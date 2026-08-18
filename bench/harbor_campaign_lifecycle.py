"""Exclusive lifecycle owner for the pinned Harbor coding campaign.

The adapter in :mod:`bench.harbor_terminal` owns one Harbor invocation.  This
module owns the longer-lived boundary: the global SparkBench lock, immutable
runtime admission, llama.cpp, telemetry, the authenticated Unix bridge, the
ordered twelve-trial campaign, and final teardown.  Raw Harbor records remain
owner-private outside the repository; the only serializable result is an exact
scalar projection validated here and by the adapter.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
from types import FrameType, SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence

from .client import stream_chat_request
from .docker_bridge import AuthenticatedUnixHttpBridge, UnixBridgeConfig
from .harbor_runtime_assets import (
    RuntimeAssetError,
    stage_immutable_asset,
    verify_normalized_tree,
)
from .harbor_terminal import (
    CampaignSpec,
    HarborAttempt,
    HarborCampaignError,
    HarborRunStatus,
    NpmArtifactAdmission,
    build_harbor_invocation,
    canonical_summary_bytes,
    derive_private_task_dataset,
    hold_campaign_lock,
    iter_trials,
    load_campaign,
    load_trial_job_result,
    prepare_runtime_overlay,
    run_harbor_invocation,
    summarize_campaign_results,
    verify_empty_python_pycache,
    verify_harbor_runtime,
    verify_npm_artifact_admission,
    verify_staged_agent_source,
)
from .journal import utc_now
from .manifest import load_models
from .runner import _preflight, results_lock_path
from .runtime import start_llamacpp, validate_llamacpp_artifacts
from .telemetry import TelemetrySampler


EXPECTED_CAMPAIGN_ID = "qwen3-coder-next-harbor-terminal-2026-08-17"
EXPECTED_MODEL_PROFILE = "qwen3-coder-next-80b-a3b-ud-q4-k-xl-llamacpp"
EXPECTED_TRIALS = 12
EXPECTED_HARD_CUTOFF_S = 23_400
EXPECTED_AUDIT_RESERVE_S = 5_400
SCHEMA_VERSION = 1

DEFAULT_CAMPAIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "manifests"
    / "campaigns"
    / "harbor_terminal_coder_next.toml"
)
DEFAULT_MODELS_PATH = Path(__file__).resolve().parents[1] / "manifests" / "models.toml"
DEFAULT_DATASET_CHECKOUT = Path.home() / ".cache" / "sparkbench" / "terminal-bench-2-1"
DEFAULT_HARBOR_RUNTIME = Path.home() / ".sparkbench-private" / "harbor-runtime-v0.21.0"
DEFAULT_TOOL_PREFIX_ROOT = (
    Path.home() / ".sparkbench-private" / "harbor-agent-prefixes-v1"
)
DEFAULT_RAW_ROOT = Path.home() / ".sparkbench-private" / "harbor-campaign-runs"

RELAY_SOURCE = Path(__file__).resolve().parent / "assets" / "harbor_uds_relay.js"
POLICY_SOURCE = (
    Path(__file__).resolve().parent / "assets" / "harbor_no_network_policy.sh"
)

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_OUTPUT_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}\.json$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")

_RESULT_STATUSES = frozenset({"aborted", "completed", "partial"})
_STOP_REASONS = frozenset(
    {
        "artifact_admission",
        "auth_failure_gate",
        "bridge_admission",
        "bridge_start",
        "canary_gate",
        "cleanup_failure",
        "completed",
        "dataset_admission",
        "endpoint_failure_gate",
        "hard_cutoff",
        "image_identity_gate",
        "interrupted",
        "model_admission",
        "model_start",
        "npm_admission",
        "platform_admission",
        "preflight",
        "runtime_admission",
        "trial_failure",
    }
)
_ENDPOINT_FAILURES = frozenset(
    {
        "ApiConnectionClosedError",
        "ApiConnectionError",
        "ApiInternalServerError",
        "ApiOverloadedError",
        "ApiResponseStalledError",
        "NetworkConnectionError",
        "RuntimeRequestError",
        "UnknownApiError",
    }
)
_IMMEDIATE_AUTH_FAILURES = frozenset(
    {
        "AgentAuthenticationError",
        "ApiKeyRejectedError",
        "AuthenticationError",
        "NotAuthenticatedError",
        "ModelNotFoundError",
        "ApiProviderResourceNotFoundError",
    }
)
_NETWORK_ADMISSION_KEYS = (
    "setup_relay_rejected",
    "agent_relay_passed",
    "wrong_auth_rejected",
    "other_loopback_rejected",
    "gost_rejected",
    "dns_rejected",
    "gateway_rejected",
    "public_rejected",
    "capabilities_dropped",
)

_EXPECTED_MODEL = {
    "profile": EXPECTED_MODEL_PROFILE,
    "source": "unsloth/Qwen3-Coder-Next-GGUF",
    "revision": "ce09c67b53bc8739eef83fe67b2f5d293c270632",
    "digest": "sha256:4bb93f0a0221ef4ff963ca9094df629c8dfdfabc3b4fdd85c1a2e4c0624fce36",
    "size_bytes": 49_608_478_720,
    "runtime_revision": "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70",
    "runtime_digest": "sha256:ae1bd49f869ff3397b2a5d757fcf010c6eaaf16c4e3071a15861312defcd4e40",
    "runtime_parallel": 1,
    "served_context_tokens": 65_536,
}


class CampaignLifecycleError(RuntimeError):
    """A payload-free failure in the outer campaign owner."""


class CampaignInterrupted(BaseException):
    """Termination translated to an exception so ``finally`` always runs."""


@dataclass(frozen=True, slots=True)
class ModelAdmission:
    chat_passed: bool
    json_passed: bool
    tool_call_passed: bool
    prompt_tokens: int
    completion_tokens: int
    wall_s: float


@dataclass(frozen=True, slots=True)
class RuntimeAdmission:
    artifact_validation: bool
    harbor_runtime_verified: bool
    node_tree_verified: bool
    agent_trees_verified: int
    npm_artifacts_verified: int
    runtime_assets_verified: int
    agent_source_files_verified: int
    python_bytecode_cache_empty: bool
    host_arm64: bool
    docker_server_arm64: bool
    model: ModelAdmission | None
    unix_bridge_verified: bool


@dataclass(frozen=True, slots=True)
class CleanupStatus:
    harbor_resources_removed: bool
    bridge_stopped: bool
    bridge_socket_removed: bool
    server_stopped: bool
    telemetry_stopped: bool
    key_removed: bool
    raw_jobs_private_retained: bool
    raw_jobs_key_free: bool
    derived_dataset_removed: bool
    runtime_overlays_removed: bool
    staged_assets_removed: bool
    socket_directory_removed: bool
    npm_scratch_removed: bool
    agent_source_removed: bool
    python_pycache_removed: bool


@dataclass(frozen=True, slots=True)
class _UnixBridgeHandle:
    bridge: AuthenticatedUnixHttpBridge
    socket_path: Path
    thread: threading.Thread
    failures: list[BaseException]

    def assert_healthy(self) -> None:
        if self.failures or not self.thread.is_alive():
            raise CampaignLifecycleError("authenticated Unix bridge stopped")
        metadata = os.lstat(self.socket_path)
        if (
            not stat.S_ISSOCK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise CampaignLifecycleError("authenticated Unix bridge changed")

    def close(self) -> tuple[bool, bool]:
        try:
            self.bridge.close()
        finally:
            self.thread.join(timeout=10)
        stopped = not self.thread.is_alive() and not self.failures
        removed = not self.socket_path.exists() and not self.socket_path.is_symlink()
        return stopped, removed


@dataclass(frozen=True, slots=True)
class _LockedOutcome:
    campaign_summary: dict[str, Any]
    model_provenance: dict[str, Any]
    admission: RuntimeAdmission
    cleanup: CleanupStatus
    trials_started: int
    trials_completed: int
    cutoff_reached: bool
    stop_reason: str
    primary_error: BaseException | None
    relay_credential: str


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _reject_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise CampaignLifecycleError("campaign path cannot be inspected") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise CampaignLifecycleError("campaign path contains a symbolic link")


def prepare_external_raw_root(path: Path, *, workspace: Path) -> Path:
    """Create one owner-only root outside the repository."""

    _reject_symlink_components(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    resolved = path.resolve(strict=True)
    repository = workspace.resolve(strict=True)
    metadata = os.lstat(resolved)
    if (
        resolved == repository
        or _is_within(resolved, repository)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise CampaignLifecycleError("raw campaign root is not isolated")
    return resolved


def _create_run_directory(raw_root: Path, campaign_id: str) -> Path:
    if _SAFE_ID.fullmatch(campaign_id) is None:
        raise CampaignLifecycleError("campaign id is invalid")
    created = Path(tempfile.mkdtemp(prefix=f"{campaign_id}-", dir=raw_root))
    created.chmod(0o700)
    return created.resolve(strict=True)


def _scalar_output_path(run_root: Path, output_name: str) -> Path:
    if _OUTPUT_NAME.fullmatch(output_name) is None:
        raise CampaignLifecycleError("scalar output name is invalid")
    candidate = run_root / output_name
    if candidate.exists() or candidate.is_symlink():
        raise CampaignLifecycleError("scalar output already exists")
    return candidate


def create_ephemeral_key(path: Path) -> str:
    """Create exactly one mode-0600 visible-ASCII key."""

    if not path.parent.exists():
        path.parent.mkdir(mode=0o700)
    _reject_symlink_components(path.parent)
    parent = os.lstat(path.parent)
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.geteuid()
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise CampaignLifecycleError("secret directory is not owner-private")
    token = secrets.token_urlsafe(48)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        payload = token.encode("ascii")
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    return token


def remove_ephemeral_key(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return not path.exists() and not path.is_symlink()


def _npm_package_specs(campaign: CampaignSpec) -> tuple[tuple[str, str], ...]:
    records: list[tuple[str, str]] = []
    for agent in campaign.agents:
        records.append((agent.npm_package, agent.version))
        if agent.platform_package is not None:
            records.append((agent.platform_package, agent.version))
    if len(records) != 3 or len(set(records)) != 3:
        raise CampaignLifecycleError("campaign npm artifact set changed")
    return tuple(records)


def _npm_environment(home: Path, cache: Path) -> dict[str, str]:
    home.mkdir(mode=0o700)
    cache.mkdir(mode=0o700)
    environment = {
        "HOME": str(home),
        "NPM_CONFIG_AUDIT": "false",
        "NPM_CONFIG_CACHE": str(cache),
        "NPM_CONFIG_FUND": "false",
        "NPM_CONFIG_LOGLEVEL": "error",
        "NPM_CONFIG_PROGRESS": "false",
        "NPM_CONFIG_UPDATE_NOTIFIER": "false",
        "NPM_CONFIG_USERCONFIG": "/dev/null",
    }
    for name in ("PATH", "LANG", "LC_ALL"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def acquire_npm_artifacts(
    campaign: CampaignSpec,
    *,
    output_dir: Path,
    repo_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> NpmArtifactAdmission:
    """Fetch exact public tarballs, then delegate pin validation to the adapter."""

    output_dir.mkdir(mode=0o700)
    environment = _npm_environment(output_dir / "home", output_dir / "cache")
    paths: dict[str, Path] = {}
    for index, (package, version) in enumerate(_npm_package_specs(campaign), start=1):
        destination = output_dir / f"artifact-{index}"
        destination.mkdir(mode=0o700)
        try:
            completed = runner(
                [
                    "npm",
                    "pack",
                    f"{package}@{version}",
                    "--ignore-scripts",
                    "--json",
                    "--pack-destination",
                    str(destination),
                ],
                cwd=output_dir,
                env=environment,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=300,
                umask=0o077,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CampaignLifecycleError("npm artifact acquisition failed") from error
        archives = tuple(destination.iterdir())
        if (
            completed.returncode != 0
            or len(archives) != 1
            or archives[0].is_symlink()
            or not archives[0].is_file()
            or archives[0].suffix != ".tgz"
        ):
            raise CampaignLifecycleError("npm artifact acquisition failed")
        archives[0].chmod(0o400)
        paths[package] = archives[0]
    return verify_npm_artifact_admission(campaign, paths, repo_root=repo_root)


def _model_port(endpoint: str) -> int:
    from urllib.parse import urlsplit

    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or parsed.path.rstrip("/") != "/v1"
        or parsed.query
        or parsed.fragment
    ):
        raise CampaignLifecycleError("model endpoint must be loopback /v1")
    return parsed.port


def _start_unix_bridge(
    *,
    socket_path: Path,
    target_port: int,
    relay_credential: str,
    idle_timeout_s: float,
) -> _UnixBridgeHandle:
    bridge = AuthenticatedUnixHttpBridge(
        UnixBridgeConfig(
            socket_path=socket_path,
            target_port=target_port,
            connect_timeout_s=5.0,
            idle_timeout_s=min(3_600.0, max(300.0, idle_timeout_s)),
        ),
        relay_credential,
    )
    failures: list[BaseException] = []

    def serve() -> None:
        try:
            bridge.serve_forever()
        except BaseException as error:
            failures.append(error)

    try:
        endpoint = bridge.start()
        thread = threading.Thread(
            target=serve,
            name="sparkbench-harbor-unix-bridge",
            daemon=True,
        )
        thread.start()
        handle = _UnixBridgeHandle(bridge, endpoint.socket_path, thread, failures)
        handle.assert_healthy()
        return handle
    except BaseException:
        bridge.close()
        raise


def _unix_http_get(socket_path: Path, relay_credential: str) -> tuple[int, bytes]:
    request = (
        "GET /v1/models HTTP/1.1\r\n"
        "Host: localhost\r\n"
        f"Authorization: Bearer {relay_credential}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(10)
    try:
        client.connect(os.fspath(socket_path))
        client.sendall(request)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = client.recv(65_536)
            if not chunk:
                break
            total += len(chunk)
            if total > 1_048_576:
                raise CampaignLifecycleError("Unix bridge response is oversized")
            chunks.append(chunk)
    except OSError as error:
        raise CampaignLifecycleError("Unix bridge admission transport failed") from error
    finally:
        client.close()
    head, separator, body = b"".join(chunks).partition(b"\r\n\r\n")
    first = head.split(b"\r\n", 1)[0].split(b" ")
    if not separator or len(first) < 2 or not first[1].isdigit():
        raise CampaignLifecycleError("Unix bridge response is malformed")
    return int(first[1]), body


def admit_unix_bridge(
    handle: _UnixBridgeHandle, *, relay_credential: str, served_name: str
) -> bool:
    invalid_status, _ = _unix_http_get(handle.socket_path, relay_credential + "x")
    if invalid_status != 401:
        raise CampaignLifecycleError("Unix bridge accepted an invalid credential")
    status_code, body = _unix_http_get(handle.socket_path, relay_credential)
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CampaignLifecycleError("Unix bridge returned invalid model metadata") from error
    models = payload.get("data") if isinstance(payload, dict) else None
    if status_code != 200 or not isinstance(models, list) or not any(
        isinstance(item, dict) and item.get("id") == served_name for item in models
    ):
        raise CampaignLifecycleError("Unix bridge returned the wrong model alias")
    handle.assert_healthy()
    return True


def run_model_admission(*, base_url: str, served_name: str) -> ModelAdmission:
    """Run ephemeral chat, JSON, and tool-call gates without persistence."""

    started = time.monotonic()
    prompt_tokens = 0
    completion_tokens = 0
    try:
        chat_marker = "CHAT-" + secrets.token_hex(8).upper()
        chat = stream_chat_request(
            base_url=base_url,
            model=served_name,
            prompt=f"Reply with exactly {chat_marker} and nothing else.",
            max_tokens=64,
            temperature=0.0,
            request_id="admission-chat-" + secrets.token_hex(8),
            timeout_s=180,
        )
        prompt_tokens += chat.prompt_tokens
        completion_tokens += chat.completion_tokens

        json_marker = "JSON-" + secrets.token_hex(8).upper()
        structured = stream_chat_request(
            base_url=base_url,
            model=served_name,
            prompt=(
                "Return one JSON object with exactly two fields: "
                f'{{"ok":true,"marker":"{json_marker}"}}.'
            ),
            max_tokens=128,
            temperature=0.0,
            request_id="admission-json-" + secrets.token_hex(8),
            extra_body={"response_format": {"type": "json_object"}},
            timeout_s=180,
        )
        prompt_tokens += structured.prompt_tokens
        completion_tokens += structured.completion_tokens

        tool_marker = "TOOL-" + secrets.token_hex(8).upper()
        tool = stream_chat_request(
            base_url=base_url,
            model=served_name,
            prompt=f"Call return_marker once with marker {tool_marker}.",
            max_tokens=256,
            temperature=0.0,
            request_id="admission-tool-" + secrets.token_hex(8),
            extra_body={
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "return_marker",
                            "description": "Return the supplied marker.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "marker": {"type": "string", "enum": [tool_marker]}
                                },
                                "required": ["marker"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "tool_choice": "required",
            },
            timeout_s=180,
        )
        prompt_tokens += tool.prompt_tokens
        completion_tokens += tool.completion_tokens
        tool_valid = False
        if len(tool.tool_calls) == 1:
            function = tool.tool_calls[0].get("function")
            if isinstance(function, dict) and function.get("name") == "return_marker":
                arguments = json.loads(str(function.get("arguments", "")))
                tool_valid = arguments == {"marker": tool_marker}
    except Exception as error:
        raise CampaignLifecycleError("model admission request failed") from error
    if (
        chat.content.strip() != chat_marker
        or json.loads(structured.content) != {"ok": True, "marker": json_marker}
        or not tool_valid
    ):
        raise CampaignLifecycleError("model admission response failed validation")
    return ModelAdmission(
        chat_passed=True,
        json_passed=True,
        tool_call_passed=True,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        wall_s=round(max(time.monotonic() - started, 0.0), 6),
    )


def _git_provenance(workspace: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=workspace,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CampaignLifecycleError("repository state admission failed") from error
    commit = revision.stdout.strip()
    if (
        revision.returncode != 0
        or status_result.returncode != 0
        or _HEX_40.fullmatch(commit) is None
    ):
        raise CampaignLifecycleError("repository state admission failed")
    return commit, not bool(status_result.stdout)


def _cross_validate_model(campaign: CampaignSpec, model: Any) -> None:
    actual = asdict(model)
    runtime_binary = actual.pop("runtime_binary")
    runtime_source_dir = actual.pop("runtime_source_dir")
    expected = {
        "id": EXPECTED_MODEL_PROFILE,
        "backend": "llamacpp",
        "source": "unsloth/Qwen3-Coder-Next-GGUF",
        "served_name": "Qwen/Qwen3-Coder-Next",
        "tasks": ("chat", "json", "tools"),
        "image": None,
        "max_context": 65_536,
        "native_context": 65_536,
        "startup_timeout_s": 1_200,
        "args": (
            "--n-gpu-layers", "all",
            "--flash-attn", "on",
            "--fit", "off",
            "--batch-size", "8192",
            "--ubatch-size", "512",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--jinja",
            "--reasoning", "off",
            "--temp", "1.0",
            "--top-p", "0.95",
            "--top-k", "40",
            "--n-predict", "8192",
            "--no-context-shift",
        ),
        "endpoint": "http://127.0.0.1:8000/v1",
        "estimated_ram_gib": 96.0,
        "revision": "ce09c67b53bc8739eef83fe67b2f5d293c270632",
        "image_digest": None,
        "architecture": "qwen3next",
        "quantization": "ud-q4_k_xl",
        "lifecycle": "subprocess",
        "description": (
            "Pinned Unsloth Qwen3-Coder-Next 80B-A3B Dynamic 2.0 "
            "UD-Q4_K_XL GGUF agent-serving baseline, single-slot at 64K "
            "and without speculation."
        ),
        "cache_dir": "user",
        "fetch_allow_patterns": ("Qwen3-Coder-Next-UD-Q4_K_XL.gguf",),
        "fetch_ignore_patterns": (),
        "weight_size_bytes": None,
        "weight_file_count": None,
        "draft_source": None,
        "draft_revision": None,
        "draft_weight_size_bytes": None,
        "draft_model_file": None,
        "draft_model_digest": None,
        "draft_model_size_bytes": None,
        "sglang_allow_hf_metadata_probe": False,
        "recipe_source": None,
        "recipe_revision": None,
        "request_body_json": None,
        "runtime_python": None,
        "runtime_digest": "sha256:ae1bd49f869ff3397b2a5d757fcf010c6eaaf16c4e3071a15861312defcd4e40",
        "runtime_parallel": 1,
        "runtime_revision": "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70",
        "model_file": "Qwen3-Coder-Next-UD-Q4_K_XL.gguf",
        "model_digest": "sha256:4bb93f0a0221ef4ff963ca9094df629c8dfdfabc3b4fdd85c1a2e4c0624fce36",
        "model_size_bytes": 49_608_478_720,
        "model_shards": (),
        "mmproj_file": None,
        "mmproj_digest": None,
        "mmproj_size_bytes": None,
        "support_status": "spark_other_backend",
    }
    locators_valid = (
        runtime_binary
        == "~/.cache/sparkbench/llama.cpp-b10453/build/bin/llama-server"
        and runtime_source_dir == "~/.cache/sparkbench/llama.cpp-b10453"
    )
    if (
        campaign.id != EXPECTED_CAMPAIGN_ID
        or campaign.model.profile != EXPECTED_MODEL_PROFILE
        or actual != expected
        or not locators_valid
        or model.served_name != campaign.model.served_name
        or model.max_context != campaign.model.context_tokens
        or model.runtime_parallel != campaign.model.parallel
        or campaign.execution.hard_campaign_cutoff_s != EXPECTED_HARD_CUTOFF_S
        or campaign.execution.reserve_for_audit_s != EXPECTED_AUDIT_RESERVE_S
        or _model_port(model.endpoint) != 8000
    ):
        raise CampaignLifecycleError("campaign and model profile do not match")


def _runtime_model(model: Any, campaign_id: str) -> SimpleNamespace:
    values = asdict(model)
    for name in ("runtime_binary", "runtime_source_dir"):
        value = values.get(name)
        if isinstance(value, str):
            values[name] = os.fspath(Path(value).expanduser())
    runtime = SimpleNamespace(**values)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runtime.run_identity = f"{campaign_id}-{stamp}-{secrets.token_hex(6)}"
    return runtime


def _model_provenance(model: Any, artifacts: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "profile": str(model.id),
        "source": str(artifacts["model_source"]),
        "revision": str(artifacts["model_revision"]),
        "digest": str(artifacts["model_sha256"]),
        "size_bytes": int(artifacts["model_size_bytes"]),
        "runtime_revision": str(artifacts["runtime_source_revision"]),
        "runtime_digest": str(artifacts["runtime_binary_sha256"]),
        "runtime_parallel": int(model.runtime_parallel),
        "served_context_tokens": int(model.max_context),
    }
    if result != _EXPECTED_MODEL:
        raise CampaignLifecycleError("model artifact provenance changed")
    return result


def admit_native_platform(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    machine: Callable[[], str] = platform.machine,
) -> tuple[bool, bool]:
    """Require an ARM64 host and Docker server before model startup."""

    host_arm64 = machine().lower() in {"aarch64", "arm64"}
    try:
        completed = runner(
            ["docker", "info", "--format", "{{.Architecture}}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CampaignLifecycleError("Docker platform admission failed") from error
    docker_architecture = completed.stdout.strip().lower()
    docker_arm64 = (
        completed.returncode == 0
        and docker_architecture in {"aarch64", "arm64"}
    )
    if not host_arm64 or not docker_arm64:
        raise CampaignLifecycleError("native ARM64 admission failed")
    return host_arm64, docker_arm64


def _remove_owned_tree(path: Path, *, owner: Path) -> bool:
    """Remove one exact ephemeral child; never follow or widen its scope."""

    try:
        owner_resolved = owner.resolve(strict=True)
        candidate = Path(os.path.abspath(path))
        if candidate.parent.resolve(strict=True) != owner_resolved and not _is_within(
            candidate.parent.resolve(strict=True), owner_resolved
        ):
            return False
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            return True
        if stat.S_ISLNK(metadata.st_mode):
            return False
        if stat.S_ISDIR(metadata.st_mode):
            shutil.rmtree(candidate)
        elif stat.S_ISREG(metadata.st_mode):
            candidate.unlink()
        else:
            return False
        return not candidate.exists() and not candidate.is_symlink()
    except OSError:
        return False


def certify_private_raw_jobs(
    path: Path, *, owner: Path, relay_credential: str
) -> tuple[bool, bool]:
    """Certify retained evidence is private and contains no exact run key."""

    try:
        owner_resolved = owner.resolve(strict=True)
        candidate = Path(os.path.abspath(path))
        if not _is_within(candidate, owner_resolved):
            return False, False
        if not candidate.exists():
            return True, True
        root_metadata = os.lstat(candidate)
        owner_metadata = os.lstat(owner_resolved)
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or root_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(root_metadata.st_mode) != 0o700
            or not stat.S_ISDIR(owner_metadata.st_mode)
            or owner_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(owner_metadata.st_mode) != 0o700
        ):
            return False, False
        needle = relay_credential.encode("ascii")
        private = True
        key_free = True
        for directory, directory_names, file_names in os.walk(
            candidate, followlinks=False
        ):
            directory_path = Path(directory)
            directory_metadata = os.lstat(directory_path)
            if (
                not stat.S_ISDIR(directory_metadata.st_mode)
                or directory_metadata.st_uid != os.geteuid()
            ):
                private = False
            for name in directory_names:
                child_metadata = os.lstat(directory_path / name)
                if stat.S_ISLNK(child_metadata.st_mode):
                    private = False
            for name in file_names:
                child = directory_path / name
                metadata = os.lstat(child)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                ):
                    private = False
                    continue
                overlap = b""
                with child.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1_048_576), b""):
                        window = overlap + chunk
                        if needle and needle in window:
                            key_free = False
                            break
                        overlap = window[-max(0, len(needle) - 1) :]
                if not key_free:
                    break
            if not key_free:
                break
        return private, key_free
    except (OSError, UnicodeEncodeError):
        return False, False


@contextmanager
def termination_guard() -> Iterator[None]:
    """Translate SIGINT/SIGTERM and protect the ensuing teardown."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous: dict[signal.Signals, Any] = {}

    def interrupt(signum: int, _frame: FrameType | None) -> None:
        for name in ("SIGINT", "SIGTERM"):
            repeated = getattr(signal, name, None)
            if repeated is not None:
                signal.signal(repeated, signal.SIG_IGN)
        raise CampaignInterrupted(f"campaign interrupted by signal {signum}")

    try:
        for name in ("SIGINT", "SIGTERM"):
            signum = getattr(signal, name, None)
            if signum is not None:
                previous[signum] = signal.signal(signum, interrupt)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _validate_trial_order(campaign: CampaignSpec) -> tuple[Any, ...]:
    trials = iter_trials(campaign)
    actual = tuple(f"{trial.task_id}:{trial.agent_id}" for trial in trials)
    if (
        len(trials) != EXPECTED_TRIALS
        or actual != campaign.execution.trial_order
        or len(set(actual)) != EXPECTED_TRIALS
        or tuple(trial.index for trial in trials) != tuple(range(1, 13))
    ):
        raise CampaignLifecycleError("adapter trial order differs from the manifest")
    return trials


def _status_network_admitted(status: HarborRunStatus) -> bool:
    return all(bool(getattr(status, key)) for key in _NETWORK_ADMISSION_KEYS)


def _canary_admitted(attempt: HarborAttempt, projection: Mapping[str, Any]) -> bool:
    status = attempt.status
    reward = projection.get("reward")
    return (
        attempt.job_result is not None
        and status.exit_code == 0
        and not status.timed_out
        and status.main_image_id is not None
        and _IMAGE_ID.fullmatch(status.main_image_id) is not None
        and status.main_image_arm64
        and status.relay_image_arm64
        and status.built_image_cleanup_succeeded
        and status.cleanup_succeeded
        and _status_network_admitted(status)
        and reward is not None
        and float(reward) in {0.0, 1.0}
        and projection.get("exception_class") is None
    )


def _attempt_gate(
    attempts: Sequence[HarborAttempt], summary: Mapping[str, Any]
) -> str | None:
    """Return one fixed stop reason, never a raw exception or message."""

    trials = summary["trials"]
    if not attempts or len(trials) != len(attempts):
        raise CampaignLifecycleError("attempt projection is inconsistent")
    latest = trials[-1]
    exception_class = latest.get("exception_class")
    if exception_class in _IMMEDIATE_AUTH_FAILURES:
        return "auth_failure_gate"
    if len(attempts) == 1 and not _canary_admitted(attempts[0], latest):
        return "canary_gate"
    status = attempts[-1].status
    if (
        not status.cleanup_succeeded
        or not status.built_image_cleanup_succeeded
        or not _status_network_admitted(status)
        or not status.main_image_arm64
        or not status.relay_image_arm64
        or status.main_image_id is None
    ):
        return "trial_failure"
    if latest.get("paired_image_match") is False:
        return "image_identity_gate"
    if len(trials) >= 2 and all(
        trial.get("exception_class") in _ENDPOINT_FAILURES for trial in trials[-2:]
    ):
        return "endpoint_failure_gate"
    return None


def _trial_timeout_s(campaign: CampaignSpec, *, deadline: float, now: float) -> float | None:
    remaining = deadline - now
    if not math.isfinite(remaining):
        raise CampaignLifecycleError("campaign deadline is invalid")
    if remaining <= 0:
        return None
    return min(float(campaign.execution.trial_wall_timeout_s), remaining)


def _deadline_limited_timeout(
    status: HarborRunStatus, *, remaining_s: float, per_trial_cap_s: float
) -> bool:
    return (
        status.timed_out
        and math.isfinite(remaining_s)
        and 0 < remaining_s <= per_trial_cap_s
    )


def _record_status_then_load(
    *,
    trial: Any,
    status: HarborRunStatus,
    attempts: list[HarborAttempt],
    loader: Callable[[], Any],
) -> None:
    """Retain cleanup status even if strict raw-result parsing fails."""

    attempts.append(HarborAttempt(trial=trial, status=status, job_result=None))
    raw_result = loader()
    attempts[-1] = HarborAttempt(
        trial=trial, status=status, job_result=raw_result
    )


def _harbor_cleanup_certified(
    invocations_started: int, statuses: Sequence[HarborRunStatus]
) -> bool:
    return (
        invocations_started == len(statuses)
        and all(status.cleanup_succeeded for status in statuses)
    )


def _exact_mapping(
    value: Any, keys: frozenset[str], context: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != keys:
        raise CampaignLifecycleError(f"{context} schema changed")
    return value


def _schema_int(
    value: Any, context: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise CampaignLifecycleError(f"{context} is outside its scalar schema")
    return value


def _schema_number(
    value: Any, context: str, *, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignLifecycleError(f"{context} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or (
        maximum is not None and numeric > maximum
    ):
        raise CampaignLifecycleError(f"{context} is outside its scalar schema")
    return numeric


def _timestamp(value: Any, context: str) -> datetime:
    if not isinstance(value, str) or len(value) > 40 or "\n" in value or "\r" in value:
        raise CampaignLifecycleError(f"{context} is not a bounded timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise CampaignLifecycleError(f"{context} is not an ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise CampaignLifecycleError(f"{context} must be UTC")
    return parsed


def validate_lifecycle_envelope(
    payload: Mapping[str, Any], *, campaign: CampaignSpec
) -> None:
    """Validate every key, value, enum, pin, and accounting equation."""

    envelope = _exact_mapping(
        payload,
        frozenset(
            {
                "schema_version",
                "campaign_id",
                "status",
                "stop_reason",
                "started_at",
                "finished_at",
                "elapsed_s",
                "git",
                "model",
                "admission",
                "execution",
                "cleanup",
                "campaign",
            }
        ),
        "lifecycle envelope",
    )
    if (
        envelope.get("schema_version") != SCHEMA_VERSION
        or envelope.get("campaign_id") != campaign.id
        or campaign.id != EXPECTED_CAMPAIGN_ID
        or envelope.get("status") not in _RESULT_STATUSES
        or envelope.get("stop_reason") not in _STOP_REASONS
    ):
        raise CampaignLifecycleError("lifecycle identity or status changed")
    started = _timestamp(envelope["started_at"], "started_at")
    finished = _timestamp(envelope["finished_at"], "finished_at")
    elapsed = _schema_number(envelope["elapsed_s"], "elapsed_s", maximum=36_000)
    if finished < started or elapsed < 0:
        raise CampaignLifecycleError("lifecycle timing is inconsistent")

    git = _exact_mapping(envelope["git"], frozenset({"revision", "clean"}), "git")
    if (
        not isinstance(git.get("revision"), str)
        or _HEX_40.fullmatch(git["revision"]) is None
        or git.get("clean") is not True
    ):
        raise CampaignLifecycleError("Git admission changed")

    model = _exact_mapping(
        envelope["model"], frozenset(_EXPECTED_MODEL), "model"
    )
    if dict(model) != _EXPECTED_MODEL:
        raise CampaignLifecycleError("model provenance changed")

    admission = _exact_mapping(
        envelope["admission"],
        frozenset(
            {
                "artifact_validation",
                "harbor_runtime_verified",
                "node_tree_verified",
                "agent_trees_verified",
                "npm_artifacts_verified",
                "runtime_assets_verified",
                "agent_source_files_verified",
                "python_bytecode_cache_empty",
                "host_arm64",
                "docker_server_arm64",
                "model",
                "unix_bridge_verified",
            }
        ),
        "admission",
    )
    for key in (
        "artifact_validation",
        "harbor_runtime_verified",
        "node_tree_verified",
        "host_arm64",
        "docker_server_arm64",
        "unix_bridge_verified",
        "python_bytecode_cache_empty",
    ):
        if not isinstance(admission[key], bool):
            raise CampaignLifecycleError("admission flags changed type")
    agent_trees = _schema_int(
        admission["agent_trees_verified"], "agent_trees_verified", maximum=2
    )
    npm_artifacts = _schema_int(
        admission["npm_artifacts_verified"], "npm_artifacts_verified", maximum=3
    )
    runtime_assets = _schema_int(
        admission["runtime_assets_verified"], "runtime_assets_verified", maximum=2
    )
    agent_source_files = _schema_int(
        admission["agent_source_files_verified"],
        "agent_source_files_verified",
        maximum=1,
    )
    model_admission = admission["model"]
    if model_admission is not None:
        model_gate = _exact_mapping(
            model_admission,
            frozenset(
                {
                    "chat_passed",
                    "json_passed",
                    "tool_call_passed",
                    "prompt_tokens",
                    "completion_tokens",
                    "wall_s",
                }
            ),
            "model admission",
        )
        if any(
            model_gate[key] is not True
            for key in ("chat_passed", "json_passed", "tool_call_passed")
        ):
            raise CampaignLifecycleError("model admission did not pass")
        _schema_int(model_gate["prompt_tokens"], "admission prompt tokens")
        _schema_int(model_gate["completion_tokens"], "admission completion tokens")
        _schema_number(model_gate["wall_s"], "admission wall", maximum=3_600)

    execution = _exact_mapping(
        envelope["execution"],
        frozenset(
            {
                "trials_planned",
                "trials_started",
                "trials_completed",
                "hard_cutoff_s",
                "audit_reserve_s",
                "cutoff_reached",
            }
        ),
        "execution",
    )
    started_count = _schema_int(
        execution["trials_started"], "trials_started", maximum=EXPECTED_TRIALS
    )
    completed_count = _schema_int(
        execution["trials_completed"], "trials_completed", maximum=started_count
    )
    if (
        execution.get("trials_planned") != EXPECTED_TRIALS
        or execution.get("hard_cutoff_s") != EXPECTED_HARD_CUTOFF_S
        or execution.get("audit_reserve_s") != EXPECTED_AUDIT_RESERVE_S
        or not isinstance(execution.get("cutoff_reached"), bool)
    ):
        raise CampaignLifecycleError("execution contract changed")

    cleanup = _exact_mapping(
        envelope["cleanup"],
        frozenset(
            {
                "harbor_resources_removed",
                "bridge_stopped",
                "bridge_socket_removed",
                "server_stopped",
                "telemetry_stopped",
                "key_removed",
                "raw_jobs_private_retained",
                "raw_jobs_key_free",
                "derived_dataset_removed",
                "runtime_overlays_removed",
                "staged_assets_removed",
                "socket_directory_removed",
                "npm_scratch_removed",
                "agent_source_removed",
                "python_pycache_removed",
            }
        ),
        "cleanup",
    )
    if any(not isinstance(value, bool) for value in cleanup.values()):
        raise CampaignLifecycleError("cleanup flags changed type")

    campaign_summary = envelope["campaign"]
    if not isinstance(campaign_summary, dict):
        raise CampaignLifecycleError("campaign summary must be an exact object")
    try:
        canonical_summary_bytes(campaign_summary)
    except HarborCampaignError as error:
        raise CampaignLifecycleError("campaign scalar schema changed") from error
    totals = campaign_summary["summary"]
    if (
        campaign_summary.get("campaign_id") != campaign.id
        or totals.get("planned_attempts") != EXPECTED_TRIALS
        or totals.get("attempts") != started_count
        or totals.get("completed_results") != completed_count
        or totals.get("campaign_cutoff_reached") != execution["cutoff_reached"]
        or elapsed < float(totals.get("wall_s", 0))
    ):
        raise CampaignLifecycleError("lifecycle and campaign accounting diverged")

    all_runtime_admitted = (
        admission["artifact_validation"]
        and admission["harbor_runtime_verified"]
        and admission["node_tree_verified"]
        and agent_trees == 2
        and npm_artifacts == 3
        and runtime_assets == 2
        and agent_source_files == 1
        and admission["python_bytecode_cache_empty"]
        and admission["host_arm64"]
        and admission["docker_server_arm64"]
        and model_admission is not None
        and admission["unix_bridge_verified"]
    )
    status_value = envelope["status"]
    stop_reason = envelope["stop_reason"]
    cutoff = execution["cutoff_reached"]
    if started_count and not all_runtime_admitted:
        raise CampaignLifecycleError("a trial started without full admission")
    if (stop_reason == "hard_cutoff" and not cutoff) or (
        cutoff and stop_reason not in {"hard_cutoff", "cleanup_failure"}
    ):
        raise CampaignLifecycleError("hard-cutoff status is inconsistent")
    if status_value == "completed":
        if (
            stop_reason != "completed"
            or started_count != EXPECTED_TRIALS
            or completed_count != EXPECTED_TRIALS
            or totals.get("campaign_complete") is not True
            or not all_runtime_admitted
            or not all(cleanup.values())
        ):
            raise CampaignLifecycleError("completed lifecycle is inconsistent")
    elif status_value == "aborted":
        if started_count != 0 or stop_reason == "completed":
            raise CampaignLifecycleError("aborted lifecycle is inconsistent")
    elif started_count == 0 or stop_reason == "completed":
        raise CampaignLifecycleError("partial lifecycle is inconsistent")
    if stop_reason == "cleanup_failure" and all(cleanup.values()):
        raise CampaignLifecycleError("cleanup failure lacks a failed cleanup scalar")


def build_lifecycle_envelope(
    *,
    campaign: CampaignSpec,
    campaign_summary: dict[str, Any],
    model_provenance: dict[str, Any],
    git_revision: str,
    git_clean: bool,
    admission: RuntimeAdmission,
    cleanup: CleanupStatus,
    status: str,
    stop_reason: str,
    started_at: str,
    finished_at: str,
    elapsed_s: float,
    trials_started: int,
    trials_completed: int,
    cutoff_reached: bool,
    sensitive_values: Sequence[str] = (),
) -> dict[str, Any]:
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign.id,
        "status": status,
        "stop_reason": stop_reason,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_s": round(max(float(elapsed_s), 0.0), 6),
        "git": {"revision": git_revision, "clean": git_clean},
        "model": dict(model_provenance),
        "admission": {
            **asdict(admission),
            "model": asdict(admission.model) if admission.model is not None else None,
        },
        "execution": {
            "trials_planned": EXPECTED_TRIALS,
            "trials_started": trials_started,
            "trials_completed": trials_completed,
            "hard_cutoff_s": campaign.execution.hard_campaign_cutoff_s,
            "audit_reserve_s": campaign.execution.reserve_for_audit_s,
            "cutoff_reached": cutoff_reached,
        },
        "cleanup": asdict(cleanup),
        "campaign": campaign_summary,
    }
    validate_lifecycle_envelope(envelope, campaign=campaign)
    encoded = json.dumps(
        envelope,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    for secret in sensitive_values:
        if secret and secret.encode("ascii") in encoded:
            raise CampaignLifecycleError("ephemeral key entered scalar output")
    return envelope


def write_scalar_result(
    path: Path, payload: Mapping[str, Any], *, campaign: CampaignSpec
) -> None:
    """Atomically persist only an exact validated lifecycle envelope."""

    validate_lifecycle_envelope(payload, campaign=campaign)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    parent = path.parent.resolve(strict=True)
    parent_metadata = os.lstat(parent)
    if (
        path.parent != parent
        or stat.S_IMODE(parent_metadata.st_mode) != 0o700
        or parent_metadata.st_uid != os.geteuid()
        or path.exists()
        or path.is_symlink()
    ):
        raise CampaignLifecycleError("scalar output destination is unsafe")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if path.exists() or path.is_symlink():
            raise CampaignLifecycleError("scalar output changed during write")
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _prepare_raw_file(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    os.close(descriptor)


def _verify_tool_prefixes(
    campaign: CampaignSpec, *, prefix_root: Path, workspace: Path
) -> tuple[Path, dict[str, Path]]:
    node_path = prefix_root / "node"
    agent_paths = {
        "qwen-coder": prefix_root / "qwen",
        "opencode": prefix_root / "opencode",
    }
    try:
        verify_normalized_tree(
            node_path,
            repo_root=workspace,
            expected_digest=campaign.toolchain.node_tree_sha256,
            expected_size_bytes=campaign.toolchain.node_tree_size_bytes,
        )
        for agent in campaign.agents:
            verify_normalized_tree(
                agent_paths[agent.id],
                repo_root=workspace,
                expected_digest=agent.install_tree_sha256,
                expected_size_bytes=agent.install_tree_size_bytes,
            )
    except (KeyError, RuntimeAssetError) as error:
        raise CampaignLifecycleError("immutable tool-prefix admission failed") from error
    return node_path, agent_paths


def _stage_runtime_assets(
    campaign: CampaignSpec, *, run_root: Path, workspace: Path
) -> tuple[Path, Path]:
    asset_root = run_root / "runtime-assets"
    asset_root.mkdir(mode=0o700)
    try:
        relay = stage_immutable_asset(
            RELAY_SOURCE,
            asset_root / "relay.js",
            repo_root=workspace,
            expected_digest=campaign.relay.relay_script_sha256,
            expected_source_mode=0o644,
            output_mode=0o444,
        )
        policy = stage_immutable_asset(
            POLICY_SOURCE,
            asset_root / "network-policy",
            repo_root=workspace,
            expected_digest=campaign.relay.network_policy_sha256,
            expected_source_mode=0o644,
            output_mode=0o555,
        )
    except RuntimeAssetError as error:
        raise CampaignLifecycleError("runtime asset admission failed") from error
    return relay, policy


def _git_head_blob(
    workspace: Path,
    revision: str,
    relative: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> bytes:
    """Read one exact regular 100644 blob and bind bytes to its Git object ID."""

    try:
        listing = runner(
            ["git", "ls-tree", revision, "--", relative],
            cwd=workspace,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        )
        blob = runner(
            ["git", "cat-file", "blob", f"{revision}:{relative}"],
            cwd=workspace,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CampaignLifecycleError("Git agent source admission failed") from error
    fields = listing.stdout.strip().split(maxsplit=3) if listing.returncode == 0 else []
    payload = blob.stdout if blob.returncode == 0 and isinstance(blob.stdout, bytes) else b""
    object_id = hashlib.sha1(
        b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload,
        usedforsecurity=False,
    ).hexdigest()
    if (
        len(fields) != 4
        or fields[0] != "100644"
        or fields[1] != "blob"
        or fields[2] != object_id
        or fields[3] != relative
        or not 0 < len(payload) <= 1_048_576
    ):
        raise CampaignLifecycleError("Git agent source blob changed")
    return payload


def _write_private_source(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        os.fchmod(descriptor, 0o444)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def stage_head_agent_source(
    campaign: CampaignSpec,
    *,
    workspace: Path,
    run_root: Path,
    revision: str,
) -> tuple[Path, Path]:
    """Stage the committed namespace-package module and an empty pycache."""

    if _HEX_40.fullmatch(revision) is None:
        raise CampaignLifecycleError("Git agent source revision is invalid")
    source_root = run_root / "agent-source"
    bench_root = source_root / "bench"
    pycache_root = run_root / "python-pycache"
    source_root.mkdir(mode=0o700)
    bench_root.mkdir(mode=0o700)
    pycache_root.mkdir(mode=0o700)
    for relative in ("bench/harbor_pinned_agents.py",):
        payload = _git_head_blob(workspace, revision, relative)
        _write_private_source(source_root / relative, payload)
    admitted_root, digest = verify_staged_agent_source(
        source_root, repo_root=workspace
    )
    if digest != campaign.harbor.agent_source_sha256:
        raise CampaignLifecycleError("staged agent source pin changed")
    admitted_cache = verify_empty_python_pycache(
        pycache_root, repo_root=workspace
    )
    return admitted_root, admitted_cache


def _execute_campaign_locked(
    *,
    lock: Any,
    workspace: Path,
    external_root: Path,
    run_root: Path,
    campaign: CampaignSpec,
    model: Any,
    runtime_model: Any,
    trials: Sequence[Any],
    harbor_runtime_root: Path,
    tool_prefix_root: Path,
    dataset_checkout: Path,
    git_revision: str,
    wall_started: float,
    monotonic: Callable[[], float],
) -> _LockedOutcome:
    """Own every campaign resource while the single lock remains active."""

    jobs_dir = run_root / "jobs"
    derived_dir = run_root / "derived-dataset"
    overlay_dir = run_root / "runtime-overlays"
    socket_dir = run_root / "relay-private"
    npm_dir = run_root / "npm-artifacts"
    telemetry_path = run_root / "telemetry.jsonl"
    server_log_path = run_root / "llama-server.log"
    process_state_path = run_root / "llama-server-state.json"
    key_path = socket_dir / PurePosixPath(campaign.relay.internal_key_path).name
    socket_path = socket_dir / PurePosixPath(campaign.relay.uds_path).name

    jobs_dir.mkdir(mode=0o700)
    overlay_dir.mkdir(mode=0o700)
    socket_dir.mkdir(mode=0o700)
    _prepare_raw_file(telemetry_path)

    patch: Any | None = None
    npm_admission: NpmArtifactAdmission | None = None
    artifacts: dict[str, Any] | None = None
    harbor_runtime_verified = False
    node_tree_verified = False
    agent_trees_verified = 0
    npm_artifacts_verified = 0
    runtime_assets_verified = 0
    agent_source_files_verified = 0
    python_bytecode_cache_empty = False
    host_arm64 = False
    docker_server_arm64 = False
    model_admission: ModelAdmission | None = None
    unix_bridge_verified = False
    relay_asset: Path | None = None
    policy_asset: Path | None = None
    node_prefix: Path | None = None
    agent_prefixes: dict[str, Path] = {}
    agent_source_root: Path | None = None
    python_pycache_root: Path | None = None
    server: Any | None = None
    sampler: TelemetrySampler | None = None
    bridge: _UnixBridgeHandle | None = None
    relay_credential = ""
    attempts: list[HarborAttempt] = []
    run_statuses: list[HarborRunStatus] = []
    invocations_started = 0
    campaign_summary: dict[str, Any] | None = None
    primary_error: BaseException | None = None
    cutoff_reached = False
    stop_reason = "artifact_admission"
    cleanup_values = {
        "harbor_resources_removed": False,
        "bridge_stopped": True,
        "bridge_socket_removed": True,
        "server_stopped": True,
        "telemetry_stopped": True,
        "key_removed": True,
        "raw_jobs_private_retained": False,
        "raw_jobs_key_free": False,
        "derived_dataset_removed": True,
        "runtime_overlays_removed": True,
        "staged_assets_removed": True,
        "socket_directory_removed": True,
        "npm_scratch_removed": True,
        "agent_source_removed": True,
        "python_pycache_removed": True,
    }

    try:
        validation_started = monotonic()
        artifacts = validate_llamacpp_artifacts(runtime_model, workspace=workspace)
        validation_s = monotonic() - validation_started
        if validation_s < 0:
            raise CampaignLifecycleError("artifact admission clock moved backwards")
        _model_provenance(model, artifacts)

        stop_reason = "dataset_admission"
        patch = derive_private_task_dataset(
            campaign,
            source_checkout=dataset_checkout,
            destination=derived_dir,
            repo_root=workspace,
        )

        stop_reason = "npm_admission"
        npm_admission = acquire_npm_artifacts(
            campaign,
            output_dir=npm_dir,
            repo_root=workspace,
        )
        npm_artifacts_verified = len(npm_admission.artifacts)

        stop_reason = "runtime_admission"
        verify_harbor_runtime(campaign, harbor_runtime_root, repo_root=workspace)
        harbor_runtime_verified = True
        node_prefix, agent_prefixes = _verify_tool_prefixes(
            campaign, prefix_root=tool_prefix_root, workspace=workspace
        )
        node_tree_verified = True
        agent_trees_verified = len(agent_prefixes)
        relay_asset, policy_asset = _stage_runtime_assets(
            campaign, run_root=run_root, workspace=workspace
        )
        runtime_assets_verified = 2
        agent_source_root, python_pycache_root = stage_head_agent_source(
            campaign,
            workspace=workspace,
            run_root=run_root,
            revision=git_revision,
        )
        agent_source_files_verified = 1
        python_bytecode_cache_empty = True

        stop_reason = "platform_admission"
        host_arm64, docker_server_arm64 = admit_native_platform()

        stop_reason = "preflight"
        _preflight(runtime_model)
        relay_credential = create_ephemeral_key(key_path)

        sampler = TelemetrySampler(telemetry_path)
        sampler.start()
        sampler.set_phase("model_start")
        stop_reason = "model_start"
        server = start_llamacpp(
            runtime_model,
            workspace=workspace,
            server_log_path=server_log_path,
            process_state_path=process_state_path,
            validated_artifacts=artifacts,
            artifact_validation_s=validation_s,
        )

        sampler.set_phase("model_admission")
        stop_reason = "model_admission"
        model_admission = run_model_admission(
            base_url=server.base_url,
            served_name=campaign.model.served_name,
        )

        stop_reason = "bridge_start"
        bridge = _start_unix_bridge(
            socket_path=socket_path,
            target_port=_model_port(server.base_url),
            relay_credential=relay_credential,
            idle_timeout_s=campaign.execution.agent_timeout_s + 120.0,
        )
        cleanup_values["bridge_stopped"] = False
        cleanup_values["bridge_socket_removed"] = False
        stop_reason = "bridge_admission"
        unix_bridge_verified = admit_unix_bridge(
            bridge,
            relay_credential=relay_credential,
            served_name=campaign.model.served_name,
        )

        deadline = wall_started + campaign.execution.hard_campaign_cutoff_s
        sampler.set_phase("harbor_trials")
        stop_reason = "trial_failure"
        for trial in trials:
            bridge.assert_healthy()
            trial_now = monotonic()
            trial_timeout = _trial_timeout_s(
                campaign, deadline=deadline, now=trial_now
            )
            if trial_timeout is None:
                cutoff_reached = True
                stop_reason = "hard_cutoff"
                break

            # Re-admit the actual launcher tree immediately before every argv,
            # and rebuild the overlay from freshly traversed Node/agent trees.
            harbor_admission = verify_harbor_runtime(
                campaign, harbor_runtime_root, repo_root=workspace
            )
            overlay_admission = prepare_runtime_overlay(
                campaign,
                trial,
                destination=overlay_dir / f"trial-{trial.index:02d}.json",
                node_prefix=node_prefix,
                agent_prefix=agent_prefixes[trial.agent_id],
                relay_script=relay_asset,
                network_policy_script=policy_asset,
                run_socket_dir=socket_dir,
                repo_root=workspace,
            )
            invocation = build_harbor_invocation(
                campaign,
                trial=trial,
                npm_artifact_admission=npm_admission,
                runtime_overlay_admission=overlay_admission,
                harbor_runtime_admission=harbor_admission,
                agent_source_root=agent_source_root,
                python_pycache_root=python_pycache_root,
                derived_dataset=patch,
                jobs_dir=jobs_dir,
                base_url=f"http://{campaign.relay.listen_host}:{campaign.relay.port}/v1",
                repo_root=workspace,
            )
            invocations_started += 1
            status = run_harbor_invocation(
                invocation,
                lock=lock,
                timeout_s=trial_timeout,
            )
            run_statuses.append(status)

            def load_admitted_result() -> Any:
                verify_empty_python_pycache(python_pycache_root, repo_root=workspace)
                return load_trial_job_result(
                    invocation, jobs_dir=jobs_dir, repo_root=workspace
                )

            _record_status_then_load(
                trial=trial,
                status=status,
                attempts=attempts,
                loader=load_admitted_result,
            )
            bridge.assert_healthy()
            campaign_summary = summarize_campaign_results(
                campaign,
                attempts,
                network_policy_patch_digest=patch.digest,
                npm_artifact_admission=npm_admission,
                campaign_cutoff_reached=False,
            )
            if _deadline_limited_timeout(
                status,
                remaining_s=deadline - trial_now,
                per_trial_cap_s=float(campaign.execution.trial_wall_timeout_s),
            ):
                cutoff_reached = True
                stop_reason = "hard_cutoff"
                break
            gate = _attempt_gate(attempts, campaign_summary)
            if gate is not None:
                stop_reason = gate
                break
        else:
            stop_reason = "completed"
    except CampaignInterrupted as error:
        stop_reason = "interrupted"
        primary_error = error
    except BaseException as error:
        primary_error = error
    finally:
        if patch is not None and npm_admission is not None:
            try:
                campaign_summary = summarize_campaign_results(
                    campaign,
                    attempts,
                    network_policy_patch_digest=patch.digest,
                    npm_artifact_admission=npm_admission,
                    campaign_cutoff_reached=cutoff_reached,
                )
            except BaseException as error:
                if primary_error is None:
                    primary_error = error
                    stop_reason = "trial_failure"

        cleanup_values["harbor_resources_removed"] = _harbor_cleanup_certified(
            invocations_started, run_statuses
        )
        if bridge is not None:
            try:
                stopped, socket_removed = bridge.close()
                cleanup_values["bridge_stopped"] = stopped
                cleanup_values["bridge_socket_removed"] = socket_removed
            except BaseException:
                cleanup_values["bridge_stopped"] = False
                cleanup_values["bridge_socket_removed"] = False
        if server is not None:
            cleanup_values["server_stopped"] = False
            try:
                server.stop()
                cleanup_values["server_stopped"] = True
            except BaseException:
                pass
        if sampler is not None:
            cleanup_values["telemetry_stopped"] = False
            try:
                sampler.stop()
                sampler_thread = getattr(sampler, "_thread", None)
                cleanup_values["telemetry_stopped"] = (
                    sampler_thread is None or not sampler_thread.is_alive()
                )
            except BaseException:
                pass

        cleanup_values["key_removed"] = remove_ephemeral_key(key_path)
        cleanup_values["derived_dataset_removed"] = _remove_owned_tree(
            derived_dir, owner=run_root
        )
        cleanup_values["runtime_overlays_removed"] = _remove_owned_tree(
            overlay_dir, owner=run_root
        )
        cleanup_values["staged_assets_removed"] = _remove_owned_tree(
            run_root / "runtime-assets", owner=run_root
        )
        cleanup_values["socket_directory_removed"] = _remove_owned_tree(
            socket_dir, owner=run_root
        )
        cleanup_values["npm_scratch_removed"] = all(
            _remove_owned_tree(npm_dir / name, owner=run_root)
            for name in ("home", "cache")
        )
        if python_pycache_root is not None:
            try:
                verify_empty_python_pycache(python_pycache_root, repo_root=workspace)
            except BaseException as error:
                python_bytecode_cache_empty = False
                if primary_error is None:
                    primary_error = error
                    stop_reason = "runtime_admission"
        cleanup_values["agent_source_removed"] = _remove_owned_tree(
            run_root / "agent-source", owner=run_root
        )
        cleanup_values["python_pycache_removed"] = _remove_owned_tree(
            run_root / "python-pycache", owner=run_root
        )
        (
            cleanup_values["raw_jobs_private_retained"],
            cleanup_values["raw_jobs_key_free"],
        ) = certify_private_raw_jobs(
            run_root,
            owner=external_root,
            relay_credential=relay_credential,
        )

    if campaign_summary is None or artifacts is None:
        # Without the adapter's exact empty schema or exact model admission,
        # there is no safe alternate output shape to serialize.
        if primary_error is not None:
            raise CampaignLifecycleError("campaign stopped before scalar admission") from primary_error
        raise CampaignLifecycleError("campaign stopped before scalar admission")

    cleanup = CleanupStatus(**cleanup_values)
    if not all(asdict(cleanup).values()):
        stop_reason = "cleanup_failure"
    admission = RuntimeAdmission(
        artifact_validation=True,
        harbor_runtime_verified=harbor_runtime_verified,
        node_tree_verified=node_tree_verified,
        agent_trees_verified=agent_trees_verified,
        npm_artifacts_verified=npm_artifacts_verified,
        runtime_assets_verified=runtime_assets_verified,
        agent_source_files_verified=agent_source_files_verified,
        python_bytecode_cache_empty=python_bytecode_cache_empty,
        host_arm64=host_arm64,
        docker_server_arm64=docker_server_arm64,
        model=model_admission,
        unix_bridge_verified=unix_bridge_verified,
    )
    return _LockedOutcome(
        campaign_summary=campaign_summary,
        model_provenance=_model_provenance(model, artifacts),
        admission=admission,
        cleanup=cleanup,
        trials_started=len(attempts),
        trials_completed=sum(1 for attempt in attempts if attempt.job_result is not None),
        cutoff_reached=cutoff_reached,
        stop_reason=stop_reason,
        primary_error=primary_error,
        relay_credential=relay_credential,
    )


def run_campaign(
    *,
    workspace: Path,
    campaign_path: Path,
    harbor_runtime_root: Path,
    tool_prefix_root: Path,
    dataset_checkout: Path,
    raw_root: Path,
    output_name: str,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run the frozen campaign under one continuous global benchmark lock."""

    wall_started = monotonic()
    started_at = utc_now()
    workspace = workspace.resolve(strict=True)
    external_root = prepare_external_raw_root(raw_root, workspace=workspace)
    campaign = load_campaign(campaign_path)
    models_path = workspace / "manifests" / "models.toml"
    if models_path.resolve(strict=True) != DEFAULT_MODELS_PATH.resolve(strict=True):
        raise CampaignLifecycleError("pinned model manifest path changed")
    models = load_models(models_path)
    if campaign.model.profile not in models:
        raise CampaignLifecycleError("campaign model profile is missing")
    model = models[campaign.model.profile]
    _cross_validate_model(campaign, model)
    trials = _validate_trial_order(campaign)
    git_revision, git_clean = _git_provenance(workspace)
    if not git_clean:
        raise CampaignLifecycleError("campaign requires a clean repository")

    run_root = _create_run_directory(external_root, campaign.id)
    scalar_output = _scalar_output_path(run_root, output_name)
    runtime_model = _runtime_model(model, campaign.id)
    with termination_guard(), hold_campaign_lock(workspace) as lock:
        if results_lock_path(workspace) != workspace / "results" / ".sparkbench.lock":
            raise CampaignLifecycleError("shared benchmark lock path changed")
        lock.assert_active()
        outcome = _execute_campaign_locked(
            lock=lock,
            workspace=workspace,
            external_root=external_root,
            run_root=run_root,
            campaign=campaign,
            model=model,
            runtime_model=runtime_model,
            trials=trials,
            harbor_runtime_root=harbor_runtime_root,
            tool_prefix_root=tool_prefix_root,
            dataset_checkout=dataset_checkout,
            git_revision=git_revision,
            wall_started=wall_started,
            monotonic=monotonic,
        )
        # Cleanup is complete before this continuous lock is released.
        lock.assert_active()

    cleanup_complete = all(asdict(outcome.cleanup).values())
    completed = (
        outcome.trials_started == EXPECTED_TRIALS
        and outcome.trials_completed == EXPECTED_TRIALS
        and outcome.stop_reason == "completed"
        and outcome.primary_error is None
        and cleanup_complete
    )
    status = "completed" if completed else (
        "partial" if outcome.trials_started else "aborted"
    )
    stop_reason = outcome.stop_reason if cleanup_complete else "cleanup_failure"
    envelope = build_lifecycle_envelope(
        campaign=campaign,
        campaign_summary=outcome.campaign_summary,
        model_provenance=outcome.model_provenance,
        git_revision=git_revision,
        git_clean=git_clean,
        admission=outcome.admission,
        cleanup=outcome.cleanup,
        status=status,
        stop_reason=stop_reason,
        started_at=started_at,
        finished_at=utc_now(),
        elapsed_s=max(monotonic() - wall_started, 0.0),
        trials_started=outcome.trials_started,
        trials_completed=outcome.trials_completed,
        cutoff_reached=outcome.cutoff_reached,
        sensitive_values=(outcome.relay_credential,),
    )
    write_scalar_result(scalar_output, envelope, campaign=campaign)
    if outcome.primary_error is not None:
        raise CampaignLifecycleError(
            f"campaign stopped safely at lifecycle stage {stop_reason}"
        ) from outcome.primary_error
    if not cleanup_complete:
        raise CampaignLifecycleError("campaign cleanup could not be certified")
    return envelope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pinned Qwen3-Coder-Next Harbor campaign safely."
    )
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN_PATH)
    parser.add_argument(
        "--harbor-runtime-root", type=Path, default=DEFAULT_HARBOR_RUNTIME
    )
    parser.add_argument(
        "--tool-prefix-root", type=Path, default=DEFAULT_TOOL_PREFIX_ROOT
    )
    parser.add_argument(
        "--dataset-checkout", type=Path, default=DEFAULT_DATASET_CHECKOUT
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-name", default="campaign-result.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(__file__).resolve().parents[1]
    result = run_campaign(
        workspace=workspace,
        campaign_path=args.campaign,
        harbor_runtime_root=args.harbor_runtime_root,
        tool_prefix_root=args.tool_prefix_root,
        dataset_checkout=args.dataset_checkout,
        raw_root=args.raw_root,
        output_name=args.output_name,
    )
    print(
        json.dumps(
            {
                "campaign_id": result["campaign_id"],
                "status": result["status"],
                "trials_completed": result["execution"]["trials_completed"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
