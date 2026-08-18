"""Pinned, privacy-safe Harbor Terminal-Bench campaign support.

The outer campaign owner must acquire :func:`hold_campaign_lock` *before*
starting the model server.  It must retain that same lock while the server,
authenticated Docker bridge, and every Harbor trial are alive, and release it
only after all owned processes and containers have been stopped.  Acquiring a
fresh lock around each trial would leave unsafe gaps and is not supported by
the runner API.

Harbor's raw job directories contain prompts, completions, tool payloads, and
identifiers.  This module therefore requires raw jobs and the derived task tree
to live outside the repository.  Only the canonical scalar projection returned
by :func:`summarize_campaign_results` is suitable for later evidence export.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import base64
import binascii
import fcntl
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import time
import tomllib
from typing import Any
from urllib.parse import urlsplit

from .harbor_runtime_assets import (
    RuntimeAssetError,
    TreeAdmission,
    verify_normalized_tree,
)


SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 2
SUMMARY_PROTOCOL = "harbor-terminal-scalar-v2"
MAX_API_KEY_BYTES = 16 * 1024
MAX_RAW_JSON_BYTES = 64 * 1024 * 1024
MAX_IMAGE_INSPECT_BYTES = 1024 * 1024
MAX_IMAGE_LAYERS = 4096
MAX_NPM_TARBALL_BYTES = 512 * 1024 * 1024
MAX_TASK_FILE_BYTES = 128 * 1024 * 1024
RELAY_LISTEN_HOST = "127.0.0.1"
RELAY_PORT = 18_080
RELAY_BASE_URL = f"http://{RELAY_LISTEN_HOST}:{RELAY_PORT}/v1"
RELAY_SENTINEL_HOST = "sparkbench-relay.invalid"
RELAY_PLACEHOLDER_API_KEY = "sparkbench-relay-placeholder-v1"
RELAY_UDS_PATH = "/run/sparkbench/model.sock"
RELAY_INTERNAL_KEY_PATH = "/run/sparkbench/internal-api-key"
NETWORK_ADMISSION_FILENAME = "sparkbench-network-admission.json"
NETWORK_ADMISSION_KEYS = (
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
RELAY_NODE_IMAGE = (
    "node@sha256:4f77a690f2f8946ab16fe1e791a3ac0667ae1c3575c3e4d0d4589e9ed5bfaf3d"
)
RELAY_SCRIPT_SHA256 = (
    "sha256:ebf7e377c0ae2b5597653979107cf39fd2582c7d200f6f08179db6c724fc5a81"
)
NETWORK_POLICY_SHA256 = (
    "sha256:5b86e7b2f9f6b05e1f54318964a1db015fe0891b1d72fe4a59c02a9d55417f2f"
)
NODE_VERSION = "v22.22.1"
NPM_BUILDER_VERSION = "10.9.4"
NODE_BINARY_SHA256 = (
    "sha256:1ea16f5e8d590ca819f1db81e61a6bc4c8e7d8b69776c9236f8ebbf717113225"
)
NODE_TREE_SHA256 = (
    "sha256:fe1c8f2403bdf6c1dbd38421ba26b8094b9faa35efc51bc2b6fb375b2fe9e147"
)
NODE_TREE_SIZE_BYTES = 121_914_536
NODE_MOUNT_PATH = "/opt/sparkbench/node"
AGENT_MOUNT_PATH = "/opt/sparkbench/agent"

HARBOR_SOURCE = "harbor-framework/harbor"
HARBOR_REVISION = "64afbbcb62165950301e1a6407c729aa26d844ff"
HARBOR_VERSION = "0.21.0"
HARBOR_RUNTIME_TREE_SHA256 = (
    "sha256:9fe1a144b228d5f540726696d9d29a5ea01e9051bb6b6e293c3c23f2a5d0386b"
)
HARBOR_RUNTIME_TREE_SIZE_BYTES = 2_106_386_494
HARBOR_RUNTIME_TREE_ENTRIES = 66_729
HARBOR_RUNTIME_TREE_FILES = 57_355
HARBOR_RUNTIME_TREE_LINKS = 1_053
HARBOR_EXECUTABLE_PATH = ".venv/bin/harbor"
HARBOR_EXECUTABLE_SIZE_BYTES = 348
HARBOR_EXECUTABLE_SHA256 = (
    "sha256:95c7c1b6da6209f66179ba411e4b22addd6ec96dd4f206ca69564c8dde337f9d"
)
HARBOR_AGENT_SOURCE_SHA256 = (
    "sha256:cc898eea830fc6a06505c7b07092234489468baf92b802e8c73bf94e133ae8c3"
)
HARBOR_AGENT_SOURCE_FILES = (
    "bench/harbor_pinned_agents.py",
)
HARBOR_PYTHON_LAUNCHER_PATH = ".venv/bin/python"
HARBOR_PYTHON_LAUNCHER_TARGET = "../../.python-runtime/bin/python3.13"
HARBOR_PYTHON_VERSION = "3.13.11"
HARBOR_PYTHON_PATH = ".python-runtime/bin/python3.13"
HARBOR_PYTHON_SIZE_BYTES = 21_558_296
HARBOR_PYTHON_SHA256 = (
    "sha256:f9a43df6c18466648773aacbd1b1a3db14a4bc23d222b866b8e5e088f9a4675a"
)
DATASET_SOURCE = "harbor-framework/terminal-bench-2-1"
DATASET_REVISION = "7131e4375048a0e408a8fb404b5f499d726b695b"
DATASET_VERSION = "2.1"
CAMPAIGN_ID = "qwen3-coder-next-harbor-terminal-2026-08-17"
MODEL_PROFILE = "qwen3-coder-next-80b-a3b-ud-q4-k-xl-llamacpp"
MODEL_SERVED_NAME = "Qwen/Qwen3-Coder-Next"
MODEL_CONTEXT_TOKENS = 65_536
MODEL_MAX_OUTPUT_TOKENS = 8_192
AGENT_TIMEOUT_S = 900
TRIAL_WALL_TIMEOUT_S = 3_600
HARD_CAMPAIGN_CUTOFF_S = 23_400
RESERVE_FOR_AUDIT_S = 5_400
SERVER_DEFAULT_TEMPERATURE = 1.0
SERVER_DEFAULT_TOP_P = 0.95
SERVER_DEFAULT_TOP_K = 40

EXPECTED_TASKS = (
    "build-cython-ext",
    "cancel-async-tasks",
    "fix-code-vulnerability",
    "kv-store-grpc",
    "polyglot-c-py",
    "query-optimize",
)
EXPECTED_AGENT_IDS = ("qwen-coder", "opencode")
EXPECTED_TRIAL_ORDER = (
    "build-cython-ext:qwen-coder",
    "build-cython-ext:opencode",
    "cancel-async-tasks:opencode",
    "cancel-async-tasks:qwen-coder",
    "fix-code-vulnerability:qwen-coder",
    "fix-code-vulnerability:opencode",
    "kv-store-grpc:opencode",
    "kv-store-grpc:qwen-coder",
    "polyglot-c-py:qwen-coder",
    "polyglot-c-py:opencode",
    "query-optimize:opencode",
    "query-optimize:qwen-coder",
)
EXPECTED_NPM_PACKAGES = {
    "qwen-coder": "@qwen-code/qwen-code",
    "opencode": "opencode-ai",
}
PINNED_AGENT_IMPORTS = {
    "qwen-coder": "bench.harbor_pinned_agents:PinnedQwenCode",
    "opencode": "bench.harbor_pinned_agents:PinnedOpenCode",
}
EXPECTED_AGENT_PINS = {
    "qwen-coder": {
        "version": "0.21.13",
        "npm_package": "@qwen-code/qwen-code",
        "npm_integrity": (
            "sha512-xXyOK166EEeTjHUh9BEdH4h7Afhz53k+jJAv5mgFxQYJbHf25oxif6WRk6jvY"
            "GwMxpEdL3vaoURP/QQiplN9lQ=="
        ),
        "npm_shasum": "ca3ec34b0cd6179fe08ed757b861241c28178460",
        "source": "QwenLM/qwen-code",
        "revision": "d959015974302fb60ebd99adb81a68c2f482eaa3",
        "install_tree_sha256": (
            "sha256:d8116cabe714aad8649d55cabe709903ce13d1944edafc13e9670f48b7f5a579"
        ),
        "install_tree_size_bytes": 132_176_032,
        "platform_package": None,
        "platform_integrity": None,
        "platform_shasum": None,
    },
    "opencode": {
        "version": "1.18.18",
        "npm_package": "opencode-ai",
        "npm_integrity": (
            "sha512-J+5HFq8tf+wPBBpBpMPSNjSytF2/EkNWYfFZh4si1d9auFbQriqDyqZv+vFUsL"
            "WERfdMU32Eajwuiq3rKBvZLQ=="
        ),
        "npm_shasum": "a78971b6affe7ed27a207218465d1a80e36a018c",
        "source": "anomalyco/opencode",
        "revision": "31406ccc51b4bd2a4e1e086b2bcaa5f7f804f26d",
        "install_tree_sha256": (
            "sha256:39a28a0bb7c08720a54d1d533e26e5280f4f494a029be9751f2d2b3b41161883"
        ),
        "install_tree_size_bytes": 366_571_144,
        "platform_package": "opencode-linux-arm64",
        "platform_integrity": (
            "sha512-e8D3g0qJEIzawEg2+ygW3vkZjAYL2ssyAx4GbihjwXwZFvlZZy5zRWWzdz5KLB"
            "oHSTl0FB73vNtnNeXONyHpVQ=="
        ),
        "platform_shasum": "5d4952bb8c1c3bbcccc52bcd07a540a845e31408",
    },
}

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "id",
        "description",
        "model",
        "harbor",
        "dataset",
        "agents",
        "execution",
        "admission",
        "relay",
        "toolchain",
    }
)
_MODEL_KEYS = frozenset(
    {
        "profile",
        "served_name",
        "context_tokens",
        "max_output_tokens",
        "parallel",
    }
)
_HARBOR_KEYS = frozenset(
    {
        "source",
        "revision",
        "version",
        "environment",
        "force_build",
        "n_attempts",
        "n_concurrent",
        "max_retries",
        "runtime_tree_sha256",
        "runtime_tree_size_bytes",
        "runtime_tree_entries",
        "runtime_tree_files",
        "runtime_tree_links",
        "executable_path",
        "executable_size_bytes",
        "executable_sha256",
        "agent_source_sha256",
        "python_launcher_path",
        "python_launcher_target",
        "python_version",
        "python_path",
        "python_size_bytes",
        "python_sha256",
    }
)
_DATASET_KEYS = frozenset({"source", "revision", "version", "tasks"})
_AGENT_REQUIRED_KEYS = frozenset(
    {
        "id",
        "version",
        "npm_package",
        "npm_integrity",
        "npm_shasum",
        "source",
        "revision",
        "install_tree_sha256",
        "install_tree_size_bytes",
    }
)
_AGENT_PLATFORM_KEYS = frozenset(
    {"platform_package", "platform_integrity", "platform_shasum"}
)
_EXECUTION_KEYS = frozenset(
    {
        "canary_task",
        "agent_timeout_s",
        "trial_wall_timeout_s",
        "hard_campaign_cutoff_s",
        "reserve_for_audit_s",
        "trial_order",
        "bridge_requires_bearer_auth",
        "bridge_target_is_loopback",
        "raw_trajectories_are_ignored",
        "publish_scalar_evidence_only",
        "harness_requests_are_not_rewritten",
        "server_default_temperature",
        "server_default_top_p",
        "server_default_top_k",
    }
)
_RELAY_KEYS = frozenset(
    {
        "listen_host",
        "port",
        "sentinel_host",
        "placeholder_api_key",
        "uds_path",
        "internal_key_path",
        "node_image",
        "relay_script_sha256",
        "network_policy_sha256",
    }
)
_TOOLCHAIN_KEYS = frozenset(
    {
        "node_version",
        "npm_builder_version",
        "node_binary_sha256",
        "node_tree_sha256",
        "node_tree_size_bytes",
        "node_mount_path",
        "agent_mount_path",
    }
)
_ADMISSION_KEYS = frozenset(
    {
        "require_native_arm64_build",
        "require_canary_container_cleanup",
        "require_model_tool_call",
        "require_verifier_result",
        "reject_unexpected_network",
        "reject_raw_payload_publication",
    }
)
_TRUE_EXECUTION_FLAGS = (
    "bridge_requires_bearer_auth",
    "bridge_target_is_loopback",
    "raw_trajectories_are_ignored",
    "publish_scalar_evidence_only",
    "harness_requests_are_not_rewritten",
)
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_REPOSITORY_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA1_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_NPM_PACKAGE_PATTERN = re.compile(r"^(?:@[a-z0-9._-]+/)?[a-z0-9._-]+$")
_IMAGE_REFERENCE_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9._-]+)?"
    r"(?:@sha256:[0-9a-f]{64})?$"
)
_SEMVER_PATTERN = re.compile(
    r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_SAFE_ENV_KEYS = frozenset(
    {
        "DOCKER_CONFIG",
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "PATH",
        "SHELL",
        "TERM",
        "TMPDIR",
        "USER",
        "XDG_CACHE_HOME",
    }
)
_ALLOWED_EXCEPTION_CLASSES = frozenset(
    {
        "AddTestsDirError",
        "AgentAuthenticationError",
        "AgentSafetyRefusalError",
        "AgentSetupTimeoutError",
        "AgentTimeoutError",
        "ApiConnectionClosedError",
        "ApiConnectionError",
        "ApiError",
        "ApiInternalServerError",
        "ApiKeyRejectedError",
        "ApiOverloadedError",
        "ApiProviderResourceNotFoundError",
        "ApiRateLimitError",
        "ApiResponseStalledError",
        "ApiUsageLimitError",
        "AuthenticationError",
        "CancelledError",
        "CampaignCutoffError",
        "ContextLengthExceededError",
        "ContextWindowExceededError",
        "DownloadVerifierDirError",
        "DspyImportError",
        "EnvironmentStartTimeoutError",
        "GKEExecStreamClosedError",
        "HarborProcessError",
        "HarborCleanupError",
        "HealthcheckError",
        "MemoryLimitExceededError",
        "MissingExtraError",
        "ModelNotFoundError",
        "NetworkConnectionError",
        "NonZeroAgentExitCodeError",
        "NotAuthenticatedError",
        "OutputLengthExceededError",
        "OutputTokenExceededError",
        "RegradeError",
        "RewardFileEmptyError",
        "RewardFileNotFoundError",
        "RuntimeRequestError",
        "SandboxBuildFailedError",
        "ServiceOperationsUnsupportedError",
        "UnknownApiError",
        "VerifierOutputParseError",
        "VerifierTimeoutError",
    }
)


class HarborCampaignError(ValueError):
    """Raised when campaign input or raw output violates the protocol."""


@dataclass(frozen=True, slots=True)
class CampaignModel:
    profile: str
    served_name: str
    context_tokens: int
    max_output_tokens: int
    parallel: int


@dataclass(frozen=True, slots=True)
class HarborPin:
    source: str
    revision: str
    version: str
    environment: str
    force_build: bool
    n_attempts: int
    n_concurrent: int
    max_retries: int
    runtime_tree_sha256: str
    runtime_tree_size_bytes: int
    runtime_tree_entries: int
    runtime_tree_files: int
    runtime_tree_links: int
    executable_path: str
    executable_size_bytes: int
    executable_sha256: str
    agent_source_sha256: str
    python_launcher_path: str
    python_launcher_target: str
    python_version: str
    python_path: str
    python_size_bytes: int
    python_sha256: str


@dataclass(frozen=True, slots=True)
class DatasetPin:
    source: str
    revision: str
    version: str
    tasks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AgentPin:
    id: str
    version: str
    npm_package: str
    npm_integrity: str
    npm_shasum: str
    source: str
    revision: str
    install_tree_sha256: str
    install_tree_size_bytes: int
    platform_package: str | None = None
    platform_integrity: str | None = None
    platform_shasum: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    canary_task: str
    agent_timeout_s: int
    trial_wall_timeout_s: int
    hard_campaign_cutoff_s: int
    reserve_for_audit_s: int
    trial_order: tuple[str, ...]
    bridge_requires_bearer_auth: bool
    bridge_target_is_loopback: bool
    raw_trajectories_are_ignored: bool
    publish_scalar_evidence_only: bool
    harness_requests_are_not_rewritten: bool
    server_default_temperature: float
    server_default_top_p: float
    server_default_top_k: int


@dataclass(frozen=True, slots=True)
class RelayPin:
    listen_host: str
    port: int
    sentinel_host: str
    placeholder_api_key: str
    uds_path: str
    internal_key_path: str
    node_image: str
    relay_script_sha256: str
    network_policy_sha256: str


@dataclass(frozen=True, slots=True)
class ToolchainPin:
    node_version: str
    npm_builder_version: str
    node_binary_sha256: str
    node_tree_sha256: str
    node_tree_size_bytes: int
    node_mount_path: str
    agent_mount_path: str


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    schema_version: int
    id: str
    description: str
    model: CampaignModel
    harbor: HarborPin
    dataset: DatasetPin
    agents: tuple[AgentPin, ...]
    relay: RelayPin
    toolchain: ToolchainPin
    execution: ExecutionSpec
    admission: tuple[tuple[str, bool], ...]

    def agent(self, agent_id: str) -> AgentPin:
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        raise HarborCampaignError("Trial references an unknown campaign agent")


@dataclass(frozen=True, slots=True)
class TrialSpec:
    index: int
    task_id: str
    agent_id: str


@dataclass(frozen=True, slots=True)
class TaskPatch:
    task_id: str
    source_task_toml_sha256: str
    source_task_toml_mode: int
    derived_task_toml_sha256: str
    derived_task_toml_mode: int
    unchanged_tree_sha256: str
    verifier_script_sha256: str
    source_verifier_script_mode: int
    derived_verifier_script_mode: int


@dataclass(frozen=True, slots=True)
class NetworkPolicyPatch:
    dataset_revision: str
    digest: str
    tasks: tuple[TaskPatch, ...]
    dataset_dir: Path = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class HarborInvocation:
    trial: TrialSpec
    job_name: str
    timeout_s: int
    npm_artifact_admission_digest: str
    runtime_overlay_admission_digest: str
    harbor_runtime_admission_digest: str
    agent_source_admission_digest: str
    task_image: str = field(repr=False)
    relay_image: str = field(repr=False)
    workspace_root: Path = field(repr=False)
    agent_source_root: Path = field(repr=False)
    python_pycache_root: Path = field(repr=False)
    raw_job_dir: Path = field(repr=False)
    argv: tuple[str, ...] = field(repr=False)
    env: Mapping[str, str] = field(repr=False)


@dataclass(frozen=True, slots=True)
class HarborRunStatus:
    trial: TrialSpec
    exit_code: int | None
    timed_out: bool
    wall_s: float
    main_image_id: str | None
    main_image_fingerprint: str | None
    main_image_arm64: bool
    relay_image_arm64: bool
    built_image_cleanup_succeeded: bool
    setup_relay_rejected: bool
    agent_relay_passed: bool
    wrong_auth_rejected: bool
    other_loopback_rejected: bool
    gost_rejected: bool
    dns_rejected: bool
    gateway_rejected: bool
    public_rejected: bool
    capabilities_dropped: bool
    cleanup_succeeded: bool
    containers_found: int
    containers_removed: int
    networks_found: int
    networks_removed: int
    volumes_found: int
    volumes_removed: int


@dataclass(frozen=True, slots=True)
class HarborCleanupStatus:
    succeeded: bool
    containers_found: int
    containers_removed: int
    networks_found: int
    networks_removed: int
    volumes_found: int
    volumes_removed: int


@dataclass(frozen=True, slots=True)
class DockerResourceSnapshot:
    containers: frozenset[str] = field(repr=False)
    networks: frozenset[str] = field(repr=False)
    volumes: frozenset[str] = field(repr=False)


@dataclass(frozen=True, slots=True)
class HarborAttempt:
    trial: TrialSpec
    status: HarborRunStatus
    job_result: "HarborRawResult | None" = field(repr=False)


@dataclass(frozen=True, slots=True)
class HarborRawResult:
    job: Mapping[str, Any] = field(repr=False)
    trial: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True, slots=True)
class NpmArtifactRecord:
    package: str
    version: str
    size_bytes: int
    shasum: str
    integrity: str


@dataclass(frozen=True, slots=True)
class NpmArtifactAdmission:
    digest: str
    artifacts: tuple[NpmArtifactRecord, ...]


@dataclass(frozen=True, slots=True)
class RuntimeOverlayAdmission:
    trial: TrialSpec
    digest: str
    compose_sha256: str
    node_tree: TreeAdmission = field(repr=False)
    agent_tree: TreeAdmission = field(repr=False)
    compose_path: Path = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class HarborRuntimeAdmission:
    digest: str
    tree: TreeAdmission = field(repr=False)
    executable_path: Path = field(repr=False, compare=False)
    python_launcher_path: Path = field(repr=False, compare=False)
    python_path: Path = field(repr=False, compare=False)


@dataclass(slots=True)
class CampaignLock:
    """Capability proving that the outer owner still holds the global lock."""

    descriptor: int = field(repr=False)
    active: bool = field(default=True, repr=False)

    def assert_active(self) -> None:
        if not self.active or self.descriptor < 0:
            raise HarborCampaignError(
                "The Harbor run requires one continuously held campaign lock"
            )


def _expect_keys(value: Mapping[str, Any], expected: frozenset[str], context: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise HarborCampaignError(f"{context} keys are invalid ({'; '.join(details)})")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise HarborCampaignError(f"{context} must be a table")
    return value


def _string(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise HarborCampaignError(f"{context} must be one non-empty string")
    return value


def _integer(value: Any, context: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HarborCampaignError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarborCampaignError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HarborCampaignError(f"{context} must be finite")
    return result


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise HarborCampaignError(f"{context} must be boolean")
    return value


def _pinned_string(value: Any, pattern: re.Pattern[str], context: str) -> str:
    result = _string(value, context)
    if pattern.fullmatch(result) is None:
        raise HarborCampaignError(f"{context} is not an exact pin")
    return result


def _integrity(value: Any, context: str) -> str:
    result = _string(value, context)
    if not result.startswith("sha512-"):
        raise HarborCampaignError(f"{context} must be an npm sha512 integrity pin")
    try:
        decoded = base64.b64decode(result.removeprefix("sha512-"), validate=True)
    except (binascii.Error, ValueError) as error:
        raise HarborCampaignError(f"{context} is not valid base64") from error
    if len(decoded) != 64:
        raise HarborCampaignError(f"{context} must encode one SHA-512 digest")
    return result


def _container_path(value: Any, context: str) -> str:
    result = _string(value, context)
    parsed = PurePosixPath(result)
    if (
        not parsed.is_absolute()
        or result != parsed.as_posix()
        or ".." in parsed.parts
        or result == "/"
    ):
        raise HarborCampaignError(f"{context} must be one normalized absolute path")
    return result


def _relative_runtime_path(value: Any, context: str) -> str:
    result = _string(value, context)
    parsed = PurePosixPath(result)
    if (
        parsed.is_absolute()
        or result != parsed.as_posix()
        or ".." in parsed.parts
        or result in {"", "."}
    ):
        raise HarborCampaignError(f"{context} must be one normalized relative path")
    return result


def load_campaign(path: Path) -> CampaignSpec:
    """Load the one frozen Harbor campaign with strict keys, types, and pins."""

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise HarborCampaignError("Could not load the Harbor campaign manifest") from error
    _expect_keys(raw, _TOP_LEVEL_KEYS, "campaign")
    schema_version = _integer(raw["schema_version"], "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise HarborCampaignError("Unsupported Harbor campaign schema version")

    campaign_id = _string(raw["id"], "id")
    if campaign_id != CAMPAIGN_ID:
        raise HarborCampaignError("Campaign id is invalid")
    description = _string(raw["description"], "description")

    model_raw = _mapping(raw["model"], "model")
    _expect_keys(model_raw, _MODEL_KEYS, "model")
    model = CampaignModel(
        profile=_string(model_raw["profile"], "model.profile"),
        served_name=_string(model_raw["served_name"], "model.served_name"),
        context_tokens=_integer(model_raw["context_tokens"], "model.context_tokens"),
        max_output_tokens=_integer(
            model_raw["max_output_tokens"], "model.max_output_tokens"
        ),
        parallel=_integer(model_raw["parallel"], "model.parallel"),
    )
    if (
        model.profile != MODEL_PROFILE
        or model.served_name != MODEL_SERVED_NAME
        or model.context_tokens != MODEL_CONTEXT_TOKENS
        or model.max_output_tokens != MODEL_MAX_OUTPUT_TOKENS
        or model.parallel != 1
    ):
        raise HarborCampaignError("Model geometry violates the serial campaign protocol")

    harbor_raw = _mapping(raw["harbor"], "harbor")
    _expect_keys(harbor_raw, _HARBOR_KEYS, "harbor")
    harbor = HarborPin(
        source=_pinned_string(harbor_raw["source"], _REPOSITORY_PATTERN, "harbor.source"),
        revision=_pinned_string(
            harbor_raw["revision"], _COMMIT_PATTERN, "harbor.revision"
        ),
        version=_pinned_string(harbor_raw["version"], _SEMVER_PATTERN, "harbor.version"),
        environment=_string(harbor_raw["environment"], "harbor.environment"),
        force_build=_boolean(harbor_raw["force_build"], "harbor.force_build"),
        n_attempts=_integer(harbor_raw["n_attempts"], "harbor.n_attempts"),
        n_concurrent=_integer(harbor_raw["n_concurrent"], "harbor.n_concurrent"),
        max_retries=_integer(
            harbor_raw["max_retries"], "harbor.max_retries", minimum=0
        ),
        runtime_tree_sha256=_pinned_string(
            harbor_raw["runtime_tree_sha256"],
            _SHA256_PATTERN,
            "harbor.runtime_tree_sha256",
        ),
        runtime_tree_size_bytes=_integer(
            harbor_raw["runtime_tree_size_bytes"],
            "harbor.runtime_tree_size_bytes",
        ),
        runtime_tree_entries=_integer(
            harbor_raw["runtime_tree_entries"], "harbor.runtime_tree_entries"
        ),
        runtime_tree_files=_integer(
            harbor_raw["runtime_tree_files"], "harbor.runtime_tree_files"
        ),
        runtime_tree_links=_integer(
            harbor_raw["runtime_tree_links"],
            "harbor.runtime_tree_links",
            minimum=0,
        ),
        executable_path=_relative_runtime_path(
            harbor_raw["executable_path"], "harbor.executable_path"
        ),
        executable_size_bytes=_integer(
            harbor_raw["executable_size_bytes"], "harbor.executable_size_bytes"
        ),
        executable_sha256=_pinned_string(
            harbor_raw["executable_sha256"],
            _SHA256_PATTERN,
            "harbor.executable_sha256",
        ),
        agent_source_sha256=_pinned_string(
            harbor_raw["agent_source_sha256"],
            _SHA256_PATTERN,
            "harbor.agent_source_sha256",
        ),
        python_launcher_path=_relative_runtime_path(
            harbor_raw["python_launcher_path"], "harbor.python_launcher_path"
        ),
        python_launcher_target=_string(
            harbor_raw["python_launcher_target"], "harbor.python_launcher_target"
        ),
        python_version=_pinned_string(
            harbor_raw["python_version"], _SEMVER_PATTERN, "harbor.python_version"
        ),
        python_path=_relative_runtime_path(
            harbor_raw["python_path"], "harbor.python_path"
        ),
        python_size_bytes=_integer(
            harbor_raw["python_size_bytes"], "harbor.python_size_bytes"
        ),
        python_sha256=_pinned_string(
            harbor_raw["python_sha256"], _SHA256_PATTERN, "harbor.python_sha256"
        ),
    )
    if (
        harbor.source != HARBOR_SOURCE
        or harbor.revision != HARBOR_REVISION
        or harbor.version != HARBOR_VERSION
        or harbor.environment != "docker"
        or not harbor.force_build
        or harbor.n_attempts != 1
        or harbor.n_concurrent != 1
        or harbor.max_retries != 0
        or harbor.runtime_tree_sha256 != HARBOR_RUNTIME_TREE_SHA256
        or harbor.runtime_tree_size_bytes != HARBOR_RUNTIME_TREE_SIZE_BYTES
        or harbor.runtime_tree_entries != HARBOR_RUNTIME_TREE_ENTRIES
        or harbor.runtime_tree_files != HARBOR_RUNTIME_TREE_FILES
        or harbor.runtime_tree_links != HARBOR_RUNTIME_TREE_LINKS
        or harbor.executable_path != HARBOR_EXECUTABLE_PATH
        or harbor.executable_size_bytes != HARBOR_EXECUTABLE_SIZE_BYTES
        or harbor.executable_sha256 != HARBOR_EXECUTABLE_SHA256
        or harbor.agent_source_sha256 != HARBOR_AGENT_SOURCE_SHA256
        or harbor.python_launcher_path != HARBOR_PYTHON_LAUNCHER_PATH
        or harbor.python_launcher_target != HARBOR_PYTHON_LAUNCHER_TARGET
        or harbor.python_version != HARBOR_PYTHON_VERSION
        or harbor.python_path != HARBOR_PYTHON_PATH
        or harbor.python_size_bytes != HARBOR_PYTHON_SIZE_BYTES
        or harbor.python_sha256 != HARBOR_PYTHON_SHA256
    ):
        raise HarborCampaignError("Harbor runtime pin or serial controls changed")

    dataset_raw = _mapping(raw["dataset"], "dataset")
    _expect_keys(dataset_raw, _DATASET_KEYS, "dataset")
    task_values = dataset_raw["tasks"]
    if not isinstance(task_values, list):
        raise HarborCampaignError("dataset.tasks must be an array")
    tasks = tuple(_string(value, "dataset.tasks entry") for value in task_values)
    dataset = DatasetPin(
        source=_pinned_string(
            dataset_raw["source"], _REPOSITORY_PATTERN, "dataset.source"
        ),
        revision=_pinned_string(
            dataset_raw["revision"], _COMMIT_PATTERN, "dataset.revision"
        ),
        version=_string(dataset_raw["version"], "dataset.version"),
        tasks=tasks,
    )
    if (
        dataset.source != DATASET_SOURCE
        or dataset.revision != DATASET_REVISION
        or dataset.version != DATASET_VERSION
        or dataset.tasks != EXPECTED_TASKS
    ):
        raise HarborCampaignError("Terminal-Bench dataset pin or task subset changed")

    agents_raw = raw["agents"]
    if not isinstance(agents_raw, list) or len(agents_raw) != len(EXPECTED_AGENT_IDS):
        raise HarborCampaignError("agents must contain the two pinned harnesses")
    agents: list[AgentPin] = []
    for index, raw_agent_value in enumerate(agents_raw):
        raw_agent = _mapping(raw_agent_value, f"agents[{index}]")
        keys = frozenset(raw_agent)
        if keys not in {
            _AGENT_REQUIRED_KEYS,
            _AGENT_REQUIRED_KEYS | _AGENT_PLATFORM_KEYS,
        }:
            raise HarborCampaignError(f"agents[{index}] keys are invalid")
        agent_id = _string(raw_agent["id"], f"agents[{index}].id")
        platform_values = tuple(raw_agent.get(key) for key in _AGENT_PLATFORM_KEYS)
        if any(value is not None for value in platform_values) and not all(
            value is not None for value in platform_values
        ):
            raise HarborCampaignError("Agent platform artifact pins must be complete")
        agent = AgentPin(
            id=agent_id,
            version=_pinned_string(
                raw_agent["version"], _SEMVER_PATTERN, f"agents[{index}].version"
            ),
            npm_package=_string(
                raw_agent["npm_package"], f"agents[{index}].npm_package"
            ),
            npm_integrity=_integrity(
                raw_agent["npm_integrity"], f"agents[{index}].npm_integrity"
            ),
            npm_shasum=_pinned_string(
                raw_agent["npm_shasum"], _SHA1_PATTERN, f"agents[{index}].npm_shasum"
            ),
            source=_pinned_string(
                raw_agent["source"], _REPOSITORY_PATTERN, f"agents[{index}].source"
            ),
            revision=_pinned_string(
                raw_agent["revision"], _COMMIT_PATTERN, f"agents[{index}].revision"
            ),
            install_tree_sha256=_pinned_string(
                raw_agent["install_tree_sha256"],
                _SHA256_PATTERN,
                f"agents[{index}].install_tree_sha256",
            ),
            install_tree_size_bytes=_integer(
                raw_agent["install_tree_size_bytes"],
                f"agents[{index}].install_tree_size_bytes",
            ),
            platform_package=(
                _string(raw_agent["platform_package"], "agent.platform_package")
                if "platform_package" in raw_agent
                else None
            ),
            platform_integrity=(
                _integrity(raw_agent["platform_integrity"], "agent.platform_integrity")
                if "platform_integrity" in raw_agent
                else None
            ),
            platform_shasum=(
                _pinned_string(
                    raw_agent["platform_shasum"], _SHA1_PATTERN, "agent.platform_shasum"
                )
                if "platform_shasum" in raw_agent
                else None
            ),
        )
        if agent.npm_package != EXPECTED_NPM_PACKAGES.get(agent.id):
            raise HarborCampaignError("Agent id and npm package do not match")
        if agent.id != "opencode" and agent.platform_package is not None:
            raise HarborCampaignError("Only OpenCode may pin a platform package")
        if agent.platform_package is not None and agent.platform_package != "opencode-linux-arm64":
            raise HarborCampaignError("OpenCode platform package must be native ARM64")
        expected_agent = EXPECTED_AGENT_PINS.get(agent.id)
        if expected_agent is None or any(
            getattr(agent, key) != value for key, value in expected_agent.items()
        ):
            raise HarborCampaignError("Agent package or source pins changed")
        agents.append(agent)
    if tuple(agent.id for agent in agents) != EXPECTED_AGENT_IDS:
        raise HarborCampaignError("Agent order changed")

    relay_raw = _mapping(raw["relay"], "relay")
    _expect_keys(relay_raw, _RELAY_KEYS, "relay")
    relay = RelayPin(
        listen_host=_string(relay_raw["listen_host"], "relay.listen_host"),
        port=_integer(relay_raw["port"], "relay.port"),
        sentinel_host=_string(relay_raw["sentinel_host"], "relay.sentinel_host"),
        placeholder_api_key=_string(
            relay_raw["placeholder_api_key"], "relay.placeholder_api_key"
        ),
        uds_path=_container_path(relay_raw["uds_path"], "relay.uds_path"),
        internal_key_path=_container_path(
            relay_raw["internal_key_path"], "relay.internal_key_path"
        ),
        node_image=_string(relay_raw["node_image"], "relay.node_image"),
        relay_script_sha256=_pinned_string(
            relay_raw["relay_script_sha256"],
            _SHA256_PATTERN,
            "relay.relay_script_sha256",
        ),
        network_policy_sha256=_pinned_string(
            relay_raw["network_policy_sha256"],
            _SHA256_PATTERN,
            "relay.network_policy_sha256",
        ),
    )
    if relay != RelayPin(
        listen_host=RELAY_LISTEN_HOST,
        port=RELAY_PORT,
        sentinel_host=RELAY_SENTINEL_HOST,
        placeholder_api_key=RELAY_PLACEHOLDER_API_KEY,
        uds_path=RELAY_UDS_PATH,
        internal_key_path=RELAY_INTERNAL_KEY_PATH,
        node_image=RELAY_NODE_IMAGE,
        relay_script_sha256=RELAY_SCRIPT_SHA256,
        network_policy_sha256=NETWORK_POLICY_SHA256,
    ):
        raise HarborCampaignError("Relay pins changed")

    toolchain_raw = _mapping(raw["toolchain"], "toolchain")
    _expect_keys(toolchain_raw, _TOOLCHAIN_KEYS, "toolchain")
    toolchain = ToolchainPin(
        node_version=_string(toolchain_raw["node_version"], "toolchain.node_version"),
        npm_builder_version=_pinned_string(
            toolchain_raw["npm_builder_version"],
            _SEMVER_PATTERN,
            "toolchain.npm_builder_version",
        ),
        node_binary_sha256=_pinned_string(
            toolchain_raw["node_binary_sha256"],
            _SHA256_PATTERN,
            "toolchain.node_binary_sha256",
        ),
        node_tree_sha256=_pinned_string(
            toolchain_raw["node_tree_sha256"],
            _SHA256_PATTERN,
            "toolchain.node_tree_sha256",
        ),
        node_tree_size_bytes=_integer(
            toolchain_raw["node_tree_size_bytes"],
            "toolchain.node_tree_size_bytes",
        ),
        node_mount_path=_container_path(
            toolchain_raw["node_mount_path"], "toolchain.node_mount_path"
        ),
        agent_mount_path=_container_path(
            toolchain_raw["agent_mount_path"], "toolchain.agent_mount_path"
        ),
    )
    if toolchain != ToolchainPin(
        node_version=NODE_VERSION,
        npm_builder_version=NPM_BUILDER_VERSION,
        node_binary_sha256=NODE_BINARY_SHA256,
        node_tree_sha256=NODE_TREE_SHA256,
        node_tree_size_bytes=NODE_TREE_SIZE_BYTES,
        node_mount_path=NODE_MOUNT_PATH,
        agent_mount_path=AGENT_MOUNT_PATH,
    ):
        raise HarborCampaignError("Offline toolchain pins changed")

    execution_raw = _mapping(raw["execution"], "execution")
    _expect_keys(execution_raw, _EXECUTION_KEYS, "execution")
    trial_order_raw = execution_raw["trial_order"]
    if not isinstance(trial_order_raw, list):
        raise HarborCampaignError("execution.trial_order must be an array")
    execution = ExecutionSpec(
        canary_task=_string(execution_raw["canary_task"], "execution.canary_task"),
        agent_timeout_s=_integer(
            execution_raw["agent_timeout_s"], "execution.agent_timeout_s"
        ),
        trial_wall_timeout_s=_integer(
            execution_raw["trial_wall_timeout_s"],
            "execution.trial_wall_timeout_s",
        ),
        hard_campaign_cutoff_s=_integer(
            execution_raw["hard_campaign_cutoff_s"],
            "execution.hard_campaign_cutoff_s",
        ),
        reserve_for_audit_s=_integer(
            execution_raw["reserve_for_audit_s"], "execution.reserve_for_audit_s"
        ),
        trial_order=tuple(
            _string(value, "execution.trial_order entry") for value in trial_order_raw
        ),
        bridge_requires_bearer_auth=_boolean(
            execution_raw["bridge_requires_bearer_auth"],
            "execution.bridge_requires_bearer_auth",
        ),
        bridge_target_is_loopback=_boolean(
            execution_raw["bridge_target_is_loopback"],
            "execution.bridge_target_is_loopback",
        ),
        raw_trajectories_are_ignored=_boolean(
            execution_raw["raw_trajectories_are_ignored"],
            "execution.raw_trajectories_are_ignored",
        ),
        publish_scalar_evidence_only=_boolean(
            execution_raw["publish_scalar_evidence_only"],
            "execution.publish_scalar_evidence_only",
        ),
        harness_requests_are_not_rewritten=_boolean(
            execution_raw["harness_requests_are_not_rewritten"],
            "execution.harness_requests_are_not_rewritten",
        ),
        server_default_temperature=_number(
            execution_raw["server_default_temperature"],
            "execution.server_default_temperature",
        ),
        server_default_top_p=_number(
            execution_raw["server_default_top_p"], "execution.server_default_top_p"
        ),
        server_default_top_k=_integer(
            execution_raw["server_default_top_k"], "execution.server_default_top_k"
        ),
    )
    if execution.canary_task not in tasks:
        raise HarborCampaignError("Canary task is outside the task subset")
    if (
        execution.canary_task != EXPECTED_TASKS[0]
        or execution.agent_timeout_s != AGENT_TIMEOUT_S
        or execution.trial_wall_timeout_s != TRIAL_WALL_TIMEOUT_S
        or execution.hard_campaign_cutoff_s != HARD_CAMPAIGN_CUTOFF_S
        or execution.reserve_for_audit_s != RESERVE_FOR_AUDIT_S
        or execution.server_default_temperature != SERVER_DEFAULT_TEMPERATURE
        or execution.server_default_top_p != SERVER_DEFAULT_TOP_P
        or execution.server_default_top_k != SERVER_DEFAULT_TOP_K
    ):
        raise HarborCampaignError("Frozen execution timing or sampling pins changed")
    if execution.reserve_for_audit_s >= execution.hard_campaign_cutoff_s:
        raise HarborCampaignError("Audit reserve must be below the hard cutoff")
    if execution.trial_wall_timeout_s <= execution.agent_timeout_s:
        raise HarborCampaignError("Trial wall timeout must include setup and verification")
    if not all(getattr(execution, key) for key in _TRUE_EXECUTION_FLAGS):
        raise HarborCampaignError("Required execution safety controls must be enabled")

    if execution.trial_order != EXPECTED_TRIAL_ORDER:
        raise HarborCampaignError("trial_order changed from the frozen counterbalance")

    admission_raw = _mapping(raw["admission"], "admission")
    _expect_keys(admission_raw, _ADMISSION_KEYS, "admission")
    admission: list[tuple[str, bool]] = []
    for key in sorted(_ADMISSION_KEYS):
        value = _boolean(admission_raw[key], f"admission.{key}")
        if not value:
            raise HarborCampaignError("All campaign admission gates must be enabled")
        admission.append((key, value))

    return CampaignSpec(
        schema_version=schema_version,
        id=campaign_id,
        description=description,
        model=model,
        harbor=harbor,
        dataset=dataset,
        agents=tuple(agents),
        relay=relay,
        toolchain=toolchain,
        execution=execution,
        admission=tuple(admission),
    )


def iter_trials(campaign: CampaignSpec) -> tuple[TrialSpec, ...]:
    """Return the exact counterbalanced order frozen in the manifest."""

    trials: list[TrialSpec] = []
    for index, entry in enumerate(campaign.execution.trial_order, start=1):
        task_id, separator, agent_id = entry.partition(":")
        if separator != ":" or task_id not in campaign.dataset.tasks:
            raise HarborCampaignError("Invalid trial_order task-agent pair")
        campaign.agent(agent_id)
        trials.append(TrialSpec(index=index, task_id=task_id, agent_id=agent_id))
    return tuple(trials)


def canonical_bridge_base_url(value: str) -> str:
    """Require the fixed loopback-only in-container relay endpoint."""

    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.username is not None or parsed.password is not None:
        raise HarborCampaignError("Bridge base URL must be credential-free HTTP")
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        port = parsed.port
    except ValueError as error:
        raise HarborCampaignError("Bridge base URL has an invalid host or port") from error
    if (
        address.version != 4
        or address.compressed != RELAY_LISTEN_HOST
        or port != RELAY_PORT
        or parsed.path.rstrip("/") != "/v1"
        or parsed.query
        or parsed.fragment
    ):
        raise HarborCampaignError("Relay endpoint changed from its fixed loopback pin")
    return RELAY_BASE_URL


def _read_all(descriptor: int, limit: int) -> bytes:
    blocks: list[bytes] = []
    remaining = limit + 1
    while remaining > 0:
        block = os.read(descriptor, min(remaining, 64 * 1024))
        if not block:
            break
        blocks.append(block)
        remaining -= len(block)
    return b"".join(blocks)


def read_api_key(path: Path) -> str:
    """Read one owner-only, single-link regular file without following links."""

    try:
        before = os.lstat(path)
    except OSError as error:
        raise HarborCampaignError("Could not inspect API key file") from error
    if not stat.S_ISREG(before.st_mode):
        raise HarborCampaignError("API key file must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise HarborCampaignError("Could not open API key file safely") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
            or (metadata.st_dev, metadata.st_ino) != (before.st_dev, before.st_ino)
            or metadata.st_size <= 0
            or metadata.st_size > MAX_API_KEY_BYTES
        ):
            raise HarborCampaignError(
                "API key file must be owner-only, single-link, regular, and bounded"
            )
        data = _read_all(descriptor, MAX_API_KEY_BYTES)
    finally:
        os.close(descriptor)
    if not data or len(data) > MAX_API_KEY_BYTES:
        raise HarborCampaignError("API key file has an invalid size")
    try:
        key = data.decode("utf-8")
    except UnicodeError as error:
        raise HarborCampaignError("API key file is not valid UTF-8") from error
    if key.endswith("\n"):
        key = key[:-1]
    if (
        not key
        or key != key.strip()
        or any(ord(character) < 33 or ord(character) > 126 for character in key)
    ):
        raise HarborCampaignError("API key file must contain exactly one ASCII token")
    return key


def _declared_npm_artifacts(campaign: CampaignSpec) -> dict[str, tuple[str, str, str]]:
    declared: dict[str, tuple[str, str, str]] = {}
    for agent in campaign.agents:
        declared[agent.npm_package] = (
            agent.version,
            agent.npm_shasum,
            agent.npm_integrity,
        )
        if agent.platform_package is not None:
            assert agent.platform_shasum is not None
            assert agent.platform_integrity is not None
            declared[agent.platform_package] = (
                agent.version,
                agent.platform_shasum,
                agent.platform_integrity,
            )
    return declared


def _hash_npm_tarball(path: Path) -> tuple[int, str, str]:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise HarborCampaignError("Could not inspect npm admission artifact") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.geteuid()
        or before.st_size <= 0
        or before.st_size > MAX_NPM_TARBALL_BYTES
    ):
        raise HarborCampaignError("npm admission artifact is unsafe or oversized")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise HarborCampaignError("Could not open npm admission artifact safely") from error
    sha1 = hashlib.sha1()
    sha512 = hashlib.sha512()
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino) != (before.st_dev, before.st_ino)
            or metadata.st_size != before.st_size
        ):
            raise HarborCampaignError("npm admission artifact changed while opening")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            sha1.update(block)
            sha512.update(block)
    finally:
        os.close(descriptor)
    integrity = "sha512-" + base64.b64encode(sha512.digest()).decode("ascii")
    return metadata.st_size, sha1.hexdigest(), integrity


def verify_npm_artifact_admission(
    campaign: CampaignSpec,
    artifact_paths: Mapping[str, Path],
    *,
    repo_root: Path,
) -> NpmArtifactAdmission:
    """Verify exact npm wrapper/platform tarballs before any model is started."""

    declared = _declared_npm_artifacts(campaign)
    if set(artifact_paths) != set(declared):
        raise HarborCampaignError("npm admission artifact set is incomplete or unknown")
    records: list[NpmArtifactRecord] = []
    resolved_paths: set[Path] = set()
    for package in sorted(declared):
        candidate = artifact_paths[package]
        if not isinstance(candidate, Path):
            raise HarborCampaignError("npm admission artifact path must be a Path")
        resolved = _resolved_external(
            candidate,
            repo_root=repo_root,
            context="npm admission artifact",
            must_exist=True,
            require_directory=False,
        )
        if resolved in resolved_paths:
            raise HarborCampaignError("npm admission artifacts must be distinct files")
        resolved_paths.add(resolved)
        version, expected_shasum, expected_integrity = declared[package]
        size_bytes, shasum, integrity = _hash_npm_tarball(resolved)
        if shasum != expected_shasum or integrity != expected_integrity:
            raise HarborCampaignError("npm admission artifact digest does not match manifest")
        records.append(
            NpmArtifactRecord(
                package=package,
                version=version,
                size_bytes=size_bytes,
                shasum=shasum,
                integrity=integrity,
            )
        )
    payload = [
        {
            "package": record.package,
            "version": record.version,
            "size_bytes": record.size_bytes,
            "shasum": record.shasum,
            "integrity": record.integrity,
        }
        for record in records
    ]
    digest = _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    return NpmArtifactAdmission(digest=digest, artifacts=tuple(records))


def _validate_npm_admission(
    campaign: CampaignSpec, admission: NpmArtifactAdmission
) -> None:
    declared = _declared_npm_artifacts(campaign)
    if not _SHA256_PATTERN.fullmatch(admission.digest):
        raise HarborCampaignError("npm artifact admission digest is invalid")
    actual = {
        record.package: (record.version, record.shasum, record.integrity)
        for record in admission.artifacts
    }
    if actual != declared or len(actual) != len(admission.artifacts):
        raise HarborCampaignError("npm artifact admission does not match the campaign")
    if any(
        isinstance(record.size_bytes, bool)
        or not isinstance(record.size_bytes, int)
        or record.size_bytes <= 0
        or record.size_bytes > MAX_NPM_TARBALL_BYTES
        for record in admission.artifacts
    ):
        raise HarborCampaignError("npm artifact admission size is invalid")
    payload = [
        {
            "package": record.package,
            "version": record.version,
            "size_bytes": record.size_bytes,
            "shasum": record.shasum,
            "integrity": record.integrity,
        }
        for record in admission.artifacts
    ]
    expected_digest = _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    if expected_digest != admission.digest:
        raise HarborCampaignError("npm artifact admission record digest changed")


def _resolved_external(
    path: Path,
    *,
    repo_root: Path,
    context: str,
    must_exist: bool,
    require_directory: bool = True,
) -> Path:
    try:
        repository = repo_root.resolve(strict=True)
        resolved = path.resolve(strict=must_exist)
    except OSError as error:
        raise HarborCampaignError(f"Could not resolve {context}") from error
    if resolved == repository or repository in resolved.parents:
        raise HarborCampaignError(f"{context} must live outside the repository")
    if must_exist:
        try:
            metadata = os.lstat(path)
        except OSError as error:
            raise HarborCampaignError(f"Could not inspect {context}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise HarborCampaignError(f"{context} must not be a symbolic link")
        if require_directory and not stat.S_ISDIR(metadata.st_mode):
            raise HarborCampaignError(f"{context} must be a directory")
    return resolved


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_owned_regular_file(path: Path, *, context: str) -> tuple[bytes, int]:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise HarborCampaignError(f"Could not inspect {context}") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.geteuid()
        or before.st_size < 0
        or before.st_size > MAX_TASK_FILE_BYTES
    ):
        raise HarborCampaignError(f"{context} is linked, special, unowned, or oversized")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise HarborCampaignError(f"Could not open {context} safely") from error
    try:
        opened = os.fstat(descriptor)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_uid,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        expected_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if identity != expected_identity:
            raise HarborCampaignError(f"{context} changed while opening")
        data = bytearray()
        while len(data) <= MAX_TASK_FILE_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, MAX_TASK_FILE_BYTES + 1 - len(data)),
            )
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(descriptor)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_uid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if after_identity != identity or len(data) != opened.st_size:
            raise HarborCampaignError(f"{context} changed while reading")
    except OSError as error:
        raise HarborCampaignError(f"Could not read {context} safely") from error
    finally:
        os.close(descriptor)
    return bytes(data), stat.S_IMODE(opened.st_mode)


def _validate_task_directory(path: Path) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise HarborCampaignError("Could not inspect task tree directory") from error
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise HarborCampaignError("Task tree contains a linked or unowned directory")


def _file_tree(path: Path) -> dict[str, str]:
    _validate_task_directory(path)
    files: dict[str, str] = {}
    for root, directory_names, file_names in os.walk(path, followlinks=False):
        root_path = Path(root)
        _validate_task_directory(root_path)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            _validate_task_directory(root_path / name)
        for name in file_names:
            candidate = root_path / name
            relative = candidate.relative_to(path).as_posix()
            data, mode = _read_owned_regular_file(candidate, context="task tree file")
            if mode not in {0o555, 0o644, 0o755}:
                raise HarborCampaignError("Task tree file mode is not Git-normalized")
            files[relative] = f"{mode:04o}:{_sha256_bytes(data)}"
    return dict(sorted(files.items()))


def _write_new_regular_file(path: Path, data: bytes, mode: int) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, mode)
        try:
            view = memoryview(data)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise OSError("short write")
                written += count
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise HarborCampaignError("Could not safely write generated task file") from error


def _copy_task_tree(
    source: Path, destination: Path, tracked_files: Mapping[str, str]
) -> None:
    _validate_task_directory(source)
    destination.mkdir(mode=0o700)
    created_directories = {PurePosixPath(".")}
    for relative, record in sorted(tracked_files.items()):
        expected_mode, expected_digest = _file_record_parts(record)
        relative_path = PurePosixPath(relative)
        for parent in reversed(relative_path.parents[:-1]):
            if parent not in created_directories:
                (destination / Path(parent.as_posix())).mkdir(mode=0o700)
                created_directories.add(parent)
        data, actual_mode = _read_owned_regular_file(
            source / Path(relative), context="source task file"
        )
        if (
            _sha256_bytes(data) != expected_digest
            or bool(actual_mode & 0o111) != (expected_mode == 0o755)
        ):
            raise HarborCampaignError("Tracked task file changed during copy")
        _write_new_regular_file(
            destination / Path(relative), data, expected_mode
        )


def _harbor_runtime_digest(campaign: CampaignSpec, tree: TreeAdmission) -> str:
    payload = {
        "protocol": "harbor-runtime-admission-v1",
        "tree": _tree_admission_projection(tree),
        "executable_path": campaign.harbor.executable_path,
        "executable_size_bytes": campaign.harbor.executable_size_bytes,
        "executable_sha256": campaign.harbor.executable_sha256,
        "python_launcher_path": campaign.harbor.python_launcher_path,
        "python_launcher_target": campaign.harbor.python_launcher_target,
        "python_version": campaign.harbor.python_version,
        "python_path": campaign.harbor.python_path,
        "python_size_bytes": campaign.harbor.python_size_bytes,
        "python_sha256": campaign.harbor.python_sha256,
    }
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    )


def _verify_runtime_member(
    path: Path, *, expected_size: int, expected_sha256: str, context: str
) -> None:
    data, mode = _read_owned_regular_file(path, context=context)
    if (
        mode != 0o555
        or len(data) != expected_size
        or _sha256_bytes(data) != expected_sha256
    ):
        raise HarborCampaignError(f"{context} mode, size, or digest changed")


def _verify_python_launcher(
    launcher: Path, python: Path, *, expected_target: str
) -> None:
    try:
        before = os.lstat(launcher)
        target = os.readlink(launcher)
        after = os.lstat(launcher)
    except OSError as error:
        raise HarborCampaignError("Harbor Python launcher admission failed") from error
    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
    if (
        not stat.S_ISLNK(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or identity(before) != identity(after)
        or target != expected_target
    ):
        raise HarborCampaignError("Harbor Python launcher symlink changed")
    try:
        resolved_target = (launcher.parent / target).resolve(strict=True)
        resolved_python = python.resolve(strict=True)
    except OSError as error:
        raise HarborCampaignError("Harbor Python launcher target is unavailable") from error
    if resolved_target != resolved_python:
        raise HarborCampaignError("Harbor Python launcher escaped its admitted runtime")


def verify_harbor_runtime(
    campaign: CampaignSpec, path: Path, *, repo_root: Path
) -> HarborRuntimeAdmission:
    """Admit the complete immutable pinned Harbor/Python runtime tree."""

    try:
        tree = verify_normalized_tree(
            path,
            repo_root=repo_root,
            expected_digest=campaign.harbor.runtime_tree_sha256,
            expected_size_bytes=campaign.harbor.runtime_tree_size_bytes,
            expected_entries=campaign.harbor.runtime_tree_entries,
            expected_files=campaign.harbor.runtime_tree_files,
            expected_links=campaign.harbor.runtime_tree_links,
        )
    except RuntimeAssetError as error:
        raise HarborCampaignError("Pinned Harbor runtime tree admission failed") from error
    executable = tree.resolved_path / campaign.harbor.executable_path
    launcher = tree.resolved_path / campaign.harbor.python_launcher_path
    python = tree.resolved_path / campaign.harbor.python_path
    _verify_runtime_member(
        executable,
        expected_size=campaign.harbor.executable_size_bytes,
        expected_sha256=campaign.harbor.executable_sha256,
        context="Harbor runtime executable",
    )
    _verify_runtime_member(
        python,
        expected_size=campaign.harbor.python_size_bytes,
        expected_sha256=campaign.harbor.python_sha256,
        context="Harbor Python executable",
    )
    _verify_python_launcher(
        launcher,
        python,
        expected_target=campaign.harbor.python_launcher_target,
    )
    return HarborRuntimeAdmission(
        digest=_harbor_runtime_digest(campaign, tree),
        tree=tree,
        executable_path=executable,
        python_launcher_path=launcher,
        python_path=python,
    )


def _validate_harbor_runtime_admission(
    campaign: CampaignSpec, admission: HarborRuntimeAdmission
) -> None:
    if (
        "sha256:" + admission.tree.digest != campaign.harbor.runtime_tree_sha256
        or admission.tree.size_bytes != campaign.harbor.runtime_tree_size_bytes
        or admission.tree.entries != campaign.harbor.runtime_tree_entries
        or admission.tree.files != campaign.harbor.runtime_tree_files
        or admission.tree.links != campaign.harbor.runtime_tree_links
        or admission.executable_path
        != admission.tree.resolved_path / campaign.harbor.executable_path
        or admission.python_launcher_path
        != admission.tree.resolved_path / campaign.harbor.python_launcher_path
        or admission.python_path
        != admission.tree.resolved_path / campaign.harbor.python_path
        or admission.digest != _harbor_runtime_digest(campaign, admission.tree)
    ):
        raise HarborCampaignError("Pinned Harbor runtime admission changed")
    _verify_runtime_member(
        admission.executable_path,
        expected_size=campaign.harbor.executable_size_bytes,
        expected_sha256=campaign.harbor.executable_sha256,
        context="Harbor runtime executable",
    )
    _verify_runtime_member(
        admission.python_path,
        expected_size=campaign.harbor.python_size_bytes,
        expected_sha256=campaign.harbor.python_sha256,
        context="Harbor Python executable",
    )
    _verify_python_launcher(
        admission.python_launcher_path,
        admission.python_path,
        expected_target=campaign.harbor.python_launcher_target,
    )


def _admit_runtime_asset(
    path: Path,
    *,
    expected_sha256: str,
    expected_mode: int,
) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise HarborCampaignError("Runtime overlay asset does not resolve") from error
    if absolute != resolved:
        raise HarborCampaignError("Runtime overlay asset path must be canonical")
    data, mode = _read_owned_regular_file(resolved, context="runtime overlay asset")
    if mode != expected_mode or _sha256_bytes(data) != expected_sha256:
        raise HarborCampaignError("Runtime overlay asset mode or digest changed")
    return resolved


def _tree_admission_projection(admission: TreeAdmission) -> dict[str, Any]:
    return {
        "protocol": admission.protocol,
        "digest": "sha256:" + admission.digest,
        "entries": admission.entries,
        "files": admission.files,
        "links": admission.links,
        "size_bytes": admission.size_bytes,
    }


def _runtime_admission_digest(
    campaign: CampaignSpec,
    trial: TrialSpec,
    node_tree: TreeAdmission,
    agent_tree: TreeAdmission,
) -> str:
    payload = {
        "protocol": "harbor-runtime-overlay-v1",
        "agent": trial.agent_id,
        "node_tree": _tree_admission_projection(node_tree),
        "agent_tree": _tree_admission_projection(agent_tree),
        "node_binary_sha256": campaign.toolchain.node_binary_sha256,
        "relay_script_sha256": campaign.relay.relay_script_sha256,
        "network_policy_sha256": campaign.relay.network_policy_sha256,
        "relay_image": campaign.relay.node_image,
        "relay_port": campaign.relay.port,
    }
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    )


def prepare_runtime_overlay(
    campaign: CampaignSpec,
    trial: TrialSpec,
    *,
    destination: Path,
    node_prefix: Path,
    agent_prefix: Path,
    relay_script: Path,
    network_policy_script: Path,
    run_socket_dir: Path,
    repo_root: Path,
) -> RuntimeOverlayAdmission:
    """Verify immutable runtime assets and write one exact external Compose overlay."""

    expected_trials = iter_trials(campaign)
    if trial not in expected_trials or expected_trials[trial.index - 1] != trial:
        raise HarborCampaignError("Runtime overlay trial changed from trial_order")
    agent = campaign.agent(trial.agent_id)
    try:
        node_tree = verify_normalized_tree(
            node_prefix,
            repo_root=repo_root,
            expected_digest=campaign.toolchain.node_tree_sha256,
            expected_size_bytes=campaign.toolchain.node_tree_size_bytes,
        )
        agent_tree = verify_normalized_tree(
            agent_prefix,
            repo_root=repo_root,
            expected_digest=agent.install_tree_sha256,
            expected_size_bytes=agent.install_tree_size_bytes,
        )
    except RuntimeAssetError as error:
        raise HarborCampaignError("Runtime tool tree admission failed") from error

    node_binary, node_mode = _read_owned_regular_file(
        node_tree.resolved_path / "bin" / "node", context="admitted Node binary"
    )
    if (
        node_mode != 0o555
        or _sha256_bytes(node_binary) != campaign.toolchain.node_binary_sha256
    ):
        raise HarborCampaignError("Admitted Node executable changed")
    relay_asset = _admit_runtime_asset(
        relay_script,
        expected_sha256=campaign.relay.relay_script_sha256,
        expected_mode=0o444,
    )
    policy_asset = _admit_runtime_asset(
        network_policy_script,
        expected_sha256=campaign.relay.network_policy_sha256,
        expected_mode=0o555,
    )

    socket_root = _resolved_external(
        run_socket_dir,
        repo_root=repo_root,
        context="relay socket directory",
        must_exist=True,
    )
    socket_root_metadata = os.lstat(socket_root)
    expected_names = {
        PurePosixPath(campaign.relay.uds_path).name,
        PurePosixPath(campaign.relay.internal_key_path).name,
    }
    if (
        socket_root_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(socket_root_metadata.st_mode) != 0o700
        or {entry.name for entry in socket_root.iterdir()} != expected_names
    ):
        raise HarborCampaignError("Relay socket directory is not an isolated 0700 pair")
    key_path = socket_root / PurePosixPath(campaign.relay.internal_key_path).name
    read_api_key(key_path)
    socket_path = socket_root / PurePosixPath(campaign.relay.uds_path).name
    socket_metadata = os.lstat(socket_path)
    if (
        not stat.S_ISSOCK(socket_metadata.st_mode)
        or socket_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(socket_metadata.st_mode) != 0o600
    ):
        raise HarborCampaignError("Relay model socket is not one owned mode-0600 socket")

    destination_path = _resolved_external(
        destination,
        repo_root=repo_root,
        context="runtime Compose overlay",
        must_exist=False,
        require_directory=False,
    )
    if destination.exists():
        raise HarborCampaignError("Runtime Compose overlay must not already exist")
    try:
        destination_path.parent.resolve(strict=True)
    except OSError as error:
        raise HarborCampaignError("Runtime overlay parent must already exist") from error

    def bind(source: Path, target: str) -> dict[str, Any]:
        return {
            "type": "bind",
            "source": str(source),
            "target": target,
            "read_only": True,
            "bind": {"create_host_path": False},
        }

    relay_service = "sparkbench-model-relay"
    sidecar_service = "harbor-docker-egress-control-sidecar"
    health_probe = (
        "const f=require('fs'),p=f.readFileSync('/proc/1/cmdline','utf8'),"
        "r=f.readFileSync('/proc/net/tcp','ascii').trim().split('\\n').slice(1);"
        "const l=r.some(x=>{const c=x.trim().split(/\\s+/);"
        "return c[1]==='0100007F:46A0'&&c[3]==='0A'});"
        "process.exit(p.includes('/usr/local/bin/node\\0/opt/sparkbench/relay.js')"
        "&&l?0:1)"
    )
    document = {
        "services": {
            "main": {
                "cap_drop": ["ALL"],
                "security_opt": ["no-new-privileges:true"],
                "depends_on": {relay_service: {"condition": "service_healthy"}},
                "volumes": [
                    bind(node_tree.resolved_path, campaign.toolchain.node_mount_path),
                    bind(agent_tree.resolved_path, campaign.toolchain.agent_mount_path),
                ],
            },
            relay_service: {
                "image": campaign.relay.node_image,
                "platform": "linux/arm64",
                "user": f"{os.geteuid()}:{os.getegid()}",
                "network_mode": f"service:{sidecar_service}",
                "depends_on": {sidecar_service: {"condition": "service_healthy"}},
                "command": ["/usr/local/bin/node", "/opt/sparkbench/relay.js"],
                "cap_drop": ["ALL"],
                "security_opt": ["no-new-privileges:true"],
                "read_only": True,
                "restart": "no",
                "stop_grace_period": "5s",
                "volumes": [
                    bind(relay_asset, "/opt/sparkbench/relay.js"),
                    bind(socket_root, "/run/sparkbench"),
                ],
                "healthcheck": {
                    "test": ["CMD", "/usr/local/bin/node", "-e", health_probe],
                    "interval": "5s",
                    "timeout": "2s",
                    "retries": 6,
                    "start_period": "1s",
                },
            },
            sidecar_service: {
                "volumes": [bind(policy_asset, "/usr/local/bin/network-policy")]
            },
        }
    }
    encoded = (
        json.dumps(document, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    _write_new_regular_file(destination_path, encoded, 0o600)
    return RuntimeOverlayAdmission(
        trial=trial,
        digest=_runtime_admission_digest(campaign, trial, node_tree, agent_tree),
        compose_sha256=_sha256_bytes(encoded),
        node_tree=node_tree,
        agent_tree=agent_tree,
        compose_path=destination_path,
    )


def _validate_runtime_overlay_admission(
    campaign: CampaignSpec,
    trial: TrialSpec,
    admission: RuntimeOverlayAdmission,
    *,
    repo_root: Path,
) -> None:
    if admission.trial != trial:
        raise HarborCampaignError("Runtime overlay admission belongs to another trial")
    agent = campaign.agent(trial.agent_id)
    if (
        "sha256:" + admission.node_tree.digest != campaign.toolchain.node_tree_sha256
        or admission.node_tree.size_bytes != campaign.toolchain.node_tree_size_bytes
        or "sha256:" + admission.agent_tree.digest != agent.install_tree_sha256
        or admission.agent_tree.size_bytes != agent.install_tree_size_bytes
        or admission.digest
        != _runtime_admission_digest(
            campaign, trial, admission.node_tree, admission.agent_tree
        )
    ):
        raise HarborCampaignError("Runtime overlay tree admission changed")
    overlay = _resolved_external(
        admission.compose_path,
        repo_root=repo_root,
        context="runtime Compose overlay",
        must_exist=True,
        require_directory=False,
    )
    encoded, mode = _read_owned_regular_file(
        overlay, context="runtime Compose overlay"
    )
    if mode != 0o600 or _sha256_bytes(encoded) != admission.compose_sha256:
        raise HarborCampaignError("Runtime Compose overlay changed after admission")
    try:
        parsed = json.loads(encoded.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise HarborCampaignError("Runtime Compose overlay is no longer JSON") from error
    if not isinstance(parsed, dict) or set(parsed) != {"services"}:
        raise HarborCampaignError("Runtime Compose overlay topology changed")


def _tree_digest(files: Mapping[str, str]) -> str:
    encoded = json.dumps(
        sorted(files.items()), ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return _sha256_bytes(encoded)


def _file_record_parts(record: str) -> tuple[int, str]:
    mode_text, separator, digest = record.partition(":")
    if (
        separator != ":"
        or mode_text not in {"0555", "0644", "0755"}
        or _SHA256_PATTERN.fullmatch(digest) is None
    ):
        raise HarborCampaignError("Task file provenance record is invalid")
    return int(mode_text, 8), digest


def _verify_clean_checkout(checkout: Path, revision: str, scopes: Sequence[str]) -> None:
    commands = [
        ["git", "-C", str(checkout), "rev-parse", "--verify", "HEAD"],
        [
            "git",
            "-C",
            str(checkout),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignored=matching",
            "--",
            *scopes,
        ],
    ]
    try:
        head = subprocess.run(
            commands[0], check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=15,
        )
        status_result = subprocess.run(
            commands[1], check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise HarborCampaignError("Could not verify pinned source checkout") from error
    if (
        head.returncode != 0
        or head.stdout.strip() != revision
        or status_result.returncode != 0
        or status_result.stdout
    ):
        raise HarborCampaignError("Pinned source checkout revision or cleanliness changed")


def _git_bound_task_trees(
    checkout: Path, task_ids: Sequence[str]
) -> dict[str, dict[str, str]]:
    scopes = [f"tasks/{task_id}" for task_id in task_ids]
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), "ls-files", "--stage", "-z", "--", *scopes],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise HarborCampaignError("Could not enumerate tracked task files") from error
    if result.returncode != 0 or not isinstance(result.stdout, bytes):
        raise HarborCampaignError("Could not enumerate tracked task files")
    index_entries: dict[str, dict[str, tuple[int, str]]] = {
        task_id: {} for task_id in task_ids
    }
    for raw_record in result.stdout.split(b"\0"):
        if not raw_record:
            continue
        metadata, separator, raw_path = raw_record.partition(b"\t")
        fields = metadata.split(b" ")
        if separator != b"\t" or len(fields) != 3 or fields[2] != b"0":
            raise HarborCampaignError("Tracked task index record is invalid")
        try:
            mode_text = fields[0].decode("ascii")
            object_id = fields[1].decode("ascii")
            relative_path = raw_path.decode("utf-8")
        except UnicodeError as error:
            raise HarborCampaignError("Tracked task path or metadata is not UTF-8") from error
        if (
            mode_text not in {"100644", "100755"}
            or re.fullmatch(r"[0-9a-f]{40}", object_id) is None
            or any(ord(character) < 32 or ord(character) > 126 for character in relative_path)
            or PurePosixPath(relative_path).as_posix() != relative_path
        ):
            raise HarborCampaignError("Tracked task entry is unsafe")
        task_id = next(
            (
                candidate
                for candidate in task_ids
                if relative_path.startswith(f"tasks/{candidate}/")
            ),
            None,
        )
        if task_id is None:
            raise HarborCampaignError("Tracked task entry escaped its selected scope")
        task_relative = relative_path.removeprefix(f"tasks/{task_id}/")
        expected_mode = 0o755 if mode_text == "100755" else 0o644
        if task_relative in index_entries[task_id]:
            raise HarborCampaignError("Tracked task topology contains a duplicate")
        index_entries[task_id][task_relative] = (expected_mode, object_id)
    if any(not tree for tree in index_entries.values()):
        raise HarborCampaignError("Tracked task topology is incomplete")

    trees: dict[str, dict[str, str]] = {}
    for task_id, expected_entries in index_entries.items():
        task_root = checkout / "tasks" / task_id
        expected_directories = {"."}
        for relative in expected_entries:
            parent = PurePosixPath(relative).parent
            while parent.as_posix() != ".":
                expected_directories.add(parent.as_posix())
                parent = parent.parent
        actual_directories = {"."}
        actual_files: dict[str, str] = {}
        _validate_task_directory(task_root)
        for root, directory_names, file_names in os.walk(
            task_root, followlinks=False
        ):
            root_path = Path(root)
            _validate_task_directory(root_path)
            relative_root = root_path.relative_to(task_root).as_posix()
            actual_directories.add(relative_root)
            directory_names.sort()
            file_names.sort()
            for name in directory_names:
                candidate = root_path / name
                _validate_task_directory(candidate)
                actual_directories.add(candidate.relative_to(task_root).as_posix())
            for name in file_names:
                candidate = root_path / name
                relative = candidate.relative_to(task_root).as_posix()
                expected = expected_entries.get(relative)
                if expected is None:
                    raise HarborCampaignError(
                        "Selected task contains an ignored or untracked entry"
                    )
                expected_mode, object_id = expected
                data, actual_mode = _read_owned_regular_file(
                    candidate, context="tracked task file"
                )
                git_blob = hashlib.sha1(
                    b"blob " + str(len(data)).encode("ascii") + b"\0" + data
                ).hexdigest()
                if (
                    git_blob != object_id
                    or bool(actual_mode & 0o111) != (expected_mode == 0o755)
                ):
                    raise HarborCampaignError("Tracked task blob or mode changed")
                actual_files[relative] = (
                    f"{expected_mode:04o}:{_sha256_bytes(data)}"
                )
        if actual_directories != expected_directories:
            raise HarborCampaignError(
                "Selected task contains an ignored or untracked directory"
            )
        if set(actual_files) != set(expected_entries):
            raise HarborCampaignError("Tracked task topology is incomplete")
        trees[task_id] = dict(sorted(actual_files.items()))
    return trees


def _patch_task_toml(source: bytes, timeout_s: int) -> bytes:
    try:
        text = source.decode("utf-8")
        parsed = tomllib.loads(text)
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise HarborCampaignError("Pinned task.toml is malformed") from error
    agent = _mapping(parsed.get("agent"), "task.agent")
    verifier = _mapping(parsed.get("verifier"), "task.verifier")
    environment = _mapping(parsed.get("environment"), "task.environment")
    timeout = agent.get("timeout_sec")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout != timeout_s:
        raise HarborCampaignError("Task agent timeout changed from the campaign pin")
    if "network_mode" in agent or "allowed_hosts" in agent:
        raise HarborCampaignError("Source task already has an explicit agent network policy")
    if "network_mode" in verifier or "allowed_hosts" in verifier:
        raise HarborCampaignError(
            "Source task already has an explicit verifier network policy"
        )
    if environment.get("allow_internet") is not True:
        raise HarborCampaignError("Task environment baseline is no longer public")
    if "network_mode" in environment or "allowed_hosts" in environment:
        raise HarborCampaignError("Task environment baseline network schema changed")
    insertions = {
        "[environment]\n": '[environment]\nnetwork_mode = "no-network"\n',
        "[agent]\n": (
            '[agent]\nnetwork_mode = "allowlist"\n'
            f'allowed_hosts = ["{RELAY_SENTINEL_HOST}"]\n'
        ),
        "[verifier]\n": '[verifier]\nnetwork_mode = "no-network"\n',
    }
    patched_text = text
    for marker, insertion in insertions.items():
        if patched_text.count(marker) != 1:
            raise HarborCampaignError("Task must contain each canonical policy table")
        patched_text = patched_text.replace(marker, insertion, 1)
    patched = patched_text.encode("utf-8")
    derived = tomllib.loads(patched.decode("utf-8"))
    if derived["agent"].get("network_mode") != "allowlist" or derived["agent"].get(
        "allowed_hosts"
    ) != [RELAY_SENTINEL_HOST]:
        raise HarborCampaignError("Could not derive the private agent network policy")
    if (
        derived["environment"].get("network_mode") != "no-network"
        or derived["verifier"].get("network_mode") != "no-network"
    ):
        raise HarborCampaignError("Could not derive baseline/verifier deny-all policy")
    return patched


def derive_private_task_dataset(
    campaign: CampaignSpec,
    *,
    source_checkout: Path,
    destination: Path,
    repo_root: Path,
    checkout_verifier: Callable[[Path, str, Sequence[str]], None] = _verify_clean_checkout,
    tracked_tree_verifier: Callable[
        [Path, Sequence[str]], dict[str, dict[str, str]]
    ] = _git_bound_task_trees,
) -> NetworkPolicyPatch:
    """Copy selected tasks and apply deterministic phase-isolated networking."""
    source_root = _resolved_external(
        source_checkout,
        repo_root=repo_root,
        context="Terminal-Bench checkout",
        must_exist=True,
    )
    destination_root = _resolved_external(
        destination,
        repo_root=repo_root,
        context="derived task directory",
        must_exist=False,
    )
    if destination.exists():
        raise HarborCampaignError("Derived task directory must not already exist")
    try:
        destination_root.parent.resolve(strict=True)
    except OSError as error:
        raise HarborCampaignError("Derived task parent directory must already exist") from error

    scoped = tuple(f"tasks/{task}" for task in campaign.dataset.tasks)
    checkout_verifier(source_root, campaign.dataset.revision, scoped)
    tracked_trees = tracked_tree_verifier(source_root, campaign.dataset.tasks)
    if set(tracked_trees) != set(campaign.dataset.tasks):
        raise HarborCampaignError("Tracked task set changed")
    tasks_root = source_root / "tasks"
    source_before: dict[str, dict[str, str]] = {}
    for task_id in campaign.dataset.tasks:
        task_root = tasks_root / task_id
        if not task_root.is_dir():
            raise HarborCampaignError("Pinned dataset is missing an expected task")
        source_before[task_id] = tracked_trees[task_id]
        if "task.toml" not in source_before[task_id]:
            raise HarborCampaignError("Pinned task is missing task.toml")

    destination_root.mkdir(mode=0o700)
    try:
        task_patches: list[TaskPatch] = []
        for task_id in campaign.dataset.tasks:
            source_task = tasks_root / task_id
            derived_task = destination_root / task_id
            _copy_task_tree(
                source_task, derived_task, source_before[task_id]
            )
            source_toml, source_actual_mode = _read_owned_regular_file(
                source_task / "task.toml", context="source task.toml"
            )
            source_mode, source_digest = _file_record_parts(
                source_before[task_id]["task.toml"]
            )
            if (
                _sha256_bytes(source_toml) != source_digest
                or bool(source_actual_mode & 0o111) != (source_mode == 0o755)
            ):
                raise HarborCampaignError("Tracked source task.toml changed")
            patched_toml = _patch_task_toml(
                source_toml, campaign.execution.agent_timeout_s
            )
            derived_toml = derived_task / "task.toml"
            os.unlink(derived_toml)
            _write_new_regular_file(derived_toml, patched_toml, source_mode)

            verifier_relative = "tests/test.sh"
            verifier_record = source_before[task_id].get(verifier_relative)
            if verifier_record is None:
                raise HarborCampaignError("Pinned task is missing tests/test.sh")
            verifier_source_mode, verifier_digest = _file_record_parts(
                verifier_record
            )
            if verifier_source_mode != 0o644:
                raise HarborCampaignError(
                    "Pinned verifier launcher mode changed from the dataset pin"
                )
            os.chmod(derived_task / verifier_relative, 0o555)

            source_after = tracked_tree_verifier(source_root, (task_id,))[task_id]
            if source_after != source_before[task_id]:
                raise HarborCampaignError("Source task tree changed during derivation")
            derived_files = _file_tree(derived_task)
            source_unchanged = {
                key: value
                for key, value in source_before[task_id].items()
                if key not in {"task.toml", verifier_relative}
            }
            derived_unchanged = {
                key: value
                for key, value in derived_files.items()
                if key not in {"task.toml", verifier_relative}
            }
            derived_verifier_mode, derived_verifier_digest = _file_record_parts(
                derived_files[verifier_relative]
            )
            if (
                source_unchanged != derived_unchanged
                or derived_verifier_digest != verifier_digest
                or derived_verifier_mode != 0o555
            ):
                raise HarborCampaignError(
                    "Derived task changed content or an unapproved file mode"
                )
            task_patches.append(
                TaskPatch(
                    task_id=task_id,
                    source_task_toml_sha256=source_digest,
                    source_task_toml_mode=source_mode,
                    derived_task_toml_sha256=_file_record_parts(
                        derived_files["task.toml"]
                    )[1],
                    derived_task_toml_mode=_file_record_parts(
                        derived_files["task.toml"]
                    )[0],
                    unchanged_tree_sha256=_tree_digest(source_unchanged),
                    verifier_script_sha256=verifier_digest,
                    source_verifier_script_mode=verifier_source_mode,
                    derived_verifier_script_mode=0o555,
                )
            )

        patch_payload = {
            "schema_version": 1,
            "dataset_revision": campaign.dataset.revision,
            "environment_network_mode": "no-network",
            "agent_network_mode": "allowlist",
            "agent_allowed_hosts": [RELAY_SENTINEL_HOST],
            "verifier_network_mode": "no-network",
            "tasks": [
                {
                    "task_id": item.task_id,
                    "source_task_toml_sha256": item.source_task_toml_sha256,
                    "source_task_toml_mode": item.source_task_toml_mode,
                    "derived_task_toml_sha256": item.derived_task_toml_sha256,
                    "derived_task_toml_mode": item.derived_task_toml_mode,
                    "unchanged_tree_sha256": item.unchanged_tree_sha256,
                    "verifier_script_sha256": item.verifier_script_sha256,
                    "source_verifier_script_mode": (
                        item.source_verifier_script_mode
                    ),
                    "derived_verifier_script_mode": (
                        item.derived_verifier_script_mode
                    ),
                }
                for item in task_patches
            ],
        }
        digest = _sha256_bytes(
            json.dumps(
                patch_payload,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
        return NetworkPolicyPatch(
            dataset_revision=campaign.dataset.revision,
            digest=digest,
            tasks=tuple(task_patches),
            dataset_dir=destination_root,
        )
    except BaseException:
        shutil.rmtree(destination_root)
        raise


def verify_private_task_dataset(
    campaign: CampaignSpec, patch: NetworkPolicyPatch, *, repo_root: Path
) -> None:
    """Fail closed if the external derived tree changed after preparation."""

    if (
        patch.dataset_revision != campaign.dataset.revision
        or not _SHA256_PATTERN.fullmatch(patch.digest)
        or tuple(item.task_id for item in patch.tasks) != campaign.dataset.tasks
    ):
        raise HarborCampaignError("Derived network patch provenance changed")
    root = _resolved_external(
        patch.dataset_dir,
        repo_root=repo_root,
        context="derived task directory",
        must_exist=True,
    )
    if {path.name for path in root.iterdir()} != set(campaign.dataset.tasks):
        raise HarborCampaignError("Derived task directory topology changed")
    for expected in patch.tasks:
        files = _file_tree(root / expected.task_id)
        record = files.get("task.toml")
        if record is None:
            raise HarborCampaignError("Derived task.toml is missing")
        derived_mode, derived_digest = _file_record_parts(record)
        if (
            derived_digest != expected.derived_task_toml_sha256
            or derived_mode != expected.derived_task_toml_mode
            or expected.source_task_toml_mode not in {0o644, 0o755}
            or expected.derived_task_toml_mode != expected.source_task_toml_mode
            or not _SHA256_PATTERN.fullmatch(expected.source_task_toml_sha256)
        ):
            raise HarborCampaignError("Derived task.toml digest changed")
        unchanged = {
            key: value
            for key, value in files.items()
            if key not in {"task.toml", "tests/test.sh"}
        }
        verifier_record = files.get("tests/test.sh")
        if verifier_record is None:
            raise HarborCampaignError("Derived verifier launcher is missing")
        verifier_mode, verifier_digest = _file_record_parts(verifier_record)
        if (
            _tree_digest(unchanged) != expected.unchanged_tree_sha256
            or expected.verifier_script_sha256 != verifier_digest
            or expected.source_verifier_script_mode != 0o644
            or expected.derived_verifier_script_mode != 0o555
            or verifier_mode != expected.derived_verifier_script_mode
        ):
            raise HarborCampaignError(
                "Derived task content or verifier launcher mode changed"
            )
        task_toml, _ = _read_owned_regular_file(
            root / expected.task_id / "task.toml", context="derived task.toml"
        )
        parsed = tomllib.loads(task_toml.decode("utf-8"))
        if (
            parsed["environment"].get("network_mode") != "no-network"
            or parsed["agent"].get("network_mode") != "allowlist"
            or parsed["agent"].get("allowed_hosts") != [RELAY_SENTINEL_HOST]
            or parsed["verifier"].get("network_mode") != "no-network"
        ):
            raise HarborCampaignError("Derived task agent network policy changed")


def verify_staged_agent_source(
    path: Path, *, repo_root: Path
) -> tuple[Path, str]:
    """Admit the exact private namespace package staged from one Git HEAD blob."""

    root = _resolved_external(
        path,
        repo_root=repo_root,
        context="staged Harbor agent source",
        must_exist=True,
    )
    bench_root = root / "bench"
    try:
        root_metadata = os.lstat(root)
        bench_metadata = os.lstat(bench_root)
    except OSError as error:
        raise HarborCampaignError("Staged Harbor agent package is incomplete") from error
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or not stat.S_ISDIR(bench_metadata.st_mode)
        or root_metadata.st_uid != os.geteuid()
        or bench_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(root_metadata.st_mode) != 0o700
        or stat.S_IMODE(bench_metadata.st_mode) != 0o700
        or {entry.name for entry in root.iterdir()} != {"bench"}
        or {entry.name for entry in bench_root.iterdir()}
        != {"harbor_pinned_agents.py"}
    ):
        raise HarborCampaignError("Staged Harbor agent package topology changed")
    records: list[dict[str, Any]] = []
    for relative in HARBOR_AGENT_SOURCE_FILES:
        data, mode = _read_owned_regular_file(
            root / Path(relative), context="staged Harbor agent source file"
        )
        if mode != 0o444:
            raise HarborCampaignError("Staged Harbor agent source mode changed")
        records.append(
            {
                "path": relative,
                "source_mode": 0o644,
                "staged_mode": 0o444,
                "sha256": _sha256_bytes(data),
            }
        )
    payload = {"protocol": "harbor-agent-source-v1", "files": records}
    digest = _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    if digest != HARBOR_AGENT_SOURCE_SHA256:
        raise HarborCampaignError("Staged Harbor agent source digest changed")
    return root, digest


def verify_empty_python_pycache(path: Path, *, repo_root: Path) -> Path:
    root = _resolved_external(
        path,
        repo_root=repo_root,
        context="private Python bytecode cache",
        must_exist=True,
    )
    try:
        metadata = os.lstat(root)
    except OSError as error:
        raise HarborCampaignError("Could not inspect private Python bytecode cache") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or any(root.iterdir())
    ):
        raise HarborCampaignError("Private Python bytecode cache must be empty mode 0700")
    return root


def _safe_process_environment(
    *,
    base_url: str,
    agent_source_root: Path,
    python_pycache_root: Path,
    ambient_env: Mapping[str, str],
) -> dict[str, str]:
    environment = {
        key: value
        for key, value in ambient_env.items()
        if key in _SAFE_ENV_KEYS and isinstance(value, str) and "\0" not in value
    }
    environment.update(
        {
            "HARBOR_TELEMETRY": "off",
            "OPENAI_API_KEY": RELAY_PLACEHOLDER_API_KEY,
            "OPENAI_BASE_URL": base_url,
            "PYTHONPATH": str(agent_source_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(python_pycache_root),
        }
    )
    return environment


def build_harbor_invocation(
    campaign: CampaignSpec,
    *,
    trial: TrialSpec,
    npm_artifact_admission: NpmArtifactAdmission,
    runtime_overlay_admission: RuntimeOverlayAdmission,
    harbor_runtime_admission: HarborRuntimeAdmission,
    agent_source_root: Path,
    python_pycache_root: Path,
    derived_dataset: NetworkPolicyPatch,
    jobs_dir: Path,
    base_url: str,
    repo_root: Path,
    ambient_env: Mapping[str, str] | None = None,
    runtime_admission_validator: Callable[
        [CampaignSpec, HarborRuntimeAdmission], None
    ] = _validate_harbor_runtime_admission,
) -> HarborInvocation:
    """Build one exact one-task Harbor invocation in counterbalanced order."""

    expected_trials = iter_trials(campaign)
    if trial not in expected_trials or expected_trials[trial.index - 1] != trial:
        raise HarborCampaignError("Trial is outside the frozen campaign order")
    _validate_npm_admission(campaign, npm_artifact_admission)
    _validate_runtime_overlay_admission(
        campaign,
        trial,
        runtime_overlay_admission,
        repo_root=repo_root,
    )
    runtime_admission_validator(campaign, harbor_runtime_admission)
    workspace_root = repo_root.resolve(strict=True)
    admitted_source_root, source_digest = verify_staged_agent_source(
        agent_source_root, repo_root=workspace_root
    )
    admitted_pycache_root = verify_empty_python_pycache(
        python_pycache_root, repo_root=workspace_root
    )
    canonical_url = canonical_bridge_base_url(base_url)
    verify_private_task_dataset(campaign, derived_dataset, repo_root=repo_root)
    raw_root = _resolved_external(
        jobs_dir,
        repo_root=repo_root,
        context="raw Harbor jobs directory",
        must_exist=True,
    )
    launcher = harbor_runtime_admission.python_launcher_path
    executable = harbor_runtime_admission.executable_path

    agent = campaign.agent(trial.agent_id)
    task_toml, _ = _read_owned_regular_file(
        derived_dataset.dataset_dir / trial.task_id / "task.toml",
        context="derived invocation task.toml",
    )
    try:
        task_definition = tomllib.loads(task_toml.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise HarborCampaignError("Derived invocation task.toml is invalid") from error
    task_image = _mapping(
        task_definition.get("environment"), "derived task environment"
    ).get("docker_image")
    if (
        not isinstance(task_image, str)
        or len(task_image) > 255
        or _IMAGE_REFERENCE_PATTERN.fullmatch(task_image) is None
    ):
        raise HarborCampaignError("Derived task image reference is unsafe")
    job_name = (
        f"{campaign.id}--{trial.index:02d}--{trial.task_id}--{trial.agent_id}"
    )
    if (raw_root / job_name).exists():
        raise HarborCampaignError("Raw Harbor job directory already exists")
    argv = (
        str(launcher),
        "-B",
        str(executable),
        "run",
        "--job-name",
        job_name,
        "--jobs-dir",
        str(raw_root),
        "--n-attempts",
        "1",
        "--n-concurrent",
        "1",
        "--n-concurrent-agents",
        "1",
        "--max-retries",
        "0",
        "--agent-import-path",
        PINNED_AGENT_IMPORTS[agent.id],
        "--model",
        f"openai/{campaign.model.served_name}",
        "--agent-kwarg",
        f"version={agent.version}",
        "--agent-env",
        "OPENAI_API_KEY=${OPENAI_API_KEY}",
        "--agent-env",
        "OPENAI_BASE_URL=${OPENAI_BASE_URL}",
        "--env",
        "docker",
        "--force-build",
        "--extra-docker-compose",
        str(runtime_overlay_admission.compose_path),
        "--no-delete",
        "--path",
        str(derived_dataset.dataset_dir),
        "--include-task-name",
        trial.task_id,
        "--n-tasks",
        "1",
        "--no-export-traces",
        "--yes",
        "--quiet",
    )
    environment = _safe_process_environment(
        base_url=canonical_url,
        agent_source_root=admitted_source_root,
        python_pycache_root=admitted_pycache_root,
        ambient_env=os.environ if ambient_env is None else ambient_env,
    )
    return HarborInvocation(
        trial=trial,
        job_name=job_name,
        timeout_s=campaign.execution.trial_wall_timeout_s,
        npm_artifact_admission_digest=npm_artifact_admission.digest,
        runtime_overlay_admission_digest=runtime_overlay_admission.digest,
        harbor_runtime_admission_digest=harbor_runtime_admission.digest,
        agent_source_admission_digest=source_digest,
        task_image=task_image,
        relay_image=campaign.relay.node_image,
        workspace_root=workspace_root,
        agent_source_root=admitted_source_root,
        python_pycache_root=admitted_pycache_root,
        raw_job_dir=raw_root / job_name,
        argv=argv,
        env=environment,
    )


@contextmanager
def hold_campaign_lock(workspace: Path) -> Iterator[CampaignLock]:
    """Hold SparkBench's global lock across the complete external lifecycle."""

    lock_path = workspace.resolve(strict=True) / "results" / ".sparkbench.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise HarborCampaignError("Could not open the SparkBench campaign lock") from error
    token = CampaignLock(descriptor=descriptor)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise HarborCampaignError("SparkBench campaign lock is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise HarborCampaignError("Another SparkBench run holds the benchmark lock") from error
        yield token
    finally:
        token.active = False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
            token.descriptor = -1


def _run_process_group(
    argv: Sequence[str], env: Mapping[str, str], timeout_s: float
) -> tuple[int | None, bool]:
    """Run Harbor in a new session and terminate the whole group on timeout."""

    process = subprocess.Popen(
        list(argv),
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        umask=0o077,
    )

    def group_alive() -> bool:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        except OSError as error:
            raise HarborCampaignError(
                "Could not certify Harbor process-group state"
            ) from error
        return True

    def wait_for_group_exit(timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while group_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))
        return True

    def drain_surviving_members() -> None:
        if not group_alive():
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        if wait_for_group_exit(10):
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        if not wait_for_group_exit(10):
            raise HarborCampaignError(
                "Harbor process group survived SIGKILL"
            )

    def terminate_and_reap() -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired as error:
                raise HarborCampaignError(
                    "Harbor process group could not be reaped after SIGKILL"
                ) from error
        drain_surviving_members()

    try:
        exit_code = process.wait(timeout=timeout_s)
        drain_surviving_members()
        return exit_code, False
    except subprocess.TimeoutExpired:
        terminate_and_reap()
        return None, True
    except BaseException:
        terminate_and_reap()
        raise


def _docker_command(
    command: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> subprocess.CompletedProcess[Any]:
    try:
        return runner(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise HarborCampaignError("Docker ownership command failed") from error


def _listed_docker_resources(
    resource: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> frozenset[str]:
    commands = {
        "container": [
            "docker",
            "ps",
            "-aq",
            "--filter",
            "label=com.docker.compose.project",
        ],
        "network": [
            "docker",
            "network",
            "ls",
            "-q",
            "--filter",
            "label=com.docker.compose.project",
        ],
        "volume": [
            "docker",
            "volume",
            "ls",
            "-q",
            "--filter",
            "label=com.docker.compose.project",
        ],
    }
    result = _docker_command(commands[resource], runner=runner)
    if result.returncode != 0 or not isinstance(result.stdout, str):
        raise HarborCampaignError("Could not inventory Harbor Docker resources")
    identifiers = frozenset(line.strip() for line in result.stdout.splitlines() if line.strip())
    pattern = (
        re.compile(r"^[0-9a-f]{12,64}$")
        if resource != "volume"
        else re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    )
    if any(pattern.fullmatch(identifier) is None for identifier in identifiers):
        raise HarborCampaignError("Docker returned an unsafe resource identifier")
    return identifiers


def snapshot_harbor_resources(
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> DockerResourceSnapshot:
    """Capture labeled Compose resources so cleanup never targets pre-existing work."""

    return DockerResourceSnapshot(
        containers=_listed_docker_resources("container", runner=runner),
        networks=_listed_docker_resources("network", runner=runner),
        volumes=_listed_docker_resources("volume", runner=runner),
    )


def _resource_project(
    resource: str,
    identifier: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> str:
    prefix = ["docker"] if resource == "container" else ["docker", resource]
    labels_path = ".Config.Labels" if resource == "container" else ".Labels"
    result = _docker_command(
        [
            *prefix,
            "inspect",
            "--format",
            f'{{{{ index {labels_path} "com.docker.compose.project" }}}}',
            identifier,
        ],
        runner=runner,
    )
    if result.returncode != 0 or not isinstance(result.stdout, str):
        raise HarborCampaignError("Could not inspect one labeled Docker resource")
    project = result.stdout.strip()
    if not project or len(project) > 200 or re.fullmatch(r"[a-z0-9_-]+", project) is None:
        raise HarborCampaignError("Docker resource has an unsafe Compose project label")
    return project


def _owned_project_pattern(invocation: HarborInvocation) -> re.Pattern[str]:
    task_prefix = invocation.trial.task_id[:32].rstrip("_-").lower()
    task_prefix = re.sub(r"[^a-z0-9_-]", "-", task_prefix)
    return re.compile(
        rf"^{re.escape(task_prefix)}__[a-z0-9]{{7}}__"
        r"(?:env|verifier__[a-z0-9_-]+)$"
    )


def _trial_directory(invocation: HarborInvocation) -> Path:
    job_dir = invocation.raw_job_dir
    try:
        job_metadata = os.lstat(job_dir)
    except OSError as error:
        raise HarborCampaignError("Harbor raw job directory is unavailable") from error
    if not stat.S_ISDIR(job_metadata.st_mode) or job_metadata.st_uid != os.geteuid():
        raise HarborCampaignError("Harbor raw job directory is unsafe")
    task_prefix = invocation.trial.task_id[:32].rstrip("_-")
    trial_pattern = re.compile(rf"^{re.escape(task_prefix)}__[A-Za-z0-9]{{7}}$")
    candidates: list[Path] = []
    for child in job_dir.iterdir():
        try:
            metadata = os.lstat(child)
        except OSError as error:
            raise HarborCampaignError("Could not inspect Harbor trial directory") from error
        if trial_pattern.fullmatch(child.name):
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise HarborCampaignError("Harbor trial directory is unsafe")
            candidates.append(child)
        elif stat.S_ISDIR(metadata.st_mode) and not child.name.startswith("."):
            raise HarborCampaignError("Harbor job contains an unexpected directory")
    if len(candidates) != 1:
        raise HarborCampaignError("Harbor job must contain one exact trial directory")
    return candidates[0]


def _known_trial_projects(invocation: HarborInvocation) -> frozenset[str]:
    if not invocation.raw_job_dir.is_dir():
        return frozenset()
    child = _trial_directory(invocation)
    return frozenset({f"{child.name.lower()}__env"})


def _owned_resources(
    resource: str,
    invocation: HarborInvocation,
    baseline: frozenset[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> dict[str, str]:
    current = _listed_docker_resources(resource, runner=runner)
    known_projects = _known_trial_projects(invocation)
    project_pattern = _owned_project_pattern(invocation)
    owned: dict[str, str] = {}
    for identifier in current:
        project = _resource_project(resource, identifier, runner=runner)
        if project in known_projects or (
            identifier not in baseline and project_pattern.fullmatch(project)
        ):
            owned[identifier] = project
    return owned


def _remove_docker_resource(
    resource: str,
    identifier: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> bool:
    if resource == "container":
        command = ["docker", "rm", "--force", identifier]
    else:
        command = ["docker", resource, "rm", identifier]
    return _docker_command(command, runner=runner).returncode == 0


def cleanup_harbor_containers(
    invocation: HarborInvocation,
    *,
    baseline: DockerResourceSnapshot,
    lock: CampaignLock,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> HarborCleanupStatus:
    """Remove only this trial's label-matched Compose resources and certify zero remain."""

    lock.assert_active()
    baseline_by_kind = {
        "container": baseline.containers,
        "network": baseline.networks,
        "volume": baseline.volumes,
    }
    found: dict[str, int] = {}
    removed: dict[str, int] = {}
    succeeded = True
    for resource in ("container", "network", "volume"):
        try:
            owned = _owned_resources(
                resource,
                invocation,
                baseline_by_kind[resource],
                runner=runner,
            )
        except HarborCampaignError:
            found[resource] = 0
            removed[resource] = 0
            succeeded = False
            continue
        found[resource] = len(owned)
        successful_removals = 0
        for identifier in sorted(owned):
            try:
                removed_resource = _remove_docker_resource(
                    resource, identifier, runner=runner
                )
            except HarborCampaignError:
                removed_resource = False
            if removed_resource:
                successful_removals += 1
            else:
                succeeded = False
        removed[resource] = successful_removals
        try:
            remaining = _owned_resources(
                resource,
                invocation,
                baseline_by_kind[resource],
                runner=runner,
            )
        except HarborCampaignError:
            succeeded = False
        else:
            if remaining:
                succeeded = False
    return HarborCleanupStatus(
        succeeded=succeeded,
        containers_found=found["container"],
        containers_removed=removed["container"],
        networks_found=found["network"],
        networks_removed=removed["network"],
        volumes_found=found["volume"],
        volumes_removed=removed["volume"],
    )


def _listed_image_ids(
    *, runner: Callable[..., subprocess.CompletedProcess[Any]]
) -> frozenset[str]:
    result = _docker_command(
        ["docker", "image", "ls", "--no-trunc", "--quiet"], runner=runner
    )
    if result.returncode != 0 or not isinstance(result.stdout, str):
        raise HarborCampaignError("Could not inventory Docker image IDs")
    identifiers = frozenset(
        line.strip() for line in result.stdout.splitlines() if line.strip()
    )
    if any(_SHA256_PATTERN.fullmatch(value) is None for value in identifiers):
        raise HarborCampaignError("Docker returned an unsafe image ID")
    return identifiers


def _built_main_image_reference(invocation: HarborInvocation) -> str:
    trial_directory = _trial_directory(invocation)
    project = f"{trial_directory.name.lower()}__env"
    if _owned_project_pattern(invocation).fullmatch(project) is None:
        raise HarborCampaignError("Harbor built-image project identity changed")
    return f"{project}-main"


def _reject_image_object_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate Docker image JSON key")
        result[key] = value
    return result


def _reject_image_constant(value: str) -> None:
    del value
    raise ValueError("non-finite Docker image JSON value")


def _parse_image_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite Docker image JSON value")
    return parsed


def _inspect_image_id(
    image: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> str:
    result = _docker_command(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        runner=runner,
    )
    if result.returncode != 0 or not isinstance(result.stdout, str):
        raise HarborCampaignError("Could not inspect one Docker image ID")
    lines = result.stdout.splitlines()
    if len(lines) != 1 or _SHA256_PATTERN.fullmatch(lines[0]) is None:
        raise HarborCampaignError("Docker image ID is invalid")
    return lines[0]


def _load_image_inspection(
    image: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> Mapping[str, Any]:
    try:
        result = _docker_command(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{json .}}",
                image,
            ],
            runner=runner,
        )
    except HarborCampaignError:
        raise
    if (
        result.returncode != 0
        or not isinstance(result.stdout, str)
        or len(result.stdout.encode("utf-8")) > MAX_IMAGE_INSPECT_BYTES
    ):
        raise HarborCampaignError("Could not inspect one admitted Docker image")
    try:
        inspected = json.loads(
            result.stdout,
            object_pairs_hook=_reject_image_object_duplicates,
            parse_constant=_reject_image_constant,
            parse_float=_parse_image_float,
        )
    except (json.JSONDecodeError, RecursionError, UnicodeError, ValueError) as error:
        raise HarborCampaignError("Docker image inspection is not JSON") from error
    if not isinstance(inspected, dict):
        raise HarborCampaignError("Docker image inspection is not an object")
    return inspected


def _inspection_image_id(inspected: Mapping[str, Any]) -> str:
    image_id = inspected.get("Id")
    if not isinstance(image_id, str) or _SHA256_PATTERN.fullmatch(image_id) is None:
        raise HarborCampaignError("Docker image ID is invalid")
    return image_id


def _project_image_inspection(
    inspected: Mapping[str, Any],
) -> tuple[str, str, bool]:
    image_id = _inspection_image_id(inspected)
    architecture = inspected.get("Architecture")
    operating_system = inspected.get("Os")
    raw_variant = inspected.get("Variant")
    rootfs = inspected.get("RootFS")
    config = inspected.get("Config")
    if (
        not isinstance(architecture, str)
        or architecture not in {"amd64", "arm64"}
        or operating_system != "linux"
        or not isinstance(rootfs, dict)
        or set(rootfs) != {"Type", "Layers"}
        or rootfs.get("Type") != "layers"
        or not isinstance(rootfs.get("Layers"), list)
        or not rootfs["Layers"]
        or len(rootfs["Layers"]) > MAX_IMAGE_LAYERS
        or any(
            not isinstance(layer, str)
            or _SHA256_PATTERN.fullmatch(layer) is None
            for layer in rootfs["Layers"]
        )
        or not isinstance(config, dict)
    ):
        raise HarborCampaignError("Docker image identity or architecture is invalid")
    if raw_variant is not None and not isinstance(raw_variant, str):
        raise HarborCampaignError("Docker image variant is invalid")
    if architecture == "arm64":
        if raw_variant not in {None, "", "v8"}:
            raise HarborCampaignError("Docker ARM64 image variant is unsupported")
        variant = "v8"
    else:
        if raw_variant not in {None, ""}:
            raise HarborCampaignError("Docker AMD64 image variant is unsupported")
        variant = ""
    config_image = config.get("Image")
    config_labels = config.get("Labels")
    if config_image is not None and not isinstance(config_image, str):
        raise HarborCampaignError("Docker image parent metadata is invalid")
    if config_labels is not None and (
        not isinstance(config_labels, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in config_labels.items()
        )
    ):
        raise HarborCampaignError("Docker image label metadata is invalid")
    semantic_config = {
        key: value for key, value in config.items() if key not in {"Image", "Labels"}
    }
    fingerprint_payload = {
        "schema_version": 1,
        "architecture": architecture,
        "os": operating_system,
        "variant": variant,
        "rootfs": rootfs,
        "config": semantic_config,
    }
    try:
        encoded = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (RecursionError, TypeError, ValueError) as error:
        raise HarborCampaignError("Docker image runtime config is invalid") from error
    return image_id, _sha256_bytes(encoded), architecture == "arm64"


def _inspect_image(
    image: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> tuple[str, str, bool]:
    return _project_image_inspection(_load_image_inspection(image, runner=runner))


def _remove_built_image(
    reference: str,
    image_id: str,
    *,
    baseline_ids: frozenset[str],
    runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> bool:
    removed_tag = _docker_command(
        ["docker", "image", "rm", "--force", reference], runner=runner
    ).returncode == 0
    tag_probe = _docker_command(
        ["docker", "image", "inspect", reference], runner=runner
    )
    if not removed_tag or tag_probe.returncode == 0:
        return False
    if image_id in baseline_ids:
        return True
    _docker_command(
        ["docker", "image", "rm", "--force", image_id], runner=runner
    )
    id_probe = _docker_command(
        ["docker", "image", "inspect", image_id], runner=runner
    )
    return id_probe.returncode != 0


def load_network_admission(invocation: HarborInvocation) -> dict[str, bool]:
    """Read the exact host-authored phase-network marker as booleans only."""

    marker = _read_bounded_json(
        _trial_directory(invocation) / NETWORK_ADMISSION_FILENAME,
        max_bytes=1_024,
        required_mode=0o600,
    )
    expected_keys = frozenset({"schema_version", *NETWORK_ADMISSION_KEYS})
    if (
        frozenset(marker) != expected_keys
        or isinstance(marker.get("schema_version"), bool)
        or marker.get("schema_version") != 1
        or any(marker.get(key) is not True for key in NETWORK_ADMISSION_KEYS)
    ):
        raise HarborCampaignError("Network phase admission marker is invalid")
    return {key: True for key in NETWORK_ADMISSION_KEYS}


def run_harbor_invocation(
    invocation: HarborInvocation,
    *,
    lock: CampaignLock,
    timeout_s: float,
    process_runner: Callable[
        [Sequence[str], Mapping[str, str], float], tuple[int | None, bool]
    ] = _run_process_group,
    docker_runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    clock: Callable[[], float] = time.monotonic,
) -> HarborRunStatus:
    """Run one trial, kill its process group, and clean only owned containers."""

    lock.assert_active()
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise HarborCampaignError("Harbor timeout must be positive and finite")
    source_root, source_digest = verify_staged_agent_source(
        invocation.agent_source_root, repo_root=invocation.workspace_root
    )
    pycache_root = verify_empty_python_pycache(
        invocation.python_pycache_root, repo_root=invocation.workspace_root
    )
    if (
        source_root != invocation.agent_source_root
        or source_digest != invocation.agent_source_admission_digest
        or pycache_root != invocation.python_pycache_root
    ):
        raise HarborCampaignError("Harbor Python source admission changed")
    baseline = snapshot_harbor_resources(runner=docker_runner)
    baseline_image_ids = _listed_image_ids(runner=docker_runner)
    started = clock()
    exit_code: int | None = 127
    timed_out = False
    main_image_id: str | None = None
    main_image_fingerprint: str | None = None
    inspected_main_image_id: str | None = None
    main_image_arm64 = False
    relay_image_arm64 = False
    built_image_cleanup_succeeded = False
    network_admission = {key: False for key in NETWORK_ADMISSION_KEYS}
    cleanup = HarborCleanupStatus(
        succeeded=False,
        containers_found=0,
        containers_removed=0,
        networks_found=0,
        networks_removed=0,
        volumes_found=0,
        volumes_removed=0,
    )
    try:
        try:
            exit_code, timed_out = process_runner(
                invocation.argv, invocation.env, timeout_s
            )
        except OSError:
            exit_code, timed_out = 127, False
    finally:
        built_image_reference: str | None = None
        try:
            built_image_reference = _built_main_image_reference(invocation)
            inspected_main = _load_image_inspection(
                built_image_reference, runner=docker_runner
            )
            inspected_main_image_id = _inspection_image_id(inspected_main)
        except HarborCampaignError:
            inspected_main = None
            try:
                if built_image_reference is None:
                    raise HarborCampaignError("Built image reference is unavailable")
                inspected_main_image_id = _inspect_image_id(
                    built_image_reference, runner=docker_runner
                )
            except HarborCampaignError:
                inspected_main_image_id = None
        try:
            if inspected_main is None:
                raise HarborCampaignError("Built image inspection is unavailable")
            (
                admitted_main_image_id,
                inspected_main_image_fingerprint,
                main_image_arm64,
            ) = _project_image_inspection(inspected_main)
            if main_image_arm64:
                main_image_id = admitted_main_image_id
                main_image_fingerprint = inspected_main_image_fingerprint
        except HarborCampaignError:
            main_image_id = None
            main_image_fingerprint = None
            main_image_arm64 = False
        try:
            _, _, relay_image_arm64 = _inspect_image(
                invocation.relay_image, runner=docker_runner
            )
        except HarborCampaignError:
            relay_image_arm64 = False
        try:
            network_admission = load_network_admission(invocation)
        except HarborCampaignError:
            network_admission = {key: False for key in NETWORK_ADMISSION_KEYS}
        try:
            cleanup = cleanup_harbor_containers(
                invocation, baseline=baseline, lock=lock, runner=docker_runner
            )
        except BaseException:
            # Cleanup evidence remains a fixed failure scalar; never include the
            # Docker exception, command, resource ID, or local path.
            cleanup = HarborCleanupStatus(
                succeeded=False,
                containers_found=0,
                containers_removed=0,
                networks_found=0,
                networks_removed=0,
                volumes_found=0,
                volumes_removed=0,
            )
        if built_image_reference is not None and inspected_main_image_id is not None:
            try:
                built_image_cleanup_succeeded = _remove_built_image(
                    built_image_reference,
                    inspected_main_image_id,
                    baseline_ids=baseline_image_ids,
                    runner=docker_runner,
                )
            except HarborCampaignError:
                built_image_cleanup_succeeded = False
    wall_s = round(max(clock() - started, 0.0), 6)
    return HarborRunStatus(
        trial=invocation.trial,
        exit_code=exit_code,
        timed_out=timed_out,
        wall_s=wall_s,
        main_image_id=main_image_id,
        main_image_fingerprint=main_image_fingerprint,
        main_image_arm64=main_image_arm64,
        relay_image_arm64=relay_image_arm64,
        built_image_cleanup_succeeded=built_image_cleanup_succeeded,
        setup_relay_rejected=network_admission["setup_relay_rejected"],
        agent_relay_passed=network_admission["agent_relay_passed"],
        wrong_auth_rejected=network_admission["wrong_auth_rejected"],
        other_loopback_rejected=network_admission["other_loopback_rejected"],
        gost_rejected=network_admission["gost_rejected"],
        dns_rejected=network_admission["dns_rejected"],
        gateway_rejected=network_admission["gateway_rejected"],
        public_rejected=network_admission["public_rejected"],
        capabilities_dropped=network_admission["capabilities_dropped"],
        cleanup_succeeded=cleanup.succeeded and built_image_cleanup_succeeded,
        containers_found=cleanup.containers_found,
        containers_removed=cleanup.containers_removed,
        networks_found=cleanup.networks_found,
        networks_removed=cleanup.networks_removed,
        volumes_found=cleanup.volumes_found,
        volumes_removed=cleanup.volumes_removed,
    )


def _read_bounded_json(
    path: Path,
    *,
    max_bytes: int | None = None,
    required_mode: int | None = None,
) -> Mapping[str, Any]:
    limit = MAX_RAW_JSON_BYTES if max_bytes is None else max_bytes
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise HarborCampaignError("Raw Harbor JSON size limit is invalid")
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise HarborCampaignError("Raw Harbor result file is missing") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or metadata.st_size <= 0
        or metadata.st_size > limit
        or (
            required_mode is not None
            and stat.S_IMODE(metadata.st_mode) != required_mode
        )
    ):
        raise HarborCampaignError("Raw Harbor result file is unsafe or oversized")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise HarborCampaignError("Could not open raw Harbor result safely") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
            or opened.st_size != metadata.st_size
            or (
                required_mode is not None
                and stat.S_IMODE(opened.st_mode) != required_mode
            )
        ):
            raise HarborCampaignError("Raw Harbor result changed while opening")
        data = bytearray()
        while len(data) <= limit:
            chunk = os.read(
                descriptor, min(1024 * 1024, limit + 1 - len(data))
            )
            if not chunk:
                break
            data.extend(chunk)
        if len(data) != opened.st_size:
            raise HarborCampaignError("Raw Harbor result changed while reading")
    except OSError as error:
        raise HarborCampaignError("Could not read raw Harbor result safely") from error
    finally:
        os.close(descriptor)
    try:
        value = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise HarborCampaignError("Raw Harbor result is not valid JSON") from error
    return _mapping(value, "raw Harbor result")


def load_trial_job_result(
    invocation: HarborInvocation, *, jobs_dir: Path, repo_root: Path
) -> HarborRawResult | None:
    """Bind Harbor's job summary to its exact one child TrialResult."""

    raw_root = _resolved_external(
        jobs_dir,
        repo_root=repo_root,
        context="raw Harbor jobs directory",
        must_exist=True,
    )
    job_dir = raw_root / invocation.job_name
    if not job_dir.exists():
        return None
    try:
        metadata = os.lstat(job_dir)
    except OSError as error:
        raise HarborCampaignError("Could not inspect raw Harbor job directory") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise HarborCampaignError("Raw Harbor job path is not a real directory")
    job_result_path = job_dir / "result.json"
    if not job_result_path.exists():
        return None
    job_result = _read_bounded_json(job_result_path)

    trial_directory = _trial_directory(invocation)
    trial_result_path = trial_directory / "result.json"
    if not trial_result_path.exists():
        return None
    trial_result = _read_bounded_json(trial_result_path)
    if trial_result.get("trial_name") != trial_directory.name:
        raise HarborCampaignError("Raw Harbor trial name does not bind to its directory")
    return HarborRawResult(job=job_result, trial=trial_result)


def _iso_datetime(value: Any, context: str) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        raise HarborCampaignError(f"{context} must be one bounded timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise HarborCampaignError(f"{context} is not an ISO timestamp") from error
    if parsed.tzinfo is None:
        raise HarborCampaignError(f"{context} must include a timezone")
    return parsed


def _duration(value: Any, context: str) -> float | None:
    if value is None:
        return None
    mapping = _mapping(value, context)
    started = mapping.get("started_at")
    finished = mapping.get("finished_at")
    if started is None and finished is None:
        return None
    if started is None or finished is None:
        raise HarborCampaignError(f"{context} has an incomplete timing pair")
    seconds = (_iso_datetime(finished, context) - _iso_datetime(started, context)).total_seconds()
    if not math.isfinite(seconds) or seconds < 0 or seconds > 86_400:
        raise HarborCampaignError(f"{context} duration is invalid")
    return round(seconds, 6)


def _optional_token(value: Any, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HarborCampaignError(f"{context} must be an exact non-negative integer")
    return value


def _reward(trial: Mapping[str, Any]) -> int | float | None:
    verifier = trial.get("verifier_result")
    if verifier is None:
        return None
    rewards = _mapping(verifier, "verifier_result").get("rewards")
    if rewards is None:
        return None
    reward_map = _mapping(rewards, "verifier_result.rewards")
    if "reward" in reward_map:
        value = reward_map["reward"]
    elif len(reward_map) == 1:
        value = next(iter(reward_map.values()))
    else:
        raise HarborCampaignError("Verifier rewards do not have one primary scalar")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarborCampaignError("Primary reward must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
        raise HarborCampaignError("Terminal-Bench primary reward must be binary")
    return int(numeric) if isinstance(value, int) else numeric


def _exception_class(trial: Mapping[str, Any]) -> str | None:
    value = trial.get("exception_info")
    if value is None:
        return None
    exception_type = _mapping(value, "exception_info").get("exception_type")
    if not isinstance(exception_type, str) or exception_type not in _ALLOWED_EXCEPTION_CLASSES:
        raise HarborCampaignError("Raw exception class is not allowlisted")
    return exception_type


def _one_trial_from_job(raw_result: HarborRawResult) -> Mapping[str, Any]:
    total_trials = raw_result.job.get("n_total_trials")
    if isinstance(total_trials, bool) or total_trials != 1:
        raise HarborCampaignError("Each counterbalanced Harbor job must contain one trial")
    stats = _mapping(raw_result.job.get("stats"), "job.stats")
    retries = stats.get("n_retries")
    if isinstance(retries, bool) or retries != 0:
        raise HarborCampaignError("Harbor job unexpectedly retried a trial")
    completed = stats.get("n_completed_trials")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed != 1:
        raise HarborCampaignError("Harbor job did not complete exactly one trial")
    return _mapping(raw_result.trial, "raw Harbor TrialResult")


def _status_projection(status: HarborRunStatus) -> dict[str, Any]:
    boolean_fields = (
        status.timed_out,
        status.main_image_arm64,
        status.relay_image_arm64,
        status.built_image_cleanup_succeeded,
        status.setup_relay_rejected,
        status.agent_relay_passed,
        status.wrong_auth_rejected,
        status.other_loopback_rejected,
        status.gost_rejected,
        status.dns_rejected,
        status.gateway_rejected,
        status.public_rejected,
        status.capabilities_dropped,
        status.cleanup_succeeded,
    )
    if (
        (status.exit_code is not None and (
            isinstance(status.exit_code, bool)
            or not isinstance(status.exit_code, int)
            or status.exit_code < -255
            or status.exit_code > 255
        ))
        or any(not isinstance(value, bool) for value in boolean_fields)
        or (
            status.main_image_id is not None
            and (
                not isinstance(status.main_image_id, str)
                or _SHA256_PATTERN.fullmatch(status.main_image_id) is None
            )
        )
        or (
            status.main_image_fingerprint is not None
            and (
                not isinstance(status.main_image_fingerprint, str)
                or _SHA256_PATTERN.fullmatch(status.main_image_fingerprint) is None
            )
        )
        or status.main_image_arm64
        != (
            status.main_image_id is not None
            and status.main_image_fingerprint is not None
        )
        or not math.isfinite(status.wall_s)
        or status.wall_s < 0
        or (status.exit_code is None) != status.timed_out
        or (
            status.cleanup_succeeded
            and not status.built_image_cleanup_succeeded
        )
    ):
        raise HarborCampaignError("Harbor run status is invalid")
    counts = (
        status.containers_found,
        status.containers_removed,
        status.networks_found,
        status.networks_removed,
        status.volumes_found,
        status.volumes_removed,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise HarborCampaignError("Harbor cleanup counts are invalid")
    if (
        status.containers_removed > status.containers_found
        or status.networks_removed > status.networks_found
        or status.volumes_removed > status.volumes_found
    ):
        raise HarborCampaignError("Harbor cleanup removed-count exceeds found-count")
    return {
        "harbor_exit_code": status.exit_code,
        "harbor_timed_out": status.timed_out,
        "main_image_id": status.main_image_id,
        "main_image_fingerprint": status.main_image_fingerprint,
        "main_image_arm64": status.main_image_arm64,
        "relay_image_arm64": status.relay_image_arm64,
        "built_image_cleanup_succeeded": status.built_image_cleanup_succeeded,
        "setup_relay_rejected": status.setup_relay_rejected,
        "agent_relay_passed": status.agent_relay_passed,
        "wrong_auth_rejected": status.wrong_auth_rejected,
        "other_loopback_rejected": status.other_loopback_rejected,
        "gost_rejected": status.gost_rejected,
        "dns_rejected": status.dns_rejected,
        "gateway_rejected": status.gateway_rejected,
        "public_rejected": status.public_rejected,
        "capabilities_dropped": status.capabilities_dropped,
        "cleanup_succeeded": status.cleanup_succeeded,
        "containers_found": status.containers_found,
        "containers_removed": status.containers_removed,
        "networks_found": status.networks_found,
        "networks_removed": status.networks_removed,
        "volumes_found": status.volumes_found,
        "volumes_removed": status.volumes_removed,
    }


def _status_exception(status: HarborRunStatus) -> str | None:
    if status.timed_out:
        return "CampaignCutoffError"
    if status.exit_code != 0:
        return "HarborProcessError"
    if not status.main_image_arm64 or not status.relay_image_arm64:
        return "HarborProcessError"
    if not all(getattr(status, key) for key in NETWORK_ADMISSION_KEYS):
        return "HarborProcessError"
    if not status.cleanup_succeeded:
        return "HarborCleanupError"
    return None


def _project_attempt(campaign: CampaignSpec, attempt: HarborAttempt) -> dict[str, Any]:
    expected = iter_trials(campaign)[attempt.trial.index - 1]
    if attempt.trial != expected or attempt.status.trial != expected:
        raise HarborCampaignError("Attempt order does not match the frozen campaign")
    status_fields = _status_projection(attempt.status)
    if (
        attempt.job_result is None
        or attempt.status.timed_out
        or attempt.status.exit_code != 0
    ):
        result = {
            "trial_index": expected.index,
            "task": expected.task_id,
            "agent": expected.agent_id,
            "passed": False,
            "result_available": False,
            "reward": None,
            "input_tokens_including_cache": None,
            "cache_tokens": None,
            "output_tokens": None,
            "environment_setup_s": None,
            "agent_setup_s": None,
            "agent_execution_s": None,
            "verifier_s": None,
            "wall_s": attempt.status.wall_s,
            "exception_class": _status_exception(attempt.status)
            or "HarborProcessError",
        }
        result.update(status_fields)
        return result

    raw_trial = _one_trial_from_job(attempt.job_result)
    if raw_trial.get("task_name") != f"terminal-bench/{expected.task_id}":
        raise HarborCampaignError("Raw Harbor task does not match trial_order")
    agent_info = _mapping(raw_trial.get("agent_info"), "trial.agent_info")
    agent_pin = campaign.agent(expected.agent_id)
    if agent_info.get("name") != agent_pin.id or agent_info.get("version") != agent_pin.version:
        raise HarborCampaignError("Raw Harbor agent name or version changed")
    model_info = _mapping(agent_info.get("model_info"), "trial.agent_info.model_info")
    if (
        model_info.get("provider") != "openai"
        or model_info.get("name") != campaign.model.served_name
    ):
        raise HarborCampaignError("Raw Harbor model identity changed")

    agent_result = raw_trial.get("agent_result")
    if agent_result is None:
        input_tokens = cache_tokens = output_tokens = None
    else:
        context = _mapping(agent_result, "trial.agent_result")
        for key in ("n_input_tokens", "n_cache_tokens", "n_output_tokens"):
            if key not in context:
                raise HarborCampaignError("Raw Harbor result omitted exact token fields")
        input_tokens = _optional_token(context["n_input_tokens"], "n_input_tokens")
        cache_tokens = _optional_token(context["n_cache_tokens"], "n_cache_tokens")
        output_tokens = _optional_token(context["n_output_tokens"], "n_output_tokens")

    reward = _reward(raw_trial)
    exception_class = _exception_class(raw_trial) or _status_exception(attempt.status)
    wall_s = _duration(
        {
            "started_at": raw_trial.get("started_at"),
            "finished_at": raw_trial.get("finished_at"),
        },
        "trial.wall",
    )
    if wall_s is None:
        raise HarborCampaignError("Raw Harbor trial omitted wall timing")
    result = {
        "trial_index": expected.index,
        "task": expected.task_id,
        "agent": expected.agent_id,
        "passed": reward == 1 and exception_class is None,
        "result_available": True,
        "reward": reward,
        "input_tokens_including_cache": input_tokens,
        "cache_tokens": cache_tokens,
        "output_tokens": output_tokens,
        "environment_setup_s": _duration(
            raw_trial.get("environment_setup"), "trial.environment_setup"
        ),
        "agent_setup_s": _duration(raw_trial.get("agent_setup"), "trial.agent_setup"),
        "agent_execution_s": _duration(
            raw_trial.get("agent_execution"), "trial.agent_execution"
        ),
        "verifier_s": _duration(raw_trial.get("verifier"), "trial.verifier"),
        "wall_s": wall_s,
        "exception_class": exception_class,
    }
    result.update(status_fields)
    return result


def _agent_pin_projection(agent: AgentPin) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": agent.id,
        "version": agent.version,
        "source": agent.source,
        "revision": agent.revision,
        "install_tree_sha256": agent.install_tree_sha256,
        "install_tree_size_bytes": agent.install_tree_size_bytes,
        "npm_package": agent.npm_package,
        "npm_integrity": agent.npm_integrity,
        "npm_shasum": agent.npm_shasum,
    }
    if agent.platform_package is not None:
        result.update(
            {
                "platform_package": agent.platform_package,
                "platform_integrity": agent.platform_integrity,
                "platform_shasum": agent.platform_shasum,
            }
        )
    return result


def summarize_campaign_results(
    campaign: CampaignSpec,
    attempts: Sequence[HarborAttempt],
    *,
    network_policy_patch_digest: str,
    npm_artifact_admission: NpmArtifactAdmission,
    campaign_cutoff_reached: bool = False,
) -> dict[str, Any]:
    """Project ordered raw Harbor jobs into a strict scalar-only campaign record."""

    if not _SHA256_PATTERN.fullmatch(network_policy_patch_digest):
        raise HarborCampaignError("Network policy patch digest is invalid")
    _validate_npm_admission(campaign, npm_artifact_admission)
    if not isinstance(campaign_cutoff_reached, bool):
        raise HarborCampaignError("campaign_cutoff_reached must be boolean")
    expected_trials = iter_trials(campaign)
    if len(attempts) > len(expected_trials):
        raise HarborCampaignError("Summary requires an ordered trial prefix")
    expected_prefix = expected_trials[: len(attempts)]
    if tuple(attempt.trial for attempt in attempts) != expected_prefix:
        raise HarborCampaignError("Attempt records are not an exact trial_order prefix")
    projected = [_project_attempt(campaign, attempt) for attempt in attempts]
    for item in projected:
        item["paired_image_match"] = None
    by_task: dict[str, list[dict[str, Any]]] = {}
    for item in projected:
        by_task.setdefault(item["task"], []).append(item)
    for task_items in by_task.values():
        if len(task_items) != 2:
            continue
        first_fingerprint = task_items[0]["main_image_fingerprint"]
        matches = (
            first_fingerprint is not None
            and first_fingerprint == task_items[1]["main_image_fingerprint"]
        )
        for item in task_items:
            item["paired_image_match"] = matches
            if not matches:
                item["passed"] = False
                item["exception_class"] = "HarborProcessError"

    passed = sum(1 for item in projected if item["passed"])
    token_fields = (
        "input_tokens_including_cache",
        "cache_tokens",
        "output_tokens",
    )
    token_totals = {
        key: sum(item[key] for item in projected if item[key] is not None)
        for key in token_fields
    }
    token_counts = {
        key: sum(1 for item in projected if item[key] is not None) for key in token_fields
    }
    result = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "protocol": SUMMARY_PROTOCOL,
        "campaign_id": campaign.id,
        "pins": {
            "harbor": {
                "source": campaign.harbor.source,
                "revision": campaign.harbor.revision,
                "version": campaign.harbor.version,
                "runtime_tree_sha256": campaign.harbor.runtime_tree_sha256,
                "runtime_tree_size_bytes": campaign.harbor.runtime_tree_size_bytes,
                "runtime_tree_entries": campaign.harbor.runtime_tree_entries,
                "runtime_tree_files": campaign.harbor.runtime_tree_files,
                "runtime_tree_links": campaign.harbor.runtime_tree_links,
                "executable_size_bytes": campaign.harbor.executable_size_bytes,
                "executable_sha256": campaign.harbor.executable_sha256,
                "agent_source_sha256": campaign.harbor.agent_source_sha256,
                "python_version": campaign.harbor.python_version,
                "python_size_bytes": campaign.harbor.python_size_bytes,
                "python_sha256": campaign.harbor.python_sha256,
            },
            "dataset": {
                "source": campaign.dataset.source,
                "revision": campaign.dataset.revision,
                "version": campaign.dataset.version,
                "network_policy_patch_digest": network_policy_patch_digest,
            },
            "model": {
                "profile": campaign.model.profile,
                "served_name": campaign.model.served_name,
                "context_tokens": campaign.model.context_tokens,
                "max_output_tokens": campaign.model.max_output_tokens,
                "parallel": campaign.model.parallel,
                "temperature": campaign.execution.server_default_temperature,
                "top_p": campaign.execution.server_default_top_p,
                "top_k": campaign.execution.server_default_top_k,
            },
            "relay": {
                "protocol": "phase-isolated-loopback-uds-relay-v1",
                "listen_host": campaign.relay.listen_host,
                "port": campaign.relay.port,
                "sentinel_host": campaign.relay.sentinel_host,
                "node_image": campaign.relay.node_image,
                "relay_script_sha256": campaign.relay.relay_script_sha256,
                "network_policy_sha256": campaign.relay.network_policy_sha256,
            },
            "toolchain": {
                "node_version": campaign.toolchain.node_version,
                "npm_builder_version": campaign.toolchain.npm_builder_version,
                "node_binary_sha256": campaign.toolchain.node_binary_sha256,
                "node_tree_sha256": campaign.toolchain.node_tree_sha256,
                "node_tree_size_bytes": campaign.toolchain.node_tree_size_bytes,
            },
            "npm_artifact_admission": {
                "protocol": "npm-pack-tarball-sha1-sri-v1",
                "digest": npm_artifact_admission.digest,
                "artifacts": [
                    {
                        "package": artifact.package,
                        "version": artifact.version,
                        "size_bytes": artifact.size_bytes,
                        "shasum": artifact.shasum,
                        "integrity": artifact.integrity,
                    }
                    for artifact in npm_artifact_admission.artifacts
                ],
            },
            "agents": [_agent_pin_projection(agent) for agent in campaign.agents],
        },
        "summary": {
            "planned_attempts": len(expected_trials),
            "attempts": len(projected),
            "completed_results": sum(
                1 for item in projected if item["result_available"]
            ),
            "missing_results": sum(
                1 for item in projected if not item["result_available"]
            ),
            "unstarted_attempts": len(expected_trials) - len(projected),
            "campaign_complete": len(projected) == len(expected_trials),
            "campaign_cutoff_reached": campaign_cutoff_reached,
            "passed": passed,
            "pass_rate": passed / len(projected) if projected else None,
            "harbor_process_failures": sum(
                1 for item in projected if item["harbor_exit_code"] not in {0, None}
            ),
            "harbor_timeouts": sum(1 for item in projected if item["harbor_timed_out"]),
            "cleanup_failures": sum(
                1 for item in projected if not item["cleanup_succeeded"]
            ),
            "native_image_admission_failures": sum(
                1
                for item in projected
                if not item["main_image_arm64"] or not item["relay_image_arm64"]
            ),
            "built_image_cleanup_failures": sum(
                1
                for item in projected
                if not item["built_image_cleanup_succeeded"]
            ),
            "network_admission_failures": sum(
                1
                for item in projected
                if not all(item[key] for key in NETWORK_ADMISSION_KEYS)
            ),
            "image_pair_mismatches": sum(
                1
                for task_items in by_task.values()
                if len(task_items) == 2
                and task_items[0]["paired_image_match"] is False
            ),
            "containers_found": sum(item["containers_found"] for item in projected),
            "containers_removed": sum(
                item["containers_removed"] for item in projected
            ),
            "networks_found": sum(item["networks_found"] for item in projected),
            "networks_removed": sum(item["networks_removed"] for item in projected),
            "volumes_found": sum(item["volumes_found"] for item in projected),
            "volumes_removed": sum(item["volumes_removed"] for item in projected),
            "input_tokens_including_cache": token_totals[
                "input_tokens_including_cache"
            ],
            "input_token_measurements": token_counts[
                "input_tokens_including_cache"
            ],
            "cache_tokens": token_totals["cache_tokens"],
            "cache_token_measurements": token_counts["cache_tokens"],
            "output_tokens": token_totals["output_tokens"],
            "output_token_measurements": token_counts["output_tokens"],
            "wall_s": round(sum(item["wall_s"] for item in projected), 6),
        },
        "trials": projected,
    }
    # This is also a final finite-number check before callers persist anything.
    canonical_summary_bytes(result)
    return result


def _output_mapping(
    value: Any, keys: frozenset[str], context: str
) -> Mapping[str, Any]:
    result = _mapping(value, context)
    _expect_keys(result, keys, context)
    return result


def _output_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise HarborCampaignError(f"{context} must be an array")
    return value


def _output_int(
    value: Any, context: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise HarborCampaignError(f"{context} integer is outside its schema")
    return value


def _output_number(
    value: Any,
    context: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarborCampaignError(f"{context} must be numeric")
    numeric = float(value)
    if (
        not math.isfinite(numeric)
        or numeric < minimum
        or (maximum is not None and numeric > maximum)
    ):
        raise HarborCampaignError(f"{context} number is outside its schema")
    return value


def _optional_output_int(value: Any, context: str) -> int | None:
    return None if value is None else _output_int(value, context)


def _optional_output_duration(value: Any, context: str) -> int | float | None:
    if value is None:
        return None
    return _output_number(value, context, maximum=86_400.0)


def _validate_pin_projection(pins_value: Any) -> None:
    pins = _output_mapping(
        pins_value,
        frozenset(
            {
                "harbor",
                "dataset",
                "model",
                "relay",
                "toolchain",
                "npm_artifact_admission",
                "agents",
            }
        ),
        "summary.pins",
    )
    harbor = _output_mapping(
        pins["harbor"],
        frozenset(
            {
                "source",
                "revision",
                "version",
                "runtime_tree_sha256",
                "runtime_tree_size_bytes",
                "runtime_tree_entries",
                "runtime_tree_files",
                "runtime_tree_links",
                "executable_size_bytes",
                "executable_sha256",
                "agent_source_sha256",
                "python_version",
                "python_size_bytes",
                "python_sha256",
            }
        ),
        "pins.harbor",
    )
    if harbor != {
        "source": HARBOR_SOURCE,
        "revision": HARBOR_REVISION,
        "version": HARBOR_VERSION,
        "runtime_tree_sha256": HARBOR_RUNTIME_TREE_SHA256,
        "runtime_tree_size_bytes": HARBOR_RUNTIME_TREE_SIZE_BYTES,
        "runtime_tree_entries": HARBOR_RUNTIME_TREE_ENTRIES,
        "runtime_tree_files": HARBOR_RUNTIME_TREE_FILES,
        "runtime_tree_links": HARBOR_RUNTIME_TREE_LINKS,
        "executable_size_bytes": HARBOR_EXECUTABLE_SIZE_BYTES,
        "executable_sha256": HARBOR_EXECUTABLE_SHA256,
        "agent_source_sha256": HARBOR_AGENT_SOURCE_SHA256,
        "python_version": HARBOR_PYTHON_VERSION,
        "python_size_bytes": HARBOR_PYTHON_SIZE_BYTES,
        "python_sha256": HARBOR_PYTHON_SHA256,
    }:
        raise HarborCampaignError("Canonical summary Harbor pins changed")

    dataset = _output_mapping(
        pins["dataset"],
        frozenset({"source", "revision", "version", "network_policy_patch_digest"}),
        "pins.dataset",
    )
    if (
        dataset.get("source") != DATASET_SOURCE
        or dataset.get("revision") != DATASET_REVISION
        or dataset.get("version") != DATASET_VERSION
        or not isinstance(dataset.get("network_policy_patch_digest"), str)
        or _SHA256_PATTERN.fullmatch(dataset["network_policy_patch_digest"]) is None
    ):
        raise HarborCampaignError("Canonical summary dataset pins changed")

    model = _output_mapping(
        pins["model"],
        frozenset(
            {
                "profile",
                "served_name",
                "context_tokens",
                "max_output_tokens",
                "parallel",
                "temperature",
                "top_p",
                "top_k",
            }
        ),
        "pins.model",
    )
    if model != {
        "profile": MODEL_PROFILE,
        "served_name": MODEL_SERVED_NAME,
        "context_tokens": MODEL_CONTEXT_TOKENS,
        "max_output_tokens": MODEL_MAX_OUTPUT_TOKENS,
        "parallel": 1,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
    }:
        raise HarborCampaignError("Canonical summary model pins changed")

    relay = _output_mapping(
        pins["relay"],
        frozenset(
            {
                "protocol",
                "listen_host",
                "port",
                "sentinel_host",
                "node_image",
                "relay_script_sha256",
                "network_policy_sha256",
            }
        ),
        "pins.relay",
    )
    if relay != {
        "protocol": "phase-isolated-loopback-uds-relay-v1",
        "listen_host": RELAY_LISTEN_HOST,
        "port": RELAY_PORT,
        "sentinel_host": RELAY_SENTINEL_HOST,
        "node_image": RELAY_NODE_IMAGE,
        "relay_script_sha256": RELAY_SCRIPT_SHA256,
        "network_policy_sha256": NETWORK_POLICY_SHA256,
    }:
        raise HarborCampaignError("Canonical summary relay pins changed")

    toolchain = _output_mapping(
        pins["toolchain"],
        frozenset(
            {
                "node_version",
                "npm_builder_version",
                "node_binary_sha256",
                "node_tree_sha256",
                "node_tree_size_bytes",
            }
        ),
        "pins.toolchain",
    )
    if toolchain != {
        "node_version": NODE_VERSION,
        "npm_builder_version": NPM_BUILDER_VERSION,
        "node_binary_sha256": NODE_BINARY_SHA256,
        "node_tree_sha256": NODE_TREE_SHA256,
        "node_tree_size_bytes": NODE_TREE_SIZE_BYTES,
    }:
        raise HarborCampaignError("Canonical summary toolchain pins changed")

    agents = _output_list(pins["agents"], "pins.agents")
    if len(agents) != len(EXPECTED_AGENT_IDS):
        raise HarborCampaignError("Canonical summary agent count changed")
    declared_artifacts: dict[str, tuple[str, str, str]] = {}
    for index, agent_id in enumerate(EXPECTED_AGENT_IDS):
        expected_pin = EXPECTED_AGENT_PINS[agent_id]
        expected_keys = {
            "id",
            "version",
            "source",
            "revision",
            "install_tree_sha256",
            "install_tree_size_bytes",
            "npm_package",
            "npm_integrity",
            "npm_shasum",
        }
        if expected_pin["platform_package"] is not None:
            expected_keys.update(
                {"platform_package", "platform_integrity", "platform_shasum"}
            )
        agent = _output_mapping(
            agents[index], frozenset(expected_keys), f"pins.agents[{index}]"
        )
        expected_projection = {
            "id": agent_id,
            **{
                key: value
                for key, value in expected_pin.items()
                if value is not None
            },
        }
        if agent != expected_projection:
            raise HarborCampaignError("Canonical summary agent pins changed")
        declared_artifacts[agent["npm_package"]] = (
            agent["version"],
            agent["npm_shasum"],
            agent["npm_integrity"],
        )
        if "platform_package" in agent:
            declared_artifacts[agent["platform_package"]] = (
                agent["version"],
                agent["platform_shasum"],
                agent["platform_integrity"],
            )

    npm = _output_mapping(
        pins["npm_artifact_admission"],
        frozenset({"protocol", "digest", "artifacts"}),
        "pins.npm_artifact_admission",
    )
    if (
        npm.get("protocol") != "npm-pack-tarball-sha1-sri-v1"
        or not isinstance(npm.get("digest"), str)
        or _SHA256_PATTERN.fullmatch(npm["digest"]) is None
    ):
        raise HarborCampaignError("Canonical summary npm admission changed")
    artifacts = _output_list(npm["artifacts"], "npm artifacts")
    records: list[dict[str, Any]] = []
    for index, value in enumerate(artifacts):
        record = _output_mapping(
            value,
            frozenset({"package", "version", "size_bytes", "shasum", "integrity"}),
            f"npm artifacts[{index}]",
        )
        package = record.get("package")
        if (
            not isinstance(package, str)
            or package not in declared_artifacts
            or (
                record.get("version"),
                record.get("shasum"),
                record.get("integrity"),
            )
            != declared_artifacts[package]
        ):
            raise HarborCampaignError("Canonical summary npm artifact pin changed")
        _output_int(record.get("size_bytes"), "npm artifact size", minimum=1)
        records.append(dict(record))
    if [record["package"] for record in records] != sorted(declared_artifacts):
        raise HarborCampaignError("Canonical summary npm artifact order changed")
    expected_digest = _sha256_bytes(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    if npm["digest"] != expected_digest:
        raise HarborCampaignError("Canonical summary npm admission digest changed")


_TRIAL_OUTPUT_KEYS = frozenset(
    {
        "trial_index",
        "task",
        "agent",
        "passed",
        "result_available",
        "reward",
        "input_tokens_including_cache",
        "cache_tokens",
        "output_tokens",
        "environment_setup_s",
        "agent_setup_s",
        "agent_execution_s",
        "verifier_s",
        "wall_s",
        "exception_class",
        "harbor_exit_code",
        "harbor_timed_out",
        "main_image_id",
        "main_image_fingerprint",
        "main_image_arm64",
        "relay_image_arm64",
        "built_image_cleanup_succeeded",
        "setup_relay_rejected",
        "agent_relay_passed",
        "wrong_auth_rejected",
        "other_loopback_rejected",
        "gost_rejected",
        "dns_rejected",
        "gateway_rejected",
        "public_rejected",
        "capabilities_dropped",
        "paired_image_match",
        "cleanup_succeeded",
        "containers_found",
        "containers_removed",
        "networks_found",
        "networks_removed",
        "volumes_found",
        "volumes_removed",
    }
)


def _validate_trial_projection(value: Any, index: int) -> Mapping[str, Any]:
    trial = _output_mapping(value, _TRIAL_OUTPUT_KEYS, f"summary.trials[{index - 1}]")
    expected_task, expected_agent = EXPECTED_TRIAL_ORDER[index - 1].split(":")
    if (
        trial.get("trial_index") != index
        or trial.get("task") != expected_task
        or trial.get("agent") != expected_agent
        or not isinstance(trial.get("passed"), bool)
        or not isinstance(trial.get("result_available"), bool)
        or any(
            not isinstance(trial.get(key), bool)
            for key in (
                "harbor_timed_out",
                "main_image_arm64",
                "relay_image_arm64",
                "built_image_cleanup_succeeded",
                *NETWORK_ADMISSION_KEYS,
                "cleanup_succeeded",
            )
        )
        or (
            trial.get("paired_image_match") is not None
            and not isinstance(trial.get("paired_image_match"), bool)
        )
    ):
        raise HarborCampaignError("Canonical trial identity or flags changed")
    image_id = trial.get("main_image_id")
    image_fingerprint = trial.get("main_image_fingerprint")
    if image_id is not None and (
        not isinstance(image_id, str) or _SHA256_PATTERN.fullmatch(image_id) is None
    ):
        raise HarborCampaignError("Canonical trial image ID is invalid")
    if image_fingerprint is not None and (
        not isinstance(image_fingerprint, str)
        or _SHA256_PATTERN.fullmatch(image_fingerprint) is None
    ):
        raise HarborCampaignError("Canonical trial image fingerprint is invalid")
    if trial["main_image_arm64"] != (
        image_id is not None and image_fingerprint is not None
    ):
        raise HarborCampaignError(
            "Canonical image admission lacks an ID and runtime fingerprint"
        )
    if trial["cleanup_succeeded"] and not trial["built_image_cleanup_succeeded"]:
        raise HarborCampaignError("Canonical built image cleanup is inconsistent")
    reward = trial.get("reward")
    if reward is not None and (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(float(reward))
        or float(reward) not in {0.0, 1.0}
    ):
        raise HarborCampaignError("Canonical trial reward is invalid")
    for key in ("input_tokens_including_cache", "cache_tokens", "output_tokens"):
        _optional_output_int(trial.get(key), f"trial.{key}")
    for key in (
        "environment_setup_s",
        "agent_setup_s",
        "agent_execution_s",
        "verifier_s",
    ):
        _optional_output_duration(trial.get(key), f"trial.{key}")
    _output_number(trial.get("wall_s"), "trial.wall_s", maximum=86_400.0)
    exception_class = trial.get("exception_class")
    if exception_class is not None and (
        not isinstance(exception_class, str)
        or exception_class not in _ALLOWED_EXCEPTION_CLASSES
    ):
        raise HarborCampaignError("Canonical trial exception class is invalid")
    expected_passed = (
        reward is not None
        and float(reward) == 1.0
        and exception_class is None
    )
    if trial["passed"] != expected_passed:
        raise HarborCampaignError("Canonical passed trial is inconsistent")
    if not trial["result_available"] and any(
        trial[key] is not None
        for key in (
            "reward",
            "input_tokens_including_cache",
            "cache_tokens",
            "output_tokens",
            "environment_setup_s",
            "agent_setup_s",
            "agent_execution_s",
            "verifier_s",
        )
    ):
        raise HarborCampaignError("Unavailable result contains raw-result scalars")
    if trial["passed"] and (
        not trial["main_image_arm64"] or not trial["relay_image_arm64"]
    ):
        raise HarborCampaignError("Canonical passed trial lacks native image admission")
    if trial["passed"] and (
        not trial["cleanup_succeeded"]
        or not all(trial[key] for key in NETWORK_ADMISSION_KEYS)
        or trial["paired_image_match"] is False
    ):
        raise HarborCampaignError("Canonical passed trial lacks security admission")
    exit_code = trial.get("harbor_exit_code")
    if exit_code is not None:
        _output_int(exit_code, "trial.harbor_exit_code", minimum=-255, maximum=255)
    if (exit_code is None) != trial["harbor_timed_out"]:
        raise HarborCampaignError("Canonical trial timeout status is inconsistent")
    for resource in ("containers", "networks", "volumes"):
        found = _output_int(trial.get(f"{resource}_found"), f"trial.{resource}_found")
        removed = _output_int(
            trial.get(f"{resource}_removed"), f"trial.{resource}_removed"
        )
        if removed > found:
            raise HarborCampaignError("Canonical trial cleanup counts are inconsistent")
    return trial


def _validate_scalar_summary(summary_value: Any) -> None:
    summary = _output_mapping(
        summary_value,
        frozenset({"schema_version", "protocol", "campaign_id", "pins", "summary", "trials"}),
        "scalar summary",
    )
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA_VERSION
        or summary.get("protocol") != SUMMARY_PROTOCOL
        or summary.get("campaign_id") != CAMPAIGN_ID
    ):
        raise HarborCampaignError("Scalar summary identity changed")
    _validate_pin_projection(summary["pins"])
    trials_raw = _output_list(summary["trials"], "summary.trials")
    if len(trials_raw) > len(EXPECTED_TRIAL_ORDER):
        raise HarborCampaignError("Scalar summary trials are not an ordered prefix")
    trials = [
        _validate_trial_projection(value, index)
        for index, value in enumerate(trials_raw, start=1)
    ]
    for offset in range(0, len(trials), 2):
        first = trials[offset]
        if offset + 1 >= len(trials):
            if first["paired_image_match"] is not None:
                raise HarborCampaignError("Incomplete image pair must remain unclassified")
            continue
        second = trials[offset + 1]
        if first["task"] != second["task"]:
            raise HarborCampaignError("Canonical trial pairing changed")
        expected_match = (
            first["main_image_fingerprint"] is not None
            and first["main_image_fingerprint"]
            == second["main_image_fingerprint"]
        )
        if (
            first["paired_image_match"] is not expected_match
            or second["paired_image_match"] is not expected_match
        ):
            raise HarborCampaignError("Canonical image-pair admission changed")

    totals = _output_mapping(
        summary["summary"],
        frozenset(
            {
                "planned_attempts",
                "attempts",
                "completed_results",
                "missing_results",
                "unstarted_attempts",
                "campaign_complete",
                "campaign_cutoff_reached",
                "passed",
                "pass_rate",
                "harbor_process_failures",
                "harbor_timeouts",
                "cleanup_failures",
                "native_image_admission_failures",
                "built_image_cleanup_failures",
                "network_admission_failures",
                "image_pair_mismatches",
                "containers_found",
                "containers_removed",
                "networks_found",
                "networks_removed",
                "volumes_found",
                "volumes_removed",
                "input_tokens_including_cache",
                "input_token_measurements",
                "cache_tokens",
                "cache_token_measurements",
                "output_tokens",
                "output_token_measurements",
                "wall_s",
            }
        ),
        "summary.summary",
    )
    attempts = len(trials)
    _output_int(totals.get("planned_attempts"), "planned_attempts")
    _output_int(totals.get("attempts"), "attempts")
    completed = _output_int(totals.get("completed_results"), "completed_results")
    missing = _output_int(totals.get("missing_results"), "missing_results")
    _output_int(totals.get("unstarted_attempts"), "unstarted_attempts")
    if not isinstance(totals.get("campaign_complete"), bool) or not isinstance(
        totals.get("campaign_cutoff_reached"), bool
    ):
        raise HarborCampaignError("Scalar summary campaign flags are invalid")
    if (
        totals.get("planned_attempts") != len(EXPECTED_TRIAL_ORDER)
        or totals.get("attempts") != attempts
        or completed != sum(1 for trial in trials if trial["result_available"])
        or missing != sum(1 for trial in trials if not trial["result_available"])
        or completed + missing != attempts
        or totals.get("unstarted_attempts") != len(EXPECTED_TRIAL_ORDER) - attempts
        or totals.get("campaign_complete") != (attempts == len(EXPECTED_TRIAL_ORDER))
    ):
        raise HarborCampaignError("Scalar summary attempt accounting is inconsistent")
    passed = sum(1 for trial in trials if trial["passed"])
    for key in (
        "passed",
        "harbor_process_failures",
        "harbor_timeouts",
        "cleanup_failures",
        "native_image_admission_failures",
        "built_image_cleanup_failures",
        "network_admission_failures",
        "image_pair_mismatches",
        "containers_found",
        "containers_removed",
        "networks_found",
        "networks_removed",
        "volumes_found",
        "volumes_removed",
        "input_tokens_including_cache",
        "input_token_measurements",
        "cache_tokens",
        "cache_token_measurements",
        "output_tokens",
        "output_token_measurements",
    ):
        _output_int(totals.get(key), f"summary.{key}")
    if attempts == 0:
        if totals.get("pass_rate") is not None:
            raise HarborCampaignError("Empty summary pass_rate must be null")
    else:
        _output_number(totals.get("pass_rate"), "summary.pass_rate", maximum=1.0)
    _output_number(totals.get("wall_s"), "summary.wall_s")
    if (
        totals.get("passed") != passed
        or totals.get("pass_rate") != (passed / attempts if attempts else None)
        or totals.get("harbor_process_failures")
        != sum(1 for trial in trials if trial["harbor_exit_code"] not in {0, None})
        or totals.get("harbor_timeouts")
        != sum(1 for trial in trials if trial["harbor_timed_out"])
        or totals.get("cleanup_failures")
        != sum(1 for trial in trials if not trial["cleanup_succeeded"])
        or totals.get("native_image_admission_failures")
        != sum(
            1
            for trial in trials
            if not trial["main_image_arm64"] or not trial["relay_image_arm64"]
        )
        or totals.get("built_image_cleanup_failures")
        != sum(
            1 for trial in trials if not trial["built_image_cleanup_succeeded"]
        )
        or totals.get("network_admission_failures")
        != sum(
            1
            for trial in trials
            if not all(trial[key] for key in NETWORK_ADMISSION_KEYS)
        )
        or totals.get("image_pair_mismatches")
        != sum(
            1
            for offset in range(0, len(trials) - 1, 2)
            if trials[offset]["task"] == trials[offset + 1]["task"]
            and trials[offset]["paired_image_match"] is False
        )
    ):
        raise HarborCampaignError("Scalar summary outcome accounting is inconsistent")
    for resource in ("containers", "networks", "volumes"):
        for action in ("found", "removed"):
            key = f"{resource}_{action}"
            if totals.get(key) != sum(trial[key] for trial in trials):
                raise HarborCampaignError("Scalar summary cleanup accounting changed")
    token_pairs = (
        ("input_tokens_including_cache", "input_token_measurements"),
        ("cache_tokens", "cache_token_measurements"),
        ("output_tokens", "output_token_measurements"),
    )
    for token_key, count_key in token_pairs:
        measured = [trial[token_key] for trial in trials if trial[token_key] is not None]
        if totals.get(token_key) != sum(measured) or totals.get(count_key) != len(measured):
            raise HarborCampaignError("Scalar summary token accounting changed")
    expected_wall = round(sum(float(trial["wall_s"]) for trial in trials), 6)
    if totals.get("wall_s") != expected_wall:
        raise HarborCampaignError("Scalar summary wall timing accounting changed")


def canonical_summary_bytes(summary: Mapping[str, Any]) -> bytes:
    """Validate the exact recursive scalar schema and encode canonical JSON."""

    _validate_scalar_summary(summary)

    try:
        return (
            json.dumps(
                summary,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise HarborCampaignError("Scalar summary is not canonical JSON") from error
