"""Safe lifecycle management for benchmark inference servers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import http.client
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import selectors
import signal
import socket
import subprocess
import threading
import time
from typing import Any, Callable, TextIO
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from .execution_admission import model_execution_blocker
from .seccomp_profile_contract import (
    DERIVED_PATH as SM121_STORAGE_SECCOMP_PATH,
    DERIVED_SHA256 as SM121_STORAGE_SECCOMP_SHA256,
    SeccompProfileContractError,
    verify_seccomp_profile_contract,
)
from .sglang_sm121_storage import (
    SM121_STORAGE_BUILD_CONTRACT_SHA256,
    SM121_STORAGE_CACHE_PAGES,
    SM121_STORAGE_CANDIDATE_ID,
    SM121_STORAGE_MAX_BATCH_PAGES,
    SM121_STORAGE_MODE,
    SM121_STORAGE_QUEUE_DEPTH,
    SM121_STORAGE_SOURCE_TREE,
    SM121StorageCandidateError,
    is_sm121_storage_candidate,
    validate_sm121_storage_candidate,
    validate_sm121_storage_image_inspection,
)
from .sglang_sm121_cache_semantic import (
    SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
    SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
    SM121_CACHE_SEMANTIC_EXECUTION_MODE,
    SM121CacheSemanticError,
    is_sm121_cache_semantic_candidate,
    validate_sm121_cache_semantic_candidate,
)
from .sglang_sm121_cache_performance import (
    SM121_CACHE_PERFORMANCE_EXECUTION_MODE,
    SM121CachePerformanceError,
    is_sm121_cache_performance_candidate,
    validate_sm121_cache_performance_candidate,
)
from .sglang_sm121_chunked_prefill_performance import (
    SM121ChunkedPrefillPerformanceError,
    is_sm121_chunked_prefill_performance_candidate,
    validate_sm121_chunked_prefill_performance_candidate,
)
from .sglang_sm121_cache_observability import (
    SM121_CACHE_RUNTIME_ATTESTATION_EVENT,
    SM121_CACHE_RUNTIME_EXPECTED,
    SM121_CACHE_SOURCE_DIGESTS,
    SM121_CACHE_STATIC_ASSERTIONS,
    SM121_CACHE_STATIC_ATTESTATION_EVENT,
    SM121_CACHE_ZERO_HIT_EXPECTED_RESPONSE,
    SM121CacheObservabilityError,
    sm121_cache_zero_hit_request_body,
    validate_sm121_cache_zero_hit_request_contract,
    validate_sm121_cache_runtime_attestation_event,
    validate_sm121_cache_static_attestation_event,
)
from .sglang_sm121_agent_admission import (
    SM121_AGENT_ADMISSION_ENDPOINT,
    SM121AgentAdmissionError,
    validate_sm121_agent_admission_runtime_identity,
)
from bench.qwen38_ple_cache import (
    PINNED_LAYOUT as QWEN38_PLE_LAYOUT,
    PLECacheError,
    PLECacheRecord,
    default_cache_path as qwen38_ple_cache_path,
    expected_marker_sha256 as qwen38_ple_marker_sha256,
    validate_ple_cache as validate_qwen38_ple_cache,
)


MANAGED_LABEL = "ai.sparkbench.managed=true"
VLLM_CONTAINER_NAME = "sparkbench-vllm"
SGLANG_CONTAINER_NAME = "sparkbench-sglang"
# Backward-compatible alias for callers and tests that target the vLLM lifecycle.
CONTAINER_NAME = VLLM_CONTAINER_NAME
LLAMACPP_BACKEND = "llamacpp"
_NATIVE_ENV_ALLOWLIST = frozenset(
    {
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "LANG",
        "LC_ALL",
        "OMP_NUM_THREADS",
        "PATH",
        "TMPDIR",
    }
)
_SPLIT_GGUF_PATTERN = re.compile(
    r"^(?P<prefix>.+)-(?P<ordinal>[0-9]+)-of-(?P<total>[0-9]+)\.gguf$",
    re.IGNORECASE,
)
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_QWEN38_LEGACY_READONLY_PLE_OVERLAYS = {
    (
        "/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py"
    ): (
        "sha256:0b513b4dc4f2394f6b1733bb0b74fa40"
        "ab59f4a04f6b33601350b2a606c67804"
    ),
    (
        "/sgl-workspace/sglang/python/sglang/srt/layers/attention/"
        "qwen_sparse_attn_backend.py"
    ): (
        "sha256:e30566492e1502f94a4c7fed42d90b5"
        "23bbb662580c628459e6e63c7b5263c75"
    ),
}
_QWEN38_ABLATION_CAPABLE_PLE_OVERLAYS = {
    (
        "/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py"
    ): (
        "sha256:bcdc2c86aa59784ffe27d53c8d214e56"
        "b6aa45c02b1d5841fd956d1f006d6030"
    ),
    (
        "/sgl-workspace/sglang/python/sglang/srt/layers/attention/"
        "qwen_sparse_attn_backend.py"
    ): (
        "sha256:e30566492e1502f94a4c7fed42d90b5"
        "23bbb662580c628459e6e63c7b5263c75"
    ),
}
_QWEN38_READONLY_PLE_OVERLAY_VARIANTS = (
    _QWEN38_LEGACY_READONLY_PLE_OVERLAYS,
    _QWEN38_ABLATION_CAPABLE_PLE_OVERLAYS,
)


class RuntimeErrorWithContext(RuntimeError):
    pass


def _ollama_model_loaded(base_url: str, model: str) -> bool:
    ps_url = base_url.removesuffix("/v1").rstrip("/") + "/api/ps"
    try:
        with urllib.request.urlopen(ps_url, timeout=5) as response:
            loaded = json.load(response).get("models", [])
    except (OSError, ValueError) as error:
        raise RuntimeErrorWithContext(
            f"Could not verify Ollama loaded-model state for lifecycle ownership: {error}"
        ) from error
    return any(
        item.get("name") == model or item.get("model") == model for item in loaded
    )


def ollama_model_loaded(base_url: str, model: str) -> bool:
    """Public fail-closed loaded-state probe used during crash recovery."""

    return _ollama_model_loaded(base_url, model)


def _run(command: list[str], *, check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _redact_text(text: str, sensitive_values: tuple[str, ...]) -> str:
    redacted = text
    for value in sensitive_values:
        if value:
            redacted = redacted.replace(value, "<redacted>")
    return redacted


def _redact_exception(
    error: BaseException, sensitive_values: tuple[str, ...]
) -> None:
    error.args = tuple(
        _redact_text(argument, sensitive_values)
        if isinstance(argument, str)
        else argument
        for argument in error.args
    )
    notes = getattr(error, "__notes__", None)
    if notes:
        error.__notes__ = [
            _redact_text(note, sensitive_values) for note in notes
        ]


def _authorization_headers(authorization: str | None) -> dict[str, str]:
    return {"Authorization": authorization} if authorization else {}


def endpoint_ready(
    base_url: str,
    timeout_s: float = 2,
    *,
    authorization: str | None = None,
) -> bool:
    root = base_url.removesuffix("/v1").rstrip("/")
    for path in ("/health", "/v1/models"):
        try:
            request = urllib.request.Request(
                root + path,
                headers=_authorization_headers(authorization),
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                if response.status < 400:
                    return True
        except (OSError, urllib.error.URLError, TimeoutError):
            continue
    return False


def wait_for_endpoint(
    base_url: str,
    timeout_s: float,
    container_id: str | None = None,
    *,
    authorization: str | None = None,
    sensitive_values: tuple[str, ...] = (),
    abort_check: Callable[[], None] | None = None,
) -> float:
    started = time.monotonic()
    auth = authorization
    while time.monotonic() - started < timeout_s:
        if abort_check is not None:
            abort_check()
        if endpoint_ready(base_url, authorization=auth):
            return time.monotonic() - started
        if container_id:
            state = _run(
                ["docker", "inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}", container_id],
                check=False,
                timeout=10,
            )
            if state.returncode or state.stdout.startswith("exited"):
                logs = _run(
                    ["docker", "logs", "--tail", "100", container_id],
                    check=False,
                    timeout=30,
                )
                raise RuntimeErrorWithContext(
                    _redact_text(
                        f"Server exited during startup: {state.stdout.strip()}\n"
                        f"{logs.stdout}{logs.stderr}",
                        sensitive_values,
                    )
                )
        if abort_check is not None:
            abort_check()
        time.sleep(2)
    if abort_check is not None:
        abort_check()
    raise RuntimeErrorWithContext(f"Server did not become ready within {timeout_s:.0f}s")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _huggingface_root(model: Any, workspace: Path) -> Path:
    configured_cache = getattr(model, "cache_dir", None)
    if configured_cache == "user":
        return (Path.home() / ".cache" / "huggingface").resolve()
    if configured_cache in {None, "project"}:
        return (workspace / "data" / "huggingface").resolve()
    raise RuntimeErrorWithContext(
        f"Unsupported Hugging Face cache selector: {configured_cache!r}"
    )


def _validate_llamacpp_snapshot_artifact(
    *,
    repository_resolved: Path,
    snapshot: Path,
    filename: str,
    expected_size: int,
    expected_digest: str,
    label: str,
) -> dict[str, Any]:
    if (
        not filename
        or filename.startswith(("-", "/", "\\", "~"))
        or "\\" in filename
        or ":" in filename
        or any(ord(character) < 32 for character in filename)
    ):
        raise RuntimeErrorWithContext(
            f"Exact cached {label} uses an unsafe snapshot path: {filename!r}"
        )
    relative = PurePosixPath(filename)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise RuntimeErrorWithContext(
            f"Exact cached {label} uses an unsafe snapshot path: {filename!r}"
        )
    artifact_path = snapshot.joinpath(*relative.parts)
    try:
        snapshot_resolved = snapshot.resolve(strict=True)
        artifact_path.parent.resolve(strict=True).relative_to(snapshot_resolved)
        artifact_target = artifact_path.resolve(strict=True)
        artifact_target.relative_to(repository_resolved)
    except (OSError, ValueError) as error:
        raise RuntimeErrorWithContext(
            f"Exact cached {label} is missing or escapes its repository: "
            f"{artifact_path}"
        ) from error
    if not artifact_target.is_file():
        raise RuntimeErrorWithContext(
            f"Exact cached {label} is not a file: {artifact_path}"
        )
    actual_size = artifact_target.stat().st_size
    if actual_size != expected_size:
        raise RuntimeErrorWithContext(
            f"{label} size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_digest = _sha256_file(artifact_target)
    if actual_digest != expected_digest:
        raise RuntimeErrorWithContext(
            f"{label} SHA-256 mismatch: expected {expected_digest}, "
            f"got {actual_digest}"
        )
    return {
        "path": str(artifact_path),
        "target": str(artifact_target),
        "sha256": actual_digest,
        "size_bytes": actual_size,
    }


def _validate_runtime_model_shards(shards: tuple[Any, ...]) -> None:
    if len(shards) < 2:
        raise RuntimeErrorWithContext(
            "Split GGUF provenance must contain at least two shards"
        )
    seen_paths: set[str] = set()
    seen_basenames: set[str] = set()
    expected_parent: PurePosixPath | None = None
    expected_prefix: str | None = None
    expected_total: int | None = None
    expected_width: int | None = None
    for index, shard in enumerate(shards, start=1):
        path = getattr(shard, "path", None)
        digest = getattr(shard, "digest", None)
        size_bytes = getattr(shard, "size_bytes", None)
        if not isinstance(path, str) or not path or path != path.strip():
            raise RuntimeErrorWithContext(
                "Split GGUF provenance contains an invalid shard path"
            )
        if (
            path.startswith(("-", "/", "\\", "~"))
            or "\\" in path
            or ":" in path
            or any(ord(character) < 32 for character in path)
        ):
            raise RuntimeErrorWithContext(
                "Split GGUF provenance contains an unsafe shard path"
            )
        relative = PurePosixPath(path)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in path.split("/")
        ):
            raise RuntimeErrorWithContext(
                "Split GGUF provenance contains an unsafe shard path"
            )
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise RuntimeErrorWithContext(
                "Split GGUF provenance must pin every shard SHA-256"
            )
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise RuntimeErrorWithContext(
                "Split GGUF provenance must pin every shard byte size"
            )
        match = _SPLIT_GGUF_PATTERN.fullmatch(relative.name)
        if (
            not path
            or path in seen_paths
            or relative.name in seen_basenames
            or match is None
        ):
            raise RuntimeErrorWithContext(
                "Split GGUF provenance contains a duplicate or invalid shard path"
            )
        seen_paths.add(path)
        seen_basenames.add(relative.name)
        ordinal_text = match.group("ordinal")
        total_text = match.group("total")
        total = int(total_text)
        if (
            int(ordinal_text) != index
            or total < 2
            or len(ordinal_text) != len(total_text)
        ):
            raise RuntimeErrorWithContext(
                "Split GGUF provenance is unordered or has a missing shard index"
            )
        if expected_parent is None:
            expected_parent = relative.parent
            expected_prefix = match.group("prefix")
            expected_total = total
            expected_width = len(total_text)
        elif (
            relative.parent != expected_parent
            or match.group("prefix") != expected_prefix
            or total != expected_total
            or len(total_text) != expected_width
        ):
            raise RuntimeErrorWithContext(
                "Split GGUF provenance does not describe one canonical shard set"
            )
    if expected_total != len(shards):
        raise RuntimeErrorWithContext(
            "Split GGUF provenance omits one or more declared shards"
        )


def validate_llamacpp_artifacts(model: Any, *, workspace: Path) -> dict[str, Any]:
    """Validate the exact source, executable, and GGUF frozen by a profile."""

    binary = Path(str(model.runtime_binary)).expanduser()
    source_dir = Path(str(model.runtime_source_dir)).expanduser()
    try:
        binary = binary.resolve(strict=True)
        source_dir = source_dir.resolve(strict=True)
        binary.relative_to(source_dir)
    except (OSError, ValueError) as error:
        raise RuntimeErrorWithContext(
            "Pinned llama.cpp source or llama-server binary is missing or escapes "
            "runtime_source_dir"
        ) from error
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeErrorWithContext(
            f"Pinned llama-server is not an executable regular file: {binary}"
        )

    revision = _run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=False,
        timeout=15,
    )
    if revision.returncode or revision.stdout.strip() != str(model.runtime_revision):
        raise RuntimeErrorWithContext(
            "llama.cpp source revision does not match the frozen profile"
        )
    dirty = _run(
        ["git", "-C", str(source_dir), "status", "--porcelain", "--untracked-files=no"],
        check=False,
        timeout=15,
    )
    if dirty.returncode or dirty.stdout.strip():
        raise RuntimeErrorWithContext(
            "llama.cpp tracked source must be clean at the frozen revision"
        )

    binary_digest = _sha256_file(binary)
    if binary_digest != str(model.runtime_digest):
        raise RuntimeErrorWithContext(
            f"llama-server SHA-256 mismatch: expected {model.runtime_digest}, "
            f"got {binary_digest}"
        )

    repository = (
        _huggingface_root(model, workspace)
        / "hub"
        / ("models--" + str(model.source).replace("/", "--"))
    )
    snapshot = repository / "snapshots" / str(model.revision)
    try:
        repository_resolved = repository.resolve(strict=True)
        snapshot.resolve(strict=True).relative_to(repository_resolved)
    except (OSError, ValueError) as error:
        raise RuntimeErrorWithContext(
            f"Exact cached GGUF snapshot is missing or escapes its repository: {snapshot}"
        ) from error
    configured_shards = tuple(getattr(model, "model_shards", ()))
    model_shards: list[dict[str, Any]] = []
    if configured_shards:
        _validate_runtime_model_shards(configured_shards)
        for index, shard in enumerate(configured_shards, start=1):
            artifact = _validate_llamacpp_snapshot_artifact(
                repository_resolved=repository_resolved,
                snapshot=snapshot,
                filename=str(shard.path),
                expected_size=int(shard.size_bytes),
                expected_digest=str(shard.digest),
                label=f"GGUF shard {index}/{len(configured_shards)}",
            )
            model_shards.append(
                {
                    "relative_path": str(shard.path),
                    **artifact,
                }
            )
        model_artifact = model_shards[0]
    else:
        model_artifact = _validate_llamacpp_snapshot_artifact(
            repository_resolved=repository_resolved,
            snapshot=snapshot,
            filename=str(model.model_file),
            expected_size=int(model.model_size_bytes),
            expected_digest=str(model.model_digest),
            label="GGUF",
        )
    result = {
        "runtime_binary": str(binary),
        "runtime_binary_sha256": binary_digest,
        "runtime_binary_size_bytes": binary.stat().st_size,
        "runtime_source_dir": str(source_dir),
        "runtime_source_revision": revision.stdout.strip(),
        "model_path": model_artifact["path"],
        "model_target": model_artifact["target"],
        "model_sha256": model_artifact["sha256"],
        "model_size_bytes": model_artifact["size_bytes"],
        "model_source": str(model.source),
        "model_revision": str(model.revision),
    }
    if model_shards:
        result.update(
            {
                "model_shards": model_shards,
                "model_shard_count": len(model_shards),
                "model_total_size_bytes": sum(
                    int(shard["size_bytes"]) for shard in model_shards
                ),
            }
        )
    mmproj_file = getattr(model, "mmproj_file", None)
    if mmproj_file is not None:
        mmproj_artifact = _validate_llamacpp_snapshot_artifact(
            repository_resolved=repository_resolved,
            snapshot=snapshot,
            filename=str(mmproj_file),
            expected_size=int(model.mmproj_size_bytes),
            expected_digest=str(model.mmproj_digest),
            label="mmproj GGUF",
        )
        result.update(
            {
                "mmproj_path": mmproj_artifact["path"],
                "mmproj_target": mmproj_artifact["target"],
                "mmproj_sha256": mmproj_artifact["sha256"],
                "mmproj_size_bytes": mmproj_artifact["size_bytes"],
                "multimodal": True,
            }
        )
    draft_model_file = getattr(model, "draft_model_file", None)
    if draft_model_file is not None:
        draft_source = str(model.draft_source)
        draft_revision = str(model.draft_revision)
        draft_repository = (
            _huggingface_root(model, workspace)
            / "hub"
            / ("models--" + draft_source.replace("/", "--"))
        )
        draft_snapshot = draft_repository / "snapshots" / draft_revision
        try:
            draft_repository_resolved = draft_repository.resolve(strict=True)
            draft_snapshot.resolve(strict=True).relative_to(
                draft_repository_resolved
            )
        except (OSError, ValueError) as error:
            raise RuntimeErrorWithContext(
                "Exact cached draft GGUF snapshot is missing or escapes its "
                f"repository: {draft_snapshot}"
            ) from error
        draft_artifact = _validate_llamacpp_snapshot_artifact(
            repository_resolved=draft_repository_resolved,
            snapshot=draft_snapshot,
            filename=str(draft_model_file),
            expected_size=int(model.draft_model_size_bytes),
            expected_digest=str(model.draft_model_digest),
            label="draft GGUF",
        )
        result.update(
            {
                "draft_model_path": draft_artifact["path"],
                "draft_model_target": draft_artifact["target"],
                "draft_model_sha256": draft_artifact["sha256"],
                "draft_model_size_bytes": draft_artifact["size_bytes"],
                "draft_model_source": draft_source,
                "draft_model_revision": draft_revision,
            }
        )
    return result


def _native_server_id(run_identity: str) -> str:
    return hashlib.sha256(("llamacpp:" + run_identity).encode()).hexdigest()[:32]


def _native_environment(run_identity: str, server_id: str) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in _NATIVE_ENV_ALLOWLIST
    }
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "LLAMA_ARG_OFFLINE": "1",
            "SPARKBENCH_RUN_ID": run_identity,
            "SPARKBENCH_SERVER_ID": server_id,
        }
    )
    return environment


def _native_command(
    model: Any, artifacts: dict[str, Any], *, port: int
) -> list[str]:
    parallel = int(model.runtime_parallel)
    total_context = int(model.max_context) * parallel
    command = [
        str(artifacts["runtime_binary"]),
        "--model",
        str(artifacts["model_path"]),
    ]
    if artifacts.get("mmproj_path"):
        command.extend(("--mmproj", str(artifacts["mmproj_path"])))
    if artifacts.get("draft_model_path"):
        command.extend(
            ("--spec-draft-model", str(artifacts["draft_model_path"]))
        )
    command.extend(
        [
            "--alias",
            str(model.served_name),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(total_context),
            "--parallel",
            str(parallel),
            "--no-ui",
            "--offline",
            "--metrics",
            "--cors-origins",
            "localhost",
            "--no-cors-credentials",
            *(str(argument) for argument in model.args),
        ]
    )
    return command


def _proc_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text()
    fields = raw[raw.rfind(")") + 2 :].split()
    if len(fields) <= 19:
        raise OSError(f"short /proc/{pid}/stat")
    return int(fields[19])


def _proc_uid(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("Uid:"):
            return int(line.split()[1])
    raise OSError(f"missing Uid in /proc/{pid}/status")


def _proc_environment(pid: int) -> set[bytes]:
    return set(Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"))


def _proc_argv(pid: int) -> list[str]:
    return [
        field.decode(errors="surrogateescape")
        for field in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if field
    ]


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeErrorWithContext(f"Refusing symlink process state path: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _write_private_secret(path: Path, value: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeErrorWithContext(
            f"Refusing to overwrite existing API-key path: {path}"
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            stream.write(value + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _unlink_private_secret(path: Path | None) -> None:
    if path is None:
        return
    path.unlink(missing_ok=True)
    if path.parent.is_dir():
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _unlink_state(path: Path) -> None:
    path.unlink(missing_ok=True)
    if path.parent.is_dir():
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _read_native_state(path: Path, run_identity: str) -> dict[str, Any]:
    if path.is_symlink():
        raise RuntimeErrorWithContext(f"Refusing symlink process state path: {path}")
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeErrorWithContext(
            f"Could not read native process ownership state: {path}"
        ) from error
    if not isinstance(state, dict) or state.get("run_identity") != run_identity:
        raise RuntimeErrorWithContext(
            "Native process state does not belong to this frozen run"
        )
    return state


def _native_process_is_owned(state: dict[str, Any]) -> bool:
    pid = int(state["pid"])
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return False
    try:
        actual = {
            "uid": _proc_uid(pid),
            "start_ticks": _proc_start_ticks(pid),
            "pgid": os.getpgid(pid),
            "argv": _proc_argv(pid),
            "runtime_binary": str(Path(f"/proc/{pid}/exe").resolve(strict=True)),
            "runtime_digest": _sha256_file(Path(f"/proc/{pid}/exe")),
            "environment": _proc_environment(pid),
        }
    except (OSError, ValueError) as error:
        if not proc.exists():
            return False
        raise RuntimeErrorWithContext(
            f"Could not prove ownership of native process {pid}"
        ) from error
    markers = {
        f"SPARKBENCH_RUN_ID={state['run_identity']}".encode(),
        f"SPARKBENCH_SERVER_ID={state['server_id']}".encode(),
    }
    expected = {
        "uid": int(state["uid"]),
        "start_ticks": int(state["start_ticks"]),
        "pgid": int(state["pgid"]),
        "argv": [str(item) for item in state["argv"]],
        "runtime_binary": str(state["runtime_binary"]),
        "runtime_digest": str(state["runtime_digest"]),
    }
    if (
        actual["uid"] != os.geteuid()
        or any(actual[key] != value for key, value in expected.items())
        or not markers.issubset(actual["environment"])
    ):
        raise RuntimeErrorWithContext(
            f"Refusing to signal native PID {pid}: process identity does not match ownership state"
        )
    return True


def _native_process_still_alive(pid: int, start_ticks: int) -> bool:
    if not Path(f"/proc/{pid}").exists():
        return False
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2 :].split()
        return (
            len(fields) > 19
            and fields[0] != "Z"
            and int(fields[19]) == start_ticks
        )
    except OSError:
        return False


def _signal_pidfd(pidfd: int | None, pid: int, sig: signal.Signals) -> None:
    if pidfd is not None and hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(pidfd, sig)
    else:
        os.kill(pid, sig)


def _terminate_owned_native_state(
    state: dict[str, Any], *, timeout_s: float = 30.0
) -> None:
    if not _native_process_is_owned(state):
        return
    pid = int(state["pid"])
    start_ticks = int(state["start_ticks"])
    pidfd = os.pidfd_open(pid) if hasattr(os, "pidfd_open") else None
    try:
        if not _native_process_is_owned(state):
            return
        try:
            _signal_pidfd(pidfd, pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not _native_process_still_alive(pid, start_ticks):
                return
            time.sleep(0.1)
        if _native_process_still_alive(pid, start_ticks):
            _signal_pidfd(pidfd, pid, signal.SIGKILL)
            kill_deadline = time.monotonic() + 5
            while time.monotonic() < kill_deadline:
                if not _native_process_still_alive(pid, start_ticks):
                    return
                time.sleep(0.05)
            raise RuntimeErrorWithContext(
                f"Owned llama-server PID {pid} did not exit after SIGKILL"
            )
    finally:
        if pidfd is not None:
            os.close(pidfd)


def _stop_native_state(path: Path, run_identity: str) -> str:
    if not path.is_file():
        return "already_absent"
    state = _read_native_state(path, run_identity)
    _terminate_owned_native_state(state)
    _unlink_state(path)
    return "stopped_owned_process"


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _existing_container(
    container_name: str = CONTAINER_NAME,
) -> tuple[str, bool, str] | None:
    result = _run(
        [
            "docker", "ps", "-a", "--filter", f"name=^{container_name}$", "--format",
            "{{.ID}} {{.Label \"ai.sparkbench.managed\"}} {{.Label \"ai.sparkbench.run\"}}",
        ],
        check=False,
        timeout=20,
    )
    if result.returncode:
        raise RuntimeErrorWithContext(
            "Could not inspect SparkBench container state: "
            + (result.stderr.strip() or "docker ps failed")
        )
    if not result.stdout.strip():
        return None
    fields = result.stdout.strip().split(maxsplit=2)
    container_id = fields[0]
    managed = len(fields) > 1 and fields[1] == "true"
    run_identity = fields[2] if len(fields) > 2 else ""
    return container_id, managed, run_identity


@dataclass
class ManagedServer:
    backend: str
    base_url: str
    container_id: str | None = None
    ollama_model: str | None = None
    startup_s: float | None = None
    run_identity: str | None = None
    unload_ollama: bool = False
    process: subprocess.Popen[Any] | None = None
    process_state_path: Path | None = None
    process_log: TextIO | None = None
    native_provenance: dict[str, Any] | None = None
    authorization: str | None = None
    api_key: str | None = None
    api_key_path: Path | None = None
    _lifecycle_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )
    _immediate_stop_complete: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )

    def _require_owned_container(self) -> None:
        if not self.container_id or not self.run_identity:
            raise RuntimeErrorWithContext(
                "Managed container has no lifecycle ownership state"
            )
        inspect = _run(
            [
                "docker", "inspect", "--format",
                "{{index .Config.Labels \"ai.sparkbench.managed\"}} "
                "{{index .Config.Labels \"ai.sparkbench.run\"}}",
                self.container_id,
            ],
            check=False,
            timeout=20,
        )
        labels = inspect.stdout.strip().split(maxsplit=1)
        owned = (
            inspect.returncode == 0
            and labels
            and labels[0] == "true"
            and len(labels) == 2
            and labels[1] == self.run_identity
        )
        if not owned:
            raise RuntimeErrorWithContext(
                "Refusing to stop a container not owned by SparkBench"
            )

    def _require_live_owned_loopback_port(
        self, *, host_port: int, container_port: int
    ) -> tuple[str, int]:
        """Require the exact owned SGLang container to own one loopback port.

        This is deliberately narrower than normal lifecycle ownership.  A
        runtime attestation needs to reject a stopped owned container paired
        with an unrelated listener on the expected host port.
        """

        if (
            type(host_port) is not int
            or type(container_port) is not int
            or not 1 <= host_port <= 65535
            or not 1 <= container_port <= 65535
            or type(self.container_id) is not str
        ):
            raise RuntimeErrorWithContext(
                "Managed container does not have a valid loopback binding"
            )
        ManagedServer._require_owned_container(self)
        port_binding_template = (
            "{{.State.Running}} {{.State.StartedAt}} {{.State.Pid}} "
            "{{range $binding := (index .NetworkSettings.Ports \""
            + str(container_port)
            + "/tcp\")}}{{$binding.HostIP}}:{{$binding.HostPort}};{{end}}"
        )
        try:
            inspect = _run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    port_binding_template,
                    self.container_id,
                ],
                check=False,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError, UnicodeError):
            raise RuntimeErrorWithContext(
                "Managed container is not live on the required loopback port"
            ) from None
        binding = re.fullmatch(
            rf"true (?P<started_at>{_SM121_AGENT_RUNTIME_STARTED_AT_PATTERN}) "
            rf"(?P<pid>[1-9]\d*) "
            rf"127\.0\.0\.1:{host_port};\n",
            inspect.stdout if type(inspect.stdout) is str else "",
        )
        if (
            inspect.returncode != 0
            or type(inspect.stderr) is not str
            or inspect.stderr
            or binding is None
        ):
            raise RuntimeErrorWithContext(
                "Managed container is not live on the required loopback port"
            )
        return binding["started_at"], int(binding["pid"])

    def interrupt_owned(self) -> None:
        """Immediately stop an exact owned SGLang container without removing it."""

        if self.backend != "sglang":
            raise RuntimeErrorWithContext(
                "Immediate host-safety interruption is supported only for SGLang"
            )
        with self._lifecycle_lock:
            if not self.container_id:
                return
            if self._immediate_stop_complete:
                return
            self._require_owned_container()
            _run(
                ["docker", "stop", "--time", "0", self.container_id],
                check=True,
                timeout=15,
            )
            self._immediate_stop_complete = True

    def stop(self, *, keep_server: bool = False) -> None:
        with self._lifecycle_lock:
            self._stop_locked(keep_server=keep_server)

    def _stop_locked(self, *, keep_server: bool = False) -> None:
        if self.backend == LLAMACPP_BACKEND:
            if keep_server:
                raise RuntimeErrorWithContext(
                    "--keep-server is not supported for managed llama.cpp processes"
                )
            try:
                if not self.process_state_path or not self.run_identity:
                    raise RuntimeErrorWithContext(
                        "Managed llama.cpp server has no ownership state"
                    )
                _stop_native_state(self.process_state_path, self.run_identity)
                if self.process is not None:
                    try:
                        self.process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                self.process = None
            finally:
                if self.process_log is not None:
                    self.process_log.flush()
                    self.process_log.close()
                    self.process_log = None
            return
        if (
            self.backend in {"sglang", "vllm"}
            and self.container_id
            and not keep_server
        ):
            self._require_owned_container()
            if not self._immediate_stop_complete:
                _run(
                    ["docker", "stop", "--time", "30", self.container_id],
                    check=True,
                    timeout=45,
                )
            _run(["docker", "rm", self.container_id], check=True, timeout=15)
            self.container_id = None
            self._immediate_stop_complete = False
            if self.backend == "sglang":
                _unlink_private_secret(self.api_key_path)
                self.api_key_path = None
                self.api_key = None
                self.authorization = None
        elif (
            self.backend == "ollama"
            and self.ollama_model
            and self.unload_ollama
            and not keep_server
        ):
            try:
                _run(["ollama", "stop", self.ollama_model], check=False, timeout=60)
            except subprocess.TimeoutExpired:
                pass
            deadline = time.monotonic() + 75
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                try:
                    if not _ollama_model_loaded(self.base_url, self.ollama_model):
                        return
                except RuntimeErrorWithContext as error:
                    last_error = error
                time.sleep(0.5)
            detail = f": {last_error}" if last_error else ""
            raise RuntimeErrorWithContext(
                f"Ollama model {self.ollama_model} did not unload within 75 seconds{detail}"
            )


def recover_owned_vllm(run_identity: str) -> str:
    """Stop only the exact vLLM container labeled for a crashed run.

    The return value describes whether an owned container was stopped, was
    already absent, or had been replaced by a differently owned container.
    """

    existing = _existing_container()
    if existing is None:
        return "already_absent"
    container_id, managed, existing_run = existing
    if not managed or existing_run != run_identity:
        return "different_container_present"
    ManagedServer(
        backend="vllm",
        base_url="http://127.0.0.1:8000/v1",
        container_id=container_id,
        run_identity=run_identity,
    ).stop()
    return "stopped_owned_container"


def recover_owned_sglang(
    run_identity: str, *, api_key_path: Path | None = None
) -> str:
    """Stop only the exact SGLang container labeled for a crashed run."""

    existing = _existing_container(SGLANG_CONTAINER_NAME)
    if existing is None:
        _unlink_private_secret(api_key_path)
        return "already_absent"
    container_id, managed, existing_run = existing
    if not managed or existing_run != run_identity:
        return "different_container_present"
    ManagedServer(
        backend="sglang",
        base_url="http://127.0.0.1:30000/v1",
        container_id=container_id,
        run_identity=run_identity,
        api_key_path=api_key_path,
    ).stop()
    return "stopped_owned_container"


def _exact_sglang_snapshot(
    hf_cache: Path, *, source: str, revision: str, role: str
) -> tuple[Path, str]:
    if (
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise RuntimeErrorWithContext(
            f"SGLang {role} revision must be a full lowercase commit SHA"
        )
    repository_name = "models--" + source.replace("/", "--")
    hub_root = hf_cache / "hub"
    repository = hub_root / repository_name
    snapshot = repository / "snapshots" / revision
    if not snapshot.is_dir():
        raise RuntimeErrorWithContext(
            f"Exact SGLang {role} snapshot is not cached: {source}@{revision}"
        )
    try:
        hub_resolved = hub_root.resolve(strict=True)
        repository_resolved = repository.resolve(strict=True)
        repository_resolved.relative_to(hub_resolved)
        snapshot.resolve(strict=True).relative_to(repository_resolved)
    except (OSError, ValueError) as error:
        raise RuntimeErrorWithContext(
            f"Exact SGLang {role} snapshot escapes its cache repository"
        ) from error
    container_snapshot = (
        f"/root/.cache/huggingface/hub/{repository_name}/snapshots/{revision}"
    )
    return repository_resolved, container_snapshot


def _sm121_storage_image_id(model: Any) -> str:
    """Recheck the mutable local tag and return its immutable Docker ID."""

    image = str(getattr(model, "image", "") or "")
    try:
        result = _run(
            ["docker", "image", "inspect", image, "--format", "{{json .}}"],
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeErrorWithContext(
            "Could not inspect the local SM121 storage image"
        ) from error
    if result.returncode:
        raise RuntimeErrorWithContext(
            "Could not inspect the local SM121 storage image"
        )
    try:
        inspection = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeErrorWithContext(
            "Local SM121 storage image inspection was not valid JSON"
        ) from error
    if not isinstance(inspection, dict):
        raise RuntimeErrorWithContext(
            "Local SM121 storage image inspection was not an object"
        )
    try:
        identity = validate_sm121_storage_image_inspection(
            inspection, image=image
        )
    except SM121StorageCandidateError as error:
        raise RuntimeErrorWithContext(str(error)) from error
    image_id = identity["docker_image_id"]
    frozen = getattr(model, "resolved_local_image_id", None)
    if frozen is None:
        frozen = getattr(model, "resolved_image", None)
    if frozen is not None and frozen != image_id:
        raise RuntimeErrorWithContext(
            "Frozen local SM121 storage image ID differs from the inspected tag"
        )
    return image_id


def _sm121_storage_seccomp_profile(workspace: Path) -> Path:
    """Return the contract-verified, per-container io_uring profile path."""

    try:
        verification = verify_seccomp_profile_contract(workspace)
    except (OSError, SeccompProfileContractError) as error:
        raise RuntimeErrorWithContext(
            "The SM121 storage seccomp profile did not verify"
        ) from error
    if verification.derived_sha256 != SM121_STORAGE_SECCOMP_SHA256:
        raise RuntimeErrorWithContext(
            "The SM121 storage seccomp profile hash did not match its pin"
        )
    try:
        root = workspace.resolve(strict=True)
        profile = (root / SM121_STORAGE_SECCOMP_PATH).resolve(strict=True)
        profile.relative_to(root)
    except (OSError, ValueError) as error:
        raise RuntimeErrorWithContext(
            "The SM121 storage seccomp profile path is unavailable or unsafe"
        ) from error
    if not profile.is_file():
        raise RuntimeErrorWithContext(
            "The SM121 storage seccomp profile is not a regular file"
        )
    return profile


def _start_sglang_sm121_storage(
    model: Any,
    *,
    workspace: Path,
    target_repository: Path,
    container_snapshot: str,
    port: int,
    server_log_path: Path | None,
    abort_check: Callable[[], None] | None,
    on_server_created: Callable[[ManagedServer], None] | None,
) -> ManagedServer:
    """Start the dedicated native-NVMe PLE canary with narrow containment."""

    semantic_candidate = is_sm121_cache_semantic_candidate(model)
    performance_candidate = is_sm121_cache_performance_candidate(model)
    chunked_prefill_candidate = is_sm121_chunked_prefill_performance_candidate(
        model
    )
    authorized = (
        getattr(model, "cache_semantic_canary_authorized", False) is True
        if semantic_candidate
        else getattr(model, "cache_performance_authorized", False) is True
        if performance_candidate
        else getattr(model, "chunked_prefill_performance_authorized", False)
        is True
        if chunked_prefill_candidate
        else getattr(model, "storage_canary_authorized", False) is True
    )
    if not authorized:
        raise RuntimeErrorWithContext(
            "SM121 storage serving requires its dedicated canary executor"
        )
    try:
        if semantic_candidate:
            validate_sm121_cache_semantic_candidate(model)
        elif performance_candidate:
            validate_sm121_cache_performance_candidate(model)
        elif chunked_prefill_candidate:
            validate_sm121_chunked_prefill_performance_candidate(model)
        else:
            validate_sm121_storage_candidate(model)
    except (
        SM121StorageCandidateError,
        SM121CacheSemanticError,
        SM121CachePerformanceError,
        SM121ChunkedPrefillPerformanceError,
    ) as error:
        raise RuntimeErrorWithContext(str(error)) from error
    if abort_check is not None:
        abort_check()
    image_id = _sm121_storage_image_id(model)
    seccomp_profile = _sm121_storage_seccomp_profile(workspace)
    run_identity = str(getattr(model, "run_identity", "unknown"))
    key = secrets.token_urlsafe(32)
    auth = f"Bearer {key}"
    sensitive_values = (key, auth)
    api_key_path = (
        server_log_path.parent / "api-key"
        if server_log_path is not None
        else None
    )
    if api_key_path is not None:
        _write_private_secret(api_key_path, key)

    command = [
        "docker",
        "run",
        "--detach",
        "--pull=never",
        "--name",
        SGLANG_CONTAINER_NAME,
        "--label",
        MANAGED_LABEL,
        "--label",
        f"ai.sparkbench.run={run_identity}",
        "--label",
        "ai.sparkbench.backend=sglang",
        "--gpus",
        "all",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--security-opt",
        f"seccomp={seccomp_profile}",
        "--shm-size",
        "16g",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,nodev,size=16g",
        "--tmpfs",
        "/root/.cache:rw,exec,nosuid,nodev,size=8g",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "--publish",
        f"127.0.0.1:{port}:30000",
        "--volume",
        f"{target_repository}:/root/.cache/huggingface/hub/"
        f"{target_repository.name}:ro",
        "--env",
        "HF_HOME=/tmp/sparkbench-hf",
        "--env",
        "HF_HUB_CACHE=/tmp/sparkbench-hf/hub",
        "--env",
        "HF_MODULES_CACHE=/tmp/sparkbench-hf/modules",
        "--env",
        "HF_TOKEN_PATH=/tmp/sparkbench-hf/token-disabled",
        "--env",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN=1",
        "--env",
        "HF_HUB_DISABLE_TELEMETRY=1",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        "SGLANG_RUST_BUILD_MODE=never",
        "--env",
        f"SGLANG_QWEN4_PLE_NVME_PATH={container_snapshot}",
        "--env",
        "SGLANG_QWEN4_PLE_NVME_BACKEND=io_uring",
        "--env",
        f"SGLANG_QWEN4_PLE_NVME_QUEUE_DEPTH={SM121_STORAGE_QUEUE_DEPTH}",
        "--env",
        "SGLANG_QWEN4_PLE_NVME_MAX_BATCH_PAGES="
        f"{SM121_STORAGE_MAX_BATCH_PAGES}",
        "--env",
        f"SGLANG_QWEN4_PLE_NVME_CACHE_PAGES={SM121_STORAGE_CACHE_PAGES}",
        "--env",
        "SGLANG_CACHE_DIR=/tmp/sglang-cache",
        "--env",
        "TRITON_CACHE_DIR=/tmp/triton-cache",
        "--env",
        "XDG_CACHE_HOME=/tmp/xdg-cache",
        "--env",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-cache",
        "--env",
        "TILELANG_CACHE_DIR=/tmp/tilelang-cache",
        "--entrypoint",
        "sglang",
        image_id,
        "serve",
        "--model-path",
        container_snapshot,
        "--api-key=" + key,
        *(str(argument) for argument in model.args),
    ]
    if abort_check is not None:
        try:
            abort_check()
        except BaseException:
            _unlink_private_secret(api_key_path)
            raise
    try:
        result = _run(command, check=False, timeout=60)
    except BaseException as launch_error:
        _unlink_private_secret(api_key_path)
        raise RuntimeErrorWithContext(
            "docker run failed: "
            + _redact_text(str(launch_error), sensitive_values)
        ) from None
    if result.returncode:
        _unlink_private_secret(api_key_path)
        raise RuntimeErrorWithContext(
            "docker run failed: "
            + _redact_text(result.stderr.strip(), sensitive_values)
        )
    container_id = result.stdout.strip()
    if not container_id:
        _unlink_private_secret(api_key_path)
        raise RuntimeErrorWithContext(
            "docker run returned no authenticated SGLang container ID"
        )
    server = ManagedServer(
        "sglang",
        f"http://127.0.0.1:{port}/v1",
        container_id=container_id,
        run_identity=run_identity,
        authorization=auth,
        api_key=key,
        api_key_path=api_key_path,
    )
    server.native_provenance = {
        "candidate_id": (
            str(getattr(model, "id", ""))
            if (
                semantic_candidate
                or performance_candidate
                or chunked_prefill_candidate
            )
            else SM121_STORAGE_CANDIDATE_ID
        ),
        "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
        "build_contract_sha256": SM121_STORAGE_BUILD_CONTRACT_SHA256,
        "docker_image_id": image_id,
        "sglang_storage_mode": SM121_STORAGE_MODE,
        "sglang_ple_nvme_backend": "io_uring",
        "sglang_ple_nvme_queue_depth": SM121_STORAGE_QUEUE_DEPTH,
        "sglang_ple_nvme_max_batch_pages": SM121_STORAGE_MAX_BATCH_PAGES,
        "sglang_ple_nvme_cache_pages": SM121_STORAGE_CACHE_PAGES,
        "sglang_rust_build_mode": "never",
        "seccomp_profile_sha256": "sha256:" + SM121_STORAGE_SECCOMP_SHA256,
        "container_rootfs": "readonly_tmpfs_writable_cache",
        "container_capabilities": "dropped_all",
        "container_no_new_privileges": True,
        "hf_network_policy": "offline",
        "network_topology": "loopback_published_bridge",
        "benchmark_scope": (
            SM121_CACHE_SEMANTIC_EXECUTION_MODE
            if semantic_candidate
            else SM121_CACHE_PERFORMANCE_EXECUTION_MODE
            if performance_candidate
            else "sm121_storage_pre_admission_canary"
        ),
        "model_acquisition": "disabled_exact_read_only_snapshot",
        "api_authentication": "ephemeral_bearer",
        "api_key_file_mode": "0600" if api_key_path is not None else None,
    }
    try:
        if on_server_created is not None:
            on_server_created(server)
        if abort_check is not None:
            abort_check()
        wait_arguments: dict[str, Any] = {
            "authorization": auth,
            "sensitive_values": sensitive_values,
        }
        if abort_check is not None:
            wait_arguments["abort_check"] = abort_check
        server.startup_s = wait_for_endpoint(
            server.base_url,
            float(model.startup_timeout_s),
            container_id,
            **wait_arguments,
        )
    except BaseException as startup_error:
        if server_log_path is not None:
            try:
                save_server_logs(server, server_log_path)
            except Exception as log_error:
                _redact_exception(log_error, sensitive_values)
                startup_error.add_note(
                    "Could not persist full startup container logs: "
                    f"{type(log_error).__name__}: {log_error}"
                )
        try:
            server.stop()
        except BaseException as cleanup_error:
            _redact_exception(cleanup_error, sensitive_values)
            startup_error.add_note(
                "Could not clean up authenticated SGLang server: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        _redact_exception(startup_error, sensitive_values)
        raise startup_error
    return server


# The B0 source check is intentionally a read-only, no-network, no-GPU
# container invocation.  It binds the reviewed cache-selection and response
# accounting files to the same local image ID used for serving, without
# retaining source text or a container filesystem path in the result journal.
_SM121_CACHE_SOURCE_FILES = {
    "arg_overrides_sha256": "python/sglang/srt/arg_groups/overrides.py",
    "cache_registry_sha256": "python/sglang/srt/mem_cache/registry.py",
    "cache_builder_sha256": "python/sglang/srt/mem_cache/kv_cache_builder.py",
    "runtime_context_sha256": "python/sglang/srt/runtime_context.py",
    "metrics_collector_sha256": "python/sglang/srt/observability/metrics_collector.py",
    "openai_utils_sha256": "python/sglang/srt/entrypoints/openai/utils.py",
    "openai_protocol_sha256": "python/sglang/srt/entrypoints/openai/protocol.py",
    "openai_serving_chat_sha256": (
        "python/sglang/srt/entrypoints/openai/serving_chat.py"
    ),
    "openai_usage_processor_sha256": (
        "python/sglang/srt/entrypoints/openai/usage_processor.py"
    ),
    "http_server_sha256": "python/sglang/srt/entrypoints/http_server.py",
}
_SM121_CACHE_STARTUP_RE = re.compile(
    r"Tree cache initialized: source=(?P<cache_source>[a-z_]+) "
    r"impl=(?P<cache_impl>[A-Za-z0-9_]+) "
    r"hybrid_swa=(?P<hybrid_swa>True|False) "
    r"hybrid_ssm=(?P<hybrid_ssm>True|False) "
    r"hicache_attached=(?P<hicache_attached>True|False) "
    r"streaming_wrapped=(?P<streaming_wrapped>True|False)"
)
_SM121_CACHE_METRIC_LINE_RE = re.compile(
    r"^(?P<name>sglang:[A-Za-z0-9_:]+)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
    r"(?:\s+[0-9]+)?$"
)
_SM121_CACHE_METRIC_TYPE_RE = re.compile(
    r"^# TYPE (?P<name>sglang:[A-Za-z0-9_:]+) (?P<metric_type>[A-Za-z_]+)$"
)
_SM121_CACHE_LABEL_RE = re.compile(
    r"(?:^|,)(?P<name>[A-Za-z_][A-Za-z0-9_]*)=\"(?P<value>(?:\\.|[^\"])*)\""
)
_SM121_CACHE_GUARDRAIL_SAMPLES = {
    "sglang:evicted_tokens_total": "evicted_tokens",
    "sglang:num_retracted_requests_total": "retracted_requests",
}
_SM121_CACHE_GUARDRAIL_FAMILY_PREFIXES = {
    "sglang:evicted_tokens_total": "sglang:evicted_tokens",
    "sglang:num_retracted_requests_total": "sglang:num_retracted_requests",
}
_SM121_CACHE_CACHE_ON_EVICTION_LABELS = {"cache_type": "UnifiedRadixCache"}
_SM121_CACHE_SCHEDULER_LABEL_FIELDS = frozenset(
    {"model_name", "engine_type", "tp_rank", "pp_rank", "moe_ep_rank"}
)
_SM121_CACHE_TOKENIZER_LABEL_FIELDS = frozenset({"model_name", "engine_type"})
_SM121_AGENT_RUNTIME_SERVER_INFO_MAX_BYTES = 2 * 1024 * 1024
_SM121_AGENT_RUNTIME_LOG_MAX_BYTES = 2 * 1024 * 1024
_SM121_AGENT_RUNTIME_LOG_TIMEOUT_S = 30.0
_SM121_AGENT_RUNTIME_LOG_MAX_LINES = 4096
_SM121_AGENT_RUNTIME_RESPONSE_TIMEOUT_S = 15.0
_SM121_AGENT_RUNTIME_RESPONSE_CHUNK_BYTES = 64 * 1024
_SM121_AGENT_RUNTIME_STARTED_AT_PATTERN = (
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z"
)


@dataclass(frozen=True)
class _SM121AgentRuntimeBinding:
    """Private immutable values that bind both C1 runtime observations."""

    container_id: str = field(repr=False)
    run_identity: str = field(repr=False)
    authorization: str = field(repr=False)
    api_key: str = field(repr=False)
    generation: tuple[str, int] = field(repr=False)


class _SM121AgentRuntimeNoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects for the fixed loopback C1 runtime read."""

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise urllib.error.URLError("SM121 agent runtime redirects are denied")


def _sm121_agent_runtime_unique_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _sm121_agent_runtime_reject_constant(_value: str) -> None:
    raise ValueError("non-standard JSON constant")


def _sm121_agent_runtime_finite_float(value: str) -> float:
    """Reject finite-looking JSON exponents that overflow Python ``float``."""

    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _require_sm121_agent_admission_server(
    server: object,
) -> tuple[ManagedServer, _SM121AgentRuntimeBinding]:
    """Require the exact managed endpoint/auth identity before any C1 read."""

    if type(server) is not ManagedServer:
        raise RuntimeErrorWithContext("SM121 agent runtime is not an owned server")
    key = server.api_key
    if (
        type(server.backend) is not str
        or server.backend != "sglang"
        or type(server.base_url) is not str
        or server.base_url != SM121_AGENT_ADMISSION_ENDPOINT
        or type(server.container_id) is not str
        or not server.container_id
        or type(server.run_identity) is not str
        or not server.run_identity
        or type(key) is not str
        or re.fullmatch(r"[A-Za-z0-9_-]{16,512}", key) is None
        or type(server.authorization) is not str
        or server.authorization != "Bearer " + key
    ):
        raise RuntimeErrorWithContext("SM121 agent runtime is not an owned server")
    try:
        generation = ManagedServer._require_live_owned_loopback_port(
            server,
            host_port=30000,
            container_port=30000,
        )
    except (
        OSError,
        RuntimeErrorWithContext,
        subprocess.SubprocessError,
        UnicodeError,
        ValueError,
    ):
        raise RuntimeErrorWithContext(
            "SM121 agent runtime is not an owned server"
        ) from None
    if (
        type(generation) is not tuple
        or len(generation) != 2
        or type(generation[0]) is not str
        or re.fullmatch(_SM121_AGENT_RUNTIME_STARTED_AT_PATTERN, generation[0])
        is None
        or type(generation[1]) is not int
        or generation[1] <= 0
    ):
        raise RuntimeErrorWithContext("SM121 agent runtime is not an owned server")
    return server, _SM121AgentRuntimeBinding(
        server.container_id,
        server.run_identity,
        server.authorization,
        key,
        generation,
    )


def _open_sm121_agent_runtime_server_info(
    request: urllib.request.Request, *, timeout_s: float
) -> Any:
    """Open one no-proxy/no-redirect C1 server-info request."""

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _SM121AgentRuntimeNoRedirect(),
    )
    return opener.open(request, timeout=timeout_s)


def _sm121_agent_runtime_set_response_timeout(
    response: object, timeout_s: float
) -> None:
    """Set the remaining C1 deadline on urllib's direct response socket."""

    try:
        sock = response.fp.raw._sock  # type: ignore[attr-defined]
        settimeout = sock.settimeout
    except AttributeError:
        raise RuntimeErrorWithContext(
            "SM121 agent runtime attestation is invalid"
        ) from None
    if not callable(settimeout):
        raise RuntimeErrorWithContext("SM121 agent runtime attestation is invalid")
    try:
        settimeout(timeout_s)
    except (OSError, ValueError):
        raise RuntimeErrorWithContext(
            "Could not read the SM121 agent runtime attestation"
        ) from None


def _read_sm121_agent_runtime_response_body(response: object) -> bytes:
    """Read a bounded response with one hard total deadline, not per I/O."""

    reader = getattr(response, "read1", None)
    if not callable(reader):
        raise RuntimeErrorWithContext("SM121 agent runtime attestation is invalid")
    payload = bytearray()
    deadline = time.monotonic() + _SM121_AGENT_RUNTIME_RESPONSE_TIMEOUT_S
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeErrorWithContext(
                "Could not read the SM121 agent runtime attestation"
            )
        _sm121_agent_runtime_set_response_timeout(response, remaining)
        try:
            chunk = reader(
                min(
                    _SM121_AGENT_RUNTIME_RESPONSE_CHUNK_BYTES,
                    _SM121_AGENT_RUNTIME_SERVER_INFO_MAX_BYTES + 1 - len(payload),
                )
            )
        except (OSError, TimeoutError, ValueError, http.client.HTTPException):
            raise RuntimeErrorWithContext(
                "Could not read the SM121 agent runtime attestation"
            ) from None
        if type(chunk) is not bytes:
            raise RuntimeErrorWithContext("SM121 agent runtime attestation is invalid")
        if not chunk:
            return bytes(payload)
        payload.extend(chunk)
        if len(payload) > _SM121_AGENT_RUNTIME_SERVER_INFO_MAX_BYTES:
            raise RuntimeErrorWithContext(
                "SM121 agent runtime attestation is invalid"
            )
        # ``http.client.HTTPResponse.read1`` closes ``fp`` after consuming the
        # final Content-Length byte.  Return before the next socket-timeout
        # update in that normal success path.
        if getattr(response, "length", None) == 0:
            return bytes(payload)


def _read_sm121_agent_runtime_server_info(authorization: str) -> dict[str, object]:
    """Read one bounded, strict, in-memory C1 ``/server_info`` object."""

    request = urllib.request.Request(
        SM121_AGENT_ADMISSION_ENDPOINT.removesuffix("/v1") + "/server_info",
        headers={"Authorization": authorization},
        method="GET",
    )
    try:
        with _open_sm121_agent_runtime_server_info(
            request, timeout_s=15.0
        ) as response:
            geturl = getattr(response, "geturl", None)
            response_url = geturl() if callable(geturl) else None
            if (
                getattr(response, "status", None) != 200
                or response_url
                != SM121_AGENT_ADMISSION_ENDPOINT.removesuffix("/v1")
                + "/server_info"
            ):
                raise RuntimeErrorWithContext(
                    "SM121 agent runtime attestation is invalid"
                )
            payload = _read_sm121_agent_runtime_response_body(response)
    except urllib.error.HTTPError as error:
        try:
            error.close()
        except OSError:
            pass
        raise RuntimeErrorWithContext(
            "Could not read the SM121 agent runtime attestation"
        ) from None
    except RuntimeErrorWithContext:
        raise
    except (
        OSError,
        TimeoutError,
        UnicodeError,
        ValueError,
        http.client.HTTPException,
        urllib.error.URLError,
    ):
        raise RuntimeErrorWithContext(
            "Could not read the SM121 agent runtime attestation"
        ) from None
    if (
        not isinstance(payload, bytes)
        or not payload
        or len(payload) > _SM121_AGENT_RUNTIME_SERVER_INFO_MAX_BYTES
    ):
        raise RuntimeErrorWithContext("SM121 agent runtime attestation is invalid")
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_sm121_agent_runtime_unique_object,
            parse_constant=_sm121_agent_runtime_reject_constant,
            parse_float=_sm121_agent_runtime_finite_float,
        )
    except (
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
        RecursionError,
    ):
        raise RuntimeErrorWithContext(
            "SM121 agent runtime attestation is invalid"
        ) from None
    if type(decoded) is not dict:
        raise RuntimeErrorWithContext("SM121 agent runtime attestation is invalid")
    return decoded


def _read_sm121_agent_runtime_startup_logs(
    container_id: str, *, started_at: str
) -> bytes:
    """Read a bounded Docker-log snapshot without retaining it outside C1."""

    try:
        process = subprocess.Popen(
            [
                "docker",
                "logs",
                "--since",
                started_at,
                "--timestamps",
                "--tail",
                "1024",
                container_id,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.SubprocessError):
        raise RuntimeErrorWithContext(
            "Could not read the SM121 agent startup attestation"
        ) from None
    selector: selectors.BaseSelector | None = None
    try:
        if process.stdout is None or process.stderr is None:
            raise RuntimeErrorWithContext(
                "Could not read the SM121 agent startup attestation"
            )
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        selector.register(process.stderr, selectors.EVENT_READ)
        chunks: list[bytes] = []
        byte_count = 0
        deadline = time.monotonic() + _SM121_AGENT_RUNTIME_LOG_TIMEOUT_S
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("SM121 agent startup log read timed out")
            events = selector.select(remaining)
            if not events:
                raise TimeoutError("SM121 agent startup log read timed out")
            for key, _event in events:
                stream = key.fileobj
                chunk = os.read(stream.fileno(), 64 * 1024)
                if not chunk:
                    selector.unregister(stream)
                    continue
                byte_count += len(chunk)
                if byte_count > _SM121_AGENT_RUNTIME_LOG_MAX_BYTES:
                    raise RuntimeErrorWithContext(
                        "SM121 agent startup attestation is invalid"
                    )
                chunks.append(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0 or process.wait(timeout=remaining) != 0:
            raise RuntimeErrorWithContext(
                "Could not read the SM121 agent startup attestation"
            )
        return b"".join(chunks)
    except (OSError, subprocess.SubprocessError, TimeoutError):
        raise RuntimeErrorWithContext(
            "Could not read the SM121 agent startup attestation"
        ) from None
    finally:
        if selector is not None:
            selector.close()
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass


def _sm121_agent_runtime_startup_identity(
    container_id: str, *, started_at: str
) -> dict[str, object]:
    """Return one unambiguous scalar cache-startup event for C1 only."""

    if (
        type(container_id) is not str
        or not container_id
        or type(started_at) is not str
        or re.fullmatch(_SM121_AGENT_RUNTIME_STARTED_AT_PATTERN, started_at) is None
    ):
        raise RuntimeErrorWithContext("SM121 agent runtime is not an owned server")
    raw_logs = _read_sm121_agent_runtime_startup_logs(
        container_id,
        started_at=started_at,
    )
    if raw_logs.count(b"\n") > _SM121_AGENT_RUNTIME_LOG_MAX_LINES:
        raise RuntimeErrorWithContext("SM121 agent startup attestation is invalid")
    try:
        lines = raw_logs.decode("utf-8").split("\n")
    except UnicodeDecodeError:
        raise RuntimeErrorWithContext(
            "SM121 agent startup attestation is invalid"
        ) from None
    matches = [
        match.groupdict()
        for line in lines
        if (match := _SM121_CACHE_STARTUP_RE.search(line)) is not None
        and match.end() == len(line)
    ]
    if len(matches) != 1:
        raise RuntimeErrorWithContext("SM121 agent startup attestation is invalid")
    parsed = matches[0]
    return {
        "cache_impl": parsed["cache_impl"],
        "cache_source": parsed["cache_source"],
        "hybrid_swa": parsed["hybrid_swa"] == "True",
        "hybrid_ssm": parsed["hybrid_ssm"] == "True",
        "hicache_attached": parsed["hicache_attached"] == "True",
        "streaming_wrapped": parsed["streaming_wrapped"] == "True",
    }


def inspect_sm121_cache_source_digests(model: Any) -> dict[str, str]:
    """Hash the reviewed cache sources from the exact local image.

    The check is read-only, networkless, and GPU-free.  It returns only the
    allowlisted digest fields, so both cache-policy arms can bind source
    semantics to their frozen local image without retaining source text or
    container filesystem paths.
    """

    image_id = _sm121_storage_image_id(model)
    source_files = tuple(_SM121_CACHE_SOURCE_FILES.values())
    command = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=128m",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--entrypoint",
        "/bin/sh",
        image_id,
        "-c",
        "cd /sgl-workspace/sglang && sha256sum " + " ".join(source_files),
    ]
    try:
        result = _run(command, check=False, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeErrorWithContext(
            "Could not inspect SM121 cache-source semantics"
        ) from error
    if result.returncode:
        raise RuntimeErrorWithContext("SM121 cache-source inspection failed")
    observed: dict[str, str] = {}
    by_path = {path: field for field, path in _SM121_CACHE_SOURCE_FILES.items()}
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or parts[1] not in by_path:
            raise RuntimeErrorWithContext("SM121 cache-source inspection was malformed")
        digest, relative_path = parts
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeErrorWithContext("SM121 cache-source digest was malformed")
        field = by_path[relative_path]
        if field in observed:
            raise RuntimeErrorWithContext("SM121 cache-source digest was duplicated")
        observed[field] = "sha256:" + digest
    if set(observed) != set(_SM121_CACHE_SOURCE_FILES):
        raise RuntimeErrorWithContext("SM121 cache-source inspection was incomplete")
    return observed


def attest_sm121_cache_observability_static_source(model: Any) -> dict[str, Any]:
    """Return an exact scalar cache-source attestation for the local image.

    This does not mount a snapshot, use a GPU, or retain raw source output.
    A digest mismatch fails closed before the serving container is started.
    """

    try:
        validate_sm121_storage_candidate(model)
    except SM121StorageCandidateError as error:
        raise RuntimeErrorWithContext(str(error)) from error
    observed = inspect_sm121_cache_source_digests(model)
    event = {
        "event": SM121_CACHE_STATIC_ATTESTATION_EVENT,
        "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
        **observed,
        **SM121_CACHE_STATIC_ASSERTIONS,
    }
    try:
        validate_sm121_cache_static_attestation_event(event)
    except SM121CacheObservabilityError as error:
        raise RuntimeErrorWithContext("SM121 cache-source semantics changed") from error
    return event


def _extract_sm121_cache_server_info_fields(payload: object) -> dict[str, object]:
    """Project resolved cache fields from an in-memory server-info object."""
    wanted = {
        "disable_radix_cache": bool,
        "mamba_radix_cache_strategy": str,
        "max_mamba_cache_size": (int, type(None)),
    }
    values: dict[str, list[object]] = {field: [] for field in wanted}

    def walk(value: object, *, depth: int = 0) -> None:
        if depth > 24:
            raise RuntimeErrorWithContext("SM121 server-info nesting is invalid")
        if isinstance(value, dict):
            for key, child in value.items():
                expected_type = wanted.get(key)
                if expected_type is not None:
                    if key == "disable_radix_cache":
                        valid = type(child) is bool
                    elif key == "mamba_radix_cache_strategy":
                        valid = isinstance(child, str) and bool(child)
                    else:
                        valid = child is None or (
                            isinstance(child, int) and not isinstance(child, bool)
                        )
                    if not valid:
                        raise RuntimeErrorWithContext(
                            "SM121 cache runtime field is invalid"
                        )
                    values[key].append(child)
                walk(child, depth=depth + 1)
        elif isinstance(value, list):
            for child in value:
                walk(child, depth=depth + 1)

    walk(payload)
    result: dict[str, object] = {}
    for field, observed in values.items():
        if not observed or any(value != observed[0] for value in observed[1:]):
            raise RuntimeErrorWithContext("SM121 cache runtime field is unavailable")
        result[field] = observed[0]
    return result


def _sm121_cache_server_info_fields(server: ManagedServer) -> dict[str, object]:
    """Read a small, allowlisted cache identity from ``/server_info`` only.

    The full server-info reply can contain operational details which must not
    enter a journal. This helper therefore searches it in memory for the
    three resolved cache fields needed by the SM121 cache contracts, verifies
    that repeated copies agree, and returns only scalar values.
    """

    root = server.base_url.removesuffix("/v1").rstrip("/")
    request = urllib.request.Request(
        root + "/server_info", headers=_authorization_headers(server.authorization)
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise RuntimeErrorWithContext(
            "Could not read the SM121 cache runtime attestation"
        ) from error
    return _extract_sm121_cache_server_info_fields(payload)


def _sm121_cache_server_info_disable_radix(server: ManagedServer) -> bool:
    """Extract the legacy B0 cache-off flag without widening its contract.

    The paired semantic lane requires all three resolved cache settings, but
    B0 predates that stronger contract and is intentionally attested only to
    ``disable_radix_cache``.  Keep this read narrow so a compatible cache-off
    server cannot regress merely because it omits unrelated Mamba fields.
    """

    root = server.base_url.removesuffix("/v1").rstrip("/")
    request = urllib.request.Request(
        root + "/server_info", headers=_authorization_headers(server.authorization)
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise RuntimeErrorWithContext(
            "Could not read the SM121 cache runtime attestation"
        ) from error
    values: list[bool] = []

    def walk(value: object, *, depth: int = 0) -> None:
        if depth > 24:
            raise RuntimeErrorWithContext("SM121 server-info nesting is invalid")
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "disable_radix_cache":
                    if type(child) is not bool:
                        raise RuntimeErrorWithContext(
                            "SM121 cache runtime flag is invalid"
                        )
                    values.append(child)
                walk(child, depth=depth + 1)
        elif isinstance(value, list):
            for child in value:
                walk(child, depth=depth + 1)

    walk(payload)
    if not values or any(value is not values[0] for value in values[1:]):
        raise RuntimeErrorWithContext("SM121 cache runtime flag is unavailable")
    return values[0]


def _sm121_cache_startup_identity(server: ManagedServer) -> dict[str, object]:
    """Project the allowlisted cache-startup identity from bounded Docker logs."""

    if server.backend != "sglang" or not server.container_id:
        raise RuntimeErrorWithContext("SM121 cache runtime is not an owned SGLang server")
    result = _run(
        ["docker", "logs", "--tail", "4096", server.container_id],
        check=False,
        timeout=30,
    )
    if result.returncode:
        raise RuntimeErrorWithContext("Could not read SM121 cache startup attestation")
    matches = []
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        found = _SM121_CACHE_STARTUP_RE.search(line)
        if found is not None:
            matches.append(found.groupdict())
    if not matches or any(match != matches[0] for match in matches[1:]):
        raise RuntimeErrorWithContext("SM121 cache startup attestation is unavailable")
    parsed = matches[0]
    return {
        "cache_impl": parsed["cache_impl"],
        "cache_source": parsed["cache_source"],
        "hybrid_swa": parsed["hybrid_swa"] == "True",
        "hybrid_ssm": parsed["hybrid_ssm"] == "True",
        "hicache_attached": parsed["hicache_attached"] == "True",
        "streaming_wrapped": parsed["streaming_wrapped"] == "True",
    }


def inspect_sm121_cache_runtime_identity(
    server: ManagedServer, *, disable_radix_cache_override: bool | None = None
) -> dict[str, Any]:
    """Return the compact resolved cache identity without retaining raw logs.

    This is shared by the cache-off B0 lane and the paired semantic canary.
    It intentionally returns resolved arguments and startup identity only; the
    caller still supplies the exact arm-specific acceptance contract.
    """

    startup = _sm121_cache_startup_identity(server)
    if disable_radix_cache_override is None:
        server_info = _sm121_cache_server_info_fields(server)
        disabled = server_info["disable_radix_cache"]
        strategy = server_info["mamba_radix_cache_strategy"]
        max_mamba_cache_size = server_info["max_mamba_cache_size"]
    else:
        if type(disable_radix_cache_override) is not bool:
            raise RuntimeErrorWithContext("SM121 cache runtime identity is invalid")
        disabled = disable_radix_cache_override
        # A disabled Radix cache makes both source-attested predicates false.
        # The legacy B0 contract only attests that cache-off implication and
        # deliberately does not retain the unrelated resolved strategy.
        strategy = "no_buffer"
        max_mamba_cache_size = None
    if type(disabled) is not bool or not isinstance(strategy, str):
        raise RuntimeErrorWithContext("SM121 cache runtime identity is invalid")
    if max_mamba_cache_size is not None and (
        not isinstance(max_mamba_cache_size, int)
        or isinstance(max_mamba_cache_size, bool)
        or max_mamba_cache_size <= 0
    ):
        raise RuntimeErrorWithContext("SM121 cache runtime identity is invalid")
    return {
        **startup,
        "disable_radix_cache": disabled,
        "mamba_radix_cache_strategy": strategy,
        "max_mamba_cache_size": max_mamba_cache_size,
        "mamba_extra_buffer_enabled": (
            not disabled and strategy in {"extra_buffer", "extra_buffer_lazy"}
        ),
        "mamba_extra_buffer_lazy_enabled": (
            not disabled and strategy == "extra_buffer_lazy"
        ),
    }


_SM121_AGENT_RUNTIME_TOP_LEVEL_FIELDS = (
    "disable_radix_cache",
    "mamba_radix_cache_strategy",
    "max_mamba_cache_size",
    "reasoning_parser",
    "tool_call_parser",
    "chunked_prefill_size",
    "max_running_requests",
    "max_total_tokens",
    "context_length",
)


def _sm121_agent_runtime_top_level_fields(
    payload: dict[str, object],
) -> dict[str, object]:
    """Read C1 cache/parser/limit facts only from documented root fields.

    The pinned image's ``/server_info`` endpoint expands
    ``resolved_config_dict(dataclasses.asdict(server_args))`` into its root
    object.  C1 therefore rejects rather than recursively searching decoys.
    """

    result: dict[str, object] = {}
    for field in _SM121_AGENT_RUNTIME_TOP_LEVEL_FIELDS:
        if field not in payload:
            raise RuntimeErrorWithContext("SM121 agent runtime field is unavailable")
        value = payload[field]
        if field == "disable_radix_cache":
            valid = type(value) is bool
        elif field in {
            "mamba_radix_cache_strategy",
            "reasoning_parser",
            "tool_call_parser",
        }:
            valid = isinstance(value, str) and bool(value)
        else:
            valid = type(value) is int and value > 0
        if not valid:
            raise RuntimeErrorWithContext("SM121 agent runtime field is invalid")
        result[field] = value
    return result


def inspect_sm121_agent_admission_runtime_identity(
    server: ManagedServer,
) -> dict[str, object]:
    """Return the exact C1 scalar runtime identity for one owned server.

    It issues one bounded, no-proxy/no-redirect ``/server_info`` read and one
    bounded Docker startup-log read. Full responses, logs, credentials,
    endpoints, container IDs, and timings never leave this function.
    """

    owned_server, binding_before = _require_sm121_agent_admission_server(server)
    payload = _read_sm121_agent_runtime_server_info(binding_before.authorization)
    top_level = _sm121_agent_runtime_top_level_fields(payload)
    startup = _sm121_agent_runtime_startup_identity(
        binding_before.container_id,
        started_at=binding_before.generation[0],
    )
    # Bind both bounded observations to the same live owned listener.  The
    # second check fails closed if the container stopped or its loopback port
    # changed while the two attestations were being read.
    post_server, binding_after = _require_sm121_agent_admission_server(
        owned_server
    )
    if post_server is not owned_server or binding_after != binding_before:
        raise RuntimeErrorWithContext("SM121 agent runtime attestation is invalid")
    disabled = top_level["disable_radix_cache"]
    strategy = top_level["mamba_radix_cache_strategy"]
    identity = {
        **startup,
        **top_level,
        "mamba_extra_buffer_enabled": (
            disabled is False and strategy in {"extra_buffer", "extra_buffer_lazy"}
        ),
        "mamba_extra_buffer_lazy_enabled": (
            disabled is False and strategy == "extra_buffer_lazy"
        ),
    }
    try:
        return validate_sm121_agent_admission_runtime_identity(identity)
    except SM121AgentAdmissionError as error:
        raise RuntimeErrorWithContext(
            "SM121 agent runtime identity is invalid"
        ) from error


def inspect_sm121_chunked_prefill_runtime_identity(
    server: ManagedServer,
) -> dict[str, Any]:
    """Return cache identity plus the resolved chunked-prefill size.

    The cache-policy lanes predate a chunk-size attestation and deliberately
    retain only their narrow cache field set.  This separate reader is for the
    1K-versus-2K performance lane: it walks ``/server_info`` in memory,
    accepts one unambiguous positive integer, and returns only that scalar.
    It never writes the full response or a container path to a journal.
    """

    observed = inspect_sm121_cache_runtime_identity(server)
    root = server.base_url.removesuffix("/v1").rstrip("/")
    request = urllib.request.Request(
        root + "/server_info", headers=_authorization_headers(server.authorization)
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise RuntimeErrorWithContext(
            "Could not read the SM121 chunked-prefill runtime attestation"
        ) from error
    values: list[int] = []

    def walk(value: object, *, depth: int = 0) -> None:
        if depth > 24:
            raise RuntimeErrorWithContext("SM121 server-info nesting is invalid")
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "chunked_prefill_size":
                    if (
                        isinstance(child, bool)
                        or not isinstance(child, int)
                        or child <= 0
                    ):
                        raise RuntimeErrorWithContext(
                            "SM121 chunked-prefill runtime field is invalid"
                        )
                    values.append(child)
                walk(child, depth=depth + 1)
        elif isinstance(value, list):
            for child in value:
                walk(child, depth=depth + 1)

    walk(payload)
    if not values or any(value != values[0] for value in values[1:]):
        raise RuntimeErrorWithContext(
            "SM121 chunked-prefill runtime field is unavailable"
        )
    return {**observed, "chunked_prefill_size": values[0]}


def attest_sm121_cache_observability_runtime(server: ManagedServer) -> dict[str, Any]:
    """Capture the one cache-off startup fact set without persisting logs/info."""
    observed = inspect_sm121_cache_runtime_identity(
        server,
        disable_radix_cache_override=_sm121_cache_server_info_disable_radix(server),
    )
    event = {
        "event": SM121_CACHE_RUNTIME_ATTESTATION_EVENT,
        **{
            field: observed[field]
            for field in SM121_CACHE_RUNTIME_EXPECTED
        },
    }
    try:
        validate_sm121_cache_runtime_attestation_event(event)
    except SM121CacheObservabilityError as error:
        raise RuntimeErrorWithContext("SM121 cache runtime semantics changed") from error
    return event


def _sm121_cache_metric_defaults() -> dict[str, Any]:
    return {
        "available": False,
        # The B0 observer does not depend on guardrail counters, but the
        # paired semantic canary does.  Keep their availability distinct so
        # adding that stronger lane cannot retrospectively change B0's
        # admitted zero-hit semantics.
        "guardrail_metrics_available": False,
        "prefill_input_tokens": 0,
        "prefill_device_hit_tokens": 0,
        "prefill_host_hit_tokens": 0,
        "prefill_storage_hit_tokens": 0,
        "cached_total_tokens": 0,
        "cached_device_tokens": 0,
        "cached_host_tokens": 0,
        "cached_storage_tokens": 0,
        "cached_total_series_present": False,
        "cached_device_series_present": False,
        "cached_host_series_present": False,
        "cached_storage_series_present": False,
        "kv_available_tokens": 0,
        "kv_evictable_tokens": 0,
        "kv_used_tokens": 0,
        "mamba_available_tokens": 0,
        "mamba_evictable_tokens": 0,
        "mamba_used_tokens": 0,
        "evicted_tokens": 0,
        "retracted_requests": 0,
    }


def _sm121_cache_labels(raw: str | None) -> dict[str, str] | None:
    if raw in {None, ""}:
        return {}
    labels: dict[str, str] = {}
    position = 0
    for match in _SM121_CACHE_LABEL_RE.finditer(raw):
        if match.start() != position:
            return None
        position = match.end()
        name = match.group("name")
        if name in labels:
            return None
        labels[name] = match.group("value")
    return labels if position == len(raw) else None


def _sm121_cache_metric_integer(raw: str) -> int | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0 or not value.is_integer():
        return None
    return int(value)


def snapshot_sm121_cache_observability_metrics(
    server: ManagedServer, *, semantic_arm: str | None = None
) -> dict[str, Any]:
    """Read a scalar-only cache metrics snapshot, failing closed on gaps.

    The paired semantic canary has a narrower, source-pinned interpretation of
    its two guardrail counters.  Its Prometheus multiprocess endpoint does not
    expose any part of a labelled counter family until a child is materialized.
    B's ``ChunkCache`` cannot materialize eviction metrics at all; A's unified
    cache and both arms' scheduler only materialize these families on positive
    activity.  A completely absent family is therefore the verified zero form
    for this exact image, while a partial family stays unavailable.  The older
    B0 caller keeps its sample-only interpretation by passing no arm.
    """

    snapshot = _sm121_cache_metric_defaults()
    if semantic_arm not in {
        None,
        SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
        SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
    }:
        return snapshot
    root = server.base_url.removesuffix("/v1").rstrip("/")
    request = urllib.request.Request(
        root + "/metrics", headers=_authorization_headers(server.authorization)
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError) as error:
        del error
        return snapshot
    prefill_seen = {
        "input": False,
        "device_hit": False,
        "host_hit": False,
        "storage_hit": False,
    }
    gauge_seen = {
        "sglang:kv_available_tokens": False,
        "sglang:kv_evictable_tokens": False,
        "sglang:kv_used_tokens": False,
        "sglang:mamba_available_tokens": False,
        "sglang:mamba_evictable_tokens": False,
        "sglang:mamba_used_tokens": False,
    }
    gauges = {
        "sglang:kv_available_tokens": "kv_available_tokens",
        "sglang:kv_evictable_tokens": "kv_evictable_tokens",
        "sglang:kv_used_tokens": "kv_used_tokens",
        "sglang:mamba_available_tokens": "mamba_available_tokens",
        "sglang:mamba_evictable_tokens": "mamba_evictable_tokens",
        "sglang:mamba_used_tokens": "mamba_used_tokens",
    }
    cached_seen = {
        "total": False,
        "device": False,
        "host": False,
        "storage": False,
    }
    guardrail_families = {
        name: {"help": False, "type": False, "sample": False, "seen": False}
        for name in _SM121_CACHE_GUARDRAIL_SAMPLES
    }
    scheduler_labels: dict[str, str] | None = None
    tokenizer_labels: dict[str, str] | None = None

    def bind_scheduler_labels(candidate: dict[str, str]) -> bool:
        """Require one stable scheduler-label vector without retaining it."""

        nonlocal scheduler_labels
        if (
            semantic_arm is not None
            and set(candidate) != _SM121_CACHE_SCHEDULER_LABEL_FIELDS
        ):
            return False
        if scheduler_labels is None:
            scheduler_labels = dict(candidate)
            return True
        return scheduler_labels == candidate

    def bind_tokenizer_labels(candidate: dict[str, str]) -> bool:
        """Bind a finished-request cache counter to its narrower label schema."""

        nonlocal tokenizer_labels
        if semantic_arm is None:
            return bind_scheduler_labels(candidate)
        if set(candidate) != _SM121_CACHE_TOKENIZER_LABEL_FIELDS:
            return False
        if tokenizer_labels is None:
            tokenizer_labels = dict(candidate)
            return True
        return tokenizer_labels == candidate

    def guardrail_family_for_name(name: str) -> str | None:
        """Return the canonical family for a relevant exact-image metric name."""

        for family, prefix in _SM121_CACHE_GUARDRAIL_FAMILY_PREFIXES.items():
            if name.startswith(prefix):
                return family
        return None

    def relevant_guardrail_line(line: str, marker: str) -> bool:
        candidate = line.lstrip()
        return any(
            candidate.startswith(marker + prefix)
            for prefix in _SM121_CACHE_GUARDRAIL_FAMILY_PREFIXES.values()
        )

    def materialized_guardrail_family_is_valid(family: str) -> bool:
        state = guardrail_families[family]
        return bool(state["help"] and state["type"] and state["sample"])

    malformed = False
    for line in text.splitlines():
        if line.startswith("# HELP "):
            help_match = re.fullmatch(
                r"# HELP (?P<name>sglang:[A-Za-z0-9_:]+) .+", line
            )
            if help_match is None:
                if relevant_guardrail_line(line, "# HELP "):
                    malformed = True
                continue
            help_name = help_match.group("name")
            family = guardrail_family_for_name(help_name)
            if family is None:
                continue
            state = guardrail_families[family]
            if help_name != family or state["seen"] or state["help"]:
                malformed = True
                continue
            state["seen"] = True
            state["help"] = True
            continue
        if line.startswith("# TYPE "):
            type_match = _SM121_CACHE_METRIC_TYPE_RE.fullmatch(line)
            if type_match is None:
                if relevant_guardrail_line(line, "# TYPE "):
                    malformed = True
                continue
            type_name = type_match.group("name")
            family = guardrail_family_for_name(type_name)
            if family is None:
                continue
            state = guardrail_families[family]
            if (
                type_name != family
                or type_match.group("metric_type") != "counter"
                or not state["help"]
                or state["type"]
            ):
                malformed = True
                continue
            state["seen"] = True
            state["type"] = True
            continue
        match = _SM121_CACHE_METRIC_LINE_RE.fullmatch(line)
        if match is None:
            if any(
                relevant_guardrail_line(line, marker)
                for marker in ("", "# HELP ", "# TYPE ")
            ):
                malformed = True
            continue
        name = match.group("name")
        family = guardrail_family_for_name(name)
        if family is not None and name != family:
            malformed = True
            continue
        if name not in {
            "sglang:prefill_effective_tokens_total",
            "sglang:cached_tokens_total",
            *gauges,
            *_SM121_CACHE_GUARDRAIL_SAMPLES,
        }:
            continue
        labels = _sm121_cache_labels(match.group("labels"))
        value = _sm121_cache_metric_integer(match.group("value"))
        if labels is None or value is None:
            malformed = True
            continue
        if name == "sglang:prefill_effective_tokens_total":
            mode = labels.get("mode")
            field = {
                "input": "prefill_input_tokens",
                "device_hit": "prefill_device_hit_tokens",
                "host_hit": "prefill_host_hit_tokens",
                "storage_hit": "prefill_storage_hit_tokens",
            }.get(mode)
            metric_base = {key: item for key, item in labels.items() if key != "mode"}
            if (
                field is None
                or prefill_seen[mode]
                or not bind_scheduler_labels(metric_base)
            ):
                malformed = True
                continue
            snapshot[field] = value
            prefill_seen[mode] = True
        elif name == "sglang:cached_tokens_total":
            source = labels.get("cache_source")
            source_kind = (
                "total"
                if source == "total"
                else "device"
                if source == "device"
                else "host"
                if source == "host"
                else "storage"
                if isinstance(source, str)
                and source.startswith("storage_")
                and len(source) > len("storage_")
                else None
            )
            metric_base = {
                key: item for key, item in labels.items() if key != "cache_source"
            }
            if (
                source_kind is None
                or cached_seen[source_kind]
                or not bind_tokenizer_labels(metric_base)
            ):
                malformed = True
                continue
            snapshot[f"cached_{source_kind}_tokens"] = value
            snapshot[f"cached_{source_kind}_series_present"] = True
            cached_seen[source_kind] = True
        elif name in _SM121_CACHE_GUARDRAIL_SAMPLES:
            state = guardrail_families[name]
            if state["sample"] or (
                semantic_arm is not None
                and (not state["help"] or not state["type"])
            ):
                malformed = True
                continue
            if name == "sglang:evicted_tokens_total":
                if semantic_arm == SM121_CACHE_SEMANTIC_CACHE_OFF_ARM:
                    malformed = True
                    continue
                if semantic_arm == SM121_CACHE_SEMANTIC_CACHE_ON_ARM:
                    labels_match = labels == _SM121_CACHE_CACHE_ON_EVICTION_LABELS
                else:
                    labels_match = bind_scheduler_labels(labels)
                if not labels_match:
                    malformed = True
                    continue
            elif not bind_scheduler_labels(labels):
                malformed = True
                continue
            if semantic_arm is not None and value == 0:
                malformed = True
                continue
            snapshot[_SM121_CACHE_GUARDRAIL_SAMPLES[name]] = value
            state["seen"] = True
            state["sample"] = True
        else:
            if not bind_scheduler_labels(labels):
                malformed = True
                continue
            field = gauges[name]
            if gauge_seen[name]:
                malformed = True
                continue
            snapshot[field] = value
            gauge_seen[name] = True
    if semantic_arm is not None and tokenizer_labels is not None:
        expected_tokenizer_labels = (
            {
                field: scheduler_labels[field]
                for field in _SM121_CACHE_TOKENIZER_LABEL_FIELDS
            }
            if scheduler_labels is not None
            and _SM121_CACHE_TOKENIZER_LABEL_FIELDS.issubset(scheduler_labels)
            else None
        )
        if tokenizer_labels != expected_tokenizer_labels:
            malformed = True
    snapshot["available"] = not malformed and all(prefill_seen.values()) and all(
        gauge_seen.values()
    )
    if semantic_arm == SM121_CACHE_SEMANTIC_CACHE_OFF_ARM:
        eviction = guardrail_families["sglang:evicted_tokens_total"]
        retraction = guardrail_families["sglang:num_retracted_requests_total"]
        snapshot["guardrail_metrics_available"] = (
            snapshot["available"] is True
            and not malformed
            and eviction["seen"] is False
            and (
                retraction["seen"] is False
                or materialized_guardrail_family_is_valid(
                    "sglang:num_retracted_requests_total"
                )
            )
        )
    elif semantic_arm == SM121_CACHE_SEMANTIC_CACHE_ON_ARM:
        snapshot["guardrail_metrics_available"] = snapshot["available"] is True and (
            not malformed
            and all(
                state["seen"] is False
                or materialized_guardrail_family_is_valid(family)
                for family, state in guardrail_families.items()
            )
        )
    else:
        snapshot["guardrail_metrics_available"] = not malformed and all(
            state["sample"] for state in guardrail_families.values()
        )
    return snapshot


def settle_sm121_cache_observability_metrics(
    server: ManagedServer,
    *,
    timeout_s: float = 45.0,
    poll_interval_s: float = 1.0,
    semantic_arm: str | None = None,
) -> tuple[dict[str, Any], float, int, bool]:
    """Wait for two identical scalar metric views; never retain raw metrics text."""

    started = time.monotonic()
    previous: dict[str, Any] | None = None
    polls = 0
    while True:
        current = snapshot_sm121_cache_observability_metrics(
            server, semantic_arm=semantic_arm
        )
        polls += 1
        if not current["available"]:
            return current, time.monotonic() - started, polls, False
        if previous == current:
            return current, time.monotonic() - started, polls, True
        if time.monotonic() - started >= timeout_s:
            return current, time.monotonic() - started, polls, False
        previous = current
        time.sleep(poll_interval_s)


def _sm121_cache_exact_json_count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeErrorWithContext(f"SM121 cache response {field} is invalid")
    return value


def _sm121_cache_optional_json_count(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _sm121_cache_exact_json_count(value, field)


def _sm121_cache_response_detail_observation(
    payload: dict[str, Any]
) -> tuple[str, int | None, int | None, int | None]:
    """Extract only cache-detail states/counts, never response content or IDs."""

    if "sglext" not in payload:
        return "omitted", None, None, None
    extension = payload["sglext"]
    if extension is None:
        return "null", None, None, None
    if not isinstance(extension, dict):
        return "unexpected", None, None, None
    if "cached_tokens_details" not in extension:
        return "omitted", None, None, None
    details = extension["cached_tokens_details"]
    if details is None:
        return "null", None, None, None
    if (
        not isinstance(details, dict)
        or set(details) - {"device", "host", "storage", "storage_backend"}
        or not {"device", "host"}.issubset(details)
        or ("storage_backend" in details and "storage" not in details)
    ):
        return "unexpected", None, None, None
    try:
        device = _sm121_cache_exact_json_count(
            details["device"], "response device cached tokens"
        )
        host = _sm121_cache_exact_json_count(
            details["host"], "response host cached tokens"
        )
        storage = _sm121_cache_exact_json_count(
            details.get("storage", 0), "response storage cached tokens"
        )
    except RuntimeErrorWithContext:
        return "unexpected", None, None, None
    state = "zero_details" if (device, host, storage) == (0, 0, 0) else "nonzero_details"
    return state, device, host, storage


def _sm121_cache_usage_detail_observation(
    usage: dict[str, Any]
) -> tuple[str, int | None]:
    """Extract only the exact OpenAI usage cache detail state/count."""

    if "prompt_tokens_details" not in usage:
        return "omitted", None
    details = usage["prompt_tokens_details"]
    if details is None:
        return "null", None
    if not isinstance(details, dict) or set(details) != {"cached_tokens"}:
        return "unexpected", None
    try:
        cached = _sm121_cache_exact_json_count(
            details["cached_tokens"], "usage cached tokens"
        )
    except RuntimeErrorWithContext:
        return "unexpected", None
    return ("zero_details" if cached == 0 else "nonzero_details"), cached


def request_sm121_cache_observability_zero_hit(
    server: ManagedServer, *, served_name: object
) -> dict[str, Any]:
    """Run B0's one non-streaming request and return scalar observations only.

    The response body remains in memory only long enough to verify the strict
    protocol fields below.  It is never returned, journaled, logged, or added
    to an exception, so generated text, reasoning, tool payloads, IDs, and
    server model strings cannot enter a tracked evidence path.
    """

    parsed = urlsplit(server.base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or parsed.path.rstrip("/") != "/v1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeErrorWithContext("SM121 cache request endpoint is not loopback")
    try:
        validate_sm121_cache_zero_hit_request_contract()
        request_body = sm121_cache_zero_hit_request_body(served_name)
    except SM121CacheObservabilityError as error:
        raise RuntimeErrorWithContext("SM121 cache request contract is invalid") from error
    request = urllib.request.Request(
        server.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(request_body, separators=(",", ":")).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **_authorization_headers(server.authorization),
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            raw = response.read(1_048_577)
    except urllib.error.HTTPError as error:
        error.close()
        raise RuntimeErrorWithContext("SM121 cache zero-hit request was rejected") from None
    except (OSError, urllib.error.URLError, TimeoutError) as error:
        raise RuntimeErrorWithContext("SM121 cache zero-hit request failed") from error
    elapsed_s = time.monotonic() - started
    if len(raw) > 1_048_576:
        raise RuntimeErrorWithContext("SM121 cache zero-hit response exceeded its limit")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeErrorWithContext("SM121 cache zero-hit response was invalid") from error
    if not isinstance(payload, dict):
        raise RuntimeErrorWithContext("SM121 cache zero-hit response was not an object")
    usage = payload.get("usage")
    choices = payload.get("choices")
    if (
        not isinstance(usage, dict)
        or not isinstance(choices, list)
        or len(choices) != 1
    ):
        raise RuntimeErrorWithContext("SM121 cache zero-hit response lacks required scalars")
    if not all(isinstance(choice, dict) for choice in choices):
        raise RuntimeErrorWithContext("SM121 cache zero-hit response choices are invalid")
    message = choices[0].get("message")
    if (
        not isinstance(message, dict)
        or not isinstance(message.get("content"), str)
        or message["content"].strip() != SM121_CACHE_ZERO_HIT_EXPECTED_RESPONSE
    ):
        raise RuntimeErrorWithContext("SM121 cache zero-hit response failed validation")
    prompt_tokens = _sm121_cache_exact_json_count(
        usage.get("prompt_tokens"), "prompt tokens"
    )
    completion_tokens = _sm121_cache_exact_json_count(
        usage.get("completion_tokens"), "completion tokens"
    )
    if completion_tokens <= 0:
        raise RuntimeErrorWithContext("SM121 cache zero-hit response has no completion")
    reasoning_tokens = _sm121_cache_optional_json_count(
        usage.get("reasoning_tokens"), "reasoning tokens"
    )
    response_state, response_device, response_host, response_storage = (
        _sm121_cache_response_detail_observation(payload)
    )
    usage_state, usage_cached = _sm121_cache_usage_detail_observation(usage)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "elapsed_s": elapsed_s,
        "response_detail_state": response_state,
        "response_device_cached_tokens": response_device,
        "response_host_cached_tokens": response_host,
        "response_storage_cached_tokens": response_storage,
        "usage_detail_state": usage_state,
        "usage_cached_tokens": usage_cached,
    }


def request_sm121_cache_semantic_turn(
    server: ManagedServer,
    *,
    served_name: object,
    messages: list[dict[str, str]],
    expected_response: str,
    max_tokens: int,
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    """Issue one non-streaming semantic-cache request without retaining text.

    Prompt token IDs are returned only to the caller's in-memory identity
    checker.  The caller must project them to counts/booleans before journaling;
    neither prompt text, response text, IDs, nor token IDs are returned in the
    scalar result projection.
    """

    parsed = urlsplit(server.base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or parsed.path.rstrip("/") != "/v1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeErrorWithContext("SM121 semantic-cache endpoint is not loopback")
    if not isinstance(served_name, str) or not served_name:
        raise RuntimeErrorWithContext("SM121 semantic-cache served name is invalid")
    if not isinstance(expected_response, str) or not expected_response:
        raise RuntimeErrorWithContext("SM121 semantic-cache expected response is invalid")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 1 <= max_tokens <= 128
    ):
        raise RuntimeErrorWithContext("SM121 semantic-cache output cap is invalid")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or not 0 < float(timeout_s) <= 900
    ):
        raise RuntimeErrorWithContext("SM121 semantic-cache request timeout is invalid")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 16:
        raise RuntimeErrorWithContext("SM121 semantic-cache messages are invalid")
    for message in messages:
        if (
            type(message) is not dict
            or set(message) != {"role", "content"}
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            or not message["content"]
        ):
            raise RuntimeErrorWithContext("SM121 semantic-cache messages are invalid")
    request_body = {
        "model": served_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "n": 1,
        "stream": False,
        "return_cached_tokens_details": True,
        "return_prompt_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(request_body, separators=(",", ":")).encode("utf-8")
    if len(encoded) > 4_194_304:
        raise RuntimeErrorWithContext("SM121 semantic-cache request exceeded its limit")
    request = urllib.request.Request(
        server.base_url.rstrip("/") + "/chat/completions",
        data=encoded,
        headers={
            "Content-Type": "application/json",
            **_authorization_headers(server.authorization),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
            raw = response.read(4_194_305)
    except urllib.error.HTTPError as error:
        error.close()
        raise RuntimeErrorWithContext("SM121 semantic-cache request was rejected") from None
    except (OSError, urllib.error.URLError, TimeoutError) as error:
        raise RuntimeErrorWithContext("SM121 semantic-cache request failed") from error
    if len(raw) > 4_194_304:
        raise RuntimeErrorWithContext("SM121 semantic-cache response exceeded its limit")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeErrorWithContext("SM121 semantic-cache response was invalid") from error
    if not isinstance(payload, dict):
        raise RuntimeErrorWithContext("SM121 semantic-cache response was not an object")
    usage = payload.get("usage")
    choices = payload.get("choices")
    if (
        not isinstance(usage, dict)
        or not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or choices[0].get("index") != 0
    ):
        raise RuntimeErrorWithContext("SM121 semantic-cache response lacks required scalars")
    message = choices[0].get("message")
    if (
        not isinstance(message, dict)
        or not isinstance(message.get("content"), str)
        or message["content"].strip() != expected_response
    ):
        raise RuntimeErrorWithContext("SM121 semantic-cache response failed validation")
    raw_prompt_token_ids = choices[0].get("prompt_token_ids")
    if (
        not isinstance(raw_prompt_token_ids, list)
        or not raw_prompt_token_ids
        or len(raw_prompt_token_ids) > 65_536
    ):
        raise RuntimeErrorWithContext("SM121 semantic-cache prompt IDs are unavailable")
    prompt_token_ids: tuple[int, ...] = tuple()
    parsed_ids: list[int] = []
    for token_id in raw_prompt_token_ids:
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            or token_id > 2**31 - 1
        ):
            raise RuntimeErrorWithContext("SM121 semantic-cache prompt IDs are invalid")
        parsed_ids.append(token_id)
    prompt_token_ids = tuple(parsed_ids)
    prompt_tokens = _sm121_cache_exact_json_count(
        usage.get("prompt_tokens"), "prompt tokens"
    )
    if prompt_tokens != len(prompt_token_ids):
        raise RuntimeErrorWithContext("SM121 semantic-cache prompt token count disagrees")
    completion_tokens = _sm121_cache_exact_json_count(
        usage.get("completion_tokens"), "completion tokens"
    )
    if completion_tokens <= 0:
        raise RuntimeErrorWithContext("SM121 semantic-cache response has no completion")
    reasoning_tokens = _sm121_cache_optional_json_count(
        usage.get("reasoning_tokens"), "reasoning tokens"
    )
    response_state, response_device, response_host, response_storage = (
        _sm121_cache_response_detail_observation(payload)
    )
    usage_state, usage_cached = _sm121_cache_usage_detail_observation(usage)
    return {
        "private_prompt_token_ids": prompt_token_ids,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "response_detail_state": response_state,
        "response_device_cached_tokens": response_device,
        "response_host_cached_tokens": response_host,
        "response_storage_cached_tokens": response_storage,
        "usage_detail_state": usage_state,
        "usage_cached_tokens": usage_cached,
    }


def _resolve_sglang_source_overlays(
    model: Any, *, workspace: Path
) -> tuple[tuple[Path, str, str, str], ...]:
    """Resolve and verify manifest-pinned SGLang source overlays."""

    configured = tuple(getattr(model, "sglang_source_overlays", ()) or ())
    if not configured:
        return ()
    try:
        workspace_root = workspace.resolve(strict=True)
    except OSError as error:
        raise RuntimeErrorWithContext(
            "Could not resolve the workspace for SGLang source overlays"
        ) from error
    container_root = PurePosixPath(
        "/sgl-workspace/sglang/python/sglang"
    )
    resolved_overlays: list[tuple[Path, str, str, str]] = []
    seen_host_paths: set[Path] = set()
    seen_container_paths: set[str] = set()
    for index, overlay in enumerate(configured):
        context = f"SGLang source overlay {index}"
        host_path = str(getattr(overlay, "host_path", "") or "")
        container_path = str(
            getattr(overlay, "container_path", "") or ""
        )
        expected_digest = str(getattr(overlay, "digest", "") or "")
        relative = PurePosixPath(host_path)
        if (
            not host_path
            or relative.is_absolute()
            or host_path != str(relative)
            or any(part in {"", ".", ".."} for part in relative.parts)
            or "\\" in host_path
            or ":" in host_path
            or relative.suffix != ".py"
        ):
            raise RuntimeErrorWithContext(
                f"{context} host path must be a safe relative Python path"
            )
        absolute_container = PurePosixPath(container_path)
        try:
            container_relative = absolute_container.relative_to(container_root)
        except ValueError as error:
            raise RuntimeErrorWithContext(
                f"{context} target must be beneath {container_root}"
            ) from error
        if (
            not absolute_container.is_absolute()
            or not container_relative.parts
            or any(part in {"", ".", ".."} for part in absolute_container.parts)
            or absolute_container.suffix != ".py"
            or relative.name != absolute_container.name
            or ":" in container_path
        ):
            raise RuntimeErrorWithContext(
                f"{context} target must be a matching SGLang Python source path"
            )
        if not _SHA256_PATTERN.fullmatch(expected_digest):
            raise RuntimeErrorWithContext(
                f"{context} must pin a full lowercase sha256 digest"
            )
        candidate = workspace_root.joinpath(*relative.parts)
        cursor = workspace_root
        try:
            for component in relative.parts:
                cursor = cursor / component
                if cursor.is_symlink():
                    raise RuntimeErrorWithContext(
                        f"{context} must not traverse a symbolic link"
                    )
            host_file = candidate.resolve(strict=True)
            host_file.relative_to(workspace_root)
        except RuntimeErrorWithContext:
            raise
        except (OSError, ValueError) as error:
            raise RuntimeErrorWithContext(
                f"{context} is missing or escapes the workspace"
            ) from error
        if not host_file.is_file():
            raise RuntimeErrorWithContext(f"{context} must be a regular file")
        if ":" in str(host_file):
            raise RuntimeErrorWithContext(
                f"{context} cannot be represented as a Docker bind mount"
            )
        actual_digest = _sha256_file(host_file)
        if actual_digest != expected_digest:
            raise RuntimeErrorWithContext(
                f"{context} digest mismatch: expected {expected_digest}, "
                f"got {actual_digest}"
            )
        if host_file in seen_host_paths or container_path in seen_container_paths:
            raise RuntimeErrorWithContext(
                "SGLang source overlay host and target paths must be unique"
            )
        seen_host_paths.add(host_file)
        seen_container_paths.add(container_path)
        resolved_overlays.append(
            (host_file, container_path, expected_digest, host_path)
        )
    return tuple(resolved_overlays)


def _private_sglang_ple_dir(model: Any) -> Path:
    """Create the sole writable backing directory for patched Qwen PLE."""

    model_id = str(getattr(model, "id", "") or "")
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", model_id) is None:
        raise RuntimeErrorWithContext(
            "SGLang PLE mmap requires a safe stable model ID"
        )
    cache_root = (Path.home() / ".cache" / "sparkbench" / "sglang").resolve()
    cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    candidate = cache_root / model_id / "ple"
    cursor = cache_root
    for component in (model_id, "ple"):
        cursor = cursor / component
        if cursor.is_symlink():
            raise RuntimeErrorWithContext(
                "SGLang PLE backing directory must not traverse a symbolic link"
            )
    candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
    ple_dir = candidate.resolve(strict=True)
    try:
        ple_dir.relative_to(cache_root)
    except ValueError as error:
        raise RuntimeErrorWithContext(
            "SGLang PLE backing directory escapes the private runtime cache"
        ) from error
    if not ple_dir.is_dir():
        raise RuntimeErrorWithContext(
            "SGLang PLE backing path must be a directory"
        )
    ple_dir.chmod(0o700)
    return ple_dir


def _sglang_argument_value(
    arguments: tuple[str, ...], option: str
) -> tuple[str, ...]:
    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                raise RuntimeErrorWithContext(
                    f"SGLang option {option} is missing its value"
                )
            values.append(arguments[index + 1])
        elif argument.startswith(option + "="):
            values.append(argument.partition("=")[2])
    return tuple(values)


def _validate_readonly_sglang_ple_loader(model: Any) -> None:
    arguments = tuple(str(argument) for argument in getattr(model, "args", ()))
    if arguments.count("--ple-offload-embedding") != 1:
        raise RuntimeErrorWithContext(
            "read-only SGLang PLE reuse requires --ple-offload-embedding"
        )
    load_formats = _sglang_argument_value(arguments, "--load-format")
    if len(load_formats) > 1 or any(value != "auto" for value in load_formats):
        raise RuntimeErrorWithContext(
            "read-only SGLang PLE reuse requires load_format=auto; "
            "fastsafetensors is incompatible"
        )
    for incompatible in (
        "--weight-loader-disable-mmap",
        "--weight-loader-prefetch-checkpoints",
    ):
        if any(
            argument == incompatible or argument.startswith(incompatible + "=")
            for argument in arguments
        ):
            raise RuntimeErrorWithContext(
                "read-only SGLang PLE reuse is incompatible with " + incompatible
            )


def _readonly_sglang_ple_dir(
    model: Any,
    source_overlays: tuple[tuple[Path, str, str, str], ...],
) -> tuple[Path, PLECacheRecord]:
    """Admit the exact completed Qwen3.8 PLE cache without changing it."""

    exact_identity = (
        str(getattr(model, "source", "") or "") == QWEN38_PLE_LAYOUT.source
        and str(getattr(model, "revision", "") or "")
        == QWEN38_PLE_LAYOUT.revision
        and str(getattr(model, "recipe_source", "") or "")
        == QWEN38_PLE_LAYOUT.recipe_source
        and str(getattr(model, "recipe_revision", "") or "")
        == QWEN38_PLE_LAYOUT.recipe_revision
    )
    if not exact_identity:
        raise RuntimeErrorWithContext(
            "read-only SGLang PLE reuse is pinned to the exact Qwen3.8 "
            "artifact and audited recipe"
        )
    expected_marker = "sha256:" + qwen38_ple_marker_sha256()
    expected_payload = "sha256:" + QWEN38_PLE_LAYOUT.payload_sha256
    if (
        getattr(model, "sglang_ple_cache_marker_digest", None) != expected_marker
        or getattr(model, "sglang_ple_cache_payload_digest", None)
        != expected_payload
    ):
        raise RuntimeErrorWithContext(
            "read-only SGLang PLE cache marker/payload pins do not match "
            "the audited layout"
        )
    actual_overlays = {
        container_path: digest
        for _host_file, container_path, digest, _relative in source_overlays
    }
    if actual_overlays not in _QWEN38_READONLY_PLE_OVERLAY_VARIANTS:
        raise RuntimeErrorWithContext(
            "read-only SGLang PLE reuse requires the exact persistent-cache "
            "source overlays"
        )
    _validate_readonly_sglang_ple_loader(model)

    try:
        cache_root = (
            Path.home() / ".cache" / "sparkbench" / "sglang"
        ).resolve(strict=True)
        cache = qwen38_ple_cache_path(home=Path.home())
        resolved = cache.resolve(strict=True)
        resolved.relative_to(cache_root)
        record = validate_qwen38_ple_cache(
            resolved, layout=QWEN38_PLE_LAYOUT, verify_payload=False
        )
    except (OSError, ValueError, PLECacheError) as error:
        raise RuntimeErrorWithContext(
            f"read-only SGLang PLE cache admission failed: {error}"
        ) from error
    if (
        "sha256:" + record.marker_sha256 != expected_marker
        or "sha256:" + record.payload_sha256 != expected_payload
    ):
        raise RuntimeErrorWithContext(
            "read-only SGLang PLE cache validation returned unexpected pins"
        )
    return resolved, record


def start_vllm(
    model: Any,
    *,
    workspace: Path,
    port: int = 8000,
    allow_download: bool = False,
    server_log_path: Path | None = None,
) -> ManagedServer:
    existing = _existing_container()
    if existing:
        container_id, managed, existing_run = existing
        ownership = "managed" if managed else "unmanaged"
        raise RuntimeErrorWithContext(
            f"Refusing to replace existing {ownership} container {CONTAINER_NAME} "
            f"({container_id}, run={existing_run or 'unknown'})"
        )
    if not _port_is_free(port):
        raise RuntimeErrorWithContext(f"Port {port} is already in use")

    cache_root = (workspace / "data").resolve()
    configured_cache = getattr(model, "cache_dir", None)
    if configured_cache == "user":
        hf_cache = Path.home() / ".cache" / "huggingface"
    elif configured_cache in {None, "project"}:
        hf_cache = cache_root / "huggingface"
    else:
        hf_cache = Path(str(configured_cache)).expanduser().resolve()
    vllm_cache = cache_root / "vllm"
    hf_cache.mkdir(parents=True, exist_ok=True)
    vllm_cache.mkdir(parents=True, exist_ok=True)
    source = str(model.source)
    repository_dir = hf_cache / "hub" / ("models--" + source.replace("/", "--"))
    revision = getattr(model, "revision", None)
    if revision:
        source_cached = (repository_dir / "snapshots" / str(revision)).is_dir()
    else:
        snapshot_root = repository_dir / "snapshots"
        source_cached = snapshot_root.is_dir() and any(snapshot_root.iterdir())
    explicit_cached = getattr(model, "cached", None)
    if explicit_cached is not None:
        source_cached = bool(explicit_cached)
    if not allow_download and not source_cached:
        raise RuntimeErrorWithContext(
            f"{source} is not marked cached; rerun with --allow-download to permit network retrieval"
        )
    run_identity = str(getattr(model, "run_identity", "unknown"))
    command = [
        "docker", "run", "--detach", "--name", CONTAINER_NAME,
        "--label", MANAGED_LABEL, "--label", f"ai.sparkbench.run={run_identity}",
        "--entrypoint", "vllm",
        "--gpus", "all", "--ipc", "host",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "--publish", f"127.0.0.1:{port}:8000",
        "--volume", f"{hf_cache}:/root/.cache/huggingface",
        "--volume", f"{vllm_cache}:/root/.cache/vllm",
        "--env", "HF_HOME=/root/.cache/huggingface",
    ]
    if not allow_download:
        command.extend(
            [
                "--pull=never",
                "--env", "HF_HUB_OFFLINE=1",
                "--env", "TRANSFORMERS_OFFLINE=1",
            ]
        )
    if os.environ.get("HF_TOKEN"):
        command.extend(["--env", "HF_TOKEN", "--env", "HUGGING_FACE_HUB_TOKEN"])
    image_reference = getattr(model, "resolved_image", None) or model.image
    command.extend([str(image_reference), "serve", source])
    arguments = [str(argument) for argument in model.args]
    if revision and "--revision" not in arguments:
        arguments.extend(["--revision", str(revision)])
    if "--served-model-name" not in arguments:
        arguments.extend(["--served-model-name", str(model.served_name)])
    command.extend(arguments)
    result = _run(command, check=False, timeout=60)
    if result.returncode:
        raise RuntimeErrorWithContext(f"docker run failed: {result.stderr.strip()}")
    container_id = result.stdout.strip()
    server = ManagedServer(
        "vllm",
        f"http://127.0.0.1:{port}/v1",
        container_id=container_id,
        run_identity=run_identity,
    )
    try:
        server.startup_s = wait_for_endpoint(server.base_url, float(model.startup_timeout_s), container_id)
    except BaseException as startup_error:
        if server_log_path is not None:
            try:
                save_server_logs(server, server_log_path)
            except Exception as log_error:
                startup_error.add_note(
                    "Could not persist full startup container logs: "
                    f"{type(log_error).__name__}: {log_error}"
                )
        server.stop()
        raise
    return server


def start_sglang(
    model: Any,
    *,
    workspace: Path,
    port: int = 30000,
    allow_download: bool = False,
    server_log_path: Path | None = None,
    abort_check: Callable[[], None] | None = None,
    on_server_created: Callable[[ManagedServer], None] | None = None,
) -> ManagedServer:
    """Start a digest-pinned SGLang server from exact cached snapshots."""

    storage_canary_authorized = (
        getattr(model, "storage_canary_authorized", False) is True
    )
    cache_semantic_canary_authorized = (
        getattr(model, "cache_semantic_canary_authorized", False) is True
    )
    cache_performance_authorized = (
        getattr(model, "cache_performance_authorized", False) is True
    )
    chunked_prefill_performance_authorized = (
        getattr(model, "chunked_prefill_performance_authorized", False) is True
    )
    blocker = model_execution_blocker(
        model,
        allow_sm121_storage_canary=storage_canary_authorized,
        allow_sm121_cache_semantic_canary=cache_semantic_canary_authorized,
        allow_sm121_cache_performance=cache_performance_authorized,
        allow_sm121_chunked_prefill_performance=(
            chunked_prefill_performance_authorized
        ),
    )
    if blocker is not None:
        raise RuntimeErrorWithContext(blocker)
    # Runtime acquisition is always forbidden. A typed profile may permit only
    # the pinned image's documented metadata probe after both snapshots exist.
    del allow_download
    if abort_check is not None:
        abort_check()
    existing = _existing_container(SGLANG_CONTAINER_NAME)
    if existing:
        container_id, managed, existing_run = existing
        ownership = "managed" if managed else "unmanaged"
        raise RuntimeErrorWithContext(
            "Refusing to replace existing "
            f"{ownership} container {SGLANG_CONTAINER_NAME} "
            f"({container_id}, run={existing_run or 'unknown'})"
        )
    if not _port_is_free(port):
        raise RuntimeErrorWithContext(f"Port {port} is already in use")

    cache_root = (workspace / "data").resolve()
    configured_cache = getattr(model, "cache_dir", None)
    if configured_cache == "user":
        hf_cache = Path.home() / ".cache" / "huggingface"
    elif configured_cache in {None, "project"}:
        hf_cache = cache_root / "huggingface"
    else:
        hf_cache = Path(str(configured_cache)).expanduser().resolve()

    source = str(model.source)
    revision = str(getattr(model, "revision", "") or "")
    target_repository, container_snapshot = _exact_sglang_snapshot(
        hf_cache, source=source, revision=revision, role="target"
    )
    if (
        is_sm121_cache_semantic_candidate(model)
        or is_sm121_cache_performance_candidate(model)
        or is_sm121_chunked_prefill_performance_candidate(model)
        or is_sm121_storage_candidate(model)
    ):
        return _start_sglang_sm121_storage(
            model,
            workspace=workspace,
            target_repository=target_repository,
            container_snapshot=container_snapshot,
            port=port,
            server_log_path=server_log_path,
            abort_check=abort_check,
            on_server_created=on_server_created,
        )
    draft_source = getattr(model, "draft_source", None)
    draft_revision = getattr(model, "draft_revision", None)
    if (draft_source is None) != (draft_revision is None):
        raise RuntimeErrorWithContext(
            "SGLang draft source and revision must be configured together"
        )
    container_draft_snapshot: str | None = None
    draft_repository: Path | None = None
    if draft_source is not None and draft_revision is not None:
        draft_repository, container_draft_snapshot = _exact_sglang_snapshot(
            hf_cache,
            source=str(draft_source),
            revision=str(draft_revision),
            role="draft",
        )

    image_reference = str(getattr(model, "resolved_image", None) or model.image)
    expected_image_digest = str(getattr(model, "image_digest", "") or "")
    if (
        len(expected_image_digest) != 71
        or not expected_image_digest.startswith("sha256:")
        or any(
            character not in "0123456789abcdef"
            for character in expected_image_digest.removeprefix("sha256:")
        )
    ):
        raise RuntimeErrorWithContext(
            "SGLang image digest must be explicitly pinned"
        )
    if "@sha256:" not in image_reference:
        image_reference = f"{model.image}@{expected_image_digest}"
    if not image_reference.endswith("@" + expected_image_digest):
        raise RuntimeErrorWithContext(
            "Resolved SGLang image does not match the manifest sha256 digest"
        )

    run_identity = str(getattr(model, "run_identity", "unknown"))
    metadata_probe = bool(
        getattr(model, "sglang_allow_hf_metadata_probe", False)
    )
    if metadata_probe and container_draft_snapshot is None:
        raise RuntimeErrorWithContext(
            "SGLang Hugging Face metadata probe requires a pinned draft snapshot"
        )
    if metadata_probe and (
        getattr(model, "recipe_source", None) is None
        or getattr(model, "recipe_revision", None) is None
    ):
        raise RuntimeErrorWithContext(
            "SGLang Hugging Face metadata probe requires pinned recipe provenance"
        )
    compile_cache: Path | None = None
    if container_draft_snapshot is not None:
        model_id = str(getattr(model, "id", ""))
        runtime_cache_root = (
            Path.home() / ".cache" / "sparkbench" / "sglang"
        ).resolve()
        compile_cache = (runtime_cache_root / model_id / "compile").resolve()
        try:
            compile_cache.relative_to(runtime_cache_root)
        except ValueError as error:
            raise RuntimeErrorWithContext(
                "SGLang compile cache escapes the private runtime cache"
            ) from error
        compile_cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    source_overlays = _resolve_sglang_source_overlays(
        model, workspace=workspace
    )
    ple_mmap = bool(getattr(model, "sglang_ple_mmap", False))
    raw_ple_omitted = getattr(model, "sglang_ple_omitted", False)
    if type(raw_ple_omitted) is not bool:
        raise RuntimeErrorWithContext("SGLang PLE omission flag must be boolean")
    ple_omitted = raw_ple_omitted
    ple_cache_mode = getattr(model, "sglang_ple_cache_mode", None)
    if ple_cache_mode not in {None, "readonly"}:
        raise RuntimeErrorWithContext(
            "SGLang PLE cache mode must be absent or 'readonly'"
        )
    if ple_cache_mode is not None and not ple_mmap:
        raise RuntimeErrorWithContext(
            "SGLang PLE cache mode requires sglang_ple_mmap"
        )
    if ple_cache_mode is None and any(
        getattr(model, field, None) is not None
        for field in (
            "sglang_ple_cache_marker_digest",
            "sglang_ple_cache_payload_digest",
        )
    ):
        raise RuntimeErrorWithContext(
            "SGLang PLE cache pins require an explicit cache mode"
        )
    if ple_mmap and not source_overlays:
        raise RuntimeErrorWithContext(
            "SGLang PLE mmap requires verified source overlays"
        )
    if ple_mmap and container_draft_snapshot is not None:
        raise RuntimeErrorWithContext(
            "SGLang PLE mmap requires an embedded draft and one backing mount"
        )
    if ple_omitted:
        if ple_mmap or ple_cache_mode is not None:
            raise RuntimeErrorWithContext(
                "SGLang PLE omission forbids PLE mmap/cache reuse"
            )
        if not source_overlays:
            raise RuntimeErrorWithContext(
                "SGLang PLE omission requires verified source overlays"
            )
        exact_identity = (
            str(getattr(model, "source", "") or "")
            == QWEN38_PLE_LAYOUT.source
            and str(getattr(model, "revision", "") or "")
            == QWEN38_PLE_LAYOUT.revision
            and str(getattr(model, "recipe_source", "") or "")
            == QWEN38_PLE_LAYOUT.recipe_source
            and str(getattr(model, "recipe_revision", "") or "")
            == QWEN38_PLE_LAYOUT.recipe_revision
        )
        if not exact_identity:
            raise RuntimeErrorWithContext(
                "SGLang PLE omission is pinned to the exact Qwen3.8 artifact "
                "and audited recipe"
            )
        if (
            getattr(model, "draft_source", None) is not None
            or container_draft_snapshot is not None
        ):
            raise RuntimeErrorWithContext(
                "SGLang PLE omission requires the embedded draft"
            )
        actual_overlays = {
            container_path: digest
            for _host_file, container_path, digest, _relative in source_overlays
        }
        if actual_overlays != _QWEN38_ABLATION_CAPABLE_PLE_OVERLAYS:
            raise RuntimeErrorWithContext(
                "SGLang PLE omission requires the exact ablation-capable "
                "source overlays"
            )
        argument_values = tuple(str(argument) for argument in model.args)
        if "--ple-offload-embedding" in argument_values:
            raise RuntimeErrorWithContext(
                "SGLang PLE omission forbids --ple-offload-embedding"
            )
        expected_override = '{"sparkbench_omit_ple":true}'
        override_positions = tuple(
            index
            for index, argument in enumerate(argument_values)
            if argument == "--json-model-override-args"
        )
        inline_overrides = tuple(
            argument
            for argument in argument_values
            if argument.startswith("--json-model-override-args=")
        )
        if (
            len(override_positions) != 1
            or override_positions[0] + 1 >= len(argument_values)
            or argument_values[override_positions[0] + 1] != expected_override
            or inline_overrides
        ):
            raise RuntimeErrorWithContext(
                "SGLang PLE omission requires the canonical model override"
            )
    ple_record: PLECacheRecord | None = None
    if ple_cache_mode == "readonly":
        ple_dir, ple_record = _readonly_sglang_ple_dir(model, source_overlays)
    else:
        ple_dir = _private_sglang_ple_dir(model) if ple_mmap else None

    key = secrets.token_urlsafe(32)
    auth = f"Bearer {key}"
    sensitive_values = (key, auth)
    api_key_path = (
        server_log_path.parent / "api-key"
        if server_log_path is not None
        else None
    )
    if api_key_path is not None:
        _write_private_secret(api_key_path, key)

    command = [
        "docker", "run", "--detach", "--pull=never",
        "--name", SGLANG_CONTAINER_NAME,
        "--label", MANAGED_LABEL,
        "--label", f"ai.sparkbench.run={run_identity}",
        "--label", "ai.sparkbench.backend=sglang",
        "--gpus", "all",
    ]
    if container_draft_snapshot is not None:
        command.extend(
            [
                "--memory", "100g",
                "--memory-swap", "100g",
                "--shm-size", "16g",
                "--entrypoint", "python3",
            ]
        )
    else:
        command.extend(["--ipc", "host", "--entrypoint", "sglang"])
    command.extend(
        [
            "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
            "--publish", f"127.0.0.1:{port}:30000",
            "--volume",
            f"{target_repository}:/root/.cache/huggingface/hub/"
            f"{target_repository.name}:ro",
            "--env", "HF_HOME=/tmp/sparkbench-hf",
            "--env", "HF_HUB_CACHE=/tmp/sparkbench-hf/hub",
            "--env", "HF_MODULES_CACHE=/tmp/sparkbench-hf/modules",
            "--env", "HF_TOKEN_PATH=/tmp/sparkbench-hf/token-disabled",
            "--env", "HF_HUB_DISABLE_IMPLICIT_TOKEN=1",
            "--env", "HF_HUB_DISABLE_TELEMETRY=1",
        ]
    )
    if draft_repository is not None:
        command.extend(
            [
                "--volume",
                f"{draft_repository}:/root/.cache/huggingface/hub/"
                f"{draft_repository.name}:ro",
            ]
        )
    for host_file, container_path, _digest, _relative_path in source_overlays:
        command.extend(
            ["--volume", f"{host_file}:{container_path}:ro"]
        )
    if ple_dir is not None:
        command.extend(
            [
                "--volume",
                f"{ple_dir}:/ple:{'ro' if ple_cache_mode == 'readonly' else 'rw'}",
                "--env",
                "SGLANG_QWEN4_PLE_MMAP_DIR=/ple",
            ]
        )
        if ple_cache_mode == "readonly":
            assert ple_record is not None
            command.extend(
                [
                    "--env",
                    "SGLANG_QWEN4_PLE_CACHE_MODE=readonly",
                    "--env",
                    "SGLANG_QWEN4_PLE_CACHE_LOADER_CONTRACT="
                    "auto-mmap-no-prefetch",
                    "--env",
                    "SGLANG_QWEN4_PLE_CACHE_MARKER_SHA256="
                    + ple_record.marker_sha256,
                ]
            )
    if compile_cache is not None:
        command.extend(
            [
                "--volume", f"{compile_cache}:/cache",
                "--env", "TORCHINDUCTOR_CACHE_DIR=/cache/inductor",
            ]
        )
    if not metadata_probe:
        command.extend(
            [
                "--env", "HF_HUB_OFFLINE=1",
                "--env", "TRANSFORMERS_OFFLINE=1",
            ]
        )
    command.append(image_reference)
    if container_draft_snapshot is not None:
        command.extend(
            ["-m", "sglang.launch_server", "--model-path", container_snapshot]
        )
        command.extend(
            ["--speculative-draft-model-path", container_draft_snapshot]
        )
    else:
        command.extend(["serve", "--model-path", container_snapshot])
    # Keep a leading '-' in the URL-safe random value from being parsed as a
    # new option by SGLang's argparse CLI.
    command.append("--api-key=" + key)
    command.extend(str(argument) for argument in model.args)
    if abort_check is not None:
        try:
            abort_check()
        except BaseException:
            _unlink_private_secret(api_key_path)
            raise
    try:
        result = _run(command, check=False, timeout=60)
    except BaseException as launch_error:
        _unlink_private_secret(api_key_path)
        raise RuntimeErrorWithContext(
            "docker run failed: "
            + _redact_text(str(launch_error), sensitive_values)
        ) from None
    if result.returncode:
        _unlink_private_secret(api_key_path)
        raise RuntimeErrorWithContext(
            "docker run failed: "
            + _redact_text(result.stderr.strip(), sensitive_values)
        )
    container_id = result.stdout.strip()
    if not container_id:
        _unlink_private_secret(api_key_path)
        raise RuntimeErrorWithContext(
            "docker run returned no authenticated SGLang container ID"
        )
    server = ManagedServer(
        "sglang",
        f"http://127.0.0.1:{port}/v1",
        container_id=container_id,
        run_identity=run_identity,
        authorization=auth,
        api_key=key,
        api_key_path=api_key_path,
    )
    server.native_provenance = {
        "target_source": source,
        "target_revision": revision,
        "target_container_snapshot": container_snapshot,
        "draft_source": draft_source,
        "draft_revision": draft_revision,
        "draft_container_snapshot": container_draft_snapshot,
        "compile_cache_dir": str(compile_cache) if compile_cache else None,
        "sglang_source_overlays": [
            {
                "host_path": relative_path,
                "container_path": container_path,
                "sha256": digest,
            }
            for _host_file, container_path, digest, relative_path in source_overlays
        ],
        "sglang_ple_mmap": ple_mmap,
        "sglang_ple_omitted": ple_omitted,
        "sglang_ple_cache_mode": ple_cache_mode,
        "sglang_ple_cache_marker_digest": (
            "sha256:" + ple_record.marker_sha256 if ple_record else None
        ),
        "sglang_ple_cache_payload_digest": (
            "sha256:" + ple_record.payload_sha256 if ple_record else None
        ),
        "sglang_ple_container_dir": "/ple" if ple_mmap else None,
        "sglang_ple_backing_policy": (
            "verified_persistent_readonly"
            if ple_cache_mode == "readonly"
            else "private_runtime_cache" if ple_mmap else None
        ),
        "sglang_allow_hf_metadata_probe": metadata_probe,
        "hf_network_policy": (
            "documented_longcat_metadata_probe"
            if metadata_probe
            else "offline"
        ),
        "docker_memory": "100g" if container_draft_snapshot else None,
        "docker_memory_swap": "100g" if container_draft_snapshot else None,
        "docker_shm_size": "16g" if container_draft_snapshot else None,
        "recipe_source": getattr(model, "recipe_source", None),
        "recipe_revision": getattr(model, "recipe_revision", None),
        "benchmark_scope": "sparkbench_suite_not_upstream_battery",
        "model_acquisition": "disabled_exact_read_only_snapshots",
        "api_authentication": "ephemeral_bearer",
        "api_key_file_mode": "0600" if api_key_path is not None else None,
    }
    try:
        if on_server_created is not None:
            on_server_created(server)
        if abort_check is not None:
            abort_check()
        wait_arguments: dict[str, Any] = {
            "authorization": auth,
            "sensitive_values": sensitive_values,
        }
        if abort_check is not None:
            wait_arguments["abort_check"] = abort_check
        server.startup_s = wait_for_endpoint(
            server.base_url,
            float(model.startup_timeout_s),
            container_id,
            **wait_arguments,
        )
    except BaseException as startup_error:
        if server_log_path is not None:
            try:
                save_server_logs(server, server_log_path)
            except Exception as log_error:
                _redact_exception(log_error, sensitive_values)
                startup_error.add_note(
                    "Could not persist full startup container logs: "
                    f"{type(log_error).__name__}: {log_error}"
                )
        try:
            server.stop()
        except BaseException as cleanup_error:
            _redact_exception(cleanup_error, sensitive_values)
            startup_error.add_note(
                "Could not clean up authenticated SGLang server: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        _redact_exception(startup_error, sensitive_values)
        raise startup_error
    return server


def _llamacpp_port(model: Any) -> int:
    parsed = urlsplit(str(model.endpoint))
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or parsed.path.rstrip("/") != "/v1"
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeErrorWithContext(
            "llama.cpp endpoint must be canonical http://127.0.0.1:<port>/v1"
        )
    return parsed.port


def _llamacpp_alias_ready(
    base_url: str, served_name: str, *, require_multimodal: bool = False
) -> bool:
    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + "/models", timeout=2
        ) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return False
    models = payload.get("data", []) if isinstance(payload, dict) else []
    alias_ready = any(
        isinstance(item, dict) and item.get("id") == served_name
        for item in models
    )
    if not alias_ready or not require_multimodal:
        return alias_ready
    capability_models = payload.get("models", [])
    return any(
        isinstance(item, dict)
        and (item.get("name") == served_name or item.get("model") == served_name)
        and isinstance(item.get("capabilities"), list)
        and "multimodal" in item["capabilities"]
        for item in capability_models
    )


def _tail_text(path: Path, lines: int = 100) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def wait_for_llamacpp(
    server: ManagedServer,
    *,
    served_name: str,
    timeout_s: float,
    require_multimodal: bool = False,
) -> float:
    """Wait for health plus the exact model alias while monitoring the child."""

    if server.process is None or server.process_state_path is None:
        raise RuntimeErrorWithContext("llama.cpp startup has no managed process")
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        returncode = server.process.poll()
        if returncode is not None:
            log_path = (
                Path(server.native_provenance["server_log_path"])
                if server.native_provenance
                else Path("/nonexistent")
            )
            detail = _tail_text(log_path)
            raise RuntimeErrorWithContext(
                f"llama-server exited during startup with code {returncode}"
                + (f"\n{detail}" if detail else "")
            )
        if endpoint_ready(server.base_url) and _llamacpp_alias_ready(
            server.base_url,
            served_name,
            require_multimodal=require_multimodal,
        ):
            state = _read_native_state(
                server.process_state_path, str(server.run_identity)
            )
            if not _native_process_is_owned(state):
                raise RuntimeErrorWithContext(
                    "llama-server disappeared while its endpoint became ready"
                )
            return time.monotonic() - started
        time.sleep(2)
    raise RuntimeErrorWithContext(
        f"llama-server did not become ready within {timeout_s:.0f}s"
    )


def start_llamacpp(
    model: Any,
    *,
    workspace: Path,
    allow_download: bool = False,
    server_log_path: Path | None = None,
    process_state_path: Path | None = None,
    validated_artifacts: dict[str, Any] | None = None,
    artifact_validation_s: float | None = None,
) -> ManagedServer:
    """Start an exact native llama-server with offline, owned lifecycle state."""

    del allow_download  # Runtime acquisition is forbidden for native profiles.
    if server_log_path is None or process_state_path is None:
        raise RuntimeErrorWithContext(
            "Managed llama.cpp requires explicit log and process-state paths"
        )
    if process_state_path.exists() or process_state_path.is_symlink():
        raise RuntimeErrorWithContext(
            f"Refusing to overwrite existing native process state: {process_state_path}"
        )
    port = _llamacpp_port(model)
    if not _port_is_free(port):
        raise RuntimeErrorWithContext(f"Port {port} is already in use")
    if validated_artifacts is None:
        validation_started = time.monotonic()
        artifacts = validate_llamacpp_artifacts(model, workspace=workspace)
        artifact_validation_s = time.monotonic() - validation_started
    else:
        artifacts = dict(validated_artifacts)
    run_identity = str(getattr(model, "run_identity", "unknown"))
    server_id = _native_server_id(run_identity)
    command = _native_command(model, artifacts, port=port)
    environment = _native_environment(run_identity, server_id)

    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    if server_log_path.is_symlink():
        raise RuntimeErrorWithContext(
            f"Refusing symlink server log path: {server_log_path}"
        )
    log_fd = os.open(
        server_log_path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT,
        0o600,
    )
    log_stream = os.fdopen(log_fd, "a", encoding="utf-8")
    log_stream.write(
        f"--- SparkBench llama.cpp attempt run={run_identity} server={server_id} ---\n"
    )
    log_stream.flush()
    os.fsync(log_stream.fileno())
    try:
        process = subprocess.Popen(
            command,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            cwd=str(artifacts["runtime_source_dir"]),
            env=environment,
            start_new_session=True,
        )
    except BaseException:
        log_stream.close()
        raise

    try:
        start_ticks = _proc_start_ticks(process.pid)
        pgid = os.getpgid(process.pid)
        if pgid != process.pid:
            raise RuntimeErrorWithContext(
                "llama-server was not isolated into its own process group"
            )
        state = {
            "schema_version": 1,
            "backend": LLAMACPP_BACKEND,
            "run_identity": run_identity,
            "server_id": server_id,
            "pid": process.pid,
            "pgid": pgid,
            "start_ticks": start_ticks,
            "uid": os.geteuid(),
            "runtime_binary": artifacts["runtime_binary"],
            "runtime_digest": artifacts["runtime_binary_sha256"],
            "model_path": artifacts["model_path"],
            "model_digest": artifacts["model_sha256"],
            "base_url": str(model.endpoint),
            "argv": command,
        }
        if artifacts.get("model_shards"):
            state.update(
                {
                    "model_shards": artifacts["model_shards"],
                    "model_total_size_bytes": artifacts[
                        "model_total_size_bytes"
                    ],
                }
            )
        if artifacts.get("mmproj_path"):
            state.update(
                {
                    "mmproj_path": artifacts["mmproj_path"],
                    "mmproj_digest": artifacts["mmproj_sha256"],
                }
            )
        if artifacts.get("draft_model_path"):
            state.update(
                {
                    "draft_model_path": artifacts["draft_model_path"],
                    "draft_model_digest": artifacts["draft_model_sha256"],
                }
            )
        _write_private_json(process_state_path, state)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log_stream.close()
        raise

    provenance = {
        **artifacts,
        "argv": command,
        "pid": process.pid,
        "process_group_id": pgid,
        "process_start_ticks": start_ticks,
        "run_identity": run_identity,
        "server_id": server_id,
        "server_log_path": str(server_log_path),
        "process_state_path": str(process_state_path),
        "artifact_validation_s": artifact_validation_s,
        "runtime_parallel": int(model.runtime_parallel),
        "runtime_total_context": int(model.max_context)
        * int(model.runtime_parallel),
        "served_max_context": int(model.max_context),
        "offline": True,
        "loopback_only": True,
        "cors_origins": "localhost",
        "cors_credentials": False,
    }
    server = ManagedServer(
        LLAMACPP_BACKEND,
        str(model.endpoint),
        run_identity=run_identity,
        process=process,
        process_state_path=process_state_path,
        process_log=log_stream,
        native_provenance=provenance,
    )
    try:
        server.startup_s = wait_for_llamacpp(
            server,
            served_name=str(model.served_name),
            timeout_s=float(model.startup_timeout_s),
            require_multimodal="vision" in getattr(model, "tasks", ()),
        )
    except BaseException as startup_error:
        try:
            save_server_logs(server, server_log_path)
        except Exception as log_error:
            startup_error.add_note(
                "Could not flush native startup logs: "
                f"{type(log_error).__name__}: {log_error}"
            )
        server.stop()
        raise
    return server


def _scan_owned_llamacpp(
    model: Any,
    *,
    workspace: Path,
    run_identity: str,
) -> list[dict[str, Any]]:
    """Find a crash-gap child by same-UID markers and exact executable/argv."""

    artifacts = validate_llamacpp_artifacts(model, workspace=workspace)
    server_id = _native_server_id(run_identity)
    port = _llamacpp_port(model)
    command = _native_command(model, artifacts, port=port)
    marker = f"SPARKBENCH_RUN_ID={run_identity}".encode()
    server_marker = f"SPARKBENCH_SERVER_ID={server_id}".encode()
    matches: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if _proc_uid(pid) != os.geteuid():
                continue
            environment = _proc_environment(pid)
            if marker not in environment or server_marker not in environment:
                continue
            state = {
                "schema_version": 1,
                "backend": LLAMACPP_BACKEND,
                "run_identity": run_identity,
                "server_id": server_id,
                "pid": pid,
                "pgid": os.getpgid(pid),
                "start_ticks": _proc_start_ticks(pid),
                "uid": os.geteuid(),
                "runtime_binary": artifacts["runtime_binary"],
                "runtime_digest": artifacts["runtime_binary_sha256"],
                "model_path": artifacts["model_path"],
                "model_digest": artifacts["model_sha256"],
                "base_url": str(model.endpoint),
                "argv": command,
            }
            if artifacts.get("mmproj_path"):
                state.update(
                    {
                        "mmproj_path": artifacts["mmproj_path"],
                        "mmproj_digest": artifacts["mmproj_sha256"],
                    }
                )
            if artifacts.get("draft_model_path"):
                state.update(
                    {
                        "draft_model_path": artifacts["draft_model_path"],
                        "draft_model_digest": artifacts["draft_model_sha256"],
                    }
                )
            if _native_process_is_owned(state):
                matches.append(state)
        except (OSError, RuntimeErrorWithContext):
            if entry.exists():
                try:
                    environment = _proc_environment(pid)
                except OSError:
                    continue
                if marker in environment:
                    raise RuntimeErrorWithContext(
                        f"Could not unambiguously verify marked llama-server PID {pid}"
                    )
    if len(matches) > 1:
        raise RuntimeErrorWithContext(
            "Multiple native processes carry this frozen run identity; refusing recovery"
        )
    return matches


def recover_owned_llamacpp(
    model: Any,
    *,
    workspace: Path,
    run_identity: str,
    process_state_path: Path,
) -> str:
    """Stop only a process proven to belong to the crashed frozen run."""

    if process_state_path.exists() or process_state_path.is_symlink():
        return _stop_native_state(process_state_path, run_identity)
    matches = _scan_owned_llamacpp(
        model, workspace=workspace, run_identity=run_identity
    )
    if not matches:
        return "already_absent"
    _terminate_owned_native_state(matches[0])
    return "stopped_owned_crash_gap_process"


def connect_ollama(model: Any) -> ManagedServer:
    base_url = str(model.endpoint or "http://127.0.0.1:11434/v1")
    if not endpoint_ready(base_url):
        raise RuntimeErrorWithContext(
            "Ollama is not reachable. Start its service, then rerun the Ollama phase."
        )
    tags_url = base_url.removesuffix("/v1").rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as response:
            tags = json.load(response).get("models", [])
    except (OSError, ValueError) as error:
        raise RuntimeErrorWithContext(f"Could not verify Ollama model revision: {error}") from error
    local = next(
        (item for item in tags if item.get("name") == str(model.source) or item.get("model") == str(model.source)),
        None,
    )
    expected_revision = getattr(model, "revision", None)
    actual_revision = (local or {}).get("digest", "").removeprefix("sha256:")
    if local is None or (expected_revision and not actual_revision.startswith(str(expected_revision))):
        raise RuntimeErrorWithContext(
            f"Ollama model revision no longer matches the frozen plan: {model.source}"
        )
    was_loaded = _ollama_model_loaded(base_url, str(model.source))
    return ManagedServer(
        "ollama",
        base_url,
        ollama_model=str(model.source),
        unload_ollama=not was_loaded,
    )


def connect_external(model: Any) -> ManagedServer:
    base_url = str(model.endpoint)
    if not base_url or not endpoint_ready(base_url):
        raise RuntimeErrorWithContext(f"External endpoint is not ready: {base_url}")
    return ManagedServer("external", base_url)


def start_server(
    model: Any,
    *,
    workspace: Path,
    allow_download: bool = False,
    server_log_path: Path | None = None,
    process_state_path: Path | None = None,
    validated_llamacpp_artifacts: dict[str, Any] | None = None,
    artifact_validation_s: float | None = None,
    abort_check: Callable[[], None] | None = None,
    on_server_created: Callable[[ManagedServer], None] | None = None,
) -> ManagedServer:
    if model.backend != "sglang" and (
        abort_check is not None or on_server_created is not None
    ):
        raise RuntimeErrorWithContext(
            "Host-safety lifecycle callbacks are supported only for SGLang"
        )
    if model.backend == LLAMACPP_BACKEND:
        return start_llamacpp(
            model,
            workspace=workspace,
            allow_download=allow_download,
            server_log_path=server_log_path,
            process_state_path=process_state_path,
            validated_artifacts=validated_llamacpp_artifacts,
            artifact_validation_s=artifact_validation_s,
        )
    if model.backend == "vllm":
        return start_vllm(
            model,
            workspace=workspace,
            allow_download=allow_download,
            server_log_path=server_log_path,
        )
    if model.backend == "sglang":
        safety_callbacks: dict[str, Any] = {}
        if abort_check is not None:
            safety_callbacks["abort_check"] = abort_check
        if on_server_created is not None:
            safety_callbacks["on_server_created"] = on_server_created
        return start_sglang(
            model,
            workspace=workspace,
            allow_download=allow_download,
            server_log_path=server_log_path,
            **safety_callbacks,
        )
    if model.backend == "ollama":
        return connect_ollama(model)
    if model.backend == "external":
        return connect_external(model)
    raise RuntimeErrorWithContext(f"Backend lifecycle is not implemented: {model.backend}")


def capture_server_provenance(server: ManagedServer) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "backend": server.backend,
        "base_url": server.base_url,
        "startup_s": server.startup_s,
    }
    if server.container_id:
        result = _run(
            ["docker", "inspect", server.container_id], check=False, timeout=20
        )
        if result.returncode == 0:
            inspect = json.loads(result.stdout)[0]
            argv = inspect["Config"].get("Entrypoint", []) + inspect[
                "Config"
            ].get("Cmd", [])
            sensitive_values = tuple(
                value
                for value in (server.api_key, server.authorization)
                if value
            )
            provenance.update(
                {
                    "container_id": inspect["Id"],
                    "image_id": inspect["Image"],
                    "argv": [
                        _redact_text(str(argument), sensitive_values)
                        for argument in argv
                    ],
                }
            )
    native_provenance = getattr(server, "native_provenance", None)
    if native_provenance:
        provenance.update(native_provenance)
    return provenance


def save_server_logs(server: ManagedServer, path: Path) -> None:
    """Append a complete Docker log snapshot without discarding prior attempts."""

    if server.backend == LLAMACPP_BACKEND:
        if server.process_log is not None:
            server.process_log.flush()
            os.fsync(server.process_log.fileno())
        return
    if not server.container_id:
        return
    result = _run(
        ["docker", "logs", server.container_id], check=False, timeout=30
    )
    payload = result.stdout
    if result.stderr:
        if payload and not payload.endswith("\n"):
            payload += "\n"
        payload += result.stderr
    sensitive_values = tuple(
        value for value in (server.api_key, server.authorization) if value
    )
    payload = _redact_text(payload, sensitive_values)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_existing = path.is_file() and path.stat().st_size > 0
    needs_newline = False
    if has_existing:
        with path.open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            needs_newline = existing.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as stream:
        if has_existing:
            if needs_newline:
                stream.write("\n")
            stream.write(
                f"--- SparkBench docker logs ({server.container_id}) ---\n"
            )
        stream.write(payload)
        if payload and not payload.endswith("\n"):
            stream.write("\n")
