"""Safe lifecycle management for benchmark inference servers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import time
from typing import Any, TextIO
import urllib.error
import urllib.request
from urllib.parse import urlsplit


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
) -> float:
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if endpoint_ready(base_url, authorization=authorization):
            return time.monotonic() - started
        if container_id:
            state = _run(
                ["docker", "inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}", container_id],
                check=False,
            )
            if state.returncode or state.stdout.startswith("exited"):
                logs = _run(["docker", "logs", "--tail", "100", container_id], check=False)
                raise RuntimeErrorWithContext(
                    _redact_text(
                        f"Server exited during startup: {state.stdout.strip()}\n"
                        f"{logs.stdout}{logs.stderr}",
                        sensitive_values,
                    )
                )
        time.sleep(2)
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
    artifact_path = snapshot / filename
    try:
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

    def stop(self, *, keep_server: bool = False) -> None:
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
            inspect = _run(
                [
                    "docker", "inspect", "--format",
                    "{{index .Config.Labels \"ai.sparkbench.managed\"}} {{index .Config.Labels \"ai.sparkbench.run\"}}",
                    self.container_id,
                ],
                check=False,
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
                raise RuntimeErrorWithContext("Refusing to stop a container not owned by SparkBench")
            _run(["docker", "stop", "--time", "30", self.container_id], check=True, timeout=45)
            _run(["docker", "rm", self.container_id], check=True, timeout=15)
            self.container_id = None
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
) -> ManagedServer:
    """Start a digest-pinned SGLang server from exact cached snapshots."""

    # Runtime acquisition is always forbidden. A typed profile may permit only
    # the pinned image's documented metadata probe after both snapshots exist.
    del allow_download
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

    api_key = secrets.token_urlsafe(32)
    authorization = f"Bearer {api_key}"
    sensitive_values = (api_key, authorization)
    api_key_path = (
        server_log_path.parent / "api-key"
        if server_log_path is not None
        else None
    )
    if api_key_path is not None:
        _write_private_secret(api_key_path, api_key)

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
    command.extend(["--api-key", api_key])
    command.extend(str(argument) for argument in model.args)
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
        authorization=authorization,
        api_key=api_key,
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
        server.startup_s = wait_for_endpoint(
            server.base_url,
            float(model.startup_timeout_s),
            container_id,
            authorization=authorization,
            sensitive_values=sensitive_values,
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
) -> ManagedServer:
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
        return start_sglang(
            model,
            workspace=workspace,
            allow_download=allow_download,
            server_log_path=server_log_path,
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
        result = _run(["docker", "inspect", server.container_id], check=False)
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
    result = _run(["docker", "logs", server.container_id], check=False)
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
