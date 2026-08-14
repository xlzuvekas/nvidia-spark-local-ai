"""Safe lifecycle management for benchmark inference servers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request


MANAGED_LABEL = "ai.sparkbench.managed=true"
CONTAINER_NAME = "sparkbench-vllm"


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


def endpoint_ready(base_url: str, timeout_s: float = 2) -> bool:
    root = base_url.removesuffix("/v1").rstrip("/")
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(root + path, timeout=timeout_s) as response:
                if response.status < 400:
                    return True
        except (OSError, urllib.error.URLError, TimeoutError):
            continue
    return False


def wait_for_endpoint(base_url: str, timeout_s: float, container_id: str | None = None) -> float:
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if endpoint_ready(base_url):
            return time.monotonic() - started
        if container_id:
            state = _run(
                ["docker", "inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}", container_id],
                check=False,
            )
            if state.returncode or state.stdout.startswith("exited"):
                logs = _run(["docker", "logs", "--tail", "100", container_id], check=False)
                raise RuntimeErrorWithContext(
                    f"Server exited during startup: {state.stdout.strip()}\n{logs.stdout}{logs.stderr}"
                )
        time.sleep(2)
    raise RuntimeErrorWithContext(f"Server did not become ready within {timeout_s:.0f}s")


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _existing_container() -> tuple[str, bool, str] | None:
    result = _run(
        [
            "docker", "ps", "-a", "--filter", f"name=^{CONTAINER_NAME}$", "--format",
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

    def stop(self, *, keep_server: bool = False) -> None:
        if self.backend == "vllm" and self.container_id and not keep_server:
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


def start_vllm(
    model: Any,
    *,
    workspace: Path,
    port: int = 8000,
    allow_download: bool = False,
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
    except BaseException:
        server.stop()
        raise
    return server


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


def start_server(model: Any, *, workspace: Path, allow_download: bool = False) -> ManagedServer:
    if model.backend == "vllm":
        return start_vllm(model, workspace=workspace, allow_download=allow_download)
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
            provenance.update(
                {
                    "container_id": inspect["Id"],
                    "image_id": inspect["Image"],
                    "argv": inspect["Config"].get("Entrypoint", []) + inspect["Config"].get("Cmd", []),
                }
            )
    return provenance


def save_server_logs(server: ManagedServer, path: Path) -> None:
    if not server.container_id:
        return
    result = _run(["docker", "logs", server.container_id], check=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.stdout + result.stderr)
