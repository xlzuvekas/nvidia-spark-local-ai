"""Pinned, offline llama.cpp perplexity execution with durable provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import time
from typing import Any

from .journal import Journal, content_hash, utc_now, write_json
from .report import _telemetry_summaries
from .runner import _preflight, results_lock_path
from .telemetry import TelemetrySampler


PINNED_RUNTIME_SOURCE = Path(
    "/home/xlz/.cache/sparkbench/llama.cpp-b10453"
)
PINNED_RUNTIME_REVISION = "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70"
PINNED_PERPLEXITY_BINARY = (
    PINNED_RUNTIME_SOURCE / "build" / "bin" / "llama-perplexity"
)
PINNED_PERPLEXITY_BINARY_SIZE_BYTES = 51_551_520
PINNED_PERPLEXITY_BINARY_SHA256 = (
    "sha256:31ec19f4d8c071d691f7f4dde4a432771a50872eb29d87e9408a39f366ed5972"
)
PINNED_DATASET_SHA256 = (
    "sha256:173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"
)
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_FINAL_PPL_PATTERN = re.compile(
    rf"Final estimate:\s*PPL\s*=\s*({_FLOAT})\s*\+/-\s*({_FLOAT})",
    flags=re.IGNORECASE,
)
_ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ENV_ALLOWLIST = frozenset(
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
_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "LLAMA_ARG_OFFLINE": "1",
}
_TERMINATE_GRACE_S = 30.0


class LlamaCppPerplexityError(RuntimeError):
    """Raised when a perplexity run cannot be verified or completed."""


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    stdout: str
    stderr: str
    wall_time_s: float
    cleanup: dict[str, Any]
    error: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _model_payload(model: Any) -> dict[str, Any]:
    if is_dataclass(model):
        return asdict(model)
    try:
        return dict(vars(model))
    except TypeError as error:
        raise LlamaCppPerplexityError("Model profile is not serializable") from error


def _validate_parameters(
    model: Any, *, chunks: int, ctx_size: int, timeout_s: float
) -> None:
    if str(getattr(model, "backend", "")) != "llamacpp":
        raise LlamaCppPerplexityError(
            "Perplexity requires a backend=llamacpp model profile"
        )
    if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks <= 0:
        raise LlamaCppPerplexityError("chunks must be a positive integer")
    if isinstance(ctx_size, bool) or not isinstance(ctx_size, int) or ctx_size <= 0:
        raise LlamaCppPerplexityError("ctx-size must be a positive integer")
    context_limit = int(
        getattr(model, "native_context", None)
        or getattr(model, "max_context", 0)
        or 0
    )
    if context_limit <= 0 or ctx_size > context_limit:
        raise LlamaCppPerplexityError(
            f"ctx-size {ctx_size} exceeds the profile context limit {context_limit}"
        )
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
        raise LlamaCppPerplexityError("timeout must be positive and finite")
    if not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0:
        raise LlamaCppPerplexityError("timeout must be positive and finite")


def _validate_profile_shape(model: Any) -> None:
    source_dir = str(getattr(model, "runtime_source_dir", ""))
    revision = str(getattr(model, "runtime_revision", ""))
    if source_dir != str(PINNED_RUNTIME_SOURCE):
        raise LlamaCppPerplexityError(
            "llama.cpp runtime source path differs from the pinned b10453 build"
        )
    if revision != PINNED_RUNTIME_REVISION:
        raise LlamaCppPerplexityError(
            "llama.cpp runtime revision differs from pinned b10453"
        )
    model_revision = str(getattr(model, "revision", ""))
    if not _REVISION_PATTERN.fullmatch(model_revision):
        raise LlamaCppPerplexityError(
            "Model profile must pin a full lowercase commit revision"
        )
    source = str(getattr(model, "source", ""))
    if (
        source.count("/") != 1
        or any(part in {"", ".", ".."} for part in source.split("/"))
    ):
        raise LlamaCppPerplexityError("Model profile has an unsafe source ID")
    filename = str(getattr(model, "model_file", ""))
    relative = PurePosixPath(filename)
    if (
        not filename
        or relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name != filename
    ):
        raise LlamaCppPerplexityError("Model profile has an unsafe GGUF filename")
    digest = str(getattr(model, "model_digest", ""))
    size = getattr(model, "model_size_bytes", None)
    if not _DIGEST_PATTERN.fullmatch(digest):
        raise LlamaCppPerplexityError("Model profile must pin the GGUF SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise LlamaCppPerplexityError("Model profile must pin the GGUF byte size")


def _hub_root(
    model: Any, *, workspace: Path, home: Path | None = None
) -> Path:
    cache_dir = str(getattr(model, "cache_dir", ""))
    if cache_dir == "user":
        return (home or Path.home()) / ".cache" / "huggingface" / "hub"
    if cache_dir == "project":
        return workspace / "data" / "huggingface" / "hub"
    raise LlamaCppPerplexityError(
        "llama.cpp perplexity requires a user or project Hugging Face cache"
    )


def _expected_model_path(
    model: Any, *, workspace: Path, home: Path | None = None
) -> Path:
    repository = "models--" + str(model.source).replace("/", "--")
    return (
        _hub_root(model, workspace=workspace, home=home)
        / repository
        / "snapshots"
        / str(model.revision)
        / str(model.model_file)
    ).absolute()


def _verified_file(
    path: Path,
    *,
    label: str,
    expected_digest: str,
    expected_size: int | None = None,
    containment_root: Path | None = None,
    executable: bool = False,
) -> dict[str, Any]:
    try:
        target = path.resolve(strict=True)
    except OSError as error:
        raise LlamaCppPerplexityError(f"Pinned {label} is missing: {path}") from error
    if containment_root is not None:
        try:
            target.relative_to(containment_root.resolve(strict=True))
        except (OSError, ValueError) as error:
            raise LlamaCppPerplexityError(
                f"Pinned {label} escapes its certified root"
            ) from error
    if not target.is_file():
        raise LlamaCppPerplexityError(f"Pinned {label} is not a regular file")
    if executable and not os.access(target, os.X_OK):
        raise LlamaCppPerplexityError(f"Pinned {label} is not executable")
    size = target.stat().st_size
    if expected_size is not None and size != expected_size:
        raise LlamaCppPerplexityError(
            f"{label} size mismatch: expected {expected_size}, got {size}"
        )
    digest = _sha256_file(target)
    if digest != expected_digest:
        raise LlamaCppPerplexityError(
            f"{label} SHA-256 mismatch: expected {expected_digest}, got {digest}"
        )
    return {
        "path": str(path),
        "target": str(target),
        "size_bytes": size,
        "sha256": digest,
    }


def _source_revision(source_dir: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LlamaCppPerplexityError(
            "Could not verify the pinned llama.cpp source revision"
        ) from error
    revision = completed.stdout.strip()
    if completed.returncode or revision != PINNED_RUNTIME_REVISION:
        raise LlamaCppPerplexityError(
            "llama.cpp source checkout does not match pinned b10453"
        )
    return revision


def verify_perplexity_inputs(
    model: Any,
    *,
    workspace: Path,
    dataset: Path,
    home: Path | None = None,
) -> dict[str, Any]:
    """Verify runtime, GGUF, and dataset identities before inference starts."""

    _validate_profile_shape(model)
    try:
        source_target = PINNED_RUNTIME_SOURCE.resolve(strict=True)
    except OSError as error:
        raise LlamaCppPerplexityError(
            "Pinned llama.cpp source directory is missing"
        ) from error
    if not source_target.is_dir():
        raise LlamaCppPerplexityError(
            "Pinned llama.cpp source path is not a directory"
        )
    runtime = _verified_file(
        PINNED_PERPLEXITY_BINARY,
        label="llama-perplexity binary",
        expected_digest=PINNED_PERPLEXITY_BINARY_SHA256,
        expected_size=PINNED_PERPLEXITY_BINARY_SIZE_BYTES,
        containment_root=source_target,
        executable=True,
    )
    source_revision = _source_revision(source_target)

    model_path = _expected_model_path(
        model, workspace=workspace, home=home
    )
    repository = model_path.parents[2]
    model_artifact = _verified_file(
        model_path,
        label="GGUF",
        expected_digest=str(model.model_digest),
        expected_size=int(model.model_size_bytes),
        containment_root=repository,
    )
    dataset_artifact = _verified_file(
        dataset.expanduser().absolute(),
        label="perplexity dataset",
        expected_digest=PINNED_DATASET_SHA256,
    )
    if dataset_artifact["size_bytes"] <= 0:
        raise LlamaCppPerplexityError("Pinned perplexity dataset is empty")
    return {
        "runtime_source_dir": str(source_target),
        "runtime_source_revision": source_revision,
        "runtime_binary_path": runtime["path"],
        "runtime_binary_target": runtime["target"],
        "runtime_binary_size_bytes": runtime["size_bytes"],
        "runtime_binary_sha256": runtime["sha256"],
        "model_source": str(model.source),
        "model_revision": str(model.revision),
        "model_file": str(model.model_file),
        "model_path": model_artifact["path"],
        "model_target": model_artifact["target"],
        "model_size_bytes": model_artifact["size_bytes"],
        "model_sha256": model_artifact["sha256"],
        "dataset_path": dataset_artifact["path"],
        "dataset_target": dataset_artifact["target"],
        "dataset_size_bytes": dataset_artifact["size_bytes"],
        "dataset_sha256": dataset_artifact["sha256"],
    }


def _offline_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if key in _ENV_ALLOWLIST
    }
    environment.update(_OFFLINE_ENVIRONMENT)
    return environment


def _perplexity_command(
    *, binary: Path, model_path: Path, dataset: Path, chunks: int, ctx_size: int
) -> tuple[str, ...]:
    return (
        str(binary),
        "--model",
        str(model_path),
        "--file",
        str(dataset),
        "--chunks",
        str(chunks),
        "--ctx-size",
        str(ctx_size),
        "--n-gpu-layers",
        "all",
        "--flash-attn",
        "on",
        "--offline",
    )


def parse_final_perplexity(stdout: str, stderr: str) -> tuple[float, float]:
    """Parse the terminal llama.cpp PPL estimate without retaining logits."""

    output = _ANSI_PATTERN.sub("", stdout + "\n" + stderr)
    matches = list(_FINAL_PPL_PATTERN.finditer(output))
    if not matches:
        raise LlamaCppPerplexityError(
            "llama-perplexity output did not contain a final PPL estimate"
        )
    perplexity = float(matches[-1].group(1))
    uncertainty = float(matches[-1].group(2))
    if not math.isfinite(perplexity) or perplexity <= 0:
        raise LlamaCppPerplexityError(
            "llama-perplexity returned a non-positive or non-finite PPL"
        )
    if not math.isfinite(uncertainty) or uncertainty < 0:
        raise LlamaCppPerplexityError(
            "llama-perplexity returned an invalid uncertainty"
        )
    return perplexity, uncertainty


def _invoke_perplexity(
    *,
    command: tuple[str, ...],
    cwd: Path,
    timeout_s: float,
) -> ProcessOutcome:
    """Run llama-perplexity synchronously and reap its private process group."""

    process: subprocess.Popen[str] | None = None
    stdout = ""
    stderr = ""
    timed_out = False
    terminate_requested = False
    kill_requested = False
    process_error: str | None = None
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=_offline_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            if process.poll() is None:
                terminate_requested = True
                os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=_TERMINATE_GRACE_S)
            except subprocess.TimeoutExpired:
                if process.poll() is None:
                    kill_requested = True
                    os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
    except BaseException as error:
        process_error = f"{type(error).__name__}: {error}"
        if process is not None and process.poll() is None:
            terminate_requested = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
                stdout, stderr = process.communicate(
                    timeout=_TERMINATE_GRACE_S
                )
            except (OSError, subprocess.TimeoutExpired):
                kill_requested = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                stdout, stderr = process.communicate()
    wall_time_s = time.monotonic() - started
    returncode = process.returncode if process is not None else None
    process_reaped = process is None or process.poll() is not None
    cleanup = {
        "pid": process.pid if process is not None else None,
        "returncode": returncode,
        "timed_out": timed_out,
        "terminate_requested": terminate_requested,
        "kill_requested": kill_requested,
        "process_reaped": process_reaped,
        "process_group_isolated": process is not None,
    }
    error = process_error
    if timed_out:
        error = f"llama-perplexity exceeded its {timeout_s:g}s deadline"
    elif returncode not in {0, None} and error is None:
        error = f"llama-perplexity exited with status {returncode}"
    elif not process_reaped and error is None:
        error = "llama-perplexity process could not be reaped"
    return ProcessOutcome(
        stdout=stdout,
        stderr=stderr,
        wall_time_s=wall_time_s,
        cleanup=cleanup,
        error=error,
    )


def _empty_cleanup() -> dict[str, Any]:
    return {
        "pid": None,
        "returncode": None,
        "timed_out": False,
        "terminate_requested": False,
        "kill_requested": False,
        "process_reaped": True,
        "process_group_isolated": False,
    }


def _create_run_dir(
    results_root: Path, *, model_id: str, fingerprint: str
) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"{stamp}-{model_id}-perplexity-{fingerprint[:8]}"
    for ordinal in range(1_000):
        suffix = "" if ordinal == 0 else f"-{ordinal}"
        candidate = results_root / f"{stem}{suffix}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise LlamaCppPerplexityError("Could not allocate a unique result directory")


def run_llamacpp_perplexity(
    *,
    model: Any,
    workspace: Path,
    results_root: Path,
    dataset: Path,
    chunks: int,
    ctx_size: int = 512,
    timeout_s: float,
) -> dict[str, Any]:
    """Run one certified perplexity measurement under SparkBench's global lock."""

    _validate_parameters(
        model, chunks=chunks, ctx_size=ctx_size, timeout_s=timeout_s
    )
    _validate_profile_shape(model)
    workspace = workspace.resolve()
    dataset_path = dataset.expanduser().absolute()
    model_path = _expected_model_path(model, workspace=workspace)
    command = _perplexity_command(
        binary=PINNED_PERPLEXITY_BINARY,
        model_path=model_path,
        dataset=dataset_path,
        chunks=chunks,
        ctx_size=ctx_size,
    )
    basis = {
        "model": _model_payload(model),
        "mode": "llamacpp_perplexity",
        "parameters": {
            "chunks": chunks,
            "ctx_size": ctx_size,
            "timeout_s": float(timeout_s),
        },
        "expected_pins": {
            "runtime_source_dir": str(PINNED_RUNTIME_SOURCE),
            "runtime_source_revision": PINNED_RUNTIME_REVISION,
            "runtime_binary": str(PINNED_PERPLEXITY_BINARY),
            "runtime_binary_size_bytes": PINNED_PERPLEXITY_BINARY_SIZE_BYTES,
            "runtime_binary_sha256": PINNED_PERPLEXITY_BINARY_SHA256,
            "dataset_path": str(dataset_path),
            "dataset_sha256": PINNED_DATASET_SHA256,
        },
        "argv": list(command),
        "environment_policy": {
            "inherited_allowlist": sorted(_ENV_ALLOWLIST),
            "fixed": dict(_OFFLINE_ENVIRONMENT),
        },
    }
    fingerprint = content_hash(basis, 64)
    lock_path = results_lock_path(workspace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LlamaCppPerplexityError(
                "Another SparkBench run holds results/.sparkbench.lock"
            ) from error

        run_dir = _create_run_dir(
            results_root, model_id=str(model.id), fingerprint=fingerprint
        )
        plan = {
            "schema_version": "llamacpp-perplexity-v1",
            "created_at": utc_now(),
            "fingerprint": fingerprint,
            **basis,
        }
        plan["integrity_hash"] = content_hash(plan, 64)
        write_json(run_dir / "plan.json", plan)
        log_dir = run_dir / "logs"
        log_dir.mkdir()
        (log_dir / "stdout.log").write_text("", encoding="utf-8")
        (log_dir / "stderr.log").write_text("", encoding="utf-8")
        journal = Journal(run_dir / "events.jsonl")
        journal.append(
            {
                "event": "run_start",
                "mode": "llamacpp_perplexity",
                "fingerprint": fingerprint,
                "model_id": str(model.id),
            }
        )

        telemetry = TelemetrySampler(run_dir / "telemetry.jsonl")
        telemetry_started = False
        verification: dict[str, Any] = {}
        cleanup = _empty_cleanup()
        metrics: dict[str, Any] | None = None
        status = "aborted"
        error_text: str | None = None
        cleanup_recorded = False
        try:
            telemetry.start()
            telemetry_started = True
            telemetry.set_phase("artifact_validation")
            verification = verify_perplexity_inputs(
                model, workspace=workspace, dataset=dataset_path
            )
            journal.append({"event": "artifact_verified", **verification})
            _preflight(model)
            journal.append({"event": "preflight_complete"})

            telemetry.set_phase("llamacpp_perplexity")
            journal.append(
                {
                    "event": "process_start",
                    "chunks": chunks,
                    "ctx_size": ctx_size,
                    "timeout_s": float(timeout_s),
                    "dataset_sha256": verification["dataset_sha256"],
                }
            )
            outcome = _invoke_perplexity(
                command=command,
                cwd=run_dir,
                timeout_s=float(timeout_s),
            )
            cleanup = outcome.cleanup
            (log_dir / "stdout.log").write_text(
                outcome.stdout, encoding="utf-8"
            )
            (log_dir / "stderr.log").write_text(
                outcome.stderr, encoding="utf-8"
            )
            journal.append({"event": "process_cleanup", **cleanup})
            cleanup_recorded = True
            if outcome.error is not None:
                raise LlamaCppPerplexityError(outcome.error)
            if not math.isfinite(outcome.wall_time_s) or outcome.wall_time_s < 0:
                raise LlamaCppPerplexityError(
                    "llama-perplexity wall time was invalid"
                )
            perplexity, uncertainty = parse_final_perplexity(
                outcome.stdout, outcome.stderr
            )
            metrics = {
                "perplexity": perplexity,
                "uncertainty": uncertainty,
                "wall_time_s": outcome.wall_time_s,
                "chunks": chunks,
                "ctx_size": ctx_size,
                "metric_source": "llama.cpp_final_estimate",
            }
            journal.append({"event": "perplexity_complete", **metrics})
            journal.append({"event": "run_complete", "status": "completed"})
            status = "complete"
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            if not cleanup_recorded:
                journal.append({"event": "process_cleanup", **cleanup})
            journal.append(
                {
                    "event": "run_aborted",
                    "error_type": type(error).__name__,
                    "error": error_text,
                }
            )
        finally:
            if telemetry_started:
                try:
                    telemetry.stop()
                except Exception as telemetry_error:
                    journal.append(
                        {
                            "event": "telemetry_stop_failed",
                            "error_type": type(telemetry_error).__name__,
                        }
                    )

        result = {
            "schema_version": "llamacpp-perplexity-result-v1",
            "status": status,
            "model_id": str(model.id),
            "dataset_sha256": PINNED_DATASET_SHA256,
            "metrics": metrics,
            "cleanup": cleanup,
            "error": error_text,
        }
        write_json(run_dir / "result.json", result)
        summary = {
            "schema_version": "llamacpp-perplexity-summary-v1",
            "run_dir": str(run_dir),
            "status": status,
            "error": error_text,
            "model": {
                "id": str(model.id),
                "source": str(model.source),
                "revision": str(model.revision),
                "backend": str(model.backend),
                "quantization": str(getattr(model, "quantization", "")),
            },
            "configuration": {
                "chunks": chunks,
                "ctx_size": ctx_size,
                "timeout_s": float(timeout_s),
            },
            "metrics": metrics,
            "artifact_verification": verification,
            "cleanup_proof": cleanup,
            "telemetry": _telemetry_summaries(run_dir / "telemetry.jsonl"),
            "artifacts": {
                "plan": "plan.json",
                "journal": "events.jsonl",
                "result": "result.json",
                "stdout_log": "logs/stdout.log",
                "stderr_log": "logs/stderr.log",
                "telemetry": "telemetry.jsonl",
            },
        }
        write_json(run_dir / "summary.json", summary)
        return summary
