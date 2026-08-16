"""Offline, process-isolated benchmark for Nemotron block diffusion generation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bench.journal import Journal, content_hash, utc_now, write_json
    from bench.report import _telemetry_summaries
    from bench.runner import _preflight, results_lock_path
    from bench.telemetry import TelemetrySampler
else:
    from .journal import Journal, content_hash, utc_now, write_json
    from .report import _telemetry_summaries
    from .runner import _preflight, results_lock_path
    from .telemetry import TelemetrySampler


DIRECT_MODEL_ID = "nemotron-labs-diffusion-14b-transformers-direct"
DIRECT_SOURCE = "nvidia/Nemotron-Labs-Diffusion-14B"
DIRECT_REVISION = "b69aaebbcfa95a7e5f1de36d6134e4c858ddbc97"
DIRECT_RUNTIME = "/home/xlz/AGENTIC-RESEARCH/diffusion-stuff/.venv/bin/python"
DIRECT_RUNTIME_LOCK = "/home/xlz/AGENTIC-RESEARCH/diffusion-stuff/uv.lock"
DIRECT_RUNTIME_CONFIG_SHA256 = (
    "31f65fe522e0e00574481aabf9d668ea1e72e0f91efb31f29a5132b16822ae64"
)
DIRECT_RUNTIME_LOCK_SHA256 = (
    "070d9faa986bfbecf8cb907e121a6fca14e91d175f4140a1b2df83730ca7b447"
)
DIRECT_RUNTIME_EXECUTABLE_SHA256 = (
    "6242e0e8650d7dbdebbc25e08bf4c9359ddaf65f54fcae0e57fa99395fa5357a"
)
DIRECT_ARGS = (
    "--block-length",
    "32",
    "--threshold",
    "0.9",
    "--seed",
    "3407",
)
DIRECT_RUNTIME_VERSIONS = {
    "python": "3.12.3",
    "torch": "2.11.0+cu128",
    "transformers": "5.9.0",
    "accelerate": "1.13.0",
}
DIRECT_PROMPT = (
    "Explain why the daytime sky appears blue. Be concise, continue until the "
    "generation limit, and do not use a list."
)
DIRECT_ARTIFACT_SHA256 = {
    "model.safetensors": (
        "272e78c39809711139deb08024b4fe8e6af83ab1316c8514bdfa35d7c880a320"
    ),
    "config.json": (
        "909d21d435993d75685cf2c70149d5f014ce55ffc69783be4faf5a912e3aea9f"
    ),
    "generation_config.json": (
        "c400f11f0da87584559a2d375c997b79bddfcde336b4085c626c1921ae7dfb6e"
    ),
    "configuration_nemotron_labs_diffusion.py": (
        "2b18216e1b4e0d89b728c1c871744088a28004564f99009809294b39ec677b57"
    ),
    "modeling_nemotron_labs_diffusion.py": (
        "29d73c5709e90e3be3c7e537edb61a84fbb3dc1c286b1eaed42d899a3a4e4760"
    ),
    "modeling_ministral.py": (
        "4f6df2d77a786a241c8c78346d22756ddb11d1a64d557a7111e8431c2095aa7c"
    ),
    "tokenizer.json": (
        "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7"
    ),
    "tokenizer_config.json": (
        "50d5c40a1e06a86e8d0fc2d4c6a9bf73ed0ffca7bdad3c9b42cedca48baa25dc"
    ),
    "special_tokens_map.json": (
        "e3a4f63da745f02317a45e00e6476c17fc66ac41faf14bb1b0be1f3211b0ca53"
    ),
    "chat_template.jinja": (
        "24901e3846b530e3ed20436b26ea1cd7b3768ab2b2645e31c47df1413ab289dc"
    ),
}


class DirectDiffusionError(RuntimeError):
    """Raised when the direct profile cannot be certified or executed safely."""


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    result: dict[str, Any] | None
    cleanup: dict[str, Any]
    error: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_path(model: Any) -> Path:
    if str(model.cache_dir) != "user":
        raise DirectDiffusionError("Direct diffusion requires the pinned user cache")
    repository = "models--" + str(model.source).replace("/", "--")
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / repository
        / "snapshots"
        / str(model.revision)
    )


def verify_direct_profile(model: Any) -> dict[str, Any]:
    """Hash every executable/model artifact before a CUDA process is started."""

    expected = {
        "id": DIRECT_MODEL_ID,
        "backend": "transformers",
        "source": DIRECT_SOURCE,
        "revision": DIRECT_REVISION,
        "runtime_python": DIRECT_RUNTIME,
        "args": DIRECT_ARGS,
    }
    actual = {
        "id": str(model.id),
        "backend": str(model.backend),
        "source": str(model.source),
        "revision": str(model.revision),
        "runtime_python": str(model.runtime_python),
        "args": tuple(model.args),
    }
    if actual != expected:
        raise DirectDiffusionError("Direct diffusion profile differs from the certified profile")

    runtime = Path(DIRECT_RUNTIME)
    runtime_config = runtime.parent.parent / "pyvenv.cfg"
    runtime_lock = Path(DIRECT_RUNTIME_LOCK)
    if (
        not runtime.is_file()
        or not runtime_config.is_file()
        or not runtime_lock.is_file()
    ):
        raise DirectDiffusionError("Certified direct-diffusion Python environment is missing")
    runtime_executable_hash = _sha256_file(runtime.resolve(strict=True))
    if runtime_executable_hash != DIRECT_RUNTIME_EXECUTABLE_SHA256:
        raise DirectDiffusionError("Direct-diffusion Python executable hash changed")
    runtime_config_hash = _sha256_file(runtime_config)
    if runtime_config_hash != DIRECT_RUNTIME_CONFIG_SHA256:
        raise DirectDiffusionError("Direct-diffusion Python environment hash changed")
    runtime_lock_hash = _sha256_file(runtime_lock)
    if runtime_lock_hash != DIRECT_RUNTIME_LOCK_SHA256:
        raise DirectDiffusionError("Direct-diffusion dependency lock hash changed")

    snapshot = _snapshot_path(model)
    if not snapshot.is_dir():
        raise DirectDiffusionError(f"Exact offline snapshot is missing: {snapshot}")
    actual_hashes: dict[str, str] = {}
    repository_root = snapshot.parents[1].resolve()
    for relative, expected_hash in DIRECT_ARTIFACT_SHA256.items():
        artifact = snapshot / relative
        if not artifact.is_file():
            raise DirectDiffusionError(f"Required artifact is missing: {artifact}")
        try:
            artifact.resolve(strict=True).relative_to(repository_root)
        except ValueError as error:
            raise DirectDiffusionError(
                f"Artifact symlink escapes its model cache: {relative}"
            ) from error
        actual_hash = _sha256_file(artifact)
        if actual_hash != expected_hash:
            raise DirectDiffusionError(f"Artifact hash mismatch: {relative}")
        actual_hashes[relative] = actual_hash
    logic_path = Path(__file__).resolve()
    return {
        "snapshot": str(snapshot),
        "artifacts_sha256": actual_hashes,
        "runtime_python": str(runtime),
        "runtime_executable_sha256": runtime_executable_hash,
        "runtime_config_sha256": runtime_config_hash,
        "runtime_lock": str(runtime_lock),
        "runtime_lock_sha256": runtime_lock_hash,
        "expected_runtime_versions": dict(DIRECT_RUNTIME_VERSIONS),
        "worker_logic": str(logic_path),
        "worker_logic_sha256": _sha256_file(logic_path),
    }


def _offline_worker_environment(
    runtime: Path, *, modules_cache: Path | None = None
) -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if "TOKEN" in name.upper() or name.upper() in {
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
        }:
            environment.pop(name, None)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "3407",
            "SPARKBENCH_PARENT_PID": str(os.getpid()),
            "PATH": f"{runtime.parent}:/usr/bin:/bin",
        }
    )
    if modules_cache is not None:
        modules_cache.mkdir(parents=True, exist_ok=True)
        environment["HF_MODULES_CACHE"] = str(modules_cache)
    return environment


def _invoke_worker(
    *,
    runtime: Path,
    config_path: Path,
    result_path: Path,
    log_dir: Path,
    timeout_s: float,
) -> WorkerOutcome:
    """Run and reap one isolated CUDA worker, killing it on the fixed deadline."""

    command = [
        str(runtime),
        "-I",
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        str(config_path),
        str(result_path),
    ]
    process: subprocess.Popen[str] | None = None
    stdout = ""
    stderr = ""
    timed_out = False
    terminate_requested = False
    kill_requested = False
    spawn_error: str | None = None
    process_started_at_ns: int | None = None
    process_start_ticks: int | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=_offline_worker_environment(
                runtime, modules_cache=log_dir / "hf_modules_cache"
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        process_started_at_ns = time.time_ns()
        try:
            process_start_ticks = int(
                Path(f"/proc/{process.pid}/stat").read_text().split()[21]
            )
        except (OSError, ValueError, IndexError):
            process_start_ticks = None
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_requested = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                kill_requested = True
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
    except BaseException as error:
        spawn_error = f"{type(error).__name__}: {error}"
        if process is not None and process.poll() is None:
            terminate_requested = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.communicate(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                kill_requested = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                process.communicate()
    finally:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "stdout.log").write_text(stdout)
        (log_dir / "stderr.log").write_text(stderr)

    returncode = process.returncode if process is not None else None
    reaped = process is None or process.poll() is not None
    cleanup = {
        "pid": process.pid if process is not None else None,
        "process_started_at_ns": process_started_at_ns,
        "process_start_ticks": process_start_ticks,
        "returncode": returncode,
        "timed_out": timed_out,
        "terminate_requested": terminate_requested,
        "kill_requested": kill_requested,
        "process_reaped": reaped,
        "cuda_context_cleanup": "worker_process_reaped" if reaped else "unverified",
    }
    result: dict[str, Any] | None = None
    result_error: str | None = spawn_error
    if result_path.is_file():
        try:
            loaded = json.loads(result_path.read_text())
            if isinstance(loaded, dict):
                result = loaded
            else:
                result_error = "worker result was not a JSON object"
        except (OSError, json.JSONDecodeError) as error:
            result_error = f"invalid worker result: {error}"
    elif result_error is None:
        result_error = "worker did not write its result artifact"
    if timed_out:
        result_error = f"worker exceeded its {timeout_s:g}s deadline"
    elif returncode not in {0, None} and result_error is None:
        result_error = f"worker exited with status {returncode}"
    return WorkerOutcome(result=result, cleanup=cleanup, error=result_error)


def _direct_summary(
    *,
    run_dir: Path,
    model: Any,
    suite: Any,
    verification: dict[str, Any],
    worker: dict[str, Any] | None,
    cleanup: dict[str, Any],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    worker_cases = (worker or {}).get("cases", []) if status == "complete" else []
    for case in worker_cases:
        requests = case.get("requests") or []
        output_tokens = sum(int(item["output_tokens"]) for item in requests)
        output_blocks = sum(int(item["output_blocks"]) for item in requests)
        total_nfe = sum(int(item["nfe"]) for item in requests)
        wall_s = float(case.get("measured_wall_time_s", 0))
        rates = [float(item["block_generation_output_tps"]) for item in requests]
        block_rates = [
            float(item["block_generation_blocks_per_s"]) for item in requests
        ]
        block_latencies = [
            float(item["mean_block_generation_latency_s"]) for item in requests
        ]
        rows.append(
            {
                "case_id": case.get("case_id"),
                "kind": "diffusion",
                "requests": len(requests),
                "output_tokens": output_tokens,
                "output_blocks": output_blocks,
                "measured_wall_time_s": wall_s,
                "median_block_generation_output_tps": (
                    statistics.median(rates) if rates else None
                ),
                "aggregate_block_generation_output_tps": (
                    output_tokens / max(wall_s, 1e-9) if requests else None
                ),
                "median_block_generation_blocks_per_s": (
                    statistics.median(block_rates) if block_rates else None
                ),
                "aggregate_block_generation_blocks_per_s": (
                    output_blocks / max(wall_s, 1e-9) if requests else None
                ),
                "median_mean_block_generation_latency_s": (
                    statistics.median(block_latencies) if block_latencies else None
                ),
                "nfe": total_nfe,
                "nfe_per_block": (
                    total_nfe / output_blocks if output_blocks else None
                ),
                "nfe_per_output_token": (
                    total_nfe / output_tokens if output_tokens else None
                ),
                "nfe_per_s": total_nfe / max(wall_s, 1e-9) if requests else None,
                "output_tokens_per_nfe": (
                    output_tokens / total_nfe if total_nfe else None
                ),
                "output_sha256": [item["output_sha256"] for item in requests],
                "outputs_stable": len(
                    {item["output_sha256"] for item in requests}
                )
                <= 1,
                "metric_source": (
                    "completion_tokens_per_end_to_end_block_generation_wall_time"
                ),
            }
        )
    return {
        "run_dir": str(run_dir),
        "status": status,
        "error": error,
        "model": {
            "id": model.id,
            "source": model.source,
            "revision": model.revision,
            "architecture": model.architecture,
            "backend": model.backend,
        },
        "suite": suite.id,
        "load_time_s": (worker or {}).get("load_time_s"),
        "runtime": (worker or {}).get("runtime"),
        "memory": (worker or {}).get("memory"),
        "artifact_verification": verification,
        "cleanup_proof": {
            **cleanup,
            "worker_cuda": (worker or {}).get("cleanup"),
        },
        "telemetry": _telemetry_summaries(run_dir / "telemetry.jsonl"),
        "cases": rows,
    }


def _validate_worker_result(
    worker: dict[str, Any], *, cases: list[dict[str, Any]], logic_hash: str
) -> None:
    """Reject incomplete or internally inconsistent measurements fail closed."""

    if worker.get("status") != "complete":
        raise DirectDiffusionError(str(worker.get("error") or "worker failed"))
    if worker.get("logic_sha256") != logic_hash:
        raise DirectDiffusionError("Worker logic hash differed from the frozen plan")
    load_time_s = worker.get("load_time_s")
    if not isinstance(load_time_s, (int, float)) or not (
        math.isfinite(float(load_time_s)) and float(load_time_s) > 0
    ):
        raise DirectDiffusionError("Worker model-load time was not positive and finite")
    runtime = worker.get("runtime")
    if not isinstance(runtime, dict) or any(
        str(runtime.get(name)) != expected
        for name, expected in DIRECT_RUNTIME_VERSIONS.items()
    ):
        raise DirectDiffusionError("Worker runtime versions differed from the certified profile")
    cleanup = worker.get("cleanup")
    if not isinstance(cleanup, dict) or not all(
        isinstance(cleanup.get(name), int) and cleanup[name] >= 0
        for name in (
            "allocated_bytes_after_model_delete",
            "reserved_bytes_after_empty_cache",
        )
    ):
        raise DirectDiffusionError("Worker did not provide CUDA cleanup measurements")
    measured_cases = worker.get("cases")
    if not isinstance(measured_cases, list) or len(measured_cases) != len(cases):
        raise DirectDiffusionError("Worker returned the wrong number of cases")
    for expected_case, measured_case in zip(cases, measured_cases, strict=True):
        if not isinstance(measured_case, dict) or (
            measured_case.get("case_id") != expected_case["case_id"]
        ):
            raise DirectDiffusionError("Worker case identity or order differed from the plan")
        requests = measured_case.get("requests")
        expected_repetitions = int(expected_case["repetitions"])
        if not isinstance(requests, list) or len(requests) != expected_repetitions:
            raise DirectDiffusionError("Worker returned the wrong repetition count")
        case_wall_s = measured_case.get("measured_wall_time_s")
        if not isinstance(case_wall_s, (int, float)) or not (
            math.isfinite(float(case_wall_s)) and float(case_wall_s) > 0
        ):
            raise DirectDiffusionError("Worker case wall time was not positive and finite")
        request_wall_total = 0.0
        for result in requests:
            if not isinstance(result, dict):
                raise DirectDiffusionError("Worker request result was not an object")
            expected_tokens = int(expected_case["max_output_tokens"])
            if (
                result.get("finish_reason") != "length"
                or result.get("output_tokens") != expected_tokens
                or result.get("completion_tokens") != expected_tokens
                or result.get("max_output_tokens") != expected_tokens
            ):
                raise DirectDiffusionError("Worker did not return the exact requested length")
            wall_s = result.get("wall_time_s")
            nfe = result.get("nfe")
            prompt_tokens = result.get("prompt_tokens")
            if not isinstance(wall_s, (int, float)) or not (
                math.isfinite(float(wall_s)) and float(wall_s) > 0
            ):
                raise DirectDiffusionError("Worker request wall time was invalid")
            if not isinstance(nfe, int) or nfe <= 0:
                raise DirectDiffusionError("Worker NFE was invalid")
            if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
                raise DirectDiffusionError("Worker prompt-token count was invalid")
            expected_rate = expected_tokens / float(wall_s)
            expected_blocks = expected_tokens // 32
            rate = result.get("block_generation_output_tps")
            if not isinstance(rate, (int, float)) or not math.isclose(
                float(rate), expected_rate, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise DirectDiffusionError("Worker block-generation rate was inconsistent")
            block_rate = result.get("block_generation_blocks_per_s")
            block_latency = result.get("mean_block_generation_latency_s")
            nfe_per_block = result.get("nfe_per_block")
            if result.get("output_blocks") != expected_blocks or not all(
                isinstance(value, (int, float))
                for value in (block_rate, block_latency, nfe_per_block)
            ):
                raise DirectDiffusionError("Worker block measurements were missing")
            if (
                not math.isclose(
                    float(block_rate),
                    expected_blocks / float(wall_s),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(block_latency),
                    float(wall_s) / expected_blocks,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(nfe_per_block),
                    nfe / expected_blocks,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ):
                raise DirectDiffusionError("Worker block measurements were inconsistent")
            nfe_per_token = result.get("nfe_per_output_token")
            tokens_per_nfe = result.get("output_tokens_per_nfe")
            if not isinstance(nfe_per_token, (int, float)) or not isinstance(
                tokens_per_nfe, (int, float)
            ):
                raise DirectDiffusionError("Worker NFE ratios were missing")
            if not math.isclose(
                float(nfe_per_token),
                nfe / expected_tokens,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ) or not math.isclose(
                float(tokens_per_nfe),
                expected_tokens / nfe,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise DirectDiffusionError("Worker NFE ratios were inconsistent")
            if (
                result.get("block_generation_metric_source")
                != "completion_tokens_per_end_to_end_block_generation_wall_time"
                or result.get("seed") != 3407
                or result.get("temperature") != 0.0
            ):
                raise DirectDiffusionError("Worker generation semantics differed from the plan")
            output_hash = result.get("output_sha256")
            if not isinstance(output_hash, str) or (
                len(output_hash) != 64
                or any(character not in "0123456789abcdef" for character in output_hash)
            ):
                raise DirectDiffusionError("Worker output hash was invalid")
            request_wall_total += float(wall_s)
        if float(case_wall_s) + 1e-6 < request_wall_total:
            raise DirectDiffusionError("Worker case wall time was shorter than request time")


def run_direct_diffusion(
    *,
    model: Any,
    suite: Any,
    workspace: Path,
    results_root: Path,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Run the certified profile under SparkBench's global matrix lock."""

    if model.backend != "transformers" or model.support_status != "spark_transformers_direct":
        raise DirectDiffusionError("Selected model is not the supported direct profile")
    if any(case.kind != "diffusion" for case in suite.cases):
        raise DirectDiffusionError("Direct diffusion requires a diffusion-only suite")
    deadline_s = float(timeout_s or model.startup_timeout_s)
    if not math.isfinite(deadline_s) or deadline_s <= 0:
        raise DirectDiffusionError("Worker timeout must be positive and finite")

    lock_path = results_lock_path(workspace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DirectDiffusionError(
                "Another SparkBench run holds results/.sparkbench.lock"
            ) from error

        verification = verify_direct_profile(model)
        logic_hash = verification["worker_logic_sha256"]
        plan_basis = {
            "model": asdict(model),
            "suite": asdict(suite),
            "verification": verification,
            "prompt_sha256": hashlib.sha256(DIRECT_PROMPT.encode()).hexdigest(),
            "worker_timeout_s": deadline_s,
        }
        fingerprint = content_hash(plan_basis, 64)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = results_root / f"{stamp}-{model.id}-{suite.id}-{fingerprint[:8]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        cases = [
            {
                **asdict(case),
                "case_id": (
                    f"{case.id}--"
                    f"{content_hash({'model': asdict(model), 'case': asdict(case)}, 12)}"
                ),
            }
            for case in suite.cases
        ]
        plan = {
            "schema_version": "direct-diffusion-v1",
            "created_at": utc_now(),
            "fingerprint": fingerprint,
            **plan_basis,
            "suite": {**asdict(suite), "cases": cases},
        }
        plan["integrity_hash"] = content_hash(plan, 64)
        write_json(run_dir / "plan.json", plan)
        journal = Journal(run_dir / "events.jsonl")
        journal.append({"event": "run_start", "mode": "direct_diffusion"})
        journal.append({"event": "artifact_verified", **verification})

        try:
            _preflight(model)
        except Exception as error:
            cleanup = {
                "pid": None,
                "returncode": None,
                "timed_out": False,
                "terminate_requested": False,
                "kill_requested": False,
                "process_reaped": True,
                "cuda_context_cleanup": "worker_not_started",
            }
            detail = f"{type(error).__name__}: {error}"
            journal.append({"event": "worker_cleanup", **cleanup})
            journal.append(
                {
                    "event": "run_aborted",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            summary = _direct_summary(
                run_dir=run_dir,
                model=model,
                suite=suite,
                verification=verification,
                worker=None,
                cleanup=cleanup,
                status="aborted",
                error=detail,
            )
            write_json(run_dir / "summary.json", summary)
            return summary
        worker_config = {
            "snapshot": verification["snapshot"],
            "logic_sha256": logic_hash,
            "prompt": DIRECT_PROMPT,
            "block_length": 32,
            "threshold": 0.9,
            "seed": 3407,
            "cases": cases,
        }
        config_path = run_dir / "worker" / "config.json"
        result_path = run_dir / "worker" / "result.json"
        write_json(config_path, worker_config)
        telemetry = TelemetrySampler(run_dir / "telemetry.jsonl")
        telemetry.start()
        telemetry.set_phase("direct_diffusion_worker")
        journal.append(
            {
                "event": "worker_start",
                "timeout_s": deadline_s,
                "worker_logic_sha256": logic_hash,
            }
        )
        try:
            outcome = _invoke_worker(
                runtime=Path(model.runtime_python),
                config_path=config_path,
                result_path=result_path,
                log_dir=run_dir / "worker",
                timeout_s=deadline_s,
            )
        finally:
            telemetry.stop()

        journal.append(
            {
                "event": "worker_cleanup",
                **outcome.cleanup,
                "worker_cuda": (outcome.result or {}).get("cleanup"),
            }
        )
        worker = outcome.result
        error = outcome.error
        if error is None and worker is not None:
            try:
                _validate_worker_result(worker, cases=cases, logic_hash=logic_hash)
            except DirectDiffusionError as validation_error:
                error = str(validation_error)
        if error is None and worker is not None:
            journal.append(
                {
                    "event": "model_loaded",
                    "load_time_s": worker.get("load_time_s"),
                    "runtime": worker.get("runtime"),
                    "memory": worker.get("memory"),
                }
            )
            for case in worker.get("cases", []):
                for index, result in enumerate(case.get("requests") or []):
                    journal.append(
                        {
                            "event": "request_complete",
                            "case_id": case["case_id"],
                            "kind": "diffusion",
                            "repetition": index,
                            "result": result,
                        }
                    )
                journal.append(
                    {
                        "event": "case_complete",
                        "case_id": case["case_id"],
                        "kind": "diffusion",
                        "elapsed_s": case.get("measured_wall_time_s"),
                        "validation_passed": True,
                    }
                )
            journal.append({"event": "run_complete", "status": "completed"})
            status = "complete"
        else:
            error = error or "worker returned no result"
            journal.append(
                {
                    "event": "run_aborted",
                    "error_type": "DirectDiffusionWorkerError",
                    "error": error,
                }
            )
            status = "aborted"
        summary = _direct_summary(
            run_dir=run_dir,
            model=model,
            suite=suite,
            verification=verification,
            worker=worker,
            cleanup=outcome.cleanup,
            status=status,
            error=error,
        )
        write_json(run_dir / "summary.json", summary)
        return summary


def _patch_causal_mask_kwargs() -> None:
    from transformers import masking_utils

    def shim(original: Any):
        def wrapper(*args: Any, **kwargs: Any):
            if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            kwargs.pop("cache_position", None)
            return original(*args, **kwargs)

        wrapper._nemotron_compat = True  # type: ignore[attr-defined]
        return wrapper

    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        original = getattr(masking_utils, name, None)
        if original is not None and not getattr(original, "_nemotron_compat", False):
            setattr(masking_utils, name, shim(original))


def _worker_run(config: dict[str, Any]) -> dict[str, Any]:
    """CUDA worker body; imported dependencies exist only in the certified venv."""

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    logic_sha256 = _sha256_file(Path(__file__).resolve())
    if logic_sha256 != config["logic_sha256"]:
        raise DirectDiffusionError("Worker logic changed after the plan was frozen")
    if not torch.cuda.is_available():
        raise DirectDiffusionError("CUDA is unavailable in the certified runtime")
    runtime_versions = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "accelerate": importlib.metadata.version("accelerate"),
    }
    if runtime_versions != DIRECT_RUNTIME_VERSIONS:
        raise DirectDiffusionError(
            f"Certified runtime versions changed: {runtime_versions}"
        )
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _patch_causal_mask_kwargs()

    snapshot = str(config["snapshot"])
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        snapshot,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    model = model.cuda().eval()
    torch.cuda.synchronize()
    load_time_s = time.perf_counter() - load_started
    load_peak = torch.cuda.max_memory_allocated()

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": str(config["prompt"])}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.cuda()
    torch.cuda.reset_peak_memory_stats()

    def generate(model_obj: Any, max_output_tokens: int) -> dict[str, Any]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        output_ids, nfe = model_obj.generate(
            prompt_ids,
            max_new_tokens=max_output_tokens,
            block_length=int(config["block_length"]),
            threshold=float(config["threshold"]),
            causal_context=True,
            temperature=0.0,
            eos_token_id=-1,
        )
        torch.cuda.synchronize()
        wall_time_s = time.perf_counter() - started
        new_ids = output_ids[0, prompt_ids.shape[1] :].detach().cpu().tolist()
        output_tokens = len(new_ids)
        output_blocks = output_tokens // int(config["block_length"])
        nfe_value = int(nfe.item() if hasattr(nfe, "item") else nfe)
        output_hash = hashlib.sha256(
            json.dumps(new_ids, separators=(",", ":")).encode()
        ).hexdigest()
        if output_tokens != max_output_tokens or nfe_value <= 0:
            raise DirectDiffusionError("Worker returned invalid output-token/NFE counts")
        return {
            "prompt_tokens": int(prompt_ids.shape[1]),
            "output_tokens": output_tokens,
            "completion_tokens": output_tokens,
            "output_blocks": output_blocks,
            "wall_time_s": wall_time_s,
            "elapsed_s": wall_time_s,
            "block_generation_output_tps": output_tokens / max(wall_time_s, 1e-9),
            "block_generation_blocks_per_s": (
                output_blocks / max(wall_time_s, 1e-9)
            ),
            "mean_block_generation_latency_s": (
                wall_time_s / output_blocks
            ),
            "block_generation_metric_source": (
                "completion_tokens_per_end_to_end_block_generation_wall_time"
            ),
            "nfe": nfe_value,
            "nfe_per_block": nfe_value / output_blocks,
            "nfe_per_output_token": nfe_value / output_tokens,
            "output_tokens_per_nfe": output_tokens / nfe_value,
            "nfe_per_s": nfe_value / max(wall_time_s, 1e-9),
            "output_sha256": output_hash,
            "max_output_tokens": max_output_tokens,
            "seed": seed,
            "temperature": 0.0,
            "finish_reason": "length",
        }

    case_results = []
    with torch.inference_mode():
        for case in config["cases"]:
            max_output_tokens = int(case["max_output_tokens"])
            for _ in range(int(case["warmups"])):
                generate(model, max_output_tokens)
            measured_started = time.perf_counter()
            requests = [
                generate(model, max_output_tokens)
                for _ in range(int(case["repetitions"]))
            ]
            case_results.append(
                {
                    "case_id": case["case_id"],
                    "requests": requests,
                    "measured_wall_time_s": time.perf_counter() - measured_started,
                }
            )
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    memory = {
        "load_peak_allocated_bytes": int(load_peak),
        "generation_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "generation_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "device_free_bytes_after_generation": int(free_bytes),
        "device_total_bytes": int(total_bytes),
    }
    result = {
        "status": "complete",
        "logic_sha256": logic_sha256,
        "load_time_s": load_time_s,
        "runtime": {
            **runtime_versions,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "seed": seed,
            "deterministic_greedy": True,
        },
        "memory": memory,
        "cases": case_results,
    }
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result["cleanup"] = {
        "allocated_bytes_after_model_delete": int(torch.cuda.memory_allocated()),
        "reserved_bytes_after_empty_cache": int(torch.cuda.memory_reserved()),
    }
    return result


def _worker_main(config_path: Path, result_path: Path) -> int:
    # Ensure an orchestrator crash cannot leave a CUDA worker occupying the Spark.
    if sys.platform.startswith("linux"):
        import ctypes

        try:
            parent_pid = int(os.environ["SPARKBENCH_PARENT_PID"])
        except (KeyError, ValueError) as error:
            raise DirectDiffusionError("Worker parent identity is missing") from error
        if os.getppid() != parent_pid:
            raise DirectDiffusionError("Worker parent exited before initialization")
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM) != 0:
            raise DirectDiffusionError("Could not install worker parent-death signal")
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGTERM)
    config = json.loads(config_path.read_text())
    try:
        result = _worker_run(config)
    except BaseException as error:
        result = {
            "status": "error",
            "logic_sha256": _sha256_file(Path(__file__).resolve()),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(limit=20),
        }
    write_json(result_path, result)
    return 0 if result["status"] == "complete" else 1


def _build_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("config", type=Path)
    parser.add_argument("result", type=Path)
    return parser


if __name__ == "__main__":
    arguments = _build_worker_parser().parse_args()
    if not arguments.worker:
        raise SystemExit("This module's executable mode is reserved for its worker")
    raise SystemExit(_worker_main(arguments.config, arguments.result))
