"""Offline TensorRT-LLM direct benchmark for Phi-4 speech transcription."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
from typing import Any
import unicodedata

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


DIRECT_MODEL_ID = "phi-4-multimodal-instruct-fp8-trtllm-audio"
DIRECT_SOURCE = "nvidia/Phi-4-multimodal-instruct-FP8"
DIRECT_REVISION = "d822efce23f65f86c165aeed435cc27092e21d60"
DIRECT_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13"
DIRECT_IMAGE_DIGEST = (
    "sha256:4f30c464ead64fb9727a24064b25057dacc07bef848022421108e544c91f0965"
)
DIRECT_ARGS = (
    "--backend",
    "pytorch",
    "--max-seq-len",
    "32768",
    "--max-batch-size",
    "1",
    "--max-num-tokens",
    "8192",
    "--disable-block-reuse",
    "--seed",
    "3407",
)
DIRECT_KV_CACHE_OPTIONS = (
    ("free_gpu_memory_fraction", 0.6),
    ("enable_block_reuse", False),
)
DIRECT_AUDIO_PATH = Path(
    "/home/xlz/voice-cloning/Spark-TTS/example/prompt_audio.wav"
)
DIRECT_AUDIO_SHA256 = (
    "335e7f7789b231cd90d9670292d561ecfe6a6bdd5e737a7bc6c29730741852de"
)
DIRECT_AUDIO_SIZE = 318_550
DIRECT_AUDIO_DURATION_S = 9.953313
DIRECT_PROMPT = (
    "Transcribe the audio clip into text, please don't add other text."
)
DIRECT_EXPECTED_TRANSCRIPTION = (
    "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。"
    "豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"
)
DIRECT_SMALL_ARTIFACT_SHA256 = {
    "config.json": "bf2609faab7f34f8203494c7904914fa0db2143c8588c9a3e0a147bac32cb4dc",
    "hf_quant_config.json": "4b604b2cce5f26c3056402a6275d86fc2115b3a3ebc0af4ee5d3fdd34bbd887c",
    "model.safetensors.index.json": "604050d50efb26ca00bc48359ffe4c449e782d3853946a44466cc5b8fd66c047",
    "modeling_phi4mm.py": "e2b44eb7a66d6cc54524cee1ff9ba92d0658d435ea8900329ea0dbdb85c6439d",
    "processing_phi4mm.py": "84914d3e12256b4e2186e040c9830c11408468b6774f42afe85e6f8de2626d50",
    "speech_conformer_encoder.py": "3742827e945732cc5deea4a95e14004da037044431a94e3f3fac26239e614e3a",
    "tokenizer.json": "4c1b9f641d4f8b7247b8d5007dd3b6a9f6a87cb5123134fe0d326f14d10c0585",
    "tokenizer_config.json": "733c9322d8f592273541e98c989da896d4527a1d9177d26d189bd0455a192083",
    "speech-lora/adapter_config.json": "ed252a6ae210888ee69f5720bd7e8d8261f0abfda90b18ea6452316c71336df8",
}
DIRECT_LFS_ARTIFACTS = {
    "model-00001-of-00002.safetensors": (
        "a59dcd3fb4d22e586ff2e1790e07c0e4bf7161191854003f3053fc252b93dd26",
        4_997_181_968,
    ),
    "model-00002-of-00002.safetensors": (
        "a91607c21bda79b3230c0e51927cebcef6df5d210e75dc749a7e97165a67809a",
        3_669_049_232,
    ),
    "speech-lora/adapter_model.safetensors": (
        "1c2237461a4d1f9292cd128147bd3f0f70326a48d5d79c8e0f7583b26c095b30",
        922_782_296,
    ),
}
_MANAGED_LABEL = "io.sparkbench.managed"
_RUN_LABEL = "io.sparkbench.run_id"
_PROFILE_LABEL = "io.sparkbench.profile"


class DirectTrtllmError(RuntimeError):
    """Raised when the pinned direct profile cannot run safely."""


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


def _normalize_transcription(content: str) -> str:
    normalized = unicodedata.normalize("NFKC", content)
    return "".join(character for character in normalized if character.isalnum()).casefold()


def _character_edit_distance(actual: str, expected: str) -> int:
    """Return Levenshtein distance without retaining an edit matrix."""

    if len(actual) < len(expected):
        actual, expected = expected, actual
    previous = list(range(len(expected) + 1))
    for actual_index, actual_character in enumerate(actual, start=1):
        current = [actual_index]
        for expected_index, expected_character in enumerate(expected, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[expected_index] + 1,
                    previous[expected_index - 1]
                    + (actual_character != expected_character),
                )
            )
        previous = current
    return previous[-1]


def _repository_path(model: Any) -> Path:
    if str(model.cache_dir) != "user":
        raise DirectTrtllmError("TRT-LLM direct requires the pinned user cache")
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / ("models--" + str(model.source).replace("/", "--"))
    )


def _snapshot_path(model: Any) -> Path:
    return _repository_path(model) / "snapshots" / str(model.revision)


def _verify_artifact_inside_repository(artifact: Path, repository: Path) -> Path:
    if not artifact.is_file():
        raise DirectTrtllmError(f"Required cached artifact is missing: {artifact.name}")
    try:
        resolved = artifact.resolve(strict=True)
        resolved.relative_to(repository.resolve(strict=True))
    except ValueError as error:
        raise DirectTrtllmError(
            f"Cached artifact symlink escapes its repository: {artifact.name}"
        ) from error
    return resolved


def verify_direct_profile(model: Any) -> dict[str, Any]:
    """Certify the manifest, model cache, fixture, and worker before Docker."""

    expected = {
        "id": DIRECT_MODEL_ID,
        "backend": "trtllm",
        "support_status": "spark_trtllm_direct",
        "source": DIRECT_SOURCE,
        "revision": DIRECT_REVISION,
        "image": DIRECT_IMAGE,
        "image_digest": DIRECT_IMAGE_DIGEST,
        "tasks": ("chat", "audio"),
        "endpoint": "offline://trtllm",
        "args": DIRECT_ARGS,
    }
    actual = {
        "id": str(model.id),
        "backend": str(model.backend),
        "support_status": str(model.support_status),
        "source": str(model.source),
        "revision": str(model.revision),
        "image": str(model.image),
        "image_digest": str(model.image_digest),
        "tasks": tuple(model.tasks),
        "endpoint": str(model.endpoint),
        "args": tuple(model.args),
    }
    if actual != expected:
        raise DirectTrtllmError("TRT-LLM direct profile differs from its certified shape")

    repository = _repository_path(model)
    snapshot = _snapshot_path(model)
    if not snapshot.is_dir():
        raise DirectTrtllmError("Exact offline Phi snapshot is missing")
    small_hashes: dict[str, str] = {}
    for relative, expected_hash in DIRECT_SMALL_ARTIFACT_SHA256.items():
        artifact = snapshot / relative
        _verify_artifact_inside_repository(artifact, repository)
        actual_hash = _sha256_file(artifact)
        if actual_hash != expected_hash:
            raise DirectTrtllmError(f"Cached artifact hash mismatch: {relative}")
        small_hashes[relative] = actual_hash

    lfs_blobs: dict[str, dict[str, Any]] = {}
    for relative, (expected_blob, expected_size) in DIRECT_LFS_ARTIFACTS.items():
        resolved = _verify_artifact_inside_repository(snapshot / relative, repository)
        if resolved.name != expected_blob or resolved.stat().st_size != expected_size:
            raise DirectTrtllmError(f"Cached LFS artifact mismatch: {relative}")
        if _sha256_file(resolved) != expected_blob:
            raise DirectTrtllmError(f"Cached LFS artifact hash mismatch: {relative}")
        lfs_blobs[relative] = {"sha256": expected_blob, "bytes": expected_size}

    if (
        not DIRECT_AUDIO_PATH.is_file()
        or DIRECT_AUDIO_PATH.stat().st_size != DIRECT_AUDIO_SIZE
        or _sha256_file(DIRECT_AUDIO_PATH) != DIRECT_AUDIO_SHA256
    ):
        raise DirectTrtllmError("Pinned audio fixture is missing or changed")
    logic_path = Path(__file__).resolve()
    return {
        "model_revision": DIRECT_REVISION,
        "image": DIRECT_IMAGE,
        "image_digest": DIRECT_IMAGE_DIGEST,
        "small_artifacts_sha256": small_hashes,
        "lfs_artifacts": lfs_blobs,
        "audio_sha256": DIRECT_AUDIO_SHA256,
        "audio_bytes": DIRECT_AUDIO_SIZE,
        "audio_duration_s": DIRECT_AUDIO_DURATION_S,
        "prompt_sha256": hashlib.sha256(DIRECT_PROMPT.encode()).hexdigest(),
        "expected_transcription_sha256": hashlib.sha256(
            _normalize_transcription(DIRECT_EXPECTED_TRANSCRIPTION).encode()
        ).hexdigest(),
        "worker_logic_sha256": _sha256_file(logic_path),
    }


def _validate_suite(suite: Any) -> None:
    if str(suite.id) != "audio-asr" or len(suite.cases) != 1:
        raise DirectTrtllmError("TRT-LLM direct requires the pinned audio-asr suite")
    case = suite.cases[0]
    expected = {
        "id": "spark-tts-chinese-exact",
        "kind": "capability",
        "requires": ("chat", "audio"),
        "warmups": 1,
        "repetitions": 3,
        "concurrency": 1,
        "prompt_repetitions": 0,
        "max_output_tokens": 128,
        "temperature": 0.0,
    }
    actual = {name: getattr(case, name) for name in expected}
    actual["requires"] = tuple(actual["requires"])
    if actual != expected:
        raise DirectTrtllmError("Audio suite differs from the certified workload shape")


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        upper = name.upper()
        if "TOKEN" in upper or upper in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}:
            environment.pop(name, None)
    return environment


def _docker_command(
    *, model: Any, run_id: str, worker_dir: Path, repository: Path
) -> tuple[list[str], Path, dict[str, str]]:
    cidfile = worker_dir / "container.cid"
    name = f"sparkbench-trtllm-{run_id[:12]}"
    labels = {
        _MANAGED_LABEL: "true",
        _RUN_LABEL: run_id,
        _PROFILE_LABEL: str(model.id),
    }
    command = [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--platform",
        "linux/arm64",
        "--gpus",
        "all",
        "--read-only",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--shm-size",
        "8g",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,nodev,size=8g",
        "--tmpfs",
        "/root/.cache:rw,exec,nosuid,nodev,size=16g",
        "--name",
        name,
        "--cidfile",
        str(cidfile),
    ]
    for key, value in labels.items():
        command.extend(["--label", f"{key}={value}"])
    command.extend(
        [
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "TRANSFORMERS_OFFLINE=1",
            "--env",
            "HF_HUB_DISABLE_TELEMETRY=1",
            "--env",
            "TOKENIZERS_PARALLELISM=false",
            "--env",
            "PYTHONHASHSEED=3407",
            # TensorRT-LLM compiles Triton/CUDA kernels during warmup. Keep
            # those caches on the existing ephemeral /tmp tmpfs so the image
            # root filesystem can remain read-only.
            "--env",
            "TRITON_CACHE_DIR=/tmp/triton-cache",
            "--env",
            "CUDA_CACHE_PATH=/tmp/cuda-cache",
            "--env",
            "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-cache",
            "--volume",
            f"{repository}:/model_repo:ro",
            "--volume",
            f"{DIRECT_AUDIO_PATH}:/inputs/prompt_audio.wav:ro",
            "--volume",
            f"{Path(__file__).resolve().parents[1]}:/sparkbench:ro",
            "--volume",
            f"{worker_dir}:/output:rw",
            "--workdir",
            "/sparkbench",
            f"{model.image}@{model.image_digest}",
            # Keep NVIDIA's image entrypoint: it adds the TensorRT libraries
            # (including libnvonnxparser) to LD_LIBRARY_PATH before exec.
            "python3",
            "-I",
            "-u",
            "/sparkbench/bench/trtllm_direct.py",
            "--worker",
            "/output/config.json",
            "/output/result.json",
        ]
    )
    return command, cidfile, labels


def _redact_sensitive(value: str) -> str:
    redacted = value.replace(str(DIRECT_AUDIO_PATH), "[audio fixture]")
    redacted = redacted.replace(
        DIRECT_EXPECTED_TRANSCRIPTION, "[transcription omitted]"
    )
    redacted = re.sub(
        r"data:audio/[^\s\"']+", "[audio data omitted]", redacted, flags=re.I
    )
    redacted = re.sub(
        r"(?i)base64,[A-Za-z0-9+/=_-]+", "base64,[omitted]", redacted
    )
    return redacted


def _run_docker_control(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_worker_environment(),
    )


def _cleanup_owned_container(
    cidfile: Path, expected_labels: dict[str, str]
) -> dict[str, Any]:
    cleanup: dict[str, Any] = {
        "cid": None,
        "container_found": False,
        "ownership_verified": False,
        "stop_requested": False,
        "remove_requested": False,
        "cleanup_verified": False,
    }
    try:
        cid = cidfile.read_text().strip()
    except OSError:
        return cleanup
    if not re.fullmatch(r"[0-9a-f]{12,64}", cid):
        cleanup["cidfile_invalid"] = True
        return cleanup
    cleanup["cid"] = cid
    inspected = _run_docker_control(
        ["docker", "inspect", "--format", "{{json .Config.Labels}}", cid]
    )
    if inspected.returncode != 0:
        cleanup["container_absent"] = True
        cleanup["cleanup_verified"] = True
        return cleanup
    cleanup["container_found"] = True
    try:
        labels = json.loads(inspected.stdout)
    except json.JSONDecodeError:
        cleanup["inspect_invalid"] = True
        return cleanup
    if not isinstance(labels, dict) or any(
        labels.get(key) != value for key, value in expected_labels.items()
    ):
        cleanup["ownership_mismatch"] = True
        return cleanup
    cleanup["ownership_verified"] = True
    stopped = _run_docker_control(["docker", "stop", "--time", "10", cid])
    cleanup["stop_requested"] = True
    cleanup["stop_returncode"] = stopped.returncode
    after_stop = _run_docker_control(["docker", "inspect", cid])
    if after_stop.returncode == 0:
        removed = _run_docker_control(["docker", "rm", "--force", cid])
        cleanup["remove_requested"] = True
        cleanup["remove_returncode"] = removed.returncode
        after_remove = _run_docker_control(["docker", "inspect", cid])
        cleanup["cleanup_verified"] = after_remove.returncode != 0
    else:
        cleanup["container_absent_after_stop"] = True
        cleanup["cleanup_verified"] = True
    return cleanup


def _invoke_worker(
    *,
    command: list[str],
    cidfile: Path,
    labels: dict[str, str],
    result_path: Path,
    log_dir: Path,
    timeout_s: float,
) -> WorkerOutcome:
    """Run and reap Docker plus its owned container on every exit path."""

    process: subprocess.Popen[str] | None = None
    stdout = ""
    stderr = ""
    timed_out = False
    terminate_requested = False
    kill_requested = False
    spawn_error: str | None = None
    container_cleanup: dict[str, Any] = {}
    try:
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=_worker_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            container_cleanup = _cleanup_owned_container(cidfile, labels)
            if process.poll() is None:
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
        final_cleanup = _cleanup_owned_container(cidfile, labels)
        if final_cleanup.get("container_found") or not container_cleanup:
            container_cleanup = final_cleanup
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "stdout.log").write_text(_redact_sensitive(stdout))
        (log_dir / "stderr.log").write_text(_redact_sensitive(stderr))

    returncode = process.returncode if process is not None else None
    reaped = process is None or process.poll() is not None
    cleanup = {
        "pid": process.pid if process is not None else None,
        "returncode": returncode,
        "timed_out": timed_out,
        "terminate_requested": terminate_requested,
        "kill_requested": kill_requested,
        "process_reaped": reaped,
        "container": container_cleanup,
    }
    result: dict[str, Any] | None = None
    error = spawn_error
    if result_path.is_file():
        try:
            loaded = json.loads(result_path.read_text())
            if isinstance(loaded, dict):
                result = loaded
            else:
                error = "worker result was not a JSON object"
        except (OSError, json.JSONDecodeError) as parse_error:
            error = f"invalid worker result: {parse_error}"
    elif error is None:
        error = "worker did not write its result artifact"
    if timed_out:
        error = f"worker exceeded its {timeout_s:g}s deadline"
    elif returncode not in {0, None} and error is None:
        error = f"worker exited with status {returncode}"
    if (
        container_cleanup.get("container_found")
        and not container_cleanup.get("cleanup_verified")
    ):
        error = "owned container cleanup could not be verified"
    return WorkerOutcome(
        result=result,
        cleanup=cleanup,
        error=_redact_sensitive(error) if error else None,
    )


def _validate_worker_result(
    worker: dict[str, Any], *, case: dict[str, Any], logic_hash: str
) -> None:
    if worker.get("status") != "complete":
        raise DirectTrtllmError(str(worker.get("error") or "worker failed"))
    if worker.get("logic_sha256") != logic_hash:
        raise DirectTrtllmError("Worker logic hash differed from the frozen plan")
    for name in ("load_time_s", "input_prepare_time_s"):
        value = worker.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise DirectTrtllmError(f"Worker {name} was not positive and finite")
    if worker.get("audio_sha256") != DIRECT_AUDIO_SHA256:
        raise DirectTrtllmError("Worker audio fixture identity differed")
    if worker.get("case_id") != case["case_id"]:
        raise DirectTrtllmError("Worker case identity differed")
    if worker.get("warmups_completed") != 1:
        raise DirectTrtllmError("Worker did not complete exactly one warmup")
    measured = worker.get("requests")
    if not isinstance(measured, list) or len(measured) != int(case["repetitions"]):
        raise DirectTrtllmError("Worker did not return exactly three measured trials")
    expected_normalized_hash = hashlib.sha256(
        _normalize_transcription(DIRECT_EXPECTED_TRANSCRIPTION).encode()
    ).hexdigest()
    expected_normalized_length = len(
        _normalize_transcription(DIRECT_EXPECTED_TRANSCRIPTION)
    )

    def validate_trial(result: Any, repetition: int) -> None:
        if not isinstance(result, dict) or result.get("repetition") != repetition:
            raise DirectTrtllmError("Worker trial identity or order differed")
        elapsed_s = result.get("elapsed_s")
        tokens = result.get("completion_tokens")
        rate = result.get("output_tps")
        if not isinstance(elapsed_s, (int, float)) or not (
            math.isfinite(float(elapsed_s)) and float(elapsed_s) > 0
        ):
            raise DirectTrtllmError("Worker trial latency was invalid")
        if type(tokens) is not int or not 0 < tokens <= int(case["max_output_tokens"]):
            raise DirectTrtllmError("Worker completion-token count was invalid")
        if not isinstance(rate, (int, float)) or not math.isclose(
            float(rate), tokens / float(elapsed_s), rel_tol=1e-9, abs_tol=1e-12
        ):
            raise DirectTrtllmError("Worker output rate was inconsistent")
        transcription_exact = result.get("transcription_exact")
        if not isinstance(transcription_exact, bool):
            raise DirectTrtllmError("Worker transcription accuracy flag was invalid")
        normalized_hash = result.get("normalized_transcription_sha256")
        if not isinstance(normalized_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", normalized_hash
        ):
            raise DirectTrtllmError("Worker normalized transcription hash was invalid")
        if result.get("expected_transcription_sha256") != expected_normalized_hash:
            raise DirectTrtllmError("Worker expected-transcription hash differed")
        normalized_length = result.get("normalized_output_length")
        edit_distance = result.get("character_edit_distance")
        error_rate = result.get("character_error_rate")
        if type(normalized_length) is not int or normalized_length < 0:
            raise DirectTrtllmError("Worker normalized output length was invalid")
        if type(edit_distance) is not int or edit_distance < 0:
            raise DirectTrtllmError("Worker character edit distance was invalid")
        if (
            not isinstance(error_rate, (int, float))
            or isinstance(error_rate, bool)
            or not (math.isfinite(float(error_rate)) and float(error_rate) >= 0)
        ):
            raise DirectTrtllmError("Worker character error rate was invalid")
        if not (
            abs(normalized_length - expected_normalized_length)
            <= edit_distance
            <= max(normalized_length, expected_normalized_length)
        ):
            raise DirectTrtllmError("Worker character edit distance was impossible")
        if not math.isclose(
            float(error_rate),
            edit_distance / expected_normalized_length,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise DirectTrtllmError("Worker character error rate was inconsistent")
        hash_exact = normalized_hash == expected_normalized_hash
        metric_exact = (
            normalized_length == expected_normalized_length
            and edit_distance == 0
            and float(error_rate) == 0.0
        )
        if transcription_exact != hash_exact or transcription_exact != metric_exact:
            raise DirectTrtllmError(
                "Worker transcription accuracy fields were inconsistent"
            )
        for hash_name in ("output_sha256", "token_ids_sha256"):
            value = result.get(hash_name)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise DirectTrtllmError(f"Worker {hash_name} was invalid")
        if "length" in str(result.get("finish_reason", "")).lower():
            raise DirectTrtllmError("Worker transcription exhausted the output limit")

    validate_trial(worker.get("warmup"), -1)
    for index, result in enumerate(measured):
        validate_trial(result, index)
    measured_wall_s = worker.get("measured_wall_s")
    if not isinstance(measured_wall_s, (int, float)) or not math.isfinite(
        float(measured_wall_s)
    ):
        raise DirectTrtllmError("Worker measured wall time was invalid")
    if float(measured_wall_s) < sum(float(item["elapsed_s"]) for item in measured):
        raise DirectTrtllmError("Worker measured wall time was internally inconsistent")


def _summary(
    *,
    run_dir: Path,
    model: Any,
    suite: Any,
    verification: dict[str, Any],
    worker: dict[str, Any] | None,
    cleanup: dict[str, Any],
    status: str,
    error: str | None,
) -> dict[str, Any]:
    requests = (worker or {}).get("requests") or []
    latencies = [float(item["elapsed_s"]) for item in requests]
    rates = [float(item["output_tps"]) for item in requests]
    output_lengths = [int(item["normalized_output_length"]) for item in requests]
    edit_distances = [int(item["character_edit_distance"]) for item in requests]
    error_rates = [float(item["character_error_rate"]) for item in requests]
    validation_passed = bool(requests) and all(
        item.get("transcription_exact") is True for item in requests
    )
    return {
        "run_dir": str(run_dir),
        "status": status,
        "error": _redact_sensitive(error) if error else None,
        "model": {
            "id": model.id,
            "source": model.source,
            "revision": model.revision,
            "backend": model.backend,
            "image": model.image,
            "image_digest": model.image_digest,
        },
        "suite": suite.id,
        "load_time_s": (worker or {}).get("load_time_s"),
        "input_prepare_time_s": (worker or {}).get("input_prepare_time_s"),
        "runtime": (worker or {}).get("runtime"),
        "artifact_verification": verification,
        "cleanup_proof": cleanup,
        "telemetry": _telemetry_summaries(run_dir / "telemetry.jsonl"),
        "cases": [
            {
                "case_id": (worker or {}).get("case_id"),
                "kind": "capability",
                "requests": len(requests),
                "warmups": (worker or {}).get("warmups_completed", 0),
                "validation_passed": validation_passed,
                "exact_matches": sum(
                    item.get("transcription_exact") is True for item in requests
                ),
                "median_elapsed_s": statistics.median(latencies) if latencies else None,
                "median_output_tps": statistics.median(rates) if rates else None,
                "median_normalized_output_length": (
                    statistics.median(output_lengths) if output_lengths else None
                ),
                "median_character_edit_distance": (
                    statistics.median(edit_distances) if edit_distances else None
                ),
                "median_character_error_rate": (
                    statistics.median(error_rates) if error_rates else None
                ),
                "audio_duration_s": DIRECT_AUDIO_DURATION_S,
                "median_realtime_factor": (
                    statistics.median(latencies) / DIRECT_AUDIO_DURATION_S
                    if latencies
                    else None
                ),
            }
        ],
    }


def run_direct_trtllm(
    *,
    model: Any,
    suite: Any,
    workspace: Path,
    results_root: Path,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Run the pinned direct adapter under SparkBench's global lock."""

    _validate_suite(suite)
    deadline_s = float(timeout_s or model.startup_timeout_s)
    if not math.isfinite(deadline_s) or deadline_s <= 0:
        raise DirectTrtllmError("Worker timeout must be positive and finite")
    lock_path = results_lock_path(workspace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DirectTrtllmError(
                "Another SparkBench run holds results/.sparkbench.lock"
            ) from error

        verification = verify_direct_profile(model)
        case = asdict(suite.cases[0])
        case["case_id"] = (
            f"{case['id']}--"
            f"{content_hash({'model': asdict(model), 'case': case}, 12)}"
        )
        basis = {
            "model": asdict(model),
            "suite": {**asdict(suite), "cases": [case]},
            "verification": verification,
            "worker_timeout_s": deadline_s,
        }
        fingerprint = content_hash(basis, 64)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{model.id}-{fingerprint[:12]}"
        run_dir = results_root / f"{stamp}-{model.id}-{suite.id}-{fingerprint[:8]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        plan = {
            "schema_version": "trtllm-direct-v1",
            "created_at": utc_now(),
            "fingerprint": fingerprint,
            **basis,
        }
        plan["integrity_hash"] = content_hash(plan, 64)
        write_json(run_dir / "plan.json", plan)
        journal = Journal(run_dir / "events.jsonl")
        journal.append({"event": "run_start", "mode": "trtllm_direct"})
        journal.append({"event": "artifact_verified", **verification})

        empty_cleanup = {
            "pid": None,
            "returncode": None,
            "timed_out": False,
            "terminate_requested": False,
            "kill_requested": False,
            "process_reaped": True,
            "container": {"container_found": False},
        }
        try:
            _preflight(model)
        except Exception as error:
            safe_error = _redact_sensitive(f"{type(error).__name__}: {error}")
            journal.append({"event": "worker_cleanup", **empty_cleanup})
            journal.append(
                {"event": "run_aborted", "error_type": type(error).__name__, "error": safe_error}
            )
            summary = _summary(
                run_dir=run_dir,
                model=model,
                suite=suite,
                verification=verification,
                worker=None,
                cleanup=empty_cleanup,
                status="aborted",
                error=safe_error,
            )
            write_json(run_dir / "summary.json", summary)
            return summary

        worker_dir = run_dir / "worker"
        worker_dir.mkdir(parents=True, exist_ok=True)
        worker_config = {
            "logic_sha256": verification["worker_logic_sha256"],
            "model_dir": f"/model_repo/snapshots/{DIRECT_REVISION}",
            "audio_path": "/inputs/prompt_audio.wav",
            "audio_sha256": DIRECT_AUDIO_SHA256,
            "audio_duration_s": DIRECT_AUDIO_DURATION_S,
            "prompt": DIRECT_PROMPT,
            "case": case,
        }
        config_path = worker_dir / "config.json"
        result_path = worker_dir / "result.json"
        write_json(config_path, worker_config)
        # The pinned NVIDIA image runs its Python entrypoint as a non-host UID.
        # Permit that isolated container to atomically create result.json.tmp in
        # the one intentionally writable bind, then restore normal permissions
        # immediately after the worker exits.
        worker_dir.chmod(0o1777)
        command, cidfile, labels = _docker_command(
            model=model,
            run_id=run_id,
            worker_dir=worker_dir,
            repository=_repository_path(model),
        )
        telemetry = TelemetrySampler(run_dir / "telemetry.jsonl")
        telemetry.start()
        telemetry.set_phase("trtllm_direct_worker")
        journal.append(
            {
                "event": "worker_start",
                "timeout_s": deadline_s,
                "image": model.image,
                "image_digest": model.image_digest,
                "run_id": run_id,
            }
        )
        try:
            outcome = _invoke_worker(
                command=command,
                cidfile=cidfile,
                labels=labels,
                result_path=result_path,
                log_dir=worker_dir,
                timeout_s=deadline_s,
            )
        finally:
            telemetry.stop()
            worker_dir.chmod(0o755)
        journal.append({"event": "worker_cleanup", **outcome.cleanup})

        worker = outcome.result
        validated_worker: dict[str, Any] | None = None
        error = outcome.error
        if error is None and worker is not None:
            try:
                _validate_worker_result(
                    worker,
                    case=case,
                    logic_hash=verification["worker_logic_sha256"],
                )
                validated_worker = worker
            except DirectTrtllmError as validation_error:
                error = str(validation_error)
        if error is None and worker is not None:
            validation_passed = all(
                request["transcription_exact"] for request in worker["requests"]
            )
            journal.append(
                {
                    "event": "model_loaded",
                    "load_time_s": worker["load_time_s"],
                    "input_prepare_time_s": worker["input_prepare_time_s"],
                    "runtime": worker["runtime"],
                }
            )
            for request in worker["requests"]:
                journal.append(
                    {
                        "event": "request_complete",
                        "case_id": case["case_id"],
                        "kind": "capability",
                        "repetition": request["repetition"],
                        "result": request,
                        "validation": {
                            "passed": request["transcription_exact"],
                            "reason": (
                                None
                                if request["transcription_exact"]
                                else "transcription_mismatch"
                            ),
                        },
                    }
                )
            journal.append(
                {
                    "event": "case_complete",
                    "case_id": case["case_id"],
                    "kind": "capability",
                    "validation_passed": validation_passed,
                }
            )
            status = "complete" if validation_passed else "partial"
            journal.append(
                {
                    "event": "run_complete",
                    "status": "completed" if validation_passed else "partial",
                }
            )
        else:
            error = _redact_sensitive(error or "worker returned no result")
            journal.append(
                {
                    "event": "run_aborted",
                    "error_type": "DirectTrtllmWorkerError",
                    "error": error,
                }
            )
            status = "aborted"
        summary = _summary(
            run_dir=run_dir,
            model=model,
            suite=suite,
            verification=verification,
            worker=validated_worker,
            cleanup=outcome.cleanup,
            status=status,
            error=error,
        )
        write_json(run_dir / "summary.json", summary)
        return summary


def _worker_run(config: dict[str, Any]) -> dict[str, Any]:
    """Container worker body; no request payload is printed or returned."""

    import torch
    import tensorrt_llm
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models import Phi4MMForCausalLM
    from tensorrt_llm.inputs import default_multimodal_input_loader
    from tensorrt_llm.llmapi import KvCacheConfig

    if _sha256_file(Path(__file__).resolve()) != config["logic_sha256"]:
        raise DirectTrtllmError("Worker logic changed after the plan was frozen")
    audio_path = Path(config["audio_path"])
    if _sha256_file(audio_path) != config["audio_sha256"]:
        raise DirectTrtllmError("Container audio fixture hash differed")
    if not torch.cuda.is_available():
        raise DirectTrtllmError("CUDA is unavailable in the TRT-LLM container")

    runtime = {
        "python": sys.version.split()[0],
        "tensorrt_llm": str(tensorrt_llm.__version__),
        "torch": str(torch.__version__),
    }
    try:
        runtime["transformers"] = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        runtime["transformers"] = None

    model_dir = str(config["model_dir"])
    lora_config = Phi4MMForCausalLM.lora_config(model_dir)
    lora_config.max_loras = 2
    lora_config.max_cpu_loras = 2
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    load_started = time.perf_counter()
    llm = LLM(
        model=model_dir,
        backend="pytorch",
        kv_cache_config=KvCacheConfig(**dict(DIRECT_KV_CACHE_OPTIONS)),
        max_seq_len=32768,
        max_batch_size=1,
        max_num_tokens=8192,
        trust_remote_code=True,
        lora_config=lora_config,
    )
    load_time_s = time.perf_counter() - load_started
    try:
        input_started = time.perf_counter()
        inputs = default_multimodal_input_loader(
            tokenizer=llm.tokenizer,
            model_dir=str(llm._hf_model_dir),
            model_type="phi4mm",
            modality="audio",
            prompts=[str(config["prompt"])],
            media=[str(audio_path)],
            image_data_format="pt",
            num_frames=8,
            device="cpu",
        )
        lora_request = Phi4MMForCausalLM.lora_request(
            len(inputs), "audio", llm._hf_model_dir
        )
        input_prepare_time_s = time.perf_counter() - input_started
        case = config["case"]
        sampling = SamplingParams(
            max_tokens=int(case["max_output_tokens"]), temperature=0.0
        )
        expected = _normalize_transcription(DIRECT_EXPECTED_TRANSCRIPTION)
        expected_hash = hashlib.sha256(expected.encode()).hexdigest()

        def generate(repetition: int) -> dict[str, Any]:
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = llm.generate(inputs, sampling, lora_request=lora_request)
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - started
            candidate = outputs[0].outputs[0]
            text = str(candidate.text)
            token_ids = [int(token) for token in candidate.token_ids]
            normalized = _normalize_transcription(text)
            edit_distance = _character_edit_distance(normalized, expected)
            finish_reason = str(getattr(candidate, "finish_reason", "unknown"))
            return {
                "repetition": repetition,
                "elapsed_s": elapsed_s,
                "completion_tokens": len(token_ids),
                "output_tps": len(token_ids) / max(elapsed_s, 1e-9),
                "finish_reason": finish_reason,
                "transcription_exact": normalized == expected,
                "normalized_output_length": len(normalized),
                "character_edit_distance": edit_distance,
                "character_error_rate": edit_distance / len(expected),
                "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "normalized_transcription_sha256": hashlib.sha256(
                    normalized.encode()
                ).hexdigest(),
                "token_ids_sha256": hashlib.sha256(
                    json.dumps(token_ids, separators=(",", ":")).encode()
                ).hexdigest(),
                "expected_transcription_sha256": expected_hash,
            }

        warmup = generate(-1)
        measured_started = time.perf_counter()
        requests = [generate(index) for index in range(int(case["repetitions"]))]
        measured_wall_s = time.perf_counter() - measured_started
        return {
            "status": "complete",
            "logic_sha256": config["logic_sha256"],
            "audio_sha256": config["audio_sha256"],
            "case_id": case["case_id"],
            "load_time_s": load_time_s,
            "input_prepare_time_s": input_prepare_time_s,
            "runtime": runtime,
            "memory": {
                "load_peak_allocated_bytes": None,
                "availability": "unavailable",
                "reason": (
                    "TRT-LLM model allocations occur in an executor process outside "
                    "the worker's PyTorch allocator"
                ),
            },
            "warmups_completed": 1,
            "warmup": warmup,
            "measured_wall_s": measured_wall_s,
            "requests": requests,
        }
    finally:
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            shutdown()
        gc.collect()
        torch.cuda.empty_cache()


def _worker_main(config_path: Path, result_path: Path) -> int:
    try:
        config = json.loads(config_path.read_text())
        result = _worker_run(config)
        exit_code = 0
    except BaseException as error:
        result = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": _redact_sensitive(str(error))[:1000],
        }
        exit_code = 1
    write_json(result_path, result)
    return exit_code


def _build_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("config", type=Path)
    parser.add_argument("result", type=Path)
    return parser


if __name__ == "__main__":
    arguments = _build_worker_parser().parse_args()
    if not arguments.worker:
        raise SystemExit("This module is launched through sparkbench.py trtllm-direct")
    raise SystemExit(_worker_main(arguments.config, arguments.result))
