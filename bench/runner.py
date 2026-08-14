"""Plan expansion and resumable execution for text-generation benchmarks."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
import time
from types import SimpleNamespace
from typing import Any
import uuid
import zlib

from .client import (
    concurrent_chat_requests,
    embedding_request,
    stream_chat_request,
    stream_ollama_chat_request,
)
from .journal import Journal, content_hash, utc_now, write_json
from .report import summarize_run
from .runtime import (
    RuntimeErrorWithContext,
    capture_server_provenance,
    ollama_model_loaded,
    recover_owned_vllm,
    save_server_logs,
    start_server,
)
from .telemetry import TelemetrySampler


class PreflightError(RuntimeError):
    pass


def _command_output(command: list[str]) -> str | None:
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    return result.stdout.strip() if result.returncode == 0 else None


def _image_digest(image: str | None) -> str | None:
    if not image:
        return None
    output = _command_output(
        ["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"]
    )
    if not output:
        return None
    try:
        digests = json.loads(output)
    except json.JSONDecodeError:
        return None
    return digests[0] if digests else None


def _host_snapshot() -> dict[str, Any]:
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, raw = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            meminfo[f"{name.lower()}_kib"] = int(raw.strip().split()[0])
    return {
        "captured_at": utc_now(),
        "uname": _command_output(["uname", "-a"]),
        "python": _command_output(["python3", "--version"]),
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "docker": _command_output(["docker", "version", "--format", "{{.Server.Version}}"]),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_status": _command_output(["git", "status", "--short"]),
        **meminfo,
    }


def _canonical_case(model: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    payload = {"model": model, "case": case}
    return {**case, "case_id": f"{case['id']}--{content_hash(payload, 12)}"}


def create_plan(
    *, model: Any, suite: Any, results_root: Path, models_path: Path, suite_path: Path
) -> Path:
    model_data = asdict(model)
    suite_data = asdict(suite)
    cases = [_canonical_case(model_data, case) for case in suite_data["cases"]]
    resolved_image = _image_digest(model.image)
    if model.image_digest and (
        not resolved_image or not resolved_image.endswith("@" + model.image_digest)
    ):
        raise RuntimeError(
            f"Local image digest for {model.image} does not match manifest {model.image_digest}"
        )
    resolved = {"image_digest": resolved_image}
    fingerprint = content_hash(
        {"model": model_data, "suite": suite_data, "resolved": resolved}
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_root / f"{stamp}-{model.id}-{suite.id}-{fingerprint[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema_version": 2,
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "models_manifest": str(models_path),
        "suite_manifest": str(suite_path),
        "model": model_data,
        "suite": {**suite_data, "cases": cases},
        "resolved": resolved,
        "host_at_plan": _host_snapshot(),
    }
    plan["integrity_hash"] = content_hash(plan, 64)
    write_json(run_dir / "plan.json", plan)
    write_json(run_dir / "inventory.json", _host_snapshot())
    return run_dir


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {key: _plain(item) for key, item in vars(value).items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _preflight(model: SimpleNamespace) -> None:
    running = _command_output(["docker", "ps", "--format", "{{.Names}}"])
    if running is None:
        raise PreflightError("Could not verify running Docker containers")
    containers = [name for name in running.splitlines() if name != "sparkbench-vllm"]
    if containers:
        raise PreflightError("Unrelated containers are running: " + ", ".join(containers))
    compute_apps = None
    for attempt in range(11):
        compute_apps = _command_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ]
        )
        if compute_apps is None:
            raise PreflightError("Could not verify active GPU compute processes")
        if not compute_apps:
            break
        if attempt < 10:
            time.sleep(0.5)
    if compute_apps:
        raise PreflightError("Unrelated GPU compute processes are active: " + compute_apps)
    available_kib = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available_kib = int(line.split()[1])
            break
    required_gib = float(getattr(model, "estimated_ram_gib", 0) or 0) + 8
    if available_kib / 1024**2 < required_gib:
        raise PreflightError(
            f"Only {available_kib / 1024**2:.1f} GiB available; model plus reserve needs {required_gib:.1f} GiB"
        )


def _needle(nonce: str) -> str:
    return "SPARK-" + hashlib.sha256(nonce.encode()).hexdigest()[:10].upper()


def _solid_red_png_data_url(size: int = 64) -> str:
    width = height = max(16, min(size, 2048))
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _prompt(case: SimpleNamespace, nonce: str) -> str:
    kind = str(case.kind)
    prompt_repetitions = int(case.prompt_repetitions)
    prefix = f"Benchmark nonce {nonce}. "
    if kind == "prefill":
        return prefix + "Read all text and reply with exactly one word. " + (
            "measurement " * prompt_repetitions
        )
    if kind == "capability" and "json" in case.requires:
        return prefix + 'Return only a JSON object with keys "benchmark" set to "spark" and "value" set to 42.'
    if kind == "capability" and "tools" in case.requires:
        return prefix + "Use the multiply tool to multiply 6 by 7. Do not answer without calling the tool."
    if kind == "capability" and str(case.id).startswith("long-context-needle"):
        filler = "archive " * max(prompt_repetitions, 1)
        midpoint = len(filler) // 2
        needle = _needle(nonce)
        return (
            prefix
            + filler[:midpoint]
            + f" The unique benchmark key is {needle}. "
            + filler[midpoint:]
            + " What is the unique benchmark key? Reply with only the key."
        )
    return (
        prefix
        + "Write an unbroken numbered list of distinct two-word phrases. "
        "Continue until the output limit; do not conclude or summarize."
    )


def _estimated_context_tokens(case: SimpleNamespace) -> tuple[int, str]:
    """Return a conservative workload estimate without model-specific tokenizers."""

    output_tokens = 1 if str(case.kind) == "prefill" else int(case.max_output_tokens)
    if "vision" in case.requires:
        image_size = max(16, min(int(case.prompt_repetitions) or 64, 2048))
        patch_tokens = ((image_size + 13) // 14) ** 2
        return patch_tokens + output_tokens + 256, "clamped_vision_patch14_plus_margin"
    if "embeddings" in case.requires:
        return (
            max(int(case.prompt_repetitions), 1) + output_tokens + 32,
            "embedding_words_plus_margin",
        )
    prompt_words = len(_prompt(case, "context-estimate").split())
    request_overhead = 128
    estimate_basis = "prompt_words_plus_request_margin"
    if "tools" in case.requires:
        request_overhead += 256
        estimate_basis = "prompt_words_plus_tool_schema_margin"
    if "json" in case.requires:
        request_overhead += 64
        if estimate_basis == "prompt_words_plus_request_margin":
            estimate_basis = "prompt_words_plus_json_schema_margin"
    return (
        prompt_words + output_tokens + request_overhead,
        estimate_basis,
    )


def _request_arguments(
    *, server: Any, model: SimpleNamespace, case: SimpleNamespace, request_id: str
) -> dict[str, Any]:
    kind = str(case.kind)
    max_tokens = 1 if kind == "prefill" else int(case.max_output_tokens)
    request_body_json = getattr(model, "request_body_json", None)
    extra_body = json.loads(request_body_json) if request_body_json else {}
    if str(case.kind) == "capability" and "json" in case.requires:
        extra_body["response_format"] = {"type": "json_object"}
    if str(case.kind) == "capability" and "tools" in case.requires:
        extra_body.update(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "multiply",
                            "description": "Multiply two integers.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer"},
                                    "b": {"type": "integer"},
                                },
                                "required": ["a", "b"],
                            },
                        },
                    }
                ],
                "tool_choice": "required",
            }
        )
    if str(case.kind) == "capability" and "vision" in case.requires:
        extra_body["messages"] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the dominant color of this image? Reply with one color word.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _solid_red_png_data_url(
                                int(case.prompt_repetitions) or 64
                            )
                        },
                    },
                ],
            }
        ]
    arguments = {
        "base_url": server.base_url,
        "model": str(model.served_name),
        "prompt": _prompt(case, request_id),
        "max_tokens": max_tokens,
        "temperature": float(case.temperature),
        "request_id": request_id,
        "extra_body": extra_body,
    }
    if server.backend == "ollama":
        arguments["context_size"] = int(model.max_context)
    return arguments


def _chat_request_function(server: Any):
    return stream_ollama_chat_request if server.backend == "ollama" else stream_chat_request


def _validate_capability(case: SimpleNamespace, result: Any) -> dict[str, Any] | None:
    if str(case.kind) in {"decode", "concurrency"}:
        expected_tokens = int(case.max_output_tokens)
        actual_tokens = int(result.completion_tokens)
        passed = result.finish_reason == "length" and actual_tokens == expected_tokens
        if result.finish_reason != "length":
            reason = f"generation ended with {result.finish_reason!r}"
        elif actual_tokens != expected_tokens:
            reason = (
                f"generation reported {actual_tokens} completion tokens; "
                f"expected {expected_tokens}"
            )
        else:
            reason = None
        return {
            "passed": passed,
            "reason": reason,
        }
    if str(case.kind) != "capability":
        return None
    if "json" in case.requires:
        try:
            payload = json.loads(result.content)
        except json.JSONDecodeError:
            return {"passed": False, "reason": "response was not valid JSON"}
        passed = payload.get("benchmark") == "spark" and payload.get("value") == 42
        return {"passed": passed, "reason": None if passed else "JSON values differed"}
    if "tools" in case.requires:
        for call in result.tool_calls:
            function = call.get("function") or {}
            if function.get("name") != "multiply":
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            passed = {arguments.get("a"), arguments.get("b")} == {6, 7}
            return {"passed": passed, "reason": None if passed else "tool arguments differed"}
        return {"passed": False, "reason": "multiply tool was not called"}
    if "vision" in case.requires:
        passed = "red" in result.content.lower()
        return {"passed": passed, "reason": None if passed else "dominant color was not red"}
    if "embeddings" in case.requires:
        passed = bool(result.finite and result.dimension > 0 and result.batch_size > 0)
        return {"passed": passed, "reason": None if passed else "embedding vector validation failed"}
    if str(case.id).startswith("long-context-needle"):
        passed = _needle(result.request_id) in result.content
        return {"passed": passed, "reason": None if passed else "needle was not returned"}
    return {"passed": False, "reason": "capability adapter is not implemented"}


def _run_warmups(server: Any, model: SimpleNamespace, case: SimpleNamespace) -> None:
    for index in range(int(case.warmups)):
        request_id = f"warmup-{case.case_id}-{index}-{time.time_ns()}"
        if "embeddings" in case.requires:
            text = "measurement " * max(int(case.prompt_repetitions), 1)
            embedding_request(
                base_url=server.base_url,
                model=str(model.served_name),
                inputs=[f"{text} batch item {item}" for item in range(int(case.concurrency))],
                request_id=request_id,
            )
        else:
            _chat_request_function(server)(
                **_request_arguments(server=server, model=model, case=case, request_id=request_id)
            )


def _prime_model(server: Any, model: SimpleNamespace) -> Any:
    request_id = f"first-request-after-start-{time.time_ns()}"
    if "embeddings" in model.tasks and "chat" not in model.tasks:
        return embedding_request(
            base_url=server.base_url,
            model=str(model.served_name),
            inputs=["Spark benchmark model-load probe."],
            request_id=request_id,
        )
    prime_case = SimpleNamespace(
        id="first-request-after-start",
        case_id="first-request-after-start",
        kind="decode",
        requires=["chat"],
        prompt_repetitions=0,
        max_output_tokens=8,
        temperature=0.0,
    )
    return _chat_request_function(server)(
        **_request_arguments(
            server=server,
            model=model,
            case=prime_case,
            request_id=request_id,
        )
    )


def _execute_case(
    *,
    server: Any,
    model: SimpleNamespace,
    case: SimpleNamespace,
    journal: Journal,
    telemetry: TelemetrySampler,
) -> None:
    attempt_id = uuid.uuid4().hex
    journal.append(
        {
            "event": "case_start",
            "case_id": case.case_id,
            "attempt_id": attempt_id,
            "kind": case.kind,
            "concurrency": case.concurrency,
        }
    )
    telemetry.set_phase(f"warmup:{case.case_id}:{attempt_id}")
    started = time.perf_counter()
    try:
        _run_warmups(server, model, case)
        telemetry.set_phase(f"case:{case.case_id}:{attempt_id}")
        measured_started = time.perf_counter()
        validation_results: list[dict[str, Any]] = []
        for repetition in range(int(case.repetitions)):
            if "embeddings" in case.requires:
                request_id = f"{case.case_id}-r{repetition}-w0-{time.time_ns()}"
                text = "measurement " * max(int(case.prompt_repetitions), 1)
                burst_started = time.perf_counter()
                results = [
                    embedding_request(
                        base_url=server.base_url,
                        model=str(model.served_name),
                        inputs=[
                            f"{text} batch item {item} nonce {request_id}"
                            for item in range(int(case.concurrency))
                        ],
                        request_id=request_id,
                    )
                ]
                burst_s = time.perf_counter() - burst_started
            else:
                requests = [
                    _request_arguments(
                        server=server,
                        model=model,
                        case=case,
                        request_id=f"{case.case_id}-r{repetition}-w{worker}-{time.time_ns()}",
                    )
                    for worker in range(int(case.concurrency))
                ]
                results, burst_s = concurrent_chat_requests(
                    requests=requests,
                    concurrency=int(case.concurrency),
                    request_function=_chat_request_function(server),
                )
            for result in results:
                validation = _validate_capability(case, result)
                if validation is not None:
                    validation_results.append(validation)
                journal.append(
                    {
                        "event": "request_complete",
                        "case_id": case.case_id,
                        "attempt_id": attempt_id,
                        "kind": case.kind,
                        "repetition": repetition,
                        "burst_elapsed_s": burst_s,
                        "result": result.to_dict(),
                        "validation": validation,
                    }
                )
        elapsed_s = time.perf_counter() - measured_started
    except Exception as error:
        journal.append(
            {
                "event": "case_failed",
                "case_id": case.case_id,
                "attempt_id": attempt_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_s": time.perf_counter() - started,
            }
        )
        raise
    else:
        journal.append(
            {
                "event": "case_complete",
                "case_id": case.case_id,
                "attempt_id": attempt_id,
                "kind": case.kind,
                "concurrency": case.concurrency,
                "elapsed_s": elapsed_s,
                "validation_passed": (
                    all(item["passed"] for item in validation_results)
                    if validation_results
                    else None
                ),
            }
        )
    finally:
        telemetry.set_phase("between_cases")


def _recover_pending_lifecycle(
    *, model: SimpleNamespace, journal: Journal
) -> bool:
    """Resolve an earlier crashed run's lifecycle before resume or finalization."""

    events = journal.events()
    last_start = max(
        (index for index, event in enumerate(events) if event.get("event") == "run_start"),
        default=-1,
    )
    if last_start < 0:
        return False
    run_finished = any(
        index > last_start and event.get("event") == "run_complete"
        for index, event in enumerate(events)
    )
    ready_indexes = [
        index
        for index, event in enumerate(events)
        if index > last_start and event.get("event") == "server_ready"
    ]
    if not ready_indexes:
        if str(model.backend) == "vllm" and not run_finished:
            action = recover_owned_vllm(str(model.run_identity))
            journal.append(
                {
                    "event": "server_stopped",
                    "backend": "vllm",
                    "recovered": True,
                    "recovery_action": action,
                }
            )
            return True
        return False
    last_ready = ready_indexes[-1]
    if any(
        index > last_ready and event.get("event") in {"server_stopped", "server_kept"}
        for index, event in enumerate(events)
    ):
        return False

    ready_event = events[last_ready]
    backend = str(model.backend)
    if backend == "vllm":
        action = recover_owned_vllm(str(model.run_identity))
        journal.append(
            {
                "event": "server_stopped",
                "backend": backend,
                "recovered": True,
                "recovery_action": action,
            }
        )
        return True

    if backend != "ollama":
        return False

    cleanup_failed = any(
        index > last_ready and event.get("event") == "cleanup_failed"
        for index, event in enumerate(events)
    )
    unload_owned = ready_event.get("ollama_unload_owned")
    if run_finished and not cleanup_failed:
        # Older completed journals predate explicit server_stopped events. Do
        # not reinterpret a later user load as benchmark-owned.
        return False
    if unload_owned is False:
        return False

    try:
        loaded = ollama_model_loaded(str(model.endpoint), str(model.source))
    except Exception as error:
        journal.append(
            {
                "event": "recovery_cleanup_pending",
                "backend": backend,
                "model": str(model.source),
                "reason": f"loaded state could not be verified: {error}",
            }
        )
        raise
    if loaded:
        journal.append(
            {
                "event": "recovery_cleanup_pending",
                "backend": backend,
                "model": str(model.source),
                "reason": "model remains loaded and crash-time ownership cannot be proven",
            }
        )
        raise RuntimeErrorWithContext(
            f"Ollama model {model.source} may still be owned by the crashed run; "
            "refusing to finalize or unload it automatically"
        )
    journal.append(
        {
            "event": "server_stopped",
            "backend": backend,
            "recovered": True,
            "recovery_action": "observed_model_absent",
        }
    )
    return True


def execute_plan(
    run_dir: Path,
    *,
    workspace: Path,
    allow_download: bool = False,
    keep_server: bool = False,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    plan_path = run_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    suite_without_case_ids = {
        **plan["suite"],
        "cases": [
            {key: value for key, value in case.items() if key != "case_id"}
            for case in plan["suite"]["cases"]
        ],
    }
    schema_version = int(plan.get("schema_version", 1))
    if schema_version not in {1, 2}:
        raise RuntimeError(f"Unsupported frozen plan schema version: {schema_version}")
    if schema_version >= 2:
        integrity_hash = plan.get("integrity_hash")
        integrity_payload = {
            key: value for key, value in plan.items() if key != "integrity_hash"
        }
        expected_fingerprint = content_hash(
            {
                "model": plan["model"],
                "suite": suite_without_case_ids,
                "resolved": plan.get("resolved", {}),
            }
        )
        integrity_length = len(str(integrity_hash)) if integrity_hash else 64
        integrity_valid = bool(integrity_hash) and (
            content_hash(integrity_payload, integrity_length) == integrity_hash
        )
    else:
        expected_fingerprint = content_hash(
            {"model": plan["model"], "suite": suite_without_case_ids}
        )
        expected_digest = plan["model"].get("image_digest")
        resolved_image = plan.get("resolved", {}).get("image_digest")
        integrity_valid = not expected_digest or (
            isinstance(resolved_image, str)
            and resolved_image.endswith("@" + expected_digest)
        )
    if not integrity_valid or expected_fingerprint != plan["fingerprint"]:
        raise RuntimeError("Frozen plan fingerprint does not match its contents")
    for case in plan["suite"]["cases"]:
        case_without_id = {key: value for key, value in case.items() if key != "case_id"}
        expected_case_id = _canonical_case(plan["model"], case_without_id)["case_id"]
        if case.get("case_id") != expected_case_id:
            raise RuntimeError("Frozen plan case identity does not match its contents")
    model = _namespace(plan["model"])
    model.resolved_image = plan.get("resolved", {}).get("image_digest")
    model.run_identity = f"{plan['fingerprint']}-{run_dir.name}"
    cases = [_namespace(case) for case in plan["suite"]["cases"]]
    journal = Journal(run_dir / "events.jsonl")
    completed = journal.completed_cases()
    terminal = journal.terminal_cases()
    lock_path = results_lock_path(workspace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry = TelemetrySampler(run_dir / "telemetry.jsonl")
    server = None
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another SparkBench run holds the benchmark lock") from error
        lifecycle_changed = _recover_pending_lifecycle(model=model, journal=journal)
        if cases and all(case.case_id in terminal for case in cases):
            existing_summary = summarize_run(run_dir)
            if lifecycle_changed:
                existing_summary = summarize_run(run_dir)
            if existing_summary["status"] in {"aborted", "incomplete", "not_started"}:
                journal.append(
                    {
                        "event": "run_complete",
                        "status": "recovered_all_terminal" if completed else "no_work",
                    }
                )
                return summarize_run(run_dir)
            return existing_summary
        journal.append({"event": "run_start", "completed_cases_at_resume": sorted(completed)})
        runnable = []
        for case in cases:
            if case.case_id in terminal:
                continue
            missing = set(case.requires) - set(model.tasks)
            if missing:
                journal.append(
                    {
                        "event": "case_skipped_unsupported",
                        "case_id": case.case_id,
                        "missing_capabilities": sorted(missing),
                    }
                )
                continue
            unavailable_adapters = {"rerank"} & set(case.requires)
            if unavailable_adapters:
                journal.append(
                    {
                        "event": "case_skipped_adapter_unimplemented",
                        "case_id": case.case_id,
                        "capabilities": sorted(unavailable_adapters),
                    }
                )
                continue
            approximate_tokens, estimate_basis = _estimated_context_tokens(case)
            if approximate_tokens > int(model.max_context):
                journal.append(
                    {
                        "event": "case_skipped_context_limit",
                        "case_id": case.case_id,
                        "approximate_required_tokens": approximate_tokens,
                        "estimate_basis": estimate_basis,
                        "model_max_context": model.max_context,
                    }
                )
                continue
            runnable.append(case)
        if not runnable:
            journal.append({"event": "run_complete", "status": "no_work"})
            return summarize_run(run_dir)

        _preflight(model)
        telemetry.start()
        telemetry.set_phase("server_startup")
        primary_error: BaseException | None = None
        try:
            server = start_server(model, workspace=workspace, allow_download=allow_download)
            write_json(run_dir / "server" / "provenance.json", capture_server_provenance(server))
            journal.append(
                {
                    "event": "server_ready",
                    "startup_s": server.startup_s,
                    "backend": server.backend,
                    "container_id": getattr(server, "container_id", None),
                    "ollama_model": getattr(server, "ollama_model", None),
                    "ollama_unload_owned": bool(
                        getattr(server, "unload_ollama", False)
                    ),
                    "keep_server_requested": keep_server,
                }
            )
            telemetry.set_phase("first_request_after_start")
            first_request = _prime_model(server, model)
            journal.append(
                {
                    "event": "first_request_complete",
                    "backend": server.backend,
                    "result": first_request.to_dict(),
                }
            )
            for case in runnable:
                try:
                    _execute_case(
                        server=server,
                        model=model,
                        case=case,
                        journal=journal,
                        telemetry=telemetry,
                    )
                except Exception:
                    if not continue_on_error:
                        raise
        except BaseException as error:
            primary_error = error
            journal.append(
                {
                    "event": "run_aborted",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise
        finally:
            telemetry.set_phase("server_shutdown")
            try:
                if server:
                    try:
                        save_server_logs(server, run_dir / "server" / "server.log")
                    finally:
                        server.stop(keep_server=keep_server)
                    journal.append(
                        {
                            "event": "server_kept" if keep_server else "server_stopped",
                            "backend": server.backend,
                        }
                    )
            except Exception as cleanup_error:
                journal.append(
                    {
                        "event": "cleanup_failed",
                        "error_type": type(cleanup_error).__name__,
                        "error": str(cleanup_error),
                    }
                )
                if primary_error is None:
                    raise
            finally:
                telemetry.stop()
        journal.append(
            {
                "event": "run_complete",
                "status": "completed_server_kept" if keep_server else "completed",
            }
        )
    return summarize_run(run_dir)


def results_lock_path(workspace: Path) -> Path:
    return workspace / "results" / ".sparkbench.lock"
