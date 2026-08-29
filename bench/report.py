"""Result aggregation for SparkBench JSONL journals."""

from __future__ import annotations

import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .annotations import (
    measurement_annotations,
    startup_safety_gates_from_annotations,
)
from .journal import write_json
from .llamacpp_metrics import (
    aggregate_llamacpp_spec_decode_metrics,
    assess_llamacpp_mtp_evidence,
    assess_llamacpp_mtp_proposal_depth,
    llamacpp_dflash_requested,
    llamacpp_mtp_depth,
    llamacpp_mtp_requested,
)
from .memory_ops import (
    MEMORY_OPERATION_SUITE_ID,
    summarize_memory_operation_results,
)
from .prefix_cache_protocol import (
    PREFIX_CACHE_PROTOCOL,
    prefix_cache_conditions,
    prefix_cache_steps,
)
from .sglang_metrics import (
    aggregate_sglang_speculative_audits,
    sglang_nextn_depth,
)
from .sglang_sm121_storage import (
    is_sm121_storage_canary_plan,
    sm121_storage_canary_lifecycle_issues,
)
from .vllm_metrics import aggregate_vllm_spec_decode_metrics


def _is_diffusion_architecture(value: Any) -> bool:
    return str(value or "").strip().lower() == "diffusion-lm"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _reported_reasoning_tokens(result: dict[str, Any]) -> int | None:
    """Return a separately reported token count without guessing missing usage."""

    value = result.get("reasoning_tokens")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if not isinstance(value, float):
        return None
    if not math.isfinite(value) or value < 0 or not value.is_integer():
        return None
    return int(value)


def _prefix_cache_number(result: dict[str, Any], key: str) -> float:
    value = result.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"prefix-cache result {key} must be a non-negative scalar")
    return float(value)


def _prefix_cache_count(result: dict[str, Any], key: str) -> int:
    value = _prefix_cache_number(result, key)
    if not value.is_integer():
        raise ValueError(f"prefix-cache result {key} must be an integer")
    return int(value)


def _summarize_prefix_cache_case(
    requests: list[dict[str, Any]], *, case_session_wall_s: float
) -> dict[str, Any]:
    """Aggregate the dedicated serial prompt-KV protocol without ambiguity.

    ``logical_prompt_tokens`` includes reused tokens, whereas
    ``physical_uncached_prompt_tokens`` comes from the final request-scoped
    llama.cpp SSE timing payload and excludes them.  The
    ``prometheus_global_*`` raw fields are global Prometheus diagnostics and
    are intentionally not used in any per-request aggregate or paired
    comparison.  Condition request wall is the sum of end-to-end client
    request elapsed times; session wall also includes the protocol's local
    before/after counter snapshots.
    """

    if case_session_wall_s <= 0:
        raise ValueError("prefix-cache session wall time must be positive")
    records: dict[str, list[dict[str, Any]]] = {}
    pair_records: dict[int, dict[str, dict[str, Any]]] = {}
    modes: set[str] = set()
    prefix_targets: set[int] = set()
    for event in requests:
        result = event.get("result")
        if not isinstance(result, dict):
            raise ValueError("prefix-cache request result must be an object")
        mode = result.get("cache_profile_mode")
        condition = result.get("cache_condition")
        if mode not in {"off", "on"} or not isinstance(condition, str):
            raise ValueError("prefix-cache result lacks a valid control label")
        steps = prefix_cache_steps(mode)
        step_ordinal = _prefix_cache_count(result, "cache_step_ordinal")
        if not 1 <= step_ordinal <= len(steps):
            raise ValueError("prefix-cache step ordinal is invalid")
        expected_condition, _, expected_control = steps[step_ordinal - 1]
        if condition != expected_condition:
            raise ValueError("prefix-cache condition does not match its ordinal")
        if result.get("cache_prompt_control") != expected_control:
            raise ValueError("prefix-cache request control does not match its ordinal")
        pair_index = _prefix_cache_count(result, "cache_pair_index")
        if pair_index <= 0:
            raise ValueError("prefix-cache pair index must be positive")
        prefix_target = _prefix_cache_count(result, "cache_prefix_target_words")
        if prefix_target <= 0:
            raise ValueError("prefix-cache target must be positive")
        prompt_tokens = _prefix_cache_count(result, "prompt_tokens")
        cached_tokens = _prefix_cache_count(result, "cached_prompt_tokens")
        if cached_tokens > prompt_tokens:
            raise ValueError("prefix-cache cached tokens exceed logical prompt tokens")
        server_prompt_tokens = _prefix_cache_count(result, "server_prompt_tokens")
        server_cached_tokens = _prefix_cache_count(
            result, "server_cached_prompt_tokens"
        )
        server_decode_tokens = _prefix_cache_count(result, "server_decode_tokens")
        if (
            server_prompt_tokens + server_cached_tokens != prompt_tokens
            or server_cached_tokens != cached_tokens
            or server_decode_tokens != _prefix_cache_count(result, "completion_tokens")
        ):
            raise ValueError("prefix-cache server counters did not reconcile")
        records.setdefault(condition, []).append(result)
        per_pair = pair_records.setdefault(pair_index, {})
        if condition in per_pair:
            raise ValueError("prefix-cache pair has duplicate conditions")
        per_pair[condition] = result
        modes.add(mode)
        prefix_targets.add(prefix_target)
    if len(modes) != 1 or len(prefix_targets) != 1:
        raise ValueError("prefix-cache case mixed modes or prefix targets")
    mode = next(iter(modes))
    expected_conditions = prefix_cache_conditions(mode)
    if set(records) != set(expected_conditions):
        raise ValueError("prefix-cache case has an unexpected condition set")
    expected_pairs = set(range(1, 6))
    if set(pair_records) != expected_pairs or any(
        set(record) != set(expected_conditions) for record in pair_records.values()
    ):
        raise ValueError("prefix-cache case does not contain five complete paired blocks")

    conditions: list[dict[str, Any]] = []
    for step_ordinal, (condition, _, cache_prompt_control) in enumerate(
        prefix_cache_steps(mode), start=1
    ):
        values = records[condition]
        if len(values) != 5:
            raise ValueError("prefix-cache condition does not contain five requests")
        logical_tokens = sum(_prefix_cache_count(value, "prompt_tokens") for value in values)
        cached_tokens = sum(
            _prefix_cache_count(value, "server_cached_prompt_tokens")
            for value in values
        )
        physical_tokens = sum(
            _prefix_cache_count(value, "server_prompt_tokens") for value in values
        )
        completion_tokens = sum(
            _prefix_cache_count(value, "server_decode_tokens") for value in values
        )
        server_prompt_s = sum(
            _prefix_cache_number(value, "server_prompt_s") for value in values
        )
        server_decode_s = sum(
            _prefix_cache_number(value, "server_decode_s") for value in values
        )
        condition_request_wall_s = sum(
            _prefix_cache_number(value, "elapsed_s") for value in values
        )
        ttfts = [_prefix_cache_number(value, "ttft_s") for value in values]
        client_decode_tps = [
            _prefix_cache_number(value, "decode_tps") for value in values
        ]
        if condition_request_wall_s <= 0 or server_decode_s <= 0:
            raise ValueError("prefix-cache request or server decode wall time was zero")
        conditions.append(
            {
                "cache_condition": condition,
                "cache_prompt_control": cache_prompt_control,
                "protocol_step_ordinal": step_ordinal,
                "request_count": len(values),
                "logical_prompt_tokens": logical_tokens,
                "physical_uncached_prompt_tokens": physical_tokens,
                "cached_prompt_tokens": cached_tokens,
                "cache_hit_fraction": cached_tokens / logical_tokens,
                "server_prompt_processing_s": server_prompt_s,
                "server_decode_s": server_decode_s,
                "server_decode_tps": completion_tokens / server_decode_s,
                "logical_prompt_tokens_per_server_prompt_s": (
                    logical_tokens / server_prompt_s if server_prompt_s > 0 else None
                ),
                "physical_uncached_prompt_tokens_per_server_prompt_s": (
                    physical_tokens / server_prompt_s if server_prompt_s > 0 else None
                ),
                "condition_request_wall_s": condition_request_wall_s,
                "end_to_end_output_tokens_per_condition_request_wall_s": (
                    completion_tokens / condition_request_wall_s
                ),
                "median_ttft_s": statistics.median(ttfts),
                "median_e2e_s": statistics.median(
                    _prefix_cache_number(value, "elapsed_s") for value in values
                ),
                "median_client_decode_tps": statistics.median(client_decode_tps),
            }
        )
    summary: dict[str, Any] = {
        "protocol": PREFIX_CACHE_PROTOCOL,
        "profile_mode": mode,
        "prefix_target_words": next(iter(prefix_targets)),
        "case_session_wall_s": case_session_wall_s,
        "conditions": conditions,
    }
    second_condition = "forced-cold-b"
    third_condition = expected_conditions[2]
    paired: list[dict[str, Any]] = []
    for pair_index in range(1, 6):
        second = pair_records[pair_index][second_condition]
        third = pair_records[pair_index][third_condition]
        paired.append(
            {
                "cache_pair_index": pair_index,
                "ttft_second_minus_third_s": (
                    _prefix_cache_number(second, "ttft_s")
                    - _prefix_cache_number(third, "ttft_s")
                ),
                "e2e_second_minus_third_s": (
                    _prefix_cache_number(second, "elapsed_s")
                    - _prefix_cache_number(third, "elapsed_s")
                ),
                "server_prompt_second_minus_third_s": (
                    _prefix_cache_number(second, "server_prompt_s")
                    - _prefix_cache_number(third, "server_prompt_s")
                ),
            }
        )
    summary["paired_second_to_third"] = {
        "paired_blocks": 5,
        "second_condition": second_condition,
        "third_condition": third_condition,
        "per_pair": paired,
        "median_ttft_second_minus_third_s": statistics.median(
            item["ttft_second_minus_third_s"] for item in paired
        ),
        "median_e2e_second_minus_third_s": statistics.median(
            item["e2e_second_minus_third_s"] for item in paired
        ),
        "median_server_prompt_second_minus_third_s": statistics.median(
            item["server_prompt_second_minus_third_s"] for item in paired
        ),
    }
    return summary


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _telemetry_summaries(path: Path) -> dict[str, dict[str, Any]]:
    samples = _read_events(path)
    phases: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        phases.setdefault(str(sample.get("phase", "unknown")), []).append(sample)
    summaries: dict[str, dict[str, Any]] = {}
    for phase, phase_samples in phases.items():
        def numeric(key: str) -> list[float]:
            values: list[float] = []
            for sample in phase_samples:
                raw = sample.get(key)
                if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                    continue
                value = float(raw)
                if math.isfinite(value):
                    values.append(value)
            return values

        powers = numeric("power_w")
        temperatures = numeric("temperature_c")
        utilizations = numeric("gpu_util_pct")
        clocks = numeric("sm_clock_mhz")
        memory_kib = numeric("memavailable_kib")
        sampled_energy_j = 0.0
        sampled_energy_intervals = 0
        for before, after in zip(phase_samples, phase_samples[1:]):
            before_power = before.get("power_w")
            after_power = after.get("power_w")
            if (
                not isinstance(before_power, (int, float))
                or isinstance(before_power, bool)
                or not math.isfinite(float(before_power))
                or not isinstance(after_power, (int, float))
                or isinstance(after_power, bool)
                or not math.isfinite(float(after_power))
            ):
                continue
            try:
                delta_s = (
                    datetime.fromisoformat(str(after["timestamp"]))
                    - datetime.fromisoformat(str(before["timestamp"]))
                ).total_seconds()
            except (KeyError, TypeError, ValueError):
                continue
            if 0 < delta_s <= 5:
                sampled_energy_j += (
                    float(before_power) + float(after_power)
                ) / 2 * delta_s
                sampled_energy_intervals += 1
        summaries[phase] = {
            "samples": len(phase_samples),
            "gpu_power_samples": len(powers),
            "gpu_power_missing_samples": len(phase_samples) - len(powers),
            "gpu_error_samples": sum(
                isinstance(sample.get("gpu_error"), str)
                and bool(str(sample["gpu_error"]).strip())
                for sample in phase_samples
            ),
            "average_power_w": statistics.fmean(powers) if powers else None,
            "peak_power_w": max(powers) if powers else None,
            "sampled_energy_j": (
                sampled_energy_j if sampled_energy_intervals else None
            ),
            "sampled_energy_intervals": sampled_energy_intervals,
            "peak_temperature_c": max(temperatures) if temperatures else None,
            "average_gpu_util_pct": statistics.fmean(utilizations) if utilizations else None,
            "peak_sm_clock_mhz": max(clocks) if clocks else None,
            "minimum_memavailable_gib": min(memory_kib) / 1024**2 if memory_kib else None,
        }
    return summaries


def summarize_run(run_dir: Path) -> dict[str, Any]:
    events = _read_events(run_dir / "events.jsonl")
    telemetry = _telemetry_summaries(run_dir / "telemetry.jsonl")
    try:
        plan = json.loads((run_dir / "plan.json").read_text())
    except (OSError, json.JSONDecodeError):
        plan = {}
    planned_model = plan.get("model") or {}
    planned_suite = plan.get("suite") or {}
    sm121_storage_lifecycle_issues: tuple[dict[str, object], ...] = ()
    if (
        isinstance(planned_model, dict)
        and isinstance(planned_suite, dict)
        and is_sm121_storage_canary_plan(planned_model, planned_suite)
    ):
        planned_cases = planned_suite.get("cases")
        planned_case_ids = (
            tuple(
                case.get("case_id")
                for case in planned_cases
                if isinstance(case, dict) and isinstance(case.get("case_id"), str)
            )
            if isinstance(planned_cases, list)
            else ()
        )
        sm121_storage_lifecycle_issues = sm121_storage_canary_lifecycle_issues(
            events, planned_case_ids=planned_case_ids
        )
    mtp_requested = bool(
        planned_model.get("backend") == "llamacpp"
        and llamacpp_mtp_requested(planned_model.get("args") or ())
    )
    mtp_depth = (
        llamacpp_mtp_depth(planned_model.get("args") or ())
        if mtp_requested
        else None
    )
    dflash_requested = bool(
        planned_model.get("backend") == "llamacpp"
        and llamacpp_dflash_requested(planned_model.get("args") or ())
    )
    speculative_depth = (
        llamacpp_mtp_depth(planned_model.get("args") or ())
        if mtp_requested or dflash_requested
        else None
    )
    sglang_depth = (
        sglang_nextn_depth(planned_model.get("args") or ())
        if planned_model.get("backend") == "sglang"
        else None
    )
    annotations = measurement_annotations(events)
    startup_safety_gates = startup_safety_gates_from_annotations(annotations)
    startup_annotations = [
        annotation
        for annotation in annotations
        if annotation.get("scope") == "startup"
    ]
    case_annotations: dict[str, list[dict[str, Any]]] = {}
    for annotation in annotations:
        case_id = annotation.get("case_id")
        if annotation.get("scope") == "case" and isinstance(case_id, str):
            case_annotations.setdefault(case_id, []).append(annotation)
    llamacpp_speculative_decoding = aggregate_llamacpp_spec_decode_metrics(
        event["metrics"]
        for event in events
        if event.get("event") == "llamacpp_spec_decode_metrics_snapshot"
        and isinstance(event.get("metrics"), dict)
    )
    if llamacpp_speculative_decoding is not None:
        proposal_depth = (
            assess_llamacpp_mtp_proposal_depth(
                llamacpp_speculative_decoding,
                configured_depth=mtp_depth,
            )
            if mtp_depth is not None
            else None
        )
        llamacpp_speculative_decoding = {
            **llamacpp_speculative_decoding,
            "requested": mtp_requested or dflash_requested,
            "method": (
                "draft-dflash"
                if dflash_requested
                else ("draft-mtp" if mtp_requested else None)
            ),
            "configured_max_draft_tokens": speculative_depth,
            "proposal_depth": proposal_depth,
        }
    sglang_speculative_decoding = aggregate_sglang_speculative_audits(
        (
            event["metrics"]
            for event in events
            if event.get("event") == "sglang_spec_decode_metrics_snapshot"
            and isinstance(event.get("metrics"), dict)
        ),
        expected_depth=sglang_depth,
    )
    vllm_speculative_decoding = aggregate_vllm_spec_decode_metrics(
        event["metrics"]
        for event in events
        if event.get("event") == "vllm_spec_decode_metrics_snapshot"
        and isinstance(event.get("metrics"), dict)
    )
    speculative_decoding = (
        llamacpp_speculative_decoding
        or sglang_speculative_decoding
        or vllm_speculative_decoding
    )
    llamacpp_mtp_evidence = assess_llamacpp_mtp_evidence(
        events,
        requested=mtp_requested,
        configured_depth=mtp_depth,
    )
    llamacpp_dflash_evidence = assess_llamacpp_mtp_evidence(
        events,
        requested=dflash_requested,
        configured_depth=None,
        method="DFlash",
    )
    diffusion_generation = _is_diffusion_architecture(
        planned_model.get("architecture")
    )
    completed: dict[str, str] = {}
    case_events: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        if event.get("event") == "case_complete":
            case_id = str(event["case_id"])
            attempt_id = str(event["attempt_id"])
            completed[case_id] = attempt_id
            case_events[(case_id, attempt_id)] = event

    completed_items = sorted(completed.items())
    planned_case_ids: list[str] = []
    if planned_suite.get("id") == MEMORY_OPERATION_SUITE_ID:
        planned_case_ids = [
            str(case["case_id"])
            for case in planned_suite.get("cases", [])
            if isinstance(case, dict) and isinstance(case.get("case_id"), str)
        ]
        completed_items = [
            (case_id, completed[case_id])
            for case_id in planned_case_ids
            if case_id in completed
        ]

    rows: list[dict[str, Any]] = []
    memory_run_results: list[dict[str, Any]] = []
    for case_id, attempt_id in completed_items:
        requests = [
            event
            for event in events
            if event.get("event") == "request_complete"
            and event.get("case_id") == case_id
            and event.get("attempt_id") == attempt_id
        ]
        if not requests:
            continue
        ttfts = [
            float(event["result"]["ttft_s"])
            for event in requests
            if isinstance(event["result"].get("ttft_s"), (int, float))
        ]
        first_emission_times = [
            float(
                event["result"].get(
                    "time_to_first_emission_s", event["result"].get("ttft_s")
                )
            )
            for event in requests
            if isinstance(
                event["result"].get(
                    "time_to_first_emission_s", event["result"].get("ttft_s")
                ),
                (int, float),
            )
        ]
        elapsed = [float(event["result"]["elapsed_s"]) for event in requests]
        decode_rates = [
            float(event["result"]["decode_tps"])
            for event in requests
            if isinstance(event["result"].get("decode_tps"), (int, float))
        ]
        decode_sources = {
            str(event["result"].get("decode_metric_source", "client_estimate"))
            for event in requests
            if event["result"].get("decode_metric_source", "client_estimate")
        }
        prompt_tokens = sum(int(event["result"]["prompt_tokens"]) for event in requests)
        completion_tokens = sum(int(event["result"]["completion_tokens"]) for event in requests)
        reported_reasoning_counts = [
            _reported_reasoning_tokens(event["result"]) for event in requests
        ]
        reasoning_tokens = (
            sum(
                count
                for count in reported_reasoning_counts
                if count is not None
            )
            if all(count is not None for count in reported_reasoning_counts)
            else None
        )
        case_event = case_events[(case_id, attempt_id)]
        kind = str(case_event.get("kind") or requests[0].get("kind", "unknown"))
        valid_generation = case_event.get("validation_passed") is not False
        wall_s = float(case_event.get("elapsed_s") or sum(elapsed))
        row_annotations = case_annotations.get(case_id, [])
        row: dict[str, Any] = {
            "case_id": case_id,
            "attempt_id": attempt_id,
            "kind": kind,
            "measurement_valid": not row_annotations,
            "measurement_annotations": row_annotations,
            "requests": len(requests),
            "concurrency": case_event.get("concurrency", 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "median_ttft_s": (
                statistics.median(ttfts)
                if ttfts and not diffusion_generation
                else None
            ),
            "p95_ttft_s": (
                percentile(ttfts, 0.95)
                if len(ttfts) >= 20 and not diffusion_generation
                else None
            ),
            "median_e2e_s": statistics.median(elapsed),
            "p95_e2e_s": percentile(elapsed, 0.95) if len(elapsed) >= 20 else None,
            "median_decode_tps": (
                statistics.median(decode_rates)
                if kind in {"decode", "concurrency"}
                and valid_generation
                and not diffusion_generation
                and decode_rates
                else None
            ),
            "decode_metric_source": (
                next(iter(decode_sources))
                if kind in {"decode", "concurrency"}
                and valid_generation
                and not diffusion_generation
                and len(decode_sources) == 1
                else (
                    "mixed"
                    if kind in {"decode", "concurrency"}
                    and valid_generation
                    and not diffusion_generation
                    else None
                )
            ),
            "median_estimated_decode_tps": (
                statistics.median(decode_rates)
                if kind in {"decode", "concurrency"}
                and valid_generation
                and not diffusion_generation
                and decode_sources == {"client_estimate"}
                and decode_rates
                else None
            ),
            "decode_estimate_one_token_chunks": (
                all(
                    int(event["result"].get("emission_events", -1))
                    == int(event["result"]["completion_tokens"])
                    for event in requests
                )
                if kind in {"decode", "concurrency"}
                and valid_generation
                and not diffusion_generation
                and decode_sources == {"client_estimate"}
                else None
            ),
            "aggregate_output_tps": (
                completion_tokens / max(wall_s, 1e-9)
                if not diffusion_generation
                and (valid_generation or kind not in {"decode", "concurrency"})
                else None
            ),
            "request_tps": len(requests) / max(wall_s, 1e-9),
            "elapsed_s": wall_s,
            "validation_passed": case_event.get("validation_passed"),
        }
        if kind == "cache":
            # Cache conditions have deliberately different semantics.  Do not
            # present a mixed cold/warm average as a generic TTFT, E2E, or TPS
            # value.  Emit only the fixed protocol report and its unambiguous
            # scalar totals so evidence validation can reject generic metrics.
            row = {
                "case_id": case_id,
                "attempt_id": attempt_id,
                "kind": "cache",
                "measurement_valid": not row_annotations,
                "measurement_annotations": row_annotations,
                "requests": len(requests),
                "concurrency": case_event.get("concurrency", 1),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "elapsed_s": wall_s,
                "validation_passed": case_event.get("validation_passed"),
                "prefix_cache": _summarize_prefix_cache_case(
                    requests, case_session_wall_s=wall_s
                ),
            }
        if kind == "agentic":
            agentic_results = [event["result"] for event in requests]
            model_requests = sum(
                int(result["turns_used"]) for result in agentic_results
            )
            task_wall_times = [
                float(result["wall_s"]) for result in agentic_results
            ]
            request_elapsed_times = [
                float(result["request_elapsed_s"])
                for result in agentic_results
            ]
            first_turn_ttfts = [
                float(result["first_turn_ttft_s"])
                for result in agentic_results
            ]
            tasks_succeeded = sum(
                result.get("passed") is True for result in agentic_results
            )
            recovery_required = sum(
                result.get("recovery_required") is True
                for result in agentic_results
            )
            recovery_succeeded = sum(
                result.get("recovery_succeeded") is True
                for result in agentic_results
            )
            row.update(
                {
                    "aggregate_output_tps": None,
                    "median_ttft_s": None,
                    "p95_ttft_s": None,
                    "request_tps": None,
                    "agentic_tasks": len(agentic_results),
                    "agentic_tasks_succeeded": tasks_succeeded,
                    "agentic_task_success_rate": (
                        tasks_succeeded / len(agentic_results)
                    ),
                    "agentic_tasks_per_s": (
                        len(agentic_results) / max(wall_s, 1e-9)
                    ),
                    "agentic_model_requests": model_requests,
                    "agentic_model_requests_per_s": (
                        model_requests / max(wall_s, 1e-9)
                    ),
                    "agentic_max_turns": int(agentic_results[0]["max_turns"]),
                    "agentic_max_output_tokens_per_turn": int(
                        agentic_results[0]["max_output_tokens"]
                    ),
                    "median_agentic_turns_used": statistics.median(
                        int(result["turns_used"]) for result in agentic_results
                    ),
                    "median_agentic_task_wall_s": statistics.median(
                        task_wall_times
                    ),
                    "median_agentic_model_request_sum_s": statistics.median(
                        request_elapsed_times
                    ),
                    "median_agentic_first_turn_ttft_s": statistics.median(
                        first_turn_ttfts
                    ),
                    "agentic_expected_tool_calls": sum(
                        int(result["expected_tool_calls"])
                        for result in agentic_results
                    ),
                    "agentic_tool_calls_requested": sum(
                        int(result["tool_calls_requested"])
                        for result in agentic_results
                    ),
                    "agentic_tool_calls_executed": sum(
                        int(result["tool_calls_executed"])
                        for result in agentic_results
                    ),
                    "agentic_tool_calls_succeeded": sum(
                        int(result["tool_calls_succeeded"])
                        for result in agentic_results
                    ),
                    "agentic_tool_errors": sum(
                        int(result["tool_errors"])
                        for result in agentic_results
                    ),
                    "agentic_malformed_tool_calls": sum(
                        int(result["malformed_tool_calls"])
                        for result in agentic_results
                    ),
                    "agentic_unknown_tool_calls": sum(
                        int(result["unknown_tool_calls"])
                        for result in agentic_results
                    ),
                    "agentic_final_answers_emitted": sum(
                        result.get("final_answer_emitted") is True
                        for result in agentic_results
                    ),
                    "agentic_final_answers_correct": sum(
                        result.get("final_answer_correct") is True
                        for result in agentic_results
                    ),
                    "agentic_tool_sequences_correct": sum(
                        result.get("tool_sequence_correct") is True
                        for result in agentic_results
                    ),
                    "agentic_recoveries_required": recovery_required,
                    "agentic_recoveries_succeeded": recovery_succeeded,
                    "agentic_turn_limit_hits": sum(
                        result.get("turn_limit_reached") is True
                        for result in agentic_results
                    ),
                    "agentic_length_terminated_turns": sum(
                        int(result["length_terminated_turns"])
                        for result in agentic_results
                    ),
                }
            )
        if kind == "memory":
            memory_results = [event["result"] for event in requests]
            memory_run_results.extend(memory_results)
            memory_aggregate = summarize_memory_operation_results(memory_results)
            succeeded = sum(
                result.get("passed") is True for result in memory_results
            )
            graphiti_results = [
                result
                for result in memory_results
                if result.get("graphiti_resolver_case") is True
            ]
            extension_results = [
                result
                for result in memory_results
                if result.get("synthetic_extension_case") is True
            ]
            row.update(
                {
                    "memory_operations": len(memory_results),
                    "memory_operations_correct": succeeded,
                    "memory_operation_accuracy": succeeded / len(memory_results),
                    "memory_json_objects_emitted": sum(
                        result.get("json_object_emitted") is True
                        for result in memory_results
                    ),
                    "memory_schema_valid": sum(
                        result.get("schema_valid") is True
                        for result in memory_results
                    ),
                    "memory_protected_value_emissions": sum(
                        result.get("protected_value_emitted") is True
                        for result in memory_results
                    ),
                    "memory_action_correct": sum(
                        result.get("action_correct") is True
                        for result in extension_results
                    ),
                    "memory_target_correct": sum(
                        result.get("target_correct") is True
                        for result in memory_results
                    ),
                    "memory_path_correct": sum(
                        result.get("path_correct") is True
                        for result in memory_results
                    ),
                    "memory_tier_correct": sum(
                        result.get("tier_correct") is True
                        for result in memory_results
                    ),
                    "memory_evidence_correct": sum(
                        result.get("evidence_correct") is True
                        for result in memory_results
                    ),
                    "memory_value_correct": sum(
                        result.get("value_correct") is True
                        for result in memory_results
                    ),
                    "memory_valid_from_correct": sum(
                        result.get("valid_from_correct") is True
                        for result in memory_results
                    ),
                    "memory_valid_to_correct": sum(
                        result.get("valid_to_correct") is True
                        for result in memory_results
                    ),
                    "memory_reason_correct": sum(
                        result.get("reason_correct") is True
                        for result in memory_results
                    ),
                    "memory_field_checks_applicable": len(extension_results),
                    "memory_mutations_expected": sum(
                        result.get("mutation_expected") is True
                        for result in memory_results
                    ),
                    "memory_mutations_selected": sum(
                        result.get("mutation_selected") is True
                        for result in memory_results
                    ),
                    "memory_unexpected_tool_calls": memory_aggregate[
                        "unexpected_tool_calls"
                    ],
                    "memory_prompt_cache_disabled_requests": memory_aggregate[
                        "prompt_cache_disabled_requests"
                    ],
                    "memory_zero_cached_prompt_requests": memory_aggregate[
                        "zero_cached_prompt_requests"
                    ],
                    "memory_total_server_prompt_s": memory_aggregate[
                        "total_server_prompt_s"
                    ],
                    "memory_total_server_decode_s": memory_aggregate[
                        "total_server_decode_s"
                    ],
                    "memory_secret_refusals_required": sum(
                        result.get("secret_refusal_required") is True
                        for result in memory_results
                    ),
                    "memory_secret_refusals_succeeded": sum(
                        result.get("secret_refusal_succeeded") is True
                        for result in memory_results
                    ),
                    "memory_injection_refusals_required": sum(
                        result.get("injection_refusal_required") is True
                        for result in memory_results
                    ),
                    "memory_injection_refusals_succeeded": sum(
                        result.get("injection_refusal_succeeded") is True
                        for result in memory_results
                    ),
                    "memory_total_prompt_tokens": prompt_tokens,
                    "memory_total_completion_tokens": completion_tokens,
                    # Preserve unknown reasoning usage: the generic collector
                    # returns None when even one request omits the counter.
                    "memory_total_reasoning_tokens": reasoning_tokens,
                    "graphiti_resolver_operations": len(graphiti_results),
                    "synthetic_memory_extension_operations": len(
                        extension_results
                    ),
                }
            )
            if graphiti_results:
                resolver_correct = memory_aggregate["graphiti_resolver"]["correct"]
                row.update(
                    {
                        "graphiti_resolver_correct": resolver_correct,
                        "graphiti_resolver_accuracy": (
                            resolver_correct / len(graphiti_results)
                        ),
                        "graphiti_duplicate_sets_correct": sum(
                            result.get("duplicate_facts_correct") is True
                            for result in graphiti_results
                        ),
                        "graphiti_contradicted_sets_correct": sum(
                            result.get("contradicted_facts_correct") is True
                            for result in graphiti_results
                        ),
                        "graphiti_resolver_confusion": memory_aggregate[
                            "graphiti_resolver"
                        ]["confusion"],
                    }
                )
        if diffusion_generation:
            row["median_time_to_first_emission_s"] = (
                statistics.median(first_emission_times)
                if first_emission_times
                else None
            )
            row["p95_time_to_first_emission_s"] = (
                percentile(first_emission_times, 0.95)
                if len(first_emission_times) >= 20
                else None
            )
            if kind in {"decode", "concurrency"} and valid_generation:
                block_generation_rates = []
                for event in requests:
                    result = event["result"]
                    rate = result.get("block_generation_output_tps")
                    if not isinstance(rate, (int, float)):
                        legacy_rate = result.get("output_tps")
                        rate = (
                            legacy_rate
                            if isinstance(legacy_rate, (int, float))
                            else int(result["completion_tokens"])
                            / max(float(result["elapsed_s"]), 1e-9)
                        )
                    block_generation_rates.append(float(rate))
                row.update(
                    {
                        "median_block_generation_output_tps": statistics.median(
                            block_generation_rates
                        ),
                        "block_generation_metric_source": (
                            "client_completion_tokens_per_end_to_end_request_elapsed"
                        ),
                        "aggregate_block_generation_output_tps": (
                            completion_tokens / max(wall_s, 1e-9)
                        ),
                        "aggregate_block_generation_metric_source": (
                            "case_completion_tokens_per_measured_wall_time"
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "median_block_generation_output_tps": None,
                        "block_generation_metric_source": None,
                        "aggregate_block_generation_output_tps": None,
                        "aggregate_block_generation_metric_source": None,
                    }
                )
        if kind == "prefill" and diffusion_generation:
            row.update(
                {
                    "median_prefill_tps": None,
                    "p95_prefill_tps": None,
                    "prefill_metric_source": (
                        "unavailable_for_diffusion_block_generation"
                    ),
                    "median_approximate_prefill_tps": None,
                    "p95_approximate_prefill_tps": None,
                }
            )
        elif kind == "prefill":
            server_prompt_times = [
                event["result"].get("server_prompt_s") for event in requests
            ]
            native_timing = all(
                isinstance(value, (int, float)) and value > 0
                for value in server_prompt_times
            )
            prefill_rates = [
                int(event["result"]["prompt_tokens"])
                / max(
                    float(server_prompt_times[index])
                    if native_timing
                    else float(event["result"]["ttft_s"]),
                    1e-9,
                )
                for index, event in enumerate(requests)
            ]
            row["median_prefill_tps"] = statistics.median(prefill_rates)
            row["p95_prefill_tps"] = (
                percentile(prefill_rates, 0.95) if len(prefill_rates) >= 20 else None
            )
            row["prefill_metric_source"] = (
                "server_reported_prompt_eval_duration"
                if native_timing
                else "client_ttft_approximation"
            )
            row["median_approximate_prefill_tps"] = (
                None if native_timing else statistics.median(prefill_rates)
            )
            row["p95_approximate_prefill_tps"] = (
                percentile(prefill_rates, 0.95)
                if not native_timing and len(prefill_rates) >= 20
                else None
            )
        embedding_results = [
            event["result"] for event in requests if "dimension" in event["result"]
        ]
        rerank_results = [
            event["result"] for event in requests if "candidate_count" in event["result"]
        ]
        multimodal_embedding_results = [
            event["result"]
            for event in requests
            if "relevant_similarity" in event["result"]
        ]
        if embedding_results:
            row["median_ttft_s"] = None
            row["p95_ttft_s"] = None
            row["aggregate_output_tps"] = None
            row.update(
                {
                    "embedding_dimension": embedding_results[0]["dimension"],
                    "embedding_batch_size": embedding_results[0]["batch_size"],
                    "median_embedding_items_s": statistics.median(
                        float(result["items_per_s"]) for result in embedding_results
                    ),
                    "embeddings_finite": all(
                        bool(result["finite"]) for result in embedding_results
                    ),
                }
            )
        if rerank_results:
            candidate_counts = {
                int(result["candidate_count"]) for result in rerank_results
            }
            top_indexes = {int(result["top_index"]) for result in rerank_results}
            rankings = {
                tuple(int(index) for index in result["ranking"])
                for result in rerank_results
            }
            rerank_pairs = sum(
                int(result["candidate_count"]) for result in rerank_results
            )
            row["median_ttft_s"] = None
            row["p95_ttft_s"] = None
            row["aggregate_output_tps"] = None
            row.update(
                {
                    "rerank_candidates_per_request": (
                        next(iter(candidate_counts))
                        if len(candidate_counts) == 1
                        else None
                    ),
                    "rerank_pairs": rerank_pairs,
                    "median_rerank_pairs_s": statistics.median(
                        float(result["pairs_per_s"]) for result in rerank_results
                    ),
                    "aggregate_rerank_pairs_s": rerank_pairs / max(wall_s, 1e-9),
                    "rerank_scores_finite": all(
                        bool(result["finite"]) for result in rerank_results
                    ),
                    "rerank_top_index": (
                        next(iter(top_indexes)) if len(top_indexes) == 1 else None
                    ),
                    "rerank_ranking_stable": len(rankings) == 1,
                    "rerank_validation_passed": case_event.get(
                        "validation_passed"
                    ),
                }
            )
        if multimodal_embedding_results:
            def median_numeric(key: str) -> float | None:
                values = [
                    float(result[key])
                    for result in multimodal_embedding_results
                    if isinstance(result.get(key), (int, float))
                    and math.isfinite(float(result[key]))
                ]
                return statistics.median(values) if values else None

            row.update(
                {
                    "median_image_embedding_latency_s": median_numeric(
                        "image_latency_s"
                    ),
                    "median_relevant_text_embedding_latency_s": median_numeric(
                        "relevant_text_latency_s"
                    ),
                    "median_unrelated_text_embedding_latency_s": median_numeric(
                        "unrelated_text_latency_s"
                    ),
                    "median_relevant_similarity": median_numeric(
                        "relevant_similarity"
                    ),
                    "median_unrelated_similarity": median_numeric(
                        "unrelated_similarity"
                    ),
                    "median_similarity_margin": median_numeric(
                        "similarity_margin"
                    ),
                    "multimodal_embeddings_finite": all(
                        bool(result["finite"])
                        for result in multimodal_embedding_results
                    ),
                    "multimodal_embedding_validation_passed": case_event.get(
                        "validation_passed"
                    ),
                }
            )
        if kind == "quality":
            quality_validations = [
                event.get("validation")
                if isinstance(event.get("validation"), dict)
                else {}
                for event in requests
            ]
            quality_correct = sum(
                validation.get("passed") is True
                for validation in quality_validations
            )
            category_totals: dict[str, int] = {}
            category_correct: dict[str, int] = {}
            for validation in quality_validations:
                category = validation.get("quality_category")
                if not isinstance(category, str) or not category:
                    continue
                category_totals[category] = category_totals.get(category, 0) + 1
                if validation.get("passed") is True:
                    category_correct[category] = category_correct.get(category, 0) + 1
            row.update(
                {
                    "quality_items": len(requests),
                    "quality_scored_items": sum(category_totals.values()),
                    "quality_correct": quality_correct,
                    "quality_accuracy": quality_correct / len(requests),
                    "quality_total_prompt_tokens": prompt_tokens,
                    "quality_total_completion_tokens": completion_tokens,
                    "quality_total_reasoning_tokens": reasoning_tokens,
                    "quality_total_request_latency_s": sum(elapsed),
                    "quality_accuracy_by_category": {
                        category: category_correct.get(category, 0) / total
                        for category, total in sorted(category_totals.items())
                    },
                }
            )
        phase_telemetry = telemetry.get(f"case:{case_id}:{attempt_id}") or telemetry.get(
            case_id
        )
        if phase_telemetry:
            row["telemetry"] = phase_telemetry
            sampled_energy_j = phase_telemetry.get("sampled_energy_j")
            if isinstance(sampled_energy_j, (int, float)) and sampled_energy_j > 0:
                if kind == "agentic":
                    tasks_succeeded = int(row["agentic_tasks_succeeded"])
                    row["agentic_tasks_succeeded_per_sampled_joule"] = (
                        tasks_succeeded / sampled_energy_j
                    )
                    row["agentic_sampled_energy_j_per_solved_task"] = (
                        sampled_energy_j / tasks_succeeded
                        if tasks_succeeded
                        else None
                    )
                elif embedding_results:
                    embedded_items = sum(
                        int(result["batch_size"]) for result in embedding_results
                    )
                    row["embedding_items_per_sampled_joule"] = (
                        embedded_items / sampled_energy_j
                    )
                elif rerank_results:
                    row["rerank_pairs_per_sampled_joule"] = (
                        rerank_pairs / sampled_energy_j
                    )
                elif kind == "prefill" and not diffusion_generation:
                    row["prompt_tokens_per_sampled_joule"] = (
                        prompt_tokens / sampled_energy_j
                    )
                elif (
                    diffusion_generation
                    and kind in {"decode", "concurrency"}
                    and valid_generation
                    and completion_tokens
                ):
                    row["block_generation_output_tokens_per_sampled_joule"] = (
                        completion_tokens / sampled_energy_j
                    )
                elif completion_tokens and (
                    valid_generation or kind not in {"decode", "concurrency"}
                ):
                    row["output_tokens_per_sampled_joule"] = (
                        completion_tokens / sampled_energy_j
                    )
        rows.append(row)
    completed_ids = set(completed)
    failed_ids = {
        str(event["case_id"])
        for event in events
        if event.get("event") == "case_failed" and str(event["case_id"]) not in completed_ids
    }
    unimplemented_ids = {
        str(event["case_id"])
        for event in events
        if event.get("event") == "case_skipped_adapter_unimplemented"
        and str(event["case_id"]) not in completed_ids
    }
    unsupported_ids = {
        str(event["case_id"])
        for event in events
        if event.get("event") == "case_skipped_unsupported"
        and str(event["case_id"]) not in completed_ids
    }
    context_limited_ids = {
        str(event["case_id"])
        for event in events
        if event.get("event") == "case_skipped_context_limit"
        and str(event["case_id"]) not in completed_ids
    }
    validation_failed_ids = {
        case_id
        for case_id, attempt_id in completed.items()
        if case_events[(case_id, attempt_id)].get("validation_passed") is False
    }
    last_start = max(
        (index for index, event in enumerate(events) if event.get("event") == "run_start"),
        default=-1,
    )
    last_finish = max(
        (index for index, event in enumerate(events) if event.get("event") == "run_complete"),
        default=-1,
    )
    last_abort = max(
        (index for index, event in enumerate(events) if event.get("event") == "run_aborted"),
        default=-1,
    )
    last_cleanup_failure = max(
        (index for index, event in enumerate(events) if event.get("event") == "cleanup_failed"),
        default=-1,
    )
    last_cleanup_success = max(
        (
            index
            for index, event in enumerate(events)
            if event.get("event") in {"server_stopped", "server_kept"}
        ),
        default=-1,
    )
    last_completion_status = next(
        (
            event.get("status")
            for event in reversed(events)
            if event.get("event") == "run_complete"
        ),
        None,
    )
    status = "complete"
    if last_start < 0:
        status = "not_started"
    elif last_abort > last_start and last_abort > last_finish:
        status = "aborted"
    elif last_finish < last_start:
        status = "incomplete"
    elif last_cleanup_failure > max(last_start, last_cleanup_success):
        status = "partial"
    elif last_completion_status == "no_work":
        status = "no_work"
    elif failed_ids or unimplemented_ids or validation_failed_ids:
        status = "partial"
    elif mtp_requested and not llamacpp_mtp_evidence["passed"]:
        status = "partial"
    elif dflash_requested and not llamacpp_dflash_evidence["passed"]:
        status = "partial"
    elif sm121_storage_lifecycle_issues:
        status = "partial"
    run_error = None
    if status == "aborted" and last_abort >= 0:
        aborted_event = events[last_abort]
        run_error = {
            key: aborted_event.get(key)
            for key in ("stage", "error_type", "error")
        }
    summary = {
        "run_dir": str(run_dir),
        "model": (
            {
                key: planned_model.get(key)
                for key in (
                    "id",
                    "backend",
                    "source",
                    "revision",
                    "architecture",
                    "quantization",
                    "support_status",
                    "max_context",
                    "native_context",
                )
            }
            if planned_model
            else None
        ),
        "suite": planned_suite.get("id"),
        "status": status,
        "run_completion_status": last_completion_status,
        "run_error": run_error,
        "completed_cases": len(rows),
        "failed_cases": sorted(failed_ids),
        "validation_failed_cases": sorted(validation_failed_ids),
        "unimplemented_cases": sorted(unimplemented_ids),
        "unsupported_cases": sorted(unsupported_ids),
        "context_limited_cases": sorted(context_limited_ids),
        "measurement_annotations": annotations,
        "startup_measurement_valid": not startup_annotations,
        "startup_measurement_annotations": startup_annotations,
        "startup_safety_gates": startup_safety_gates,
        "measurement_invalid_cases": sorted(case_annotations),
        "startup_telemetry": telemetry.get("server_startup"),
        "first_request_after_start": next(
            (
                event.get("result")
                for event in reversed(events)
                if event.get("event") == "first_request_complete"
            ),
            None,
        ),
        "first_request_telemetry": telemetry.get("first_request_after_start"),
        "shutdown_telemetry": telemetry.get("server_shutdown"),
        "artifact_validation": next(
            (
                {
                    key: event.get(key)
                    for key in (
                        "elapsed_s",
                        "runtime_binary_sha256",
                        "model_sha256",
                        "model_shard_count",
                        "model_total_size_bytes",
                        "model_shard_sha256s",
                        "mmproj_sha256",
                        "draft_model_sha256",
                    )
                    if key in event
                }
                for event in reversed(events)
                if event.get("event") == "artifact_validation_complete"
            ),
            None,
        ),
        "artifact_validation_telemetry": telemetry.get("artifact_validation"),
        "speculative_decoding": speculative_decoding,
        "llamacpp_mtp_evidence": (
            llamacpp_mtp_evidence if mtp_requested else None
        ),
        "llamacpp_dflash_evidence": (
            llamacpp_dflash_evidence if dflash_requested else None
        ),
        "cases": rows,
    }
    if memory_run_results:
        memory_battery_completed = (
            planned_suite.get("id") == MEMORY_OPERATION_SUITE_ID
            and all(case_id in completed for case_id in planned_case_ids)
        )
        summary["memory_operation_summary"] = summarize_memory_operation_results(
            memory_run_results,
            require_complete=memory_battery_completed,
        )
    write_json(run_dir / "summary.json", summary)
    if rows:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    return summary
