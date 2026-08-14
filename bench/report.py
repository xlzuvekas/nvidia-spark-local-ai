"""Result aggregation for SparkBench JSONL journals."""

from __future__ import annotations

import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .journal import write_json


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
        numeric = lambda key: [
            float(sample[key])
            for sample in phase_samples
            if isinstance(sample.get(key), (int, float))
        ]
        powers = numeric("power_w")
        temperatures = numeric("temperature_c")
        utilizations = numeric("gpu_util_pct")
        clocks = numeric("sm_clock_mhz")
        memory_kib = numeric("memavailable_kib")
        sampled_energy_j = 0.0
        for before, after in zip(phase_samples, phase_samples[1:]):
            if not isinstance(before.get("power_w"), (int, float)) or not isinstance(
                after.get("power_w"), (int, float)
            ):
                continue
            try:
                delta_s = (
                    datetime.fromisoformat(str(after["timestamp"]))
                    - datetime.fromisoformat(str(before["timestamp"]))
                ).total_seconds()
            except (KeyError, ValueError):
                continue
            if 0 < delta_s <= 5:
                sampled_energy_j += (
                    float(before["power_w"]) + float(after["power_w"])
                ) / 2 * delta_s
        summaries[phase] = {
            "samples": len(phase_samples),
            "average_power_w": statistics.fmean(powers) if powers else None,
            "peak_power_w": max(powers) if powers else None,
            "sampled_energy_j": sampled_energy_j if len(powers) >= 2 else None,
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
    completed: dict[str, str] = {}
    case_events: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        if event.get("event") == "case_complete":
            case_id = str(event["case_id"])
            attempt_id = str(event["attempt_id"])
            completed[case_id] = attempt_id
            case_events[(case_id, attempt_id)] = event

    rows: list[dict[str, Any]] = []
    for case_id, attempt_id in sorted(completed.items()):
        requests = [
            event
            for event in events
            if event.get("event") == "request_complete"
            and event.get("case_id") == case_id
            and event.get("attempt_id") == attempt_id
        ]
        if not requests:
            continue
        ttfts = [float(event["result"]["ttft_s"]) for event in requests]
        elapsed = [float(event["result"]["elapsed_s"]) for event in requests]
        decode_rates = [float(event["result"]["decode_tps"]) for event in requests]
        decode_sources = {
            str(event["result"].get("decode_metric_source", "client_estimate"))
            for event in requests
        }
        prompt_tokens = sum(int(event["result"]["prompt_tokens"]) for event in requests)
        completion_tokens = sum(int(event["result"]["completion_tokens"]) for event in requests)
        case_event = case_events[(case_id, attempt_id)]
        kind = str(case_event.get("kind") or requests[0].get("kind", "unknown"))
        valid_generation = case_event.get("validation_passed") is not False
        wall_s = float(case_event.get("elapsed_s") or sum(elapsed))
        row: dict[str, Any] = {
            "case_id": case_id,
            "attempt_id": attempt_id,
            "kind": kind,
            "requests": len(requests),
            "concurrency": case_event.get("concurrency", 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "median_ttft_s": statistics.median(ttfts),
            "p95_ttft_s": percentile(ttfts, 0.95) if len(ttfts) >= 20 else None,
            "median_e2e_s": statistics.median(elapsed),
            "p95_e2e_s": percentile(elapsed, 0.95) if len(elapsed) >= 20 else None,
            "median_decode_tps": (
                statistics.median(decode_rates)
                if kind in {"decode", "concurrency"} and valid_generation
                else None
            ),
            "decode_metric_source": (
                next(iter(decode_sources))
                if kind in {"decode", "concurrency"}
                and valid_generation
                and len(decode_sources) == 1
                else (
                    "mixed"
                    if kind in {"decode", "concurrency"} and valid_generation
                    else None
                )
            ),
            "median_estimated_decode_tps": (
                statistics.median(decode_rates)
                if kind in {"decode", "concurrency"}
                and valid_generation
                and decode_sources == {"client_estimate"}
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
                and decode_sources == {"client_estimate"}
                else None
            ),
            "aggregate_output_tps": (
                completion_tokens / max(wall_s, 1e-9)
                if valid_generation or kind not in {"decode", "concurrency"}
                else None
            ),
            "request_tps": len(requests) / max(wall_s, 1e-9),
            "elapsed_s": wall_s,
            "validation_passed": case_event.get("validation_passed"),
        }
        if kind == "prefill":
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
        phase_telemetry = telemetry.get(f"case:{case_id}:{attempt_id}") or telemetry.get(
            case_id
        )
        if phase_telemetry:
            row["telemetry"] = phase_telemetry
            sampled_energy_j = phase_telemetry.get("sampled_energy_j")
            if isinstance(sampled_energy_j, (int, float)) and sampled_energy_j > 0:
                if embedding_results:
                    embedded_items = sum(
                        int(result["batch_size"]) for result in embedding_results
                    )
                    row["embedding_items_per_sampled_joule"] = (
                        embedded_items / sampled_energy_j
                    )
                elif kind == "prefill":
                    row["prompt_tokens_per_sampled_joule"] = (
                        prompt_tokens / sampled_energy_j
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
        "completed_cases": len(rows),
        "failed_cases": sorted(failed_ids),
        "validation_failed_cases": sorted(validation_failed_ids),
        "unimplemented_cases": sorted(unimplemented_ids),
        "unsupported_cases": sorted(unsupported_ids),
        "context_limited_cases": sorted(context_limited_ids),
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
        "cases": rows,
    }
    write_json(run_dir / "summary.json", summary)
    if rows:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    return summary
