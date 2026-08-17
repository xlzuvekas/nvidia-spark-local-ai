"""Result aggregation for SparkBench JSONL journals."""

from __future__ import annotations

import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .annotations import measurement_annotations
from .journal import write_json
from .llamacpp_metrics import (
    aggregate_llamacpp_spec_decode_metrics,
    assess_llamacpp_mtp_evidence,
    assess_llamacpp_mtp_proposal_depth,
    llamacpp_dflash_requested,
    llamacpp_mtp_depth,
    llamacpp_mtp_requested,
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
    annotations = measurement_annotations(events)
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
    speculative_decoding = llamacpp_speculative_decoding or (
        aggregate_vllm_spec_decode_metrics(
            event["metrics"]
            for event in events
            if event.get("event") == "vllm_spec_decode_metrics_snapshot"
            and isinstance(event.get("metrics"), dict)
        )
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
                if embedding_results:
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
                        "mmproj_sha256",
                        "draft_model_sha256",
                    )
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
    write_json(run_dir / "summary.json", summary)
    if rows:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    return summary
