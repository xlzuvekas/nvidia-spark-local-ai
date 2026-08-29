"""Dependency-free, read-only validation of completed SparkBench matrices."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

from .sglang_sm121_storage import (
    SM121StorageCandidateError,
    is_sm121_storage_canary_plan,
    sm121_storage_canary_lifecycle_issues,
    validate_sm121_storage_candidate,
    validate_sm121_storage_suite,
)
from .sglang_sm121_cache_observability import (
    SM121CacheObservabilityError,
    is_sm121_cache_observability_plan,
    sm121_cache_observability_lifecycle_issues,
    validate_sm121_cache_observability_candidate,
    validate_sm121_cache_observability_suite,
)
from .sglang_sm121_cache_semantic import (
    SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
    SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID,
    SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
    SM121_CACHE_SEMANTIC_EXECUTION_MODE,
    SM121_CACHE_SEMANTIC_PAIR_BINDING_SCHEMA_VERSION,
    SM121CacheSemanticError,
    is_sm121_cache_semantic_plan,
    sm121_cache_semantic_arm,
    sm121_cache_semantic_cache_off_receipt_sha256,
    sm121_cache_semantic_lifecycle_issues,
    sm121_cache_semantic_pair_binding_sha256,
    sm121_cache_semantic_pair_instance_sha256,
    validate_sm121_cache_semantic_pair_binding,
    validate_sm121_cache_semantic_candidate,
    validate_sm121_cache_semantic_suite,
)
from .sglang_sm121_cache_performance import (
    SM121_CACHE_PERFORMANCE_ARM_ORDER,
    SM121_CACHE_PERFORMANCE_CAMPAIGN_ID,
    SM121_CACHE_PERFORMANCE_EXECUTION_MODE,
    SM121_CACHE_PERFORMANCE_RUNTIME_EVENT,
    SM121_CACHE_PERFORMANCE_STATIC_EVENT,
    SM121_CACHE_PERFORMANCE_TIMED_TURNS,
    SM121_CACHE_PERFORMANCE_TURN_EVENT,
    SM121CachePerformanceError,
    score_sm121_cache_performance_campaign,
    validate_sm121_cache_performance_runtime_event,
    validate_sm121_cache_performance_static_event,
    validate_sm121_cache_performance_turn_event,
)
from . import sm121_chunked_prefill_evidence as chunked_prefill_evidence


IssueAdder = Callable[..., None]

_CASE_OUTCOMES = {
    "case_complete",
    "case_failed",
    "case_skipped_adapter_unimplemented",
    "case_skipped_context_limit",
    "case_skipped_unsupported",
}
_CASE_EVENTS = _CASE_OUTCOMES | {"case_start", "request_complete"}

_SM121_CACHE_SEMANTIC_ADMISSION_ISSUE_CODES = frozenset(
    {
        "semantic_t0_prompt_window",
        "semantic_shared_prefix_window",
        "semantic_append_identity",
        "semantic_metrics_unavailable",
        "semantic_guardrail_metrics_unavailable",
        "semantic_metric_settle_polls",
        "semantic_metric_settle",
        "semantic_input_delta",
        "semantic_cache_guardrail",
        "semantic_zero_hit_details",
        "semantic_zero_hit_native",
        "semantic_positive_detail",
        "semantic_positive_native_reconciliation",
        "semantic_usage_reconciliation",
        "semantic_case_validation",
    }
)
_SM121_CACHE_SEMANTIC_AUDIT_PATH_KEYS = frozenset(
    {"path", "run_dir", "summary_run_dir"}
)


def _redact_sm121_cache_semantic_audit_value(value: Any, *, root: Path) -> Any:
    """Remove local filesystem identity from a semantic audit report."""

    if isinstance(value, dict):
        return {
            key: _redact_sm121_cache_semantic_audit_value(item, root=root)
            for key, item in value.items()
            if key not in _SM121_CACHE_SEMANTIC_AUDIT_PATH_KEYS
        }
    if isinstance(value, list):
        return [
            _redact_sm121_cache_semantic_audit_value(item, root=root)
            for item in value
        ]
    if isinstance(value, str):
        return value.replace(str(root), "<local-run>")
    return value


def _sm121_cache_semantic_audit_run_id(root: Path) -> str:
    """Return the only local-run identifier permitted in semantic audit output."""

    return root.name


def _resolve_matrix_run_reference(raw: str, matrix_root: Path) -> Path:
    """Resolve a run reference without consulting the process working directory."""

    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()

    matrix_relative = (matrix_root / candidate).resolve()
    if matrix_relative.is_dir():
        return matrix_relative

    # Older matrix writers serialized the complete path assembled from a
    # relative --results argument (for example
    # results/matrices/<matrix>/<run>). Recover only that exact suffix shape,
    # without resolving its discarded prefix against the current directory.
    parts = candidate.parts
    is_legacy_reference = (
        len(parts) >= 3
        and parts[-3] == "matrices"
        and parts[-2] == matrix_root.name
        and all(part not in {"", ".", ".."} for part in parts)
    )
    if is_legacy_reference:
        legacy_run_dir = (matrix_root / parts[-1]).resolve()
        if legacy_run_dir.parent == matrix_root and legacy_run_dir.is_dir():
            return legacy_run_dir

    return matrix_relative


def _content_hash(value: Any, length: int = 16) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]


def _load_json_object(
    path: Path,
    *,
    add_issue: IssueAdder,
    run: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        add_issue(
            "missing_json_file",
            f"required JSON file is missing: {path}",
            run=run,
            path=str(path),
        )
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        add_issue(
            "invalid_json_file",
            f"cannot read JSON object {path}: {error}",
            run=run,
            path=str(path),
        )
        return None
    if not isinstance(value, dict):
        add_issue(
            "invalid_json_object",
            f"expected a JSON object in {path}",
            run=run,
            path=str(path),
        )
        return None
    return value


def _load_jsonl(
    path: Path,
    *,
    add_issue: IssueAdder,
    run: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        text = path.read_text()
    except FileNotFoundError:
        add_issue(
            "missing_events_jsonl",
            f"event journal is missing: {path}",
            run=run,
            path=str(path),
        )
        return []
    except (OSError, UnicodeError) as error:
        add_issue(
            "unreadable_events_jsonl",
            f"cannot read event journal {path}: {error}",
            run=run,
            path=str(path),
        )
        return []

    if not text:
        add_issue(
            "empty_events_jsonl",
            f"event journal is empty: {path}",
            run=run,
            path=str(path),
        )
        return []
    if not text.endswith("\n"):
        add_issue(
            "unterminated_events_jsonl",
            f"event journal does not end with a newline: {path}",
            run=run,
            path=str(path),
        )

    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            add_issue(
                "blank_jsonl_record",
                f"blank event journal record at line {line_number}",
                run=run,
                path=str(path),
                line=line_number,
            )
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            add_issue(
                "invalid_jsonl_record",
                f"invalid event journal record at line {line_number}: {error}",
                run=run,
                path=str(path),
                line=line_number,
            )
            continue
        if not isinstance(event, dict):
            add_issue(
                "non_object_jsonl_record",
                f"event journal line {line_number} is not an object",
                run=run,
                path=str(path),
                line=line_number,
            )
            continue
        if not isinstance(event.get("event"), str) or not event["event"]:
            add_issue(
                "missing_event_type",
                f"event journal line {line_number} has no event type",
                run=run,
                path=str(path),
                line=line_number,
            )
        events.append(event)
    return events


def _max_num_seqs(model: dict[str, Any]) -> int | None:
    args = model.get("args")
    if not isinstance(args, list):
        return None
    value: Any = None
    for index, argument in enumerate(args):
        if argument == "--max-num-seqs" and index + 1 < len(args):
            value = args[index + 1]
        elif isinstance(argument, str) and argument.startswith("--max-num-seqs="):
            value = argument.split("=", 1)[1]
    try:
        parsed = int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed and parsed > 0 else None


def _audit_plan(
    plan: dict[str, Any],
    *,
    matrix_suite: str | None,
    model_id: str,
    run_dir: Path,
    add_issue: IssueAdder,
    run: dict[str, Any],
) -> tuple[set[str], int | None]:
    model = plan.get("model")
    suite = plan.get("suite")
    if not isinstance(model, dict):
        add_issue("invalid_plan_model", "plan.model must be an object", run=run)
        return set(), None
    if not isinstance(suite, dict):
        add_issue("invalid_plan_suite", "plan.suite must be an object", run=run)
        return set(), _max_num_seqs(model)

    if model.get("id") != model_id:
        add_issue(
            "plan_model_mismatch",
            f"matrix model {model_id!r} does not match plan model {model.get('id')!r}",
            run=run,
        )
    if matrix_suite is not None and suite.get("id") != matrix_suite:
        add_issue(
            "plan_suite_mismatch",
            f"matrix suite {matrix_suite!r} does not match plan suite {suite.get('id')!r}",
            run=run,
        )

    fingerprint = plan.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        add_issue("missing_plan_fingerprint", "plan fingerprint is missing", run=run)
    elif not run_dir.name.endswith("-" + fingerprint[:8]):
        add_issue(
            "run_directory_fingerprint_mismatch",
            "run directory does not end with the plan fingerprint prefix",
            run=run,
            fingerprint=fingerprint,
        )

    cases = suite.get("cases")
    if not isinstance(cases, list):
        add_issue("invalid_plan_cases", "plan suite cases must be a list", run=run)
        return set(), _max_num_seqs(model)

    case_ids: list[str] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            add_issue(
                "invalid_plan_case",
                f"plan case {index} is not an object",
                run=run,
                case_index=index,
            )
            continue
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            add_issue(
                "missing_plan_case_id",
                f"plan case {index} has no case_id",
                run=run,
                case_index=index,
            )
            continue
        case_ids.append(case_id)
        case_without_id = {key: value for key, value in case.items() if key != "case_id"}
        expected_case_id = (
            f"{case.get('id')}--"
            f"{_content_hash({'model': model, 'case': case_without_id}, 12)}"
        )
        if case_id != expected_case_id:
            add_issue(
                "plan_case_identity_mismatch",
                f"case identity does not match frozen model and case: {case_id}",
                run=run,
                case_id=case_id,
                expected_case_id=expected_case_id,
            )
    if len(case_ids) != len(set(case_ids)):
        add_issue("duplicate_plan_case_id", "plan contains duplicate case IDs", run=run)

    suite_without_case_ids = {
        **suite,
        "cases": [
            {key: value for key, value in case.items() if key != "case_id"}
            for case in cases
            if isinstance(case, dict)
        ],
    }
    try:
        schema_version = int(plan.get("schema_version", 1))
    except (TypeError, ValueError):
        schema_version = -1
    if schema_version not in {1, 2}:
        add_issue(
            "unsupported_plan_schema",
            f"unsupported frozen plan schema: {plan.get('schema_version')!r}",
            run=run,
        )
    elif schema_version >= 2:
        integrity_hash = plan.get("integrity_hash")
        if not isinstance(integrity_hash, str) or not integrity_hash:
            add_issue("missing_plan_integrity_hash", "plan integrity hash is missing", run=run)
        else:
            if len(integrity_hash) != 64:
                add_issue(
                    "invalid_plan_integrity_hash_length",
                    "schema-2 plan integrity hash must contain 64 hex characters",
                    run=run,
                )
            integrity_payload = {
                key: value for key, value in plan.items() if key != "integrity_hash"
            }
            if _content_hash(integrity_payload, len(integrity_hash)) != integrity_hash:
                add_issue(
                    "plan_integrity_mismatch",
                    "plan integrity hash does not match its contents",
                    run=run,
                )
        expected_fingerprint = _content_hash(
            {
                "model": model,
                "suite": suite_without_case_ids,
                "resolved": plan.get("resolved", {}),
            }
        )
        if isinstance(fingerprint, str) and fingerprint != expected_fingerprint:
            add_issue(
                "plan_fingerprint_mismatch",
                "plan fingerprint does not match its frozen model, suite, and resolution",
                run=run,
                expected_fingerprint=expected_fingerprint,
            )
    else:
        expected_fingerprint = _content_hash(
            {"model": model, "suite": suite_without_case_ids}
        )
        if isinstance(fingerprint, str) and fingerprint != expected_fingerprint:
            add_issue(
                "plan_fingerprint_mismatch",
                "plan fingerprint does not match its frozen model and suite",
                run=run,
                expected_fingerprint=expected_fingerprint,
            )
        expected_digest = model.get("image_digest")
        resolved_digest = (plan.get("resolved") or {}).get("image_digest")
        if expected_digest and not (
            isinstance(resolved_digest, str)
            and resolved_digest.endswith("@" + str(expected_digest))
        ):
            add_issue(
                "legacy_plan_image_mismatch",
                "legacy plan image digest does not match its resolved image",
                run=run,
            )

    return set(case_ids), _max_num_seqs(model)


def _event_indexes(events: list[dict[str, Any]], event_type: str) -> list[int]:
    return [index for index, event in enumerate(events) if event.get("event") == event_type]


def _audit_lifecycle(
    events: list[dict[str, Any]],
    *,
    planned_case_ids: set[str],
    add_issue: IssueAdder,
    run: dict[str, Any],
) -> None:
    starts = _event_indexes(events, "run_start")
    finishes = _event_indexes(events, "run_complete")
    aborts = _event_indexes(events, "run_aborted")
    if not starts:
        add_issue("missing_run_start", "journal has no run_start event", run=run)
        return
    final_start = starts[-1]
    final_finish = max((index for index in finishes if index > final_start), default=-1)
    final_abort = max((index for index in aborts if index > final_start), default=-1)
    final_terminal = max(final_finish, final_abort)
    if final_terminal < 0:
        add_issue(
            "missing_run_complete",
            "final run_start has no later run_complete or run_aborted event",
            run=run,
        )
        final_terminal = len(events)
    aborted_terminal = final_abort > final_finish
    if aborted_terminal:
        run["terminal_event"] = "run_aborted"
    elif final_finish >= 0:
        run["terminal_event"] = "run_complete"
    else:
        run["terminal_event"] = None
    if aborted_terminal:
        aborted_event = events[final_abort]
        for field in ("stage", "error_type", "error"):
            if not isinstance(aborted_event.get(field), str) or not str(
                aborted_event[field]
            ).strip():
                add_issue(
                    "invalid_run_abort_context",
                    f"terminal run_aborted event has no non-empty {field}",
                    run=run,
                    field=field,
                    event_index=final_abort,
                )

    for index, event in enumerate(events):
        event_type = event.get("event")
        if event_type not in _CASE_EVENTS:
            continue
        case_id = event.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            add_issue(
                "case_event_missing_case_id",
                f"{event_type} at journal index {index} has no case_id",
                run=run,
                event_index=index,
            )
        elif case_id not in planned_case_ids:
            add_issue(
                "unknown_event_case_id",
                f"{event_type} references unplanned case {case_id}",
                run=run,
                case_id=case_id,
                event_index=index,
            )

    measured_indexes = [
        index
        for index, event in enumerate(events)
        if event.get("event") in {"case_start", "request_complete", "case_complete"}
        and final_start < index < final_terminal
    ]
    ready_indexes = [
        index
        for index, event in enumerate(events)
        if event.get("event") == "server_ready" and final_start < index < final_terminal
    ]
    if measured_indexes and not ready_indexes:
        add_issue(
            "measured_without_server_ready",
            "measured case events occur without server_ready in the final run attempt",
            run=run,
        )
    if ready_indexes:
        last_ready = ready_indexes[-1]
        first_case = min(measured_indexes) if measured_indexes else final_terminal
        prime_indexes = [
            index
            for index, event in enumerate(events)
            if event.get("event") == "first_request_complete"
            and last_ready < index < first_case
        ]
        if measured_indexes and not prime_indexes:
            add_issue(
                "missing_first_request",
                "server_ready is not followed by first_request_complete before cases",
                run=run,
            )
        cleanup_indexes = [
            index
            for index, event in enumerate(events)
            if event.get("event") in {"server_stopped", "server_kept"}
            and last_ready < index < (
                len(events) if aborted_terminal else final_terminal
            )
        ]
        if not cleanup_indexes:
            add_issue(
                "missing_server_cleanup",
                "server_ready is not followed by server_stopped or server_kept",
                run=run,
            )

    started_attempts: dict[tuple[str, str], list[int]] = {}
    requests_by_attempt: dict[tuple[str, str], list[int]] = {}
    completed_by_case: dict[str, list[int]] = {}
    outcomes_by_case: dict[str, list[tuple[int, str]]] = {}
    for index, event in enumerate(events):
        event_type = event.get("event")
        case_id = event.get("case_id")
        attempt_id = event.get("attempt_id")
        if event_type in {"case_start", "request_complete", "case_complete", "case_failed"}:
            if not isinstance(attempt_id, str) or not attempt_id:
                add_issue(
                    "case_event_missing_attempt_id",
                    f"{event_type} for {case_id!r} has no attempt_id",
                    run=run,
                    case_id=case_id,
                    event_index=index,
                )
                continue
            if isinstance(case_id, str):
                key = (case_id, attempt_id)
                if event_type == "case_start":
                    started_attempts.setdefault(key, []).append(index)
                elif event_type == "request_complete":
                    requests_by_attempt.setdefault(key, []).append(index)
                elif event_type == "case_complete":
                    completed_by_case.setdefault(case_id, []).append(index)
        if event_type in _CASE_OUTCOMES and isinstance(case_id, str):
            outcomes_by_case.setdefault(case_id, []).append((index, str(event_type)))

    for (case_id, attempt_id), positions in started_attempts.items():
        if len(positions) > 1:
            add_issue(
                "duplicate_case_start",
                f"case attempt has {len(positions)} case_start events",
                run=run,
                case_id=case_id,
                attempt_id=attempt_id,
            )
    for index, event in enumerate(events):
        if event.get("event") not in {"request_complete", "case_complete", "case_failed"}:
            continue
        case_id = event.get("case_id")
        attempt_id = event.get("attempt_id")
        if not isinstance(case_id, str) or not isinstance(attempt_id, str):
            continue
        starts_for_attempt = started_attempts.get((case_id, attempt_id), [])
        if not any(position < index for position in starts_for_attempt):
            add_issue(
                "attempt_event_without_start",
                f"{event.get('event')} has no preceding matching case_start",
                run=run,
                case_id=case_id,
                attempt_id=attempt_id,
                event_index=index,
            )
    for case_id, positions in completed_by_case.items():
        if len(positions) > 1:
            add_issue(
                "duplicate_case_complete",
                f"case has {len(positions)} case_complete events",
                run=run,
                case_id=case_id,
            )
        event = events[positions[-1]]
        key = (case_id, str(event.get("attempt_id")))
        if not requests_by_attempt.get(key):
            add_issue(
                "completed_case_without_requests",
                "case_complete has no request_complete records for its attempt",
                run=run,
                case_id=case_id,
                attempt_id=key[1],
            )

    for case_id in sorted(planned_case_ids):
        outcomes = outcomes_by_case.get(case_id, [])
        if not outcomes:
            if aborted_terminal and not any(
                started_case_id == case_id
                for started_case_id, _ in started_attempts
            ):
                continue
            add_issue(
                "missing_case_outcome",
                "planned case has no complete, failed, or skipped outcome",
                run=run,
                case_id=case_id,
            )
            continue
        skips = [event_type for _, event_type in outcomes if event_type.startswith("case_skipped_")]
        completes = [event_type for _, event_type in outcomes if event_type == "case_complete"]
        if skips and completes:
            add_issue(
                "conflicting_case_outcomes",
                "case has both skipped and completed outcomes",
                run=run,
                case_id=case_id,
            )


def _derived_status(events: list[dict[str, Any]]) -> tuple[str, dict[str, list[str]]]:
    completed: dict[str, str] = {}
    completed_events: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        if event.get("event") == "case_complete":
            case_id = str(event.get("case_id"))
            attempt_id = str(event.get("attempt_id"))
            completed[case_id] = attempt_id
            completed_events[(case_id, attempt_id)] = event
    completed_ids = set(completed)
    failed = {
        str(event.get("case_id"))
        for event in events
        if event.get("event") == "case_failed"
        and str(event.get("case_id")) not in completed_ids
    }
    unimplemented = {
        str(event.get("case_id"))
        for event in events
        if event.get("event") == "case_skipped_adapter_unimplemented"
        and str(event.get("case_id")) not in completed_ids
    }
    unsupported = {
        str(event.get("case_id"))
        for event in events
        if event.get("event") == "case_skipped_unsupported"
        and str(event.get("case_id")) not in completed_ids
    }
    context_limited = {
        str(event.get("case_id"))
        for event in events
        if event.get("event") == "case_skipped_context_limit"
        and str(event.get("case_id")) not in completed_ids
    }
    validation_failed = {
        case_id
        for case_id, attempt_id in completed.items()
        if completed_events[(case_id, attempt_id)].get("validation_passed") is False
    }
    last_start = max(_event_indexes(events, "run_start"), default=-1)
    last_finish = max(_event_indexes(events, "run_complete"), default=-1)
    last_abort = max(_event_indexes(events, "run_aborted"), default=-1)
    last_cleanup_failure = max(_event_indexes(events, "cleanup_failed"), default=-1)
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
    elif failed or unimplemented or validation_failed:
        status = "partial"
    return status, {
        "failed_cases": sorted(failed),
        "validation_failed_cases": sorted(validation_failed),
        "unimplemented_cases": sorted(unimplemented),
        "unsupported_cases": sorted(unsupported),
        "context_limited_cases": sorted(context_limited),
    }


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _audit_summary(
    summary: dict[str, Any],
    *,
    events: list[dict[str, Any]],
    plan: dict[str, Any],
    entry: dict[str, Any],
    run_dir: Path,
    matrix_root: Path,
    add_issue: IssueAdder,
    run: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_status, expected_lists = _derived_status(events)
    summary_status = summary.get("status")
    run["summary_status"] = summary_status
    run["derived_status"] = expected_status
    if summary_status != expected_status:
        add_issue(
            "summary_status_mismatch",
            f"summary status {summary_status!r} != journal-derived {expected_status!r}",
            run=run,
        )
    index_status = entry.get("status")
    compatible_terminal_status = (
        index_status == summary_status
        or (index_status == "failed" and summary_status == "aborted")
    )
    if not compatible_terminal_status:
        add_issue(
            "index_summary_status_mismatch",
            f"matrix status {index_status!r} != summary status {summary_status!r}",
            run=run,
        )
    if expected_status == "aborted":
        last_start = max(_event_indexes(events, "run_start"), default=-1)
        last_abort = max(
            (
                index
                for index in _event_indexes(events, "run_aborted")
                if index > last_start
            ),
            default=-1,
        )
        expected_error = (
            {
                key: events[last_abort].get(key)
                for key in ("stage", "error_type", "error")
            }
            if last_abort >= 0
            else None
        )
        if summary.get("run_error") != expected_error:
            add_issue(
                "summary_run_error_mismatch",
                "summary run_error does not match the terminal run_aborted event",
                run=run,
                expected=expected_error,
                actual=summary.get("run_error"),
            )
        if index_status == "failed" and expected_error is not None:
            for field in ("error_type", "error"):
                if entry.get(field) != expected_error.get(field):
                    add_issue(
                        "index_run_error_mismatch",
                        f"matrix {field} does not match terminal run_aborted context",
                        run=run,
                        field=field,
                        expected=expected_error.get(field),
                        actual=entry.get(field),
                    )

    summary_run_dir = summary.get("run_dir")
    resolved_summary_run_dir = (
        _resolve_matrix_run_reference(summary_run_dir, matrix_root)
        if isinstance(summary_run_dir, str) and summary_run_dir
        else None
    )
    if resolved_summary_run_dir != run_dir:
        add_issue(
            "summary_run_directory_mismatch",
            "summary run_dir does not identify its indexed run directory",
            run=run,
            summary_run_dir=summary_run_dir,
        )
    summary_model = summary.get("model")
    plan_model = plan.get("model") or {}
    if not isinstance(summary_model, dict) or summary_model.get("id") != plan_model.get("id"):
        add_issue(
            "summary_model_mismatch",
            "summary model identity does not match the frozen plan",
            run=run,
        )
    plan_suite = plan.get("suite") or {}
    if summary.get("suite") != plan_suite.get("id"):
        add_issue(
            "summary_suite_mismatch",
            "summary suite identity does not match the frozen plan",
            run=run,
        )

    for key, expected in expected_lists.items():
        actual = summary.get(key)
        if actual != expected:
            add_issue(
                "summary_outcome_list_mismatch",
                f"summary {key} does not match journal outcomes",
                run=run,
                field=key,
                expected=expected,
                actual=actual,
            )

    completed: dict[str, str] = {}
    case_events: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        if event.get("event") == "case_complete":
            case_id = str(event.get("case_id"))
            attempt_id = str(event.get("attempt_id"))
            completed[case_id] = attempt_id
            case_events[(case_id, attempt_id)] = event

    rows = summary.get("cases")
    if not isinstance(rows, list):
        add_issue("invalid_summary_cases", "summary cases must be a list", run=run)
        rows = []
    rows_by_case: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            add_issue(
                "invalid_summary_case",
                f"summary case {index} is not an object",
                run=run,
                case_index=index,
            )
            continue
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            add_issue(
                "summary_case_missing_id",
                f"summary case {index} has no case_id",
                run=run,
                case_index=index,
            )
            continue
        if case_id in rows_by_case:
            add_issue(
                "duplicate_summary_case",
                f"summary contains duplicate case {case_id}",
                run=run,
                case_id=case_id,
            )
        rows_by_case[case_id] = row

    recomputed: list[dict[str, Any]] = []
    expected_measured_cases: set[str] = set()
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
        expected_measured_cases.add(case_id)
        case_event = case_events[(case_id, attempt_id)]
        results: list[dict[str, Any]] = []
        malformed = False
        for request in requests:
            result = request.get("result")
            if not isinstance(result, dict):
                malformed = True
                break
            results.append(result)
        if malformed:
            add_issue(
                "invalid_request_result",
                "request_complete result must be an object",
                run=run,
                case_id=case_id,
                attempt_id=attempt_id,
            )
            continue
        try:
            prompt_tokens = sum(int(result["prompt_tokens"]) for result in results)
            completion_tokens = sum(int(result["completion_tokens"]) for result in results)
            request_elapsed = sum(float(result["elapsed_s"]) for result in results)
            wall_s = float(case_event.get("elapsed_s") or request_elapsed)
        except (KeyError, TypeError, ValueError) as error:
            add_issue(
                "invalid_request_metrics",
                f"cannot recompute aggregate request metrics: {error}",
                run=run,
                case_id=case_id,
                attempt_id=attempt_id,
            )
            continue
        kind = str(case_event.get("kind") or requests[0].get("kind", "unknown"))
        valid_generation = case_event.get("validation_passed") is not False
        expected_aggregate: float | None = completion_tokens / max(wall_s, 1e-9)
        if (kind in {"decode", "concurrency"} and not valid_generation) or any(
            "dimension" in result or "candidate_count" in result for result in results
        ):
            expected_aggregate = None

        row = rows_by_case.get(case_id)
        matches = True
        if row is None:
            matches = False
            add_issue(
                "missing_summary_case",
                "completed measured case is absent from summary",
                run=run,
                case_id=case_id,
            )
        else:
            exact_fields = {
                "attempt_id": attempt_id,
                "requests": len(requests),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            for field, expected in exact_fields.items():
                if row.get(field) != expected:
                    matches = False
                    add_issue(
                        "summary_case_count_mismatch",
                        f"summary {field} does not match request journal",
                        run=run,
                        case_id=case_id,
                        field=field,
                        expected=expected,
                        actual=row.get(field),
                    )
            actual_wall = row.get("elapsed_s")
            if not _numeric(actual_wall) or float(actual_wall) != wall_s:
                matches = False
                add_issue(
                    "summary_case_wall_mismatch",
                    "summary elapsed_s does not match case wall time",
                    run=run,
                    case_id=case_id,
                    expected=wall_s,
                    actual=actual_wall,
                )
            actual_aggregate = row.get("aggregate_output_tps")
            aggregate_matches = (
                actual_aggregate is None
                if expected_aggregate is None
                else _numeric(actual_aggregate)
                and math.isfinite(float(actual_aggregate))
                and float(actual_aggregate) == expected_aggregate
            )
            if not aggregate_matches:
                matches = False
                add_issue(
                    "aggregate_output_tps_mismatch",
                    "summary aggregate_output_tps is not exact tokens / case wall time",
                    run=run,
                    case_id=case_id,
                    expected=expected_aggregate,
                    actual=actual_aggregate,
                )
        recomputed.append(
            {
                "case_id": case_id,
                "attempt_id": attempt_id,
                "requests": len(requests),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "wall_s": wall_s,
                "aggregate_output_tps": expected_aggregate,
                "summary_matches": matches,
            }
        )

    extra_rows = sorted(set(rows_by_case) - expected_measured_cases)
    if extra_rows:
        add_issue(
            "extra_summary_cases",
            "summary contains cases without a completed measured attempt",
            run=run,
            case_ids=extra_rows,
        )
    expected_count = len(expected_measured_cases)
    if summary.get("completed_cases") != expected_count:
        add_issue(
            "summary_completed_count_mismatch",
            "summary completed_cases does not match measured completed cases",
            run=run,
            expected=expected_count,
            actual=summary.get("completed_cases"),
        )
    if len(rows_by_case) != expected_count:
        add_issue(
            "summary_case_row_count_mismatch",
            "summary case row count does not match measured completed cases",
            run=run,
            expected=expected_count,
            actual=len(rows_by_case),
        )
    if "completed_cases" in entry and (
        entry.get("completed_cases") != summary.get("completed_cases")
    ):
        add_issue(
            "index_completed_count_mismatch",
            "matrix completed_cases does not match summary",
            run=run,
            index_completed_cases=entry.get("completed_cases"),
            summary_completed_cases=summary.get("completed_cases"),
        )
    return recomputed


def _planned_case_id_order(suite: dict[str, Any]) -> tuple[str, ...]:
    """Return frozen case IDs in their declared order, without coercion."""

    cases = suite.get("cases")
    if not isinstance(cases, list):
        return ()
    return tuple(
        case.get("case_id")
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
    )


def audit_sm121_storage_canary_run(run_dir: Path) -> dict[str, Any]:
    """Read-only audit of the dedicated two-lifetime SM121 storage canary.

    This intentionally operates on a direct run directory instead of a matrix.
    It validates the frozen singleton records, plan integrity, and the raw
    journal's no-primer/fresh-server topology without launching any runtime.
    """

    root = run_dir.resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_dir": str(root),
        "read_only": True,
        "ok": False,
        "errors": [],
    }

    def add_issue(code: str, message: str, **context: Any) -> None:
        report["errors"].append({"code": code, "message": message, **context})

    plan = _load_json_object(root / "plan.json", add_issue=add_issue)
    events = _load_jsonl(
        root / "events.jsonl", add_issue=add_issue, run={"run_dir": str(root)}
    )
    if plan is None:
        report["error_count"] = len(report["errors"])
        return report

    model = plan.get("model")
    suite = plan.get("suite")
    if not isinstance(model, dict):
        add_issue("invalid_plan_model", "plan.model must be an object")
    if not isinstance(suite, dict):
        add_issue("invalid_plan_suite", "plan.suite must be an object")
    if not isinstance(model, dict) or not isinstance(suite, dict):
        report["error_count"] = len(report["errors"])
        return report

    if not is_sm121_storage_canary_plan(model, suite):
        add_issue(
            "not_sm121_storage_canary_plan",
            "run plan does not select the dedicated SM121 storage canary",
        )
    else:
        try:
            validate_sm121_storage_candidate(model)
            validate_sm121_storage_suite(suite)
        except SM121StorageCandidateError as error:
            add_issue("invalid_sm121_storage_canary_plan", str(error))

    model_id = model.get("id")
    suite_id = suite.get("id")
    if isinstance(model_id, str) and isinstance(suite_id, str):
        planned_case_ids, _ = _audit_plan(
            plan,
            matrix_suite=suite_id,
            model_id=model_id,
            run_dir=root,
            add_issue=add_issue,
            run={"run_dir": str(root)},
        )
    else:
        planned_case_ids = set()
    planned_case_order = _planned_case_id_order(suite)
    report["planned_case_ids"] = list(planned_case_order)
    if set(planned_case_order) != planned_case_ids:
        add_issue(
            "sm121_storage_plan_case_identity_mismatch",
            "frozen plan case IDs are not a complete ordered set",
        )

    for issue in sm121_storage_canary_lifecycle_issues(
        events, planned_case_ids=planned_case_order
    ):
        code = issue.get("code")
        message = issue.get("message")
        if isinstance(code, str) and isinstance(message, str):
            add_issue(code, message, **{
                key: value
                for key, value in issue.items()
                if key not in {"code", "message"}
            })

    report["error_count"] = len(report["errors"])
    report["ok"] = not report["errors"]
    return report


def audit_sm121_cache_observability_run(run_dir: Path) -> dict[str, Any]:
    """Read-only audit of B0's one-fresh-server cache-off observation lane."""

    root = run_dir.resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_dir": str(root),
        "read_only": True,
        "ok": False,
        "errors": [],
    }

    def add_issue(code: str, message: str, **context: Any) -> None:
        report["errors"].append({"code": code, "message": message, **context})

    plan = _load_json_object(root / "plan.json", add_issue=add_issue)
    events = _load_jsonl(
        root / "events.jsonl", add_issue=add_issue, run={"run_dir": str(root)}
    )
    if plan is None:
        report["error_count"] = len(report["errors"])
        return report
    model = plan.get("model")
    suite = plan.get("suite")
    if not isinstance(model, dict):
        add_issue("invalid_plan_model", "plan.model must be an object")
    if not isinstance(suite, dict):
        add_issue("invalid_plan_suite", "plan.suite must be an object")
    if not isinstance(model, dict) or not isinstance(suite, dict):
        report["error_count"] = len(report["errors"])
        return report
    if not is_sm121_cache_observability_plan(model, suite):
        add_issue(
            "not_sm121_cache_observability_plan",
            "run plan does not select the dedicated SM121 B0 canary",
        )
    else:
        try:
            validate_sm121_cache_observability_candidate(model)
            validate_sm121_cache_observability_suite(suite)
        except SM121CacheObservabilityError as error:
            add_issue("invalid_sm121_cache_observability_plan", str(error))

    model_id = model.get("id")
    suite_id = suite.get("id")
    if isinstance(model_id, str) and isinstance(suite_id, str):
        planned_case_ids, _ = _audit_plan(
            plan,
            matrix_suite=suite_id,
            model_id=model_id,
            run_dir=root,
            add_issue=add_issue,
            run={"run_dir": str(root)},
        )
    else:
        planned_case_ids = set()
    planned_case_order = _planned_case_id_order(suite)
    report["planned_case_ids"] = list(planned_case_order)
    if set(planned_case_order) != planned_case_ids:
        add_issue(
            "b0_plan_case_identity_mismatch",
            "frozen B0 case IDs are not a complete ordered set",
        )
    for lifecycle_issue in sm121_cache_observability_lifecycle_issues(
        events, planned_case_ids=planned_case_order
    ):
        code = lifecycle_issue.get("code")
        message = lifecycle_issue.get("message")
        if isinstance(code, str) and isinstance(message, str):
            add_issue(
                code,
                message,
                **{
                    key: value
                    for key, value in lifecycle_issue.items()
                    if key not in {"code", "message"}
                },
            )
    report["error_count"] = len(report["errors"])
    report["ok"] = not report["errors"]
    return report


def _audit_sm121_cache_semantic_plan_binding(
    plan: dict[str, Any],
    *,
    model: dict[str, Any],
    suite: dict[str, Any],
    arm: str,
    add_issue: IssueAdder,
) -> dict[str, str] | None:
    """Validate one arm's non-sensitive cross-plan binding.

    The plan binding deliberately contains only immutable plan fingerprints and
    the fixed B/A order.  It must never grow a run path, nonce, prompt digest,
    token identity, or request identifier.
    """

    pair = plan.get("semantic_pair")
    try:
        validate_sm121_cache_semantic_pair_binding(pair, model, suite)
    except SM121CacheSemanticError as error:
        add_issue("semantic_pair_binding", str(error))
        return None
    assert isinstance(pair, dict)  # established by the contract validator
    if pair.get("schema_version") != SM121_CACHE_SEMANTIC_PAIR_BINDING_SCHEMA_VERSION:
        add_issue("semantic_pair_binding", "semantic pair binding schema changed")
        return None
    if pair.get("arm") != arm:
        add_issue(
            "semantic_pair_binding_arm",
            "semantic plan binding arm disagrees with its frozen profile",
        )
        return None
    # Public audit output carries only the relationship scalars, not the
    # renderer metadata duplicated in the frozen plan.
    return {
        key: str(pair[key])
        for key in ("peer_plan_fingerprint", "pair_binding_sha256")
    }


def _sm121_cache_semantic_expected_cache_off_receipt(
    cache_on_binding: dict[str, Any], *, cache_on_plan_fingerprint: object
) -> str:
    """Derive A's B-terminal receipt from its scalar plan binding."""

    cache_off_plan_fingerprint = cache_on_binding.get("peer_plan_fingerprint")
    pair_instance_sha256 = cache_on_binding.get("pair_instance_sha256")
    cache_off_binding = dict(cache_on_binding)
    cache_off_binding.update(
        {
            "arm": SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
            "profile_id": SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID,
            "peer_plan_fingerprint": cache_on_plan_fingerprint,
        }
    )
    cache_off_binding["pair_binding_sha256"] = (
        sm121_cache_semantic_pair_binding_sha256(cache_off_binding)
    )
    return sm121_cache_semantic_cache_off_receipt_sha256(
        pair_instance_sha256,
        cache_off_plan_fingerprint,
        cache_off_binding["pair_binding_sha256"],
    )


def audit_sm121_cache_semantic_arm_run(run_dir: Path) -> dict[str, Any]:
    """Read-only audit of one cache-policy semantic canary arm.

    A successful arm is intentionally not a standalone cache-performance
    claim.  It validates the arm's two fresh lifetimes and scalar semantics;
    ``audit_sm121_cache_semantic_pair`` adds the reciprocal B-then-A binding.
    """

    root = run_dir.resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": _sm121_cache_semantic_audit_run_id(root),
        "read_only": True,
        "ok": False,
        "errors": [],
    }

    def add_issue(code: str, message: str, **context: Any) -> None:
        issue = _redact_sm121_cache_semantic_audit_value(
            {"code": code, "message": message, **context}, root=root
        )
        assert isinstance(issue, dict)
        report["errors"].append(issue)

    plan = _load_json_object(root / "plan.json", add_issue=add_issue)
    events = _load_jsonl(
        root / "events.jsonl",
        add_issue=add_issue,
        run={"run_id": _sm121_cache_semantic_audit_run_id(root)},
    )
    if plan is None:
        report["error_count"] = len(report["errors"])
        return report
    model = plan.get("model")
    suite = plan.get("suite")
    if not isinstance(model, dict):
        add_issue("invalid_plan_model", "plan.model must be an object")
    if not isinstance(suite, dict):
        add_issue("invalid_plan_suite", "plan.suite must be an object")
    if not isinstance(model, dict) or not isinstance(suite, dict):
        report["error_count"] = len(report["errors"])
        return report

    arm = ""
    if not is_sm121_cache_semantic_plan(model, suite):
        add_issue(
            "not_sm121_cache_semantic_plan",
            "run plan does not select the dedicated SM121 semantic canary",
        )
    else:
        try:
            validate_sm121_cache_semantic_candidate(model)
            validate_sm121_cache_semantic_suite(suite)
            arm = sm121_cache_semantic_arm(model)
        except SM121CacheSemanticError as error:
            add_issue("invalid_sm121_cache_semantic_plan", str(error))
    if arm:
        report["arm"] = arm
        binding = _audit_sm121_cache_semantic_plan_binding(
            plan, model=model, suite=suite, arm=arm, add_issue=add_issue
        )
        if binding is not None:
            report["pair_binding"] = binding
            run_starts = [event for event in events if event.get("event") == "run_start"]
            if len(run_starts) != 1:
                add_issue(
                    "semantic_run_start_binding",
                    "semantic arm requires exactly one run_start record",
                )
            else:
                peer_fingerprint = binding["peer_plan_fingerprint"]
                expected_start = {
                    "event": "run_start",
                    "execution_mode": SM121_CACHE_SEMANTIC_EXECUTION_MODE,
                    "arm": arm,
                    "plan_fingerprint": plan.get("fingerprint"),
                    "semantic_pair_binding_sha256": binding["pair_binding_sha256"],
                    "cache_off_plan_fingerprint": (
                        None if arm == SM121_CACHE_SEMANTIC_CACHE_OFF_ARM else peer_fingerprint
                    ),
                    "cache_off_audit_passed": (
                        None if arm == SM121_CACHE_SEMANTIC_CACHE_OFF_ARM else True
                    ),
                }
                observed_start = {
                    key: value
                    for key, value in run_starts[0].items()
                    if key != "timestamp"
                }
                if arm == SM121_CACHE_SEMANTIC_CACHE_OFF_ARM:
                    expected_start["cache_off_terminal_receipt_sha256"] = None
                else:
                    try:
                        expected_start["cache_off_terminal_receipt_sha256"] = (
                            _sm121_cache_semantic_expected_cache_off_receipt(
                                plan["semantic_pair"],
                                cache_on_plan_fingerprint=plan.get("fingerprint"),
                            )
                        )
                    except (
                        KeyError,
                        SM121CacheSemanticError,
                        TypeError,
                        ValueError,
                    ):
                        add_issue(
                            "semantic_run_start_binding",
                            "semantic cache-on receipt cannot be derived",
                        )
                if observed_start != expected_start:
                    add_issue(
                        "semantic_run_start_binding",
                        "semantic run_start disagrees with its frozen plan binding",
                    )

    model_id = model.get("id")
    suite_id = suite.get("id")
    if isinstance(model_id, str) and isinstance(suite_id, str):
        planned_case_ids, _ = _audit_plan(
            plan,
            matrix_suite=suite_id,
            model_id=model_id,
            run_dir=root,
            add_issue=add_issue,
            run={"run_id": _sm121_cache_semantic_audit_run_id(root)},
        )
    else:
        planned_case_ids = set()
    planned_case_order = _planned_case_id_order(suite)
    report["planned_case_ids"] = list(planned_case_order)
    if set(planned_case_order) != planned_case_ids:
        add_issue(
            "semantic_plan_case_identity_mismatch",
            "frozen semantic plan case IDs are not a complete ordered set",
        )
    if arm:
        for lifecycle_issue in sm121_cache_semantic_lifecycle_issues(
            events, planned_case_ids=planned_case_order, arm=arm
        ):
            code = lifecycle_issue.get("code")
            message = lifecycle_issue.get("message")
            if isinstance(code, str) and isinstance(message, str):
                add_issue(
                    code,
                    message,
                    **{
                        key: value
                        for key, value in lifecycle_issue.items()
                        if key not in {"code", "message"}
                    },
                )
    report["error_count"] = len(report["errors"])
    report["ok"] = not report["errors"]
    return report


def _sm121_cache_semantic_files_are_truly_unstarted(root: Path) -> bool:
    """Return whether the controller has not created either execution artifact."""

    events_path = root / "events.jsonl"
    summary_path = root / "summary.json"
    return not (
        events_path.exists()
        or events_path.is_symlink()
        or summary_path.exists()
        or summary_path.is_symlink()
    )


def _audit_sm121_cache_semantic_unstarted_arm(run_dir: Path) -> dict[str, Any]:
    """Audit the frozen but intentionally untouched A plan without a journal."""

    root = run_dir.resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": _sm121_cache_semantic_audit_run_id(root),
        "execution_state": "unstarted",
        "read_only": True,
        "ok": False,
        "errors": [],
    }

    def add_issue(code: str, message: str, **context: Any) -> None:
        issue = _redact_sm121_cache_semantic_audit_value(
            {"code": code, "message": message, **context}, root=root
        )
        assert isinstance(issue, dict)
        report["errors"].append(issue)

    if not _sm121_cache_semantic_files_are_truly_unstarted(root):
        add_issue(
            "semantic_unstarted_artifact",
            "unstarted semantic arm must not contain events.jsonl or summary.json",
        )
    plan = _load_json_object(root / "plan.json", add_issue=add_issue)
    if plan is None:
        report["error_count"] = len(report["errors"])
        return report
    model = plan.get("model")
    suite = plan.get("suite")
    if not isinstance(model, dict):
        add_issue("invalid_plan_model", "plan.model must be an object")
    if not isinstance(suite, dict):
        add_issue("invalid_plan_suite", "plan.suite must be an object")
    if not isinstance(model, dict) or not isinstance(suite, dict):
        report["error_count"] = len(report["errors"])
        return report
    arm = ""
    if not is_sm121_cache_semantic_plan(model, suite):
        add_issue(
            "not_sm121_cache_semantic_plan",
            "run plan does not select the dedicated SM121 semantic canary",
        )
    else:
        try:
            validate_sm121_cache_semantic_candidate(model)
            validate_sm121_cache_semantic_suite(suite)
            arm = sm121_cache_semantic_arm(model)
        except SM121CacheSemanticError as error:
            add_issue("invalid_sm121_cache_semantic_plan", str(error))
    if arm:
        report["arm"] = arm
        binding = _audit_sm121_cache_semantic_plan_binding(
            plan, model=model, suite=suite, arm=arm, add_issue=add_issue
        )
        if binding is not None:
            report["pair_binding"] = binding
    model_id = model.get("id")
    suite_id = suite.get("id")
    if isinstance(model_id, str) and isinstance(suite_id, str):
        planned_case_ids, _ = _audit_plan(
            plan,
            matrix_suite=suite_id,
            model_id=model_id,
            run_dir=root,
            add_issue=add_issue,
            run={"run_id": _sm121_cache_semantic_audit_run_id(root)},
        )
    else:
        planned_case_ids = set()
    planned_case_order = _planned_case_id_order(suite)
    report["planned_case_ids"] = list(planned_case_order)
    if set(planned_case_order) != planned_case_ids:
        add_issue(
            "semantic_plan_case_identity_mismatch",
            "frozen semantic plan case IDs are not a complete ordered set",
        )
    if arm != SM121_CACHE_SEMANTIC_CACHE_ON_ARM:
        add_issue(
            "semantic_unstarted_arm",
            "only the cache-on A arm may remain unstarted",
        )
    report["error_count"] = len(report["errors"])
    report["ok"] = not report["errors"]
    return report


def _sm121_cache_semantic_terminal_partial_control(
    report: dict[str, Any], root: Path
) -> bool:
    """Recognize the sole policy-failure topology allowed to suppress A."""

    if report.get("arm") != SM121_CACHE_SEMANTIC_CACHE_OFF_ARM:
        return False
    errors = report.get("errors")
    if not isinstance(errors, list) or not errors:
        return False
    if any(
        not isinstance(issue, dict)
        or issue.get("code") not in _SM121_CACHE_SEMANTIC_ADMISSION_ISSUE_CODES
        for issue in errors
    ):
        return False
    try:
        summary = json.loads((root / "summary.json").read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(summary, dict) and summary.get("status") == "partial"


def audit_sm121_cache_semantic_pair(
    cache_off_run: Path, cache_on_run: Path
) -> dict[str, Any]:
    """Read-only cross-arm audit of the ordered B-then-A semantic pair.

    This checks reciprocal immutable fingerprints on top of the individual
    journal audits.  It intentionally does not read or materialize prompts,
    completions, token sequences, request identifiers, or timing data.
    """

    cache_off_root = cache_off_run.resolve()
    cache_on_root = cache_on_run.resolve()
    cache_on_unstarted = _sm121_cache_semantic_files_are_truly_unstarted(
        cache_on_root
    )
    cache_off = audit_sm121_cache_semantic_arm_run(cache_off_root)
    cache_on = (
        _audit_sm121_cache_semantic_unstarted_arm(cache_on_root)
        if cache_on_unstarted
        else audit_sm121_cache_semantic_arm_run(cache_on_root)
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "read_only": True,
        "ok": False,
        "errors": [],
        "arms": {
            SM121_CACHE_SEMANTIC_CACHE_OFF_ARM: cache_off,
            SM121_CACHE_SEMANTIC_CACHE_ON_ARM: cache_on,
        },
    }

    def add_issue(code: str, message: str, **context: Any) -> None:
        issue: Any = {"code": code, "message": message, **context}
        for root in (cache_off_root, cache_on_root):
            issue = _redact_sm121_cache_semantic_audit_value(issue, root=root)
        assert isinstance(issue, dict)
        report["errors"].append(issue)

    if cache_off.get("arm") != SM121_CACHE_SEMANTIC_CACHE_OFF_ARM:
        add_issue("semantic_pair_off_arm", "first run is not the cache-off B arm")
    if cache_on.get("arm") != SM121_CACHE_SEMANTIC_CACHE_ON_ARM:
        add_issue("semantic_pair_on_arm", "second run is not the cache-on A arm")
    def load_plan(run: Path) -> dict[str, Any] | None:
        try:
            value = json.loads((run.resolve() / "plan.json").read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    off_plan = load_plan(cache_off_root)
    on_plan = load_plan(cache_on_root)
    pair_binding_valid = False
    pair_instance_sha256: str | None = None
    off_fingerprint: str | None = None
    on_fingerprint: str | None = None
    off_binding: dict[str, Any] | None = None
    on_binding: dict[str, Any] | None = None
    if off_plan is None or on_plan is None:
        add_issue("semantic_pair_plan", "paired semantic plan is unavailable")
    else:
        off_fingerprint = off_plan.get("fingerprint")
        on_fingerprint = on_plan.get("fingerprint")
        off_binding = off_plan.get("semantic_pair")
        on_binding = on_plan.get("semantic_pair")
        if not isinstance(off_fingerprint, str) or not isinstance(on_fingerprint, str):
            add_issue("semantic_pair_fingerprint", "paired semantic fingerprint is invalid")
        elif not isinstance(off_binding, dict) or not isinstance(on_binding, dict):
            add_issue("semantic_pair_binding", "paired semantic binding is missing")
        else:
            off_model = off_plan.get("model")
            off_suite = off_plan.get("suite")
            on_model = on_plan.get("model")
            on_suite = on_plan.get("suite")
            try:
                pair_instance_sha256 = sm121_cache_semantic_pair_instance_sha256(
                    off_plan.get("run_nonce"), on_plan.get("run_nonce")
                )
                if (
                    off_binding.get("pair_instance_sha256") != pair_instance_sha256
                    or on_binding.get("pair_instance_sha256") != pair_instance_sha256
                ):
                    raise SM121CacheSemanticError(
                        "semantic pair instance does not match frozen plan nonces"
                    )
                validate_sm121_cache_semantic_pair_binding(
                    off_binding,
                    off_model,
                    off_suite,
                    peer_plan_fingerprint=on_fingerprint,
                    peer_binding=on_binding,
                )
                validate_sm121_cache_semantic_pair_binding(
                    on_binding,
                    on_model,
                    on_suite,
                    peer_plan_fingerprint=off_fingerprint,
                    peer_binding=off_binding,
                )
                pair_binding_valid = True
            except SM121CacheSemanticError as error:
                add_issue(
                    "semantic_pair_binding_mismatch",
                    str(error),
                )

            def run_start(run: Path) -> dict[str, Any] | None:
                try:
                    lines = (run.resolve() / "events.jsonl").read_text().splitlines()
                    first = json.loads(lines[0]) if lines else None
                except (OSError, UnicodeError, json.JSONDecodeError, IndexError):
                    return None
                return first if isinstance(first, dict) else None

            off_start = run_start(cache_off_root)
            if off_start is None:
                add_issue(
                    "semantic_pair_run_start",
                    "cache-off semantic run_start record is unavailable",
                )
            elif (
                off_start.get("execution_mode") != SM121_CACHE_SEMANTIC_EXECUTION_MODE
                or off_start.get("arm") != SM121_CACHE_SEMANTIC_CACHE_OFF_ARM
                or off_start.get("plan_fingerprint") != off_fingerprint
                or off_start.get("semantic_pair_binding_sha256")
                != off_binding.get("pair_binding_sha256")
                or off_start.get("cache_off_plan_fingerprint") is not None
                or off_start.get("cache_off_audit_passed") is not None
                or off_start.get("cache_off_terminal_receipt_sha256") is not None
            ):
                add_issue(
                    "semantic_pair_control_start",
                    "cache-off control run_start does not preserve B-first semantics",
                )

            if not cache_on_unstarted:
                on_start = run_start(cache_on_root)
                if on_start is None:
                    add_issue(
                        "semantic_pair_run_start",
                        "cache-on semantic run_start record is unavailable",
                    )
                else:
                    try:
                        expected_receipt = (
                            sm121_cache_semantic_cache_off_receipt_sha256(
                                pair_instance_sha256,
                                off_fingerprint,
                                off_binding.get("pair_binding_sha256"),
                            )
                        )
                    except SM121CacheSemanticError:
                        expected_receipt = None
                        add_issue(
                            "semantic_pair_candidate_receipt",
                            "cache-on control receipt cannot be derived",
                        )
                    if (
                        on_start.get("execution_mode")
                        != SM121_CACHE_SEMANTIC_EXECUTION_MODE
                        or on_start.get("arm") != SM121_CACHE_SEMANTIC_CACHE_ON_ARM
                        or on_start.get("plan_fingerprint") != on_fingerprint
                        or on_start.get("semantic_pair_binding_sha256")
                        != on_binding.get("pair_binding_sha256")
                        or on_start.get("cache_off_plan_fingerprint") != off_fingerprint
                        or on_start.get("cache_off_audit_passed") is not True
                        or on_start.get("cache_off_terminal_receipt_sha256")
                        != expected_receipt
                    ):
                        add_issue(
                            "semantic_pair_candidate_start",
                            "cache-on candidate run_start does not attest the completed B control",
                        )

    if cache_on_unstarted:
        report["topology"] = "cache_off_terminal_partial_cache_on_unstarted"
        if (
            pair_binding_valid
            and cache_on.get("ok") is True
            and _sm121_cache_semantic_terminal_partial_control(
                cache_off, cache_off_root
            )
        ):
            report["authorized_terminal_partial"] = True
        elif cache_off.get("ok") is True:
            add_issue(
                "semantic_pair_unstarted_candidate",
                "completed cache-off B control requires a started cache-on A candidate",
            )
        else:
            add_issue(
                "semantic_pair_partial_control",
                "unstarted cache-on A requires a terminal policy-partial B control",
            )
    else:
        report["topology"] = "cache_off_complete_cache_on_completed"
        if not cache_off.get("ok") or not cache_on.get("ok"):
            add_issue("semantic_pair_arm_audit", "one or both semantic arm audits failed")

    report["error_count"] = len(report["errors"])
    report["ok"] = not report["errors"]
    return report


def audit_sm121_cache_performance_campaign(
    campaign_dir: Path, *, evidence_root: Path
) -> dict[str, Any]:
    """Read-only verification of one frozen SM121 A/B/B/A timing campaign.

    The source validator is shared with the scalar exporter so audit and
    publication cannot disagree about fresh-lifetime topology, score reduction,
    or which fields are safe to retain.  It never opens server logs, prompts,
    completions, token IDs, request identifiers, or credentials.
    """

    report: dict[str, Any] = {
        "schema_version": 1,
        "read_only": True,
        "campaign_id": SM121_CACHE_PERFORMANCE_CAMPAIGN_ID,
        "ok": False,
        "errors": [],
    }

    def add_issue(code: str, message: str) -> None:
        report["errors"].append({"code": code, "message": message})

    try:
        root = campaign_dir.resolve(strict=True)
        results_root = root.parent.parent
        if root.parent.name != "cache-policy-campaigns" or not results_root.is_dir():
            raise ValueError
        from .evidence import (
            EvidenceError,
            _validate_sm121_cache_performance_source,
            verify_sm121_cache_performance_prerequisites,
        )

        verify_sm121_cache_performance_prerequisites(evidence_root)
        source = _validate_sm121_cache_performance_source(root, results_root)
        if source is None:
            add_issue(
                "campaign_not_terminal",
                "frozen cache-performance campaign has not reached a terminal summary",
            )
        else:
            summary = source["summary"]
            report.update(
                {
                    "status": summary["status"],
                    "decision": summary["decision"],
                    "completed_arms": summary["completed_arms"],
                    "score": summary["score"],
                    "static_attestation_count": len(source["static_events"]),
                    "runtime_attestation_count": len(source["runtime_events"]),
                }
            )
    except (OSError, ValueError, KeyError, TypeError, EvidenceError):
        add_issue(
            "campaign_contract_invalid",
            "cache-performance campaign does not meet the frozen scalar contract",
        )
    report["error_count"] = len(report["errors"])
    report["ok"] = report["error_count"] == 0
    return report


def audit_sm121_chunked_prefill_performance_campaign(
    campaign_dir: Path,
) -> dict[str, Any]:
    """Read-only validation of one SM121 1K/2K A/B/B/A campaign.

    It shares the scalar source validator with publication and never opens
    prompt content, responses, token IDs, request identifiers, logs, or keys.
    """

    report: dict[str, Any] = {
        "schema_version": 1,
        "read_only": True,
        "campaign_id": "qwen38-flash-next-sm121-chunked-prefill-performance-v1",
        "ok": False,
        "errors": [],
    }

    def add_issue(code: str, message: str) -> None:
        report["errors"].append({"code": code, "message": message})

    try:
        root = campaign_dir.resolve(strict=True)
        results_root = root.parent.parent
        if (
            root.parent.name != chunked_prefill_evidence.RESULT_ROOT
            or not results_root.is_dir()
        ):
            raise ValueError
        source = chunked_prefill_evidence.validate_source(root, results_root)
        if source is None:
            add_issue(
                "campaign_not_terminal",
                "frozen chunked-prefill campaign has not reached a terminal summary",
            )
        else:
            summary = source["summary"]
            report.update(
                {
                    "status": summary["status"],
                    "decision": summary["decision"],
                    "completed_arms": summary["completed_arms"],
                    "score": summary["score"],
                    "static_attestation_count": len(source["static_events"]),
                    "runtime_attestation_count": len(source["runtime_events"]),
                }
            )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        chunked_prefill_evidence.ChunkedPrefillEvidenceError,
    ):
        add_issue(
            "campaign_contract_invalid",
            "chunked-prefill campaign does not meet the frozen scalar contract",
        )
    report["error_count"] = len(report["errors"])
    report["ok"] = report["error_count"] == 0
    return report


def audit_matrix(matrix_dir: Path) -> dict[str, Any]:
    """Audit a matrix without writing files or consulting external runtimes."""

    root = matrix_dir.resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "matrix_dir": str(root),
        "read_only": True,
        "ok": False,
        "errors": [],
        "matrix": {},
        "runs": [],
    }

    def add_issue(
        code: str,
        message: str,
        *,
        run: dict[str, Any] | None = None,
        **context: Any,
    ) -> None:
        run_context = (
            {"model": run.get("model"), "run_dir": run.get("run_dir")}
            if run is not None
            else {}
        )
        issue = {"code": code, "message": message, **run_context, **context}
        report["errors"].append(issue)
        if run is not None:
            run["error_codes"].append(code)

    index = _load_json_object(root / "matrix.json", add_issue=add_issue)
    if index is None:
        report["error_count"] = len(report["errors"])
        return report

    suite = index.get("suite") if isinstance(index.get("suite"), str) else None
    if suite is None:
        add_issue("invalid_matrix_suite", "matrix suite must be a non-empty string")
    raw_models = index.get("models")
    raw_runs = index.get("runs")
    if not isinstance(raw_models, list):
        add_issue("invalid_matrix_models", "matrix models must be a list")
        raw_models = []
    if not isinstance(raw_runs, list):
        add_issue("invalid_matrix_runs", "matrix runs must be a list")
        raw_runs = []
    models = [model for model in raw_models if isinstance(model, str) and model]
    if len(models) != len(raw_models):
        add_issue("invalid_matrix_model_id", "matrix contains a non-string model ID")
    if not models:
        add_issue("empty_matrix", "matrix has no models")
    if len(models) != len(set(models)):
        add_issue("duplicate_matrix_model", "matrix contains duplicate model IDs")

    run_models = [
        entry.get("model")
        for entry in raw_runs
        if isinstance(entry, dict) and isinstance(entry.get("model"), str)
    ]
    if len(run_models) != len(raw_runs):
        add_issue("invalid_matrix_run", "matrix contains an invalid run entry")
    if run_models != models:
        add_issue(
            "matrix_run_completeness_mismatch",
            "matrix run model IDs do not exactly match the frozen model list",
            expected=models,
            actual=run_models,
        )
    if len(run_models) != len(set(run_models)):
        add_issue("duplicate_matrix_run", "matrix contains duplicate run model IDs")

    indexed_dirs: set[Path] = set()
    for index_number, entry_value in enumerate(raw_runs):
        if not isinstance(entry_value, dict):
            continue
        entry = entry_value
        model_id = entry.get("model")
        if not isinstance(model_id, str) or not model_id:
            continue
        run_report: dict[str, Any] = {
            "model": model_id,
            "run_dir": entry.get("run_dir"),
            "index_status": entry.get("status"),
            "error_codes": [],
            "recomputed_cases": [],
        }
        report["runs"].append(run_report)
        if entry.get("status") in {"planned", "running"}:
            add_issue(
                "matrix_run_not_finished",
                f"matrix run is still {entry.get('status')}",
                run=run_report,
            )
        raw_run_dir = entry.get("run_dir")
        if not isinstance(raw_run_dir, str) or not raw_run_dir:
            add_issue(
                "missing_matrix_run_directory",
                "matrix run entry has no run_dir",
                run=run_report,
                run_index=index_number,
            )
            run_report["ok"] = False
            continue
        run_dir = _resolve_matrix_run_reference(raw_run_dir, root)
        run_report["run_dir"] = str(run_dir)
        try:
            run_dir.relative_to(root)
        except ValueError:
            add_issue(
                "run_directory_outside_matrix",
                "indexed run directory resolves outside the matrix directory",
                run=run_report,
            )
            run_report["ok"] = False
            continue
        indexed_dirs.add(run_dir)
        if not run_dir.is_dir():
            add_issue(
                "missing_run_directory",
                f"indexed run directory is missing: {run_dir}",
                run=run_report,
            )
            run_report["ok"] = False
            continue

        plan = _load_json_object(run_dir / "plan.json", add_issue=add_issue, run=run_report)
        events = _load_jsonl(run_dir / "events.jsonl", add_issue=add_issue, run=run_report)
        summary = _load_json_object(
            run_dir / "summary.json", add_issue=add_issue, run=run_report
        )
        if plan is None:
            run_report["ok"] = False
            continue
        planned_case_ids, max_num_seqs = _audit_plan(
            plan,
            matrix_suite=suite,
            model_id=model_id,
            run_dir=run_dir,
            add_issue=add_issue,
            run=run_report,
        )
        run_report["fingerprint"] = plan.get("fingerprint")
        run_report["planned_cases"] = len(planned_case_ids)
        run_report["max_num_seqs"] = max_num_seqs
        run_report["scheduling_label"] = (
            "serialized_queue"
            if max_num_seqs == 1
            else "parallel_configured"
            if max_num_seqs and max_num_seqs > 1
            else "unspecified"
        )
        _audit_lifecycle(
            events,
            planned_case_ids=planned_case_ids,
            add_issue=add_issue,
            run=run_report,
        )
        if summary is not None:
            run_report["recomputed_cases"] = _audit_summary(
                summary,
                events=events,
                plan=plan,
                entry=entry,
                run_dir=run_dir,
                matrix_root=root,
                add_issue=add_issue,
                run=run_report,
            )
        run_report["ok"] = not run_report["error_codes"]

    try:
        disk_dirs = {
            child.resolve()
            for child in root.iterdir()
            if child.is_dir() and (child / "plan.json").is_file()
        }
    except OSError as error:
        add_issue("unreadable_matrix_directory", f"cannot scan matrix directory: {error}")
        disk_dirs = set()
    unindexed = sorted(str(path) for path in disk_dirs - indexed_dirs)
    if unindexed:
        add_issue(
            "unindexed_run_directory",
            "matrix directory contains run plans absent from matrix.json",
            run_dirs=unindexed,
        )

    report["matrix"] = {
        "suite": suite,
        "expected_models": len(models),
        "indexed_runs": len(raw_runs),
        "audited_runs": len(report["runs"]),
        "serialized_queue_runs": [
            run["model"]
            for run in report["runs"]
            if run.get("scheduling_label") == "serialized_queue"
        ],
    }
    report["error_count"] = len(report["errors"])
    report["ok"] = not report["errors"]
    return report
