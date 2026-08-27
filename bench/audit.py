"""Dependency-free, read-only validation of completed SparkBench matrices."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable


IssueAdder = Callable[..., None]

_CASE_OUTCOMES = {
    "case_complete",
    "case_failed",
    "case_skipped_adapter_unimplemented",
    "case_skipped_context_limit",
    "case_skipped_unsupported",
}
_CASE_EVENTS = _CASE_OUTCOMES | {"case_start", "request_complete"}


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
