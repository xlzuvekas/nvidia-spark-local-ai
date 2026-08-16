"""Append-only measurement validity annotations for completed benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .journal import Journal


ANNOTATION_EVENT = "measurement_annotation"
ANNOTATION_SCOPES = frozenset({"startup", "case"})


def _frozen_case_ids(run_dir: Path) -> set[str]:
    plan_path = run_dir / "plan.json"
    try:
        plan = json.loads(plan_path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"frozen plan is missing: {plan_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"frozen plan is not valid JSON: {plan_path}") from error
    if not isinstance(plan, dict):
        raise ValueError(f"frozen plan must be a JSON object: {plan_path}")
    suite = plan.get("suite")
    cases = suite.get("cases") if isinstance(suite, dict) else None
    if not isinstance(cases, list):
        raise ValueError("frozen plan has no suite case list")
    case_ids = {
        case["case_id"]
        for case in cases
        if isinstance(case, dict)
        and isinstance(case.get("case_id"), str)
        and case["case_id"]
    }
    if len(case_ids) != len(cases):
        raise ValueError("every frozen plan case must have a unique non-empty case_id")
    return case_ids


def append_measurement_annotation(
    run_dir: Path,
    *,
    scope: str,
    reason: str,
    case_id: str | None = None,
    evidence: Iterable[str] = (),
) -> dict[str, Any]:
    """Validate and append one measurement-invalidating journal record."""

    run_dir = run_dir.resolve()
    if scope not in ANNOTATION_SCOPES:
        raise ValueError(
            f"annotation scope must be one of {sorted(ANNOTATION_SCOPES)}"
        )
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise ValueError("annotation reason must be non-empty")
    normalized_evidence: list[str] = []
    for item in evidence:
        value = item.strip()
        if not value:
            raise ValueError("annotation evidence values must be non-empty")
        normalized_evidence.append(value)

    planned_case_ids = _frozen_case_ids(run_dir)
    if scope == "case":
        if not case_id:
            raise ValueError("case annotations require --case-id")
        if case_id not in planned_case_ids:
            raise ValueError(f"case_id {case_id!r} is not present in the frozen plan")
    elif case_id is not None:
        raise ValueError("startup annotations must not include --case-id")

    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        raise ValueError(f"benchmark event journal is missing: {events_path}")
    event: dict[str, Any] = {
        "event": ANNOTATION_EVENT,
        "schema_version": 1,
        "scope": scope,
        "reason": normalized_reason,
        "evidence": normalized_evidence,
        "measurement_valid": False,
    }
    if case_id is not None:
        event["case_id"] = case_id

    journal = Journal(events_path)
    journal.append(event)
    return journal.events()[-1]


def measurement_annotations(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return normalized measurement annotations in journal order."""

    annotations: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != ANNOTATION_EVENT:
            continue
        annotation: dict[str, Any] = {
            "timestamp": event.get("timestamp"),
            "scope": event.get("scope"),
            "reason": event.get("reason"),
            "evidence": (
                list(event["evidence"])
                if isinstance(event.get("evidence"), list)
                else []
            ),
            "measurement_valid": False,
        }
        if event.get("scope") == "case":
            annotation["case_id"] = event.get("case_id")
        annotations.append(annotation)
    return annotations
