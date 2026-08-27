"""Append-only measurement validity annotations for completed benchmark runs."""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .journal import Journal


ANNOTATION_EVENT = "measurement_annotation"
ANNOTATION_SCOPES = frozenset({"startup", "case"})
STARTUP_SAFETY_GATE_SCHEMA_VERSION = 2
STARTUP_SAFETY_GATE_REGISTRY = {
    "host_memavailable": ("gib", "lt"),
    "startup_swap_growth": ("mib", "gt"),
}
_STARTUP_SAFETY_GATE_MAX_VALUES = {
    "host_memavailable": 1024.0 * 1024.0,
    "startup_swap_growth": 1024.0 * 1024.0,
}
_STARTUP_SAFETY_GATE_FIELDS = frozenset(
    {"comparison", "limit", "metric", "observed", "unit"}
)
_STARTUP_SAFETY_GATE_EVENT_FIELDS = frozenset(
    {
        "event",
        "measurement_valid",
        "safety_gate",
        "schema_version",
        "scope",
        "timestamp",
    }
)
_STARTUP_SAFETY_GATE_ANNOTATION_FIELDS = frozenset(
    {"measurement_valid", "safety_gate", "scope", "timestamp"}
)


def normalize_startup_safety_gate(value: Any) -> dict[str, Any]:
    """Validate and normalize one closed-registry startup safety gate."""

    if not isinstance(value, dict) or set(value) != _STARTUP_SAFETY_GATE_FIELDS:
        raise ValueError(
            "startup safety gate must contain exactly comparison, limit, metric, "
            "observed, and unit"
        )
    metric = value.get("metric")
    if not isinstance(metric, str) or metric not in STARTUP_SAFETY_GATE_REGISTRY:
        raise ValueError(f"unknown startup safety-gate metric: {metric!r}")
    expected_unit, expected_comparison = STARTUP_SAFETY_GATE_REGISTRY[metric]
    if (
        value.get("unit") != expected_unit
        or value.get("comparison") != expected_comparison
    ):
        raise ValueError(
            f"startup safety gate {metric!r} requires unit={expected_unit!r} "
            f"and comparison={expected_comparison!r}"
        )

    normalized_numbers: dict[str, float] = {}
    for field in ("observed", "limit"):
        number = value.get(field)
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError(f"startup safety-gate {field} must be numeric")
        try:
            normalized = float(number)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                f"startup safety-gate {field} must be a bounded number"
            ) from error
        if not math.isfinite(normalized):
            raise ValueError(f"startup safety-gate {field} must be finite")
        normalized_numbers[field] = normalized
    observed = normalized_numbers["observed"]
    limit = normalized_numbers["limit"]
    if observed < 0:
        raise ValueError("startup safety-gate observed value must be nonnegative")
    if limit <= 0:
        raise ValueError("startup safety-gate limit must be positive")
    maximum = _STARTUP_SAFETY_GATE_MAX_VALUES[metric]
    if observed > maximum or limit > maximum:
        raise ValueError("startup safety-gate value exceeds the supported range")
    breached = (
        observed < limit
        if expected_comparison == "lt"
        else observed > limit
    )
    if not breached:
        raise ValueError("startup safety gate must describe a true breach")
    return {
        "metric": metric,
        "observed": observed,
        "limit": limit,
        "unit": expected_unit,
        "comparison": expected_comparison,
    }


def _startup_safety_gate_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("startup safety-gate journal timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("startup safety-gate journal timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("startup safety-gate journal timestamp lacks a timezone")
    return value


def _startup_safety_gate_annotation(event: dict[str, Any]) -> dict[str, Any]:
    if set(event) != _STARTUP_SAFETY_GATE_EVENT_FIELDS:
        raise ValueError("startup safety-gate journal event schema changed")
    if (
        event.get("event") != ANNOTATION_EVENT
        or type(event.get("schema_version")) is not int
        or event.get("schema_version") != STARTUP_SAFETY_GATE_SCHEMA_VERSION
        or event.get("scope") != "startup"
        or event.get("measurement_valid") is not False
    ):
        raise ValueError("startup safety-gate journal classification changed")
    timestamp = _startup_safety_gate_timestamp(event.get("timestamp"))
    return {
        "timestamp": timestamp,
        "scope": "startup",
        "measurement_valid": False,
        "safety_gate": normalize_startup_safety_gate(event.get("safety_gate")),
    }


def startup_safety_gate_annotations_from_annotations(
    annotations: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return unique normalized typed annotations in their journal order."""

    normalized_annotations: list[dict[str, Any]] = []
    seen_metrics: set[str] = set()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError("measurement annotation must be an object")
        if "safety_gate" not in annotation:
            continue
        if set(annotation) != _STARTUP_SAFETY_GATE_ANNOTATION_FIELDS:
            raise ValueError("normalized startup safety-gate annotation schema changed")
        if (
            annotation.get("scope") != "startup"
            or annotation.get("measurement_valid") is not False
        ):
            raise ValueError("normalized startup safety-gate classification changed")
        timestamp = _startup_safety_gate_timestamp(annotation.get("timestamp"))
        gate = normalize_startup_safety_gate(annotation.get("safety_gate"))
        metric = gate["metric"]
        if metric in seen_metrics:
            raise ValueError(f"duplicate startup safety-gate metric: {metric}")
        seen_metrics.add(metric)
        normalized_annotations.append(
            {
                "timestamp": timestamp,
                "scope": "startup",
                "measurement_valid": False,
                "safety_gate": gate,
            }
        )
    return normalized_annotations


def startup_safety_gates_from_annotations(
    annotations: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return unique typed gates in canonical metric order."""

    gates = [
        annotation["safety_gate"]
        for annotation in startup_safety_gate_annotations_from_annotations(
            annotations
        )
    ]
    return sorted(gates, key=lambda gate: gate["metric"])


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


def append_startup_safety_gate(
    run_dir: Path,
    *,
    metric: str,
    observed: float,
    limit: float,
    unit: str,
    comparison: str,
) -> dict[str, Any]:
    """Append one typed, measurement-invalidating startup gate breach."""

    run_dir = run_dir.resolve()
    _frozen_case_ids(run_dir)
    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        raise ValueError(f"benchmark event journal is missing: {events_path}")
    safety_gate = normalize_startup_safety_gate(
        {
            "metric": metric,
            "observed": observed,
            "limit": limit,
            "unit": unit,
            "comparison": comparison,
        }
    )
    journal = Journal(events_path)
    existing = startup_safety_gates_from_annotations(
        measurement_annotations(journal.events())
    )
    if any(gate["metric"] == safety_gate["metric"] for gate in existing):
        raise ValueError(
            f"duplicate startup safety-gate metric: {safety_gate['metric']}"
        )
    event: dict[str, Any] = {
        "event": ANNOTATION_EVENT,
        "schema_version": STARTUP_SAFETY_GATE_SCHEMA_VERSION,
        "scope": "startup",
        "measurement_valid": False,
        "safety_gate": safety_gate,
    }
    journal.append(event)
    return journal.events()[-1]


def measurement_annotations(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return normalized measurement annotations in journal order."""

    annotations: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != ANNOTATION_EVENT:
            continue
        schema_version = event.get("schema_version", 1)
        if type(schema_version) is not int:
            raise ValueError(
                f"unsupported measurement annotation schema version: {schema_version!r}"
            )
        if schema_version == STARTUP_SAFETY_GATE_SCHEMA_VERSION:
            annotations.append(_startup_safety_gate_annotation(event))
            continue
        if schema_version != 1:
            raise ValueError(
                f"unsupported measurement annotation schema version: {schema_version!r}"
            )
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
    startup_safety_gates_from_annotations(annotations)
    return annotations
