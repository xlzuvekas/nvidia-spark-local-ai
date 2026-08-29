"""Scalar-only source and bundle validation for SM121 chunked-prefill runs.

The implementation is deliberately independent from the cache-policy evidence
lane. It accepts only dedicated chunk-size A/B/B/A roots, projects no raw
journal payloads, and exposes a small API used by the exporter and audit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .journal import content_hash
from .sglang_sm121_chunked_prefill_performance import (
    ChunkedPrefillPerformanceStudy,
    SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S,
    SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT,
    SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EVENT,
    SM121_CHUNKED_PREFILL_PERFORMANCE_STATIC_EVENT,
    SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS,
    SM121_CHUNKED_PREFILL_PERFORMANCE_TURN_EVENT,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY,
    SM121ChunkedPrefillPerformanceError,
    score_sm121_chunked_prefill_performance_campaign,
    sm121_chunked_prefill_performance_arm,
    sm121_chunked_prefill_performance_study,
    sm121_chunked_prefill_performance_pair_binding_sha256,
    sm121_chunked_prefill_performance_pair_instance_sha256,
    validate_sm121_chunked_prefill_performance_candidate,
    validate_sm121_chunked_prefill_performance_lifetimes,
    validate_sm121_chunked_prefill_performance_pair_binding,
    validate_sm121_chunked_prefill_performance_recorded_turn_event,
    validate_sm121_chunked_prefill_performance_runtime_event,
    validate_sm121_chunked_prefill_performance_static_event,
    validate_sm121_chunked_prefill_performance_suite,
    validate_sm121_chunked_prefill_performance_turn_event,
)
from .sglang_sm121_storage import (
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_SOURCE_TREE,
)


RESULT_ROOT = "chunked-prefill-campaigns"
EVIDENCE_KIND = "sm121_chunked_prefill_performance"

_CAMPAIGN_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "created_at",
        "execution_mode",
        "pair_binding",
        "run_directories",
        "integrity_hash",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "execution_mode",
        "pair_binding_sha256",
        "status",
        "decision",
        "completed_arms",
        "lifetimes",
        "score",
        "integrity_hash",
    }
)
_SAFE_EVENT_NAMES = frozenset(
    {
        "run_start",
        "measurement_started",
        "measurement_complete",
        "run_complete",
        "run_aborted",
        "host_safety_breach",
        "host_safety_interrupt_failed",
        "server_ready",
        "server_stopped",
        "sm121_chunked_prefill_performance_lifetime_complete",
        "sm121_chunked_prefill_performance_quality_case_start",
        "sm121_chunked_prefill_performance_quality_case_complete",
        "sm121_chunked_prefill_performance_timed_case_start",
        "sm121_chunked_prefill_performance_timed_case_complete",
        SM121_CHUNKED_PREFILL_PERFORMANCE_STATIC_EVENT,
        SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EVENT,
        SM121_CHUNKED_PREFILL_PERFORMANCE_TURN_EVENT,
    }
)


class ChunkedPrefillEvidenceError(ValueError):
    """Raised when raw provenance or scalar publication drifts from contract."""


def _study_from_campaign_id(value: object) -> ChunkedPrefillPerformanceStudy:
    try:
        study = sm121_chunked_prefill_performance_study(value)
    except SM121ChunkedPrefillPerformanceError as error:
        raise ChunkedPrefillEvidenceError("chunked-prefill campaign study is invalid") from error
    if value != study.campaign_id:
        raise ChunkedPrefillEvidenceError("chunked-prefill campaign study is invalid")
    return study


def _require_publication_admission(study: ChunkedPrefillPerformanceStudy) -> None:
    """Keep the prospective 8K study out of every public evidence route."""

    if study == SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY:
        raise ChunkedPrefillEvidenceError(
            "chunked-prefill v3 requires a verified 8K admission receipt"
        )


def _expected_chunk_size(study: ChunkedPrefillPerformanceStudy, arm: object) -> int:
    if arm == "A":
        return study.control_chunk_size
    if arm == "B":
        return study.candidate_chunk_size
    raise ChunkedPrefillEvidenceError("chunked-prefill attestation arm is invalid")


def _validate_study_attestation(
    event: Mapping[str, object], *, study: ChunkedPrefillPerformanceStudy
) -> None:
    if event.get("chunked_prefill_size") != _expected_chunk_size(
        study, event.get("arm")
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill attestation study changed")


def _safe_resolve(path: Path, root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ChunkedPrefillEvidenceError("chunked-prefill source path is unsafe") from error
    if path.is_symlink():
        raise ChunkedPrefillEvidenceError("chunked-prefill source path is unsafe")
    return resolved


def _load_json(path: Path, root: Path) -> object:
    resolved = _safe_resolve(path, root)
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ChunkedPrefillEvidenceError("chunked-prefill JSON is invalid") from error


def _load_json_lines(path: Path, root: Path) -> list[dict[str, Any]]:
    resolved = _safe_resolve(path, root)
    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
        parsed = [json.loads(line) for line in lines if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ChunkedPrefillEvidenceError("chunked-prefill journal is invalid") from error
    if any(type(event) is not dict for event in parsed):
        raise ChunkedPrefillEvidenceError("chunked-prefill journal is invalid")
    return parsed


def campaign_dirs(results_root: Path) -> list[Path]:
    root = results_root / RESULT_ROOT
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise ChunkedPrefillEvidenceError("chunked-prefill result root is invalid")
    campaigns: list[Path] = []
    for child in sorted(root.iterdir()):
        if (
            child.is_symlink()
            or not child.is_dir()
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", child.name) is None
            or not (child / "campaign.json").is_file()
        ):
            raise ChunkedPrefillEvidenceError(
                "chunked-prefill campaign directory is invalid"
            )
        campaigns.append(child)
    return campaigns


def _without_timestamp(event: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "timestamp"}


def _validate_plan(
    *,
    plan: object,
    binding: Mapping[str, object],
    study: ChunkedPrefillPerformanceStudy,
    ordinal: int,
    expected_arm: str,
) -> tuple[str, str]:
    if type(plan) is not dict:
        raise ChunkedPrefillEvidenceError("chunked-prefill plan is invalid")
    model, suite, resolved = plan.get("model"), plan.get("suite"), plan.get("resolved")
    if (
        type(model) is not dict
        or type(suite) is not dict
        or type(resolved) is not dict
        or type(suite.get("cases")) is not list
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill plan core fields are invalid")
    try:
        validate_sm121_chunked_prefill_performance_candidate(model)
        validate_sm121_chunked_prefill_performance_suite(suite)
        if (
            sm121_chunked_prefill_performance_study(model) != study
            or sm121_chunked_prefill_performance_study(suite.get("id")) != study
            or sm121_chunked_prefill_performance_arm(model) != expected_arm
        ):
            raise SM121ChunkedPrefillPerformanceError("chunked-prefill arm changed")
    except SM121ChunkedPrefillPerformanceError as error:
        raise ChunkedPrefillEvidenceError("chunked-prefill plan contract is invalid") from error
    local_image = resolved.get("local_image")
    if (
        type(local_image) is not dict
        or local_image
        != {
            "docker_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
            "platform": SM121_STORAGE_PLATFORM,
            "source_tree": SM121_STORAGE_SOURCE_TREE,
        }
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill plan image changed")
    cases = suite["cases"]
    if any(type(case) is not dict for case in cases):
        raise ChunkedPrefillEvidenceError("chunked-prefill plan cases are invalid")
    case_ids: dict[str, str] = {}
    for case in cases:
        case_name, case_id = case.get("id"), case.get("case_id")
        if (
            not isinstance(case_name, str)
            or not isinstance(case_id, str)
            or re.fullmatch(rf"{re.escape(case_name)}--[0-9a-f]{{12}}", case_id)
            is None
        ):
            raise ChunkedPrefillEvidenceError("chunked-prefill case identity is invalid")
        case_ids[case_name] = case_id
    quality_case_id = case_ids.get(SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID)
    timed_case_id = case_ids.get(study.timed_case_id)
    if not isinstance(quality_case_id, str) or not isinstance(timed_case_id, str):
        raise ChunkedPrefillEvidenceError("chunked-prefill case topology is invalid")
    suite_without_case_ids = {
        **suite,
        "cases": [{key: value for key, value in case.items() if key != "case_id"} for case in cases],
    }
    if plan.get("fingerprint") != content_hash(
        {"model": model, "suite": suite_without_case_ids, "resolved": resolved}
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill plan fingerprint is invalid")
    integrity = plan.get("integrity_hash")
    if not isinstance(integrity, str) or content_hash(
        {key: value for key, value in plan.items() if key != "integrity_hash"},
        len(integrity),
    ) != integrity:
        raise ChunkedPrefillEvidenceError("chunked-prefill plan integrity is invalid")
    if (
        plan.get("chunked_prefill_performance_ordinal") != ordinal
        or plan.get("chunked_prefill_performance_pair") != binding
        or not isinstance(plan.get("run_nonce"), str)
        or re.fullmatch(r"[0-9a-f]{32}", plan["run_nonce"]) is None
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill plan binding is invalid")
    return quality_case_id, timed_case_id


def _quality_attestation(
    *, events: Sequence[dict[str, Any]], arm: str, ordinal: int, case_id: str
) -> dict[str, object] | None:
    quality_ordinal = ordinal * 2 - 1
    completed = [
        event
        for event in events
        if event.get("event")
        == "sm121_chunked_prefill_performance_quality_case_complete"
    ]
    if not completed:
        return None
    if len(completed) != 1:
        raise ChunkedPrefillEvidenceError("chunked-prefill quality lifecycle changed")
    event = completed[0]
    if (
        event.get("arm"),
        event.get("lifetime_ordinal"),
        event.get("case_id"),
        event.get("quality_admitted"),
        event.get("item_count"),
    ) != (
        arm,
        quality_ordinal,
        case_id,
        True,
        SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT,
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill quality lifecycle changed")
    return {
        "arm": arm,
        "quality_lifetime_ordinal": quality_ordinal,
        "case_id": case_id,
        "quality_admitted": True,
        "item_count": SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT,
    }


def _validate_run_start(
    *,
    events: Sequence[dict[str, Any]],
    study: ChunkedPrefillPerformanceStudy,
    binding: Mapping[str, object],
    arm: str,
    ordinal: int,
    plan: Mapping[str, object],
) -> None:
    if not events:
        return
    starts = [event for event in events if event.get("event") == "run_start"]
    if len(starts) != 1 or (
        starts[0].get("execution_mode"),
        starts[0].get("arm"),
        starts[0].get("campaign_ordinal"),
        starts[0].get("plan_fingerprint"),
        starts[0].get("chunked_prefill_performance_pair_binding_sha256"),
    ) != (
        study.execution_mode,
        arm,
        ordinal,
        plan.get("fingerprint"),
        binding.get("pair_binding_sha256"),
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill run binding changed")


def _validate_completed_lifecycle(
    *,
    events: Sequence[dict[str, Any]],
    arm: str,
    ordinal: int,
    quality_case_id: str,
    timed_case_id: str,
) -> None:
    quality_ordinal, timed_ordinal = ordinal * 2 - 1, ordinal * 2

    def matching(name: str) -> list[dict[str, Any]]:
        return [event for event in events if event.get("event") == name]

    static = matching(SM121_CHUNKED_PREFILL_PERFORMANCE_STATIC_EVENT)
    runtime = matching(SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EVENT)
    if [event.get("lifetime_ordinal") for event in static] != [
        quality_ordinal,
        timed_ordinal,
    ] or [event.get("lifetime_ordinal") for event in runtime] != [
        quality_ordinal,
        timed_ordinal,
    ]:
        raise ChunkedPrefillEvidenceError("chunked-prefill attestation topology changed")
    ready = matching("server_ready")
    stopped = matching("server_stopped")
    if (
        len(ready) != 2
        or [
            (event.get("backend"), event.get("lifetime_ordinal"), event.get("phase"), event.get("first_inference_is_case"), event.get("case_id"))
            for event in ready
        ]
        != [
            ("sglang", quality_ordinal, "quality", True, quality_case_id),
            ("sglang", timed_ordinal, "timed", True, timed_case_id),
        ]
        or len(stopped) != 2
        or [event.get("lifetime_ordinal") for event in stopped]
        != [quality_ordinal, timed_ordinal]
        or any(event.get("backend") != "sglang" for event in stopped)
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill server lifecycle changed")
    lifetime = matching("sm121_chunked_prefill_performance_lifetime_complete")
    if len(lifetime) != 2 or [
        (
            event.get("arm"),
            event.get("lifetime_ordinal"),
            event.get("phase"),
            event.get("within_timeout"),
            event.get("admitted"),
        )
        for event in lifetime
    ] != [
        (arm, quality_ordinal, "quality", True, True),
        (arm, timed_ordinal, "timed", True, True),
    ] or any(
        type(event.get("lifetime_wall_s")) not in {int, float}
        or event["lifetime_wall_s"] <= 0
        or not math.isfinite(float(event["lifetime_wall_s"]))
        for event in lifetime
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill lifetime admission changed")
    starts = matching("sm121_chunked_prefill_performance_quality_case_start")
    completes = matching("sm121_chunked_prefill_performance_quality_case_complete")
    if len(starts) != 1 or len(completes) != 1 or (
        starts[0].get("arm"),
        starts[0].get("lifetime_ordinal"),
        starts[0].get("case_id"),
    ) != (arm, quality_ordinal, quality_case_id):
        raise ChunkedPrefillEvidenceError("chunked-prefill quality lifecycle changed")
    _quality_attestation(events=events, arm=arm, ordinal=ordinal, case_id=quality_case_id)
    timed_starts = matching("sm121_chunked_prefill_performance_timed_case_start")
    timed_completes = matching("sm121_chunked_prefill_performance_timed_case_complete")
    if len(timed_starts) != 1 or len(timed_completes) != 1 or (
        timed_starts[0].get("arm"),
        timed_starts[0].get("lifetime_ordinal"),
        timed_starts[0].get("case_id"),
    ) != (arm, timed_ordinal, timed_case_id) or (
        timed_completes[0].get("arm"),
        timed_completes[0].get("lifetime_ordinal"),
        timed_completes[0].get("case_id"),
        timed_completes[0].get("timed_admitted"),
    ) != (arm, timed_ordinal, timed_case_id, True):
        raise ChunkedPrefillEvidenceError("chunked-prefill timing lifecycle changed")
    if (
        len(matching("run_start")) != 1
        or len(matching("measurement_started")) != 1
        or len(matching("measurement_complete")) != 1
        or len(matching("run_complete")) != 1
        or matching("run_aborted")
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill arm terminal lifecycle changed")


def _validate_attestation_topology(
    *,
    study: ChunkedPrefillPerformanceStudy,
    lifetimes: Sequence[dict[str, object]],
    static_events: Sequence[dict[str, Any]],
    runtime_events: Sequence[dict[str, Any]],
) -> None:
    seen_static = 0
    seen_runtime = 0
    for campaign_ordinal, (lifetime, arm) in enumerate(
        zip(lifetimes, SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER, strict=True),
        start=1,
    ):
        expected = [campaign_ordinal * 2 - 1, campaign_ordinal * 2]
        static = [
            event
            for event in static_events
            if event.get("arm") == arm and event.get("lifetime_ordinal") in expected
        ]
        runtime = [
            event
            for event in runtime_events
            if event.get("arm") == arm and event.get("lifetime_ordinal") in expected
        ]
        seen_static += len(static)
        seen_runtime += len(runtime)
        static_ordinals = [event["lifetime_ordinal"] for event in static]
        runtime_ordinals = [event["lifetime_ordinal"] for event in runtime]
        for event in [*static, *runtime]:
            _validate_study_attestation(event, study=study)
        if (
            static_ordinals != expected[: len(static_ordinals)]
            or runtime_ordinals != expected[: len(runtime_ordinals)]
            or any(value not in static_ordinals for value in runtime_ordinals)
        ):
            raise ChunkedPrefillEvidenceError(
                "chunked-prefill attestation topology changed"
            )
        admitted = (
            lifetime["quality_admitted"] is True
            and lifetime["timed_admitted"] is True
            and lifetime["within_timeout"] is True
        )
        if admitted and (static_ordinals != expected or runtime_ordinals != expected):
            raise ChunkedPrefillEvidenceError(
                "chunked-prefill completed attestation count changed"
            )
    if seen_static != len(static_events) or seen_runtime != len(runtime_events):
        raise ChunkedPrefillEvidenceError("chunked-prefill attestation arm changed")


def validate_source(campaign_dir: Path, results_root: Path) -> dict[str, Any] | None:
    """Validate a nested source campaign before projecting any scalar data."""

    root = results_root.resolve(strict=True)
    campaign_dir = _safe_resolve(campaign_dir, root)
    campaign = _load_json(campaign_dir / "campaign.json", root)
    if type(campaign) is not dict or frozenset(campaign) != _CAMPAIGN_FIELDS:
        raise ChunkedPrefillEvidenceError("chunked-prefill campaign fields are invalid")
    integrity = campaign.get("integrity_hash")
    if not isinstance(integrity, str) or content_hash(
        {key: value for key, value in campaign.items() if key != "integrity_hash"},
        len(integrity),
    ) != integrity:
        raise ChunkedPrefillEvidenceError("chunked-prefill campaign integrity is invalid")
    study = _study_from_campaign_id(campaign.get("campaign_id"))
    _require_publication_admission(study)
    if (
        campaign.get("schema_version") != 1
        or campaign.get("execution_mode") != study.execution_mode
        or not isinstance(campaign.get("created_at"), str)
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill campaign contract is invalid")
    binding = campaign.get("pair_binding")
    try:
        validate_sm121_chunked_prefill_performance_pair_binding(binding)
    except SM121ChunkedPrefillPerformanceError as error:
        raise ChunkedPrefillEvidenceError("chunked-prefill pair binding is invalid") from error
    assert isinstance(binding, dict)
    if binding.get("suite_id") != study.suite_id:
        raise ChunkedPrefillEvidenceError("chunked-prefill pair binding study changed")
    names = campaign.get("run_directories")
    if (
        type(names) is not list
        or len(names) != len(SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER)
        or len(set(names)) != len(names)
        or any(
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", name) is None
            for name in names
        )
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill run topology is invalid")
    runs_root = campaign_dir / "runs"
    if runs_root.is_symlink() or not runs_root.is_dir():
        raise ChunkedPrefillEvidenceError("chunked-prefill run root is invalid")
    plans: list[dict[str, Any]] = []
    events_by_arm: list[list[dict[str, Any]]] = []
    case_ids_by_arm: list[tuple[str, str]] = []
    static_events: list[dict[str, Any]] = []
    runtime_events: list[dict[str, Any]] = []
    for ordinal, (name, arm) in enumerate(
        zip(names, SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER, strict=True),
        start=1,
    ):
        run_dir = runs_root / name
        if run_dir.is_symlink() or not run_dir.is_dir() or run_dir.parent != runs_root:
            raise ChunkedPrefillEvidenceError("chunked-prefill run directory is invalid")
        plan = _load_json(run_dir / "plan.json", root)
        quality_case_id, timed_case_id = _validate_plan(
            plan=plan,
            binding=binding,
            study=study,
            ordinal=ordinal,
            expected_arm=arm,
        )
        assert isinstance(plan, dict)
        plans.append(plan)
        case_ids_by_arm.append((quality_case_id, timed_case_id))
        events_path = run_dir / "events.jsonl"
        events = _load_json_lines(events_path, root) if events_path.is_file() else []
        if any(event.get("event") not in _SAFE_EVENT_NAMES for event in events):
            raise ChunkedPrefillEvidenceError(
                "chunked-prefill journal contains an unexpected event"
            )
        _validate_run_start(
            events=events,
            study=study,
            binding=binding,
            arm=arm,
            ordinal=ordinal,
            plan=plan,
        )
        for event in events:
            try:
                if event.get("event") == SM121_CHUNKED_PREFILL_PERFORMANCE_STATIC_EVENT:
                    validate_sm121_chunked_prefill_performance_static_event(event)
                    _validate_study_attestation(event, study=study)
                    static_events.append(_without_timestamp(event))
                elif event.get("event") == SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EVENT:
                    validate_sm121_chunked_prefill_performance_runtime_event(event)
                    _validate_study_attestation(event, study=study)
                    runtime_events.append(_without_timestamp(event))
                elif event.get("event") == SM121_CHUNKED_PREFILL_PERFORMANCE_TURN_EVENT:
                    validate_sm121_chunked_prefill_performance_recorded_turn_event(event)
                    if event.get("protocol_case_id") != study.timed_case_id:
                        raise ChunkedPrefillEvidenceError(
                            "chunked-prefill turn study changed"
                        )
            except SM121ChunkedPrefillPerformanceError as error:
                raise ChunkedPrefillEvidenceError(
                    "chunked-prefill journal event is invalid"
                ) from error
        events_by_arm.append(events)
    try:
        instance = sm121_chunked_prefill_performance_pair_instance_sha256(
            [plan["run_nonce"] for plan in plans]
        )
    except SM121ChunkedPrefillPerformanceError as error:
        raise ChunkedPrefillEvidenceError("chunked-prefill nonce binding is invalid") from error
    if (
        binding.get("campaign_instance_sha256") != instance
        or binding.get("plan_fingerprints") != [plan.get("fingerprint") for plan in plans]
        or binding.get("pair_binding_sha256")
        != sm121_chunked_prefill_performance_pair_binding_sha256(binding)
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill pair binding moved")
    summary_path = campaign_dir / "summary.json"
    has_events = any(events for events in events_by_arm)
    if not summary_path.is_file():
        if has_events:
            raise ChunkedPrefillEvidenceError(
                "started chunked-prefill campaign lacks summary"
            )
        return None
    summary = _load_json(summary_path, root)
    if type(summary) is not dict or frozenset(summary) != _SUMMARY_FIELDS:
        raise ChunkedPrefillEvidenceError("chunked-prefill summary fields are invalid")
    summary_integrity = summary.get("integrity_hash")
    if not isinstance(summary_integrity, str) or content_hash(
        {key: value for key, value in summary.items() if key != "integrity_hash"},
        len(summary_integrity),
    ) != summary_integrity:
        raise ChunkedPrefillEvidenceError("chunked-prefill summary integrity is invalid")
    if (
        summary.get("schema_version") != 1
        or summary.get("campaign_id") != study.campaign_id
        or summary.get("execution_mode") != study.execution_mode
        or summary.get("pair_binding_sha256") != binding["pair_binding_sha256"]
        or summary.get("status") not in {"complete", "partial"}
        or not isinstance(summary.get("lifetimes"), list)
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill summary contract is invalid")
    try:
        score = score_sm121_chunked_prefill_performance_campaign(
            summary["lifetimes"], study=study
        )
        lifetime_rows = validate_sm121_chunked_prefill_performance_lifetimes(
            summary["lifetimes"], study=study
        )
    except SM121ChunkedPrefillPerformanceError as error:
        raise ChunkedPrefillEvidenceError("chunked-prefill summary score is invalid") from error
    if (
        summary.get("status") != score.status
        or summary.get("decision") != score.decision
        or summary.get("score") != score.to_mapping()
        or summary.get("completed_arms")
        != sum(
            1
            for lifetime in summary["lifetimes"]
            if isinstance(lifetime, dict)
            and lifetime.get("quality_admitted") is True
            and lifetime.get("timed_admitted") is True
            and lifetime.get("within_timeout") is True
        )
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill summary reduction changed")
    _validate_attestation_topology(
        study=study,
        lifetimes=lifetime_rows,
        static_events=static_events,
        runtime_events=runtime_events,
    )
    quality_attestations: list[dict[str, object]] = []
    terminal = False
    for ordinal, (lifetime, events, case_ids) in enumerate(
        zip(lifetime_rows, events_by_arm, case_ids_by_arm, strict=True), start=1
    ):
        complete = (
            lifetime["quality_admitted"] is True
            and lifetime["timed_admitted"] is True
            and lifetime["within_timeout"] is True
        )
        if terminal:
            if events or complete:
                raise ChunkedPrefillEvidenceError(
                    "chunked-prefill continued after terminal arm"
                )
            continue
        turns = [
            _without_timestamp(event)
            for event in events
            if event.get("event") == SM121_CHUNKED_PREFILL_PERFORMANCE_TURN_EVENT
        ]
        if lifetime["turns"] != turns or any(
            event.get("arm") != SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER[ordinal - 1]
            or event.get("lifetime_ordinal") != ordinal * 2
            for event in turns
        ):
            raise ChunkedPrefillEvidenceError("chunked-prefill turn journal changed")
        quality = _quality_attestation(
            events=events,
            arm=SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER[ordinal - 1],
            ordinal=ordinal,
            case_id=case_ids[0],
        )
        if lifetime["quality_admitted"] is True:
            if quality is None:
                raise ChunkedPrefillEvidenceError("chunked-prefill quality proof is absent")
            quality_attestations.append(quality)
        elif quality is not None:
            raise ChunkedPrefillEvidenceError("chunked-prefill quality proof changed")
        if not complete:
            terminal = True
            continue
        if len(turns) != len(SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS):
            raise ChunkedPrefillEvidenceError("chunked-prefill completed turns changed")
        _validate_completed_lifecycle(
            events=events,
            arm=SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER[ordinal - 1],
            ordinal=ordinal,
            quality_case_id=case_ids[0],
            timed_case_id=case_ids[1],
        )
    if not terminal and summary["status"] != "complete":
        raise ChunkedPrefillEvidenceError("chunked-prefill terminal status changed")
    if terminal and summary["status"] != "partial":
        raise ChunkedPrefillEvidenceError("chunked-prefill partial status changed")
    return {
        "study": study,
        "binding": binding,
        "summary": summary,
        "static_events": static_events,
        "runtime_events": runtime_events,
        "quality_attestations": quality_attestations,
    }


def evidence_id(
    binding: Mapping[str, object], *, study: ChunkedPrefillPerformanceStudy
) -> str:
    instance = binding.get("campaign_instance_sha256")
    if not isinstance(instance, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", instance) is None:
        raise ChunkedPrefillEvidenceError("chunked-prefill public instance is invalid")
    return (
        study.campaign_id
        + "-"
        + instance.removeprefix("sha256:")[:12]
    )


def manifest_from_source(
    source: Mapping[str, Any], *, schema_version: str, sanitization_policy: str
) -> dict[str, Any]:
    binding = source.get("binding")
    summary = source.get("summary")
    study = source.get("study")
    if (
        not isinstance(binding, Mapping)
        or not isinstance(summary, Mapping)
        or not isinstance(study, ChunkedPrefillPerformanceStudy)
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill source projection is invalid")
    _require_publication_admission(study)
    return {
        "schema_version": schema_version,
        "evidence_kind": EVIDENCE_KIND,
        "campaign_id": evidence_id(binding, study=study),
        "protocol": {
            "campaign_id": study.campaign_id,
            "suite_id": study.suite_id,
            "execution_mode": study.execution_mode,
            "arm_order": list(SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER),
            "chunked_prefill_sizes": [
                study.control_chunk_size,
                study.candidate_chunk_size,
            ],
            "cell_timeout_s": SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S,
            "quality_item_count": SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT,
            "timed_turns": list(SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS),
            "measurement": "non_streaming_request_wall_s_only",
            "primary": "cache_cold_t0_request_wall_s",
            "ttft": None,
        },
        "binding": {
            "campaign_instance_sha256": binding["campaign_instance_sha256"],
            "pair_binding_sha256": binding["pair_binding_sha256"],
        },
        "status": summary["status"],
        "decision": summary["decision"],
        "completed_arms": summary["completed_arms"],
        "lifetimes": summary["lifetimes"],
        "score": summary["score"],
        "static_attestations": source["static_events"],
        "runtime_attestations": source["runtime_events"],
        "quality_attestations": source["quality_attestations"],
        "sanitization": {
            "free_form_text_included": False,
            "payloads_included": False,
            "policy": sanitization_policy,
            "raw_identifiers_included": False,
        },
    }


def verify_manifest(
    manifest: object,
    entry: Mapping[str, object],
    *,
    schema_version: str,
    sanitization_policy: str,
) -> None:
    """Verify a published scalar bundle without opening its source results."""

    expected_fields = {
        "schema_version",
        "evidence_kind",
        "campaign_id",
        "protocol",
        "binding",
        "status",
        "decision",
        "completed_arms",
        "lifetimes",
        "score",
        "static_attestations",
        "runtime_attestations",
        "quality_attestations",
        "sanitization",
    }
    if type(manifest) is not dict or set(manifest) != expected_fields:
        raise ChunkedPrefillEvidenceError("chunked-prefill manifest fields changed")
    protocol = manifest.get("protocol")
    if type(protocol) is not dict:
        raise ChunkedPrefillEvidenceError("chunked-prefill protocol changed")
    study = _study_from_campaign_id(protocol.get("campaign_id"))
    _require_publication_admission(study)
    if (
        manifest.get("schema_version") != schema_version
        or manifest.get("evidence_kind") != EVIDENCE_KIND
        or manifest.get("campaign_id") != entry.get("campaign_id")
        or not isinstance(manifest.get("campaign_id"), str)
        or not manifest["campaign_id"].startswith(
            study.campaign_id + "-"
        )
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill manifest identity changed")
    expected_protocol = {
        "campaign_id": study.campaign_id,
        "suite_id": study.suite_id,
        "execution_mode": study.execution_mode,
        "arm_order": list(SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER),
        "chunked_prefill_sizes": [
            study.control_chunk_size,
            study.candidate_chunk_size,
        ],
        "cell_timeout_s": SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S,
        "quality_item_count": SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT,
        "timed_turns": list(SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS),
        "measurement": "non_streaming_request_wall_s_only",
        "primary": "cache_cold_t0_request_wall_s",
        "ttft": None,
    }
    if manifest.get("protocol") != expected_protocol:
        raise ChunkedPrefillEvidenceError("chunked-prefill protocol changed")
    binding = manifest.get("binding")
    if (
        type(binding) is not dict
        or set(binding) != {"campaign_instance_sha256", "pair_binding_sha256"}
        or not isinstance(binding.get("campaign_instance_sha256"), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", binding["campaign_instance_sha256"])
        is None
        or not isinstance(binding.get("pair_binding_sha256"), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", binding["pair_binding_sha256"])
        is None
        or not manifest["campaign_id"].endswith(
            binding["campaign_instance_sha256"].removeprefix("sha256:")[:12]
        )
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill public binding changed")
    lifetimes = manifest.get("lifetimes")
    try:
        score = score_sm121_chunked_prefill_performance_campaign(lifetimes, study=study)
        lifetime_rows = validate_sm121_chunked_prefill_performance_lifetimes(
            lifetimes, study=study
        )
    except SM121ChunkedPrefillPerformanceError as error:
        raise ChunkedPrefillEvidenceError("chunked-prefill public score is invalid") from error
    if (
        manifest.get("status") != score.status
        or manifest.get("decision") != score.decision
        or manifest.get("score") != score.to_mapping()
        or manifest.get("completed_arms")
        != sum(
            1
            for lifetime in lifetimes
            if isinstance(lifetime, dict)
            and lifetime.get("quality_admitted") is True
            and lifetime.get("timed_admitted") is True
            and lifetime.get("within_timeout") is True
        )
    ):
        raise ChunkedPrefillEvidenceError("chunked-prefill public reduction changed")
    static, runtime = manifest.get("static_attestations"), manifest.get("runtime_attestations")
    if not isinstance(static, list) or not isinstance(runtime, list):
        raise ChunkedPrefillEvidenceError("chunked-prefill public attestations are invalid")
    try:
        for event in static:
            if not isinstance(event, dict) or "timestamp" in event:
                raise SM121ChunkedPrefillPerformanceError("public static timestamp changed")
            validate_sm121_chunked_prefill_performance_static_event(event)
            _validate_study_attestation(event, study=study)
        for event in runtime:
            if not isinstance(event, dict) or "timestamp" in event:
                raise SM121ChunkedPrefillPerformanceError("public runtime timestamp changed")
            validate_sm121_chunked_prefill_performance_runtime_event(event)
            _validate_study_attestation(event, study=study)
    except SM121ChunkedPrefillPerformanceError as error:
        raise ChunkedPrefillEvidenceError("chunked-prefill public attestation changed") from error
    _validate_attestation_topology(
        study=study,
        lifetimes=lifetime_rows,
        static_events=static,
        runtime_events=runtime,
    )
    quality = manifest.get("quality_attestations")
    expected_quality = sum(
        1 for lifetime in lifetime_rows if lifetime["quality_admitted"] is True
    )
    if not isinstance(quality, list) or len(quality) != expected_quality:
        raise ChunkedPrefillEvidenceError("chunked-prefill public quality count changed")
    quality_index = 0
    for ordinal, (lifetime, arm) in enumerate(
        zip(lifetime_rows, SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER, strict=True),
        start=1,
    ):
        if lifetime["quality_admitted"] is not True:
            continue
        event = quality[quality_index]
        quality_index += 1
        if (
            type(event) is not dict
            or set(event)
            != {
                "arm",
                "quality_lifetime_ordinal",
                "case_id",
                "quality_admitted",
                "item_count",
            }
            or (
                event.get("arm"),
                event.get("quality_lifetime_ordinal"),
                event.get("quality_admitted"),
                event.get("item_count"),
            )
            != (
                arm,
                ordinal * 2 - 1,
                True,
                SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT,
            )
            or not isinstance(event.get("case_id"), str)
            or re.fullmatch(
                rf"{re.escape(SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID)}--[0-9a-f]{{12}}",
                event["case_id"],
            )
            is None
        ):
            raise ChunkedPrefillEvidenceError("chunked-prefill public quality changed")
    if manifest.get("sanitization") != {
        "free_form_text_included": False,
        "payloads_included": False,
        "policy": sanitization_policy,
        "raw_identifiers_included": False,
    }:
        raise ChunkedPrefillEvidenceError("chunked-prefill sanitization changed")
