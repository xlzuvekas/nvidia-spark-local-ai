"""Current-runtime autoresearch rounds for bounded SM121 serving experiments.

The original ``autoresearch`` controller is intentionally inseparable from the
expired 2026-08-28 TRT-LLM campaign.  This module does not weaken, clone, or
resume that controller.  Instead it provides a small runner registry for
newly-admitted SM121 experiments.  A round has one immutable control, one
immutable one-axis candidate, an explicit cutoff, and a non-resumable child
executor.  Only scalar controller state is written here; the child executor
owns all inference and its existing scalar evidence contract.

The first registry entry deliberately reuses the audited cache-policy A/B/B/A
executor.  It gives the product track a real propose -> freeze -> run -> score
loop without allowing an agent to compose arbitrary serving flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping

from .audit import audit_sm121_cache_performance_campaign
from .journal import Journal, content_hash, utc_now, write_json
from .manifest import ManifestError, load_models, load_suite
from .runner import (
    create_sm121_cache_performance_campaign,
    execute_sm121_cache_performance_campaign,
)
from .sglang_sm121_cache_performance import (
    SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID,
    SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID,
    SM121_CACHE_PERFORMANCE_CAMPAIGN_ID,
    SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S,
    SM121_CACHE_PERFORMANCE_EXECUTION_MODE,
    SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S,
    SM121_CACHE_PERFORMANCE_SUITE_ID,
)


AUTORESEARCH_V2_SCHEMA_VERSION = 1
AUTORESEARCH_V2_RESULT_ROOT = "autoresearch-v2"
AUTORESEARCH_V2_RUNNER_CACHE_POLICY = "sm121_cache_policy_performance_v1"
AUTORESEARCH_V2_CAMPAIGN_ID = "qwen38-flash-next-sm121-autoresearch-v2-cache-policy"
AUTORESEARCH_V2_EXECUTION_MODE = "sm121_autoresearch_v2_cache_policy_round"
AUTORESEARCH_V2_AXIS = "cache_policy"
AUTORESEARCH_V2_PRIMARY = "later_turn_request_wall_s"
AUTORESEARCH_V2_PROMOTION_RATIO = 0.95
AUTORESEARCH_V2_FULL_WALL_GUARDRAIL_RATIO = 1.05
AUTORESEARCH_V2_REQUIRED_LIFETIMES = 4
# The child contract has an independent 1,200-second budget for each quality
# and timing lifetime.  The additional ten-minute reserve covers handoff and
# terminal scalar audit; it is an admission bound, not a padding instruction.
AUTORESEARCH_V2_MIN_CUTOFF_REMAINING_S = (
    AUTORESEARCH_V2_REQUIRED_LIFETIMES
    * 2
    * SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S
    + 600
)

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SAFE_CHILD_DIRECTORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PREFIXED_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_FAILURE_STAGES = frozenset(
    {
        "child_execution",
        "child_audit",
        "projection",
    }
)
_DEFINITION_KEYS = frozenset({"schema_version", "campaign"})
_CAMPAIGN_KEYS = frozenset(
    {
        "id",
        "runner",
        "axis",
        "control",
        "candidate",
        "control_arm",
        "candidate_arm",
        "models",
        "suite",
        "cell_timeout_s",
        "required_lifetimes",
        "primary",
        "promotion_ratio",
        "full_wall_guardrail_ratio",
        "min_cutoff_remaining_s",
    }
)
_ROUND_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "created_at",
        "cutoff",
        "execution_mode",
        "definition_sha256",
        "runner",
        "axis",
        "control_profile_id",
        "candidate_profile_id",
        "control_arm",
        "candidate_arm",
        "suite_id",
        "cell_timeout_s",
        "required_lifetimes",
        "primary",
        "promotion_ratio",
        "full_wall_guardrail_ratio",
        "min_cutoff_remaining_s",
        "child_campaign_id",
        "child_campaign_directory",
        "child_campaign_integrity_hash",
        "child_pair_binding_sha256",
        "child_prerequisite_bundle_sha256s",
        "integrity_hash",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "execution_mode",
        "child_campaign_id",
        "child_pair_binding_sha256",
        "status",
        "decision",
        "child_status",
        "child_decision",
        "audit_ok",
        "completed_arms",
        "score",
        "integrity_hash",
    }
)
_FAILURE_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "execution_mode",
        "child_campaign_id",
        "child_pair_binding_sha256",
        "status",
        "decision",
        "failure_stage",
        "integrity_hash",
    }
)
_EVENT_FIELDS = {
    "autoresearch_v2_round_started": frozenset(
        {
            "event",
            "campaign_id",
            "execution_mode",
            "definition_sha256",
            "child_pair_binding_sha256",
        }
    ),
    "autoresearch_v2_round_scored": frozenset(
        {
            "event",
            "campaign_id",
            "child_pair_binding_sha256",
            "status",
            "decision",
            "child_status",
            "child_decision",
            "audit_ok",
            "completed_arms",
        }
    ),
    "autoresearch_v2_round_complete": frozenset(
        {
            "event",
            "campaign_id",
            "status",
            "decision",
        }
    ),
    "autoresearch_v2_round_failed": frozenset(
        {
            "event",
            "campaign_id",
            "child_pair_binding_sha256",
            "stage",
        }
    ),
}


class AutoresearchV2Error(ValueError):
    """Raised when a v2 controller contract cannot be safely honored."""


class AutoresearchV2ExecutionFailure(AutoresearchV2Error):
    """A started round preserved a terminal scalar failure record.

    The original exception stays chained for direct local diagnosis.  The CLI
    routes this type to its stable stage label and scalar summary only, so it
    does not accidentally render child request details.
    """

    def __init__(self, *, stage: str, summary: Mapping[str, object]) -> None:
        super().__init__(f"autoresearch-v2 round terminated during {stage}")
        self.stage = stage
        self.summary = dict(summary)


@dataclass(frozen=True, slots=True)
class AutoresearchV2Definition:
    """One immutable, registry-backed product-track hypothesis."""

    campaign_id: str
    runner: str
    axis: str
    control_profile_id: str
    candidate_profile_id: str
    control_arm: str
    candidate_arm: str
    models_path: Path
    suite_path: Path
    cell_timeout_s: int
    required_lifetimes: int
    primary: str
    promotion_ratio: float
    full_wall_guardrail_ratio: float
    min_cutoff_remaining_s: int

    def public_mapping(self) -> dict[str, object]:
        """Return the stable, scalar proposal projection.

        Paths are intentionally represented only by their manifest-relative
        strings in the frozen source file; the public controller projection
        does not publish machine-local paths.
        """

        return {
            "schema_version": AUTORESEARCH_V2_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "runner": self.runner,
            "axis": self.axis,
            "control_profile_id": self.control_profile_id,
            "candidate_profile_id": self.candidate_profile_id,
            "control_arm": self.control_arm,
            "candidate_arm": self.candidate_arm,
            "suite_id": SM121_CACHE_PERFORMANCE_SUITE_ID,
            "cell_timeout_s": self.cell_timeout_s,
            "required_lifetimes": self.required_lifetimes,
            "primary": self.primary,
            "promotion_ratio": self.promotion_ratio,
            "full_wall_guardrail_ratio": self.full_wall_guardrail_ratio,
            "min_cutoff_remaining_s": self.min_cutoff_remaining_s,
        }

    @property
    def definition_sha256(self) -> str:
        return content_hash(self.public_mapping(), 64)


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        detail = []
        if unknown:
            detail.append(f"unknown keys: {sorted(unknown)!r}")
        if missing:
            detail.append(f"missing keys: {sorted(missing)!r}")
        raise AutoresearchV2Error(f"{name} has {'; '.join(detail)}")


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AutoresearchV2Error(f"{name} must be a non-empty string")
    return value


def _require_id(value: object, *, name: str) -> str:
    parsed = _require_string(value, name=name)
    if _SAFE_ID.fullmatch(parsed) is None:
        raise AutoresearchV2Error(f"{name} must be a stable lowercase identifier")
    return parsed


def _require_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AutoresearchV2Error(f"{name} must be an integer at least {minimum}")
    return value


def _require_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AutoresearchV2Error(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise AutoresearchV2Error(f"{name} must be finite")
    return parsed


def _parse_cutoff(value: str, *, now: datetime | None = None) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise AutoresearchV2Error("cutoff must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AutoresearchV2Error("cutoff must include a UTC offset")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise AutoresearchV2Error("controller clock must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _resolve_manifest_path(definition_path: Path, value: str, *, name: str) -> Path:
    candidate = (definition_path.parent / value).resolve()
    try:
        # Campaign declarations conventionally live in ``manifests/campaigns``
        # and refer to their sibling model/suite files through ``..``.  Keep
        # that useful form, but reject paths escaping the manifests tree.
        candidate.relative_to(definition_path.parent.parent.resolve())
    except ValueError as error:
        raise AutoresearchV2Error(f"{name} escapes the manifests directory") from error
    if not candidate.is_file():
        raise AutoresearchV2Error(f"{name} is unavailable")
    return candidate


def load_autoresearch_v2_definition(path: Path) -> AutoresearchV2Definition:
    """Load the one supported immutable v2 proposal declaration."""

    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise AutoresearchV2Error("autoresearch-v2 campaign is unreadable") from error
    if not isinstance(raw, dict):
        raise AutoresearchV2Error("autoresearch-v2 campaign is invalid")
    _require_exact_keys(raw, _DEFINITION_KEYS, "campaign")
    if raw.get("schema_version") != AUTORESEARCH_V2_SCHEMA_VERSION:
        raise AutoresearchV2Error("autoresearch-v2 schema version is invalid")
    campaign = raw.get("campaign")
    if not isinstance(campaign, dict):
        raise AutoresearchV2Error("campaign section is invalid")
    _require_exact_keys(campaign, _CAMPAIGN_KEYS, "campaign")
    campaign_id = _require_id(campaign["id"], name="campaign.id")
    runner = _require_id(campaign["runner"], name="campaign.runner")
    axis = _require_id(campaign["axis"], name="campaign.axis")
    control = _require_id(campaign["control"], name="campaign.control")
    candidate = _require_id(campaign["candidate"], name="campaign.candidate")
    control_arm = _require_string(campaign["control_arm"], name="campaign.control_arm")
    candidate_arm = _require_string(campaign["candidate_arm"], name="campaign.candidate_arm")
    models_value = _require_string(campaign["models"], name="campaign.models")
    suite_value = _require_string(campaign["suite"], name="campaign.suite")
    models_path = _resolve_manifest_path(path, models_value, name="campaign.models")
    suite_path = _resolve_manifest_path(path, suite_value, name="campaign.suite")
    cell_timeout_s = _require_int(campaign["cell_timeout_s"], name="campaign.cell_timeout_s")
    required_lifetimes = _require_int(
        campaign["required_lifetimes"], name="campaign.required_lifetimes", minimum=1
    )
    primary = _require_id(campaign["primary"], name="campaign.primary")
    promotion_ratio = _require_float(campaign["promotion_ratio"], name="campaign.promotion_ratio")
    guardrail_ratio = _require_float(
        campaign["full_wall_guardrail_ratio"], name="campaign.full_wall_guardrail_ratio"
    )
    minimum_remaining = _require_int(
        campaign["min_cutoff_remaining_s"],
        name="campaign.min_cutoff_remaining_s",
        minimum=1,
    )
    definition = AutoresearchV2Definition(
        campaign_id=campaign_id,
        runner=runner,
        axis=axis,
        control_profile_id=control,
        candidate_profile_id=candidate,
        control_arm=control_arm,
        candidate_arm=candidate_arm,
        models_path=models_path,
        suite_path=suite_path,
        cell_timeout_s=cell_timeout_s,
        required_lifetimes=required_lifetimes,
        primary=primary,
        promotion_ratio=promotion_ratio,
        full_wall_guardrail_ratio=guardrail_ratio,
        min_cutoff_remaining_s=minimum_remaining,
    )
    _validate_supported_definition(definition)
    return definition


def _validate_supported_definition(definition: AutoresearchV2Definition) -> None:
    """Fail closed unless every declared field names the reviewed runner."""

    expected = {
        "campaign_id": AUTORESEARCH_V2_CAMPAIGN_ID,
        "runner": AUTORESEARCH_V2_RUNNER_CACHE_POLICY,
        "axis": AUTORESEARCH_V2_AXIS,
        "control_profile_id": SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID,
        "candidate_profile_id": SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID,
        "control_arm": "A",
        "candidate_arm": "B",
        "cell_timeout_s": SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S,
        "required_lifetimes": AUTORESEARCH_V2_REQUIRED_LIFETIMES,
        "primary": AUTORESEARCH_V2_PRIMARY,
        "promotion_ratio": AUTORESEARCH_V2_PROMOTION_RATIO,
        "full_wall_guardrail_ratio": AUTORESEARCH_V2_FULL_WALL_GUARDRAIL_RATIO,
        "min_cutoff_remaining_s": AUTORESEARCH_V2_MIN_CUTOFF_REMAINING_S,
    }
    for field, expected_value in expected.items():
        if getattr(definition, field) != expected_value:
            raise AutoresearchV2Error(
                f"{field} does not match the reviewed autoresearch-v2 runner"
            )


def _expected_definition_sha256() -> str:
    """Return the registry's immutable scalar proposal commitment."""

    return content_hash(
        {
            "schema_version": AUTORESEARCH_V2_SCHEMA_VERSION,
            "campaign_id": AUTORESEARCH_V2_CAMPAIGN_ID,
            "runner": AUTORESEARCH_V2_RUNNER_CACHE_POLICY,
            "axis": AUTORESEARCH_V2_AXIS,
            "control_profile_id": SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID,
            "candidate_profile_id": SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID,
            "control_arm": "A",
            "candidate_arm": "B",
            "suite_id": SM121_CACHE_PERFORMANCE_SUITE_ID,
            "cell_timeout_s": SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S,
            "required_lifetimes": AUTORESEARCH_V2_REQUIRED_LIFETIMES,
            "primary": AUTORESEARCH_V2_PRIMARY,
            "promotion_ratio": AUTORESEARCH_V2_PROMOTION_RATIO,
            "full_wall_guardrail_ratio": AUTORESEARCH_V2_FULL_WALL_GUARDRAIL_RATIO,
            "min_cutoff_remaining_s": AUTORESEARCH_V2_MIN_CUTOFF_REMAINING_S,
        },
        64,
    )


def preview_autoresearch_v2(path: Path) -> dict[str, object]:
    """Return the scalar hypothesis that would be frozen, without writing."""

    return load_autoresearch_v2_definition(path).public_mapping()


def _round_root(results_root: Path) -> Path:
    return results_root / AUTORESEARCH_V2_RESULT_ROOT


def _round_payload_without_hash(payload: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != "integrity_hash"}


def _validate_hashed_payload(
    payload: object,
    *,
    fields: frozenset[str],
    name: str,
) -> dict[str, object]:
    if not isinstance(payload, dict) or frozenset(payload) != fields:
        raise AutoresearchV2Error(f"{name} fields are invalid")
    integrity_hash = payload.get("integrity_hash")
    if not isinstance(integrity_hash, str) or _SHA256.fullmatch(integrity_hash) is None:
        raise AutoresearchV2Error(f"{name} integrity hash is invalid")
    if content_hash(_round_payload_without_hash(payload), 64) != integrity_hash:
        raise AutoresearchV2Error(f"{name} integrity hash does not match")
    return dict(payload)


def _load_round(round_dir: Path) -> dict[str, object]:
    try:
        candidate = round_dir.resolve(strict=True)
    except OSError as error:
        raise AutoresearchV2Error("autoresearch-v2 round is unavailable") from error
    if candidate.parent.name != AUTORESEARCH_V2_RESULT_ROOT:
        raise AutoresearchV2Error("autoresearch-v2 round is outside its result root")
    try:
        payload = json.loads((candidate / "round.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AutoresearchV2Error("autoresearch-v2 round is unreadable") from error
    return _validate_hashed_payload(payload, fields=_ROUND_FIELDS, name="round")


def _results_root_for_round(round_dir: Path) -> Path:
    root = round_dir.resolve(strict=True)
    if root.parent.name != AUTORESEARCH_V2_RESULT_ROOT:
        raise AutoresearchV2Error("autoresearch-v2 round has an invalid parent")
    return root.parent.parent


def _child_campaign_path(round_dir: Path, round_payload: Mapping[str, object]) -> Path:
    name = round_payload.get("child_campaign_directory")
    if not isinstance(name, str) or _SAFE_CHILD_DIRECTORY.fullmatch(name) is None:
        raise AutoresearchV2Error("autoresearch-v2 child campaign name is invalid")
    child = _results_root_for_round(round_dir) / "cache-policy-campaigns" / name
    try:
        child.resolve(strict=True).relative_to(
            (_results_root_for_round(round_dir) / "cache-policy-campaigns").resolve(
                strict=True
            )
        )
    except (OSError, ValueError) as error:
        raise AutoresearchV2Error("autoresearch-v2 child campaign is unavailable") from error
    return child


def _validate_child_binding(round_dir: Path, round_payload: Mapping[str, object]) -> Path:
    child = _child_campaign_path(round_dir, round_payload)
    try:
        campaign = json.loads((child / "campaign.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AutoresearchV2Error("autoresearch-v2 child campaign is unreadable") from error
    if not isinstance(campaign, dict):
        raise AutoresearchV2Error("autoresearch-v2 child campaign is invalid")
    binding = campaign.get("pair_binding")
    prerequisites = campaign.get("prerequisite_bundle_sha256s")
    if (
        campaign.get("campaign_id") != round_payload["child_campaign_id"]
        or campaign.get("campaign_id") != SM121_CACHE_PERFORMANCE_CAMPAIGN_ID
        or campaign.get("integrity_hash") != round_payload["child_campaign_integrity_hash"]
        or not isinstance(binding, dict)
        or binding.get("pair_binding_sha256") != round_payload["child_pair_binding_sha256"]
        or prerequisites
        != list(SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S)
        or round_payload.get("child_prerequisite_bundle_sha256s")
        != list(SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S)
    ):
        raise AutoresearchV2Error("autoresearch-v2 child campaign binding changed")
    return child


def _validate_round_definition(round_payload: Mapping[str, object]) -> None:
    expected = {
        "schema_version": AUTORESEARCH_V2_SCHEMA_VERSION,
        "campaign_id": AUTORESEARCH_V2_CAMPAIGN_ID,
        "execution_mode": AUTORESEARCH_V2_EXECUTION_MODE,
        "runner": AUTORESEARCH_V2_RUNNER_CACHE_POLICY,
        "axis": AUTORESEARCH_V2_AXIS,
        "control_profile_id": SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID,
        "candidate_profile_id": SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID,
        "control_arm": "A",
        "candidate_arm": "B",
        "suite_id": SM121_CACHE_PERFORMANCE_SUITE_ID,
        "cell_timeout_s": SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S,
        "required_lifetimes": AUTORESEARCH_V2_REQUIRED_LIFETIMES,
        "primary": AUTORESEARCH_V2_PRIMARY,
        "promotion_ratio": AUTORESEARCH_V2_PROMOTION_RATIO,
        "full_wall_guardrail_ratio": AUTORESEARCH_V2_FULL_WALL_GUARDRAIL_RATIO,
        "min_cutoff_remaining_s": AUTORESEARCH_V2_MIN_CUTOFF_REMAINING_S,
        "child_campaign_id": SM121_CACHE_PERFORMANCE_CAMPAIGN_ID,
    }
    for field, expected_value in expected.items():
        if round_payload.get(field) != expected_value:
            raise AutoresearchV2Error(f"autoresearch-v2 round {field} changed")
    created_at = round_payload.get("created_at")
    if not isinstance(created_at, str):
        raise AutoresearchV2Error("autoresearch-v2 creation time is invalid")
    _parse_cutoff(created_at)
    for field in ("definition_sha256", "child_campaign_integrity_hash"):
        value = round_payload.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise AutoresearchV2Error(f"autoresearch-v2 round {field} is invalid")
    if round_payload["definition_sha256"] != _expected_definition_sha256():
        raise AutoresearchV2Error("autoresearch-v2 definition binding changed")
    pair_binding = round_payload.get("child_pair_binding_sha256")
    if not isinstance(pair_binding, str) or _PREFIXED_SHA256.fullmatch(pair_binding) is None:
        raise AutoresearchV2Error("autoresearch-v2 round child pair binding is invalid")
    prerequisites = round_payload.get("child_prerequisite_bundle_sha256s")
    if prerequisites != list(SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S):
        raise AutoresearchV2Error("autoresearch-v2 prerequisite binding changed")
    child_name = round_payload.get("child_campaign_directory")
    if (
        not isinstance(child_name, str)
        or _SAFE_CHILD_DIRECTORY.fullmatch(child_name) is None
    ):
        raise AutoresearchV2Error("autoresearch-v2 child campaign directory is invalid")
    cutoff = round_payload.get("cutoff")
    if not isinstance(cutoff, str):
        raise AutoresearchV2Error("autoresearch-v2 cutoff is invalid")
    _parse_cutoff(cutoff)


def freeze_autoresearch_v2(
    definition_path: Path,
    *,
    results_root: Path,
    evidence_root: Path,
    cutoff: str,
    now: datetime | None = None,
) -> Path:
    """Freeze a current-runtime v2 round and its exact child campaign.

    Planning invokes no server.  It validates the existing cache-policy
    prerequisite evidence via the child planner before a fresh four-arm plan
    is materialized.
    """

    definition = load_autoresearch_v2_definition(definition_path)
    current = now or datetime.now(timezone.utc)
    cutoff_at = _parse_cutoff(cutoff, now=current)
    remaining_s = (cutoff_at - current.astimezone(timezone.utc)).total_seconds()
    if not math.isfinite(remaining_s) or remaining_s < definition.min_cutoff_remaining_s:
        raise AutoresearchV2Error(
            "cutoff leaves insufficient time for the non-resumable four-arm round"
        )
    try:
        models = load_models(definition.models_path)
        suite = load_suite(definition.suite_path)
        control = models[definition.control_profile_id]
        candidate = models[definition.candidate_profile_id]
    except (ManifestError, KeyError) as error:
        raise AutoresearchV2Error("autoresearch-v2 model or suite is unavailable") from error
    if suite.id != SM121_CACHE_PERFORMANCE_SUITE_ID:
        raise AutoresearchV2Error("autoresearch-v2 suite is not the registered runner suite")
    stamp = current.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    round_dir = _round_root(results_root) / f"{stamp}-{definition.campaign_id}"
    round_dir.mkdir(parents=True, exist_ok=False)
    try:
        child = create_sm121_cache_performance_campaign(
            cache_on_model=control,
            cache_off_model=candidate,
            suite=suite,
            results_root=results_root / "cache-policy-campaigns",
            models_path=definition.models_path,
            suite_path=definition.suite_path,
            evidence_root=evidence_root,
        )
        campaign = json.loads((child / "campaign.json").read_text())
        if not isinstance(campaign, dict) or not isinstance(campaign.get("pair_binding"), dict):
            raise AutoresearchV2Error("autoresearch-v2 child campaign is invalid")
        binding = campaign["pair_binding"]
        child_name = child.name
        if (
            _SAFE_CHILD_DIRECTORY.fullmatch(child_name) is None
            or not isinstance(campaign.get("integrity_hash"), str)
        or not isinstance(binding.get("pair_binding_sha256"), str)
        ):
            raise AutoresearchV2Error("autoresearch-v2 child campaign binding is invalid")
        payload: dict[str, object] = {
            "schema_version": AUTORESEARCH_V2_SCHEMA_VERSION,
            "campaign_id": definition.campaign_id,
            "created_at": utc_now(),
            "cutoff": cutoff,
            "execution_mode": AUTORESEARCH_V2_EXECUTION_MODE,
            "definition_sha256": definition.definition_sha256,
            "runner": definition.runner,
            "axis": definition.axis,
            "control_profile_id": definition.control_profile_id,
            "candidate_profile_id": definition.candidate_profile_id,
            "control_arm": definition.control_arm,
            "candidate_arm": definition.candidate_arm,
            "suite_id": SM121_CACHE_PERFORMANCE_SUITE_ID,
            "cell_timeout_s": definition.cell_timeout_s,
            "required_lifetimes": definition.required_lifetimes,
            "primary": definition.primary,
            "promotion_ratio": definition.promotion_ratio,
            "full_wall_guardrail_ratio": definition.full_wall_guardrail_ratio,
            "min_cutoff_remaining_s": definition.min_cutoff_remaining_s,
            "child_campaign_id": SM121_CACHE_PERFORMANCE_CAMPAIGN_ID,
            "child_campaign_directory": child_name,
            "child_campaign_integrity_hash": campaign["integrity_hash"],
            "child_pair_binding_sha256": binding["pair_binding_sha256"],
            "child_prerequisite_bundle_sha256s": list(
                SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S
            ),
        }
        payload["integrity_hash"] = content_hash(payload, 64)
        write_json(round_dir / "round.json", payload)
    except BaseException:
        # Preserve a failed planning root as ignored raw provenance.  It never
        # starts inference and a later round cannot reuse its child plans.
        raise
    return round_dir


def _decision_from_child(child_summary: Mapping[str, object], audit: Mapping[str, object]) -> tuple[str, str]:
    """Map the registered child reducer into v2 retain/reject/inconclusive."""

    child_status = child_summary.get("status")
    child_decision = child_summary.get("decision")
    if audit.get("ok") is not True or not isinstance(child_decision, str):
        return "partial", "inconclusive"
    if child_status == "complete" and child_decision == "retain_b":
        return "complete", "retain"
    if child_status == "complete" and child_decision == "retain_a":
        return "complete", "reject"
    if child_decision in {"no_retention", "guardrail_reject", "not_evaluated"}:
        return ("complete" if child_status == "complete" else "partial"), "inconclusive"
    return "partial", "inconclusive"


def _scalar_score(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AutoresearchV2Error("autoresearch-v2 child score is invalid")
    expected = {
        "status",
        "decision",
        "a_later_wall_s",
        "b_later_wall_s",
        "a_full_wall_s",
        "b_full_wall_s",
        "winner_later_wall_ratio",
        "winner_full_wall_ratio",
    }
    if set(value) != expected:
        raise AutoresearchV2Error("autoresearch-v2 child score fields changed")
    projected: dict[str, object] = {}
    for key, item in value.items():
        if key in {"status", "decision"}:
            if not isinstance(item, str) or len(item) > 64:
                raise AutoresearchV2Error("autoresearch-v2 child score text is invalid")
        elif item is not None and (
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item))
        ):
            raise AutoresearchV2Error("autoresearch-v2 child score number is invalid")
        projected[key] = item
    return projected


def _summary_payload(
    round_payload: Mapping[str, object],
    *,
    child_summary: Mapping[str, object],
    audit: Mapping[str, object],
) -> dict[str, object]:
    status, decision = _decision_from_child(child_summary, audit)
    completed_arms = child_summary.get("completed_arms")
    if isinstance(completed_arms, bool) or not isinstance(completed_arms, int):
        raise AutoresearchV2Error("autoresearch-v2 child arm count is invalid")
    if completed_arms < 0 or completed_arms > AUTORESEARCH_V2_REQUIRED_LIFETIMES:
        raise AutoresearchV2Error("autoresearch-v2 child arm count is out of range")
    child_status = child_summary.get("status")
    child_decision = child_summary.get("decision")
    if not isinstance(child_status, str) or not isinstance(child_decision, str):
        raise AutoresearchV2Error("autoresearch-v2 child terminal state is invalid")
    payload: dict[str, object] = {
        "schema_version": AUTORESEARCH_V2_SCHEMA_VERSION,
        "campaign_id": round_payload["campaign_id"],
        "execution_mode": round_payload["execution_mode"],
        "child_campaign_id": round_payload["child_campaign_id"],
        "child_pair_binding_sha256": round_payload["child_pair_binding_sha256"],
        "status": status,
        "decision": decision,
        "child_status": child_status,
        "child_decision": child_decision,
        "audit_ok": audit.get("ok") is True,
        "completed_arms": completed_arms,
        "score": _scalar_score(child_summary.get("score")),
    }
    payload["integrity_hash"] = content_hash(payload, 64)
    return payload


def _failure_summary_payload(
    round_payload: Mapping[str, object], *, stage: str
) -> dict[str, object]:
    """Return the scalar-only terminal result for a wrapper-level failure.

    A child controller normally owns its own partial/failed summary.  This
    narrow fallback covers failures before that summary can be trusted (for
    example a child-executor or audit exception).  It deliberately carries no
    exception type, message, path, request, or child-derived timing.
    """

    if stage not in _FAILURE_STAGES:
        raise AutoresearchV2Error("autoresearch-v2 failure stage is invalid")
    payload: dict[str, object] = {
        "schema_version": AUTORESEARCH_V2_SCHEMA_VERSION,
        "campaign_id": round_payload["campaign_id"],
        "execution_mode": round_payload["execution_mode"],
        "child_campaign_id": round_payload["child_campaign_id"],
        "child_pair_binding_sha256": round_payload["child_pair_binding_sha256"],
        "status": "partial",
        "decision": "inconclusive",
        "failure_stage": stage,
    }
    payload["integrity_hash"] = content_hash(payload, 64)
    return payload


def _failure_stage_requires_terminal_child(stage: str) -> bool:
    """Return whether a wrapper failure can only follow a terminal child."""

    if stage not in _FAILURE_STAGES:
        raise AutoresearchV2Error("autoresearch-v2 failure stage is invalid")
    return stage != "child_execution"


def _record_terminal_failure(
    round_dir: Path,
    round_payload: Mapping[str, object],
    *,
    child: Path,
    stage: str,
) -> dict[str, object]:
    """Durably retain one sanitized terminal wrapper failure.

    This record is intentionally non-resumable.  If its own durable writes
    fail, the caller must fail closed rather than infer a terminal result from
    a partial journal.
    """

    _validate_failure_child_source(round_dir, child, stage=stage)
    summary = _failure_summary_payload(round_payload, stage=stage)
    journal = Journal(round_dir / "events.jsonl")
    try:
        journal.append(
            {
                "event": "autoresearch_v2_round_failed",
                "campaign_id": round_payload["campaign_id"],
                "child_pair_binding_sha256": round_payload[
                    "child_pair_binding_sha256"
                ],
                "stage": stage,
            }
        )
        write_json(round_dir / "summary.json", summary)
    except Exception as error:
        raise AutoresearchV2Error(
            "autoresearch-v2 terminal failure record could not be persisted"
        ) from error
    return summary


def _validate_events(
    round_dir: Path,
    round_payload: Mapping[str, object],
    *,
    terminal: bool,
    failure_stage: str | None = None,
) -> str:
    journal_path = round_dir / "events.jsonl"
    journal = Journal(journal_path)
    try:
        events = journal.strict_events()
    except (OSError, ValueError) as error:
        raise AutoresearchV2Error("autoresearch-v2 event journal is invalid") from error
    if not events:
        if terminal:
            raise AutoresearchV2Error("terminal autoresearch-v2 round has no journal")
        return "unstarted"
    normal_names = (
        "autoresearch_v2_round_started",
        "autoresearch_v2_round_scored",
        "autoresearch_v2_round_complete",
    )
    failed_names = (
        "autoresearch_v2_round_started",
        "autoresearch_v2_round_failed",
    )
    names = tuple(event.get("event") for event in events)
    if names == normal_names:
        state = "complete"
    elif names == failed_names:
        state = "failed"
    elif not terminal and names == normal_names[: len(names)]:
        state = ("started", "scored")[len(names) - 1]
    else:
        raise AutoresearchV2Error("autoresearch-v2 event order is invalid")
    for event in events:
        name = event.get("event")
        if not isinstance(name, str):
            raise AutoresearchV2Error("autoresearch-v2 event order is invalid")
        scalar = {key: value for key, value in event.items() if key != "timestamp"}
        if frozenset(scalar) != _EVENT_FIELDS[name]:
            raise AutoresearchV2Error("autoresearch-v2 event fields changed")
        if scalar.get("campaign_id") != round_payload["campaign_id"]:
            raise AutoresearchV2Error("autoresearch-v2 event campaign changed")
        if name == "autoresearch_v2_round_started":
            if scalar.get("execution_mode") != round_payload["execution_mode"]:
                raise AutoresearchV2Error(
                    "autoresearch-v2 started event execution mode changed"
                )
            if scalar.get("definition_sha256") != round_payload["definition_sha256"]:
                raise AutoresearchV2Error(
                    "autoresearch-v2 started event definition binding changed"
                )
            if (
                scalar.get("child_pair_binding_sha256")
                != round_payload["child_pair_binding_sha256"]
            ):
                raise AutoresearchV2Error(
                    "autoresearch-v2 started event child binding changed"
                )
        if name == "autoresearch_v2_round_failed":
            if scalar.get("stage") not in _FAILURE_STAGES:
                raise AutoresearchV2Error("autoresearch-v2 failure stage is invalid")
            if failure_stage is not None and scalar.get("stage") != failure_stage:
                raise AutoresearchV2Error("autoresearch-v2 failure stage changed")
            if (
                scalar.get("child_pair_binding_sha256")
                != round_payload["child_pair_binding_sha256"]
            ):
                raise AutoresearchV2Error(
                    "autoresearch-v2 failure child binding changed"
                )
    return state


def _read_child_summary(child: Path) -> dict[str, object]:
    try:
        raw = json.loads((child / "summary.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AutoresearchV2Error("autoresearch-v2 child has no terminal summary") from error
    if not isinstance(raw, dict):
        raise AutoresearchV2Error("autoresearch-v2 child summary is invalid")
    return raw


def _validate_failure_child_source(
    round_dir: Path, child: Path, *, stage: str
) -> None:
    """Match the exporter's terminal-child policy for a failed wrapper.

    The scalar source validator is intentionally shared with evidence.  An
    execution-stage exception may leave a valid unstarted child or a valid
    terminal child; later stages can occur only after a valid terminal child.
    No child audit is re-run here, because the failure record specifically
    covers an audit or projection that already raised.
    """

    requires_terminal = _failure_stage_requires_terminal_child(stage)
    try:
        from .evidence import (
            EvidenceError,
            _validate_sm121_cache_performance_source,
        )

        source = _validate_sm121_cache_performance_source(
            child, _results_root_for_round(round_dir)
        )
    except (OSError, ValueError, KeyError, TypeError, EvidenceError) as error:
        raise AutoresearchV2Error(
            "autoresearch-v2 failure child does not meet the scalar contract"
        ) from error
    if source is None and requires_terminal:
        raise AutoresearchV2Error("autoresearch-v2 failure child is not terminal")


def run_autoresearch_v2(
    round_dir: Path,
    *,
    workspace: Path,
    evidence_root: Path,
    now: datetime | None = None,
) -> dict[str, object]:
    """Run one frozen v2 child once, then append its deterministic decision."""

    payload = _load_round(round_dir)
    _validate_round_definition(payload)
    if (round_dir / "summary.json").exists():
        raise AutoresearchV2Error("autoresearch-v2 round is terminal; freeze a new round")
    _validate_events(round_dir, payload, terminal=False)
    if Journal(round_dir / "events.jsonl").strict_events():
        raise AutoresearchV2Error(
            "autoresearch-v2 round was started and cannot be resumed; freeze a new round"
        )
    current = now or datetime.now(timezone.utc)
    cutoff_at = _parse_cutoff(str(payload["cutoff"]), now=current)
    remaining_s = (cutoff_at - current.astimezone(timezone.utc)).total_seconds()
    if not math.isfinite(remaining_s) or remaining_s < AUTORESEARCH_V2_MIN_CUTOFF_REMAINING_S:
        raise AutoresearchV2Error(
            "cutoff leaves insufficient time for the non-resumable four-arm round"
        )
    child = _validate_child_binding(round_dir, payload)
    journal = Journal(round_dir / "events.jsonl")
    journal.append(
        {
            "event": "autoresearch_v2_round_started",
            "campaign_id": payload["campaign_id"],
            "execution_mode": payload["execution_mode"],
            "definition_sha256": payload["definition_sha256"],
            "child_pair_binding_sha256": payload["child_pair_binding_sha256"],
        }
    )
    try:
        child_summary = execute_sm121_cache_performance_campaign(
            child, workspace=workspace, evidence_root=evidence_root
        )
    except Exception as error:
        summary = _record_terminal_failure(
            round_dir, payload, child=child, stage="child_execution"
        )
        raise AutoresearchV2ExecutionFailure(
            stage="child_execution", summary=summary
        ) from error
    try:
        audit = audit_sm121_cache_performance_campaign(child, evidence_root=evidence_root)
    except Exception as error:
        summary = _record_terminal_failure(
            round_dir, payload, child=child, stage="child_audit"
        )
        raise AutoresearchV2ExecutionFailure(
            stage="child_audit", summary=summary
        ) from error
    try:
        summary = _summary_payload(payload, child_summary=child_summary, audit=audit)
    except Exception as error:
        failure_summary = _record_terminal_failure(
            round_dir, payload, child=child, stage="projection"
        )
        raise AutoresearchV2ExecutionFailure(
            stage="projection", summary=failure_summary
        ) from error
    journal.append(
        {
            "event": "autoresearch_v2_round_scored",
            "campaign_id": payload["campaign_id"],
            "child_pair_binding_sha256": payload["child_pair_binding_sha256"],
            "status": summary["status"],
            "decision": summary["decision"],
            "child_status": summary["child_status"],
            "child_decision": summary["child_decision"],
            "audit_ok": summary["audit_ok"],
            "completed_arms": summary["completed_arms"],
        }
    )
    journal.append(
        {
            "event": "autoresearch_v2_round_complete",
            "campaign_id": payload["campaign_id"],
            "status": summary["status"],
            "decision": summary["decision"],
        }
    )
    write_json(round_dir / "summary.json", summary)
    return summary


def summarize_autoresearch_v2(
    round_dir: Path, *, evidence_root: Path
) -> dict[str, object]:
    """Read-only audit of a terminal v2 round and its child campaign."""

    payload = _load_round(round_dir)
    _validate_round_definition(payload)
    child = _validate_child_binding(round_dir, payload)
    try:
        saved = json.loads((round_dir / "summary.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AutoresearchV2Error("autoresearch-v2 round has no terminal summary") from error
    event_state = _validate_events(round_dir, payload, terminal=True)
    if event_state == "failed":
        saved_payload = _validate_hashed_payload(
            saved, fields=_FAILURE_SUMMARY_FIELDS, name="failure summary"
        )
        stage = saved_payload.get("failure_stage")
        if not isinstance(stage, str):
            raise AutoresearchV2Error("autoresearch-v2 failure stage is invalid")
        _validate_events(
            round_dir, payload, terminal=True, failure_stage=stage
        )
        expected_failure = _failure_summary_payload(payload, stage=stage)
        if saved_payload != expected_failure:
            raise AutoresearchV2Error(
                "autoresearch-v2 failure summary does not match its round"
            )
        _validate_failure_child_source(round_dir, child, stage=stage)
        return saved_payload
    saved_payload = _validate_hashed_payload(saved, fields=_SUMMARY_FIELDS, name="summary")
    source_summary = _read_child_summary(child)
    audit = audit_sm121_cache_performance_campaign(child, evidence_root=evidence_root)
    expected = _summary_payload(payload, child_summary=source_summary, audit=audit)
    if saved_payload != expected:
        raise AutoresearchV2Error("autoresearch-v2 summary does not match its child")
    return saved_payload
