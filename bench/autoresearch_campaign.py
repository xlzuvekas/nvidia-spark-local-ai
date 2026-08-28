"""Strict planning primitives for the single-user Qwen autoresearch campaign.

The campaign planner is deliberately separate from the execution controller.
It validates a finite queue of semantic one-axis candidates, constructs the
immutable policy, and can freeze fresh SparkBench plans for calibration,
screening, and reverse-order confirmation.  Planning never executes a plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import tomllib
from typing import Any, Callable, Mapping

from .autoresearch import (
    CampaignPolicy,
    CandidateDelta,
    EligibilityInputs,
    PairObservation,
    SimplificationEvidence,
    TimingInputs,
    validate_one_axis_delta,
)
from .journal import canonical_json, content_hash, utc_now, write_json
from .manifest import (
    ManifestError,
    ModelSpec,
    SuiteSpec,
    load_models,
    load_suite,
    validate_benchmark_selection,
)
from .runner import create_plan


CAMPAIGN_SCHEMA_VERSION = 1
FROZEN_CAMPAIGN_SCHEMA_VERSION = 1
EXPECTED_SUITE_ID = "qwen38-flash-next-sglang-agent64k-autoresearch"
EXPECTED_CASE_IDS = (
    "json-smoke",
    "tools-smoke",
    "synthetic-exact-answer-v2",
    "agentic-select-and-call",
    "agentic-no-tool",
    "agentic-two-hop",
    "agentic-tool-error-recovery",
    "long-context-needle-60000-agent-c1",
    "agent64k-decode-256-c1-v1",
)
EXPECTED_PRIMARY_CASE_IDS = (
    "agentic-select-and-call",
    "agentic-no-tool",
    "agentic-two-hop",
    "agentic-tool-error-recovery",
    "long-context-needle-60000-agent-c1",
    "agent64k-decode-256-c1-v1",
)
EXPECTED_AXES = (
    "reasoning_policy",
    "chunked_prefill_size",
    "nextn_bundle",
)
HOST_SAFETY_MIN_MEMAVAILABLE_KIB = 14 * 1024 * 1024
HOST_SAFETY_MAX_SWAP_GROWTH_KIB = 512 * 1024
HOST_SAFETY_MAX_STARTING_SWAP_KIB = 64 * 1024

_TOP_LEVEL_KEYS = frozenset({"schema_version", "campaign", "candidates"})
_CAMPAIGN_KEYS = frozenset(
    {
        "id",
        "cutoff",
        "models",
        "suite",
        "baseline",
        "primary_case_ids",
        "allowed_axes",
    }
)
_CANDIDATE_KEYS = frozenset({"id", "axis"})


class CampaignPlanningError(ValueError):
    """Raised when a campaign definition or frozen plan is not admissible."""


class CellProjectionError(CampaignPlanningError):
    """A typed, scalar-only reason that one cell cannot enter a score."""

    def __init__(self, message: str, *, failure_kind: str = "measurement") -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    id: str
    axis: str


@dataclass(frozen=True, slots=True)
class CampaignDefinition:
    id: str
    cutoff: str
    models_path: Path
    suite_path: Path
    baseline_id: str
    primary_case_ids: tuple[str, ...]
    allowed_axes: tuple[str, ...]
    candidates: tuple[CandidateProposal, ...]

    @property
    def cutoff_datetime(self) -> datetime:
        parsed = datetime.fromisoformat(self.cutoff)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise CampaignPlanningError("campaign cutoff must include a timezone")
        return parsed


@dataclass(frozen=True, slots=True)
class ValidatedProposal:
    candidate_id: str
    axis: str
    delta: CandidateDelta

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "axis": self.axis,
            "delta": asdict(self.delta),
            "delta_digest": self.delta.digest,
        }


@dataclass(frozen=True, slots=True)
class CampaignPreview:
    definition: CampaignDefinition
    policy: CampaignPolicy
    proposals: tuple[ValidatedProposal, ...]
    suite_id: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": self.definition.id,
            "cutoff": self.definition.cutoff,
            "baseline_id": self.definition.baseline_id,
            "suite_id": self.suite_id,
            "policy": self.policy.to_mapping(),
            "policy_digest": self.policy.digest,
            "proposals": [proposal.to_mapping() for proposal in self.proposals],
            "planned_cell_count": 2 + 4 * len(self.proposals),
            "execution_started": False,
        }

    @property
    def digest(self) -> str:
        return content_hash(self.to_mapping(), 64)


@dataclass(frozen=True, slots=True)
class CaseMeasurement:
    case_id: str
    speed_value: float
    speed_direction: str
    ttft_s: float


@dataclass(frozen=True, slots=True)
class CellProjection:
    profile_id: str
    plan_fingerprint: str
    measurements: tuple[CaseMeasurement, ...]
    measurement_elapsed_s: float
    cleanup_elapsed_s: float
    minimum_memavailable_gib: float
    maximum_swap_growth_mib: float
    normalized_flags: tuple[str, ...]

    def measurement(self, case_id: str) -> CaseMeasurement:
        matches = tuple(
            measurement
            for measurement in self.measurements
            if measurement.case_id == case_id
        )
        if len(matches) != 1:
            raise CellProjectionError(
                f"cell projection has {len(matches)} measurements for {case_id!r}"
            )
        return matches[0]


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, context: str
) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise CampaignPlanningError(
            f"{context} has unknown keys: {sorted(unknown)!r}"
        )
    if missing:
        raise CampaignPlanningError(
            f"{context} is missing keys: {sorted(missing)!r}"
        )


def _require_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CampaignPlanningError(f"{context} must be a non-empty string")
    if any(ord(character) < 32 for character in value):
        raise CampaignPlanningError(f"{context} contains control characters")
    return value


def _require_string_tuple(value: Any, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise CampaignPlanningError(f"{context} must be a non-empty array")
    parsed = tuple(
        _require_string(item, context=f"{context}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(parsed)) != len(parsed):
        raise CampaignPlanningError(f"{context} must not contain duplicates")
    return parsed


def _contained_file(base: Path, raw: Any, *, workspace: Path, context: str) -> Path:
    text = _require_string(raw, context=context)
    relative = Path(text)
    if relative.is_absolute():
        raise CampaignPlanningError(f"{context} must be relative")
    try:
        resolved = (base / relative).resolve(strict=True)
        resolved.relative_to(workspace.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise CampaignPlanningError(
            f"{context} must resolve to a file inside the workspace"
        ) from error
    if not resolved.is_file():
        raise CampaignPlanningError(f"{context} must resolve to a regular file")
    return resolved


def load_campaign_definition(
    path: str | Path, *, workspace: Path
) -> CampaignDefinition:
    manifest_path = Path(path).resolve(strict=True)
    try:
        manifest_path.relative_to(workspace.resolve(strict=True))
    except ValueError as error:
        raise CampaignPlanningError(
            "campaign manifest must be inside the workspace"
        ) from error
    try:
        document = tomllib.loads(manifest_path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise CampaignPlanningError(
            f"invalid campaign TOML: {error}"
        ) from error
    if not isinstance(document, dict):
        raise CampaignPlanningError("campaign manifest must be a TOML object")
    _require_exact_keys(document, _TOP_LEVEL_KEYS, context="campaign manifest")
    if document["schema_version"] != CAMPAIGN_SCHEMA_VERSION:
        raise CampaignPlanningError(
            f"campaign schema_version must equal {CAMPAIGN_SCHEMA_VERSION}"
        )
    campaign = document["campaign"]
    if not isinstance(campaign, dict):
        raise CampaignPlanningError("campaign must be a table")
    _require_exact_keys(campaign, _CAMPAIGN_KEYS, context="campaign")
    candidates_value = document["candidates"]
    if not isinstance(candidates_value, list) or not candidates_value:
        raise CampaignPlanningError("candidates must be a non-empty table array")
    candidates: list[CandidateProposal] = []
    for index, raw in enumerate(candidates_value):
        context = f"candidates[{index}]"
        if not isinstance(raw, dict):
            raise CampaignPlanningError(f"{context} must be a table")
        _require_exact_keys(raw, _CANDIDATE_KEYS, context=context)
        candidates.append(
            CandidateProposal(
                id=_require_string(raw["id"], context=f"{context}.id"),
                axis=_require_string(raw["axis"], context=f"{context}.axis"),
            )
        )
    if len({candidate.id for candidate in candidates}) != len(candidates):
        raise CampaignPlanningError("candidate IDs must not contain duplicates")

    cutoff = _require_string(campaign["cutoff"], context="campaign.cutoff")
    definition = CampaignDefinition(
        id=_require_string(campaign["id"], context="campaign.id"),
        cutoff=cutoff,
        models_path=_contained_file(
            manifest_path.parent,
            campaign["models"],
            workspace=workspace,
            context="campaign.models",
        ),
        suite_path=_contained_file(
            manifest_path.parent,
            campaign["suite"],
            workspace=workspace,
            context="campaign.suite",
        ),
        baseline_id=_require_string(
            campaign["baseline"], context="campaign.baseline"
        ),
        primary_case_ids=_require_string_tuple(
            campaign["primary_case_ids"], context="campaign.primary_case_ids"
        ),
        allowed_axes=_require_string_tuple(
            campaign["allowed_axes"], context="campaign.allowed_axes"
        ),
        candidates=tuple(candidates),
    )
    definition.cutoff_datetime
    if definition.primary_case_ids != EXPECTED_PRIMARY_CASE_IDS:
        raise CampaignPlanningError(
            "campaign primary_case_ids do not match the audited six-case score"
        )
    if definition.allowed_axes != EXPECTED_AXES:
        raise CampaignPlanningError(
            "campaign allowed_axes do not match the audited semantic axes"
        )
    if tuple(candidate.axis for candidate in definition.candidates) != EXPECTED_AXES:
        raise CampaignPlanningError(
            "candidate queue must follow reasoning, chunk, then NEXTN depth"
        )
    return definition


def _arg_value(model: ModelSpec, option: str) -> str:
    indexes = [index for index, value in enumerate(model.args) if value == option]
    if len(indexes) != 1:
        raise CampaignPlanningError(
            f"{model.id} must declare exactly one {option} option"
        )
    index = indexes[0]
    if index + 1 >= len(model.args) or model.args[index + 1].startswith("--"):
        raise CampaignPlanningError(f"{model.id} has no value for {option}")
    return model.args[index + 1]


def _reasoning_policy(model: ModelSpec) -> Any:
    if model.request_body_json is None:
        raise CampaignPlanningError(
            f"{model.id} must declare an explicit request reasoning policy"
        )
    try:
        value = json.loads(model.request_body_json)
    except json.JSONDecodeError as error:
        raise CampaignPlanningError(
            f"{model.id} request reasoning policy is invalid JSON"
        ) from error
    if not isinstance(value, dict):
        raise CampaignPlanningError(
            f"{model.id} request reasoning policy must be an object"
        )
    return value


def semantic_config(model: ModelSpec) -> dict[str, Any]:
    """Return the three audited serving axes for one campaign profile."""

    return {
        "reasoning_policy": _reasoning_policy(model),
        "chunked_prefill_size": int(_arg_value(model, "--chunked-prefill-size")),
        "nextn_bundle": {
            "steps": int(_arg_value(model, "--speculative-num-steps")),
            "draft_tokens": int(
                _arg_value(model, "--speculative-num-draft-tokens")
            ),
        },
    }


def _invariant_model_projection(model: ModelSpec) -> dict[str, Any]:
    projection = asdict(model)
    for key in ("id", "description", "served_name", "request_body_json"):
        projection.pop(key)
    arguments = list(model.args)
    replacements = {
        "--served-model-name": "<served-name>",
        "--chunked-prefill-size": "<chunked-prefill-size>",
        "--speculative-num-steps": "<nextn-steps>",
        "--speculative-num-draft-tokens": "<nextn-draft-tokens>",
    }
    for option, replacement in replacements.items():
        indexes = [index for index, value in enumerate(arguments) if value == option]
        if len(indexes) != 1 or indexes[0] + 1 >= len(arguments):
            raise CampaignPlanningError(
                f"{model.id} must declare exactly one {option} option"
            )
        arguments[indexes[0] + 1] = replacement
    projection["args"] = arguments
    return projection


def validate_campaign(
    definition: CampaignDefinition,
) -> tuple[CampaignPreview, dict[str, ModelSpec], SuiteSpec]:
    models = load_models(definition.models_path)
    suite = load_suite(definition.suite_path)
    if suite.id != EXPECTED_SUITE_ID:
        raise CampaignPlanningError(
            f"campaign suite must be {EXPECTED_SUITE_ID!r}"
        )
    if tuple(case.id for case in suite.cases) != EXPECTED_CASE_IDS:
        raise CampaignPlanningError("campaign suite case topology changed")
    if any(case.concurrency != 1 or case.temperature != 0.0 for case in suite.cases):
        raise CampaignPlanningError("campaign suite must remain C1 at temperature zero")
    try:
        baseline = models[definition.baseline_id]
    except KeyError as error:
        raise CampaignPlanningError("campaign baseline profile is unknown") from error
    try:
        validate_benchmark_selection(baseline, suite, context="autoresearch")
    except ManifestError as error:
        raise CampaignPlanningError(str(error)) from error

    baseline_config = semantic_config(baseline)
    invariant = canonical_json(_invariant_model_projection(baseline))
    proposals: list[ValidatedProposal] = []
    for proposal in definition.candidates:
        try:
            candidate = models[proposal.id]
        except KeyError as error:
            raise CampaignPlanningError(
                f"unknown campaign candidate {proposal.id!r}"
            ) from error
        try:
            validate_benchmark_selection(candidate, suite, context="autoresearch")
        except ManifestError as error:
            raise CampaignPlanningError(str(error)) from error
        if canonical_json(_invariant_model_projection(candidate)) != invariant:
            raise CampaignPlanningError(
                f"candidate {candidate.id!r} changes a non-axis model field"
            )
        delta = validate_one_axis_delta(
            baseline_config,
            semantic_config(candidate),
            allowed_axes=definition.allowed_axes,
        )
        if delta.axis != proposal.axis:
            raise CampaignPlanningError(
                f"candidate {candidate.id!r} changes {delta.axis!r}, not "
                f"declared axis {proposal.axis!r}"
            )
        proposals.append(
            ValidatedProposal(
                candidate_id=candidate.id,
                axis=proposal.axis,
                delta=delta,
            )
        )

    policy = CampaignPolicy(
        primary_case_ids=definition.primary_case_ids,
        allowed_axes=definition.allowed_axes,
    )
    preview = CampaignPreview(
        definition=definition,
        policy=policy,
        proposals=tuple(proposals),
        suite_id=suite.id,
    )
    return preview, models, suite


def preview_campaign(path: str | Path, *, workspace: Path) -> CampaignPreview:
    definition = load_campaign_definition(path, workspace=workspace)
    preview, _models, _suite = validate_campaign(definition)
    return preview


def _cell_specs(preview: CampaignPreview) -> tuple[dict[str, str], ...]:
    baseline = preview.definition.baseline_id
    cells: list[dict[str, str]] = [
        {
            "cell_id": "calibration-control-a",
            "stage": "calibration",
            "candidate_id": "control",
            "arm": "control_a",
            "profile_id": baseline,
        },
        {
            "cell_id": "calibration-control-b",
            "stage": "calibration",
            "candidate_id": "control",
            "arm": "control_b",
            "profile_id": baseline,
        },
    ]
    for proposal in preview.proposals:
        for stage, arms in (
            ("screen", (("champion", baseline), ("candidate", proposal.candidate_id))),
            (
                "confirmation",
                (("candidate", proposal.candidate_id), ("champion", baseline)),
            ),
        ):
            for arm, profile_id in arms:
                cells.append(
                    {
                        "cell_id": f"{proposal.candidate_id}-{stage}-{arm}",
                        "stage": stage,
                        "candidate_id": proposal.candidate_id,
                        "arm": arm,
                        "profile_id": profile_id,
                    }
                )
    return tuple(cells)


def freeze_campaign(
    path: str | Path,
    *,
    workspace: Path,
    results_root: Path,
    create_plan_fn: Callable[..., Path] = create_plan,
) -> Path:
    """Freeze every fresh cell plan without executing any of them."""

    definition = load_campaign_definition(path, workspace=workspace)
    preview, models, suite = validate_campaign(definition)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    campaign_dir = results_root / f"{stamp}-{definition.id}-{preview.digest[:8]}"
    campaign_dir.mkdir(parents=True, exist_ok=False)
    frozen_cells: list[dict[str, Any]] = []
    for ordinal, cell in enumerate(_cell_specs(preview), start=1):
        cell_root = campaign_dir / "cells" / f"{ordinal:02d}-{cell['cell_id']}"
        cell_root.mkdir(parents=True, exist_ok=False)
        run_dir = create_plan_fn(
            model=models[cell["profile_id"]],
            suite=suite,
            results_root=cell_root,
            models_path=definition.models_path,
            suite_path=definition.suite_path,
        )
        plan = json.loads((run_dir / "plan.json").read_text())
        frozen_cells.append(
            {
                **cell,
                "ordinal": ordinal,
                "run_dir": str(run_dir.relative_to(campaign_dir)),
                "plan_fingerprint": plan["fingerprint"],
                "plan_integrity_hash": plan.get("integrity_hash"),
            }
        )
    frozen = {
        "schema_version": FROZEN_CAMPAIGN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "preview": preview.to_mapping(),
        "preview_digest": preview.digest,
        "cells": frozen_cells,
        "execution_started": False,
    }
    frozen["integrity_hash"] = content_hash(frozen, 64)
    write_json(campaign_dir / "campaign.json", frozen)
    return campaign_dir


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CellProjectionError(f"JSON object contains duplicate key {key!r}")
        value[key] = item
    return value


def _read_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(), object_pairs_hook=_reject_duplicate_pairs
        )
    except (OSError, json.JSONDecodeError) as error:
        raise CellProjectionError(f"{context} is not readable canonical JSON") from error
    if not isinstance(value, dict):
        raise CellProjectionError(f"{context} must be a JSON object")
    return value


def _read_jsonl(path: Path, *, context: str) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise CellProjectionError(f"{context} is not readable") from error
    if not lines:
        raise CellProjectionError(f"{context} must not be empty")
    events: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
        except json.JSONDecodeError as error:
            raise CellProjectionError(
                f"{context} line {index + 1} is invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise CellProjectionError(
                f"{context} line {index + 1} must be an object"
            )
        events.append(value)
    return tuple(events)


def _finite_positive(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CellProjectionError(f"{context} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise CellProjectionError(f"{context} must be finite and positive")
    return parsed


def _finite_nonnegative(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CellProjectionError(f"{context} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise CellProjectionError(f"{context} must be finite and nonnegative")
    return parsed


def _one_event(
    events: tuple[dict[str, Any], ...], name: str
) -> dict[str, Any]:
    matches = tuple(event for event in events if event.get("event") == name)
    if len(matches) != 1:
        raise CellProjectionError(
            f"cell must contain exactly one {name!r} event, found {len(matches)}"
        )
    return matches[0]


def _validate_plan_integrity(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != 2:
        raise CellProjectionError("cell plan must use frozen schema version 2")
    integrity_hash = plan.get("integrity_hash")
    if (
        not isinstance(integrity_hash, str)
        or len(integrity_hash) != 64
        or any(character not in "0123456789abcdef" for character in integrity_hash)
    ):
        raise CellProjectionError("cell plan has no full integrity hash")
    payload = {key: value for key, value in plan.items() if key != "integrity_hash"}
    if content_hash(payload, 64) != integrity_hash:
        raise CellProjectionError("cell plan integrity hash does not match")
    suite = plan.get("suite")
    model = plan.get("model")
    resolved = plan.get("resolved")
    if not isinstance(suite, dict) or not isinstance(model, dict) or not isinstance(
        resolved, dict
    ):
        raise CellProjectionError("cell plan has invalid model, suite, or resolution")
    cases = suite.get("cases")
    if not isinstance(cases, list):
        raise CellProjectionError("cell plan suite cases must be an array")
    suite_without_case_ids = {
        **suite,
        "cases": [
            {key: value for key, value in case.items() if key != "case_id"}
            for case in cases
            if isinstance(case, dict)
        ],
    }
    if len(suite_without_case_ids["cases"]) != len(cases):
        raise CellProjectionError("cell plan contains a non-object case")
    expected = content_hash(
        {"model": model, "suite": suite_without_case_ids, "resolved": resolved}
    )
    if plan.get("fingerprint") != expected:
        raise CellProjectionError("cell plan fingerprint does not match")


def _normalized_flags(model: Mapping[str, Any]) -> tuple[str, ...]:
    raw = model.get("args")
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise CellProjectionError("cell plan model args must be a string array")
    arguments = list(raw)
    indexes = [
        index for index, argument in enumerate(arguments) if argument == "--served-model-name"
    ]
    if len(indexes) != 1 or indexes[0] + 1 >= len(arguments):
        raise CellProjectionError("cell plan has invalid served-model-name args")
    arguments[indexes[0] + 1] = "<served-name>"
    return tuple(arguments)


def _expected_nextn_depth(model: Mapping[str, Any]) -> int:
    raw = model.get("args")
    if not isinstance(raw, list):
        raise CellProjectionError("cell plan model args must be an array")
    indexes = [
        index
        for index, argument in enumerate(raw)
        if argument == "--speculative-num-steps"
    ]
    if len(indexes) != 1 or indexes[0] + 1 >= len(raw):
        raise CellProjectionError("cell plan must declare one NEXTN depth")
    try:
        depth = int(raw[indexes[0] + 1])
    except (TypeError, ValueError) as error:
        raise CellProjectionError("cell plan NEXTN depth is invalid") from error
    if depth <= 0:
        raise CellProjectionError("cell plan NEXTN depth must be positive")
    return depth


def _validate_native_audit(event: Mapping[str, Any], *, expected_depth: int) -> None:
    metrics = event.get("metrics")
    if not isinstance(metrics, dict):
        raise CellProjectionError("SGLang NEXTN audit metrics are missing")
    positions = metrics.get("accepted_tokens_per_position")
    expected_positions = {str(index) for index in range(expected_depth)}
    try:
        num_drafts = _finite_positive(
            metrics.get("num_drafts"), context="audit.num_drafts"
        )
        num_accepted = _finite_positive(
            metrics.get("num_accepted_tokens"),
            context="audit.num_accepted_tokens",
        )
    except CellProjectionError as error:
        raise CellProjectionError(
            "SGLang NEXTN audit counters are invalid", failure_kind="audit"
        ) from error
    if (
        metrics.get("requested") is not True
        or metrics.get("method") != "NEXTN"
        or metrics.get("configured_max_draft_tokens") != expected_depth
        or not isinstance(positions, dict)
        or set(positions) != expected_positions
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in positions.values()
        )
        or num_drafts <= 0
        or num_accepted <= 0
        or sum(positions.values()) != num_accepted
    ):
        raise CellProjectionError(
            "SGLang NEXTN audit does not prove the configured active depth",
            failure_kind="audit",
        )


def _telemetry_extrema(path: Path) -> tuple[float, float]:
    samples = _read_jsonl(path, context="cell telemetry")
    memavailable: list[int] = []
    swap_used: list[int] = []
    swap_total: int | None = None
    for index, sample in enumerate(samples):
        values: dict[str, int] = {}
        for key in ("memavailable_kib", "swaptotal_kib", "swapfree_kib"):
            raw = sample.get(key)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise CellProjectionError(
                    f"cell telemetry sample {index + 1} has invalid {key}",
                    failure_kind="measurement",
                )
            values[key] = raw
        if values["swapfree_kib"] > values["swaptotal_kib"]:
            raise CellProjectionError(
                "cell telemetry reports SwapFree above SwapTotal",
                failure_kind="measurement",
            )
        if swap_total is None:
            swap_total = values["swaptotal_kib"]
        elif values["swaptotal_kib"] != swap_total:
            raise CellProjectionError(
                "cell telemetry SwapTotal changed during the run",
                failure_kind="measurement",
            )
        memavailable.append(values["memavailable_kib"])
        swap_used.append(values["swaptotal_kib"] - values["swapfree_kib"])
    baseline_swap = swap_used[0]
    if baseline_swap > HOST_SAFETY_MAX_STARTING_SWAP_KIB:
        raise CellProjectionError(
            "cell began above the clean-start swap admission",
            failure_kind="swap_pressure",
        )
    minimum_memory = min(memavailable)
    growth = max(swap_used) - baseline_swap
    if minimum_memory < HOST_SAFETY_MIN_MEMAVAILABLE_KIB:
        raise CellProjectionError(
            "cell crossed the runtime MemAvailable floor",
            failure_kind="memory_pressure",
        )
    if growth > HOST_SAFETY_MAX_SWAP_GROWTH_KIB:
        raise CellProjectionError(
            "cell crossed the runtime swap-growth limit",
            failure_kind="swap_pressure",
        )
    return minimum_memory / 1024**2, growth / 1024


def _case_measurement(stable_id: str, case: Mapping[str, Any]) -> CaseMeasurement:
    if stable_id.startswith("agentic-"):
        speed_value = _finite_positive(
            case.get("median_agentic_task_wall_s"),
            context=f"{stable_id}.median_agentic_task_wall_s",
        )
        ttft = _finite_positive(
            case.get("median_agentic_first_turn_ttft_s"),
            context=f"{stable_id}.median_agentic_first_turn_ttft_s",
        )
        direction = "lower"
    elif stable_id == "long-context-needle-60000-agent-c1":
        speed_value = _finite_positive(
            case.get("median_e2e_s"), context=f"{stable_id}.median_e2e_s"
        )
        ttft = _finite_positive(
            case.get("median_ttft_s"), context=f"{stable_id}.median_ttft_s"
        )
        direction = "lower"
    elif stable_id == "agent64k-decode-256-c1-v1":
        speed_value = _finite_positive(
            case.get("aggregate_output_tps"),
            context=f"{stable_id}.aggregate_output_tps",
        )
        ttft = _finite_positive(
            case.get("median_ttft_s"), context=f"{stable_id}.median_ttft_s"
        )
        direction = "higher"
    else:
        raise CellProjectionError(f"unsupported primary case {stable_id!r}")
    return CaseMeasurement(stable_id, speed_value, direction, ttft)


def project_completed_cell(run_dir: Path) -> CellProjection:
    """Project one complete raw run to the strict scalar campaign contract."""

    plan = _read_json_object(run_dir / "plan.json", context="cell plan")
    _validate_plan_integrity(plan)
    summary = _read_json_object(run_dir / "summary.json", context="cell summary")
    events = _read_jsonl(run_dir / "events.jsonl", context="cell event journal")
    model = plan["model"]
    suite = plan["suite"]
    if not isinstance(model, dict) or not isinstance(suite, dict):
        raise CellProjectionError("cell plan model and suite must be objects")
    if model.get("backend") != "sglang" or suite.get("id") != EXPECTED_SUITE_ID:
        raise CellProjectionError("cell does not use the frozen SGLang campaign")
    profile_id = model.get("id")
    if not isinstance(profile_id, str):
        raise CellProjectionError("cell profile ID is missing")

    if summary.get("status") != "complete" or summary.get("completed_cases") != 9:
        raise CellProjectionError("cell summary is not a complete nine-case run")
    for field in (
        "failed_cases",
        "validation_failed_cases",
        "measurement_invalid_cases",
        "unsupported_cases",
        "unimplemented_cases",
        "context_limited_cases",
        "measurement_annotations",
        "startup_measurement_annotations",
    ):
        if summary.get(field) != []:
            raise CellProjectionError(f"cell summary {field} must be an empty array")
    if summary.get("startup_measurement_valid") is not True:
        raise CellProjectionError("cell startup measurement is invalid")

    run_start = _one_event(events, "run_start")
    if run_start.get("completed_cases_at_resume") != []:
        raise CellProjectionError("campaign cells must be fresh, not resumed")
    _one_event(events, "server_ready")
    measurement_complete = _one_event(events, "measurement_complete")
    server_stopped = _one_event(events, "server_stopped")
    _one_event(events, "run_complete")
    forbidden = {
        "run_aborted",
        "case_failed",
        "cleanup_failed",
        "server_kept",
        "lifecycle_recovery",
    }
    if any(event.get("event") in forbidden for event in events):
        raise CellProjectionError("cell journal contains a failure or recovery event")
    audit = _one_event(events, "sglang_spec_decode_metrics_snapshot")
    _validate_native_audit(audit, expected_depth=_expected_nextn_depth(model))

    plan_cases = suite.get("cases")
    summary_cases = summary.get("cases")
    if not isinstance(plan_cases, list) or not isinstance(summary_cases, list):
        raise CellProjectionError("cell plan and summary cases must be arrays")
    if len(plan_cases) != 9 or len(summary_cases) != 9:
        raise CellProjectionError("cell plan and summary must each contain nine cases")
    stable_to_frozen: dict[str, str] = {}
    for case in plan_cases:
        if not isinstance(case, dict):
            raise CellProjectionError("cell plan case must be an object")
        stable_id = case.get("id")
        frozen_id = case.get("case_id")
        if not isinstance(stable_id, str) or not isinstance(frozen_id, str):
            raise CellProjectionError("cell plan case identity is invalid")
        stable_to_frozen[stable_id] = frozen_id
    if tuple(stable_to_frozen) != EXPECTED_CASE_IDS:
        raise CellProjectionError("cell plan case order or identity changed")
    frozen_summaries: dict[str, dict[str, Any]] = {}
    for case in summary_cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise CellProjectionError("cell summary case identity is invalid")
        frozen_id = str(case["case_id"])
        if frozen_id in frozen_summaries:
            raise CellProjectionError("cell summary has duplicate case identity")
        frozen_summaries[frozen_id] = case
    if set(frozen_summaries) != set(stable_to_frozen.values()):
        raise CellProjectionError("cell summary cases do not match the frozen plan")
    for stable_id, frozen_id in stable_to_frozen.items():
        case = frozen_summaries[frozen_id]
        if case.get("measurement_valid") is not True:
            raise CellProjectionError(f"{stable_id} measurement is invalid")
        if case.get("validation_passed") is not True:
            raise CellProjectionError(f"{stable_id} validation did not pass")

    measurements = tuple(
        _case_measurement(stable_id, frozen_summaries[stable_to_frozen[stable_id]])
        for stable_id in EXPECTED_PRIMARY_CASE_IDS
    )
    minimum_memory, swap_growth = _telemetry_extrema(run_dir / "telemetry.jsonl")
    return CellProjection(
        profile_id=profile_id,
        plan_fingerprint=str(plan["fingerprint"]),
        measurements=measurements,
        measurement_elapsed_s=_finite_nonnegative(
            measurement_complete.get("elapsed_s"),
            context="measurement_complete.elapsed_s",
        ),
        cleanup_elapsed_s=_finite_nonnegative(
            server_stopped.get("cleanup_elapsed_s"),
            context="server_stopped.cleanup_elapsed_s",
        ),
        minimum_memavailable_gib=minimum_memory,
        maximum_swap_growth_mib=swap_growth,
        normalized_flags=_normalized_flags(model),
    )


def observation_from_cells(
    policy: CampaignPolicy,
    *,
    pair_index: int,
    champion: CellProjection,
    candidate: CellProjection,
    audit_reserve_remaining_s: float,
) -> PairObservation:
    """Build one role-normalized score; execution order never changes ratios."""

    ratios: list[float] = []
    ttft_ratios: list[float] = []
    for case_id in policy.primary_case_ids:
        champion_metric = champion.measurement(case_id)
        candidate_metric = candidate.measurement(case_id)
        if champion_metric.speed_direction != candidate_metric.speed_direction:
            raise CellProjectionError(f"{case_id} metric direction changed")
        if champion_metric.speed_direction == "lower":
            ratios.append(champion_metric.speed_value / candidate_metric.speed_value)
        elif champion_metric.speed_direction == "higher":
            ratios.append(candidate_metric.speed_value / champion_metric.speed_value)
        else:
            raise CellProjectionError(f"{case_id} has an unknown metric direction")
        ttft_ratios.append(candidate_metric.ttft_s / champion_metric.ttft_s)

    elapsed_by_role = {
        "champion": champion.measurement_elapsed_s,
        "candidate": candidate.measurement_elapsed_s,
    }
    order = (
        ("champion", "candidate")
        if pair_index % 2 == 0
        else ("candidate", "champion")
    )
    return PairObservation(
        pair_index=pair_index,
        primary_case_ids=policy.primary_case_ids,
        primary_speed_ratios=tuple(ratios),
        median_ttft_ratio=statistics.median(ttft_ratios),
        timing=TimingInputs(
            cell_elapsed_s=tuple(elapsed_by_role[role] for role in order),  # type: ignore[arg-type]
            pair_elapsed_s=(
                champion.measurement_elapsed_s + candidate.measurement_elapsed_s
            ),
            cleanup_elapsed_s=max(
                champion.cleanup_elapsed_s, candidate.cleanup_elapsed_s
            ),
            audit_reserve_remaining_s=_finite_nonnegative(
                audit_reserve_remaining_s, context="audit_reserve_remaining_s"
            ),
        ),
        eligibility=EligibilityInputs(
            cells_completed=True,
            measurement_valid=True,
            validation_passed=True,
            workload_matched=True,
            artifact_identity_verified=True,
            audit_requirement_passed=True,
            cleanup_verified=True,
            memory_pressure=False,
            swap_pressure=False,
            oom=False,
            ownership_ambiguous=False,
            cleanup_breach=False,
        ),
        simplification=SimplificationEvidence(
            minimum_memavailable_gain_gib=(
                candidate.minimum_memavailable_gib
                - champion.minimum_memavailable_gib
            ),
            champion_flags=champion.normalized_flags,
            candidate_flags=candidate.normalized_flags,
        ),
    )
