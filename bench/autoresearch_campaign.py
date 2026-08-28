"""Strict planning primitives for the single-user Qwen autoresearch campaign.

The campaign planner is deliberately separate from the execution controller.
It validates a finite queue of semantic one-axis candidates, constructs the
immutable policy, and can freeze fresh SparkBench plans for calibration,
screening, and reverse-order confirmation.  Planning never executes a plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import tomllib
from typing import Any, Callable, Mapping

from .autoresearch import (
    CampaignPolicy,
    CandidateDelta,
    EligibilityInputs,
    PairObservation,
    SimplificationEvidence,
    TimingInputs,
    append_transition,
    evaluate_calibration,
    evaluate_promotion,
    evaluate_screen,
    pair_order,
    replay_transitions,
    validate_one_axis_delta,
)
from .host_safety import parse_meminfo, read_host_meminfo
from .journal import Journal, canonical_json, content_hash, utc_now, write_json
from .manifest import (
    ManifestError,
    ModelSpec,
    SuiteSpec,
    load_models,
    load_suite,
    validate_benchmark_selection,
)
from .runner import create_plan
from .runtime import recover_owned_sglang


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
PAIR_ADMISSION_REMAINING_S = 4_620
EXPECTED_HOST_SAFETY_THRESHOLDS = (14, 512, 64)

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


@dataclass(frozen=True, slots=True)
class FrozenCell:
    cell_id: str
    stage: str
    candidate_id: str
    arm: str
    profile_id: str
    ordinal: int
    run_dir: Path
    plan_fingerprint: str
    plan_integrity_hash: str


@dataclass(frozen=True, slots=True)
class FrozenCampaign:
    campaign_dir: Path
    campaign_id: str
    cutoff: datetime
    baseline_id: str
    policy: CampaignPolicy
    policy_digest: str
    proposals: tuple[ValidatedProposal, ...]
    cells: tuple[FrozenCell, ...]

    def cells_for(self, *, candidate_id: str, stage: str) -> dict[str, FrozenCell]:
        matches = {
            cell.arm: cell
            for cell in self.cells
            if cell.candidate_id == candidate_id and cell.stage == stage
        }
        expected = (
            {"control_a", "control_b"}
            if stage == "calibration"
            else {"champion", "candidate"}
        )
        if set(matches) != expected:
            raise CampaignPlanningError(
                f"frozen {stage} cells for {candidate_id!r} are incomplete"
            )
        return matches


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
    baseline_safety = (
        baseline.host_safety_min_memavailable_gib,
        baseline.host_safety_max_swap_growth_mib,
        baseline.host_safety_max_starting_swap_mib,
    )
    if baseline_safety != EXPECTED_HOST_SAFETY_THRESHOLDS:
        raise CampaignPlanningError(
            "campaign profiles must freeze the 14 GiB, 512 MiB growth, and "
            "64 MiB starting-swap host-safety gates"
        )

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


def _require_safe_relative_path(
    base: Path, raw: Any, *, context: str
) -> Path:
    text = _require_string(raw, context=context)
    relative = Path(text)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise CampaignPlanningError(f"{context} must be a safe relative path")
    try:
        resolved = (base / relative).resolve(strict=True)
        resolved.relative_to(base.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise CampaignPlanningError(f"{context} escapes the campaign directory") from error
    if not resolved.is_dir():
        raise CampaignPlanningError(f"{context} must resolve to a directory")
    return resolved


def load_frozen_campaign(campaign_dir: Path) -> FrozenCampaign:
    root = campaign_dir.resolve(strict=True)
    frozen = _read_json_object(root / "campaign.json", context="frozen campaign")
    expected_top = {
        "schema_version",
        "created_at",
        "preview",
        "preview_digest",
        "cells",
        "execution_started",
        "integrity_hash",
    }
    if set(frozen) != expected_top:
        raise CampaignPlanningError("frozen campaign has an unknown or missing field")
    if frozen["schema_version"] != FROZEN_CAMPAIGN_SCHEMA_VERSION:
        raise CampaignPlanningError("unsupported frozen campaign schema version")
    integrity_hash = frozen["integrity_hash"]
    if not isinstance(integrity_hash, str) or len(integrity_hash) != 64:
        raise CampaignPlanningError("frozen campaign integrity hash is invalid")
    payload = {key: value for key, value in frozen.items() if key != "integrity_hash"}
    if content_hash(payload, 64) != integrity_hash:
        raise CampaignPlanningError("frozen campaign integrity hash does not match")
    preview = frozen["preview"]
    if not isinstance(preview, dict):
        raise CampaignPlanningError("frozen campaign preview must be an object")
    if content_hash(preview, 64) != frozen["preview_digest"]:
        raise CampaignPlanningError("frozen campaign preview digest does not match")
    required_preview = {
        "schema_version",
        "campaign_id",
        "cutoff",
        "baseline_id",
        "suite_id",
        "policy",
        "policy_digest",
        "proposals",
        "planned_cell_count",
        "execution_started",
    }
    if set(preview) != required_preview:
        raise CampaignPlanningError("frozen campaign preview topology changed")
    if preview["schema_version"] != CAMPAIGN_SCHEMA_VERSION:
        raise CampaignPlanningError("frozen campaign preview schema is unsupported")
    if (
        preview["planned_cell_count"] != 14
        or preview["execution_started"] is not False
        or frozen["execution_started"] is not False
    ):
        raise CampaignPlanningError("frozen campaign execution topology changed")
    if preview["suite_id"] != EXPECTED_SUITE_ID:
        raise CampaignPlanningError("frozen campaign suite identity changed")
    policy = CampaignPolicy.from_mapping(preview["policy"])
    if policy.digest != preview["policy_digest"]:
        raise CampaignPlanningError("frozen campaign policy digest does not match")
    cutoff = datetime.fromisoformat(
        _require_string(preview["cutoff"], context="frozen cutoff")
    )
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise CampaignPlanningError("frozen campaign cutoff must be timezone-aware")
    proposals_value = preview["proposals"]
    if not isinstance(proposals_value, list) or not proposals_value:
        raise CampaignPlanningError("frozen campaign proposals must be an array")
    proposals: list[ValidatedProposal] = []
    for index, raw in enumerate(proposals_value):
        if not isinstance(raw, dict) or set(raw) != {
            "candidate_id",
            "axis",
            "delta",
            "delta_digest",
        }:
            raise CampaignPlanningError(
                f"frozen campaign proposal {index} topology changed"
            )
        delta_value = raw["delta"]
        if not isinstance(delta_value, dict):
            raise CampaignPlanningError("frozen candidate delta must be an object")
        try:
            delta = CandidateDelta(**delta_value)
        except TypeError as error:
            raise CampaignPlanningError("frozen candidate delta is malformed") from error
        if delta.digest != raw["delta_digest"] or delta.axis != raw["axis"]:
            raise CampaignPlanningError("frozen candidate delta digest does not match")
        proposals.append(
            ValidatedProposal(
                candidate_id=_require_string(
                    raw["candidate_id"], context="frozen candidate ID"
                ),
                axis=_require_string(raw["axis"], context="frozen candidate axis"),
                delta=delta,
            )
        )
    if tuple(proposal.axis for proposal in proposals) != EXPECTED_AXES:
        raise CampaignPlanningError("frozen candidate order changed")

    cells_value = frozen["cells"]
    if not isinstance(cells_value, list) or len(cells_value) != 14:
        raise CampaignPlanningError("frozen campaign must contain fourteen cells")
    cell_keys = {
        "cell_id",
        "stage",
        "candidate_id",
        "arm",
        "profile_id",
        "ordinal",
        "run_dir",
        "plan_fingerprint",
        "plan_integrity_hash",
    }
    cells: list[FrozenCell] = []
    expected_cells: list[dict[str, str]] = [
        {
            "cell_id": "calibration-control-a",
            "stage": "calibration",
            "candidate_id": "control",
            "arm": "control_a",
            "profile_id": str(preview["baseline_id"]),
        },
        {
            "cell_id": "calibration-control-b",
            "stage": "calibration",
            "candidate_id": "control",
            "arm": "control_b",
            "profile_id": str(preview["baseline_id"]),
        },
    ]
    for proposal in proposals:
        expected_cells.extend(
            (
                {
                    "cell_id": f"{proposal.candidate_id}-screen-champion",
                    "stage": "screen",
                    "candidate_id": proposal.candidate_id,
                    "arm": "champion",
                    "profile_id": str(preview["baseline_id"]),
                },
                {
                    "cell_id": f"{proposal.candidate_id}-screen-candidate",
                    "stage": "screen",
                    "candidate_id": proposal.candidate_id,
                    "arm": "candidate",
                    "profile_id": proposal.candidate_id,
                },
                {
                    "cell_id": f"{proposal.candidate_id}-confirmation-candidate",
                    "stage": "confirmation",
                    "candidate_id": proposal.candidate_id,
                    "arm": "candidate",
                    "profile_id": proposal.candidate_id,
                },
                {
                    "cell_id": f"{proposal.candidate_id}-confirmation-champion",
                    "stage": "confirmation",
                    "candidate_id": proposal.candidate_id,
                    "arm": "champion",
                    "profile_id": str(preview["baseline_id"]),
                },
            )
        )
    for index, raw in enumerate(cells_value, start=1):
        if not isinstance(raw, dict) or set(raw) != cell_keys:
            raise CampaignPlanningError(f"frozen cell {index} topology changed")
        if raw["ordinal"] != index:
            raise CampaignPlanningError("frozen cell ordinals are not contiguous")
        expected_identity = expected_cells[index - 1]
        if any(raw.get(key) != value for key, value in expected_identity.items()):
            raise CampaignPlanningError("frozen cell schedule or profile binding changed")
        run_dir = _require_safe_relative_path(
            root, raw["run_dir"], context=f"frozen cell {index} run_dir"
        )
        plan = _read_json_object(run_dir / "plan.json", context="frozen cell plan")
        _validate_plan_integrity(plan)
        if (
            plan.get("fingerprint") != raw["plan_fingerprint"]
            or plan.get("integrity_hash") != raw["plan_integrity_hash"]
        ):
            raise CampaignPlanningError("frozen cell plan binding changed")
        model = plan.get("model")
        if not isinstance(model, dict) or model.get("id") != raw["profile_id"]:
            raise CampaignPlanningError("frozen cell profile binding changed")
        if (
            model.get("host_safety_min_memavailable_gib"),
            model.get("host_safety_max_swap_growth_mib"),
            model.get("host_safety_max_starting_swap_mib"),
        ) != EXPECTED_HOST_SAFETY_THRESHOLDS:
            raise CampaignPlanningError("frozen cell host-safety gates changed")
        cells.append(
            FrozenCell(
                cell_id=_require_string(raw["cell_id"], context="frozen cell ID"),
                stage=_require_string(raw["stage"], context="frozen cell stage"),
                candidate_id=_require_string(
                    raw["candidate_id"], context="frozen cell candidate"
                ),
                arm=_require_string(raw["arm"], context="frozen cell arm"),
                profile_id=_require_string(
                    raw["profile_id"], context="frozen cell profile"
                ),
                ordinal=index,
                run_dir=run_dir,
                plan_fingerprint=str(raw["plan_fingerprint"]),
                plan_integrity_hash=str(raw["plan_integrity_hash"]),
            )
        )
    if len({cell.cell_id for cell in cells}) != 14:
        raise CampaignPlanningError("frozen cell IDs must be unique")
    campaign = FrozenCampaign(
        campaign_dir=root,
        campaign_id=_require_string(
            preview["campaign_id"], context="frozen campaign ID"
        ),
        cutoff=cutoff,
        baseline_id=_require_string(
            preview["baseline_id"], context="frozen baseline ID"
        ),
        policy=policy,
        policy_digest=policy.digest,
        proposals=tuple(proposals),
        cells=tuple(cells),
    )
    campaign.cells_for(candidate_id="control", stage="calibration")
    for proposal in campaign.proposals:
        campaign.cells_for(candidate_id=proposal.candidate_id, stage="screen")
        campaign.cells_for(
            candidate_id=proposal.candidate_id, stage="confirmation"
        )
    return campaign


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


def _terminal_failure_kind(
    events: tuple[dict[str, Any], ...], summary: Mapping[str, Any]
) -> str | None:
    """Classify durable terminal artifacts before rejecting an incomplete cell."""

    if any(event.get("event") == "cleanup_failed" for event in events):
        return "cleanup_breach"
    breaches = tuple(
        event for event in events if event.get("event") == "host_safety_breach"
    )
    if breaches:
        code = breaches[-1].get("code")
        if code == "memavailable_below_minimum":
            return "memory_pressure"
        if code in {
            "starting_swap_above_maximum",
            "swap_growth_above_maximum",
            "swap_total_changed",
        }:
            return "swap_pressure"
        return "measurement"
    if any(
        event.get("event") == "run_aborted"
        and event.get("stage") == "sglang_speculative_acceptance_audit"
        for event in events
    ):
        return "audit"
    validation_failures = summary.get("validation_failed_cases")
    if isinstance(validation_failures, list) and validation_failures:
        return "validation"
    if any(
        event.get("event") == "case_complete"
        and event.get("validation_passed") is False
        for event in events
    ):
        return "validation"
    return None


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

    terminal_failure = _terminal_failure_kind(events, summary)
    if terminal_failure is not None:
        raise CellProjectionError(
            "cell contains a terminal failure artifact",
            failure_kind=terminal_failure,
        )
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


def _aware_now() -> datetime:
    return datetime.now(timezone.utc)


def campaign_admission(
    campaign: FrozenCampaign,
    *,
    now: datetime | None = None,
    meminfo_reader: Callable[[], str] = read_host_meminfo,
) -> tuple[str, ...]:
    """Return scalar pre-start blockers without starting an inference process."""

    current = now or _aware_now()
    if current.tzinfo is None or current.utcoffset() is None:
        raise CampaignPlanningError("campaign admission time must be timezone-aware")
    blockers: list[str] = []
    remaining_s = (campaign.cutoff - current).total_seconds()
    if remaining_s < PAIR_ADMISSION_REMAINING_S:
        blockers.append("insufficient_time_for_pair")
    try:
        sample = parse_meminfo(meminfo_reader())
    except (OSError, ValueError) as error:
        raise CampaignPlanningError("campaign host-memory admission failed closed") from error
    if sample.swap_used_kib > HOST_SAFETY_MAX_STARTING_SWAP_KIB:
        blockers.append("starting_swap_above_clean_limit")
    calibration = campaign.cells_for(
        candidate_id="control", stage="calibration"
    )
    plan = _read_json_object(
        calibration["control_a"].run_dir / "plan.json",
        context="calibration control plan",
    )
    model = plan.get("model")
    if not isinstance(model, dict):
        raise CampaignPlanningError("calibration control model is malformed")
    estimated = model.get("estimated_ram_gib")
    if isinstance(estimated, bool) or not isinstance(estimated, (int, float)):
        raise CampaignPlanningError("calibration control RAM estimate is missing")
    required_kib = int((float(estimated) + 8.0) * 1024**2)
    if sample.memavailable_kib < required_kib:
        blockers.append("insufficient_preflight_memavailable")
    return tuple(blockers)


def _cell_run_identity(cell: FrozenCell) -> str:
    return f"{cell.plan_fingerprint}-{cell.run_dir.name}"


def _recover_cell(cell: FrozenCell) -> str:
    try:
        outcome = recover_owned_sglang(
            _cell_run_identity(cell),
            api_key_path=cell.run_dir / "server" / "api-key",
        )
    except Exception as error:
        raise CellProjectionError(
            "exact owned SGLang recovery failed", failure_kind="cleanup_breach"
        ) from error
    if outcome == "different_container_present":
        raise CellProjectionError(
            "a differently owned SGLang container blocks recovery",
            failure_kind="ownership_ambiguity",
        )
    return outcome


def run_frozen_cell(
    cell: FrozenCell,
    *,
    workspace: Path,
    cell_timeout_s: int,
    cleanup_timeout_s: int,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    monotonic: Callable[[], float] = time.monotonic,
) -> CellProjection:
    """Execute one pristine plan with SIGINT unwind and exact owned recovery."""

    events_path = cell.run_dir / "events.jsonl"
    if events_path.exists() and events_path.stat().st_size:
        try:
            projection = project_completed_cell(cell.run_dir)
        except CellProjectionError as error:
            _recover_cell(cell)
            raise CellProjectionError(
                "a previously started cell is not scoreable and will not be resumed",
                failure_kind=error.failure_kind,
            ) from error
        if projection.profile_id != cell.profile_id:
            raise CellProjectionError("completed cell profile binding changed")
        return projection

    command = [
        sys.executable,
        str((workspace / "sparkbench.py").resolve(strict=True)),
        "run",
        str(cell.run_dir),
        "--fail-fast",
    ]
    log_path = cell.run_dir / "controller.log"
    started = monotonic()
    timed_out = False
    forced_kill = False
    with log_path.open("a", encoding="utf-8") as log:
        process = popen_factory(
            command,
            cwd=str(workspace),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=cell_timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGINT)
            try:
                return_code = process.wait(timeout=cleanup_timeout_s)
            except subprocess.TimeoutExpired:
                forced_kill = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
                return_code = -signal.SIGKILL
    wall_s = monotonic() - started
    if wall_s < 0:
        raise CellProjectionError("cell monotonic clock moved backwards")
    if forced_kill:
        _recover_cell(cell)
        raise CellProjectionError(
            "cell exceeded timeout and owned cleanup grace",
            failure_kind="cleanup_breach",
        )
    if timed_out:
        _recover_cell(cell)
        raise CellProjectionError(
            "cell exceeded its causal timeout and is invalid",
            failure_kind="measurement",
        )
    if return_code != 0:
        _recover_cell(cell)
        try:
            project_completed_cell(cell.run_dir)
        except CellProjectionError as error:
            raise error
        raise CellProjectionError("cell process returned a nonzero status")
    projection = project_completed_cell(cell.run_dir)
    if projection.profile_id != cell.profile_id:
        raise CellProjectionError("completed cell profile binding changed")
    return projection


def _transition_id(*parts: object) -> str:
    value = "-".join(str(part).lower().replace("_", "-") for part in parts)
    if len(value) <= 128:
        return value
    return value[:111] + "-" + content_hash(value, 16)


def _append_campaign_started(
    journal: Journal, campaign: FrozenCampaign
) -> None:
    if journal.events():
        replay_transitions(campaign.policy, journal.events())
        return
    append_transition(
        journal,
        campaign.policy,
        {
            "event": "autoresearch_campaign_started",
            "transition_id": "campaign-started",
            "campaign_id": campaign.campaign_id,
            "policy_digest": campaign.policy_digest,
        },
    )


def _append_terminal(
    journal: Journal,
    campaign: FrozenCampaign,
    *,
    failure_kind: str,
    cleanup_verified: bool,
) -> None:
    state = replay_transitions(campaign.policy, journal.events())
    if state.phase == "terminal":
        return
    append_transition(
        journal,
        campaign.policy,
        {
            "event": "autoresearch_campaign_terminated",
            "transition_id": _transition_id("campaign", "terminated", failure_kind),
            "failure_kind": failure_kind,
            "cleanup_verified": cleanup_verified,
            "restored_preflight": False,
        },
    )


def _calibration_path(campaign: FrozenCampaign) -> Path:
    return campaign.campaign_dir / "calibration.json"


def _write_calibration(
    campaign: FrozenCampaign,
    observation: PairObservation,
    *,
    passed: bool,
    reasons: tuple[str, ...],
) -> None:
    record = {
        "schema_version": 1,
        "observation": observation.to_mapping(),
        "passed": passed,
        "reasons": list(reasons),
    }
    record["integrity_hash"] = content_hash(record, 64)
    write_json(_calibration_path(campaign), record)


def _load_calibration(campaign: FrozenCampaign) -> PairObservation | None:
    path = _calibration_path(campaign)
    if not path.exists():
        return None
    record = _read_json_object(path, context="calibration record")
    expected = {"schema_version", "observation", "passed", "reasons", "integrity_hash"}
    if set(record) != expected or record.get("schema_version") != 1:
        raise CampaignPlanningError("calibration record topology changed")
    integrity = record.pop("integrity_hash")
    if not isinstance(integrity, str) or content_hash(record, 64) != integrity:
        raise CampaignPlanningError("calibration record integrity hash does not match")
    observation = PairObservation.from_mapping(record["observation"])
    decision = evaluate_calibration(campaign.policy, observation)
    if record["passed"] is not decision.passed or record["reasons"] != list(
        decision.reasons
    ):
        raise CampaignPlanningError("calibration record decision does not replay")
    if not decision.passed:
        raise CellProjectionError("control-to-control calibration did not pass")
    return observation


def _remaining_s(campaign: FrozenCampaign, now: Callable[[], datetime]) -> float:
    value = now()
    if value.tzinfo is None or value.utcoffset() is None:
        raise CampaignPlanningError("campaign clock must be timezone-aware")
    return max(0.0, (campaign.cutoff - value).total_seconds())


def _run_calibration(
    campaign: FrozenCampaign,
    *,
    workspace: Path,
    cell_runner: Callable[[FrozenCell], CellProjection],
    now: Callable[[], datetime],
) -> PairObservation:
    existing = _load_calibration(campaign)
    if existing is not None:
        return existing
    if _remaining_s(campaign, now) < PAIR_ADMISSION_REMAINING_S:
        raise CellProjectionError("insufficient time remains for calibration")
    cells = campaign.cells_for(candidate_id="control", stage="calibration")
    control_a = cell_runner(cells["control_a"])
    if (
        control_a.measurement_elapsed_s > campaign.policy.cell_timeout_s
        or control_a.cleanup_elapsed_s > campaign.policy.cleanup_timeout_s
    ):
        raise CellProjectionError(
            "first calibration cell exceeded its measurement or cleanup budget"
        )
    stopped_path = cells["control_a"].run_dir / "events.jsonl"
    control_a_stopped = (
        _server_stopped_at(cells["control_a"])
        if stopped_path.exists() and stopped_path.stat().st_size
        else now()
    )
    calibration_gap_s = (now() - control_a_stopped).total_seconds()
    if calibration_gap_s < 0 or calibration_gap_s > campaign.policy.cleanup_timeout_s:
        raise CellProjectionError(
            "calibration inter-cell gap exceeded the frozen cleanup bound"
        )
    control_b = cell_runner(cells["control_b"])
    observation = observation_from_cells(
        campaign.policy,
        pair_index=0,
        champion=control_a,
        candidate=control_b,
        audit_reserve_remaining_s=_remaining_s(campaign, now),
    )
    decision = evaluate_calibration(campaign.policy, observation)
    _write_calibration(
        campaign,
        observation,
        passed=decision.passed,
        reasons=decision.reasons,
    )
    if not decision.passed:
        raise CellProjectionError("control-to-control calibration did not pass")
    return observation


def _candidate_decisions(events: tuple[dict[str, Any], ...]) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for event in events:
        if event.get("event") != "autoresearch_candidate_decided":
            continue
        candidate_id = event.get("candidate_id")
        decision = event.get("decision")
        if not isinstance(candidate_id, str) or not isinstance(decision, str):
            raise CampaignPlanningError("candidate decision journal is malformed")
        decisions[candidate_id] = decision
    return decisions


def _require_score_eligible(
    policy: CampaignPolicy, observation: PairObservation
) -> None:
    failed_gates = observation.eligibility.failed_gates
    failed_budgets = observation.timing.failed_budgets(policy)
    if not failed_gates and not failed_budgets:
        return
    failure_kind = "measurement"
    for gate, kind in (
        ("cleanup_breach", "cleanup_breach"),
        ("ownership_ambiguous", "ownership_ambiguity"),
        ("oom", "oom"),
        ("swap_pressure", "swap_pressure"),
        ("memory_pressure", "memory_pressure"),
        ("audit_requirement_passed", "audit"),
        ("validation_passed", "validation"),
    ):
        if gate in failed_gates:
            failure_kind = kind
            break
    raise CellProjectionError(
        "score-bearing pair failed a frozen eligibility or timing gate",
        failure_kind=failure_kind,
    )


def _server_stopped_at(cell: FrozenCell) -> datetime:
    event = _one_event(
        _read_jsonl(cell.run_dir / "events.jsonl", context="cell event journal"),
        "server_stopped",
    )
    raw = event.get("timestamp")
    if not isinstance(raw, str):
        raise CellProjectionError("server_stopped event has no timestamp")
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as error:
        raise CellProjectionError("server_stopped timestamp is invalid") from error
    if value.tzinfo is None or value.utcoffset() is None:
        raise CellProjectionError("server_stopped timestamp is not timezone-aware")
    return value


def _run_search_pair(
    campaign: FrozenCampaign,
    journal: Journal,
    *,
    candidate_id: str,
    stage: str,
    cell_runner: Callable[[FrozenCell], CellProjection],
    now: Callable[[], datetime],
) -> PairObservation:
    state = replay_transitions(campaign.policy, journal.events())
    if state.phase == "candidate":
        if _remaining_s(campaign, now) < PAIR_ADMISSION_REMAINING_S:
            raise CellProjectionError("insufficient time remains for a search pair")
        append_transition(
            journal,
            campaign.policy,
            {
                "event": "autoresearch_pair_started",
                "transition_id": _transition_id(
                    candidate_id, "pair", state.next_pair_index, "started"
                ),
                "candidate_id": candidate_id,
                "pair_index": state.next_pair_index,
                "order": list(pair_order(state.next_pair_index)),
            },
        )
        state = replay_transitions(campaign.policy, journal.events())
    if state.phase != "pair" or state.active_pair_index is None:
        raise CampaignPlanningError("candidate is not in an executable pair phase")
    cells = campaign.cells_for(candidate_id=candidate_id, stage=stage)
    projections: dict[str, CellProjection] = {}
    last_completion: datetime | None = None
    for completed_arm in state.completed_arms:
        projections[completed_arm] = project_completed_cell(
            cells[completed_arm].run_dir
        )
        last_completion = _server_stopped_at(cells[completed_arm])
    for arm in state.active_order[len(state.completed_arms) :]:
        if last_completion is not None:
            current = now()
            if current.tzinfo is None or current.utcoffset() is None:
                raise CampaignPlanningError("campaign clock must be timezone-aware")
            gap_s = (current - last_completion).total_seconds()
            if gap_s < 0 or gap_s > campaign.policy.cleanup_timeout_s:
                raise CellProjectionError(
                    "inter-cell gap exceeded the frozen cleanup bound"
                )
            prior = projections[state.active_order[len(projections) - 1]]
            if (
                prior.measurement_elapsed_s > campaign.policy.cell_timeout_s
                or prior.cleanup_elapsed_s > campaign.policy.cleanup_timeout_s
            ):
                raise CellProjectionError(
                    "first cell exceeded its measurement or cleanup budget"
                )
        projection = cell_runner(cells[arm])
        if projection.profile_id != cells[arm].profile_id:
            raise CellProjectionError("cell projection profile does not match schedule")
        projections[arm] = projection
        append_transition(
            journal,
            campaign.policy,
            {
                "event": "autoresearch_cell_completed",
                "transition_id": _transition_id(
                    candidate_id, "pair", state.active_pair_index, arm, "completed"
                ),
                "candidate_id": candidate_id,
                "pair_index": state.active_pair_index,
                "arm": arm,
            },
        )
        state = replay_transitions(campaign.policy, journal.events())
        last_completion = now()
    observation = observation_from_cells(
        campaign.policy,
        pair_index=state.active_pair_index,
        champion=projections["champion"],
        candidate=projections["candidate"],
        audit_reserve_remaining_s=_remaining_s(campaign, now),
    )
    append_transition(
        journal,
        campaign.policy,
        {
            "event": "autoresearch_pair_scored",
            "transition_id": _transition_id(
                candidate_id, "pair", state.active_pair_index, "scored"
            ),
            "candidate_id": candidate_id,
            "pair_index": state.active_pair_index,
            "observation": observation.to_mapping(),
        },
    )
    return observation


def summarize_campaign(campaign_dir: Path) -> dict[str, Any]:
    campaign = load_frozen_campaign(campaign_dir)
    journal = Journal(campaign.campaign_dir / "events.jsonl")
    events = tuple(journal.events())
    state = replay_transitions(campaign.policy, events) if events else None
    calibration_path = _calibration_path(campaign)
    summary = {
        "schema_version": 1,
        "campaign_id": campaign.campaign_id,
        "status": (
            "planned"
            if state is None
            else "complete"
            if state.phase == "terminal" and state.terminal_reason == "completed"
            else "terminated"
            if state.phase == "terminal"
            else "active"
        ),
        "terminal_reason": state.terminal_reason if state else None,
        "calibration_recorded": calibration_path.exists(),
        "next_pair_index": state.next_pair_index if state else 0,
        "candidate_decisions": _candidate_decisions(events),
        "policy_digest": campaign.policy_digest,
    }
    write_json(campaign.campaign_dir / "summary.json", summary)
    return summary


def run_campaign(
    campaign_dir: Path,
    *,
    workspace: Path,
    now: Callable[[], datetime] = _aware_now,
    meminfo_reader: Callable[[], str] = read_host_meminfo,
    cell_runner: Callable[[FrozenCell], CellProjection] | None = None,
) -> dict[str, Any]:
    """Run or replay the finite queue, never resuming an interrupted cell."""

    campaign = load_frozen_campaign(campaign_dir)
    lock_path = campaign.campaign_dir / ".autoresearch.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignPlanningError(
                "another autoresearch controller holds the campaign lock"
            ) from error
        existing_journal = Journal(campaign.campaign_dir / "events.jsonl")
        existing_events = existing_journal.events()
        if existing_events:
            existing_state = replay_transitions(campaign.policy, existing_events)
            if existing_state.phase == "terminal":
                return summarize_campaign(campaign.campaign_dir)
        blockers = campaign_admission(
            campaign, now=now(), meminfo_reader=meminfo_reader
        )
        if blockers:
            summary = summarize_campaign(campaign.campaign_dir)
            summary.update(
                {
                    "status": "blocked_environment",
                    "blockers": list(blockers),
                }
            )
            write_json(campaign.campaign_dir / "summary.json", summary)
            return summary

        runner = cell_runner or (
            lambda cell: run_frozen_cell(
                cell,
                workspace=workspace,
                cell_timeout_s=campaign.policy.cell_timeout_s,
                cleanup_timeout_s=campaign.policy.cleanup_timeout_s,
            )
        )
        journal = existing_journal
        _append_campaign_started(journal, campaign)
        try:
            calibration_already_recorded = _calibration_path(campaign).exists()
            _run_calibration(
                campaign,
                workspace=workspace,
                cell_runner=runner,
                now=now,
            )
            if not calibration_already_recorded:
                return summarize_campaign(campaign.campaign_dir)
            while True:
                state = replay_transitions(campaign.policy, journal.events())
                if state.phase == "terminal":
                    break
                decisions = _candidate_decisions(tuple(journal.events()))
                if any(
                    decision in {"promote", "promote_simplification"}
                    for decision in decisions.values()
                ):
                    append_transition(
                        journal,
                        campaign.policy,
                        {
                            "event": "autoresearch_campaign_completed",
                            "transition_id": "campaign-completed-after-promotion",
                        },
                    )
                    break
                if state.phase == "idle":
                    proposal = next(
                        (
                            item
                            for item in campaign.proposals
                            if item.candidate_id not in decisions
                        ),
                        None,
                    )
                    if proposal is None:
                        append_transition(
                            journal,
                            campaign.policy,
                            {
                                "event": "autoresearch_campaign_completed",
                                "transition_id": "campaign-completed-queue-exhausted",
                            },
                        )
                        break
                    append_transition(
                        journal,
                        campaign.policy,
                        {
                            "event": "autoresearch_candidate_started",
                            "transition_id": _transition_id(
                                proposal.candidate_id, "started"
                            ),
                            "candidate_id": proposal.candidate_id,
                            "axis": proposal.axis,
                            "delta_digest": proposal.delta.digest,
                        },
                    )
                    state = replay_transitions(campaign.policy, journal.events())
                if state.candidate_id is None:
                    raise CampaignPlanningError("active search state has no candidate")
                if state.phase in {"candidate", "pair"}:
                    stage = (
                        "screen"
                        if len(state.candidate_observations) == 0
                        else "confirmation"
                    )
                    _run_search_pair(
                        campaign,
                        journal,
                        candidate_id=state.candidate_id,
                        stage=stage,
                        cell_runner=runner,
                        now=now,
                    )
                    state = replay_transitions(campaign.policy, journal.events())
                if state.phase != "scored" or state.candidate_id is None:
                    raise CampaignPlanningError("search pair did not reach scored state")
                observations = state.candidate_observations
                _require_score_eligible(campaign.policy, observations[-1])
                if len(observations) == 1:
                    decision = (
                        "confirm"
                        if evaluate_screen(campaign.policy, observations[0]).passed
                        else "reject"
                    )
                elif len(observations) == 2:
                    decision = (
                        "promote"
                        if evaluate_promotion(
                            campaign.policy, observations[0], observations[1]
                        ).passed
                        else "reject"
                    )
                else:
                    raise CampaignPlanningError("candidate has an invalid score count")
                append_transition(
                    journal,
                    campaign.policy,
                    {
                        "event": "autoresearch_candidate_decided",
                        "transition_id": _transition_id(
                            state.candidate_id, "decision", decision
                        ),
                        "candidate_id": state.candidate_id,
                        "decision": decision,
                    },
                )
                if decision in {"promote", "promote_simplification"}:
                    append_transition(
                        journal,
                        campaign.policy,
                        {
                            "event": "autoresearch_campaign_completed",
                            "transition_id": "campaign-completed-after-promotion",
                        },
                    )
                elif decision == "reject":
                    final_decisions = _candidate_decisions(tuple(journal.events()))
                    if all(
                        final_decisions.get(proposal.candidate_id) == "reject"
                        for proposal in campaign.proposals
                    ):
                        append_transition(
                            journal,
                            campaign.policy,
                            {
                                "event": "autoresearch_campaign_completed",
                                "transition_id": "campaign-completed-queue-exhausted",
                            },
                        )
                # One invocation crosses at most one calibration/search pair.
                # The caller can publish and push its scalar checkpoint before
                # explicitly resuming the next pair.
                break
        except CellProjectionError as error:
            _append_terminal(
                journal,
                campaign,
                failure_kind=error.failure_kind,
                cleanup_verified=error.failure_kind != "cleanup_breach",
            )
        return summarize_campaign(campaign.campaign_dir)
