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
from pathlib import Path
import tomllib
from typing import Any, Callable, Mapping

from .autoresearch import CampaignPolicy, CandidateDelta, validate_one_axis_delta
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
