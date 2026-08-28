"""Strict planning primitives for the single-user Qwen autoresearch campaign.

The campaign planner is deliberately separate from the execution controller.
It validates a finite queue of semantic one-axis candidates, constructs the
immutable policy, and can freeze fresh SparkBench plans for calibration,
screening, and reverse-order confirmation.  Planning never executes a plan.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
import subprocess
import sys
import time
import tomllib
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping

if TYPE_CHECKING:
    from .autoresearch_checkpoint import CheckpointAcknowledgement

from .autoresearch import (
    AUDIT_RESERVE_S,
    CELL_TIMEOUT_S,
    CLEANUP_TIMEOUT_S,
    CampaignPolicy,
    CandidateDelta,
    EligibilityInputs,
    PairObservation,
    ReplayState,
    SimplificationEvidence,
    TimingInputs,
    append_transition,
    evaluate_calibration,
    evaluate_promotion,
    evaluate_screen,
    evaluate_simplification_promotion,
    evaluate_simplification_screen,
    pair_order,
    replay_transitions,
    validate_one_axis_delta,
)
from .autoresearch_admission import (
    AdmissionBinding,
    AdmissionJournalError,
    AdmissionTarget,
    append_admission_record,
    controller_prefix_sha256,
    observe_admission,
    read_admission_journal,
)
from .autoresearch_worker import (
    WorkerLifecycleError,
    WorkerProgress,
    WorkerRunResult,
    recover_owned_worker,
    run_owned_worker,
)
from .execution_admission import model_execution_blocker
from .host_safety import parse_meminfo, read_host_meminfo
from .journal import Journal, canonical_json, content_hash, utc_now, write_json
from .manifest import (
    ManifestError,
    ModelSpec,
    SuiteSpec,
    load_models,
    load_suite,
    model_spec_to_dict,
    validate_benchmark_selection,
)
from .runner import _canonical_case, create_plan
from .runtime import recover_owned_sglang


CAMPAIGN_SCHEMA_VERSION = 1
FROZEN_CAMPAIGN_SCHEMA_VERSION = 3
EXPECTED_SUITE_ID = "qwen38-flash-next-sglang-agent64k-autoresearch"
EXPECTED_CAMPAIGN_RELATIVE_PATH = Path(
    "manifests/campaigns/qwen38_flash_next_single_user_autoresearch.toml"
)
EXPECTED_CAMPAIGN_ID = "qwen38-flash-next-single-user-autoresearch-2026-08-28"
EXPECTED_CAMPAIGN_CUTOFF = "2026-08-28T07:00:00-07:00"
EXPECTED_BASELINE_ID = (
    "qwen38-flash-next-nvfp4-mtp2-agent64k-low-ple-mapped-sglang"
)
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
EXPECTED_CANDIDATE_IDS = (
    "qwen38-flash-next-nvfp4-mtp2-agent64k-none-ple-mapped-sglang",
    "qwen38-flash-next-nvfp4-mtp2-agent64k-low-chunk2k-ple-mapped-sglang",
    "qwen38-flash-next-nvfp4-mtp3-agent64k-low-ple-mapped-sglang",
)
# These canonical manifest projections make the protocol self-anchoring: a
# manifest-only rewrite cannot silently freeze a different workload or model.
EXPECTED_MODEL_SPEC_DIGESTS = {
    EXPECTED_BASELINE_ID: "f966e9527cb5b9bf18bab737484e0cdb03cb4d358cf5a89d20150d1052b6ecbd",
    EXPECTED_CANDIDATE_IDS[0]: "d41c6843824ffcb15e5198354dc522cf660cba940c4d5e0af8ad60ca0cc10312",
    EXPECTED_CANDIDATE_IDS[1]: "5efdecbac20414bb36d595d72af823e362f0c31437051c635616f1be12bba38d",
    EXPECTED_CANDIDATE_IDS[2]: "f9ccd82a08006b3a7badc032bdebaedc217101365b79444e748bc6354460f825",
}
EXPECTED_SUITE_SPEC_DIGEST = (
    "260506c71f890e714b50829e69289fdc1e2490b1c7d5a8a08218c5369128a063"
)
HOST_SAFETY_MIN_MEMAVAILABLE_KIB = 14 * 1024 * 1024
HOST_SAFETY_MAX_SWAP_GROWTH_KIB = 512 * 1024
HOST_SAFETY_MAX_STARTING_SWAP_KIB = 64 * 1024
EXPECTED_HOST_SAFETY_THRESHOLDS = (14, 512, 64)
START_MARKER_TIMEOUT_S = 30
FINALIZATION_TIMEOUT_S = 10
PAIR_ADMISSION_REMAINING_S = (
    2 * CELL_TIMEOUT_S
    + 2 * CLEANUP_TIMEOUT_S
    + 2 * START_MARKER_TIMEOUT_S
    + CLEANUP_TIMEOUT_S
    + FINALIZATION_TIMEOUT_S
    + AUDIT_RESERVE_S
)
MAX_PROGRESS_LINE_BYTES = 16 * 1024 * 1024
MAX_PROGRESS_JOURNAL_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _LegacyBlockedCampaignSeal:
    campaign_id: str
    created_at: str
    cutoff: str
    integrity_sha256: str
    preview_sha256: str
    policy_sha256: str
    harness_tree_sha256: str
    harness_file_count: int
    campaign_json_sha256: str
    campaign_json_size: int
    summary_json_sha256: str
    summary_json_size: int
    tree_sha256: str
    tree_size: int


_LEGACY_BLOCKED_CAMPAIGN_SEAL = _LegacyBlockedCampaignSeal(
    campaign_id=EXPECTED_CAMPAIGN_ID,
    created_at="2026-08-28T08:09:19.396+00:00",
    cutoff=EXPECTED_CAMPAIGN_CUTOFF,
    integrity_sha256=(
        "ea576eaf6540bd842e956bbaec719227b60389df52505390ad0d26025bdf7d92"
    ),
    preview_sha256=(
        "be2b70f0c0415d258e7d43979566e59606c55bc3adc87534d93945a067d8d1cb"
    ),
    policy_sha256=(
        "ff3237c4106ebafbc50710e9a2222007611fc972ccc001b28a7100a6c01a50e7"
    ),
    harness_tree_sha256=(
        "33170881721d0dce0f4466495110b336a7451fcd1635c5667f7fc5f722f7599f"
    ),
    harness_file_count=84,
    campaign_json_sha256=(
        "523112428589a338faab93bf2eaa94a474bc0722c9ff396ca0fff4f34269f421"
    ),
    campaign_json_size=14_874,
    summary_json_sha256=(
        "8c197a3f0dbda0bb06c3fca1e04940f54cb8f918754aa164acd8603f4afa847d"
    ),
    summary_json_size=471,
    tree_sha256=(
        "5b121756b6a95644cb99969f309adaf43ea9d9b5d153749d869cd8bac420e988"
    ),
    tree_size=156_253,
)
_LEGACY_BLOCKED_CAMPAIGN_BLOCKERS = (
    "insufficient_time_for_pair",
    "starting_swap_above_clean_limit",
    "insufficient_preflight_memavailable",
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
    run_nonce: str | None = None
    plan_integrity_hash: str | None = None
    run_completed_at: datetime | None = None

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
    run_nonce: str


@dataclass(frozen=True, slots=True)
class FrozenCampaign:
    campaign_dir: Path
    created_at: str
    integrity_hash: str
    campaign_id: str
    cutoff: datetime
    baseline_id: str
    policy: CampaignPolicy
    policy_digest: str
    preview_digest: str
    harness_tree_sha256: str
    harness_file_count: int
    admission_journal_required: bool
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
    if definition.id != EXPECTED_CAMPAIGN_ID:
        raise CampaignPlanningError("campaign ID does not match the audited protocol")
    if definition.cutoff != EXPECTED_CAMPAIGN_CUTOFF:
        raise CampaignPlanningError("campaign cutoff does not match the audited protocol")
    if definition.baseline_id != EXPECTED_BASELINE_ID:
        raise CampaignPlanningError(
            "campaign baseline does not match the audited protocol"
        )
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
    if tuple(candidate.id for candidate in definition.candidates) != EXPECTED_CANDIDATE_IDS:
        raise CampaignPlanningError(
            "candidate queue does not match the audited protocol"
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


def _mapping_arg_value(model: Mapping[str, Any], option: str) -> str:
    arguments = model.get("args")
    if not isinstance(arguments, list) or any(
        not isinstance(argument, str) for argument in arguments
    ):
        raise CampaignPlanningError("frozen model args must be a string array")
    indexes = [index for index, value in enumerate(arguments) if value == option]
    if len(indexes) != 1 or indexes[0] + 1 >= len(arguments):
        raise CampaignPlanningError(f"frozen model must declare one {option}")
    value = arguments[indexes[0] + 1]
    if value.startswith("--"):
        raise CampaignPlanningError(f"frozen model has no value for {option}")
    return value


def _mapping_semantic_config(model: Mapping[str, Any]) -> dict[str, Any]:
    request_body = model.get("request_body_json")
    if not isinstance(request_body, str):
        raise CampaignPlanningError("frozen model reasoning policy is missing")
    try:
        reasoning_policy = json.loads(request_body)
    except json.JSONDecodeError as error:
        raise CampaignPlanningError("frozen model reasoning policy is invalid") from error
    if not isinstance(reasoning_policy, dict):
        raise CampaignPlanningError("frozen model reasoning policy must be an object")
    return {
        "reasoning_policy": reasoning_policy,
        "chunked_prefill_size": int(
            _mapping_arg_value(model, "--chunked-prefill-size")
        ),
        "nextn_bundle": {
            "steps": int(_mapping_arg_value(model, "--speculative-num-steps")),
            "draft_tokens": int(
                _mapping_arg_value(model, "--speculative-num-draft-tokens")
            ),
        },
    }


def _mapping_invariant_model_projection(model: Mapping[str, Any]) -> dict[str, Any]:
    projection = dict(model)
    for key in ("id", "description", "served_name", "request_body_json"):
        projection.pop(key, None)
    arguments = list(projection.get("args", []))
    replacements = {
        "--served-model-name": "<served-name>",
        "--chunked-prefill-size": "<chunked-prefill-size>",
        "--speculative-num-steps": "<nextn-steps>",
        "--speculative-num-draft-tokens": "<nextn-draft-tokens>",
    }
    for option, replacement in replacements.items():
        indexes = [index for index, value in enumerate(arguments) if value == option]
        if len(indexes) != 1 or indexes[0] + 1 >= len(arguments):
            raise CampaignPlanningError(f"frozen model must declare one {option}")
        arguments[indexes[0] + 1] = replacement
    projection["args"] = arguments
    return projection


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
    suite_projection = asdict(suite)
    if suite_projection.get("protocol_digest") is None:
        suite_projection.pop("protocol_digest", None)
    if content_hash(suite_projection, 64) != EXPECTED_SUITE_SPEC_DIGEST:
        raise CampaignPlanningError("campaign suite content changed")
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
    audited_ids = (definition.baseline_id,) + tuple(
        candidate.id for candidate in definition.candidates
    )
    for profile_id in audited_ids:
        profile = models.get(profile_id)
        if profile is None or content_hash(model_spec_to_dict(profile), 64) != (
            EXPECTED_MODEL_SPEC_DIGESTS[profile_id]
        ):
            raise CampaignPlanningError(
                f"campaign model profile {profile_id!r} changed"
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


def harness_tree_identity(workspace: Path) -> tuple[str, int]:
    """Hash executable and frozen-protocol inputs, excluding evidence outputs."""

    root = workspace.resolve(strict=True)
    bench_root = root / "bench"
    manifests_root = root / "manifests"
    if bench_root.is_symlink() or not bench_root.is_dir():
        raise CampaignPlanningError("benchmark harness directory is unsafe")
    if manifests_root.is_symlink() or not manifests_root.is_dir():
        raise CampaignPlanningError("benchmark manifest directory is unsafe")
    paths = sorted(
        {
            *root.glob("*.py"),
            *bench_root.rglob("*.py"),
            *manifests_root.rglob("*.toml"),
        }
    )
    if root / "sparkbench.py" not in paths:
        raise CampaignPlanningError("benchmark CLI source is missing")
    digest = hashlib.sha256()
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise CampaignPlanningError("benchmark harness contains an unsafe source")
        try:
            relative = path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as error:
            raise CampaignPlanningError("benchmark harness source escapes workspace") from error
        relative_bytes = relative.as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative_bytes).to_bytes(4, "big"))
        digest.update(relative_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest(), len(paths)


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
    selected_profile_ids = (
        definition.baseline_id,
        *(proposal.candidate_id for proposal in preview.proposals),
    )
    for profile_id in selected_profile_ids:
        blocker = model_execution_blocker(models[profile_id])
        if blocker is not None:
            raise CampaignPlanningError(blocker)
    harness_tree_sha256, harness_file_count = harness_tree_identity(workspace)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    campaign_dir = results_root / f"{stamp}-{definition.id}-{preview.digest[:8]}"
    campaign_dir.mkdir(parents=True, exist_ok=False)
    # Freeze the lock topology with the campaign itself.  Read-only consumers
    # can then take a shared lock without creating or chmodding anything in the
    # source tree they are inspecting.
    with _campaign_lock(campaign_dir / ".autoresearch.lock"):
        pass
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
        run_nonce = plan.get("run_nonce")
        if (
            not isinstance(run_nonce, str)
            or len(run_nonce) != 32
            or any(character not in "0123456789abcdef" for character in run_nonce)
        ):
            raise CampaignPlanningError("fresh cell plan has no ownership nonce")
        frozen_cells.append(
            {
                **cell,
                "ordinal": ordinal,
                "run_dir": str(run_dir.relative_to(campaign_dir)),
                "plan_fingerprint": plan["fingerprint"],
                "plan_integrity_hash": plan.get("integrity_hash"),
                "run_nonce": run_nonce,
            }
        )
    if harness_tree_identity(workspace) != (
        harness_tree_sha256,
        harness_file_count,
    ):
        raise CampaignPlanningError("benchmark harness changed while plans were frozen")
    frozen = {
        "schema_version": FROZEN_CAMPAIGN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "harness_tree_sha256": harness_tree_sha256,
        "harness_file_count": harness_file_count,
        "admission_journal_required": True,
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
    schema_version = frozen.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in {2, FROZEN_CAMPAIGN_SCHEMA_VERSION}
    ):
        raise CampaignPlanningError("unsupported frozen campaign schema version")
    expected_top = {
        "schema_version",
        "created_at",
        "harness_tree_sha256",
        "harness_file_count",
        "preview",
        "preview_digest",
        "cells",
        "execution_started",
        "integrity_hash",
    }
    if schema_version == FROZEN_CAMPAIGN_SCHEMA_VERSION:
        expected_top.add("admission_journal_required")
    if set(frozen) != expected_top:
        raise CampaignPlanningError("frozen campaign has an unknown or missing field")
    admission_journal_required = schema_version == FROZEN_CAMPAIGN_SCHEMA_VERSION
    if admission_journal_required and frozen["admission_journal_required"] is not True:
        raise CampaignPlanningError(
            "frozen campaign admission-journal requirement changed"
        )
    created_at = _require_string(
        frozen["created_at"], context="frozen campaign creation time"
    )
    if len(created_at) > 64:
        raise CampaignPlanningError("frozen campaign creation time is invalid")
    try:
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise CampaignPlanningError(
            "frozen campaign creation time is invalid"
        ) from error
    if parsed_created_at.tzinfo is None or parsed_created_at.utcoffset() is None:
        raise CampaignPlanningError(
            "frozen campaign creation time must be timezone-aware"
        )
    harness_tree_sha256 = frozen["harness_tree_sha256"]
    harness_file_count = frozen["harness_file_count"]
    if (
        not isinstance(harness_tree_sha256, str)
        or len(harness_tree_sha256) != 64
        or any(character not in "0123456789abcdef" for character in harness_tree_sha256)
        or isinstance(harness_file_count, bool)
        or not isinstance(harness_file_count, int)
        or harness_file_count <= 0
    ):
        raise CampaignPlanningError("frozen campaign harness identity is invalid")
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
    if preview["campaign_id"] != EXPECTED_CAMPAIGN_ID:
        raise CampaignPlanningError("frozen campaign identity changed")
    if preview["baseline_id"] != EXPECTED_BASELINE_ID:
        raise CampaignPlanningError("frozen campaign baseline changed")
    policy = CampaignPolicy.from_mapping(preview["policy"])
    if policy.digest != preview["policy_digest"]:
        raise CampaignPlanningError("frozen campaign policy digest does not match")
    if (
        policy.primary_case_ids != EXPECTED_PRIMARY_CASE_IDS
        or policy.allowed_axes != EXPECTED_AXES
    ):
        raise CampaignPlanningError("frozen campaign evaluator policy changed")
    if preview["cutoff"] != EXPECTED_CAMPAIGN_CUTOFF:
        raise CampaignPlanningError("frozen campaign cutoff changed")
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
    if tuple(proposal.candidate_id for proposal in proposals) != EXPECTED_CANDIDATE_IDS:
        raise CampaignPlanningError("frozen candidate queue changed")

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
        "run_nonce",
    }
    cells: list[FrozenCell] = []
    profile_models: dict[str, dict[str, Any]] = {}
    frozen_suite_basis: dict[str, Any] | None = None
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
        run_nonce = raw["run_nonce"]
        if (
            not isinstance(run_nonce, str)
            or len(run_nonce) != 32
            or any(character not in "0123456789abcdef" for character in run_nonce)
            or plan.get("run_nonce") != run_nonce
        ):
            raise CampaignPlanningError("frozen cell ownership nonce changed")
        model = plan.get("model")
        if not isinstance(model, dict) or model.get("id") != raw["profile_id"]:
            raise CampaignPlanningError("frozen cell profile binding changed")
        profile_id = str(raw["profile_id"])
        prior_model = profile_models.get(profile_id)
        if prior_model is not None and canonical_json(prior_model) != canonical_json(model):
            raise CampaignPlanningError("repeated frozen profile records changed")
        profile_models[profile_id] = model
        suite = plan.get("suite")
        if not isinstance(suite, dict) or not isinstance(suite.get("cases"), list):
            raise CampaignPlanningError("frozen cell suite binding is malformed")
        suite_basis = {
            **suite,
            "cases": [
                {key: value for key, value in case.items() if key != "case_id"}
                for case in suite["cases"]
                if isinstance(case, dict)
            ],
        }
        if len(suite_basis["cases"]) != len(suite["cases"]):
            raise CampaignPlanningError("frozen cell suite contains a non-object case")
        if frozen_suite_basis is None:
            frozen_suite_basis = suite_basis
        elif canonical_json(frozen_suite_basis) != canonical_json(suite_basis):
            raise CampaignPlanningError("frozen cell suite protocols changed")
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
                run_nonce=run_nonce,
            )
        )
    if len({cell.cell_id for cell in cells}) != 14:
        raise CampaignPlanningError("frozen cell IDs must be unique")
    if len({cell.run_dir for cell in cells}) != 14:
        raise CampaignPlanningError("frozen cell run directories must be unique")
    if len({cell.run_nonce for cell in cells}) != 14:
        raise CampaignPlanningError("frozen cell ownership nonces must be unique")
    expected_profiles = {
        str(preview["baseline_id"]),
        *(proposal.candidate_id for proposal in proposals),
    }
    if set(profile_models) != expected_profiles:
        raise CampaignPlanningError("frozen campaign profile set changed")
    for profile_id, model in profile_models.items():
        if content_hash(model, 64) != EXPECTED_MODEL_SPEC_DIGESTS[profile_id]:
            raise CampaignPlanningError(
                f"frozen campaign model profile {profile_id!r} changed"
            )
    if (
        frozen_suite_basis is None
        or frozen_suite_basis.get("id") != EXPECTED_SUITE_ID
        or tuple(
            case.get("id")
            for case in frozen_suite_basis.get("cases", [])
            if isinstance(case, dict)
        )
        != EXPECTED_CASE_IDS
    ):
        raise CampaignPlanningError("frozen campaign suite topology changed")
    if content_hash(frozen_suite_basis, 64) != EXPECTED_SUITE_SPEC_DIGEST:
        raise CampaignPlanningError("frozen campaign suite content changed")
    baseline_model = profile_models[str(preview["baseline_id"])]
    invariant = canonical_json(_mapping_invariant_model_projection(baseline_model))
    baseline_semantic = _mapping_semantic_config(baseline_model)
    for proposal in proposals:
        candidate_model = profile_models[proposal.candidate_id]
        if canonical_json(_mapping_invariant_model_projection(candidate_model)) != invariant:
            raise CampaignPlanningError("frozen candidate changes a non-axis model field")
        actual_delta = validate_one_axis_delta(
            baseline_semantic,
            _mapping_semantic_config(candidate_model),
            allowed_axes=EXPECTED_AXES,
        )
        if actual_delta != proposal.delta:
            raise CampaignPlanningError("frozen candidate semantic delta changed")
    campaign = FrozenCampaign(
        campaign_dir=root,
        created_at=created_at,
        integrity_hash=integrity_hash,
        campaign_id=_require_string(
            preview["campaign_id"], context="frozen campaign ID"
        ),
        cutoff=cutoff,
        baseline_id=_require_string(
            preview["baseline_id"], context="frozen baseline ID"
        ),
        policy=policy,
        policy_digest=policy.digest,
        preview_digest=str(frozen["preview_digest"]),
        harness_tree_sha256=harness_tree_sha256,
        harness_file_count=harness_file_count,
        admission_journal_required=admission_journal_required,
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


def _progress_timestamp(value: Any, *, context: str) -> None:
    if not isinstance(value, str):
        raise WorkerLifecycleError(
            "worker_progress_malformed", f"{context} timestamp is missing"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise WorkerLifecycleError(
            "worker_progress_malformed", f"{context} timestamp is invalid"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WorkerLifecycleError(
            "worker_progress_malformed", f"{context} timestamp has no timezone"
        )


def _progress_monotonic_ns(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkerLifecycleError(
            "worker_progress_malformed",
            f"{context} monotonic clock must be a positive integer",
        )
    return value


class _CellLifecycleProgress:
    """Incrementally validate durable lifecycle markers for one frozen cell."""

    def __init__(
        self,
        *,
        events_path: Path,
        plan_fingerprint: str,
        run_nonce: str,
        measurement_timeout_s: int,
        cleanup_timeout_s: int,
        finalization_timeout_s: int = FINALIZATION_TIMEOUT_S,
    ) -> None:
        self.events_path = events_path
        self.plan_fingerprint = plan_fingerprint
        self.run_nonce = run_nonce
        self.measurement_timeout_s = measurement_timeout_s
        self.cleanup_timeout_s = cleanup_timeout_s
        self.finalization_timeout_s = finalization_timeout_s
        self._identity: tuple[int, int] | None = None
        self._offset = 0
        self._tail = b""
        self._marker_lines: list[tuple[int, bytes]] = []
        self._run_started = False
        self._measurement_started_ns: int | None = None
        self._measurement_complete_ns: int | None = None
        self._server_stopped_ns: int | None = None
        self._run_completed = False
        self.measurement_elapsed_s: float | None = None
        self.cleanup_elapsed_s: float | None = None

    def _require_keys(
        self, event: Mapping[str, Any], expected: frozenset[str], *, context: str
    ) -> None:
        if set(event) != expected:
            raise WorkerLifecycleError(
                "worker_progress_malformed", f"{context} marker schema changed"
            )
        _progress_timestamp(event.get("timestamp"), context=context)

    def _consume(self, event: Mapping[str, Any]) -> None:
        name = event.get("event")
        if name == "run_start":
            self._require_keys(
                event,
                frozenset(
                    {
                        "event",
                        "timestamp",
                        "completed_cases_at_resume",
                        "plan_fingerprint",
                        "run_nonce",
                    }
                ),
                context="run_start",
            )
            if self._run_started or self._measurement_started_ns is not None:
                raise WorkerLifecycleError(
                    "worker_progress_malformed", "run_start marker is duplicated"
                )
            if event.get("completed_cases_at_resume") != []:
                raise WorkerLifecycleError(
                    "worker_progress_malformed", "campaign worker attempted a resume"
                )
            if (
                event.get("plan_fingerprint") != self.plan_fingerprint
                or event.get("run_nonce") != self.run_nonce
            ):
                raise WorkerLifecycleError(
                    "worker_progress_binding_mismatch",
                    "run_start marker does not match the frozen cell",
                )
            self._run_started = True
            return
        if name == "measurement_started":
            self._require_keys(
                event,
                frozenset(
                    {
                        "event",
                        "timestamp",
                        "monotonic_ns",
                        "plan_fingerprint",
                        "run_nonce",
                    }
                ),
                context="measurement_started",
            )
            if not self._run_started or self._measurement_started_ns is not None:
                raise WorkerLifecycleError(
                    "worker_progress_malformed",
                    "measurement_started marker is out of order or duplicated",
                )
            if (
                event.get("plan_fingerprint") != self.plan_fingerprint
                or event.get("run_nonce") != self.run_nonce
            ):
                raise WorkerLifecycleError(
                    "worker_progress_binding_mismatch",
                    "measurement marker does not match the frozen cell",
                )
            self._measurement_started_ns = _progress_monotonic_ns(
                event.get("monotonic_ns"), context="measurement_started"
            )
            return
        if name == "measurement_complete":
            self._require_keys(
                event,
                frozenset({"event", "timestamp", "elapsed_s", "monotonic_ns"}),
                context="measurement_complete",
            )
            if (
                self._measurement_started_ns is None
                or self._measurement_complete_ns is not None
                or self._server_stopped_ns is not None
            ):
                raise WorkerLifecycleError(
                    "worker_progress_malformed",
                    "measurement_complete marker is out of order or duplicated",
                )
            completed_ns = _progress_monotonic_ns(
                event.get("monotonic_ns"), context="measurement_complete"
            )
            try:
                elapsed_s = _finite_nonnegative(
                    event.get("elapsed_s"), context="measurement_complete.elapsed_s"
                )
            except CellProjectionError as error:
                raise WorkerLifecycleError(
                    "worker_progress_malformed",
                    "measurement_complete elapsed time is invalid",
                ) from error
            elapsed_ns = completed_ns - self._measurement_started_ns
            if (
                elapsed_ns < 0
                or elapsed_ns > self.measurement_timeout_s * 1_000_000_000
                or elapsed_s > self.measurement_timeout_s
                or abs(elapsed_s - elapsed_ns / 1_000_000_000) > 0.000001
            ):
                raise WorkerLifecycleError(
                    "worker_progress_measurement_budget",
                    "measurement marker exceeds or disagrees with its causal budget",
                )
            self._measurement_complete_ns = completed_ns
            self.measurement_elapsed_s = elapsed_s
            return
        if name == "server_stopped":
            self._require_keys(
                event,
                frozenset(
                    {
                        "event",
                        "timestamp",
                        "backend",
                        "cleanup_elapsed_s",
                        "monotonic_ns",
                    }
                ),
                context="server_stopped",
            )
            if (
                self._measurement_started_ns is None
                or self._server_stopped_ns is not None
            ):
                raise WorkerLifecycleError(
                    "worker_progress_malformed",
                    "server_stopped marker is out of order or duplicated",
                )
            stopped_ns = _progress_monotonic_ns(
                event.get("monotonic_ns"), context="server_stopped"
            )
            try:
                cleanup_s = _finite_nonnegative(
                    event.get("cleanup_elapsed_s"),
                    context="server_stopped.cleanup_elapsed_s",
                )
            except CellProjectionError as error:
                raise WorkerLifecycleError(
                    "worker_progress_malformed", "server cleanup elapsed time is invalid"
                ) from error
            if stopped_ns < self._measurement_started_ns or cleanup_s > (
                self.cleanup_timeout_s
            ):
                raise WorkerLifecycleError(
                    "worker_progress_cleanup_budget",
                    "server cleanup marker exceeds its causal budget",
                )
            if self._measurement_complete_ns is not None and (
                stopped_ns < self._measurement_complete_ns
                or stopped_ns - self._measurement_complete_ns
                > self.cleanup_timeout_s * 1_000_000_000
            ):
                raise WorkerLifecycleError(
                    "worker_progress_cleanup_budget",
                    "server cleanup marker exceeds its causal budget",
                )
            self._server_stopped_ns = stopped_ns
            self.cleanup_elapsed_s = cleanup_s
            return
        if name == "run_complete":
            self._require_keys(
                event,
                frozenset({"event", "timestamp", "status"}),
                context="run_complete",
            )
            if self._server_stopped_ns is None or self._run_completed:
                raise WorkerLifecycleError(
                    "worker_progress_malformed",
                    "run_complete marker is out of order or duplicated",
                )
            self._run_completed = True

    def _read_new_events(self) -> None:
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.events_path, flags)
        except FileNotFoundError:
            if self._identity is not None:
                raise WorkerLifecycleError(
                    "worker_progress_regressed", "cell journal disappeared"
                )
            return
        except OSError as error:
            raise WorkerLifecycleError(
                "worker_progress_unreadable", "cell journal could not be opened"
            ) from error
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.geteuid()
            ):
                raise WorkerLifecycleError(
                    "worker_progress_unsafe", "cell journal topology is unsafe"
                )
            if metadata.st_size > MAX_PROGRESS_JOURNAL_BYTES:
                raise WorkerLifecycleError(
                    "worker_progress_malformed", "cell journal exceeds its size bound"
                )
            identity = (metadata.st_dev, metadata.st_ino)
            if self._identity is None:
                self._identity = identity
            elif identity != self._identity or metadata.st_size < self._offset:
                raise WorkerLifecycleError(
                    "worker_progress_regressed", "cell journal was replaced or truncated"
                )
            for marker_offset, marker_line in self._marker_lines:
                if os.pread(descriptor, len(marker_line), marker_offset) != marker_line:
                    raise WorkerLifecycleError(
                        "worker_progress_changed",
                        "a durable cell lifecycle marker changed in place",
                    )
            payload_start = self._offset - len(self._tail)
            os.lseek(descriptor, self._offset, os.SEEK_SET)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 65_536)
                if not chunk:
                    break
                chunks.append(chunk)
                self._offset += len(chunk)
            payload = self._tail + b"".join(chunks)
            lines = payload.split(b"\n")
            self._tail = lines.pop()
            if len(self._tail) > MAX_PROGRESS_LINE_BYTES:
                raise WorkerLifecycleError(
                    "worker_progress_malformed", "cell journal line is too large"
                )
            line_offset = payload_start
            for line in lines:
                if not line or len(line) > MAX_PROGRESS_LINE_BYTES:
                    raise WorkerLifecycleError(
                        "worker_progress_malformed", "cell journal line is invalid"
                    )
                try:
                    event = json.loads(
                        line.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs
                    )
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    CellProjectionError,
                ) as error:
                    raise WorkerLifecycleError(
                        "worker_progress_malformed",
                        "cell journal has a malformed complete line",
                    ) from error
                if not isinstance(event, dict):
                    raise WorkerLifecycleError(
                        "worker_progress_malformed", "cell journal event is not an object"
                    )
                self._consume(event)
                if event.get("event") in {
                    "run_start",
                    "measurement_started",
                    "measurement_complete",
                    "server_stopped",
                    "run_complete",
                }:
                    self._marker_lines.append((line_offset, line))
                line_offset += len(line) + 1
        finally:
            os.close(descriptor)

    def __call__(self) -> WorkerProgress | None:
        self._read_new_events()
        if self._server_stopped_ns is not None:
            return WorkerProgress(
                "finalization",
                self._server_stopped_ns / 1_000_000_000
                + self.finalization_timeout_s,
            )
        if self._measurement_complete_ns is not None:
            return WorkerProgress(
                "cleanup",
                self._measurement_complete_ns / 1_000_000_000
                + self.cleanup_timeout_s,
            )
        if self._measurement_started_ns is not None:
            return WorkerProgress(
                "measurement",
                self._measurement_started_ns / 1_000_000_000
                + self.measurement_timeout_s,
            )
        return None

    @property
    def run_completed(self) -> bool:
        return self._run_completed


def _one_event(
    events: tuple[dict[str, Any], ...], name: str
) -> dict[str, Any]:
    matches = tuple(event for event in events if event.get("event") == name)
    if len(matches) != 1:
        raise CellProjectionError(
            f"cell must contain exactly one {name!r} event, found {len(matches)}"
        )
    return matches[0]


def _aware_event_timestamp(event: Mapping[str, Any], *, context: str) -> datetime:
    raw = event.get("timestamp")
    if not isinstance(raw, str):
        raise CellProjectionError(f"{context} event has no timestamp")
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as error:
        raise CellProjectionError(f"{context} timestamp is invalid") from error
    if value.tzinfo is None or value.utcoffset() is None:
        raise CellProjectionError(f"{context} timestamp is not timezone-aware")
    return value


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
    protocol_digest = suite_without_case_ids.get("protocol_digest")
    expected_cases = [
        _canonical_case(
            model,
            case,
            protocol_digest=(
                protocol_digest if isinstance(protocol_digest, str) else None
            ),
        )
        for case in suite_without_case_ids["cases"]
    ]
    if canonical_json(cases) != canonical_json(expected_cases):
        raise CellProjectionError("cell plan case identities are not model-bound")
    expected = content_hash(
        {"model": model, "suite": suite_without_case_ids, "resolved": resolved}
    )
    if plan.get("fingerprint") != expected:
        raise CellProjectionError("cell plan fingerprint does not match")


def _bound_frozen_cell_plan_model(cell: FrozenCell) -> dict[str, Any]:
    """Read and revalidate the exact model bound to one frozen cell."""

    plan = _read_json_object(
        cell.run_dir / "plan.json", context="frozen cell plan"
    )
    _validate_plan_integrity(plan)
    if (
        plan.get("fingerprint") != cell.plan_fingerprint
        or plan.get("integrity_hash") != cell.plan_integrity_hash
        or plan.get("run_nonce") != cell.run_nonce
    ):
        raise CellProjectionError("frozen cell plan binding changed")
    model = plan.get("model")
    if not isinstance(model, dict) or model.get("id") != cell.profile_id:
        raise CellProjectionError("frozen cell profile binding changed")
    return model


def _normalized_flags(model: Mapping[str, Any]) -> tuple[str, ...]:
    raw = model.get("args")
    if not isinstance(raw, (list, tuple)) or any(
        not isinstance(value, str) for value in raw
    ):
        raise CellProjectionError("cell plan model args must be a string array")
    arguments = list(raw)
    indexes = [
        index for index, argument in enumerate(arguments) if argument == "--served-model-name"
    ]
    if len(indexes) != 1 or indexes[0] + 1 >= len(arguments):
        raise CellProjectionError("cell plan has invalid served-model-name args")
    arguments[indexes[0] + 1] = "<served-name>"
    request_body = model.get("request_body_json")
    if not isinstance(request_body, str):
        raise CellProjectionError("cell plan has no explicit request policy")
    try:
        request_policy = json.loads(request_body)
    except json.JSONDecodeError as error:
        raise CellProjectionError("cell plan request policy is invalid JSON") from error
    if not isinstance(request_policy, dict) or set(request_policy) != {
        "chat_template_kwargs"
    }:
        raise CellProjectionError("cell plan request policy topology changed")
    template = request_policy["chat_template_kwargs"]
    if not isinstance(template, dict):
        raise CellProjectionError("cell plan chat-template policy is invalid")
    thinking = template.get("enable_thinking")
    if not isinstance(thinking, bool):
        raise CellProjectionError("cell plan thinking policy must be boolean")
    if thinking:
        if set(template) != {"enable_thinking", "reasoning_effort"}:
            raise CellProjectionError("thinking request policy topology changed")
        effort = template.get("reasoning_effort")
        if not isinstance(effort, str) or not effort:
            raise CellProjectionError("cell plan reasoning effort is invalid")
        arguments.extend(
            ("<request:enable-thinking>", f"<request:reasoning-effort={effort}>")
        )
    elif set(template) != {"enable_thinking"}:
        raise CellProjectionError("no-thinking request policy topology changed")
    # This is a semantic flag bundle, not argv: repeated scalar values (for
    # example two independent options both set to ``4``) must not make an
    # otherwise strict simplification record structurally invalid.
    return tuple(dict.fromkeys(arguments))


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
    plan_fingerprint = plan.get("fingerprint")
    run_nonce = plan.get("run_nonce")
    if not isinstance(plan_fingerprint, str) or not isinstance(run_nonce, str):
        raise CellProjectionError("cell plan lifecycle binding is missing")
    lifecycle = _CellLifecycleProgress(
        events_path=run_dir / "events.jsonl",
        plan_fingerprint=plan_fingerprint,
        run_nonce=run_nonce,
        measurement_timeout_s=CELL_TIMEOUT_S,
        cleanup_timeout_s=CLEANUP_TIMEOUT_S,
    )
    try:
        lifecycle_progress = lifecycle()
    except WorkerLifecycleError as error:
        raise CellProjectionError(
            "cell lifecycle markers are invalid",
            failure_kind=(
                "cleanup_breach"
                if error.code == "worker_progress_cleanup_budget"
                else "measurement"
                if error.code == "worker_progress_measurement_budget"
                else "audit"
            ),
        ) from error
    if (
        lifecycle_progress is None
        or lifecycle_progress.phase != "finalization"
        or not lifecycle.run_completed
        or lifecycle.measurement_elapsed_s is None
        or lifecycle.cleanup_elapsed_s is None
    ):
        raise CellProjectionError("cell lifecycle journal is not terminal")

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
    _one_event(events, "measurement_started")
    _one_event(events, "server_ready")
    measurement_complete = _one_event(events, "measurement_complete")
    server_stopped = _one_event(events, "server_stopped")
    run_complete = _one_event(events, "run_complete")
    server_stopped_at = _aware_event_timestamp(
        server_stopped, context="server_stopped"
    )
    run_completed_at = _aware_event_timestamp(run_complete, context="run_complete")
    finalization_elapsed_s = (run_completed_at - server_stopped_at).total_seconds()
    if not 0 <= finalization_elapsed_s <= FINALIZATION_TIMEOUT_S:
        raise CellProjectionError(
            "run completion exceeds the frozen finalization budget",
            failure_kind="cleanup_breach",
        )
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
        plan_fingerprint=plan_fingerprint,
        measurements=measurements,
        measurement_elapsed_s=lifecycle.measurement_elapsed_s,
        cleanup_elapsed_s=lifecycle.cleanup_elapsed_s,
        minimum_memavailable_gib=minimum_memory,
        maximum_swap_growth_mib=swap_growth,
        normalized_flags=_normalized_flags(model),
        run_nonce=run_nonce,
        plan_integrity_hash=str(plan["integrity_hash"]),
        run_completed_at=run_completed_at,
    )


def _project_frozen_cell(cell: FrozenCell) -> CellProjection:
    """Project raw artifacts only when they still bind to the frozen cell."""

    projection = project_completed_cell(cell.run_dir)
    if (
        projection.profile_id != cell.profile_id
        or projection.plan_fingerprint != cell.plan_fingerprint
        or projection.plan_integrity_hash != cell.plan_integrity_hash
        or projection.run_nonce != cell.run_nonce
        or projection.run_completed_at is None
    ):
        raise CellProjectionError(
            "completed cell projection does not match the frozen schedule",
            failure_kind="audit",
        )
    return projection


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


def _required_preflight_memavailable_kib(campaign: FrozenCampaign) -> int:
    calibration = campaign.cells_for(
        candidate_id="control", stage="calibration"
    )
    model = _bound_frozen_cell_plan_model(calibration["control_a"])
    estimated = model.get("estimated_ram_gib")
    if (
        isinstance(estimated, bool)
        or not isinstance(estimated, (int, float))
        or not math.isfinite(float(estimated))
        or float(estimated) <= 0
    ):
        raise CampaignPlanningError("calibration control RAM estimate is missing")
    return int((float(estimated) + 8.0) * 1024**2)


def _campaign_admission_binding(campaign: FrozenCampaign) -> AdmissionBinding:
    try:
        return AdmissionBinding(
            campaign_id=campaign.campaign_id,
            campaign_integrity_sha256=campaign.integrity_hash,
            preview_sha256=campaign.preview_digest,
            policy_sha256=campaign.policy_digest,
            cutoff_at=campaign.cutoff.isoformat(),
            required_remaining_s=float(PAIR_ADMISSION_REMAINING_S),
            required_memavailable_kib=_required_preflight_memavailable_kib(
                campaign
            ),
            max_starting_swap_kib=HOST_SAFETY_MAX_STARTING_SWAP_KIB,
            harness_sha256=campaign.harness_tree_sha256,
            harness_file_count=campaign.harness_file_count,
        )
    except AdmissionJournalError as error:
        raise CampaignPlanningError(
            "frozen campaign admission binding is invalid"
        ) from error


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
    required_kib = _required_preflight_memavailable_kib(campaign)
    if sample.memavailable_kib < required_kib:
        blockers.append("insufficient_preflight_memavailable")
    return tuple(blockers)


def _next_admission_target(
    campaign: FrozenCampaign,
    state: ReplayState | None,
    events: tuple[dict[str, Any], ...],
    *,
    calibration_ready: bool,
) -> AdmissionTarget | None:
    """Return the exact pair whose next cell could launch this invocation."""

    if not calibration_ready:
        if state is not None and not (
            state.phase == "idle"
            and state.next_pair_index == 0
            and len(events) == 1
            and events[0].get("event") == "autoresearch_campaign_started"
        ):
            raise CampaignPlanningError(
                "calibration is missing after search controller activity"
            )
        return AdmissionTarget(
            kind="calibration", candidate_id="control", pair_index=0
        )
    if state is None:
        raise CampaignPlanningError(
            "recorded calibration has no controller admission boundary"
        )
    if state.phase == "pair":
        if (
            state.candidate_id is None
            or state.active_pair_index is None
        ):
            raise CampaignPlanningError("active pair has no frozen identity")
        if state.completed_arms == state.active_order:
            return None
        return AdmissionTarget(
            kind=(
                "screen"
                if len(state.candidate_observations) == 0
                else "confirmation"
            ),
            candidate_id=state.candidate_id,
            pair_index=state.active_pair_index,
        )
    if state.phase in {"scored", "terminal"}:
        return None
    if state.phase == "candidate":
        if state.candidate_id is None:
            raise CampaignPlanningError("active candidate has no ID")
        return AdmissionTarget(
            kind=(
                "screen"
                if len(state.candidate_observations) == 0
                else "confirmation"
            ),
            candidate_id=state.candidate_id,
            pair_index=state.next_pair_index,
        )
    if state.phase != "idle":
        raise CampaignPlanningError(
            "controller is not at a pair admission boundary"
        )
    decisions = _candidate_decisions(events)
    if any(
        decision in {"promote", "promote_simplification"}
        for decision in decisions.values()
    ):
        return None
    proposal = next(
        (
            item
            for item in campaign.proposals
            if item.candidate_id not in decisions
        ),
        None,
    )
    if proposal is None:
        return None
    return AdmissionTarget(
        kind="screen",
        candidate_id=proposal.candidate_id,
        pair_index=state.next_pair_index,
    )


def _append_live_admission(
    campaign: FrozenCampaign,
    *,
    target: AdmissionTarget,
    events: tuple[dict[str, Any], ...],
    workspace: Path,
    now: Callable[[], datetime],
    meminfo_reader: Callable[[], str],
    harness_identity_reader: Callable[[Path], tuple[str, int]],
) -> dict[str, Any]:
    binding = _campaign_admission_binding(campaign)
    _read_campaign_admissions(campaign, events=events, binding=binding)
    try:
        sample = parse_meminfo(meminfo_reader())
    except (OSError, ValueError) as error:
        raise CampaignPlanningError(
            "campaign host-memory admission failed closed"
        ) from error
    try:
        harness_sha256, harness_file_count = harness_identity_reader(workspace)
    except (OSError, ValueError) as error:
        raise CampaignPlanningError(
            "campaign harness identity could not be verified"
        ) from error
    observed_at = now()
    if (
        not isinstance(observed_at, datetime)
        or observed_at.tzinfo is None
        or observed_at.utcoffset() is None
    ):
        raise CampaignPlanningError(
            "campaign admission time must be timezone-aware"
        )
    try:
        observation = observe_admission(
            binding,
            observed_at=observed_at,
            memavailable_kib=sample.memavailable_kib,
            swap_used_kib=sample.swap_used_kib,
            observed_harness_sha256=harness_sha256,
            observed_harness_file_count=harness_file_count,
        )
        appended = append_admission_record(
            campaign.campaign_dir / "admissions.jsonl",
            binding=binding,
            target=target,
            observation=observation,
            controller_events=events,
        )
        admissions = _read_campaign_admissions(
            campaign, events=events, binding=binding
        )
        if not admissions or admissions[-1] != appended:
            raise AdmissionJournalError(
                "appended admission is not the verified journal tail"
            )
        return appended
    except AdmissionJournalError as error:
        raise CampaignPlanningError(
            "campaign admission journal is unsafe or malformed"
        ) from error


def _admission_failure_kind(blockers: tuple[str, ...]) -> str:
    for blocker, failure_kind in (
        ("harness_code_changed", "audit"),
        ("starting_swap_above_clean_limit", "swap_pressure"),
        ("insufficient_preflight_memavailable", "memory_pressure"),
        ("insufficient_time_for_pair", "cutoff"),
    ):
        if blocker in blockers:
            return failure_kind
    raise CampaignPlanningError("blocked admission has no known causal blocker")


def _historical_search_admission_target(
    campaign: FrozenCampaign,
    *,
    events: tuple[dict[str, Any], ...],
) -> AdmissionTarget | None:
    if not events:
        return None
    state = _replay_frozen_controller(campaign, events)
    if state.phase == "pair":
        if state.candidate_id is None or state.active_pair_index is None:
            raise CampaignPlanningError("historical admission pair has no identity")
        if state.completed_arms == state.active_order:
            return None
        return AdmissionTarget(
            kind=(
                "screen"
                if len(state.candidate_observations) == 0
                else "confirmation"
            ),
            candidate_id=state.candidate_id,
            pair_index=state.active_pair_index,
        )
    if state.phase == "candidate":
        if state.candidate_id is None:
            raise CampaignPlanningError(
                "historical admission candidate has no identity"
            )
        return AdmissionTarget(
            kind=(
                "screen"
                if len(state.candidate_observations) == 0
                else "confirmation"
            ),
            candidate_id=state.candidate_id,
            pair_index=state.next_pair_index,
        )
    if state.phase != "idle":
        return None
    decisions = _candidate_decisions(events)
    if any(
        decision in {"promote", "promote_simplification"}
        for decision in decisions.values()
    ):
        return None
    proposal = next(
        (
            item
            for item in campaign.proposals
            if item.candidate_id not in decisions
        ),
        None,
    )
    if proposal is None:
        return None
    return AdmissionTarget(
        kind="screen",
        candidate_id=proposal.candidate_id,
        pair_index=state.next_pair_index,
    )


def _validate_admission_target_history(
    campaign: FrozenCampaign,
    admissions: tuple[dict[str, Any], ...],
    *,
    events: tuple[dict[str, Any], ...],
) -> None:
    search_seen = False
    for record in admissions:
        try:
            target = AdmissionTarget(
                kind=record["target_kind"],
                candidate_id=record["candidate_id"],
                pair_index=record["pair_index"],
            )
        except (AdmissionJournalError, KeyError) as error:
            raise CampaignPlanningError(
                "campaign admission target is malformed"
            ) from error
        count = record["controller_event_count"]
        prefix = events[:count]
        if target.kind == "calibration":
            if search_seen:
                raise CampaignPlanningError(
                    "calibration admission follows a search admission"
                )
            if prefix and not (
                len(prefix) == 1
                and prefix[0].get("event")
                == "autoresearch_campaign_started"
            ):
                raise CampaignPlanningError(
                    "calibration admission controller prefix is invalid"
                )
            continue
        search_seen = True
        try:
            expected = _historical_search_admission_target(
                campaign, events=prefix
            )
        except AdmissionJournalError as error:
            raise CampaignPlanningError(
                "campaign admission target binding is invalid"
            ) from error
        if expected != target:
            raise CampaignPlanningError(
                "campaign admission target does not match its controller prefix"
            )


def _admission_record_target(record: Mapping[str, Any]) -> AdmissionTarget:
    try:
        return AdmissionTarget(
            kind=record["target_kind"],
            candidate_id=record["candidate_id"],
            pair_index=record["pair_index"],
        )
    except (AdmissionJournalError, KeyError) as error:
        raise CampaignPlanningError(
            "campaign admission target is malformed"
        ) from error


def _admission_observed_at(record: Mapping[str, Any]) -> datetime:
    raw = record.get("observed_at")
    if not isinstance(raw, str):
        raise CampaignPlanningError("campaign admission timestamp is malformed")
    try:
        observed = datetime.fromisoformat(raw)
    except ValueError as error:
        raise CampaignPlanningError(
            "campaign admission timestamp is malformed"
        ) from error
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise CampaignPlanningError(
            "campaign admission timestamp has no timezone"
        )
    return observed


def _validate_admission_execution_coverage(
    campaign: FrozenCampaign,
    admissions: tuple[dict[str, Any], ...],
    *,
    events: tuple[dict[str, Any], ...],
) -> None:
    if not campaign.admission_journal_required:
        return
    calibration_cells = campaign.cells_for(
        candidate_id="control", stage="calibration"
    )
    calibration_raw = tuple(
        cell
        for cell in calibration_cells.values()
        if _safe_cell_journal_size(cell.run_dir / "events.jsonl") > 0
    )
    calibration_exists = _calibration_path(campaign).exists()
    if calibration_raw or calibration_exists:
        admitted_calibrations = tuple(
            record
            for record in admissions
            if record["outcome"] == "admitted"
            and _admission_record_target(record).kind == "calibration"
        )
        if not admitted_calibrations:
            raise CampaignPlanningError(
                "calibration execution has no admitted provenance"
            )
        if calibration_raw:
            first_start = min(_run_started_at(cell) for cell in calibration_raw)
            if not any(
                _admission_observed_at(record) <= first_start
                for record in admitted_calibrations
            ):
                raise CampaignPlanningError(
                    "calibration admission does not precede execution"
                )

    for event_index, event in enumerate(events):
        if event.get("event") != "autoresearch_pair_started":
            continue
        expected = _historical_search_admission_target(
            campaign, events=events[:event_index]
        )
        if expected is None:
            raise CampaignPlanningError(
                "search pair start has no valid admission target"
            )
        raw_timestamp = event.get("timestamp")
        if not isinstance(raw_timestamp, str):
            raise CampaignPlanningError("search pair start timestamp is malformed")
        try:
            pair_started_at = datetime.fromisoformat(raw_timestamp)
        except ValueError as error:
            raise CampaignPlanningError(
                "search pair start timestamp is malformed"
            ) from error
        if pair_started_at.tzinfo is None or pair_started_at.utcoffset() is None:
            raise CampaignPlanningError(
                "search pair start timestamp has no timezone"
            )
        covered = any(
            record["outcome"] == "admitted"
            and record["controller_event_count"] <= event_index
            and _admission_record_target(record) == expected
            and _admission_observed_at(record) <= pair_started_at
            for record in admissions
        )
        if not covered:
            raise CampaignPlanningError(
                "search pair execution has no admitted provenance"
            )


def _read_campaign_admissions(
    campaign: FrozenCampaign,
    *,
    events: tuple[dict[str, Any], ...],
    binding: AdmissionBinding | None = None,
) -> tuple[dict[str, Any], ...]:
    active_binding = binding or _campaign_admission_binding(campaign)
    try:
        admissions = read_admission_journal(
            campaign.campaign_dir / "admissions.jsonl",
            binding=active_binding,
            controller_events=events,
        )
    except AdmissionJournalError as error:
        raise CampaignPlanningError(
            "campaign admission journal is unsafe or malformed"
        ) from error
    _validate_admission_target_history(
        campaign, admissions, events=events
    )
    _validate_admission_execution_coverage(
        campaign, admissions, events=events
    )
    return admissions


def _cell_run_identity(cell: FrozenCell) -> str:
    return f"{cell.plan_fingerprint}-{cell.run_nonce}"


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


@contextmanager
def _private_append_log(path: Path) -> Iterator[Any]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise CellProjectionError("controller log requires no-follow support")
    try:
        descriptor = os.open(path, flags | nofollow, 0o600)
    except OSError as error:
        raise CellProjectionError("controller log path is unsafe") from error
    stream = None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
        ):
            raise CellProjectionError("controller log must be a private regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            os.fchmod(descriptor, 0o600)
        stream = os.fdopen(descriptor, "a", encoding="utf-8")
        descriptor = -1
        yield stream
    finally:
        if stream is not None:
            stream.close()
        if descriptor >= 0:
            os.close(descriptor)


def _safe_cell_journal_size(path: Path) -> int:
    """Return a fresh cell journal size without following links."""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return 0
    except OSError as error:
        raise CellProjectionError(
            "cell journal topology could not be inspected", failure_kind="audit"
        ) from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
    ):
        raise CellProjectionError(
            "cell journal must be an owned single-link regular file",
            failure_kind="audit",
        )
    return metadata.st_size


_WORKER_OWNERSHIP_FAILURES = frozenset(
    {
        "identity_mismatch",
        "ownership_ambiguous",
        "process_group_probe_failed",
        "process_lookup_race",
        "proc_stat_malformed",
        "proc_stat_unreadable",
        "run_nonce_mismatch",
        "worker_state_changed",
        "worker_state_malformed",
        "worker_state_unreadable",
        "worker_state_unsafe",
    }
)


def _worker_failure_kind(error: WorkerLifecycleError) -> str:
    if error.code in _WORKER_OWNERSHIP_FAILURES:
        return "ownership_ambiguity"
    if error.code == "worker_progress_measurement_budget":
        return "measurement"
    if error.code == "worker_progress_cleanup_budget":
        return "cleanup_breach"
    if error.code.startswith("worker_progress_"):
        return "audit"
    return "cleanup_breach"


def run_frozen_cell(
    cell: FrozenCell,
    *,
    workspace: Path,
    cell_timeout_s: int,
    cleanup_timeout_s: int,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    worker_runner: Callable[..., WorkerRunResult] = run_owned_worker,
    monotonic: Callable[[], float] = time.monotonic,
) -> CellProjection:
    """Execute one pristine plan with SIGINT unwind and exact owned recovery."""

    blocker = model_execution_blocker(_bound_frozen_cell_plan_model(cell))
    if blocker is not None:
        raise CellProjectionError(blocker, failure_kind="audit")
    events_path = cell.run_dir / "events.jsonl"
    if _safe_cell_journal_size(events_path):
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
    progress_probe = _CellLifecycleProgress(
        events_path=events_path,
        plan_fingerprint=cell.plan_fingerprint,
        run_nonce=cell.run_nonce,
        measurement_timeout_s=cell_timeout_s,
        cleanup_timeout_s=cleanup_timeout_s,
    )
    started = monotonic()
    with _private_append_log(log_path) as log:
        try:
            worker = worker_runner(
                command,
                cell_run_dir=cell.run_dir,
                run_nonce=cell.run_nonce,
                timeout_s=cell_timeout_s,
                interrupt_grace_s=cleanup_timeout_s,
                popen_kwargs={
                    "cwd": str(workspace),
                    "stdin": subprocess.DEVNULL,
                    "stdout": log,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                },
                popen_factory=popen_factory,
                progress_probe=progress_probe,
                start_marker_timeout_s=START_MARKER_TIMEOUT_S,
            )
        except BaseException as error:
            try:
                _recover_cell(cell)
            except BaseException as recovery_error:
                raise CellProjectionError(
                    "cell worker and exact server recovery could not be certified",
                    failure_kind="cleanup_breach",
                ) from recovery_error
            if isinstance(error, WorkerLifecycleError):
                raise CellProjectionError(
                    "cell worker ownership, progress, or cleanup failed",
                    failure_kind=_worker_failure_kind(error),
                ) from error
            raise
    wall_s = monotonic() - started
    if wall_s < 0:
        raise CellProjectionError("cell monotonic clock moved backwards")
    if worker.timed_out:
        _recover_cell(cell)
        if (
            worker.timeout_phase in {"cleanup", "finalization"}
            or worker.cleanup.outcome == "killed"
        ):
            raise CellProjectionError(
                "cell exceeded timeout and owned cleanup grace",
                failure_kind="cleanup_breach",
            )
        raise CellProjectionError(
            "cell exceeded its causal timeout and is invalid",
            failure_kind="measurement",
        )
    if worker.return_code != 0:
        _recover_cell(cell)
        try:
            project_completed_cell(cell.run_dir)
        except CellProjectionError as error:
            raise error
        raise CellProjectionError("cell process returned a nonzero status")
    try:
        projection = project_completed_cell(cell.run_dir)
    except CellProjectionError:
        _recover_cell(cell)
        raise
    if projection.profile_id != cell.profile_id:
        raise CellProjectionError("completed cell profile binding changed")
    return projection


def _recover_interrupted_cells(campaign: FrozenCampaign) -> None:
    """Recover every owned worker/container before inspecting raw semantics."""

    worker_recovered: dict[str, bool] = {}
    container_recovered: dict[str, bool] = {}
    journal_sizes: dict[str, int | None] = {}
    failures: list[tuple[str, BaseException]] = []
    container_mismatches: list[CellProjectionError] = []
    exact_container_stopped = False

    # Never let one ambiguous worker prevent cleanup attempts for later cells.
    for cell in campaign.cells:
        try:
            cleanup = recover_owned_worker(
                cell.run_dir,
                expected_run_nonce=cell.run_nonce,
                interrupt_grace_s=campaign.policy.cleanup_timeout_s,
            )
        except WorkerLifecycleError as error:
            worker_recovered[cell.cell_id] = True
            failures.append((_worker_failure_kind(error), error))
        else:
            worker_recovered[cell.cell_id] = cleanup.outcome != "no_state"

    # Journal topology is evidence for exact container cleanup, not permission
    # to semantically project a possibly incomplete cell during recovery.
    for cell in campaign.cells:
        try:
            journal_sizes[cell.cell_id] = _safe_cell_journal_size(
                cell.run_dir / "events.jsonl"
            )
        except CellProjectionError as error:
            journal_sizes[cell.cell_id] = None
            failures.append((error.failure_kind, error))

    for cell in campaign.cells:
        container_recovered[cell.cell_id] = False
        try:
            outcome = _recover_cell(cell)
        except CellProjectionError as error:
            if error.failure_kind == "ownership_ambiguity":
                if exact_container_stopped:
                    failures.append((error.failure_kind, error))
                else:
                    container_mismatches.append(error)
            else:
                failures.append((error.failure_kind, error))
        else:
            if outcome == "stopped_owned_container":
                container_recovered[cell.cell_id] = True
                exact_container_stopped = True
                container_mismatches.clear()

    # SGLang has one fixed container name. Probing that sole container against
    # an earlier frozen identity can mismatch before a later exact identity
    # stops it. Keep mismatches terminal unless this campaign proves ownership
    # by stopping the exact named container during the same exhaustive pass.
    failures.extend(
        ("ownership_ambiguity", error) for error in container_mismatches
    )

    for cell in campaign.cells:
        if (
            worker_recovered[cell.cell_id] or container_recovered[cell.cell_id]
        ) and journal_sizes[cell.cell_id] == 0:
            failures.append(
                (
                    "measurement",
                    CellProjectionError(
                        "interrupted cell exited before a durable measurement journal"
                    ),
                )
            )

    if failures:
        priority = {
            "measurement": 0,
            "audit": 1,
            "cleanup_breach": 2,
            "ownership_ambiguity": 3,
        }
        failure_kind, cause = max(
            failures, key=lambda item: priority.get(item[0], 0)
        )
        raise CellProjectionError(
            f"interrupted cleanup failed for {len(failures)} frozen cell condition(s)",
            failure_kind=failure_kind,
        ) from cause


def _validate_search_artifact_admission(
    campaign: FrozenCampaign, events: tuple[dict[str, Any], ...]
) -> None:
    """Reject raw search cells that have no preceding durable pair start."""

    admitted: set[str] = set()
    occurrences: dict[str, int] = {}
    for event in _deduplicated_controller_events(events):
        if event.get("event") != "autoresearch_pair_started":
            continue
        candidate_id = event.get("candidate_id")
        if not isinstance(candidate_id, str):
            raise CampaignPlanningError("controller pair candidate is malformed")
        occurrence = occurrences.get(candidate_id, 0)
        stage = "screen" if occurrence == 0 else "confirmation"
        if occurrence > 1:
            raise CampaignPlanningError(
                "controller candidate exceeds screen and confirmation"
            )
        admitted.update(
            cell.cell_id
            for cell in campaign.cells_for(
                candidate_id=candidate_id, stage=stage
            ).values()
        )
        occurrences[candidate_id] = occurrence + 1
    for cell in campaign.cells:
        if cell.stage == "calibration" or cell.cell_id in admitted:
            continue
        if _safe_cell_journal_size(cell.run_dir / "events.jsonl"):
            raise CellProjectionError(
                "raw search cell has no preceding durable pair admission",
                failure_kind="audit",
            )


def _bound_transition_id(
    campaign: FrozenCampaign, event: Mapping[str, Any]
) -> str:
    """Bind one timestamp-free controller transition to this exact freeze."""

    name = event.get("event")
    if not isinstance(name, str) or not name.startswith("autoresearch_"):
        raise CampaignPlanningError("controller transition has no stable event name")
    payload = {
        key: value
        for key, value in event.items()
        if key not in {"timestamp", "transition_id"}
    }
    kind = name.removeprefix("autoresearch_").replace("_", "-")
    digest = content_hash(
        {
            "campaign_integrity_hash": campaign.integrity_hash,
            "preview_digest": campaign.preview_digest,
            "event": payload,
        },
        64,
    )
    return f"{kind}-{digest}"


def _deduplicated_controller_events(
    events: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], ...]:
    unique: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for event in events:
        transition_id = event.get("transition_id")
        if not isinstance(transition_id, str):
            raise CampaignPlanningError("controller transition ID is malformed")
        payload = canonical_json(
            {key: value for key, value in event.items() if key != "timestamp"}
        )
        prior = seen.get(transition_id)
        if prior is not None:
            if prior != payload:
                raise CampaignPlanningError("controller transition ID was reused")
            continue
        seen[transition_id] = payload
        unique.append(event)
    return tuple(unique)


def _durable_pair_audit_reserve_s(
    campaign: FrozenCampaign, projections: Mapping[str, CellProjection]
) -> float:
    """Derive score-time reserve from the pair's durable final completion."""

    if set(projections) != {"champion", "candidate"}:
        raise CampaignPlanningError("score-bearing pair projections are incomplete")
    completed_at = tuple(
        projection.run_completed_at for projection in projections.values()
    )
    if any(value is None for value in completed_at):
        raise CampaignPlanningError("score-bearing projection has no completion time")
    last_completed = max(value for value in completed_at if value is not None)
    return max(0.0, (campaign.cutoff - last_completed).total_seconds())


def _replay_frozen_campaign(
    campaign: FrozenCampaign,
    events: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    verify_raw: bool = True,
) -> ReplayState:
    """Replay plus exact frozen-instance and queue bindings."""

    records = tuple(events)
    state = replay_transitions(campaign.policy, records)
    if not records:
        return state
    unique = _deduplicated_controller_events(records)
    if not unique or unique[0].get("event") != "autoresearch_campaign_started":
        raise CampaignPlanningError("controller journal has no first campaign start")

    proposal_cursor = 0
    pair_counts: dict[str, int] = {}
    pair_stages: dict[tuple[str, int], str] = {}
    pair_cells: dict[tuple[str, int], dict[str, FrozenCell]] = {}
    pair_projections: dict[
        tuple[str, int], dict[str, CellProjection]
    ] = {}
    started_candidates: set[str] = set()
    for event in unique:
        expected_id = _bound_transition_id(campaign, event)
        if event.get("transition_id") != expected_id:
            raise CampaignPlanningError(
                "controller transition is not bound to this frozen campaign"
            )
        name = event.get("event")
        if name == "autoresearch_campaign_started":
            if (
                event.get("campaign_id") != campaign.campaign_id
                or event.get("policy_digest") != campaign.policy_digest
            ):
                raise CampaignPlanningError(
                    "controller start does not match the frozen campaign"
                )
            continue
        if name == "autoresearch_candidate_started":
            if proposal_cursor >= len(campaign.proposals):
                raise CampaignPlanningError("controller started an extra candidate")
            proposal = campaign.proposals[proposal_cursor]
            if (
                event.get("candidate_id") != proposal.candidate_id
                or event.get("axis") != proposal.axis
                or event.get("delta_digest") != proposal.delta.digest
            ):
                raise CampaignPlanningError(
                    "controller candidate is not the next frozen proposal"
                )
            if proposal.candidate_id in started_candidates:
                raise CampaignPlanningError("controller candidate was started twice")
            started_candidates.add(proposal.candidate_id)
            proposal_cursor += 1
            continue
        if name == "autoresearch_pair_started":
            candidate_id = event.get("candidate_id")
            pair_index = event.get("pair_index")
            if not isinstance(candidate_id, str) or not isinstance(pair_index, int):
                raise CampaignPlanningError("controller pair identity is malformed")
            occurrence = pair_counts.get(candidate_id, 0)
            if occurrence > 1:
                raise CampaignPlanningError(
                    "controller candidate exceeds screen and confirmation"
                )
            stage = "screen" if occurrence == 0 else "confirmation"
            cells = campaign.cells_for(candidate_id=candidate_id, stage=stage)
            pair_counts[candidate_id] = occurrence + 1
            pair_key = (candidate_id, pair_index)
            pair_stages[pair_key] = stage
            pair_cells[pair_key] = cells
            pair_projections[pair_key] = {}
            continue
        if name in {"autoresearch_cell_completed", "autoresearch_pair_scored"}:
            candidate_id = event.get("candidate_id")
            pair_index = event.get("pair_index")
            if not isinstance(candidate_id, str) or not isinstance(pair_index, int):
                raise CampaignPlanningError("controller pair binding is malformed")
            pair_key = (candidate_id, pair_index)
            stage = pair_stages.get(pair_key)
            if stage is None:
                raise CampaignPlanningError(
                    "controller completion has no frozen pair occurrence"
                )
            cells = pair_cells[pair_key]
            projections = pair_projections[pair_key]
            if name == "autoresearch_cell_completed":
                arm = event.get("arm")
                if not isinstance(arm, str):
                    raise CampaignPlanningError("controller cell arm is malformed")
                if arm not in cells:
                    raise CampaignPlanningError(
                        "controller cell does not map to the frozen schedule"
                    )
                if verify_raw:
                    projections[arm] = _project_frozen_cell(cells[arm])
                if verify_raw and set(projections) == {"champion", "candidate"}:
                    _validate_completed_pair_gap(
                        campaign, cells, pair_index=pair_index
                    )
                continue
            if not verify_raw:
                continue
            if set(projections) != {"champion", "candidate"}:
                raise CampaignPlanningError(
                    "controller score has no complete raw frozen pair"
                )
            expected_observation = observation_from_cells(
                campaign.policy,
                pair_index=pair_index,
                champion=projections["champion"],
                candidate=projections["candidate"],
                audit_reserve_remaining_s=_durable_pair_audit_reserve_s(
                    campaign, projections
                ),
            )
            if canonical_json(event.get("observation")) != canonical_json(
                expected_observation.to_mapping()
            ):
                raise CampaignPlanningError(
                    "controller score does not match the frozen raw pair"
                )

    return state


def _replay_frozen_controller(
    campaign: FrozenCampaign,
    events: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> ReplayState:
    """Replay exact frozen controller bindings without dereferencing raw cells."""

    return _replay_frozen_campaign(campaign, events, verify_raw=False)


def _append_frozen_transition(
    journal: Journal,
    campaign: FrozenCampaign,
    event: Mapping[str, Any],
) -> ReplayState:
    """Append one exact-freeze-bound transition and audit the resulting journal."""

    existing = _controller_events(journal)
    _replay_frozen_campaign(campaign, existing)
    payload = {
        key: value
        for key, value in event.items()
        if key not in {"timestamp", "transition_id"}
    }
    payload["transition_id"] = _bound_transition_id(campaign, payload)
    _replay_frozen_campaign(campaign, [*existing, payload])
    append_transition(journal, campaign.policy, payload)
    return _replay_frozen_campaign(campaign, _controller_events(journal))


def _controller_events(journal: Journal) -> list[dict[str, Any]]:
    """Read controller truth without silently filtering durable corruption."""

    try:
        return journal.strict_events(max_bytes=MAX_PROGRESS_JOURNAL_BYTES)
    except (OSError, ValueError) as error:
        raise CampaignPlanningError("controller journal is unsafe or malformed") from error


def _transition_id(*parts: object) -> str:
    value = "-".join(str(part).lower().replace("_", "-") for part in parts)
    if len(value) <= 128:
        return value
    return value[:111] + "-" + content_hash(value, 16)


def _append_campaign_started(
    journal: Journal, campaign: FrozenCampaign
) -> None:
    if _controller_events(journal):
        _replay_frozen_campaign(campaign, _controller_events(journal))
        return
    _append_frozen_transition(
        journal,
        campaign,
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
    state = _replay_frozen_campaign(campaign, _controller_events(journal))
    if state.phase == "terminal":
        return
    _append_frozen_transition(
        journal,
        campaign,
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


def _calibration_record_payload(
    observation: PairObservation,
    *,
    passed: bool,
    reasons: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "observation": observation.to_mapping(),
        "passed": passed,
        "reasons": list(reasons),
    }


def _write_calibration(
    campaign: FrozenCampaign,
    observation: PairObservation,
    *,
    passed: bool,
    reasons: tuple[str, ...],
) -> None:
    record = _calibration_record_payload(
        observation, passed=passed, reasons=reasons
    )
    record["integrity_hash"] = content_hash(record, 64)
    write_json(_calibration_path(campaign), record)


def _calibration_raw_topology(
    campaign: FrozenCampaign,
) -> tuple[dict[str, FrozenCell], dict[str, CellProjection]]:
    """Project the valid one-use calibration prefix without launching work."""

    cells = campaign.cells_for(candidate_id="control", stage="calibration")
    journal_sizes = {
        arm: _safe_cell_journal_size(cells[arm].run_dir / "events.jsonl")
        for arm in ("control_a", "control_b")
    }
    if journal_sizes["control_b"] and not journal_sizes["control_a"]:
        raise CampaignPlanningError(
            "calibration control B exists without its control A predecessor"
        )
    projections: dict[str, CellProjection] = {}
    for arm in ("control_a", "control_b"):
        if journal_sizes[arm]:
            projections[arm] = _project_frozen_cell(cells[arm])
    return cells, projections


def _recomputed_calibration(
    campaign: FrozenCampaign,
    cells: Mapping[str, FrozenCell],
    projections: Mapping[str, CellProjection],
) -> tuple[PairObservation, bool, tuple[str, ...]]:
    if set(cells) != {"control_a", "control_b"} or set(projections) != {
        "control_a",
        "control_b",
    }:
        raise CampaignPlanningError(
            "calibration requires both exact frozen raw controls"
        )
    gap_s = (
        _run_started_at(cells["control_b"])
        - _server_stopped_at(cells["control_a"])
    ).total_seconds()
    if gap_s < 0 or gap_s > campaign.policy.cleanup_timeout_s:
        raise CellProjectionError(
            "calibration inter-cell gap exceeded the frozen cleanup bound"
        )
    score_projections = {
        "champion": projections["control_a"],
        "candidate": projections["control_b"],
    }
    observation = observation_from_cells(
        campaign.policy,
        pair_index=0,
        champion=score_projections["champion"],
        candidate=score_projections["candidate"],
        audit_reserve_remaining_s=_durable_pair_audit_reserve_s(
            campaign, score_projections
        ),
    )
    decision = evaluate_calibration(campaign.policy, observation)
    return observation, decision.passed, decision.reasons


def _verify_calibration_record(
    campaign: FrozenCampaign,
) -> tuple[PairObservation, bool] | None:
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
    cells, projections = _calibration_raw_topology(campaign)
    observation, passed, reasons = _recomputed_calibration(
        campaign, cells, projections
    )
    expected = _calibration_record_payload(
        observation, passed=passed, reasons=reasons
    )
    if canonical_json(record) != canonical_json(expected):
        raise CampaignPlanningError(
            "calibration record does not match its exact frozen raw controls"
        )
    return observation, passed


def _load_calibration(campaign: FrozenCampaign) -> PairObservation | None:
    verified = _verify_calibration_record(campaign)
    if verified is None:
        return None
    observation, passed = verified
    if not passed:
        raise CellProjectionError("control-to-control calibration did not pass")
    return observation


def _reconcile_raw_calibration(campaign: FrozenCampaign) -> bool:
    """Write a missing record from two raw-complete controls, without inference."""

    if _load_calibration(campaign) is not None:
        return False
    cells, projections = _calibration_raw_topology(campaign)
    if set(projections) != {"control_a", "control_b"}:
        return False
    observation, passed, reasons = _recomputed_calibration(
        campaign, cells, projections
    )
    _write_calibration(
        campaign, observation, passed=passed, reasons=reasons
    )
    if not passed:
        raise CellProjectionError("control-to-control calibration did not pass")
    return True


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
    cells, projections = _calibration_raw_topology(campaign)
    if set(projections) == {"control_a", "control_b"}:
        observation, passed, reasons = _recomputed_calibration(
            campaign, cells, projections
        )
        _write_calibration(
            campaign, observation, passed=passed, reasons=reasons
        )
        if not passed:
            raise CellProjectionError("control-to-control calibration did not pass")
        return observation
    if "control_a" not in projections:
        launched_a = cell_runner(cells["control_a"])
        if (
            launched_a.profile_id != cells["control_a"].profile_id
            or launched_a.plan_fingerprint != cells["control_a"].plan_fingerprint
        ):
            raise CellProjectionError("calibration control A does not match schedule")
        projections["control_a"] = _project_frozen_cell(cells["control_a"])
    control_a = projections["control_a"]
    if (
        control_a.measurement_elapsed_s > campaign.policy.cell_timeout_s
        or control_a.cleanup_elapsed_s > campaign.policy.cleanup_timeout_s
    ):
        raise CellProjectionError(
            "first calibration cell exceeded its measurement or cleanup budget"
        )
    control_a_stopped = _server_stopped_at(cells["control_a"])
    calibration_gap_s = (now() - control_a_stopped).total_seconds()
    if calibration_gap_s < 0 or calibration_gap_s > campaign.policy.cleanup_timeout_s:
        raise CellProjectionError(
            "calibration inter-cell gap exceeded the frozen cleanup bound"
        )
    launched_b = cell_runner(cells["control_b"])
    if (
        launched_b.profile_id != cells["control_b"].profile_id
        or launched_b.plan_fingerprint != cells["control_b"].plan_fingerprint
    ):
        raise CellProjectionError("calibration control B does not match schedule")
    projections["control_b"] = _project_frozen_cell(cells["control_b"])
    observation, passed, reasons = _recomputed_calibration(
        campaign, cells, projections
    )
    _write_calibration(
        campaign,
        observation,
        passed=passed,
        reasons=reasons,
    )
    if not passed:
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


def _cell_event_at(cell: FrozenCell, event_name: str) -> datetime:
    event = _one_event(
        _read_jsonl(cell.run_dir / "events.jsonl", context="cell event journal"),
        event_name,
    )
    raw = event.get("timestamp")
    if not isinstance(raw, str):
        raise CellProjectionError(f"{event_name} event has no timestamp")
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as error:
        raise CellProjectionError(f"{event_name} timestamp is invalid") from error
    if value.tzinfo is None or value.utcoffset() is None:
        raise CellProjectionError(
            f"{event_name} timestamp is not timezone-aware"
        )
    return value


def _server_stopped_at(cell: FrozenCell) -> datetime:
    return _cell_event_at(cell, "server_stopped")


def _run_started_at(cell: FrozenCell) -> datetime:
    return _cell_event_at(cell, "run_start")


def _validate_completed_pair_gap(
    campaign: FrozenCampaign,
    cells: Mapping[str, FrozenCell],
    *,
    pair_index: int,
) -> float:
    order = pair_order(pair_index)
    gap_s = (
        _run_started_at(cells[order[1]])
        - _server_stopped_at(cells[order[0]])
    ).total_seconds()
    if gap_s < 0 or gap_s > campaign.policy.cleanup_timeout_s:
        raise CellProjectionError(
            "inter-cell gap exceeded the frozen cleanup bound"
        )
    return gap_s


def _run_search_pair(
    campaign: FrozenCampaign,
    journal: Journal,
    *,
    candidate_id: str,
    stage: str,
    cell_runner: Callable[[FrozenCell], CellProjection],
    now: Callable[[], datetime],
) -> PairObservation:
    state = _replay_frozen_campaign(campaign, _controller_events(journal))
    cells = campaign.cells_for(candidate_id=candidate_id, stage=stage)
    if state.phase == "candidate":
        if any(
            _safe_cell_journal_size(cell.run_dir / "events.jsonl")
            for cell in cells.values()
        ):
            raise CellProjectionError(
                "raw search cell exists before its durable pair admission",
                failure_kind="audit",
            )
        _append_frozen_transition(
            journal,
            campaign,
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
        state = _replay_frozen_campaign(campaign, _controller_events(journal))
    if state.phase != "pair" or state.active_pair_index is None:
        raise CampaignPlanningError("candidate is not in an executable pair phase")
    raw_prefix = frozenset(_raw_search_prefix(campaign, state))
    projections: dict[str, CellProjection] = {}
    last_completion: datetime | None = None
    for completed_arm in state.completed_arms:
        projections[completed_arm] = _project_frozen_cell(cells[completed_arm])
        last_completion = _server_stopped_at(cells[completed_arm])
    for arm in state.active_order[len(state.completed_arms) :]:
        raw_complete = arm in raw_prefix
        if last_completion is not None:
            current = _run_started_at(cells[arm]) if raw_complete else now()
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
        if raw_complete:
            projection = _project_frozen_cell(cells[arm])
        else:
            projection = cell_runner(cells[arm])
            if (
                projection.profile_id != cells[arm].profile_id
                or projection.plan_fingerprint != cells[arm].plan_fingerprint
            ):
                raise CellProjectionError("cell projection does not match schedule")
        projections[arm] = _project_frozen_cell(cells[arm])
        _append_frozen_transition(
            journal,
            campaign,
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
        state = _replay_frozen_campaign(campaign, _controller_events(journal))
        last_completion = _server_stopped_at(cells[arm])
    observation = observation_from_cells(
        campaign.policy,
        pair_index=state.active_pair_index,
        champion=projections["champion"],
        candidate=projections["candidate"],
        audit_reserve_remaining_s=_durable_pair_audit_reserve_s(
            campaign, projections
        ),
    )
    _append_frozen_transition(
        journal,
        campaign,
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


def _raw_search_prefix(
    campaign: FrozenCampaign, state: ReplayState | None
) -> tuple[str, ...]:
    """Return the active ordered nonempty raw prefix without projecting it."""

    if (
        state is None
        or state.phase != "pair"
        or state.candidate_id is None
        or state.active_pair_index is None
    ):
        return ()
    stage = (
        "screen" if len(state.candidate_observations) == 0 else "confirmation"
    )
    cells = campaign.cells_for(candidate_id=state.candidate_id, stage=stage)
    missing_arms = state.active_order[len(state.completed_arms) :]
    raw_prefix: list[str] = []
    saw_pristine = False
    for arm in missing_arms:
        raw_complete = (
            _safe_cell_journal_size(cells[arm].run_dir / "events.jsonl") > 0
        )
        if not raw_complete:
            saw_pristine = True
            continue
        if saw_pristine:
            raise CellProjectionError(
                "later search arm has raw artifacts before its ordered predecessor",
                failure_kind="audit",
            )
        raw_prefix.append(arm)
    return tuple(raw_prefix)


def _reconcile_raw_search_prefix(
    campaign: FrozenCampaign,
    journal: Journal,
    state: ReplayState | None,
    *,
    raw_prefix: tuple[str, ...] | None = None,
) -> ReplayState | None:
    """Durably index an ordered raw-complete prefix without launching work."""

    if (
        state is None
        or state.phase != "pair"
        or state.candidate_id is None
        or state.active_pair_index is None
    ):
        return state
    stage = (
        "screen" if len(state.candidate_observations) == 0 else "confirmation"
    )
    cells = campaign.cells_for(candidate_id=state.candidate_id, stage=stage)
    prefix = _raw_search_prefix(campaign, state) if raw_prefix is None else raw_prefix
    for arm in prefix:
        _project_frozen_cell(cells[arm])
        state = _append_frozen_transition(
            journal,
            campaign,
            {
                "event": "autoresearch_cell_completed",
                "transition_id": _transition_id(
                    state.candidate_id,
                    "pair",
                    state.active_pair_index,
                    arm,
                    "completed",
                ),
                "candidate_id": state.candidate_id,
                "pair_index": state.active_pair_index,
                "arm": arm,
            },
        )
    return state


def _checkpoint_campaign_binding(campaign: FrozenCampaign) -> Any:
    from .autoresearch_checkpoint import CampaignBinding

    return CampaignBinding(
        campaign_id=campaign.campaign_id,
        campaign_integrity_sha256=campaign.integrity_hash,
        preview_sha256=campaign.preview_digest,
        policy_sha256=campaign.policy_digest,
    )


def _checkpoint_completion_for_cells(
    campaign: FrozenCampaign,
    *,
    sequence: int,
    pair_kind: str,
    candidate_id: str,
    search_pair_index: int | None,
    ordered_cells: tuple[FrozenCell, FrozenCell],
    observation: PairObservation,
) -> Any:
    from .autoresearch_checkpoint import (
        PairCompletion,
        autoresearch_published_run_id,
    )

    return PairCompletion(
        sequence=sequence,
        pair_kind=pair_kind,
        candidate_id=candidate_id,
        search_pair_index=search_pair_index,
        ordered_cell_ids=tuple(cell.cell_id for cell in ordered_cells),
        ordered_evidence_run_ids=tuple(
            autoresearch_published_run_id(
                campaign_id=campaign.campaign_id,
                cell_id=cell.cell_id,
                ordinal=cell.ordinal,
                created_at=campaign.created_at,
            )
            for cell in ordered_cells
        ),
        cell_plan_integrity_sha256s=tuple(
            cell.plan_integrity_hash for cell in ordered_cells
        ),
        observation_sha256=content_hash(observation.to_mapping(), 64),
    )


def _latest_checkpoint_completion(campaign: FrozenCampaign) -> Any:
    """Return the latest strictly raw-attested pair identity, if one exists."""

    verified_calibration = _verify_calibration_record(campaign)
    calibration_completion = None
    if verified_calibration is not None:
        calibration_observation, _passed = verified_calibration
        calibration_cells = campaign.cells_for(
            candidate_id="control", stage="calibration"
        )
        calibration_completion = _checkpoint_completion_for_cells(
            campaign,
            sequence=1,
            pair_kind="calibration",
            candidate_id="control",
            search_pair_index=None,
            ordered_cells=(
                calibration_cells["control_a"],
                calibration_cells["control_b"],
            ),
            observation=calibration_observation,
        )

    journal = Journal(campaign.campaign_dir / "events.jsonl")
    events = tuple(_controller_events(journal))
    if not events:
        return calibration_completion
    _replay_frozen_campaign(campaign, events)
    unique = _deduplicated_controller_events(events)
    pair_occurrences: dict[str, int] = {}
    pair_metadata: dict[tuple[str, int], tuple[str, tuple[str, str]]] = {}
    latest_score: dict[str, Any] | None = None
    for event in unique:
        name = event.get("event")
        if name == "autoresearch_pair_started":
            candidate_id = str(event["candidate_id"])
            pair_index = int(event["pair_index"])
            occurrence = pair_occurrences.get(candidate_id, 0)
            stage = "screen" if occurrence == 0 else "confirmation"
            pair_occurrences[candidate_id] = occurrence + 1
            order = tuple(event["order"])
            if len(order) != 2:
                raise CampaignPlanningError("checkpoint pair order is incomplete")
            pair_metadata[(candidate_id, pair_index)] = (stage, order)
        elif name == "autoresearch_pair_scored":
            latest_score = event

    if latest_score is None:
        return calibration_completion
    if verified_calibration is None or not verified_calibration[1]:
        raise CampaignPlanningError(
            "search score exists without a passing verified calibration"
        )
    candidate_id = str(latest_score["candidate_id"])
    pair_index = int(latest_score["pair_index"])
    try:
        stage, order = pair_metadata[(candidate_id, pair_index)]
    except KeyError as error:  # Strict replay should make this unreachable.
        raise CampaignPlanningError(
            "checkpoint score has no frozen pair occurrence"
        ) from error
    cells = campaign.cells_for(candidate_id=candidate_id, stage=stage)
    projections = {
        arm: _project_frozen_cell(cells[arm]) for arm in ("champion", "candidate")
    }
    _validate_completed_pair_gap(campaign, cells, pair_index=pair_index)
    observation = observation_from_cells(
        campaign.policy,
        pair_index=pair_index,
        champion=projections["champion"],
        candidate=projections["candidate"],
        audit_reserve_remaining_s=_durable_pair_audit_reserve_s(
            campaign, projections
        ),
    )
    if canonical_json(latest_score.get("observation")) != canonical_json(
        observation.to_mapping()
    ):
        raise CampaignPlanningError(
            "checkpoint score does not match the recomputed frozen raw pair"
        )
    return _checkpoint_completion_for_cells(
        campaign,
        sequence=pair_index + 2,
        pair_kind=stage,
        candidate_id=candidate_id,
        search_pair_index=pair_index,
        ordered_cells=(cells[order[0]], cells[order[1]]),
        observation=observation,
    )


def _checkpoint_gate_required(
    campaign: FrozenCampaign,
    state: ReplayState | None,
    events: tuple[dict[str, Any], ...],
) -> bool:
    calibration = _verify_calibration_record(campaign)
    if calibration is None or not calibration[1] or state is None:
        return False
    if state.phase == "candidate":
        return (
            len(state.candidate_observations) == 1
            and state.confirmation_mode in {"standard", "simplification"}
        )
    if state.phase != "idle":
        return False
    decisions = _candidate_decisions(events)
    if any(
        decision in {"promote", "promote_simplification"}
        for decision in decisions.values()
    ):
        return False
    return not all(
        decisions.get(proposal.candidate_id) == "reject"
        for proposal in campaign.proposals
    )


def _checkpoint_gate_for_campaign(campaign: FrozenCampaign, workspace: Path) -> Any:
    from .autoresearch_checkpoint import checkpoint_gate

    return checkpoint_gate(
        workspace=workspace,
        campaign=_checkpoint_campaign_binding(campaign),
        completion=_latest_checkpoint_completion(campaign),
        journal_path=campaign.campaign_dir / "events.jsonl",
    )


def _campaign_summary_payload(campaign: FrozenCampaign) -> dict[str, Any]:
    journal = Journal(campaign.campaign_dir / "events.jsonl")
    events = tuple(_controller_events(journal))
    state = _replay_frozen_campaign(campaign, events) if events else None
    calibration = _verify_calibration_record(campaign)
    binding = _campaign_admission_binding(campaign)
    try:
        admissions = _read_campaign_admissions(
            campaign, events=events, binding=binding
        )
        controller_hash = controller_prefix_sha256(events)
    except AdmissionJournalError as error:
        raise CampaignPlanningError(
            "campaign admission journal is unsafe or malformed"
        ) from error
    terminal_reason = state.terminal_reason if state else None
    controller_status = (
        "planned"
        if state is None
        else "complete"
        if state.phase == "terminal" and terminal_reason == "completed"
        else "terminated"
        if state.phase == "terminal"
        else "active"
    )
    last_admission = admissions[-1] if admissions else None
    status = controller_status
    if controller_status == "terminated" and terminal_reason == "cutoff":
        status = "expired"
    elif controller_status == "planned" and last_admission is not None:
        if last_admission["outcome"] == "cutoff":
            status = "expired"
        elif last_admission["outcome"] == "blocked_environment":
            status = "blocked_environment"
    return {
        "schema_version": 2,
        "campaign_id": campaign.campaign_id,
        "status": status,
        "controller_status": controller_status,
        "controller_phase": state.phase if state else "new",
        "terminal_reason": terminal_reason,
        "calibration_recorded": calibration is not None,
        "next_pair_index": state.next_pair_index if state else 0,
        "candidate_decisions": _candidate_decisions(events),
        "admission_count": len(admissions),
        "last_admission": last_admission,
        "last_admission_sha256": (
            last_admission["record_sha256"] if last_admission else None
        ),
        "blockers": list(last_admission["blockers"]) if last_admission else [],
        "campaign_integrity_sha256": campaign.integrity_hash,
        "preview_sha256": campaign.preview_digest,
        "policy_sha256": campaign.policy_digest,
        "controller_event_count": len(events),
        "controller_prefix_sha256": controller_hash,
        "policy_digest": campaign.policy_digest,
    }


def _is_legacy_blocked_campaign(campaign: FrozenCampaign) -> bool:
    seal = _LEGACY_BLOCKED_CAMPAIGN_SEAL
    return (
        campaign.campaign_id == seal.campaign_id
        and campaign.created_at == seal.created_at
        and campaign.cutoff.isoformat() == seal.cutoff
        and campaign.integrity_hash == seal.integrity_sha256
        and campaign.preview_digest == seal.preview_sha256
        and campaign.policy_digest == seal.policy_sha256
        and campaign.harness_tree_sha256 == seal.harness_tree_sha256
        and campaign.harness_file_count == seal.harness_file_count
    )


def _legacy_blocked_summary_payload() -> dict[str, Any]:
    seal = _LEGACY_BLOCKED_CAMPAIGN_SEAL
    return {
        "blockers": list(_LEGACY_BLOCKED_CAMPAIGN_BLOCKERS),
        "calibration_recorded": False,
        "campaign_id": seal.campaign_id,
        "candidate_decisions": {},
        "next_pair_index": 0,
        "policy_digest": seal.policy_sha256,
        "schema_version": 1,
        "status": "blocked_environment",
        "terminal_reason": None,
    }


def _read_sealed_regular_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    context: str,
) -> bytes:
    """Read one exact owned file without accepting links or replacement races."""

    payload = _read_owned_regular_file(
        path,
        max_size=expected_size + 1,
        context=context,
    )
    if (
        len(payload) != expected_size
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise CampaignPlanningError(f"{context} content changed")
    return payload


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _pread_bounded(descriptor: int, limit: int) -> bytes:
    payload = bytearray()
    while len(payload) < limit:
        chunk = os.pread(descriptor, min(65_536, limit - len(payload)), len(payload))
        if not chunk:
            break
        payload.extend(chunk)
    return bytes(payload)


def _read_owned_regular_file(
    path: Path,
    *,
    max_size: int,
    context: str,
) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise CampaignPlanningError(f"{context} requires no-follow support")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | nofollow)
    except OSError as error:
        raise CampaignPlanningError(f"{context} is absent or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or before.st_size < 0
            or before.st_size > max_size
        ):
            raise CampaignPlanningError(f"{context} topology changed")
        payload = _pread_bounded(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if (
            _metadata_identity(before) != _metadata_identity(after)
            or len(payload) != before.st_size
        ):
            raise CampaignPlanningError(f"{context} content changed")
        return payload
    finally:
        os.close(descriptor)


def _legacy_expected_topology(
    campaign: FrozenCampaign,
) -> tuple[set[Path], set[Path]]:
    directories = {Path(".")}
    files = {
        Path(".autoresearch.lock"),
        Path("campaign.json"),
        Path("summary.json"),
    }
    for cell in campaign.cells:
        relative = cell.run_dir.relative_to(campaign.campaign_dir)
        directories.add(relative)
        directories.update(relative.parents)
        files.add(relative / "inventory.json")
        files.add(relative / "plan.json")
    return directories, files


def _topology_snapshot_row(
    relative: Path, kind: str, metadata: os.stat_result
) -> tuple[object, ...]:
    return (relative.as_posix(), kind, *_metadata_identity(metadata))


def _require_legacy_topology(
    campaign: FrozenCampaign,
) -> tuple[tuple[object, ...], ...]:
    """Require the sealed campaign to contain planning artifacts and nothing else."""

    root = campaign.campaign_dir
    expected_directories, expected_files = _legacy_expected_topology(campaign)
    actual_directories: set[Path] = {Path(".")}
    actual_files: set[Path] = set()
    snapshot: list[tuple[object, ...]] = []

    def walk_error(error: OSError) -> None:
        raise error

    try:
        root_metadata = os.lstat(root)
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or stat.S_ISLNK(root_metadata.st_mode)
            or root_metadata.st_uid != os.geteuid()
        ):
            raise CampaignPlanningError(
                "preserved legacy campaign root topology changed"
            )
        snapshot.append(_topology_snapshot_row(Path("."), "directory", root_metadata))
        for current, directory_names, file_names in os.walk(
            root, topdown=True, onerror=walk_error, followlinks=False
        ):
            current_path = Path(current)
            for name in directory_names:
                path = current_path / name
                metadata = os.lstat(path)
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or stat.S_ISLNK(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                ):
                    raise CampaignPlanningError(
                        "preserved legacy campaign directory topology changed"
                    )
                relative = path.relative_to(root)
                actual_directories.add(relative)
                snapshot.append(
                    _topology_snapshot_row(relative, "directory", metadata)
                )
            for name in file_names:
                path = current_path / name
                metadata = os.lstat(path)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or metadata.st_uid != os.geteuid()
                ):
                    raise CampaignPlanningError(
                        "preserved legacy campaign file topology changed"
                    )
                relative = path.relative_to(root)
                actual_files.add(relative)
                snapshot.append(_topology_snapshot_row(relative, "file", metadata))
    except OSError as error:
        raise CampaignPlanningError(
            "preserved legacy campaign topology is unreadable"
        ) from error
    if (
        actual_directories != expected_directories
        or actual_files != expected_files
    ):
        raise CampaignPlanningError(
            "preserved legacy campaign topology changed"
        )
    return tuple(sorted(snapshot))


def _legacy_tree_identity(campaign: FrozenCampaign) -> tuple[str, int]:
    _expected_directories, expected_files = _legacy_expected_topology(campaign)
    digest = hashlib.sha256()
    total_size = 0
    for relative in sorted(expected_files, key=lambda path: path.as_posix()):
        payload = _read_owned_regular_file(
            campaign.campaign_dir / relative,
            max_size=16 * 1024 * 1024,
            context="preserved legacy campaign artifact",
        )
        relative_bytes = relative.as_posix().encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(4, "big"))
        digest.update(relative_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        total_size += len(payload)
        if total_size > 64 * 1024 * 1024:
            raise CampaignPlanningError(
                "preserved legacy campaign tree exceeds its size bound"
            )
    return digest.hexdigest(), total_size


@contextmanager
def _legacy_campaign_lock(path: Path) -> Iterator[None]:
    """Share an existing immutable lock without creating or changing it."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise CampaignPlanningError("legacy campaign lock requires no-follow support")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | nofollow)
    except OSError as error:
        raise CampaignPlanningError("legacy campaign lock is absent or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != 0
        ):
            raise CampaignPlanningError("legacy campaign lock topology changed")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignPlanningError(
                "another autoresearch controller holds the campaign lock"
            ) from error
        try:
            path_metadata = os.lstat(path)
        except OSError as error:
            raise CampaignPlanningError(
                "legacy campaign lock path changed"
            ) from error
        locked_identity = _metadata_identity(metadata)
        if _metadata_identity(path_metadata) != locked_identity:
            raise CampaignPlanningError("legacy campaign lock path changed")
        try:
            yield
        finally:
            try:
                path_metadata = os.lstat(path)
            except OSError as error:
                raise CampaignPlanningError(
                    "legacy campaign lock path changed"
                ) from error
            if _metadata_identity(path_metadata) != locked_identity:
                raise CampaignPlanningError("legacy campaign lock path changed")
    finally:
        os.close(descriptor)


def _read_legacy_blocked_summary(campaign: FrozenCampaign) -> dict[str, Any]:
    seal = _LEGACY_BLOCKED_CAMPAIGN_SEAL
    before = _require_legacy_topology(campaign)
    if _legacy_tree_identity(campaign) != (seal.tree_sha256, seal.tree_size):
        raise CampaignPlanningError(
            "preserved legacy campaign tree content changed"
        )
    _read_sealed_regular_file(
        campaign.campaign_dir / "campaign.json",
        expected_size=seal.campaign_json_size,
        expected_sha256=seal.campaign_json_sha256,
        context="preserved legacy campaign manifest",
    )
    payload = _read_sealed_regular_file(
        campaign.campaign_dir / "summary.json",
        expected_size=seal.summary_json_size,
        expected_sha256=seal.summary_json_sha256,
        context="preserved legacy campaign summary",
    )
    try:
        summary = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError, CellProjectionError) as error:
        raise CampaignPlanningError(
            "preserved legacy campaign summary is malformed"
        ) from error
    if summary != _legacy_blocked_summary_payload():
        raise CampaignPlanningError(
            "preserved legacy campaign summary payload changed"
        )
    after = _require_legacy_topology(campaign)
    if after != before:
        raise CampaignPlanningError(
            "preserved legacy campaign topology changed while reading"
        )
    return summary


def _summarize_campaign_locked(campaign: FrozenCampaign) -> dict[str, Any]:
    if _is_legacy_blocked_campaign(campaign):
        raise CampaignPlanningError(
            "the preserved legacy campaign summary is immutable"
        )
    summary = _campaign_summary_payload(campaign)
    write_json(campaign.campaign_dir / "summary.json", summary)
    return summary


def summarize_campaign(campaign_dir: Path) -> dict[str, Any]:
    initial = load_frozen_campaign(campaign_dir)
    lock_path = initial.campaign_dir / ".autoresearch.lock"
    if _is_legacy_blocked_campaign(initial):
        with _legacy_campaign_lock(lock_path):
            campaign = load_frozen_campaign(initial.campaign_dir)
            if not _is_legacy_blocked_campaign(campaign):
                raise CampaignPlanningError(
                    "preserved legacy campaign identity changed under lock"
                )
            return _read_legacy_blocked_summary(campaign)
    with _campaign_lock(lock_path):
        campaign = load_frozen_campaign(initial.campaign_dir)
        if _is_legacy_blocked_campaign(campaign):
            raise CampaignPlanningError(
                "campaign identity changed into the preserved legacy campaign"
            )
        return _summarize_campaign_locked(campaign)


@contextmanager
def _campaign_lock(path: Path) -> Iterator[None]:
    """Hold a private, single-link lock without following or truncating files."""

    flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise CampaignPlanningError("campaign lock requires no-follow support")
    try:
        descriptor = os.open(path, flags | nofollow, 0o600)
    except OSError as error:
        raise CampaignPlanningError("campaign lock path is unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
        ):
            raise CampaignPlanningError("campaign lock must be a single-link file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            os.fchmod(descriptor, 0o600)
            metadata = os.fstat(descriptor)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignPlanningError(
                "another autoresearch controller holds the campaign lock"
            ) from error
        try:
            path_metadata = os.lstat(path)
        except OSError as error:
            raise CampaignPlanningError("campaign lock path changed") from error
        locked_identity = _metadata_identity(metadata)
        if _metadata_identity(path_metadata) != locked_identity:
            raise CampaignPlanningError("campaign lock path changed")
        try:
            yield
        finally:
            try:
                path_metadata = os.lstat(path)
            except OSError as error:
                raise CampaignPlanningError("campaign lock path changed") from error
            if _metadata_identity(path_metadata) != locked_identity:
                raise CampaignPlanningError("campaign lock path changed")
    finally:
        os.close(descriptor)


@contextmanager
def _read_only_campaign_lock(path: Path) -> Iterator[None]:
    """Share an existing frozen campaign lock without mutating its topology."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise CampaignPlanningError("campaign read lock requires no-follow support")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | nofollow)
    except OSError as error:
        raise CampaignPlanningError(
            "campaign read lock is absent or unsafe"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != 0
        ):
            raise CampaignPlanningError("campaign read lock topology changed")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignPlanningError(
                "another autoresearch controller holds the campaign lock"
            ) from error
        try:
            path_metadata = os.lstat(path)
        except OSError as error:
            raise CampaignPlanningError("campaign read lock path changed") from error
        locked_identity = _metadata_identity(metadata)
        if _metadata_identity(path_metadata) != locked_identity:
            raise CampaignPlanningError("campaign read lock path changed")
        try:
            yield
        finally:
            try:
                path_metadata = os.lstat(path)
            except OSError as error:
                raise CampaignPlanningError(
                    "campaign read lock path changed"
                ) from error
            if _metadata_identity(path_metadata) != locked_identity:
                raise CampaignPlanningError("campaign read lock path changed")
    finally:
        os.close(descriptor)


def _controller_event_counts(
    events: tuple[dict[str, Any], ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        name = event.get("event")
        if not isinstance(name, str) or not name:
            raise CampaignPlanningError("controller journal event name is malformed")
        counts[name] = counts.get(name, 0) + 1
    return {name: counts[name] for name in sorted(counts)}


def _campaign_snapshot_payload(
    campaign: FrozenCampaign,
    *,
    summary: Mapping[str, Any],
    admissions: tuple[dict[str, Any], ...],
    events: tuple[dict[str, Any], ...],
    provenance_mode: str,
) -> dict[str, Any]:
    return {
        "snapshot_schema_version": 1,
        "frozen_campaign_schema_version": (
            FROZEN_CAMPAIGN_SCHEMA_VERSION
            if campaign.admission_journal_required
            else 2
        ),
        "campaign_id": campaign.campaign_id,
        "created_at": campaign.created_at,
        "cutoff_at": campaign.cutoff.isoformat(),
        "baseline_id": campaign.baseline_id,
        "suite_id": EXPECTED_SUITE_ID,
        "planned_cell_count": len(campaign.cells),
        "campaign_integrity_sha256": campaign.integrity_hash,
        "preview_sha256": campaign.preview_digest,
        "policy_sha256": campaign.policy_digest,
        "harness_tree_sha256": campaign.harness_tree_sha256,
        "harness_file_count": campaign.harness_file_count,
        "admission_journal_required": campaign.admission_journal_required,
        "provenance_mode": provenance_mode,
        "proposals": [
            {
                "candidate_id": proposal.candidate_id,
                "axis": proposal.axis,
                "delta_sha256": proposal.delta.digest,
            }
            for proposal in campaign.proposals
        ],
        "summary": json.loads(canonical_json(summary)),
        "admissions": json.loads(canonical_json(admissions)),
        "controller_event_counts": _controller_event_counts(events),
    }


def campaign_evidence_snapshot(campaign_dir: Path) -> dict[str, Any]:
    """Return a validated, scalar-only campaign snapshot without source writes."""

    initial = load_frozen_campaign(campaign_dir)
    lock_path = initial.campaign_dir / ".autoresearch.lock"
    if _is_legacy_blocked_campaign(initial):
        with _legacy_campaign_lock(lock_path):
            campaign = load_frozen_campaign(initial.campaign_dir)
            if not _is_legacy_blocked_campaign(campaign):
                raise CampaignPlanningError(
                    "preserved legacy campaign identity changed under lock"
                )
            summary = _read_legacy_blocked_summary(campaign)
            return _campaign_snapshot_payload(
                campaign,
                summary=summary,
                admissions=(),
                events=(),
                provenance_mode="sealed_legacy_unjournaled",
            )

    with _read_only_campaign_lock(lock_path):
        campaign = load_frozen_campaign(initial.campaign_dir)
        if (
            _is_legacy_blocked_campaign(campaign)
            or campaign.integrity_hash != initial.integrity_hash
            or campaign.created_at != initial.created_at
        ):
            raise CampaignPlanningError("campaign identity changed under read lock")
        events = tuple(
            _controller_events(Journal(campaign.campaign_dir / "events.jsonl"))
        )
        summary = _campaign_summary_payload(campaign)
        admissions = _read_campaign_admissions(campaign, events=events)
        controller_hash = controller_prefix_sha256(events)
        if (
            summary.get("controller_event_count") != len(events)
            or summary.get("controller_prefix_sha256") != controller_hash
            or summary.get("admission_count") != len(admissions)
            or summary.get("last_admission")
            != (admissions[-1] if admissions else None)
        ):
            raise CampaignPlanningError(
                "campaign state changed while evidence was snapshotted"
            )
        if tuple(
            _controller_events(Journal(campaign.campaign_dir / "events.jsonl"))
        ) != events:
            raise CampaignPlanningError(
                "controller journal changed while evidence was snapshotted"
            )
        final = load_frozen_campaign(initial.campaign_dir)
        if (
            final.integrity_hash != campaign.integrity_hash
            or final.created_at != campaign.created_at
        ):
            raise CampaignPlanningError("campaign identity changed under read lock")
        return _campaign_snapshot_payload(
            campaign,
            summary=summary,
            admissions=admissions,
            events=events,
            provenance_mode=(
                "required" if campaign.admission_journal_required else "optional"
            ),
        )


def _require_checkpoint_boundary(campaign: FrozenCampaign) -> None:
    """Reject states that may still settle or advance an already-admitted pair."""

    from .autoresearch_checkpoint import CheckpointError

    journal = Journal(campaign.campaign_dir / "events.jsonl")
    events = tuple(_controller_events(journal))
    _validate_search_artifact_admission(campaign, events)
    state = _replay_frozen_campaign(campaign, events) if events else None
    calibration = _verify_calibration_record(campaign)
    if calibration is None:
        cells = campaign.cells_for(candidate_id="control", stage="calibration")
        raw_started = any(
            _safe_cell_journal_size(cells[arm].run_dir / "events.jsonl") > 0
            for arm in ("control_a", "control_b")
        )
        if raw_started or (state is not None and state.phase != "terminal"):
            raise CheckpointError(
                "checkpoint_boundary_unsettled",
                "calibration boundary is not settled",
            )
        return
    if state is None:
        raise CheckpointError(
            "checkpoint_boundary_unsettled",
            "calibration has no durable controller boundary",
        )
    if not calibration[1] and state.phase != "terminal":
        raise CheckpointError(
            "checkpoint_boundary_unsettled",
            "failed calibration has not reached a terminal boundary",
        )
    if state.phase in {"pair", "scored"}:
        raise CheckpointError(
            "checkpoint_boundary_unsettled",
            "search pair boundary is not settled",
        )
    if state.phase == "candidate" and not (
        len(state.candidate_observations) == 1
        and state.confirmation_mode in {"standard", "simplification"}
    ):
        raise CheckpointError(
            "checkpoint_boundary_unsettled",
            "candidate admission boundary is not settled",
        )
    if state.phase == "idle":
        decisions = _candidate_decisions(events)
        terminal_decision = any(
            decision in {"promote", "promote_simplification"}
            for decision in decisions.values()
        ) or all(
            decisions.get(proposal.candidate_id) == "reject"
            for proposal in campaign.proposals
        )
        if terminal_decision:
            raise CheckpointError(
                "checkpoint_boundary_unsettled",
                "campaign terminal decision has not settled",
            )


def acknowledge_campaign_checkpoint(
    campaign_dir: Path,
    workspace: Path,
    *,
    evidence_verifier: Callable[..., Any] | None = None,
    repository_verifier: Callable[..., Any] | None = None,
    evidence_snapshot_reader: Callable[..., Any] | None = None,
    repository_snapshot_reader: Callable[..., Any] | None = None,
    state_root: Path | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> "CheckpointAcknowledgement":
    """Acknowledge one stable, strictly re-audited campaign pair under lock."""

    from .autoresearch_checkpoint import (
        CheckpointError,
        acknowledge_checkpoint,
    )

    initial = load_frozen_campaign(campaign_dir)
    if _is_legacy_blocked_campaign(initial):
        raise CampaignPlanningError(
            "the preserved legacy campaign is immutable and cannot be acknowledged"
        )
    lock_path = initial.campaign_dir / ".autoresearch.lock"
    with _campaign_lock(lock_path):
        campaign = load_frozen_campaign(initial.campaign_dir)
        if _is_legacy_blocked_campaign(campaign):
            raise CampaignPlanningError(
                "campaign identity changed into the preserved legacy campaign"
            )
        try:
            _recover_interrupted_cells(campaign)
        except CellProjectionError as error:
            raise CheckpointError(
                "checkpoint_boundary_unsettled",
                "campaign cleanup has not reached a stable checkpoint boundary",
            ) from error
        _require_checkpoint_boundary(campaign)
        binding = _checkpoint_campaign_binding(campaign)

        def completion_reader() -> Any:
            refreshed = load_frozen_campaign(campaign.campaign_dir)
            if _checkpoint_campaign_binding(refreshed) != binding:
                raise CheckpointError(
                    "checkpoint_race",
                    "frozen campaign binding changed during verification",
                )
            _require_checkpoint_boundary(refreshed)
            return _latest_checkpoint_completion(refreshed)

        return acknowledge_checkpoint(
            workspace=workspace,
            campaign=binding,
            journal_path=campaign.campaign_dir / "events.jsonl",
            completion_reader=completion_reader,
            evidence_verifier=evidence_verifier,
            repository_verifier=repository_verifier,
            evidence_snapshot_reader=evidence_snapshot_reader,
            repository_snapshot_reader=repository_snapshot_reader,
            state_root=state_root,
            now=now,
        )


def run_campaign(
    campaign_dir: Path,
    *,
    workspace: Path,
    now: Callable[[], datetime] = _aware_now,
    meminfo_reader: Callable[[], str] = read_host_meminfo,
    harness_identity_reader: Callable[[Path], tuple[str, int]] = harness_tree_identity,
    cell_runner: Callable[[FrozenCell], CellProjection] | None = None,
    checkpoint_gate_reader: Callable[[FrozenCampaign, Path], Any] | None = None,
) -> dict[str, Any]:
    """Run or replay the finite queue, never resuming an interrupted cell."""

    campaign = load_frozen_campaign(campaign_dir)
    if _is_legacy_blocked_campaign(campaign):
        raise CampaignPlanningError(
            "the preserved legacy campaign is immutable and cannot be run"
        )
    for cell in campaign.cells:
        blocker = model_execution_blocker(_bound_frozen_cell_plan_model(cell))
        if blocker is not None:
            raise CampaignPlanningError(blocker)
    lock_path = campaign.campaign_dir / ".autoresearch.lock"
    with _campaign_lock(lock_path):
        refreshed = load_frozen_campaign(campaign.campaign_dir)
        if _is_legacy_blocked_campaign(refreshed):
            raise CampaignPlanningError(
                "campaign identity changed into the preserved legacy campaign"
            )
        if refreshed.integrity_hash != campaign.integrity_hash:
            raise CampaignPlanningError("campaign identity changed under lock")
        campaign = refreshed
        existing_journal = Journal(campaign.campaign_dir / "events.jsonl")
        recovery_error: CellProjectionError | None = None
        try:
            _recover_interrupted_cells(campaign)
        except CellProjectionError as error:
            recovery_error = error

        # Controller corruption must not prevent exact-owned cleanup. Structural
        # replay is sufficient to establish whether a terminal append is legal;
        # raw-attested replay follows topology inspection below.
        existing_events = _controller_events(existing_journal)
        existing_state = (
            _replay_frozen_controller(campaign, existing_events)
            if existing_events
            else None
        )
        # Exact-owned cleanup is always permitted, but an unsafe admission
        # history must be rejected before recovery or reconciliation can append
        # new controller/calibration truth.
        _read_campaign_admissions(
            campaign,
            events=tuple(existing_events),
        )
        if recovery_error is not None:
            if existing_state is not None and existing_state.phase == "terminal":
                raise CampaignPlanningError(
                    "terminal campaign cleanup could not be reverified"
                ) from recovery_error
            try:
                _append_campaign_started(existing_journal, campaign)
                _append_terminal(
                    existing_journal,
                    campaign,
                    failure_kind=recovery_error.failure_kind,
                    cleanup_verified=False,
                )
            except CampaignPlanningError as replay_error:
                raise CampaignPlanningError(
                    "cleanup failed and controller truth could not be terminalized"
                ) from replay_error
            return _summarize_campaign_locked(campaign)

        raw_prefix: tuple[str, ...] = ()
        topology_error: CellProjectionError | None = None
        try:
            _validate_search_artifact_admission(
                campaign, tuple(existing_events)
            )
            raw_prefix = _raw_search_prefix(campaign, existing_state)
        except CellProjectionError as error:
            topology_error = error

        # An already-indexed raw mutation is frozen/controller corruption, not a
        # new recoverable cell failure. Cleanup is complete, but do not mutate
        # controller truth when its exact prior state no longer attests.
        try:
            existing_state = (
                _replay_frozen_campaign(campaign, existing_events)
                if existing_events
                else None
            )
        except CellProjectionError as error:
            raise CampaignPlanningError(
                "controller-bound raw artifacts could not be exactly replayed"
            ) from error

        if topology_error is not None:
            if existing_state is not None and existing_state.phase == "terminal":
                raise CampaignPlanningError(
                    "terminal campaign raw topology could not be reverified"
                ) from topology_error
            _append_campaign_started(existing_journal, campaign)
            _append_terminal(
                existing_journal,
                campaign,
                failure_kind=topology_error.failure_kind,
                cleanup_verified=True,
            )
            return _summarize_campaign_locked(campaign)

        if existing_state is not None and existing_state.phase == "terminal":
            return _summarize_campaign_locked(campaign)

        calibration_reconciled = False
        calibration_boundary_recovered = False
        try:
            existing_state = _reconcile_raw_search_prefix(
                campaign,
                existing_journal,
                existing_state,
                raw_prefix=raw_prefix,
            )
            existing_events = _controller_events(existing_journal)
            calibration_reconciled = _reconcile_raw_calibration(campaign)
            if (
                existing_state is None
                and _verify_calibration_record(campaign) is not None
            ):
                _append_campaign_started(existing_journal, campaign)
                existing_events = _controller_events(existing_journal)
                existing_state = _replay_frozen_campaign(
                    campaign, existing_events
                )
                calibration_boundary_recovered = True
        except CellProjectionError as error:
            if existing_state is not None and existing_state.phase == "terminal":
                raise CampaignPlanningError(
                    "terminal campaign cleanup could not be reverified"
                ) from error
            _append_campaign_started(existing_journal, campaign)
            _append_terminal(
                existing_journal,
                campaign,
                failure_kind=error.failure_kind,
                cleanup_verified=error.failure_kind
                not in {"cleanup_breach", "ownership_ambiguity"},
            )
            return _summarize_campaign_locked(campaign)
        if calibration_reconciled or calibration_boundary_recovered:
            return _summarize_campaign_locked(campaign)
        calibration_ready = _calibration_path(campaign).exists()
        current_events = tuple(_controller_events(existing_journal))
        if _checkpoint_gate_required(
            campaign,
            existing_state,
            current_events,
        ):
            gate_reader = checkpoint_gate_reader or _checkpoint_gate_for_campaign
            gate = gate_reader(campaign, Path(workspace))
            if not gate.ready:
                gate_payload = gate.to_mapping()
                if gate_payload.get("status") != "checkpoint_required":
                    raise CampaignPlanningError(
                        "checkpoint gate returned an invalid non-ready status"
                    )
                summary = _campaign_summary_payload(campaign)
                summary.update(gate_payload)
                return summary
        admission_target = _next_admission_target(
            campaign,
            existing_state,
            current_events,
            calibration_ready=calibration_ready,
        )
        if admission_target is not None:
            admission = _append_live_admission(
                campaign,
                target=admission_target,
                events=current_events,
                workspace=Path(workspace),
                now=now,
                meminfo_reader=meminfo_reader,
                harness_identity_reader=harness_identity_reader,
            )
            blockers = tuple(admission["blockers"])
            if blockers:
                if current_events:
                    _append_terminal(
                        existing_journal,
                        campaign,
                        failure_kind=_admission_failure_kind(blockers),
                        cleanup_verified=True,
                    )
                return _summarize_campaign_locked(campaign)

        base_runner = cell_runner or (
            lambda cell: run_frozen_cell(
                cell,
                workspace=workspace,
                cell_timeout_s=campaign.policy.cell_timeout_s,
                cleanup_timeout_s=campaign.policy.cleanup_timeout_s,
            )
        )

        def runner(cell: FrozenCell) -> CellProjection:
            try:
                current = harness_identity_reader(workspace)
            except (OSError, ValueError) as error:
                raise CellProjectionError(
                    "campaign harness identity could not be reverified",
                    failure_kind="audit",
                ) from error
            if current != (
                campaign.harness_tree_sha256,
                campaign.harness_file_count,
            ):
                raise CellProjectionError(
                    "campaign harness changed before cell launch",
                    failure_kind="audit",
                )
            return base_runner(cell)
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
                return _summarize_campaign_locked(campaign)
            while True:
                state = _replay_frozen_campaign(campaign, _controller_events(journal))
                if state.phase == "terminal":
                    break
                decisions = _candidate_decisions(tuple(_controller_events(journal)))
                if any(
                    decision in {"promote", "promote_simplification"}
                    for decision in decisions.values()
                ):
                    _append_frozen_transition(
                        journal,
                        campaign,
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
                        _append_frozen_transition(
                            journal,
                            campaign,
                            {
                                "event": "autoresearch_campaign_completed",
                                "transition_id": "campaign-completed-queue-exhausted",
                            },
                        )
                        break
                    _append_frozen_transition(
                        journal,
                        campaign,
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
                    state = _replay_frozen_campaign(
                        campaign, _controller_events(journal)
                    )
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
                    state = _replay_frozen_campaign(
                        campaign, _controller_events(journal)
                    )
                if state.phase != "scored" or state.candidate_id is None:
                    raise CampaignPlanningError("search pair did not reach scored state")
                observations = state.candidate_observations
                _require_score_eligible(campaign.policy, observations[-1])
                proposal = next(
                    (
                        item
                        for item in campaign.proposals
                        if item.candidate_id == state.candidate_id
                    ),
                    None,
                )
                if proposal is None:
                    raise CampaignPlanningError(
                        "scored candidate is absent from the frozen queue"
                    )
                if len(observations) == 1:
                    if (
                        proposal.axis == "reasoning_policy"
                        and evaluate_simplification_screen(
                            campaign.policy, observations[0]
                        ).passed
                    ):
                        decision = "confirm_simplification"
                    else:
                        decision = (
                            "confirm"
                            if evaluate_screen(
                                campaign.policy, observations[0]
                            ).passed
                            else "reject"
                        )
                elif len(observations) == 2:
                    if state.confirmation_mode == "simplification":
                        decision = (
                            "promote_simplification"
                            if evaluate_simplification_promotion(
                                campaign.policy, observations[0], observations[1]
                            ).passed
                            else "reject"
                        )
                    else:
                        decision = (
                            "promote"
                            if evaluate_promotion(
                                campaign.policy, observations[0], observations[1]
                            ).passed
                            else "reject"
                        )
                else:
                    raise CampaignPlanningError("candidate has an invalid score count")
                _append_frozen_transition(
                    journal,
                    campaign,
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
                    _append_frozen_transition(
                        journal,
                        campaign,
                        {
                            "event": "autoresearch_campaign_completed",
                            "transition_id": "campaign-completed-after-promotion",
                        },
                    )
                elif decision == "reject":
                    final_decisions = _candidate_decisions(
                        tuple(_controller_events(journal))
                    )
                    if all(
                        final_decisions.get(proposal.candidate_id) == "reject"
                        for proposal in campaign.proposals
                    ):
                        _append_frozen_transition(
                            journal,
                            campaign,
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
                cleanup_verified=error.failure_kind
                not in {"cleanup_breach", "ownership_ambiguity"},
            )
        return _summarize_campaign_locked(campaign)
