"""Bounded, offline policy primitives for paired autoresearch campaigns.

This module deliberately does not execute benchmarks, inspect raw results, or
mutate manifests.  It provides the immutable policy, deterministic scoring,
candidate-delta validation, and a small append/replay state machine that a
future campaign driver can use around :class:`bench.journal.Journal`.

All ratios are candidate/champion.  Pair execution order is counterbalanced by
the zero-based global pair index: even pairs run champion then candidate, odd
pairs run candidate then champion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .journal import Journal, canonical_json


POLICY_SCHEMA_VERSION = 1
CELL_TIMEOUT_S = 1_800
PAIR_TIMEOUT_S = 3_600
CLEANUP_TIMEOUT_S = 120
AUDIT_RESERVE_S = 900

SCREEN_GEOMEAN_MIN = 1.03
SCREEN_PRIMARY_RATIO_MIN = 0.95
SCREEN_TTFT_RATIO_MAX = 1.10
PROMOTION_PAIR_GEOMEAN_STRICT_MIN = 1.00
PROMOTION_COMBINED_GEOMEAN_MIN = 1.03
SIMPLIFICATION_COMBINED_GEOMEAN_MIN = 0.99
SIMPLIFICATION_MEMAVAILABLE_GAIN_GIB = 1.0
CALIBRATION_GEOMEAN_MIN = 0.97
CALIBRATION_GEOMEAN_MAX = 1.03
CALIBRATION_PRIMARY_RATIO_MIN = 0.95
CALIBRATION_PRIMARY_RATIO_MAX = 1.05
CALIBRATION_TTFT_RATIO_MIN = 0.90
CALIBRATION_TTFT_RATIO_MAX = 1.10

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AutoresearchError(ValueError):
    """Base error for malformed policy inputs and scores."""


class TransitionError(AutoresearchError):
    """Raised when an append-only autoresearch transition is invalid."""


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AutoresearchError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise AutoresearchError(f"{name} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, name: str
) -> None:
    unknown = set(value) - expected
    if unknown:
        raise AutoresearchError(f"{name} has unknown keys: {sorted(unknown)!r}")
    missing = expected - set(value)
    if missing:
        raise AutoresearchError(f"{name} is missing keys: {sorted(missing)!r}")


def _require_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise AutoresearchError(f"{name} must be boolean")
    return value


def _require_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AutoresearchError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise AutoresearchError(f"{name} must be at least {minimum}")
    return value


def _require_finite(
    value: Any,
    *,
    name: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AutoresearchError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise AutoresearchError(f"{name} must be finite")
    if positive and parsed <= 0:
        raise AutoresearchError(f"{name} must be positive")
    return parsed


def _require_id(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise AutoresearchError(f"{name} must be a stable lowercase identifier")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise AutoresearchError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_id_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise AutoresearchError(f"{name} must be a non-empty array")
    parsed = tuple(
        _require_id(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(parsed)) != len(parsed):
        raise AutoresearchError(f"{name} must not contain duplicates")
    return parsed


def _require_flags(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise AutoresearchError(f"{name} must be an array")
    flags: list[str] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, str)
            or not item
            or len(item) > 256
            or any(ord(character) < 32 for character in item)
        ):
            raise AutoresearchError(f"{name}[{index}] must be a bounded flag string")
        flags.append(item)
    if len(set(flags)) != len(flags):
        raise AutoresearchError(f"{name} must not contain duplicates")
    return tuple(flags)


def _canonical_json_value(value: Any, *, name: str) -> str:
    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise AutoresearchError(f"{name} must be finite JSON data") from error
    return serialized


@dataclass(frozen=True, slots=True)
class CampaignPolicy:
    """Frozen campaign contract.

    Primary case IDs and mutable axes are campaign-specific.  Every timing and
    decision threshold is fixed to the audited bounded policy; construction
    with a different value fails rather than silently creating a new protocol.
    """

    primary_case_ids: tuple[str, ...]
    allowed_axes: tuple[str, ...]
    schema_version: int = POLICY_SCHEMA_VERSION
    cell_timeout_s: int = CELL_TIMEOUT_S
    pair_timeout_s: int = PAIR_TIMEOUT_S
    cleanup_timeout_s: int = CLEANUP_TIMEOUT_S
    audit_reserve_s: int = AUDIT_RESERVE_S
    screen_geomean_min: float = SCREEN_GEOMEAN_MIN
    screen_primary_ratio_min: float = SCREEN_PRIMARY_RATIO_MIN
    screen_ttft_ratio_max: float = SCREEN_TTFT_RATIO_MAX
    promotion_pair_geomean_strict_min: float = PROMOTION_PAIR_GEOMEAN_STRICT_MIN
    promotion_combined_geomean_min: float = PROMOTION_COMBINED_GEOMEAN_MIN
    simplification_combined_geomean_min: float = (
        SIMPLIFICATION_COMBINED_GEOMEAN_MIN
    )
    simplification_memavailable_gain_gib: float = (
        SIMPLIFICATION_MEMAVAILABLE_GAIN_GIB
    )
    calibration_geomean_min: float = CALIBRATION_GEOMEAN_MIN
    calibration_geomean_max: float = CALIBRATION_GEOMEAN_MAX
    calibration_primary_ratio_min: float = CALIBRATION_PRIMARY_RATIO_MIN
    calibration_primary_ratio_max: float = CALIBRATION_PRIMARY_RATIO_MAX
    calibration_ttft_ratio_min: float = CALIBRATION_TTFT_RATIO_MIN
    calibration_ttft_ratio_max: float = CALIBRATION_TTFT_RATIO_MAX

    def __post_init__(self) -> None:
        if not isinstance(self.primary_case_ids, tuple):
            raise AutoresearchError("primary_case_ids must be an immutable tuple")
        if not isinstance(self.allowed_axes, tuple):
            raise AutoresearchError("allowed_axes must be an immutable tuple")
        _require_id_tuple(self.primary_case_ids, name="primary_case_ids")
        _require_id_tuple(self.allowed_axes, name="allowed_axes")

        exact_integers = {
            "schema_version": POLICY_SCHEMA_VERSION,
            "cell_timeout_s": CELL_TIMEOUT_S,
            "pair_timeout_s": PAIR_TIMEOUT_S,
            "cleanup_timeout_s": CLEANUP_TIMEOUT_S,
            "audit_reserve_s": AUDIT_RESERVE_S,
        }
        for field, expected in exact_integers.items():
            actual = _require_int(getattr(self, field), name=field)
            if actual != expected:
                raise AutoresearchError(f"{field} must equal {expected}")

        exact_floats = {
            "screen_geomean_min": SCREEN_GEOMEAN_MIN,
            "screen_primary_ratio_min": SCREEN_PRIMARY_RATIO_MIN,
            "screen_ttft_ratio_max": SCREEN_TTFT_RATIO_MAX,
            "promotion_pair_geomean_strict_min": (
                PROMOTION_PAIR_GEOMEAN_STRICT_MIN
            ),
            "promotion_combined_geomean_min": PROMOTION_COMBINED_GEOMEAN_MIN,
            "simplification_combined_geomean_min": (
                SIMPLIFICATION_COMBINED_GEOMEAN_MIN
            ),
            "simplification_memavailable_gain_gib": (
                SIMPLIFICATION_MEMAVAILABLE_GAIN_GIB
            ),
            "calibration_geomean_min": CALIBRATION_GEOMEAN_MIN,
            "calibration_geomean_max": CALIBRATION_GEOMEAN_MAX,
            "calibration_primary_ratio_min": CALIBRATION_PRIMARY_RATIO_MIN,
            "calibration_primary_ratio_max": CALIBRATION_PRIMARY_RATIO_MAX,
            "calibration_ttft_ratio_min": CALIBRATION_TTFT_RATIO_MIN,
            "calibration_ttft_ratio_max": CALIBRATION_TTFT_RATIO_MAX,
        }
        for field, expected in exact_floats.items():
            actual = _require_finite(getattr(self, field), name=field)
            if actual != expected:
                raise AutoresearchError(f"{field} must equal {expected}")
        if self.pair_timeout_s != 2 * self.cell_timeout_s:
            raise AutoresearchError("pair_timeout_s must equal two cell timeouts")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CampaignPolicy:
        mapping = _require_mapping(value, name="policy")
        _require_exact_keys(mapping, _POLICY_KEYS, name="policy")
        return cls(
            primary_case_ids=_require_id_tuple(
                mapping["primary_case_ids"], name="policy.primary_case_ids"
            ),
            allowed_axes=_require_id_tuple(
                mapping["allowed_axes"], name="policy.allowed_axes"
            ),
            **{
                key: mapping[key]
                for key in _POLICY_KEYS
                if key not in {"primary_case_ids", "allowed_axes"}
            },
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "primary_case_ids": list(self.primary_case_ids),
            "allowed_axes": list(self.allowed_axes),
            "cell_timeout_s": self.cell_timeout_s,
            "pair_timeout_s": self.pair_timeout_s,
            "cleanup_timeout_s": self.cleanup_timeout_s,
            "audit_reserve_s": self.audit_reserve_s,
            "screen_geomean_min": self.screen_geomean_min,
            "screen_primary_ratio_min": self.screen_primary_ratio_min,
            "screen_ttft_ratio_max": self.screen_ttft_ratio_max,
            "promotion_pair_geomean_strict_min": (
                self.promotion_pair_geomean_strict_min
            ),
            "promotion_combined_geomean_min": (
                self.promotion_combined_geomean_min
            ),
            "simplification_combined_geomean_min": (
                self.simplification_combined_geomean_min
            ),
            "simplification_memavailable_gain_gib": (
                self.simplification_memavailable_gain_gib
            ),
            "calibration_geomean_min": self.calibration_geomean_min,
            "calibration_geomean_max": self.calibration_geomean_max,
            "calibration_primary_ratio_min": self.calibration_primary_ratio_min,
            "calibration_primary_ratio_max": self.calibration_primary_ratio_max,
            "calibration_ttft_ratio_min": self.calibration_ttft_ratio_min,
            "calibration_ttft_ratio_max": self.calibration_ttft_ratio_max,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.to_mapping()).encode()).hexdigest()


_POLICY_KEYS = frozenset(
    {
        "schema_version",
        "primary_case_ids",
        "allowed_axes",
        "cell_timeout_s",
        "pair_timeout_s",
        "cleanup_timeout_s",
        "audit_reserve_s",
        "screen_geomean_min",
        "screen_primary_ratio_min",
        "screen_ttft_ratio_max",
        "promotion_pair_geomean_strict_min",
        "promotion_combined_geomean_min",
        "simplification_combined_geomean_min",
        "simplification_memavailable_gain_gib",
        "calibration_geomean_min",
        "calibration_geomean_max",
        "calibration_primary_ratio_min",
        "calibration_primary_ratio_max",
        "calibration_ttft_ratio_min",
        "calibration_ttft_ratio_max",
    }
)


@dataclass(frozen=True, slots=True)
class CandidateDelta:
    """Digest-bound description of an exactly one-axis candidate change."""

    axis: str
    champion_value_json: str
    candidate_value_json: str
    champion_config_digest: str
    candidate_config_digest: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self).encode()).hexdigest()


def validate_one_axis_delta(
    champion: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    allowed_axes: Iterable[str],
) -> CandidateDelta:
    """Validate equal config topology with exactly one allowed changed value."""

    champion_mapping = _require_mapping(champion, name="champion")
    candidate_mapping = _require_mapping(candidate, name="candidate")
    champion_keys = set(champion_mapping)
    candidate_keys = set(candidate_mapping)
    if champion_keys != candidate_keys:
        raise AutoresearchError(
            "candidate config keys must exactly match champion config keys"
        )
    if not champion_keys:
        raise AutoresearchError("candidate configs must not be empty")
    for key in champion_keys:
        _require_id(key, name="candidate config axis")

    allowed = tuple(allowed_axes)
    if not allowed:
        raise AutoresearchError("allowed_axes must not be empty")
    for index, axis in enumerate(allowed):
        _require_id(axis, name=f"allowed_axes[{index}]")
    if len(set(allowed)) != len(allowed):
        raise AutoresearchError("allowed_axes must not contain duplicates")

    champion_values = {
        key: _canonical_json_value(champion_mapping[key], name=f"champion.{key}")
        for key in sorted(champion_keys)
    }
    candidate_values = {
        key: _canonical_json_value(candidate_mapping[key], name=f"candidate.{key}")
        for key in sorted(candidate_keys)
    }
    changed = [
        key for key in sorted(champion_keys) if champion_values[key] != candidate_values[key]
    ]
    if len(changed) != 1:
        raise AutoresearchError("candidate must change exactly one configuration axis")
    axis = changed[0]
    if axis not in set(allowed):
        raise AutoresearchError(f"candidate changed disallowed axis {axis!r}")

    champion_json = _canonical_json_value(champion_mapping, name="champion")
    candidate_json = _canonical_json_value(candidate_mapping, name="candidate")
    return CandidateDelta(
        axis=axis,
        champion_value_json=champion_values[axis],
        candidate_value_json=candidate_values[axis],
        champion_config_digest=hashlib.sha256(champion_json.encode()).hexdigest(),
        candidate_config_digest=hashlib.sha256(candidate_json.encode()).hexdigest(),
    )


def strictly_simpler_flag_bundle(
    champion_flags: Sequence[str], candidate_flags: Sequence[str]
) -> bool:
    """Return whether the candidate declares a strict subset of champion flags."""

    champion = _require_flags(champion_flags, name="champion_flags")
    candidate = _require_flags(candidate_flags, name="candidate_flags")
    return set(candidate) < set(champion)


_ELIGIBILITY_POSITIVE_FIELDS = (
    "cells_completed",
    "measurement_valid",
    "validation_passed",
    "workload_matched",
    "artifact_identity_verified",
    "audit_requirement_passed",
    "cleanup_verified",
)
_ELIGIBILITY_HAZARD_FIELDS = (
    "memory_pressure",
    "swap_pressure",
    "oom",
    "ownership_ambiguous",
    "cleanup_breach",
)
_ELIGIBILITY_KEYS = frozenset(
    _ELIGIBILITY_POSITIVE_FIELDS + _ELIGIBILITY_HAZARD_FIELDS
)


@dataclass(frozen=True, slots=True)
class EligibilityInputs:
    """Explicit fail-closed inputs for a score-bearing pair."""

    cells_completed: bool
    measurement_valid: bool
    validation_passed: bool
    workload_matched: bool
    artifact_identity_verified: bool
    audit_requirement_passed: bool
    cleanup_verified: bool
    memory_pressure: bool
    swap_pressure: bool
    oom: bool
    ownership_ambiguous: bool
    cleanup_breach: bool

    def __post_init__(self) -> None:
        for field in _ELIGIBILITY_KEYS:
            _require_bool(getattr(self, field), name=f"eligibility.{field}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EligibilityInputs:
        mapping = _require_mapping(value, name="eligibility")
        _require_exact_keys(mapping, _ELIGIBILITY_KEYS, name="eligibility")
        return cls(**{key: mapping[key] for key in _ELIGIBILITY_KEYS})

    def to_mapping(self) -> dict[str, bool]:
        return {key: getattr(self, key) for key in sorted(_ELIGIBILITY_KEYS)}

    @property
    def failed_gates(self) -> tuple[str, ...]:
        failures = [
            field for field in _ELIGIBILITY_POSITIVE_FIELDS if not getattr(self, field)
        ]
        failures.extend(
            field for field in _ELIGIBILITY_HAZARD_FIELDS if getattr(self, field)
        )
        return tuple(failures)

    @property
    def eligible(self) -> bool:
        return not self.failed_gates

    @property
    def campaign_terminal_pressure(self) -> bool:
        return any(getattr(self, field) for field in _ELIGIBILITY_HAZARD_FIELDS)


@dataclass(frozen=True, slots=True)
class SimplificationEvidence:
    minimum_memavailable_gain_gib: float
    champion_flags: tuple[str, ...] = ()
    candidate_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_finite(
            self.minimum_memavailable_gain_gib,
            name="simplification.minimum_memavailable_gain_gib",
        )
        if not isinstance(self.champion_flags, tuple) or not isinstance(
            self.candidate_flags, tuple
        ):
            raise AutoresearchError("simplification flag bundles must be tuples")
        _require_flags(self.champion_flags, name="simplification.champion_flags")
        _require_flags(self.candidate_flags, name="simplification.candidate_flags")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SimplificationEvidence:
        mapping = _require_mapping(value, name="simplification")
        _require_exact_keys(
            mapping,
            frozenset(
                {
                    "minimum_memavailable_gain_gib",
                    "champion_flags",
                    "candidate_flags",
                }
            ),
            name="simplification",
        )
        return cls(
            minimum_memavailable_gain_gib=_require_finite(
                mapping["minimum_memavailable_gain_gib"],
                name="simplification.minimum_memavailable_gain_gib",
            ),
            champion_flags=_require_flags(
                mapping["champion_flags"], name="simplification.champion_flags"
            ),
            candidate_flags=_require_flags(
                mapping["candidate_flags"], name="simplification.candidate_flags"
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "minimum_memavailable_gain_gib": self.minimum_memavailable_gain_gib,
            "champion_flags": list(self.champion_flags),
            "candidate_flags": list(self.candidate_flags),
        }

    @property
    def strictly_simpler(self) -> bool:
        return strictly_simpler_flag_bundle(
            self.champion_flags, self.candidate_flags
        )


_OBSERVATION_KEYS = frozenset(
    {
        "pair_index",
        "primary_case_ids",
        "primary_speed_ratios",
        "median_ttft_ratio",
        "timing",
        "eligibility",
        "simplification",
    }
)


_TIMING_KEYS = frozenset(
    {
        "cell_elapsed_s",
        "pair_elapsed_s",
        "cleanup_elapsed_s",
        "audit_reserve_remaining_s",
    }
)


@dataclass(frozen=True, slots=True)
class TimingInputs:
    """Observed pair timings checked against the immutable campaign budgets."""

    cell_elapsed_s: tuple[float, float]
    pair_elapsed_s: float
    cleanup_elapsed_s: float
    audit_reserve_remaining_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.cell_elapsed_s, tuple) or len(self.cell_elapsed_s) != 2:
            raise AutoresearchError("cell_elapsed_s must be an immutable two-item tuple")
        for index, value in enumerate(self.cell_elapsed_s):
            parsed = _require_finite(value, name=f"cell_elapsed_s[{index}]")
            if parsed < 0:
                raise AutoresearchError(f"cell_elapsed_s[{index}] must not be negative")
        for field in (
            "pair_elapsed_s",
            "cleanup_elapsed_s",
            "audit_reserve_remaining_s",
        ):
            parsed = _require_finite(getattr(self, field), name=field)
            if parsed < 0:
                raise AutoresearchError(f"{field} must not be negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TimingInputs:
        mapping = _require_mapping(value, name="timing")
        _require_exact_keys(mapping, _TIMING_KEYS, name="timing")
        cells = mapping["cell_elapsed_s"]
        if not isinstance(cells, (list, tuple)) or len(cells) != 2:
            raise AutoresearchError("timing.cell_elapsed_s must contain two values")
        return cls(
            cell_elapsed_s=tuple(
                _require_finite(item, name=f"timing.cell_elapsed_s[{index}]")
                for index, item in enumerate(cells)
            ),  # type: ignore[arg-type]
            pair_elapsed_s=_require_finite(
                mapping["pair_elapsed_s"], name="timing.pair_elapsed_s"
            ),
            cleanup_elapsed_s=_require_finite(
                mapping["cleanup_elapsed_s"], name="timing.cleanup_elapsed_s"
            ),
            audit_reserve_remaining_s=_require_finite(
                mapping["audit_reserve_remaining_s"],
                name="timing.audit_reserve_remaining_s",
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "cell_elapsed_s": list(self.cell_elapsed_s),
            "pair_elapsed_s": self.pair_elapsed_s,
            "cleanup_elapsed_s": self.cleanup_elapsed_s,
            "audit_reserve_remaining_s": self.audit_reserve_remaining_s,
        }

    def failed_budgets(self, policy: CampaignPolicy) -> tuple[str, ...]:
        failures: list[str] = []
        if any(value > policy.cell_timeout_s for value in self.cell_elapsed_s):
            failures.append("cell_timeout")
        if self.pair_elapsed_s > policy.pair_timeout_s:
            failures.append("pair_timeout")
        if self.cleanup_elapsed_s > policy.cleanup_timeout_s:
            failures.append("cleanup_timeout")
        if self.audit_reserve_remaining_s < policy.audit_reserve_s:
            failures.append("audit_reserve")
        return tuple(failures)


@dataclass(frozen=True, slots=True)
class PairObservation:
    """One complete candidate/champion pair, independent of execution order."""

    pair_index: int
    primary_case_ids: tuple[str, ...]
    primary_speed_ratios: tuple[float, ...]
    median_ttft_ratio: float
    timing: TimingInputs
    eligibility: EligibilityInputs
    simplification: SimplificationEvidence

    def __post_init__(self) -> None:
        _require_int(self.pair_index, name="pair_index", minimum=0)
        if not isinstance(self.primary_case_ids, tuple):
            raise AutoresearchError("primary_case_ids must be a tuple")
        _require_id_tuple(self.primary_case_ids, name="primary_case_ids")
        if not isinstance(self.primary_speed_ratios, tuple):
            raise AutoresearchError("primary_speed_ratios must be a tuple")
        if len(self.primary_speed_ratios) != len(self.primary_case_ids):
            raise AutoresearchError(
                "primary_speed_ratios must align exactly with primary_case_ids"
            )
        for index, ratio in enumerate(self.primary_speed_ratios):
            _require_finite(
                ratio, name=f"primary_speed_ratios[{index}]", positive=True
            )
        _require_finite(
            self.median_ttft_ratio, name="median_ttft_ratio", positive=True
        )
        if not isinstance(self.timing, TimingInputs):
            raise AutoresearchError("timing must be TimingInputs")
        if not isinstance(self.eligibility, EligibilityInputs):
            raise AutoresearchError("eligibility must be EligibilityInputs")
        if not isinstance(self.simplification, SimplificationEvidence):
            raise AutoresearchError(
                "simplification must be SimplificationEvidence"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PairObservation:
        mapping = _require_mapping(value, name="observation")
        _require_exact_keys(mapping, _OBSERVATION_KEYS, name="observation")
        ratios_value = mapping["primary_speed_ratios"]
        if not isinstance(ratios_value, (list, tuple)) or not ratios_value:
            raise AutoresearchError(
                "observation.primary_speed_ratios must be a non-empty array"
            )
        return cls(
            pair_index=_require_int(
                mapping["pair_index"], name="observation.pair_index", minimum=0
            ),
            primary_case_ids=_require_id_tuple(
                mapping["primary_case_ids"],
                name="observation.primary_case_ids",
            ),
            primary_speed_ratios=tuple(
                _require_finite(
                    item,
                    name=f"observation.primary_speed_ratios[{index}]",
                    positive=True,
                )
                for index, item in enumerate(ratios_value)
            ),
            median_ttft_ratio=_require_finite(
                mapping["median_ttft_ratio"],
                name="observation.median_ttft_ratio",
                positive=True,
            ),
            timing=TimingInputs.from_mapping(mapping["timing"]),
            eligibility=EligibilityInputs.from_mapping(mapping["eligibility"]),
            simplification=SimplificationEvidence.from_mapping(
                mapping["simplification"]
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "pair_index": self.pair_index,
            "primary_case_ids": list(self.primary_case_ids),
            "primary_speed_ratios": list(self.primary_speed_ratios),
            "median_ttft_ratio": self.median_ttft_ratio,
            "timing": self.timing.to_mapping(),
            "eligibility": self.eligibility.to_mapping(),
            "simplification": self.simplification.to_mapping(),
        }

    @property
    def speed_geomean(self) -> float:
        return geometric_mean(self.primary_speed_ratios)


@dataclass(frozen=True, slots=True)
class GateDecision:
    stage: str
    passed: bool
    geometric_mean_ratio: float
    reasons: tuple[str, ...]


def geometric_mean(values: Iterable[float]) -> float:
    """Return a stable geometric mean of finite, strictly positive values."""

    parsed = tuple(values)
    if not parsed:
        raise AutoresearchError("geometric mean requires at least one value")
    logs = [
        math.log(_require_finite(value, name=f"ratio[{index}]", positive=True))
        for index, value in enumerate(parsed)
    ]
    return math.exp(math.fsum(logs) / len(logs))


def pair_order(pair_index: int) -> tuple[str, str]:
    index = _require_int(pair_index, name="pair_index", minimum=0)
    return (
        ("champion", "candidate")
        if index % 2 == 0
        else ("candidate", "champion")
    )


def _pair_reasons(policy: CampaignPolicy, observation: PairObservation) -> list[str]:
    reasons: list[str] = []
    if observation.primary_case_ids != policy.primary_case_ids:
        reasons.append("primary_cases_mismatch")
    reasons.extend(f"ineligible:{gate}" for gate in observation.eligibility.failed_gates)
    reasons.extend(
        f"timing:{budget}" for budget in observation.timing.failed_budgets(policy)
    )
    return reasons


def _screen_guardrail_reasons(
    policy: CampaignPolicy,
    observation: PairObservation,
    *,
    prefix: str = "",
) -> list[str]:
    reasons: list[str] = []
    if min(observation.primary_speed_ratios) < policy.screen_primary_ratio_min:
        reasons.append(f"{prefix}primary_ratio_below_screen_floor")
    if observation.median_ttft_ratio > policy.screen_ttft_ratio_max:
        reasons.append(f"{prefix}ttft_ratio_above_screen_ceiling")
    return reasons


def evaluate_screen(
    policy: CampaignPolicy, observation: PairObservation
) -> GateDecision:
    reasons = _pair_reasons(policy, observation)
    score = observation.speed_geomean
    if score < policy.screen_geomean_min:
        reasons.append("geomean_below_screen_min")
    reasons.extend(_screen_guardrail_reasons(policy, observation))
    return GateDecision("screen", not reasons, score, tuple(reasons))


def _confirmation_reasons(
    policy: CampaignPolicy,
    first: PairObservation,
    reverse: PairObservation,
) -> list[str]:
    reasons = _pair_reasons(policy, first)
    reasons.extend(_pair_reasons(policy, reverse))
    if reverse.pair_index != first.pair_index + 1:
        reasons.append("confirmation_pair_index_not_consecutive")
    elif pair_order(first.pair_index) == pair_order(reverse.pair_index):
        reasons.append("confirmation_order_not_reversed")
    if first.primary_case_ids != reverse.primary_case_ids:
        reasons.append("confirmation_primary_cases_mismatch")
    if (
        first.simplification.champion_flags
        != reverse.simplification.champion_flags
        or first.simplification.candidate_flags
        != reverse.simplification.candidate_flags
    ):
        reasons.append("confirmation_flag_bundles_mismatch")
    return reasons


def evaluate_promotion(
    policy: CampaignPolicy,
    first: PairObservation,
    reverse: PairObservation,
) -> GateDecision:
    reasons = _confirmation_reasons(policy, first, reverse)
    screen = evaluate_screen(policy, first)
    if not screen.passed:
        reasons.append("first_pair_did_not_pass_screen")
    reasons.extend(
        _screen_guardrail_reasons(policy, reverse, prefix="reverse_")
    )
    for label, observation in (("first", first), ("reverse", reverse)):
        if not observation.speed_geomean > policy.promotion_pair_geomean_strict_min:
            reasons.append(f"{label}_pair_geomean_not_strictly_above_one")
    combined = geometric_mean(
        (*first.primary_speed_ratios, *reverse.primary_speed_ratios)
    )
    if combined < policy.promotion_combined_geomean_min:
        reasons.append("combined_geomean_below_promotion_min")
    return GateDecision("promotion", not reasons, combined, tuple(dict.fromkeys(reasons)))


def evaluate_simplification_screen(
    policy: CampaignPolicy, observation: PairObservation
) -> GateDecision:
    reasons = _pair_reasons(policy, observation)
    score = observation.speed_geomean
    if score < policy.simplification_combined_geomean_min:
        reasons.append("geomean_below_simplification_floor")
    reasons.extend(_screen_guardrail_reasons(policy, observation))
    evidence = observation.simplification
    if not (
        evidence.minimum_memavailable_gain_gib
        >= policy.simplification_memavailable_gain_gib
        or evidence.strictly_simpler
    ):
        reasons.append("simplification_benefit_missing")
    return GateDecision("simplification_screen", not reasons, score, tuple(reasons))


def evaluate_simplification_promotion(
    policy: CampaignPolicy,
    first: PairObservation,
    reverse: PairObservation,
) -> GateDecision:
    reasons = _confirmation_reasons(policy, first, reverse)
    combined = geometric_mean(
        (*first.primary_speed_ratios, *reverse.primary_speed_ratios)
    )
    if combined < policy.simplification_combined_geomean_min:
        reasons.append("combined_geomean_below_simplification_floor")
    reasons.extend(
        _screen_guardrail_reasons(policy, first, prefix="first_")
    )
    reasons.extend(
        _screen_guardrail_reasons(policy, reverse, prefix="reverse_")
    )
    memory_confirmed = all(
        observation.simplification.minimum_memavailable_gain_gib
        >= policy.simplification_memavailable_gain_gib
        for observation in (first, reverse)
    )
    flags_confirmed = all(
        observation.simplification.strictly_simpler
        for observation in (first, reverse)
    )
    if not (memory_confirmed or flags_confirmed):
        reasons.append("simplification_benefit_not_confirmed_twice")
    return GateDecision(
        "simplification_promotion",
        not reasons,
        combined,
        tuple(dict.fromkeys(reasons)),
    )


def evaluate_calibration(
    policy: CampaignPolicy, observation: PairObservation
) -> GateDecision:
    reasons = _pair_reasons(policy, observation)
    score = observation.speed_geomean
    if not policy.calibration_geomean_min <= score <= policy.calibration_geomean_max:
        reasons.append("calibration_geomean_out_of_range")
    if any(
        not policy.calibration_primary_ratio_min
        <= ratio
        <= policy.calibration_primary_ratio_max
        for ratio in observation.primary_speed_ratios
    ):
        reasons.append("calibration_primary_ratio_out_of_range")
    if not policy.calibration_ttft_ratio_min <= observation.median_ttft_ratio <= (
        policy.calibration_ttft_ratio_max
    ):
        reasons.append("calibration_ttft_ratio_out_of_range")
    return GateDecision("calibration", not reasons, score, tuple(reasons))


class FailureKind(str, Enum):
    CUTOFF = "cutoff"
    CANDIDATE_SYNTAX = "candidate_syntax"
    CANDIDATE_STARTUP = "candidate_startup"
    MEASUREMENT = "measurement"
    VALIDATION = "validation"
    AUDIT = "audit"
    MEMORY_PRESSURE = "memory_pressure"
    SWAP_PRESSURE = "swap_pressure"
    OOM = "oom"
    OWNERSHIP_AMBIGUITY = "ownership_ambiguity"
    CLEANUP_BREACH = "cleanup_breach"


def _require_failure_kind(value: Any) -> FailureKind:
    try:
        return FailureKind(value)
    except (TypeError, ValueError) as error:
        raise AutoresearchError(f"unknown failure kind {value!r}") from error


def failure_disposition(
    failure_kind: FailureKind | str,
    *,
    cleanup_verified: bool,
    restored_preflight: bool,
) -> str:
    """Return ``discard_candidate`` only for fully recovered syntax/startup errors."""

    kind = _require_failure_kind(failure_kind)
    cleanup = _require_bool(cleanup_verified, name="cleanup_verified")
    preflight = _require_bool(restored_preflight, name="restored_preflight")
    if (
        kind in {FailureKind.CANDIDATE_SYNTAX, FailureKind.CANDIDATE_STARTUP}
        and cleanup
        and preflight
    ):
        return "discard_candidate"
    return "terminate_campaign"


_COMMON_EVENT_KEYS = frozenset({"event", "transition_id"})
_EVENT_KEYS = {
    "autoresearch_campaign_started": _COMMON_EVENT_KEYS
    | {"campaign_id", "policy_digest"},
    "autoresearch_candidate_started": _COMMON_EVENT_KEYS
    | {"candidate_id", "axis", "delta_digest"},
    "autoresearch_pair_started": _COMMON_EVENT_KEYS
    | {"candidate_id", "pair_index", "order"},
    "autoresearch_cell_completed": _COMMON_EVENT_KEYS
    | {"candidate_id", "pair_index", "arm"},
    "autoresearch_pair_scored": _COMMON_EVENT_KEYS
    | {"candidate_id", "pair_index", "observation"},
    "autoresearch_candidate_decided": _COMMON_EVENT_KEYS
    | {"candidate_id", "decision"},
    "autoresearch_candidate_discarded": _COMMON_EVENT_KEYS
    | {
        "candidate_id",
        "failure_kind",
        "cleanup_verified",
        "restored_preflight",
    },
    "autoresearch_campaign_terminated": _COMMON_EVENT_KEYS
    | {"failure_kind", "cleanup_verified", "restored_preflight"},
    "autoresearch_campaign_completed": _COMMON_EVENT_KEYS,
}
_DECISIONS = frozenset(
    {"reject", "confirm", "confirm_simplification", "promote", "promote_simplification"}
)


@dataclass(frozen=True, slots=True)
class ReplayState:
    phase: str
    campaign_id: str | None
    candidate_id: str | None
    candidate_axis: str | None
    next_pair_index: int
    active_pair_index: int | None
    active_order: tuple[str, ...]
    completed_arms: tuple[str, ...]
    candidate_observations: tuple[PairObservation, ...]
    confirmation_mode: str | None
    terminal_reason: str | None
    seen_transition_ids: tuple[str, ...]


def _empty_replay_state() -> ReplayState:
    return ReplayState(
        phase="new",
        campaign_id=None,
        candidate_id=None,
        candidate_axis=None,
        next_pair_index=0,
        active_pair_index=None,
        active_order=(),
        completed_arms=(),
        candidate_observations=(),
        confirmation_mode=None,
        terminal_reason=None,
        seen_transition_ids=(),
    )


def _validate_timestamp(value: Any) -> None:
    if not isinstance(value, str):
        raise TransitionError("event.timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise TransitionError("event.timestamp must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TransitionError("event.timestamp must include a timezone")


def _validated_event(raw: Any) -> dict[str, Any]:
    try:
        mapping = dict(_require_mapping(raw, name="transition"))
    except AutoresearchError as error:
        raise TransitionError(str(error)) from error
    name = mapping.get("event")
    if name not in _EVENT_KEYS:
        raise TransitionError(f"unknown autoresearch event {name!r}")
    expected = frozenset(_EVENT_KEYS[str(name)])
    if "timestamp" in mapping:
        _validate_timestamp(mapping["timestamp"])
        expected = expected | {"timestamp"}
    try:
        _require_exact_keys(mapping, expected, name=str(name))
        _require_id(mapping["transition_id"], name="transition_id")
    except AutoresearchError as error:
        raise TransitionError(str(error)) from error
    return mapping


def _transition_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "timestamp"}


def replay_transitions(
    policy: CampaignPolicy, events: Iterable[Mapping[str, Any]]
) -> ReplayState:
    """Replay and validate an append-only campaign journal.

    Identical transition IDs with identical timestamp-free payloads are
    idempotent.  Reusing an ID for a different payload is rejected.
    """

    if not isinstance(policy, CampaignPolicy):
        raise TransitionError("policy must be CampaignPolicy")
    phase = "new"
    campaign_id: str | None = None
    candidate_id: str | None = None
    candidate_axis: str | None = None
    next_pair_index = 0
    active_pair_index: int | None = None
    active_order: tuple[str, ...] = ()
    completed_arms: tuple[str, ...] = ()
    observations: tuple[PairObservation, ...] = ()
    confirmation_mode: str | None = None
    terminal_reason: str | None = None
    seen: dict[str, str] = {}

    for raw in events:
        event = _validated_event(raw)
        transition_id = str(event["transition_id"])
        payload_json = canonical_json(_transition_payload(event))
        if transition_id in seen:
            if seen[transition_id] != payload_json:
                raise TransitionError(
                    f"transition ID collision for {transition_id!r}"
                )
            continue
        seen[transition_id] = payload_json
        name = str(event["event"])

        if phase == "terminal":
            raise TransitionError("no transitions are allowed after campaign terminal")

        try:
            if name == "autoresearch_campaign_started":
                if phase != "new":
                    raise TransitionError("campaign can only start once")
                campaign_id = _require_id(event["campaign_id"], name="campaign_id")
                if _require_sha256(
                    event["policy_digest"], name="policy_digest"
                ) != policy.digest:
                    raise TransitionError("campaign policy digest does not match")
                phase = "idle"

            elif name == "autoresearch_candidate_started":
                if phase != "idle":
                    raise TransitionError("candidate can only start while campaign is idle")
                candidate_id = _require_id(event["candidate_id"], name="candidate_id")
                candidate_axis = _require_id(event["axis"], name="axis")
                if candidate_axis not in policy.allowed_axes:
                    raise TransitionError("candidate axis is not allowed by policy")
                _require_sha256(event["delta_digest"], name="delta_digest")
                observations = ()
                confirmation_mode = None
                phase = "candidate"

            elif name == "autoresearch_pair_started":
                if phase != "candidate" or candidate_id is None:
                    raise TransitionError("pair can only start for an active candidate")
                if _require_id(event["candidate_id"], name="candidate_id") != candidate_id:
                    raise TransitionError("pair candidate does not match active candidate")
                pair_index = _require_int(
                    event["pair_index"], name="pair_index", minimum=0
                )
                if pair_index != next_pair_index:
                    raise TransitionError("pair index is not the next global pair index")
                order_value = event["order"]
                if not isinstance(order_value, (list, tuple)):
                    raise TransitionError("pair order must be an array")
                order = tuple(order_value)
                if order != pair_order(pair_index):
                    raise TransitionError("pair order does not match alternating policy")
                active_pair_index = pair_index
                active_order = order
                completed_arms = ()
                phase = "pair"

            elif name == "autoresearch_cell_completed":
                if phase != "pair" or candidate_id is None or active_pair_index is None:
                    raise TransitionError("cell can only complete in an active pair")
                if _require_id(event["candidate_id"], name="candidate_id") != candidate_id:
                    raise TransitionError("cell candidate does not match active candidate")
                if _require_int(
                    event["pair_index"], name="pair_index", minimum=0
                ) != active_pair_index:
                    raise TransitionError("cell pair index does not match active pair")
                arm = event["arm"]
                if arm not in {"champion", "candidate"}:
                    raise TransitionError("cell arm must be champion or candidate")
                if len(completed_arms) >= len(active_order):
                    raise TransitionError("active pair already completed both cells")
                expected_arm = active_order[len(completed_arms)]
                if arm != expected_arm:
                    raise TransitionError("cells must complete in the frozen pair order")
                completed_arms = (*completed_arms, str(arm))

            elif name == "autoresearch_pair_scored":
                if phase != "pair" or candidate_id is None or active_pair_index is None:
                    raise TransitionError("score requires an active pair")
                if completed_arms != active_order:
                    raise TransitionError("both ordered cells must complete before scoring")
                if _require_id(event["candidate_id"], name="candidate_id") != candidate_id:
                    raise TransitionError("score candidate does not match active candidate")
                if _require_int(
                    event["pair_index"], name="pair_index", minimum=0
                ) != active_pair_index:
                    raise TransitionError("score pair index does not match active pair")
                observation = PairObservation.from_mapping(event["observation"])
                if observation.pair_index != active_pair_index:
                    raise TransitionError("observation pair index does not match event")
                observations = (*observations, observation)
                if len(observations) > 2:
                    raise TransitionError("candidate cannot exceed one reverse confirmation")
                next_pair_index += 1
                phase = "scored"

            elif name == "autoresearch_candidate_decided":
                if phase != "scored" or candidate_id is None:
                    raise TransitionError("candidate decision requires a scored pair")
                if _require_id(event["candidate_id"], name="candidate_id") != candidate_id:
                    raise TransitionError("decision candidate does not match active candidate")
                decision = event["decision"]
                if decision not in _DECISIONS:
                    raise TransitionError(f"unknown candidate decision {decision!r}")
                latest = observations[-1]
                if _pair_reasons(policy, latest):
                    raise TransitionError(
                        "hard-ineligible pair must terminate the campaign, not decide candidate"
                    )
                if decision == "confirm":
                    if len(observations) != 1 or not evaluate_screen(
                        policy, latest
                    ).passed:
                        raise TransitionError("confirmation requires a passing screen")
                    confirmation_mode = "standard"
                    phase = "candidate"
                    active_pair_index = None
                    active_order = ()
                    completed_arms = ()
                    continue
                if decision == "confirm_simplification":
                    if len(observations) != 1 or not evaluate_simplification_screen(
                        policy, latest
                    ).passed:
                        raise TransitionError(
                            "simplification confirmation requires a passing first pair"
                        )
                    confirmation_mode = "simplification"
                    phase = "candidate"
                    active_pair_index = None
                    active_order = ()
                    completed_arms = ()
                    continue
                if decision == "promote":
                    if (
                        len(observations) != 2
                        or confirmation_mode != "standard"
                        or not evaluate_promotion(
                            policy, observations[0], observations[1]
                        ).passed
                    ):
                        raise TransitionError(
                            "promotion requires passing reverse-order confirmation"
                        )
                elif decision == "promote_simplification":
                    if (
                        len(observations) != 2
                        or confirmation_mode != "simplification"
                        or not evaluate_simplification_promotion(
                            policy, observations[0], observations[1]
                        ).passed
                    ):
                        raise TransitionError(
                            "simplification promotion requires two passing pairs"
                        )
                elif decision != "reject":
                    raise TransitionError("invalid candidate decision")
                phase = "idle"
                candidate_id = None
                candidate_axis = None
                active_pair_index = None
                active_order = ()
                completed_arms = ()
                observations = ()
                confirmation_mode = None

            elif name == "autoresearch_candidate_discarded":
                if phase not in {"candidate", "pair"} or candidate_id is None:
                    raise TransitionError(
                        "only an unscored active candidate can be discarded"
                    )
                if _require_id(event["candidate_id"], name="candidate_id") != candidate_id:
                    raise TransitionError("discard candidate does not match active candidate")
                disposition = failure_disposition(
                    event["failure_kind"],
                    cleanup_verified=event["cleanup_verified"],
                    restored_preflight=event["restored_preflight"],
                )
                if disposition != "discard_candidate":
                    raise TransitionError(
                        "failure is campaign-terminal and cannot discard candidate"
                    )
                phase = "idle"
                candidate_id = None
                candidate_axis = None
                active_pair_index = None
                active_order = ()
                completed_arms = ()
                observations = ()
                confirmation_mode = None

            elif name == "autoresearch_campaign_terminated":
                if phase == "new":
                    raise TransitionError("campaign cannot terminate before it starts")
                disposition = failure_disposition(
                    event["failure_kind"],
                    cleanup_verified=event["cleanup_verified"],
                    restored_preflight=event["restored_preflight"],
                )
                if disposition != "terminate_campaign":
                    raise TransitionError(
                        "fully recovered syntax/startup failure must discard candidate"
                    )
                terminal_reason = str(_require_failure_kind(event["failure_kind"]).value)
                phase = "terminal"

            elif name == "autoresearch_campaign_completed":
                if phase != "idle":
                    raise TransitionError("campaign can only complete while idle")
                terminal_reason = "completed"
                phase = "terminal"

            else:  # pragma: no cover - exact event registry makes this unreachable.
                raise TransitionError(f"unhandled autoresearch event {name!r}")

        except AutoresearchError as error:
            if isinstance(error, TransitionError):
                raise
            raise TransitionError(str(error)) from error

    return ReplayState(
        phase=phase,
        campaign_id=campaign_id,
        candidate_id=candidate_id,
        candidate_axis=candidate_axis,
        next_pair_index=next_pair_index,
        active_pair_index=active_pair_index,
        active_order=active_order,
        completed_arms=completed_arms,
        candidate_observations=observations,
        confirmation_mode=confirmation_mode,
        terminal_reason=terminal_reason,
        seen_transition_ids=tuple(seen),
    )


def append_transition(
    journal: Journal,
    policy: CampaignPolicy,
    event: Mapping[str, Any],
) -> ReplayState:
    """Validate then durably append one transition for a single journal owner.

    Retrying an already present transition ID with the same payload is a no-op.
    The caller must provide no timestamp; :class:`Journal` adds it atomically
    with its normal flush/fsync convention.
    """

    if not isinstance(journal, Journal):
        raise TransitionError("journal must be bench.journal.Journal")
    candidate = dict(_require_mapping(event, name="transition"))
    if "timestamp" in candidate:
        raise TransitionError("append input must not provide timestamp")
    existing = journal.events()
    state = replay_transitions(policy, [*existing, candidate])
    validated = _validated_event(candidate)
    transition_id = str(validated["transition_id"])
    payload_json = canonical_json(_transition_payload(validated))
    for prior_raw in existing:
        prior = _validated_event(prior_raw)
        if prior["transition_id"] != transition_id:
            continue
        if canonical_json(_transition_payload(prior)) != payload_json:
            raise TransitionError(f"transition ID collision for {transition_id!r}")
        return state
    journal.append(candidate)
    return state
