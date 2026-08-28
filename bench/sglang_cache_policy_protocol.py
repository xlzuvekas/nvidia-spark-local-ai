"""Draft-only scalar reducer for a future SGLang cache-policy study.

This module has no serving, HTTP, manifest, or evidence integration.  It
freezes provisional topology and diagnostic arithmetic while making the
missing admission work impossible to paper over with caller assertions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, DecimalException
import hashlib
import json
import math
import re
from typing import Any, Final


PROTOCOL_SCHEMA_VERSION: Final = 3
PROTOCOL_ID: Final = "qwen38-flash-next-sglang-cache-policy-draft-v3"
VALIDATOR_ID: Final = "sparkbench.sglang-cache-policy-draft/3"

PROTOCOL_PHASE: Final = "draft"
PROTOCOL_STATUS: Final = "not_admitted"
DRAFT_BLOCKERS: Final = (
    "runtime_image_model_admission_record_absent",
    "zero_hit_canary_record_absent",
    "native_device_cache_reconciliation_contract_absent",
    "native_residency_state_pool_gauge_contract_absent",
    "workload_request_correctness_contract_absent",
    "native_host_storage_cache_reconciliation_contract_absent",
)

ARM_A: Final = "A"
ARM_B: Final = "B"
LIFETIME_ORDER: Final = (ARM_A, ARM_B, ARM_B, ARM_A)
TURN_ORDER: Final = ("T0", "T1", "T2")
ARM_A_CACHE_IMPL: Final = "UnifiedRadixCache"
ARM_B_CACHE_IMPL: Final = "ChunkCache"

PROMOTION_RATIO: Final = Decimal("0.95")
GUARDRAIL_RATIO: Final = Decimal("1.05")
MIN_COLD_INPUT_TOKENS: Final = 32 * 1024
MAX_COLD_INPUT_TOKENS: Final = 48 * 1024

PREFILL_DEVICE_HIT_METRIC: Final = (
    'sglang:prefill_effective_tokens_total{mode="device_hit"}'
)
FINISHED_REQUEST_DEVICE_HIT_METRIC: Final = (
    'sglang:cached_tokens_total{cache_source="device"}'
)

_ENVELOPE_KEYS: Final = frozenset(
    {
        "schema_version",
        "protocol_descriptor",
        "protocol_sha256",
        "provisional_runtime_source_contract_sha256",
        "validator_id",
        "lifetimes",
    }
)
_DESCRIPTOR_KEYS: Final = frozenset(
    {
        "schema_version",
        "protocol_id",
        "validator_id",
        "provisional_runtime_source_contract_sha256",
        "runtime_source_contract",
        "lifecycle",
        "future_admission_requirements",
        "topology",
        "arms",
        "provisional_device_hit_intervals",
        "provisional_cache_observation_contract",
        "unfrozen_native_contracts",
        "diagnostic_gate_order",
        "diagnostic_gate_rules",
        "schemas",
        "diagnostic_scoring",
        "output_contract",
    }
)
_RUNTIME_SOURCE_KEYS: Final = frozenset(
    {
        "provisional_source_tree",
        "digest_kind",
        "status",
        "required_source_predicates",
    }
)
_LIFETIME_KEYS: Final = frozenset(
    {
        "lifetime_ordinal",
        "arm",
        "cache_impl",
        "mamba_extra_buffer_of",
        "mamba_extra_buffer_lazy_of",
        "provisional_fresh_server_observed",
        "pre_t0_request_count",
        "pre_t0_warmup_count",
        "provisional_startup_identity_match",
        "turns",
    }
)
_TURN_KEYS: Final = frozenset(
    {
        "turn",
        "input_tokens",
        "common_prefix_tokens",
        "cache_observation",
        "provisional_prompt_identity_match",
        "provisional_correctness_passed",
        "eviction_count",
        "retraction_count",
        "other_request_count",
        "pressure_breach",
        "ttft_s",
        "wall_s",
    }
)
_CACHE_OBSERVATION_KEYS: Final = frozenset(
    {
        "request_detail_state",
        "request_device_tokens",
        "request_host_tokens",
        "request_storage_tokens",
        "provisional_settled_sglang_prefill_device_hit_tokens_delta",
        "provisional_settled_sglang_finished_request_device_hit_tokens_delta",
    }
)
_INTERVAL_KEYS: Final = frozenset({"minimum", "maximum"})

_DIAGNOSTIC_GATE_ORDER: Final = (
    "fresh_server",
    "pre_t0_request_count",
    "pre_t0_warmup_count",
    "startup_identity",
    "cache_implementation",
    "lazy_predicates",
    "cache_hit",
    "prompt_identity",
    "correctness",
    "eviction",
    "retraction",
    "other_request",
    "pressure",
)

_FUTURE_WORKLOAD_IDENTITY_REQUIREMENTS: Final = {
    "tokenizer": "pinned tokenizer artifact, revision, and digest",
    "model": "pinned model artifact, revision, and digest",
    "template": "pinned chat template and rendering implementation",
    "tools": "pinned tool schemas, order, and serialization",
    "turn_plan": "pinned T0/T1/T2 messages, tool calls, and tool results",
    "reasoning": "pinned reasoning policy and reasoning-effort value",
    "sampling": "pinned sampling parameters",
    "output_cap": "pinned output-token cap",
    "correctness_validator": (
        "pinned validator identity, version, fixtures, and acceptance rules"
    ),
}

_SHA256_PATTERN: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_TREE_PATTERN: Final = re.compile(r"[0-9a-f]{40}\Z")
_OBVIOUS_PLACEHOLDER_DIGESTS: Final = frozenset(
    {
        "0" * 64,
        "f" * 64,
        "deadbeef" * 8,
        "0123456789abcdef" * 4,
    }
)


class SGLangCachePolicyProtocolError(ValueError):
    """A fail-closed draft protocol schema or topology error."""


@dataclass(frozen=True, slots=True)
class _CacheObservation:
    request_device_tokens: int
    request_host_tokens: int
    request_storage_tokens: int
    provisional_prefill_device_delta: int
    provisional_finished_request_device_delta: int


@dataclass(frozen=True, slots=True)
class _Turn:
    turn: str
    input_tokens: int
    common_prefix_tokens: int
    cache: _CacheObservation
    provisional_prompt_identity_match: bool
    provisional_correctness_passed: bool
    eviction_count: int
    retraction_count: int
    other_request_count: int
    pressure_breach: bool
    ttft_s: Decimal
    wall_s: Decimal


@dataclass(frozen=True, slots=True)
class _Lifetime:
    lifetime_ordinal: int
    arm: str
    cache_impl: str
    mamba_extra_buffer_of: bool
    mamba_extra_buffer_lazy_of: bool
    provisional_fresh_server_observed: bool
    pre_t0_request_count: int
    pre_t0_warmup_count: int
    provisional_startup_identity_match: bool
    turns: tuple[_Turn, _Turn, _Turn]


@dataclass(frozen=True, slots=True)
class _ParsedProtocol:
    sha256: str
    provisional_runtime_source_contract_sha256: str
    provisional_source_tree: str
    arm_a_hit_intervals: dict[str, tuple[int, int]]


def _require_object(value: object, *, context: str) -> dict[str, object]:
    if type(value) is not dict:
        raise SGLangCachePolicyProtocolError(
            f"{context} must be an exact JSON object"
        )
    if any(not isinstance(key, str) for key in value):
        raise SGLangCachePolicyProtocolError(
            f"{context} field names must be strings"
        )
    return value


def _require_exact_keys(
    value: dict[str, object], expected: frozenset[str], *, context: str
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unknown:
        details.append("unknown " + ", ".join(unknown))
    raise SGLangCachePolicyProtocolError(
        f"{context} fields changed: " + "; ".join(details)
    )


def _require_bool(value: object, *, context: str) -> bool:
    if type(value) is not bool:
        raise SGLangCachePolicyProtocolError(f"{context} must be boolean")
    return value


def _require_int(
    value: object,
    *,
    context: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise SGLangCachePolicyProtocolError(
            f"{context} must be an integer at least {minimum}"
        )
    if maximum is not None and value > maximum:
        raise SGLangCachePolicyProtocolError(
            f"{context} must not exceed {maximum}"
        )
    return value


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise SGLangCachePolicyProtocolError(f"{context} must be a string")
    return value


def _require_enum(
    value: object, allowed: frozenset[str], *, context: str
) -> str:
    parsed = _require_string(value, context=context)
    if parsed not in allowed:
        raise SGLangCachePolicyProtocolError(
            f"{context} must be one of {sorted(allowed)}"
        )
    return parsed


def _require_sha256(value: object, *, context: str) -> str:
    parsed = _require_string(value, context=context)
    if _SHA256_PATTERN.fullmatch(parsed) is None:
        raise SGLangCachePolicyProtocolError(
            f"{context} must be a lowercase sha256 digest"
        )
    return parsed


def _require_pinned_sha256(value: object, *, context: str) -> str:
    parsed = _require_sha256(value, context=context)
    body = parsed.removeprefix("sha256:")
    if body in _OBVIOUS_PLACEHOLDER_DIGESTS or len(set(body)) < 8:
        raise SGLangCachePolicyProtocolError(
            f"{context} must be a non-placeholder pinned digest"
        )
    return parsed


def _require_git_tree(value: object, *, context: str) -> str:
    parsed = _require_string(value, context=context)
    if _GIT_TREE_PATTERN.fullmatch(parsed) is None:
        raise SGLangCachePolicyProtocolError(
            f"{context} must be a full lowercase 40-hex Git tree"
        )
    return parsed


def _require_positive_decimal(value: object, *, context: str) -> Decimal:
    if type(value) not in (int, float):
        raise SGLangCachePolicyProtocolError(
            f"{context} must be a positive finite JSON number"
        )
    try:
        finite_value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise SGLangCachePolicyProtocolError(
            f"{context} must be a positive finite JSON number"
        ) from exc
    if not math.isfinite(finite_value) or finite_value <= 0:
        raise SGLangCachePolicyProtocolError(
            f"{context} must be a positive finite JSON number"
        )
    parsed = Decimal(str(value))
    if not parsed.is_finite() or parsed <= 0:
        raise SGLangCachePolicyProtocolError(
            f"{context} must be a positive finite JSON number"
        )
    return parsed


def _canonical_json(value: object, *, context: str) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SGLangCachePolicyProtocolError(
            f"{context} must contain only finite JSON values"
        ) from exc


def protocol_sha256(descriptor: object) -> str:
    """Return the canonical SHA-256 identity of a JSON descriptor."""

    canonical = _canonical_json(descriptor, context="protocol descriptor")
    return "sha256:" + hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _provisional_interval(
    minimum: object, maximum: object, *, turn: str
) -> dict[str, int]:
    parsed_minimum = _require_int(
        minimum,
        context=f"arm A {turn} provisional device-hit minimum",
        minimum=1,
    )
    parsed_maximum = _require_int(
        maximum,
        context=f"arm A {turn} provisional device-hit maximum",
        minimum=1,
    )
    if parsed_minimum > parsed_maximum:
        raise SGLangCachePolicyProtocolError(
            f"arm A {turn} provisional device-hit interval is reversed"
        )
    return {"minimum": parsed_minimum, "maximum": parsed_maximum}


def protocol_descriptor(
    *,
    provisional_runtime_source_contract_sha256: str,
    provisional_source_tree: str,
    provisional_arm_a_t1_device_hit_minimum: int,
    provisional_arm_a_t1_device_hit_maximum: int,
    provisional_arm_a_t2_device_hit_minimum: int,
    provisional_arm_a_t2_device_hit_maximum: int,
) -> dict[str, Any]:
    """Build the hashable draft descriptor from provisional source inputs."""

    runtime_digest = _require_pinned_sha256(
        provisional_runtime_source_contract_sha256,
        context="provisional runtime/source contract digest",
    )
    source_tree = _require_git_tree(
        provisional_source_tree, context="provisional source tree"
    )
    t1_interval = _provisional_interval(
        provisional_arm_a_t1_device_hit_minimum,
        provisional_arm_a_t1_device_hit_maximum,
        turn="T1",
    )
    t2_interval = _provisional_interval(
        provisional_arm_a_t2_device_hit_minimum,
        provisional_arm_a_t2_device_hit_maximum,
        turn="T2",
    )
    zero_interval = {"minimum": 0, "maximum": 0}

    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "validator_id": VALIDATOR_ID,
        "provisional_runtime_source_contract_sha256": runtime_digest,
        "runtime_source_contract": {
            "provisional_source_tree": source_tree,
            "digest_kind": "provisional source contract; not runtime admission",
            "status": "draft_only",
            "required_source_predicates": [
                "mamba_extra_buffer_of",
                "mamba_extra_buffer_lazy_of",
            ],
        },
        "lifecycle": {
            "phase": PROTOCOL_PHASE,
            "status": PROTOCOL_STATUS,
            "protocol_runnable": False,
            "measurement_admissible": False,
            "decision": PROTOCOL_STATUS,
            "selected_arm": "none",
            "blockers": list(DRAFT_BLOCKERS),
            "caller_cannot_override": True,
        },
        "future_admission_requirements": {
            "runtime_image_model_admission_record": "required but absent",
            "zero_hit_canary_record": "required but absent",
            "native_device_cache_reconciliation_contract": (
                "required but absent; current equality check is diagnostic only"
            ),
            "native_residency_state_pool_gauge_contract": "required but absent",
            "native_host_storage_cache_reconciliation_contract": (
                "required but absent"
            ),
            "workload_identity": dict(_FUTURE_WORKLOAD_IDENTITY_REQUIREMENTS),
            "request_contract": "required but absent",
            "correctness_contract": "required but absent",
        },
        "topology": {
            "lifetime_order": list(LIFETIME_ORDER),
            "turn_order": list(TURN_ORDER),
            "fresh_server_per_lifetime": True,
            "pre_t0_request_count": 0,
            "pre_t0_warmup_count": 0,
            "cold_input_min_tokens": MIN_COLD_INPUT_TOKENS,
            "cold_input_max_tokens": MAX_COLD_INPUT_TOKENS,
            "later_turn_history": "append-only",
        },
        "arms": {
            ARM_A: {
                "observed_impl_label": ARM_A_CACHE_IMPL,
                "mamba_extra_buffer_of": True,
                "mamba_extra_buffer_lazy_of": True,
            },
            ARM_B: {
                "observed_impl_label": ARM_B_CACHE_IMPL,
                "mamba_extra_buffer_of": False,
                "mamba_extra_buffer_lazy_of": False,
            },
        },
        "provisional_device_hit_intervals": {
            "status": "diagnostic_only_until_admitted_calibration",
            ARM_A: {
                "T0": dict(zero_interval),
                "T1": t1_interval,
                "T2": t2_interval,
            },
            ARM_B: {
                "T0": dict(zero_interval),
                "T1": dict(zero_interval),
                "T2": dict(zero_interval),
            },
        },
        "provisional_cache_observation_contract": {
            "status": "hash_bound_diagnostic_only",
            "accepted_request_detail_state": "reported",
            "omitted_or_null_rule": (
                "reject; no admitted zero-hit canary record exists"
            ),
            "prefill_device_hit_metric": PREFILL_DEVICE_HIT_METRIC,
            "finished_request_device_hit_metric": (
                FINISHED_REQUEST_DEVICE_HIT_METRIC
            ),
            "metric_sampling": "provisional settled per-request deltas",
            "device_rule": (
                "reported request device count must equal both provisional "
                "settled native device deltas"
            ),
            "host_storage_rule": (
                "reported request values retained as diagnostics only; zero "
                "does not prove no spill because native reconciliation is absent"
            ),
        },
        "unfrozen_native_contracts": {
            "device_cache": (
                "absent; request/native equality check is provisional only"
            ),
            "residency": "absent; no native gauges accepted",
            "state_pool": "absent; no native gauges accepted",
            "host_cache": "absent; request value is not native proof",
            "storage_cache": "absent; request value is not native proof",
        },
        "diagnostic_gate_order": list(_DIAGNOSTIC_GATE_ORDER),
        "diagnostic_gate_rules": {
            "fresh_server": (
                "provisional_fresh_server_observed is true per lifetime"
            ),
            "pre_t0_request_count": "pre_t0_request_count == 0 per lifetime",
            "pre_t0_warmup_count": "pre_t0_warmup_count == 0 per lifetime",
            "startup_identity": (
                "provisional_startup_identity_match is true per lifetime"
            ),
            "cache_implementation": (
                "observed cache_impl equals the arm implementation label"
            ),
            "lazy_predicates": (
                "both observed Mamba predicate booleans equal the arm contract"
            ),
            "cache_hit": (
                "reported request T0 and B counts are zero; A T1/T2 device "
                "counts are inside provisional intervals; host/storage are zero"
            ),
            "prompt_identity": (
                "provisional prompt booleans are true and same-index scalar "
                "input/common-prefix counts match; never admission proof"
            ),
            "correctness": (
                "provisional correctness booleans are true; never admission proof"
            ),
            "eviction": "eviction_count == 0 for every turn",
            "retraction": "retraction_count == 0 for every turn",
            "other_request": "other_request_count == 0 for every turn",
            "pressure": "pressure_breach is false for every turn",
        },
        "schemas": {
            "envelope_fields": sorted(_ENVELOPE_KEYS),
            "lifetime_fields": sorted(_LIFETIME_KEYS),
            "turn_fields": sorted(_TURN_KEYS),
            "cache_observation_fields": sorted(_CACHE_OBSERVATION_KEYS),
            "unknown_fields": "reject",
            "nullable_fields": [],
            "container_types": {
                "envelope": "exact JSON object",
                "lifetimes": "exact four-item JSON list",
                "turns": "exact three-item JSON list",
            },
            "field_types": {
                "envelope": {
                    "exact_integer": ["schema_version"],
                    "bounded_string": [
                        "protocol_sha256",
                        "provisional_runtime_source_contract_sha256",
                        "validator_id",
                    ],
                    "object": ["protocol_descriptor"],
                    "list": ["lifetimes"],
                },
                "lifetime": {
                    "nonnegative_integer": [
                        "lifetime_ordinal",
                        "pre_t0_request_count",
                        "pre_t0_warmup_count",
                    ],
                    "exact_boolean": [
                        "mamba_extra_buffer_of",
                        "mamba_extra_buffer_lazy_of",
                        "provisional_fresh_server_observed",
                        "provisional_startup_identity_match",
                    ],
                    "bounded_string": ["arm", "cache_impl"],
                    "list": ["turns"],
                },
                "turn": {
                    "nonnegative_integer": [
                        "input_tokens",
                        "common_prefix_tokens",
                        "eviction_count",
                        "retraction_count",
                        "other_request_count",
                    ],
                    "exact_boolean": [
                        "provisional_prompt_identity_match",
                        "provisional_correctness_passed",
                        "pressure_breach",
                    ],
                    "positive_finite_number": ["ttft_s", "wall_s"],
                    "bounded_string": ["turn"],
                    "object": ["cache_observation"],
                },
                "cache_observation": {
                    "nonnegative_integer": [
                        "request_device_tokens",
                        "request_host_tokens",
                        "request_storage_tokens",
                        "provisional_settled_sglang_prefill_device_hit_tokens_delta",
                        "provisional_settled_sglang_finished_request_device_hit_tokens_delta",
                    ],
                    "bounded_string": ["request_detail_state"],
                },
            },
        },
        "diagnostic_scoring": {
            "lifetime_later_wall_formula": "T1.wall_s + T2.wall_s",
            "arm_later_wall_formula": (
                "unweighted arithmetic mean of two lifetime later walls"
            ),
            "b_over_a_formula": "arm B mean later wall / arm A mean later wall",
            "a_over_b_formula": "arm A mean later wall / arm B mean later wall",
            "b_faster_if_b_over_a_lte": float(PROMOTION_RATIO),
            "a_faster_if_a_over_b_lte": float(PROMOTION_RATIO),
            "speed_threshold_inclusive": True,
            "otherwise": "inconclusive",
            "lifetime_later_ttft_formula": "mean(T1.ttft_s, T2.ttft_s)",
            "arm_later_ttft_formula": (
                "unweighted arithmetic mean of two lifetime later TTFTs"
            ),
            "lifetime_full_wall_formula": (
                "T0.wall_s + T1.wall_s + T2.wall_s"
            ),
            "arm_full_wall_formula": (
                "unweighted arithmetic mean of two lifetime full walls"
            ),
            "candidate_guardrail_formula": (
                "diagnostic candidate arm mean / other arm mean"
            ),
            "candidate_later_ttft_ratio_lte": float(GUARDRAIL_RATIO),
            "candidate_full_wall_ratio_lte": float(GUARDRAIL_RATIO),
            "guardrail_threshold_inclusive": True,
            "t0_cold_wall_role": "diagnostic only; not in speed comparison",
            "arithmetic": "Decimal comparisons with no threshold rounding",
        },
        "output_contract": {
            "scalar_leaves_only": True,
            "malformed_schema_or_topology": "raise custom protocol error",
            "diagnostic_gate_failure": "diagnostic result not_evaluated",
            "unevaluated_diagnostic_ratios": None,
            "measurement_admissible": False,
            "decision": PROTOCOL_STATUS,
            "selected_arm": "none",
            "never_emits_promotion_or_retention": True,
        },
    }


def _parse_descriptor(value: object) -> _ParsedProtocol:
    row = _require_object(value, context="protocol descriptor")
    _require_exact_keys(row, _DESCRIPTOR_KEYS, context="protocol descriptor")
    schema_version = _require_int(
        row["schema_version"], context="descriptor schema version"
    )
    if schema_version != PROTOCOL_SCHEMA_VERSION:
        raise SGLangCachePolicyProtocolError(
            "protocol descriptor schema version is unsupported"
        )
    if row["protocol_id"] != PROTOCOL_ID:
        raise SGLangCachePolicyProtocolError("protocol descriptor ID changed")
    if row["validator_id"] != VALIDATOR_ID:
        raise SGLangCachePolicyProtocolError(
            "protocol descriptor validator identity changed"
        )
    runtime_digest = _require_pinned_sha256(
        row["provisional_runtime_source_contract_sha256"],
        context="descriptor provisional runtime/source contract digest",
    )
    runtime_contract = _require_object(
        row["runtime_source_contract"], context="runtime/source contract"
    )
    _require_exact_keys(
        runtime_contract,
        _RUNTIME_SOURCE_KEYS,
        context="runtime/source contract",
    )
    source_tree = _require_git_tree(
        runtime_contract["provisional_source_tree"],
        context="descriptor provisional source tree",
    )

    intervals = _require_object(
        row["provisional_device_hit_intervals"],
        context="provisional device-hit intervals",
    )
    _require_exact_keys(
        intervals,
        frozenset({"status", ARM_A, ARM_B}),
        context="provisional device-hit intervals",
    )
    arm_a = _require_object(intervals[ARM_A], context="arm A hit intervals")
    _require_exact_keys(
        arm_a, frozenset(TURN_ORDER), context="arm A hit intervals"
    )
    parsed_intervals: dict[str, tuple[int, int]] = {}
    for turn in ("T1", "T2"):
        interval = _require_object(
            arm_a[turn], context=f"arm A {turn} hit interval"
        )
        _require_exact_keys(
            interval, _INTERVAL_KEYS, context=f"arm A {turn} hit interval"
        )
        provisional = _provisional_interval(
            interval["minimum"], interval["maximum"], turn=turn
        )
        parsed_intervals[turn] = (
            provisional["minimum"],
            provisional["maximum"],
        )

    expected = protocol_descriptor(
        provisional_runtime_source_contract_sha256=runtime_digest,
        provisional_source_tree=source_tree,
        provisional_arm_a_t1_device_hit_minimum=parsed_intervals["T1"][0],
        provisional_arm_a_t1_device_hit_maximum=parsed_intervals["T1"][1],
        provisional_arm_a_t2_device_hit_minimum=parsed_intervals["T2"][0],
        provisional_arm_a_t2_device_hit_maximum=parsed_intervals["T2"][1],
    )
    if _canonical_json(row, context="protocol descriptor") != _canonical_json(
        expected, context="expected protocol descriptor"
    ):
        raise SGLangCachePolicyProtocolError(
            "protocol descriptor semantics or nested fields changed"
        )
    return _ParsedProtocol(
        sha256=protocol_sha256(expected),
        provisional_runtime_source_contract_sha256=runtime_digest,
        provisional_source_tree=source_tree,
        arm_a_hit_intervals=parsed_intervals,
    )


def normalize_cache_observation(value: object) -> dict[str, int | bool | str]:
    """Reconcile one reported request with provisional native device deltas."""

    row = _require_object(value, context="cache observation")
    _require_exact_keys(
        row, _CACHE_OBSERVATION_KEYS, context="cache observation"
    )
    state = _require_enum(
        row["request_detail_state"],
        frozenset({"reported", "omitted", "null"}),
        context="request cache-detail state",
    )
    if state != "reported":
        raise SGLangCachePolicyProtocolError(
            "draft protocol rejects omitted or null cache details because "
            "no admitted zero-hit canary record exists"
        )
    request_device = _require_int(
        row["request_device_tokens"], context="request device cached tokens"
    )
    request_host = _require_int(
        row["request_host_tokens"], context="request host cached tokens"
    )
    request_storage = _require_int(
        row["request_storage_tokens"], context="request storage cached tokens"
    )
    prefill_device = _require_int(
        row["provisional_settled_sglang_prefill_device_hit_tokens_delta"],
        context="provisional settled prefill device-hit delta",
    )
    finished_device = _require_int(
        row[
            "provisional_settled_sglang_finished_request_device_hit_tokens_delta"
        ],
        context="provisional settled finished-request device-hit delta",
    )
    if prefill_device != finished_device:
        raise SGLangCachePolicyProtocolError(
            "the two provisional settled native device-hit deltas disagree"
        )
    if request_device != prefill_device:
        raise SGLangCachePolicyProtocolError(
            "reported request device count disagrees with provisional native deltas"
        )
    return {
        "request_detail_state": state,
        "reported_request_device_tokens": request_device,
        "reported_request_host_tokens": request_host,
        "reported_request_storage_tokens": request_storage,
        "provisional_native_prefill_device_hit_tokens_delta": prefill_device,
        "provisional_native_finished_request_device_hit_tokens_delta": (
            finished_device
        ),
        "provisional_device_reconciliation_matched": True,
    }


def _parse_turn(
    value: object,
    *,
    expected_turn: str,
    arm: str,
    arm_a_hit_intervals: dict[str, tuple[int, int]],
) -> _Turn:
    row = _require_object(value, context=f"turn {expected_turn}")
    _require_exact_keys(row, _TURN_KEYS, context=f"turn {expected_turn}")
    turn = _require_enum(row["turn"], frozenset(TURN_ORDER), context="turn")
    if turn != expected_turn:
        raise SGLangCachePolicyProtocolError(
            "cache-policy turns must be ordered exactly T0, T1, T2"
        )
    cache = normalize_cache_observation(row["cache_observation"])
    parsed = _Turn(
        turn=turn,
        input_tokens=_require_int(
            row["input_tokens"], context=f"{turn} input tokens", minimum=1
        ),
        common_prefix_tokens=_require_int(
            row["common_prefix_tokens"],
            context=f"{turn} common-prefix tokens",
        ),
        cache=_CacheObservation(
            request_device_tokens=int(cache["reported_request_device_tokens"]),
            request_host_tokens=int(cache["reported_request_host_tokens"]),
            request_storage_tokens=int(cache["reported_request_storage_tokens"]),
            provisional_prefill_device_delta=int(
                cache["provisional_native_prefill_device_hit_tokens_delta"]
            ),
            provisional_finished_request_device_delta=int(
                cache[
                    "provisional_native_finished_request_device_hit_tokens_delta"
                ]
            ),
        ),
        provisional_prompt_identity_match=_require_bool(
            row["provisional_prompt_identity_match"],
            context=f"{turn} provisional prompt identity",
        ),
        provisional_correctness_passed=_require_bool(
            row["provisional_correctness_passed"],
            context=f"{turn} provisional correctness",
        ),
        eviction_count=_require_int(
            row["eviction_count"], context=f"{turn} eviction count"
        ),
        retraction_count=_require_int(
            row["retraction_count"], context=f"{turn} retraction count"
        ),
        other_request_count=_require_int(
            row["other_request_count"],
            context=f"{turn} other-request count",
        ),
        pressure_breach=_require_bool(
            row["pressure_breach"], context=f"{turn} pressure breach"
        ),
        ttft_s=_require_positive_decimal(
            row["ttft_s"], context=f"{turn} TTFT"
        ),
        wall_s=_require_positive_decimal(
            row["wall_s"], context=f"{turn} wall"
        ),
    )
    if parsed.ttft_s > parsed.wall_s:
        raise SGLangCachePolicyProtocolError(
            f"{turn} TTFT must not exceed request wall time"
        )
    if turn == "T0":
        if parsed.common_prefix_tokens != 0:
            raise SGLangCachePolicyProtocolError(
                "T0 must have zero prior common-prefix tokens"
            )
    else:
        if not 0 < parsed.common_prefix_tokens < parsed.input_tokens:
            raise SGLangCachePolicyProtocolError(
                f"{turn} common-prefix tokens must be within the rendered input"
            )
        if arm == ARM_A:
            maximum_reusable = min(
                parsed.common_prefix_tokens, parsed.input_tokens - 1
            )
            if arm_a_hit_intervals[turn][1] > maximum_reusable:
                raise SGLangCachePolicyProtocolError(
                    f"provisional arm A {turn} device-hit interval exceeds "
                    "the rendered reusable prefix"
                )
    return parsed


def _parse_lifetime(
    value: object,
    *,
    expected_ordinal: int,
    expected_arm: str,
    arm_a_hit_intervals: dict[str, tuple[int, int]],
) -> tuple[_Lifetime, set[str]]:
    row = _require_object(value, context=f"lifetime {expected_ordinal}")
    _require_exact_keys(
        row, _LIFETIME_KEYS, context=f"lifetime {expected_ordinal}"
    )
    ordinal = _require_int(
        row["lifetime_ordinal"],
        context="lifetime ordinal",
        minimum=1,
        maximum=len(LIFETIME_ORDER),
    )
    if ordinal != expected_ordinal:
        raise SGLangCachePolicyProtocolError(
            "cache-policy lifetime ordinals must be contiguous"
        )
    arm = _require_enum(row["arm"], frozenset({ARM_A, ARM_B}), context="arm")
    if arm != expected_arm:
        raise SGLangCachePolicyProtocolError(
            "cache-policy lifetimes must be ordered exactly A, B, B, A"
        )
    cache_impl = _require_enum(
        row["cache_impl"],
        frozenset({ARM_A_CACHE_IMPL, ARM_B_CACHE_IMPL}),
        context="cache implementation",
    )
    extra_buffer = _require_bool(
        row["mamba_extra_buffer_of"], context="mamba_extra_buffer_of observation"
    )
    extra_buffer_lazy = _require_bool(
        row["mamba_extra_buffer_lazy_of"],
        context="mamba_extra_buffer_lazy_of observation",
    )
    fresh_server = _require_bool(
        row["provisional_fresh_server_observed"],
        context="provisional fresh-server observation",
    )
    pre_t0_requests = _require_int(
        row["pre_t0_request_count"], context="pre-T0 request count"
    )
    pre_t0_warmups = _require_int(
        row["pre_t0_warmup_count"], context="pre-T0 warmup count"
    )
    startup_identity = _require_bool(
        row["provisional_startup_identity_match"],
        context="provisional startup identity",
    )
    raw_turns = row["turns"]
    if type(raw_turns) is not list or len(raw_turns) != len(TURN_ORDER):
        raise SGLangCachePolicyProtocolError(
            "each cache-policy lifetime must contain an exact T0, T1, T2 list"
        )
    turns = tuple(
        _parse_turn(
            raw,
            expected_turn=expected_turn,
            arm=arm,
            arm_a_hit_intervals=arm_a_hit_intervals,
        )
        for raw, expected_turn in zip(raw_turns, TURN_ORDER, strict=True)
    )
    typed_turns = (turns[0], turns[1], turns[2])
    if not (
        typed_turns[0].input_tokens
        < typed_turns[1].input_tokens
        < typed_turns[2].input_tokens
    ):
        raise SGLangCachePolicyProtocolError(
            "T0, T1, T2 rendered input lengths must increase strictly"
        )
    if not (
        typed_turns[1].common_prefix_tokens
        <= typed_turns[0].input_tokens
        and typed_turns[2].common_prefix_tokens
        <= typed_turns[1].input_tokens
        and typed_turns[1].common_prefix_tokens
        <= typed_turns[2].common_prefix_tokens
    ):
        raise SGLangCachePolicyProtocolError(
            "later-turn common prefixes do not match append-only history"
        )
    if not (
        MIN_COLD_INPUT_TOKENS
        <= typed_turns[0].input_tokens
        <= MAX_COLD_INPUT_TOKENS
    ):
        raise SGLangCachePolicyProtocolError(
            "T0 rendered input must be within the frozen 32K-48K window"
        )

    gates: set[str] = set()
    if not fresh_server:
        gates.add("fresh_server")
    if pre_t0_requests:
        gates.add("pre_t0_request_count")
    if pre_t0_warmups:
        gates.add("pre_t0_warmup_count")
    if not startup_identity:
        gates.add("startup_identity")
    expected_impl = ARM_A_CACHE_IMPL if arm == ARM_A else ARM_B_CACHE_IMPL
    if cache_impl != expected_impl:
        gates.add("cache_implementation")
    expected_predicate = arm == ARM_A
    if (
        extra_buffer is not expected_predicate
        or extra_buffer_lazy is not expected_predicate
    ):
        gates.add("lazy_predicates")

    for turn in typed_turns:
        if not turn.provisional_prompt_identity_match:
            gates.add("prompt_identity")
        if not turn.provisional_correctness_passed:
            gates.add("correctness")
        if turn.eviction_count:
            gates.add("eviction")
        if turn.retraction_count:
            gates.add("retraction")
        if turn.other_request_count:
            gates.add("other_request")
        if turn.pressure_breach:
            gates.add("pressure")

        request_counts = (
            turn.cache.request_device_tokens,
            turn.cache.request_host_tokens,
            turn.cache.request_storage_tokens,
        )
        if turn.turn == "T0" and any(request_counts):
            gates.add("cache_hit")
        if turn.cache.request_host_tokens or turn.cache.request_storage_tokens:
            gates.add("cache_hit")
        if arm == ARM_B and any(request_counts):
            gates.add("cache_hit")
        if arm == ARM_A and turn.turn != "T0":
            minimum, maximum = arm_a_hit_intervals[turn.turn]
            if not minimum <= turn.cache.request_device_tokens <= maximum:
                gates.add("cache_hit")

    return (
        _Lifetime(
            ordinal,
            arm,
            cache_impl,
            extra_buffer,
            extra_buffer_lazy,
            fresh_server,
            pre_t0_requests,
            pre_t0_warmups,
            startup_identity,
            typed_turns,
        ),
        gates,
    )


def _ordered_gates(values: set[str]) -> list[str]:
    unknown = values - set(_DIAGNOSTIC_GATE_ORDER)
    if unknown:
        raise AssertionError("internal diagnostic gate vocabulary changed")
    return [gate for gate in _DIAGNOSTIC_GATE_ORDER if gate in values]


def _sum_finite(values: tuple[Decimal, ...], *, context: str) -> Decimal:
    try:
        result = sum(values, Decimal(0))
    except DecimalException as exc:
        raise SGLangCachePolicyProtocolError(
            f"{context} aggregate is not finite"
        ) from exc
    if not result.is_finite():
        raise SGLangCachePolicyProtocolError(
            f"{context} aggregate is not finite"
        )
    return result


def _mean_finite(values: tuple[Decimal, ...], *, context: str) -> Decimal:
    if not values:
        raise AssertionError("diagnostic reduction requires non-empty values")
    total = _sum_finite(values, context=context)
    try:
        result = total / Decimal(len(values))
    except DecimalException as exc:
        raise SGLangCachePolicyProtocolError(
            f"{context} aggregate is not finite"
        ) from exc
    if not result.is_finite():
        raise SGLangCachePolicyProtocolError(
            f"{context} aggregate is not finite"
        )
    return result


def _ratio_finite(
    numerator: Decimal, denominator: Decimal, *, context: str
) -> Decimal:
    if denominator <= 0:
        raise SGLangCachePolicyProtocolError(
            f"{context} ratio denominator must be positive"
        )
    try:
        result = numerator / denominator
    except DecimalException as exc:
        raise SGLangCachePolicyProtocolError(
            f"{context} ratio is not finite"
        ) from exc
    if not result.is_finite():
        raise SGLangCachePolicyProtocolError(
            f"{context} ratio is not finite"
        )
    return result


def _json_number(value: Decimal, *, context: str) -> float:
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise SGLangCachePolicyProtocolError(
            f"{context} is not a finite JSON number"
        ) from exc
    if not math.isfinite(result) or (result == 0.0 and value != 0):
        raise SGLangCachePolicyProtocolError(
            f"{context} is not a finite representable JSON number"
        )
    return result


def summarize_cache_policy_campaign(envelope: object) -> dict[str, Any]:
    """Validate and diagnostically reduce an always-unadmitted draft campaign."""

    row = _require_object(envelope, context="cache-policy draft envelope")
    _require_exact_keys(
        row, _ENVELOPE_KEYS, context="cache-policy draft envelope"
    )
    schema_version = _require_int(
        row["schema_version"], context="envelope schema version"
    )
    if schema_version != PROTOCOL_SCHEMA_VERSION:
        raise SGLangCachePolicyProtocolError(
            "cache-policy envelope schema version is unsupported"
        )
    parsed_protocol = _parse_descriptor(row["protocol_descriptor"])
    supplied_protocol_sha256 = _require_sha256(
        row["protocol_sha256"], context="protocol digest"
    )
    if supplied_protocol_sha256 != parsed_protocol.sha256:
        raise SGLangCachePolicyProtocolError(
            "protocol digest does not match the canonical descriptor"
        )
    runtime_digest = _require_pinned_sha256(
        row["provisional_runtime_source_contract_sha256"],
        context="envelope provisional runtime/source contract digest",
    )
    if (
        runtime_digest
        != parsed_protocol.provisional_runtime_source_contract_sha256
    ):
        raise SGLangCachePolicyProtocolError(
            "envelope provisional runtime/source digest does not match descriptor"
        )
    validator_id = _require_string(
        row["validator_id"], context="validator identity"
    )
    if validator_id != VALIDATOR_ID:
        raise SGLangCachePolicyProtocolError(
            "cache-policy validator identity changed"
        )
    raw_lifetimes = row["lifetimes"]
    if type(raw_lifetimes) is not list or len(raw_lifetimes) != len(
        LIFETIME_ORDER
    ):
        raise SGLangCachePolicyProtocolError(
            "cache-policy lifetimes must be an exact four-item JSON list"
        )

    parsed: list[_Lifetime] = []
    lifetime_gates: list[set[str]] = []
    for ordinal, (value, arm) in enumerate(
        zip(raw_lifetimes, LIFETIME_ORDER, strict=True), start=1
    ):
        lifetime, gates = _parse_lifetime(
            value,
            expected_ordinal=ordinal,
            expected_arm=arm,
            arm_a_hit_intervals=parsed_protocol.arm_a_hit_intervals,
        )
        parsed.append(lifetime)
        lifetime_gates.append(gates)

    for turn_index in range(len(TURN_ORDER)):
        identities = {
            (
                lifetime.turns[turn_index].input_tokens,
                lifetime.turns[turn_index].common_prefix_tokens,
            )
            for lifetime in parsed
        }
        if len(identities) != 1:
            for gates in lifetime_gates:
                gates.add("prompt_identity")

    lifetime_summaries: list[dict[str, Any]] = []
    later_wall_by_arm: dict[str, list[Decimal]] = {ARM_A: [], ARM_B: []}
    later_ttft_by_arm: dict[str, list[Decimal]] = {ARM_A: [], ARM_B: []}
    full_wall_by_arm: dict[str, list[Decimal]] = {ARM_A: [], ARM_B: []}
    cold_wall_by_arm: dict[str, list[Decimal]] = {ARM_A: [], ARM_B: []}
    all_gates: set[str] = set()

    for lifetime, gates in zip(parsed, lifetime_gates, strict=True):
        later_wall = _sum_finite(
            (lifetime.turns[1].wall_s, lifetime.turns[2].wall_s),
            context=f"lifetime {lifetime.lifetime_ordinal} later wall",
        )
        later_ttft = _mean_finite(
            (lifetime.turns[1].ttft_s, lifetime.turns[2].ttft_s),
            context=f"lifetime {lifetime.lifetime_ordinal} later TTFT",
        )
        full_wall = _sum_finite(
            tuple(turn.wall_s for turn in lifetime.turns),
            context=f"lifetime {lifetime.lifetime_ordinal} full wall",
        )
        later_wall_by_arm[lifetime.arm].append(later_wall)
        later_ttft_by_arm[lifetime.arm].append(later_ttft)
        full_wall_by_arm[lifetime.arm].append(full_wall)
        cold_wall_by_arm[lifetime.arm].append(lifetime.turns[0].wall_s)
        all_gates.update(gates)
        ordered = _ordered_gates(gates)
        turn_summaries = [
            {
                "turn": turn.turn,
                "input_tokens": turn.input_tokens,
                "common_prefix_tokens": turn.common_prefix_tokens,
                "request_detail_state": "reported",
                "reported_request_device_tokens": (
                    turn.cache.request_device_tokens
                ),
                "reported_request_host_tokens": turn.cache.request_host_tokens,
                "reported_request_storage_tokens": (
                    turn.cache.request_storage_tokens
                ),
                "provisional_native_prefill_device_hit_tokens_delta": (
                    turn.cache.provisional_prefill_device_delta
                ),
                "provisional_native_finished_request_device_hit_tokens_delta": (
                    turn.cache.provisional_finished_request_device_delta
                ),
                "provisional_device_reconciliation_matched": True,
                "native_host_storage_reconciliation_available": False,
                "provisional_prompt_identity_match": (
                    turn.provisional_prompt_identity_match
                ),
                "provisional_correctness_passed": (
                    turn.provisional_correctness_passed
                ),
                "eviction_count": turn.eviction_count,
                "retraction_count": turn.retraction_count,
                "other_request_count": turn.other_request_count,
                "pressure_breach": turn.pressure_breach,
                "ttft_s": _json_number(
                    turn.ttft_s,
                    context=(
                        f"lifetime {lifetime.lifetime_ordinal} {turn.turn} TTFT"
                    ),
                ),
                "wall_s": _json_number(
                    turn.wall_s,
                    context=(
                        f"lifetime {lifetime.lifetime_ordinal} {turn.turn} wall"
                    ),
                ),
            }
            for turn in lifetime.turns
        ]
        lifetime_summaries.append(
            {
                "lifetime_ordinal": lifetime.lifetime_ordinal,
                "arm": lifetime.arm,
                "cache_impl": lifetime.cache_impl,
                "mamba_extra_buffer_of": lifetime.mamba_extra_buffer_of,
                "mamba_extra_buffer_lazy_of": (
                    lifetime.mamba_extra_buffer_lazy_of
                ),
                "provisional_fresh_server_observed": (
                    lifetime.provisional_fresh_server_observed
                ),
                "pre_t0_request_count": lifetime.pre_t0_request_count,
                "pre_t0_warmup_count": lifetime.pre_t0_warmup_count,
                "provisional_startup_identity_match": (
                    lifetime.provisional_startup_identity_match
                ),
                "diagnostic_gate_passed": not ordered,
                "diagnostic_invalid_gate_count": len(ordered),
                "diagnostic_invalid_gates": ordered,
                "later_wall_s": _json_number(
                    later_wall,
                    context=f"lifetime {lifetime.lifetime_ordinal} later wall",
                ),
                "later_ttft_s": _json_number(
                    later_ttft,
                    context=f"lifetime {lifetime.lifetime_ordinal} later TTFT",
                ),
                "full_wall_s": _json_number(
                    full_wall,
                    context=f"lifetime {lifetime.lifetime_ordinal} full wall",
                ),
                "turns": turn_summaries,
            }
        )

    a_later_wall = _mean_finite(
        tuple(later_wall_by_arm[ARM_A]), context="arm A mean later wall"
    )
    b_later_wall = _mean_finite(
        tuple(later_wall_by_arm[ARM_B]), context="arm B mean later wall"
    )
    a_later_ttft = _mean_finite(
        tuple(later_ttft_by_arm[ARM_A]), context="arm A mean later TTFT"
    )
    b_later_ttft = _mean_finite(
        tuple(later_ttft_by_arm[ARM_B]), context="arm B mean later TTFT"
    )
    a_full_wall = _mean_finite(
        tuple(full_wall_by_arm[ARM_A]), context="arm A mean full wall"
    )
    b_full_wall = _mean_finite(
        tuple(full_wall_by_arm[ARM_B]), context="arm B mean full wall"
    )
    a_cold_wall = _mean_finite(
        tuple(cold_wall_by_arm[ARM_A]), context="arm A mean T0 cold wall"
    )
    b_cold_wall = _mean_finite(
        tuple(cold_wall_by_arm[ARM_B]), context="arm B mean T0 cold wall"
    )
    b_over_a = _ratio_finite(
        b_later_wall, a_later_wall, context="B/A later wall"
    )
    a_over_b = _ratio_finite(
        a_later_wall, b_later_wall, context="A/B later wall"
    )

    ordered_all_gates = _ordered_gates(all_gates)
    diagnostic_observations_valid = not ordered_all_gates
    diagnostic_speed_result = "not_evaluated"
    diagnostic_candidate_arm = "none"
    diagnostic_guardrails_evaluated = False
    diagnostic_guardrails_passed = False
    diagnostic_later_ttft_ratio: Decimal | None = None
    diagnostic_full_wall_ratio: Decimal | None = None

    if diagnostic_observations_valid:
        if b_later_wall <= a_later_wall * PROMOTION_RATIO:
            diagnostic_speed_result = "b_faster"
            diagnostic_candidate_arm = ARM_B
            diagnostic_later_ttft_ratio = _ratio_finite(
                b_later_ttft,
                a_later_ttft,
                context="diagnostic B later TTFT guardrail",
            )
            diagnostic_full_wall_ratio = _ratio_finite(
                b_full_wall,
                a_full_wall,
                context="diagnostic B full-wall guardrail",
            )
        elif a_later_wall <= b_later_wall * PROMOTION_RATIO:
            diagnostic_speed_result = "a_faster"
            diagnostic_candidate_arm = ARM_A
            diagnostic_later_ttft_ratio = _ratio_finite(
                a_later_ttft,
                b_later_ttft,
                context="diagnostic A later TTFT guardrail",
            )
            diagnostic_full_wall_ratio = _ratio_finite(
                a_full_wall,
                b_full_wall,
                context="diagnostic A full-wall guardrail",
            )
        else:
            diagnostic_speed_result = "inconclusive"

        if diagnostic_candidate_arm != "none":
            diagnostic_guardrails_evaluated = True
            if (
                diagnostic_later_ttft_ratio is None
                or diagnostic_full_wall_ratio is None
            ):
                raise AssertionError("diagnostic guardrail ratios missing")
            diagnostic_guardrails_passed = (
                diagnostic_later_ttft_ratio <= GUARDRAIL_RATIO
                and diagnostic_full_wall_ratio <= GUARDRAIL_RATIO
            )

    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": parsed_protocol.sha256,
        "provisional_runtime_source_contract_sha256": runtime_digest,
        "provisional_source_tree": parsed_protocol.provisional_source_tree,
        "validator_id": VALIDATOR_ID,
        "protocol_phase": PROTOCOL_PHASE,
        "protocol_status": PROTOCOL_STATUS,
        "protocol_runnable": False,
        "measurement_admissible": False,
        "decision": PROTOCOL_STATUS,
        "selected_arm": "none",
        "draft_blocker_count": len(DRAFT_BLOCKERS),
        "draft_blockers": list(DRAFT_BLOCKERS),
        "native_host_storage_reconciliation_available": False,
        "lifetime_count": len(parsed),
        "arm_a_lifetime_count": len(later_wall_by_arm[ARM_A]),
        "arm_b_lifetime_count": len(later_wall_by_arm[ARM_B]),
        "diagnostic_observations_valid": diagnostic_observations_valid,
        "diagnostic_invalid_gate_count": len(ordered_all_gates),
        "diagnostic_invalid_gates": ordered_all_gates,
        "lifetimes": lifetime_summaries,
        "arm_a_mean_later_wall_s": _json_number(
            a_later_wall, context="arm A mean later wall"
        ),
        "arm_b_mean_later_wall_s": _json_number(
            b_later_wall, context="arm B mean later wall"
        ),
        "b_over_a_later_wall_ratio": _json_number(
            b_over_a, context="B/A later wall ratio"
        ),
        "a_over_b_later_wall_ratio": _json_number(
            a_over_b, context="A/B later wall ratio"
        ),
        "arm_a_mean_later_ttft_s": _json_number(
            a_later_ttft, context="arm A mean later TTFT"
        ),
        "arm_b_mean_later_ttft_s": _json_number(
            b_later_ttft, context="arm B mean later TTFT"
        ),
        "arm_a_mean_full_wall_s": _json_number(
            a_full_wall, context="arm A mean full wall"
        ),
        "arm_b_mean_full_wall_s": _json_number(
            b_full_wall, context="arm B mean full wall"
        ),
        "arm_a_mean_t0_cold_wall_s": _json_number(
            a_cold_wall, context="arm A mean T0 cold wall"
        ),
        "arm_b_mean_t0_cold_wall_s": _json_number(
            b_cold_wall, context="arm B mean T0 cold wall"
        ),
        "diagnostic_speed_result": diagnostic_speed_result,
        "diagnostic_candidate_arm": diagnostic_candidate_arm,
        "diagnostic_guardrails_evaluated": diagnostic_guardrails_evaluated,
        "diagnostic_later_ttft_ratio": (
            None
            if diagnostic_later_ttft_ratio is None
            else _json_number(
                diagnostic_later_ttft_ratio,
                context="diagnostic later TTFT guardrail ratio",
            )
        ),
        "diagnostic_full_wall_ratio": (
            None
            if diagnostic_full_wall_ratio is None
            else _json_number(
                diagnostic_full_wall_ratio,
                context="diagnostic full-wall guardrail ratio",
            )
        ),
        "diagnostic_guardrails_passed": diagnostic_guardrails_passed,
        "diagnostic_speed_ratio_threshold": float(PROMOTION_RATIO),
        "diagnostic_guardrail_ratio_threshold": float(GUARDRAIL_RATIO),
    }
