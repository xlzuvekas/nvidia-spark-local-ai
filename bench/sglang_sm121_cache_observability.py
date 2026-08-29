"""Pinned B0 cache-observability contract for the SM121 storage candidate.

The completed native-NVMe canary deliberately runs with Radix disabled.  This
module defines a separate, equally narrow follow-on lane that observes that
cache-off configuration before any cache-enabled configuration is admitted.
It is not a throughput experiment and it cannot select a cache-policy arm.

Only scalar source hashes, startup facts, response-detail states, and
Prometheus counters/gauges may leave the ignored run directory.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .sglang_sm121_storage import (
    SM121_STORAGE_PROFILE_ID,
    SM121_STORAGE_SOURCE_TREE,
    SM121StorageCandidateError,
    is_sm121_storage_candidate,
)


SM121_CACHE_OBSERVABILITY_SUITE_ID = (
    "qwen38-flash-next-sm121-triton-storage-cache-observability-canary"
)
SM121_CACHE_OBSERVABILITY_EXECUTION_MODE = (
    "sm121_storage_cache_observability_b0_fresh_lifetime"
)
SM121_CACHE_STATIC_ATTESTATION_EVENT = "sm121_cache_static_attestation"
SM121_CACHE_RUNTIME_ATTESTATION_EVENT = "sm121_cache_runtime_attestation"
SM121_CACHE_ZERO_HIT_EVENT = "sm121_cache_zero_hit_observation"
SM121_CACHE_ZERO_HIT_CASE_ID = "sm121-cache-zero-hit-observability-v1"
SM121_CACHE_ZERO_HIT_PROMPT = (
    "This is a synthetic cache-observability request. "
    "Reply with exactly CACHE-OBS-41."
)
SM121_CACHE_ZERO_HIT_EXPECTED_RESPONSE = "CACHE-OBS-41"
SM121_CACHE_ZERO_HIT_MAX_OUTPUT_TOKENS = 16
SM121_CACHE_ZERO_HIT_PROMPT_SHA256 = (
    "sha256:7465e7388b09b9b4fba1acee41ddaf579e9ec66ffce8e6fe444ca58c267e7ad2"
)
SM121_CACHE_ZERO_HIT_REQUEST_CONTRACT_SHA256 = (
    "sha256:aebacb6a7767f6bcd82caff7294d08e6d323281f463770c2c9187457af6f2c8a"
)

# These digests were read from the immutable local image after its Docker
# config ID and final source-tree label had already been verified.  The key
# names intentionally identify reviewed roles, not host or container paths.
SM121_CACHE_SOURCE_DIGESTS = {
    "arg_overrides_sha256": (
        "sha256:d3a1ccc96359d544b124ca9d666e161f2de4cdbfa649b8f0239bcdb6fe3f8692"
    ),
    "cache_registry_sha256": (
        "sha256:b60134d1ba8c6fdad7b30f268b5d96264102bdfe9881e2401264cb7f99fd31cd"
    ),
    "cache_builder_sha256": (
        "sha256:0907673eb3cafe12dd7d84a0f5874f7054b789bf8984fbf7914e0c100673e861"
    ),
    "runtime_context_sha256": (
        "sha256:a41f510571622a89a4549fa62f89fe47f55a9b9a715392564ea731d3cac720ba"
    ),
    "metrics_collector_sha256": (
        "sha256:9b2469b149e58d6e427dfafe82af073e61e04c825ff7d2b5dec138fd77b5c134"
    ),
    "openai_utils_sha256": (
        "sha256:6057816db2a0851dada70399266f4c678b0a56c576e145d3dfd442e2b2300624"
    ),
    "openai_protocol_sha256": (
        "sha256:3f19b264c155f773400ba5a29780d8d28a57c477c9d6c587353821be47da6d01"
    ),
    "openai_serving_chat_sha256": (
        "sha256:652a3b7cdb77e6552df902735f47a5ef87fdb9fb0c05125429e296016b27ccd4"
    ),
    "openai_usage_processor_sha256": (
        "sha256:883c4c3ba95eb52fa57d5bf732d128063bac0b032f484776331220f61b7498a5"
    ),
    "http_server_sha256": (
        "sha256:979a5fd0be624a5f0ddb9ef6d020b4d2f079a558413b5edccb75a02149882412"
    ),
}
SM121_CACHE_STATIC_ASSERTIONS = {
    "cache_off_selects_chunk_cache": True,
    "cache_off_suppresses_mamba_extra_buffer": True,
    "cache_off_suppresses_mamba_extra_buffer_lazy": True,
    "cache_startup_log_exposes_impl": True,
    "cache_detail_extension_available": True,
    "zero_cache_details_may_be_omitted_or_null": True,
    "native_cache_counters_exposed": True,
    "cached_total_counter_absent_when_zero": True,
    "native_residency_gauges_exposed": True,
}

SM121_CACHE_RUNTIME_EXPECTED = {
    "cache_impl": "ChunkCache",
    "cache_source": "default",
    "disable_radix_cache": True,
    "hicache_attached": False,
    "hybrid_ssm": True,
    "hybrid_swa": False,
    "mamba_extra_buffer_enabled": False,
    "mamba_extra_buffer_lazy_enabled": False,
    "streaming_wrapped": False,
}

_SOURCE_DIGEST_FIELDS = frozenset(SM121_CACHE_SOURCE_DIGESTS)
_STATIC_ASSERTION_FIELDS = frozenset(SM121_CACHE_STATIC_ASSERTIONS)
_RUNTIME_FIELDS = frozenset(SM121_CACHE_RUNTIME_EXPECTED)
_DETAIL_STATES = frozenset(
    {"omitted", "null", "zero_details", "nonzero_details", "unexpected"}
)
_ZERO_HIT_BASES = frozenset(
    {
        "explicit_details",
        "omitted_or_null_with_native_counters",
        "not_admitted",
    }
)
_METRIC_PREFIXES = ("before", "after", "delta")
_HIT_METRICS = (
    "prefill_device_hit_tokens",
    "prefill_host_hit_tokens",
    "prefill_storage_hit_tokens",
    "cached_total_tokens",
    "cached_device_tokens",
    "cached_host_tokens",
    "cached_storage_tokens",
)
_POOL_METRICS = (
    "kv_available_tokens",
    "kv_evictable_tokens",
    "kv_used_tokens",
    "mamba_available_tokens",
    "mamba_evictable_tokens",
    "mamba_used_tokens",
)
SM121_CACHE_OBSERVABILITY_METRIC_FIELDS = (
    "prefill_input_tokens",
    *_HIT_METRICS,
    *_POOL_METRICS,
)
SM121_CACHE_OBSERVABILITY_CACHED_SERIES = (
    "total",
    "device",
    "host",
    "storage",
)


class SM121CacheObservabilityError(ValueError):
    """Raised when B0 deviates from its small, fixed admission contract."""


def _value(item: Any, field: str) -> object:
    if isinstance(item, Mapping):
        return item.get(field)
    return getattr(item, field, None)


def _require(value: object, expected: object, field: str) -> None:
    if type(expected) is bool:
        matches = type(value) is bool and value is expected
    else:
        matches = value == expected
    if not matches:
        raise SM121CacheObservabilityError(f"{field} does not match the B0 contract")


def _require_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SM121CacheObservabilityError(f"{field} must be a non-negative integer")
    return value


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SM121CacheObservabilityError(f"{field} must be an integer")
    return value


def _require_optional_nonnegative_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _require_nonnegative_int(value, field)


def _require_nonnegative_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SM121CacheObservabilityError(f"{field} must be a finite duration")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise SM121CacheObservabilityError(f"{field} must be a finite duration")
    return parsed


def sm121_cache_zero_hit_request_contract() -> dict[str, object]:
    """Return the scalar-only, fixed B0 request shape without its prompt text."""

    return {
        "schema_version": 1,
        "endpoint": "/v1/chat/completions",
        "prompt_sha256": SM121_CACHE_ZERO_HIT_PROMPT_SHA256,
        "max_tokens": SM121_CACHE_ZERO_HIT_MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
        "n": 1,
        "stream": False,
        "return_cached_tokens_details": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def validate_sm121_cache_zero_hit_request_contract() -> None:
    """Fail closed if the code-visible B0 request shape diverges from its pin."""

    prompt_digest = "sha256:" + hashlib.sha256(
        SM121_CACHE_ZERO_HIT_PROMPT.encode("utf-8")
    ).hexdigest()
    _require(
        prompt_digest,
        SM121_CACHE_ZERO_HIT_PROMPT_SHA256,
        "B0 synthetic prompt digest",
    )
    serialized = json.dumps(
        sm121_cache_zero_hit_request_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    _require(
        "sha256:" + hashlib.sha256(serialized).hexdigest(),
        SM121_CACHE_ZERO_HIT_REQUEST_CONTRACT_SHA256,
        "B0 request contract",
    )


def sm121_cache_zero_hit_request_body(served_name: object) -> dict[str, object]:
    """Build the exact non-streaming B0 body while retaining no response data."""

    if not isinstance(served_name, str) or not served_name:
        raise SM121CacheObservabilityError("B0 served model name is invalid")
    validate_sm121_cache_zero_hit_request_contract()
    return {
        "model": served_name,
        "messages": [{"role": "user", "content": SM121_CACHE_ZERO_HIT_PROMPT}],
        "max_tokens": SM121_CACHE_ZERO_HIT_MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
        "n": 1,
        "stream": False,
        "return_cached_tokens_details": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def is_sm121_cache_observability_plan(model: Any, suite: Any) -> bool:
    """Return whether an exact model/suite pair selects B0."""

    return (
        is_sm121_storage_candidate(model)
        and _value(suite, "id") == SM121_CACHE_OBSERVABILITY_SUITE_ID
    )


def validate_sm121_cache_observability_suite(suite: Any) -> None:
    """Require the two fixed B0 cases in their quality-then-observation order."""

    _require(_value(suite, "id"), SM121_CACHE_OBSERVABILITY_SUITE_ID, "suite.id")
    _require(_value(suite, "protocol_digest"), None, "suite.protocol_digest")
    _require(
        _value(suite, "description"),
        (
            "B0 cache-off observability for the pre-admission Qwen3.8 Flash-Next "
            "SM121 native-NVMe runtime. It first requires a fresh-process exact-answer "
            "quality gate, then records one non-streaming zero-hit cache-detail and "
            "settled-metrics observation. This is not a cache-policy or throughput "
            "comparison."
        ),
        "suite.description",
    )
    cases = _value(suite, "cases")
    if not isinstance(cases, (list, tuple)) or len(cases) != 2:
        raise SM121CacheObservabilityError("B0 suite must contain exactly two cases")
    expected = (
        {
            "id": "synthetic-exact-answer-v2",
            "kind": "quality",
            "requires": ("chat",),
            "warmups": 0,
            "repetitions": 1,
            "max_output_tokens": 512,
            "temperature": 0.0,
            "concurrency": 1,
            "prompt_repetitions": 0,
            "max_turns": 1,
        },
        {
            "id": SM121_CACHE_ZERO_HIT_CASE_ID,
            "kind": "capability",
            "requires": ("chat",),
            "warmups": 0,
            "repetitions": 1,
            "max_output_tokens": SM121_CACHE_ZERO_HIT_MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
            "concurrency": 1,
            "prompt_repetitions": 0,
            "max_turns": 1,
        },
    )
    for index, (case, required) in enumerate(zip(cases, expected, strict=True)):
        for field, value in required.items():
            actual = _value(case, field)
            if field == "requires" and isinstance(actual, (list, tuple)):
                actual = tuple(actual)
            if actual != value:
                raise SM121CacheObservabilityError(
                    f"B0 suite case {index} field {field} does not match its pin"
                )


def validate_sm121_cache_static_attestation_event(event: object) -> None:
    """Validate a scalar-only final-tree cache-semantics attestation."""

    if type(event) is not dict:
        raise SM121CacheObservabilityError("B0 static attestation is not an object")
    expected_fields = {
        "event",
        "candidate_source_tree",
        *_SOURCE_DIGEST_FIELDS,
        *_STATIC_ASSERTION_FIELDS,
    }
    fields = frozenset(event)
    if fields not in {frozenset(expected_fields), frozenset(expected_fields | {"timestamp"})}:
        raise SM121CacheObservabilityError("B0 static attestation fields are invalid")
    if event.get("event") != SM121_CACHE_STATIC_ATTESTATION_EVENT:
        raise SM121CacheObservabilityError("B0 static attestation event is invalid")
    _require(event.get("candidate_source_tree"), SM121_STORAGE_SOURCE_TREE, "source tree")
    for field, expected in SM121_CACHE_SOURCE_DIGESTS.items():
        _require(event.get(field), expected, field)
    for field, expected in SM121_CACHE_STATIC_ASSERTIONS.items():
        _require(event.get(field), expected, field)
    if "timestamp" in event and not isinstance(event["timestamp"], str):
        raise SM121CacheObservabilityError("B0 static attestation timestamp is invalid")


def validate_sm121_cache_runtime_attestation_event(event: object) -> None:
    """Validate the scalar cache-off runtime identity observed at startup."""

    if type(event) is not dict:
        raise SM121CacheObservabilityError("B0 runtime attestation is not an object")
    expected_fields = {"event", *_RUNTIME_FIELDS}
    fields = frozenset(event)
    if fields not in {frozenset(expected_fields), frozenset(expected_fields | {"timestamp"})}:
        raise SM121CacheObservabilityError("B0 runtime attestation fields are invalid")
    if event.get("event") != SM121_CACHE_RUNTIME_ATTESTATION_EVENT:
        raise SM121CacheObservabilityError("B0 runtime attestation event is invalid")
    for field, expected in SM121_CACHE_RUNTIME_EXPECTED.items():
        _require(event.get(field), expected, field)
    if "timestamp" in event and not isinstance(event["timestamp"], str):
        raise SM121CacheObservabilityError("B0 runtime attestation timestamp is invalid")


def validate_sm121_cache_zero_hit_event(event: object) -> None:
    """Validate a B0 response/metric observation without inferring a cache hit."""

    if type(event) is not dict:
        raise SM121CacheObservabilityError("B0 zero-hit observation is not an object")
    fields = {
        "event",
        "case_id",
        "protocol_case_id",
        "attempt_id",
        "request_contract_sha256",
        "cache_details_requested",
        "streaming",
        "thinking_disabled",
        "response_detail_state",
        "usage_detail_state",
        "response_device_cached_tokens",
        "response_host_cached_tokens",
        "response_storage_cached_tokens",
        "usage_cached_tokens",
        "metrics_available",
        "metrics_before_polls",
        "metrics_after_polls",
        "metrics_before_settle_s",
        "metrics_after_settle_s",
        "metrics_before_settled",
        "metrics_after_settled",
        "zero_hit_basis",
        "zero_hit_admitted",
    }
    for prefix in _METRIC_PREFIXES:
        fields.add(f"{prefix}_prefill_input_tokens")
        for metric in _HIT_METRICS + _POOL_METRICS:
            fields.add(f"{prefix}_{metric}")
    for prefix in ("before", "after"):
        for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
            fields.add(f"{prefix}_cached_{source}_series_present")
    actual_fields = frozenset(event)
    accepted = {frozenset(fields), frozenset(fields | {"timestamp"})}
    if actual_fields not in accepted:
        raise SM121CacheObservabilityError("B0 zero-hit observation fields are invalid")
    if event.get("event") != SM121_CACHE_ZERO_HIT_EVENT:
        raise SM121CacheObservabilityError("B0 zero-hit observation event is invalid")
    case_id = event.get("case_id")
    case_prefix = SM121_CACHE_ZERO_HIT_CASE_ID + "--"
    case_suffix = (
        case_id[len(case_prefix) :]
        if isinstance(case_id, str) and case_id.startswith(case_prefix)
        else ""
    )
    if len(case_suffix) != 12 or any(
        character not in "0123456789abcdef" for character in case_suffix
    ):
        raise SM121CacheObservabilityError("B0 frozen case ID is invalid")
    _require(
        event.get("protocol_case_id"),
        SM121_CACHE_ZERO_HIT_CASE_ID,
        "B0 protocol case ID",
    )
    if not isinstance(event.get("attempt_id"), str) or not event["attempt_id"]:
        raise SM121CacheObservabilityError("B0 zero-hit attempt ID is invalid")
    _require(
        event.get("request_contract_sha256"),
        SM121_CACHE_ZERO_HIT_REQUEST_CONTRACT_SHA256,
        "B0 request contract",
    )
    for field, expected in (
        ("cache_details_requested", True),
        ("streaming", False),
        ("thinking_disabled", True),
    ):
        _require(event.get(field), expected, f"B0 {field}")
    for field in ("response_detail_state", "usage_detail_state"):
        if event.get(field) not in _DETAIL_STATES:
            raise SM121CacheObservabilityError(f"B0 {field} is invalid")
    for field in (
        "response_device_cached_tokens",
        "response_host_cached_tokens",
        "response_storage_cached_tokens",
        "usage_cached_tokens",
        "metrics_before_polls",
        "metrics_after_polls",
    ):
        _require_optional_nonnegative_int(event.get(field), field)
    for field in ("metrics_before_polls", "metrics_after_polls"):
        _require_nonnegative_int(event.get(field), field)
    for field in ("metrics_before_settle_s", "metrics_after_settle_s"):
        _require_nonnegative_float(event.get(field), field)
    if type(event.get("metrics_available")) is not bool:
        raise SM121CacheObservabilityError("B0 metrics availability is invalid")
    for field in ("metrics_before_settled", "metrics_after_settled"):
        if type(event.get(field)) is not bool:
            raise SM121CacheObservabilityError(f"B0 {field} is invalid")
    if event.get("zero_hit_basis") not in _ZERO_HIT_BASES:
        raise SM121CacheObservabilityError("B0 zero-hit basis is invalid")
    if type(event.get("zero_hit_admitted")) is not bool:
        raise SM121CacheObservabilityError("B0 zero-hit admission is invalid")
    for prefix in _METRIC_PREFIXES:
        _require_nonnegative_int(event.get(f"{prefix}_prefill_input_tokens"), f"{prefix}_prefill_input_tokens")
        for metric in _HIT_METRICS:
            _require_nonnegative_int(event.get(f"{prefix}_{metric}"), f"{prefix}_{metric}")
        for metric in _POOL_METRICS:
            value = event.get(f"{prefix}_{metric}")
            if prefix in {"before", "after"}:
                _require_nonnegative_int(value, f"{prefix}_{metric}")
            else:
                _require_int(value, f"{prefix}_{metric}")
    for prefix in ("before", "after"):
        for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
            if type(event.get(f"{prefix}_cached_{source}_series_present")) is not bool:
                raise SM121CacheObservabilityError(
                    f"B0 {prefix} cached {source} series marker is invalid"
                )
    for metric in SM121_CACHE_OBSERVABILITY_METRIC_FIELDS:
        if event[f"delta_{metric}"] != (
            event[f"after_{metric}"] - event[f"before_{metric}"]
        ):
            raise SM121CacheObservabilityError(
                f"B0 delta for {metric} does not reconcile its snapshots"
            )
    if "timestamp" in event and not isinstance(event["timestamp"], str):
        raise SM121CacheObservabilityError("B0 zero-hit timestamp is invalid")

    response_state = event["response_detail_state"]
    response_counts = tuple(
        event[field]
        for field in (
            "response_device_cached_tokens",
            "response_host_cached_tokens",
            "response_storage_cached_tokens",
        )
    )
    usage_state = event["usage_detail_state"]
    usage_count = event["usage_cached_tokens"]
    if response_state == "zero_details":
        if response_counts != (0, 0, 0):
            raise SM121CacheObservabilityError(
                "B0 explicit response cache details must all be zero"
            )
    elif response_state == "nonzero_details":
        if (
            any(value is None for value in response_counts)
            or not any(value > 0 for value in response_counts)
        ):
            raise SM121CacheObservabilityError(
                "B0 nonzero response cache details are invalid"
            )
    elif any(value is not None for value in response_counts):
        raise SM121CacheObservabilityError(
            "B0 absent or unexpected response details must remain unavailable"
        )
    if usage_state == "zero_details":
        if usage_count != 0:
            raise SM121CacheObservabilityError(
                "B0 explicit usage cache detail must be zero"
            )
    elif usage_state == "nonzero_details":
        if usage_count is None or usage_count <= 0:
            raise SM121CacheObservabilityError(
                "B0 nonzero usage cache detail is invalid"
            )
    elif usage_count is not None:
        raise SM121CacheObservabilityError(
            "B0 absent or unexpected usage detail must remain unavailable"
        )

    metrics_available = event["metrics_available"]
    settled = event["metrics_before_settled"] and event["metrics_after_settled"]
    if metrics_available and not settled:
        raise SM121CacheObservabilityError(
            "B0 available metrics require settled before and after snapshots"
        )
    if not metrics_available and settled:
        raise SM121CacheObservabilityError(
            "B0 unavailable metrics cannot be marked settled"
        )
    if settled and (
        event["metrics_before_polls"] < 2 or event["metrics_after_polls"] < 2
    ):
        raise SM121CacheObservabilityError(
            "B0 settled metrics require two matching polls per snapshot"
        )
    admitted, expected_basis = derive_sm121_cache_zero_hit_admission(event)
    if event["zero_hit_admitted"] is not admitted:
        raise SM121CacheObservabilityError(
            "B0 zero-hit admission does not match its scalar observation"
        )
    if event["zero_hit_basis"] != expected_basis:
        raise SM121CacheObservabilityError(
            "B0 zero-hit basis does not match its scalar observation"
        )


def derive_sm121_cache_zero_hit_admission(event: Mapping[str, object]) -> tuple[bool, str]:
    """Derive B0's decision from already-normalized scalar event fields."""

    response_state = event.get("response_detail_state")
    usage_state = event.get("usage_detail_state")
    explicit_details = (
        response_state == "zero_details" and usage_state == "zero_details"
    )
    omitted_or_null_details = (
        response_state in {"omitted", "null"}
        and usage_state in {"omitted", "null"}
    )
    settled = bool(
        event.get("metrics_before_settled") is True
        and event.get("metrics_after_settled") is True
    )
    zero_hit_counters = all(
        event.get(f"delta_{metric}") == 0 for metric in _HIT_METRICS
    )
    admitted = bool(
        event.get("metrics_available") is True
        and settled
        and event.get("delta_prefill_input_tokens", 0) > 0
        and zero_hit_counters
        and (explicit_details or omitted_or_null_details)
    )
    basis = (
        "explicit_details"
        if admitted and explicit_details
        else (
            "omitted_or_null_with_native_counters"
            if admitted and omitted_or_null_details
            else "not_admitted"
        )
    )
    return admitted, basis


def sm121_cache_observability_lifecycle_issues(
    events: Sequence[Mapping[str, object]], *, planned_case_ids: Sequence[str]
) -> tuple[dict[str, object], ...]:
    """Return topology violations for one completed, fresh-server B0 run.

    B0 deliberately permits a terminal *partial* summary when the observation
    is well-formed but not admitted.  It does not permit a resumed journal,
    primer, second server, omitted quality gate, or a loosely attached cache
    event.  The checks below operate only on event shape, ordering, and scalar
    pins; they never copy raw response values into a finding.
    """

    def issue(code: str, message: str, **context: object) -> None:
        issues.append({"code": code, "message": message, **context})

    issues: list[dict[str, object]] = []
    expected_case_ids = tuple(planned_case_ids)
    if (
        len(expected_case_ids) != 2
        or any(not isinstance(case_id, str) or not case_id for case_id in expected_case_ids)
        or len(set(expected_case_ids)) != 2
        or not expected_case_ids[1].startswith(SM121_CACHE_ZERO_HIT_CASE_ID + "--")
    ):
        return (
            {
                "code": "b0_invalid_planned_case_order",
                "message": "B0 requires two distinct frozen cases in its pinned order",
            },
        )
    if any(not isinstance(event, Mapping) for event in events):
        for index, event in enumerate(events):
            if not isinstance(event, Mapping):
                issue(
                    "b0_non_object_event",
                    "B0 journal contains a non-object event",
                    event_index=index,
                )
        return tuple(issues)

    expected_events = expected_sm121_cache_observability_event_counts()
    allowed_events = frozenset(expected_events)
    positions = {
        name: [index for index, event in enumerate(events) if event.get("event") == name]
        for name in expected_events
    }
    for name, expected in expected_events.items():
        actual = len(positions[name])
        if actual != expected:
            issue(
                "b0_event_count",
                "B0 event count does not match the frozen topology",
                event=name,
                expected=expected,
                actual=actual,
            )
    for index, event in enumerate(events):
        event_type = event.get("event")
        if event_type not in allowed_events:
            issue(
                "b0_unexpected_event",
                "B0 journal contains an event outside its frozen topology",
                event_index=index,
            )
    if any(len(positions[name]) != count for name, count in expected_events.items()):
        return tuple(issues)

    run_start = positions["run_start"][0]
    measurement_started = positions["measurement_started"][0]
    static_index = positions[SM121_CACHE_STATIC_ATTESTATION_EVENT][0]
    runtime_index = positions[SM121_CACHE_RUNTIME_ATTESTATION_EVENT][0]
    ready_index = positions["server_ready"][0]
    stopped_index = positions["server_stopped"][0]
    measurement_complete = positions["measurement_complete"][0]
    run_complete = positions["run_complete"][0]
    zero_index = positions[SM121_CACHE_ZERO_HIT_EVENT][0]
    if run_start != 0:
        issue("b0_run_start_not_first", "B0 run_start must be the first journal event")
    if events[run_start].get("execution_mode") != SM121_CACHE_OBSERVABILITY_EXECUTION_MODE:
        issue("b0_execution_mode", "B0 run_start has the wrong execution mode")
    if events[run_complete].get("status") != "completed":
        issue("b0_run_complete_status", "B0 run_complete must record completed")
    if run_complete != len(events) - 1:
        issue("b0_run_complete_not_final", "B0 run_complete must be final")
    if not (
        run_start
        < measurement_started
        < static_index
        < runtime_index
        < ready_index
        < stopped_index
        < measurement_complete
        < run_complete
    ):
        issue("b0_lifecycle_order", "B0 lifecycle events are out of order")
    if runtime_index + 1 != ready_index:
        issue(
            "b0_runtime_attestation_order",
            "B0 runtime attestation must immediately precede server_ready",
        )

    try:
        validate_sm121_cache_static_attestation_event(events[static_index])
    except SM121CacheObservabilityError as error:
        issue("b0_static_attestation", str(error))
    try:
        validate_sm121_cache_runtime_attestation_event(events[runtime_index])
    except SM121CacheObservabilityError as error:
        issue("b0_runtime_attestation", str(error))
    try:
        validate_sm121_cache_zero_hit_event(events[zero_index])
    except SM121CacheObservabilityError as error:
        issue("b0_zero_hit_observation", str(error))

    ready = events[ready_index]
    stopped = events[stopped_index]
    if ready.get("backend") != "sglang":
        issue("b0_ready_backend", "B0 server_ready must identify SGLang")
    if ready.get("fresh_server_lifetime") != 1 or stopped.get("fresh_server_lifetime") != 1:
        issue("b0_lifetime", "B0 must use exactly fresh server lifetime one")
    if ready.get("first_inference_is_case") is not True:
        issue("b0_first_inference", "B0 quality must be the first chat inference")
    if ready.get("case_id") != expected_case_ids[0]:
        issue("b0_ready_case", "B0 server_ready must bind the quality case")

    case_starts = positions["case_start"]
    case_completes = positions["case_complete"]
    if [events[index].get("case_id") for index in case_starts] != list(expected_case_ids):
        issue("b0_case_start_order", "B0 cases did not start in frozen order")
    if [events[index].get("case_id") for index in case_completes] != list(expected_case_ids):
        issue("b0_case_complete_order", "B0 cases did not complete in frozen order")
    if not (
        ready_index < case_starts[0] < case_completes[0] < case_starts[1]
        < case_completes[1] < stopped_index
    ):
        issue("b0_case_order", "B0 case blocks are outside the one server lifetime")

    request_positions = positions["request_complete"]
    expected_requests = (4, 1)
    case_request_positions: list[list[int]] = []
    for case_number, (case_start, case_complete, expected_count) in enumerate(
        zip(case_starts, case_completes, expected_requests, strict=True)
    ):
        start_event = events[case_start]
        complete_event = events[case_complete]
        expected_case_id = expected_case_ids[case_number]
        attempt_id = start_event.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            issue(
                "b0_case_attempt",
                "B0 case_start must carry an attempt ID",
                case_number=case_number,
            )
        for event_index, event in ((case_start, start_event), (case_complete, complete_event)):
            if event.get("case_id") != expected_case_id:
                issue(
                    "b0_case_id",
                    "B0 case event has the wrong frozen case ID",
                    event_index=event_index,
                )
            if event.get("attempt_id") != attempt_id:
                issue(
                    "b0_case_attempt_mismatch",
                    "B0 case event does not match its start attempt",
                    event_index=event_index,
                )
        requests = [
            index
            for index in request_positions
            if case_start < index < case_complete
        ]
        case_request_positions.append(requests)
        if len(requests) != expected_count:
            issue(
                "b0_case_request_count",
                "B0 case has the wrong number of measured requests",
                case_number=case_number,
                expected=expected_count,
                actual=len(requests),
            )
        for request_index in requests:
            request = events[request_index]
            if (
                request.get("case_id") != expected_case_id
                or request.get("attempt_id") != attempt_id
            ):
                issue(
                    "b0_request_binding",
                    "B0 request does not match its case attempt",
                    event_index=request_index,
                )
    if case_completes[0] < len(events) and events[case_completes[0]].get("validation_passed") is not True:
        issue("b0_quality_gate", "B0 exact-answer quality gate was not clean")

    zero_event = events[zero_index]
    zero_attempt = events[case_starts[1]].get("attempt_id")
    if (
        zero_event.get("case_id") != expected_case_ids[1]
        or zero_event.get("attempt_id") != zero_attempt
    ):
        issue("b0_zero_binding", "B0 zero-hit event is not bound to its case attempt")
    second_requests = case_request_positions[1]
    if len(second_requests) == 1 and not (
        case_starts[1] < zero_index < second_requests[0] < case_completes[1]
    ):
        issue(
            "b0_zero_hit_order",
            "B0 zero-hit observation must immediately precede its scalar request record",
        )
    if len(second_requests) == 1 and zero_index + 1 != second_requests[0]:
        issue(
            "b0_zero_request_adjacency",
            "B0 zero-hit observation must be adjacent to its request record",
        )
    if events[case_completes[1]].get("validation_passed") is not zero_event.get(
        "zero_hit_admitted"
    ):
        issue(
            "b0_capability_validation",
            "B0 capability result must equal the recomputed zero-hit admission",
        )
    return tuple(issues)


def expected_sm121_cache_observability_event_counts() -> dict[str, int]:
    """Return the frozen successful B0 lifecycle counts for tests/export."""

    return {
        "case_complete": 2,
        "case_start": 2,
        "measurement_complete": 1,
        "measurement_started": 1,
        "request_complete": 5,
        "run_complete": 1,
        "run_start": 1,
        "server_ready": 1,
        "server_stopped": 1,
        SM121_CACHE_STATIC_ATTESTATION_EVENT: 1,
        SM121_CACHE_RUNTIME_ATTESTATION_EVENT: 1,
        SM121_CACHE_ZERO_HIT_EVENT: 1,
    }


def validate_sm121_cache_observability_candidate(model: Any) -> None:
    """Make the B0 selector explicitly depend on the unchanged target profile."""

    if not is_sm121_storage_candidate(model):
        raise SM121CacheObservabilityError("B0 requires the SM121 storage profile")
    if _value(model, "id") != SM121_STORAGE_PROFILE_ID:
        raise SM121CacheObservabilityError("B0 storage profile identity is invalid")
    # The base candidate validator remains the source of truth for every
    # storage, image, argument, containment, and PLE-reader pin.
    try:
        from .sglang_sm121_storage import validate_sm121_storage_candidate

        validate_sm121_storage_candidate(model)
    except SM121StorageCandidateError as error:
        raise SM121CacheObservabilityError(str(error)) from error
