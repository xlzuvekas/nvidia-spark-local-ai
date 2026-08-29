"""Immutable non-evidence admission contract for the prospective 8K profile.

This is intentionally narrower than the paired chunked-prefill performance
lane.  It admits only the exact prospective V3 B profile through two fresh
lifetimes: the existing four-item quality gate and one cache-cold 60K T0.
It records no request wall time, TPS, score, or ratio, so a successful
admission cannot be confused with a reusable performance observation.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .journal import content_hash
from .sglang_sm121_cache_observability import (
    SM121_CACHE_OBSERVABILITY_CACHED_SERIES,
    SM121_CACHE_SOURCE_DIGESTS,
)
from .sglang_sm121_cache_semantic import (
    SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED,
    SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
)
from .sglang_sm121_chunked_prefill_performance import (
    SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S,
    SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MAX_TOKENS,
    SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MIN_TOKENS,
    SM121_CHUNKED_PREFILL_PERFORMANCE_MAX_MAMBA_CACHE_SIZE,
    SM121_CHUNKED_PREFILL_PERFORMANCE_METRIC_FIELDS,
    SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CAMPAIGN_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CANDIDATE_CHUNK_SIZE,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CANDIDATE_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CASE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CONTROL_CHUNK_SIZE,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CONTROL_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_EXECUTION_MODE,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_SUITE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY,
    SM121ChunkedPrefillPerformanceError,
    sm121_chunked_prefill_performance_study,
    validate_sm121_chunked_prefill_performance_candidate,
)
from .sglang_sm121_storage import SM121_STORAGE_SOURCE_TREE


SM121_CHUNKED_PREFILL_8K_ADMISSION_ID = (
    "qwen38-flash-next-sm121-chunked-prefill-8k-admission-v1"
)
SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE_ID = (
    "qwen38-flash-next-sm121-triton-storage-chunked-prefill-8k-admission-v1"
)
SM121_CHUNKED_PREFILL_8K_ADMISSION_EXECUTION_MODE = (
    "sm121_storage_chunked_prefill_8k_admission_fresh_lifetimes_v1"
)
SM121_CHUNKED_PREFILL_8K_ADMISSION_TIMED_CASE_ID = (
    "sm121-chunked-prefill-60k-static-history-8k-admission-v1"
)
SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE_DESCRIPTION = (
    "Admission-only two-lifetime validation of the prospective 8K current-SM121 "
    "native-NVMe Qwen3.8 Flash-Next cache-on bundle. It proves exact quality "
    "and a cache-cold 60K T0 request can complete safely; it records no "
    "comparative timing result and does not claim agentic coding performance."
)
SM121_CHUNKED_PREFILL_8K_ADMISSION_CELL_TIMEOUT_S = (
    SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S
)
SM121_CHUNKED_PREFILL_8K_ADMISSION_STATIC_EVENT = (
    "sm121_chunked_prefill_8k_admission_static_attestation"
)
SM121_CHUNKED_PREFILL_8K_ADMISSION_RUNTIME_EVENT = (
    "sm121_chunked_prefill_8k_admission_runtime_attestation"
)
SM121_CHUNKED_PREFILL_8K_ADMISSION_T0_EVENT = (
    "sm121_chunked_prefill_8k_admission_t0_observation"
)
SM121_CHUNKED_PREFILL_8K_ADMISSION_LIFETIME_PHASES = {
    1: "quality",
    2: "cold_t0",
}
SM121_CHUNKED_PREFILL_8K_ADMISSION_RUNTIME_EXPECTED = dict(
    SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED["A"]
)
SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_GENERIC = "admission_failure"
SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_EXACT_RESPONSE = (
    "cold_exact_response"
)
SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_PROMPT_IDS = (
    "cold_prompt_ids"
)
SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_RESPONSE_CONTRACT = (
    "cold_response_contract"
)
SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_REQUEST_CONTRACT = (
    "cold_request_contract"
)
SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_TRANSPORT = (
    "cold_transport"
)
SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_HTTP = "cold_http"
SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODES = frozenset(
    {
        SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_GENERIC,
        SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_EXACT_RESPONSE,
        SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_PROMPT_IDS,
        SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_RESPONSE_CONTRACT,
        SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_REQUEST_CONTRACT,
        SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_TRANSPORT,
        SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_HTTP,
    }
)
SM121_CHUNKED_PREFILL_8K_ADMISSION_RECEIPT_SCHEMA_VERSION = 1
SM121_CHUNKED_PREFILL_8K_ADMISSION_RECEIPT_ID = (
    "qwen38-flash-next-sm121-chunked-prefill-v3-receipt-v1"
)
_ADMISSION_RECEIPT_TARGET_FIELDS = frozenset(
    {
        "performance_campaign_id",
        "performance_suite_id",
        "performance_execution_mode",
        "arm_order",
        "control_profile_id",
        "candidate_profile_id",
        "control_chunk_size",
        "candidate_chunk_size",
        "performance_timed_case_id",
        "performance_workload_contract_sha256",
        "admission_id",
        "admission_suite_id",
        "admission_execution_mode",
        "admission_timed_case_id",
    }
)
_ADMISSION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_id",
        "target",
        "target_contract_sha256",
        "admission_plan_integrity_hash",
        "admission_summary_integrity_hash",
        "admission_model_contract_sha256",
        "admission_local_image_contract_sha256",
        "admission_audit_sha256",
        "receipt_integrity_hash",
    }
)


class SM121ChunkedPrefill8KAdmissionError(ValueError):
    """Raised when the singleton 8K admission contract changes."""


def _value(item: Any, field: str) -> object:
    return item.get(field) if isinstance(item, Mapping) else getattr(item, field, None)


def _profile_id(value: Any) -> str | None:
    candidate = value if isinstance(value, str) else _value(value, "id")
    return candidate if isinstance(candidate, str) else None


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise SM121ChunkedPrefill8KAdmissionError(f"{name} must be boolean")
    return bool(value)


def _require_int(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < 0 or (positive and value == 0):
        raise SM121ChunkedPrefill8KAdmissionError(
            f"{name} must be a non-negative integer"
        )
    return value


def _require_optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, name)


def _event_fields(
    value: object, expected: frozenset[str], name: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise SM121ChunkedPrefill8KAdmissionError(f"{name} fields are invalid")
    actual = set(value)
    if actual != set(expected) and actual != set(expected) | {"timestamp"}:
        raise SM121ChunkedPrefill8KAdmissionError(f"{name} fields are invalid")
    if "timestamp" in value and not isinstance(value["timestamp"], str):
        raise SM121ChunkedPrefill8KAdmissionError(f"{name} timestamp is invalid")
    return value


def _lifetime_phase(row: Mapping[str, object], *, name: str) -> None:
    ordinal = _require_int(row["fresh_lifetime"], f"{name} lifetime", positive=True)
    if (
        ordinal not in SM121_CHUNKED_PREFILL_8K_ADMISSION_LIFETIME_PHASES
        or row["phase"]
        != SM121_CHUNKED_PREFILL_8K_ADMISSION_LIFETIME_PHASES[ordinal]
    ):
        raise SM121ChunkedPrefill8KAdmissionError(f"{name} lifetime changed")


def is_sm121_chunked_prefill_8k_admission_profile(model: Any) -> bool:
    """Return whether a model is the one exact profile admitted by this lane."""

    return _profile_id(model) == SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CANDIDATE_PROFILE_ID


def validate_sm121_chunked_prefill_8k_admission_profile(model: Any) -> None:
    """Require the prospective V3 8K candidate without a broad selector."""

    if not is_sm121_chunked_prefill_8k_admission_profile(model):
        raise SM121ChunkedPrefill8KAdmissionError("8K admission profile is invalid")
    try:
        validate_sm121_chunked_prefill_performance_candidate(model)
        if sm121_chunked_prefill_performance_study(model) != SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY:
            raise SM121ChunkedPrefillPerformanceError("8K study changed")
    except SM121ChunkedPrefillPerformanceError as error:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission profile changed") from error


def validate_sm121_chunked_prefill_8k_admission_suite(suite: Any) -> None:
    """Require the quality-plus-cold-T0, non-measurement admission suite."""

    if _value(suite, "id") != SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE_ID:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission suite ID is invalid")
    if _value(suite, "description") != SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE_DESCRIPTION:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission suite description changed")
    if _value(suite, "protocol_digest") is not None:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission suite digest is invalid")
    cases = _value(suite, "cases")
    if not isinstance(cases, (list, tuple)) or len(cases) != 2:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission suite cases are invalid")
    expected_cases = (
        {
            "id": SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID,
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
            "id": SM121_CHUNKED_PREFILL_8K_ADMISSION_TIMED_CASE_ID,
            "kind": "capability",
            "requires": ("chat",),
            "warmups": 0,
            "repetitions": 1,
            "max_output_tokens": 32,
            "temperature": 0.0,
            "concurrency": 1,
            "prompt_repetitions": 0,
            "max_turns": 1,
        },
    )
    for case, expected in zip(cases, expected_cases, strict=True):
        for field, wanted in expected.items():
            actual = _value(case, field)
            if field == "requires" and isinstance(actual, (list, tuple)):
                actual = tuple(actual)
            if actual != wanted:
                raise SM121ChunkedPrefill8KAdmissionError(
                    f"8K admission suite field {field} changed"
                )


_STATIC_EVENT_FIELDS = frozenset(
    {
        "event",
        "fresh_lifetime",
        "phase",
        "candidate_source_tree",
        "chunked_prefill_size",
        *SM121_CACHE_SOURCE_DIGESTS,
        *SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
    }
)
_RUNTIME_EVENT_FIELDS = frozenset(
    {
        "event",
        "fresh_lifetime",
        "phase",
        "mamba_radix_cache_strategy",
        "max_mamba_cache_size",
        "chunked_prefill_size",
        *SM121_CHUNKED_PREFILL_8K_ADMISSION_RUNTIME_EXPECTED,
    }
)
_T0_EVENT_FIELDS = frozenset(
    {
        "event",
        "fresh_lifetime",
        "case_id",
        "protocol_case_id",
        "cache_details_requested",
        "prompt_token_ids_requested",
        "prompt_token_ids_verified",
        "streaming",
        "thinking_disabled",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "response_detail_state",
        "usage_detail_state",
        "response_device_cached_tokens",
        "response_host_cached_tokens",
        "response_storage_cached_tokens",
        "usage_cached_tokens",
        "metrics_available",
        "guardrail_metrics_available",
        "metrics_before_polls",
        "metrics_after_polls",
        "metrics_before_settled",
        "metrics_after_settled",
        "cold_t0_admitted",
        "cold_t0_basis",
        *(
            f"{prefix}_{metric}"
            for prefix in ("before", "after", "delta")
            for metric in SM121_CHUNKED_PREFILL_PERFORMANCE_METRIC_FIELDS
        ),
        *(
            f"{prefix}_cached_{source}_series_present"
            for prefix in ("before", "after")
            for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES
        ),
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "admission_id",
        "execution_mode",
        "status",
        "decision",
        "terminal_stage",
        "failure_code",
        "profile_id",
        "suite_id",
        "quality_admitted",
        "cold_t0_admitted",
        "quality_within_timeout",
        "cold_t0_within_timeout",
        "static_attestations",
        "runtime_attestations",
        "integrity_hash",
    }
)


def validate_sm121_chunked_prefill_8k_admission_static_event(event: object) -> None:
    row = _event_fields(event, _STATIC_EVENT_FIELDS, "8K admission static event")
    if row["event"] != SM121_CHUNKED_PREFILL_8K_ADMISSION_STATIC_EVENT:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission static event changed")
    _lifetime_phase(row, name="8K admission static")
    if row["candidate_source_tree"] != SM121_STORAGE_SOURCE_TREE:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission source tree changed")
    if row["chunked_prefill_size"] != SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CANDIDATE_CHUNK_SIZE:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission chunk size changed")
    for field, wanted in SM121_CACHE_SOURCE_DIGESTS.items():
        if row[field] != wanted:
            raise SM121ChunkedPrefill8KAdmissionError(
                f"8K admission {field} changed"
            )
    for field, wanted in SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS.items():
        if row[field] is not wanted:
            raise SM121ChunkedPrefill8KAdmissionError(
                f"8K admission {field} changed"
            )


def validate_sm121_chunked_prefill_8k_admission_runtime_event(event: object) -> None:
    row = _event_fields(event, _RUNTIME_EVENT_FIELDS, "8K admission runtime event")
    if row["event"] != SM121_CHUNKED_PREFILL_8K_ADMISSION_RUNTIME_EVENT:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission runtime event changed")
    _lifetime_phase(row, name="8K admission runtime")
    if row["mamba_radix_cache_strategy"] != "extra_buffer_lazy":
        raise SM121ChunkedPrefill8KAdmissionError("8K admission cache strategy changed")
    if row["max_mamba_cache_size"] != SM121_CHUNKED_PREFILL_PERFORMANCE_MAX_MAMBA_CACHE_SIZE:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission Mamba size changed")
    if row["chunked_prefill_size"] != SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CANDIDATE_CHUNK_SIZE:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission runtime size changed")
    for field, wanted in SM121_CHUNKED_PREFILL_8K_ADMISSION_RUNTIME_EXPECTED.items():
        if row[field] != wanted:
            raise SM121ChunkedPrefill8KAdmissionError(
                f"8K admission {field} changed"
            )


def _validate_detail_state(
    state: object,
    values: tuple[int | None, int | None, int | None],
    *,
    name: str,
) -> None:
    if state not in {
        "omitted",
        "null",
        "zero_details",
        "nonzero_details",
        "unexpected",
    }:
        raise SM121ChunkedPrefill8KAdmissionError(f"8K admission {name} state is invalid")
    if state in {"omitted", "null", "unexpected"}:
        if any(value is not None for value in values):
            raise SM121ChunkedPrefill8KAdmissionError(
                f"8K admission {name} omitted details are invalid"
            )
    elif state == "zero_details":
        if values != (0, 0, 0):
            raise SM121ChunkedPrefill8KAdmissionError(
                f"8K admission {name} zero details are invalid"
            )
    elif (
        any(value is None for value in values)
        or values[0] is None
        or values[0] <= 0
        or values[1:] != (0, 0)
    ):
        raise SM121ChunkedPrefill8KAdmissionError(
            f"8K admission {name} nonzero details are invalid"
        )


def _cold_t0_issues(row: Mapping[str, object]) -> tuple[str, ...]:
    issues: list[str] = []
    if row["response_detail_state"] not in {"omitted", "null", "zero_details"}:
        issues.append("response_detail_state")
    if row["usage_detail_state"] not in {"omitted", "null", "zero_details"}:
        issues.append("usage_detail_state")
    if row["metrics_available"] is not True or row["guardrail_metrics_available"] is not True:
        issues.append("metrics_unavailable")
    if (
        row["metrics_before_settled"] is not True
        or row["metrics_after_settled"] is not True
        or int(row["metrics_before_polls"]) < 2
        or int(row["metrics_after_polls"]) < 2
    ):
        issues.append("metrics_unsettled")
    if row["prompt_token_ids_verified"] is not True:
        issues.append("prompt_token_ids")
    if int(row["delta_prefill_input_tokens"]) <= 0:
        issues.append("input_counter")
    for metric in (
        "prefill_host_hit_tokens",
        "prefill_storage_hit_tokens",
        "cached_host_tokens",
        "cached_storage_tokens",
        "evicted_tokens",
        "retracted_requests",
    ):
        if any(
            int(row[f"{prefix}_{metric}"]) != 0
            for prefix in ("before", "after", "delta")
        ):
            issues.append("cache_guardrail")
            break
    if any(
        int(row[f"before_{metric}"]) != 0
        for metric in (
            "prefill_device_hit_tokens",
            "prefill_host_hit_tokens",
            "prefill_storage_hit_tokens",
            "cached_device_tokens",
            "cached_host_tokens",
            "cached_storage_tokens",
            "cached_total_tokens",
        )
    ):
        issues.append("cold_lifetime")
    response_values = (
        row["response_device_cached_tokens"],
        row["response_host_cached_tokens"],
        row["response_storage_cached_tokens"],
        row["usage_cached_tokens"],
    )
    if (
        any(value not in {None, 0} for value in response_values)
        or any(
            int(row[f"{prefix}_{metric}"]) != 0
            for prefix in ("after", "delta")
            for metric in (
                "prefill_device_hit_tokens",
                "cached_device_tokens",
                "cached_total_tokens",
            )
        )
    ):
        issues.append("cold_hit")
    return tuple(issues)


def derive_sm121_chunked_prefill_8k_admission_t0(
    event: object,
) -> tuple[bool, str]:
    """Reduce a well-formed scalar T0 observation to one admission decision."""

    row = _event_fields(event, _T0_EVENT_FIELDS, "8K admission T0 event")
    issues = _cold_t0_issues(row)
    return not issues, "admitted" if not issues else issues[0]


def validate_sm121_chunked_prefill_8k_admission_t0_event(event: object) -> None:
    row = _event_fields(event, _T0_EVENT_FIELDS, "8K admission T0 event")
    if row["event"] != SM121_CHUNKED_PREFILL_8K_ADMISSION_T0_EVENT:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission T0 event changed")
    if row["fresh_lifetime"] != 2:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission T0 lifetime changed")
    if row["protocol_case_id"] != SM121_CHUNKED_PREFILL_8K_ADMISSION_TIMED_CASE_ID:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission T0 case changed")
    case_id = row["case_id"]
    if not isinstance(case_id, str) or re.fullmatch(
        rf"{re.escape(SM121_CHUNKED_PREFILL_8K_ADMISSION_TIMED_CASE_ID)}--[0-9a-f]{{12}}",
        case_id,
    ) is None:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission T0 identifier is invalid")
    for field, wanted in (
        ("cache_details_requested", True),
        ("prompt_token_ids_requested", True),
        ("prompt_token_ids_verified", True),
        ("streaming", False),
        ("thinking_disabled", True),
    ):
        if row[field] is not wanted:
            raise SM121ChunkedPrefill8KAdmissionError(
                f"8K admission {field} changed"
            )
    prompt_tokens = _require_int(row["prompt_tokens"], "8K admission prompt", positive=True)
    if not (
        SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MIN_TOKENS
        <= prompt_tokens
        <= SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MAX_TOKENS
    ):
        raise SM121ChunkedPrefill8KAdmissionError("8K admission T0 shape is invalid")
    _require_int(row["completion_tokens"], "8K admission completion", positive=True)
    if _require_int(row["reasoning_tokens"], "8K admission reasoning") != 0:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission reasoning is enabled")
    response_details = tuple(
        _require_optional_int(row[field], field)
        for field in (
            "response_device_cached_tokens",
            "response_host_cached_tokens",
            "response_storage_cached_tokens",
        )
    )
    _validate_detail_state(
        row["response_detail_state"], response_details, name="response"
    )
    usage = _require_optional_int(row["usage_cached_tokens"], "8K admission usage")
    usage_state = row["usage_detail_state"]
    if usage_state not in {
        "omitted",
        "null",
        "zero_details",
        "nonzero_details",
        "unexpected",
    }:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission usage state is invalid")
    if usage_state in {"omitted", "null", "unexpected"} and usage is not None:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission usage details are invalid")
    if usage_state == "zero_details" and usage != 0:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission zero usage is invalid")
    if usage_state == "nonzero_details" and (usage is None or usage <= 0):
        raise SM121ChunkedPrefill8KAdmissionError("8K admission nonzero usage is invalid")
    for field in (
        "metrics_available",
        "guardrail_metrics_available",
        "metrics_before_settled",
        "metrics_after_settled",
    ):
        _require_bool(row[field], f"8K admission {field}")
    _require_int(row["metrics_before_polls"], "8K admission metrics-before polls")
    _require_int(row["metrics_after_polls"], "8K admission metrics-after polls")
    for prefix in ("before", "after", "delta"):
        for metric in SM121_CHUNKED_PREFILL_PERFORMANCE_METRIC_FIELDS:
            value = row[f"{prefix}_{metric}"]
            if prefix == "delta" and metric.endswith(
                ("available_tokens", "evictable_tokens", "used_tokens")
            ):
                if type(value) is not int:
                    raise SM121ChunkedPrefill8KAdmissionError(
                        "8K admission gauge delta is invalid"
                    )
            else:
                _require_int(value, f"8K admission {prefix} {metric}")
            if row[f"delta_{metric}"] != row[f"after_{metric}"] - row[f"before_{metric}"]:
                raise SM121ChunkedPrefill8KAdmissionError(
                    "8K admission metric delta changed"
                )
    for prefix in ("before", "after"):
        for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
            _require_bool(
                row[f"{prefix}_cached_{source}_series_present"],
                "8K admission cache series marker",
            )
    admitted, basis = derive_sm121_chunked_prefill_8k_admission_t0(row)
    if row["cold_t0_admitted"] is not admitted or row["cold_t0_basis"] != basis:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission T0 decision changed")


def validate_sm121_chunked_prefill_8k_admission_summary(summary: object) -> None:
    """Require a scalar-only terminal admission record."""

    if type(summary) is not dict or set(summary) != _SUMMARY_FIELDS:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission summary fields are invalid")
    integrity = summary["integrity_hash"]
    if not isinstance(integrity, str) or content_hash(
        {key: value for key, value in summary.items() if key != "integrity_hash"},
        len(integrity),
    ) != integrity:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission summary integrity is invalid")
    if (
        summary["schema_version"] != 1
        or summary["admission_id"] != SM121_CHUNKED_PREFILL_8K_ADMISSION_ID
        or summary["execution_mode"]
        != SM121_CHUNKED_PREFILL_8K_ADMISSION_EXECUTION_MODE
        or summary["profile_id"]
        != SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CANDIDATE_PROFILE_ID
        or summary["suite_id"] != SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE_ID
    ):
        raise SM121ChunkedPrefill8KAdmissionError("8K admission summary changed")
    for field in (
        "quality_admitted",
        "cold_t0_admitted",
        "quality_within_timeout",
        "cold_t0_within_timeout",
    ):
        _require_bool(summary[field], f"8K admission {field}")
    for field in ("static_attestations", "runtime_attestations"):
        value = _require_int(summary[field], f"8K admission {field}")
        if value > 2:
            raise SM121ChunkedPrefill8KAdmissionError(
                f"8K admission {field} is invalid"
            )
    complete = (
        summary["quality_admitted"] is True
        and summary["cold_t0_admitted"] is True
        and summary["quality_within_timeout"] is True
        and summary["cold_t0_within_timeout"] is True
        and summary["static_attestations"] == 2
        and summary["runtime_attestations"] == 2
    )
    if complete:
        if (
            summary["status"] != "complete"
            or summary["decision"] != "admitted"
            or summary["terminal_stage"] != "complete"
            or summary["failure_code"] is not None
        ):
            raise SM121ChunkedPrefill8KAdmissionError("8K admission summary changed")
    elif (
        summary["status"] != "partial"
        or summary["decision"] != "blocked"
        or summary["terminal_stage"]
        not in {"preflight", "quality_lifetime", "cold_t0_lifetime"}
        or not isinstance(summary["failure_code"], str)
        or summary["failure_code"]
        not in SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODES
    ):
        raise SM121ChunkedPrefill8KAdmissionError("8K admission summary changed")


def _receipt_hash(value: object, *, domain: str) -> str:
    """Return a full domain-separated digest for scalar receipt material."""

    return content_hash({"domain": domain, "value": value}, 64)


def _require_receipt_hash(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SM121ChunkedPrefill8KAdmissionError(f"8K admission {name} is invalid")
    return value


def _receipt_target() -> dict[str, object]:
    """Return the immutable V3 and timing-free admission target contract."""

    workload = {
        "cold_input_min_tokens": SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MIN_TOKENS,
        "cold_input_max_tokens": SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MAX_TOKENS,
        "max_output_tokens": 32,
        "temperature": 0.0,
        "concurrency": 1,
        "repetitions": 1,
        "warmups": 0,
        "prompt_repetitions": 0,
        "max_turns": 1,
        "streaming": False,
        "thinking_disabled": True,
        "quality_item_count": SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT,
        "fresh_lifetime_count": 2,
    }
    return {
        "performance_campaign_id": SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CAMPAIGN_ID,
        "performance_suite_id": SM121_CHUNKED_PREFILL_PERFORMANCE_V3_SUITE_ID,
        "performance_execution_mode": SM121_CHUNKED_PREFILL_PERFORMANCE_V3_EXECUTION_MODE,
        "arm_order": ["A", "B", "B", "A"],
        "control_profile_id": SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CONTROL_PROFILE_ID,
        "candidate_profile_id": SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CANDIDATE_PROFILE_ID,
        "control_chunk_size": SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CONTROL_CHUNK_SIZE,
        "candidate_chunk_size": SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CANDIDATE_CHUNK_SIZE,
        "performance_timed_case_id": SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CASE_ID,
        "performance_workload_contract_sha256": _receipt_hash(
            workload,
            domain="sm121-chunked-prefill-v3-performance-workload-v1",
        ),
        "admission_id": SM121_CHUNKED_PREFILL_8K_ADMISSION_ID,
        "admission_suite_id": SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE_ID,
        "admission_execution_mode": SM121_CHUNKED_PREFILL_8K_ADMISSION_EXECUTION_MODE,
        "admission_timed_case_id": SM121_CHUNKED_PREFILL_8K_ADMISSION_TIMED_CASE_ID,
    }


def sm121_chunked_prefill_8k_admission_receipt(
    summary: object,
    *,
    admission_plan_integrity_hash: object,
    admission_model_contract_sha256: object,
    admission_local_image_contract_sha256: object,
    admission_audit_sha256: object,
) -> dict[str, object]:
    """Project one audited complete admission to a path-free V3 receipt.

    All dynamic source material is reduced to full domain-separated hashes.
    The receipt has no local path, timestamps, request identifiers, prompt
    data, response data, token IDs, log content, or timing observation.
    """

    validate_sm121_chunked_prefill_8k_admission_summary(summary)
    assert isinstance(summary, dict)
    if (
        summary["status"] != "complete"
        or summary["decision"] != "admitted"
        or summary["terminal_stage"] != "complete"
        or summary["failure_code"] is not None
        or summary["quality_admitted"] is not True
        or summary["cold_t0_admitted"] is not True
        or summary["quality_within_timeout"] is not True
        or summary["cold_t0_within_timeout"] is not True
        or summary["static_attestations"] != 2
        or summary["runtime_attestations"] != 2
    ):
        raise SM121ChunkedPrefill8KAdmissionError("8K admission receipt is blocked")
    target = _receipt_target()
    receipt: dict[str, object] = {
        "schema_version": SM121_CHUNKED_PREFILL_8K_ADMISSION_RECEIPT_SCHEMA_VERSION,
        "receipt_id": SM121_CHUNKED_PREFILL_8K_ADMISSION_RECEIPT_ID,
        "target": target,
        "target_contract_sha256": _receipt_hash(
            target, domain="sm121-chunked-prefill-v3-target-v1"
        ),
        "admission_plan_integrity_hash": _require_receipt_hash(
            admission_plan_integrity_hash, "plan integrity"
        ),
        "admission_summary_integrity_hash": _require_receipt_hash(
            summary["integrity_hash"], "summary integrity"
        ),
        "admission_model_contract_sha256": _require_receipt_hash(
            admission_model_contract_sha256, "model contract"
        ),
        "admission_local_image_contract_sha256": _require_receipt_hash(
            admission_local_image_contract_sha256, "local image contract"
        ),
        "admission_audit_sha256": _require_receipt_hash(
            admission_audit_sha256, "audit proof"
        ),
    }
    receipt["receipt_integrity_hash"] = _receipt_hash(
        receipt, domain="sm121-chunked-prefill-v3-receipt-v1"
    )
    validate_sm121_chunked_prefill_8k_admission_receipt(receipt)
    return receipt


def validate_sm121_chunked_prefill_8k_admission_receipt(receipt: object) -> None:
    """Require the immutable scalar receipt that admits only V3 execution."""

    if type(receipt) is not dict or set(receipt) != _ADMISSION_RECEIPT_FIELDS:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission receipt fields are invalid")
    target = receipt["target"]
    if type(target) is not dict or set(target) != _ADMISSION_RECEIPT_TARGET_FIELDS:
        raise SM121ChunkedPrefill8KAdmissionError("8K admission receipt target is invalid")
    if target != _receipt_target() or receipt["target_contract_sha256"] != _receipt_hash(
        target, domain="sm121-chunked-prefill-v3-target-v1"
    ):
        raise SM121ChunkedPrefill8KAdmissionError("8K admission receipt target changed")
    expected = {
        "schema_version": SM121_CHUNKED_PREFILL_8K_ADMISSION_RECEIPT_SCHEMA_VERSION,
        "receipt_id": SM121_CHUNKED_PREFILL_8K_ADMISSION_RECEIPT_ID,
    }
    if any(receipt[field] != value for field, value in expected.items()):
        raise SM121ChunkedPrefill8KAdmissionError("8K admission receipt changed")
    for field in (
        "target_contract_sha256",
        "admission_plan_integrity_hash",
        "admission_summary_integrity_hash",
        "admission_model_contract_sha256",
        "admission_local_image_contract_sha256",
        "admission_audit_sha256",
        "receipt_integrity_hash",
    ):
        _require_receipt_hash(receipt[field], field)
    if receipt["receipt_integrity_hash"] != _receipt_hash(
        {key: value for key, value in receipt.items() if key != "receipt_integrity_hash"},
        domain="sm121-chunked-prefill-v3-receipt-v1",
    ):
        raise SM121ChunkedPrefill8KAdmissionError("8K admission receipt integrity is invalid")


def validate_sm121_chunked_prefill_8k_admission_receipt_for_v3_candidate_plan(
    receipt: object, candidate_plan: object
) -> None:
    """Bind the admission's exact 8K model and image to a frozen V3 B plan."""

    validate_sm121_chunked_prefill_8k_admission_receipt(receipt)
    if type(candidate_plan) is not dict:
        raise SM121ChunkedPrefill8KAdmissionError("8K receipt candidate plan is invalid")
    model = candidate_plan.get("model")
    resolved = candidate_plan.get("resolved")
    suite = candidate_plan.get("suite")
    if type(model) is not dict or type(resolved) is not dict or type(suite) is not dict:
        raise SM121ChunkedPrefill8KAdmissionError("8K receipt candidate plan is invalid")
    local_image = resolved.get("local_image")
    if type(local_image) is not dict:
        raise SM121ChunkedPrefill8KAdmissionError("8K receipt local image is invalid")
    try:
        validate_sm121_chunked_prefill_performance_candidate(model)
        if sm121_chunked_prefill_performance_study(model) != SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY:
            raise SM121ChunkedPrefillPerformanceError("V3 candidate changed")
    except SM121ChunkedPrefillPerformanceError as error:
        raise SM121ChunkedPrefill8KAdmissionError("8K receipt candidate changed") from error
    target = receipt["target"]
    assert isinstance(target, dict)
    if (
        model.get("id") != target["candidate_profile_id"]
        or suite.get("id") != target["performance_suite_id"]
    ):
        raise SM121ChunkedPrefill8KAdmissionError("8K receipt candidate changed")
    if _receipt_hash(model, domain="sm121-chunked-prefill-v3-candidate-model-v1") != receipt[
        "admission_model_contract_sha256"
    ]:
        raise SM121ChunkedPrefill8KAdmissionError("8K receipt model binding changed")
    if _receipt_hash(
        local_image, domain="sm121-chunked-prefill-v3-local-image-v1"
    ) != receipt["admission_local_image_contract_sha256"]:
        raise SM121ChunkedPrefill8KAdmissionError("8K receipt image binding changed")
