"""Exact manifest contract for the paired SM121 cache-policy semantic canary.

This is an admission-only B-then-A probe.  It freezes two sibling profiles
against the same immutable native-storage artifact while changing only the
Radix-cache policy flag.  The suite identifies a dedicated semantic workload;
it deliberately does not describe timing, throughput, or generic execution.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .sglang_sm121_cache_observability import (
    SM121_CACHE_OBSERVABILITY_CACHED_SERIES,
    SM121_CACHE_OBSERVABILITY_METRIC_FIELDS,
    SM121_CACHE_SOURCE_DIGESTS,
)

from .sglang_sm121_storage import (
    SM121_STORAGE_BUILD_CONTRACT_SHA256,
    SM121_STORAGE_CACHE_PAGES,
    SM121_STORAGE_CONTEXT_LENGTH,
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_LOCAL_IMAGE_TAG,
    SM121_STORAGE_MAX_BATCH_PAGES,
    SM121_STORAGE_MODE,
    SM121_STORAGE_NATIVE_CONTEXT,
    SM121_STORAGE_QUEUE_DEPTH,
    SM121_STORAGE_REVISION,
    SM121_STORAGE_SOURCE,
    SM121_STORAGE_SOURCE_TREE,
    SM121_STORAGE_WEIGHT_FILE_COUNT,
    SM121_STORAGE_WEIGHT_SIZE_BYTES,
)


SM121_CACHE_SEMANTIC_SUITE_ID = (
    "qwen38-flash-next-sm121-triton-storage-cache-policy-semantic-canary"
)
SM121_CACHE_SEMANTIC_EXECUTION_MODE = (
    "sm121_storage_cache_policy_semantic_b_then_a_fresh_lifetimes"
)
SM121_CACHE_SEMANTIC_PAIR_BINDING_SCHEMA_VERSION = 2
SM121_CACHE_SEMANTIC_STATIC_ATTESTATION_EVENT = (
    "sm121_cache_semantic_static_attestation"
)
SM121_CACHE_SEMANTIC_RUNTIME_ATTESTATION_EVENT = (
    "sm121_cache_semantic_runtime_attestation"
)
SM121_CACHE_SEMANTIC_TURN_OBSERVATION_EVENT = (
    "sm121_cache_semantic_turn_observation"
)
SM121_CACHE_SEMANTIC_CACHE_OFF_ARM = "B"
SM121_CACHE_SEMANTIC_CACHE_ON_ARM = "A"
SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID = (
    "qwen38-flash-next-nvfp4-sm121-triton-storage-cache-policy-off-sglang"
)
SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID = (
    "qwen38-flash-next-nvfp4-sm121-triton-storage-cache-policy-on-sglang"
)
SM121_CACHE_SEMANTIC_PROFILE_IDS = frozenset(
    {
        SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID,
        SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID,
    }
)
SM121_CACHE_SEMANTIC_PROFILE_ORDER = (
    SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID,
    SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID,
)
SM121_CACHE_SEMANTIC_ARM_ORDER = (
    SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
    SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
)
SM121_CACHE_SEMANTIC_PROFILE_ID_BY_ARM = {
    SM121_CACHE_SEMANTIC_CACHE_OFF_ARM: SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID,
    SM121_CACHE_SEMANTIC_CACHE_ON_ARM: SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID,
}
SM121_CACHE_SEMANTIC_LIFETIME_PHASES = ("quality", "semantic")
SM121_CACHE_SEMANTIC_LIFETIME_ORDER = (
    (SM121_CACHE_SEMANTIC_CACHE_OFF_ARM, "quality"),
    (SM121_CACHE_SEMANTIC_CACHE_OFF_ARM, "semantic"),
    (SM121_CACHE_SEMANTIC_CACHE_ON_ARM, "quality"),
    (SM121_CACHE_SEMANTIC_CACHE_ON_ARM, "semantic"),
)
SM121_CACHE_SEMANTIC_LOCAL_LIFETIME_ORDER = SM121_CACHE_SEMANTIC_LIFETIME_PHASES
SM121_CACHE_SEMANTIC_LOCAL_LIFETIME_ORDINALS = (1, 2)
SM121_CACHE_SEMANTIC_LIFETIME_COUNT = len(SM121_CACHE_SEMANTIC_LIFETIME_ORDER)
SM121_CACHE_SEMANTIC_TURN_ORDER = ("T0", "T1", "T2")
SM121_CACHE_SEMANTIC_QUALITY_CASE_ID = "synthetic-exact-answer-v2"
SM121_CACHE_SEMANTIC_CASE_ID = "sm121-cache-policy-shared-prefix-semantic-v1"
SM121_CACHE_SEMANTIC_CASE_MAX_OUTPUT_TOKENS = 32
SM121_CACHE_SEMANTIC_COLD_INPUT_MIN_TOKENS = 32 * 1024
SM121_CACHE_SEMANTIC_COLD_INPUT_MAX_TOKENS = 48 * 1024
SM121_CACHE_SEMANTIC_MAX_MAMBA_CACHE_SIZE = 4
SM121_CACHE_SEMANTIC_SERVED_NAME = (
    "qwen38-flash-next-nvfp4-sm121-storage-cache-policy-semantic"
)
SM121_CACHE_SEMANTIC_SUITE_DESCRIPTION = (
    "Admission-only paired cache-policy semantic canary for the pre-admission "
    "Qwen3.8 Flash-Next SM121 native-NVMe runtime. It runs cache-off B before "
    "cache-on A in separate fresh lifetimes, with an exact-answer quality gate "
    "and one dedicated append-only shared-prefix semantic probe. No timing or "
    "throughput claim is part of this suite."
)
SM121_CACHE_SEMANTIC_CACHE_OFF_DESCRIPTION = (
    "Paired SM121 cache-policy semantic canary B: cache-off control. Only the "
    "dedicated B-then-A semantic canary may execute this profile."
)
SM121_CACHE_SEMANTIC_CACHE_ON_DESCRIPTION = (
    "Paired SM121 cache-policy semantic canary A: cache-on candidate. Only the "
    "dedicated B-then-A semantic canary may execute this profile."
)

# The profiles intentionally share one served name: they cannot be co-resident,
# and doing so makes the cache-off argument the sole serving-configuration
# difference between their pinned command lines.
SM121_CACHE_SEMANTIC_COMMON_ARGS = (
    "--served-model-name",
    SM121_CACHE_SEMANTIC_SERVED_NAME,
    "--tp-size",
    "1",
    "--attention-backend",
    "triton",
    "--moe-runner-backend",
    "flashinfer_cutlass",
    "--quantization",
    "modelopt_fp4",
    "--load-format",
    "auto",
    "--no-ple-offload-embedding",
    "--weight-loader-drop-cache-after-load",
    "--language-only",
    "--mamba-radix-cache-strategy",
    "extra_buffer_lazy",
    "--max-mamba-cache-size",
    str(SM121_CACHE_SEMANTIC_MAX_MAMBA_CACHE_SIZE),
    "--page-size",
    "64",
    "--mem-fraction-static",
    "0.85",
    "--max-total-tokens",
    str(SM121_STORAGE_CONTEXT_LENGTH),
    "--context-length",
    str(SM121_STORAGE_CONTEXT_LENGTH),
    "--chunked-prefill-size",
    "1024",
    "--max-running-requests",
    "1",
    "--cuda-graph-backend-decode",
    "disabled",
    "--cuda-graph-backend-prefill",
    "disabled",
    "--enable-metrics",
    "--host",
    "0.0.0.0",
    "--port",
    "30000",
)
_CACHE_OFF_INSERTION_INDEX = SM121_CACHE_SEMANTIC_COMMON_ARGS.index("--page-size")
SM121_CACHE_SEMANTIC_CACHE_OFF_ARGS = (
    SM121_CACHE_SEMANTIC_COMMON_ARGS[:_CACHE_OFF_INSERTION_INDEX]
    + ("--disable-radix-cache",)
    + SM121_CACHE_SEMANTIC_COMMON_ARGS[_CACHE_OFF_INSERTION_INDEX:]
)
SM121_CACHE_SEMANTIC_CACHE_ON_ARGS = SM121_CACHE_SEMANTIC_COMMON_ARGS

SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED = {
    SM121_CACHE_SEMANTIC_CACHE_OFF_ARM: {
        "cache_impl": "ChunkCache",
        "cache_source": "default",
        "disable_radix_cache": True,
        "hicache_attached": False,
        "hybrid_ssm": True,
        "hybrid_swa": False,
        "mamba_extra_buffer_enabled": False,
        "mamba_extra_buffer_lazy_enabled": False,
        "streaming_wrapped": False,
    },
    SM121_CACHE_SEMANTIC_CACHE_ON_ARM: {
        "cache_impl": "UnifiedRadixCache",
        "cache_source": "default",
        "disable_radix_cache": False,
        "hicache_attached": False,
        "hybrid_ssm": True,
        "hybrid_swa": False,
        "mamba_extra_buffer_enabled": True,
        "mamba_extra_buffer_lazy_enabled": True,
        "streaming_wrapped": False,
    },
}
SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED_BY_PROFILE = {
    SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID: SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED[
        SM121_CACHE_SEMANTIC_CACHE_OFF_ARM
    ],
    SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID: SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED[
        SM121_CACHE_SEMANTIC_CACHE_ON_ARM
    ],
}
SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS = {
    "cache_on_selects_unified_radix_cache": True,
    "cache_off_selects_chunk_cache": True,
    "cache_off_suppresses_mamba_extra_buffer": True,
    "cache_off_suppresses_mamba_extra_buffer_lazy": True,
    "cache_on_enables_mamba_extra_buffer": True,
    "cache_on_enables_mamba_extra_buffer_lazy": True,
    "cache_startup_log_exposes_impl": True,
    "prompt_token_ids_available": True,
    "cache_detail_extension_available": True,
    "native_cache_counters_exposed": True,
    "native_residency_gauges_exposed": True,
    "native_guardrail_counters_exposed": True,
}
SM121_CACHE_SEMANTIC_GUARDRAIL_METRIC_FIELDS = (
    "evicted_tokens",
    "retracted_requests",
)
SM121_CACHE_SEMANTIC_METRIC_FIELDS = (
    *SM121_CACHE_OBSERVABILITY_METRIC_FIELDS,
    *SM121_CACHE_SEMANTIC_GUARDRAIL_METRIC_FIELDS,
)
SM121_CACHE_SEMANTIC_TURN_EVENT_FIELDS = frozenset(
    {
        "event",
        "case_id",
        "protocol_case_id",
        "attempt_id",
        "turn",
        "arm",
        "cache_details_requested",
        "prompt_token_ids_requested",
        "streaming",
        "thinking_disabled",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "append_only_prompt_identity_verified",
        "cross_arm_prompt_identity_verified",
        "shared_prefix_tokens",
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
        "semantic_turn_admitted",
        "semantic_turn_basis",
        *(
            f"{prefix}_{metric}"
            for prefix in ("before", "after", "delta")
            for metric in SM121_CACHE_SEMANTIC_METRIC_FIELDS
        ),
        *(
            f"{prefix}_cached_{source}_series_present"
            for prefix in ("before", "after")
            for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES
        ),
    }
)
SM121_CACHE_SEMANTIC_PAIR_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "suite_id",
        "execution_mode",
        "arm",
        "profile_id",
        "arm_order",
        "local_lifetime_order",
        "quality_case_id",
        "semantic_case_id",
        "semantic_case_metadata",
        "peer_plan_fingerprint",
        "pair_instance_sha256",
        "pair_binding_sha256",
    }
)
_SEMANTIC_DETAIL_STATES = frozenset(
    {"omitted", "null", "zero_details", "nonzero_details", "unexpected"}
)
SM121_CACHE_SEMANTIC_TURN_BASES = frozenset(
    {
        "admitted",
        "zero_hit_not_reconciled",
        "positive_device_hit_not_reconciled",
        "metric_unavailable",
        "guardrail_activity",
        "identity_mismatch",
    }
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{16}\Z")

_COMMON_MODEL_FIELDS = {
    "backend": "sglang",
    "source": SM121_STORAGE_SOURCE,
    "revision": SM121_STORAGE_REVISION,
    "served_name": SM121_CACHE_SEMANTIC_SERVED_NAME,
    "tasks": ("chat",),
    "architecture": "moe+qsa+gdn",
    "quantization": "nvfp4+ple-fp8-nvme-io-uring",
    "lifecycle": "docker",
    "image": SM121_STORAGE_LOCAL_IMAGE_TAG,
    "image_digest": None,
    "local_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
    "cache_dir": "user",
    "max_context": SM121_STORAGE_CONTEXT_LENGTH,
    "native_context": SM121_STORAGE_NATIVE_CONTEXT,
    "startup_timeout_s": 1_800,
    "estimated_ram_gib": 101.0,
    "host_safety_min_memavailable_gib": 10,
    "host_safety_max_swap_growth_mib": 512,
    "host_safety_max_starting_swap_mib": 512,
    "endpoint": "http://127.0.0.1:30000/v1",
    "fetch_allow_patterns": (),
    "fetch_ignore_patterns": (),
    "weight_size_bytes": SM121_STORAGE_WEIGHT_SIZE_BYTES,
    "weight_file_count": SM121_STORAGE_WEIGHT_FILE_COUNT,
    "draft_source": None,
    "draft_revision": None,
    "draft_weight_size_bytes": None,
    "draft_model_file": None,
    "draft_model_digest": None,
    "draft_model_size_bytes": None,
    "sglang_allow_hf_metadata_probe": False,
    "sglang_source_overlays": (),
    "sglang_ple_mmap": False,
    "sglang_ple_omitted": False,
    "sglang_ple_cache_mode": None,
    "sglang_ple_cache_marker_digest": None,
    "sglang_ple_cache_payload_digest": None,
    "sglang_storage_mode": SM121_STORAGE_MODE,
    "sglang_ple_nvme_queue_depth": SM121_STORAGE_QUEUE_DEPTH,
    "sglang_ple_nvme_max_batch_pages": SM121_STORAGE_MAX_BATCH_PAGES,
    "sglang_ple_nvme_cache_pages": SM121_STORAGE_CACHE_PAGES,
    "recipe_source": None,
    "recipe_revision": None,
    "runtime_python": None,
    "runtime_binary": None,
    "runtime_digest": None,
    "runtime_parallel": None,
    "runtime_source_dir": None,
    "runtime_revision": None,
    "model_file": None,
    "model_digest": None,
    "model_size_bytes": None,
    "model_shards": (),
    "mmproj_file": None,
    "mmproj_digest": None,
    "mmproj_size_bytes": None,
    "prefix_cache_mode": None,
    "support_status": "exploratory",
}


class SM121CacheSemanticError(ValueError):
    """Raised when the paired semantic canary loses its immutable identity."""


def _value(item: Any, field: str) -> object:
    if isinstance(item, Mapping):
        return item.get(field)
    return getattr(item, field, None)


def _require(value: object, expected: object, field: str) -> None:
    if type(expected) is bool:
        matches = type(value) is bool and value is expected
    elif type(expected) is int:
        matches = type(value) is int and value == expected
    elif type(expected) is float:
        matches = (
            type(value) in {int, float}
            and type(value) is not bool
            and value == expected
        )
    else:
        matches = value == expected
    if not matches:
        raise SM121CacheSemanticError(
            f"{field} does not match the SM121 cache-policy semantic contract"
        )


def _require_nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise SM121CacheSemanticError(f"{field} must be a non-negative integer")
    return value


def _require_positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise SM121CacheSemanticError(f"{field} must be a positive integer")
    return value


def _require_optional_nonnegative_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _require_nonnegative_int(value, field)


def _require_event_fields(
    event: object, fields: frozenset[str], name: str
) -> dict[str, object]:
    if type(event) is not dict:
        raise SM121CacheSemanticError(f"{name} is not an object")
    actual = frozenset(event)
    if actual not in {fields, fields | {"timestamp"}}:
        raise SM121CacheSemanticError(f"{name} fields are invalid")
    if "timestamp" in event and not isinstance(event["timestamp"], str):
        raise SM121CacheSemanticError(f"{name} timestamp is invalid")
    return event


def _require_arm(value: object, field: str) -> str:
    if value not in SM121_CACHE_SEMANTIC_ARM_ORDER:
        raise SM121CacheSemanticError(f"{field} is not a semantic canary arm")
    return str(value)


def _require_case_id(value: object, protocol_case_id: str, field: str) -> str:
    prefix = protocol_case_id + "--"
    if not isinstance(value, str) or not value.startswith(prefix):
        raise SM121CacheSemanticError(f"{field} is invalid")
    suffix = value[len(prefix) :]
    if len(suffix) != 12 or any(
        character not in "0123456789abcdef" for character in suffix
    ):
        raise SM121CacheSemanticError(f"{field} is invalid")
    return value


def _profile_id(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    candidate = _value(value, "id")
    return candidate if isinstance(candidate, str) else None


def sm121_cache_semantic_arm(value: Any) -> str:
    """Return the fixed A/B arm for one exact semantic profile."""

    profile_id = _profile_id(value)
    if profile_id == SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID:
        return SM121_CACHE_SEMANTIC_CACHE_OFF_ARM
    if profile_id == SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID:
        return SM121_CACHE_SEMANTIC_CACHE_ON_ARM
    raise SM121CacheSemanticError("profile is not part of the SM121 semantic pair")


def sm121_cache_semantic_runtime_expected(value: Any) -> dict[str, object]:
    """Return a detached scalar startup-identity contract for one arm."""

    profile_id = _profile_id(value)
    expected = SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED_BY_PROFILE.get(profile_id)
    if expected is None:
        raise SM121CacheSemanticError("profile is not part of the SM121 semantic pair")
    return dict(expected)


def sm121_cache_semantic_case_metadata() -> dict[str, object]:
    """Return the fixed scalar metadata for the dedicated semantic probe.

    The specialized executor owns the deterministic renderer and private
    correctness checks.  This identity is intentionally sufficient to reject
    generic case execution while exposing no prompt or completion material.
    """

    return {
        "schema_version": 1,
        "case_id": SM121_CACHE_SEMANTIC_CASE_ID,
        "turn_order": list(SM121_CACHE_SEMANTIC_TURN_ORDER),
        "cold_input_min_tokens": SM121_CACHE_SEMANTIC_COLD_INPUT_MIN_TOKENS,
        "cold_input_max_tokens": SM121_CACHE_SEMANTIC_COLD_INPUT_MAX_TOKENS,
        "t0_common_prefix_tokens": 0,
        "later_turns_require_shared_prefix": True,
        "history": "append_only",
        "measurement": "semantic_only",
        "timing_claims": "forbidden",
    }


def is_sm121_cache_semantic_candidate(model: Any) -> bool:
    """Return whether a model ID belongs to the exact paired semantic lane."""

    return _profile_id(model) in SM121_CACHE_SEMANTIC_PROFILE_IDS


def is_sm121_cache_semantic_plan(model: Any, suite: Any) -> bool:
    """Return whether one model/suite selection belongs to the pair canary."""

    return (
        is_sm121_cache_semantic_candidate(model)
        and _value(suite, "id") == SM121_CACHE_SEMANTIC_SUITE_ID
    )


def _canonical_request_body(value: object) -> dict[str, object]:
    if not isinstance(value, str):
        raise SM121CacheSemanticError(
            "request_body_json must be a canonical no-thinking object"
        )
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise SM121CacheSemanticError("request_body_json must be valid JSON") from error
    if type(decoded) is not dict:
        raise SM121CacheSemanticError(
            "request_body_json must be a canonical no-thinking object"
        )
    return decoded


def _expected_profile_fields(profile_id: str) -> dict[str, object]:
    if profile_id == SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID:
        return {
            "id": profile_id,
            "description": SM121_CACHE_SEMANTIC_CACHE_OFF_DESCRIPTION,
            "args": SM121_CACHE_SEMANTIC_CACHE_OFF_ARGS,
        }
    if profile_id == SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID:
        return {
            "id": profile_id,
            "description": SM121_CACHE_SEMANTIC_CACHE_ON_DESCRIPTION,
            "args": SM121_CACHE_SEMANTIC_CACHE_ON_ARGS,
        }
    raise SM121CacheSemanticError("profile is not part of the SM121 semantic pair")


def validate_sm121_cache_semantic_candidate(model: Any) -> None:
    """Fail closed unless one sibling profile exactly matches its pinned arm."""

    if not is_sm121_cache_semantic_candidate(model):
        return
    profile_id = _profile_id(model)
    if profile_id is None:
        raise SM121CacheSemanticError("semantic profile ID is invalid")
    expected = _expected_profile_fields(profile_id)
    for field, value in _COMMON_MODEL_FIELDS.items():
        actual = _value(model, field)
        if field in {
            "tasks",
            "fetch_allow_patterns",
            "fetch_ignore_patterns",
            "sglang_source_overlays",
            "model_shards",
        } and isinstance(actual, (list, tuple)):
            actual = tuple(actual)
        _require(actual, value, field)
    for field, value in expected.items():
        actual = _value(model, field)
        if field == "args" and isinstance(actual, (list, tuple)):
            actual = tuple(actual)
        _require(actual, value, field)
    request_body = _canonical_request_body(_value(model, "request_body_json"))
    template = request_body.get("chat_template_kwargs")
    if (
        set(request_body) != {"chat_template_kwargs"}
        or type(template) is not dict
        or set(template) != {"enable_thinking"}
        or type(template.get("enable_thinking")) is not bool
        or template["enable_thinking"] is not False
    ):
        raise SM121CacheSemanticError(
            "request_body_json does not match the no-thinking contract"
        )


def validate_sm121_cache_semantic_pair(
    cache_off_model: Any, cache_on_model: Any
) -> None:
    """Validate B then A and prove the policy flag is their sole args delta."""

    _require(
        _profile_id(cache_off_model),
        SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID,
        "cache-off profile ID",
    )
    _require(
        _profile_id(cache_on_model),
        SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID,
        "cache-on profile ID",
    )
    validate_sm121_cache_semantic_candidate(cache_off_model)
    validate_sm121_cache_semantic_candidate(cache_on_model)
    off_args = tuple(_value(cache_off_model, "args") or ())
    on_args = tuple(_value(cache_on_model, "args") or ())
    _require(
        off_args,
        on_args[:_CACHE_OFF_INSERTION_INDEX]
        + ("--disable-radix-cache",)
        + on_args[_CACHE_OFF_INSERTION_INDEX:],
        "cache-off arguments",
    )


def validate_sm121_cache_semantic_suite(suite: Any) -> None:
    """Require the exact quality-plus-dedicated-semantic suite identity."""

    _require(_value(suite, "id"), SM121_CACHE_SEMANTIC_SUITE_ID, "suite.id")
    _require(_value(suite, "protocol_digest"), None, "suite.protocol_digest")
    _require(
        _value(suite, "description"),
        SM121_CACHE_SEMANTIC_SUITE_DESCRIPTION,
        "suite.description",
    )
    cases = _value(suite, "cases")
    if not isinstance(cases, (list, tuple)) or len(cases) != 2:
        raise SM121CacheSemanticError(
            "SM121 semantic suite must contain exactly two ordered cases"
        )
    expected_cases = (
        {
            "id": SM121_CACHE_SEMANTIC_QUALITY_CASE_ID,
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
            "id": SM121_CACHE_SEMANTIC_CASE_ID,
            "kind": "capability",
            "requires": ("chat",),
            "warmups": 0,
            "repetitions": 1,
            "max_output_tokens": SM121_CACHE_SEMANTIC_CASE_MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
            "concurrency": 1,
            "prompt_repetitions": 0,
            "max_turns": 1,
        },
    )
    for index, (case, expected) in enumerate(zip(cases, expected_cases, strict=True)):
        for field, value in expected.items():
            actual = _value(case, field)
            if field == "requires" and isinstance(actual, (list, tuple)):
                actual = tuple(actual)
            _require(actual, value, f"suite case {index} field {field}")


def validate_sm121_cache_semantic_static_attestation_event(event: object) -> None:
    """Validate one scalar-only source attestation for either semantic arm."""

    fields = frozenset(
        {
            "event",
            "arm",
            "fresh_server_lifetime",
            "candidate_source_tree",
            *SM121_CACHE_SOURCE_DIGESTS,
            *SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
        }
    )
    parsed = _require_event_fields(event, fields, "semantic static attestation")
    _require(
        parsed.get("event"),
        SM121_CACHE_SEMANTIC_STATIC_ATTESTATION_EVENT,
        "semantic static attestation event",
    )
    _require_arm(parsed.get("arm"), "semantic static attestation arm")
    lifetime = _require_positive_int(
        parsed.get("fresh_server_lifetime"), "semantic static lifetime"
    )
    if lifetime not in SM121_CACHE_SEMANTIC_LOCAL_LIFETIME_ORDINALS:
        raise SM121CacheSemanticError("semantic static lifetime is invalid")
    _require(
        parsed.get("candidate_source_tree"),
        SM121_STORAGE_SOURCE_TREE,
        "semantic static source tree",
    )
    for field, value in SM121_CACHE_SOURCE_DIGESTS.items():
        _require(parsed.get(field), value, field)
    for field, value in SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS.items():
        _require(parsed.get(field), value, field)


def validate_sm121_cache_semantic_runtime_attestation_event(event: object) -> None:
    """Validate a resolved scalar cache identity for its declared semantic arm."""

    runtime_fields = frozenset(next(iter(SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED.values())))
    fields = frozenset(
        {
            "event",
            "arm",
            "fresh_server_lifetime",
            "mamba_radix_cache_strategy",
            "max_mamba_cache_size",
            *runtime_fields,
        }
    )
    parsed = _require_event_fields(event, fields, "semantic runtime attestation")
    _require(
        parsed.get("event"),
        SM121_CACHE_SEMANTIC_RUNTIME_ATTESTATION_EVENT,
        "semantic runtime attestation event",
    )
    arm = _require_arm(parsed.get("arm"), "semantic runtime attestation arm")
    lifetime = _require_positive_int(
        parsed.get("fresh_server_lifetime"), "semantic runtime lifetime"
    )
    if lifetime not in SM121_CACHE_SEMANTIC_LOCAL_LIFETIME_ORDINALS:
        raise SM121CacheSemanticError("semantic runtime lifetime is invalid")
    for field, value in SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED[arm].items():
        _require(parsed.get(field), value, field)
    _require(
        parsed.get("mamba_radix_cache_strategy"),
        "extra_buffer_lazy",
        "mamba radix cache strategy",
    )
    _require(
        parsed.get("max_mamba_cache_size"),
        SM121_CACHE_SEMANTIC_MAX_MAMBA_CACHE_SIZE,
        "max Mamba cache size",
    )


def _validate_semantic_detail_fields(event: Mapping[str, object]) -> None:
    response_state = event.get("response_detail_state")
    usage_state = event.get("usage_detail_state")
    if response_state not in _SEMANTIC_DETAIL_STATES:
        raise SM121CacheSemanticError("semantic response detail state is invalid")
    if usage_state not in _SEMANTIC_DETAIL_STATES:
        raise SM121CacheSemanticError("semantic usage detail state is invalid")
    response_counts = tuple(
        _require_optional_nonnegative_int(
            event.get(field), f"semantic {field}"
        )
        for field in (
            "response_device_cached_tokens",
            "response_host_cached_tokens",
            "response_storage_cached_tokens",
        )
    )
    usage_count = _require_optional_nonnegative_int(
        event.get("usage_cached_tokens"), "semantic usage cached tokens"
    )
    if response_state == "zero_details":
        if response_counts != (0, 0, 0):
            raise SM121CacheSemanticError(
                "semantic zero response details must all be zero"
            )
    elif response_state == "nonzero_details":
        if (
            any(value is None for value in response_counts)
            or response_counts[0] is None
            or response_counts[0] <= 0
            or response_counts[1:] != (0, 0)
        ):
            raise SM121CacheSemanticError(
                "semantic nonzero response details must be device-only"
            )
    elif any(value is not None for value in response_counts):
        raise SM121CacheSemanticError(
            "semantic omitted response details must be null"
        )
    if usage_state == "zero_details" and usage_count != 0:
        raise SM121CacheSemanticError("semantic zero usage detail must be zero")
    if usage_state == "nonzero_details" and (
        usage_count is None or usage_count <= 0
    ):
        raise SM121CacheSemanticError(
            "semantic nonzero usage detail must be positive"
        )
    if usage_state in {"omitted", "null", "unexpected"} and usage_count is not None:
        raise SM121CacheSemanticError("semantic omitted usage detail must be null")


def validate_sm121_cache_semantic_turn_event(event: object) -> None:
    """Validate one scalar-only T0/T1/T2 observation without timing fields."""

    parsed = _require_event_fields(
        event, SM121_CACHE_SEMANTIC_TURN_EVENT_FIELDS, "semantic turn observation"
    )
    _require(
        parsed.get("event"),
        SM121_CACHE_SEMANTIC_TURN_OBSERVATION_EVENT,
        "semantic turn observation event",
    )
    arm = _require_arm(parsed.get("arm"), "semantic turn arm")
    _require(
        parsed.get("protocol_case_id"),
        SM121_CACHE_SEMANTIC_CASE_ID,
        "semantic protocol case ID",
    )
    _require_case_id(
        parsed.get("case_id"), SM121_CACHE_SEMANTIC_CASE_ID, "semantic case ID"
    )
    attempt_id = parsed.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise SM121CacheSemanticError("semantic turn attempt ID is invalid")
    turn = parsed.get("turn")
    if turn not in SM121_CACHE_SEMANTIC_TURN_ORDER:
        raise SM121CacheSemanticError("semantic turn is invalid")
    for field, value in (
        ("cache_details_requested", True),
        ("prompt_token_ids_requested", True),
        ("streaming", False),
        ("thinking_disabled", True),
    ):
        _require(parsed.get(field), value, f"semantic {field}")
    prompt_tokens = _require_positive_int(
        parsed.get("prompt_tokens"), "semantic prompt tokens"
    )
    _require_positive_int(parsed.get("completion_tokens"), "semantic completion tokens")
    _require(parsed.get("reasoning_tokens"), 0, "semantic reasoning tokens")
    for field in ("append_only_prompt_identity_verified",):
        if type(parsed.get(field)) is not bool:
            raise SM121CacheSemanticError(f"semantic {field} is invalid")
    cross_arm = parsed.get("cross_arm_prompt_identity_verified")
    if arm == SM121_CACHE_SEMANTIC_CACHE_OFF_ARM:
        _require(cross_arm, None, "semantic cache-off cross-arm identity")
    elif cross_arm is not True:
        raise SM121CacheSemanticError("semantic cache-on cross-arm identity is invalid")
    shared_prefix_tokens = _require_nonnegative_int(
        parsed.get("shared_prefix_tokens"), "semantic shared prefix tokens"
    )
    if turn == "T0":
        _require(shared_prefix_tokens, 0, "semantic T0 shared prefix tokens")
    elif not 0 < shared_prefix_tokens < prompt_tokens:
        raise SM121CacheSemanticError(
            "semantic later-turn shared prefix is outside its prompt"
        )
    _validate_semantic_detail_fields(parsed)
    for field in (
        "metrics_available",
        "guardrail_metrics_available",
        "metrics_before_settled",
        "metrics_after_settled",
    ):
        if type(parsed.get(field)) is not bool:
            raise SM121CacheSemanticError(f"semantic {field} is invalid")
    for field in ("metrics_before_polls", "metrics_after_polls"):
        _require_nonnegative_int(parsed.get(field), f"semantic {field}")
    for prefix in ("before", "after", "delta"):
        for metric in SM121_CACHE_SEMANTIC_METRIC_FIELDS:
            value = parsed.get(f"{prefix}_{metric}")
            if prefix == "delta" and metric in {
                "kv_available_tokens",
                "kv_evictable_tokens",
                "kv_used_tokens",
                "mamba_available_tokens",
                "mamba_evictable_tokens",
                "mamba_used_tokens",
            }:
                if type(value) is not int:
                    raise SM121CacheSemanticError(
                        f"semantic {prefix}_{metric} is invalid"
                    )
            else:
                _require_nonnegative_int(value, f"semantic {prefix}_{metric}")
            if parsed[f"delta_{metric}"] != (
                parsed[f"after_{metric}"] - parsed[f"before_{metric}"]
            ):
                raise SM121CacheSemanticError(
                    f"semantic delta for {metric} does not reconcile"
                )
    for prefix in ("before", "after"):
        for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
            if type(parsed.get(f"{prefix}_cached_{source}_series_present")) is not bool:
                raise SM121CacheSemanticError(
                    f"semantic {prefix} cached {source} marker is invalid"
                )
    if type(parsed.get("semantic_turn_admitted")) is not bool:
        raise SM121CacheSemanticError("semantic turn admission is invalid")
    if parsed.get("semantic_turn_basis") not in SM121_CACHE_SEMANTIC_TURN_BASES:
        raise SM121CacheSemanticError("semantic turn basis is invalid")
    admitted, basis = _derive_sm121_cache_semantic_turn_admission(parsed)
    _require(parsed.get("semantic_turn_admitted"), admitted, "semantic turn admission")
    _require(parsed.get("semantic_turn_basis"), basis, "semantic turn basis")


def _semantic_issue(
    code: str, message: str, **context: object
) -> dict[str, object]:
    return {"code": code, "message": message, **context}


def _semantic_turn_policy_issues(
    event: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Derive policy findings after scalar shape and arithmetic are checked."""
    issues: list[dict[str, object]] = []
    turn = str(event["turn"])
    arm = str(event["arm"])

    def issue(code: str, message: str) -> None:
        issues.append(_semantic_issue(code, message, turn=turn, arm=arm))

    prompt_tokens = int(event["prompt_tokens"])
    shared_prefix_tokens = int(event["shared_prefix_tokens"])
    if turn == "T0" and not (
        SM121_CACHE_SEMANTIC_COLD_INPUT_MIN_TOKENS
        <= prompt_tokens
        <= SM121_CACHE_SEMANTIC_COLD_INPUT_MAX_TOKENS
    ):
        issue("semantic_t0_prompt_window", "T0 prompt is outside the cold window")
    if turn != "T0" and shared_prefix_tokens < SM121_CACHE_SEMANTIC_COLD_INPUT_MIN_TOKENS:
        issue(
            "semantic_shared_prefix_window",
            "Later semantic turn has less than the required shared prefix",
        )
    if event["append_only_prompt_identity_verified"] is not True:
        issue(
            "semantic_append_identity",
            "Semantic turn did not verify append-only prompt identity",
        )
    if event["metrics_available"] is not True:
        issue("semantic_metrics_unavailable", "Semantic turn lacks native cache metrics")
    if event["guardrail_metrics_available"] is not True:
        issue(
            "semantic_guardrail_metrics_unavailable",
            "Semantic turn lacks native guardrail metrics",
        )
    for field in ("metrics_before_polls", "metrics_after_polls"):
        if int(event[field]) < 2:
            issue("semantic_metric_settle_polls", "Semantic metrics were not settled")
            break
    if (
        event["metrics_before_settled"] is not True
        or event["metrics_after_settled"] is not True
    ):
        issue("semantic_metric_settle", "Semantic metrics were not settled")
    if int(event["delta_prefill_input_tokens"]) <= 0:
        issue("semantic_input_delta", "Semantic turn has no native input delta")
    for metric in (
        "prefill_host_hit_tokens",
        "prefill_storage_hit_tokens",
        "cached_host_tokens",
        "cached_storage_tokens",
        "evicted_tokens",
        "retracted_requests",
    ):
        if int(event[f"delta_{metric}"]) != 0:
            issue(
                "semantic_cache_guardrail",
                "Semantic turn observed host/storage cache or guardrail activity",
            )
            break

    zero_required = arm == SM121_CACHE_SEMANTIC_CACHE_OFF_ARM or turn == "T0"
    if zero_required:
        if event["response_detail_state"] not in {
            "zero_details",
            "omitted",
            "null",
        } or event["usage_detail_state"] not in {
            "zero_details",
            "omitted",
            "null",
        }:
            issue(
                "semantic_zero_hit_details",
                "Zero-hit turn lacks a valid zero-cache detail state",
            )
        for field in (
            "response_device_cached_tokens",
            "response_host_cached_tokens",
            "response_storage_cached_tokens",
            "usage_cached_tokens",
        ):
            value = event[field]
            if value is not None and value != 0:
                issue("semantic_zero_hit_details", "Zero-hit turn reported cached tokens")
                break
        for metric in (
            "prefill_device_hit_tokens",
            "prefill_host_hit_tokens",
            "prefill_storage_hit_tokens",
            "cached_total_tokens",
            "cached_device_tokens",
            "cached_host_tokens",
            "cached_storage_tokens",
        ):
            if int(event[f"delta_{metric}"]) != 0:
                issue("semantic_zero_hit_native", "Zero-hit turn has native cache hits")
                break
    else:
        response_device = event["response_device_cached_tokens"]
        response_state = event["response_detail_state"]
        prefill_device = int(event["delta_prefill_device_hit_tokens"])
        cached_device = int(event["delta_cached_device_tokens"])
        if response_state != "nonzero_details" or not isinstance(response_device, int):
            issue(
                "semantic_positive_detail",
                "Cache-on later turn lacks explicit device cache details",
            )
        elif (
            response_device <= 0
            or response_device > shared_prefix_tokens
            or prefill_device <= 0
            or cached_device <= 0
            or response_device != prefill_device
            or response_device != cached_device
        ):
            issue(
                "semantic_positive_native_reconciliation",
                "Cache-on device details do not reconcile native hit counters",
            )
        usage_cached = event["usage_cached_tokens"]
        if usage_cached is not None and usage_cached != response_device:
            issue(
                "semantic_usage_reconciliation",
                "Cache-on usage detail does not reconcile device cache detail",
            )
    return tuple(issues)


def _derive_sm121_cache_semantic_turn_admission(
    event: Mapping[str, object],
) -> tuple[bool, str]:
    """Return the deterministic admission result for an already-shaped turn."""

    issues = _semantic_turn_policy_issues(event)
    if not issues:
        return True, "admitted"
    codes = {str(issue["code"]) for issue in issues}
    if codes.intersection(
        {
            "semantic_t0_prompt_window",
            "semantic_shared_prefix_window",
            "semantic_append_identity",
        }
    ):
        return False, "identity_mismatch"
    if codes.intersection(
        {
            "semantic_metrics_unavailable",
            "semantic_metric_settle_polls",
            "semantic_metric_settle",
            "semantic_input_delta",
        }
    ):
        return False, "metric_unavailable"
    if codes.intersection(
        {
            "semantic_guardrail_metrics_unavailable",
            "semantic_cache_guardrail",
        }
    ):
        return False, "guardrail_activity"
    zero_required = (
        event["arm"] == SM121_CACHE_SEMANTIC_CACHE_OFF_ARM
        or event["turn"] == "T0"
    )
    if zero_required:
        return False, "zero_hit_not_reconciled"
    return False, "positive_device_hit_not_reconciled"


def derive_sm121_cache_semantic_turn_admission(
    event: Mapping[str, object],
) -> tuple[bool, str]:
    """Derive admission before adding its two persisted result fields.

    Callers construct every other scalar field first, call this helper, then
    add ``semantic_turn_admitted`` and ``semantic_turn_basis``.  The finished
    event is checked again by :func:`validate_sm121_cache_semantic_turn_event`.
    """

    if type(event) is not dict:
        raise SM121CacheSemanticError("semantic turn observation is not an object")
    try:
        return _derive_sm121_cache_semantic_turn_admission(event)
    except (KeyError, TypeError, ValueError) as error:
        raise SM121CacheSemanticError("semantic turn admission inputs are invalid") from error


def sm121_cache_semantic_turn_issues(event: object) -> tuple[dict[str, object], ...]:
    """Return semantic admission violations for one structurally valid turn."""

    try:
        validate_sm121_cache_semantic_turn_event(event)
    except SM121CacheSemanticError:
        return (
            _semantic_issue(
                "semantic_turn_schema",
                "Semantic turn observation does not match the scalar contract",
            ),
        )
    assert isinstance(event, Mapping)
    return _semantic_turn_policy_issues(event)


def expected_sm121_cache_semantic_event_counts() -> dict[str, int]:
    """Return the exact successful event topology for one B or A arm journal."""

    return {
        "case_complete": 2,
        "case_start": 2,
        "measurement_complete": 1,
        "measurement_started": 1,
        "request_complete": 7,
        "run_complete": 1,
        "run_start": 1,
        "server_ready": 2,
        "server_stopped": 2,
        SM121_CACHE_SEMANTIC_STATIC_ATTESTATION_EVENT: 2,
        SM121_CACHE_SEMANTIC_RUNTIME_ATTESTATION_EVENT: 2,
        SM121_CACHE_SEMANTIC_TURN_OBSERVATION_EVENT: 3,
    }


def sm121_cache_semantic_lifecycle_issues(
    events: Sequence[Mapping[str, object]],
    *,
    planned_case_ids: Sequence[str],
    arm: str,
) -> tuple[dict[str, object], ...]:
    """Return scalar topology and semantic violations for one arm journal.

    Each arm uses two local fresh lifetimes: a quality-only lifetime first, then
    a semantic T0/T1/T2 lifetime with no prior inference.  The function never
    copies a prompt, completion, token ID, timing, request tag, or server text
    into its findings.
    """

    issues: list[dict[str, object]] = []

    def issue(code: str, message: str, **context: object) -> None:
        issues.append(_semantic_issue(code, message, **context))

    if arm not in SM121_CACHE_SEMANTIC_ARM_ORDER:
        issue("semantic_arm", "Semantic lifecycle arm is invalid")
        return tuple(issues)
    expected_cases = tuple(planned_case_ids)
    if len(expected_cases) != 2 or any(
        not isinstance(case_id, str) or not case_id for case_id in expected_cases
    ):
        issue("semantic_planned_cases", "Semantic lifecycle has invalid planned cases")
        return tuple(issues)
    if any(type(event) is not dict for event in events):
        issue("semantic_event_object", "Semantic journal contains a non-object event")
        return tuple(issues)
    expected_counts = expected_sm121_cache_semantic_event_counts()
    positions = {
        event_type: [
            index
            for index, event in enumerate(events)
            if event.get("event") == event_type
        ]
        for event_type in expected_counts
    }
    for event_type, expected in expected_counts.items():
        actual = len(positions[event_type])
        if actual != expected:
            issue(
                "semantic_event_count",
                "Semantic event count does not match the frozen topology",
                event=event_type,
                expected=expected,
                actual=actual,
            )
    allowed_events = set(expected_counts)
    for index, event in enumerate(events):
        if event.get("event") not in allowed_events:
            issue(
                "semantic_unexpected_event",
                "Semantic journal contains an event outside its frozen topology",
                event_index=index,
            )
    if any(len(positions[name]) != count for name, count in expected_counts.items()):
        return tuple(issues)

    run_start = positions["run_start"][0]
    measurement_started = positions["measurement_started"][0]
    measurement_complete = positions["measurement_complete"][0]
    run_complete = positions["run_complete"][0]
    if run_start != 0:
        issue("semantic_run_start", "Semantic run_start must be first")
    if events[run_start].get("execution_mode") != SM121_CACHE_SEMANTIC_EXECUTION_MODE:
        issue("semantic_execution_mode", "Semantic run uses the wrong execution mode")
    if events[run_complete].get("status") != "completed":
        issue("semantic_run_status", "Semantic run_complete must be completed")
    if run_complete != len(events) - 1:
        issue("semantic_run_terminal", "Semantic run_complete must be final")
    if not run_start < measurement_started < measurement_complete < run_complete:
        issue("semantic_measurement_order", "Semantic measurement lifecycle is invalid")

    static_positions = positions[SM121_CACHE_SEMANTIC_STATIC_ATTESTATION_EVENT]
    runtime_positions = positions[SM121_CACHE_SEMANTIC_RUNTIME_ATTESTATION_EVENT]
    ready_positions = positions["server_ready"]
    stopped_positions = positions["server_stopped"]
    case_starts = positions["case_start"]
    case_completes = positions["case_complete"]
    for index, expected_lifetime in enumerate(
        SM121_CACHE_SEMANTIC_LOCAL_LIFETIME_ORDINALS
    ):
        for event_index, validator, code in (
            (
                static_positions[index],
                validate_sm121_cache_semantic_static_attestation_event,
                "semantic_static_attestation",
            ),
            (
                runtime_positions[index],
                validate_sm121_cache_semantic_runtime_attestation_event,
                "semantic_runtime_attestation",
            ),
        ):
            event = events[event_index]
            try:
                validator(event)
            except SM121CacheSemanticError:
                issue(code, "Semantic attestation does not match its scalar contract")
                continue
            if event.get("arm") != arm:
                issue(code, "Semantic attestation has the wrong arm")
            if event.get("fresh_server_lifetime") != expected_lifetime:
                issue(code, "Semantic attestation has the wrong local lifetime")
        ready = events[ready_positions[index]]
        stopped = events[stopped_positions[index]]
        expected_case_id = expected_cases[index]
        if ready.get("backend") != "sglang":
            issue("semantic_ready_backend", "Semantic server_ready must identify SGLang")
        if ready.get("fresh_server_lifetime") != expected_lifetime:
            issue("semantic_ready_lifetime", "Semantic server_ready lifetime is invalid")
        if stopped.get("fresh_server_lifetime") != expected_lifetime:
            issue("semantic_stop_lifetime", "Semantic server_stopped lifetime is invalid")
        if ready.get("first_inference_is_case") is not True:
            issue("semantic_first_inference", "Semantic lifetime has a prior inference")
        if ready.get("case_id") != expected_case_id:
            issue("semantic_ready_case", "Semantic server_ready has the wrong case")
        if not (
            static_positions[index]
            < runtime_positions[index]
            < ready_positions[index]
            < case_starts[index]
            < case_completes[index]
            < stopped_positions[index]
        ):
            issue("semantic_lifetime_order", "Semantic lifetime event order is invalid")
    if not (
        stopped_positions[0]
        < static_positions[1]
        < runtime_positions[1]
        < ready_positions[1]
        < stopped_positions[1]
        < measurement_complete
    ):
        issue("semantic_lifetime_boundary", "Semantic lifetimes are not isolated")

    request_positions = positions["request_complete"]
    request_blocks: list[list[int]] = []
    for case_index, expected_count in enumerate((4, 3)):
        start_index = case_starts[case_index]
        complete_index = case_completes[case_index]
        start = events[start_index]
        complete = events[complete_index]
        expected_case_id = expected_cases[case_index]
        attempt_id = start.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            issue("semantic_case_attempt", "Semantic case start lacks an attempt ID")
        for event_index, case_event in ((start_index, start), (complete_index, complete)):
            if case_event.get("case_id") != expected_case_id:
                issue(
                    "semantic_case_id",
                    "Semantic case event has the wrong frozen case ID",
                    event_index=event_index,
                )
            if case_event.get("attempt_id") != attempt_id:
                issue(
                    "semantic_case_attempt",
                    "Semantic case event does not match its start attempt",
                    event_index=event_index,
                )
        if case_index == 0 and complete.get("validation_passed") is not True:
            issue("semantic_case_validation", "Semantic quality validation is not clean")
        block = [
            index
            for index in request_positions
            if start_index < index < complete_index
        ]
        request_blocks.append(block)
        if len(block) != expected_count:
            issue(
                "semantic_case_request_count",
                "Semantic case has the wrong request count",
                case_number=case_index,
                expected=expected_count,
                actual=len(block),
            )
        for request_index in block:
            request = events[request_index]
            if (
                request.get("case_id") != expected_case_id
                or request.get("attempt_id") != attempt_id
            ):
                issue(
                    "semantic_request_binding",
                    "Semantic request is not bound to its case attempt",
                    event_index=request_index,
                )

    turn_positions = positions[SM121_CACHE_SEMANTIC_TURN_OBSERVATION_EVENT]
    semantic_attempt = events[case_starts[1]].get("attempt_id")
    turn_events: list[Mapping[str, object]] = []
    for position, expected_turn in zip(
        turn_positions, SM121_CACHE_SEMANTIC_TURN_ORDER, strict=True
    ):
        turn_event = events[position]
        turn_events.append(turn_event)
        try:
            validate_sm121_cache_semantic_turn_event(turn_event)
        except SM121CacheSemanticError:
            issue("semantic_turn_attestation", "Semantic turn has an invalid schema")
            continue
        if (
            turn_event.get("case_id") != expected_cases[1]
            or turn_event.get("attempt_id") != semantic_attempt
            or turn_event.get("arm") != arm
            or turn_event.get("turn") != expected_turn
        ):
            issue("semantic_turn_binding", "Semantic turn has the wrong binding")
        issues.extend(sm121_cache_semantic_turn_issues(turn_event))
    if len(turn_events) == len(SM121_CACHE_SEMANTIC_TURN_ORDER) and all(
        type(event.get("semantic_turn_admitted")) is bool
        for event in turn_events
    ):
        semantic_validation = all(
            event["semantic_turn_admitted"] for event in turn_events
        )
        if events[case_completes[1]].get("validation_passed") is not semantic_validation:
            issue(
                "semantic_case_validation",
                "Semantic case validation disagrees with turn admission",
            )
    if len(request_blocks[1]) == 3:
        for turn_index, (turn_position, request_position) in enumerate(
            zip(turn_positions, request_blocks[1], strict=True)
        ):
            if turn_position + 1 != request_position:
                issue(
                    "semantic_turn_request_adjacency",
                    "Semantic turn must immediately precede its request record",
                    turn_index=turn_index,
                )
        if turn_positions[0] + 1 != request_blocks[1][0]:
            issue(
                "semantic_t0_first_request",
                "T0 must be the first request in the semantic lifetime",
            )
    if len(turn_events) == len(SM121_CACHE_SEMANTIC_TURN_ORDER):
        prompts = [int(event["prompt_tokens"]) for event in turn_events]
        shared_prefixes = [int(event["shared_prefix_tokens"]) for event in turn_events]
        if not prompts[0] < prompts[1] < prompts[2]:
            issue("semantic_prompt_growth", "Semantic prompts do not grow by turn")
        if not (
            shared_prefixes[1] <= prompts[0]
            and shared_prefixes[2] <= prompts[1]
            and shared_prefixes[1] <= shared_prefixes[2]
        ):
            issue(
                "semantic_append_prefix",
                "Semantic shared prefixes do not match append-only history",
            )
    return tuple(issues)


def _pair_binding_payload(binding: Mapping[str, object]) -> dict[str, object]:
    return {
        field: binding[field]
        for field in sorted(
            SM121_CACHE_SEMANTIC_PAIR_BINDING_FIELDS - {"pair_binding_sha256"}
        )
    }


def _sm121_cache_semantic_sha256(payload: Mapping[str, object], *, name: str) -> str:
    """Hash a private-free semantic binding payload with one canonical form."""

    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SM121CacheSemanticError(
            f"semantic {name} is not canonical JSON"
        ) from error
    return "sha256:" + hashlib.sha256(canonical.encode("ascii")).hexdigest()


def sm121_cache_semantic_pair_instance_sha256(
    cache_off_run_nonce: object, cache_on_run_nonce: object
) -> str:
    """Commit a frozen B/A plan pair without retaining either nonce publicly.

    A profile's plan fingerprint is deterministic. This opaque commitment ties
    the two otherwise independent frozen plans to one controller-created pair
    while keeping their random run nonces raw-only.
    """

    if (
        not isinstance(cache_off_run_nonce, str)
        or re.fullmatch(r"[0-9a-f]{32}", cache_off_run_nonce) is None
        or not isinstance(cache_on_run_nonce, str)
        or re.fullmatch(r"[0-9a-f]{32}", cache_on_run_nonce) is None
    ):
        raise SM121CacheSemanticError("semantic pair run nonce is invalid")
    if cache_off_run_nonce == cache_on_run_nonce:
        raise SM121CacheSemanticError("semantic pair run nonces must differ")
    return _sm121_cache_semantic_sha256(
        {
            "schema_version": 1,
            "cache_off_run_nonce": cache_off_run_nonce,
            "cache_on_run_nonce": cache_on_run_nonce,
        },
        name="pair instance",
    )


def sm121_cache_semantic_cache_off_receipt_sha256(
    pair_instance_sha256: object,
    cache_off_plan_fingerprint: object,
    cache_off_pair_binding_sha256: object,
) -> str:
    """Return A's scalar receipt for the completed B control.

    The input is restricted to opaque binding commitments: no prompt,
    completion, token identity, timing, or raw nonce is carried forward.
    """

    if (
        not isinstance(pair_instance_sha256, str)
        or _SHA256_PATTERN.fullmatch(pair_instance_sha256) is None
        or not isinstance(cache_off_plan_fingerprint, str)
        or _FINGERPRINT_PATTERN.fullmatch(cache_off_plan_fingerprint) is None
        or not isinstance(cache_off_pair_binding_sha256, str)
        or _SHA256_PATTERN.fullmatch(cache_off_pair_binding_sha256) is None
    ):
        raise SM121CacheSemanticError("semantic cache-off receipt input is invalid")
    return _sm121_cache_semantic_sha256(
        {
            "schema_version": 1,
            "status": "complete",
            "pair_instance_sha256": pair_instance_sha256,
            "cache_off_plan_fingerprint": cache_off_plan_fingerprint,
            "cache_off_pair_binding_sha256": cache_off_pair_binding_sha256,
        },
        name="cache-off receipt",
    )


def sm121_cache_semantic_pair_binding_sha256(binding: Mapping[str, object]) -> str:
    """Return the canonical digest of a binding excluding its own digest field."""

    if type(binding) is not dict:
        raise SM121CacheSemanticError("semantic pair binding is not an object")
    fields = frozenset(binding)
    payload_fields = SM121_CACHE_SEMANTIC_PAIR_BINDING_FIELDS - {
        "pair_binding_sha256"
    }
    if fields not in {payload_fields, SM121_CACHE_SEMANTIC_PAIR_BINDING_FIELDS}:
        raise SM121CacheSemanticError("semantic pair binding fields are invalid")
    try:
        canonical = json.dumps(
            _pair_binding_payload(binding),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SM121CacheSemanticError("semantic pair binding is not canonical JSON") from error
    return "sha256:" + hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _validate_sm121_cache_semantic_pair_binding_shape(
    binding: object,
) -> dict[str, object]:
    if type(binding) is not dict:
        raise SM121CacheSemanticError("semantic pair binding is not an object")
    if frozenset(binding) != SM121_CACHE_SEMANTIC_PAIR_BINDING_FIELDS:
        raise SM121CacheSemanticError("semantic pair binding fields are invalid")
    _require(
        binding.get("schema_version"),
        SM121_CACHE_SEMANTIC_PAIR_BINDING_SCHEMA_VERSION,
        "semantic pair binding schema",
    )
    _require(
        binding.get("suite_id"),
        SM121_CACHE_SEMANTIC_SUITE_ID,
        "semantic pair binding suite",
    )
    _require(
        binding.get("execution_mode"),
        SM121_CACHE_SEMANTIC_EXECUTION_MODE,
        "semantic pair binding execution mode",
    )
    arm = _require_arm(binding.get("arm"), "semantic pair binding arm")
    _require(
        binding.get("profile_id"),
        SM121_CACHE_SEMANTIC_PROFILE_ID_BY_ARM[arm],
        "semantic pair binding profile",
    )
    _require(
        binding.get("arm_order"),
        list(SM121_CACHE_SEMANTIC_ARM_ORDER),
        "semantic pair binding arm order",
    )
    _require(
        binding.get("local_lifetime_order"),
        list(SM121_CACHE_SEMANTIC_LOCAL_LIFETIME_ORDER),
        "semantic pair binding local lifetime order",
    )
    _require(
        binding.get("quality_case_id"),
        SM121_CACHE_SEMANTIC_QUALITY_CASE_ID,
        "semantic pair binding quality case",
    )
    _require(
        binding.get("semantic_case_id"),
        SM121_CACHE_SEMANTIC_CASE_ID,
        "semantic pair binding semantic case",
    )
    _require(
        binding.get("semantic_case_metadata"),
        sm121_cache_semantic_case_metadata(),
        "semantic pair binding semantic metadata",
    )
    peer_fingerprint = binding.get("peer_plan_fingerprint")
    if not isinstance(peer_fingerprint, str) or _FINGERPRINT_PATTERN.fullmatch(
        peer_fingerprint
    ) is None:
        raise SM121CacheSemanticError("semantic pair peer fingerprint is invalid")
    pair_instance = binding.get("pair_instance_sha256")
    if not isinstance(pair_instance, str) or _SHA256_PATTERN.fullmatch(pair_instance) is None:
        raise SM121CacheSemanticError("semantic pair instance digest is invalid")
    binding_digest = binding.get("pair_binding_sha256")
    if not isinstance(binding_digest, str) or _SHA256_PATTERN.fullmatch(binding_digest) is None:
        raise SM121CacheSemanticError("semantic pair binding digest is invalid")
    _require(
        binding_digest,
        sm121_cache_semantic_pair_binding_sha256(binding),
        "semantic pair binding digest",
    )
    return binding


def validate_sm121_cache_semantic_pair_binding(
    plan_binding: object,
    model: Any,
    suite: Any,
    peer_plan_fingerprint: object | None = None,
    peer_binding: object | None = None,
) -> None:
    """Validate one frozen arm binding without reading a plan or raw workload.

    The campaign controller owns reciprocal plan-fingerprint and peer-digest
    checks.  This pure validator pins the local arm, profile, suite, renderer
    metadata, and canonical binding digest before those cross-plan checks run.
    """

    validate_sm121_cache_semantic_candidate(model)
    validate_sm121_cache_semantic_suite(suite)
    if not is_sm121_cache_semantic_plan(model, suite):
        raise SM121CacheSemanticError("semantic pair binding has an invalid selection")
    binding = _validate_sm121_cache_semantic_pair_binding_shape(plan_binding)
    arm = sm121_cache_semantic_arm(model)
    _require(binding.get("arm"), arm, "semantic pair binding selected arm")
    _require(
        binding.get("profile_id"),
        _profile_id(model),
        "semantic pair binding selected profile",
    )
    if peer_plan_fingerprint is not None:
        if (
            not isinstance(peer_plan_fingerprint, str)
            or _FINGERPRINT_PATTERN.fullmatch(peer_plan_fingerprint) is None
        ):
            raise SM121CacheSemanticError("semantic pair supplied peer fingerprint is invalid")
        _require(
            binding.get("peer_plan_fingerprint"),
            peer_plan_fingerprint,
            "semantic pair binding peer fingerprint",
        )
    if peer_binding is not None:
        peer = _validate_sm121_cache_semantic_pair_binding_shape(peer_binding)
        peer_arm = _require_arm(peer.get("arm"), "semantic pair peer arm")
        if peer_arm == arm:
            raise SM121CacheSemanticError("semantic pair binding has a duplicate arm")
        _require(
            binding.get("pair_instance_sha256"),
            peer.get("pair_instance_sha256"),
            "semantic pair instance digest",
        )


def sm121_cache_semantic_storage_identity() -> dict[str, object]:
    """Return the immutable storage/image identity shared by both arms."""

    return {
        "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
        "build_contract_sha256": SM121_STORAGE_BUILD_CONTRACT_SHA256,
        "docker_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
        "image": SM121_STORAGE_LOCAL_IMAGE_TAG,
        "source": SM121_STORAGE_SOURCE,
        "revision": SM121_STORAGE_REVISION,
    }
