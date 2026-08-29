"""Frozen profile contract for the current-SM121 chunked-prefill study.

This is intentionally distinct from the cache-policy performance lane.  Both
arms retain the admitted UnifiedRadixCache/lazy-Mamba bundle; the only serving
configuration difference is ``--chunked-prefill-size``.  The profiles are
blocked from generic execution until their dedicated fresh-lifetime controller
and scalar evidence contract are present.

The immediate study is a long-context prefill proxy.  It is not an agentic
coding result: the current admitted SM121 profile is chat/no-thinking only.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, DecimalException
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .sglang_sm121_cache_observability import (
    SM121_CACHE_OBSERVABILITY_CACHED_SERIES,
    SM121_CACHE_OBSERVABILITY_METRIC_FIELDS,
    SM121_CACHE_SOURCE_DIGESTS,
)
from .sglang_sm121_cache_semantic import (
    SM121_CACHE_SEMANTIC_GUARDRAIL_METRIC_FIELDS,
    SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED,
    SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
)

from .sglang_sm121_storage import (
    SM121_STORAGE_CACHE_PAGES,
    SM121_STORAGE_CONTEXT_LENGTH,
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_LOCAL_IMAGE_TAG,
    SM121_STORAGE_MAX_BATCH_PAGES,
    SM121_STORAGE_MODE,
    SM121_STORAGE_NATIVE_CONTEXT,
    SM121_STORAGE_SOURCE_TREE,
    SM121_STORAGE_QUEUE_DEPTH,
    SM121_STORAGE_REVISION,
    SM121_STORAGE_SOURCE,
    SM121_STORAGE_WEIGHT_FILE_COUNT,
    SM121_STORAGE_WEIGHT_SIZE_BYTES,
)


SM121_CHUNKED_PREFILL_PERFORMANCE_SUITE_ID = (
    "qwen38-flash-next-sm121-triton-storage-chunked-prefill-performance-v1"
)
SM121_CHUNKED_PREFILL_PERFORMANCE_CAMPAIGN_ID = (
    "qwen38-flash-next-sm121-chunked-prefill-performance-v1"
)
SM121_CHUNKED_PREFILL_PERFORMANCE_EXECUTION_MODE = (
    "sm121_storage_chunked_prefill_performance_abba_fresh_lifetimes"
)
SM121_CHUNKED_PREFILL_PERFORMANCE_SCHEMA_VERSION = 1
SM121_CHUNKED_PREFILL_PERFORMANCE_PAIR_BINDING_SCHEMA_VERSION = 1
SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID = (
    "qwen38-flash-next-nvfp4-sm121-triton-storage-chunked-prefill-performance-1k-sglang"
)
SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID = (
    "qwen38-flash-next-nvfp4-sm121-triton-storage-chunked-prefill-performance-2k-sglang"
)
SM121_CHUNKED_PREFILL_PERFORMANCE_PROFILE_IDS = frozenset(
    {
        SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID,
        SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID,
    }
)
SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_ARM = "A"
SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_ARM = "B"
SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER = ("A", "B", "B", "A")
SM121_CHUNKED_PREFILL_PERFORMANCE_LIFETIME_ARMS = {
    1: "A",
    2: "A",
    3: "B",
    4: "B",
    5: "B",
    6: "B",
    7: "A",
    8: "A",
}
SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S = 1_200
SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID = "synthetic-exact-answer-v2"
SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT = 4
SM121_CHUNKED_PREFILL_PERFORMANCE_CASE_ID = (
    "sm121-chunked-prefill-60k-static-history-v1"
)
SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS = ("T0", "T1", "T2")
SM121_CHUNKED_PREFILL_PERFORMANCE_SERVED_NAME = (
    "qwen38-flash-next-nvfp4-sm121-storage-chunked-prefill-performance"
)
SM121_CHUNKED_PREFILL_PERFORMANCE_MAX_MAMBA_CACHE_SIZE = 4
SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_CHUNK_SIZE = 1_024
SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_CHUNK_SIZE = 2_048
SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MIN_TOKENS = 56 * 1_024
SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MAX_TOKENS = 62 * 1_024
SM121_CHUNKED_PREFILL_PERFORMANCE_PROMOTION_RATIO = Decimal("0.95")
SM121_CHUNKED_PREFILL_PERFORMANCE_APPEND_WALL_GUARDRAIL_RATIO = Decimal("1.05")
SM121_CHUNKED_PREFILL_PERFORMANCE_FULL_WALL_GUARDRAIL_RATIO = Decimal("1.05")
SM121_CHUNKED_PREFILL_PERFORMANCE_STATIC_EVENT = (
    "sm121_chunked_prefill_performance_static_attestation"
)
SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EVENT = (
    "sm121_chunked_prefill_performance_runtime_attestation"
)
SM121_CHUNKED_PREFILL_PERFORMANCE_TURN_EVENT = (
    "sm121_chunked_prefill_performance_turn_observation"
)
SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EXPECTED = dict(
    SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED["A"]
)
SM121_CHUNKED_PREFILL_PERFORMANCE_METRIC_FIELDS = (
    *SM121_CACHE_OBSERVABILITY_METRIC_FIELDS,
    *SM121_CACHE_SEMANTIC_GUARDRAIL_METRIC_FIELDS,
)
SM121_CHUNKED_PREFILL_PERFORMANCE_SUITE_DESCRIPTION = (
    "Frozen fresh-lifetime A/B/B/A long-context prefill study of the current "
    "SM121 native-NVMe Qwen3.8 Flash-Next cache-on bundle. The 1K and 2K "
    "profiles differ only in chunked-prefill size; the dedicated controller "
    "uses a 60K deterministic static-history proxy and does not claim an "
    "agentic coding result."
)
SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_DESCRIPTION = (
    "SM121 chunked-prefill performance A: current cache-on lazy-Mamba 1,024 "
    "token control. Only the dedicated A/B/B/A performance controller may "
    "execute it."
)
SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_DESCRIPTION = (
    "SM121 chunked-prefill performance B: current cache-on lazy-Mamba 2,048 "
    "token candidate. Only the dedicated A/B/B/A performance controller may "
    "execute it."
)

SM121_CHUNKED_PREFILL_PERFORMANCE_COMMON_ARGS = (
    "--served-model-name",
    SM121_CHUNKED_PREFILL_PERFORMANCE_SERVED_NAME,
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
    str(SM121_CHUNKED_PREFILL_PERFORMANCE_MAX_MAMBA_CACHE_SIZE),
    "--page-size",
    "64",
    "--mem-fraction-static",
    "0.85",
    "--max-total-tokens",
    str(SM121_STORAGE_CONTEXT_LENGTH),
    "--context-length",
    str(SM121_STORAGE_CONTEXT_LENGTH),
    "--chunked-prefill-size",
    "pending",
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
_CHUNK_SIZE_INDEX = SM121_CHUNKED_PREFILL_PERFORMANCE_COMMON_ARGS.index(
    "--chunked-prefill-size"
) + 1

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FINGERPRINT = re.compile(r"[0-9a-f]{16}\Z")
_PAIR_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "suite_id",
        "execution_mode",
        "arm_order",
        "profile_ids",
        "chunked_prefill_sizes",
        "quality_case_id",
        "timed_case_id",
        "cell_timeout_s",
        "campaign_instance_sha256",
        "plan_fingerprints",
        "pair_binding_sha256",
    }
)
_TURN_EVENT_FIELDS = frozenset(
    {
        "event",
        "arm",
        "lifetime_ordinal",
        "case_id",
        "protocol_case_id",
        "turn",
        "cache_details_requested",
        "prompt_token_ids_requested",
        "streaming",
        "thinking_disabled",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "shared_prefix_tokens",
        "append_only_prompt_identity_verified",
        "cross_lifetime_prompt_identity_verified",
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
        "request_wall_s",
        "timed_turn_admitted",
        "timed_turn_basis",
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


def _with_chunk_size(size: int) -> tuple[str, ...]:
    arguments = list(SM121_CHUNKED_PREFILL_PERFORMANCE_COMMON_ARGS)
    arguments[_CHUNK_SIZE_INDEX] = str(size)
    return tuple(arguments)


SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_ARGS = _with_chunk_size(
    SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_CHUNK_SIZE
)
SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_ARGS = _with_chunk_size(
    SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_CHUNK_SIZE
)


class SM121ChunkedPrefillPerformanceError(ValueError):
    """Raised when a proposed 1K/2K prefill study drifts from its contract."""


@dataclass(frozen=True, slots=True)
class ChunkedPrefillPerformanceScore:
    """Scalar reducer output for the one-axis 1K-versus-2K comparison."""

    status: str
    decision: str
    a_t0_wall_s: float | None
    b_t0_wall_s: float | None
    a_later_wall_s: float | None
    b_later_wall_s: float | None
    a_full_wall_s: float | None
    b_full_wall_s: float | None
    candidate_t0_wall_ratio: float | None
    candidate_later_wall_ratio: float | None
    candidate_full_wall_ratio: float | None

    def to_mapping(self) -> dict[str, object]:
        return {
            "status": self.status,
            "decision": self.decision,
            "a_t0_wall_s": self.a_t0_wall_s,
            "b_t0_wall_s": self.b_t0_wall_s,
            "a_later_wall_s": self.a_later_wall_s,
            "b_later_wall_s": self.b_later_wall_s,
            "a_full_wall_s": self.a_full_wall_s,
            "b_full_wall_s": self.b_full_wall_s,
            "candidate_t0_wall_ratio": self.candidate_t0_wall_ratio,
            "candidate_later_wall_ratio": self.candidate_later_wall_ratio,
            "candidate_full_wall_ratio": self.candidate_full_wall_ratio,
        }


def _value(item: Any, field: str) -> object:
    return item.get(field) if isinstance(item, Mapping) else getattr(item, field, None)


def _profile_id(value: Any) -> str | None:
    candidate = value if isinstance(value, str) else _value(value, "id")
    return candidate if isinstance(candidate, str) else None


def sm121_chunked_prefill_performance_arm(value: Any) -> str:
    profile_id = _profile_id(value)
    if profile_id == SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID:
        return SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_ARM
    if profile_id == SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID:
        return SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_ARM
    raise SM121ChunkedPrefillPerformanceError(
        "profile is not a chunked-prefill performance arm"
    )


def is_sm121_chunked_prefill_performance_candidate(model: Any) -> bool:
    """Return whether a profile belongs exclusively to this frozen experiment."""

    return _profile_id(model) in SM121_CHUNKED_PREFILL_PERFORMANCE_PROFILE_IDS


def is_sm121_chunked_prefill_performance_plan(model: Any, suite: Any) -> bool:
    return (
        is_sm121_chunked_prefill_performance_candidate(model)
        and _value(suite, "id") == SM121_CHUNKED_PREFILL_PERFORMANCE_SUITE_ID
    )


def _expected_profile(profile_id: str) -> dict[str, object]:
    if profile_id == SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID:
        return {
            "description": SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_DESCRIPTION,
            "args": SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_ARGS,
        }
    if profile_id == SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID:
        return {
            "description": SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_DESCRIPTION,
            "args": SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_ARGS,
        }
    raise SM121ChunkedPrefillPerformanceError(
        "profile is not a chunked-prefill performance arm"
    )


def validate_sm121_chunked_prefill_performance_candidate(model: Any) -> None:
    """Require one exact current-SM121 profile and nothing broadly runnable."""

    if not is_sm121_chunked_prefill_performance_candidate(model):
        return
    profile_id = _profile_id(model)
    assert profile_id is not None
    expected = {
        "backend": "sglang",
        "source": SM121_STORAGE_SOURCE,
        "revision": SM121_STORAGE_REVISION,
        "served_name": SM121_CHUNKED_PREFILL_PERFORMANCE_SERVED_NAME,
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
        "startup_timeout_s": SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S,
        "estimated_ram_gib": 101.0,
        "host_safety_min_memavailable_gib": 10,
        "host_safety_max_swap_growth_mib": 512,
        "host_safety_max_starting_swap_mib": 512,
        "endpoint": "http://127.0.0.1:30000/v1",
        "fetch_allow_patterns": (),
        "fetch_ignore_patterns": (),
        "sglang_storage_mode": SM121_STORAGE_MODE,
        "sglang_ple_nvme_queue_depth": SM121_STORAGE_QUEUE_DEPTH,
        "sglang_ple_nvme_max_batch_pages": SM121_STORAGE_MAX_BATCH_PAGES,
        "sglang_ple_nvme_cache_pages": SM121_STORAGE_CACHE_PAGES,
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
    sequence_fields = {
        "tasks",
        "fetch_allow_patterns",
        "fetch_ignore_patterns",
        "sglang_source_overlays",
        "model_shards",
    }
    for field, wanted in expected.items():
        actual = _value(model, field)
        if field in sequence_fields and isinstance(actual, (list, tuple)):
            actual = tuple(actual)
        if actual != wanted:
            raise SM121ChunkedPrefillPerformanceError(
                f"{field} does not match the chunked-prefill performance contract"
            )
    request = _value(model, "request_body_json")
    try:
        parsed = json.loads(request) if isinstance(request, str) else None
    except json.JSONDecodeError as error:
        raise SM121ChunkedPrefillPerformanceError(
            "request_body_json is invalid"
        ) from error
    if parsed != {"chat_template_kwargs": {"enable_thinking": False}}:
        raise SM121ChunkedPrefillPerformanceError(
            "request_body_json is not no-thinking"
        )
    expected_profile = _expected_profile(profile_id)
    if _value(model, "description") != expected_profile["description"]:
        raise SM121ChunkedPrefillPerformanceError("profile description changed")
    args = _value(model, "args")
    if isinstance(args, (list, tuple)):
        args = tuple(args)
    if args != expected_profile["args"]:
        raise SM121ChunkedPrefillPerformanceError("profile args changed")


def validate_sm121_chunked_prefill_performance_pair(
    control: Any, candidate: Any
) -> None:
    """Require the cache-on 1K/2K pair with no other configuration delta."""

    if _profile_id(control) != SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID:
        raise SM121ChunkedPrefillPerformanceError("control profile is invalid")
    if _profile_id(candidate) != SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID:
        raise SM121ChunkedPrefillPerformanceError("candidate profile is invalid")
    validate_sm121_chunked_prefill_performance_candidate(control)
    validate_sm121_chunked_prefill_performance_candidate(candidate)
    control_args = list(_value(control, "args") or ())
    candidate_args = list(_value(candidate, "args") or ())
    for arguments in (control_args, candidate_args):
        if arguments.count("--chunked-prefill-size") != 1:
            raise SM121ChunkedPrefillPerformanceError(
                "chunked-prefill argument is invalid"
            )
        index = arguments.index("--chunked-prefill-size")
        arguments[index + 1] = "normalized"
        served_index = arguments.index("--served-model-name")
        arguments[served_index + 1] = "normalized"
    if control_args != candidate_args:
        raise SM121ChunkedPrefillPerformanceError(
            "profiles differ beyond chunked-prefill size"
        )


def validate_sm121_chunked_prefill_performance_suite(suite: Any) -> None:
    """Require the small quality gate plus one controller-owned 60K proxy."""

    if _value(suite, "id") != SM121_CHUNKED_PREFILL_PERFORMANCE_SUITE_ID:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill suite ID is invalid")
    if _value(suite, "description") != SM121_CHUNKED_PREFILL_PERFORMANCE_SUITE_DESCRIPTION:
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill suite description changed"
        )
    if _value(suite, "protocol_digest") is not None:
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill suite digest is invalid"
        )
    cases = _value(suite, "cases")
    if not isinstance(cases, (list, tuple)) or len(cases) != 2:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill suite cases are invalid")
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
            "id": SM121_CHUNKED_PREFILL_PERFORMANCE_CASE_ID,
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
                raise SM121ChunkedPrefillPerformanceError(
                    f"chunked-prefill suite field {field} changed"
                )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _require_exact_keys(
    value: object, expected: frozenset[str], name: str
) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise SM121ChunkedPrefillPerformanceError(f"{name} fields are invalid")
    return value


def _event_fields(
    value: object, expected: frozenset[str], name: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise SM121ChunkedPrefillPerformanceError(f"{name} fields are invalid")
    actual = set(value)
    if actual != set(expected) and actual != set(expected) | {"timestamp"}:
        raise SM121ChunkedPrefillPerformanceError(f"{name} fields are invalid")
    if "timestamp" in value and not isinstance(value["timestamp"], str):
        raise SM121ChunkedPrefillPerformanceError(f"{name} timestamp is invalid")
    return value


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise SM121ChunkedPrefillPerformanceError(f"{name} must be boolean")
    return bool(value)


def _require_int(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < 0 or (positive and value == 0):
        raise SM121ChunkedPrefillPerformanceError(
            f"{name} must be a non-negative integer"
        )
    return value


def _require_optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, name)


def _require_finite(value: object, name: str, *, positive: bool = False) -> Decimal:
    if type(value) not in {int, float}:
        raise SM121ChunkedPrefillPerformanceError(f"{name} must be numeric")
    try:
        parsed = Decimal(str(value))
    except (DecimalException, ValueError) as error:
        raise SM121ChunkedPrefillPerformanceError(f"{name} must be finite") from error
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise SM121ChunkedPrefillPerformanceError(f"{name} must be finite")
    return parsed


def _require_arm(value: object, name: str) -> str:
    if value not in {
        SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_ARM,
        SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_ARM,
    }:
        raise SM121ChunkedPrefillPerformanceError(f"{name} is invalid")
    return str(value)


def sm121_chunked_prefill_performance_pair_instance_sha256(
    nonces: Sequence[object],
) -> str:
    if len(nonces) != len(SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER):
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill nonce count is invalid"
        )
    parsed: list[str] = []
    for nonce in nonces:
        if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
            raise SM121ChunkedPrefillPerformanceError(
                "chunked-prefill plan nonce is invalid"
            )
        parsed.append(nonce)
    return "sha256:" + hashlib.sha256(
        _canonical({"domain": "sm121-chunked-prefill-performance-v1", "nonces": parsed})
    ).hexdigest()


def sm121_chunked_prefill_performance_pair_binding_sha256(
    binding: Mapping[str, object],
) -> str:
    row = dict(binding)
    row.pop("pair_binding_sha256", None)
    return "sha256:" + hashlib.sha256(_canonical(row)).hexdigest()


def validate_sm121_chunked_prefill_performance_pair_binding(
    binding: object,
) -> None:
    row = _require_exact_keys(binding, _PAIR_BINDING_FIELDS, "chunked-prefill binding")
    if row["schema_version"] != SM121_CHUNKED_PREFILL_PERFORMANCE_PAIR_BINDING_SCHEMA_VERSION:
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill binding schema changed"
        )
    if row["suite_id"] != SM121_CHUNKED_PREFILL_PERFORMANCE_SUITE_ID:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill binding suite changed")
    if row["execution_mode"] != SM121_CHUNKED_PREFILL_PERFORMANCE_EXECUTION_MODE:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill binding mode changed")
    if row["arm_order"] != list(SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER):
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill arm order changed")
    if row["profile_ids"] != [
        SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID,
        SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID,
    ]:
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill profile binding changed"
        )
    if row["chunked_prefill_sizes"] != [
        SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_CHUNK_SIZE,
        SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_CHUNK_SIZE,
    ]:
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill size binding changed"
        )
    if (
        row["quality_case_id"] != SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID
        or row["timed_case_id"] != SM121_CHUNKED_PREFILL_PERFORMANCE_CASE_ID
        or row["cell_timeout_s"] != SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S
    ):
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill workload binding changed"
        )
    if not isinstance(row["campaign_instance_sha256"], str) or not _SHA256.fullmatch(
        row["campaign_instance_sha256"]
    ):
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill instance binding is invalid"
        )
    fingerprints = row["plan_fingerprints"]
    if (
        not isinstance(fingerprints, list)
        or len(fingerprints) != len(SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER)
        or any(not isinstance(value, str) or not _FINGERPRINT.fullmatch(value) for value in fingerprints)
    ):
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill plan binding is invalid"
        )
    if row["pair_binding_sha256"] != sm121_chunked_prefill_performance_pair_binding_sha256(row):
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill binding digest changed"
        )


def _arm_chunk_size(arm: str) -> int:
    if arm == SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_ARM:
        return SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_CHUNK_SIZE
    if arm == SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_ARM:
        return SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_CHUNK_SIZE
    raise SM121ChunkedPrefillPerformanceError("chunked-prefill arm is invalid")


def validate_sm121_chunked_prefill_performance_static_event(event: object) -> None:
    fields = frozenset(
        {
            "event",
            "arm",
            "lifetime_ordinal",
            "candidate_source_tree",
            "chunked_prefill_size",
            *SM121_CACHE_SOURCE_DIGESTS,
            *SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
        }
    )
    row = _event_fields(event, fields, "chunked-prefill static event")
    if row["event"] != SM121_CHUNKED_PREFILL_PERFORMANCE_STATIC_EVENT:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill static event changed")
    arm = _require_arm(row["arm"], "chunked-prefill static arm")
    ordinal = _require_int(
        row["lifetime_ordinal"], "chunked-prefill static ordinal", positive=True
    )
    if SM121_CHUNKED_PREFILL_PERFORMANCE_LIFETIME_ARMS.get(ordinal) != arm:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill static ordinal is invalid")
    if row["candidate_source_tree"] != SM121_STORAGE_SOURCE_TREE:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill source tree changed")
    if row["chunked_prefill_size"] != _arm_chunk_size(arm):
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill static size changed")
    for field, wanted in SM121_CACHE_SOURCE_DIGESTS.items():
        if row[field] != wanted:
            raise SM121ChunkedPrefillPerformanceError(
                f"chunked-prefill {field} changed"
            )
    for field, wanted in SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS.items():
        if row[field] is not wanted:
            raise SM121ChunkedPrefillPerformanceError(
                f"chunked-prefill {field} changed"
            )


def validate_sm121_chunked_prefill_performance_runtime_event(event: object) -> None:
    fields = frozenset(
        {
            "event",
            "arm",
            "lifetime_ordinal",
            "mamba_radix_cache_strategy",
            "max_mamba_cache_size",
            "chunked_prefill_size",
            *SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EXPECTED,
        }
    )
    row = _event_fields(event, fields, "chunked-prefill runtime event")
    if row["event"] != SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EVENT:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill runtime event changed")
    arm = _require_arm(row["arm"], "chunked-prefill runtime arm")
    ordinal = _require_int(
        row["lifetime_ordinal"], "chunked-prefill runtime ordinal", positive=True
    )
    if SM121_CHUNKED_PREFILL_PERFORMANCE_LIFETIME_ARMS.get(ordinal) != arm:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill runtime ordinal is invalid")
    if row["mamba_radix_cache_strategy"] != "extra_buffer_lazy":
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill cache strategy changed")
    if row["max_mamba_cache_size"] != SM121_CHUNKED_PREFILL_PERFORMANCE_MAX_MAMBA_CACHE_SIZE:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill Mamba size changed")
    if row["chunked_prefill_size"] != _arm_chunk_size(arm):
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill runtime size changed")
    for field, wanted in SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EXPECTED.items():
        if row[field] != wanted:
            raise SM121ChunkedPrefillPerformanceError(
                f"chunked-prefill {field} changed"
            )


def _optional_detail_state(
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
        raise SM121ChunkedPrefillPerformanceError(f"{name} detail state is invalid")
    if state in {"omitted", "null", "unexpected"}:
        if any(value is not None for value in values):
            raise SM121ChunkedPrefillPerformanceError(
                f"{name} omitted details are not null"
            )
    elif state == "zero_details":
        if values != (0, 0, 0):
            raise SM121ChunkedPrefillPerformanceError(
                f"{name} zero details are invalid"
            )
    elif (
        any(value is None for value in values)
        or values[0] is None
        or values[0] <= 0
        or values[1:] != (0, 0)
    ):
        raise SM121ChunkedPrefillPerformanceError(
            f"{name} nonzero details are invalid"
        )


def _turn_issues(row: Mapping[str, object]) -> tuple[str, ...]:
    """Return the one public admission basis for a 60K request observation."""

    issues: list[str] = []
    turn = str(row["turn"])
    if row["metrics_available"] is not True or row["guardrail_metrics_available"] is not True:
        issues.append("metrics_unavailable")
    if (
        row["metrics_before_settled"] is not True
        or row["metrics_after_settled"] is not True
        or int(row["metrics_before_polls"]) < 2
        or int(row["metrics_after_polls"]) < 2
    ):
        issues.append("metrics_unsettled")
    if row["append_only_prompt_identity_verified"] is not True:
        issues.append("append_identity")
    if row["cross_lifetime_prompt_identity_verified"] is not True:
        issues.append("cross_lifetime_identity")
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
    response_device = row["response_device_cached_tokens"]
    native_device = int(row["delta_prefill_device_hit_tokens"])
    native_cached = int(row["delta_cached_device_tokens"])
    if turn == "T0":
        # SGLang may issue a short bootstrap prefill while bringing a fresh
        # server to its ready state.  ``prefill_input_tokens`` is a global
        # cumulative counter, so it is diagnostic rather than proof that the
        # controller's first measured request inherited a cache entry.  A
        # cache-cold T0 is instead authenticated by the zero hit/residency
        # counters below, alongside the fresh-lifetime controller boundary.
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
        if any(
            value not in {None, 0}
            for value in (
                response_device,
                row["response_host_cached_tokens"],
                row["response_storage_cached_tokens"],
                row["usage_cached_tokens"],
            )
        ) or (
            native_device != 0
            or native_cached != 0
            or int(row["delta_cached_total_tokens"]) != 0
            or int(row["after_cached_total_tokens"]) != 0
        ):
            issues.append("cold_hit")
    elif (
        row["response_detail_state"] != "nonzero_details"
        or not isinstance(response_device, int)
        or response_device <= 0
        or response_device > int(row["shared_prefix_tokens"])
        or response_device != native_device
        or response_device != native_cached
        or (
            row["usage_cached_tokens"] is not None
            and row["usage_cached_tokens"] != response_device
        )
    ):
        issues.append("device_hit_reconciliation")
    return tuple(issues)


def derive_sm121_chunked_prefill_performance_turn_admission(
    event: object,
) -> tuple[bool, str]:
    row = _event_fields(event, _TURN_EVENT_FIELDS, "chunked-prefill turn event")
    issues = _turn_issues(row)
    return not issues, "admitted" if not issues else issues[0]


def validate_sm121_chunked_prefill_performance_turn_event(event: object) -> None:
    row = _event_fields(event, _TURN_EVENT_FIELDS, "chunked-prefill turn event")
    if row["event"] != SM121_CHUNKED_PREFILL_PERFORMANCE_TURN_EVENT:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill turn event changed")
    arm = _require_arm(row["arm"], "chunked-prefill turn arm")
    ordinal = _require_int(
        row["lifetime_ordinal"], "chunked-prefill turn ordinal", positive=True
    )
    if ordinal not in {2, 4, 6, 8}:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill turn must be timed")
    if SM121_CHUNKED_PREFILL_PERFORMANCE_LIFETIME_ARMS.get(ordinal) != arm:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill turn arm is invalid")
    if row["protocol_case_id"] != SM121_CHUNKED_PREFILL_PERFORMANCE_CASE_ID:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill case ID changed")
    case_id = row["case_id"]
    if not isinstance(case_id, str) or re.fullmatch(
        rf"{re.escape(SM121_CHUNKED_PREFILL_PERFORMANCE_CASE_ID)}--[0-9a-f]{{12}}",
        case_id,
    ) is None:
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill case identifier is invalid"
        )
    turn = row["turn"]
    if turn not in SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill turn is invalid")
    for field, wanted in (
        ("cache_details_requested", True),
        ("prompt_token_ids_requested", True),
        ("streaming", False),
        ("thinking_disabled", True),
    ):
        if row[field] is not wanted:
            raise SM121ChunkedPrefillPerformanceError(
                f"chunked-prefill {field} changed"
            )
    prompt = _require_int(
        row["prompt_tokens"], "chunked-prefill prompt tokens", positive=True
    )
    _require_int(
        row["completion_tokens"], "chunked-prefill completion tokens", positive=True
    )
    if _require_int(row["reasoning_tokens"], "chunked-prefill reasoning tokens") != 0:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill reasoning is not disabled")
    shared = _require_int(
        row["shared_prefix_tokens"], "chunked-prefill shared prefix"
    )
    if turn == "T0":
        if shared != 0 or not (
            SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MIN_TOKENS
            <= prompt
            <= SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MAX_TOKENS
        ):
            raise SM121ChunkedPrefillPerformanceError("chunked-prefill T0 shape is invalid")
    elif not (
        SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MIN_TOKENS <= shared < prompt
    ):
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill shared prefix is invalid"
        )
    _require_bool(
        row["append_only_prompt_identity_verified"], "chunked-prefill append identity"
    )
    _require_bool(
        row["cross_lifetime_prompt_identity_verified"], "chunked-prefill cross identity"
    )
    detail_values = tuple(
        _require_optional_int(row[field], field)
        for field in (
            "response_device_cached_tokens",
            "response_host_cached_tokens",
            "response_storage_cached_tokens",
        )
    )
    _optional_detail_state(row["response_detail_state"], detail_values, name="response")
    usage = _require_optional_int(row["usage_cached_tokens"], "chunked-prefill usage")
    if row["usage_detail_state"] not in {
        "omitted",
        "null",
        "zero_details",
        "nonzero_details",
        "unexpected",
    }:
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill usage detail state is invalid"
        )
    if row["usage_detail_state"] in {"omitted", "null", "unexpected"} and usage is not None:
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill usage details are invalid"
        )
    if row["usage_detail_state"] == "zero_details" and usage != 0:
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill zero usage is invalid")
    if row["usage_detail_state"] == "nonzero_details" and (usage is None or usage <= 0):
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill nonzero usage is invalid"
        )
    for field in (
        "metrics_available",
        "guardrail_metrics_available",
        "metrics_before_settled",
        "metrics_after_settled",
    ):
        _require_bool(row[field], f"chunked-prefill {field}")
    _require_int(
        row["metrics_before_polls"], "chunked-prefill metrics-before polls"
    )
    _require_int(
        row["metrics_after_polls"], "chunked-prefill metrics-after polls"
    )
    _require_finite(row["request_wall_s"], "chunked-prefill request wall", positive=True)
    for prefix in ("before", "after", "delta"):
        for metric in SM121_CHUNKED_PREFILL_PERFORMANCE_METRIC_FIELDS:
            value = row[f"{prefix}_{metric}"]
            if prefix == "delta" and metric.endswith(
                ("available_tokens", "evictable_tokens", "used_tokens")
            ):
                if type(value) is not int:
                    raise SM121ChunkedPrefillPerformanceError(
                        "chunked-prefill gauge delta is invalid"
                    )
            else:
                _require_int(value, f"chunked-prefill {prefix} {metric}")
            if row[f"delta_{metric}"] != row[f"after_{metric}"] - row[f"before_{metric}"]:
                raise SM121ChunkedPrefillPerformanceError(
                    "chunked-prefill metric delta changed"
                )
    for prefix in ("before", "after"):
        for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
            _require_bool(
                row[f"{prefix}_cached_{source}_series_present"],
                "chunked-prefill cache series marker",
            )
    admitted, basis = derive_sm121_chunked_prefill_performance_turn_admission(row)
    if row["timed_turn_admitted"] is not admitted or row["timed_turn_basis"] != basis:
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill turn admission changed"
        )
    if turn != "T0" and detail_values[0] is None:
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill device details are absent"
        )


def _is_legacy_bootstrap_counter_partial_turn(event: object) -> bool:
    """Recognize only the pre-correction terminal T0 observation.

    The first live v1 campaign was already terminal when the ready-state
    bootstrap behavior was discovered.  It may be retained as an audited
    partial record, but this compatibility path is deliberately unavailable
    to the controller that creates new observations.
    """

    try:
        row = _event_fields(event, _TURN_EVENT_FIELDS, "chunked-prefill turn event")
    except SM121ChunkedPrefillPerformanceError:
        return False
    if (
        row["turn"] != "T0"
        or row["timed_turn_admitted"] is not False
        or row["timed_turn_basis"] != "cold_lifetime"
        or type(row["before_prefill_input_tokens"]) is not int
        or row["before_prefill_input_tokens"] <= 0
    ):
        return False
    normalized = dict(row)
    normalized["timed_turn_admitted"] = True
    normalized["timed_turn_basis"] = "admitted"
    try:
        validate_sm121_chunked_prefill_performance_turn_event(normalized)
    except SM121ChunkedPrefillPerformanceError:
        return False
    return True


def validate_sm121_chunked_prefill_performance_recorded_turn_event(event: object) -> None:
    """Validate a current observation or the one audited legacy partial T0."""

    try:
        validate_sm121_chunked_prefill_performance_turn_event(event)
    except SM121ChunkedPrefillPerformanceError:
        if not _is_legacy_bootstrap_counter_partial_turn(event):
            raise


def validate_sm121_chunked_prefill_performance_lifetimes(
    lifetimes: object,
) -> tuple[dict[str, object], ...]:
    """Validate an A/B/B/A public summary, including a terminal prefix."""

    if type(lifetimes) is not list or len(lifetimes) != len(
        SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER
    ):
        raise SM121ChunkedPrefillPerformanceError("chunked-prefill lifetimes are invalid")
    expected_fields = frozenset(
        {
            "ordinal",
            "arm",
            "quality_admitted",
            "timed_admitted",
            "within_timeout",
            "turns",
        }
    )
    rows: list[dict[str, object]] = []
    terminal = False
    for ordinal, (raw, expected_arm) in enumerate(
        zip(lifetimes, SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER, strict=True),
        start=1,
    ):
        row = _require_exact_keys(raw, expected_fields, "chunked-prefill lifetime")
        if row["ordinal"] != ordinal or row["arm"] != expected_arm:
            raise SM121ChunkedPrefillPerformanceError(
                "chunked-prefill lifetime order changed"
            )
        quality_admitted = _require_bool(
            row["quality_admitted"], "chunked-prefill quality admission"
        )
        timed_admitted = _require_bool(
            row["timed_admitted"], "chunked-prefill timed admission"
        )
        within_timeout = _require_bool(
            row["within_timeout"], "chunked-prefill timeout"
        )
        turns = row["turns"]
        if type(turns) is not list or len(turns) > len(
            SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS
        ):
            raise SM121ChunkedPrefillPerformanceError("chunked-prefill turns are invalid")
        for turn_index, (expected_turn, event) in enumerate(
            zip(SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS, turns, strict=False)
        ):
            if type(event) is not dict or "timestamp" in event:
                raise SM121ChunkedPrefillPerformanceError(
                    "chunked-prefill public turn is invalid"
                )
            validate_sm121_chunked_prefill_performance_recorded_turn_event(event)
            if (
                event["arm"] != expected_arm
                or event["lifetime_ordinal"] != ordinal * 2
                or event["turn"] != expected_turn
            ):
                raise SM121ChunkedPrefillPerformanceError(
                    "chunked-prefill turn topology changed"
                )
            if (
                turn_index + 1 < len(turns)
                and event["timed_turn_admitted"] is not True
            ):
                raise SM121ChunkedPrefillPerformanceError(
                    "chunked-prefill failed turn is not terminal"
                )
        if not quality_admitted and (timed_admitted or turns):
            raise SM121ChunkedPrefillPerformanceError(
                "chunked-prefill quality failure retained timings"
            )
        if timed_admitted and (
            not quality_admitted
            or len(turns) != len(SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS)
            or any(event["timed_turn_admitted"] is not True for event in turns)
        ):
            raise SM121ChunkedPrefillPerformanceError(
                "chunked-prefill timed admission is inconsistent"
            )
        admitted = quality_admitted and timed_admitted and within_timeout
        if terminal:
            if quality_admitted or timed_admitted or within_timeout or turns:
                raise SM121ChunkedPrefillPerformanceError(
                    "chunked-prefill continued after terminal lifetime"
                )
        elif not admitted:
            terminal = True
        rows.append(row)
    return tuple(rows)


def score_sm121_chunked_prefill_performance_campaign(
    lifetimes: object,
) -> ChunkedPrefillPerformanceScore:
    """Reduce two replicas per arm against the frozen 60K decision rule."""

    rows = validate_sm121_chunked_prefill_performance_lifetimes(lifetimes)
    by_arm: dict[str, list[tuple[Decimal, Decimal, Decimal]]] = {"A": [], "B": []}
    for row, arm in zip(
        rows, SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER, strict=True
    ):
        admitted = (
            row["quality_admitted"] is True
            and row["timed_admitted"] is True
            and row["within_timeout"] is True
        )
        if not admitted:
            return ChunkedPrefillPerformanceScore(
                "partial",
                "not_evaluated",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        turns = row["turns"]
        assert isinstance(turns, list)
        parsed: list[Mapping[str, object]] = []
        for expected_turn, event in zip(
            SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS, turns, strict=True
        ):
            assert isinstance(event, Mapping)
            if event["turn"] != expected_turn or event["timed_turn_admitted"] is not True:
                raise SM121ChunkedPrefillPerformanceError(
                    "chunked-prefill timing admission changed"
                )
            parsed.append(event)
        walls = [
            _require_finite(event["request_wall_s"], "chunked-prefill request wall", positive=True)
            for event in parsed
        ]
        by_arm[arm].append((walls[0], walls[1] + walls[2], sum(walls, Decimal(0))))
    if any(len(values) != 2 for values in by_arm.values()):
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill replication is incomplete"
        )

    def mean(index: int, arm: str) -> Decimal:
        return sum((value[index] for value in by_arm[arm]), Decimal(0)) / Decimal(2)

    a_t0, b_t0 = mean(0, "A"), mean(0, "B")
    a_later, b_later = mean(1, "A"), mean(1, "B")
    a_full, b_full = mean(2, "A"), mean(2, "B")
    if any(value <= 0 for value in (a_t0, b_t0, a_later, b_later, a_full, b_full)):
        raise SM121ChunkedPrefillPerformanceError(
            "chunked-prefill timing aggregate is invalid"
        )
    t0_ratio = b_t0 / a_t0
    later_ratio = b_later / a_later
    full_ratio = b_full / a_full
    if t0_ratio <= SM121_CHUNKED_PREFILL_PERFORMANCE_PROMOTION_RATIO:
        if (
            later_ratio <= SM121_CHUNKED_PREFILL_PERFORMANCE_APPEND_WALL_GUARDRAIL_RATIO
            and full_ratio <= SM121_CHUNKED_PREFILL_PERFORMANCE_FULL_WALL_GUARDRAIL_RATIO
        ):
            decision = "retain_b"
        else:
            decision = "guardrail_reject"
    else:
        decision = "no_retention"

    def number(value: Decimal) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise SM121ChunkedPrefillPerformanceError(
                "chunked-prefill aggregate is non-finite"
            )
        return result

    return ChunkedPrefillPerformanceScore(
        "complete",
        decision,
        number(a_t0),
        number(b_t0),
        number(a_later),
        number(b_later),
        number(a_full),
        number(b_full),
        number(t0_ratio),
        number(later_ratio),
        number(full_ratio),
    )
