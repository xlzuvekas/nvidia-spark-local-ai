"""Immutable contract for the SM121 cache-policy wall-time campaign.

The paired semantic canary intentionally excludes timings.  This sibling lane
does not loosen that canary: it freezes its own profiles, workload identity,
fresh-lifetime A/B/B/A topology, and scalar-only timing reducer.  Prompt text,
responses, prompt token IDs, and request identifiers must remain transient.
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
    SM121_STORAGE_QUEUE_DEPTH,
    SM121_STORAGE_REVISION,
    SM121_STORAGE_SOURCE,
    SM121_STORAGE_SOURCE_TREE,
    SM121_STORAGE_WEIGHT_FILE_COUNT,
    SM121_STORAGE_WEIGHT_SIZE_BYTES,
)


SM121_CACHE_PERFORMANCE_SUITE_ID = (
    "qwen38-flash-next-sm121-triton-storage-cache-policy-performance-v1"
)
SM121_CACHE_PERFORMANCE_CAMPAIGN_ID = (
    "qwen38-flash-next-sm121-cache-policy-performance-v1"
)
SM121_CACHE_PERFORMANCE_EXECUTION_MODE = (
    "sm121_storage_cache_policy_performance_abba_fresh_lifetimes"
)
SM121_CACHE_PERFORMANCE_SCHEMA_VERSION = 1
SM121_CACHE_PERFORMANCE_PAIR_BINDING_SCHEMA_VERSION = 1
SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID = (
    "qwen38-flash-next-nvfp4-sm121-triton-storage-cache-performance-on-sglang"
)
SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID = (
    "qwen38-flash-next-nvfp4-sm121-triton-storage-cache-performance-off-sglang"
)
SM121_CACHE_PERFORMANCE_PROFILE_IDS = frozenset(
    {
        SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID,
        SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID,
    }
)
SM121_CACHE_PERFORMANCE_CACHE_ON_ARM = "A"
SM121_CACHE_PERFORMANCE_CACHE_OFF_ARM = "B"
SM121_CACHE_PERFORMANCE_ARM_ORDER = ("A", "B", "B", "A")
SM121_CACHE_PERFORMANCE_LIFETIME_ARMS = {
    1: "A",
    2: "A",
    3: "B",
    4: "B",
    5: "B",
    6: "B",
    7: "A",
    8: "A",
}
SM121_CACHE_PERFORMANCE_TIMED_TURNS = ("T0", "T1", "T2")
SM121_CACHE_PERFORMANCE_QUALITY_CASE_ID = "synthetic-exact-answer-v2"
SM121_CACHE_PERFORMANCE_QUALITY_ITEM_COUNT = 4
SM121_CACHE_PERFORMANCE_CASE_ID = (
    "sm121-cache-policy-shared-prefix-performance-v1"
)
SM121_CACHE_PERFORMANCE_SERVED_NAME = (
    "qwen38-flash-next-nvfp4-sm121-storage-cache-policy-performance"
)
SM121_CACHE_PERFORMANCE_MAX_MAMBA_CACHE_SIZE = 4
SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S = 1_200
SM121_CACHE_PERFORMANCE_COLD_INPUT_MIN_TOKENS = 32 * 1024
SM121_CACHE_PERFORMANCE_COLD_INPUT_MAX_TOKENS = 48 * 1024
SM121_CACHE_PERFORMANCE_PROMOTION_RATIO = Decimal("0.95")
SM121_CACHE_PERFORMANCE_FULL_WALL_GUARDRAIL_RATIO = Decimal("1.05")
SM121_CACHE_PERFORMANCE_STATIC_EVENT = "sm121_cache_performance_static_attestation"
SM121_CACHE_PERFORMANCE_RUNTIME_EVENT = "sm121_cache_performance_runtime_attestation"
SM121_CACHE_PERFORMANCE_TURN_EVENT = "sm121_cache_performance_turn_observation"
# These are the verified scalar bundle commitments for the target admission,
# clean B0 observation, and final complete B/A semantic pair. They bind a
# performance campaign to the already-published capability evidence without
# retaining source run paths, prompts, or request identities.
SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S = (
    "3e0238661eb20ddc66c6a32bc2c3ac35f5a3f184017f07364225a0b4cd7083d0",
    "b1017ce93f547c4dbba8884bd15b14cd4d65a7224ceaf9cfc1c819c8fd21077f",
    "82a6fcc2895212b53f46486bae2edd9d32d154b6813e665c6a994c2b283afa37",
    "f40f58fffeaf28a860087bb42e0148ce25f0efdc301a14e2cc83e98b9489aba2",
)

SM121_CACHE_PERFORMANCE_SUITE_DESCRIPTION = (
    "Frozen fresh-lifetime A/B/B/A wall-time study of the exact SM121 native-"
    "NVMe Qwen3.8 Flash-Next cache-policy bundle. Each arm has a separate "
    "quality lifetime and a cold T0 plus append-only T1/T2 timing lifetime; "
    "only the dedicated controller may execute it."
)
SM121_CACHE_PERFORMANCE_CACHE_ON_DESCRIPTION = (
    "SM121 cache-policy performance A: UnifiedRadixCache with lazy Mamba "
    "state. Only the dedicated A/B/B/A performance controller may execute it."
)
SM121_CACHE_PERFORMANCE_CACHE_OFF_DESCRIPTION = (
    "SM121 cache-policy performance B: ChunkCache cache-off control. Only the "
    "dedicated A/B/B/A performance controller may execute it."
)

SM121_CACHE_PERFORMANCE_COMMON_ARGS = (
    "--served-model-name",
    SM121_CACHE_PERFORMANCE_SERVED_NAME,
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
    str(SM121_CACHE_PERFORMANCE_MAX_MAMBA_CACHE_SIZE),
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
_CACHE_OFF_INSERTION_INDEX = SM121_CACHE_PERFORMANCE_COMMON_ARGS.index("--page-size")
SM121_CACHE_PERFORMANCE_CACHE_OFF_ARGS = (
    SM121_CACHE_PERFORMANCE_COMMON_ARGS[:_CACHE_OFF_INSERTION_INDEX]
    + ("--disable-radix-cache",)
    + SM121_CACHE_PERFORMANCE_COMMON_ARGS[_CACHE_OFF_INSERTION_INDEX:]
)
SM121_CACHE_PERFORMANCE_CACHE_ON_ARGS = SM121_CACHE_PERFORMANCE_COMMON_ARGS

SM121_CACHE_PERFORMANCE_RUNTIME_EXPECTED = {
    "A": SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED["A"],
    "B": SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED["B"],
}
SM121_CACHE_PERFORMANCE_METRIC_FIELDS = (
    *SM121_CACHE_OBSERVABILITY_METRIC_FIELDS,
    *SM121_CACHE_SEMANTIC_GUARDRAIL_METRIC_FIELDS,
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FINGERPRINT = re.compile(r"[0-9a-f]{16}\Z")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
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
            for metric in SM121_CACHE_PERFORMANCE_METRIC_FIELDS
        ),
        *(
            f"{prefix}_cached_{source}_series_present"
            for prefix in ("before", "after")
            for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES
        ),
    }
)
_PAIR_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "suite_id",
        "execution_mode",
        "arm_order",
        "profile_ids",
        "quality_case_id",
        "timed_case_id",
        "cell_timeout_s",
        "campaign_instance_sha256",
        "plan_fingerprints",
        "pair_binding_sha256",
    }
)


class SM121CachePerformanceError(ValueError):
    """Raised when the exact cache-performance contract is not met."""


@dataclass(frozen=True, slots=True)
class CachePerformanceScore:
    status: str
    decision: str
    a_later_wall_s: float | None
    b_later_wall_s: float | None
    a_full_wall_s: float | None
    b_full_wall_s: float | None
    winner_later_wall_ratio: float | None
    winner_full_wall_ratio: float | None

    def to_mapping(self) -> dict[str, object]:
        return {
            "status": self.status,
            "decision": self.decision,
            "a_later_wall_s": self.a_later_wall_s,
            "b_later_wall_s": self.b_later_wall_s,
            "a_full_wall_s": self.a_full_wall_s,
            "b_full_wall_s": self.b_full_wall_s,
            "winner_later_wall_ratio": self.winner_later_wall_ratio,
            "winner_full_wall_ratio": self.winner_full_wall_ratio,
        }


def _value(item: Any, field: str) -> object:
    return item.get(field) if isinstance(item, Mapping) else getattr(item, field, None)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _require_exact_keys(value: object, expected: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise SM121CachePerformanceError(f"{name} fields are invalid")
    return value


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise SM121CachePerformanceError(f"{name} must be boolean")
    return bool(value)


def _require_int(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < 0 or (positive and value == 0):
        raise SM121CachePerformanceError(f"{name} must be a non-negative integer")
    return value


def _require_optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, name)


def _require_finite(value: object, name: str, *, positive: bool = False) -> Decimal:
    if type(value) not in {int, float}:
        raise SM121CachePerformanceError(f"{name} must be numeric")
    try:
        parsed = Decimal(str(value))
    except (DecimalException, ValueError) as error:
        raise SM121CachePerformanceError(f"{name} must be finite") from error
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise SM121CachePerformanceError(f"{name} must be finite")
    return parsed


def _require_arm(value: object, name: str) -> str:
    if value not in {"A", "B"}:
        raise SM121CachePerformanceError(f"{name} is invalid")
    return str(value)


def _profile_id(value: Any) -> str | None:
    candidate = value if isinstance(value, str) else _value(value, "id")
    return candidate if isinstance(candidate, str) else None


def sm121_cache_performance_arm(value: Any) -> str:
    profile_id = _profile_id(value)
    if profile_id == SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID:
        return "A"
    if profile_id == SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID:
        return "B"
    raise SM121CachePerformanceError("profile is not a cache-performance arm")


def is_sm121_cache_performance_candidate(model: Any) -> bool:
    return _profile_id(model) in SM121_CACHE_PERFORMANCE_PROFILE_IDS


def is_sm121_cache_performance_plan(model: Any, suite: Any) -> bool:
    return (
        is_sm121_cache_performance_candidate(model)
        and _value(suite, "id") == SM121_CACHE_PERFORMANCE_SUITE_ID
    )


def _expected_profile(profile_id: str) -> dict[str, object]:
    if profile_id == SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID:
        return {
            "description": SM121_CACHE_PERFORMANCE_CACHE_ON_DESCRIPTION,
            "args": SM121_CACHE_PERFORMANCE_CACHE_ON_ARGS,
        }
    if profile_id == SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID:
        return {
            "description": SM121_CACHE_PERFORMANCE_CACHE_OFF_DESCRIPTION,
            "args": SM121_CACHE_PERFORMANCE_CACHE_OFF_ARGS,
        }
    raise SM121CachePerformanceError("profile is not a cache-performance arm")


def validate_sm121_cache_performance_candidate(model: Any) -> None:
    """Require one immutable sibling profile and no generic serving variant."""

    if not is_sm121_cache_performance_candidate(model):
        return
    profile_id = _profile_id(model)
    assert profile_id is not None
    expected = {
        "backend": "sglang",
        "source": SM121_STORAGE_SOURCE,
        "revision": SM121_STORAGE_REVISION,
        "served_name": SM121_CACHE_PERFORMANCE_SERVED_NAME,
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
        "startup_timeout_s": SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S,
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
    for field, wanted in expected.items():
        actual = _value(model, field)
        if field in {
            "tasks",
            "fetch_allow_patterns",
            "fetch_ignore_patterns",
            "sglang_source_overlays",
            "model_shards",
        } and isinstance(actual, (list, tuple)):
            actual = tuple(actual)
        if actual != wanted:
            raise SM121CachePerformanceError(
                f"{field} does not match the cache-performance contract"
            )
    request = _value(model, "request_body_json")
    try:
        parsed = json.loads(request) if isinstance(request, str) else None
    except json.JSONDecodeError as error:
        raise SM121CachePerformanceError("request_body_json is invalid") from error
    if parsed != {"chat_template_kwargs": {"enable_thinking": False}}:
        raise SM121CachePerformanceError("request_body_json is not no-thinking")
    profile = _expected_profile(profile_id)
    if _value(model, "description") != profile["description"]:
        raise SM121CachePerformanceError("profile description changed")
    args = _value(model, "args")
    if isinstance(args, (list, tuple)):
        args = tuple(args)
    if args != profile["args"]:
        raise SM121CachePerformanceError("profile args changed")


def validate_sm121_cache_performance_pair(cache_on: Any, cache_off: Any) -> None:
    if _profile_id(cache_on) != SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID:
        raise SM121CachePerformanceError("cache-on profile is invalid")
    if _profile_id(cache_off) != SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID:
        raise SM121CachePerformanceError("cache-off profile is invalid")
    validate_sm121_cache_performance_candidate(cache_on)
    validate_sm121_cache_performance_candidate(cache_off)


def validate_sm121_cache_performance_suite(suite: Any) -> None:
    if _value(suite, "id") != SM121_CACHE_PERFORMANCE_SUITE_ID:
        raise SM121CachePerformanceError("cache-performance suite ID is invalid")
    if _value(suite, "description") != SM121_CACHE_PERFORMANCE_SUITE_DESCRIPTION:
        raise SM121CachePerformanceError("cache-performance suite description changed")
    if _value(suite, "protocol_digest") is not None:
        raise SM121CachePerformanceError("cache-performance suite digest is invalid")
    cases = _value(suite, "cases")
    if not isinstance(cases, (list, tuple)) or len(cases) != 2:
        raise SM121CachePerformanceError("cache-performance suite cases are invalid")
    expected_cases = (
        {
            "id": SM121_CACHE_PERFORMANCE_QUALITY_CASE_ID,
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
            "id": SM121_CACHE_PERFORMANCE_CASE_ID,
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
                raise SM121CachePerformanceError(
                    f"cache-performance suite field {field} changed"
                )


def sm121_cache_performance_pair_instance_sha256(nonces: Sequence[object]) -> str:
    if len(nonces) != len(SM121_CACHE_PERFORMANCE_ARM_ORDER):
        raise SM121CachePerformanceError("cache-performance nonce count is invalid")
    parsed: list[str] = []
    for nonce in nonces:
        if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
            raise SM121CachePerformanceError("cache-performance plan nonce is invalid")
        parsed.append(nonce)
    return "sha256:" + hashlib.sha256(
        _canonical({"domain": "sm121-cache-performance-v1", "nonces": parsed})
    ).hexdigest()


def sm121_cache_performance_pair_binding_sha256(binding: Mapping[str, object]) -> str:
    row = dict(binding)
    row.pop("pair_binding_sha256", None)
    return "sha256:" + hashlib.sha256(_canonical(row)).hexdigest()


def validate_sm121_cache_performance_pair_binding(binding: object) -> None:
    row = _require_exact_keys(binding, _PAIR_BINDING_FIELDS, "cache-performance binding")
    if row["schema_version"] != SM121_CACHE_PERFORMANCE_PAIR_BINDING_SCHEMA_VERSION:
        raise SM121CachePerformanceError("cache-performance binding schema changed")
    if row["suite_id"] != SM121_CACHE_PERFORMANCE_SUITE_ID:
        raise SM121CachePerformanceError("cache-performance binding suite changed")
    if row["execution_mode"] != SM121_CACHE_PERFORMANCE_EXECUTION_MODE:
        raise SM121CachePerformanceError("cache-performance binding mode changed")
    if row["arm_order"] != list(SM121_CACHE_PERFORMANCE_ARM_ORDER):
        raise SM121CachePerformanceError("cache-performance arm order changed")
    if row["profile_ids"] != [
        SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID,
        SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID,
    ]:
        raise SM121CachePerformanceError("cache-performance profile binding changed")
    if (
        row["quality_case_id"] != SM121_CACHE_PERFORMANCE_QUALITY_CASE_ID
        or row["timed_case_id"] != SM121_CACHE_PERFORMANCE_CASE_ID
        or row["cell_timeout_s"] != SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S
    ):
        raise SM121CachePerformanceError("cache-performance workload binding changed")
    if not isinstance(row["campaign_instance_sha256"], str) or not _SHA256.fullmatch(
        row["campaign_instance_sha256"]
    ):
        raise SM121CachePerformanceError("cache-performance instance binding is invalid")
    fingerprints = row["plan_fingerprints"]
    if (
        not isinstance(fingerprints, list)
        or len(fingerprints) != len(SM121_CACHE_PERFORMANCE_ARM_ORDER)
        or any(not isinstance(value, str) or not _FINGERPRINT.fullmatch(value) for value in fingerprints)
    ):
        raise SM121CachePerformanceError("cache-performance plan binding is invalid")
    if row["pair_binding_sha256"] != sm121_cache_performance_pair_binding_sha256(row):
        raise SM121CachePerformanceError("cache-performance binding digest changed")


def _event_fields(value: object, expected: frozenset[str], name: str) -> dict[str, object]:
    actual = set(value) if type(value) is dict else set()
    if type(value) is not dict or actual not in (set(expected), set(expected) | {"timestamp"}):
        raise SM121CachePerformanceError(f"{name} fields are invalid")
    row = value
    if "timestamp" in row and not isinstance(row["timestamp"], str):
        raise SM121CachePerformanceError(f"{name} timestamp is invalid")
    return row


def validate_sm121_cache_performance_static_event(event: object) -> None:
    fields = frozenset(
        {
            "event",
            "arm",
            "lifetime_ordinal",
            "candidate_source_tree",
            *SM121_CACHE_SOURCE_DIGESTS,
            *SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
        }
    )
    row = _event_fields(event, fields, "cache-performance static event")
    if row["event"] != SM121_CACHE_PERFORMANCE_STATIC_EVENT:
        raise SM121CachePerformanceError("cache-performance static event changed")
    _require_arm(row["arm"], "cache-performance static arm")
    ordinal = _require_int(row["lifetime_ordinal"], "cache-performance static ordinal", positive=True)
    if SM121_CACHE_PERFORMANCE_LIFETIME_ARMS.get(ordinal) != row["arm"]:
        raise SM121CachePerformanceError("cache-performance static ordinal is invalid")
    if row["candidate_source_tree"] != SM121_STORAGE_SOURCE_TREE:
        raise SM121CachePerformanceError("cache-performance source tree changed")
    for field, wanted in SM121_CACHE_SOURCE_DIGESTS.items():
        if row[field] != wanted:
            raise SM121CachePerformanceError(f"cache-performance {field} changed")
    for field, wanted in SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS.items():
        if row[field] is not wanted:
            raise SM121CachePerformanceError(f"cache-performance {field} changed")


def validate_sm121_cache_performance_runtime_event(event: object) -> None:
    runtime_fields = frozenset(next(iter(SM121_CACHE_PERFORMANCE_RUNTIME_EXPECTED.values())))
    fields = frozenset(
        {
            "event",
            "arm",
            "lifetime_ordinal",
            "mamba_radix_cache_strategy",
            "max_mamba_cache_size",
            *runtime_fields,
        }
    )
    row = _event_fields(event, fields, "cache-performance runtime event")
    if row["event"] != SM121_CACHE_PERFORMANCE_RUNTIME_EVENT:
        raise SM121CachePerformanceError("cache-performance runtime event changed")
    arm = _require_arm(row["arm"], "cache-performance runtime arm")
    ordinal = _require_int(
        row["lifetime_ordinal"], "cache-performance runtime ordinal", positive=True
    )
    if SM121_CACHE_PERFORMANCE_LIFETIME_ARMS.get(ordinal) != arm:
        raise SM121CachePerformanceError("cache-performance runtime ordinal is invalid")
    if row["mamba_radix_cache_strategy"] != "extra_buffer_lazy":
        raise SM121CachePerformanceError("cache-performance cache strategy changed")
    if row["max_mamba_cache_size"] != SM121_CACHE_PERFORMANCE_MAX_MAMBA_CACHE_SIZE:
        raise SM121CachePerformanceError("cache-performance Mamba size changed")
    for field, wanted in SM121_CACHE_PERFORMANCE_RUNTIME_EXPECTED[arm].items():
        if row[field] != wanted:
            raise SM121CachePerformanceError(f"cache-performance {field} changed")


def _optional_detail_state(
    state: object,
    values: tuple[int | None, int | None, int | None],
    *,
    name: str,
) -> None:
    if state not in {"omitted", "null", "zero_details", "nonzero_details"}:
        raise SM121CachePerformanceError(f"{name} detail state is invalid")
    if state in {"omitted", "null"}:
        if any(value is not None for value in values):
            raise SM121CachePerformanceError(f"{name} omitted details are not null")
    elif state == "zero_details":
        if values != (0, 0, 0):
            raise SM121CachePerformanceError(f"{name} zero details are invalid")
    elif (
        any(value is None for value in values)
        or values[0] is None
        or values[0] <= 0
        or values[1:] != (0, 0)
    ):
        raise SM121CachePerformanceError(f"{name} nonzero details are invalid")


def _turn_issues(row: Mapping[str, object]) -> tuple[str, ...]:
    issues: list[str] = []
    arm = str(row["arm"])
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
        if any(int(row[f"{prefix}_{metric}"]) != 0 for prefix in ("before", "after", "delta")):
            issues.append("cache_guardrail")
            break
    response_device = row["response_device_cached_tokens"]
    native_device = int(row["delta_prefill_device_hit_tokens"])
    native_cached = int(row["delta_cached_device_tokens"])
    zero_required = arm == "B" or turn == "T0"
    if turn == "T0" and any(
        int(row[f"before_{metric}"]) != 0
        for metric in (
            "prefill_input_tokens",
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
    if zero_required:
        if any(value not in {None, 0} for value in (
            response_device,
            row["response_host_cached_tokens"],
            row["response_storage_cached_tokens"],
            row["usage_cached_tokens"],
        )) or (
            native_device != 0
            or native_cached != 0
            or int(row["delta_cached_total_tokens"]) != 0
            or int(row["after_cached_total_tokens"]) != 0
        ):
            issues.append("zero_hit")
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


def derive_sm121_cache_performance_turn_admission(event: object) -> tuple[bool, str]:
    """Return whether a validated turn is a legitimate timing observation."""

    row = _event_fields(event, _TURN_EVENT_FIELDS, "cache-performance turn event")
    issues = _turn_issues(row)
    return (not issues, "admitted" if not issues else issues[0])


def validate_sm121_cache_performance_turn_event(event: object) -> None:
    row = _event_fields(event, _TURN_EVENT_FIELDS, "cache-performance turn event")
    if row["event"] != SM121_CACHE_PERFORMANCE_TURN_EVENT:
        raise SM121CachePerformanceError("cache-performance turn event changed")
    arm = _require_arm(row["arm"], "cache-performance turn arm")
    ordinal = _require_int(row["lifetime_ordinal"], "cache-performance turn ordinal", positive=True)
    if ordinal not in {2, 4, 6, 8}:
        raise SM121CachePerformanceError("cache-performance turn must be timed")
    if SM121_CACHE_PERFORMANCE_LIFETIME_ARMS.get(ordinal) != arm:
        raise SM121CachePerformanceError("cache-performance turn arm is invalid")
    if row["protocol_case_id"] != SM121_CACHE_PERFORMANCE_CASE_ID:
        raise SM121CachePerformanceError("cache-performance case ID changed")
    case_id = row["case_id"]
    if not isinstance(case_id, str) or re.fullmatch(
        rf"{re.escape(SM121_CACHE_PERFORMANCE_CASE_ID)}--[0-9a-f]{{12}}",
        case_id,
    ) is None:
        raise SM121CachePerformanceError("cache-performance case identifier is invalid")
    turn = row["turn"]
    if turn not in SM121_CACHE_PERFORMANCE_TIMED_TURNS:
        raise SM121CachePerformanceError("cache-performance turn is invalid")
    for field, wanted in (
        ("cache_details_requested", True),
        ("prompt_token_ids_requested", True),
        ("streaming", False),
        ("thinking_disabled", True),
    ):
        if row[field] is not wanted:
            raise SM121CachePerformanceError(f"cache-performance {field} changed")
    prompt = _require_int(row["prompt_tokens"], "cache-performance prompt tokens", positive=True)
    _require_int(row["completion_tokens"], "cache-performance completion tokens", positive=True)
    if _require_int(row["reasoning_tokens"], "cache-performance reasoning tokens") != 0:
        raise SM121CachePerformanceError("cache-performance reasoning is not disabled")
    shared = _require_int(row["shared_prefix_tokens"], "cache-performance shared prefix")
    if turn == "T0":
        if shared != 0 or not (
            SM121_CACHE_PERFORMANCE_COLD_INPUT_MIN_TOKENS
            <= prompt
            <= SM121_CACHE_PERFORMANCE_COLD_INPUT_MAX_TOKENS
        ):
            raise SM121CachePerformanceError("cache-performance T0 shape is invalid")
    elif not (
        SM121_CACHE_PERFORMANCE_COLD_INPUT_MIN_TOKENS <= shared < prompt
    ):
        raise SM121CachePerformanceError("cache-performance shared prefix is invalid")
    _require_bool(row["append_only_prompt_identity_verified"], "cache-performance append identity")
    _require_bool(row["cross_lifetime_prompt_identity_verified"], "cache-performance cross identity")
    detail_values = tuple(
        _require_optional_int(row[field], field)
        for field in (
            "response_device_cached_tokens",
            "response_host_cached_tokens",
            "response_storage_cached_tokens",
        )
    )
    _optional_detail_state(row["response_detail_state"], detail_values, name="response")
    usage = _require_optional_int(row["usage_cached_tokens"], "cache-performance usage cached tokens")
    if row["usage_detail_state"] not in {"omitted", "null", "zero_details", "nonzero_details"}:
        raise SM121CachePerformanceError("cache-performance usage detail state is invalid")
    if row["usage_detail_state"] in {"omitted", "null"} and usage is not None:
        raise SM121CachePerformanceError("cache-performance usage details are invalid")
    if row["usage_detail_state"] == "zero_details" and usage != 0:
        raise SM121CachePerformanceError("cache-performance zero usage is invalid")
    if row["usage_detail_state"] == "nonzero_details" and (usage is None or usage <= 0):
        raise SM121CachePerformanceError("cache-performance nonzero usage is invalid")
    for field in ("metrics_available", "guardrail_metrics_available", "metrics_before_settled", "metrics_after_settled"):
        _require_bool(row[field], f"cache-performance {field}")
    _require_int(row["metrics_before_polls"], "cache-performance metrics-before polls")
    _require_int(row["metrics_after_polls"], "cache-performance metrics-after polls")
    _require_finite(row["request_wall_s"], "cache-performance request wall", positive=True)
    for prefix in ("before", "after", "delta"):
        for metric in SM121_CACHE_PERFORMANCE_METRIC_FIELDS:
            value = row[f"{prefix}_{metric}"]
            if prefix == "delta" and metric.endswith(("available_tokens", "evictable_tokens", "used_tokens")):
                if type(value) is not int:
                    raise SM121CachePerformanceError("cache-performance gauge delta is invalid")
            else:
                _require_int(value, f"cache-performance {prefix} {metric}")
            if row[f"delta_{metric}"] != row[f"after_{metric}"] - row[f"before_{metric}"]:
                raise SM121CachePerformanceError("cache-performance metric delta changed")
    for prefix in ("before", "after"):
        for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
            _require_bool(row[f"{prefix}_cached_{source}_series_present"], "cache-performance cache series marker")
    admitted, basis = derive_sm121_cache_performance_turn_admission(row)
    if row["timed_turn_admitted"] is not admitted or row["timed_turn_basis"] != basis:
        raise SM121CachePerformanceError("cache-performance turn admission changed")
    if arm == "A" and turn != "T0" and detail_values[0] is None:
        raise SM121CachePerformanceError("cache-performance A device details are absent")


def validate_sm121_cache_performance_lifetimes(
    lifetimes: object,
) -> tuple[dict[str, object], ...]:
    """Validate every public lifetime, including a terminal partial prefix.

    A failed lifetime is still allowed to retain the admitted timing turns it
    completed before the failure.  Those rows must remain an exact prefix of
    T0/T1/T2 and must not become an untyped side channel in a partial scalar
    summary.  Later rows are the controller's explicit unstarted tombstones.
    """

    if type(lifetimes) is not list or len(lifetimes) != len(
        SM121_CACHE_PERFORMANCE_ARM_ORDER
    ):
        raise SM121CachePerformanceError("cache-performance lifetimes are invalid")
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
        zip(lifetimes, SM121_CACHE_PERFORMANCE_ARM_ORDER, strict=True), start=1
    ):
        row = _require_exact_keys(raw, expected_fields, "cache-performance lifetime")
        if row["ordinal"] != ordinal or row["arm"] != expected_arm:
            raise SM121CachePerformanceError("cache-performance lifetime order changed")
        quality_admitted = _require_bool(
            row["quality_admitted"], "cache-performance quality admission"
        )
        timed_admitted = _require_bool(
            row["timed_admitted"], "cache-performance timed admission"
        )
        within_timeout = _require_bool(
            row["within_timeout"], "cache-performance timeout"
        )
        turns = row["turns"]
        if type(turns) is not list or len(turns) > len(
            SM121_CACHE_PERFORMANCE_TIMED_TURNS
        ):
            raise SM121CachePerformanceError("cache-performance turns are invalid")
        for turn_index, (expected_turn, event) in enumerate(
            zip(SM121_CACHE_PERFORMANCE_TIMED_TURNS, turns, strict=False)
        ):
            # Summaries and public evidence deliberately omit Journal's
            # wall-clock timestamp.  It is allowed only in raw provenance.
            if type(event) is not dict or "timestamp" in event:
                raise SM121CachePerformanceError(
                    "cache-performance public turn is invalid"
                )
            validate_sm121_cache_performance_turn_event(event)
            if (
                event["arm"] != expected_arm
                or event["lifetime_ordinal"] != ordinal * 2
                or event["turn"] != expected_turn
            ):
                raise SM121CachePerformanceError(
                    "cache-performance turn topology changed"
                )
            if (
                turn_index + 1 < len(turns)
                and event["timed_turn_admitted"] is not True
            ):
                raise SM121CachePerformanceError(
                    "cache-performance failed turn is not terminal"
                )
        if not quality_admitted:
            if timed_admitted or turns:
                raise SM121CachePerformanceError(
                    "cache-performance quality failure retained timings"
                )
        if timed_admitted:
            if not quality_admitted or len(turns) != len(
                SM121_CACHE_PERFORMANCE_TIMED_TURNS
            ) or any(event["timed_turn_admitted"] is not True for event in turns):
                raise SM121CachePerformanceError(
                    "cache-performance timed admission is inconsistent"
                )
        admitted = quality_admitted and timed_admitted and within_timeout
        if terminal:
            if quality_admitted or timed_admitted or within_timeout or turns:
                raise SM121CachePerformanceError(
                    "cache-performance continued after terminal lifetime"
                )
        elif not admitted:
            terminal = True
        rows.append(row)
    return tuple(rows)


def score_sm121_cache_performance_campaign(lifetimes: object) -> CachePerformanceScore:
    """Reduce four scalar timing lifetimes with exact, unrounded thresholds."""

    rows = validate_sm121_cache_performance_lifetimes(lifetimes)
    by_arm: dict[str, list[tuple[Decimal, Decimal]]] = {"A": [], "B": []}
    for ordinal, (row, expected_arm) in enumerate(
        zip(rows, SM121_CACHE_PERFORMANCE_ARM_ORDER, strict=True), start=1
    ):
        admitted = (
            row["quality_admitted"] is True
            and row["timed_admitted"] is True
            and row["within_timeout"] is True
        )
        turns = row["turns"]
        if not admitted:
            return CachePerformanceScore("partial", "not_evaluated", None, None, None, None, None, None)
        assert isinstance(turns, list)
        parsed_turns: list[Mapping[str, object]] = []
        for expected_turn, event in zip(SM121_CACHE_PERFORMANCE_TIMED_TURNS, turns, strict=True):
            assert isinstance(event, Mapping)
            if event["timed_turn_admitted"] is not True:
                raise SM121CachePerformanceError("cache-performance timing admission changed")
            parsed_turns.append(event)
        walls = [_require_finite(event["request_wall_s"], "cache-performance request wall", positive=True) for event in parsed_turns]
        by_arm[expected_arm].append((walls[1] + walls[2], sum(walls, Decimal(0))))
    if any(len(values) != 2 for values in by_arm.values()):
        raise SM121CachePerformanceError("cache-performance replication is incomplete")
    a_later = sum((value[0] for value in by_arm["A"]), Decimal(0)) / Decimal(2)
    b_later = sum((value[0] for value in by_arm["B"]), Decimal(0)) / Decimal(2)
    a_full = sum((value[1] for value in by_arm["A"]), Decimal(0)) / Decimal(2)
    b_full = sum((value[1] for value in by_arm["B"]), Decimal(0)) / Decimal(2)
    if a_later <= 0 or b_later <= 0 or a_full <= 0 or b_full <= 0:
        raise SM121CachePerformanceError("cache-performance timing aggregate is invalid")
    if a_later <= b_later * SM121_CACHE_PERFORMANCE_PROMOTION_RATIO:
        winner, later_ratio, full_ratio = "A", a_later / b_later, a_full / b_full
    elif b_later <= a_later * SM121_CACHE_PERFORMANCE_PROMOTION_RATIO:
        winner, later_ratio, full_ratio = "B", b_later / a_later, b_full / a_full
    else:
        winner, later_ratio, full_ratio = None, None, None
    decision = "no_retention"
    if winner is not None:
        decision = f"retain_{winner.lower()}" if full_ratio <= SM121_CACHE_PERFORMANCE_FULL_WALL_GUARDRAIL_RATIO else "guardrail_reject"
    def number(value: Decimal | None) -> float | None:
        if value is None:
            return None
        number_value = float(value)
        if not math.isfinite(number_value):
            raise SM121CachePerformanceError("cache-performance aggregate is non-finite")
        return number_value
    return CachePerformanceScore(
        "complete",
        decision,
        number(a_later),
        number(b_later),
        number(a_full),
        number(b_full),
        number(later_ratio),
        number(full_ratio),
    )
