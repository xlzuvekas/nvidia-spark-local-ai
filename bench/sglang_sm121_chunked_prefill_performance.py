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

import json
from typing import Any, Mapping

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
SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S = 1_200
SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID = "synthetic-exact-answer-v2"
SM121_CHUNKED_PREFILL_PERFORMANCE_CASE_ID = (
    "sm121-chunked-prefill-60k-static-history-v1"
)
SM121_CHUNKED_PREFILL_PERFORMANCE_SERVED_NAME = (
    "qwen38-flash-next-nvfp4-sm121-storage-chunked-prefill-performance"
)
SM121_CHUNKED_PREFILL_PERFORMANCE_MAX_MAMBA_CACHE_SIZE = 4
SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_CHUNK_SIZE = 1_024
SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_CHUNK_SIZE = 2_048
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
