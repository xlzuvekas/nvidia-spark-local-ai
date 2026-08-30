"""Static execution tombstones for retired model artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .sglang_sm121_storage import is_sm121_storage_candidate
from .sglang_sm121_agent_admission import is_sm121_agent_admission_candidate
from .sglang_sm121_cache_semantic import is_sm121_cache_semantic_candidate
from .sglang_sm121_cache_performance import is_sm121_cache_performance_candidate
from .sglang_sm121_chunked_prefill_performance import (
    is_sm121_chunked_prefill_performance_candidate,
)


RETIRED_SGLANG_SOURCE_OVERLAY_DIGESTS = frozenset(
    {
        (
            "sha256:e30566492e1502f94a4c7fed42d90b5"
            "23bbb662580c628459e6e63c7b5263c75"
        ),
    }
)
_RETIRED_SGLANG_SOURCE_OVERLAY_MESSAGE = (
    "This model profile uses a retired SGLang source overlay "
    "and cannot be executed"
)
_SM121_STORAGE_CANARY_MESSAGE = (
    "This SM121 native-storage profile is pre-admission and requires the "
    "dedicated sm121-storage-canary command"
)
_SM121_CACHE_SEMANTIC_CANARY_MESSAGE = (
    "This SM121 cache-policy semantic profile is pre-admission and requires "
    "the dedicated sm121-cache-policy-semantic-canary command"
)
_SM121_CACHE_PERFORMANCE_MESSAGE = (
    "This SM121 cache-policy performance profile requires the dedicated "
    "sm121-cache-policy-performance command"
)
_SM121_CHUNKED_PREFILL_PERFORMANCE_MESSAGE = (
    "This SM121 chunked-prefill performance profile requires the dedicated "
    "sm121-chunked-prefill-performance command"
)
_SM121_AGENT_ADMISSION_MESSAGE = (
    "This SM121 low-thinking/tool profile is prospective and requires the "
    "dedicated parser/tool admission controller"
)


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def model_execution_blocker(
    model: Any,
    *,
    allow_sm121_storage_canary: bool = False,
    allow_sm121_cache_semantic_canary: bool = False,
    allow_sm121_cache_performance: bool = False,
    allow_sm121_chunked_prefill_performance: bool = False,
    allow_sm121_agent_admission: bool = False,
) -> str | None:
    """Return a stable blocker when a model contains a retired artifact."""

    semantic_candidate = is_sm121_cache_semantic_candidate(model)
    performance_candidate = is_sm121_cache_performance_candidate(model)
    chunked_prefill_candidate = is_sm121_chunked_prefill_performance_candidate(
        model
    )
    if (
        is_sm121_agent_admission_candidate(model)
        and not allow_sm121_agent_admission
    ):
        return _SM121_AGENT_ADMISSION_MESSAGE
    if (
        semantic_candidate
        and not allow_sm121_cache_semantic_canary
    ):
        return _SM121_CACHE_SEMANTIC_CANARY_MESSAGE
    if (
        performance_candidate
        and not allow_sm121_cache_performance
    ):
        return _SM121_CACHE_PERFORMANCE_MESSAGE
    if (
        chunked_prefill_candidate
        and not allow_sm121_chunked_prefill_performance
    ):
        return _SM121_CHUNKED_PREFILL_PERFORMANCE_MESSAGE
    if (
        is_sm121_storage_candidate(model)
        and not is_sm121_agent_admission_candidate(model)
        and not semantic_candidate
        and not performance_candidate
        and not chunked_prefill_candidate
        and not allow_sm121_storage_canary
    ):
        return _SM121_STORAGE_CANARY_MESSAGE
    overlays = _field(model, "sglang_source_overlays")
    if overlays is None:
        return None
    if isinstance(overlays, Mapping):
        overlays = (overlays,)
    if isinstance(overlays, (str, bytes, bytearray)):
        return None
    try:
        iterator = iter(overlays)
    except TypeError:
        return None
    for overlay in iterator:
        digest = _field(overlay, "digest")
        if (
            isinstance(digest, str)
            and digest in RETIRED_SGLANG_SOURCE_OVERLAY_DIGESTS
        ):
            return _RETIRED_SGLANG_SOURCE_OVERLAY_MESSAGE
    return None
