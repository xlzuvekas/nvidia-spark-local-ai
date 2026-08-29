"""Static execution tombstones for retired model artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .sglang_sm121_storage import is_sm121_storage_candidate
from .sglang_sm121_cache_semantic import is_sm121_cache_semantic_candidate


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


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def model_execution_blocker(
    model: Any,
    *,
    allow_sm121_storage_canary: bool = False,
    allow_sm121_cache_semantic_canary: bool = False,
) -> str | None:
    """Return a stable blocker when a model contains a retired artifact."""

    semantic_candidate = is_sm121_cache_semantic_candidate(model)
    if (
        semantic_candidate
        and not allow_sm121_cache_semantic_canary
    ):
        return _SM121_CACHE_SEMANTIC_CANARY_MESSAGE
    if (
        is_sm121_storage_candidate(model)
        and not semantic_candidate
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
