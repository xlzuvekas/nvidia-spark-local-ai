"""Strict scalar parsing for llama.cpp global Prometheus cache counters."""

from __future__ import annotations

import math
import re
import urllib.request


PROMPT_TOKENS = "llamacpp:prompt_tokens_total"
CACHED_PROMPT_TOKENS = "llamacpp:prompt_tokens_cached_total"
PROMPT_SECONDS = "llamacpp:prompt_seconds_total"
DECODE_TOKENS = "llamacpp:tokens_predicted_total"
DECODE_SECONDS = "llamacpp:tokens_predicted_seconds_total"
_COUNTERS = (
    PROMPT_TOKENS,
    CACHED_PROMPT_TOKENS,
    PROMPT_SECONDS,
    DECODE_TOKENS,
    DECODE_SECONDS,
)
_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{[^}]*\})?\s+(?P<value>[^\s]+)(?:\s+\d+)?$"
)


class LlamaCppCacheMetricsError(RuntimeError):
    """Raised when a cache benchmark cannot retain a valid metrics delta."""


def _scalar(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def parse_llamacpp_cache_metrics(exposition: str) -> dict[str, int | float] | None:
    """Parse the five cumulative prompt/cache/decode counters from ``/metrics``.

    Unknown exposition is intentionally ignored, while missing, non-finite, or
    negative required counters make the snapshot unavailable.  A labelled
    counter is summed defensively, matching the existing speculative-counter
    parser's treatment of Prometheus samples.
    """

    totals = {counter: 0.0 for counter in _COUNTERS}
    seen: set[str] = set()
    for line in exposition.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            continue
        name = match.group("name")
        if name not in totals:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            return None
        if not math.isfinite(value) or value < 0:
            return None
        totals[name] += value
        seen.add(name)
    if seen != set(_COUNTERS):
        return None
    return {
        "prompt_tokens": _scalar(totals[PROMPT_TOKENS]),
        "cached_prompt_tokens": _scalar(totals[CACHED_PROMPT_TOKENS]),
        "prompt_s": _scalar(totals[PROMPT_SECONDS]),
        "decode_tokens": _scalar(totals[DECODE_TOKENS]),
        "decode_s": _scalar(totals[DECODE_SECONDS]),
    }


def snapshot_llamacpp_cache_metrics(
    base_url: str, *, timeout_s: float = 2.0
) -> dict[str, int | float] | None:
    """Read one loopback cumulative counter snapshot from llama.cpp."""

    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(root + "/metrics", timeout=timeout_s) as response:
            exposition = response.read().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return parse_llamacpp_cache_metrics(exposition)


def delta_llamacpp_cache_metrics(
    before: dict[str, int | float] | None,
    after: dict[str, int | float] | None,
) -> dict[str, int | float]:
    """Return one non-negative global Prometheus delta or fail closed.

    llama.cpp exposes these counters across server batches and slots.  They are
    useful scalar diagnostics for a serial cache run, but are not attributed to
    a single request.  Per-request cache accounting comes from the final SSE
    usage and timings payload instead.
    """

    if before is None or after is None:
        raise LlamaCppCacheMetricsError(
            "native llama.cpp prompt-cache counters were unavailable"
        )
    delta: dict[str, int | float] = {}
    for key in ("prompt_tokens", "cached_prompt_tokens", "prompt_s", "decode_tokens", "decode_s"):
        left = before.get(key)
        right = after.get(key)
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, (int, float))
            or not isinstance(right, (int, float))
            or not math.isfinite(float(left))
            or not math.isfinite(float(right))
        ):
            raise LlamaCppCacheMetricsError(
                "native llama.cpp prompt-cache counters were invalid"
            )
        value = float(right) - float(left)
        if value < -1e-9:
            raise LlamaCppCacheMetricsError(
                "native llama.cpp prompt-cache counters decreased"
            )
        delta[key] = _scalar(max(value, 0.0))
    return delta


def require_llamacpp_cache_delta(
    delta: dict[str, int | float],
) -> None:
    """Validate a non-negative global Prometheus diagnostic delta.

    These metrics intentionally do *not* reconcile with a single request:
    llama.cpp accumulates them at server/batch scope.  Exact per-request token
    and timing identities are enforced from the final SSE payload by the cache
    runner and evidence validator.
    """

    for key in (
        "prompt_tokens",
        "cached_prompt_tokens",
        "decode_tokens",
        "prompt_s",
        "decode_s",
    ):
        value = delta.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise LlamaCppCacheMetricsError(
                "native llama.cpp prompt-cache metrics delta was invalid"
            )
