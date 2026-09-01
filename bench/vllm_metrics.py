"""Read vLLM speculative-decoding counters from its Prometheus endpoint."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
import re
from typing import Any
import urllib.request


NUM_DRAFTS = "vllm:spec_decode_num_drafts_total"
NUM_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
NUM_ACCEPTED_TOKENS = "vllm:spec_decode_num_accepted_tokens_total"
NUM_ACCEPTED_TOKENS_PER_POSITION = "vllm:spec_decode_num_accepted_tokens_per_pos_total"

VLLM_SPEC_DECODE_SOURCE = "vllm_prometheus_cumulative_counters"
VLLM_SPEC_DECODE_CUMULATIVE_SCOPE = (
    "single_vllm_server_lifetime_including_prime_warmups_and_measured_requests"
)
VLLM_SPEC_DECODE_REQUEST_DELTA_SCOPE = "request_scoped_cumulative_counter_delta"
VLLM_SPEC_DECODE_CASE_AGGREGATE_SCOPE = (
    "case_measurement_request_deltas_aggregate"
)

_CORE_COUNTERS = (NUM_DRAFTS, NUM_DRAFT_TOKENS, NUM_ACCEPTED_TOKENS)
_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[^\s]+)(?:\s+\d+)?$"
)
_POSITION = re.compile(r'(?:^|,)\s*position="(?P<position>\d+)"(?:\s*,|$)')
_POSITION_KEY = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_CUMULATIVE_SNAPSHOT_KEYS = {
    "accepted_tokens_per_position",
    "draft_acceptance_rate",
    "mean_accepted_length",
    "num_accepted_tokens",
    "num_draft_tokens",
    "num_drafts",
    "scope",
    "source",
}


class VLLMSpecDecodeMetricsDeltaError(ValueError):
    """A cumulative vLLM counter pair cannot yield a trustworthy delta."""


def _count(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _snapshot(
    *,
    drafts: float,
    draft_tokens: float,
    accepted_tokens: float,
    per_position: dict[int, float],
) -> dict[str, Any]:
    return {
        "source": VLLM_SPEC_DECODE_SOURCE,
        "scope": VLLM_SPEC_DECODE_CUMULATIVE_SCOPE,
        "num_drafts": _count(drafts),
        "num_draft_tokens": _count(draft_tokens),
        "num_accepted_tokens": _count(accepted_tokens),
        "accepted_tokens_per_position": {
            str(position): _count(value)
            for position, value in sorted(per_position.items())
        },
        "draft_acceptance_rate": (
            accepted_tokens / draft_tokens if draft_tokens > 0 else None
        ),
        # vLLM's mean acceptance length includes the verifier's bonus token.
        "mean_accepted_length": (
            1.0 + accepted_tokens / drafts if drafts > 0 else None
        ),
    }


def _strict_count(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VLLMSpecDecodeMetricsDeltaError(
            f"vLLM speculative counter {name} must be a non-negative integer"
        )
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise VLLMSpecDecodeMetricsDeltaError(
                f"vLLM speculative counter {name} must be a non-negative integer"
            )
        value = int(value)
    if value < 0:
        raise VLLMSpecDecodeMetricsDeltaError(
            f"vLLM speculative counter {name} must be a non-negative integer"
        )
    return value


def _strict_derived_value(
    snapshot: Mapping[str, Any],
    *,
    name: str,
    expected: float | None,
    snapshot_name: str,
) -> None:
    if name not in snapshot:
        raise VLLMSpecDecodeMetricsDeltaError(
            f"{snapshot_name} vLLM speculative snapshot is missing {name}"
        )
    value = snapshot[name]
    if expected is None:
        if value is not None:
            raise VLLMSpecDecodeMetricsDeltaError(
                f"{snapshot_name} vLLM speculative {name} is inconsistent"
            )
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not math.isclose(float(value), expected, rel_tol=1e-9, abs_tol=1e-12)
    ):
        raise VLLMSpecDecodeMetricsDeltaError(
            f"{snapshot_name} vLLM speculative {name} is inconsistent"
        )


def _strict_positions(value: Any, *, snapshot_name: str) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise VLLMSpecDecodeMetricsDeltaError(
            f"{snapshot_name} accepted_tokens_per_position must be an object"
        )
    positions: dict[int, int] = {}
    for raw_position, raw_value in value.items():
        if (
            not isinstance(raw_position, str)
            or _POSITION_KEY.fullmatch(raw_position) is None
        ):
            raise VLLMSpecDecodeMetricsDeltaError(
                f"{snapshot_name} accepted-token position is not a canonical index"
            )
        position = int(raw_position)
        positions[position] = _strict_count(
            raw_value,
            name=f"{snapshot_name}.accepted_tokens_per_position[{raw_position}]",
        )
    if positions and set(positions) != set(range(max(positions) + 1)):
        raise VLLMSpecDecodeMetricsDeltaError(
            f"{snapshot_name} accepted-token positions are not contiguous"
        )
    return positions


def _strict_cumulative_snapshot(
    snapshot: Any, *, snapshot_name: str
) -> tuple[int, int, int, dict[int, int]]:
    if not isinstance(snapshot, Mapping) or set(snapshot) != _CUMULATIVE_SNAPSHOT_KEYS:
        raise VLLMSpecDecodeMetricsDeltaError(
            f"{snapshot_name} vLLM speculative snapshot does not match its schema"
        )
    if (
        snapshot.get("source") != VLLM_SPEC_DECODE_SOURCE
        or snapshot.get("scope") != VLLM_SPEC_DECODE_CUMULATIVE_SCOPE
    ):
        raise VLLMSpecDecodeMetricsDeltaError(
            f"{snapshot_name} vLLM speculative snapshot is not one cumulative lifetime"
        )

    drafts = _strict_count(
        snapshot.get("num_drafts"), name=f"{snapshot_name}.num_drafts"
    )
    draft_tokens = _strict_count(
        snapshot.get("num_draft_tokens"),
        name=f"{snapshot_name}.num_draft_tokens",
    )
    accepted_tokens = _strict_count(
        snapshot.get("num_accepted_tokens"),
        name=f"{snapshot_name}.num_accepted_tokens",
    )
    if (
        (drafts == 0) != (draft_tokens == 0)
        or draft_tokens < drafts
        or accepted_tokens > draft_tokens
    ):
        raise VLLMSpecDecodeMetricsDeltaError(
            f"{snapshot_name} vLLM speculative core counters are inconsistent"
        )

    positions = _strict_positions(
        snapshot.get("accepted_tokens_per_position"),
        snapshot_name=snapshot_name,
    )
    ordered = [positions[position] for position in sorted(positions)]
    if any(value > drafts for value in ordered) or any(
        left < right for left, right in zip(ordered, ordered[1:])
    ):
        raise VLLMSpecDecodeMetricsDeltaError(
            f"{snapshot_name} vLLM speculative position counters are inconsistent"
        )

    _strict_derived_value(
        snapshot,
        name="draft_acceptance_rate",
        expected=(accepted_tokens / draft_tokens if draft_tokens else None),
        snapshot_name=snapshot_name,
    )
    _strict_derived_value(
        snapshot,
        name="mean_accepted_length",
        expected=(1.0 + accepted_tokens / drafts if drafts else None),
        snapshot_name=snapshot_name,
    )
    return drafts, draft_tokens, accepted_tokens, positions


def delta_vllm_spec_decode_metrics(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Subtract two same-lifetime cumulative snapshots for one request window.

    The result is admitted only when every core and per-position counter is
    monotonic and the delta forms a self-consistent speculative-decoding
    record. Counter resets, partial position series, and malformed or
    arithmetically impossible inputs raise ``VLLMSpecDecodeMetricsDeltaError``.
    """

    before_values = _strict_cumulative_snapshot(before, snapshot_name="before")
    after_values = _strict_cumulative_snapshot(after, snapshot_name="after")
    before_drafts, before_draft_tokens, before_accepted, before_positions = (
        before_values
    )
    after_drafts, after_draft_tokens, after_accepted, after_positions = after_values

    if any(
        after_value < before_value
        for before_value, after_value in zip(before_values[:3], after_values[:3])
    ):
        raise VLLMSpecDecodeMetricsDeltaError(
            "vLLM speculative core counter reset between snapshots"
        )
    missing_positions = set(before_positions) - set(after_positions)
    if missing_positions:
        raise VLLMSpecDecodeMetricsDeltaError(
            "vLLM speculative position counter disappeared between snapshots"
        )

    drafts = after_drafts - before_drafts
    draft_tokens = after_draft_tokens - before_draft_tokens
    accepted_tokens = after_accepted - before_accepted
    per_position: dict[int, int] = {}
    for position, after_value in sorted(after_positions.items()):
        before_value = before_positions.get(position, 0)
        if after_value < before_value:
            raise VLLMSpecDecodeMetricsDeltaError(
                "vLLM speculative position counter reset between snapshots"
            )
        per_position[position] = after_value - before_value

    ordered = [per_position[position] for position in sorted(per_position)]
    if (
        (drafts == 0) != (draft_tokens == 0)
        or draft_tokens < drafts
        or accepted_tokens > draft_tokens
        or any(value > drafts for value in ordered)
        or any(left < right for left, right in zip(ordered, ordered[1:]))
        or sum(ordered) != accepted_tokens
    ):
        raise VLLMSpecDecodeMetricsDeltaError(
            "vLLM speculative request-scoped counter delta is inconsistent"
        )

    result = _snapshot(
        drafts=float(drafts),
        draft_tokens=float(draft_tokens),
        accepted_tokens=float(accepted_tokens),
        per_position={
            position: float(value) for position, value in per_position.items()
        },
    )
    result["scope"] = VLLM_SPEC_DECODE_REQUEST_DELTA_SCOPE
    return result


def _strict_request_delta(
    snapshot: Mapping[str, Any], *, snapshot_name: str
) -> tuple[int, int, int, dict[int, int]]:
    normalized = dict(snapshot)
    if normalized.get("scope") != VLLM_SPEC_DECODE_REQUEST_DELTA_SCOPE:
        raise VLLMSpecDecodeMetricsDeltaError(
            f"{snapshot_name} vLLM speculative snapshot is not request scoped"
        )
    normalized["scope"] = VLLM_SPEC_DECODE_CUMULATIVE_SCOPE
    values = _strict_cumulative_snapshot(
        normalized,
        snapshot_name=snapshot_name,
    )
    _drafts, _draft_tokens, accepted_tokens, positions = values
    if sum(positions.values()) != accepted_tokens:
        raise VLLMSpecDecodeMetricsDeltaError(
            f"{snapshot_name} request-scoped position counters are inconsistent"
        )
    return values


def aggregate_vllm_spec_decode_metric_deltas(
    deltas: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine exact per-request deltas from one measured case window."""

    drafts = 0
    draft_tokens = 0
    accepted_tokens = 0
    per_position: dict[int, int] = {}
    request_count = 0
    for request_count, delta in enumerate(deltas, start=1):
        values = _strict_request_delta(
            delta,
            snapshot_name=f"request_delta[{request_count - 1}]",
        )
        request_drafts, request_draft_tokens, request_accepted, positions = values
        drafts += request_drafts
        draft_tokens += request_draft_tokens
        accepted_tokens += request_accepted
        for position, value in positions.items():
            per_position[position] = per_position.get(position, 0) + value

    if request_count == 0:
        raise VLLMSpecDecodeMetricsDeltaError(
            "at least one request-scoped vLLM speculative delta is required"
        )
    result = _snapshot(
        drafts=float(drafts),
        draft_tokens=float(draft_tokens),
        accepted_tokens=float(accepted_tokens),
        per_position={
            position: float(value) for position, value in per_position.items()
        },
    )
    result["scope"] = VLLM_SPEC_DECODE_CASE_AGGREGATE_SCOPE
    result["request_count"] = request_count
    return result


def parse_vllm_spec_decode_metrics(exposition: str) -> dict[str, Any] | None:
    """Parse and aggregate the exact speculative counters exposed by vLLM 0.24."""

    totals = {name: 0.0 for name in _CORE_COUNTERS}
    seen: set[str] = set()
    per_position: dict[int, float] = {}
    for line in exposition.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        name = match.group("name")
        if name not in {*_CORE_COUNTERS, NUM_ACCEPTED_TOKENS_PER_POSITION}:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if not math.isfinite(value) or value < 0:
            continue
        if name in totals:
            totals[name] += value
            seen.add(name)
            continue
        position_match = _POSITION.search(match.group("labels") or "")
        if position_match:
            position = int(position_match.group("position"))
            per_position[position] = per_position.get(position, 0.0) + value

    if seen != set(_CORE_COUNTERS):
        return None

    drafts = totals[NUM_DRAFTS]
    draft_tokens = totals[NUM_DRAFT_TOKENS]
    accepted_tokens = totals[NUM_ACCEPTED_TOKENS]
    return _snapshot(
        drafts=drafts,
        draft_tokens=draft_tokens,
        accepted_tokens=accepted_tokens,
        per_position=per_position,
    )


def aggregate_vllm_spec_decode_metrics(
    snapshots: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Combine disjoint cumulative snapshots from resumed server lifetimes."""

    drafts = 0.0
    draft_tokens = 0.0
    accepted_tokens = 0.0
    per_position: dict[int, float] = {}
    snapshot_count = 0
    for snapshot in snapshots:
        values = (
            snapshot.get("num_drafts"),
            snapshot.get("num_draft_tokens"),
            snapshot.get("num_accepted_tokens"),
        )
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
            for value in values
        ):
            continue
        drafts += float(values[0])
        draft_tokens += float(values[1])
        accepted_tokens += float(values[2])
        raw_positions = snapshot.get("accepted_tokens_per_position", {})
        if isinstance(raw_positions, dict):
            for raw_position, raw_value in raw_positions.items():
                try:
                    position = int(raw_position)
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if position >= 0 and math.isfinite(value) and value >= 0:
                    per_position[position] = per_position.get(position, 0.0) + value
        snapshot_count += 1

    if snapshot_count == 0:
        return None
    combined = _snapshot(
        drafts=drafts,
        draft_tokens=draft_tokens,
        accepted_tokens=accepted_tokens,
        per_position=per_position,
    )
    combined["snapshot_count"] = snapshot_count
    combined["scope"] = "all_persisted_vllm_server_lifetimes"
    return combined


def snapshot_vllm_spec_decode_metrics(
    base_url: str, *, timeout_s: float = 2.0
) -> dict[str, Any] | None:
    """Fetch one cumulative snapshot; absent/unavailable metrics are optional."""

    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(root + "/metrics", timeout=timeout_s) as response:
            exposition = response.read().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return parse_vllm_spec_decode_metrics(exposition)
