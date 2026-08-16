"""Read vLLM speculative-decoding counters from its Prometheus endpoint."""

from __future__ import annotations

from collections.abc import Iterable
import math
import re
from typing import Any
import urllib.request


NUM_DRAFTS = "vllm:spec_decode_num_drafts_total"
NUM_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
NUM_ACCEPTED_TOKENS = "vllm:spec_decode_num_accepted_tokens_total"
NUM_ACCEPTED_TOKENS_PER_POSITION = (
    "vllm:spec_decode_num_accepted_tokens_per_pos_total"
)

_CORE_COUNTERS = (NUM_DRAFTS, NUM_DRAFT_TOKENS, NUM_ACCEPTED_TOKENS)
_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[^\s]+)(?:\s+\d+)?$"
)
_POSITION = re.compile(r'(?:^|,)\s*position="(?P<position>\d+)"(?:\s*,|$)')


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
        "source": "vllm_prometheus_cumulative_counters",
        "scope": (
            "single_vllm_server_lifetime_including_prime_warmups_and_measured_requests"
        ),
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
