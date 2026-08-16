"""Read llama.cpp speculative-decoding counters from its metrics endpoint."""

from __future__ import annotations

from collections.abc import Iterable
import math
import re
from typing import Any
import urllib.request


NUM_DRAFTS = "llamacpp:spec_decode_num_drafts_total"
NUM_DRAFT_TOKENS = "llamacpp:spec_decode_num_draft_tokens_total"
NUM_ACCEPTED_TOKENS = "llamacpp:spec_decode_num_accepted_tokens_total"
NUM_ACCEPTED_TOKENS_PER_POSITION = (
    "llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
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
        "source": "llamacpp_prometheus_cumulative_counters",
        "scope": (
            "single_llamacpp_server_lifetime_including_prime_warmups_and_"
            "measured_requests"
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
        "mean_accepted_length": (
            1.0 + accepted_tokens / drafts if drafts > 0 else None
        ),
    }


def parse_llamacpp_spec_decode_metrics(
    exposition: str,
) -> dict[str, Any] | None:
    """Parse the exact speculative counters emitted by llama.cpp b10453."""

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
    return _snapshot(
        drafts=totals[NUM_DRAFTS],
        draft_tokens=totals[NUM_DRAFT_TOKENS],
        accepted_tokens=totals[NUM_ACCEPTED_TOKENS],
        per_position=per_position,
    )


def aggregate_llamacpp_spec_decode_metrics(
    snapshots: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Combine disjoint cumulative snapshots from resumed native lifetimes."""

    drafts = 0.0
    draft_tokens = 0.0
    accepted_tokens = 0.0
    per_position: dict[int, float] = {}
    count = 0
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
        positions = snapshot.get("accepted_tokens_per_position", {})
        if isinstance(positions, dict):
            for raw_position, raw_value in positions.items():
                try:
                    position = int(raw_position)
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if position >= 0 and math.isfinite(value) and value >= 0:
                    per_position[position] = per_position.get(position, 0.0) + value
        count += 1
    if not count:
        return None
    combined = _snapshot(
        drafts=drafts,
        draft_tokens=draft_tokens,
        accepted_tokens=accepted_tokens,
        per_position=per_position,
    )
    combined["snapshot_count"] = count
    combined["scope"] = "all_persisted_llamacpp_server_lifetimes"
    return combined


def snapshot_llamacpp_spec_decode_metrics(
    base_url: str, *, timeout_s: float = 2.0
) -> dict[str, Any] | None:
    """Fetch one cumulative snapshot from the loopback metrics endpoint."""

    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(root + "/metrics", timeout=timeout_s) as response:
            exposition = response.read().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return parse_llamacpp_spec_decode_metrics(exposition)


def require_mtp_activity(metrics: dict[str, Any] | None) -> None:
    """Reject an MTP run that did not draft and accept at least one token."""

    if metrics is None:
        raise RuntimeError("llama.cpp MTP metrics were unavailable")
    drafted = metrics.get("num_draft_tokens")
    accepted = metrics.get("num_accepted_tokens")
    steps = metrics.get("num_drafts")
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
        for value in (drafted, accepted, steps)
    ):
        raise RuntimeError(
            "llama.cpp MTP was requested but drafted/accepted counters were not all positive"
        )
    if float(accepted) > float(drafted):
        raise RuntimeError("llama.cpp accepted-token counter exceeds drafted tokens")


def llamacpp_mtp_requested(arguments: Iterable[Any]) -> bool:
    """Return whether frozen llama.cpp arguments request embedded MTP drafting."""

    values = [str(argument) for argument in arguments]
    for index, argument in enumerate(values):
        if argument.startswith("--spec-type="):
            configured = argument.split("=", 1)[1]
        elif argument == "--spec-type" and index + 1 < len(values):
            configured = values[index + 1]
        else:
            continue
        if "draft-mtp" in {
            item.strip() for item in configured.split(",") if item.strip()
        }:
            return True
    return False


def llamacpp_mtp_depth(arguments: Iterable[Any]) -> int | None:
    """Return one explicit positive ``--spec-draft-n-max`` value."""

    values = [str(argument) for argument in arguments]
    configured: list[str] = []
    for index, argument in enumerate(values):
        if argument.startswith("--spec-draft-n-max="):
            configured.append(argument.split("=", 1)[1])
        elif argument == "--spec-draft-n-max" and index + 1 < len(values):
            configured.append(values[index + 1])
    if len(configured) != 1:
        return None
    try:
        depth = int(configured[0])
    except ValueError:
        return None
    return depth if depth > 0 else None


def assess_llamacpp_mtp_proposal_depth(
    metrics: dict[str, Any] | None, *, configured_depth: int
) -> dict[str, Any]:
    """Prove counters exercised an explicitly configured maximum draft length."""

    evidence: dict[str, Any] = {
        "configured_max_draft_tokens": configured_depth,
        "passed": False,
        "average_draft_tokens_per_draft": None,
        "deepest_accepted_position": None,
        "deepest_accepted_draft_depth": None,
        "reason": None,
    }
    if isinstance(configured_depth, bool) or configured_depth <= 0:
        evidence["reason"] = "configured MTP draft length must be positive"
        return evidence
    try:
        require_mtp_activity(metrics)
    except RuntimeError as error:
        evidence["reason"] = str(error)
        return evidence
    assert metrics is not None
    drafts = float(metrics["num_drafts"])
    drafted_tokens = float(metrics["num_draft_tokens"])
    average_width = drafted_tokens / drafts
    evidence["average_draft_tokens_per_draft"] = average_width

    if average_width > configured_depth + 1e-9:
        evidence["reason"] = (
            "observed average proposal width exceeds the configured maximum"
        )
        return evidence
    if average_width <= configured_depth - 1:
        evidence["reason"] = (
            "observed counters do not prove the configured maximum draft "
            "length was exercised"
        )
        return evidence

    positions = metrics.get("accepted_tokens_per_position", {})
    if not isinstance(positions, dict):
        evidence["reason"] = "accepted-token position counters are not an object"
        return evidence
    deepest: int | None = None
    for raw_position, raw_count in positions.items():
        try:
            position = int(raw_position)
            count = float(raw_count)
        except (TypeError, ValueError):
            evidence["reason"] = "accepted-token position counter is malformed"
            return evidence
        if (
            isinstance(raw_count, bool)
            or position < 0
            or not math.isfinite(count)
            or count < 0
        ):
            evidence["reason"] = "accepted-token position counter is invalid"
            return evidence
        if position >= configured_depth:
            evidence["reason"] = (
                "accepted-token position exceeds the configured maximum draft length"
            )
            return evidence
        if count > 0 and (deepest is None or position > deepest):
            deepest = position
    evidence.update(
        {
            "passed": True,
            "deepest_accepted_position": deepest,
            "deepest_accepted_draft_depth": (
                deepest + 1 if deepest is not None else None
            ),
        }
    )
    return evidence


def assess_llamacpp_mtp_evidence(
    events: Iterable[dict[str, Any]],
    *,
    requested: bool,
    configured_depth: int | None = None,
) -> dict[str, Any]:
    """Check that every reported case lifetime has a later valid MTP snapshot."""

    records = list(events)
    evidence: dict[str, Any] = {
        "requested": requested,
        "configured_max_draft_tokens": configured_depth,
        "passed": True,
        "contributing_lifetimes": 0,
        "validated_lifetimes": 0,
        "proposal_depth_validated_lifetimes": 0,
        "reason": None,
    }
    if not requested:
        return evidence

    starts = [
        index
        for index, event in enumerate(records)
        if event.get("event") == "run_start"
    ]
    final_completions: dict[str, int] = {}
    for index, event in enumerate(records):
        if event.get("event") == "case_complete":
            final_completions[str(event.get("case_id"))] = index
    if not final_completions:
        return evidence
    if not starts or min(final_completions.values()) < starts[0]:
        evidence.update(
            {
                "passed": False,
                "reason": "completed llama.cpp case has no owning run lifetime",
            }
        )
        return evidence

    boundaries = [*starts, len(records)]
    contributing = []
    for position, start in enumerate(starts):
        end = boundaries[position + 1]
        completions = tuple(
            index
            for index in final_completions.values()
            if start < index < end
        )
        if completions:
            contributing.append((start, end, completions))
    evidence["contributing_lifetimes"] = len(contributing)
    for start, end, completions in contributing:
        last_completion = max(completions)
        snapshots = [
            event.get("metrics")
            for index, event in enumerate(records)
            if last_completion < index < end
            and event.get("event") == "llamacpp_spec_decode_metrics_snapshot"
            and isinstance(event.get("metrics"), dict)
        ]
        if not snapshots:
            evidence.update(
                {
                    "passed": False,
                    "reason": (
                        "completed llama.cpp MTP lifetime has no later "
                        "speculative-decoding metrics snapshot"
                    ),
                }
            )
            return evidence
        try:
            require_mtp_activity(snapshots[-1])
        except RuntimeError as error:
            evidence.update({"passed": False, "reason": str(error)})
            return evidence
        evidence["validated_lifetimes"] += 1
        if configured_depth is not None:
            depth_evidence = assess_llamacpp_mtp_proposal_depth(
                snapshots[-1], configured_depth=configured_depth
            )
            if not depth_evidence["passed"]:
                evidence.update(
                    {
                        "passed": False,
                        "reason": depth_evidence["reason"],
                    }
                )
                return evidence
            evidence["proposal_depth_validated_lifetimes"] += 1
    return evidence


def require_llamacpp_mtp_evidence(
    arguments: Iterable[Any], events: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Fail closed when completed MTP measurements lack per-lifetime proof."""

    evidence = assess_llamacpp_mtp_evidence(
        events,
        requested=llamacpp_mtp_requested(arguments),
        configured_depth=llamacpp_mtp_depth(arguments),
    )
    if evidence["requested"] and not evidence["passed"]:
        raise RuntimeError(f"llama.cpp MTP evidence is incomplete: {evidence['reason']}")
    return evidence
