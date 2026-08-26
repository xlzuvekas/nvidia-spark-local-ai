"""Deadline-bounded local RLM and HALO benchmark campaign.

The campaign intentionally lives beside SparkBench's request-oriented suites:
recursive episodes and trace-analysis agents have multi-call lifecycles that do
not fit the one-request case schema.  It still reuses SparkBench's exclusive
lock, managed vLLM lifecycle, telemetry, pinned model registry, and journal.

Only scalar measurements are journaled.  BABILong contexts, model responses,
RLM code, HALO tool payloads, and synthetic trace rows stay transient or under
the ignored ``results/`` tree.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from types import SimpleNamespace
from typing import Any, Iterable, Mapping
import urllib.error
import urllib.parse
import urllib.request

from .journal import Journal, utc_now
from .manifest import load_models
from .runner import _preflight, results_lock_path
from .runtime import (
    ManagedServer,
    recover_owned_vllm,
    save_server_logs,
    start_server,
)
from .telemetry import TelemetrySampler


PROTOCOL_VERSION = 2
PLAN_SCHEMA_VERSION = 2
SUPPORTED_PLAN_SCHEMA_VERSIONS = frozenset({1, PLAN_SCHEMA_VERSION})
SUMMARY_SCHEMA_VERSION = "sparkbench-loop-campaign-summary-v2"
DEFAULT_MANIFEST = Path("manifests/campaigns/rlm_halo_overnight.toml")
DEFAULT_MODELS = Path("manifests/models.toml")
DEFAULT_RESULTS = Path("results/loop-campaigns")
DEFAULT_LOOP_PYTHON = Path.home() / ".cache" / "local-llm-loop-env" / "bin" / "python"
MAX_WORKER_OUTPUT_BYTES = 1024 * 1024
WORKER_IMAGE = (
    "nvcr.io/nvidia/vllm:26.07-py3@"
    "sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268"
)

BABILONG_SOURCE = "RMT-team/babilong"
BABILONG_REVISION = "ee0d588794c7ac098062ee0d247c733d62e94fe2"
BABILONG_LENGTHS = frozenset({"4k", "8k", "16k", "32k", "64k", "128k"})
BABILONG_TASKS = frozenset({f"qa{index}" for index in range(1, 11)})
BABILONG_LOCATION_LABELS = (
    "bathroom",
    "bedroom",
    "garden",
    "hallway",
    "kitchen",
    "office",
)

RLM_SOURCE = "alexzhang13/rlm"
RLM_REVISION = "0b45df99c43fb3844a3b796a15d13c0f9d07afd8"
RLM_VERSION = "0.1.3"
RLM_COMPACTION_CONTEXT_TOKENS = 32_768
RLM_COMPACTION_THRESHOLD_PCT = 0.85
RLM_FORCED_COMPACTION_THRESHOLD_PCT = 0.20
RLM_ADMISSION_ADMITTED = "admitted"
RLM_DEPTH2_HOLD = "held_child_compaction_unverified"
HALO_SOURCE = "context-labs/HALO"
HALO_REVISION = "b7f8509745d67b499b4e80efe20ea37c03426a74"
HALO_VERSION = "0.3.5"
OPENAI_AGENTS_VERSION = "0.14.7"
OPENAI_VERSION = "2.32.0"
PYARROW_VERSION = "21.0.0"
HALO_SUBAGENT_ARGUMENT_ERROR = (
    "Invalid call_subagent arguments. Retry with a JSON object containing a string "
    "field named input."
)

HALO_FAILURE_FAMILIES = (
    "search_timeout",
    "write_conflict",
    "duplicate_edge",
    "entity_split",
    "invalid_arguments",
    "empty_retrieval",
)

PROMETHEUS_COUNTERS = frozenset(
    {
        "vllm:prompt_tokens_total",
        "vllm:prompt_tokens_cached_total",
        "vllm:generation_tokens_total",
        "vllm:request_success_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
    }
)
_PROMETHEUS_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[^\s]+)(?:\s+\d+)?$"
)

_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "id", "description", "window", "upstreams", "rlm", "halo"}
)
_WINDOW_KEYS = frozenset(
    {"rlm_stop_at", "measurement_stop_at", "hard_stop_at", "cleanup_reserve_s"}
)
_UPSTREAM_KEYS = frozenset(
    {
        "rlm_source",
        "rlm_revision",
        "rlm_version",
        "halo_source",
        "halo_revision",
        "halo_version",
        "openai_agents_version",
        "openai_version",
        "pyarrow_version",
        "babilong_source",
        "babilong_revision",
    }
)
_RLM_KEYS = frozenset(
    {
        "model_profile",
        "reasoning_control",
        "lengths",
        "direct_lengths",
        "tasks",
        "row_indices",
        "max_iterations",
        "max_concurrent_subcalls",
        "max_total_tokens",
        "max_output_tokens",
        "compaction",
        "compaction_threshold_pct",
        "direct_timeout_s",
        "episode_timeout_s",
        "recursive_depth2_tasks",
        "recursive_depth2_lengths",
        "recursive_depth2_rows",
        "worker_isolation",
    }
)
_RLM_OPTIONAL_KEYS = frozenset({"compaction_diagnostic"})
_RLM_COMPACTION_DIAGNOSTIC_KEYS = frozenset(
    {"threshold_pct", "tasks", "lengths", "row_indices"}
)
_HALO_KEYS = frozenset(
    {
        "model_profiles",
        "reasoning_effort",
        "trace_counts",
        "seeds",
        "depths",
        "max_parallel",
        "max_turns",
        "max_output_tokens",
        "episode_timeout_s",
        "depth2_trace_counts",
        "depth2_seeds",
    }
)


class LoopCampaignError(RuntimeError):
    """A public-safe campaign failure."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(value: Any, length: int = 64) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()[:length]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _parse_datetime(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str):
        raise LoopCampaignError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise LoopCampaignError(f"{name} is not valid ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LoopCampaignError(f"{name} must include an explicit UTC offset")
    return parsed


def _require_exact_keys(value: Any, expected: frozenset[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LoopCampaignError(f"{name} must be a table")
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise LoopCampaignError(
            f"{name} keys do not match the frozen schema "
            f"(missing={sorted(missing)}, unknown={sorted(unknown)})"
        )
    return value


def _string_list(value: Any, *, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
        or len(set(value)) != len(value)
    ):
        raise LoopCampaignError(f"{name} must be a non-empty unique string list")
    return list(value)


def _int_list(value: Any, *, name: str, minimum: int = 0) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or not all(type(item) is int and item >= minimum for item in value)
        or len(set(value)) != len(value)
    ):
        raise LoopCampaignError(f"{name} must be a non-empty unique integer list")
    return list(value)


def _positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise LoopCampaignError(f"{name} must be a positive integer")
    return value


def load_campaign_manifest(path: Path) -> dict[str, Any]:
    """Load and strictly validate the date-specific overnight campaign."""

    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise LoopCampaignError("Could not read the loop campaign manifest") from error
    _require_exact_keys(document, _TOP_LEVEL_KEYS, name="campaign")
    if document["schema_version"] != PLAN_SCHEMA_VERSION:
        raise LoopCampaignError("Unsupported loop campaign manifest schema")
    if not isinstance(document["id"], str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_-]*", document["id"]
    ):
        raise LoopCampaignError("Campaign id is invalid")
    if not isinstance(document["description"], str) or not document["description"]:
        raise LoopCampaignError("Campaign description is required")

    window = _require_exact_keys(document["window"], _WINDOW_KEYS, name="window")
    rlm_stop = _parse_datetime(window["rlm_stop_at"], name="window.rlm_stop_at")
    measurement_stop = _parse_datetime(
        window["measurement_stop_at"], name="window.measurement_stop_at"
    )
    hard_stop = _parse_datetime(window["hard_stop_at"], name="window.hard_stop_at")
    cleanup_reserve = _positive_int(
        window["cleanup_reserve_s"], name="window.cleanup_reserve_s"
    )
    if not rlm_stop < measurement_stop < hard_stop:
        raise LoopCampaignError("Campaign window deadlines must be strictly ordered")
    if (hard_stop - measurement_stop).total_seconds() < cleanup_reserve:
        raise LoopCampaignError("Measurement stop does not preserve the cleanup reserve")

    upstreams = _require_exact_keys(
        document["upstreams"], _UPSTREAM_KEYS, name="upstreams"
    )
    exact_upstreams = {
        "rlm_source": RLM_SOURCE,
        "rlm_revision": RLM_REVISION,
        "rlm_version": RLM_VERSION,
        "halo_source": HALO_SOURCE,
        "halo_revision": HALO_REVISION,
        "halo_version": HALO_VERSION,
        "openai_agents_version": OPENAI_AGENTS_VERSION,
        "openai_version": OPENAI_VERSION,
        "pyarrow_version": PYARROW_VERSION,
        "babilong_source": BABILONG_SOURCE,
        "babilong_revision": BABILONG_REVISION,
    }
    if upstreams != exact_upstreams:
        raise LoopCampaignError("Campaign upstream pins do not match the protocol")

    rlm_value = document["rlm"]
    if not isinstance(rlm_value, dict):
        raise LoopCampaignError("rlm must be a table")
    rlm_unknown = set(rlm_value) - (_RLM_KEYS | _RLM_OPTIONAL_KEYS)
    rlm_missing = _RLM_KEYS - set(rlm_value)
    if rlm_unknown or rlm_missing:
        raise LoopCampaignError(
            "rlm keys do not match the frozen schema "
            f"(missing={sorted(rlm_missing)}, unknown={sorted(rlm_unknown)})"
        )
    rlm = rlm_value
    if not isinstance(rlm["model_profile"], str) or not rlm["model_profile"]:
        raise LoopCampaignError("rlm.model_profile is required")
    if rlm["reasoning_control"] != "fixed_unsupported":
        raise LoopCampaignError(
            "rlm.reasoning_control must be fixed_unsupported for the pinned model"
        )
    lengths = _string_list(rlm["lengths"], name="rlm.lengths")
    direct_lengths = _string_list(rlm["direct_lengths"], name="rlm.direct_lengths")
    tasks = _string_list(rlm["tasks"], name="rlm.tasks")
    rows = _int_list(rlm["row_indices"], name="rlm.row_indices")
    depth2_tasks = _string_list(
        rlm["recursive_depth2_tasks"], name="rlm.recursive_depth2_tasks"
    )
    depth2_lengths = _string_list(
        rlm["recursive_depth2_lengths"], name="rlm.recursive_depth2_lengths"
    )
    depth2_rows = _int_list(
        rlm["recursive_depth2_rows"], name="rlm.recursive_depth2_rows"
    )
    if not set(lengths) <= BABILONG_LENGTHS or not set(tasks) <= BABILONG_TASKS:
        raise LoopCampaignError("RLM BABILong selection is unsupported")
    if not set(direct_lengths) <= set(lengths):
        raise LoopCampaignError("Direct lengths must be a subset of RLM lengths")
    if not set(depth2_tasks) <= set(tasks) or not set(depth2_lengths) <= set(lengths):
        raise LoopCampaignError("Depth-2 selection must be a subset of the core matrix")
    if not set(depth2_rows) <= set(rows) or any(row >= 100 for row in rows):
        raise LoopCampaignError("BABILong row indices must be unique values below 100")
    for field in (
        "max_iterations",
        "max_concurrent_subcalls",
        "max_total_tokens",
        "max_output_tokens",
        "direct_timeout_s",
        "episode_timeout_s",
    ):
        _positive_int(rlm[field], name=f"rlm.{field}")
    if rlm["compaction"] is not True:
        raise LoopCampaignError("rlm.compaction must be true for the frozen protocol")
    if (
        type(rlm["compaction_threshold_pct"]) is not float
        or rlm["compaction_threshold_pct"] != RLM_COMPACTION_THRESHOLD_PCT
    ):
        raise LoopCampaignError(
            "rlm.compaction_threshold_pct must be exactly 0.85"
        )
    diagnostic = rlm.get("compaction_diagnostic")
    if diagnostic is not None:
        diagnostic = _require_exact_keys(
            diagnostic,
            _RLM_COMPACTION_DIAGNOSTIC_KEYS,
            name="rlm.compaction_diagnostic",
        )
        if (
            type(diagnostic["threshold_pct"]) is not float
            or diagnostic["threshold_pct"]
            != RLM_FORCED_COMPACTION_THRESHOLD_PCT
        ):
            raise LoopCampaignError(
                "rlm.compaction_diagnostic.threshold_pct must be exactly 0.20"
            )
        diagnostic_tasks = _string_list(
            diagnostic["tasks"], name="rlm.compaction_diagnostic.tasks"
        )
        diagnostic_lengths = _string_list(
            diagnostic["lengths"], name="rlm.compaction_diagnostic.lengths"
        )
        diagnostic_rows = _int_list(
            diagnostic["row_indices"],
            name="rlm.compaction_diagnostic.row_indices",
        )
        if (
            not set(diagnostic_tasks) <= set(tasks)
            or not set(diagnostic_lengths) <= set(lengths)
            or not set(diagnostic_rows) <= set(rows)
        ):
            raise LoopCampaignError(
                "RLM compaction diagnostic selectors must be core subsets"
            )
    if rlm["worker_isolation"] != "docker":
        raise LoopCampaignError("The frozen RLM protocol requires Docker worker isolation")

    halo = _require_exact_keys(document["halo"], _HALO_KEYS, name="halo")
    if halo["reasoning_effort"] != "none":
        raise LoopCampaignError(
            "halo.reasoning_effort must be none for the frozen control campaign"
        )
    halo_profiles = _string_list(halo["model_profiles"], name="halo.model_profiles")
    if len(halo_profiles) > 2:
        raise LoopCampaignError(
            "halo.model_profiles permits at most one fallback profile"
        )
    _int_list(halo["trace_counts"], name="halo.trace_counts", minimum=1)
    _int_list(halo["seeds"], name="halo.seeds")
    depths = _int_list(halo["depths"], name="halo.depths")
    if depths != [0, 1]:
        raise LoopCampaignError("HALO core depths must be exactly [0, 1]")
    _int_list(halo["depth2_trace_counts"], name="halo.depth2_trace_counts", minimum=1)
    _int_list(halo["depth2_seeds"], name="halo.depth2_seeds")
    for field in (
        "max_parallel",
        "max_turns",
        "max_output_tokens",
        "episode_timeout_s",
    ):
        _positive_int(halo[field], name=f"halo.{field}")
    return document


def _case_id(case: Mapping[str, Any]) -> str:
    prefix = str(case["phase"])
    return f"{prefix}-{_content_hash(case, 16)}"


def build_cases(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build a deterministic, counterbalanced scalar case list."""

    rlm = config["rlm"]
    cases: list[dict[str, Any]] = []
    lengths = list(rlm["lengths"])
    tasks = list(rlm["tasks"])
    direct_lengths = set(rlm["direct_lengths"])
    diagnostic = rlm.get("compaction_diagnostic")
    if diagnostic is not None:
        for row_index in diagnostic["row_indices"]:
            for length in diagnostic["lengths"]:
                for task in diagnostic["tasks"]:
                    case = {
                        "phase": "rlm",
                        "treatment": "rlm_depth1_forced_compaction",
                        "reasoning_control": rlm["reasoning_control"],
                        "context_length": length,
                        "task": task,
                        "row_index": row_index,
                        "replicate": list(rlm["row_indices"]).index(row_index),
                        "max_depth": 1,
                        "max_iterations": rlm["max_iterations"],
                        "max_concurrent_subcalls": rlm[
                            "max_concurrent_subcalls"
                        ],
                        "max_total_tokens": rlm["max_total_tokens"],
                        "max_output_tokens": rlm["max_output_tokens"],
                        "compaction": True,
                        "compaction_threshold_pct": diagnostic["threshold_pct"],
                        "admission_status": RLM_ADMISSION_ADMITTED,
                        "timeout_s": rlm["episode_timeout_s"],
                    }
                    cases.append({**case, "case_id": _case_id(case)})
    for replicate, row_index in enumerate(rlm["row_indices"]):
        length_order = lengths[replicate % len(lengths) :] + lengths[: replicate % len(lengths)]
        for length_index, length in enumerate(length_order):
            task_shift = (replicate + length_index) % len(tasks)
            task_order = tasks[task_shift:] + tasks[:task_shift]
            for task_index, task in enumerate(task_order):
                treatments = ["rlm_depth1"]
                if length in direct_lengths:
                    treatments.append("direct")
                if (replicate + length_index + task_index) % 2:
                    treatments.reverse()
                for treatment in treatments:
                    case = {
                        "phase": "rlm",
                        "treatment": treatment,
                        "reasoning_control": rlm["reasoning_control"],
                        "context_length": length,
                        "task": task,
                        "row_index": row_index,
                        "replicate": replicate,
                        "max_depth": 1 if treatment == "rlm_depth1" else None,
                        "max_iterations": rlm["max_iterations"],
                        "max_concurrent_subcalls": rlm["max_concurrent_subcalls"],
                        "max_total_tokens": rlm["max_total_tokens"],
                        "max_output_tokens": rlm["max_output_tokens"],
                        "compaction": (
                            rlm["compaction"] if treatment != "direct" else False
                        ),
                        "compaction_threshold_pct": (
                            rlm["compaction_threshold_pct"]
                            if treatment != "direct"
                            else None
                        ),
                        "admission_status": RLM_ADMISSION_ADMITTED,
                        "timeout_s": (
                            rlm["direct_timeout_s"]
                            if treatment == "direct"
                            else rlm["episode_timeout_s"]
                        ),
                    }
                    cases.append({**case, "case_id": _case_id(case)})

    for row_index in rlm["recursive_depth2_rows"]:
        for length in rlm["recursive_depth2_lengths"]:
            for task in rlm["recursive_depth2_tasks"]:
                case = {
                    "phase": "rlm",
                    "treatment": "rlm_depth2",
                    "reasoning_control": rlm["reasoning_control"],
                    "context_length": length,
                    "task": task,
                    "row_index": row_index,
                    "replicate": list(rlm["row_indices"]).index(row_index),
                    "max_depth": 2,
                    "max_iterations": rlm["max_iterations"],
                    "max_concurrent_subcalls": rlm["max_concurrent_subcalls"],
                    "max_total_tokens": rlm["max_total_tokens"],
                    "max_output_tokens": rlm["max_output_tokens"],
                    "compaction": rlm["compaction"],
                    "compaction_threshold_pct": rlm["compaction_threshold_pct"],
                    "admission_status": RLM_DEPTH2_HOLD,
                    "timeout_s": rlm["episode_timeout_s"],
                }
                cases.append({**case, "case_id": _case_id(case)})

    halo = config["halo"]
    for seed_index, seed in enumerate(halo["seeds"]):
        counts = list(halo["trace_counts"])
        count_order = counts[seed_index % len(counts) :] + counts[: seed_index % len(counts)]
        for count_index, trace_count in enumerate(count_order):
            depths = list(halo["depths"])
            if (seed_index + count_index) % 2:
                depths.reverse()
            for depth in depths:
                case = {
                    "phase": "halo",
                    "treatment": f"halo_depth{depth}",
                    "reasoning_effort": halo["reasoning_effort"],
                    "trace_count": trace_count,
                    "seed": seed,
                    "max_depth": depth,
                    "max_parallel": 1 if depth == 0 else halo["max_parallel"],
                    "max_turns": halo["max_turns"],
                    "max_output_tokens": halo["max_output_tokens"],
                    "timeout_s": halo["episode_timeout_s"],
                }
                cases.append({**case, "case_id": _case_id(case)})
    for trace_count in halo["depth2_trace_counts"]:
        for seed in halo["depth2_seeds"]:
            case = {
                "phase": "halo",
                "treatment": "halo_depth2",
                "reasoning_effort": halo["reasoning_effort"],
                "trace_count": trace_count,
                "seed": seed,
                "max_depth": 2,
                "max_parallel": halo["max_parallel"],
                "max_turns": halo["max_turns"],
                "max_output_tokens": halo["max_output_tokens"],
                "timeout_s": halo["episode_timeout_s"],
            }
            cases.append({**case, "case_id": _case_id(case)})
    if len({case["case_id"] for case in cases}) != len(cases):
        raise LoopCampaignError("Generated case identifiers are not unique")
    return cases


def _profile_plan_record(profile: Any) -> dict[str, Any]:
    record = asdict(profile)
    record["tasks"] = list(profile.tasks)
    record["args"] = list(profile.args)
    record["model_shards"] = [asdict(shard) for shard in profile.model_shards]
    return record


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _dataset_inventory(config: Mapping[str, Any]) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for length in config["rlm"]["lengths"]:
        for task in config["rlm"]["tasks"]:
            path = babilong_arrow_path(length, task)
            # The controller re-execs into the pinned loop environment before
            # planning, so this validates the full Arrow schema and row count.
            read_babilong_row(length, task, 0)
            selected.append(
                {
                    "context_length": length,
                    "task": task,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return {
        "source": BABILONG_SOURCE,
        "revision": BABILONG_REVISION,
        "rows_per_split": 100,
        "selected_files": selected,
    }


def _verify_loop_environment() -> None:
    probe = r'''
import importlib.metadata as metadata
import json

def record(name):
    dist = metadata.distribution(name)
    direct = dist.read_text("direct_url.json")
    return {"version": dist.version, "direct_url": json.loads(direct) if direct else None}

print(json.dumps({
    "rlms": record("rlms"),
    "halo-engine": record("halo-engine"),
    "openai-agents": record("openai-agents"),
    "openai": record("openai"),
    "pyarrow": record("pyarrow"),
}, sort_keys=True))
'''
    try:
        result = subprocess.run(
            [str(DEFAULT_LOOP_PYTHON), "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        installed = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        raise LoopCampaignError("Could not verify the pinned loop environment") from error
    expected_versions = {
        "rlms": RLM_VERSION,
        "halo-engine": HALO_VERSION,
        "openai-agents": OPENAI_AGENTS_VERSION,
        "openai": OPENAI_VERSION,
        "pyarrow": PYARROW_VERSION,
    }
    if result.returncode or not isinstance(installed, dict):
        raise LoopCampaignError("Could not verify the pinned loop environment")
    for name, expected in expected_versions.items():
        record = installed.get(name)
        if not isinstance(record, dict) or record.get("version") != expected:
            raise LoopCampaignError("Pinned loop environment versions have drifted")
    revision_expectations = {"rlms": RLM_REVISION, "halo-engine": HALO_REVISION}
    for name, expected_revision in revision_expectations.items():
        direct = installed[name].get("direct_url")
        commit = (
            direct.get("vcs_info", {}).get("commit_id")
            if isinstance(direct, dict)
            else None
        )
        if commit != expected_revision:
            raise LoopCampaignError("Pinned loop environment revisions have drifted")


def _verify_worker_image() -> None:
    result = subprocess.run(
        ["/usr/bin/docker", "image", "inspect", WORKER_IMAGE],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    if result.returncode:
        raise LoopCampaignError("The exact isolated-worker image is not cached")


def _repository_provenance(workspace: Path, *, require_clean: bool) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LoopCampaignError("Could not verify repository provenance") from error
    revision = head.stdout.strip()
    if (
        head.returncode
        or status.returncode
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        raise LoopCampaignError("Could not verify repository provenance")
    clean = not bool(status.stdout.strip())
    if require_clean and not clean:
        raise LoopCampaignError("Campaign planning requires a clean repository")
    return {"revision": revision, "clean": clean}


def _stage_worker_source(
    *, plan: Mapping[str, Any], run_dir: Path, workspace: Path, journal: Journal
) -> Path:
    """Export exact HEAD into an isolated mount with no ignored worktree data."""

    private_root = (run_dir / "private").resolve()
    private_root.mkdir(parents=True, exist_ok=True)
    destination = Path(
        tempfile.mkdtemp(prefix="worker-source-", dir=str(private_root))
    ).resolve()
    try:
        destination.relative_to(private_root)
    except ValueError as error:
        raise LoopCampaignError("Worker source staging escaped the private run root") from error
    archive = subprocess.Popen(
        ["git", "archive", "--format=tar", str(plan["repository"]["revision"])],
        cwd=workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if archive.stdout is None:
        archive.kill()
        raise LoopCampaignError("Could not stage exact worker source")
    extractor = subprocess.Popen(
        [
            "/usr/bin/tar",
            "--extract",
            "--file=-",
            "--directory",
            str(destination),
            "--no-same-owner",
            "--no-same-permissions",
        ],
        stdin=archive.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    archive.stdout.close()
    try:
        extractor_status = extractor.wait(timeout=60)
        archive_status = archive.wait(timeout=60)
    except subprocess.TimeoutExpired as error:
        extractor.kill()
        archive.kill()
        extractor.wait()
        archive.wait()
        raise LoopCampaignError("Exact worker source staging timed out") from error
    if archive_status or extractor_status:
        raise LoopCampaignError("Could not stage exact worker source")
    required = (destination / "loop_campaign.py", destination / "bench" / "loop_campaign.py")
    if not all(path.is_file() and not path.is_symlink() for path in required):
        raise LoopCampaignError("Exact worker source staging is incomplete")
    journal.append(
        {
            "event": "worker_source_staged",
            "repository_revision": plan["repository"]["revision"],
        }
    )
    return destination


def _verify_frozen_admission(plan: Mapping[str, Any]) -> None:
    """Recheck mutable local inputs before every run or resume."""

    _verify_loop_environment()
    _verify_worker_image()
    if _dataset_inventory(plan) != plan.get("dataset"):
        raise LoopCampaignError("Frozen BABILong artifacts have drifted since planning")


def _validate_reasoning_profiles(
    config: Mapping[str, Any], models: Mapping[str, Any]
) -> None:
    rlm_profile = models[str(config["rlm"]["model_profile"])]
    if getattr(rlm_profile, "request_body_json", None) is not None:
        raise LoopCampaignError(
            "Fixed-unsupported RLM profile must not advertise a reasoning request knob"
        )
    expected = {"chat_template_kwargs": {"enable_thinking": False}}
    for profile_id in config["halo"]["model_profiles"]:
        profile = models[str(profile_id)]
        try:
            request_default = json.loads(str(profile.request_body_json))
            arguments = list(profile.args)
            index = arguments.index("--default-chat-template-kwargs")
            server_default = json.loads(arguments[index + 1])
        except (AttributeError, IndexError, ValueError, json.JSONDecodeError) as error:
            raise LoopCampaignError(
                "HALO control profile lacks a valid thinking-off default"
            ) from error
        if request_default != expected or server_default != expected["chat_template_kwargs"]:
            raise LoopCampaignError(
                "HALO control profile must enforce enable_thinking=false"
            )


def _rlm_compaction_admission(
    rlm: Mapping[str, Any], *, served_context_tokens: int
) -> dict[str, Any]:
    """Return the frozen scalar context envelope for root-history compaction."""

    if rlm.get("compaction") is not True:
        raise LoopCampaignError("RLM root-history compaction is not admitted")
    threshold_pct = rlm.get("compaction_threshold_pct")
    output_tokens = rlm.get("max_output_tokens")
    if (
        type(threshold_pct) is not float
        or threshold_pct != RLM_COMPACTION_THRESHOLD_PCT
        or type(output_tokens) is not int
        or output_tokens <= 0
        or type(served_context_tokens) is not int
        or served_context_tokens <= 0
    ):
        raise LoopCampaignError("RLM compaction admission inputs are invalid")
    threshold_tokens = int(threshold_pct * RLM_COMPACTION_CONTEXT_TOKENS)
    headroom_tokens = served_context_tokens - threshold_tokens - output_tokens
    if headroom_tokens <= 0:
        raise LoopCampaignError("RLM compaction does not preserve output headroom")
    return {
        "enabled": True,
        "threshold_pct": threshold_pct,
        "package_context_tokens": RLM_COMPACTION_CONTEXT_TOKENS,
        "threshold_tokens": threshold_tokens,
        "served_context_tokens": served_context_tokens,
        "output_reserve_tokens": output_tokens,
        "headroom_tokens": headroom_tokens,
        "depth1_admitted": True,
        "depth2_admitted": False,
    }


def create_campaign_plan(
    *,
    campaign_path: Path,
    models_path: Path,
    results_root: Path,
) -> Path:
    """Freeze model profiles, cases, deadlines, and cached dataset admission."""

    config = load_campaign_manifest(campaign_path)
    models = load_models(models_path)
    profile_ids = [config["rlm"]["model_profile"], *config["halo"]["model_profiles"]]
    if len(set(profile_ids)) != len(profile_ids):
        raise LoopCampaignError("RLM and HALO profiles must be distinct")
    missing = [profile_id for profile_id in profile_ids if profile_id not in models]
    if missing:
        raise LoopCampaignError(f"Campaign references missing model profiles: {missing}")
    if "chat" not in models[config["rlm"]["model_profile"]].tasks:
        raise LoopCampaignError("RLM model profile lacks chat capability")
    for profile_id in config["halo"]["model_profiles"]:
        if "tools" not in models[profile_id].tasks:
            raise LoopCampaignError("Every HALO profile must declare tool capability")
    _validate_reasoning_profiles(config, models)
    rlm_profile = models[config["rlm"]["model_profile"]]
    compaction_admission = _rlm_compaction_admission(
        config["rlm"], served_context_tokens=int(rlm_profile.max_context)
    )
    if not DEFAULT_LOOP_PYTHON.is_file():
        raise LoopCampaignError("Pinned loop environment is absent")
    if not Path("/usr/bin/docker").is_file():
        raise LoopCampaignError("Docker is required for RLM worker isolation")
    _verify_loop_environment()
    _verify_worker_image()

    hard_stop = _parse_datetime(config["window"]["hard_stop_at"], name="hard_stop_at")
    if datetime.now(hard_stop.tzinfo) >= hard_stop:
        raise LoopCampaignError("The frozen campaign hard deadline has already passed")

    cases = build_cases(config)
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "campaign_id": config["id"],
        "description": config["description"],
        "created_at": utc_now(),
        "window": config["window"],
        "upstreams": config["upstreams"],
        "rlm": config["rlm"],
        "rlm_compaction_admission": compaction_admission,
        "halo": config["halo"],
        "models": {
            profile_id: _profile_plan_record(models[profile_id])
            for profile_id in profile_ids
        },
        "dataset": _dataset_inventory(config),
        "worker": {"isolation": "docker", "image": WORKER_IMAGE},
        "repository": _repository_provenance(
            Path(__file__).parents[1], require_clean=True
        ),
        "cases": cases,
    }
    plan["fingerprint"] = _content_hash(plan)
    plan["integrity_hash"] = _content_hash(plan)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_root / f"{stamp}-{config['id']}-{plan['fingerprint'][:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "plan.json", plan)
    return run_dir


def load_campaign_plan(run_dir: Path) -> dict[str, Any]:
    try:
        plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LoopCampaignError("Could not read the frozen campaign plan") from error
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") not in SUPPORTED_PLAN_SCHEMA_VERSIONS
    ):
        raise LoopCampaignError("Frozen campaign plan schema is unsupported")
    integrity_hash = plan.get("integrity_hash")
    payload = {key: value for key, value in plan.items() if key != "integrity_hash"}
    if not isinstance(integrity_hash, str) or _content_hash(payload) != integrity_hash:
        raise LoopCampaignError("Frozen campaign plan integrity check failed")
    fingerprint_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"fingerprint"}
    }
    if plan.get("fingerprint") != _content_hash(fingerprint_payload):
        raise LoopCampaignError("Frozen campaign fingerprint does not match")
    for case in plan.get("cases", []):
        if not isinstance(case, dict):
            raise LoopCampaignError("Frozen campaign case is invalid")
        raw = {key: value for key, value in case.items() if key != "case_id"}
        if case.get("case_id") != _case_id(raw):
            raise LoopCampaignError("Frozen campaign case identity does not match")
    if plan["schema_version"] == PLAN_SCHEMA_VERSION:
        _validate_v2_plan_semantics(plan)
    return plan


def _validate_v2_plan_semantics(plan: Mapping[str, Any]) -> None:
    if plan.get("protocol_version") != PROTOCOL_VERSION:
        raise LoopCampaignError("Frozen campaign protocol version is unsupported")
    rlm = plan.get("rlm")
    halo = plan.get("halo")
    if (
        not isinstance(rlm, dict)
        or rlm.get("reasoning_control") != "fixed_unsupported"
    ):
        raise LoopCampaignError("Frozen RLM reasoning control is invalid")
    if not isinstance(halo, dict) or halo.get("reasoning_effort") != "none":
        raise LoopCampaignError("Frozen HALO reasoning effort is invalid")
    try:
        profile = plan["models"][rlm["model_profile"]]
        expected_compaction = _rlm_compaction_admission(
            rlm, served_context_tokens=int(profile["max_context"])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LoopCampaignError("Frozen RLM compaction admission is invalid") from error
    if plan.get("rlm_compaction_admission") != expected_compaction:
        raise LoopCampaignError("Frozen RLM compaction admission has drifted")
    diagnostic = rlm.get("compaction_diagnostic")
    if diagnostic is not None:
        if (
            not isinstance(diagnostic, dict)
            or set(diagnostic) != _RLM_COMPACTION_DIAGNOSTIC_KEYS
            or diagnostic.get("threshold_pct")
            != RLM_FORCED_COMPACTION_THRESHOLD_PCT
            or not all(
                isinstance(diagnostic.get(name), list) and diagnostic[name]
                for name in ("tasks", "lengths", "row_indices")
            )
            or not set(diagnostic["tasks"]) <= set(rlm.get("tasks", ()))
            or not set(diagnostic["lengths"]) <= set(rlm.get("lengths", ()))
            or not set(diagnostic["row_indices"])
            <= set(rlm.get("row_indices", ()))
        ):
            raise LoopCampaignError(
                "Frozen RLM compaction diagnostic is invalid"
            )
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        raise LoopCampaignError("Frozen campaign cases are missing")
    for case in cases:
        phase = case.get("phase")
        if phase == "rlm":
            if (
                case.get("reasoning_control") != rlm["reasoning_control"]
                or "reasoning_effort" in case
            ):
                raise LoopCampaignError("Frozen RLM case reasoning control is invalid")
            treatment = case.get("treatment")
            expected_status = (
                RLM_DEPTH2_HOLD
                if treatment == "rlm_depth2"
                else RLM_ADMISSION_ADMITTED
            )
            if case.get("admission_status") != expected_status:
                raise LoopCampaignError("Frozen RLM case admission status is invalid")
            if treatment == "direct":
                if (
                    case.get("compaction") is not False
                    or case.get("compaction_threshold_pct") is not None
                ):
                    raise LoopCampaignError("Frozen direct case compaction is invalid")
            else:
                expected_threshold = rlm["compaction_threshold_pct"]
                if treatment == "rlm_depth1_forced_compaction":
                    diagnostic = rlm.get("compaction_diagnostic")
                    if not isinstance(diagnostic, dict):
                        raise LoopCampaignError(
                            "Frozen forced-compaction case lacks its diagnostic"
                        )
                    expected_threshold = diagnostic.get("threshold_pct")
                    if (
                        case.get("task") not in diagnostic.get("tasks", ())
                        or case.get("context_length")
                        not in diagnostic.get("lengths", ())
                        or case.get("row_index")
                        not in diagnostic.get("row_indices", ())
                    ):
                        raise LoopCampaignError(
                            "Frozen forced-compaction case selectors are invalid"
                        )
                if (
                    case.get("compaction") is not True
                    or case.get("compaction_threshold_pct")
                    != expected_threshold
                ):
                    raise LoopCampaignError("Frozen RLM case compaction is invalid")
        elif phase == "halo":
            if (
                case.get("reasoning_effort") != halo["reasoning_effort"]
                or "reasoning_control" in case
            ):
                raise LoopCampaignError("Frozen HALO case reasoning effort is invalid")
        else:
            raise LoopCampaignError("Frozen campaign case phase is invalid")


def babilong_arrow_path(context_length: str, task: str) -> Path:
    if context_length not in BABILONG_LENGTHS or task not in BABILONG_TASKS:
        raise LoopCampaignError("Unsupported BABILong path selection")
    root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "datasets"
        / "RMT-team___babilong"
    ).resolve()
    path = (
        root
        / context_length
        / "0.0.0"
        / BABILONG_REVISION
        / f"babilong-{task}.arrow"
    )
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise LoopCampaignError("Exact cached BABILong artifact is unavailable") from error
    if not resolved.is_file():
        raise LoopCampaignError("Exact cached BABILong artifact is not a file")
    return resolved


def read_babilong_row(context_length: str, task: str, row_index: int) -> dict[str, str]:
    """Read one public cached row without ever serializing it into the journal."""

    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as error:
        raise LoopCampaignError("The pinned pyarrow runtime is unavailable") from error
    if type(row_index) is not int or not 0 <= row_index < 100:
        raise LoopCampaignError("BABILong row index is out of range")
    path = babilong_arrow_path(context_length, task)
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.open_stream(source)
        except pa.ArrowInvalid:
            reader = ipc.open_file(source)
        table = reader.read_all()
    if table.column_names != ["input", "question", "target"] or table.num_rows != 100:
        raise LoopCampaignError("Cached BABILong Arrow schema has drifted")
    values = {
        name: table[name][row_index].as_py()
        for name in ("input", "question", "target")
    }
    if not all(isinstance(value, str) and value for value in values.values()):
        raise LoopCampaignError("Cached BABILong row contains invalid values")
    return values


def compare_babilong_answer(
    *, target: str, output: str, question: str, task: str
) -> bool:
    """Apply the official label-uniqueness semantics for selected BABILong tasks."""

    if task not in {"qa1", "qa2", "qa3", "qa4"}:
        raise LoopCampaignError("This campaign scorer supports BABILong qa1-qa4 only")
    normalized = output.lower().split(".", 1)[0]
    for marker in ("<context>", "<example>", "question"):
        normalized = normalized.split(marker, 1)[0]
    labels = set(BABILONG_LOCATION_LABELS)
    question_labels = {label for label in labels if label in question.lower()}
    output_labels = {label for label in labels if label in normalized} - question_labels
    return target.lower() in output_labels and len(output_labels) == 1


def _babilong_root_prompt(question: str) -> str:
    return (
        "Answer the question using only the hidden facts in the context variable. "
        "Track the relevant person or object through its latest moves. Return only "
        f"one location name and no explanation. Question: {question}"
    )


def _direct_babilong_prompt(context: str, question: str) -> str:
    return (
        "Answer the question using only the hidden facts in the context. Track the "
        "relevant person or object through its latest moves. Return only one location "
        f"name and no explanation.\n<context>\n{context}\n</context>\nQuestion: {question}"
    )


def _trace_identifier(seed: int, index: int) -> str:
    return hashlib.sha256(f"halo-trace-v1:{seed}:{index}".encode()).hexdigest()[:32]


def _span_identifier(trace_id: str, role: str) -> str:
    return hashlib.sha256(f"{trace_id}:{role}".encode()).hexdigest()[:16]


def _active_halo_families(seed: int) -> tuple[str, ...]:
    start = seed % len(HALO_FAILURE_FAMILIES)
    rotated = HALO_FAILURE_FAMILIES[start:] + HALO_FAILURE_FAMILIES[:start]
    return tuple(rotated[:4])


def _failure_span_fields(family: str) -> tuple[str, str, str, str]:
    records = {
        "search_timeout": (
            "graphiti.search",
            "STATUS_CODE_ERROR",
            "deadline exceeded while searching memory graph",
            '{"query":"project status","timeout_ms":5000}',
        ),
        "write_conflict": (
            "graphiti.write_episode",
            "STATUS_CODE_ERROR",
            "optimistic version conflict on memory write",
            '{"expected_version":3,"current_version":5}',
        ),
        "duplicate_edge": (
            "graphiti.add_edge",
            "STATUS_CODE_OK",
            "",
            '{"created":true,"duplicate_count":2,"deduplicated":false}',
        ),
        "entity_split": (
            "graphiti.resolve_entity",
            "STATUS_CODE_OK",
            "",
            '{"resolved":false,"canonical_candidates":2,"confidence":0.41}',
        ),
        "invalid_arguments": (
            "graphiti.search",
            "STATUS_CODE_ERROR",
            "invalid tool arguments: group_id is required",
            '{"error_type":"validation","missing_field":"group_id"}',
        ),
        "empty_retrieval": (
            "graphiti.search",
            "STATUS_CODE_OK",
            "",
            '{"expected_entity":true,"result_count":0,"fallback_used":false}',
        ),
    }
    try:
        return records[family]
    except KeyError as error:
        raise LoopCampaignError("Unknown synthetic HALO failure family") from error


def generate_halo_trace_fixture(
    path: Path, *, trace_count: int, seed: int
) -> dict[str, Any]:
    """Materialize deterministic synthetic OTel spans and return transient truth.

    An existing fixture is left byte-for-byte untouched.  That matters because
    HALO's sidecar index is keyed to the source file metadata and a resumed run
    must not invalidate an already completed index merely by rebuilding truth.
    """

    if type(trace_count) is not int or trace_count < 32 or type(seed) is not int or seed < 0:
        raise LoopCampaignError("Synthetic HALO fixture parameters are invalid")
    active = _active_halo_families(seed)
    assignments: dict[int, str] = {}
    family_width = max(2, trace_count // 40)
    cursor = (seed * 17) % trace_count
    for family in active:
        assigned = 0
        while assigned < family_width:
            index = cursor % trace_count
            cursor += 7
            if index in assignments:
                continue
            assignments[index] = family
            assigned += 1

    truth_ids: dict[str, list[str]] = {family: [] for family in active}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    stream = None
    if not path.is_file():
        stream = temporary.open("w", encoding="utf-8")
    try:
        for index in range(trace_count):
            trace_id = _trace_identifier(seed, index)
            root_span = _span_identifier(trace_id, "root")
            start_second = index % 50
            start_time = f"2026-08-20T12:00:{start_second:02d}.000000000Z"
            end_time = f"2026-08-20T12:00:{start_second:02d}.020000000Z"
            common_attributes = {
                "inference.schema_version": "1",
                "inference.project_id": f"synthetic-memory-{seed}",
                "inference.agent_name": "memory-agent",
                "inference.agent_id": f"agent-{seed}",
            }
            root = {
                "trace_id": trace_id,
                "span_id": root_span,
                "parent_span_id": "",
                "trace_state": "",
                "name": "memory.reflect",
                "kind": "SPAN_KIND_INTERNAL",
                "start_time": start_time,
                "end_time": end_time,
                "status": {"code": "STATUS_CODE_OK", "message": ""},
                "resource": {"attributes": {"service.name": "synthetic-graphiti"}},
                "scope": {"name": "sparkbench.synthetic", "version": "1"},
                "attributes": {
                    **common_attributes,
                    "openinference.span.kind": "AGENT",
                },
            }
            family = assignments.get(index)
            if family is None:
                tool_name, status_code, status_message, output_value = (
                    "graphiti.search",
                    "STATUS_CODE_OK",
                    "",
                    '{"expected_entity":true,"result_count":3,"fallback_used":false}',
                )
            else:
                tool_name, status_code, status_message, output_value = _failure_span_fields(family)
                truth_ids[family].append(trace_id)
            tool = {
                "trace_id": trace_id,
                "span_id": _span_identifier(trace_id, "tool"),
                "parent_span_id": root_span,
                "trace_state": "",
                "name": tool_name,
                "kind": "SPAN_KIND_INTERNAL",
                "start_time": start_time,
                "end_time": end_time,
                "status": {"code": status_code, "message": status_message},
                "resource": {"attributes": {"service.name": "synthetic-graphiti"}},
                "scope": {"name": "sparkbench.synthetic", "version": "1"},
                "attributes": {
                    **common_attributes,
                    "openinference.span.kind": "TOOL",
                    "tool.name": tool_name,
                    "input.value": '{"operation":"memory"}',
                    "output.value": output_value,
                },
            }
            if stream is not None:
                stream.write(_canonical_json(root) + "\n")
                stream.write(_canonical_json(tool) + "\n")
        if stream is not None:
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if stream is not None:
            stream.close()
    if temporary.is_file():
        temporary.replace(path)
    return {
        "active_families": list(active),
        "family_counts": {family: len(ids) for family, ids in truth_ids.items()},
        "family_trace_ids": truth_ids,
        "trace_count": trace_count,
        "span_count": trace_count * 2,
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def score_halo_answer(answer: str | None, truth: Mapping[str, Any]) -> dict[str, Any]:
    """Grade a transient HALO answer and return scalar-only measurements."""

    empty = {
        "json_valid": False,
        "predicted_family_count": 0,
        "family_precision": 0.0,
        "family_recall": 0.0,
        "family_f1": 0.0,
        "mean_count_accuracy": 0.0,
        "exact_count_rate": 0.0,
        "citation_precision": 0.0,
        "citation_family_coverage": 0.0,
    }
    if not isinstance(answer, str):
        return empty
    payload = _extract_json_object(answer)
    if (
        payload is None
        or set(payload) != {"families"}
        or not isinstance(payload.get("families"), list)
    ):
        return empty
    predictions: dict[str, tuple[int, list[str]]] = {}
    schema_valid = True
    for row in payload["families"]:
        if not isinstance(row, dict) or set(row) != {"id", "count", "example_trace_ids"}:
            schema_valid = False
            continue
        family = row.get("id")
        count = row.get("count")
        examples = row.get("example_trace_ids")
        if (
            family not in HALO_FAILURE_FAMILIES
            or type(count) is not int
            or count < 0
            or not isinstance(examples, list)
            or not all(isinstance(item, str) for item in examples)
            or family in predictions
        ):
            schema_valid = False
            continue
        predictions[family] = (count, list(examples))

    active = set(truth["active_families"])
    predicted = set(predictions)
    true_positive = len(active & predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(active) if active else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    count_scores: list[float] = []
    exact_counts = 0
    cited = sum(len(examples) for _, examples in predictions.values())
    valid_cited = 0
    covered = 0
    truth_counts = truth["family_counts"]
    truth_ids = truth["family_trace_ids"]
    for family in active:
        expected = int(truth_counts[family])
        predicted_count, examples = predictions.get(family, (0, []))
        count_scores.append(max(0.0, 1.0 - abs(predicted_count - expected) / max(expected, 1)))
        exact_counts += int(predicted_count == expected)
        family_valid = 0
        allowed = set(truth_ids[family])
        for example in examples:
            if example in allowed:
                valid_cited += 1
                family_valid += 1
        covered += int(family_valid > 0)
    return {
        "json_valid": schema_valid,
        "predicted_family_count": len(predicted),
        "family_precision": precision,
        "family_recall": recall,
        "family_f1": f1,
        "mean_count_accuracy": statistics.fmean(count_scores) if count_scores else 0.0,
        "exact_count_rate": exact_counts / len(active) if active else 0.0,
        "citation_precision": valid_cited / cited if cited else 0.0,
        "citation_family_coverage": covered / len(active) if active else 0.0,
    }


def _halo_prompt() -> str:
    allowed = ", ".join(HALO_FAILURE_FAMILIES)
    return (
        "Analyze this synthetic Graphiti-like trace dataset. Determine which of the "
        f"following failure families are actually present and count affected traces: {allowed}. "
        "Use trace tools to verify counts and examples; do not infer a family merely because it "
        "appears in this instruction. Return only one JSON object with exactly this shape: "
        '{"families":[{"id":"family","count":1,"example_trace_ids":["trace-id"]}]}. '
        "Omit absent families. Include one or two verified example trace IDs per present family."
    )


def parse_prometheus_counters(exposition: str) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for line in exposition.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _PROMETHEUS_SAMPLE.match(line)
        if not match or match.group("name") not in PROMETHEUS_COUNTERS:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if math.isfinite(value) and value >= 0:
            totals[match.group("name")] += value
    return dict(totals)


def snapshot_prometheus_counters(base_url: str, timeout_s: float = 3.0) -> dict[str, float]:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}:
        raise LoopCampaignError("Metrics endpoint must be literal loopback HTTP")
    root = base_url.rstrip("/").removesuffix("/v1")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(root + "/metrics", timeout=timeout_s) as response:
            exposition = response.read(8 * 1024 * 1024 + 1)
    except (OSError, urllib.error.URLError) as error:
        raise LoopCampaignError("Could not read local vLLM metrics") from error
    if len(exposition) > 8 * 1024 * 1024:
        raise LoopCampaignError("Local vLLM metrics response exceeded the bound")
    try:
        return parse_prometheus_counters(exposition.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise LoopCampaignError("Local vLLM metrics were not UTF-8") from error


def prometheus_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for name in PROMETHEUS_COUNTERS:
        if name not in before or name not in after:
            continue
        value = float(after[name]) - float(before[name])
        if value < 0 or not math.isfinite(value):
            raise LoopCampaignError("vLLM counter reset during an exclusive episode")
        delta[name] = value
    return delta


def _request_json(
    *, base_url: str, model: str, body: Mapping[str, Any], timeout_s: float
) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or parsed.port is None
        or parsed.path.rstrip("/") != "/v1"
        or parsed.query
        or parsed.fragment
    ):
        raise LoopCampaignError("Worker endpoint must be a literal loopback /v1 URL")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=_canonical_json(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_s) as response:
            payload = response.read(4 * 1024 * 1024 + 1)
    except (OSError, urllib.error.URLError) as error:
        raise LoopCampaignError("Local model request failed") from error
    if len(payload) > 4 * 1024 * 1024:
        raise LoopCampaignError("Local model response exceeded the worker bound")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LoopCampaignError("Local model response was invalid JSON") from error
    if not isinstance(value, dict):
        raise LoopCampaignError("Local model response was not an object")
    return value


def _worker_direct_babilong(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("reasoning_control") != "fixed_unsupported":
        raise LoopCampaignError("RLM reasoning control is unsupported")
    started = time.perf_counter()
    response = _request_json(
        base_url=str(payload["base_url"]),
        model=str(payload["model"]),
        timeout_s=float(payload["request_timeout_s"]),
        body={
            "model": payload["model"],
            "messages": [
                {
                    "role": "user",
                    "content": _direct_babilong_prompt(
                        str(payload["context"]), str(payload["question"])
                    ),
                }
            ],
            "temperature": 0.0,
            "max_tokens": 64,
        },
    )
    choices = response.get("choices")
    usage = response.get("usage")
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], dict)
        or not isinstance(choices[0].get("message"), dict)
        or not isinstance(choices[0]["message"].get("content"), str)
        or not isinstance(usage, dict)
    ):
        raise LoopCampaignError("Direct response omitted content or usage")
    answer = choices[0]["message"]["content"]
    correct = compare_babilong_answer(
        target=str(payload["target"]),
        output=answer,
        question=str(payload["question"]),
        task=str(payload["task"]),
    )
    result = {
        "status": "ok",
        "correct": correct,
        "wall_s": time.perf_counter() - started,
        "engine_wall_s": time.perf_counter() - started,
        "reported_calls": 1,
        "reported_input_tokens": _usage_int(usage, "prompt_tokens"),
        "reported_output_tokens": _usage_int(usage, "completion_tokens"),
        "usage_includes_recursive_children": True,
        "iterations": 0,
        "recursive_subcalls": 0,
        "output_chars": len(answer),
    }
    del answer, response
    return result


def _usage_int(usage: Mapping[str, Any], name: str) -> int:
    value = usage.get(name)
    if type(value) is not int or value < 0:
        raise LoopCampaignError("Model usage scalar is invalid")
    return value


def _worker_rlm_babilong(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("reasoning_control") != "fixed_unsupported":
        raise LoopCampaignError("RLM reasoning control is unsupported")
    threshold_pct = payload.get("compaction_threshold_pct")
    if payload.get("compaction") is not True or threshold_pct not in (
        RLM_COMPACTION_THRESHOLD_PCT,
        RLM_FORCED_COMPACTION_THRESHOLD_PCT,
    ):
        raise LoopCampaignError("RLM compaction controls are not admitted")
    try:
        from rlm import RLM
    except ImportError as error:
        raise LoopCampaignError("Pinned RLM package is unavailable") from error
    counters = {"iterations": 0, "recursive_subcalls": 0, "compactions": 0}

    class CountingRLM(RLM):
        def _completion_turn(self, *args: Any, **kwargs: Any) -> Any:
            counters["iterations"] += 1
            return super()._completion_turn(*args, **kwargs)

        def _compact_history(self, *args: Any, **kwargs: Any) -> Any:
            counters["compactions"] += 1
            return super()._compact_history(*args, **kwargs)

    def on_subcall_start(*_: Any) -> None:
        counters["recursive_subcalls"] += 1

    sampling = {
        "temperature": 0.0,
        "max_tokens": int(payload["max_output_tokens"]),
    }
    rlm = CountingRLM(
        backend="vllm",
        backend_kwargs={
            "api_key": "local",
            "model_name": str(payload["model"]),
            "base_url": str(payload["base_url"]),
            "max_retries": 0,
            "timeout": float(payload["request_timeout_s"]),
        },
        environment="local",
        max_depth=int(payload["max_depth"]),
        max_iterations=int(payload["max_iterations"]),
        max_timeout=float(payload["engine_timeout_s"]),
        max_tokens=int(payload["max_total_tokens"]),
        max_errors=3,
        max_concurrent_subcalls=int(payload["max_concurrent_subcalls"]),
        compaction=True,
        compaction_threshold_pct=float(threshold_pct),
        sampling_args=sampling,
        sub_sampling_args=sampling,
        on_subcall_start=on_subcall_start,
        verbose=False,
        logger=None,
        persistent=False,
    )
    started = time.perf_counter()
    result = rlm.completion(
        str(payload["context"]),
        root_prompt=_babilong_root_prompt(str(payload["question"])),
    )
    wall_s = time.perf_counter() - started
    if result.metadata is not None:
        raise LoopCampaignError("RLM unexpectedly retained trajectory metadata")
    answer = result.response
    correct = compare_babilong_answer(
        target=str(payload["target"]),
        output=answer,
        question=str(payload["question"]),
        task=str(payload["task"]),
    )
    calls = 0
    for usage in result.usage_summary.model_usage_summaries.values():
        calls += int(usage.total_calls)
    output = {
        "status": "ok",
        "correct": correct,
        "wall_s": wall_s,
        "engine_wall_s": float(result.execution_time),
        "reported_calls": calls,
        "reported_input_tokens": int(result.usage_summary.total_input_tokens),
        "reported_output_tokens": int(result.usage_summary.total_output_tokens),
        "usage_includes_recursive_children": False,
        "iterations": counters["iterations"],
        "recursive_subcalls": counters["recursive_subcalls"],
        "compaction_enabled": True,
        "compaction_threshold_pct": float(payload["compaction_threshold_pct"]),
        "compaction_count": counters["compactions"],
        "output_chars": len(answer),
    }
    del answer, result, rlm
    return output


def _worker_halo_index(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import asyncio
        from engine.traces.models.trace_index_config import TraceIndexConfig
        from engine.traces.trace_index_builder import TraceIndexBuilder
    except ImportError as error:
        raise LoopCampaignError("Pinned HALO package is unavailable") from error
    trace_path = Path(str(payload["trace_path"]))
    started = time.perf_counter()
    index_path = asyncio.run(
        TraceIndexBuilder.ensure_index_exists(
            trace_path=trace_path,
            config=TraceIndexConfig(),
        )
    )
    wall_s = time.perf_counter() - started
    meta_path = (
        index_path.with_name(index_path.name[: -len(".jsonl")] + ".meta.json")
        if index_path.name.endswith(".jsonl")
        else index_path.with_name(index_path.name + ".meta.json")
    )
    trace_count = None
    if meta_path.is_file():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if type(metadata.get("trace_count")) is int:
                trace_count = metadata["trace_count"]
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    return {
        "status": "ok",
        "index_wall_s": wall_s,
        "index_size_bytes": index_path.stat().st_size,
        "indexed_trace_count": trace_count,
    }


def _bounded_halo_chat_create(original: Any, max_output_tokens: int) -> Any:
    """Wrap the exact async OpenAI surface with deterministic HALO bounds."""

    async def create(resource: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs["temperature"] = 0.0
        bounded_key = None
        for key in ("max_completion_tokens", "max_tokens"):
            value = kwargs.get(key)
            if type(value) is int and value > 0:
                kwargs[key] = min(value, max_output_tokens)
                bounded_key = key
                break
        if bounded_key is None:
            kwargs.pop("max_completion_tokens", None)
            kwargs["max_tokens"] = max_output_tokens
        elif bounded_key == "max_completion_tokens":
            kwargs.pop("max_tokens", None)
        else:
            kwargs.pop("max_completion_tokens", None)
        return await original(resource, *args, **kwargs)

    return create


def _halo_subagent_builder_with_validation_recovery(
    original_builder: Any, validation_error_type: type[BaseException]
) -> Any:
    """Keep malformed ``call_subagent`` arguments model-visible and recoverable."""

    def build(*args: Any, **kwargs: Any) -> Any:
        tool = original_builder(*args, **kwargs)
        original_invoke = tool.on_invoke_tool

        async def invoke(context: Any, raw_arguments: str) -> Any:
            try:
                return await original_invoke(context, raw_arguments)
            except validation_error_type:
                return HALO_SUBAGENT_ARGUMENT_ERROR

        tool.on_invoke_tool = invoke
        return tool

    return build


def _worker_halo(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("reasoning_effort") != "none":
        raise LoopCampaignError("HALO reasoning effort is unsupported by this control")
    os.environ["OPENAI_AGENTS_DONT_LOG_MODEL_DATA"] = "1"
    os.environ["OPENAI_AGENTS_DONT_LOG_TOOL_DATA"] = "1"
    os.environ["OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA"] = "false"
    try:
        from agents import set_default_openai_api, set_trace_processors
        from engine.agents.agent_config import AgentConfig
        from engine.engine_config import EngineConfig
        from engine.main import stream_engine_output
        from engine.model_config import ModelConfig
        from engine.model_provider_config import ModelProviderConfig
        from engine.models.messages import AgentMessage
        from engine.sandbox.sandbox import Sandbox
        from engine.tools import subagent_tool_factory
        from engine.tools.subagent_result import SubagentToolResult
        from openai.resources.chat.completions import AsyncCompletions
        from pydantic import ValidationError
    except ImportError as error:
        raise LoopCampaignError("Pinned HALO runtime is unavailable") from error

    set_default_openai_api("chat_completions")
    set_trace_processors([])
    model = ModelConfig(
        name=str(payload["model"]),
        temperature=0.0,
        maximum_output_tokens=int(payload["max_output_tokens"]),
        parallel_tool_calls=True,
        reasoning_effort=None,
    )
    config = EngineConfig(
        root_agent=AgentConfig(
            name="root",
            model=model,
            maximum_turns=int(payload["max_turns"]),
            refusal_retries=0,
            final_answer_reprompts=1,
        ),
        subagent=AgentConfig(
            name="sub",
            model=model,
            maximum_turns=max(3, int(payload["max_turns"]) - 2),
            refusal_retries=0,
        ),
        synthesis_model=model,
        compaction_model=model,
        model_provider=ModelProviderConfig(
            base_url=str(payload["base_url"]),
            api_key="local",
        ),
        maximum_depth=int(payload["max_depth"]),
        maximum_parallel_subagents=int(payload["max_parallel"]),
        emit_run_checkpoints=False,
        dataset_context=(
            "A deterministic synthetic Graphiti-like memory-agent corpus. Each trace "
            "contains one AGENT span and one TOOL span. Failures may be explicit OTel "
            "errors or semantic anomalies inside output.value."
        ),
        repo_path=None,
    )
    tool_histogram: Counter[str] = Counter()
    child_ids: set[str] = set()
    completed_child_ids: set[str] = set()
    max_observed_depth = 0
    durable_items = 0
    assistant_items = 0
    child_turns = 0
    child_tool_calls = 0
    final_answer: str | None = None
    started = time.perf_counter()
    # HALO 0.3.5 otherwise prepares a Deno/Pyodide code sandbox lazily and may
    # download npm/wheel assets.  The frozen offline protocol deliberately
    # removes run_code; trace, aggregation, synthesis, and delegation tools
    # remain available and require no unpinned artifacts.
    from unittest.mock import patch

    bounded_create = _bounded_halo_chat_create(
        AsyncCompletions.create, int(payload["max_output_tokens"])
    )
    guarded_subagent_builder = _halo_subagent_builder_with_validation_recovery(
        subagent_tool_factory._build_subagent_as_tool,
        ValidationError,
    )
    with (
        patch.object(Sandbox, "get", return_value=None),
        patch.object(AsyncCompletions, "create", new=bounded_create),
        # HALO 0.3.5 replaces the SDK agent-as-tool invocation wrapper and loses
        # its validation-error handler. Keep only that model-input failure
        # recoverable while preserving every other exception and return value.
        patch.object(
            subagent_tool_factory,
            "_build_subagent_as_tool",
            new=guarded_subagent_builder,
        ),
    ):
        for event in stream_engine_output(
            [AgentMessage(role="user", content=_halo_prompt())],
            config,
            Path(str(payload["trace_path"])),
            telemetry=False,
        ):
            durable_items += 1
            max_observed_depth = max(max_observed_depth, event.depth)
            message = event.item
            assistant_items += int(message.role == "assistant")
            for call in message.tool_calls or ():
                tool_histogram[call.function.name] += 1
            if event.depth > 0:
                child_ids.add(event.agent_id)
            if (
                message.role == "tool"
                and message.name == "call_subagent"
                and isinstance(message.content, str)
            ):
                try:
                    child = SubagentToolResult.model_validate_json(message.content)
                except Exception:
                    child = None
                if child is not None:
                    child_ids.add(child.child_agent_id)
                    completed_child_ids.add(child.child_agent_id)
                    child_turns += int(child.turns_used)
                    child_tool_calls += int(child.tool_calls_made)
                    del child
            if event.final and event.depth == 0 and isinstance(message.content, str):
                final_answer = message.content
    wall_s = time.perf_counter() - started
    scores = score_halo_answer(final_answer, payload["truth"])
    output = {
        "status": "ok",
        "wall_s": wall_s,
        "durable_items": durable_items,
        "assistant_items": assistant_items,
        "tool_calls": sum(tool_histogram.values()),
        "subagent_requests": tool_histogram.get("call_subagent", 0),
        "observed_subagents": len(child_ids),
        "completed_subagents": len(completed_child_ids),
        "child_turns": child_turns,
        "child_tool_calls": child_tool_calls,
        "max_observed_depth": max_observed_depth,
        "root_finalized": final_answer is not None,
        "run_code_disabled": True,
        "final_answer_chars": len(final_answer) if final_answer is not None else 0,
        **scores,
    }
    del final_answer, config, model
    return output


def _read_worker_payload() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise LoopCampaignError("Worker input exceeded the bound")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LoopCampaignError("Worker input was invalid JSON") from error
    if not isinstance(value, dict):
        raise LoopCampaignError("Worker input must be a JSON object")
    return value


def _worker_entry(kind: str) -> int:
    result_fd = os.dup(sys.stdout.fileno())
    os.set_inheritable(result_fd, False)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, sys.stdout.fileno())
    os.dup2(null_fd, sys.stderr.fileno())
    os.close(null_fd)
    try:
        payload = _read_worker_payload()
        if kind == "direct":
            result = _worker_direct_babilong(payload)
        elif kind == "rlm":
            result = _worker_rlm_babilong(payload)
        elif kind == "halo-index":
            result = _worker_halo_index(payload)
        elif kind == "halo":
            result = _worker_halo(payload)
        else:
            raise LoopCampaignError("Unknown worker kind")
    except BaseException as error:
        result = _safe_error_result(error)
    os.write(result_fd, (_canonical_json(result) + "\n").encode())
    os.close(result_fd)
    return 0 if result.get("status") == "ok" else 1


def _docker_worker_command(
    worker_source: Path,
    *,
    worker_kind: str,
    isolation_plan: str,
    isolation_case: str,
) -> tuple[list[str], str]:
    if not re.fullmatch(r"[0-9a-f]{64}", isolation_plan):
        raise LoopCampaignError("Worker isolation plan identity is invalid")
    if not re.fullmatch(r"[a-z]+-[0-9a-f]{16}", isolation_case):
        raise LoopCampaignError("Worker isolation case identity is invalid")
    real_python = DEFAULT_LOOP_PYTHON.resolve(strict=True)
    runtime_root = real_python.parents[1]
    loop_environment = DEFAULT_LOOP_PYTHON.parents[1]
    container_name = "sparkbench-loop-worker-" + _content_hash(
        {"plan": isolation_plan, "case": isolation_case, "pid": os.getpid()}, 16
    )
    network_name = f"sparkbench-loop-{isolation_plan[:12]}"
    command = [
        "/usr/bin/docker",
        "run",
        "--rm",
        "--interactive",
        "--pull=never",
        "--name",
        container_name,
        "--label",
        "ai.sparkbench.loop-worker=true",
        "--label",
        f"ai.sparkbench.loop-plan={isolation_plan}",
        "--network",
        network_name,
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=1g",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=128",
        "--memory=8g",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--volume",
        f"{runtime_root}:/opt/python:ro",
        "--volume",
        f"{loop_environment}:/opt/loop-env:ro",
        "--volume",
        f"{worker_source}:/repo:ro",
        "--workdir",
        "/repo",
        "--env",
        "HOME=/tmp",
        "--env",
        "PYTHONNOUSERSITE=1",
        "--env",
        "PYTHONPATH=/opt/loop-env/lib/python3.12/site-packages:/repo",
        "--entrypoint",
        "/opt/python/bin/python3.12",
        WORKER_IMAGE,
        "/repo/loop_campaign.py",
        f"_worker-{worker_kind}",
    ]
    return command, container_name


def _host_worker_command(*, worker_kind: str) -> list[str]:
    return [str(DEFAULT_LOOP_PYTHON), str(Path(__file__).parents[1] / "loop_campaign.py"), f"_worker-{worker_kind}"]


def _terminate_worker_process(
    process: subprocess.Popen[bytes], *, container_name: str | None
) -> bool:
    """Remove an exact worker container and reap its attached client process."""

    cleanup_ok = True
    if container_name is not None:
        try:
            removed = subprocess.run(
                ["/usr/bin/docker", "rm", "-f", container_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            cleanup_ok = False
        else:
            cleanup_ok = removed.returncode == 0
    if process.poll() is not None:
        process.wait(timeout=0)
        return cleanup_ok
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cleanup_ok = False
    return cleanup_ok


def run_worker(
    *,
    worker_kind: str,
    payload: Mapping[str, Any],
    timeout_s: float,
    workspace: Path,
    isolated: bool,
    isolation_plan: str | None = None,
    isolation_case: str | None = None,
    worker_source: Path | None = None,
) -> dict[str, Any]:
    """Run one killable worker and expose only its final scalar JSON object."""

    if timeout_s <= 0:
        return {"status": "timeout"}
    if _STOP_REQUESTED:
        return {"status": "stopped"}
    container_name = None
    if isolated:
        if isolation_plan is None or isolation_case is None or worker_source is None:
            raise LoopCampaignError("Isolated workers require frozen identities")
        command, container_name = _docker_worker_command(
            worker_source,
            worker_kind=worker_kind,
            isolation_plan=isolation_plan,
            isolation_case=isolation_case,
        )
    else:
        command = _host_worker_command(worker_kind=worker_kind)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout_s
    worker_input: bytes | None = _canonical_json(payload).encode()
    while True:
        if _STOP_REQUESTED:
            cleanup_ok = _terminate_worker_process(
                process, container_name=container_name
            )
            return {
                "status": "stopped" if cleanup_ok else "error",
                **({} if cleanup_ok else {"error_type": "WorkerCleanupError"}),
            }
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            cleanup_ok = _terminate_worker_process(
                process, container_name=container_name
            )
            return {
                "status": "timeout" if cleanup_ok else "error",
                **({} if cleanup_ok else {"error_type": "WorkerCleanupError"}),
            }
        try:
            stdout, _ = process.communicate(
                input=worker_input,
                timeout=min(1.0, remaining),
            )
            break
        except subprocess.TimeoutExpired:
            # Python retains partially written input and captured output across
            # communicate() retries. Subsequent calls must not resend stdin.
            worker_input = None
    if len(stdout) > MAX_WORKER_OUTPUT_BYTES:
        return {"status": "error", "error_type": "WorkerOutputBound"}
    result: Any = None
    for line in reversed(stdout.decode("utf-8", errors="replace").splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            result = candidate
            break
    if not isinstance(result, dict) or result.get("status") not in {"ok", "error"}:
        return {"status": "error", "error_type": "WorkerProtocolError"}
    return result


def _resolved_image_reference(model: Mapping[str, Any]) -> str:
    image = model.get("image")
    digest = model.get("image_digest")
    if not isinstance(image, str) or not isinstance(digest, str):
        raise LoopCampaignError("Frozen vLLM profile lacks an image digest")
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode:
        raise LoopCampaignError("Frozen vLLM image is unavailable")
    try:
        repo_digests = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise LoopCampaignError("Could not verify the frozen vLLM image") from error
    expected = image.split(":", 1)[0] + "@" + digest
    matching = [item for item in repo_digests if item.endswith("@" + digest)]
    if not matching:
        raise LoopCampaignError("Local vLLM image digest does not match the plan")
    return matching[0] if matching else expected


def _model_namespace(
    model: Mapping[str, Any], *, run_identity: str
) -> SimpleNamespace:
    value = dict(model)
    value["run_identity"] = run_identity
    value["resolved_image"] = _resolved_image_reference(model)
    return SimpleNamespace(**value)


def _seconds_until(value: str) -> float:
    deadline = _parse_datetime(value, name="deadline")
    return (deadline - datetime.now(deadline.tzinfo)).total_seconds()


def _event_attempts(events: Iterable[Mapping[str, Any]]) -> Counter[str]:
    attempts: Counter[str] = Counter()
    for event in events:
        if event.get("event") == "case_started":
            case_id = event.get("case_id")
            if isinstance(case_id, str):
                attempts[case_id] += 1
    return attempts


def _completed_case_ids(events: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        str(event["case_id"])
        for event in events
        if event.get("event") == "case_complete" and isinstance(event.get("case_id"), str)
    }


_SAFE_ERROR_TEXT_FIELDS = frozenset(
    {
        "error_type",
        "error_cause_type",
        "error_code",
        "error_frame_file",
        "error_frame_function",
        "error_cause_frame_file",
        "error_cause_frame_function",
    }
)
_SAFE_ERROR_TEXT = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_SAFE_ERROR_CODES = frozenset(
    {
        "content_filter",
        "context_length_exceeded",
        "invalid_function_parameters",
        "missing_required_parameter",
        "string_above_max_length",
        "unknown_parameter",
    }
)


def _error_frame_fields(error: BaseException, *, prefix: str) -> dict[str, Any]:
    traceback = error.__traceback__
    if traceback is None:
        return {}
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    filename = Path(traceback.tb_frame.f_code.co_filename).name
    function = traceback.tb_frame.f_code.co_name
    if not _SAFE_ERROR_TEXT.fullmatch(filename) or not _SAFE_ERROR_TEXT.fullmatch(function):
        return {}
    return {
        f"{prefix}_frame_file": filename,
        f"{prefix}_frame_function": function,
        f"{prefix}_frame_line": traceback.tb_lineno,
    }


def _safe_error_result(error: BaseException) -> dict[str, Any]:
    """Return scalar diagnostics without exception prose, payloads, or local paths."""

    result: dict[str, Any] = {
        "status": "error",
        "error_type": type(error).__name__,
        **_error_frame_fields(error, prefix="error"),
    }
    chain = [error]
    seen = {id(error)}
    for _ in range(8):
        current = chain[-1]
        next_cause = current.__cause__ or current.__context__
        if next_cause is None or id(next_cause) in seen:
            break
        chain.append(next_cause)
        seen.add(id(next_cause))
    cause = chain[-1]
    if cause is not error:
        result["error_cause_type"] = type(cause).__name__
        result.update(_error_frame_fields(cause, prefix="error_cause"))

    for source in chain:
        for attribute, field in (
            ("tokens_used", "error_tokens_used"),
            ("token_limit", "error_token_limit"),
        ):
            value = getattr(source, attribute, None)
            if type(value) is int and value >= 0:
                result[field] = value
        status_code = getattr(source, "status_code", None)
        if type(status_code) is not int or status_code < 0:
            response = getattr(source, "response", None)
            status_code = getattr(response, "status_code", None)
        if type(status_code) is int and status_code >= 0:
            result["error_http_status"] = status_code
        code = getattr(source, "code", None)
        if isinstance(code, str) and code in _SAFE_ERROR_CODES:
            result["error_code"] = code
    return result


def _scalar_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that a worker result contains JSON scalar values only."""

    output: dict[str, Any] = {}
    for key, value in result.items():
        if key == "status":
            continue
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise LoopCampaignError("Worker returned an invalid scalar field")
        if value is None or isinstance(value, (str, bool)):
            if isinstance(value, str) and (
                key not in _SAFE_ERROR_TEXT_FIELDS
                or not _SAFE_ERROR_TEXT.fullmatch(value)
            ):
                raise LoopCampaignError("Worker returned non-allowlisted text")
            output[key] = value
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                raise LoopCampaignError("Worker returned a non-finite scalar")
            output[key] = value
            continue
        raise LoopCampaignError("Worker returned a non-scalar result")
    return output


def _case_metrics_payload(
    *,
    case: Mapping[str, Any],
    result: Mapping[str, Any],
    metrics_delta: Mapping[str, float],
) -> dict[str, Any]:
    payload = _scalar_result(result)
    prompt_tokens = metrics_delta.get("vllm:prompt_tokens_total")
    cached_tokens = metrics_delta.get("vllm:prompt_tokens_cached_total")
    generation_tokens = metrics_delta.get("vllm:generation_tokens_total")
    queries = metrics_delta.get("vllm:prefix_cache_queries_total")
    hits = metrics_delta.get("vllm:prefix_cache_hits_total")
    wall_s = payload.get("wall_s")
    payload.update(
        {
            "vllm_prompt_tokens": prompt_tokens,
            "vllm_cached_prompt_tokens": cached_tokens,
            "vllm_generation_tokens": generation_tokens,
            "vllm_successful_requests": metrics_delta.get("vllm:request_success_total"),
            "vllm_prefix_cache_queries": queries,
            "vllm_prefix_cache_hits": hits,
            "vllm_prefix_cache_hit_rate": (
                hits / queries if hits is not None and queries is not None and queries > 0 else None
            ),
            "effective_generation_tps": (
                generation_tokens / float(wall_s)
                if generation_tokens is not None
                and isinstance(wall_s, (int, float))
                and wall_s > 0
                else None
            ),
            "treatment": case["treatment"],
        }
    )
    return payload


def _stop_server(server: ManagedServer, log_path: Path) -> None:
    try:
        save_server_logs(server, log_path)
    finally:
        server.stop()


def _start_campaign_server(
    *,
    plan: Mapping[str, Any],
    profile_id: str,
    workspace: Path,
    run_dir: Path,
    journal: Journal,
    startup_budget_s: float | None = None,
) -> ManagedServer:
    run_identity = f"loop-{str(plan['fingerprint'])[:12]}-{profile_id}"
    recovery = recover_owned_vllm(run_identity)
    if recovery == "different_container_present":
        raise LoopCampaignError("A differently owned vLLM container is present")
    model = _model_namespace(plan["models"][profile_id], run_identity=run_identity)
    if startup_budget_s is not None:
        if startup_budget_s < 30:
            raise LoopCampaignError("Insufficient phase budget for server startup")
        model.startup_timeout_s = min(
            float(model.startup_timeout_s), float(startup_budget_s)
        )
    _preflight(model)
    journal.append(
        {
            "event": "server_starting",
            "profile_id": profile_id,
            "recovery": recovery,
        }
    )
    server = start_server(
        model,
        workspace=workspace,
        allow_download=False,
        server_log_path=run_dir / "server" / profile_id / "server.log",
    )
    journal.append(
        {
            "event": "server_ready",
            "profile_id": profile_id,
            "startup_s": server.startup_s,
        }
    )
    return server


def _run_rlm_case(
    *,
    case: Mapping[str, Any],
    plan: Mapping[str, Any],
    server: ManagedServer,
    workspace: Path,
    worker_source: Path,
    timeout_s: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    row = read_babilong_row(
        str(case["context_length"]), str(case["task"]), int(case["row_index"])
    )
    profile_id = plan["rlm"]["model_profile"]
    profile = plan["models"][profile_id]
    worker_base_url = (
        server.base_url
        if case["treatment"] == "direct"
        else "http://sparkbench-vllm:8000/v1"
    )
    payload: dict[str, Any] = {
        "context": row["input"],
        "question": row["question"],
        "target": row["target"],
        "task": case["task"],
        "reasoning_control": case["reasoning_control"],
        "base_url": worker_base_url,
        "model": profile["served_name"],
        "request_timeout_s": min(150.0, max(30.0, timeout_s - 10)),
    }
    if case["treatment"] != "direct":
        payload.update(
            {
                "max_depth": case["max_depth"],
                "max_iterations": case["max_iterations"],
                "max_concurrent_subcalls": case["max_concurrent_subcalls"],
                "max_total_tokens": case["max_total_tokens"],
                "max_output_tokens": case["max_output_tokens"],
                "compaction": case["compaction"],
                "compaction_threshold_pct": case["compaction_threshold_pct"],
                "engine_timeout_s": max(20.0, timeout_s - 15),
            }
        )
    before = snapshot_prometheus_counters(server.base_url)
    result = run_worker(
        worker_kind="direct" if case["treatment"] == "direct" else "rlm",
        payload=payload,
        timeout_s=timeout_s,
        workspace=workspace,
        isolated=case["treatment"] != "direct",
        isolation_plan=(
            str(plan["fingerprint"])
            if case["treatment"] != "direct"
            else None
        ),
        isolation_case=(
            str(case["case_id"])
            if case["treatment"] != "direct"
            else None
        ),
        worker_source=(worker_source if case["treatment"] != "direct" else None),
    )
    after = snapshot_prometheus_counters(server.base_url)
    del payload, row
    return result, prometheus_delta(before, after)


def _fixture_path(run_dir: Path, *, trace_count: int, seed: int) -> Path:
    return run_dir / "private" / f"halo-traces-n{trace_count}-s{seed}.jsonl"


def _ensure_halo_fixture_and_index(
    *,
    run_dir: Path,
    trace_count: int,
    seed: int,
    workspace: Path,
    journal: Journal,
) -> tuple[Path, dict[str, Any]]:
    path = _fixture_path(run_dir, trace_count=trace_count, seed=seed)
    truth = generate_halo_trace_fixture(path, trace_count=trace_count, seed=seed)
    marker = f"n{trace_count}-s{seed}"
    existing = {
        event.get("fixture_id")
        for event in journal.events()
        if event.get("event") == "halo_index_complete"
    }
    if marker not in existing:
        result = run_worker(
            worker_kind="halo-index",
            payload={"trace_path": str(path)},
            timeout_s=300,
            workspace=workspace,
            isolated=False,
        )
        if result.get("status") != "ok":
            raise LoopCampaignError("HALO trace indexing failed")
        journal.append(
            {
                "event": "halo_index_complete",
                "fixture_id": marker,
                "trace_count": trace_count,
                "seed": seed,
                **_scalar_result(result),
            }
        )
    return path, truth


def _run_halo_case(
    *,
    case: Mapping[str, Any],
    plan: Mapping[str, Any],
    profile_id: str,
    server: ManagedServer,
    run_dir: Path,
    workspace: Path,
    journal: Journal,
    timeout_s: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    trace_path, truth = _ensure_halo_fixture_and_index(
        run_dir=run_dir,
        trace_count=int(case["trace_count"]),
        seed=int(case["seed"]),
        workspace=workspace,
        journal=journal,
    )
    profile = plan["models"][profile_id]
    payload = {
        "trace_path": str(trace_path),
        "truth": truth,
        "base_url": server.base_url,
        "model": profile["served_name"],
        "reasoning_effort": case["reasoning_effort"],
        "max_depth": case["max_depth"],
        "max_parallel": case["max_parallel"],
        "max_turns": case["max_turns"],
        "max_output_tokens": case["max_output_tokens"],
    }
    before = snapshot_prometheus_counters(server.base_url)
    result = run_worker(
        worker_kind="halo",
        payload=payload,
        timeout_s=timeout_s,
        workspace=workspace,
        isolated=False,
    )
    after = snapshot_prometheus_counters(server.base_url)
    del payload, truth
    return result, prometheus_delta(before, after)


_STOP_REQUESTED = False


def _request_stop(_signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _case_dimensions(case: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "phase",
        "treatment",
        "reasoning_control",
        "reasoning_effort",
        "context_length",
        "task",
        "row_index",
        "replicate",
        "trace_count",
        "seed",
        "max_depth",
        "compaction",
        "compaction_threshold_pct",
        "admission_status",
    )
    return {key: case[key] for key in allowed if key in case}


def _terminal_case_ids(events: Iterable[Mapping[str, Any]]) -> set[str]:
    terminal_events = {
        "case_complete",
        "case_exhausted",
        "case_skipped_held",
        "case_skipped_deadline",
        "case_skipped_campaign_stop",
    }
    return {
        str(event["case_id"])
        for event in events
        if event.get("event") in terminal_events
        and isinstance(event.get("case_id"), str)
    }


def _journal_held_rlm_cases(
    *, plan: Mapping[str, Any], journal: Journal
) -> None:
    terminal = _terminal_case_ids(journal.events())
    for case in plan["cases"]:
        if (
            case.get("phase") == "rlm"
            and case.get("admission_status") == RLM_DEPTH2_HOLD
            and case["case_id"] not in terminal
        ):
            journal.append(
                {
                    "event": "case_skipped_held",
                    "case_id": case["case_id"],
                    **_case_dimensions(case),
                }
            )


def _journal_deadline_skips(
    *,
    plan: Mapping[str, Any],
    journal: Journal,
    phase: str,
    reason_event: str,
) -> None:
    terminal = _terminal_case_ids(journal.events())
    for case in plan["cases"]:
        if case["phase"] != phase or case["case_id"] in terminal:
            continue
        journal.append(
            {
                "event": reason_event,
                "case_id": case["case_id"],
                **_case_dimensions(case),
            }
        )


def _safe_stop_server(
    *, server: ManagedServer | None, profile_id: str, run_dir: Path, journal: Journal
) -> bool:
    if server is None:
        return True
    try:
        _stop_server(server, run_dir / "server" / profile_id / "server.log")
    except BaseException as error:
        journal.append(
            {
                "event": "server_stop_failed",
                "profile_id": profile_id,
                "error_type": type(error).__name__,
            }
        )
        return False
    else:
        journal.append({"event": "server_stopped", "profile_id": profile_id})
        return True


def _recover_plan_servers(plan: Mapping[str, Any], journal: Journal) -> None:
    """Recover only containers whose exact identities are frozen in this plan."""

    saw_different = False
    for profile_id in plan["models"]:
        run_identity = f"loop-{str(plan['fingerprint'])[:12]}-{profile_id}"
        recovery = recover_owned_vllm(run_identity)
        if recovery == "stopped_owned_container":
            journal.append(
                {
                    "event": "server_recovered",
                    "profile_id": profile_id,
                    "recovery": recovery,
                }
            )
            return
        if recovery == "already_absent":
            return
        saw_different = True
    if saw_different:
        raise LoopCampaignError("An unrelated or differently owned container is present")


def _recover_plan_workers(plan: Mapping[str, Any], journal: Journal) -> None:
    result = subprocess.run(
        [
            "/usr/bin/docker",
            "ps",
            "-aq",
            "--filter",
            "label=ai.sparkbench.loop-worker=true",
            "--filter",
            f"label=ai.sparkbench.loop-plan={plan['fingerprint']}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode:
        raise LoopCampaignError("Could not inspect isolated worker containers")
    container_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not container_ids:
        return
    removed = subprocess.run(
        ["/usr/bin/docker", "rm", "-f", *container_ids],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    if removed.returncode:
        raise LoopCampaignError("Could not recover exact-plan worker containers")
    journal.append({"event": "workers_recovered", "container_count": len(container_ids)})


def _worker_network_name(plan: Mapping[str, Any]) -> str:
    return f"sparkbench-loop-{str(plan['fingerprint'])[:12]}"


def _ensure_worker_network(plan: Mapping[str, Any], journal: Journal) -> str:
    name = _worker_network_name(plan)
    inspected = subprocess.run(
        ["/usr/bin/docker", "network", "inspect", name],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if inspected.returncode:
        created = subprocess.run(
            [
                "/usr/bin/docker",
                "network",
                "create",
                "--internal",
                "--label",
                "ai.sparkbench.loop-network=true",
                "--label",
                f"ai.sparkbench.loop-plan={plan['fingerprint']}",
                name,
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        if created.returncode:
            raise LoopCampaignError("Could not create the isolated worker network")
        inspected = subprocess.run(
            ["/usr/bin/docker", "network", "inspect", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    try:
        records = json.loads(inspected.stdout)
        record = records[0]
        labels = record["Labels"] or {}
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as error:
        raise LoopCampaignError("Could not verify the isolated worker network") from error
    if (
        inspected.returncode
        or record.get("Name") != name
        or record.get("Internal") is not True
        or labels.get("ai.sparkbench.loop-network") != "true"
        or labels.get("ai.sparkbench.loop-plan") != plan["fingerprint"]
    ):
        raise LoopCampaignError("Isolated worker network ownership does not match")
    journal.append({"event": "worker_network_ready"})
    return name


def _connect_server_to_worker_network(
    *, server: ManagedServer, network_name: str
) -> None:
    if not server.container_id:
        raise LoopCampaignError("RLM server has no managed container identity")
    connected = subprocess.run(
        [
            "/usr/bin/docker",
            "network",
            "connect",
            "--alias",
            "sparkbench-vllm",
            network_name,
            server.container_id,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )
    if connected.returncode:
        raise LoopCampaignError("Could not connect the RLM server to the worker network")


def _remove_worker_network(plan: Mapping[str, Any], journal: Journal) -> bool:
    name = _worker_network_name(plan)
    inspected = subprocess.run(
        [
            "/usr/bin/docker",
            "network",
            "inspect",
            "--format",
            '{{index .Labels "ai.sparkbench.loop-plan"}} {{.Internal}}',
            name,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if inspected.returncode:
        listed = subprocess.run(
            ["/usr/bin/docker", "network", "ls", "--format", "{{.Name}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if listed.returncode:
            journal.append({"event": "worker_network_cleanup_inspection_failed"})
            return False
        if name not in {line.strip() for line in listed.stdout.splitlines()}:
            return True
        journal.append({"event": "worker_network_cleanup_inspection_failed"})
        return False
    if inspected.stdout.strip() != f"{plan['fingerprint']} true":
        journal.append({"event": "worker_network_cleanup_refused"})
        return False
    removed = subprocess.run(
        ["/usr/bin/docker", "network", "rm", name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )
    journal.append(
        {
            "event": (
                "worker_network_removed"
                if removed.returncode == 0
                else "worker_network_cleanup_failed"
            )
        }
    )
    return removed.returncode == 0


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    return statistics.fmean(numbers) if numbers else None


def _complete_sum(
    observations: Iterable[Mapping[str, Any]], field: str
) -> float | None:
    rows = list(observations)
    values = [row.get(field) for row in rows]
    if not rows or not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in values
    ):
        return None
    return sum(float(value) for value in values)


def summarize_campaign(run_dir: Path) -> dict[str, Any]:
    """Build a deterministic scalar-only campaign summary from the journal."""

    plan = load_campaign_plan(run_dir)
    journal = Journal(run_dir / "journal.jsonl")
    events = journal.events()
    completed_by_id: dict[str, Mapping[str, Any]] = {}
    for event in events:
        case_id = event.get("case_id")
        if event.get("event") == "case_complete" and isinstance(case_id, str):
            completed_by_id[case_id] = event
    terminal = _terminal_case_ids(events)
    exhausted = {
        str(event["case_id"])
        for event in events
        if event.get("event") == "case_exhausted"
        and isinstance(event.get("case_id"), str)
    }
    deadline_skipped = {
        str(event["case_id"])
        for event in events
        if event.get("event") in {"case_skipped_deadline", "case_skipped_campaign_stop"}
        and isinstance(event.get("case_id"), str)
    }
    held = {
        str(event["case_id"])
        for event in events
        if event.get("event") == "case_skipped_held"
        and isinstance(event.get("case_id"), str)
    }

    groups: list[dict[str, Any]] = []
    rlm_profile = str(plan["rlm"]["model_profile"])
    halo_profiles = {
        str(event["profile_id"])
        for event in completed_by_id.values()
        if event.get("phase") == "halo" and isinstance(event.get("profile_id"), str)
    }
    selected_fallbacks = [
        str(event["profile_id"])
        for event in events
        if event.get("event") == "halo_fallback_selected"
        and isinstance(event.get("profile_id"), str)
    ]
    if selected_fallbacks:
        halo_profiles.add(selected_fallbacks[-1])
    if not halo_profiles:
        halo_profiles.add(str(plan["halo"]["model_profiles"][0]))
    group_keys = {
        (
            case["phase"],
            case["treatment"],
            rlm_profile,
            case.get("reasoning_control"),
            case.get("reasoning_effort"),
        )
        for case in plan["cases"]
        if case["phase"] == "rlm"
    }
    group_keys.update(
        (
            case["phase"],
            case["treatment"],
            profile_id,
            case.get("reasoning_control"),
            case.get("reasoning_effort"),
        )
        for case in plan["cases"]
        if case["phase"] == "halo"
        for profile_id in halo_profiles
    )
    for phase, treatment, profile_id, reasoning_control, reasoning_effort in sorted(
        group_keys, key=lambda item: tuple("" if value is None else str(value) for value in item)
    ):
        planned = [
            case
            for case in plan["cases"]
            if case["phase"] == phase
            and case["treatment"] == treatment
            and case.get("reasoning_control") == reasoning_control
            and case.get("reasoning_effort") == reasoning_effort
        ]
        observations = [
            completed_by_id[case["case_id"]]
            for case in planned
            if case["case_id"] in completed_by_id
            and completed_by_id[case["case_id"]].get("profile_id") == profile_id
        ]
        prompt_tokens = _complete_sum(observations, "vllm_prompt_tokens")
        cached_tokens = _complete_sum(observations, "vllm_cached_prompt_tokens")
        generation_tokens = _complete_sum(observations, "vllm_generation_tokens")
        wall_s = _complete_sum(observations, "wall_s")
        group: dict[str, Any] = {
            "phase": phase,
            "treatment": treatment,
            "profile_id": profile_id,
            "reasoning_control": reasoning_control,
            "reasoning_effort": reasoning_effort,
            "planned_cases": len(planned),
            "completed_cases": len(observations),
            "mean_wall_s": _mean(event.get("wall_s") for event in observations),
            "prompt_tokens": prompt_tokens,
            "cached_prompt_tokens": cached_tokens,
            "generation_tokens": generation_tokens,
            "cache_fraction": (
                cached_tokens / prompt_tokens
                if cached_tokens is not None
                and prompt_tokens is not None
                and prompt_tokens > 0
                else None
            ),
            "effective_generation_tps": (
                generation_tokens / wall_s
                if generation_tokens is not None
                and wall_s is not None
                and wall_s > 0
                else None
            ),
        }
        if phase == "rlm":
            correct = sum(int(event.get("correct") is True) for event in observations)
            group.update(
                {
                    "correct_cases": correct,
                    "accuracy": correct / len(observations) if observations else None,
                    "mean_reported_calls": _mean(
                        event.get("reported_calls") for event in observations
                    ),
                    "mean_vllm_successful_requests": _mean(
                        event.get("vllm_successful_requests")
                        for event in observations
                    ),
                }
            )
        else:
            group.update(
                {
                    "json_valid_rate": _mean(
                        int(event.get("json_valid") is True) for event in observations
                    ),
                    "mean_family_f1": _mean(
                        event.get("family_f1") for event in observations
                    ),
                    "mean_count_accuracy": _mean(
                        event.get("mean_count_accuracy") for event in observations
                    ),
                    "mean_citation_precision": _mean(
                        event.get("citation_precision") for event in observations
                    ),
                }
            )
        groups.append(group)

    completed = len(completed_by_id)
    planned_count = len(plan["cases"])
    last_start_index = max(
        (
            index
            for index, event in enumerate(events)
            if event.get("event") in {"campaign_started", "campaign_resumed"}
        ),
        default=-1,
    )
    latest_run_events = events[last_start_index + 1 :]
    last_cleanup_state = next(
        (
            str(event["event"])
            for event in reversed(latest_run_events)
            if event.get("event")
            in {"campaign_cleanup_verified", "campaign_cleanup_failed"}
        ),
        None,
    )
    cleanup_verified_for_latest_run = last_cleanup_state == "campaign_cleanup_verified"
    if last_cleanup_state == "campaign_cleanup_failed":
        status = "cleanup_failed"
    elif completed == planned_count and cleanup_verified_for_latest_run:
        status = "complete"
    elif completed == planned_count:
        status = "measurements_complete_cleanup_pending"
    elif terminal:
        status = "partial"
    else:
        status = "not_started"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "campaign_id": plan["campaign_id"],
        "plan_fingerprint": plan["fingerprint"],
        "status": status,
        "planned_cases": planned_count,
        "completed_cases": completed,
        "exhausted_cases": len(exhausted),
        "held_cases": len(held),
        "deadline_skipped_cases": len(deadline_skipped),
        "failed_attempts": sum(
            event.get("event") in {"case_failed", "case_timeout"} for event in events
        ),
        "groups": groups,
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def _record_case_result(
    *,
    case: Mapping[str, Any],
    profile_id: str,
    attempt: int,
    result: Mapping[str, Any],
    metrics_delta: Mapping[str, float],
    journal: Journal,
) -> bool:
    base = {
        "case_id": case["case_id"],
        "profile_id": profile_id,
        "attempt": attempt,
        **_case_dimensions(case),
    }
    if result.get("status") == "ok":
        journal.append(
            {
                "event": "case_complete",
                **base,
                **_case_metrics_payload(
                    case=case, result=result, metrics_delta=metrics_delta
                ),
            }
        )
        return True
    if result.get("status") == "stopped":
        journal.append({"event": "case_skipped_campaign_stop", **base})
        return False
    event_name = "case_timeout" if result.get("status") == "timeout" else "case_failed"
    failure: dict[str, Any] = {"event": event_name, **base}
    if event_name == "case_failed":
        diagnostics = _scalar_result(result)
        diagnostics.setdefault("error_type", "WorkerError")
        failure.update(diagnostics)
    journal.append(failure)
    return False


def _run_phase_case(
    *,
    case: Mapping[str, Any],
    plan: Mapping[str, Any],
    profile_id: str,
    server: ManagedServer,
    run_dir: Path,
    workspace: Path,
    worker_source: Path | None,
    journal: Journal,
    cutoff_name: str,
) -> tuple[bool, bool]:
    """Run one episode; return ``(complete, restart_server)``."""

    remaining = _seconds_until(str(plan["window"][cutoff_name]))
    timeout_s = min(float(case["timeout_s"]), max(0.0, remaining - 75.0))
    if timeout_s < 15:
        return False, False
    attempts = _event_attempts(journal.events())
    attempt = attempts[case["case_id"]] + 1
    journal.append(
        {
            "event": "case_started",
            "case_id": case["case_id"],
            "profile_id": profile_id,
            "attempt": attempt,
            "timeout_s": timeout_s,
            **_case_dimensions(case),
        }
    )
    try:
        if case["phase"] == "rlm":
            if worker_source is None:
                raise LoopCampaignError("RLM worker source was not staged")
            result, metrics_delta = _run_rlm_case(
                case=case,
                plan=plan,
                server=server,
                workspace=workspace,
                worker_source=worker_source,
                timeout_s=timeout_s,
            )
        else:
            result, metrics_delta = _run_halo_case(
                case=case,
                plan=plan,
                profile_id=profile_id,
                server=server,
                run_dir=run_dir,
                workspace=workspace,
                journal=journal,
                timeout_s=timeout_s,
            )
    except BaseException as error:
        result = _safe_error_result(error)
        metrics_delta = {}
    complete = _record_case_result(
        case=case,
        profile_id=profile_id,
        attempt=attempt,
        result=result,
        metrics_delta=metrics_delta,
        journal=journal,
    )
    summarize_campaign(run_dir)
    return complete, not complete and result.get("status") != "stopped"


def _phase_cases(plan: Mapping[str, Any], phase: str) -> list[Mapping[str, Any]]:
    return [case for case in plan["cases"] if case["phase"] == phase]


def _needs_server_restart(*, pending_count: int, current_attempt: int) -> bool:
    """Avoid a cold restart when the final pending case just exhausted retries."""

    if pending_count <= 0 or current_attempt <= 0:
        raise ValueError("pending_count and current_attempt must be positive")
    return current_attempt < 2 or pending_count > 1


def _prepare_halo_fixtures(
    *, plan: Mapping[str, Any], run_dir: Path, workspace: Path, journal: Journal
) -> None:
    selections = sorted(
        {
            (int(case["trace_count"]), int(case["seed"]))
            for case in _phase_cases(plan, "halo")
        }
    )
    for trace_count, seed in selections:
        if _STOP_REQUESTED or _seconds_until(plan["window"]["measurement_stop_at"]) < 330:
            return
        _ensure_halo_fixture_and_index(
            run_dir=run_dir,
            trace_count=trace_count,
            seed=seed,
            workspace=workspace,
            journal=journal,
        )


def _run_rlm_phase(
    *, plan: Mapping[str, Any], run_dir: Path, workspace: Path,
    worker_source: Path, journal: Journal,
    telemetry: TelemetrySampler
) -> None:
    phase = "rlm"
    cutoff_name = "rlm_stop_at"
    _journal_held_rlm_cases(plan=plan, journal=journal)
    if _STOP_REQUESTED or _seconds_until(plan["window"][cutoff_name]) < 120:
        _journal_deadline_skips(
            plan=plan,
            journal=journal,
            phase=phase,
            reason_event=(
                "case_skipped_campaign_stop"
                if _STOP_REQUESTED
                else "case_skipped_deadline"
            ),
        )
        return
    if not any(
        case["case_id"] not in _terminal_case_ids(journal.events())
        for case in _phase_cases(plan, phase)
    ):
        return
    profile_id = str(plan["rlm"]["model_profile"])
    server: ManagedServer | None = None
    network_ready = False
    try:
        network_name = _ensure_worker_network(plan, journal)
        network_ready = True
        telemetry.set_phase("rlm_server_start")
        server = _start_campaign_server(
            plan=plan,
            profile_id=profile_id,
            workspace=workspace,
            run_dir=run_dir,
            journal=journal,
            startup_budget_s=_seconds_until(plan["window"][cutoff_name]) - 90,
        )
        _connect_server_to_worker_network(server=server, network_name=network_name)
        telemetry.set_phase("rlm_cases")
        while True:
            events = journal.events()
            terminal = _terminal_case_ids(events)
            pending = [
                case
                for case in _phase_cases(plan, phase)
                if case["case_id"] not in terminal
            ]
            if not pending or _STOP_REQUESTED:
                break
            case = pending[0]
            attempts = _event_attempts(events)[case["case_id"]]
            if attempts >= 2:
                journal.append(
                    {
                        "event": "case_exhausted",
                        "case_id": case["case_id"],
                        **_case_dimensions(case),
                    }
                )
                continue
            if _seconds_until(plan["window"][cutoff_name]) < 90:
                break
            complete, restart = _run_phase_case(
                case=case,
                plan=plan,
                profile_id=profile_id,
                server=server,
                run_dir=run_dir,
                workspace=workspace,
                worker_source=worker_source,
                journal=journal,
                cutoff_name=cutoff_name,
            )
            if restart and not complete:
                if attempts + 1 >= 2:
                    journal.append(
                        {
                            "event": "case_exhausted",
                            "case_id": case["case_id"],
                            **_case_dimensions(case),
                        }
                    )
                if not _needs_server_restart(
                    pending_count=len(pending), current_attempt=attempts + 1
                ):
                    break
                telemetry.set_phase("rlm_server_restart")
                if not _safe_stop_server(
                    server=server,
                    profile_id=profile_id,
                    run_dir=run_dir,
                    journal=journal,
                ):
                    raise LoopCampaignError("RLM server restart cleanup failed")
                server = None
                if _seconds_until(plan["window"][cutoff_name]) < 180:
                    break
                server = _start_campaign_server(
                    plan=plan,
                    profile_id=profile_id,
                    workspace=workspace,
                    run_dir=run_dir,
                    journal=journal,
                    startup_budget_s=_seconds_until(plan["window"][cutoff_name]) - 90,
                )
                _connect_server_to_worker_network(
                    server=server, network_name=network_name
                )
                telemetry.set_phase("rlm_cases")
    finally:
        telemetry.set_phase("rlm_cleanup")
        stop_ok = _safe_stop_server(
            server=server,
            profile_id=profile_id,
            run_dir=run_dir,
            journal=journal,
        )
        network_ok = True
        if network_ready:
            network_ok = _remove_worker_network(plan, journal)
        if not stop_ok or not network_ok:
            raise LoopCampaignError("RLM phase cleanup was not verified")
    _journal_deadline_skips(
        plan=plan,
        journal=journal,
        phase=phase,
        reason_event=(
            "case_skipped_campaign_stop" if _STOP_REQUESTED else "case_skipped_deadline"
        ),
    )


def _start_first_halo_profile(
    *,
    plan: Mapping[str, Any],
    run_dir: Path,
    workspace: Path,
    journal: Journal,
    cutoff_name: str,
    profile_ids: Iterable[str],
) -> tuple[str, ManagedServer]:
    for profile_id in profile_ids:
        try:
            server = _start_campaign_server(
                plan=plan,
                profile_id=str(profile_id),
                workspace=workspace,
                run_dir=run_dir,
                journal=journal,
                startup_budget_s=_seconds_until(plan["window"][cutoff_name]) - 90,
            )
        except BaseException as error:
            journal.append(
                {
                    "event": "halo_profile_rejected",
                    "profile_id": profile_id,
                    "error_type": type(error).__name__,
                }
            )
            continue
        return str(profile_id), server
    raise LoopCampaignError("No frozen HALO serving profile admitted")


def _halo_profile_candidates(
    plan: Mapping[str, Any], events: Iterable[Mapping[str, Any]]
) -> list[str]:
    ordered = [str(item) for item in plan["halo"]["model_profiles"]]
    event_list = list(events)
    completed_profiles = [
        str(event["profile_id"])
        for event in event_list
        if event.get("event") == "case_complete"
        and event.get("phase") == "halo"
        and isinstance(event.get("profile_id"), str)
    ]
    if completed_profiles:
        selected = completed_profiles[-1]
        if selected not in ordered:
            raise LoopCampaignError("HALO journal references an unfrozen profile")
        return [selected]
    fallback_events = [
        str(event["profile_id"])
        for event in event_list
        if event.get("event") == "halo_fallback_selected"
        and isinstance(event.get("profile_id"), str)
    ]
    primary_failed = any(
        event.get("event") in {"case_failed", "case_timeout"}
        and event.get("phase") == "halo"
        and event.get("profile_id") == ordered[0]
        for event in event_list
    )
    if fallback_events:
        selected = fallback_events[-1]
        if selected not in ordered[1:]:
            raise LoopCampaignError("HALO fallback journal identity is invalid")
        return [selected]
    if primary_failed and len(ordered) > 1:
        return [ordered[1]]
    return ordered


def _run_halo_phase(
    *, plan: Mapping[str, Any], run_dir: Path, workspace: Path, journal: Journal,
    telemetry: TelemetrySampler
) -> None:
    phase = "halo"
    cutoff_name = "measurement_stop_at"
    if _STOP_REQUESTED or _seconds_until(plan["window"][cutoff_name]) < 420:
        _journal_deadline_skips(
            plan=plan,
            journal=journal,
            phase=phase,
            reason_event=(
                "case_skipped_campaign_stop" if _STOP_REQUESTED else "case_skipped_deadline"
            ),
        )
        return
    if not any(
        case["case_id"] not in _terminal_case_ids(journal.events())
        for case in _phase_cases(plan, phase)
    ):
        return
    telemetry.set_phase("halo_index")
    _prepare_halo_fixtures(
        plan=plan, run_dir=run_dir, workspace=workspace, journal=journal
    )
    if _STOP_REQUESTED or _seconds_until(plan["window"][cutoff_name]) < 180:
        _journal_deadline_skips(
            plan=plan, journal=journal, phase=phase, reason_event="case_skipped_deadline"
        )
        return

    profile_id = ""
    server: ManagedServer | None = None
    ordered_profiles = [str(item) for item in plan["halo"]["model_profiles"]]
    candidates = _halo_profile_candidates(plan, journal.events())
    fallback_used = candidates[0] != ordered_profiles[0]
    try:
        telemetry.set_phase("halo_server_start")
        profile_id, server = _start_first_halo_profile(
            plan=plan,
            run_dir=run_dir,
            workspace=workspace,
            journal=journal,
            cutoff_name=cutoff_name,
            profile_ids=candidates,
        )
        if profile_id != ordered_profiles[0] and not any(
            event.get("event") == "halo_fallback_selected"
            for event in journal.events()
        ):
            journal.append(
                {"event": "halo_fallback_selected", "profile_id": profile_id}
            )
        fallback_used = profile_id != ordered_profiles[0]
        telemetry.set_phase("halo_cases")
        while True:
            events = journal.events()
            terminal = _terminal_case_ids(events)
            pending = [
                case
                for case in _phase_cases(plan, phase)
                if case["case_id"] not in terminal
            ]
            if not pending or _STOP_REQUESTED:
                break
            case = pending[0]
            attempts = _event_attempts(events)[case["case_id"]]
            if attempts >= 2:
                journal.append(
                    {
                        "event": "case_exhausted",
                        "case_id": case["case_id"],
                        **_case_dimensions(case),
                    }
                )
                continue
            if _seconds_until(plan["window"][cutoff_name]) < 90:
                break
            complete, restart = _run_phase_case(
                case=case,
                plan=plan,
                profile_id=profile_id,
                server=server,
                run_dir=run_dir,
                workspace=workspace,
                worker_source=None,
                journal=journal,
                cutoff_name=cutoff_name,
            )
            if restart and not complete:
                if attempts + 1 >= 2:
                    journal.append(
                        {
                            "event": "case_exhausted",
                            "case_id": case["case_id"],
                            **_case_dimensions(case),
                        }
                    )
                if not _needs_server_restart(
                    pending_count=len(pending), current_attempt=attempts + 1
                ):
                    break
                telemetry.set_phase("halo_server_restart")
                if not _safe_stop_server(
                    server=server,
                    profile_id=profile_id,
                    run_dir=run_dir,
                    journal=journal,
                ):
                    raise LoopCampaignError("HALO server restart cleanup failed")
                server = None
                halo_completed = any(
                    event.get("event") == "case_complete"
                    and event.get("phase") == "halo"
                    for event in journal.events()
                )
                if (
                    not fallback_used
                    and not halo_completed
                    and profile_id == ordered_profiles[0]
                    and len(ordered_profiles) > 1
                ):
                    fallback_used = True
                    profile_id = ordered_profiles[1]
                    journal.append(
                        {
                            "event": "halo_fallback_selected",
                            "profile_id": profile_id,
                        }
                    )
                if _seconds_until(plan["window"][cutoff_name]) < 180:
                    break
                server = _start_campaign_server(
                    plan=plan,
                    profile_id=profile_id,
                    workspace=workspace,
                    run_dir=run_dir,
                    journal=journal,
                    startup_budget_s=_seconds_until(plan["window"][cutoff_name]) - 90,
                )
                telemetry.set_phase("halo_cases")
    finally:
        telemetry.set_phase("halo_cleanup")
        if not _safe_stop_server(
            server=server,
            profile_id=profile_id,
            run_dir=run_dir,
            journal=journal,
        ):
            raise LoopCampaignError("HALO phase cleanup was not verified")
    _journal_deadline_skips(
        plan=plan,
        journal=journal,
        phase=phase,
        reason_event=(
            "case_skipped_campaign_stop" if _STOP_REQUESTED else "case_skipped_deadline"
        ),
    )


def cleanup_campaign(*, run_dir: Path, workspace: Path) -> dict[str, Any]:
    """Idempotently remove only resources owned by one valid frozen plan."""

    plan = load_campaign_plan(run_dir)
    journal = Journal(run_dir / "journal.jsonl")
    lock_path = results_lock_path(workspace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LoopCampaignError(
                "Another SparkBench run holds the results lock"
            ) from error
        failures: list[BaseException] = []
        try:
            _recover_plan_workers(plan, journal)
        except BaseException as error:
            failures.append(error)
        try:
            _recover_plan_servers(plan, journal)
        except BaseException as error:
            failures.append(error)
        try:
            network_ok = _remove_worker_network(plan, journal)
        except BaseException as error:
            failures.append(error)
        else:
            if not network_ok:
                failures.append(
                    LoopCampaignError("Exact-plan worker network cleanup failed")
                )
        if failures:
            journal.append(
                {
                    "event": "campaign_cleanup_failed",
                    "error_type": type(failures[0]).__name__,
                    "failure_count": len(failures),
                }
            )
            raise LoopCampaignError("Exact-plan cleanup was not verified") from failures[0]
        journal.append({"event": "campaign_cleanup_verified"})
    return {"status": "cleanup_verified"}


def execute_campaign(*, run_dir: Path, workspace: Path) -> dict[str, Any]:
    """Execute or resume a frozen campaign under the repository-wide lock."""

    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    plan = load_campaign_plan(run_dir)
    repository = _repository_provenance(workspace, require_clean=True)
    if repository != plan.get("repository"):
        raise LoopCampaignError("Repository provenance no longer matches the frozen plan")
    if _seconds_until(plan["window"]["hard_stop_at"]) <= 0:
        raise LoopCampaignError("The campaign hard deadline has passed")
    _verify_frozen_admission(plan)
    journal = Journal(run_dir / "journal.jsonl")
    lock_path = results_lock_path(workspace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry = TelemetrySampler(run_dir / "telemetry.jsonl", interval_s=2.0)
    previous_handlers: dict[int, Any] = {}
    with lock_path.open("a+") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LoopCampaignError("Another SparkBench run holds the results lock") from error
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, _request_stop)
        telemetry.start()
        journal.append(
            {
                "event": "campaign_resumed"
                if journal.events()
                else "campaign_started",
                "plan_fingerprint": plan["fingerprint"],
            }
        )
        campaign_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        try:
            _recover_plan_workers(plan, journal)
            _recover_plan_servers(plan, journal)
            if not _remove_worker_network(plan, journal):
                raise LoopCampaignError("Pre-run worker network cleanup failed")
            worker_source = _stage_worker_source(
                plan=plan,
                run_dir=run_dir,
                workspace=workspace,
                journal=journal,
            )
            _run_rlm_phase(
                plan=plan,
                run_dir=run_dir,
                workspace=workspace,
                worker_source=worker_source,
                journal=journal,
                telemetry=telemetry,
            )
            if not _STOP_REQUESTED:
                _run_halo_phase(
                    plan=plan,
                    run_dir=run_dir,
                    workspace=workspace,
                    journal=journal,
                    telemetry=telemetry,
                )
        except BaseException as error:
            campaign_error = error
            journal.append(
                {
                    "event": "campaign_failed",
                    "error_type": type(error).__name__,
                }
            )
        finally:
            try:
                _recover_plan_workers(plan, journal)
                _recover_plan_servers(plan, journal)
                if not _remove_worker_network(plan, journal):
                    raise LoopCampaignError("Final worker network cleanup failed")
            except BaseException as error:
                cleanup_error = error
                journal.append(
                    {
                        "event": "campaign_cleanup_failed",
                        "error_type": type(error).__name__,
                    }
                )
            else:
                journal.append({"event": "campaign_cleanup_verified"})
            telemetry.set_phase("finalize")
            telemetry.stop()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        if campaign_error is not None:
            raise campaign_error
        if cleanup_error is not None:
            raise cleanup_error
        summary = summarize_campaign(run_dir)
        journal.append(
            {
                "event": "campaign_finished",
                "status": summary["status"],
                "completed_cases": summary["completed_cases"],
            }
        )
        return summarize_campaign(run_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, execute, resume, clean up, and summarize local RLM/HALO "
            "campaigns."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="freeze a campaign plan")
    plan_parser.add_argument("--campaign", type=Path, default=DEFAULT_MANIFEST)
    plan_parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    plan_parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    for command in ("run", "resume", "cleanup", "summarize"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("run_dir", type=Path)
        if command in {"run", "resume", "cleanup"}:
            command_parser.add_argument("--workspace", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0].startswith("_worker-"):
        return _worker_entry(arguments[0].removeprefix("_worker-"))
    try:
        DEFAULT_LOOP_PYTHON.resolve(strict=True)
        loop_prefix = DEFAULT_LOOP_PYTHON.parents[1].resolve(strict=True)
    except OSError:
        print("error: pinned loop environment is absent", file=sys.stderr)
        return 2
    if Path(sys.prefix).resolve() != loop_prefix:
        os.execv(
            str(DEFAULT_LOOP_PYTHON),
            [
                str(DEFAULT_LOOP_PYTHON),
                str(Path(__file__).parents[1] / "loop_campaign.py"),
                *arguments,
            ],
        )
    parser = _build_parser()
    options = parser.parse_args(arguments)
    try:
        if options.command == "plan":
            run_dir = create_campaign_plan(
                campaign_path=options.campaign,
                models_path=options.models,
                results_root=options.results,
            )
            print(run_dir)
            return 0
        if options.command in {"run", "resume"}:
            summary = execute_campaign(
                run_dir=options.run_dir.resolve(), workspace=options.workspace.resolve()
            )
        elif options.command == "cleanup":
            summary = cleanup_campaign(
                run_dir=options.run_dir.resolve(), workspace=options.workspace.resolve()
            )
        else:
            summary = summarize_campaign(options.run_dir.resolve())
    except (LoopCampaignError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(_canonical_json(summary))
    return 0
