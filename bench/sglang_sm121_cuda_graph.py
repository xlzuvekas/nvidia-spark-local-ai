"""Static contract for the matched SM121 breakable CUDA-graph screen.

The two profiles in this contract are exact clones of the prospective C1
agent-admission profile except for an explicit, matched batch-one decode graph
bundle. This module only admits their manifest topology; it deliberately does
not grant either profile authority to use the dedicated SM121 storage runtime.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from .sglang_sm121_agent_admission import (
    SM121_AGENT_ADMISSION_ARGS,
    SM121_AGENT_ADMISSION_DESCRIPTION,
    SM121_AGENT_ADMISSION_PROFILE_ID,
    validate_sm121_agent_admission_profile,
)


SM121_CUDA_GRAPH_BREAKABLE_PROFILE_ID = (
    "qwen38-flash-next-nvfp4-sm121-triton-storage-c1-cuda-graph-breakable-sglang"
)
SM121_CUDA_GRAPH_DISABLED_PROFILE_ID = (
    "qwen38-flash-next-nvfp4-sm121-triton-storage-c1-cuda-graph-disabled-sglang"
)
SM121_CUDA_GRAPH_PROFILE_IDS = frozenset(
    {
        SM121_CUDA_GRAPH_BREAKABLE_PROFILE_ID,
        SM121_CUDA_GRAPH_DISABLED_PROFILE_ID,
    }
)
SM121_CUDA_GRAPH_SUITE_ID = (
    "qwen38-flash-next-sm121-triton-storage-c1-cuda-graph"
)
SM121_CUDA_GRAPH_CASE_ID = (
    "matched-prompt-qwen38-flash-next-sm121-triton-storage-"
    "cuda-graph-d256-c1-v1"
)
SM121_CUDA_GRAPH_SUITE_DESCRIPTION = (
    "Matched single-user D256 decode CUDA-graph screen cloned from the "
    "prospective current-SM121 Qwen3.8 Flash-Next agent profile: batch-one "
    "BreakableCUDAGraphCapture versus disabled, with eager prefill."
)
SM121_CUDA_GRAPH_BREAKABLE_DESCRIPTION = (
    "Matched C1 D256 decode CUDA-graph candidate cloned from the prospective "
    "current-SM121 agent profile, with explicit batch-one breakable decode "
    "graphs, eager prefill, and metrics."
)
SM121_CUDA_GRAPH_DISABLED_DESCRIPTION = (
    "Matched C1 D256 decode CUDA-graph control cloned from the prospective "
    "current-SM121 agent profile, with explicit batch-one disabled decode "
    "graphs, eager prefill, and metrics."
)

_DESCRIPTION_BY_PROFILE_ID = {
    SM121_CUDA_GRAPH_BREAKABLE_PROFILE_ID: SM121_CUDA_GRAPH_BREAKABLE_DESCRIPTION,
    SM121_CUDA_GRAPH_DISABLED_PROFILE_ID: SM121_CUDA_GRAPH_DISABLED_DESCRIPTION,
}
_BACKEND_BY_PROFILE_ID = {
    SM121_CUDA_GRAPH_BREAKABLE_PROFILE_ID: "breakable",
    SM121_CUDA_GRAPH_DISABLED_PROFILE_ID: "disabled",
}

# Full capture cannot cross the PLE path's device-to-host ``tolist()``,
# thread/Future and io_uring work, ctypes copy, or host-to-device/event edges.
# The pinned image marks those regions eager, so only SGLang's breakable graph
# backend can capture the surrounding decode work without swallowing them.


class SM121CudaGraphError(ValueError):
    """Raised when the static CUDA-graph screen contract drifts."""


def _value(item: Any, field: str) -> object:
    if isinstance(item, Mapping):
        return item.get(field)
    return getattr(item, field, None)


def sm121_cuda_graph_args(backend: str) -> tuple[str, ...]:
    """Return the exact one-axis graph bundle derived from the C1 source."""

    if backend not in {"breakable", "disabled"}:
        raise SM121CudaGraphError("decode graph backend is invalid")
    args = list(SM121_AGENT_ADMISSION_ARGS)
    flag = "--cuda-graph-backend-decode"
    if args.count(flag) != 1:
        raise SM121CudaGraphError("source decode graph argument is invalid")
    index = args.index(flag)
    if args[index : index + 5] != [
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--enable-metrics",
    ]:
        raise SM121CudaGraphError("source graph bundle changed")
    args[index : index + 5] = [
        "--cuda-graph-backend-decode",
        backend,
        "--cuda-graph-bs-decode",
        "1",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--enable-metrics",
    ]
    return tuple(args)


def is_sm121_cuda_graph_candidate(model: Any) -> bool:
    """Return whether ``model`` selects one exact static screen profile."""

    return _value(model, "id") in SM121_CUDA_GRAPH_PROFILE_IDS


def _source_projection(model: Any) -> dict[str, object]:
    if isinstance(model, Mapping):
        projection = dict(model)
    elif is_dataclass(model) and not isinstance(model, type):
        projection = asdict(model)
    else:
        try:
            projection = dict(vars(model))
        except TypeError as error:
            raise SM121CudaGraphError("CUDA-graph profile is not projectable") from error
    projection.update(
        {
            "id": SM121_AGENT_ADMISSION_PROFILE_ID,
            "description": SM121_AGENT_ADMISSION_DESCRIPTION,
            "args": SM121_AGENT_ADMISSION_ARGS,
        }
    )
    return projection


def validate_sm121_cuda_graph_candidate(model: Any) -> None:
    """Require an exact source clone plus the explicit graph bundle."""

    profile_id = _value(model, "id")
    if profile_id not in SM121_CUDA_GRAPH_PROFILE_IDS:
        raise SM121CudaGraphError("SM121 CUDA-graph profile is invalid")
    if _value(model, "description") != _DESCRIPTION_BY_PROFILE_ID[profile_id]:
        raise SM121CudaGraphError("CUDA-graph profile description changed")
    args = _value(model, "args")
    if not isinstance(args, (list, tuple)):
        raise SM121CudaGraphError("CUDA-graph profile args are invalid")
    if "full" in args:
        raise SM121CudaGraphError("full CUDA graphs are prohibited for PLE")
    expected_args = sm121_cuda_graph_args(_BACKEND_BY_PROFILE_ID[profile_id])
    if tuple(args) != expected_args:
        raise SM121CudaGraphError("CUDA-graph profile args changed")
    try:
        validate_sm121_agent_admission_profile(_source_projection(model))
    except ValueError as error:
        raise SM121CudaGraphError(
            "CUDA-graph profile changed beyond its graph bundle"
        ) from error


def validate_sm121_cuda_graph_suite(suite: Any) -> None:
    """Require the one-case deterministic C1 D256 screening suite."""

    if _value(suite, "id") != SM121_CUDA_GRAPH_SUITE_ID:
        raise SM121CudaGraphError("CUDA-graph suite id changed")
    if _value(suite, "description") != SM121_CUDA_GRAPH_SUITE_DESCRIPTION:
        raise SM121CudaGraphError("CUDA-graph suite description changed")
    if _value(suite, "schema_version") != 1:
        raise SM121CudaGraphError("CUDA-graph suite schema changed")
    if _value(suite, "protocol_digest") is not None:
        raise SM121CudaGraphError("CUDA-graph suite digest is invalid")
    cases = _value(suite, "cases")
    if not isinstance(cases, (list, tuple)) or len(cases) != 1:
        raise SM121CudaGraphError("CUDA-graph suite cases are invalid")
    expected = {
        "id": SM121_CUDA_GRAPH_CASE_ID,
        "kind": "decode",
        "requires": ("chat",),
        "warmups": 1,
        "repetitions": 5,
        "max_output_tokens": 256,
        "temperature": 0.0,
        "concurrency": 1,
        "prompt_repetitions": 0,
        "max_turns": 1,
    }
    case = cases[0]
    for field, wanted in expected.items():
        actual = _value(case, field)
        if field == "requires" and isinstance(actual, (list, tuple)):
            actual = tuple(actual)
        if actual != wanted:
            raise SM121CudaGraphError(
                f"CUDA-graph suite case field {field!r} changed"
            )
