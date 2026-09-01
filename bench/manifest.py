"""Typed, dependency-free loaders for Spark benchmark manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path, PurePosixPath
import re
import tomllib
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

from .densespark import (
    DENSESPARK_FAST_CODING_SUITE_ID,
    DENSESPARK_PROFILE_IDS,
    DENSESPARK_SUITE_ID,
    DENSESPARK_TOOL_SUITE_ID,
    DenseSparkContractError,
    is_densespark_profile,
    validate_densespark_profile,
    validate_densespark_selection,
    validate_densespark_suite,
)
from .memory_ops import (
    MEMORY_OPERATION_CONTEXT_TOKENS,
    MEMORY_OPERATION_LLAMACPP_DIGEST,
    MEMORY_OPERATION_LLAMACPP_REVISION,
    MEMORY_OPERATION_OUTPUT_TOKENS,
    MEMORY_OPERATION_SCENARIO_IDS,
    MEMORY_OPERATION_SUITE_DESCRIPTION,
    MEMORY_OPERATION_SUITE_ID,
    MEMORY_OPERATION_VARIANT_COUNT,
    memory_operation_llamacpp_args,
    require_memory_operation_protocol_digest,
)
from .prefix_cache_protocol import (
    PREFIX_CACHE_PREFIX_TARGETS,
    PREFIX_CACHE_SUITE_ID,
)
from .sglang_sm121_storage import (
    SM121_STORAGE_MODE,
    SM121_STORAGE_PROFILE_ID,
    SM121_STORAGE_SUITE_ID,
    SM121StorageCandidateError,
    validate_sm121_storage_candidate,
    validate_sm121_storage_suite,
)
from .sglang_sm121_agent_admission import (
    SM121_AGENT_ADMISSION_PROFILE_ID,
    SM121_AGENT_ADMISSION_SUITE_ID,
    SM121AgentAdmissionError,
    is_sm121_agent_admission_candidate,
    validate_sm121_agent_admission_profile,
    validate_sm121_agent_admission_suite,
)
from .sglang_sm121_cuda_graph import (
    SM121_CUDA_GRAPH_CASE_ID,
    SM121_CUDA_GRAPH_PROFILE_IDS,
    SM121_CUDA_GRAPH_SUITE_DESCRIPTION,
    SM121_CUDA_GRAPH_SUITE_ID,
    SM121CudaGraphError,
    is_sm121_cuda_graph_candidate,
    validate_sm121_cuda_graph_candidate,
    validate_sm121_cuda_graph_suite,
)
from .sglang_sm121_cache_observability import (
    SM121_CACHE_OBSERVABILITY_SUITE_ID,
    SM121CacheObservabilityError,
    validate_sm121_cache_observability_suite,
)
from .sglang_sm121_cache_semantic import (
    SM121_CACHE_SEMANTIC_PROFILE_IDS,
    SM121_CACHE_SEMANTIC_SUITE_ID,
    SM121CacheSemanticError,
    is_sm121_cache_semantic_candidate,
    validate_sm121_cache_semantic_candidate,
    validate_sm121_cache_semantic_suite,
)
from .sglang_sm121_cache_performance import (
    SM121_CACHE_PERFORMANCE_PROFILE_IDS,
    SM121_CACHE_PERFORMANCE_SUITE_ID,
    SM121CachePerformanceError,
    is_sm121_cache_performance_candidate,
    validate_sm121_cache_performance_candidate,
    validate_sm121_cache_performance_suite,
)
from .sglang_sm121_chunked_prefill_performance import (
    SM121_CHUNKED_PREFILL_PERFORMANCE_PROFILE_IDS,
    SM121ChunkedPrefillPerformanceError,
    is_sm121_chunked_prefill_performance_candidate,
    sm121_chunked_prefill_performance_study,
    validate_sm121_chunked_prefill_performance_candidate,
    validate_sm121_chunked_prefill_performance_suite,
)
from .sglang_sm121_chunked_prefill_admission import (
    SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE_ID,
    SM121ChunkedPrefill8KAdmissionError,
    is_sm121_chunked_prefill_8k_admission_profile,
    validate_sm121_chunked_prefill_8k_admission_suite,
)


SCHEMA_VERSION = 1
HOST_SAFETY_MODEL_FIELDS = (
    "host_safety_min_memavailable_gib",
    "host_safety_max_swap_growth_mib",
    "host_safety_max_starting_swap_mib",
)
MATCHED_PROMPT_CASE_PREFIX = "matched-prompt-"
MATCHED_REQUEST_UNIQUE_PROTOCOL = "matched-request-unique-v1"
QWEN38_27B_DSPARK_CUDA_GRAPH_SUITE_ID = (
    "qwen38-27b-dspark-c1-cuda-graph"
)
QWEN38_27B_DSPARK_CUDA_GRAPH_SUITE_DESCRIPTION = (
    "Matched single-user D256 screen for the pinned RadixArk Qwen3.8-27B "
    "plus DSpark full-versus-disabled decode CUDA-graph ablation."
)
QWEN38_27B_DSPARK_CUDA_GRAPH_CASE_ID = (
    "matched-prompt-qwen38-27b-dspark-cuda-graph-d256-c1-v1"
)
QWEN38_27B_DSPARK_CUDA_GRAPH_PROFILE_IDS = frozenset(
    {
        "qwen38-27b-nvfp4-dspark-c1-cuda-graph-full-sglang",
        "qwen38-27b-nvfp4-dspark-c1-cuda-graph-disabled-sglang",
    }
)
QWEN38_27B_DSPARK_SOURCE_PROFILE_ID = "qwen38-27b-nvfp4-dspark-sglang"
QWEN38_27B_DSPARK_CUDA_GRAPH_FULL_PROFILE_ID = (
    "qwen38-27b-nvfp4-dspark-c1-cuda-graph-full-sglang"
)
QWEN38_27B_DSPARK_CUDA_GRAPH_DISABLED_PROFILE_ID = (
    "qwen38-27b-nvfp4-dspark-c1-cuda-graph-disabled-sglang"
)
QWEN38_27B_DSPARK_CUDA_GRAPH_FULL_DESCRIPTION = (
    "Matched C1 D256 CUDA-graph-on arm cloned from the pinned RadixArk/DSpark "
    "profile, with an explicit batch-one full decode graph, eager prefill, "
    "and metrics."
)
QWEN38_27B_DSPARK_CUDA_GRAPH_DISABLED_DESCRIPTION = (
    "Matched C1 D256 CUDA-graph-off arm cloned from the pinned RadixArk/DSpark "
    "profile, with explicit batch-one decode geometry, eager prefill, and "
    "metrics."
)
QWEN38_27B_DSPARK_SOURCE_DESCRIPTION = (
    "Managed serving-config reproduction of the pinned upstream recipe; "
    "SparkBench cases and results remain independent. Apache-2.0 target with "
    "the card-license=other DSpark draft."
)
QWEN38_27B_DSPARK_SOURCE_ARGS = (
    "--trust-remote-code",
    "--tp-size",
    "1",
    "--served-model-name",
    "qwen3.8-27b",
    "--mem-fraction-static",
    "0.50",
    "--attention-backend",
    "flashinfer",
    "--chunked-prefill-size",
    "8192",
    "--disable-prefill-cuda-graph",
    "--cuda-graph-max-bs",
    "4",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-dspark-block-size",
    "7",
    "--speculative-draft-model-quantization",
    "unquant",
    "--mamba-radix-cache-strategy",
    "extra_buffer_lazy",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--max-mamba-cache-size",
    "96",
    "--max-running-requests",
    "8",
    "--enable-torch-compile",
    "--torch-compile-max-bs",
    "4",
    "--num-continuous-decode-steps",
    "2",
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen3_coder",
    "--host",
    "0.0.0.0",
    "--port",
    "30000",
)


def qwen38_27b_dspark_cuda_graph_args(backend: str) -> tuple[str, ...]:
    """Return the one reviewed graph delta over the pinned DSpark source."""

    if backend not in {"full", "disabled"}:
        raise ValueError("Qwen3.8 27B graph backend must be full or disabled")
    args = list(QWEN38_27B_DSPARK_SOURCE_ARGS)
    graph_index = args.index("--disable-prefill-cuda-graph")
    if args[graph_index : graph_index + 3] != [
        "--disable-prefill-cuda-graph",
        "--cuda-graph-max-bs",
        "4",
    ]:
        raise AssertionError("Qwen3.8 27B source graph bundle changed")
    args[graph_index : graph_index + 3] = [
        "--cuda-graph-backend-decode",
        backend,
        "--cuda-graph-bs-decode",
        "1",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--enable-metrics",
    ]
    return tuple(args)


@dataclass(frozen=True, slots=True)
class MatchedPromptGraphStudy:
    """One exact model/suite/case binding for a matched prompt schedule."""

    suite_id: str
    suite_description: str
    case_id: str
    profile_ids: frozenset[str]


MATCHED_PROMPT_GRAPH_STUDIES = (
    MatchedPromptGraphStudy(
        suite_id=QWEN38_27B_DSPARK_CUDA_GRAPH_SUITE_ID,
        suite_description=QWEN38_27B_DSPARK_CUDA_GRAPH_SUITE_DESCRIPTION,
        case_id=QWEN38_27B_DSPARK_CUDA_GRAPH_CASE_ID,
        profile_ids=QWEN38_27B_DSPARK_CUDA_GRAPH_PROFILE_IDS,
    ),
    MatchedPromptGraphStudy(
        suite_id=SM121_CUDA_GRAPH_SUITE_ID,
        suite_description=SM121_CUDA_GRAPH_SUITE_DESCRIPTION,
        case_id=SM121_CUDA_GRAPH_CASE_ID,
        profile_ids=SM121_CUDA_GRAPH_PROFILE_IDS,
    ),
)
MATCHED_PROMPT_GRAPH_SUITE_IDS = frozenset(
    study.suite_id for study in MATCHED_PROMPT_GRAPH_STUDIES
)
MATCHED_PROMPT_GRAPH_PROFILE_IDS = frozenset(
    profile_id
    for study in MATCHED_PROMPT_GRAPH_STUDIES
    for profile_id in study.profile_ids
)
_QWEN38_PLE_ABLATION_IDENTITY = (
    "RadixArk/Qwen3.8-Flash-Next-NVFP4",
    "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
    "hashd1ve/qwen38-flash-next-one-dgx-spark",
    "bf2b7c75870d3703730b6bd8f3bb93dc622c278d",
)
_QWEN38_PLE_ABLATION_OVERLAYS = {
    "/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py": (
        "sha256:bcdc2c86aa59784ffe27d53c8d214e56"
        "b6aa45c02b1d5841fd956d1f006d6030"
    ),
    (
        "/sgl-workspace/sglang/python/sglang/srt/layers/attention/"
        "qwen_sparse_attn_backend.py"
    ): (
        "sha256:e30566492e1502f94a4c7fed42d90b5"
        "23bbb662580c628459e6e63c7b5263c75"
    ),
}
KNOWN_TASKS = frozenset(
    {
        "audio",
        "chat",
        "diffusion",
        "embeddings",
        "json",
        "ocr",
        "rerank",
        "thinking",
        "tools",
        "vision",
    }
)
KNOWN_BACKENDS = frozenset(
    {"external", "llamacpp", "ollama", "sglang", "transformers", "trtllm", "vllm"}
)
KNOWN_SUPPORT_STATUSES = frozenset(
    {
        "exploratory",
        "incompatible",
        "spark_other_backend",
        "spark_transformers_direct",
        "spark_trtllm_direct",
        "spark_vllm_matrix",
        "spark_vllm_recipe",
    }
)
KNOWN_CASE_KINDS = frozenset(
    {
        "agentic",
        "capability",
        "cache",
        "concurrency",
        "decode",
        "diffusion",
        "memory",
        "prefill",
        "quality",
    }
)
PREFIX_CACHE_MODES = frozenset({"off", "on"})
KNOWN_AGENTIC_CASE_IDS = frozenset(
    {
        "agentic-no-tool",
        "agentic-select-and-call",
        "agentic-tool-error-recovery",
        "agentic-two-hop",
    }
)
KNOWN_MEMORY_OPERATION_CASE_IDS = frozenset(MEMORY_OPERATION_SCENARIO_IDS)
_TPS_PROFILE_IDS_BY_SUITE = {
    "long-context-tps": frozenset(
        {
            "qwen36-35b-a3b-nvfp4-mtp3-long-tps",
            "qwen38-27b-nvfp4-mtp3-long-tps",
        }
    ),
    "throughput-saturation": frozenset(
        {
            "qwen36-35b-a3b-nvfp4-mtp3-tps64",
            "qwen38-27b-nvfp4-mtp3-tps64",
        }
    ),
}
_TPS_SUITE_BY_PROFILE_ID = {
    profile_id: suite_id
    for suite_id, profile_ids in _TPS_PROFILE_IDS_BY_SUITE.items()
    for profile_id in profile_ids
}
_FLASH_NEXT_LONG_PROFILE_ID = "qwen38-flash-next-nvfp4-long-sglang"
_FLASH_NEXT_LONG_SUITE_ID = "qwen38-flash-next-sglang-long-context"
VARIED_CONTEXT_NEEDLE_CASE_PREFIX = "sm121-varied-context-needle-"
VARIED_CONTEXT_NEEDLE_PROTOCOL_VERSION = 1
VARIED_CONTEXT_NEEDLE_MIN_OUTPUT_TOKENS = 32
_VARIED_CONTEXT_NEEDLE_CASE_ID_PATTERN = re.compile(
    r"^sm121-varied-context-needle-"
    r"(?P<filler_records>[1-9][0-9]*)-"
    r"(?P<depth>mid|tail)-"
    r"s(?P<seed>[a-z0-9][a-z0-9-]*)-"
    r"c1-v1$"
)
_FLASH_NEXT_AGENT64K_SUITE_ID = "qwen38-flash-next-sglang-agent64k-autoresearch"
_FLASH_NEXT_AGENT64K_PROFILE_IDS = frozenset(
    {
        "qwen38-flash-next-nvfp4-mtp2-agent64k-low-ple-mapped-sglang",
        "qwen38-flash-next-nvfp4-mtp3-agent64k-low-ple-mapped-sglang",
        "qwen38-flash-next-nvfp4-mtp2-agent64k-low-chunk2k-ple-mapped-sglang",
        "qwen38-flash-next-nvfp4-mtp2-agent64k-none-ple-mapped-sglang",
    }
)
_SM121_STORAGE_PROFILE_ID = SM121_STORAGE_PROFILE_ID
_SM121_STORAGE_SUITE_ID = SM121_STORAGE_SUITE_ID
_SM121_STORAGE_CACHE_OBSERVABILITY_SUITE_ID = SM121_CACHE_OBSERVABILITY_SUITE_ID
_SM121_CACHE_SEMANTIC_SUITE_ID = SM121_CACHE_SEMANTIC_SUITE_ID
_SM121_CACHE_SEMANTIC_PROFILE_IDS = SM121_CACHE_SEMANTIC_PROFILE_IDS
_SM121_STORAGE_SUITE_IDS = frozenset(
    {
        _SM121_STORAGE_SUITE_ID,
        _SM121_STORAGE_CACHE_OBSERVABILITY_SUITE_ID,
    }
)
_PLE_STUDY_PROFILE_IDS_BY_SUITE = {
    "qwen38-flash-next-sglang-ple-depth-c8": frozenset(
        {
            "qwen38-flash-next-nvfp4-mtp1-c8-lazy-ple-mapped-sglang",
            "qwen38-flash-next-nvfp4-mtp2-c8-lazy-ple-mapped-sglang",
            "qwen38-flash-next-nvfp4-mtp3-c8-lazy-ple-mapped-sglang",
            "qwen38-flash-next-nvfp4-mtp3-c8-lazy-ple-omitted-sglang",
        }
    ),
    "qwen38-flash-next-sglang-lazy-depth3-interaction": frozenset(
        {
            "qwen38-flash-next-nvfp4-mtp2-c6-extra-ple-mapped-sglang",
            "qwen38-flash-next-nvfp4-mtp3-c6-extra-ple-mapped-sglang",
            "qwen38-flash-next-nvfp4-mtp2-c8-lazy-ple-mapped-sglang",
            "qwen38-flash-next-nvfp4-mtp3-c8-lazy-ple-mapped-sglang",
        }
    ),
    "qwen38-flash-next-sglang-lazy-depth3-interaction-core-v2": frozenset(
        {
            "qwen38-flash-next-nvfp4-mtp2-c6-extra-ple-mapped-sglang",
            "qwen38-flash-next-nvfp4-mtp3-c6-extra-ple-mapped-sglang",
            "qwen38-flash-next-nvfp4-mtp2-c8-lazy-ple-mapped-sglang",
            "qwen38-flash-next-nvfp4-mtp3-c8-lazy-ple-mapped-sglang",
        }
    ),
    "qwen38-flash-next-sglang-quality-v2": frozenset(
        {
            "qwen38-flash-next-nvfp4-mtp3-quality-v2-ple-mapped-sglang",
            "qwen38-flash-next-nvfp4-mtp3-quality-v2-ple-omitted-sglang",
        }
    ),
}
_PLE_STUDY_SUITE_IDS_BY_PROFILE_ID = {
    profile_id: frozenset(
        suite_id
        for suite_id, profile_ids in _PLE_STUDY_PROFILE_IDS_BY_SUITE.items()
        if profile_id in profile_ids
    )
    for profile_id in frozenset().union(
        *_PLE_STUDY_PROFILE_IDS_BY_SUITE.values()
    )
}
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_HF_REPOSITORY_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)
_LLAMACPP_SPLIT_GGUF_PATTERN = re.compile(
    r"^(?P<prefix>.+)-(?P<ordinal>[0-9]+)-of-(?P<total>[0-9]+)\.gguf$",
    re.IGNORECASE,
)
_LLAMACPP_RESERVED_ARGS = frozenset(
    {
        "-m",
        "--model",
        "-mu",
        "--model-url",
        "-dr",
        "--docker-repo",
        "-hf",
        "-hfr",
        "--hf-repo",
        "-hff",
        "--hf-file",
        "-hft",
        "--hf-token",
        "--host",
        "--port",
        "-a",
        "--alias",
        "-c",
        "--ctx-size",
        "-np",
        "--parallel",
        "-mm",
        "--mmproj",
        "-mmu",
        "--mmproj-url",
        "--mmproj-auto",
        "--no-mmproj",
        "--no-mmproj-auto",
        "--mmproj-offload",
        "--no-mmproj-offload",
        "--metrics",
        "--offline",
        "--ui",
        "--webui",
        "--no-ui",
        "--no-webui",
        "-hfd",
        "-hfrd",
        "--spec-draft-hf",
        "--hf-repo-draft",
        "-md",
        "--spec-draft-model",
        "--model-draft",
        "--cors-origins",
        "--cors-methods",
        "--cors-headers",
        "--cors-credentials",
        "--no-cors-credentials",
        "--api-key",
        "--api-key-file",
    }
)
_LLAMACPP_FORBIDDEN_ARGS = frozenset(
    {
        "--rpc",
        "--reuse-port",
        "--api-prefix",
        "--ui-config",
        "--webui-config",
        "--ui-config-file",
        "--webui-config-file",
        "--ui-mcp-proxy",
        "--webui-mcp-proxy",
        "--no-ui-mcp-proxy",
        "--no-webui-mcp-proxy",
        "--tools",
        "--tools-runtime",
        "--mcp-servers-config",
        "--mcp-servers-json",
        "-ag",
        "--agent",
        "-no-ag",
        "--no-agent",
        "--props",
        "--models-dir",
        "--models-preset",
        "--models-max",
        "--models-autoload",
        "--no-models-autoload",
        "--log-disable",
        "--log-file",
        "--log-prompts-dir",
        "--log-colors",
        "-v",
        "--verbose",
        "--log-verbose",
        "-lv",
        "--verbosity",
        "--log-verbosity",
        "--log-prefix",
        "--no-log-prefix",
        "--log-timestamps",
        "--no-log-timestamps",
        "--slot-save-path",
        "--media-path",
        "--path",
        "--ssl-key-file",
        "--ssl-cert-file",
        "--mtp",
        "--dflash",
        "--eagle3",
    }
)
_SGLANG_RESERVED_ARGS = frozenset(
    {
        "--api-key",
        "--model-path",
        "--speculative-draft-model-path",
    }
)
_MODEL_KEYS = frozenset(
    {
        "id",
        "description",
        "backend",
        "source",
        "revision",
        "served_name",
        "tasks",
        "architecture",
        "quantization",
        "lifecycle",
        "image",
        "image_digest",
        "local_image_id",
        "max_context",
        "native_context",
        "startup_timeout_s",
        "estimated_ram_gib",
        "host_safety_min_memavailable_gib",
        "host_safety_max_swap_growth_mib",
        "host_safety_max_starting_swap_mib",
        "endpoint",
        "args",
        "cache_dir",
        "fetch_allow_patterns",
        "fetch_ignore_patterns",
        "weight_size_bytes",
        "weight_file_count",
        "draft_source",
        "draft_revision",
        "draft_weight_size_bytes",
        "draft_model_file",
        "draft_model_digest",
        "draft_model_size_bytes",
        "densespark_pq_file",
        "densespark_pq_digest",
        "densespark_pq_size_bytes",
        "sglang_allow_hf_metadata_probe",
        "sglang_source_overlays",
        "sglang_ple_mmap",
        "sglang_ple_omitted",
        "sglang_ple_cache_mode",
        "sglang_ple_cache_marker_digest",
        "sglang_ple_cache_payload_digest",
        "sglang_storage_mode",
        "sglang_ple_nvme_queue_depth",
        "sglang_ple_nvme_max_batch_pages",
        "sglang_ple_nvme_cache_pages",
        "recipe_source",
        "recipe_revision",
        "request_body_json",
        "runtime_python",
        "runtime_binary",
        "runtime_digest",
        "runtime_parallel",
        "runtime_source_dir",
        "runtime_revision",
        "model_file",
        "model_digest",
        "model_size_bytes",
        "model_shards",
        "mmproj_file",
        "mmproj_digest",
        "mmproj_size_bytes",
        "prefix_cache_mode",
        "support_status",
    }
)
_MODEL_SHARD_KEYS = frozenset({"path", "digest", "size_bytes"})
_SGLANG_SOURCE_OVERLAY_KEYS = frozenset(
    {"host_path", "container_path", "digest"}
)
_CASE_KEYS = frozenset(
    {
        "id",
        "kind",
        "requires",
        "warmups",
        "repetitions",
        "concurrency",
        "prompt_repetitions",
        "max_output_tokens",
        "max_turns",
        "temperature",
    }
)


class ManifestError(ValueError):
    """Raised when a manifest is malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ModelShard:
    """One ordered, exact GGUF shard within a pinned snapshot."""

    path: str
    digest: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SGLangSourceOverlay:
    """One digest-pinned host file mounted over an SGLang image source file."""

    host_path: str
    container_path: str
    digest: str


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One reproducible model-serving configuration."""

    id: str
    backend: str
    source: str
    served_name: str
    tasks: tuple[str, ...]
    image: str | None = None
    max_context: int = 0
    native_context: int | None = None
    startup_timeout_s: int = 1_800
    args: tuple[str, ...] = ()
    endpoint: str = "http://127.0.0.1:8000/v1"
    estimated_ram_gib: float | None = None
    host_safety_min_memavailable_gib: int | None = None
    host_safety_max_swap_growth_mib: int | None = None
    host_safety_max_starting_swap_mib: int | None = None
    revision: str | None = None
    image_digest: str | None = None
    local_image_id: str | None = None
    architecture: str = "unknown"
    quantization: str | None = None
    lifecycle: str = "docker"
    description: str = ""
    cache_dir: str = "project"
    fetch_allow_patterns: tuple[str, ...] = ()
    fetch_ignore_patterns: tuple[str, ...] = ()
    weight_size_bytes: int | None = None
    weight_file_count: int | None = None
    draft_source: str | None = None
    draft_revision: str | None = None
    draft_weight_size_bytes: int | None = None
    draft_model_file: str | None = None
    draft_model_digest: str | None = None
    draft_model_size_bytes: int | None = None
    densespark_pq_file: str | None = None
    densespark_pq_digest: str | None = None
    densespark_pq_size_bytes: int | None = None
    sglang_allow_hf_metadata_probe: bool = False
    sglang_source_overlays: tuple[SGLangSourceOverlay, ...] = ()
    sglang_ple_mmap: bool = False
    sglang_ple_omitted: bool = False
    sglang_ple_cache_mode: str | None = None
    sglang_ple_cache_marker_digest: str | None = None
    sglang_ple_cache_payload_digest: str | None = None
    sglang_storage_mode: str | None = None
    sglang_ple_nvme_queue_depth: int | None = None
    sglang_ple_nvme_max_batch_pages: int | None = None
    sglang_ple_nvme_cache_pages: int | None = None
    recipe_source: str | None = None
    recipe_revision: str | None = None
    request_body_json: str | None = None
    runtime_python: str | None = None
    runtime_binary: str | None = None
    runtime_digest: str | None = None
    runtime_parallel: int | None = None
    runtime_source_dir: str | None = None
    runtime_revision: str | None = None
    model_file: str | None = None
    model_digest: str | None = None
    model_size_bytes: int | None = None
    model_shards: tuple[ModelShard, ...] = ()
    mmproj_file: str | None = None
    mmproj_digest: str | None = None
    mmproj_size_bytes: int | None = None
    prefix_cache_mode: str | None = None
    support_status: str = "exploratory"


def model_spec_to_dict(model: ModelSpec) -> dict[str, Any]:
    """Serialize a model without changing legacy unconfigured records.

    Host-safety gates are an opt-in profile contract. Omitting their three
    nullable defaults preserves frozen fingerprints and strict projections for
    profiles that predate that contract, while configured profiles retain the
    complete all-or-none threshold tuple.
    """

    record = asdict(model)
    configured = {
        name: record[name]
        for name in HOST_SAFETY_MODEL_FIELDS
        if record[name] is not None
    }
    if not configured:
        for name in HOST_SAFETY_MODEL_FIELDS:
            record.pop(name)
    elif len(configured) != len(HOST_SAFETY_MODEL_FIELDS):
        raise ManifestError(
            "model host-safety thresholds must be configured together"
        )
    storage_fields = (
        "local_image_id",
        "sglang_storage_mode",
        "sglang_ple_nvme_queue_depth",
        "sglang_ple_nvme_max_batch_pages",
        "sglang_ple_nvme_cache_pages",
    )
    if not any(record[name] is not None for name in storage_fields):
        for name in storage_fields:
            record.pop(name)
    elif is_densespark_profile(model):
        for name in storage_fields[1:]:
            record.pop(name)
    densespark_fields = (
        "densespark_pq_file",
        "densespark_pq_digest",
        "densespark_pq_size_bytes",
    )
    if not any(record[name] is not None for name in densespark_fields):
        for name in densespark_fields:
            record.pop(name)
    return record


def qwen38_27b_dspark_cuda_graph_expected_model(
    profile_id: str,
) -> ModelSpec:
    """Construct one complete reviewed Qwen3.8 27B graph-arm identity."""

    source = ModelSpec(
        id=QWEN38_27B_DSPARK_SOURCE_PROFILE_ID,
        backend="sglang",
        source="RadixArk/Qwen3.8-27B-NVFP4",
        served_name="qwen3.8-27b",
        tasks=("chat", "json", "thinking", "tools"),
        image=(
            "lmsysorg/sglang@sha256:"
            "febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1"
        ),
        max_context=262_144,
        native_context=262_144,
        startup_timeout_s=1_800,
        args=QWEN38_27B_DSPARK_SOURCE_ARGS,
        endpoint="http://127.0.0.1:30000/v1",
        estimated_ram_gib=92.0,
        revision="52d1adc5f38aa5ebf099c29ed7025ba34cfbb854",
        image_digest=(
            "sha256:"
            "febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1"
        ),
        architecture="dense+dspark-draft",
        quantization="nvfp4+unquant-dspark",
        description=QWEN38_27B_DSPARK_SOURCE_DESCRIPTION,
        cache_dir="user",
        weight_size_bytes=21_921_697_280,
        weight_file_count=3,
        draft_source="RadixArk/Qwen3.8-27B-DSpark",
        draft_revision="923ed3a8572615643f0137e424e4ce4edd7f1cda",
        draft_weight_size_bytes=2_718_576_122,
        sglang_allow_hf_metadata_probe=True,
        recipe_source="hasso5703/dgx-spark-qwen38",
        recipe_revision="3590fb29296b1babd85405daad1eef1c4a3ebe0f",
        support_status="spark_other_backend",
    )
    if profile_id == QWEN38_27B_DSPARK_CUDA_GRAPH_FULL_PROFILE_ID:
        return replace(
            source,
            id=profile_id,
            description=QWEN38_27B_DSPARK_CUDA_GRAPH_FULL_DESCRIPTION,
            args=qwen38_27b_dspark_cuda_graph_args("full"),
        )
    if profile_id == QWEN38_27B_DSPARK_CUDA_GRAPH_DISABLED_PROFILE_ID:
        return replace(
            source,
            id=profile_id,
            description=QWEN38_27B_DSPARK_CUDA_GRAPH_DISABLED_DESCRIPTION,
            args=qwen38_27b_dspark_cuda_graph_args("disabled"),
        )
    raise ValueError(f"{profile_id!r} is not a Qwen3.8 27B graph profile")


def is_qwen38_27b_dspark_cuda_graph_candidate(model: Any) -> bool:
    """Return whether ``model`` claims one reviewed Qwen3.8 27B graph ID."""

    model_id = (
        model.get("id") if isinstance(model, Mapping) else getattr(model, "id", None)
    )
    return model_id in QWEN38_27B_DSPARK_CUDA_GRAPH_PROFILE_IDS


def _coerce_qwen38_27b_dspark_cuda_graph_model(model: Any) -> ModelSpec:
    if isinstance(model, ModelSpec):
        return model
    if isinstance(model, Mapping):
        record = dict(model)
    else:
        try:
            record = dict(vars(model))
        except TypeError as error:
            raise ManifestError(
                "Qwen3.8 27B graph profile must be a model record"
            ) from error
    unknown = set(record) - set(ModelSpec.__dataclass_fields__)
    if unknown:
        raise ManifestError(
            "Qwen3.8 27B graph profile has unknown fields: "
            + ", ".join(sorted(unknown))
        )
    for name in (
        "args",
        "fetch_allow_patterns",
        "fetch_ignore_patterns",
        "tasks",
    ):
        value = record.get(name)
        if isinstance(value, (list, tuple)):
            record[name] = tuple(value)
    overlays = record.get("sglang_source_overlays")
    if isinstance(overlays, (list, tuple)):
        record["sglang_source_overlays"] = tuple(
            item
            if isinstance(item, SGLangSourceOverlay)
            else SGLangSourceOverlay(**item)
            if isinstance(item, Mapping)
            else item
            for item in overlays
        )
    shards = record.get("model_shards")
    if isinstance(shards, (list, tuple)):
        record["model_shards"] = tuple(
            item
            if isinstance(item, ModelShard)
            else ModelShard(**item)
            if isinstance(item, Mapping)
            else item
            for item in shards
        )
    try:
        return ModelSpec(**record)
    except (TypeError, ValueError) as error:
        raise ManifestError(
            "Qwen3.8 27B graph profile is not a complete model record"
        ) from error


def validate_qwen38_27b_dspark_cuda_graph_candidate(model: Any) -> None:
    """Require the complete exact source clone plus its one graph-arm delta."""

    candidate = _coerce_qwen38_27b_dspark_cuda_graph_model(model)
    if candidate.id not in QWEN38_27B_DSPARK_CUDA_GRAPH_PROFILE_IDS:
        raise ManifestError("model is not a reviewed Qwen3.8 27B graph profile")
    expected = qwen38_27b_dspark_cuda_graph_expected_model(candidate.id)
    candidate_json = json.dumps(
        asdict(candidate), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    expected_json = json.dumps(
        asdict(expected), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    if candidate_json != expected_json:
        raise ManifestError(
            f"{candidate.id!r} changed beyond its exact graph bundle"
        )


@dataclass(frozen=True, slots=True)
class VariedContextNeedleSpec:
    """Typed identity parsed from a fixed varied-context needle case ID.

    The protocol keeps the depth and deterministic seed inside the stable case
    identity so generated text never depends on a request identifier.  It is
    deliberately a derived type rather than additional nullable CaseSpec
    fields: serializations of every pre-existing suite remain unchanged.
    """

    filler_records: int
    depth: str
    seed: str
    protocol_version: int = VARIED_CONTEXT_NEEDLE_PROTOCOL_VERSION


def varied_context_needle_spec(case_id: str) -> VariedContextNeedleSpec | None:
    """Parse one fixed varied-context needle case ID into its typed contract.

    ``None`` means the case does not use this protocol.  A case using the
    reserved prefix but not its complete v1 grammar is rejected instead of
    silently falling back to a generic capability prompt.
    """

    if not case_id.startswith(VARIED_CONTEXT_NEEDLE_CASE_PREFIX):
        return None
    match = _VARIED_CONTEXT_NEEDLE_CASE_ID_PATTERN.fullmatch(case_id)
    if match is None:
        raise ValueError(
            "varied-context needle IDs must use "
            "sm121-varied-context-needle-<records>-<mid|tail>-"
            "s<seed>-c1-v1"
        )
    return VariedContextNeedleSpec(
        filler_records=int(match.group("filler_records")),
        depth=match.group("depth"),
        seed=match.group("seed"),
    )


def matched_prompt_graph_study(
    *,
    case_id: str | None = None,
    suite_id: str | None = None,
    profile_id: str | None = None,
) -> MatchedPromptGraphStudy | None:
    """Resolve exactly one graph study identity without fuzzy matching."""

    selectors = tuple(
        value for value in (case_id, suite_id, profile_id) if value is not None
    )
    if len(selectors) != 1:
        raise ValueError("exactly one matched-prompt graph selector is required")
    for study in MATCHED_PROMPT_GRAPH_STUDIES:
        if case_id is not None and case_id == study.case_id:
            return study
        if suite_id is not None and suite_id == study.suite_id:
            return study
        if profile_id is not None and profile_id in study.profile_ids:
            return study
    return None


def validate_matched_prompt_graph_model(
    model: Any,
    *,
    context: str = "model",
) -> None:
    """Dispatch a graph arm only to its exact pair-specific profile contract."""

    model_id = (
        model.get("id") if isinstance(model, Mapping) else getattr(model, "id", None)
    )
    study = matched_prompt_graph_study(profile_id=model_id)
    if study is None:
        raise ManifestError(f"{context}.id is not a matched-prompt graph profile")
    if study.suite_id == QWEN38_27B_DSPARK_CUDA_GRAPH_SUITE_ID:
        try:
            validate_qwen38_27b_dspark_cuda_graph_candidate(model)
        except ManifestError as error:
            raise ManifestError(f"{context}: {error}") from error
        return
    if study.suite_id != SM121_CUDA_GRAPH_SUITE_ID:
        raise ManifestError(f"{context}.id is not a reviewed graph profile")
    try:
        validate_sm121_cuda_graph_candidate(model)
    except SM121CudaGraphError as error:
        raise ManifestError(f"{context}: {error}") from error


def matched_prompt_protocol(case_id: str) -> str | None:
    """Return the exact matched-prompt protocol selected by ``case_id``.

    The reserved prefix is fail-closed: a typo or an unreviewed protocol
    variant cannot silently fall back to the generic model-bound, clock-based
    nonce schedule.
    """

    if not case_id.startswith(MATCHED_PROMPT_CASE_PREFIX):
        return None
    if matched_prompt_graph_study(case_id=case_id) is None:
        raise ValueError(
            "matched-prompt IDs must use one of the exact reviewed "
            "CUDA-graph v1 case identities"
        )
    return MATCHED_REQUEST_UNIQUE_PROTOCOL


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """A workload case expanded across compatible model configurations."""

    id: str
    kind: str
    requires: tuple[str, ...]
    warmups: int = 0
    repetitions: int = 1
    max_output_tokens: int = 1
    temperature: float = 0.0
    concurrency: int = 1
    prompt_repetitions: int = 0
    max_turns: int = 1

    @property
    def varied_context_needle(self) -> VariedContextNeedleSpec | None:
        """Return the typed v1 varied-context contract, when this case uses it."""

        return varied_context_needle_spec(self.id)


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    """A named collection of benchmark cases."""

    id: str
    cases: tuple[CaseSpec, ...]
    description: str = ""
    schema_version: int = SCHEMA_VERSION
    protocol_digest: str | None = None


def load_models(path: str | Path) -> dict[str, ModelSpec]:
    """Load and validate a model registry keyed by stable configuration ID."""

    manifest_path = Path(path)
    document = _read_toml(manifest_path)
    _reject_unknown(document, {"schema_version", "models"}, str(manifest_path))
    _check_schema_version(document, manifest_path)
    rows = document.get("models")
    if not isinstance(rows, list) or not rows:
        raise ManifestError(f"{manifest_path}: 'models' must be a non-empty array")

    models: dict[str, ModelSpec] = {}
    for index, row in enumerate(rows):
        context = f"{manifest_path}: models[{index}]"
        if not isinstance(row, dict):
            raise ManifestError(f"{context} must be a table")
        _reject_unknown(row, _MODEL_KEYS, context)
        backend = _required_string(row, "backend", context).lower()
        model = ModelSpec(
            id=_required_string(row, "id", context),
            backend=backend,
            source=_required_string(row, "source", context),
            served_name=_required_string(row, "served_name", context),
            tasks=_string_tuple(row, "tasks", context, required=True),
            image=_optional_string(row, "image", context),
            max_context=_required_int(row, "max_context", context),
            native_context=_optional_int(
                row, "native_context", context, default=None
            ),
            startup_timeout_s=_optional_int(
                row, "startup_timeout_s", context, default=1_800
            ),
            args=_string_tuple(row, "args", context),
            endpoint=_optional_string(row, "endpoint", context)
            or _default_endpoint(backend),
            estimated_ram_gib=_optional_number(
                row, "estimated_ram_gib", context
            ),
            host_safety_min_memavailable_gib=_optional_int(
                row,
                "host_safety_min_memavailable_gib",
                context,
                default=None,
            ),
            host_safety_max_swap_growth_mib=_optional_int(
                row,
                "host_safety_max_swap_growth_mib",
                context,
                default=None,
            ),
            host_safety_max_starting_swap_mib=_optional_int(
                row,
                "host_safety_max_starting_swap_mib",
                context,
                default=None,
            ),
            revision=_optional_string(row, "revision", context),
            image_digest=_optional_string(row, "image_digest", context),
            local_image_id=_optional_string(row, "local_image_id", context),
            architecture=_optional_string(row, "architecture", context)
            or "unknown",
            quantization=_optional_string(row, "quantization", context),
            lifecycle=_optional_string(row, "lifecycle", context) or "docker",
            description=_optional_string(row, "description", context) or "",
            cache_dir=_optional_string(row, "cache_dir", context) or "project",
            fetch_allow_patterns=_string_tuple(
                row, "fetch_allow_patterns", context
            ),
            fetch_ignore_patterns=_string_tuple(
                row, "fetch_ignore_patterns", context
            ),
            weight_size_bytes=_optional_int(
                row, "weight_size_bytes", context, default=None
            ),
            weight_file_count=_optional_int(
                row, "weight_file_count", context, default=None
            ),
            draft_source=_optional_string(row, "draft_source", context),
            draft_revision=_optional_string(row, "draft_revision", context),
            draft_weight_size_bytes=_optional_int(
                row, "draft_weight_size_bytes", context, default=None
            ),
            draft_model_file=_optional_string(row, "draft_model_file", context),
            draft_model_digest=_optional_string(
                row, "draft_model_digest", context
            ),
            draft_model_size_bytes=_optional_int(
                row, "draft_model_size_bytes", context, default=None
            ),
            densespark_pq_file=_optional_string(
                row, "densespark_pq_file", context
            ),
            densespark_pq_digest=_optional_string(
                row, "densespark_pq_digest", context
            ),
            densespark_pq_size_bytes=_optional_int(
                row, "densespark_pq_size_bytes", context, default=None
            ),
            sglang_allow_hf_metadata_probe=_optional_bool(
                row, "sglang_allow_hf_metadata_probe", context, default=False
            ),
            sglang_source_overlays=_sglang_source_overlays(
                row, "sglang_source_overlays", context
            ),
            sglang_ple_mmap=_optional_bool(
                row, "sglang_ple_mmap", context, default=False
            ),
            sglang_ple_omitted=_optional_bool(
                row, "sglang_ple_omitted", context, default=False
            ),
            sglang_ple_cache_mode=_optional_string(
                row, "sglang_ple_cache_mode", context
            ),
            sglang_ple_cache_marker_digest=_optional_string(
                row, "sglang_ple_cache_marker_digest", context
            ),
            sglang_ple_cache_payload_digest=_optional_string(
                row, "sglang_ple_cache_payload_digest", context
            ),
            sglang_storage_mode=_optional_string(
                row, "sglang_storage_mode", context
            ),
            sglang_ple_nvme_queue_depth=_optional_int(
                row, "sglang_ple_nvme_queue_depth", context, default=None
            ),
            sglang_ple_nvme_max_batch_pages=_optional_int(
                row, "sglang_ple_nvme_max_batch_pages", context, default=None
            ),
            sglang_ple_nvme_cache_pages=_optional_int(
                row, "sglang_ple_nvme_cache_pages", context, default=None
            ),
            recipe_source=_optional_string(row, "recipe_source", context),
            recipe_revision=_optional_string(row, "recipe_revision", context),
            request_body_json=_optional_string(
                row, "request_body_json", context
            ),
            runtime_python=_optional_string(row, "runtime_python", context),
            runtime_binary=_optional_string(row, "runtime_binary", context),
            runtime_digest=_optional_string(row, "runtime_digest", context),
            runtime_parallel=_optional_int(
                row, "runtime_parallel", context, default=None
            ),
            runtime_source_dir=_optional_string(
                row, "runtime_source_dir", context
            ),
            runtime_revision=_optional_string(row, "runtime_revision", context),
            model_file=_optional_string(row, "model_file", context),
            model_digest=_optional_string(row, "model_digest", context),
            model_size_bytes=_optional_int(
                row, "model_size_bytes", context, default=None
            ),
            model_shards=_model_shards(row, "model_shards", context),
            mmproj_file=_optional_string(row, "mmproj_file", context),
            mmproj_digest=_optional_string(row, "mmproj_digest", context),
            mmproj_size_bytes=_optional_int(
                row, "mmproj_size_bytes", context, default=None
            ),
            prefix_cache_mode=_optional_string(
                row, "prefix_cache_mode", context
            ),
            support_status=(
                _optional_string(row, "support_status", context) or "exploratory"
            ),
        )
        validate_model(model, context=context)
        if model.id in models:
            raise ManifestError(f"{context}: duplicate model id {model.id!r}")
        models[model.id] = model
    return models


def load_suite(path: str | Path) -> SuiteSpec:
    """Load and validate one benchmark suite."""

    manifest_path = Path(path)
    document = _read_toml(manifest_path)
    _reject_unknown(
        document, {"schema_version", "suite", "cases"}, str(manifest_path)
    )
    version = _check_schema_version(document, manifest_path)
    suite_row = document.get("suite")
    if not isinstance(suite_row, dict):
        raise ManifestError(f"{manifest_path}: 'suite' must be a table")
    _reject_unknown(
        suite_row,
        {"id", "description", "protocol_digest"},
        f"{manifest_path}: suite",
    )

    rows = document.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ManifestError(f"{manifest_path}: 'cases' must be a non-empty array")
    cases: list[CaseSpec] = []
    for index, row in enumerate(rows):
        context = f"{manifest_path}: cases[{index}]"
        if not isinstance(row, dict):
            raise ManifestError(f"{context} must be a table")
        _reject_unknown(row, _CASE_KEYS, context)
        case = CaseSpec(
            id=_required_string(row, "id", context),
            kind=_required_string(row, "kind", context).lower(),
            requires=_string_tuple(row, "requires", context, required=True),
            warmups=_optional_int(row, "warmups", context, default=0),
            repetitions=_optional_int(row, "repetitions", context, default=1),
            max_output_tokens=_optional_int(
                row, "max_output_tokens", context, default=1
            ),
            temperature=_optional_number(row, "temperature", context, default=0.0),
            concurrency=_optional_int(row, "concurrency", context, default=1),
            prompt_repetitions=_optional_int(
                row, "prompt_repetitions", context, default=0
            ),
            max_turns=_optional_int(row, "max_turns", context, default=1),
        )
        validate_case(case, context=context)
        cases.append(case)

    suite = SuiteSpec(
        id=_required_string(suite_row, "id", f"{manifest_path}: suite"),
        description=_optional_string(
            suite_row, "description", f"{manifest_path}: suite"
        )
        or "",
        cases=tuple(cases),
        schema_version=version,
        protocol_digest=_optional_string(
            suite_row, "protocol_digest", f"{manifest_path}: suite"
        ),
    )
    validate_suite(suite, context=str(manifest_path))
    return suite


def validate_model(model: ModelSpec, *, context: str = "model") -> None:
    """Validate a model dataclass, including instances built outside the loader."""

    _validate_id(model.id, f"{context}.id")
    for name, value in (
        ("backend", model.backend),
        ("source", model.source),
        ("served_name", model.served_name),
        ("architecture", model.architecture),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ManifestError(f"{context}.{name} must be a non-empty string")
    if model.backend not in KNOWN_BACKENDS:
        raise ManifestError(
            f"{context}.backend must be one of {sorted(KNOWN_BACKENDS)}"
        )
    _validate_unique_values(model.tasks, f"{context}.tasks", KNOWN_TASKS)
    if model.max_context <= 0:
        raise ManifestError(f"{context}.max_context must be positive")
    if model.native_context is not None and model.native_context < model.max_context:
        raise ManifestError(
            f"{context}.native_context must be at least max_context"
        )
    if model.startup_timeout_s <= 0:
        raise ManifestError(f"{context}.startup_timeout_s must be positive")
    if model.estimated_ram_gib is not None and model.estimated_ram_gib <= 0:
        raise ManifestError(f"{context}.estimated_ram_gib must be positive")
    densespark_fields = (
        model.densespark_pq_file,
        model.densespark_pq_digest,
        model.densespark_pq_size_bytes,
    )
    densespark_candidate = is_densespark_profile(model) or any(
        value is not None for value in densespark_fields
    )
    if densespark_candidate:
        try:
            validate_densespark_profile(model)
        except DenseSparkContractError as error:
            raise ManifestError(f"{context}: {error}") from error
    if model.id in MATCHED_PROMPT_GRAPH_PROFILE_IDS:
        validate_matched_prompt_graph_model(model, context=context)
    host_safety_fields = {
        "host_safety_min_memavailable_gib": (
            model.host_safety_min_memavailable_gib
        ),
        "host_safety_max_swap_growth_mib": (
            model.host_safety_max_swap_growth_mib
        ),
        "host_safety_max_starting_swap_mib": (
            model.host_safety_max_starting_swap_mib
        ),
    }
    configured_host_safety = {
        name: value
        for name, value in host_safety_fields.items()
        if value is not None
    }
    if configured_host_safety and len(configured_host_safety) != len(
        host_safety_fields
    ):
        raise ManifestError(
            f"{context} host-safety thresholds must be configured together"
        )
    if configured_host_safety and model.backend != "sglang":
        raise ManifestError(
            f"{context} host-safety thresholds are supported only for sglang"
        )
    for name, value in configured_host_safety.items():
        if value is None or value <= 0:
            raise ManifestError(f"{context}.{name} must be positive")
    if model.lifecycle not in {"docker", "existing", "subprocess"}:
        raise ManifestError(
            f"{context}.lifecycle must be 'docker', 'existing', or 'subprocess'"
        )
    if model.cache_dir not in {"project", "user"}:
        raise ManifestError(f"{context}.cache_dir must be 'project' or 'user'")
    _validate_fetch_patterns(
        model.fetch_allow_patterns, f"{context}.fetch_allow_patterns"
    )
    _validate_fetch_patterns(
        model.fetch_ignore_patterns, f"{context}.fetch_ignore_patterns"
    )
    for name, value in (
        ("weight_size_bytes", model.weight_size_bytes),
        ("weight_file_count", model.weight_file_count),
        ("draft_weight_size_bytes", model.draft_weight_size_bytes),
        ("draft_model_size_bytes", model.draft_model_size_bytes),
    ):
        if value is not None and value <= 0:
            raise ManifestError(f"{context}.{name} must be positive")
    draft_fields = (model.draft_source, model.draft_revision)
    if any(value is not None for value in draft_fields) and not all(
        value is not None for value in draft_fields
    ):
        raise ManifestError(
            f"{context}.draft_source and draft_revision must be set together"
        )
    if model.draft_source is not None:
        if model.backend not in {"llamacpp", "sglang"}:
            raise ManifestError(
                f"{context}.draft_source is supported only for llamacpp or sglang"
            )
        if not _HF_REPOSITORY_PATTERN.fullmatch(model.draft_source):
            raise ManifestError(
                f"{context}.draft_source must be a Hugging Face repository ID"
            )
        if not model.draft_revision or not _COMMIT_PATTERN.fullmatch(
            model.draft_revision
        ):
            raise ManifestError(
                f"{context}.draft_revision must be a full lowercase commit SHA"
            )
        if not model.revision or not _COMMIT_PATTERN.fullmatch(model.revision):
            raise ManifestError(
                f"{context}.revision must be a full lowercase commit SHA "
                "when a draft snapshot is configured"
            )
        if not _HF_REPOSITORY_PATTERN.fullmatch(model.source):
            raise ManifestError(
                f"{context}.source must be a Hugging Face repository ID "
                "when a draft snapshot is configured"
            )
        same_snapshot = (model.source, model.revision) == (
            model.draft_source,
            model.draft_revision,
        )
        target_model_paths = (
            (model.model_file,)
            if model.model_file is not None
            else tuple(shard.path for shard in model.model_shards)
        )
        target_model_digests = (
            (model.model_digest,)
            if model.model_digest is not None
            else tuple(shard.digest for shard in model.model_shards)
        )
        if (
            same_snapshot
            and model.backend == "llamacpp"
            and model.draft_model_file in target_model_paths
        ):
            raise ManifestError(
                f"{context}.draft_model_file must differ from every target "
                "model artifact in a shared snapshot"
            )
        same_snapshot_sidecar = (
            model.backend == "llamacpp"
            and bool(target_model_paths)
            and model.draft_model_file is not None
            and model.draft_model_file not in target_model_paths
        )
        if same_snapshot and not same_snapshot_sidecar:
            raise ManifestError(
                f"{context}.draft snapshot must differ from the target snapshot "
                "unless it pins a distinct llama.cpp sidecar file"
            )
        if (
            same_snapshot
            and same_snapshot_sidecar
            and model.draft_model_digest is not None
            and model.draft_model_digest in target_model_digests
        ):
            raise ManifestError(
                f"{context}.draft_model_digest must differ from every target "
                "model artifact digest in a shared snapshot"
            )
    elif any(
        value is not None
        for value in (
            model.draft_weight_size_bytes,
            model.draft_model_file,
            model.draft_model_digest,
            model.draft_model_size_bytes,
        )
    ):
        raise ManifestError(
            f"{context}.draft artifact fields require a draft snapshot"
        )
    draft_model_fields = (
        model.draft_model_file,
        model.draft_model_digest,
        model.draft_model_size_bytes,
    )
    if any(value is not None for value in draft_model_fields) and not all(
        value is not None for value in draft_model_fields
    ):
        raise ManifestError(
            f"{context}.draft_model_file, draft_model_digest, and "
            "draft_model_size_bytes must be set together"
        )
    if model.backend != "llamacpp" and any(
        value is not None for value in draft_model_fields
    ):
        raise ManifestError(
            f"{context}.draft_model_* fields are supported only for llamacpp"
        )
    if model.sglang_allow_hf_metadata_probe and (
        model.backend != "sglang" or model.draft_source is None
    ):
        raise ManifestError(
            f"{context}.sglang_allow_hf_metadata_probe requires an sglang "
            "draft snapshot"
        )
    if (
        model.sglang_source_overlays
        or model.sglang_ple_mmap
        or model.sglang_ple_omitted
        or model.sglang_ple_cache_mode is not None
    ) and (
        model.backend != "sglang"
    ):
        raise ManifestError(
            f"{context}.sglang source overlays and PLE mmap are supported "
            "only for sglang"
        )
    if model.sglang_source_overlays:
        if len(model.sglang_source_overlays) > 8:
            raise ManifestError(
                f"{context}.sglang_source_overlays supports at most 8 files"
            )
        seen_host_paths: set[str] = set()
        seen_container_paths: set[str] = set()
        container_root = PurePosixPath(
            "/sgl-workspace/sglang/python/sglang"
        )
        for index, overlay in enumerate(model.sglang_source_overlays):
            overlay_context = f"{context}.sglang_source_overlays[{index}]"
            if not isinstance(overlay, SGLangSourceOverlay):
                raise ManifestError(
                    f"{overlay_context} must be an SGLangSourceOverlay"
                )
            host_path = overlay.host_path
            if (
                not isinstance(host_path, str)
                or not host_path
                or host_path != host_path.strip()
                or host_path.startswith(("-", "/", "\\", "~"))
                or "\\" in host_path
                or ":" in host_path
                or any(character in host_path for character in "*?[")
                or any(ord(character) < 32 for character in host_path)
            ):
                raise ManifestError(
                    f"{overlay_context}.host_path must be a safe relative path"
                )
            relative_host = PurePosixPath(host_path)
            if relative_host.is_absolute() or any(
                part in {"", ".", ".."} for part in relative_host.parts
            ):
                raise ManifestError(
                    f"{overlay_context}.host_path must be a safe relative path"
                )
            if relative_host.suffix != ".py":
                raise ManifestError(
                    f"{overlay_context}.host_path must name a Python source file"
                )
            container_path = overlay.container_path
            if (
                not isinstance(container_path, str)
                or not container_path
                or container_path != container_path.strip()
                or "\\" in container_path
                or ":" in container_path
                or any(ord(character) < 32 for character in container_path)
                or any(
                    component in {"", ".", ".."}
                    for component in container_path.removeprefix("/").split("/")
                )
            ):
                raise ManifestError(
                    f"{overlay_context}.container_path must be a safe absolute path"
                )
            absolute_container = PurePosixPath(container_path)
            try:
                container_relative = absolute_container.relative_to(
                    container_root
                )
            except ValueError as error:
                raise ManifestError(
                    f"{overlay_context}.container_path must be beneath "
                    f"{container_root}"
                ) from error
            if (
                not absolute_container.is_absolute()
                or not container_relative.parts
                or absolute_container.suffix != ".py"
            ):
                raise ManifestError(
                    f"{overlay_context}.container_path must name a Python "
                    f"source file beneath {container_root}"
                )
            if relative_host.name != absolute_container.name:
                raise ManifestError(
                    f"{overlay_context} host and container basenames must match"
                )
            if not _DIGEST_PATTERN.fullmatch(overlay.digest):
                raise ManifestError(
                    f"{overlay_context}.digest must be a sha256 digest"
                )
            if host_path in seen_host_paths:
                raise ManifestError(
                    f"{context}.sglang_source_overlays has duplicate host paths"
                )
            if container_path in seen_container_paths:
                raise ManifestError(
                    f"{context}.sglang_source_overlays has duplicate container paths"
                )
            seen_host_paths.add(host_path)
            seen_container_paths.add(container_path)
    if model.sglang_ple_mmap and not model.sglang_source_overlays:
        raise ManifestError(
            f"{context}.sglang_ple_mmap requires digest-pinned source overlays"
        )
    if model.sglang_ple_mmap and model.draft_source is not None:
        raise ManifestError(
            f"{context}.sglang_ple_mmap requires an embedded draft and a "
            "single PLE backing mount"
        )
    omission_override_values: list[object] = []
    for index, argument in enumerate(model.args):
        if argument == "--json-model-override-args":
            if index + 1 >= len(model.args):
                raise ManifestError(
                    f"{context}.args leaves --json-model-override-args without a value"
                )
            try:
                omission_override_values.append(json.loads(model.args[index + 1]))
            except json.JSONDecodeError as error:
                raise ManifestError(
                    f"{context}.args has invalid --json-model-override-args JSON"
                ) from error
        elif argument.startswith("--json-model-override-args="):
            try:
                omission_override_values.append(
                    json.loads(argument.split("=", 1)[1])
                )
            except json.JSONDecodeError as error:
                raise ManifestError(
                    f"{context}.args has invalid --json-model-override-args JSON"
                ) from error
    omission_sentinels = [
        value
        for value in omission_override_values
        if isinstance(value, dict) and "sparkbench_omit_ple" in value
    ]
    if model.sglang_ple_omitted:
        if model.sglang_ple_mmap or model.sglang_ple_cache_mode is not None:
            raise ManifestError(
                f"{context}.sglang_ple_omitted forbids PLE mmap/cache reuse"
            )
        if any(
            value is not None
            for value in (
                model.sglang_ple_cache_marker_digest,
                model.sglang_ple_cache_payload_digest,
            )
        ):
            raise ManifestError(
                f"{context}.sglang_ple_omitted forbids PLE cache digests"
            )
        if not model.sglang_source_overlays:
            raise ManifestError(
                f"{context}.sglang_ple_omitted requires digest-pinned overlays"
            )
        if (
            model.source,
            model.revision,
            model.recipe_source,
            model.recipe_revision,
        ) != _QWEN38_PLE_ABLATION_IDENTITY:
            raise ManifestError(
                f"{context}.sglang_ple_omitted is pinned to the exact Qwen3.8 "
                "artifact and audited recipe"
            )
        if model.draft_source is not None:
            raise ManifestError(
                f"{context}.sglang_ple_omitted requires the embedded draft"
            )
        actual_omission_overlays = {
            overlay.container_path: overlay.digest
            for overlay in model.sglang_source_overlays
        }
        if actual_omission_overlays != _QWEN38_PLE_ABLATION_OVERLAYS:
            raise ManifestError(
                f"{context}.sglang_ple_omitted requires the exact "
                "ablation-capable overlays"
            )
        if "--ple-offload-embedding" in model.args:
            raise ManifestError(
                f"{context}.sglang_ple_omitted forbids --ple-offload-embedding"
            )
        canonical_override_pairs = tuple(
            tuple(model.args[index : index + 2])
            for index, argument in enumerate(model.args[:-1])
            if argument == "--json-model-override-args"
        )
        inline_overrides = tuple(
            argument
            for argument in model.args
            if argument.startswith("--json-model-override-args=")
        )
        if (
            omission_override_values != [{"sparkbench_omit_ple": True}]
            or canonical_override_pairs
            != (("--json-model-override-args", '{"sparkbench_omit_ple":true}'),)
            or inline_overrides
        ):
            raise ManifestError(
                f"{context}.sglang_ple_omitted requires exactly one canonical "
                "sparkbench_omit_ple=true model override and no other model "
                "overrides"
            )
        if "ple-omitted" not in (model.quantization or ""):
            raise ManifestError(
                f"{context}.sglang_ple_omitted requires an explicit "
                "ple-omitted quantization label"
            )
    elif omission_sentinels:
        raise ManifestError(
            f"{context}.sparkbench_omit_ple override requires "
            "sglang_ple_omitted=true"
        )
    ple_cache_pins = (
        model.sglang_ple_cache_marker_digest,
        model.sglang_ple_cache_payload_digest,
    )
    if model.sglang_ple_cache_mode is None:
        if any(value is not None for value in ple_cache_pins):
            raise ManifestError(
                f"{context}.sglang_ple_cache_* digests require an explicit "
                "sglang_ple_cache_mode"
            )
    elif model.sglang_ple_cache_mode != "readonly":
        raise ManifestError(
            f"{context}.sglang_ple_cache_mode must be 'readonly' when set"
        )
    elif not model.sglang_ple_mmap:
        raise ManifestError(
            f"{context}.sglang_ple_cache_mode requires sglang_ple_mmap=true"
        )
    elif not all(
        isinstance(value, str) and _DIGEST_PATTERN.fullmatch(value)
        for value in ple_cache_pins
    ):
        raise ManifestError(
            f"{context}.sglang_ple_cache_mode requires exact marker and "
            "payload sha256 digests"
        )
    recipe_fields = (model.recipe_source, model.recipe_revision)
    if any(value is not None for value in recipe_fields) and not all(
        value is not None for value in recipe_fields
    ):
        raise ManifestError(
            f"{context}.recipe_source and recipe_revision must be set together"
        )
    if model.recipe_source is not None:
        if not _HF_REPOSITORY_PATTERN.fullmatch(model.recipe_source):
            raise ManifestError(
                f"{context}.recipe_source must be an owner/repository name"
            )
        if not model.recipe_revision or not _COMMIT_PATTERN.fullmatch(
            model.recipe_revision
        ):
            raise ManifestError(
                f"{context}.recipe_revision must be a full lowercase commit SHA"
            )
    if model.sglang_allow_hf_metadata_probe and model.recipe_source is None:
        raise ManifestError(
            f"{context}.sglang_allow_hf_metadata_probe requires pinned recipe "
            "provenance"
        )
    if model.sglang_source_overlays and model.recipe_source is None:
        raise ManifestError(
            f"{context}.sglang_source_overlays requires pinned recipe provenance"
        )
    if model.support_status not in KNOWN_SUPPORT_STATUSES:
        raise ManifestError(
            f"{context}.support_status must be one of {sorted(KNOWN_SUPPORT_STATUSES)}"
        )
    if model.lifecycle == "docker" and not model.image:
        raise ManifestError(f"{context}.image is required for Docker lifecycle")
    if model.backend == "transformers":
        if model.lifecycle != "subprocess":
            raise ManifestError(
                f"{context}.lifecycle must be 'subprocess' for transformers"
            )
        if not model.runtime_python or not Path(model.runtime_python).is_absolute():
            raise ManifestError(
                f"{context}.runtime_python must be an absolute path for transformers"
            )
        if model.endpoint != "offline://transformers":
            raise ManifestError(
                f"{context}.endpoint must be 'offline://transformers'"
            )
    if model.backend == "trtllm":
        if model.lifecycle != "docker":
            raise ManifestError(
                f"{context}.lifecycle must be 'docker' for trtllm"
            )
        if model.endpoint != "offline://trtllm":
            raise ManifestError(
                f"{context}.endpoint must be 'offline://trtllm'"
            )
        if not model.image_digest:
            raise ManifestError(
                f"{context}.image_digest is required for trtllm"
            )
    mmproj_fields = (
        model.mmproj_file,
        model.mmproj_digest,
        model.mmproj_size_bytes,
    )
    if any(value is not None for value in mmproj_fields) and not all(
        value is not None for value in mmproj_fields
    ):
        raise ManifestError(
            f"{context}.mmproj_file, mmproj_digest, and mmproj_size_bytes "
            "must be set together"
        )
    if model.backend != "llamacpp" and any(
        value is not None for value in mmproj_fields
    ):
        raise ManifestError(
            f"{context}.mmproj_* fields are supported only for llamacpp"
        )
    model_file_fields = (
        model.model_file,
        model.model_digest,
        model.model_size_bytes,
    )
    if any(value is not None for value in model_file_fields) and not all(
        value is not None for value in model_file_fields
    ):
        raise ManifestError(
            f"{context}.model_file, model_digest, and model_size_bytes "
            "must be set together"
        )
    if model.model_shards and any(
        value is not None for value in model_file_fields
    ):
        raise ManifestError(
            f"{context}.model_shards cannot be combined with single-file "
            "model_* fields"
        )
    if model.backend != "llamacpp" and model.model_shards:
        raise ManifestError(
            f"{context}.model_shards is supported only for llamacpp"
        )
    if model.backend == "llamacpp":
        if model.lifecycle != "subprocess":
            raise ManifestError(
                f"{context}.lifecycle must be 'subprocess' for llamacpp"
            )
        if model.image is not None:
            raise ManifestError(f"{context}.image must be omitted for llamacpp")
        for name, value in (
            ("runtime_binary", model.runtime_binary),
            ("runtime_source_dir", model.runtime_source_dir),
        ):
            portable_home_path = bool(
                value
                and value.startswith("~/")
                and len(value) > 2
                and ".." not in Path(value[2:]).parts
                and not Path(value[2:]).is_absolute()
            )
            if not value or not (
                Path(value).is_absolute() or portable_home_path
            ):
                raise ManifestError(
                    f"{context}.{name} must be an absolute or portable ~/ path "
                    "for llamacpp"
                )
        if not model.runtime_revision or not _COMMIT_PATTERN.fullmatch(
            model.runtime_revision
        ):
            raise ManifestError(
                f"{context}.runtime_revision must be a full lowercase commit SHA"
            )
        if not model.revision or not _COMMIT_PATTERN.fullmatch(model.revision):
            raise ManifestError(
                f"{context}.revision must be a full lowercase commit SHA for llamacpp"
            )
        has_model_file = model.model_file is not None
        has_model_shards = bool(model.model_shards)
        if has_model_file == has_model_shards:
            raise ManifestError(
                f"{context} must configure exactly one of model_file or "
                "model_shards for llamacpp"
            )
        if has_model_file:
            if (
                Path(str(model.model_file)).name != model.model_file
                or not model.model_file.lower().endswith(".gguf")
            ):
                raise ManifestError(
                    f"{context}.model_file must be one safe GGUF filename"
                )
            if model.model_size_bytes is None or model.model_size_bytes <= 0:
                raise ManifestError(
                    f"{context}.model_size_bytes must be positive for llamacpp"
                )
        else:
            _validate_model_shards(
                model.model_shards, f"{context}.model_shards"
            )
            shard_bytes = sum(shard.size_bytes for shard in model.model_shards)
            if model.weight_size_bytes != shard_bytes:
                raise ManifestError(
                    f"{context}.weight_size_bytes must equal the exact GGUF "
                    f"shard total {shard_bytes}"
                )
            if model.weight_file_count != len(model.model_shards):
                raise ManifestError(
                    f"{context}.weight_file_count must equal the exact GGUF "
                    f"shard count {len(model.model_shards)}"
                )
        has_draft_model = model.draft_model_file is not None
        if has_draft_model:
            if (
                Path(str(model.draft_model_file)).name != model.draft_model_file
                or not model.draft_model_file.lower().endswith(".gguf")
            ):
                raise ManifestError(
                    f"{context}.draft_model_file must be one safe GGUF filename"
                )
            if model.draft_source is None or model.draft_revision is None:
                raise ManifestError(
                    f"{context}.draft_model_file requires a pinned draft snapshot"
                )
            spec_types = _llamacpp_spec_types(model.args)
            if spec_types not in {("draft-dflash",), ("draft-mtp",)}:
                raise ManifestError(
                    f"{context}.llamacpp draft sidecar requires exactly "
                    "--spec-type draft-dflash or draft-mtp"
                )
            if _llamacpp_positive_option(model.args, "--spec-draft-n-max") is None:
                raise ManifestError(
                    f"{context}.llamacpp draft sidecar requires one positive "
                    "--spec-draft-n-max"
                )
        elif model.draft_source is not None:
            raise ManifestError(
                f"{context}.llamacpp draft snapshot requires draft_model_file, "
                "draft_model_digest, and draft_model_size_bytes"
            )
        has_mmproj = model.mmproj_file is not None
        if has_mmproj:
            if (
                Path(str(model.mmproj_file)).name != model.mmproj_file
                or not model.mmproj_file.lower().endswith(".gguf")
            ):
                raise ManifestError(
                    f"{context}.mmproj_file must be one safe GGUF filename"
                )
            if model.mmproj_file == model.model_file:
                raise ManifestError(
                    f"{context}.mmproj_file must differ from model_file"
                )
            if any(
                model.mmproj_file == shard.path for shard in model.model_shards
            ):
                raise ManifestError(
                    f"{context}.mmproj_file must differ from model_shards"
                )
            if model.mmproj_size_bytes is None or model.mmproj_size_bytes <= 0:
                raise ManifestError(
                    f"{context}.mmproj_size_bytes must be positive for llamacpp"
                )
        if "vision" in model.tasks and not has_mmproj:
            raise ManifestError(
                f"{context}.vision task requires the complete mmproj artifact"
            )
        if has_mmproj and "vision" not in model.tasks:
            raise ManifestError(
                f"{context}.mmproj artifact requires the vision task"
            )
        target_files = (
            tuple(shard.path for shard in model.model_shards)
            if has_model_shards
            else (str(model.model_file),)
        )
        exact_files = (
            (*target_files, str(model.mmproj_file))
            if has_mmproj
            else target_files
        )
        if model.fetch_allow_patterns != exact_files:
            raise ManifestError(
                f"{context}.fetch_allow_patterns must contain only the exact "
                "model_file and configured mmproj_file for llamacpp"
            )
        if model.native_context is None:
            raise ManifestError(
                f"{context}.native_context is required for llamacpp"
            )
        if model.runtime_parallel is None or model.runtime_parallel <= 0:
            raise ManifestError(
                f"{context}.runtime_parallel must be positive for llamacpp"
            )
        if model.native_context != model.max_context * model.runtime_parallel:
            raise ManifestError(
                f"{context}.native_context must equal max_context * "
                "runtime_parallel for llamacpp"
            )
        parsed_endpoint = urlsplit(model.endpoint)
        if (
            parsed_endpoint.scheme != "http"
            or parsed_endpoint.hostname != "127.0.0.1"
            or parsed_endpoint.port is None
            or parsed_endpoint.path.rstrip("/") != "/v1"
            or parsed_endpoint.username is not None
            or parsed_endpoint.password is not None
            or parsed_endpoint.query
            or parsed_endpoint.fragment
        ):
            raise ManifestError(
                f"{context}.endpoint must be canonical http://127.0.0.1:<port>/v1 for llamacpp"
            )
        reserved = sorted(
            argument.split("=", 1)[0]
            for argument in model.args
            if argument.split("=", 1)[0] in _LLAMACPP_RESERVED_ARGS
        )
        if reserved:
            raise ManifestError(
                f"{context}.args contains runtime-owned llamacpp option(s): "
                + ", ".join(reserved)
            )
        forbidden = sorted(
            argument.split("=", 1)[0]
            for argument in model.args
            if argument.split("=", 1)[0] in _LLAMACPP_FORBIDDEN_ARGS
        )
        if forbidden:
            raise ManifestError(
                f"{context}.args contains unsafe llamacpp option(s): "
                + ", ".join(forbidden)
            )
    if model.prefix_cache_mode is not None:
        if model.prefix_cache_mode not in PREFIX_CACHE_MODES:
            raise ManifestError(
                f"{context}.prefix_cache_mode must be one of "
                f"{sorted(PREFIX_CACHE_MODES)}"
            )
        if model.backend != "llamacpp":
            raise ManifestError(
                f"{context}.prefix_cache_mode is supported only for llamacpp"
            )
        if model.runtime_parallel != 1:
            raise ManifestError(
                f"{context}.prefix_cache_mode requires runtime_parallel = 1"
            )
        if model.request_body_json is not None:
            raise ManifestError(
                f"{context}.prefix_cache_mode reserves request_body_json for "
                "the native cache adapter"
            )
        argument_values = tuple(str(argument) for argument in model.args)
        argument_names = [argument.split("=", 1)[0] for argument in argument_values]
        cache_controls = [
            argument
            for argument in argument_values
            if argument.split("=", 1)[0]
            in {"--cache-prompt", "--no-cache-prompt"}
        ]
        if any(
            argument not in {"--cache-prompt", "--no-cache-prompt"}
            for argument in cache_controls
        ):
            raise ManifestError(
                f"{context}.prefix_cache_mode requires a literal standalone "
                "--cache-prompt or --no-cache-prompt control"
            )
        enabled = argument_values.count("--cache-prompt")
        disabled = argument_values.count("--no-cache-prompt")
        if enabled + disabled != 1:
            raise ManifestError(
                f"{context}.prefix_cache_mode requires exactly one explicit "
                "--cache-prompt or --no-cache-prompt control"
            )
        if model.prefix_cache_mode == "on" and enabled != 1:
            raise ManifestError(
                f"{context}.prefix_cache_mode = 'on' requires --cache-prompt"
            )
        if model.prefix_cache_mode == "off" and disabled != 1:
            raise ManifestError(
                f"{context}.prefix_cache_mode = 'off' requires --no-cache-prompt"
            )
        if "--cache-reuse" in argument_names:
            raise ManifestError(
                f"{context}.prefix_cache_mode must not enable --cache-reuse"
            )
    if model.backend == "sglang":
        reserved = sorted(
            argument.split("=", 1)[0]
            for argument in model.args
            if argument.split("=", 1)[0] in _SGLANG_RESERVED_ARGS
        )
        if reserved:
            raise ManifestError(
                f"{context}.args contains runtime-owned sglang option(s): "
                + ", ".join(reserved)
            )
    storage_fields = {
        "local_image_id": model.local_image_id,
        "sglang_ple_nvme_queue_depth": model.sglang_ple_nvme_queue_depth,
        "sglang_ple_nvme_max_batch_pages": (
            model.sglang_ple_nvme_max_batch_pages
        ),
        "sglang_ple_nvme_cache_pages": model.sglang_ple_nvme_cache_pages,
    }
    configured_storage_fields = {
        name: value for name, value in storage_fields.items() if value is not None
    }
    if model.sglang_storage_mode is None:
        native_storage_fields = {
            name: value
            for name, value in configured_storage_fields.items()
            if name != "local_image_id"
        }
        if native_storage_fields or (
            model.local_image_id is not None and not densespark_candidate
        ):
            raise ManifestError(
                f"{context}.local_image_id and native PLE fields require "
                "sglang_storage_mode or the exact managed DenseSpark profile"
            )
    else:
        if model.sglang_storage_mode != SM121_STORAGE_MODE:
            raise ManifestError(
                f"{context}.sglang_storage_mode is not a supported native "
                "storage mode"
            )
        if len(configured_storage_fields) != len(storage_fields):
            raise ManifestError(
                f"{context}.sglang_storage_mode requires local image identity "
                "and all native PLE parameters"
            )
        if model.backend != "sglang":
            raise ManifestError(
                f"{context}.sglang_storage_mode is supported only for sglang"
            )
        try:
            if is_sm121_agent_admission_candidate(model):
                validate_sm121_agent_admission_profile(model)
            elif is_sm121_cuda_graph_candidate(model):
                validate_sm121_cuda_graph_candidate(model)
            elif is_sm121_cache_semantic_candidate(model):
                validate_sm121_cache_semantic_candidate(model)
            elif is_sm121_cache_performance_candidate(model):
                validate_sm121_cache_performance_candidate(model)
            elif is_sm121_chunked_prefill_performance_candidate(model):
                validate_sm121_chunked_prefill_performance_candidate(model)
            else:
                validate_sm121_storage_candidate(model)
        except (
            SM121StorageCandidateError,
            SM121AgentAdmissionError,
            SM121CudaGraphError,
            SM121CacheSemanticError,
            SM121CachePerformanceError,
            SM121ChunkedPrefillPerformanceError,
        ) as error:
            raise ManifestError(f"{context}: {error}") from error
    if model.image_digest and not _DIGEST_PATTERN.fullmatch(model.image_digest):
        raise ManifestError(f"{context}.image_digest must be a sha256 digest")
    for name, digest in (
        ("runtime_digest", model.runtime_digest),
        ("model_digest", model.model_digest),
        ("mmproj_digest", model.mmproj_digest),
        ("draft_model_digest", model.draft_model_digest),
    ):
        if digest is not None and not _DIGEST_PATTERN.fullmatch(digest):
            raise ManifestError(f"{context}.{name} must be a sha256 digest")
    if model.backend == "llamacpp" and model.runtime_digest is None:
        raise ManifestError(
            f"{context}.runtime_digest is required for llamacpp"
        )
    if model.backend not in {"transformers", "trtllm"}:
        _validate_endpoint(model.endpoint, f"{context}.endpoint")
    if any(not isinstance(argument, str) or not argument for argument in model.args):
        raise ManifestError(f"{context}.args must contain non-empty strings")
    if model.request_body_json is not None:
        try:
            request_body = json.loads(model.request_body_json)
        except json.JSONDecodeError as error:
            raise ManifestError(
                f"{context}.request_body_json must contain valid JSON"
            ) from error
        if not isinstance(request_body, dict):
            raise ManifestError(
                f"{context}.request_body_json must decode to a JSON object"
            )


def validate_case(case: CaseSpec, *, context: str = "case") -> None:
    """Validate a case dataclass."""

    _validate_id(case.id, f"{context}.id")
    if case.kind not in KNOWN_CASE_KINDS:
        raise ManifestError(
            f"{context}.kind must be one of {sorted(KNOWN_CASE_KINDS)}"
        )
    _validate_unique_values(case.requires, f"{context}.requires", KNOWN_TASKS)
    if case.warmups < 0:
        raise ManifestError(f"{context}.warmups must not be negative")
    if case.repetitions <= 0:
        raise ManifestError(f"{context}.repetitions must be positive")
    if case.concurrency <= 0:
        raise ManifestError(f"{context}.concurrency must be positive")
    if case.prompt_repetitions < 0:
        raise ManifestError(f"{context}.prompt_repetitions must not be negative")
    if case.max_output_tokens <= 0:
        raise ManifestError(f"{context}.max_output_tokens must be positive")
    if case.max_turns <= 0:
        raise ManifestError(f"{context}.max_turns must be positive")
    if not 0 <= case.temperature <= 2:
        raise ManifestError(f"{context}.temperature must be between 0 and 2")
    try:
        matched_prompt = matched_prompt_protocol(case.id)
    except ValueError as error:
        raise ManifestError(f"{context}.id {error}") from error
    if matched_prompt is not None:
        expected = {
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
        for field, wanted in expected.items():
            if getattr(case, field) != wanted:
                raise ManifestError(
                    f"{context}.{field} must be {wanted!r} for "
                    f"{matched_prompt!r}"
                )
    try:
        varied_context = case.varied_context_needle
    except ValueError as error:
        raise ManifestError(f"{context}.id {error}") from error
    if varied_context is not None:
        if case.kind != "capability":
            raise ManifestError(
                f"{context}.kind must be 'capability' for varied-context needle cases"
            )
        if case.requires != ("chat",):
            raise ManifestError(
                f"{context}.requires must be ['chat'] for varied-context needle cases"
            )
        if case.warmups != 0:
            raise ManifestError(
                f"{context}.warmups must be 0 for varied-context needle cases"
            )
        if case.repetitions != 1:
            raise ManifestError(
                f"{context}.repetitions must be 1 for varied-context needle cases"
            )
        if case.concurrency != 1:
            raise ManifestError(
                f"{context}.concurrency must be 1 for varied-context needle cases"
            )
        if case.temperature != 0:
            raise ManifestError(
                f"{context}.temperature must be 0 for varied-context needle cases"
            )
        if case.prompt_repetitions != varied_context.filler_records:
            raise ManifestError(
                f"{context}.prompt_repetitions must match the varied-context "
                "needle record count"
            )
        if case.max_output_tokens < VARIED_CONTEXT_NEEDLE_MIN_OUTPUT_TOKENS:
            raise ManifestError(
                f"{context}.max_output_tokens must be at least "
                f"{VARIED_CONTEXT_NEEDLE_MIN_OUTPUT_TOKENS} for varied-context "
                "needle cases"
            )
    if case.kind == "decode" and case.prompt_repetitions != 0:
        raise ManifestError(
            f"{context}.prompt_repetitions must be 0 for decode cases"
        )
    if case.kind == "prefill" and case.prompt_repetitions <= 0:
        raise ManifestError(
            f"{context}.prompt_repetitions must be positive for prefill cases"
        )
    if case.kind == "cache":
        if case.requires != ("chat",):
            raise ManifestError(
                f"{context}.requires must be ['chat'] for cache cases"
            )
        if case.warmups != 0:
            raise ManifestError(f"{context}.warmups must be 0 for cache cases")
        if case.repetitions != 5:
            raise ManifestError(
                f"{context}.repetitions must be 5 for cache paired blocks"
            )
        if case.max_output_tokens != 128:
            raise ManifestError(
                f"{context}.max_output_tokens must be 128 for cache cases"
            )
        if case.concurrency != 1:
            raise ManifestError(
                f"{context}.concurrency must be 1 for cache cases"
            )
        if case.prompt_repetitions <= 0:
            raise ManifestError(
                f"{context}.prompt_repetitions must be positive for cache cases"
            )
        if case.temperature != 0:
            raise ManifestError(f"{context}.temperature must be 0 for cache cases")
    if case.kind == "quality":
        if case.requires != ("chat",):
            raise ManifestError(
                f"{context}.requires must be ['chat'] for quality cases"
            )
        if case.prompt_repetitions != 0:
            raise ManifestError(
                f"{context}.prompt_repetitions must be 0 for quality cases"
            )
        if case.temperature != 0:
            raise ManifestError(f"{context}.temperature must be 0 for quality cases")
    if case.kind == "agentic":
        if case.id not in KNOWN_AGENTIC_CASE_IDS:
            raise ManifestError(f"{context}.id is not a supported agentic scenario")
        if case.repetitions != 3:
            raise ManifestError(
                f"{context}.repetitions must be 3 for the three fixed agentic variants"
            )
        if case.requires != ("chat", "tools"):
            raise ManifestError(
                f"{context}.requires must be ['chat', 'tools'] for agentic cases"
            )
        if case.warmups != 0:
            raise ManifestError(f"{context}.warmups must be 0 for agentic cases")
        if case.concurrency != 1:
            raise ManifestError(f"{context}.concurrency must be 1 for agentic cases")
        if case.prompt_repetitions != 0:
            raise ManifestError(
                f"{context}.prompt_repetitions must be 0 for agentic cases"
            )
        if case.temperature != 0:
            raise ManifestError(f"{context}.temperature must be 0 for agentic cases")
        if not 2 <= case.max_turns <= 8:
            raise ManifestError(
                f"{context}.max_turns must be between 2 and 8 for agentic cases"
            )
        if case.max_output_tokens < 2048:
            raise ManifestError(
                f"{context}.max_output_tokens must be at least 2048 for agentic cases"
            )
    if case.kind == "memory":
        if case.id not in KNOWN_MEMORY_OPERATION_CASE_IDS:
            raise ManifestError(
                f"{context}.id is not a supported memory-operation scenario"
            )
        if case.repetitions != MEMORY_OPERATION_VARIANT_COUNT:
            raise ManifestError(
                f"{context}.repetitions must be 3 for the three fixed memory variants"
            )
        if case.requires != ("chat", "json"):
            raise ManifestError(
                f"{context}.requires must be ['chat', 'json'] for memory cases"
            )
        if case.warmups != 0:
            raise ManifestError(f"{context}.warmups must be 0 for memory cases")
        if case.concurrency != 1:
            raise ManifestError(f"{context}.concurrency must be 1 for memory cases")
        if case.prompt_repetitions != 0:
            raise ManifestError(
                f"{context}.prompt_repetitions must be 0 for memory cases"
            )
        if case.temperature != 0:
            raise ManifestError(f"{context}.temperature must be 0 for memory cases")
        if case.max_output_tokens != MEMORY_OPERATION_OUTPUT_TOKENS:
            raise ManifestError(
                f"{context}.max_output_tokens must be "
                f"{MEMORY_OPERATION_OUTPUT_TOKENS} for memory cases"
            )
    if case.kind != "agentic" and case.max_turns != 1:
        raise ManifestError(f"{context}.max_turns must be 1 for non-agentic cases")
    if case.kind == "diffusion":
        if case.requires != ("diffusion",):
            raise ManifestError(
                f"{context}.requires must be ['diffusion'] for diffusion cases"
            )
        if case.temperature != 0:
            raise ManifestError(f"{context}.temperature must be 0 for diffusion cases")
        if case.concurrency != 1:
            raise ManifestError(f"{context}.concurrency must be 1 for diffusion cases")
        if case.max_output_tokens % 32:
            raise ManifestError(
                f"{context}.max_output_tokens must be divisible by block size 32"
            )


def validate_qwen38_27b_dspark_cuda_graph_suite(
    suite: SuiteSpec,
    *,
    context: str = "suite",
) -> None:
    """Require the exact matched-request-unique C1 D256 screen."""

    if getattr(suite, "id", None) != QWEN38_27B_DSPARK_CUDA_GRAPH_SUITE_ID:
        raise ManifestError(
            f"{context}.id must be {QWEN38_27B_DSPARK_CUDA_GRAPH_SUITE_ID!r}"
        )
    if (
        getattr(suite, "description", None)
        != QWEN38_27B_DSPARK_CUDA_GRAPH_SUITE_DESCRIPTION
    ):
        raise ManifestError(f"{context}.description changed")
    if getattr(suite, "schema_version", None) != SCHEMA_VERSION:
        raise ManifestError(f"{context}.schema_version changed")
    if getattr(suite, "protocol_digest", None) is not None:
        raise ManifestError(f"{context}.protocol_digest must be omitted")
    cases = getattr(suite, "cases", None)
    if (
        not isinstance(cases, (list, tuple))
        or len(cases) != 1
        or getattr(cases[0], "id", None)
        != QWEN38_27B_DSPARK_CUDA_GRAPH_CASE_ID
    ):
        raise ManifestError(
            f"{context}.cases must contain only the exact matched-prompt "
            "Qwen3.8 27B CUDA-graph case"
        )
    expected = {
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
        actual = getattr(case, field, None)
        if field == "requires" and isinstance(actual, (list, tuple)):
            actual = tuple(actual)
        if actual != wanted:
            raise ManifestError(f"{context}.cases[0].{field} changed")


def validate_matched_prompt_graph_suite(
    suite: SuiteSpec,
    *,
    context: str = "suite",
) -> None:
    """Dispatch an exact graph suite only to its pair-specific validator."""

    study = matched_prompt_graph_study(
        suite_id=getattr(suite, "id", None)
    )
    if study is None:
        raise ManifestError(f"{context}.id is not a matched-prompt graph suite")
    if study.suite_id == QWEN38_27B_DSPARK_CUDA_GRAPH_SUITE_ID:
        validate_qwen38_27b_dspark_cuda_graph_suite(suite, context=context)
        return
    if study.suite_id != SM121_CUDA_GRAPH_SUITE_ID:
        raise ManifestError(f"{context}.id is not a reviewed graph suite")
    try:
        validate_sm121_cuda_graph_suite(suite)
    except SM121CudaGraphError as error:
        raise ManifestError(f"{context}: {error}") from error


def validate_suite(suite: SuiteSpec, *, context: str = "suite") -> None:
    """Validate suite-level identity and case uniqueness."""

    _validate_id(suite.id, f"{context}.id")
    if suite.schema_version != SCHEMA_VERSION:
        raise ManifestError(
            f"{context}.schema_version must be {SCHEMA_VERSION}, "
            f"got {suite.schema_version}"
        )
    if (
        suite.protocol_digest is not None
        and _DIGEST_PATTERN.fullmatch(suite.protocol_digest) is None
    ):
        raise ManifestError(f"{context}.protocol_digest must be a sha256 digest")
    if not suite.cases:
        raise ManifestError(f"{context}.cases must not be empty")
    seen: set[str] = set()
    for index, case in enumerate(suite.cases):
        validate_case(case, context=f"{context}.cases[{index}]")
        if case.id in seen:
            raise ManifestError(f"{context}: duplicate case id {case.id!r}")
        seen.add(case.id)
    matched_prompt_studies = tuple(
        matched_prompt_graph_study(case_id=case.id)
        for case in suite.cases
        if matched_prompt_protocol(case.id) is not None
    )
    if matched_prompt_studies and (
        len(set(matched_prompt_studies)) != 1
        or matched_prompt_studies[0] is None
        or suite.id != matched_prompt_studies[0].suite_id
    ):
        expected_suite_ids = sorted(
            study.suite_id
            for study in matched_prompt_studies
            if study is not None
        )
        raise ManifestError(
            f"{context}.id must match its exact paired matched-prompt suite "
            f"{expected_suite_ids!r}"
        )
    if suite.id in MATCHED_PROMPT_GRAPH_SUITE_IDS:
        validate_matched_prompt_graph_suite(suite, context=context)
    if suite.id == DENSESPARK_SUITE_ID:
        try:
            validate_densespark_suite(suite)
        except DenseSparkContractError as error:
            raise ManifestError(f"{context}: {error}") from error
    if suite.id == _SM121_STORAGE_SUITE_ID:
        try:
            validate_sm121_storage_suite(suite)
        except SM121StorageCandidateError as error:
            raise ManifestError(f"{context}: {error}") from error
    if suite.id == _SM121_STORAGE_CACHE_OBSERVABILITY_SUITE_ID:
        try:
            validate_sm121_cache_observability_suite(suite)
        except SM121CacheObservabilityError as error:
            raise ManifestError(f"{context}: {error}") from error
    if suite.id == _SM121_CACHE_SEMANTIC_SUITE_ID:
        try:
            validate_sm121_cache_semantic_suite(suite)
        except SM121CacheSemanticError as error:
            raise ManifestError(f"{context}: {error}") from error
    if suite.id == SM121_CACHE_PERFORMANCE_SUITE_ID:
        try:
            validate_sm121_cache_performance_suite(suite)
        except SM121CachePerformanceError as error:
            raise ManifestError(f"{context}: {error}") from error
    try:
        chunked_prefill_suite_study = sm121_chunked_prefill_performance_study(
            suite.id
        )
    except SM121ChunkedPrefillPerformanceError:
        chunked_prefill_suite_study = None
    if (
        chunked_prefill_suite_study is not None
        and suite.id == chunked_prefill_suite_study.suite_id
    ):
        try:
            validate_sm121_chunked_prefill_performance_suite(suite)
        except SM121ChunkedPrefillPerformanceError as error:
            raise ManifestError(f"{context}: {error}") from error
    if suite.id == SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE_ID:
        try:
            validate_sm121_chunked_prefill_8k_admission_suite(suite)
        except SM121ChunkedPrefill8KAdmissionError as error:
            raise ManifestError(f"{context}: {error}") from error
    cache_cases = [case for case in suite.cases if case.kind == "cache"]
    cache_suite = suite.id == PREFIX_CACHE_SUITE_ID
    if cache_suite or cache_cases:
        if not cache_suite:
            raise ManifestError(
                f"{context}.id must be '{PREFIX_CACHE_SUITE_ID}' for cache cases"
            )
        expected = PREFIX_CACHE_PREFIX_TARGETS
        observed = {case.id: case.prompt_repetitions for case in cache_cases}
        if (
            observed != expected
            or len(cache_cases) != len(expected)
            or len(suite.cases) != len(expected)
        ):
            raise ManifestError(
                f"{context} cache cases must be the fixed 8192 and 32768 "
                "prefix protocol"
            )
    memory_cases = [case for case in suite.cases if case.kind == "memory"]
    memory_suite = suite.id == MEMORY_OPERATION_SUITE_ID
    if memory_suite or memory_cases:
        if not memory_suite:
            raise ManifestError(
                f"{context}.id must be '{MEMORY_OPERATION_SUITE_ID}' for memory cases"
            )
        observed = tuple(case.id for case in memory_cases)
        try:
            require_memory_operation_protocol_digest(suite.protocol_digest)
        except ValueError as error:
            raise ManifestError(
                f"{context} memory-operation protocol digest is stale or invalid"
            ) from error
        if (
            observed != MEMORY_OPERATION_SCENARIO_IDS
            or len(memory_cases) != len(KNOWN_MEMORY_OPERATION_CASE_IDS)
            or len(suite.cases) != len(KNOWN_MEMORY_OPERATION_CASE_IDS)
            or suite.description != MEMORY_OPERATION_SUITE_DESCRIPTION
        ):
            raise ManifestError(
                f"{context} memory cases must contain the fixed ordered "
                "memory-operation battery, description, and protocol digest"
            )


def validate_benchmark_selection(
    model: ModelSpec,
    suite: SuiteSpec,
    *,
    context: str = "selection",
    allow_sm121_agent_admission: bool = False,
) -> None:
    """Require protocol-specific profiles and suites to travel together."""

    cache_profile = getattr(model, "prefix_cache_mode", None) is not None
    cache_suite = suite.id == PREFIX_CACHE_SUITE_ID
    if cache_profile and not cache_suite:
        raise ManifestError(
            f"{context}: prefix_cache_mode profiles require the "
            f"{PREFIX_CACHE_SUITE_ID!r} suite"
        )
    if cache_suite and not cache_profile:
        raise ManifestError(
            f"{context}: the {PREFIX_CACHE_SUITE_ID!r} suite requires a "
            "prefix_cache_mode profile"
        )
    model_id = getattr(model, "id", None)
    if model_id in MATCHED_PROMPT_GRAPH_PROFILE_IDS:
        validate_matched_prompt_graph_model(model, context=f"{context}.model")
    densespark_profile = is_densespark_profile(model)
    densespark_suite = suite.id == DENSESPARK_SUITE_ID
    densespark_fast_suite = suite.id == DENSESPARK_FAST_CODING_SUITE_ID
    densespark_tool_suite = suite.id == DENSESPARK_TOOL_SUITE_ID
    densespark_known_suite = (
        densespark_suite or densespark_fast_suite or densespark_tool_suite
    )
    if densespark_profile and not densespark_known_suite:
        raise ManifestError(
            f"{context}: managed DenseSpark profiles require the "
            f"{DENSESPARK_SUITE_ID!r}, {DENSESPARK_FAST_CODING_SUITE_ID!r}, "
            f"or {DENSESPARK_TOOL_SUITE_ID!r} suite"
        )
    if (densespark_suite or densespark_fast_suite) and not densespark_profile:
        raise ManifestError(
            f"{context}: the {suite.id!r} suite requires the "
            f"managed DenseSpark profiles {sorted(DENSESPARK_PROFILE_IDS)!r}"
        )
    if densespark_profile and densespark_known_suite:
        try:
            validate_densespark_selection(model, suite)
        except DenseSparkContractError as error:
            raise ManifestError(f"{context}: {error}") from error
    tps_suite_profiles = _TPS_PROFILE_IDS_BY_SUITE.get(suite.id)
    tps_profile_suite = _TPS_SUITE_BY_PROFILE_ID.get(model_id)
    if tps_suite_profiles is not None and model_id not in tps_suite_profiles:
        raise ManifestError(
            f"{context}: the {suite.id!r} suite requires one of its exact "
            "dedicated cache-off profiles"
        )
    if tps_profile_suite is not None and suite.id != tps_profile_suite:
        raise ManifestError(
            f"{context}: the {model.id!r} profile requires the "
            f"{tps_profile_suite!r} suite"
        )
    flash_next_long_profile = model_id == _FLASH_NEXT_LONG_PROFILE_ID
    flash_next_long_suite = suite.id == _FLASH_NEXT_LONG_SUITE_ID
    if flash_next_long_profile and not flash_next_long_suite:
        raise ManifestError(
            f"{context}: the {_FLASH_NEXT_LONG_PROFILE_ID!r} profile requires "
            f"the {_FLASH_NEXT_LONG_SUITE_ID!r} suite"
        )
    if flash_next_long_suite and not flash_next_long_profile:
        raise ManifestError(
            f"{context}: the {_FLASH_NEXT_LONG_SUITE_ID!r} suite requires "
            f"the {_FLASH_NEXT_LONG_PROFILE_ID!r} profile"
        )
    flash_next_agent64k_profile = model_id in _FLASH_NEXT_AGENT64K_PROFILE_IDS
    flash_next_agent64k_suite = suite.id == _FLASH_NEXT_AGENT64K_SUITE_ID
    if flash_next_agent64k_profile and not flash_next_agent64k_suite:
        raise ManifestError(
            f"{context}: the {model.id!r} profile requires the "
            f"{_FLASH_NEXT_AGENT64K_SUITE_ID!r} suite"
        )
    if flash_next_agent64k_suite and not flash_next_agent64k_profile:
        raise ManifestError(
            f"{context}: the {_FLASH_NEXT_AGENT64K_SUITE_ID!r} suite requires "
            "one of its exact dedicated agent64k profiles"
        )
    qwen38_27b_graph_profile = (
        model_id in QWEN38_27B_DSPARK_CUDA_GRAPH_PROFILE_IDS
    )
    qwen38_27b_graph_suite = (
        suite.id == QWEN38_27B_DSPARK_CUDA_GRAPH_SUITE_ID
    )
    if qwen38_27b_graph_profile and not qwen38_27b_graph_suite:
        raise ManifestError(
            f"{context}: the {model.id!r} profile requires "
            f"{QWEN38_27B_DSPARK_CUDA_GRAPH_SUITE_ID!r}"
        )
    if qwen38_27b_graph_suite and not qwen38_27b_graph_profile:
        raise ManifestError(
            f"{context}: the {QWEN38_27B_DSPARK_CUDA_GRAPH_SUITE_ID!r} "
            "suite requires one of its exact paired CUDA-graph profiles"
        )
    if qwen38_27b_graph_suite:
        validate_qwen38_27b_dspark_cuda_graph_suite(
            suite, context=f"{context}.suite"
        )
    sm121_agent_admission_profile = (
        model_id == SM121_AGENT_ADMISSION_PROFILE_ID
    )
    sm121_agent_admission_suite = suite.id == SM121_AGENT_ADMISSION_SUITE_ID
    if sm121_agent_admission_profile:
        if not allow_sm121_agent_admission:
            raise ManifestError(
                f"{context}: the {SM121_AGENT_ADMISSION_PROFILE_ID!r} profile is "
                "tombstoned pending its dedicated parser/tool admission controller"
            )
        if not sm121_agent_admission_suite:
            raise ManifestError(
                f"{context}: the {SM121_AGENT_ADMISSION_PROFILE_ID!r} profile "
                f"requires {SM121_AGENT_ADMISSION_SUITE_ID!r}"
            )
        try:
            validate_sm121_agent_admission_suite(suite)
        except SM121AgentAdmissionError as error:
            raise ManifestError(f"{context}: {error}") from error
    if sm121_agent_admission_suite and not sm121_agent_admission_profile:
        raise ManifestError(
            f"{context}: the {SM121_AGENT_ADMISSION_SUITE_ID!r} suite requires "
            f"the {SM121_AGENT_ADMISSION_PROFILE_ID!r} profile"
        )
    sm121_cuda_graph_profile = model_id in SM121_CUDA_GRAPH_PROFILE_IDS
    sm121_cuda_graph_suite = suite.id == SM121_CUDA_GRAPH_SUITE_ID
    if sm121_cuda_graph_profile and not sm121_cuda_graph_suite:
        raise ManifestError(
            f"{context}: the {model.id!r} profile requires "
            f"{SM121_CUDA_GRAPH_SUITE_ID!r}"
        )
    if sm121_cuda_graph_suite and not sm121_cuda_graph_profile:
        raise ManifestError(
            f"{context}: the {SM121_CUDA_GRAPH_SUITE_ID!r} suite requires "
            "one of its exact paired CUDA-graph profiles"
        )
    if sm121_cuda_graph_suite:
        try:
            validate_sm121_cuda_graph_suite(suite)
        except SM121CudaGraphError as error:
            raise ManifestError(f"{context}: {error}") from error
    sm121_storage_profile = model_id == _SM121_STORAGE_PROFILE_ID
    sm121_storage_suite = suite.id in _SM121_STORAGE_SUITE_IDS
    if sm121_storage_profile and not sm121_storage_suite:
        raise ManifestError(
            f"{context}: the {_SM121_STORAGE_PROFILE_ID!r} profile requires "
            "one of its dedicated pre-admission suites"
        )
    if sm121_storage_suite and not sm121_storage_profile:
        raise ManifestError(
            f"{context}: the SM121 pre-admission suite requires "
            f"the {_SM121_STORAGE_PROFILE_ID!r} profile"
        )
    sm121_cache_semantic_profile = model_id in _SM121_CACHE_SEMANTIC_PROFILE_IDS
    sm121_cache_semantic_suite = suite.id == _SM121_CACHE_SEMANTIC_SUITE_ID
    if sm121_cache_semantic_profile and not sm121_cache_semantic_suite:
        raise ManifestError(
            f"{context}: the {model.id!r} profile requires "
            f"the {_SM121_CACHE_SEMANTIC_SUITE_ID!r} suite"
        )
    if sm121_cache_semantic_suite and not sm121_cache_semantic_profile:
        raise ManifestError(
            f"{context}: the {_SM121_CACHE_SEMANTIC_SUITE_ID!r} suite requires "
            "one of its exact paired cache-policy profiles"
        )
    sm121_cache_performance_profile = model_id in SM121_CACHE_PERFORMANCE_PROFILE_IDS
    sm121_cache_performance_suite = suite.id == SM121_CACHE_PERFORMANCE_SUITE_ID
    if sm121_cache_performance_profile and not sm121_cache_performance_suite:
        raise ManifestError(
            f"{context}: the {model.id!r} profile requires "
            f"the {SM121_CACHE_PERFORMANCE_SUITE_ID!r} suite"
        )
    if sm121_cache_performance_suite and not sm121_cache_performance_profile:
        raise ManifestError(
            f"{context}: the {SM121_CACHE_PERFORMANCE_SUITE_ID!r} suite requires "
            "one of its exact paired cache-policy performance profiles"
        )
    try:
        chunked_prefill_profile_study = sm121_chunked_prefill_performance_study(
            model_id
        )
    except SM121ChunkedPrefillPerformanceError:
        chunked_prefill_profile_study = None
    try:
        chunked_prefill_suite_study = sm121_chunked_prefill_performance_study(
            suite.id
        )
    except SM121ChunkedPrefillPerformanceError:
        chunked_prefill_suite_study = None
    sm121_chunked_prefill_performance_profile = (
        model_id in SM121_CHUNKED_PREFILL_PERFORMANCE_PROFILE_IDS
    )
    sm121_chunked_prefill_performance_suite = (
        chunked_prefill_suite_study is not None
        and suite.id == chunked_prefill_suite_study.suite_id
    )
    sm121_chunked_prefill_8k_admission_profile = (
        is_sm121_chunked_prefill_8k_admission_profile(model)
    )
    sm121_chunked_prefill_8k_admission_suite = (
        suite.id == SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE_ID
    )
    if (
        sm121_chunked_prefill_performance_profile
        and not (
            sm121_chunked_prefill_8k_admission_profile
            and sm121_chunked_prefill_8k_admission_suite
        )
        and (
            not sm121_chunked_prefill_performance_suite
            or chunked_prefill_profile_study != chunked_prefill_suite_study
        )
    ):
        assert chunked_prefill_profile_study is not None
        raise ManifestError(
            f"{context}: the {model.id!r} profile requires "
            f"the {chunked_prefill_profile_study.suite_id!r} suite"
        )
    if (
        sm121_chunked_prefill_performance_suite
        and (
            not sm121_chunked_prefill_performance_profile
            or chunked_prefill_profile_study != chunked_prefill_suite_study
        )
    ):
        assert chunked_prefill_suite_study is not None
        raise ManifestError(
            f"{context}: the {chunked_prefill_suite_study.suite_id!r} "
            "suite requires one of its exact chunked-prefill profiles"
        )
    if (
        sm121_chunked_prefill_8k_admission_suite
        and not sm121_chunked_prefill_8k_admission_profile
    ):
        raise ManifestError(
            f"{context}: the {SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE_ID!r} "
            "suite requires the exact prospective 8K profile"
        )
    ple_study_suite_profiles = _PLE_STUDY_PROFILE_IDS_BY_SUITE.get(suite.id)
    ple_study_profile_suites = _PLE_STUDY_SUITE_IDS_BY_PROFILE_ID.get(model_id)
    if (
        ple_study_suite_profiles is not None
        and model_id not in ple_study_suite_profiles
    ):
        raise ManifestError(
            f"{context}: the {suite.id!r} suite requires one of its exact "
            "PLE-study profiles"
        )
    if (
        ple_study_profile_suites is not None
        and suite.id not in ple_study_profile_suites
    ):
        allowed_suites = ", ".join(
            repr(suite_id) for suite_id in sorted(ple_study_profile_suites)
        )
        raise ManifestError(
            f"{context}: the {model.id!r} profile requires one of the "
            f"following suites: {allowed_suites}"
        )
    if suite.id == MEMORY_OPERATION_SUITE_ID:
        try:
            require_memory_operation_protocol_digest(suite.protocol_digest)
        except ValueError as error:
            raise ManifestError(
                f"{context}: memory-operation protocol digest is stale or invalid"
            ) from error
        if (
            model.backend != "llamacpp"
            or model.lifecycle != "subprocess"
            or model.runtime_revision != MEMORY_OPERATION_LLAMACPP_REVISION
            or model.runtime_digest != MEMORY_OPERATION_LLAMACPP_DIGEST
            or model.runtime_parallel != 1
            or model.max_context != MEMORY_OPERATION_CONTEXT_TOKENS
            or model.native_context != MEMORY_OPERATION_CONTEXT_TOKENS
        ):
            raise ManifestError(
                f"{context}: the {MEMORY_OPERATION_SUITE_ID!r} suite requires "
                "the fixed single-slot 32K llama.cpp geometry"
            )
        try:
            request_body = json.loads(model.request_body_json or "")
        except (json.JSONDecodeError, TypeError) as error:
            raise ManifestError(
                f"{context}: memory-operation profiles require an explicit "
                "thinking policy"
            ) from error
        if (
            not isinstance(request_body, dict)
            or set(request_body) != {"chat_template_kwargs"}
            or not isinstance(request_body["chat_template_kwargs"], dict)
            or set(request_body["chat_template_kwargs"]) != {"enable_thinking"}
            or not isinstance(
                request_body["chat_template_kwargs"]["enable_thinking"], bool
            )
        ):
            raise ManifestError(
                f"{context}: memory-operation profiles require exactly one "
                "boolean enable_thinking policy"
            )
        enable_thinking = request_body["chat_template_kwargs"]["enable_thinking"]
        if (
            not {"chat", "json"}.issubset(model.tasks)
            or ("thinking" in model.tasks) is not enable_thinking
        ):
            raise ManifestError(
                f"{context}: memory-operation capabilities do not match the "
                "thinking policy"
            )
        if model.args != memory_operation_llamacpp_args(
            enable_thinking=enable_thinking
        ):
            raise ManifestError(
                f"{context}: memory-operation llama.cpp arguments do not "
                "match the fixed protocol"
            )


def validate_models(models: Mapping[str, ModelSpec] | Iterable[ModelSpec]) -> None:
    """Validate a registry or iterable and reject duplicate or mismatched IDs."""

    items = models.items() if isinstance(models, Mapping) else None
    values = models.values() if isinstance(models, Mapping) else models
    seen: set[str] = set()
    for index, model in enumerate(values):
        validate_model(model, context=f"models[{index}]")
        if model.id in seen:
            raise ManifestError(f"duplicate model id {model.id!r}")
        seen.add(model.id)
    if items is not None:
        for key, model in items:
            if key != model.id:
                raise ManifestError(
                    f"model registry key {key!r} does not match id {model.id!r}"
                )


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except OSError as error:
        raise ManifestError(f"cannot read {path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ManifestError(f"invalid TOML in {path}: {error}") from error


def _check_schema_version(document: Mapping[str, Any], path: Path) -> int:
    version = document.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ManifestError(f"{path}: schema_version must be an integer")
    if version != SCHEMA_VERSION:
        raise ManifestError(
            f"{path}: unsupported schema_version {version}; expected {SCHEMA_VERSION}"
        )
    return version


def _reject_unknown(
    table: Mapping[str, Any], allowed: set[str] | frozenset[str], context: str
) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise ManifestError(f"{context}: unknown field(s): {', '.join(unknown)}")


def _required_string(table: Mapping[str, Any], key: str, context: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _optional_string(
    table: Mapping[str, Any], key: str, context: str
) -> str | None:
    if key not in table:
        return None
    return _required_string(table, key, context)


def _required_int(table: Mapping[str, Any], key: str, context: str) -> int:
    if key not in table:
        raise ManifestError(f"{context}.{key} is required")
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{context}.{key} must be an integer")
    return value


def _optional_int(
    table: Mapping[str, Any], key: str, context: str, *, default: int | None
) -> int | None:
    if key not in table:
        return default
    return _required_int(table, key, context)


def _optional_bool(
    table: Mapping[str, Any], key: str, context: str, *, default: bool
) -> bool:
    if key not in table:
        return default
    value = table[key]
    if not isinstance(value, bool):
        raise ManifestError(f"{context}.{key} must be a boolean")
    return value


def _optional_number(
    table: Mapping[str, Any],
    key: str,
    context: str,
    *,
    default: float | None = None,
) -> float | None:
    if key not in table:
        return default
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{context}.{key} must be a number")
    return float(value)


def _string_tuple(
    table: Mapping[str, Any],
    key: str,
    context: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    if key not in table:
        if required:
            raise ManifestError(f"{context}.{key} is required")
        return ()
    value = table[key]
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ManifestError(f"{context}.{key} must be an array of strings")
    if required and not value:
        raise ManifestError(f"{context}.{key} must not be empty")
    return tuple(item.strip() for item in value)


def _model_shards(
    table: Mapping[str, Any], key: str, context: str
) -> tuple[ModelShard, ...]:
    if key not in table:
        return ()
    value = table[key]
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{context}.{key} must be a non-empty array of tables")
    shards: list[ModelShard] = []
    for index, row in enumerate(value):
        shard_context = f"{context}.{key}[{index}]"
        if not isinstance(row, dict):
            raise ManifestError(f"{shard_context} must be a table")
        _reject_unknown(row, _MODEL_SHARD_KEYS, shard_context)
        shards.append(
            ModelShard(
                path=_required_string(row, "path", shard_context),
                digest=_required_string(row, "digest", shard_context),
                size_bytes=_required_int(row, "size_bytes", shard_context),
            )
        )
    return tuple(shards)


def _sglang_source_overlays(
    table: Mapping[str, Any], key: str, context: str
) -> tuple[SGLangSourceOverlay, ...]:
    if key not in table:
        return ()
    value = table[key]
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{context}.{key} must be a non-empty array of tables")
    overlays: list[SGLangSourceOverlay] = []
    for index, row in enumerate(value):
        overlay_context = f"{context}.{key}[{index}]"
        if not isinstance(row, dict):
            raise ManifestError(f"{overlay_context} must be a table")
        _reject_unknown(row, _SGLANG_SOURCE_OVERLAY_KEYS, overlay_context)
        overlays.append(
            SGLangSourceOverlay(
                host_path=_required_string(row, "host_path", overlay_context),
                container_path=_required_string(
                    row, "container_path", overlay_context
                ),
                digest=_required_string(row, "digest", overlay_context),
            )
        )
    return tuple(overlays)


def _validate_id(value: str, context: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ManifestError(
            f"{context} must match {_ID_PATTERN.pattern!r}; got {value!r}"
        )


def _validate_unique_values(
    values: tuple[str, ...], context: str, allowed: frozenset[str]
) -> None:
    if not values:
        raise ManifestError(f"{context} must not be empty")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ManifestError(f"{context} contains unknown values: {', '.join(unknown)}")
    if len(values) != len(set(values)):
        raise ManifestError(f"{context} must not contain duplicates")


def _validate_endpoint(endpoint: str, context: str) -> None:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ManifestError(f"{context} must be an HTTP(S) URL")
    if parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ManifestError(f"{context} must use a loopback host")


def _validate_fetch_patterns(values: tuple[str, ...], context: str) -> None:
    if len(values) != len(set(values)):
        raise ManifestError(f"{context} must not contain duplicates")
    for pattern in values:
        unsafe_component = any(
            component in {"", ".", ".."} for component in pattern.split("/")
        )
        unsafe_character = any(ord(character) < 32 for character in pattern)
        if (
            pattern.startswith(("-", "/", "\\", "~"))
            or "\\" in pattern
            or ":" in pattern
            or unsafe_component
            or unsafe_character
        ):
            raise ManifestError(f"{context} contains unsafe pattern {pattern!r}")


def _validate_model_shards(
    shards: tuple[ModelShard, ...], context: str
) -> None:
    if len(shards) < 2:
        raise ManifestError(f"{context} must contain at least two split GGUF files")
    seen_paths: set[str] = set()
    seen_basenames: set[str] = set()
    expected_parent: PurePosixPath | None = None
    expected_prefix: str | None = None
    expected_total: int | None = None
    expected_width: int | None = None
    for index, shard in enumerate(shards, start=1):
        shard_context = f"{context}[{index - 1}]"
        if not isinstance(shard, ModelShard):
            raise ManifestError(f"{shard_context} must be a ModelShard")
        path = shard.path
        if not isinstance(path, str) or not path or path != path.strip():
            raise ManifestError(f"{shard_context}.path must be a non-empty string")
        if (
            path.startswith(("-", "/", "\\", "~"))
            or "\\" in path
            or ":" in path
            or any(character in path for character in "*?[")
            or any(ord(character) < 32 for character in path)
        ):
            raise ManifestError(
                f"{shard_context}.path must be a safe relative snapshot path"
            )
        relative = PurePosixPath(path)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ManifestError(
                f"{shard_context}.path must be a safe relative snapshot path"
            )
        if path in seen_paths:
            raise ManifestError(f"{context} must not contain duplicate paths")
        if relative.name in seen_basenames:
            raise ManifestError(f"{context} must not contain duplicate basenames")
        seen_paths.add(path)
        seen_basenames.add(relative.name)
        if not _DIGEST_PATTERN.fullmatch(shard.digest):
            raise ManifestError(f"{shard_context}.digest must be a sha256 digest")
        if (
            isinstance(shard.size_bytes, bool)
            or not isinstance(shard.size_bytes, int)
            or shard.size_bytes <= 0
        ):
            raise ManifestError(f"{shard_context}.size_bytes must be positive")
        match = _LLAMACPP_SPLIT_GGUF_PATTERN.fullmatch(relative.name)
        if match is None:
            raise ManifestError(
                f"{shard_context}.path must use canonical "
                "<name>-<index>-of-<total>.gguf split naming"
            )
        ordinal_text = match.group("ordinal")
        total_text = match.group("total")
        ordinal = int(ordinal_text)
        total = int(total_text)
        if total < 2 or ordinal != index:
            raise ManifestError(
                f"{context} must be ordered without missing shard indexes"
            )
        if len(ordinal_text) != len(total_text):
            raise ManifestError(
                f"{shard_context}.path split indexes must use equal widths"
            )
        if expected_parent is None:
            expected_parent = relative.parent
            expected_prefix = match.group("prefix")
            expected_total = total
            expected_width = len(total_text)
        elif (
            relative.parent != expected_parent
            or match.group("prefix") != expected_prefix
            or total != expected_total
            or len(total_text) != expected_width
        ):
            raise ManifestError(
                f"{context} must describe one canonical split GGUF set"
            )
    if expected_total != len(shards):
        raise ManifestError(
            f"{context} must include every shard declared by the split total"
        )


def _llamacpp_spec_types(arguments: Iterable[Any]) -> tuple[str, ...]:
    values = tuple(str(argument) for argument in arguments)
    configured: list[str] = []
    for index, argument in enumerate(values):
        if argument.startswith("--spec-type="):
            configured.append(argument.split("=", 1)[1])
        elif argument == "--spec-type" and index + 1 < len(values):
            configured.append(values[index + 1])
    if len(configured) != 1:
        return ()
    return tuple(
        item.strip() for item in configured[0].split(",") if item.strip()
    )


def _llamacpp_positive_option(
    arguments: Iterable[Any], option: str
) -> int | None:
    values = tuple(str(argument) for argument in arguments)
    configured: list[str] = []
    for index, argument in enumerate(values):
        if argument.startswith(option + "="):
            configured.append(argument.split("=", 1)[1])
        elif argument == option and index + 1 < len(values):
            configured.append(values[index + 1])
    if len(configured) != 1:
        return None
    try:
        parsed = int(configured[0])
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _default_endpoint(backend: str) -> str:
    if backend == "ollama":
        return "http://127.0.0.1:11434/v1"
    if backend == "transformers":
        return "offline://transformers"
    if backend == "trtllm":
        return "offline://trtllm"
    return "http://127.0.0.1:8000/v1"
