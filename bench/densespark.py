"""Fail-closed local provenance helpers for the DenseSpark adapter.

This module deliberately does not start containers or inspect GPUs.  It binds the
small set of inputs that a serving adapter needs to trust before it can do either
of those things: an exact upstream identity, an immutable PQ artifact, an
allowlisted configuration, and a locally resolved Docker image ID.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import subprocess
from typing import Protocol


DENSESPARK_RECIPE_SOURCE = "albond/DenseSpark-Qwen3.8-27B"
DENSESPARK_RECIPE_REVISION = "0abecc3005cebe6f5e1e0c0e1f16552f95fe0228"
DENSESPARK_RECIPE_TREE = "347468a41f8431b0c5a94a56e316566a06489a43"
DENSESPARK_MODEL_SOURCE = "Frozenlock/Qwen3.8-27B-int4-AutoRound"
DENSESPARK_MODEL_REVISION = "b4c61732c4f2d8af323d75ba5702b5c7f3361539"
DENSESPARK_PROFILE_ID = "qwen38-27b-int4-autoround-densespark-c1"
DENSESPARK_WARMUP_SYNC_PROFILE_ID = (
    "qwen38-27b-int4-autoround-densespark-c1-experimental-warmup-sync"
)
DENSESPARK_PROFILE_IDS = frozenset(
    {DENSESPARK_PROFILE_ID, DENSESPARK_WARMUP_SYNC_PROFILE_ID}
)
DENSESPARK_SUITE_ID = "qwen38-27b-densespark-c1"
DENSESPARK_TOOL_SUITE_ID = "agentic-tools"
DENSESPARK_IMAGE = "local/densespark:qwen38-27b-v1.2-0abecc3"
DENSESPARK_LOCAL_IMAGE_ID = (
    "sha256:d8d02859a49ebf452d9e20b5fbc0790c"
    "d4c38fe9a1f5184096b06e3cc6a751d1"
)
DENSESPARK_WARMUP_SYNC_IMAGE = (
    "local/densespark:qwen38-27b-v1.2-warmup-probe-hardened-572e66d5"
)
DENSESPARK_WARMUP_SYNC_LOCAL_IMAGE_ID = (
    "sha256:c7adf2163f7dd04b52eb5ec91f373bf8"
    "fcd1cc63a51f61c2d457ad2976564153"
)
DENSESPARK_WARMUP_SYNC_MODE = "sync-only"
DENSESPARK_WARMUP_SYNC_PROBE_SHA256 = (
    "sha256:95089265e60f67da8d8f33d6fb249e4c"
    "79300f0891c38f2f15d4e125001821d3"
)
DENSESPARK_WARMUP_SYNC_IMAGE_RECIPE_SHA256 = (
    "sha256:572e66d585ed74a5f0b278e2feb2cf7d"
    "ba260ca84c53ba93be66d7c2e69c571a"
)
DENSESPARK_WARMUP_SYNC_DOCKERIGNORE_SHA256 = (
    "sha256:100ee126af6ef26dd45e85b9e90f5cc0"
    "adb8d6b0c51d391c37117fc7168627ea"
)
DENSESPARK_WARMUP_SYNC_QWEN_WARMUP_SOURCE_SHA256 = (
    "sha256:2b08d94662e7b04ce61c0f7a818e0cd1"
    "768fe7602a89df04ec6148f62fe3acdb"
)
DENSESPARK_WARMUP_SYNC_KERNEL_WARMUP_SOURCE_SHA256 = (
    "sha256:452ae5db905110df8eb7aac90a93ac808"
    "63d166f8ea7d52b8cec02c477477aed"
)
DENSESPARK_WARMUP_SYNC_QWEN_GDN_SOURCE_SHA256 = (
    "sha256:d42cdc95d8d221b49693a46119c714fee"
    "3f290282bdfefa63f92f9725f1b20ea"
)
DENSESPARK_WARMUP_SYNC_MAMBA_UTILS_SOURCE_SHA256 = (
    "sha256:53eaae681b5a0327465b28b7b1983303"
    "335db852ac9667ae05faa3682d8c6b8c"
)
DENSESPARK_WARMUP_SYNC_FUSED_SIGMOID_SOURCE_SHA256 = (
    "sha256:000ab8996af9788fdb8843a6a3b91833"
    "e7a14c8acc0e1ea073a536330f64cb6f"
)
DENSESPARK_WARMUP_SYNC_VLLM_ENTRYPOINT_SHA256 = (
    "sha256:6f6395c128e80861f7f7d21b8e1e4547"
    "261ab9e928390aa7a7a89ce0d701ff36"
)
DENSESPARK_WARMUP_SYNC_PROBE_RELATIVE_PATH = PurePosixPath(
    "bench/assets/densespark_qwen_warmup_probe.py"
)
DENSESPARK_WARMUP_SYNC_IMAGE_RECIPE_RELATIVE_PATH = PurePosixPath(
    "patches/vllm/Dockerfile.densespark-qwen-warmup-probe"
)
DENSESPARK_WARMUP_SYNC_DOCKERIGNORE_RELATIVE_PATH = PurePosixPath(
    "patches/vllm/Dockerfile.densespark-qwen-warmup-probe.dockerignore"
)
DENSESPARK_SERVED_NAME = "densespark-qwen3.8-27b"
DENSESPARK_TOOL_CALL_PARSER = "qwen3_xml"
DENSESPARK_MAX_CONTEXT = 65_536
DENSESPARK_NATIVE_CONTEXT = 262_144
DENSESPARK_WEIGHT_FILE_COUNT = 8
DENSESPARK_WEIGHT_SIZE_BYTES = 18_996_706_072
DENSESPARK_WEIGHT_FILES = (
    *(f"model-{index:05d}-of-00007.safetensors" for index in range(1, 8)),
    "model_extra_tensors.safetensors",
)
DENSESPARK_WEIGHT_BLOB_SHA256S = {
    "model-00001-of-00007.safetensors": (
        "83c0adb0f1141142a25c5f7937a09272fef2baee7d6b07afe60a076feffe3b66"
    ),
    "model-00002-of-00007.safetensors": (
        "6c86c3917cd228d5699cb1435d9216e2baf281582ba4282a98638163605be7ec"
    ),
    "model-00003-of-00007.safetensors": (
        "bfb9613b3f6ba3d0c5b3efa74f92cc14ec5c9266b341a79bf6b45247a5857770"
    ),
    "model-00004-of-00007.safetensors": (
        "8e9fbd476aa3055c3c0a886298f9beb93c5275c748c533835746f6648f6dae55"
    ),
    "model-00005-of-00007.safetensors": (
        "f0cd639621eb26410d96b8c073344ae008f646572a9c4713326ff43d406cf276"
    ),
    "model-00006-of-00007.safetensors": (
        "55a14ee79d3e5a65a8731d89426f4df477e8bdc7daa7976d41254d8afb9432f0"
    ),
    "model-00007-of-00007.safetensors": (
        "6866cf8adcccc4cc6a00e74bc025f1a774fb52103b70f2674d0288272951a733"
    ),
    "model_extra_tensors.safetensors": (
        "94102b67c6b84e65dbb9bae37c00bd88ac1a43ff577ce65fd8842d231c7e89de"
    ),
}
# Complete top-level inventory for the pinned Hugging Face snapshot.  Each
# value is ``(blob name, size in bytes, content SHA-256)``.  The small Git
# blobs use SHA-1 names in the Hugging Face cache, so their content SHA-256 is
# recorded separately.  Repository metadata is pinned too: rejecting every
# unknown top-level entry keeps the mounted snapshot identical to the one that
# was inspected, even when a file is not currently consumed by vLLM.
DENSESPARK_SNAPSHOT_FILE_PINS: Mapping[str, tuple[str, int, str]] = {
    ".gitattributes": (
        "52373fe24473b1aa44333d318f578ae6bf04b49b",
        1_570,
        "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    ),
    "README.md": (
        "08b94c415e7d7bf0e46f1c4b0c7c2fb07e361923",
        3_541,
        "4e1683d008085df47351b2814a70c5389888ab6732b36e105159c06a50e047f8",
    ),
    "chat_template.jinja": (
        "c0c686f9c38d70d179fb7b5f5aa7530bc913dda3",
        8_952,
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    ),
    "config.json": (
        "f2efc10cb1160a95bf9f9d45b12ccb5823dea753",
        15_741,
        "15173d7a487c88112a02804701ab2cc8f8dd4631a3ef67c8d9bb2c66d30debce",
    ),
    "generation_config.json": (
        "8b9f95da7b8d22a1ef9de7d28c2dc52eeb04d41e",
        214,
        "b8eb74d15e0a56623d00ccd14950a4bb87fabbf84b5cc030dcc904b899fb1eb5",
    ),
    "model-00001-of-00007.safetensors": (
        DENSESPARK_WEIGHT_BLOB_SHA256S["model-00001-of-00007.safetensors"],
        3_216_489_680,
        DENSESPARK_WEIGHT_BLOB_SHA256S["model-00001-of-00007.safetensors"],
    ),
    "model-00002-of-00007.safetensors": (
        DENSESPARK_WEIGHT_BLOB_SHA256S["model-00002-of-00007.safetensors"],
        3_190_151_576,
        DENSESPARK_WEIGHT_BLOB_SHA256S["model-00002-of-00007.safetensors"],
    ),
    "model-00003-of-00007.safetensors": (
        DENSESPARK_WEIGHT_BLOB_SHA256S["model-00003-of-00007.safetensors"],
        3_219_130_488,
        DENSESPARK_WEIGHT_BLOB_SHA256S["model-00003-of-00007.safetensors"],
    ),
    "model-00004-of-00007.safetensors": (
        DENSESPARK_WEIGHT_BLOB_SHA256S["model-00004-of-00007.safetensors"],
        3_216_860_120,
        DENSESPARK_WEIGHT_BLOB_SHA256S["model-00004-of-00007.safetensors"],
    ),
    "model-00005-of-00007.safetensors": (
        DENSESPARK_WEIGHT_BLOB_SHA256S["model-00005-of-00007.safetensors"],
        770_164_464,
        DENSESPARK_WEIGHT_BLOB_SHA256S["model-00005-of-00007.safetensors"],
    ),
    "model-00006-of-00007.safetensors": (
        DENSESPARK_WEIGHT_BLOB_SHA256S["model-00006-of-00007.safetensors"],
        2_542_807_272,
        DENSESPARK_WEIGHT_BLOB_SHA256S["model-00006-of-00007.safetensors"],
    ),
    "model-00007-of-00007.safetensors": (
        DENSESPARK_WEIGHT_BLOB_SHA256S["model-00007-of-00007.safetensors"],
        2_542_796_896,
        DENSESPARK_WEIGHT_BLOB_SHA256S["model-00007-of-00007.safetensors"],
    ),
    "model.safetensors.index.json": (
        "41cb335983bf0f9ce2ce528ecbab010a0089f491",
        192_181,
        "adf387dee183d109e95cdc4d4988fcd966de326861ff4d66876692170f1e03ad",
    ),
    "model_extra_tensors.safetensors": (
        DENSESPARK_WEIGHT_BLOB_SHA256S["model_extra_tensors.safetensors"],
        298_305_576,
        DENSESPARK_WEIGHT_BLOB_SHA256S["model_extra_tensors.safetensors"],
    ),
    "preprocessor_config.json": (
        "8ed39680d90d989c35a3e308338a24875bafbc42",
        443,
        "3a159dfec9978a186a72ba085e0ad6a050f3968d8b364218d7bd13f5c89381f2",
    ),
    "processor_config.json": (
        "33818c7f9e991ad735fd240209f4fa73e6c28c50",
        1_191,
        "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
    ),
    "quantization_config.json": (
        "a6d675703033186d4a7bde8b24b10e1e58228605",
        11_142,
        "7ca7c7290d50e6155120067e2fe7a31ef2913158b1b07441f95b451bcec6db94",
    ),
    "tokenizer.json": (
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        19_989_325,
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    ),
    "tokenizer_config.json": (
        "1d134cd298be1e3be25db393d93a1cefe80e3214",
        1_165,
        "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b",
    ),
}
DENSESPARK_PQ_CACHE_RELATIVE_PATH = PurePosixPath(
    "densespark-repro-b4c61732/pq_head_m128.pt"
)
DENSESPARK_PQ_CONTAINER_PATH = "/opt/densespark/pq_head_m128.pt"
DENSESPARK_PQ_SIZE_BYTES = 34_906_281
DENSESPARK_PQ_SHA256 = (
    "sha256:4e794c398d700002479b914e2c5d530e"
    "ad57ca5861862ab4230bc470cf95cea9"
)
DENSESPARK_CACHE_ROOT_NAME = "densespark-vllm-repro-b4c61732"
DENSESPARK_STARTUP_MIN_MEMAVAILABLE_GIB = 14
DENSESPARK_STARTUP_MAX_SWAP_GROWTH_MIB = 512
DENSESPARK_STARTUP_MAX_STARTING_SWAP_MIB = 512
DENSESPARK_LAUNCH_POLICY_SCHEMA = "sparkbench.densespark.launch-policy.v1"
DENSESPARK_LAUNCH_HOST = "127.0.0.1"
DENSESPARK_LAUNCH_HOST_PORT = 8000
DENSESPARK_LAUNCH_CONTAINER_PORT = 8000
DENSESPARK_LAUNCH_DOCKER_NETWORK = "bridge"
DENSESPARK_CONTAINER_SNAPSHOT = (
    "/root/.cache/huggingface/hub/"
    "models--Frozenlock--Qwen3.8-27B-int4-AutoRound/snapshots/"
    + DENSESPARK_MODEL_REVISION
)

# These are the complete, reviewed v1.2 C1 launch controls.  They are code,
# rather than manifest fields, so a profile cannot inject arbitrary environment
# variables or mounts into the managed Docker boundary.
DENSESPARK_C1_ENVIRONMENT = (
    ("DENSESPARK_MARLIN_NSPLIT", "0"),
    ("DENSESPARK_MARLIN_NSPLIT_MIN_M", "256"),
    ("DENSESPARK_LAB89_HYBRID_LINEAR", "1"),
    ("DENSESPARK_LAB89_HUMMING_MIN_M", "256"),
    ("DENSESPARK_LAB113_ENABLE", "1"),
    ("DENSESPARK_LAB118_ENABLE", "1"),
    ("DENSESPARK_LAB133_M8000", "1"),
    ("DENSESPARK_INT8_LMHEAD", "1"),
    ("DENSESPARK_HEAD_BATCH_DOT", "1"),
    ("DENSESPARK_HEAD_AUTOTUNE", "0"),
    ("DENSESPARK_HEAD_CHUNK16", "1"),
    ("DENSESPARK_HEAD_INTERLEAVED", "1"),
    ("DENSESPARK_PQ_DRAFT", "1"),
    ("DENSESPARK_PQ_CANDIDATES", "2048"),
    ("DENSESPARK_PQ_BATCH_SCAN", "1"),
    ("DENSESPARK_DRAFT_MATCH_FILTERS", "0"),
    ("DENSESPARK_LAB86_SPARSE_PQ", "0"),
    ("DENSESPARK_LAB90_EXACT_SAMPLER", "0"),
    ("HF_HUB_OFFLINE", "1"),
    ("VLLM_NO_USAGE_STATS", "1"),
    ("VLLM_HUMMING_INPUT_QUANT_CONFIG", '{"dtype":"int8"}'),
)

# Upstream C1 used 0.90. The Spark reproduction reduced this to 0.86 to retain
# more MemAvailable on unified memory. This is a focused configuration change,
# not a substitute for the separate memory and swap watchdog.
DENSESPARK_C1_ARGS = (
    "--max-model-len",
    "65536",
    "--tool-call-parser",
    DENSESPARK_TOOL_CALL_PARSER,
    "--enable-auto-tool-choice",
    "--gpu-memory-utilization",
    "0.86",
    "--limit-mm-per-prompt",
    '{"image":0,"video":0}',
    "--no-enable-prefix-caching",
    "--mamba-cache-mode",
    "none",
    "--max-num-batched-tokens",
    "8192",
    "--mamba-ssm-cache-dtype",
    "bfloat16",
    "--linear-backend",
    "humming",
    "--gdn-prefill-backend",
    "flashinfer",
    "--reasoning-parser",
    "qwen3",
    "--speculative-config",
    '{"method":"mtp","num_speculative_tokens":8,'
    '"draft_sample_method":"probabilistic"}',
)

DENSESPARK_C1_CASES = (
    {
        "id": "densespark-c1-decode-256",
        "kind": "decode",
        "requires": ("chat",),
        "warmups": 1,
        "repetitions": 3,
        "max_output_tokens": 256,
        "temperature": 0.0,
        "concurrency": 1,
        "prompt_repetitions": 0,
        "max_turns": 1,
    },
)
DENSESPARK_TOOL_SUITE_DESCRIPTION = (
    "Deterministic multi-turn tool selection, abstention, dependency, and "
    "recovery checks with scalar-only results."
)
DENSESPARK_TOOL_CASES = tuple(
    {
        "id": case_id,
        "kind": "agentic",
        "requires": ("chat", "tools"),
        "warmups": 0,
        "repetitions": 3,
        "max_output_tokens": 4_096,
        "temperature": 0.0,
        "concurrency": 1,
        "prompt_repetitions": 0,
        "max_turns": 6,
    }
    for case_id in (
        "agentic-select-and-call",
        "agentic-no-tool",
        "agentic-two-hop",
        "agentic-tool-error-recovery",
    )
)

DENSESPARK_CONFIG_SCHEMA = "sparkbench.densespark.config.v1"
DENSESPARK_CONFIG_KEYS = frozenset(
    {
        "DENSESPARK_CONCURRENCY",
        "DENSESPARK_AUTO_TOOL_CHOICE",
        "DENSESPARK_DRAFT_MATCH_FILTERS",
        "DENSESPARK_DRAFT_SAMPLE_METHOD",
        "DENSESPARK_HEAD_AUTOTUNE",
        "DENSESPARK_HEAD_BATCH_DOT",
        "DENSESPARK_HEAD_CHUNK16",
        "DENSESPARK_HEAD_INTERLEAVED",
        "DENSESPARK_GDN_PREFILL_BACKEND",
        "DENSESPARK_GPU_MEMORY_UTILIZATION",
        "DENSESPARK_IMAGE_ID",
        "DENSESPARK_INT8_LMHEAD",
        "DENSESPARK_LAB113_ENABLE",
        "DENSESPARK_LAB118_ENABLE",
        "DENSESPARK_LAB133_M8000",
        "DENSESPARK_LAB86_SPARSE_PQ",
        "DENSESPARK_LAB89_HUMMING_MIN_M",
        "DENSESPARK_LAB89_HYBRID_LINEAR",
        "DENSESPARK_LAB90_EXACT_SAMPLER",
        "DENSESPARK_MARLIN_NSPLIT",
        "DENSESPARK_MARLIN_NSPLIT_MIN_M",
        "DENSESPARK_LINEAR_BACKEND",
        "DENSESPARK_LIMIT_MM",
        "DENSESPARK_MAMBA_CACHE_MODE",
        "DENSESPARK_MAMBA_SSM_CACHE_DTYPE",
        "DENSESPARK_MAX_MODEL_LEN",
        "DENSESPARK_MAX_NUM_BATCHED_TOKENS",
        "DENSESPARK_MODEL_REVISION",
        "DENSESPARK_PQ_ARTIFACT_SHA256",
        "DENSESPARK_PQ_BATCH_SCAN",
        "DENSESPARK_PQ_CANDIDATES",
        "DENSESPARK_PQ_DRAFT",
        "DENSESPARK_SPEC_TOKENS",
        "DENSESPARK_TOOL_CALL_PARSER",
        "DENSESPARK_WARMUP_MODE",
        "DENSESPARK_PREFIX_CACHING",
        "DENSESPARK_REASONING_PARSER",
        "VLLM_HUMMING_INPUT_QUANT_CONFIG",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IMAGE_REFERENCE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,254}")
_CONFIG_INTEGER_MAX = (1 << 63) - 1
_READ_SIZE = 1024 * 1024


class DenseSparkContractError(RuntimeError):
    """Raised when a DenseSpark provenance or isolation contract fails."""


class CommandResult(Protocol):
    """Minimal result shape accepted from an injected command runner."""

    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str]], CommandResult]


@dataclass(frozen=True, slots=True)
class DenseSparkPQArtifact:
    """Verified scalar receipt for a local PQ artifact."""

    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DenseSparkSnapshotReceipt:
    """Scalar receipt for the exact cached DenseSpark weight set."""

    weight_file_count: int
    weight_size_bytes: int


def is_densespark_profile_identity(
    *,
    recipe_source: object,
    recipe_revision: object,
    model_source: object,
    model_revision: object,
) -> bool:
    """Return whether four simple values identify the pinned DenseSpark profile."""

    values = (recipe_source, recipe_revision, model_source, model_revision)
    return (
        all(type(value) is str for value in values)
        and recipe_source == DENSESPARK_RECIPE_SOURCE
        and recipe_revision == DENSESPARK_RECIPE_REVISION
        and model_source == DENSESPARK_MODEL_SOURCE
        and model_revision == DENSESPARK_MODEL_REVISION
    )


def require_densespark_profile_identity(
    *,
    recipe_source: object,
    recipe_revision: object,
    model_source: object,
    model_revision: object,
) -> None:
    """Require the exact pinned DenseSpark recipe and model identities."""

    if not is_densespark_profile_identity(
        recipe_source=recipe_source,
        recipe_revision=recipe_revision,
        model_source=model_source,
        model_revision=model_revision,
    ):
        raise DenseSparkContractError("DenseSpark profile identity is not the pinned profile")


def is_densespark_profile(model: object) -> bool:
    """Return whether a model-shaped object selects a managed C1 profile."""

    return type(getattr(model, "id", None)) is str and (
        getattr(model, "id") in DENSESPARK_PROFILE_IDS
    )


def is_densespark_warmup_sync_profile(model: object) -> bool:
    """Return whether an object selects the instrumentation-only sync arm."""

    return type(getattr(model, "id", None)) is str and (
        getattr(model, "id") == DENSESPARK_WARMUP_SYNC_PROFILE_ID
    )


def densespark_image_for_profile(profile_id: str) -> str:
    """Return the exact image tag for one managed profile ID."""

    if type(profile_id) is not str or profile_id not in DENSESPARK_PROFILE_IDS:
        raise DenseSparkContractError("DenseSpark managed profile ID does not match")
    if profile_id == DENSESPARK_WARMUP_SYNC_PROFILE_ID:
        return DENSESPARK_WARMUP_SYNC_IMAGE
    return DENSESPARK_IMAGE


def densespark_local_image_id_for_profile(profile_id: str) -> str:
    """Return the immutable local image ID for one managed profile ID."""

    if type(profile_id) is not str or profile_id not in DENSESPARK_PROFILE_IDS:
        raise DenseSparkContractError("DenseSpark managed profile ID does not match")
    if profile_id == DENSESPARK_WARMUP_SYNC_PROFILE_ID:
        return DENSESPARK_WARMUP_SYNC_LOCAL_IMAGE_ID
    return DENSESPARK_LOCAL_IMAGE_ID


def _exact_sequence(value: object) -> tuple[object, ...] | None:
    if type(value) not in {list, tuple}:
        return None
    return tuple(value)  # type: ignore[arg-type]


def validate_densespark_profile(model: object) -> None:
    """Require every execution-relevant field of the managed C1 profile."""

    if not is_densespark_profile(model):
        raise DenseSparkContractError("DenseSpark managed profile ID does not match")
    profile_id = getattr(model, "id")
    require_densespark_profile_identity(
        recipe_source=getattr(model, "recipe_source", None),
        recipe_revision=getattr(model, "recipe_revision", None),
        model_source=getattr(model, "source", None),
        model_revision=getattr(model, "revision", None),
    )
    exact = {
        "architecture": "dense+gdn",
        "backend": "vllm",
        "cache_dir": "user",
        "densespark_pq_digest": DENSESPARK_PQ_SHA256,
        "densespark_pq_file": DENSESPARK_PQ_CACHE_RELATIVE_PATH.as_posix(),
        "densespark_pq_size_bytes": DENSESPARK_PQ_SIZE_BYTES,
        "endpoint": "http://127.0.0.1:8000/v1",
        "estimated_ram_gib": 92.0,
        "image": densespark_image_for_profile(profile_id),
        "image_digest": None,
        "lifecycle": "docker",
        "local_image_id": densespark_local_image_id_for_profile(profile_id),
        "max_context": DENSESPARK_MAX_CONTEXT,
        "native_context": DENSESPARK_NATIVE_CONTEXT,
        "prefix_cache_mode": None,
        "quantization": "int4-autoround+densespark-pq",
        "request_body_json": None,
        "served_name": DENSESPARK_SERVED_NAME,
        "startup_timeout_s": 1_800,
        "support_status": (
            "exploratory"
            if profile_id == DENSESPARK_WARMUP_SYNC_PROFILE_ID
            else "spark_vllm_recipe"
        ),
        "weight_file_count": DENSESPARK_WEIGHT_FILE_COUNT,
        "weight_size_bytes": DENSESPARK_WEIGHT_SIZE_BYTES,
    }
    for name, expected in exact.items():
        value = getattr(model, name, None)
        if type(value) is not type(expected) or value != expected:
            raise DenseSparkContractError(
                f"DenseSpark managed profile field {name} does not match"
            )
    if _exact_sequence(getattr(model, "tasks", None)) != (
        "chat",
        "thinking",
        "tools",
    ):
        raise DenseSparkContractError(
            "DenseSpark managed profile capabilities do not match v1.2"
        )
    if _exact_sequence(getattr(model, "args", None)) != DENSESPARK_C1_ARGS:
        raise DenseSparkContractError("DenseSpark managed profile arguments do not match")


def validate_densespark_suite(suite: object) -> None:
    """Require one exact exportable timing or tool suite for DenseSpark."""

    suite_id = getattr(suite, "id", None)
    if type(suite_id) is not str or suite_id not in {
        DENSESPARK_SUITE_ID,
        DENSESPARK_TOOL_SUITE_ID,
    }:
        raise DenseSparkContractError("DenseSpark suite ID does not match")
    if suite_id == DENSESPARK_TOOL_SUITE_ID and getattr(
        suite, "description", None
    ) != DENSESPARK_TOOL_SUITE_DESCRIPTION:
        raise DenseSparkContractError(
            "DenseSpark tool suite description does not match"
        )
    expected_cases = (
        DENSESPARK_C1_CASES
        if suite_id == DENSESPARK_SUITE_ID
        else DENSESPARK_TOOL_CASES
    )
    cases = _exact_sequence(getattr(suite, "cases", None))
    if cases is None or len(cases) != len(expected_cases):
        raise DenseSparkContractError(
            "DenseSpark suite must contain its exact ordered cases"
        )
    for case, expected_case in zip(cases, expected_cases, strict=True):
        for name, expected in expected_case.items():
            value = getattr(case, name, None)
            if name == "requires":
                value = _exact_sequence(value)
            if type(value) is not type(expected) or value != expected:
                raise DenseSparkContractError(
                    f"DenseSpark C1 suite field {name} does not match"
                )


def validate_densespark_snapshot_receipt(
    *, weight_file_count: object, weight_size_bytes: object
) -> None:
    """Validate the scalar receipt produced by exact snapshot inspection."""

    if (
        type(weight_file_count) is not int
        or weight_file_count != DENSESPARK_WEIGHT_FILE_COUNT
        or type(weight_size_bytes) is not int
        or weight_size_bytes != DENSESPARK_WEIGHT_SIZE_BYTES
    ):
        raise DenseSparkContractError(
            "DenseSpark cached snapshot does not match its weight contract"
        )


def densespark_c1_environment() -> dict[str, str]:
    """Return a fresh copy of the fixed, non-secret C1 container environment."""

    return dict(DENSESPARK_C1_ENVIRONMENT)


def densespark_c1_cache_config(
    profile_id: str = DENSESPARK_PROFILE_ID,
) -> dict[str, bool | int | str]:
    """Return all graph- or generation-relevant C1 values used for cache isolation."""

    image_id = densespark_local_image_id_for_profile(profile_id)

    config: dict[str, bool | int | str] = {
        key: value
        for key, value in DENSESPARK_C1_ENVIRONMENT
        if key in DENSESPARK_CONFIG_KEYS
    }
    config.update(
        {
            "DENSESPARK_AUTO_TOOL_CHOICE": True,
            "DENSESPARK_CONCURRENCY": 1,
            "DENSESPARK_DRAFT_SAMPLE_METHOD": "probabilistic",
            "DENSESPARK_GDN_PREFILL_BACKEND": "flashinfer",
            "DENSESPARK_GPU_MEMORY_UTILIZATION": "0.86",
            "DENSESPARK_IMAGE_ID": image_id,
            "DENSESPARK_LIMIT_MM": '{"image":0,"video":0}',
            "DENSESPARK_LINEAR_BACKEND": "humming",
            "DENSESPARK_MAMBA_CACHE_MODE": "none",
            "DENSESPARK_MAMBA_SSM_CACHE_DTYPE": "bfloat16",
            "DENSESPARK_MAX_MODEL_LEN": DENSESPARK_MAX_CONTEXT,
            "DENSESPARK_MAX_NUM_BATCHED_TOKENS": 8_192,
            "DENSESPARK_MODEL_REVISION": DENSESPARK_MODEL_REVISION,
            "DENSESPARK_PQ_ARTIFACT_SHA256": DENSESPARK_PQ_SHA256,
            "DENSESPARK_PREFIX_CACHING": False,
            "DENSESPARK_REASONING_PARSER": "qwen3",
            "DENSESPARK_SPEC_TOKENS": 8,
            "DENSESPARK_TOOL_CALL_PARSER": DENSESPARK_TOOL_CALL_PARSER,
        }
    )
    if profile_id == DENSESPARK_WARMUP_SYNC_PROFILE_ID:
        config["DENSESPARK_WARMUP_MODE"] = DENSESPARK_WARMUP_SYNC_MODE
    return config


def densespark_pq_artifact_path(*, home: Path | None = None) -> Path:
    """Resolve the fixed PQ artifact beneath the user's cache root."""

    home_path = Path.home() if home is None else Path(home)
    return home_path / ".cache" / Path(*DENSESPARK_PQ_CACHE_RELATIVE_PATH.parts)


def densespark_compile_cache_path(
    *,
    home: Path | None = None,
    profile_id: str = DENSESPARK_PROFILE_ID,
) -> Path:
    """Resolve the C1 compile cache under its full configuration namespace."""

    home_path = Path.home() if home is None else Path(home)
    return (
        home_path
        / ".cache"
        / DENSESPARK_CACHE_ROOT_NAME
        / densespark_cache_namespace(densespark_c1_cache_config(profile_id))
    )


def _normalize_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise DenseSparkContractError(f"{field} must be a lowercase SHA-256 string")
    raw = value.removeprefix("sha256:")
    if _SHA256_RE.fullmatch(raw) is None:
        raise DenseSparkContractError(f"{field} must be a lowercase SHA-256 string")
    return raw


def validate_densespark_warmup_sync_sources(
    *, repository_root: Path | None = None
) -> dict[str, str]:
    """Verify the two checked-in inputs used to build the derived sync image.

    The returned receipt is scalar and path-free. Container-side source pins are
    separately bound by the Dockerfile and immutable derived image ID.
    """

    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root)
    )
    expected = {
        "probe_sha256": (
            DENSESPARK_WARMUP_SYNC_PROBE_RELATIVE_PATH,
            DENSESPARK_WARMUP_SYNC_PROBE_SHA256,
        ),
        "image_recipe_sha256": (
            DENSESPARK_WARMUP_SYNC_IMAGE_RECIPE_RELATIVE_PATH,
            DENSESPARK_WARMUP_SYNC_IMAGE_RECIPE_SHA256,
        ),
        "dockerignore_sha256": (
            DENSESPARK_WARMUP_SYNC_DOCKERIGNORE_RELATIVE_PATH,
            DENSESPARK_WARMUP_SYNC_DOCKERIGNORE_SHA256,
        ),
    }
    receipt: dict[str, str] = {}
    try:
        resolved_root = root.resolve(strict=True)
        if root.is_symlink() or not resolved_root.is_dir():
            raise DenseSparkContractError(
                "DenseSpark warmup-sync source root is unsafe"
            )
        for name, (relative, expected_sha256) in expected.items():
            source = resolved_root / Path(*relative.parts)
            if source.is_symlink():
                raise DenseSparkContractError(
                    "DenseSpark warmup-sync source must not be a symlink"
                )
            resolved = source.resolve(strict=True)
            resolved.relative_to(resolved_root)
            metadata_before = resolved.stat()
            if not stat.S_ISREG(metadata_before.st_mode):
                raise DenseSparkContractError(
                    "DenseSpark warmup-sync source must be a regular file"
                )
            payload = resolved.read_bytes()
            metadata_after = resolved.stat()
            if _file_metadata(metadata_before) != _file_metadata(metadata_after):
                raise DenseSparkContractError(
                    "DenseSpark warmup-sync source changed while being hashed"
                )
            actual = f"sha256:{hashlib.sha256(payload).hexdigest()}"
            if not secrets.compare_digest(actual, expected_sha256):
                raise DenseSparkContractError(
                    "DenseSpark warmup-sync source digest does not match its pin"
                )
            receipt[name] = actual
    except DenseSparkContractError:
        raise
    except (OSError, ValueError) as error:
        raise DenseSparkContractError(
            "DenseSpark warmup-sync sources could not be verified"
        ) from error
    return receipt


def _file_metadata(
    stat_result: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_mode,
        stat_result.st_nlink,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def validate_densespark_pq_artifact(
    path: str | os.PathLike[str],
    *,
    expected_size_bytes: int,
    expected_sha256: str,
) -> DenseSparkPQArtifact:
    """Verify an exact, regular, non-symlink PQ artifact without trusting its name.

    The descriptor is opened without following symlinks where the platform offers
    ``O_NOFOLLOW``.  Metadata is checked before and after hashing, and the path is
    checked once more afterward so replacement during validation is rejected.
    """

    if type(expected_size_bytes) is not int or expected_size_bytes <= 0:
        raise DenseSparkContractError("expected PQ artifact size must be a positive integer")
    expected_digest = _normalize_sha256(expected_sha256, field="expected PQ artifact digest")

    try:
        artifact_path = os.fspath(path)
    except TypeError as exc:
        raise DenseSparkContractError("PQ artifact path must be path-like") from exc
    if not isinstance(artifact_path, str) or not artifact_path:
        raise DenseSparkContractError("PQ artifact path must be a non-empty text path")

    descriptor = -1
    try:
        path_before = os.stat(artifact_path, follow_symlinks=False)
        if stat.S_ISLNK(path_before.st_mode):
            raise DenseSparkContractError("DenseSpark PQ artifact must not be a symlink")
        if not stat.S_ISREG(path_before.st_mode):
            raise DenseSparkContractError("DenseSpark PQ artifact must be a regular file")

        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(artifact_path, flags)
        descriptor_before = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_before.st_mode):
            raise DenseSparkContractError("DenseSpark PQ artifact must be a regular file")
        if (
            path_before.st_dev != descriptor_before.st_dev
            or path_before.st_ino != descriptor_before.st_ino
        ):
            raise DenseSparkContractError("DenseSpark PQ artifact changed while being opened")
        if descriptor_before.st_size != expected_size_bytes:
            raise DenseSparkContractError("DenseSpark PQ artifact size does not match its pin")

        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, _READ_SIZE)
            if not chunk:
                break
            digest.update(chunk)

        descriptor_after = os.fstat(descriptor)
        if _file_metadata(descriptor_before) != _file_metadata(descriptor_after):
            raise DenseSparkContractError("DenseSpark PQ artifact changed while being hashed")
        path_after = os.stat(artifact_path, follow_symlinks=False)
        if not stat.S_ISREG(path_after.st_mode):
            raise DenseSparkContractError("DenseSpark PQ artifact path changed after hashing")
        if _file_metadata(descriptor_after) != _file_metadata(path_after):
            raise DenseSparkContractError("DenseSpark PQ artifact path changed after hashing")
    except DenseSparkContractError:
        raise
    except OSError as exc:
        raise DenseSparkContractError("DenseSpark PQ artifact could not be verified") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    actual_digest = digest.hexdigest()
    if not secrets.compare_digest(actual_digest, expected_digest):
        raise DenseSparkContractError("DenseSpark PQ artifact digest does not match its pin")
    return DenseSparkPQArtifact(
        size_bytes=expected_size_bytes,
        sha256=f"sha256:{actual_digest}",
    )


def validate_densespark_snapshot(
    snapshot: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
) -> DenseSparkSnapshotReceipt:
    """Content-verify the complete pinned Hugging Face snapshot.

    The snapshot must contain exactly the reviewed top-level inventory.  Every
    entry must be the canonical relative Hugging Face symlink to its pinned
    blob.  Blobs are opened relative to an already-open blob-directory
    descriptor with symlink following disabled, then size-checked and SHA-256
    hashed.  Link, path, descriptor, and directory metadata are compared after
    hashing so same-size tampering, replacement, and concurrent mutation fail
    closed.  This deliberately reads all 19 GB before every managed launch.
    """

    snapshot_path = Path(snapshot)
    repository_path = Path(repository_root)
    snapshot_descriptor = -1
    blobs_descriptor = -1
    try:
        repository_before = os.stat(repository_path, follow_symlinks=False)
        if not stat.S_ISDIR(repository_before.st_mode):
            raise DenseSparkContractError(
                "DenseSpark repository root must be a non-symlink directory"
            )
        repository_resolved = repository_path.resolve(strict=True)
        blobs = repository_resolved / "blobs"
        snapshots = repository_resolved / "snapshots"
        blobs_before = os.stat(blobs, follow_symlinks=False)
        snapshots_before = os.stat(snapshots, follow_symlinks=False)
        snapshot_before = os.stat(snapshot_path, follow_symlinks=False)
        if not all(
            stat.S_ISDIR(metadata.st_mode)
            for metadata in (blobs_before, snapshots_before, snapshot_before)
        ):
            raise DenseSparkContractError(
                "DenseSpark snapshot topology must use non-symlink directories"
            )
        blobs_resolved = blobs.resolve(strict=True)
        blobs_resolved.relative_to(repository_resolved)
        snapshot_resolved = snapshot_path.resolve(strict=True)
        snapshot_resolved.relative_to(repository_resolved)
        expected_snapshot = (
            repository_resolved / "snapshots" / DENSESPARK_MODEL_REVISION
        )
        if snapshot_resolved != expected_snapshot:
            raise DenseSparkContractError(
                "DenseSpark cached snapshot is not the exact pinned revision path"
            )

        directory_flags = os.O_RDONLY
        directory_flags |= getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        blobs_descriptor = os.open(blobs_resolved, directory_flags)
        snapshot_descriptor = os.open(snapshot_resolved, directory_flags)
        blobs_opened = os.fstat(blobs_descriptor)
        snapshot_opened = os.fstat(snapshot_descriptor)
        if _file_metadata(blobs_before) != _file_metadata(blobs_opened):
            raise DenseSparkContractError(
                "DenseSpark repository blob directory changed while being opened"
            )
        if _file_metadata(snapshot_before) != _file_metadata(snapshot_opened):
            raise DenseSparkContractError(
                "DenseSpark snapshot directory changed while being opened"
            )
    except DenseSparkContractError:
        if snapshot_descriptor >= 0:
            os.close(snapshot_descriptor)
        if blobs_descriptor >= 0:
            os.close(blobs_descriptor)
        raise
    except (OSError, ValueError) as exc:
        if snapshot_descriptor >= 0:
            os.close(snapshot_descriptor)
        if blobs_descriptor >= 0:
            os.close(blobs_descriptor)
        raise DenseSparkContractError(
            "DenseSpark exact cached snapshot is unavailable or unsafe"
        ) from exc

    try:
        observed = set(os.listdir(snapshot_descriptor))
        if observed != set(DENSESPARK_SNAPSHOT_FILE_PINS):
            raise DenseSparkContractError(
                "DenseSpark cached snapshot has a different file layout"
            )

        identities: set[tuple[int, int]] = set()
        weight_identities: set[tuple[int, int]] = set()
        weight_total = 0
        for filename, pin in DENSESPARK_SNAPSHOT_FILE_PINS.items():
            blob_name, expected_size, expected_digest = pin
            if (
                type(filename) is not str
                or not filename
                or "/" in filename
                or "\\" in filename
                or type(blob_name) is not str
                or not blob_name
                or "/" in blob_name
                or "\\" in blob_name
                or type(expected_size) is not int
                or expected_size <= 0
                or _SHA256_RE.fullmatch(expected_digest) is None
            ):
                raise DenseSparkContractError(
                    "DenseSpark snapshot file pin is malformed"
                )

            entry_before = os.stat(
                filename,
                dir_fd=snapshot_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISLNK(entry_before.st_mode):
                raise DenseSparkContractError(
                    "DenseSpark snapshot entries must use pinned blob symlinks"
                )
            expected_link = f"../../blobs/{blob_name}"
            link_before = os.readlink(filename, dir_fd=snapshot_descriptor)
            if link_before != expected_link:
                raise DenseSparkContractError(
                    "DenseSpark snapshot entry selects an unsafe or unexpected blob"
                )

            blob_before = os.stat(
                blob_name,
                dir_fd=blobs_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(blob_before.st_mode)
                or blob_before.st_nlink != 1
            ):
                raise DenseSparkContractError(
                    "DenseSpark snapshot blob must be a single-link regular file"
                )
            if blob_before.st_size != expected_size:
                raise DenseSparkContractError(
                    "DenseSpark snapshot blob size does not match its pin"
                )

            blob_descriptor = -1
            try:
                file_flags = os.O_RDONLY
                file_flags |= getattr(os, "O_CLOEXEC", 0)
                file_flags |= getattr(os, "O_NOFOLLOW", 0)
                blob_descriptor = os.open(
                    blob_name,
                    file_flags,
                    dir_fd=blobs_descriptor,
                )
                descriptor_before = os.fstat(blob_descriptor)
                if (
                    not stat.S_ISREG(descriptor_before.st_mode)
                    or _file_metadata(blob_before)
                    != _file_metadata(descriptor_before)
                ):
                    raise DenseSparkContractError(
                        "DenseSpark snapshot blob changed while being opened"
                    )

                digest = hashlib.sha256()
                while True:
                    chunk = os.read(blob_descriptor, _READ_SIZE)
                    if not chunk:
                        break
                    digest.update(chunk)

                descriptor_after = os.fstat(blob_descriptor)
                if _file_metadata(descriptor_before) != _file_metadata(
                    descriptor_after
                ):
                    raise DenseSparkContractError(
                        "DenseSpark snapshot blob changed while being hashed"
                    )
                blob_after = os.stat(
                    blob_name,
                    dir_fd=blobs_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(blob_after.st_mode)
                    or _file_metadata(descriptor_after)
                    != _file_metadata(blob_after)
                ):
                    raise DenseSparkContractError(
                        "DenseSpark snapshot blob path changed while being hashed"
                    )
            finally:
                if blob_descriptor >= 0:
                    os.close(blob_descriptor)

            if not secrets.compare_digest(digest.hexdigest(), expected_digest):
                raise DenseSparkContractError(
                    "DenseSpark snapshot blob digest does not match its pin"
                )

            entry_after = os.stat(
                filename,
                dir_fd=snapshot_descriptor,
                follow_symlinks=False,
            )
            link_after = os.readlink(filename, dir_fd=snapshot_descriptor)
            if (
                _file_metadata(entry_before) != _file_metadata(entry_after)
                or link_after != link_before
            ):
                raise DenseSparkContractError(
                    "DenseSpark snapshot entry changed while being hashed"
                )

            identity = (descriptor_after.st_dev, descriptor_after.st_ino)
            if identity in identities:
                raise DenseSparkContractError(
                    "DenseSpark cached snapshot aliases two pinned files"
                )
            identities.add(identity)
            if filename in DENSESPARK_WEIGHT_FILES:
                weight_identities.add(identity)
                weight_total += descriptor_after.st_size

        blobs_after = os.fstat(blobs_descriptor)
        snapshot_after = os.fstat(snapshot_descriptor)
        if _file_metadata(blobs_opened) != _file_metadata(blobs_after):
            raise DenseSparkContractError(
                "DenseSpark blob directory changed during snapshot validation"
            )
        if _file_metadata(snapshot_opened) != _file_metadata(snapshot_after):
            raise DenseSparkContractError(
                "DenseSpark snapshot directory changed during validation"
            )
        blobs_path_after = os.stat(blobs_resolved, follow_symlinks=False)
        snapshot_path_after = os.stat(snapshot_resolved, follow_symlinks=False)
        repository_after = os.stat(repository_resolved, follow_symlinks=False)
        if _file_metadata(blobs_after) != _file_metadata(blobs_path_after):
            raise DenseSparkContractError(
                "DenseSpark blob directory path changed during validation"
            )
        if _file_metadata(snapshot_after) != _file_metadata(snapshot_path_after):
            raise DenseSparkContractError(
                "DenseSpark snapshot directory path changed during validation"
            )
        if _file_metadata(repository_before) != _file_metadata(repository_after):
            raise DenseSparkContractError(
                "DenseSpark repository root changed during validation"
            )
    except DenseSparkContractError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise DenseSparkContractError(
            "DenseSpark cached snapshot contains an unsafe pinned file"
        ) from exc
    finally:
        if snapshot_descriptor >= 0:
            os.close(snapshot_descriptor)
        if blobs_descriptor >= 0:
            os.close(blobs_descriptor)

    validate_densespark_snapshot_receipt(
        weight_file_count=len(weight_identities),
        weight_size_bytes=weight_total,
    )
    return DenseSparkSnapshotReceipt(
        weight_file_count=len(weight_identities),
        weight_size_bytes=weight_total,
    )


def canonical_densespark_config(config: Mapping[str, object]) -> dict[str, bool | int | str]:
    """Return a sorted copy of a scalar DenseSpark config after allowlist checks."""

    if not isinstance(config, Mapping):
        raise DenseSparkContractError("DenseSpark configuration must be a mapping")

    canonical: dict[str, bool | int | str] = {}
    for key, value in config.items():
        if type(key) is not str or key not in DENSESPARK_CONFIG_KEYS:
            raise DenseSparkContractError("DenseSpark configuration contains an unknown key")
        if type(value) is bool:
            canonical[key] = value
        elif type(value) is int:
            if value < 0 or value > _CONFIG_INTEGER_MAX:
                raise DenseSparkContractError("DenseSpark integer configuration is out of range")
            canonical[key] = value
        elif type(value) is str:
            if not value or len(value) > 4096 or any(ord(character) < 32 for character in value):
                raise DenseSparkContractError("DenseSpark string configuration is invalid")
            canonical[key] = value
        else:
            raise DenseSparkContractError("DenseSpark configuration values must be scalar")
    return {key: canonical[key] for key in sorted(canonical)}


def densespark_configuration_digest(config: Mapping[str, object]) -> str:
    """Return a deterministic, profile-bound SHA-256 digest for a configuration."""

    payload = {
        "config": canonical_densespark_config(config),
        "profile": {
            "model_revision": DENSESPARK_MODEL_REVISION,
            "model_source": DENSESPARK_MODEL_SOURCE,
            "recipe_revision": DENSESPARK_RECIPE_REVISION,
            "recipe_source": DENSESPARK_RECIPE_SOURCE,
        },
        "schema": DENSESPARK_CONFIG_SCHEMA,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def densespark_cache_namespace(config: Mapping[str, object]) -> str:
    """Return a path-component-safe namespace for configuration-specific caches."""

    digest = densespark_configuration_digest(config).removeprefix("sha256:")
    return f"densespark-v1-{digest}"


def densespark_expected_launch_policy() -> dict[str, int | str]:
    """Return the exact path-free managed-launch policy and its digest.

    This receipt is intentionally separate from the immutable artifact receipt:
    old plans can therefore be identified honestly as pre-policy evidence while
    every newly frozen plan must carry this complete launch contract.
    """

    policy: dict[str, int | str] = {
        "artifact_download_policy": "disabled",
        "docker_network": DENSESPARK_LAUNCH_DOCKER_NETWORK,
        "docker_network_egress": "capable",
        "docker_network_isolation": "none",
        "docker_pull_policy": "never",
        "environment_hf_hub_offline": "1",
        "environment_vllm_no_usage_stats": "1",
        "host_safety_max_starting_swap_bytes": (
            DENSESPARK_STARTUP_MAX_STARTING_SWAP_MIB * 1024**2
        ),
        "host_safety_max_swap_growth_bytes": (
            DENSESPARK_STARTUP_MAX_SWAP_GROWTH_MIB * 1024**2
        ),
        "host_safety_min_memavailable_bytes": (
            DENSESPARK_STARTUP_MIN_MEMAVAILABLE_GIB * 1024**3
        ),
        "label_backend": "ai.sparkbench.backend=vllm",
        "label_managed": "ai.sparkbench.managed=true",
        "label_run_binding": "ai.sparkbench.run=frozen-run-identity",
        "publish_container_port": DENSESPARK_LAUNCH_CONTAINER_PORT,
        "publish_host": DENSESPARK_LAUNCH_HOST,
        "publish_host_port": DENSESPARK_LAUNCH_HOST_PORT,
    }
    payload = {
        "policy": policy,
        "profile": {
            "model_revision": DENSESPARK_MODEL_REVISION,
            "model_source": DENSESPARK_MODEL_SOURCE,
            "recipe_revision": DENSESPARK_RECIPE_REVISION,
            "recipe_source": DENSESPARK_RECIPE_SOURCE,
        },
        "schema": DENSESPARK_LAUNCH_POLICY_SCHEMA,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        **policy,
        "schema": DENSESPARK_LAUNCH_POLICY_SCHEMA,
        "sha256": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
    }


def densespark_expected_resolved_provenance(
    profile_id: str = DENSESPARK_PROFILE_ID,
) -> dict[str, int | str]:
    """Return the exact scalar receipt frozen into a managed C1 plan.

    Keeping this receipt beside the configuration contract prevents execution
    and evidence publication from independently reconstructing (and drifting
    on) the immutable local image, PQ artifact, or compile-cache identity.
    """

    config = densespark_c1_cache_config(profile_id)
    receipt: dict[str, int | str] = {
        "cache_namespace": densespark_cache_namespace(config),
        "configuration_sha256": densespark_configuration_digest(config),
        "docker_image_id": densespark_local_image_id_for_profile(profile_id),
        "model_revision": DENSESPARK_MODEL_REVISION,
        "pq_artifact_sha256": DENSESPARK_PQ_SHA256,
        "pq_artifact_size_bytes": DENSESPARK_PQ_SIZE_BYTES,
        "weight_file_count": DENSESPARK_WEIGHT_FILE_COUNT,
        "weight_size_bytes": DENSESPARK_WEIGHT_SIZE_BYTES,
    }
    if profile_id == DENSESPARK_WARMUP_SYNC_PROFILE_ID:
        receipt.update(
            {
                "base_docker_image_id": DENSESPARK_LOCAL_IMAGE_ID,
                "dockerignore_sha256": (
                    DENSESPARK_WARMUP_SYNC_DOCKERIGNORE_SHA256
                ),
                "fused_sigmoid_source_sha256": (
                    DENSESPARK_WARMUP_SYNC_FUSED_SIGMOID_SOURCE_SHA256
                ),
                "image_recipe_sha256": (
                    DENSESPARK_WARMUP_SYNC_IMAGE_RECIPE_SHA256
                ),
                "kernel_warmup_source_sha256": (
                    DENSESPARK_WARMUP_SYNC_KERNEL_WARMUP_SOURCE_SHA256
                ),
                "mamba_utils_source_sha256": (
                    DENSESPARK_WARMUP_SYNC_MAMBA_UTILS_SOURCE_SHA256
                ),
                "mode": DENSESPARK_WARMUP_SYNC_MODE,
                "probe_sha256": DENSESPARK_WARMUP_SYNC_PROBE_SHA256,
                "qwen_gdn_source_sha256": (
                    DENSESPARK_WARMUP_SYNC_QWEN_GDN_SOURCE_SHA256
                ),
                "qwen_warmup_source_sha256": (
                    DENSESPARK_WARMUP_SYNC_QWEN_WARMUP_SOURCE_SHA256
                ),
                "vllm_entrypoint_sha256": (
                    DENSESPARK_WARMUP_SYNC_VLLM_ENTRYPOINT_SHA256
                ),
            }
        )
    return receipt


def _subprocess_runner(command: Sequence[str]) -> CommandResult:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )


def validate_densespark_local_image(
    image: str,
    *,
    expected_image_id: str,
    runner: CommandRunner = _subprocess_runner,
) -> str:
    """Require that a local Docker tag resolves to one exact image ID.

    Tests and callers that already centralize subprocess policy can inject a runner.
    Only the result protocol above is required; stderr is deliberately never copied
    into contract errors.
    """

    if type(image) is not str or _IMAGE_REFERENCE_RE.fullmatch(image) is None:
        raise DenseSparkContractError("DenseSpark Docker image reference is invalid")
    expected_digest = _normalize_sha256(expected_image_id, field="expected Docker image ID")
    if not expected_image_id.startswith("sha256:"):
        raise DenseSparkContractError("expected Docker image ID must include the sha256: prefix")
    expected = f"sha256:{expected_digest}"
    command = ("docker", "image", "inspect", "--format", "{{.Id}}", image)
    try:
        result = runner(command)
    except (OSError, subprocess.SubprocessError) as exc:
        raise DenseSparkContractError("DenseSpark Docker image could not be inspected") from exc

    returncode = getattr(result, "returncode", None)
    stdout = getattr(result, "stdout", None)
    if type(returncode) is not int or returncode != 0:
        raise DenseSparkContractError("DenseSpark Docker image is not available locally")
    if not isinstance(stdout, str):
        raise DenseSparkContractError("DenseSpark Docker inspection returned invalid output")
    lines = stdout.splitlines()
    if len(lines) != 1 or _SHA256_RE.fullmatch(lines[0].removeprefix("sha256:")) is None:
        raise DenseSparkContractError("DenseSpark Docker inspection returned invalid output")
    if not lines[0].startswith("sha256:"):
        raise DenseSparkContractError("DenseSpark Docker inspection returned invalid output")
    if not secrets.compare_digest(lines[0], expected):
        raise DenseSparkContractError("DenseSpark Docker image ID does not match its pin")
    return expected
