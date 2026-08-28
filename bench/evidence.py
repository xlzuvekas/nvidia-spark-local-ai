"""Deterministic, privacy-safe publication of ignored benchmark results.

Raw result bundles deliberately contain prompts, completions, request identifiers,
commands, host paths, and free-form errors.  This module never copies those
objects.  It constructs a small, typed scalar evidence corpus from explicit
allowlists and validates the materialized output before publication.
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import statistics
import subprocess
import tempfile
from typing import Any, Sequence
import unicodedata

from .annotations import (
    measurement_annotations,
    normalize_startup_safety_gate,
    startup_safety_gate_annotations_from_annotations,
    startup_safety_gates_from_annotations,
)
from .manifest import KNOWN_AGENTIC_CASE_IDS
from .memory_ops import (
    MEMORY_OPERATION_CONTEXT_TOKENS,
    MEMORY_OPERATION_LLAMACPP_DIGEST,
    MEMORY_OPERATION_LLAMACPP_REVISION,
    MEMORY_OPERATION_OUTPUT_TOKENS,
    MEMORY_OPERATION_PROTOCOL_DIGEST,
    MEMORY_OPERATION_SCENARIO_IDS,
    MEMORY_OPERATION_SERVER_TIMING_TOLERANCE_S,
    MEMORY_OPERATION_SUITE_ID,
    MEMORY_OPERATION_VARIANT_COUNT,
    memory_operation_llamacpp_args,
    require_memory_operation_protocol_digest,
    summarize_memory_operation_results,
)
from .prefix_cache_protocol import (
    PREFIX_CACHE_CONTEXT_TOKENS,
    PREFIX_CACHE_PREFIX_TARGETS,
    PREFIX_CACHE_PROTOCOL,
    PREFIX_CACHE_SUITE_ID,
    prefix_cache_conditions,
    prefix_cache_steps,
)
from .report import _summarize_prefix_cache_case


SCHEMA_VERSION = "sparkbench-evidence-v1"
SANITIZATION_POLICY = "strict-scalar-allowlist-v1"
HARBOR_EVIDENCE_KIND = "harbor_terminal_campaign"
HARBOR_SCHEMA_VERSION = "sparkbench-harbor-evidence-v1"
HARBOR_CAMPAIGN_ID = "qwen3-coder-next-harbor-terminal-offline-2026-08-18"
HARBOR_EXPECTED_GIT_REVISION = "26600d4abe48c082ce6764a61618516837069b9c"
HARBOR_EXPECTED_DERIVATION_DIGEST = (
    "sha256:4749be56af707f6d7615ac5cdb0fb7fa8d50fcdd49e5d4c9a9bfebb71677b4ef"
)
HARBOR_REPLICATE_COUNT = 2
LOOP_EVIDENCE_KIND = "rlm_halo_loop_campaign"
LOOP_RESULT_ROOTS = ("loop-campaigns", "loop-smoke-plans", "loop-smokes")
AUTORESEARCH_RESULT_ROOT = "autoresearch"
AUTORESEARCH_CAMPAIGN_SCHEMA_VERSION = 2
AUTORESEARCH_CELL_COUNT = 14
AUTORESEARCH_SUITE_ID = "qwen38-flash-next-sglang-agent64k-autoresearch"
AUTORESEARCH_SUITE_DESCRIPTION = (
    "Immutable single-user coding/cowork proxy for the exact "
    "Qwen3.8-Flash-Next 64K autoresearch baseline and its three one-axis "
    "candidates."
)
AUTORESEARCH_SUITE_SPEC_DIGEST = (
    "260506c71f890e714b50829e69289fdc1e2490b1c7d5a8a08218c5369128a063"
)
MAX_SOURCE_JSON_BYTES = 16 * 1024 * 1024
MAX_SOURCE_LINE_BYTES = 1024 * 1024
MAX_OUTPUT_FILE_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_TOTAL_BYTES = 25 * 1024 * 1024
MAX_STRING_LENGTH = 256
TELEMETRY_SAMPLES_PER_CHUNK = 2_000
TELEMETRY_COLUMNS = (
    "elapsed_s",
    "gpu_error_present",
    "gpu_util_pct",
    "memory_util_pct",
    "power_w",
    "sm_clock_mhz",
    "temperature_c",
    "cached_bytes",
    "memavailable_bytes",
    "memfree_bytes",
    "swapfree_bytes",
    "swaptotal_bytes",
)

_SGLANG_SOURCE_OVERLAY_FIELDS = frozenset(
    {"container_path", "digest", "host_path"}
)
_SGLANG_SOURCE_OVERLAY_HOST_PREFIX = ("results", "runtime-overlays")
_SGLANG_SOURCE_OVERLAY_CONTAINER_ROOT = PurePosixPath(
    "/sgl-workspace/sglang/python/sglang"
)
_SGLANG_PLE_CACHE_SOURCE_FIELDS = frozenset(
    {
        "sglang_ple_cache_marker_digest",
        "sglang_ple_cache_mode",
        "sglang_ple_cache_payload_digest",
    }
)
_SGLANG_PLE_CACHE_RUNTIME_HASH_FIELDS = frozenset(
    {
        "sglang_ple_cache_marker_sha256",
        "sglang_ple_cache_payload_sha256",
    }
)
_SGLANG_PROVENANCE_VERSION = 1
_SGLANG_PROVENANCE_CURRENT_VERSION = 2
_SGLANG_PROVENANCE_V1_RUNTIME_FIELDS = frozenset(
    {
        "sglang_ple_cache_mode",
        "sglang_ple_mmap",
        "sglang_provenance_version",
        "sglang_source_overlay_artifacts",
    }
)
_SGLANG_PROVENANCE_V2_RUNTIME_FIELDS = frozenset(
    _SGLANG_PROVENANCE_V1_RUNTIME_FIELDS | {"sglang_ple_omitted"}
)
_SGLANG_PROVENANCE_RUNTIME_FIELDS = frozenset(
    _SGLANG_PROVENANCE_V2_RUNTIME_FIELDS
    | _SGLANG_PLE_CACHE_RUNTIME_HASH_FIELDS
)
_SGLANG_PLE_CACHE_MODES = frozenset(
    {
        "disabled",
        "legacy_unspecified",
        "readonly",
        "writable",
    }
)
_QWEN38_PLE_OMISSION_IDENTITY = {
    "source": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
    "revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
    "recipe_source": "hashd1ve/qwen38-flash-next-one-dgx-spark",
    "recipe_revision": "bf2b7c75870d3703730b6bd8f3bb93dc622c278d",
}
_QWEN38_PLE_OMISSION_ARTIFACTS = [
    {
        "sha256": (
            "bcdc2c86aa59784ffe27d53c8d214e56"
            "b6aa45c02b1d5841fd956d1f006d6030"
        ),
        "target": "qwen4_exp.py",
    },
    {
        "sha256": (
            "e30566492e1502f94a4c7fed42d90b5"
            "23bbb662580c628459e6e63c7b5263c75"
        ),
        "target": "qwen_sparse_attn_backend.py",
    },
]
_QWEN38_PLE_STUDY_CACHE = {
    "sglang_ple_cache_marker_digest": (
        "sha256:f0ef55e4e4dec9b6b936a42af4ca2e"
        "b9b2f24ced373b1e216f7a6d507b171665"
    ),
    "sglang_ple_cache_payload_digest": (
        "sha256:b070f9644adf93794d8a1030584ab705"
        "809387e64396a9327a68fa3a3a6666b3"
    ),
}

# These bundles predate explicit SGLang PLE/overlay provenance.  Compatibility
# is deliberately bound to their complete bundle digests: accepting arbitrary
# v1 SGLang manifests without the new discriminator would make wholesale
# removal of provenance indistinguishable from a legacy bundle.
_LEGACY_SGLANG_PROVENANCE_BUNDLES = {
    "20260816T163510Z-phi-4-multimodal-instruct-nvfp4-smoke-faa306b5": (
        "898dc03f240e70d60b1ec19d3ca1ad40503813115f07427bf642f7d428f2a092"
    ),
    "20260816T163733Z-phi-4-multimodal-instruct-nvfp4-smoke-9adcae20": (
        "101ade1d1887a9a7d00d0369667ae5d6fac1eae4388b6cc681ce5ef2fb6a9026"
    ),
    "20260816T164642Z-phi-4-multimodal-instruct-fp8-smoke-7d4c8353": (
        "acdbba1789132e12def491e4ab740c2c4d59a82baee5f757a9bac24174e5fe3b"
    ),
    "20260816T164847Z-phi-4-multimodal-instruct-fp8-vision-eb08efc1": (
        "4458afe96b3c2be28bb41cf0020dab0bb650b7dd69bf582d09b5387c7fcce7d6"
    ),
    "20260816T165046Z-phi-4-multimodal-instruct-fp8-quick-65e98f95": (
        "b2b9bb81ee9f2dcaa3cfd8912b711b6448b80fe1c3acbb0995fe897ba7b7125e"
    ),
    "20260816T165332Z-phi-4-multimodal-instruct-fp8-reasoning-core-7a48a87e": (
        "f8d4ec1f3a0bcd2390331aa5af32d7ed1fcc08b0b2d51a81d5d201851d3d417e"
    ),
    "20260816T174341Z-phi-4-multimodal-instruct-fp8-audio-audio-asr-56e2f4bb": (
        "bae72ca5bf42998478ced7ec2a5b5e886bffa57d2595bf12bb39dec2aa586370"
    ),
    "20260817T065426Z-qwen38-27b-nvfp4-dspark-sglang-core-6b79826a": (
        "eb16431841c91266685a93dbc4cd3c059c84a5f3cdfa24b01be6c6d4379fc263"
    ),
    "20260817T065612Z-qwen38-27b-nvfp4-dspark-sglang-core-6b79826a": (
        "4ded70a72907dac2424b83c96a2c35f0b2a951ec17865932b6d4f5a23abb96fa"
    ),
    "20260826T190843Z-qwen38-flash-next-tiny-qsa-disabled-sglang-smoke-30d30d00": (
        "1238b75f558e9cadecae3a514150ef8f7ebe8ff06397532cd299c52970d9482e"
    ),
    "20260826T190953Z-qwen38-flash-next-tiny-dummy-nextn-sglang-smoke-931e5c58": (
        "a9abb5b45e7a0a86770d220910c8f3f563b78e16844d395169d122237849e432"
    ),
}

_COLD_START_SAFETY_SCALARS = {
    "cold_start_swap_growth_exceeded_safety_limit": (
        "swap_growth_mib",
        "safety_limit_mib",
        "memavailable_gib",
        "memory_psi_full_avg10",
    ),
    "ple_materialization_swap_growth_exceeded_safety_limit": (
        "swap_growth_mib",
        "safety_limit_mib",
        "memavailable_gib",
        "memory_psi_full_avg10",
        "ple_allocated_blocks",
    ),
}
_COLD_START_SOURCE_ANNOTATION_FIELDS = frozenset(
    {"evidence", "measurement_valid", "reason", "scope", "timestamp"}
)
_COLD_START_DECIMAL_RE = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_COLD_START_INTEGER_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_COLD_START_MAX_MEMORY_MIB = 1024.0 * 1024.0
_COLD_START_MAX_MEMORY_GIB = 1024.0 * 1024.0


class EvidenceError(RuntimeError):
    """Raised when source or generated evidence fails closed validation."""


_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+@/-]{0,255}\Z")
_SAFE_TEXT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._:+@/(),=+-]{0,255}\Z")
_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")
_REVISION_RE = re.compile(r"[0-9a-f]{7,64}\Z")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}\Z")
_RUN_ID_RE = re.compile(r"20[0-9]{6}T[0-9]{6,12}Z-[A-Za-z0-9_.-]+\Z")
_AUTORESEARCH_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_AUTORESEARCH_PATH_COMPONENT_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,255}\Z"
)
_AUTORESEARCH_CAMPAIGN_FIELDS = frozenset(
    {
        "schema_version",
        "created_at",
        "harness_tree_sha256",
        "harness_file_count",
        "preview",
        "preview_digest",
        "cells",
        "execution_started",
        "integrity_hash",
    }
)
_AUTORESEARCH_PREVIEW_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "cutoff",
        "baseline_id",
        "suite_id",
        "policy",
        "policy_digest",
        "proposals",
        "planned_cell_count",
        "execution_started",
    }
)
_AUTORESEARCH_CELL_FIELDS = frozenset(
    {
        "cell_id",
        "stage",
        "candidate_id",
        "arm",
        "profile_id",
        "ordinal",
        "run_dir",
        "plan_fingerprint",
        "plan_integrity_hash",
        "run_nonce",
    }
)
_GROUPED_RUN_ROOT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"qwen36-core-(20[0-9]{6}T[0-9]{6})\Z"), "%Y%m%dT%H%M%S"),
    (re.compile(r"reasoning-(20[0-9]{6})\Z"), "%Y%m%d"),
)

_LOOP_PLAN_TOP_FIELDS = frozenset(
    {
        "campaign_id",
        "cases",
        "created_at",
        "dataset",
        "description",
        "fingerprint",
        "halo",
        "integrity_hash",
        "models",
        "protocol_version",
        "repository",
        "rlm",
        "schema_version",
        "upstreams",
        "window",
        "worker",
    }
)
_LOOP_MODEL_SOURCE_FIELDS = frozenset(
    {
        "architecture",
        "args",
        "backend",
        "cache_dir",
        "description",
        "draft_model_digest",
        "draft_model_file",
        "draft_model_size_bytes",
        "draft_revision",
        "draft_source",
        "draft_weight_size_bytes",
        "endpoint",
        "estimated_ram_gib",
        "fetch_allow_patterns",
        "fetch_ignore_patterns",
        "id",
        "image",
        "image_digest",
        "lifecycle",
        "max_context",
        "mmproj_digest",
        "mmproj_file",
        "mmproj_size_bytes",
        "model_digest",
        "model_file",
        "model_shards",
        "model_size_bytes",
        "native_context",
        "prefix_cache_mode",
        "quantization",
        "recipe_revision",
        "recipe_source",
        "request_body_json",
        "revision",
        "runtime_binary",
        "runtime_digest",
        "runtime_parallel",
        "runtime_python",
        "runtime_revision",
        "runtime_source_dir",
        "served_name",
        "sglang_allow_hf_metadata_probe",
        "sglang_ple_mmap",
        "sglang_ple_omitted",
        "sglang_source_overlays",
        "source",
        "startup_timeout_s",
        "support_status",
        "tasks",
        "weight_file_count",
        "weight_size_bytes",
    }
)
_LOOP_MODEL_SOURCE_FIELDS_WITH_PLE_CACHE = (
    _LOOP_MODEL_SOURCE_FIELDS | _SGLANG_PLE_CACHE_SOURCE_FIELDS
)
_LOOP_MODEL_PRE_OMISSION_SOURCE_FIELDS = _LOOP_MODEL_SOURCE_FIELDS - {
    "sglang_ple_omitted"
}
_LOOP_MODEL_PRE_OMISSION_SOURCE_FIELDS_WITH_PLE_CACHE = (
    _LOOP_MODEL_PRE_OMISSION_SOURCE_FIELDS | _SGLANG_PLE_CACHE_SOURCE_FIELDS
)
_LOOP_MODEL_LEGACY_SOURCE_FIELDS = _LOOP_MODEL_SOURCE_FIELDS - {
    "sglang_ple_mmap",
    "sglang_ple_omitted",
    "sglang_source_overlays",
}
_LOOP_CASE_FIELDS = frozenset(
    {
        "admission_status",
        "case_id",
        "compaction",
        "compaction_threshold_pct",
        "context_length",
        "max_concurrent_subcalls",
        "max_depth",
        "max_iterations",
        "max_output_tokens",
        "max_parallel",
        "max_total_tokens",
        "max_turns",
        "phase",
        "reasoning_control",
        "reasoning_effort",
        "replicate",
        "row_index",
        "seed",
        "task",
        "timeout_s",
        "trace_count",
        "treatment",
    }
)
_LOOP_CASE_STRING_FIELDS = frozenset(
    {
        "admission_status",
        "case_id",
        "context_length",
        "phase",
        "reasoning_control",
        "reasoning_effort",
        "task",
        "treatment",
    }
)
_LOOP_CASE_BOOLEAN_FIELDS = frozenset({"compaction"})
_LOOP_CASE_INTEGER_FIELDS = frozenset(
    {
        "max_concurrent_subcalls",
        "max_depth",
        "max_iterations",
        "max_output_tokens",
        "max_parallel",
        "max_total_tokens",
        "max_turns",
        "replicate",
        "row_index",
        "seed",
        "timeout_s",
        "trace_count",
    }
)
_LOOP_BOUND_DIMENSION_FIELDS = frozenset(
    {
        "admission_status",
        "compaction",
        "compaction_threshold_pct",
        "context_length",
        "max_depth",
        "phase",
        "reasoning_control",
        "reasoning_effort",
        "replicate",
        "row_index",
        "seed",
        "task",
        "trace_count",
        "treatment",
    }
)
_LOOP_MEASUREMENT_BOOLEAN_FIELDS = frozenset(
    {
        "compaction",
        "compaction_enabled",
        "correct",
        "json_valid",
        "root_finalized",
        "run_code_disabled",
        "usage_includes_recursive_children",
    }
)
_LOOP_MEASUREMENT_INTEGER_FIELDS = frozenset(
    {
        "assistant_items",
        "attempt",
        "child_tool_calls",
        "child_turns",
        "compaction_count",
        "completed_subagents",
        "durable_items",
        "final_answer_chars",
        "iterations",
        "max_depth",
        "max_observed_depth",
        "observed_subagents",
        "output_chars",
        "predicted_family_count",
        "recursive_subcalls",
        "replicate",
        "reported_calls",
        "reported_input_tokens",
        "reported_output_tokens",
        "row_index",
        "seed",
        "subagent_requests",
        "tool_calls",
        "trace_count",
    }
)
_LOOP_MEASUREMENT_NUMERIC_FIELDS = frozenset(
    {
        "citation_family_coverage",
        "citation_precision",
        "compaction_threshold_pct",
        "effective_generation_tps",
        "engine_wall_s",
        "exact_count_rate",
        "family_f1",
        "family_precision",
        "family_recall",
        "mean_count_accuracy",
        "vllm_cached_prompt_tokens",
        "vllm_generation_tokens",
        "vllm_prefix_cache_hit_rate",
        "vllm_prefix_cache_hits",
        "vllm_prefix_cache_queries",
        "vllm_prompt_tokens",
        "vllm_successful_requests",
        "wall_s",
    }
)
_LOOP_MEASUREMENT_STRING_FIELDS = _LOOP_CASE_STRING_FIELDS | frozenset(
    {"profile_id"}
)
_LOOP_MEASUREMENT_FIELDS = (
    _LOOP_MEASUREMENT_BOOLEAN_FIELDS
    | _LOOP_MEASUREMENT_INTEGER_FIELDS
    | _LOOP_MEASUREMENT_NUMERIC_FIELDS
    | _LOOP_MEASUREMENT_STRING_FIELDS
)
_LOOP_MEASUREMENT_OUTPUT_FIELDS = (
    _LOOP_MEASUREMENT_FIELDS - {"tool_calls"}
) | {"executed_tool_call_count"}
_LOOP_EVENT_TYPES = frozenset(
    {
        "campaign_cleanup_failed",
        "campaign_cleanup_verified",
        "campaign_failed",
        "campaign_finished",
        "campaign_resumed",
        "campaign_started",
        "case_complete",
        "case_exhausted",
        "case_failed",
        "case_skipped_campaign_stop",
        "case_skipped_deadline",
        "case_skipped_held",
        "case_started",
        "case_timeout",
        "halo_fallback_selected",
        "halo_index_complete",
        "halo_profile_rejected",
        "server_ready",
        "server_recovered",
        "server_starting",
        "server_stop_failed",
        "server_stopped",
        "worker_network_cleanup_failed",
        "worker_network_cleanup_inspection_failed",
        "worker_network_cleanup_refused",
        "worker_network_ready",
        "worker_network_removed",
        "worker_source_staged",
        "workers_recovered",
    }
)
_LOOP_EVENT_FIELDS = _LOOP_MEASUREMENT_FIELDS | frozenset(
    {
        "completed_cases",
        "container_count",
        "error_cause_frame_file",
        "error_cause_frame_function",
        "error_cause_frame_line",
        "error_cause_type",
        "error_code",
        "error_frame_file",
        "error_frame_function",
        "error_frame_line",
        "error_http_status",
        "error_token_limit",
        "error_tokens_used",
        "error_type",
        "event",
        "fixture_id",
        "index_size_bytes",
        "index_wall_s",
        "indexed_trace_count",
        "plan_fingerprint",
        "recovery",
        "repository_revision",
        "startup_s",
        "status",
        "timestamp",
        "timeout_s",
    }
)
_LOOP_SUMMARY_COMMON_GROUP_FIELDS = frozenset(
    {
        "cache_fraction",
        "cached_prompt_tokens",
        "completed_cases",
        "effective_generation_tps",
        "generation_tokens",
        "mean_wall_s",
        "phase",
        "planned_cases",
        "profile_id",
        "prompt_tokens",
        "reasoning_control",
        "reasoning_effort",
        "treatment",
    }
)
_LOOP_SUMMARY_RLM_GROUP_FIELDS = _LOOP_SUMMARY_COMMON_GROUP_FIELDS | frozenset(
    {
        "accuracy",
        "correct_cases",
        "mean_reported_calls",
        "mean_vllm_successful_requests",
    }
)
_LOOP_SUMMARY_HALO_GROUP_FIELDS = _LOOP_SUMMARY_COMMON_GROUP_FIELDS | frozenset(
    {
        "json_valid_rate",
        "mean_citation_precision",
        "mean_count_accuracy",
        "mean_family_f1",
    }
)
_LOOP_UPSTREAMS = {
    "babilong_revision": "ee0d588794c7ac098062ee0d247c733d62e94fe2",
    "babilong_source": "RMT-team/babilong",
    "halo_revision": "b7f8509745d67b499b4e80efe20ea37c03426a74",
    "halo_source": "context-labs/HALO",
    "halo_version": "0.3.5",
    "openai_agents_version": "0.14.7",
    "openai_version": "2.32.0",
    "pyarrow_version": "21.0.0",
    "rlm_revision": "0b45df99c43fb3844a3b796a15d13c0f9d07afd8",
    "rlm_source": "alexzhang13/rlm",
    "rlm_version": "0.1.3",
}
_LOOP_WORKER_IMAGE = (
    "nvcr.io/nvidia/vllm:26.07-py3@"
    "sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268"
)
_LOOP_CONTEXT_LENGTHS = frozenset({"4k", "8k", "16k", "32k", "64k", "128k"})
_LOOP_TASKS = frozenset(f"qa{index}" for index in range(1, 11))

_HARBOR_ZERO_INFRASTRUCTURE_FIELDS = (
    "harbor_process_failures",
    "harbor_timeouts",
    "cleanup_failures",
    "native_image_admission_failures",
    "built_image_cleanup_failures",
    "network_admission_failures",
    "image_pair_mismatches",
)
_HARBOR_TRIAL_SECURITY_FIELDS = (
    "main_image_arm64",
    "relay_image_arm64",
    "built_image_cleanup_succeeded",
    "setup_relay_rejected",
    "agent_relay_passed",
    "wrong_auth_rejected",
    "other_loopback_rejected",
    "gost_rejected",
    "dns_rejected",
    "gateway_rejected",
    "public_rejected",
    "capabilities_dropped",
    "paired_image_match",
    "cleanup_succeeded",
)

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private-key", re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")),
    ("huggingface-token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    (
        "github-token",
        re.compile(r"\b(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,})\b"),
    ),
    (
        "api-token",
        re.compile(
            r"\b(?:sk-(?:proj-|ant-(?:api[0-9]+-)?)?[A-Za-z0-9_-]{20,}|"
            r"nvapi-[A-Za-z0-9_-]{16,}|glpat-[A-Za-z0-9_-]{16,})\b"
        ),
    ),
    ("aws-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("google-key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b")),
    (
        "package-token",
        re.compile(r"\b(?:sk_live_[0-9A-Za-z]{16,}|npm_[A-Za-z0-9]{20,}|pypi-AgEI[A-Za-z0-9_-]{20,})\b"),
    ),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    ),
    (
        "authorization",
        re.compile(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{12,}"),
    ),
    (
        "credential-url",
        re.compile(
            r"(?i)(?:https?|s3)://[^\s/:@]+:[^\s/@]+@|"
            r"[?&](?:x-amz-signature|signature|sig|token|key)=[A-Za-z0-9%._~+/-]{12,}"
        ),
    ),
    (
        "secret-assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
            r"client[_-]?secret|password|passwd|secret|authorization|cookie)\b"
            r"\s*[:=]\s*[\"']?(?!null\b|none\b|false\b|true\b|<redacted>|"
            r"\[redacted\])[^\s\"',;]{8,}"
        ),
    ),
)

_FORBIDDEN_OUTPUT_KEYS = {
    "content",
    "reasoning",
    "tool_calls",
    "prompt",
    "prompts",
    "messages",
    "response",
    "completion",
    "output",
    "output_text",
    "raw",
    "request_id",
    "request_tag",
    "token_ids",
    "transcription",
    "normalized_transcription",
    "image_url",
    "audio_url",
    "media",
    "stdout",
    "stderr",
    "log",
    "logs",
    "args",
    "argv",
    "command",
    "environment",
    "env",
    "error",
    "authorization",
    "cookie",
    "password",
    "secret",
    "api_key",
    "access_token",
    "refresh_token",
    "client_secret",
}

_KNOWN_EVENTS = {
    "artifact_validation_complete",
    "artifact_verified",
    "case_complete",
    "case_failed",
    "case_skipped_adapter_unimplemented",
    "case_skipped_context_limit",
    "case_skipped_unsupported",
    "case_start",
    "first_request_complete",
    "llamacpp_spec_decode_metrics_snapshot",
    "measurement_complete",
    "measurement_started",
    "measurement_annotation",
    "model_loaded",
    "perplexity_complete",
    "preflight_complete",
    "process_cleanup",
    "process_start",
    "request_complete",
    "run_aborted",
    "run_complete",
    "run_start",
    "server_kept",
    "server_ready",
    "server_stopped",
    "sglang_spec_decode_metrics_snapshot",
    "vllm_spec_decode_metrics_snapshot",
    "worker_cleanup",
    "worker_start",
}

_REQUEST_NUMERIC_FIELDS = {
    "audio_duration_s",
    "batch_size",
    "block_generation_blocks_per_s",
    "block_generation_output_tps",
    "cache_pair_index",
    "cache_step_ordinal",
    "cache_prefix_target_words",
    "candidate_count",
    "cached_prompt_tokens",
    "character_edit_distance",
    "character_error_rate",
    "completion_tokens",
    "decode_s",
    "decode_tps",
    "dimension",
    "elapsed_s",
    "emission_events",
    "image_latency_s",
    "items_per_s",
    "load_s",
    "max_output_tokens",
    "mean_block_generation_latency_s",
    "nfe",
    "nfe_per_block",
    "nfe_per_output_token",
    "nfe_per_s",
    "normalized_output_length",
    "output_blocks",
    "output_tokens",
    "output_tokens_per_nfe",
    "output_tps",
    "pairs_per_s",
    "prompt_tokens",
    "relevant_similarity",
    "relevant_text_latency_s",
    "repetition",
    "seed",
    "server_prompt_s",
    "server_cached_prompt_tokens",
    "server_decode_s",
    "server_decode_tokens",
    "server_prompt_tokens",
    "similarity_margin",
    "temperature",
    "top_index",
    "ttft_s",
    "unrelated_similarity",
    "unrelated_text_latency_s",
    "wall_time_s",
}
_REQUEST_NULLABLE_NUMERIC_FIELDS = {"reasoning_tokens"}
_REQUEST_BOOLEAN_FIELDS = {
    "finite",
    "transcription_exact",
}
_REQUEST_NUMERIC_SEQUENCE_FIELDS = {"norms", "ranking", "scores"}
_REQUEST_STRING_FIELDS = {
    "block_generation_metric_source",
    "cache_condition",
    "cache_prompt_control",
    "cache_profile_mode",
    "decode_metric_source",
    "finish_reason",
}
_REQUEST_DROPPED_FIELDS = {
    "content",
    "expected_transcription_sha256",
    "normalized_transcription_sha256",
    "output_sha256",
    "reasoning",
    "request_id",
    "response_model",
    "started_at_ns",
    "token_ids_sha256",
    "tool_calls",
}

# The llama.cpp prefix-KV experiment is intentionally narrower than a generic
# serving result.  Keeping this contract separate prevents an otherwise
# allowlisted metric (for example an audio score) from quietly becoming part of
# the cache evidence corpus.  ``server_*`` fields are request-scoped final SSE
# measurements; ``prometheus_global_*`` fields are only global Prometheus
# diagnostics.
_PREFIX_CACHE_RAW_RESULT_FIELDS = frozenset(
    {
        "cache_condition",
        "cache_pair_index",
        "cache_prefix_target_words",
        "cache_profile_mode",
        "cache_prompt_control",
        "cache_step_ordinal",
        "cached_prompt_tokens",
        "completion_tokens",
        "decode_metric_source",
        "decode_s",
        "decode_tps",
        "elapsed_s",
        "emission_events",
        "finish_reason",
        "prometheus_global_cached_prompt_tokens",
        "prometheus_global_decode_s",
        "prometheus_global_decode_tokens",
        "prometheus_global_prompt_s",
        "prometheus_global_prompt_tokens",
        "output_tps",
        "prompt_tokens",
        "reasoning_tokens",
        "server_cached_prompt_tokens",
        "server_decode_s",
        "server_decode_tokens",
        "server_prompt_s",
        "server_prompt_tokens",
        "ttft_s",
    }
)
_PREFIX_CACHE_RAW_RESULT_INTEGER_FIELDS = frozenset(
    {
        "cache_pair_index",
        "cache_prefix_target_words",
        "cache_step_ordinal",
        "cached_prompt_tokens",
        "completion_tokens",
        "emission_events",
        "prometheus_global_cached_prompt_tokens",
        "prometheus_global_decode_tokens",
        "prometheus_global_prompt_tokens",
        "prompt_tokens",
        "server_cached_prompt_tokens",
        "server_decode_tokens",
        "server_prompt_tokens",
    }
)
_PREFIX_CACHE_RAW_RESULT_POSITIVE_INTEGER_FIELDS = frozenset(
    {
        "cache_pair_index",
        "cache_prefix_target_words",
        "cache_step_ordinal",
        "completion_tokens",
        "emission_events",
        "prompt_tokens",
        "server_decode_tokens",
    }
)

_AGENTIC_RESULT_FIELDS = frozenset(
    {
        "completion_tokens",
        "decode_metric_source",
        "decode_s",
        "decode_tps",
        "elapsed_s",
        "emission_events",
        "expected_tool_calls",
        "failure_code",
        "final_answer_correct",
        "final_answer_emitted",
        "finish_reason",
        "first_turn_ttft_s",
        "length_terminated_turns",
        "malformed_tool_calls",
        "max_output_tokens",
        "max_turns",
        "output_tps",
        "passed",
        "prompt_tokens",
        "recovery_required",
        "recovery_succeeded",
        "request_elapsed_s",
        "scenario_id",
        "schema_version",
        "tool_calls_executed",
        "tool_calls_requested",
        "tool_calls_succeeded",
        "tool_errors",
        "tool_sequence_correct",
        "ttft_s",
        "turn_limit_reached",
        "turns_used",
        "unknown_tool_calls",
        "variant",
        "wall_s",
    }
)
_AGENTIC_FAILURE_CODES = frozenset(
    {
        "final_answer",
        "malformed_tool_call",
        "missing_final",
        "output_limit",
        "tool_call_limit",
        "tool_sequence",
        "turn_limit",
        "unknown_tool",
    }
)
_AGENTIC_FINISH_REASONS = frozenset(
    {
        "content_filter",
        "length",
        "other",
        "stop",
        "tool_call_limit",
        "tool_calls",
        "turn_limit",
    }
)
_AGENTIC_EXPECTED_CALLS = {
    "agentic-no-tool": 0,
    "agentic-select-and-call": 1,
    "agentic-tool-error-recovery": 2,
    "agentic-two-hop": 2,
}
_AGENTIC_TOOL_COUNTS = {
    "agentic-no-tool": (0, 0, 0, 0, 1),
    "agentic-select-and-call": (1, 1, 1, 0, 2),
    "agentic-tool-error-recovery": (2, 2, 1, 1, 3),
    "agentic-two-hop": (2, 2, 2, 0, 3),
}
_AGENTIC_SUITE_FIELDS = frozenset({"cases", "id", "schema_version"})
_AGENTIC_SUITE_DESCRIPTION = (
    "Deterministic multi-turn tool selection, abstention, dependency, and "
    "recovery checks with scalar-only results."
)
_AGENTIC_SUITE_CASE_FIELDS = frozenset(
    {
        "case_id",
        "concurrency",
        "id",
        "kind",
        "max_output_tokens",
        "max_turns",
        "prompt_repetitions",
        "repetitions",
        "requires",
        "temperature",
        "warmups",
    }
)
_AGENTIC_PROJECTED_RESULT_FIELDS = (
    _AGENTIC_RESULT_FIELDS - {"emission_events", "output_tps"}
) | {"emission_event_count"}
_AGENTIC_SAMPLE_FIELDS = _AGENTIC_PROJECTED_RESULT_FIELDS | {
    "burst_elapsed_s",
    "case_attempt",
    "case_id",
    "case_sample_index",
    "kind",
    "repetition",
    "sample_index",
    "sample_type",
    "selected_attempt",
    "validation_passed",
}

# The memory-operation protocol is an exact, self-contained evidence family.
# Its Graphiti-inspired resolver cases deliberately retain only confusion
# labels and exact-set correctness; the synthetic extension retains only
# transaction-field correctness.  Neither projection can carry facts, paths,
# values, nonces, prompts, model output, hidden reasoning, tool payloads, or
# request identifiers.
_MEMORY_SUITE_DESCRIPTION = (
    "Graphiti-style edge resolution followed by explicitly synthetic "
    "MemFS/transaction extension cases; exact JSON grading and scalar-only "
    "results."
)
_MEMORY_SUITE_FIELDS = frozenset(
    {"cases", "id", "protocol_digest", "schema_version"}
)
_MEMORY_SUITE_CASE_FIELDS = frozenset(
    {
        "case_id",
        "concurrency",
        "id",
        "kind",
        "max_output_tokens",
        "max_turns",
        "prompt_repetitions",
        "repetitions",
        "requires",
        "temperature",
        "warmups",
    }
)
_MEMORY_RESULT_FIELDS = frozenset(
    {
        "action_correct",
        "completion_tokens",
        "contradicted_facts_correct",
        "cached_prompt_tokens",
        "decode_metric_source",
        "decode_s",
        "decode_tps",
        "duplicate_facts_correct",
        "elapsed_s",
        "emission_events",
        "evidence_correct",
        "expected_resolver_action",
        "failure_code",
        "finish_reason",
        "graphiti_resolver_case",
        "injection_refusal_required",
        "injection_refusal_succeeded",
        "json_object_emitted",
        "max_output_tokens",
        "mutation_expected",
        "mutation_selected",
        "output_tps",
        "passed",
        "path_correct",
        "prompt_tokens",
        "prompt_cache_disabled",
        "protected_value_emitted",
        "reason_correct",
        "reasoning_tokens",
        "resolver_decision_correct",
        "scenario_id",
        "schema_valid",
        "schema_version",
        "secret_refusal_required",
        "secret_refusal_succeeded",
        "selected_resolver_action",
        "server_cached_prompt_tokens",
        "server_decode_s",
        "server_decode_tokens",
        "server_prompt_s",
        "server_prompt_tokens",
        "synthetic_extension_case",
        "target_correct",
        "tier_correct",
        "ttft_s",
        "unexpected_field_count",
        "unexpected_tool_call_count",
        "valid_from_correct",
        "valid_to_correct",
        "value_correct",
        "variant",
    }
)
_MEMORY_PROJECTED_RESULT_FIELDS = _MEMORY_RESULT_FIELDS - {
    "emission_events",
    "output_tps",
}
_MEMORY_SAMPLE_FIELDS = _MEMORY_PROJECTED_RESULT_FIELDS | {
    "burst_elapsed_s",
    "case_attempt",
    "case_id",
    "case_sample_index",
    "kind",
    "repetition",
    "sample_index",
    "sample_type",
    "selected_attempt",
    "validation_passed",
}

# Evidence v1 is intentionally limited to the frozen first memory panel.  The
# fifth entry is the separately-labelled exploratory thinking profile.  The
# pinned llama.cpp build does not expose a trustworthy reasoning-token
# partition for either policy, so reasoning usage remains unavailable/null in
# every v1 sample.
_MEMORY_PANEL_MODELS: dict[str, dict[str, Any]] = {
    "laguna-xs21-33b-a3b-q4-k-m-llamacpp": {
        "architecture": "laguna",
        "backend": "llamacpp",
        "estimated_ram_gib": 48.0,
        "id": "laguna-xs21-33b-a3b-q4-k-m-llamacpp",
        "lifecycle": "subprocess",
        "max_context": MEMORY_OPERATION_CONTEXT_TOKENS,
        "memory_thinking_enabled": False,
        "native_context": MEMORY_OPERATION_CONTEXT_TOKENS,
        "quantization": "q4_k_m",
        "revision": "1a37c0a5fb8c7a18e6106decb6be6327d1b63fa6",
        "runtime_parallel": 1,
        "source": "poolside/Laguna-XS-2.1-GGUF",
        "startup_timeout_s": 600,
        "support_status": "spark_other_backend",
        "tasks": ["chat", "json", "tools"],
    },
    "laguna-s21-118b-a8b-ud-q4-k-xl-llamacpp": {
        "architecture": "laguna",
        "backend": "llamacpp",
        "estimated_ram_gib": 96.0,
        "id": "laguna-s21-118b-a8b-ud-q4-k-xl-llamacpp",
        "lifecycle": "subprocess",
        "max_context": MEMORY_OPERATION_CONTEXT_TOKENS,
        "memory_thinking_enabled": False,
        "native_context": MEMORY_OPERATION_CONTEXT_TOKENS,
        "quantization": "ud-q4_k_xl",
        "revision": "750f92f90cf54159c4d7a610cb7b3e74498e75c6",
        "runtime_parallel": 1,
        "source": "unsloth/Laguna-S-2.1-GGUF",
        "startup_timeout_s": 1200,
        "support_status": "spark_other_backend",
        "tasks": ["chat", "json", "tools"],
        "weight_file_count": 3,
        "weight_size_bytes": 73395172000,
    },
    "ornith15-35b-a3b-q4-k-m-llamacpp": {
        "architecture": "qwen35moe",
        "backend": "llamacpp",
        "estimated_ram_gib": 56.0,
        "id": "ornith15-35b-a3b-q4-k-m-llamacpp",
        "lifecycle": "subprocess",
        "max_context": MEMORY_OPERATION_CONTEXT_TOKENS,
        "memory_thinking_enabled": False,
        "native_context": MEMORY_OPERATION_CONTEXT_TOKENS,
        "quantization": "q4_k_m",
        "revision": "12393612fd4f730ff5aadc23e9b8f9648aa49ceb",
        "runtime_parallel": 1,
        "source": "ornith-ai/Ornith-1.5-35B-A3B-GGUF",
        "startup_timeout_s": 600,
        "support_status": "spark_other_backend",
        "tasks": ["chat", "json", "tools"],
    },
    "ornith15-35b-a3b-q4-k-m-llamacpp-thinking": {
        "architecture": "qwen35moe",
        "backend": "llamacpp",
        "estimated_ram_gib": 56.0,
        "id": "ornith15-35b-a3b-q4-k-m-llamacpp-thinking",
        "lifecycle": "subprocess",
        "max_context": MEMORY_OPERATION_CONTEXT_TOKENS,
        "memory_thinking_enabled": True,
        "native_context": MEMORY_OPERATION_CONTEXT_TOKENS,
        "quantization": "q4_k_m",
        "revision": "12393612fd4f730ff5aadc23e9b8f9648aa49ceb",
        "runtime_parallel": 1,
        "source": "ornith-ai/Ornith-1.5-35B-A3B-GGUF",
        "startup_timeout_s": 600,
        "support_status": "spark_other_backend",
        "tasks": ["chat", "json", "tools", "thinking"],
    },
    "qwen36-35b-a3b-ud-q4-k-xl-llamacpp-32k": {
        "architecture": "qwen35moe",
        "backend": "llamacpp",
        "estimated_ram_gib": 56.0,
        "id": "qwen36-35b-a3b-ud-q4-k-xl-llamacpp-32k",
        "lifecycle": "subprocess",
        "max_context": MEMORY_OPERATION_CONTEXT_TOKENS,
        "memory_thinking_enabled": False,
        "native_context": MEMORY_OPERATION_CONTEXT_TOKENS,
        "quantization": "ud-q4_k_xl",
        "revision": "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
        "runtime_parallel": 1,
        "source": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        "startup_timeout_s": 600,
        "support_status": "spark_other_backend",
        "tasks": ["chat", "json", "tools"],
    },
}

_MEMORY_PANEL_MODEL_ARTIFACTS: dict[str, tuple[dict[str, Any], ...]] = {
    "laguna-xs21-33b-a3b-q4-k-m-llamacpp": (
        {
            "role": "model",
            "sha256": "1ac7079101fca5a6df8c5a7523a3c30ea7d1c0e4b1258090e7d6d4039287f6cb",
            "size_bytes": 20274300032,
            "source": "poolside/Laguna-XS-2.1-GGUF",
            "revision": "1a37c0a5fb8c7a18e6106decb6be6327d1b63fa6",
            "target": "Laguna-XS-2.1-Q4_K_M.gguf",
        },
    ),
    "laguna-s21-118b-a8b-ud-q4-k-xl-llamacpp": (
        {
            "role": "model_shard_1",
            "sha256": "0cfaf46917260d253773e5e2fab64329fa5c9c60fdf0db0f59f31205b5f5dd32",
            "size_bytes": 3683648,
            "source": "unsloth/Laguna-S-2.1-GGUF",
            "revision": "750f92f90cf54159c4d7a610cb7b3e74498e75c6",
            "target": "Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003.gguf",
        },
        {
            "role": "model_shard_2",
            "sha256": "2296102462b02edca70163121ac62bacf7a82078c0eafc91625c8822850769bf",
            "size_bytes": 49971821312,
            "source": "unsloth/Laguna-S-2.1-GGUF",
            "revision": "750f92f90cf54159c4d7a610cb7b3e74498e75c6",
            "target": "Laguna-S-2.1-UD-Q4_K_XL-00002-of-00003.gguf",
        },
        {
            "role": "model_shard_3",
            "sha256": "9150e2338f7690af29685b6a2ca621a8fda7ecf9724678266c4b04b7c6dd0ef3",
            "size_bytes": 23419667040,
            "source": "unsloth/Laguna-S-2.1-GGUF",
            "revision": "750f92f90cf54159c4d7a610cb7b3e74498e75c6",
            "target": "Laguna-S-2.1-UD-Q4_K_XL-00003-of-00003.gguf",
        },
    ),
    "ornith15-35b-a3b-q4-k-m-llamacpp": (
        {
            "role": "model",
            "sha256": "42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f",
            "size_bytes": 21713463040,
            "source": "ornith-ai/Ornith-1.5-35B-A3B-GGUF",
            "revision": "12393612fd4f730ff5aadc23e9b8f9648aa49ceb",
            "target": "Ornith-1.5-35B-Q4_K_M.gguf",
        },
    ),
    "ornith15-35b-a3b-q4-k-m-llamacpp-thinking": (
        {
            "role": "model",
            "sha256": "42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f",
            "size_bytes": 21713463040,
            "source": "ornith-ai/Ornith-1.5-35B-A3B-GGUF",
            "revision": "12393612fd4f730ff5aadc23e9b8f9648aa49ceb",
            "target": "Ornith-1.5-35B-Q4_K_M.gguf",
        },
    ),
    "qwen36-35b-a3b-ud-q4-k-xl-llamacpp-32k": (
        {
            "role": "model",
            "sha256": "55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1",
            "size_bytes": 22853663008,
            "source": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
            "revision": "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
            "target": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        },
    ),
}
_MEMORY_FAILURE_CODES = frozenset(
    {
        "invalid_json",
        "operation_mismatch",
        "output_limit",
        "protected_value",
        "schema_mismatch",
        "unexpected_tool_call",
    }
)
_MEMORY_FINISH_REASONS = frozenset(
    {"content_filter", "length", "other", "stop", "tool_calls"}
)
_MEMORY_DECODE_SOURCES = frozenset(
    {"client_estimate", "other", "server_reported_eval_duration"}
)
_MEMORY_RESOLVER_ACTIONS = frozenset(
    {"CREATE_AND_INVALIDATE", "CREATE_FACT", "INVALID", "REUSE_FACT"}
)
_MEMORY_EXPECTED_RESOLVER_ACTION = {
    "graphiti-reuse-fact": "REUSE_FACT",
    "graphiti-invalidate-fact": "CREATE_AND_INVALIDATE",
    "graphiti-create-fact": "CREATE_FACT",
}
_MEMORY_MUTATION_EXPECTED = frozenset(
    {
        "memory-add",
        "memory-delete",
        "memory-supersede",
        "memory-temporal-invalidate",
        "memory-tier-placement",
    }
)
_MEMORY_SUMMARY_FIELDS = frozenset(
    {
        "graphiti_resolver",
        "json_objects_emitted",
        "operation_accuracy",
        "operations",
        "operations_correct",
        "prompt_cache_disabled_requests",
        "protected_value_emissions",
        "schema_valid",
        "schema_version",
        "synthetic_extension",
        "total_completion_tokens",
        "total_reasoning_tokens",
        "total_request_elapsed_s",
        "total_prompt_tokens",
        "total_server_decode_s",
        "total_server_prompt_s",
        "unexpected_tool_calls",
        "zero_cached_prompt_requests",
    }
)
_MEMORY_GRAPHITI_SUMMARY_FIELDS = frozenset(
    {
        "accuracy",
        "confusion",
        "contradicted_sets_correct",
        "correct",
        "duplicate_sets_correct",
        "operations",
    }
)
_MEMORY_EXTENSION_SUMMARY_FIELDS = frozenset(
    {
        "accuracy",
        "action_correct",
        "correct",
        "evidence_correct",
        "field_checks_applicable",
        "injection_refusals_required",
        "injection_refusals_succeeded",
        "mutations_expected",
        "mutations_selected",
        "operations",
        "path_correct",
        "reason_correct",
        "secret_refusals_required",
        "secret_refusals_succeeded",
        "target_correct",
        "tier_correct",
        "valid_from_correct",
        "valid_to_correct",
        "value_correct",
    }
)
_VALIDATION_FIELDS = {
    "expected_answer",
    "expected_transcription",
    "extracted_answer",
    "normalized_transcription",
    "passed",
    "quality_category",
    "quality_item_id",
    "reason",
}

_CASE_FIELDS = {
    "agentic_expected_tool_calls",
    "agentic_final_answers_correct",
    "agentic_final_answers_emitted",
    "agentic_length_terminated_turns",
    "agentic_malformed_tool_calls",
    "agentic_max_output_tokens_per_turn",
    "agentic_max_turns",
    "agentic_model_requests",
    "agentic_model_requests_per_s",
    "agentic_recoveries_required",
    "agentic_recoveries_succeeded",
    "agentic_sampled_energy_j_per_solved_task",
    "agentic_task_success_rate",
    "agentic_tasks",
    "agentic_tasks_per_s",
    "agentic_tasks_succeeded",
    "agentic_tasks_succeeded_per_sampled_joule",
    "agentic_tool_calls_executed",
    "agentic_tool_calls_requested",
    "agentic_tool_calls_succeeded",
    "agentic_tool_errors",
    "agentic_tool_sequences_correct",
    "agentic_turn_limit_hits",
    "agentic_unknown_tool_calls",
    "aggregate_block_generation_blocks_per_s",
    "aggregate_block_generation_output_tps",
    "aggregate_output_tps",
    "aggregate_rerank_pairs_s",
    "audio_duration_s",
    "case_id",
    "completion_tokens",
    "concurrency",
    "decode_estimate_one_token_chunks",
    "decode_metric_source",
    "elapsed_s",
    "embedding_batch_size",
    "embedding_dimension",
    "embeddings_finite",
    "exact_matches",
    "graphiti_contradicted_sets_correct",
    "graphiti_duplicate_sets_correct",
    "graphiti_resolver_accuracy",
    "graphiti_resolver_confusion",
    "graphiti_resolver_correct",
    "graphiti_resolver_operations",
    "kind",
    "measured_wall_time_s",
    "measurement_annotation_count",
    "measurement_valid",
    "median_approximate_prefill_tps",
    "median_agentic_first_turn_ttft_s",
    "median_agentic_model_request_sum_s",
    "median_agentic_task_wall_s",
    "median_agentic_turns_used",
    "median_block_generation_blocks_per_s",
    "median_block_generation_output_tps",
    "median_character_edit_distance",
    "median_character_error_rate",
    "median_decode_tps",
    "median_e2e_s",
    "median_elapsed_s",
    "median_embedding_items_s",
    "median_estimated_decode_tps",
    "median_image_embedding_latency_s",
    "median_mean_block_generation_latency_s",
    "median_normalized_output_length",
    "median_output_tps",
    "median_prefill_tps",
    "median_realtime_factor",
    "median_relevant_similarity",
    "median_relevant_text_embedding_latency_s",
    "median_rerank_pairs_s",
    "median_similarity_margin",
    "median_ttft_s",
    "median_unrelated_similarity",
    "median_unrelated_text_embedding_latency_s",
    "memory_action_correct",
    "memory_evidence_correct",
    "memory_field_checks_applicable",
    "memory_injection_refusals_required",
    "memory_injection_refusals_succeeded",
    "memory_json_objects_emitted",
    "memory_operation_accuracy",
    "memory_operations",
    "memory_operations_correct",
    "memory_path_correct",
    "memory_prompt_cache_disabled_requests",
    "memory_protected_value_emissions",
    "memory_schema_valid",
    "memory_secret_refusals_required",
    "memory_secret_refusals_succeeded",
    "memory_target_correct",
    "memory_tier_correct",
    "memory_total_completion_tokens",
    "memory_total_prompt_tokens",
    "memory_total_reasoning_tokens",
    "memory_total_server_decode_s",
    "memory_total_server_prompt_s",
    "memory_unexpected_tool_calls",
    "memory_valid_from_correct",
    "memory_valid_to_correct",
    "memory_value_correct",
    "memory_reason_correct",
    "memory_mutations_expected",
    "memory_mutations_selected",
    "memory_zero_cached_prompt_requests",
    "metric_source",
    "multimodal_embedding_validation_passed",
    "multimodal_embeddings_finite",
    "nfe",
    "nfe_per_block",
    "nfe_per_output_token",
    "nfe_per_s",
    "output_blocks",
    "output_tokens",
    "output_tokens_per_nfe",
    "output_tokens_per_sampled_joule",
    "outputs_stable",
    "p95_approximate_prefill_tps",
    "p95_e2e_s",
    "p95_prefill_tps",
    "p95_ttft_s",
    "prefill_metric_source",
    "prefix_cache",
    "prompt_tokens",
    "prompt_tokens_per_sampled_joule",
    "quality_accuracy",
    "quality_accuracy_by_category",
    "quality_correct",
    "quality_items",
    "quality_scored_items",
    "quality_total_completion_tokens",
    "quality_total_prompt_tokens",
    "quality_total_reasoning_tokens",
    "reasoning_tokens",
    "quality_total_request_latency_s",
    "request_tps",
    "requests",
    "rerank_candidates_per_request",
    "rerank_pairs",
    "rerank_ranking_stable",
    "rerank_scores_finite",
    "rerank_top_index",
    "rerank_validation_passed",
    "synthetic_memory_extension_operations",
    "telemetry",
    "validation_passed",
    "warmups",
}
_CASE_DROPPED_FIELDS = {"attempt_id", "measurement_annotations", "output_sha256"}
_CASE_STRING_FIELDS = {
    "case_id",
    "decode_metric_source",
    "kind",
    "metric_source",
    "prefill_metric_source",
}
_CASE_BOOLEAN_FIELDS = {
    "decode_estimate_one_token_chunks",
    "embeddings_finite",
    "measurement_valid",
    "multimodal_embedding_validation_passed",
    "multimodal_embeddings_finite",
    "outputs_stable",
    "rerank_ranking_stable",
    "rerank_scores_finite",
    "rerank_validation_passed",
    "validation_passed",
}
_CASE_OBJECT_FIELDS = {
    "graphiti_resolver_confusion",
    "prefix_cache",
    "quality_accuracy_by_category",
    "telemetry",
}
_CASE_NULLABLE_FIELDS = {
    "agentic_sampled_energy_j_per_solved_task",
    "aggregate_output_tps",
    "case_id",
    "decode_estimate_one_token_chunks",
    "decode_metric_source",
    "median_approximate_prefill_tps",
    "median_decode_tps",
    "median_e2e_s",
    "median_elapsed_s",
    "median_estimated_decode_tps",
    "median_output_tps",
    "median_realtime_factor",
    "median_ttft_s",
    "memory_total_reasoning_tokens",
    "p95_approximate_prefill_tps",
    "p95_e2e_s",
    "p95_prefill_tps",
    "p95_ttft_s",
    "quality_total_reasoning_tokens",
    "reasoning_tokens",
    "request_tps",
    "validation_passed",
}
_AGENTIC_CASE_FIELDS = frozenset(
    {
        "agentic_expected_tool_calls",
        "agentic_final_answers_correct",
        "agentic_final_answers_emitted",
        "agentic_length_terminated_turns",
        "agentic_malformed_tool_calls",
        "agentic_max_output_tokens_per_turn",
        "agentic_max_turns",
        "agentic_model_requests",
        "agentic_model_requests_per_s",
        "agentic_recoveries_required",
        "agentic_recoveries_succeeded",
        "agentic_sampled_energy_j_per_solved_task",
        "agentic_task_success_rate",
        "agentic_tasks",
        "agentic_tasks_per_s",
        "agentic_tasks_succeeded",
        "agentic_tasks_succeeded_per_sampled_joule",
        "agentic_tool_calls_executed",
        "agentic_tool_calls_requested",
        "agentic_tool_calls_succeeded",
        "agentic_tool_errors",
        "agentic_tool_sequences_correct",
        "agentic_turn_limit_hits",
        "agentic_unknown_tool_calls",
        "median_agentic_first_turn_ttft_s",
        "median_agentic_model_request_sum_s",
        "median_agentic_task_wall_s",
        "median_agentic_turns_used",
    }
)
_AGENTIC_CASE_REQUIRED_FIELDS = _AGENTIC_CASE_FIELDS - {
    "agentic_sampled_energy_j_per_solved_task",
    "agentic_tasks_succeeded_per_sampled_joule",
}
_AGENTIC_CASE_ENERGY_FIELDS = frozenset(
    {
        "agentic_sampled_energy_j_per_solved_task",
        "agentic_tasks_succeeded_per_sampled_joule",
    }
)

_MEMORY_CASE_METRIC_FIELDS = frozenset(
    {
        "graphiti_resolver_operations",
        "memory_action_correct",
        "memory_evidence_correct",
        "memory_field_checks_applicable",
        "memory_injection_refusals_required",
        "memory_injection_refusals_succeeded",
        "memory_json_objects_emitted",
        "memory_operation_accuracy",
        "memory_operations",
        "memory_operations_correct",
        "memory_path_correct",
        "memory_prompt_cache_disabled_requests",
        "memory_protected_value_emissions",
        "memory_schema_valid",
        "memory_secret_refusals_required",
        "memory_secret_refusals_succeeded",
        "memory_target_correct",
        "memory_tier_correct",
        "memory_total_completion_tokens",
        "memory_total_prompt_tokens",
        "memory_total_reasoning_tokens",
        "memory_total_server_decode_s",
        "memory_total_server_prompt_s",
        "memory_unexpected_tool_calls",
        "memory_valid_from_correct",
        "memory_valid_to_correct",
        "memory_value_correct",
        "memory_reason_correct",
        "memory_mutations_expected",
        "memory_mutations_selected",
        "memory_zero_cached_prompt_requests",
        "synthetic_memory_extension_operations",
    }
)
_GRAPHITI_CASE_METRIC_FIELDS = frozenset(
    {
        "graphiti_contradicted_sets_correct",
        "graphiti_duplicate_sets_correct",
        "graphiti_resolver_accuracy",
        "graphiti_resolver_confusion",
        "graphiti_resolver_correct",
    }
)
_MEMORY_PROJECTED_CASE_BASE_FIELDS = _MEMORY_CASE_METRIC_FIELDS | {
    "aggregate_output_tps",
    "case_id",
    "completion_tokens",
    "concurrency",
    "decode_estimate_one_token_chunks",
    "decode_metric_source",
    "elapsed_s",
    "kind",
    "measurement_annotation_count",
    "measurement_valid",
    "median_decode_tps",
    "median_e2e_s",
    "median_estimated_decode_tps",
    "median_ttft_s",
    "p95_e2e_s",
    "p95_ttft_s",
    "prompt_tokens",
    "reasoning_tokens",
    "request_tps",
    "requests",
    "validation_passed",
}
_MEMORY_CASE_OPTIONAL_TELEMETRY_FIELDS = frozenset({"telemetry"})
_MEMORY_CASE_OPTIONAL_ENERGY_FIELDS = frozenset(
    {"output_tokens_per_sampled_joule"}
)
_AGENTIC_PROJECTED_CASE_BASE_FIELDS = _AGENTIC_CASE_REQUIRED_FIELDS | {
    "aggregate_output_tps",
    "case_id",
    "completion_tokens",
    "concurrency",
    "decode_estimate_one_token_chunks",
    "decode_metric_source",
    "elapsed_s",
    "kind",
    "measurement_annotation_count",
    "measurement_valid",
    "median_decode_tps",
    "median_e2e_s",
    "median_estimated_decode_tps",
    "median_ttft_s",
    "p95_e2e_s",
    "p95_ttft_s",
    "prompt_tokens",
    "request_tps",
    "requests",
    "telemetry",
    "validation_passed",
}
_TELEMETRY_SUMMARY_FIELDS = {
    "average_gpu_util_pct",
    "average_power_w",
    "energy_j",
    "gpu_error_samples",
    "gpu_power_missing_samples",
    "gpu_power_samples",
    "minimum_memavailable_gib",
    "peak_power_w",
    "peak_sm_clock_mhz",
    "peak_temperature_c",
    "sampled_energy_intervals",
    "sampled_energy_j",
    "samples",
}

_SUMMARY_KEYS = {
    "artifact_validation",
    "artifact_validation_telemetry",
    "artifact_verification",
    "artifacts",
    "cases",
    "cleanup_proof",
    "completed_cases",
    "configuration",
    "context_limited_cases",
    "error",
    "failed_cases",
    "first_request_after_start",
    "first_request_telemetry",
    "input_prepare_time_s",
    "llamacpp_dflash_evidence",
    "llamacpp_mtp_evidence",
    "load_time_s",
    "measurement_annotations",
    "measurement_invalid_cases",
    "memory",
    "memory_operation_summary",
    "metrics",
    "model",
    "run_completion_status",
    "run_dir",
    "run_error",
    "runtime",
    "schema_version",
    "shutdown_telemetry",
    "speculative_decoding",
    "startup_measurement_annotations",
    "startup_measurement_valid",
    "startup_safety_gates",
    "startup_telemetry",
    "status",
    "suite",
    "telemetry",
    "unimplemented_cases",
    "unsupported_cases",
    "validation_failed_cases",
}

_SAFE_SUMMARY_STRING_KEYS = {
    "accelerate",
    "architecture",
    "backend",
    "case_id",
    "cid",
    "cuda_runtime",
    "cuda_context_cleanup",
    "decode_metric_source",
    "device",
    "id",
    "image",
    "kind",
    "lifecycle",
    "method",
    "metric_source",
    "model_revision",
    "model_source",
    "prefill_metric_source",
    "python",
    "quantization",
    "revision",
    "scope",
    "source",
    "status",
    "suite",
    "support_status",
    "tensorrt_llm",
    "tool",
    "torch",
    "transformers",
}

_SAFE_SUMMARY_KEYS_BY_ROOT = {
    "artifact_validation": {
        "draft_model_sha256",
        "elapsed_s",
        "mmproj_sha256",
        "model_sha256",
        "model_shard_count",
        "model_shard_sha256s",
        "model_total_size_bytes",
        "runtime_binary_sha256",
    },
    "artifact_validation_telemetry": {
        "average_gpu_util_pct",
        "average_power_w",
        "gpu_error_samples",
        "gpu_power_missing_samples",
        "gpu_power_samples",
        "minimum_memavailable_gib",
        "peak_power_w",
        "peak_sm_clock_mhz",
        "peak_temperature_c",
        "sampled_energy_intervals",
        "sampled_energy_j",
        "samples",
    },
    "cleanup_proof": {
        "container",
        "cuda_context_cleanup",
        "kill_requested",
        "pid",
        "process_group_isolated",
        "process_reaped",
        "process_start_ticks",
        "process_started_at_ns",
        "returncode",
        "terminate_requested",
        "timed_out",
        "worker_cuda",
    },
    "configuration": {"chunks", "ctx_size", "timeout_s"},
    "first_request_telemetry": {
        "average_gpu_util_pct",
        "average_power_w",
        "gpu_error_samples",
        "gpu_power_missing_samples",
        "gpu_power_samples",
        "minimum_memavailable_gib",
        "peak_power_w",
        "peak_sm_clock_mhz",
        "peak_temperature_c",
        "sampled_energy_intervals",
        "sampled_energy_j",
        "samples",
    },
    "llamacpp_dflash_evidence": {
        "configured_max_draft_tokens",
        "contributing_lifetimes",
        "passed",
        "proposal_depth_validated_lifetimes",
        "reason",
        "requested",
        "validated_lifetimes",
    },
    "llamacpp_mtp_evidence": {
        "configured_max_draft_tokens",
        "contributing_lifetimes",
        "passed",
        "proposal_depth_validated_lifetimes",
        "reason",
        "requested",
        "validated_lifetimes",
    },
    "memory": {
        "device_free_bytes_after_generation",
        "device_total_bytes",
        "generation_peak_allocated_bytes",
        "generation_peak_reserved_bytes",
        "load_peak_allocated_bytes",
    },
    "metrics": {
        "chunks",
        "ctx_size",
        "metric_source",
        "perplexity",
        "uncertainty",
        "wall_time_s",
    },
    "runtime": {
        "accelerate",
        "cuda_runtime",
        "deterministic_greedy",
        "device",
        "python",
        "seed",
        "tensorrt_llm",
        "torch",
        "transformers",
    },
    "shutdown_telemetry": {
        "average_gpu_util_pct",
        "average_power_w",
        "energy_j",
        "gpu_error_samples",
        "gpu_power_missing_samples",
        "gpu_power_samples",
        "minimum_memavailable_gib",
        "peak_power_w",
        "peak_sm_clock_mhz",
        "peak_temperature_c",
        "sampled_energy_intervals",
        "sampled_energy_j",
        "samples",
    },
    "speculative_decoding": {
        "accepted_tokens_per_position",
        "configured_max_draft_tokens",
        "draft_acceptance_rate",
        "mean_accepted_length",
        "method",
        "num_accepted_tokens",
        "num_draft_tokens",
        "num_drafts",
        "proposal_depth",
        "requested",
        "scope",
        "snapshot_count",
        "source",
    },
    "startup_telemetry": {
        "average_gpu_util_pct",
        "average_power_w",
        "energy_j",
        "gpu_error_samples",
        "gpu_power_missing_samples",
        "gpu_power_samples",
        "minimum_memavailable_gib",
        "peak_power_w",
        "peak_sm_clock_mhz",
        "peak_temperature_c",
        "sampled_energy_intervals",
        "sampled_energy_j",
        "samples",
    },
    "telemetry": {
        "artifact_validation",
        "direct_diffusion_worker",
        "idle",
        "llamacpp_perplexity",
        "trtllm_direct_worker",
    },
}

_AGGREGATE_TELEMETRY_FIELDS = {
    "average_gpu_util_pct",
    "average_power_w",
    "gpu_error_samples",
    "gpu_power_missing_samples",
    "gpu_power_samples",
    "minimum_memavailable_gib",
    "peak_power_w",
    "peak_sm_clock_mhz",
    "peak_temperature_c",
    "sampled_energy_intervals",
    "sampled_energy_j",
    "samples",
}
_SAFE_SUMMARY_CHILD_KEYS = {
    ("cleanup_proof", "container"): {
        "cid",
        "cleanup_verified",
        "container_absent",
        "container_found",
        "ownership_verified",
        "remove_requested",
        "stop_requested",
    },
    ("cleanup_proof", "worker_cuda"): {
        "allocated_bytes_after_model_delete",
        "reserved_bytes_after_empty_cache",
    },
    ("speculative_decoding", "proposal_depth"): {
        "average_draft_tokens_per_draft",
        "configured_max_draft_tokens",
        "deepest_accepted_draft_depth",
        "deepest_accepted_position",
        "passed",
        "reason",
    },
    **{
        ("telemetry", phase): _AGGREGATE_TELEMETRY_FIELDS
        for phase in {
            "artifact_validation",
            "direct_diffusion_worker",
            "idle",
            "llamacpp_perplexity",
            "trtllm_direct_worker",
        }
    },
}

_SAFE_SUMMARY_OBJECT_FIELDS_BY_ROOT = {
    "cleanup_proof": {"container", "worker_cuda"},
    "speculative_decoding": {"accepted_tokens_per_position", "proposal_depth"},
    "telemetry": {
        "artifact_validation",
        "direct_diffusion_worker",
        "idle",
        "llamacpp_perplexity",
        "trtllm_direct_worker",
    },
}
_SAFE_SUMMARY_BOOLEAN_FIELDS = {
    "cleanup_verified",
    "container_absent",
    "container_found",
    "deterministic_greedy",
    "kill_requested",
    "ownership_verified",
    "passed",
    "process_group_isolated",
    "process_reaped",
    "remove_requested",
    "requested",
    "stop_requested",
    "terminate_requested",
    "timed_out",
}
_SAFE_SUMMARY_LIST_FIELDS = {"model_shard_sha256s"}
_SAFE_SUMMARY_NULLABLE_FIELDS = {
    ("artifact_validation", "draft_model_sha256"),
    ("artifact_validation", "mmproj_sha256"),
    ("first_request_telemetry", "sampled_energy_j"),
    ("llamacpp_dflash_evidence", "configured_max_draft_tokens"),
    ("llamacpp_dflash_evidence", "reason"),
    ("llamacpp_mtp_evidence", "reason"),
    ("shutdown_telemetry", "energy_j"),
    ("shutdown_telemetry", "sampled_energy_j"),
    ("speculative_decoding", "configured_max_draft_tokens"),
    ("speculative_decoding", "draft_acceptance_rate"),
    ("speculative_decoding", "mean_accepted_length"),
    ("speculative_decoding", "method"),
    ("speculative_decoding", "proposal_depth"),
    ("speculative_decoding", "reason"),
    ("telemetry", "sampled_energy_j"),
}
_SAFE_SUMMARY_NULLABLE_ROOTS = {
    "artifact_validation",
    "artifact_validation_telemetry",
    "first_request_telemetry",
    "llamacpp_dflash_evidence",
    "llamacpp_mtp_evidence",
    "runtime",
    "shutdown_telemetry",
    "speculative_decoding",
    "startup_telemetry",
}


def _reject_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON constant {value!r}")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


_DECODER = json.JSONDecoder(
    object_pairs_hook=_pairs_no_duplicates,
    parse_constant=_reject_constant,
)


def _secure_read(path: Path, root: Path, *, maximum: int) -> str:
    root = root.resolve(strict=True)
    path = Path(os.path.abspath(path))
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise EvidenceError("source file escapes results root") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise EvidenceError("source file has an unsafe relative path")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        metadata = cursor.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise EvidenceError(f"source symlink is not allowed: {relative}")
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise EvidenceError(f"source must be a single-link regular file: {relative}")
    if metadata.st_size > maximum:
        raise EvidenceError(f"source file exceeds size limit: {relative}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
        ):
            raise EvidenceError(f"source changed while opening: {relative}")
        data = b""
        while len(data) <= maximum:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(data)))
            if not chunk:
                break
            data += chunk
    finally:
        os.close(descriptor)
    if len(data) > maximum or b"\x00" in data:
        raise EvidenceError(f"invalid source bytes: {relative}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"source is not UTF-8: {relative}") from error


def _load_json(path: Path, root: Path) -> Any:
    text = _secure_read(path, root, maximum=MAX_SOURCE_JSON_BYTES)
    try:
        value, end = _DECODER.raw_decode(text)
    except json.JSONDecodeError as error:
        raise EvidenceError(f"invalid JSON in {path.name}: {error.msg}") from error
    if text[end:].strip():
        raise EvidenceError(f"trailing data in JSON file {path.name}")
    return value


def _secure_owner_read(path: Path, *, maximum: int) -> bytes:
    """Read one explicit owner-private file without following links."""

    absolute = Path(os.path.abspath(path))
    try:
        directory_descriptor = os.open(
            absolute.anchor,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0),
        )
    except OSError as error:
        raise EvidenceError("Harbor result cannot be inspected") from error
    try:
        for component in absolute.parts[1:-1]:
            component_metadata = os.stat(
                component, dir_fd=directory_descriptor, follow_symlinks=False
            )
            if stat.S_ISLNK(component_metadata.st_mode):
                raise EvidenceError("Harbor result path contains a symbolic link")
            next_descriptor = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            try:
                opened_component = os.fstat(next_descriptor)
            except BaseException:
                os.close(next_descriptor)
                raise
            if (
                not stat.S_ISDIR(opened_component.st_mode)
                or (opened_component.st_dev, opened_component.st_ino)
                != (component_metadata.st_dev, component_metadata.st_ino)
            ):
                os.close(next_descriptor)
                raise EvidenceError("Harbor result path changed while opening")
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        name = absolute.parts[-1]
        metadata = os.stat(
            name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if stat.S_ISLNK(metadata.st_mode):
            raise EvidenceError("Harbor result path contains a symbolic link")
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise EvidenceError(
                "Harbor result must be an owner-mode-0600 single-link regular file"
            )
        if metadata.st_size > maximum:
            raise EvidenceError("Harbor result exceeds the source size limit")
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise EvidenceError("Harbor result cannot be opened safely") from error
    finally:
        os.close(directory_descriptor)
    try:
        opened = os.fstat(descriptor)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_nlink,
            opened.st_uid,
            stat.S_IMODE(opened.st_mode),
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        expected_identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_nlink,
            metadata.st_uid,
            stat.S_IMODE(metadata.st_mode),
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        if identity != expected_identity:
            raise EvidenceError("Harbor result changed while opening")
        chunks: list[bytes] = []
        total = 0
        while total <= maximum:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        final = os.fstat(descriptor)
        final_identity = (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_nlink,
            final.st_uid,
            stat.S_IMODE(final.st_mode),
            final.st_mtime_ns,
            final.st_ctime_ns,
        )
        if final_identity != identity:
            raise EvidenceError("Harbor result changed while reading")
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if len(data) != metadata.st_size or len(data) > maximum or b"\x00" in data:
        raise EvidenceError("Harbor result has invalid source bytes")
    return data


def _json_strict_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int or int/float coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_strict_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return bool(left == right)


def _expected_harbor_pin_sections(campaign: Any) -> dict[str, Any]:
    agents: list[dict[str, Any]] = []
    for agent in campaign.agents:
        projection = {
            "id": agent.id,
            "version": agent.version,
            "source": agent.source,
            "revision": agent.revision,
            "install_tree_sha256": agent.install_tree_sha256,
            "install_tree_size_bytes": agent.install_tree_size_bytes,
            "npm_package": agent.npm_package,
            "npm_integrity": agent.npm_integrity,
            "npm_shasum": agent.npm_shasum,
        }
        if agent.platform_package is not None:
            projection.update(
                {
                    "platform_package": agent.platform_package,
                    "platform_integrity": agent.platform_integrity,
                    "platform_shasum": agent.platform_shasum,
                }
            )
        agents.append(projection)
    return {
        "harbor": {
            "source": campaign.harbor.source,
            "revision": campaign.harbor.revision,
            "version": campaign.harbor.version,
            "runtime_tree_sha256": campaign.harbor.runtime_tree_sha256,
            "runtime_tree_size_bytes": campaign.harbor.runtime_tree_size_bytes,
            "runtime_tree_entries": campaign.harbor.runtime_tree_entries,
            "runtime_tree_files": campaign.harbor.runtime_tree_files,
            "runtime_tree_links": campaign.harbor.runtime_tree_links,
            "executable_size_bytes": campaign.harbor.executable_size_bytes,
            "executable_sha256": campaign.harbor.executable_sha256,
            "agent_source_sha256": campaign.harbor.agent_source_sha256,
            "python_version": campaign.harbor.python_version,
            "python_size_bytes": campaign.harbor.python_size_bytes,
            "python_sha256": campaign.harbor.python_sha256,
        },
        "dataset": {
            "source": campaign.dataset.source,
            "revision": campaign.dataset.revision,
            "version": campaign.dataset.version,
            "network_policy_patch_digest": HARBOR_EXPECTED_DERIVATION_DIGEST,
        },
        "model": {
            "profile": campaign.model.profile,
            "served_name": campaign.model.served_name,
            "context_tokens": campaign.model.context_tokens,
            "max_output_tokens": campaign.model.max_output_tokens,
            "parallel": campaign.model.parallel,
            "temperature": campaign.execution.server_default_temperature,
            "top_p": campaign.execution.server_default_top_p,
            "top_k": campaign.execution.server_default_top_k,
        },
        "relay": {
            "protocol": "phase-isolated-loopback-uds-relay-v1",
            "listen_host": campaign.relay.listen_host,
            "port": campaign.relay.port,
            "sentinel_host": campaign.relay.sentinel_host,
            "node_image": campaign.relay.node_image,
            "relay_script_sha256": campaign.relay.relay_script_sha256,
            "network_policy_sha256": campaign.relay.network_policy_sha256,
        },
        "toolchain": {
            "node_version": campaign.toolchain.node_version,
            "npm_builder_version": campaign.toolchain.npm_builder_version,
            "node_binary_sha256": campaign.toolchain.node_binary_sha256,
            "node_tree_sha256": campaign.toolchain.node_tree_sha256,
            "node_tree_size_bytes": campaign.toolchain.node_tree_size_bytes,
        },
        "agents": agents,
    }


def _validate_harbor_envelope(payload: Any) -> dict[str, Any]:
    """Apply the tracked lifecycle schema plus publication-only success gates."""

    if not isinstance(payload, dict):
        raise EvidenceError("Harbor result must be a JSON object")
    try:
        _validate_output_value(payload, pointer="/harbor-result")
    except EvidenceError as error:
        raise EvidenceError("Harbor result violates scalar publication policy") from error
    try:
        from .harbor_campaign_lifecycle import (
            CampaignLifecycleError,
            DEFAULT_CAMPAIGN_PATH,
            _EXPECTED_MODEL,
            validate_lifecycle_envelope,
        )
        from .harbor_terminal import HarborCampaignError, load_campaign

        campaign = load_campaign(DEFAULT_CAMPAIGN_PATH)
        validate_lifecycle_envelope(payload, campaign=campaign)
    except (CampaignLifecycleError, HarborCampaignError, OSError, ValueError) as error:
        raise EvidenceError("Harbor lifecycle envelope failed exact validation") from error

    if (
        campaign.id != HARBOR_CAMPAIGN_ID
        or payload["campaign_id"] != HARBOR_CAMPAIGN_ID
        or payload["status"] != "completed"
        or payload["stop_reason"] != "completed"
        or payload["git"]
        != {"clean": True, "revision": HARBOR_EXPECTED_GIT_REVISION}
        or not _json_strict_equal(payload["model"], _EXPECTED_MODEL)
        or type(payload["schema_version"]) is not int
        or type(payload["campaign"]["schema_version"]) is not int
    ):
        raise EvidenceError("Harbor result is not the completed tracked campaign")
    admission = payload["admission"]
    model_admission = admission["model"]
    if (
        model_admission is None
        or any(
            model_admission[key] is not True
            for key in ("chat_passed", "json_passed", "tool_call_passed")
        )
        or admission["agent_trees_verified"] != 2
        or admission["npm_artifacts_verified"] != 3
        or admission["runtime_assets_verified"] != 2
        or admission["agent_source_files_verified"] != 1
        or any(
            admission[key] is not True
            for key in (
                "artifact_validation",
                "harbor_runtime_verified",
                "node_tree_verified",
                "python_bytecode_cache_empty",
                "host_arm64",
                "docker_server_arm64",
                "unix_bridge_verified",
            )
        )
    ):
        raise EvidenceError("Harbor runtime admission is incomplete")
    if any(value is not True for value in payload["cleanup"].values()):
        raise EvidenceError("Harbor lifecycle cleanup is incomplete")

    execution = payload["execution"]
    totals = payload["campaign"]["summary"]
    trials = payload["campaign"]["trials"]
    if (
        type(execution["trials_planned"]) is not int
        or type(execution["hard_cutoff_s"]) is not int
        or type(execution["audit_reserve_s"]) is not int
        or execution["trials_planned"] != 12
        or execution["trials_started"] != 12
        or execution["trials_completed"] != 12
        or execution["cutoff_reached"] is not False
        or totals["planned_attempts"] != 12
        or totals["attempts"] != 12
        or totals["completed_results"] != 12
        or totals["missing_results"] != 0
        or totals["unstarted_attempts"] != 0
        or totals["campaign_complete"] is not True
        or totals["campaign_cutoff_reached"] is not False
        or any(totals[field] != 0 for field in _HARBOR_ZERO_INFRASTRUCTURE_FIELDS)
    ):
        raise EvidenceError("Harbor campaign execution or infrastructure gates failed")
    for trial in trials:
        if (
            type(trial["trial_index"]) is not int
            or trial["result_available"] is not True
            or trial["reward"] is None
            or trial["harbor_exit_code"] != 0
            or trial["harbor_timed_out"] is not False
            or any(trial[field] is not True for field in _HARBOR_TRIAL_SECURITY_FIELDS)
            or any(
                trial[f"{resource}_found"] != trial[f"{resource}_removed"]
                for resource in ("containers", "networks", "volumes")
            )
        ):
            raise EvidenceError("Harbor trial infrastructure gates failed")
    pins = payload["campaign"]["pins"]
    expected_pin_sections = _expected_harbor_pin_sections(campaign)
    if any(
        not _json_strict_equal(pins[key], expected)
        for key, expected in expected_pin_sections.items()
    ):
        raise EvidenceError("Harbor campaign pins or scalar types changed")
    return payload


def _load_harbor_result(path: Path) -> dict[str, Any]:
    data = _secure_owner_read(path, maximum=MAX_SOURCE_JSON_BYTES)
    try:
        text = data.decode("utf-8")
        value, end = _DECODER.raw_decode(text)
    except EvidenceError as error:
        raise EvidenceError("Harbor result violates the strict JSON grammar") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError("Harbor result is not valid UTF-8 JSON") from error
    if text[end:].strip():
        raise EvidenceError("Harbor result contains trailing JSON data")
    if _canonical(value) != data:
        raise EvidenceError("Harbor result is not canonical JSON")
    return _validate_harbor_envelope(value)


def _order_harbor_replicates(
    replicates: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(replicates) != HARBOR_REPLICATE_COUNT:
        raise EvidenceError("Harbor evidence requires exactly two result files")
    encoded = [_canonical(replicate) for replicate in replicates]
    if encoded[0] == encoded[1]:
        raise EvidenceError("Harbor result inputs are duplicates")
    started = [
        _parse_timestamp(replicate["started_at"], name="Harbor replicate started_at")
        for replicate in replicates
    ]
    if started[0] == started[1]:
        raise EvidenceError("Harbor result inputs have duplicate start timestamps")
    ordered_pairs = sorted(zip(started, replicates), key=lambda item: item[0])
    ordered = (ordered_pairs[0][1], ordered_pairs[1][1])
    first, second = ordered
    if (
        not _json_strict_equal(first["git"], second["git"])
        or not _json_strict_equal(first["model"], second["model"])
        or not _json_strict_equal(
            first["campaign"]["pins"], second["campaign"]["pins"]
        )
        or first["campaign_id"] != second["campaign_id"]
    ):
        raise EvidenceError("Harbor result inputs do not share exact campaign pins")
    return ordered


def _load_json_lines(path: Path, root: Path) -> list[dict[str, Any]]:
    text = _secure_read(path, root, maximum=MAX_SOURCE_JSON_BYTES)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if len(line.encode()) > MAX_SOURCE_LINE_BYTES:
            raise EvidenceError(f"oversized JSONL line in {path.name}:{line_number}")
        if not line.strip():
            continue
        try:
            value, end = _DECODER.raw_decode(line)
        except json.JSONDecodeError as error:
            raise EvidenceError(
                f"invalid JSONL in {path.name}:{line_number}: {error.msg}"
            ) from error
        if line[end:].strip() or not isinstance(value, dict):
            raise EvidenceError(f"invalid JSONL object in {path.name}:{line_number}")
        records.append(value)
    return records


def _load_complete_array_objects(path: Path, root: Path) -> tuple[list[Any], bool]:
    """Load JSON, retaining only complete values from an intentionally torn array."""

    text = _secure_read(path, root, maximum=MAX_SOURCE_JSON_BYTES)
    try:
        value, end = _DECODER.raw_decode(text)
    except json.JSONDecodeError:
        value = None
    else:
        if text[end:].strip():
            raise EvidenceError(f"trailing data in campaign JSON {path.name}")
        if not isinstance(value, list):
            raise EvidenceError(f"campaign JSON must be an array: {path.name}")
        return value, False
    stripped = text.lstrip()
    if not stripped.startswith("["):
        raise EvidenceError(f"invalid campaign JSON in {path.name}")
    position = len(text) - len(stripped) + 1
    values: list[Any] = []
    while True:
        while position < len(text) and (text[position].isspace() or text[position] == ","):
            position += 1
        if position >= len(text):
            break
        if text[position] == "]":
            raise EvidenceError(f"invalid complete campaign array: {path.name}")
        try:
            item, end = _DECODER.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        values.append(item)
        position = end
    if not values:
        raise EvidenceError(f"campaign JSON has no complete objects: {path.name}")
    remaining = text[position:].lstrip()
    if remaining.startswith("]"):
        raise EvidenceError(f"campaign truncation is not an incomplete value: {path.name}")
    return values, True


def _finite(value: Any, *, name: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{name} must be numeric or null")
    if not math.isfinite(float(value)):
        raise EvidenceError(f"{name} must be finite")
    return value


def _safe_id(value: Any, *, name: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise EvidenceError(f"unsafe {name}")
    return value


def _safe_text(value: Any, *, name: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _SAFE_TEXT_RE.fullmatch(value):
        raise EvidenceError(f"unsafe {name}")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise EvidenceError(f"{name} must be SHA-256 text")
    normalized = value.removeprefix("sha256:").lower()
    if not _HEX_RE.fullmatch(normalized):
        raise EvidenceError(f"invalid SHA-256 for {name}")
    return normalized


def _revision(value: Any, *, name: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _REVISION_RE.fullmatch(value.lower()):
        raise EvidenceError(f"invalid revision for {name}")
    return value.lower()


def _date_from_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise EvidenceError(f"unsafe run ID {run_id!r}")
    try:
        return datetime.strptime(run_id[:8], "%Y%m%d").date().isoformat()
    except ValueError as error:
        raise EvidenceError(f"invalid date in run ID {run_id!r}") from error


def _is_grouped_run_root(name: str) -> bool:
    """Recognize the two audited timestamped group layouts under ``results``."""

    for pattern, timestamp_format in _GROUPED_RUN_ROOT_PATTERNS:
        match = pattern.fullmatch(name)
        if match is None:
            continue
        try:
            datetime.strptime(match.group(1), timestamp_format)
        except ValueError as error:
            raise EvidenceError(f"invalid date in grouped results root {name!r}") from error
        return True
    return False


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalize_status(summary: dict[str, Any] | None, events: list[dict[str, Any]]) -> str:
    status = summary.get("status") if summary else None
    terminal = next(
        (event for event in reversed(events) if event.get("event") in {"run_complete", "run_aborted"}),
        None,
    )
    if terminal and terminal.get("event") == "run_aborted":
        if status not in {None, "aborted"}:
            raise EvidenceError("summary status conflicts with run_aborted")
        return "aborted"
    if terminal:
        if status not in {None, "complete", "partial"}:
            raise EvidenceError("summary status conflicts with run_complete")
        return str(status or "legacy_terminal")
    if status is not None:
        raise EvidenceError("summary claims a status without a terminal journal event")
    return "nonterminal"


def _project_model(plan: dict[str, Any], summary: dict[str, Any] | None) -> dict[str, Any]:
    model = plan.get("model")
    if not isinstance(model, dict):
        model = summary.get("model") if summary else None
    if not isinstance(model, dict):
        return {}
    result: dict[str, Any] = {}
    for key in (
        "id",
        "backend",
        "architecture",
        "prefix_cache_mode",
        "quantization",
        "source",
    ):
        if model.get(key) is not None:
            result[key] = _safe_id(model[key], name=f"model.{key}")
    if model.get("revision") is not None:
        result["revision"] = _revision(model["revision"], name="model.revision")
    if model.get("support_status") is not None:
        result["support_status"] = _safe_id(
            model["support_status"], name="model.support_status"
        )
    for key in (
        "estimated_ram_gib",
        "max_context",
        "native_context",
        "runtime_parallel",
        "startup_timeout_s",
        "weight_file_count",
        "weight_size_bytes",
    ):
        if key in model and model[key] is not None:
            result[key] = _finite(model[key], name=f"model.{key}")
    if model.get("lifecycle") is not None:
        result["lifecycle"] = _safe_id(model["lifecycle"], name="model.lifecycle")
    tasks = model.get("tasks")
    if isinstance(tasks, list):
        result["tasks"] = [_safe_id(item, name="model.task") for item in tasks]
    if result.get("prefix_cache_mode") is not None:
        return _project_prefix_cache_model(result)
    return result


_MEMORY_MODEL_REQUIRED_FIELDS = frozenset(
    {
        "architecture",
        "backend",
        "estimated_ram_gib",
        "id",
        "lifecycle",
        "max_context",
        "memory_thinking_enabled",
        "native_context",
        "quantization",
        "revision",
        "runtime_parallel",
        "source",
        "startup_timeout_s",
        "support_status",
        "tasks",
    }
)
_MEMORY_MODEL_OPTIONAL_FIELDS = frozenset(
    {"weight_file_count", "weight_size_bytes"}
)


def _validate_projected_memory_model(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or not _MEMORY_MODEL_REQUIRED_FIELDS <= set(value)
        or set(value)
        - _MEMORY_MODEL_REQUIRED_FIELDS
        - _MEMORY_MODEL_OPTIONAL_FIELDS
    ):
        raise EvidenceError("memory model does not match its exact evidence schema")
    if value.get("backend") != "llamacpp":
        raise EvidenceError("memory evidence requires the fixed llama.cpp backend")
    for key, expected in (
        ("max_context", MEMORY_OPERATION_CONTEXT_TOKENS),
        ("native_context", MEMORY_OPERATION_CONTEXT_TOKENS),
        ("runtime_parallel", 1),
    ):
        if type(value.get(key)) is not int or value[key] != expected:
            raise EvidenceError(f"memory model {key} changed")
    for key in (
        "id",
        "architecture",
        "quantization",
        "source",
        "support_status",
        "lifecycle",
    ):
        _safe_id(value.get(key), name=f"memory model.{key}")
    _revision(value.get("revision"), name="memory model.revision")
    for key in ("estimated_ram_gib", "startup_timeout_s"):
        number = _finite(value.get(key), name=f"memory model.{key}")
        if number is None or number <= 0:
            raise EvidenceError(f"memory model {key} must be positive")
    for key in _MEMORY_MODEL_OPTIONAL_FIELDS & set(value):
        if type(value[key]) is not int or value[key] <= 0:
            raise EvidenceError(f"memory model {key} must be a positive integer")
    tasks = value.get("tasks")
    if (
        not isinstance(tasks, list)
        or any(not isinstance(task, str) for task in tasks)
        or len(tasks) != len(set(tasks))
        or not {"chat", "json"} <= set(tasks)
    ):
        raise EvidenceError("memory model tasks must include unique chat and json tasks")
    for task in tasks:
        _safe_id(task, name="memory model.task")
    thinking_enabled = value.get("memory_thinking_enabled")
    if not isinstance(thinking_enabled, bool):
        raise EvidenceError("memory model thinking policy must be boolean")
    if ("thinking" in tasks) is not thinking_enabled:
        raise EvidenceError("memory thinking task and policy disagree")
    expected = _MEMORY_PANEL_MODELS.get(str(value.get("id")))
    if expected is None or not _json_strict_equal(value, expected):
        raise EvidenceError("memory model is outside the exact frozen panel")
    return value


def _project_memory_model(
    source: Any, projected: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise EvidenceError("memory source model must be an object")
    try:
        request_body = json.loads(source.get("request_body_json", ""))
    except (json.JSONDecodeError, TypeError) as error:
        raise EvidenceError("memory model thinking policy is invalid") from error
    if (
        not isinstance(request_body, dict)
        or set(request_body) != {"chat_template_kwargs"}
        or not isinstance(request_body["chat_template_kwargs"], dict)
        or set(request_body["chat_template_kwargs"]) != {"enable_thinking"}
        or not isinstance(
            request_body["chat_template_kwargs"]["enable_thinking"], bool
        )
    ):
        raise EvidenceError("memory model thinking policy changed")
    thinking_enabled = request_body["chat_template_kwargs"]["enable_thinking"]
    arguments = source.get("args")
    if (
        not isinstance(arguments, list)
        or tuple(arguments)
        != memory_operation_llamacpp_args(enable_thinking=thinking_enabled)
    ):
        raise EvidenceError("memory model llama.cpp arguments changed")
    if source.get("lifecycle") != "subprocess":
        raise EvidenceError("memory model lifecycle must be subprocess")
    if (
        _revision(source.get("runtime_revision"), name="memory runtime revision")
        != MEMORY_OPERATION_LLAMACPP_REVISION
        or _sha256(source.get("runtime_digest"), name="memory runtime binary")
        != _sha256(
            MEMORY_OPERATION_LLAMACPP_DIGEST,
            name="fixed memory runtime binary",
        )
    ):
        raise EvidenceError("memory model llama.cpp pin changed")
    result = {**projected, "memory_thinking_enabled": thinking_enabled}
    return _validate_projected_memory_model(result)


def _bind_memory_summary_model(
    *, source_model: dict[str, Any], summary: dict[str, Any] | None
) -> None:
    if not isinstance(summary, dict) or summary.get("model") is None:
        raise EvidenceError("memory summary must retain its frozen model identity")
    summary_model = summary.get("model")
    if not isinstance(summary_model, dict):
        raise EvidenceError("memory summary model must be an object")
    expected_fields = {
        "architecture",
        "backend",
        "id",
        "max_context",
        "native_context",
        "quantization",
        "revision",
        "source",
        "support_status",
    }
    if set(summary_model) != expected_fields:
        raise EvidenceError("memory summary model does not match its exact schema")
    if not _json_strict_equal(
        summary_model, {key: source_model.get(key) for key in expected_fields}
    ):
        raise EvidenceError("memory summary model disagrees with the frozen model")


def _agentic_case_identifier(case_id: Any, scenario_id: str) -> str:
    value = _safe_id(case_id, name="agentic suite case_id")
    if not re.fullmatch(rf"{re.escape(scenario_id)}--[0-9a-f]{{12}}", value):
        raise EvidenceError("agentic case identifier does not match its scenario")
    return value


def _agentic_exact_integer(value: Any, expected: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise EvidenceError(f"{name} must be the integer {expected}")
    return value


def _project_agentic_suite(suite: Any) -> dict[str, Any]:
    if not isinstance(suite, dict) or frozenset(suite) not in {
        _AGENTIC_SUITE_FIELDS,
        _AGENTIC_SUITE_FIELDS | {"description"},
    }:
        raise EvidenceError("agentic suite does not match its exact schema")
    if "description" in suite and suite["description"] != _AGENTIC_SUITE_DESCRIPTION:
        raise EvidenceError("agentic suite description changed")
    if suite.get("id") != "agentic-tools":
        raise EvidenceError("agentic suite identifier must be agentic-tools")
    _agentic_exact_integer(
        suite.get("schema_version"), 1, name="agentic suite schema_version"
    )
    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) != len(KNOWN_AGENTIC_CASE_IDS):
        raise EvidenceError("agentic suite must contain exactly four cases")

    projected_cases: list[dict[str, Any]] = []
    scenarios: list[str] = []
    case_ids: list[str] = []
    for case in cases:
        if not isinstance(case, dict) or set(case) != _AGENTIC_SUITE_CASE_FIELDS:
            raise EvidenceError("agentic suite case does not match its exact schema")
        scenario_id = case.get("id")
        if scenario_id not in KNOWN_AGENTIC_CASE_IDS:
            raise EvidenceError("agentic suite contains an unsupported scenario")
        if case.get("kind") != "agentic":
            raise EvidenceError("agentic suite case kind must be agentic")
        case_id = _agentic_case_identifier(case.get("case_id"), scenario_id)
        for key, expected in (
            ("concurrency", 1),
            ("max_output_tokens", 4_096),
            ("max_turns", 6),
            ("prompt_repetitions", 0),
            ("repetitions", 3),
            ("warmups", 0),
        ):
            _agentic_exact_integer(
                case.get(key), expected, name=f"agentic suite case {key}"
            )
        temperature = case.get("temperature")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or float(temperature) != 0.0
        ):
            raise EvidenceError("agentic suite case temperature must be numeric zero")
        if case.get("requires") != ["chat", "tools"]:
            raise EvidenceError(
                "agentic suite case requires must be exactly ['chat', 'tools']"
            )
        scenarios.append(scenario_id)
        case_ids.append(case_id)
        projected_cases.append(dict(case))

    if len(case_ids) != len(set(case_ids)):
        raise EvidenceError("agentic suite case identifiers must be unique")
    if len(scenarios) != len(set(scenarios)) or set(scenarios) != set(
        KNOWN_AGENTIC_CASE_IDS
    ):
        raise EvidenceError("agentic suite must contain each known scenario once")
    return {
        "cases": projected_cases,
        "id": "agentic-tools",
        "schema_version": 1,
    }


def _memory_case_identifier(case_id: Any, scenario_id: str) -> str:
    value = _safe_id(case_id, name="memory suite case_id")
    if not re.fullmatch(rf"{re.escape(scenario_id)}--[0-9a-f]{{12}}", value):
        raise EvidenceError("memory case identifier does not match its scenario")
    return value


def _memory_source_case_identifier(
    *,
    model: dict[str, Any],
    case: dict[str, Any],
    case_id: str,
    protocol_digest: str,
) -> None:
    unbound_case = {key: value for key, value in case.items() if key != "case_id"}
    payload = {
        "model": model,
        "case": unbound_case,
        "protocol_digest": protocol_digest,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:12]
    expected = f"{case['id']}--{digest}"
    if case_id != expected:
        raise EvidenceError("memory case identifier is not bound to its frozen model")


def _memory_bound_case_identifier(
    *, model: dict[str, Any], case: dict[str, Any], protocol_digest: str
) -> str:
    unbound_case = {key: value for key, value in case.items() if key != "case_id"}
    digest = hashlib.sha256(
        json.dumps(
            {
                "model": model,
                "case": unbound_case,
                "protocol_digest": protocol_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"{case['id']}--{digest}"


def _project_memory_suite(
    suite: Any, *, source_model: Any = None, binding_model: Any = None
) -> dict[str, Any]:
    fields = frozenset(suite) if isinstance(suite, dict) else frozenset()
    if fields not in {
        _MEMORY_SUITE_FIELDS,
        _MEMORY_SUITE_FIELDS | {"description"},
    }:
        raise EvidenceError("memory suite does not match its exact schema")
    assert isinstance(suite, dict)
    if "description" in suite and suite["description"] != _MEMORY_SUITE_DESCRIPTION:
        raise EvidenceError("memory suite description changed")
    if suite.get("id") != MEMORY_OPERATION_SUITE_ID:
        raise EvidenceError("memory suite identifier changed")
    _agentic_exact_integer(
        suite.get("schema_version"), 1, name="memory suite schema_version"
    )
    try:
        require_memory_operation_protocol_digest(suite.get("protocol_digest"))
    except ValueError as error:
        raise EvidenceError("memory suite protocol digest changed") from error
    protocol_digest = MEMORY_OPERATION_PROTOCOL_DIGEST
    if source_model is not None and not isinstance(source_model, dict):
        raise EvidenceError("memory suite source model must be an object")
    if binding_model is not None:
        _validate_projected_memory_model(binding_model)
    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) != len(
        MEMORY_OPERATION_SCENARIO_IDS
    ):
        raise EvidenceError("memory suite must contain exactly eleven cases")

    projected_cases: list[dict[str, Any]] = []
    for case, expected_scenario in zip(
        cases, MEMORY_OPERATION_SCENARIO_IDS, strict=True
    ):
        if not isinstance(case, dict) or set(case) != _MEMORY_SUITE_CASE_FIELDS:
            raise EvidenceError("memory suite case does not match its exact schema")
        if case.get("id") != expected_scenario or case.get("kind") != "memory":
            raise EvidenceError("memory suite case order or identity changed")
        case_id = _memory_case_identifier(case.get("case_id"), expected_scenario)
        if isinstance(source_model, dict):
            _memory_source_case_identifier(
                model=source_model,
                case=case,
                case_id=case_id,
                protocol_digest=protocol_digest,
            )
        for key, expected in (
            ("concurrency", 1),
            ("max_output_tokens", MEMORY_OPERATION_OUTPUT_TOKENS),
            ("max_turns", 1),
            ("prompt_repetitions", 0),
            ("repetitions", MEMORY_OPERATION_VARIANT_COUNT),
            ("warmups", 0),
        ):
            _agentic_exact_integer(
                case.get(key), expected, name=f"memory suite case {key}"
            )
        if type(case.get("temperature")) is not float or case["temperature"] != 0.0:
            raise EvidenceError("memory suite case temperature must be JSON 0.0")
        if case.get("requires") != ["chat", "json"]:
            raise EvidenceError(
                "memory suite case requires must be exactly ['chat', 'json']"
            )
        projected_case = dict(case)
        if isinstance(binding_model, dict):
            bound_case_id = _memory_bound_case_identifier(
                model=binding_model,
                case=projected_case,
                protocol_digest=protocol_digest,
            )
            if source_model is None and case_id != bound_case_id:
                raise EvidenceError(
                    "memory case identifier is not bound to its published model"
                )
            projected_case["case_id"] = bound_case_id
        projected_cases.append(projected_case)
    if len({case["case_id"] for case in projected_cases}) != len(projected_cases):
        raise EvidenceError("memory suite case identifiers must be unique")
    return {
        "cases": projected_cases,
        "id": MEMORY_OPERATION_SUITE_ID,
        "protocol_digest": protocol_digest,
        "schema_version": 1,
    }


def _project_autoresearch_suite(
    suite: Any, *, source_model: Any = None
) -> dict[str, Any]:
    """Validate and project the exact mixed nine-case autoresearch suite."""

    if not isinstance(suite, dict) or frozenset(suite) not in {
        frozenset({"cases", "id", "schema_version"}),
        frozenset({"cases", "description", "id", "schema_version"}),
    }:
        raise EvidenceError("autoresearch suite does not match its exact schema")
    if suite.get("id") != AUTORESEARCH_SUITE_ID:
        raise EvidenceError("autoresearch suite identifier changed")
    if suite.get("schema_version") != 1:
        raise EvidenceError("autoresearch suite schema version changed")
    description = suite.get("description", AUTORESEARCH_SUITE_DESCRIPTION)
    if description != AUTORESEARCH_SUITE_DESCRIPTION:
        raise EvidenceError("autoresearch suite description changed")
    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) != 9 or any(
        not isinstance(case, dict) for case in cases
    ):
        raise EvidenceError("autoresearch suite must contain exactly nine cases")
    unbound_cases = [
        {key: value for key, value in case.items() if key != "case_id"}
        for case in cases
    ]
    suite_basis = {
        "id": AUTORESEARCH_SUITE_ID,
        "cases": unbound_cases,
        "description": AUTORESEARCH_SUITE_DESCRIPTION,
        "schema_version": 1,
    }
    if _autoresearch_content_hash(suite_basis) != AUTORESEARCH_SUITE_SPEC_DIGEST:
        raise EvidenceError("autoresearch suite content changed")
    if source_model is not None and not isinstance(source_model, dict):
        raise EvidenceError("autoresearch source model must be an object")
    projected_cases: list[dict[str, Any]] = []
    for case, unbound in zip(cases, unbound_cases, strict=True):
        case_id = _safe_id(case.get("case_id"), name="autoresearch case ID")
        scenario_id = _safe_id(case.get("id"), name="autoresearch scenario ID")
        if not re.fullmatch(rf"{re.escape(scenario_id)}--[0-9a-f]{{12}}", case_id):
            raise EvidenceError("autoresearch case identifier changed")
        if isinstance(source_model, dict):
            expected = (
                f"{scenario_id}--"
                f"{_autoresearch_content_hash({'model': source_model, 'case': unbound}, length=12)}"
            )
            if case_id != expected:
                raise EvidenceError(
                    "autoresearch case identifier is not model-bound"
                )
        projected_cases.append(dict(case))
    if len({case["case_id"] for case in projected_cases}) != len(projected_cases):
        raise EvidenceError("autoresearch case identifiers are duplicated")
    return {
        "cases": projected_cases,
        "id": AUTORESEARCH_SUITE_ID,
        "schema_version": 1,
    }


def _project_suite(plan: dict[str, Any]) -> dict[str, Any] | None:
    suite = plan.get("suite")
    if not isinstance(suite, dict):
        parameters = plan.get("parameters")
        if isinstance(parameters, dict):
            return {
                "id": "llamacpp-perplexity",
                "parameters": {
                    key: _finite(value, name=f"parameters.{key}")
                    for key, value in parameters.items()
                    if key in {"chunks", "ctx_size", "timeout_s"}
                },
            }
        return None
    raw_cases = suite.get("cases")
    if suite.get("id") == AUTORESEARCH_SUITE_ID:
        return _project_autoresearch_suite(suite, source_model=plan.get("model"))
    if suite.get("id") == "agentic-tools" or (
        isinstance(raw_cases, list)
        and any(
            isinstance(case, dict) and case.get("kind") == "agentic"
            for case in raw_cases
        )
    ):
        return _project_agentic_suite(suite)
    if suite.get("id") == MEMORY_OPERATION_SUITE_ID or (
        isinstance(raw_cases, list)
        and any(
            isinstance(case, dict) and case.get("kind") == "memory"
            for case in raw_cases
        )
    ):
        return _project_memory_suite(suite, source_model=plan.get("model"))
    result: dict[str, Any] = {
        "id": _safe_id(suite.get("id"), name="suite.id"),
        "schema_version": suite.get("schema_version"),
        "cases": [],
    }
    if not isinstance(result["schema_version"], (int, str)):
        raise EvidenceError("invalid suite schema version")
    cases = suite.get("cases")
    if not isinstance(cases, list):
        raise EvidenceError("suite cases must be a list")
    allowed = {
        "case_id",
        "id",
        "kind",
        "max_output_tokens",
        "max_turns",
        "prompt_repetitions",
        "repetitions",
        "requires",
        "temperature",
        "warmups",
        "concurrency",
    }
    for case in cases:
        if not isinstance(case, dict):
            raise EvidenceError("suite case must be an object")
        projected: dict[str, Any] = {}
        for key, value in case.items():
            if key not in allowed:
                continue
            if key in {"case_id", "id", "kind"}:
                projected[key] = _safe_id(value, name=f"suite.case.{key}")
            elif key == "requires":
                if not isinstance(value, list):
                    raise EvidenceError("case requires must be a list")
                projected[key] = [_safe_id(item, name="suite.case.require") for item in value]
            else:
                projected[key] = _finite(value, name=f"suite.case.{key}")
        result["cases"].append(projected)
    if result["id"] == PREFIX_CACHE_SUITE_ID:
        return _project_prefix_cache_suite(result)
    return result


def _artifact_target(value: Any, *, fallback: str) -> str:
    if isinstance(value, str) and value:
        target = Path(value).name
        if _ID_RE.fullmatch(target):
            return target
    return fallback


def _sglang_source_overlay_declarations(
    model: dict[str, Any],
) -> list[tuple[Path, str, str, str]]:
    """Return plan-bound overlays without exposing either source path.

    The returned tuple is ``(results-relative path, basename, digest,
    container path)``.  Both paths are validated here, while callers publish
    only the basename and digest.
    """

    raw_overlays = model.get("sglang_source_overlays")
    if raw_overlays is None:
        return []
    if not isinstance(raw_overlays, list):
        raise EvidenceError("SGLang source overlays must be a list")
    if raw_overlays and model.get("backend") != "sglang":
        raise EvidenceError("SGLang source overlays require the sglang backend")

    declarations: list[tuple[Path, str, str, str]] = []
    seen_host_paths: set[Path] = set()
    seen_container_paths: set[str] = set()
    seen_basenames: set[str] = set()
    for index, overlay in enumerate(raw_overlays, 1):
        if not isinstance(overlay, dict) or set(overlay) != _SGLANG_SOURCE_OVERLAY_FIELDS:
            raise EvidenceError(
                f"SGLang source overlay {index} does not match its exact schema"
            )
        host_value = overlay.get("host_path")
        container_value = overlay.get("container_path")
        if not isinstance(host_value, str) or not isinstance(container_value, str):
            raise EvidenceError("SGLang source overlay paths must be text")

        host_path = PurePosixPath(host_value)
        host_parts = host_path.parts
        if (
            host_path.is_absolute()
            or host_path.as_posix() != host_value
            or len(host_parts) != 4
            or host_parts[:2] != _SGLANG_SOURCE_OVERLAY_HOST_PREFIX
            or any(part in {"", ".", ".."} for part in host_parts)
        ):
            raise EvidenceError("SGLang source overlay has an unsafe host path")
        recipe = _safe_id(host_parts[2], name="SGLang source overlay recipe")
        basename = _safe_id(host_parts[3], name="SGLang source overlay basename")
        if not recipe or not basename or not basename.endswith(".py"):
            raise EvidenceError("SGLang source overlay must identify one Python file")

        container_path = PurePosixPath(container_value)
        try:
            container_relative = container_path.relative_to(
                _SGLANG_SOURCE_OVERLAY_CONTAINER_ROOT
            )
        except ValueError as error:
            raise EvidenceError(
                "SGLang source overlay container path is outside the source tree"
            ) from error
        if (
            not container_path.is_absolute()
            or container_path.as_posix() != container_value
            or any(part in {"", ".", ".."} for part in container_relative.parts)
            or container_path.name != basename
        ):
            raise EvidenceError("SGLang source overlay has an unsafe container path")

        relative_path = Path(*host_parts[1:])
        if relative_path in seen_host_paths:
            raise EvidenceError("SGLang source overlay host path is duplicated")
        if container_value in seen_container_paths:
            raise EvidenceError("SGLang source overlay container path is duplicated")
        if basename in seen_basenames:
            raise EvidenceError("SGLang source overlay basename is duplicated")
        seen_host_paths.add(relative_path)
        seen_container_paths.add(container_value)
        seen_basenames.add(basename)
        declarations.append(
            (
                relative_path,
                basename,
                _sha256(
                    overlay.get("digest"),
                    name=f"SGLang source overlay {index}",
                ),
                container_value,
            )
        )
    return sorted(declarations, key=lambda item: (item[1], item[2], item[0].as_posix()))


def _project_sglang_source_overlay_artifacts(
    model: dict[str, Any],
) -> list[dict[str, str]]:
    """Project overlay identities without publishing either source path."""

    return [
        {"sha256": digest, "target": basename}
        for _, basename, digest, _ in _sglang_source_overlay_declarations(model)
    ]


def _collect_artifacts(plan: dict[str, Any], summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    model = plan.get("model") if isinstance(plan.get("model"), dict) else {}

    def add(
        role: str,
        digest: Any,
        *,
        size: Any = None,
        source: Any = None,
        revision: Any = None,
        target: Any = None,
        duration: Any = None,
    ) -> None:
        if digest is None:
            return
        item: dict[str, Any] = {
            "role": _safe_id(role, name="artifact.role"),
            "sha256": _sha256(digest, name=f"artifact.{role}"),
        }
        if size is not None:
            item["size_bytes"] = _finite(size, name=f"artifact.{role}.size")
        if source is not None:
            item["source"] = _safe_id(source, name=f"artifact.{role}.source")
        if revision is not None:
            item["revision"] = _revision(revision, name=f"artifact.{role}.revision")
        item["target"] = _artifact_target(target, fallback=role)
        if duration is not None:
            item["duration_s"] = _finite(duration, name=f"artifact.{role}.duration")
        artifacts.append(item)

    if isinstance(model, dict):
        add(
            "model",
            model.get("model_digest"),
            size=model.get("model_size_bytes"),
            source=model.get("source"),
            revision=model.get("revision"),
            target=model.get("model_file"),
        )
        shards = model.get("model_shards")
        if isinstance(shards, list):
            for index, shard in enumerate(shards, 1):
                if not isinstance(shard, dict):
                    raise EvidenceError("model shard must be an object")
                add(
                    f"model_shard_{index}",
                    shard.get("digest"),
                    size=shard.get("size_bytes"),
                    source=model.get("source"),
                    revision=model.get("revision"),
                    target=shard.get("path"),
                )
        add(
            "draft_model",
            model.get("draft_model_digest"),
            size=model.get("draft_model_size_bytes") or model.get("draft_weight_size_bytes"),
            source=model.get("draft_source"),
            revision=model.get("draft_revision"),
            target=model.get("draft_model_file"),
        )
        add(
            "multimodal_projector",
            model.get("mmproj_digest"),
            size=model.get("mmproj_size_bytes"),
            source=model.get("source"),
            revision=model.get("revision"),
            target=model.get("mmproj_file"),
        )
        add(
            "runtime_binary",
            model.get("runtime_digest"),
            revision=model.get("runtime_revision"),
            target=model.get("runtime_binary"),
        )
        add("container_image", model.get("image_digest"), target="container-image")
        for index, overlay in enumerate(
            _project_sglang_source_overlay_artifacts(model), 1
        ):
            add(
                f"sglang_source_overlay_{index}",
                overlay["sha256"],
                target=overlay["target"],
            )

    verification = plan.get("verification")
    if not isinstance(verification, dict) and summary:
        verification = summary.get("artifact_verification")
    if isinstance(verification, dict):
        add(
            "audio_fixture",
            verification.get("audio_sha256"),
            size=verification.get("audio_bytes"),
            target="audio-fixture",
            duration=verification.get("audio_duration_s"),
        )
        add(
            "dataset",
            verification.get("dataset_sha256"),
            size=verification.get("dataset_size_bytes"),
            target="dataset",
        )
        add(
            "runtime_binary",
            verification.get("runtime_binary_sha256")
            or verification.get("runtime_executable_sha256"),
            size=verification.get("runtime_binary_size_bytes"),
            revision=verification.get("runtime_source_revision"),
            target="runtime-binary",
        )
        add(
            "runtime_lock",
            verification.get("runtime_lock_sha256"),
            target="runtime-lock",
        )
        add(
            "harness",
            verification.get("worker_logic_sha256"),
            target="worker-logic",
        )
        for container_key in ("lfs_artifacts",):
            values = verification.get(container_key)
            if isinstance(values, dict):
                for index, (target, item) in enumerate(sorted(values.items()), 1):
                    if not isinstance(item, dict):
                        raise EvidenceError("verified artifact must be an object")
                    add(
                        f"checkpoint_file_{index}",
                        item.get("sha256"),
                        size=item.get("bytes"),
                        source=model.get("source") if isinstance(model, dict) else None,
                        revision=model.get("revision") if isinstance(model, dict) else None,
                        target=target,
                    )
        for container_key in ("artifacts_sha256", "small_artifacts_sha256"):
            values = verification.get(container_key)
            if isinstance(values, dict):
                for index, (target, digest) in enumerate(sorted(values.items()), 1):
                    add(
                        f"verified_file_{index}",
                        digest,
                        source=model.get("source") if isinstance(model, dict) else None,
                        revision=model.get("revision") if isinstance(model, dict) else None,
                        target=target,
                    )

    expected = plan.get("expected_pins")
    if isinstance(expected, dict):
        add(
            "dataset",
            expected.get("dataset_sha256"),
            target="dataset",
        )
        add(
            "runtime_binary",
            expected.get("runtime_binary_sha256"),
            size=expected.get("runtime_binary_size_bytes"),
            revision=expected.get("runtime_source_revision"),
            target="llama-perplexity",
        )

    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for artifact in artifacts:
        key = (artifact["role"], artifact["sha256"], artifact["target"])
        unique[key] = artifact
    return [unique[key] for key in sorted(unique)]


def _project_sglang_provenance(model: dict[str, Any]) -> dict[str, Any]:
    """Publish an explicit, non-downgradeable SGLang runtime state."""

    overlays = _project_sglang_source_overlay_artifacts(model)
    source = {key: model.get(key) for key in _SGLANG_PLE_CACHE_SOURCE_FIELDS}
    populated = {key for key, value in source.items() if value is not None}
    backend = model.get("backend")
    mmap_present = "sglang_ple_mmap" in model
    mmap = model.get("sglang_ple_mmap")
    omitted_present = "sglang_ple_omitted" in model
    omitted = model.get("sglang_ple_omitted")

    if backend != "sglang":
        if (
            populated
            or overlays
            or (mmap_present and mmap is not None and mmap is not False)
            or (omitted_present and omitted is not False)
        ):
            raise EvidenceError("SGLang provenance requires the sglang backend")
        return {}

    if mmap_present and type(mmap) is not bool:
        raise EvidenceError("runtime.sglang_ple_mmap must be boolean")
    if not mmap_present:
        mmap = None
    if omitted_present and type(omitted) is not bool:
        raise EvidenceError("runtime.sglang_ple_omitted must be boolean")
    if not omitted_present:
        omitted = False
    omission_labeled = (
        isinstance(model.get("quantization"), str)
        and "ple-omitted" in model["quantization"]
    )
    mapped_labeled = (
        isinstance(model.get("quantization"), str)
        and "ple-fp8-mapped" in model["quantization"]
    )
    if bool(omitted) != omission_labeled:
        raise EvidenceError(
            "SGLang PLE omission flag and quantization label disagree"
        )
    if omitted and (mmap is not False or populated or not overlays):
        raise EvidenceError(
            "SGLang PLE omission requires mmap=false, no cache, and overlays"
        )
    if omitted:
        for key, expected in _QWEN38_PLE_OMISSION_IDENTITY.items():
            if model.get(key) != expected:
                raise EvidenceError(
                    "SGLang PLE omission artifact/recipe identity changed"
                )
        if overlays != _QWEN38_PLE_OMISSION_ARTIFACTS:
            raise EvidenceError(
                "SGLang PLE omission overlay identities changed"
            )
    if mapped_labeled:
        if not omitted_present or omitted:
            raise EvidenceError(
                "SGLang mapped-PLE study label requires an explicit false "
                "omission flag"
            )
        if (
            mmap is not True
            or populated != _SGLANG_PLE_CACHE_SOURCE_FIELDS
            or source["sglang_ple_cache_mode"] != "readonly"
        ):
            raise EvidenceError(
                "SGLang mapped-PLE study requires exact read-only cache state"
            )
        for key, expected in _QWEN38_PLE_OMISSION_IDENTITY.items():
            if model.get(key) != expected:
                raise EvidenceError(
                    "SGLang mapped-PLE study artifact/recipe identity changed"
                )
        if overlays != _QWEN38_PLE_OMISSION_ARTIFACTS:
            raise EvidenceError(
                "SGLang mapped-PLE study overlay identities changed"
            )
        for key, expected in _QWEN38_PLE_STUDY_CACHE.items():
            if model.get(key) != expected:
                raise EvidenceError(
                    "SGLang mapped-PLE study cache identity changed"
                )

    result: dict[str, Any] = {
        "sglang_ple_mmap": mmap,
        "sglang_provenance_version": (
            _SGLANG_PROVENANCE_CURRENT_VERSION
            if omitted_present
            else _SGLANG_PROVENANCE_VERSION
        ),
        "sglang_source_overlay_artifacts": overlays,
    }
    if omitted_present:
        result["sglang_ple_omitted"] = omitted
    if not populated:
        result["sglang_ple_cache_mode"] = (
            "legacy_unspecified"
            if mmap is None
            else ("writable" if mmap else "disabled")
        )
        return result
    if populated != _SGLANG_PLE_CACHE_SOURCE_FIELDS:
        raise EvidenceError("SGLang PLE cache provenance must be all present or absent")
    if mmap is not True:
        raise EvidenceError("SGLang PLE cache provenance requires SGLang PLE mmap")
    if source["sglang_ple_cache_mode"] != "readonly":
        raise EvidenceError("SGLang PLE cache mode must be readonly")
    result.update(
        {
            "sglang_ple_cache_marker_sha256": _sha256(
                source["sglang_ple_cache_marker_digest"],
                name="runtime.SGLang PLE cache marker",
            ),
            "sglang_ple_cache_mode": "readonly",
            "sglang_ple_cache_payload_sha256": _sha256(
                source["sglang_ple_cache_payload_digest"],
                name="runtime.SGLang PLE cache payload",
            ),
        }
    )
    return result


def _validate_projected_sglang_overlay_artifacts(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise EvidenceError("published SGLang source overlays must be a list")
    projected: list[dict[str, str]] = []
    seen_targets: set[str] = set()
    for index, artifact in enumerate(value, 1):
        if not isinstance(artifact, dict) or set(artifact) != {"sha256", "target"}:
            raise EvidenceError(
                f"published SGLang source overlay {index} has an invalid schema"
            )
        digest = _sha256(
            artifact.get("sha256"), name="published SGLang source overlay"
        )
        target = _safe_id(
            artifact.get("target"), name="published SGLang source overlay target"
        )
        if not target.endswith(".py") or Path(target).name != target:
            raise EvidenceError("published SGLang source overlay target changed")
        if target in seen_targets:
            raise EvidenceError("published SGLang source overlay target is duplicated")
        seen_targets.add(target)
        projected.append({"sha256": digest, "target": target})
    if projected != sorted(
        projected, key=lambda item: (item["target"], item["sha256"])
    ):
        raise EvidenceError("published SGLang source overlay order changed")
    return projected


def _validate_projected_sglang_provenance(
    runtime: dict[str, Any], model: Any, *, allow_absent: bool = False
) -> list[dict[str, str]]:
    sglang_fields = {key for key in runtime if key.startswith("sglang_")}
    if not sglang_fields:
        if allow_absent:
            return []
        raise EvidenceError("published SGLang provenance is required")
    if sglang_fields - _SGLANG_PROVENANCE_RUNTIME_FIELDS:
        raise EvidenceError("published SGLang provenance has unknown fields")
    version = runtime.get("sglang_provenance_version")
    if type(version) is not int or version not in {
        _SGLANG_PROVENANCE_VERSION,
        _SGLANG_PROVENANCE_CURRENT_VERSION,
    }:
        raise EvidenceError("published SGLang provenance version is invalid")
    required_fields = (
        _SGLANG_PROVENANCE_V2_RUNTIME_FIELDS
        if version == _SGLANG_PROVENANCE_CURRENT_VERSION
        else _SGLANG_PROVENANCE_V1_RUNTIME_FIELDS
    )
    if not required_fields <= sglang_fields:
        raise EvidenceError("published SGLang provenance is incomplete")
    if version == _SGLANG_PROVENANCE_VERSION and "sglang_ple_omitted" in runtime:
        raise EvidenceError("published SGLang v1 provenance contains v2 fields")
    if (
        runtime.get("backend") != "sglang"
        or not isinstance(model, dict)
        or model.get("backend") != "sglang"
    ):
        raise EvidenceError("published SGLang provenance is inconsistent")

    mode = runtime.get("sglang_ple_cache_mode")
    mmap = runtime.get("sglang_ple_mmap")
    omitted = runtime.get("sglang_ple_omitted", False)
    if type(omitted) is not bool:
        raise EvidenceError("published SGLang PLE omission flag must be boolean")
    omission_labeled = (
        isinstance(model.get("quantization"), str)
        and "ple-omitted" in model["quantization"]
    )
    mapped_labeled = (
        isinstance(model.get("quantization"), str)
        and "ple-fp8-mapped" in model["quantization"]
    )
    if omitted != omission_labeled:
        raise EvidenceError(
            "published SGLang PLE omission flag and model label disagree"
        )
    if mapped_labeled and (
        version != _SGLANG_PROVENANCE_CURRENT_VERSION or omitted
    ):
        raise EvidenceError(
            "published mapped-PLE study lost its explicit omission dimension"
        )
    hashes = _SGLANG_PLE_CACHE_RUNTIME_HASH_FIELDS & sglang_fields
    if not isinstance(mode, str) or mode not in _SGLANG_PLE_CACHE_MODES:
        raise EvidenceError("published SGLang PLE cache mode is invalid")
    if mode == "legacy_unspecified":
        valid_state = mmap is None and not hashes
    elif mode == "disabled":
        valid_state = mmap is False and not hashes
    elif mode == "writable":
        valid_state = mmap is True and not hashes
    else:
        if hashes != _SGLANG_PLE_CACHE_RUNTIME_HASH_FIELDS:
            raise EvidenceError("published SGLang PLE cache provenance is incomplete")
        valid_state = mmap is True
    if not valid_state:
        raise EvidenceError("published SGLang PLE cache provenance is inconsistent")
    if mode == "readonly":
        _sha256(
            runtime.get("sglang_ple_cache_marker_sha256"),
            name="published SGLang PLE cache marker",
        )
        _sha256(
            runtime.get("sglang_ple_cache_payload_sha256"),
            name="published SGLang PLE cache payload",
        )
    overlays = _validate_projected_sglang_overlay_artifacts(
        runtime.get("sglang_source_overlay_artifacts")
    )
    if omitted or mapped_labeled:
        expected_mode = "disabled" if omitted else "readonly"
        if mode != expected_mode or overlays != _QWEN38_PLE_OMISSION_ARTIFACTS:
            raise EvidenceError(
                "published SGLang PLE-study state or overlays changed"
            )
        if (
            model.get("source") != _QWEN38_PLE_OMISSION_IDENTITY["source"]
            or model.get("revision")
            != _QWEN38_PLE_OMISSION_IDENTITY["revision"]
        ):
            raise EvidenceError(
                "published SGLang PLE-study artifact identity changed"
            )
        if runtime.get("recipe_revision") != _QWEN38_PLE_OMISSION_IDENTITY[
            "recipe_revision"
        ]:
            raise EvidenceError(
                "published SGLang PLE-study recipe identity changed"
            )
    if mapped_labeled:
        if mmap is not True:
            raise EvidenceError(
                "published mapped-PLE study lost its read-only mapping"
            )
        if (
            runtime.get("sglang_ple_cache_marker_sha256")
            != _QWEN38_PLE_STUDY_CACHE[
                "sglang_ple_cache_marker_digest"
            ].removeprefix("sha256:")
            or runtime.get("sglang_ple_cache_payload_sha256")
            != _QWEN38_PLE_STUDY_CACHE[
                "sglang_ple_cache_payload_digest"
            ].removeprefix("sha256:")
        ):
            raise EvidenceError(
                "published mapped-PLE study cache identity changed"
            )
    return overlays


def _project_runtime(plan: dict[str, Any], summary: dict[str, Any] | None) -> dict[str, Any]:
    model = plan.get("model") if isinstance(plan.get("model"), dict) else {}
    result: dict[str, Any] = {}
    if isinstance(model, dict):
        for key in ("backend", "lifecycle"):
            if model.get(key) is not None:
                result[key] = _safe_id(model[key], name=f"runtime.{key}")
        if model.get("image") is not None:
            result["image"] = _safe_id(model["image"], name="runtime.image")
        if model.get("image_digest") is not None:
            result["image_sha256"] = _sha256(
                model["image_digest"], name="runtime.image_digest"
            )
        for key in ("runtime_revision", "recipe_revision"):
            if model.get(key) is not None:
                result[key] = _revision(model[key], name=f"runtime.{key}")
        result.update(_project_sglang_provenance(model))
    resolved = plan.get("resolved")
    if isinstance(resolved, dict) and isinstance(resolved.get("llamacpp"), dict):
        llama = resolved["llamacpp"]
        if llama.get("runtime_binary_sha256") is not None:
            result["binary_sha256"] = _sha256(
                llama["runtime_binary_sha256"], name="runtime.binary"
            )
        if llama.get("runtime_source_revision") is not None:
            result["source_revision"] = _revision(
                llama["runtime_source_revision"], name="runtime.source_revision"
            )
    runtime = summary.get("runtime") if summary else None
    if isinstance(runtime, dict):
        versions: dict[str, Any] = {}
        for key, value in runtime.items():
            if key == "device":
                continue
            if isinstance(value, bool) or isinstance(value, (int, float)):
                versions[key] = value
            elif isinstance(value, str) and _VERSION_RE.fullmatch(value):
                versions[key] = value
        if versions:
            result["versions"] = versions
    return result


def _project_hardware(plan: dict[str, Any]) -> dict[str, Any]:
    host = plan.get("host_at_plan")
    if not isinstance(host, dict):
        return {}
    result: dict[str, Any] = {}
    nvidia_smi = host.get("nvidia_smi")
    if isinstance(nvidia_smi, str):
        parts = [part.strip() for part in nvidia_smi.split(",")]
        if len(parts) != 3 or parts[0] != "NVIDIA GB10":
            raise EvidenceError("unrecognized public GPU identity")
        result.update(
            {
                "compute_capability": _safe_id(
                    parts[2], name="hardware.compute_capability"
                ),
                "driver_version": _safe_id(
                    parts[1], name="hardware.driver_version"
                ),
                "gpu": "NVIDIA GB10",
                "platform": "NVIDIA DGX Spark",
            }
        )
    if host.get("memtotal_kib") is not None:
        memory_kib = _finite(host["memtotal_kib"], name="hardware.memtotal_kib")
        result["unified_memory_bytes"] = int(memory_kib) * 1024
    if host.get("git_commit") is not None:
        result["harness_revision"] = _revision(
            host["git_commit"], name="hardware.harness_revision"
        )
        result["harness_worktree_dirty"] = bool(host.get("git_status"))
    return result


def _agentic_int(
    result: dict[str, Any], key: str, *, minimum: int = 0
) -> int:
    value = result[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceError(f"agentic request {key} must be an integer >= {minimum}")
    return value


def _agentic_number(
    result: dict[str, Any], key: str, *, nullable: bool = False
) -> float | None:
    value = result[key]
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"agentic request {key} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise EvidenceError(f"agentic request {key} must be finite and nonnegative")
    return number


def _project_agentic_request_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict) or set(result) != _AGENTIC_RESULT_FIELDS:
        raise EvidenceError("agentic request result does not match schema version 1")
    if result["schema_version"] != 1 or isinstance(result["schema_version"], bool):
        raise EvidenceError("agentic request schema_version must be 1")
    scenario_id = result["scenario_id"]
    if scenario_id not in KNOWN_AGENTIC_CASE_IDS:
        raise EvidenceError("agentic request scenario is unsupported")
    variant = _agentic_int(result, "variant")
    if variant > 2:
        raise EvidenceError("agentic request variant must be between 0 and 2")
    max_turns = _agentic_int(result, "max_turns", minimum=2)
    if max_turns > 8:
        raise EvidenceError("agentic request max_turns must not exceed 8")
    max_output_tokens = _agentic_int(
        result, "max_output_tokens", minimum=2_048
    )
    turns_used = _agentic_int(result, "turns_used", minimum=1)
    if turns_used > max_turns:
        raise EvidenceError("agentic request turns_used exceeds max_turns")

    integer_keys = (
        "completion_tokens",
        "emission_events",
        "expected_tool_calls",
        "length_terminated_turns",
        "malformed_tool_calls",
        "prompt_tokens",
        "tool_calls_executed",
        "tool_calls_requested",
        "tool_calls_succeeded",
        "tool_errors",
        "unknown_tool_calls",
    )
    integers = {key: _agentic_int(result, key) for key in integer_keys}
    if integers["expected_tool_calls"] != _AGENTIC_EXPECTED_CALLS[scenario_id]:
        raise EvidenceError("agentic request expected-tool count changed")
    if not (
        integers["tool_calls_succeeded"]
        <= integers["tool_calls_executed"]
        <= integers["tool_calls_requested"]
    ):
        raise EvidenceError("agentic request tool-call counters are inconsistent")
    if (
        integers["tool_calls_succeeded"] + integers["tool_errors"]
        != integers["tool_calls_executed"]
    ):
        raise EvidenceError("agentic request tool outcomes do not match executions")
    if (
        integers["malformed_tool_calls"] + integers["unknown_tool_calls"]
        + integers["tool_calls_executed"]
        > integers["tool_calls_requested"]
    ):
        raise EvidenceError("agentic request classified too many tool calls")
    if integers["length_terminated_turns"] > turns_used:
        raise EvidenceError("agentic request length terminations exceed turns")

    boolean_keys = (
        "final_answer_correct",
        "final_answer_emitted",
        "passed",
        "recovery_required",
        "recovery_succeeded",
        "tool_sequence_correct",
        "turn_limit_reached",
    )
    booleans: dict[str, bool] = {}
    for key in boolean_keys:
        value = result[key]
        if not isinstance(value, bool):
            raise EvidenceError(f"agentic request {key} must be boolean")
        booleans[key] = value
    if booleans["final_answer_correct"] and not booleans["final_answer_emitted"]:
        raise EvidenceError("agentic correct final answer was not emitted")
    if booleans["tool_sequence_correct"]:
        (
            expected_requested,
            expected_executed,
            expected_succeeded,
            expected_errors,
            minimum_turns,
        ) = _AGENTIC_TOOL_COUNTS[scenario_id]
        for key, expected in (
            ("tool_calls_requested", expected_requested),
            ("tool_calls_executed", expected_executed),
            ("tool_calls_succeeded", expected_succeeded),
            ("tool_errors", expected_errors),
            ("malformed_tool_calls", 0),
            ("unknown_tool_calls", 0),
        ):
            if integers[key] != expected:
                raise EvidenceError(
                    f"agentic request {key} disagrees with scenario {scenario_id}"
                )
        if turns_used < minimum_turns:
            raise EvidenceError("agentic request used too few turns for its scenario")
    recovery_expected = scenario_id == "agentic-tool-error-recovery"
    if booleans["recovery_required"] != recovery_expected:
        raise EvidenceError("agentic recovery requirement disagrees with scenario")
    expected_recovery = bool(
        recovery_expected
        and booleans["tool_sequence_correct"]
        and integers["tool_errors"] == 1
        and integers["tool_calls_succeeded"] == 1
    )
    if booleans["recovery_succeeded"] != expected_recovery:
        raise EvidenceError("agentic recovery outcome is inconsistent")
    expected_pass = bool(
        booleans["tool_sequence_correct"]
        and booleans["final_answer_emitted"]
        and booleans["final_answer_correct"]
        and not booleans["turn_limit_reached"]
        and integers["length_terminated_turns"] == 0
    )
    if booleans["passed"] != expected_pass:
        raise EvidenceError("agentic pass flag is internally inconsistent")

    failure_code = result["failure_code"]
    if integers["tool_calls_requested"] > 16:
        expected_failure = "tool_call_limit"
    elif integers["length_terminated_turns"]:
        expected_failure = "output_limit"
    elif integers["malformed_tool_calls"]:
        expected_failure = "malformed_tool_call"
    elif integers["unknown_tool_calls"]:
        expected_failure = "unknown_tool"
    elif booleans["turn_limit_reached"]:
        expected_failure = "turn_limit"
    elif not booleans["tool_sequence_correct"]:
        expected_failure = "tool_sequence"
    elif not booleans["final_answer_emitted"]:
        expected_failure = "missing_final"
    elif not booleans["final_answer_correct"]:
        expected_failure = "final_answer"
    else:
        expected_failure = None
    if failure_code != expected_failure:
        raise EvidenceError("agentic request failure code is inconsistent")
    if failure_code is not None and failure_code not in _AGENTIC_FAILURE_CODES:
        raise EvidenceError("failed agentic request has an invalid failure code")
    finish_reason = result["finish_reason"]
    if finish_reason is not None and finish_reason not in _AGENTIC_FINISH_REASONS:
        raise EvidenceError("agentic request has an invalid finish reason")
    if booleans["turn_limit_reached"] and finish_reason != "turn_limit":
        raise EvidenceError("agentic turn-limit result has the wrong finish reason")
    if any(result[key] is not None for key in ("decode_s", "decode_tps", "decode_metric_source")):
        raise EvidenceError("agentic request must not publish decode-token metrics")

    first_ttft = _agentic_number(result, "first_turn_ttft_s", nullable=True)
    ttft = _agentic_number(result, "ttft_s", nullable=True)
    request_elapsed = _agentic_number(result, "request_elapsed_s")
    wall_s = _agentic_number(result, "wall_s")
    elapsed_s = _agentic_number(result, "elapsed_s")
    output_tps = _agentic_number(result, "output_tps")
    assert request_elapsed is not None and wall_s is not None
    assert elapsed_s is not None and output_tps is not None
    if wall_s <= 0 or request_elapsed > wall_s + 1e-6:
        raise EvidenceError("agentic request wall time is inconsistent")
    if not math.isclose(elapsed_s, wall_s, rel_tol=1e-9, abs_tol=1e-9):
        raise EvidenceError("agentic elapsed and wall times disagree")
    if first_ttft != ttft:
        raise EvidenceError("agentic first-turn TTFT fields disagree")
    expected_tps = integers["completion_tokens"] / wall_s
    if not math.isclose(output_tps, expected_tps, rel_tol=1e-9, abs_tol=1e-9):
        raise EvidenceError("agentic request output rate is inconsistent")

    return {
        "completion_tokens": integers["completion_tokens"],
        "decode_metric_source": None,
        "decode_s": None,
        "decode_tps": None,
        "elapsed_s": elapsed_s,
        "emission_event_count": integers["emission_events"],
        "expected_tool_calls": integers["expected_tool_calls"],
        "failure_code": failure_code,
        "final_answer_correct": booleans["final_answer_correct"],
        "final_answer_emitted": booleans["final_answer_emitted"],
        "finish_reason": finish_reason,
        "first_turn_ttft_s": first_ttft,
        "length_terminated_turns": integers["length_terminated_turns"],
        "malformed_tool_calls": integers["malformed_tool_calls"],
        "max_output_tokens": max_output_tokens,
        "max_turns": max_turns,
        "passed": booleans["passed"],
        "prompt_tokens": integers["prompt_tokens"],
        "recovery_required": booleans["recovery_required"],
        "recovery_succeeded": booleans["recovery_succeeded"],
        "request_elapsed_s": request_elapsed,
        "scenario_id": scenario_id,
        "schema_version": 1,
        "tool_calls_executed": integers["tool_calls_executed"],
        "tool_calls_requested": integers["tool_calls_requested"],
        "tool_calls_succeeded": integers["tool_calls_succeeded"],
        "tool_errors": integers["tool_errors"],
        "tool_sequence_correct": booleans["tool_sequence_correct"],
        "ttft_s": ttft,
        "turn_limit_reached": booleans["turn_limit_reached"],
        "turns_used": turns_used,
        "unknown_tool_calls": integers["unknown_tool_calls"],
        "variant": variant,
        "wall_s": wall_s,
    }


def _memory_integer(
    result: dict[str, Any], key: str, *, positive: bool = False
) -> int:
    value = result[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"memory request {key} must be a JSON integer")
    if value < 0 or (positive and value <= 0):
        qualifier = "positive" if positive else "nonnegative"
        raise EvidenceError(f"memory request {key} must be {qualifier}")
    return value


def _memory_number(
    result: dict[str, Any],
    key: str,
    *,
    nullable: bool = False,
    positive: bool = False,
) -> float | None:
    value = result[key]
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"memory request {key} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number <= 0):
        raise EvidenceError(f"memory request {key} is outside its numeric domain")
    return number


def _memory_boolean(result: dict[str, Any], key: str) -> bool:
    value = result[key]
    if not isinstance(value, bool):
        raise EvidenceError(f"memory request {key} must be boolean")
    return value


def _memory_nullable_boolean(result: dict[str, Any], key: str) -> bool | None:
    value = result[key]
    if value is not None and not isinstance(value, bool):
        raise EvidenceError(f"memory request {key} must be boolean or null")
    return value


def _project_memory_request_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict) or set(result) != _MEMORY_RESULT_FIELDS:
        raise EvidenceError("memory request result does not match schema version 1")
    if type(result.get("schema_version")) is not int or result["schema_version"] != 1:
        raise EvidenceError("memory request schema_version must be the integer 1")
    scenario_id = result.get("scenario_id")
    if scenario_id not in MEMORY_OPERATION_SCENARIO_IDS:
        raise EvidenceError("memory request scenario is unsupported")
    variant = _memory_integer(result, "variant")
    if variant >= MEMORY_OPERATION_VARIANT_COUNT:
        raise EvidenceError("memory request variant is outside the fixed battery")
    max_output_tokens = _memory_integer(result, "max_output_tokens", positive=True)
    if max_output_tokens != MEMORY_OPERATION_OUTPUT_TOKENS:
        raise EvidenceError("memory request output budget changed")

    integer_keys = (
        "cached_prompt_tokens",
        "completion_tokens",
        "emission_events",
        "prompt_tokens",
        "server_cached_prompt_tokens",
        "server_decode_tokens",
        "server_prompt_tokens",
        "unexpected_field_count",
        "unexpected_tool_call_count",
    )
    integers = {key: _memory_integer(result, key) for key in integer_keys}
    reasoning = result.get("reasoning_tokens")
    if reasoning is not None:
        raise EvidenceError(
            "memory evidence v1 requires unavailable reasoning_tokens under b10453"
        )
    if integers["prompt_tokens"] <= 0:
        raise EvidenceError("memory request prompt_tokens must be positive")
    if integers["completion_tokens"] <= 0 or integers["emission_events"] <= 0:
        raise EvidenceError("memory request decode counters must be positive")
    if integers["emission_events"] > integers["completion_tokens"]:
        raise EvidenceError("memory request emission count exceeds decoded tokens")
    if integers["completion_tokens"] > max_output_tokens:
        raise EvidenceError("memory request completion exceeds its fixed output cap")
    if (
        integers["prompt_tokens"] + integers["completion_tokens"]
        > MEMORY_OPERATION_CONTEXT_TOKENS
    ):
        raise EvidenceError("memory request exceeds its fixed context admission")
    if integers["cached_prompt_tokens"] != 0:
        raise EvidenceError("memory request reused prompt cache state")
    if integers["server_cached_prompt_tokens"] != 0:
        raise EvidenceError("memory request server cache counter is nonzero")
    if integers["server_prompt_tokens"] != integers["prompt_tokens"]:
        raise EvidenceError("memory request prompt token counters disagree")
    if integers["server_decode_tokens"] != integers["completion_tokens"]:
        raise EvidenceError("memory request completion token counters disagree")

    boolean_keys = (
        "action_correct",
        "graphiti_resolver_case",
        "injection_refusal_required",
        "injection_refusal_succeeded",
        "json_object_emitted",
        "mutation_expected",
        "mutation_selected",
        "passed",
        "prompt_cache_disabled",
        "protected_value_emitted",
        "schema_valid",
        "secret_refusal_required",
        "secret_refusal_succeeded",
        "synthetic_extension_case",
    )
    booleans = {key: _memory_boolean(result, key) for key in boolean_keys}
    if booleans["prompt_cache_disabled"] is not True:
        raise EvidenceError("memory request did not disable prompt caching")
    nullable_boolean_keys = (
        "contradicted_facts_correct",
        "duplicate_facts_correct",
        "evidence_correct",
        "path_correct",
        "reason_correct",
        "resolver_decision_correct",
        "target_correct",
        "tier_correct",
        "valid_from_correct",
        "valid_to_correct",
        "value_correct",
    )
    nullable_booleans = {
        key: _memory_nullable_boolean(result, key)
        for key in nullable_boolean_keys
    }

    graphiti = scenario_id.startswith("graphiti-")
    if (
        booleans["graphiti_resolver_case"] is not graphiti
        or booleans["synthetic_extension_case"] is graphiti
    ):
        raise EvidenceError("memory request family flags disagree with its scenario")
    expected_resolver_action = result.get("expected_resolver_action")
    selected_resolver_action = result.get("selected_resolver_action")
    if graphiti:
        if expected_resolver_action != _MEMORY_EXPECTED_RESOLVER_ACTION[scenario_id]:
            raise EvidenceError("memory resolver oracle label changed")
        if selected_resolver_action not in _MEMORY_RESOLVER_ACTIONS:
            raise EvidenceError("memory resolver selected label is invalid")
        if any(
            nullable_booleans[key] is not None
            for key in (
                "evidence_correct",
                "path_correct",
                "reason_correct",
                "target_correct",
                "tier_correct",
                "valid_from_correct",
                "valid_to_correct",
                "value_correct",
            )
        ):
            raise EvidenceError("Graphiti resolver result contains extension metrics")
        if any(
            nullable_booleans[key] is None
            for key in ("contradicted_facts_correct", "duplicate_facts_correct")
        ):
            raise EvidenceError("Graphiti resolver set metrics are missing")
        expected_action_correct = selected_resolver_action == expected_resolver_action
        if booleans["action_correct"] is not expected_action_correct:
            raise EvidenceError("memory resolver action correctness is inconsistent")
        expected_resolver_correct = bool(
            expected_action_correct
            and nullable_booleans["duplicate_facts_correct"] is True
            and nullable_booleans["contradicted_facts_correct"] is True
        )
        if nullable_booleans["resolver_decision_correct"] is not expected_resolver_correct:
            raise EvidenceError("memory resolver decision correctness is inconsistent")
        if (
            nullable_booleans["duplicate_facts_correct"] is True
            and nullable_booleans["contradicted_facts_correct"] is True
            and (
                selected_resolver_action != expected_resolver_action
                or booleans["action_correct"] is not True
            )
        ):
            raise EvidenceError("exact Graphiti sets imply the resolver oracle action")
        if booleans["mutation_expected"] or booleans["mutation_selected"]:
            raise EvidenceError("Graphiti resolver result selected a memory mutation")
    else:
        if (
            expected_resolver_action is not None
            or selected_resolver_action is not None
            or nullable_booleans["resolver_decision_correct"] is not None
            or nullable_booleans["duplicate_facts_correct"] is not None
            or nullable_booleans["contradicted_facts_correct"] is not None
        ):
            raise EvidenceError("synthetic memory result contains resolver metrics")
        if any(
            nullable_booleans[key] is None
            for key in (
                "evidence_correct",
                "path_correct",
                "reason_correct",
                "target_correct",
                "tier_correct",
                "valid_from_correct",
                "valid_to_correct",
                "value_correct",
            )
        ):
            raise EvidenceError("synthetic memory field metrics are missing")
        if booleans["mutation_expected"] is not (
            scenario_id in _MEMORY_MUTATION_EXPECTED
        ):
            raise EvidenceError("memory mutation oracle flag changed")
        if (
            booleans["schema_valid"]
            and booleans["action_correct"]
            and booleans["mutation_selected"] is not booleans["mutation_expected"]
        ):
            raise EvidenceError("correct memory action selected the wrong mutation class")

    secret_required = scenario_id == "memory-secret-refusal"
    injection_required = scenario_id == "memory-injection-refusal"
    if (
        booleans["secret_refusal_required"] is not secret_required
        or booleans["injection_refusal_required"] is not injection_required
    ):
        raise EvidenceError("memory refusal requirement changed")
    if booleans["protected_value_emitted"] and not (
        secret_required or injection_required
    ):
        raise EvidenceError("memory request reports a protected value for an unprotected case")

    finish_reason = result.get("finish_reason")
    if finish_reason not in {"length", "stop", "tool_calls"}:
        raise EvidenceError("memory request finish reason is invalid")
    if (finish_reason == "tool_calls") is not bool(
        integers["unexpected_tool_call_count"]
    ):
        raise EvidenceError("memory request tool-call finish reason is inconsistent")
    decode_source = result.get("decode_metric_source")
    if decode_source not in _MEMORY_DECODE_SOURCES:
        raise EvidenceError("memory request decode source is invalid")
    # The frozen memory protocol is llama.cpp-only and requests both native
    # server counters and the established client decode estimate.
    if decode_source != "client_estimate":
        raise EvidenceError("memory request must use the client decode metric")
    elapsed_s = _memory_number(result, "elapsed_s", positive=True)
    ttft_s = _memory_number(result, "ttft_s")
    decode_s = _memory_number(result, "decode_s", positive=True)
    decode_tps = _memory_number(result, "decode_tps")
    output_tps = _memory_number(result, "output_tps")
    server_prompt_s = _memory_number(result, "server_prompt_s", positive=True)
    server_decode_s = _memory_number(result, "server_decode_s", positive=True)
    assert elapsed_s is not None and ttft_s is not None
    assert decode_s is not None and decode_tps is not None and output_tps is not None
    assert server_prompt_s is not None and server_decode_s is not None
    if ttft_s > elapsed_s + 1e-9 or decode_s > elapsed_s + 1e-9:
        raise EvidenceError("memory request timing counters are inconsistent")
    if not math.isclose(
        ttft_s + decode_s, elapsed_s, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise EvidenceError("memory request TTFT and decode time do not reconcile")
    if (
        server_prompt_s + server_decode_s
        > elapsed_s + MEMORY_OPERATION_SERVER_TIMING_TOLERANCE_S
    ):
        raise EvidenceError("memory native server time exceeds client request time")
    expected_output_tps = integers["completion_tokens"] / elapsed_s
    expected_decode_tps = max(integers["completion_tokens"] - 1, 0) / decode_s
    if not math.isclose(output_tps, expected_output_tps, rel_tol=1e-6, abs_tol=1e-6):
        raise EvidenceError("memory request output rate is inconsistent")
    if not math.isclose(decode_tps, expected_decode_tps, rel_tol=1e-6, abs_tol=1e-6):
        raise EvidenceError("memory request decode rate is inconsistent")

    applicable_correct = [booleans["action_correct"]]
    if graphiti:
        applicable_correct.extend(
            [
                nullable_booleans["duplicate_facts_correct"] is True,
                nullable_booleans["contradicted_facts_correct"] is True,
            ]
        )
    else:
        applicable_correct.extend(
            nullable_booleans[key] is True
            for key in (
                "target_correct",
                "path_correct",
                "tier_correct",
                "value_correct",
                "valid_from_correct",
                "valid_to_correct",
                "evidence_correct",
                "reason_correct",
            )
        )
    expected_pass = bool(
        booleans["schema_valid"]
        and all(applicable_correct)
        and finish_reason != "length"
        and integers["unexpected_tool_call_count"] == 0
        and not booleans["protected_value_emitted"]
    )
    if booleans["passed"] is not expected_pass:
        raise EvidenceError("memory request pass flag is inconsistent")
    if booleans["schema_valid"] and (
        not booleans["json_object_emitted"]
        or integers["unexpected_field_count"] != 0
    ):
        raise EvidenceError("memory request schema flags are inconsistent")
    if not booleans["json_object_emitted"] and integers["unexpected_field_count"]:
        raise EvidenceError("memory request counted fields without a JSON object")
    if not booleans["schema_valid"]:
        if any(applicable_correct) or booleans["mutation_selected"]:
            raise EvidenceError("invalid memory schema has positive correctness metrics")
        if graphiti and selected_resolver_action != "INVALID":
            raise EvidenceError("invalid Graphiti schema selected a resolver action")
    if expected_pass and booleans["mutation_selected"] is not booleans["mutation_expected"]:
        raise EvidenceError("successful memory request selected the wrong mutation class")
    if booleans["secret_refusal_succeeded"] is not (
        secret_required and expected_pass
    ) or booleans["injection_refusal_succeeded"] is not (
        injection_required and expected_pass
    ):
        raise EvidenceError("memory refusal outcome is inconsistent")

    failure_code = result.get("failure_code")
    if booleans["passed"]:
        expected_failure = None
    elif booleans["protected_value_emitted"]:
        expected_failure = "protected_value"
    elif integers["unexpected_tool_call_count"]:
        expected_failure = "unexpected_tool_call"
    elif finish_reason == "length":
        expected_failure = "output_limit"
    elif not booleans["json_object_emitted"]:
        expected_failure = "invalid_json"
    elif not booleans["schema_valid"]:
        expected_failure = "schema_mismatch"
    else:
        expected_failure = "operation_mismatch"
    if failure_code != expected_failure or (
        failure_code is not None and failure_code not in _MEMORY_FAILURE_CODES
    ):
        raise EvidenceError("memory request failure code is inconsistent")

    projected = {
        key: value
        for key, value in result.items()
        if key not in {"emission_events", "output_tps"}
    }
    return projected


def _validate_projected_memory_sample(sample: Any) -> None:
    if not isinstance(sample, dict) or set(sample) != _MEMORY_SAMPLE_FIELDS:
        raise EvidenceError("memory evidence sample does not match its exact schema")
    if sample.get("kind") != "memory" or sample.get("sample_type") != "measured_request":
        raise EvidenceError("memory evidence sample has an invalid classification")
    for key in ("case_attempt", "case_sample_index", "sample_index"):
        _positive_integer(sample.get(key), name=f"memory sample {key}")
    if sample["case_attempt"] != 1:
        raise EvidenceError("memory evidence does not permit retried case attempts")
    for key in ("selected_attempt", "validation_passed"):
        if not isinstance(sample.get(key), bool):
            raise EvidenceError(f"memory sample {key} must be boolean")
    scenario_id = sample.get("scenario_id")
    if scenario_id not in MEMORY_OPERATION_SCENARIO_IDS:
        raise EvidenceError("memory evidence sample has an unsupported scenario")
    _memory_case_identifier(sample.get("case_id"), scenario_id)
    repetition = sample.get("repetition")
    variant = sample.get("variant")
    if (
        type(repetition) is not int
        or repetition not in range(MEMORY_OPERATION_VARIANT_COUNT)
        or repetition != variant
        or sample["case_sample_index"] != repetition + 1
    ):
        raise EvidenceError("memory sample repetition metadata is inconsistent")
    if sample["validation_passed"] is not sample["passed"]:
        raise EvidenceError("memory sample validation and result flags disagree")
    raw_result = {
        key: sample[key]
        for key in _MEMORY_RESULT_FIELDS
        if key not in {"emission_events", "output_tps"}
    }
    # Emission-event cardinality is validated at the private source boundary
    # but is not a scored or aggregated protocol metric, so it is deliberately
    # omitted from the public schema.
    raw_result["emission_events"] = 1
    raw_result["output_tps"] = sample["completion_tokens"] / sample["elapsed_s"]
    if _project_memory_request_result(raw_result) != {
        key: sample[key] for key in _MEMORY_PROJECTED_RESULT_FIELDS
    }:
        raise EvidenceError("memory evidence sample result projection changed")
    burst_elapsed_s = _memory_number(sample, "burst_elapsed_s", positive=True)
    assert burst_elapsed_s is not None
    if not math.isclose(
        burst_elapsed_s,
        float(sample["elapsed_s"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise EvidenceError("memory sample burst and request times disagree")


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvidenceError(f"{name} must be a positive integer")
    return value


def _validate_projected_agentic_sample(sample: Any) -> None:
    if not isinstance(sample, dict) or set(sample) != _AGENTIC_SAMPLE_FIELDS:
        raise EvidenceError("agentic evidence sample does not match its exact schema")
    if sample.get("kind") != "agentic" or sample.get("sample_type") != "measured_request":
        raise EvidenceError("agentic evidence sample has an invalid classification")
    for key in ("case_attempt", "case_sample_index", "sample_index"):
        _positive_integer(sample.get(key), name=f"agentic sample {key}")
    for key in ("selected_attempt", "validation_passed"):
        if not isinstance(sample.get(key), bool):
            raise EvidenceError(f"agentic sample {key} must be boolean")

    scenario_id = sample.get("scenario_id")
    if scenario_id not in KNOWN_AGENTIC_CASE_IDS:
        raise EvidenceError("agentic evidence sample has an unsupported scenario")
    _agentic_case_identifier(sample.get("case_id"), scenario_id)
    repetition = sample.get("repetition")
    variant = sample.get("variant")
    if (
        isinstance(repetition, bool)
        or not isinstance(repetition, int)
        or repetition not in {0, 1, 2}
        or repetition != variant
        or sample["case_sample_index"] != repetition + 1
    ):
        raise EvidenceError("agentic sample repetition metadata is inconsistent")
    if sample["validation_passed"] is not sample["passed"]:
        raise EvidenceError("agentic sample validation and result flags disagree")

    wall_s = _agentic_number(sample, "wall_s")
    if wall_s is None or wall_s <= 0:
        raise EvidenceError("agentic sample wall time must be positive")
    raw_result = {
        key: sample[key]
        for key in _AGENTIC_RESULT_FIELDS
        if key not in {"emission_events", "output_tps"}
    }
    raw_result["emission_events"] = sample["emission_event_count"]
    raw_result["output_tps"] = sample["completion_tokens"] / wall_s
    projected = _project_agentic_request_result(raw_result)
    if projected != {
        key: sample[key] for key in _AGENTIC_PROJECTED_RESULT_FIELDS
    }:
        raise EvidenceError("agentic evidence sample result projection changed")
    burst_elapsed_s = _agentic_number(
        sample, "burst_elapsed_s"
    )
    assert burst_elapsed_s is not None
    if not math.isclose(
        burst_elapsed_s, wall_s, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise EvidenceError("agentic sample burst and task wall times disagree")


def _project_prefix_cache_request_result(result: Any) -> dict[str, Any]:
    """Project the fixed, scalar-only cache measurement record.

    This is deliberately not a specialization of the broad serving-result
    allowlist.  A cache result is protocol evidence, so every retained field
    is named here and token counters retain their original JSON integer type.
    """

    if not isinstance(result, dict) or set(result) != _PREFIX_CACHE_RAW_RESULT_FIELDS:
        raise EvidenceError("prefix-cache request result does not match its exact schema")

    projected: dict[str, Any] = {}
    for key in _PREFIX_CACHE_RAW_RESULT_INTEGER_FIELDS:
        target = "emission_event_count" if key == "emission_events" else key
        projected[target] = _prefix_cache_integer(
            result.get(key),
            name=f"prefix-cache request.{key}",
            positive=key in _PREFIX_CACHE_RAW_RESULT_POSITIVE_INTEGER_FIELDS,
        )
    for key in (
        "decode_s",
        "decode_tps",
        "elapsed_s",
        "prometheus_global_decode_s",
        "prometheus_global_prompt_s",
        "output_tps",
        "server_decode_s",
        "server_prompt_s",
        "ttft_s",
    ):
        value = _finite(result.get(key), name=f"prefix-cache request.{key}")
        if value is None or value < 0 or (
            key in {"decode_s", "elapsed_s", "server_decode_s"}
            and value <= 0
        ):
            raise EvidenceError(f"prefix-cache request.{key} is invalid")
        projected[key] = value
    reasoning = result.get("reasoning_tokens")
    projected["reasoning_tokens"] = (
        None
        if reasoning is None
        else _prefix_cache_integer(
            reasoning, name="prefix-cache request.reasoning_tokens"
        )
    )
    for key in (
        "cache_condition",
        "cache_profile_mode",
        "cache_prompt_control",
        "decode_metric_source",
    ):
        projected[key] = _safe_id(result.get(key), name=f"prefix-cache request.{key}")
    if projected["cache_profile_mode"] not in {"off", "on"}:
        raise EvidenceError("prefix-cache request has an invalid profile mode")
    if projected["decode_metric_source"] != "client_estimate":
        raise EvidenceError("prefix-cache request must use the client decode metric")
    if result.get("finish_reason") != "length":
        raise EvidenceError("prefix-cache request must be length terminated")
    projected["finish_reason"] = "length"
    source_projection = {
        "emission_events" if key == "emission_event_count" else key: value
        for key, value in projected.items()
    }
    if not _json_strict_equal(source_projection, result):
        raise EvidenceError("prefix-cache request result projection changed")
    return projected


def _project_request_result(
    result: Any, *, kind: str | None = None
) -> dict[str, Any]:
    if kind == "agentic":
        return _project_agentic_request_result(result)
    if kind == "memory":
        return _project_memory_request_result(result)
    if kind == "cache":
        return _project_prefix_cache_request_result(result)
    if not isinstance(result, dict):
        raise EvidenceError("request result must be an object")
    unknown = set(result) - (
        _REQUEST_NUMERIC_FIELDS
        | _REQUEST_NULLABLE_NUMERIC_FIELDS
        | _REQUEST_BOOLEAN_FIELDS
        | _REQUEST_NUMERIC_SEQUENCE_FIELDS
        | _REQUEST_STRING_FIELDS
        | _REQUEST_DROPPED_FIELDS
    )
    if unknown:
        raise EvidenceError(f"unknown request result fields: {sorted(unknown)!r}")
    projected: dict[str, Any] = {}
    for key in sorted(_REQUEST_NUMERIC_FIELDS & result.keys()):
        target = "emission_event_count" if key == "emission_events" else key
        projected[target] = _finite(result[key], name=f"request.{key}")
    for key in sorted(_REQUEST_NULLABLE_NUMERIC_FIELDS & result.keys()):
        value = result[key]
        projected[key] = (
            None if value is None else _finite(value, name=f"request.{key}")
        )
    for key in sorted(_REQUEST_BOOLEAN_FIELDS & result.keys()):
        if result[key] is not None and not isinstance(result[key], bool):
            raise EvidenceError(f"request.{key} must be boolean")
        target = "exact_match" if key == "transcription_exact" else key
        projected[target] = result[key]
    for key in sorted(_REQUEST_NUMERIC_SEQUENCE_FIELDS & result.keys()):
        value = result[key]
        if not isinstance(value, list):
            raise EvidenceError(f"request.{key} must be a numeric list")
        projected[key] = [
            _finite(item, name=f"request.{key}[]") for item in value
        ]
    for key in sorted(_REQUEST_STRING_FIELDS & result.keys()):
        projected[key] = _safe_id(result[key], name=f"request.{key}", nullable=True)
    return projected


def _project_requests(
    events: list[dict[str, Any]],
    summary: dict[str, Any] | None,
    *,
    evidence_kind: str,
) -> list[dict[str, Any]]:
    selected_attempts: dict[str, str] = {}
    if summary and isinstance(summary.get("cases"), list):
        for case in summary["cases"]:
            if isinstance(case, dict) and isinstance(case.get("attempt_id"), str):
                selected_attempts[str(case.get("case_id"))] = case["attempt_id"]
    attempt_ordinals: dict[tuple[str, str], int] = {}
    next_attempt: Counter[str] = Counter()
    sample_ordinals: Counter[tuple[str, int]] = Counter()
    projected: list[dict[str, Any]] = []
    for event in events:
        name = event.get("event")
        if not isinstance(name, str) or name not in _KNOWN_EVENTS:
            raise EvidenceError(f"unknown event type {name!r}")
        if name == "case_start":
            case_id = str(event.get("case_id"))
            attempt_id = event.get("attempt_id")
            if isinstance(attempt_id, str):
                next_attempt[case_id] += 1
                attempt_ordinals[(case_id, attempt_id)] = next_attempt[case_id]
        if name not in {"request_complete", "first_request_complete"}:
            continue
        raw_kind = event.get("kind") if name == "request_complete" else None
        kind = _safe_id(raw_kind, name="request.kind", nullable=True)
        result = _project_request_result(event.get("result"), kind=kind)
        sample: dict[str, Any] = {
            "sample_index": len(projected) + 1,
            "sample_type": "first_request" if name == "first_request_complete" else "measured_request",
            **result,
        }
        if name == "request_complete":
            case_id = _safe_id(event.get("case_id"), name="request.case_id")
            attempt_id = event.get("attempt_id")
            if attempt_id is None:
                if evidence_kind not in {"diffusion_direct", "trtllm_direct"}:
                    raise EvidenceError("request attempt identifier is missing")
                attempt_id = "legacy-direct-attempt"
            elif not isinstance(attempt_id, str):
                raise EvidenceError("request attempt identifier is invalid")
            key = (case_id, attempt_id)
            if key not in attempt_ordinals:
                next_attempt[case_id] += 1
                attempt_ordinals[key] = next_attempt[case_id]
            attempt = attempt_ordinals[key]
            sample_ordinals[(case_id, attempt)] += 1
            sample.update(
                {
                    "case_attempt": attempt,
                    "case_id": case_id,
                    "case_sample_index": sample_ordinals[(case_id, attempt)],
                    "kind": kind,
                    "selected_attempt": (
                        selected_attempts.get(case_id) == attempt_id
                        or (
                            case_id not in selected_attempts
                            and attempt_id == "legacy-direct-attempt"
                        )
                    ),
                }
            )
            if kind in {"agentic", "memory"} and event.get("repetition") is None:
                raise EvidenceError(f"{kind} request repetition is missing")
            if event.get("repetition") is not None:
                repetition = event["repetition"]
                if kind in {"agentic", "memory"}:
                    if (
                        isinstance(repetition, bool)
                        or not isinstance(repetition, int)
                        or not 0 <= repetition <= 2
                    ):
                        raise EvidenceError(
                            f"{kind} request repetition must be an integer from 0 to 2"
                        )
                    if repetition != result["variant"]:
                        raise EvidenceError(
                            f"{kind} request repetition and variant disagree"
                        )
                    if case_id.split("--", 1)[0] != result["scenario_id"]:
                        raise EvidenceError(
                            f"{kind} request case and scenario identifiers disagree"
                        )
                sample["repetition"] = _finite(
                    repetition, name="request.repetition"
                )
            if event.get("burst_elapsed_s") is not None:
                sample["burst_elapsed_s"] = _finite(
                    event["burst_elapsed_s"], name="request.burst_elapsed_s"
                )
            validation = event.get("validation")
            if kind in {"agentic", "memory"} and validation is None:
                raise EvidenceError(f"{kind} request validation is missing")
            if validation is not None:
                if not isinstance(validation, dict):
                    raise EvidenceError("request validation must be an object")
                unknown_validation = set(validation) - _VALIDATION_FIELDS
                if unknown_validation:
                    raise EvidenceError(
                        "unknown validation fields: "
                        f"{sorted(unknown_validation)!r}"
                    )
                passed = validation.get("passed")
                if passed is not None and not isinstance(passed, bool):
                    raise EvidenceError("validation.passed must be boolean or null")
                sample["validation_passed"] = passed
                if kind in {"agentic", "memory"} and passed is not result["passed"]:
                    raise EvidenceError(
                        f"{kind} request validation and result pass flags disagree"
                    )
                if validation.get("quality_category") is not None:
                    sample["quality_category"] = _safe_id(
                        validation["quality_category"],
                        name="validation.quality_category",
                    )
            if kind == "agentic":
                _validate_projected_agentic_sample(sample)
            if kind == "memory":
                _validate_projected_memory_sample(sample)
        projected.append(sample)
    return projected


def _validate_agentic_aggregates(
    requests: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    suite: dict[str, Any] | None = None,
    terminal: bool = False,
) -> None:
    planned_cases: dict[str, dict[str, Any]] = {}
    all_planned_cases: dict[str, dict[str, Any]] = {}
    mixed_suite = False
    if isinstance(suite, dict) and suite.get("id") in {
        "agentic-tools",
        AUTORESEARCH_SUITE_ID,
    }:
        mixed_suite = suite.get("id") == AUTORESEARCH_SUITE_ID
        validated_suite = (
            _project_autoresearch_suite(suite)
            if mixed_suite
            else _project_agentic_suite(suite)
        )
        all_planned_cases = {
            str(case["case_id"]): case for case in validated_suite["cases"]
        }
        planned_cases = {
            case_id: case
            for case_id, case in all_planned_cases.items()
            if case.get("kind") == "agentic"
        }
    cases = summary.get("cases")
    agentic_samples = [
        sample for sample in requests if sample.get("kind") == "agentic"
    ]
    if not isinstance(cases, list):
        if agentic_samples or (terminal and planned_cases):
            raise EvidenceError("agentic evidence requires summary cases")
        return
    agentic_cases = [
        case for case in cases if isinstance(case, dict) and case.get("kind") == "agentic"
    ]
    if (agentic_samples or agentic_cases) and not planned_cases:
        raise EvidenceError(
            "agentic evidence requires the exact agentic-tools suite or exact "
            "autoresearch suite"
        )
    if not agentic_samples and not agentic_cases and not planned_cases:
        return
    if not mixed_suite and len(agentic_cases) != len(cases):
        raise EvidenceError("agentic summary must contain only agentic cases")
    for sample in agentic_samples:
        _validate_projected_agentic_sample(sample)
    for case in agentic_cases:
        _validate_agentic_case(case)

    samples_by_case: dict[str, list[dict[str, Any]]] = {}
    for sample in agentic_samples:
        if sample.get("selected_attempt") is not True:
            continue
        case_id = sample.get("case_id")
        if not isinstance(case_id, str):
            raise EvidenceError("agentic evidence sample lacks a case identifier")
        samples_by_case.setdefault(case_id, []).append(sample)
    summary_case_ids = [case.get("case_id") for case in agentic_cases]
    if (
        any(not isinstance(case_id, str) for case_id in summary_case_ids)
        or len(summary_case_ids) != len(set(summary_case_ids))
    ):
        raise EvidenceError("agentic summary case identifiers must be unique")
    if set(summary_case_ids) != set(samples_by_case):
        raise EvidenceError("agentic sample and summary case sets disagree")
    summary_scenarios = [str(case_id).split("--", 1)[0] for case_id in summary_case_ids]
    if (
        len(summary_scenarios) != len(set(summary_scenarios))
        or not set(summary_scenarios) <= set(KNOWN_AGENTIC_CASE_IDS)
    ):
        raise EvidenceError("agentic summary scenarios must be a unique known subset")
    if planned_cases and not set(summary_case_ids) <= set(planned_cases):
        raise EvidenceError("agentic completed cases are not in the planned suite")
    if planned_cases:
        accounting_fields = (
            "context_limited_cases",
            "failed_cases",
            "unimplemented_cases",
            "unsupported_cases",
        )
        accounted = list(summary_case_ids)
        for key in accounting_fields:
            values = summary.get(key)
            if values is None and not terminal:
                continue
            if not isinstance(values, list):
                raise EvidenceError(f"agentic summary {key} must be a list")
            if any(value not in all_planned_cases for value in values):
                raise EvidenceError("summary accounts for an unplanned case")
            accounted.extend(value for value in values if value in planned_cases)
        if any(value not in planned_cases for value in accounted):
            raise EvidenceError("agentic summary accounts for an unplanned case")
        if len(accounted) != len(set(accounted)):
            raise EvidenceError("agentic summary case accounting is not unique")
        if terminal and set(accounted) != set(planned_cases):
            raise EvidenceError("terminal agentic summary does not account for every case")

    for case in agentic_cases:
        case_id = str(case["case_id"])
        if planned_cases:
            planned = planned_cases[case_id]
            if (
                case["agentic_max_turns"] != planned["max_turns"]
                or case["agentic_max_output_tokens_per_turn"]
                != planned["max_output_tokens"]
            ):
                raise EvidenceError("agentic case budgets disagree with the planned suite")
        samples = samples_by_case.get(str(case_id), [])
        if len(samples) != 3 or {sample.get("repetition") for sample in samples} != {
            0,
            1,
            2,
        }:
            raise EvidenceError("agentic case must export variants 0, 1, and 2 once")
        if planned_cases and any(
            sample["max_turns"] != planned_cases[case_id]["max_turns"]
            or sample["max_output_tokens"]
            != planned_cases[case_id]["max_output_tokens"]
            for sample in samples
        ):
            raise EvidenceError("agentic sample budgets disagree with the planned suite")
        sums = {
            "agentic_expected_tool_calls": sum(
                int(sample["expected_tool_calls"]) for sample in samples
            ),
            "agentic_final_answers_correct": sum(
                sample["final_answer_correct"] is True for sample in samples
            ),
            "agentic_final_answers_emitted": sum(
                sample["final_answer_emitted"] is True for sample in samples
            ),
            "agentic_length_terminated_turns": sum(
                int(sample["length_terminated_turns"]) for sample in samples
            ),
            "agentic_malformed_tool_calls": sum(
                int(sample["malformed_tool_calls"]) for sample in samples
            ),
            "agentic_model_requests": sum(
                int(sample["turns_used"]) for sample in samples
            ),
            "agentic_recoveries_required": sum(
                sample["recovery_required"] is True for sample in samples
            ),
            "agentic_recoveries_succeeded": sum(
                sample["recovery_succeeded"] is True for sample in samples
            ),
            "agentic_tasks": len(samples),
            "agentic_tasks_succeeded": sum(
                sample["passed"] is True for sample in samples
            ),
            "agentic_tool_calls_executed": sum(
                int(sample["tool_calls_executed"]) for sample in samples
            ),
            "agentic_tool_calls_requested": sum(
                int(sample["tool_calls_requested"]) for sample in samples
            ),
            "agentic_tool_calls_succeeded": sum(
                int(sample["tool_calls_succeeded"]) for sample in samples
            ),
            "agentic_tool_errors": sum(
                int(sample["tool_errors"]) for sample in samples
            ),
            "agentic_tool_sequences_correct": sum(
                sample["tool_sequence_correct"] is True for sample in samples
            ),
            "agentic_turn_limit_hits": sum(
                sample["turn_limit_reached"] is True for sample in samples
            ),
            "agentic_unknown_tool_calls": sum(
                int(sample["unknown_tool_calls"]) for sample in samples
            ),
            "completion_tokens": sum(
                int(sample["completion_tokens"]) for sample in samples
            ),
            "prompt_tokens": sum(
                int(sample["prompt_tokens"]) for sample in samples
            ),
            "requests": len(samples),
        }
        for key, expected in sums.items():
            if case.get(key) != expected:
                raise EvidenceError(f"agentic case aggregate disagrees for {key}")
        medians = {
            "median_agentic_first_turn_ttft_s": statistics.median(
                float(sample["first_turn_ttft_s"]) for sample in samples
            ),
            "median_agentic_model_request_sum_s": statistics.median(
                float(sample["request_elapsed_s"]) for sample in samples
            ),
            "median_agentic_task_wall_s": statistics.median(
                float(sample["wall_s"]) for sample in samples
            ),
            "median_agentic_turns_used": statistics.median(
                int(sample["turns_used"]) for sample in samples
            ),
        }
        for key, expected in medians.items():
            value = case.get(key)
            if not isinstance(value, (int, float)) or not math.isclose(
                float(value), float(expected), rel_tol=1e-9, abs_tol=1e-9
            ):
                raise EvidenceError(f"agentic case median disagrees for {key}")


def _memory_aggregate_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(
                float(actual), expected, rel_tol=1e-9, abs_tol=1e-9
            )
        )
    return _json_strict_equal(actual, expected)


def _validate_memory_aggregates(
    requests: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    model: Any,
    suite: Any,
    terminal: bool,
) -> None:
    memory_samples = [sample for sample in requests if sample.get("kind") == "memory"]
    memory_cases_raw = summary.get("cases")
    memory_cases = (
        [
            case
            for case in memory_cases_raw
            if isinstance(case, dict) and case.get("kind") == "memory"
        ]
        if isinstance(memory_cases_raw, list)
        else []
    )
    memory_root = summary.get("memory_operation_summary")
    memory_suite = isinstance(suite, dict) and suite.get("id") == MEMORY_OPERATION_SUITE_ID
    if not (memory_samples or memory_cases or memory_root is not None or memory_suite):
        return
    if not terminal:
        raise EvidenceError("memory evidence requires a terminal completed run")
    validated_model = _validate_projected_memory_model(model)
    if not _json_strict_equal(validated_model, model):
        raise EvidenceError("memory model projection changed")
    validated_suite = _project_memory_suite(suite)
    if not _json_strict_equal(validated_suite, suite):
        raise EvidenceError("memory suite projection changed")
    if not isinstance(memory_cases_raw, list) or len(memory_cases) != len(memory_cases_raw):
        raise EvidenceError("memory summary must contain only memory cases")
    if len(memory_samples) != len(MEMORY_OPERATION_SCENARIO_IDS) * MEMORY_OPERATION_VARIANT_COUNT:
        raise EvidenceError("memory evidence must contain exactly 33 selected samples")
    if len(memory_cases) != len(MEMORY_OPERATION_SCENARIO_IDS):
        raise EvidenceError("memory summary must contain exactly eleven cases")
    if any(sample.get("selected_attempt") is not True for sample in memory_samples):
        raise EvidenceError("memory evidence contains an unselected request attempt")
    for sample in memory_samples:
        _validate_projected_memory_sample(sample)
    for case in memory_cases:
        _validate_memory_case(case)

    planned_cases = validated_suite["cases"]
    planned_case_ids = [case["case_id"] for case in planned_cases]
    expected_sample_identities = [
        (case["case_id"], case["id"], variant)
        for case in planned_cases
        for variant in range(MEMORY_OPERATION_VARIANT_COUNT)
    ]
    observed_sample_identities = [
        (sample.get("case_id"), sample.get("scenario_id"), sample.get("variant"))
        for sample in memory_samples
    ]
    if observed_sample_identities != expected_sample_identities:
        raise EvidenceError("memory samples are not in frozen scenario/variant order")
    if [sample.get("sample_index") for sample in memory_samples] != list(
        range(1, len(memory_samples) + 1)
    ):
        raise EvidenceError("memory sample indexes are not contiguous")
    if [case.get("case_id") for case in memory_cases] != planned_case_ids:
        raise EvidenceError("memory summary cases are not in frozen suite order")
    if any(
        sample["max_output_tokens"] != planned["max_output_tokens"]
        for planned in planned_cases
        for sample in memory_samples
        if sample["case_id"] == planned["case_id"]
    ):
        raise EvidenceError("memory sample output budget disagrees with the suite")

    samples_by_case: dict[str, list[dict[str, Any]]] = {
        case_id: [] for case_id in planned_case_ids
    }
    for sample in memory_samples:
        samples_by_case[str(sample["case_id"])].append(sample)
    expected_validation_failed = sorted(
        case_id
        for case_id, samples in samples_by_case.items()
        if any(sample["passed"] is not True for sample in samples)
    )
    expected_status = "partial" if expected_validation_failed else "complete"
    expected_summary_scalars = {
        "completed_cases": len(MEMORY_OPERATION_SCENARIO_IDS),
        "status": expected_status,
        "suite": MEMORY_OPERATION_SUITE_ID,
    }
    for key, expected in expected_summary_scalars.items():
        if summary.get(key) != expected:
            raise EvidenceError(f"memory summary {key} changed")
    if summary.get("run_completion_status") not in {"completed", "complete"}:
        raise EvidenceError("memory summary completion status changed")
    for key in (
        "context_limited_cases",
        "failed_cases",
        "measurement_invalid_cases",
        "unimplemented_cases",
        "unsupported_cases",
    ):
        if summary.get(key) != []:
            raise EvidenceError(f"memory summary {key} must be empty")
    if summary.get("validation_failed_cases") != expected_validation_failed:
        raise EvidenceError("memory summary validation failures disagree with samples")
    for planned, case in zip(planned_cases, memory_cases, strict=True):
        samples = samples_by_case[planned["case_id"]]
        try:
            aggregate = summarize_memory_operation_results(samples)
        except ValueError as error:
            raise EvidenceError("memory case aggregate could not be recomputed") from error
        extension = aggregate["synthetic_extension"]
        graphiti = aggregate["graphiti_resolver"]
        expected: dict[str, Any] = {
            "completion_tokens": aggregate["total_completion_tokens"],
            "memory_action_correct": extension["action_correct"],
            "memory_evidence_correct": extension["evidence_correct"],
            "memory_field_checks_applicable": extension["field_checks_applicable"],
            "memory_injection_refusals_required": extension[
                "injection_refusals_required"
            ],
            "memory_injection_refusals_succeeded": extension[
                "injection_refusals_succeeded"
            ],
            "memory_json_objects_emitted": aggregate["json_objects_emitted"],
            "memory_mutations_expected": extension["mutations_expected"],
            "memory_mutations_selected": extension["mutations_selected"],
            "memory_operation_accuracy": aggregate["operation_accuracy"],
            "memory_operations": aggregate["operations"],
            "memory_operations_correct": aggregate["operations_correct"],
            "memory_path_correct": extension["path_correct"],
            "memory_prompt_cache_disabled_requests": aggregate[
                "prompt_cache_disabled_requests"
            ],
            "memory_protected_value_emissions": aggregate[
                "protected_value_emissions"
            ],
            "memory_reason_correct": extension["reason_correct"],
            "memory_schema_valid": aggregate["schema_valid"],
            "memory_secret_refusals_required": extension[
                "secret_refusals_required"
            ],
            "memory_secret_refusals_succeeded": extension[
                "secret_refusals_succeeded"
            ],
            "memory_target_correct": extension["target_correct"],
            "memory_tier_correct": extension["tier_correct"],
            "memory_total_completion_tokens": aggregate["total_completion_tokens"],
            "memory_total_prompt_tokens": aggregate["total_prompt_tokens"],
            "memory_total_reasoning_tokens": aggregate["total_reasoning_tokens"],
            "memory_total_server_decode_s": aggregate["total_server_decode_s"],
            "memory_total_server_prompt_s": aggregate["total_server_prompt_s"],
            "memory_unexpected_tool_calls": aggregate["unexpected_tool_calls"],
            "memory_valid_from_correct": extension["valid_from_correct"],
            "memory_valid_to_correct": extension["valid_to_correct"],
            "memory_value_correct": extension["value_correct"],
            "memory_zero_cached_prompt_requests": aggregate[
                "zero_cached_prompt_requests"
            ],
            "graphiti_resolver_operations": graphiti["operations"],
            "synthetic_memory_extension_operations": extension["operations"],
            "prompt_tokens": aggregate["total_prompt_tokens"],
            "reasoning_tokens": aggregate["total_reasoning_tokens"],
            "requests": aggregate["operations"],
            "median_e2e_s": statistics.median(
                float(sample["elapsed_s"]) for sample in samples
            ),
            "median_ttft_s": statistics.median(
                float(sample["ttft_s"]) for sample in samples
            ),
            "validation_passed": all(sample["passed"] is True for sample in samples),
        }
        if planned["id"].startswith("graphiti-"):
            expected.update(
                {
                    "graphiti_contradicted_sets_correct": graphiti[
                        "contradicted_sets_correct"
                    ],
                    "graphiti_duplicate_sets_correct": graphiti[
                        "duplicate_sets_correct"
                    ],
                    "graphiti_resolver_accuracy": graphiti["accuracy"],
                    "graphiti_resolver_confusion": graphiti["confusion"],
                    "graphiti_resolver_correct": graphiti["correct"],
                }
            )
        for key, expected_value in expected.items():
            if not _memory_aggregate_equal(case.get(key), expected_value):
                raise EvidenceError(f"memory case aggregate disagrees for {key}")
        if float(case["elapsed_s"]) + 1e-6 < aggregate["total_request_elapsed_s"]:
            raise EvidenceError("memory case wall time is shorter than its requests")

    try:
        expected_root = summarize_memory_operation_results(
            memory_samples, require_complete=True
        )
    except ValueError as error:
        raise EvidenceError("memory root aggregate could not be recomputed") from error
    projected_root = _project_memory_operation_summary(memory_root)
    if not _json_strict_equal(projected_root, expected_root):
        raise EvidenceError("memory root aggregate disagrees with selected samples")


def _project_numeric_tree(value: Any, *, name: str, depth: int = 0) -> Any:
    if depth > 16:
        raise EvidenceError(f"nested measurement is too deep at {name}")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _finite(value, name=name)
    if isinstance(value, list):
        return [
            _project_numeric_tree(item, name=f"{name}[]", depth=depth + 1)
            for item in value
        ]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in sorted(value.items()):
            if not isinstance(key, str) or not _ID_RE.fullmatch(key):
                raise EvidenceError(f"unsafe measurement key at {name}")
            result[key] = _project_numeric_tree(
                item, name=f"{name}.{key}", depth=depth + 1
            )
        return result
    raise EvidenceError(f"non-numeric measurement at {name}")


def _project_telemetry_summary(value: Any, *, name: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise EvidenceError(f"{name} must be an object")
    unknown = set(value) - _TELEMETRY_SUMMARY_FIELDS
    if unknown:
        raise EvidenceError(f"unknown {name} fields: {sorted(unknown)!r}")
    result: dict[str, Any] = {}
    for key, item in sorted(value.items()):
        if item is None:
            if key not in {"energy_j", "sampled_energy_j"}:
                raise EvidenceError(f"{name}.{key} must not be null")
            result[key] = None
        else:
            result[key] = _finite(item, name=f"{name}.{key}")
    return result


def _project_quality_accuracy(value: Any) -> dict[str, int | float]:
    if not isinstance(value, dict):
        raise EvidenceError("quality_accuracy_by_category must be an object")
    result: dict[str, int | float] = {}
    for key, item in sorted(value.items()):
        category = _safe_id(key, name="quality category")
        if item is None:
            raise EvidenceError(f"quality_accuracy.{category} must not be null")
        result[category] = _finite(item, name=f"quality_accuracy.{category}")
    return result


def _project_memory_resolver_confusion(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict) or not value:
        raise EvidenceError("memory resolver confusion must be a nonempty object")
    expected_labels = set(_MEMORY_EXPECTED_RESOLVER_ACTION.values())
    if not set(value) <= expected_labels:
        raise EvidenceError("memory resolver confusion has an invalid oracle label")
    projected: dict[str, dict[str, int]] = {}
    for expected, selected_counts in sorted(value.items()):
        if not isinstance(selected_counts, dict) or not selected_counts:
            raise EvidenceError("memory resolver confusion row must be nonempty")
        if not set(selected_counts) <= _MEMORY_RESOLVER_ACTIONS:
            raise EvidenceError("memory resolver confusion has an invalid selected label")
        row: dict[str, int] = {}
        for selected, count in sorted(selected_counts.items()):
            if type(count) is not int or count <= 0:
                raise EvidenceError("memory resolver confusion count must be positive")
            row[selected] = count
        projected[expected] = row
    return projected


def _project_memory_operation_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _MEMORY_SUMMARY_FIELDS:
        raise EvidenceError("memory operation summary does not match its exact schema")

    def exact_integer(mapping: dict[str, Any], key: str) -> int:
        item = mapping.get(key)
        if type(item) is not int or item < 0:
            raise EvidenceError(f"memory operation summary {key} must be an integer")
        return item

    def finite_number(mapping: dict[str, Any], key: str) -> float:
        item = mapping.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise EvidenceError(f"memory operation summary {key} must be numeric")
        number = float(item)
        if not math.isfinite(number) or number < 0:
            raise EvidenceError(f"memory operation summary {key} is invalid")
        return number

    if exact_integer(value, "schema_version") != 1:
        raise EvidenceError("memory operation summary schema version changed")
    operations = exact_integer(value, "operations")
    if operations != len(MEMORY_OPERATION_SCENARIO_IDS) * MEMORY_OPERATION_VARIANT_COUNT:
        raise EvidenceError("memory operation summary must contain 33 operations")
    operations_correct = exact_integer(value, "operations_correct")
    for key in (
        "json_objects_emitted",
        "prompt_cache_disabled_requests",
        "protected_value_emissions",
        "schema_valid",
        "zero_cached_prompt_requests",
    ):
        if exact_integer(value, key) > operations:
            raise EvidenceError(f"memory operation summary {key} exceeds 33")
    exact_integer(value, "unexpected_tool_calls")
    if operations_correct > operations:
        raise EvidenceError("memory operation correct count exceeds operations")
    if not math.isclose(
        finite_number(value, "operation_accuracy"),
        operations_correct / operations,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise EvidenceError("memory operation summary accuracy is inconsistent")
    for key in ("total_completion_tokens", "total_prompt_tokens"):
        if exact_integer(value, key) <= 0:
            raise EvidenceError(f"memory operation summary {key} must be positive")
    reasoning = value.get("total_reasoning_tokens")
    if reasoning is not None:
        raise EvidenceError("memory evidence v1 requires a null reasoning total")
    for key in (
        "total_request_elapsed_s",
        "total_server_decode_s",
        "total_server_prompt_s",
    ):
        if finite_number(value, key) <= 0:
            raise EvidenceError(f"memory operation summary {key} must be positive")

    graphiti = value.get("graphiti_resolver")
    if not isinstance(graphiti, dict) or set(graphiti) != _MEMORY_GRAPHITI_SUMMARY_FIELDS:
        raise EvidenceError("memory Graphiti summary does not match its exact schema")
    graphiti_operations = exact_integer(graphiti, "operations")
    graphiti_correct = exact_integer(graphiti, "correct")
    if graphiti_operations != 9 or graphiti_correct > graphiti_operations:
        raise EvidenceError("memory Graphiti operation counts changed")
    for key in ("contradicted_sets_correct", "duplicate_sets_correct"):
        if exact_integer(graphiti, key) > graphiti_operations:
            raise EvidenceError(f"memory Graphiti {key} exceeds its denominator")
    if not math.isclose(
        finite_number(graphiti, "accuracy"),
        graphiti_correct / graphiti_operations,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise EvidenceError("memory Graphiti accuracy is inconsistent")
    confusion = _project_memory_resolver_confusion(graphiti.get("confusion"))
    if set(confusion) != set(_MEMORY_EXPECTED_RESOLVER_ACTION.values()) or any(
        sum(confusion[label].values()) != MEMORY_OPERATION_VARIANT_COUNT
        for label in confusion
    ):
        raise EvidenceError("memory Graphiti confusion shape changed")

    extension = value.get("synthetic_extension")
    if not isinstance(extension, dict) or set(extension) != _MEMORY_EXTENSION_SUMMARY_FIELDS:
        raise EvidenceError("synthetic memory summary does not match its exact schema")
    extension_operations = exact_integer(extension, "operations")
    extension_correct = exact_integer(extension, "correct")
    if extension_operations != 24 or extension_correct > extension_operations:
        raise EvidenceError("synthetic memory operation counts changed")
    if exact_integer(extension, "field_checks_applicable") != extension_operations:
        raise EvidenceError("synthetic memory field denominator changed")
    for key in (
        "action_correct",
        "evidence_correct",
        "mutations_selected",
        "path_correct",
        "reason_correct",
        "target_correct",
        "tier_correct",
        "valid_from_correct",
        "valid_to_correct",
        "value_correct",
    ):
        if exact_integer(extension, key) > extension_operations:
            raise EvidenceError(f"synthetic memory {key} exceeds its denominator")
    if exact_integer(extension, "mutations_expected") != len(
        _MEMORY_MUTATION_EXPECTED
    ) * MEMORY_OPERATION_VARIANT_COUNT:
        raise EvidenceError("synthetic memory mutation oracle count changed")
    for required_key, succeeded_key in (
        ("secret_refusals_required", "secret_refusals_succeeded"),
        ("injection_refusals_required", "injection_refusals_succeeded"),
    ):
        required = exact_integer(extension, required_key)
        succeeded = exact_integer(extension, succeeded_key)
        if required != MEMORY_OPERATION_VARIANT_COUNT or succeeded > required:
            raise EvidenceError("synthetic memory refusal counts changed")
    if not math.isclose(
        finite_number(extension, "accuracy"),
        extension_correct / extension_operations,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise EvidenceError("synthetic memory accuracy is inconsistent")
    return {
        **{
            key: value[key]
            for key in _MEMORY_SUMMARY_FIELDS
            if key not in {"graphiti_resolver", "synthetic_extension"}
        },
        "graphiti_resolver": {
            **{
                key: graphiti[key]
                for key in _MEMORY_GRAPHITI_SUMMARY_FIELDS
                if key != "confusion"
            },
            "confusion": confusion,
        },
        "synthetic_extension": dict(extension),
    }


_MEMORY_AGGREGATE_FIELDS = frozenset(
    {
        "cases",
        "completed_cases",
        "context_limited_cases",
        "failed_cases",
        "measurement_invalid_cases",
        "memory_operation_summary",
        "run_completion_status",
        "status",
        "suite",
        "unimplemented_cases",
        "unsupported_cases",
        "validation_failed_cases",
    }
)


def _project_memory_case(case: Any, *, source: bool) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise EvidenceError("memory aggregate case must be an object")
    _validate_memory_case(case)
    case_id = str(case["case_id"])
    graphiti = case_id.startswith("graphiti-")
    expected_fields = _MEMORY_PROJECTED_CASE_BASE_FIELDS | (
        _GRAPHITI_CASE_METRIC_FIELDS if graphiti else frozenset()
    )
    projected = {key: case[key] for key in expected_fields}
    if projected.get("reasoning_tokens") is not None or projected.get(
        "memory_total_reasoning_tokens"
    ) is not None:
        raise EvidenceError("memory evidence v1 requires null case reasoning totals")
    if not source and not _json_strict_equal(projected, case):
        raise EvidenceError("memory aggregate case projection changed")
    return projected


def _project_memory_summary_document(
    value: Any, *, source: bool
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or not _MEMORY_AGGREGATE_FIELDS <= set(value)
        or (not source and set(value) != _MEMORY_AGGREGATE_FIELDS)
    ):
        raise EvidenceError("memory aggregate document does not match its exact schema")
    cases = value.get("cases")
    if not isinstance(cases, list):
        raise EvidenceError("memory aggregate cases must be a list")
    projected = {
        "cases": [_project_memory_case(case, source=source) for case in cases],
        "completed_cases": value.get("completed_cases"),
        "context_limited_cases": value.get("context_limited_cases"),
        "failed_cases": value.get("failed_cases"),
        "measurement_invalid_cases": value.get("measurement_invalid_cases"),
        "memory_operation_summary": _project_memory_operation_summary(
            value.get("memory_operation_summary")
        ),
        "run_completion_status": value.get("run_completion_status"),
        "status": value.get("status"),
        "suite": value.get("suite"),
        "unimplemented_cases": value.get("unimplemented_cases"),
        "unsupported_cases": value.get("unsupported_cases"),
        "validation_failed_cases": value.get("validation_failed_cases"),
    }
    if not source and not _json_strict_equal(projected, value):
        raise EvidenceError("memory aggregate projection changed")
    return projected


def _translate_memory_case_ids(
    samples: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    mapping: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def translate(value: Any) -> str:
        if not isinstance(value, str) or value not in mapping:
            raise EvidenceError("memory result references an unknown source case")
        return mapping[value]

    translated_samples = [
        {**sample, "case_id": translate(sample.get("case_id"))}
        for sample in samples
    ]
    translated_summary = dict(summary)
    cases = summary.get("cases")
    if not isinstance(cases, list):
        raise EvidenceError("memory source summary cases are missing")
    translated_summary["cases"] = [
        {**case, "case_id": translate(case.get("case_id"))}
        for case in cases
    ]
    for key in (
        "context_limited_cases",
        "failed_cases",
        "measurement_invalid_cases",
        "unimplemented_cases",
        "unsupported_cases",
        "validation_failed_cases",
    ):
        values = summary.get(key)
        if not isinstance(values, list):
            raise EvidenceError(f"memory source summary {key} is missing")
        translated_summary[key] = sorted(translate(value) for value in values)
    return translated_samples, translated_summary


def _case_integer(case: dict[str, Any], key: str) -> int:
    value = case[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceError(f"agentic case {key} must be a nonnegative integer")
    return value


def _case_number(case: dict[str, Any], key: str) -> float:
    value = case[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"agentic case {key} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise EvidenceError(f"agentic case {key} must be finite and nonnegative")
    return number


def _validate_agentic_case(case: dict[str, Any]) -> None:
    fields = frozenset(case)
    if fields not in {
        _AGENTIC_PROJECTED_CASE_BASE_FIELDS,
        _AGENTIC_PROJECTED_CASE_BASE_FIELDS | _AGENTIC_CASE_ENERGY_FIELDS,
    }:
        raise EvidenceError("agentic case does not match its exact evidence schema")
    missing = _AGENTIC_CASE_REQUIRED_FIELDS - set(case)
    if missing:
        raise EvidenceError(f"agentic case lacks required fields: {sorted(missing)!r}")
    case_id = case.get("case_id")
    if not isinstance(case_id, str):
        raise EvidenceError("agentic case must have a stable case identifier")
    scenario_id = case_id.split("--", 1)[0]
    if scenario_id not in KNOWN_AGENTIC_CASE_IDS:
        raise EvidenceError("agentic case identifier is unsupported")
    _agentic_case_identifier(case_id, scenario_id)
    if case.get("kind") != "agentic":
        raise EvidenceError("agentic case kind must be agentic")

    integer_keys = (
        "agentic_expected_tool_calls",
        "agentic_final_answers_correct",
        "agentic_final_answers_emitted",
        "agentic_length_terminated_turns",
        "agentic_malformed_tool_calls",
        "agentic_max_output_tokens_per_turn",
        "agentic_max_turns",
        "agentic_model_requests",
        "agentic_recoveries_required",
        "agentic_recoveries_succeeded",
        "agentic_tasks",
        "agentic_tasks_succeeded",
        "agentic_tool_calls_executed",
        "agentic_tool_calls_requested",
        "agentic_tool_calls_succeeded",
        "agentic_tool_errors",
        "agentic_tool_sequences_correct",
        "agentic_turn_limit_hits",
        "agentic_unknown_tool_calls",
    )
    values = {key: _case_integer(case, key) for key in integer_keys}
    tasks = values["agentic_tasks"]
    if tasks != 3 or case.get("requests") != tasks or case.get("concurrency") != 1:
        raise EvidenceError("agentic case must contain three single-stream episodes")
    max_turns = values["agentic_max_turns"]
    if not 2 <= max_turns <= 8:
        raise EvidenceError("agentic case max-turn bound is invalid")
    if values["agentic_max_output_tokens_per_turn"] < 2_048:
        raise EvidenceError("agentic case output bound is too small")
    model_requests = values["agentic_model_requests"]
    if not tasks <= model_requests <= tasks * max_turns:
        raise EvidenceError("agentic case model-request count is inconsistent")
    if values["agentic_expected_tool_calls"] != (
        _AGENTIC_EXPECTED_CALLS[scenario_id] * tasks
    ):
        raise EvidenceError("agentic case expected-tool count changed")
    minimum_turns = _AGENTIC_TOOL_COUNTS[scenario_id][4]
    if values["agentic_tool_sequences_correct"] == tasks:
        (
            expected_requested,
            expected_executed,
            expected_succeeded,
            expected_errors,
            _,
        ) = _AGENTIC_TOOL_COUNTS[scenario_id]
        for key, expected in (
            ("agentic_tool_calls_requested", expected_requested * tasks),
            ("agentic_tool_calls_executed", expected_executed * tasks),
            ("agentic_tool_calls_succeeded", expected_succeeded * tasks),
            ("agentic_tool_errors", expected_errors * tasks),
            ("agentic_malformed_tool_calls", 0),
            ("agentic_unknown_tool_calls", 0),
        ):
            if values[key] != expected:
                raise EvidenceError(
                    f"agentic case {key} disagrees with scenario {scenario_id}"
                )
    if not (
        values["agentic_tool_calls_succeeded"]
        <= values["agentic_tool_calls_executed"]
        <= values["agentic_tool_calls_requested"]
    ):
        raise EvidenceError("agentic case tool-call counters are inconsistent")
    if (
        values["agentic_tool_calls_succeeded"] + values["agentic_tool_errors"]
        != values["agentic_tool_calls_executed"]
    ):
        raise EvidenceError("agentic case tool outcomes do not match executions")
    if (
        values["agentic_malformed_tool_calls"]
        + values["agentic_unknown_tool_calls"]
        + values["agentic_tool_calls_executed"]
        > values["agentic_tool_calls_requested"]
    ):
        raise EvidenceError("agentic case classified too many tool calls")
    for key in (
        "agentic_final_answers_correct",
        "agentic_final_answers_emitted",
        "agentic_recoveries_required",
        "agentic_recoveries_succeeded",
        "agentic_tasks_succeeded",
        "agentic_tool_sequences_correct",
        "agentic_turn_limit_hits",
    ):
        if values[key] > tasks:
            raise EvidenceError(f"agentic case {key} exceeds episode count")
    if values["agentic_final_answers_correct"] > values["agentic_final_answers_emitted"]:
        raise EvidenceError("agentic case final-answer counters are inconsistent")
    recovery_tasks = tasks if scenario_id == "agentic-tool-error-recovery" else 0
    if values["agentic_recoveries_required"] != recovery_tasks:
        raise EvidenceError("agentic case recovery requirement changed")
    if values["agentic_recoveries_succeeded"] > recovery_tasks:
        raise EvidenceError("agentic case recovery count is inconsistent")
    if values["agentic_length_terminated_turns"] > model_requests:
        raise EvidenceError("agentic case length terminations exceed model requests")
    if (
        values["agentic_tool_sequences_correct"] == tasks
        and model_requests < tasks * minimum_turns
    ):
        raise EvidenceError("agentic case used too few model turns for its scenario")

    numeric_keys = (
        "agentic_model_requests_per_s",
        "agentic_task_success_rate",
        "agentic_tasks_per_s",
        "median_agentic_first_turn_ttft_s",
        "median_agentic_model_request_sum_s",
        "median_agentic_task_wall_s",
        "median_agentic_turns_used",
    )
    numbers = {key: _case_number(case, key) for key in numeric_keys}
    minimum_median_turns = (
        minimum_turns if values["agentic_tool_sequences_correct"] == tasks else 1
    )
    if not minimum_median_turns <= numbers["median_agentic_turns_used"] <= max_turns:
        raise EvidenceError("agentic case median turns is out of range")
    elapsed_s = _case_number(case, "elapsed_s")
    if elapsed_s <= 0:
        raise EvidenceError("agentic case elapsed time must be positive")
    expected_success_rate = values["agentic_tasks_succeeded"] / tasks
    if not math.isclose(
        numbers["agentic_task_success_rate"],
        expected_success_rate,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise EvidenceError("agentic case success rate is inconsistent")
    for key, numerator in (
        ("agentic_tasks_per_s", tasks),
        ("agentic_model_requests_per_s", model_requests),
    ):
        if not math.isclose(
            numbers[key], numerator / elapsed_s, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise EvidenceError(f"agentic case {key} is inconsistent")
    validation_passed = case.get("validation_passed")
    if not isinstance(validation_passed, bool) or validation_passed != (
        values["agentic_tasks_succeeded"] == tasks
    ):
        raise EvidenceError("agentic case validation flag is inconsistent")
    for key in ("aggregate_output_tps", "median_ttft_s", "p95_ttft_s", "request_tps"):
        if case.get(key) is not None:
            raise EvidenceError(f"agentic case must suppress generic metric {key}")
    for key in (
        "decode_estimate_one_token_chunks",
        "decode_metric_source",
        "median_decode_tps",
        "median_estimated_decode_tps",
        "p95_e2e_s",
    ):
        if case.get(key) is not None:
            raise EvidenceError(f"agentic case must suppress generic metric {key}")
    if not isinstance(case.get("measurement_valid"), bool):
        raise EvidenceError("agentic case measurement_valid must be boolean")
    for key in (
        "completion_tokens",
        "measurement_annotation_count",
        "prompt_tokens",
    ):
        _case_integer(case, key)
    if not math.isclose(
        _case_number(case, "median_e2e_s"),
        numbers["median_agentic_task_wall_s"],
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise EvidenceError("agentic case generic and task wall medians disagree")

    telemetry = case.get("telemetry")
    if _project_telemetry_summary(telemetry, name="case.telemetry") != telemetry:
        raise EvidenceError("agentic case telemetry projection changed")
    has_energy_metrics = any(
        key in case
        for key in (
            "agentic_sampled_energy_j_per_solved_task",
            "agentic_tasks_succeeded_per_sampled_joule",
        )
    )
    sampled_energy = telemetry.get("sampled_energy_j") if isinstance(telemetry, dict) else None
    if isinstance(sampled_energy, (int, float)) and sampled_energy > 0:
        if not has_energy_metrics:
            raise EvidenceError("agentic case omitted sampled-energy metrics")
        tasks_per_joule = _case_number(
            case, "agentic_tasks_succeeded_per_sampled_joule"
        )
        expected_tasks_per_joule = values["agentic_tasks_succeeded"] / sampled_energy
        if not math.isclose(
            tasks_per_joule,
            expected_tasks_per_joule,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise EvidenceError("agentic tasks-per-joule metric is inconsistent")
        joules_per_task = case.get("agentic_sampled_energy_j_per_solved_task")
        if values["agentic_tasks_succeeded"] == 0:
            if joules_per_task is not None:
                raise EvidenceError("unsolved agentic case has joules-per-task")
        else:
            measured_joules = _case_number(
                case, "agentic_sampled_energy_j_per_solved_task"
            )
            if not math.isclose(
                measured_joules,
                sampled_energy / values["agentic_tasks_succeeded"],
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise EvidenceError("agentic joules-per-task metric is inconsistent")
    elif has_energy_metrics:
        raise EvidenceError("agentic case has energy metrics without sampled energy")


def _prefix_cache_integer(value: Any, *, name: str, positive: bool = False) -> int:
    # These fields originate as native token/counter values.  Do not accept an
    # integral float here: cache evidence must retain the distinction between
    # a server-reported JSON integer and a value that was coerced somewhere in
    # the publication path.
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{name} must be an integer")
    result = value
    if result < 0 or (positive and result <= 0):
        raise EvidenceError(f"{name} must be non-negative")
    return result


def _project_prefix_cache_metrics(value: Any) -> dict[str, Any]:
    """Project the exact scalar report for the same-slot KV protocol."""

    if not isinstance(value, dict):
        raise EvidenceError("prefix-cache summary must be an object")
    mode = value.get("profile_mode")
    expected_root = {
        "protocol",
        "profile_mode",
        "prefix_target_words",
        "case_session_wall_s",
        "conditions",
        "paired_second_to_third",
    }
    if set(value) != expected_root:
        raise EvidenceError("prefix-cache summary does not match its exact schema")
    if value.get("protocol") != PREFIX_CACHE_PROTOCOL:
        raise EvidenceError("prefix-cache protocol identifier changed")
    if mode not in {"off", "on"}:
        raise EvidenceError("prefix-cache profile mode is invalid")

    prefix_target = _prefix_cache_integer(
        value.get("prefix_target_words"),
        name="prefix_cache.prefix_target_words",
        positive=True,
    )
    session_wall_s = _finite(
        value.get("case_session_wall_s"), name="prefix_cache.case_session_wall_s"
    )
    if session_wall_s is None or session_wall_s <= 0:
        raise EvidenceError("prefix-cache session wall must be positive")
    conditions = value.get("conditions")
    steps = prefix_cache_steps(mode)
    if not isinstance(conditions, list) or len(conditions) != len(steps):
        raise EvidenceError("prefix-cache condition count is invalid")
    condition_fields = {
        "cache_condition",
        "cache_prompt_control",
        "protocol_step_ordinal",
        "request_count",
        "logical_prompt_tokens",
        "physical_uncached_prompt_tokens",
        "cached_prompt_tokens",
        "cache_hit_fraction",
        "server_prompt_processing_s",
        "server_decode_s",
        "server_decode_tps",
        "logical_prompt_tokens_per_server_prompt_s",
        "physical_uncached_prompt_tokens_per_server_prompt_s",
        "condition_request_wall_s",
        "end_to_end_output_tokens_per_condition_request_wall_s",
        "median_ttft_s",
        "median_e2e_s",
        "median_client_decode_tps",
    }
    projected_conditions: list[dict[str, Any]] = []
    for ordinal, ((expected_condition, _, expected_control), condition) in enumerate(
        zip(steps, conditions), start=1
    ):
        if not isinstance(condition, dict) or set(condition) != condition_fields:
            raise EvidenceError("prefix-cache condition does not match its schema")
        if (
            condition.get("cache_condition") != expected_condition
            or condition.get("cache_prompt_control") != expected_control
            or _prefix_cache_integer(
                condition.get("protocol_step_ordinal"),
                name="prefix_cache.protocol_step_ordinal",
                positive=True,
            )
            != ordinal
        ):
            raise EvidenceError("prefix-cache conditions are not in protocol order")
        request_count = _prefix_cache_integer(
            condition.get("request_count"),
            name="prefix_cache.request_count",
            positive=True,
        )
        if request_count != 5:
            raise EvidenceError("prefix-cache condition must contain five requests")
        logical_tokens = _prefix_cache_integer(
            condition.get("logical_prompt_tokens"),
            name="prefix_cache.logical_prompt_tokens",
            positive=True,
        )
        physical_tokens = _prefix_cache_integer(
            condition.get("physical_uncached_prompt_tokens"),
            name="prefix_cache.physical_uncached_prompt_tokens",
        )
        cached_tokens = _prefix_cache_integer(
            condition.get("cached_prompt_tokens"),
            name="prefix_cache.cached_prompt_tokens",
        )
        if physical_tokens + cached_tokens != logical_tokens:
            raise EvidenceError("prefix-cache logical and physical prompt totals disagree")
        cache_fraction = _finite(
            condition.get("cache_hit_fraction"), name="prefix_cache.cache_hit_fraction"
        )
        if (
            cache_fraction is None
            or not 0 <= cache_fraction <= 1
            or not math.isclose(
                float(cache_fraction),
                cached_tokens / logical_tokens,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise EvidenceError("prefix-cache hit fraction is inconsistent")
        prompt_s = _finite(
            condition.get("server_prompt_processing_s"),
            name="prefix_cache.server_prompt_processing_s",
        )
        decode_s = _finite(
            condition.get("server_decode_s"), name="prefix_cache.server_decode_s"
        )
        condition_wall_s = _finite(
            condition.get("condition_request_wall_s"),
            name="prefix_cache.condition_request_wall_s",
        )
        if (
            prompt_s is None
            or prompt_s < 0
            or decode_s is None
            or decode_s <= 0
            or condition_wall_s is None
            or condition_wall_s <= 0
        ):
            raise EvidenceError("prefix-cache server timing or request wall is invalid")
        projected: dict[str, Any] = {
            "cache_condition": expected_condition,
            "cache_prompt_control": expected_control,
            "protocol_step_ordinal": ordinal,
            "request_count": request_count,
            "logical_prompt_tokens": logical_tokens,
            "physical_uncached_prompt_tokens": physical_tokens,
            "cached_prompt_tokens": cached_tokens,
            "cache_hit_fraction": cache_fraction,
            "server_prompt_processing_s": prompt_s,
            "server_decode_s": decode_s,
            "condition_request_wall_s": condition_wall_s,
        }
        for field in (
            "server_decode_tps",
            "end_to_end_output_tokens_per_condition_request_wall_s",
            "median_ttft_s",
            "median_e2e_s",
            "median_client_decode_tps",
        ):
            item = _finite(condition.get(field), name=f"prefix_cache.{field}")
            if item is None or item < 0:
                raise EvidenceError(f"prefix-cache {field} must be non-negative")
            projected[field] = item
        for field in (
            "logical_prompt_tokens_per_server_prompt_s",
            "physical_uncached_prompt_tokens_per_server_prompt_s",
        ):
            item = _finite(condition.get(field), name=f"prefix_cache.{field}")
            if prompt_s == 0:
                if item is not None:
                    raise EvidenceError("prefix-cache zero server prompt time needs null rates")
            elif item is None or item < 0:
                raise EvidenceError("prefix-cache server prompt rate is invalid")
            projected[field] = item
        if expected_condition.startswith("forced-cold") and cached_tokens != 0:
            raise EvidenceError("prefix-cache forced-cold condition reused tokens")
        if expected_condition == "warm-prefix-hit" and cache_fraction < 0.90:
            raise EvidenceError("prefix-cache warm condition did not prove substantial reuse")
        projected_conditions.append(projected)

    paired = value.get("paired_second_to_third")
    paired_fields = {
        "paired_blocks",
        "second_condition",
        "third_condition",
        "per_pair",
        "median_ttft_second_minus_third_s",
        "median_e2e_second_minus_third_s",
        "median_server_prompt_second_minus_third_s",
    }
    if not isinstance(paired, dict) or set(paired) != paired_fields:
        raise EvidenceError("prefix-cache paired summary does not match its schema")
    expected_third = prefix_cache_conditions(mode)[2]
    if (
        _prefix_cache_integer(
            paired.get("paired_blocks"),
            name="prefix_cache.paired_blocks",
            positive=True,
        )
        != 5
        or paired.get("second_condition") != "forced-cold-b"
        or paired.get("third_condition") != expected_third
        or not isinstance(paired.get("per_pair"), list)
    ):
        raise EvidenceError("prefix-cache paired protocol is invalid")
    pair_fields = {
        "cache_pair_index",
        "ttft_second_minus_third_s",
        "e2e_second_minus_third_s",
        "server_prompt_second_minus_third_s",
    }
    projected_pairs: list[dict[str, Any]] = []
    for expected_index, item in enumerate(paired["per_pair"], start=1):
        if not isinstance(item, dict) or set(item) != pair_fields:
            raise EvidenceError("prefix-cache paired observation schema is invalid")
        if _prefix_cache_integer(
            item.get("cache_pair_index"),
            name="prefix_cache.cache_pair_index",
            positive=True,
        ) != expected_index:
            raise EvidenceError("prefix-cache paired observations are not ordered")
        projected_item: dict[str, Any] = {"cache_pair_index": expected_index}
        for field in pair_fields - {"cache_pair_index"}:
            scalar = _finite(item.get(field), name=f"prefix_cache.{field}")
            if scalar is None:
                raise EvidenceError("prefix-cache paired observation is invalid")
            projected_item[field] = scalar
        projected_pairs.append(projected_item)
    if len(projected_pairs) != 5:
        raise EvidenceError("prefix-cache paired summary must contain five blocks")
    projected_paired: dict[str, Any] = {
        "paired_blocks": 5,
        "second_condition": "forced-cold-b",
        "third_condition": expected_third,
        "per_pair": projected_pairs,
    }
    for field in paired_fields - {
        "paired_blocks",
        "second_condition",
        "third_condition",
        "per_pair",
    }:
        scalar = _finite(paired.get(field), name=f"prefix_cache.{field}")
        if scalar is None:
            raise EvidenceError("prefix-cache paired summary is invalid")
        projected_paired[field] = scalar
    return {
        "protocol": PREFIX_CACHE_PROTOCOL,
        "profile_mode": mode,
        "prefix_target_words": prefix_target,
        "case_session_wall_s": session_wall_s,
        "conditions": projected_conditions,
        "paired_second_to_third": projected_paired,
    }


_PREFIX_CACHE_MODEL_FIELDS = frozenset(
    {
        "architecture",
        "backend",
        "estimated_ram_gib",
        "id",
        "lifecycle",
        "max_context",
        "native_context",
        "prefix_cache_mode",
        "quantization",
        "revision",
        "runtime_parallel",
        "source",
        "startup_timeout_s",
        "support_status",
        "tasks",
        "weight_file_count",
        "weight_size_bytes",
    }
)
_PREFIX_CACHE_MODEL_REQUIRED_FIELDS = frozenset(
    {
        "architecture",
        "backend",
        "id",
        "lifecycle",
        "max_context",
        "native_context",
        "prefix_cache_mode",
        "quantization",
        "revision",
        "runtime_parallel",
        "source",
        "startup_timeout_s",
        "support_status",
        "tasks",
    }
)
_PREFIX_CACHE_SUITE_FIELDS = frozenset({"cases", "id", "schema_version"})
_PREFIX_CACHE_SUITE_CASE_FIELDS = frozenset(
    {
        "case_id",
        "concurrency",
        "id",
        "kind",
        "max_output_tokens",
        "max_turns",
        "prompt_repetitions",
        "repetitions",
        "requires",
        "temperature",
        "warmups",
    }
)
_PREFIX_CACHE_CASE_FIELDS = frozenset(
    {
        "case_id",
        "completion_tokens",
        "concurrency",
        "elapsed_s",
        "kind",
        "measurement_annotation_count",
        "measurement_valid",
        "prefix_cache",
        "prompt_tokens",
        "reasoning_tokens",
        "requests",
        "validation_passed",
    }
)
_PREFIX_CACHE_RAW_CASE_FIELDS = (
    _PREFIX_CACHE_CASE_FIELDS - {"measurement_annotation_count"}
) | frozenset({"attempt_id", "measurement_annotations"})
_PREFIX_CACHE_RAW_CASE_TELEMETRY_FIELDS = _PREFIX_CACHE_RAW_CASE_FIELDS | frozenset(
    {"telemetry"}
)
_PREFIX_CACHE_RAW_CASE_TELEMETRY_ENERGY_FIELDS = (
    _PREFIX_CACHE_RAW_CASE_TELEMETRY_FIELDS
    | frozenset({"output_tokens_per_sampled_joule"})
)
_PREFIX_CACHE_SAMPLE_FIELDS = frozenset(
    {
        "burst_elapsed_s",
        "cache_condition",
        "cache_pair_index",
        "cache_prefix_target_words",
        "cache_profile_mode",
        "cache_prompt_control",
        "cache_step_ordinal",
        "cached_prompt_tokens",
        "case_attempt",
        "case_id",
        "case_sample_index",
        "completion_tokens",
        "decode_metric_source",
        "decode_s",
        "decode_tps",
        "elapsed_s",
        "emission_event_count",
        "finish_reason",
        "kind",
        "prometheus_global_cached_prompt_tokens",
        "prometheus_global_decode_s",
        "prometheus_global_decode_tokens",
        "prometheus_global_prompt_s",
        "prometheus_global_prompt_tokens",
        "output_tps",
        "prompt_tokens",
        "reasoning_tokens",
        "repetition",
        "sample_index",
        "sample_type",
        "selected_attempt",
        "server_cached_prompt_tokens",
        "server_decode_s",
        "server_decode_tokens",
        "server_prompt_s",
        "server_prompt_tokens",
        "ttft_s",
        "validation_passed",
    }
)
_PREFIX_CACHE_SUMMARY_FIELDS = frozenset(
    {
        "cases",
        "completed_cases",
        "run_completion_status",
        "startup_measurement_valid",
        "status",
        "suite",
    }
)
_PREFIX_CACHE_SUMMARY_SOURCE_EMPTY_LIST_FIELDS = frozenset(
    {
        "context_limited_cases",
        "failed_cases",
        "measurement_invalid_cases",
        "startup_safety_gates",
        "unimplemented_cases",
        "unsupported_cases",
        "validation_failed_cases",
    }
)
_PREFIX_CACHE_SUMMARY_SOURCE_ZERO_FIELDS = frozenset(
    {
        "measurement_annotations_count",
        "startup_measurement_annotations_count",
    }
)
_PREFIX_CACHE_SUMMARY_SOURCE_DROPPED_FIELDS = frozenset(
    {
        "artifact_validation",
        "artifact_validation_telemetry",
        "first_request",
        "first_request_telemetry",
        "shutdown_telemetry",
        "startup_telemetry",
    }
)
_PREFIX_CACHE_SUMMARY_SOURCE_NULL_FIELDS = frozenset(
    {"llamacpp_dflash_evidence", "llamacpp_mtp_evidence", "speculative_decoding"}
)

# A non-MTP llama.cpp server still emits the generic speculative-decoding
# rollup.  It is not cache-protocol evidence, but a fully disabled counter
# snapshot is expected source metadata and is safe to discard after exact
# validation.  Keep its scope/source constants fixed so this exception cannot
# silently admit a draft-enabled or differently scoped report.
_PREFIX_CACHE_DISABLED_SPECULATIVE_DECODING = {
    "accepted_tokens_per_position": {},
    "configured_max_draft_tokens": None,
    "draft_acceptance_rate": None,
    "mean_accepted_length": None,
    "method": None,
    "num_accepted_tokens": 0,
    "num_draft_tokens": 0,
    "num_drafts": 0,
    "proposal_depth": None,
    "requested": False,
    "scope": "all_persisted_llamacpp_server_lifetimes",
    "snapshot_count": 1,
    "source": "llamacpp_prometheus_cumulative_counters",
}

# A cache result is a tightly scoped protocol bundle rather than a generic
# serving report.  In particular, do not let the broad generic manifest grow
# an unreviewed field which could carry trace text or another non-protocol
# value after its checksums have been refreshed.  The source manifest has one
# extra runtime-only field (``versions``) which is intentionally discarded;
# the published cache manifest is reconstructed from this smaller contract.
_PREFIX_CACHE_MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "artifacts",
        "evidence_kind",
        "lifecycle",
        "model",
        "run_date_utc",
        "runtime",
        "sanitization",
        "schema_version",
        "source_run_id",
        "status",
        "suite",
    }
)
_PREFIX_CACHE_MANIFEST_OPTIONAL_FIELDS = frozenset({"hardware", "matrix_id"})
_PREFIX_CACHE_RUNTIME_FIELDS = frozenset(
    {
        "backend",
        "binary_sha256",
        "image",
        "image_sha256",
        "lifecycle",
        "recipe_revision",
        "runtime_revision",
        "source_revision",
    }
)
_PREFIX_CACHE_RUNTIME_SOURCE_FIELDS = _PREFIX_CACHE_RUNTIME_FIELDS | frozenset(
    {"versions"}
)
_PREFIX_CACHE_HARDWARE_FIELDS = frozenset(
    {
        "compute_capability",
        "driver_version",
        "gpu",
        "harness_revision",
        "harness_worktree_dirty",
        "platform",
        "unified_memory_bytes",
    }
)
_PREFIX_CACHE_ARTIFACT_REQUIRED_FIELDS = frozenset({"role", "sha256", "target"})
_PREFIX_CACHE_ARTIFACT_FIELDS = _PREFIX_CACHE_ARTIFACT_REQUIRED_FIELDS | frozenset(
    {"duration_s", "revision", "size_bytes", "source"}
)
_PREFIX_CACHE_LIFECYCLE_REQUIRED_FIELDS = frozenset(
    {"event_count", "event_counts", "terminal", "terminal_event"}
)
_PREFIX_CACHE_LIFECYCLE_FIELDS = _PREFIX_CACHE_LIFECYCLE_REQUIRED_FIELDS | frozenset(
    {"failure", "journal_elapsed_s"}
)
_PREFIX_CACHE_SANITIZATION = {
    "free_form_text_included": False,
    "payloads_included": False,
    "policy": SANITIZATION_POLICY,
    "raw_identifiers_included": False,
}

def _project_prefix_cache_model(value: Any) -> dict[str, Any]:
    """Canonicalize the cache profile provenance retained in a manifest."""

    if not isinstance(value, dict):
        raise EvidenceError("prefix-cache model metadata must be an object")
    if set(value) - _PREFIX_CACHE_MODEL_FIELDS or not _PREFIX_CACHE_MODEL_REQUIRED_FIELDS <= set(value):
        raise EvidenceError("prefix-cache model metadata does not match its exact schema")
    projected: dict[str, Any] = {}
    for key in (
        "id",
        "backend",
        "architecture",
        "prefix_cache_mode",
        "quantization",
        "source",
        "lifecycle",
        "support_status",
    ):
        projected[key] = _safe_id(value.get(key), name=f"prefix-cache model.{key}")
    projected["revision"] = _revision(
        value.get("revision"), name="prefix-cache model.revision"
    )
    if projected["backend"] != "llamacpp":
        raise EvidenceError("prefix-cache model backend must be llamacpp")
    if projected["prefix_cache_mode"] not in {"off", "on"}:
        raise EvidenceError("prefix-cache model mode is invalid")
    projected["runtime_parallel"] = _prefix_cache_integer(
        value.get("runtime_parallel"),
        name="prefix-cache model.runtime_parallel",
        positive=True,
    )
    if projected["runtime_parallel"] != 1:
        raise EvidenceError("prefix-cache model must use exactly one runtime slot")
    for key in ("max_context", "native_context", "startup_timeout_s"):
        projected[key] = _prefix_cache_integer(
            value.get(key), name=f"prefix-cache model.{key}", positive=True
        )
    if (
        projected["max_context"] != PREFIX_CACHE_CONTEXT_TOKENS
        or projected["native_context"] != PREFIX_CACHE_CONTEXT_TOKENS
    ):
        raise EvidenceError(
            "prefix-cache model contexts must equal the fixed protocol context"
        )
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise EvidenceError("prefix-cache model tasks must be a non-empty list")
    projected["tasks"] = [
        _safe_id(task, name="prefix-cache model.task") for task in tasks
    ]
    if "chat" not in projected["tasks"]:
        raise EvidenceError("prefix-cache model must declare chat support")
    if "estimated_ram_gib" in value:
        ram = _finite(value["estimated_ram_gib"], name="prefix-cache model.estimated_ram_gib")
        if ram is None or ram <= 0:
            raise EvidenceError("prefix-cache model estimated RAM is invalid")
        projected["estimated_ram_gib"] = ram
    for key in ("weight_file_count", "weight_size_bytes"):
        if key in value:
            projected[key] = _prefix_cache_integer(
                value[key], name=f"prefix-cache model.{key}", positive=True
            )
    if not _json_strict_equal(projected, value):
        raise EvidenceError("prefix-cache model projection changed")
    return projected


def _project_prefix_cache_suite(value: Any) -> dict[str, Any]:
    """Retain an exact, typed copy of the fixed cache-suite controls."""

    if not isinstance(value, dict) or set(value) != _PREFIX_CACHE_SUITE_FIELDS:
        raise EvidenceError("prefix-cache suite does not match its exact schema")
    if value.get("id") != PREFIX_CACHE_SUITE_ID:
        raise EvidenceError("prefix-cache suite identifier changed")
    _prefix_cache_integer(
        value.get("schema_version"),
        name="prefix-cache suite.schema_version",
        positive=True,
    )
    if value["schema_version"] != 1:
        raise EvidenceError("prefix-cache suite schema version is invalid")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != len(PREFIX_CACHE_PREFIX_TARGETS):
        raise EvidenceError("prefix-cache suite must contain both fixed cases")
    projected_cases: list[dict[str, Any]] = []
    observed: dict[str, int] = {}
    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != _PREFIX_CACHE_SUITE_CASE_FIELDS:
            raise EvidenceError("prefix-cache suite case does not match its exact schema")
        case_name = _safe_id(case.get("id"), name="prefix-cache suite case.id")
        case_id = _safe_id(case.get("case_id"), name="prefix-cache suite case.case_id")
        if not re.fullmatch(rf"{re.escape(case_name)}--[0-9a-f]{{12}}", case_id):
            raise EvidenceError("prefix-cache suite case identifier is invalid")
        if case.get("kind") != "cache" or case.get("requires") != ["chat"]:
            raise EvidenceError("prefix-cache suite case classification is invalid")
        for key, expected in (
            ("warmups", 0),
            ("repetitions", 5),
            ("max_output_tokens", 128),
            ("max_turns", 1),
            ("concurrency", 1),
        ):
            if _prefix_cache_integer(
                case.get(key), name=f"prefix-cache suite case.{key}"
            ) != expected:
                raise EvidenceError(f"prefix-cache suite case {key} is invalid")
        temperature = case.get("temperature")
        if type(temperature) is not float or temperature != 0.0:
            raise EvidenceError("prefix-cache suite case temperature must be JSON 0.0")
        target = _prefix_cache_integer(
            case.get("prompt_repetitions"),
            name="prefix-cache suite case.prompt_repetitions",
            positive=True,
        )
        if case_name in observed or case_id in case_ids:
            raise EvidenceError("prefix-cache suite cases are duplicated")
        observed[case_name] = target
        case_ids.add(case_id)
        projected_cases.append(
            {
                "case_id": case_id,
                "concurrency": 1,
                "id": case_name,
                "kind": "cache",
                "max_output_tokens": 128,
                "max_turns": 1,
                "prompt_repetitions": target,
                "repetitions": 5,
                "requires": ["chat"],
                "temperature": 0.0,
                "warmups": 0,
            }
        )
    if observed != PREFIX_CACHE_PREFIX_TARGETS or [
        case["id"] for case in projected_cases
    ] != list(PREFIX_CACHE_PREFIX_TARGETS):
        raise EvidenceError("prefix-cache suite targets do not match protocol")
    projected = {
        "cases": projected_cases,
        "id": PREFIX_CACHE_SUITE_ID,
        "schema_version": 1,
    }
    if not _json_strict_equal(projected, value):
        raise EvidenceError("prefix-cache suite projection changed")
    return projected


def _project_prefix_cache_artifacts(value: Any) -> list[dict[str, Any]]:
    """Validate the scalar model/runtime pins retained by a cache bundle."""

    if not isinstance(value, list):
        raise EvidenceError("prefix-cache artifacts must be a list")
    projected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) - _PREFIX_CACHE_ARTIFACT_FIELDS
            or not _PREFIX_CACHE_ARTIFACT_REQUIRED_FIELDS <= set(item)
        ):
            raise EvidenceError("prefix-cache artifact does not match its exact schema")
        artifact = {
            "role": _safe_id(item.get("role"), name="prefix-cache artifact.role"),
            "sha256": _sha256(item.get("sha256"), name="prefix-cache artifact.sha256"),
            "target": _safe_id(item.get("target"), name="prefix-cache artifact.target"),
        }
        if "size_bytes" in item:
            artifact["size_bytes"] = _prefix_cache_integer(
                item["size_bytes"],
                name="prefix-cache artifact.size_bytes",
                positive=True,
            )
        if "source" in item:
            artifact["source"] = _safe_id(
                item["source"], name="prefix-cache artifact.source"
            )
        if "revision" in item:
            artifact["revision"] = _revision(
                item["revision"], name="prefix-cache artifact.revision"
            )
        if "duration_s" in item:
            duration_s = _finite(
                item["duration_s"], name="prefix-cache artifact.duration_s"
            )
            if duration_s is None or duration_s < 0:
                raise EvidenceError("prefix-cache artifact duration is invalid")
            artifact["duration_s"] = duration_s
        key = (artifact["role"], artifact["sha256"], artifact["target"])
        if key in seen:
            raise EvidenceError("prefix-cache artifacts are duplicated")
        seen.add(key)
        projected.append(artifact)
    if projected != sorted(
        projected, key=lambda item: (item["role"], item["sha256"], item["target"])
    ):
        raise EvidenceError("prefix-cache artifacts are not in canonical order")
    roles = {item["role"] for item in projected}
    if not {"model", "runtime_binary"} <= roles:
        raise EvidenceError("prefix-cache artifacts must pin model and runtime binary")
    if not _json_strict_equal(projected, value):
        raise EvidenceError("prefix-cache artifact projection changed")
    return projected


def _project_prefix_cache_runtime(
    value: Any,
    *,
    model: dict[str, Any],
    source: bool,
) -> dict[str, Any]:
    """Keep a fixed runtime provenance subset and discard generic versions."""

    allowed = (
        _PREFIX_CACHE_RUNTIME_SOURCE_FIELDS
        if source
        else _PREFIX_CACHE_RUNTIME_FIELDS
    )
    if (
        not isinstance(value, dict)
        or set(value) - allowed
        or not {"backend", "lifecycle"} <= set(value)
    ):
        raise EvidenceError("prefix-cache runtime does not match its exact schema")
    projected = {
        "backend": _safe_id(value.get("backend"), name="prefix-cache runtime.backend"),
        "lifecycle": _safe_id(
            value.get("lifecycle"), name="prefix-cache runtime.lifecycle"
        ),
    }
    if (
        projected["backend"] != "llamacpp"
        or projected["backend"] != model["backend"]
        or projected["lifecycle"] != model["lifecycle"]
    ):
        raise EvidenceError("prefix-cache runtime does not match the frozen model")
    for key in ("image",):
        if key in value:
            projected[key] = _safe_id(value[key], name=f"prefix-cache runtime.{key}")
    for key in ("image_sha256", "binary_sha256"):
        if key in value:
            projected[key] = _sha256(value[key], name=f"prefix-cache runtime.{key}")
    for key in ("recipe_revision", "runtime_revision", "source_revision"):
        if key in value:
            projected[key] = _revision(value[key], name=f"prefix-cache runtime.{key}")
    if source and "versions" in value:
        versions = value["versions"]
        if not isinstance(versions, dict):
            raise EvidenceError("prefix-cache source runtime versions must be an object")
        for key, item in versions.items():
            _safe_id(key, name="prefix-cache source runtime version key")
            if isinstance(item, bool) or isinstance(item, (int, float)):
                _finite(item, name=f"prefix-cache source runtime version.{key}")
            elif isinstance(item, str) and _VERSION_RE.fullmatch(item):
                pass
            else:
                raise EvidenceError("prefix-cache source runtime version is invalid")
    if not source and not _json_strict_equal(projected, value):
        raise EvidenceError("prefix-cache runtime projection changed")
    return projected


def _project_prefix_cache_hardware(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value or set(value) - _PREFIX_CACHE_HARDWARE_FIELDS:
        raise EvidenceError("prefix-cache hardware does not match its exact schema")
    projected: dict[str, Any] = {}
    gpu_fields = {"compute_capability", "driver_version", "gpu", "platform"}
    if set(value) & gpu_fields:
        if not gpu_fields <= set(value):
            raise EvidenceError("prefix-cache GPU hardware metadata is incomplete")
        projected["compute_capability"] = _safe_id(
            value["compute_capability"], name="prefix-cache hardware.compute_capability"
        )
        projected["driver_version"] = _safe_id(
            value["driver_version"], name="prefix-cache hardware.driver_version"
        )
        if value["gpu"] != "NVIDIA GB10" or value["platform"] != "NVIDIA DGX Spark":
            raise EvidenceError("prefix-cache hardware identity changed")
        projected["gpu"] = "NVIDIA GB10"
        projected["platform"] = "NVIDIA DGX Spark"
    if "unified_memory_bytes" in value:
        projected["unified_memory_bytes"] = _prefix_cache_integer(
            value["unified_memory_bytes"],
            name="prefix-cache hardware.unified_memory_bytes",
            positive=True,
        )
    if "harness_revision" in value:
        projected["harness_revision"] = _revision(
            value["harness_revision"], name="prefix-cache hardware.harness_revision"
        )
    if "harness_worktree_dirty" in value:
        if not isinstance(value["harness_worktree_dirty"], bool):
            raise EvidenceError("prefix-cache hardware worktree flag must be boolean")
        projected["harness_worktree_dirty"] = value["harness_worktree_dirty"]
    if not _json_strict_equal(projected, value):
        raise EvidenceError("prefix-cache hardware projection changed")
    return projected


def _project_prefix_cache_lifecycle(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) - _PREFIX_CACHE_LIFECYCLE_FIELDS
        or not _PREFIX_CACHE_LIFECYCLE_REQUIRED_FIELDS <= set(value)
    ):
        raise EvidenceError("prefix-cache lifecycle does not match its exact schema")
    event_count = _prefix_cache_integer(
        value.get("event_count"), name="prefix-cache lifecycle.event_count", positive=True
    )
    event_counts = value.get("event_counts")
    if not isinstance(event_counts, dict) or not event_counts:
        raise EvidenceError("prefix-cache lifecycle event counts are invalid")
    projected_counts: dict[str, int] = {}
    for event, count in event_counts.items():
        if event not in _KNOWN_EVENTS:
            raise EvidenceError("prefix-cache lifecycle has an unknown event")
        projected_counts[event] = _prefix_cache_integer(
            count, name=f"prefix-cache lifecycle.{event}", positive=True
        )
    if (
        sum(projected_counts.values()) != event_count
        or projected_counts.get("run_start") != 1
        or projected_counts.get("run_complete") != 1
        or value.get("terminal") is not True
        or value.get("terminal_event") != "run_complete"
        or "failure" in value
    ):
        raise EvidenceError("prefix-cache lifecycle is not a completed protocol run")
    projected: dict[str, Any] = {
        "event_count": event_count,
        "event_counts": dict(sorted(projected_counts.items())),
        "terminal": True,
        "terminal_event": "run_complete",
    }
    if "journal_elapsed_s" in value:
        journal_elapsed_s = _finite(
            value["journal_elapsed_s"], name="prefix-cache lifecycle.journal_elapsed_s"
        )
        if journal_elapsed_s is None or journal_elapsed_s < 0:
            raise EvidenceError("prefix-cache lifecycle elapsed time is invalid")
        projected["journal_elapsed_s"] = journal_elapsed_s
    if not _json_strict_equal(projected, value):
        raise EvidenceError("prefix-cache lifecycle projection changed")
    return projected


def _project_prefix_cache_manifest(
    value: Any, *, source: bool = False
) -> dict[str, Any]:
    """Canonicalize the full outer cache manifest and reject every extra key.

    ``source=True`` admits the generic runtime ``versions`` map only long
    enough to validate and discard it.  The materialized bundle always uses
    the exact output schema, and verification invokes this with ``False``.
    """

    allowed = _PREFIX_CACHE_MANIFEST_REQUIRED_FIELDS | _PREFIX_CACHE_MANIFEST_OPTIONAL_FIELDS
    if not isinstance(value, dict) or set(value) - allowed or not _PREFIX_CACHE_MANIFEST_REQUIRED_FIELDS <= set(value):
        raise EvidenceError("prefix-cache manifest does not match its exact schema")
    schema_version = value.get("schema_version")
    if schema_version != SCHEMA_VERSION or type(schema_version) is not str:
        raise EvidenceError("prefix-cache manifest schema version changed")
    if value.get("evidence_kind") != "serving" or value.get("status") != "complete":
        raise EvidenceError("prefix-cache manifest is not a completed serving run")
    source_run_id = _safe_id(
        value.get("source_run_id"), name="prefix-cache manifest.source_run_id"
    )
    if value.get("run_date_utc") != _date_from_run_id(source_run_id):
        raise EvidenceError("prefix-cache manifest run date does not match its run ID")
    model = _project_prefix_cache_model(value.get("model"))
    suite = _project_prefix_cache_suite(value.get("suite"))
    runtime = _project_prefix_cache_runtime(
        value.get("runtime"), model=model, source=source
    )
    projected: dict[str, Any] = {
        "artifacts": _project_prefix_cache_artifacts(value.get("artifacts")),
        "evidence_kind": "serving",
        "lifecycle": _project_prefix_cache_lifecycle(value.get("lifecycle")),
        "model": model,
        "run_date_utc": _date_from_run_id(source_run_id),
        "runtime": runtime,
        "sanitization": dict(_PREFIX_CACHE_SANITIZATION),
        "schema_version": SCHEMA_VERSION,
        "source_run_id": source_run_id,
        "status": "complete",
        "suite": suite,
    }
    if not _json_strict_equal(value.get("sanitization"), _PREFIX_CACHE_SANITIZATION):
        raise EvidenceError("prefix-cache manifest sanitization changed")
    if "hardware" in value:
        projected["hardware"] = _project_prefix_cache_hardware(value["hardware"])
    if "matrix_id" in value:
        projected["matrix_id"] = _safe_id(
            value["matrix_id"], name="prefix-cache manifest.matrix_id"
        )
    if not source and not _json_strict_equal(projected, value):
        raise EvidenceError("prefix-cache manifest projection changed")
    return projected


_MEMORY_MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "artifact_validation",
        "artifacts",
        "evidence_kind",
        "hardware",
        "lifecycle",
        "model",
        "run_date_utc",
        "runtime",
        "sanitization",
        "schema_version",
        "source_run_id",
        "status",
        "suite",
    }
)
_MEMORY_MANIFEST_OPTIONAL_FIELDS = frozenset({"matrix_id"})
_MEMORY_RUNTIME_FIELDS = frozenset(
    {
        "backend",
        "binary_sha256",
        "lifecycle",
        "runtime_revision",
        "source_revision",
    }
)
_MEMORY_RUNTIME_SOURCE_FIELDS = _MEMORY_RUNTIME_FIELDS | frozenset({"versions"})
_MEMORY_ARTIFACT_FIELDS = frozenset(
    {"revision", "role", "sha256", "size_bytes", "source", "target"}
)
_MEMORY_ARTIFACT_VALIDATION_SINGLE_FIELDS = frozenset(
    {"model_sha256", "runtime_binary_sha256"}
)
_MEMORY_ARTIFACT_VALIDATION_SHARD_FIELDS = frozenset(
    {
        "model_sha256",
        "model_shard_count",
        "model_shard_sha256s",
        "model_total_size_bytes",
        "runtime_binary_sha256",
    }
)
_MEMORY_PROTOCOL_EVENT_COUNTS = {
    "artifact_validation_complete": 1,
    "case_complete": len(MEMORY_OPERATION_SCENARIO_IDS),
    "case_start": len(MEMORY_OPERATION_SCENARIO_IDS),
    "first_request_complete": 1,
    "request_complete": len(MEMORY_OPERATION_SCENARIO_IDS)
    * MEMORY_OPERATION_VARIANT_COUNT,
    "run_complete": 1,
    "run_start": 1,
    "server_ready": 1,
    "server_stopped": 1,
}
_MEMORY_LIFECYCLE_FIELDS = frozenset(
    {"event_counts", "protocol_event_count", "terminal", "terminal_event"}
)
_MEMORY_SANITIZATION = {
    "free_form_text_included": False,
    "payloads_included": False,
    "policy": SANITIZATION_POLICY,
    "raw_identifiers_included": False,
}


def _project_memory_runtime(value: Any, *, source: bool) -> dict[str, Any]:
    allowed = _MEMORY_RUNTIME_SOURCE_FIELDS if source else _MEMORY_RUNTIME_FIELDS
    if not isinstance(value, dict) or set(value) - allowed:
        raise EvidenceError("memory runtime does not match its exact schema")
    expected_digest = _sha256(
        MEMORY_OPERATION_LLAMACPP_DIGEST,
        name="fixed memory runtime digest",
    )
    projected = {
        "backend": _safe_id(value.get("backend"), name="memory runtime.backend"),
        "binary_sha256": _sha256(
            value.get("binary_sha256"), name="memory runtime.binary_sha256"
        ),
        "lifecycle": _safe_id(
            value.get("lifecycle"), name="memory runtime.lifecycle"
        ),
        "runtime_revision": _revision(
            value.get("runtime_revision"), name="memory runtime.runtime_revision"
        ),
        "source_revision": _revision(
            value.get("source_revision"), name="memory runtime.source_revision"
        ),
    }
    if projected != {
        "backend": "llamacpp",
        "binary_sha256": expected_digest,
        "lifecycle": "subprocess",
        "runtime_revision": MEMORY_OPERATION_LLAMACPP_REVISION,
        "source_revision": MEMORY_OPERATION_LLAMACPP_REVISION,
    }:
        raise EvidenceError("memory runtime pin or lifecycle changed")
    if source and "versions" in value and not isinstance(value["versions"], dict):
        raise EvidenceError("memory source runtime versions must be an object")
    if not source and not _json_strict_equal(projected, value):
        raise EvidenceError("memory runtime projection changed")
    return projected


def _project_memory_artifacts(
    value: Any, *, model: dict[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise EvidenceError("memory artifacts must be a list")
    projected: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or frozenset(item) not in {
            _MEMORY_ARTIFACT_FIELDS,
            frozenset({"revision", "role", "sha256", "target"}),
        }:
            raise EvidenceError("memory artifact does not match its exact schema")
        artifact: dict[str, Any] = {
            "revision": _revision(
                item.get("revision"), name="memory artifact.revision"
            ),
            "role": _safe_id(item.get("role"), name="memory artifact.role"),
            "sha256": _sha256(item.get("sha256"), name="memory artifact.sha256"),
            "target": _safe_id(item.get("target"), name="memory artifact.target"),
        }
        if "size_bytes" in item:
            artifact["size_bytes"] = _prefix_cache_integer(
                item["size_bytes"], name="memory artifact.size_bytes", positive=True
            )
        if "source" in item:
            artifact["source"] = _safe_id(
                item["source"], name="memory artifact.source"
            )
        projected.append(artifact)
    expected_model = _MEMORY_PANEL_MODEL_ARTIFACTS.get(str(model.get("id")))
    if expected_model is None:
        raise EvidenceError("memory artifacts reference an unsupported model")
    expected = [dict(item) for item in expected_model]
    expected.append(
        {
            "revision": MEMORY_OPERATION_LLAMACPP_REVISION,
            "role": "runtime_binary",
            "sha256": _sha256(
                MEMORY_OPERATION_LLAMACPP_DIGEST,
                name="fixed memory runtime artifact",
            ),
            "target": "llama-server",
        }
    )
    expected.sort(key=lambda item: (item["role"], item["sha256"], item["target"]))
    if not _json_strict_equal(projected, expected):
        raise EvidenceError("memory model/runtime artifacts changed")
    return projected


def _project_memory_artifact_validation(
    value: Any, *, model: dict[str, Any], source: bool
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError("memory artifact validation must be an object")
    source_value = dict(value)
    if source and "elapsed_s" in source_value:
        elapsed = _finite(
            source_value.pop("elapsed_s"), name="memory artifact validation.elapsed_s"
        )
        if elapsed is None or elapsed < 0:
            raise EvidenceError("memory artifact validation elapsed time is invalid")
    expected_model = _MEMORY_PANEL_MODEL_ARTIFACTS.get(str(model.get("id")))
    if expected_model is None:
        raise EvidenceError("memory artifact validation model is unsupported")
    runtime_digest = _sha256(
        MEMORY_OPERATION_LLAMACPP_DIGEST,
        name="fixed memory validation runtime",
    )
    if expected_model[0]["role"] == "model":
        expected = {
            "model_sha256": expected_model[0]["sha256"],
            "runtime_binary_sha256": runtime_digest,
        }
        expected_fields = _MEMORY_ARTIFACT_VALIDATION_SINGLE_FIELDS
    else:
        shard_digests = [str(item["sha256"]) for item in expected_model]
        expected = {
            "model_sha256": shard_digests[0],
            "model_shard_count": len(expected_model),
            "model_shard_sha256s": shard_digests,
            "model_total_size_bytes": sum(
                int(item["size_bytes"]) for item in expected_model
            ),
            "runtime_binary_sha256": runtime_digest,
        }
        expected_fields = _MEMORY_ARTIFACT_VALIDATION_SHARD_FIELDS
    if set(source_value) != expected_fields:
        raise EvidenceError("memory artifact validation schema changed")
    projected: dict[str, Any] = {}
    for key, item in expected.items():
        if key.endswith("sha256"):
            projected[key] = _sha256(
                source_value.get(key), name=f"memory artifact validation.{key}"
            )
        elif key == "model_shard_sha256s":
            values = source_value.get(key)
            if not isinstance(values, list):
                raise EvidenceError("memory artifact validation shard list changed")
            projected[key] = [
                _sha256(item, name="memory artifact validation shard")
                for item in values
            ]
        else:
            projected[key] = _prefix_cache_integer(
                source_value.get(key),
                name=f"memory artifact validation.{key}",
                positive=True,
            )
    if not _json_strict_equal(projected, expected):
        raise EvidenceError("memory artifact validation disagrees with frozen pins")
    return projected


def _project_memory_lifecycle(value: Any, *, source: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError("memory lifecycle must be an object")
    if source:
        counts = value.get("event_counts")
        if not isinstance(counts, dict):
            raise EvidenceError("memory source lifecycle event counts are missing")
        if any(counts.get(key) != expected for key, expected in _MEMORY_PROTOCOL_EVENT_COUNTS.items()):
            raise EvidenceError("memory source lifecycle protocol counts changed")
        if value.get("terminal") is not True or value.get("terminal_event") != "run_complete":
            raise EvidenceError("memory source lifecycle is not terminal")
    elif set(value) != _MEMORY_LIFECYCLE_FIELDS:
        raise EvidenceError("memory lifecycle does not match its exact schema")
    projected = {
        "event_counts": dict(_MEMORY_PROTOCOL_EVENT_COUNTS),
        "protocol_event_count": sum(_MEMORY_PROTOCOL_EVENT_COUNTS.values()),
        "terminal": True,
        "terminal_event": "run_complete",
    }
    if not source and not _json_strict_equal(projected, value):
        raise EvidenceError("memory lifecycle projection changed")
    return projected


def _project_memory_manifest(value: Any, *, source: bool = False) -> dict[str, Any]:
    allowed = _MEMORY_MANIFEST_REQUIRED_FIELDS | _MEMORY_MANIFEST_OPTIONAL_FIELDS
    if (
        not isinstance(value, dict)
        or set(value) - allowed
        or not _MEMORY_MANIFEST_REQUIRED_FIELDS <= set(value)
    ):
        raise EvidenceError("memory manifest does not match its exact schema")
    if value.get("schema_version") != SCHEMA_VERSION or type(value["schema_version"]) is not str:
        raise EvidenceError("memory manifest schema version changed")
    if value.get("evidence_kind") != "serving" or value.get("status") not in {
        "complete",
        "partial",
    }:
        raise EvidenceError("memory manifest classification changed")
    source_run_id = _safe_id(
        value.get("source_run_id"), name="memory manifest.source_run_id"
    )
    if value.get("run_date_utc") != _date_from_run_id(source_run_id):
        raise EvidenceError("memory manifest run date changed")
    model = _validate_projected_memory_model(value.get("model"))
    suite = _project_memory_suite(
        value.get("suite"), binding_model=model
    )
    artifacts = _project_memory_artifacts(value.get("artifacts"), model=model)
    hardware = _project_prefix_cache_hardware(value.get("hardware"))
    if set(hardware) != _PREFIX_CACHE_HARDWARE_FIELDS:
        raise EvidenceError("memory Spark hardware provenance is incomplete")
    projected: dict[str, Any] = {
        "artifact_validation": _project_memory_artifact_validation(
            value.get("artifact_validation"), model=model, source=source
        ),
        "artifacts": artifacts,
        "evidence_kind": "serving",
        "hardware": hardware,
        "lifecycle": _project_memory_lifecycle(value.get("lifecycle"), source=source),
        "model": model,
        "run_date_utc": _date_from_run_id(source_run_id),
        "runtime": _project_memory_runtime(value.get("runtime"), source=source),
        "sanitization": dict(_MEMORY_SANITIZATION),
        "schema_version": SCHEMA_VERSION,
        "source_run_id": source_run_id,
        "status": value["status"],
        "suite": suite,
    }
    if not _json_strict_equal(value.get("sanitization"), _MEMORY_SANITIZATION):
        raise EvidenceError("memory manifest sanitization changed")
    if "matrix_id" in value:
        projected["matrix_id"] = _safe_id(
            value["matrix_id"], name="memory manifest.matrix_id"
        )
    if not source and not _json_strict_equal(projected, value):
        raise EvidenceError("memory manifest projection changed")
    return projected


def _project_prefix_cache_case(case: Any) -> dict[str, Any]:
    """Project only the cache protocol's case-level scalar report."""

    if not isinstance(case, dict) or set(case) not in {
        _PREFIX_CACHE_CASE_FIELDS,
        _PREFIX_CACHE_RAW_CASE_FIELDS,
        _PREFIX_CACHE_RAW_CASE_TELEMETRY_FIELDS,
        _PREFIX_CACHE_RAW_CASE_TELEMETRY_ENERGY_FIELDS,
    }:
        raise EvidenceError("prefix-cache summary case does not match its exact schema")
    source_is_canonical = set(case) == _PREFIX_CACHE_CASE_FIELDS
    raw_annotations = case.get("measurement_annotations")
    has_count = "measurement_annotation_count" in case
    if raw_annotations is not None:
        if not isinstance(raw_annotations, list) or raw_annotations:
            raise EvidenceError("prefix-cache case must not contain measurement annotations")
    if has_count and (
        isinstance(case.get("measurement_annotation_count"), bool)
        or not isinstance(case.get("measurement_annotation_count"), int)
        or case.get("measurement_annotation_count") != 0
    ):
        raise EvidenceError("prefix-cache case annotation count is invalid")
    if raw_annotations is None and not has_count:
        raise EvidenceError("prefix-cache case annotation count is missing")
    if "attempt_id" in case:
        _safe_id(case.get("attempt_id"), name="prefix-cache case.attempt_id")
    if "telemetry" in case:
        # Source summaries may include the generic scalar telemetry rollup.
        # Validate it before discarding it: the cache evidence schema remains
        # minimal and protocol-only, so case telemetry is never materialized.
        _project_telemetry_summary(
            case.get("telemetry"), name="prefix-cache case.telemetry"
        )
    if "output_tokens_per_sampled_joule" in case:
        output_tokens_per_sampled_joule = _finite(
            case.get("output_tokens_per_sampled_joule"),
            name="prefix-cache case.output_tokens_per_sampled_joule",
        )
        if (
            output_tokens_per_sampled_joule is None
            or output_tokens_per_sampled_joule < 0
        ):
            raise EvidenceError(
                "prefix-cache case.output_tokens_per_sampled_joule must be finite "
                "and nonnegative"
            )
    projected: dict[str, Any] = {
        "case_id": _safe_id(case.get("case_id"), name="prefix-cache case.case_id"),
        "kind": "cache",
        "measurement_annotation_count": 0,
        "measurement_valid": True,
        "validation_passed": True,
        "prefix_cache": _project_prefix_cache_metrics(case.get("prefix_cache")),
    }
    if case.get("kind") != "cache" or case.get("measurement_valid") is not True or case.get("validation_passed") is not True:
        raise EvidenceError("prefix-cache summary case is not a valid completed cache case")
    for key, positive in (
        ("requests", True),
        ("concurrency", True),
        ("prompt_tokens", True),
        ("completion_tokens", True),
    ):
        projected[key] = _prefix_cache_integer(
            case.get(key), name=f"prefix-cache case.{key}", positive=positive
        )
    if projected["requests"] != 15 or projected["concurrency"] != 1:
        raise EvidenceError("prefix-cache case request geometry is invalid")
    elapsed_s = _finite(case.get("elapsed_s"), name="prefix-cache case.elapsed_s")
    if elapsed_s is None or elapsed_s <= 0:
        raise EvidenceError("prefix-cache case elapsed time is invalid")
    projected["elapsed_s"] = elapsed_s
    reasoning = case.get("reasoning_tokens")
    projected["reasoning_tokens"] = (
        None
        if reasoning is None
        else _prefix_cache_integer(
            reasoning, name="prefix-cache case.reasoning_tokens"
        )
    )
    if source_is_canonical and not _json_strict_equal(projected, case):
        raise EvidenceError("prefix-cache summary case projection changed")
    return projected


def _project_prefix_cache_sample(sample: Any) -> dict[str, Any]:
    """Re-project one published cache sample and reject every extra field."""

    if not isinstance(sample, dict) or set(sample) != _PREFIX_CACHE_SAMPLE_FIELDS:
        raise EvidenceError("prefix-cache evidence sample does not match its exact schema")
    if sample.get("sample_type") != "measured_request" or sample.get("kind") != "cache":
        raise EvidenceError("prefix-cache evidence sample has an invalid classification")
    for key in ("case_attempt", "case_sample_index", "sample_index"):
        _prefix_cache_integer(
            sample.get(key), name=f"prefix-cache sample.{key}", positive=True
        )
    repetition = _prefix_cache_integer(
        sample.get("repetition"), name="prefix-cache sample.repetition"
    )
    for key in ("selected_attempt", "validation_passed"):
        if not isinstance(sample.get(key), bool):
            raise EvidenceError(f"prefix-cache sample.{key} must be boolean")
    _safe_id(sample.get("case_id"), name="prefix-cache sample.case_id")
    raw_result = {
        key: sample[key]
        for key in _PREFIX_CACHE_RAW_RESULT_FIELDS
        if key != "emission_events"
    }
    raw_result["emission_events"] = sample["emission_event_count"]
    projected_result = _project_prefix_cache_request_result(raw_result)
    projected_sample_result = {
        "emission_event_count" if key == "emission_events" else key: value
        for key, value in projected_result.items()
    }
    expected = {
        "sample_index": sample["sample_index"],
        "sample_type": "measured_request",
        "case_attempt": sample["case_attempt"],
        "case_id": sample["case_id"],
        "case_sample_index": sample["case_sample_index"],
        "kind": "cache",
        "selected_attempt": sample["selected_attempt"],
        "validation_passed": sample["validation_passed"],
        "repetition": repetition,
        "burst_elapsed_s": sample["burst_elapsed_s"],
        **projected_sample_result,
    }
    burst = _finite(sample.get("burst_elapsed_s"), name="prefix-cache sample.burst_elapsed_s")
    elapsed = _finite(sample.get("elapsed_s"), name="prefix-cache sample.elapsed_s")
    if burst is None or elapsed is None or burst <= 0 or not math.isclose(
        float(burst), float(elapsed), rel_tol=1e-9, abs_tol=1e-9
    ):
        raise EvidenceError("prefix-cache sample burst and request walls disagree")
    if not _json_strict_equal(expected, sample):
        raise EvidenceError("prefix-cache evidence sample projection changed")
    return expected


def _validate_prefix_cache_disabled_speculative_decoding(value: Any) -> None:
    """Admit only llama.cpp's exact, disabled no-draft source rollup.

    The generic aggregate projector owns the field/type allowlist.  Reuse it
    here before comparing the fully projected value, so malformed nested
    objects and unsafe text still fail at the same source boundary.  Cache
    bundles deliberately do not retain this generic serving metric.
    """

    projected = _safe_summary_tree(value, name="speculative_decoding")
    if not _json_strict_equal(
        projected, _PREFIX_CACHE_DISABLED_SPECULATIVE_DECODING
    ):
        raise EvidenceError(
            "prefix-cache aggregate speculative_decoding must be the exact "
            "disabled no-draft llama.cpp summary"
        )


def _project_prefix_cache_summary(summary: Any) -> dict[str, Any]:
    """Publish a minimal, protocol-only cache aggregate summary."""

    if not isinstance(summary, dict):
        raise EvidenceError("prefix-cache aggregate summary must be an object")
    allowed_source = (
        _PREFIX_CACHE_SUMMARY_FIELDS
        | _PREFIX_CACHE_SUMMARY_SOURCE_EMPTY_LIST_FIELDS
        | _PREFIX_CACHE_SUMMARY_SOURCE_ZERO_FIELDS
        | _PREFIX_CACHE_SUMMARY_SOURCE_DROPPED_FIELDS
        | _PREFIX_CACHE_SUMMARY_SOURCE_NULL_FIELDS
    )
    if set(summary) - allowed_source:
        raise EvidenceError("prefix-cache aggregate summary contains nonprotocol fields")
    for key in _PREFIX_CACHE_SUMMARY_SOURCE_EMPTY_LIST_FIELDS:
        if key in summary and summary[key] != []:
            raise EvidenceError(f"prefix-cache aggregate {key} must be empty")
    for key in _PREFIX_CACHE_SUMMARY_SOURCE_ZERO_FIELDS:
        if key in summary and _prefix_cache_integer(
            summary[key], name=f"prefix-cache aggregate {key}"
        ) != 0:
            raise EvidenceError(f"prefix-cache aggregate {key} must be zero")
    for key in _PREFIX_CACHE_SUMMARY_SOURCE_NULL_FIELDS:
        if key not in summary or summary[key] is None:
            continue
        if key == "speculative_decoding":
            _validate_prefix_cache_disabled_speculative_decoding(summary[key])
            continue
        raise EvidenceError(f"prefix-cache aggregate {key} is not protocol evidence")
    if set(summary) == _PREFIX_CACHE_SUMMARY_FIELDS:
        source_is_canonical = True
    else:
        source_is_canonical = False
    cases = summary.get("cases")
    if not isinstance(cases, list) or len(cases) != len(PREFIX_CACHE_PREFIX_TARGETS):
        raise EvidenceError("prefix-cache aggregate summary must contain both cases")
    projected = {
        "cases": [_project_prefix_cache_case(case) for case in cases],
        "completed_cases": _prefix_cache_integer(
            summary.get("completed_cases"),
            name="prefix-cache aggregate completed_cases",
            positive=True,
        ),
        "run_completion_status": summary.get("run_completion_status"),
        "startup_measurement_valid": summary.get("startup_measurement_valid"),
        "status": summary.get("status"),
        "suite": summary.get("suite"),
    }
    if (
        projected["completed_cases"] != len(PREFIX_CACHE_PREFIX_TARGETS)
        or projected["run_completion_status"] != "completed"
        or projected["startup_measurement_valid"] is not True
        or projected["status"] != "complete"
        or projected["suite"] != PREFIX_CACHE_SUITE_ID
    ):
        raise EvidenceError("prefix-cache aggregate summary is not a completed protocol run")
    if not _json_strict_equal(projected, summary) and source_is_canonical:
        raise EvidenceError("prefix-cache aggregate summary projection changed")
    return projected


def _memory_case_integer(case: dict[str, Any], key: str) -> int:
    value = case.get(key)
    if type(value) is not int or value < 0:
        raise EvidenceError(f"memory case {key} must be a nonnegative JSON integer")
    return value


def _memory_case_number(case: dict[str, Any], key: str) -> float:
    value = case.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"memory case {key} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise EvidenceError(f"memory case {key} must be finite and nonnegative")
    return number


def _validate_memory_case(case: dict[str, Any]) -> None:
    case_id = case.get("case_id")
    if not isinstance(case_id, str):
        raise EvidenceError("memory case lacks a stable identifier")
    scenario_id = case_id.split("--", 1)[0]
    if scenario_id not in MEMORY_OPERATION_SCENARIO_IDS:
        raise EvidenceError("memory case identifier is unsupported")
    _memory_case_identifier(case_id, scenario_id)
    graphiti = scenario_id.startswith("graphiti-")
    expected_fields = _MEMORY_PROJECTED_CASE_BASE_FIELDS | (
        _GRAPHITI_CASE_METRIC_FIELDS if graphiti else frozenset()
    )
    allowed_shapes = {
        frozenset(expected_fields),
        frozenset(expected_fields | _MEMORY_CASE_OPTIONAL_TELEMETRY_FIELDS),
        frozenset(
            expected_fields
            | _MEMORY_CASE_OPTIONAL_TELEMETRY_FIELDS
            | _MEMORY_CASE_OPTIONAL_ENERGY_FIELDS
        ),
    }
    if frozenset(case) not in allowed_shapes:
        raise EvidenceError("memory case does not match its exact evidence schema")
    if case.get("kind") != "memory":
        raise EvidenceError("memory case kind changed")

    integer_keys = (
        "completion_tokens",
        "concurrency",
        "graphiti_resolver_operations",
        "measurement_annotation_count",
        "memory_action_correct",
        "memory_evidence_correct",
        "memory_field_checks_applicable",
        "memory_injection_refusals_required",
        "memory_injection_refusals_succeeded",
        "memory_json_objects_emitted",
        "memory_mutations_expected",
        "memory_mutations_selected",
        "memory_operations",
        "memory_operations_correct",
        "memory_path_correct",
        "memory_prompt_cache_disabled_requests",
        "memory_protected_value_emissions",
        "memory_reason_correct",
        "memory_schema_valid",
        "memory_secret_refusals_required",
        "memory_secret_refusals_succeeded",
        "memory_target_correct",
        "memory_tier_correct",
        "memory_total_completion_tokens",
        "memory_total_prompt_tokens",
        "memory_unexpected_tool_calls",
        "memory_valid_from_correct",
        "memory_valid_to_correct",
        "memory_value_correct",
        "memory_zero_cached_prompt_requests",
        "prompt_tokens",
        "requests",
        "synthetic_memory_extension_operations",
    )
    counts = {key: _memory_case_integer(case, key) for key in integer_keys}
    operations = MEMORY_OPERATION_VARIANT_COUNT
    if (
        counts["requests"] != operations
        or counts["memory_operations"] != operations
        or counts["concurrency"] != 1
    ):
        raise EvidenceError("memory case must contain exactly three single-stream requests")
    graphiti_operations = operations if graphiti else 0
    extension_operations = 0 if graphiti else operations
    if (
        counts["graphiti_resolver_operations"] != graphiti_operations
        or counts["synthetic_memory_extension_operations"] != extension_operations
        or counts["memory_field_checks_applicable"] != extension_operations
    ):
        raise EvidenceError("memory case family counts are inconsistent")
    for key in (
        "memory_json_objects_emitted",
        "memory_mutations_selected",
        "memory_operations_correct",
        "memory_prompt_cache_disabled_requests",
        "memory_protected_value_emissions",
        "memory_schema_valid",
        "memory_zero_cached_prompt_requests",
    ):
        if counts[key] > operations:
            raise EvidenceError(f"memory case {key} exceeds its request denominator")
    for key in (
        "memory_action_correct",
        "memory_evidence_correct",
        "memory_path_correct",
        "memory_reason_correct",
        "memory_target_correct",
        "memory_tier_correct",
        "memory_valid_from_correct",
        "memory_valid_to_correct",
        "memory_value_correct",
    ):
        if counts[key] > extension_operations:
            raise EvidenceError(f"memory case {key} exceeds its field denominator")
    expected_mutations = operations if scenario_id in _MEMORY_MUTATION_EXPECTED else 0
    if counts["memory_mutations_expected"] != expected_mutations:
        raise EvidenceError("memory case mutation oracle count changed")
    secret_required = operations if scenario_id == "memory-secret-refusal" else 0
    injection_required = (
        operations if scenario_id == "memory-injection-refusal" else 0
    )
    if (
        counts["memory_secret_refusals_required"] != secret_required
        or counts["memory_injection_refusals_required"] != injection_required
        or counts["memory_secret_refusals_succeeded"] > secret_required
        or counts["memory_injection_refusals_succeeded"] > injection_required
    ):
        raise EvidenceError("memory case refusal counts are inconsistent")
    if (
        counts["memory_total_prompt_tokens"] != counts["prompt_tokens"]
        or counts["memory_total_completion_tokens"] != counts["completion_tokens"]
    ):
        raise EvidenceError("memory case token totals disagree")
    reasoning_tokens = case.get("reasoning_tokens")
    total_reasoning_tokens = case.get("memory_total_reasoning_tokens")
    if reasoning_tokens is not None or total_reasoning_tokens is not None:
        raise EvidenceError("memory evidence v1 requires null case reasoning totals")

    accuracy = _memory_case_number(case, "memory_operation_accuracy")
    if not math.isclose(
        accuracy,
        counts["memory_operations_correct"] / operations,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise EvidenceError("memory case accuracy is inconsistent")
    elapsed_s = _memory_case_number(case, "elapsed_s")
    if elapsed_s <= 0:
        raise EvidenceError("memory case elapsed time must be positive")
    for key, expected in (
        ("aggregate_output_tps", counts["completion_tokens"] / elapsed_s),
        ("request_tps", operations / elapsed_s),
    ):
        value = _memory_case_number(case, key)
        if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise EvidenceError(f"memory case {key} is inconsistent")
    for key in ("median_e2e_s", "median_ttft_s"):
        _memory_case_number(case, key)
    for key in (
        "decode_estimate_one_token_chunks",
        "decode_metric_source",
        "median_decode_tps",
        "median_estimated_decode_tps",
        "p95_e2e_s",
        "p95_ttft_s",
    ):
        if case.get(key) is not None:
            raise EvidenceError(f"memory case must suppress generic metric {key}")
    if (
        case.get("measurement_valid") is not True
        or counts["measurement_annotation_count"] != 0
        or case.get("validation_passed")
        is not (counts["memory_operations_correct"] == operations)
    ):
        raise EvidenceError("memory case validity flags are inconsistent")
    for key in ("memory_total_server_decode_s", "memory_total_server_prompt_s"):
        if _memory_case_number(case, key) <= 0:
            raise EvidenceError(f"memory case {key} must be positive")

    if graphiti:
        for key in (
            "graphiti_contradicted_sets_correct",
            "graphiti_duplicate_sets_correct",
            "graphiti_resolver_correct",
        ):
            if _memory_case_integer(case, key) > operations:
                raise EvidenceError(f"memory case {key} exceeds resolver operations")
        resolver_accuracy = _memory_case_number(case, "graphiti_resolver_accuracy")
        if not math.isclose(
            resolver_accuracy,
            case["graphiti_resolver_correct"] / operations,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise EvidenceError("memory resolver accuracy is inconsistent")
        confusion = _project_memory_resolver_confusion(
            case.get("graphiti_resolver_confusion")
        )
        expected_label = _MEMORY_EXPECTED_RESOLVER_ACTION[scenario_id]
        if set(confusion) != {expected_label} or sum(
            confusion[expected_label].values()
        ) != operations:
            raise EvidenceError("memory resolver confusion denominator changed")

    telemetry = case.get("telemetry")
    if telemetry is not None and _project_telemetry_summary(
        telemetry, name="case.telemetry"
    ) != telemetry:
        raise EvidenceError("memory case telemetry projection changed")
    sampled_energy = telemetry.get("sampled_energy_j") if isinstance(telemetry, dict) else None
    if isinstance(sampled_energy, (int, float)) and sampled_energy > 0:
        if "output_tokens_per_sampled_joule" not in case:
            raise EvidenceError("memory case omitted its sampled-energy metric")
        expected_rate = counts["completion_tokens"] / float(sampled_energy)
        if not math.isclose(
            _memory_case_number(case, "output_tokens_per_sampled_joule"),
            expected_rate,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise EvidenceError("memory case sampled-energy metric is inconsistent")
    elif "output_tokens_per_sampled_joule" in case:
        raise EvidenceError("memory case has an energy metric without sampled energy")


def _project_case(case: Any) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise EvidenceError("summary case must be an object")
    if case.get("kind") == "cache":
        return _project_prefix_cache_case(case)
    if "measurement_annotation_count" in case:
        annotation_count = case["measurement_annotation_count"]
        if (
            "measurement_annotations" in case
            or isinstance(annotation_count, bool)
            or not isinstance(annotation_count, int)
            or annotation_count < 0
        ):
            raise EvidenceError("case measurement annotation count is invalid")
    unknown = set(case) - _CASE_FIELDS - _CASE_DROPPED_FIELDS
    if unknown:
        raise EvidenceError(f"unknown summary case fields: {sorted(unknown)!r}")
    kind = case.get("kind")
    present_agentic = set(case) & _AGENTIC_CASE_FIELDS
    if kind != "agentic" and present_agentic:
        raise EvidenceError("non-agentic case contains agentic metrics")
    present_memory = set(case) & (
        _MEMORY_CASE_METRIC_FIELDS | _GRAPHITI_CASE_METRIC_FIELDS
    )
    if kind != "memory" and present_memory:
        raise EvidenceError("non-memory case contains memory metrics")
    projected: dict[str, Any] = {}
    for key, value in case.items():
        if key in _CASE_DROPPED_FIELDS:
            if key == "measurement_annotations" and isinstance(value, list):
                projected["measurement_annotation_count"] = len(value)
            continue
        if value is None:
            if key not in _CASE_NULLABLE_FIELDS:
                raise EvidenceError(f"case.{key} must not be null")
            projected[key] = None
        elif key in _CASE_STRING_FIELDS:
            projected[key] = _safe_id(value, name=f"case.{key}", nullable=True)
        elif key in _CASE_BOOLEAN_FIELDS:
            if not isinstance(value, bool):
                raise EvidenceError(f"case.{key} must be boolean")
            projected[key] = value
        elif key == "measurement_annotation_count":
            projected[key] = value
        elif key == "prefix_cache":
            projected[key] = _project_prefix_cache_metrics(value)
        elif key == "telemetry":
            projected[key] = _project_telemetry_summary(value, name="case.telemetry")
        elif key == "graphiti_resolver_confusion":
            projected[key] = _project_memory_resolver_confusion(value)
        elif key == "quality_accuracy_by_category":
            projected[key] = _project_quality_accuracy(value)
        elif key in _CASE_FIELDS - _CASE_STRING_FIELDS - _CASE_BOOLEAN_FIELDS - _CASE_OBJECT_FIELDS:
            projected[key] = _finite(value, name=f"case.{key}")
        else:
            raise EvidenceError(f"invalid summary case value for {key}")
    if kind == "cache":
        if (
            "prefix_cache" not in projected
            or projected.get("measurement_valid") is not True
            or projected.get("validation_passed") is not True
        ):
            raise EvidenceError(
                "cache evidence requires a valid completed prefix-cache summary"
            )
    elif "prefix_cache" in projected:
        raise EvidenceError("non-cache case contains prefix-cache metrics")
    if kind == "agentic":
        _validate_agentic_case(projected)
    if kind == "memory":
        _validate_memory_case(projected)
    return projected


def _prefix_cache_numbers_equal(left: Any, right: Any) -> bool:
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, (int, float))
        or not isinstance(right, (int, float))
    ):
        return False
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)


def _prefix_cache_values_equal(actual: Any, expected: Any) -> bool:
    """Compare recomputed cache aggregates while preserving their exact shape."""

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return _prefix_cache_numbers_equal(actual, expected)
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _prefix_cache_values_equal(actual[key], expected[key])
            for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _prefix_cache_values_equal(left, right)
            for left, right in zip(actual, expected)
        )
    return actual == expected


def _validate_prefix_cache_suite_and_model(
    *, model: Any, suite: Any
) -> tuple[str, dict[str, int]]:
    if not isinstance(model, dict) or not isinstance(suite, dict):
        raise EvidenceError("prefix-cache evidence requires model and suite metadata")
    projected_model = _project_prefix_cache_model(model)
    if not _json_strict_equal(projected_model, model):
        raise EvidenceError("prefix-cache model projection changed")
    projected_suite = _project_prefix_cache_suite(suite)
    if not _json_strict_equal(projected_suite, suite):
        raise EvidenceError("prefix-cache suite projection changed")
    mode = projected_model["prefix_cache_mode"]
    cases = projected_suite["cases"]
    case_targets: dict[str, int] = {}
    for case in cases:
        case_targets[case["case_id"]] = case["prompt_repetitions"]
    return mode, case_targets


def _prefix_cache_sample_number(
    sample: dict[str, Any], key: str, *, integer: bool = False, positive: bool = False
) -> int | float:
    if integer:
        return _prefix_cache_integer(
            sample.get(key),
            name=f"prefix-cache sample.{key}",
            positive=positive,
        )
    value = _finite(sample.get(key), name=f"prefix-cache sample.{key}")
    if value is None or value < 0 or (positive and value <= 0):
        raise EvidenceError(f"prefix-cache sample {key} is invalid")
    return float(value)


def _validate_prefix_cache_sample(
    sample: dict[str, Any],
    *,
    mode: str,
    prefix_target: int,
) -> tuple[int, str]:
    if (
        sample.get("sample_type") != "measured_request"
        or sample.get("kind") != "cache"
        or sample.get("selected_attempt") is not True
        or sample.get("validation_passed") is not True
        or sample.get("cache_profile_mode") != mode
        or sample.get("finish_reason") != "length"
    ):
        raise EvidenceError("prefix-cache sample classification is invalid")
    pair_index = _prefix_cache_sample_number(
        sample, "cache_pair_index", integer=True, positive=True
    )
    step_ordinal = _prefix_cache_sample_number(
        sample, "cache_step_ordinal", integer=True, positive=True
    )
    if pair_index not in range(1, 6) or step_ordinal not in range(1, 4):
        raise EvidenceError("prefix-cache sample protocol ordinal is invalid")
    expected_case_sample_index = (pair_index - 1) * 3 + step_ordinal
    if _prefix_cache_sample_number(
        sample, "case_sample_index", integer=True, positive=True
    ) != expected_case_sample_index:
        raise EvidenceError("prefix-cache sample event order does not match its fixed schedule")
    expected_condition, _, expected_control = prefix_cache_steps(mode)[step_ordinal - 1]
    if (
        sample.get("cache_condition") != expected_condition
        or sample.get("cache_prompt_control") != expected_control
        or _prefix_cache_sample_number(
            sample, "cache_prefix_target_words", integer=True, positive=True
        )
        != prefix_target
        or _prefix_cache_sample_number(sample, "repetition", integer=True)
        != pair_index - 1
    ):
        raise EvidenceError("prefix-cache sample does not match its fixed schedule")
    prompt_tokens = _prefix_cache_sample_number(
        sample, "prompt_tokens", integer=True, positive=True
    )
    cached_tokens = _prefix_cache_sample_number(
        sample, "cached_prompt_tokens", integer=True
    )
    completion_tokens = _prefix_cache_sample_number(
        sample, "completion_tokens", integer=True, positive=True
    )
    if cached_tokens > prompt_tokens or completion_tokens != 128:
        raise EvidenceError("prefix-cache sample token counts are invalid")
    # ``prometheus_global_*`` fields are global Prometheus deltas.  They must
    # remain scalar diagnostics, but cannot be assigned to one request or used
    # to reconcile the exact request-scoped final SSE counters below.
    for key in (
        "prometheus_global_prompt_tokens",
        "prometheus_global_cached_prompt_tokens",
        "prometheus_global_decode_tokens",
    ):
        _prefix_cache_sample_number(sample, key, integer=True)
    server_prompt_tokens = _prefix_cache_sample_number(
        sample, "server_prompt_tokens", integer=True
    )
    server_cached_tokens = _prefix_cache_sample_number(
        sample, "server_cached_prompt_tokens", integer=True
    )
    server_decode_tokens = _prefix_cache_sample_number(
        sample, "server_decode_tokens", integer=True, positive=True
    )
    if (
        server_prompt_tokens + server_cached_tokens != prompt_tokens
        or server_cached_tokens != cached_tokens
        or server_decode_tokens != completion_tokens
    ):
        raise EvidenceError("prefix-cache sample server counters do not reconcile")
    for key, positive in (
        ("ttft_s", False),
        ("elapsed_s", True),
        ("decode_s", True),
        ("decode_tps", False),
        ("output_tps", False),
        ("prometheus_global_prompt_s", False),
        ("prometheus_global_decode_s", False),
        ("server_prompt_s", False),
        ("server_decode_s", True),
    ):
        _prefix_cache_sample_number(sample, key, positive=positive)
    if sample.get("decode_metric_source") != "client_estimate":
        raise EvidenceError("prefix-cache sample has an unsupported decode metric")
    ttft_s = _prefix_cache_sample_number(sample, "ttft_s")
    elapsed_s = _prefix_cache_sample_number(sample, "elapsed_s", positive=True)
    decode_s = _prefix_cache_sample_number(sample, "decode_s", positive=True)
    decode_tps = _prefix_cache_sample_number(sample, "decode_tps")
    output_tps = _prefix_cache_sample_number(sample, "output_tps")
    expected_decode_tps = max(completion_tokens - 1, 0) / decode_s
    expected_output_tps = completion_tokens / elapsed_s
    if (
        ttft_s > elapsed_s
        or not math.isclose(
            decode_tps, expected_decode_tps, rel_tol=1e-9, abs_tol=1e-9
        )
        or not math.isclose(
            output_tps, expected_output_tps, rel_tol=1e-9, abs_tol=1e-9
        )
    ):
        raise EvidenceError("prefix-cache sample client timing metrics are inconsistent")
    reasoning = sample.get("reasoning_tokens")
    if reasoning is not None:
        _prefix_cache_sample_number(
            sample, "reasoning_tokens", integer=True
        )
    if expected_condition.startswith("forced-cold") and cached_tokens != 0:
        raise EvidenceError("prefix-cache forced-cold sample reused tokens")
    if expected_condition == "warm-prefix-hit" and cached_tokens / prompt_tokens < 0.90:
        raise EvidenceError("prefix-cache warm sample did not prove substantial reuse")
    return int(pair_index), expected_condition


def _validate_prefix_cache_aggregates(
    samples: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    model: Any,
    suite: Any,
) -> None:
    """Bind every cache aggregate to selected samples and frozen controls.

    This runs identically during export and evidence verification.  It rejects
    any unpaired profile/suite, stale summary, missing cache measurement, or
    altered condition median rather than publishing a merely well-typed value.
    """

    raw_cases = summary.get("cases", [])
    if raw_cases is not None and not isinstance(raw_cases, list):
        raise EvidenceError("prefix-cache summary cases must be a list")
    for case in raw_cases or []:
        if not isinstance(case, dict) or not _json_strict_equal(
            _project_case(case), case
        ):
            raise EvidenceError("summary case projection changed")
    cache_cases = [
        case
        for case in raw_cases or []
        if isinstance(case, dict) and case.get("kind") == "cache"
    ]
    cache_samples = [
        sample for sample in samples if isinstance(sample, dict) and sample.get("kind") == "cache"
    ]
    has_cache_material = bool(cache_cases or cache_samples)
    profile_mode = model.get("prefix_cache_mode") if isinstance(model, dict) else None
    if not has_cache_material:
        if profile_mode is not None or (
            isinstance(suite, dict) and suite.get("id") == PREFIX_CACHE_SUITE_ID
        ):
            raise EvidenceError("prefix-cache profile or suite lacks valid cache evidence")
        return
    expected_sample_count = len(PREFIX_CACHE_PREFIX_TARGETS) * 15
    if (
        len(samples) != expected_sample_count
        or len(cache_samples) != expected_sample_count
        or len(cache_samples) != len(samples)
    ):
        raise EvidenceError(
            "prefix-cache evidence must contain exactly its thirty protocol samples"
        )
    mode, planned_targets = _validate_prefix_cache_suite_and_model(
        model=model, suite=suite
    )
    if not _json_strict_equal(_project_prefix_cache_summary(summary), summary):
        raise EvidenceError("prefix-cache aggregate summary projection changed")
    if mode != profile_mode:
        raise EvidenceError("prefix-cache model mode is inconsistent")
    if not cache_cases:
        raise EvidenceError("prefix-cache samples lack a completed cache summary")
    if len(cache_cases) != len(raw_cases or []):
        raise EvidenceError("prefix-cache summary must contain only cache cases")
    if len({case.get("case_id") for case in cache_cases}) != len(cache_cases):
        raise EvidenceError("prefix-cache summary case IDs are duplicated")
    expected_case_ids = list(planned_targets)
    for sample_index, sample in enumerate(cache_samples, start=1):
        _project_prefix_cache_sample(sample)
        expected_case_id = expected_case_ids[(sample_index - 1) // 15]
        if (
            _prefix_cache_sample_number(
                sample, "sample_index", integer=True, positive=True
            )
            != sample_index
            or sample.get("case_id") != expected_case_id
            or _prefix_cache_sample_number(
                sample, "case_attempt", integer=True, positive=True
            )
            != 1
        ):
            raise EvidenceError(
                "prefix-cache evidence samples do not match the fixed run order"
            )
    selected_samples = [
        sample for sample in cache_samples if sample.get("selected_attempt") is True
    ]
    if len(selected_samples) != expected_sample_count:
        raise EvidenceError("prefix-cache evidence retains an unselected attempt")
    selected_by_case: dict[str, list[dict[str, Any]]] = {}
    for sample in selected_samples:
        case_id = sample.get("case_id")
        if not isinstance(case_id, str):
            raise EvidenceError("prefix-cache selected sample lacks a case ID")
        selected_by_case.setdefault(case_id, []).append(sample)
    summary_ids = {case.get("case_id") for case in cache_cases}
    if (
        summary_ids != set(planned_targets)
        or set(selected_by_case) != summary_ids
    ):
        raise EvidenceError("prefix-cache selected samples and summary cases disagree")

    for case in cache_cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id not in planned_targets:
            raise EvidenceError("prefix-cache summary case is not in frozen suite")
        prefix_target = planned_targets[case_id]
        records = selected_by_case[case_id]
        if len(records) != 15:
            raise EvidenceError("prefix-cache case must contain fifteen selected samples")
        for expected_index, sample in enumerate(records, start=1):
            if _prefix_cache_sample_number(
                sample, "case_sample_index", integer=True, positive=True
            ) != expected_index:
                raise EvidenceError(
                    "prefix-cache sample event order does not match its fixed schedule"
                )
        pairs: dict[int, set[str]] = {}
        for sample in records:
            pair_index, condition = _validate_prefix_cache_sample(
                sample, mode=mode, prefix_target=prefix_target
            )
            pairs.setdefault(pair_index, set()).add(condition)
        expected_conditions = set(prefix_cache_conditions(mode))
        if set(pairs) != set(range(1, 6)) or any(
            conditions != expected_conditions for conditions in pairs.values()
        ):
            raise EvidenceError("prefix-cache selected samples are not complete blocks")
        if (
            _prefix_cache_sample_number(case, "requests", integer=True, positive=True)
            != len(records)
            or _prefix_cache_sample_number(case, "concurrency", integer=True, positive=True)
            != 1
        ):
            raise EvidenceError("prefix-cache generic case counts disagree")
        prompt_total = sum(
            _prefix_cache_sample_number(sample, "prompt_tokens", integer=True)
            for sample in records
        )
        completion_total = sum(
            _prefix_cache_sample_number(sample, "completion_tokens", integer=True)
            for sample in records
        )
        if (
            _prefix_cache_sample_number(case, "prompt_tokens", integer=True)
            != prompt_total
            or _prefix_cache_sample_number(case, "completion_tokens", integer=True)
            != completion_total
        ):
            raise EvidenceError("prefix-cache generic token totals disagree")
        reasoning = [sample.get("reasoning_tokens") for sample in records]
        expected_reasoning = (
            None
            if any(value is None for value in reasoning)
            else sum(int(value) for value in reasoning)
        )
        if case.get("reasoning_tokens") != expected_reasoning:
            raise EvidenceError("prefix-cache generic reasoning total disagrees")
        for field in (
            "aggregate_output_tps",
            "median_ttft_s",
            "median_e2e_s",
            "p95_e2e_s",
            "p95_ttft_s",
            "request_tps",
        ):
            if case.get(field) is not None:
                raise EvidenceError("prefix-cache case exposes an ambiguous generic rate")
        cache_metrics = case.get("prefix_cache")
        if not isinstance(cache_metrics, dict) or cache_metrics.get("profile_mode") != mode:
            raise EvidenceError("prefix-cache summary mode disagrees with model")
        if cache_metrics.get("prefix_target_words") != prefix_target:
            raise EvidenceError("prefix-cache summary target disagrees with frozen suite")
        session_wall_s = _prefix_cache_sample_number(case, "elapsed_s", positive=True)
        request_wall_s = sum(
            _prefix_cache_sample_number(sample, "elapsed_s", positive=True)
            for sample in records
        )
        if session_wall_s + 1e-9 < request_wall_s:
            raise EvidenceError("prefix-cache session wall is shorter than request walls")
        if not _prefix_cache_numbers_equal(
            cache_metrics.get("case_session_wall_s"), session_wall_s
        ):
            raise EvidenceError("prefix-cache session wall disagrees with case wall")
        expected_metrics = _summarize_prefix_cache_case(
            [{"result": sample} for sample in records],
            case_session_wall_s=session_wall_s,
        )
        if not _prefix_cache_values_equal(cache_metrics, expected_metrics):
            raise EvidenceError("prefix-cache summary aggregates disagree with samples")


def _prefix_cache_telemetry_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Accept only a JSON float in a cache telemetry floating-point column."""

    if type(value) is not float or not math.isfinite(value):
        raise EvidenceError(f"prefix-cache telemetry.{name} must be a finite JSON float")
    if minimum is not None and value < minimum:
        raise EvidenceError(f"prefix-cache telemetry.{name} is below its valid range")
    if maximum is not None and value > maximum:
        raise EvidenceError(f"prefix-cache telemetry.{name} is above its valid range")
    return value


def _validate_prefix_cache_telemetry_documents(
    telemetry: Any,
    chunks: list[Any],
    *,
    suite: Any,
) -> None:
    """Validate every cache telemetry scalar, phase, and row ordering.

    Telemetry is observational, but it remains part of a scalar-only cache
    evidence bundle.  A row-width check is insufficient: an arbitrary string
    in an otherwise width-correct row would then survive refreshed checksums.
    """

    projected_suite = _project_prefix_cache_suite(suite)
    case_ids = {case["case_id"] for case in projected_suite["cases"]}
    telemetry = _expect_object_keys(
        telemetry,
        {
            "chunk_count",
            "chunks",
            "columns",
            "sample_count",
            "schema_version",
            "segment_count",
        },
        name="prefix-cache telemetry index",
    )
    for key in ("chunk_count", "sample_count", "segment_count"):
        _prefix_cache_integer(
            telemetry[key], name=f"prefix-cache telemetry.{key}"
        )
    if (
        telemetry["schema_version"] != SCHEMA_VERSION
        or type(telemetry["schema_version"]) is not str
        or telemetry["columns"] != list(TELEMETRY_COLUMNS)
        or not isinstance(telemetry["chunks"], list)
        or telemetry["chunk_count"] != len(telemetry["chunks"])
        or telemetry["chunk_count"] != len(chunks)
        or telemetry["chunks"]
        != [f"telemetry-{index:04d}.json" for index in range(1, len(chunks) + 1)]
    ):
        raise EvidenceError("prefix-cache telemetry index does not match its exact schema")

    allowed_fixed_phases = {
        "artifact_validation",
        "between_cases",
        "first_request_after_start",
        "idle",
        "server_shutdown",
        "server_startup",
    }
    sample_count = 0
    segment_count = 0
    expected_sample_index = 1
    previous_phase: str | None = None
    previous_phase_segment: int | None = None
    previous_phase_sample_index = 0
    for chunk_index, chunk in enumerate(chunks, start=1):
        chunk = _expect_object_keys(
            chunk,
            {"sample_count", "schema_version", "segments"},
            name=f"prefix-cache telemetry chunk {chunk_index}",
        )
        if (
            chunk["schema_version"] != SCHEMA_VERSION
            or type(chunk["schema_version"]) is not str
            or not isinstance(chunk["segments"], list)
        ):
            raise EvidenceError("prefix-cache telemetry chunk does not match its schema")
        rows_in_chunk = 0
        for segment in chunk["segments"]:
            segment = _expect_object_keys(
                segment,
                {
                    "first_phase_sample_index",
                    "first_sample_index",
                    "phase",
                    "phase_segment",
                    "rows",
                },
                name="prefix-cache telemetry segment",
            )
            phase = _normalize_phase(segment["phase"])
            if phase != segment["phase"]:
                raise EvidenceError("prefix-cache telemetry phase is not canonical")
            if phase.startswith("case:"):
                if phase.removeprefix("case:") not in case_ids:
                    raise EvidenceError("prefix-cache telemetry phase has an unknown case")
            elif phase not in allowed_fixed_phases:
                raise EvidenceError("prefix-cache telemetry phase is not protocol allowlisted")
            phase_segment = _prefix_cache_integer(
                segment["phase_segment"],
                name="prefix-cache telemetry.phase_segment",
                positive=True,
            )
            first_phase_sample_index = _prefix_cache_integer(
                segment["first_phase_sample_index"],
                name="prefix-cache telemetry.first_phase_sample_index",
                positive=True,
            )
            first_sample_index = _prefix_cache_integer(
                segment["first_sample_index"],
                name="prefix-cache telemetry.first_sample_index",
                positive=True,
            )
            if first_sample_index != expected_sample_index:
                raise EvidenceError("prefix-cache telemetry sample order is inconsistent")
            if previous_phase_segment is None:
                if phase_segment != 1 or first_phase_sample_index != 1:
                    raise EvidenceError("prefix-cache telemetry first phase is inconsistent")
            elif phase_segment == previous_phase_segment:
                if (
                    phase != previous_phase
                    or first_phase_sample_index != previous_phase_sample_index + 1
                ):
                    raise EvidenceError("prefix-cache telemetry phase continuation is inconsistent")
            elif (
                phase_segment != previous_phase_segment + 1
                or first_phase_sample_index != 1
            ):
                raise EvidenceError("prefix-cache telemetry phase sequence is inconsistent")
            rows = segment["rows"]
            if not isinstance(rows, list) or not rows:
                raise EvidenceError("prefix-cache telemetry segment rows are invalid")
            previous_elapsed_s: float | None = None
            for row_index, row in enumerate(rows, start=1):
                if not isinstance(row, list) or len(row) != len(TELEMETRY_COLUMNS):
                    raise EvidenceError("prefix-cache telemetry row does not match its schema")
                elapsed_s = _prefix_cache_telemetry_float(
                    row[0], name="elapsed_s", minimum=0.0
                )
                if (
                    (row_index == 1 and not math.isclose(elapsed_s, 0.0, abs_tol=1e-9))
                    or (
                        previous_elapsed_s is not None
                        and elapsed_s + 1e-9 < previous_elapsed_s
                    )
                ):
                    raise EvidenceError("prefix-cache telemetry elapsed times are inconsistent")
                previous_elapsed_s = elapsed_s
                if not isinstance(row[1], bool):
                    raise EvidenceError("prefix-cache telemetry.gpu_error_present must be boolean")
                _prefix_cache_telemetry_float(
                    row[2], name="gpu_util_pct", minimum=0.0, maximum=100.0
                ) if row[2] is not None else None
                _prefix_cache_telemetry_float(
                    row[3], name="memory_util_pct", minimum=0.0, maximum=100.0
                ) if row[3] is not None else None
                _prefix_cache_telemetry_float(
                    row[4], name="power_w", minimum=0.0
                ) if row[4] is not None else None
                _prefix_cache_telemetry_float(
                    row[5], name="sm_clock_mhz", minimum=0.0
                ) if row[5] is not None else None
                _prefix_cache_telemetry_float(
                    row[6], name="temperature_c", minimum=-273.15, maximum=1_000.0
                ) if row[6] is not None else None
                for column, value in zip(TELEMETRY_COLUMNS[7:], row[7:], strict=True):
                    if value is not None:
                        _prefix_cache_integer(
                            value, name=f"prefix-cache telemetry.{column}"
                        )
                expected_sample_index += 1
                rows_in_chunk += 1
                sample_count += 1
            previous_phase = phase
            previous_phase_segment = phase_segment
            previous_phase_sample_index = first_phase_sample_index + len(rows) - 1
            segment_count += 1
        if _prefix_cache_integer(
            chunk["sample_count"], name="prefix-cache telemetry chunk.sample_count"
        ) != rows_in_chunk:
            raise EvidenceError("prefix-cache telemetry chunk sample count disagrees")
    if (
        telemetry["sample_count"] != sample_count
        or telemetry["segment_count"] != segment_count
    ):
        raise EvidenceError("prefix-cache telemetry totals disagree")


def _validate_prefix_cache_bundle_file_set(
    names: set[str], chunks: Any, *, include_checksums: bool
) -> None:
    """Require the complete cache bundle to use only protocol filenames.

    Generic bundles intentionally permit new scalar documents when their
    checksums are refreshed.  Cache evidence is a narrower protocol: accepting
    an otherwise checksummed ``trace.json`` would reintroduce arbitrary text
    into this scalar-only archive.  Telemetry chunk names are fixed too, rather
    than inferred from a mutable index document.
    """

    if not isinstance(chunks, list) or any(type(name) is not str for name in chunks):
        raise EvidenceError("prefix-cache bundle telemetry filenames are invalid")
    expected_chunks = [
        f"telemetry-{index:04d}.json" for index in range(1, len(chunks) + 1)
    ]
    if chunks != expected_chunks:
        raise EvidenceError("prefix-cache bundle telemetry filenames changed")
    expected = {
        "manifest.json",
        "samples.json",
        "summary.json",
        "telemetry.json",
        *expected_chunks,
    }
    if include_checksums:
        expected.add("checksums.json")
    if names != expected:
        raise EvidenceError("prefix-cache bundle file set does not match its protocol")


def _validate_memory_bundle_file_set(
    names: set[str], chunks: Any, *, include_checksums: bool
) -> None:
    if chunks != []:
        raise EvidenceError("memory evidence v1 must not publish telemetry chunks")
    expected = {
        "manifest.json",
        "samples.json",
        "summary.json",
        "telemetry.json",
    }
    if include_checksums:
        expected.add("checksums.json")
    if names != expected:
        raise EvidenceError("memory bundle file set does not match its protocol")


def _validate_summary_field_type(
    *,
    root_name: str,
    key: str,
    value: Any,
    name: str,
) -> None:
    if value is None:
        if (root_name, key) not in _SAFE_SUMMARY_NULLABLE_FIELDS:
            raise EvidenceError(f"aggregate field must not be null at {name}.{key}")
        return
    if key in _SAFE_SUMMARY_OBJECT_FIELDS_BY_ROOT.get(root_name, set()):
        if not isinstance(value, dict):
            raise EvidenceError(f"aggregate field must be an object at {name}.{key}")
        return
    if key in _SAFE_SUMMARY_BOOLEAN_FIELDS:
        if not isinstance(value, bool):
            raise EvidenceError(f"aggregate field must be boolean at {name}.{key}")
        return
    if key in _SAFE_SUMMARY_LIST_FIELDS:
        if not isinstance(value, list):
            raise EvidenceError(f"aggregate field must be a list at {name}.{key}")
        return
    if key in _SAFE_SUMMARY_STRING_KEYS or key == "reason" or "sha256" in key or key.endswith("digest"):
        if not isinstance(value, str):
            raise EvidenceError(f"aggregate field must be text at {name}.{key}")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"aggregate field must be numeric at {name}.{key}")


def _safe_summary_tree(value: Any, *, name: str, depth: int = 0) -> Any:
    """Project aggregate-only structures with explicit string and digest rules."""

    if depth > 16:
        raise EvidenceError(f"aggregate is too deep at {name}")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _finite(value, name=name)
    if isinstance(value, str):
        if "sha256" in name or name.endswith("digest"):
            return _sha256(value, name=name)
        raise EvidenceError(f"unsupported aggregate text at {name}")
    if isinstance(value, list):
        if not name.endswith("model_shard_sha256s"):
            raise EvidenceError(f"unexpected aggregate list at {name}")
        if any(not isinstance(item, str) for item in value):
            raise EvidenceError(f"aggregate digest list must be flat at {name}")
        return [_sha256(item, name=f"{name}[]") for item in value]
    if isinstance(value, dict):
        root_name = name.split(".", 1)[0].removesuffix("[]")
        root_keys = _SAFE_SUMMARY_KEYS_BY_ROOT.get(root_name)
        if root_keys is None:
            raise EvidenceError(f"unknown aggregate root at {name}")
        child_name = name.rsplit(".", 1)[-1].removesuffix("[]")
        is_root = name == root_name
        position_mapping = (
            root_name == "speculative_decoding"
            and child_name == "accepted_tokens_per_position"
        )
        if is_root:
            allowed_keys = root_keys
        elif position_mapping:
            allowed_keys = set()
        else:
            allowed_keys = _SAFE_SUMMARY_CHILD_KEYS.get((root_name, child_name))
            if allowed_keys is None:
                raise EvidenceError(f"unexpected nested aggregate object at {name}")
        result: dict[str, Any] = {}
        for key, item in sorted(value.items()):
            is_position_counter = position_mapping and key.isdigit()
            if key not in allowed_keys and not is_position_counter:
                raise EvidenceError(f"unknown aggregate field at {name}.{key}")
            if is_position_counter:
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise EvidenceError(
                        f"aggregate position counter must be numeric at {name}.{key}"
                    )
            else:
                _validate_summary_field_type(
                    root_name=root_name,
                    key=key,
                    value=item,
                    name=name,
                )
            if key in {"reason", "evidence", "timestamp", "case_id"} and "annotation" in name:
                continue
            if key == "reason":
                continue
            if key in {"cid", "pid", "process_start_ticks", "process_started_at_ns"}:
                continue
            if key.endswith("_path") or key.endswith("_target") or key in {
                "journal",
                "plan",
                "result",
                "stderr_log",
                "stdout_log",
                "telemetry",
                "worker_logic",
                "runtime_lock",
                "runtime_python",
                "snapshot",
            }:
                continue
            if isinstance(item, str):
                if "sha256" in key or key.endswith("digest"):
                    result[key] = _sha256(item, name=f"{name}.{key}")
                elif key in _SAFE_SUMMARY_STRING_KEYS:
                    result[key] = _safe_text(item, name=f"{name}.{key}")
                elif key in {"error_type", "stage"}:
                    result[key] = _safe_id(item, name=f"{name}.{key}")
                else:
                    raise EvidenceError(f"unsupported aggregate text at {name}.{key}")
            else:
                result[key] = _safe_summary_tree(
                    item, name=f"{name}.{key}", depth=depth + 1
                )
        return result
    raise EvidenceError(f"unsupported aggregate at {name}")


def _cold_start_scalar(value: str, *, key: str) -> int | float:
    if len(value) > 32:
        raise EvidenceError(f"cold-start safety scalar is too long: {key}")
    if key == "ple_allocated_blocks":
        if not _COLD_START_INTEGER_RE.fullmatch(value):
            raise EvidenceError("PLE allocated blocks must be an unsigned integer")
        try:
            result = int(value)
        except ValueError as error:
            raise EvidenceError("PLE allocated blocks are invalid") from error
        if result > 2**63 - 1:
            raise EvidenceError("PLE allocated blocks exceed the supported range")
        return result
    if not _COLD_START_DECIMAL_RE.fullmatch(value):
        raise EvidenceError(f"cold-start safety scalar has invalid syntax: {key}")
    try:
        result = float(value)
    except ValueError as error:
        raise EvidenceError(f"cold-start safety scalar is invalid: {key}") from error
    if not math.isfinite(result):
        raise EvidenceError(f"cold-start safety scalar is not finite: {key}")
    return result


def _validate_cold_start_safety_scalars(
    values: dict[str, Any], *, reason: str, source: bool
) -> dict[str, Any]:
    fields = _COLD_START_SAFETY_SCALARS[reason]
    if set(values) != set(fields):
        raise EvidenceError("cold-start safety annotation scalar set changed")
    projected: dict[str, Any] = {}
    for key in fields:
        value = values[key]
        if source:
            if not isinstance(value, str):
                raise EvidenceError("cold-start source scalar must be text")
            projected[key] = _cold_start_scalar(value, key=key)
        elif key == "ple_allocated_blocks":
            if type(value) is not int or not 0 <= value <= 2**63 - 1:
                raise EvidenceError(
                    "published PLE allocated blocks exceed the supported range"
                )
            projected[key] = value
        else:
            if type(value) is not float or not math.isfinite(value):
                raise EvidenceError("published cold-start scalar must be a finite float")
            projected[key] = value

    swap_growth = float(projected["swap_growth_mib"])
    safety_limit = float(projected["safety_limit_mib"])
    memavailable = float(projected["memavailable_gib"])
    psi = float(projected["memory_psi_full_avg10"])
    if (
        safety_limit <= 0
        or safety_limit > _COLD_START_MAX_MEMORY_MIB
        or swap_growth <= safety_limit
        or swap_growth > _COLD_START_MAX_MEMORY_MIB
        or memavailable < 0
        or memavailable > _COLD_START_MAX_MEMORY_GIB
        or psi < 0
        or psi > 100
    ):
        raise EvidenceError("cold-start safety scalars are outside supported ranges")
    return projected


def _project_cold_start_safety_annotation(
    value: Any, *, source: bool
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError("cold-start safety annotation must be an object")
    reason = value.get("reason")
    if reason not in _COLD_START_SAFETY_SCALARS:
        raise EvidenceError("cold-start safety annotation reason changed")
    assert isinstance(reason, str)

    if source:
        if set(value) != _COLD_START_SOURCE_ANNOTATION_FIELDS:
            raise EvidenceError("cold-start source annotation schema changed")
        if value.get("scope") != "startup" or value.get("measurement_valid") is not False:
            raise EvidenceError("cold-start source annotation classification changed")
        timestamp = _parse_timestamp(
            value.get("timestamp"), name="cold-start source annotation timestamp"
        )
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise EvidenceError("cold-start source annotation timestamp lacks a timezone")
        evidence = value.get("evidence")
        fields = _COLD_START_SAFETY_SCALARS[reason]
        if not isinstance(evidence, list) or len(evidence) != len(fields):
            raise EvidenceError("cold-start source annotation evidence changed")
        raw_scalars: dict[str, str] = {}
        for item in evidence:
            if not isinstance(item, str) or item.count("=") != 1:
                raise EvidenceError("cold-start source evidence syntax changed")
            key, scalar = item.split("=", 1)
            if key not in fields or key in raw_scalars:
                raise EvidenceError("cold-start source evidence keys changed")
            raw_scalars[key] = scalar
        scalars = _validate_cold_start_safety_scalars(
            raw_scalars, reason=reason, source=True
        )
    else:
        expected_fields = {"reason", *_COLD_START_SAFETY_SCALARS[reason]}
        if set(value) != expected_fields:
            raise EvidenceError("published cold-start annotation schema changed")
        scalars = _validate_cold_start_safety_scalars(
            {key: value[key] for key in _COLD_START_SAFETY_SCALARS[reason]},
            reason=reason,
            source=False,
        )
    return {"reason": reason, **scalars}


def _project_cold_start_safety_annotations(
    summary: dict[str, Any], *, source: bool
) -> list[dict[str, Any]]:
    if not source:
        annotations = summary.get("cold_start_safety_annotations")
        if not isinstance(annotations, list) or not annotations:
            raise EvidenceError("published cold-start annotations must be a nonempty list")
        projected = [
            _project_cold_start_safety_annotation(annotation, source=False)
            for annotation in annotations
        ]
        identities = [_canonical(annotation) for annotation in projected]
        if len(identities) != len(set(identities)):
            raise EvidenceError("published cold-start annotations are duplicated")
        ordered = sorted(projected, key=lambda annotation: annotation["reason"])
        if not _json_strict_equal(ordered, annotations):
            raise EvidenceError("published cold-start annotation order changed")
        return ordered

    by_summary_field: dict[str, list[dict[str, Any]]] = {}
    for field in ("measurement_annotations", "startup_measurement_annotations"):
        raw_annotations = summary.get(field)
        if raw_annotations is None:
            continue
        if not isinstance(raw_annotations, list):
            raise EvidenceError(f"{field} must be a list")
        projected: list[dict[str, Any]] = []
        for annotation in raw_annotations:
            reason = annotation.get("reason") if isinstance(annotation, dict) else None
            if reason not in _COLD_START_SAFETY_SCALARS:
                continue
            projected.append(
                _project_cold_start_safety_annotation(annotation, source=True)
            )
        identities = [_canonical(annotation) for annotation in projected]
        if len(identities) != len(set(identities)):
            raise EvidenceError("cold-start source annotations are duplicated")
        by_summary_field[field] = sorted(
            projected, key=lambda annotation: annotation["reason"]
        )

    measurement = by_summary_field.get("measurement_annotations", [])
    startup = by_summary_field.get("startup_measurement_annotations")
    if startup is not None and measurement != startup:
        raise EvidenceError("cold-start annotations disagree across summary fields")
    return startup if startup is not None else measurement


def _project_startup_safety_gates(value: Any) -> list[dict[str, Any]]:
    """Validate the scalar-only published form of typed startup gates."""

    if not isinstance(value, list):
        raise EvidenceError("startup_safety_gates must be a list")
    projected: list[dict[str, Any]] = []
    seen_metrics: set[str] = set()
    for raw_gate in value:
        try:
            gate = normalize_startup_safety_gate(raw_gate)
        except ValueError as error:
            raise EvidenceError(f"invalid startup safety gate: {error}") from error
        if not _json_strict_equal(gate, raw_gate):
            raise EvidenceError("startup safety-gate scalar schema changed")
        metric = gate["metric"]
        if metric in seen_metrics:
            raise EvidenceError(f"duplicate startup safety-gate metric: {metric}")
        seen_metrics.add(metric)
        projected.append(gate)
    ordered = sorted(projected, key=lambda gate: gate["metric"])
    if not _json_strict_equal(ordered, value):
        raise EvidenceError("startup safety gates are not in canonical order")
    return ordered


def _source_summary_startup_safety_gates(
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Require the typed summary list to equal both annotation mirrors."""

    mirror_gates: dict[str, list[dict[str, Any]]] = {}
    for field in ("measurement_annotations", "startup_measurement_annotations"):
        raw_annotations = summary.get(field)
        if raw_annotations is None:
            mirror_gates[field] = []
            continue
        if not isinstance(raw_annotations, list):
            raise EvidenceError(f"{field} must be a list")
        try:
            mirror_gates[field] = startup_safety_gates_from_annotations(
                raw_annotations
            )
        except ValueError as error:
            raise EvidenceError(
                f"invalid startup safety-gate annotation mirror: {error}"
            ) from error

    measurement = mirror_gates["measurement_annotations"]
    startup = mirror_gates["startup_measurement_annotations"]
    if not _json_strict_equal(measurement, startup):
        raise EvidenceError("startup safety gates disagree across summary mirrors")

    raw_gates = summary.get("startup_safety_gates")
    if raw_gates is None:
        if measurement:
            raise EvidenceError("startup_safety_gates is missing from the summary")
        return []
    projected = _project_startup_safety_gates(raw_gates)
    if not _json_strict_equal(projected, measurement):
        raise EvidenceError("startup safety gates disagree with annotation mirrors")
    if projected and summary.get("startup_measurement_valid") is not False:
        raise EvidenceError(
            "startup safety gates require startup_measurement_valid=false"
        )
    return projected


def _source_summary_startup_safety_gate_annotations(
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Require both source summary mirrors to retain typed journal identity."""

    mirrored: dict[str, list[dict[str, Any]]] = {}
    for field in ("measurement_annotations", "startup_measurement_annotations"):
        annotations = summary.get(field)
        if annotations is None:
            mirrored[field] = []
            continue
        if not isinstance(annotations, list):
            raise EvidenceError(f"{field} must be a list")
        try:
            mirrored[field] = startup_safety_gate_annotations_from_annotations(
                annotations
            )
        except ValueError as error:
            raise EvidenceError(
                f"invalid startup safety-gate annotation mirror: {error}"
            ) from error
    measurement = mirrored["measurement_annotations"]
    startup = mirrored["startup_measurement_annotations"]
    if not _json_strict_equal(measurement, startup):
        raise EvidenceError(
            "startup safety-gate annotations disagree across summary mirrors"
        )
    return measurement


def _validate_startup_safety_representation_consistency(
    gates: list[dict[str, Any]],
    cold_start_annotations: list[dict[str, Any]],
) -> None:
    """Reject contradictory typed and legacy swap-gate projections."""

    typed_swap = next(
        (gate for gate in gates if gate["metric"] == "startup_swap_growth"),
        None,
    )
    if typed_swap is None:
        return
    for annotation in cold_start_annotations:
        if (
            annotation.get("swap_growth_mib") != typed_swap["observed"]
            or annotation.get("safety_limit_mib") != typed_swap["limit"]
        ):
            raise EvidenceError(
                "typed and legacy startup swap safety gates disagree"
            )


def _project_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {"startup_safety_gates": []}
    unknown = set(summary) - _SUMMARY_KEYS
    if unknown:
        raise EvidenceError(f"unknown summary fields: {sorted(unknown)!r}")
    result: dict[str, Any] = {}
    scalar_fields = {
        "completed_cases",
        "input_prepare_time_s",
        "load_time_s",
    }
    for key in scalar_fields:
        if key in summary:
            value = summary[key]
            if value is None and key in {"input_prepare_time_s", "load_time_s"}:
                result[key] = None
            elif value is None:
                raise EvidenceError(f"{key} must not be null")
            else:
                result[key] = _finite(value, name=key)
    if "startup_measurement_valid" in summary:
        if not isinstance(summary["startup_measurement_valid"], bool):
            raise EvidenceError("startup_measurement_valid must be boolean")
        result["startup_measurement_valid"] = summary["startup_measurement_valid"]
    for key in (
        "context_limited_cases",
        "failed_cases",
        "measurement_invalid_cases",
        "unimplemented_cases",
        "unsupported_cases",
        "validation_failed_cases",
    ):
        if key not in summary:
            continue
        values = summary[key]
        if not isinstance(values, list):
            raise EvidenceError(f"{key} must be a list")
        result[key] = [_safe_id(value, name=f"{key}[]") for value in values]
    for key in ("status", "run_completion_status", "suite", "schema_version"):
        if key in summary and summary[key] is not None:
            result[key] = _safe_id(summary[key], name=key)
    if "run_completion_status" in summary and summary["run_completion_status"] is None:
        result["run_completion_status"] = None
    cases = summary.get("cases")
    if cases is not None:
        if not isinstance(cases, list):
            raise EvidenceError("summary cases must be a list")
        result["cases"] = [_project_case(case) for case in cases]
    if "memory_operation_summary" in summary:
        result["memory_operation_summary"] = _project_memory_operation_summary(
            summary["memory_operation_summary"]
        )
    first = summary.get("first_request_after_start")
    if first is not None:
        result["first_request"] = _project_request_result(first)
    for key in (
        "artifact_validation",
        "artifact_validation_telemetry",
        "cleanup_proof",
        "configuration",
        "first_request_telemetry",
        "llamacpp_dflash_evidence",
        "llamacpp_mtp_evidence",
        "memory",
        "metrics",
        "runtime",
        "shutdown_telemetry",
        "speculative_decoding",
        "startup_telemetry",
        "telemetry",
    ):
        if key in summary:
            value = summary[key]
            if value is None:
                if key not in _SAFE_SUMMARY_NULLABLE_ROOTS:
                    raise EvidenceError(f"aggregate root must not be null: {key}")
            elif not isinstance(value, dict):
                raise EvidenceError(f"aggregate root must be an object: {key}")
            if key == "artifact_validation" and isinstance(value, dict):
                value = dict(value)
                shard_keys = {
                    "model_shard_count",
                    "model_shard_sha256s",
                    "model_total_size_bytes",
                }
                populated = {name for name in shard_keys if value.get(name) is not None}
                if not populated:
                    for name in shard_keys:
                        value.pop(name, None)
                elif populated != shard_keys:
                    raise EvidenceError(
                        "artifact shard count, hashes, and total bytes must be all present"
                    )
                else:
                    count = value["model_shard_count"]
                    total_size = value["model_total_size_bytes"]
                    hashes = value["model_shard_sha256s"]
                    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                        raise EvidenceError("artifact shard count must be a positive integer")
                    if (
                        isinstance(total_size, bool)
                        or not isinstance(total_size, int)
                        or total_size <= 0
                    ):
                        raise EvidenceError(
                            "artifact shard total bytes must be a positive integer"
                        )
                    if not isinstance(hashes, list) or len(hashes) != count:
                        raise EvidenceError(
                            "artifact shard hashes must match the declared shard count"
                        )
            result[key] = _safe_summary_tree(value, name=key)
    for key in ("measurement_annotations", "startup_measurement_annotations"):
        annotations = summary.get(key)
        if annotations is not None:
            if not isinstance(annotations, list):
                raise EvidenceError(f"{key} must be a list")
            result[f"{key}_count"] = len(annotations)
    startup_safety_gates = _source_summary_startup_safety_gates(summary)
    result["startup_safety_gates"] = startup_safety_gates
    cold_start_annotations = _project_cold_start_safety_annotations(
        summary, source=True
    )
    _validate_startup_safety_representation_consistency(
        startup_safety_gates,
        cold_start_annotations,
    )
    if cold_start_annotations:
        result["cold_start_safety_annotations"] = cold_start_annotations
    dropped_types: dict[str, tuple[type, ...]] = {
        "artifact_verification": (dict,),
        "artifacts": (dict,),
        "error": (str, type(None)),
        "model": (dict,),
        "run_dir": (str,),
        "run_error": (dict, type(None)),
    }
    for key, expected_types in dropped_types.items():
        if key in summary and not isinstance(summary[key], expected_types):
            raise EvidenceError(f"{key} has an unexpected type")
    if any(
        isinstance(case, dict) and case.get("kind") == "cache"
        for case in result.get("cases", [])
    ):
        return _project_prefix_cache_summary(result)
    return result


def _normalize_phase(value: Any) -> str:
    if not isinstance(value, str):
        raise EvidenceError("telemetry phase must be text")
    if value.startswith("case:") or value.startswith("warmup:"):
        prefix, remainder = value.split(":", 1)
        case_id = remainder.rsplit(":", 1)[0]
        return f"{prefix}:{_safe_id(case_id, name='telemetry.case_id')}"
    return str(_safe_id(value, name="telemetry.phase"))


def _parse_timestamp(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str):
        raise EvidenceError(f"{name} must be an ISO timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvidenceError(f"invalid timestamp for {name}") from error


def _project_telemetry(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {
        "cached_kib",
        "gpu_error",
        "gpu_timestamp",
        "gpu_util_pct",
        "memavailable_kib",
        "memfree_kib",
        "memory_util_pct",
        "phase",
        "power_w",
        "sm_clock_mhz",
        "swapfree_kib",
        "swaptotal_kib",
        "temperature_c",
        "timestamp",
    }
    projected: list[dict[str, Any]] = []
    previous_phase: str | None = None
    phase_segment = 0
    phase_start: datetime | None = None
    phase_sample_index = 0
    for index, record in enumerate(records, 1):
        unknown = set(record) - allowed
        if unknown:
            raise EvidenceError(f"unknown telemetry fields: {sorted(unknown)!r}")
        phase = _normalize_phase(record.get("phase"))
        timestamp = _parse_timestamp(record.get("timestamp"), name="telemetry.timestamp")
        if phase != previous_phase:
            phase_segment += 1
            phase_sample_index = 0
            phase_start = timestamp
            previous_phase = phase
        phase_sample_index += 1
        assert phase_start is not None
        elapsed = (timestamp - phase_start).total_seconds()
        if elapsed < 0:
            raise EvidenceError("telemetry timestamps move backwards within a phase")
        sample: dict[str, Any] = {
            "elapsed_s": elapsed,
            "gpu_error_present": bool(record.get("gpu_error")),
            "phase": phase,
            "phase_sample_index": phase_sample_index,
            "phase_segment": phase_segment,
            "sample_index": index,
        }
        numeric = {
            "gpu_util_pct": "gpu_util_pct",
            "memory_util_pct": "memory_util_pct",
            "power_w": "power_w",
            "sm_clock_mhz": "sm_clock_mhz",
            "temperature_c": "temperature_c",
        }
        for source, target in numeric.items():
            if source in record:
                sample[target] = _finite(record[source], name=f"telemetry.{source}")
        for source, target in {
            "cached_kib": "cached_bytes",
            "memavailable_kib": "memavailable_bytes",
            "memfree_kib": "memfree_bytes",
            "swapfree_kib": "swapfree_bytes",
            "swaptotal_kib": "swaptotal_bytes",
        }.items():
            if source in record:
                value = _finite(record[source], name=f"telemetry.{source}")
                sample[target] = None if value is None else int(value) * 1024
        projected.append(sample)
    return projected


def _lifecycle(events: list[dict[str, Any]]) -> dict[str, Any]:
    names: list[str] = []
    timestamps: list[datetime] = []
    failure: dict[str, Any] | None = None
    for event in events:
        name = event.get("event")
        if not isinstance(name, str) or name not in _KNOWN_EVENTS:
            raise EvidenceError(f"unknown event type {name!r}")
        names.append(name)
        if event.get("timestamp") is not None:
            timestamps.append(_parse_timestamp(event["timestamp"], name="event.timestamp"))
        if name in {"run_aborted", "case_failed"} and failure is None:
            candidate: dict[str, Any] = {}
            for source, target in (("stage", "stage"), ("error_type", "exception_type")):
                if event.get(source) is not None:
                    candidate[target] = _safe_id(event[source], name=f"failure.{target}")
            if name == "case_failed" and event.get("case_id") is not None:
                candidate["case_id"] = _safe_id(event["case_id"], name="failure.case_id")
            failure = candidate or {"stage": "unspecified"}
    result: dict[str, Any] = {
        "event_count": len(events),
        "event_counts": dict(sorted(Counter(names).items())),
        "terminal": any(name in {"run_complete", "run_aborted"} for name in names),
        "terminal_event": next(
            (name for name in reversed(names) if name in {"run_complete", "run_aborted"}),
            None,
        ),
    }
    if len(timestamps) >= 2:
        elapsed = (timestamps[-1] - timestamps[0]).total_seconds()
        if elapsed < 0:
            raise EvidenceError("journal timestamps move backwards")
        result["journal_elapsed_s"] = elapsed
    if failure:
        result["failure"] = failure
    return result


def _validate_memory_source_events(
    events: list[dict[str, Any]],
    *,
    model: dict[str, Any],
    suite: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """Validate the no-resume, one-attempt memory journal topology."""

    cases = suite.get("cases")
    if not isinstance(cases, list):
        raise EvidenceError("memory source suite cases are missing")
    allowed_events = set(_MEMORY_PROTOCOL_EVENT_COUNTS) | {
        "llamacpp_spec_decode_metrics_snapshot",
        "measurement_complete",
        "measurement_started",
    }
    names = [event.get("event") for event in events]
    if any(name not in allowed_events for name in names) or names.count(
        "llamacpp_spec_decode_metrics_snapshot"
    ) > 1:
        raise EvidenceError("memory source journal contains a nonprotocol event")
    marker_counts = (
        names.count("measurement_started"),
        names.count("measurement_complete"),
    )
    if marker_counts not in {(0, 0), (1, 1)}:
        raise EvidenceError("memory source lifecycle marker cardinality changed")
    if marker_counts == (1, 1) and not (
        names.index("run_start")
        < names.index("measurement_started")
        < names.index("measurement_complete")
        < names.index("server_stopped")
    ):
        raise EvidenceError("memory source lifecycle markers are out of order")
    if "llamacpp_spec_decode_metrics_snapshot" in names and names.index(
        "llamacpp_spec_decode_metrics_snapshot"
    ) != names.index("server_stopped") - 1:
        raise EvidenceError("memory source metrics snapshot is out of protocol order")
    expected_names = [
        "run_start",
        "artifact_validation_complete",
        "server_ready",
        "first_request_complete",
    ]
    for _case in cases:
        expected_names.extend(
            [
                "case_start",
                *("request_complete" for _ in range(MEMORY_OPERATION_VARIANT_COUNT)),
                "case_complete",
            ]
        )
    expected_names.extend(["server_stopped", "run_complete"])
    protocol_events = [
        event for event in events if event.get("event") in _MEMORY_PROTOCOL_EVENT_COUNTS
    ]
    if [event.get("event") for event in protocol_events] != expected_names:
        raise EvidenceError("memory source journal protocol order changed")
    run_start, artifact, ready, first = protocol_events[:4]
    stopped, run_complete = protocol_events[-2:]
    if run_start.get("completed_cases_at_resume") != []:
        raise EvidenceError("memory source journal is a resumed run")
    if artifact.get("backend") != "llamacpp":
        raise EvidenceError("memory source artifact admission backend changed")
    if (
        ready.get("backend") != "llamacpp"
        or ready.get("keep_server_requested") is not False
        or first.get("backend") != "llamacpp"
        or stopped.get("backend") != "llamacpp"
        or run_complete.get("status") != "completed"
    ):
        raise EvidenceError("memory source server lifecycle changed")
    artifact_payload = {
        key: artifact[key]
        for key in (
            "elapsed_s",
            "model_sha256",
            "model_shard_count",
            "model_shard_sha256s",
            "model_total_size_bytes",
            "runtime_binary_sha256",
        )
        if key in artifact
    }
    _project_memory_artifact_validation(
        artifact_payload, model=model, source=True
    )
    _validate_memory_prime_result(first.get("result"))

    summary_cases = summary.get("cases")
    if not isinstance(summary_cases, list):
        raise EvidenceError("memory source summary cases are missing")
    summary_by_case = {
        case.get("case_id"): case for case in summary_cases if isinstance(case, dict)
    }
    if len(summary_by_case) != len(summary_cases):
        raise EvidenceError("memory source summary cases are duplicated")

    cursor = 4
    attempt_ids: set[str] = set()
    for case in cases:
        case_id = case["case_id"]
        start = protocol_events[cursor]
        attempt_id = start.get("attempt_id")
        if (
            start.get("case_id") != case_id
            or start.get("kind") != "memory"
            or type(start.get("concurrency")) is not int
            or start.get("concurrency") != 1
            or not isinstance(attempt_id, str)
            or not attempt_id
            or attempt_id in attempt_ids
        ):
            raise EvidenceError("memory source case_start identity changed")
        attempt_ids.add(attempt_id)
        cursor += 1
        for variant in range(MEMORY_OPERATION_VARIANT_COUNT):
            request = protocol_events[cursor]
            if (
                request.get("case_id") != case_id
                or request.get("attempt_id") != attempt_id
                or request.get("kind") != "memory"
                or type(request.get("repetition")) is not int
                or request.get("repetition") != variant
            ):
                raise EvidenceError("memory source request topology changed")
            projected_result = _project_memory_request_result(request.get("result"))
            validation = request.get("validation")
            if (
                not isinstance(validation, dict)
                or validation.get("passed") is not projected_result["passed"]
            ):
                raise EvidenceError("memory source request validation changed")
            cursor += 1
        complete = protocol_events[cursor]
        if (
            complete.get("case_id") != case_id
            or complete.get("attempt_id") != attempt_id
            or complete.get("kind") != "memory"
            or type(complete.get("concurrency")) is not int
            or complete.get("concurrency") != 1
        ):
            raise EvidenceError("memory source case_complete identity changed")
        request_events = protocol_events[
            cursor - MEMORY_OPERATION_VARIANT_COUNT : cursor
        ]
        request_elapsed = sum(
            float(request["result"]["elapsed_s"]) for request in request_events
        )
        case_elapsed = _finite(
            complete.get("elapsed_s"), name="memory source case_complete.elapsed_s"
        )
        case_passed = all(request["result"]["passed"] is True for request in request_events)
        summary_case = summary_by_case.get(case_id)
        if (
            case_elapsed is None
            or case_elapsed < request_elapsed - 1e-6
            or case_elapsed > request_elapsed + 5.0
            or complete.get("validation_passed") is not case_passed
            or not isinstance(summary_case, dict)
            or not _memory_aggregate_equal(summary_case.get("elapsed_s"), case_elapsed)
            or summary_case.get("validation_passed") is not case_passed
        ):
            raise EvidenceError("memory source case_complete outcome changed")
        cursor += 1


def _validate_memory_prime_result(value: Any) -> None:
    if not isinstance(value, dict):
        raise EvidenceError("memory source prime result must be an object")
    required = {
        "completion_tokens",
        "elapsed_s",
        "emission_events",
        "finish_reason",
        "prompt_tokens",
        "reasoning_tokens",
        "tool_calls",
        "ttft_s",
    }
    if not required <= set(value):
        raise EvidenceError("memory source prime result is incomplete")
    prompt_tokens = value.get("prompt_tokens")
    completion_tokens = value.get("completion_tokens")
    emission_events = value.get("emission_events")
    if (
        type(prompt_tokens) is not int
        or prompt_tokens <= 0
        or type(completion_tokens) is not int
        or completion_tokens <= 0
        or completion_tokens > 8
        or type(emission_events) is not int
        or emission_events <= 0
        or emission_events > completion_tokens
        or prompt_tokens + completion_tokens > MEMORY_OPERATION_CONTEXT_TOKENS
        or value.get("reasoning_tokens") is not None
        or value.get("finish_reason") not in {"length", "stop"}
        or value.get("tool_calls") != []
    ):
        raise EvidenceError("memory source prime counters changed")
    elapsed_s = _finite(value.get("elapsed_s"), name="memory source prime.elapsed_s")
    ttft_s = _finite(value.get("ttft_s"), name="memory source prime.ttft_s")
    if (
        elapsed_s is None
        or elapsed_s <= 0
        or ttft_s is None
        or ttft_s < 0
        or ttft_s > elapsed_s
    ):
        raise EvidenceError("memory source prime timing changed")


def _run_kind(plan: dict[str, Any]) -> str:
    schema = plan.get("schema_version")
    if schema == "direct-diffusion-v1":
        return "diffusion_direct"
    if schema == "trtllm-direct-v1":
        return "trtllm_direct"
    if schema == "llamacpp-perplexity-v1":
        return "llamacpp_perplexity"
    if schema in {1, 2}:
        return "serving"
    raise EvidenceError(f"unsupported plan schema {schema!r}")


def _parse_perplexity_progress(run_dir: Path, results_root: Path) -> list[dict[str, Any]]:
    stdout = run_dir / "logs" / "stdout.log"
    if not stdout.is_file():
        return []
    text = _secure_read(stdout, results_root, maximum=MAX_SOURCE_JSON_BYTES)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    data = lines[-1]
    matches = list(re.finditer(r"\[([1-9][0-9]*)\]([0-9]+(?:\.[0-9]+)?),", data))
    if not matches or "".join(match.group(0) for match in matches) != data:
        raise EvidenceError("perplexity stdout does not match the numeric grammar")
    return [
        {"chunk_index": int(match.group(1)), "cumulative_perplexity": float(match.group(2))}
        for match in matches
    ]


def _write_bundle(
    root: Path,
    relative: Path,
    files: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=False)
    checksums: dict[str, str] = {}
    for name, value in sorted(files.items()):
        data = _canonical(value)
        if len(data) > MAX_OUTPUT_FILE_BYTES:
            raise EvidenceError(f"generated evidence file exceeds limit: {relative / name}")
        (directory / name).write_bytes(data)
        checksums[name] = _hash_bytes(data)
    checksum_data = _canonical({"schema_version": SCHEMA_VERSION, "files": checksums})
    (directory / "checksums.json").write_bytes(checksum_data)
    bundle_hash = _hash_bytes(checksum_data)
    return bundle_hash, checksums


def _telemetry_files(samples: list[dict[str, Any]]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    names: list[str] = []
    segments: list[dict[str, Any]] = []
    for start in range(0, len(samples), TELEMETRY_SAMPLES_PER_CHUNK):
        chunk = samples[start : start + TELEMETRY_SAMPLES_PER_CHUNK]
        chunk_segments: list[dict[str, Any]] = []
        for sample in chunk:
            phase = sample["phase"]
            phase_segment = sample["phase_segment"]
            if (
                not chunk_segments
                or chunk_segments[-1]["phase"] != phase
                or chunk_segments[-1]["phase_segment"] != phase_segment
            ):
                chunk_segments.append(
                    {
                        "first_phase_sample_index": sample["phase_sample_index"],
                        "first_sample_index": sample["sample_index"],
                        "phase": phase,
                        "phase_segment": phase_segment,
                        "rows": [],
                    }
                )
            chunk_segments[-1]["rows"].append(
                [sample.get(column) for column in TELEMETRY_COLUMNS]
            )
        name = f"telemetry-{len(names) + 1:04d}.json"
        files[name] = {
            "sample_count": len(chunk),
            "schema_version": SCHEMA_VERSION,
            "segments": chunk_segments,
        }
        names.append(name)
        segments.extend(chunk_segments)
    files["telemetry.json"] = {
        "chunk_count": len(names),
        "chunks": names,
        "columns": list(TELEMETRY_COLUMNS),
        "sample_count": len(samples),
        "segment_count": len(segments),
        "schema_version": SCHEMA_VERSION,
    }
    return files


def _export_run(
    run_dir: Path,
    results_root: Path,
    output_root: Path,
    matrix_id: str | None,
    *,
    published_run_id: str | None = None,
) -> dict[str, Any]:
    source_run_id = run_dir.name
    _date_from_run_id(source_run_id)
    run_id = published_run_id if published_run_id is not None else source_run_id
    run_date = _date_from_run_id(run_id)
    plan = _load_json(run_dir / "plan.json", results_root)
    if not isinstance(plan, dict):
        raise EvidenceError("plan must be an object")
    events = (
        _load_json_lines(run_dir / "events.jsonl", results_root)
        if (run_dir / "events.jsonl").is_file()
        else []
    )
    summary = (
        _load_json(run_dir / "summary.json", results_root)
        if (run_dir / "summary.json").is_file()
        else None
    )
    if summary is not None and not isinstance(summary, dict):
        raise EvidenceError("summary must be an object")
    telemetry_records = (
        _load_json_lines(run_dir / "telemetry.jsonl", results_root)
        if (run_dir / "telemetry.jsonl").is_file()
        else []
    )
    kind = _run_kind(plan)
    status = _normalize_status(summary, events)
    lifecycle = _lifecycle(events)
    projected_model = _project_model(plan, summary)
    suite = _project_suite(plan)
    memory_protocol = bool(
        isinstance(suite, dict) and suite.get("id") == MEMORY_OPERATION_SUITE_ID
    )
    memory_case_id_mapping: dict[str, str] = {}
    if memory_protocol:
        source_model = plan.get("model")
        source_suite = plan.get("suite")
        if not isinstance(source_model, dict) or not isinstance(source_suite, dict):
            raise EvidenceError("memory run lacks a frozen model")
        assert isinstance(suite, dict)
        source_projected_suite = suite
        _bind_memory_summary_model(source_model=source_model, summary=summary)
        projected_model = _project_memory_model(source_model, projected_model)
        assert isinstance(summary, dict)
        _validate_memory_source_events(
            events,
            model=projected_model,
            suite=source_suite,
            summary=summary,
        )
        suite = _project_memory_suite(
            source_suite,
            source_model=source_model,
            binding_model=projected_model,
        )
        memory_case_id_mapping = {
            str(source_case["case_id"]): str(published_case["case_id"])
            for source_case, published_case in zip(
                source_projected_suite["cases"], suite["cases"], strict=True
            )
        }
    cache_protocol = (
        projected_model.get("prefix_cache_mode") is not None
        or (isinstance(suite, dict) and suite.get("id") == PREFIX_CACHE_SUITE_ID)
    )
    manifest: dict[str, Any] = {
        "artifacts": _collect_artifacts(plan, summary),
        "evidence_kind": kind,
        "lifecycle": lifecycle,
        "model": projected_model,
        "run_date_utc": run_date,
        "runtime": _project_runtime(plan, summary),
        "sanitization": {
            "free_form_text_included": False,
            "payloads_included": False,
            "policy": SANITIZATION_POLICY,
            "raw_identifiers_included": False,
        },
        "schema_version": SCHEMA_VERSION,
        "source_run_id": run_id,
        "status": status,
    }
    hardware = _project_hardware(plan)
    if hardware:
        manifest["hardware"] = hardware
    if matrix_id:
        manifest["matrix_id"] = _safe_id(matrix_id, name="matrix_id")
    if suite:
        manifest["suite"] = suite
    if memory_protocol:
        manifest["artifact_validation"] = (
            summary.get("artifact_validation") if isinstance(summary, dict) else None
        )
        manifest = _project_memory_manifest(manifest, source=True)
    if cache_protocol:
        # Materialize an exact protocol manifest rather than publishing the
        # broader serving manifest.  This is also the export-side counterpart
        # to the verifier's strict re-projection below.
        manifest = _project_prefix_cache_manifest(manifest, source=True)
    requests = _project_requests(events, summary, evidence_kind=kind)
    if memory_protocol:
        unexpected = [
            sample
            for sample in requests
            if sample.get("sample_type") == "measured_request"
            and sample.get("kind") != "memory"
        ]
        if unexpected:
            raise EvidenceError("memory run contains a nonprotocol measured sample")
        raw_memory_samples = [sample for sample in requests if sample.get("kind") == "memory"]
        if len(raw_memory_samples) != len(MEMORY_OPERATION_SCENARIO_IDS) * MEMORY_OPERATION_VARIANT_COUNT or any(
            sample.get("selected_attempt") is not True for sample in raw_memory_samples
        ):
            raise EvidenceError("memory run contains missing, retried, or unselected samples")
        requests = [
            sample
            for sample in requests
            if sample.get("kind") == "memory"
            and sample.get("selected_attempt") is True
        ]
        requests = [
            {**sample, "sample_index": index}
            for index, sample in enumerate(requests, start=1)
        ]
    if cache_protocol:
        unexpected = [
            sample
            for sample in requests
            if sample.get("sample_type") == "measured_request"
            and sample.get("kind") != "cache"
        ]
        if unexpected:
            raise EvidenceError("prefix-cache run contains a nonprotocol measured sample")
        requests = [sample for sample in requests if sample.get("kind") == "cache"]
        requests = [
            {**sample, "sample_index": index}
            for index, sample in enumerate(requests, start=1)
        ]
    if kind == "llamacpp_perplexity":
        requests.extend(
            {
                "sample_index": len(requests) + index,
                "sample_type": "perplexity_progress",
                **sample,
            }
            for index, sample in enumerate(
                _parse_perplexity_progress(run_dir, results_root), 1
            )
        )
    telemetry = _project_telemetry(telemetry_records)
    if memory_protocol:
        telemetry = []
    projected_summary = _project_summary(summary)
    try:
        journal_annotations = startup_safety_gate_annotations_from_annotations(
            measurement_annotations(events)
        )
    except ValueError as error:
        raise EvidenceError(f"invalid startup safety-gate journal: {error}") from error
    summary_annotations = (
        _source_summary_startup_safety_gate_annotations(summary)
        if summary is not None
        else []
    )
    if not _json_strict_equal(journal_annotations, summary_annotations):
        raise EvidenceError(
            "startup safety-gate annotations disagree between journal and summary"
        )
    journal_safety_gates = sorted(
        (annotation["safety_gate"] for annotation in journal_annotations),
        key=lambda gate: gate["metric"],
    )
    if not _json_strict_equal(
        journal_safety_gates,
        projected_summary.get("startup_safety_gates", []),
    ):
        raise EvidenceError("startup safety gates disagree between journal and summary")
    if memory_protocol:
        requests, projected_summary = _translate_memory_case_ids(
            requests,
            projected_summary,
            mapping=memory_case_id_mapping,
        )
        projected_summary = _project_memory_summary_document(
            projected_summary, source=True
        )
        if manifest["status"] != projected_summary["status"]:
            raise EvidenceError("memory manifest status disagrees with its aggregates")
    _validate_agentic_aggregates(
        requests,
        projected_summary,
        suite=suite,
        terminal=lifecycle.get("terminal_event") == "run_complete",
    )
    _validate_memory_aggregates(
        requests,
        projected_summary,
        model=manifest["model"],
        suite=suite,
        terminal=lifecycle.get("terminal_event") == "run_complete",
    )
    _validate_prefix_cache_aggregates(
        requests,
        projected_summary,
        model=manifest["model"],
        suite=suite,
    )
    telemetry_files = _telemetry_files(telemetry)
    if memory_protocol and telemetry_files != _telemetry_files([]):
        raise EvidenceError("memory evidence telemetry projection changed")
    if cache_protocol:
        telemetry_index = telemetry_files["telemetry.json"]
        assert isinstance(telemetry_index, dict)
        _validate_prefix_cache_telemetry_documents(
            telemetry_index,
            [telemetry_files[name] for name in telemetry_index["chunks"]],
            suite=suite,
        )
    relative = Path("runs") / run_id
    bundle_files: dict[str, Any] = {
        "manifest.json": manifest,
        "samples.json": {
            "sample_count": len(requests),
            "samples": requests,
            "schema_version": SCHEMA_VERSION,
        },
        "summary.json": {
            "aggregates": projected_summary,
            "schema_version": SCHEMA_VERSION,
        },
        **telemetry_files,
    }
    if cache_protocol:
        telemetry_index = bundle_files["telemetry.json"]
        assert isinstance(telemetry_index, dict)
        _validate_prefix_cache_bundle_file_set(
            set(bundle_files),
            telemetry_index["chunks"],
            include_checksums=False,
        )
    if memory_protocol:
        telemetry_index = bundle_files["telemetry.json"]
        assert isinstance(telemetry_index, dict)
        _validate_memory_bundle_file_set(
            set(bundle_files), telemetry_index["chunks"], include_checksums=False
        )
    bundle_hash, _ = _write_bundle(
        output_root,
        relative,
        bundle_files,
    )
    return {
        "bundle_sha256": bundle_hash,
        "evidence_kind": kind,
        "file": str(relative / "manifest.json"),
        "matrix_id": matrix_id,
        "measurement_terminal": lifecycle.get("terminal_event") == "run_complete",
        "run_id": run_id,
        "status": manifest["status"],
    }


def _loop_content_hash(value: Any, *, length: int = 64) -> str:
    return hashlib.sha256(_canonical(value).rstrip(b"\n")).hexdigest()[:length]


def _project_loop_case(value: Any, *, variant: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise EvidenceError("loop case must be an object")
    phase = value.get("phase")
    rlm_fields = {
        "case_id",
        "context_length",
        "max_concurrent_subcalls",
        "max_depth",
        "max_iterations",
        "max_output_tokens",
        "max_total_tokens",
        "phase",
        "replicate",
        "row_index",
        "task",
        "timeout_s",
        "treatment",
    }
    halo_fields = {
        "case_id",
        "max_depth",
        "max_output_tokens",
        "max_parallel",
        "max_turns",
        "phase",
        "seed",
        "timeout_s",
        "trace_count",
        "treatment",
    }
    if variant in {"legacy_v2", "current_v2"}:
        rlm_fields.add("reasoning_control")
        halo_fields.add("reasoning_effort")
    if variant == "current_v2":
        rlm_fields.update(
            {"admission_status", "compaction", "compaction_threshold_pct"}
        )
    expected_fields = rlm_fields if phase == "rlm" else halo_fields
    if variant not in {"legacy_v1", "legacy_v2", "current_v2"} or set(value) != expected_fields:
        raise EvidenceError("loop case schema changed")
    projected: dict[str, Any] = {}
    for key, item in value.items():
        if key in _LOOP_CASE_STRING_FIELDS:
            projected[key] = _safe_id(
                item,
                name=f"loop case.{key}",
                nullable=key in {"reasoning_control", "reasoning_effort"},
            )
        elif key in _LOOP_CASE_BOOLEAN_FIELDS:
            if item is not None and type(item) is not bool:
                raise EvidenceError(f"loop case {key} must be boolean or null")
            projected[key] = item
        elif key in _LOOP_CASE_INTEGER_FIELDS:
            if item is not None and (type(item) is not int or item < 0):
                raise EvidenceError(
                    f"loop case {key} must be a non-negative integer or null"
                )
            projected[key] = item
        elif key == "compaction_threshold_pct":
            number = _finite(item, name=f"loop case.{key}")
            if number is not None and not 0 < float(number) < 1:
                raise EvidenceError("loop compaction threshold is invalid")
            projected[key] = number
        else:  # pragma: no cover - guarded by the exact field partition above
            raise EvidenceError(f"loop case field is unprojected: {key}")
    phase = projected.get("phase")
    treatment = projected.get("treatment")
    expected_treatments = {
        "rlm": {"direct", "rlm_depth1", "rlm_depth1_forced_compaction", "rlm_depth2"},
        "halo": {"halo_depth0", "halo_depth1", "halo_depth2"},
    }
    if phase not in expected_treatments or treatment not in expected_treatments[phase]:
        raise EvidenceError("loop case phase or treatment is unsupported")
    if (
        (phase == "rlm" and projected.get("context_length") not in _LOOP_CONTEXT_LENGTHS)
        or (phase == "rlm" and projected.get("task") not in _LOOP_TASKS)
        or projected.get("reasoning_control") not in {None, "fixed_unsupported"}
        or projected.get("reasoning_effort") not in {None, "none"}
        or projected.get("admission_status")
        not in {None, "admitted", "held_child_compaction_unverified"}
    ):
        raise EvidenceError("loop case dimensions are outside the frozen domains")
    if variant == "current_v2" and phase == "rlm":
        expected_admission = (
            "held_child_compaction_unverified"
            if treatment == "rlm_depth2"
            else "admitted"
        )
        expected_compaction = treatment != "direct"
        expected_threshold = (
            None
            if treatment == "direct"
            else (0.20 if treatment == "rlm_depth1_forced_compaction" else 0.85)
        )
        if (
            projected.get("admission_status") != expected_admission
            or projected.get("compaction") is not expected_compaction
            or projected.get("compaction_threshold_pct") != expected_threshold
        ):
            raise EvidenceError("loop case compaction semantics changed")
    case_id = projected["case_id"]
    if not re.fullmatch(rf"{phase}-[0-9a-f]{{16}}", str(case_id)):
        raise EvidenceError("loop case identifier is invalid")
    expected_id = f"{phase}-{_loop_content_hash({key: value[key] for key in value if key != 'case_id'}, length=16)}"
    if case_id != expected_id:
        raise EvidenceError("loop case identifier does not bind its dimensions")
    return projected


def _project_loop_protocol_section(
    value: Any,
    *,
    name: str,
    allowed: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, dict) or not set(value) <= allowed:
        raise EvidenceError(f"loop {name} protocol schema changed")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key == "compaction_diagnostic":
            if not isinstance(item, dict) or set(item) != {
                "lengths",
                "row_indices",
                "tasks",
                "threshold_pct",
            }:
                raise EvidenceError("loop compaction diagnostic schema changed")
            result[key] = _project_loop_protocol_section(
                item,
                name="compaction diagnostic",
                allowed=frozenset(item),
            )
        elif isinstance(item, list):
            if not item:
                raise EvidenceError(f"loop {name}.{key} must not be empty")
            projected_items: list[Any] = []
            for entry in item:
                if type(entry) is int and entry >= 0:
                    projected_items.append(entry)
                else:
                    projected_items.append(
                        _safe_id(entry, name=f"loop {name}.{key}")
                    )
            if len({_canonical(entry) for entry in projected_items}) != len(
                projected_items
            ):
                raise EvidenceError(f"loop {name}.{key} contains duplicates")
            result[key] = projected_items
        elif item is None or type(item) is bool:
            result[key] = item
        elif type(item) is int:
            if item < 0:
                raise EvidenceError(f"loop {name}.{key} must be non-negative")
            result[key] = item
        elif isinstance(item, float):
            number = _finite(item, name=f"loop {name}.{key}")
            if number is None or number < 0:
                raise EvidenceError(f"loop {name}.{key} must be non-negative")
            result[key] = number
        else:
            result[key] = _safe_id(item, name=f"loop {name}.{key}")
    return result


def _project_loop_model(value: Any, *, profile_id: str) -> dict[str, Any]:
    required = {
        "architecture",
        "backend",
        "estimated_ram_gib",
        "id",
        "image",
        "image_digest",
        "lifecycle",
        "max_context",
        "native_context",
        "quantization",
        "revision",
        "source",
        "startup_timeout_s",
        "support_status",
        "tasks",
    }
    if (
        not isinstance(value, dict)
        or not required <= set(value)
        or frozenset(value)
        not in (
            _LOOP_MODEL_LEGACY_SOURCE_FIELDS,
            _LOOP_MODEL_PRE_OMISSION_SOURCE_FIELDS,
            _LOOP_MODEL_PRE_OMISSION_SOURCE_FIELDS_WITH_PLE_CACHE,
            _LOOP_MODEL_SOURCE_FIELDS,
            _LOOP_MODEL_SOURCE_FIELDS_WITH_PLE_CACHE,
        )
        or value.get("id") != profile_id
    ):
        raise EvidenceError("loop model source schema changed")
    projected = _project_model({"model": value}, None)
    projected.update(_project_sglang_provenance(value))
    if value.get("recipe_revision") is not None:
        projected["recipe_revision"] = _revision(
            value["recipe_revision"], name="loop model.recipe_revision"
        )
    projected["container_image"] = _safe_id(
        value.get("image"), name="loop model.container_image"
    )
    projected["container_image_sha256"] = _sha256(
        value.get("image_digest"), name="loop model.container_image"
    )
    return projected


def _project_loop_plan(
    value: Any, *, run_id: str, source_group: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError("loop plan must be an object")
    schema = value.get("schema_version")
    protocol = value.get("protocol_version")
    expected_fields = set(_LOOP_PLAN_TOP_FIELDS)
    if "rlm_compaction_admission" in value:
        expected_fields.add("rlm_compaction_admission")
    if set(value) != expected_fields or (schema, protocol) not in {(1, 1), (2, 2)}:
        raise EvidenceError("loop plan schema changed")
    if schema == 1 and "rlm_compaction_admission" in value:
        raise EvidenceError("legacy loop plan has a current-only admission section")
    variant = (
        "legacy_v1"
        if schema == 1
        else (
            "current_v2"
            if "rlm_compaction_admission" in value
            else "legacy_v2"
        )
    )
    _parse_timestamp(value.get("created_at"), name="loop plan.created_at")
    if not isinstance(value.get("description"), str):
        raise EvidenceError("loop plan description must be text")
    fingerprint = _sha256(value.get("fingerprint"), name="loop plan fingerprint")
    integrity = _sha256(value.get("integrity_hash"), name="loop plan integrity")
    without_integrity = {
        key: item for key, item in value.items() if key != "integrity_hash"
    }
    if _loop_content_hash(without_integrity) != integrity:
        raise EvidenceError("loop plan integrity hash changed")
    fingerprint_payload = {
        key: item for key, item in without_integrity.items() if key != "fingerprint"
    }
    if _loop_content_hash(fingerprint_payload) != fingerprint:
        raise EvidenceError("loop plan fingerprint changed")
    if not run_id.endswith(f"-{fingerprint[:8]}"):
        raise EvidenceError("loop run directory does not bind its plan fingerprint")

    campaign_id = _safe_id(value.get("campaign_id"), name="loop campaign ID")
    repository = value.get("repository")
    if (
        not isinstance(repository, dict)
        or set(repository) != {"clean", "revision"}
        or repository.get("clean") is not True
    ):
        raise EvidenceError("loop repository provenance changed")
    repository_revision = _revision(
        repository.get("revision"), name="loop repository revision"
    )
    if value.get("upstreams") != _LOOP_UPSTREAMS:
        raise EvidenceError("loop upstream pins changed")
    worker = value.get("worker")
    if worker != {"image": _LOOP_WORKER_IMAGE, "isolation": "docker"}:
        raise EvidenceError("loop worker pin changed")

    dataset = value.get("dataset")
    if not isinstance(dataset, dict) or set(dataset) != {
        "revision",
        "rows_per_split",
        "selected_files",
        "source",
    }:
        raise EvidenceError("loop dataset schema changed")
    if (
        dataset.get("source") != _LOOP_UPSTREAMS["babilong_source"]
        or dataset.get("revision") != _LOOP_UPSTREAMS["babilong_revision"]
        or type(dataset.get("rows_per_split")) is not int
        or dataset["rows_per_split"] <= 0
        or not isinstance(dataset.get("selected_files"), list)
    ):
        raise EvidenceError("loop dataset pin changed")
    dataset_artifacts: list[dict[str, Any]] = []
    seen_dataset_targets: set[str] = set()
    for artifact in dataset["selected_files"]:
        if not isinstance(artifact, dict) or set(artifact) != {
            "context_length",
            "sha256",
            "size_bytes",
            "task",
        }:
            raise EvidenceError("loop dataset artifact schema changed")
        context_length = _safe_id(
            artifact.get("context_length"), name="loop dataset context"
        )
        task = _safe_id(artifact.get("task"), name="loop dataset task")
        if context_length not in _LOOP_CONTEXT_LENGTHS or task not in _LOOP_TASKS:
            raise EvidenceError("loop dataset artifact selection changed")
        target = f"{context_length}-{task}"
        if target in seen_dataset_targets:
            raise EvidenceError("loop dataset artifact is duplicated")
        seen_dataset_targets.add(target)
        size = artifact.get("size_bytes")
        if type(size) is not int or size <= 0:
            raise EvidenceError("loop dataset artifact size is invalid")
        dataset_artifacts.append(
            {
                "sha256": _sha256(
                    artifact.get("sha256"), name="loop dataset artifact"
                ),
                "size_bytes": size,
                "target": target,
            }
        )

    models = value.get("models")
    if not isinstance(models, dict) or not models:
        raise EvidenceError("loop plan models are missing")
    projected_models = [
        _project_loop_model(models[profile_id], profile_id=profile_id)
        for profile_id in sorted(models)
    ]
    model_ids = {model["id"] for model in projected_models}
    rlm_allowed = frozenset(
        {
            "compaction",
            "compaction_diagnostic",
            "compaction_threshold_pct",
            "direct_lengths",
            "direct_timeout_s",
            "episode_timeout_s",
            "lengths",
            "max_concurrent_subcalls",
            "max_iterations",
            "max_output_tokens",
            "max_total_tokens",
            "model_profile",
            "reasoning_control",
            "recursive_depth2_lengths",
            "recursive_depth2_rows",
            "recursive_depth2_tasks",
            "row_indices",
            "tasks",
            "worker_isolation",
        }
    )
    halo_allowed = frozenset(
        {
            "depth2_seeds",
            "depth2_trace_counts",
            "depths",
            "episode_timeout_s",
            "max_output_tokens",
            "max_parallel",
            "max_turns",
            "model_profiles",
            "reasoning_effort",
            "seeds",
            "trace_counts",
        }
    )
    rlm_v1_fields = {
        "direct_lengths",
        "direct_timeout_s",
        "episode_timeout_s",
        "lengths",
        "max_concurrent_subcalls",
        "max_iterations",
        "max_output_tokens",
        "max_total_tokens",
        "model_profile",
        "recursive_depth2_lengths",
        "recursive_depth2_rows",
        "recursive_depth2_tasks",
        "row_indices",
        "tasks",
        "worker_isolation",
    }
    rlm_v2_fields = rlm_v1_fields | {"reasoning_control"}
    rlm_current_fields = rlm_v2_fields | {
        "compaction",
        "compaction_threshold_pct",
    }
    expected_rlm_fields = {
        "legacy_v1": rlm_v1_fields,
        "legacy_v2": rlm_v2_fields,
        "current_v2": rlm_current_fields,
    }[variant]
    raw_rlm = value.get("rlm")
    if not isinstance(raw_rlm, dict):
        raise EvidenceError("loop RLM protocol is missing")
    if variant == "current_v2" and "compaction_diagnostic" in raw_rlm:
        expected_rlm_fields = expected_rlm_fields | {"compaction_diagnostic"}
    if set(raw_rlm) != expected_rlm_fields:
        raise EvidenceError("loop RLM protocol variant changed")
    halo_v1_fields = {
        "depth2_seeds",
        "depth2_trace_counts",
        "depths",
        "episode_timeout_s",
        "max_output_tokens",
        "max_parallel",
        "max_turns",
        "model_profiles",
        "seeds",
        "trace_counts",
    }
    expected_halo_fields = (
        halo_v1_fields
        if variant == "legacy_v1"
        else halo_v1_fields | {"reasoning_effort"}
    )
    raw_halo = value.get("halo")
    if not isinstance(raw_halo, dict) or set(raw_halo) != expected_halo_fields:
        raise EvidenceError("loop HALO protocol variant changed")
    rlm = _project_loop_protocol_section(
        raw_rlm, name="rlm", allowed=rlm_allowed
    )
    halo = _project_loop_protocol_section(
        raw_halo, name="halo", allowed=halo_allowed
    )
    if rlm.get("model_profile") not in model_ids:
        raise EvidenceError("loop RLM profile is not frozen in the plan")
    halo_profiles = halo.get("model_profiles")
    referenced_models = {str(rlm.get("model_profile")), *map(str, halo_profiles or [])}
    if (
        not isinstance(halo_profiles, list)
        or not halo_profiles
        or not set(halo_profiles) <= model_ids
        or model_ids != referenced_models
    ):
        raise EvidenceError("loop HALO profiles are not frozen in the plan")

    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise EvidenceError("loop plan cases are missing")
    projected_cases = [
        _project_loop_case(case, variant=variant) for case in cases
    ]
    case_ids = [case["case_id"] for case in projected_cases]
    if len(case_ids) != len(set(case_ids)):
        raise EvidenceError("loop plan contains duplicate case identifiers")

    window = value.get("window")
    if not isinstance(window, dict) or set(window) != {
        "cleanup_reserve_s",
        "hard_stop_at",
        "measurement_stop_at",
        "rlm_stop_at",
    }:
        raise EvidenceError("loop window schema changed")
    for key in ("rlm_stop_at", "measurement_stop_at", "hard_stop_at"):
        _parse_timestamp(window.get(key), name=f"loop window.{key}")
    if type(window.get("cleanup_reserve_s")) is not int or window["cleanup_reserve_s"] <= 0:
        raise EvidenceError("loop cleanup reserve is invalid")

    compaction_admission = value.get("rlm_compaction_admission")
    if compaction_admission is not None:
        compaction_admission = _project_loop_protocol_section(
            compaction_admission,
            name="rlm compaction admission",
            allowed=frozenset(
                {
                    "depth1_admitted",
                    "depth2_admitted",
                    "enabled",
                    "headroom_tokens",
                    "output_reserve_tokens",
                    "package_context_tokens",
                    "served_context_tokens",
                    "threshold_pct",
                    "threshold_tokens",
                }
            ),
        )
        if set(compaction_admission) != {
            "depth1_admitted",
            "depth2_admitted",
            "enabled",
            "headroom_tokens",
            "output_reserve_tokens",
            "package_context_tokens",
            "served_context_tokens",
            "threshold_pct",
            "threshold_tokens",
        }:
            raise EvidenceError("loop compaction admission schema changed")
    if variant == "current_v2":
        if (
            rlm.get("compaction") is not True
            or rlm.get("compaction_threshold_pct") != 0.85
            or rlm.get("reasoning_control") != "fixed_unsupported"
            or halo.get("reasoning_effort") != "none"
            or compaction_admission is None
        ):
            raise EvidenceError("current loop protocol semantics changed")
    elif variant == "legacy_v2" and (
        rlm.get("reasoning_control") != "fixed_unsupported"
        or halo.get("reasoning_effort") != "none"
        or compaction_admission is not None
    ):
        raise EvidenceError("legacy-v2 loop protocol semantics changed")

    return {
        "campaign_id": campaign_id,
        "cases": projected_cases,
        "dataset": {
            "artifacts": sorted(dataset_artifacts, key=lambda item: item["target"]),
            "revision": _revision(dataset["revision"], name="loop dataset revision"),
            "rows_per_split": dataset["rows_per_split"],
            "source": _safe_id(dataset["source"], name="loop dataset source"),
        },
        "models": projected_models,
        "plan_fingerprint": fingerprint,
        "plan_integrity_sha256": integrity,
        "plan_schema_version": schema,
        "protocol": {
            "halo": halo,
            "rlm": rlm,
            "rlm_compaction_admission": compaction_admission,
        },
        "protocol_version": protocol,
        "repository": {
            "clean": True,
            "revision": repository_revision,
        },
        "source_group": _safe_id(source_group, name="loop source group"),
        "upstreams": dict(_LOOP_UPSTREAMS),
        "worker": {
            "container_image": _LOOP_WORKER_IMAGE.split("@", 1)[0],
            "container_image_sha256": _sha256(
                _LOOP_WORKER_IMAGE.split("@", 1)[1], name="loop worker image"
            ),
            "isolation": "docker",
        },
    }


def _loop_case_dimensions_match(event: dict[str, Any], case: dict[str, Any]) -> bool:
    return all(
        key not in event or _json_strict_equal(event[key], case.get(key))
        for key in _LOOP_BOUND_DIMENSION_FIELDS
    )


def _project_loop_measurement(value: Any, *, source: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError("loop measurement must be an object")
    ignored = {"event", "timestamp"} if source else set()
    permitted = (
        _LOOP_MEASUREMENT_FIELDS
        if source
        else _LOOP_MEASUREMENT_OUTPUT_FIELDS | {"sample_index"}
    )
    if not set(value) - ignored <= permitted:
        raise EvidenceError("loop completed-case schema changed")
    required = {"attempt", "case_id", "phase", "profile_id", "treatment"}
    if not required <= set(value):
        raise EvidenceError("loop completed-case identity is incomplete")
    if source and value.get("event") != "case_complete":
        raise EvidenceError("loop measurement source event is invalid")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in ignored:
            continue
        if key == "sample_index":
            if type(item) is not int or item <= 0:
                raise EvidenceError("loop sample index must be positive")
            result[key] = item
        elif key in _LOOP_MEASUREMENT_STRING_FIELDS:
            result[key] = _safe_id(
                item,
                name=f"loop measurement.{key}",
                nullable=key in {"reasoning_control", "reasoning_effort"},
            )
        elif key in _LOOP_MEASUREMENT_BOOLEAN_FIELDS:
            if item is not None and type(item) is not bool:
                raise EvidenceError(f"loop measurement {key} must be boolean or null")
            result[key] = item
        elif key in _LOOP_MEASUREMENT_INTEGER_FIELDS or key == "executed_tool_call_count":
            if item is not None and (type(item) is not int or item < 0):
                raise EvidenceError(
                    f"loop measurement {key} must be a non-negative integer or null"
                )
            result[
                "executed_tool_call_count" if source and key == "tool_calls" else key
            ] = item
        elif key in _LOOP_MEASUREMENT_NUMERIC_FIELDS:
            number = _finite(item, name=f"loop measurement.{key}")
            if number is not None and number < 0:
                raise EvidenceError(f"loop measurement {key} must be non-negative")
            result[key] = number
        else:  # pragma: no cover - guarded by the exact field partition above
            raise EvidenceError(f"loop measurement field is unprojected: {key}")
    return result


def _validate_loop_journal(
    events: list[dict[str, Any]], *, plan: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = {case["case_id"]: case for case in plan["cases"]}
    models = {model["id"] for model in plan["models"]}
    starts: set[tuple[str, int, str]] = set()
    completed: set[str] = set()
    terminal_cases: set[str] = set()
    measurements: list[dict[str, Any]] = []
    previous_timestamp: datetime | None = None
    for event in events:
        if not isinstance(event, dict) or not set(event) <= _LOOP_EVENT_FIELDS:
            raise EvidenceError("loop journal event schema changed")
        event_type = event.get("event")
        if event_type not in _LOOP_EVENT_TYPES:
            raise EvidenceError("loop journal event type changed")
        timestamp = _parse_timestamp(event.get("timestamp"), name="loop event.timestamp")
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise EvidenceError("loop journal timestamps move backwards")
        previous_timestamp = timestamp
        for key, item in event.items():
            if isinstance(item, (dict, list)):
                raise EvidenceError("loop journal contains a nonscalar field")
            if isinstance(item, float) and not math.isfinite(item):
                raise EvidenceError("loop journal contains a non-finite scalar")
        case_id = event.get("case_id")
        if case_id is not None:
            if case_id not in cases or not _loop_case_dimensions_match(event, cases[case_id]):
                raise EvidenceError("loop journal case does not bind the frozen plan")
            if case_id in terminal_cases:
                raise EvidenceError("loop journal continues after a terminal case event")
        profile_id = event.get("profile_id")
        if profile_id is not None and profile_id not in models:
            raise EvidenceError("loop journal profile is not frozen in the plan")
        if case_id is not None and profile_id is not None:
            case = cases[case_id]
            valid_profiles = (
                {plan["protocol"]["rlm"]["model_profile"]}
                if case["phase"] == "rlm"
                else set(plan["protocol"]["halo"]["model_profiles"])
            )
            if profile_id not in valid_profiles:
                raise EvidenceError("loop journal profile does not match the case phase")
        if event_type == "case_started":
            attempt = event.get("attempt")
            if type(attempt) is not int or not 1 <= attempt <= 2 or profile_id is None:
                raise EvidenceError("loop case start attempt is invalid")
            start = (str(case_id), attempt, str(profile_id))
            if start in starts:
                raise EvidenceError("loop journal contains a duplicate case start")
            starts.add(start)
        elif event_type == "case_complete":
            measurement = _project_loop_measurement(event, source=True)
            identity = (
                str(measurement["case_id"]),
                int(measurement["attempt"]),
                str(measurement["profile_id"]),
            )
            if (
                type(measurement["attempt"]) is not int
                or not 1 <= measurement["attempt"] <= 2
                or identity not in starts
                or measurement["case_id"] in completed
            ):
                raise EvidenceError("loop completion does not bind one unique start")
            completed.add(str(measurement["case_id"]))
            terminal_cases.add(str(measurement["case_id"]))
            measurements.append(
                {"sample_index": len(measurements) + 1, **measurement}
            )
        elif event_type in {
            "case_exhausted",
            "case_skipped_campaign_stop",
            "case_skipped_deadline",
            "case_skipped_held",
        }:
            if not isinstance(case_id, str):
                raise EvidenceError("loop case has duplicate terminal events")
            terminal_cases.add(case_id)
    terminal_outcomes = {
        "case_complete": "complete",
        "case_exhausted": "exhausted",
        "case_skipped_campaign_stop": "skipped_campaign_stop",
        "case_skipped_deadline": "skipped_deadline",
        "case_skipped_held": "held",
    }
    starts_by_case = Counter(
        str(event["case_id"])
        for event in events
        if event.get("event") == "case_started"
    )
    failures_by_case = Counter(
        str(event["case_id"])
        for event in events
        if event.get("event") == "case_failed"
    )
    timeouts_by_case = Counter(
        str(event["case_id"])
        for event in events
        if event.get("event") == "case_timeout"
    )
    outcome_by_case = {
        str(event["case_id"]): terminal_outcomes[str(event["event"])]
        for event in events
        if event.get("event") in terminal_outcomes
    }
    outcomes: list[dict[str, Any]] = []
    for case in plan["cases"]:
        case_id = str(case["case_id"])
        attempts = starts_by_case[case_id]
        failures = failures_by_case[case_id]
        timeouts = timeouts_by_case[case_id]
        if attempts > 2 or failures + timeouts > attempts:
            raise EvidenceError("loop case attempt accounting changed")
        outcome = outcome_by_case.get(
            case_id, "incomplete" if attempts else "not_started"
        )
        outcomes.append(
            {
                "attempt_count": attempts,
                "case_id": case_id,
                "failed_attempt_count": failures,
                "outcome": outcome,
                "timeout_attempt_count": timeouts,
            }
        )
    return measurements, outcomes


def _loop_mean(values: Sequence[Any]) -> float | None:
    numbers = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    return statistics.fmean(numbers) if numbers else None


def _loop_complete_sum(values: Sequence[dict[str, Any]], field: str) -> float | None:
    items = [value.get(field) for value in values]
    if not values or not all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        for item in items
    ):
        return None
    return sum(float(item) for item in items)


def _loop_summary_expected(
    *, plan: dict[str, Any], events: list[dict[str, Any]], measurements: list[dict[str, Any]]
) -> dict[str, Any]:
    completed_by_id = {row["case_id"]: row for row in measurements}
    terminal_types = {
        "case_complete",
        "case_exhausted",
        "case_skipped_campaign_stop",
        "case_skipped_deadline",
        "case_skipped_held",
    }
    terminal_ids = {
        event.get("case_id")
        for event in events
        if event.get("event") in terminal_types and event.get("case_id") is not None
    }
    exhausted = {
        event.get("case_id")
        for event in events
        if event.get("event") == "case_exhausted"
    }
    held = {
        event.get("case_id")
        for event in events
        if event.get("event") == "case_skipped_held"
    }
    deadline = {
        event.get("case_id")
        for event in events
        if event.get("event")
        in {"case_skipped_campaign_stop", "case_skipped_deadline"}
    }
    rlm_profile = plan["protocol"]["rlm"]["model_profile"]
    halo_profiles = {
        row["profile_id"] for row in measurements if row.get("phase") == "halo"
    }
    fallback_profiles = [
        event.get("profile_id")
        for event in events
        if event.get("event") == "halo_fallback_selected"
    ]
    if fallback_profiles:
        halo_profiles.add(fallback_profiles[-1])
    if not halo_profiles:
        halo_profiles.add(plan["protocol"]["halo"]["model_profiles"][0])
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
    groups: list[dict[str, Any]] = []
    for phase, treatment, profile_id, reasoning_control, reasoning_effort in sorted(
        group_keys,
        key=lambda item: tuple("" if value is None else str(value) for value in item),
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
        prompt = _loop_complete_sum(observations, "vllm_prompt_tokens")
        cached = _loop_complete_sum(observations, "vllm_cached_prompt_tokens")
        generation = _loop_complete_sum(observations, "vllm_generation_tokens")
        wall = _loop_complete_sum(observations, "wall_s")
        group: dict[str, Any] = {
            "cache_fraction": (
                cached / prompt
                if cached is not None and prompt is not None and prompt > 0
                else None
            ),
            "cached_prompt_tokens": cached,
            "completed_cases": len(observations),
            "effective_generation_tps": (
                generation / wall
                if generation is not None and wall is not None and wall > 0
                else None
            ),
            "generation_tokens": generation,
            "mean_wall_s": _loop_mean([row.get("wall_s") for row in observations]),
            "phase": phase,
            "planned_cases": len(planned),
            "profile_id": profile_id,
            "prompt_tokens": prompt,
            "reasoning_control": reasoning_control,
            "reasoning_effort": reasoning_effort,
            "treatment": treatment,
        }
        if phase == "rlm":
            correct = sum(int(row.get("correct") is True) for row in observations)
            group.update(
                {
                    "accuracy": correct / len(observations) if observations else None,
                    "correct_cases": correct,
                    "mean_reported_calls": _loop_mean(
                        [row.get("reported_calls") for row in observations]
                    ),
                    "mean_vllm_successful_requests": _loop_mean(
                        [row.get("vllm_successful_requests") for row in observations]
                    ),
                }
            )
        else:
            group.update(
                {
                    "json_valid_rate": _loop_mean(
                        [int(row.get("json_valid") is True) for row in observations]
                    ),
                    "mean_citation_precision": _loop_mean(
                        [row.get("citation_precision") for row in observations]
                    ),
                    "mean_count_accuracy": _loop_mean(
                        [row.get("mean_count_accuracy") for row in observations]
                    ),
                    "mean_family_f1": _loop_mean(
                        [row.get("family_f1") for row in observations]
                    ),
                }
            )
        groups.append(group)

    last_start = max(
        (
            index
            for index, event in enumerate(events)
            if event.get("event") in {"campaign_resumed", "campaign_started"}
        ),
        default=-1,
    )
    latest = events[last_start + 1 :]
    cleanup = next(
        (
            event.get("event")
            for event in reversed(latest)
            if event.get("event")
            in {"campaign_cleanup_failed", "campaign_cleanup_verified"}
        ),
        None,
    )
    completed_count = len(completed_by_id)
    if cleanup == "campaign_cleanup_failed":
        status = "cleanup_failed"
    elif completed_count == len(plan["cases"]) and cleanup == "campaign_cleanup_verified":
        status = "complete"
    elif completed_count == len(plan["cases"]):
        status = "measurements_complete_cleanup_pending"
    elif terminal_ids:
        status = "partial"
    else:
        status = "not_started"
    return {
        "completed_cases": completed_count,
        "deadline_skipped_cases": len(deadline),
        "exhausted_cases": len(exhausted),
        "failed_attempts": sum(
            event.get("event") in {"case_failed", "case_timeout"}
            for event in events
        ),
        "groups": groups,
        "held_cases": len(held),
        "planned_cases": len(plan["cases"]),
        "status": status,
    }


def _loop_values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
        )
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(
            _loop_values_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _loop_values_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return _json_strict_equal(actual, expected)


def _project_loop_source_summary(
    value: Any, *, plan: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError("loop summary must be an object")
    source_schema = value.get("schema_version")
    expected_schema = f"sparkbench-loop-campaign-summary-v{plan['plan_schema_version']}"
    required = {
        "campaign_id",
        "completed_cases",
        "deadline_skipped_cases",
        "exhausted_cases",
        "failed_attempts",
        "generated_at",
        "groups",
        "plan_fingerprint",
        "planned_cases",
        "schema_version",
        "status",
    }
    if plan["protocol"]["rlm_compaction_admission"] is not None:
        required.add("held_cases")
    if set(value) != required or source_schema != expected_schema:
        raise EvidenceError("loop summary schema changed")
    _parse_timestamp(value.get("generated_at"), name="loop summary.generated_at")
    if (
        value.get("campaign_id") != plan["campaign_id"]
        or value.get("plan_fingerprint") != plan["plan_fingerprint"]
    ):
        raise EvidenceError("loop summary does not bind its plan")
    normalized_groups: list[dict[str, Any]] = []
    groups = value.get("groups")
    if not isinstance(groups, list):
        raise EvidenceError("loop summary groups must be a list")
    for group in groups:
        if not isinstance(group, dict) or group.get("phase") not in {"rlm", "halo"}:
            raise EvidenceError("loop summary group is invalid")
        expected_fields = (
            _LOOP_SUMMARY_RLM_GROUP_FIELDS
            if group["phase"] == "rlm"
            else _LOOP_SUMMARY_HALO_GROUP_FIELDS
        )
        legacy_fields = expected_fields - {"reasoning_control", "reasoning_effort"}
        if set(group) not in {expected_fields, legacy_fields}:
            raise EvidenceError("loop summary group schema changed")
        normalized = dict(group)
        normalized.setdefault("reasoning_control", None)
        normalized.setdefault("reasoning_effort", None)
        for key in ("phase", "profile_id", "treatment"):
            _safe_id(normalized.get(key), name=f"loop summary group.{key}")
        for key in ("reasoning_control", "reasoning_effort"):
            _safe_id(
                normalized.get(key),
                name=f"loop summary group.{key}",
                nullable=True,
            )
        for key, item in normalized.items():
            if key in {
                "phase",
                "profile_id",
                "reasoning_control",
                "reasoning_effort",
                "treatment",
            }:
                continue
            number = _finite(item, name=f"loop summary group.{key}")
            if number is not None and number < 0:
                raise EvidenceError("loop summary group contains a negative metric")
        normalized_groups.append(normalized)
    normalized_source = {
        "completed_cases": value.get("completed_cases"),
        "deadline_skipped_cases": value.get("deadline_skipped_cases"),
        "exhausted_cases": value.get("exhausted_cases"),
        "failed_attempts": value.get("failed_attempts"),
        "groups": normalized_groups,
        "held_cases": value.get("held_cases", 0),
        "planned_cases": value.get("planned_cases"),
        "status": value.get("status"),
    }
    if not _loop_values_equal(normalized_source, expected):
        raise EvidenceError("loop summary aggregates disagree with its journal")
    return expected


def _project_loop_lifecycle(
    events: list[dict[str, Any]], *, plan: dict[str, Any]
) -> dict[str, Any]:
    names = [str(event["event"]) for event in events]
    timestamps = [
        _parse_timestamp(event["timestamp"], name="loop lifecycle timestamp")
        for event in events
    ]
    elapsed: float | None = None
    if len(timestamps) >= 2:
        elapsed = (timestamps[-1] - timestamps[0]).total_seconds()
        if elapsed < 0:
            raise EvidenceError("loop journal timestamps move backwards")
    latest_start = max(
        (
            index
            for index, name in enumerate(names)
            if name in {"campaign_resumed", "campaign_started"}
        ),
        default=-1,
    )
    latest = names[latest_start + 1 :]
    cleanup = next(
        (
            name
            for name in reversed(latest)
            if name in {"campaign_cleanup_failed", "campaign_cleanup_verified"}
        ),
        None,
    )
    selected_halo_profiles = [
        event.get("profile_id")
        for event in events
        if event.get("event") == "case_complete" and event.get("phase") == "halo"
    ]
    selected_halo_profiles.extend(
        event.get("profile_id")
        for event in events
        if event.get("event") == "halo_fallback_selected"
    )
    selected_halo_profile = (
        selected_halo_profiles[-1]
        if selected_halo_profiles
        else plan["protocol"]["halo"]["model_profiles"][0]
    )
    return {
        "cleanup_verified": cleanup == "campaign_cleanup_verified",
        "event_count": len(names),
        "event_counts": dict(sorted(Counter(names).items())),
        "journal_elapsed_s": elapsed,
        "selected_halo_profile": selected_halo_profile,
        "terminal": "campaign_finished" in names,
        "terminal_event": "campaign_finished" if "campaign_finished" in names else None,
    }


def _project_loop_telemetry(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected_fields = {
        "cached_kib",
        "gpu_timestamp",
        "gpu_util_pct",
        "memavailable_kib",
        "memfree_kib",
        "memory_util_pct",
        "phase",
        "power_w",
        "sm_clock_mhz",
        "swapfree_kib",
        "swaptotal_kib",
        "temperature_c",
        "timestamp",
    }
    if any(not isinstance(record, dict) or set(record) != expected_fields for record in records):
        raise EvidenceError("loop telemetry source schema changed")
    return _project_telemetry(records)


def _export_loop_campaign(
    run_dir: Path,
    results_root: Path,
    output_root: Path,
    *,
    source_group: str,
) -> dict[str, Any]:
    run_id = run_dir.name
    _date_from_run_id(run_id)
    direct_entries = {path.name for path in run_dir.iterdir()}
    lifecycle_entries = {
        "journal.jsonl",
        "plan.json",
        "private",
        "server",
        "summary.json",
        "telemetry.jsonl",
    }
    if direct_entries not in ({"plan.json"}, lifecycle_entries):
        raise EvidenceError("loop source directory layout changed")
    for name in direct_entries:
        path = run_dir / name
        metadata = path.lstat()
        if name in {"private", "server"}:
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise EvidenceError("loop raw source directory is unsafe")
        elif not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise EvidenceError("loop public source file is unsafe")
    plan = _project_loop_plan(
        _load_json(run_dir / "plan.json", results_root),
        run_id=run_id,
        source_group=source_group,
    )
    lifecycle_names = {"journal.jsonl", "summary.json", "telemetry.jsonl"}
    present_lifecycle = direct_entries & lifecycle_names
    if present_lifecycle and present_lifecycle != lifecycle_names:
        raise EvidenceError("loop source lifecycle files are incomplete")
    events: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    if present_lifecycle:
        events = _load_json_lines(run_dir / "journal.jsonl", results_root)
        measurements, outcomes = _validate_loop_journal(events, plan=plan)
        expected_summary = _loop_summary_expected(
            plan=plan, events=events, measurements=measurements
        )
        summary = _project_loop_source_summary(
            _load_json(run_dir / "summary.json", results_root),
            plan=plan,
            expected=expected_summary,
        )
        telemetry = _project_loop_telemetry(
            _load_json_lines(run_dir / "telemetry.jsonl", results_root)
        )
        status = summary["status"]
    else:
        outcomes = [
            {
                "attempt_count": 0,
                "case_id": case["case_id"],
                "failed_attempt_count": 0,
                "outcome": "not_started",
                "timeout_attempt_count": 0,
            }
            for case in plan["cases"]
        ]
        summary = {
            "completed_cases": 0,
            "deadline_skipped_cases": 0,
            "exhausted_cases": 0,
            "failed_attempts": 0,
            "groups": [],
            "held_cases": 0,
            "planned_cases": len(plan["cases"]),
            "status": "planned",
        }
        status = "planned"
    lifecycle = _project_loop_lifecycle(events, plan=plan)
    manifest = {
        **plan,
        "evidence_kind": LOOP_EVIDENCE_KIND,
        "lifecycle": lifecycle,
        "run_date_utc": _date_from_run_id(run_id),
        "sanitization": {
            "free_form_text_included": False,
            "payloads_included": False,
            "policy": SANITIZATION_POLICY,
            "raw_identifiers_included": False,
        },
        "schema_version": SCHEMA_VERSION,
        "source_run_id": run_id,
        "status": status,
    }
    telemetry_files = _telemetry_files(telemetry)
    relative = Path("campaigns") / run_id
    bundle_hash, _ = _write_bundle(
        output_root,
        relative,
        {
            "manifest.json": manifest,
            "measurements.json": {
                "measurement_count": len(measurements),
                "measurements": measurements,
                "schema_version": SCHEMA_VERSION,
            },
            "outcomes.json": {
                "outcome_count": len(outcomes),
                "outcomes": outcomes,
                "schema_version": SCHEMA_VERSION,
            },
            "summary.json": {
                "aggregates": summary,
                "schema_version": SCHEMA_VERSION,
            },
            **telemetry_files,
        },
    )
    return {
        "bundle_sha256": bundle_hash,
        "campaign_id": run_id,
        "evidence_kind": LOOP_EVIDENCE_KIND,
        "file": str(relative / "manifest.json"),
        "status": status,
    }


_LLAMA_BENCH_FIELDS = {
    "avg_ns",
    "avg_ts",
    "backends",
    "build_commit",
    "build_number",
    "cpu_info",
    "cpu_mask",
    "cpu_strict",
    "devices",
    "embeddings",
    "fit_min_ctx",
    "fit_target",
    "flash_attn",
    "gpu_info",
    "load_mode",
    "main_gpu",
    "model_filename",
    "model_n_params",
    "model_size",
    "model_type",
    "n_batch",
    "n_cpu_moe",
    "n_depth",
    "n_gen",
    "n_gpu_layers",
    "n_prompt",
    "n_threads",
    "n_ubatch",
    "no_host",
    "no_kv_offload",
    "no_op_offload",
    "poll",
    "samples_ns",
    "samples_ts",
    "split_mode",
    "stddev_ns",
    "stddev_ts",
    "tensor_buft_overrides",
    "tensor_split",
    "test_time",
    "type_k",
    "type_v",
}

_LLAMA_BENCH_ARTIFACTS = {
    "Qwen3.8-27B-UD-Q4_K_XL.gguf": {
        "role": "qwen38-27b-ud-q4-k-xl",
        "revision": "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
        "sha256": "bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372",
        "size_bytes": 17_923_394_624,
        "source": "unsloth/Qwen3.8-27B-GGUF",
        "target": "Qwen3.8-27B-UD-Q4_K_XL.gguf",
    },
    "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf": {
        "role": "qwen36-35b-a3b-ud-q4-k-xl",
        "revision": "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
        "sha256": "55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1",
        "size_bytes": 22_853_663_008,
        "source": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        "target": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    },
    "Muse-Glimmer-30B-UD-Q4_K_XL.gguf": {
        "role": "muse-glimmer-30b-ud-q4-k-xl",
        "revision": "faa5b025c584459c13febfa5c59883516710ae39",
        "sha256": "82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38",
        "size_bytes": 15_878_222_368,
        "source": "unsloth/Muse-Glimmer-30B-GGUF",
        "target": "Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
    },
    "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf": {
        "role": "nemotron35-lightning-30b-a3b-q4-0",
        "revision": "9d425fe18d84ab04da6aabb757d2e2807083d054",
        "sha256": "61f87e75974e4b535dcdf9aad056541a9514f1dfa4538b463b081d19b7a00e3c",
        "size_bytes": 18_898_091_584,
        "source": "ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
        "target": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf",
    },
    "sha256-78b329e716e7e9775973d392cd132b1f1ff1c8287a992887caeb6fd6c56ba9cc": {
        "role": "ollama-qwen3-30b-a3b-q4-k-m",
        "revision": "19e422b02313",
        "sha256": "78b329e716e7e9775973d392cd132b1f1ff1c8287a992887caeb6fd6c56ba9cc",
        "size_bytes": 18_556_685_856,
        "source": "qwen3:30b-a3b-instruct-2507-q4_K_M",
        "target": "ollama-model-blob",
    },
    "sha256-9e0c827cfd6a6d000032be3da3d0914668b0c1112977e927186d29c4487466c4": {
        "role": "ollama-nemotron-cascade-2-q4-k-m",
        "revision": "e0705e3fe8f7",
        "sha256": "9e0c827cfd6a6d000032be3da3d0914668b0c1112977e927186d29c4487466c4",
        "size_bytes": 24_272_433_056,
        "source": "nemotron-cascade-2:latest",
        "target": "ollama-model-blob",
    },
}


def _project_llama_bench_row(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError("llama-bench row must be an object")
    if set(value) != _LLAMA_BENCH_FIELDS:
        raise EvidenceError("llama-bench source schema changed")
    filename = value.get("model_filename")
    if not isinstance(filename, str):
        raise EvidenceError("llama-bench model filename is missing")
    artifact = _LLAMA_BENCH_ARTIFACTS.get(Path(filename).name)
    if artifact is None:
        raise EvidenceError("unknown llama-bench artifact")
    numeric = {
        "avg_ns",
        "avg_ts",
        "build_number",
        "embeddings",
        "fit_min_ctx",
        "fit_target",
        "flash_attn",
        "main_gpu",
        "model_n_params",
        "model_size",
        "n_batch",
        "n_cpu_moe",
        "n_depth",
        "n_gen",
        "n_gpu_layers",
        "n_prompt",
        "n_threads",
        "n_ubatch",
        "no_host",
        "no_kv_offload",
        "no_op_offload",
        "poll",
        "stddev_ns",
        "stddev_ts",
    }
    strings = {
        "backends",
        "build_commit",
        "load_mode",
        "model_type",
        "split_mode",
        "type_k",
        "type_v",
    }
    result: dict[str, Any] = {"artifact_role": artifact["role"]}
    for key in numeric:
        if key in value:
            raw = value[key]
            result[key] = raw if isinstance(raw, bool) else _finite(raw, name=f"llama_bench.{key}")
    for key in strings:
        if key in value:
            if key == "build_commit":
                result[key] = _revision(value[key], name="llama_bench.build_commit")
            else:
                result[key] = _safe_text(value[key], name=f"llama_bench.{key}")
    for source, target in (("samples_ns", "samples_ns"), ("samples_ts", "samples_tps")):
        raw = value.get(source)
        if raw is not None:
            if not isinstance(raw, list):
                raise EvidenceError(f"llama-bench {source} must be a list")
            result[target] = [_finite(item, name=f"llama_bench.{source}") for item in raw]
    return result


_NINFER_TOP_FIELDS = {
    "artifact",
    "artifact_type",
    "command",
    "config",
    "environment",
    "load",
    "memory",
    "schema_version",
    "tests",
    "tool",
}
_NINFER_ARTIFACT_FIELDS = {"file_size_bytes", "path"}
_NINFER_ENVIRONMENT_FIELDS = {
    "cuda_driver_version",
    "cuda_runtime_version",
    "device_id",
    "gpu_name",
}
_NINFER_LOAD_FIELDS = {
    "artifact_bytes_read",
    "host_to_device_bytes",
    "load_seconds",
    "peak_staging_bytes",
    "resource_count",
    "target",
    "tensor_count",
    "upload_seconds",
    "weights_id",
}
_NINFER_CONFIG_FIELDS = {
    "corpus_path",
    "corpus_tokens",
    "decode_graph_prime",
    "decode_path",
    "kv_cache",
    "max_context",
    "mtp_draft_tokens",
    "prefill_chunk",
    "proposal_head",
    "repetitions",
    "use_cuda_graph",
    "warmup",
}
_NINFER_MEMORY_FIELDS = {
    "available_after_startup_bytes",
    "available_after_weights_bytes",
    "cuda_graph_allowance_bytes",
    "cuda_graph_observed_bytes",
    "device",
    "kv_cache",
    "kv_capacity",
    "kv_capacity_headroom_bytes",
    "kv_capacity_increment_bytes",
    "kv_capacity_max_page_groups",
    "kv_capacity_mode",
    "kv_capacity_page_groups",
    "kv_payload_bytes",
    "max_context",
    "minimum_runtime_reservation_bytes",
    "planned_slack_bytes",
    "request_transient",
    "runtime_reservation_bytes",
    "sequence",
    "weights",
    "workspace",
}
_NINFER_TEST_FIELDS = {
    "decode_engine_tok_s_mean",
    "decode_engine_tok_s_stddev",
    "decode_output_tok_s_mean",
    "decode_output_tok_s_stddev",
    "decode_seconds_mean",
    "decode_seconds_stddev",
    "kind",
    "label",
    "n_gen",
    "n_prompt",
    "prefill_seconds_mean",
    "prefill_seconds_stddev",
    "prefill_tok_s_mean",
    "prefill_tok_s_stddev",
    "prepare_seconds_mean",
    "prepare_seconds_stddev",
    "reps",
    "requested_output_tokens",
    "speculative",
    "total_seconds_mean",
    "total_seconds_stddev",
    "workspace_allocator_peak_bytes",
    "workspace_peak_bytes",
}
_NINFER_REP_FIELDS = {
    "decode_engine_tokens",
    "decode_output_tokens",
    "generated_output_tokens",
    "speculative",
    "timings",
}
_NINFER_TIMING_FIELDS = {
    "decode_seconds",
    "prefill_seconds",
    "prepare_seconds",
    "total_seconds",
    "vision_seconds",
}
_NINFER_SPECULATIVE_FIELDS = {
    "acceptance_length",
    "acceptance_rate",
    "accepted_per_position",
    "accepted_tokens",
    "draft_window",
    "drafted_tokens",
    "enabled",
    "fallback_steps",
    "rounds",
}


def _project_ninfer_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError("NInfer report must be an object")
    if (
        set(value) != _NINFER_TOP_FIELDS
        or value.get("schema_version") != 11
        or value.get("artifact_type") != "ninfer_bench_report"
        or value.get("tool") != "ninfer_bench"
    ):
        raise EvidenceError("unrecognized NInfer report")
    artifact = value.get("artifact")
    if not isinstance(artifact, dict) or set(artifact) != _NINFER_ARTIFACT_FIELDS:
        raise EvidenceError("NInfer artifact block changed")
    hardware = value.get("environment")
    if not isinstance(hardware, dict) or set(hardware) != _NINFER_ENVIRONMENT_FIELDS:
        raise EvidenceError("NInfer hardware block is missing")
    report: dict[str, Any] = {
        "artifact": _project_numeric_tree(
            {"file_size_bytes": artifact.get("file_size_bytes")},
            name="ninfer.artifact",
        ),
        "artifact_type": "ninfer_bench_report",
        "configuration": {},
        "hardware": {},
        "load": {},
        "memory": {},
        "schema_version": 11,
        "tests": [],
        "tool": "ninfer_bench",
    }
    for key in ("gpu_name", "cuda_runtime_version", "cuda_driver_version"):
        report["hardware"][key] = _safe_text(hardware.get(key), name=f"ninfer.hardware.{key}")
    if hardware.get("device_id") is not None:
        report["hardware"]["device_id"] = _finite(
            hardware["device_id"], name="ninfer.hardware.device_id"
        )
    load = value.get("load")
    if not isinstance(load, dict) or set(load) != _NINFER_LOAD_FIELDS:
        raise EvidenceError("NInfer load block is missing")
    for key, item in load.items():
        if key in {"target", "weights_id"}:
            report["load"][key] = _safe_id(item, name=f"ninfer.load.{key}")
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            report["load"][key] = _finite(item, name=f"ninfer.load.{key}")
    config = value.get("config")
    if not isinstance(config, dict) or set(config) != _NINFER_CONFIG_FIELDS:
        raise EvidenceError("NInfer config block is missing")
    for key, item in config.items():
        if key == "corpus_path":
            continue
        if key in {"decode_path", "kv_cache", "proposal_head"}:
            target = "decode_mode" if key == "decode_path" else key
            report["configuration"][target] = _safe_id(
                item, name=f"ninfer.configuration.{key}"
            )
        else:
            report["configuration"][key] = _project_numeric_tree(
                item, name=f"ninfer.configuration.{key}"
            )
    memory = value.get("memory")
    if not isinstance(memory, dict) or set(memory) != _NINFER_MEMORY_FIELDS:
        raise EvidenceError("NInfer memory block is missing")
    for key, item in memory.items():
        if key in {"kv_cache", "kv_capacity_mode"}:
            report["memory"][key] = _safe_id(item, name=f"ninfer.memory.{key}")
        else:
            report["memory"][key] = _project_numeric_tree(
                item, name=f"ninfer.memory.{key}"
            )
    tests = value.get("tests")
    if not isinstance(tests, list):
        raise EvidenceError("NInfer tests must be a list")
    for test in tests:
        if not isinstance(test, dict) or set(test) != _NINFER_TEST_FIELDS:
            raise EvidenceError("NInfer test must be an object")
        speculative = test.get("speculative")
        if (
            not isinstance(speculative, dict)
            or set(speculative) != _NINFER_SPECULATIVE_FIELDS
        ):
            raise EvidenceError("NInfer test speculative fields changed")
        repetitions = test.get("reps")
        if not isinstance(repetitions, list):
            raise EvidenceError("NInfer repetitions must be a list")
        for repetition in repetitions:
            if not isinstance(repetition, dict) or set(repetition) != _NINFER_REP_FIELDS:
                raise EvidenceError("NInfer repetition fields changed")
            timings = repetition.get("timings")
            repetition_speculative = repetition.get("speculative")
            if not isinstance(timings, dict) or set(timings) != _NINFER_TIMING_FIELDS:
                raise EvidenceError("NInfer timing fields changed")
            if (
                not isinstance(repetition_speculative, dict)
                or set(repetition_speculative) != _NINFER_SPECULATIVE_FIELDS
            ):
                raise EvidenceError("NInfer repetition speculative fields changed")
        projected: dict[str, Any] = {}
        for key, item in test.items():
            if key in {"kind", "label"}:
                projected[key] = _safe_id(item, name=f"ninfer.test.{key}")
            else:
                projected[key] = _project_numeric_tree(item, name=f"ninfer.test.{key}")
        report["tests"].append(projected)
    return report


def _project_artifact_inspect(value: Any) -> dict[str, Any]:
    expected = {
        "encodings",
        "file_bytes",
        "formats",
        "layouts",
        "model_id",
        "objects",
        "path",
        "payload_offset",
        "resources",
        "tensors",
        "weights_id",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise EvidenceError("artifact inspection must be an object")
    result: dict[str, Any] = {}
    for key in ("model_id", "weights_id"):
        result[key] = _safe_id(value.get(key), name=f"artifact_inspect.{key}")
    for key in ("file_bytes", "payload_offset", "objects", "tensors", "resources"):
        result[key] = _finite(value.get(key), name=f"artifact_inspect.{key}")
    for key in ("formats", "layouts", "encodings"):
        result[key] = _project_numeric_tree(value.get(key), name=f"artifact_inspect.{key}")
    path = value.get("path")
    if isinstance(path, str) and _HEX_RE.fullmatch(Path(path).name):
        result["artifact_sha256"] = Path(path).name
    return result


def _project_image_inspect(value: Any) -> dict[str, Any]:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise EvidenceError("image inspection must contain one object")
    item = value[0]
    result: dict[str, Any] = {}
    if item.get("Id") is not None:
        result["image_sha256"] = _sha256(item["Id"], name="image.id")
    for source, target in (("Architecture", "architecture"), ("Os", "os")):
        if item.get(source) is not None:
            result[target] = _safe_id(item[source], name=f"image.{target}")
    if item.get("Size") is not None:
        result["size_bytes"] = _finite(item["Size"], name="image.size")
    config = item.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    if isinstance(labels, dict):
        for source, target in (
            ("org.opencontainers.image.revision", "source_revision"),
            ("org.opencontainers.image.version", "base_version"),
            ("io.sparkbench.port", "port_label"),
        ):
            if labels.get(source) is not None:
                raw = labels[source]
                result[target] = (
                    _revision(raw, name=f"image.{target}")
                    if target == "source_revision"
                    else _safe_id(raw, name=f"image.{target}")
                )
    return result


def _project_ninfer_telemetry(path: Path, results_root: Path) -> list[dict[str, Any]]:
    text = _secure_read(path, results_root, maximum=MAX_SOURCE_JSON_BYTES)
    samples: list[dict[str, Any]] = []
    start: datetime | None = None
    for line_number, row in enumerate(csv.reader(text.splitlines()), 1):
        if len(row) != 5:
            raise EvidenceError(f"invalid NInfer telemetry row {path.name}:{line_number}")
        try:
            timestamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            power, temperature, utilization, clock = (float(value.strip()) for value in row[1:])
        except ValueError as error:
            raise EvidenceError(f"invalid NInfer telemetry row {path.name}:{line_number}") from error
        if start is None:
            start = timestamp
        elapsed = (timestamp - start).total_seconds()
        if elapsed < 0:
            raise EvidenceError("NInfer telemetry timestamps move backwards")
        samples.append(
            {
                "elapsed_s": elapsed,
                "gpu_util_pct": utilization,
                "power_w": power,
                "sample_index": line_number,
                "sm_clock_mhz": clock,
                "temperature_c": temperature,
            }
        )
    return samples


def _ninfer_parity(root: Path, results_root: Path) -> dict[str, Any] | None:
    stdout_zero = root / "parity-mtp0.stdout"
    stdout_spec = root / "parity-mtp3.stdout"
    stderr_zero = root / "parity-mtp0.stderr"
    stderr_spec = root / "parity-mtp3.stderr"
    if not all(path.is_file() for path in (stdout_zero, stdout_spec, stderr_zero, stderr_spec)):
        return None
    zero_bytes = _secure_read(stdout_zero, results_root, maximum=MAX_SOURCE_JSON_BYTES).encode()
    spec_bytes = _secure_read(stdout_spec, results_root, maximum=MAX_SOURCE_JSON_BYTES).encode()
    token_pattern = re.compile(r"^tokens\s+generated ids\s+([0-9 ]+)$", re.MULTILINE)
    zero_stderr = _secure_read(stderr_zero, results_root, maximum=MAX_SOURCE_JSON_BYTES)
    spec_stderr = _secure_read(stderr_spec, results_root, maximum=MAX_SOURCE_JSON_BYTES)
    zero_match = token_pattern.search(zero_stderr)
    spec_match = token_pattern.search(spec_stderr)
    if not zero_match or not spec_match:
        raise EvidenceError("NInfer parity token line is missing")
    zero_ids = tuple(int(value) for value in zero_match.group(1).split())
    spec_ids = tuple(int(value) for value in spec_match.group(1).split())
    return {
        "output_bytes": len(zero_bytes),
        "output_bytes_equal": zero_bytes == spec_bytes,
        "token_count": len(zero_ids),
        "token_ids_equal": zero_ids == spec_ids,
    }


def _export_campaign(
    campaign: Path,
    results_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    campaign_id = _safe_id(campaign.name, name="campaign.id")
    expected_files = {
        "moe-bandwidth-20260817T1539Z": {
            "cascade.json",
            "cascade.log",
            "control-post.json",
            "control-post.log",
            "control-pre.json",
            "control-pre.log",
            "muse-control-post.json",
            "muse-control-post.log",
            "muse-control-pre.json",
            "muse-control-pre.log",
            "muse-glimmer.json",
            "muse-glimmer.log",
            "nemotron35.json",
            "nemotron35.log",
            "panel-part2.json",
            "panel-part2.log",
            "panel.json",
            "panel.log",
        },
        "ninfer-experimental-sm121a-20260817T181134Z": {
            "admission-mtp0.log",
            "artifact-inspect.json",
            "build.log",
            "experimental-sm121a-admission-mtp0.json",
            "experimental-sm121a-native-eager-mtp0.json",
            "experimental-sm121a-native-eager-mtp3.json",
            "image-inspect.json",
            "native-eager-mtp0-telemetry.csv",
            "native-eager-mtp0.log",
            "native-eager-mtp3-telemetry.csv",
            "native-eager-mtp3.log",
            "native-mtp0-telemetry.csv",
            "native-mtp0.log",
            "packages.tsv",
            "parity-mtp0.stderr",
            "parity-mtp0.stdout",
            "parity-mtp3.stderr",
            "parity-mtp3.stdout",
        },
        "ninfer-gb10-20260817": {
            "stock-cmake-default.log",
            "stock-cmake-sm121.log",
        },
        "ninfer-qwen38-nvfp4-sm121a-20260817T200147Z": {
            "SHA256SUMS",
            "admission-mtp0.json",
            "admission-mtp0.log",
            "admission-mtp3.json",
            "admission-mtp3.log",
            "artifact-inspect.json",
            "image-inspect.json",
            "mtp0-telemetry.csv",
            "mtp0.json",
            "mtp0.log",
            "mtp3-telemetry.csv",
            "mtp3.json",
            "mtp3.log",
            "parity-mtp0.stderr",
            "parity-mtp0.stdout",
            "parity-mtp3.stderr",
            "parity-mtp3.stdout",
        },
    }
    expected = expected_files.get(campaign.name)
    if expected is None:
        raise EvidenceError(f"unsupported custom campaign {campaign.name!r}")
    actual = {path.name for path in campaign.iterdir()}
    if actual != expected or any(not path.is_file() for path in campaign.iterdir()):
        raise EvidenceError(f"custom campaign file set changed: {campaign.name}")
    relative = Path("campaigns") / campaign_id
    manifest: dict[str, Any] = {
        "campaign_id": campaign_id,
        "payloads_included": False,
        "sanitization_policy": SANITIZATION_POLICY,
        "schema_version": SCHEMA_VERSION,
    }
    measurements: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    status = "complete"
    if campaign.name == "moe-bandwidth-20260817T1539Z":
        manifest["evidence_kind"] = "llama_bench"
        used_artifacts: dict[str, dict[str, Any]] = {}
        for path in sorted(campaign.glob("*.json")):
            values, truncated = _load_complete_array_objects(path, results_root)
            for ordinal, value in enumerate(values, 1):
                if not isinstance(value, dict) or not isinstance(
                    value.get("model_filename"), str
                ):
                    raise EvidenceError("llama-bench artifact identity is missing")
                artifact = _LLAMA_BENCH_ARTIFACTS.get(
                    Path(value["model_filename"]).name
                )
                if artifact is None:
                    raise EvidenceError("unknown llama-bench artifact")
                used_artifacts[str(artifact["role"])] = artifact
                measurements.append(
                    {
                        "capture_id": _safe_id(path.stem, name="capture.id"),
                        "capture_ordinal": ordinal,
                        "source_truncated": truncated,
                        **_project_llama_bench_row(value),
                    }
                )
            if truncated:
                status = "partial"
        if len(measurements) != 40 or len(used_artifacts) != 6:
            raise EvidenceError("llama-bench campaign measurement set changed")
        manifest["artifacts"] = [
            used_artifacts[role] for role in sorted(used_artifacts)
        ]
        manifest["runtime"] = {
            "binary_sha256": "cc16b06acc899a8fa4f1231c341abec5eb27b7f96a18a57ec75a8703e46ff3fc",
            "binary_size_bytes": 51_559_272,
            "source": "ggml-org/llama.cpp",
            "source_revision": "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70",
        }
    elif campaign.name in {
        "ninfer-experimental-sm121a-20260817T181134Z",
        "ninfer-qwen38-nvfp4-sm121a-20260817T200147Z",
    }:
        manifest["evidence_kind"] = "ninfer"
        for path in sorted(campaign.glob("*.json")):
            value = _load_json(path, results_root)
            if path.name == "artifact-inspect.json":
                projected = _project_artifact_inspect(value)
                record_type = "artifact_inspection"
            elif path.name == "image-inspect.json":
                projected = _project_image_inspect(value)
                record_type = "image_inspection"
            else:
                projected = _project_ninfer_report(value)
                record_type = "benchmark_report"
            measurements.append(
                {
                    "capture_id": _safe_id(path.stem, name="capture.id"),
                    "record_type": record_type,
                    "value": projected,
                }
            )
        for path in sorted(campaign.glob("*telemetry.csv")):
            telemetry.append(
                {
                    "capture_id": _safe_id(path.stem, name="telemetry.capture_id"),
                    "samples": _project_ninfer_telemetry(path, results_root),
                }
            )
        parity = _ninfer_parity(campaign, results_root)
        if parity:
            manifest["parity"] = parity
    elif campaign.name == "ninfer-gb10-20260817":
        manifest["evidence_kind"] = "ninfer_admission"
        status = "unsupported"
        codes: list[str] = []
        checks = {
            "stock-cmake-sm121.log": (
                "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got '121a'",
                "unsupported_compute_architecture",
            ),
            "stock-cmake-default.log": (
                "NInfer requires CUDA 13.1 or newer; found CUDA compiler 13.0.88",
                "cuda_toolkit_too_old",
            ),
        }
        for name, (needle, code) in checks.items():
            text = _secure_read(campaign / name, results_root, maximum=MAX_SOURCE_JSON_BYTES)
            if needle not in text:
                raise EvidenceError(f"unrecognized NInfer admission failure in {name}")
            codes.append(code)
        manifest["failure_codes"] = codes
    else:
        raise EvidenceError(f"unsupported custom campaign {campaign.name!r}")
    manifest["status"] = status
    bundle_hash, _ = _write_bundle(
        output_root,
        relative,
        {
            "manifest.json": manifest,
            "measurements.json": {
                "measurement_count": len(measurements),
                "measurements": measurements,
                "schema_version": SCHEMA_VERSION,
            },
            "telemetry.json": {
                "capture_count": len(telemetry),
                "captures": telemetry,
                "schema_version": SCHEMA_VERSION,
            },
        },
    )
    return {
        "bundle_sha256": bundle_hash,
        "campaign_id": campaign_id,
        "evidence_kind": manifest["evidence_kind"],
        "file": str(relative / "manifest.json"),
        "status": status,
    }


def _harbor_manifest(replicates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    first = replicates[0]
    return {
        "campaign_id": HARBOR_CAMPAIGN_ID,
        "derivation_digest": HARBOR_EXPECTED_DERIVATION_DIGEST,
        "evidence_kind": HARBOR_EVIDENCE_KIND,
        "git_revision": HARBOR_EXPECTED_GIT_REVISION,
        "model_profile": first["model"]["profile"],
        "payloads_included": False,
        "replicate_count": HARBOR_REPLICATE_COUNT,
        "replicate_order": "started_at_utc",
        "sanitization_policy": SANITIZATION_POLICY,
        "schema_version": HARBOR_SCHEMA_VERSION,
        "status": "complete",
    }


def _export_harbor_campaign(
    paths: Sequence[Path], output_root: Path
) -> dict[str, Any]:
    if len(paths) != HARBOR_REPLICATE_COUNT:
        raise EvidenceError("Harbor evidence requires exactly two result files")
    absolute_paths = tuple(Path(os.path.abspath(path)) for path in paths)
    if len(set(absolute_paths)) != HARBOR_REPLICATE_COUNT:
        raise EvidenceError("Harbor result inputs are duplicates")
    replicates = _order_harbor_replicates(
        tuple(_load_harbor_result(path) for path in absolute_paths)
    )
    relative = Path("campaigns") / HARBOR_CAMPAIGN_ID
    manifest = _harbor_manifest(replicates)
    bundle_hash, _ = _write_bundle(
        output_root,
        relative,
        {
            "manifest.json": manifest,
            "replicates.json": {
                "replicate_count": HARBOR_REPLICATE_COUNT,
                "replicates": list(replicates),
                "schema_version": HARBOR_SCHEMA_VERSION,
            },
        },
    )
    return {
        "bundle_sha256": bundle_hash,
        "campaign_id": HARBOR_CAMPAIGN_ID,
        "evidence_kind": HARBOR_EVIDENCE_KIND,
        "file": str(relative / "manifest.json"),
        "status": "complete",
    }


def _reuse_existing_harbor_campaign(
    evidence_root: Path, output_root: Path
) -> dict[str, Any]:
    relative = Path("campaigns") / HARBOR_CAMPAIGN_ID
    directory = evidence_root / relative
    bundle_hash = _verify_bundle(directory, evidence_root)
    manifest = _load_json(directory / "manifest.json", evidence_root)
    replicates = _load_json(directory / "replicates.json", evidence_root)
    entry = {
        "bundle_sha256": bundle_hash,
        "campaign_id": HARBOR_CAMPAIGN_ID,
        "evidence_kind": HARBOR_EVIDENCE_KIND,
        "file": str(relative / "manifest.json"),
        "status": "complete",
    }
    if not isinstance(manifest, dict):
        raise EvidenceError("Harbor evidence manifest changed")
    _verify_harbor_bundle(evidence_root, directory, entry, manifest)
    regenerated_hash, _ = _write_bundle(
        output_root,
        relative,
        {
            "manifest.json": manifest,
            "replicates.json": replicates,
        },
    )
    if regenerated_hash != bundle_hash:
        raise EvidenceError("Harbor evidence is not canonically reproducible")
    return entry


def _project_metric_mapping(
    value: Any,
    *,
    name: str,
    numeric: set[str],
    strings: set[str] | None = None,
    dropped: set[str] | None = None,
    exact: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{name} must be an object")
    strings = strings or set()
    dropped = dropped or set()
    unknown = set(value) - numeric - strings - dropped
    if unknown:
        raise EvidenceError(f"unknown {name} fields: {sorted(unknown)!r}")
    expected = numeric | strings | dropped
    if exact and set(value) != expected:
        raise EvidenceError(f"incomplete {name} fields")
    projected: dict[str, Any] = {}
    for key, item in sorted(value.items()):
        if key in dropped:
            continue
        if key in numeric:
            projected[key] = _finite(item, name=f"{name}.{key}")
        else:
            projected[key] = _safe_id(item, name=f"{name}.{key}")
    return projected


_BATTERY_SUMMARY_NUMERIC = {
    "aggregate_decode_tps",
    "aggregate_output_tps",
    "completion_tokens",
    "maximum_decode_tps",
    "median_decode_tps",
    "median_e2e_s",
    "median_ttft_s",
    "minimum_decode_tps",
    "prompt_tokens",
    "requests",
}
_BATTERY_SAMPLE_NUMERIC = {
    "completion_tokens",
    "decode_s",
    "decode_tps",
    "e2e_s",
    "emission_events",
    "measured_order",
    "output_tps",
    "prompt_tokens",
    "repetition",
    "ttft_s",
}


def _export_content_battery(path: Path, results_root: Path, output_root: Path) -> dict[str, Any]:
    value = _load_json(path, results_root)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise EvidenceError("unrecognized content battery")
    expected_top = {
        "battery",
        "endpoint",
        "model",
        "probes",
        "protocol",
        "schema_version",
        "summary",
        "warmup",
    }
    if set(value) != expected_top:
        raise EvidenceError("unknown content battery top-level fields")
    battery = value.get("battery")
    if not isinstance(battery, dict):
        raise EvidenceError("content battery identity is missing")
    if set(battery) != {"id", "prompt_set_version", "protocol_version"}:
        raise EvidenceError("unknown content battery identity fields")
    if battery.get("id") != "dgx-spark-qwen38-content" or value.get("model") != "qwen3.8-27b":
        raise EvidenceError("unexpected content battery identity")
    measurements: list[dict[str, Any]] = []
    probes = value.get("probes")
    if not isinstance(probes, list):
        raise EvidenceError("content battery probes are missing")
    expected_probe_ids = set(_UPSTREAM_PROBES.values())
    if len(probes) != len(expected_probe_ids) or {
        probe.get("id") for probe in probes if isinstance(probe, dict)
    } != expected_probe_ids:
        raise EvidenceError("content battery probe set changed")
    for probe in probes:
        if not isinstance(probe, dict):
            raise EvidenceError("content battery probe must be an object")
        if set(probe) != {"category", "id", "language", "samples", "summary"}:
            raise EvidenceError("unknown content battery probe fields")
        projected_probe = {
            "category": _safe_id(probe.get("category"), name="battery.category"),
            "id": _safe_id(probe.get("id"), name="battery.probe_id"),
            "language": _safe_id(probe.get("language"), name="battery.language"),
            "samples": [],
            "summary": _project_metric_mapping(
                probe.get("summary"),
                name="battery.probe.summary",
                numeric=_BATTERY_SUMMARY_NUMERIC,
                exact=True,
            ),
        }
        samples = probe.get("samples")
        if not isinstance(samples, list):
            raise EvidenceError("content battery samples are missing")
        if len(samples) != 3:
            raise EvidenceError("content battery must contain three measured samples per probe")
        for sample_index, sample in enumerate(samples, 1):
            if not isinstance(sample, dict):
                raise EvidenceError("content battery sample must be an object")
            projected_probe["samples"].append(
                {
                    "sample_index": sample_index,
                    **_project_metric_mapping(
                        sample,
                        name="battery.sample",
                        numeric=_BATTERY_SAMPLE_NUMERIC,
                        dropped={"sample_id"},
                        exact=True,
                    ),
                }
            )
        measurements.append(projected_probe)
    warmup = value.get("warmup")
    if not isinstance(warmup, dict):
        raise EvidenceError("content battery warmup is missing")
    result = {
        "battery": {
            "id": _safe_id(battery.get("id"), name="battery.id"),
            "prompt_set_version": _finite(
                battery.get("prompt_set_version"), name="battery.prompt_set_version"
            ),
            "protocol_version": _finite(
                battery.get("protocol_version"), name="battery.protocol_version"
            ),
        },
        "evidence_kind": "content_battery",
        "model_id": _safe_id(value.get("model"), name="battery.model"),
        "probes": measurements,
        "protocol": _project_metric_mapping(
            value.get("protocol"),
            name="battery.protocol",
            numeric={
                "max_output_tokens",
                "minimum_output_tokens",
                "repetitions_per_prompt",
                "temperature",
                "warmups",
            },
            strings={"aggregate_decode_tps", "aggregate_output_tps", "transport"},
            dropped={"fresh_prompt_tags"},
            exact=True,
        ),
        "schema_version": SCHEMA_VERSION,
        "summary": _project_metric_mapping(
            value.get("summary"),
            name="battery.summary",
            numeric=_BATTERY_SUMMARY_NUMERIC,
            exact=True,
        ),
        "warmup": _project_metric_mapping(
            warmup,
            name="battery.warmup",
            numeric=_BATTERY_SAMPLE_NUMERIC - {"measured_order", "repetition"},
            dropped={"id"},
            exact=True,
        ),
    }
    relative = Path("standalone") / "content-battery-dspark-sglang-20260817"
    bundle_hash, _ = _write_bundle(output_root, relative, {"summary.json": result})
    return {
        "bundle_sha256": bundle_hash,
        "evidence_kind": "content_battery",
        "file": str(relative / "summary.json"),
        "id": relative.name,
        "status": "complete",
    }


_UPSTREAM_PROBES = {
    "math (EN, eval-style)": "math-en-eval-style",
    "code (EN)": "code-en",
    "code (DE)": "code-de",
    "technical explain (FR)": "technical-explanation-fr",
    "reasoning (FR)": "reasoning-fr",
    "free prose (EN)": "free-prose-en",
    "free prose (FR)": "free-prose-fr",
    "free prose (DE)": "free-prose-de",
}


def _export_upstream_battery(path: Path, results_root: Path, output_root: Path) -> dict[str, Any]:
    value = _load_json(path, results_root)
    if not isinstance(value, dict) or value.get("battery_version") != 1:
        raise EvidenceError("unrecognized upstream battery")
    if set(value) != {"battery_version", "endpoint", "method", "model_id", "results"}:
        raise EvidenceError("unknown upstream battery fields")
    rows = value.get("results")
    if not isinstance(rows, list):
        raise EvidenceError("upstream battery rows are missing")
    if len(rows) != len(_UPSTREAM_PROBES) or {
        row.get("probe") for row in rows if isinstance(row, dict)
    } != set(_UPSTREAM_PROBES):
        raise EvidenceError("upstream battery probe set changed")
    measurements: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if (
            not isinstance(row, dict)
            or set(row) != {"probe", "tok_s"}
            or row.get("probe") not in _UPSTREAM_PROBES
        ):
            raise EvidenceError("unknown upstream battery probe")
        measurements.append(
            {
                "probe_id": _UPSTREAM_PROBES[row["probe"]],
                "probe_index": index,
                "tokens_per_s": _finite(row.get("tok_s"), name="upstream.tokens_per_s"),
            }
        )
    result = {
        "evidence_kind": "upstream_content_battery",
        "measurements": measurements,
        "model_id": _safe_id(value.get("model_id"), name="upstream.model_id"),
        "protocol_id": "two-call-delta-temperature0-output80-680",
        "schema_version": SCHEMA_VERSION,
    }
    relative = Path("standalone") / "upstream-bench-matrix-dspark-sglang-20260817"
    bundle_hash, _ = _write_bundle(output_root, relative, {"summary.json": result})
    return {
        "bundle_sha256": bundle_hash,
        "evidence_kind": "upstream_content_battery",
        "file": str(relative / "summary.json"),
        "id": relative.name,
        "status": "complete",
    }


def _export_matrix(path: Path, results_root: Path, output_root: Path) -> dict[str, Any]:
    value = _load_json(path, results_root)
    if not isinstance(value, dict):
        raise EvidenceError("matrix ledger must be an object")
    matrix_id = _safe_id(path.parent.name, name="matrix.id")
    models = value.get("models")
    if not isinstance(models, list):
        raise EvidenceError("matrix model list is missing")
    runs = value.get("runs")
    if not isinstance(runs, list):
        raise EvidenceError("matrix run list is missing")
    projected_runs: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            raise EvidenceError("matrix run must be an object")
        item: dict[str, Any] = {
            "model_id": _safe_id(run.get("model"), name="matrix.run.model"),
            "status": _safe_id(run.get("status"), name="matrix.run.status"),
        }
        if run.get("completed_cases") is not None:
            item["completed_cases"] = _finite(
                run["completed_cases"], name="matrix.run.completed_cases"
            )
        run_dir = run.get("run_dir")
        if isinstance(run_dir, str):
            item["run_id"] = _safe_id(Path(run_dir).name, name="matrix.run.run_id")
        if run.get("error_type") is not None:
            item["exception_type"] = _safe_id(
                run["error_type"], name="matrix.run.exception_type"
            )
        projected_runs.append(item)
    result = {
        "evidence_kind": "matrix",
        "matrix_id": matrix_id,
        "models": [_safe_id(model, name="matrix.model") for model in models],
        "run_count": len(projected_runs),
        "runs": projected_runs,
        "schema_version": SCHEMA_VERSION,
        "status_counts": dict(sorted(Counter(item["status"] for item in projected_runs).items())),
        "suite_id": _safe_id(value.get("suite"), name="matrix.suite"),
    }
    output = output_root / "matrices"
    output.mkdir(parents=True, exist_ok=True)
    data = _canonical(result)
    target = output / f"{matrix_id}.json"
    target.write_bytes(data)
    return {
        "file": str(Path("matrices") / target.name),
        "matrix_id": matrix_id,
        "sha256": _hash_bytes(data),
        "status": "complete",
    }


def _scan_string(value: str, *, pointer: str) -> None:
    normalized = unicodedata.normalize("NFKC", value)
    if len(normalized) > MAX_STRING_LENGTH or not normalized.isascii():
        raise EvidenceError(f"unsafe string at {pointer}")
    if any(ord(character) < 32 for character in normalized):
        raise EvidenceError(f"control character at {pointer}")
    if (
        normalized.startswith(("/", "~", "file://", "data:", "\\\\"))
        or re.match(r"[A-Za-z]:[\\/]", normalized)
        or "../" in normalized
        or "..\\" in normalized
    ):
        raise EvidenceError(f"path or payload value at {pointer}")
    for detector, pattern in _SECRET_PATTERNS:
        if pattern.search(normalized):
            raise EvidenceError(f"{detector} detector matched at {pointer}")


def _validate_output_value(value: Any, *, pointer: str = "") -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise EvidenceError(f"non-finite number at {pointer}")
        return
    if isinstance(value, str):
        _scan_string(value, pointer=pointer)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_output_value(item, pointer=f"{pointer}/{index}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvidenceError(f"non-string key at {pointer}")
            normalized = key.lower().replace("-", "_")
            if normalized in _FORBIDDEN_OUTPUT_KEYS or normalized.endswith("_path"):
                raise EvidenceError(f"forbidden key at {pointer}/{key}")
            _scan_string(key, pointer=f"{pointer}/<key>")
            _validate_output_value(item, pointer=f"{pointer}/{key}")
        return
    raise EvidenceError(f"unsupported output type at {pointer}")


def _expect_object_keys(value: Any, expected: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise EvidenceError(f"invalid {name} schema")
    return value


def _verify_bundle(directory: Path, root: Path) -> str:
    if not directory.is_dir() or directory.is_symlink():
        raise EvidenceError(f"missing or unsafe evidence bundle: {directory.name}")
    entries = list(directory.iterdir())
    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise EvidenceError(f"evidence bundle must contain only files: {directory.name}")
    checksums_path = directory / "checksums.json"
    checksums = _load_json(checksums_path, root)
    checksums = _expect_object_keys(
        checksums, {"files", "schema_version"}, name="bundle checksums"
    )
    if checksums["schema_version"] != SCHEMA_VERSION:
        raise EvidenceError("bundle checksum schema version changed")
    expected = checksums["files"]
    if not isinstance(expected, dict):
        raise EvidenceError("bundle checksum file map is invalid")
    actual_names = {entry.name for entry in entries}
    if actual_names != set(expected) | {"checksums.json"}:
        raise EvidenceError(f"bundle file set does not match checksums: {directory.name}")
    for name, digest in expected.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise EvidenceError("unsafe bundle checksum filename")
        expected_digest = _sha256(digest, name=f"bundle.{name}")
        data = _secure_read(
            directory / name, root, maximum=MAX_OUTPUT_FILE_BYTES
        ).encode()
        if _hash_bytes(data) != expected_digest:
            raise EvidenceError(f"bundle checksum mismatch: {directory.name}/{name}")
    return _hash_bytes(
        _secure_read(checksums_path, root, maximum=MAX_OUTPUT_FILE_BYTES).encode()
    )


def _verify_run_bundle(root: Path, entry: dict[str, Any]) -> None:
    _expect_object_keys(
        entry,
        {
            "bundle_sha256",
            "evidence_kind",
            "file",
            "matrix_id",
            "measurement_terminal",
            "run_id",
            "status",
        },
        name="run index entry",
    )
    run_id = _safe_id(entry["run_id"], name="run index ID")
    directory = root / "runs" / run_id
    bundle_sha256 = _sha256(entry["bundle_sha256"], name="run bundle")
    if _verify_bundle(directory, root) != bundle_sha256:
        raise EvidenceError(f"run bundle digest mismatch: {run_id}")
    expected_manifest = f"runs/{run_id}/manifest.json"
    if entry["file"] != expected_manifest:
        raise EvidenceError(f"run manifest pointer mismatch: {run_id}")
    manifest = _load_json(directory / "manifest.json", root)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError(f"run manifest schema mismatch: {run_id}")
    manifest_model = manifest.get("model")
    manifest_suite = manifest.get("suite")
    manifest_runtime = manifest.get("runtime")
    model_backend = (
        manifest_model.get("backend") if isinstance(manifest_model, dict) else None
    )
    runtime_backend = (
        manifest_runtime.get("backend")
        if isinstance(manifest_runtime, dict)
        else None
    )
    if (
        model_backend == "sglang" or runtime_backend == "sglang"
    ) and model_backend != runtime_backend:
        raise EvidenceError(f"SGLang runtime provenance backend changed: {run_id}")
    runtime_sglang_fields = (
        {key for key in manifest_runtime if key.startswith("sglang_")}
        if isinstance(manifest_runtime, dict)
        else set()
    )
    legacy_sglang = (
        model_backend == "sglang"
        and not runtime_sglang_fields
        and _LEGACY_SGLANG_PROVENANCE_BUNDLES.get(run_id) == bundle_sha256
    )
    projected_overlays: list[dict[str, str]] = []
    if model_backend == "sglang":
        if not isinstance(manifest_runtime, dict):
            raise EvidenceError(f"SGLang runtime provenance is missing: {run_id}")
        if legacy_sglang:
            projected_overlays = []
        else:
            projected_overlays = _validate_projected_sglang_provenance(
                manifest_runtime, manifest_model
            )
    elif runtime_sglang_fields:
        raise EvidenceError(f"SGLang runtime provenance backend changed: {run_id}")
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, list):
        if model_backend == "sglang":
            raise EvidenceError(f"SGLang overlay artifacts are missing: {run_id}")
        overlay_artifacts: list[Any] = []
    else:
        overlay_artifacts = [
            artifact
            for artifact in manifest_artifacts
            if isinstance(artifact, dict)
            and isinstance(artifact.get("role"), str)
            and artifact["role"].startswith("sglang_source_overlay_")
        ]
    expected_overlay_artifacts = [
        {
            "role": f"sglang_source_overlay_{index}",
            "sha256": artifact["sha256"],
            "target": artifact["target"],
        }
        for index, artifact in enumerate(projected_overlays, 1)
    ]
    if overlay_artifacts != expected_overlay_artifacts:
        raise EvidenceError(f"SGLang overlay artifact binding changed: {run_id}")
    is_prefix_cache_manifest = (
        isinstance(manifest_model, dict)
        and manifest_model.get("prefix_cache_mode") is not None
    ) or (
        isinstance(manifest_suite, dict)
        and manifest_suite.get("id") == PREFIX_CACHE_SUITE_ID
    )
    is_memory_manifest = (
        isinstance(manifest_suite, dict)
        and manifest_suite.get("id") == MEMORY_OPERATION_SUITE_ID
    )
    if is_prefix_cache_manifest:
        # Cache bundles are intentionally a complete, exact outer document.
        # Do not rely on checksums alone: an attacker can refresh checksums
        # after adding a generic manifest field such as trace text.
        _project_prefix_cache_manifest(manifest)
    if is_memory_manifest:
        _project_memory_manifest(manifest)
    for manifest_key, index_key in (
        ("source_run_id", "run_id"),
        ("evidence_kind", "evidence_kind"),
        ("status", "status"),
        ("matrix_id", "matrix_id"),
    ):
        if manifest.get(manifest_key) != entry.get(index_key):
            raise EvidenceError(f"run index mismatch for {run_id}: {manifest_key}")
    lifecycle = manifest.get("lifecycle")
    if not isinstance(lifecycle, dict):
        raise EvidenceError(f"run lifecycle is missing: {run_id}")
    if bool(entry["measurement_terminal"]) != (
        lifecycle.get("terminal_event") == "run_complete"
    ):
        raise EvidenceError(f"run terminal classification mismatch: {run_id}")
    suite = manifest.get("suite")
    agentic_suite: dict[str, Any] | None = None
    memory_suite: dict[str, Any] | None = None
    manifest_cases = suite.get("cases") if isinstance(suite, dict) else None
    if isinstance(suite, dict) and suite.get("id") == AUTORESEARCH_SUITE_ID:
        agentic_suite = _project_autoresearch_suite(suite)
        if agentic_suite != suite:
            raise EvidenceError(f"autoresearch suite projection mismatch: {run_id}")
    elif isinstance(suite, dict) and (
        suite.get("id") == "agentic-tools"
        or (
            isinstance(manifest_cases, list)
            and any(
                isinstance(case, dict) and case.get("kind") == "agentic"
                for case in manifest_cases
            )
        )
    ):
        agentic_suite = _project_agentic_suite(suite)
        if agentic_suite != suite:
            raise EvidenceError(f"agentic suite projection mismatch: {run_id}")
    if isinstance(suite, dict) and (
        suite.get("id") == MEMORY_OPERATION_SUITE_ID
        or (
            isinstance(manifest_cases, list)
            and any(
                isinstance(case, dict) and case.get("kind") == "memory"
                for case in manifest_cases
            )
        )
    ):
        memory_suite = _project_memory_suite(
            suite, binding_model=manifest.get("model")
        )
        if not _json_strict_equal(memory_suite, suite):
            raise EvidenceError(f"memory suite projection mismatch: {run_id}")
        validated_memory_model = _validate_projected_memory_model(manifest.get("model"))
        if not _json_strict_equal(validated_memory_model, manifest.get("model")):
            raise EvidenceError(f"memory model projection mismatch: {run_id}")

    samples = _load_json(directory / "samples.json", root)
    samples = _expect_object_keys(
        samples, {"sample_count", "samples", "schema_version"}, name="run samples"
    )
    if (
        samples["schema_version"] != SCHEMA_VERSION
        or not isinstance(samples["samples"], list)
        or samples["sample_count"] != len(samples["samples"])
    ):
        raise EvidenceError(f"run sample count mismatch: {run_id}")
    if is_prefix_cache_manifest and (
        type(samples["sample_count"]) is not int
        or samples["sample_count"] != len(PREFIX_CACHE_PREFIX_TARGETS) * 15
        or type(samples["schema_version"]) is not str
    ):
        raise EvidenceError("prefix-cache samples document does not match its exact schema")
    if memory_suite is not None and (
        type(samples["sample_count"]) is not int
        or samples["sample_count"]
        != len(MEMORY_OPERATION_SCENARIO_IDS) * MEMORY_OPERATION_VARIANT_COUNT
        or type(samples["schema_version"]) is not str
    ):
        raise EvidenceError("memory samples document does not match its exact schema")
    summary = _load_json(directory / "summary.json", root)
    summary = _expect_object_keys(
        summary, {"aggregates", "schema_version"}, name="run summary"
    )
    if summary["schema_version"] != SCHEMA_VERSION:
        raise EvidenceError(f"run summary schema mismatch: {run_id}")
    aggregates = summary["aggregates"]
    if not isinstance(aggregates, dict):
        raise EvidenceError(f"run aggregates must be an object: {run_id}")
    cold_start_annotations = (
        _project_cold_start_safety_annotations(aggregates, source=False)
        if "cold_start_safety_annotations" in aggregates
        else []
    )
    exact_protocol = is_prefix_cache_manifest or memory_suite is not None
    if exact_protocol:
        gates: list[dict[str, Any]] = []
    else:
        if "startup_safety_gates" not in aggregates:
            raise EvidenceError("published startup safety gates are missing")
        gates = _project_startup_safety_gates(aggregates["startup_safety_gates"])
        if gates:
            if aggregates.get("startup_measurement_valid") is not False:
                raise EvidenceError(
                    "startup safety gates require startup_measurement_valid=false"
                )
            startup_count = aggregates.get(
                "startup_measurement_annotations_count"
            )
            measurement_count = aggregates.get("measurement_annotations_count")
            if (
                type(startup_count) is not int
                or startup_count < len(gates)
                or type(measurement_count) is not int
                or measurement_count < startup_count
            ):
                raise EvidenceError(
                    "startup safety-gate annotation counts are inconsistent"
                )
    _validate_startup_safety_representation_consistency(
        gates,
        cold_start_annotations,
    )
    if memory_suite is not None:
        aggregates = _project_memory_summary_document(aggregates, source=False)
        if manifest.get("status") != aggregates.get("status"):
            raise EvidenceError("memory manifest status disagrees with its aggregates")
    _validate_agentic_aggregates(
        samples["samples"],
        aggregates,
        suite=agentic_suite,
        terminal=bool(entry["measurement_terminal"]),
    )
    _validate_memory_aggregates(
        samples["samples"],
        aggregates,
        model=manifest.get("model"),
        suite=memory_suite,
        terminal=bool(entry["measurement_terminal"]),
    )
    _validate_prefix_cache_aggregates(
        samples["samples"],
        aggregates,
        model=manifest.get("model"),
        suite=suite,
    )

    telemetry = _load_json(directory / "telemetry.json", root)
    telemetry = _expect_object_keys(
        telemetry,
        {
            "chunk_count",
            "chunks",
            "columns",
            "sample_count",
            "schema_version",
            "segment_count",
        },
        name="telemetry index",
    )
    chunks = telemetry["chunks"]
    if (
        telemetry["schema_version"] != SCHEMA_VERSION
        or telemetry["columns"] != list(TELEMETRY_COLUMNS)
        or not isinstance(chunks, list)
        or telemetry["chunk_count"] != len(chunks)
        or chunks != [f"telemetry-{index:04d}.json" for index in range(1, len(chunks) + 1)]
    ):
        raise EvidenceError(f"telemetry index mismatch: {run_id}")
    if memory_suite is not None:
        expected_memory_telemetry = _telemetry_files([])["telemetry.json"]
        if not _json_strict_equal(telemetry, expected_memory_telemetry):
            raise EvidenceError("memory telemetry index changed")
        _validate_memory_bundle_file_set(
            {path.name for path in directory.iterdir()},
            chunks,
            include_checksums=True,
        )
    sample_count = 0
    segment_count = 0
    expected_sample_index = 1
    telemetry_chunks: list[dict[str, Any]] = []
    for chunk_name in chunks:
        chunk = _load_json(directory / chunk_name, root)
        chunk = _expect_object_keys(
            chunk,
            {"sample_count", "schema_version", "segments"},
            name="telemetry chunk",
        )
        if chunk["schema_version"] != SCHEMA_VERSION or not isinstance(
            chunk["segments"], list
        ):
            raise EvidenceError(f"telemetry chunk schema mismatch: {run_id}")
        telemetry_chunks.append(chunk)
        rows_in_chunk = 0
        for segment in chunk["segments"]:
            segment = _expect_object_keys(
                segment,
                {
                    "first_phase_sample_index",
                    "first_sample_index",
                    "phase",
                    "phase_segment",
                    "rows",
                },
                name="telemetry segment",
            )
            if segment["first_sample_index"] != expected_sample_index:
                raise EvidenceError(f"telemetry sample order mismatch: {run_id}")
            if not isinstance(segment["rows"], list):
                raise EvidenceError(f"telemetry rows are invalid: {run_id}")
            for row in segment["rows"]:
                if not isinstance(row, list) or len(row) != len(TELEMETRY_COLUMNS):
                    raise EvidenceError(f"telemetry row width mismatch: {run_id}")
            rows = len(segment["rows"])
            expected_sample_index += rows
            rows_in_chunk += rows
            segment_count += 1
        if chunk["sample_count"] != rows_in_chunk:
            raise EvidenceError(f"telemetry chunk count mismatch: {run_id}")
        sample_count += rows_in_chunk
    if (
        telemetry["sample_count"] != sample_count
        or telemetry["segment_count"] != segment_count
    ):
        raise EvidenceError(f"telemetry total mismatch: {run_id}")
    if is_prefix_cache_manifest:
        _validate_prefix_cache_telemetry_documents(
            telemetry, telemetry_chunks, suite=suite
        )
        _validate_prefix_cache_bundle_file_set(
            {path.name for path in directory.iterdir()},
            telemetry["chunks"],
            include_checksums=True,
        )


def _verify_harbor_bundle(
    root: Path,
    directory: Path,
    entry: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if {path.name for path in directory.iterdir()} != {
        "checksums.json",
        "manifest.json",
        "replicates.json",
    }:
        raise EvidenceError("Harbor evidence bundle file set changed")
    replicates_document = _load_json(directory / "replicates.json", root)
    replicates_document = _expect_object_keys(
        replicates_document,
        {"replicate_count", "replicates", "schema_version"},
        name="Harbor replicates",
    )
    replicates = replicates_document["replicates"]
    if (
        replicates_document["schema_version"] != HARBOR_SCHEMA_VERSION
        or type(replicates_document["replicate_count"]) is not int
        or replicates_document["replicate_count"] != HARBOR_REPLICATE_COUNT
        or not isinstance(replicates, list)
        or len(replicates) != HARBOR_REPLICATE_COUNT
    ):
        raise EvidenceError("Harbor replicate document changed")
    validated = tuple(_validate_harbor_envelope(value) for value in replicates)
    ordered = _order_harbor_replicates(validated)
    if not _json_strict_equal(list(ordered), replicates):
        raise EvidenceError("Harbor replicates are not ordered by started_at")
    expected_manifest = _harbor_manifest(ordered)
    if not _json_strict_equal(manifest, expected_manifest):
        raise EvidenceError("Harbor evidence manifest changed")
    if (
        entry["campaign_id"] != HARBOR_CAMPAIGN_ID
        or entry["evidence_kind"] != HARBOR_EVIDENCE_KIND
        or entry["status"] != "complete"
    ):
        raise EvidenceError("Harbor campaign index entry changed")


def _validate_projected_loop_manifest(value: Any) -> dict[str, Any]:
    expected_fields = {
        "campaign_id",
        "cases",
        "dataset",
        "evidence_kind",
        "lifecycle",
        "models",
        "plan_fingerprint",
        "plan_integrity_sha256",
        "plan_schema_version",
        "protocol",
        "protocol_version",
        "repository",
        "run_date_utc",
        "sanitization",
        "schema_version",
        "source_group",
        "source_run_id",
        "status",
        "upstreams",
        "worker",
    }
    manifest = _expect_object_keys(value, expected_fields, name="loop manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("evidence_kind") != LOOP_EVIDENCE_KIND
        or manifest.get("source_group") not in LOOP_RESULT_ROOTS
    ):
        raise EvidenceError("loop manifest identity changed")
    run_id = _safe_id(manifest.get("source_run_id"), name="loop source run ID")
    if manifest.get("run_date_utc") != _date_from_run_id(run_id):
        raise EvidenceError("loop manifest date changed")
    _safe_id(manifest.get("campaign_id"), name="loop protocol campaign ID")
    fingerprint = _sha256(
        manifest.get("plan_fingerprint"), name="loop manifest fingerprint"
    )
    _sha256(manifest.get("plan_integrity_sha256"), name="loop manifest integrity")
    if not run_id.endswith(f"-{fingerprint[:8]}"):
        raise EvidenceError("loop manifest run ID does not bind its fingerprint")
    schema = manifest.get("plan_schema_version")
    protocol_version = manifest.get("protocol_version")
    protocol = manifest.get("protocol")
    if (
        (schema, protocol_version) not in {(1, 1), (2, 2)}
        or not isinstance(protocol, dict)
        or set(protocol)
        != {"halo", "rlm", "rlm_compaction_admission"}
    ):
        raise EvidenceError("loop manifest protocol version changed")
    variant = (
        "legacy_v1"
        if schema == 1
        else (
            "current_v2"
            if protocol.get("rlm_compaction_admission") is not None
            else "legacy_v2"
        )
    )
    rlm_v1_fields = {
        "direct_lengths",
        "direct_timeout_s",
        "episode_timeout_s",
        "lengths",
        "max_concurrent_subcalls",
        "max_iterations",
        "max_output_tokens",
        "max_total_tokens",
        "model_profile",
        "recursive_depth2_lengths",
        "recursive_depth2_rows",
        "recursive_depth2_tasks",
        "row_indices",
        "tasks",
        "worker_isolation",
    }
    expected_rlm = {
        "legacy_v1": rlm_v1_fields,
        "legacy_v2": rlm_v1_fields | {"reasoning_control"},
        "current_v2": rlm_v1_fields
        | {"compaction", "compaction_threshold_pct", "reasoning_control"},
    }[variant]
    rlm = protocol.get("rlm")
    if not isinstance(rlm, dict):
        raise EvidenceError("loop manifest RLM protocol is invalid")
    if variant == "current_v2" and "compaction_diagnostic" in rlm:
        expected_rlm = expected_rlm | {"compaction_diagnostic"}
    if set(rlm) != expected_rlm:
        raise EvidenceError("loop manifest RLM protocol schema changed")
    validated_rlm = _project_loop_protocol_section(
        rlm, name="published rlm", allowed=frozenset(expected_rlm)
    )
    if not _json_strict_equal(validated_rlm, rlm):
        raise EvidenceError("loop manifest RLM protocol projection changed")
    halo_v1_fields = {
        "depth2_seeds",
        "depth2_trace_counts",
        "depths",
        "episode_timeout_s",
        "max_output_tokens",
        "max_parallel",
        "max_turns",
        "model_profiles",
        "seeds",
        "trace_counts",
    }
    expected_halo = (
        halo_v1_fields
        if variant == "legacy_v1"
        else halo_v1_fields | {"reasoning_effort"}
    )
    halo = protocol.get("halo")
    if not isinstance(halo, dict) or set(halo) != expected_halo:
        raise EvidenceError("loop manifest HALO protocol schema changed")
    validated_halo = _project_loop_protocol_section(
        halo, name="published halo", allowed=frozenset(expected_halo)
    )
    if not _json_strict_equal(validated_halo, halo):
        raise EvidenceError("loop manifest HALO protocol projection changed")
    admission = protocol.get("rlm_compaction_admission")
    if admission is not None:
        admission_fields = frozenset(
            {
                "depth1_admitted",
                "depth2_admitted",
                "enabled",
                "headroom_tokens",
                "output_reserve_tokens",
                "package_context_tokens",
                "served_context_tokens",
                "threshold_pct",
                "threshold_tokens",
            }
        )
        if not isinstance(admission, dict) or set(admission) != admission_fields:
            raise EvidenceError("loop manifest compaction admission schema changed")
        if not _json_strict_equal(
            _project_loop_protocol_section(
                admission,
                name="published compaction admission",
                allowed=admission_fields,
            ),
            admission,
        ):
            raise EvidenceError("loop manifest compaction admission changed")
    if variant == "current_v2" and (
        rlm.get("compaction") is not True
        or rlm.get("compaction_threshold_pct") != 0.85
        or rlm.get("reasoning_control") != "fixed_unsupported"
        or halo.get("reasoning_effort") != "none"
    ):
        raise EvidenceError("loop manifest current protocol semantics changed")
    if variant == "legacy_v2" and (
        rlm.get("reasoning_control") != "fixed_unsupported"
        or halo.get("reasoning_effort") != "none"
    ):
        raise EvidenceError("loop manifest legacy-v2 semantics changed")

    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise EvidenceError("loop manifest cases are missing")
    projected_cases = [
        _project_loop_case(case, variant=variant) for case in cases
    ]
    if not _json_strict_equal(projected_cases, cases):
        raise EvidenceError("loop manifest case projection changed")
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise EvidenceError("loop manifest case IDs are duplicated")

    models = manifest.get("models")
    model_allowed = {
        "architecture",
        "backend",
        "container_image",
        "container_image_sha256",
        "estimated_ram_gib",
        "id",
        "lifecycle",
        "max_context",
        "native_context",
        "prefix_cache_mode",
        "quantization",
        "recipe_revision",
        "revision",
        "runtime_parallel",
        "source",
        "startup_timeout_s",
        "support_status",
        "tasks",
        "weight_file_count",
        "weight_size_bytes",
    } | set(_SGLANG_PROVENANCE_RUNTIME_FIELDS)
    model_required = {
        "architecture",
        "backend",
        "container_image",
        "container_image_sha256",
        "estimated_ram_gib",
        "id",
        "lifecycle",
        "max_context",
        "native_context",
        "quantization",
        "revision",
        "source",
        "startup_timeout_s",
        "support_status",
        "tasks",
    }
    if not isinstance(models, list) or not models:
        raise EvidenceError("loop manifest models are missing")
    model_ids: set[str] = set()
    for model in models:
        if (
            not isinstance(model, dict)
            or not model_required <= set(model)
            or not set(model) <= model_allowed
        ):
            raise EvidenceError("loop manifest model schema changed")
        model_id = _safe_id(model.get("id"), name="loop manifest model ID")
        if model_id in model_ids:
            raise EvidenceError("loop manifest model IDs are duplicated")
        model_ids.add(model_id)
        _safe_id(model.get("container_image"), name="loop manifest model image")
        _sha256(
            model.get("container_image_sha256"), name="loop manifest model image"
        )
        model_sglang_fields = _SGLANG_PROVENANCE_RUNTIME_FIELDS & set(model)
        if model.get("backend") == "sglang":
            _validate_projected_sglang_provenance(model, model)
        elif model_sglang_fields:
            raise EvidenceError("loop manifest SGLang provenance backend changed")
        base = {
            key: item
            for key, item in model.items()
            if key
            not in {
                "container_image",
                "container_image_sha256",
                "recipe_revision",
                *_SGLANG_PROVENANCE_RUNTIME_FIELDS,
            }
        }
        if not _json_strict_equal(_project_model({"model": base}, None), base):
            raise EvidenceError("loop manifest model projection changed")
    halo_profiles = halo.get("model_profiles")
    referenced_models = {str(rlm.get("model_profile")), *map(str, halo_profiles)}
    if model_ids != referenced_models:
        raise EvidenceError("loop manifest model set changed")

    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or set(dataset) != {
        "artifacts",
        "revision",
        "rows_per_split",
        "source",
    }:
        raise EvidenceError("loop manifest dataset schema changed")
    if (
        dataset.get("source") != _LOOP_UPSTREAMS["babilong_source"]
        or dataset.get("revision") != _LOOP_UPSTREAMS["babilong_revision"]
        or type(dataset.get("rows_per_split")) is not int
        or dataset["rows_per_split"] <= 0
        or not isinstance(dataset.get("artifacts"), list)
    ):
        raise EvidenceError("loop manifest dataset pin changed")
    targets: list[str] = []
    for artifact in dataset["artifacts"]:
        artifact = _expect_object_keys(
            artifact,
            {"sha256", "size_bytes", "target"},
            name="loop manifest dataset artifact",
        )
        _sha256(artifact.get("sha256"), name="loop manifest dataset artifact")
        target = _safe_id(
            artifact.get("target"), name="loop manifest dataset target"
        )
        if not any(
            target == f"{context_length}-{task}"
            for context_length in _LOOP_CONTEXT_LENGTHS
            for task in _LOOP_TASKS
        ):
            raise EvidenceError("loop manifest dataset target changed")
        if type(artifact.get("size_bytes")) is not int or artifact["size_bytes"] <= 0:
            raise EvidenceError("loop manifest dataset artifact size changed")
        targets.append(artifact["target"])
    if targets != sorted(set(targets)):
        raise EvidenceError("loop manifest dataset artifact order changed")
    if manifest.get("upstreams") != _LOOP_UPSTREAMS:
        raise EvidenceError("loop manifest upstreams changed")
    repository = _expect_object_keys(
        manifest.get("repository"), {"clean", "revision"}, name="loop repository"
    )
    if repository.get("clean") is not True:
        raise EvidenceError("loop manifest repository was dirty")
    _revision(repository.get("revision"), name="loop manifest repository revision")
    if manifest.get("worker") != {
        "container_image": _LOOP_WORKER_IMAGE.split("@", 1)[0],
        "container_image_sha256": _sha256(
            _LOOP_WORKER_IMAGE.split("@", 1)[1], name="fixed loop worker image"
        ),
        "isolation": "docker",
    }:
        raise EvidenceError("loop manifest worker pin changed")
    if manifest.get("sanitization") != {
        "free_form_text_included": False,
        "payloads_included": False,
        "policy": SANITIZATION_POLICY,
        "raw_identifiers_included": False,
    }:
        raise EvidenceError("loop manifest sanitization claim changed")
    lifecycle = _expect_object_keys(
        manifest.get("lifecycle"),
        {
            "cleanup_verified",
            "event_count",
            "event_counts",
            "journal_elapsed_s",
            "selected_halo_profile",
            "terminal",
            "terminal_event",
        },
        name="loop lifecycle",
    )
    event_counts = lifecycle.get("event_counts")
    if not isinstance(event_counts, dict) or any(
        name not in _LOOP_EVENT_TYPES or type(count) is not int or count < 0
        for name, count in event_counts.items()
    ):
        raise EvidenceError("loop lifecycle event counts changed")
    if (
        type(lifecycle.get("event_count")) is not int
        or lifecycle["event_count"] != sum(event_counts.values())
        or type(lifecycle.get("cleanup_verified")) is not bool
        or type(lifecycle.get("terminal")) is not bool
    ):
        raise EvidenceError("loop lifecycle counters changed")
    elapsed = _finite(lifecycle.get("journal_elapsed_s"), name="loop lifecycle elapsed")
    if elapsed is not None and elapsed < 0:
        raise EvidenceError("loop lifecycle elapsed time changed")
    selected = _safe_id(
        lifecycle.get("selected_halo_profile"), name="loop selected HALO profile"
    )
    if selected not in set(halo_profiles):
        raise EvidenceError("loop selected HALO profile is not frozen")
    expected_terminal = event_counts.get("campaign_finished", 0) > 0
    if (
        lifecycle["terminal"] is not expected_terminal
        or lifecycle.get("terminal_event")
        != ("campaign_finished" if expected_terminal else None)
        or lifecycle["cleanup_verified"]
        is not (event_counts.get("campaign_cleanup_verified", 0) > 0)
    ):
        raise EvidenceError("loop lifecycle terminal classification changed")
    allowed_statuses = {
        "cleanup_failed",
        "complete",
        "measurements_complete_cleanup_pending",
        "not_started",
        "partial",
        "planned",
    }
    if manifest.get("status") not in allowed_statuses:
        raise EvidenceError("loop manifest status changed")
    return manifest


def _verify_loop_telemetry(
    directory: Path, root: Path, *, campaign_id: str
) -> None:
    telemetry = _load_json(directory / "telemetry.json", root)
    telemetry = _expect_object_keys(
        telemetry,
        {
            "chunk_count",
            "chunks",
            "columns",
            "sample_count",
            "schema_version",
            "segment_count",
        },
        name="loop telemetry index",
    )
    chunks = telemetry.get("chunks")
    if (
        telemetry.get("schema_version") != SCHEMA_VERSION
        or telemetry.get("columns") != list(TELEMETRY_COLUMNS)
        or not isinstance(chunks, list)
        or telemetry.get("chunk_count") != len(chunks)
        or chunks
        != [f"telemetry-{index:04d}.json" for index in range(1, len(chunks) + 1)]
    ):
        raise EvidenceError(f"loop telemetry index mismatch: {campaign_id}")
    sample_count = 0
    segment_count = 0
    expected_sample_index = 1
    previous_phase_segment = 0
    previous_phase: str | None = None
    previous_phase_sample_index = 0
    previous_elapsed: float | None = None
    allowed_phases = {
        "finalize",
        "halo_cases",
        "halo_cleanup",
        "halo_index",
        "halo_server_restart",
        "halo_server_start",
        "idle",
        "rlm_cases",
        "rlm_cleanup",
        "rlm_server_restart",
        "rlm_server_start",
    }
    for chunk_index, chunk_name in enumerate(chunks):
        chunk = _expect_object_keys(
            _load_json(directory / chunk_name, root),
            {"sample_count", "schema_version", "segments"},
            name="loop telemetry chunk",
        )
        if chunk.get("schema_version") != SCHEMA_VERSION or not isinstance(
            chunk.get("segments"), list
        ):
            raise EvidenceError("loop telemetry chunk schema changed")
        rows_in_chunk = 0
        for segment_index, segment in enumerate(chunk["segments"]):
            segment = _expect_object_keys(
                segment,
                {
                    "first_phase_sample_index",
                    "first_sample_index",
                    "phase",
                    "phase_segment",
                    "rows",
                },
                name="loop telemetry segment",
            )
            phase = _safe_id(segment.get("phase"), name="loop telemetry phase")
            rows = segment.get("rows")
            phase_segment = segment.get("phase_segment")
            first_phase_sample_index = segment.get("first_phase_sample_index")
            if (
                phase not in allowed_phases
                or type(first_phase_sample_index) is not int
                or type(segment.get("first_sample_index")) is not int
                or segment["first_sample_index"] != expected_sample_index
                or type(phase_segment) is not int
                or not isinstance(rows, list)
                or not rows
            ):
                raise EvidenceError("loop telemetry segment topology changed")
            continuing = phase_segment == previous_phase_segment
            if continuing:
                if (
                    chunk_index == 0
                    or segment_index != 0
                    or phase != previous_phase
                    or first_phase_sample_index != previous_phase_sample_index + 1
                ):
                    raise EvidenceError("loop telemetry continuation changed")
            elif (
                phase_segment != previous_phase_segment + 1
                or first_phase_sample_index != 1
                or (previous_phase is not None and phase == previous_phase)
                or not isinstance(rows[0], list)
                or len(rows[0]) != len(TELEMETRY_COLUMNS)
                or rows[0][TELEMETRY_COLUMNS.index("elapsed_s")] != 0
            ):
                raise EvidenceError("loop telemetry phase transition changed")
            elapsed_index = TELEMETRY_COLUMNS.index("elapsed_s")
            error_index = TELEMETRY_COLUMNS.index("gpu_error_present")
            gpu_index = TELEMETRY_COLUMNS.index("gpu_util_pct")
            memory_index = TELEMETRY_COLUMNS.index("memory_util_pct")
            byte_indices = {
                TELEMETRY_COLUMNS.index(name)
                for name in (
                    "cached_bytes",
                    "memavailable_bytes",
                    "memfree_bytes",
                    "swapfree_bytes",
                    "swaptotal_bytes",
                )
            }
            segment_previous_elapsed = previous_elapsed if continuing else None
            for row in rows:
                if not isinstance(row, list) or len(row) != len(TELEMETRY_COLUMNS):
                    raise EvidenceError("loop telemetry row width changed")
                for index, item in enumerate(row):
                    if item is None:
                        raise EvidenceError("loop telemetry row contains a missing reading")
                    if index == error_index:
                        if type(item) is not bool:
                            raise EvidenceError("loop telemetry error flag changed")
                        continue
                    if item is not None and (
                        not isinstance(item, (int, float)) or isinstance(item, bool)
                    ):
                        raise EvidenceError("loop telemetry row contains nonnumeric data")
                    if isinstance(item, float) and not math.isfinite(item):
                        raise EvidenceError("loop telemetry row is non-finite")
                    if index in byte_indices and item is not None and (
                        type(item) is not int or item < 0
                    ):
                        raise EvidenceError("loop telemetry byte counter changed")
                    if index in {gpu_index, memory_index} and item is not None and not 0 <= item <= 100:
                        raise EvidenceError("loop telemetry utilization changed")
                    if index not in {gpu_index, memory_index, elapsed_index, error_index} and index not in byte_indices and item is not None and item < 0:
                        raise EvidenceError("loop telemetry physical metric changed")
                elapsed = row[elapsed_index]
                if (
                    not isinstance(elapsed, (int, float))
                    or isinstance(elapsed, bool)
                    or elapsed < 0
                    or (
                        segment_previous_elapsed is not None
                        and elapsed < segment_previous_elapsed
                    )
                ):
                    raise EvidenceError("loop telemetry elapsed time changed")
                segment_previous_elapsed = float(elapsed)
            row_count = len(rows)
            expected_sample_index += row_count
            rows_in_chunk += row_count
            segment_count += 1
            previous_phase_segment = phase_segment
            previous_phase = phase
            previous_phase_sample_index = first_phase_sample_index + row_count - 1
            previous_elapsed = float(rows[-1][elapsed_index])
        if chunk.get("sample_count") != rows_in_chunk:
            raise EvidenceError("loop telemetry chunk count changed")
        sample_count += rows_in_chunk
    if (
        telemetry.get("sample_count") != sample_count
        or telemetry.get("segment_count") != segment_count
    ):
        raise EvidenceError("loop telemetry totals changed")
    expected_files = {
        "checksums.json",
        "manifest.json",
        "measurements.json",
        "outcomes.json",
        "summary.json",
        "telemetry.json",
        *chunks,
    }
    if {path.name for path in directory.iterdir()} != expected_files:
        raise EvidenceError("loop campaign bundle file set changed")


def _verify_loop_campaign_bundle(
    root: Path,
    directory: Path,
    entry: dict[str, Any],
    primary: Any,
) -> None:
    manifest = _validate_projected_loop_manifest(primary)
    campaign_id = str(entry["campaign_id"])
    if (
        manifest.get("source_run_id") != campaign_id
        or manifest.get("status") != entry.get("status")
        or entry.get("evidence_kind") != LOOP_EVIDENCE_KIND
    ):
        raise EvidenceError("loop campaign index binding changed")
    measurements_document = _expect_object_keys(
        _load_json(directory / "measurements.json", root),
        {"measurement_count", "measurements", "schema_version"},
        name="loop measurements",
    )
    measurements = measurements_document.get("measurements")
    if (
        measurements_document.get("schema_version") != SCHEMA_VERSION
        or not isinstance(measurements, list)
        or measurements_document.get("measurement_count") != len(measurements)
    ):
        raise EvidenceError("loop measurement count changed")
    cases = {case["case_id"]: case for case in manifest["cases"]}
    model_ids = {model["id"] for model in manifest["models"]}
    seen_cases: set[str] = set()
    for index, measurement in enumerate(measurements, 1):
        projected = _project_loop_measurement(measurement, source=False)
        if not _json_strict_equal(projected, measurement):
            raise EvidenceError("loop measurement projection changed")
        if measurement.get("sample_index") != index:
            raise EvidenceError("loop measurement order changed")
        case_id = measurement.get("case_id")
        if case_id not in cases or case_id in seen_cases:
            raise EvidenceError("loop measurement case binding changed")
        seen_cases.add(str(case_id))
        case = cases[str(case_id)]
        required_dimensions = _LOOP_BOUND_DIMENSION_FIELDS & set(case)
        if not required_dimensions <= set(measurement) or not _loop_case_dimensions_match(
            measurement, case
        ):
            raise EvidenceError("loop measurement dimensions changed")
        profile_id = measurement.get("profile_id")
        valid_profiles = (
            {manifest["protocol"]["rlm"]["model_profile"]}
            if case["phase"] == "rlm"
            else set(manifest["protocol"]["halo"]["model_profiles"])
        )
        if profile_id not in model_ids or profile_id not in valid_profiles:
            raise EvidenceError("loop measurement profile binding changed")

    outcomes_document = _expect_object_keys(
        _load_json(directory / "outcomes.json", root),
        {"outcome_count", "outcomes", "schema_version"},
        name="loop outcomes",
    )
    outcomes = outcomes_document.get("outcomes")
    if (
        outcomes_document.get("schema_version") != SCHEMA_VERSION
        or not isinstance(outcomes, list)
        or outcomes_document.get("outcome_count") != len(outcomes)
        or len(outcomes) != len(manifest["cases"])
    ):
        raise EvidenceError("loop outcome count changed")
    allowed_outcomes = {
        "complete",
        "exhausted",
        "held",
        "incomplete",
        "not_started",
        "skipped_campaign_stop",
        "skipped_deadline",
    }
    for case, outcome in zip(manifest["cases"], outcomes, strict=True):
        outcome = _expect_object_keys(
            outcome,
            {
                "attempt_count",
                "case_id",
                "failed_attempt_count",
                "outcome",
                "timeout_attempt_count",
            },
            name="loop outcome",
        )
        if outcome.get("case_id") != case["case_id"]:
            raise EvidenceError("loop outcome order or case binding changed")
        _safe_id(outcome.get("case_id"), name="loop outcome case ID")
        if outcome.get("outcome") not in allowed_outcomes:
            raise EvidenceError("loop outcome status changed")
        for key in (
            "attempt_count",
            "failed_attempt_count",
            "timeout_attempt_count",
        ):
            if type(outcome.get(key)) is not int or outcome[key] < 0:
                raise EvidenceError("loop outcome attempt counter changed")
        if (
            outcome["attempt_count"] > 2
            or outcome["failed_attempt_count"]
            + outcome["timeout_attempt_count"]
            > outcome["attempt_count"]
            or (
                outcome["outcome"] in {"not_started", "held"}
                and outcome["attempt_count"] != 0
            )
            or (
                outcome["outcome"] == "incomplete"
                and outcome["attempt_count"] == 0
            )
            or (
                outcome["outcome"] == "complete"
                and outcome["attempt_count"] == 0
            )
            or (
                outcome["outcome"] == "exhausted"
                and (
                    outcome["attempt_count"] != 2
                    or outcome["failed_attempt_count"]
                    + outcome["timeout_attempt_count"]
                    != 2
                )
            )
        ):
            raise EvidenceError("loop outcome attempt accounting changed")
    completed_outcomes = {
        row["case_id"] for row in outcomes if row["outcome"] == "complete"
    }
    if completed_outcomes != seen_cases:
        raise EvidenceError("loop complete outcomes disagree with measurements")
    outcomes_by_case = {row["case_id"]: row for row in outcomes}
    if any(
        measurement["attempt"]
        != outcomes_by_case[str(measurement["case_id"])]["attempt_count"]
        for measurement in measurements
    ):
        raise EvidenceError("loop completion attempt disagrees with its outcome")

    summary_document = _expect_object_keys(
        _load_json(directory / "summary.json", root),
        {"aggregates", "schema_version"},
        name="loop summary",
    )
    summary = summary_document.get("aggregates")
    summary_fields = {
        "completed_cases",
        "deadline_skipped_cases",
        "exhausted_cases",
        "failed_attempts",
        "groups",
        "held_cases",
        "planned_cases",
        "status",
    }
    if (
        summary_document.get("schema_version") != SCHEMA_VERSION
        or not isinstance(summary, dict)
        or set(summary) != summary_fields
    ):
        raise EvidenceError("loop published summary schema changed")
    for key in (
        "completed_cases",
        "deadline_skipped_cases",
        "exhausted_cases",
        "failed_attempts",
        "held_cases",
        "planned_cases",
    ):
        if type(summary.get(key)) is not int or summary[key] < 0:
            raise EvidenceError("loop published summary counter changed")
    lifecycle = manifest["lifecycle"]
    event_counts = lifecycle["event_counts"]
    expected_counts = {
        "completed_cases": sum(row["outcome"] == "complete" for row in outcomes),
        "deadline_skipped_cases": sum(
            row["outcome"] in {"skipped_deadline", "skipped_campaign_stop"}
            for row in outcomes
        ),
        "exhausted_cases": sum(row["outcome"] == "exhausted" for row in outcomes),
        "failed_attempts": sum(
            row["failed_attempt_count"] + row["timeout_attempt_count"]
            for row in outcomes
        ),
        "held_cases": sum(row["outcome"] == "held" for row in outcomes),
        "planned_cases": len(cases),
    }
    if any(summary[key] != expected for key, expected in expected_counts.items()):
        raise EvidenceError("loop published summary counters disagree with lifecycle")
    if summary["completed_cases"] != len(measurements):
        raise EvidenceError("loop summary disagrees with its measurements")
    expected_event_counts = {
        "case_complete": expected_counts["completed_cases"],
        "case_exhausted": expected_counts["exhausted_cases"],
        "case_failed": sum(row["failed_attempt_count"] for row in outcomes),
        "case_skipped_campaign_stop": sum(
            row["outcome"] == "skipped_campaign_stop" for row in outcomes
        ),
        "case_skipped_deadline": sum(
            row["outcome"] == "skipped_deadline" for row in outcomes
        ),
        "case_skipped_held": expected_counts["held_cases"],
        "case_started": sum(row["attempt_count"] for row in outcomes),
        "case_timeout": sum(row["timeout_attempt_count"] for row in outcomes),
    }
    if any(
        event_counts.get(name, 0) != count
        for name, count in expected_event_counts.items()
    ):
        raise EvidenceError("loop outcomes disagree with lifecycle event counts")
    pseudo_events: list[dict[str, Any]] = [
        {"event": "case_complete", "case_id": row["case_id"]}
        for row in measurements
    ]
    selected = lifecycle["selected_halo_profile"]
    if selected != manifest["protocol"]["halo"]["model_profiles"][0]:
        pseudo_events.append(
            {"event": "halo_fallback_selected", "profile_id": selected}
        )
    for outcome in outcomes:
        event_name = {
            "exhausted": "case_exhausted",
            "held": "case_skipped_held",
            "skipped_campaign_stop": "case_skipped_campaign_stop",
            "skipped_deadline": "case_skipped_deadline",
        }.get(outcome["outcome"])
        if event_name is not None:
            pseudo_events.append(
                {"event": event_name, "case_id": outcome["case_id"]}
            )
        pseudo_events.extend(
            {"event": "case_failed", "case_id": outcome["case_id"]}
            for _ in range(outcome["failed_attempt_count"])
        )
        pseudo_events.extend(
            {"event": "case_timeout", "case_id": outcome["case_id"]}
            for _ in range(outcome["timeout_attempt_count"])
        )
    if event_counts.get("campaign_cleanup_failed", 0):
        pseudo_events.append({"event": "campaign_cleanup_failed"})
    elif lifecycle["cleanup_verified"]:
        pseudo_events.append({"event": "campaign_cleanup_verified"})
    expected_summary = _loop_summary_expected(
        plan=manifest, events=pseudo_events, measurements=measurements
    )
    expected_status = "planned" if summary["status"] == "planned" else expected_summary["status"]
    groups_match = (
        summary.get("groups") == []
        if summary["status"] == "planned"
        else _loop_values_equal(summary.get("groups"), expected_summary["groups"])
    )
    if (
        summary.get("status") != expected_status
        or manifest.get("status") != summary.get("status")
        or not groups_match
    ):
        raise EvidenceError("loop published summary aggregates changed")
    if summary["status"] == "planned" and (
        measurements
        or lifecycle["event_count"] != 0
        or any(row["outcome"] != "not_started" for row in outcomes)
        or _load_json(directory / "telemetry.json", root).get("sample_count") != 0
    ):
        raise EvidenceError("loop planned bundle contains execution evidence")
    _verify_loop_telemetry(directory, root, campaign_id=campaign_id)


def _verify_simple_bundle(
    root: Path,
    entry: dict[str, Any],
    *,
    category: str,
    identity_key: str,
) -> None:
    expected_keys = {
        "bundle_sha256",
        "evidence_kind",
        "file",
        identity_key,
        "status",
    }
    _expect_object_keys(entry, expected_keys, name=f"{category} index entry")
    identity = _safe_id(entry[identity_key], name=f"{category} ID")
    directory = root / category / identity
    if _verify_bundle(directory, root) != _sha256(
        entry["bundle_sha256"], name=f"{category} bundle"
    ):
        raise EvidenceError(f"{category} bundle digest mismatch: {identity}")
    relative = Path(entry["file"])
    if relative.parts[:2] != (category, identity) or len(relative.parts) != 3:
        raise EvidenceError(f"unsafe {category} manifest pointer")
    primary = _load_json(root / relative, root)
    if not isinstance(primary, dict):
        raise EvidenceError(f"{category} schema mismatch: {identity}")
    if primary.get("evidence_kind") != entry["evidence_kind"]:
        raise EvidenceError(f"{category} kind mismatch: {identity}")
    if category == "campaigns" and (
        identity == HARBOR_CAMPAIGN_ID
        or entry["evidence_kind"] == HARBOR_EVIDENCE_KIND
    ):
        if (
            identity != HARBOR_CAMPAIGN_ID
            or entry["evidence_kind"] != HARBOR_EVIDENCE_KIND
        ):
            raise EvidenceError("Harbor campaign identity or evidence kind changed")
        _verify_harbor_bundle(root, directory, entry, primary)
        return
    if category == "campaigns" and (
        entry.get("evidence_kind") == LOOP_EVIDENCE_KIND
        or primary.get("evidence_kind") == LOOP_EVIDENCE_KIND
        or primary.get("source_group") in LOOP_RESULT_ROOTS
        or _RUN_ID_RE.fullmatch(str(identity)) is not None
    ):
        if (
            entry.get("evidence_kind") != LOOP_EVIDENCE_KIND
            or primary.get("evidence_kind") != LOOP_EVIDENCE_KIND
        ):
            raise EvidenceError("loop campaign evidence kind changed")
        _verify_loop_campaign_bundle(root, directory, entry, primary)
        return
    if primary.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError(f"{category} schema mismatch: {identity}")
    measurements_path = directory / "measurements.json"
    if measurements_path.is_file():
        measurements = _load_json(measurements_path, root)
        measurements = _expect_object_keys(
            measurements,
            {"measurement_count", "measurements", "schema_version"},
            name="campaign measurements",
        )
        if (
            measurements["schema_version"] != SCHEMA_VERSION
            or not isinstance(measurements["measurements"], list)
            or measurements["measurement_count"] != len(measurements["measurements"])
        ):
            raise EvidenceError(f"campaign measurement count mismatch: {identity}")
        telemetry = _load_json(directory / "telemetry.json", root)
        telemetry = _expect_object_keys(
            telemetry,
            {"capture_count", "captures", "schema_version"},
            name="campaign telemetry",
        )
        if (
            telemetry["schema_version"] != SCHEMA_VERSION
            or not isinstance(telemetry["captures"], list)
            or telemetry["capture_count"] != len(telemetry["captures"])
        ):
            raise EvidenceError(f"campaign telemetry count mismatch: {identity}")


def _verify_evidence_topology(root: Path) -> None:
    expected_top = {"README.md", "campaigns", "checksums.json", "index.json", "matrices", "runs", "standalone"}
    if {entry.name for entry in root.iterdir()} != expected_top:
        raise EvidenceError("evidence top-level layout changed")
    index = _load_json(root / "index.json", root)
    index = _expect_object_keys(
        index,
        {
            "campaign_count",
            "campaigns",
            "matrix_count",
            "matrices",
            "run_count",
            "run_status_counts",
            "runs",
            "sanitization_policy",
            "schema_version",
            "source_control_files_ignored",
            "source_file_count",
            "source_size_bytes",
            "standalone",
            "standalone_count",
        },
        name="evidence index",
    )
    if (
        index["schema_version"] != SCHEMA_VERSION
        or index["sanitization_policy"] != SANITIZATION_POLICY
    ):
        raise EvidenceError("evidence index version changed")
    for key, count_key in (
        ("runs", "run_count"),
        ("campaigns", "campaign_count"),
        ("matrices", "matrix_count"),
        ("standalone", "standalone_count"),
    ):
        if not isinstance(index[key], list) or index[count_key] != len(index[key]):
            raise EvidenceError(f"evidence index count mismatch: {key}")

    run_ids = [entry.get("run_id") for entry in index["runs"] if isinstance(entry, dict)]
    if len(run_ids) != len(set(run_ids)) or set(run_ids) != {
        directory.name for directory in (root / "runs").iterdir()
    }:
        raise EvidenceError("run bundle index mismatch")
    for entry in index["runs"]:
        if not isinstance(entry, dict):
            raise EvidenceError("run index entry must be an object")
        _verify_run_bundle(root, entry)
    expected_status_counts = dict(
        sorted(Counter(entry["status"] for entry in index["runs"]).items())
    )
    if index["run_status_counts"] != expected_status_counts:
        raise EvidenceError("run status counts do not match entries")

    campaign_ids = [
        entry.get("campaign_id") for entry in index["campaigns"] if isinstance(entry, dict)
    ]
    if len(campaign_ids) != len(set(campaign_ids)) or set(campaign_ids) != {
        directory.name for directory in (root / "campaigns").iterdir()
    }:
        raise EvidenceError("campaign bundle index mismatch")
    for entry in index["campaigns"]:
        if not isinstance(entry, dict):
            raise EvidenceError("campaign index entry must be an object")
        _verify_simple_bundle(
            root, entry, category="campaigns", identity_key="campaign_id"
        )

    standalone_ids = [entry.get("id") for entry in index["standalone"] if isinstance(entry, dict)]
    if len(standalone_ids) != len(set(standalone_ids)) or set(standalone_ids) != {
        directory.name for directory in (root / "standalone").iterdir()
    }:
        raise EvidenceError("standalone bundle index mismatch")
    for entry in index["standalone"]:
        if not isinstance(entry, dict):
            raise EvidenceError("standalone index entry must be an object")
        _verify_simple_bundle(root, entry, category="standalone", identity_key="id")

    matrix_ids = [entry.get("matrix_id") for entry in index["matrices"] if isinstance(entry, dict)]
    expected_matrix_files = {f"{matrix_id}.json" for matrix_id in matrix_ids}
    if len(matrix_ids) != len(set(matrix_ids)) or expected_matrix_files != {
        path.name for path in (root / "matrices").iterdir()
    }:
        raise EvidenceError("matrix evidence index mismatch")
    for entry in index["matrices"]:
        entry = _expect_object_keys(
            entry, {"file", "matrix_id", "sha256", "status"}, name="matrix index entry"
        )
        matrix_id = _safe_id(entry["matrix_id"], name="matrix ID")
        relative = Path(entry["file"])
        if relative != Path("matrices") / f"{matrix_id}.json":
            raise EvidenceError("matrix pointer mismatch")
        data = _secure_read(root / relative, root, maximum=MAX_OUTPUT_FILE_BYTES).encode()
        if _hash_bytes(data) != _sha256(entry["sha256"], name="matrix evidence"):
            raise EvidenceError("matrix evidence digest mismatch")


def verify_evidence(root: Path) -> dict[str, Any]:
    root = Path(root)
    if root.is_symlink():
        raise EvidenceError("evidence root must be a regular directory")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise EvidenceError("evidence root must be a regular directory")
    files: list[Path] = []
    for directory, directories, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directories:
            path = base / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise EvidenceError(f"evidence contains an unsafe directory: {name}")
        for name in filenames:
            path = base / name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise EvidenceError(f"evidence contains a non-regular file: {name}")
            files.append(path)
    files.sort()
    total = 0
    for path in files:
        metadata = path.lstat()
        if metadata.st_size > MAX_OUTPUT_FILE_BYTES:
            raise EvidenceError(f"evidence file exceeds size limit: {path.name}")
        total += metadata.st_size
        data = path.read_bytes()
        for detector, pattern in _SECRET_PATTERNS:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as error:
                raise EvidenceError(f"evidence is not UTF-8: {path.name}") from error
            if pattern.search(unicodedata.normalize("NFKC", text)):
                raise EvidenceError(f"{detector} detector matched in {path.name}")
        if path.suffix == ".json":
            value = _load_json(path, root)
            _validate_output_value(value, pointer=f"/{path.relative_to(root)}")
        elif path.name != "README.md":
            raise EvidenceError(f"unexpected evidence file type: {path.name}")
    if total > MAX_OUTPUT_TOTAL_BYTES:
        raise EvidenceError("evidence corpus exceeds aggregate size limit")
    checksums_path = root / "checksums.json"
    if not checksums_path.is_file():
        raise EvidenceError("top-level checksums.json is missing")
    checksums = _load_json(checksums_path, root)
    checksums = _expect_object_keys(
        checksums, {"files", "schema_version"}, name="top-level checksums"
    )
    if checksums["schema_version"] != SCHEMA_VERSION:
        raise EvidenceError("invalid top-level checksums")
    expected = checksums.get("files")
    if not isinstance(expected, dict):
        raise EvidenceError("invalid top-level checksum map")
    actual: dict[str, str] = {}
    for path in files:
        if path == checksums_path:
            continue
        actual[str(path.relative_to(root))] = _hash_bytes(path.read_bytes())
    if expected != actual:
        raise EvidenceError("evidence checksums do not match materialized files")
    _verify_evidence_topology(root)
    return {"files": len(files), "size_bytes": total, "status": "verified"}


def _git_bytes(repo_root: Path, *arguments: str) -> bytes:
    process = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise EvidenceError(f"Git index inspection failed during {arguments[0]}")
    return process.stdout


def _scan_staged_blob(data: bytes, *, name: str) -> None:
    if b"\x00" in data:
        raise EvidenceError(f"staged file is binary: {name}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"staged file is not UTF-8: {name}") from error
    normalized = unicodedata.normalize("NFKC", text)
    for detector, pattern in _SECRET_PATTERNS:
        if pattern.search(normalized):
            raise EvidenceError(f"{detector} detector matched staged file {name}")


def verify_staged_evidence(
    *,
    repo_root: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    """Verify the exact Git-index evidence tree and scan every staged text blob."""

    repo_root = repo_root.resolve(strict=True)
    if not (repo_root / ".git").exists():
        raise EvidenceError("staged evidence verification requires a Git worktree")
    evidence_root = Path(evidence_root)
    evidence_candidate = (
        evidence_root
        if evidence_root.is_absolute()
        else repo_root / evidence_root
    ).absolute()
    try:
        evidence_relative = evidence_candidate.relative_to(repo_root)
    except ValueError as error:
        raise EvidenceError("staged evidence must be inside the repository") from error
    if (
        not evidence_relative.parts
        or any(part in {"", ".", ".."} for part in evidence_relative.parts)
        or evidence_relative.is_absolute()
    ):
        raise EvidenceError("unsafe staged evidence path")

    tree_before = _git_bytes(repo_root, "write-tree").strip()
    raw_entries = _git_bytes(repo_root, "ls-files", "--stage", "-z")
    index: dict[str, tuple[str, str, str]] = {}
    for record in raw_entries.split(b"\x00"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise EvidenceError("Git index contains an invalid entry") from error
        if path in index:
            raise EvidenceError(f"Git index contains duplicate path {path}")
        index[path] = (mode, object_id, stage)

    changed_raw = _git_bytes(
        repo_root,
        "diff",
        "--cached",
        "--name-only",
        "--diff-filter=ACMRT",
        "-z",
    )
    changed_paths: list[str] = []
    for raw_path in changed_raw.split(b"\x00"):
        if not raw_path:
            continue
        try:
            changed_paths.append(raw_path.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise EvidenceError("staged path is not UTF-8") from error
    for path in changed_paths:
        _scan_staged_blob(path.encode("utf-8"), name="<staged-path>")
        path_parts = Path(path).parts
        if (
            Path(path).is_absolute()
            or not path_parts
            or any(part in {"", ".", ".."} for part in path_parts)
            or "\\" in path
        ):
            raise EvidenceError("staged path is unsafe")
        entry = index.get(path)
        if entry is None:
            raise EvidenceError(f"staged path is absent from the index: {path}")
        mode, object_id, stage = entry
        if mode != "100644" or stage != "0":
            raise EvidenceError(f"staged path has an unsafe mode: {path}")
        blob = _git_bytes(repo_root, "cat-file", "blob", object_id)
        _scan_staged_blob(blob, name=path)

    evidence_prefix = evidence_relative.as_posix()
    evidence_entries = {
        path: entry
        for path, entry in index.items()
        if path == evidence_prefix or path.startswith(f"{evidence_prefix}/")
    }
    if not evidence_entries:
        raise EvidenceError("no evidence tree is staged")
    with tempfile.TemporaryDirectory(prefix="sparkbench-staged-evidence-") as temporary:
        temporary_root = Path(temporary)
        materialized_root = temporary_root / evidence_relative
        for path, (mode, object_id, stage) in evidence_entries.items():
            relative = Path(path).relative_to(evidence_relative)
            if (
                mode != "100644"
                or stage != "0"
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise EvidenceError(f"unsafe staged evidence entry: {path}")
            blob = _git_bytes(repo_root, "cat-file", "blob", object_id)
            if len(blob) > MAX_OUTPUT_FILE_BYTES:
                raise EvidenceError(f"staged evidence file exceeds size limit: {path}")
            target = materialized_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
        verification = verify_evidence(materialized_root)

    tree_after = _git_bytes(repo_root, "write-tree").strip()
    if tree_before != tree_after:
        raise EvidenceError("Git index changed during evidence verification")
    return {
        **verification,
        "staged_file_count": len(changed_paths),
        "status": "staged_verified",
        "tree_sha256": hashlib.sha256(tree_after).hexdigest(),
    }


_EVIDENCE_README = """# Sanitized Benchmark Evidence

This directory is generated by `python3 sparkbench.py export-evidence`.

Raw `results/` remain ignored because journals and summaries can contain prompts,
completions, reasoning, tool arguments, request identifiers, commands, host paths,
and free-form errors.  The tracked corpus contains only allowlisted scalar samples,
aggregate metrics, validation flags, speculative-decoding counters, telemetry, and
public artifact/runtime identities.  It contains no raw request or response text,
media, token sequences, credentials, API keys, commands, environment variables, or
absolute paths.

`index.json` accounts for every discovered run, declared nested autoresearch cell,
matrix, custom campaign, and standalone battery.  Aborted and nonterminal attempts
remain explicitly classified; they are not promoted to completed measurements.
`checksums.json` covers only the sanitized files in this directory.
"""


def _assert_source_tree(root: Path) -> tuple[int, int]:
    if root.is_symlink() or not root.is_dir():
        raise EvidenceError("results root must be a real directory")
    count = 0
    size = 0
    for directory, directories, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directories:
            path = base / name
            if path.is_symlink():
                raise EvidenceError(f"results contains a directory symlink: {name}")
        for name in files:
            path = base / name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise EvidenceError(f"results contains a special or linked file: {name}")
            count += 1
            size += metadata.st_size
    return count, size


def _validate_runtime_overlay_tree(
    results_root: Path,
    run_dirs: Sequence[Path],
    loop_dirs: Sequence[Path] = (),
) -> bool:
    """Bind the exact runtime-overlay tree to declarations in frozen plans."""

    declared: dict[Path, tuple[str, str, str]] = {}

    def add_model(model: dict[str, Any]) -> None:
        for relative_path, basename, digest, container_path in (
            _sglang_source_overlay_declarations(model)
        ):
            identity = (basename, digest, container_path)
            previous = declared.get(relative_path)
            if previous is not None and previous != identity:
                raise EvidenceError(
                    "SGLang source overlay declarations conflict across frozen plans"
                )
            declared[relative_path] = identity

    for run_dir in run_dirs:
        plan = _load_json(run_dir / "plan.json", results_root)
        if not isinstance(plan, dict):
            raise EvidenceError("plan must be an object")
        model = plan.get("model")
        if not isinstance(model, dict):
            continue
        add_model(model)

    for loop_dir in loop_dirs:
        plan = _load_json(loop_dir / "plan.json", results_root)
        if not isinstance(plan, dict):
            raise EvidenceError("loop plan must be an object")
        models = plan.get("models")
        if not isinstance(models, dict) or not models:
            raise EvidenceError("loop plan models are missing")
        for profile_id in sorted(models):
            _safe_id(profile_id, name="loop plan model ID")
            model = models[profile_id]
            if not isinstance(model, dict):
                raise EvidenceError("loop plan model must be an object")
            add_model(model)

    overlay_root = results_root / "runtime-overlays"
    if not overlay_root.exists():
        if declared:
            raise EvidenceError("a frozen SGLang source overlay is missing")
        return False
    metadata = overlay_root.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise EvidenceError("runtime-overlays must be a real directory")
    if not declared:
        raise EvidenceError("runtime-overlays contains no plan-declared files")

    actual_files: set[Path] = set()
    actual_directories: set[Path] = {Path("runtime-overlays")}
    for directory, directories, files in os.walk(overlay_root, followlinks=False):
        base = Path(directory)
        for name in directories:
            path = base / name
            child_metadata = path.lstat()
            if not stat.S_ISDIR(child_metadata.st_mode) or stat.S_ISLNK(
                child_metadata.st_mode
            ):
                raise EvidenceError("runtime-overlays contains an unsafe directory")
            actual_directories.add(path.relative_to(results_root))
        for name in files:
            path = base / name
            file_metadata = path.lstat()
            if (
                not stat.S_ISREG(file_metadata.st_mode)
                or file_metadata.st_nlink != 1
            ):
                raise EvidenceError("runtime-overlays contains an unsafe file")
            actual_files.add(path.relative_to(results_root))

    expected_files = set(declared)
    if actual_files != expected_files:
        undeclared = sorted(
            path.as_posix() for path in actual_files - expected_files
        )
        missing = sorted(path.as_posix() for path in expected_files - actual_files)
        raise EvidenceError(
            "runtime-overlays file set changed; "
            f"undeclared={undeclared!r}, missing={missing!r}"
        )
    expected_directories = {Path("runtime-overlays")}
    for relative_path in expected_files:
        parent = relative_path.parent
        while parent != Path("."):
            expected_directories.add(parent)
            if parent == Path("runtime-overlays"):
                break
            parent = parent.parent
    if actual_directories != expected_directories:
        raise EvidenceError("runtime-overlays directory topology changed")

    for relative_path, (_, expected_digest, _) in sorted(declared.items()):
        source = _secure_read(
            results_root / relative_path,
            results_root,
            maximum=MAX_SOURCE_JSON_BYTES,
        ).encode("utf-8")
        if _hash_bytes(source) != expected_digest:
            raise EvidenceError(
                f"runtime overlay digest mismatch: {relative_path.name}"
            )
    return True


def _grouped_run_dirs(results_root: Path) -> dict[str, set[Path]]:
    """Validate and enumerate the narrowly allowlisted grouped-run layouts."""

    grouped: dict[str, set[Path]] = {}
    for root in sorted(results_root.iterdir()):
        if not _is_grouped_run_root(root.name):
            continue
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise EvidenceError("grouped results root must be a real directory")
        children = sorted(root.iterdir())
        if not children:
            raise EvidenceError("grouped results root must contain at least one run")
        runs: set[Path] = set()
        for child in children:
            child_metadata = child.lstat()
            if not stat.S_ISDIR(child_metadata.st_mode) or stat.S_ISLNK(
                child_metadata.st_mode
            ):
                raise EvidenceError(
                    "grouped results root must contain only direct run directories"
                )
            _date_from_run_id(child.name)
            plan_path = child / "plan.json"
            if not plan_path.is_file():
                raise EvidenceError("grouped run directory is missing plan.json")
            runs.add(child)
        grouped[root.name] = runs
    return grouped


def _loop_campaign_dirs(results_root: Path) -> list[tuple[str, Path]]:
    """Enumerate only direct loop campaign directories under exact source roots."""

    campaigns: list[tuple[str, Path]] = []
    for root_name in LOOP_RESULT_ROOTS:
        root = results_root / root_name
        if not root.exists():
            continue
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise EvidenceError("loop results root must be a real directory")
        children = sorted(root.iterdir())
        if not children:
            raise EvidenceError("loop results root must contain at least one plan")
        for child in children:
            child_metadata = child.lstat()
            if not stat.S_ISDIR(child_metadata.st_mode) or stat.S_ISLNK(
                child_metadata.st_mode
            ):
                raise EvidenceError(
                    "loop results root must contain only direct campaign directories"
                )
            _date_from_run_id(child.name)
            if not (child / "plan.json").is_file():
                raise EvidenceError("loop campaign directory is missing plan.json")
            campaigns.append((root_name, child))
    identities = [path.name for _, path in campaigns]
    if len(identities) != len(set(identities)):
        raise EvidenceError("loop campaign run identifiers are duplicated")
    return campaigns


def _autoresearch_content_hash(value: Any, *, length: int = 64) -> str:
    return hashlib.sha256(_canonical(value).rstrip(b"\n")).hexdigest()[:length]


def _autoresearch_id(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _AUTORESEARCH_ID_RE.fullmatch(value) is None:
        raise EvidenceError(f"unsafe autoresearch {name}")
    return value


def _autoresearch_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _HEX_RE.fullmatch(value) is None:
        raise EvidenceError(f"invalid autoresearch {name}")
    return value


def _autoresearch_real_directory(path: Path, *, name: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise EvidenceError(f"autoresearch {name} is missing") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise EvidenceError(f"autoresearch {name} must be a real directory")
    return path.resolve(strict=True)


def _autoresearch_created_stamp(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 64:
        raise EvidenceError("autoresearch campaign creation time is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvidenceError("autoresearch campaign creation time is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvidenceError("autoresearch campaign creation time must be timezone-aware")
    return parsed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _autoresearch_published_run_id(
    *,
    campaign_id: str,
    cell_id: str,
    ordinal: int,
    created_at: str,
) -> str:
    stamp = _autoresearch_created_stamp(created_at)
    identity = {
        "campaign_id": campaign_id,
        "cell_id": cell_id,
        "created_at": created_at,
        "ordinal": ordinal,
    }
    identity_digest = _autoresearch_content_hash(identity, length=16)
    run_id = (
        f"{stamp}-autoresearch-{campaign_id[:64]}-{ordinal:02d}-"
        f"{cell_id[:64]}-{identity_digest}"
    )
    _date_from_run_id(run_id)
    checked = _safe_id(run_id, name="autoresearch published run ID")
    assert isinstance(checked, str)
    return checked


def _autoresearch_run_dirs(results_root: Path) -> list[tuple[Path, str]]:
    """Validate frozen campaigns and enumerate only their declared cell plans."""

    autoresearch_root = results_root / AUTORESEARCH_RESULT_ROOT
    if not autoresearch_root.exists():
        return []
    autoresearch_root = _autoresearch_real_directory(
        autoresearch_root, name="results root"
    )
    campaign_dirs = sorted(autoresearch_root.iterdir())
    if not campaign_dirs:
        raise EvidenceError("autoresearch results root must contain a campaign")

    published: list[tuple[Path, str]] = []
    for campaign_dir in campaign_dirs:
        if (
            _AUTORESEARCH_PATH_COMPONENT_RE.fullmatch(campaign_dir.name) is None
        ):
            raise EvidenceError("unsafe autoresearch campaign directory name")
        campaign_dir = _autoresearch_real_directory(
            campaign_dir, name="campaign directory"
        )
        campaign_entries = {entry.name: entry for entry in campaign_dir.iterdir()}
        allowed_campaign_files = {
            ".autoresearch.lock",
            "calibration.json",
            "campaign.json",
            "events.jsonl",
            "summary.json",
        }
        if "campaign.json" not in campaign_entries or "cells" not in campaign_entries:
            raise EvidenceError("autoresearch campaign topology is incomplete")
        unknown_campaign_entries = set(campaign_entries) - (
            allowed_campaign_files | {"cells"}
        )
        if unknown_campaign_entries:
            raise EvidenceError("autoresearch campaign topology contains unknown entries")
        for name, entry in campaign_entries.items():
            metadata = entry.lstat()
            if name == "cells":
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(
                    metadata.st_mode
                ):
                    raise EvidenceError("autoresearch cells must be a real directory")
            elif stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(
                metadata.st_mode
            ):
                raise EvidenceError("autoresearch campaign control must be a real file")

        campaign = _load_json(campaign_dir / "campaign.json", results_root)
        if not isinstance(campaign, dict) or set(campaign) != set(
            _AUTORESEARCH_CAMPAIGN_FIELDS
        ):
            raise EvidenceError("autoresearch campaign schema changed")
        if campaign.get("schema_version") != AUTORESEARCH_CAMPAIGN_SCHEMA_VERSION:
            raise EvidenceError("unsupported autoresearch campaign schema")
        campaign_integrity = _autoresearch_sha256(
            campaign.get("integrity_hash"), name="campaign integrity hash"
        )
        campaign_payload = {
            key: value for key, value in campaign.items() if key != "integrity_hash"
        }
        if _autoresearch_content_hash(campaign_payload) != campaign_integrity:
            raise EvidenceError("autoresearch campaign integrity hash does not match")
        if campaign.get("execution_started") is not False:
            raise EvidenceError("autoresearch frozen campaign execution state changed")
        _autoresearch_sha256(
            campaign.get("harness_tree_sha256"), name="harness tree hash"
        )
        harness_file_count = campaign.get("harness_file_count")
        if (
            isinstance(harness_file_count, bool)
            or not isinstance(harness_file_count, int)
            or harness_file_count <= 0
        ):
            raise EvidenceError("autoresearch harness file count is invalid")
        created_at = campaign.get("created_at")
        _autoresearch_created_stamp(created_at)
        assert isinstance(created_at, str)

        preview = campaign.get("preview")
        if not isinstance(preview, dict) or set(preview) != set(
            _AUTORESEARCH_PREVIEW_FIELDS
        ):
            raise EvidenceError("autoresearch campaign preview schema changed")
        preview_digest = _autoresearch_sha256(
            campaign.get("preview_digest"), name="preview digest"
        )
        if _autoresearch_content_hash(preview) != preview_digest:
            raise EvidenceError("autoresearch campaign preview digest does not match")
        if (
            preview.get("schema_version") != 1
            or preview.get("execution_started") is not False
            or preview.get("planned_cell_count") != AUTORESEARCH_CELL_COUNT
        ):
            raise EvidenceError("autoresearch campaign preview topology changed")
        campaign_id = _autoresearch_id(
            preview.get("campaign_id"), name="campaign ID"
        )
        baseline_id = _autoresearch_id(
            preview.get("baseline_id"), name="baseline ID"
        )
        suite_id = _autoresearch_id(preview.get("suite_id"), name="suite ID")
        policy = preview.get("policy")
        policy_digest = _autoresearch_sha256(
            preview.get("policy_digest"), name="policy digest"
        )
        if (
            not isinstance(policy, dict)
            or _autoresearch_content_hash(policy) != policy_digest
        ):
            raise EvidenceError("autoresearch campaign policy digest does not match")
        proposals = preview.get("proposals")
        if not isinstance(proposals, list):
            raise EvidenceError("autoresearch campaign proposals must be an array")
        proposal_ids: list[str] = []
        for proposal in proposals:
            if not isinstance(proposal, dict) or set(proposal) != {
                "candidate_id",
                "axis",
                "delta",
                "delta_digest",
            }:
                raise EvidenceError("autoresearch campaign proposal schema changed")
            proposal_ids.append(
                _autoresearch_id(proposal.get("candidate_id"), name="candidate ID")
            )
            _autoresearch_id(proposal.get("axis"), name="candidate axis")
            delta = proposal.get("delta")
            delta_digest = _autoresearch_sha256(
                proposal.get("delta_digest"), name="candidate delta digest"
            )
            if (
                not isinstance(delta, dict)
                or _autoresearch_content_hash(delta) != delta_digest
            ):
                raise EvidenceError("autoresearch candidate delta digest does not match")
        if len(proposal_ids) != len(set(proposal_ids)):
            raise EvidenceError("autoresearch candidate identifiers are duplicated")

        expected_cells: list[dict[str, str]] = [
            {
                "arm": "control_a",
                "candidate_id": "control",
                "cell_id": "calibration-control-a",
                "profile_id": baseline_id,
                "stage": "calibration",
            },
            {
                "arm": "control_b",
                "candidate_id": "control",
                "cell_id": "calibration-control-b",
                "profile_id": baseline_id,
                "stage": "calibration",
            },
        ]
        for candidate_id in proposal_ids:
            expected_cells.extend(
                (
                    {
                        "arm": "champion",
                        "candidate_id": candidate_id,
                        "cell_id": f"{candidate_id}-screen-champion",
                        "profile_id": baseline_id,
                        "stage": "screen",
                    },
                    {
                        "arm": "candidate",
                        "candidate_id": candidate_id,
                        "cell_id": f"{candidate_id}-screen-candidate",
                        "profile_id": candidate_id,
                        "stage": "screen",
                    },
                    {
                        "arm": "candidate",
                        "candidate_id": candidate_id,
                        "cell_id": f"{candidate_id}-confirmation-candidate",
                        "profile_id": candidate_id,
                        "stage": "confirmation",
                    },
                    {
                        "arm": "champion",
                        "candidate_id": candidate_id,
                        "cell_id": f"{candidate_id}-confirmation-champion",
                        "profile_id": baseline_id,
                        "stage": "confirmation",
                    },
                )
            )
        if len(expected_cells) != AUTORESEARCH_CELL_COUNT:
            raise EvidenceError("autoresearch campaign proposal topology changed")

        cells = campaign.get("cells")
        if not isinstance(cells, list) or len(cells) != AUTORESEARCH_CELL_COUNT:
            raise EvidenceError("autoresearch campaign must declare fourteen cells")
        cells_root = _autoresearch_real_directory(
            campaign_dir / "cells", name="cells directory"
        )
        expected_cell_directories: set[str] = set()
        declared_plan_paths: set[Path] = set()
        run_dirs: set[Path] = set()
        cell_ids: list[str] = []
        run_nonces: list[str] = []
        for ordinal, cell in enumerate(cells, start=1):
            if not isinstance(cell, dict) or set(cell) != set(
                _AUTORESEARCH_CELL_FIELDS
            ):
                raise EvidenceError("autoresearch campaign cell schema changed")
            if cell.get("ordinal") != ordinal:
                raise EvidenceError("autoresearch cell ordinals are not contiguous")
            cell_id = _autoresearch_id(cell.get("cell_id"), name="cell ID")
            cell_ids.append(cell_id)
            for field in ("stage", "candidate_id", "arm", "profile_id"):
                _autoresearch_id(cell.get(field), name=f"cell {field}")
            expected_cell = expected_cells[ordinal - 1]
            if any(cell.get(key) != value for key, value in expected_cell.items()):
                raise EvidenceError(
                    "autoresearch cell schedule or profile binding changed"
                )
            run_nonce = cell.get("run_nonce")
            if (
                not isinstance(run_nonce, str)
                or re.fullmatch(r"[0-9a-f]{32}", run_nonce) is None
            ):
                raise EvidenceError("autoresearch cell ownership nonce is invalid")
            run_nonces.append(run_nonce)
            plan_fingerprint = cell.get("plan_fingerprint")
            if (
                not isinstance(plan_fingerprint, str)
                or re.fullmatch(r"[0-9a-f]{16}", plan_fingerprint) is None
            ):
                raise EvidenceError("autoresearch cell plan fingerprint is invalid")
            plan_integrity = _autoresearch_sha256(
                cell.get("plan_integrity_hash"), name="cell plan integrity hash"
            )

            raw_run_dir = cell.get("run_dir")
            if not isinstance(raw_run_dir, str) or len(raw_run_dir) > 768:
                raise EvidenceError("autoresearch cell run directory is unsafe")
            relative_run_dir = PurePosixPath(raw_run_dir)
            expected_cell_directory = f"{ordinal:02d}-{cell_id}"
            if (
                relative_run_dir.is_absolute()
                or relative_run_dir.as_posix() != raw_run_dir
                or len(relative_run_dir.parts) != 3
                or relative_run_dir.parts[:2]
                != ("cells", expected_cell_directory)
                or any(part in {"", ".", ".."} for part in relative_run_dir.parts)
            ):
                raise EvidenceError("autoresearch cell run directory topology changed")
            raw_run_id = relative_run_dir.parts[2]
            _date_from_run_id(raw_run_id)
            expected_cell_directories.add(expected_cell_directory)
            run_dir = campaign_dir.joinpath(*relative_run_dir.parts)
            resolved_run_dir = _autoresearch_real_directory(
                run_dir, name="cell run directory"
            )
            if resolved_run_dir != run_dir or resolved_run_dir in run_dirs:
                raise EvidenceError("autoresearch cell run directories are duplicated")
            try:
                resolved_run_dir.relative_to(campaign_dir)
            except ValueError as error:
                raise EvidenceError(
                    "autoresearch cell run directory escapes its campaign"
                ) from error
            run_dirs.add(resolved_run_dir)
            plan_path = resolved_run_dir / "plan.json"
            declared_plan_paths.add(plan_path)
            plan = _load_json(plan_path, results_root)
            if not isinstance(plan, dict) or plan.get("schema_version") != 2:
                raise EvidenceError("autoresearch cell plan schema changed")
            if (
                plan.get("fingerprint") != plan_fingerprint
                or plan.get("integrity_hash") != plan_integrity
            ):
                raise EvidenceError("autoresearch cell plan binding changed")
            if plan.get("run_nonce") != run_nonce:
                raise EvidenceError("autoresearch cell ownership nonce changed")
            plan_payload = {
                key: value for key, value in plan.items() if key != "integrity_hash"
            }
            if _autoresearch_content_hash(plan_payload) != plan_integrity:
                raise EvidenceError("autoresearch cell plan integrity hash does not match")
            model = plan.get("model")
            suite = plan.get("suite")
            resolved = plan.get("resolved")
            if (
                not isinstance(model, dict)
                or not isinstance(suite, dict)
                or not isinstance(resolved, dict)
                or model.get("id") != cell.get("profile_id")
                or suite.get("id") != suite_id
            ):
                raise EvidenceError("autoresearch cell profile or suite binding changed")
            if suite_id == AUTORESEARCH_SUITE_ID:
                _project_autoresearch_suite(suite, source_model=model)
            cases = suite.get("cases")
            if not isinstance(cases, list) or any(
                not isinstance(case, dict) for case in cases
            ):
                raise EvidenceError("autoresearch cell plan cases are invalid")
            suite_without_case_ids = {
                **suite,
                "cases": [
                    {key: value for key, value in case.items() if key != "case_id"}
                    for case in cases
                ],
            }
            expected_fingerprint = _autoresearch_content_hash(
                {
                    "model": model,
                    "suite": suite_without_case_ids,
                    "resolved": resolved,
                },
                length=16,
            )
            if expected_fingerprint != plan_fingerprint:
                raise EvidenceError("autoresearch cell plan fingerprint does not match")

            published_run_id = _autoresearch_published_run_id(
                campaign_id=campaign_id,
                cell_id=cell_id,
                ordinal=ordinal,
                created_at=created_at,
            )
            published.append((resolved_run_dir, published_run_id))

        if len(cell_ids) != len(set(cell_ids)):
            raise EvidenceError("autoresearch cell identifiers are duplicated")
        if len(run_nonces) != len(set(run_nonces)):
            raise EvidenceError("autoresearch cell ownership nonces are duplicated")
        actual_cell_entries = {entry.name: entry for entry in cells_root.iterdir()}
        if set(actual_cell_entries) != expected_cell_directories:
            raise EvidenceError("autoresearch cell directory topology changed")
        for cell_directory, entry in actual_cell_entries.items():
            resolved_cell = _autoresearch_real_directory(
                entry, name="cell directory"
            )
            expected_runs = {
                run_dir.name for run_dir in run_dirs if run_dir.parent == resolved_cell
            }
            actual_runs = {child.name: child for child in resolved_cell.iterdir()}
            if set(actual_runs) != expected_runs or len(expected_runs) != 1:
                raise EvidenceError("autoresearch cell run topology changed")
            for child in actual_runs.values():
                _autoresearch_real_directory(child, name="cell run directory")
        actual_plan_paths = set(campaign_dir.rglob("plan.json"))
        if actual_plan_paths != declared_plan_paths:
            raise EvidenceError("autoresearch campaign contains an undeclared plan")

    published_ids = [run_id for _, run_id in published]
    if len(published_ids) != len(set(published_ids)):
        raise EvidenceError("autoresearch published run identifiers are duplicated")
    return published


def _export_evidence_locked(
    *,
    results_root: Path,
    output_root: Path,
    harbor_results: Sequence[Path] = (),
    replace: bool = False,
) -> dict[str, Any]:
    if len(harbor_results) not in {0, HARBOR_REPLICATE_COUNT}:
        raise EvidenceError("Harbor evidence requires zero or exactly two result files")
    results_root = results_root.resolve(strict=True)
    source_file_count, source_size_bytes = _assert_source_tree(results_root)
    if output_root.name in {"", ".", ".."} or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", output_root.name
    ):
        raise EvidenceError("unsafe evidence output name")
    output_parent = output_root.parent.resolve(strict=True)
    output_candidate = output_parent / output_root.name
    if output_candidate.is_symlink():
        raise EvidenceError("evidence output must not be a symlink")
    output_target = output_candidate.resolve(strict=False)
    if (
        output_target == results_root
        or output_target in results_root.parents
        or results_root in output_target.parents
    ):
        raise EvidenceError("unsafe evidence output target")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_target.name}.tmp-", dir=output_parent)
    )
    try:
        for category in ("campaigns", "matrices", "runs", "standalone"):
            (temporary / category).mkdir()
        runs: list[dict[str, Any]] = []
        loop_campaign_dirs = _loop_campaign_dirs(results_root)
        loop_campaign_paths = {path for _, path in loop_campaign_dirs}
        autoresearch_runs = _autoresearch_run_dirs(results_root)
        autoresearch_root = results_root / AUTORESEARCH_RESULT_ROOT
        run_dirs = sorted(
            {
                path.parent
                for path in results_root.rglob("plan.json")
                if not any(
                    ancestor in loop_campaign_paths for ancestor in path.parents
                )
                and autoresearch_root not in path.parents
            }
        )
        all_run_dirs = [*run_dirs, *(path for path, _ in autoresearch_runs)]
        has_runtime_overlays = _validate_runtime_overlay_tree(
            results_root,
            all_run_dirs,
            [path for _, path in loop_campaign_dirs],
        )
        grouped_runs = _grouped_run_dirs(results_root)
        recognized_top = {
            ".sparkbench.lock",
            "content-battery-dspark-sglang-20260817.json",
            "matrices",
            "moe-bandwidth-20260817T1539Z",
            "ninfer-experimental-sm121a-20260817T181134Z",
            "ninfer-gb10-20260817",
            "ninfer-qwen38-nvfp4-sm121a-20260817T200147Z",
            "upstream-bench-matrix-dspark-sglang-20260817.json",
        }
        if has_runtime_overlays:
            recognized_top.add("runtime-overlays")
        if autoresearch_runs:
            recognized_top.add(AUTORESEARCH_RESULT_ROOT)
        published_run_ids = [run_dir.name for run_dir in run_dirs] + [
            run_id for _, run_id in autoresearch_runs
        ]
        if len(published_run_ids) != len(set(published_run_ids)):
            raise EvidenceError("published run identifiers are duplicated")
        for run_dir in run_dirs:
            relative_run = run_dir.relative_to(results_root)
            if len(relative_run.parts) == 1:
                recognized_top.add(relative_run.parts[0])
            elif (
                len(relative_run.parts) == 2
                and relative_run.parts[0] in grouped_runs
            ):
                if run_dir not in grouped_runs[relative_run.parts[0]]:
                    raise EvidenceError(
                        "grouped run directory does not match the declared topology"
                    )
                recognized_top.add(relative_run.parts[0])
            elif relative_run.parts[0] != "matrices" or len(relative_run.parts) != 3:
                raise EvidenceError(f"run directory has an unknown layout: {run_dir.name}")
            matrix_id = (
                run_dir.parent.name
                if run_dir.parent.parent == results_root / "matrices"
                else None
            )
            runs.append(_export_run(run_dir, results_root, temporary, matrix_id))
        for run_dir, published_run_id in autoresearch_runs:
            runs.append(
                _export_run(
                    run_dir,
                    results_root,
                    temporary,
                    None,
                    published_run_id=published_run_id,
                )
            )
        recognized_top.update(root_name for root_name, _ in loop_campaign_dirs)
        actual_top = {path.name for path in results_root.iterdir()}
        if actual_top != recognized_top:
            unknown = sorted(actual_top - recognized_top)
            missing = sorted(recognized_top - actual_top)
            raise EvidenceError(
                f"results top-level entries changed; unknown={unknown!r}, missing={missing!r}"
            )

        matrices = [
            _export_matrix(path, results_root, temporary)
            for path in sorted((results_root / "matrices").glob("*/matrix.json"))
        ]
        campaign_names = (
            "moe-bandwidth-20260817T1539Z",
            "ninfer-experimental-sm121a-20260817T181134Z",
            "ninfer-gb10-20260817",
            "ninfer-qwen38-nvfp4-sm121a-20260817T200147Z",
        )
        campaigns = [
            _export_campaign(results_root / name, results_root, temporary)
            for name in campaign_names
        ]
        campaigns.extend(
            _export_loop_campaign(
                campaign,
                results_root,
                temporary,
                source_group=root_name,
            )
            for root_name, campaign in loop_campaign_dirs
        )
        existing_harbor = output_target / "campaigns" / HARBOR_CAMPAIGN_ID
        if harbor_results:
            campaigns.append(_export_harbor_campaign(harbor_results, temporary))
        elif existing_harbor.is_dir():
            campaigns.append(
                _reuse_existing_harbor_campaign(output_target, temporary)
            )
        campaigns.sort(key=lambda campaign: campaign["campaign_id"])
        standalone = [
            _export_content_battery(
                results_root / "content-battery-dspark-sglang-20260817.json",
                results_root,
                temporary,
            ),
            _export_upstream_battery(
                results_root / "upstream-bench-matrix-dspark-sglang-20260817.json",
                results_root,
                temporary,
            ),
        ]
        status_counts = Counter(run["status"] for run in runs)
        index = {
            "campaign_count": len(campaigns),
            "campaigns": campaigns,
            "matrix_count": len(matrices),
            "matrices": matrices,
            "run_count": len(runs),
            "run_status_counts": dict(sorted(status_counts.items())),
            "runs": runs,
            "sanitization_policy": SANITIZATION_POLICY,
            "schema_version": SCHEMA_VERSION,
            "source_control_files_ignored": 1,
            "source_file_count": source_file_count,
            "source_size_bytes": source_size_bytes,
            "standalone": standalone,
            "standalone_count": len(standalone),
        }
        (temporary / "index.json").write_bytes(_canonical(index))
        (temporary / "README.md").write_text(_EVIDENCE_README, encoding="utf-8")
        checksum_files = sorted(
            path for path in temporary.rglob("*") if path.is_file()
        )
        checksums = {
            str(path.relative_to(temporary)): _hash_bytes(path.read_bytes())
            for path in checksum_files
        }
        (temporary / "checksums.json").write_bytes(
            _canonical({"files": checksums, "schema_version": SCHEMA_VERSION})
        )
        verification = verify_evidence(temporary)

        if output_target.exists():
            if not output_target.is_dir():
                raise EvidenceError("existing evidence target is not a directory")
            verify_evidence(output_target)
            existing_files = {
                str(path.relative_to(output_target)): _secure_read(
                    path, output_target, maximum=MAX_OUTPUT_FILE_BYTES
                ).encode()
                for path in output_target.rglob("*")
                if path.is_file()
            }
            new_files = {
                str(path.relative_to(temporary)): path.read_bytes()
                for path in temporary.rglob("*")
                if path.is_file()
            }
            if existing_files == new_files:
                shutil.rmtree(temporary)
                return {**verification, "changed": False, "runs": len(runs)}
            if not replace:
                raise EvidenceError("evidence differs; rerun with --replace")
            backup = output_parent / f".{output_target.name}.old"
            if backup.exists():
                raise EvidenceError("stale evidence backup exists")
            os.replace(output_target, backup)
            try:
                os.replace(temporary, output_target)
            except BaseException:
                os.replace(backup, output_target)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temporary, output_target)
        return {**verification, "changed": True, "runs": len(runs)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def export_evidence(
    *,
    results_root: Path,
    output_root: Path,
    harbor_results: Sequence[Path] = (),
    replace: bool = False,
) -> dict[str, Any]:
    if len(harbor_results) not in {0, HARBOR_REPLICATE_COUNT}:
        raise EvidenceError("Harbor evidence requires zero or exactly two result files")
    results_root = results_root.resolve(strict=True)
    lock_path = results_root / ".sparkbench.lock"
    descriptor = os.open(lock_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise EvidenceError("benchmark lock must be a single-link regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise EvidenceError(
                "Another SparkBench run holds results/.sparkbench.lock"
            ) from error
        return _export_evidence_locked(
            results_root=results_root,
            output_root=output_root,
            harbor_results=tuple(Path(path) for path in harbor_results),
            replace=replace,
        )
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
