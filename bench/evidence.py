"""Deterministic, privacy-safe publication of ignored benchmark results.

Raw result bundles deliberately contain prompts, completions, request identifiers,
commands, host paths, and free-form errors.  This module never copies those
objects.  It constructs a small, typed scalar evidence corpus from explicit
allowlists and validates the materialized output before publication.
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import statistics
import subprocess
import tempfile
from typing import Any, Sequence
import unicodedata

from .manifest import KNOWN_AGENTIC_CASE_IDS
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


class EvidenceError(RuntimeError):
    """Raised when source or generated evidence fails closed validation."""


_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+@/-]{0,255}\Z")
_SAFE_TEXT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._:+@/(),=+-]{0,255}\Z")
_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")
_REVISION_RE = re.compile(r"[0-9a-f]{7,64}\Z")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}\Z")
_RUN_ID_RE = re.compile(r"20[0-9]{6}T[0-9]{6,12}Z-[A-Za-z0-9_.-]+\Z")
_GROUPED_RUN_ROOT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"qwen36-core-(20[0-9]{6}T[0-9]{6})\Z"), "%Y%m%dT%H%M%S"),
    (re.compile(r"reasoning-(20[0-9]{6})\Z"), "%Y%m%d"),
)

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
_CASE_OBJECT_FIELDS = {"prefix_cache", "quality_accuracy_by_category", "telemetry"}
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
    if suite.get("id") == "agentic-tools" or (
        isinstance(raw_cases, list)
        and any(
            isinstance(case, dict) and case.get("kind") == "agentic"
            for case in raw_cases
        )
    ):
        return _project_agentic_suite(suite)
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
            if kind == "agentic" and event.get("repetition") is None:
                raise EvidenceError("agentic request repetition is missing")
            if event.get("repetition") is not None:
                repetition = event["repetition"]
                if kind == "agentic":
                    if (
                        isinstance(repetition, bool)
                        or not isinstance(repetition, int)
                        or not 0 <= repetition <= 2
                    ):
                        raise EvidenceError(
                            "agentic request repetition must be an integer from 0 to 2"
                        )
                    if repetition != result["variant"]:
                        raise EvidenceError(
                            "agentic request repetition and variant disagree"
                        )
                    if case_id.split("--", 1)[0] != result["scenario_id"]:
                        raise EvidenceError(
                            "agentic request case and scenario identifiers disagree"
                        )
                sample["repetition"] = _finite(
                    repetition, name="request.repetition"
                )
            if event.get("burst_elapsed_s") is not None:
                sample["burst_elapsed_s"] = _finite(
                    event["burst_elapsed_s"], name="request.burst_elapsed_s"
                )
            validation = event.get("validation")
            if kind == "agentic" and validation is None:
                raise EvidenceError("agentic request validation is missing")
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
                if kind == "agentic" and passed is not result["passed"]:
                    raise EvidenceError(
                        "agentic request validation and result pass flags disagree"
                    )
                if validation.get("quality_category") is not None:
                    sample["quality_category"] = _safe_id(
                        validation["quality_category"],
                        name="validation.quality_category",
                    )
            if kind == "agentic":
                _validate_projected_agentic_sample(sample)
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
    if isinstance(suite, dict) and suite.get("id") == "agentic-tools":
        validated_suite = _project_agentic_suite(suite)
        planned_cases = {
            str(case["case_id"]): case for case in validated_suite["cases"]
        }
    cases = summary.get("cases")
    agentic_samples = [
        sample for sample in requests if sample.get("kind") == "agentic"
    ]
    if not isinstance(cases, list):
        if agentic_samples or planned_cases:
            raise EvidenceError("agentic evidence requires summary cases")
        return
    agentic_cases = [
        case for case in cases if isinstance(case, dict) and case.get("kind") == "agentic"
    ]
    if (agentic_samples or agentic_cases) and not planned_cases:
        raise EvidenceError("agentic evidence requires the exact agentic-tools suite")
    if not agentic_samples and not agentic_cases and not planned_cases:
        return
    if len(agentic_cases) != len(cases):
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
            accounted.extend(values)
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


def _project_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {}
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
) -> dict[str, Any]:
    run_id = run_dir.name
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
    if cache_protocol:
        # Materialize an exact protocol manifest rather than publishing the
        # broader serving manifest.  This is also the export-side counterpart
        # to the verifier's strict re-projection below.
        manifest = _project_prefix_cache_manifest(manifest, source=True)
    requests = _project_requests(events, summary, evidence_kind=kind)
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
    projected_summary = _project_summary(summary)
    _validate_agentic_aggregates(
        requests,
        projected_summary,
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
    if _verify_bundle(directory, root) != _sha256(
        entry["bundle_sha256"], name="run bundle"
    ):
        raise EvidenceError(f"run bundle digest mismatch: {run_id}")
    expected_manifest = f"runs/{run_id}/manifest.json"
    if entry["file"] != expected_manifest:
        raise EvidenceError(f"run manifest pointer mismatch: {run_id}")
    manifest = _load_json(directory / "manifest.json", root)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError(f"run manifest schema mismatch: {run_id}")
    manifest_model = manifest.get("model")
    manifest_suite = manifest.get("suite")
    is_prefix_cache_manifest = (
        isinstance(manifest_model, dict)
        and manifest_model.get("prefix_cache_mode") is not None
    ) or (
        isinstance(manifest_suite, dict)
        and manifest_suite.get("id") == PREFIX_CACHE_SUITE_ID
    )
    if is_prefix_cache_manifest:
        # Cache bundles are intentionally a complete, exact outer document.
        # Do not rely on checksums alone: an attacker can refresh checksums
        # after adding a generic manifest field such as trace text.
        _project_prefix_cache_manifest(manifest)
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
    manifest_cases = suite.get("cases") if isinstance(suite, dict) else None
    if isinstance(suite, dict) and (
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
    summary = _load_json(directory / "summary.json", root)
    summary = _expect_object_keys(
        summary, {"aggregates", "schema_version"}, name="run summary"
    )
    if summary["schema_version"] != SCHEMA_VERSION:
        raise EvidenceError(f"run summary schema mismatch: {run_id}")
    aggregates = summary["aggregates"]
    if not isinstance(aggregates, dict):
        raise EvidenceError(f"run aggregates must be an object: {run_id}")
    _validate_agentic_aggregates(
        samples["samples"],
        aggregates,
        suite=agentic_suite,
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

`index.json` accounts for every discovered run, matrix, custom campaign, and
standalone battery.  Aborted and nonterminal attempts remain explicitly classified;
they are not promoted to completed measurements.  `checksums.json` covers only the
sanitized files in this directory.
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
        run_dirs = sorted({path.parent for path in results_root.rglob("plan.json")})
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
        if harbor_results:
            campaigns.append(_export_harbor_campaign(harbor_results, temporary))
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
