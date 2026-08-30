"""Pinned static preflight for the prospective current-SM121 agent profile.

This module deliberately proves only image-local prerequisites: the exact
current native-storage SGLang image imports and initializes the Qwen reasoning
and tool parsers, and accepts the exact C1 parser/limit argv with a dummy
model. The probe does not mount weights, expose a port, request a GPU, start a
server, or issue model requests. It is not an agentic, reasoning-quality, or
performance result.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import secrets
import subprocess
from typing import Any, Callable, Mapping

from . import agentic_tools
from .sglang_sm121_storage import (
    SM121_STORAGE_CACHE_PAGES,
    SM121_STORAGE_CONTEXT_LENGTH,
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_LOCAL_IMAGE_TAG,
    SM121_STORAGE_MAX_BATCH_PAGES,
    SM121_STORAGE_MODE,
    SM121_STORAGE_NATIVE_CONTEXT,
    SM121_STORAGE_QUEUE_DEPTH,
    SM121_STORAGE_REVISION,
    SM121_STORAGE_SOURCE,
    SM121_STORAGE_SOURCE_TREE,
    SM121_STORAGE_WEIGHT_FILE_COUNT,
    SM121_STORAGE_WEIGHT_SIZE_BYTES,
    SM121StorageCandidateError,
    validate_sm121_storage_image_inspection,
)
from .sglang_sm121_cache_semantic import (
    SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
    SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED,
)


SM121_AGENT_ADMISSION_PROFILE_ID = (
    "qwen38-flash-next-nvfp4-sm121-triton-storage-agent-admission-sglang"
)
SM121_AGENT_ADMISSION_SUITE_ID = (
    "qwen38-flash-next-sm121-triton-storage-agent-admission-v1"
)
SM121_AGENT_ADMISSION_SERVED_NAME = (
    "qwen38-flash-next-nvfp4-sm121-storage-agent-admission"
)
SM121_AGENT_ADMISSION_ENDPOINT = "http://127.0.0.1:30000/v1"
SM121_AGENT_ADMISSION_DESCRIPTION = (
    "Prospective C1 Qwen3.8 Flash-Next NVFP4 Pi/cowork admission profile on "
    "the current SM121 Triton/io_uring storage runtime. Only its dedicated "
    "non-resumable parser, quality, tool-loop, long-context, cache, and "
    "host-safety admission controller may execute it."
)
SM121_AGENT_ADMISSION_REQUEST_BODY = (
    '{"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"low"}}'
)
SM121_AGENT_ADMISSION_STARTUP_TIMEOUT_S = 1_200
SM121_AGENT_ADMISSION_ESTIMATED_RAM_GIB = 101.0
SM121_AGENT_ADMISSION_MIN_MEMAVAILABLE_GIB = 14
SM121_AGENT_ADMISSION_MAX_SWAP_MIB = 64
SM121_AGENT_ADMISSION_CHUNKED_PREFILL_SIZE = 4_096
SM121_AGENT_ADMISSION_MAX_MAMBA_CACHE_SIZE = 4
SM121_AGENT_ADMISSION_QUALITY_CASE_ID = "synthetic-exact-answer-v2"
SM121_AGENT_ADMISSION_TOOL_CASE_IDS = (
    "agentic-select-and-call",
    "agentic-no-tool",
    "agentic-two-hop",
    "agentic-tool-error-recovery",
)
SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID = "sm121-agent-long-context-cache-zero-v1"
SM121_AGENT_ADMISSION_LONG_CONTEXT_PROMPT_REPETITIONS = 60_000
# Offline tokenizer/template count for the exact client-rendered first turn:
# 60K ``archive `` fillers, its two fixed instructions, the canonical three
# tool schemas, the Qwen3.8 low-thinking chat template, and the 128-token
# output reservation.  The private controller rechecks these values in an
# image-local, no-network/no-GPU tokenizer probe before it starts C1.
SM121_AGENT_ADMISSION_LONG_CONTEXT_RAW_PROMPT_SHA256 = (
    "7e7e5e087b6f4585a004a5e0369ed9be71358fa978a1a6d0cbaf682655fd5918"
)
SM121_AGENT_ADMISSION_LONG_CONTEXT_TOOLS_SHA256 = (
    "6aacd08332e48b6aaee06a27776b5bf6656a38680177502fdf8aaf9a72ee7831"
)
SM121_AGENT_ADMISSION_LONG_CONTEXT_TOKENIZER_SHA256 = (
    "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
)
SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_TEMPLATE_SHA256 = (
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
)
SM121_AGENT_ADMISSION_LONG_CONTEXT_RENDERED_PROMPT_SHA256 = (
    "a83f3b4585628a4f45b612a35de53c2d968e2ab4e9cc91e20aac49523bef269f"
)
SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_PROMPT_TOKENS = 60_489
SM121_AGENT_ADMISSION_LONG_CONTEXT_OUTPUT_TOKENS = 128
SM121_AGENT_ADMISSION_LONG_CONTEXT_BUDGET_TOKENS = (
    SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_PROMPT_TOKENS
    + SM121_AGENT_ADMISSION_LONG_CONTEXT_OUTPUT_TOKENS
)
# Retain the historical name for read-only consumers, but it now means the
# exact pinned client/template count rather than a permissive filler lower
# bound. C1 rejects any server-reported count that differs from it.
SM121_AGENT_ADMISSION_LONG_CONTEXT_MIN_PROMPT_TOKENS = (
    SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_PROMPT_TOKENS
)
SM121_AGENT_ADMISSION_NATIVE_CACHE_METRIC_FIELDS = (
    "prefill_input_tokens",
    "prefill_device_hit_tokens",
    "prefill_host_hit_tokens",
    "prefill_storage_hit_tokens",
    "cached_total_tokens",
    "cached_device_tokens",
    "cached_host_tokens",
    "cached_storage_tokens",
    "evicted_tokens",
    "retracted_requests",
)
SM121_AGENT_ADMISSION_NATIVE_CACHE_MAX_POLLS = 64
SM121_AGENT_ADMISSION_CASE_IDS = (
    SM121_AGENT_ADMISSION_QUALITY_CASE_ID,
    *SM121_AGENT_ADMISSION_TOOL_CASE_IDS,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID,
)
SM121_AGENT_ADMISSION_SUITE_DESCRIPTION = (
    "Admission-only C1 six-case current-SM121 Qwen3.8 Flash-Next Pi/cowork "
    "tool-and-reasoning gate. It freezes exact-answer quality, four bounded "
    "tool-loop semantics, and one rendered 60K low-thinking-plus-tools "
    "long-context/cache-zero-first-turn probe; only the dedicated controller "
    "may execute it and it does not measure Pi/cowork performance."
)
SM121_AGENT_ADMISSION_STATIC_PROBE_ID = (
    "qwen38-flash-next-sm121-agent-parser-static-preflight-v2"
)
SM121_AGENT_ADMISSION_STATIC_PROBE_SCHEMA_VERSION = 2
SM121_AGENT_ADMISSION_LONG_CONTEXT_BUDGET_PROBE_ID = (
    "qwen38-flash-next-sm121-agent-long-context-tokenizer-preflight-v1"
)
SM121_AGENT_ADMISSION_LONG_CONTEXT_BUDGET_PROBE_SCHEMA_VERSION = 1
_STATIC_CONTAINER_NAME_PREFIX = "sparkbench-sm121-agent-parser-"
_STATIC_CONTAINER_LABEL = "io.sparkbench.sm121-agent-parser-preflight"
_STATIC_CONTAINER_NONCE_PATTERN = re.compile(r"[0-9a-f]{32}")

SM121_AGENT_ADMISSION_ARGS = (
    "--served-model-name",
    SM121_AGENT_ADMISSION_SERVED_NAME,
    "--tp-size",
    "1",
    "--attention-backend",
    "triton",
    "--moe-runner-backend",
    "flashinfer_cutlass",
    "--quantization",
    "modelopt_fp4",
    "--load-format",
    "auto",
    "--no-ple-offload-embedding",
    "--weight-loader-drop-cache-after-load",
    "--language-only",
    "--mamba-radix-cache-strategy",
    "extra_buffer_lazy",
    "--max-mamba-cache-size",
    str(SM121_AGENT_ADMISSION_MAX_MAMBA_CACHE_SIZE),
    "--page-size",
    "64",
    "--mem-fraction-static",
    "0.85",
    "--max-total-tokens",
    str(SM121_STORAGE_CONTEXT_LENGTH),
    "--context-length",
    str(SM121_STORAGE_CONTEXT_LENGTH),
    "--chunked-prefill-size",
    str(SM121_AGENT_ADMISSION_CHUNKED_PREFILL_SIZE),
    "--max-running-requests",
    "1",
    "--cuda-graph-backend-decode",
    "disabled",
    "--cuda-graph-backend-prefill",
    "disabled",
    "--enable-metrics",
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen3_coder",
    "--host",
    "0.0.0.0",
    "--port",
    "30000",
)

# This is the scalar-only runtime projection that the dedicated controller
# obtains from one freshly started owned server. Keep the leaf contract here so
# its inspector and private journal audit cannot drift apart.
SM121_AGENT_ADMISSION_RUNTIME_EXPECTED = {
    **SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED[SM121_CACHE_SEMANTIC_CACHE_ON_ARM],
    "mamba_radix_cache_strategy": "extra_buffer_lazy",
    "max_mamba_cache_size": SM121_AGENT_ADMISSION_MAX_MAMBA_CACHE_SIZE,
    "chunked_prefill_size": SM121_AGENT_ADMISSION_CHUNKED_PREFILL_SIZE,
    "reasoning_parser": "qwen3",
    "tool_call_parser": "qwen3_coder",
    "max_running_requests": 1,
    "max_total_tokens": SM121_STORAGE_CONTEXT_LENGTH,
    "context_length": SM121_STORAGE_CONTEXT_LENGTH,
}
_RUNTIME_IDENTITY_FIELDS = frozenset(SM121_AGENT_ADMISSION_RUNTIME_EXPECTED)

_STATIC_PROBE_FIELDS = frozenset(
    {
        "schema_version",
        "probe_id",
        "docker_image_id",
        "source_tree",
        "reasoning_parser_qwen3",
        "tool_call_parser_qwen3_coder",
        "reasoning_parser_instantiated",
        "tool_call_parser_instantiated",
        "reasoning_parser",
        "tool_call_parser",
        "chunked_prefill_size",
        "max_running_requests",
        "max_total_tokens",
        "context_length",
    }
)
_LONG_CONTEXT_BUDGET_PROBE_FIELDS = frozenset(
    {
        "schema_version",
        "probe_id",
        "raw_prompt_sha256",
        "tools_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "rendered_prompt_sha256",
        "chat_prompt_tokens",
        "output_tokens",
        "budget_tokens",
        "context_length",
        "within_context",
    }
)
_STATIC_PROBE_ARGV = ("--model-path", "dummy", *SM121_AGENT_ADMISSION_ARGS)
_STATIC_PROBE_SCRIPT = "\n".join(
    (
        "import json",
        "from sglang.srt.function_call.function_call_parser import FunctionCallParser",
        "from sglang.srt.parser.reasoning_parser import ReasoningParser",
        "from sglang.srt.server_args import prepare_server_args",
        f"server_args = prepare_server_args({list(_STATIC_PROBE_ARGV)!r})",
        "reasoning_parser = ReasoningParser('qwen3')",
        "tool_call_parser = FunctionCallParser([], 'qwen3_coder')",
        "print(json.dumps({",
        "    'reasoning_parser_qwen3': 'qwen3' in ReasoningParser.DetectorMap,",
        "    'tool_call_parser_qwen3_coder': 'qwen3_coder' in FunctionCallParser.ToolCallParserEnum,",
        "    'reasoning_parser_instantiated': reasoning_parser is not None,",
        "    'tool_call_parser_instantiated': tool_call_parser is not None,",
        "    'reasoning_parser': server_args.reasoning_parser,",
        "    'tool_call_parser': server_args.tool_call_parser,",
        "    'chunked_prefill_size': server_args.chunked_prefill_size,",
        "    'max_running_requests': server_args.max_running_requests,",
        "    'max_total_tokens': server_args.max_total_tokens,",
        "    'context_length': server_args.context_length,",
        "}, sort_keys=True, separators=(',', ':')))",
    )
)


def _static_long_context_tools() -> list[dict[str, Any]]:
    """Return the same fixed first-turn schemas as the private direct client."""

    scenario = agentic_tools._scenario("agentic-select-and-call", 0)
    return agentic_tools._rotated_tool_schemas(scenario)


def _static_long_context_tokenizer_script() -> str:
    """Build a no-network image-local count script without embedding the prompt."""

    tools = _static_long_context_tools()
    tools_json = json.dumps(
        tools, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return "\n".join(
        (
            "import hashlib",
            "import json",
            "from pathlib import Path",
            "from transformers import AutoTokenizer",
            "model_path = Path('/model')",
            "def digest(name):",
            "    return hashlib.sha256((model_path / name).read_bytes()).hexdigest()",
            "prompt = ('Read the complete synthetic context before answering. '"
            " + 'archive ' * 60000"
            " + 'Do not call a tool. Reply with exactly LONG-CONTEXT-READY.')",
            f"tools = json.loads({tools_json!r})",
            "tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)",
            "kwargs = {'tools': tools, 'add_generation_prompt': True, 'enable_thinking': True, 'reasoning_effort': 'low'}",
            "rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, **kwargs)",
            "token_ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True, return_dict=False, **kwargs)",
            "chat_prompt_tokens = len(token_ids)",
            "output_tokens = 128",
            "budget_tokens = chat_prompt_tokens + output_tokens",
            "print(json.dumps({",
            "  'raw_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),",
            "  'tools_sha256': hashlib.sha256(json.dumps(tools, ensure_ascii=True, separators=(',', ':'), sort_keys=True).encode()).hexdigest(),",
            "  'tokenizer_sha256': digest('tokenizer.json'),",
            "  'chat_template_sha256': digest('chat_template.jinja'),",
            "  'rendered_prompt_sha256': hashlib.sha256(rendered.encode()).hexdigest(),",
            "  'chat_prompt_tokens': chat_prompt_tokens,",
            "  'output_tokens': output_tokens,",
            "  'budget_tokens': budget_tokens,",
            "  'context_length': 65536,",
            "  'within_context': budget_tokens < 65536,",
            "}, sort_keys=True, separators=(',', ':')))",
        )
    )


_STATIC_LONG_CONTEXT_TOKENIZER_SCRIPT = _static_long_context_tokenizer_script()


class SM121AgentAdmissionError(ValueError):
    """Raised when the prospective agent admission contract drifts."""


def _value(item: Any, field: str) -> object:
    return item.get(field) if isinstance(item, Mapping) else getattr(item, field, None)


def _require(value: object, expected: object, field: str) -> None:
    if type(expected) is bool:
        matches = type(value) is bool and value is expected
    elif type(expected) is int:
        matches = type(value) is int and value == expected
    elif type(expected) is float:
        matches = (
            type(value) in {int, float}
            and type(value) is not bool
            and value == expected
        )
    else:
        matches = value == expected
    if not matches:
        raise SM121AgentAdmissionError(
            f"{field} does not match the pinned SM121 agent admission profile"
        )


def _canonical_request_body(value: object) -> dict[str, object]:
    if not isinstance(value, str):
        raise SM121AgentAdmissionError("request_body_json must be a canonical object")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise SM121AgentAdmissionError("request_body_json must be valid JSON") from error
    if type(decoded) is not dict:
        raise SM121AgentAdmissionError("request_body_json must be a canonical object")
    return decoded


def is_sm121_agent_admission_candidate(model: Any) -> bool:
    """Return whether ``model`` is the sole prospective C1 agent profile."""

    return _value(model, "id") == SM121_AGENT_ADMISSION_PROFILE_ID


def validate_sm121_agent_admission_candidate(model: Any) -> None:
    """Require exactly the one prospective C1 agent-admission profile."""

    if not is_sm121_agent_admission_candidate(model):
        raise SM121AgentAdmissionError("SM121 agent admission profile is invalid")
    validate_sm121_agent_admission_profile(model)


def validate_sm121_agent_admission_profile(model: Any) -> None:
    """Require the immutable current-SM121 low-thinking/parser profile."""

    if not is_sm121_agent_admission_candidate(model):
        return
    expected = {
        "description": SM121_AGENT_ADMISSION_DESCRIPTION,
        "support_status": "exploratory",
        "backend": "sglang",
        "source": SM121_STORAGE_SOURCE,
        "revision": SM121_STORAGE_REVISION,
        "weight_file_count": SM121_STORAGE_WEIGHT_FILE_COUNT,
        "weight_size_bytes": SM121_STORAGE_WEIGHT_SIZE_BYTES,
        "served_name": SM121_AGENT_ADMISSION_SERVED_NAME,
        "tasks": ("chat", "json", "thinking", "tools"),
        "architecture": "moe+qsa+gdn",
        "quantization": "nvfp4+ple-fp8-nvme-io-uring",
        "lifecycle": "docker",
        "image": SM121_STORAGE_LOCAL_IMAGE_TAG,
        "local_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
        "image_digest": None,
        "cache_dir": "user",
        "max_context": SM121_STORAGE_CONTEXT_LENGTH,
        "native_context": SM121_STORAGE_NATIVE_CONTEXT,
        "startup_timeout_s": SM121_AGENT_ADMISSION_STARTUP_TIMEOUT_S,
        "estimated_ram_gib": SM121_AGENT_ADMISSION_ESTIMATED_RAM_GIB,
        "host_safety_min_memavailable_gib": SM121_AGENT_ADMISSION_MIN_MEMAVAILABLE_GIB,
        "host_safety_max_swap_growth_mib": SM121_AGENT_ADMISSION_MAX_SWAP_MIB,
        "host_safety_max_starting_swap_mib": SM121_AGENT_ADMISSION_MAX_SWAP_MIB,
        "endpoint": SM121_AGENT_ADMISSION_ENDPOINT,
        "fetch_allow_patterns": (),
        "fetch_ignore_patterns": (),
        "sglang_storage_mode": SM121_STORAGE_MODE,
        "sglang_ple_nvme_queue_depth": SM121_STORAGE_QUEUE_DEPTH,
        "sglang_ple_nvme_max_batch_pages": SM121_STORAGE_MAX_BATCH_PAGES,
        "sglang_ple_nvme_cache_pages": SM121_STORAGE_CACHE_PAGES,
        "draft_source": None,
        "draft_revision": None,
        "draft_weight_size_bytes": None,
        "draft_model_file": None,
        "draft_model_digest": None,
        "draft_model_size_bytes": None,
        "sglang_allow_hf_metadata_probe": False,
        "sglang_source_overlays": (),
        "sglang_ple_mmap": False,
        "sglang_ple_omitted": False,
        "sglang_ple_cache_mode": None,
        "sglang_ple_cache_marker_digest": None,
        "sglang_ple_cache_payload_digest": None,
        "recipe_source": None,
        "recipe_revision": None,
        "runtime_python": None,
        "runtime_binary": None,
        "runtime_digest": None,
        "runtime_parallel": None,
        "runtime_source_dir": None,
        "runtime_revision": None,
        "model_file": None,
        "model_digest": None,
        "model_size_bytes": None,
        "model_shards": (),
        "mmproj_file": None,
        "mmproj_digest": None,
        "mmproj_size_bytes": None,
        "prefix_cache_mode": None,
    }
    sequence_fields = {
        "tasks",
        "fetch_allow_patterns",
        "fetch_ignore_patterns",
        "sglang_source_overlays",
        "model_shards",
    }
    for field, wanted in expected.items():
        actual = _value(model, field)
        if field in sequence_fields and isinstance(actual, (list, tuple)):
            actual = tuple(actual)
        _require(actual, wanted, field)
    _require(_value(model, "args"), SM121_AGENT_ADMISSION_ARGS, "args")
    request_body_json = _value(model, "request_body_json")
    _require(
        request_body_json,
        SM121_AGENT_ADMISSION_REQUEST_BODY,
        "request_body_json",
    )
    request_body = _canonical_request_body(request_body_json)
    _require(
        request_body,
        {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}},
        "request_body_json",
    )
    argument_names = tuple(
        str(argument).split("=", 1)[0]
        for argument in tuple(_value(model, "args") or ())
    )
    if any(name.startswith("--speculative-") for name in argument_names):
        raise SM121AgentAdmissionError(
            "SM121 agent admission forbids speculative decoding arguments"
        )
    if "--enable-auto-tool-choice" in argument_names:
        raise SM121AgentAdmissionError(
            "SM121 agent admission does not import a non-SGLang tool-choice flag"
        )


def validate_sm121_agent_admission_runtime_identity(
    identity: object,
) -> dict[str, object]:
    """Require the exact scalar C1 runtime projection.

    This validates a controller-produced allowlist, not an untrusted full
    ``/server_info`` response. It deliberately has no server, Docker, network,
    timestamp, or journal side effect.
    """

    if (
        type(identity) is not dict
        or frozenset(identity) != _RUNTIME_IDENTITY_FIELDS
    ):
        raise SM121AgentAdmissionError("SM121 agent runtime identity is invalid")
    for field, expected in SM121_AGENT_ADMISSION_RUNTIME_EXPECTED.items():
        _require(identity[field], expected, field)
    return dict(identity)


def validate_sm121_agent_native_cache_metrics_receipt(
    receipt: object,
) -> dict[str, object]:
    """Validate the scalar-only native cache-zero receipt for C1.

    A controller writes this record only after two settled owned-server metric
    views before and after the sole 60K request.  The provenance binding stays
    opaque in runtime; this pure validator receives no endpoint, credential,
    container, process, label, metric text, prompt, response, or timing data.
    """

    fields = {
        "event",
        "schema_version",
        "fresh_lifetime",
        "same_owned_generation",
        "metrics_available",
        "guardrail_metrics_available",
        "metrics_before_settled",
        "metrics_after_settled",
        "metrics_before_polls",
        "metrics_after_polls",
        "metrics_before",
        "metrics_after",
        "native_input_observed",
        "zero_metric_cache_hits",
        "guardrails_clean",
    }
    if type(receipt) is not dict or set(receipt) != fields:
        raise SM121AgentAdmissionError("SM121 agent native cache receipt is invalid")
    if (
        receipt["event"] != "sm121_agent_native_cache_metrics_receipt"
        or type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or type(receipt["fresh_lifetime"]) is not int
        or receipt["fresh_lifetime"] != 3
    ):
        raise SM121AgentAdmissionError("SM121 agent native cache receipt is invalid")
    for field in (
        "same_owned_generation",
        "metrics_available",
        "guardrail_metrics_available",
        "metrics_before_settled",
        "metrics_after_settled",
        "native_input_observed",
        "zero_metric_cache_hits",
        "guardrails_clean",
    ):
        if type(receipt[field]) is not bool or receipt[field] is not True:
            raise SM121AgentAdmissionError("SM121 agent native cache receipt is invalid")
    for field in ("metrics_before_polls", "metrics_after_polls"):
        value = receipt[field]
        if (
            type(value) is not int
            or not 2 <= value <= SM121_AGENT_ADMISSION_NATIVE_CACHE_MAX_POLLS
        ):
            raise SM121AgentAdmissionError("SM121 agent native cache receipt is invalid")
    maps: dict[str, dict[str, int]] = {}
    for name in ("metrics_before", "metrics_after"):
        values = receipt[name]
        if type(values) is not dict or set(values) != set(
            SM121_AGENT_ADMISSION_NATIVE_CACHE_METRIC_FIELDS
        ):
            raise SM121AgentAdmissionError("SM121 agent native cache receipt is invalid")
        normalized: dict[str, int] = {}
        for field in SM121_AGENT_ADMISSION_NATIVE_CACHE_METRIC_FIELDS:
            value = values[field]
            if type(value) is not int or value < 0:
                raise SM121AgentAdmissionError(
                    "SM121 agent native cache receipt is invalid"
                )
            normalized[field] = value
        maps[name] = normalized
    before = maps["metrics_before"]
    after = maps["metrics_after"]
    hit_fields = tuple(
        field
        for field in SM121_AGENT_ADMISSION_NATIVE_CACHE_METRIC_FIELDS
        if field
        not in {"prefill_input_tokens", "evicted_tokens", "retracted_requests"}
    )
    native_input_observed = (
        after["prefill_input_tokens"] > before["prefill_input_tokens"]
    )
    zero_metric_cache_hits = all(
        before[field] == after[field] == 0 for field in hit_fields
    )
    guardrails_clean = all(
        before[field] == after[field] == 0
        for field in ("evicted_tokens", "retracted_requests")
    )
    if (
        receipt["native_input_observed"] is not native_input_observed
        or receipt["zero_metric_cache_hits"] is not zero_metric_cache_hits
        or receipt["guardrails_clean"] is not guardrails_clean
    ):
        raise SM121AgentAdmissionError("SM121 agent native cache receipt is invalid")
    return {
        **{field: receipt[field] for field in fields - {"metrics_before", "metrics_after"}},
        "metrics_before": dict(before),
        "metrics_after": dict(after),
    }


def validate_sm121_agent_admission_suite(suite: Any) -> None:
    """Require the exact six-case, controller-owned C1 admission suite.

    This is a static manifest contract only.  It contains no prompts, rendered
    payloads, expected completions, or timing thresholds; a later dedicated
    controller owns those transient inputs and scalar-only outcomes.
    """

    def require_suite_field(value: object, expected: object, field: str) -> None:
        try:
            _require(value, expected, field)
        except SM121AgentAdmissionError as error:
            raise SM121AgentAdmissionError(
                f"SM121 agent admission suite field {field} changed"
            ) from error

    require_suite_field(
        _value(suite, "id"), SM121_AGENT_ADMISSION_SUITE_ID, "id"
    )
    require_suite_field(
        _value(suite, "description"),
        SM121_AGENT_ADMISSION_SUITE_DESCRIPTION,
        "description",
    )
    require_suite_field(_value(suite, "schema_version"), 1, "schema_version")
    require_suite_field(_value(suite, "protocol_digest"), None, "protocol_digest")
    cases = _value(suite, "cases")
    if not isinstance(cases, (list, tuple)) or len(cases) != len(
        SM121_AGENT_ADMISSION_CASE_IDS
    ):
        raise SM121AgentAdmissionError("SM121 agent admission suite cases are invalid")
    expected_cases = (
        {
            "id": SM121_AGENT_ADMISSION_QUALITY_CASE_ID,
            "kind": "quality",
            "requires": ("chat",),
            "warmups": 0,
            "repetitions": 2,
            "max_output_tokens": 512,
            "temperature": 0.0,
            "concurrency": 1,
            "prompt_repetitions": 0,
            "max_turns": 1,
        },
        *(
            {
                "id": case_id,
                "kind": "agentic",
                "requires": ("chat", "tools"),
                "warmups": 0,
                "repetitions": 3,
                "max_output_tokens": 4096,
                "temperature": 0.0,
                "concurrency": 1,
                "prompt_repetitions": 0,
                "max_turns": 6,
            }
            for case_id in SM121_AGENT_ADMISSION_TOOL_CASE_IDS
        ),
        {
            "id": SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID,
            "kind": "capability",
            "requires": ("chat", "thinking", "tools"),
            "warmups": 0,
            "repetitions": 1,
            "max_output_tokens": 128,
            "temperature": 0.0,
            "concurrency": 1,
            "prompt_repetitions": SM121_AGENT_ADMISSION_LONG_CONTEXT_PROMPT_REPETITIONS,
            "max_turns": 1,
        },
    )
    for case, expected in zip(cases, expected_cases, strict=True):
        for field, wanted in expected.items():
            actual = _value(case, field)
            if field == "requires" and isinstance(actual, (list, tuple)):
                actual = tuple(actual)
            require_suite_field(actual, wanted, field)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def validate_sm121_agent_parser_static_probe(probe: object) -> dict[str, object]:
    """Validate the scalar-only static parser registry attestation."""

    if type(probe) is not dict or frozenset(probe) != _STATIC_PROBE_FIELDS:
        raise SM121AgentAdmissionError("SM121 agent parser preflight is invalid")
    expected = {
        "schema_version": SM121_AGENT_ADMISSION_STATIC_PROBE_SCHEMA_VERSION,
        "probe_id": SM121_AGENT_ADMISSION_STATIC_PROBE_ID,
        "docker_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
        "source_tree": SM121_STORAGE_SOURCE_TREE,
        "reasoning_parser_qwen3": True,
        "tool_call_parser_qwen3_coder": True,
        "reasoning_parser_instantiated": True,
        "tool_call_parser_instantiated": True,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
        "chunked_prefill_size": SM121_AGENT_ADMISSION_CHUNKED_PREFILL_SIZE,
        "max_running_requests": 1,
        "max_total_tokens": SM121_STORAGE_CONTEXT_LENGTH,
        "context_length": SM121_STORAGE_CONTEXT_LENGTH,
    }
    for field, value in expected.items():
        actual = probe.get(field)
        valid = actual is value if isinstance(value, bool) else actual == value
        if not valid:
            raise SM121AgentAdmissionError("SM121 agent parser preflight is invalid")
    return dict(probe)


def validate_sm121_agent_long_context_budget_probe(
    probe: object,
) -> dict[str, object]:
    """Require the pinned tokenizer/template C1 first-turn budget proof."""

    if type(probe) is not dict or frozenset(probe) != _LONG_CONTEXT_BUDGET_PROBE_FIELDS:
        raise SM121AgentAdmissionError("SM121 agent long-context budget is invalid")
    expected = {
        "schema_version": SM121_AGENT_ADMISSION_LONG_CONTEXT_BUDGET_PROBE_SCHEMA_VERSION,
        "probe_id": SM121_AGENT_ADMISSION_LONG_CONTEXT_BUDGET_PROBE_ID,
        "raw_prompt_sha256": SM121_AGENT_ADMISSION_LONG_CONTEXT_RAW_PROMPT_SHA256,
        "tools_sha256": SM121_AGENT_ADMISSION_LONG_CONTEXT_TOOLS_SHA256,
        "tokenizer_sha256": SM121_AGENT_ADMISSION_LONG_CONTEXT_TOKENIZER_SHA256,
        "chat_template_sha256": SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_TEMPLATE_SHA256,
        "rendered_prompt_sha256": SM121_AGENT_ADMISSION_LONG_CONTEXT_RENDERED_PROMPT_SHA256,
        "chat_prompt_tokens": SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_PROMPT_TOKENS,
        "output_tokens": SM121_AGENT_ADMISSION_LONG_CONTEXT_OUTPUT_TOKENS,
        "budget_tokens": SM121_AGENT_ADMISSION_LONG_CONTEXT_BUDGET_TOKENS,
        "context_length": SM121_STORAGE_CONTEXT_LENGTH,
        "within_context": True,
    }
    for field, value in expected.items():
        actual = probe.get(field)
        valid = actual is value if isinstance(value, bool) else actual == value
        if not valid:
            raise SM121AgentAdmissionError("SM121 agent long-context budget is invalid")
    if probe["budget_tokens"] >= probe["context_length"]:
        raise SM121AgentAdmissionError("SM121 agent long-context budget is invalid")
    return dict(probe)


def _run_static_command(
    runner: Callable[..., subprocess.CompletedProcess[str]], command: list[str]
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SM121AgentAdmissionError("SM121 agent parser preflight is unavailable") from error


def _new_static_container_identity() -> tuple[str, str]:
    """Create an unguessable, Docker-safe identity for one static probe."""

    nonce = secrets.token_hex(16)
    if _STATIC_CONTAINER_NONCE_PATTERN.fullmatch(nonce) is None:
        raise SM121AgentAdmissionError("SM121 agent parser preflight is unavailable")
    return f"{_STATIC_CONTAINER_NAME_PREFIX}{nonce}", nonce


def _best_effort_static_container_cleanup(
    runner: Callable[..., subprocess.CompletedProcess[str]],
    *,
    container_name: str,
    nonce: str,
) -> None:
    """Remove only the exact labelled probe container after an interrupted run.

    ``docker run --rm`` normally removes the container itself.  A timeout can
    kill the foreground Docker client while its container is still alive, so
    this verifies the random run label before using the destructive cleanup.
    It deliberately ignores cleanup failures: the primary admission error is
    still the useful result and no unrelated container is ever targeted.
    """

    try:
        inspection = runner(
            [
                "docker",
                "container",
                "inspect",
                container_name,
                "--format",
                "{{ index .Config.Labels \"io.sparkbench.sm121-agent-parser-preflight\" }}",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except BaseException:
        return
    if inspection.returncode != 0 or inspection.stdout.strip() != nonce:
        return
    try:
        runner(
            ["docker", "container", "rm", "--force", container_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except BaseException:
        return


def _run_static_container_command(
    runner: Callable[..., subprocess.CompletedProcess[str]],
    command: list[str],
    *,
    container_name: str,
    nonce: str,
) -> subprocess.CompletedProcess[str]:
    """Run the static container and clean its verified name if interrupted."""

    try:
        return _run_static_command(runner, command)
    except BaseException:
        _best_effort_static_container_cleanup(
            runner,
            container_name=container_name,
            nonce=nonce,
        )
        raise


def _inspect_static_image(
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, str]:
    completed = _run_static_command(
        runner,
        [
            "docker",
            "image",
            "inspect",
            SM121_STORAGE_LOCAL_IMAGE_TAG,
            "--format",
            "{{json .}}",
        ],
    )
    if completed.returncode != 0:
        raise SM121AgentAdmissionError("SM121 agent parser preflight is unavailable")
    try:
        inspection = json.loads(completed.stdout, object_pairs_hook=_unique_json_object)
    except (TypeError, json.JSONDecodeError, ValueError) as error:
        raise SM121AgentAdmissionError("SM121 agent parser preflight is unavailable") from error
    if type(inspection) is not dict:
        raise SM121AgentAdmissionError("SM121 agent parser preflight is unavailable")
    try:
        return validate_sm121_storage_image_inspection(
            inspection, image=SM121_STORAGE_LOCAL_IMAGE_TAG
        )
    except SM121StorageCandidateError as error:
        raise SM121AgentAdmissionError("SM121 agent parser image identity changed") from error


def probe_sm121_agent_parser_static_preflight(
    model: Any,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """Run one no-network, no-GPU parser/CLI preflight in the pinned image."""

    validate_sm121_agent_admission_profile(model)
    image_identity = _inspect_static_image(runner)
    container_name, nonce = _new_static_container_identity()
    completed = _run_static_container_command(
        runner,
        [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--name",
            container_name,
            "--label",
            f"{_STATIC_CONTAINER_LABEL}={nonce}",
            "--runtime",
            "runc",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "2g",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=64m",
            "--env",
            "HOME=/tmp",
            "--env",
            "XDG_CACHE_HOME=/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--entrypoint",
            "python3",
            image_identity["docker_image_id"],
            "-c",
            _STATIC_PROBE_SCRIPT,
        ],
        container_name=container_name,
        nonce=nonce,
    )
    if completed.returncode != 0:
        _best_effort_static_container_cleanup(
            runner,
            container_name=container_name,
            nonce=nonce,
        )
        raise SM121AgentAdmissionError("SM121 agent parser preflight failed")
    try:
        observed = json.loads(completed.stdout, object_pairs_hook=_unique_json_object)
    except (TypeError, json.JSONDecodeError, ValueError) as error:
        raise SM121AgentAdmissionError("SM121 agent parser preflight failed") from error
    if type(observed) is not dict or frozenset(observed) != (
        _STATIC_PROBE_FIELDS
        - {"schema_version", "probe_id", "docker_image_id", "source_tree"}
    ):
        raise SM121AgentAdmissionError("SM121 agent parser preflight failed")
    probe = {
        "schema_version": SM121_AGENT_ADMISSION_STATIC_PROBE_SCHEMA_VERSION,
        "probe_id": SM121_AGENT_ADMISSION_STATIC_PROBE_ID,
        "docker_image_id": image_identity["docker_image_id"],
        "source_tree": image_identity["source_tree"],
        **observed,
    }
    return validate_sm121_agent_parser_static_probe(probe)


def probe_sm121_agent_long_context_budget_preflight(
    model: Any,
    *,
    snapshot_path: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """Tokenize the exact C1 first turn before any inference server starts.

    The target snapshot is supplied by the runtime's exact-cache resolver and
    mounted read-only into the already pinned local image. This CPU-only,
    no-network probe checks the target tokenizer and chat template rather than
    assuming the repeated filler is one token or discovering overflow on the
    first live long-context request.
    """

    validate_sm121_agent_admission_candidate(model)
    if not isinstance(snapshot_path, Path) or snapshot_path.is_symlink():
        raise SM121AgentAdmissionError("SM121 agent long-context budget is unavailable")
    try:
        resolved_snapshot = snapshot_path.resolve(strict=True)
    except OSError as error:
        raise SM121AgentAdmissionError(
            "SM121 agent long-context budget is unavailable"
        ) from error
    if not resolved_snapshot.is_dir():
        raise SM121AgentAdmissionError("SM121 agent long-context budget is unavailable")
    image_identity = _inspect_static_image(runner)
    container_name, nonce = _new_static_container_identity()
    completed = _run_static_container_command(
        runner,
        [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--name",
            container_name,
            "--label",
            f"{_STATIC_CONTAINER_LABEL}={nonce}",
            "--runtime",
            "runc",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "2g",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=64m",
            "--mount",
            f"type=bind,src={resolved_snapshot},dst=/model,readonly",
            "--env",
            "HOME=/tmp",
            "--env",
            "XDG_CACHE_HOME=/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--entrypoint",
            "python3",
            image_identity["docker_image_id"],
            "-c",
            _STATIC_LONG_CONTEXT_TOKENIZER_SCRIPT,
        ],
        container_name=container_name,
        nonce=nonce,
    )
    if completed.returncode != 0:
        _best_effort_static_container_cleanup(
            runner,
            container_name=container_name,
            nonce=nonce,
        )
        raise SM121AgentAdmissionError("SM121 agent long-context budget failed")
    try:
        observed = json.loads(
            completed.stdout, object_pairs_hook=_unique_json_object
        )
    except (TypeError, json.JSONDecodeError, ValueError) as error:
        raise SM121AgentAdmissionError(
            "SM121 agent long-context budget failed"
        ) from error
    try:
        return validate_sm121_agent_long_context_budget_probe(observed)
    except SM121AgentAdmissionError as error:
        raise SM121AgentAdmissionError(
            "SM121 agent long-context budget failed"
        ) from error
