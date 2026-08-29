"""Pinned contract for the SM121 native-NVMe SGLang canary.

This is intentionally a singleton rather than a generic SGLang tuning API.
The image has no registry manifest digest, so its Docker config ID, platform,
and build labels are all part of the admission identity.  Ordinary benchmark
entry points must not execute this profile before the dedicated canary has
established its quality and long-context evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from .seccomp_profile_contract import DERIVED_SHA256 as SM121_STORAGE_SECCOMP_SHA256


SM121_STORAGE_MODE = "qwen4_ple_nvme_io_uring"
SM121_STORAGE_CANDIDATE_ID = "sglang-sm121-triton-storage-v1"
SM121_STORAGE_PROFILE_ID = (
    "qwen38-flash-next-nvfp4-sm121-triton-storage-target-only-sglang"
)
SM121_STORAGE_SUITE_ID = "qwen38-flash-next-sm121-triton-storage-canary"
SM121_STORAGE_EXECUTION_MODE = "sm121_storage_canary_fresh_lifetimes"
SM121_STORAGE_RUNTIME_PROVENANCE_EVENT = "sm121_storage_runtime_provenance"
SM121_STORAGE_LIFETIME_COUNT = 2
SM121_STORAGE_LOCAL_IMAGE_TAG = "local/sglang:sm121-storage-274ee330-runtime"
SM121_STORAGE_LOCAL_IMAGE_ID = (
    "sha256:b14c39fb7cb2e0b82f2f8cae1e115a55f2bb69b5ec6fd7ccc4099b219d1096b0"
)
SM121_STORAGE_SOURCE_TREE = "274ee330db7ea9653807b868c0fb8693d50ed7b2"
SM121_STORAGE_BUILD_CONTRACT_SHA256 = (
    "sha256:c9c7c5bb958a8cf4c0fbc904b40c5e51fac82ef97c6e1fc391e2b67b5c9d9975"
)
SM121_STORAGE_PLATFORM = "linux/arm64"
SM121_STORAGE_SOURCE = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
SM121_STORAGE_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
SM121_STORAGE_SERVED_NAME = "qwen38-flash-next-nvfp4-sm121-storage-target-only"
SM121_STORAGE_QUEUE_DEPTH = 512
SM121_STORAGE_MAX_BATCH_PAGES = 4096
SM121_STORAGE_CACHE_PAGES = 0
SM121_STORAGE_CONTEXT_LENGTH = 65_536
SM121_STORAGE_NATIVE_CONTEXT = 262_144
SM121_STORAGE_WEIGHT_SIZE_BYTES = 135_195_303_851
SM121_STORAGE_WEIGHT_FILE_COUNT = 206
SM121_STORAGE_VARIED_CONTEXT_RECORDS = 19_000
SM121_STORAGE_VARIED_CONTEXT_CASE_ID = (
    "sm121-varied-context-needle-19000-mid-s20260828-c1-v1"
)
# Offline count against the pinned target snapshot's Qwen tokenizer and its
# ``enable_thinking=False`` chat template.  The prompt digest lets the
# regression test detect text-generator drift without retaining the prompt.
SM121_STORAGE_VARIED_CONTEXT_PROMPT_SHA256 = (
    "49470ca55dc7ce67d505badd8e3bc5e3f192711ebdc368db80b7ee63cd9a4f3f"
)
SM121_STORAGE_VARIED_CONTEXT_RAW_PROMPT_TOKENS = 62_324
SM121_STORAGE_VARIED_CONTEXT_CHAT_PROMPT_TOKENS = 62_336
SM121_STORAGE_VARIED_CONTEXT_OUTPUT_TOKENS = 64
SM121_STORAGE_VARIED_CONTEXT_BUDGET_TOKENS = (
    SM121_STORAGE_VARIED_CONTEXT_CHAT_PROMPT_TOKENS
    + SM121_STORAGE_VARIED_CONTEXT_OUTPUT_TOKENS
)
SM121_STORAGE_RUNTIME_PROVENANCE_FIELDS = frozenset(
    {
        "candidate_id",
        "candidate_source_tree",
        "build_contract_sha256",
        "docker_image_id",
        "sglang_storage_mode",
        "sglang_ple_nvme_backend",
        "sglang_ple_nvme_queue_depth",
        "sglang_ple_nvme_max_batch_pages",
        "sglang_ple_nvme_cache_pages",
        "sglang_rust_build_mode",
        "seccomp_profile_sha256",
        "container_rootfs",
        "container_capabilities",
        "container_no_new_privileges",
        "hf_network_policy",
        "network_topology",
        "benchmark_scope",
        "model_acquisition",
        "api_authentication",
        "api_key_file_mode",
    }
)
_SM121_STORAGE_RUNTIME_PROVENANCE_EXPECTED = {
    "candidate_id": SM121_STORAGE_CANDIDATE_ID,
    "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
    "build_contract_sha256": SM121_STORAGE_BUILD_CONTRACT_SHA256,
    "docker_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
    "sglang_storage_mode": SM121_STORAGE_MODE,
    "sglang_ple_nvme_backend": "io_uring",
    "sglang_ple_nvme_queue_depth": SM121_STORAGE_QUEUE_DEPTH,
    "sglang_ple_nvme_max_batch_pages": SM121_STORAGE_MAX_BATCH_PAGES,
    "sglang_ple_nvme_cache_pages": SM121_STORAGE_CACHE_PAGES,
    "sglang_rust_build_mode": "never",
    "seccomp_profile_sha256": "sha256:" + SM121_STORAGE_SECCOMP_SHA256,
    "container_rootfs": "readonly_tmpfs_writable_cache",
    "container_capabilities": "dropped_all",
    "container_no_new_privileges": True,
    "hf_network_policy": "offline",
    "network_topology": "loopback_published_bridge",
    "benchmark_scope": "sm121_storage_pre_admission_canary",
    "model_acquisition": "disabled_exact_read_only_snapshot",
    "api_authentication": "ephemeral_bearer",
    "api_key_file_mode": "0600",
}

# A cache-disabled fresh-process canary deliberately trades a production cache
# setting for isolation.  Page size 64 remains valid because the candidate
# source accepts --disable-radix-cache as the alternative to extra_buffer.
SM121_STORAGE_ARGS = (
    "--served-model-name",
    SM121_STORAGE_SERVED_NAME,
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
    "--disable-radix-cache",
    "--page-size",
    "64",
    "--mem-fraction-static",
    "0.85",
    "--max-total-tokens",
    str(SM121_STORAGE_CONTEXT_LENGTH),
    "--context-length",
    str(SM121_STORAGE_CONTEXT_LENGTH),
    "--chunked-prefill-size",
    "1024",
    "--max-running-requests",
    "1",
    "--cuda-graph-backend-decode",
    "disabled",
    "--cuda-graph-backend-prefill",
    "disabled",
    "--enable-metrics",
    "--host",
    "0.0.0.0",
    "--port",
    "30000",
)

_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


class SM121StorageCandidateError(ValueError):
    """Raised when an attempted native-storage canary deviates from its pin."""


def validate_sm121_storage_runtime_provenance_event(
    event: object, *, fresh_server_lifetime: int | None = None
) -> None:
    """Validate one scalar-only SM121 native-provenance journal event.

    ``Journal.append`` injects a timestamp after validation, so replayed
    events may contain that one additional journal field. The contract permits
    no free-form runtime text, paths, URLs, arguments, container IDs, or
    credentials.
    """

    if type(event) is not dict:
        raise SM121StorageCandidateError(
            "SM121 storage native provenance event is not an object"
        )
    required_fields = SM121_STORAGE_RUNTIME_PROVENANCE_FIELDS | {
        "event",
        "fresh_server_lifetime",
    }
    event_fields = frozenset(event)
    if event_fields not in {required_fields, required_fields | {"timestamp"}}:
        raise SM121StorageCandidateError(
            "SM121 storage native provenance event fields are invalid"
        )
    if event.get("event") != SM121_STORAGE_RUNTIME_PROVENANCE_EVENT:
        raise SM121StorageCandidateError(
            "SM121 storage native provenance event type is invalid"
        )
    lifetime = event.get("fresh_server_lifetime")
    if isinstance(lifetime, bool) or not isinstance(lifetime, int) or lifetime <= 0:
        raise SM121StorageCandidateError("SM121 storage provenance lifetime is invalid")
    if fresh_server_lifetime is not None and lifetime != fresh_server_lifetime:
        raise SM121StorageCandidateError(
            "SM121 storage provenance lifetime does not match"
        )
    if "timestamp" in event and not isinstance(event["timestamp"], str):
        raise SM121StorageCandidateError("SM121 storage provenance timestamp is invalid")
    for field, expected in _SM121_STORAGE_RUNTIME_PROVENANCE_EXPECTED.items():
        actual = event.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise SM121StorageCandidateError(
                "SM121 storage native provenance values are invalid"
            )


def _value(item: Any, field: str) -> object:
    """Read a frozen mapping or live manifest object without coercion."""

    if isinstance(item, Mapping):
        return item.get(field)
    return getattr(item, field, None)


def is_sm121_storage_candidate(model: Any) -> bool:
    """Return whether ``model`` selects the singleton native-storage path."""

    return _value(model, "sglang_storage_mode") == SM121_STORAGE_MODE


def is_sm121_storage_canary_plan(model: Any, suite: Any) -> bool:
    """Return whether frozen records select the singleton canary topology."""

    return (
        is_sm121_storage_candidate(model)
        and _value(suite, "id") == SM121_STORAGE_SUITE_ID
    )


def _require(value: object, expected: object, field: str) -> None:
    if value != expected:
        raise SM121StorageCandidateError(
            f"{field} does not match the pinned SM121 storage candidate"
        )


def _canonical_request_body(value: object) -> dict[str, object]:
    if not isinstance(value, str):
        raise SM121StorageCandidateError(
            "request_body_json must be a canonical target-only object"
        )
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise SM121StorageCandidateError(
            "request_body_json must be valid JSON"
        ) from error
    if type(decoded) is not dict:
        raise SM121StorageCandidateError(
            "request_body_json must be a canonical target-only object"
        )
    return decoded


def validate_sm121_storage_candidate(model: Any) -> None:
    """Fail closed unless a model exactly describes the one canary lane."""

    if not is_sm121_storage_candidate(model):
        return
    _require(_value(model, "id"), SM121_STORAGE_PROFILE_ID, "id")
    _require(_value(model, "backend"), "sglang", "backend")
    _require(_value(model, "source"), SM121_STORAGE_SOURCE, "source")
    _require(_value(model, "revision"), SM121_STORAGE_REVISION, "revision")
    _require(
        _value(model, "served_name"),
        SM121_STORAGE_SERVED_NAME,
        "served_name",
    )
    _require(tuple(_value(model, "tasks") or ()), ("chat",), "tasks")
    _require(_value(model, "lifecycle"), "docker", "lifecycle")
    _require(_value(model, "image"), SM121_STORAGE_LOCAL_IMAGE_TAG, "image")
    _require(
        _value(model, "local_image_id"),
        SM121_STORAGE_LOCAL_IMAGE_ID,
        "local_image_id",
    )
    _require(_value(model, "image_digest"), None, "image_digest")
    _require(_value(model, "cache_dir"), "user", "cache_dir")
    _require(
        _value(model, "max_context"),
        SM121_STORAGE_CONTEXT_LENGTH,
        "max_context",
    )
    _require(
        _value(model, "native_context"),
        SM121_STORAGE_NATIVE_CONTEXT,
        "native_context",
    )
    _require(
        _value(model, "endpoint"),
        "http://127.0.0.1:30000/v1",
        "endpoint",
    )
    _require(
        _value(model, "weight_size_bytes"),
        SM121_STORAGE_WEIGHT_SIZE_BYTES,
        "weight_size_bytes",
    )
    _require(
        _value(model, "weight_file_count"),
        SM121_STORAGE_WEIGHT_FILE_COUNT,
        "weight_file_count",
    )
    _require(
        _value(model, "sglang_ple_nvme_queue_depth"),
        SM121_STORAGE_QUEUE_DEPTH,
        "sglang_ple_nvme_queue_depth",
    )
    _require(
        _value(model, "sglang_ple_nvme_max_batch_pages"),
        SM121_STORAGE_MAX_BATCH_PAGES,
        "sglang_ple_nvme_max_batch_pages",
    )
    _require(
        _value(model, "sglang_ple_nvme_cache_pages"),
        SM121_STORAGE_CACHE_PAGES,
        "sglang_ple_nvme_cache_pages",
    )
    _require(
        tuple(_value(model, "args") or ()), SM121_STORAGE_ARGS, "args"
    )
    _require(_value(model, "draft_source"), None, "draft_source")
    _require(_value(model, "draft_revision"), None, "draft_revision")
    _require(
        tuple(_value(model, "sglang_source_overlays") or ()),
        (),
        "sglang_source_overlays",
    )
    _require(_value(model, "sglang_ple_mmap"), False, "sglang_ple_mmap")
    _require(
        _value(model, "sglang_ple_omitted"), False, "sglang_ple_omitted"
    )
    _require(
        _value(model, "sglang_ple_cache_mode"), None, "sglang_ple_cache_mode"
    )
    for field in (
        "sglang_ple_cache_marker_digest",
        "sglang_ple_cache_payload_digest",
        "recipe_source",
        "recipe_revision",
    ):
        _require(_value(model, field), None, field)
    request_body = _canonical_request_body(_value(model, "request_body_json"))
    _require(
        request_body,
        {"chat_template_kwargs": {"enable_thinking": False}},
        "request_body_json",
    )
    argument_names = tuple(
        str(argument).split("=", 1)[0]
        for argument in tuple(_value(model, "args") or ())
    )
    if any(name.startswith("--speculative-") for name in argument_names):
        raise SM121StorageCandidateError(
            "native-storage canary forbids speculative decoding arguments"
        )
    forbidden = {
        "--mtp",
        "--dflash",
        "--eagle3",
        "--ple-offload-embedding",
        "--prefill-attention-backend",
        "--decode-attention-backend",
        "--mamba-radix-cache-strategy",
        "--max-mamba-cache-size",
        "--json-model-override-args",
    }
    used = forbidden.intersection(argument_names)
    if used:
        raise SM121StorageCandidateError(
            "native-storage canary contains forbidden argument(s): "
            + ", ".join(sorted(used))
        )


def _case_value(case: Any, name: str) -> object:
    return _value(case, name)


def _require_case(case: Any, expected: Mapping[str, object], index: int) -> None:
    for field, value in expected.items():
        actual = _case_value(case, field)
        if field == "requires" and isinstance(actual, (list, tuple)):
            actual = tuple(actual)
        if actual != value:
            raise SM121StorageCandidateError(
                f"canary suite case {index} field {field} does not match its pin"
            )


def validate_sm121_storage_suite(suite: Any) -> None:
    """Require the two fixed fresh-process admission cases in order."""

    _require(_value(suite, "id"), SM121_STORAGE_SUITE_ID, "suite.id")
    cases = _value(suite, "cases")
    if not isinstance(cases, (list, tuple)) or len(cases) != 2:
        raise SM121StorageCandidateError(
            "SM121 storage canary suite must contain exactly two cases"
        )
    _require_case(
        cases[0],
        {
            "id": "synthetic-exact-answer-v2",
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
        0,
    )
    _require_case(
        cases[1],
        {
            "id": SM121_STORAGE_VARIED_CONTEXT_CASE_ID,
            "kind": "capability",
            "requires": ("chat",),
            "warmups": 0,
            "repetitions": 1,
            "max_output_tokens": 64,
            "temperature": 0.0,
            "concurrency": 1,
            "prompt_repetitions": SM121_STORAGE_VARIED_CONTEXT_RECORDS,
            "max_turns": 1,
        },
        1,
    )


def _lifecycle_issue(
    code: str, message: str, **context: object
) -> dict[str, object]:
    """Build a scalar-only lifecycle finding without copying journal payloads."""

    return {"code": code, "message": message, **context}


def _event_positions(
    events: Sequence[Mapping[str, object]], event_type: str
) -> list[int]:
    return [
        index
        for index, event in enumerate(events)
        if event.get("event") == event_type
    ]


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def sm121_storage_canary_lifecycle_issues(
    events: Sequence[Mapping[str, object]], *, planned_case_ids: Sequence[str]
) -> tuple[dict[str, object], ...]:
    """Return topology violations for one completed fresh-process canary.

    The canary deliberately uses one server per case and its first inference is
    the measured case itself.  A normal benchmark's ``first_request_complete``
    primer is therefore forbidden, rather than merely absent.  The function is
    pure and does not expose request bodies, completions, or other raw journal
    values in its scalar findings.
    """

    issues: list[dict[str, object]] = []
    expected_case_ids = tuple(planned_case_ids)
    if (
        len(expected_case_ids) != SM121_STORAGE_LIFETIME_COUNT
        or any(not _nonempty_string(case_id) for case_id in expected_case_ids)
        or len(set(expected_case_ids)) != SM121_STORAGE_LIFETIME_COUNT
    ):
        issues.append(
            _lifecycle_issue(
                "sm121_storage_invalid_planned_case_order",
                "SM121 storage canary requires two distinct ordered planned cases",
            )
        )
        return tuple(issues)

    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            issues.append(
                _lifecycle_issue(
                    "sm121_storage_non_object_event",
                    "SM121 storage canary journal contains a non-object event",
                    event_index=index,
                )
            )

    starts = _event_positions(events, "run_start")
    if len(starts) != 1:
        issues.append(
            _lifecycle_issue(
                "sm121_storage_run_start_count",
                "SM121 storage canary must have exactly one non-resumed run_start",
                actual_count=len(starts),
            )
        )
    start_index = starts[0] if len(starts) == 1 else None
    if start_index is not None:
        if start_index != 0:
            issues.append(
                _lifecycle_issue(
                    "sm121_storage_run_start_not_first",
                    "SM121 storage canary run_start must be the first journal event",
                    event_index=start_index,
                )
            )
        if events[start_index].get("execution_mode") != SM121_STORAGE_EXECUTION_MODE:
            issues.append(
                _lifecycle_issue(
                    "sm121_storage_execution_mode_mismatch",
                    "SM121 storage canary run_start has the wrong execution mode",
                    event_index=start_index,
                )
            )

    primer_indexes = _event_positions(events, "first_request_complete")
    if primer_indexes:
        issues.append(
            _lifecycle_issue(
                "sm121_storage_unexpected_primer",
                "SM121 storage canary forbids first_request_complete primer events",
                event_indexes=primer_indexes,
            )
        )

    abort_indexes = _event_positions(events, "run_aborted")
    if abort_indexes:
        issues.append(
            _lifecycle_issue(
                "sm121_storage_run_aborted",
                "SM121 storage lifecycle audit requires a completed canary run",
                event_indexes=abort_indexes,
            )
        )

    measurement_starts = _event_positions(events, "measurement_started")
    measurement_completes = _event_positions(events, "measurement_complete")
    completes = _event_positions(events, "run_complete")
    for event_type, positions in (
        ("measurement_started", measurement_starts),
        ("measurement_complete", measurement_completes),
        ("run_complete", completes),
    ):
        if len(positions) != 1:
            issues.append(
                _lifecycle_issue(
                    f"sm121_storage_{event_type}_count",
                    f"SM121 storage canary must have exactly one {event_type}",
                    actual_count=len(positions),
                )
            )

    measurement_start = (
        measurement_starts[0] if len(measurement_starts) == 1 else None
    )
    measurement_complete = (
        measurement_completes[0] if len(measurement_completes) == 1 else None
    )
    run_complete = completes[0] if len(completes) == 1 else None
    if run_complete is not None:
        if run_complete != len(events) - 1:
            issues.append(
                _lifecycle_issue(
                    "sm121_storage_run_complete_not_final",
                    "SM121 storage canary run_complete must be the final journal event",
                    event_index=run_complete,
                )
            )
        if events[run_complete].get("status") != "completed":
            issues.append(
                _lifecycle_issue(
                    "sm121_storage_run_completion_status",
                    "SM121 storage canary run_complete must record completed status",
                    event_index=run_complete,
                )
            )

    ready_positions = _event_positions(events, "server_ready")
    stopped_positions = _event_positions(events, "server_stopped")
    provenance_positions = _event_positions(
        events, SM121_STORAGE_RUNTIME_PROVENANCE_EVENT
    )
    for event_type, positions in (
        ("server_ready", ready_positions),
        ("server_stopped", stopped_positions),
        (SM121_STORAGE_RUNTIME_PROVENANCE_EVENT, provenance_positions),
    ):
        if len(positions) != SM121_STORAGE_LIFETIME_COUNT:
            issues.append(
                _lifecycle_issue(
                    f"sm121_storage_{event_type}_count",
                    f"SM121 storage canary must have exactly two {event_type} events",
                    actual_count=len(positions),
                )
            )

    def positions_by_lifetime(
        positions: Sequence[int], event_type: str
    ) -> dict[int, list[int]]:
        grouped: dict[int, list[int]] = {}
        for index in positions:
            lifetime = events[index].get("fresh_server_lifetime")
            if isinstance(lifetime, bool) or not isinstance(lifetime, int):
                issues.append(
                    _lifecycle_issue(
                        "sm121_storage_invalid_lifetime",
                        f"{event_type} has no integer fresh_server_lifetime",
                        event_index=index,
                    )
                )
                continue
            grouped.setdefault(lifetime, []).append(index)
        return grouped

    ready_by_lifetime = positions_by_lifetime(ready_positions, "server_ready")
    stopped_by_lifetime = positions_by_lifetime(stopped_positions, "server_stopped")
    provenance_by_lifetime = positions_by_lifetime(
        provenance_positions, SM121_STORAGE_RUNTIME_PROVENANCE_EVENT
    )
    for lifetime in range(1, SM121_STORAGE_LIFETIME_COUNT + 1):
        for event_type, grouped in (
            ("server_ready", ready_by_lifetime),
            ("server_stopped", stopped_by_lifetime),
            (SM121_STORAGE_RUNTIME_PROVENANCE_EVENT, provenance_by_lifetime),
        ):
            actual_count = len(grouped.get(lifetime, []))
            if actual_count != 1:
                issues.append(
                    _lifecycle_issue(
                        "sm121_storage_lifetime_event_count",
                        f"lifetime {lifetime} must have exactly one {event_type}",
                        lifetime=lifetime,
                        event_type=event_type,
                        actual_count=actual_count,
                    )
                )
    allowed_lifetimes = set(range(1, SM121_STORAGE_LIFETIME_COUNT + 1))
    for event_type, grouped in (
        ("server_ready", ready_by_lifetime),
        ("server_stopped", stopped_by_lifetime),
        (SM121_STORAGE_RUNTIME_PROVENANCE_EVENT, provenance_by_lifetime),
    ):
        for lifetime, positions in grouped.items():
            if lifetime not in allowed_lifetimes:
                issues.append(
                    _lifecycle_issue(
                        "sm121_storage_unknown_lifetime",
                        f"{event_type} references an unknown server lifetime",
                        lifetime=lifetime,
                        event_indexes=positions,
                    )
                )

    case_event_types = {
        "case_start",
        "request_complete",
        "case_complete",
        "case_failed",
        "case_skipped_adapter_unimplemented",
        "case_skipped_context_limit",
        "case_skipped_unsupported",
    }
    valid_case_event_indexes: set[int] = set()
    previous_stop = measurement_start
    for lifetime, expected_case_id in enumerate(expected_case_ids, start=1):
        ready_entries = ready_by_lifetime.get(lifetime, [])
        stop_entries = stopped_by_lifetime.get(lifetime, [])
        provenance_entries = provenance_by_lifetime.get(lifetime, [])
        if not (
            len(ready_entries) == len(stop_entries) == len(provenance_entries) == 1
        ):
            continue
        ready_index = ready_entries[0]
        stop_index = stop_entries[0]
        provenance_index = provenance_entries[0]
        try:
            validate_sm121_storage_runtime_provenance_event(
                events[provenance_index], fresh_server_lifetime=lifetime
            )
        except SM121StorageCandidateError:
            issues.append(
                _lifecycle_issue(
                    "sm121_storage_invalid_runtime_provenance",
                    "runtime provenance does not match the pinned scalar contract",
                    lifetime=lifetime,
                    event_index=provenance_index,
                )
            )
        if previous_stop is not None and not previous_stop < provenance_index:
            issues.append(
                _lifecycle_issue(
                    "sm121_storage_lifetime_order",
                    "SM121 storage lifetimes must execute in ascending fresh order",
                    lifetime=lifetime,
                )
            )
        if not provenance_index + 1 == ready_index:
            issues.append(
                _lifecycle_issue(
                    "sm121_storage_provenance_order",
                    "runtime provenance must be immediately followed by server_ready",
                    lifetime=lifetime,
                    provenance_index=provenance_index,
                    server_ready_index=ready_index,
                )
            )
        if ready_index >= stop_index:
            issues.append(
                _lifecycle_issue(
                    "sm121_storage_lifetime_stop_order",
                    "server_stopped must follow server_ready in each lifetime",
                    lifetime=lifetime,
                )
            )
            previous_stop = stop_index
            continue
        ready_event = events[ready_index]
        if ready_event.get("backend") != "sglang":
            issues.append(
                _lifecycle_issue(
                    "sm121_storage_ready_backend",
                    "SM121 storage canary server_ready must identify the SGLang backend",
                    lifetime=lifetime,
                    event_index=ready_index,
                )
            )
        if ready_event.get("first_inference_is_case") is not True:
            issues.append(
                _lifecycle_issue(
                    "sm121_storage_first_inference_marker",
                    "server_ready must state that the first inference is the case",
                    lifetime=lifetime,
                    event_index=ready_index,
                )
            )
        if ready_event.get("case_id") != expected_case_id:
            issues.append(
                _lifecycle_issue(
                    "sm121_storage_case_order_mismatch",
                    "server_ready case does not match the ordered canary lifetime",
                    lifetime=lifetime,
                )
            )

        segment_positions = range(ready_index + 1, stop_index)
        case_starts = [
            index for index in segment_positions if events[index].get("event") == "case_start"
        ]
        case_completes = [
            index
            for index in range(ready_index + 1, stop_index)
            if events[index].get("event") == "case_complete"
        ]
        if len(case_starts) != 1:
            issues.append(
                _lifecycle_issue(
                    "sm121_storage_case_start_count",
                    "each fresh server lifetime must start exactly one measured case",
                    lifetime=lifetime,
                    actual_count=len(case_starts),
                )
            )
        if len(case_completes) != 1:
            issues.append(
                _lifecycle_issue(
                    "sm121_storage_case_complete_count",
                    "each fresh server lifetime must complete exactly one measured case",
                    lifetime=lifetime,
                    actual_count=len(case_completes),
                )
            )
        if len(case_starts) == 1 and len(case_completes) == 1:
            case_start = case_starts[0]
            case_complete = case_completes[0]
            start_event = events[case_start]
            complete_event = events[case_complete]
            valid_case_event_indexes.update(
                index
                for index in range(case_start, case_complete + 1)
                if events[index].get("event")
                in {"case_start", "request_complete", "case_complete"}
            )
            attempt_id = start_event.get("attempt_id")
            if not _nonempty_string(attempt_id):
                issues.append(
                    _lifecycle_issue(
                        "sm121_storage_case_attempt_id",
                        "case_start has no non-empty attempt_id",
                        lifetime=lifetime,
                        event_index=case_start,
                    )
                )
            if case_start >= case_complete:
                issues.append(
                    _lifecycle_issue(
                        "sm121_storage_case_event_order",
                        "case_complete must follow case_start",
                        lifetime=lifetime,
                    )
                )
            for label, event, event_index in (
                ("case_start", start_event, case_start),
                ("case_complete", complete_event, case_complete),
            ):
                if event.get("case_id") != expected_case_id:
                    issues.append(
                        _lifecycle_issue(
                            "sm121_storage_case_order_mismatch",
                            f"{label} does not match the ordered canary lifetime",
                            lifetime=lifetime,
                            event_index=event_index,
                        )
                    )
                if event.get("attempt_id") != attempt_id:
                    issues.append(
                        _lifecycle_issue(
                            "sm121_storage_case_attempt_mismatch",
                            f"{label} does not match its case_start attempt",
                            lifetime=lifetime,
                            event_index=event_index,
                        )
                    )
            requests = [
                index
                for index in range(case_start + 1, case_complete)
                if events[index].get("event") == "request_complete"
            ]
            if not requests:
                issues.append(
                    _lifecycle_issue(
                        "sm121_storage_missing_case_request",
                        "each canary lifetime needs at least one measured request",
                        lifetime=lifetime,
                    )
                )
            for request_index in requests:
                request = events[request_index]
                if (
                    request.get("case_id") != expected_case_id
                    or request.get("attempt_id") != attempt_id
                ):
                    issues.append(
                        _lifecycle_issue(
                            "sm121_storage_request_case_mismatch",
                            "request_complete does not match its lifetime case attempt",
                            lifetime=lifetime,
                            event_index=request_index,
                        )
                    )
        previous_stop = stop_index

    if (
        measurement_start is not None
        and ready_by_lifetime.get(1)
        and measurement_start >= ready_by_lifetime[1][0]
    ):
        issues.append(
            _lifecycle_issue(
                "sm121_storage_measurement_start_order",
                "measurement_started must precede the first server lifetime",
            )
        )
    second_stop = stopped_by_lifetime.get(SM121_STORAGE_LIFETIME_COUNT, [])
    if (
        measurement_complete is not None
        and len(second_stop) == 1
        and measurement_complete <= second_stop[0]
    ):
        issues.append(
            _lifecycle_issue(
                "sm121_storage_measurement_complete_order",
                "measurement_complete must follow the second server lifetime",
            )
        )
    if (
        measurement_complete is not None
        and run_complete is not None
        and measurement_complete >= run_complete
    ):
        issues.append(
            _lifecycle_issue(
                "sm121_storage_terminal_order",
                "measurement_complete must precede run_complete",
            )
        )

    for index, event in enumerate(events):
        event_type = event.get("event")
        if event_type in case_event_types and index not in valid_case_event_indexes:
            issues.append(
                _lifecycle_issue(
                    "sm121_storage_case_event_outside_lifetime",
                    "case event is outside its one fresh server lifetime",
                    event_index=index,
                )
            )
        if event_type in {"server_kept", "cleanup_failed"}:
            issues.append(
                _lifecycle_issue(
                    "sm121_storage_unexpected_lifecycle_event",
                    "SM121 storage canary must stop, not retain or fail cleanup of, each server",
                    event_index=index,
                )
            )

    return tuple(issues)


def validate_sm121_storage_image_inspection(
    inspection: Mapping[str, object], *, image: str
) -> dict[str, str]:
    """Validate a Docker inspect object without running Docker.

    The caller supplies a decoded ``docker image inspect`` response.  Keeping
    parsing pure makes the immutable-local-ID rule testable and lets planner
    and runtime use the exact same admission checks.
    """

    _require(image, SM121_STORAGE_LOCAL_IMAGE_TAG, "image")
    image_id = inspection.get("Id")
    if not isinstance(image_id, str) or _IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        raise SM121StorageCandidateError("local image inspect has no sha256 image ID")
    _require(image_id, SM121_STORAGE_LOCAL_IMAGE_ID, "docker image ID")
    repo_tags = inspection.get("RepoTags")
    if not isinstance(repo_tags, list) or SM121_STORAGE_LOCAL_IMAGE_TAG not in repo_tags:
        raise SM121StorageCandidateError("local image tag does not resolve to the candidate")
    _require(inspection.get("Os"), "linux", "docker image OS")
    _require(inspection.get("Architecture"), "arm64", "docker image architecture")
    config = inspection.get("Config")
    if not isinstance(config, Mapping):
        raise SM121StorageCandidateError("local image has no config metadata")
    labels = config.get("Labels")
    if not isinstance(labels, Mapping):
        raise SM121StorageCandidateError("local image has no candidate build labels")
    for field in (
        "ai.sglang.build.commit",
        "org.opencontainers.image.revision",
    ):
        _require(labels.get(field), SM121_STORAGE_SOURCE_TREE, f"image label {field}")
    return {
        "docker_image_id": image_id,
        "platform": SM121_STORAGE_PLATFORM,
        "source_tree": SM121_STORAGE_SOURCE_TREE,
    }
