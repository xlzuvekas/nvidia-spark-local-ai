"""Plan expansion and resumable execution for text-generation benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import base64
import fcntl
import functools
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import struct
import time
from types import SimpleNamespace
from typing import Any, Iterator
import unicodedata
import uuid
import zlib

from .agentic_tools import estimate_agentic_context_tokens, run_agentic_scenario
from .client import (
    concurrent_chat_requests,
    concurrent_multimodal_embedding_requests,
    concurrent_score_requests,
    embedding_request,
    multimodal_embedding_request,
    score_request,
    stream_audio_chat_request,
    stream_chat_request,
    stream_ollama_chat_request,
)
from .execution_admission import model_execution_blocker
from .host_safety import HostSafetyError, HostSafetyWatchdog
from .journal import Journal, content_hash, utc_now, write_json
from .llamacpp_cache_metrics import (
    LlamaCppCacheMetricsError,
    delta_llamacpp_cache_metrics,
    require_llamacpp_cache_delta,
    snapshot_llamacpp_cache_metrics,
)
from .llamacpp_metrics import (
    llamacpp_dflash_requested,
    llamacpp_mtp_requested,
    require_llamacpp_dflash_evidence,
    require_llamacpp_mtp_evidence,
    require_mtp_activity,
    require_speculative_activity,
    snapshot_llamacpp_spec_decode_metrics,
)
from .manifest import (
    VARIED_CONTEXT_NEEDLE_CASE_PREFIX,
    VariedContextNeedleSpec,
    model_spec_to_dict,
    validate_benchmark_selection,
    varied_context_needle_spec,
)
from .memory_ops import (
    MEMORY_OPERATION_CONTEXT_TOKENS,
    MEMORY_OPERATION_LLAMACPP_DIGEST,
    MEMORY_OPERATION_LLAMACPP_REVISION,
    MEMORY_OPERATION_OUTPUT_TOKENS,
    MEMORY_OPERATION_PROTOCOL_DIGEST,
    MEMORY_OPERATION_SCENARIO_IDS,
    MEMORY_OPERATION_SUITE_DESCRIPTION,
    MEMORY_OPERATION_SUITE_ID,
    MEMORY_OPERATION_VARIANT_COUNT,
    MemoryOperationError,
    estimate_memory_operation_context_tokens,
    memory_operation_llamacpp_args,
    require_memory_operation_protocol_digest,
    run_memory_operation_scenario,
)
from .prefix_cache_protocol import (
    PREFIX_CACHE_CONTEXT_TOKENS,
    PREFIX_CACHE_PREFIX_TARGETS,
    PREFIX_CACHE_SUITE_ID,
    prefix_cache_llamacpp_args,
    prefix_cache_steps,
)
from .report import summarize_run
from .runtime import (
    RuntimeErrorWithContext,
    attest_sm121_cache_observability_runtime,
    attest_sm121_cache_observability_static_source,
    capture_server_provenance,
    inspect_sm121_cache_runtime_identity,
    inspect_sm121_cache_source_digests,
    ollama_model_loaded,
    recover_owned_llamacpp,
    recover_owned_sglang,
    recover_owned_vllm,
    save_server_logs,
    request_sm121_cache_observability_zero_hit,
    request_sm121_cache_semantic_turn,
    settle_sm121_cache_observability_metrics,
    start_server,
    validate_llamacpp_artifacts,
)
from .sglang_metrics import (
    SGLangSpeculativeAuditError,
    request_sglang_speculative_audit,
    sglang_nextn_depth,
)
from .sglang_sm121_storage import (
    SM121_STORAGE_EXECUTION_MODE,
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_RUNTIME_PROVENANCE_EVENT,
    SM121_STORAGE_SOURCE_TREE,
    SM121_STORAGE_VARIED_CONTEXT_BUDGET_TOKENS,
    SM121_STORAGE_VARIED_CONTEXT_CASE_ID,
    SM121StorageCandidateError,
    is_sm121_storage_candidate,
    validate_sm121_storage_candidate,
    validate_sm121_storage_image_inspection,
    validate_sm121_storage_runtime_provenance_event,
    validate_sm121_storage_suite,
)
from .sglang_sm121_cache_semantic import (
    SM121_CACHE_SEMANTIC_ARM_ORDER,
    SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID,
    SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID,
    SM121_CACHE_SEMANTIC_CASE_ID,
    SM121_CACHE_SEMANTIC_COLD_INPUT_MAX_TOKENS,
    SM121_CACHE_SEMANTIC_COLD_INPUT_MIN_TOKENS,
    SM121_CACHE_SEMANTIC_EXECUTION_MODE,
    SM121_CACHE_SEMANTIC_LOCAL_LIFETIME_ORDER,
    SM121_CACHE_SEMANTIC_METRIC_FIELDS,
    SM121_CACHE_SEMANTIC_PAIR_BINDING_SCHEMA_VERSION,
    SM121_CACHE_SEMANTIC_RUNTIME_ATTESTATION_EVENT,
    SM121_CACHE_SEMANTIC_QUALITY_CASE_ID,
    SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
    SM121_CACHE_SEMANTIC_STATIC_ATTESTATION_EVENT,
    SM121_CACHE_SEMANTIC_TURN_OBSERVATION_EVENT,
    SM121_CACHE_SEMANTIC_TURN_ORDER,
    SM121_CACHE_SEMANTIC_SUITE_ID,
    SM121CacheSemanticError,
    derive_sm121_cache_semantic_turn_admission,
    is_sm121_cache_semantic_candidate,
    is_sm121_cache_semantic_plan,
    sm121_cache_semantic_arm,
    sm121_cache_semantic_case_metadata,
    sm121_cache_semantic_cache_off_receipt_sha256,
    sm121_cache_semantic_lifecycle_issues,
    sm121_cache_semantic_pair_binding_sha256,
    sm121_cache_semantic_pair_instance_sha256,
    validate_sm121_cache_semantic_runtime_attestation_event,
    validate_sm121_cache_semantic_static_attestation_event,
    validate_sm121_cache_semantic_candidate,
    validate_sm121_cache_semantic_pair,
    validate_sm121_cache_semantic_pair_binding,
    validate_sm121_cache_semantic_suite,
    validate_sm121_cache_semantic_turn_event,
)
from .sglang_sm121_cache_performance import (
    SM121_CACHE_PERFORMANCE_ARM_ORDER,
    SM121_CACHE_PERFORMANCE_CAMPAIGN_ID,
    SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID,
    SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID,
    SM121_CACHE_PERFORMANCE_CASE_ID,
    SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S,
    SM121_CACHE_PERFORMANCE_COLD_INPUT_MAX_TOKENS,
    SM121_CACHE_PERFORMANCE_COLD_INPUT_MIN_TOKENS,
    SM121_CACHE_PERFORMANCE_EXECUTION_MODE,
    SM121_CACHE_PERFORMANCE_METRIC_FIELDS,
    SM121_CACHE_PERFORMANCE_PAIR_BINDING_SCHEMA_VERSION,
    SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S,
    SM121_CACHE_PERFORMANCE_QUALITY_CASE_ID,
    SM121_CACHE_PERFORMANCE_QUALITY_ITEM_COUNT,
    SM121_CACHE_PERFORMANCE_RUNTIME_EVENT,
    SM121_CACHE_PERFORMANCE_STATIC_EVENT,
    SM121_CACHE_PERFORMANCE_SUITE_ID,
    SM121_CACHE_PERFORMANCE_TIMED_TURNS,
    SM121_CACHE_PERFORMANCE_TURN_EVENT,
    SM121CachePerformanceError,
    derive_sm121_cache_performance_turn_admission,
    is_sm121_cache_performance_candidate,
    is_sm121_cache_performance_plan,
    score_sm121_cache_performance_campaign,
    sm121_cache_performance_arm,
    sm121_cache_performance_pair_binding_sha256,
    sm121_cache_performance_pair_instance_sha256,
    validate_sm121_cache_performance_candidate,
    validate_sm121_cache_performance_pair,
    validate_sm121_cache_performance_pair_binding,
    validate_sm121_cache_performance_runtime_event,
    validate_sm121_cache_performance_static_event,
    validate_sm121_cache_performance_suite,
    validate_sm121_cache_performance_turn_event,
)
from .sglang_sm121_cache_observability import (
    SM121_CACHE_OBSERVABILITY_CACHED_SERIES,
    SM121_CACHE_OBSERVABILITY_EXECUTION_MODE,
    SM121_CACHE_OBSERVABILITY_METRIC_FIELDS,
    SM121_CACHE_OBSERVABILITY_SUITE_ID,
    SM121_CACHE_RUNTIME_ATTESTATION_EVENT,
    SM121_CACHE_STATIC_ATTESTATION_EVENT,
    SM121_CACHE_ZERO_HIT_CASE_ID,
    SM121_CACHE_ZERO_HIT_EVENT,
    SM121_CACHE_ZERO_HIT_REQUEST_CONTRACT_SHA256,
    SM121CacheObservabilityError,
    derive_sm121_cache_zero_hit_admission,
    is_sm121_cache_observability_plan,
    validate_sm121_cache_observability_candidate,
    validate_sm121_cache_observability_suite,
    validate_sm121_cache_zero_hit_event,
)
from .telemetry import TelemetrySampler
from .vllm_metrics import snapshot_vllm_spec_decode_metrics


class PreflightError(RuntimeError):
    pass


_MULTI_HOP_FAILURE_MESSAGE = "multi-hop case failed; error details omitted"
_VARIED_CONTEXT_NEEDLE_FAILURE_MESSAGE = (
    "varied-context needle case failed; error details omitted"
)
_PREFIX_CACHE_FAILURE_MESSAGE = "prefix-cache case failed; error details omitted"
_SM121_STORAGE_QUALITY_GATE_FAILURE_MESSAGE = (
    "SM121 storage quality gate failed; long-context lifetime was not started"
)
_SM121_CACHE_OBSERVABILITY_FAILURE_MESSAGE = (
    "SM121 cache observability request failed; details omitted"
)
_SM121_CACHE_SEMANTIC_FAILURE_MESSAGE = (
    "SM121 cache-policy semantic request failed; details omitted"
)
_SM121_CACHE_PERFORMANCE_FAILURE_MESSAGE = (
    "SM121 cache-policy performance request failed; details omitted"
)
# These are synthetic fixed protocol strings, kept private to the executor.
# They are never written to journals, reports, or evidence. The repeated
# ledger token was locally tokenizer-checked to place T0 in the 32--48 KiB
# cold-input window once Qwen's chat template is applied.
_SM121_CACHE_SEMANTIC_LEDGER_WORD = "shared-ledger-entry "
_SM121_CACHE_SEMANTIC_LEDGER_REPETITIONS = 10_240
_SM121_CACHE_SEMANTIC_EXPECTED_RESPONSES = (
    "SEMANTIC-CACHE-T0-17",
    "SEMANTIC-CACHE-T1-29",
    "SEMANTIC-CACHE-T2-43",
)


class MultiHopNeedleError(RuntimeError):
    """A public-safe failure for a generated multi-hop needle case."""

    def __init__(self) -> None:
        super().__init__(_MULTI_HOP_FAILURE_MESSAGE)


class VariedContextNeedleError(RuntimeError):
    """A public-safe failure for a generated varied-context needle case."""

    def __init__(self) -> None:
        super().__init__(_VARIED_CONTEXT_NEEDLE_FAILURE_MESSAGE)


class SM121StorageQualityGateError(RuntimeError):
    """Public-safe failure when the first SM121 canary gate is not clean."""

    def __init__(self) -> None:
        super().__init__(_SM121_STORAGE_QUALITY_GATE_FAILURE_MESSAGE)


class SM121CacheObservabilityRequestError(RuntimeError):
    """Public-safe failure for the B0 non-streaming cache observation."""

    def __init__(self) -> None:
        super().__init__(_SM121_CACHE_OBSERVABILITY_FAILURE_MESSAGE)


class SM121CacheSemanticRequestError(RuntimeError):
    """A public-safe failure for the paired cache-policy semantic probe."""

    def __init__(self) -> None:
        super().__init__(_SM121_CACHE_SEMANTIC_FAILURE_MESSAGE)


class SM121CachePerformanceRequestError(RuntimeError):
    """A public-safe failure for the cache-policy performance probe."""

    def __init__(self) -> None:
        super().__init__(_SM121_CACHE_PERFORMANCE_FAILURE_MESSAGE)


class PrefixCacheError(RuntimeError):
    """A public-safe failure for the native llama.cpp prompt-KV protocol."""

    def __init__(self) -> None:
        super().__init__(_PREFIX_CACHE_FAILURE_MESSAGE)


def _command_output(command: list[str]) -> str | None:
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    return result.stdout.strip() if result.returncode == 0 else None


def _image_digest(image: str | None) -> str | None:
    if not image:
        return None
    output = _command_output(
        ["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"]
    )
    if not output:
        return None
    try:
        digests = json.loads(output)
    except json.JSONDecodeError:
        return None
    return digests[0] if digests else None


def _sm121_storage_image_identity(model: Any) -> dict[str, str]:
    """Inspect the tagged local candidate before freezing its Docker ID."""

    image = str(getattr(model, "image", "") or "")
    output = _command_output(
        ["docker", "image", "inspect", image, "--format", "{{json .}}"]
    )
    if not output:
        raise RuntimeError("Could not inspect the local SM121 storage image")
    try:
        inspection = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Local SM121 storage image inspection was not valid JSON"
        ) from error
    if not isinstance(inspection, dict):
        raise RuntimeError("Local SM121 storage image inspection was not an object")
    try:
        return validate_sm121_storage_image_inspection(inspection, image=image)
    except SM121StorageCandidateError as error:
        raise RuntimeError(str(error)) from error


def _host_snapshot() -> dict[str, Any]:
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, raw = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            meminfo[f"{name.lower()}_kib"] = int(raw.strip().split()[0])
    return {
        "captured_at": utc_now(),
        "uname": _command_output(["uname", "-a"]),
        "python": _command_output(["python3", "--version"]),
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "docker": _command_output(["docker", "version", "--format", "{{.Server.Version}}"]),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_status": _command_output(["git", "status", "--short"]),
        **meminfo,
    }


def _canonical_case(
    model: dict[str, Any],
    case: dict[str, Any],
    *,
    protocol_digest: str | None = None,
) -> dict[str, Any]:
    payload = {"model": model, "case": case}
    if protocol_digest is not None:
        payload["protocol_digest"] = protocol_digest
    return {**case, "case_id": f"{case['id']}--{content_hash(payload, 12)}"}


def create_plan(
    *,
    model: Any,
    suite: Any,
    results_root: Path,
    models_path: Path,
    suite_path: Path,
    allow_sm121_storage_canary: bool = False,
    allow_sm121_cache_semantic_canary: bool = False,
    allow_sm121_cache_performance: bool = False,
    run_label: str | None = None,
) -> Path:
    semantic_candidate = is_sm121_cache_semantic_candidate(model)
    performance_candidate = is_sm121_cache_performance_candidate(model)
    storage_candidate = (
        is_sm121_storage_candidate(model)
        and not semantic_candidate
        and not performance_candidate
    )
    blocker = model_execution_blocker(
        model,
        allow_sm121_storage_canary=allow_sm121_storage_canary,
        allow_sm121_cache_semantic_canary=allow_sm121_cache_semantic_canary,
        allow_sm121_cache_performance=allow_sm121_cache_performance,
    )
    if blocker is not None:
        raise RuntimeError(blocker)
    if str(getattr(model, "support_status", "")) == "incompatible":
        raise RuntimeError("Incompatible model profiles cannot be planned")
    direct_commands = {
        "transformers": "diffusion-direct",
        "trtllm": "trtllm-direct",
    }
    direct_command = direct_commands.get(str(model.backend))
    if direct_command:
        raise RuntimeError(
            f"{model.backend} direct profiles require the {direct_command} command"
        )
    validate_benchmark_selection(model, suite, context="plan")
    if semantic_candidate:
        try:
            validate_sm121_cache_semantic_candidate(model)
        except SM121CacheSemanticError as error:
            raise RuntimeError(str(error)) from error
    elif performance_candidate:
        try:
            validate_sm121_cache_performance_candidate(model)
        except SM121CachePerformanceError as error:
            raise RuntimeError(str(error)) from error
    elif storage_candidate:
        try:
            validate_sm121_storage_candidate(model)
        except SM121StorageCandidateError as error:
            raise RuntimeError(str(error)) from error
    model_data = model_spec_to_dict(model)
    suite_data = asdict(suite)
    protocol_digest = suite_data.get("protocol_digest")
    if protocol_digest is None:
        # Preserve the frozen fingerprints and case identifiers of unrelated
        # suites that predate protocol-level provenance binding.
        suite_data.pop("protocol_digest", None)
    cases = [
        _canonical_case(
            model_data,
            case,
            protocol_digest=protocol_digest,
        )
        for case in suite_data["cases"]
    ]
    resolved_image = _image_digest(model.image)
    if storage_candidate or semantic_candidate or performance_candidate:
        local_image = _sm121_storage_image_identity(model)
        resolved: dict[str, Any] = {
            "image_digest": resolved_image,
            "local_image": local_image,
        }
    else:
        if model.image_digest and (
            not resolved_image or not resolved_image.endswith("@" + model.image_digest)
        ):
            raise RuntimeError(
                f"Local image digest for {model.image} does not match manifest {model.image_digest}"
            )
        resolved = {"image_digest": resolved_image}
    if str(model.backend) == "llamacpp":
        resolved["llamacpp"] = validate_llamacpp_artifacts(
            model, workspace=models_path.resolve().parents[1]
        )
    fingerprint = content_hash(
        {"model": model_data, "suite": suite_data, "resolved": resolved}
    )
    if run_label is not None and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_label) is None:
        raise RuntimeError("Frozen run label is invalid")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label_suffix = f"-{run_label}" if run_label is not None else ""
    run_dir = results_root / f"{stamp}-{model.id}-{suite.id}-{fingerprint[:8]}{label_suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema_version": 2,
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "run_nonce": uuid.uuid4().hex,
        "models_manifest": str(models_path),
        "suite_manifest": str(suite_path),
        "model": model_data,
        "suite": {**suite_data, "cases": cases},
        "resolved": resolved,
        "host_at_plan": _host_snapshot(),
    }
    plan["integrity_hash"] = content_hash(plan, 64)
    write_json(run_dir / "plan.json", plan)
    write_json(run_dir / "inventory.json", _host_snapshot())
    return run_dir


def create_sm121_storage_canary_plan(
    *, model: Any, suite: Any, results_root: Path, models_path: Path, suite_path: Path
) -> Path:
    """Freeze the one pre-admission SM121 storage canary plan.

    This is deliberately separate from the ordinary plan command.  It is the
    only caller permitted to get past the candidate execution tombstone, and
    it still freezes the exact local Docker ID before work begins.
    """

    if not is_sm121_storage_candidate(model):
        raise RuntimeError("The dedicated canary requires the SM121 storage profile")
    try:
        validate_sm121_storage_candidate(model)
        validate_sm121_storage_suite(suite)
    except SM121StorageCandidateError as error:
        raise RuntimeError(str(error)) from error
    return create_plan(
        model=model,
        suite=suite,
        results_root=results_root,
        models_path=models_path,
        suite_path=suite_path,
        allow_sm121_storage_canary=True,
    )


def create_sm121_cache_observability_plan(
    *, model: Any, suite: Any, results_root: Path, models_path: Path, suite_path: Path
) -> Path:
    """Freeze the separate cache-off B0 plan without admitting generic serving."""

    if not is_sm121_cache_observability_plan(model, suite):
        raise RuntimeError(
            "The dedicated cache-observability canary requires its exact SM121 profile and suite"
        )
    try:
        validate_sm121_cache_observability_candidate(model)
        validate_sm121_cache_observability_suite(suite)
    except SM121CacheObservabilityError as error:
        raise RuntimeError(str(error)) from error
    return create_plan(
        model=model,
        suite=suite,
        results_root=results_root,
        models_path=models_path,
        suite_path=suite_path,
        allow_sm121_storage_canary=True,
    )


def create_sm121_cache_semantic_pair_plans(
    *,
    cache_off_model: Any,
    cache_on_model: Any,
    suite: Any,
    results_root: Path,
    models_path: Path,
    suite_path: Path,
) -> tuple[Path, Path]:
    """Freeze the two semantic-cache arm plans in fixed B-then-A order.

    This only freezes the profiles.  The paired executor is the sole path
    that may serve either plan, and it runs B completely before it permits A.
    Keeping the plans independent lets the normal scalar exporter validate
    each arm without carrying any prompt/token material between directories.
    """

    if (
        str(getattr(cache_off_model, "id", ""))
        != SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID
        or str(getattr(cache_on_model, "id", ""))
        != SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID
    ):
        raise RuntimeError("The semantic canary requires the exact B-then-A profiles")
    try:
        validate_sm121_cache_semantic_pair(cache_off_model, cache_on_model)
        validate_sm121_cache_semantic_suite(suite)
    except SM121CacheSemanticError as error:
        raise RuntimeError(str(error)) from error
    cache_off_run = create_plan(
        model=cache_off_model,
        suite=suite,
        results_root=results_root,
        models_path=models_path,
        suite_path=suite_path,
        allow_sm121_cache_semantic_canary=True,
    )
    try:
        cache_on_run = create_plan(
            model=cache_on_model,
            suite=suite,
            results_root=results_root,
            models_path=models_path,
            suite_path=suite_path,
            allow_sm121_cache_semantic_canary=True,
        )
    except BaseException:
        # The B plan is harmless and intentionally left as a frozen, ignored
        # record if creating A fails; no server has started at this point.
        raise
    _bind_sm121_cache_semantic_pair_plans(cache_off_run, cache_on_run)
    return cache_off_run, cache_on_run


def _bind_sm121_cache_semantic_pair_plans(
    cache_off_run: Path, cache_on_run: Path
) -> None:
    """Add one deterministic, non-sensitive cross-plan binding to both arms."""

    plans: dict[str, tuple[Path, dict[str, Any]]] = {}
    for arm, run_dir in zip(SM121_CACHE_SEMANTIC_ARM_ORDER, (cache_off_run, cache_on_run)):
        try:
            plan = json.loads((run_dir / "plan.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Could not bind the frozen semantic-cache plans") from error
        fingerprint = plan.get("fingerprint") if isinstance(plan, dict) else None
        if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{16}", fingerprint) is None:
            raise RuntimeError("Frozen semantic-cache plan fingerprint is invalid")
        plans[arm] = (run_dir, plan)
    try:
        pair_instance_sha256 = sm121_cache_semantic_pair_instance_sha256(
            plans[SM121_CACHE_SEMANTIC_ARM_ORDER[0]][1].get("run_nonce"),
            plans[SM121_CACHE_SEMANTIC_ARM_ORDER[1]][1].get("run_nonce"),
        )
    except SM121CacheSemanticError as error:
        raise RuntimeError("Frozen semantic-cache pair instance is invalid") from error
    for index, arm in enumerate(SM121_CACHE_SEMANTIC_ARM_ORDER):
        run_dir, plan = plans[arm]
        peer_arm = SM121_CACHE_SEMANTIC_ARM_ORDER[1 - index]
        model = plan.get("model")
        suite = plan.get("suite")
        if type(model) is not dict or type(suite) is not dict:
            raise RuntimeError("Frozen semantic-cache plan has invalid binding inputs")
        binding: dict[str, object] = {
            "schema_version": SM121_CACHE_SEMANTIC_PAIR_BINDING_SCHEMA_VERSION,
            "suite_id": SM121_CACHE_SEMANTIC_SUITE_ID,
            "execution_mode": SM121_CACHE_SEMANTIC_EXECUTION_MODE,
            "arm": arm,
            "profile_id": model.get("id"),
            "arm_order": list(SM121_CACHE_SEMANTIC_ARM_ORDER),
            "local_lifetime_order": list(SM121_CACHE_SEMANTIC_LOCAL_LIFETIME_ORDER),
            "quality_case_id": SM121_CACHE_SEMANTIC_QUALITY_CASE_ID,
            "semantic_case_id": SM121_CACHE_SEMANTIC_CASE_ID,
            "semantic_case_metadata": sm121_cache_semantic_case_metadata(),
            "peer_plan_fingerprint": plans[peer_arm][1]["fingerprint"],
            "pair_instance_sha256": pair_instance_sha256,
        }
        try:
            binding["pair_binding_sha256"] = sm121_cache_semantic_pair_binding_sha256(
                binding
            )
            validate_sm121_cache_semantic_pair_binding(
                binding,
                model,
                suite,
                peer_plan_fingerprint=plans[peer_arm][1]["fingerprint"],
            )
        except SM121CacheSemanticError as error:
            raise RuntimeError("Frozen semantic-cache pair binding is invalid") from error
        plan["semantic_pair"] = binding
        plan["integrity_hash"] = content_hash(
            {key: value for key, value in plan.items() if key != "integrity_hash"},
            64,
        )
        write_json(run_dir / "plan.json", plan)


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {key: _plain(item) for key, item in vars(value).items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _load_sm121_storage_canary_plan(run_dir: Path) -> tuple[dict[str, Any], Any, Any]:
    """Load and authenticate a newly frozen native-storage canary plan."""

    try:
        plan = json.loads((run_dir / "plan.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError("SM121 storage canary plan is unavailable or invalid") from error
    if type(plan) is not dict or int(plan.get("schema_version", 0)) != 2:
        raise PreflightError("SM121 storage canary plan schema is invalid")
    model_data = plan.get("model")
    suite_data = plan.get("suite")
    resolved = plan.get("resolved")
    if (
        type(model_data) is not dict
        or type(suite_data) is not dict
        or type(resolved) is not dict
        or type(suite_data.get("cases")) is not list
    ):
        raise PreflightError("SM121 storage canary plan has invalid core fields")
    suite_without_case_ids = {
        **suite_data,
        "cases": [
            {key: value for key, value in case.items() if key != "case_id"}
            for case in suite_data["cases"]
            if type(case) is dict
        ],
    }
    if len(suite_without_case_ids["cases"]) != len(suite_data["cases"]):
        raise PreflightError("SM121 storage canary case records are invalid")
    expected_fingerprint = content_hash(
        {"model": model_data, "suite": suite_without_case_ids, "resolved": resolved}
    )
    integrity_hash = plan.get("integrity_hash")
    integrity_payload = {key: value for key, value in plan.items() if key != "integrity_hash"}
    integrity_valid = isinstance(integrity_hash, str) and (
        content_hash(integrity_payload, len(integrity_hash)) == integrity_hash
    )
    if not integrity_valid or plan.get("fingerprint") != expected_fingerprint:
        raise PreflightError("SM121 storage canary plan fingerprint is invalid")
    protocol_digest = suite_data.get("protocol_digest")
    for case in suite_data["cases"]:
        assert isinstance(case, dict)
        case_without_id = {key: value for key, value in case.items() if key != "case_id"}
        expected_case_id = _canonical_case(
            model_data, case_without_id, protocol_digest=protocol_digest
        )["case_id"]
        if case.get("case_id") != expected_case_id:
            raise PreflightError("SM121 storage canary case identity is invalid")
    model = _namespace(model_data)
    suite = _namespace(suite_data)
    try:
        validate_sm121_storage_candidate(model)
        validate_sm121_storage_suite(suite)
    except SM121StorageCandidateError as error:
        raise PreflightError(str(error)) from error
    local_image = resolved.get("local_image")
    if (
        type(local_image) is not dict
        or set(local_image) != {"docker_image_id", "platform", "source_tree"}
        or local_image.get("docker_image_id") != SM121_STORAGE_LOCAL_IMAGE_ID
        or local_image.get("platform") != SM121_STORAGE_PLATFORM
        or local_image.get("source_tree") != SM121_STORAGE_SOURCE_TREE
    ):
        raise PreflightError("SM121 storage canary local image identity is invalid")
    run_nonce = plan.get("run_nonce")
    if not isinstance(run_nonce, str) or re.fullmatch(r"[0-9a-f]{32}", run_nonce) is None:
        raise PreflightError("SM121 storage canary run nonce is invalid")
    model.resolved_local_image_id = local_image["docker_image_id"]
    model.run_identity = f"{plan['fingerprint']}-{run_nonce}"
    model.storage_canary_authorized = True
    return plan, model, suite


def _load_sm121_cache_observability_plan(
    run_dir: Path,
) -> tuple[dict[str, Any], Any, Any]:
    """Load and authenticate a newly frozen cache-off B0 plan."""

    try:
        plan = json.loads((run_dir / "plan.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError("SM121 cache-observability plan is unavailable or invalid") from error
    if type(plan) is not dict or int(plan.get("schema_version", 0)) != 2:
        raise PreflightError("SM121 cache-observability plan schema is invalid")
    model_data = plan.get("model")
    suite_data = plan.get("suite")
    resolved = plan.get("resolved")
    if (
        type(model_data) is not dict
        or type(suite_data) is not dict
        or type(resolved) is not dict
        or type(suite_data.get("cases")) is not list
    ):
        raise PreflightError("SM121 cache-observability plan has invalid core fields")
    suite_without_case_ids = {
        **suite_data,
        "cases": [
            {key: value for key, value in case.items() if key != "case_id"}
            for case in suite_data["cases"]
            if type(case) is dict
        ],
    }
    if len(suite_without_case_ids["cases"]) != len(suite_data["cases"]):
        raise PreflightError("SM121 cache-observability case records are invalid")
    expected_fingerprint = content_hash(
        {"model": model_data, "suite": suite_without_case_ids, "resolved": resolved}
    )
    integrity_hash = plan.get("integrity_hash")
    integrity_payload = {key: value for key, value in plan.items() if key != "integrity_hash"}
    integrity_valid = isinstance(integrity_hash, str) and (
        content_hash(integrity_payload, len(integrity_hash)) == integrity_hash
    )
    if not integrity_valid or plan.get("fingerprint") != expected_fingerprint:
        raise PreflightError("SM121 cache-observability plan fingerprint is invalid")
    protocol_digest = suite_data.get("protocol_digest")
    for case in suite_data["cases"]:
        assert isinstance(case, dict)
        case_without_id = {key: value for key, value in case.items() if key != "case_id"}
        expected_case_id = _canonical_case(
            model_data, case_without_id, protocol_digest=protocol_digest
        )["case_id"]
        if case.get("case_id") != expected_case_id:
            raise PreflightError("SM121 cache-observability case identity is invalid")
    model = _namespace(model_data)
    suite = _namespace(suite_data)
    try:
        if not is_sm121_cache_observability_plan(model, suite):
            raise SM121CacheObservabilityError("B0 plan selector is invalid")
        validate_sm121_cache_observability_candidate(model)
        validate_sm121_cache_observability_suite(suite)
    except SM121CacheObservabilityError as error:
        raise PreflightError(str(error)) from error
    local_image = resolved.get("local_image")
    if (
        type(local_image) is not dict
        or set(local_image) != {"docker_image_id", "platform", "source_tree"}
        or local_image.get("docker_image_id") != SM121_STORAGE_LOCAL_IMAGE_ID
        or local_image.get("platform") != SM121_STORAGE_PLATFORM
        or local_image.get("source_tree") != SM121_STORAGE_SOURCE_TREE
    ):
        raise PreflightError("SM121 cache-observability local image identity is invalid")
    run_nonce = plan.get("run_nonce")
    if not isinstance(run_nonce, str) or re.fullmatch(r"[0-9a-f]{32}", run_nonce) is None:
        raise PreflightError("SM121 cache-observability run nonce is invalid")
    model.resolved_local_image_id = local_image["docker_image_id"]
    model.run_identity = f"{plan['fingerprint']}-{run_nonce}"
    model.storage_canary_authorized = True
    return plan, model, suite


def _load_sm121_cache_semantic_plan(
    run_dir: Path,
) -> tuple[dict[str, Any], Any, Any]:
    """Load one immutable arm plan; pair binding is checked by the controller."""

    try:
        plan = json.loads((run_dir / "plan.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError("SM121 semantic-cache plan is unavailable or invalid") from error
    if type(plan) is not dict or int(plan.get("schema_version", 0)) != 2:
        raise PreflightError("SM121 semantic-cache plan schema is invalid")
    model_data = plan.get("model")
    suite_data = plan.get("suite")
    resolved = plan.get("resolved")
    if (
        type(model_data) is not dict
        or type(suite_data) is not dict
        or type(resolved) is not dict
        or type(suite_data.get("cases")) is not list
    ):
        raise PreflightError("SM121 semantic-cache plan has invalid core fields")
    suite_without_case_ids = {
        **suite_data,
        "cases": [
            {key: value for key, value in case.items() if key != "case_id"}
            for case in suite_data["cases"]
            if type(case) is dict
        ],
    }
    if len(suite_without_case_ids["cases"]) != len(suite_data["cases"]):
        raise PreflightError("SM121 semantic-cache case records are invalid")
    expected_fingerprint = content_hash(
        {"model": model_data, "suite": suite_without_case_ids, "resolved": resolved}
    )
    integrity_hash = plan.get("integrity_hash")
    integrity_payload = {
        key: value for key, value in plan.items() if key != "integrity_hash"
    }
    integrity_valid = isinstance(integrity_hash, str) and (
        content_hash(integrity_payload, len(integrity_hash)) == integrity_hash
    )
    if not integrity_valid or plan.get("fingerprint") != expected_fingerprint:
        raise PreflightError("SM121 semantic-cache plan fingerprint is invalid")
    protocol_digest = suite_data.get("protocol_digest")
    for case in suite_data["cases"]:
        assert isinstance(case, dict)
        case_without_id = {key: value for key, value in case.items() if key != "case_id"}
        expected_case_id = _canonical_case(
            model_data, case_without_id, protocol_digest=protocol_digest
        )["case_id"]
        if case.get("case_id") != expected_case_id:
            raise PreflightError("SM121 semantic-cache case identity is invalid")
    model = _namespace(model_data)
    suite = _namespace(suite_data)
    try:
        if not is_sm121_cache_semantic_plan(model, suite):
            raise SM121CacheSemanticError("semantic-cache plan selector is invalid")
        validate_sm121_cache_semantic_candidate(model)
        validate_sm121_cache_semantic_suite(suite)
    except SM121CacheSemanticError as error:
        raise PreflightError(str(error)) from error
    local_image = resolved.get("local_image")
    if (
        type(local_image) is not dict
        or set(local_image) != {"docker_image_id", "platform", "source_tree"}
        or local_image.get("docker_image_id") != SM121_STORAGE_LOCAL_IMAGE_ID
        or local_image.get("platform") != SM121_STORAGE_PLATFORM
        or local_image.get("source_tree") != SM121_STORAGE_SOURCE_TREE
    ):
        raise PreflightError("SM121 semantic-cache local image identity is invalid")
    run_nonce = plan.get("run_nonce")
    if not isinstance(run_nonce, str) or re.fullmatch(r"[0-9a-f]{32}", run_nonce) is None:
        raise PreflightError("SM121 semantic-cache run nonce is invalid")
    pair = plan.get("semantic_pair")
    try:
        validate_sm121_cache_semantic_pair_binding(pair, model, suite)
    except SM121CacheSemanticError as error:
        raise PreflightError("SM121 semantic-cache plan binding is invalid") from error
    model.resolved_local_image_id = local_image["docker_image_id"]
    model.run_identity = f"{plan['fingerprint']}-{run_nonce}"
    model.cache_semantic_canary_authorized = True
    model.semantic_pair_binding = pair
    return plan, model, suite


def _preflight(model: SimpleNamespace) -> None:
    running = _command_output(["docker", "ps", "--format", "{{.Names}}"])
    if running is None:
        raise PreflightError("Could not verify running Docker containers")
    backend = str(getattr(model, "backend", ""))
    managed_name = (
        f"sparkbench-{backend}" if backend in {"sglang", "vllm"} else None
    )
    containers = [
        name for name in running.splitlines() if name != managed_name
    ]
    if containers:
        raise PreflightError("Unrelated containers are running: " + ", ".join(containers))
    compute_apps = None
    for attempt in range(11):
        compute_apps = _command_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ]
        )
        if compute_apps is None:
            raise PreflightError("Could not verify active GPU compute processes")
        if not compute_apps:
            break
        if attempt < 10:
            time.sleep(0.5)
    if compute_apps:
        raise PreflightError("Unrelated GPU compute processes are active: " + compute_apps)
    available_kib = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available_kib = int(line.split()[1])
            break
    required_gib = float(getattr(model, "estimated_ram_gib", 0) or 0) + 8
    if available_kib / 1024**2 < required_gib:
        raise PreflightError(
            f"Only {available_kib / 1024**2:.1f} GiB available; model plus reserve needs {required_gib:.1f} GiB"
        )


def _host_safety_watchdog(model: SimpleNamespace) -> HostSafetyWatchdog | None:
    thresholds = {
        "min_memavailable_gib": getattr(
            model, "host_safety_min_memavailable_gib", None
        ),
        "max_swap_growth_mib": getattr(
            model, "host_safety_max_swap_growth_mib", None
        ),
        "max_starting_swap_mib": getattr(
            model, "host_safety_max_starting_swap_mib", None
        ),
    }
    configured = {
        name: value for name, value in thresholds.items() if value is not None
    }
    if not configured:
        return None
    if len(configured) != len(thresholds):
        raise PreflightError(
            "Frozen host-safety thresholds must be configured together"
        )
    if str(getattr(model, "backend", "")) != "sglang":
        raise PreflightError(
            "Frozen host-safety thresholds are supported only for SGLang"
        )
    return HostSafetyWatchdog(**configured)


def _record_host_safety_breach(
    journal: Journal, error: HostSafetyError, *, stage: str
) -> None:
    events = journal.events()
    last_start = max(
        (
            index
            for index, event in enumerate(events)
            if event.get("event") == "run_start"
        ),
        default=-1,
    )
    if any(
        index > last_start and event.get("event") == "host_safety_breach"
        for index, event in enumerate(events)
    ):
        return
    sample = error.sample
    journal.append(
        {
            "event": "host_safety_breach",
            "stage": stage,
            "code": error.code,
            "observed_kib": error.observed_kib,
            "limit_kib": error.limit_kib,
            "starting_swap_used_kib": error.starting_swap_used_kib,
            "memavailable_kib": (
                sample.memavailable_kib if sample is not None else None
            ),
            "swap_used_kib": sample.swap_used_kib if sample is not None else None,
        }
    )


def _record_host_safety_interrupt_failure(
    journal: Journal, watchdog: HostSafetyWatchdog, *, stage: str
) -> None:
    error = watchdog.abort_callback_error
    if error is None:
        return
    events = journal.events()
    last_start = max(
        (
            index
            for index, event in enumerate(events)
            if event.get("event") == "run_start"
        ),
        default=-1,
    )
    if any(
        index > last_start
        and event.get("event") == "host_safety_interrupt_failed"
        for index, event in enumerate(events)
    ):
        return
    journal.append(
        {
            "event": "host_safety_interrupt_failed",
            "stage": stage,
            "error_type": type(error).__name__,
        }
    )


def _retry_host_safety_interrupt_if_needed(
    server: Any, watchdog: HostSafetyWatchdog
) -> None:
    """Retry only a failed watchdog callback before ordinary owned cleanup.

    A successful callback has already stopped the container, so a second
    Docker stop would race with final removal. Retrying an actual callback
    failure preserves the host-safety fallback for the narrow error path.
    """

    if watchdog.tripped and watchdog.abort_callback_error is not None:
        server.interrupt_owned()


def _needle(nonce: str) -> str:
    return "SPARK-" + hashlib.sha256(nonce.encode()).hexdigest()[:10].upper()


def _multi_hop_path(nonce: str, path_id: str) -> tuple[str, str, str]:
    """Return one nonce-derived source-to-relay-to-final relation path."""

    source = "SPARK-SOURCE-" + hashlib.sha256(
        f"{nonce}:{path_id}:source".encode()
    ).hexdigest()[:10].upper()
    relay = "SPARK-RELAY-" + hashlib.sha256(
        f"{source}:{path_id}:relay".encode()
    ).hexdigest()[:10].upper()
    final = "SPARK-FINAL-" + hashlib.sha256(
        f"{relay}:{path_id}:final".encode()
    ).hexdigest()[:10].upper()
    return source, relay, final


def _multi_hop_values(nonce: str) -> tuple[str, str, str]:
    """Return the nonce-derived target path for one two-hop task."""

    return _multi_hop_path(nonce, "target")


def _multi_hop_needle(nonce: str) -> str:
    """Return only the final answer expected from a two-hop needle task."""

    return _multi_hop_values(nonce)[2]


def _is_multi_hop_needle(case: Any) -> bool:
    return str(getattr(case, "id", "")).startswith(
        "long-context-multi-hop-needle-"
    )


def _multi_hop_needle_prompt(
    *, prompt_repetitions: int, nonce: str, prefix: str
) -> str:
    """Build a two-hop retrieval prompt with separated target and decoy paths."""

    source, relay, final = _multi_hop_values(nonce)
    decoy_one = _multi_hop_path(nonce, "decoy-one")
    decoy_two = _multi_hop_path(nonce, "decoy-two")
    records = (
        f"Source record: source {source} routes through relay {relay}. ",
        (
            f"Source record: source {decoy_one[0]} routes through relay "
            f"{decoy_one[1]}. "
        ),
        (
            f"Relay record: relay {decoy_two[1]} has final archive key "
            f"{decoy_two[2]}. "
        ),
        f"Relay record: relay {relay} has final archive key {final}. ",
        (
            f"Relay record: relay {decoy_one[1]} has final archive key "
            f"{decoy_one[2]}. "
        ),
        (
            f"Source record: source {decoy_two[0]} routes through relay "
            f"{decoy_two[1]}. "
        ),
    )
    filler_count = max(prompt_repetitions, 1)
    block_count = len(records) + 1
    base_filler_count, extra_filler_count = divmod(filler_count, block_count)
    filler_counts = tuple(
        base_filler_count + (1 if index < extra_filler_count else 0)
        for index in range(block_count)
    )
    parts = [prefix]
    for index, record in enumerate(records):
        parts.append("archive " * filler_counts[index])
        parts.append(record)
    parts.extend(
        (
            "archive " * filler_counts[-1],
            f"Start at source {source}. Follow its relay, then use that relay to find "
            "the final archive key. Reply with only the final archive key.",
        )
    )
    return "".join(parts)


_VARIED_CONTEXT_FILLER_LEXICON = (
    "acorn",
    "apron",
    "basket",
    "beacon",
    "bicycle",
    "blanket",
    "bridge",
    "brook",
    "cabin",
    "canvas",
    "candle",
    "canyon",
    "carpet",
    "cedar",
    "circle",
    "cloud",
    "copper",
    "corner",
    "cricket",
    "cushion",
    "dawn",
    "drizzle",
    "farmer",
    "feather",
    "fence",
    "field",
    "forest",
    "garden",
    "guitar",
    "harbor",
    "helmet",
    "honey",
    "island",
    "jacket",
    "kettle",
    "ladder",
    "lantern",
    "market",
    "meadow",
    "mirror",
    "mountain",
    "napkin",
    "ocean",
    "orchard",
    "pencil",
    "pillow",
    "porch",
    "river",
    "robin",
    "saddle",
    "sailor",
    "shovel",
    "spring",
    "station",
    "stone",
    "sunset",
    "thistle",
    "ticket",
    "timber",
    "valley",
    "velvet",
    "window",
    "winter",
    "wren",
)
_VARIED_CONTEXT_ANSWER_LEXICON = (
    "alder",
    "beryl",
    "citron",
    "dahlia",
    "ember",
    "fable",
    "garnet",
    "hazel",
    "indigo",
    "juniper",
    "kepler",
    "lilac",
    "marigold",
    "nectar",
    "onyx",
    "piper",
    "quartz",
    "raven",
    "saffron",
    "topaz",
    "umber",
    "violet",
    "willow",
    "zephyr",
)
_VARIED_CONTEXT_ANSWER_WORD_COUNT = 12
_VARIED_CONTEXT_MASK_64 = (1 << 64) - 1


def _is_varied_context_needle(case: Any) -> bool:
    """Whether ``case`` selects the strict SM121 varied-context protocol."""

    return str(getattr(case, "id", "")).startswith(
        VARIED_CONTEXT_NEEDLE_CASE_PREFIX
    )


def _varied_context_spec(case: Any) -> VariedContextNeedleSpec:
    """Get the typed protocol contract from a frozen or loaded case."""

    case_id = str(getattr(case, "id", ""))
    spec = varied_context_needle_spec(case_id)
    if spec is None:
        raise ValueError("case does not select the varied-context needle protocol")
    return spec


def _splitmix64(value: int) -> int:
    """Advance a local deterministic stream without global random state."""

    value = (value + 0x9E3779B97F4A7C15) & _VARIED_CONTEXT_MASK_64
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
    value &= _VARIED_CONTEXT_MASK_64
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB
    value &= _VARIED_CONTEXT_MASK_64
    return (value ^ (value >> 31)) & _VARIED_CONTEXT_MASK_64


def _varied_context_stream_seed(case_id: str, stream: str) -> int:
    digest = hashlib.sha256(
        f"sm121-varied-context-v1:{stream}:{case_id}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _varied_context_needle(case: Any) -> str:
    """Return the stable, multiword visible answer for a case's stable ID."""

    case_id = str(getattr(case, "id", ""))
    _varied_context_spec(case)
    words = list(_VARIED_CONTEXT_ANSWER_LEXICON)
    state = _varied_context_stream_seed(case_id, "answer")
    for index in range(len(words) - 1, 0, -1):
        state = _splitmix64(state)
        selected = state % (index + 1)
        words[index], words[selected] = words[selected], words[index]
    return " ".join(words[:_VARIED_CONTEXT_ANSWER_WORD_COUNT])


def _varied_context_insert_index(spec: VariedContextNeedleSpec) -> int:
    """Place the unique answer at a reproducible semantic depth."""

    if spec.depth == "mid":
        return spec.filler_records // 2
    if spec.depth == "tail":
        return (spec.filler_records * 7) // 8
    raise ValueError(f"unsupported varied-context depth: {spec.depth}")


def _varied_context_filler_records(
    *, case_id: str, record_count: int
) -> Iterator[tuple[str, str]]:
    """Build deterministic two-word filler records from a fixed lexicon."""

    state = _varied_context_stream_seed(case_id, "filler")
    for _ in range(record_count):
        state = _splitmix64(state)
        left = _VARIED_CONTEXT_FILLER_LEXICON[
            state % len(_VARIED_CONTEXT_FILLER_LEXICON)
        ]
        state = _splitmix64(state)
        right = _VARIED_CONTEXT_FILLER_LEXICON[
            state % len(_VARIED_CONTEXT_FILLER_LEXICON)
        ]
        yield left, right


def _varied_context_needle_prompt(*, case: Any) -> str:
    """Build a nonce-independent varied-context retrieval prompt.

    The answer and every filler word derive only from the stable case ID.  A
    caller can therefore retry a request without changing the retrieval task,
    while the prompt itself never enters journal or evidence payloads.
    """

    spec = _varied_context_spec(case)
    case_id = str(getattr(case, "id", ""))
    target = _varied_context_needle(case)
    insert_index = _varied_context_insert_index(spec)
    parts = [
        "Synthetic varied-context retrieval protocol v1. Read every record. "
    ]
    for index, (left, right) in enumerate(
        _varied_context_filler_records(
            case_id=case_id, record_count=spec.filler_records
        )
    ):
        if index == insert_index:
            parts.append(f"Recovery phrase: {target}. ")
        parts.append(f"{left} {right}. ")
    if insert_index == spec.filler_records:
        parts.append(f"Recovery phrase: {target}. ")
    parts.append(
        "Return only the recovery phrase with every word in order and no punctuation."
    )
    return "".join(parts)


def _normalize_varied_context_answer(answer: str) -> str:
    """Normalize only whitespace and case; punctuation remains a mismatch."""

    return " ".join(answer.split()).casefold()


def _is_scalar_safe_needle(case: Any) -> bool:
    """Whether a generated needle must omit text-bearing result fields."""

    return _is_multi_hop_needle(case) or _is_varied_context_needle(case)


_PREFIX_CACHE_RESULT_SCALAR_FIELDS = (
    "cache_condition",
    "cache_pair_index",
    "cache_prompt_control",
    "cache_prefix_target_words",
    "cache_profile_mode",
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
)


def _is_prefix_cache_case(case: Any) -> bool:
    return str(getattr(case, "kind", "")) == "cache"


def _prefix_cache_shared_prefix(case: Any, pair_index: int) -> str:
    """Create a deterministic shared prefix for one serial cold/warm block.

    The pair key changes before the filler so blocks cannot accidentally reuse
    a preceding block.  It is deterministic case metadata, not the request
    nonce; each request nonce is appended only in the short suffix.
    """

    if pair_index <= 0:
        raise ValueError("prefix-cache pair index must be positive")
    # Use the stable suite case ID, not the profile-specific frozen case ID,
    # so matching cache-off/cache-on runs exercise the exact same long prefix.
    pair_key = hashlib.sha256(
        f"prefix-cache-v1:{case.id}:{pair_index}".encode()
    ).hexdigest()[:16]
    repetitions = max(int(case.prompt_repetitions), 1)
    return (
        "Synthetic static prefix-cache corpus. "
        f"Pair control {pair_key}. "
        + "shared-ledger-entry " * repetitions
    )


def _prefix_cache_prompt(case: Any, pair_index: int, request_id: str) -> str:
    """Append the only request-unique nonce material after a shared prefix."""

    suffix_nonce = hashlib.sha256(request_id.encode()).hexdigest()[:16]
    return (
        _prefix_cache_shared_prefix(case, pair_index)
        + f" Request suffix nonce {suffix_nonce}. "
        "Write an unbroken numbered list of distinct two-word phrases. "
        "Continue until the output limit; do not conclude or summarize."
    )


def _prefix_cache_steps(
    mode: str,
) -> tuple[tuple[str, bool | None, str], ...]:
    """Return the exact serial control/treatment schedule for one block."""

    try:
        return prefix_cache_steps(mode)
    except ValueError as error:
        raise PrefixCacheError() from error


def _prefix_cache_scalar(
    value: Any,
    *,
    name: str,
    integer: bool = False,
    positive: bool = False,
    nullable: bool = False,
) -> int | float | None:
    """Validate a scalar retained by the privacy-safe cache journal."""

    if value is None and nullable:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (integer and not isinstance(value, int))
        or not math.isfinite(float(value))
        or float(value) < 0
        or (positive and float(value) <= 0)
    ):
        raise PrefixCacheError()
    return int(value) if integer else float(value)


def _prefix_cache_result_payload(
    result: Any,
    *,
    mode: str,
    condition: str,
    pair_index: int,
    step_ordinal: int,
    cache_prompt_control: str,
    prefix_target_words: int,
    prometheus_metrics: dict[str, int | float],
) -> dict[str, Any]:
    """Serialize one scalar-only cache record.

    ``server_*`` values are request-scoped final SSE counters and timings.
    ``prometheus_global_*`` values are only the corresponding global
    Prometheus deltas; retain them as diagnostics without treating them as
    request attribution.
    """

    payload = {
        "cache_condition": condition,
        "cache_pair_index": pair_index,
        "cache_prompt_control": cache_prompt_control,
        "cache_prefix_target_words": prefix_target_words,
        "cache_profile_mode": mode,
        "cache_step_ordinal": step_ordinal,
        "cached_prompt_tokens": result.cached_prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "decode_metric_source": result.decode_metric_source,
        "decode_s": result.decode_s,
        "decode_tps": result.decode_tps,
        "elapsed_s": result.elapsed_s,
        "emission_events": result.emission_events,
        "finish_reason": result.finish_reason,
        "prometheus_global_cached_prompt_tokens": prometheus_metrics[
            "cached_prompt_tokens"
        ],
        "prometheus_global_decode_s": prometheus_metrics["decode_s"],
        "prometheus_global_decode_tokens": prometheus_metrics["decode_tokens"],
        "prometheus_global_prompt_s": prometheus_metrics["prompt_s"],
        "prometheus_global_prompt_tokens": prometheus_metrics["prompt_tokens"],
        "output_tps": result.output_tps,
        "prompt_tokens": result.prompt_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "server_cached_prompt_tokens": result.server_cached_prompt_tokens,
        "server_decode_s": result.server_decode_s,
        "server_decode_tokens": result.server_decode_tokens,
        "server_prompt_s": result.server_prompt_s,
        "server_prompt_tokens": result.server_prompt_tokens,
        "ttft_s": result.ttft_s,
    }
    if set(payload) != set(_PREFIX_CACHE_RESULT_SCALAR_FIELDS):
        raise PrefixCacheError()
    for field in (
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
    ):
        _prefix_cache_scalar(payload[field], name=field, integer=True)
    for field in (
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
        _prefix_cache_scalar(
            payload[field],
            name=field,
            positive=field in {"decode_s", "elapsed_s", "server_decode_s"},
        )
    _prefix_cache_scalar(
        payload["reasoning_tokens"],
        name="reasoning_tokens",
        integer=True,
        nullable=True,
    )
    if (
        payload["cache_condition"] not in {
            "forced-cold-a",
            "forced-cold-b",
            "forced-cold-c",
            "warm-prefix-hit",
        }
        or payload["cache_prompt_control"]
        not in {"profile-default", "force-off"}
        or payload["cache_profile_mode"] not in {"off", "on"}
        or not isinstance(payload["decode_metric_source"], str)
        or payload["finish_reason"] != "length"
    ):
        raise PrefixCacheError()
    return payload


def _prefix_cache_control_mode(model: Any, server: Any, case: Any) -> str:
    """Validate the frozen same-slot native cache control before measuring."""

    mode = getattr(model, "prefix_cache_mode", None)
    if (
        str(getattr(server, "backend", "")) != "llamacpp"
        or mode not in {"off", "on"}
        or int(getattr(model, "runtime_parallel", 0)) != 1
        or int(getattr(case, "concurrency", 0)) != 1
    ):
        raise PrefixCacheError()
    argument_values = tuple(str(argument) for argument in getattr(model, "args", ()))
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
        raise PrefixCacheError()
    if "--cache-reuse" in argument_names:
        raise PrefixCacheError()
    enabled = argument_values.count("--cache-prompt")
    disabled = argument_values.count("--no-cache-prompt")
    if enabled + disabled != 1:
        raise PrefixCacheError()
    if (mode == "on" and enabled != 1) or (mode == "off" and disabled != 1):
        raise PrefixCacheError()
    return str(mode)


def _prefix_cache_request_arguments(
    *,
    server: Any,
    model: Any,
    case: Any,
    pair_index: int,
    request_id: str,
    cache_prompt: bool | None,
) -> dict[str, Any]:
    """Build one forced-cold or warm same-slot native request."""

    extra_body: dict[str, Any] = {"id_slot": 0}
    if cache_prompt is not None:
        extra_body["cache_prompt"] = cache_prompt
    arguments: dict[str, Any] = {
        "base_url": server.base_url,
        "model": str(model.served_name),
        "prompt": _prefix_cache_prompt(case, pair_index, request_id),
        "max_tokens": int(case.max_output_tokens),
        "temperature": float(case.temperature),
        "request_id": request_id,
        "extra_body": extra_body,
        "require_native_cache_metrics": True,
        "require_native_timing": True,
    }
    if getattr(server, "authorization", None):
        arguments["authorization"] = str(server.authorization)
    return arguments


def _validate_prefix_cache_result(
    *,
    result: Any,
    prometheus_metrics: dict[str, int | float],
    case: Any,
    condition: str,
) -> dict[str, Any]:
    """Prove one request is a real cache control or a substantial prefix hit.

    The final SSE fields are request-scoped and therefore authoritative for
    token identity and server timing.  Prometheus snapshots are validated only
    as non-negative, server-global diagnostics.
    """

    cached = getattr(result, "cached_prompt_tokens", None)
    prompt_tokens = getattr(result, "prompt_tokens", None)
    completion_tokens = getattr(result, "completion_tokens", None)
    ttft_s = _prefix_cache_scalar(getattr(result, "ttft_s", None), name="ttft_s")
    elapsed_s = _prefix_cache_scalar(
        getattr(result, "elapsed_s", None), name="elapsed_s", positive=True
    )
    decode_s = _prefix_cache_scalar(
        getattr(result, "decode_s", None), name="decode_s", positive=True
    )
    decode_tps = _prefix_cache_scalar(
        getattr(result, "decode_tps", None), name="decode_tps"
    )
    output_tps = _prefix_cache_scalar(
        getattr(result, "output_tps", None), name="output_tps"
    )
    server_prompt_tokens = getattr(result, "server_prompt_tokens", None)
    server_cached_prompt_tokens = getattr(result, "server_cached_prompt_tokens", None)
    server_decode_tokens = getattr(result, "server_decode_tokens", None)
    valid = (
        isinstance(cached, int)
        and not isinstance(cached, bool)
        and cached >= 0
        and isinstance(prompt_tokens, int)
        and not isinstance(prompt_tokens, bool)
        and prompt_tokens > 0
        and cached <= prompt_tokens
        and isinstance(completion_tokens, int)
        and not isinstance(completion_tokens, bool)
        and completion_tokens == int(case.max_output_tokens)
        and all(
            type(value) is int and value >= 0
            for value in (
                server_prompt_tokens,
                server_cached_prompt_tokens,
                server_decode_tokens,
            )
        )
        and int(server_prompt_tokens) + int(server_cached_prompt_tokens)
        == int(prompt_tokens)
        and int(server_cached_prompt_tokens) == int(cached)
        and int(server_decode_tokens) == int(completion_tokens)
        and getattr(result, "finish_reason", None) == "length"
        and _prefix_cache_scalar(
            getattr(result, "server_prompt_s", None), name="server_prompt_s"
        ) is not None
        and _prefix_cache_scalar(
            getattr(result, "server_decode_s", None),
            name="server_decode_s",
            positive=True,
        ) is not None
        and getattr(result, "decode_metric_source", None) == "client_estimate"
        and ttft_s is not None
        and elapsed_s is not None
        and decode_s is not None
        and decode_tps is not None
        and output_tps is not None
        and ttft_s <= elapsed_s
        and math.isclose(
            float(decode_tps),
            max(int(completion_tokens) - 1, 0) / float(decode_s),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        and math.isclose(
            float(output_tps),
            int(completion_tokens) / float(elapsed_s),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    )
    if valid:
        try:
            require_llamacpp_cache_delta(prometheus_metrics)
        except LlamaCppCacheMetricsError:
            valid = False
    if valid and condition.startswith("forced-cold"):
        valid = cached == 0
    if valid and condition == "warm-prefix-hit":
        valid = cached / prompt_tokens >= 0.90
    return {
        "passed": valid,
        "reason": None if valid else "prefix-cache control did not validate",
    }


def _solid_color_png_data_url(
    red: int, green: int, blue: int, size: int = 64
) -> str:
    width = height = max(16, min(size, 2048))
    pixel = bytes((red, green, blue))
    raw = b"".join(b"\x00" + pixel * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _solid_red_png_data_url(size: int = 64) -> str:
    return _solid_color_png_data_url(255, 0, 0, size)


def _solid_blue_png_data_url(size: int = 64) -> str:
    return _solid_color_png_data_url(0, 0, 255, size)


_AUDIO_FIXTURE_PATH = Path(
    "/home/xlz/voice-cloning/Spark-TTS/example/prompt_audio.wav"
)
_AUDIO_FIXTURE_SHA256 = (
    "335e7f7789b231cd90d9670292d561ecfe6a6bdd5e737a7bc6c29730741852de"
)
_AUDIO_EXPECTED_TRANSCRIPTION = (
    "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。"
    "豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"
)
_AUDIO_PROMPT = "Transcribe the audio clip into text."
_AUDIO_LORA_NAME = "speech"


def _normalize_audio_transcription(content: str) -> str:
    """Normalize punctuation and width while preserving transcript wording."""

    normalized = unicodedata.normalize("NFKC", content)
    return "".join(character for character in normalized if character.isalnum()).casefold()


_OCR_EXPECTED_TRANSCRIPTION = "SPARKOCR4827"
# DeepSeek-OCR is unusually sensitive to this exact, punctuated prompt; Ollama's
# model card and the upstream reference both use it for layout-free OCR.
_OCR_PROMPT = "Free OCR."
_OCR_IMAGE_WIDTH = 1024
_OCR_IMAGE_HEIGHT = 256
_OCR_GLYPH_WIDTH = 5
_OCR_GLYPH_HEIGHT = 7
_OCR_GLYPH_SCALE = 12
_OCR_GLYPHS = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("11111", "10000", "10000", "11111", "00001", "00001", "11111"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "4": ("00100", "01100", "10100", "11111", "00100", "00100", "00100"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
}


@functools.lru_cache(maxsize=1)
def _ocr_png_data_url() -> str:
    """Render the fixed OCR token into a dependency-free, deterministic PNG."""

    pixels = bytearray(b"\xff" * (_OCR_IMAGE_WIDTH * _OCR_IMAGE_HEIGHT * 3))
    token_width = (
        len(_OCR_EXPECTED_TRANSCRIPTION) * _OCR_GLYPH_WIDTH
        + len(_OCR_EXPECTED_TRANSCRIPTION)
        - 1
    ) * _OCR_GLYPH_SCALE
    start_x = (_OCR_IMAGE_WIDTH - token_width) // 2
    start_y = (
        _OCR_IMAGE_HEIGHT - _OCR_GLYPH_HEIGHT * _OCR_GLYPH_SCALE
    ) // 2
    for character_index, character in enumerate(_OCR_EXPECTED_TRANSCRIPTION):
        glyph = _OCR_GLYPHS[character]
        glyph_x = start_x + character_index * (
            _OCR_GLYPH_WIDTH + 1
        ) * _OCR_GLYPH_SCALE
        for glyph_y, row in enumerate(glyph):
            for glyph_column, enabled in enumerate(row):
                if enabled != "1":
                    continue
                left = glyph_x + glyph_column * _OCR_GLYPH_SCALE
                top = start_y + glyph_y * _OCR_GLYPH_SCALE
                for pixel_y in range(top, top + _OCR_GLYPH_SCALE):
                    offset = (pixel_y * _OCR_IMAGE_WIDTH + left) * 3
                    pixels[offset : offset + _OCR_GLYPH_SCALE * 3] = (
                        b"\x00" * (_OCR_GLYPH_SCALE * 3)
                    )

    raw = b"".join(
        b"\x00"
        + pixels[
            row * _OCR_IMAGE_WIDTH * 3 : (row + 1) * _OCR_IMAGE_WIDTH * 3
        ]
        for row in range(_OCR_IMAGE_HEIGHT)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            struct.pack(
                ">IIBBBBB",
                _OCR_IMAGE_WIDTH,
                _OCR_IMAGE_HEIGHT,
                8,
                2,
                0,
                0,
                0,
            ),
        )
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _normalize_ocr_transcription(content: str) -> str:
    """Apply Unicode normalization and trim framing whitespace only."""

    return unicodedata.normalize("NFKC", content).strip()


_RERANK_QUERY = "Which planet is known as the Red Planet?"
_RERANK_RELEVANT_INDEX = 1
_RERANK_BASE_CANDIDATES = (
    "Venus is the second planet from the Sun and has a dense atmosphere.",
    "Mars is widely known as the Red Planet because iron oxides color its surface.",
    "Jupiter is the largest planet in the Solar System and is a gas giant.",
    "Neptune is an ice giant with strong winds and a deep blue appearance.",
)
_RERANK_DISTRACTORS = (
    "Saturn has a prominent ring system made mostly of ice particles.",
    "Mercury is the closest planet to the Sun and has a cratered surface.",
    "Uranus rotates on its side and is classified as an ice giant.",
    "Earth has liquid surface oceans and a nitrogen-rich atmosphere.",
)
_MULTIMODAL_RELEVANT_TEXT = "A solid red square with no other colors or objects."
_MULTIMODAL_UNRELATED_TEXT = (
    "A photograph of snow-covered mountains beneath a cloudy blue sky."
)
_MULTIMODAL_RERANK_QUERY = "A solid red square with no other colors or objects."
_MULTIMODAL_RERANK_INSTRUCTION = (
    "Retrieve images relevant to the user's query."
)
_MULTIMODAL_RERANK_CANDIDATE_COUNT = 2
_MULTIMODAL_RERANK_RELEVANT_INDEX = 1


@dataclass(frozen=True, slots=True)
class QualityItem:
    """One embedded exact-answer item; model output is always treated as text."""

    id: str
    category: str
    question: str
    expected_answer: str


_QUALITY_ITEMS = (
    QualityItem(
        id="arithmetic-01",
        category="arithmetic",
        question="What is (17 * 6) - 19?",
        expected_answer="83",
    ),
    QualityItem(
        id="logic-01",
        category="logic",
        question=(
            "All flerns are torps. No torps are silver. Can any flern be silver? "
            "Answer yes or no."
        ),
        expected_answer="no",
    ),
    QualityItem(
        id="instruction-01",
        category="instruction_following",
        question="Return the third word in this sequence: amber cobalt silver jade.",
        expected_answer="silver",
    ),
    QualityItem(
        id="code-01",
        category="code_reasoning",
        question=(
            "Without running it, what integer does this Python snippet print?\n"
            "```python\n"
            "values = [2, 5, 8]\n"
            "total = 0\n"
            "for value in values:\n"
            "    if value % 2 == 0:\n"
            "        total += value\n"
            "    else:\n"
            "        total -= 1\n"
            "print(total)\n"
            "```"
        ),
        expected_answer="9",
    ),
)
_QUALITY_FINAL_PATTERN = re.compile(
    r"^\s*(?:\*\*|__|`)?final"
    r"[ \t]*"
    r"(?::(?:\*\*|__|`)[ \t]*|:[ \t]*|(?:\*\*|__|`)[ \t]*:)"
    r"\s*(?P<answer>.+?)\s*$",
    re.IGNORECASE,
)
_QUALITY_FINAL_OCCURRENCE_PATTERN = re.compile(
    r"\bfinal(?::\s*(?:\*\*|__|`)?|(?:\*\*|__|`)?\s*:)",
    re.IGNORECASE,
)
_NUMERIC_ANSWER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def _is_multimodal_embedding_case(case: SimpleNamespace) -> bool:
    return {"embeddings", "vision"}.issubset(set(case.requires))


def _is_multimodal_rerank_case(case: SimpleNamespace) -> bool:
    return {"rerank", "vision"}.issubset(
        set(getattr(case, "requires", ()))
    )


def _is_diffusion_model(model: Any) -> bool:
    """Return whether generation uses non-autoregressive diffusion semantics."""

    return str(getattr(model, "architecture", "")).strip().lower() == "diffusion-lm"


def _request_result_payload(model: Any, result: Any) -> dict[str, Any]:
    """Serialize a result without assigning AR timing semantics to diffusion output."""

    payload = result.to_dict()
    if not _is_diffusion_model(model):
        return payload
    payload.update(
        {
            "time_to_first_emission_s": payload.get("ttft_s"),
            "block_generation_output_tps": payload.get("output_tps"),
            "block_generation_metric_source": (
                "client_completion_tokens_per_end_to_end_request_elapsed"
            ),
            "ttft_s": None,
            "decode_s": None,
            "decode_tps": None,
            "decode_metric_source": None,
            "output_tps": None,
        }
    )
    return payload


_MULTI_HOP_RESULT_SCALAR_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "ttft_s",
    "elapsed_s",
    "decode_s",
    "decode_tps",
    "output_tps",
    "emission_events",
    "finish_reason",
    "response_model",
    "decode_metric_source",
    "load_s",
    "server_prompt_s",
)


def _multi_hop_result_payload(model: Any, result: Any) -> dict[str, Any]:
    """Return the fixed scalar result schema used by generated needle cases.

    The generated prompt, nonce-derived request identifiers and start times,
    relation values, visible completion, hidden reasoning, and tool payloads
    are intentionally excluded from raw results.  The historical function
    name remains for compatibility with the multi-hop protocol.
    """

    payload = _request_result_payload(model, result)
    scalar_payload: dict[str, Any] = {}
    for field in _MULTI_HOP_RESULT_SCALAR_FIELDS:
        if field not in payload:
            continue
        value = payload[field]
        if value is not None and not isinstance(value, (bool, int, float, str)):
            raise RuntimeError(
                f"Generated needle result field {field!r} must be a scalar or null"
            )
        scalar_payload[field] = value
    return scalar_payload


def _rerank_inputs(case: SimpleNamespace) -> tuple[str, list[str], int]:
    requested = int(case.prompt_repetitions)
    candidate_count = max(requested, 2) if requested else len(_RERANK_BASE_CANDIDATES)
    candidates = list(_RERANK_BASE_CANDIDATES[:candidate_count])
    while len(candidates) < candidate_count:
        index = len(candidates)
        distractor = _RERANK_DISTRACTORS[
            (index - len(_RERANK_BASE_CANDIDATES)) % len(_RERANK_DISTRACTORS)
        ]
        candidates.append(f"{distractor} Candidate record {index:03d}.")
    return _RERANK_QUERY, candidates, _RERANK_RELEVANT_INDEX


def _multimodal_rerank_inputs(
    case: SimpleNamespace,
) -> tuple[str, list[dict[str, Any]], int]:
    image_size = int(case.prompt_repetitions) or 64
    documents = [
        {
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _solid_blue_png_data_url(image_size)},
                }
            ]
        },
        {
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _solid_red_png_data_url(image_size)},
                }
            ]
        },
    ]
    return (
        _MULTIMODAL_RERANK_QUERY,
        documents,
        _MULTIMODAL_RERANK_RELEVANT_INDEX,
    )


def _quality_prompt(
    item: QualityItem, nonce: str, *, protocol_version: int = 2
) -> str:
    if protocol_version == 1:
        header = f"Benchmark exact-answer item {item.id}; nonce {nonce}."
    elif protocol_version == 2:
        header = (
            f"Benchmark exact-answer protocol v2; item {item.id}; "
            f"variant {nonce}."
        )
    else:
        raise ValueError(f"unsupported quality protocol version: {protocol_version}")
    return (
        f"{header}\n{item.question}\n"
        "Return one line exactly in the form `FINAL: <answer>`. "
        "Do not include an explanation."
    )


def _uses_matched_prompt_protocol(case: SimpleNamespace) -> bool:
    return str(case.id).startswith("ple-study-")


def _extract_quality_answer(content: str) -> tuple[str | None, str | None]:
    """Extract an explicit final answer without interpreting or executing output."""

    if len(_QUALITY_FINAL_OCCURRENCE_PATTERN.findall(content)) > 1:
        return None, "response contained multiple FINAL markers"
    lines = content.strip().splitlines()
    if (
        len(lines) >= 2
        and re.fullmatch(r"\s*```[A-Za-z0-9_-]*\s*", lines[0])
        and re.fullmatch(r"\s*```\s*", lines[-1])
    ):
        lines = lines[1:-1]
    nonempty = [line.strip() for line in lines if line.strip()]
    marked: list[str] = []
    for line in nonempty:
        marker_line = line
        for wrapper in ("**", "__", "`"):
            if len(marker_line) > 2 * len(wrapper) and marker_line.startswith(
                wrapper
            ) and marker_line.endswith(wrapper):
                marker_line = marker_line[len(wrapper) : -len(wrapper)].strip()
                break
        match = _QUALITY_FINAL_PATTERN.fullmatch(marker_line)
        if match:
            marked.append(match.group("answer").strip())
    if marked:
        return marked[0], None
    if len(nonempty) == 1:
        return nonempty[0], None
    return None, "response had neither one FINAL marker nor one bare-answer line"


def _normalize_quality_answer(answer: str) -> str:
    value = " ".join(answer.strip().split())
    paired_wrappers = (
        ("**", "**"),
        ("__", "__"),
        ("`", "`"),
        ('"', '"'),
        ("'", "'"),
    )
    for _ in range(3):
        previous = value
        if value.endswith("."):
            value = value[:-1].rstrip()
        for opening, closing in paired_wrappers:
            if len(value) > len(opening) + len(closing) and value.startswith(
                opening
            ) and value.endswith(closing):
                value = value[len(opening) : -len(closing)].strip()
                break
        if value == previous:
            break
    return value.casefold()


def _quality_answers_match(extracted: str, expected: str) -> bool:
    actual = _normalize_quality_answer(extracted)
    target = _normalize_quality_answer(expected)
    if _NUMERIC_ANSWER_PATTERN.fullmatch(actual) and _NUMERIC_ANSWER_PATTERN.fullmatch(
        target
    ):
        try:
            return Decimal(actual) == Decimal(target)
        except InvalidOperation:
            return False
    return actual == target


def _validate_quality_item(item: QualityItem, result: Any) -> dict[str, Any]:
    extracted, extraction_error = _extract_quality_answer(str(result.content))
    passed = bool(
        extraction_error is None
        and extracted is not None
        and _quality_answers_match(extracted, item.expected_answer)
    )
    reason = extraction_error
    if reason is None and not passed:
        reason = "extracted answer did not match the exact answer key"
    return {
        "passed": passed,
        "reason": reason,
        "quality_item_id": item.id,
        "quality_category": item.category,
        "expected_answer": item.expected_answer,
        "extracted_answer": extracted[:160] if extracted is not None else None,
    }


def _prompt(case: SimpleNamespace, nonce: str) -> str:
    kind = str(case.kind)
    prompt_repetitions = int(case.prompt_repetitions)
    prefix = f"Benchmark nonce {nonce}. "
    if kind == "prefill":
        return prefix + "Read all text and reply with exactly one word. " + (
            "measurement " * prompt_repetitions
        )
    if kind == "capability" and "json" in case.requires:
        return prefix + 'Return only a JSON object with keys "benchmark" set to "spark" and "value" set to 42.'
    if kind == "capability" and "tools" in case.requires:
        return prefix + "Use the multiply tool to multiply 6 by 7. Do not answer without calling the tool."
    if kind == "capability" and _is_varied_context_needle(case):
        return _varied_context_needle_prompt(case=case)
    if kind == "capability" and _is_multi_hop_needle(case):
        return _multi_hop_needle_prompt(
            prompt_repetitions=prompt_repetitions,
            nonce=nonce,
            prefix=prefix,
        )
    if kind == "capability" and str(case.id).startswith("long-context-needle"):
        filler = "archive " * max(prompt_repetitions, 1)
        midpoint = len(filler) // 2
        needle = _needle(nonce)
        return (
            prefix
            + filler[:midpoint]
            + f" The unique benchmark key is {needle}. "
            + filler[midpoint:]
            + " What is the unique benchmark key? Reply with only the key."
        )
    if kind == "concurrency" and prompt_repetitions > 0:
        return (
            prefix
            + "Read all of the synthetic context before answering. "
            + ("measurement " * prompt_repetitions)
            + "Now write an unbroken numbered list of distinct two-word phrases. "
            "Continue until the output limit; do not conclude or summarize."
        )
    return (
        prefix
        + "Write an unbroken numbered list of distinct two-word phrases. "
        "Continue until the output limit; do not conclude or summarize."
    )


def _estimated_context_tokens(case: SimpleNamespace) -> tuple[int, str]:
    """Return a conservative workload estimate without model-specific tokenizers."""

    output_tokens = int(case.max_output_tokens)
    if _is_varied_context_needle(case):
        spec = _varied_context_spec(case)
        if str(case.id) == SM121_STORAGE_VARIED_CONTEXT_CASE_ID:
            return (
                SM121_STORAGE_VARIED_CONTEXT_BUDGET_TOKENS,
                "pinned_qwen_tokenizer_chat_template_plus_output",
            )
        return (
            spec.filler_records * 2 + output_tokens + 256,
            "varied_context_two_words_per_record_plus_output_and_template_margin",
        )
    if _is_prefix_cache_case(case):
        # ``shared-ledger-entry`` tokenizes to multiple pieces on common BPE
        # vocabularies.  Six tokens per synthetic corpus word leaves generous
        # headroom without pretending this workload is a one-token repetition.
        return (
            max(int(case.prompt_repetitions), 1) * 6 + output_tokens + 1_024,
            "prefix_cache_words_times_six_plus_output_and_template_margin",
        )
    if str(case.kind) == "agentic":
        return (
            estimate_agentic_context_tokens(
                max_turns=int(case.max_turns),
                max_output_tokens=output_tokens,
            ),
            "agentic_episode_max_turns_times_output_plus_tool_history_margin",
        )
    if str(case.kind) == "memory":
        return (
            estimate_memory_operation_context_tokens(
                max_output_tokens=output_tokens
            ),
            "memory_operation_fixed_prompt_plus_json_output_margin",
        )
    if str(case.kind) == "quality":
        quality_protocol_version = (
            2 if str(case.id) == "synthetic-exact-answer-v2" else 1
        )
        prompt_words = max(
            len(
                _quality_prompt(
                    item,
                    "context-estimate",
                    protocol_version=quality_protocol_version,
                ).split()
            )
            for item in _QUALITY_ITEMS
        )
        return prompt_words + output_tokens + 128, "quality_prompt_words_plus_margin"
    if "audio" in case.requires:
        return output_tokens + 4096, "fixed_9.953313s_audio_plus_margin"
    if "ocr" in case.requires:
        patch_tokens = (
            (_OCR_IMAGE_WIDTH + 13) // 14
        ) * ((_OCR_IMAGE_HEIGHT + 13) // 14)
        return patch_tokens + output_tokens + 256, "ocr_image_patch14_plus_margin"
    if "vision" in case.requires:
        image_size = max(16, min(int(case.prompt_repetitions) or 64, 2048))
        patch_tokens = ((image_size + 13) // 14) ** 2
        return patch_tokens + output_tokens + 256, "clamped_vision_patch14_plus_margin"
    if "embeddings" in case.requires:
        return (
            max(int(case.prompt_repetitions), 1) + output_tokens + 32,
            "embedding_words_plus_margin",
        )
    if "rerank" in case.requires:
        query, candidates, _ = _rerank_inputs(case)
        longest_pair_words = len(query.split()) + max(
            len(candidate.split()) for candidate in candidates
        )
        return longest_pair_words + 128, "rerank_pair_words_plus_margin"
    prompt_words = len(_prompt(case, "context-estimate").split())
    request_overhead = 128
    estimate_basis = "prompt_words_plus_request_margin"
    if "tools" in case.requires:
        request_overhead += 256
        estimate_basis = "prompt_words_plus_tool_schema_margin"
    if "json" in case.requires:
        request_overhead += 64
        if estimate_basis == "prompt_words_plus_request_margin":
            estimate_basis = "prompt_words_plus_json_schema_margin"
    return (
        prompt_words + output_tokens + request_overhead,
        estimate_basis,
    )


def _request_arguments(
    *, server: Any, model: SimpleNamespace, case: SimpleNamespace, request_id: str
) -> dict[str, Any]:
    max_tokens = int(case.max_output_tokens)
    audio_case = str(case.kind) == "capability" and "audio" in case.requires
    if audio_case and server.backend != "sglang":
        raise RuntimeError(
            "The audio adapter requires SGLang with a registered speech LoRA"
        )
    request_body_json = getattr(model, "request_body_json", None)
    extra_body = json.loads(request_body_json) if request_body_json else {}
    if str(case.kind) == "capability" and "json" in case.requires:
        extra_body["response_format"] = {"type": "json_object"}
    if str(case.kind) == "capability" and "tools" in case.requires:
        extra_body.update(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "multiply",
                            "description": "Multiply two integers.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer"},
                                    "b": {"type": "integer"},
                                },
                                "required": ["a", "b"],
                            },
                        },
                    }
                ],
                "tool_choice": "required",
            }
        )
    if audio_case:
        if "messages" in extra_body or "lora_path" in extra_body:
            raise RuntimeError(
                "Audio cases reserve messages and lora_path request fields"
            )
    elif str(case.kind) == "capability" and "ocr" in case.requires:
        extra_body["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": _ocr_png_data_url()},
                    },
                ],
            }
        ]
    elif str(case.kind) == "capability" and "vision" in case.requires:
        extra_body["messages"] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the dominant color of this image? Reply with one color word.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _solid_red_png_data_url(
                                int(case.prompt_repetitions) or 64
                            )
                        },
                    },
                ],
            }
        ]
    arguments = {
        "base_url": server.base_url,
        "model": str(model.served_name),
        "prompt": (
            _AUDIO_PROMPT
            if audio_case
            else _OCR_PROMPT
            if "ocr" in case.requires
            else _prompt(case, request_id)
        ),
        "max_tokens": max_tokens,
        "temperature": float(case.temperature),
        "request_id": request_id,
        "extra_body": extra_body,
    }
    if audio_case:
        arguments.update(
            {
                "audio_path": _AUDIO_FIXTURE_PATH,
                "expected_audio_sha256": _AUDIO_FIXTURE_SHA256,
                "lora_path": _AUDIO_LORA_NAME,
            }
        )
    if server.backend == "ollama":
        arguments["context_size"] = int(model.max_context)
        arguments["require_native_decode_timing"] = not (
            str(case.kind) == "capability" and "vision" in case.requires
        )
    elif getattr(server, "authorization", None):
        arguments["authorization"] = str(server.authorization)
    return arguments


def _quality_request_arguments(
    *,
    server: Any,
    model: SimpleNamespace,
    case: SimpleNamespace,
    item: QualityItem,
    request_id: str,
    prompt_tag: str,
) -> dict[str, Any]:
    arguments = _request_arguments(
        server=server,
        model=model,
        case=case,
        request_id=request_id,
    )
    if str(case.id) == "synthetic-exact-answer-v2":
        arguments["prompt"] = _quality_prompt(
            item, prompt_tag, protocol_version=2
        )
    else:
        arguments["prompt"] = _quality_prompt(
            item, request_id, protocol_version=1
        )
    return arguments


def _chat_request_function(server: Any, case: SimpleNamespace | None = None):
    if case is not None and "audio" in case.requires:
        if server.backend != "sglang":
            raise RuntimeError(
                "The audio adapter requires SGLang with a registered speech LoRA"
            )
        return stream_audio_chat_request
    return stream_ollama_chat_request if server.backend == "ollama" else stream_chat_request


def _authorization_argument(server: Any) -> dict[str, str]:
    managed_auth = getattr(server, "authorization", None)
    return {"authorization": str(managed_auth)} if managed_auth else {}


def _rerank_request_arguments(
    *, server: Any, model: SimpleNamespace, case: SimpleNamespace, request_id: str
) -> dict[str, Any]:
    if server.backend != "vllm":
        raise RuntimeError("The rerank adapter requires a vLLM /score endpoint")
    if _is_multimodal_rerank_case(case):
        query, candidates, _ = _multimodal_rerank_inputs(case)
        instruction = _MULTIMODAL_RERANK_INSTRUCTION
    else:
        query, candidates, _ = _rerank_inputs(case)
        instruction = None
    arguments = {
        "base_url": server.base_url,
        "model": str(model.served_name),
        "query": query,
        "candidates": candidates,
        "request_id": request_id,
    }
    if getattr(server, "authorization", None):
        arguments["authorization"] = str(server.authorization)
    if instruction is not None:
        arguments["instruction"] = instruction
    return arguments


def _multimodal_embedding_request_arguments(
    *, server: Any, model: SimpleNamespace, case: SimpleNamespace, request_id: str
) -> dict[str, Any]:
    if server.backend != "vllm":
        raise RuntimeError(
            "Multimodal embeddings require the vLLM Chat Embeddings endpoint"
        )
    image_size = int(case.prompt_repetitions) or 64
    arguments = {
        "base_url": server.base_url,
        "model": str(model.served_name),
        "image_data_url": _solid_red_png_data_url(image_size),
        "relevant_text": _MULTIMODAL_RELEVANT_TEXT,
        "unrelated_text": _MULTIMODAL_UNRELATED_TEXT,
        "request_id": request_id,
    }
    if getattr(server, "authorization", None):
        arguments["authorization"] = str(server.authorization)
    return arguments


def _validate_capability(
    case: SimpleNamespace, result: Any, model: Any | None = None
) -> dict[str, Any] | None:
    if str(case.kind) in {"decode", "concurrency"}:
        expected_tokens = int(case.max_output_tokens)
        actual_tokens = int(result.completion_tokens)
        diffusion_generation = model is not None and _is_diffusion_model(model)
        output_rate = getattr(result, "output_tps", None)
        rate_valid = (
            isinstance(output_rate, (int, float))
            and math.isfinite(float(output_rate))
            and float(output_rate) > 0
        )
        passed = (
            result.finish_reason == "length"
            and actual_tokens == expected_tokens
            and (rate_valid or not diffusion_generation)
        )
        if result.finish_reason != "length":
            reason = f"generation ended with {result.finish_reason!r}"
        elif actual_tokens != expected_tokens:
            reason = (
                f"generation reported {actual_tokens} completion tokens; "
                f"expected {expected_tokens}"
            )
        elif diffusion_generation and not rate_valid:
            reason = "end-to-end block-generation output rate was not positive and finite"
        else:
            reason = None
        validation = {
            "passed": passed,
            "reason": reason,
        }
        if diffusion_generation:
            validation.update(
                {
                    "generation_mode": "diffusion_block_generation",
                    "throughput_metric": (
                        "completion_tokens_per_end_to_end_request_elapsed"
                    ),
                }
            )
        return validation
    if str(case.kind) != "capability":
        return None
    if "json" in case.requires:
        try:
            payload = json.loads(result.content)
        except json.JSONDecodeError:
            return {"passed": False, "reason": "response was not valid JSON"}
        passed = payload.get("benchmark") == "spark" and payload.get("value") == 42
        return {"passed": passed, "reason": None if passed else "JSON values differed"}
    if "tools" in case.requires:
        for call in result.tool_calls:
            function = call.get("function") or {}
            if function.get("name") != "multiply":
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            passed = {arguments.get("a"), arguments.get("b")} == {6, 7}
            return {"passed": passed, "reason": None if passed else "tool arguments differed"}
        return {"passed": False, "reason": "multiply tool was not called"}
    if _is_multimodal_embedding_case(case):
        if not result.finite or int(result.dimension) <= 0:
            return {
                "passed": False,
                "reason": "multimodal embedding vectors were not finite and nonzero",
            }
        relevant = result.relevant_similarity
        unrelated = result.unrelated_similarity
        if not isinstance(relevant, (int, float)) or not isinstance(
            unrelated, (int, float)
        ):
            return {
                "passed": False,
                "reason": "multimodal embedding similarities were unavailable",
            }
        passed = math.isfinite(float(relevant)) and math.isfinite(
            float(unrelated)
        ) and float(relevant) > float(unrelated)
        return {
            "passed": passed,
            "reason": (
                None
                if passed
                else "red image did not rank the relevant red text higher"
            ),
        }
    if "audio" in case.requires:
        transcription = _normalize_audio_transcription(str(result.content))
        expected = _normalize_audio_transcription(_AUDIO_EXPECTED_TRANSCRIPTION)
        passed = transcription == expected
        return {
            "passed": passed,
            "reason": (
                None
                if passed
                else "normalized transcription did not exactly match the known fixture"
            ),
            "expected_transcription": expected,
            "normalized_transcription": transcription[:160],
        }
    if "ocr" in case.requires:
        transcription = _normalize_ocr_transcription(str(result.content))
        passed = transcription == _OCR_EXPECTED_TRANSCRIPTION
        return {
            "passed": passed,
            "reason": (
                None
                if passed
                else "normalized transcription did not exactly match the expected token"
            ),
            "expected_transcription": _OCR_EXPECTED_TRANSCRIPTION,
            "normalized_transcription": transcription[:160],
        }
    if "vision" in case.requires and "rerank" not in case.requires:
        passed = "red" in result.content.lower()
        return {"passed": passed, "reason": None if passed else "dominant color was not red"}
    if "rerank" in case.requires:
        if _is_multimodal_rerank_case(case):
            candidate_count = _MULTIMODAL_RERANK_CANDIDATE_COUNT
            relevant_index = _MULTIMODAL_RERANK_RELEVANT_INDEX
        else:
            _, candidates, relevant_index = _rerank_inputs(case)
            candidate_count = len(candidates)
        expected_indexes = list(range(candidate_count))
        if not result.finite:
            return {"passed": False, "reason": "rerank scores were not finite"}
        if int(result.candidate_count) != candidate_count:
            return {
                "passed": False,
                "reason": "rerank response returned the wrong candidate count",
            }
        scores = list(result.scores)
        ranking = list(result.ranking)
        if len(scores) != candidate_count:
            return {
                "passed": False,
                "reason": "rerank response returned the wrong score count",
            }
        if sorted(ranking) != expected_indexes:
            return {"passed": False, "reason": "rerank ranking was not a permutation"}
        if not ranking or int(ranking[0]) != relevant_index:
            return {
                "passed": False,
                "reason": f"rerank top candidate was not index {relevant_index}",
            }
        relevant_score = float(scores[relevant_index])
        passed = all(
            relevant_score > float(score)
            for index, score in enumerate(scores)
            if index != relevant_index
        )
        return {
            "passed": passed,
            "reason": None if passed else "rerank top score was not uniquely highest",
        }
    if "embeddings" in case.requires:
        passed = bool(result.finite and result.dimension > 0 and result.batch_size > 0)
        return {"passed": passed, "reason": None if passed else "embedding vector validation failed"}
    if _is_varied_context_needle(case):
        actual = _normalize_varied_context_answer(str(result.content))
        expected = _normalize_varied_context_answer(_varied_context_needle(case))
        passed = actual == expected
        return {
            "passed": passed,
            "reason": (
                None
                if passed
                else "recovery phrase did not exactly match the visible answer"
            ),
        }
    if _is_multi_hop_needle(case):
        passed = str(result.content).strip() == _multi_hop_needle(result.request_id)
        return {
            "passed": passed,
            "reason": (
                None if passed else "final multi-hop needle key was not returned exactly"
            ),
        }
    if str(case.id).startswith("long-context-needle"):
        passed = _needle(result.request_id) in result.content
        return {"passed": passed, "reason": None if passed else "needle was not returned"}
    return {"passed": False, "reason": "capability adapter is not implemented"}


def _run_warmups(server: Any, model: SimpleNamespace, case: SimpleNamespace) -> None:
    for index in range(int(case.warmups)):
        if str(case.kind) == "quality":
            item = _QUALITY_ITEMS[index % len(_QUALITY_ITEMS)]
            request_id = (
                f"warmup-{case.case_id}-{index}-{item.id}-{time.time_ns()}"
            )
            _chat_request_function(server, case)(
                **_quality_request_arguments(
                    server=server,
                    model=model,
                    case=case,
                    item=item,
                    request_id=request_id,
                    prompt_tag=f"warmup-{index}",
                )
            )
            continue
        request_id = (
            f"warmup-{case.id}-{index}"
            if _uses_matched_prompt_protocol(case)
            else f"warmup-{case.case_id}-{index}-{time.time_ns()}"
        )
        if _is_multimodal_embedding_case(case):
            multimodal_embedding_request(
                **_multimodal_embedding_request_arguments(
                    server=server,
                    model=model,
                    case=case,
                    request_id=request_id,
                )
            )
        elif "rerank" in case.requires:
            score_request(
                **_rerank_request_arguments(
                    server=server,
                    model=model,
                    case=case,
                    request_id=request_id,
                )
            )
        elif "embeddings" in case.requires:
            text = "measurement " * max(int(case.prompt_repetitions), 1)
            embedding_request(
                base_url=server.base_url,
                model=str(model.served_name),
                inputs=[f"{text} batch item {item}" for item in range(int(case.concurrency))],
                request_id=request_id,
                **_authorization_argument(server),
            )
        else:
            _chat_request_function(server, case)(
                **_request_arguments(server=server, model=model, case=case, request_id=request_id)
            )


def _prime_model(server: Any, model: SimpleNamespace) -> Any:
    request_id = f"first-request-after-start-{time.time_ns()}"
    if "rerank" in model.tasks and "chat" not in model.tasks:
        prime_case = SimpleNamespace(prompt_repetitions=0)
        return score_request(
            **_rerank_request_arguments(
                server=server,
                model=model,
                case=prime_case,
                request_id=request_id,
            )
        )
    if {"embeddings", "vision"}.issubset(set(model.tasks)) and "chat" not in model.tasks:
        prime_case = SimpleNamespace(prompt_repetitions=64, requires=["embeddings", "vision"])
        return multimodal_embedding_request(
            **_multimodal_embedding_request_arguments(
                server=server,
                model=model,
                case=prime_case,
                request_id=request_id,
            )
        )
    if "embeddings" in model.tasks and "chat" not in model.tasks:
        return embedding_request(
            base_url=server.base_url,
            model=str(model.served_name),
            inputs=["Spark benchmark model-load probe."],
            request_id=request_id,
            **_authorization_argument(server),
        )
    prime_case = SimpleNamespace(
        id="first-request-after-start",
        case_id="first-request-after-start",
        kind="decode",
        requires=["chat"],
        prompt_repetitions=0,
        max_output_tokens=8,
        temperature=0.0,
    )
    return _chat_request_function(server)(
        **_request_arguments(
            server=server,
            model=model,
            case=prime_case,
            request_id=request_id,
        )
    )


def _execute_prefix_cache_case(
    *,
    server: Any,
    model: SimpleNamespace,
    case: SimpleNamespace,
    journal: Journal,
    telemetry: TelemetrySampler,
) -> None:
    """Run the fixed serial same-slot llama.cpp prompt-KV A/B protocol."""

    attempt_id = uuid.uuid4().hex
    journal.append(
        {
            "event": "case_start",
            "case_id": case.case_id,
            "attempt_id": attempt_id,
            "kind": case.kind,
            "concurrency": case.concurrency,
        }
    )
    telemetry.set_phase(f"case:{case.case_id}:{attempt_id}")
    started = time.perf_counter()
    try:
        mode = _prefix_cache_control_mode(model, server, case)
        validations: list[dict[str, Any]] = []
        for pair_index in range(1, int(case.repetitions) + 1):
            for step_ordinal, (
                condition,
                cache_prompt,
                cache_prompt_control,
            ) in enumerate(
                _prefix_cache_steps(mode), start=1
            ):
                request_id = (
                    f"prefix-cache-{case.case_id}-p{pair_index}-{condition}-"
                    f"{time.time_ns()}"
                )
                before = snapshot_llamacpp_cache_metrics(server.base_url)
                result = stream_chat_request(
                    **_prefix_cache_request_arguments(
                        server=server,
                        model=model,
                        case=case,
                        pair_index=pair_index,
                        request_id=request_id,
                        cache_prompt=cache_prompt,
                    )
                )
                after = snapshot_llamacpp_cache_metrics(server.base_url)
                prometheus_metrics = delta_llamacpp_cache_metrics(before, after)
                validation = _validate_prefix_cache_result(
                    result=result,
                    prometheus_metrics=prometheus_metrics,
                    case=case,
                    condition=condition,
                )
                validations.append(validation)
                journal.append(
                    {
                        "event": "request_complete",
                        "case_id": case.case_id,
                        "attempt_id": attempt_id,
                        "kind": case.kind,
                        "repetition": pair_index - 1,
                        "burst_elapsed_s": result.elapsed_s,
                        "result": _prefix_cache_result_payload(
                            result,
                            mode=mode,
                            condition=condition,
                            pair_index=pair_index,
                            step_ordinal=step_ordinal,
                            cache_prompt_control=cache_prompt_control,
                            prefix_target_words=int(case.prompt_repetitions),
                            prometheus_metrics=prometheus_metrics,
                        ),
                        "validation": validation,
                    }
                )
        journal.append(
            {
                "event": "case_complete",
                "case_id": case.case_id,
                "attempt_id": attempt_id,
                "kind": case.kind,
                "concurrency": case.concurrency,
                "elapsed_s": time.perf_counter() - started,
                "validation_passed": all(item["passed"] for item in validations),
            }
        )
    except Exception as error:
        safe_error = PrefixCacheError()
        journal.append(
            {
                "event": "case_failed",
                "case_id": case.case_id,
                "attempt_id": attempt_id,
                "error_type": type(safe_error).__name__,
                "error": str(safe_error),
                "elapsed_s": time.perf_counter() - started,
            }
        )
        raise safe_error from error
    finally:
        telemetry.set_phase("between_cases")


def _execute_case(
    *,
    server: Any,
    model: SimpleNamespace,
    case: SimpleNamespace,
    journal: Journal,
    telemetry: TelemetrySampler,
) -> None:
    if _is_prefix_cache_case(case):
        _execute_prefix_cache_case(
            server=server,
            model=model,
            case=case,
            journal=journal,
            telemetry=telemetry,
        )
        return
    attempt_id = uuid.uuid4().hex
    journal.append(
        {
            "event": "case_start",
            "case_id": case.case_id,
            "attempt_id": attempt_id,
            "kind": case.kind,
            "concurrency": case.concurrency,
        }
    )
    telemetry.set_phase(f"warmup:{case.case_id}:{attempt_id}")
    started = time.perf_counter()
    try:
        _run_warmups(server, model, case)
        telemetry.set_phase(f"case:{case.case_id}:{attempt_id}")
        measured_started = time.perf_counter()
        validation_results: list[dict[str, Any]] = []
        for repetition in range(int(case.repetitions)):
            quality_items_by_request_id: dict[str, QualityItem] = {}
            quality_bursts_by_request_id: dict[str, float] = {}
            if str(case.kind) == "agentic":
                request_id_prefix = (
                    f"{case.case_id}-r{repetition}-w0-{time.time_ns()}"
                )
                request_body_json = getattr(model, "request_body_json", None)
                extra_body = json.loads(request_body_json) if request_body_json else {}
                result = run_agentic_scenario(
                    scenario_id=str(case.id),
                    variant=repetition,
                    request_function=stream_chat_request,
                    request_kwargs={
                        "base_url": server.base_url,
                        "model": str(model.served_name),
                        **_authorization_argument(server),
                    },
                    request_id_prefix=request_id_prefix,
                    max_turns=int(case.max_turns),
                    max_output_tokens=int(case.max_output_tokens),
                    temperature=float(case.temperature),
                    extra_body=extra_body,
                )
                results = [result]
                burst_s = float(result.wall_s)
            elif str(case.kind) == "memory":
                request_id_prefix = (
                    f"{case.case_id}-r{repetition}-w0-{time.time_ns()}"
                )
                request_body_json = getattr(model, "request_body_json", None)
                extra_body = json.loads(request_body_json) if request_body_json else {}
                request_kwargs: dict[str, Any] = {
                    "base_url": server.base_url,
                    "model": str(model.served_name),
                    "require_native_timing": True,
                    **_authorization_argument(server),
                }
                extra_body["cache_prompt"] = False
                result = run_memory_operation_scenario(
                    scenario_id=str(case.id),
                    variant=repetition,
                    request_function=_chat_request_function(server, case),
                    request_kwargs=request_kwargs,
                    request_id_prefix=request_id_prefix,
                    max_output_tokens=int(case.max_output_tokens),
                    temperature=float(case.temperature),
                    extra_body=extra_body,
                )
                results = [result]
                burst_s = float(result.elapsed_s)
            elif str(case.kind) == "quality":
                results = []
                configured_concurrency = int(case.concurrency)
                for offset in range(0, len(_QUALITY_ITEMS), configured_concurrency):
                    batch_items = _QUALITY_ITEMS[
                        offset : offset + configured_concurrency
                    ]
                    requests = []
                    for item in batch_items:
                        request_id = (
                            f"{case.case_id}-r{repetition}-{item.id}-{time.time_ns()}"
                        )
                        quality_items_by_request_id[request_id] = item
                        requests.append(
                            _quality_request_arguments(
                                server=server,
                                model=model,
                                case=case,
                                item=item,
                                request_id=request_id,
                                prompt_tag=f"r{repetition}",
                            )
                        )
                    batch_results, batch_s = concurrent_chat_requests(
                        requests=requests,
                        concurrency=len(requests),
                        request_function=_chat_request_function(server, case),
                    )
                    if len(batch_results) != len(requests):
                        raise RuntimeError(
                            "Quality batch returned the wrong number of responses"
                        )
                    for result in batch_results:
                        if result.request_id not in quality_items_by_request_id:
                            raise RuntimeError(
                                "Quality response request_id did not match its request"
                            )
                        quality_bursts_by_request_id[result.request_id] = batch_s
                    results.extend(batch_results)
                burst_s = 0.0
            elif _is_multimodal_embedding_case(case):
                requests = [
                    _multimodal_embedding_request_arguments(
                        server=server,
                        model=model,
                        case=case,
                        request_id=(
                            f"{case.case_id}-r{repetition}-w{worker}-{time.time_ns()}"
                        ),
                    )
                    for worker in range(int(case.concurrency))
                ]
                results, burst_s = concurrent_multimodal_embedding_requests(
                    requests=requests,
                    concurrency=int(case.concurrency),
                )
            elif "rerank" in case.requires:
                requests = [
                    _rerank_request_arguments(
                        server=server,
                        model=model,
                        case=case,
                        request_id=(
                            f"{case.case_id}-r{repetition}-w{worker}-{time.time_ns()}"
                        ),
                    )
                    for worker in range(int(case.concurrency))
                ]
                results, burst_s = concurrent_score_requests(
                    requests=requests,
                    concurrency=int(case.concurrency),
                )
            elif "embeddings" in case.requires:
                request_id = f"{case.case_id}-r{repetition}-w0-{time.time_ns()}"
                text = "measurement " * max(int(case.prompt_repetitions), 1)
                burst_started = time.perf_counter()
                results = [
                    embedding_request(
                        base_url=server.base_url,
                        model=str(model.served_name),
                        inputs=[
                            f"{text} batch item {item} nonce {request_id}"
                            for item in range(int(case.concurrency))
                        ],
                        request_id=request_id,
                        **_authorization_argument(server),
                    )
                ]
                burst_s = time.perf_counter() - burst_started
            else:
                requests = [
                    _request_arguments(
                        server=server,
                        model=model,
                        case=case,
                        request_id=(
                            f"{case.id}-r{repetition}-w{worker}"
                            if _uses_matched_prompt_protocol(case)
                            else (
                                f"{case.case_id}-r{repetition}-w{worker}-"
                                f"{time.time_ns()}"
                            )
                        ),
                    )
                    for worker in range(int(case.concurrency))
                ]
                results, burst_s = concurrent_chat_requests(
                    requests=requests,
                    concurrency=int(case.concurrency),
                    request_function=_chat_request_function(server, case),
                )
            for result in results:
                if str(case.kind) in {"agentic", "memory"}:
                    validation = {
                        "passed": bool(result.passed),
                        "reason": (
                            None if result.passed else str(result.failure_code)
                        ),
                    }
                    result_burst_s = burst_s
                elif str(case.kind) == "quality":
                    validation = _validate_quality_item(
                        quality_items_by_request_id[result.request_id], result
                    )
                    result_burst_s = quality_bursts_by_request_id[result.request_id]
                else:
                    validation = _validate_capability(case, result, model=model)
                    result_burst_s = burst_s
                if validation is not None:
                    validation_results.append(validation)
                journal.append(
                    {
                        "event": "request_complete",
                        "case_id": case.case_id,
                        "attempt_id": attempt_id,
                        "kind": case.kind,
                        "repetition": repetition,
                        "burst_elapsed_s": result_burst_s,
                        "result": (
                            _multi_hop_result_payload(model, result)
                            if _is_scalar_safe_needle(case)
                            else _request_result_payload(model, result)
                        ),
                        "validation": validation,
                    }
                )
        elapsed_s = time.perf_counter() - measured_started
    except Exception as error:
        if _is_scalar_safe_needle(case):
            safe_error = (
                VariedContextNeedleError()
                if _is_varied_context_needle(case)
                else MultiHopNeedleError()
            )
            journal.append(
                {
                    "event": "case_failed",
                    "case_id": case.case_id,
                    "attempt_id": attempt_id,
                    "error_type": type(safe_error).__name__,
                    "error": str(safe_error),
                    "elapsed_s": time.perf_counter() - started,
                }
            )
            raise safe_error from error
        if str(case.kind) == "memory":
            safe_error = MemoryOperationError(
                "memory-operation case failed; details omitted"
            )
            journal.append(
                {
                    "event": "case_failed",
                    "case_id": case.case_id,
                    "attempt_id": attempt_id,
                    "error_type": type(safe_error).__name__,
                    "error": str(safe_error),
                    "elapsed_s": time.perf_counter() - started,
                }
            )
            raise safe_error from error
        journal.append(
            {
                "event": "case_failed",
                "case_id": case.case_id,
                "attempt_id": attempt_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_s": time.perf_counter() - started,
            }
        )
        raise
    else:
        journal.append(
            {
                "event": "case_complete",
                "case_id": case.case_id,
                "attempt_id": attempt_id,
                "kind": case.kind,
                "concurrency": case.concurrency,
                "elapsed_s": elapsed_s,
                "validation_passed": (
                    all(item["passed"] for item in validation_results)
                    if validation_results
                    else None
                ),
            }
        )
    finally:
        telemetry.set_phase("between_cases")


def _require_sm121_storage_quality_gate(
    journal: Journal, case: SimpleNamespace
) -> None:
    """Require exactly one clean durable quality result before lifetime two.

    The generic executor intentionally records validation failures without
    throwing, so ordinary suites can finish and report every case. This
    candidate is different: its first lifetime is a strict admission gate.
    Consult the just-written scalar journal record rather than response text,
    and fail closed if the record is missing, duplicated, or not clean.
    """

    completions = [
        event
        for event in journal.events()
        if event.get("event") == "case_complete"
        and event.get("case_id") == case.case_id
    ]
    if len(completions) != 1 or completions[0].get("validation_passed") is not True:
        raise SM121StorageQualityGateError()


def _sm121_cache_semantic_messages() -> tuple[list[dict[str, str]], ...]:
    """Build T0/T1/T2 only in memory for the paired semantic probe.

    The renderer has no external inputs.  Each later turn appends fixed,
    already-verified synthetic assistant history to the immediately prior
    message list; it never uses model text as a new prompt component.
    """

    system = {
        "role": "system",
        "content": (
            "Follow the synthetic ledger protocol exactly. Reply with only the "
            "requested token and no explanation."
        ),
    }
    ledger = _SM121_CACHE_SEMANTIC_LEDGER_WORD * _SM121_CACHE_SEMANTIC_LEDGER_REPETITIONS
    initial_user = {
        "role": "user",
        "content": (
            "Read the complete synthetic ledger before replying.\n"
            + ledger
            + "\nReturn exactly "
            + _SM121_CACHE_SEMANTIC_EXPECTED_RESPONSES[0]
        ),
    }
    t0 = [system, initial_user]
    t1 = [
        *t0,
        {"role": "assistant", "content": _SM121_CACHE_SEMANTIC_EXPECTED_RESPONSES[0]},
        {
            "role": "user",
            "content": (
                "Keep the same ledger history and return exactly "
                + _SM121_CACHE_SEMANTIC_EXPECTED_RESPONSES[1]
            ),
        },
    ]
    t2 = [
        *t1,
        {"role": "assistant", "content": _SM121_CACHE_SEMANTIC_EXPECTED_RESPONSES[1]},
        {
            "role": "user",
            "content": (
                "Keep the same ledger history and return exactly "
                + _SM121_CACHE_SEMANTIC_EXPECTED_RESPONSES[2]
            ),
        },
    ]
    return t0, t1, t2


def _sm121_cache_semantic_common_prefix_tokens(
    first: tuple[int, ...], second: tuple[int, ...]
) -> int:
    """Count a private token-prefix overlap without storing token IDs."""

    common = 0
    for left, right in zip(first, second):
        if left != right:
            break
        common += 1
    return common


def _sm121_cache_semantic_static_event(
    *, model: SimpleNamespace, arm: str, lifetime: int
) -> dict[str, Any]:
    """Bind one fresh semantic lifetime to reviewed cache-source roles."""

    event = {
        "event": SM121_CACHE_SEMANTIC_STATIC_ATTESTATION_EVENT,
        "arm": arm,
        "fresh_server_lifetime": lifetime,
        "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
        **inspect_sm121_cache_source_digests(model),
        **SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
    }
    try:
        validate_sm121_cache_semantic_static_attestation_event(event)
    except SM121CacheSemanticError as error:
        raise SM121CacheSemanticRequestError() from error
    return event


def _sm121_cache_semantic_runtime_event(
    *, server: Any, arm: str, lifetime: int
) -> dict[str, Any]:
    """Record only the resolved scalar cache identity for a fresh server."""

    event = {
        "event": SM121_CACHE_SEMANTIC_RUNTIME_ATTESTATION_EVENT,
        "arm": arm,
        "fresh_server_lifetime": lifetime,
        **inspect_sm121_cache_runtime_identity(server),
    }
    try:
        validate_sm121_cache_semantic_runtime_attestation_event(event)
    except SM121CacheSemanticError as error:
        raise SM121CacheSemanticRequestError() from error
    return event


def _sm121_cache_semantic_turn_event(
    *,
    case: SimpleNamespace,
    attempt_id: str,
    arm: str,
    turn: str,
    result: dict[str, Any],
    before: dict[str, Any],
    before_polls: int,
    before_settled: bool,
    after: dict[str, Any],
    after_polls: int,
    after_settled: bool,
    append_only_prompt_identity_verified: bool,
    cross_arm_prompt_identity_verified: bool | None,
    shared_prefix_tokens: int,
) -> dict[str, Any]:
    """Project one private turn into a validated scalar-only observation."""

    event: dict[str, Any] = {
        "event": SM121_CACHE_SEMANTIC_TURN_OBSERVATION_EVENT,
        "case_id": case.case_id,
        "protocol_case_id": SM121_CACHE_SEMANTIC_CASE_ID,
        "attempt_id": attempt_id,
        "turn": turn,
        "arm": arm,
        "cache_details_requested": True,
        "prompt_token_ids_requested": True,
        "streaming": False,
        "thinking_disabled": True,
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "reasoning_tokens": result["reasoning_tokens"],
        "append_only_prompt_identity_verified": append_only_prompt_identity_verified,
        "cross_arm_prompt_identity_verified": cross_arm_prompt_identity_verified,
        "shared_prefix_tokens": shared_prefix_tokens,
        "response_detail_state": result["response_detail_state"],
        "usage_detail_state": result["usage_detail_state"],
        "response_device_cached_tokens": result["response_device_cached_tokens"],
        "response_host_cached_tokens": result["response_host_cached_tokens"],
        "response_storage_cached_tokens": result["response_storage_cached_tokens"],
        "usage_cached_tokens": result["usage_cached_tokens"],
        "metrics_available": bool(
            before.get("available") is True and after.get("available") is True
        ),
        "guardrail_metrics_available": bool(
            before.get("guardrail_metrics_available") is True
            and after.get("guardrail_metrics_available") is True
        ),
        "metrics_before_polls": before_polls,
        "metrics_after_polls": after_polls,
        "metrics_before_settled": before_settled,
        "metrics_after_settled": after_settled,
    }
    for prefix, snapshot in (("before", before), ("after", after)):
        for metric in SM121_CACHE_SEMANTIC_METRIC_FIELDS:
            event[f"{prefix}_{metric}"] = snapshot[metric]
        for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
            event[f"{prefix}_cached_{source}_series_present"] = snapshot[
                f"cached_{source}_series_present"
            ]
    for metric in SM121_CACHE_SEMANTIC_METRIC_FIELDS:
        event[f"delta_{metric}"] = event[f"after_{metric}"] - event[
            f"before_{metric}"
        ]
    try:
        admitted, basis = derive_sm121_cache_semantic_turn_admission(event)
    except SM121CacheSemanticError as error:
        raise SM121CacheSemanticRequestError() from error
    event["semantic_turn_admitted"] = admitted
    event["semantic_turn_basis"] = basis
    try:
        validate_sm121_cache_semantic_turn_event(event)
    except SM121CacheSemanticError as error:
        raise SM121CacheSemanticRequestError() from error
    return event


def _execute_sm121_cache_semantic_case(
    *,
    server: Any,
    model: SimpleNamespace,
    case: SimpleNamespace,
    arm: str,
    control_prompt_token_ids: tuple[tuple[int, ...], ...] | None,
    journal: Journal,
    telemetry: TelemetrySampler,
) -> tuple[tuple[int, ...], ...]:
    """Run T0/T1/T2 in one fresh lifetime and retain prompt IDs only in RAM."""

    if str(case.id) != SM121_CACHE_SEMANTIC_CASE_ID:
        raise SM121CacheSemanticRequestError()
    if arm == "A" and (
        control_prompt_token_ids is None
        or len(control_prompt_token_ids) != len(SM121_CACHE_SEMANTIC_TURN_ORDER)
    ):
        raise SM121CacheSemanticRequestError()
    if arm == "B" and control_prompt_token_ids is not None:
        raise SM121CacheSemanticRequestError()
    attempt_id = uuid.uuid4().hex
    journal.append(
        {
            "event": "case_start",
            "case_id": case.case_id,
            "attempt_id": attempt_id,
            "kind": case.kind,
            "concurrency": case.concurrency,
        }
    )
    telemetry.set_phase(f"case:{case.case_id}:{attempt_id}")
    started = time.perf_counter()
    private_turn_ids: list[tuple[int, ...]] = []
    turn_events: list[dict[str, Any]] = []
    try:
        for index, (turn, messages, expected_response) in enumerate(
            zip(
                SM121_CACHE_SEMANTIC_TURN_ORDER,
                _sm121_cache_semantic_messages(),
                _SM121_CACHE_SEMANTIC_EXPECTED_RESPONSES,
                strict=True,
            )
        ):
            before, _ignored_before_wait_s, before_polls, before_settled = (
                settle_sm121_cache_observability_metrics(server, semantic_arm=arm)
            )
            request_started = time.perf_counter()
            result = request_sm121_cache_semantic_turn(
                server,
                served_name=model.served_name,
                messages=messages,
                expected_response=expected_response,
                max_tokens=int(case.max_output_tokens),
            )
            request_elapsed_s = time.perf_counter() - request_started
            after, _ignored_after_wait_s, after_polls, after_settled = (
                settle_sm121_cache_observability_metrics(server, semantic_arm=arm)
            )
            prompt_token_ids = result.pop("private_prompt_token_ids", None)
            if (
                not isinstance(prompt_token_ids, tuple)
                or not prompt_token_ids
                or any(type(token) is not int or token < 0 for token in prompt_token_ids)
            ):
                raise SM121CacheSemanticRequestError()
            if index == 0:
                shared_prefix_tokens = 0
                append_verified = True
            else:
                shared_prefix_tokens = _sm121_cache_semantic_common_prefix_tokens(
                    private_turn_ids[-1], prompt_token_ids
                )
                append_verified = (
                    shared_prefix_tokens >= SM121_CACHE_SEMANTIC_COLD_INPUT_MIN_TOKENS
                )
            if arm == "A":
                assert control_prompt_token_ids is not None
                if prompt_token_ids != control_prompt_token_ids[index]:
                    # Do not journal an ID-derived mismatch or a digest. The A
                    # arm is not an admissible matched workload in that case.
                    raise SM121CacheSemanticRequestError()
                cross_arm_verified: bool | None = True
            else:
                cross_arm_verified = None
            event = _sm121_cache_semantic_turn_event(
                case=case,
                attempt_id=attempt_id,
                arm=arm,
                turn=turn,
                result=result,
                before=before,
                before_polls=before_polls,
                before_settled=before_settled,
                after=after,
                after_polls=after_polls,
                after_settled=after_settled,
                append_only_prompt_identity_verified=append_verified,
                cross_arm_prompt_identity_verified=cross_arm_verified,
                shared_prefix_tokens=shared_prefix_tokens,
            )
            journal.append(event)
            journal.append(
                {
                    "event": "request_complete",
                    "case_id": case.case_id,
                    "attempt_id": attempt_id,
                    "kind": case.kind,
                    "repetition": index,
                    "burst_elapsed_s": request_elapsed_s,
                    "result": {
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "reasoning_tokens": result["reasoning_tokens"],
                        "ttft_s": None,
                        "elapsed_s": request_elapsed_s,
                        "decode_s": None,
                        "decode_tps": None,
                        "output_tps": None,
                        "emission_events": 1,
                        "finish_reason": None,
                        "response_model": None,
                        "decode_metric_source": None,
                    },
                    "validation": {"passed": event["semantic_turn_admitted"]},
                }
            )
            turn_events.append(event)
            private_turn_ids.append(prompt_token_ids)
        validation_passed = all(
            event["semantic_turn_admitted"] for event in turn_events
        )
        journal.append(
            {
                "event": "case_complete",
                "case_id": case.case_id,
                "attempt_id": attempt_id,
                "kind": case.kind,
                "concurrency": case.concurrency,
                "elapsed_s": time.perf_counter() - started,
                "validation_passed": validation_passed,
            }
        )
        return tuple(private_turn_ids)
    except BaseException as error:
        safe_error = SM121CacheSemanticRequestError()
        journal.append(
            {
                "event": "case_failed",
                "case_id": case.case_id,
                "attempt_id": attempt_id,
                "error_type": type(safe_error).__name__,
                "error": str(safe_error),
                "elapsed_s": time.perf_counter() - started,
            }
        )
        raise safe_error from None
    finally:
        telemetry.set_phase("between_cases")


def _execute_sm121_cache_semantic_lifetime(
    *,
    run_dir: Path,
    workspace: Path,
    model: SimpleNamespace,
    arm: str,
    lifetime: int,
    case: SimpleNamespace,
    control_prompt_token_ids: tuple[tuple[int, ...], ...] | None,
    journal: Journal,
    telemetry: TelemetrySampler,
) -> tuple[tuple[int, ...], ...] | None:
    """Execute one fully isolated quality or semantic server lifetime."""

    server = None
    watchdog: HostSafetyWatchdog | None = None
    terminal_error: BaseException | None = None
    private_ids: tuple[tuple[int, ...], ...] | None = None
    phase = "quality" if lifetime == 1 else "semantic"
    try:
        journal.append(
            _sm121_cache_semantic_static_event(
                model=model, arm=arm, lifetime=lifetime
            )
        )
        watchdog = _host_safety_watchdog(model)
        if watchdog is not None:
            watchdog.start()
        telemetry.set_phase(f"server_startup:{lifetime}")
        callbacks: dict[str, Any] = {}
        if watchdog is not None:
            callbacks = {
                "abort_check": watchdog.raise_if_tripped,
                "on_server_created": (
                    lambda created_server: watchdog.register_abort_callback(
                        created_server.interrupt_owned
                    )
                ),
            }
        server = start_server(
            model,
            workspace=workspace,
            allow_download=False,
            server_log_path=run_dir / "server" / f"lifetime-{lifetime}" / "server.log",
            **callbacks,
        )
        if watchdog is not None:
            watchdog.raise_if_tripped()
        journal.append(
            _sm121_cache_semantic_runtime_event(
                server=server, arm=arm, lifetime=lifetime
            )
        )
        journal.append(
            {
                "event": "server_ready",
                "backend": server.backend,
                "fresh_server_lifetime": lifetime,
                "first_inference_is_case": True,
                "case_id": case.case_id,
            }
        )
        telemetry.set_phase(f"first_case_after_start:{lifetime}")
        if phase == "quality":
            _execute_case(
                server=server,
                model=model,
                case=case,
                journal=journal,
                telemetry=telemetry,
            )
            _require_sm121_storage_quality_gate(journal, case)
        else:
            private_ids = _execute_sm121_cache_semantic_case(
                server=server,
                model=model,
                case=case,
                arm=arm,
                control_prompt_token_ids=control_prompt_token_ids,
                journal=journal,
                telemetry=telemetry,
            )
        if watchdog is not None:
            watchdog.raise_if_tripped()
    except BaseException as error:
        terminal_error = watchdog.failure if watchdog is not None and watchdog.failure else error
    finally:
        telemetry.set_phase(f"server_shutdown:{lifetime}")
        cleanup_error: BaseException | None = None
        if server is not None:
            if watchdog is not None and watchdog.tripped:
                try:
                    _retry_host_safety_interrupt_if_needed(server, watchdog)
                    _record_host_safety_interrupt_failure(
                        journal, watchdog, stage=f"semantic_{phase}_lifetime"
                    )
                except BaseException as error:
                    cleanup_error = error
            try:
                save_server_logs(
                    server, run_dir / "server" / f"lifetime-{lifetime}" / "server.log"
                )
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            try:
                server.stop()
                journal.append(
                    {
                        "event": "server_stopped",
                        "backend": server.backend,
                        "fresh_server_lifetime": lifetime,
                    }
                )
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        if watchdog is not None:
            watchdog.stop()
            if terminal_error is None:
                try:
                    watchdog.raise_if_tripped()
                except BaseException as error:
                    terminal_error = error
        if cleanup_error is not None:
            journal.append(
                {"event": "cleanup_failed", "error_type": type(cleanup_error).__name__}
            )
            if terminal_error is None:
                terminal_error = cleanup_error
    if terminal_error is not None:
        raise terminal_error
    return private_ids


def _sm121_cache_semantic_case_ids(suite: SimpleNamespace) -> tuple[str, str]:
    cases = list(getattr(suite, "cases", ()))
    if len(cases) != 2:
        raise PreflightError("SM121 semantic-cache cases are invalid")
    quality, semantic = cases
    if (
        str(getattr(quality, "id", "")) != SM121_CACHE_SEMANTIC_QUALITY_CASE_ID
        or str(getattr(semantic, "id", "")) != SM121_CACHE_SEMANTIC_CASE_ID
        or not isinstance(getattr(quality, "case_id", None), str)
        or not isinstance(getattr(semantic, "case_id", None), str)
    ):
        raise PreflightError("SM121 semantic-cache cases are invalid")
    return quality.case_id, semantic.case_id


def _sm121_cache_semantic_arm_complete(
    *,
    summary: dict[str, Any],
    journal: Journal,
    suite: SimpleNamespace,
    arm: str,
) -> bool:
    """Require a complete summary and exact scalar-only arm lifecycle."""

    planned_case_ids = _sm121_cache_semantic_case_ids(suite)
    issues = sm121_cache_semantic_lifecycle_issues(
        journal.events(), planned_case_ids=planned_case_ids, arm=arm
    )
    return summary.get("status") == "complete" and not issues


def _execute_sm121_cache_semantic_arm(
    *,
    plan: dict[str, Any],
    model: SimpleNamespace,
    suite: SimpleNamespace,
    run_dir: Path,
    workspace: Path,
    cache_off_plan_fingerprint: str | None,
    cache_off_audit_passed: bool | None,
    cache_off_terminal_receipt_sha256: str | None,
    control_prompt_token_ids: tuple[tuple[int, ...], ...] | None,
) -> tuple[dict[str, Any], tuple[tuple[int, ...], ...] | None]:
    """Execute one arm with two fresh lifetimes; it is intentionally non-resumable."""

    journal = Journal(run_dir / "events.jsonl")
    if journal.events():
        raise PreflightError(
            "SM121 semantic-cache canary is non-resumable; freeze a new pair"
        )
    arm = sm121_cache_semantic_arm(model)
    cases = list(suite.cases)
    planned_case_ids = _sm121_cache_semantic_case_ids(suite)
    for case in cases:
        missing = set(case.requires) - set(model.tasks)
        if missing:
            raise PreflightError(
                "SM121 semantic-cache case has unsupported capabilities"
            )
    if (
        SM121_CACHE_SEMANTIC_COLD_INPUT_MAX_TOKENS
        + int(cases[1].max_output_tokens)
        + 1_024
        > int(model.max_context)
    ):
        raise PreflightError("SM121 semantic-cache context admission is insufficient")
    binding = getattr(model, "semantic_pair_binding", None)
    if not isinstance(binding, dict):
        raise PreflightError("SM121 semantic-cache pair binding is unavailable")
    journal.append(
        {
            "event": "run_start",
            "execution_mode": SM121_CACHE_SEMANTIC_EXECUTION_MODE,
            "arm": arm,
            "plan_fingerprint": str(plan["fingerprint"]),
            "semantic_pair_binding_sha256": binding["pair_binding_sha256"],
            "cache_off_plan_fingerprint": cache_off_plan_fingerprint,
            "cache_off_audit_passed": cache_off_audit_passed,
            "cache_off_terminal_receipt_sha256": cache_off_terminal_receipt_sha256,
        }
    )
    # These are ordering markers, not duration measurements.  The semantic
    # evidence lane rejects elapsed/monotonic fields for this protocol.
    journal.append({"event": "measurement_started"})
    telemetry = TelemetrySampler(run_dir / "telemetry.jsonl")
    stage = "preflight"
    private_ids: tuple[tuple[int, ...], ...] | None = None
    try:
        _preflight(model)
        stage = "quality_lifetime"
        _execute_sm121_cache_semantic_lifetime(
            run_dir=run_dir,
            workspace=workspace,
            model=model,
            arm=arm,
            lifetime=1,
            case=cases[0],
            control_prompt_token_ids=None,
            journal=journal,
            telemetry=telemetry,
        )
        stage = "semantic_lifetime"
        private_ids = _execute_sm121_cache_semantic_lifetime(
            run_dir=run_dir,
            workspace=workspace,
            model=model,
            arm=arm,
            lifetime=2,
            case=cases[1],
            control_prompt_token_ids=control_prompt_token_ids,
            journal=journal,
            telemetry=telemetry,
        )
        if private_ids is None:
            raise SM121CacheSemanticRequestError()
    except BaseException as error:
        safe_error: BaseException
        if isinstance(error, (HostSafetyError, PreflightError, SM121CacheSemanticRequestError)):
            safe_error = error
        else:
            safe_error = SM121CacheSemanticRequestError()
        if isinstance(safe_error, HostSafetyError):
            _record_host_safety_breach(journal, safe_error, stage=stage)
        _record_run_aborted(journal, safe_error, stage=stage)
        try:
            summarize_run(run_dir)
        except Exception as summary_error:
            journal.append(
                {"event": "summary_failed", "error_type": type(summary_error).__name__}
            )
        raise safe_error from None
    finally:
        # Not started: this is only a no-op phase carrier for the generic
        # quality plumbing, and therefore produces no telemetry file.
        telemetry.stop()
    journal.append({"event": "measurement_complete"})
    journal.append({"event": "run_complete", "status": "completed"})
    summary = summarize_run(run_dir)
    if not _sm121_cache_semantic_arm_complete(
        summary=summary, journal=journal, suite=suite, arm=arm
    ):
        # The journal remains complete and auditable; the controller will not
        # promote a partial B control into the cache-on arm.
        return summary, private_ids
    return summary, private_ids


def _validate_sm121_cache_semantic_pair_plans(
    cache_off_plan: dict[str, Any],
    cache_off_model: SimpleNamespace,
    cache_off_suite: SimpleNamespace,
    cache_on_plan: dict[str, Any],
    cache_on_model: SimpleNamespace,
    cache_on_suite: SimpleNamespace,
) -> None:
    """Verify reciprocal scalar plan bindings before any arm is allowed to start."""

    try:
        validate_sm121_cache_semantic_pair(cache_off_model, cache_on_model)
        validate_sm121_cache_semantic_suite(cache_off_suite)
        validate_sm121_cache_semantic_suite(cache_on_suite)
        off_binding = cache_off_plan.get("semantic_pair")
        on_binding = cache_on_plan.get("semantic_pair")
        off_fingerprint = cache_off_plan.get("fingerprint")
        on_fingerprint = cache_on_plan.get("fingerprint")
        validate_sm121_cache_semantic_pair_binding(
            off_binding,
            cache_off_model,
            cache_off_suite,
            peer_plan_fingerprint=on_fingerprint,
            peer_binding=on_binding,
        )
        validate_sm121_cache_semantic_pair_binding(
            on_binding,
            cache_on_model,
            cache_on_suite,
            peer_plan_fingerprint=off_fingerprint,
            peer_binding=off_binding,
        )
        if (
            not isinstance(off_binding, dict)
            or not isinstance(on_binding, dict)
            or off_binding.get("peer_plan_fingerprint") != on_fingerprint
            or on_binding.get("peer_plan_fingerprint") != off_fingerprint
        ):
            raise SM121CacheSemanticError("semantic pair reciprocal binding is invalid")
        pair_instance_sha256 = sm121_cache_semantic_pair_instance_sha256(
            cache_off_plan.get("run_nonce"), cache_on_plan.get("run_nonce")
        )
        if (
            off_binding.get("pair_instance_sha256") != pair_instance_sha256
            or on_binding.get("pair_instance_sha256") != pair_instance_sha256
        ):
            raise SM121CacheSemanticError("semantic pair instance binding is invalid")
    except SM121CacheSemanticError as error:
        raise PreflightError("SM121 semantic-cache pair binding is invalid") from error


def execute_sm121_cache_semantic_canary(
    cache_off_run_dir: Path, cache_on_run_dir: Path, *, workspace: Path
) -> dict[str, Any]:
    """Run B then A in four fresh lifetimes without making a speed claim."""

    cache_off_plan, cache_off_model, cache_off_suite = _load_sm121_cache_semantic_plan(
        cache_off_run_dir
    )
    cache_on_plan, cache_on_model, cache_on_suite = _load_sm121_cache_semantic_plan(
        cache_on_run_dir
    )
    _validate_sm121_cache_semantic_pair_plans(
        cache_off_plan,
        cache_off_model,
        cache_off_suite,
        cache_on_plan,
        cache_on_model,
        cache_on_suite,
    )
    lock_path = results_lock_path(workspace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another SparkBench run holds the benchmark lock") from error
        cache_off_summary, control_prompt_token_ids = _execute_sm121_cache_semantic_arm(
            plan=cache_off_plan,
            model=cache_off_model,
            suite=cache_off_suite,
            run_dir=cache_off_run_dir,
            workspace=workspace,
            cache_off_plan_fingerprint=None,
            cache_off_audit_passed=None,
            cache_off_terminal_receipt_sha256=None,
            control_prompt_token_ids=None,
        )
        cache_off_journal = Journal(cache_off_run_dir / "events.jsonl")
        cache_off_complete = _sm121_cache_semantic_arm_complete(
            summary=cache_off_summary,
            journal=cache_off_journal,
            suite=cache_off_suite,
            arm="B",
        )
        if not cache_off_complete or control_prompt_token_ids is None:
            return {
                "status": "partial",
                "cache_off": cache_off_summary,
                "cache_on": None,
                "cache_on_started": False,
            }
        cache_off_binding = cache_off_plan.get("semantic_pair")
        cache_on_binding = cache_on_plan.get("semantic_pair")
        if not isinstance(cache_off_binding, dict) or not isinstance(cache_on_binding, dict):
            raise PreflightError("SM121 semantic-cache pair binding is unavailable")
        try:
            cache_off_terminal_receipt_sha256 = (
                sm121_cache_semantic_cache_off_receipt_sha256(
                    cache_on_binding.get("pair_instance_sha256"),
                    cache_off_plan.get("fingerprint"),
                    cache_off_binding.get("pair_binding_sha256"),
                )
            )
        except SM121CacheSemanticError as error:
            raise PreflightError(
                "SM121 semantic-cache cache-off receipt is unavailable"
            ) from error
        cache_on_summary, _private_cache_on_ids = _execute_sm121_cache_semantic_arm(
            plan=cache_on_plan,
            model=cache_on_model,
            suite=cache_on_suite,
            run_dir=cache_on_run_dir,
            workspace=workspace,
            cache_off_plan_fingerprint=str(cache_off_plan["fingerprint"]),
            cache_off_audit_passed=True,
            cache_off_terminal_receipt_sha256=cache_off_terminal_receipt_sha256,
            control_prompt_token_ids=control_prompt_token_ids,
        )
        del control_prompt_token_ids
        del _private_cache_on_ids
        cache_on_complete = _sm121_cache_semantic_arm_complete(
            summary=cache_on_summary,
            journal=Journal(cache_on_run_dir / "events.jsonl"),
            suite=cache_on_suite,
            arm="A",
        )
        return {
            "status": "complete" if cache_on_complete else "partial",
            "cache_off": cache_off_summary,
            "cache_on": cache_on_summary,
            "cache_on_started": True,
        }


def create_sm121_cache_performance_campaign(
    *,
    cache_on_model: Any,
    cache_off_model: Any,
    suite: Any,
    results_root: Path,
    models_path: Path,
    suite_path: Path,
    evidence_root: Path,
) -> Path:
    """Freeze one fresh, non-resumable A/B/B/A cache-performance campaign.

    The campaign is intentionally separate from normal plans and from the
    semantic B-then-A lane.  It does not start a server.  Four plan nonces are
    committed only through one opaque instance digest; neither nonce is
    exported and no prompt material is ever written here.
    """

    try:
        validate_sm121_cache_performance_pair(cache_on_model, cache_off_model)
        validate_sm121_cache_performance_suite(suite)
        _validate_sm121_cache_performance_prerequisites(evidence_root)
    except SM121CachePerformanceError as error:
        raise RuntimeError("SM121 cache-performance admission is unavailable") from error
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign_dir = results_root / (
        f"{stamp}-qwen38-flash-next-sm121-cache-policy-performance-v1"
    )
    campaign_dir.mkdir(parents=True, exist_ok=False)
    runs_root = campaign_dir / "runs"
    arm_models = {
        "A": cache_on_model,
        "B": cache_off_model,
    }
    run_dirs: list[Path] = []
    try:
        for ordinal, arm in enumerate(SM121_CACHE_PERFORMANCE_ARM_ORDER, start=1):
            run_dirs.append(
                create_plan(
                    model=arm_models[arm],
                    suite=suite,
                    results_root=runs_root,
                    models_path=models_path,
                    suite_path=suite_path,
                    allow_sm121_cache_performance=True,
                    run_label=f"performance-{ordinal}-{arm.lower()}",
                )
            )
        _bind_sm121_cache_performance_campaign_plans(run_dirs)
        plans = [json.loads((run_dir / "plan.json").read_text()) for run_dir in run_dirs]
        binding = plans[0].get("cache_performance_pair")
        if not isinstance(binding, dict):
            raise RuntimeError("SM121 cache-performance plan binding is unavailable")
        campaign = {
            "schema_version": 1,
            "campaign_id": SM121_CACHE_PERFORMANCE_CAMPAIGN_ID,
            "created_at": utc_now(),
            "execution_mode": SM121_CACHE_PERFORMANCE_EXECUTION_MODE,
            "prerequisite_bundle_sha256s": list(
                SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S
            ),
            "pair_binding": binding,
            "run_directories": [run_dir.name for run_dir in run_dirs],
        }
        campaign["integrity_hash"] = content_hash(campaign, 64)
        write_json(campaign_dir / "campaign.json", campaign)
    except BaseException:
        # The root is deliberately retained as ignored raw provenance.  It has
        # no execution side effect and a later evidence export will reject an
        # incomplete topology rather than treating it as a measurement.
        raise
    return campaign_dir


def _validate_sm121_cache_performance_prerequisites(evidence_root: Path) -> None:
    """Require the exact prior scalar capability evidence before planning."""

    # An index row by itself is not evidence: verify its checksums, topology,
    # scalar schemas, and required bundle-hash containment before accepting an
    # opaque prerequisite commitment.
    from .evidence import EvidenceError, verify_sm121_cache_performance_prerequisites

    try:
        verify_sm121_cache_performance_prerequisites(evidence_root)
    except (OSError, EvidenceError) as error:
        raise SM121CachePerformanceError(
            "cache-performance prerequisite evidence is unverifiable"
        ) from error


def _bind_sm121_cache_performance_campaign_plans(run_dirs: list[Path]) -> None:
    """Bind four frozen A/B/B/A plans with a scalar-only commitment."""

    if len(run_dirs) != len(SM121_CACHE_PERFORMANCE_ARM_ORDER):
        raise RuntimeError("SM121 cache-performance plan count is invalid")
    plans: list[dict[str, Any]] = []
    for ordinal, (run_dir, arm) in enumerate(
        zip(run_dirs, SM121_CACHE_PERFORMANCE_ARM_ORDER, strict=True), start=1
    ):
        try:
            plan = json.loads((run_dir / "plan.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("SM121 cache-performance plan is unreadable") from error
        if type(plan) is not dict:
            raise RuntimeError("SM121 cache-performance plan is invalid")
        model = plan.get("model")
        suite = plan.get("suite")
        if type(model) is not dict or type(suite) is not dict:
            raise RuntimeError("SM121 cache-performance plan is invalid")
        try:
            if sm121_cache_performance_arm(model) != arm:
                raise SM121CachePerformanceError("cache-performance plan arm changed")
            validate_sm121_cache_performance_candidate(model)
            validate_sm121_cache_performance_suite(_namespace(suite))
        except SM121CachePerformanceError as error:
            raise RuntimeError("SM121 cache-performance plan is invalid") from error
        if not isinstance(plan.get("fingerprint"), str) or re.fullmatch(
            r"[0-9a-f]{16}", plan["fingerprint"]
        ) is None:
            raise RuntimeError("SM121 cache-performance fingerprint is invalid")
        plan["cache_performance_ordinal"] = ordinal
        plans.append(plan)
    try:
        instance = sm121_cache_performance_pair_instance_sha256(
            [plan.get("run_nonce") for plan in plans]
        )
    except SM121CachePerformanceError as error:
        raise RuntimeError("SM121 cache-performance plan nonce is invalid") from error
    binding: dict[str, object] = {
        "schema_version": SM121_CACHE_PERFORMANCE_PAIR_BINDING_SCHEMA_VERSION,
        "suite_id": SM121_CACHE_PERFORMANCE_SUITE_ID,
        "execution_mode": SM121_CACHE_PERFORMANCE_EXECUTION_MODE,
        "arm_order": list(SM121_CACHE_PERFORMANCE_ARM_ORDER),
        "profile_ids": [
            SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID,
            SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID,
        ],
        "quality_case_id": SM121_CACHE_PERFORMANCE_QUALITY_CASE_ID,
        "timed_case_id": SM121_CACHE_PERFORMANCE_CASE_ID,
        "cell_timeout_s": SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S,
        "campaign_instance_sha256": instance,
        "plan_fingerprints": [str(plan["fingerprint"]) for plan in plans],
    }
    binding["pair_binding_sha256"] = sm121_cache_performance_pair_binding_sha256(
        binding
    )
    try:
        validate_sm121_cache_performance_pair_binding(binding)
    except SM121CachePerformanceError as error:
        raise RuntimeError("SM121 cache-performance pair binding is invalid") from error
    for run_dir, plan in zip(run_dirs, plans, strict=True):
        plan["cache_performance_pair"] = binding
        plan["integrity_hash"] = content_hash(
            {key: value for key, value in plan.items() if key != "integrity_hash"},
            64,
        )
        write_json(run_dir / "plan.json", plan)


def _load_sm121_cache_performance_plan(
    run_dir: Path,
) -> tuple[dict[str, Any], SimpleNamespace, SimpleNamespace]:
    """Load one frozen performance arm without accepting generic plans."""

    try:
        plan = json.loads((run_dir / "plan.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError("SM121 cache-performance plan is unavailable") from error
    if type(plan) is not dict or plan.get("schema_version") != 2:
        raise PreflightError("SM121 cache-performance plan schema is invalid")
    model_data = plan.get("model")
    suite_data = plan.get("suite")
    resolved = plan.get("resolved")
    if type(model_data) is not dict or type(suite_data) is not dict or type(resolved) is not dict:
        raise PreflightError("SM121 cache-performance plan core fields are invalid")
    integrity = plan.get("integrity_hash")
    if not isinstance(integrity, str) or content_hash(
        {key: value for key, value in plan.items() if key != "integrity_hash"},
        len(integrity),
    ) != integrity:
        raise PreflightError("SM121 cache-performance plan integrity is invalid")
    cases = suite_data.get("cases")
    if not isinstance(cases, list) or any(type(case) is not dict for case in cases):
        raise PreflightError("SM121 cache-performance plan cases are invalid")
    suite_without_case_ids = {
        **suite_data,
        "cases": [
            {key: value for key, value in case.items() if key != "case_id"}
            for case in cases
        ],
    }
    expected_fingerprint = content_hash(
        {"model": model_data, "suite": suite_without_case_ids, "resolved": resolved}
    )
    if plan.get("fingerprint") != expected_fingerprint:
        raise PreflightError("SM121 cache-performance plan fingerprint is invalid")
    protocol_digest = suite_data.get("protocol_digest")
    for case in cases:
        case_without_id = {key: value for key, value in case.items() if key != "case_id"}
        expected_case_id = _canonical_case(
            model_data, case_without_id, protocol_digest=protocol_digest
        )["case_id"]
        if case.get("case_id") != expected_case_id:
            raise PreflightError("SM121 cache-performance case identity is invalid")
    model = _namespace(model_data)
    suite = _namespace(suite_data)
    try:
        if not is_sm121_cache_performance_plan(model, suite):
            raise SM121CachePerformanceError("cache-performance selector is invalid")
        validate_sm121_cache_performance_candidate(model)
        validate_sm121_cache_performance_suite(suite)
    except SM121CachePerformanceError as error:
        raise PreflightError("SM121 cache-performance plan contract is invalid") from error
    local_image = resolved.get("local_image")
    if (
        type(local_image) is not dict
        or set(local_image) != {"docker_image_id", "platform", "source_tree"}
        or local_image.get("docker_image_id") != SM121_STORAGE_LOCAL_IMAGE_ID
        or local_image.get("platform") != SM121_STORAGE_PLATFORM
        or local_image.get("source_tree") != SM121_STORAGE_SOURCE_TREE
    ):
        raise PreflightError("SM121 cache-performance local image changed")
    binding = plan.get("cache_performance_pair")
    try:
        validate_sm121_cache_performance_pair_binding(binding)
    except SM121CachePerformanceError as error:
        raise PreflightError("SM121 cache-performance pair binding is invalid") from error
    ordinal = plan.get("cache_performance_ordinal")
    if type(ordinal) is not int or not 1 <= ordinal <= len(SM121_CACHE_PERFORMANCE_ARM_ORDER):
        raise PreflightError("SM121 cache-performance plan ordinal is invalid")
    if sm121_cache_performance_arm(model) != SM121_CACHE_PERFORMANCE_ARM_ORDER[ordinal - 1]:
        raise PreflightError("SM121 cache-performance plan arm is invalid")
    fingerprints = binding.get("plan_fingerprints") if isinstance(binding, dict) else None
    if not isinstance(fingerprints, list) or fingerprints[ordinal - 1] != plan.get("fingerprint"):
        raise PreflightError("SM121 cache-performance plan binding moved")
    run_nonce = plan.get("run_nonce")
    if not isinstance(run_nonce, str) or re.fullmatch(r"[0-9a-f]{32}", run_nonce) is None:
        raise PreflightError("SM121 cache-performance run nonce is invalid")
    model.resolved_local_image_id = local_image["docker_image_id"]
    model.run_identity = f"{plan['fingerprint']}-{run_nonce}"
    model.cache_performance_authorized = True
    model.cache_performance_pair = binding
    return plan, model, suite


def _sm121_cache_performance_case_ids(
    suite: SimpleNamespace,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Return the exact quality and timed cases from one frozen arm plan."""

    cases = list(getattr(suite, "cases", ()))
    if len(cases) != 2:
        raise PreflightError("SM121 cache-performance cases are invalid")
    quality, timed = cases
    if (
        str(getattr(quality, "id", ""))
        != SM121_CACHE_PERFORMANCE_QUALITY_CASE_ID
        or str(getattr(timed, "id", "")) != SM121_CACHE_PERFORMANCE_CASE_ID
        or not isinstance(getattr(quality, "case_id", None), str)
        or not isinstance(getattr(timed, "case_id", None), str)
    ):
        raise PreflightError("SM121 cache-performance cases are invalid")
    return quality, timed


def _sm121_cache_performance_static_event(
    *, model: SimpleNamespace, arm: str, lifetime_ordinal: int
) -> dict[str, Any]:
    """Attest a fresh lifetime to reviewed source semantics without source text."""

    event = {
        "event": SM121_CACHE_PERFORMANCE_STATIC_EVENT,
        "arm": arm,
        "lifetime_ordinal": lifetime_ordinal,
        "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
        **inspect_sm121_cache_source_digests(model),
        **SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
    }
    try:
        validate_sm121_cache_performance_static_event(event)
    except SM121CachePerformanceError as error:
        raise SM121CachePerformanceRequestError() from error
    return event


def _sm121_cache_performance_runtime_event(
    *, server: Any, arm: str, lifetime_ordinal: int
) -> dict[str, Any]:
    """Record only the resolved cache identity for one fresh lifetime."""

    event = {
        "event": SM121_CACHE_PERFORMANCE_RUNTIME_EVENT,
        "arm": arm,
        "lifetime_ordinal": lifetime_ordinal,
        **inspect_sm121_cache_runtime_identity(server),
    }
    try:
        validate_sm121_cache_performance_runtime_event(event)
    except SM121CachePerformanceError as error:
        raise SM121CachePerformanceRequestError() from error
    return event


def _sm121_cache_performance_remaining_s(deadline: float) -> float:
    """Return bounded remaining lifetime budget or fail before another action."""

    remaining_s = deadline - time.monotonic()
    if not math.isfinite(remaining_s) or remaining_s <= 0:
        raise SM121CachePerformanceRequestError()
    return remaining_s


def _sm121_cache_performance_abort_check(
    *, watchdog: HostSafetyWatchdog | None, deadline: float
) -> None:
    if watchdog is not None:
        watchdog.raise_if_tripped()
    _sm121_cache_performance_remaining_s(deadline)


def _sm121_cache_performance_interrupt_terminal_server(
    *, server: Any, deadline: float, terminal_error: BaseException | None
) -> None:
    """Stop owned GPU work before slow diagnostic cleanup after a terminal cell.

    The 1,200-second value is an admission deadline: any work that reaches it
    is rejected even if log collection or container removal takes longer. On
    an observed expiry (or another terminal error), stop the exact owned
    SGLang container before collecting logs so failed work cannot continue on
    the GPU during diagnostic cleanup. ``ManagedServer.interrupt_owned`` is
    idempotent and ``stop`` below still removes the owned container.
    """

    if terminal_error is not None or time.monotonic() >= deadline:
        server.interrupt_owned()


def _sm121_cache_performance_turn_event(
    *,
    case: SimpleNamespace,
    arm: str,
    lifetime_ordinal: int,
    turn: str,
    result: dict[str, Any],
    request_wall_s: float,
    before: dict[str, Any],
    before_polls: int,
    before_settled: bool,
    after: dict[str, Any],
    after_polls: int,
    after_settled: bool,
    append_only_prompt_identity_verified: bool,
    cross_lifetime_prompt_identity_verified: bool,
    shared_prefix_tokens: int,
) -> dict[str, Any]:
    """Project one timed request into scalar-only validated evidence."""

    event: dict[str, Any] = {
        "event": SM121_CACHE_PERFORMANCE_TURN_EVENT,
        "arm": arm,
        "lifetime_ordinal": lifetime_ordinal,
        "case_id": case.case_id,
        "protocol_case_id": SM121_CACHE_PERFORMANCE_CASE_ID,
        "turn": turn,
        "cache_details_requested": True,
        "prompt_token_ids_requested": True,
        "streaming": False,
        "thinking_disabled": True,
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "reasoning_tokens": result["reasoning_tokens"],
        "shared_prefix_tokens": shared_prefix_tokens,
        "append_only_prompt_identity_verified": append_only_prompt_identity_verified,
        "cross_lifetime_prompt_identity_verified": (
            cross_lifetime_prompt_identity_verified
        ),
        "response_detail_state": result["response_detail_state"],
        "usage_detail_state": result["usage_detail_state"],
        "response_device_cached_tokens": result["response_device_cached_tokens"],
        "response_host_cached_tokens": result["response_host_cached_tokens"],
        "response_storage_cached_tokens": result["response_storage_cached_tokens"],
        "usage_cached_tokens": result["usage_cached_tokens"],
        "metrics_available": bool(
            before.get("available") is True and after.get("available") is True
        ),
        "guardrail_metrics_available": bool(
            before.get("guardrail_metrics_available") is True
            and after.get("guardrail_metrics_available") is True
        ),
        "metrics_before_polls": before_polls,
        "metrics_after_polls": after_polls,
        "metrics_before_settled": before_settled,
        "metrics_after_settled": after_settled,
        "request_wall_s": request_wall_s,
        # ``derive_sm121_cache_performance_turn_admission`` authenticates the
        # exact public field set.  Seed its two derived fields before asking it
        # for their final values; these placeholders are overwritten below and
        # can never reach a journal or evidence record.
        "timed_turn_admitted": False,
        "timed_turn_basis": "pending",
    }
    for prefix, snapshot in (("before", before), ("after", after)):
        for metric in SM121_CACHE_PERFORMANCE_METRIC_FIELDS:
            event[f"{prefix}_{metric}"] = snapshot[metric]
        for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
            event[f"{prefix}_cached_{source}_series_present"] = snapshot[
                f"cached_{source}_series_present"
            ]
    for metric in SM121_CACHE_PERFORMANCE_METRIC_FIELDS:
        event[f"delta_{metric}"] = event[f"after_{metric}"] - event[
            f"before_{metric}"
        ]
    try:
        admitted, basis = derive_sm121_cache_performance_turn_admission(event)
    except (KeyError, TypeError, ValueError, SM121CachePerformanceError) as error:
        raise SM121CachePerformanceRequestError() from error
    event["timed_turn_admitted"] = admitted
    event["timed_turn_basis"] = basis
    try:
        validate_sm121_cache_performance_turn_event(event)
    except SM121CachePerformanceError as error:
        raise SM121CachePerformanceRequestError() from error
    return event


def _sm121_cache_performance_timed_turn_prefix_from_journal(
    *, journal: Journal, arm: str, campaign_ordinal: int
) -> list[dict[str, Any]]:
    """Recover only a validated scalar timed prefix after a terminal failure.

    The timed worker journals a turn before it can observe a failed admission
    or teardown.  Its exception path cannot return private prompt IDs, so the
    outer campaign arm rebuilds the public summary prefix from those already
    durable scalar events.  Any malformed or out-of-order record is omitted;
    the raw journal will then fail the later source audit rather than leaking
    an unchecked field into a partial summary.
    """

    events = [
        event
        for event in journal.events()
        if event.get("event") == SM121_CACHE_PERFORMANCE_TURN_EVENT
    ]
    if len(events) > len(SM121_CACHE_PERFORMANCE_TIMED_TURNS):
        return []
    prefix: list[dict[str, Any]] = []
    for index, (expected_turn, raw) in enumerate(
        zip(SM121_CACHE_PERFORMANCE_TIMED_TURNS, events, strict=False)
    ):
        scalar = {key: value for key, value in raw.items() if key != "timestamp"}
        try:
            validate_sm121_cache_performance_turn_event(scalar)
        except SM121CachePerformanceError:
            return []
        if (
            scalar["arm"] != arm
            or scalar["lifetime_ordinal"] != campaign_ordinal * 2
            or scalar["turn"] != expected_turn
            or (
                index + 1 < len(events)
                and scalar["timed_turn_admitted"] is not True
            )
        ):
            return []
        prefix.append(scalar)
    return prefix


def _execute_sm121_cache_performance_quality_case(
    *,
    server: Any,
    model: SimpleNamespace,
    case: SimpleNamespace,
    journal: Journal,
    arm: str,
    lifetime_ordinal: int,
    watchdog: HostSafetyWatchdog | None,
    deadline: float,
) -> None:
    """Run the exact quality gate without retaining responses or request IDs."""

    if len(_QUALITY_ITEMS) != SM121_CACHE_PERFORMANCE_QUALITY_ITEM_COUNT:
        raise PreflightError("SM121 cache-performance quality item count changed")
    journal.append(
        {
            "event": "sm121_cache_performance_quality_case_start",
            "arm": arm,
            "lifetime_ordinal": lifetime_ordinal,
            "case_id": case.case_id,
        }
    )
    try:
        for item in _QUALITY_ITEMS:
            _sm121_cache_performance_abort_check(watchdog=watchdog, deadline=deadline)
            request = _quality_request_arguments(
                server=server,
                model=model,
                case=case,
                item=item,
                request_id=uuid.uuid4().hex,
                # Preserve the exact v2 prompt variant already admitted by
                # the paired semantic canary. The item ID still distinguishes
                # all four quality questions; this tag is not a benchmark arm
                # or timing variable.
                prompt_tag="r0",
            )
            request["timeout_s"] = min(
                900.0, _sm121_cache_performance_remaining_s(deadline)
            )
            result = stream_chat_request(**request)
            validation = _validate_quality_item(item, result)
            if validation.get("passed") is not True:
                raise SM121StorageQualityGateError()
            _sm121_cache_performance_abort_check(watchdog=watchdog, deadline=deadline)
    except (HostSafetyError, SM121StorageQualityGateError, SM121CachePerformanceRequestError):
        raise
    except BaseException:
        raise SM121CachePerformanceRequestError() from None
    journal.append(
        {
            "event": "sm121_cache_performance_quality_case_complete",
            "arm": arm,
            "lifetime_ordinal": lifetime_ordinal,
            "case_id": case.case_id,
            "quality_admitted": True,
            "item_count": len(_QUALITY_ITEMS),
        }
    )


def _execute_sm121_cache_performance_quality_lifetime(
    *,
    run_dir: Path,
    workspace: Path,
    model: SimpleNamespace,
    arm: str,
    lifetime_ordinal: int,
    case: SimpleNamespace,
    journal: Journal,
    telemetry: TelemetrySampler,
) -> float:
    """Run the isolated quality gate and return its start-to-teardown wall time."""

    started = time.monotonic()
    deadline = started + SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S
    server = None
    watchdog: HostSafetyWatchdog | None = None
    terminal_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        _sm121_cache_performance_remaining_s(deadline)
        journal.append(
            _sm121_cache_performance_static_event(
                model=model, arm=arm, lifetime_ordinal=lifetime_ordinal
            )
        )
        watchdog = _host_safety_watchdog(model)
        if watchdog is not None:
            watchdog.start()
        telemetry.set_phase(f"cache_performance_quality_start:{lifetime_ordinal}")
        callbacks: dict[str, Any] = {
            "abort_check": lambda: _sm121_cache_performance_abort_check(
                watchdog=watchdog, deadline=deadline
            )
        }
        if watchdog is not None:
            callbacks["on_server_created"] = (
                lambda created_server: watchdog.register_abort_callback(
                    created_server.interrupt_owned
                )
            )
        server = start_server(
            model,
            workspace=workspace,
            allow_download=False,
            server_log_path=(
                run_dir
                / "server"
                / f"lifetime-{lifetime_ordinal}"
                / "server.log"
            ),
            **callbacks,
        )
        _sm121_cache_performance_abort_check(watchdog=watchdog, deadline=deadline)
        journal.append(
            _sm121_cache_performance_runtime_event(
                server=server, arm=arm, lifetime_ordinal=lifetime_ordinal
            )
        )
        journal.append(
            {
                "event": "server_ready",
                "backend": server.backend,
                "lifetime_ordinal": lifetime_ordinal,
                "phase": "quality",
                "first_inference_is_case": True,
                "case_id": case.case_id,
            }
        )
        telemetry.set_phase(f"cache_performance_quality_case:{lifetime_ordinal}")
        _execute_sm121_cache_performance_quality_case(
            server=server,
            model=model,
            case=case,
            journal=journal,
            arm=arm,
            lifetime_ordinal=lifetime_ordinal,
            watchdog=watchdog,
            deadline=deadline,
        )
        _sm121_cache_performance_abort_check(watchdog=watchdog, deadline=deadline)
    except BaseException as error:
        terminal_error = (
            watchdog.failure if watchdog is not None and watchdog.failure else error
        )
    finally:
        telemetry.set_phase(f"cache_performance_quality_stop:{lifetime_ordinal}")
        if server is not None:
            if watchdog is not None and watchdog.tripped:
                try:
                    _retry_host_safety_interrupt_if_needed(server, watchdog)
                    _record_host_safety_interrupt_failure(
                        journal, watchdog, stage="cache_performance_quality"
                    )
                except BaseException as error:
                    cleanup_error = error
            try:
                _sm121_cache_performance_interrupt_terminal_server(
                    server=server,
                    deadline=deadline,
                    terminal_error=terminal_error,
                )
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            try:
                save_server_logs(
                    server,
                    run_dir
                    / "server"
                    / f"lifetime-{lifetime_ordinal}"
                    / "server.log",
                )
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            try:
                server.stop()
                journal.append(
                    {
                        "event": "server_stopped",
                        "backend": server.backend,
                        "lifetime_ordinal": lifetime_ordinal,
                    }
                )
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                try:
                    server.interrupt_owned()
                    server.stop()
                    journal.append(
                        {
                            "event": "server_stopped",
                            "backend": server.backend,
                            "lifetime_ordinal": lifetime_ordinal,
                        }
                    )
                except BaseException:
                    pass
        if watchdog is not None:
            watchdog.stop()
            if terminal_error is None:
                try:
                    watchdog.raise_if_tripped()
                except BaseException as error:
                    terminal_error = error
        if cleanup_error is not None and terminal_error is None:
            terminal_error = cleanup_error
    elapsed_s = time.monotonic() - started
    within_timeout = elapsed_s <= SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S
    journal.append(
        {
            "event": "sm121_cache_performance_lifetime_complete",
            "arm": arm,
            "lifetime_ordinal": lifetime_ordinal,
            "phase": "quality",
            "lifetime_wall_s": elapsed_s,
            "within_timeout": within_timeout,
            "admitted": terminal_error is None and within_timeout,
        }
    )
    if terminal_error is not None:
        if isinstance(
            terminal_error,
            (
                HostSafetyError,
                SM121StorageQualityGateError,
                SM121CachePerformanceRequestError,
            ),
        ):
            raise terminal_error
        raise SM121CachePerformanceRequestError() from None
    if not within_timeout:
        raise SM121CachePerformanceRequestError()
    return elapsed_s


def _execute_sm121_cache_performance_timed_lifetime(
    *,
    run_dir: Path,
    workspace: Path,
    model: SimpleNamespace,
    arm: str,
    lifetime_ordinal: int,
    case: SimpleNamespace,
    journal: Journal,
    telemetry: TelemetrySampler,
    reference_prompt_token_ids: tuple[tuple[int, ...], ...] | None,
) -> tuple[tuple[dict[str, Any], ...], tuple[tuple[int, ...], ...], float]:
    """Run cold T0 plus append-only T1/T2 in one fresh timed lifetime.

    Token IDs are retained only until the next A/B/B/A lifetime is checked;
    journal records contain only the resulting boolean identity attestations.
    """

    started = time.monotonic()
    deadline = started + SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S
    server = None
    watchdog: HostSafetyWatchdog | None = None
    terminal_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    private_turn_ids: list[tuple[int, ...]] = []
    turn_events: list[dict[str, Any]] = []
    try:
        _sm121_cache_performance_remaining_s(deadline)
        journal.append(
            _sm121_cache_performance_static_event(
                model=model, arm=arm, lifetime_ordinal=lifetime_ordinal
            )
        )
        watchdog = _host_safety_watchdog(model)
        if watchdog is not None:
            watchdog.start()
        telemetry.set_phase(f"cache_performance_timed_start:{lifetime_ordinal}")
        callbacks: dict[str, Any] = {
            "abort_check": lambda: _sm121_cache_performance_abort_check(
                watchdog=watchdog, deadline=deadline
            )
        }
        if watchdog is not None:
            callbacks["on_server_created"] = (
                lambda created_server: watchdog.register_abort_callback(
                    created_server.interrupt_owned
                )
            )
        server = start_server(
            model,
            workspace=workspace,
            allow_download=False,
            server_log_path=(
                run_dir
                / "server"
                / f"lifetime-{lifetime_ordinal}"
                / "server.log"
            ),
            **callbacks,
        )
        _sm121_cache_performance_abort_check(watchdog=watchdog, deadline=deadline)
        journal.append(
            _sm121_cache_performance_runtime_event(
                server=server, arm=arm, lifetime_ordinal=lifetime_ordinal
            )
        )
        journal.append(
            {
                "event": "server_ready",
                "backend": server.backend,
                "lifetime_ordinal": lifetime_ordinal,
                "phase": "timed",
                "first_inference_is_case": True,
                "case_id": case.case_id,
            }
        )
        journal.append(
            {
                "event": "sm121_cache_performance_timed_case_start",
                "arm": arm,
                "lifetime_ordinal": lifetime_ordinal,
                "case_id": case.case_id,
            }
        )
        telemetry.set_phase(f"cache_performance_timed_case:{lifetime_ordinal}")
        for index, (turn, messages, expected_response) in enumerate(
            zip(
                SM121_CACHE_PERFORMANCE_TIMED_TURNS,
                _sm121_cache_semantic_messages(),
                _SM121_CACHE_SEMANTIC_EXPECTED_RESPONSES,
                strict=True,
            )
        ):
            _sm121_cache_performance_abort_check(watchdog=watchdog, deadline=deadline)
            before_timeout_s = min(
                45.0, _sm121_cache_performance_remaining_s(deadline)
            )
            before, _ignored_before_wait_s, before_polls, before_settled = (
                settle_sm121_cache_observability_metrics(
                    server,
                    timeout_s=before_timeout_s,
                    poll_interval_s=min(1.0, max(0.001, before_timeout_s / 4)),
                    semantic_arm=arm,
                )
            )
            _sm121_cache_performance_abort_check(watchdog=watchdog, deadline=deadline)
            request_started = time.perf_counter()
            result = request_sm121_cache_semantic_turn(
                server,
                served_name=model.served_name,
                messages=messages,
                expected_response=expected_response,
                max_tokens=int(case.max_output_tokens),
                timeout_s=min(
                    900.0, _sm121_cache_performance_remaining_s(deadline)
                ),
            )
            request_wall_s = time.perf_counter() - request_started
            _sm121_cache_performance_abort_check(watchdog=watchdog, deadline=deadline)
            after_timeout_s = min(
                45.0, _sm121_cache_performance_remaining_s(deadline)
            )
            after, _ignored_after_wait_s, after_polls, after_settled = (
                settle_sm121_cache_observability_metrics(
                    server,
                    timeout_s=after_timeout_s,
                    poll_interval_s=min(1.0, max(0.001, after_timeout_s / 4)),
                    semantic_arm=arm,
                )
            )
            _sm121_cache_performance_abort_check(watchdog=watchdog, deadline=deadline)
            prompt_token_ids = result.pop("private_prompt_token_ids", None)
            if (
                not isinstance(prompt_token_ids, tuple)
                or not prompt_token_ids
                or any(type(token) is not int or token < 0 for token in prompt_token_ids)
            ):
                raise SM121CachePerformanceRequestError()
            if index == 0:
                shared_prefix_tokens = 0
                append_verified = True
            else:
                shared_prefix_tokens = _sm121_cache_semantic_common_prefix_tokens(
                    private_turn_ids[-1], prompt_token_ids
                )
                append_verified = (
                    shared_prefix_tokens
                    >= SM121_CACHE_PERFORMANCE_COLD_INPUT_MIN_TOKENS
                )
            if reference_prompt_token_ids is None:
                if lifetime_ordinal != 2 or arm != "A":
                    raise SM121CachePerformanceRequestError()
                cross_lifetime_verified = True
            else:
                if prompt_token_ids != reference_prompt_token_ids[index]:
                    raise SM121CachePerformanceRequestError()
                cross_lifetime_verified = True
            event = _sm121_cache_performance_turn_event(
                case=case,
                arm=arm,
                lifetime_ordinal=lifetime_ordinal,
                turn=turn,
                result=result,
                request_wall_s=request_wall_s,
                before=before,
                before_polls=before_polls,
                before_settled=before_settled,
                after=after,
                after_polls=after_polls,
                after_settled=after_settled,
                append_only_prompt_identity_verified=append_verified,
                cross_lifetime_prompt_identity_verified=cross_lifetime_verified,
                shared_prefix_tokens=shared_prefix_tokens,
            )
            journal.append(event)
            if event["timed_turn_admitted"] is not True:
                raise SM121CachePerformanceRequestError()
            turn_events.append(event)
            private_turn_ids.append(prompt_token_ids)
        journal.append(
            {
                "event": "sm121_cache_performance_timed_case_complete",
                "arm": arm,
                "lifetime_ordinal": lifetime_ordinal,
                "case_id": case.case_id,
                "timed_admitted": True,
            }
        )
        _sm121_cache_performance_abort_check(watchdog=watchdog, deadline=deadline)
    except BaseException as error:
        terminal_error = (
            watchdog.failure if watchdog is not None and watchdog.failure else error
        )
    finally:
        telemetry.set_phase(f"cache_performance_timed_stop:{lifetime_ordinal}")
        if server is not None:
            if watchdog is not None and watchdog.tripped:
                try:
                    _retry_host_safety_interrupt_if_needed(server, watchdog)
                    _record_host_safety_interrupt_failure(
                        journal, watchdog, stage="cache_performance_timed"
                    )
                except BaseException as error:
                    cleanup_error = error
            try:
                _sm121_cache_performance_interrupt_terminal_server(
                    server=server,
                    deadline=deadline,
                    terminal_error=terminal_error,
                )
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            try:
                save_server_logs(
                    server,
                    run_dir
                    / "server"
                    / f"lifetime-{lifetime_ordinal}"
                    / "server.log",
                )
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            try:
                server.stop()
                journal.append(
                    {
                        "event": "server_stopped",
                        "backend": server.backend,
                        "lifetime_ordinal": lifetime_ordinal,
                    }
                )
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                try:
                    server.interrupt_owned()
                    server.stop()
                    journal.append(
                        {
                            "event": "server_stopped",
                            "backend": server.backend,
                            "lifetime_ordinal": lifetime_ordinal,
                        }
                    )
                except BaseException:
                    pass
        if watchdog is not None:
            watchdog.stop()
            if terminal_error is None:
                try:
                    watchdog.raise_if_tripped()
                except BaseException as error:
                    terminal_error = error
        if cleanup_error is not None and terminal_error is None:
            terminal_error = cleanup_error
    elapsed_s = time.monotonic() - started
    within_timeout = elapsed_s <= SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S
    journal.append(
        {
            "event": "sm121_cache_performance_lifetime_complete",
            "arm": arm,
            "lifetime_ordinal": lifetime_ordinal,
            "phase": "timed",
            "lifetime_wall_s": elapsed_s,
            "within_timeout": within_timeout,
            "admitted": terminal_error is None and within_timeout,
        }
    )
    if terminal_error is not None:
        if isinstance(terminal_error, (HostSafetyError, SM121CachePerformanceRequestError)):
            raise terminal_error
        raise SM121CachePerformanceRequestError() from None
    if not within_timeout:
        raise SM121CachePerformanceRequestError()
    return tuple(turn_events), tuple(private_turn_ids), elapsed_s


def _load_sm121_cache_performance_campaign(
    campaign_dir: Path, *, evidence_root: Path
) -> tuple[
    dict[str, Any],
    list[tuple[Path, dict[str, Any], SimpleNamespace, SimpleNamespace]],
]:
    """Authenticate a frozen A/B/B/A topology before any server can start."""

    try:
        root = campaign_dir.resolve(strict=True)
        campaign = json.loads((root / "campaign.json").read_text())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise PreflightError("SM121 cache-performance campaign is unavailable") from error
    expected_fields = {
        "schema_version",
        "campaign_id",
        "created_at",
        "execution_mode",
        "prerequisite_bundle_sha256s",
        "pair_binding",
        "run_directories",
        "integrity_hash",
    }
    if type(campaign) is not dict or set(campaign) != expected_fields:
        raise PreflightError("SM121 cache-performance campaign fields are invalid")
    integrity = campaign.get("integrity_hash")
    if not isinstance(integrity, str) or content_hash(
        {key: value for key, value in campaign.items() if key != "integrity_hash"},
        len(integrity),
    ) != integrity:
        raise PreflightError("SM121 cache-performance campaign integrity is invalid")
    if (
        campaign.get("schema_version") != 1
        or campaign.get("campaign_id") != SM121_CACHE_PERFORMANCE_CAMPAIGN_ID
        or campaign.get("execution_mode") != SM121_CACHE_PERFORMANCE_EXECUTION_MODE
        or not isinstance(campaign.get("created_at"), str)
        or campaign.get("prerequisite_bundle_sha256s")
        != list(SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S)
    ):
        raise PreflightError("SM121 cache-performance campaign contract is invalid")
    binding = campaign.get("pair_binding")
    try:
        validate_sm121_cache_performance_pair_binding(binding)
        _validate_sm121_cache_performance_prerequisites(evidence_root)
    except SM121CachePerformanceError as error:
        raise PreflightError("SM121 cache-performance admission is unavailable") from error
    if not isinstance(binding, dict):
        raise PreflightError("SM121 cache-performance pair binding is invalid")
    names = campaign.get("run_directories")
    if (
        type(names) is not list
        or len(names) != len(SM121_CACHE_PERFORMANCE_ARM_ORDER)
        or len(set(names)) != len(names)
        or any(
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", name) is None
            for name in names
        )
    ):
        raise PreflightError("SM121 cache-performance run topology is invalid")
    runs_root = (root / "runs").resolve(strict=True)
    try:
        runs_root.relative_to(root)
    except ValueError as error:
        raise PreflightError("SM121 cache-performance runs escape campaign") from error
    loaded: list[tuple[Path, dict[str, Any], SimpleNamespace, SimpleNamespace]] = []
    for ordinal, (name, expected_arm) in enumerate(
        zip(names, SM121_CACHE_PERFORMANCE_ARM_ORDER, strict=True), start=1
    ):
        run_dir = runs_root / str(name)
        try:
            resolved_run_dir = run_dir.resolve(strict=True)
            resolved_run_dir.relative_to(runs_root)
        except (OSError, ValueError) as error:
            raise PreflightError("SM121 cache-performance run directory is invalid") from error
        if resolved_run_dir.parent != runs_root or run_dir.is_symlink():
            raise PreflightError("SM121 cache-performance run directory is invalid")
        plan, model, suite = _load_sm121_cache_performance_plan(resolved_run_dir)
        if (
            plan.get("cache_performance_ordinal") != ordinal
            or sm121_cache_performance_arm(model) != expected_arm
            or plan.get("cache_performance_pair") != binding
        ):
            raise PreflightError("SM121 cache-performance run binding moved")
        loaded.append((resolved_run_dir, plan, model, suite))
    fingerprints = [plan["fingerprint"] for _path, plan, _model, _suite in loaded]
    nonces = [plan["run_nonce"] for _path, plan, _model, _suite in loaded]
    try:
        instance = sm121_cache_performance_pair_instance_sha256(nonces)
    except SM121CachePerformanceError as error:
        raise PreflightError("SM121 cache-performance run nonce is invalid") from error
    if (
        binding.get("campaign_instance_sha256") != instance
        or binding.get("plan_fingerprints") != fingerprints
        or binding.get("pair_binding_sha256")
        != sm121_cache_performance_pair_binding_sha256(binding)
    ):
        raise PreflightError("SM121 cache-performance binding is invalid")
    return campaign, loaded


def _execute_sm121_cache_performance_arm(
    *,
    run_dir: Path,
    plan: dict[str, Any],
    model: SimpleNamespace,
    suite: SimpleNamespace,
    campaign_ordinal: int,
    workspace: Path,
    reference_prompt_token_ids: tuple[tuple[int, ...], ...] | None,
) -> tuple[dict[str, Any], tuple[tuple[int, ...], ...] | None, bool]:
    """Execute one frozen A/B/B/A arm as two clean server lifetimes."""

    journal = Journal(run_dir / "events.jsonl")
    if journal.events():
        raise PreflightError(
            "SM121 cache-performance campaign is non-resumable; freeze a new campaign"
        )
    arm = sm121_cache_performance_arm(model)
    quality_case, timed_case = _sm121_cache_performance_case_ids(suite)
    if set(quality_case.requires) - set(model.tasks) or set(timed_case.requires) - set(
        model.tasks
    ):
        raise PreflightError("SM121 cache-performance case capabilities are invalid")
    if (
        SM121_CACHE_PERFORMANCE_COLD_INPUT_MAX_TOKENS
        + int(timed_case.max_output_tokens)
        + 1_024
        > int(model.max_context)
    ):
        raise PreflightError("SM121 cache-performance context admission is insufficient")
    binding = getattr(model, "cache_performance_pair", None)
    if not isinstance(binding, dict):
        raise PreflightError("SM121 cache-performance pair binding is unavailable")
    journal.append(
        {
            "event": "run_start",
            "execution_mode": SM121_CACHE_PERFORMANCE_EXECUTION_MODE,
            "arm": arm,
            "campaign_ordinal": campaign_ordinal,
            "plan_fingerprint": str(plan["fingerprint"]),
            "cache_performance_pair_binding_sha256": binding[
                "pair_binding_sha256"
            ],
        }
    )
    journal.append({"event": "measurement_started"})
    telemetry = TelemetrySampler(run_dir / "telemetry.jsonl")
    quality_admitted = False
    timed_admitted = False
    within_timeout = False
    turns: list[dict[str, Any]] = []
    next_reference = reference_prompt_token_ids
    stage = "preflight"
    try:
        _preflight(model)
        stage = "quality_lifetime"
        quality_elapsed_s = _execute_sm121_cache_performance_quality_lifetime(
            run_dir=run_dir,
            workspace=workspace,
            model=model,
            arm=arm,
            lifetime_ordinal=campaign_ordinal * 2 - 1,
            case=quality_case,
            journal=journal,
            telemetry=telemetry,
        )
        quality_admitted = True
        stage = "timed_lifetime"
        turns_tuple, private_ids, timed_elapsed_s = (
            _execute_sm121_cache_performance_timed_lifetime(
                run_dir=run_dir,
                workspace=workspace,
                model=model,
                arm=arm,
                lifetime_ordinal=campaign_ordinal * 2,
                case=timed_case,
                journal=journal,
                telemetry=telemetry,
                reference_prompt_token_ids=reference_prompt_token_ids,
            )
        )
        turns = list(turns_tuple)
        timed_admitted = all(
            event["timed_turn_admitted"] is True for event in turns
        )
        if not timed_admitted or len(private_ids) != len(SM121_CACHE_PERFORMANCE_TIMED_TURNS):
            raise SM121CachePerformanceRequestError()
        if reference_prompt_token_ids is None:
            next_reference = private_ids
        within_timeout = (
            quality_elapsed_s <= SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S
            and timed_elapsed_s <= SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S
        )
        if not within_timeout:
            raise SM121CachePerformanceRequestError()
    except BaseException as error:
        if isinstance(
            error,
            (
                HostSafetyError,
                PreflightError,
                SM121StorageQualityGateError,
                SM121CachePerformanceRequestError,
            ),
        ):
            safe_error = error
        else:
            safe_error = SM121CachePerformanceRequestError()
        if isinstance(safe_error, HostSafetyError):
            _record_host_safety_breach(journal, safe_error, stage=stage)
        if stage == "timed_lifetime":
            turns = _sm121_cache_performance_timed_turn_prefix_from_journal(
                journal=journal,
                arm=arm,
                campaign_ordinal=campaign_ordinal,
            )
        _record_run_aborted(journal, safe_error, stage=stage)
        return (
            {
                "ordinal": campaign_ordinal,
                "arm": arm,
                "quality_admitted": quality_admitted,
                "timed_admitted": False,
                "within_timeout": within_timeout,
                "turns": turns,
            },
            next_reference,
            False,
        )
    finally:
        # This controller uses phase tracking only; no telemetry file is
        # started because scalar cache events are the sole retained evidence.
        telemetry.stop()
    journal.append({"event": "measurement_complete"})
    journal.append({"event": "run_complete", "status": "completed"})
    return (
        {
            "ordinal": campaign_ordinal,
            "arm": arm,
            "quality_admitted": quality_admitted,
            "timed_admitted": timed_admitted,
            "within_timeout": within_timeout,
            "turns": turns,
        },
        next_reference,
        True,
    )


def _sm121_cache_performance_unstarted_lifetime(
    *, ordinal: int, arm: str
) -> dict[str, Any]:
    """Represent an intentionally unstarted arm after a terminal predecessor."""

    return {
        "ordinal": ordinal,
        "arm": arm,
        "quality_admitted": False,
        "timed_admitted": False,
        "within_timeout": False,
        "turns": [],
    }


def execute_sm121_cache_performance_campaign(
    campaign_dir: Path, *, workspace: Path, evidence_root: Path
) -> dict[str, Any]:
    """Run one non-resumable fresh-lifetime A/B/B/A wall-time campaign."""

    campaign, loaded = _load_sm121_cache_performance_campaign(
        campaign_dir, evidence_root=evidence_root
    )
    if (campaign_dir / "summary.json").exists():
        raise PreflightError(
            "SM121 cache-performance campaign is terminal; freeze a new campaign"
        )
    for run_dir, _plan, _model, _suite in loaded:
        if Journal(run_dir / "events.jsonl").events():
            raise PreflightError(
                "SM121 cache-performance campaign is non-resumable; freeze a new campaign"
            )
    lock_path = results_lock_path(workspace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another SparkBench run holds the benchmark lock") from error
        lifetimes: list[dict[str, Any]] = []
        reference_prompt_token_ids: tuple[tuple[int, ...], ...] | None = None
        terminal = False
        for ordinal, (run_dir, plan, model, suite) in enumerate(loaded, start=1):
            arm = SM121_CACHE_PERFORMANCE_ARM_ORDER[ordinal - 1]
            if terminal:
                lifetimes.append(
                    _sm121_cache_performance_unstarted_lifetime(
                        ordinal=ordinal, arm=arm
                    )
                )
                continue
            lifetime, reference_prompt_token_ids, completed = (
                _execute_sm121_cache_performance_arm(
                    run_dir=run_dir,
                    plan=plan,
                    model=model,
                    suite=suite,
                    campaign_ordinal=ordinal,
                    workspace=workspace,
                    reference_prompt_token_ids=reference_prompt_token_ids,
                )
            )
            lifetimes.append(lifetime)
            terminal = not completed
        try:
            score = score_sm121_cache_performance_campaign(lifetimes)
        except SM121CachePerformanceError as error:
            raise PreflightError("SM121 cache-performance score is invalid") from error
        summary = {
            "schema_version": 1,
            "campaign_id": SM121_CACHE_PERFORMANCE_CAMPAIGN_ID,
            "execution_mode": SM121_CACHE_PERFORMANCE_EXECUTION_MODE,
            "pair_binding_sha256": campaign["pair_binding"]["pair_binding_sha256"],
            "status": score.status,
            "decision": score.decision,
            "completed_arms": sum(
                1
                for lifetime in lifetimes
                if lifetime["quality_admitted"] is True
                and lifetime["timed_admitted"] is True
                and lifetime["within_timeout"] is True
            ),
            "lifetimes": lifetimes,
            "score": score.to_mapping(),
        }
        summary["integrity_hash"] = content_hash(summary, 64)
        write_json(campaign_dir / "summary.json", summary)
        return summary


def _recover_pending_lifecycle(
    *, model: SimpleNamespace, journal: Journal, run_dir: Path, workspace: Path
) -> bool:
    """Resolve an earlier crashed run's lifecycle before resume or finalization."""

    events = journal.events()
    last_start = max(
        (index for index, event in enumerate(events) if event.get("event") == "run_start"),
        default=-1,
    )
    if last_start < 0:
        return False
    run_finished = any(
        index > last_start and event.get("event") == "run_complete"
        for index, event in enumerate(events)
    )
    ready_indexes = [
        index
        for index, event in enumerate(events)
        if index > last_start and event.get("event") == "server_ready"
    ]
    if not ready_indexes:
        backend = str(model.backend)
        if backend == "llamacpp" and not run_finished:
            action = recover_owned_llamacpp(
                model,
                workspace=workspace,
                run_identity=str(model.run_identity),
                process_state_path=run_dir / "server" / "process.json",
            )
            journal.append(
                {
                    "event": "server_stopped",
                    "backend": backend,
                    "recovered": True,
                    "recovery_action": action,
                }
            )
            return True
        if backend in {"sglang", "vllm"} and not run_finished:
            if backend == "sglang":
                action = recover_owned_sglang(
                    str(model.run_identity),
                    api_key_path=run_dir / "server" / "api-key",
                )
            else:
                action = recover_owned_vllm(str(model.run_identity))
            journal.append(
                {
                    "event": "server_stopped",
                    "backend": backend,
                    "recovered": True,
                    "recovery_action": action,
                }
            )
            return True
        return False
    last_ready = ready_indexes[-1]
    if any(
        index > last_ready and event.get("event") in {"server_stopped", "server_kept"}
        for index, event in enumerate(events)
    ):
        return False

    ready_event = events[last_ready]
    backend = str(model.backend)
    if backend == "llamacpp":
        action = recover_owned_llamacpp(
            model,
            workspace=workspace,
            run_identity=str(model.run_identity),
            process_state_path=run_dir / "server" / "process.json",
        )
        journal.append(
            {
                "event": "server_stopped",
                "backend": backend,
                "recovered": True,
                "recovery_action": action,
            }
        )
        return True
    if backend in {"sglang", "vllm"}:
        if backend == "sglang":
            action = recover_owned_sglang(
                str(model.run_identity),
                api_key_path=run_dir / "server" / "api-key",
            )
        else:
            action = recover_owned_vllm(str(model.run_identity))
        journal.append(
            {
                "event": "server_stopped",
                "backend": backend,
                "recovered": True,
                "recovery_action": action,
            }
        )
        return True

    if backend != "ollama":
        return False

    cleanup_failed = any(
        index > last_ready and event.get("event") == "cleanup_failed"
        for index, event in enumerate(events)
    )
    unload_owned = ready_event.get("ollama_unload_owned")
    if run_finished and not cleanup_failed:
        # Older completed journals predate explicit server_stopped events. Do
        # not reinterpret a later user load as benchmark-owned.
        return False
    if unload_owned is False:
        return False

    try:
        loaded = ollama_model_loaded(str(model.endpoint), str(model.source))
    except Exception as error:
        journal.append(
            {
                "event": "recovery_cleanup_pending",
                "backend": backend,
                "model": str(model.source),
                "reason": f"loaded state could not be verified: {error}",
            }
        )
        raise
    if loaded:
        journal.append(
            {
                "event": "recovery_cleanup_pending",
                "backend": backend,
                "model": str(model.source),
                "reason": "model remains loaded and crash-time ownership cannot be proven",
            }
        )
        raise RuntimeErrorWithContext(
            f"Ollama model {model.source} may still be owned by the crashed run; "
            "refusing to finalize or unload it automatically"
        )
    journal.append(
        {
            "event": "server_stopped",
            "backend": backend,
            "recovered": True,
            "recovery_action": "observed_model_absent",
        }
    )
    return True


def _record_run_aborted(
    journal: Journal, error: BaseException, *, stage: str
) -> None:
    """Persist one terminal abort record for the current run attempt."""

    events = journal.events()
    last_start = max(
        (
            index
            for index, event in enumerate(events)
            if event.get("event") == "run_start"
        ),
        default=-1,
    )
    if last_start < 0:
        return
    recorded_aborts = [
        event
        for index, event in enumerate(events)
        if index > last_start and event.get("event") == "run_aborted"
    ]
    if recorded_aborts:
        safety_override = isinstance(error, HostSafetyError) and (
            recorded_aborts[-1].get("error_type") != "HostSafetyError"
        )
        if not safety_override:
            return
    journal.append(
        {
            "event": "run_aborted",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )


def _validate_prefix_cache_plan_selection(model: Any, suite: Any) -> None:
    """Reject mismatched or legacy cache plans before any server is started."""

    raw_suite_id = getattr(suite, "id", None)
    suite_id = raw_suite_id if isinstance(raw_suite_id, str) else ""
    raw_cases = getattr(suite, "cases", ())
    cases = list(raw_cases) if isinstance(raw_cases, (list, tuple)) else []
    cache_cases = [case for case in cases if _is_prefix_cache_case(case)]
    mode = getattr(model, "prefix_cache_mode", None)
    cache_profile = mode in {"off", "on"}
    cache_suite = suite_id == PREFIX_CACHE_SUITE_ID
    if mode is not None and not cache_profile:
        raise PreflightError("Frozen prefix-cache profile mode is invalid")
    if (
        cache_profile != cache_suite
        or bool(cache_cases) != cache_suite
        or (cache_suite and len(cache_cases) != len(cases))
    ):
        raise PreflightError(
            "Frozen prefix-cache model profile and suite do not match"
        )
    if not cache_suite:
        return
    if not isinstance(raw_cases, list):
        raise PreflightError("Frozen prefix-cache suite cases must be a JSON list")
    if (
        getattr(model, "backend", None) != "llamacpp"
        or type(getattr(model, "runtime_parallel", None)) is not int
        or getattr(model, "runtime_parallel", None) != 1
        or type(getattr(model, "max_context", None)) is not int
        or type(getattr(model, "native_context", None)) is not int
        or getattr(model, "max_context", None) != PREFIX_CACHE_CONTEXT_TOKENS
        or getattr(model, "native_context", None) != PREFIX_CACHE_CONTEXT_TOKENS
    ):
        raise PreflightError("Frozen prefix-cache model does not match its protocol")
    expected_case_fields = {
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
    expected_cases = tuple(PREFIX_CACHE_PREFIX_TARGETS.items())
    if (
        len(cases) != len(expected_cases)
        or len(cache_cases) != len(expected_cases)
    ):
        raise PreflightError("Frozen prefix-cache plan does not match its protocol")
    for case, (expected_id, expected_target) in zip(cases, expected_cases, strict=True):
        try:
            fields = vars(case)
        except TypeError as error:
            raise PreflightError(
                "Frozen prefix-cache case is not a typed plan object"
            ) from error
        if set(fields) != expected_case_fields:
            raise PreflightError("Frozen prefix-cache case schema does not match protocol")
        case_id = fields["case_id"]
        if (
            fields["id"] != expected_id
            or fields["kind"] != "cache"
            or type(fields["requires"]) is not list
            or fields["requires"] != ["chat"]
            or type(fields["warmups"]) is not int
            or fields["warmups"] != 0
            or type(fields["repetitions"]) is not int
            or fields["repetitions"] != 5
            or type(fields["max_output_tokens"]) is not int
            or fields["max_output_tokens"] != 128
            or type(fields["max_turns"]) is not int
            or fields["max_turns"] != 1
            or type(fields["temperature"]) is not float
            or fields["temperature"] != 0.0
            or type(fields["concurrency"]) is not int
            or fields["concurrency"] != 1
            or type(fields["prompt_repetitions"]) is not int
            or fields["prompt_repetitions"] != expected_target
            or not isinstance(case_id, str)
            or re.fullmatch(rf"{re.escape(expected_id)}--[0-9a-f]{{12}}", case_id)
            is None
        ):
            raise PreflightError("Frozen prefix-cache case does not match protocol")
        required_context = expected_target * 6 + 128 + 1_024
        if required_context > getattr(model, "max_context") or required_context > getattr(
            model, "native_context"
        ):
            raise PreflightError("Frozen prefix-cache context admission is insufficient")
    raw_arguments = getattr(model, "args", None)
    if not isinstance(raw_arguments, list) or not all(
        isinstance(argument, str) for argument in raw_arguments
    ):
        raise PreflightError("Frozen prefix-cache arguments must be a JSON string list")
    arguments = tuple(raw_arguments)
    if arguments != prefix_cache_llamacpp_args(mode):
        raise PreflightError(
            "Frozen prefix-cache arguments do not match the exact protocol"
        )


def _validate_memory_operation_plan_selection(model: Any, suite: Any) -> None:
    """Reject altered memory-operation plans before any server is started."""

    raw_suite_id = getattr(suite, "id", None)
    suite_id = raw_suite_id if isinstance(raw_suite_id, str) else ""
    raw_cases = getattr(suite, "cases", ())
    cases = list(raw_cases) if isinstance(raw_cases, (list, tuple)) else []
    memory_cases = [case for case in cases if getattr(case, "kind", None) == "memory"]
    memory_suite = suite_id == MEMORY_OPERATION_SUITE_ID
    if bool(memory_cases) != memory_suite or (
        memory_suite and len(memory_cases) != len(cases)
    ):
        raise PreflightError("Frozen memory-operation suite does not match protocol")
    if not memory_suite:
        return
    if not isinstance(raw_cases, list):
        raise PreflightError("Frozen memory-operation cases must be a JSON list")
    try:
        require_memory_operation_protocol_digest(
            getattr(suite, "protocol_digest", None)
        )
    except ValueError as error:
        raise PreflightError(
            "Frozen memory-operation protocol digest is stale or invalid"
        ) from error
    try:
        suite_fields = vars(suite)
    except TypeError as error:
        raise PreflightError(
            "Frozen memory-operation suite is not a typed plan object"
        ) from error
    if (
        set(suite_fields)
        != {
            "id",
            "cases",
            "description",
            "protocol_digest",
            "schema_version",
        }
        or type(suite_fields.get("schema_version")) is not int
        or suite_fields.get("schema_version") != 1
        or suite_fields.get("description") != MEMORY_OPERATION_SUITE_DESCRIPTION
        or suite_fields.get("protocol_digest")
        != MEMORY_OPERATION_PROTOCOL_DIGEST
    ):
        raise PreflightError(
            "Frozen memory-operation suite schema does not match protocol"
        )
    if (
        getattr(model, "backend", None) != "llamacpp"
        or getattr(model, "lifecycle", None) != "subprocess"
        or getattr(model, "runtime_revision", None)
        != MEMORY_OPERATION_LLAMACPP_REVISION
        or getattr(model, "runtime_digest", None)
        != MEMORY_OPERATION_LLAMACPP_DIGEST
        or type(getattr(model, "runtime_parallel", None)) is not int
        or getattr(model, "runtime_parallel", None) != 1
        or type(getattr(model, "max_context", None)) is not int
        or getattr(model, "max_context", None) != MEMORY_OPERATION_CONTEXT_TOKENS
        or type(getattr(model, "native_context", None)) is not int
        or getattr(model, "native_context", None) != MEMORY_OPERATION_CONTEXT_TOKENS
    ):
        raise PreflightError(
            "Frozen memory-operation model does not match fixed llama.cpp geometry"
        )
    raw_tasks = getattr(model, "tasks", None)
    if not isinstance(raw_tasks, list) or not all(
        isinstance(task, str) for task in raw_tasks
    ):
        raise PreflightError("Frozen memory-operation tasks must be a JSON string list")
    raw_request_body = getattr(model, "request_body_json", None)
    if not isinstance(raw_request_body, str):
        raise PreflightError(
            "Frozen memory-operation profile lacks an explicit thinking policy"
        )
    try:
        request_body = json.loads(raw_request_body)
    except json.JSONDecodeError as error:
        raise PreflightError(
            "Frozen memory-operation thinking policy is invalid"
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
        raise PreflightError(
            "Frozen memory-operation thinking policy does not match protocol"
        )
    enable_thinking = request_body["chat_template_kwargs"]["enable_thinking"]
    task_set = set(raw_tasks)
    if (
        len(task_set) != len(raw_tasks)
        or not {"chat", "json"}.issubset(task_set)
        or ("thinking" in task_set) is not enable_thinking
    ):
        raise PreflightError(
            "Frozen memory-operation task capabilities do not match protocol"
        )
    raw_arguments = getattr(model, "args", None)
    if not isinstance(raw_arguments, list) or not all(
        isinstance(argument, str) for argument in raw_arguments
    ):
        raise PreflightError(
            "Frozen memory-operation arguments must be a JSON string list"
        )
    if tuple(raw_arguments) != memory_operation_llamacpp_args(
        enable_thinking=enable_thinking
    ):
        raise PreflightError(
            "Frozen memory-operation arguments do not match protocol"
        )
    expected_case_fields = {
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
    if len(cases) != len(MEMORY_OPERATION_SCENARIO_IDS):
        raise PreflightError("Frozen memory-operation case count is invalid")
    for case, expected_id in zip(
        cases, MEMORY_OPERATION_SCENARIO_IDS, strict=True
    ):
        try:
            fields = vars(case)
        except TypeError as error:
            raise PreflightError(
                "Frozen memory-operation case is not a typed plan object"
            ) from error
        case_id = fields.get("case_id")
        if (
            set(fields) != expected_case_fields
            or fields.get("id") != expected_id
            or fields.get("kind") != "memory"
            or type(fields.get("requires")) is not list
            or fields.get("requires") != ["chat", "json"]
            or type(fields.get("warmups")) is not int
            or fields.get("warmups") != 0
            or type(fields.get("repetitions")) is not int
            or fields.get("repetitions") != MEMORY_OPERATION_VARIANT_COUNT
            or type(fields.get("max_output_tokens")) is not int
            or fields.get("max_output_tokens") != MEMORY_OPERATION_OUTPUT_TOKENS
            or type(fields.get("max_turns")) is not int
            or fields.get("max_turns") != 1
            or type(fields.get("temperature")) is not float
            or fields.get("temperature") != 0.0
            or type(fields.get("concurrency")) is not int
            or fields.get("concurrency") != 1
            or type(fields.get("prompt_repetitions")) is not int
            or fields.get("prompt_repetitions") != 0
            or not isinstance(case_id, str)
            or re.fullmatch(rf"{re.escape(expected_id)}--[0-9a-f]{{12}}", case_id)
            is None
        ):
            raise PreflightError(
                "Frozen memory-operation case does not match protocol"
            )
    required_context = estimate_memory_operation_context_tokens(
        max_output_tokens=MEMORY_OPERATION_OUTPUT_TOKENS
    )
    if required_context > model.max_context or required_context > model.native_context:
        raise PreflightError(
            "Frozen memory-operation context admission is insufficient"
        )


def execute_plan(
    run_dir: Path,
    *,
    workspace: Path,
    allow_download: bool = False,
    keep_server: bool = False,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    plan_path = run_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    suite_without_case_ids = {
        **plan["suite"],
        "cases": [
            {key: value for key, value in case.items() if key != "case_id"}
            for case in plan["suite"]["cases"]
        ],
    }
    schema_version = int(plan.get("schema_version", 1))
    if schema_version not in {1, 2}:
        raise RuntimeError(f"Unsupported frozen plan schema version: {schema_version}")
    if schema_version >= 2:
        integrity_hash = plan.get("integrity_hash")
        integrity_payload = {
            key: value for key, value in plan.items() if key != "integrity_hash"
        }
        expected_fingerprint = content_hash(
            {
                "model": plan["model"],
                "suite": suite_without_case_ids,
                "resolved": plan.get("resolved", {}),
            }
        )
        integrity_length = len(str(integrity_hash)) if integrity_hash else 64
        integrity_valid = bool(integrity_hash) and (
            content_hash(integrity_payload, integrity_length) == integrity_hash
        )
    else:
        expected_fingerprint = content_hash(
            {"model": plan["model"], "suite": suite_without_case_ids}
        )
        expected_digest = plan["model"].get("image_digest")
        resolved_image = plan.get("resolved", {}).get("image_digest")
        integrity_valid = not expected_digest or (
            isinstance(resolved_image, str)
            and resolved_image.endswith("@" + expected_digest)
        )
    if not integrity_valid or expected_fingerprint != plan["fingerprint"]:
        raise RuntimeError("Frozen plan fingerprint does not match its contents")
    for case in plan["suite"]["cases"]:
        case_without_id = {key: value for key, value in case.items() if key != "case_id"}
        expected_case_id = _canonical_case(
            plan["model"],
            case_without_id,
            protocol_digest=plan["suite"].get("protocol_digest"),
        )["case_id"]
        if case.get("case_id") != expected_case_id:
            raise RuntimeError("Frozen plan case identity does not match its contents")
    blocker = model_execution_blocker(plan["model"])
    if blocker is not None:
        raise PreflightError(blocker)
    model = _namespace(plan["model"])
    model.resolved_image = plan.get("resolved", {}).get("image_digest")
    run_nonce = plan.get("run_nonce")
    if run_nonce is None:
        # Compatibility for schema-v2 plans frozen before per-plan ownership
        # nonces were introduced. New plans always take the collision-proof path.
        model.run_identity = f"{plan['fingerprint']}-{run_dir.name}"
    else:
        if (
            not isinstance(run_nonce, str)
            or re.fullmatch(r"[0-9a-f]{32}", run_nonce) is None
        ):
            raise RuntimeError("Frozen plan run nonce is invalid")
        model.run_identity = f"{plan['fingerprint']}-{run_nonce}"
    if str(getattr(model, "support_status", "")) == "incompatible":
        raise PreflightError(
            "This frozen model profile is marked incompatible and cannot be executed"
        )
    direct_commands = {
        "transformers": "diffusion-direct",
        "trtllm": "trtllm-direct",
    }
    direct_command = direct_commands.get(str(model.backend))
    if direct_command:
        raise PreflightError(
            f"{model.backend} direct profiles require the {direct_command} command"
        )
    suite = _namespace(plan["suite"])
    cases = list(suite.cases)
    _validate_prefix_cache_plan_selection(model, suite)
    _validate_memory_operation_plan_selection(model, suite)
    journal = Journal(run_dir / "events.jsonl")
    completed = journal.completed_cases()
    terminal = journal.terminal_cases()
    lock_path = results_lock_path(workspace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry = TelemetrySampler(run_dir / "telemetry.jsonl")
    host_safety: HostSafetyWatchdog | None = None
    server = None
    primary_error: BaseException | None = None
    primary_error_stage: str | None = None
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another SparkBench run holds the benchmark lock") from error
        try:
            lifecycle_changed = _recover_pending_lifecycle(
                model=model,
                journal=journal,
                run_dir=run_dir,
                workspace=workspace,
            )
        except BaseException as error:
            _record_run_aborted(journal, error, stage="lifecycle_recovery")
            try:
                summarize_run(run_dir)
            except Exception as summary_error:
                journal.append(
                    {
                        "event": "summary_failed",
                        "error_type": type(summary_error).__name__,
                        "error": str(summary_error),
                    }
                )
            raise
        if str(model.backend) == "llamacpp":
            try:
                require_llamacpp_mtp_evidence(
                    getattr(model, "args", ()), journal.events()
                )
                if llamacpp_dflash_requested(getattr(model, "args", ())):
                    require_llamacpp_dflash_evidence(
                        getattr(model, "args", ()), journal.events()
                    )
            except BaseException as error:
                stage = (
                    "dflash_evidence"
                    if llamacpp_dflash_requested(getattr(model, "args", ()))
                    else "mtp_evidence"
                )
                _record_run_aborted(journal, error, stage=stage)
                try:
                    summarize_run(run_dir)
                except Exception as summary_error:
                    journal.append(
                        {
                            "event": "summary_failed",
                            "error_type": type(summary_error).__name__,
                            "error": str(summary_error),
                        }
                    )
                raise
        if cases and all(case.case_id in terminal for case in cases):
            existing_summary = summarize_run(run_dir)
            if lifecycle_changed:
                existing_summary = summarize_run(run_dir)
            if existing_summary["status"] in {"aborted", "incomplete", "not_started"}:
                journal.append(
                    {
                        "event": "run_complete",
                        "status": "recovered_all_terminal" if completed else "no_work",
                    }
                )
                return summarize_run(run_dir)
            return existing_summary
        journal.append(
            {
                "event": "run_start",
                "completed_cases_at_resume": sorted(completed),
                "plan_fingerprint": str(plan["fingerprint"]),
                "run_nonce": run_nonce,
            }
        )
        if keep_server and str(model.backend) == "llamacpp":
            error = PreflightError(
                "--keep-server is not supported for managed llama.cpp processes"
            )
            _record_run_aborted(journal, error, stage="preflight")
            raise error
        runnable = []
        for case in cases:
            if case.case_id in terminal:
                continue
            missing = set(case.requires) - set(model.tasks)
            if missing:
                journal.append(
                    {
                        "event": "case_skipped_unsupported",
                        "case_id": case.case_id,
                        "missing_capabilities": sorted(missing),
                    }
                )
                continue
            unavailable_adapters: set[str] = set()
            adapter_reasons: list[str] = []
            if str(model.backend) != "vllm" and "rerank" in case.requires:
                unavailable_adapters.add("rerank")
                adapter_reasons.append("rerank requires the vLLM /score adapter")
            if str(model.backend) != "sglang" and "audio" in case.requires:
                unavailable_adapters.add("audio")
                adapter_reasons.append(
                    "audio requires SGLang with a registered speech LoRA"
                )
            if str(model.backend) != "vllm" and _is_multimodal_embedding_case(case):
                unavailable_adapters.update({"embeddings", "vision"})
                adapter_reasons.append(
                    "multimodal embeddings require the vLLM Chat Embeddings adapter"
                )
            if str(case.kind) == "agentic" and str(model.backend) == "ollama":
                unavailable_adapters.add("tools")
                adapter_reasons.append(
                    "multi-turn tool history requires an OpenAI-compatible chat adapter"
                )
            if unavailable_adapters:
                journal.append(
                    {
                        "event": "case_skipped_adapter_unimplemented",
                        "case_id": case.case_id,
                        "capabilities": sorted(unavailable_adapters),
                        "backend": str(model.backend),
                        "reason": "; ".join(adapter_reasons),
                    }
                )
                continue
            approximate_tokens, estimate_basis = _estimated_context_tokens(case)
            if approximate_tokens > int(model.max_context):
                journal.append(
                    {
                        "event": "case_skipped_context_limit",
                        "case_id": case.case_id,
                        "approximate_required_tokens": approximate_tokens,
                        "estimate_basis": estimate_basis,
                        "model_max_context": model.max_context,
                    }
                )
                continue
            runnable.append(case)
        if not runnable:
            journal.append({"event": "run_complete", "status": "no_work"})
            return summarize_run(run_dir)

        failure_stage = "host_safety_policy"
        measurement_started_ns = time.monotonic_ns()
        journal.append(
            {
                "event": "measurement_started",
                "monotonic_ns": measurement_started_ns,
                "plan_fingerprint": str(plan["fingerprint"]),
                "run_nonce": run_nonce,
            }
        )
        measurement_complete_ns: int | None = None
        try:
            host_safety = _host_safety_watchdog(model)
            if host_safety is not None and keep_server:
                raise PreflightError(
                    "--keep-server is not supported when host-safety "
                    "monitoring is enabled"
                )
            failure_stage = "preflight"
            _preflight(model)
            failure_stage = "telemetry_start"
            telemetry.start()
            try:
                failure_stage = "host_safety_start"
                if host_safety is not None:
                    host_safety.start()
                validated_llamacpp_artifacts: dict[str, Any] | None = None
                artifact_validation_s: float | None = None
                if str(model.backend) == "llamacpp":
                    failure_stage = "artifact_validation"
                    telemetry.set_phase("artifact_validation")
                    validation_started = time.perf_counter()
                    validated_llamacpp_artifacts = validate_llamacpp_artifacts(
                        model, workspace=workspace
                    )
                    artifact_validation_s = time.perf_counter() - validation_started
                    artifact_event = {
                        "event": "artifact_validation_complete",
                        "backend": str(model.backend),
                        "elapsed_s": artifact_validation_s,
                        "runtime_binary_sha256": (
                            validated_llamacpp_artifacts[
                                "runtime_binary_sha256"
                            ]
                        ),
                        "model_sha256": validated_llamacpp_artifacts[
                            "model_sha256"
                        ],
                    }
                    if "mmproj_sha256" in validated_llamacpp_artifacts:
                        artifact_event["mmproj_sha256"] = (
                            validated_llamacpp_artifacts["mmproj_sha256"]
                        )
                    if "draft_model_sha256" in validated_llamacpp_artifacts:
                        artifact_event["draft_model_sha256"] = (
                            validated_llamacpp_artifacts["draft_model_sha256"]
                        )
                    model_shards = validated_llamacpp_artifacts.get(
                        "model_shards"
                    )
                    if model_shards:
                        artifact_event.update(
                            {
                                "model_shard_count": len(model_shards),
                                "model_total_size_bytes": sum(
                                    int(shard["size_bytes"])
                                    for shard in model_shards
                                ),
                                "model_shard_sha256s": [
                                    str(shard["sha256"])
                                    for shard in model_shards
                                ],
                            }
                        )
                    journal.append(artifact_event)
                telemetry.set_phase("server_startup")
                failure_stage = "server_start"
                host_safety_callbacks: dict[str, Any] = {}
                if host_safety is not None:
                    host_safety_callbacks = {
                        "abort_check": host_safety.raise_if_tripped,
                        "on_server_created": (
                            lambda created_server: host_safety.register_abort_callback(
                                created_server.interrupt_owned
                            )
                        ),
                    }
                server = start_server(
                    model,
                    workspace=workspace,
                    allow_download=allow_download,
                    server_log_path=run_dir / "server" / "server.log",
                    process_state_path=run_dir / "server" / "process.json",
                    validated_llamacpp_artifacts=validated_llamacpp_artifacts,
                    artifact_validation_s=artifact_validation_s,
                    **host_safety_callbacks,
                )
                if host_safety is not None:
                    host_safety.raise_if_tripped()
                failure_stage = "server_provenance"
                provenance = capture_server_provenance(server)
                write_json(
                    run_dir / "server" / "provenance.json",
                    provenance,
                )
                if server.backend == "llamacpp":
                    Journal(run_dir / "server" / "provenance.jsonl").append(
                        {"event": "server_provenance", **provenance}
                    )
                journal.append(
                    {
                        "event": "server_ready",
                        "startup_s": server.startup_s,
                        "backend": server.backend,
                        "container_id": getattr(server, "container_id", None),
                        "process_pid": getattr(
                            getattr(server, "process", None), "pid", None
                        ),
                        "ollama_model": getattr(server, "ollama_model", None),
                        "ollama_unload_owned": bool(
                            getattr(server, "unload_ollama", False)
                        ),
                        "keep_server_requested": keep_server,
                    }
                )
                if host_safety is not None:
                    host_safety.raise_if_tripped()
                failure_stage = "first_request"
                telemetry.set_phase("first_request_after_start")
                first_request = _prime_model(server, model)
                if host_safety is not None:
                    host_safety.raise_if_tripped()
                journal.append(
                    {
                        "event": "first_request_complete",
                        "backend": server.backend,
                        "result": _request_result_payload(model, first_request),
                    }
                )
                failure_stage = "case_execution"
                for case in runnable:
                    if host_safety is not None:
                        host_safety.raise_if_tripped()
                    try:
                        _execute_case(
                            server=server,
                            model=model,
                            case=case,
                            journal=journal,
                            telemetry=telemetry,
                        )
                    except Exception:
                        if host_safety is not None:
                            host_safety.raise_if_tripped()
                        if not continue_on_error:
                            raise
                    if host_safety is not None:
                        host_safety.raise_if_tripped()
                nextn_depth = (
                    sglang_nextn_depth(getattr(model, "args", ()))
                    if server.backend == "sglang"
                    else None
                )
                if nextn_depth is not None:
                    failure_stage = "sglang_speculative_acceptance_audit"
                    telemetry.set_phase("sglang_speculative_acceptance_audit")
                    if (
                        not getattr(server, "container_id", None)
                        or getattr(server, "run_identity", None)
                        != str(model.run_identity)
                    ):
                        raise SGLangSpeculativeAuditError(
                            "Managed SGLang NEXTN server ownership is unavailable"
                        )
                    request_body_json = getattr(model, "request_body_json", None)
                    extra_body = (
                        json.loads(request_body_json) if request_body_json else {}
                    )
                    if not isinstance(extra_body, dict):
                        raise SGLangSpeculativeAuditError(
                            "SGLang profile request body must be an object"
                        )
                    chat_template_kwargs = extra_body.get("chat_template_kwargs")
                    if chat_template_kwargs is not None and not isinstance(
                        chat_template_kwargs, dict
                    ):
                        raise SGLangSpeculativeAuditError(
                            "SGLang chat-template arguments must be an object"
                        )
                    auth = getattr(server, "authorization", None)
                    spec_decode_metrics = request_sglang_speculative_audit(
                        base_url=server.base_url,
                        model=str(model.served_name),
                        authorization=auth,
                        expected_depth=nextn_depth,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                    journal.append(
                        {
                            "event": "sglang_spec_decode_metrics_snapshot",
                            "backend": server.backend,
                            "metrics": spec_decode_metrics,
                        }
                    )
                    if host_safety is not None:
                        host_safety.raise_if_tripped()
                failure_stage = "measurement_complete"
                measurement_complete_ns = time.monotonic_ns()
                journal.append(
                    {
                        "event": "measurement_complete",
                        "elapsed_s": (
                            measurement_complete_ns - measurement_started_ns
                        )
                        / 1_000_000_000,
                        "monotonic_ns": measurement_complete_ns,
                    }
                )
            except BaseException as error:
                safety_error = (
                    host_safety.failure if host_safety is not None else None
                )
                primary_error = safety_error or error
                primary_error_stage = failure_stage
                raise primary_error
            finally:
                failure_stage = "server_cleanup"
                telemetry.set_phase("server_shutdown")
                effective_keep_server = keep_server
                cleanup_started_ns = (
                    measurement_complete_ns
                    if measurement_complete_ns is not None
                    else time.monotonic_ns()
                )
                try:
                    if server:
                        try:
                            if host_safety is not None and host_safety.tripped:
                                _retry_host_safety_interrupt_if_needed(
                                    server, host_safety
                                )
                                _record_host_safety_interrupt_failure(
                                    journal,
                                    host_safety,
                                    stage=failure_stage,
                                )
                            # A vLLM metrics scrape is valid only for a server that this
                            # run actually started.  In particular, never probe a
                            # loopback endpoint supplied by an external or mocked
                            # server object: another benchmark may own that port.
                            if (
                                server.backend == "vllm"
                                and getattr(server, "container_id", None)
                                and getattr(server, "run_identity", None)
                                == str(model.run_identity)
                            ):
                                spec_decode_metrics = (
                                    snapshot_vllm_spec_decode_metrics(server.base_url)
                                )
                                if spec_decode_metrics is not None:
                                    journal.append(
                                        {
                                            "event": "vllm_spec_decode_metrics_snapshot",
                                            "backend": server.backend,
                                            "metrics": spec_decode_metrics,
                                        }
                                    )
                            if server.backend == "llamacpp":
                                spec_decode_metrics = (
                                    snapshot_llamacpp_spec_decode_metrics(
                                        server.base_url
                                    )
                                )
                                if spec_decode_metrics is not None:
                                    journal.append(
                                        {
                                            "event": (
                                                "llamacpp_spec_decode_metrics_snapshot"
                                            ),
                                            "backend": server.backend,
                                            "metrics": spec_decode_metrics,
                                        }
                                    )
                                if llamacpp_mtp_requested(
                                    getattr(model, "args", ())
                                ):
                                    require_mtp_activity(spec_decode_metrics)
                                if llamacpp_dflash_requested(
                                    getattr(model, "args", ())
                                ):
                                    require_speculative_activity(
                                        spec_decode_metrics, method="DFlash"
                                    )
                            save_server_logs(
                                server, run_dir / "server" / "server.log"
                            )
                        finally:
                            if host_safety is not None:
                                effective_keep_server = (
                                    keep_server and not host_safety.tripped
                                )
                            server.stop(keep_server=effective_keep_server)
                            if host_safety is not None:
                                # Keep monitoring through the exact owned stop.
                                # The server lifecycle lock serializes a racing
                                # watchdog callback with this cleanup path.
                                host_safety.stop()
                                if host_safety.tripped and effective_keep_server:
                                    effective_keep_server = False
                                    server.stop(keep_server=False)
                        server_stopped_ns = time.monotonic_ns()
                        journal.append(
                            {
                                "event": (
                                    "server_kept"
                                    if effective_keep_server
                                    else "server_stopped"
                                ),
                                "backend": server.backend,
                                "cleanup_elapsed_s": (
                                    server_stopped_ns - cleanup_started_ns
                                )
                                / 1_000_000_000,
                                "monotonic_ns": server_stopped_ns,
                            }
                        )
                except Exception as cleanup_error:
                    journal.append(
                        {
                            "event": "cleanup_failed",
                            "error_type": type(cleanup_error).__name__,
                            "error": str(cleanup_error),
                        }
                    )
                    if primary_error is None:
                        raise
                finally:
                    if host_safety is not None:
                        host_safety.stop()
                    telemetry.stop()
            if host_safety is not None:
                host_safety.raise_if_tripped()
            failure_stage = "mtp_evidence"
            if str(model.backend) == "llamacpp":
                require_llamacpp_mtp_evidence(
                    getattr(model, "args", ()), journal.events()
                )
                if llamacpp_dflash_requested(getattr(model, "args", ())):
                    failure_stage = "dflash_evidence"
                    require_llamacpp_dflash_evidence(
                        getattr(model, "args", ()), journal.events()
                    )
            journal.append(
                {
                    "event": "run_complete",
                    "status": (
                        "completed_server_kept"
                        if effective_keep_server
                        else "completed"
                    ),
                }
            )
        except BaseException as error:
            safety_error = host_safety.failure if host_safety is not None else None
            terminal_error = safety_error or error
            terminal_stage = (
                primary_error_stage
                if primary_error is terminal_error
                and primary_error_stage is not None
                else failure_stage
            )
            if isinstance(terminal_error, HostSafetyError):
                _record_host_safety_breach(
                    journal, terminal_error, stage=terminal_stage
                )
                if host_safety is not None:
                    _record_host_safety_interrupt_failure(
                        journal, host_safety, stage=terminal_stage
                    )
            _record_run_aborted(journal, terminal_error, stage=terminal_stage)
            try:
                summarize_run(run_dir)
            except Exception as summary_error:
                journal.append(
                    {
                        "event": "summary_failed",
                        "error_type": type(summary_error).__name__,
                        "error": str(summary_error),
                    }
                )
            raise terminal_error
    return summarize_run(run_dir)


def _sm121_storage_runtime_provenance_event(
    server: Any, *, fresh_server_lifetime: int
) -> dict[str, Any]:
    """Return the exact scalar-only native provenance event for one lifetime."""

    provenance = getattr(server, "native_provenance", None)
    if type(provenance) is not dict:
        raise RuntimeError("SM121 storage native provenance is unavailable")
    event = {
        "event": SM121_STORAGE_RUNTIME_PROVENANCE_EVENT,
        "fresh_server_lifetime": fresh_server_lifetime,
        **provenance,
    }
    try:
        validate_sm121_storage_runtime_provenance_event(
            event, fresh_server_lifetime=fresh_server_lifetime
        )
    except SM121StorageCandidateError as error:
        raise RuntimeError("SM121 storage native provenance is invalid") from error
    return event


def execute_sm121_storage_canary(run_dir: Path, *, workspace: Path) -> dict[str, Any]:
    """Execute the two SM121 admission cases in separate fresh lifetimes.

    The long-context retrieval request is the first inference request after
    its server becomes ready.  This intentionally does not call
    :func:`_prime_model`, resume a partial journal, retain a container, or
    reuse a process between the quality and long-context gates.
    """

    plan, model, suite = _load_sm121_storage_canary_plan(run_dir)
    journal = Journal(run_dir / "events.jsonl")
    if journal.events():
        raise PreflightError(
            "SM121 storage canary is non-resumable; freeze a new fresh plan"
        )
    cases = list(suite.cases)
    for case in cases:
        missing = set(case.requires) - set(model.tasks)
        if missing:
            raise PreflightError(
                "SM121 storage canary case has unsupported capabilities"
            )
        estimated_tokens, estimate_basis = _estimated_context_tokens(case)
        if estimated_tokens > int(model.max_context):
            raise PreflightError(
                "SM121 storage canary context admission is insufficient: "
                f"{estimated_tokens} via {estimate_basis}"
            )
    lock_path = results_lock_path(workspace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry = TelemetrySampler(run_dir / "telemetry.jsonl")
    measurement_started_ns = time.monotonic_ns()
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another SparkBench run holds the benchmark lock") from error
        journal.append(
            {
                "event": "run_start",
                "execution_mode": SM121_STORAGE_EXECUTION_MODE,
                "plan_fingerprint": str(plan["fingerprint"]),
                "run_nonce": str(plan["run_nonce"]),
            }
        )
        journal.append(
            {
                "event": "measurement_started",
                "monotonic_ns": measurement_started_ns,
                "plan_fingerprint": str(plan["fingerprint"]),
                "run_nonce": str(plan["run_nonce"]),
            }
        )
        stage = "preflight"
        active_watchdog: HostSafetyWatchdog | None = None
        try:
            telemetry.start()
            for lifetime, case in enumerate(cases, start=1):
                server = None
                active_watchdog = _host_safety_watchdog(model)
                try:
                    stage = f"preflight_lifetime_{lifetime}"
                    _preflight(model)
                    if active_watchdog is not None:
                        active_watchdog.start()
                    stage = f"server_start_lifetime_{lifetime}"
                    telemetry.set_phase(f"server_startup:{lifetime}")
                    callbacks: dict[str, Any] = {}
                    if active_watchdog is not None:
                        callbacks = {
                            "abort_check": active_watchdog.raise_if_tripped,
                            "on_server_created": (
                                lambda created_server: active_watchdog.register_abort_callback(
                                    created_server.interrupt_owned
                                )
                            ),
                        }
                    server = start_server(
                        model,
                        workspace=workspace,
                        allow_download=False,
                        server_log_path=(
                            run_dir / "server" / f"lifetime-{lifetime}" / "server.log"
                        ),
                        **callbacks,
                    )
                    if active_watchdog is not None:
                        active_watchdog.raise_if_tripped()
                    stage = f"server_provenance_lifetime_{lifetime}"
                    provenance = capture_server_provenance(server)
                    write_json(
                        run_dir
                        / "server"
                        / f"lifetime-{lifetime}"
                        / "provenance.json",
                        provenance,
                    )
                    journal.append(
                        _sm121_storage_runtime_provenance_event(
                            server, fresh_server_lifetime=lifetime
                        )
                    )
                    journal.append(
                        {
                            "event": "server_ready",
                            "backend": server.backend,
                            "startup_s": server.startup_s,
                            "fresh_server_lifetime": lifetime,
                            "first_inference_is_case": True,
                            "case_id": case.case_id,
                        }
                    )
                    stage = f"case_execution_lifetime_{lifetime}"
                    telemetry.set_phase(f"first_case_after_start:{lifetime}")
                    _execute_case(
                        server=server,
                        model=model,
                        case=case,
                        journal=journal,
                        telemetry=telemetry,
                    )
                    if str(case.kind) == "quality":
                        _require_sm121_storage_quality_gate(journal, case)
                    if active_watchdog is not None:
                        active_watchdog.raise_if_tripped()
                except BaseException as error:
                    safety_error = (
                        active_watchdog.failure
                        if active_watchdog is not None
                        else None
                    )
                    raise safety_error or error
                finally:
                    telemetry.set_phase(f"server_shutdown:{lifetime}")
                    cleanup_error: BaseException | None = None
                    try:
                        if server is not None:
                            if active_watchdog is not None and active_watchdog.tripped:
                                _retry_host_safety_interrupt_if_needed(
                                    server, active_watchdog
                                )
                                _record_host_safety_interrupt_failure(
                                    journal, active_watchdog, stage=stage
                                )
                            save_server_logs(
                                server,
                                run_dir
                                / "server"
                                / f"lifetime-{lifetime}"
                                / "server.log",
                            )
                            server.stop()
                            journal.append(
                                {
                                    "event": "server_stopped",
                                    "backend": server.backend,
                                    "fresh_server_lifetime": lifetime,
                                }
                            )
                    except BaseException as error:
                        cleanup_error = error
                        journal.append(
                            {
                                "event": "cleanup_failed",
                                "error_type": type(error).__name__,
                            }
                        )
                    finally:
                        if active_watchdog is not None:
                            active_watchdog.stop()
                    if cleanup_error is not None:
                        raise cleanup_error
                    if active_watchdog is not None:
                        active_watchdog.raise_if_tripped()
                    active_watchdog = None
            measurement_complete_ns = time.monotonic_ns()
            journal.append(
                {
                    "event": "measurement_complete",
                    "elapsed_s": (
                        measurement_complete_ns - measurement_started_ns
                    )
                    / 1_000_000_000,
                    "monotonic_ns": measurement_complete_ns,
                }
            )
            journal.append({"event": "run_complete", "status": "completed"})
        except BaseException as error:
            safety_error = (
                active_watchdog.failure if active_watchdog is not None else None
            )
            terminal_error = safety_error or error
            if isinstance(terminal_error, HostSafetyError):
                _record_host_safety_breach(journal, terminal_error, stage=stage)
                if active_watchdog is not None:
                    _record_host_safety_interrupt_failure(
                        journal, active_watchdog, stage=stage
                    )
            _record_run_aborted(journal, terminal_error, stage=stage)
            try:
                summarize_run(run_dir)
            except Exception as summary_error:
                journal.append(
                    {
                        "event": "summary_failed",
                        "error_type": type(summary_error).__name__,
                    }
                )
            raise terminal_error
        finally:
            if active_watchdog is not None:
                active_watchdog.stop()
            telemetry.stop()
    return summarize_run(run_dir)


def _sm121_cache_observability_zero_event(
    *,
    case: SimpleNamespace,
    attempt_id: str,
    result: dict[str, Any],
    before: dict[str, Any],
    before_settle_s: float,
    before_polls: int,
    before_settled: bool,
    after: dict[str, Any],
    after_settle_s: float,
    after_polls: int,
    after_settled: bool,
) -> dict[str, Any]:
    """Bind B0's scalar response and metric windows into one checked event."""

    event: dict[str, Any] = {
        "event": SM121_CACHE_ZERO_HIT_EVENT,
        "case_id": case.case_id,
        "protocol_case_id": SM121_CACHE_ZERO_HIT_CASE_ID,
        "attempt_id": attempt_id,
        "request_contract_sha256": SM121_CACHE_ZERO_HIT_REQUEST_CONTRACT_SHA256,
        "cache_details_requested": True,
        "streaming": False,
        "thinking_disabled": True,
        "response_detail_state": result["response_detail_state"],
        "usage_detail_state": result["usage_detail_state"],
        "response_device_cached_tokens": result["response_device_cached_tokens"],
        "response_host_cached_tokens": result["response_host_cached_tokens"],
        "response_storage_cached_tokens": result["response_storage_cached_tokens"],
        "usage_cached_tokens": result["usage_cached_tokens"],
        "metrics_available": bool(
            before.get("available") is True and after.get("available") is True
        ),
        "metrics_before_polls": before_polls,
        "metrics_after_polls": after_polls,
        "metrics_before_settle_s": before_settle_s,
        "metrics_after_settle_s": after_settle_s,
        "metrics_before_settled": before_settled,
        "metrics_after_settled": after_settled,
        "zero_hit_basis": "not_admitted",
        "zero_hit_admitted": False,
    }
    for prefix, snapshot in (("before", before), ("after", after)):
        for metric in SM121_CACHE_OBSERVABILITY_METRIC_FIELDS:
            event[f"{prefix}_{metric}"] = snapshot[metric]
        for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
            event[f"{prefix}_cached_{source}_series_present"] = snapshot[
                f"cached_{source}_series_present"
            ]
    for metric in SM121_CACHE_OBSERVABILITY_METRIC_FIELDS:
        event[f"delta_{metric}"] = event[f"after_{metric}"] - event[
            f"before_{metric}"
        ]
    admitted, basis = derive_sm121_cache_zero_hit_admission(event)
    event["zero_hit_admitted"] = admitted
    event["zero_hit_basis"] = basis
    try:
        validate_sm121_cache_zero_hit_event(event)
    except SM121CacheObservabilityError as error:
        raise SM121CacheObservabilityRequestError() from error
    return event


def _execute_sm121_cache_observability_case(
    *,
    server: Any,
    model: SimpleNamespace,
    case: SimpleNamespace,
    journal: Journal,
    telemetry: TelemetrySampler,
) -> None:
    """Run B0's one non-streaming request without recording response payloads."""

    if str(case.id) != SM121_CACHE_ZERO_HIT_CASE_ID:
        raise SM121CacheObservabilityRequestError()
    attempt_id = uuid.uuid4().hex
    journal.append(
        {
            "event": "case_start",
            "case_id": case.case_id,
            "attempt_id": attempt_id,
            "kind": case.kind,
            "concurrency": case.concurrency,
        }
    )
    telemetry.set_phase(f"case:{case.case_id}:{attempt_id}")
    started = time.perf_counter()
    try:
        before, before_settle_s, before_polls, before_settled = (
            settle_sm121_cache_observability_metrics(server)
        )
        result = request_sm121_cache_observability_zero_hit(
            server, served_name=model.served_name
        )
        after, after_settle_s, after_polls, after_settled = (
            settle_sm121_cache_observability_metrics(server)
        )
        event = _sm121_cache_observability_zero_event(
            case=case,
            attempt_id=attempt_id,
            result=result,
            before=before,
            before_settle_s=before_settle_s,
            before_polls=before_polls,
            before_settled=before_settled,
            after=after,
            after_settle_s=after_settle_s,
            after_polls=after_polls,
            after_settled=after_settled,
        )
        journal.append(event)
        journal.append(
            {
                "event": "request_complete",
                "case_id": case.case_id,
                "attempt_id": attempt_id,
                "kind": case.kind,
                "repetition": 0,
                "burst_elapsed_s": result["elapsed_s"],
                "result": {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "reasoning_tokens": result["reasoning_tokens"],
                    "ttft_s": None,
                    "elapsed_s": result["elapsed_s"],
                    "decode_s": None,
                    "decode_tps": None,
                    "output_tps": None,
                    "emission_events": 1,
                    "finish_reason": None,
                    "response_model": None,
                    "decode_metric_source": None,
                },
                "validation": {"passed": event["zero_hit_admitted"]},
            }
        )
        journal.append(
            {
                "event": "case_complete",
                "case_id": case.case_id,
                "attempt_id": attempt_id,
                "kind": case.kind,
                "concurrency": case.concurrency,
                "elapsed_s": time.perf_counter() - started,
                "validation_passed": event["zero_hit_admitted"],
            }
        )
    except BaseException as error:
        safe_error = SM121CacheObservabilityRequestError()
        journal.append(
            {
                "event": "case_failed",
                "case_id": case.case_id,
                "attempt_id": attempt_id,
                "error_type": type(safe_error).__name__,
                "error": str(safe_error),
                "elapsed_s": time.perf_counter() - started,
            }
        )
        raise safe_error from error
    finally:
        telemetry.set_phase("between_cases")


def execute_sm121_cache_observability_canary(
    run_dir: Path, *, workspace: Path
) -> dict[str, Any]:
    """Execute B0 on one fresh cache-off server, with no cache-policy claim."""

    plan, model, suite = _load_sm121_cache_observability_plan(run_dir)
    journal = Journal(run_dir / "events.jsonl")
    if journal.events():
        raise PreflightError(
            "SM121 cache-observability canary is non-resumable; freeze a new fresh plan"
        )
    cases = list(suite.cases)
    if len(cases) != 2 or str(cases[1].id) != SM121_CACHE_ZERO_HIT_CASE_ID:
        raise PreflightError("SM121 cache-observability cases are invalid")
    for case in cases:
        missing = set(case.requires) - set(model.tasks)
        if missing:
            raise PreflightError(
                "SM121 cache-observability case has unsupported capabilities"
            )
        estimated_tokens, estimate_basis = _estimated_context_tokens(case)
        if estimated_tokens > int(model.max_context):
            raise PreflightError(
                "SM121 cache-observability context admission is insufficient: "
                f"{estimated_tokens} via {estimate_basis}"
            )
    lock_path = results_lock_path(workspace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry = TelemetrySampler(run_dir / "telemetry.jsonl")
    measurement_started_ns = time.monotonic_ns()
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another SparkBench run holds the benchmark lock") from error
        journal.append(
            {
                "event": "run_start",
                "execution_mode": SM121_CACHE_OBSERVABILITY_EXECUTION_MODE,
                "plan_fingerprint": str(plan["fingerprint"]),
                "run_nonce": str(plan["run_nonce"]),
            }
        )
        journal.append(
            {
                "event": "measurement_started",
                "monotonic_ns": measurement_started_ns,
                "plan_fingerprint": str(plan["fingerprint"]),
                "run_nonce": str(plan["run_nonce"]),
            }
        )
        stage = "preflight"
        active_watchdog: HostSafetyWatchdog | None = None
        server = None
        terminal_error: BaseException | None = None
        try:
            telemetry.start()
            _preflight(model)
            stage = "static_cache_source_attestation"
            journal.append(attest_sm121_cache_observability_static_source(model))
            active_watchdog = _host_safety_watchdog(model)
            if active_watchdog is not None:
                active_watchdog.start()
            stage = "server_start"
            telemetry.set_phase("server_startup:1")
            callbacks: dict[str, Any] = {}
            if active_watchdog is not None:
                callbacks = {
                    "abort_check": active_watchdog.raise_if_tripped,
                    "on_server_created": (
                        lambda created_server: active_watchdog.register_abort_callback(
                            created_server.interrupt_owned
                        )
                    ),
                }
            server = start_server(
                model,
                workspace=workspace,
                allow_download=False,
                server_log_path=run_dir / "server" / "lifetime-1" / "server.log",
                **callbacks,
            )
            if active_watchdog is not None:
                active_watchdog.raise_if_tripped()
            stage = "server_provenance"
            write_json(
                run_dir / "server" / "lifetime-1" / "provenance.json",
                capture_server_provenance(server),
            )
            stage = "runtime_cache_attestation"
            journal.append(attest_sm121_cache_observability_runtime(server))
            journal.append(
                {
                    "event": "server_ready",
                    "backend": server.backend,
                    "startup_s": server.startup_s,
                    "fresh_server_lifetime": 1,
                    "first_inference_is_case": True,
                    "case_id": cases[0].case_id,
                }
            )
            stage = "quality_case"
            telemetry.set_phase("first_case_after_start:1")
            _execute_case(
                server=server,
                model=model,
                case=cases[0],
                journal=journal,
                telemetry=telemetry,
            )
            _require_sm121_storage_quality_gate(journal, cases[0])
            if active_watchdog is not None:
                active_watchdog.raise_if_tripped()
            stage = "zero_hit_observation"
            _execute_sm121_cache_observability_case(
                server=server,
                model=model,
                case=cases[1],
                journal=journal,
                telemetry=telemetry,
            )
            if active_watchdog is not None:
                active_watchdog.raise_if_tripped()
        except BaseException as error:
            safety_error = active_watchdog.failure if active_watchdog is not None else None
            terminal_error = safety_error or error
        finally:
            telemetry.set_phase("server_shutdown:1")
            cleanup_error: BaseException | None = None
            if server is not None:
                if active_watchdog is not None and active_watchdog.tripped:
                    try:
                        _retry_host_safety_interrupt_if_needed(server, active_watchdog)
                        _record_host_safety_interrupt_failure(
                            journal, active_watchdog, stage=stage
                        )
                    except BaseException as error:
                        cleanup_error = error
                try:
                    save_server_logs(
                        server, run_dir / "server" / "lifetime-1" / "server.log"
                    )
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
                try:
                    server.stop()
                    journal.append(
                        {
                            "event": "server_stopped",
                            "backend": server.backend,
                            "fresh_server_lifetime": 1,
                        }
                    )
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
            if cleanup_error is not None:
                journal.append(
                    {
                        "event": "cleanup_failed",
                        "error_type": type(cleanup_error).__name__,
                    }
                )
            if active_watchdog is not None:
                active_watchdog.stop()
            if cleanup_error is not None and terminal_error is None:
                terminal_error = cleanup_error
            if terminal_error is None and active_watchdog is not None:
                try:
                    active_watchdog.raise_if_tripped()
                except BaseException as error:
                    terminal_error = error
        if terminal_error is not None:
            telemetry.stop()
            if isinstance(terminal_error, HostSafetyError):
                _record_host_safety_breach(journal, terminal_error, stage=stage)
                if active_watchdog is not None:
                    _record_host_safety_interrupt_failure(
                        journal, active_watchdog, stage=stage
                    )
            _record_run_aborted(journal, terminal_error, stage=stage)
            try:
                summarize_run(run_dir)
            except Exception as summary_error:
                journal.append(
                    {"event": "summary_failed", "error_type": type(summary_error).__name__}
                )
            raise terminal_error
        try:
            if active_watchdog is not None:
                active_watchdog.raise_if_tripped()
            measurement_complete_ns = time.monotonic_ns()
            journal.append(
                {
                    "event": "measurement_complete",
                    "elapsed_s": (
                        measurement_complete_ns - measurement_started_ns
                    )
                    / 1_000_000_000,
                    "monotonic_ns": measurement_complete_ns,
                }
            )
            journal.append({"event": "run_complete", "status": "completed"})
        except BaseException as error:
            safety_error = active_watchdog.failure if active_watchdog is not None else None
            terminal_error = safety_error or error
            if isinstance(terminal_error, HostSafetyError):
                _record_host_safety_breach(journal, terminal_error, stage=stage)
                if active_watchdog is not None:
                    _record_host_safety_interrupt_failure(
                        journal, active_watchdog, stage=stage
                    )
            _record_run_aborted(journal, terminal_error, stage=stage)
            try:
                summarize_run(run_dir)
            except Exception as summary_error:
                journal.append(
                    {"event": "summary_failed", "error_type": type(summary_error).__name__}
                )
            raise terminal_error
        finally:
            telemetry.stop()
    return summarize_run(run_dir)


def results_lock_path(workspace: Path) -> Path:
    return workspace / "results" / ".sparkbench.lock"
