"""Dedicated fresh-lifetime executor for SM121 chunked-prefill studies.

This controller is intentionally independent of the cache-policy performance
lane.  Both arms keep UnifiedRadixCache and lazy Mamba state; only the
resolved chunked-prefill size changes.  Prompt text, completions, prompt token
IDs, request IDs, credentials, logs, and server-info bodies remain transient
or in ignored raw run directories.  Public summaries contain scalar-only
attestations and timings.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from types import SimpleNamespace
from typing import Any, Mapping
import uuid

from . import runner as base_runner
from .host_safety import HostSafetyError, HostSafetyWatchdog
from .journal import Journal, content_hash, utc_now, write_json
from .sm121_chunked_prefill_admission_runner import (
    load_verified_sm121_chunked_prefill_8k_admission_receipt,
)
from .runtime import (
    inspect_sm121_cache_source_digests,
    inspect_sm121_chunked_prefill_runtime_identity,
    request_sm121_cache_semantic_turn,
    save_server_logs,
    settle_sm121_cache_observability_metrics,
)
from .sglang_sm121_cache_semantic import SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS
from .sglang_sm121_chunked_prefill_performance import (
    ChunkedPrefillPerformanceStudy,
    SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_ARM,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_ARM,
    SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MAX_TOKENS,
    SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MIN_TOKENS,
    SM121_CHUNKED_PREFILL_PERFORMANCE_METRIC_FIELDS,
    SM121_CHUNKED_PREFILL_PERFORMANCE_PAIR_BINDING_SCHEMA_VERSION,
    SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT,
    SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EVENT,
    SM121_CHUNKED_PREFILL_PERFORMANCE_STATIC_EVENT,
    SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS,
    SM121_CHUNKED_PREFILL_PERFORMANCE_TURN_EVENT,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_PAIR_BINDING_SCHEMA_VERSION,
    SM121ChunkedPrefillPerformanceError,
    derive_sm121_chunked_prefill_performance_turn_admission,
    is_sm121_chunked_prefill_performance_plan,
    score_sm121_chunked_prefill_performance_campaign,
    sm121_chunked_prefill_performance_arm,
    sm121_chunked_prefill_performance_study,
    sm121_chunked_prefill_performance_pair_binding_sha256,
    sm121_chunked_prefill_performance_pair_instance_sha256,
    validate_sm121_chunked_prefill_performance_candidate,
    validate_sm121_chunked_prefill_performance_pair,
    validate_sm121_chunked_prefill_performance_pair_binding,
    validate_sm121_chunked_prefill_performance_runtime_event,
    validate_sm121_chunked_prefill_performance_static_event,
    validate_sm121_chunked_prefill_performance_suite,
    validate_sm121_chunked_prefill_performance_turn_event,
)
from .sglang_sm121_chunked_prefill_admission import (
    SM121ChunkedPrefill8KAdmissionError,
    validate_sm121_chunked_prefill_8k_admission_receipt,
    validate_sm121_chunked_prefill_8k_admission_receipt_for_v3_candidate_plan,
)
from .sglang_sm121_storage import (
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_SOURCE_TREE,
)
from .telemetry import TelemetrySampler


_FAILURE_MESSAGE = "SM121 chunked-prefill performance request failed; details omitted"
_LEDGER_WORD = "shared-ledger-entry "
# The former 10,240-word ledger measured 41,017 prompt tokens with this
# model/template.  15,000 is intentionally checked at request time against a
# scalar 56--62Ki token window; the text itself is never persisted.
_LEDGER_REPETITIONS = 15_000
_EXPECTED_RESPONSES = (
    "CHUNKED-PREFILL-T0-17",
    "CHUNKED-PREFILL-T1-29",
    "CHUNKED-PREFILL-T2-43",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_V3_LOGS_ROOT = _REPOSITORY_ROOT / "logs"


class SM121ChunkedPrefillPerformanceRequestError(RuntimeError):
    """Public-safe terminal request failure for this controller."""

    def __init__(self) -> None:
        super().__init__(_FAILURE_MESSAGE)


def _require_private_v3_directory(path: Path, *, create: bool) -> Path:
    """Require V3 raw output under an owned private ignored logs subtree."""

    logs_root = _V3_LOGS_ROOT
    if logs_root.is_symlink():
        raise RuntimeError("SM121 chunked-prefill V3 output topology is invalid")
    if not logs_root.exists():
        if not create:
            raise RuntimeError("SM121 chunked-prefill V3 output is unavailable")
        logs_root.mkdir(mode=0o700)
        os.chmod(logs_root, 0o700)
    if not logs_root.is_dir():
        raise RuntimeError("SM121 chunked-prefill V3 output topology is invalid")
    candidate = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = candidate.relative_to(logs_root)
    except ValueError as error:
        raise RuntimeError(
            "SM121 chunked-prefill V3 output must remain under ignored private logs"
        ) from error
    if candidate == logs_root:
        raise RuntimeError("SM121 chunked-prefill V3 output location is invalid")
    current = logs_root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise RuntimeError("SM121 chunked-prefill V3 output topology is invalid")
        if current.exists():
            metadata = current.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise RuntimeError("SM121 chunked-prefill V3 output is not private")
            continue
        if not create:
            raise RuntimeError("SM121 chunked-prefill V3 output is unavailable")
        current.mkdir(mode=0o700)
        os.chmod(current, 0o700)
    return candidate


def _require_private_v3_run_directory(run_dir: Path, *, harden: bool = True) -> None:
    """Require or harden one V3 plan leaf and its direct raw plan files."""

    metadata = run_dir.lstat()
    if (
        run_dir.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise RuntimeError("SM121 chunked-prefill V3 run topology is invalid")
    if harden:
        os.chmod(run_dir, 0o700)
    _require_private_v3_directory(run_dir, create=False)
    for child in run_dir.iterdir():
        _require_private_v3_regular_file(child, harden=harden)


def _require_private_v3_regular_file(path: Path, *, harden: bool = True) -> None:
    """Require or harden one V3 raw file to owned, unlinked mode 0600."""

    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
    ):
        raise RuntimeError("SM121 chunked-prefill V3 run topology is invalid")
    if harden:
        os.chmod(path, 0o600)
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise RuntimeError("SM121 chunked-prefill V3 output is not private")


def create_sm121_chunked_prefill_performance_campaign(
    *,
    control_model: Any,
    candidate_model: Any,
    suite: Any,
    results_root: Path,
    models_path: Path,
    suite_path: Path,
    admission_run_dir: Path | None = None,
) -> Path:
    """Freeze one non-resumable A/B/B/A chunk-size campaign without serving."""

    try:
        validate_sm121_chunked_prefill_performance_pair(control_model, candidate_model)
        validate_sm121_chunked_prefill_performance_suite(suite)
        study = sm121_chunked_prefill_performance_study(control_model)
        if study != sm121_chunked_prefill_performance_study(getattr(suite, "id", None)):
            raise SM121ChunkedPrefillPerformanceError(
                "chunked-prefill profile and suite studies differ"
            )
    except SM121ChunkedPrefillPerformanceError as error:
        raise RuntimeError("SM121 chunked-prefill admission is unavailable") from error
    admission_receipt: dict[str, object] | None = None
    if study == SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY:
        if admission_run_dir is None:
            raise RuntimeError(
                "SM121 chunked-prefill v3 requires a verified 8K admission receipt"
            )
        try:
            admission_receipt = (
                load_verified_sm121_chunked_prefill_8k_admission_receipt(
                    admission_run_dir
                )
            )
            validate_sm121_chunked_prefill_8k_admission_receipt(admission_receipt)
        except (base_runner.PreflightError, SM121ChunkedPrefill8KAdmissionError) as error:
            raise RuntimeError(
                "SM121 chunked-prefill v3 requires a verified 8K admission receipt"
            ) from error
    elif admission_run_dir is not None:
        raise RuntimeError("SM121 chunked-prefill admission receipt is only valid for v3")
    is_v3 = study == SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY
    if is_v3:
        results_root = _require_private_v3_directory(results_root, create=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign_dir = results_root / f"{stamp}-{study.campaign_id}"
    if is_v3:
        campaign_dir.mkdir(mode=0o700)
        _require_private_v3_directory(campaign_dir, create=False)
    else:
        campaign_dir.mkdir(parents=True, exist_ok=False)
    runs_root = campaign_dir / "runs"
    if is_v3:
        runs_root.mkdir(mode=0o700)
        _require_private_v3_directory(runs_root, create=False)
    arm_models = {
        SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_ARM: control_model,
        SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_ARM: candidate_model,
    }
    run_dirs: list[Path] = []
    for ordinal, arm in enumerate(SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER, start=1):
        run_dir = base_runner.create_plan(
            model=arm_models[arm],
            suite=suite,
            results_root=runs_root,
            models_path=models_path,
            suite_path=suite_path,
            allow_sm121_chunked_prefill_performance=True,
            run_label=f"prefill-{ordinal}-{arm.lower()}",
        )
        if is_v3:
            _require_private_v3_run_directory(run_dir)
        run_dirs.append(run_dir)
    _bind_campaign_plans(
        run_dirs,
        study=study,
        admission_receipt=admission_receipt,
    )
    if is_v3:
        for run_dir in run_dirs:
            _require_private_v3_run_directory(run_dir)
    plans = [json.loads((run_dir / "plan.json").read_text()) for run_dir in run_dirs]
    binding = plans[0].get("chunked_prefill_performance_pair")
    if not isinstance(binding, dict):
        raise RuntimeError("SM121 chunked-prefill plan binding is unavailable")
    campaign = {
        "schema_version": 1,
        "campaign_id": study.campaign_id,
        "created_at": utc_now(),
        "execution_mode": study.execution_mode,
        "pair_binding": binding,
        "run_directories": [run_dir.name for run_dir in run_dirs],
    }
    if admission_receipt is not None:
        campaign["v3_admission_receipt"] = admission_receipt
    campaign["integrity_hash"] = content_hash(campaign, 64)
    write_json(campaign_dir / "campaign.json", campaign)
    if is_v3:
        _require_private_v3_regular_file(campaign_dir / "campaign.json")
    return campaign_dir


def _bind_campaign_plans(
    run_dirs: list[Path],
    *,
    study: ChunkedPrefillPerformanceStudy,
    admission_receipt: dict[str, object] | None,
) -> None:
    """Bind four frozen plans to one opaque, scalar-only instance digest."""

    if len(run_dirs) != len(SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER):
        raise RuntimeError("SM121 chunked-prefill plan count is invalid")
    plans: list[dict[str, Any]] = []
    for ordinal, (run_dir, arm) in enumerate(
        zip(run_dirs, SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER, strict=True),
        start=1,
    ):
        try:
            plan = json.loads((run_dir / "plan.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("SM121 chunked-prefill plan is unreadable") from error
        if type(plan) is not dict or type(plan.get("model")) is not dict or type(plan.get("suite")) is not dict:
            raise RuntimeError("SM121 chunked-prefill plan is invalid")
        try:
            if sm121_chunked_prefill_performance_arm(plan["model"]) != arm:
                raise SM121ChunkedPrefillPerformanceError("chunked-prefill plan arm changed")
            if sm121_chunked_prefill_performance_study(plan["model"]) != study:
                raise SM121ChunkedPrefillPerformanceError("chunked-prefill plan study changed")
            validate_sm121_chunked_prefill_performance_candidate(plan["model"])
            validate_sm121_chunked_prefill_performance_suite(
                base_runner._namespace(plan["suite"])
            )
        except SM121ChunkedPrefillPerformanceError as error:
            raise RuntimeError("SM121 chunked-prefill plan is invalid") from error
        if not isinstance(plan.get("fingerprint"), str) or re.fullmatch(
            r"[0-9a-f]{16}", plan["fingerprint"]
        ) is None:
            raise RuntimeError("SM121 chunked-prefill fingerprint is invalid")
        plan["chunked_prefill_performance_ordinal"] = ordinal
        plans.append(plan)
    if study == SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY:
        if admission_receipt is None:
            raise RuntimeError("SM121 chunked-prefill V3 admission receipt is unavailable")
        try:
            for plan in plans:
                if sm121_chunked_prefill_performance_arm(plan["model"]) == "B":
                    validate_sm121_chunked_prefill_8k_admission_receipt_for_v3_candidate_plan(
                        admission_receipt, plan
                    )
        except SM121ChunkedPrefill8KAdmissionError as error:
            raise RuntimeError("SM121 chunked-prefill V3 admission receipt is invalid") from error
    elif admission_receipt is not None:
        raise RuntimeError("SM121 chunked-prefill admission receipt is only valid for v3")
    try:
        instance = sm121_chunked_prefill_performance_pair_instance_sha256(
            [plan.get("run_nonce") for plan in plans]
        )
    except SM121ChunkedPrefillPerformanceError as error:
        raise RuntimeError("SM121 chunked-prefill plan nonce is invalid") from error
    binding: dict[str, object] = {
        "schema_version": (
            SM121_CHUNKED_PREFILL_PERFORMANCE_V3_PAIR_BINDING_SCHEMA_VERSION
            if study == SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY
            else SM121_CHUNKED_PREFILL_PERFORMANCE_PAIR_BINDING_SCHEMA_VERSION
        ),
        "suite_id": study.suite_id,
        "execution_mode": study.execution_mode,
        "arm_order": list(SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER),
        "profile_ids": [
            study.control_profile_id,
            study.candidate_profile_id,
        ],
        "chunked_prefill_sizes": [
            study.control_chunk_size,
            study.candidate_chunk_size,
        ],
        "quality_case_id": SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID,
        "timed_case_id": study.timed_case_id,
        "cell_timeout_s": SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S,
        "campaign_instance_sha256": instance,
        "plan_fingerprints": [str(plan["fingerprint"]) for plan in plans],
    }
    if admission_receipt is not None:
        binding["admission_receipt_sha256"] = admission_receipt[
            "receipt_integrity_hash"
        ]
    binding["pair_binding_sha256"] = (
        sm121_chunked_prefill_performance_pair_binding_sha256(binding)
    )
    try:
        validate_sm121_chunked_prefill_performance_pair_binding(binding)
    except SM121ChunkedPrefillPerformanceError as error:
        raise RuntimeError("SM121 chunked-prefill pair binding is invalid") from error
    for run_dir, plan in zip(run_dirs, plans, strict=True):
        plan["chunked_prefill_performance_pair"] = binding
        plan["integrity_hash"] = content_hash(
            {key: value for key, value in plan.items() if key != "integrity_hash"},
            64,
        )
        write_json(run_dir / "plan.json", plan)


def _load_plan(
    run_dir: Path,
) -> tuple[
    dict[str, Any],
    SimpleNamespace,
    SimpleNamespace,
    ChunkedPrefillPerformanceStudy,
]:
    """Authenticate one frozen performance arm before it can create a server."""

    try:
        plan = json.loads((run_dir / "plan.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise base_runner.PreflightError("SM121 chunked-prefill plan is unavailable") from error
    if type(plan) is not dict or plan.get("schema_version") != 2:
        raise base_runner.PreflightError("SM121 chunked-prefill plan schema is invalid")
    model_data, suite_data, resolved = (
        plan.get("model"),
        plan.get("suite"),
        plan.get("resolved"),
    )
    if type(model_data) is not dict or type(suite_data) is not dict or type(resolved) is not dict:
        raise base_runner.PreflightError("SM121 chunked-prefill plan core fields are invalid")
    integrity = plan.get("integrity_hash")
    if not isinstance(integrity, str) or content_hash(
        {key: value for key, value in plan.items() if key != "integrity_hash"},
        len(integrity),
    ) != integrity:
        raise base_runner.PreflightError("SM121 chunked-prefill plan integrity is invalid")
    cases = suite_data.get("cases")
    if not isinstance(cases, list) or any(type(case) is not dict for case in cases):
        raise base_runner.PreflightError("SM121 chunked-prefill plan cases are invalid")
    suite_without_case_ids = {
        **suite_data,
        "cases": [{key: value for key, value in case.items() if key != "case_id"} for case in cases],
    }
    if plan.get("fingerprint") != content_hash(
        {"model": model_data, "suite": suite_without_case_ids, "resolved": resolved}
    ):
        raise base_runner.PreflightError("SM121 chunked-prefill plan fingerprint is invalid")
    for case in cases:
        case_without_id = {key: value for key, value in case.items() if key != "case_id"}
        expected_case_id = base_runner._canonical_case(
            model_data,
            case_without_id,
            protocol_digest=suite_data.get("protocol_digest"),
        )["case_id"]
        if case.get("case_id") != expected_case_id:
            raise base_runner.PreflightError("SM121 chunked-prefill case identity is invalid")
    model = base_runner._namespace(model_data)
    suite = base_runner._namespace(suite_data)
    try:
        if not is_sm121_chunked_prefill_performance_plan(model, suite):
            raise SM121ChunkedPrefillPerformanceError("chunked-prefill selector is invalid")
        validate_sm121_chunked_prefill_performance_candidate(model)
        validate_sm121_chunked_prefill_performance_suite(suite)
        study = sm121_chunked_prefill_performance_study(model)
        if study != sm121_chunked_prefill_performance_study(getattr(suite, "id", None)):
            raise SM121ChunkedPrefillPerformanceError("chunked-prefill plan study changed")
    except SM121ChunkedPrefillPerformanceError as error:
        raise base_runner.PreflightError("SM121 chunked-prefill plan contract is invalid") from error
    local_image = resolved.get("local_image")
    if (
        type(local_image) is not dict
        or set(local_image) != {"docker_image_id", "platform", "source_tree"}
        or local_image.get("docker_image_id") != SM121_STORAGE_LOCAL_IMAGE_ID
        or local_image.get("platform") != SM121_STORAGE_PLATFORM
        or local_image.get("source_tree") != SM121_STORAGE_SOURCE_TREE
    ):
        raise base_runner.PreflightError("SM121 chunked-prefill local image changed")
    binding = plan.get("chunked_prefill_performance_pair")
    try:
        validate_sm121_chunked_prefill_performance_pair_binding(binding)
    except SM121ChunkedPrefillPerformanceError as error:
        raise base_runner.PreflightError("SM121 chunked-prefill pair binding is invalid") from error
    ordinal = plan.get("chunked_prefill_performance_ordinal")
    if type(ordinal) is not int or not 1 <= ordinal <= len(SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER):
        raise base_runner.PreflightError("SM121 chunked-prefill plan ordinal is invalid")
    if sm121_chunked_prefill_performance_arm(model) != SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER[ordinal - 1]:
        raise base_runner.PreflightError("SM121 chunked-prefill plan arm is invalid")
    if not isinstance(binding, dict) or binding["plan_fingerprints"][ordinal - 1] != plan.get("fingerprint"):
        raise base_runner.PreflightError("SM121 chunked-prefill plan binding moved")
    if not isinstance(binding, dict) or binding.get("suite_id") != study.suite_id:
        raise base_runner.PreflightError("SM121 chunked-prefill plan study changed")
    run_nonce = plan.get("run_nonce")
    if not isinstance(run_nonce, str) or re.fullmatch(r"[0-9a-f]{32}", run_nonce) is None:
        raise base_runner.PreflightError("SM121 chunked-prefill run nonce is invalid")
    model.resolved_local_image_id = local_image["docker_image_id"]
    model.run_identity = f"{plan['fingerprint']}-{run_nonce}"
    model.chunked_prefill_performance_authorized = True
    model.chunked_prefill_performance_pair = binding
    return plan, model, suite, study


def _case_pair(
    suite: SimpleNamespace, *, study: ChunkedPrefillPerformanceStudy
) -> tuple[SimpleNamespace, SimpleNamespace]:
    cases = list(getattr(suite, "cases", ()))
    if len(cases) != 2:
        raise base_runner.PreflightError("SM121 chunked-prefill cases are invalid")
    quality, timed = cases
    if (
        getattr(quality, "id", None) != SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID
        or getattr(timed, "id", None) != study.timed_case_id
        or not isinstance(getattr(quality, "case_id", None), str)
        or not isinstance(getattr(timed, "case_id", None), str)
    ):
        raise base_runner.PreflightError("SM121 chunked-prefill cases are invalid")
    return quality, timed


def _static_event(
    *,
    model: SimpleNamespace,
    study: ChunkedPrefillPerformanceStudy,
    arm: str,
    lifetime_ordinal: int,
) -> dict[str, Any]:
    event = {
        "event": SM121_CHUNKED_PREFILL_PERFORMANCE_STATIC_EVENT,
        "arm": arm,
        "lifetime_ordinal": lifetime_ordinal,
        "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
        "chunked_prefill_size": (
            study.control_chunk_size
            if arm == SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_ARM
            else study.candidate_chunk_size
        ),
        **inspect_sm121_cache_source_digests(model),
        **SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
    }
    try:
        validate_sm121_chunked_prefill_performance_static_event(event)
    except SM121ChunkedPrefillPerformanceError as error:
        raise SM121ChunkedPrefillPerformanceRequestError() from error
    return event


def _runtime_event(
    *,
    server: Any,
    study: ChunkedPrefillPerformanceStudy,
    arm: str,
    lifetime_ordinal: int,
) -> dict[str, Any]:
    event = {
        "event": SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EVENT,
        "arm": arm,
        "lifetime_ordinal": lifetime_ordinal,
        **inspect_sm121_chunked_prefill_runtime_identity(server),
    }
    try:
        validate_sm121_chunked_prefill_performance_runtime_event(event)
    except SM121ChunkedPrefillPerformanceError as error:
        raise SM121ChunkedPrefillPerformanceRequestError() from error
    expected_chunk_size = (
        study.control_chunk_size
        if arm == SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_ARM
        else study.candidate_chunk_size
    )
    if event["chunked_prefill_size"] != expected_chunk_size:
        raise SM121ChunkedPrefillPerformanceRequestError()
    return event


def _remaining_s(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise SM121ChunkedPrefillPerformanceRequestError()
    return remaining


def _abort_check(*, watchdog: HostSafetyWatchdog | None, deadline: float) -> None:
    if watchdog is not None:
        watchdog.raise_if_tripped()
    _remaining_s(deadline)


def _interrupt_terminal_server(
    *, server: Any, deadline: float, terminal_error: BaseException | None
) -> None:
    if terminal_error is not None or time.monotonic() >= deadline:
        server.interrupt_owned()


def _messages() -> tuple[list[dict[str, str]], ...]:
    """Build fixed T0/T1/T2 history only in memory.

    No model-generated text is ever reused.  The append messages use known
    synthetic replies, so a private token-ID equality check proves the exact
    input remained matched across arms and fresh lifetimes.
    """

    system = {
        "role": "system",
        "content": "Follow the synthetic ledger protocol exactly. Reply with only the requested token and no explanation.",
    }
    ledger = _LEDGER_WORD * _LEDGER_REPETITIONS
    t0 = [
        system,
        {
            "role": "user",
            "content": "Read the complete synthetic ledger before replying.\n"
            + ledger
            + "\nReturn exactly "
            + _EXPECTED_RESPONSES[0],
        },
    ]
    t1 = [
        *t0,
        {"role": "assistant", "content": _EXPECTED_RESPONSES[0]},
        {
            "role": "user",
            "content": "Keep the same ledger history and return exactly "
            + _EXPECTED_RESPONSES[1],
        },
    ]
    t2 = [
        *t1,
        {"role": "assistant", "content": _EXPECTED_RESPONSES[1]},
        {
            "role": "user",
            "content": "Keep the same ledger history and return exactly "
            + _EXPECTED_RESPONSES[2],
        },
    ]
    return t0, t1, t2


def _common_prefix_tokens(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    shared = 0
    for left, right in zip(first, second):
        if left != right:
            break
        shared += 1
    return shared


def _turn_event(
    *,
    study: ChunkedPrefillPerformanceStudy,
    case: SimpleNamespace,
    arm: str,
    lifetime_ordinal: int,
    turn: str,
    result: Mapping[str, Any],
    request_wall_s: float,
    before: Mapping[str, Any],
    before_polls: int,
    before_settled: bool,
    after: Mapping[str, Any],
    after_polls: int,
    after_settled: bool,
    append_only_prompt_identity_verified: bool,
    cross_lifetime_prompt_identity_verified: bool,
    shared_prefix_tokens: int,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event": SM121_CHUNKED_PREFILL_PERFORMANCE_TURN_EVENT,
        "arm": arm,
        "lifetime_ordinal": lifetime_ordinal,
        "case_id": case.case_id,
        "protocol_case_id": study.timed_case_id,
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
        "cross_lifetime_prompt_identity_verified": cross_lifetime_prompt_identity_verified,
        "response_detail_state": result["response_detail_state"],
        "usage_detail_state": result["usage_detail_state"],
        "response_device_cached_tokens": result["response_device_cached_tokens"],
        "response_host_cached_tokens": result["response_host_cached_tokens"],
        "response_storage_cached_tokens": result["response_storage_cached_tokens"],
        "usage_cached_tokens": result["usage_cached_tokens"],
        "metrics_available": (
            before.get("available") is True and after.get("available") is True
        ),
        "guardrail_metrics_available": (
            before.get("guardrail_metrics_available") is True
            and after.get("guardrail_metrics_available") is True
        ),
        "metrics_before_polls": before_polls,
        "metrics_after_polls": after_polls,
        "metrics_before_settled": before_settled,
        "metrics_after_settled": after_settled,
        "request_wall_s": request_wall_s,
        "timed_turn_admitted": False,
        "timed_turn_basis": "pending",
    }
    for prefix, snapshot in (("before", before), ("after", after)):
        for metric in SM121_CHUNKED_PREFILL_PERFORMANCE_METRIC_FIELDS:
            event[f"{prefix}_{metric}"] = snapshot[metric]
        for source in base_runner.SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
            event[f"{prefix}_cached_{source}_series_present"] = snapshot[
                f"cached_{source}_series_present"
            ]
    for metric in SM121_CHUNKED_PREFILL_PERFORMANCE_METRIC_FIELDS:
        event[f"delta_{metric}"] = event[f"after_{metric}"] - event[f"before_{metric}"]
    try:
        admitted, basis = derive_sm121_chunked_prefill_performance_turn_admission(event)
    except (KeyError, TypeError, ValueError, SM121ChunkedPrefillPerformanceError) as error:
        raise SM121ChunkedPrefillPerformanceRequestError() from error
    event["timed_turn_admitted"] = admitted
    event["timed_turn_basis"] = basis
    try:
        validate_sm121_chunked_prefill_performance_turn_event(event)
    except SM121ChunkedPrefillPerformanceError as error:
        raise SM121ChunkedPrefillPerformanceRequestError() from error
    return event


def _run_quality_lifetime(
    *,
    run_dir: Path,
    workspace: Path,
    model: SimpleNamespace,
    study: ChunkedPrefillPerformanceStudy,
    arm: str,
    lifetime_ordinal: int,
    case: SimpleNamespace,
    journal: Journal,
    telemetry: TelemetrySampler,
) -> float:
    """Run one isolated exact-answer quality lifetime and tear it down."""

    started = time.monotonic()
    deadline = started + SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S
    server = None
    watchdog: HostSafetyWatchdog | None = None
    terminal_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        _remaining_s(deadline)
        journal.append(
            _static_event(
                model=model,
                study=study,
                arm=arm,
                lifetime_ordinal=lifetime_ordinal,
            )
        )
        watchdog = base_runner._host_safety_watchdog(model)
        if watchdog is not None:
            watchdog.start()
        telemetry.set_phase(f"chunked_prefill_quality_start:{lifetime_ordinal}")
        callbacks: dict[str, Any] = {
            "abort_check": lambda: _abort_check(watchdog=watchdog, deadline=deadline)
        }
        if watchdog is not None:
            callbacks["on_server_created"] = (
                lambda created: watchdog.register_abort_callback(created.interrupt_owned)
            )
        server = base_runner.start_server(
            model,
            workspace=workspace,
            allow_download=False,
            server_log_path=(
                run_dir / "server" / f"lifetime-{lifetime_ordinal}" / "server.log"
            ),
            **callbacks,
        )
        _abort_check(watchdog=watchdog, deadline=deadline)
        journal.append(
            _runtime_event(
                server=server,
                study=study,
                arm=arm,
                lifetime_ordinal=lifetime_ordinal,
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
        if len(base_runner._QUALITY_ITEMS) != SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT:
            raise base_runner.PreflightError("SM121 chunked-prefill quality item count changed")
        journal.append(
            {
                "event": "sm121_chunked_prefill_performance_quality_case_start",
                "arm": arm,
                "lifetime_ordinal": lifetime_ordinal,
                "case_id": case.case_id,
            }
        )
        telemetry.set_phase(f"chunked_prefill_quality_case:{lifetime_ordinal}")
        for item in base_runner._QUALITY_ITEMS:
            _abort_check(watchdog=watchdog, deadline=deadline)
            request = base_runner._quality_request_arguments(
                server=server,
                model=model,
                case=case,
                item=item,
                request_id=uuid.uuid4().hex,
                prompt_tag="r0",
            )
            request["timeout_s"] = min(900.0, _remaining_s(deadline))
            result = base_runner.stream_chat_request(**request)
            if base_runner._validate_quality_item(item, result).get("passed") is not True:
                raise base_runner.SM121StorageQualityGateError()
            _abort_check(watchdog=watchdog, deadline=deadline)
        journal.append(
            {
                "event": "sm121_chunked_prefill_performance_quality_case_complete",
                "arm": arm,
                "lifetime_ordinal": lifetime_ordinal,
                "case_id": case.case_id,
                "quality_admitted": True,
                "item_count": len(base_runner._QUALITY_ITEMS),
            }
        )
    except BaseException as error:
        terminal_error = watchdog.failure if watchdog is not None and watchdog.failure else error
    finally:
        telemetry.set_phase(f"chunked_prefill_quality_stop:{lifetime_ordinal}")
        if server is not None:
            if watchdog is not None and watchdog.tripped:
                try:
                    base_runner._retry_host_safety_interrupt_if_needed(server, watchdog)
                    base_runner._record_host_safety_interrupt_failure(
                        journal, watchdog, stage="chunked_prefill_quality"
                    )
                except BaseException as error:
                    cleanup_error = error
            try:
                _interrupt_terminal_server(
                    server=server, deadline=deadline, terminal_error=terminal_error
                )
            except BaseException as error:
                cleanup_error = cleanup_error or error
            try:
                save_server_logs(
                    server,
                    run_dir / "server" / f"lifetime-{lifetime_ordinal}" / "server.log",
                )
            except BaseException as error:
                cleanup_error = cleanup_error or error
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
                cleanup_error = cleanup_error or error
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
    within_timeout = elapsed_s <= SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S
    journal.append(
        {
            "event": "sm121_chunked_prefill_performance_lifetime_complete",
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
                base_runner.SM121StorageQualityGateError,
                SM121ChunkedPrefillPerformanceRequestError,
            ),
        ):
            raise terminal_error
        raise SM121ChunkedPrefillPerformanceRequestError() from None
    if not within_timeout:
        raise SM121ChunkedPrefillPerformanceRequestError()
    return elapsed_s


def _timed_turn_prefix(
    *,
    journal: Journal,
    study: ChunkedPrefillPerformanceStudy,
    arm: str,
    campaign_ordinal: int,
) -> list[dict[str, Any]]:
    """Return only a validated, scalar prefix after a terminal timed failure."""

    events = [
        event
        for event in journal.events()
        if event.get("event") == SM121_CHUNKED_PREFILL_PERFORMANCE_TURN_EVENT
    ]
    if len(events) > len(SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS):
        return []
    prefix: list[dict[str, Any]] = []
    for index, (expected_turn, raw) in enumerate(
        zip(SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS, events, strict=False)
    ):
        scalar = {key: value for key, value in raw.items() if key != "timestamp"}
        try:
            validate_sm121_chunked_prefill_performance_turn_event(scalar)
        except SM121ChunkedPrefillPerformanceError:
            return []
        if (
            scalar["arm"] != arm
            or scalar["lifetime_ordinal"] != campaign_ordinal * 2
            or scalar["turn"] != expected_turn
            or scalar["protocol_case_id"] != study.timed_case_id
            or (
                index + 1 < len(events)
                and scalar["timed_turn_admitted"] is not True
            )
        ):
            return []
        prefix.append(scalar)
    return prefix


def _run_timed_lifetime(
    *,
    run_dir: Path,
    workspace: Path,
    model: SimpleNamespace,
    study: ChunkedPrefillPerformanceStudy,
    arm: str,
    lifetime_ordinal: int,
    case: SimpleNamespace,
    journal: Journal,
    telemetry: TelemetrySampler,
    reference_prompt_token_ids: tuple[tuple[int, ...], ...] | None,
) -> tuple[tuple[dict[str, Any], ...], tuple[tuple[int, ...], ...], float]:
    """Run cold T0 then append-only T1/T2 in one fresh server lifetime."""

    started = time.monotonic()
    deadline = started + SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S
    server = None
    watchdog: HostSafetyWatchdog | None = None
    terminal_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    private_ids: list[tuple[int, ...]] = []
    events: list[dict[str, Any]] = []
    try:
        _remaining_s(deadline)
        journal.append(
            _static_event(
                model=model,
                study=study,
                arm=arm,
                lifetime_ordinal=lifetime_ordinal,
            )
        )
        watchdog = base_runner._host_safety_watchdog(model)
        if watchdog is not None:
            watchdog.start()
        telemetry.set_phase(f"chunked_prefill_timed_start:{lifetime_ordinal}")
        callbacks: dict[str, Any] = {
            "abort_check": lambda: _abort_check(watchdog=watchdog, deadline=deadline)
        }
        if watchdog is not None:
            callbacks["on_server_created"] = (
                lambda created: watchdog.register_abort_callback(created.interrupt_owned)
            )
        server = base_runner.start_server(
            model,
            workspace=workspace,
            allow_download=False,
            server_log_path=(
                run_dir / "server" / f"lifetime-{lifetime_ordinal}" / "server.log"
            ),
            **callbacks,
        )
        _abort_check(watchdog=watchdog, deadline=deadline)
        journal.append(
            _runtime_event(
                server=server,
                study=study,
                arm=arm,
                lifetime_ordinal=lifetime_ordinal,
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
                "event": "sm121_chunked_prefill_performance_timed_case_start",
                "arm": arm,
                "lifetime_ordinal": lifetime_ordinal,
                "case_id": case.case_id,
            }
        )
        telemetry.set_phase(f"chunked_prefill_timed_case:{lifetime_ordinal}")
        for index, (turn, messages, expected_response) in enumerate(
            zip(
                SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS,
                _messages(),
                _EXPECTED_RESPONSES,
                strict=True,
            )
        ):
            _abort_check(watchdog=watchdog, deadline=deadline)
            before_timeout = min(45.0, _remaining_s(deadline))
            before, _wait, before_polls, before_settled = (
                settle_sm121_cache_observability_metrics(
                    server,
                    timeout_s=before_timeout,
                    poll_interval_s=min(1.0, max(0.001, before_timeout / 4)),
                    semantic_arm="A",
                )
            )
            _abort_check(watchdog=watchdog, deadline=deadline)
            request_started = time.perf_counter()
            result = request_sm121_cache_semantic_turn(
                server,
                served_name=model.served_name,
                messages=messages,
                expected_response=expected_response,
                max_tokens=int(case.max_output_tokens),
                timeout_s=min(900.0, _remaining_s(deadline)),
            )
            request_wall_s = time.perf_counter() - request_started
            _abort_check(watchdog=watchdog, deadline=deadline)
            after_timeout = min(45.0, _remaining_s(deadline))
            after, _wait, after_polls, after_settled = (
                settle_sm121_cache_observability_metrics(
                    server,
                    timeout_s=after_timeout,
                    poll_interval_s=min(1.0, max(0.001, after_timeout / 4)),
                    semantic_arm="A",
                )
            )
            _abort_check(watchdog=watchdog, deadline=deadline)
            prompt_ids = result.pop("private_prompt_token_ids", None)
            if (
                not isinstance(prompt_ids, tuple)
                or not prompt_ids
                or any(type(token) is not int or token < 0 for token in prompt_ids)
            ):
                raise SM121ChunkedPrefillPerformanceRequestError()
            if index == 0:
                shared = 0
                append_verified = True
            else:
                shared = _common_prefix_tokens(private_ids[-1], prompt_ids)
                append_verified = (
                    shared >= SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MIN_TOKENS
                )
            if reference_prompt_token_ids is None:
                if lifetime_ordinal != 2 or arm != "A":
                    raise SM121ChunkedPrefillPerformanceRequestError()
                cross_lifetime_verified = True
            else:
                if prompt_ids != reference_prompt_token_ids[index]:
                    raise SM121ChunkedPrefillPerformanceRequestError()
                cross_lifetime_verified = True
            event = _turn_event(
                study=study,
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
                shared_prefix_tokens=shared,
            )
            journal.append(event)
            if event["timed_turn_admitted"] is not True:
                raise SM121ChunkedPrefillPerformanceRequestError()
            events.append(event)
            private_ids.append(prompt_ids)
        journal.append(
            {
                "event": "sm121_chunked_prefill_performance_timed_case_complete",
                "arm": arm,
                "lifetime_ordinal": lifetime_ordinal,
                "case_id": case.case_id,
                "timed_admitted": True,
            }
        )
        _abort_check(watchdog=watchdog, deadline=deadline)
    except BaseException as error:
        terminal_error = watchdog.failure if watchdog is not None and watchdog.failure else error
    finally:
        telemetry.set_phase(f"chunked_prefill_timed_stop:{lifetime_ordinal}")
        if server is not None:
            if watchdog is not None and watchdog.tripped:
                try:
                    base_runner._retry_host_safety_interrupt_if_needed(server, watchdog)
                    base_runner._record_host_safety_interrupt_failure(
                        journal, watchdog, stage="chunked_prefill_timed"
                    )
                except BaseException as error:
                    cleanup_error = error
            try:
                _interrupt_terminal_server(
                    server=server, deadline=deadline, terminal_error=terminal_error
                )
            except BaseException as error:
                cleanup_error = cleanup_error or error
            try:
                save_server_logs(
                    server,
                    run_dir / "server" / f"lifetime-{lifetime_ordinal}" / "server.log",
                )
            except BaseException as error:
                cleanup_error = cleanup_error or error
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
                cleanup_error = cleanup_error or error
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
    within_timeout = elapsed_s <= SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S
    journal.append(
        {
            "event": "sm121_chunked_prefill_performance_lifetime_complete",
            "arm": arm,
            "lifetime_ordinal": lifetime_ordinal,
            "phase": "timed",
            "lifetime_wall_s": elapsed_s,
            "within_timeout": within_timeout,
            "admitted": terminal_error is None and within_timeout,
        }
    )
    if terminal_error is not None:
        if isinstance(terminal_error, (HostSafetyError, SM121ChunkedPrefillPerformanceRequestError)):
            raise terminal_error
        raise SM121ChunkedPrefillPerformanceRequestError() from None
    if not within_timeout:
        raise SM121ChunkedPrefillPerformanceRequestError()
    return tuple(events), tuple(private_ids), elapsed_s


def _load_campaign(
    campaign_dir: Path,
) -> tuple[
    dict[str, Any],
    ChunkedPrefillPerformanceStudy,
    list[tuple[Path, dict[str, Any], SimpleNamespace, SimpleNamespace]],
]:
    """Load only one complete, untouched frozen A/B/B/A campaign."""

    try:
        root = campaign_dir.resolve(strict=True)
        campaign = json.loads((root / "campaign.json").read_text())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise base_runner.PreflightError("SM121 chunked-prefill campaign is unavailable") from error
    base_fields = {
        "schema_version",
        "campaign_id",
        "created_at",
        "execution_mode",
        "pair_binding",
        "run_directories",
        "integrity_hash",
    }
    if type(campaign) is not dict:
        raise base_runner.PreflightError("SM121 chunked-prefill campaign fields are invalid")
    try:
        study = sm121_chunked_prefill_performance_study(campaign.get("campaign_id"))
    except SM121ChunkedPrefillPerformanceError as error:
        raise base_runner.PreflightError(
            "SM121 chunked-prefill campaign contract is invalid"
        ) from error
    expected_fields = set(base_fields)
    if study == SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY:
        expected_fields.add("v3_admission_receipt")
    if set(campaign) != expected_fields:
        raise base_runner.PreflightError("SM121 chunked-prefill campaign fields are invalid")
    integrity = campaign.get("integrity_hash")
    if not isinstance(integrity, str) or content_hash(
        {key: value for key, value in campaign.items() if key != "integrity_hash"},
        len(integrity),
    ) != integrity:
        raise base_runner.PreflightError("SM121 chunked-prefill campaign integrity is invalid")
    if study == SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY:
        try:
            validate_sm121_chunked_prefill_8k_admission_receipt(
                campaign.get("v3_admission_receipt")
            )
        except SM121ChunkedPrefill8KAdmissionError as error:
            raise base_runner.PreflightError(
                "SM121 chunked-prefill v3 admission receipt is invalid"
            ) from error
    if (
        campaign.get("schema_version") != 1
        or campaign.get("campaign_id") != study.campaign_id
        or campaign.get("execution_mode") != study.execution_mode
        or not isinstance(campaign.get("created_at"), str)
    ):
        raise base_runner.PreflightError("SM121 chunked-prefill campaign contract is invalid")
    binding = campaign.get("pair_binding")
    try:
        validate_sm121_chunked_prefill_performance_pair_binding(binding)
    except SM121ChunkedPrefillPerformanceError as error:
        raise base_runner.PreflightError("SM121 chunked-prefill admission is unavailable") from error
    if not isinstance(binding, dict):
        raise base_runner.PreflightError("SM121 chunked-prefill pair binding is invalid")
    if binding.get("suite_id") != study.suite_id:
        raise base_runner.PreflightError("SM121 chunked-prefill campaign study changed")
    names = campaign.get("run_directories")
    if (
        type(names) is not list
        or len(names) != len(SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER)
        or len(set(names)) != len(names)
        or any(
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", name) is None
            for name in names
        )
    ):
        raise base_runner.PreflightError("SM121 chunked-prefill run topology is invalid")
    try:
        runs_root = (root / "runs").resolve(strict=True)
        runs_root.relative_to(root)
    except (OSError, ValueError) as error:
        raise base_runner.PreflightError("SM121 chunked-prefill runs escape campaign") from error
    loaded: list[tuple[Path, dict[str, Any], SimpleNamespace, SimpleNamespace]] = []
    for ordinal, (name, arm) in enumerate(
        zip(names, SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER, strict=True), start=1
    ):
        run_dir = runs_root / name
        try:
            resolved_run_dir = run_dir.resolve(strict=True)
            resolved_run_dir.relative_to(runs_root)
        except (OSError, ValueError) as error:
            raise base_runner.PreflightError(
                "SM121 chunked-prefill run directory is invalid"
            ) from error
        if resolved_run_dir.parent != runs_root or run_dir.is_symlink():
            raise base_runner.PreflightError("SM121 chunked-prefill run directory is invalid")
        plan, model, suite, plan_study = _load_plan(resolved_run_dir)
        if (
            plan.get("chunked_prefill_performance_ordinal") != ordinal
            or sm121_chunked_prefill_performance_arm(model) != arm
            or plan.get("chunked_prefill_performance_pair") != binding
            or plan_study != study
        ):
            raise base_runner.PreflightError("SM121 chunked-prefill run binding moved")
        loaded.append((resolved_run_dir, plan, model, suite))
    fingerprints = [plan["fingerprint"] for _path, plan, _model, _suite in loaded]
    nonces = [plan["run_nonce"] for _path, plan, _model, _suite in loaded]
    try:
        instance = sm121_chunked_prefill_performance_pair_instance_sha256(nonces)
    except SM121ChunkedPrefillPerformanceError as error:
        raise base_runner.PreflightError("SM121 chunked-prefill run nonce is invalid") from error
    if (
        binding.get("campaign_instance_sha256") != instance
        or binding.get("plan_fingerprints") != fingerprints
        or binding.get("pair_binding_sha256")
        != sm121_chunked_prefill_performance_pair_binding_sha256(binding)
    ):
        raise base_runner.PreflightError("SM121 chunked-prefill binding is invalid")
    if study == SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY:
        receipt = campaign.get("v3_admission_receipt")
        if (
            not isinstance(receipt, dict)
            or binding.get("admission_receipt_sha256")
            != receipt.get("receipt_integrity_hash")
        ):
            raise base_runner.PreflightError(
                "SM121 chunked-prefill V3 admission binding is invalid"
            )
        try:
            for _run_dir, plan, model, _suite in loaded:
                if sm121_chunked_prefill_performance_arm(model) == "B":
                    validate_sm121_chunked_prefill_8k_admission_receipt_for_v3_candidate_plan(
                        receipt, plan
                    )
        except SM121ChunkedPrefill8KAdmissionError as error:
            raise base_runner.PreflightError(
                "SM121 chunked-prefill V3 admission binding is invalid"
            ) from error
    return campaign, study, loaded


def _execute_arm(
    *,
    run_dir: Path,
    plan: dict[str, Any],
    model: SimpleNamespace,
    suite: SimpleNamespace,
    study: ChunkedPrefillPerformanceStudy,
    campaign_ordinal: int,
    workspace: Path,
    reference_prompt_token_ids: tuple[tuple[int, ...], ...] | None,
) -> tuple[dict[str, Any], tuple[tuple[int, ...], ...] | None, bool]:
    """Execute one campaign arm as separate quality and timed lifetimes."""

    journal = Journal(run_dir / "events.jsonl")
    if journal.events():
        raise base_runner.PreflightError(
            "SM121 chunked-prefill campaign is non-resumable; freeze a new campaign"
        )
    arm = sm121_chunked_prefill_performance_arm(model)
    quality_case, timed_case = _case_pair(suite, study=study)
    if set(quality_case.requires) - set(model.tasks) or set(timed_case.requires) - set(model.tasks):
        raise base_runner.PreflightError("SM121 chunked-prefill case capabilities are invalid")
    if (
        SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MAX_TOKENS
        + int(timed_case.max_output_tokens)
        + 1024
        > int(model.max_context)
    ):
        raise base_runner.PreflightError("SM121 chunked-prefill context admission is insufficient")
    binding = getattr(model, "chunked_prefill_performance_pair", None)
    if not isinstance(binding, dict):
        raise base_runner.PreflightError("SM121 chunked-prefill pair binding is unavailable")
    journal.append(
        {
            "event": "run_start",
            "execution_mode": study.execution_mode,
            "arm": arm,
            "campaign_ordinal": campaign_ordinal,
            "plan_fingerprint": str(plan["fingerprint"]),
            "chunked_prefill_performance_pair_binding_sha256": binding[
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
        base_runner._preflight(model)
        stage = "quality_lifetime"
        quality_elapsed = _run_quality_lifetime(
            run_dir=run_dir,
            workspace=workspace,
            model=model,
            study=study,
            arm=arm,
            lifetime_ordinal=campaign_ordinal * 2 - 1,
            case=quality_case,
            journal=journal,
            telemetry=telemetry,
        )
        quality_admitted = True
        stage = "timed_lifetime"
        timed_events, private_ids, timed_elapsed = _run_timed_lifetime(
            run_dir=run_dir,
            workspace=workspace,
            model=model,
            study=study,
            arm=arm,
            lifetime_ordinal=campaign_ordinal * 2,
            case=timed_case,
            journal=journal,
            telemetry=telemetry,
            reference_prompt_token_ids=reference_prompt_token_ids,
        )
        turns = list(timed_events)
        timed_admitted = all(event["timed_turn_admitted"] is True for event in turns)
        if not timed_admitted or len(private_ids) != len(SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS):
            raise SM121ChunkedPrefillPerformanceRequestError()
        if reference_prompt_token_ids is None:
            next_reference = private_ids
        within_timeout = (
            quality_elapsed <= SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S
            and timed_elapsed <= SM121_CHUNKED_PREFILL_PERFORMANCE_CELL_TIMEOUT_S
        )
        if not within_timeout:
            raise SM121ChunkedPrefillPerformanceRequestError()
    except BaseException as error:
        if isinstance(
            error,
            (
                HostSafetyError,
                base_runner.PreflightError,
                base_runner.SM121StorageQualityGateError,
                SM121ChunkedPrefillPerformanceRequestError,
            ),
        ):
            safe_error = error
        else:
            safe_error = SM121ChunkedPrefillPerformanceRequestError()
        if isinstance(safe_error, HostSafetyError):
            base_runner._record_host_safety_breach(journal, safe_error, stage=stage)
        if stage == "timed_lifetime":
            turns = _timed_turn_prefix(
                journal=journal,
                study=study,
                arm=arm,
                campaign_ordinal=campaign_ordinal,
            )
        base_runner._record_run_aborted(journal, safe_error, stage=stage)
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


def _unstarted_lifetime(*, ordinal: int, arm: str) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "arm": arm,
        "quality_admitted": False,
        "timed_admitted": False,
        "within_timeout": False,
        "turns": [],
    }


def execute_sm121_chunked_prefill_performance_campaign(
    campaign_dir: Path,
    *,
    workspace: Path,
    admission_run_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one frozen non-resumable A/B/B/A chunk-size measurement."""

    lock_path = base_runner.results_lock_path(workspace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another SparkBench run holds the benchmark lock") from error
        campaign, study, loaded = _load_campaign(campaign_dir)
        if study == SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY:
            if admission_run_dir is None:
                raise base_runner.PreflightError(
                    "SM121 chunked-prefill v3 requires a verified 8K admission receipt"
                )
            try:
                _require_private_v3_directory(campaign_dir, create=False)
                _require_private_v3_directory(campaign_dir / "runs", create=False)
                _require_private_v3_regular_file(
                    campaign_dir / "campaign.json", harden=False
                )
                for run_dir, _plan, _model, _suite in loaded:
                    _require_private_v3_run_directory(run_dir, harden=False)
                current_receipt = load_verified_sm121_chunked_prefill_8k_admission_receipt(
                    admission_run_dir
                )
            except (RuntimeError, base_runner.PreflightError) as error:
                raise base_runner.PreflightError(
                    "SM121 chunked-prefill v3 admission receipt is invalid"
                ) from error
            if current_receipt != campaign.get("v3_admission_receipt"):
                raise base_runner.PreflightError(
                    "SM121 chunked-prefill v3 admission receipt changed"
                )
        elif admission_run_dir is not None:
            raise base_runner.PreflightError(
                "SM121 chunked-prefill admission receipt is only valid for v3"
            )
        if (campaign_dir / "summary.json").exists():
            raise base_runner.PreflightError(
                "SM121 chunked-prefill campaign is terminal; freeze a new campaign"
            )
        for run_dir, _plan, _model, _suite in loaded:
            if Journal(run_dir / "events.jsonl").events():
                raise base_runner.PreflightError(
                    "SM121 chunked-prefill campaign is non-resumable; freeze a new campaign"
                )
        lifetimes: list[dict[str, Any]] = []
        reference_prompt_token_ids: tuple[tuple[int, ...], ...] | None = None
        terminal = False
        for ordinal, (run_dir, plan, model, suite) in enumerate(loaded, start=1):
            arm = SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER[ordinal - 1]
            if terminal:
                lifetimes.append(_unstarted_lifetime(ordinal=ordinal, arm=arm))
                continue
            lifetime, reference_prompt_token_ids, completed = _execute_arm(
                run_dir=run_dir,
                plan=plan,
                model=model,
                suite=suite,
                study=study,
                campaign_ordinal=ordinal,
                workspace=workspace,
                reference_prompt_token_ids=reference_prompt_token_ids,
            )
            lifetimes.append(lifetime)
            terminal = not completed
        try:
            score = score_sm121_chunked_prefill_performance_campaign(
                lifetimes, study=study
            )
        except SM121ChunkedPrefillPerformanceError as error:
            raise base_runner.PreflightError("SM121 chunked-prefill score is invalid") from error
        summary = {
            "schema_version": 1,
            "campaign_id": study.campaign_id,
            "execution_mode": study.execution_mode,
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
        if study == SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY:
            try:
                _require_private_v3_regular_file(campaign_dir / "summary.json")
            except RuntimeError as error:
                raise base_runner.PreflightError(
                    "SM121 chunked-prefill V3 output is not private"
                ) from error
        return summary
