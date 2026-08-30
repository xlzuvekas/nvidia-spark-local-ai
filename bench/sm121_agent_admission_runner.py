"""Private, non-evidence C1 controller and audit for the SM121 agent profile.

The controller accepts only one authenticated frozen C1 plan and owns its
three fresh server lifetimes, direct client, runtime checks, cache receipt,
and cleanup.  It writes scalar-only private records under ``logs/`` and never
uses generic/resumable execution or caller-provided request hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import fcntl
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

from . import runner as base_runner
from .host_safety import HostSafetyError, HostSafetyWatchdog
from .journal import Journal, content_hash, write_json
from .runtime import (
    RuntimeErrorWithContext,
    inspect_sm121_agent_admission_runtime_identity,
    inspect_sm121_cache_source_digests,
    recover_owned_sglang,
    require_sm121_agent_admission_clean_start,
    settle_sm121_agent_admission_metrics,
    sm121_agent_admission_target_snapshot,
    start_sm121_agent_admission_server,
)
from .sm121_agent_admission_client import (
    SM121AgentAdmissionRequestError,
    create_sm121_agent_admission_client,
    validate_c1_payload_diagnostics,
)
from .sglang_sm121_agent_admission import (
    SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_BUDGET_PROBE_ID,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_BUDGET_PROBE_SCHEMA_VERSION,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_BUDGET_TOKENS,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_PROMPT_TOKENS,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_OUTPUT_TOKENS,
    SM121_AGENT_ADMISSION_NATIVE_CACHE_METRIC_FIELDS,
    SM121_AGENT_ADMISSION_PROFILE_ID,
    SM121_AGENT_ADMISSION_QUALITY_CASE_ID,
    SM121_AGENT_ADMISSION_RUNTIME_EXPECTED,
    SM121_AGENT_ADMISSION_SUITE_ID,
    SM121_AGENT_ADMISSION_TOOL_CASE_IDS,
    SM121AgentAdmissionError,
    probe_sm121_agent_parser_static_preflight,
    probe_sm121_agent_long_context_budget_preflight,
    validate_sm121_agent_admission_candidate,
    validate_sm121_agent_native_cache_metrics_receipt,
    validate_sm121_agent_admission_suite,
    validate_sm121_agent_long_context_budget_probe,
    validate_sm121_agent_parser_static_probe,
)
from .sglang_sm121_cache_observability import SM121_CACHE_SOURCE_DIGESTS
from .sglang_sm121_cache_semantic import SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS
from .sglang_sm121_storage import (
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_SOURCE_TREE,
)


SM121_AGENT_ADMISSION_ID = "qwen38-flash-next-sm121-agent-c1-admission-v1"
SM121_AGENT_ADMISSION_EXECUTION_MODE = "private_non_evidence_c1_admission"
SM121_AGENT_ADMISSION_LIFETIME_COUNT = 3
SM121_AGENT_ADMISSION_LIFETIME_TIMEOUT_S = 2_700.0
SM121_AGENT_ADMISSION_METRICS_SETTLE_TIMEOUT_S = 45.0
SM121_AGENT_ADMISSION_METRICS_POLL_INTERVAL_S = 1.0
SM121_AGENT_ADMISSION_FAILURE_CODES = frozenset(
    {
        "cleanup",
        "dependency_unavailable",
        "generic",
        "host_safety",
        "long_context",
        "preflight",
        "quality",
        "runtime_identity",
        "static_parser",
        "timeout",
        "tool",
    }
)

_ROOT = Path(__file__).resolve().parents[1]
_LOGS_ROOT = _ROOT / "logs"
_JSON_MAX_BYTES = 8 * 1024 * 1024
_RUN_LABEL = "agent-c1-admission"
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_FINGERPRINT_HEX = re.compile(r"[0-9a-f]{16}")
_RUN_NONCE_HEX = re.compile(r"[0-9a-f]{32}")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+00:00")
_PHASES = ("quality", "tools", "long_context")
_TERMINAL_STAGES = frozenset(
    {
        "parser_static",
        "quality_lifetime",
        "tool_lifetime",
        "long_context_lifetime",
        "record_audit",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "completion",
        "content",
        "container_id",
        "message",
        "messages",
        "prompt",
        "prompt_token_ids",
        "reasoning",
        "request_id",
        "response",
        "tool_arguments",
        "tool_calls",
        "tool_payload",
        "token_ids",
        "wall_s",
    }
)

@dataclass(frozen=True, eq=False)
class _SM121AgentAdmissionControllerSession:
    """Opaque execution context registered only by the C1 controller."""


@dataclass(frozen=True, eq=False)
class _SM121AgentAdmissionLaunchLease:
    """Opaque one-shot lease; its authority lives exclusively in the registry."""


@dataclass
class _SM121AgentAdmissionControllerRegistration:
    session: _SM121AgentAdmissionControllerSession
    model: object = field(repr=False)
    fingerprint: str
    integrity_hash: str
    run_nonce: str
    run_identity: str
    local_image_id: str
    active: bool = True
    issued_lifetimes: set[int] = field(default_factory=set, repr=False)
    active_lease_id: int | None = field(default=None, repr=False)


@dataclass
class _SM121AgentAdmissionLeaseRegistration:
    lease: _SM121AgentAdmissionLaunchLease
    controller: _SM121AgentAdmissionControllerRegistration = field(repr=False)
    fresh_lifetime: int
    consumed: bool = False


# C1 launch authority is deliberately a process-local, identity-registered
# capability. A model-shaped object, a freshly constructed private dataclass,
# or a binding returned by the frozen-plan reader cannot enter this registry.
# The controller closes its session after cleanup, revoking every lease.
_SM121_AGENT_ADMISSION_REGISTRY_LOCK = threading.RLock()
_SM121_AGENT_ADMISSION_CONTROLLERS: dict[
    int, _SM121AgentAdmissionControllerRegistration
] = {}
_SM121_AGENT_ADMISSION_LEASES: dict[
    int, _SM121AgentAdmissionLeaseRegistration
] = {}


def _controller_registration(
    session: object,
) -> _SM121AgentAdmissionControllerRegistration:
    if type(session) is not _SM121AgentAdmissionControllerSession:
        raise SM121AgentAdmissionError("SM121 agent admission launch is not authorized")
    registration = _SM121_AGENT_ADMISSION_CONTROLLERS.get(id(session))
    if (
        registration is None
        or registration.session is not session
        or registration.active is not True
    ):
        raise SM121AgentAdmissionError("SM121 agent admission launch is not authorized")
    return registration


def _registered_model_matches(
    model: object,
    registration: _SM121AgentAdmissionControllerRegistration,
) -> bool:
    return (
        registration.model is model
        and getattr(model, "id", None) == SM121_AGENT_ADMISSION_PROFILE_ID
        and getattr(model, "run_identity", None) == registration.run_identity
        and getattr(model, "resolved_local_image_id", None)
        == registration.local_image_id
        and registration.run_identity
        == f"{registration.fingerprint}-{registration.run_nonce}"
        and registration.local_image_id == SM121_STORAGE_LOCAL_IMAGE_ID
        and _FINGERPRINT_HEX.fullmatch(registration.fingerprint) is not None
        and _SHA256_HEX.fullmatch(registration.integrity_hash) is not None
        and _RUN_NONCE_HEX.fullmatch(registration.run_nonce) is not None
    )


def _issue_sm121_agent_admission_launch_lease(
    model: object,
    session: object,
    *,
    fresh_lifetime: int,
) -> _SM121AgentAdmissionLaunchLease:
    """Issue one fresh launch lease inside an already registered controller."""

    with _SM121_AGENT_ADMISSION_REGISTRY_LOCK:
        registration = _controller_registration(session)
        validate_sm121_agent_admission_candidate(model)
        if (
            not _registered_model_matches(model, registration)
            or type(fresh_lifetime) is not int
            or not 1 <= fresh_lifetime <= SM121_AGENT_ADMISSION_LIFETIME_COUNT
            or fresh_lifetime in registration.issued_lifetimes
            or registration.active_lease_id is not None
        ):
            raise SM121AgentAdmissionError("SM121 agent admission launch is not authorized")
        registration.issued_lifetimes.add(fresh_lifetime)
        lease = _SM121AgentAdmissionLaunchLease()
        _SM121_AGENT_ADMISSION_LEASES[id(lease)] = (
            _SM121AgentAdmissionLeaseRegistration(
                lease=lease,
                controller=registration,
                fresh_lifetime=fresh_lifetime,
            )
        )
        registration.active_lease_id = id(lease)
    return lease


def _require_sm121_agent_admission_plan_binding(
    model: object, binding: object
) -> _SM121AgentAdmissionLaunchLease:
    """Validate an unconsumed lease immediately before a C1 launch step."""

    if type(binding) is not _SM121AgentAdmissionLaunchLease:
        raise SM121AgentAdmissionError("SM121 agent admission launch is not authorized")
    with _SM121_AGENT_ADMISSION_REGISTRY_LOCK:
        lease = _SM121_AGENT_ADMISSION_LEASES.get(id(binding))
        if (
            lease is None
            or lease.lease is not binding
            or lease.consumed
            or not 1 <= lease.fresh_lifetime <= SM121_AGENT_ADMISSION_LIFETIME_COUNT
            or lease.fresh_lifetime not in lease.controller.issued_lifetimes
            or lease.controller.active_lease_id != id(binding)
        ):
            raise SM121AgentAdmissionError("SM121 agent admission launch is not authorized")
        registration = _controller_registration(lease.controller.session)
        validate_sm121_agent_admission_candidate(model)
        if lease.controller is not registration or not _registered_model_matches(
            model, registration
        ):
            raise SM121AgentAdmissionError("SM121 agent admission launch is not authorized")
    return binding


def _consume_sm121_agent_admission_plan_binding(
    model: object, binding: object
) -> _SM121AgentAdmissionLaunchLease:
    """Consume a lease just before the Docker launch side effect."""

    lease = _require_sm121_agent_admission_plan_binding(model, binding)
    with _SM121_AGENT_ADMISSION_REGISTRY_LOCK:
        registration = _SM121_AGENT_ADMISSION_LEASES.get(id(lease))
        if registration is None or registration.lease is not lease or registration.consumed:
            raise SM121AgentAdmissionError("SM121 agent admission launch is not authorized")
        registration.consumed = True
    return lease


def _revoke_sm121_agent_admission_launch_lease(binding: object) -> None:
    """Drop a lease after its lifetime, whether or not it reached Docker."""

    if type(binding) is not _SM121AgentAdmissionLaunchLease:
        return
    with _SM121_AGENT_ADMISSION_REGISTRY_LOCK:
        registration = _SM121_AGENT_ADMISSION_LEASES.get(id(binding))
        if registration is not None and registration.lease is binding:
            if registration.controller.active_lease_id == id(binding):
                registration.controller.active_lease_id = None
            del _SM121_AGENT_ADMISSION_LEASES[id(binding)]


def _close_sm121_agent_admission_controller(session: object) -> None:
    """Revoke an execution session and every remaining one-shot lease."""

    if type(session) is not _SM121AgentAdmissionControllerSession:
        return
    with _SM121_AGENT_ADMISSION_REGISTRY_LOCK:
        registration = _SM121_AGENT_ADMISSION_CONTROLLERS.get(id(session))
        if registration is None or registration.session is not session:
            return
        registration.active = False
        for lease_id, lease in tuple(_SM121_AGENT_ADMISSION_LEASES.items()):
            if lease.controller is registration:
                del _SM121_AGENT_ADMISSION_LEASES[lease_id]
        del _SM121_AGENT_ADMISSION_CONTROLLERS[id(session)]


def _scalar_static_assertions() -> dict[str, object]:
    """Project shared static facts into names safe for durable C1 records.

    The general cache contract calls one boolean ``prompt_token_ids_available``.
    It is only a capability flag, but this controller must not retain anything
    that even suggests token identifiers.  Keep the fact while making clear
    that no identifiers were written.
    """

    value = dict(SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS)
    available = value.pop("prompt_token_ids_available", None)
    if available is not True:
        raise SM121AgentAdmissionError("SM121 scalar static assertion is invalid")
    value["input_tokenization_available"] = True
    return value


_SCALAR_STATIC_ASSERTIONS = _scalar_static_assertions()


def _path(
    path: Path,
    *,
    existing: bool,
    allow_logs_root: bool,
    create_logs_root: bool,
) -> Path:
    """Return a non-symlink descendant of the repository's ignored logs tree."""

    if _LOGS_ROOT.is_symlink():
        raise base_runner.PreflightError("SM121 agent admission logs topology is invalid")
    if not _LOGS_ROOT.exists():
        if not create_logs_root:
            raise base_runner.PreflightError("SM121 agent admission logs are unavailable")
        _LOGS_ROOT.mkdir(mode=0o700)
    if not _LOGS_ROOT.is_dir():
        raise base_runner.PreflightError("SM121 agent admission logs topology is invalid")
    candidate = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = candidate.relative_to(_LOGS_ROOT)
    except ValueError as error:
        raise base_runner.PreflightError(
            "SM121 agent admission output must remain under ignored logs"
        ) from error
    if candidate == _LOGS_ROOT and not allow_logs_root:
        raise base_runner.PreflightError("SM121 agent admission location is invalid")
    cursor = _LOGS_ROOT
    for index, part in enumerate(relative.parts):
        cursor = cursor / part
        if cursor.is_symlink() or (
            cursor.exists() and index + 1 < len(relative.parts) and not cursor.is_dir()
        ):
            raise base_runner.PreflightError("SM121 agent admission logs topology is invalid")
    if existing and (candidate.is_symlink() or not candidate.is_dir()):
        raise base_runner.PreflightError("SM121 agent admission location is invalid")
    if not existing and candidate.is_symlink():
        raise base_runner.PreflightError("SM121 agent admission location is invalid")
    return candidate


def _owned_file(path: Path) -> bool:
    try:
        item = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(item.st_mode)
        and item.st_nlink == 1
        and item.st_uid == os.geteuid()
    )


def _owned_directory(path: Path) -> bool:
    try:
        item = path.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(item.st_mode) and item.st_uid == os.geteuid()


def _private_directory(path: Path, *, create: bool) -> None:
    if path.exists() or path.is_symlink():
        if not _owned_directory(path) or stat.S_IMODE(path.lstat().st_mode) & 0o077:
            raise base_runner.PreflightError("SM121 agent admission directory is not private")
        return
    if not create:
        raise base_runner.PreflightError("SM121 agent admission directory is unavailable")
    try:
        path.mkdir(parents=True, mode=0o700)
        os.chmod(path, 0o700)
    except OSError as error:
        raise base_runner.PreflightError("SM121 agent admission directory is unavailable") from error
    if not _owned_directory(path) or stat.S_IMODE(path.lstat().st_mode) != 0o700:
        raise base_runner.PreflightError("SM121 agent admission directory is not private")


def _private_run_directory(path: Path) -> None:
    if not _owned_directory(path) or stat.S_IMODE(path.lstat().st_mode) != 0o700:
        raise base_runner.PreflightError("SM121 agent admission run is not private")


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    """Read a bounded owned object without link traversal or duplicate keys."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise base_runner.PreflightError(f"{context} requires no-follow support")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | nofollow)
    except OSError as error:
        raise base_runner.PreflightError(f"{context} is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or before.st_size < 0
            or before.st_size > _JSON_MAX_BYTES
        ):
            raise base_runner.PreflightError(f"{context} topology is invalid")
        payload = os.pread(descriptor, before.st_size, 0)
        after = os.fstat(descriptor)
        if (
            len(payload) != before.st_size
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        ):
            raise base_runner.PreflightError(f"{context} changed while being read")
    finally:
        os.close(descriptor)

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise base_runner.PreflightError(f"{context} is invalid") from error
    if type(value) is not dict:
        raise base_runner.PreflightError(f"{context} is invalid")
    return value


def _require_safe_tree(root: Path) -> None:
    if not all(_owned_file(root / name) for name in ("plan.json", "inventory.json")):
        raise base_runner.PreflightError("SM121 agent admission plan inputs are invalid")
    for name in ("events.jsonl", "admission.json"):
        item = root / name
        if item.is_symlink() or (item.exists() and not _owned_file(item)):
            raise base_runner.PreflightError("SM121 agent admission artifacts are invalid")
    server_root = root / "server"
    if server_root.is_symlink() or (server_root.exists() and not _owned_directory(server_root)):
        raise base_runner.PreflightError("SM121 agent admission server topology is invalid")
    for lifetime in range(1, SM121_AGENT_ADMISSION_LIFETIME_COUNT + 1):
        lifetime_root = server_root / f"lifetime-{lifetime}"
        key = lifetime_root / "api-key"
        if lifetime_root.is_symlink() or (
            lifetime_root.exists() and not _owned_directory(lifetime_root)
        ):
            raise base_runner.PreflightError("SM121 agent admission server topology is invalid")
        if key.is_symlink() or (key.exists() and not _owned_file(key)):
            raise base_runner.PreflightError("SM121 agent admission server topology is invalid")


def _server_issues(root: Path) -> tuple[str, ...]:
    server_root = root / "server"
    try:
        item = server_root.lstat()
    except FileNotFoundError:
        return ()
    if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
        return ("server_topology",)
    issues: list[str] = []
    for lifetime in range(1, SM121_AGENT_ADMISSION_LIFETIME_COUNT + 1):
        lifetime_root = server_root / f"lifetime-{lifetime}"
        try:
            item = lifetime_root.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
            issues.append("server_topology")
        elif (lifetime_root / "api-key").exists() or (lifetime_root / "api-key").is_symlink():
            issues.append("api_key_residue")
    return tuple(issues)


def create_sm121_agent_admission_plan(
    *,
    model: Any,
    suite: Any,
    output_root: Path,
    models_path: Path,
    suite_path: Path,
) -> Path:
    """Freeze one exact private C1 plan; this never starts a server."""

    try:
        validate_sm121_agent_admission_candidate(model)
        validate_sm121_agent_admission_suite(suite)
    except SM121AgentAdmissionError as error:
        raise RuntimeError("SM121 agent admission is unavailable") from error
    target = _path(
        output_root, existing=False, allow_logs_root=False, create_logs_root=True
    )
    _private_directory(target, create=True)
    run_dir = base_runner._create_sm121_agent_admission_plan(
        model=model,
        suite=suite,
        results_root=target,
        models_path=models_path,
        suite_path=suite_path,
        run_label=_RUN_LABEL,
    )
    os.chmod(run_dir, 0o700)
    _private_run_directory(run_dir)
    return run_dir


def _load_plan(
    root: Path,
) -> tuple[dict[str, Any], SimpleNamespace, SimpleNamespace]:
    plan = _read_json(root / "plan.json", context="SM121 agent admission plan")
    if (
        type(plan) is not dict
        or type(plan.get("schema_version")) is not int
        or plan["schema_version"] != 2
    ):
        raise base_runner.PreflightError("SM121 agent admission plan schema is invalid")
    model_data, suite_data, resolved = (
        plan.get("model"),
        plan.get("suite"),
        plan.get("resolved"),
    )
    if type(model_data) is not dict or type(suite_data) is not dict or type(resolved) is not dict:
        raise base_runner.PreflightError("SM121 agent admission plan fields are invalid")
    integrity = plan.get("integrity_hash")
    if (
        not isinstance(integrity, str)
        or _SHA256_HEX.fullmatch(integrity) is None
        or content_hash(
            {key: value for key, value in plan.items() if key != "integrity_hash"},
            64,
        )
        != integrity
    ):
        raise base_runner.PreflightError("SM121 agent admission plan integrity is invalid")
    cases = suite_data.get("cases")
    if not isinstance(cases, list) or any(type(case) is not dict for case in cases):
        raise base_runner.PreflightError("SM121 agent admission cases are invalid")
    stripped_suite = {
        **suite_data,
        "cases": [{key: value for key, value in case.items() if key != "case_id"} for case in cases],
    }
    if plan.get("fingerprint") != content_hash(
        {"model": model_data, "suite": stripped_suite, "resolved": resolved}
    ):
        raise base_runner.PreflightError("SM121 agent admission plan fingerprint is invalid")
    for case in cases:
        raw = {key: value for key, value in case.items() if key != "case_id"}
        if case.get("case_id") != base_runner._canonical_case(
            model_data, raw, protocol_digest=suite_data.get("protocol_digest")
        )["case_id"]:
            raise base_runner.PreflightError("SM121 agent admission case identity is invalid")
    model = base_runner._namespace(model_data)
    suite = base_runner._namespace(suite_data)
    # Frozen JSON arrays round-trip as lists; the pinned profile contract uses
    # a tuple for argv so preserve its manifest representation before checking.
    if isinstance(getattr(model, "args", None), list):
        model.args = tuple(model.args)
    try:
        validate_sm121_agent_admission_candidate(model)
        validate_sm121_agent_admission_suite(suite)
    except SM121AgentAdmissionError as error:
        raise base_runner.PreflightError("SM121 agent admission plan contract is invalid") from error
    image = resolved.get("local_image")
    if (
        type(image) is not dict
        or image != {
            "docker_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
            "platform": SM121_STORAGE_PLATFORM,
            "source_tree": SM121_STORAGE_SOURCE_TREE,
        }
    ):
        raise base_runner.PreflightError("SM121 agent admission local image changed")
    nonce = plan.get("run_nonce")
    if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
        raise base_runner.PreflightError("SM121 agent admission run nonce is invalid")
    model.resolved_local_image_id = image["docker_image_id"]
    model.run_identity = f"{plan['fingerprint']}-{nonce}"
    fingerprint = plan["fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or _FINGERPRINT_HEX.fullmatch(fingerprint) is None
        or not isinstance(integrity, str)
        or _SHA256_HEX.fullmatch(integrity) is None
        or model.run_identity != f"{fingerprint}-{nonce}"
    ):
        raise base_runner.PreflightError("SM121 agent admission plan binding is invalid")
    # Frozen-plan loading authenticates data only. It must not return a launch
    # authority: execute registers a controller session after a clean-start
    # gate, then issues one fresh lease for each server lifetime.
    return plan, model, suite


def _runtime_expected() -> dict[str, object]:
    return dict(SM121_AGENT_ADMISSION_RUNTIME_EXPECTED)


def _payload_diagnostics(
    *, outbound: int, tools: int, cache_zero: int
) -> dict[str, int | bool]:
    return {
        "outbound_body_count": outbound,
        "validated_low_thinking_body_count": outbound,
        "validated_tool_body_count": tools,
        "validated_cache_zero_body_count": cache_zero,
        "transport_attempt_count": outbound,
        "transport_retry_count": 0,
        "payload_contract_verified": True,
    }


def _quality_gate_expected() -> dict[str, object]:
    return {
        "event": "sm121_agent_quality_gate",
        "fresh_lifetime": 1,
        "quality_item_count": 4,
        "quality_repetitions": 2,
        "completed_quality_turns": 8,
        "quality_admitted": True,
        "payload_diagnostics": _payload_diagnostics(
            outbound=8, tools=0, cache_zero=0
        ),
    }


def _tool_gate_expected() -> dict[str, object]:
    return {
        "event": "sm121_agent_tool_gate",
        "fresh_lifetime": 2,
        "scenario_count": 4,
        "scenario_repetitions": 3,
        "completed_episodes": {
            case_id: 3 for case_id in SM121_AGENT_ADMISSION_TOOL_CASE_IDS
        },
        "scenario_passes": {
            case_id: True for case_id in SM121_AGENT_ADMISSION_TOOL_CASE_IDS
        },
        "tools_admitted": True,
        "payload_diagnostics": _payload_diagnostics(
            outbound=27, tools=27, cache_zero=0
        ),
    }


def _long_context_gate_expected() -> dict[str, object]:
    return {
        "event": "sm121_agent_long_context_gate",
        "fresh_lifetime": 3,
        "input_tokenization_verified": True,
        "context_fit": True,
        "zero_metric_cache_hits": True,
        "zero_response_cache_hits": True,
        "guardrails_clean": True,
        "long_context_admitted": True,
        "payload_diagnostics": _payload_diagnostics(
            outbound=1, tools=1, cache_zero=1
        ),
    }


class SM121AgentAdmissionExecutionError(RuntimeError):
    """Fixed scalar failure for the dedicated C1 controller."""

    def __init__(self, failure_code: str = "generic") -> None:
        if failure_code not in SM121_AGENT_ADMISSION_FAILURE_CODES:
            failure_code = "generic"
        self.failure_code = failure_code
        super().__init__("SM121 agent admission execution failed")


def _remaining_s(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise SM121AgentAdmissionExecutionError("timeout")
    return remaining


def _abort_check(*, watchdog: HostSafetyWatchdog | None, deadline: float) -> None:
    if watchdog is not None:
        watchdog.raise_if_tripped()
    _remaining_s(deadline)


def _admission_cases(
    suite: SimpleNamespace,
) -> tuple[SimpleNamespace, tuple[SimpleNamespace, ...], SimpleNamespace]:
    cases = tuple(getattr(suite, "cases", ()))
    expected_ids = (
        SM121_AGENT_ADMISSION_QUALITY_CASE_ID,
        *SM121_AGENT_ADMISSION_TOOL_CASE_IDS,
        SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID,
    )
    if len(cases) != len(expected_ids) or tuple(
        getattr(case, "id", None) for case in cases
    ) != expected_ids:
        raise base_runner.PreflightError("SM121 agent admission cases are invalid")
    if any(not isinstance(getattr(case, "case_id", None), str) for case in cases):
        raise base_runner.PreflightError("SM121 agent admission cases are invalid")
    quality = cases[0]
    tools = cases[1:-1]
    long_context = cases[-1]
    return quality, tools, long_context


def _require_context_and_tasks(
    *,
    model: SimpleNamespace,
    quality: SimpleNamespace,
    tools: tuple[SimpleNamespace, ...],
    long_context: SimpleNamespace,
) -> None:
    available_tasks = set(getattr(model, "tasks", ()))
    for case in (quality, *tools, long_context):
        if set(getattr(case, "requires", ())) - available_tasks:
            raise base_runner.PreflightError("SM121 agent admission tasks are invalid")
    if (
        int(getattr(long_context, "prompt_repetitions", 0))
        + int(getattr(long_context, "max_output_tokens", 0))
        + 1024
        > int(getattr(model, "max_context", 0))
    ):
        raise base_runner.PreflightError("SM121 agent admission context is insufficient")


def _static_event(
    *, model: SimpleNamespace, fresh_lifetime: int, phase: str
) -> dict[str, object]:
    try:
        observed = inspect_sm121_cache_source_digests(model)
    except RuntimeErrorWithContext as error:
        raise SM121AgentAdmissionExecutionError("runtime_identity") from error
    if observed != SM121_CACHE_SOURCE_DIGESTS:
        raise SM121AgentAdmissionExecutionError("runtime_identity")
    return {
        "event": "sm121_agent_static_attestation",
        "fresh_lifetime": fresh_lifetime,
        "phase": phase,
        "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
        **observed,
        **_SCALAR_STATIC_ASSERTIONS,
    }


def _runtime_event(
    *, server: object, fresh_lifetime: int, phase: str
) -> dict[str, object]:
    try:
        observed = inspect_sm121_agent_admission_runtime_identity(server)  # type: ignore[arg-type]
    except RuntimeErrorWithContext as error:
        raise SM121AgentAdmissionExecutionError("runtime_identity") from error
    return {
        "event": "sm121_agent_runtime_attestation",
        "fresh_lifetime": fresh_lifetime,
        "phase": phase,
        **observed,
    }


def _combine_payload_diagnostics(values: list[dict[str, int | bool]]) -> dict[str, int | bool]:
    if not values:
        raise SM121AgentAdmissionExecutionError("generic")
    totals = {
        "outbound_body_count": 0,
        "validated_low_thinking_body_count": 0,
        "validated_tool_body_count": 0,
        "validated_cache_zero_body_count": 0,
        "transport_attempt_count": 0,
        "transport_retry_count": 0,
        "payload_contract_verified": True,
    }
    for value in values:
        try:
            checked = validate_c1_payload_diagnostics(value)
        except SM121AgentAdmissionRequestError as error:
            raise SM121AgentAdmissionExecutionError("generic") from error
        for field in totals:
            if field == "payload_contract_verified":
                totals[field] = bool(totals[field] and checked[field])
            else:
                totals[field] = int(totals[field]) + int(checked[field])
    try:
        return validate_c1_payload_diagnostics(totals)
    except SM121AgentAdmissionRequestError as error:
        raise SM121AgentAdmissionExecutionError("generic") from error


def _quality_gate(
    *,
    server: object,
    model: SimpleNamespace,
    case: SimpleNamespace,
    deadline: float,
    watchdog: HostSafetyWatchdog | None,
) -> dict[str, object]:
    repetitions = getattr(case, "repetitions", None)
    if type(repetitions) is not int or repetitions <= 0:
        raise SM121AgentAdmissionExecutionError("quality")
    diagnostics: list[dict[str, int | bool]] = []
    client = create_sm121_agent_admission_client(
        server=server,
        model=model,
        case_id=SM121_AGENT_ADMISSION_QUALITY_CASE_ID,
    )
    client._bind_controller_deadline(deadline)
    try:
        for _rep in range(repetitions):
            for item in base_runner._QUALITY_ITEMS:
                _abort_check(watchdog=watchdog, deadline=deadline)
                # Reuse the benchmark's exact-answer response grammar instead
                # of asking the bare question.  The prior bare-question form
                # allowed an otherwise correct explanation to fail the
                # synthetic exact-answer parser, so it could only yield a
                # partial C1 record.  The prompt remains in memory only.
                result = client.run_quality_turn(
                    prompt=base_runner._quality_prompt(item, f"c1-{_rep}")
                )
                if base_runner._validate_quality_item(item, result).get("passed") is not True:
                    raise SM121AgentAdmissionExecutionError("quality")
                _abort_check(watchdog=watchdog, deadline=deadline)
        diagnostics.append(client.diagnostics().to_dict())
    except SM121AgentAdmissionExecutionError:
        raise
    except Exception as error:
        raise SM121AgentAdmissionExecutionError("quality") from error
    payload = _combine_payload_diagnostics(diagnostics)
    expected_turns = len(base_runner._QUALITY_ITEMS) * repetitions
    if (
        payload["outbound_body_count"] != expected_turns
        or payload["validated_low_thinking_body_count"] != expected_turns
        or payload["validated_tool_body_count"] != 0
        or payload["validated_cache_zero_body_count"] != 0
        or payload["transport_attempt_count"] != expected_turns
        or payload["transport_retry_count"] != 0
        or payload["payload_contract_verified"] is not True
    ):
        raise SM121AgentAdmissionExecutionError("quality")
    return {
        "event": "sm121_agent_quality_gate",
        "fresh_lifetime": 1,
        "quality_item_count": len(base_runner._QUALITY_ITEMS),
        "quality_repetitions": repetitions,
        "completed_quality_turns": expected_turns,
        "quality_admitted": True,
        "payload_diagnostics": payload,
    }


def _tool_gate(
    *,
    server: object,
    model: SimpleNamespace,
    cases: tuple[SimpleNamespace, ...],
    deadline: float,
    watchdog: HostSafetyWatchdog | None,
) -> dict[str, object]:
    if tuple(getattr(case, "id", None) for case in cases) != SM121_AGENT_ADMISSION_TOOL_CASE_IDS:
        raise SM121AgentAdmissionExecutionError("tool")
    scenario_passes: dict[str, bool] = {}
    completed_episodes: dict[str, int] = {}
    diagnostics: list[dict[str, int | bool]] = []
    for case in cases:
        repetitions = getattr(case, "repetitions", None)
        if type(repetitions) is not int or repetitions <= 0:
            raise SM121AgentAdmissionExecutionError("tool")
        client = create_sm121_agent_admission_client(
            server=server,
            model=model,
            case_id=case.id,
        )
        client._bind_controller_deadline(deadline)
        completed = 0
        try:
            for _rep in range(repetitions):
                _abort_check(watchdog=watchdog, deadline=deadline)
                if client.run_agentic().passed is not True:
                    raise SM121AgentAdmissionExecutionError("tool")
                completed += 1
                _abort_check(watchdog=watchdog, deadline=deadline)
        except SM121AgentAdmissionExecutionError:
            raise
        except Exception as error:
            raise SM121AgentAdmissionExecutionError("tool") from error
        scenario_passes[case.id] = completed == repetitions
        completed_episodes[case.id] = completed
        diagnostics.append(client.diagnostics().to_dict())
    payload = _combine_payload_diagnostics(diagnostics)
    expected_repetitions = {getattr(case, "repetitions") for case in cases}
    if len(expected_repetitions) != 1:
        raise SM121AgentAdmissionExecutionError("tool")
    repetitions = expected_repetitions.pop()
    if not isinstance(repetitions, int):
        raise SM121AgentAdmissionExecutionError("tool")
    expected_bodies = 9 * repetitions
    if (
        set(scenario_passes) != set(SM121_AGENT_ADMISSION_TOOL_CASE_IDS)
        or not all(scenario_passes.values())
        or any(count != repetitions for count in completed_episodes.values())
        or payload["outbound_body_count"] != expected_bodies
        or payload["validated_low_thinking_body_count"] != expected_bodies
        or payload["validated_tool_body_count"] != expected_bodies
        or payload["validated_cache_zero_body_count"] != 0
        or payload["transport_attempt_count"] != expected_bodies
        or payload["transport_retry_count"] != 0
        or payload["payload_contract_verified"] is not True
    ):
        raise SM121AgentAdmissionExecutionError("tool")
    return {
        "event": "sm121_agent_tool_gate",
        "fresh_lifetime": 2,
        "scenario_count": len(cases),
        "scenario_repetitions": repetitions,
        "completed_episodes": completed_episodes,
        "scenario_passes": scenario_passes,
        "tools_admitted": True,
        "payload_diagnostics": payload,
    }


def _native_cache_metrics_receipt(
    *,
    before: dict[str, Any],
    before_lease: object,
    before_polls: int,
    before_settled: bool,
    after: dict[str, Any],
    after_lease: object,
    after_polls: int,
    after_settled: bool,
) -> dict[str, object]:
    try:
        metrics_before = {
            field: before[field]
            for field in SM121_AGENT_ADMISSION_NATIVE_CACHE_METRIC_FIELDS
        }
        metrics_after = {
            field: after[field]
            for field in SM121_AGENT_ADMISSION_NATIVE_CACHE_METRIC_FIELDS
        }
    except (KeyError, TypeError) as error:
        raise SM121AgentAdmissionExecutionError("long_context") from error
    hit_fields = tuple(
        field
        for field in SM121_AGENT_ADMISSION_NATIVE_CACHE_METRIC_FIELDS
        if field
        not in {"prefill_input_tokens", "evicted_tokens", "retracted_requests"}
    )
    receipt: dict[str, object] = {
        "event": "sm121_agent_native_cache_metrics_receipt",
        "schema_version": 1,
        "fresh_lifetime": 3,
        "same_owned_generation": before_lease == after_lease,
        "metrics_available": (
            before.get("available") is True and after.get("available") is True
        ),
        "guardrail_metrics_available": (
            before.get("guardrail_metrics_available") is True
            and after.get("guardrail_metrics_available") is True
        ),
        "metrics_before_settled": before_settled,
        "metrics_after_settled": after_settled,
        "metrics_before_polls": before_polls,
        "metrics_after_polls": after_polls,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "native_input_observed": (
            type(metrics_before.get("prefill_input_tokens")) is int
            and type(metrics_after.get("prefill_input_tokens")) is int
            and metrics_after["prefill_input_tokens"]
            > metrics_before["prefill_input_tokens"]
        ),
        "zero_metric_cache_hits": all(
            metrics_before.get(field) == metrics_after.get(field) == 0
            for field in hit_fields
        ),
        "guardrails_clean": all(
            metrics_before.get(field) == metrics_after.get(field) == 0
            for field in ("evicted_tokens", "retracted_requests")
        ),
    }
    try:
        return validate_sm121_agent_native_cache_metrics_receipt(receipt)
    except SM121AgentAdmissionError as error:
        raise SM121AgentAdmissionExecutionError("long_context") from error


def _long_context_gate(
    *,
    server: object,
    model: SimpleNamespace,
    deadline: float,
    watchdog: HostSafetyWatchdog | None,
    journal: Journal,
) -> dict[str, object]:
    client = create_sm121_agent_admission_client(
        server=server,
        model=model,
        case_id=SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID,
    )
    client._bind_controller_deadline(deadline)
    try:
        _abort_check(watchdog=watchdog, deadline=deadline)
        before, before_lease, before_polls, before_settled = (
            settle_sm121_agent_admission_metrics(
                server,  # type: ignore[arg-type]
                deadline=deadline,
                timeout_s=min(
                    SM121_AGENT_ADMISSION_METRICS_SETTLE_TIMEOUT_S,
                    _remaining_s(deadline),
                ),
                poll_interval_s=SM121_AGENT_ADMISSION_METRICS_POLL_INTERVAL_S,
            )
        )
        _abort_check(watchdog=watchdog, deadline=deadline)
        client.run_long_context_turn()
        _abort_check(watchdog=watchdog, deadline=deadline)
        after, after_lease, after_polls, after_settled = (
            settle_sm121_agent_admission_metrics(
                server,  # type: ignore[arg-type]
                deadline=deadline,
                timeout_s=min(
                    SM121_AGENT_ADMISSION_METRICS_SETTLE_TIMEOUT_S,
                    _remaining_s(deadline),
                ),
                poll_interval_s=SM121_AGENT_ADMISSION_METRICS_POLL_INTERVAL_S,
                expected_lease=before_lease,
            )
        )
        _abort_check(watchdog=watchdog, deadline=deadline)
    except SM121AgentAdmissionExecutionError:
        raise
    except Exception as error:
        raise SM121AgentAdmissionExecutionError("long_context") from error
    receipt = _native_cache_metrics_receipt(
        before=before,
        before_lease=before_lease,
        before_polls=before_polls,
        before_settled=before_settled,
        after=after,
        after_lease=after_lease,
        after_polls=after_polls,
        after_settled=after_settled,
    )
    journal.append(receipt)
    try:
        long_context = client.long_context_receipt()
        payload = validate_c1_payload_diagnostics(client.diagnostics().to_dict())
    except SM121AgentAdmissionRequestError as error:
        raise SM121AgentAdmissionExecutionError("long_context") from error
    if (
        long_context.get("input_tokenization_verified") is not True
        or long_context.get("context_fit") is not True
        or long_context.get("zero_response_cache_hits") is not True
        or long_context.get("response_semantics_verified") is not True
        or long_context.get("first_turn_only") is not True
        or payload != {
            "outbound_body_count": 1,
            "validated_low_thinking_body_count": 1,
            "validated_tool_body_count": 1,
            "validated_cache_zero_body_count": 1,
            "transport_attempt_count": 1,
            "transport_retry_count": 0,
            "payload_contract_verified": True,
        }
    ):
        raise SM121AgentAdmissionExecutionError("long_context")
    return {
        "event": "sm121_agent_long_context_gate",
        "fresh_lifetime": 3,
        "input_tokenization_verified": True,
        "context_fit": True,
        "zero_metric_cache_hits": receipt["zero_metric_cache_hits"],
        "zero_response_cache_hits": True,
        "guardrails_clean": receipt["guardrails_clean"],
        "long_context_admitted": True,
        "payload_diagnostics": payload,
    }


def _summary(
    *,
    parser_static_admitted: bool,
    quality_admitted: bool,
    tools_admitted: bool,
    long_context_admitted: bool,
    source_static_attestations: int,
    runtime_attestations: int,
    completed_lifetimes: int,
    terminal_stage: str,
    failure_code: str,
    record_valid: bool = True,
) -> dict[str, object]:
    complete = (
        parser_static_admitted
        and quality_admitted
        and tools_admitted
        and long_context_admitted
        and source_static_attestations == 3
        and runtime_attestations == 3
        and completed_lifetimes == 3
        and record_valid is True
    )
    value: dict[str, object] = {
        "schema_version": 1,
        "admission_id": SM121_AGENT_ADMISSION_ID,
        "execution_mode": SM121_AGENT_ADMISSION_EXECUTION_MODE,
        "status": "complete" if complete else "partial",
        "decision": "admitted" if complete else "blocked",
        "terminal_stage": "complete" if complete else terminal_stage,
        "failure_code": None if complete else failure_code,
        "profile_id": SM121_AGENT_ADMISSION_PROFILE_ID,
        "suite_id": SM121_AGENT_ADMISSION_SUITE_ID,
        "parser_static_admitted": parser_static_admitted,
        "quality_admitted": quality_admitted,
        "tools_admitted": tools_admitted,
        "long_context_admitted": long_context_admitted,
        "source_static_attestations": source_static_attestations,
        "runtime_attestations": runtime_attestations,
        "completed_lifetimes": completed_lifetimes,
        "record_valid": record_valid,
    }
    value["integrity_hash"] = content_hash(value, 64)
    _validate_summary(value)
    return value


def _validate_summary(value: object) -> dict[str, object]:
    fields = {
        "schema_version", "admission_id", "execution_mode", "status", "decision",
        "terminal_stage", "failure_code", "profile_id", "suite_id",
        "parser_static_admitted", "quality_admitted", "tools_admitted",
        "long_context_admitted", "source_static_attestations", "runtime_attestations",
        "completed_lifetimes", "record_valid", "integrity_hash",
    }
    if type(value) is not dict or set(value) != fields:
        raise SM121AgentAdmissionError("SM121 agent admission summary fields are invalid")
    digest = value["integrity_hash"]
    if (
        not isinstance(digest, str)
        or _SHA256_HEX.fullmatch(digest) is None
        or content_hash(
            {key: item for key, item in value.items() if key != "integrity_hash"},
            64,
        )
        != digest
    ):
        raise SM121AgentAdmissionError("SM121 agent admission summary integrity is invalid")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["admission_id"] != SM121_AGENT_ADMISSION_ID
        or value["execution_mode"] != SM121_AGENT_ADMISSION_EXECUTION_MODE
        or value["profile_id"] != SM121_AGENT_ADMISSION_PROFILE_ID
        or value["suite_id"] != SM121_AGENT_ADMISSION_SUITE_ID
    ):
        raise SM121AgentAdmissionError("SM121 agent admission summary identity is invalid")
    if (
        type(value["status"]) is not str
        or type(value["decision"]) is not str
        or type(value["terminal_stage"]) is not str
        or (
            value["failure_code"] is not None
            and type(value["failure_code"]) is not str
        )
    ):
        raise SM121AgentAdmissionError("SM121 agent admission summary types are invalid")
    for name in (
        "parser_static_admitted",
        "quality_admitted",
        "tools_admitted",
        "long_context_admitted",
        "record_valid",
    ):
        if type(value[name]) is not bool:
            raise SM121AgentAdmissionError("SM121 agent admission summary booleans are invalid")
    for name in ("source_static_attestations", "runtime_attestations", "completed_lifetimes"):
        if type(value[name]) is not int or not 0 <= value[name] <= 3:
            raise SM121AgentAdmissionError("SM121 agent admission summary counts are invalid")
    complete = value["record_valid"] is True and all(
        value[name] is True
        for name in ("parser_static_admitted", "quality_admitted", "tools_admitted", "long_context_admitted")
    ) and all(value[name] == 3 for name in ("source_static_attestations", "runtime_attestations", "completed_lifetimes"))
    valid = (
        value["status"] == "complete"
        and value["terminal_stage"] == "complete"
        and value["failure_code"] is None
    ) if complete else (
        value["status"] == "partial"
        and value["decision"] == "blocked"
        and value["terminal_stage"] in _TERMINAL_STAGES
        and value["failure_code"] in SM121_AGENT_ADMISSION_FAILURE_CODES
    )
    if not valid:
        raise SM121AgentAdmissionError("SM121 agent admission summary decision is invalid")
    return value


def _require_fresh_admission_topology(root: Path) -> None:
    _require_safe_tree(root)
    try:
        entries = {entry.name for entry in root.iterdir()}
    except OSError as error:
        raise base_runner.PreflightError("SM121 agent admission run is unavailable") from error
    if entries != {"plan.json", "inventory.json"}:
        raise base_runner.PreflightError("SM121 agent admission plan is not fresh")


def _recover_incomplete_admission_lifetimes(
    *, model: SimpleNamespace, root: Path
) -> None:
    """Recover only an exact stale C1 container before rejecting a resume."""

    for fresh_lifetime in range(1, SM121_AGENT_ADMISSION_LIFETIME_COUNT + 1):
        try:
            action = recover_owned_sglang(
                str(model.run_identity),
                api_key_path=root / "server" / f"lifetime-{fresh_lifetime}" / "api-key",
            )
        except Exception:
            raise base_runner.PreflightError(
                "SM121 agent admission recovery failed"
            ) from None
        if action == "different_container_present":
            raise base_runner.PreflightError("SM121 agent admission recovery is blocked")


def _failure_code_for(
    error: BaseException, *, phase: str
) -> str:
    if isinstance(error, SM121AgentAdmissionExecutionError):
        return error.failure_code
    if isinstance(error, HostSafetyError):
        return "host_safety"
    if isinstance(error, base_runner.PreflightError):
        return "preflight"
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, SM121AgentAdmissionRequestError):
        return {
            "quality": "quality",
            "tool": "tool",
            "tools": "tool",
            "long_context": "long_context",
        }.get(phase, "generic")
    if isinstance(error, RuntimeErrorWithContext):
        return "runtime_identity"
    return "generic"


def _run_lifetime(
    *,
    run_dir: Path,
    workspace: Path,
    model: SimpleNamespace,
    fresh_lifetime: int,
    phase: str,
    first_case_id: str,
    controller_session: object,
    journal: Journal,
    operation: Callable[[object, float, HostSafetyWatchdog | None], dict[str, object]],
) -> None:
    """Run one C1 gate in one fresh owned server lifetime and clean it up."""

    started = time.monotonic()
    deadline = started + SM121_AGENT_ADMISSION_LIFETIME_TIMEOUT_S
    server: object | None = None
    watchdog: HostSafetyWatchdog | None = None
    launch_lease: object | None = None
    launch_attempted = False
    server_stopped = False
    terminal_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    api_key_path = run_dir / "server" / f"lifetime-{fresh_lifetime}" / "api-key"
    run_identity = getattr(model, "run_identity", None)
    try:
        _remaining_s(deadline)
        base_runner._preflight(model)
        _remaining_s(deadline)
        try:
            require_sm121_agent_admission_clean_start()
        except RuntimeErrorWithContext as error:
            raise SM121AgentAdmissionExecutionError("preflight") from error
        _remaining_s(deadline)
        watchdog = base_runner._host_safety_watchdog(model)
        if watchdog is not None:
            watchdog.start()
        _abort_check(watchdog=watchdog, deadline=deadline)
        journal.append(
            _static_event(
                model=model,
                fresh_lifetime=fresh_lifetime,
                phase=phase,
            )
        )
        try:
            # A separate one-shot lease is minted only after this exact
            # lifetime passed its own clean-start check. It is consumed at the
            # final Docker-launch boundary and revoked during cleanup.
            launch_lease = _issue_sm121_agent_admission_launch_lease(
                model, controller_session, fresh_lifetime=fresh_lifetime
            )
        except SM121AgentAdmissionError as error:
            raise SM121AgentAdmissionExecutionError("preflight") from error
        callbacks: dict[str, Any] = {
            "abort_check": lambda: _abort_check(
                watchdog=watchdog, deadline=deadline
            )
        }
        if watchdog is not None:
            callbacks["on_server_created"] = (
                lambda created: watchdog.register_abort_callback(
                    created.interrupt_owned
                )
            )
        try:
            launch_attempted = True
            server = start_sm121_agent_admission_server(
                model,
                workspace=workspace,
                server_log_path=(
                    run_dir
                    / "server"
                    / f"lifetime-{fresh_lifetime}"
                    / "server.log"
                ),
                _launch_capability=launch_lease,
                **callbacks,
            )
        except RuntimeErrorWithContext as error:
            # Startup can fail because an exact snapshot/image/seccomp/port or
            # Docker prerequisite disappeared. Keep that distinct from the
            # post-launch runtime-identity attestation below.
            raise SM121AgentAdmissionExecutionError(
                "dependency_unavailable"
            ) from error
        _abort_check(watchdog=watchdog, deadline=deadline)
        journal.append(
            _runtime_event(
                server=server,
                fresh_lifetime=fresh_lifetime,
                phase=phase,
            )
        )
        journal.append(
            {
                "event": "server_ready",
                "backend": "sglang",
                "fresh_lifetime": fresh_lifetime,
                "phase": phase,
                "first_inference_is_admission_gate": True,
                "first_protocol_case": first_case_id,
            }
        )
        journal.append(operation(server, deadline, watchdog))
        _abort_check(watchdog=watchdog, deadline=deadline)
    except BaseException as error:
        terminal_error = (
            watchdog.failure if watchdog is not None and watchdog.failure else error
        )
    finally:
        if server is not None:
            if watchdog is not None and watchdog.tripped:
                try:
                    server.interrupt_owned()  # type: ignore[attr-defined]
                except BaseException as error:
                    cleanup_error = error
            if terminal_error is not None or time.monotonic() >= deadline:
                try:
                    server.interrupt_owned()  # type: ignore[attr-defined]
                except BaseException as error:
                    cleanup_error = cleanup_error or error
            try:
                server.stop()  # type: ignore[attr-defined]
                if api_key_path.exists() or api_key_path.is_symlink():
                    raise RuntimeError("C1 API-key residue")
                journal.append(
                    {
                        "event": "server_stopped",
                        "backend": "sglang",
                        "fresh_lifetime": fresh_lifetime,
                    }
                )
                server_stopped = True
            except BaseException as error:
                cleanup_error = cleanup_error or error
                try:
                    server.interrupt_owned()  # type: ignore[attr-defined]
                    server.stop()  # type: ignore[attr-defined]
                    if api_key_path.exists() or api_key_path.is_symlink():
                        raise RuntimeError("C1 API-key residue")
                    journal.append(
                        {
                            "event": "server_stopped",
                            "backend": "sglang",
                            "fresh_lifetime": fresh_lifetime,
                        }
                    )
                    server_stopped = True
                    cleanup_error = None
                except BaseException:
                    pass
        if (
            launch_attempted
            and isinstance(run_identity, str)
            and run_identity
            and (server is None or cleanup_error is not None)
        ):
            # start_sm121_agent_admission_server can create an owned Docker
            # container and then fail before returning a ManagedServer (for
            # example, readiness or its first cleanup can fail). Recover the
            # exact run identity here as a second bounded cleanup path; never
            # touch a differently owned replacement.
            try:
                action = recover_owned_sglang(
                    run_identity, api_key_path=api_key_path
                )
                if action not in {"already_absent", "stopped_owned_container"}:
                    raise RuntimeError("C1 startup recovery did not own the server")
                if api_key_path.exists() or api_key_path.is_symlink():
                    raise RuntimeError("C1 API-key residue")
                if not server_stopped and (
                    action == "stopped_owned_container" or server is not None
                ):
                    journal.append(
                        {
                            "event": "server_stopped",
                            "backend": "sglang",
                            "fresh_lifetime": fresh_lifetime,
                        }
                    )
                    server_stopped = True
                cleanup_error = None
            except BaseException as error:
                cleanup_error = cleanup_error or error
        if watchdog is not None:
            try:
                watchdog.stop()
                if terminal_error is None:
                    watchdog.raise_if_tripped()
            except BaseException as error:
                terminal_error = terminal_error or error
        if cleanup_error is not None:
            terminal_error = SM121AgentAdmissionExecutionError("cleanup")
        _revoke_sm121_agent_admission_launch_lease(launch_lease)
    within_timeout = time.monotonic() - started <= SM121_AGENT_ADMISSION_LIFETIME_TIMEOUT_S
    journal.append(
        {
            "event": "sm121_agent_lifetime_complete",
            "fresh_lifetime": fresh_lifetime,
            "phase": phase,
            "within_timeout": within_timeout,
            "admitted": terminal_error is None and within_timeout,
        }
    )
    if terminal_error is not None:
        raise SM121AgentAdmissionExecutionError(
            _failure_code_for(terminal_error, phase=phase)
        ) from None
    if not within_timeout:
        raise SM121AgentAdmissionExecutionError("timeout")


def _complete_candidate_errors(
    events: list[dict[str, Any]],
    summary: dict[str, object],
    root: Path,
) -> list[dict[str, str]]:
    """Audit a complete terminal shape before it can be written as admitted."""

    timestamp = "2000-01-01T00:00:00.000+00:00"
    candidate = [
        *events,
        {
            "timestamp": timestamp,
            "event": "measurement_complete",
            "status": "complete",
        },
        {"timestamp": timestamp, "event": "run_complete", "status": "admitted"},
    ]
    return _complete_errors(candidate, summary, _server_issues(root))


def execute_sm121_agent_admission(
    run_dir: Path,
    *,
    workspace: Path,
) -> dict[str, object]:
    """Run the one non-resumable private C1 agent-admission controller."""

    root = _path(
        run_dir, existing=True, allow_logs_root=False, create_logs_root=False
    )
    _private_run_directory(root)
    _private_directory(root.parent, create=False)
    _require_safe_tree(root)
    plan, model, suite = _load_plan(root)
    lock_path = base_runner.results_lock_path(workspace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another SparkBench run holds the benchmark lock") from error
        summary_path = root / "admission.json"
        journal = Journal(root / "events.jsonl")
        _require_safe_tree(root)
        if summary_path.exists():
            # A prior non-resumable partial may have failed after Docker
            # created a container but before the launcher returned it. Make one
            # exact-ownership recovery attempt before refusing the old plan.
            _recover_incomplete_admission_lifetimes(model=model, root=root)
            raise base_runner.PreflightError(
                "SM121 agent admission is non-resumable; freeze a new plan"
            )
        try:
            existing_events = journal.strict_events()
        except (OSError, ValueError) as error:
            raise base_runner.PreflightError(
                "SM121 agent admission record is invalid"
            ) from error
        if existing_events:
            _recover_incomplete_admission_lifetimes(model=model, root=root)
            raise base_runner.PreflightError(
                "SM121 agent admission is non-resumable; freeze a new plan"
            )
        _require_fresh_admission_topology(root)
        quality_case, tool_cases, long_context_case = _admission_cases(suite)
        _require_context_and_tasks(
            model=model,
            quality=quality_case,
            tools=tool_cases,
            long_context=long_context_case,
        )
        journal.append(
            {
                "event": "run_start",
                "execution_mode": SM121_AGENT_ADMISSION_EXECUTION_MODE,
                "admission_id": SM121_AGENT_ADMISSION_ID,
                "profile_id": model.id,
                "suite_id": suite.id,
            }
        )
        journal.append({"event": "measurement_started"})
        parser_static_admitted = False
        quality_admitted = False
        tools_admitted = False
        long_context_admitted = False
        terminal_stage = "parser_static"
        failure_code = "generic"
        controller_session: _SM121AgentAdmissionControllerSession | None = None
        try:
            base_runner._preflight(model)
            try:
                require_sm121_agent_admission_clean_start()
            except RuntimeErrorWithContext as error:
                raise SM121AgentAdmissionExecutionError("preflight") from error
            host_gate = base_runner._host_safety_watchdog(model)
            if host_gate is not None:
                try:
                    host_gate.start()
                    host_gate.raise_if_tripped()
                finally:
                    host_gate.stop()
                host_gate.raise_if_tripped()
            try:
                journal.append(probe_sm121_agent_parser_static_preflight(model))
            except SM121AgentAdmissionError as error:
                raise SM121AgentAdmissionExecutionError("static_parser") from error
            try:
                snapshot_path = sm121_agent_admission_target_snapshot(
                    model, workspace=workspace
                )
                budget_probe = probe_sm121_agent_long_context_budget_preflight(
                    model, snapshot_path=snapshot_path
                )
                journal.append(
                    {
                        "event": "sm121_agent_long_context_budget_preflight",
                        **budget_probe,
                    }
                )
            except (RuntimeErrorWithContext, SM121AgentAdmissionError) as error:
                raise SM121AgentAdmissionExecutionError("long_context") from error
            # Registration is deliberately inline and later than the top-level
            # clean-start, host, parser, and exact-tokenizer gates. There is no
            # callable plan-to-authority factory: _load_plan returns only
            # authenticated data, so generic code cannot turn it into a C1
            # launch authorization without entering this gated executor.
            model_data = plan.get("model")
            suite_data = plan.get("suite")
            resolved = plan.get("resolved")
            fingerprint = plan.get("fingerprint")
            integrity = plan.get("integrity_hash")
            nonce = plan.get("run_nonce")
            if (
                type(model_data) is not dict
                or type(suite_data) is not dict
                or type(resolved) is not dict
                or not isinstance(fingerprint, str)
                or _FINGERPRINT_HEX.fullmatch(fingerprint) is None
                or not isinstance(integrity, str)
                or _SHA256_HEX.fullmatch(integrity) is None
                or not isinstance(nonce, str)
                or _RUN_NONCE_HEX.fullmatch(nonce) is None
                or integrity
                != content_hash(
                    {key: value for key, value in plan.items() if key != "integrity_hash"},
                    64,
                )
            ):
                raise SM121AgentAdmissionExecutionError("preflight")
            cases = suite_data.get("cases")
            if not isinstance(cases, list) or any(type(case) is not dict for case in cases):
                raise SM121AgentAdmissionExecutionError("preflight")
            stripped_suite = {
                **suite_data,
                "cases": [
                    {key: value for key, value in case.items() if key != "case_id"}
                    for case in cases
                ],
            }
            if fingerprint != content_hash(
                {
                    "model": model_data,
                    "suite": stripped_suite,
                    "resolved": resolved,
                }
            ):
                raise SM121AgentAdmissionExecutionError("preflight")
            session = _SM121AgentAdmissionControllerSession()
            registration = _SM121AgentAdmissionControllerRegistration(
                session=session,
                model=model,
                fingerprint=fingerprint,
                integrity_hash=integrity,
                run_nonce=nonce,
                run_identity=f"{fingerprint}-{nonce}",
                local_image_id=SM121_STORAGE_LOCAL_IMAGE_ID,
            )
            # Assign before registry insertion so the enclosing finally can
            # revoke an entry even if an asynchronous exception lands at the
            # narrow registration boundary.
            controller_session = session
            with _SM121_AGENT_ADMISSION_REGISTRY_LOCK:
                if not _registered_model_matches(model, registration):
                    raise SM121AgentAdmissionExecutionError("preflight")
                _SM121_AGENT_ADMISSION_CONTROLLERS[id(session)] = registration
            parser_static_admitted = True
            terminal_stage = "quality_lifetime"
            _run_lifetime(
                run_dir=root,
                workspace=workspace,
                model=model,
                fresh_lifetime=1,
                phase="quality",
                first_case_id=quality_case.id,
                controller_session=controller_session,
                journal=journal,
                operation=lambda server, deadline, watchdog: _quality_gate(
                    server=server,
                    model=model,
                    case=quality_case,
                    deadline=deadline,
                    watchdog=watchdog,
                ),
            )
            quality_admitted = True
            terminal_stage = "tool_lifetime"
            _run_lifetime(
                run_dir=root,
                workspace=workspace,
                model=model,
                fresh_lifetime=2,
                phase="tools",
                first_case_id=tool_cases[0].id,
                controller_session=controller_session,
                journal=journal,
                operation=lambda server, deadline, watchdog: _tool_gate(
                    server=server,
                    model=model,
                    cases=tool_cases,
                    deadline=deadline,
                    watchdog=watchdog,
                ),
            )
            tools_admitted = True
            terminal_stage = "long_context_lifetime"
            _run_lifetime(
                run_dir=root,
                workspace=workspace,
                model=model,
                fresh_lifetime=3,
                phase="long_context",
                first_case_id=long_context_case.id,
                controller_session=controller_session,
                journal=journal,
                operation=lambda server, deadline, watchdog: _long_context_gate(
                    server=server,
                    model=model,
                    deadline=deadline,
                    watchdog=watchdog,
                    journal=journal,
                ),
            )
            long_context_admitted = True
        except BaseException as error:
            failure_code = _failure_code_for(error, phase=terminal_stage.removesuffix("_lifetime"))
            journal.append(
                {
                    "event": "sm121_agent_blocked",
                    "terminal_stage": terminal_stage,
                    "failure_code": failure_code,
                }
            )
        finally:
            _close_sm121_agent_admission_controller(controller_session)
        events = journal.events()
        summary = _summary(
            parser_static_admitted=parser_static_admitted,
            quality_admitted=quality_admitted,
            tools_admitted=tools_admitted,
            long_context_admitted=long_context_admitted,
            source_static_attestations=sum(
                event.get("event") == "sm121_agent_static_attestation"
                for event in events
            ),
            runtime_attestations=sum(
                event.get("event") == "sm121_agent_runtime_attestation"
                for event in events
            ),
            completed_lifetimes=sum(
                event.get("event") == "sm121_agent_lifetime_complete"
                and event.get("admitted") is True
                for event in events
            ),
            terminal_stage=terminal_stage,
            failure_code=failure_code,
        )
        if summary["decision"] == "admitted":
            # A structurally invalid complete record must never turn into a
            # successful process exit merely because gate booleans happened to
            # be true. Validate the exact terminal topology in memory before
            # appending it or writing the admitted summary.
            if _complete_candidate_errors(events, summary, root):
                terminal_stage = "record_audit"
                failure_code = "generic"
                summary = _summary(
                    parser_static_admitted=parser_static_admitted,
                    quality_admitted=quality_admitted,
                    tools_admitted=tools_admitted,
                    long_context_admitted=long_context_admitted,
                    source_static_attestations=sum(
                        event.get("event") == "sm121_agent_static_attestation"
                        for event in events
                    ),
                    runtime_attestations=sum(
                        event.get("event") == "sm121_agent_runtime_attestation"
                        for event in events
                    ),
                    completed_lifetimes=sum(
                        event.get("event") == "sm121_agent_lifetime_complete"
                        and event.get("admitted") is True
                        for event in events
                    ),
                    terminal_stage=terminal_stage,
                    failure_code=failure_code,
                    record_valid=False,
                )
                journal.append(
                    {
                        "event": "sm121_agent_blocked",
                        "terminal_stage": terminal_stage,
                        "failure_code": failure_code,
                    }
                )
        journal.append({"event": "measurement_complete", "status": summary["status"]})
        journal.append({"event": "run_complete", "status": summary["decision"]})
        write_json(summary_path, summary)
        return summary


def _untimestamped(event: object) -> dict[str, object] | None:
    if type(event) is not dict or not isinstance(event.get("timestamp"), str):
        return None
    return {key: value for key, value in event.items() if key != "timestamp"}


def _unsafe(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            not isinstance(key, str) or key in _FORBIDDEN_KEYS or _unsafe(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_unsafe(item) for item in value)
    return False


def _event_schema_safe(event: object) -> bool:
    """Require one bounded event schema before inspecting a private journal.

    Partial records are never accepted as an admission, but they still must
    not become a loophole for retaining arbitrary text.  Until a live
    controller defines additional terminal events, unknown event shapes fail
    closed here.
    """

    if (
        type(event) is not dict
        or not isinstance(event.get("timestamp"), str)
        or _TIMESTAMP.fullmatch(event["timestamp"]) is None
    ):
        return False
    row = {key: value for key, value in event.items() if key != "timestamp"}
    if _unsafe(row):
        return False
    name = row.get("event")
    if name == "run_start":
        return row == {
            "event": "run_start",
            "execution_mode": SM121_AGENT_ADMISSION_EXECUTION_MODE,
            "admission_id": SM121_AGENT_ADMISSION_ID,
            "profile_id": SM121_AGENT_ADMISSION_PROFILE_ID,
            "suite_id": SM121_AGENT_ADMISSION_SUITE_ID,
        }
    if name == "measurement_started":
        return row == {"event": "measurement_started"}
    if name == "sm121_agent_parser_static_attestation":
        try:
            validate_sm121_agent_parser_static_probe(
                {key: value for key, value in row.items() if key != "event"}
            )
        except SM121AgentAdmissionError:
            return False
        return set(row) == {
            "event",
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
    if name == "sm121_agent_long_context_budget_preflight":
        try:
            validate_sm121_agent_long_context_budget_probe(
                {key: value for key, value in row.items() if key != "event"}
            )
        except SM121AgentAdmissionError:
            return False
        return set(row) == {
            "event",
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
    expected_static = {
        "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
        **SM121_CACHE_SOURCE_DIGESTS,
        **_SCALAR_STATIC_ASSERTIONS,
    }
    if name == "sm121_agent_static_attestation":
        return any(
            row
            == {
                "event": "sm121_agent_static_attestation",
                "fresh_lifetime": lifetime,
                "phase": phase,
                **expected_static,
            }
            for lifetime, phase in enumerate(_PHASES, start=1)
        )
    if name == "sm121_agent_runtime_attestation":
        return any(
            row
            == {
                "event": "sm121_agent_runtime_attestation",
                "fresh_lifetime": lifetime,
                "phase": phase,
                **_runtime_expected(),
            }
            for lifetime, phase in enumerate(_PHASES, start=1)
        )
    if name == "server_ready":
        first_cases = (
            SM121_AGENT_ADMISSION_QUALITY_CASE_ID,
            SM121_AGENT_ADMISSION_TOOL_CASE_IDS[0],
            SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID,
        )
        return any(
            row
            == {
                "event": "server_ready",
                "backend": "sglang",
                "fresh_lifetime": lifetime,
                "phase": phase,
                "first_inference_is_admission_gate": True,
                "first_protocol_case": first_case,
            }
            for lifetime, (phase, first_case) in enumerate(
                zip(_PHASES, first_cases, strict=True), start=1
            )
        )
    if name == "sm121_agent_quality_gate":
        return row == _quality_gate_expected()
    if name == "sm121_agent_tool_gate":
        return row == _tool_gate_expected()
    if name == "sm121_agent_native_cache_metrics_receipt":
        try:
            validate_sm121_agent_native_cache_metrics_receipt(row)
        except SM121AgentAdmissionError:
            return False
        return True
    if name == "sm121_agent_long_context_gate":
        return row == _long_context_gate_expected()
    if name == "server_stopped":
        return (
            set(row) == {"event", "backend", "fresh_lifetime"}
            and row["backend"] == "sglang"
            and type(row["fresh_lifetime"]) is int
            and 1 <= row["fresh_lifetime"] <= SM121_AGENT_ADMISSION_LIFETIME_COUNT
        )
    if name == "sm121_agent_lifetime_complete":
        return (
            set(row)
            == {
                "event",
                "fresh_lifetime",
                "phase",
                "within_timeout",
                "admitted",
            }
            and type(row["fresh_lifetime"]) is int
            and (row["fresh_lifetime"], row["phase"])
            in tuple(enumerate(_PHASES, start=1))
            and type(row["within_timeout"]) is bool
            and type(row["admitted"]) is bool
        )
    if name == "sm121_agent_blocked":
        return (
            set(row) == {"event", "terminal_stage", "failure_code"}
            and type(row["terminal_stage"]) is str
            and type(row["failure_code"]) is str
            and row["terminal_stage"] in _TERMINAL_STAGES
            and row["failure_code"] in SM121_AGENT_ADMISSION_FAILURE_CODES
        )
    if name == "measurement_complete":
        return (
            set(row) == {"event", "status"}
            and type(row["status"]) is str
            and row["status"] in {"complete", "partial"}
        )
    if name == "run_complete":
        return (
            set(row) == {"event", "status"}
            and type(row["status"]) is str
            and row["status"] in {"admitted", "blocked"}
        )
    return False


def _complete_errors(
    events: list[dict[str, Any]],
    summary: dict[str, object],
    server_issues: tuple[str, ...],
) -> list[dict[str, str]]:
    """Validate a complete scalar journal using one exact C1 topology."""

    errors: list[dict[str, str]] = []

    def add(code: str, message: str) -> None:
        errors.append({"code": code, "message": message})

    expected_names = (
        "run_start",
        "measurement_started",
        "sm121_agent_parser_static_attestation",
        "sm121_agent_long_context_budget_preflight",
        "sm121_agent_static_attestation",
        "sm121_agent_runtime_attestation",
        "server_ready",
        "sm121_agent_quality_gate",
        "server_stopped",
        "sm121_agent_lifetime_complete",
        "sm121_agent_static_attestation",
        "sm121_agent_runtime_attestation",
        "server_ready",
        "sm121_agent_tool_gate",
        "server_stopped",
        "sm121_agent_lifetime_complete",
        "sm121_agent_static_attestation",
        "sm121_agent_runtime_attestation",
        "server_ready",
        "sm121_agent_native_cache_metrics_receipt",
        "sm121_agent_long_context_gate",
        "server_stopped",
        "sm121_agent_lifetime_complete",
        "measurement_complete",
        "run_complete",
    )
    if tuple(event.get("event") for event in events) != expected_names:
        add("event_topology", "SM121 agent admission journal topology is invalid")
        return errors
    rows = [_untimestamped(event) for event in events]
    if any(row is None for row in rows):
        add("event_timestamp", "SM121 agent admission timestamps are invalid")
        return errors
    if any(not _event_schema_safe(event) for event in events):
        add("scalar_safety", "SM121 agent admission journal contains forbidden data")

    parser = rows[2]
    if parser is None:
        add("parser_static", "SM121 agent parser attestation is invalid")
    else:
        try:
            validate_sm121_agent_parser_static_probe(
                {key: value for key, value in parser.items() if key != "event"}
            )
        except SM121AgentAdmissionError:
            add("parser_static", "SM121 agent parser attestation is invalid")

    budget_probe = rows[3]
    if budget_probe is None:
        add("long_context_budget", "SM121 agent long-context budget is invalid")
    else:
        try:
            validate_sm121_agent_long_context_budget_probe(
                {key: value for key, value in budget_probe.items() if key != "event"}
            )
        except SM121AgentAdmissionError:
            add("long_context_budget", "SM121 agent long-context budget is invalid")

    expected_static = {
        "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
        **SM121_CACHE_SOURCE_DIGESTS,
        **_SCALAR_STATIC_ASSERTIONS,
    }
    lifetimes = (
        (1, "quality", SM121_AGENT_ADMISSION_QUALITY_CASE_ID, _quality_gate_expected()),
        (2, "tools", SM121_AGENT_ADMISSION_TOOL_CASE_IDS[0], _tool_gate_expected()),
        (3, "long_context", SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID, _long_context_gate_expected()),
    )
    cursor = 4
    receipt: dict[str, object] | None = None
    for lifetime, phase, first_case, gate in lifetimes:
        static = rows[cursor]
        cursor += 1
        runtime = rows[cursor]
        cursor += 1
        ready = rows[cursor]
        cursor += 1
        if static != {
            "event": "sm121_agent_static_attestation",
            "fresh_lifetime": lifetime,
            "phase": phase,
            **expected_static,
        }:
            add("static_attestation", "SM121 agent static attestation is invalid")
        if runtime != {
            "event": "sm121_agent_runtime_attestation",
            "fresh_lifetime": lifetime,
            "phase": phase,
            **_runtime_expected(),
        }:
            add("runtime_attestation", "SM121 agent runtime attestation is invalid")
        if ready != {
            "event": "server_ready",
            "backend": "sglang",
            "fresh_lifetime": lifetime,
            "phase": phase,
            "first_inference_is_admission_gate": True,
            "first_protocol_case": first_case,
        }:
            add("server_ready", "SM121 agent server readiness is invalid")
        if lifetime == 3:
            candidate_receipt = rows[cursor]
            cursor += 1
            try:
                receipt = validate_sm121_agent_native_cache_metrics_receipt(
                    candidate_receipt
                )
            except SM121AgentAdmissionError:
                add("native_cache_receipt", "SM121 agent native cache receipt is invalid")
        if rows[cursor] != gate:
            add(
                "long_context_gate" if lifetime == 3 else f"{phase}_gate",
                "SM121 agent admission gate is invalid",
            )
        cursor += 1
        if rows[cursor] != {
            "event": "server_stopped",
            "backend": "sglang",
            "fresh_lifetime": lifetime,
        }:
            add("server_stopped", "SM121 agent server cleanup is invalid")
        cursor += 1
        if rows[cursor] != {
            "event": "sm121_agent_lifetime_complete",
            "fresh_lifetime": lifetime,
            "phase": phase,
            "within_timeout": True,
            "admitted": True,
        }:
            add("lifetime_complete", "SM121 agent lifetime completion is invalid")
        cursor += 1
    if rows[0] != {
        "event": "run_start",
        "execution_mode": SM121_AGENT_ADMISSION_EXECUTION_MODE,
        "admission_id": SM121_AGENT_ADMISSION_ID,
        "profile_id": SM121_AGENT_ADMISSION_PROFILE_ID,
        "suite_id": SM121_AGENT_ADMISSION_SUITE_ID,
    }:
        add("run_start", "SM121 agent admission start is invalid")
    if rows[cursor] != {"event": "measurement_complete", "status": "complete"} or rows[cursor + 1] != {"event": "run_complete", "status": "admitted"}:
        add("completion", "SM121 agent admission terminal records are invalid")
    if receipt is not None:
        long_gate = _long_context_gate_expected()
        if (
            receipt["zero_metric_cache_hits"] is not long_gate["zero_metric_cache_hits"]
            or receipt["guardrails_clean"] is not long_gate["guardrails_clean"]
        ):
            add("native_cache_receipt", "SM121 agent native cache receipt disagrees with the gate")
    if server_issues:
        add("server_cleanup", "SM121 agent admission retained unsafe server artifacts")
    if any(
        summary[name] is not True
        for name in (
            "parser_static_admitted",
            "quality_admitted",
            "tools_admitted",
            "long_context_admitted",
            "record_valid",
        )
    ) or any(
        summary[name] != 3
        for name in (
            "source_static_attestations",
            "runtime_attestations",
            "completed_lifetimes",
        )
    ):
        add("summary", "SM121 agent admission summary is incomplete")
    return errors


def audit_sm121_agent_admission(run_dir: Path) -> dict[str, object]:
    """Audit a terminal private C1 record without starting or stopping anything."""

    report: dict[str, object] = {"schema_version": 1, "admission_id": SM121_AGENT_ADMISSION_ID, "read_only": True, "ok": False, "errors": []}
    errors = report["errors"]
    assert isinstance(errors, list)
    try:
        root = _path(run_dir, existing=True, allow_logs_root=False, create_logs_root=False)
        _private_run_directory(root)
        _private_directory(root.parent, create=False)
        _require_safe_tree(root)
        _plan, _model, _suite = _load_plan(root)
    except (OSError, ValueError, base_runner.PreflightError):
        errors.append({"code": "invalid_location", "message": "SM121 agent admission location is invalid"})
        report["error_count"] = len(errors)
        return report
    if not _owned_file(root / "events.jsonl") or not _owned_file(root / "admission.json"):
        errors.append({"code": "missing_record", "message": "SM121 agent admission journal or summary is unavailable"})
        report["error_count"] = len(errors)
        return report
    try:
        events = Journal(root / "events.jsonl").strict_events()
        summary = _validate_summary(_read_json(root / "admission.json", context="SM121 agent admission summary"))
    except (OSError, ValueError, base_runner.PreflightError, SM121AgentAdmissionError):
        errors.append({"code": "invalid_record", "message": "SM121 agent admission record is invalid"})
        report["error_count"] = len(errors)
        return report
    scalar_unsafe = any(not _event_schema_safe(event) for event in events)
    server_issues = _server_issues(root)
    if scalar_unsafe:
        errors.append(
            {
                "code": "scalar_safety",
                "message": "SM121 agent admission journal violates the scalar schema",
            }
        )
    if summary["status"] != "complete" or summary["decision"] != "admitted":
        if server_issues:
            errors.append(
                {
                    "code": "server_cleanup",
                    "message": "SM121 agent admission retained unsafe server artifacts",
                }
            )
        errors.append({"code": "not_admitted", "message": "SM121 agent admission did not reach admission"})
    else:
        errors.extend(_complete_errors(events, summary, server_issues))
    report["status"] = summary["status"]
    report["decision"] = summary["decision"]
    report["error_count"] = len(errors)
    report["ok"] = not errors
    return report
