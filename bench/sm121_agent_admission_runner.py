"""Private, non-evidence C1 plan and audit scaffold for the SM121 agent profile.

This module freezes a C1-only plan under ``logs/`` and structurally audits a
future scalar-only terminal record.  It deliberately contains no server,
request, tool, or hook execution path.  A later live controller must add
reviewed in-repository adapters for final-body observation and runtime
parser/limit inspection; it must not accept caller-provided callables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
from types import SimpleNamespace
from typing import Any

from . import runner as base_runner
from .journal import Journal, content_hash
from .sglang_sm121_agent_admission import (
    SM121_AGENT_ADMISSION_CHUNKED_PREFILL_SIZE,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID,
    SM121_AGENT_ADMISSION_MAX_MAMBA_CACHE_SIZE,
    SM121_AGENT_ADMISSION_PROFILE_ID,
    SM121_AGENT_ADMISSION_QUALITY_CASE_ID,
    SM121_AGENT_ADMISSION_SUITE_ID,
    SM121_AGENT_ADMISSION_TOOL_CASE_IDS,
    SM121AgentAdmissionError,
    validate_sm121_agent_admission_candidate,
    validate_sm121_agent_admission_suite,
    validate_sm121_agent_parser_static_probe,
)
from .sglang_sm121_cache_observability import SM121_CACHE_SOURCE_DIGESTS
from .sglang_sm121_cache_semantic import (
    SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
    SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED,
    SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
)
from .sglang_sm121_storage import (
    SM121_STORAGE_CONTEXT_LENGTH,
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_SOURCE_TREE,
)


SM121_AGENT_ADMISSION_ID = "qwen38-flash-next-sm121-agent-c1-admission-v1"
SM121_AGENT_ADMISSION_EXECUTION_MODE = "private_non_evidence_c1_admission"
SM121_AGENT_ADMISSION_LIFETIME_COUNT = 3
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
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+00:00")
_PHASES = ("quality", "tools", "long_context")
_TERMINAL_STAGES = frozenset(
    {"parser_static", "quality_lifetime", "tool_lifetime", "long_context_lifetime"}
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


def _load_plan(root: Path) -> tuple[dict[str, Any], SimpleNamespace, SimpleNamespace]:
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
    return plan, model, suite


def _runtime_expected() -> dict[str, object]:
    return {
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
) -> dict[str, object]:
    complete = (
        parser_static_admitted
        and quality_admitted
        and tools_admitted
        and long_context_admitted
        and source_static_attestations == 3
        and runtime_attestations == 3
        and completed_lifetimes == 3
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
        "completed_lifetimes", "integrity_hash",
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
    for name in ("parser_static_admitted", "quality_admitted", "tools_admitted", "long_context_admitted"):
        if type(value[name]) is not bool:
            raise SM121AgentAdmissionError("SM121 agent admission summary booleans are invalid")
    for name in ("source_static_attestations", "runtime_attestations", "completed_lifetimes"):
        if type(value[name]) is not int or not 0 <= value[name] <= 3:
            raise SM121AgentAdmissionError("SM121 agent admission summary counts are invalid")
    complete = all(
        value[name] is True
        for name in ("parser_static_admitted", "quality_admitted", "tools_admitted", "long_context_admitted")
    ) and all(value[name] == 3 for name in ("source_static_attestations", "runtime_attestations", "completed_lifetimes"))
    valid = (
        value["status"] == "complete"
        and value["decision"] == "admitted"
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


def execute_sm121_agent_admission(
    run_dir: Path,
    *,
    workspace: Path,
) -> None:
    """Refuse live C1 execution until concrete in-repository adapters exist.

    A caller-supplied callable can return a fabricated scalar success record;
    it therefore cannot be treated as an admission primitive. The frozen
    private plan and read-only auditor are useful implementation scaffolding,
    but a later change must add reviewed, in-repository request construction,
    final-body observation, and runtime parser/limit inspection before this
    entry point is allowed to touch Docker or the model.
    """

    del run_dir, workspace
    raise RuntimeError(
        "SM121 agent admission live adapters are not implemented; "
        "no model server was started"
    )


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
        return row == {
            "event": "sm121_agent_quality_gate",
            "fresh_lifetime": 1,
            "quality_item_count": 4,
            "quality_admitted": True,
            "payload_contract_verified": True,
        }
    if name == "sm121_agent_tool_gate":
        return row == {
            "event": "sm121_agent_tool_gate",
            "fresh_lifetime": 2,
            "variant": 0,
            "scenario_count": 4,
            "scenario_passes": {
                case_id: True for case_id in SM121_AGENT_ADMISSION_TOOL_CASE_IDS
            },
            "tools_admitted": True,
            "payload_contract_verified": True,
        }
    if name == "sm121_agent_long_context_gate":
        return row == {
            "event": "sm121_agent_long_context_gate",
            "fresh_lifetime": 3,
            "input_tokenization_verified": True,
            "context_fit": True,
            "zero_metric_cache_hits": True,
            "zero_response_cache_hits": True,
            "guardrails_clean": True,
            "long_context_admitted": True,
            "payload_contract_verified": True,
        }
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


def _complete_errors(events: list[dict[str, Any]], summary: dict[str, object], server_issues: tuple[str, ...]) -> list[dict[str, str]]:
    """Validate a complete scalar journal using one exact event topology."""

    errors: list[dict[str, str]] = []
    add = lambda code, message: errors.append({"code": code, "message": message})
    expected_names = (
        "run_start", "measurement_started", "sm121_agent_parser_static_attestation",
        "sm121_agent_static_attestation", "sm121_agent_runtime_attestation", "server_ready", "sm121_agent_quality_gate", "server_stopped", "sm121_agent_lifetime_complete",
        "sm121_agent_static_attestation", "sm121_agent_runtime_attestation", "server_ready", "sm121_agent_tool_gate", "server_stopped", "sm121_agent_lifetime_complete",
        "sm121_agent_static_attestation", "sm121_agent_runtime_attestation", "server_ready", "sm121_agent_long_context_gate", "server_stopped", "sm121_agent_lifetime_complete",
        "measurement_complete", "run_complete",
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
            validate_sm121_agent_parser_static_probe({key: value for key, value in parser.items() if key != "event"})
        except SM121AgentAdmissionError:
            add("parser_static", "SM121 agent parser attestation is invalid")
    expected_static = {
        "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
        **SM121_CACHE_SOURCE_DIGESTS,
        **_SCALAR_STATIC_ASSERTIONS,
    }
    for index, (lifetime, phase, first_case) in enumerate(zip(
        range(1, 4), _PHASES,
        (SM121_AGENT_ADMISSION_QUALITY_CASE_ID, SM121_AGENT_ADMISSION_TOOL_CASE_IDS[0], SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID),
        strict=True,
    )):
        offset = 3 + index * 6
        if rows[offset] != {"event": "sm121_agent_static_attestation", "fresh_lifetime": lifetime, "phase": phase, **expected_static}:
            add("static_attestation", "SM121 agent static attestation is invalid")
        if rows[offset + 1] != {"event": "sm121_agent_runtime_attestation", "fresh_lifetime": lifetime, "phase": phase, **_runtime_expected()}:
            add("runtime_attestation", "SM121 agent runtime attestation is invalid")
        if rows[offset + 2] != {"event": "server_ready", "backend": "sglang", "fresh_lifetime": lifetime, "phase": phase, "first_inference_is_admission_gate": True, "first_protocol_case": first_case}:
            add("server_ready", "SM121 agent server readiness is invalid")
        if rows[offset + 4] != {"event": "server_stopped", "backend": "sglang", "fresh_lifetime": lifetime}:
            add("server_stopped", "SM121 agent server cleanup is invalid")
        if rows[offset + 5] != {"event": "sm121_agent_lifetime_complete", "fresh_lifetime": lifetime, "phase": phase, "within_timeout": True, "admitted": True}:
            add("lifetime_complete", "SM121 agent lifetime completion is invalid")
    if rows[6] != {"event": "sm121_agent_quality_gate", "fresh_lifetime": 1, "quality_item_count": 4, "quality_admitted": True, "payload_contract_verified": True}:
        add("quality_gate", "SM121 agent quality gate is invalid")
    if rows[12] != {"event": "sm121_agent_tool_gate", "fresh_lifetime": 2, "variant": 0, "scenario_count": 4, "scenario_passes": {case_id: True for case_id in SM121_AGENT_ADMISSION_TOOL_CASE_IDS}, "tools_admitted": True, "payload_contract_verified": True}:
        add("tool_gate", "SM121 agent tool gate is invalid")
    if rows[18] != {"event": "sm121_agent_long_context_gate", "fresh_lifetime": 3, "input_tokenization_verified": True, "context_fit": True, "zero_metric_cache_hits": True, "zero_response_cache_hits": True, "guardrails_clean": True, "long_context_admitted": True, "payload_contract_verified": True}:
        add("long_context_gate", "SM121 agent long-context gate is invalid")
    if rows[0] != {"event": "run_start", "execution_mode": SM121_AGENT_ADMISSION_EXECUTION_MODE, "admission_id": SM121_AGENT_ADMISSION_ID, "profile_id": SM121_AGENT_ADMISSION_PROFILE_ID, "suite_id": SM121_AGENT_ADMISSION_SUITE_ID}:
        add("run_start", "SM121 agent admission start is invalid")
    if rows[-2] != {"event": "measurement_complete", "status": "complete"} or rows[-1] != {"event": "run_complete", "status": "admitted"}:
        add("completion", "SM121 agent admission terminal records are invalid")
    if server_issues:
        add("server_cleanup", "SM121 agent admission retained unsafe server artifacts")
    if any(summary[name] is not True for name in ("parser_static_admitted", "quality_admitted", "tools_admitted", "long_context_admitted")) or any(summary[name] != 3 for name in ("source_static_attestations", "runtime_attestations", "completed_lifetimes")):
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
        errors.append(
            {
                "code": "controller_unimplemented",
                "message": (
                    "SM121 agent admission execution is not implemented; "
                    "a structurally valid record is not accepted as admission"
                ),
            }
        )
    report["status"] = summary["status"]
    report["decision"] = summary["decision"]
    report["error_count"] = len(errors)
    report["ok"] = not errors
    return report
