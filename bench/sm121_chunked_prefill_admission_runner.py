"""Dedicated, non-evidence executor for the prospective SM121 8K profile.

The paired A/B/B/A controller remains the only source of chunked-prefill
performance evidence.  This module freezes a single, exact 8K plan under the
ignored logs tree, runs two fresh server lifetimes, and emits only scalar
admission records.  It intentionally omits all timing observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from types import SimpleNamespace
from typing import Any, Callable
import uuid

from . import runner as base_runner
from .host_safety import HostSafetyError, HostSafetyWatchdog
from .journal import Journal, content_hash, write_json
from .runtime import (
    RuntimeErrorWithContext,
    inspect_sm121_cache_source_digests,
    inspect_sm121_chunked_prefill_runtime_identity,
    recover_owned_sglang,
    request_sm121_cache_semantic_turn,
    save_server_logs,
    settle_sm121_cache_observability_metrics,
)
from .sglang_sm121_cache_semantic import SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS
from .sglang_sm121_chunked_prefill_admission import (
    SM121_CHUNKED_PREFILL_8K_ADMISSION_CELL_TIMEOUT_S,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_EXECUTION_MODE,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_EXACT_RESPONSE,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_HTTP,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_PROMPT_IDS,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_REQUEST_CONTRACT,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_RESPONSE_CONTRACT,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_TRANSPORT,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_GENERIC,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODES,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_ID,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_RUNTIME_EVENT,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_STATIC_EVENT,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE_ID,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_TIMED_CASE_ID,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_T0_EVENT,
    SM121ChunkedPrefill8KAdmissionError,
    derive_sm121_chunked_prefill_8k_admission_t0,
    validate_sm121_chunked_prefill_8k_admission_profile,
    validate_sm121_chunked_prefill_8k_admission_runtime_event,
    sm121_chunked_prefill_8k_admission_receipt,
    validate_sm121_chunked_prefill_8k_admission_receipt,
    validate_sm121_chunked_prefill_8k_admission_static_event,
    validate_sm121_chunked_prefill_8k_admission_suite,
    validate_sm121_chunked_prefill_8k_admission_summary,
    validate_sm121_chunked_prefill_8k_admission_t0_event,
)
from .sglang_sm121_chunked_prefill_performance import (
    SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MAX_TOKENS,
    SM121_CHUNKED_PREFILL_PERFORMANCE_METRIC_FIELDS,
    SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CANDIDATE_PROFILE_ID,
)
from .sglang_sm121_storage import (
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_SOURCE_TREE,
)
from .telemetry import TelemetrySampler


_FAILURE_MESSAGE = "SM121 chunked-prefill 8K admission request failed; details omitted"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ADMISSION_LOGS_ROOT = _REPOSITORY_ROOT / "logs"
_ADMISSION_JSON_MAX_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class _AdmissionReceiptSnapshot:
    """One stable private admission view used for a V3 receipt decision."""

    plan: dict[str, Any]
    suite: SimpleNamespace
    summary: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    server_topology_issues: tuple[str, ...]
    audit_hash: str
    root_identity: tuple[int, ...]
    artifact_identities: tuple[tuple[str, tuple[int, ...]], ...]


class SM121ChunkedPrefill8KAdmissionRequestError(RuntimeError):
    """Public-safe terminal failure for the singleton admission controller."""

    def __init__(
        self,
        failure_code: str = SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_GENERIC,
    ) -> None:
        if (
            not isinstance(failure_code, str)
            or failure_code not in SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODES
        ):
            failure_code = SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_GENERIC
        self.failure_code = failure_code
        super().__init__(_FAILURE_MESSAGE)


def _cold_semantic_failure_code(error: RuntimeErrorWithContext) -> str:
    """Project known runtime failures to scalar-only admission diagnostics."""

    detail = str(error)
    if detail == "SM121 semantic-cache response failed validation":
        return SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_EXACT_RESPONSE
    if detail == "SM121 semantic-cache prompt token count disagrees" or detail.startswith(
        "SM121 semantic-cache prompt IDs "
    ):
        return SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_PROMPT_IDS
    if detail == "SM121 semantic-cache request was rejected":
        return SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_HTTP
    if detail == "SM121 semantic-cache request failed":
        return SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_TRANSPORT
    if detail.startswith("SM121 semantic-cache response ") or detail.startswith(
        "SM121 cache response "
    ):
        return SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_RESPONSE_CONTRACT
    if detail.startswith("SM121 semantic-cache "):
        return SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_REQUEST_CONTRACT
    return SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_GENERIC


def _admission_path(
    path: Path,
    *,
    require_existing: bool,
    allow_logs_root: bool,
    create_logs_root: bool,
) -> Path:
    """Return a real, non-symlink descendant of the private logs tree."""

    logs_root = _ADMISSION_LOGS_ROOT
    if logs_root.is_symlink():
        raise base_runner.PreflightError("SM121 8K admission logs topology is invalid")
    if not logs_root.exists():
        if not create_logs_root:
            raise base_runner.PreflightError("SM121 8K admission logs are unavailable")
        logs_root.mkdir(mode=0o700)
    if not logs_root.is_dir():
        raise base_runner.PreflightError("SM121 8K admission logs topology is invalid")
    candidate = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = candidate.relative_to(logs_root)
    except ValueError as error:
        raise base_runner.PreflightError(
            "SM121 8K admission output must remain under the ignored logs root"
        ) from error
    if candidate == logs_root and not allow_logs_root:
        raise base_runner.PreflightError("SM121 8K admission run location is invalid")
    current = logs_root
    for index, component in enumerate(relative.parts):
        current = current / component
        if current.is_symlink():
            raise base_runner.PreflightError("SM121 8K admission logs topology is invalid")
        if current.exists() and index + 1 < len(relative.parts) and not current.is_dir():
            raise base_runner.PreflightError("SM121 8K admission logs topology is invalid")
    if require_existing:
        if candidate.is_symlink() or not candidate.is_dir():
            raise base_runner.PreflightError("SM121 8K admission run location is invalid")
    elif candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
        raise base_runner.PreflightError("SM121 8K admission output location is invalid")
    return candidate


def _owned_regular_file(path: Path) -> bool:
    """Return whether a planned input is an owned, single-link regular file."""

    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and metadata.st_uid == os.geteuid()
    )


def _owned_directory(path: Path) -> bool:
    """Return whether a path is an owned directory without link traversal."""

    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == os.geteuid()


def _admission_metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Return the stable fields needed to detect a same-path replacement."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_owned_admission_json(path: Path, *, context: str) -> dict[str, Any]:
    """Read one immutable admission record without following a replacement link."""

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
            or before.st_size > _ADMISSION_JSON_MAX_BYTES
        ):
            raise base_runner.PreflightError(f"{context} topology is invalid")
        payload = bytearray()
        while len(payload) < before.st_size:
            chunk = os.pread(
                descriptor,
                min(65_536, before.st_size - len(payload)),
                len(payload),
            )
            if not chunk:
                raise base_runner.PreflightError(f"{context} changed while being read")
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if _admission_metadata_identity(before) != _admission_metadata_identity(after):
            raise base_runner.PreflightError(f"{context} changed while being read")
    finally:
        os.close(descriptor)

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    try:
        value = json.loads(
            bytes(payload).decode("utf-8"), object_pairs_hook=reject_duplicate_pairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise base_runner.PreflightError(f"{context} is invalid") from error
    if type(value) is not dict:
        raise base_runner.PreflightError(f"{context} is invalid")
    return value


def _admission_run_identity(root: Path) -> tuple[int, ...]:
    """Authenticate the private admission leaf before and after a snapshot."""

    try:
        metadata = root.lstat()
    except OSError as error:
        raise base_runner.PreflightError("8K admission run is unavailable") from error
    if (
        root.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise base_runner.PreflightError("8K admission run topology is invalid")
    return _admission_metadata_identity(metadata)


def _require_private_admission_output_root(
    output_root: Path, *, create: bool
) -> None:
    """Create or require the private directory that contains raw artifacts."""

    if output_root.exists() or output_root.is_symlink():
        if not _owned_directory(output_root) or stat.S_IMODE(output_root.lstat().st_mode) & 0o077:
            raise base_runner.PreflightError(
                "SM121 8K admission output directory is not private"
            )
        return
    if not create:
        raise base_runner.PreflightError(
            "SM121 8K admission output directory is unavailable"
        )
    try:
        output_root.mkdir(parents=True, mode=0o700)
        os.chmod(output_root, 0o700)
    except OSError as error:
        raise base_runner.PreflightError(
            "SM121 8K admission output directory is unavailable"
        ) from error
    if not _owned_directory(output_root) or stat.S_IMODE(output_root.lstat().st_mode) != 0o700:
        raise base_runner.PreflightError(
            "SM121 8K admission output directory is not private"
        )


def _require_private_admission_run_directory(run_dir: Path) -> None:
    """Require a private, owned leaf directory for one raw admission run."""

    if (
        not _owned_directory(run_dir)
        or stat.S_IMODE(run_dir.lstat().st_mode) != 0o700
    ):
        raise base_runner.PreflightError(
            "SM121 8K admission run directory is not private"
        )


def _require_admission_plan_inputs(root: Path) -> None:
    if not all(
        _owned_regular_file(root / name) for name in ("plan.json", "inventory.json")
    ):
        raise base_runner.PreflightError("SM121 8K admission plan inputs are invalid")


def _require_safe_admission_server_tree(root: Path) -> None:
    """Reject link or special-file traversal before any key cleanup can run."""

    server_root = root / "server"
    if server_root.is_symlink() or (server_root.exists() and not _owned_directory(server_root)):
        raise base_runner.PreflightError("SM121 8K admission server topology is invalid")
    for fresh_lifetime in (1, 2):
        lifetime_root = server_root / f"lifetime-{fresh_lifetime}"
        api_key_path = lifetime_root / "api-key"
        if lifetime_root.is_symlink() or (
            lifetime_root.exists() and not _owned_directory(lifetime_root)
        ):
            raise base_runner.PreflightError(
                "SM121 8K admission server topology is invalid"
            )
        if api_key_path.is_symlink() or (
            api_key_path.exists() and not _owned_regular_file(api_key_path)
        ):
            raise base_runner.PreflightError(
                "SM121 8K admission server topology is invalid"
            )


def _require_safe_admission_artifacts(root: Path) -> None:
    """Validate pre-existing raw records before reading or cleaning anything."""

    _require_admission_plan_inputs(root)
    for name in ("events.jsonl", "admission.json", "telemetry.jsonl"):
        path = root / name
        if path.is_symlink() or (path.exists() and not _owned_regular_file(path)):
            raise base_runner.PreflightError(
                "SM121 8K admission artifacts are invalid"
            )
    _require_safe_admission_server_tree(root)


def _admission_artifact_identities(
    root: Path, *, names: tuple[str, ...]
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Capture the direct records that must not be replaced during a receipt read."""

    identities: list[tuple[str, tuple[int, ...]]] = []
    for name in names:
        path = root / name
        try:
            metadata = path.lstat()
        except OSError as error:
            raise base_runner.PreflightError(
                "8K admission receipt inputs are unavailable"
            ) from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
        ):
            raise base_runner.PreflightError(
                "8K admission receipt inputs are unavailable"
            )
        identities.append((name, _admission_metadata_identity(metadata)))
    return tuple(identities)


def _admission_server_topology_issues(root: Path) -> tuple[str, ...]:
    """Snapshot terminal server cleanup state without traversing unsafe links."""

    def lstat_or_missing(path: Path) -> os.stat_result | None:
        try:
            return path.lstat()
        except FileNotFoundError:
            return None

    issues: list[str] = []
    server_root = root / "server"
    server_metadata = lstat_or_missing(server_root)
    if server_metadata is None:
        return ()
    if stat.S_ISLNK(server_metadata.st_mode) or not stat.S_ISDIR(server_metadata.st_mode):
        return ("server_topology",)
    for fresh_lifetime in (1, 2):
        lifetime_root = server_root / f"lifetime-{fresh_lifetime}"
        lifetime_metadata = lstat_or_missing(lifetime_root)
        if lifetime_metadata is None:
            continue
        if (
            stat.S_ISLNK(lifetime_metadata.st_mode)
            or not stat.S_ISDIR(lifetime_metadata.st_mode)
        ):
            issues.append("server_topology")
            continue
        api_key_metadata = lstat_or_missing(lifetime_root / "api-key")
        if api_key_metadata is not None:
            issues.append("api_key_residue")
    return tuple(issues)


def _require_fresh_admission_topology(root: Path) -> None:
    """Refuse pre-created write targets before telemetry or a server can start."""

    _require_safe_admission_artifacts(root)
    try:
        entries = {entry.name for entry in root.iterdir()}
    except OSError as error:
        raise base_runner.PreflightError("SM121 8K admission run is unavailable") from error
    if entries != {"plan.json", "inventory.json"}:
        raise base_runner.PreflightError("SM121 8K admission plan is not fresh")


def _recover_incomplete_admission_lifetimes(
    *, model: SimpleNamespace, root: Path
) -> None:
    """Stop only an exact stale admission container and remove its two keys."""

    for fresh_lifetime in (1, 2):
        try:
            action = recover_owned_sglang(
                str(model.run_identity),
                api_key_path=(
                    root / "server" / f"lifetime-{fresh_lifetime}" / "api-key"
                ),
            )
        except Exception:
            raise base_runner.PreflightError(
                "SM121 8K admission recovery failed"
            ) from None
        if action == "different_container_present":
            raise base_runner.PreflightError("SM121 8K admission recovery is blocked")


def create_sm121_chunked_prefill_8k_admission_plan(
    *,
    model: Any,
    suite: Any,
    output_root: Path,
    models_path: Path,
    suite_path: Path,
) -> Path:
    """Freeze one exact 8K admission plan without starting a server."""

    try:
        validate_sm121_chunked_prefill_8k_admission_profile(model)
        validate_sm121_chunked_prefill_8k_admission_suite(suite)
    except SM121ChunkedPrefill8KAdmissionError as error:
        raise RuntimeError("SM121 chunked-prefill 8K admission is unavailable") from error
    try:
        output_target = _admission_path(
            output_root,
            require_existing=False,
            allow_logs_root=False,
            create_logs_root=True,
        )
        _require_private_admission_output_root(output_target, create=True)
    except base_runner.PreflightError as error:
        raise RuntimeError(str(error)) from error
    run_dir = base_runner.create_plan(
        model=model,
        suite=suite,
        results_root=output_target,
        models_path=models_path,
        suite_path=suite_path,
        allow_sm121_chunked_prefill_performance=True,
        run_label="prefill-8k-admission",
    )
    try:
        os.chmod(run_dir, 0o700)
        _require_private_admission_run_directory(run_dir)
    except (OSError, base_runner.PreflightError) as error:
        raise RuntimeError("SM121 8K admission run directory is not private") from error
    return run_dir


def _load_sm121_chunked_prefill_8k_admission_plan(
    run_dir: Path,
) -> tuple[dict[str, Any], SimpleNamespace, SimpleNamespace]:
    """Authenticate a direct, unpaired 8K admission plan before serving."""

    try:
        plan = _read_owned_admission_json(
            run_dir / "plan.json", context="SM121 8K admission plan"
        )
    except base_runner.PreflightError as error:
        raise base_runner.PreflightError("SM121 8K admission plan is unavailable") from error
    if type(plan) is not dict or plan.get("schema_version") != 2:
        raise base_runner.PreflightError("SM121 8K admission plan schema is invalid")
    model_data, suite_data, resolved = (
        plan.get("model"),
        plan.get("suite"),
        plan.get("resolved"),
    )
    if (
        type(model_data) is not dict
        or type(suite_data) is not dict
        or type(resolved) is not dict
    ):
        raise base_runner.PreflightError("SM121 8K admission plan core fields are invalid")
    integrity = plan.get("integrity_hash")
    if not isinstance(integrity, str) or content_hash(
        {key: value for key, value in plan.items() if key != "integrity_hash"},
        len(integrity),
    ) != integrity:
        raise base_runner.PreflightError("SM121 8K admission plan integrity is invalid")
    cases = suite_data.get("cases")
    if not isinstance(cases, list) or any(type(case) is not dict for case in cases):
        raise base_runner.PreflightError("SM121 8K admission plan cases are invalid")
    suite_without_case_ids = {
        **suite_data,
        "cases": [
            {key: value for key, value in case.items() if key != "case_id"}
            for case in cases
        ],
    }
    if plan.get("fingerprint") != content_hash(
        {"model": model_data, "suite": suite_without_case_ids, "resolved": resolved}
    ):
        raise base_runner.PreflightError("SM121 8K admission plan fingerprint is invalid")
    for case in cases:
        case_without_id = {key: value for key, value in case.items() if key != "case_id"}
        expected_case_id = base_runner._canonical_case(
            model_data,
            case_without_id,
            protocol_digest=suite_data.get("protocol_digest"),
        )["case_id"]
        if case.get("case_id") != expected_case_id:
            raise base_runner.PreflightError("SM121 8K admission case identity is invalid")
    if (
        "chunked_prefill_performance_pair" in plan
        or "chunked_prefill_performance_ordinal" in plan
    ):
        raise base_runner.PreflightError("SM121 8K admission plan must be unpaired")
    model = base_runner._namespace(model_data)
    suite = base_runner._namespace(suite_data)
    try:
        validate_sm121_chunked_prefill_8k_admission_profile(model)
        validate_sm121_chunked_prefill_8k_admission_suite(suite)
    except SM121ChunkedPrefill8KAdmissionError as error:
        raise base_runner.PreflightError("SM121 8K admission plan contract is invalid") from error
    local_image = resolved.get("local_image")
    if (
        type(local_image) is not dict
        or set(local_image) != {"docker_image_id", "platform", "source_tree"}
        or local_image.get("docker_image_id") != SM121_STORAGE_LOCAL_IMAGE_ID
        or local_image.get("platform") != SM121_STORAGE_PLATFORM
        or local_image.get("source_tree") != SM121_STORAGE_SOURCE_TREE
    ):
        raise base_runner.PreflightError("SM121 8K admission local image changed")
    run_nonce = plan.get("run_nonce")
    if not isinstance(run_nonce, str) or re.fullmatch(r"[0-9a-f]{32}", run_nonce) is None:
        raise base_runner.PreflightError("SM121 8K admission run nonce is invalid")
    model.resolved_local_image_id = local_image["docker_image_id"]
    model.run_identity = f"{plan['fingerprint']}-{run_nonce}"
    model.chunked_prefill_performance_authorized = True
    return plan, model, suite


def _case_pair(suite: SimpleNamespace) -> tuple[SimpleNamespace, SimpleNamespace]:
    cases = list(getattr(suite, "cases", ()))
    if len(cases) != 2:
        raise base_runner.PreflightError("SM121 8K admission cases are invalid")
    quality, cold_t0 = cases
    if (
        getattr(quality, "id", None) != SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID
        or getattr(cold_t0, "id", None)
        != SM121_CHUNKED_PREFILL_8K_ADMISSION_TIMED_CASE_ID
        or not isinstance(getattr(quality, "case_id", None), str)
        or not isinstance(getattr(cold_t0, "case_id", None), str)
    ):
        raise base_runner.PreflightError("SM121 8K admission cases are invalid")
    return quality, cold_t0


def _static_event(
    *, model: SimpleNamespace, fresh_lifetime: int, phase: str
) -> dict[str, Any]:
    event = {
        "event": SM121_CHUNKED_PREFILL_8K_ADMISSION_STATIC_EVENT,
        "fresh_lifetime": fresh_lifetime,
        "phase": phase,
        "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
        "chunked_prefill_size": 8192,
        **inspect_sm121_cache_source_digests(model),
        **SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
    }
    try:
        validate_sm121_chunked_prefill_8k_admission_static_event(event)
    except SM121ChunkedPrefill8KAdmissionError as error:
        raise SM121ChunkedPrefill8KAdmissionRequestError() from error
    return event


def _runtime_event(
    *, server: Any, fresh_lifetime: int, phase: str
) -> dict[str, Any]:
    event = {
        "event": SM121_CHUNKED_PREFILL_8K_ADMISSION_RUNTIME_EVENT,
        "fresh_lifetime": fresh_lifetime,
        "phase": phase,
        **inspect_sm121_chunked_prefill_runtime_identity(server),
    }
    try:
        validate_sm121_chunked_prefill_8k_admission_runtime_event(event)
    except SM121ChunkedPrefill8KAdmissionError as error:
        raise SM121ChunkedPrefill8KAdmissionRequestError() from error
    return event


def _remaining_s(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise SM121ChunkedPrefill8KAdmissionRequestError()
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


def _run_lifetime(
    *,
    run_dir: Path,
    workspace: Path,
    model: SimpleNamespace,
    fresh_lifetime: int,
    phase: str,
    case: SimpleNamespace,
    journal: Journal,
    telemetry: TelemetrySampler,
    operation: Callable[[Any, float, HostSafetyWatchdog | None], None],
) -> None:
    """Run one exact fresh server lifetime and always tear it down."""

    started = time.monotonic()
    deadline = started + SM121_CHUNKED_PREFILL_8K_ADMISSION_CELL_TIMEOUT_S
    server = None
    watchdog: HostSafetyWatchdog | None = None
    terminal_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        _remaining_s(deadline)
        base_runner._preflight(model)
        _remaining_s(deadline)
        journal.append(
            _static_event(
                model=model,
                fresh_lifetime=fresh_lifetime,
                phase=phase,
            )
        )
        watchdog = base_runner._host_safety_watchdog(model)
        if watchdog is not None:
            watchdog.start()
        telemetry.set_phase(f"chunked_prefill_8k_admission_{phase}_start")
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
                run_dir / "server" / f"lifetime-{fresh_lifetime}" / "server.log"
            ),
            **callbacks,
        )
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
                "backend": server.backend,
                "fresh_lifetime": fresh_lifetime,
                "phase": phase,
                "first_inference_is_case": True,
                "case_id": case.case_id,
            }
        )
        operation(server, deadline, watchdog)
        _abort_check(watchdog=watchdog, deadline=deadline)
    except BaseException as error:
        terminal_error = watchdog.failure if watchdog is not None and watchdog.failure else error
    finally:
        telemetry.set_phase(f"chunked_prefill_8k_admission_{phase}_stop")
        if server is not None:
            if watchdog is not None and watchdog.tripped:
                try:
                    base_runner._retry_host_safety_interrupt_if_needed(server, watchdog)
                    base_runner._record_host_safety_interrupt_failure(
                        journal, watchdog, stage=f"chunked_prefill_8k_admission_{phase}"
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
                    run_dir / "server" / f"lifetime-{fresh_lifetime}" / "server.log",
                )
            except BaseException as error:
                cleanup_error = cleanup_error or error
            try:
                server.stop()
                journal.append(
                    {
                        "event": "server_stopped",
                        "backend": server.backend,
                        "fresh_lifetime": fresh_lifetime,
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
                            "fresh_lifetime": fresh_lifetime,
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
    within_timeout = time.monotonic() - started <= SM121_CHUNKED_PREFILL_8K_ADMISSION_CELL_TIMEOUT_S
    journal.append(
        {
            "event": "sm121_chunked_prefill_8k_admission_lifetime_complete",
            "fresh_lifetime": fresh_lifetime,
            "phase": phase,
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
                SM121ChunkedPrefill8KAdmissionRequestError,
            ),
        ):
            raise terminal_error
        raise SM121ChunkedPrefill8KAdmissionRequestError() from None
    if not within_timeout:
        raise SM121ChunkedPrefill8KAdmissionRequestError()


def _run_quality_case(
    *,
    server: Any,
    deadline: float,
    watchdog: HostSafetyWatchdog | None,
    model: SimpleNamespace,
    case: SimpleNamespace,
    journal: Journal,
    telemetry: TelemetrySampler,
) -> None:
    if len(base_runner._QUALITY_ITEMS) != SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT:
        raise base_runner.PreflightError("SM121 8K admission quality item count changed")
    journal.append(
        {
            "event": "sm121_chunked_prefill_8k_admission_quality_case_start",
            "fresh_lifetime": 1,
            "case_id": case.case_id,
        }
    )
    telemetry.set_phase("chunked_prefill_8k_admission_quality_case")
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
            "event": "sm121_chunked_prefill_8k_admission_quality_case_complete",
            "fresh_lifetime": 1,
            "case_id": case.case_id,
            "quality_admitted": True,
            "item_count": len(base_runner._QUALITY_ITEMS),
        }
    )


def _t0_event(
    *,
    case: SimpleNamespace,
    result: dict[str, Any],
    before: dict[str, Any],
    before_polls: int,
    before_settled: bool,
    after: dict[str, Any],
    after_polls: int,
    after_settled: bool,
) -> dict[str, Any]:
    prompt_ids = result.pop("private_prompt_token_ids", None)
    if (
        not isinstance(prompt_ids, tuple)
        or not prompt_ids
        or any(type(token) is not int or token < 0 for token in prompt_ids)
    ):
        raise SM121ChunkedPrefill8KAdmissionRequestError(
            SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_COLD_PROMPT_IDS
        )
    event: dict[str, Any] = {
        "event": SM121_CHUNKED_PREFILL_8K_ADMISSION_T0_EVENT,
        "fresh_lifetime": 2,
        "case_id": case.case_id,
        "protocol_case_id": SM121_CHUNKED_PREFILL_8K_ADMISSION_TIMED_CASE_ID,
        "cache_details_requested": True,
        "prompt_token_ids_requested": True,
        "prompt_token_ids_verified": True,
        "streaming": False,
        "thinking_disabled": True,
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "reasoning_tokens": result["reasoning_tokens"],
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
        "cold_t0_admitted": False,
        "cold_t0_basis": "pending",
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
        admitted, basis = derive_sm121_chunked_prefill_8k_admission_t0(event)
    except (KeyError, TypeError, ValueError, SM121ChunkedPrefill8KAdmissionError) as error:
        raise SM121ChunkedPrefill8KAdmissionRequestError() from error
    event["cold_t0_admitted"] = admitted
    event["cold_t0_basis"] = basis
    try:
        validate_sm121_chunked_prefill_8k_admission_t0_event(event)
    except SM121ChunkedPrefill8KAdmissionError as error:
        raise SM121ChunkedPrefill8KAdmissionRequestError() from error
    return event


def _run_cold_t0_case(
    *,
    server: Any,
    deadline: float,
    watchdog: HostSafetyWatchdog | None,
    model: SimpleNamespace,
    case: SimpleNamespace,
    journal: Journal,
    telemetry: TelemetrySampler,
) -> None:
    from .sm121_chunked_prefill_runner import _EXPECTED_RESPONSES, _messages

    journal.append(
        {
            "event": "sm121_chunked_prefill_8k_admission_cold_t0_case_start",
            "fresh_lifetime": 2,
            "case_id": case.case_id,
        }
    )
    telemetry.set_phase("chunked_prefill_8k_admission_cold_t0_case")
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
    try:
        result = request_sm121_cache_semantic_turn(
            server,
            served_name=model.served_name,
            messages=_messages()[0],
            expected_response=_EXPECTED_RESPONSES[0],
            max_tokens=int(case.max_output_tokens),
            timeout_s=min(900.0, _remaining_s(deadline)),
        )
    except RuntimeErrorWithContext as error:
        raise SM121ChunkedPrefill8KAdmissionRequestError(
            _cold_semantic_failure_code(error)
        ) from None
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
    event = _t0_event(
        case=case,
        result=result,
        before=before,
        before_polls=before_polls,
        before_settled=before_settled,
        after=after,
        after_polls=after_polls,
        after_settled=after_settled,
    )
    journal.append(event)
    if event["cold_t0_admitted"] is not True:
        raise SM121ChunkedPrefill8KAdmissionRequestError()
    journal.append(
        {
            "event": "sm121_chunked_prefill_8k_admission_cold_t0_case_complete",
            "fresh_lifetime": 2,
            "case_id": case.case_id,
            "cold_t0_admitted": True,
        }
    )


def _summary(
    *,
    quality_admitted: bool,
    cold_t0_admitted: bool,
    quality_within_timeout: bool,
    cold_t0_within_timeout: bool,
    static_attestations: int,
    runtime_attestations: int,
    terminal_stage: str,
    failure_code: str,
) -> dict[str, Any]:
    complete = (
        quality_admitted
        and cold_t0_admitted
        and quality_within_timeout
        and cold_t0_within_timeout
        and static_attestations == 2
        and runtime_attestations == 2
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "admission_id": SM121_CHUNKED_PREFILL_8K_ADMISSION_ID,
        "execution_mode": SM121_CHUNKED_PREFILL_8K_ADMISSION_EXECUTION_MODE,
        "status": "complete" if complete else "partial",
        "decision": "admitted" if complete else "blocked",
        "terminal_stage": "complete" if complete else terminal_stage,
        "failure_code": None if complete else failure_code,
        "profile_id": SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CANDIDATE_PROFILE_ID,
        "suite_id": SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE_ID,
        "quality_admitted": quality_admitted,
        "cold_t0_admitted": cold_t0_admitted,
        "quality_within_timeout": quality_within_timeout,
        "cold_t0_within_timeout": cold_t0_within_timeout,
        "static_attestations": static_attestations,
        "runtime_attestations": runtime_attestations,
    }
    summary["integrity_hash"] = content_hash(summary, 64)
    validate_sm121_chunked_prefill_8k_admission_summary(summary)
    return summary


def execute_sm121_chunked_prefill_8k_admission(
    run_dir: Path, *, workspace: Path
) -> dict[str, Any]:
    """Run the one non-resumable quality-plus-cold-T0 8K admission check."""

    root = _admission_path(
        run_dir,
        require_existing=True,
        allow_logs_root=False,
        create_logs_root=False,
    )
    _require_private_admission_run_directory(root)
    _require_admission_plan_inputs(root)
    _require_private_admission_output_root(root.parent, create=False)
    plan, model, suite = _load_sm121_chunked_prefill_8k_admission_plan(root)
    lock_path = base_runner.results_lock_path(workspace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another SparkBench run holds the benchmark lock") from error
        summary_path = root / "admission.json"
        journal = Journal(root / "events.jsonl")
        _require_safe_admission_artifacts(root)
        if summary_path.exists():
            raise base_runner.PreflightError(
                "SM121 8K admission is non-resumable; freeze a new plan"
            )
        if journal.events():
            _recover_incomplete_admission_lifetimes(model=model, root=root)
            raise base_runner.PreflightError(
                "SM121 8K admission is non-resumable; freeze a new plan"
            )
        _require_fresh_admission_topology(root)
        quality_case, cold_t0_case = _case_pair(suite)
        if (
            set(quality_case.requires) - set(model.tasks)
            or set(cold_t0_case.requires) - set(model.tasks)
            or SM121_CHUNKED_PREFILL_PERFORMANCE_COLD_INPUT_MAX_TOKENS
            + int(cold_t0_case.max_output_tokens)
            + 1024
            > int(model.max_context)
        ):
            raise base_runner.PreflightError(
                "SM121 8K admission context is insufficient"
            )
        journal.append(
            {
                "event": "run_start",
                "execution_mode": SM121_CHUNKED_PREFILL_8K_ADMISSION_EXECUTION_MODE,
                "admission_id": SM121_CHUNKED_PREFILL_8K_ADMISSION_ID,
                "profile_id": model.id,
                "suite_id": suite.id,
            }
        )
        journal.append({"event": "measurement_started"})
        telemetry = TelemetrySampler(root / "telemetry.jsonl")
        telemetry.start()
        quality_admitted = False
        cold_t0_admitted = False
        quality_within_timeout = False
        cold_t0_within_timeout = False
        terminal_stage = "preflight"
        failure_code = SM121_CHUNKED_PREFILL_8K_ADMISSION_FAILURE_CODE_GENERIC
        try:
            terminal_stage = "quality_lifetime"
            _run_lifetime(
                run_dir=root,
                workspace=workspace,
                model=model,
                fresh_lifetime=1,
                phase="quality",
                case=quality_case,
                journal=journal,
                telemetry=telemetry,
                operation=lambda server, deadline, watchdog: _run_quality_case(
                    server=server,
                    deadline=deadline,
                    watchdog=watchdog,
                    model=model,
                    case=quality_case,
                    journal=journal,
                    telemetry=telemetry,
                ),
            )
            quality_admitted = True
            quality_within_timeout = True
            terminal_stage = "cold_t0_lifetime"
            _run_lifetime(
                run_dir=root,
                workspace=workspace,
                model=model,
                fresh_lifetime=2,
                phase="cold_t0",
                case=cold_t0_case,
                journal=journal,
                telemetry=telemetry,
                operation=lambda server, deadline, watchdog: _run_cold_t0_case(
                    server=server,
                    deadline=deadline,
                    watchdog=watchdog,
                    model=model,
                    case=cold_t0_case,
                    journal=journal,
                    telemetry=telemetry,
                ),
            )
            cold_t0_admitted = True
            cold_t0_within_timeout = True
        except BaseException as error:
            if isinstance(error, HostSafetyError):
                base_runner._record_host_safety_breach(
                    journal, error, stage=terminal_stage
                )
            safe_error = (
                error
                if isinstance(error, SM121ChunkedPrefill8KAdmissionRequestError)
                else SM121ChunkedPrefill8KAdmissionRequestError()
            )
            failure_code = safe_error.failure_code
            base_runner._record_run_aborted(journal, safe_error, stage=terminal_stage)
        finally:
            telemetry.stop()
        events = journal.events()
        summary = _summary(
            quality_admitted=quality_admitted,
            cold_t0_admitted=cold_t0_admitted,
            quality_within_timeout=quality_within_timeout,
            cold_t0_within_timeout=cold_t0_within_timeout,
            static_attestations=sum(
                1
                for event in events
                if event.get("event") == SM121_CHUNKED_PREFILL_8K_ADMISSION_STATIC_EVENT
            ),
            runtime_attestations=sum(
                1
                for event in events
                if event.get("event") == SM121_CHUNKED_PREFILL_8K_ADMISSION_RUNTIME_EVENT
            ),
            terminal_stage=terminal_stage,
            failure_code=failure_code,
        )
        journal.append(
            {
                "event": "measurement_complete",
                "status": summary["status"],
            }
        )
        journal.append({"event": "run_complete", "status": summary["decision"]})
        write_json(summary_path, summary)
        return summary


def _audit_sm121_chunked_prefill_8k_admission_snapshot(
    *,
    plan: dict[str, Any],
    suite: SimpleNamespace,
    events: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    summary: dict[str, Any],
    server_topology_issues: tuple[str, ...],
) -> dict[str, Any]:
    """Audit one already-read admission snapshot without touching its paths."""

    report: dict[str, Any] = {
        "schema_version": 1,
        "admission_id": SM121_CHUNKED_PREFILL_8K_ADMISSION_ID,
        "read_only": True,
        "ok": False,
        "errors": [],
    }

    def add(code: str, message: str) -> None:
        report["errors"].append({"code": code, "message": message})

    quality_case, cold_t0_case = _case_pair(suite)

    def without_timestamp(event: object) -> dict[str, object] | None:
        if type(event) is not dict or not isinstance(event.get("timestamp"), str):
            return None
        return {key: value for key, value in event.items() if key != "timestamp"}

    def require_one(
        event_name: str, expected: dict[str, object], code: str, message: str
    ) -> None:
        matching = [event for event in events if event.get("event") == event_name]
        if len(matching) != 1 or without_timestamp(matching[0]) != expected:
            add(code, message)

    expected_event_names = (
        "run_start",
        "measurement_started",
        SM121_CHUNKED_PREFILL_8K_ADMISSION_STATIC_EVENT,
        SM121_CHUNKED_PREFILL_8K_ADMISSION_RUNTIME_EVENT,
        "server_ready",
        "sm121_chunked_prefill_8k_admission_quality_case_start",
        "sm121_chunked_prefill_8k_admission_quality_case_complete",
        "server_stopped",
        "sm121_chunked_prefill_8k_admission_lifetime_complete",
        SM121_CHUNKED_PREFILL_8K_ADMISSION_STATIC_EVENT,
        SM121_CHUNKED_PREFILL_8K_ADMISSION_RUNTIME_EVENT,
        "server_ready",
        "sm121_chunked_prefill_8k_admission_cold_t0_case_start",
        SM121_CHUNKED_PREFILL_8K_ADMISSION_T0_EVENT,
        "sm121_chunked_prefill_8k_admission_cold_t0_case_complete",
        "server_stopped",
        "sm121_chunked_prefill_8k_admission_lifetime_complete",
        "measurement_complete",
        "run_complete",
    )
    if tuple(event.get("event") for event in events) != expected_event_names:
        add("event_topology", "8K admission journal topology is invalid")
    if any(without_timestamp(event) is None for event in events):
        add("event_timestamp", "8K admission journal timestamp is invalid")
    require_one(
        "run_start",
        {
            "event": "run_start",
            "execution_mode": SM121_CHUNKED_PREFILL_8K_ADMISSION_EXECUTION_MODE,
            "admission_id": SM121_CHUNKED_PREFILL_8K_ADMISSION_ID,
            "profile_id": SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CANDIDATE_PROFILE_ID,
            "suite_id": SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE_ID,
        },
        "run_start",
        "8K admission run_start is invalid",
    )
    require_one(
        "measurement_started",
        {"event": "measurement_started"},
        "measurement_started",
        "8K admission measurement start is invalid",
    )
    for event_name, validator in (
        (
            SM121_CHUNKED_PREFILL_8K_ADMISSION_STATIC_EVENT,
            validate_sm121_chunked_prefill_8k_admission_static_event,
        ),
        (
            SM121_CHUNKED_PREFILL_8K_ADMISSION_RUNTIME_EVENT,
            validate_sm121_chunked_prefill_8k_admission_runtime_event,
        ),
        (
            SM121_CHUNKED_PREFILL_8K_ADMISSION_T0_EVENT,
            validate_sm121_chunked_prefill_8k_admission_t0_event,
        ),
    ):
        matching = [event for event in events if event.get("event") == event_name]
        expected_count = 1 if event_name == SM121_CHUNKED_PREFILL_8K_ADMISSION_T0_EVENT else 2
        if len(matching) != expected_count:
            add("attestation_count", "8K admission attestation count is invalid")
        for event in matching:
            try:
                validator(event)
            except SM121ChunkedPrefill8KAdmissionError:
                add("invalid_attestation", "8K admission attestation is invalid")
                break
    expected_lifetimes = (
        (1, "quality", quality_case.case_id),
        (2, "cold_t0", cold_t0_case.case_id),
    )
    for fresh_lifetime, phase, case_id in expected_lifetimes:
        ready = [
            event
            for event in events
            if event.get("event") == "server_ready"
            and event.get("fresh_lifetime") == fresh_lifetime
        ]
        stopped = [
            event
            for event in events
            if event.get("event") == "server_stopped"
            and event.get("fresh_lifetime") == fresh_lifetime
        ]
        complete = [
            event
            for event in events
            if event.get("event")
            == "sm121_chunked_prefill_8k_admission_lifetime_complete"
            and event.get("fresh_lifetime") == fresh_lifetime
        ]
        expected_ready = {
            "event": "server_ready",
            "backend": "sglang",
            "fresh_lifetime": fresh_lifetime,
            "phase": phase,
            "first_inference_is_case": True,
            "case_id": case_id,
        }
        if len(ready) != 1 or without_timestamp(ready[0]) != expected_ready:
            add("server_ready", "8K admission server readiness is invalid")
        expected_stopped = {
            "event": "server_stopped",
            "backend": "sglang",
            "fresh_lifetime": fresh_lifetime,
        }
        if len(stopped) != 1 or without_timestamp(stopped[0]) != expected_stopped:
            add("server_stopped", "8K admission server cleanup is invalid")
        expected_complete = {
            "event": "sm121_chunked_prefill_8k_admission_lifetime_complete",
            "fresh_lifetime": fresh_lifetime,
            "phase": phase,
            "within_timeout": True,
            "admitted": True,
        }
        if len(complete) != 1 or without_timestamp(complete[0]) != expected_complete:
            add("lifetime_complete", "8K admission lifetime completion is invalid")

    indexed_lifecycle = (
        (
            2,
            SM121_CHUNKED_PREFILL_8K_ADMISSION_STATIC_EVENT,
            1,
            "quality",
            None,
        ),
        (
            3,
            SM121_CHUNKED_PREFILL_8K_ADMISSION_RUNTIME_EVENT,
            1,
            "quality",
            None,
        ),
        (4, "server_ready", 1, "quality", quality_case.case_id),
        (7, "server_stopped", 1, None, None),
        (
            8,
            "sm121_chunked_prefill_8k_admission_lifetime_complete",
            1,
            "quality",
            None,
        ),
        (
            9,
            SM121_CHUNKED_PREFILL_8K_ADMISSION_STATIC_EVENT,
            2,
            "cold_t0",
            None,
        ),
        (
            10,
            SM121_CHUNKED_PREFILL_8K_ADMISSION_RUNTIME_EVENT,
            2,
            "cold_t0",
            None,
        ),
        (11, "server_ready", 2, "cold_t0", cold_t0_case.case_id),
        (15, "server_stopped", 2, None, None),
        (
            16,
            "sm121_chunked_prefill_8k_admission_lifetime_complete",
            2,
            "cold_t0",
            None,
        ),
    )
    for index, event_name, fresh_lifetime, phase, case_id in indexed_lifecycle:
        event = events[index] if index < len(events) else {}
        if (
            event.get("event") != event_name
            or event.get("fresh_lifetime") != fresh_lifetime
            or (phase is not None and event.get("phase") != phase)
            or (case_id is not None and event.get("case_id") != case_id)
        ):
            add("lifecycle_order", "8K admission lifetime order is invalid")
            break
    require_one(
        "sm121_chunked_prefill_8k_admission_quality_case_start",
        {
            "event": "sm121_chunked_prefill_8k_admission_quality_case_start",
            "fresh_lifetime": 1,
            "case_id": quality_case.case_id,
        },
        "quality_start",
        "8K admission quality start is invalid",
    )
    require_one(
        "sm121_chunked_prefill_8k_admission_quality_case_complete",
        {
            "event": "sm121_chunked_prefill_8k_admission_quality_case_complete",
            "fresh_lifetime": 1,
            "case_id": quality_case.case_id,
            "quality_admitted": True,
            "item_count": SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT,
        },
        "quality_gate",
        "8K admission quality gate is invalid",
    )
    require_one(
        "sm121_chunked_prefill_8k_admission_cold_t0_case_start",
        {
            "event": "sm121_chunked_prefill_8k_admission_cold_t0_case_start",
            "fresh_lifetime": 2,
            "case_id": cold_t0_case.case_id,
        },
        "cold_t0_start",
        "8K admission cold T0 start is invalid",
    )
    require_one(
        "sm121_chunked_prefill_8k_admission_cold_t0_case_complete",
        {
            "event": "sm121_chunked_prefill_8k_admission_cold_t0_case_complete",
            "fresh_lifetime": 2,
            "case_id": cold_t0_case.case_id,
            "cold_t0_admitted": True,
        },
        "cold_t0",
        "8K admission cold T0 gate is invalid",
    )
    require_one(
        "measurement_complete",
        {"event": "measurement_complete", "status": "complete"},
        "measurement_complete",
        "8K admission measurement completion is invalid",
    )
    require_one(
        "run_complete",
        {"event": "run_complete", "status": "admitted"},
        "run_complete",
        "8K admission terminal completion is invalid",
    )
    for issue in server_topology_issues:
        if issue == "server_topology":
            add("server_topology", "8K admission server artifacts are invalid")
        elif issue == "api_key_residue":
            add("api_key_residue", "8K admission retained an ephemeral API key")
        else:
            add("server_topology", "8K admission server artifacts are invalid")
    if summary["status"] != "complete" or summary["decision"] != "admitted":
        add("summary_decision", "8K admission did not reach admission")
    if (
        summary["quality_admitted"] is not True
        or summary["cold_t0_admitted"] is not True
        or summary["static_attestations"] != 2
        or summary["runtime_attestations"] != 2
    ):
        add("summary_attestations", "8K admission summary is incomplete")
    report["status"] = summary["status"]
    report["decision"] = summary["decision"]
    report["error_count"] = len(report["errors"])
    report["ok"] = report["error_count"] == 0
    return report


def audit_sm121_chunked_prefill_8k_admission(run_dir: Path) -> dict[str, Any]:
    """Read one terminal 8K admission record without changing local state."""

    report: dict[str, Any] = {
        "schema_version": 1,
        "admission_id": SM121_CHUNKED_PREFILL_8K_ADMISSION_ID,
        "read_only": True,
        "ok": False,
        "errors": [],
    }

    def add(code: str, message: str) -> None:
        report["errors"].append({"code": code, "message": message})

    try:
        root = _admission_path(
            run_dir,
            require_existing=True,
            allow_logs_root=False,
            create_logs_root=False,
        )
        _require_private_admission_run_directory(root)
        _require_admission_plan_inputs(root)
        _require_private_admission_output_root(root.parent, create=False)
    except (OSError, base_runner.PreflightError):
        add("invalid_location", "8K admission run location is invalid")
        report["error_count"] = len(report["errors"])
        return report
    events_path = root / "events.jsonl"
    summary_path = root / "admission.json"
    if not _owned_regular_file(events_path) or not _owned_regular_file(summary_path):
        add("missing_record", "8K admission journal or summary is unavailable")
        report["error_count"] = len(report["errors"])
        return report
    try:
        plan, _model, suite = _load_sm121_chunked_prefill_8k_admission_plan(root)
    except (OSError, ValueError, base_runner.PreflightError):
        add("invalid_plan", "8K admission plan is invalid")
        plan = None
        suite = None
    try:
        events = Journal(events_path).strict_events()
    except (OSError, ValueError):
        add("invalid_journal", "8K admission journal is invalid")
        events = []
    try:
        summary = _read_owned_admission_json(
            summary_path, context="SM121 8K admission summary"
        )
        validate_sm121_chunked_prefill_8k_admission_summary(summary)
    except (base_runner.PreflightError, SM121ChunkedPrefill8KAdmissionError):
        add("invalid_summary", "8K admission summary is invalid")
        summary = None
    if plan is None or suite is None or summary is None:
        report["error_count"] = len(report["errors"])
        return report
    try:
        server_topology_issues = _admission_server_topology_issues(root)
    except OSError:
        server_topology_issues = ("server_topology",)
    return _audit_sm121_chunked_prefill_8k_admission_snapshot(
        plan=plan,
        suite=suite,
        events=events,
        summary=summary,
        server_topology_issues=server_topology_issues,
    )


def _receipt_snapshot(
    root: Path,
) -> _AdmissionReceiptSnapshot:
    """Read one strict, timestamp-free scalar snapshot for receipt issuance."""

    artifact_names = ("plan.json", "inventory.json", "admission.json", "events.jsonl")
    root_identity = _admission_run_identity(root)
    _require_private_admission_run_directory(root)
    _require_admission_plan_inputs(root)
    _require_private_admission_output_root(root.parent, create=False)
    artifact_identities = _admission_artifact_identities(root, names=artifact_names)
    plan, _model, suite = _load_sm121_chunked_prefill_8k_admission_plan(root)
    summary_path = root / "admission.json"
    events_path = root / "events.jsonl"
    summary = _read_owned_admission_json(
        summary_path, context="8K admission receipt summary"
    )
    validate_sm121_chunked_prefill_8k_admission_summary(summary)
    events = tuple(Journal(events_path).strict_events())
    server_topology_issues = _admission_server_topology_issues(root)
    terminal_journal = [
        {key: value for key, value in event.items() if key != "timestamp"}
        for event in events
    ]
    if (
        _admission_run_identity(root) != root_identity
        or _admission_artifact_identities(root, names=artifact_names)
        != artifact_identities
    ):
        raise base_runner.PreflightError("8K admission receipt changed while being read")
    audit_hash = content_hash(
        {
            "domain": "sm121-chunked-prefill-8k-admission-audit-v2",
            "plan_integrity_hash": plan["integrity_hash"],
            "summary_integrity_hash": summary["integrity_hash"],
            "terminal_journal": terminal_journal,
            "server_topology_issues": server_topology_issues,
        },
        64,
    )
    return _AdmissionReceiptSnapshot(
        plan=plan,
        suite=suite,
        summary=summary,
        events=events,
        server_topology_issues=server_topology_issues,
        audit_hash=audit_hash,
        root_identity=root_identity,
        artifact_identities=artifact_identities,
    )


def load_verified_sm121_chunked_prefill_8k_admission_receipt(
    run_dir: Path,
) -> dict[str, object]:
    """Re-audit and project one private admission without retaining its path."""

    try:
        root = _admission_path(
            run_dir,
            require_existing=True,
            allow_logs_root=False,
            create_logs_root=False,
        )
        before = _receipt_snapshot(root)
        report = _audit_sm121_chunked_prefill_8k_admission_snapshot(
            plan=before.plan,
            suite=before.suite,
            events=before.events,
            summary=before.summary,
            server_topology_issues=before.server_topology_issues,
        )
        if report.get("ok") is not True:
            raise base_runner.PreflightError("8K admission audit is invalid")
        after = _receipt_snapshot(root)
        if (
            before.audit_hash != after.audit_hash
            or before.root_identity != after.root_identity
            or before.artifact_identities != after.artifact_identities
        ):
            raise base_runner.PreflightError("8K admission receipt changed during audit")
        model = before.plan["model"]
        resolved = before.plan["resolved"]
        local_image = resolved.get("local_image")
        if type(model) is not dict or type(local_image) is not dict:
            raise base_runner.PreflightError("8K admission receipt inputs are invalid")
        receipt = sm121_chunked_prefill_8k_admission_receipt(
            before.summary,
            admission_plan_integrity_hash=before.plan["integrity_hash"],
            admission_model_contract_sha256=content_hash(
                {
                    "domain": "sm121-chunked-prefill-v3-candidate-model-v1",
                    "value": model,
                },
                64,
            ),
            admission_local_image_contract_sha256=content_hash(
                {
                    "domain": "sm121-chunked-prefill-v3-local-image-v1",
                    "value": local_image,
                },
                64,
            ),
            admission_audit_sha256=before.audit_hash,
        )
        validate_sm121_chunked_prefill_8k_admission_receipt(receipt)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        base_runner.PreflightError,
        SM121ChunkedPrefill8KAdmissionError,
    ) as error:
        raise base_runner.PreflightError(
            "SM121 chunked-prefill V3 admission receipt is invalid"
        ) from error
    return receipt
