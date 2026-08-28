"""Fail-closed ownership and cleanup for autoresearch cell workers.

The worker state is deliberately small and scalar-only.  In particular, it
never records the child command, environment, output, or other benchmark
payloads.  A persisted Linux ``/proc`` start time prevents a restarted
controller from signaling a process that merely reused the recorded PID.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time
from typing import Any, Literal, TypeAlias


WORKER_STATE_SCHEMA_VERSION = 1
WORKER_STATE_FILENAME = "worker.json"
DEFAULT_POLL_INTERVAL_S = 0.05
MAX_WORKER_STATE_BYTES = 4_096

_RUN_NONCE_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
_WORKER_STATE_KEYS = frozenset(
    {"schema_version", "pid", "pgid", "start_ticks", "run_nonce"}
)

ProcStatReader: TypeAlias = Callable[[int], str]
ProcessGroupSignaler: TypeAlias = Callable[[int, int], None]
ProcessSignaler: TypeAlias = Callable[[int, int], None]
MonotonicClock: TypeAlias = Callable[[], float]
Sleeper: TypeAlias = Callable[[float], None]
PopenFactory: TypeAlias = Callable[..., subprocess.Popen[Any]]


class WorkerLifecycleError(RuntimeError):
    """A stable, fail-closed worker ownership or cleanup failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        identity: WorkerIdentity | None = None,
        cleanup: WorkerCleanupResult | None = None,
    ) -> None:
        self.code = code
        self.identity = identity
        self.cleanup = cleanup
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class WorkerIdentity:
    """The durable identity of one new-session child process."""

    pid: int
    pgid: int
    start_ticks: int
    run_nonce: str

    def to_mapping(self) -> dict[str, int | str]:
        return {
            "schema_version": WORKER_STATE_SCHEMA_VERSION,
            "pid": self.pid,
            "pgid": self.pgid,
            "start_ticks": self.start_ticks,
            "run_nonce": self.run_nonce,
        }


@dataclass(frozen=True, slots=True)
class ProcessSnapshot:
    """The process identity fields parsed from one ``/proc/<pid>/stat`` read."""

    pid: int
    pgid: int
    start_ticks: int


CleanupOutcome = Literal[
    "completed",
    "no_state",
    "already_exited",
    "interrupted",
    "killed",
]


@dataclass(frozen=True, slots=True)
class WorkerCleanupResult:
    """A certified terminal outcome for a worker process group."""

    outcome: CleanupOutcome
    identity: WorkerIdentity | None
    return_code: int | None
    sigint_sent: bool
    sigkill_sent: bool
    process_lookup_race: bool
    state_removed: bool


@dataclass(frozen=True, slots=True)
class WorkerRunResult:
    """The bounded result of a fresh worker invocation."""

    return_code: int
    timed_out: bool
    cleanup: WorkerCleanupResult


def worker_state_path(cell_run_dir: Path) -> Path:
    """Return the per-cell worker state path beside the frozen run artifacts."""

    return cell_run_dir / WORKER_STATE_FILENAME


def read_proc_stat(pid: int) -> str:
    """Read the Linux process identity record for ``pid``."""

    return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")


def parse_proc_stat(raw: str) -> ProcessSnapshot:
    """Parse PID, process-group ID, and start ticks from Linux proc stat text."""

    if not isinstance(raw, str):
        raise WorkerLifecycleError("proc_stat_malformed", "proc stat was not text")
    open_paren = raw.find("(")
    close_paren = raw.rfind(")")
    if open_paren <= 0 or close_paren <= open_paren:
        raise WorkerLifecycleError(
            "proc_stat_malformed", "proc stat has no bounded command field"
        )
    try:
        pid = int(raw[:open_paren].strip(), 10)
    except ValueError as error:
        raise WorkerLifecycleError(
            "proc_stat_malformed", "proc stat PID is not an integer"
        ) from error
    fields = raw[close_paren + 1 :].split()
    # fields[0] is kernel field 3 (state), so pgrp is index 2 and the
    # process start time (kernel field 22) is index 19.
    if len(fields) <= 19 or len(fields[0]) != 1:
        raise WorkerLifecycleError(
            "proc_stat_malformed", "proc stat is missing identity fields"
        )
    try:
        pgid = int(fields[2], 10)
        start_ticks = int(fields[19], 10)
    except ValueError as error:
        raise WorkerLifecycleError(
            "proc_stat_malformed", "proc stat identity is not integral"
        ) from error
    if pid <= 0 or pgid <= 0 or start_ticks <= 0:
        raise WorkerLifecycleError(
            "proc_stat_malformed", "proc stat identity is not positive"
        )
    return ProcessSnapshot(pid=pid, pgid=pgid, start_ticks=start_ticks)


def _validate_run_nonce(run_nonce: str) -> None:
    if not isinstance(run_nonce, str) or _RUN_NONCE_PATTERN.fullmatch(run_nonce) is None:
        raise WorkerLifecycleError(
            "run_nonce_invalid", "worker run nonce must be 32 lowercase hex characters"
        )


def _validate_positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkerLifecycleError(
            "worker_state_malformed", f"worker state {field} must be positive"
        )
    return value


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_state_absent(path: Path) -> None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as error:
        raise WorkerLifecycleError(
            "worker_state_unreadable", "worker state existence could not be certified"
        ) from error
    raise WorkerLifecycleError(
        "worker_state_exists",
        "worker state already exists; recover it before launching another worker",
    )


def _persist_worker_identity(path: Path, identity: WorkerIdentity) -> None:
    """Create, flush, and directory-sync a mode-0600 ownership record."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise WorkerLifecycleError(
            "worker_state_write_failed",
            "worker ownership directory could not be prepared",
            identity=identity,
        ) from error
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        payload = (
            json.dumps(identity.to_mapping(), sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("short worker state write")
            written += count
        os.fsync(descriptor)
    except FileExistsError as error:
        raise WorkerLifecycleError(
            "worker_state_exists",
            "worker state appeared concurrently; launch ownership is ambiguous",
            identity=identity,
        ) from error
    except WorkerLifecycleError:
        raise
    except OSError as error:
        raise WorkerLifecycleError(
            "worker_state_write_failed",
            "worker ownership state could not be persisted",
            identity=identity,
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        _fsync_directory(path.parent)
    except OSError as error:
        raise WorkerLifecycleError(
            "worker_state_write_failed",
            "worker ownership directory could not be synchronized",
            identity=identity,
        ) from error


def load_worker_identity(
    cell_run_dir: Path, *, expected_run_nonce: str
) -> WorkerIdentity | None:
    """Load and strictly validate a persisted worker identity."""

    _validate_run_nonce(expected_run_nonce)
    path = worker_state_path(cell_run_dir)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise WorkerLifecycleError(
            "worker_state_unreadable", "worker state could not be opened safely"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise WorkerLifecycleError(
                "worker_state_unsafe", "worker state is not a singly linked regular file"
            )
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise WorkerLifecycleError(
                "worker_state_unsafe", "worker state ownership or mode is unsafe"
            )
        if metadata.st_size <= 0 or metadata.st_size > MAX_WORKER_STATE_BYTES:
            raise WorkerLifecycleError(
                "worker_state_malformed", "worker state size is outside its bound"
            )
        chunks: list[bytes] = []
        remaining = MAX_WORKER_STATE_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 1_024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > MAX_WORKER_STATE_BYTES:
            raise WorkerLifecycleError(
                "worker_state_malformed", "worker state grew beyond its size bound"
            )
    except OSError as error:
        raise WorkerLifecycleError(
            "worker_state_unreadable", "worker state could not be read safely"
        ) from error
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerLifecycleError(
            "worker_state_malformed", "worker state is not valid JSON"
        ) from error
    if not isinstance(value, dict) or set(value) != _WORKER_STATE_KEYS:
        raise WorkerLifecycleError(
            "worker_state_malformed", "worker state keys are not exact"
        )
    if value.get("schema_version") != WORKER_STATE_SCHEMA_VERSION:
        raise WorkerLifecycleError(
            "worker_state_malformed", "worker state schema is unsupported"
        )
    run_nonce = value.get("run_nonce")
    _validate_run_nonce(run_nonce)
    if run_nonce != expected_run_nonce:
        raise WorkerLifecycleError(
            "run_nonce_mismatch", "worker state belongs to a different frozen run"
        )
    identity = WorkerIdentity(
        pid=_validate_positive_integer(value.get("pid"), field="pid"),
        pgid=_validate_positive_integer(value.get("pgid"), field="pgid"),
        start_ticks=_validate_positive_integer(
            value.get("start_ticks"), field="start_ticks"
        ),
        run_nonce=run_nonce,
    )
    if identity.pid != identity.pgid:
        raise WorkerLifecycleError(
            "worker_state_malformed", "new-session worker PID and PGID must match"
        )
    return identity


def _snapshot_for_pid(pid: int, proc_reader: ProcStatReader) -> ProcessSnapshot | None:
    try:
        raw = proc_reader(pid)
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as error:
        raise WorkerLifecycleError(
            "proc_stat_unreadable", "worker proc identity could not be read"
        ) from error
    snapshot = parse_proc_stat(raw)
    if snapshot.pid != pid:
        raise WorkerLifecycleError(
            "proc_stat_malformed", "worker proc identity reported a different PID"
        )
    return snapshot


def _assert_owned_process(
    identity: WorkerIdentity, proc_reader: ProcStatReader
) -> ProcessSnapshot:
    snapshot = _snapshot_for_pid(identity.pid, proc_reader)
    if snapshot is None:
        raise WorkerLifecycleError(
            "owned_process_missing",
            "recorded worker process no longer exists",
            identity=identity,
        )
    if (
        snapshot.pid != identity.pid
        or snapshot.pgid != identity.pgid
        or snapshot.start_ticks != identity.start_ticks
    ):
        raise WorkerLifecycleError(
            "identity_mismatch",
            "recorded worker PID was reused or changed process group",
            identity=identity,
        )
    return snapshot


def _group_exists(pgid: int, signal_group: ProcessGroupSignaler) -> bool:
    try:
        signal_group(pgid, 0)
    except ProcessLookupError:
        return False
    except OSError as error:
        raise WorkerLifecycleError(
            "process_group_probe_failed", "worker process-group state is uncertain"
        ) from error
    return True


def _remove_matching_state(cell_run_dir: Path, identity: WorkerIdentity) -> None:
    current = load_worker_identity(
        cell_run_dir, expected_run_nonce=identity.run_nonce
    )
    if current != identity:
        raise WorkerLifecycleError(
            "worker_state_changed",
            "worker state changed before cleanup could be committed",
            identity=identity,
        )
    path = worker_state_path(cell_run_dir)
    try:
        path.unlink()
        _fsync_directory(path.parent)
    except OSError as error:
        raise WorkerLifecycleError(
            "worker_state_remove_failed",
            "terminal worker state could not be removed durably",
            identity=identity,
        ) from error


def _result(
    outcome: CleanupOutcome,
    identity: WorkerIdentity | None,
    *,
    return_code: int | None = None,
    sigint_sent: bool = False,
    sigkill_sent: bool = False,
    process_lookup_race: bool = False,
    state_removed: bool = False,
) -> WorkerCleanupResult:
    return WorkerCleanupResult(
        outcome=outcome,
        identity=identity,
        return_code=return_code,
        sigint_sent=sigint_sent,
        sigkill_sent=sigkill_sent,
        process_lookup_race=process_lookup_race,
        state_removed=state_removed,
    )


def _certify_absent_and_remove(
    cell_run_dir: Path,
    identity: WorkerIdentity,
    *,
    outcome: CleanupOutcome,
    return_code: int | None,
    sigint_sent: bool,
    sigkill_sent: bool,
    process_lookup_race: bool,
    signal_group: ProcessGroupSignaler,
    require_state: bool = True,
) -> WorkerCleanupResult:
    if _group_exists(identity.pgid, signal_group):
        partial = _result(
            outcome,
            identity,
            return_code=return_code,
            sigint_sent=sigint_sent,
            sigkill_sent=sigkill_sent,
            process_lookup_race=process_lookup_race,
        )
        raise WorkerLifecycleError(
            "process_group_survived",
            "worker process group survived after its leader was reaped",
            identity=identity,
            cleanup=partial,
        )
    state_removed = False
    if require_state:
        _remove_matching_state(cell_run_dir, identity)
        state_removed = True
    else:
        # State installation may have failed before a complete record existed.
        # Remove only an exact complete record; malformed, absent, or unrelated
        # state remains untouched and therefore blocks a later launch.
        try:
            current = load_worker_identity(
                cell_run_dir, expected_run_nonce=identity.run_nonce
            )
        except WorkerLifecycleError:
            current = None
        if current == identity:
            _remove_matching_state(cell_run_dir, identity)
            state_removed = True
    return _result(
        outcome,
        identity,
        return_code=return_code,
        sigint_sent=sigint_sent,
        sigkill_sent=sigkill_sent,
        process_lookup_race=process_lookup_race,
        state_removed=state_removed,
    )


def _signal_exact_group(
    identity: WorkerIdentity,
    action: int,
    *,
    proc_reader: ProcStatReader,
    signal_group: ProcessGroupSignaler,
) -> bool:
    """Signal a group only after a fresh exact identity comparison.

    ``False`` means the group disappeared between validation and signaling.
    That race is not treated as proof of cleanup; callers still have to reap or
    independently certify group absence before removing durable state.
    """

    _assert_owned_process(identity, proc_reader)
    try:
        signal_group(identity.pgid, action)
    except ProcessLookupError:
        return False
    except OSError as error:
        raise WorkerLifecycleError(
            "process_group_signal_failed",
            "owned worker process group could not be signaled",
            identity=identity,
        ) from error
    return True


def _terminate_current_process(
    process: subprocess.Popen[Any],
    cell_run_dir: Path,
    identity: WorkerIdentity,
    *,
    interrupt_grace_s: float,
    kill_grace_s: float,
    proc_reader: ProcStatReader,
    signal_group: ProcessGroupSignaler,
    require_state: bool = True,
) -> WorkerCleanupResult:
    if interrupt_grace_s < 0 or kill_grace_s <= 0:
        raise WorkerLifecycleError(
            "cleanup_timeout_invalid", "worker cleanup bounds are invalid"
        )
    sigint_sent = _signal_exact_group(
        identity,
        signal.SIGINT,
        proc_reader=proc_reader,
        signal_group=signal_group,
    )
    lookup_race = not sigint_sent
    try:
        return_code = process.wait(timeout=interrupt_grace_s)
    except subprocess.TimeoutExpired:
        if not sigint_sent:
            partial = _result(
                "interrupted", identity, process_lookup_race=True
            )
            raise WorkerLifecycleError(
                "process_lookup_race",
                "worker vanished during SIGINT but could not be reaped",
                identity=identity,
                cleanup=partial,
            )
        sigkill_sent = _signal_exact_group(
            identity,
            signal.SIGKILL,
            proc_reader=proc_reader,
            signal_group=signal_group,
        )
        lookup_race = lookup_race or not sigkill_sent
        try:
            return_code = process.wait(timeout=kill_grace_s)
        except subprocess.TimeoutExpired as error:
            partial = _result(
                "killed",
                identity,
                sigint_sent=sigint_sent,
                sigkill_sent=sigkill_sent,
                process_lookup_race=lookup_race,
            )
            raise WorkerLifecycleError(
                "reap_timeout",
                "worker could not be reaped within the SIGKILL bound",
                identity=identity,
                cleanup=partial,
            ) from error
        except BaseException as error:
            partial = _result(
                "killed",
                identity,
                sigint_sent=sigint_sent,
                sigkill_sent=sigkill_sent,
                process_lookup_race=lookup_race,
            )
            raise WorkerLifecycleError(
                "reap_failed",
                "worker reap failed after SIGKILL",
                identity=identity,
                cleanup=partial,
            ) from error
        return _certify_absent_and_remove(
            cell_run_dir,
            identity,
            outcome="killed",
            return_code=return_code,
            sigint_sent=sigint_sent,
            sigkill_sent=sigkill_sent,
            process_lookup_race=lookup_race,
            signal_group=signal_group,
            require_state=require_state,
        )
    except BaseException as error:
        partial = _result(
            "interrupted",
            identity,
            sigint_sent=sigint_sent,
            process_lookup_race=lookup_race,
        )
        raise WorkerLifecycleError(
            "reap_failed",
            "worker reap failed after SIGINT",
            identity=identity,
            cleanup=partial,
        ) from error
    return _certify_absent_and_remove(
        cell_run_dir,
        identity,
        outcome="interrupted",
        return_code=return_code,
        sigint_sent=sigint_sent,
        sigkill_sent=False,
        process_lookup_race=lookup_race,
        signal_group=signal_group,
        require_state=require_state,
    )


def _capture_identity(
    pid: int, run_nonce: str, proc_reader: ProcStatReader
) -> WorkerIdentity:
    _validate_run_nonce(run_nonce)
    snapshot = _snapshot_for_pid(pid, proc_reader)
    if snapshot is None:
        raise WorkerLifecycleError(
            "identity_capture_failed", "new worker exited before ownership was captured"
        )
    if snapshot.pid != pid or snapshot.pgid != pid:
        raise WorkerLifecycleError(
            "identity_capture_failed",
            "new-session worker did not become its own process-group leader",
        )
    return WorkerIdentity(
        pid=pid,
        pgid=snapshot.pgid,
        start_ticks=snapshot.start_ticks,
        run_nonce=run_nonce,
    )


def _cleanup_uncaptured_process(
    process: subprocess.Popen[Any],
    *,
    kill_grace_s: float,
    signal_group: ProcessGroupSignaler,
    signal_process: ProcessSignaler,
) -> None:
    """Kill a just-created child whose durable proc identity was unavailable.

    The unreaped child retains its PID, so neither its PID nor its intended
    new-session PGID can be reused while these signals are sent.
    """

    signal_failure: OSError | None = None
    for target, signaler in (
        (process.pid, signal_group),
        (process.pid, signal_process),
    ):
        try:
            signaler(target, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except OSError as error:
            signal_failure = error
    try:
        process.wait(timeout=kill_grace_s)
    except subprocess.TimeoutExpired as error:
        raise WorkerLifecycleError(
            "identity_capture_cleanup_failed",
            "uncaptured worker could not be reaped within the kill bound",
        ) from error
    except BaseException as error:
        raise WorkerLifecycleError(
            "identity_capture_cleanup_failed",
            "uncaptured worker reap failed",
        ) from error
    if signal_failure is not None:
        raise WorkerLifecycleError(
            "identity_capture_cleanup_failed",
            "uncaptured worker group cleanup could not be certified",
        ) from signal_failure


def run_owned_worker(
    argv: Sequence[str],
    *,
    cell_run_dir: Path,
    run_nonce: str,
    timeout_s: float,
    interrupt_grace_s: float,
    kill_grace_s: float = 10.0,
    popen_kwargs: Mapping[str, Any] | None = None,
    popen_factory: PopenFactory = subprocess.Popen,
    proc_reader: ProcStatReader = read_proc_stat,
    signal_group: ProcessGroupSignaler = os.killpg,
    signal_process: ProcessSignaler = os.kill,
) -> WorkerRunResult:
    """Run one worker in a new session with durable, bounded ownership.

    A causal timeout returns ``timed_out=True`` after certified cleanup.  Any
    other ``BaseException`` (including ``KeyboardInterrupt``) is re-raised only
    after certified cleanup; cleanup ambiguity instead raises
    :class:`WorkerLifecycleError` chained from the original exception.
    """

    if not argv or not all(isinstance(part, str) and part for part in argv):
        raise WorkerLifecycleError(
            "worker_argv_invalid", "worker argv must contain nonempty strings"
        )
    _validate_run_nonce(run_nonce)
    if timeout_s <= 0 or interrupt_grace_s < 0 or kill_grace_s <= 0:
        raise WorkerLifecycleError(
            "worker_timeout_invalid", "worker and cleanup bounds must be positive"
        )
    state_path = worker_state_path(cell_run_dir)
    _ensure_state_absent(state_path)
    options = dict(popen_kwargs or {})
    forbidden = {"args", "shell", "start_new_session"}.intersection(options)
    if forbidden:
        raise WorkerLifecycleError(
            "popen_options_invalid",
            "worker launch options may not override ownership controls",
        )
    try:
        process = popen_factory(
            list(argv), shell=False, start_new_session=True, **options
        )
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise WorkerLifecycleError(
            "worker_launch_failed", "worker process could not be launched"
        ) from error
    try:
        identity = _capture_identity(process.pid, run_nonce, proc_reader)
    except BaseException as capture_error:
        try:
            _cleanup_uncaptured_process(
                process,
                kill_grace_s=kill_grace_s,
                signal_group=signal_group,
                signal_process=signal_process,
            )
        except WorkerLifecycleError as cleanup_error:
            raise cleanup_error from capture_error
        raise
    try:
        _persist_worker_identity(state_path, identity)
    except BaseException as persistence_error:
        # We have a validated live identity even if its durable record could
        # not be installed, so perform exact bounded cleanup before failing.
        try:
            cleanup = _terminate_current_process(
                process,
                cell_run_dir,
                identity,
                interrupt_grace_s=interrupt_grace_s,
                kill_grace_s=kill_grace_s,
                proc_reader=proc_reader,
                signal_group=signal_group,
                require_state=False,
            )
        except WorkerLifecycleError as cleanup_error:
            raise cleanup_error from persistence_error
        error = WorkerLifecycleError(
            "worker_state_write_failed",
            "worker was cleaned after ownership state could not be persisted",
            identity=identity,
            cleanup=cleanup,
        )
        raise error from persistence_error

    try:
        return_code = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        cleanup = _terminate_current_process(
            process,
            cell_run_dir,
            identity,
            interrupt_grace_s=interrupt_grace_s,
            kill_grace_s=kill_grace_s,
            proc_reader=proc_reader,
            signal_group=signal_group,
        )
        if cleanup.return_code is None:
            raise WorkerLifecycleError(
                "reap_failed",
                "timed-out worker has no certified return code",
                identity=identity,
                cleanup=cleanup,
            )
        return WorkerRunResult(
            return_code=cleanup.return_code,
            timed_out=True,
            cleanup=cleanup,
        )
    except BaseException as original_error:
        try:
            _terminate_current_process(
                process,
                cell_run_dir,
                identity,
                interrupt_grace_s=interrupt_grace_s,
                kill_grace_s=kill_grace_s,
                proc_reader=proc_reader,
                signal_group=signal_group,
            )
        except WorkerLifecycleError as cleanup_error:
            raise cleanup_error from original_error
        raise

    cleanup = _certify_absent_and_remove(
        cell_run_dir,
        identity,
        outcome="completed",
        return_code=return_code,
        sigint_sent=False,
        sigkill_sent=False,
        process_lookup_race=False,
        signal_group=signal_group,
    )
    return WorkerRunResult(
        return_code=return_code,
        timed_out=False,
        cleanup=cleanup,
    )


def _wait_for_recovery_exit(
    identity: WorkerIdentity,
    *,
    timeout_s: float,
    proc_reader: ProcStatReader,
    signal_group: ProcessGroupSignaler,
    monotonic: MonotonicClock,
    sleep: Sleeper,
    poll_interval_s: float,
) -> Literal["absent", "alive"]:
    deadline = monotonic() + timeout_s
    while True:
        snapshot = _snapshot_for_pid(identity.pid, proc_reader)
        if snapshot is None:
            if _group_exists(identity.pgid, signal_group):
                raise WorkerLifecycleError(
                    "ownership_ambiguous",
                    "worker leader vanished while its process group remained",
                    identity=identity,
                )
            return "absent"
        if (
            snapshot.pgid != identity.pgid
            or snapshot.start_ticks != identity.start_ticks
        ):
            raise WorkerLifecycleError(
                "identity_mismatch",
                "worker PID was reused during recovery",
                identity=identity,
            )
        remaining = deadline - monotonic()
        if remaining <= 0:
            return "alive"
        sleep(min(poll_interval_s, remaining))


def recover_owned_worker(
    cell_run_dir: Path,
    *,
    expected_run_nonce: str,
    interrupt_grace_s: float,
    kill_grace_s: float = 10.0,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    proc_reader: ProcStatReader = read_proc_stat,
    signal_group: ProcessGroupSignaler = os.killpg,
    monotonic: MonotonicClock = time.monotonic,
    sleep: Sleeper = time.sleep,
) -> WorkerCleanupResult:
    """Recover and terminate one exactly owned worker after controller restart."""

    if interrupt_grace_s < 0 or kill_grace_s <= 0 or poll_interval_s <= 0:
        raise WorkerLifecycleError(
            "cleanup_timeout_invalid", "worker recovery bounds are invalid"
        )
    identity = load_worker_identity(
        cell_run_dir, expected_run_nonce=expected_run_nonce
    )
    if identity is None:
        return _result("no_state", None)
    snapshot = _snapshot_for_pid(identity.pid, proc_reader)
    if snapshot is None:
        if _group_exists(identity.pgid, signal_group):
            raise WorkerLifecycleError(
                "ownership_ambiguous",
                "recorded worker leader is gone but its group remains",
                identity=identity,
            )
        _remove_matching_state(cell_run_dir, identity)
        return _result(
            "already_exited", identity, process_lookup_race=True, state_removed=True
        )
    _assert_owned_process(identity, proc_reader)
    sigint_sent = _signal_exact_group(
        identity,
        signal.SIGINT,
        proc_reader=proc_reader,
        signal_group=signal_group,
    )
    lookup_race = not sigint_sent
    state = _wait_for_recovery_exit(
        identity,
        timeout_s=interrupt_grace_s,
        proc_reader=proc_reader,
        signal_group=signal_group,
        monotonic=monotonic,
        sleep=sleep,
        poll_interval_s=poll_interval_s,
    )
    if state == "absent":
        _remove_matching_state(cell_run_dir, identity)
        return _result(
            "interrupted",
            identity,
            sigint_sent=sigint_sent,
            process_lookup_race=lookup_race,
            state_removed=True,
        )
    if not sigint_sent:
        partial = _result(
            "interrupted", identity, process_lookup_race=True
        )
        raise WorkerLifecycleError(
            "process_lookup_race",
            "worker survived after its SIGINT target disappeared",
            identity=identity,
            cleanup=partial,
        )
    sigkill_sent = _signal_exact_group(
        identity,
        signal.SIGKILL,
        proc_reader=proc_reader,
        signal_group=signal_group,
    )
    lookup_race = lookup_race or not sigkill_sent
    state = _wait_for_recovery_exit(
        identity,
        timeout_s=kill_grace_s,
        proc_reader=proc_reader,
        signal_group=signal_group,
        monotonic=monotonic,
        sleep=sleep,
        poll_interval_s=poll_interval_s,
    )
    if state != "absent":
        partial = _result(
            "killed",
            identity,
            sigint_sent=sigint_sent,
            sigkill_sent=sigkill_sent,
            process_lookup_race=lookup_race,
        )
        raise WorkerLifecycleError(
            "termination_timeout",
            "worker survived the bounded SIGKILL recovery interval",
            identity=identity,
            cleanup=partial,
        )
    _remove_matching_state(cell_run_dir, identity)
    return _result(
        "killed",
        identity,
        sigint_sent=sigint_sent,
        sigkill_sent=sigkill_sent,
        process_lookup_race=lookup_race,
        state_removed=True,
    )
