from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import tempfile
import unittest

from bench.autoresearch_worker import (
    WORKER_STATE_SCHEMA_VERSION,
    WorkerLifecycleError,
    WorkerProgress,
    parse_proc_stat,
    recover_owned_worker,
    run_owned_worker,
    worker_state_path,
)


RUN_NONCE = "0123456789abcdef0123456789abcdef"
PID = 4_321
START_TICKS = 987_654


def _proc_stat(
    *, pid: int = PID, pgid: int = PID, start_ticks: int = START_TICKS
) -> str:
    # Kernel fields 3 through 22.  The deliberately awkward command exercises
    # parsing from the final closing parenthesis rather than splitting on spaces.
    fields = ["S", "1", str(pgid), str(pgid)] + ["0"] * 15 + [str(start_ticks)]
    return f"{pid} (synthetic worker ) name) {' '.join(fields)}\n"


class _FakeProcess:
    def __init__(self, waits: list[object], *, pid: int = PID) -> None:
        self.pid = pid
        self._waits = waits
        self.timeouts: list[float] = []

    def wait(self, timeout: float) -> int:
        self.timeouts.append(timeout)
        if not self._waits:
            raise AssertionError("unexpected process.wait call")
        outcome = self._waits.pop(0)
        if callable(outcome):
            outcome = outcome()
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, int)
        return outcome


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += duration


class _TimedProcess:
    def __init__(self, clock: _Clock, *, exit_at: float = float("inf")) -> None:
        self.pid = PID
        self.clock = clock
        self.exit_at = exit_at
        self.return_code = 0
        self.exited = False
        self.timeouts: list[float] = []

    def wait(self, timeout: float) -> int:
        self.timeouts.append(timeout)
        if self.exited:
            return self.return_code
        deadline = self.clock.value + timeout
        if self.exit_at <= deadline:
            self.clock.value = self.exit_at
            self.exited = True
            return self.return_code
        self.clock.value = deadline
        raise subprocess.TimeoutExpired("synthetic", timeout)


def _missing_group(_pgid: int, action: int) -> None:
    if action != 0:
        raise AssertionError("unexpected terminating signal")
    raise ProcessLookupError


def _write_state(run_dir: Path, **overrides: object) -> Path:
    value: dict[str, object] = {
        "schema_version": WORKER_STATE_SCHEMA_VERSION,
        "pid": PID,
        "pgid": PID,
        "start_ticks": START_TICKS,
        "run_nonce": RUN_NONCE,
    }
    value.update(overrides)
    path = worker_state_path(run_dir)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


class AutoresearchWorkerTests(unittest.TestCase):
    def test_parse_proc_stat_handles_spaces_and_parenthesis_in_command(self) -> None:
        snapshot = parse_proc_stat(_proc_stat())

        self.assertEqual(snapshot.pid, PID)
        self.assertEqual(snapshot.pgid, PID)
        self.assertEqual(snapshot.start_ticks, START_TICKS)

    def test_fresh_run_persists_minimal_mode_0600_identity_before_wait(self) -> None:
        captured_state: dict[str, object] = {}
        captured_launch: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            state_path = worker_state_path(run_dir)

            def inspect_state() -> int:
                captured_state.update(json.loads(state_path.read_text()))
                captured_state["mode"] = stat.S_IMODE(state_path.stat().st_mode)
                return 0

            process = _FakeProcess([inspect_state])

            def popen(argv: list[str], **kwargs: object) -> _FakeProcess:
                captured_launch["argv"] = argv
                captured_launch.update(kwargs)
                return process

            result = run_owned_worker(
                ["python3", "synthetic-worker.py"],
                cell_run_dir=run_dir,
                run_nonce=RUN_NONCE,
                timeout_s=30,
                interrupt_grace_s=2,
                popen_kwargs={"env": {"SYNTHETIC_SECRET": "not-persisted"}},
                popen_factory=popen,
                proc_reader=lambda _pid: _proc_stat(),
                signal_group=_missing_group,
            )

            self.assertFalse(state_path.exists())

        self.assertFalse(result.timed_out)
        self.assertEqual(result.return_code, 0)
        self.assertEqual(result.cleanup.outcome, "completed")
        self.assertTrue(result.cleanup.state_removed)
        self.assertTrue(captured_launch["start_new_session"])
        self.assertFalse(captured_launch["shell"])
        self.assertEqual(
            set(captured_state),
            {
                "schema_version",
                "pid",
                "pgid",
                "start_ticks",
                "run_nonce",
                "mode",
            },
        )
        self.assertEqual(captured_state["mode"], 0o600)
        serialized = json.dumps(captured_state)
        self.assertNotIn("synthetic-worker.py", serialized)
        self.assertNotIn("SYNTHETIC_SECRET", serialized)

    def test_identity_capture_failure_kills_and_reaps_uncaptured_process(self) -> None:
        process = _FakeProcess([0])
        group_actions: list[tuple[int, int]] = []
        process_actions: list[tuple[int, int]] = []

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(WorkerLifecycleError) as raised:
                run_owned_worker(
                    ["synthetic"],
                    cell_run_dir=Path(directory),
                    run_nonce=RUN_NONCE,
                    timeout_s=30,
                    interrupt_grace_s=4,
                    kill_grace_s=7,
                    popen_factory=lambda *_args, **_kwargs: process,
                    proc_reader=lambda _pid: _proc_stat(pgid=PID + 1),
                    signal_group=lambda target, action: group_actions.append(
                        (target, action)
                    ),
                    signal_process=lambda target, action: process_actions.append(
                        (target, action)
                    ),
                )

        self.assertEqual(raised.exception.code, "identity_capture_failed")
        self.assertEqual(group_actions, [(PID, signal.SIGKILL)])
        self.assertEqual(process_actions, [(PID, signal.SIGKILL)])
        self.assertEqual(process.timeouts, [7])

    def test_timeout_interrupts_exact_group_and_returns_typed_result(self) -> None:
        timeout = subprocess.TimeoutExpired("synthetic", 30)
        process = _FakeProcess([timeout, -signal.SIGINT])
        actions: list[int] = []

        def signal_group(_pgid: int, action: int) -> None:
            actions.append(action)
            if action == 0:
                raise ProcessLookupError

        with tempfile.TemporaryDirectory() as directory:
            result = run_owned_worker(
                ["synthetic"],
                cell_run_dir=Path(directory),
                run_nonce=RUN_NONCE,
                timeout_s=30,
                interrupt_grace_s=4,
                popen_factory=lambda *_args, **_kwargs: process,
                proc_reader=lambda _pid: _proc_stat(),
                signal_group=signal_group,
            )

        self.assertTrue(result.timed_out)
        self.assertEqual(result.cleanup.outcome, "interrupted")
        self.assertTrue(result.cleanup.sigint_sent)
        self.assertFalse(result.cleanup.sigkill_sent)
        self.assertEqual(actions, [signal.SIGINT, 0])
        self.assertEqual(process.timeouts, [30, 4])

    def test_timeout_escalates_to_sigkill_and_reaps_within_bound(self) -> None:
        process = _FakeProcess(
            [
                subprocess.TimeoutExpired("synthetic", 30),
                subprocess.TimeoutExpired("synthetic", 4),
                -signal.SIGKILL,
            ]
        )
        actions: list[int] = []

        def signal_group(_pgid: int, action: int) -> None:
            actions.append(action)
            if action == 0:
                raise ProcessLookupError

        with tempfile.TemporaryDirectory() as directory:
            result = run_owned_worker(
                ["synthetic"],
                cell_run_dir=Path(directory),
                run_nonce=RUN_NONCE,
                timeout_s=30,
                interrupt_grace_s=4,
                kill_grace_s=7,
                popen_factory=lambda *_args, **_kwargs: process,
                proc_reader=lambda _pid: _proc_stat(),
                signal_group=signal_group,
            )

        self.assertTrue(result.timed_out)
        self.assertEqual(result.cleanup.outcome, "killed")
        self.assertEqual(actions, [signal.SIGINT, signal.SIGKILL, 0])
        self.assertEqual(process.timeouts, [30, 4, 7])

    def test_final_reap_timeout_is_typed_and_preserves_recovery_state(self) -> None:
        process = _FakeProcess(
            [
                subprocess.TimeoutExpired("synthetic", 30),
                subprocess.TimeoutExpired("synthetic", 4),
                subprocess.TimeoutExpired("synthetic", 7),
            ]
        )
        actions: list[int] = []

        def signal_group(_pgid: int, action: int) -> None:
            actions.append(action)

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with self.assertRaises(WorkerLifecycleError) as raised:
                run_owned_worker(
                    ["synthetic"],
                    cell_run_dir=run_dir,
                    run_nonce=RUN_NONCE,
                    timeout_s=30,
                    interrupt_grace_s=4,
                    kill_grace_s=7,
                    popen_factory=lambda *_args, **_kwargs: process,
                    proc_reader=lambda _pid: _proc_stat(),
                    signal_group=signal_group,
                )
            self.assertTrue(worker_state_path(run_dir).exists())

        self.assertEqual(raised.exception.code, "reap_timeout")
        self.assertIsNotNone(raised.exception.cleanup)
        self.assertEqual(actions, [signal.SIGINT, signal.SIGKILL])

    def test_process_lookup_race_is_explicit_only_after_absence_is_certified(self) -> None:
        process = _FakeProcess(
            [subprocess.TimeoutExpired("synthetic", 30), -signal.SIGINT]
        )

        def signal_group(_pgid: int, _action: int) -> None:
            raise ProcessLookupError

        with tempfile.TemporaryDirectory() as directory:
            result = run_owned_worker(
                ["synthetic"],
                cell_run_dir=Path(directory),
                run_nonce=RUN_NONCE,
                timeout_s=30,
                interrupt_grace_s=4,
                popen_factory=lambda *_args, **_kwargs: process,
                proc_reader=lambda _pid: _proc_stat(),
                signal_group=signal_group,
            )

        self.assertTrue(result.cleanup.process_lookup_race)
        self.assertFalse(result.cleanup.sigint_sent)
        self.assertTrue(result.cleanup.state_removed)

    def test_late_measurement_switches_to_separate_cleanup_deadline(self) -> None:
        clock = _Clock()
        process = _TimedProcess(clock, exit_at=1.821)
        actions: list[int] = []

        def progress() -> WorkerProgress:
            if clock.value < 1.79:
                return WorkerProgress("measurement", 1.8)
            if clock.value < 1.82:
                return WorkerProgress("cleanup", 1.91)
            return WorkerProgress("finalization", 1.83)

        def signal_group(_pgid: int, action: int) -> None:
            actions.append(action)
            if action == 0 and process.exited:
                raise ProcessLookupError

        with tempfile.TemporaryDirectory() as directory:
            result = run_owned_worker(
                ["synthetic"],
                cell_run_dir=Path(directory),
                run_nonce=RUN_NONCE,
                timeout_s=1.8,
                interrupt_grace_s=0.12,
                progress_probe=progress,
                progress_poll_interval_s=0.05,
                popen_factory=lambda *_args, **_kwargs: process,
                proc_reader=lambda _pid: _proc_stat(),
                signal_group=signal_group,
                monotonic=clock.monotonic,
            )

        self.assertFalse(result.timed_out)
        self.assertIsNone(result.timeout_phase)
        self.assertEqual(actions, [0])
        self.assertGreater(clock.value, 1.8)

    def test_deadline_reprobe_observes_marker_before_signaling(self) -> None:
        clock = _Clock()
        process = _TimedProcess(clock, exit_at=1.81)
        boundary_probes = 0
        actions: list[int] = []

        def progress() -> WorkerProgress:
            nonlocal boundary_probes
            if clock.value < 1.8:
                return WorkerProgress("measurement", 1.8)
            boundary_probes += 1
            if boundary_probes == 1:
                return WorkerProgress("measurement", 1.8)
            return WorkerProgress("cleanup", 1.92)

        def signal_group(_pgid: int, action: int) -> None:
            actions.append(action)
            if action == 0 and process.exited:
                raise ProcessLookupError

        with tempfile.TemporaryDirectory() as directory:
            result = run_owned_worker(
                ["synthetic"],
                cell_run_dir=Path(directory),
                run_nonce=RUN_NONCE,
                timeout_s=1.8,
                interrupt_grace_s=0.12,
                progress_probe=progress,
                progress_poll_interval_s=0.05,
                popen_factory=lambda *_args, **_kwargs: process,
                proc_reader=lambda _pid: _proc_stat(),
                signal_group=signal_group,
                monotonic=clock.monotonic,
            )

        self.assertFalse(result.timed_out)
        self.assertGreaterEqual(boundary_probes, 2)
        self.assertEqual(actions, [0])

    def test_cleanup_timeout_gets_no_second_cleanup_grace(self) -> None:
        clock = _Clock()
        process = _TimedProcess(clock)
        actions: list[int] = []

        def progress() -> WorkerProgress:
            if clock.value < 1.0:
                return WorkerProgress("measurement", 1.8)
            return WorkerProgress("cleanup", 1.12)

        def signal_group(_pgid: int, action: int) -> None:
            actions.append(action)
            if action == signal.SIGKILL:
                process.return_code = -signal.SIGKILL
                process.exit_at = clock.value
            elif action == 0 and process.exited:
                raise ProcessLookupError

        def proc_reader(_pid: int) -> str:
            if process.exited:
                raise FileNotFoundError
            return _proc_stat()

        with tempfile.TemporaryDirectory() as directory:
            result = run_owned_worker(
                ["synthetic"],
                cell_run_dir=Path(directory),
                run_nonce=RUN_NONCE,
                timeout_s=1.8,
                interrupt_grace_s=0.12,
                progress_probe=progress,
                progress_poll_interval_s=0.05,
                popen_factory=lambda *_args, **_kwargs: process,
                proc_reader=proc_reader,
                signal_group=signal_group,
                monotonic=clock.monotonic,
            )

        self.assertTrue(result.timed_out)
        self.assertEqual(result.timeout_phase, "cleanup")
        self.assertEqual(result.cleanup.outcome, "killed")
        self.assertEqual(actions, [signal.SIGINT, signal.SIGKILL, 0])
        self.assertIn(0.0, process.timeouts)

    def test_measurement_timeout_retains_owned_interrupt_grace(self) -> None:
        clock = _Clock()
        process = _TimedProcess(clock)
        actions: list[int] = []

        def signal_group(_pgid: int, action: int) -> None:
            actions.append(action)
            if action == signal.SIGINT:
                process.return_code = -signal.SIGINT
                process.exit_at = clock.value + 0.05
            elif action == 0 and process.exited:
                raise ProcessLookupError

        with tempfile.TemporaryDirectory() as directory:
            result = run_owned_worker(
                ["synthetic"],
                cell_run_dir=Path(directory),
                run_nonce=RUN_NONCE,
                timeout_s=1.8,
                interrupt_grace_s=0.12,
                progress_probe=lambda: WorkerProgress("measurement", 1.8),
                progress_poll_interval_s=0.05,
                popen_factory=lambda *_args, **_kwargs: process,
                proc_reader=lambda _pid: _proc_stat(),
                signal_group=signal_group,
                monotonic=clock.monotonic,
            )

        self.assertTrue(result.timed_out)
        self.assertEqual(result.timeout_phase, "measurement")
        self.assertEqual(result.cleanup.outcome, "interrupted")
        self.assertEqual(actions, [signal.SIGINT, 0])
        self.assertIn(0.12, process.timeouts)

    def test_keyboard_interrupt_cleans_then_reraises_original_exception(self) -> None:
        process = _FakeProcess([KeyboardInterrupt(), -signal.SIGINT])
        actions: list[int] = []

        def signal_group(_pgid: int, action: int) -> None:
            actions.append(action)
            if action == 0:
                raise ProcessLookupError

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with self.assertRaises(KeyboardInterrupt):
                run_owned_worker(
                    ["synthetic"],
                    cell_run_dir=run_dir,
                    run_nonce=RUN_NONCE,
                    timeout_s=30,
                    interrupt_grace_s=4,
                    popen_factory=lambda *_args, **_kwargs: process,
                    proc_reader=lambda _pid: _proc_stat(),
                    signal_group=signal_group,
                )
            self.assertFalse(worker_state_path(run_dir).exists())

        self.assertEqual(actions, [signal.SIGINT, 0])

    def test_restart_recovery_interrupts_matching_identity(self) -> None:
        calls = 0
        actions: list[int] = []

        def proc_reader(_pid: int) -> str:
            nonlocal calls
            calls += 1
            if calls >= 4:
                raise FileNotFoundError
            return _proc_stat()

        def signal_group(_pgid: int, action: int) -> None:
            actions.append(action)
            if action == 0:
                raise ProcessLookupError

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_state(run_dir)
            result = recover_owned_worker(
                run_dir,
                expected_run_nonce=RUN_NONCE,
                interrupt_grace_s=2,
                proc_reader=proc_reader,
                signal_group=signal_group,
            )
            self.assertFalse(worker_state_path(run_dir).exists())

        self.assertEqual(result.outcome, "interrupted")
        self.assertTrue(result.sigint_sent)
        self.assertEqual(actions, [signal.SIGINT, 0])

    def test_restart_recovery_escalates_to_sigkill(self) -> None:
        killed = False
        actions: list[int] = []

        def proc_reader(_pid: int) -> str:
            if killed:
                raise FileNotFoundError
            return _proc_stat()

        def signal_group(_pgid: int, action: int) -> None:
            nonlocal killed
            actions.append(action)
            if action == signal.SIGKILL:
                killed = True
            elif action == 0:
                raise ProcessLookupError

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_state(run_dir)
            result = recover_owned_worker(
                run_dir,
                expected_run_nonce=RUN_NONCE,
                interrupt_grace_s=0,
                kill_grace_s=2,
                proc_reader=proc_reader,
                signal_group=signal_group,
            )

        self.assertEqual(result.outcome, "killed")
        self.assertEqual(actions, [signal.SIGINT, signal.SIGKILL, 0])

    def test_restart_never_signals_reused_pid(self) -> None:
        actions: list[int] = []
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            state_path = _write_state(run_dir)
            with self.assertRaises(WorkerLifecycleError) as raised:
                recover_owned_worker(
                    run_dir,
                    expected_run_nonce=RUN_NONCE,
                    interrupt_grace_s=2,
                    proc_reader=lambda _pid: _proc_stat(start_ticks=START_TICKS + 1),
                    signal_group=lambda _pgid, action: actions.append(action),
                )
            self.assertTrue(state_path.exists())

        self.assertEqual(raised.exception.code, "identity_mismatch")
        self.assertEqual(actions, [])

    def test_restart_without_state_is_a_typed_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = recover_owned_worker(
                Path(directory),
                expected_run_nonce=RUN_NONCE,
                interrupt_grace_s=2,
            )

        self.assertEqual(result.outcome, "no_state")
        self.assertIsNone(result.identity)

    def test_unsafe_state_mode_is_rejected_without_signaling(self) -> None:
        actions: list[int] = []
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            state_path = _write_state(run_dir)
            state_path.chmod(0o644)
            with self.assertRaises(WorkerLifecycleError) as raised:
                recover_owned_worker(
                    run_dir,
                    expected_run_nonce=RUN_NONCE,
                    interrupt_grace_s=2,
                    signal_group=lambda _pgid, action: actions.append(action),
                )

        self.assertEqual(raised.exception.code, "worker_state_unsafe")
        self.assertEqual(actions, [])

    def test_recovery_sigkill_timeout_is_typed_and_leaves_state(self) -> None:
        clock = _Clock()
        actions: list[int] = []

        def signal_group(_pgid: int, action: int) -> None:
            actions.append(action)

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            state_path = _write_state(run_dir)
            with self.assertRaises(WorkerLifecycleError) as raised:
                recover_owned_worker(
                    run_dir,
                    expected_run_nonce=RUN_NONCE,
                    interrupt_grace_s=0,
                    kill_grace_s=0.2,
                    poll_interval_s=0.1,
                    proc_reader=lambda _pid: _proc_stat(),
                    signal_group=signal_group,
                    monotonic=clock.monotonic,
                    sleep=clock.sleep,
                )
            self.assertTrue(state_path.exists())

        self.assertEqual(raised.exception.code, "termination_timeout")
        self.assertEqual(actions, [signal.SIGINT, signal.SIGKILL])


if __name__ == "__main__":
    unittest.main()
