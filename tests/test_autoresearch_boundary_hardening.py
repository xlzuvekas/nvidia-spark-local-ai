from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest

from bench.autoresearch import AUDIT_RESERVE_S, CELL_TIMEOUT_S, CLEANUP_TIMEOUT_S
from bench.autoresearch_campaign import (
    FINALIZATION_TIMEOUT_S,
    PAIR_ADMISSION_REMAINING_S,
    START_MARKER_TIMEOUT_S,
)
from bench.autoresearch_worker import (
    WorkerLifecycleError,
    WorkerProgress,
    run_owned_worker,
    worker_state_path,
)
from bench.journal import Journal


RUN_NONCE = "0123456789abcdef0123456789abcdef"
PID = 4_321
START_TICKS = 987_654


def _proc_stat() -> str:
    fields = ["S", "1", str(PID), str(PID)] + ["0"] * 15 + [str(START_TICKS)]
    return f"{PID} (synthetic boundary worker) {' '.join(fields)}\n"


class _InterruptEscalationProcess:
    def __init__(self) -> None:
        self.pid = PID
        self.alive = True
        self.timeouts: list[float] = []

    def poll(self) -> int | None:
        return None if self.alive else -signal.SIGKILL

    def wait(self, timeout: float) -> int:
        self.timeouts.append(timeout)
        call = len(self.timeouts)
        if call == 1:
            raise KeyboardInterrupt
        if call == 2:
            raise subprocess.TimeoutExpired("synthetic", timeout)
        if call == 3:
            self.alive = False
            return -signal.SIGKILL
        raise AssertionError("unexpected process.wait call")


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value


class _DeadlineExitRaceProcess:
    """Exit after the boundary poll loses its zero-time wait race."""

    def __init__(self, clock: _Clock) -> None:
        self.pid = PID
        self.clock = clock
        self.exited = False
        self.timeouts: list[float] = []

    def poll(self) -> int | None:
        return 0 if self.exited else None

    def wait(self, timeout: float) -> int:
        self.timeouts.append(timeout)
        call = len(self.timeouts)
        if call == 1:
            self.clock.value += timeout
            raise subprocess.TimeoutExpired("synthetic", timeout)
        if call == 2:
            self.exited = True
            raise subprocess.TimeoutExpired("synthetic", timeout)
        raise AssertionError("unexpected process.wait call")


class JournalBoundaryHardeningTests(unittest.TestCase):
    def test_pair_admission_reserves_every_bounded_phase_for_both_cells(self) -> None:
        expected = (
            2 * CELL_TIMEOUT_S
            + 2 * CLEANUP_TIMEOUT_S
            + 2 * START_MARKER_TIMEOUT_S
            + CLEANUP_TIMEOUT_S
            + FINALIZATION_TIMEOUT_S
            + AUDIT_RESERVE_S
        )

        self.assertEqual(PAIR_ADMISSION_REMAINING_S, expected)
        self.assertEqual(PAIR_ADMISSION_REMAINING_S, 4_930)

    def test_append_rejects_dangling_symlink_without_creating_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "missing-target.jsonl"
            journal_path = root / "events.jsonl"
            journal_path.symlink_to(target)

            with self.assertRaises(OSError):
                Journal(journal_path).append({"event": "synthetic"})

            self.assertTrue(journal_path.is_symlink())
            self.assertFalse(target.exists())

    def test_strict_read_rejects_middle_corruption_duplicate_keys_and_torn_tail(
        self,
    ) -> None:
        fixtures = (
            b'{"event":"first"}\nnot-json\n{"event":"last"}\n',
            b'{"event":"first","event":"second"}\n',
            b'{"event":"first"}\n{"event":"torn"',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            journal = Journal(path)
            for payload in fixtures:
                with self.subTest(payload=payload):
                    path.write_bytes(payload)
                    with self.assertRaises(ValueError):
                        journal.strict_events()

    def test_append_rejects_hardlink_without_mutating_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.jsonl"
            journal_path = root / "events.jsonl"
            original = b'{"sentinel":true}\n'
            target.write_bytes(original)
            target.chmod(0o600)
            os.link(target, journal_path)
            before = target.stat()

            with self.assertRaisesRegex(OSError, "single-link regular file"):
                Journal(journal_path).append({"event": "synthetic"})

            after = target.stat()
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(after.st_size, before.st_size)
            self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(after.st_nlink, 2)


class WorkerBoundaryHardeningTests(unittest.TestCase):
    def test_keyboard_interrupt_sigkill_escalation_is_typed_and_cleans_state(self) -> None:
        process = _InterruptEscalationProcess()
        actions: list[int] = []

        def signal_group(_pgid: int, action: int) -> None:
            actions.append(action)
            if action == 0:
                if process.alive:
                    return
                raise ProcessLookupError

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
            self.assertFalse(worker_state_path(run_dir).exists())

        error = raised.exception
        self.assertEqual(error.code, "interrupt_cleanup_escalated")
        self.assertIsInstance(error.__cause__, KeyboardInterrupt)
        self.assertIsNotNone(error.cleanup)
        assert error.cleanup is not None
        self.assertEqual(error.cleanup.outcome, "killed")
        self.assertTrue(error.cleanup.sigint_sent)
        self.assertTrue(error.cleanup.sigkill_sent)
        self.assertTrue(error.cleanup.state_removed)
        self.assertEqual(actions, [signal.SIGINT, signal.SIGKILL, 0])
        self.assertEqual(process.timeouts, [30, 4, 7])

    def test_deadline_exit_race_removes_state_and_preserves_timeout_phase(self) -> None:
        clock = _Clock()
        process = _DeadlineExitRaceProcess(clock)
        actions: list[int] = []

        def signal_group(_pgid: int, action: int) -> None:
            actions.append(action)
            if action == 0 and process.exited:
                raise ProcessLookupError
            if action != 0:
                raise AssertionError("deadline exit race must not be signaled")

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            result = run_owned_worker(
                ["synthetic"],
                cell_run_dir=run_dir,
                run_nonce=RUN_NONCE,
                timeout_s=1,
                interrupt_grace_s=4,
                kill_grace_s=7,
                progress_probe=lambda: WorkerProgress("measurement", 1.0),
                progress_poll_interval_s=1,
                popen_factory=lambda *_args, **_kwargs: process,
                proc_reader=lambda _pid: _proc_stat(),
                signal_group=signal_group,
                monotonic=clock.monotonic,
            )
            self.assertFalse(worker_state_path(run_dir).exists())

        self.assertTrue(result.timed_out)
        self.assertEqual(result.timeout_phase, "measurement")
        self.assertEqual(result.return_code, 0)
        self.assertEqual(result.cleanup.outcome, "already_exited")
        self.assertTrue(result.cleanup.state_removed)
        self.assertFalse(result.cleanup.sigint_sent)
        self.assertFalse(result.cleanup.sigkill_sent)
        self.assertEqual(actions, [0])
        self.assertEqual(process.timeouts, [1.0, 0])


if __name__ == "__main__":
    unittest.main()
