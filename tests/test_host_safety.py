from __future__ import annotations

from decimal import Decimal
import threading
import unittest

from bench.host_safety import (
    DEFAULT_INTERVAL_S,
    HostSafetyError,
    HostSafetyWatchdog,
    MeminfoError,
    parse_meminfo,
)


KIB_PER_MIB = 1024
KIB_PER_GIB = 1024**2


def _meminfo(
    *,
    mem_total_kib: int = 64 * KIB_PER_GIB,
    memavailable_kib: int = 32 * KIB_PER_GIB,
    swap_total_kib: int = 8 * KIB_PER_GIB,
    swap_used_kib: int = 0,
) -> str:
    return "\n".join(
        [
            f"MemTotal: {mem_total_kib} kB",
            "MemFree: 1234 kB",
            f"MemAvailable: {memavailable_kib} kB",
            f"SwapTotal: {swap_total_kib} kB",
            f"SwapFree: {swap_total_kib - swap_used_kib} kB",
            "Cached: 5678 kB",
        ]
    )


def _watchdog(
    reader: object,
    **kwargs: object,
) -> HostSafetyWatchdog:
    options = {
        "min_memavailable_gib": 14,
        "max_swap_growth_mib": 512,
        "max_starting_swap_mib": 512,
        "meminfo_reader": reader,
    }
    options.update(kwargs)
    return HostSafetyWatchdog(**options)  # type: ignore[arg-type]


class MeminfoParsingTests(unittest.TestCase):
    def test_parser_returns_integer_kib_sample(self) -> None:
        sample = parse_meminfo(
            _meminfo(
                memavailable_kib=20 * KIB_PER_GIB,
                swap_total_kib=4 * KIB_PER_GIB,
                swap_used_kib=73 * KIB_PER_MIB,
            )
        )

        self.assertEqual(sample.mem_total_kib, 64 * KIB_PER_GIB)
        self.assertEqual(sample.memavailable_kib, 20 * KIB_PER_GIB)
        self.assertEqual(sample.swap_total_kib, 4 * KIB_PER_GIB)
        self.assertEqual(sample.swap_used_kib, 73 * KIB_PER_MIB)

    def test_missing_required_field_is_rejected(self) -> None:
        raw = _meminfo().replace("MemAvailable: 33554432 kB\n", "")

        with self.assertRaises(MeminfoError) as raised:
            parse_meminfo(raw)

        self.assertEqual(raised.exception.code, "meminfo_missing")
        self.assertIn("MemAvailable", str(raised.exception))

    def test_malformed_or_duplicate_required_field_is_rejected(self) -> None:
        malformed_samples = [
            _meminfo().replace("MemAvailable: 33554432 kB", "MemAvailable: nope kB"),
            _meminfo().replace("MemAvailable: 33554432 kB", "MemAvailable: 2 MB"),
            _meminfo() + "\nSwapFree: 0 kB",
        ]

        for raw in malformed_samples:
            with self.subTest(raw=raw):
                with self.assertRaises(MeminfoError) as raised:
                    parse_meminfo(raw)
                self.assertEqual(raised.exception.code, "meminfo_malformed")

    def test_inconsistent_values_are_rejected(self) -> None:
        inconsistent_samples = [
            _meminfo(mem_total_kib=100, memavailable_kib=101),
            _meminfo(swap_total_kib=100, swap_used_kib=-1),
            _meminfo().replace("MemTotal: 67108864 kB", "MemTotal: -1 kB"),
        ]

        for raw in inconsistent_samples:
            with self.subTest(raw=raw):
                with self.assertRaises(MeminfoError) as raised:
                    parse_meminfo(raw)
                self.assertEqual(raised.exception.code, "meminfo_inconsistent")


class HostSafetyWatchdogTests(unittest.TestCase):
    def test_thresholds_are_required_and_converted_to_integer_kib(self) -> None:
        watchdog = HostSafetyWatchdog(
            min_memavailable_gib=14,
            max_swap_growth_mib=Decimal("512"),
            max_starting_swap_mib=0.5,
            meminfo_reader=_meminfo,
        )

        self.assertEqual(watchdog.min_memavailable_kib, 14 * KIB_PER_GIB)
        self.assertEqual(watchdog.max_swap_growth_kib, 512 * KIB_PER_MIB)
        self.assertEqual(watchdog.max_starting_swap_kib, 512)
        self.assertEqual(watchdog.interval_s, DEFAULT_INTERVAL_S)

        with self.assertRaises(TypeError):
            HostSafetyWatchdog()  # type: ignore[call-arg]
        with self.assertRaises(ValueError):
            _watchdog(_meminfo, max_swap_growth_mib=0.1)
        with self.assertRaises(ValueError):
            _watchdog(_meminfo, interval_s=0)

    def test_start_samples_synchronously_and_fixes_swap_baseline(self) -> None:
        calls: list[str] = []

        def reader() -> str:
            calls.append("read")
            return _meminfo(swap_used_kib=81 * KIB_PER_MIB)

        watchdog = _watchdog(reader)
        try:
            returned = watchdog.start()
            self.assertIs(returned, watchdog)
            self.assertEqual(calls, ["read"])
            self.assertEqual(
                watchdog.starting_swap_used_kib, 81 * KIB_PER_MIB
            )
        finally:
            watchdog.stop()

    def test_initial_memavailable_failure_is_synchronous(self) -> None:
        callbacks: list[str] = []
        watchdog = _watchdog(
            lambda: _meminfo(memavailable_kib=14 * KIB_PER_GIB - 1)
        )
        watchdog.register_abort_callback(lambda: callbacks.append("abort"))

        with self.assertRaises(HostSafetyError) as raised:
            watchdog.start()

        self.assertEqual(raised.exception.code, "memavailable_below_minimum")
        self.assertEqual(raised.exception.observed_kib, 14 * KIB_PER_GIB - 1)
        self.assertEqual(raised.exception.limit_kib, 14 * KIB_PER_GIB)
        self.assertEqual(callbacks, ["abort"])

    def test_starting_swap_maximum_is_inclusive(self) -> None:
        samples = iter(
            [
                _meminfo(swap_used_kib=512 * KIB_PER_MIB),
                _meminfo(swap_used_kib=512 * KIB_PER_MIB + 1),
            ]
        )
        safe = _watchdog(lambda: next(samples))
        safe.sample_once()
        safe.raise_if_tripped()

        unsafe = _watchdog(lambda: next(samples))
        unsafe.sample_once()
        with self.assertRaises(HostSafetyError) as raised:
            unsafe.raise_if_tripped()

        self.assertEqual(raised.exception.code, "starting_swap_above_maximum")
        self.assertEqual(raised.exception.limit_kib, 512 * KIB_PER_MIB)

    def test_swap_growth_uses_fixed_baseline_and_maximum_is_inclusive(self) -> None:
        baseline = 100 * KIB_PER_MIB
        samples = iter(
            [
                _meminfo(swap_used_kib=baseline),
                _meminfo(swap_used_kib=0),
                _meminfo(swap_used_kib=baseline + 512 * KIB_PER_MIB),
                _meminfo(swap_used_kib=baseline + 512 * KIB_PER_MIB + 1),
            ]
        )
        watchdog = _watchdog(lambda: next(samples))

        for _ in range(3):
            watchdog.sample_once()
            watchdog.raise_if_tripped()
            self.assertEqual(watchdog.starting_swap_used_kib, baseline)
        watchdog.sample_once()

        with self.assertRaises(HostSafetyError) as raised:
            watchdog.raise_if_tripped()
        self.assertEqual(raised.exception.code, "swap_growth_above_maximum")
        self.assertEqual(raised.exception.observed_kib, 512 * KIB_PER_MIB + 1)
        self.assertEqual(raised.exception.starting_swap_used_kib, baseline)

    def test_swap_total_change_trips_before_using_incomparable_delta(self) -> None:
        samples = iter(
            [
                _meminfo(
                    swap_total_kib=8 * KIB_PER_GIB,
                    swap_used_kib=100 * KIB_PER_MIB,
                ),
                _meminfo(
                    swap_total_kib=9 * KIB_PER_GIB,
                    swap_used_kib=100 * KIB_PER_MIB,
                ),
            ]
        )
        watchdog = _watchdog(lambda: next(samples))
        watchdog.sample_once()
        watchdog.sample_once()

        with self.assertRaises(HostSafetyError) as raised:
            watchdog.raise_if_tripped()

        self.assertEqual(raised.exception.code, "swap_total_changed")
        self.assertEqual(raised.exception.observed_kib, 9 * KIB_PER_GIB)
        self.assertEqual(raised.exception.limit_kib, 8 * KIB_PER_GIB)

    def test_memavailable_minimum_is_inclusive(self) -> None:
        watchdog = _watchdog(
            lambda: _meminfo(memavailable_kib=14 * KIB_PER_GIB)
        )

        watchdog.sample_once()

        watchdog.raise_if_tripped()
        self.assertFalse(watchdog.tripped)

    def test_reader_and_parser_failures_trip_closed(self) -> None:
        def fail_read() -> str:
            raise OSError("synthetic read failure")

        cases = [
            (fail_read, "meminfo_read_failed"),
            (lambda: "MemTotal: 1 kB", "meminfo_missing"),
            (
                lambda: _meminfo().replace(
                    "MemAvailable: 33554432 kB", "MemAvailable: invalid kB"
                ),
                "meminfo_malformed",
            ),
            (
                lambda: _meminfo(swap_total_kib=1, swap_used_kib=-1),
                "meminfo_inconsistent",
            ),
        ]

        for reader, code in cases:
            with self.subTest(code=code):
                watchdog = _watchdog(reader)
                watchdog.sample_once()
                with self.assertRaises(HostSafetyError) as raised:
                    watchdog.raise_if_tripped()
                self.assertEqual(raised.exception.code, code)

    def test_first_failure_wins_and_callback_runs_only_once(self) -> None:
        calls = 0
        callback_calls = 0

        def reader() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                return _meminfo(memavailable_kib=1)
            return "malformed"

        def abort() -> None:
            nonlocal callback_calls
            callback_calls += 1

        watchdog = _watchdog(reader)
        watchdog.register_abort_callback(abort)
        watchdog.sample_once()
        original = watchdog.failure
        watchdog.sample_once()

        self.assertIs(watchdog.failure, original)
        self.assertEqual(original.code, "memavailable_below_minimum")  # type: ignore[union-attr]
        self.assertEqual(calls, 1)
        self.assertEqual(callback_calls, 1)
        with self.assertRaises(HostSafetyError) as raised:
            watchdog.raise_if_tripped()
        self.assertIs(raised.exception, original)

    def test_late_callback_is_invoked_and_only_one_can_be_registered(self) -> None:
        watchdog = _watchdog(lambda: _meminfo(memavailable_kib=1))
        watchdog.sample_once()
        calls: list[str] = []

        watchdog.register_abort_callback(lambda: calls.append("late"))

        self.assertEqual(calls, ["late"])
        with self.assertRaises(RuntimeError):
            watchdog.register_abort_callback(lambda: calls.append("second"))
        self.assertEqual(calls, ["late"])

    def test_callback_error_does_not_replace_first_safety_failure(self) -> None:
        marker = RuntimeError("synthetic callback failure")

        def abort() -> None:
            raise marker

        watchdog = _watchdog(lambda: _meminfo(memavailable_kib=1))
        watchdog.register_abort_callback(abort)
        watchdog.sample_once()

        self.assertEqual(watchdog.failure.code, "memavailable_below_minimum")  # type: ignore[union-attr]
        self.assertIs(watchdog.abort_callback_error, marker)

    def test_monitor_thread_failure_trips_and_invokes_abort(self) -> None:
        callback_called = threading.Event()
        release_wait = threading.Event()

        def fail_wait(_interval_s: float) -> bool:
            release_wait.wait(timeout=1)
            raise RuntimeError("synthetic monitor failure")

        watchdog = _watchdog(
            _meminfo,
            wait_function=fail_wait,
        )
        watchdog.register_abort_callback(callback_called.set)
        watchdog.start()
        release_wait.set()

        self.assertTrue(callback_called.wait(timeout=1))
        with self.assertRaises(HostSafetyError) as raised:
            watchdog.raise_if_tripped()
        self.assertEqual(raised.exception.code, "monitor_thread_failed")
        self.assertIn("synthetic monitor failure", str(raised.exception))
        watchdog.stop()

    def test_stopped_watchdog_cannot_be_restarted(self) -> None:
        watchdog = _watchdog(_meminfo)
        watchdog.start()
        watchdog.stop()

        with self.assertRaises(RuntimeError):
            watchdog.start()

    def test_bounded_stop_trips_closed_if_monitor_does_not_quiesce(self) -> None:
        release_wait = threading.Event()
        callbacks: list[str] = []

        def blocked_wait(_interval_s: float) -> bool:
            release_wait.wait(timeout=2)
            return True

        watchdog = _watchdog(_meminfo, wait_function=blocked_wait)
        watchdog.register_abort_callback(lambda: callbacks.append("abort"))
        watchdog.start()
        try:
            watchdog.stop()

            with self.assertRaises(HostSafetyError) as raised:
                watchdog.raise_if_tripped()
            self.assertEqual(raised.exception.code, "monitor_thread_failed")
            self.assertEqual(callbacks, ["abort"])
        finally:
            release_wait.set()
            watchdog.stop()


if __name__ == "__main__":
    unittest.main()
