"""Fail-closed host-memory safety monitoring for managed benchmarks.

The observational telemetry sampler deliberately tolerates probe failures.  This
module has the opposite contract: once enabled by a caller, an unreadable or
internally inconsistent ``/proc/meminfo`` sample is itself a safety failure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
import threading
from typing import TypeAlias


DEFAULT_INTERVAL_S = 0.25
KIB_PER_MIB = 1024
KIB_PER_GIB = 1024**2

Threshold: TypeAlias = int | float | Decimal
MeminfoReader: TypeAlias = Callable[[], str]
WaitFunction: TypeAlias = Callable[[float], bool]
AbortCallback: TypeAlias = Callable[[], None]


@dataclass(frozen=True)
class HostMemorySample:
    """The meminfo values needed by the safety gates, all in integer KiB."""

    mem_total_kib: int
    memavailable_kib: int
    swap_total_kib: int
    swap_free_kib: int

    @property
    def swap_used_kib(self) -> int:
        return self.swap_total_kib - self.swap_free_kib


class MeminfoError(ValueError):
    """Raised when a host-memory sample cannot be trusted."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class HostSafetyError(RuntimeError):
    """A stable first-failure record for a tripped host-safety watchdog."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        sample: HostMemorySample | None = None,
        observed_kib: int | None = None,
        limit_kib: int | None = None,
        starting_swap_used_kib: int | None = None,
    ) -> None:
        self.code = code
        self.sample = sample
        self.observed_kib = observed_kib
        self.limit_kib = limit_kib
        self.starting_swap_used_kib = starting_swap_used_kib
        super().__init__(message)


def read_host_meminfo() -> str:
    """Read the kernel host-memory snapshot used by the watchdog."""

    return Path("/proc/meminfo").read_text(encoding="utf-8")


def parse_meminfo(raw: str) -> HostMemorySample:
    """Parse and cross-check the required fields from one meminfo snapshot."""

    if not isinstance(raw, str):
        raise MeminfoError("meminfo_malformed", "meminfo was not text")

    required = {
        "MemTotal": "mem_total_kib",
        "MemAvailable": "memavailable_kib",
        "SwapTotal": "swap_total_kib",
        "SwapFree": "swap_free_kib",
    }
    values: dict[str, int] = {}
    for line in raw.splitlines():
        name, separator, value_text = line.partition(":")
        name = name.strip()
        if name not in required:
            continue
        if not separator:
            raise MeminfoError(
                "meminfo_malformed", f"meminfo field {name} has no separator"
            )
        if name in values:
            raise MeminfoError(
                "meminfo_malformed", f"meminfo field {name} is duplicated"
            )
        parts = value_text.split()
        if len(parts) != 2 or parts[1] != "kB":
            raise MeminfoError(
                "meminfo_malformed",
                f"meminfo field {name} is not an integer KiB value",
            )
        try:
            value = int(parts[0], 10)
        except ValueError as error:
            raise MeminfoError(
                "meminfo_malformed",
                f"meminfo field {name} is not an integer KiB value",
            ) from error
        if value < 0:
            raise MeminfoError(
                "meminfo_inconsistent", f"meminfo field {name} is negative"
            )
        values[name] = value

    missing = sorted(set(required) - set(values))
    if missing:
        raise MeminfoError(
            "meminfo_missing",
            "meminfo is missing required fields: " + ", ".join(missing),
        )

    sample = HostMemorySample(
        **{attribute: values[name] for name, attribute in required.items()}
    )
    if sample.memavailable_kib > sample.mem_total_kib:
        raise MeminfoError(
            "meminfo_inconsistent", "MemAvailable exceeds MemTotal"
        )
    if sample.swap_free_kib > sample.swap_total_kib:
        raise MeminfoError("meminfo_inconsistent", "SwapFree exceeds SwapTotal")
    return sample


def _threshold_to_kib(value: Threshold, multiplier: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative finite number")
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be a non-negative finite number") from error
    if not decimal_value.is_finite() or decimal_value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    kib = decimal_value * multiplier
    integral_kib = kib.to_integral_value()
    if kib != integral_kib:
        raise ValueError(f"{name} must resolve to a whole number of KiB")
    return int(integral_kib)


class HostSafetyWatchdog:
    """Monitor host memory and request one exact, caller-owned abort on failure.

    Thresholds are required keyword arguments so monitoring is explicitly opted
    into.  ``start()`` reads and evaluates the starting sample before it creates
    the background thread.  The first successful sample fixes the swap-used
    baseline for the lifetime of the instance.

    The optional reader and wait function are dependency-injection seams for
    unit tests.  A wait function has the same contract as ``Event.wait``: it
    returns true only after the watchdog has been asked to stop.
    """

    def __init__(
        self,
        *,
        min_memavailable_gib: Threshold,
        max_swap_growth_mib: Threshold,
        max_starting_swap_mib: Threshold,
        interval_s: float = DEFAULT_INTERVAL_S,
        meminfo_reader: MeminfoReader = read_host_meminfo,
        wait_function: WaitFunction | None = None,
    ) -> None:
        if isinstance(interval_s, bool) or not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("interval_s must be a positive finite number")
        if not callable(meminfo_reader):
            raise TypeError("meminfo_reader must be callable")
        if wait_function is not None and not callable(wait_function):
            raise TypeError("wait_function must be callable")

        self.min_memavailable_kib = _threshold_to_kib(
            min_memavailable_gib, KIB_PER_GIB, "min_memavailable_gib"
        )
        self.max_swap_growth_kib = _threshold_to_kib(
            max_swap_growth_mib, KIB_PER_MIB, "max_swap_growth_mib"
        )
        self.max_starting_swap_kib = _threshold_to_kib(
            max_starting_swap_mib, KIB_PER_MIB, "max_starting_swap_mib"
        )
        self.interval_s = float(interval_s)
        self._meminfo_reader = meminfo_reader
        self._stop_event = threading.Event()
        self._wait_function = wait_function or self._stop_event.wait
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._failure: HostSafetyError | None = None
        self._last_sample: HostMemorySample | None = None
        self._starting_swap_total_kib: int | None = None
        self._starting_swap_used_kib: int | None = None
        self._abort_callback: AbortCallback | None = None
        self._abort_callback_registered = False
        self._abort_callback_invoked = False
        self._abort_callback_error: BaseException | None = None

    @property
    def failure(self) -> HostSafetyError | None:
        with self._lock:
            return self._failure

    @property
    def tripped(self) -> bool:
        return self.failure is not None

    @property
    def last_sample(self) -> HostMemorySample | None:
        with self._lock:
            return self._last_sample

    @property
    def starting_swap_used_kib(self) -> int | None:
        with self._lock:
            return self._starting_swap_used_kib

    @property
    def abort_callback_error(self) -> BaseException | None:
        """Return a callback exception without replacing the safety failure."""

        with self._lock:
            return self._abort_callback_error

    def register_abort_callback(self, callback: AbortCallback) -> None:
        """Register the sole abort callback, invoking it now if already tripped."""

        if not callable(callback):
            raise TypeError("abort callback must be callable")
        invoke: AbortCallback | None = None
        with self._lock:
            if self._abort_callback_registered:
                raise RuntimeError("an abort callback is already registered")
            self._abort_callback_registered = True
            self._abort_callback = callback
            if self._failure is not None and not self._abort_callback_invoked:
                self._abort_callback_invoked = True
                invoke = callback
        if invoke is not None:
            self._invoke_abort_callback(invoke)

    def sample_once(self) -> HostMemorySample | None:
        """Read and evaluate one sample, tripping instead of propagating probes."""

        with self._lock:
            if self._failure is not None:
                return None
        try:
            raw = self._meminfo_reader()
        except BaseException as error:
            self._trip(
                HostSafetyError(
                    "meminfo_read_failed",
                    "host meminfo read failed: "
                    f"{type(error).__name__}: {str(error)[:300]}",
                )
            )
            return None
        try:
            sample = parse_meminfo(raw)
        except MeminfoError as error:
            self._trip(HostSafetyError(error.code, str(error)))
            return None

        failure: HostSafetyError | None = None
        with self._lock:
            if self._failure is not None:
                return None
            self._last_sample = sample
            starting = self._starting_swap_used_kib is None
            if starting:
                self._starting_swap_total_kib = sample.swap_total_kib
                self._starting_swap_used_kib = sample.swap_used_kib
            starting_swap_total_kib = self._starting_swap_total_kib
            starting_swap_used_kib = self._starting_swap_used_kib

            if (
                not starting
                and sample.swap_total_kib != starting_swap_total_kib
            ):
                failure = HostSafetyError(
                    "swap_total_changed",
                    f"SwapTotal changed from {starting_swap_total_kib} KiB to "
                    f"{sample.swap_total_kib} KiB after the safety baseline",
                    sample=sample,
                    observed_kib=sample.swap_total_kib,
                    limit_kib=starting_swap_total_kib,
                    starting_swap_used_kib=starting_swap_used_kib,
                )
            elif sample.memavailable_kib < self.min_memavailable_kib:
                failure = HostSafetyError(
                    "memavailable_below_minimum",
                    f"MemAvailable {sample.memavailable_kib} KiB is below minimum "
                    f"{self.min_memavailable_kib} KiB",
                    sample=sample,
                    observed_kib=sample.memavailable_kib,
                    limit_kib=self.min_memavailable_kib,
                    starting_swap_used_kib=starting_swap_used_kib,
                )
            elif starting and sample.swap_used_kib > self.max_starting_swap_kib:
                failure = HostSafetyError(
                    "starting_swap_above_maximum",
                    f"starting swap used {sample.swap_used_kib} KiB exceeds maximum "
                    f"{self.max_starting_swap_kib} KiB",
                    sample=sample,
                    observed_kib=sample.swap_used_kib,
                    limit_kib=self.max_starting_swap_kib,
                    starting_swap_used_kib=starting_swap_used_kib,
                )
            else:
                swap_growth_kib = sample.swap_used_kib - starting_swap_used_kib
                if swap_growth_kib > self.max_swap_growth_kib:
                    failure = HostSafetyError(
                        "swap_growth_above_maximum",
                        f"swap growth {swap_growth_kib} KiB exceeds maximum "
                        f"{self.max_swap_growth_kib} KiB",
                        sample=sample,
                        observed_kib=swap_growth_kib,
                        limit_kib=self.max_swap_growth_kib,
                        starting_swap_used_kib=starting_swap_used_kib,
                    )
        if failure is not None:
            self._trip(failure)
        return sample

    def start(self) -> HostSafetyWatchdog:
        """Synchronously establish the baseline, then begin live monitoring."""

        with self._lock:
            if self._started:
                raise RuntimeError("host-safety watchdog is already started")
            if self._closed:
                raise RuntimeError("host-safety watchdog is already stopped")

        self.sample_once()
        self.raise_if_tripped()

        thread = threading.Thread(
            target=self._run,
            name="sparkbench-host-safety",
            daemon=True,
        )
        with self._lock:
            self._thread = thread
        try:
            thread.start()
        except BaseException as error:
            with self._lock:
                self._thread = None
            self._trip(
                HostSafetyError(
                    "monitor_thread_failed",
                    "host-safety monitor could not start: "
                    f"{type(error).__name__}: {str(error)[:300]}",
                )
            )
            self.raise_if_tripped()
        with self._lock:
            self._started = True
        self.raise_if_tripped()
        return self

    def stop(self) -> None:
        """Signal stop and make a bounded join from non-monitor threads.

        This method is safe after a trip and in ordinary ownership ``finally``
        cleanup.  It never attempts to join itself when an abort callback runs
        on the monitor thread.
        """

        with self._lock:
            self._closed = True
            self._stop_event.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.interval_s * 4))
            if thread.is_alive():
                self._trip(
                    HostSafetyError(
                        "monitor_thread_failed",
                        "host-safety monitor did not stop within its bounded join",
                    )
                )

    def raise_if_tripped(self) -> None:
        """Raise the watchdog's original failure, if any."""

        with self._lock:
            failure = self._failure
            thread = self._thread
            unexpectedly_stopped = (
                failure is None
                and self._started
                and not self._closed
                and thread is not None
                and thread is not threading.current_thread()
                and not thread.is_alive()
            )
        if unexpectedly_stopped:
            self._trip(
                HostSafetyError(
                    "monitor_thread_failed",
                    "host-safety monitor thread stopped unexpectedly",
                )
            )
            with self._lock:
                failure = self._failure
        if failure is not None:
            raise failure

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                wait_returned = self._wait_function(self.interval_s)
                if wait_returned:
                    if self._stop_event.is_set():
                        return
                    raise RuntimeError("wait function returned before stop was requested")
                if self._stop_event.is_set():
                    return
                self.sample_once()
        except BaseException as error:
            self._trip(
                HostSafetyError(
                    "monitor_thread_failed",
                    "host-safety monitor failed: "
                    f"{type(error).__name__}: {str(error)[:300]}",
                )
            )

    def _trip(self, failure: HostSafetyError) -> None:
        callback: AbortCallback | None = None
        with self._lock:
            if self._failure is not None:
                return
            self._failure = failure
            self._stop_event.set()
            if (
                self._abort_callback is not None
                and not self._abort_callback_invoked
            ):
                self._abort_callback_invoked = True
                callback = self._abort_callback
        if callback is not None:
            self._invoke_abort_callback(callback)

    def _invoke_abort_callback(self, callback: AbortCallback) -> None:
        try:
            callback()
        except BaseException as error:
            with self._lock:
                self._abort_callback_error = error


__all__ = [
    "DEFAULT_INTERVAL_S",
    "HostMemorySample",
    "HostSafetyError",
    "HostSafetyWatchdog",
    "MeminfoError",
    "parse_meminfo",
    "read_host_meminfo",
]
