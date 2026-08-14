"""Low-overhead host and GPU telemetry sampling."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

from .journal import utc_now


GPU_QUERY = (
    "timestamp,power.draw,temperature.gpu,utilization.gpu,"
    "utilization.memory,clocks.current.sm"
)


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, raw = line.split(":", 1)
        number = raw.strip().split()[0]
        if name in {"MemAvailable", "MemFree", "Cached", "SwapTotal", "SwapFree"}:
            values[f"{name.lower()}_kib"] = int(number)
    return values


def _gpu_sample() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--query-gpu={GPU_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    if result.returncode:
        return {"gpu_error": result.stderr.strip()[:500]}
    row = next(csv.reader([result.stdout.strip()]))
    names = ["gpu_timestamp", "power_w", "temperature_c", "gpu_util_pct", "memory_util_pct", "sm_clock_mhz"]
    parsed: dict[str, Any] = {}
    for name, raw in zip(names, row, strict=False):
        value = raw.strip()
        try:
            parsed[name] = float(value)
        except ValueError:
            parsed[name] = value
    return parsed


class TelemetrySampler:
    def __init__(self, path: Path, interval_s: float = 1.0) -> None:
        self.path = path
        self.interval_s = interval_s
        self.phase = "idle"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="sparkbench-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(5, self.interval_s * 2))

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                phase = self.phase
            sample = {"timestamp": utc_now(), "phase": phase, **_meminfo(), **_gpu_sample()}
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(sample, sort_keys=True) + "\n")
                stream.flush()
            self._stop.wait(self.interval_s)
