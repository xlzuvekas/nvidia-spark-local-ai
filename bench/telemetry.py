"""Low-overhead host and GPU telemetry sampling."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
import threading
from typing import Any

from .journal import utc_now


GPU_QUERY = (
    "timestamp,power.draw,temperature.gpu,utilization.gpu,"
    "utilization.memory,clocks.current.sm"
)
GPU_SAMPLE_FIELDS = (
    "gpu_timestamp",
    "power_w",
    "temperature_c",
    "gpu_util_pct",
    "memory_util_pct",
    "sm_clock_mhz",
)
GPU_MISSING_MARKERS = {"", "n/a", "[not supported]", "not supported", "unknown"}


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, raw = line.split(":", 1)
        number = raw.strip().split()[0]
        if name in {"MemAvailable", "MemFree", "Cached", "SwapTotal", "SwapFree"}:
            values[f"{name.lower()}_kib"] = int(number)
    return values


def _missing_gpu_sample(error: str) -> dict[str, Any]:
    return {
        **dict.fromkeys(GPU_SAMPLE_FIELDS),
        "gpu_error": error.strip()[:500] or "unknown nvidia-smi error",
    }


def _gpu_sample() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--query-gpu={GPU_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return _missing_gpu_sample("nvidia-smi timed out after 5 seconds")
    except OSError as error:
        return _missing_gpu_sample(
            f"nvidia-smi could not be executed: {type(error).__name__}: {error}"
        )
    if result.returncode:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        return _missing_gpu_sample(f"nvidia-smi failed: {detail}")
    try:
        row = next(csv.reader(result.stdout.splitlines()))
    except (csv.Error, StopIteration) as error:
        return _missing_gpu_sample(
            f"nvidia-smi returned invalid CSV: {type(error).__name__}: {error}"
        )
    if not row:
        return _missing_gpu_sample("nvidia-smi returned an empty row")
    parsed: dict[str, Any] = dict.fromkeys(GPU_SAMPLE_FIELDS)
    for name, raw in zip(GPU_SAMPLE_FIELDS, row, strict=False):
        value = raw.strip()
        if value.lower() in GPU_MISSING_MARKERS:
            continue
        if name == "gpu_timestamp":
            parsed[name] = value
            continue
        try:
            numeric = float(value)
        except ValueError:
            continue
        if math.isfinite(numeric):
            parsed[name] = numeric
    if len(row) != len(GPU_SAMPLE_FIELDS):
        parsed["gpu_error"] = (
            "nvidia-smi returned "
            f"{len(row)} fields; expected {len(GPU_SAMPLE_FIELDS)}"
        )
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
            try:
                gpu_sample = _gpu_sample()
            except Exception as error:
                # Telemetry is observational: an unexpected probe failure must not
                # abort a benchmark or permanently stop later samples.
                gpu_sample = _missing_gpu_sample(
                    f"unexpected GPU telemetry error: {type(error).__name__}: {error}"
                )
            sample = {
                "timestamp": utc_now(),
                "phase": phase,
                **_meminfo(),
                **gpu_sample,
            }
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(sample, sort_keys=True) + "\n")
                stream.flush()
            self._stop.wait(self.interval_s)
