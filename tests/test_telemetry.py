from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from bench.report import _telemetry_summaries
from bench.telemetry import GPU_SAMPLE_FIELDS, TelemetrySampler, _gpu_sample


class GpuTelemetryTests(unittest.TestCase):
    def test_timeout_returns_explicit_missing_values(self) -> None:
        with patch(
            "bench.telemetry.subprocess.run",
            side_effect=subprocess.TimeoutExpired("nvidia-smi", 5),
        ) as run:
            sample = _gpu_sample()

        run.assert_called_once()
        self.assertEqual(
            sample["gpu_error"], "nvidia-smi timed out after 5 seconds"
        )
        for field in GPU_SAMPLE_FIELDS:
            self.assertIn(field, sample)
            self.assertIsNone(sample[field])

    def test_command_error_returns_explicit_missing_values(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=9,
            stdout="",
            stderr="driver temporarily unavailable",
        )
        with patch("bench.telemetry.subprocess.run", return_value=completed):
            sample = _gpu_sample()

        self.assertEqual(
            sample["gpu_error"],
            "nvidia-smi failed: driver temporarily unavailable",
        )
        self.assertTrue(all(sample[field] is None for field in GPU_SAMPLE_FIELDS))

    def test_sampler_continues_after_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            sampler = TelemetrySampler(path, interval_s=0)
            calls = 0

            def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise subprocess.TimeoutExpired("nvidia-smi", 5)
                sampler._stop.set()
                return subprocess.CompletedProcess(
                    args=["nvidia-smi"],
                    returncode=0,
                    stdout="2026/08/16 03:00:00, 48.5, 52, 91, 4, 1200\n",
                    stderr="",
                )

            with (
                patch("bench.telemetry.subprocess.run", side_effect=run),
                patch(
                    "bench.telemetry._meminfo",
                    return_value={"memavailable_kib": 1024},
                ),
            ):
                sampler._run()

            samples = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(calls, 2)
        self.assertEqual(len(samples), 2)
        self.assertIsNone(samples[0]["power_w"])
        self.assertIn("timed out", samples[0]["gpu_error"])
        self.assertEqual(samples[1]["power_w"], 48.5)
        self.assertNotIn("gpu_error", samples[1])


class TelemetrySummaryTests(unittest.TestCase):
    def test_missing_power_is_not_reported_as_zero_energy(self) -> None:
        samples = [
            {
                "timestamp": "2026-08-16T03:00:00+00:00",
                "phase": "case:decode:attempt",
                "power_w": 100.0,
            },
            {
                "timestamp": "2026-08-16T03:00:01+00:00",
                "phase": "case:decode:attempt",
                "power_w": None,
                "gpu_error": "nvidia-smi timed out after 5 seconds",
            },
            {
                "timestamp": "2026-08-16T03:00:02+00:00",
                "phase": "case:decode:attempt",
                "power_w": 120.0,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            path.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
            summary = _telemetry_summaries(path)["case:decode:attempt"]

        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["gpu_power_samples"], 2)
        self.assertEqual(summary["gpu_power_missing_samples"], 1)
        self.assertEqual(summary["gpu_error_samples"], 1)
        self.assertEqual(summary["average_power_w"], 110.0)
        self.assertIsNone(summary["sampled_energy_j"])
        self.assertEqual(summary["sampled_energy_intervals"], 0)

    def test_all_missing_gpu_metrics_remain_none(self) -> None:
        samples = [
            {
                "timestamp": f"2026-08-16T03:00:0{second}+00:00",
                "phase": "startup",
                "power_w": None,
                "gpu_util_pct": None,
                "gpu_error": "temporary failure",
            }
            for second in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            path.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
            summary = _telemetry_summaries(path)["startup"]

        self.assertIsNone(summary["average_power_w"])
        self.assertIsNone(summary["peak_power_w"])
        self.assertIsNone(summary["average_gpu_util_pct"])
        self.assertIsNone(summary["sampled_energy_j"])
        self.assertEqual(summary["gpu_power_missing_samples"], 2)


if __name__ == "__main__":
    unittest.main()
