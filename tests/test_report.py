from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from bench.report import percentile, summarize_run


def _request(
    case_id: str,
    attempt_id: str,
    *,
    ttft_s: float,
    elapsed_s: float,
    decode_tps: float,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    kind: str = "decode",
) -> dict[str, object]:
    return {
        "event": "request_complete",
        "case_id": case_id,
        "attempt_id": attempt_id,
        "kind": kind,
        "result": {
            "ttft_s": ttft_s,
            "elapsed_s": elapsed_s,
            "decode_tps": decode_tps,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = Path(self.temporary.name)

    def _write_events(self, events: list[dict[str, object]]) -> None:
        lines = [json.dumps(event) for event in events]
        lines.insert(1, "partial invalid journal line")
        (self.run_dir / "events.jsonl").write_text("\n".join(lines) + "\n")

    def test_percentile_interpolates_and_handles_empty_input(self) -> None:
        self.assertIsNone(percentile([], 0.95))
        self.assertEqual(percentile([4.0], 0.95), 4.0)
        self.assertAlmostEqual(percentile([0.0, 10.0], 0.25), 2.5)
        self.assertAlmostEqual(percentile([float(value) for value in range(20)], 0.95), 18.05)

    def test_summarizes_only_latest_completed_attempt_and_writes_outputs(self) -> None:
        events = [
            _request("decode", "old", ttft_s=9, elapsed_s=9, decode_tps=1),
            {"event": "case_complete", "case_id": "decode", "attempt_id": "old", "elapsed_s": 9},
            _request("decode", "new", ttft_s=0.1, elapsed_s=1.0, decode_tps=19),
            _request("decode", "new", ttft_s=0.2, elapsed_s=1.2, decode_tps=20),
            _request("decode", "new", ttft_s=0.3, elapsed_s=1.4, decode_tps=21),
            {
                "event": "case_complete",
                "case_id": "decode",
                "attempt_id": "new",
                "elapsed_s": 2.0,
                "concurrency": 2,
                "validation_passed": True,
            },
            _request("unfinished", "attempt", ttft_s=1, elapsed_s=1, decode_tps=1),
        ]
        self._write_events(events)

        summary = summarize_run(self.run_dir)

        self.assertEqual(summary["completed_cases"], 1)
        row = summary["cases"][0]
        self.assertEqual(row["attempt_id"], "new")
        self.assertEqual(row["requests"], 3)
        self.assertEqual(row["concurrency"], 2)
        self.assertEqual(row["prompt_tokens"], 30)
        self.assertEqual(row["completion_tokens"], 60)
        self.assertAlmostEqual(row["median_ttft_s"], 0.2)
        self.assertAlmostEqual(row["median_e2e_s"], 1.2)
        self.assertAlmostEqual(row["median_estimated_decode_tps"], 20)
        self.assertAlmostEqual(row["aggregate_output_tps"], 30)
        self.assertAlmostEqual(row["request_tps"], 1.5)
        self.assertIsNone(row["p95_ttft_s"])
        self.assertTrue((self.run_dir / "summary.json").is_file())
        with (self.run_dir / "summary.csv").open(newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual(len(csv_rows), 1)
        self.assertEqual(csv_rows[0]["attempt_id"], "new")

    def test_prefill_and_p95_metrics(self) -> None:
        events = [
            _request(
                "prefill",
                "one",
                ttft_s=0.25,
                elapsed_s=0.3,
                decode_tps=0,
                prompt_tokens=1000,
                completion_tokens=1,
                kind="prefill",
            ),
            {
                "event": "case_complete",
                "case_id": "prefill",
                "attempt_id": "one",
                "elapsed_s": 0.3,
            },
        ]
        for index in range(20):
            events.append(
                _request(
                    "many",
                    "many-attempt",
                    ttft_s=float(index),
                    elapsed_s=float(index + 1),
                    decode_tps=float(index + 10),
                )
            )
        events.append(
            {
                "event": "case_complete",
                "case_id": "many",
                "attempt_id": "many-attempt",
                "elapsed_s": 30,
            }
        )
        self._write_events(events)

        summary = summarize_run(self.run_dir)
        rows = {row["case_id"]: row for row in summary["cases"]}

        self.assertAlmostEqual(rows["prefill"]["median_approximate_prefill_tps"], 4000)
        self.assertAlmostEqual(rows["many"]["p95_ttft_s"], 18.05)
        self.assertAlmostEqual(rows["many"]["p95_e2e_s"], 19.05)

    def test_prefill_uses_native_server_timing_or_explicit_client_approximation(self) -> None:
        native = _request(
            "native-prefill",
            "native-attempt",
            ttft_s=1.0,
            elapsed_s=1.1,
            decode_tps=0,
            prompt_tokens=1000,
            completion_tokens=1,
            kind="prefill",
        )
        native["result"]["server_prompt_s"] = 0.1  # type: ignore[index]
        approximate = _request(
            "approximate-prefill",
            "approximate-attempt",
            ttft_s=1.0,
            elapsed_s=1.1,
            decode_tps=0,
            prompt_tokens=1000,
            completion_tokens=1,
            kind="prefill",
        )
        self._write_events(
            [
                native,
                {
                    "event": "case_complete",
                    "case_id": "native-prefill",
                    "attempt_id": "native-attempt",
                    "kind": "prefill",
                    "elapsed_s": 1.1,
                },
                approximate,
                {
                    "event": "case_complete",
                    "case_id": "approximate-prefill",
                    "attempt_id": "approximate-attempt",
                    "kind": "prefill",
                    "elapsed_s": 1.1,
                },
            ]
        )

        rows = {row["case_id"]: row for row in summarize_run(self.run_dir)["cases"]}
        native_row = rows["native-prefill"]
        self.assertAlmostEqual(native_row["median_prefill_tps"], 10_000)
        self.assertEqual(
            native_row["prefill_metric_source"],
            "server_reported_prompt_eval_duration",
        )
        self.assertIsNone(native_row["median_approximate_prefill_tps"])

        approximate_row = rows["approximate-prefill"]
        self.assertAlmostEqual(approximate_row["median_prefill_tps"], 1_000)
        self.assertAlmostEqual(
            approximate_row["median_approximate_prefill_tps"], 1_000
        )
        self.assertEqual(
            approximate_row["prefill_metric_source"], "client_ttft_approximation"
        )

    def test_capability_validation_is_preserved_in_summary(self) -> None:
        self._write_events(
            [
                _request("json", "attempt", ttft_s=0.1, elapsed_s=0.2, decode_tps=10, kind="capability"),
                {
                    "event": "case_complete",
                    "case_id": "json",
                    "attempt_id": "attempt",
                    "elapsed_s": 0.2,
                    "validation_passed": False,
                },
            ]
        )

        row = summarize_run(self.run_dir)["cases"][0]

        self.assertIn("validation_passed", row)
        self.assertFalse(row["validation_passed"])

    def test_status_distinguishes_not_started_and_no_applicable_work(self) -> None:
        self.assertEqual(summarize_run(self.run_dir)["status"], "not_started")

        self._write_events(
            [
                {"event": "run_start"},
                {
                    "event": "case_skipped_unsupported",
                    "case_id": "unsupported",
                },
                {"event": "run_complete", "status": "no_work"},
            ]
        )

        self.assertEqual(summarize_run(self.run_dir)["status"], "no_work")

    def test_adapter_only_no_work_preserves_status_and_skip_details(self) -> None:
        self._write_events(
            [
                {"event": "run_start"},
                {
                    "event": "case_skipped_adapter_unimplemented",
                    "case_id": "rerank-adapter",
                    "capabilities": ["rerank"],
                },
                {"event": "run_complete", "status": "no_work"},
            ]
        )

        summary = summarize_run(self.run_dir)

        self.assertEqual(summary["status"], "no_work")
        self.assertEqual(summary["run_completion_status"], "no_work")
        self.assertEqual(summary["unimplemented_cases"], ["rerank-adapter"])

    def test_invalid_decode_is_excluded_from_throughput_metrics(self) -> None:
        self._write_events(
            [
                _request(
                    "decode",
                    "failed",
                    ttft_s=0.1,
                    elapsed_s=0.2,
                    decode_tps=9999,
                    completion_tokens=1,
                ),
                {
                    "event": "case_complete",
                    "case_id": "decode",
                    "attempt_id": "failed",
                    "kind": "decode",
                    "elapsed_s": 0.2,
                    "validation_passed": False,
                },
            ]
        )

        row = summarize_run(self.run_dir)["cases"][0]
        self.assertIsNone(row["median_decode_tps"])
        self.assertIsNone(row["median_estimated_decode_tps"])
        self.assertIsNone(row["aggregate_output_tps"])


if __name__ == "__main__":
    unittest.main()
