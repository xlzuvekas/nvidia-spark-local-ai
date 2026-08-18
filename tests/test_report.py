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
    decode_tps: float | None,
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

    def test_agentic_summary_reports_task_metrics_without_token_throughput(self) -> None:
        requests = []
        for index, passed in enumerate((True, False, True)):
            request = _request(
                "agentic-two-hop",
                "attempt",
                ttft_s=0.1 + index * 0.1,
                elapsed_s=2.0 + index,
                decode_tps=None,
                prompt_tokens=100 + index,
                completion_tokens=20 + index,
                kind="agentic",
            )
            request["result"].update(  # type: ignore[union-attr]
                {
                    "passed": passed,
                    "wall_s": 2.0 + index,
                    "request_elapsed_s": 1.5 + index,
                    "first_turn_ttft_s": 0.1 + index * 0.1,
                    "max_turns": 6,
                    "max_output_tokens": 4096,
                    "turns_used": 3 + index,
                    "expected_tool_calls": 2,
                    "tool_calls_requested": 2,
                    "tool_calls_executed": 2,
                    "tool_calls_succeeded": 2 if passed else 1,
                    "tool_errors": 0,
                    "malformed_tool_calls": 0 if passed else 1,
                    "unknown_tool_calls": 0,
                    "final_answer_emitted": True,
                    "final_answer_correct": passed,
                    "tool_sequence_correct": passed,
                    "recovery_required": False,
                    "recovery_succeeded": False,
                    "turn_limit_reached": False,
                    "length_terminated_turns": 0,
                }
            )
            request["repetition"] = index
            requests.append(request)
        self._write_events(
            [
                *requests,
                {
                    "event": "case_complete",
                    "case_id": "agentic-two-hop",
                    "attempt_id": "attempt",
                    "kind": "agentic",
                    "elapsed_s": 9.0,
                    "concurrency": 1,
                    "validation_passed": False,
                },
            ]
        )

        row = summarize_run(self.run_dir)["cases"][0]

        self.assertEqual(row["agentic_tasks"], 3)
        self.assertEqual(row["agentic_tasks_succeeded"], 2)
        self.assertEqual(row["agentic_model_requests"], 12)
        self.assertAlmostEqual(row["agentic_model_requests_per_s"], 12 / 9)
        self.assertAlmostEqual(row["agentic_task_success_rate"], 2 / 3)
        self.assertEqual(row["agentic_expected_tool_calls"], 6)
        self.assertEqual(row["agentic_tool_calls_executed"], 6)
        self.assertEqual(row["agentic_malformed_tool_calls"], 1)
        self.assertEqual(row["agentic_final_answers_correct"], 2)
        self.assertEqual(row["agentic_tool_sequences_correct"], 2)
        self.assertEqual(row["median_agentic_turns_used"], 4)
        self.assertEqual(row["median_agentic_task_wall_s"], 3)
        self.assertEqual(row["median_agentic_first_turn_ttft_s"], 0.2)
        self.assertIsNone(row["aggregate_output_tps"])
        self.assertIsNone(row["median_ttft_s"])
        self.assertIsNone(row["request_tps"])
        self.assertFalse(row["validation_passed"])

    def test_capability_with_unavailable_native_decode_timing_stays_null(self) -> None:
        request = _request(
            "ocr",
            "attempt",
            ttft_s=0.1,
            elapsed_s=0.2,
            decode_tps=None,
            completion_tokens=6,
            kind="capability",
        )
        request["result"].update(  # type: ignore[union-attr]
            {"decode_s": None, "decode_metric_source": None}
        )
        self._write_events(
            [
                request,
                {
                    "event": "case_complete",
                    "case_id": "ocr",
                    "attempt_id": "attempt",
                    "kind": "capability",
                    "elapsed_s": 0.2,
                    "validation_passed": True,
                },
            ]
        )

        row = summarize_run(self.run_dir)["cases"][0]

        self.assertIsNone(row["median_decode_tps"])
        self.assertIsNone(row["median_estimated_decode_tps"])
        self.assertIsNone(row["decode_metric_source"])
        self.assertTrue(row["validation_passed"])

    def test_rerank_summary_reports_pair_throughput_and_validation(self) -> None:
        requests = []
        for index, pairs_per_s in enumerate((40.0, 20.0)):
            request = _request(
                "rerank",
                "attempt",
                ttft_s=0.1,
                elapsed_s=0.2,
                decode_tps=0.0,
                prompt_tokens=12,
                completion_tokens=0,
                kind="capability",
            )
            request["result"].update(  # type: ignore[union-attr]
                {
                    "candidate_count": 4,
                    "pairs_per_s": pairs_per_s,
                    "finite": True,
                    "top_index": 1,
                    "ranking": [1, 2, 0, 3],
                    "scores": [0.1, 0.9, 0.2, 0.05],
                }
            )
            request["repetition"] = index
            requests.append(request)
        self._write_events(
            [
                *requests,
                {
                    "event": "case_complete",
                    "case_id": "rerank",
                    "attempt_id": "attempt",
                    "kind": "capability",
                    "elapsed_s": 0.5,
                    "validation_passed": True,
                },
            ]
        )

        row = summarize_run(self.run_dir)["cases"][0]

        self.assertEqual(row["rerank_candidates_per_request"], 4)
        self.assertEqual(row["rerank_pairs"], 8)
        self.assertAlmostEqual(row["median_rerank_pairs_s"], 30.0)
        self.assertAlmostEqual(row["aggregate_rerank_pairs_s"], 16.0)
        self.assertTrue(row["rerank_scores_finite"])
        self.assertEqual(row["rerank_top_index"], 1)
        self.assertTrue(row["rerank_ranking_stable"])
        self.assertTrue(row["rerank_validation_passed"])
        self.assertIsNone(row["aggregate_output_tps"])
        self.assertIsNone(row["median_ttft_s"])

    def test_multimodal_embedding_summary_reports_latency_similarity_and_validation(
        self,
    ) -> None:
        requests = []
        measurements = ((0.01, 0.9, 0.1), (0.03, 0.8, 0.2))
        for latency, relevant, unrelated in measurements:
            request = _request(
                "multimodal",
                "attempt",
                ttft_s=latency,
                elapsed_s=latency * 3,
                decode_tps=0.0,
                prompt_tokens=30,
                completion_tokens=0,
                kind="capability",
            )
            request["result"].update(  # type: ignore[union-attr]
                {
                    "dimension": 3,
                    "batch_size": 3,
                    "items_per_s": 100.0,
                    "finite": True,
                    "image_latency_s": latency,
                    "relevant_text_latency_s": latency + 0.01,
                    "unrelated_text_latency_s": latency + 0.02,
                    "relevant_similarity": relevant,
                    "unrelated_similarity": unrelated,
                    "similarity_margin": relevant - unrelated,
                }
            )
            requests.append(request)
        self._write_events(
            [
                *requests,
                {
                    "event": "case_complete",
                    "case_id": "multimodal",
                    "attempt_id": "attempt",
                    "kind": "capability",
                    "elapsed_s": 0.1,
                    "validation_passed": True,
                },
            ]
        )

        row = summarize_run(self.run_dir)["cases"][0]

        self.assertEqual(row["embedding_dimension"], 3)
        self.assertTrue(row["embeddings_finite"])
        self.assertAlmostEqual(row["median_image_embedding_latency_s"], 0.02)
        self.assertAlmostEqual(row["median_relevant_similarity"], 0.85)
        self.assertAlmostEqual(row["median_unrelated_similarity"], 0.15)
        self.assertAlmostEqual(row["median_similarity_margin"], 0.7)
        self.assertTrue(row["multimodal_embeddings_finite"])
        self.assertTrue(row["multimodal_embedding_validation_passed"])

    def test_quality_summary_reports_accuracy_latency_and_token_totals(self) -> None:
        categories = (
            ("arithmetic", True),
            ("logic", True),
            ("instruction_following", False),
            ("code_reasoning", True),
        )
        requests = []
        for index, (category, passed) in enumerate(categories):
            request = _request(
                "quality",
                "attempt",
                ttft_s=0.01 * (index + 1),
                elapsed_s=0.1 * (index + 1),
                decode_tps=10.0,
                prompt_tokens=25,
                completion_tokens=2,
                kind="quality",
            )
            request["validation"] = {
                "passed": passed,
                "quality_item_id": f"item-{index}",
                "quality_category": category,
            }
            requests.append(request)
        self._write_events(
            [
                *requests,
                {
                    "event": "case_complete",
                    "case_id": "quality",
                    "attempt_id": "attempt",
                    "kind": "quality",
                    "elapsed_s": 1.0,
                    "validation_passed": False,
                },
            ]
        )

        row = summarize_run(self.run_dir)["cases"][0]

        self.assertEqual(row["quality_items"], 4)
        self.assertEqual(row["quality_scored_items"], 4)
        self.assertEqual(row["quality_correct"], 3)
        self.assertEqual(row["quality_accuracy"], 0.75)
        self.assertEqual(row["quality_total_prompt_tokens"], 100)
        self.assertEqual(row["quality_total_completion_tokens"], 8)
        self.assertAlmostEqual(row["quality_total_request_latency_s"], 1.0)
        self.assertEqual(row["quality_accuracy_by_category"]["logic"], 1.0)
        self.assertEqual(
            row["quality_accuracy_by_category"]["instruction_following"], 0.0
        )

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
