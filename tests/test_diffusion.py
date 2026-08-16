from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from bench.report import summarize_run
from bench.runner import _request_result_payload, _validate_capability


class _Result:
    def __init__(
        self,
        *,
        completion_tokens: int = 64,
        finish_reason: str = "length",
        output_tps: float = 8.0,
    ) -> None:
        self.completion_tokens = completion_tokens
        self.finish_reason = finish_reason
        self.output_tps = output_tps

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_tokens": 12,
            "completion_tokens": self.completion_tokens,
            "ttft_s": 1.5,
            "elapsed_s": 8.0,
            "decode_s": 6.5,
            "decode_tps": 9.7,
            "output_tps": self.output_tps,
            "decode_metric_source": "client_estimate",
            "emission_events": 1,
            "finish_reason": self.finish_reason,
        }


class DiffusionMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SimpleNamespace(architecture="diffusion-lm")

    def test_serialization_removes_autoregressive_metric_labels(self) -> None:
        payload = _request_result_payload(self.model, _Result())

        self.assertIsNone(payload["decode_s"])
        self.assertIsNone(payload["decode_tps"])
        self.assertIsNone(payload["decode_metric_source"])
        self.assertIsNone(payload["ttft_s"])
        self.assertIsNone(payload["output_tps"])
        self.assertEqual(payload["time_to_first_emission_s"], 1.5)
        self.assertEqual(payload["block_generation_output_tps"], 8.0)
        self.assertEqual(
            payload["block_generation_metric_source"],
            "client_completion_tokens_per_end_to_end_request_elapsed",
        )

    def test_validation_identifies_diffusion_semantics_and_rejects_bad_rate(self) -> None:
        case = SimpleNamespace(kind="decode", max_output_tokens=64)

        valid = _validate_capability(case, _Result(), model=self.model)
        invalid = _validate_capability(
            case, _Result(output_tps=math.nan), model=self.model
        )

        self.assertTrue(valid["passed"])
        self.assertEqual(valid["generation_mode"], "diffusion_block_generation")
        self.assertEqual(
            valid["throughput_metric"],
            "completion_tokens_per_end_to_end_request_elapsed",
        )
        self.assertFalse(invalid["passed"])
        self.assertIn("block-generation", invalid["reason"])

    def test_report_uses_only_end_to_end_block_generation_rates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "plan.json").write_text(
                json.dumps(
                    {
                        "model": {"id": "diffusion", "architecture": "diffusion-lm"},
                        "suite": {"id": "quick"},
                    }
                )
            )
            result = _request_result_payload(self.model, _Result())
            events = [
                {
                    "event": "request_complete",
                    "case_id": "decode",
                    "attempt_id": "one",
                    "kind": "decode",
                    "result": result,
                },
                {
                    "event": "case_complete",
                    "case_id": "decode",
                    "attempt_id": "one",
                    "kind": "decode",
                    "elapsed_s": 8.0,
                    "validation_passed": True,
                },
            ]
            (run_dir / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events)
            )

            row = summarize_run(run_dir)["cases"][0]

        self.assertIsNone(row["median_decode_tps"])
        self.assertIsNone(row["decode_metric_source"])
        self.assertIsNone(row["median_estimated_decode_tps"])
        self.assertIsNone(row["aggregate_output_tps"])
        self.assertIsNone(row["median_ttft_s"])
        self.assertEqual(row["median_time_to_first_emission_s"], 1.5)
        self.assertEqual(row["median_block_generation_output_tps"], 8.0)
        self.assertEqual(row["aggregate_block_generation_output_tps"], 8.0)
        self.assertEqual(
            row["block_generation_metric_source"],
            "client_completion_tokens_per_end_to_end_request_elapsed",
        )

    def test_diffusion_prefill_never_uses_first_emission_as_prefill_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "plan.json").write_text(
                json.dumps(
                    {
                        "model": {"id": "diffusion", "architecture": "diffusion-lm"},
                        "suite": {"id": "quick"},
                    }
                )
            )
            events = [
                {
                    "event": "request_complete",
                    "case_id": "prefill",
                    "attempt_id": "one",
                    "kind": "prefill",
                    "result": {
                        "prompt_tokens": 1024,
                        "completion_tokens": 1,
                        "ttft_s": 0.01,
                        "elapsed_s": 1.0,
                        "decode_tps": 99999.0,
                    },
                },
                {
                    "event": "case_complete",
                    "case_id": "prefill",
                    "attempt_id": "one",
                    "kind": "prefill",
                    "elapsed_s": 1.0,
                },
            ]
            (run_dir / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events)
            )

            row = summarize_run(run_dir)["cases"][0]

        self.assertIsNone(row["median_prefill_tps"])
        self.assertIsNone(row["median_approximate_prefill_tps"])
        self.assertEqual(
            row["prefill_metric_source"],
            "unavailable_for_diffusion_block_generation",
        )


if __name__ == "__main__":
    unittest.main()
