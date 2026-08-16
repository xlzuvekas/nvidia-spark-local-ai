from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.journal import content_hash
from bench.report import summarize_run
from bench.runner import execute_plan
from bench.vllm_metrics import (
    aggregate_vllm_spec_decode_metrics,
    parse_vllm_spec_decode_metrics,
    snapshot_vllm_spec_decode_metrics,
)


PROMETHEUS_EXPOSITION = """
# HELP vllm:spec_decode_num_drafts_total Number of speculative drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="target"} 4.0
vllm:spec_decode_num_drafts_total{engine="1",model_name="target"} 2.0
vllm:spec_decode_num_drafts_created{engine="0",model_name="target"} 1.0
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="target"} 60.0
vllm:spec_decode_num_draft_tokens_total{engine="1",model_name="target"} 30.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="target"} 30.0
vllm:spec_decode_num_accepted_tokens_total{engine="1",model_name="target"} 15.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="target",position="0"} 4.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",model_name="target",position="0"} 2.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="target",position="1"} 3.0
"""


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class VLLMMetricsTests(unittest.TestCase):
    def test_parser_uses_exact_vllm_counters_and_derives_acceptance(self) -> None:
        metrics = parse_vllm_spec_decode_metrics(PROMETHEUS_EXPOSITION)

        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertEqual(metrics["num_drafts"], 6)
        self.assertEqual(metrics["num_draft_tokens"], 90)
        self.assertEqual(metrics["num_accepted_tokens"], 45)
        self.assertEqual(metrics["accepted_tokens_per_position"], {"0": 6, "1": 3})
        self.assertAlmostEqual(metrics["draft_acceptance_rate"], 0.5)
        self.assertAlmostEqual(metrics["mean_accepted_length"], 8.5)

    def test_absent_or_unavailable_spec_metrics_are_optional(self) -> None:
        self.assertIsNone(
            parse_vllm_spec_decode_metrics(
                'vllm:num_requests_running{engine="0",model_name="target"} 1.0\n'
            )
        )
        with patch(
            "bench.vllm_metrics.urllib.request.urlopen",
            side_effect=ConnectionResetError("server stopped"),
        ):
            self.assertIsNone(
                snapshot_vllm_spec_decode_metrics("http://127.0.0.1:8000/v1")
            )

    def test_snapshot_uses_root_metrics_endpoint_without_process_probes(self) -> None:
        with patch(
            "bench.vllm_metrics.urllib.request.urlopen",
            return_value=_Response(PROMETHEUS_EXPOSITION.encode()),
        ) as urlopen:
            metrics = snapshot_vllm_spec_decode_metrics(
                "http://127.0.0.1:8000/v1/", timeout_s=0.25
            )

        self.assertIsNotNone(metrics)
        urlopen.assert_called_once_with(
            "http://127.0.0.1:8000/metrics", timeout=0.25
        )

    def test_report_combines_snapshots_from_resumed_server_lifetimes(self) -> None:
        snapshots = [
            {
                "num_drafts": 4,
                "num_draft_tokens": 60,
                "num_accepted_tokens": 30,
                "accepted_tokens_per_position": {"0": 4, "1": 3},
            },
            {
                "num_drafts": 2,
                "num_draft_tokens": 30,
                "num_accepted_tokens": 15,
                "accepted_tokens_per_position": {"0": 2, "1": 1},
            },
        ]
        combined = aggregate_vllm_spec_decode_metrics(snapshots)
        self.assertIsNotNone(combined)
        assert combined is not None
        self.assertEqual(combined["snapshot_count"], 2)
        self.assertEqual(combined["num_drafts"], 6)
        self.assertEqual(combined["accepted_tokens_per_position"], {"0": 6, "1": 4})
        self.assertAlmostEqual(combined["draft_acceptance_rate"], 0.5)
        self.assertAlmostEqual(combined["mean_accepted_length"], 8.5)

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            events = [
                {
                    "event": "vllm_spec_decode_metrics_snapshot",
                    "metrics": snapshot,
                }
                for snapshot in snapshots
            ]
            (run_dir / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events)
            )
            summary = summarize_run(run_dir)

        self.assertEqual(summary["speculative_decoding"], combined)


class VLLMMetricsLifecycleTests(unittest.TestCase):
    def _write_plan(self, root: Path) -> Path:
        model = {
            "id": "vllm-target",
            "backend": "vllm",
            "source": "example/model",
            "served_name": "target",
            "tasks": ["chat"],
            "max_context": 8192,
            "endpoint": "http://127.0.0.1:8000/v1",
            "image": "example/image",
            "args": [],
            "startup_timeout_s": 1,
            "cache_dir": "project",
        }
        case = {
            "id": "decode",
            "kind": "decode",
            "requires": ["chat"],
            "warmups": 0,
            "repetitions": 1,
            "max_output_tokens": 8,
            "temperature": 0.0,
            "concurrency": 1,
            "prompt_repetitions": 0,
        }
        suite = {
            "id": "quick",
            "description": "",
            "schema_version": 1,
            "cases": [case],
        }
        fingerprint = content_hash({"model": model, "suite": suite})
        frozen_case = {
            **case,
            "case_id": f"decode--{content_hash({'model': model, 'case': case}, 12)}",
        }
        run_dir = root / "run"
        run_dir.mkdir()
        (run_dir / "plan.json").write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "model": model,
                    "suite": {**suite, "cases": [frozen_case]},
                    "resolved": {},
                }
            )
        )
        return run_dir

    def test_vllm_snapshot_is_journaled_before_server_teardown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_plan(root)
            plan = json.loads((run_dir / "plan.json").read_text())
            server = SimpleNamespace(
                backend="vllm",
                base_url="http://127.0.0.1:8000/v1",
                startup_s=0.1,
                container_id="owned-container",
                run_identity=f"{plan['fingerprint']}-{run_dir.name}",
                ollama_model=None,
                unload_ollama=False,
                stop=Mock(),
            )
            telemetry = Mock()
            first_request = Mock()
            first_request.to_dict.return_value = {}
            parsed = parse_vllm_spec_decode_metrics(PROMETHEUS_EXPOSITION)
            assert parsed is not None

            def snapshot(base_url: str) -> dict[str, object]:
                self.assertEqual(base_url, server.base_url)
                server.stop.assert_not_called()
                return parsed

            def complete_case(**kwargs: object) -> None:
                case = kwargs["case"]
                journal = kwargs["journal"]
                journal.append(
                    {
                        "event": "request_complete",
                        "case_id": case.case_id,
                        "attempt_id": "attempt",
                        "kind": "decode",
                        "result": {
                            "ttft_s": 0.1,
                            "elapsed_s": 1.0,
                            "decode_tps": 8.0,
                            "prompt_tokens": 10,
                            "completion_tokens": 8,
                        },
                    }
                )
                journal.append(
                    {
                        "event": "case_complete",
                        "case_id": case.case_id,
                        "attempt_id": "attempt",
                        "kind": "decode",
                        "elapsed_s": 1.0,
                        "validation_passed": True,
                    }
                )

            with (
                patch("bench.runner._preflight"),
                patch("bench.runner.TelemetrySampler", return_value=telemetry),
                patch("bench.runner.start_server", return_value=server),
                patch("bench.runner._prime_model", return_value=first_request),
                patch("bench.runner._execute_case", side_effect=complete_case),
                patch("bench.runner.save_server_logs") as save_logs,
                patch(
                    "bench.runner.snapshot_vllm_spec_decode_metrics",
                    side_effect=snapshot,
                ) as scrape,
            ):
                summary = execute_plan(run_dir, workspace=root)

            scrape.assert_called_once_with(server.base_url)
            save_logs.assert_called_once()
            server.stop.assert_called_once_with(keep_server=False)
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            event_names = [event["event"] for event in events]
            self.assertLess(
                event_names.index("vllm_spec_decode_metrics_snapshot"),
                event_names.index("server_stopped"),
            )
            self.assertEqual(summary["speculative_decoding"]["num_drafts"], 6)


if __name__ == "__main__":
    unittest.main()
