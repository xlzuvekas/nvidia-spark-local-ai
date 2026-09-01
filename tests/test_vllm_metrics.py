from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.client import RequestResult
from bench.journal import Journal, content_hash
from bench.report import summarize_run
from bench.runner import _execute_case, execute_plan
from bench.vllm_metrics import (
    VLLMSpecDecodeMetricsDeltaError,
    VLLM_SPEC_DECODE_CASE_AGGREGATE_SCOPE,
    VLLM_SPEC_DECODE_CUMULATIVE_SCOPE,
    VLLM_SPEC_DECODE_REQUEST_DELTA_SCOPE,
    VLLM_SPEC_DECODE_SOURCE,
    aggregate_vllm_spec_decode_metrics,
    aggregate_vllm_spec_decode_metric_deltas,
    delta_vllm_spec_decode_metrics,
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


def _cumulative_snapshot(
    *,
    drafts: int,
    draft_tokens: int,
    accepted_tokens: int,
    positions: list[int],
) -> dict[str, object]:
    return {
        "source": VLLM_SPEC_DECODE_SOURCE,
        "scope": VLLM_SPEC_DECODE_CUMULATIVE_SCOPE,
        "num_drafts": drafts,
        "num_draft_tokens": draft_tokens,
        "num_accepted_tokens": accepted_tokens,
        "accepted_tokens_per_position": {
            str(position): value for position, value in enumerate(positions)
        },
        "draft_acceptance_rate": (
            accepted_tokens / draft_tokens if draft_tokens else None
        ),
        "mean_accepted_length": (1.0 + accepted_tokens / drafts if drafts else None),
    }


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
        urlopen.assert_called_once_with("http://127.0.0.1:8000/metrics", timeout=0.25)

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


class VLLMMetricsDeltaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.before = _cumulative_snapshot(
            drafts=10,
            draft_tokens=80,
            accepted_tokens=35,
            positions=[10, 8, 7, 5, 3, 2, 0, 0],
        )
        self.after = _cumulative_snapshot(
            drafts=12,
            draft_tokens=96,
            accepted_tokens=41,
            positions=[12, 10, 8, 6, 3, 2, 0, 0],
        )

    def test_delta_subtracts_core_and_position_counters_and_derives_rates(
        self,
    ) -> None:
        delta = delta_vllm_spec_decode_metrics(self.before, self.after)

        self.assertEqual(delta["source"], VLLM_SPEC_DECODE_SOURCE)
        self.assertEqual(delta["scope"], VLLM_SPEC_DECODE_REQUEST_DELTA_SCOPE)
        self.assertEqual(delta["num_drafts"], 2)
        self.assertEqual(delta["num_draft_tokens"], 16)
        self.assertEqual(delta["num_accepted_tokens"], 6)
        self.assertEqual(
            delta["accepted_tokens_per_position"],
            {"0": 2, "1": 2, "2": 1, "3": 1, "4": 0, "5": 0, "6": 0, "7": 0},
        )
        self.assertAlmostEqual(delta["draft_acceptance_rate"], 6 / 16)
        self.assertAlmostEqual(delta["mean_accepted_length"], 4.0)

    def test_delta_preserves_zero_activity_and_new_position_series(self) -> None:
        unchanged = delta_vllm_spec_decode_metrics(self.before, self.before)
        self.assertEqual(unchanged["num_drafts"], 0)
        self.assertEqual(unchanged["num_draft_tokens"], 0)
        self.assertEqual(unchanged["num_accepted_tokens"], 0)
        self.assertEqual(
            unchanged["accepted_tokens_per_position"],
            {str(position): 0 for position in range(8)},
        )
        self.assertIsNone(unchanged["draft_acceptance_rate"])
        self.assertIsNone(unchanged["mean_accepted_length"])

        before = _cumulative_snapshot(
            drafts=5,
            draft_tokens=10,
            accepted_tokens=4,
            positions=[3, 1],
        )
        after = _cumulative_snapshot(
            drafts=6,
            draft_tokens=13,
            accepted_tokens=7,
            positions=[4, 2, 1],
        )
        delta = delta_vllm_spec_decode_metrics(before, after)
        self.assertEqual(
            delta["accepted_tokens_per_position"], {"0": 1, "1": 1, "2": 1}
        )
        self.assertEqual(delta["num_accepted_tokens"], 3)

    def test_request_deltas_aggregate_into_one_exact_case_window(self) -> None:
        first = delta_vllm_spec_decode_metrics(self.before, self.after)
        second = deepcopy(first)

        aggregate = aggregate_vllm_spec_decode_metric_deltas([first, second])

        self.assertEqual(
            aggregate["scope"], VLLM_SPEC_DECODE_CASE_AGGREGATE_SCOPE
        )
        self.assertEqual(aggregate["source"], VLLM_SPEC_DECODE_SOURCE)
        self.assertEqual(aggregate["request_count"], 2)
        self.assertEqual(aggregate["num_drafts"], 4)
        self.assertEqual(aggregate["num_draft_tokens"], 32)
        self.assertEqual(aggregate["num_accepted_tokens"], 12)
        self.assertEqual(
            aggregate["accepted_tokens_per_position"],
            {"0": 4, "1": 4, "2": 2, "3": 2, "4": 0, "5": 0, "6": 0, "7": 0},
        )
        self.assertAlmostEqual(aggregate["draft_acceptance_rate"], 12 / 32)
        self.assertAlmostEqual(aggregate["mean_accepted_length"], 4.0)

        with self.assertRaisesRegex(
            VLLMSpecDecodeMetricsDeltaError, "at least one request-scoped"
        ):
            aggregate_vllm_spec_decode_metric_deltas([])

        malformed = deepcopy(first)
        malformed["num_accepted_tokens"] = 7
        malformed["draft_acceptance_rate"] = 7 / 16
        with self.assertRaises(VLLMSpecDecodeMetricsDeltaError):
            aggregate_vllm_spec_decode_metric_deltas([malformed])

    def test_delta_rejects_core_and_position_counter_resets(self) -> None:
        reset_core = _cumulative_snapshot(
            drafts=9,
            draft_tokens=72,
            accepted_tokens=30,
            positions=[9, 7, 6, 4, 2, 2, 0, 0],
        )
        with self.assertRaisesRegex(
            VLLMSpecDecodeMetricsDeltaError, "core counter reset"
        ):
            delta_vllm_spec_decode_metrics(self.before, reset_core)

        reset_position = _cumulative_snapshot(
            drafts=12,
            draft_tokens=96,
            accepted_tokens=38,
            positions=[12, 7, 7, 5, 3, 2, 1, 1],
        )
        with self.assertRaisesRegex(
            VLLMSpecDecodeMetricsDeltaError, "position counter reset"
        ):
            delta_vllm_spec_decode_metrics(self.before, reset_position)

        disappeared = deepcopy(self.after)
        positions = disappeared["accepted_tokens_per_position"]
        assert isinstance(positions, dict)
        positions.pop("7")
        with self.assertRaisesRegex(
            VLLMSpecDecodeMetricsDeltaError, "position counter disappeared"
        ):
            delta_vllm_spec_decode_metrics(self.before, disappeared)

    def test_delta_rejects_arithmetically_inconsistent_results(self) -> None:
        inconsistent = (
            _cumulative_snapshot(
                drafts=12,
                draft_tokens=96,
                accepted_tokens=41,
                positions=[12, 10, 8, 5, 3, 2, 0, 0],
            ),
            _cumulative_snapshot(
                drafts=12,
                draft_tokens=81,
                accepted_tokens=37,
                positions=[11, 9, 7, 5, 3, 2, 0, 0],
            ),
            _cumulative_snapshot(
                drafts=12,
                draft_tokens=96,
                accepted_tokens=38,
                positions=[11, 10, 7, 5, 3, 2, 0, 0],
            ),
            _cumulative_snapshot(
                drafts=10,
                draft_tokens=81,
                accepted_tokens=35,
                positions=[10, 8, 7, 5, 3, 2, 0, 0],
            ),
        )
        for after in inconsistent:
            with self.subTest(after=after):
                with self.assertRaisesRegex(
                    VLLMSpecDecodeMetricsDeltaError,
                    "request-scoped counter delta is inconsistent",
                ):
                    delta_vllm_spec_decode_metrics(self.before, after)

    def test_delta_rejects_malformed_or_misproven_snapshots(self) -> None:
        invalid_snapshots: list[dict[str, object]] = []

        missing_source = deepcopy(self.before)
        missing_source.pop("source")
        invalid_snapshots.append(missing_source)

        wrong_scope = deepcopy(self.before)
        wrong_scope["scope"] = "all_persisted_vllm_server_lifetimes"
        invalid_snapshots.append(wrong_scope)

        fractional = deepcopy(self.before)
        fractional["num_drafts"] = 10.5
        invalid_snapshots.append(fractional)

        nonfinite = deepcopy(self.before)
        nonfinite["num_draft_tokens"] = float("nan")
        invalid_snapshots.append(nonfinite)

        noncanonical_position = deepcopy(self.before)
        positions = noncanonical_position["accepted_tokens_per_position"]
        assert isinstance(positions, dict)
        positions["08"] = positions.pop("7")
        invalid_snapshots.append(noncanonical_position)

        invalid_position_value = deepcopy(self.before)
        positions = invalid_position_value["accepted_tokens_per_position"]
        assert isinstance(positions, dict)
        positions["0"] = True
        invalid_snapshots.append(invalid_position_value)

        wrong_rate = deepcopy(self.before)
        wrong_rate["draft_acceptance_rate"] = 0.0
        invalid_snapshots.append(wrong_rate)

        for before in invalid_snapshots:
            with self.subTest(before=before):
                with self.assertRaises(VLLMSpecDecodeMetricsDeltaError):
                    delta_vllm_spec_decode_metrics(before, self.after)


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

    def test_fast_case_records_and_aggregates_one_delta_per_request(self) -> None:
        case = SimpleNamespace(
            id="densespark-fast-code-tests-d256",
            case_id="densespark-fast-code-tests-d256--fixture",
            kind="decode",
            requires=("chat",),
            warmups=0,
            repetitions=2,
            max_output_tokens=256,
            temperature=0.0,
            concurrency=1,
            prompt_repetitions=0,
            max_turns=1,
        )
        model = SimpleNamespace(
            id=(
                "qwen38-27b-int4-autoround-densespark-c1-native-262k-"
                "fast-mtp8-warmup-sync"
            ),
            served_name="densespark-qwen3.8-27b",
            request_body_json=(
                '{"chat_template_kwargs":{"enable_thinking":false}}'
            ),
        )
        server = SimpleNamespace(
            backend="vllm",
            base_url="http://127.0.0.1:8000/v1",
            authorization=None,
        )
        snapshots = [
            _cumulative_snapshot(
                drafts=10,
                draft_tokens=80,
                accepted_tokens=35,
                positions=[10, 8, 7, 5, 3, 2, 0, 0],
            ),
            _cumulative_snapshot(
                drafts=12,
                draft_tokens=96,
                accepted_tokens=41,
                positions=[12, 10, 8, 6, 3, 2, 0, 0],
            ),
            _cumulative_snapshot(
                drafts=12,
                draft_tokens=96,
                accepted_tokens=41,
                positions=[12, 10, 8, 6, 3, 2, 0, 0],
            ),
            _cumulative_snapshot(
                drafts=14,
                draft_tokens=112,
                accepted_tokens=47,
                positions=[14, 12, 9, 7, 3, 2, 0, 0],
            ),
        ]

        def request_batch(**kwargs: object) -> tuple[list[RequestResult], float]:
            requests = kwargs["requests"]
            assert isinstance(requests, list)
            request_id = str(requests[0]["request_id"])
            return (
                [
                    RequestResult(
                        request_id=request_id,
                        started_at_ns=1,
                        prompt_tokens=40,
                        completion_tokens=256,
                        reasoning_tokens=0,
                        ttft_s=0.1,
                        elapsed_s=4.0,
                        decode_s=3.9,
                        decode_tps=65.4,
                        output_tps=64.0,
                        emission_events=256,
                        finish_reason="length",
                        response_model=model.served_name,
                        content="def merge_sort(values): pass\ndef test_merge_000(): pass",
                        reasoning="",
                        tool_calls=[],
                    )
                ],
                4.0,
            )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            journal = Journal(run_dir / "events.jsonl")
            telemetry = SimpleNamespace(set_phase=lambda _phase: None)
            with (
                patch("bench.runner._run_warmups"),
                patch(
                    "bench.runner.concurrent_chat_requests",
                    side_effect=request_batch,
                ),
                patch(
                    "bench.runner.snapshot_vllm_spec_decode_metrics",
                    side_effect=snapshots,
                ) as scrape,
            ):
                _execute_case(
                    server=server,
                    model=model,
                    case=case,
                    journal=journal,
                    telemetry=telemetry,
                )

            scrape.assert_has_calls(
                [
                    unittest.mock.call(server.base_url),
                    unittest.mock.call(server.base_url),
                    unittest.mock.call(server.base_url),
                    unittest.mock.call(server.base_url),
                ]
            )
            delta_events = [
                event
                for event in journal.events()
                if event.get("event") == "vllm_spec_decode_metrics_delta"
            ]
            self.assertEqual([0, 1], [event["repetition"] for event in delta_events])
            self.assertTrue(
                all(
                    event["metrics"]["scope"]
                    == VLLM_SPEC_DECODE_REQUEST_DELTA_SCOPE
                    for event in delta_events
                )
            )
            (run_dir / "plan.json").write_text(
                json.dumps(
                    {
                        "model": {},
                        "suite": {
                            "cases": [
                                {
                                    "case_id": case.case_id,
                                    "repetitions": case.repetitions,
                                }
                            ]
                        },
                    }
                )
            )
            summary = summarize_run(run_dir)

        self.assertEqual(1, len(summary["cases"]))
        aggregate = summary["cases"][0]["speculative_decoding"]
        self.assertEqual(2, aggregate["request_count"])
        self.assertEqual(4, aggregate["num_drafts"])
        self.assertEqual(12, aggregate["num_accepted_tokens"])
        self.assertEqual(
            VLLM_SPEC_DECODE_CASE_AGGREGATE_SCOPE,
            aggregate["scope"],
        )

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
