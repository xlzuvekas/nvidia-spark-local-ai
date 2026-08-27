from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import urllib.error
from unittest.mock import Mock, patch

from bench.evidence import _lifecycle, _project_summary
from bench.journal import content_hash
from bench.runner import execute_plan
from bench.sglang_metrics import (
    SGLangSpeculativeAuditError,
    aggregate_sglang_speculative_audits,
    parse_sglang_speculative_response,
    request_sglang_speculative_audit,
    sglang_nextn_depth,
)


RAW_TEXT = "GENERATED_TEXT_MUST_NOT_SURVIVE"
RAW_TOKEN = "private-ephemeral-token"


def _native_response(
    *,
    accepted: int = 9,
    proposed: int = 15,
    verify_count: int = 5,
    histogram: list[int] | None = None,
) -> dict[str, object]:
    histogram = [0, 2, 2, 1] if histogram is None else histogram
    return {
        "text": RAW_TEXT,
        "output_ids": [9001, 9002],
        "meta_info": {
            "id": "private-request-id",
            "finish_reason": {"type": "length", "length": 256},
            "spec_num_correct_drafts": accepted,
            "spec_num_proposed_drafts": proposed,
            "spec_verify_ct": verify_count,
            "spec_correct_drafts_histogram": histogram,
            "spec_accept_rate": accepted / proposed,
            "spec_accept_length": 2.8,
            "spec_accepted_drafts": accepted,
            "spec_proposed_drafts": proposed,
            "spec_accept_histogram": histogram,
        },
    }


class _Response(io.BytesIO):
    status = 200

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class SGLangSpeculativeParserTests(unittest.TestCase):
    def test_parser_projects_only_exact_cumulative_scalars(self) -> None:
        snapshot = parse_sglang_speculative_response(
            _native_response(), expected_depth=3
        )

        self.assertEqual(snapshot["num_drafts"], 5)
        self.assertEqual(snapshot["num_draft_tokens"], 15)
        self.assertEqual(snapshot["num_accepted_tokens"], 9)
        self.assertEqual(
            snapshot["accepted_tokens_per_position"],
            {"0": 5, "1": 3, "2": 1},
        )
        self.assertAlmostEqual(snapshot["draft_acceptance_rate"], 0.6)
        self.assertAlmostEqual(snapshot["mean_accepted_length"], 2.8)
        serialized = json.dumps(snapshot, sort_keys=True)
        self.assertNotIn(RAW_TEXT, serialized)
        self.assertNotIn("output_ids", serialized)
        self.assertNotIn("private-request-id", serialized)
        self.assertNotIn("finish_reason", serialized)

    def test_parser_accepts_a_short_histogram_with_implicit_trailing_zeros(self) -> None:
        snapshot = parse_sglang_speculative_response(
            _native_response(
                accepted=2,
                proposed=15,
                verify_count=5,
                histogram=[3, 2],
            ),
            expected_depth=3,
        )
        self.assertEqual(
            snapshot["accepted_tokens_per_position"],
            {"0": 2, "1": 0, "2": 0},
        )

    def test_parser_tolerates_float32_rounding_in_reported_acceptance_rate(self) -> None:
        payload = _native_response()
        meta = payload["meta_info"]
        assert isinstance(meta, dict)
        meta["spec_accept_rate"] = 0.6000000238418579

        snapshot = parse_sglang_speculative_response(payload, expected_depth=3)

        self.assertEqual(snapshot["draft_acceptance_rate"], 0.6)

    def test_parser_rejects_malformed_or_inconsistent_counters(self) -> None:
        mutations = {
            "boolean count": lambda meta: meta.__setitem__(
                "spec_verify_ct", True
            ),
            "geometry": lambda meta: meta.__setitem__(
                "spec_num_proposed_drafts", 14
            ),
            "accepted exceeds proposed": lambda meta: meta.__setitem__(
                "spec_num_correct_drafts", 16
            ),
            "alias mismatch": lambda meta: meta.__setitem__(
                "spec_accepted_drafts", 8
            ),
            "histogram steps": lambda meta: meta.__setitem__(
                "spec_correct_drafts_histogram", [0, 2, 1, 1]
            ),
            "histogram accepted": lambda meta: meta.__setitem__(
                "spec_correct_drafts_histogram", [0, 1, 3, 1]
            ),
            "histogram too deep": lambda meta: meta.__setitem__(
                "spec_correct_drafts_histogram", [0, 2, 2, 1, 0]
            ),
            "rate mismatch": lambda meta: meta.__setitem__(
                "spec_accept_rate", 0.4
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                payload = _native_response()
                meta = payload["meta_info"]
                assert isinstance(meta, dict)
                mutate(meta)
                with self.assertRaises(SGLangSpeculativeAuditError):
                    parse_sglang_speculative_response(payload, expected_depth=3)

    def test_missing_activity_fails_closed_without_echoing_response(self) -> None:
        payload = {
            "text": RAW_TEXT,
            "output_ids": [1],
            "meta_info": {"completion_tokens": 1},
        }
        with self.assertRaises(SGLangSpeculativeAuditError) as raised:
            parse_sglang_speculative_response(payload, expected_depth=3)
        self.assertNotIn(RAW_TEXT, str(raised.exception))

    def test_nextn_depth_uses_pinned_proposals_per_verify_geometry(self) -> None:
        self.assertEqual(
            sglang_nextn_depth(
                (
                    "--speculative-algorithm",
                    "NEXTN",
                    "--speculative-num-steps",
                    "3",
                    "--speculative-num-draft-tokens",
                    "4",
                )
            ),
            3,
        )
        self.assertEqual(
            sglang_nextn_depth(
                (
                    "--speculative-algorithm=nextn",
                    "--speculative-num-draft-tokens=2",
                )
            ),
            1,
        )
        self.assertIsNone(sglang_nextn_depth(()))
        with self.assertRaises(SGLangSpeculativeAuditError):
            sglang_nextn_depth(
                ("--speculative-algorithm", "NEXTN", "--speculative-num-draft-tokens")
            )

    def test_aggregate_combines_disjoint_native_audit_requests(self) -> None:
        first = parse_sglang_speculative_response(
            _native_response(), expected_depth=3
        )
        second = parse_sglang_speculative_response(
            _native_response(
                accepted=2,
                proposed=6,
                verify_count=2,
                histogram=[1, 0, 1],
            ),
            expected_depth=3,
        )
        combined = aggregate_sglang_speculative_audits([first, second])
        assert combined is not None
        self.assertEqual(combined["snapshot_count"], 2)
        self.assertEqual(combined["num_drafts"], 7)
        self.assertEqual(combined["num_draft_tokens"], 21)
        self.assertEqual(combined["num_accepted_tokens"], 11)
        self.assertEqual(
            combined["accepted_tokens_per_position"],
            {"0": 6, "1": 4, "2": 1},
        )
        self.assertTrue(combined["proposal_depth"]["passed"])


class SGLangSpeculativeTransportTests(unittest.TestCase):
    def test_request_uses_authenticated_tokenize_then_native_generate(self) -> None:
        responses = [
            {
                "tokens": [101, 102, 103, 104],
                "count": 4,
                "max_model_len": 262144,
            },
            _native_response(),
        ]
        requests: list[tuple[object, float]] = []

        def open_request(request: object, *, timeout: float) -> _Response:
            requests.append((request, timeout))
            payload = responses[len(requests) - 1]
            return _Response(json.dumps(payload).encode("utf-8"))

        with patch("bench.sglang_metrics._urlopen", side_effect=open_request):
            snapshot = request_sglang_speculative_audit(
                base_url="http://127.0.0.1:30000/v1",
                model="qwen38-flash-next",
                authorization=f"Bearer {RAW_TOKEN}",
                expected_depth=3,
                chat_template_kwargs={"enable_thinking": False},
                max_new_tokens=256,
                timeout_s=12.5,
            )

        self.assertEqual(snapshot["num_accepted_tokens"], 9)
        self.assertEqual(len(requests), 2)
        tokenize_request, tokenize_timeout = requests[0]
        generate_request, generate_timeout = requests[1]
        self.assertEqual(tokenize_request.full_url, "http://127.0.0.1:30000/v1/tokenize")
        self.assertEqual(generate_request.full_url, "http://127.0.0.1:30000/generate")
        self.assertEqual(tokenize_request.get_method(), "POST")
        self.assertEqual(generate_request.get_method(), "POST")
        self.assertEqual(tokenize_timeout, 12.5)
        self.assertEqual(generate_timeout, 12.5)
        for request, _timeout in requests:
            self.assertEqual(request.get_header("Authorization"), f"Bearer {RAW_TOKEN}")
            self.assertEqual(request.get_header("Content-type"), "application/json")

        tokenize_body = json.loads(tokenize_request.data)
        self.assertEqual(
            tokenize_body,
            {
                "model": "qwen38-flash-next",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Write an unbroken numbered list of distinct two-word phrases. "
                            "Continue until the output limit; do not conclude or summarize."
                        ),
                    }
                ],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        generate_body = json.loads(generate_request.data)
        self.assertEqual(
            generate_body,
            {
                "input_ids": [101, 102, 103, 104],
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 256,
                },
                "stream": False,
                "no_logs": True,
                "log_metrics": False,
            },
        )
        self.assertNotIn("return_meta_info", generate_body)

    def test_transport_failure_never_echoes_body_or_authorization(self) -> None:
        error = urllib.error.HTTPError(
            "http://127.0.0.1:30000/v1/tokenize",
            401,
            "unauthorized",
            {},
            io.BytesIO(f"{RAW_TEXT} Bearer {RAW_TOKEN}".encode()),
        )
        with (
            patch("bench.sglang_metrics._urlopen", side_effect=error),
            self.assertRaises(SGLangSpeculativeAuditError) as raised,
        ):
            request_sglang_speculative_audit(
                base_url="http://127.0.0.1:30000/v1",
                model="qwen38-flash-next",
                authorization=f"Bearer {RAW_TOKEN}",
                expected_depth=3,
            )
        message = str(raised.exception)
        self.assertIn("HTTP status 401", message)
        self.assertNotIn(RAW_TEXT, message)
        self.assertNotIn(RAW_TOKEN, message)

    def test_endpoint_and_authorization_fail_closed(self) -> None:
        for endpoint in (
            "https://127.0.0.1:30000/v1",
            "http://localhost:30000/v1",
            "http://127.0.0.1:30000/v1?redirect=external",
            "http://user@127.0.0.1:30000/v1",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(SGLangSpeculativeAuditError):
                    request_sglang_speculative_audit(
                        base_url=endpoint,
                        model="qwen38-flash-next",
                        authorization=f"Bearer {RAW_TOKEN}",
                        expected_depth=3,
                    )
        with self.assertRaises(SGLangSpeculativeAuditError) as raised:
            request_sglang_speculative_audit(
                base_url="http://127.0.0.1:30000/v1",
                model="qwen38-flash-next",
                authorization="Bearer ",
                expected_depth=3,
            )
        self.assertNotIn(RAW_TOKEN, str(raised.exception))


class SGLangSpeculativeEvidenceTests(unittest.TestCase):
    def test_generic_evidence_projects_scalar_aggregate_and_knows_event(self) -> None:
        snapshot = parse_sglang_speculative_response(
            _native_response(), expected_depth=3
        )
        aggregate = aggregate_sglang_speculative_audits([snapshot])
        assert aggregate is not None

        projected = _project_summary({"speculative_decoding": aggregate})
        lifecycle = _lifecycle(
            [{"event": "sglang_spec_decode_metrics_snapshot", "metrics": snapshot}]
        )

        self.assertEqual(projected["speculative_decoding"]["source"], aggregate["source"])
        self.assertEqual(
            projected["speculative_decoding"]["num_accepted_tokens"], 9
        )
        self.assertEqual(
            lifecycle["event_counts"], {"sglang_spec_decode_metrics_snapshot": 1}
        )
        serialized = json.dumps(projected, sort_keys=True)
        self.assertNotIn(RAW_TEXT, serialized)
        self.assertNotIn("output_ids", serialized)


class SGLangSpeculativeLifecycleTests(unittest.TestCase):
    def _write_plan(self, root: Path) -> Path:
        model = {
            "id": "sglang-nextn-target",
            "backend": "sglang",
            "source": "example/model",
            "served_name": "qwen38-flash-next",
            "tasks": ["chat"],
            "max_context": 8192,
            "native_context": 8192,
            "endpoint": "http://127.0.0.1:30000/v1",
            "image": "example/image",
            "args": [
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-num-steps",
                "3",
                "--speculative-num-draft-tokens",
                "4",
            ],
            "request_body_json": '{"chat_template_kwargs":{"enable_thinking":false}}',
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

    def test_owned_nextn_audit_is_journaled_before_teardown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_plan(root)
            plan = json.loads((run_dir / "plan.json").read_text())
            server = SimpleNamespace(
                backend="sglang",
                base_url="http://127.0.0.1:30000/v1",
                startup_s=0.1,
                container_id="owned-container",
                run_identity=f"{plan['fingerprint']}-{run_dir.name}",
                authorization=f"Bearer {RAW_TOKEN}",
                ollama_model=None,
                unload_ollama=False,
                stop=Mock(),
            )
            telemetry = Mock()
            first_request = Mock()
            first_request.to_dict.return_value = {}
            snapshot = parse_sglang_speculative_response(
                _native_response(), expected_depth=3
            )

            def complete_case(**kwargs: object) -> None:
                case = kwargs["case"]
                journal = kwargs["journal"]
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

            def audit(**kwargs: object) -> dict[str, object]:
                self.assertEqual(kwargs["base_url"], server.base_url)
                self.assertEqual(kwargs["authorization"], f"Bearer {RAW_TOKEN}")
                self.assertEqual(kwargs["expected_depth"], 3)
                self.assertEqual(
                    kwargs["chat_template_kwargs"], {"enable_thinking": False}
                )
                server.stop.assert_not_called()
                return snapshot

            with (
                patch("bench.runner._preflight"),
                patch("bench.runner.TelemetrySampler", return_value=telemetry),
                patch("bench.runner.start_server", return_value=server),
                patch("bench.runner.capture_server_provenance", return_value={}),
                patch("bench.runner._prime_model", return_value=first_request),
                patch("bench.runner._execute_case", side_effect=complete_case),
                patch("bench.runner.save_server_logs"),
                patch(
                    "bench.runner.request_sglang_speculative_audit",
                    side_effect=audit,
                ) as native_audit,
            ):
                summary = execute_plan(run_dir, workspace=root)

            native_audit.assert_called_once()
            server.stop.assert_called_once_with(keep_server=False)
            phases = [call.args[0] for call in telemetry.set_phase.call_args_list]
            self.assertIn("sglang_speculative_acceptance_audit", phases)
            self.assertLess(
                phases.index("sglang_speculative_acceptance_audit"),
                phases.index("server_shutdown"),
            )
            serialized = (run_dir / "events.jsonl").read_text()
            events = [json.loads(line) for line in serialized.splitlines()]
            names = [event["event"] for event in events]
            self.assertLess(
                names.index("sglang_spec_decode_metrics_snapshot"),
                names.index("server_stopped"),
            )
            event = next(
                item
                for item in events
                if item["event"] == "sglang_spec_decode_metrics_snapshot"
            )
            self.assertEqual(event["metrics"], snapshot)
            self.assertEqual(
                summary["speculative_decoding"]["source"],
                "sglang_native_generate_per_request_counters",
            )
            self.assertEqual(
                summary["speculative_decoding"]["num_accepted_tokens"], 9
            )
            self.assertEqual(summary["speculative_decoding"]["snapshot_count"], 1)
            self.assertNotIn(RAW_TEXT, serialized)
            self.assertNotIn(RAW_TOKEN, serialized)


if __name__ == "__main__":
    unittest.main()
