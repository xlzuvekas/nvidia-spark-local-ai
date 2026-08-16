from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.client import RerankResult
from bench.journal import Journal
from bench.runner import (
    _execute_case,
    _multimodal_rerank_inputs,
    _rerank_request_arguments,
    _run_warmups,
    _validate_capability,
)


def _result(request_id: str, *, relevant_top: bool = True) -> RerankResult:
    scores = [0.1, 0.9] if relevant_top else [0.9, 0.1]
    ranking = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return RerankResult(
        request_id=request_id,
        started_at_ns=1,
        prompt_tokens=40,
        completion_tokens=0,
        ttft_s=0.02,
        elapsed_s=0.02,
        decode_s=0.0,
        decode_tps=0.0,
        output_tps=0.0,
        emission_events=1,
        finish_reason="stop",
        response_model="qwen3-vl-reranker",
        candidate_count=2,
        scores=scores,
        ranking=ranking,
        top_index=ranking[0],
        pairs_per_s=100.0,
        finite=True,
    )


def _case(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": "red-over-blue-image",
        "case_id": "red-over-blue-image--test",
        "kind": "capability",
        "requires": ["rerank", "vision"],
        "warmups": 1,
        "repetitions": 2,
        "max_output_tokens": 1,
        "temperature": 0.0,
        "concurrency": 2,
        "prompt_repetitions": 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class MultimodalRerankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = SimpleNamespace(
            backend="vllm", base_url="http://127.0.0.1:8000/v1"
        )
        self.model = SimpleNamespace(
            served_name="qwen3-vl-reranker",
            tasks=["rerank", "vision"],
            max_context=8192,
        )

    def test_inputs_are_local_deterministic_image_documents(self) -> None:
        query, documents, relevant_index = _multimodal_rerank_inputs(_case())
        second = _multimodal_rerank_inputs(_case())

        self.assertEqual((query, documents, relevant_index), second)
        self.assertIn("red square", query)
        self.assertEqual(relevant_index, 1)
        self.assertEqual(len(documents), 2)
        urls = [
            document["content"][0]["image_url"]["url"]
            for document in documents
        ]
        self.assertTrue(all(url.startswith("data:image/png;base64,") for url in urls))
        self.assertNotEqual(urls[0], urls[1])

    def test_arguments_follow_vllm_score_multimodal_schema(self) -> None:
        arguments = _rerank_request_arguments(
            server=self.server,
            model=self.model,
            case=_case(),
            request_id="multimodal-rerank-request",
        )

        self.assertEqual(arguments["base_url"], self.server.base_url)
        self.assertEqual(arguments["query"], "A solid red square with no other colors or objects.")
        self.assertEqual(
            arguments["instruction"],
            "Retrieve images relevant to the user's query.",
        )
        self.assertEqual(len(arguments["candidates"]), 2)
        self.assertTrue(
            arguments["candidates"][1]["content"][0]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )

    def test_validation_requires_red_candidate_to_be_uniquely_highest(self) -> None:
        passing = _validate_capability(_case(), _result("passing"))
        failing = _validate_capability(
            _case(), _result("failing", relevant_top=False)
        )

        self.assertEqual(passing, {"passed": True, "reason": None})
        self.assertFalse(failing["passed"])
        self.assertIn("index 1", failing["reason"])

    def test_warmup_and_measured_events_never_journal_image_payloads(self) -> None:
        case = _case()

        def measured(*, requests: list[dict[str, object]], concurrency: int):
            self.assertEqual(concurrency, 2)
            self.assertEqual(len(requests), 2)
            for request in requests:
                self.assertEqual(len(request["candidates"]), 2)
                self.assertIn("instruction", request)
                self.assertTrue(
                    request["candidates"][1]["content"][0]["image_url"][
                        "url"
                    ].startswith("data:image/png;base64,")
                )
            return [
                _result(str(request["request_id"])) for request in requests
            ], 0.04

        with tempfile.TemporaryDirectory() as directory:
            events_path = Path(directory) / "events.jsonl"
            journal = Journal(events_path)
            with (
                patch(
                    "bench.runner.score_request", return_value=_result("warmup")
                ) as score,
                patch(
                    "bench.runner.concurrent_score_requests", side_effect=measured
                ) as concurrent,
            ):
                _execute_case(
                    server=self.server,
                    model=self.model,
                    case=case,
                    journal=journal,
                    telemetry=Mock(),
                )
            events = journal.events()
            serialized_events = events_path.read_text()

        self.assertEqual(score.call_count, 1)
        self.assertEqual(concurrent.call_count, 2)
        self.assertIn("data:image/png;base64,", str(score.call_args.kwargs))
        self.assertNotIn("data:image/png;base64,", serialized_events)
        request_events = [
            event for event in events if event["event"] == "request_complete"
        ]
        self.assertEqual(len(request_events), 4)
        self.assertTrue(all(event["validation"]["passed"] for event in request_events))
        completed = [event for event in events if event["event"] == "case_complete"]
        self.assertTrue(completed[0]["validation_passed"])

    def test_direct_warmup_uses_multimodal_score_documents(self) -> None:
        with patch(
            "bench.runner.score_request", return_value=_result("warmup")
        ) as score:
            _run_warmups(self.server, self.model, _case(warmups=2))

        self.assertEqual(score.call_count, 2)
        self.assertTrue(
            all(
                isinstance(call.kwargs["candidates"][0], dict)
                for call in score.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
