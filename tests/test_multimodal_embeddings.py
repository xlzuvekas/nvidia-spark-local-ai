from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.client import MultimodalEmbeddingResult
from bench.journal import Journal
from bench.runner import (
    _execute_case,
    _multimodal_embedding_request_arguments,
    _prime_model,
    _run_warmups,
    _validate_capability,
)


def _result(
    request_id: str,
    *,
    relevant_similarity: float = 0.9,
    unrelated_similarity: float = 0.1,
    finite: bool = True,
) -> MultimodalEmbeddingResult:
    return MultimodalEmbeddingResult(
        request_id=request_id,
        started_at_ns=1,
        prompt_tokens=30,
        completion_tokens=0,
        ttft_s=0.01,
        elapsed_s=0.03,
        decode_s=0.0,
        decode_tps=0.0,
        output_tps=0.0,
        emission_events=3,
        finish_reason="stop",
        response_model="qwen3-vl-embedding",
        dimension=3,
        batch_size=3,
        items_per_s=100.0,
        finite=finite,
        norms=[1.0, 1.0, 1.0],
        image_latency_s=0.01,
        relevant_text_latency_s=0.01,
        unrelated_text_latency_s=0.01,
        relevant_similarity=relevant_similarity,
        unrelated_similarity=unrelated_similarity,
        similarity_margin=relevant_similarity - unrelated_similarity,
    )


def _case(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": "red-image-cross-modal",
        "case_id": "red-image-cross-modal--test",
        "kind": "capability",
        "requires": ["embeddings", "vision"],
        "warmups": 1,
        "repetitions": 2,
        "max_output_tokens": 1,
        "temperature": 0.0,
        "concurrency": 2,
        "prompt_repetitions": 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class MultimodalEmbeddingRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = SimpleNamespace(
            backend="vllm", base_url="http://127.0.0.1:8000/v1"
        )
        self.model = SimpleNamespace(
            served_name="qwen3-vl-embedding",
            tasks=["embeddings", "vision"],
            max_context=8192,
        )

    def test_arguments_are_local_deterministic_and_validation_compares_similarity(
        self,
    ) -> None:
        case = _case()
        first = _multimodal_embedding_request_arguments(
            server=self.server,
            model=self.model,
            case=case,
            request_id="first",
        )
        second = _multimodal_embedding_request_arguments(
            server=self.server,
            model=self.model,
            case=case,
            request_id="second",
        )

        self.assertTrue(first["image_data_url"].startswith("data:image/png;base64,"))
        self.assertEqual(first["image_data_url"], second["image_data_url"])
        self.assertIn("red", first["relevant_text"].lower())
        passing = _validate_capability(case, _result("passing"))
        self.assertEqual(passing, {"passed": True, "reason": None})

        failing = _validate_capability(
            case,
            _result(
                "failing", relevant_similarity=0.1, unrelated_similarity=0.9
            ),
        )
        self.assertFalse(failing["passed"])
        self.assertIn("rank", failing["reason"])

    def test_warmup_and_prime_use_multimodal_chat_embeddings(self) -> None:
        case = _case(warmups=2)
        with patch(
            "bench.runner.multimodal_embedding_request",
            return_value=_result("warmup"),
        ) as request:
            _run_warmups(self.server, self.model, case)

        self.assertEqual(request.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["image_data_url"].startswith("data:image/png;base64,")
                for call in request.call_args_list
            )
        )

        with patch(
            "bench.runner.multimodal_embedding_request",
            return_value=_result("prime"),
        ) as request:
            result = _prime_model(self.server, self.model)

        self.assertIsInstance(result, MultimodalEmbeddingResult)
        self.assertIn("red", request.call_args.kwargs["relevant_text"].lower())

    def test_measured_batches_are_journaled_without_vectors_or_image_data(self) -> None:
        case = _case()

        def measured(*, requests: list[dict[str, object]], concurrency: int):
            self.assertEqual(concurrency, 2)
            self.assertEqual(len(requests), 2)
            return [
                _result(str(request["request_id"])) for request in requests
            ], 0.04

        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "events.jsonl")
            telemetry = Mock()
            with (
                patch(
                    "bench.runner.multimodal_embedding_request",
                    return_value=_result("warmup"),
                ),
                patch(
                    "bench.runner.concurrent_multimodal_embedding_requests",
                    side_effect=measured,
                ) as concurrent,
            ):
                _execute_case(
                    server=self.server,
                    model=self.model,
                    case=case,
                    journal=journal,
                    telemetry=telemetry,
                )
            events = journal.events()

        self.assertEqual(concurrent.call_count, 2)
        request_events = [event for event in events if event["event"] == "request_complete"]
        self.assertEqual(len(request_events), 4)
        self.assertTrue(all(event["validation"]["passed"] for event in request_events))
        self.assertTrue(
            all("embedding" not in event["result"] for event in request_events)
        )
        completed = [event for event in events if event["event"] == "case_complete"]
        self.assertTrue(completed[0]["validation_passed"])


if __name__ == "__main__":
    unittest.main()
