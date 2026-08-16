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
    _prime_model,
    _rerank_inputs,
    _run_warmups,
    _validate_capability,
)


def _result(request_id: str, candidate_count: int) -> RerankResult:
    scores = [0.05 / (index + 1) for index in range(candidate_count)]
    scores[1] = 0.95
    ranking = sorted(range(candidate_count), key=lambda index: (-scores[index], index))
    return RerankResult(
        request_id=request_id,
        started_at_ns=1,
        prompt_tokens=24,
        completion_tokens=0,
        ttft_s=0.01,
        elapsed_s=0.01,
        decode_s=0.0,
        decode_tps=0.0,
        output_tps=0.0,
        emission_events=1,
        finish_reason="stop",
        response_model="reranker",
        candidate_count=candidate_count,
        scores=scores,
        ranking=ranking,
        top_index=ranking[0],
        pairs_per_s=candidate_count / 0.01,
        finite=True,
    )


def _case(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": "rerank-candidates",
        "case_id": "rerank-candidates--test",
        "kind": "capability",
        "requires": ["rerank"],
        "warmups": 1,
        "repetitions": 2,
        "max_output_tokens": 1,
        "temperature": 0.0,
        "concurrency": 2,
        "prompt_repetitions": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RerankRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = SimpleNamespace(
            backend="vllm", base_url="http://127.0.0.1:8000/v1"
        )
        self.model = SimpleNamespace(
            served_name="reranker", tasks=["rerank"], max_context=8192
        )

    def test_inputs_and_validation_are_deterministic(self) -> None:
        case = _case(prompt_repetitions=32)
        first = _rerank_inputs(case)
        second = _rerank_inputs(case)

        self.assertEqual(first, second)
        query, candidates, relevant_index = first
        self.assertIn("Red Planet", query)
        self.assertEqual(len(candidates), 32)
        self.assertEqual(relevant_index, 1)
        result = _result("rerank-validation", len(candidates))
        self.assertEqual(
            _validate_capability(case, result), {"passed": True, "reason": None}
        )

        result.ranking = [0, *range(1, len(candidates))]
        result.top_index = 0
        validation = _validate_capability(case, result)
        self.assertFalse(validation["passed"])
        self.assertIn("top candidate", validation["reason"])

    def test_warmup_and_prime_use_vllm_score(self) -> None:
        case = _case(warmups=2, prompt_repetitions=6)
        with patch("bench.runner.score_request", return_value=_result("mock", 6)) as score:
            _run_warmups(self.server, self.model, case)

        self.assertEqual(score.call_count, 2)
        self.assertTrue(all(len(call.kwargs["candidates"]) == 6 for call in score.call_args_list))

        with patch("bench.runner.score_request", return_value=_result("prime", 4)) as score:
            result = _prime_model(self.server, self.model)

        self.assertIsInstance(result, RerankResult)
        self.assertEqual(len(score.call_args.kwargs["candidates"]), 4)
        self.assertEqual(score.call_args.kwargs["query"], _rerank_inputs(_case(prompt_repetitions=0))[0])

    def test_measured_rerank_runs_concurrent_batches_and_journals_validation(self) -> None:
        case = _case()

        def measured(*, requests: list[dict[str, object]], concurrency: int):
            self.assertEqual(concurrency, 2)
            self.assertEqual(len(requests), 2)
            self.assertTrue(all(len(request["candidates"]) == 8 for request in requests))
            return [
                _result(str(request["request_id"]), len(request["candidates"]))
                for request in requests
            ], 0.02

        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "events.jsonl")
            telemetry = Mock()
            with (
                patch("bench.runner.score_request", return_value=_result("warmup", 8)) as score,
                patch("bench.runner.concurrent_score_requests", side_effect=measured) as concurrent,
            ):
                _execute_case(
                    server=self.server,
                    model=self.model,
                    case=case,
                    journal=journal,
                    telemetry=telemetry,
                )

            self.assertEqual(score.call_count, 1)
            self.assertEqual(concurrent.call_count, 2)
            events = journal.events()

        request_events = [event for event in events if event["event"] == "request_complete"]
        self.assertEqual(len(request_events), 4)
        self.assertTrue(all(event["validation"]["passed"] for event in request_events))
        completed = [event for event in events if event["event"] == "case_complete"]
        self.assertEqual(len(completed), 1)
        self.assertTrue(completed[0]["validation_passed"])


if __name__ == "__main__":
    unittest.main()
