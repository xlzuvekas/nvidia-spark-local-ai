from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.client import RequestResult
from bench.journal import Journal
from bench.runner import (
    _QUALITY_ITEMS,
    _execute_case,
    _extract_quality_answer,
    _quality_answers_match,
    _quality_prompt,
    _validate_quality_item,
)


def _case(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": "synthetic-exact-answer",
        "case_id": "synthetic-exact-answer--test",
        "kind": "quality",
        "requires": ["chat"],
        "warmups": 0,
        "repetitions": 1,
        "max_output_tokens": 64,
        "temperature": 0.0,
        "concurrency": 2,
        "prompt_repetitions": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _result(request_id: str, content: str, *, reasoning: str = "") -> RequestResult:
    return RequestResult(
        request_id=request_id,
        started_at_ns=1,
        prompt_tokens=30,
        completion_tokens=3,
        reasoning_tokens=None,
        ttft_s=0.01,
        elapsed_s=0.03,
        decode_s=0.02,
        decode_tps=150.0,
        output_tps=100.0,
        emission_events=3,
        finish_reason="stop",
        response_model="quality-test",
        content=content,
        reasoning=reasoning,
        tool_calls=[],
    )


class ChatQualityTests(unittest.TestCase):
    def test_embedded_items_cover_four_categories_with_stable_prompts(self) -> None:
        self.assertEqual(
            {item.category for item in _QUALITY_ITEMS},
            {"arithmetic", "logic", "instruction_following", "code_reasoning"},
        )
        self.assertEqual(len(_QUALITY_ITEMS), 4)
        self.assertEqual(len({item.id for item in _QUALITY_ITEMS}), 4)
        for item in _QUALITY_ITEMS:
            with self.subTest(item=item.id):
                prompt = _quality_prompt(item, "fixed-nonce")
                self.assertEqual(prompt, _quality_prompt(item, "fixed-nonce"))
                self.assertIn(item.id, prompt)
                self.assertIn("FINAL: <answer>", prompt)

    def test_answer_extraction_accepts_bounded_format_variations(self) -> None:
        examples = (
            ("FINAL: 83", "83", "83"),
            ("  final : 83.0  ", "83.0", "83"),
            ("```text\n**FINAL:** `silver`\n```", "`silver`", "silver"),
            ("FINAL: **83**.", "**83**.", "83"),
            ("`FINAL: 83`", "83", "83"),
            ("**FINAL: no**", "no", "no"),
            ('"No."', '"No."', "no"),
            ("No.", "No.", "no"),
            ("Brief work omitted.\nFINAL: 9", "9", "9"),
        )
        for content, extracted_expected, answer_expected in examples:
            with self.subTest(content=content):
                extracted, error = _extract_quality_answer(content)
                self.assertIsNone(error)
                self.assertEqual(extracted, extracted_expected)
                self.assertIsNotNone(extracted)
                self.assertTrue(_quality_answers_match(extracted, answer_expected))

    def test_answer_extraction_rejects_ambiguous_or_inexact_output(self) -> None:
        extracted, error = _extract_quality_answer(
            "FINAL: 83\nOn reconsideration, FINAL: 84"
        )
        self.assertIsNone(extracted)
        self.assertIn("multiple", str(error))

        extracted, error = _extract_quality_answer("The answer\nis 83")
        self.assertIsNone(extracted)
        self.assertIn("neither", str(error))
        self.assertFalse(_quality_answers_match("about 83", "83"))

    def test_validation_reads_content_only_and_never_executes_output(self) -> None:
        item = _QUALITY_ITEMS[-1]
        malicious = "FINAL: __import__('os').system('false')"
        validation = _validate_quality_item(
            item,
            _result("malicious", malicious, reasoning="FINAL: 9"),
        )

        self.assertFalse(validation["passed"])
        self.assertEqual(validation["quality_item_id"], "code-01")
        self.assertEqual(validation["extracted_answer"], malicious.removeprefix("FINAL: "))

    def test_measured_workload_uses_both_chat_adapters_and_journals_scores(self) -> None:
        answers = {
            "arithmetic-01": "FINAL: 83",
            "logic-01": "No.",
            "instruction-01": "Explanation.\nFINAL: silver",
            "code-01": "```text\nFINAL: 9.0\n```",
        }
        model = SimpleNamespace(served_name="quality-test", max_context=4096)

        for backend in ("vllm", "ollama"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                server = SimpleNamespace(
                    backend=backend,
                    base_url=(
                        "http://127.0.0.1:8000/v1"
                        if backend == "vllm"
                        else "http://127.0.0.1:11434/v1"
                    ),
                )

                def measured(
                    *,
                    requests: list[dict[str, object]],
                    concurrency: int,
                    request_function: object,
                ) -> tuple[list[RequestResult], float]:
                    self.assertEqual(concurrency, 2)
                    self.assertEqual(len(requests), 2)
                    expected_function = (
                        "stream_chat_request"
                        if backend == "vllm"
                        else "stream_ollama_chat_request"
                    )
                    self.assertEqual(
                        getattr(request_function, "__name__"), expected_function
                    )
                    for request in requests:
                        if backend == "ollama":
                            self.assertEqual(request["context_size"], 4096)
                        else:
                            self.assertNotIn("context_size", request)
                    results = []
                    for request in requests:
                        request_id = str(request["request_id"])
                        item_id = next(
                            item_id for item_id in answers if item_id in request_id
                        )
                        results.append(_result(request_id, answers[item_id]))
                    return results, 0.05

                journal = Journal(Path(directory) / "events.jsonl")
                with patch(
                    "bench.runner.concurrent_chat_requests", side_effect=measured
                ) as concurrent:
                    _execute_case(
                        server=server,
                        model=model,
                        case=_case(),
                        journal=journal,
                        telemetry=Mock(),
                    )
                events = journal.events()

                self.assertEqual(concurrent.call_count, 2)
                request_events = [
                    event for event in events if event["event"] == "request_complete"
                ]
                self.assertEqual(len(request_events), 4)
                self.assertTrue(
                    all(event["validation"]["passed"] for event in request_events)
                )
                self.assertEqual(
                    {
                        event["validation"]["quality_category"]
                        for event in request_events
                    },
                    {"arithmetic", "logic", "instruction_following", "code_reasoning"},
                )
                completed = [
                    event for event in events if event["event"] == "case_complete"
                ]
                self.assertTrue(completed[0]["validation_passed"])


if __name__ == "__main__":
    unittest.main()
