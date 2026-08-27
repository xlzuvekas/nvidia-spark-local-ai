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
    _quality_request_arguments,
    _validate_quality_item,
)


def _case(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": "synthetic-exact-answer-v2",
        "case_id": "synthetic-exact-answer-v2--test",
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

    def test_request_ids_are_out_of_band_from_stable_quality_prompts(self) -> None:
        server = SimpleNamespace(
            backend="vllm",
            base_url="http://127.0.0.1:8000/v1",
        )
        model = SimpleNamespace(served_name="quality-test", max_context=4096)
        item = _QUALITY_ITEMS[-1]
        first = _quality_request_arguments(
            server=server,
            model=model,
            case=_case(),
            item=item,
            request_id="request-with-clock-value-1",
            prompt_tag="r0",
        )
        second = _quality_request_arguments(
            server=server,
            model=model,
            case=_case(),
            item=item,
            request_id="request-with-clock-value-2",
            prompt_tag="r0",
        )

        self.assertNotEqual(first["request_id"], second["request_id"])
        self.assertEqual(first["prompt"], second["prompt"])
        self.assertNotIn("clock-value", str(first["prompt"]))
        self.assertIn("protocol v2", str(first["prompt"]))

    def test_frozen_v1_quality_case_preserves_request_id_nonce(self) -> None:
        server = SimpleNamespace(
            backend="vllm",
            base_url="http://127.0.0.1:8000/v1",
        )
        model = SimpleNamespace(served_name="quality-test", max_context=4096)
        legacy_case = _case(
            id="synthetic-exact-answer",
            case_id="synthetic-exact-answer--frozen-v1",
        )
        item = _QUALITY_ITEMS[-1]
        first = _quality_request_arguments(
            server=server,
            model=model,
            case=legacy_case,
            item=item,
            request_id="legacy-request-1",
            prompt_tag="ignored-v2-tag",
        )
        second = _quality_request_arguments(
            server=server,
            model=model,
            case=legacy_case,
            item=item,
            request_id="legacy-request-2",
            prompt_tag="ignored-v2-tag",
        )

        self.assertNotEqual(first["prompt"], second["prompt"])
        self.assertIn("nonce legacy-request-1", str(first["prompt"]))
        self.assertNotIn("protocol v2", str(first["prompt"]))

    def test_ple_study_prompts_match_across_model_derived_case_ids(self) -> None:
        server = SimpleNamespace(
            backend="vllm",
            base_url="http://127.0.0.1:8000/v1",
        )
        case_values = {
            "id": "ple-study-fresh-short-c2-v1",
            "kind": "concurrency",
            "requires": ["chat"],
            "warmups": 0,
            "repetitions": 2,
            "max_output_tokens": 64,
            "temperature": 0.0,
            "concurrency": 2,
            "prompt_repetitions": 0,
        }
        captured: list[list[dict[str, object]]] = []

        def measured(
            *,
            requests: list[dict[str, object]],
            concurrency: int,
            request_function: object,
        ) -> tuple[list[RequestResult], float]:
            del request_function
            self.assertEqual(concurrency, 2)
            captured.append(requests)
            return (
                [
                    _result(str(request["request_id"]), "measurement")
                    for request in requests
                ],
                0.05,
            )

        with tempfile.TemporaryDirectory() as directory, patch(
            "bench.runner.concurrent_chat_requests", side_effect=measured
        ):
            for arm in ("mapped", "omitted"):
                _execute_case(
                    server=server,
                    model=SimpleNamespace(
                        served_name=f"quality-test-{arm}", max_context=4096
                    ),
                    case=SimpleNamespace(
                        **case_values,
                        case_id=f"ple-study-fresh-short-c2-v1--{arm}",
                    ),
                    journal=Journal(Path(directory) / f"{arm}.jsonl"),
                    telemetry=Mock(),
                )

        self.assertEqual(len(captured), 4)
        for repetition in range(2):
            mapped = captured[repetition]
            omitted = captured[repetition + 2]
            self.assertEqual(
                [request["prompt"] for request in mapped],
                [request["prompt"] for request in omitted],
            )
            self.assertEqual(
                [request["request_id"] for request in mapped],
                [request["request_id"] for request in omitted],
            )
            self.assertEqual(
                len({request["request_id"] for request in mapped}), 2
            )
            self.assertTrue(
                all(
                    "--mapped" not in str(request["prompt"])
                    and "--omitted" not in str(request["prompt"])
                    for request in (*mapped, *omitted)
                )
            )

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
