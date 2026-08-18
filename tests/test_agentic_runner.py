from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.agentic_tools import AgenticRunResult
from bench.journal import Journal
from bench.runner import _estimated_context_tokens, _execute_case


def _case() -> SimpleNamespace:
    return SimpleNamespace(
        id="agentic-two-hop",
        case_id="agentic-two-hop--test",
        kind="agentic",
        requires=["chat", "tools"],
        warmups=0,
        repetitions=3,
        max_output_tokens=4096,
        max_turns=6,
        temperature=0.0,
        concurrency=1,
        prompt_repetitions=0,
    )


def _result(variant: int) -> AgenticRunResult:
    return AgenticRunResult(
        schema_version=1,
        scenario_id="agentic-two-hop",
        variant=variant,
        passed=True,
        failure_code=None,
        max_turns=6,
        max_output_tokens=4096,
        turns_used=3,
        expected_tool_calls=2,
        tool_calls_requested=2,
        tool_calls_executed=2,
        tool_calls_succeeded=2,
        tool_errors=0,
        malformed_tool_calls=0,
        unknown_tool_calls=0,
        final_answer_emitted=True,
        final_answer_correct=True,
        tool_sequence_correct=True,
        recovery_required=False,
        recovery_succeeded=False,
        turn_limit_reached=False,
        prompt_tokens=100,
        completion_tokens=20,
        emission_events=4,
        first_turn_ttft_s=0.1,
        request_elapsed_s=1.4,
        wall_s=1.5,
        length_terminated_turns=0,
        elapsed_s=1.5,
        ttft_s=0.1,
        finish_reason="stop",
        output_tps=20 / 1.5,
        decode_s=None,
        decode_tps=None,
        decode_metric_source=None,
    )


class AgenticRunnerTests(unittest.TestCase):
    def test_context_estimate_covers_every_turn_and_tool_history(self) -> None:
        estimate, basis = _estimated_context_tokens(_case())

        self.assertEqual(estimate, 26_624)
        self.assertIn("agentic_episode", basis)
        self.assertLess(estimate, 32_768)

    def test_execute_case_journals_only_scalar_episode_results(self) -> None:
        server = SimpleNamespace(
            backend="llamacpp", base_url="http://127.0.0.1:8000/v1"
        )
        model = SimpleNamespace(
            served_name="agentic-model",
            tasks=["chat", "tools"],
            max_context=32_768,
            request_body_json='{"chat_template_kwargs":{"enable_thinking":false}}',
            architecture="qwen",
        )

        def run(**kwargs: object) -> AgenticRunResult:
            self.assertEqual(kwargs["scenario_id"], "agentic-two-hop")
            self.assertEqual(kwargs["max_turns"], 6)
            self.assertEqual(kwargs["max_output_tokens"], 4096)
            self.assertEqual(
                kwargs["extra_body"],
                {"chat_template_kwargs": {"enable_thinking": False}},
            )
            return _result(int(kwargs["variant"]))

        with tempfile.TemporaryDirectory() as directory:
            events_path = Path(directory) / "events.jsonl"
            journal = Journal(events_path)
            telemetry = Mock()
            with patch("bench.runner.run_agentic_scenario", side_effect=run) as call:
                _execute_case(
                    server=server,
                    model=model,
                    case=_case(),
                    journal=journal,
                    telemetry=telemetry,
                )
            events = journal.events()
            serialized = events_path.read_text(encoding="utf-8")

        self.assertEqual(call.call_count, 3)
        self.assertEqual(
            [item.kwargs["variant"] for item in call.call_args_list], [0, 1, 2]
        )
        measured = [
            event for event in events if event["event"] == "request_complete"
        ]
        self.assertEqual(len(measured), 3)
        self.assertTrue(all(event["validation"]["passed"] for event in measured))
        self.assertTrue(all(event["result"]["passed"] for event in measured))
        self.assertNotIn('"messages"', serialized)
        self.assertNotIn('"tool_calls"', serialized)
        self.assertNotIn('"content"', serialized)
        completed = [event for event in events if event["event"] == "case_complete"]
        self.assertTrue(completed[0]["validation_passed"])


if __name__ == "__main__":
    unittest.main()
