from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.agentic_tools import (
    AGENTIC_SCENARIO_IDS,
    AGENTIC_VARIANT_COUNT,
    AgenticScenarioError,
    DEFAULT_MAX_TURNS,
    MIN_OUTPUT_TOKENS,
    estimate_agentic_context_tokens,
    is_agentic_scenario,
    run_agentic_scenario,
)
from bench.manifest import KNOWN_AGENTIC_CASE_IDS, load_suite
from bench.journal import Journal
from bench.runner import _execute_case


ROOT = Path(__file__).resolve().parents[1]


def _result(
    *,
    content: str = "",
    tool_calls: list[dict[str, object]] | None = None,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=2,
        ttft_s=0.01,
        elapsed_s=0.03,
        emission_events=2,
        finish_reason=finish_reason,
        content=content,
        tool_calls=tool_calls or [],
    )


def _tool_call(
    name: str, arguments: str, *, call_id: str = "call_1"
) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class _ScriptedRequests:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def __call__(self, **arguments: object) -> SimpleNamespace:
        self.calls.append(arguments)
        if not self.responses:
            raise AssertionError("agentic loop made an unexpected request")
        return self.responses.pop(0)


def _run(
    scenario_id: str,
    variant: int,
    script: _ScriptedRequests,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    extra_body: dict[str, object] | None = None,
):
    return run_agentic_scenario(
        scenario_id=scenario_id,
        variant=variant,
        request_function=script,
        request_kwargs={
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "synthetic-model",
        },
        request_id_prefix="ephemeral-test",
        max_turns=max_turns,
        max_output_tokens=4_096,
        temperature=0.0,
        extra_body=extra_body,
    )


class AgenticToolBatteryTests(unittest.TestCase):
    def test_manifest_and_runner_scenario_ids_do_not_drift(self) -> None:
        self.assertEqual(KNOWN_AGENTIC_CASE_IDS, frozenset(AGENTIC_SCENARIO_IDS))

    def test_suite_has_four_long_bounded_three_variant_cases(self) -> None:
        suite = load_suite(ROOT / "manifests" / "suites" / "agentic_tools.toml")

        self.assertEqual(suite.id, "agentic-tools")
        self.assertEqual(tuple(case.id for case in suite.cases), AGENTIC_SCENARIO_IDS)
        for case in suite.cases:
            with self.subTest(case=case.id):
                self.assertEqual(case.kind, "agentic")
                self.assertEqual(case.requires, ("chat", "tools"))
                self.assertEqual(case.warmups, 0)
                self.assertEqual(case.repetitions, AGENTIC_VARIANT_COUNT)
                self.assertEqual(case.concurrency, 1)
                self.assertEqual(case.max_turns, 6)
                self.assertEqual(case.max_output_tokens, 4_096)
                self.assertEqual(case.temperature, 0.0)

    def test_selects_multiply_and_returns_only_scalar_evidence(self) -> None:
        script = _ScriptedRequests(
            [
                _result(
                    tool_calls=[_tool_call("multiply", '{"a":6,"b":7}')],
                    finish_reason="tool_calls",
                ),
                _result(content="FINAL: 42"),
            ]
        )

        result = _run("agentic-select-and-call", 0, script)

        self.assertTrue(result.passed)
        self.assertIsNone(result.failure_code)
        self.assertEqual(result.turns_used, 2)
        self.assertEqual(result.expected_tool_calls, 1)
        self.assertEqual(result.tool_calls_requested, 1)
        self.assertEqual(result.tool_calls_executed, 1)
        self.assertEqual(result.tool_calls_succeeded, 1)
        self.assertTrue(result.tool_sequence_correct)
        self.assertTrue(result.final_answer_correct)
        self.assertEqual(result.prompt_tokens, 20)
        self.assertEqual(result.completion_tokens, 4)
        self.assertEqual(result.emission_events, 4)
        self.assertAlmostEqual(result.request_elapsed_s, 0.06)
        self.assertEqual(result.ttft_s, 0.01)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.elapsed_s, result.wall_s)
        self.assertAlmostEqual(
            result.output_tps, result.completion_tokens / result.wall_s
        )
        self.assertIsNone(result.decode_s)
        self.assertIsNone(result.decode_tps)
        self.assertIsNone(result.decode_metric_source)

        first = script.calls[0]
        self.assertEqual(first["max_tokens"], 4_096)
        self.assertEqual(first["temperature"], 0.0)
        body = first["extra_body"]
        self.assertIsInstance(body, dict)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][1]["role"], "user")
        tool_names = {
            tool["function"]["name"] for tool in body["tools"]
        }
        self.assertEqual(
            tool_names, {"multiply", "add_integers", "lookup_number"}
        )
        second_messages = script.calls[1]["extra_body"]["messages"]
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertEqual(second_messages[-1]["content"], '{"ok":true,"value":42}')

        payload = result.to_dict()
        banned_keys = {
            "arguments",
            "content",
            "expected_answer",
            "messages",
            "prompt",
            "reasoning",
            "request_id",
            "tool_calls",
        }
        self.assertFalse(banned_keys & set(payload))
        scalar_types = (bool, float, int, str, type(None))
        self.assertTrue(all(isinstance(value, scalar_types) for value in payload.values()))

    def test_runner_journals_only_the_scalar_episode_result(self) -> None:
        script = _ScriptedRequests(
            [
                _result(
                    tool_calls=[_tool_call("multiply", '{"a":6,"b":7}')],
                    finish_reason="tool_calls",
                ),
                _result(content="FINAL: 42"),
            ]
        )
        server = SimpleNamespace(
            backend="vllm",
            base_url="http://127.0.0.1:8000/v1",
            authorization=None,
        )
        model = SimpleNamespace(
            served_name="synthetic-model",
            request_body_json=None,
        )
        case = SimpleNamespace(
            id="agentic-select-and-call",
            case_id="agentic-select-and-call--synthetic",
            kind="agentic",
            requires=["chat", "tools"],
            warmups=0,
            repetitions=1,
            max_output_tokens=4_096,
            max_turns=6,
            temperature=0.0,
            concurrency=1,
            prompt_repetitions=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "events.jsonl")
            with patch("bench.runner.stream_chat_request", side_effect=script):
                _execute_case(
                    server=server,
                    model=model,
                    case=case,
                    journal=journal,
                    telemetry=Mock(),
                )

            events = journal.events()
            request_event = next(
                event for event in events if event["event"] == "request_complete"
            )
            result = request_event["result"]
            self.assertTrue(request_event["validation"]["passed"])
            self.assertTrue(result["passed"])
            self.assertNotIn("content", result)
            self.assertNotIn("reasoning", result)
            self.assertNotIn("request_id", result)
            self.assertNotIn("tool_calls", result)
            serialized = journal.path.read_text()
            self.assertNotIn("FINAL: 42", serialized)
            self.assertNotIn('{\\"a\\":6,\\"b\\":7}', serialized)

    def test_no_tool_scenario_abstains_for_every_variant(self) -> None:
        answers = ("ORCHID-27", "EMBER-41", "QUARTZ-63")
        for variant, answer in enumerate(answers):
            with self.subTest(variant=variant):
                script = _ScriptedRequests([_result(content=f"FINAL: {answer}")])

                result = _run("agentic-no-tool", variant, script)

                self.assertTrue(result.passed)
                self.assertEqual(result.turns_used, 1)
                self.assertEqual(result.expected_tool_calls, 0)
                self.assertEqual(result.tool_calls_requested, 0)
                self.assertEqual(result.tool_calls_executed, 0)
                self.assertTrue(result.tool_sequence_correct)

    def test_two_hop_requires_sequential_lookup_then_multiply(self) -> None:
        script = _ScriptedRequests(
            [
                _result(
                    tool_calls=[_tool_call("lookup_number", '{"label":"amber"}')],
                    finish_reason="tool_calls",
                ),
                _result(
                    tool_calls=[
                        _tool_call("multiply", '{"a":23,"b":4}', call_id="call_2")
                    ],
                    finish_reason="tool_calls",
                ),
                _result(content="FINAL: 92"),
            ]
        )

        result = _run("agentic-two-hop", 1, script)

        self.assertTrue(result.passed)
        self.assertEqual(result.turns_used, 3)
        self.assertEqual(result.tool_calls_requested, 2)
        self.assertEqual(result.tool_calls_executed, 2)
        second_messages = script.calls[1]["extra_body"]["messages"]
        self.assertEqual(second_messages[-1]["content"], '{"ok":true,"value":23}')
        third_messages = script.calls[2]["extra_body"]["messages"]
        self.assertEqual(third_messages[-1]["content"], '{"ok":true,"value":92}')

    def test_transient_tool_error_must_be_retried_before_final_answer(self) -> None:
        script = _ScriptedRequests(
            [
                _result(
                    tool_calls=[_tool_call("unstable_lookup", '{"key":"east"}')],
                    finish_reason="tool_calls",
                ),
                _result(
                    tool_calls=[
                        _tool_call(
                            "unstable_lookup", '{"key":"east"}', call_id="call_2"
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                _result(content="FINAL: 58"),
            ]
        )

        result = _run("agentic-tool-error-recovery", 1, script)

        self.assertTrue(result.passed)
        self.assertTrue(result.recovery_required)
        self.assertTrue(result.recovery_succeeded)
        self.assertEqual(result.tool_errors, 1)
        self.assertEqual(result.tool_calls_executed, 2)
        self.assertEqual(result.tool_calls_succeeded, 1)
        second_messages = script.calls[1]["extra_body"]["messages"]
        self.assertEqual(
            second_messages[-1]["content"],
            '{"error":"transient_error","ok":false,"retryable":true}',
        )
        third_messages = script.calls[2]["extra_body"]["messages"]
        self.assertEqual(third_messages[-1]["content"], '{"ok":true,"value":58}')

    def test_unknown_and_malformed_calls_fail_with_allowlisted_codes(self) -> None:
        cases = (
            (
                _tool_call("not_allowlisted", '{"value":1}'),
                "unknown_tool",
                0,
                1,
            ),
            (_tool_call("multiply", "not-json"), "malformed_tool_call", 1, 0),
        )
        for call, code, malformed, unknown in cases:
            with self.subTest(code=code):
                script = _ScriptedRequests(
                    [
                        _result(tool_calls=[call], finish_reason="tool_calls"),
                        _result(content="FINAL: 42"),
                    ]
                )

                result = _run("agentic-select-and-call", 0, script)

                self.assertFalse(result.passed)
                self.assertEqual(result.failure_code, code)
                self.assertEqual(result.malformed_tool_calls, malformed)
                self.assertEqual(result.unknown_tool_calls, unknown)
                self.assertEqual(result.tool_calls_executed, 0)
                self.assertFalse(result.tool_sequence_correct)

    def test_turn_and_tool_call_limits_are_terminal_scalar_failures(self) -> None:
        repeated = [
            _result(
                tool_calls=[_tool_call("multiply", '{"a":6,"b":7}')],
                finish_reason="tool_calls",
            )
            for _ in range(2)
        ]
        turn_limited = _run(
            "agentic-select-and-call",
            0,
            _ScriptedRequests(repeated),
            max_turns=2,
        )
        self.assertFalse(turn_limited.passed)
        self.assertTrue(turn_limited.turn_limit_reached)
        self.assertEqual(turn_limited.failure_code, "turn_limit")
        self.assertEqual(turn_limited.finish_reason, "turn_limit")

        too_many = [
            _tool_call("multiply", '{"a":6,"b":7}', call_id=f"call_{index}")
            for index in range(17)
        ]
        call_limited = _run(
            "agentic-select-and-call",
            0,
            _ScriptedRequests(
                [_result(tool_calls=too_many, finish_reason="tool_calls")]
            ),
        )
        self.assertFalse(call_limited.passed)
        self.assertEqual(call_limited.failure_code, "tool_call_limit")
        self.assertEqual(call_limited.finish_reason, "tool_call_limit")
        self.assertEqual(call_limited.tool_calls_executed, 0)

    def test_output_limit_malformed_shapes_and_metrics_fail_closed(self) -> None:
        limited = _run(
            "agentic-no-tool",
            0,
            _ScriptedRequests(
                [_result(content="FINAL: ORCHID-27", finish_reason="length")]
            ),
        )
        self.assertFalse(limited.passed)
        self.assertEqual(limited.failure_code, "output_limit")
        self.assertEqual(limited.length_terminated_turns, 1)

        duplicate_or_nonstandard = (
            '{"a":6,"a":6,"b":7}',
            '{"a":NaN,"b":7}',
        )
        for arguments in duplicate_or_nonstandard:
            with self.subTest(arguments=arguments):
                result = _run(
                    "agentic-select-and-call",
                    0,
                    _ScriptedRequests(
                        [
                            _result(
                                tool_calls=[_tool_call("multiply", arguments)],
                                finish_reason="tool_calls",
                            ),
                            _result(content="FINAL: 42"),
                        ]
                    ),
                )
                self.assertFalse(result.passed)
                self.assertEqual(result.failure_code, "malformed_tool_call")

        malformed_shape = _result(content="FINAL: ORCHID-27")
        malformed_shape.tool_calls = {}  # type: ignore[assignment]
        with self.assertRaisesRegex(ValueError, "tool_calls must be a list"):
            _run("agentic-no-tool", 0, _ScriptedRequests([malformed_shape]))

        with self.assertRaisesRegex(
            AgenticScenarioError, "tool turn finish reason"
        ):
            _run(
                "agentic-select-and-call",
                0,
                _ScriptedRequests(
                    [_result(tool_calls=[_tool_call("multiply", '{"a":6,"b":7}')])]
                ),
            )
        with self.assertRaisesRegex(
            AgenticScenarioError, "final turn finish reason"
        ):
            _run(
                "agentic-no-tool",
                0,
                _ScriptedRequests(
                    [_result(content="FINAL: ORCHID-27", finish_reason="content_filter")]
                ),
            )

        fractional_metric = _result(content="FINAL: ORCHID-27")
        fractional_metric.prompt_tokens = 1.5
        with self.assertRaisesRegex(ValueError, "prompt_tokens"):
            _run("agentic-no-tool", 0, _ScriptedRequests([fractional_metric]))

    def test_request_failures_are_wrapped_without_response_text(self) -> None:
        marker = "private-response-body"

        def fail(**_kwargs: object) -> SimpleNamespace:
            raise RuntimeError(marker)

        with self.assertRaises(AgenticScenarioError) as caught:
            run_agentic_scenario(
                scenario_id="agentic-no-tool",
                variant=0,
                request_function=fail,
                request_kwargs={"model": "synthetic"},
                request_id_prefix="ephemeral-test",
                max_turns=6,
                max_output_tokens=4_096,
            )
        self.assertNotIn(marker, str(caught.exception))

    def test_input_bounds_and_controlled_request_fields_are_enforced(self) -> None:
        self.assertTrue(is_agentic_scenario(AGENTIC_SCENARIO_IDS[0]))
        self.assertFalse(is_agentic_scenario("tool-call-correctness"))
        self.assertEqual(
            estimate_agentic_context_tokens(
                max_turns=6, max_output_tokens=4_096
            ),
            26_624,
        )
        with self.assertRaisesRegex(ValueError, "at least 2048"):
            estimate_agentic_context_tokens(
                max_turns=6, max_output_tokens=MIN_OUTPUT_TOKENS - 1
            )
        with self.assertRaisesRegex(ValueError, "between 1 and 8"):
            estimate_agentic_context_tokens(max_turns=9, max_output_tokens=4_096)
        with self.assertRaisesRegex(ValueError, "temperature 0"):
            run_agentic_scenario(
                scenario_id=AGENTIC_SCENARIO_IDS[0],
                variant=0,
                request_function=_ScriptedRequests([]),
                request_kwargs={"model": "synthetic"},
                request_id_prefix="ephemeral-test",
                max_turns=6,
                max_output_tokens=4_096,
                temperature=0.1,
            )
        with self.assertRaisesRegex(ValueError, "controlled fields"):
            run_agentic_scenario(
                scenario_id=AGENTIC_SCENARIO_IDS[0],
                variant=0,
                request_function=_ScriptedRequests([]),
                request_kwargs={"model": "synthetic", "prompt": "override"},
                request_id_prefix="ephemeral-test",
                max_turns=6,
                max_output_tokens=4_096,
            )
        with self.assertRaisesRegex(ValueError, "agentic fields"):
            _run(
                AGENTIC_SCENARIO_IDS[0],
                0,
                _ScriptedRequests([]),
                extra_body={"tool_choice": "required"},
            )
        with self.assertRaisesRegex(ValueError, "unsupported agentic settings"):
            _run(
                AGENTIC_SCENARIO_IDS[0],
                0,
                _ScriptedRequests([]),
                extra_body={"stop": ["unsafe-override"]},
            )


if __name__ == "__main__":
    unittest.main()
