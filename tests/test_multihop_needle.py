from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.manifest import load_models, load_suite
from bench.journal import Journal
from bench.runner import (
    MultiHopNeedleError,
    _MULTI_HOP_FAILURE_MESSAGE,
    _estimated_context_tokens,
    _execute_case,
    _multi_hop_needle,
    _multi_hop_path,
    _multi_hop_values,
    _prompt,
    _validate_capability,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "manifests" / "suites" / "llamacpp_multihop_long_context.toml"


class _Result:
    def __init__(self, *, request_id: str, content: str, reasoning: str) -> None:
        self.request_id = request_id
        self.content = content
        self.reasoning = reasoning
        self.tool_calls: list[dict[str, object]] = []

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "started_at_ns": 1,
            "prompt_tokens": 64,
            "completion_tokens": 8,
            "reasoning_tokens": 4,
            "ttft_s": 0.1,
            "elapsed_s": 0.2,
            "decode_s": 0.1,
            "decode_tps": 80.0,
            "output_tps": 40.0,
            "emission_events": 2,
            "finish_reason": "stop",
            "response_model": "synthetic",
            "decode_metric_source": "client_estimate",
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
            "prompt": "sensitive generated prompt",
        }


class MultiHopNeedleTests(unittest.TestCase):
    def _case(self) -> SimpleNamespace:
        return SimpleNamespace(
            id="long-context-multi-hop-needle-32",
            case_id="long-context-multi-hop-needle-32--test",
            kind="capability",
            requires=("chat",),
            warmups=0,
            repetitions=1,
            max_output_tokens=32,
            temperature=0.0,
            concurrency=1,
            prompt_repetitions=32,
        )

    def test_prompt_places_two_nonce_derived_relations_amid_decoys(self) -> None:
        case = self._case()
        nonce = "fixed-nonce"
        source, relay, final = _multi_hop_values(nonce)
        decoy_one = _multi_hop_path(nonce, "decoy-one")
        decoy_two = _multi_hop_path(nonce, "decoy-two")

        prompt = _prompt(case, nonce)
        first_relation = f"Source record: source {source} routes through relay {relay}."
        second_relation = f"Relay record: relay {relay} has final archive key {final}."

        self.assertEqual(_multi_hop_needle(nonce), final)
        self.assertIn(first_relation, prompt)
        self.assertIn(second_relation, prompt)
        self.assertIn(decoy_one[2], prompt)
        self.assertIn(decoy_two[2], prompt)
        self.assertLess(prompt.index(first_relation), prompt.index(second_relation))
        self.assertEqual(prompt.count("Source record:"), 3)
        self.assertEqual(prompt.count("Relay record:"), 3)
        self.assertEqual(prompt.count(final), 1)

    def test_validator_requires_only_the_exact_visible_final_key(self) -> None:
        case = self._case()
        nonce = "validator-nonce"
        _, relay, final = _multi_hop_values(nonce)

        hidden_only = SimpleNamespace(
            request_id=nonce,
            content=relay,
            reasoning=f"The final key is {final}",
        )
        extra_text = SimpleNamespace(
            request_id=nonce,
            content=f"Answer: {final}",
            reasoning="",
        )
        exact_visible = SimpleNamespace(
            request_id=nonce,
            content=f"  {final}\n",
            reasoning="",
        )

        self.assertFalse(_validate_capability(case, hidden_only)["passed"])
        self.assertFalse(_validate_capability(case, extra_text)["passed"])
        self.assertTrue(_validate_capability(case, exact_visible)["passed"])

    def test_multihop_suite_fits_native_long_context_profiles(self) -> None:
        suite = load_suite(SUITE_PATH)
        self.assertEqual(suite.id, "llamacpp-multihop-long-context")
        self.assertEqual(
            tuple(
                (case.prompt_repetitions, case.repetitions) for case in suite.cases
            ),
            ((32_768, 3), (65_536, 3), (131_072, 3), (245_760, 1)),
        )

        models = load_models(ROOT / "manifests" / "models.toml")
        profiles = (
            models["qwen36-35b-a3b-ud-q4-k-xl-llamacpp"],
            models["qwen36-35b-a3b-ud-q4-k-xl-llamacpp-mtp2"],
            models["qwen38-27b-ud-q4-k-xl-llamacpp-long-context"],
            models["qwen38-27b-ud-q4-k-xl-llamacpp-mtp5-long-context"],
        )
        for case in suite.cases:
            with self.subTest(case=case.id):
                self.assertEqual(
                    case.id,
                    f"long-context-multi-hop-needle-{case.prompt_repetitions}",
                )
                self.assertEqual(case.kind, "capability")
                self.assertEqual(case.requires, ("chat",))
                self.assertEqual(case.warmups, 0)
                self.assertEqual(case.max_output_tokens, 32)
                self.assertEqual(case.temperature, 0.0)
                self.assertEqual(case.concurrency, 1)
                estimated_tokens, basis = _estimated_context_tokens(case)
                self.assertEqual(basis, "prompt_words_plus_request_margin")
                self.assertGreater(estimated_tokens, case.prompt_repetitions)
                for profile in profiles:
                    self.assertLessEqual(estimated_tokens, profile.max_context)

    def test_measured_multihop_event_is_scalar_only(self) -> None:
        case = self._case()
        request_id = f"{case.case_id}-r0-w0-123"
        source, relay, final = _multi_hop_values(request_id)
        result = _Result(
            request_id=request_id,
            content=final,
            reasoning=f"First use {source}, then {relay}, then {final}.",
        )
        server = SimpleNamespace(
            backend="llamacpp", base_url="http://127.0.0.1:8000/v1"
        )
        model = SimpleNamespace(architecture="qwen", served_name="synthetic")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            journal = Journal(path)
            def run_request(
                *, requests: list[dict[str, object]], **_: object
            ) -> tuple[list[_Result], float]:
                self.assertEqual(len(requests), 1)
                self.assertIn(source, str(requests[0]["prompt"]))
                self.assertIn(relay, str(requests[0]["prompt"]))
                self.assertIn(final, str(requests[0]["prompt"]))
                return [result], 0.2

            with (
                patch("bench.runner.time.time_ns", return_value=123),
                patch("bench.runner.concurrent_chat_requests", side_effect=run_request),
            ):
                _execute_case(
                    server=server,
                    model=model,
                    case=case,
                    journal=journal,
                    telemetry=Mock(),
                )
            serialized = path.read_text(encoding="utf-8")
            events = journal.events()

        measured = [event for event in events if event["event"] == "request_complete"]
        self.assertEqual(len(measured), 1)
        self.assertTrue(measured[0]["validation"]["passed"])
        self.assertNotIn("content", measured[0]["result"])
        self.assertNotIn("reasoning", measured[0]["result"])
        self.assertNotIn("tool_calls", measured[0]["result"])
        self.assertNotIn("prompt", measured[0]["result"])
        self.assertNotIn("request_id", measured[0]["result"])
        self.assertNotIn("started_at_ns", measured[0]["result"])
        for sensitive_value in (
            "sensitive generated prompt",
            source,
            relay,
            final,
        ):
            self.assertNotIn(sensitive_value, serialized)

    def test_failed_multihop_request_journals_only_safe_error_scalars(self) -> None:
        case = self._case()
        request_id = f"{case.case_id}-r0-w0-123"
        source, relay, final = _multi_hop_values(request_id)
        marker = "synthetic-http-error-body-marker"
        server = SimpleNamespace(
            backend="llamacpp", base_url="http://127.0.0.1:8000/v1"
        )
        model = SimpleNamespace(architecture="qwen", served_name="synthetic")

        def failing_request(
            *, requests: list[dict[str, object]], **_: object
        ) -> tuple[list[_Result], float]:
            self.assertEqual(len(requests), 1)
            prompt = str(requests[0]["prompt"])
            self.assertIn(source, prompt)
            self.assertIn(relay, prompt)
            self.assertIn(final, prompt)
            raise RuntimeError(
                f"HTTP 500: {marker}; echoed prompt={prompt}; final key={final}"
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            journal = Journal(path)
            with (
                patch("bench.runner.time.time_ns", return_value=123),
                patch(
                    "bench.runner.concurrent_chat_requests", side_effect=failing_request
                ),
            ):
                with self.assertRaises(MultiHopNeedleError) as raised:
                    _execute_case(
                        server=server,
                        model=model,
                        case=case,
                        journal=journal,
                        telemetry=Mock(),
                    )
            serialized = path.read_text(encoding="utf-8")
            events = journal.events()

        self.assertEqual(str(raised.exception), _MULTI_HOP_FAILURE_MESSAGE)
        failed = [event for event in events if event["event"] == "case_failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["error_type"], "MultiHopNeedleError")
        self.assertEqual(failed[0]["error"], _MULTI_HOP_FAILURE_MESSAGE)
        for sensitive_value in (marker, source, relay, final):
            self.assertNotIn(sensitive_value, serialized)


if __name__ == "__main__":
    unittest.main()
