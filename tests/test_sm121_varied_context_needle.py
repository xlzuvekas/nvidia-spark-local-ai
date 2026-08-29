from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.journal import Journal
from bench.manifest import CaseSpec, ManifestError, validate_case
from bench.runner import (
    VariedContextNeedleError,
    _VARIED_CONTEXT_FILLER_LEXICON,
    _VARIED_CONTEXT_NEEDLE_FAILURE_MESSAGE,
    _estimated_context_tokens,
    _execute_case,
    _prompt,
    _validate_capability,
    _varied_context_filler_records,
    _varied_context_needle,
)
from bench.sglang_sm121_storage import (
    SM121_STORAGE_CONTEXT_LENGTH,
    SM121_STORAGE_VARIED_CONTEXT_BUDGET_TOKENS,
    SM121_STORAGE_VARIED_CONTEXT_CASE_ID,
    SM121_STORAGE_VARIED_CONTEXT_CHAT_PROMPT_TOKENS,
    SM121_STORAGE_VARIED_CONTEXT_OUTPUT_TOKENS,
    SM121_STORAGE_VARIED_CONTEXT_PROMPT_SHA256,
    SM121_STORAGE_VARIED_CONTEXT_RECORDS,
)


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
            "completion_tokens": 12,
            "reasoning_tokens": 4,
            "ttft_s": 0.1,
            "elapsed_s": 0.2,
            "decode_s": 0.1,
            "decode_tps": 120.0,
            "output_tps": 60.0,
            "emission_events": 2,
            "finish_reason": "stop",
            "response_model": "synthetic",
            "decode_metric_source": "client_estimate",
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
            "prompt": "sensitive generated prompt",
        }


class Sm121VariedContextNeedleTests(unittest.TestCase):
    def _case(
        self,
        *,
        records: int = 512,
        depth: str = "mid",
        seed: str = "1",
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id=(
                f"sm121-varied-context-needle-{records}-{depth}-s{seed}-c1-v1"
            ),
            case_id=(
                f"sm121-varied-context-needle-{records}-{depth}-s{seed}-"
                "c1-v1--test"
            ),
            kind="capability",
            requires=("chat",),
            warmups=0,
            repetitions=1,
            max_output_tokens=64,
            temperature=0.0,
            concurrency=1,
            prompt_repetitions=records,
            max_turns=1,
        )

    def test_manifest_contract_is_typed_and_rejects_protocol_drift(self) -> None:
        case = CaseSpec(
            id="sm121-varied-context-needle-512-tail-s17-c1-v1",
            kind="capability",
            requires=("chat",),
            max_output_tokens=64,
            prompt_repetitions=512,
        )
        validate_case(case)
        self.assertIsNotNone(case.varied_context_needle)
        assert case.varied_context_needle is not None
        self.assertEqual(case.varied_context_needle.filler_records, 512)
        self.assertEqual(case.varied_context_needle.depth, "tail")
        self.assertEqual(case.varied_context_needle.seed, "17")
        self.assertEqual(case.varied_context_needle.protocol_version, 1)

        invalid_cases = (
            replace(case, prompt_repetitions=511),
            replace(case, concurrency=2),
            replace(case, repetitions=2),
            replace(case, max_output_tokens=31),
            replace(case, id="sm121-varied-context-needle-512-middle-s17-c1-v1"),
        )
        for invalid in invalid_cases:
            with self.subTest(case=invalid):
                with self.assertRaises(ManifestError):
                    validate_case(invalid)

    def test_prompt_is_stable_varied_and_depth_controlled(self) -> None:
        mid = self._case(depth="mid", seed="1")
        tail = self._case(depth="tail", seed="1")
        alternate_seed = self._case(depth="mid", seed="2")
        mid_target = _varied_context_needle(mid)

        prompt_one = _prompt(mid, "request-nonce-one")
        prompt_two = _prompt(mid, "request-nonce-two")
        tail_prompt = _prompt(tail, "request-nonce-one")
        alternate_prompt = _prompt(alternate_seed, "request-nonce-one")

        self.assertEqual(prompt_one, prompt_two)
        self.assertNotIn("Benchmark nonce", prompt_one)
        self.assertEqual(prompt_one.count(mid_target), 1)
        self.assertNotEqual(prompt_one, alternate_prompt)
        self.assertGreater(
            tail_prompt.index("Recovery phrase:"),
            prompt_one.index("Recovery phrase:"),
        )
        filler_words = {
            word
            for record in _varied_context_filler_records(
                case_id=mid.id,
                record_count=mid.prompt_repetitions,
            )
            for word in record
        }
        self.assertEqual(len(_VARIED_CONTEXT_FILLER_LEXICON), 64)
        self.assertEqual(filler_words, set(_VARIED_CONTEXT_FILLER_LEXICON))

        estimated_tokens, basis = _estimated_context_tokens(mid)
        self.assertEqual(
            basis,
            "varied_context_two_words_per_record_plus_output_and_template_margin",
        )
        self.assertGreaterEqual(estimated_tokens, len(prompt_one.split()))

    def test_pinned_canary_budget_is_tokenizer_verified_and_has_margin(self) -> None:
        case = self._case(
            records=SM121_STORAGE_VARIED_CONTEXT_RECORDS,
            depth="mid",
            seed="20260828",
        )
        self.assertEqual(case.id, SM121_STORAGE_VARIED_CONTEXT_CASE_ID)
        prompt = _prompt(case, "request-nonce")
        self.assertEqual(
            hashlib.sha256(prompt.encode()).hexdigest(),
            SM121_STORAGE_VARIED_CONTEXT_PROMPT_SHA256,
        )
        estimated_tokens, basis = _estimated_context_tokens(case)
        self.assertEqual(
            basis, "pinned_qwen_tokenizer_chat_template_plus_output"
        )
        self.assertEqual(
            estimated_tokens, SM121_STORAGE_VARIED_CONTEXT_BUDGET_TOKENS
        )
        self.assertEqual(
            estimated_tokens,
            SM121_STORAGE_VARIED_CONTEXT_CHAT_PROMPT_TOKENS
            + SM121_STORAGE_VARIED_CONTEXT_OUTPUT_TOKENS,
        )
        self.assertLess(estimated_tokens, SM121_STORAGE_CONTEXT_LENGTH)

    def test_visible_answer_must_match_every_multiword_token_exactly(self) -> None:
        case = self._case()
        target = _varied_context_needle(case)
        accepted = SimpleNamespace(
            request_id="ignored", content=target.upper().replace(" ", "  ")
        )
        wrong_cases = (
            f"{target}!",
            target.split()[0],
            f"{target} {target}",
            "alder beryl citron dahlia ember fable garnet hazel indigo juniper kepler lilac",
        )

        self.assertTrue(_validate_capability(case, accepted, model=None)["passed"])
        for content in wrong_cases:
            with self.subTest(content=content):
                result = SimpleNamespace(request_id="ignored", content=content)
                validation = _validate_capability(case, result, model=None)
                self.assertFalse(validation["passed"])
                self.assertNotIn(target, str(validation))

    def test_execution_journals_scalar_only_and_redacts_failures(self) -> None:
        case = self._case()
        target = _varied_context_needle(case)
        request_id = f"{case.case_id}-r0-w0-123"
        result = _Result(
            request_id=request_id,
            content=target,
            reasoning=f"The recovery phrase is {target}.",
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
                self.assertEqual(str(requests[0]["prompt"]).count(target), 1)
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
        for field in (
            "content",
            "reasoning",
            "tool_calls",
            "prompt",
            "request_id",
            "started_at_ns",
        ):
            self.assertNotIn(field, measured[0]["result"])
        for sensitive_value in ("sensitive generated prompt", target):
            self.assertNotIn(sensitive_value, serialized)

        marker = "synthetic-http-error-body-marker"

        def failing_request(
            *, requests: list[dict[str, object]], **_: object
        ) -> tuple[list[_Result], float]:
            prompt = str(requests[0]["prompt"])
            raise RuntimeError(
                f"HTTP 500: {marker}; echoed prompt={prompt}; target={target}"
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed-events.jsonl"
            journal = Journal(path)
            with (
                patch("bench.runner.time.time_ns", return_value=123),
                patch(
                    "bench.runner.concurrent_chat_requests", side_effect=failing_request
                ),
            ):
                with self.assertRaises(VariedContextNeedleError) as raised:
                    _execute_case(
                        server=server,
                        model=model,
                        case=case,
                        journal=journal,
                        telemetry=Mock(),
                    )
            serialized = path.read_text(encoding="utf-8")
            events = journal.events()

        self.assertEqual(str(raised.exception), _VARIED_CONTEXT_NEEDLE_FAILURE_MESSAGE)
        failed = [event for event in events if event["event"] == "case_failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["error_type"], "VariedContextNeedleError")
        self.assertEqual(failed[0]["error"], _VARIED_CONTEXT_NEEDLE_FAILURE_MESSAGE)
        for sensitive_value in (marker, target):
            self.assertNotIn(sensitive_value, serialized)


if __name__ == "__main__":
    unittest.main()
