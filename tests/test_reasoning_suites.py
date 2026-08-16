from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from bench.manifest import CaseSpec, load_suite
from bench.runner import (
    _estimated_context_tokens,
    _needle,
    _request_arguments,
    _validate_capability,
)


ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "manifests" / "suites"


class ReasoningSuiteTests(unittest.TestCase):
    def _load_pair(self, stem: str):
        return (
            load_suite(SUITES / f"{stem}.toml"),
            load_suite(SUITES / f"reasoning_{stem}.toml"),
        )

    def test_reasoning_variants_preserve_base_case_shapes(self) -> None:
        for stem in ("quick", "core"):
            with self.subTest(suite=stem):
                base, reasoning = self._load_pair(stem)
                base_cases = {case.id: case for case in base.cases}
                reasoning_cases = {case.id: case for case in reasoning.cases}
                self.assertEqual(set(reasoning_cases), set(base_cases))
                self.assertEqual(reasoning.id, f"reasoning-{stem}")
                self.assertIn("low-effort reasoning", reasoning.description)

                for case_id, original in base_cases.items():
                    variant = reasoning_cases[case_id]
                    for field in (
                        "id",
                        "kind",
                        "warmups",
                        "repetitions",
                        "temperature",
                        "concurrency",
                        "prompt_repetitions",
                    ):
                        self.assertEqual(
                            getattr(variant, field),
                            getattr(original, field),
                            f"{reasoning.id}:{case_id}:{field}",
                        )
                    self.assertEqual(variant.requires, original.requires)
                    if original.kind == "prefill":
                        self.assertEqual(variant.max_output_tokens, 64)
                    elif case_id.startswith("long-context-needle"):
                        self.assertEqual(variant.max_output_tokens, 128)
                    else:
                        self.assertEqual(
                            variant.max_output_tokens, original.max_output_tokens
                        )

    def test_existing_quick_and_core_budgets_remain_unchanged(self) -> None:
        for stem in ("quick", "core"):
            suite = load_suite(SUITES / f"{stem}.toml")
            prefill = [case for case in suite.cases if case.kind == "prefill"]
            needles = [
                case
                for case in suite.cases
                if case.id.startswith("long-context-needle")
            ]
            with self.subTest(suite=stem):
                self.assertTrue(prefill)
                self.assertTrue(all(case.max_output_tokens == 1 for case in prefill))
                self.assertEqual(len(needles), 1)
                self.assertEqual(needles[0].max_output_tokens, 32)

    def test_prefill_request_honors_manifest_output_budget(self) -> None:
        quick, reasoning = self._load_pair("quick")
        normal_case = next(case for case in quick.cases if case.kind == "prefill")
        reasoning_case = next(
            case for case in reasoning.cases if case.kind == "prefill"
        )
        server = SimpleNamespace(backend="vllm", base_url="http://127.0.0.1:8000")
        model = SimpleNamespace(
            served_name="openai/gpt-oss-120b",
            request_body_json='{"reasoning_effort":"low"}',
            max_context=32768,
        )

        normal = _request_arguments(
            server=server,
            model=model,
            case=normal_case,
            request_id="normal-prefill",
        )
        expanded = _request_arguments(
            server=server,
            model=model,
            case=reasoning_case,
            request_id="reasoning-prefill",
        )

        self.assertEqual(normal["max_tokens"], 1)
        self.assertEqual(expanded["max_tokens"], 64)
        self.assertEqual(expanded["extra_body"], {"reasoning_effort": "low"})
        self.assertEqual(normal_case.prompt_repetitions, reasoning_case.prompt_repetitions)

    def test_decode_and_concurrency_request_arguments_remain_comparable(self) -> None:
        server = SimpleNamespace(backend="vllm", base_url="http://127.0.0.1:8000")
        model = SimpleNamespace(
            served_name="openai/gpt-oss-120b",
            request_body_json='{"reasoning_effort":"low"}',
            max_context=32768,
        )
        for stem in ("quick", "core"):
            base, reasoning = self._load_pair(stem)
            reasoning_cases = {case.id: case for case in reasoning.cases}
            for original in base.cases:
                if original.kind not in {"decode", "concurrency"}:
                    continue
                variant = reasoning_cases[original.id]
                with self.subTest(suite=stem, case=original.id):
                    self.assertEqual(
                        _request_arguments(
                            server=server,
                            model=model,
                            case=original,
                            request_id="comparable-request",
                        ),
                        _request_arguments(
                            server=server,
                            model=model,
                            case=variant,
                            request_id="comparable-request",
                        ),
                    )

    def test_prefill_context_estimate_includes_manifest_output_budget(self) -> None:
        shared = {
            "id": "prefill-repeat-2048",
            "kind": "prefill",
            "requires": ["chat"],
            "prompt_repetitions": 2048,
        }
        one_token = SimpleNamespace(**shared, max_output_tokens=1)
        reasoning_budget = SimpleNamespace(**shared, max_output_tokens=64)

        original_estimate, original_basis = _estimated_context_tokens(one_token)
        expanded_estimate, expanded_basis = _estimated_context_tokens(reasoning_budget)

        self.assertEqual(original_basis, "prompt_words_plus_request_margin")
        self.assertEqual(expanded_basis, original_basis)
        self.assertEqual(expanded_estimate - original_estimate, 63)

    def test_needle_validation_ignores_hidden_reasoning(self) -> None:
        suite = load_suite(SUITES / "reasoning_quick.toml")
        case: CaseSpec = next(
            case for case in suite.cases if case.id.startswith("long-context-needle")
        )
        request_id = "needle-content-only"
        key = _needle(request_id)

        hidden_only = SimpleNamespace(
            request_id=request_id,
            content="SPARK-TRUNCATED",
            reasoning=f"The answer is {key}",
        )
        visible = SimpleNamespace(
            request_id=request_id,
            content=key,
            reasoning="",
        )

        self.assertFalse(_validate_capability(case, hidden_only)["passed"])
        self.assertTrue(_validate_capability(case, visible)["passed"])


if __name__ == "__main__":
    unittest.main()
