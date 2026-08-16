from __future__ import annotations

from pathlib import Path
import unittest

from bench.manifest import load_models, load_suite
from bench.runner import _estimated_context_tokens


ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "manifests" / "suites"

PREFILL_REPAIR_MODELS = (
    "ollama-mistral-medium-3.5-128b-q4-k-m",
    "ollama-gemma4-31b-q4-k-m",
    "ollama-nemotron-cascade-2-q4-k-m",
    "ollama-nemotron-3-super-q4-k-m",
    "ollama-qwen3.5-35b-a3b-q4-k-m",
    "ollama-llama3.3-70b-q4-k-m",
    "ollama-qwen3-30b-a3b-q4-k-m",
    "ollama-gemma3-12b-q4-k-m",
    "ollama-glm-4.7-flash-q4-k-m",
)

REASONING_NEEDLE_MODELS = (
    "ollama-mistral-medium-3.5-128b-q4-k-m",
    "ollama-gemma4-31b-q4-k-m",
    "ollama-nemotron-cascade-2-q4-k-m",
    "ollama-nemotron-3-super-q4-k-m",
    "ollama-qwen3.5-35b-a3b-q4-k-m",
    "ollama-glm-4.7-flash-q4-k-m",
)


class OllamaRepairSuiteTests(unittest.TestCase):
    def test_prefill_repair_only_changes_the_quick_output_budget(self) -> None:
        quick = load_suite(SUITES / "quick.toml")
        repair = load_suite(SUITES / "ollama_prefill_repair.toml")
        quick_prefill = {case.id: case for case in quick.cases if case.kind == "prefill"}
        repair_cases = {case.id: case for case in repair.cases}

        self.assertEqual(repair.id, "ollama-prefill-repair")
        self.assertEqual(set(repair_cases), set(quick_prefill))
        self.assertTrue(all(case.kind == "prefill" for case in repair.cases))
        self.assertTrue(all(case.max_output_tokens == 2 for case in repair.cases))
        self.assertNotIn(
            "decode", {case.kind for case in repair.cases}
        )
        self.assertNotIn(
            "concurrency", {case.kind for case in repair.cases}
        )

        for case_id, repaired in repair_cases.items():
            original = quick_prefill[case_id]
            for field in (
                "id",
                "kind",
                "requires",
                "warmups",
                "repetitions",
                "temperature",
                "concurrency",
                "prompt_repetitions",
            ):
                self.assertEqual(
                    getattr(repaired, field),
                    getattr(original, field),
                    f"{case_id}:{field}",
                )

    def test_reasoning_needle_reuses_the_existing_content_oracle(self) -> None:
        reasoning_quick = load_suite(SUITES / "reasoning_quick.toml")
        repair = load_suite(SUITES / "ollama_reasoning_needle_repair.toml")
        expected = next(
            case
            for case in reasoning_quick.cases
            if case.id == "long-context-needle-8192"
        )

        self.assertEqual(repair.id, "ollama-reasoning-needle-repair")
        self.assertEqual(repair.cases, (expected,))
        self.assertEqual(repair.cases[0].max_output_tokens, 128)
        self.assertNotIn(
            repair.cases[0].kind,
            {"decode", "concurrency"},
        )

    def test_recommended_profiles_have_conservative_context_headroom(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        prefill = load_suite(SUITES / "ollama_prefill_repair.toml")
        needle = load_suite(SUITES / "ollama_reasoning_needle_repair.toml")

        largest_prefill = max(
            _estimated_context_tokens(case)[0] for case in prefill.cases
        )
        needle_estimate = _estimated_context_tokens(needle.cases[0])[0]

        for model_id in PREFILL_REPAIR_MODELS:
            model = models[model_id]
            with self.subTest(suite="prefill", model=model_id):
                self.assertEqual(model.backend, "ollama")
                self.assertIn("chat", model.tasks)
                self.assertLessEqual(largest_prefill, model.max_context)

        for model_id in REASONING_NEEDLE_MODELS:
            model = models[model_id]
            with self.subTest(suite="needle", model=model_id):
                self.assertIn("thinking", model.tasks)
                self.assertLessEqual(needle_estimate, model.max_context)

        deepseek_ocr = models["ollama-deepseek-ocr-f16"]
        self.assertLess(deepseek_ocr.max_context, largest_prefill)
        self.assertLess(deepseek_ocr.max_context, needle_estimate)


if __name__ == "__main__":
    unittest.main()
