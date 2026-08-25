from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from bench.manifest import (
    ManifestError,
    load_models,
    load_suite,
    validate_benchmark_selection,
)
from bench.runner import _estimated_context_tokens, _prompt


ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "manifests" / "suites"


class LongContextTpsTests(unittest.TestCase):
    def test_long_concurrency_prompt_uses_requested_synthetic_context(self) -> None:
        case = SimpleNamespace(
            id="long-decode-8192-c8",
            kind="concurrency",
            requires=("chat",),
            prompt_repetitions=17,
            max_output_tokens=256,
        )

        prompt = _prompt(case, "unique-request")

        self.assertEqual(prompt.count("measurement "), 17)
        self.assertLess(prompt.index("unique-request"), prompt.index("measurement "))
        self.assertIn("Continue until the output limit", prompt)
        estimate, basis = _estimated_context_tokens(case)
        self.assertGreaterEqual(estimate, 17 + 256)
        self.assertEqual(basis, "prompt_words_plus_request_margin")

    def test_long_suite_has_conservative_c1_c2_geometry(self) -> None:
        suite = load_suite(SUITES / "long_context_tps.toml")
        cases = {case.id: case for case in suite.cases}

        self.assertEqual(suite.id, "long-context-tps")
        self.assertEqual(len(cases), 10)
        expected_decode_geometry = {
            "long-decode-61440-c2": (61440, 2),
            "long-decode-122880-c2": (122880, 2),
            "long-decode-245760-c2": (245760, 2),
        }
        for case_id, (
            prompt_repetitions,
            concurrency,
        ) in expected_decode_geometry.items():
            with self.subTest(case_id=case_id):
                case = cases[case_id]
                self.assertEqual(case.kind, "concurrency")
                self.assertEqual(case.prompt_repetitions, prompt_repetitions)
                self.assertEqual(case.concurrency, concurrency)
                self.assertEqual(case.repetitions, 3)
                self.assertEqual(case.max_output_tokens, 256)
                estimate, _ = _estimated_context_tokens(case)
                self.assertLessEqual(estimate, 262144)

        self.assertLess(262144 * 2, 726747)

        for case in cases.values():
            if case.id.startswith("long-context-needle"):
                self.assertEqual(case.kind, "capability")
                self.assertEqual(case.repetitions, 1)
                self.assertEqual(case.max_output_tokens, 32)

    def test_saturation_suite_stays_below_measured_qwen38_kv_ceiling(self) -> None:
        suite = load_suite(SUITES / "throughput_saturation.toml")
        cases = {case.id: case for case in suite.cases}

        self.assertEqual(suite.id, "throughput-saturation")
        self.assertEqual(len(cases), 15)
        near_limit = cases["long-decode-30720-c32"]
        estimate, _ = _estimated_context_tokens(near_limit)
        self.assertLess(estimate * near_limit.concurrency, 1144149)
        self.assertEqual(cases["decode-saturation-c64"].concurrency, 64)
        self.assertEqual(cases["long-decode-8192-c64"].concurrency, 64)
        self.assertEqual(cases["long-context-needle-30720-c32"].concurrency, 32)

    def test_profiles_split_native_context_from_saturation(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        for model_id in (
            "qwen36-35b-a3b-nvfp4-mtp3-long-tps",
            "qwen38-27b-nvfp4-mtp3-long-tps",
        ):
            with self.subTest(model_id=model_id):
                model = models[model_id]
                self.assertEqual(model.max_context, 262144)
                self.assertEqual(model.native_context, 262144)
                self.assertIn("--no-enable-prefix-caching", model.args)
                self.assertNotIn("--enable-prefix-caching", model.args)
                self.assertEqual(
                    model.args[model.args.index("--max-model-len") + 1],
                    "262144",
                )
                self.assertEqual(
                    model.args[model.args.index("--max-num-seqs") + 1], "2"
                )

        expected_long_tuning = {
            "qwen36-35b-a3b-nvfp4-mtp3-long-tps": ("0.40", "8192"),
            "qwen38-27b-nvfp4-mtp3-long-tps": ("0.52", "4096"),
        }
        for model_id, (utilization, batch_tokens) in expected_long_tuning.items():
            model = models[model_id]
            self.assertEqual(
                model.args[model.args.index("--gpu-memory-utilization") + 1],
                utilization,
            )
            self.assertEqual(
                model.args[model.args.index("--max-num-batched-tokens") + 1],
                batch_tokens,
            )

        for model_id in (
            "qwen36-35b-a3b-nvfp4-mtp3-tps64",
            "qwen38-27b-nvfp4-mtp3-tps64",
        ):
            with self.subTest(model_id=model_id):
                model = models[model_id]
                self.assertEqual(model.max_context, 32768)
                self.assertEqual(model.native_context, 262144)
                self.assertIn("--no-enable-prefix-caching", model.args)
                self.assertNotIn("--enable-prefix-caching", model.args)
                self.assertEqual(
                    model.args[model.args.index("--max-num-seqs") + 1], "64"
                )

    def test_suites_and_dedicated_profiles_are_fail_closed_paired(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        suites = {
            "long-context-tps": load_suite(SUITES / "long_context_tps.toml"),
            "throughput-saturation": load_suite(
                SUITES / "throughput_saturation.toml"
            ),
        }
        smoke = load_suite(SUITES / "smoke.toml")
        profile_ids_by_suite = {
            "long-context-tps": (
                "qwen36-35b-a3b-nvfp4-mtp3-long-tps",
                "qwen38-27b-nvfp4-mtp3-long-tps",
            ),
            "throughput-saturation": (
                "qwen36-35b-a3b-nvfp4-mtp3-tps64",
                "qwen38-27b-nvfp4-mtp3-tps64",
            ),
        }

        for suite_id, profile_ids in profile_ids_by_suite.items():
            suite = suites[suite_id]
            other_suite = suites[
                "throughput-saturation"
                if suite_id == "long-context-tps"
                else "long-context-tps"
            ]
            for profile_id in profile_ids:
                with self.subTest(suite_id=suite_id, profile_id=profile_id):
                    validate_benchmark_selection(models[profile_id], suite)
                    with self.assertRaisesRegex(ManifestError, "profile requires"):
                        validate_benchmark_selection(models[profile_id], smoke)
                    with self.assertRaisesRegex(
                        ManifestError, "exact dedicated cache-off profiles"
                    ):
                        validate_benchmark_selection(models[profile_id], other_suite)

        historical_profiles = {
            "long-context-tps": "qwen38-27b-nvfp4-mtp3",
            "throughput-saturation": "qwen38-27b-nvfp4-mtp3-throughput",
        }
        for suite_id, profile_id in historical_profiles.items():
            with self.subTest(suite_id=suite_id, historical_profile=profile_id):
                with self.assertRaisesRegex(
                    ManifestError, "exact dedicated cache-off profiles"
                ):
                    validate_benchmark_selection(
                        models[profile_id], suites[suite_id]
                    )


if __name__ == "__main__":
    unittest.main()
