from __future__ import annotations

from pathlib import Path
import unittest

from bench.manifest import (
    ManifestError,
    load_models,
    load_suite,
    validate_benchmark_selection,
)
from bench.runner import _estimated_context_tokens


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = (
    ROOT
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sglang_native.toml"
)
LONG_SUITE_PATH = (
    ROOT
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sglang_long_context.toml"
)
DEPTH_SUITE_PATH = (
    ROOT
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sglang_ple_depth_c8.toml"
)
QUALITY_V2_SUITE_PATH = (
    ROOT
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sglang_quality_v2.toml"
)


class Qwen38FlashNextSglangNativeSuiteTests(unittest.TestCase):
    def test_suite_geometry_is_bounded_and_native_backend_safe(self) -> None:
        suite = load_suite(SUITE_PATH)
        cases = {case.id: case for case in suite.cases}

        self.assertEqual(suite.id, "qwen38-flash-next-sglang-native")
        self.assertIn("NVMe-PLE", suite.description)
        self.assertEqual(len(cases), 12)
        self.assertNotIn("cache", {case.kind for case in cases.values()})
        self.assertEqual(cases["chat-smoke"].kind, "decode")
        self.assertEqual(cases["json-smoke"].requires, ("chat", "json"))
        self.assertEqual(cases["tools-smoke"].requires, ("chat", "tools"))
        self.assertEqual(cases["synthetic-exact-answer-v2"].kind, "quality")

        decode = cases["decode-256-c1"]
        self.assertEqual(
            (decode.kind, decode.max_output_tokens, decode.concurrency),
            ("decode", 256, 1),
        )

        for concurrency in (1, 2, 4):
            case = cases[f"fresh-short-c{concurrency}"]
            with self.subTest(case=case.id):
                self.assertEqual(case.kind, "concurrency")
                self.assertEqual(case.concurrency, concurrency)
                self.assertEqual(case.prompt_repetitions, 0)
                self.assertEqual(case.warmups, 0)

        for target in (8192, 32768):
            with self.subTest(target=target):
                prefill = cases[f"prefill-repeat-{target}"]
                needle = cases[f"long-context-needle-{target}-c1"]
                self.assertEqual(prefill.kind, "prefill")
                self.assertEqual(prefill.prompt_repetitions, target)
                self.assertEqual(prefill.concurrency, 1)
                self.assertEqual(needle.kind, "capability")
                self.assertEqual(needle.prompt_repetitions, target)
                self.assertEqual(needle.concurrency, 1)

        for target in (131072, 245760):
            self.assertNotIn(f"long-context-needle-{target}-c1", cases)

        for case in cases.values():
            with self.subTest(context_estimate=case.id):
                estimated_tokens, _ = _estimated_context_tokens(case)
                self.assertLess(estimated_tokens, 262144)

    def test_ple_depth_suite_matches_depth_and_concurrency_arms(self) -> None:
        suite = load_suite(DEPTH_SUITE_PATH)
        cases = {case.id: case for case in suite.cases}

        self.assertEqual(suite.id, "qwen38-flash-next-sglang-ple-depth-c8")
        self.assertEqual(
            set(cases),
            {
                "ple-study-decode-256-c1-v1",
                "ple-study-fresh-short-c1-v1",
                "ple-study-fresh-short-c2-v1",
                "ple-study-fresh-short-c4-v1",
                "ple-study-fresh-short-c8-v1",
            },
        )
        for case in cases.values():
            with self.subTest(case=case.id):
                self.assertEqual(case.repetitions, 3)
                self.assertEqual(case.max_output_tokens, 256)
                self.assertEqual(case.temperature, 0.0)
        for concurrency in (1, 2, 4, 8):
            self.assertEqual(
                cases[f"ple-study-fresh-short-c{concurrency}-v1"].concurrency,
                concurrency,
            )

    def test_quality_v2_suite_is_strict_c1_n2(self) -> None:
        suite = load_suite(QUALITY_V2_SUITE_PATH)

        self.assertEqual(suite.id, "qwen38-flash-next-sglang-quality-v2")
        self.assertEqual(len(suite.cases), 1)
        case = suite.cases[0]
        self.assertEqual(case.id, "synthetic-exact-answer-v2")
        self.assertEqual(case.kind, "quality")
        self.assertEqual(case.repetitions, 2)
        self.assertEqual(case.concurrency, 1)
        self.assertEqual(case.max_output_tokens, 512)
        self.assertEqual(case.temperature, 0.0)

    def test_ple_study_profiles_are_bound_to_their_exact_suites(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        depth_suite = load_suite(DEPTH_SUITE_PATH)
        quality_suite = load_suite(QUALITY_V2_SUITE_PATH)
        depth_ids = (
            "qwen38-flash-next-nvfp4-mtp1-c8-lazy-ple-mapped-sglang",
            "qwen38-flash-next-nvfp4-mtp2-c8-lazy-ple-mapped-sglang",
            "qwen38-flash-next-nvfp4-mtp3-c8-lazy-ple-mapped-sglang",
            "qwen38-flash-next-nvfp4-mtp3-c8-lazy-ple-omitted-sglang",
        )
        quality_ids = (
            "qwen38-flash-next-nvfp4-mtp3-quality-v2-ple-mapped-sglang",
            "qwen38-flash-next-nvfp4-mtp3-quality-v2-ple-omitted-sglang",
        )

        for profile_id in depth_ids:
            validate_benchmark_selection(models[profile_id], depth_suite)
            with self.assertRaisesRegex(
                ManifestError, "(profile|suite) requires"
            ):
                validate_benchmark_selection(models[profile_id], quality_suite)
        for profile_id in quality_ids:
            validate_benchmark_selection(models[profile_id], quality_suite)
            with self.assertRaisesRegex(
                ManifestError, "(profile|suite) requires"
            ):
                validate_benchmark_selection(models[profile_id], depth_suite)
        unrelated = models["qwen38-flash-next-nvfp4-mtp-sglang"]
        for suite in (depth_suite, quality_suite):
            with self.subTest(suite=suite.id), self.assertRaisesRegex(
                ManifestError, "suite requires"
            ):
                validate_benchmark_selection(unrelated, suite)

    def test_long_suite_is_single_request_and_within_native_window(self) -> None:
        suite = load_suite(LONG_SUITE_PATH)
        cases = {case.id: case for case in suite.cases}

        self.assertEqual(suite.id, "qwen38-flash-next-sglang-long-context")
        self.assertIn("NVMe-PLE", suite.description)
        self.assertEqual(set(cases), {"long-context-needle-245760-c1"})
        case = cases["long-context-needle-245760-c1"]
        self.assertEqual(case.kind, "capability")
        self.assertEqual(case.repetitions, 1)
        self.assertEqual(case.warmups, 0)
        self.assertEqual(case.concurrency, 1)
        estimated_tokens, _ = _estimated_context_tokens(case)
        self.assertLess(estimated_tokens, 262144)

    def test_long_profile_and_suite_must_travel_together(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        native_suite = load_suite(SUITE_PATH)
        long_suite = load_suite(LONG_SUITE_PATH)
        throughput = models["qwen38-flash-next-nvfp4-mtp-sglang"]
        long_context = models[
            "qwen38-flash-next-nvfp4-long-sglang"
        ]

        validate_benchmark_selection(long_context, long_suite)
        with self.assertRaisesRegex(ManifestError, "profile requires"):
            validate_benchmark_selection(long_context, native_suite)
        with self.assertRaisesRegex(ManifestError, "suite requires"):
            validate_benchmark_selection(throughput, long_suite)


if __name__ == "__main__":
    unittest.main()
