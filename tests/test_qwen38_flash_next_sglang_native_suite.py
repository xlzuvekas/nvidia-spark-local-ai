from __future__ import annotations

from pathlib import Path
import unittest

from bench.manifest import load_suite
from bench.runner import _estimated_context_tokens


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = (
    ROOT
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sglang_native.toml"
)


class Qwen38FlashNextSglangNativeSuiteTests(unittest.TestCase):
    def test_suite_geometry_is_bounded_and_native_backend_safe(self) -> None:
        suite = load_suite(SUITE_PATH)
        cases = {case.id: case for case in suite.cases}

        self.assertEqual(suite.id, "qwen38-flash-next-sglang-native")
        self.assertEqual(len(cases), 14)
        self.assertNotIn("cache", {case.kind for case in cases.values()})
        self.assertEqual(cases["chat-smoke"].kind, "decode")
        self.assertEqual(cases["json-smoke"].requires, ("chat", "json"))
        self.assertEqual(cases["tools-smoke"].requires, ("chat", "tools"))
        self.assertEqual(cases["synthetic-exact-answer"].kind, "quality")

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
            case = cases[f"long-context-needle-{target}-c1"]
            with self.subTest(case=case.id):
                self.assertEqual(case.kind, "capability")
                self.assertEqual(case.repetitions, 1)
                self.assertEqual(case.warmups, 0)
                self.assertEqual(case.concurrency, 1)

        for case in cases.values():
            with self.subTest(context_estimate=case.id):
                estimated_tokens, _ = _estimated_context_tokens(case)
                self.assertLess(estimated_tokens, 262144)


if __name__ == "__main__":
    unittest.main()
