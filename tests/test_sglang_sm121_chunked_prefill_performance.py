"""Offline contract tests for the SM121 1K-versus-2K prefill study."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from bench.manifest import ManifestError, load_models, load_suite, validate_benchmark_selection
from bench.sglang_sm121_chunked_prefill_performance import (
    SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_CHUNK_SIZE,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CASE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_CHUNK_SIZE,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_SUITE_ID,
    SM121ChunkedPrefillPerformanceError,
    sm121_chunked_prefill_performance_arm,
    validate_sm121_chunked_prefill_performance_candidate,
    validate_sm121_chunked_prefill_performance_pair,
    validate_sm121_chunked_prefill_performance_suite,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "manifests" / "models.toml"
SUITE = (
    ROOT
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sm121_triton_storage_chunked_prefill_performance_v1.toml"
)


def _argument_value(arguments: tuple[str, ...], flag: str) -> str:
    indexes = [index for index, value in enumerate(arguments) if value == flag]
    if len(indexes) != 1 or indexes[0] + 1 >= len(arguments):
        raise AssertionError(f"invalid {flag} placement")
    return arguments[indexes[0] + 1]


class SM121ChunkedPrefillPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        models = load_models(MODELS)
        cls.control = models[SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID]
        cls.candidate = models[SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID]
        cls.suite = load_suite(SUITE)

    def test_pair_is_current_cache_on_and_differs_only_by_chunk_size(self) -> None:
        validate_sm121_chunked_prefill_performance_pair(
            self.control, self.candidate
        )
        self.assertEqual("A", sm121_chunked_prefill_performance_arm(self.control))
        self.assertEqual("B", sm121_chunked_prefill_performance_arm(self.candidate))
        self.assertEqual(
            str(SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_CHUNK_SIZE),
            _argument_value(self.control.args, "--chunked-prefill-size"),
        )
        self.assertEqual(
            str(SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_CHUNK_SIZE),
            _argument_value(self.candidate.args, "--chunked-prefill-size"),
        )
        self.assertEqual(("chat",), self.control.tasks)
        self.assertEqual(("chat",), self.candidate.tasks)
        self.assertNotIn("--tool-call-parser", self.control.args)
        self.assertNotIn("--tool-call-parser", self.candidate.args)

    def test_candidate_rejects_any_other_serving_delta(self) -> None:
        arguments = list(self.candidate.args)
        index = arguments.index("--max-running-requests")
        arguments[index + 1] = "2"
        drifted = replace(self.candidate, args=tuple(arguments))
        with self.assertRaisesRegex(
            SM121ChunkedPrefillPerformanceError, "profile args changed"
        ):
            validate_sm121_chunked_prefill_performance_candidate(drifted)

    def test_pair_rejects_same_chunk_size(self) -> None:
        arguments = list(self.candidate.args)
        index = arguments.index("--chunked-prefill-size")
        arguments[index + 1] = str(
            SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_CHUNK_SIZE
        )
        drifted = replace(self.candidate, args=tuple(arguments))
        with self.assertRaisesRegex(
            SM121ChunkedPrefillPerformanceError, "profile args changed"
        ):
            validate_sm121_chunked_prefill_performance_pair(self.control, drifted)

    def test_suite_is_exact_and_profiles_cannot_select_other_suites(self) -> None:
        validate_sm121_chunked_prefill_performance_suite(self.suite)
        validate_benchmark_selection(self.control, self.suite)
        self.assertEqual(
            SM121_CHUNKED_PREFILL_PERFORMANCE_SUITE_ID, self.suite.id
        )
        self.assertEqual(
            (
                SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID,
                SM121_CHUNKED_PREFILL_PERFORMANCE_CASE_ID,
            ),
            tuple(case.id for case in self.suite.cases),
        )
        unrelated = load_suite(ROOT / "manifests" / "suites" / "smoke.toml")
        with self.assertRaisesRegex(ManifestError, "requires"):
            validate_benchmark_selection(self.control, unrelated)

    def test_suite_rejects_a_shape_rewrite(self) -> None:
        changed = replace(
            self.suite,
            cases=(
                self.suite.cases[0],
                replace(self.suite.cases[1], max_output_tokens=64),
            ),
        )
        with self.assertRaisesRegex(
            SM121ChunkedPrefillPerformanceError, "max_output_tokens"
        ):
            validate_sm121_chunked_prefill_performance_suite(changed)


if __name__ == "__main__":
    unittest.main()
