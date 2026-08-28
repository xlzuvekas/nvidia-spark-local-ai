from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from bench.manifest import (
    ManifestError,
    ModelSpec,
    load_models,
    load_suite,
    validate_benchmark_selection,
)
from bench.runner import _estimated_context_tokens


ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = ROOT / "manifests" / "models.toml"
SUITE_PATH = (
    ROOT
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sglang_agent64k_autoresearch.toml"
)
AGENTIC_SUITE_PATH = ROOT / "manifests" / "suites" / "agentic_tools.toml"
NATIVE_SUITE_PATH = (
    ROOT
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sglang_native.toml"
)

BASELINE_ID = (
    "qwen38-flash-next-nvfp4-mtp2-agent64k-low-ple-mapped-sglang"
)
MTP3_ID = "qwen38-flash-next-nvfp4-mtp3-agent64k-low-ple-mapped-sglang"
CHUNK2K_ID = (
    "qwen38-flash-next-nvfp4-mtp2-agent64k-low-chunk2k-ple-mapped-sglang"
)
NONE_ID = "qwen38-flash-next-nvfp4-mtp2-agent64k-none-ple-mapped-sglang"
PROFILE_IDS = (BASELINE_ID, MTP3_ID, CHUNK2K_ID, NONE_ID)
SOURCE_ID = "qwen38-flash-next-nvfp4-mtp2-c8-lazy-ple-mapped-sglang"
LOW_BODY = (
    '{"chat_template_kwargs":{"enable_thinking":true,'
    '"reasoning_effort":"low"}}'
)
NONE_BODY = '{"chat_template_kwargs":{"enable_thinking":false}}'


def _arg_value(model: ModelSpec, flag: str) -> str:
    indexes = [index for index, argument in enumerate(model.args) if argument == flag]
    if len(indexes) != 1:
        raise AssertionError(
            f"expected exactly one {flag!r} in {model.id}, found {len(indexes)}"
        )
    index = indexes[0]
    if index + 1 >= len(model.args):
        raise AssertionError(f"{flag!r} has no value in {model.id}")
    return model.args[index + 1]


def _replace_arg(args: tuple[str, ...], flag: str, value: str) -> tuple[str, ...]:
    updated = list(args)
    index = updated.index(flag)
    updated[index + 1] = value
    return tuple(updated)


def _candidate_from_baseline(
    baseline: ModelSpec,
    candidate: ModelSpec,
    *,
    arg_changes: tuple[tuple[str, str], ...] = (),
    request_body_json: str | None = None,
) -> ModelSpec:
    args = _replace_arg(
        baseline.args,
        "--served-model-name",
        candidate.served_name,
    )
    for flag, value in arg_changes:
        args = _replace_arg(args, flag, value)
    return replace(
        baseline,
        id=candidate.id,
        description=candidate.description,
        served_name=candidate.served_name,
        request_body_json=(
            baseline.request_body_json
            if request_body_json is None
            else request_body_json
        ),
        args=args,
    )


def _baseline_args_from_source(
    source: ModelSpec,
    *,
    served_name: str,
) -> tuple[str, ...]:
    args = _replace_arg(
        source.args,
        "--served-model-name",
        served_name,
    )
    for flag, value in (
        ("--max-mamba-cache-size", "4"),
        ("--max-total-tokens", "65536"),
        ("--context-length", "65536"),
        ("--max-running-requests", "1"),
    ):
        args = _replace_arg(args, flag, value)
    updated = list(args)
    parser_index = updated.index("--reasoning-parser") + 2
    updated[parser_index:parser_index] = [
        "--tool-call-parser",
        "qwen3_coder",
    ]
    max_prefill_index = updated.index("--max-prefill-tokens")
    del updated[max_prefill_index : max_prefill_index + 2]
    graph_index = updated.index("--cuda-graph-bs-decode")
    graph_end = graph_index + 1
    while graph_end < len(updated) and not updated[graph_end].startswith("--"):
        graph_end += 1
    updated[graph_index + 1 : graph_end] = ["1"]
    return tuple(updated)


class Qwen38FlashNextAutoresearchProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = load_models(MODELS_PATH)

    def test_baseline_clones_mapped_ple_pins_and_bounds_c1_geometry(self) -> None:
        source = self.models[SOURCE_ID]
        baseline = self.models[BASELINE_ID]

        self.assertEqual(
            baseline,
            replace(
                source,
                id=baseline.id,
                description=baseline.description,
                served_name=baseline.served_name,
                tasks=("chat", "json", "thinking", "tools"),
                request_body_json=LOW_BODY,
                max_context=65536,
                args=_baseline_args_from_source(
                    source,
                    served_name=baseline.served_name,
                ),
            ),
        )

        self.assertEqual(baseline.tasks, ("chat", "json", "thinking", "tools"))
        self.assertEqual(baseline.request_body_json, LOW_BODY)
        self.assertEqual(
            json.loads(baseline.request_body_json),
            {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "reasoning_effort": "low",
                }
            },
        )
        self.assertEqual(baseline.max_context, 65536)
        self.assertEqual(baseline.native_context, 262144)
        expected_args = {
            "--reasoning-parser": "qwen3",
            "--tool-call-parser": "qwen3_coder",
            "--mamba-radix-cache-strategy": "extra_buffer_lazy",
            "--max-mamba-cache-size": "4",
            "--mem-fraction-static": "0.85",
            "--max-total-tokens": "65536",
            "--context-length": "65536",
            "--chunked-prefill-size": "1024",
            "--max-running-requests": "1",
            "--cuda-graph-backend-decode": "full",
            "--cuda-graph-bs-decode": "1",
            "--cuda-graph-backend-prefill": "disabled",
            "--speculative-algorithm": "NEXTN",
            "--speculative-num-steps": "2",
            "--speculative-eagle-topk": "1",
            "--speculative-num-draft-tokens": "3",
            "--speculative-draft-model-quantization": "unquant",
        }
        for flag, expected in expected_args.items():
            with self.subTest(flag=flag):
                self.assertEqual(_arg_value(baseline, flag), expected)
        self.assertNotIn("--max-prefill-tokens", baseline.args)
        graph_index = baseline.args.index("--cuda-graph-bs-decode")
        self.assertEqual(
            baseline.args[graph_index + 1 : graph_index + 2],
            ("1",),
        )
        self.assertEqual(
            baseline.args[graph_index + 2],
            "--cuda-graph-backend-prefill",
        )

    def test_candidates_change_only_their_named_axis(self) -> None:
        baseline = self.models[BASELINE_ID]
        mtp3 = self.models[MTP3_ID]
        chunk2k = self.models[CHUNK2K_ID]
        no_thinking = self.models[NONE_ID]

        self.assertEqual(
            mtp3,
            _candidate_from_baseline(
                baseline,
                mtp3,
                arg_changes=(
                    ("--speculative-num-steps", "3"),
                    ("--speculative-num-draft-tokens", "4"),
                ),
            ),
        )
        self.assertEqual(
            chunk2k,
            _candidate_from_baseline(
                baseline,
                chunk2k,
                arg_changes=(("--chunked-prefill-size", "2048"),),
            ),
        )
        self.assertEqual(
            no_thinking,
            _candidate_from_baseline(
                baseline,
                no_thinking,
                request_body_json=NONE_BODY,
            ),
        )
        self.assertEqual(
            json.loads(no_thinking.request_body_json or ""),
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_profile_and_suite_binding_rejects_unknown_or_wrong_selection(
        self,
    ) -> None:
        suite = load_suite(SUITE_PATH)
        native_suite = load_suite(NATIVE_SUITE_PATH)
        agentic_suite = load_suite(AGENTIC_SUITE_PATH)

        for profile_id in PROFILE_IDS:
            profile = self.models[profile_id]
            with self.subTest(profile=profile_id):
                validate_benchmark_selection(profile, suite)
            for wrong_suite in (native_suite, agentic_suite):
                with self.subTest(
                    profile=profile_id, wrong_suite=wrong_suite.id
                ), self.assertRaisesRegex(ManifestError, "profile requires"):
                    validate_benchmark_selection(profile, wrong_suite)

        unrelated = self.models[SOURCE_ID]
        with self.assertRaisesRegex(ManifestError, "suite requires"):
            validate_benchmark_selection(unrelated, suite)

        unknown_profile = replace(
            self.models[BASELINE_ID],
            id="qwen38-flash-next-nvfp4-unknown-agent64k-sglang",
        )
        with self.assertRaisesRegex(ManifestError, "suite requires"):
            validate_benchmark_selection(unknown_profile, suite)

        unknown_suite = replace(suite, id="unknown-agent64k-suite")
        with self.assertRaisesRegex(ManifestError, "profile requires"):
            validate_benchmark_selection(self.models[BASELINE_ID], unknown_suite)

    def test_suite_case_order_and_shapes_are_immutable(self) -> None:
        suite = load_suite(SUITE_PATH)
        agentic_suite = load_suite(AGENTIC_SUITE_PATH)
        cases = {case.id: case for case in suite.cases}

        self.assertEqual(
            suite.id,
            "qwen38-flash-next-sglang-agent64k-autoresearch",
        )
        self.assertIn("immutable", suite.description.lower())
        self.assertIn("single-user", suite.description.lower())
        self.assertIn("coding/cowork", suite.description.lower())
        self.assertEqual(
            tuple(case.id for case in suite.cases),
            (
                "json-smoke",
                "tools-smoke",
                "synthetic-exact-answer-v2",
                "agentic-select-and-call",
                "agentic-no-tool",
                "agentic-two-hop",
                "agentic-tool-error-recovery",
                "long-context-needle-60000-agent-c1",
                "agent64k-decode-256-c1-v1",
            ),
        )
        self.assertEqual(suite.cases[3:7], agentic_suite.cases)
        self.assertTrue(all(case.concurrency == 1 for case in suite.cases))
        self.assertTrue(all(case.temperature == 0.0 for case in suite.cases))

        json_smoke = cases["json-smoke"]
        self.assertEqual(
            (
                json_smoke.kind,
                json_smoke.requires,
                json_smoke.warmups,
                json_smoke.repetitions,
                json_smoke.max_output_tokens,
            ),
            ("capability", ("chat", "json"), 0, 1, 64),
        )
        tools_smoke = cases["tools-smoke"]
        self.assertEqual(
            (
                tools_smoke.kind,
                tools_smoke.requires,
                tools_smoke.warmups,
                tools_smoke.repetitions,
                tools_smoke.max_output_tokens,
            ),
            ("capability", ("chat", "tools"), 0, 1, 64),
        )
        quality = cases["synthetic-exact-answer-v2"]
        self.assertEqual(
            (
                quality.kind,
                quality.requires,
                quality.warmups,
                quality.repetitions,
                quality.max_output_tokens,
            ),
            ("quality", ("chat",), 0, 2, 512),
        )
        needle = cases["long-context-needle-60000-agent-c1"]
        self.assertEqual(
            (
                needle.kind,
                needle.requires,
                needle.warmups,
                needle.repetitions,
                needle.max_output_tokens,
                needle.prompt_repetitions,
            ),
            ("capability", ("chat",), 0, 1, 128, 60000),
        )
        decode = cases["agent64k-decode-256-c1-v1"]
        self.assertEqual(
            (
                decode.kind,
                decode.requires,
                decode.warmups,
                decode.repetitions,
                decode.max_output_tokens,
                decode.prompt_repetitions,
            ),
            ("decode", ("chat",), 1, 5, 256, 0),
        )

    def test_every_case_fits_each_profile_context_admission(self) -> None:
        suite = load_suite(SUITE_PATH)

        for profile_id in PROFILE_IDS:
            profile = self.models[profile_id]
            for case in suite.cases:
                estimated_tokens, basis = _estimated_context_tokens(case)
                with self.subTest(
                    profile=profile_id,
                    case=case.id,
                    basis=basis,
                ):
                    self.assertLess(estimated_tokens, profile.max_context)


if __name__ == "__main__":
    unittest.main()
