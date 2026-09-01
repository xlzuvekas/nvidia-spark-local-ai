from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.client import RequestResult
from bench.journal import Journal
from bench.manifest import (
    MATCHED_REQUEST_UNIQUE_PROTOCOL,
    QWEN38_27B_DSPARK_CUDA_GRAPH_CASE_ID,
    QWEN38_27B_DSPARK_CUDA_GRAPH_DISABLED_DESCRIPTION,
    QWEN38_27B_DSPARK_CUDA_GRAPH_DISABLED_PROFILE_ID,
    QWEN38_27B_DSPARK_CUDA_GRAPH_FULL_DESCRIPTION,
    QWEN38_27B_DSPARK_CUDA_GRAPH_FULL_PROFILE_ID,
    QWEN38_27B_DSPARK_CUDA_GRAPH_SUITE_ID,
    QWEN38_27B_DSPARK_SOURCE_PROFILE_ID,
    ManifestError,
    ModelSpec,
    load_models,
    load_suite,
    matched_prompt_protocol,
    model_spec_to_dict,
    validate_benchmark_selection,
    validate_model,
    validate_qwen38_27b_dspark_cuda_graph_candidate,
    validate_suite,
)
from bench.runner import (
    PreflightError,
    _execute_case,
    _require_frozen_matched_prompt_suite,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = ROOT / "manifests" / "models.toml"
SUITE_PATH = (
    ROOT
    / "manifests"
    / "suites"
    / "qwen38_27b_dspark_c1_cuda_graph.toml"
)

SOURCE_ID = QWEN38_27B_DSPARK_SOURCE_PROFILE_ID
FULL_ID = QWEN38_27B_DSPARK_CUDA_GRAPH_FULL_PROFILE_ID
DISABLED_ID = QWEN38_27B_DSPARK_CUDA_GRAPH_DISABLED_PROFILE_ID
FULL_DESCRIPTION = QWEN38_27B_DSPARK_CUDA_GRAPH_FULL_DESCRIPTION
DISABLED_DESCRIPTION = QWEN38_27B_DSPARK_CUDA_GRAPH_DISABLED_DESCRIPTION


def _request_result(request_id: str) -> RequestResult:
    return RequestResult(
        request_id=request_id,
        started_at_ns=1,
        prompt_tokens=32,
        completion_tokens=256,
        reasoning_tokens=None,
        ttft_s=0.01,
        elapsed_s=1.0,
        decode_s=0.99,
        decode_tps=258.0,
        output_tps=256.0,
        emission_events=256,
        finish_reason="length",
        response_model="qwen3.8-27b",
        content="synthetic result",
        reasoning="",
        tool_calls=[],
    )


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


def _arg_list(model: ModelSpec, flag: str) -> tuple[str, ...]:
    start = model.args.index(flag) + 1
    end = start
    while end < len(model.args) and not model.args[end].startswith("--"):
        end += 1
    return model.args[start:end]


def _c1_graph_args(source: ModelSpec, backend: str) -> tuple[str, ...]:
    args = list(source.args)
    graph_index = args.index("--disable-prefill-cuda-graph")
    if args[graph_index : graph_index + 3] != [
        "--disable-prefill-cuda-graph",
        "--cuda-graph-max-bs",
        "4",
    ]:
        raise AssertionError("source graph bundle no longer matches the pinned recipe")
    args[graph_index : graph_index + 3] = [
        "--cuda-graph-backend-decode",
        backend,
        "--cuda-graph-bs-decode",
        "1",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--enable-metrics",
    ]
    return tuple(args)


class Qwen3827BDsparkCudaGraphProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = load_models(MODELS_PATH)
        cls.source = cls.models[SOURCE_ID]
        cls.full = cls.models[FULL_ID]
        cls.disabled = cls.models[DISABLED_ID]

    def test_profiles_are_exact_source_clones_with_only_the_graph_bundle_changed(
        self,
    ) -> None:
        self.assertEqual(
            self.full,
            replace(
                self.source,
                id=FULL_ID,
                description=FULL_DESCRIPTION,
                args=_c1_graph_args(self.source, "full"),
            ),
        )
        self.assertEqual(
            self.disabled,
            replace(
                self.source,
                id=DISABLED_ID,
                description=DISABLED_DESCRIPTION,
                args=_c1_graph_args(self.source, "disabled"),
            ),
        )

    def test_arms_differ_only_by_decode_graph_backend(self) -> None:
        expected_args = list(self.full.args)
        backend_index = expected_args.index("--cuda-graph-backend-decode") + 1
        expected_args[backend_index] = "disabled"
        self.assertEqual(
            self.disabled,
            replace(
                self.full,
                id=DISABLED_ID,
                description=DISABLED_DESCRIPTION,
                args=tuple(expected_args),
            ),
        )

        differences = [
            (index, full_arg, disabled_arg)
            for index, (full_arg, disabled_arg) in enumerate(
                zip(self.full.args, self.disabled.args, strict=True)
            )
            if full_arg != disabled_arg
        ]
        self.assertEqual(differences, [(backend_index, "full", "disabled")])

    def test_exact_profile_contract_rejects_argument_and_provenance_drift(self) -> None:
        validate_qwen38_27b_dspark_cuda_graph_candidate(self.full)
        mutations = (
            {"args": (*self.full.args, "--cuda-graph-max-bs", "8")},
            {"source": "example/changed"},
            {"revision": "0" * 40},
            {"image": "example/changed:latest"},
            {"description": "changed"},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                ManifestError, "changed beyond its exact graph bundle"
            ):
                validate_model(replace(self.full, **mutation))

    def test_both_arms_have_explicit_matched_c1_graph_geometry(self) -> None:
        for model, backend in (
            (self.full, "full"),
            (self.disabled, "disabled"),
        ):
            with self.subTest(model=model.id):
                self.assertEqual(
                    _arg_value(model, "--cuda-graph-backend-decode"), backend
                )
                self.assertEqual(_arg_list(model, "--cuda-graph-bs-decode"), ("1",))
                self.assertEqual(
                    _arg_value(model, "--cuda-graph-backend-prefill"), "disabled"
                )
                self.assertEqual(model.args.count("--enable-metrics"), 1)
                self.assertNotIn("--disable-prefill-cuda-graph", model.args)
                self.assertNotIn("--cuda-graph-max-bs", model.args)
                self.assertEqual(_arg_value(model, "--max-running-requests"), "8")
                self.assertEqual(_arg_value(model, "--torch-compile-max-bs"), "4")

    def test_screening_suite_is_one_deterministic_c1_d256_case(self) -> None:
        suite = load_suite(SUITE_PATH)

        self.assertEqual(suite.id, QWEN38_27B_DSPARK_CUDA_GRAPH_SUITE_ID)
        self.assertIn("D256", suite.description)
        self.assertEqual(len(suite.cases), 1)
        case = suite.cases[0]
        self.assertEqual(case.id, QWEN38_27B_DSPARK_CUDA_GRAPH_CASE_ID)
        self.assertEqual(
            matched_prompt_protocol(case.id), MATCHED_REQUEST_UNIQUE_PROTOCOL
        )
        self.assertEqual(case.kind, "decode")
        self.assertEqual(case.requires, ("chat",))
        self.assertEqual(case.warmups, 1)
        self.assertEqual(case.repetitions, 5)
        self.assertEqual(case.max_output_tokens, 256)
        self.assertEqual(case.temperature, 0.0)
        self.assertEqual(case.concurrency, 1)
        self.assertEqual(case.prompt_repetitions, 0)

        validate_benchmark_selection(self.full, suite)
        validate_benchmark_selection(self.disabled, suite)

    def test_reserved_schedule_and_pair_selection_fail_closed(self) -> None:
        suite = load_suite(SUITE_PATH)
        case = suite.cases[0]
        with self.assertRaisesRegex(ManifestError, "matched-prompt IDs"):
            validate_suite(
                replace(
                    suite,
                    cases=(replace(case, id="matched-prompt-unreviewed-v2"),),
                )
            )
        with self.assertRaisesRegex(ManifestError, "max_output_tokens"):
            validate_suite(
                replace(
                    suite,
                    cases=(replace(case, max_output_tokens=255),),
                )
            )
        with self.assertRaisesRegex(ManifestError, "requires one of its exact"):
            validate_benchmark_selection(self.source, suite)
        with self.assertRaisesRegex(ManifestError, "requires"):
            validate_benchmark_selection(
                self.full,
                replace(suite, id="unrelated-screen"),
            )

        frozen_suite = asdict(suite)
        frozen_suite.pop("protocol_digest", None)
        frozen_suite["cases"] = [
            {**frozen_suite["cases"][0], "case_id": f"{case.id}--0123456789ab"}
        ]
        frozen_model = SimpleNamespace(
            **json.loads(json.dumps(model_spec_to_dict(self.full)))
        )
        _require_frozen_matched_prompt_suite(frozen_model, frozen_suite)
        tampered_model = SimpleNamespace(**vars(frozen_model))
        tampered_model.args = [*tampered_model.args, "--cuda-graph-max-bs", "8"]
        with self.assertRaisesRegex(PreflightError, "contract is invalid"):
            _require_frozen_matched_prompt_suite(tampered_model, frozen_suite)
        stale_suite = {
            **frozen_suite,
            "cases": [{**frozen_suite["cases"][0], "id": "decode-256-c1"}],
        }
        with self.assertRaisesRegex(PreflightError, "contract is invalid"):
            _require_frozen_matched_prompt_suite(
                frozen_model, stale_suite
            )

    def test_prompt_bytes_match_across_arms_and_remain_unique_within_arm(
        self,
    ) -> None:
        case = load_suite(SUITE_PATH).cases[0]
        captured: dict[str, list[dict[str, object]]] = {}

        with tempfile.TemporaryDirectory() as directory:
            for arm in ("full", "disabled"):
                requests: list[dict[str, object]] = []

                def warmup(**kwargs: object) -> RequestResult:
                    requests.append(dict(kwargs))
                    return _request_result(str(kwargs["request_id"]))

                def measured(
                    *,
                    requests: list[dict[str, object]],
                    concurrency: int,
                    request_function: object,
                ) -> tuple[list[RequestResult], float]:
                    del request_function
                    self.assertEqual(concurrency, 1)
                    captured_requests = [dict(request) for request in requests]
                    captured[arm].extend(captured_requests)
                    return (
                        [
                            _request_result(str(request["request_id"]))
                            for request in requests
                        ],
                        1.0,
                    )

                captured[arm] = requests
                with patch("bench.runner.stream_chat_request", side_effect=warmup), patch(
                    "bench.runner.concurrent_chat_requests", side_effect=measured
                ):
                    _execute_case(
                        server=SimpleNamespace(
                            backend="sglang",
                            base_url="http://127.0.0.1:30000/v1",
                        ),
                        model=SimpleNamespace(
                            served_name="qwen3.8-27b", max_context=262_144
                        ),
                        case=SimpleNamespace(
                            **asdict(case),
                            case_id=f"{case.id}--{arm}",
                        ),
                        journal=Journal(Path(directory) / f"{arm}.jsonl"),
                        telemetry=Mock(),
                    )

        full_ids = [str(request["request_id"]) for request in captured["full"]]
        disabled_ids = [
            str(request["request_id"]) for request in captured["disabled"]
        ]
        full_prompts = [str(request["prompt"]) for request in captured["full"]]
        disabled_prompts = [
            str(request["prompt"]) for request in captured["disabled"]
        ]
        self.assertEqual(len(full_ids), 6)
        self.assertEqual(full_ids, disabled_ids)
        self.assertEqual(full_prompts, disabled_prompts)
        self.assertEqual(len(set(full_ids)), 6)
        self.assertEqual(len(set(full_prompts)), 6)
        self.assertEqual(
            full_ids,
            [f"warmup-{case.id}-0"]
            + [f"{case.id}-r{index}-w0" for index in range(5)],
        )
        self.assertTrue(
            all(
                prompt.startswith(f"Benchmark nonce {request_id}. ")
                for prompt, request_id in zip(full_prompts, full_ids, strict=True)
            )
        )
        self.assertTrue(
            all("--full" not in prompt and "--disabled" not in prompt for prompt in full_prompts)
        )


if __name__ == "__main__":
    unittest.main()
