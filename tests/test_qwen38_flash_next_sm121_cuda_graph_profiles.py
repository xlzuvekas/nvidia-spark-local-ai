from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.client import RequestResult
from bench.execution_admission import model_execution_blocker
from bench.journal import Journal
from bench.manifest import (
    MATCHED_REQUEST_UNIQUE_PROTOCOL,
    ManifestError,
    ModelSpec,
    load_models,
    load_suite,
    matched_prompt_protocol,
    model_spec_to_dict,
    validate_benchmark_selection,
    validate_model,
    validate_suite,
)
from bench.runner import (
    PreflightError,
    _execute_case,
    _require_frozen_matched_prompt_suite,
)
from bench.sglang_sm121_cuda_graph import (
    SM121_CUDA_GRAPH_BREAKABLE_DESCRIPTION,
    SM121_CUDA_GRAPH_BREAKABLE_PROFILE_ID,
    SM121_CUDA_GRAPH_CASE_ID,
    SM121_CUDA_GRAPH_DISABLED_DESCRIPTION,
    SM121_CUDA_GRAPH_DISABLED_PROFILE_ID,
    SM121_CUDA_GRAPH_SUITE_ID,
    is_sm121_cuda_graph_candidate,
    sm121_cuda_graph_args,
    validate_sm121_cuda_graph_candidate,
    validate_sm121_cuda_graph_suite,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = ROOT / "manifests" / "models.toml"
SUITE_PATH = (
    ROOT
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sm121_triton_storage_c1_cuda_graph.toml"
)
SOURCE_ID = (
    "qwen38-flash-next-nvfp4-sm121-triton-storage-agent-admission-sglang"
)
QWEN38_27B_GRAPH_FULL_ID = (
    "qwen38-27b-nvfp4-dspark-c1-cuda-graph-full-sglang"
)


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
        response_model="qwen3.8-flash-next",
        content="synthetic result",
        reasoning="",
        tool_calls=[],
    )


def _arg_value(model: ModelSpec, flag: str) -> str:
    indexes = [
        index for index, argument in enumerate(model.args) if argument == flag
    ]
    if len(indexes) != 1:
        raise AssertionError(
            f"expected exactly one {flag!r} in {model.id}, found {len(indexes)}"
        )
    index = indexes[0]
    if index + 1 >= len(model.args):
        raise AssertionError(f"{flag!r} has no value in {model.id}")
    return model.args[index + 1]


class Qwen38FlashNextSM121CudaGraphProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = load_models(MODELS_PATH)
        cls.source = cls.models[SOURCE_ID]
        cls.breakable = cls.models[SM121_CUDA_GRAPH_BREAKABLE_PROFILE_ID]
        cls.disabled = cls.models[SM121_CUDA_GRAPH_DISABLED_PROFILE_ID]
        cls.suite = load_suite(SUITE_PATH)

    def test_profiles_are_exact_prospective_source_clones(self) -> None:
        self.assertEqual(
            self.breakable,
            replace(
                self.source,
                id=SM121_CUDA_GRAPH_BREAKABLE_PROFILE_ID,
                description=SM121_CUDA_GRAPH_BREAKABLE_DESCRIPTION,
                args=sm121_cuda_graph_args("breakable"),
            ),
        )
        self.assertEqual(
            self.disabled,
            replace(
                self.source,
                id=SM121_CUDA_GRAPH_DISABLED_PROFILE_ID,
                description=SM121_CUDA_GRAPH_DISABLED_DESCRIPTION,
                args=sm121_cuda_graph_args("disabled"),
            ),
        )
        for model in (self.breakable, self.disabled):
            self.assertTrue(is_sm121_cuda_graph_candidate(model))
            validate_sm121_cuda_graph_candidate(model)

    def test_arms_have_one_decode_backend_delta_and_full_is_prohibited(self) -> None:
        expected_args = list(self.breakable.args)
        backend_index = expected_args.index("--cuda-graph-backend-decode") + 1
        expected_args[backend_index] = "disabled"
        self.assertEqual(
            self.disabled,
            replace(
                self.breakable,
                id=SM121_CUDA_GRAPH_DISABLED_PROFILE_ID,
                description=SM121_CUDA_GRAPH_DISABLED_DESCRIPTION,
                args=tuple(expected_args),
            ),
        )
        differences = [
            (index, candidate, control)
            for index, (candidate, control) in enumerate(
                zip(self.breakable.args, self.disabled.args, strict=True)
            )
            if candidate != control
        ]
        self.assertEqual(differences, [(backend_index, "breakable", "disabled")])

        for model, backend in (
            (self.breakable, "breakable"),
            (self.disabled, "disabled"),
        ):
            with self.subTest(model=model.id):
                self.assertEqual(
                    _arg_value(model, "--cuda-graph-backend-decode"), backend
                )
                self.assertEqual(_arg_value(model, "--cuda-graph-bs-decode"), "1")
                self.assertEqual(
                    _arg_value(model, "--cuda-graph-backend-prefill"), "disabled"
                )
                self.assertEqual(model.args.count("--enable-metrics"), 1)
                self.assertNotIn("full", model.args)

        full_args = list(self.breakable.args)
        full_args[backend_index] = "full"
        with self.assertRaisesRegex(ManifestError, "full CUDA graphs are prohibited"):
            validate_model(replace(self.breakable, args=tuple(full_args)))

    def test_source_runtime_and_ple_pin_drift_is_rejected(self) -> None:
        mutations = (
            {"source": "example/changed"},
            {"revision": "0" * 40},
            {"image": "local/sglang:changed"},
            {"local_image_id": "sha256:" + "0" * 64},
            {"quantization": "changed"},
            {"sglang_ple_nvme_queue_depth": 256},
            {"sglang_ple_nvme_max_batch_pages": 2048},
            {"sglang_ple_nvme_cache_pages": 1},
            {"request_body_json": '{"chat_template_kwargs":{"enable_thinking":false}}'},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(
                    ManifestError, "changed beyond its graph bundle"
                ):
                    validate_model(replace(self.breakable, **mutation))

    def test_suite_is_exact_minimal_c1_d256_screen_and_pair_bound(self) -> None:
        validate_sm121_cuda_graph_suite(self.suite)
        self.assertEqual(self.suite.id, SM121_CUDA_GRAPH_SUITE_ID)
        self.assertEqual(len(self.suite.cases), 1)
        case = self.suite.cases[0]
        self.assertEqual(case.id, SM121_CUDA_GRAPH_CASE_ID)
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

        validate_benchmark_selection(self.breakable, self.suite)
        validate_benchmark_selection(self.disabled, self.suite)
        smoke = load_suite(ROOT / "manifests" / "suites" / "smoke.toml")
        with self.assertRaisesRegex(ManifestError, "requires"):
            validate_benchmark_selection(self.breakable, smoke)
        with self.assertRaisesRegex(ManifestError, "exact paired"):
            validate_benchmark_selection(
                replace(self.breakable, id="unrelated-sm121-profile"), self.suite
            )

    def test_reserved_schedule_and_pair_selection_fail_closed(self) -> None:
        case = self.suite.cases[0]
        with self.assertRaisesRegex(ManifestError, "matched-prompt IDs"):
            validate_suite(
                replace(
                    self.suite,
                    cases=(replace(case, id="matched-prompt-unreviewed-v2"),),
                )
            )
        with self.assertRaisesRegex(ManifestError, "max_output_tokens"):
            validate_suite(
                replace(
                    self.suite,
                    cases=(replace(case, max_output_tokens=255),),
                )
            )

        qwen38_27b_suite = load_suite(
            ROOT
            / "manifests"
            / "suites"
            / "qwen38_27b_dspark_c1_cuda_graph.toml"
        )
        with self.assertRaisesRegex(ManifestError, "requires"):
            validate_benchmark_selection(self.breakable, qwen38_27b_suite)
        with self.assertRaisesRegex(ManifestError, "requires"):
            validate_benchmark_selection(
                self.models[QWEN38_27B_GRAPH_FULL_ID], self.suite
            )

        frozen_suite = asdict(self.suite)
        frozen_suite.pop("protocol_digest", None)
        frozen_suite["cases"] = [
            {**frozen_suite["cases"][0], "case_id": f"{case.id}--0123456789ab"}
        ]
        frozen_model = SimpleNamespace(
            **json.loads(json.dumps(model_spec_to_dict(self.breakable)))
        )
        _require_frozen_matched_prompt_suite(frozen_model, frozen_suite)
        tampered_model = SimpleNamespace(**vars(frozen_model))
        backend_index = tampered_model.args.index(
            "--cuda-graph-backend-decode"
        ) + 1
        tampered_model.args[backend_index] = "disabled"
        with self.assertRaisesRegex(PreflightError, "contract is invalid"):
            _require_frozen_matched_prompt_suite(tampered_model, frozen_suite)
        stale_suite = {
            **frozen_suite,
            "cases": [{**frozen_suite["cases"][0], "id": "decode-256-c1"}],
        }
        with self.assertRaisesRegex(PreflightError, "contract is invalid"):
            _require_frozen_matched_prompt_suite(
                frozen_model,
                stale_suite,
            )
        with self.assertRaisesRegex(PreflightError, "contract is invalid"):
            _require_frozen_matched_prompt_suite(
                SimpleNamespace(
                    **json.loads(
                        json.dumps(
                            model_spec_to_dict(
                                self.models[QWEN38_27B_GRAPH_FULL_ID]
                            )
                        )
                    )
                ),
                frozen_suite,
            )

    def test_prompt_bytes_match_across_arms_and_remain_unique_within_arm(
        self,
    ) -> None:
        case = self.suite.cases[0]
        captured: dict[str, list[dict[str, object]]] = {}

        with tempfile.TemporaryDirectory() as directory:
            for arm in ("breakable", "disabled"):
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
                    captured[arm].extend(dict(request) for request in requests)
                    return (
                        [
                            _request_result(str(request["request_id"]))
                            for request in requests
                        ],
                        1.0,
                    )

                captured[arm] = requests
                with patch(
                    "bench.runner.stream_chat_request", side_effect=warmup
                ), patch(
                    "bench.runner.concurrent_chat_requests", side_effect=measured
                ):
                    _execute_case(
                        server=SimpleNamespace(
                            backend="sglang",
                            base_url="http://127.0.0.1:30000/v1",
                        ),
                        model=SimpleNamespace(
                            served_name="qwen3.8-flash-next",
                            max_context=262_144,
                        ),
                        case=SimpleNamespace(
                            **asdict(case),
                            case_id=f"{case.id}--{arm}",
                        ),
                        journal=Journal(Path(directory) / f"{arm}.jsonl"),
                        telemetry=Mock(),
                    )

        candidate_ids = [
            str(request["request_id"]) for request in captured["breakable"]
        ]
        control_ids = [
            str(request["request_id"]) for request in captured["disabled"]
        ]
        candidate_prompt_bytes = [
            str(request["prompt"]).encode("utf-8")
            for request in captured["breakable"]
        ]
        control_prompt_bytes = [
            str(request["prompt"]).encode("utf-8")
            for request in captured["disabled"]
        ]
        self.assertEqual(len(candidate_ids), 6)
        self.assertEqual(candidate_ids, control_ids)
        self.assertEqual(candidate_prompt_bytes, control_prompt_bytes)
        self.assertEqual(len(set(candidate_ids)), 6)
        self.assertEqual(len(set(candidate_prompt_bytes)), 6)
        self.assertEqual(
            candidate_ids,
            [f"warmup-{case.id}-0"]
            + [f"{case.id}-r{index}-w0" for index in range(5)],
        )
        self.assertTrue(
            all(
                prompt.startswith(f"Benchmark nonce {request_id}. ".encode())
                for prompt, request_id in zip(
                    candidate_prompt_bytes, candidate_ids, strict=True
                )
            )
        )
        self.assertTrue(
            all(
                b"breakable" not in prompt and b"disabled" not in prompt
                for prompt in candidate_prompt_bytes
            )
        )

    def test_static_recognition_does_not_broaden_runtime_authority(self) -> None:
        for model in (self.breakable, self.disabled):
            with self.subTest(model=model.id):
                self.assertIn(
                    "dedicated sm121-storage-canary command",
                    model_execution_blocker(model) or "",
                )


if __name__ == "__main__":
    unittest.main()
