from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import textwrap
import unittest

from bench.manifest import (
    CaseSpec,
    ManifestError,
    ModelSpec,
    load_models,
    load_suite,
    validate_case,
    validate_model,
    validate_models,
)


ROOT = Path(__file__).resolve().parents[1]


class ManifestLoaderTests(unittest.TestCase):
    def _write(self, body: str) -> Path:
        temporary = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".toml", delete=False
        )
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        with temporary:
            temporary.write(textwrap.dedent(body))
        return Path(temporary.name)

    def test_repository_manifests_load(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        smoke = load_suite(ROOT / "manifests" / "suites" / "smoke.toml")
        core = load_suite(ROOT / "manifests" / "suites" / "core.toml")
        multimodal = load_suite(
            ROOT / "manifests" / "suites" / "multimodal_embeddings.toml"
        )
        capabilities = load_suite(
            ROOT / "manifests" / "suites" / "capabilities.toml"
        )
        chat_quality = load_suite(
            ROOT / "manifests" / "suites" / "chat_quality.toml"
        )
        multimodal_rerank = load_suite(
            ROOT / "manifests" / "suites" / "multimodal_rerank.toml"
        )

        self.assertGreaterEqual(len(models), 1)
        self.assertEqual(smoke.id, "smoke")
        self.assertEqual(core.id, "core")
        self.assertEqual(len(core.cases), len({case.id for case in core.cases}))
        self.assertEqual(multimodal.id, "multimodal-embeddings")
        self.assertEqual(multimodal.cases[0].requires, ("embeddings", "vision"))
        self.assertEqual(capabilities.id, "capabilities")
        self.assertEqual(chat_quality.id, "chat-quality")
        self.assertEqual(multimodal_rerank.id, "multimodal-rerank")
        self.assertEqual(
            multimodal_rerank.cases[0].requires, ("rerank", "vision")
        )

    def test_capabilities_suite_is_lightweight_and_uses_deterministic_oracles(
        self,
    ) -> None:
        suite = load_suite(ROOT / "manifests" / "suites" / "capabilities.toml")
        cases = {case.id: case for case in suite.cases}

        self.assertEqual(
            set(cases),
            {
                "json-correctness",
                "tool-call-correctness",
                "long-context-needle-4096",
            },
        )
        self.assertTrue(all(case.kind == "capability" for case in cases.values()))
        self.assertTrue(all(case.warmups == 0 for case in cases.values()))
        self.assertTrue(all(case.repetitions == 1 for case in cases.values()))
        self.assertTrue(all(case.concurrency == 1 for case in cases.values()))
        self.assertEqual(cases["json-correctness"].requires, ("chat", "json"))
        self.assertEqual(
            cases["tool-call-correctness"].requires, ("chat", "tools")
        )
        needle = cases["long-context-needle-4096"]
        self.assertEqual(needle.requires, ("chat",))
        self.assertEqual(needle.prompt_repetitions, 4096)
        self.assertEqual(needle.max_output_tokens, 32)
        self.assertNotIn(
            "thinking", {requirement for case in suite.cases for requirement in case.requires}
        )
        self.assertIn("reasoning", suite.description.lower())
        self.assertIn("excluded", suite.description.lower())

    def test_qwen3_vl_pooling_profiles_are_pinned_and_conservative(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        embedding = models["qwen3-vl-embedding-2b-bf16"]
        reranker = models["qwen3-vl-reranker-2b-bf16"]
        expected_image = "nvcr.io/nvidia/vllm:26.07-py3"
        expected_digest = (
            "sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268"
        )

        for model in (embedding, reranker):
            with self.subTest(model=model.id):
                self.assertEqual(model.image, expected_image)
                self.assertEqual(model.image_digest, expected_digest)
                self.assertEqual(model.cache_dir, "user")
                self.assertEqual(model.max_context, 8192)
                self.assertEqual(model.native_context, 32768)
                self.assertEqual(model.estimated_ram_gib, 24.0)
                self.assertIn("--runner", model.args)
                self.assertEqual(model.args[model.args.index("--runner") + 1], "pooling")
                self.assertIn("--trust-remote-code", model.args)
                self.assertEqual(
                    model.args[model.args.index("--max-model-len") + 1], "8192"
                )
                self.assertEqual(
                    model.args[model.args.index("--gpu-memory-utilization") + 1],
                    "0.30",
                )

        self.assertEqual(
            embedding.revision, "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
        )
        self.assertEqual(embedding.tasks, ("embeddings", "vision"))
        self.assertEqual(
            reranker.revision, "4bd860ac4f15ad1897a214615cccc700f8f71818"
        )
        self.assertEqual(reranker.tasks, ("rerank", "vision"))
        overrides = json.loads(
            reranker.args[reranker.args.index("--hf-overrides") + 1]
        )
        self.assertEqual(
            overrides,
            {
                "architectures": ["Qwen3VLForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
            },
        )
        self.assertEqual(
            reranker.args[reranker.args.index("--chat-template") + 1],
            "/opt/vllm/vllm-src/examples/pooling/score/template/qwen3_vl_reranker.jinja",
        )

    def test_next_official_download_profiles_are_exactly_pinned(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        expected_image = "nvcr.io/nvidia/vllm:26.07-py3"
        expected_digest = (
            "sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268"
        )
        expected = {
            "qwen36-35b-a3b-nvfp4-mtp3": {
                "source": "nvidia/Qwen3.6-35B-A3B-NVFP4",
                "revision": "491c2f1ea524c639598bf8fa787a93fed5a6fbce",
                "support_status": "spark_vllm_recipe",
                "quantization": "nvfp4+mtp3",
                "native_context": 262144,
                "reasoning_parser": "qwen3",
                "tool_parser": "qwen3_coder",
            },
            "gpt-oss-20b-mxfp4": {
                "source": "openai/gpt-oss-20b",
                "revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
                "support_status": "spark_vllm_matrix",
                "quantization": "mxfp4",
                "native_context": 131072,
                "reasoning_parser": "openai_gptoss",
                "tool_parser": "openai",
            },
            "qwen3-8b-nvfp4": {
                "source": "nvidia/Qwen3-8B-NVFP4",
                "revision": "ccd10a893cbca613259517c3efe08e151ddf2b8e",
                "support_status": "spark_vllm_matrix",
                "quantization": "nvfp4",
                "native_context": 131072,
                "reasoning_parser": "qwen3",
                "tool_parser": "hermes",
            },
            "qwen3-8b-fp8": {
                "source": "nvidia/Qwen3-8B-FP8",
                "revision": "2cebc4c89e25abc17668c81b01dceaf3d8b914d5",
                "support_status": "spark_vllm_matrix",
                "quantization": "fp8",
                "native_context": 131072,
                "reasoning_parser": "qwen3",
                "tool_parser": "hermes",
            },
        }

        for model_id, fields in expected.items():
            model = models[model_id]
            with self.subTest(model=model_id):
                self.assertEqual(model.source, fields["source"])
                self.assertEqual(model.served_name, fields["source"])
                self.assertEqual(model.revision, fields["revision"])
                self.assertEqual(model.support_status, fields["support_status"])
                self.assertEqual(model.quantization, fields["quantization"])
                self.assertEqual(model.image, expected_image)
                self.assertEqual(model.image_digest, expected_digest)
                self.assertEqual(model.cache_dir, "user")
                self.assertEqual(model.max_context, 32768)
                self.assertEqual(model.native_context, fields["native_context"])
                self.assertEqual(
                    model.args[model.args.index("--max-model-len") + 1], "32768"
                )
                self.assertEqual(
                    model.args[model.args.index("--reasoning-parser") + 1],
                    fields["reasoning_parser"],
                )
                self.assertEqual(
                    model.args[model.args.index("--tool-call-parser") + 1],
                    fields["tool_parser"],
                )
                self.assertIn("--enable-auto-tool-choice", model.args)
                self.assertIn("--enable-chunked-prefill", model.args)
                self.assertIn("--enable-prefix-caching", model.args)

        qwen36 = models["qwen36-35b-a3b-nvfp4-mtp3"]
        self.assertEqual(
            set(qwen36.tasks), {"chat", "json", "vision", "tools", "thinking"}
        )
        self.assertIn("--trust-remote-code", qwen36.args)
        self.assertEqual(qwen36.args[qwen36.args.index("--moe-backend") + 1], "marlin")
        self.assertEqual(
            json.loads(qwen36.args[qwen36.args.index("--speculative-config") + 1]),
            {
                "method": "mtp",
                "num_speculative_tokens": 3,
                "moe_backend": "triton",
            },
        )
        gpt_oss = models["gpt-oss-20b-mxfp4"]
        self.assertEqual(json.loads(gpt_oss.request_body_json or "{}"), {"reasoning_effort": "low"})
        for model_id in ("qwen3-8b-nvfp4", "qwen3-8b-fp8"):
            model = models[model_id]
            self.assertEqual(model.args[model.args.index("--kv-cache-dtype") + 1], "fp8")

    def test_phi4_multimodal_profile_is_pinned_and_does_not_overclaim(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        model = models["phi-4-multimodal-instruct-nvfp4"]

        self.assertEqual(
            model.source, "nvidia/Phi-4-multimodal-instruct-NVFP4"
        )
        self.assertEqual(model.served_name, model.source)
        self.assertEqual(
            model.revision, "617cfabb9ad6c2c6e318fd21c1961536b84f65a1"
        )
        self.assertEqual(model.tasks, ("chat", "vision"))
        self.assertEqual(model.backend, "sglang")
        self.assertEqual(model.support_status, "spark_other_backend")
        self.assertEqual(model.quantization, "nvfp4")
        self.assertEqual(model.cache_dir, "user")
        self.assertEqual(model.max_context, 32768)
        self.assertEqual(model.native_context, 131072)
        self.assertEqual(model.estimated_ram_gib, 40.0)
        self.assertEqual(model.endpoint, "http://127.0.0.1:30000/v1")
        self.assertEqual(model.image, "scitrera/dgx-spark-sglang:0.5.10rc0")
        self.assertEqual(
            model.image_digest,
            "sha256:3f51e3b127bd0fe8f261a84c6ad54ce42bdb65eb2e57e228a9f6359e89bd08ec",
        )
        self.assertIn("--trust-remote-code", model.args)
        self.assertEqual(
            model.args[model.args.index("--quantization") + 1], "modelopt_fp4"
        )
        self.assertEqual(
            model.args[model.args.index("--attention-backend") + 1],
            "flashinfer",
        )
        self.assertEqual(
            model.args[model.args.index("--context-length") + 1], "32768"
        )
        self.assertEqual(
            model.args[model.args.index("--mem-fraction-static") + 1],
            "0.50",
        )
        self.assertEqual(
            model.args[model.args.index("--max-running-requests") + 1], "8"
        )
        self.assertEqual(
            model.args[model.args.index("--chunked-prefill-size") + 1],
            "8192",
        )
        self.assertIn("--enable-multimodal", model.args)
        for unverified_capability in ("audio", "json", "thinking", "tools"):
            self.assertNotIn(unverified_capability, model.tasks)

    def test_phi4_reasoning_fp8_profile_is_pinned_and_conservative(self) -> None:
        model = load_models(ROOT / "manifests" / "models.toml")[
            "phi-4-reasoning-plus-fp8"
        ]

        self.assertEqual(model.source, "nvidia/Phi-4-reasoning-plus-FP8")
        self.assertEqual(
            model.revision, "18abf8a59bd8ff0b79ec712863a153becc6cdaeb"
        )
        self.assertEqual(model.tasks, ("chat", "thinking"))
        self.assertEqual(model.support_status, "spark_vllm_matrix")
        self.assertEqual(model.backend, "vllm")
        self.assertEqual(model.architecture, "dense-reasoning")
        self.assertEqual(model.quantization, "fp8")
        self.assertEqual(model.max_context, 32768)
        self.assertEqual(model.native_context, 32768)
        self.assertEqual(model.estimated_ram_gib, 48.0)
        self.assertEqual(model.cache_dir, "user")
        self.assertEqual(model.image, "nvcr.io/nvidia/vllm:26.07-py3")
        self.assertEqual(
            model.image_digest,
            "sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268",
        )
        self.assertEqual(
            model.args[model.args.index("--max-num-seqs") + 1], "8"
        )
        self.assertNotIn("--reasoning-parser", model.args)
        self.assertNotIn("--enable-auto-tool-choice", model.args)

    def test_phi4_multimodal_fp8_profile_is_pinned_and_reuses_longrope_fix(
        self,
    ) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        model = models["phi-4-multimodal-instruct-fp8"]
        nvfp4 = models["phi-4-multimodal-instruct-nvfp4"]

        self.assertEqual(model.backend, "sglang")
        self.assertEqual(model.source, "nvidia/Phi-4-multimodal-instruct-FP8")
        self.assertEqual(model.served_name, model.source)
        self.assertEqual(
            model.revision, "d822efce23f65f86c165aeed435cc27092e21d60"
        )
        self.assertEqual(model.tasks, ("chat", "vision"))
        self.assertEqual(model.quantization, "fp8")
        self.assertEqual(model.support_status, "spark_other_backend")
        self.assertEqual(model.cache_dir, "user")
        self.assertEqual(model.max_context, 32768)
        self.assertEqual(model.native_context, 131072)
        self.assertEqual(model.estimated_ram_gib, 48.0)
        self.assertEqual(model.endpoint, "http://127.0.0.1:30000/v1")
        self.assertEqual(model.image, "scitrera/dgx-spark-sglang:0.5.10rc0")
        self.assertEqual(
            model.image_digest,
            "sha256:3f51e3b127bd0fe8f261a84c6ad54ce42bdb65eb2e57e228a9f6359e89bd08ec",
        )
        self.assertNotIn("--quantization", model.args)
        self.assertIn("--enable-multimodal", model.args)
        self.assertEqual(
            model.args[model.args.index("--max-running-requests") + 1], "8"
        )
        override = model.args[
            model.args.index("--json-model-override-args") + 1
        ]
        nvfp4_override = nvfp4.args[
            nvfp4.args.index("--json-model-override-args") + 1
        ]
        self.assertEqual(override, nvfp4_override)
        rope = json.loads(override)["rope_parameters"]
        self.assertEqual(rope["type"], "longrope")
        self.assertEqual(rope["rope_theta"], 10000.0)
        self.assertEqual(len(rope["long_factor"]), 48)
        self.assertEqual(len(rope["short_factor"]), 48)
        for unverified_capability in ("audio", "json", "thinking", "tools"):
            self.assertNotIn(unverified_capability, model.tasks)

    def test_phi4_multimodal_fp8_sglang_audio_profile_is_incompatible(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        model = models["phi-4-multimodal-instruct-fp8-audio"]
        image_text = models["phi-4-multimodal-instruct-fp8"]

        self.assertNotEqual(model.id, image_text.id)
        self.assertEqual(model.source, image_text.source)
        self.assertEqual(model.served_name, image_text.served_name)
        self.assertEqual(model.revision, image_text.revision)
        self.assertEqual(model.image, image_text.image)
        self.assertEqual(model.image_digest, image_text.image_digest)
        self.assertEqual(model.backend, "sglang")
        self.assertEqual(model.support_status, "incompatible")
        self.assertEqual(model.tasks, ("chat",))
        self.assertNotIn("audio", model.tasks)
        self.assertNotIn("vision", model.tasks)
        self.assertEqual(model.quantization, "fp8")
        self.assertEqual(model.cache_dir, "user")
        self.assertEqual(model.max_context, 32768)
        self.assertEqual(model.native_context, 131072)
        self.assertEqual(model.estimated_ram_gib, 50.0)
        self.assertEqual(model.endpoint, "http://127.0.0.1:30000/v1")
        self.assertNotIn("--quantization", model.args)
        self.assertIn("--enable-multimodal", model.args)
        self.assertEqual(
            model.args[model.args.index("--json-model-override-args") + 1],
            image_text.args[
                image_text.args.index("--json-model-override-args") + 1
            ],
        )
        self.assertEqual(
            model.args[model.args.index("--lora-paths") + 1],
            "speech=/root/.cache/huggingface/hub/"
            "models--nvidia--Phi-4-multimodal-instruct-FP8/snapshots/"
            "d822efce23f65f86c165aeed435cc27092e21d60/speech-lora",
        )
        self.assertEqual(
            model.args[model.args.index("--max-loras-per-batch") + 1], "2"
        )

    def test_phi4_multimodal_fp8_trtllm_audio_profile_is_exactly_pinned(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        model = models["phi-4-multimodal-instruct-fp8-trtllm-audio"]

        self.assertEqual(model.backend, "trtllm")
        self.assertEqual(model.support_status, "spark_trtllm_direct")
        self.assertEqual(model.source, "nvidia/Phi-4-multimodal-instruct-FP8")
        self.assertEqual(
            model.revision,
            "d822efce23f65f86c165aeed435cc27092e21d60",
        )
        self.assertEqual(model.tasks, ("chat", "audio"))
        self.assertEqual(model.lifecycle, "docker")
        self.assertEqual(
            model.image,
            "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13",
        )
        self.assertEqual(
            model.image_digest,
            "sha256:4f30c464ead64fb9727a24064b25057dacc07bef848022421108e544c91f0965",
        )
        self.assertEqual(model.cache_dir, "user")
        self.assertEqual(model.max_context, 32768)
        self.assertEqual(model.native_context, 131072)
        self.assertEqual(model.startup_timeout_s, 7200)
        self.assertEqual(model.estimated_ram_gib, 64.0)
        self.assertEqual(model.endpoint, "offline://trtllm")
        self.assertEqual(
            model.args,
            (
                "--backend",
                "pytorch",
                "--max-seq-len",
                "32768",
                "--max-batch-size",
                "1",
                "--max-num-tokens",
                "8192",
                "--disable-block-reuse",
                "--seed",
                "3407",
            ),
        )

    def test_chat_quality_suite_is_deterministic_and_backend_neutral(self) -> None:
        suite = load_suite(ROOT / "manifests" / "suites" / "chat_quality.toml")

        self.assertEqual(len(suite.cases), 1)
        case = suite.cases[0]
        self.assertEqual(case.id, "synthetic-exact-answer")
        self.assertEqual(case.kind, "quality")
        self.assertEqual(case.requires, ("chat",))
        self.assertEqual(case.temperature, 0.0)
        self.assertEqual(case.prompt_repetitions, 0)
        self.assertEqual(case.warmups, 0)
        self.assertEqual(case.concurrency, 1)

    def test_quality_case_rejects_non_deterministic_or_non_chat_settings(self) -> None:
        valid = CaseSpec(id="quality", kind="quality", requires=("chat",))
        validate_case(valid)

        invalid_cases = (
            replace(valid, temperature=0.1),
            replace(valid, prompt_repetitions=1),
            replace(valid, requires=("chat", "tools")),
        )
        for case in invalid_cases:
            with self.subTest(case=case), self.assertRaises(ManifestError):
                validate_case(case)

    def test_multimodal_rerank_suite_is_small_and_vllm_compatible(self) -> None:
        suite = load_suite(
            ROOT / "manifests" / "suites" / "multimodal_rerank.toml"
        )

        self.assertEqual(len(suite.cases), 1)
        case = suite.cases[0]
        self.assertEqual(case.id, "red-over-blue-image")
        self.assertEqual(case.kind, "capability")
        self.assertEqual(case.requires, ("rerank", "vision"))
        self.assertEqual(case.prompt_repetitions, 64)
        self.assertEqual(case.concurrency, 1)
        self.assertEqual(case.temperature, 0.0)
        self.assertIn("never journaled", suite.description)

    def test_loads_valid_model_and_normalizes_values(self) -> None:
        path = self._write(
            """
            schema_version = 1
            [[models]]
            id = "tiny-test"
            backend = "VLLM"
            source = "example/tiny"
            served_name = "tiny"
            tasks = ["chat", "tools"]
            image = "example/vllm:test"
            image_digest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            max_context = 4096
            request_body_json = '{"seed": 7}'
            """
        )

        model = load_models(path)["tiny-test"]

        self.assertEqual(model.backend, "vllm")
        self.assertEqual(model.endpoint, "http://127.0.0.1:8000/v1")
        self.assertEqual(model.tasks, ("chat", "tools"))
        self.assertEqual(model.startup_timeout_s, 1800)

    def test_unknown_fields_and_duplicate_ids_are_rejected(self) -> None:
        unknown = self._write(
            """
            schema_version = 1
            [[models]]
            id = "tiny"
            backend = "vllm"
            source = "example/tiny"
            served_name = "tiny"
            tasks = ["chat"]
            image = "example/vllm:test"
            max_context = 1024
            typo_field = true
            """
        )
        with self.assertRaisesRegex(ManifestError, "unknown field"):
            load_models(unknown)

        duplicate = self._write(
            """
            schema_version = 1
            [[models]]
            id = "same"
            backend = "ollama"
            source = "same:latest"
            served_name = "same:latest"
            tasks = ["chat"]
            lifecycle = "existing"
            max_context = 1024
            [[models]]
            id = "same"
            backend = "ollama"
            source = "other:latest"
            served_name = "other:latest"
            tasks = ["chat"]
            lifecycle = "existing"
            max_context = 1024
            """
        )
        with self.assertRaisesRegex(ManifestError, "duplicate model id"):
            load_models(duplicate)

    def test_suite_rejects_duplicate_cases(self) -> None:
        path = self._write(
            """
            schema_version = 1
            [suite]
            id = "duplicate"
            [[cases]]
            id = "same"
            kind = "decode"
            requires = ["chat"]
            [[cases]]
            id = "same"
            kind = "decode"
            requires = ["chat"]
            """
        )

        with self.assertRaisesRegex(ManifestError, "duplicate case id"):
            load_suite(path)

    def test_case_numeric_and_kind_invariants(self) -> None:
        valid = CaseSpec(id="prefill", kind="prefill", requires=("chat",), prompt_repetitions=4)
        validate_case(valid)

        invalid_cases = (
            replace(valid, warmups=-1),
            replace(valid, repetitions=0),
            replace(valid, concurrency=0),
            replace(valid, prompt_repetitions=0),
            replace(valid, max_output_tokens=0),
            replace(valid, temperature=2.1),
            replace(valid, kind="unsupported"),
        )
        for case in invalid_cases:
            with self.subTest(case=case), self.assertRaises(ManifestError):
                validate_case(case)

    def test_model_rejects_remote_endpoint_bad_json_and_bad_digest(self) -> None:
        valid = ModelSpec(
            id="safe",
            backend="ollama",
            source="safe:latest",
            served_name="safe:latest",
            tasks=("chat",),
            lifecycle="existing",
            max_context=1024,
        )
        validate_model(valid)

        invalid_models = (
            replace(valid, endpoint="http://example.com:11434/v1"),
            replace(valid, request_body_json="[]"),
            replace(valid, image_digest="sha256:not-a-digest"),
            replace(valid, tasks=("chat", "chat")),
        )
        for model in invalid_models:
            with self.subTest(model=model), self.assertRaises(ManifestError):
                validate_model(model)

    def test_registry_key_must_match_model_id(self) -> None:
        model = ModelSpec(
            id="actual",
            backend="ollama",
            source="actual:latest",
            served_name="actual:latest",
            tasks=("chat",),
            lifecycle="existing",
            max_context=1024,
        )
        with self.assertRaisesRegex(ManifestError, "does not match"):
            validate_models({"different": model})

    def test_unknown_backend_is_rejected_during_validation(self) -> None:
        model = ModelSpec(
            id="unknown-runtime",
            backend="typo",
            source="example/model",
            served_name="example/model",
            tasks=("chat",),
            lifecycle="existing",
            max_context=1024,
        )

        with self.assertRaisesRegex(ManifestError, "backend"):
            validate_model(model)


if __name__ == "__main__":
    unittest.main()
