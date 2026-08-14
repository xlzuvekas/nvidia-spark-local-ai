from __future__ import annotations

from dataclasses import replace
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

        self.assertGreaterEqual(len(models), 1)
        self.assertEqual(smoke.id, "smoke")
        self.assertEqual(core.id, "core")
        self.assertEqual(len(core.cases), len({case.id for case in core.cases}))

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
