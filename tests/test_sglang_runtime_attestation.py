from __future__ import annotations

from copy import deepcopy
import hashlib
import unittest

from bench import sglang_runtime_attestation as attestation


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("ascii")).hexdigest()


def _record() -> dict[str, object]:
    return {
        "schema_version": attestation.RUNTIME_ATTESTATION_SCHEMA_VERSION,
        "kind": attestation.RUNTIME_ATTESTATION_KIND,
        "candidate_id": attestation.SM121_TRITON_CANDIDATE_ID,
        "status": attestation.RUNTIME_ATTESTATION_STATUS,
        "source_tree": attestation.SM121_TRITON_SOURCE_TREE,
        "build_contract_sha256": (
            attestation.SM121_TRITON_BUILD_CONTRACT_SHA256
        ),
        "oci_image_digest": _digest("synthetic immutable OCI manifest"),
        "platform": attestation.SM121_TRITON_PLATFORM,
        "model_sha256": _digest("synthetic pinned model artifact"),
        "tokenizer_sha256": _digest("synthetic pinned tokenizer artifact"),
        "revision_sha256": _digest("synthetic pinned revision artifact"),
        "profile_sha256": _digest("synthetic pinned profile artifact"),
        "retired_overlay_rejected": True,
        "storage_import_passed": True,
        "io_uring_passed": True,
        "ple_rows_passed": True,
        "sm121_triton_passed": True,
        "quality_passed": True,
        "long_context_passed": True,
    }


class SGLangRuntimeAttestationTests(unittest.TestCase):
    def test_accepts_complete_scalar_only_admission(self) -> None:
        record = _record()

        parsed = attestation.validate_sglang_runtime_attestation(record)

        self.assertIsInstance(parsed, attestation.SGLangRuntimeAttestation)
        self.assertEqual(parsed.to_mapping(), record)
        self.assertTrue(
            all(
                type(value) in {bool, int, str}
                for value in parsed.to_mapping().values()
            )
        )

    def test_fixed_future_candidate_identity_rejects_legacy_route(self) -> None:
        cases = {
            "candidate_id": "qwen38-flash-next-nvfp4-mtp-sglang",
            "source_tree": "d91c3682b0b429e4c70df63cd57f819588ce29b0",
            "build_contract_sha256": _digest("a different build contract"),
            "platform": "linux/amd64",
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                record = _record()
                record[field] = value
                with self.assertRaises(attestation.SGLangRuntimeAttestationError):
                    attestation.validate_sglang_runtime_attestation(record)

    def test_requires_exact_field_set_and_exact_object_type(self) -> None:
        missing = _record()
        del missing["quality_passed"]
        unknown = _record()
        unknown["legacy_overlay_path"] = "results/runtime-overlays/legacy.py"

        for value in (missing, unknown, [], {1: "not-a-string-key"}):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(attestation.SGLangRuntimeAttestationError):
                    attestation.validate_sglang_runtime_attestation(value)

    def test_rejects_nested_non_scalar_known_fields(self) -> None:
        cases: tuple[tuple[str, object], ...] = (
            ("oci_image_digest", {"digest": _digest("nested")}),
            ("model_sha256", [_digest("list")]),
            ("quality_passed", {"passed": True}),
            ("schema_version", True),
        )
        for field, value in cases:
            with self.subTest(field=field):
                record = _record()
                record[field] = value
                with self.assertRaises(attestation.SGLangRuntimeAttestationError):
                    attestation.validate_sglang_runtime_attestation(record)

    def test_rejects_tags_paths_and_placeholder_digests(self) -> None:
        cases = {
            "oci_image_digest": "local/sglang:sm121-storage-274ee330-runtime",
            "model_sha256": "../model.safetensors",
            "tokenizer_sha256": "sha256:" + "0" * 64,
            "revision_sha256": "sha256:" + "F" * 64,
            "profile_sha256": "sha256:" + "deadbeef" * 8,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                record = _record()
                record[field] = value
                with self.assertRaises(attestation.SGLangRuntimeAttestationError):
                    attestation.validate_sglang_runtime_attestation(record)

    def test_all_required_gates_and_admitted_status_are_strict(self) -> None:
        for field in (
            "retired_overlay_rejected",
            "storage_import_passed",
            "io_uring_passed",
            "ple_rows_passed",
            "sm121_triton_passed",
            "quality_passed",
            "long_context_passed",
        ):
            with self.subTest(field=field):
                record = _record()
                record[field] = False
                with self.assertRaises(attestation.SGLangRuntimeAttestationError):
                    attestation.validate_sglang_runtime_attestation(record)

        record = _record()
        record["status"] = "partial"
        with self.assertRaises(attestation.SGLangRuntimeAttestationError):
            attestation.validate_sglang_runtime_attestation(record)

    def test_does_not_reflect_path_or_secret_like_input(self) -> None:
        secret = "hf_example_secret_value"
        record = _record()
        record["model_sha256"] = secret
        with self.assertRaises(attestation.SGLangRuntimeAttestationError) as raised:
            attestation.validate_sglang_runtime_attestation(record)
        self.assertNotIn(secret, str(raised.exception))

        unknown = _record()
        unknown["api_token"] = secret
        with self.assertRaises(attestation.SGLangRuntimeAttestationError) as raised:
            attestation.validate_sglang_runtime_attestation(unknown)
        self.assertNotIn(secret, str(raised.exception))

    def test_dataclass_constructor_is_fail_closed_too(self) -> None:
        record = _record()
        record["long_context_passed"] = False
        with self.assertRaises(attestation.SGLangRuntimeAttestationError):
            attestation.SGLangRuntimeAttestation(**deepcopy(record))


if __name__ == "__main__":
    unittest.main()
