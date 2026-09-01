from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.densespark import (
    DENSESPARK_C1_ARGS,
    DENSESPARK_C1_CASES,
    DENSESPARK_C1_ENVIRONMENT,
    DENSESPARK_IMAGE,
    DENSESPARK_LOCAL_IMAGE_ID,
    DENSESPARK_MAX_CONTEXT,
    DENSESPARK_MODEL_REVISION,
    DENSESPARK_MODEL_SOURCE,
    DENSESPARK_NATIVE_CONTEXT,
    DENSESPARK_PQ_CACHE_RELATIVE_PATH,
    DENSESPARK_PQ_SHA256,
    DENSESPARK_PQ_SIZE_BYTES,
    DENSESPARK_PROFILE_ID,
    DENSESPARK_RECIPE_REVISION,
    DENSESPARK_RECIPE_SOURCE,
    DENSESPARK_SERVED_NAME,
    DENSESPARK_SNAPSHOT_FILE_PINS,
    DENSESPARK_TOOL_CALL_PARSER,
    DENSESPARK_TOOL_CASES,
    DENSESPARK_TOOL_SUITE_DESCRIPTION,
    DENSESPARK_TOOL_SUITE_ID,
    DENSESPARK_SUITE_ID,
    DENSESPARK_WEIGHT_FILE_COUNT,
    DENSESPARK_WEIGHT_FILES,
    DENSESPARK_WEIGHT_SIZE_BYTES,
    DenseSparkContractError,
    DenseSparkSnapshotReceipt,
    canonical_densespark_config,
    densespark_c1_cache_config,
    densespark_c1_environment,
    densespark_cache_namespace,
    densespark_compile_cache_path,
    densespark_configuration_digest,
    densespark_expected_launch_policy,
    densespark_pq_artifact_path,
    is_densespark_profile_identity,
    require_densespark_profile_identity,
    validate_densespark_local_image,
    validate_densespark_pq_artifact,
    validate_densespark_profile,
    validate_densespark_snapshot,
    validate_densespark_suite,
)


def _profile(**changes: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": DENSESPARK_PROFILE_ID,
        "architecture": "dense+gdn",
        "backend": "vllm",
        "source": DENSESPARK_MODEL_SOURCE,
        "revision": DENSESPARK_MODEL_REVISION,
        "recipe_source": DENSESPARK_RECIPE_SOURCE,
        "recipe_revision": DENSESPARK_RECIPE_REVISION,
        "served_name": DENSESPARK_SERVED_NAME,
        "tasks": ("chat", "thinking", "tools"),
        "image": DENSESPARK_IMAGE,
        "local_image_id": DENSESPARK_LOCAL_IMAGE_ID,
        "cache_dir": "user",
        "max_context": DENSESPARK_MAX_CONTEXT,
        "native_context": DENSESPARK_NATIVE_CONTEXT,
        "startup_timeout_s": 1_800,
        "endpoint": "http://127.0.0.1:8000/v1",
        "estimated_ram_gib": 92.0,
        "lifecycle": "docker",
        "support_status": "spark_vllm_recipe",
        "quantization": "int4-autoround+densespark-pq",
        "weight_file_count": DENSESPARK_WEIGHT_FILE_COUNT,
        "weight_size_bytes": DENSESPARK_WEIGHT_SIZE_BYTES,
        "densespark_pq_file": DENSESPARK_PQ_CACHE_RELATIVE_PATH.as_posix(),
        "densespark_pq_digest": DENSESPARK_PQ_SHA256,
        "densespark_pq_size_bytes": DENSESPARK_PQ_SIZE_BYTES,
        "args": DENSESPARK_C1_ARGS,
    }
    values.update(changes)
    return SimpleNamespace(**values)


class DenseSparkIdentityTests(unittest.TestCase):
    def identity(self) -> dict[str, object]:
        return {
            "recipe_source": DENSESPARK_RECIPE_SOURCE,
            "recipe_revision": DENSESPARK_RECIPE_REVISION,
            "model_source": DENSESPARK_MODEL_SOURCE,
            "model_revision": DENSESPARK_MODEL_REVISION,
        }

    def test_recognizes_only_the_exact_pinned_identity(self) -> None:
        identity = self.identity()
        self.assertTrue(is_densespark_profile_identity(**identity))
        require_densespark_profile_identity(**identity)

        for key in identity:
            mismatch = dict(identity)
            mismatch[key] = f"{mismatch[key]}-different"
            self.assertFalse(is_densespark_profile_identity(**mismatch), key)
            with self.assertRaises(DenseSparkContractError, msg=key):
                require_densespark_profile_identity(**mismatch)

    def test_non_string_values_do_not_coerce_into_identity(self) -> None:
        class StringSubclass(str):
            pass

        identity = self.identity()
        for impostor in (Path(DENSESPARK_RECIPE_SOURCE), StringSubclass(DENSESPARK_RECIPE_SOURCE)):
            identity["recipe_source"] = impostor
            self.assertFalse(is_densespark_profile_identity(**identity))

    def test_managed_profile_contract_is_exact_and_requires_v12_tools(self) -> None:
        validate_densespark_profile(_profile())
        mutations = (
            {"id": "different"},
            {"image": "local/densespark:different"},
            {"local_image_id": f"sha256:{'0' * 64}"},
            {"max_context": 32_768},
            {"tasks": ("chat", "thinking")},
            {"args": (*DENSESPARK_C1_ARGS, "--enable-auto-tool-choice")},
            {"densespark_pq_size_bytes": DENSESPARK_PQ_SIZE_BYTES + 1},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(DenseSparkContractError):
                    validate_densespark_profile(_profile(**mutation))

    def test_c1_suite_contract_is_single_concurrency_one_decode(self) -> None:
        case = SimpleNamespace(**DENSESPARK_C1_CASES[0])
        suite = SimpleNamespace(id=DENSESPARK_SUITE_ID, cases=(case,))
        validate_densespark_suite(suite)

        for change in (
            {"concurrency": 2},
            {"requires": ("chat", "tools")},
            {"max_output_tokens": 512},
        ):
            invalid_case = SimpleNamespace(**{**DENSESPARK_C1_CASES[0], **change})
            with self.subTest(change=change):
                with self.assertRaises(DenseSparkContractError):
                    validate_densespark_suite(
                        SimpleNamespace(
                            id=DENSESPARK_SUITE_ID,
                            cases=(invalid_case,),
                        )
                    )

    def test_tool_suite_contract_is_the_canonical_exportable_battery(self) -> None:
        suite = SimpleNamespace(
            id=DENSESPARK_TOOL_SUITE_ID,
            description=DENSESPARK_TOOL_SUITE_DESCRIPTION,
            cases=tuple(SimpleNamespace(**case) for case in DENSESPARK_TOOL_CASES),
        )
        validate_densespark_suite(suite)
        invalid = SimpleNamespace(
            **{**DENSESPARK_TOOL_CASES[0], "repetitions": 1}
        )
        with self.assertRaises(DenseSparkContractError):
            validate_densespark_suite(
                SimpleNamespace(
                    id=DENSESPARK_TOOL_SUITE_ID,
                    description=DENSESPARK_TOOL_SUITE_DESCRIPTION,
                    cases=(invalid, *suite.cases[1:]),
                )
            )


class DenseSparkPQArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "pq.bin"
        self.payload = b"synthetic-pq-artifact\x00" * 31
        self.path.write_bytes(self.payload)
        self.digest = hashlib.sha256(self.payload).hexdigest()

    def test_validates_exact_regular_file_and_returns_scalar_receipt(self) -> None:
        receipt = validate_densespark_pq_artifact(
            self.path,
            expected_size_bytes=len(self.payload),
            expected_sha256=self.digest,
        )
        self.assertEqual(receipt.size_bytes, len(self.payload))
        self.assertEqual(receipt.sha256, f"sha256:{self.digest}")

        prefixed = validate_densespark_pq_artifact(
            self.path,
            expected_size_bytes=len(self.payload),
            expected_sha256=f"sha256:{self.digest}",
        )
        self.assertEqual(prefixed, receipt)

    def test_rejects_wrong_size_or_digest(self) -> None:
        with self.assertRaisesRegex(DenseSparkContractError, "size"):
            validate_densespark_pq_artifact(
                self.path,
                expected_size_bytes=len(self.payload) + 1,
                expected_sha256=self.digest,
            )
        with self.assertRaisesRegex(DenseSparkContractError, "digest"):
            validate_densespark_pq_artifact(
                self.path,
                expected_size_bytes=len(self.payload),
                expected_sha256="0" * 64,
            )

    def test_rejects_symlink_and_non_regular_path(self) -> None:
        symlink = self.path.with_name("pq-link.bin")
        symlink.symlink_to(self.path)
        with self.assertRaisesRegex(DenseSparkContractError, "symlink"):
            validate_densespark_pq_artifact(
                symlink,
                expected_size_bytes=len(self.payload),
                expected_sha256=self.digest,
            )
        with self.assertRaisesRegex(DenseSparkContractError, "regular"):
            validate_densespark_pq_artifact(
                Path(self.temporary_directory.name),
                expected_size_bytes=len(self.payload),
                expected_sha256=self.digest,
            )

    def test_rejects_malformed_pins(self) -> None:
        for invalid_size in (True, 0, -1):
            with self.assertRaises(DenseSparkContractError):
                validate_densespark_pq_artifact(
                    self.path,
                    expected_size_bytes=invalid_size,
                    expected_sha256=self.digest,
                )
        for invalid_digest in ("A" * 64, "short", "sha256:" + "g" * 64):
            with self.assertRaises(DenseSparkContractError):
                validate_densespark_pq_artifact(
                    self.path,
                    expected_size_bytes=len(self.payload),
                    expected_sha256=invalid_digest,
                )


class DenseSparkSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repository = Path(self.temporary_directory.name) / "repository"
        self.snapshot = (
            self.repository / "snapshots" / DENSESPARK_MODEL_REVISION
        )
        self.blobs = self.repository / "blobs"
        self.snapshot.mkdir(parents=True)
        self.blobs.mkdir()
        self.file_pins: dict[str, tuple[str, int, str]] = {}
        self.payloads: dict[str, bytes] = {}
        self.blob_paths: dict[str, Path] = {}
        self.total_size = 0
        for index, (filename, pin) in enumerate(
            DENSESPARK_SNAPSHOT_FILE_PINS.items(),
            start=1,
        ):
            blob_name, _real_size, _real_digest = pin
            payload = (f"synthetic-{index}-{filename}\n".encode("utf-8")) * index
            blob = self.blobs / blob_name
            blob.write_bytes(payload)
            (self.snapshot / filename).symlink_to(
                Path("../../blobs") / blob_name
            )
            self.file_pins[filename] = (
                blob_name,
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            )
            self.payloads[filename] = payload
            self.blob_paths[filename] = blob
            if filename in DENSESPARK_WEIGHT_FILES:
                self.total_size += len(payload)

    def validate(self) -> DenseSparkSnapshotReceipt:
        with (
            patch(
                "bench.densespark.DENSESPARK_SNAPSHOT_FILE_PINS",
                self.file_pins,
            ),
            patch(
                "bench.densespark.DENSESPARK_WEIGHT_SIZE_BYTES",
                self.total_size,
            ),
        ):
            return validate_densespark_snapshot(
                self.snapshot,
                repository_root=self.repository,
            )

    def test_exact_huggingface_symlink_layout_returns_scalar_receipt(self) -> None:
        receipt = self.validate()
        self.assertEqual(receipt.weight_file_count, DENSESPARK_WEIGHT_FILE_COUNT)
        self.assertEqual(receipt.weight_size_bytes, self.total_size)

    def test_rejects_unknown_missing_or_non_symlink_snapshot_entries(self) -> None:
        (self.snapshot / "extra.safetensors").write_bytes(b"extra")
        with self.assertRaisesRegex(DenseSparkContractError, "layout"):
            self.validate()
        (self.snapshot / "extra.safetensors").unlink()

        unknown_blob = self.blobs / "unknown"
        unknown_blob.write_bytes(b"unknown")
        (self.snapshot / "unknown.json").symlink_to(
            Path("../../blobs/unknown")
        )
        with self.assertRaisesRegex(DenseSparkContractError, "layout"):
            self.validate()
        (self.snapshot / "unknown.json").unlink()

        config = self.snapshot / "config.json"
        config.unlink()
        with self.assertRaisesRegex(DenseSparkContractError, "layout"):
            self.validate()
        config.write_bytes(self.payloads["config.json"])
        with self.assertRaisesRegex(DenseSparkContractError, "symlinks"):
            self.validate()

    def test_rejects_repository_escape(self) -> None:
        escaped = Path(self.temporary_directory.name) / "escaped"
        escaped.write_bytes(b"escaped")
        first = self.snapshot / DENSESPARK_WEIGHT_FILES[0]
        first.unlink()
        first.symlink_to(escaped)
        with self.assertRaisesRegex(DenseSparkContractError, "unsafe"):
            self.validate()

        first.unlink()
        same_repository_nonblob = self.repository / "wrong-weight"
        same_repository_nonblob.write_bytes(b"escaped")
        first.symlink_to(same_repository_nonblob)
        with self.assertRaisesRegex(DenseSparkContractError, "unsafe"):
            self.validate()

    def test_rejects_same_size_weight_and_operational_file_tampering(self) -> None:
        for filename in (DENSESPARK_WEIGHT_FILES[0], "config.json"):
            with self.subTest(filename=filename):
                blob = self.blob_paths[filename]
                original = self.payloads[filename]
                tampered = bytes([original[0] ^ 1]) + original[1:]
                self.assertEqual(len(tampered), len(original))
                blob.write_bytes(tampered)
                with self.assertRaisesRegex(DenseSparkContractError, "digest"):
                    self.validate()
                blob.write_bytes(original)

    def test_rejects_blob_path_replacement_during_descriptor_hash(self) -> None:
        filename = next(iter(self.file_pins))
        blob = self.blob_paths[filename]
        replacement = self.blobs / "replacement"
        replacement.write_bytes(self.payloads[filename])
        real_read = os.read
        replaced = False

        def racing_read(descriptor: int, size: int) -> bytes:
            nonlocal replaced
            payload = real_read(descriptor, size)
            if payload and not replaced:
                os.replace(replacement, blob)
                replaced = True
            return payload

        with patch("bench.densespark.os.read", side_effect=racing_read):
            with self.assertRaisesRegex(DenseSparkContractError, "changed while"):
                self.validate()
        self.assertTrue(replaced)


class DenseSparkConfigurationTests(unittest.TestCase):
    def test_launch_policy_is_exact_scalar_path_free_and_deterministic(self) -> None:
        first = densespark_expected_launch_policy()
        second = densespark_expected_launch_policy()
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], "sparkbench.densespark.launch-policy.v1")
        self.assertEqual(
            first["sha256"],
            "sha256:55f5ee63b5daea287e89038a19fab816"
            "953d64ed71751111d57f88dbd210cda8",
        )
        self.assertEqual(first["host_safety_min_memavailable_bytes"], 14 * 1024**3)
        self.assertEqual(first["host_safety_max_swap_growth_bytes"], 512 * 1024**2)
        self.assertEqual(first["host_safety_max_starting_swap_bytes"], 512 * 1024**2)
        self.assertEqual(first["environment_hf_hub_offline"], "1")
        self.assertEqual(first["environment_vllm_no_usage_stats"], "1")
        self.assertEqual(first["docker_pull_policy"], "never")
        self.assertEqual(first["docker_network"], "bridge")
        self.assertEqual(first["docker_network_egress"], "capable")
        self.assertEqual(first["docker_network_isolation"], "none")
        self.assertEqual(first["publish_host"], "127.0.0.1")
        self.assertEqual(first["publish_host_port"], 8000)
        self.assertEqual(first["publish_container_port"], 8000)
        self.assertTrue(all(isinstance(value, (int, str)) for value in first.values()))
        self.assertFalse(any(key.endswith("_path") for key in first))

    def test_digest_and_namespace_are_order_independent_and_path_safe(self) -> None:
        first = {
            "DENSESPARK_SPEC_TOKENS": 3,
            "DENSESPARK_HEAD_AUTOTUNE": True,
            "DENSESPARK_MARLIN_NSPLIT": "4",
        }
        second = dict(reversed(list(first.items())))

        digest = densespark_configuration_digest(first)
        self.assertEqual(digest, densespark_configuration_digest(second))
        self.assertEqual(
            digest,
            "sha256:ee980683f4a5d5d855633b8c99d026dfe2723cee844eded5e814bf2e9819fdf6",
        )
        self.assertRegex(digest, r"^sha256:[0-9a-f]{64}$")
        namespace = densespark_cache_namespace(first)
        self.assertEqual(namespace, densespark_cache_namespace(second))
        self.assertRegex(namespace, r"^densespark-v1-[0-9a-f]{64}$")
        self.assertIsNone(re.search(r"[/\\\s]", namespace))

    def test_digest_changes_with_value_or_scalar_type(self) -> None:
        integer = {"DENSESPARK_SPEC_TOKENS": 1}
        string = {"DENSESPARK_SPEC_TOKENS": "1"}
        boolean = {"DENSESPARK_SPEC_TOKENS": True}
        self.assertNotEqual(
            densespark_configuration_digest(integer),
            densespark_configuration_digest(string),
        )
        self.assertNotEqual(
            densespark_configuration_digest(integer),
            densespark_configuration_digest(boolean),
        )

    def test_canonical_config_is_sorted_without_mutating_input(self) -> None:
        source = {
            "DENSESPARK_SPEC_TOKENS": 3,
            "DENSESPARK_CONCURRENCY": 1,
        }
        before = dict(source)
        canonical = canonical_densespark_config(source)
        self.assertEqual(list(canonical), sorted(source))
        self.assertEqual(source, before)

    def test_rejects_unknown_keys_and_non_scalar_values(self) -> None:
        invalid_configs = (
            {"PATH": "/synthetic"},
            {1: "value"},
            {"DENSESPARK_SPEC_TOKENS": 1.0},
            {"DENSESPARK_SPEC_TOKENS": [1]},
            {"DENSESPARK_SPEC_TOKENS": -1},
            {"DENSESPARK_DRAFT_SAMPLE_METHOD": "line\nbreak"},
        )
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(DenseSparkContractError):
                    densespark_configuration_digest(config)  # type: ignore[arg-type]

    def test_c1_environment_and_cache_paths_are_fixed_and_non_secret(self) -> None:
        environment = densespark_c1_environment()
        self.assertEqual(tuple(environment.items()), DENSESPARK_C1_ENVIRONMENT)
        self.assertEqual(environment["DENSESPARK_LAB90_EXACT_SAMPLER"], "0")
        self.assertNotIn("HF_TOKEN", environment)
        self.assertNotIn("DENSESPARK_LAB118_RUNTIME_AUDIT", environment)
        self.assertEqual(
            densespark_c1_cache_config()["DENSESPARK_DRAFT_SAMPLE_METHOD"],
            "probabilistic",
        )
        self.assertEqual(
            densespark_c1_cache_config()["DENSESPARK_TOOL_CALL_PARSER"],
            DENSESPARK_TOOL_CALL_PARSER,
        )
        self.assertTrue(
            densespark_c1_cache_config()["DENSESPARK_AUTO_TOOL_CHOICE"]
        )
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self.assertEqual(
                densespark_pq_artifact_path(home=home),
                home / ".cache" / Path(*DENSESPARK_PQ_CACHE_RELATIVE_PATH.parts),
            )
            compile_cache = densespark_compile_cache_path(home=home)
            self.assertEqual(
                compile_cache.name,
                densespark_cache_namespace(densespark_c1_cache_config()),
            )
            self.assertEqual(
                compile_cache.parent.name,
                "densespark-vllm-repro-b4c61732",
            )


class DenseSparkDockerImageTests(unittest.TestCase):
    image = "local/densespark:qwen3.8-27b"
    image_id = f"sha256:{'a' * 64}"

    def test_injected_runner_resolves_exact_local_image_id(self) -> None:
        commands: list[tuple[str, ...]] = []

        def runner(command: object) -> subprocess.CompletedProcess[str]:
            commands.append(tuple(command))  # type: ignore[arg-type]
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=f"{self.image_id}\n",
                stderr="",
            )

        resolved = validate_densespark_local_image(
            self.image,
            expected_image_id=self.image_id,
            runner=runner,
        )
        self.assertEqual(resolved, self.image_id)
        self.assertEqual(
            commands,
            [("docker", "image", "inspect", "--format", "{{.Id}}", self.image)],
        )

    def test_rejects_mismatch_nonzero_and_malformed_output(self) -> None:
        cases = (
            subprocess.CompletedProcess([], 0, f"sha256:{'b' * 64}\n", ""),
            subprocess.CompletedProcess([], 1, "", "synthetic-secret"),
            subprocess.CompletedProcess([], 0, f"{self.image_id}\n{self.image_id}\n", ""),
            subprocess.CompletedProcess([], 0, f"{'a' * 64}\n", ""),
        )
        for result in cases:
            with self.subTest(result=result):
                with self.assertRaises(DenseSparkContractError) as raised:
                    validate_densespark_local_image(
                        self.image,
                        expected_image_id=self.image_id,
                        runner=lambda command, result=result: result,
                    )
                self.assertNotIn("synthetic-secret", str(raised.exception))

    def test_rejects_invalid_reference_and_pin_without_calling_runner(self) -> None:
        calls = 0

        def runner(command: object) -> subprocess.CompletedProcess[str]:
            nonlocal calls
            calls += 1
            return subprocess.CompletedProcess(command, 0, self.image_id, "")

        invalid_cases = (
            ("--help", self.image_id),
            ("image with spaces", self.image_id),
            (self.image, "a" * 64),
            (self.image, f"sha256:{'A' * 64}"),
        )
        for image, image_id in invalid_cases:
            with self.subTest(image=image, image_id=image_id):
                with self.assertRaises(DenseSparkContractError):
                    validate_densespark_local_image(
                        image,
                        expected_image_id=image_id,
                        runner=runner,
                    )
        self.assertEqual(calls, 0)

    def test_runner_os_error_is_wrapped(self) -> None:
        def runner(command: object) -> subprocess.CompletedProcess[str]:
            raise FileNotFoundError("docker")

        with self.assertRaisesRegex(DenseSparkContractError, "inspected"):
            validate_densespark_local_image(
                self.image,
                expected_image_id=self.image_id,
                runner=runner,
            )

    def test_malformed_runner_result_is_a_contract_failure(self) -> None:
        with self.assertRaises(DenseSparkContractError):
            validate_densespark_local_image(
                self.image,
                expected_image_id=self.image_id,
                runner=lambda command: object(),  # type: ignore[arg-type,return-value]
            )


if __name__ == "__main__":
    unittest.main()
