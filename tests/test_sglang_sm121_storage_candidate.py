"""Unit contracts for the future native SM121 PLE-storage SGLang route.

These tests intentionally mock Docker and endpoint I/O.  They describe the
small candidate-only launch surface: a pinned local image ID, direct NVMe
PLE reads, and a read-only container with the narrowly derived io_uring
seccomp profile.  The normal SGLang route remains covered by ``test_sglang``.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench import seccomp_profile_contract
from bench.runtime import RuntimeErrorWithContext, start_sglang
from bench.sglang_sm121_storage import (
    SM121StorageCandidateError,
    SM121_STORAGE_ARGS,
    SM121_STORAGE_CACHE_PAGES,
    SM121_STORAGE_CONTEXT_LENGTH,
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_LOCAL_IMAGE_TAG,
    SM121_STORAGE_MAX_BATCH_PAGES,
    SM121_STORAGE_NATIVE_CONTEXT,
    SM121_STORAGE_PROFILE_ID,
    SM121_STORAGE_QUEUE_DEPTH,
    SM121_STORAGE_SERVED_NAME,
    SM121_STORAGE_SOURCE,
    SM121_STORAGE_SOURCE_TREE,
    SM121_STORAGE_WEIGHT_FILE_COUNT,
    SM121_STORAGE_WEIGHT_SIZE_BYTES,
    is_sm121_storage_candidate,
    validate_sm121_storage_image_inspection,
)


SOURCE = SM121_STORAGE_SOURCE
REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
STORAGE_MODE = "qwen4_ple_nvme_io_uring"


def _completed(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


class SGLangSm121StorageCandidateTests(unittest.TestCase):
    def _model(self) -> SimpleNamespace:
        return SimpleNamespace(
            backend="sglang",
            id=SM121_STORAGE_PROFILE_ID,
            source=SOURCE,
            revision=REVISION,
            served_name=SM121_STORAGE_SERVED_NAME,
            tasks=("chat",),
            lifecycle="docker",
            image=SM121_STORAGE_LOCAL_IMAGE_TAG,
            image_digest=None,
            local_image_id=SM121_STORAGE_LOCAL_IMAGE_ID,
            resolved_image=SM121_STORAGE_LOCAL_IMAGE_ID,
            cache_dir="user",
            max_context=SM121_STORAGE_CONTEXT_LENGTH,
            native_context=SM121_STORAGE_NATIVE_CONTEXT,
            endpoint="http://127.0.0.1:30000/v1",
            weight_size_bytes=SM121_STORAGE_WEIGHT_SIZE_BYTES,
            weight_file_count=SM121_STORAGE_WEIGHT_FILE_COUNT,
            startup_timeout_s=1_200,
            run_identity="sm121-storage-test",
            storage_canary_authorized=True,
            sglang_storage_mode=STORAGE_MODE,
            sglang_ple_nvme_queue_depth=SM121_STORAGE_QUEUE_DEPTH,
            sglang_ple_nvme_max_batch_pages=SM121_STORAGE_MAX_BATCH_PAGES,
            sglang_ple_nvme_cache_pages=SM121_STORAGE_CACHE_PAGES,
            sglang_source_overlays=(),
            sglang_ple_mmap=False,
            sglang_ple_omitted=False,
            sglang_ple_cache_mode=None,
            sglang_ple_cache_marker_digest=None,
            sglang_ple_cache_payload_digest=None,
            draft_source=None,
            draft_revision=None,
            recipe_source=None,
            recipe_revision=None,
            request_body_json='{"chat_template_kwargs":{"enable_thinking":false}}',
            args=SM121_STORAGE_ARGS,
        )

    def _write_snapshot(self, workspace: Path) -> Path:
        snapshot = (
            workspace
            / "home"
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--RadixArk--Qwen3.8-Flash-Next-NVFP4"
            / "snapshots"
            / REVISION
        )
        snapshot.mkdir(parents=True)
        return snapshot

    @staticmethod
    def _write_seccomp_profile(workspace: Path) -> Path:
        profile = workspace / seccomp_profile_contract.DERIVED_PATH
        profile.parent.mkdir(parents=True)
        profile.write_text("{}\n", encoding="utf-8")
        return profile

    @staticmethod
    def _option_values(command: list[str], option: str) -> list[str]:
        return [
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == option
        ]

    @staticmethod
    def _local_image_inspection() -> dict[str, object]:
        return {
            "Id": SM121_STORAGE_LOCAL_IMAGE_ID,
            "RepoTags": [SM121_STORAGE_LOCAL_IMAGE_TAG],
            "Os": "linux",
            "Architecture": "arm64",
            "Config": {
                "Labels": {
                    "ai.sglang.build.commit": SM121_STORAGE_SOURCE_TREE,
                    "org.opencontainers.image.revision": SM121_STORAGE_SOURCE_TREE,
                }
            },
        }

    @staticmethod
    def _verifier_result() -> SimpleNamespace:
        return SimpleNamespace(
            candidate_id=seccomp_profile_contract.CONTRACT_CANDIDATE_ID,
            status=seccomp_profile_contract.CONTRACT_STATUS,
            derived_sha256=seccomp_profile_contract.DERIVED_SHA256,
            as_dict=lambda: {
                "candidate_id": seccomp_profile_contract.CONTRACT_CANDIDATE_ID,
                "derived_sha256": seccomp_profile_contract.DERIVED_SHA256,
                "status": seccomp_profile_contract.CONTRACT_STATUS,
                "verified": True,
            },
        )

    def _assert_no_docker_run(self, run: object) -> None:
        calls = getattr(run, "call_args_list")
        self.assertFalse(
            any(
                call.args[0][:2] == ["docker", "run"]
                for call in calls
                if call.args and isinstance(call.args[0], list)
            )
        )

    def test_storage_mode_is_explicit_candidate_selector(self) -> None:
        model = self._model()

        self.assertTrue(is_sm121_storage_candidate(model))
        model.sglang_storage_mode = "readonly_mmap"
        self.assertFalse(is_sm121_storage_candidate(model))
        model.sglang_storage_mode = None
        self.assertFalse(is_sm121_storage_candidate(model))

    def test_local_image_inspection_is_a_tag_and_source_tree_bound_identity(
        self,
    ) -> None:
        inspection = self._local_image_inspection()

        resolved = validate_sm121_storage_image_inspection(
            inspection, image=SM121_STORAGE_LOCAL_IMAGE_TAG
        )

        self.assertEqual(
            resolved["docker_image_id"], SM121_STORAGE_LOCAL_IMAGE_ID
        )
        self.assertEqual(resolved["source_tree"], SM121_STORAGE_SOURCE_TREE)
        for field, value in (
            ("Id", "sha256:" + "0" * 64),
            ("RepoTags", []),
            ("Architecture", "amd64"),
        ):
            with self.subTest(field=field):
                altered = dict(inspection)
                altered[field] = value
                with self.assertRaises(SM121StorageCandidateError):
                    validate_sm121_storage_image_inspection(
                        altered, image=SM121_STORAGE_LOCAL_IMAGE_TAG
                    )

    def test_launches_only_pinned_local_image_with_io_uring_hardening(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._write_snapshot(workspace)
            self._write_seccomp_profile(workspace)
            model = self._model()
            with (
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime.Path.home", return_value=workspace / "home"),
                patch(
                    "bench.runtime.verify_seccomp_profile_contract",
                    return_value=self._verifier_result(),
                    create=True,
                ) as verify_seccomp,
                patch(
                    "bench.runtime.secrets.token_urlsafe",
                    return_value="test-key",
                ),
                patch(
                    "bench.runtime._run",
                    side_effect=(
                        _completed(stdout=json.dumps(self._local_image_inspection())),
                        _completed(stdout="sm121-container\n"),
                    ),
                ) as run,
                patch("bench.runtime.wait_for_endpoint", return_value=1.25),
            ):
                server = start_sglang(model, workspace=workspace)

        verify_seccomp.assert_called_once()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args_list[0].args[0],
            [
                "docker",
                "image",
                "inspect",
                SM121_STORAGE_LOCAL_IMAGE_TAG,
                "--format",
                "{{json .}}",
            ],
        )
        launch = run.call_args.args[0]
        self.assertEqual(launch[:4], ["docker", "run", "--detach", "--pull=never"])
        self.assertIn("--read-only", launch)
        self.assertEqual(self._option_values(launch, "--cap-drop"), ["ALL"])
        security_options = self._option_values(launch, "--security-opt")
        self.assertTrue(
            any(option.startswith("no-new-privileges") for option in security_options)
        )
        self.assertIn(
            "seccomp="
            + str(
                (
                    workspace
                    / seccomp_profile_contract.DERIVED_PATH
                ).resolve()
            ),
            security_options,
        )
        self.assertNotIn("--privileged", launch)
        self.assertNotIn("--ipc", launch)
        self.assertNotIn("seccomp=unconfined", security_options)
        self.assertEqual(
            self._option_values(launch, "--tmpfs"),
            [
                "/tmp:rw,exec,nosuid,nodev,size=16g",
                "/root/.cache:rw,exec,nosuid,nodev,size=8g",
            ],
        )
        self.assertIn("127.0.0.1:30000:30000", launch)
        self.assertEqual(
            self._option_values(launch, "--entrypoint"), ["sglang"]
        )

        repository = (
            workspace
            / "home"
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--RadixArk--Qwen3.8-Flash-Next-NVFP4"
        )
        self.assertEqual(
            self._option_values(launch, "--volume"),
            [
                f"{repository}:/root/.cache/huggingface/hub/"
                "models--RadixArk--Qwen3.8-Flash-Next-NVFP4:ro"
            ],
        )
        env = self._option_values(launch, "--env")
        expected_env = {
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "SGLANG_RUST_BUILD_MODE=never",
            "SGLANG_QWEN4_PLE_NVME_BACKEND=io_uring",
            "SGLANG_QWEN4_PLE_NVME_QUEUE_DEPTH=512",
            "SGLANG_QWEN4_PLE_NVME_MAX_BATCH_PAGES=4096",
            "SGLANG_QWEN4_PLE_NVME_CACHE_PAGES=0",
            "SGLANG_CACHE_DIR=/tmp/sglang-cache",
            "TRITON_CACHE_DIR=/tmp/triton-cache",
            "XDG_CACHE_HOME=/tmp/xdg-cache",
            "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-cache",
            "TILELANG_CACHE_DIR=/tmp/tilelang-cache",
        }
        self.assertTrue(expected_env.issubset(env))
        self.assertIn(
            "SGLANG_QWEN4_PLE_NVME_PATH="
            "/root/.cache/huggingface/hub/"
            "models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/"
            + REVISION,
            env,
        )
        self.assertFalse(any(value.startswith("HF_TOKEN=") for value in env))
        self.assertFalse(any(value.startswith("SGLANG_QWEN4_PLE_MMAP_DIR=") for value in env))

        image_index = launch.index(SM121_STORAGE_LOCAL_IMAGE_ID)
        self.assertNotIn(SM121_STORAGE_LOCAL_IMAGE_TAG, launch)
        self.assertEqual(
            launch[image_index + 1 : image_index + 4],
            [
                "serve",
                "--model-path",
                "/root/.cache/huggingface/hub/"
                "models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/"
                + REVISION,
            ],
        )
        self.assertIn("--no-ple-offload-embedding", launch)
        self.assertNotIn("--ple-offload-embedding", launch)
        self.assertNotIn("--json-model-override-args", launch)
        self.assertFalse(
            any(argument.startswith("--speculative-") for argument in launch)
        )
        self.assertEqual(
            launch[launch.index("--attention-backend") + 1], "triton"
        )
        self.assertEqual(
            launch[launch.index("--moe-runner-backend") + 1],
            "flashinfer_cutlass",
        )
        self.assertIn("--disable-radix-cache", launch)

        self.assertEqual(server.backend, "sglang")
        self.assertEqual(server.base_url, "http://127.0.0.1:30000/v1")
        self.assertEqual(server.authorization, "Bearer test-key")
        self.assertIsNotNone(server.native_provenance)
        assert server.native_provenance is not None
        self.assertIn(SM121_STORAGE_SOURCE_TREE, server.native_provenance.values())
        self.assertIn(STORAGE_MODE, server.native_provenance.values())
        self.assertIn("io_uring", server.native_provenance.values())
        self.assertIn(SM121_STORAGE_QUEUE_DEPTH, server.native_provenance.values())
        self.assertIn(SM121_STORAGE_CACHE_PAGES, server.native_provenance.values())
        self.assertIn(
            "sha256:" + seccomp_profile_contract.DERIVED_SHA256,
            server.native_provenance.values(),
        )
        self.assertNotIn(str(workspace), str(server.native_provenance))

    def test_rejects_mutable_or_mismatched_local_image_before_docker(self) -> None:
        for resolved_image in (
            SM121_STORAGE_LOCAL_IMAGE_TAG,
            "sha256:" + "0" * 64,
        ):
            with self.subTest(resolved_image=resolved_image):
                with tempfile.TemporaryDirectory() as directory:
                    workspace = Path(directory)
                    self._write_snapshot(workspace)
                    model = self._model()
                    model.resolved_image = resolved_image
                    with (
                        patch(
                            "bench.runtime._existing_container",
                            return_value=None,
                        ),
                        patch("bench.runtime._port_is_free", return_value=True),
                        patch(
                            "bench.runtime.Path.home",
                            return_value=workspace / "home",
                        ),
                        patch("bench.runtime._run") as run,
                    ):
                        with self.assertRaises(RuntimeErrorWithContext):
                            start_sglang(model, workspace=workspace)
                    self._assert_no_docker_run(run)

    def test_storage_mode_requires_the_dedicated_canary_executor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._write_snapshot(workspace)
            model = self._model()
            model.storage_canary_authorized = False
            with (
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime.Path.home", return_value=workspace / "home"),
                patch("bench.runtime._run") as run,
            ):
                with self.assertRaisesRegex(
                    RuntimeErrorWithContext, "dedicated sm121-storage-canary command"
                ):
                    start_sglang(model, workspace=workspace)
            self._assert_no_docker_run(run)

    def test_retagged_local_image_is_rejected_before_container_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._write_snapshot(workspace)
            inspection = self._local_image_inspection()
            inspection["Id"] = "sha256:" + "0" * 64
            with (
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime.Path.home", return_value=workspace / "home"),
                patch(
                    "bench.runtime.verify_seccomp_profile_contract",
                    return_value=self._verifier_result(),
                    create=True,
                ),
                patch(
                    "bench.runtime._run",
                    return_value=_completed(stdout=json.dumps(inspection)),
                ) as run,
            ):
                with self.assertRaises(RuntimeErrorWithContext):
                    start_sglang(self._model(), workspace=workspace)

        self.assertIn(
            [
                "docker",
                "image",
                "inspect",
                SM121_STORAGE_LOCAL_IMAGE_TAG,
                "--format",
                "{{json .}}",
            ],
            [call.args[0] for call in run.call_args_list],
        )
        self._assert_no_docker_run(run)

    def test_rejects_legacy_mmap_or_overlay_fields_before_docker(self) -> None:
        mutations = (
            ("sglang_ple_mmap", True),
            ("sglang_ple_cache_mode", "readonly"),
            (
                "sglang_source_overlays",
                (
                    SimpleNamespace(
                        host_path="legacy.py",
                        container_path="/sgl-workspace/sglang/python/sglang/legacy.py",
                        digest="sha256:" + "a" * 64,
                    ),
                ),
            ),
            ("draft_source", "RadixArk/Qwen3.8-Flash-Next-Draft"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as directory:
                    workspace = Path(directory)
                    self._write_snapshot(workspace)
                    model = self._model()
                    setattr(model, field, value)
                    if field == "draft_source":
                        model.draft_revision = REVISION
                    with (
                        patch(
                            "bench.runtime._existing_container",
                            return_value=None,
                        ),
                        patch("bench.runtime._port_is_free", return_value=True),
                        patch(
                            "bench.runtime.Path.home",
                            return_value=workspace / "home",
                        ),
                        patch("bench.runtime._run") as run,
                    ):
                        with self.assertRaises(RuntimeErrorWithContext):
                            start_sglang(model, workspace=workspace)
                    self._assert_no_docker_run(run)


if __name__ == "__main__":
    unittest.main()
