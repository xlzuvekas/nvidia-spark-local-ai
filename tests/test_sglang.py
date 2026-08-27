from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.inventory import (
    DockerImage,
    HuggingFaceSnapshot,
    Inventory,
    assess_model_availability,
)
from bench.journal import Journal
from bench.manifest import (
    ManifestError,
    SGLangSourceOverlay,
    load_models,
    validate_model,
)
from bench.runner import _recover_pending_lifecycle, _request_arguments
from bench.qwen38_ple_cache import PLECacheRecord
from bench.runtime import (
    ManagedServer,
    RuntimeErrorWithContext,
    capture_server_provenance,
    recover_owned_sglang,
    save_server_logs,
    start_server,
    start_sglang,
    _validate_readonly_sglang_ple_loader,
)


ROOT = Path(__file__).resolve().parents[1]


def _completed(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


class SGLangRuntimeTests(unittest.TestCase):
    def _model(self) -> SimpleNamespace:
        digest = (
            "sha256:3f51e3b127bd0fe8f261a84c6ad54ce"
            "42bdb65eb2e57e228a9f6359e89bd08ec"
        )
        return SimpleNamespace(
            backend="sglang",
            id="phi-4-multimodal-instruct-nvfp4",
            source="nvidia/Phi-4-multimodal-instruct-NVFP4",
            revision="617cfabb9ad6c2c6e318fd21c1961536b84f65a1",
            served_name="nvidia/Phi-4-multimodal-instruct-NVFP4",
            image="scitrera/dgx-spark-sglang:0.5.10rc0",
            image_digest=digest,
            resolved_image=f"scitrera/dgx-spark-sglang@{digest}",
            cache_dir="project",
            startup_timeout_s=1200,
            run_identity="run-1",
            args=[
                "--served-model-name",
                "nvidia/Phi-4-multimodal-instruct-NVFP4",
                "--host",
                "0.0.0.0",
                "--port",
                "30000",
                "--quantization",
                "modelopt_fp4",
            ],
        )

    def _dspark_model(self) -> SimpleNamespace:
        profile = replace(
            load_models(ROOT / "manifests" / "models.toml")[
                "qwen38-27b-nvfp4-dspark-sglang"
            ],
            cache_dir="project",
        )
        return SimpleNamespace(
            **asdict(profile),
            resolved_image=profile.image,
            run_identity="run-dspark",
        )

    def _write_dspark_snapshots(
        self, workspace: Path, model: SimpleNamespace
    ) -> None:
        for source, revision in (
            (model.source, model.revision),
            (model.draft_source, model.draft_revision),
        ):
            snapshot = (
                workspace
                / "data"
                / "huggingface"
                / "hub"
                / ("models--" + source.replace("/", "--"))
                / "snapshots"
                / revision
            )
            snapshot.mkdir(parents=True)

    def test_flash_next_profile_pins_overlays_ple_and_recipe(self) -> None:
        profile = load_models(ROOT / "manifests" / "models.toml")[
            "qwen38-flash-next-nvfp4-mtp-sglang"
        ]
        self.assertEqual(
            profile.revision,
            "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
        )
        self.assertEqual(
            profile.recipe_revision,
            "bf2b7c75870d3703730b6bd8f3bb93dc622c278d",
        )
        self.assertTrue(profile.sglang_ple_mmap)
        self.assertEqual(profile.sglang_ple_cache_mode, "readonly")
        self.assertEqual(
            profile.sglang_ple_cache_marker_digest,
            "sha256:f0ef55e4e4dec9b6b936a42af4ca2eb9b2f24ced373b1e216f7a6d507b171665",
        )
        self.assertEqual(
            profile.sglang_ple_cache_payload_digest,
            "sha256:b070f9644adf93794d8a1030584ab705809387e64396a9327a68fa3a3a6666b3",
        )
        self.assertEqual(profile.estimated_ram_gib, 105.0)
        self.assertEqual(
            profile.args[profile.args.index("--mem-fraction-static") + 1],
            "0.85",
        )
        self.assertIn("--weight-loader-drop-cache-after-load", profile.args)
        self.assertEqual(len(profile.sglang_source_overlays), 2)
        self.assertTrue(
            all(
                isinstance(overlay, SGLangSourceOverlay)
                for overlay in profile.sglang_source_overlays
            )
        )
        self.assertEqual(
            {overlay.digest for overlay in profile.sglang_source_overlays},
            {
                "sha256:0b513b4dc4f2394f6b1733bb0b74fa40ab59f4a04f6b33601350b2a606c67804",
                "sha256:e30566492e1502f94a4c7fed42d90b523bbb662580c628459e6e63c7b5263c75",
            },
        )

    def test_manifest_rejects_unsafe_or_unprovenanced_sglang_overlays(
        self,
    ) -> None:
        profile = load_models(ROOT / "manifests" / "models.toml")[
            "qwen38-flash-next-nvfp4-mtp-sglang"
        ]
        valid_overlay = profile.sglang_source_overlays[0]
        with self.assertRaisesRegex(ManifestError, "safe relative path"):
            validate_model(
                replace(
                    profile,
                    sglang_source_overlays=(
                        replace(valid_overlay, host_path="../qwen4_exp.py"),
                    ),
                )
            )
        with self.assertRaisesRegex(ManifestError, "must be beneath"):
            validate_model(
                replace(
                    profile,
                    sglang_source_overlays=(
                        replace(
                            valid_overlay,
                            container_path="/tmp/qwen4_exp.py",
                        ),
                    ),
                )
            )
        with self.assertRaisesRegex(ManifestError, "recipe provenance"):
            validate_model(
                replace(
                    profile,
                    recipe_source=None,
                    recipe_revision=None,
                )
            )
        with self.assertRaisesRegex(ManifestError, "explicit.*cache_mode"):
            validate_model(replace(profile, sglang_ple_cache_mode=None))
        with self.assertRaisesRegex(ManifestError, "must be 'readonly'"):
            validate_model(
                replace(profile, sglang_ple_cache_mode="automatic")
            )
        with self.assertRaisesRegex(ManifestError, "sglang_ple_mmap=true"):
            validate_model(replace(profile, sglang_ple_mmap=False))

    def test_launch_is_offline_digest_pinned_and_snapshot_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            model = self._model()
            snapshot = (
                workspace
                / "data"
                / "huggingface"
                / "hub"
                / "models--nvidia--Phi-4-multimodal-instruct-NVFP4"
                / "snapshots"
                / model.revision
            )
            snapshot.mkdir(parents=True)
            with (
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime.wait_for_endpoint", return_value=2.5) as wait,
                patch(
                    "bench.runtime.secrets.token_urlsafe",
                    return_value="-generic-ephemeral-key",
                ),
                patch(
                    "bench.runtime._run",
                    return_value=_completed(stdout="container-id\n"),
                ) as run,
            ):
                server = start_sglang(
                    model, workspace=workspace, allow_download=True
                )

        launch = run.call_args.args[0]
        self.assertEqual(launch[:4], ["docker", "run", "--detach", "--pull=never"])
        self.assertIn("127.0.0.1:30000:30000", launch)
        self.assertIn("ai.sparkbench.managed=true", launch)
        self.assertIn("ai.sparkbench.run=run-1", launch)
        self.assertIn("ai.sparkbench.backend=sglang", launch)
        self.assertIn("HF_HUB_OFFLINE=1", launch)
        self.assertIn("TRANSFORMERS_OFFLINE=1", launch)
        self.assertNotIn("HF_TOKEN", launch)
        repository = (
            workspace
            / "data"
            / "huggingface"
            / "hub"
            / "models--nvidia--Phi-4-multimodal-instruct-NVFP4"
        )
        self.assertIn(
            f"{repository}:/root/.cache/huggingface/hub/"
            "models--nvidia--Phi-4-multimodal-instruct-NVFP4:ro",
            launch,
        )
        self.assertIn("HF_HOME=/tmp/sparkbench-hf", launch)
        self.assertIn("HF_HUB_CACHE=/tmp/sparkbench-hf/hub", launch)
        self.assertIn(
            "HF_TOKEN_PATH=/tmp/sparkbench-hf/token-disabled", launch
        )
        self.assertIn("HF_HUB_DISABLE_IMPLICIT_TOKEN=1", launch)
        self.assertIn("--api-key=-generic-ephemeral-key", launch)
        self.assertNotIn("-generic-ephemeral-key", launch)
        image_index = launch.index(model.resolved_image)
        self.assertEqual(
            launch[image_index + 1 : image_index + 4],
            ["serve", "--model-path", (
                "/root/.cache/huggingface/hub/"
                "models--nvidia--Phi-4-multimodal-instruct-NVFP4/"
                f"snapshots/{model.revision}"
            )],
        )
        self.assertEqual(
            launch[launch.index("--quantization") + 1], "modelopt_fp4"
        )
        self.assertEqual(server.backend, "sglang")
        self.assertEqual(server.base_url, "http://127.0.0.1:30000/v1")
        self.assertEqual(
            server.authorization, "Bearer -generic-ephemeral-key"
        )
        wait.assert_called_once_with(
            server.base_url,
            1200.0,
            "container-id",
            authorization="Bearer -generic-ephemeral-key",
            sensitive_values=(
                "-generic-ephemeral-key",
                "Bearer -generic-ephemeral-key",
            ),
        )

    def test_source_overlays_and_ple_mmap_are_verified_and_minimally_mounted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            model = self._model()
            snapshot = (
                workspace
                / "data"
                / "huggingface"
                / "hub"
                / "models--nvidia--Phi-4-multimodal-instruct-NVFP4"
                / "snapshots"
                / model.revision
            )
            snapshot.mkdir(parents=True)
            overlay_root = workspace / "results" / "runtime-overlays" / "recipe"
            model_overlay = overlay_root / "qwen4_exp.py"
            backend_overlay = overlay_root / "qwen_sparse_attn_backend.py"
            overlay_root.mkdir(parents=True)
            model_overlay.write_text("PLE_PATCH = True\n", encoding="utf-8")
            backend_overlay.write_text("QSA_PATCH = True\n", encoding="utf-8")

            def digest(path: Path) -> str:
                return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

            model.sglang_source_overlays = (
                SimpleNamespace(
                    host_path=(
                        "results/runtime-overlays/recipe/qwen4_exp.py"
                    ),
                    container_path=(
                        "/sgl-workspace/sglang/python/sglang/srt/models/"
                        "qwen4_exp.py"
                    ),
                    digest=digest(model_overlay),
                ),
                SimpleNamespace(
                    host_path=(
                        "results/runtime-overlays/recipe/"
                        "qwen_sparse_attn_backend.py"
                    ),
                    container_path=(
                        "/sgl-workspace/sglang/python/sglang/srt/layers/"
                        "attention/quantized_sparse_attention/"
                        "qwen_sparse_attn_backend.py"
                    ),
                    digest=digest(backend_overlay),
                ),
            )
            model.sglang_ple_mmap = True
            with (
                patch("bench.runtime.Path.home", return_value=workspace),
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime.wait_for_endpoint", return_value=2.5),
                patch(
                    "bench.runtime.secrets.token_urlsafe",
                    return_value="overlay-ephemeral-key",
                ),
                patch(
                    "bench.runtime._run",
                    return_value=_completed(stdout="overlay-container\n"),
                ) as run,
            ):
                server = start_sglang(model, workspace=workspace)

            launch = run.call_args.args[0]
            target_model = (
                "/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py"
            )
            target_backend = (
                "/sgl-workspace/sglang/python/sglang/srt/layers/attention/"
                "quantized_sparse_attention/qwen_sparse_attn_backend.py"
            )
            self.assertIn(f"{model_overlay}:{target_model}:ro", launch)
            self.assertIn(f"{backend_overlay}:{target_backend}:ro", launch)
            ple_dir = (
                workspace
                / ".cache"
                / "sparkbench"
                / "sglang"
                / model.id
                / "ple"
            )
            self.assertTrue(ple_dir.is_dir())
            self.assertEqual(os.stat(ple_dir).st_mode & 0o777, 0o700)
            self.assertIn(f"{ple_dir}:/ple:rw", launch)
            self.assertIn("SGLANG_QWEN4_PLE_MMAP_DIR=/ple", launch)
            writable_mounts = [
                launch[index + 1]
                for index, value in enumerate(launch)
                if value == "--volume" and launch[index + 1].endswith(":rw")
            ]
            self.assertEqual(writable_mounts, [f"{ple_dir}:/ple:rw"])
            assert server.native_provenance is not None
            self.assertTrue(server.native_provenance["sglang_ple_mmap"])
            self.assertEqual(
                server.native_provenance["sglang_ple_backing_policy"],
                "private_runtime_cache",
            )
            serialized = json.dumps(server.native_provenance)
            self.assertNotIn(str(workspace), serialized)
            self.assertIn(digest(model_overlay), serialized)

    def test_explicit_readonly_ple_cache_is_verified_and_mounted_ro(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            profile = load_models(ROOT / "manifests" / "models.toml")[
                "qwen38-flash-next-nvfp4-mtp-sglang"
            ]
            model = SimpleNamespace(
                **asdict(replace(profile, cache_dir="project")),
                resolved_image=profile.image,
                run_identity="readonly-ple-run",
            )
            snapshot = (
                workspace
                / "data"
                / "huggingface"
                / "hub"
                / "models--RadixArk--Qwen3.8-Flash-Next-NVFP4"
                / "snapshots"
                / model.revision
            )
            snapshot.mkdir(parents=True)
            ple_dir = workspace / "completed-ple"
            ple_dir.mkdir()
            record = PLECacheRecord(
                marker_sha256=model.sglang_ple_cache_marker_digest.removeprefix(
                    "sha256:"
                ),
                payload_sha256=model.sglang_ple_cache_payload_digest.removeprefix(
                    "sha256:"
                ),
                payload_size_bytes=51_200_245_760,
            )
            resolved_overlays = tuple(
                (
                    workspace / Path(overlay.host_path).name,
                    overlay.container_path,
                    overlay.digest,
                    overlay.host_path,
                )
                for overlay in profile.sglang_source_overlays
            )
            with (
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch(
                    "bench.runtime._resolve_sglang_source_overlays",
                    return_value=resolved_overlays,
                ),
                patch(
                    "bench.runtime._readonly_sglang_ple_dir",
                    return_value=(ple_dir, record),
                ) as admit,
                patch("bench.runtime.wait_for_endpoint", return_value=1.0),
                patch(
                    "bench.runtime.secrets.token_urlsafe",
                    return_value="readonly-ephemeral-key",
                ),
                patch(
                    "bench.runtime._run",
                    return_value=_completed(stdout="readonly-container\n"),
                ) as run,
            ):
                server = start_sglang(model, workspace=workspace)

            admit.assert_called_once_with(model, resolved_overlays)
            launch = run.call_args.args[0]
            self.assertIn(f"{ple_dir}:/ple:ro", launch)
            self.assertNotIn(f"{ple_dir}:/ple:rw", launch)
            self.assertIn("SGLANG_QWEN4_PLE_CACHE_MODE=readonly", launch)
            self.assertIn(
                "SGLANG_QWEN4_PLE_CACHE_LOADER_CONTRACT="
                "auto-mmap-no-prefetch",
                launch,
            )
            self.assertIn(
                "SGLANG_QWEN4_PLE_CACHE_MARKER_SHA256="
                + record.marker_sha256,
                launch,
            )
            assert server.native_provenance is not None
            self.assertEqual(
                server.native_provenance["sglang_ple_backing_policy"],
                "verified_persistent_readonly",
            )
            self.assertEqual(
                server.native_provenance["sglang_ple_cache_payload_digest"],
                model.sglang_ple_cache_payload_digest,
            )

    def test_readonly_ple_rejects_nondefault_weight_loader_modes(self) -> None:
        good = SimpleNamespace(args=["--ple-offload-embedding"])
        _validate_readonly_sglang_ple_loader(good)
        _validate_readonly_sglang_ple_loader(
            SimpleNamespace(
                args=["--ple-offload-embedding", "--load-format=auto"]
            )
        )
        incompatible = (
            ["--ple-offload-embedding", "--load-format", "fastsafetensors"],
            ["--ple-offload-embedding", "--weight-loader-disable-mmap"],
            [
                "--ple-offload-embedding",
                "--weight-loader-prefetch-checkpoints",
            ],
        )
        for arguments in incompatible:
            with self.subTest(arguments=arguments), self.assertRaises(
                RuntimeErrorWithContext
            ):
                _validate_readonly_sglang_ple_loader(
                    SimpleNamespace(args=arguments)
                )
        with self.assertRaisesRegex(
            RuntimeErrorWithContext, "ple-offload-embedding"
        ):
            _validate_readonly_sglang_ple_loader(SimpleNamespace(args=[]))

    def test_source_overlay_digest_mismatch_fails_before_docker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            model = self._model()
            snapshot = (
                workspace
                / "data"
                / "huggingface"
                / "hub"
                / "models--nvidia--Phi-4-multimodal-instruct-NVFP4"
                / "snapshots"
                / model.revision
            )
            snapshot.mkdir(parents=True)
            overlay = workspace / "results" / "overlay" / "qwen4_exp.py"
            overlay.parent.mkdir(parents=True)
            overlay.write_text("changed\n", encoding="utf-8")
            model.sglang_source_overlays = (
                SimpleNamespace(
                    host_path="results/overlay/qwen4_exp.py",
                    container_path=(
                        "/sgl-workspace/sglang/python/sglang/srt/models/"
                        "qwen4_exp.py"
                    ),
                    digest="sha256:" + "0" * 64,
                ),
            )
            with (
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime._run") as run,
            ):
                with self.assertRaisesRegex(
                    RuntimeErrorWithContext, "digest mismatch"
                ):
                    start_sglang(model, workspace=workspace)
            run.assert_not_called()

    def test_source_overlay_symlink_fails_before_docker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            model = self._model()
            snapshot = (
                workspace
                / "data"
                / "huggingface"
                / "hub"
                / "models--nvidia--Phi-4-multimodal-instruct-NVFP4"
                / "snapshots"
                / model.revision
            )
            snapshot.mkdir(parents=True)
            real = workspace / "real.py"
            real.write_text("patched\n", encoding="utf-8")
            overlay_dir = workspace / "results" / "overlay"
            overlay_dir.mkdir(parents=True)
            overlay = overlay_dir / "qwen4_exp.py"
            overlay.symlink_to(real)
            model.sglang_source_overlays = (
                SimpleNamespace(
                    host_path="results/overlay/qwen4_exp.py",
                    container_path=(
                        "/sgl-workspace/sglang/python/sglang/srt/models/"
                        "qwen4_exp.py"
                    ),
                    digest=(
                        "sha256:" + hashlib.sha256(real.read_bytes()).hexdigest()
                    ),
                ),
            )
            with (
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime._run") as run,
            ):
                with self.assertRaisesRegex(
                    RuntimeErrorWithContext, "symbolic link"
                ):
                    start_sglang(model, workspace=workspace)
            run.assert_not_called()

    def test_unmanaged_named_container_is_never_replaced(self) -> None:
        with patch(
            "bench.runtime._existing_container", return_value=("abc", False, "")
        ):
            with self.assertRaisesRegex(
                RuntimeErrorWithContext, "unmanaged container sparkbench-sglang"
            ):
                start_sglang(self._model(), workspace=Path("/unused"))

    def test_request_arguments_thread_only_managed_authorization(self) -> None:
        model = SimpleNamespace(served_name="served", request_body_json=None)
        case = SimpleNamespace(
            id="chat-smoke",
            kind="decode",
            requires=["chat"],
            prompt_repetitions=0,
            max_output_tokens=8,
            temperature=0.0,
        )
        authenticated = ManagedServer(
            backend="sglang",
            base_url="http://127.0.0.1:30000/v1",
            authorization="Bearer private-key",
        )
        arguments = _request_arguments(
            server=authenticated,
            model=model,
            case=case,
            request_id="auth-request",
        )
        self.assertEqual(arguments["authorization"], "Bearer private-key")

        unchanged = _request_arguments(
            server=ManagedServer(
                backend="vllm", base_url="http://127.0.0.1:8000/v1"
            ),
            model=model,
            case=case,
            request_id="plain-request",
        )
        self.assertNotIn("authorization", unchanged)

    def test_dspark_launch_is_exact_capped_loopback_and_provenance_visible(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            model = self._dspark_model()
            self._write_dspark_snapshots(workspace, model)
            host_hf_root = workspace / "data" / "huggingface"
            (host_hf_root / "token").write_text(
                "host-token-must-not-be-mounted", encoding="utf-8"
            )
            unrelated = host_hf_root / "hub" / "models--private--unrelated"
            unrelated.mkdir(parents=True)
            server_log = workspace / "results" / "run" / "server" / "server.log"
            with (
                patch("bench.runtime.Path.home", return_value=workspace),
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime.wait_for_endpoint", return_value=8.0) as wait,
                patch(
                    "bench.runtime.secrets.token_urlsafe",
                    return_value="dspark-ephemeral-key",
                ),
                patch(
                    "bench.runtime._run",
                    return_value=_completed(stdout="dspark-container\n"),
                ) as run,
            ):
                server = start_sglang(
                    model,
                    workspace=workspace,
                    allow_download=True,
                    server_log_path=server_log,
                )

            launch = run.call_args.args[0]
            self.assertEqual(
                launch[:4], ["docker", "run", "--detach", "--pull=never"]
            )
            self.assertEqual(launch[launch.index("--memory") + 1], "100g")
            self.assertEqual(launch[launch.index("--memory-swap") + 1], "100g")
            self.assertEqual(launch[launch.index("--shm-size") + 1], "16g")
            self.assertNotIn("--network", launch)
            self.assertNotIn("--ipc", launch)
            self.assertIn("127.0.0.1:30000:30000", launch)
            self.assertIn("ai.sparkbench.managed=true", launch)
            self.assertIn("ai.sparkbench.run=run-dspark", launch)
            self.assertIn("ai.sparkbench.backend=sglang", launch)
            self.assertNotIn("HF_HUB_OFFLINE=1", launch)
            self.assertNotIn("TRANSFORMERS_OFFLINE=1", launch)
            self.assertIn("HF_HUB_DISABLE_TELEMETRY=1", launch)
            self.assertIn("HF_HUB_DISABLE_IMPLICIT_TOKEN=1", launch)
            self.assertIn("HF_HOME=/tmp/sparkbench-hf", launch)
            self.assertIn("HF_HUB_CACHE=/tmp/sparkbench-hf/hub", launch)
            self.assertIn(
                "HF_TOKEN_PATH=/tmp/sparkbench-hf/token-disabled", launch
            )
            joined = " ".join(launch)
            for secret_or_proxy in (
                "HF_TOKEN=",
                "HUGGING_FACE_HUB_TOKEN=",
                "HTTP_PROXY",
                "HTTPS_PROXY",
            ):
                self.assertNotIn(secret_or_proxy, joined)
            self.assertNotIn(str(host_hf_root / "token"), joined)
            self.assertNotIn("host-token-must-not-be-mounted", joined)
            self.assertNotIn(str(unrelated), joined)
            self.assertNotIn(
                f"{host_hf_root}:/root/.cache/huggingface:ro", launch
            )
            for source in (model.source, model.draft_source):
                repository_name = "models--" + source.replace("/", "--")
                self.assertIn(
                    f"{host_hf_root / 'hub' / repository_name}:"
                    f"/root/.cache/huggingface/hub/{repository_name}:ro",
                    launch,
                )
            repository_mounts = [
                launch[index + 1]
                for index, value in enumerate(launch)
                if value == "--volume"
                and "/root/.cache/huggingface/hub/models--" in launch[index + 1]
            ]
            self.assertEqual(len(repository_mounts), 2)
            self.assertIn("--api-key=dspark-ephemeral-key", launch)
            self.assertNotIn("dspark-ephemeral-key", launch)
            self.assertEqual(
                server.authorization, "Bearer dspark-ephemeral-key"
            )
            key_path = server_log.parent / "api-key"
            self.assertEqual(key_path.read_text(), "dspark-ephemeral-key\n")
            self.assertEqual(os.stat(key_path).st_mode & 0o777, 0o600)
            wait.assert_called_once_with(
                server.base_url,
                1800.0,
                "dspark-container",
                authorization="Bearer dspark-ephemeral-key",
                sensitive_values=(
                    "dspark-ephemeral-key",
                    "Bearer dspark-ephemeral-key",
                ),
            )
            compile_cache = (
                workspace
                / ".cache"
                / "sparkbench"
                / "sglang"
                / model.id
                / "compile"
            )
            self.assertTrue(compile_cache.is_dir())
            self.assertIn(f"{compile_cache}:/cache", launch)
            self.assertIn("TORCHINDUCTOR_CACHE_DIR=/cache/inductor", launch)
            image_index = launch.index(model.resolved_image)
            self.assertEqual(
                launch[image_index + 1 : image_index + 4],
                ["-m", "sglang.launch_server", "--model-path"],
            )
            self.assertEqual(
                launch[image_index + 4],
                "/root/.cache/huggingface/hub/"
                "models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/"
                "52d1adc5f38aa5ebf099c29ed7025ba34cfbb854",
            )
            draft_index = launch.index("--speculative-draft-model-path")
            self.assertEqual(
                launch[draft_index + 1],
                "/root/.cache/huggingface/hub/"
                "models--RadixArk--Qwen3.8-27B-DSpark/snapshots/"
                "923ed3a8572615643f0137e424e4ce4edd7f1cda",
            )
            self.assertEqual(server.base_url, "http://127.0.0.1:30000/v1")
            assert server.native_provenance is not None
            self.assertEqual(
                server.native_provenance["hf_network_policy"],
                "documented_longcat_metadata_probe",
            )
            self.assertEqual(
                server.native_provenance["recipe_revision"],
                "3590fb29296b1babd85405daad1eef1c4a3ebe0f",
            )
            self.assertEqual(
                server.native_provenance["benchmark_scope"],
                "sparkbench_suite_not_upstream_battery",
            )
            self.assertEqual(
                server.native_provenance["api_authentication"],
                "ephemeral_bearer",
            )

    def test_dspark_key_is_redacted_and_private_until_owned_stop(self) -> None:
        key = "private-dspark-ephemeral-key"
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            model = self._dspark_model()
            self._write_dspark_snapshots(workspace, model)
            server_log = workspace / "results" / "run" / "server" / "server.log"
            with (
                patch("bench.runtime.Path.home", return_value=workspace),
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime.wait_for_endpoint", return_value=8.0),
                patch("bench.runtime.secrets.token_urlsafe", return_value=key),
                patch(
                    "bench.runtime._run",
                    return_value=_completed(stdout="dspark-container\n"),
                ),
            ):
                server = start_sglang(
                    model,
                    workspace=workspace,
                    server_log_path=server_log,
                )

            key_path = server_log.parent / "api-key"
            inspect_payload = json.dumps(
                [
                    {
                        "Id": "dspark-container",
                        "Image": "sha256:image",
                        "Config": {
                            "Entrypoint": ["python3"],
                            "Cmd": [
                                "-m",
                                "sglang.launch_server",
                                "--api-key",
                                key,
                            ],
                        },
                    }
                ]
            )
            with patch(
                "bench.runtime._run",
                return_value=_completed(stdout=inspect_payload),
            ):
                provenance = capture_server_provenance(server)
            serialized = json.dumps(provenance)
            self.assertNotIn(key, serialized)
            self.assertIn("<redacted>", serialized)
            journal_path = workspace / "results" / "run" / "events.jsonl"
            Journal(journal_path).append(
                {"event": "server_provenance", **provenance}
            )
            self.assertNotIn(key, journal_path.read_text())

            with patch(
                "bench.runtime._run",
                return_value=_completed(
                    stdout=f"startup configuration api_key={key}\n"
                ),
            ):
                save_server_logs(server, server_log)
            persisted = server_log.read_text()
            self.assertNotIn(key, persisted)
            self.assertIn("<redacted>", persisted)

            with patch("bench.runtime._run") as run:
                server.stop(keep_server=True)
            run.assert_not_called()
            self.assertTrue(key_path.is_file())

            with patch(
                "bench.runtime._run",
                side_effect=[
                    _completed(stdout="true run-dspark\n"),
                    _completed(),
                    _completed(),
                ],
            ):
                server.stop()
            self.assertFalse(key_path.exists())
            self.assertIsNone(server.authorization)
            self.assertIsNone(server.api_key)

    def test_dspark_startup_failure_removes_only_the_owned_container(self) -> None:
        key = "startup-error-secret-key"
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            model = self._dspark_model()
            self._write_dspark_snapshots(workspace, model)
            server_log = workspace / "results" / "run" / "server" / "server.log"

            def runtime_command(
                command: list[str], **_: object
            ) -> subprocess.CompletedProcess[str]:
                if command[:2] == ["docker", "run"]:
                    return _completed(stdout="dspark-container\n")
                if command[:2] == ["docker", "inspect"]:
                    return _completed(stdout="true run-dspark\n")
                if command[:2] == ["docker", "logs"]:
                    return _completed(stdout=f"server echoed {key}\n")
                return _completed()

            with (
                patch("bench.runtime.Path.home", return_value=workspace),
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch(
                    "bench.runtime.wait_for_endpoint",
                    side_effect=RuntimeErrorWithContext(
                        f"startup failed with {key}"
                    ),
                ),
                patch("bench.runtime.secrets.token_urlsafe", return_value=key),
                patch("bench.runtime._run", side_effect=runtime_command) as run,
            ):
                with self.assertRaises(RuntimeErrorWithContext) as raised:
                    start_sglang(
                        model,
                        workspace=workspace,
                        server_log_path=server_log,
                    )

            commands = [call.args[0] for call in run.call_args_list]
            self.assertIn(
                ["docker", "stop", "--time", "30", "dspark-container"],
                commands,
            )
            self.assertIn(["docker", "rm", "dspark-container"], commands)
            self.assertTrue(
                (
                    workspace
                    / ".cache"
                    / "sparkbench"
                    / "sglang"
                    / model.id
                    / "compile"
                ).is_dir()
            )
            self.assertNotIn(key, str(raised.exception))
            self.assertNotIn(key, server_log.read_text())
            self.assertIn("<redacted>", server_log.read_text())
            self.assertFalse((server_log.parent / "api-key").exists())

    def test_dspark_rejects_a_mismatched_resolved_image_before_docker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            model = self._dspark_model()
            model.resolved_image = "lmsysorg/sglang@sha256:" + "0" * 64
            self._write_dspark_snapshots(workspace, model)
            with (
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime._run") as run,
            ):
                with self.assertRaisesRegex(
                    RuntimeErrorWithContext, "does not match"
                ):
                    start_sglang(model, workspace=workspace)
            run.assert_not_called()

    def test_inventory_requires_both_exact_dspark_snapshots(self) -> None:
        profile = replace(
            load_models(ROOT / "manifests" / "models.toml")[
                "qwen38-27b-nvfp4-dspark-sglang"
            ],
            cache_dir="project",
        )
        target = HuggingFaceSnapshot(
            source=profile.source,
            revision=str(profile.revision),
            path=Path("/mock/project/target"),
        )
        draft = HuggingFaceSnapshot(
            source=str(profile.draft_source),
            revision=str(profile.draft_revision),
            path=Path("/mock/project/draft"),
        )
        image = DockerImage(
            repository="lmsysorg/sglang",
            tag="qwen38-27b",
            digest=profile.image_digest,
            image_id="sha256:image",
        )

        def inventory(*snapshots: HuggingFaceSnapshot) -> Inventory:
            return Inventory(
                collected_at="now",
                python_version="3",
                platform="test",
                machine="aarch64",
                huggingface_snapshots=snapshots,
                docker_images=(image,),
                ollama_models=(),
            )

        missing = assess_model_availability(
            {profile.id: profile}, inventory(target)
        )[profile.id]
        self.assertFalse(missing.source_available)
        self.assertIn("draft checkpoint revision is not cached", missing.details)
        complete = assess_model_availability(
            {profile.id: profile}, inventory(target, draft)
        )[profile.id]
        self.assertTrue(complete.available)

    def test_recovery_never_stops_a_differently_owned_sglang_container(self) -> None:
        with (
            patch(
                "bench.runtime._existing_container",
                return_value=("container-id", True, "different-run"),
            ),
            patch("bench.runtime._run") as run,
        ):
            status = recover_owned_sglang("run-dspark")
        self.assertEqual(status, "different_container_present")
        run.assert_not_called()

    def test_recovery_removes_private_key_after_owned_server_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            key_path = Path(directory) / "server" / "api-key"
            key_path.parent.mkdir()
            key_path.write_text("ephemeral-key\n", encoding="utf-8")
            key_path.chmod(0o600)
            with patch("bench.runtime._existing_container", return_value=None):
                status = recover_owned_sglang(
                    "run-dspark", api_key_path=key_path
                )
            self.assertEqual(status, "already_absent")
            self.assertFalse(key_path.exists())

    def test_pre_ready_recovery_forwards_private_key_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            journal = Journal(run_dir / "events.jsonl")
            journal.append({"event": "run_start"})
            model = SimpleNamespace(backend="sglang", run_identity="run-dspark")
            key_path = run_dir / "server" / "api-key"
            with patch(
                "bench.runner.recover_owned_sglang",
                return_value="stopped_owned_container",
            ) as recover:
                changed = _recover_pending_lifecycle(
                    model=model,
                    journal=journal,
                    run_dir=run_dir,
                    workspace=run_dir,
                )
            self.assertTrue(changed)
            recover.assert_called_once_with(
                "run-dspark", api_key_path=key_path
            )

    def test_dispatch_forwards_append_safe_log_path(self) -> None:
        model = self._model()
        log_path = Path("/mock/run/server/server.log")
        expected = object()
        with patch("bench.runtime.start_sglang", return_value=expected) as start:
            actual = start_server(
                model,
                workspace=Path("/mock/workspace"),
                allow_download=True,
                server_log_path=log_path,
            )
        self.assertIs(actual, expected)
        start.assert_called_once_with(
            model,
            workspace=Path("/mock/workspace"),
            allow_download=True,
            server_log_path=log_path,
        )

    def test_owned_sglang_stop_checks_exact_run_label(self) -> None:
        server = ManagedServer(
            backend="sglang",
            base_url="http://127.0.0.1:30000/v1",
            container_id="container-id",
            run_identity="run-1",
        )
        with patch(
            "bench.runtime._run",
            side_effect=[
                _completed(stdout="true run-1\n"),
                _completed(),
                _completed(),
            ],
        ) as run:
            server.stop()

        self.assertEqual(
            [call.args[0][:2] for call in run.call_args_list],
            [["docker", "inspect"], ["docker", "stop"], ["docker", "rm"]],
        )
        self.assertIsNone(server.container_id)


if __name__ == "__main__":
    unittest.main()
