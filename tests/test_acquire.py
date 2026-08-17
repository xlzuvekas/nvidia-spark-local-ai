from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from bench.acquire import (
    AcquisitionError,
    fetch_model_snapshot,
    resolve_huggingface_hub_root,
    verify_exact_snapshot,
)
from bench.manifest import ManifestError, ModelSpec, load_models, validate_model
from sparkbench import build_parser, command_fetch


ROOT = Path(__file__).resolve().parents[1]
REVISION = "0123456789abcdef0123456789abcdef01234567"


def _model(**changes: object) -> ModelSpec:
    model = ModelSpec(
        id="fixture-model",
        backend="vllm",
        source="example/model",
        revision=REVISION,
        served_name="example/model",
        tasks=("chat",),
        image="example/image:fixed",
        max_context=1024,
        lifecycle="docker",
        cache_dir="project",
        fetch_allow_patterns=("*.json", "*.safetensors"),
        fetch_ignore_patterns=("original/**",),
    )
    return replace(model, **changes)


def _write_complete_snapshot(hub_root: Path, model: ModelSpec) -> Path:
    snapshot = (
        hub_root
        / "models--example--model"
        / "snapshots"
        / str(model.revision)
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"second")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 11},
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    return snapshot


def _llamacpp_model(
    payload: bytes, *, mmproj_payload: bytes | None = None
) -> ModelSpec:
    model_file = "model.gguf"
    mmproj_file = "mmproj.gguf" if mmproj_payload is not None else None
    return _model(
        id="llamacpp-fixture",
        backend="llamacpp",
        image=None,
        lifecycle="subprocess",
        support_status="spark_other_backend",
        tasks=("chat", "vision") if mmproj_payload is not None else ("chat",),
        fetch_allow_patterns=(
            (model_file, str(mmproj_file))
            if mmproj_file is not None
            else (model_file,)
        ),
        fetch_ignore_patterns=(),
        runtime_binary="/opt/llama.cpp/llama-server",
        runtime_digest="sha256:" + "1" * 64,
        runtime_parallel=2,
        runtime_source_dir="/opt/llama.cpp",
        runtime_revision="a" * 40,
        model_file=model_file,
        model_size_bytes=len(payload),
        model_digest="sha256:" + hashlib.sha256(payload).hexdigest(),
        mmproj_file=mmproj_file,
        mmproj_size_bytes=(
            len(mmproj_payload) if mmproj_payload is not None else None
        ),
        mmproj_digest=(
            "sha256:" + hashlib.sha256(mmproj_payload).hexdigest()
            if mmproj_payload is not None
            else None
        ),
        native_context=2048,
    )


class AcquisitionTests(unittest.TestCase):
    def test_cache_roots_match_runtime_and_inventory_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            home = root / "home"
            self.assertEqual(
                resolve_huggingface_hub_root(
                    "project", workspace=workspace, home=home
                ),
                workspace / "data" / "huggingface" / "hub",
            )
            self.assertEqual(
                resolve_huggingface_hub_root(
                    "user", workspace=workspace, home=home
                ),
                home / ".cache" / "huggingface" / "hub",
            )

    def test_fetch_runs_only_pinned_hf_download_and_verifies_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            model = _model()
            hub_root = workspace / "data" / "huggingface" / "hub"

            def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                self.assertEqual(command[0:3], ["/mock/hf", "download", model.source])
                self.assertEqual(command[command.index("--revision") + 1], REVISION)
                self.assertEqual(
                    command[command.index("--cache-dir") + 1], str(hub_root)
                )
                include = command.index("--include")
                exclude = command.index("--exclude")
                self.assertEqual(command[include + 1 : exclude], list(model.fetch_allow_patterns))
                self.assertEqual(command[exclude + 1 :], list(model.fetch_ignore_patterns))
                self.assertNotIn("--token", command)
                self.assertEqual(kwargs["cwd"], workspace)
                self.assertTrue(kwargs["capture_output"])
                self.assertNotIn("shell", kwargs)
                _write_complete_snapshot(hub_root, model)
                return subprocess.CompletedProcess(command, 0, stdout="ignored", stderr="")

            with (
                patch("bench.acquire.shutil.which", return_value="/mock/hf"),
                patch("bench.acquire.subprocess.run", side_effect=run) as run_mock,
            ):
                result = fetch_model_snapshot(model, workspace=workspace)

            run_mock.assert_called_once()
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["revision"], REVISION)
            self.assertEqual(result["cache_root"], str(hub_root))
            self.assertEqual(result["weight_index_count"], 1)
            self.assertEqual(result["referenced_shard_count"], 2)
            self.assertEqual(result["weight_file_count"], 2)
            self.assertGreater(result["snapshot_bytes"], 11)

    def test_fetch_passes_the_exact_user_hub_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            home = Path(temporary) / "home"
            workspace.mkdir()
            model = _model(cache_dir="user")
            hub_root = home / ".cache" / "huggingface" / "hub"

            def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                self.assertEqual(
                    command[command.index("--cache-dir") + 1], str(hub_root)
                )
                _write_complete_snapshot(hub_root, model)
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with (
                patch("bench.acquire.shutil.which", return_value="/mock/hf"),
                patch("bench.acquire.subprocess.run", side_effect=run),
            ):
                result = fetch_model_snapshot(
                    model, workspace=workspace, home=home
                )
            self.assertEqual(result["cache_root"], str(hub_root))

    def test_fetch_accepts_sglang_hugging_face_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            model = _model(backend="sglang")
            hub_root = workspace / "data" / "huggingface" / "hub"

            def run(
                command: list[str], **_: object
            ) -> subprocess.CompletedProcess[str]:
                _write_complete_snapshot(hub_root, model)
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with (
                patch("bench.acquire.shutil.which", return_value="/mock/hf"),
                patch("bench.acquire.subprocess.run", side_effect=run),
            ):
                result = fetch_model_snapshot(model, workspace=workspace)

            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["revision"], REVISION)

    def test_sglang_fetch_acquires_and_verifies_exact_target_and_draft(self) -> None:
        draft_revision = "a" * 40
        model = _model(
            backend="sglang",
            weight_size_bytes=11,
            weight_file_count=2,
            draft_source="example/draft",
            draft_revision=draft_revision,
            draft_weight_size_bytes=5,
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            hub_root = workspace / "data" / "huggingface" / "hub"

            def run(
                command: list[str], **_: object
            ) -> subprocess.CompletedProcess[str]:
                source = command[2]
                if source == model.source:
                    self.assertEqual(
                        command[command.index("--revision") + 1], model.revision
                    )
                    self.assertIn("--include", command)
                    self.assertIn("--exclude", command)
                    _write_complete_snapshot(hub_root, model)
                else:
                    self.assertEqual(source, model.draft_source)
                    self.assertEqual(
                        command[command.index("--revision") + 1], draft_revision
                    )
                    self.assertNotIn("--include", command)
                    self.assertNotIn("--exclude", command)
                    draft = (
                        hub_root
                        / "models--example--draft"
                        / "snapshots"
                        / draft_revision
                    )
                    draft.mkdir(parents=True)
                    (draft / "model.safetensors").write_bytes(b"draft")
                self.assertNotIn("--token", command)
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with (
                patch("bench.acquire.shutil.which", return_value="/mock/hf"),
                patch("bench.acquire.subprocess.run", side_effect=run) as run_mock,
            ):
                result = fetch_model_snapshot(model, workspace=workspace)

            self.assertEqual(run_mock.call_count, 2)
            self.assertEqual(result["weight_bytes"], 11)
            self.assertEqual(result["weight_file_count"], 2)
            self.assertEqual(result["draft"]["source"], "example/draft")
            self.assertEqual(result["draft"]["revision"], draft_revision)
            self.assertEqual(result["draft"]["weight_bytes"], 5)
            with self.assertRaisesRegex(AcquisitionError, "draft weight size"):
                verify_exact_snapshot(
                    replace(model, draft_weight_size_bytes=6),
                    hub_root=hub_root,
                    draft=True,
                )

    def test_fetch_refuses_unpinned_incompatible_and_non_hf_profiles(self) -> None:
        cases = (
            (_model(revision="main"), "full 40-character"),
            (_model(support_status="incompatible"), "Incompatible profile"),
            (
                _model(
                    backend="ollama",
                    lifecycle="existing",
                    endpoint="http://127.0.0.1:11434/v1",
                ),
                "not a Hugging Face-backed profile",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for model, expected in cases:
                with self.subTest(expected=expected):
                    with patch("bench.acquire.subprocess.run") as run_mock:
                        with self.assertRaisesRegex(AcquisitionError, expected):
                            fetch_model_snapshot(model, workspace=temporary)
                    run_mock.assert_not_called()

    def test_verification_rejects_a_missing_index_referenced_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hub_root = Path(temporary)
            model = _model()
            snapshot = _write_complete_snapshot(hub_root, model)
            (snapshot / "model-00002-of-00002.safetensors").unlink()
            with self.assertRaisesRegex(AcquisitionError, "index-referenced shard"):
                verify_exact_snapshot(model, hub_root=hub_root)

    def test_verification_rejects_empty_or_unindexed_sharded_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = _model()
            repository = root / "models--example--model" / "snapshots" / REVISION
            repository.mkdir(parents=True)
            (repository / "model.safetensors").write_bytes(b"")
            with self.assertRaisesRegex(AcquisitionError, "weight file is empty"):
                verify_exact_snapshot(model, hub_root=root)

            (repository / "model.safetensors").unlink()
            (repository / "model-00001-of-00002.safetensors").write_bytes(b"one")
            with self.assertRaisesRegex(AcquisitionError, "require.*weight index"):
                verify_exact_snapshot(model, hub_root=root)

    def test_verification_rejects_malformed_weight_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hub_root = Path(temporary)
            model = _model()
            snapshot = _write_complete_snapshot(hub_root, model)
            (snapshot / "model.safetensors.index.json").write_text(
                '{"metadata": {}}', encoding="utf-8"
            )
            with self.assertRaisesRegex(AcquisitionError, "no weight_map"):
                verify_exact_snapshot(model, hub_root=hub_root)

    def test_llamacpp_fetch_verifies_exact_gguf_size_and_sha256(self) -> None:
        payload = b"manifest-pinned GGUF fixture"
        model = _llamacpp_model(payload)
        with tempfile.TemporaryDirectory() as temporary:
            hub_root = Path(temporary)
            snapshot = (
                hub_root
                / "models--example--model"
                / "snapshots"
                / REVISION
            )
            snapshot.mkdir(parents=True)
            gguf = snapshot / str(model.model_file)
            gguf.write_bytes(payload)

            verified = verify_exact_snapshot(model, hub_root=hub_root)
            self.assertEqual(verified.exact_model_file, model.model_file)
            self.assertEqual(verified.exact_model_size_bytes, len(payload))
            self.assertEqual(verified.exact_model_sha256, model.model_digest)

            with self.assertRaisesRegex(AcquisitionError, "size mismatch"):
                verify_exact_snapshot(
                    replace(model, model_size_bytes=len(payload) + 1),
                    hub_root=hub_root,
                )

            gguf.write_bytes(b"X" * len(payload))
            with self.assertRaisesRegex(AcquisitionError, "SHA-256 mismatch"):
                verify_exact_snapshot(model, hub_root=hub_root)

    def test_q5_repository_profile_fetches_only_its_pinned_gguf(self) -> None:
        repository_model = load_models(ROOT / "manifests" / "models.toml")[
            "qwen38-27b-ud-q5-k-xl-llamacpp"
        ]
        payload = b"Q5 acquisition fixture"
        model = replace(
            repository_model,
            model_size_bytes=len(payload),
            model_digest="sha256:" + hashlib.sha256(payload).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            home = root / "home"
            workspace.mkdir()
            snapshot = (
                home
                / ".cache"
                / "huggingface"
                / "hub"
                / "models--unsloth--Qwen3.8-27B-GGUF"
                / "snapshots"
                / str(model.revision)
            )

            def run(
                command: list[str], **_: object
            ) -> subprocess.CompletedProcess[str]:
                self.assertEqual(command[:3], ["/mock/hf", "download", model.source])
                revision = command.index("--revision")
                self.assertEqual(command[revision + 1], model.revision)
                cache_dir = command.index("--cache-dir")
                self.assertEqual(
                    command[cache_dir + 1],
                    str(home / ".cache" / "huggingface" / "hub"),
                )
                include = command.index("--include")
                self.assertEqual(
                    command[include + 1 :], ["Qwen3.8-27B-UD-Q5_K_XL.gguf"]
                )
                snapshot.mkdir(parents=True)
                (snapshot / str(model.model_file)).write_bytes(payload)
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with (
                patch("bench.acquire.shutil.which", return_value="/mock/hf"),
                patch("bench.acquire.subprocess.run", side_effect=run),
            ):
                result = fetch_model_snapshot(
                    model, workspace=workspace, home=home
                )

            self.assertEqual(
                result["exact_model_file"], "Qwen3.8-27B-UD-Q5_K_XL.gguf"
            )
            self.assertEqual(result["exact_model_size_bytes"], len(payload))
            self.assertEqual(result["exact_model_sha256"], model.model_digest)

    def test_llamacpp_dflash_fetches_and_verifies_exact_target_and_sidecar(self) -> None:
        repository_model = load_models(ROOT / "manifests" / "models.toml")[
            "muse-glimmer-30b-ud-q4-k-xl-llamacpp-dflash15"
        ]
        target_payload = b"Muse target acquisition fixture"
        draft_payload = b"Muse DFlash acquisition fixture"
        model = replace(
            repository_model,
            model_size_bytes=len(target_payload),
            model_digest=(
                "sha256:" + hashlib.sha256(target_payload).hexdigest()
            ),
            draft_model_size_bytes=len(draft_payload),
            draft_model_digest=(
                "sha256:" + hashlib.sha256(draft_payload).hexdigest()
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            home = root / "home"
            workspace.mkdir()
            hub = home / ".cache" / "huggingface" / "hub"
            target_snapshot = (
                hub
                / "models--unsloth--Muse-Glimmer-30B-GGUF"
                / "snapshots"
                / str(model.revision)
            )
            draft_snapshot = (
                hub
                / "models--meta-models--Muse-Glimmer-30B-GGUF"
                / "snapshots"
                / str(model.draft_revision)
            )

            def run(
                command: list[str], **_: object
            ) -> subprocess.CompletedProcess[str]:
                source = command[2]
                include = command.index("--include")
                self.assertEqual(len(command[include + 1 :]), 1)
                if source == model.source:
                    self.assertEqual(
                        command[include + 1], model.model_file
                    )
                    target_snapshot.mkdir(parents=True)
                    (target_snapshot / str(model.model_file)).write_bytes(
                        target_payload
                    )
                else:
                    self.assertEqual(source, model.draft_source)
                    self.assertEqual(
                        command[include + 1], model.draft_model_file
                    )
                    draft_snapshot.mkdir(parents=True)
                    (draft_snapshot / str(model.draft_model_file)).write_bytes(
                        draft_payload
                    )
                return subprocess.CompletedProcess(
                    command, 0, stdout="", stderr=""
                )

            with (
                patch("bench.acquire.shutil.which", return_value="/mock/hf"),
                patch("bench.acquire.subprocess.run", side_effect=run),
            ):
                result = fetch_model_snapshot(
                    model, workspace=workspace, home=home
                )

            self.assertEqual(result["exact_model_file"], model.model_file)
            self.assertEqual(
                result["exact_model_sha256"], model.model_digest
            )
            self.assertEqual(
                result["draft"]["exact_model_file"],
                model.draft_model_file,
            )
            self.assertEqual(
                result["draft"]["exact_model_sha256"],
                model.draft_model_digest,
            )

            (draft_snapshot / str(model.draft_model_file)).write_bytes(
                b"X" * len(draft_payload)
            )
            with self.assertRaisesRegex(
                AcquisitionError, "draft model SHA-256 mismatch"
            ):
                verify_exact_snapshot(model, hub_root=hub, draft=True)

    def test_llamacpp_fetch_includes_and_verifies_exact_projector(self) -> None:
        payload = b"manifest-pinned GGUF fixture"
        mmproj_payload = b"manifest-pinned projector fixture"
        model = _llamacpp_model(payload, mmproj_payload=mmproj_payload)
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            hub_root = workspace / "data" / "huggingface" / "hub"
            snapshot = (
                hub_root
                / "models--example--model"
                / "snapshots"
                / REVISION
            )

            def run(
                command: list[str], **_: object
            ) -> subprocess.CompletedProcess[str]:
                include = command.index("--include")
                self.assertEqual(
                    command[include + 1 :], list(model.fetch_allow_patterns)
                )
                snapshot.mkdir(parents=True)
                (snapshot / str(model.model_file)).write_bytes(payload)
                (snapshot / str(model.mmproj_file)).write_bytes(mmproj_payload)
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with (
                patch("bench.acquire.shutil.which", return_value="/mock/hf"),
                patch("bench.acquire.subprocess.run", side_effect=run),
            ):
                result = fetch_model_snapshot(model, workspace=workspace)

            self.assertEqual(result["exact_mmproj_file"], model.mmproj_file)
            self.assertEqual(
                result["exact_mmproj_size_bytes"], len(mmproj_payload)
            )
            self.assertEqual(
                result["exact_mmproj_sha256"], model.mmproj_digest
            )

            mmproj = snapshot / str(model.mmproj_file)
            mmproj.write_bytes(b"short")
            with self.assertRaisesRegex(AcquisitionError, "mmproj size mismatch"):
                verify_exact_snapshot(model, hub_root=hub_root)
            mmproj.write_bytes(b"X" * len(mmproj_payload))
            with self.assertRaisesRegex(AcquisitionError, "mmproj SHA-256"):
                verify_exact_snapshot(model, hub_root=hub_root)

    def test_gpt_oss_zero_based_terminal_index_layout_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hub_root = Path(temporary)
            model = _model()
            snapshot = (
                hub_root
                / "models--example--model"
                / "snapshots"
                / REVISION
            )
            snapshot.mkdir(parents=True)
            shard_names = [
                "model-00000-of-00002.safetensors",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ]
            for ordinal, shard_name in enumerate(shard_names):
                (snapshot / shard_name).write_bytes(f"shard-{ordinal}".encode())
            index_path = snapshot / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            f"layer.{ordinal}": shard_name
                            for ordinal, shard_name in enumerate(shard_names)
                        }
                    }
                ),
                encoding="utf-8",
            )

            verified = verify_exact_snapshot(model, hub_root=hub_root)
            self.assertEqual(verified.weight_file_count, 3)
            self.assertEqual(verified.referenced_shard_count, 3)

            index_path.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layer.0": shard_names[0],
                            "layer.1": shard_names[1],
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AcquisitionError, "not referenced"):
                verify_exact_snapshot(model, hub_root=hub_root)

    def test_verification_rejects_repository_symlink_outside_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            hub_root = base / "hub"
            hub_root.mkdir()
            outside_repository = base / "outside-repository"
            snapshot = outside_repository / "snapshots" / REVISION
            snapshot.mkdir(parents=True)
            (snapshot / "model.safetensors").write_bytes(b"weights")
            (hub_root / "models--example--model").symlink_to(
                outside_repository, target_is_directory=True
            )
            with self.assertRaisesRegex(AcquisitionError, "escapes.*cache root"):
                verify_exact_snapshot(_model(), hub_root=hub_root)

    def test_fetch_failure_suppresses_all_client_output(self) -> None:
        token = "hf_thismustneverappear"
        failed = subprocess.CompletedProcess(
            ["/mock/hf"],
            7,
            stdout="",
            stderr=(
                f"authorization: Bearer {token} "
                "https://person:password@example.invalid/file?token=also-secret&"
                "X-Amz-Credential=credential&X-Amz-Signature=signature"
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.dict(os.environ, {"HF_TOKEN": token}),
                patch("bench.acquire.shutil.which", return_value="/mock/hf"),
                patch("bench.acquire.subprocess.run", return_value=failed),
            ):
                with self.assertRaises(AcquisitionError) as caught:
                    fetch_model_snapshot(_model(), workspace=temporary)
        message = str(caught.exception)
        self.assertNotIn(token, message)
        self.assertNotIn("password", message)
        self.assertNotIn("also-secret", message)
        self.assertNotIn("credential", message)
        self.assertNotIn("signature", message)
        self.assertIn("client output was suppressed", message)

    def test_manifest_patterns_are_safe_and_gpt_oss_avoids_duplicate_layouts(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        gpt_oss = models["gpt-oss-20b-mxfp4"]
        self.assertEqual(gpt_oss.fetch_allow_patterns, ())
        self.assertEqual(
            gpt_oss.fetch_ignore_patterns, ("metal/**", "original/**")
        )
        with self.assertRaisesRegex(ManifestError, "unsafe pattern"):
            validate_model(_model(fetch_ignore_patterns=("../secret",)))

    def test_cli_exposes_fetch_without_inference_options(self) -> None:
        args = build_parser().parse_args(
            ["fetch", "fixture-model", "--models", "fixture-models.toml"]
        )
        self.assertIs(args.function, command_fetch)
        self.assertEqual(args.model, "fixture-model")
        self.assertEqual(args.models, Path("fixture-models.toml"))
        self.assertFalse(hasattr(args, "allow_download"))


if __name__ == "__main__":
    unittest.main()
