from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.densespark import (
    DENSESPARK_C1_ARGS,
    DENSESPARK_C1_ENVIRONMENT,
    DENSESPARK_LOCAL_IMAGE_ID,
    DENSESPARK_PROFILE_ID,
    DENSESPARK_SUITE_ID,
    DENSESPARK_TOOL_SUITE_ID,
    DENSESPARK_WARMUP_SYNC_IMAGE,
    DENSESPARK_WARMUP_SYNC_LOCAL_IMAGE_ID,
    DENSESPARK_WARMUP_SYNC_MODE,
    DENSESPARK_WARMUP_SYNC_PROFILE_ID,
    DENSESPARK_WARMUP_SYNC_PROBE_SHA256,
    DENSESPARK_WEIGHT_FILE_COUNT,
    DENSESPARK_WEIGHT_SIZE_BYTES,
    DENSESPARK_MODEL_REVISION,
    DENSESPARK_PQ_SHA256,
    DENSESPARK_PQ_SIZE_BYTES,
    DenseSparkContractError,
    DenseSparkPQArtifact,
    DenseSparkSnapshotReceipt,
    densespark_c1_cache_config,
    densespark_cache_namespace,
    densespark_compile_cache_path,
    densespark_expected_launch_policy,
    densespark_expected_resolved_provenance,
    validate_densespark_profile,
    validate_densespark_warmup_sync_sources,
)
from bench.manifest import (
    CaseSpec,
    ManifestError,
    SuiteSpec,
    load_models,
    load_suite,
    model_spec_to_dict,
    validate_benchmark_selection,
    validate_model,
)
from bench.runner import (
    PreflightError,
    _require_frozen_densespark_contract,
    _resolve_densespark_contract,
)
from bench.runtime import RuntimeErrorWithContext, start_vllm


ROOT = Path(__file__).resolve().parents[1]
CONTAINER_ID = "b" * 64


def _model() -> object:
    return load_models(ROOT / "manifests" / "models.toml")[
        DENSESPARK_WARMUP_SYNC_PROFILE_ID
    ]


def _completed(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class _Watchdog:
    def __init__(self) -> None:
        self.abort_callback_error = None
        self.starting_swap_used_kib = 0
        self.started = False
        self.stopped = False

    def start(self) -> _Watchdog:
        self.started = True
        return self

    def stop(self) -> None:
        self.stopped = True

    def register_abort_callback(self, callback: object) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")

    def raise_if_tripped(self) -> None:
        return None


class DenseSparkWarmupSyncManifestTests(unittest.TestCase):
    def test_profile_is_a_distinct_exact_experimental_sync_arm(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        base = models[DENSESPARK_PROFILE_ID]
        sync = models[DENSESPARK_WARMUP_SYNC_PROFILE_ID]
        validate_model(base)
        validate_model(sync)
        validate_densespark_profile(base)
        validate_densespark_profile(sync)

        self.assertIn("experimental-warmup-sync", sync.id)
        self.assertNotIn("rank4", sync.id)
        self.assertNotIn("skip", sync.id)
        self.assertEqual("exploratory", sync.support_status)
        self.assertEqual(DENSESPARK_WARMUP_SYNC_IMAGE, sync.image)
        self.assertEqual(DENSESPARK_WARMUP_SYNC_LOCAL_IMAGE_ID, sync.local_image_id)
        self.assertEqual(base.args, sync.args)
        self.assertEqual(DENSESPARK_C1_ARGS, sync.args)

        for mutation in (
            {"image": base.image},
            {"local_image_id": DENSESPARK_LOCAL_IMAGE_ID},
            {"support_status": base.support_status},
            {"args": (*sync.args, "--skip-warmup")},
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises(ManifestError):
                    validate_model(replace(sync, **mutation))

    def test_existing_d256_and_canonical_agentic_suites_bind_both_profiles(self) -> None:
        model = _model()
        for filename, suite_id in (
            ("qwen38_27b_densespark_c1.toml", DENSESPARK_SUITE_ID),
            ("agentic_tools.toml", DENSESPARK_TOOL_SUITE_ID),
        ):
            suite = load_suite(ROOT / "manifests" / "suites" / filename)
            self.assertEqual(suite_id, suite.id)
            validate_benchmark_selection(model, suite)

        with self.assertRaises(ManifestError):
            validate_benchmark_selection(
                model,
                SuiteSpec(
                    id="smoke",
                    cases=(CaseSpec(id="decode", kind="decode", requires=("chat",)),),
                ),
            )


class DenseSparkWarmupSyncContractTests(unittest.TestCase):
    def test_sync_mode_changes_only_cache_identity_not_c1_environment(self) -> None:
        base_config = densespark_c1_cache_config()
        sync_config = densespark_c1_cache_config(
            DENSESPARK_WARMUP_SYNC_PROFILE_ID
        )
        self.assertNotIn("DENSESPARK_WARMUP_MODE", base_config)
        self.assertEqual(
            DENSESPARK_WARMUP_SYNC_MODE,
            sync_config["DENSESPARK_WARMUP_MODE"],
        )
        self.assertEqual(
            DENSESPARK_WARMUP_SYNC_LOCAL_IMAGE_ID,
            sync_config["DENSESPARK_IMAGE_ID"],
        )
        base_common = {
            key: value
            for key, value in base_config.items()
            if key != "DENSESPARK_IMAGE_ID"
        }
        sync_common = {
            key: value
            for key, value in sync_config.items()
            if key not in {"DENSESPARK_IMAGE_ID", "DENSESPARK_WARMUP_MODE"}
        }
        self.assertEqual(base_common, sync_common)
        self.assertNotEqual(
            densespark_cache_namespace(base_config),
            densespark_cache_namespace(sync_config),
        )
        self.assertNotEqual(
            densespark_compile_cache_path(),
            densespark_compile_cache_path(
                profile_id=DENSESPARK_WARMUP_SYNC_PROFILE_ID
            ),
        )
        self.assertFalse(
            any("SPARKBENCH_QWEN_WARMUP" in name for name, _ in DENSESPARK_C1_ENVIRONMENT)
        )

    def test_checked_in_probe_and_image_recipe_match_the_receipt(self) -> None:
        source_receipt = validate_densespark_warmup_sync_sources(
            repository_root=ROOT
        )
        expected = densespark_expected_resolved_provenance(
            DENSESPARK_WARMUP_SYNC_PROFILE_ID
        )
        self.assertEqual(DENSESPARK_WARMUP_SYNC_MODE, expected["mode"])
        self.assertEqual(DENSESPARK_WARMUP_SYNC_PROBE_SHA256, source_receipt["probe_sha256"])
        self.assertEqual(expected["probe_sha256"], source_receipt["probe_sha256"])
        self.assertEqual(
            expected["image_recipe_sha256"],
            source_receipt["image_recipe_sha256"],
        )
        self.assertEqual(
            expected["dockerignore_sha256"],
            source_receipt["dockerignore_sha256"],
        )
        self.assertTrue(all(isinstance(value, str) for value in source_receipt.values()))
        self.assertFalse(any("/" in value for value in source_receipt.values()))

    def test_plan_and_frozen_receipts_bind_the_derived_image_and_mode(self) -> None:
        pq = DenseSparkPQArtifact(DENSESPARK_PQ_SIZE_BYTES, DENSESPARK_PQ_SHA256)
        snapshot = DenseSparkSnapshotReceipt(
            DENSESPARK_WEIGHT_FILE_COUNT, DENSESPARK_WEIGHT_SIZE_BYTES
        )
        with (
            patch(
                "bench.runner.validate_densespark_warmup_sync_sources"
            ) as sources,
            patch(
                "bench.runner.validate_densespark_local_image",
                return_value=DENSESPARK_WARMUP_SYNC_LOCAL_IMAGE_ID,
            ) as image,
            patch("bench.runner.validate_densespark_pq_artifact", return_value=pq),
            patch("bench.runner.validate_densespark_snapshot", return_value=snapshot),
        ):
            receipt = _resolve_densespark_contract(_model())
        sources.assert_called_once_with()
        image.assert_called_once()
        self.assertEqual(
            DENSESPARK_WARMUP_SYNC_LOCAL_IMAGE_ID,
            receipt["docker_image_id"],
        )
        self.assertEqual(DENSESPARK_WARMUP_SYNC_MODE, receipt["mode"])
        self.assertFalse(any(key.endswith("_path") for key in receipt))

        frozen = SimpleNamespace(**model_spec_to_dict(_model()))
        launch_policy = densespark_expected_launch_policy()
        _require_frozen_densespark_contract(frozen, receipt, launch_policy)
        self.assertEqual(
            DENSESPARK_WARMUP_SYNC_LOCAL_IMAGE_ID,
            frozen.resolved_local_image_id,
        )
        tampered = dict(receipt)
        tampered["mode"] = "rank4"
        with self.assertRaises(PreflightError):
            _require_frozen_densespark_contract(frozen, tampered, launch_policy)


class DenseSparkWarmupSyncRuntimeTests(unittest.TestCase):
    def prepare_home(self, home: Path) -> None:
        repository = (
            home
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--Frozenlock--Qwen3.8-27B-int4-AutoRound"
        )
        (repository / "snapshots" / DENSESPARK_MODEL_REVISION).mkdir(parents=True)
        (home / ".cache").chmod(0o700)

    def runtime_model(self) -> SimpleNamespace:
        return SimpleNamespace(
            **model_spec_to_dict(_model()),
            run_identity="run-sync",
            resolved_densespark_launch_policy=densespark_expected_launch_policy(),
        )

    def patches(self, home: Path, watchdog: _Watchdog) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch("bench.densespark.Path.home", return_value=home))
        stack.enter_context(patch("bench.runtime._existing_container", return_value=None))
        stack.enter_context(patch("bench.runtime._port_is_free", return_value=True))
        stack.enter_context(
            patch(
                "bench.runtime.validate_densespark_snapshot",
                return_value=DenseSparkSnapshotReceipt(
                    DENSESPARK_WEIGHT_FILE_COUNT,
                    DENSESPARK_WEIGHT_SIZE_BYTES,
                ),
            )
        )
        stack.enter_context(
            patch(
                "bench.runtime.validate_densespark_pq_artifact",
                return_value=DenseSparkPQArtifact(
                    DENSESPARK_PQ_SIZE_BYTES,
                    DENSESPARK_PQ_SHA256,
                ),
            )
        )
        stack.enter_context(
            patch("bench.runtime._densespark_startup_watchdog", return_value=watchdog)
        )
        stack.enter_context(patch("bench.runtime.wait_for_endpoint", return_value=0.25))
        return stack

    def test_launch_uses_exact_derived_image_sync_only_and_same_watchdog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self.prepare_home(home)
            commands: list[list[str]] = []
            watchdog = _Watchdog()

            def run(
                command: list[str],
                *,
                check: bool = True,
                timeout: float | None = None,
            ) -> subprocess.CompletedProcess[str]:
                del check, timeout
                commands.append(command)
                if command[:3] == ["docker", "image", "inspect"]:
                    return _completed(stdout=DENSESPARK_WARMUP_SYNC_LOCAL_IMAGE_ID + "\n")
                if command[:2] == ["docker", "run"]:
                    return _completed(stdout=CONTAINER_ID + "\n")
                raise AssertionError(f"unexpected command: {command}")

            with self.patches(home, watchdog), patch(
                "bench.runtime._run", side_effect=run
            ):
                server = start_vllm(self.runtime_model(), workspace=home)

        self.assertTrue(watchdog.started)
        self.assertTrue(watchdog.stopped)
        self.assertEqual(2, len(commands))
        launch = commands[1]
        self.assertIn(DENSESPARK_WARMUP_SYNC_LOCAL_IMAGE_ID, launch)
        self.assertNotIn(DENSESPARK_WARMUP_SYNC_IMAGE, launch)
        environment = tuple(
            launch[index + 1]
            for index, argument in enumerate(launch[:-1])
            if argument == "--env"
        )
        self.assertEqual(
            tuple(f"{name}={value}" for name, value in DENSESPARK_C1_ENVIRONMENT),
            environment,
        )
        joined = " ".join(launch)
        self.assertNotIn("SPARKBENCH_QWEN_WARMUP_SKIP", joined)
        self.assertNotIn("SPARKBENCH_QWEN_WARMUP_RANK4_STATE", joined)
        self.assertEqual(1, launch.count("--max-model-len"))
        self.assertEqual(list(DENSESPARK_C1_ARGS), launch[-len(DENSESPARK_C1_ARGS) :])
        mounts = [
            launch[index + 1]
            for index, argument in enumerate(launch[:-1])
            if argument == "--volume"
        ]
        expected_namespace = densespark_cache_namespace(
            densespark_c1_cache_config(DENSESPARK_WARMUP_SYNC_PROFILE_ID)
        )
        self.assertIn(expected_namespace, mounts[-1])
        self.assertNotIn(densespark_cache_namespace(densespark_c1_cache_config()), mounts[-1])
        assert server.native_provenance is not None
        self.assertEqual(
            DENSESPARK_WARMUP_SYNC_PROFILE_ID,
            server.native_provenance["densespark_profile"],
        )
        self.assertEqual(
            DENSESPARK_WARMUP_SYNC_MODE,
            server.native_provenance["densespark_warmup_mode"],
        )

    def test_source_drift_fails_before_any_runtime_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self.prepare_home(home)
            watchdog = _Watchdog()
            with (
                self.patches(home, watchdog),
                patch(
                    "bench.runtime.validate_densespark_warmup_sync_sources",
                    side_effect=DenseSparkContractError("synthetic drift"),
                ),
                patch("bench.runtime._run") as run,
                self.assertRaisesRegex(RuntimeErrorWithContext, "source provenance"),
            ):
                start_vllm(self.runtime_model(), workspace=home)
            run.assert_not_called()
            self.assertFalse(watchdog.started)


if __name__ == "__main__":
    unittest.main()
