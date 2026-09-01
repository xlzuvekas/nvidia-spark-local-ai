from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.densespark import (
    DENSESPARK_C1_ARGS,
    DENSESPARK_C1_ENVIRONMENT,
    DENSESPARK_CONTAINER_SNAPSHOT,
    DENSESPARK_IMAGE,
    DENSESPARK_LOCAL_IMAGE_ID,
    DENSESPARK_MAX_CONTEXT,
    DENSESPARK_MODEL_REVISION,
    DENSESPARK_MODEL_SOURCE,
    DENSESPARK_NATIVE_CONTEXT,
    DENSESPARK_PQ_CACHE_RELATIVE_PATH,
    DENSESPARK_PQ_CONTAINER_PATH,
    DENSESPARK_PQ_SHA256,
    DENSESPARK_PQ_SIZE_BYTES,
    DENSESPARK_PROFILE_ID,
    DENSESPARK_RECIPE_REVISION,
    DENSESPARK_RECIPE_SOURCE,
    DENSESPARK_SERVED_NAME,
    DENSESPARK_STARTUP_MAX_STARTING_SWAP_MIB,
    DENSESPARK_STARTUP_MAX_SWAP_GROWTH_MIB,
    DENSESPARK_STARTUP_MIN_MEMAVAILABLE_GIB,
    DENSESPARK_SUITE_ID,
    DENSESPARK_TOOL_SUITE_ID,
    DENSESPARK_TOOL_CALL_PARSER,
    DENSESPARK_WEIGHT_FILE_COUNT,
    DENSESPARK_WEIGHT_SIZE_BYTES,
    DenseSparkContractError,
    DenseSparkPQArtifact,
    DenseSparkSnapshotReceipt,
    densespark_c1_cache_config,
    densespark_cache_namespace,
    densespark_compile_cache_path,
    densespark_expected_launch_policy,
    densespark_pq_artifact_path,
)
from bench.inventory import (
    DockerImage,
    HuggingFaceSnapshot,
    Inventory,
    assess_model_availability,
)
from bench.manifest import (
    CaseSpec,
    ManifestError,
    ModelSpec,
    SuiteSpec,
    load_suite,
    model_spec_to_dict,
    validate_benchmark_selection,
    validate_model,
    validate_suite,
)
from bench.runner import (
    PreflightError,
    _require_frozen_densespark_contract,
    _require_frozen_densespark_suite,
    _resolve_densespark_contract,
    create_plan,
)
from bench.host_safety import HostSafetyError
from bench.runtime import (
    RuntimeErrorWithContext,
    _densespark_startup_watchdog,
    _prepare_densespark_compile_cache,
    _require_densespark_cache_node,
    _validate_densespark_launch_command,
    start_vllm,
)


ROOT = Path(__file__).resolve().parents[1]
CONTAINER_ID = "a" * 64


class _FakeStartupWatchdog:
    def __init__(self) -> None:
        self.abort_callback_error: BaseException | None = None
        self.starting_swap_used_kib = 27 * 1_024
        self.callback = None
        self.started = False
        self.stopped = False

    def start(self) -> _FakeStartupWatchdog:
        self.started = True
        return self

    def stop(self) -> None:
        self.stopped = True

    def register_abort_callback(self, callback: object) -> None:
        self.callback = callback

    def raise_if_tripped(self) -> None:
        return None


class _TrippedStartupWatchdog(_FakeStartupWatchdog):
    def __init__(self) -> None:
        super().__init__()
        self.failure = HostSafetyError(
            "swap_growth_above_maximum",
            "synthetic DenseSpark swap-growth breach",
            observed_kib=513 * 1_024,
            limit_kib=512 * 1_024,
            starting_swap_used_kib=27 * 1_024,
        )

    def register_abort_callback(self, callback: object) -> None:
        super().register_abort_callback(callback)
        if not callable(callback):
            raise TypeError("callback is not callable")
        callback()

    def raise_if_tripped(self) -> None:
        raise self.failure


def _model() -> ModelSpec:
    return ModelSpec(
        id=DENSESPARK_PROFILE_ID,
        backend="vllm",
        source=DENSESPARK_MODEL_SOURCE,
        revision=DENSESPARK_MODEL_REVISION,
        recipe_source=DENSESPARK_RECIPE_SOURCE,
        recipe_revision=DENSESPARK_RECIPE_REVISION,
        served_name=DENSESPARK_SERVED_NAME,
        tasks=("chat", "thinking", "tools"),
        image=DENSESPARK_IMAGE,
        local_image_id=DENSESPARK_LOCAL_IMAGE_ID,
        cache_dir="user",
        max_context=DENSESPARK_MAX_CONTEXT,
        native_context=DENSESPARK_NATIVE_CONTEXT,
        startup_timeout_s=1_800,
        endpoint="http://127.0.0.1:8000/v1",
        estimated_ram_gib=92.0,
        lifecycle="docker",
        support_status="spark_vllm_recipe",
        architecture="dense+gdn",
        quantization="int4-autoround+densespark-pq",
        description=(
            "0.86 is a MemAvailable-focused configuration reduction from "
            "upstream 0.90."
        ),
        weight_file_count=DENSESPARK_WEIGHT_FILE_COUNT,
        weight_size_bytes=DENSESPARK_WEIGHT_SIZE_BYTES,
        densespark_pq_file=DENSESPARK_PQ_CACHE_RELATIVE_PATH.as_posix(),
        densespark_pq_digest=DENSESPARK_PQ_SHA256,
        densespark_pq_size_bytes=DENSESPARK_PQ_SIZE_BYTES,
        args=DENSESPARK_C1_ARGS,
    )


def _suite() -> SuiteSpec:
    return SuiteSpec(
        id=DENSESPARK_SUITE_ID,
        description="Synthetic minimal C1 suite.",
        cases=(
            CaseSpec(
                id="densespark-c1-decode-256",
                kind="decode",
                requires=("chat",),
                warmups=1,
                repetitions=3,
                max_output_tokens=256,
                temperature=0.0,
                concurrency=1,
                prompt_repetitions=0,
            ),
        ),
    )


def _completed(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _launch_validation_fixture() -> tuple[list[str], Path, Path, Path]:
    repository = Path(
        "/synthetic/huggingface/hub/"
        "models--Frozenlock--Qwen3.8-27B-int4-AutoRound"
    )
    pq_artifact = Path("/synthetic/densespark/pq_head_m128.pt")
    compile_cache = Path("/synthetic/vllm-cache")
    environment = [
        item
        for name, value in DENSESPARK_C1_ENVIRONMENT
        for item in ("--env", f"{name}={value}")
    ]
    command = [
        "docker",
        "run",
        "--detach",
        "--pull=never",
        "--network",
        "bridge",
        "--name",
        "sparkbench-vllm",
        "--label",
        "ai.sparkbench.managed=true",
        "--label",
        "ai.sparkbench.run=run-1",
        "--label",
        "ai.sparkbench.backend=vllm",
        "--entrypoint",
        "vllm",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "--publish",
        "127.0.0.1:8000:8000",
        "--volume",
        f"{repository}:/root/.cache/huggingface/hub/{repository.name}:ro",
        "--volume",
        f"{pq_artifact}:{DENSESPARK_PQ_CONTAINER_PATH}:ro",
        "--volume",
        f"{compile_cache}:/root/.cache/vllm:rw",
        *environment,
        DENSESPARK_LOCAL_IMAGE_ID,
        "serve",
        DENSESPARK_CONTAINER_SNAPSHOT,
        "--served-model-name",
        DENSESPARK_SERVED_NAME,
        *DENSESPARK_C1_ARGS,
    ]
    return command, repository, pq_artifact, compile_cache


class DenseSparkManifestAdapterTests(unittest.TestCase):
    def test_exact_profile_serializes_without_generic_env_or_mount_fields(self) -> None:
        model = _model()
        validate_model(model)
        record = model_spec_to_dict(model)
        self.assertEqual(record["local_image_id"], DENSESPARK_LOCAL_IMAGE_ID)
        self.assertEqual(record["densespark_pq_digest"], DENSESPARK_PQ_SHA256)
        self.assertNotIn("sglang_storage_mode", record)
        self.assertFalse(any("env" in key or "mount" in key for key in record))

    def test_every_execution_identity_mutation_fails_closed(self) -> None:
        mutations = (
            {"source": "example/different"},
            {"revision": "0" * 40},
            {"recipe_revision": "1" * 40},
            {"image": "local/densespark:different"},
            {"local_image_id": f"sha256:{'0' * 64}"},
            {"cache_dir": "project"},
            {"max_context": 32_768},
            {"estimated_ram_gib": 1.0},
            {"args": (*DENSESPARK_C1_ARGS, "--enable-auto-tool-choice")},
            {"tasks": ("chat", "thinking")},
            {"args": tuple(
                "different" if value == DENSESPARK_TOOL_CALL_PARSER else value
                for value in DENSESPARK_C1_ARGS
            )},
            {"densespark_pq_digest": f"sha256:{'0' * 64}"},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(ManifestError):
                    validate_model(replace(_model(), **mutation))

    def test_local_image_id_remains_forbidden_for_ordinary_vllm(self) -> None:
        ordinary = ModelSpec(
            id="ordinary-vllm",
            backend="vllm",
            source="example/model",
            served_name="example/model",
            tasks=("chat",),
            image="example/image:tag",
            local_image_id=f"sha256:{'1' * 64}",
            max_context=8_192,
        )
        with self.assertRaisesRegex(ManifestError, "local_image_id"):
            validate_model(ordinary)

    def test_exact_suite_loads_and_is_bound_to_exact_profile(self) -> None:
        suite = load_suite(
            ROOT / "manifests" / "suites" / "qwen38_27b_densespark_c1.toml"
        )
        validate_benchmark_selection(_model(), suite)
        tool_suite = load_suite(ROOT / "manifests" / "suites" / "agentic_tools.toml")
        self.assertEqual(tool_suite.id, DENSESPARK_TOOL_SUITE_ID)
        validate_benchmark_selection(_model(), tool_suite)
        with self.assertRaises(ManifestError):
            validate_benchmark_selection(
                replace(_model(), id="ordinary-vllm"), suite
            )
        with self.assertRaises(ManifestError):
            validate_benchmark_selection(
                _model(),
                SuiteSpec(
                    id="smoke",
                    cases=(
                        CaseSpec(
                            id="decode",
                            kind="decode",
                            requires=("chat",),
                        ),
                    ),
                ),
            )

    def test_suite_mutation_rejects_non_c1_or_tool_case(self) -> None:
        validate_suite(_suite())
        case = _suite().cases[0]
        for mutation in (
            {"concurrency": 2},
            {"requires": ("chat", "tools")},
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises(ManifestError):
                    mutated = replace(
                        _suite(),
                        cases=(replace(case, **mutation),),
                    )
                    validate_suite(mutated)
                with self.assertRaises(ManifestError):
                    validate_benchmark_selection(_model(), mutated)


class DenseSparkInventoryAdapterTests(unittest.TestCase):
    def inventory(self, image_id: str) -> Inventory:
        return Inventory(
            collected_at="2026-08-31T00:00:00+00:00",
            python_version="3.12",
            platform="synthetic",
            machine="aarch64",
            huggingface_snapshots=(
                HuggingFaceSnapshot(
                    source=DENSESPARK_MODEL_SOURCE,
                    revision=DENSESPARK_MODEL_REVISION,
                    path=Path("/synthetic/repository/snapshots")
                    / DENSESPARK_MODEL_REVISION,
                ),
            ),
            docker_images=(
                DockerImage(
                    repository="local/densespark",
                    tag="qwen38-27b-v1.2-0abecc3",
                    digest=None,
                    image_id=image_id,
                ),
            ),
            ollama_models=(),
        )

    def test_availability_requires_exact_image_snapshot_and_pq(self) -> None:
        with (
            patch("bench.inventory._snapshot_cache_dir", return_value="user"),
            patch("bench.inventory.validate_densespark_profile"),
            patch("bench.inventory.validate_densespark_snapshot"),
            patch("bench.inventory.validate_densespark_pq_artifact"),
        ):
            available = assess_model_availability(
                {DENSESPARK_PROFILE_ID: _model()},
                self.inventory(DENSESPARK_LOCAL_IMAGE_ID),
            )[DENSESPARK_PROFILE_ID]
            wrong_image = assess_model_availability(
                {DENSESPARK_PROFILE_ID: _model()},
                self.inventory(f"sha256:{'0' * 64}"),
            )[DENSESPARK_PROFILE_ID]

        self.assertTrue(available.available)
        self.assertTrue(available.source_available)
        self.assertTrue(available.runtime_available)
        self.assertFalse(wrong_image.runtime_available)

    def test_missing_pq_marks_source_unavailable(self) -> None:
        with (
            patch("bench.inventory._snapshot_cache_dir", return_value="user"),
            patch("bench.inventory.validate_densespark_profile"),
            patch("bench.inventory.validate_densespark_snapshot"),
            patch(
                "bench.inventory.validate_densespark_pq_artifact",
                side_effect=DenseSparkContractError("missing"),
            ),
        ):
            availability = assess_model_availability(
                {DENSESPARK_PROFILE_ID: _model()},
                self.inventory(DENSESPARK_LOCAL_IMAGE_ID),
            )[DENSESPARK_PROFILE_ID]
        self.assertFalse(availability.source_available)
        self.assertIn("PQ artifact", " ".join(availability.details))


class DenseSparkRunnerAdapterTests(unittest.TestCase):
    def test_plan_receipt_is_scalar_only_and_deterministic(self) -> None:
        pq = DenseSparkPQArtifact(DENSESPARK_PQ_SIZE_BYTES, DENSESPARK_PQ_SHA256)
        snapshot = DenseSparkSnapshotReceipt(
            DENSESPARK_WEIGHT_FILE_COUNT,
            DENSESPARK_WEIGHT_SIZE_BYTES,
        )
        with (
            patch("bench.runner.validate_densespark_local_image", return_value=DENSESPARK_LOCAL_IMAGE_ID),
            patch("bench.runner.validate_densespark_pq_artifact", return_value=pq),
            patch("bench.runner.validate_densespark_snapshot", return_value=snapshot),
        ):
            first = _resolve_densespark_contract(_model())
            second = _resolve_densespark_contract(_model())

        self.assertEqual(first, second)
        self.assertTrue(all(isinstance(value, (int, str)) for value in first.values()))
        self.assertFalse(any("/home/" in str(value) for value in first.values()))
        frozen = SimpleNamespace(**model_spec_to_dict(_model()))
        launch_policy = densespark_expected_launch_policy()
        _require_frozen_densespark_contract(frozen, first, launch_policy)
        self.assertEqual(frozen.resolved_local_image_id, DENSESPARK_LOCAL_IMAGE_ID)
        self.assertEqual(
            frozen.resolved_densespark_launch_policy,
            launch_policy,
        )

        tampered = dict(first)
        tampered["configuration_sha256"] = f"sha256:{'0' * 64}"
        with self.assertRaises(PreflightError):
            _require_frozen_densespark_contract(frozen, tampered, launch_policy)
        with self.assertRaisesRegex(PreflightError, "launch policy"):
            _require_frozen_densespark_contract(frozen, first, None)

        for key, value in (
            ("host_safety_min_memavailable_bytes", 1),
            ("environment_hf_hub_offline", "0"),
            ("docker_pull_policy", "always"),
            ("publish_host", "0.0.0.0"),
            ("label_backend", "ai.sparkbench.backend=other"),
            ("docker_network", "none"),
            ("sha256", f"sha256:{'0' * 64}"),
        ):
            with self.subTest(policy_key=key):
                changed_policy = dict(launch_policy)
                changed_policy[key] = value
                with self.assertRaises(PreflightError):
                    _require_frozen_densespark_contract(
                        frozen,
                        first,
                        changed_policy,
                    )

        frozen_suite = {
            "id": DENSESPARK_SUITE_ID,
            "cases": [
                {
                    **asdict(case),
                    "case_id": f"synthetic-{index}",
                }
                for index, case in enumerate(_suite().cases)
            ],
        }
        _require_frozen_densespark_suite(frozen, frozen_suite)
        frozen_suite["cases"][0]["concurrency"] = 2
        with self.assertRaises(PreflightError):
            _require_frozen_densespark_suite(frozen, frozen_suite)

    def test_new_plan_freezes_launch_policy_as_a_sibling_receipt(self) -> None:
        artifact_receipt = {
            "cache_namespace": "densespark-v1-" + "0" * 64,
            "configuration_sha256": "sha256:" + "1" * 64,
            "docker_image_id": DENSESPARK_LOCAL_IMAGE_ID,
            "model_revision": DENSESPARK_MODEL_REVISION,
            "pq_artifact_sha256": DENSESPARK_PQ_SHA256,
            "pq_artifact_size_bytes": DENSESPARK_PQ_SIZE_BYTES,
            "weight_file_count": DENSESPARK_WEIGHT_FILE_COUNT,
            "weight_size_bytes": DENSESPARK_WEIGHT_SIZE_BYTES,
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "bench.runner._resolve_densespark_contract",
                return_value=artifact_receipt,
            ),
            patch("bench.runner._host_snapshot", return_value={}),
        ):
            root = Path(directory)
            run_dir = create_plan(
                model=_model(),
                suite=_suite(),
                results_root=root / "results",
                models_path=root / "models.toml",
                suite_path=root / "suite.toml",
            )
            plan = json.loads(
                (run_dir / "plan.json").read_text(encoding="utf-8")
            )
        self.assertEqual(plan["resolved"]["densespark"], artifact_receipt)
        self.assertEqual(
            plan["resolved"]["densespark_launch_policy"],
            densespark_expected_launch_policy(),
        )


class DenseSparkCompileCacheTests(unittest.TestCase):
    def test_creates_exact_private_namespace_and_accepts_safe_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            cache = _prepare_densespark_compile_cache(home=home)
            self.assertEqual(cache, cache.resolve(strict=True))
            self.assertEqual(stat.S_IMODE(cache.stat().st_mode), 0o700)

            artifact_dir = cache / "root-container-output"
            artifact_dir.mkdir(mode=0o755)
            artifact = artifact_dir / "kernel.cubin"
            artifact.write_bytes(b"synthetic")
            artifact.chmod(0o644)
            self.assertEqual(
                _prepare_densespark_compile_cache(home=home),
                cache,
            )

    def test_root_owned_artifacts_are_allowed_only_below_user_anchors(self) -> None:
        metadata = SimpleNamespace(
            st_mode=stat.S_IFREG | 0o600,
            st_nlink=1,
            st_uid=0,
        )
        with patch("bench.runtime.os.geteuid", return_value=12_345):
            _require_densespark_cache_node(metadata, anchor=False)
            with self.assertRaisesRegex(RuntimeErrorWithContext, "owner"):
                _require_densespark_cache_node(metadata, anchor=True)

    def test_rejects_group_or_world_writable_anchors_and_descendants(self) -> None:
        for mode in (0o720, 0o702):
            with self.subTest(mode=oct(mode)):
                with tempfile.TemporaryDirectory() as directory:
                    home = Path(directory)
                    cache = _prepare_densespark_compile_cache(home=home)
                    cache.chmod(mode)
                    with self.assertRaisesRegex(
                        RuntimeErrorWithContext,
                        "group/world-writable",
                    ):
                        _prepare_densespark_compile_cache(home=home)

        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            cache = _prepare_densespark_compile_cache(home=home)
            unsafe = cache / "unsafe.bin"
            unsafe.write_bytes(b"unsafe")
            unsafe.chmod(0o660)
            with self.assertRaisesRegex(
                RuntimeErrorWithContext,
                "group/world-writable",
            ):
                _prepare_densespark_compile_cache(home=home)

    def test_rejects_cache_symlinks_and_special_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            cache = _prepare_densespark_compile_cache(home=home)
            outside = home / "outside"
            outside.mkdir()
            (cache / "redirect").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeErrorWithContext, "symlinks"):
                _prepare_densespark_compile_cache(home=home)

        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            cache = _prepare_densespark_compile_cache(home=home)
            cache.rmdir()
            outside = home / "outside"
            outside.mkdir()
            cache.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeErrorWithContext, "symlinks"):
                _prepare_densespark_compile_cache(home=home)
            self.assertEqual(tuple(outside.iterdir()), ())

        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            cache = _prepare_densespark_compile_cache(home=home)
            os.mkfifo(cache / "compiler.pipe")
            with self.assertRaisesRegex(
                RuntimeErrorWithContext,
                "directories and regular files",
            ):
                _prepare_densespark_compile_cache(home=home)

    def test_rejects_redirected_namespace_before_creating_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            outside = home.parent / f"{home.name}-outside-densespark-cache"
            with patch(
                "bench.runtime.densespark_compile_cache_path",
                return_value=outside,
            ):
                with self.assertRaises(RuntimeErrorWithContext):
                    _prepare_densespark_compile_cache(home=home)
            self.assertFalse(outside.exists())


class DenseSparkRuntimeAdapterTests(unittest.TestCase):
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

    def model(self) -> SimpleNamespace:
        return SimpleNamespace(
            **model_spec_to_dict(_model()),
            run_identity="run-1",
            resolved_densespark_launch_policy=densespark_expected_launch_policy(),
        )

    def runtime_patches(self, home: Path) -> ExitStack:
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
            patch(
                "bench.runtime._densespark_startup_watchdog",
                return_value=_FakeStartupWatchdog(),
            )
        )
        stack.enter_context(patch("bench.runtime.wait_for_endpoint", return_value=0.25))
        return stack

    def test_startup_watchdog_factory_uses_exact_cold_compile_limits(self) -> None:
        sentinel = object()
        with patch("bench.runtime.HostSafetyWatchdog", return_value=sentinel) as factory:
            self.assertIs(_densespark_startup_watchdog(), sentinel)
        factory.assert_called_once_with(
            min_memavailable_gib=DENSESPARK_STARTUP_MIN_MEMAVAILABLE_GIB,
            max_swap_growth_mib=DENSESPARK_STARTUP_MAX_SWAP_GROWTH_MIB,
            max_starting_swap_mib=DENSESPARK_STARTUP_MAX_STARTING_SWAP_MIB,
        )

    def test_startup_watchdog_sources_units_from_launch_policy_receipt(self) -> None:
        policy = densespark_expected_launch_policy()
        policy.update(
            {
                "host_safety_min_memavailable_bytes": 15 * 1024**3,
                "host_safety_max_swap_growth_bytes": 513 * 1024**2,
                "host_safety_max_starting_swap_bytes": 514 * 1024**2,
            }
        )
        sentinel = object()
        with (
            patch(
                "bench.runtime.densespark_expected_launch_policy",
                return_value=policy,
            ),
            patch("bench.runtime.HostSafetyWatchdog", return_value=sentinel) as factory,
        ):
            self.assertIs(_densespark_startup_watchdog(), sentinel)
        factory.assert_called_once_with(
            min_memavailable_gib=15,
            max_swap_growth_mib=513,
            max_starting_swap_mib=514,
        )

    def test_launch_uses_immutable_inputs_fixed_env_and_owned_loopback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self.prepare_home(home)
            commands: list[list[str]] = []

            def run(
                command: list[str],
                *,
                check: bool = True,
                timeout: float | None = None,
            ) -> subprocess.CompletedProcess[str]:
                del check, timeout
                commands.append(command)
                if command[:3] == ["docker", "image", "inspect"]:
                    return _completed(stdout=DENSESPARK_LOCAL_IMAGE_ID + "\n")
                if command[:2] == ["docker", "run"]:
                    return _completed(stdout=CONTAINER_ID + "\n")
                raise AssertionError(f"unexpected command: {command}")

            with self.runtime_patches(home), patch("bench.runtime._run", side_effect=run):
                server = start_vllm(
                    self.model(),
                    workspace=Path(directory),
                    allow_download=False,
                )

        self.assertEqual(server.container_id, CONTAINER_ID)
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][:3], ["docker", "image", "inspect"])
        launch = commands[1]
        self.assertEqual(
            launch[:4], ["docker", "run", "--detach", "--pull=never"]
        )
        self.assertEqual(
            launch[launch.index("--network") + 1],
            "bridge",
        )
        self.assertIn(DENSESPARK_LOCAL_IMAGE_ID, launch)
        self.assertNotIn(DENSESPARK_IMAGE, launch)
        self.assertIn("127.0.0.1:8000:8000", launch)
        self.assertIn(DENSESPARK_CONTAINER_SNAPSHOT, launch)
        self.assertIn("--no-enable-prefix-caching", launch)
        self.assertIn("--enable-auto-tool-choice", launch)
        parser_index = launch.index("--tool-call-parser")
        self.assertEqual(launch[parser_index + 1], DENSESPARK_TOOL_CALL_PARSER)

        mounts = [
            launch[index + 1]
            for index, argument in enumerate(launch[:-1])
            if argument == "--volume"
        ]
        self.assertEqual(len(mounts), 3)
        self.assertTrue(mounts[0].endswith(":ro"))
        self.assertEqual(
            mounts[1].split(":", 1)[1],
            f"{DENSESPARK_PQ_CONTAINER_PATH}:ro",
        )
        self.assertTrue(mounts[2].endswith(":/root/.cache/vllm:rw"))
        self.assertIn(
            densespark_cache_namespace(densespark_c1_cache_config()),
            mounts[2],
        )
        environment = tuple(
            launch[index + 1]
            for index, argument in enumerate(launch[:-1])
            if argument == "--env"
        )
        self.assertEqual(
            environment,
            tuple(f"{name}={value}" for name, value in DENSESPARK_C1_ENVIRONMENT),
        )
        self.assertNotIn("HF_TOKEN", " ".join(launch))
        self.assertNotIn("HTTPS_PROXY", " ".join(launch))
        _validate_densespark_launch_command(
            launch,
            image_id=DENSESPARK_LOCAL_IMAGE_ID,
            run_identity="run-1",
            repository=(
                home
                / ".cache"
                / "huggingface"
                / "hub"
                / "models--Frozenlock--Qwen3.8-27B-int4-AutoRound"
            ),
            pq_artifact=densespark_pq_artifact_path(home=home),
            compile_cache=densespark_compile_cache_path(home=home),
        )
        assert server.native_provenance is not None
        self.assertEqual(
            server.native_provenance["densespark_launch_policy"],
            densespark_expected_launch_policy(),
        )
        self.assertEqual(server.native_provenance["hf_hub_policy"], "offline")
        self.assertEqual(server.native_provenance["network_isolation"], "none")
        self.assertEqual(
            server.native_provenance["network_topology"],
            "loopback_published_bridge_egress_capable",
        )
        self.assertEqual(
            server.native_provenance["gpu_memory_configuration_reduction"],
            "upstream_0.90_to_0.86_memavailable_margin",
        )
        self.assertEqual(server.native_provenance["draft_sample_method"], "probabilistic")
        self.assertTrue(server.native_provenance["tool_calling"])
        self.assertEqual(
            server.native_provenance["tool_call_parser"],
            DENSESPARK_TOOL_CALL_PARSER,
        )
        self.assertEqual(
            server.native_provenance["startup_min_memavailable_gib"],
            DENSESPARK_STARTUP_MIN_MEMAVAILABLE_GIB,
        )
        self.assertEqual(
            server.native_provenance["startup_max_swap_growth_mib"],
            DENSESPARK_STARTUP_MAX_SWAP_GROWTH_MIB,
        )

    def test_launch_policy_and_fixed_port_fail_before_container_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self.prepare_home(home)
            with self.runtime_patches(home) as stack:
                run = stack.enter_context(patch("bench.runtime._run"))
                model = self.model()
                model.resolved_densespark_launch_policy = None
                with self.assertRaisesRegex(RuntimeErrorWithContext, "frozen plan"):
                    start_vllm(model, workspace=Path(directory))
                run.assert_not_called()

            with self.runtime_patches(home) as stack:
                run = stack.enter_context(patch("bench.runtime._run"))
                with self.assertRaisesRegex(RuntimeErrorWithContext, "loopback port"):
                    start_vllm(
                        self.model(),
                        workspace=Path(directory),
                        port=8001,
                    )
                run.assert_not_called()

    def test_launch_command_validator_rejects_each_policy_shape_mutation(self) -> None:
        command, repository, pq_artifact, compile_cache = (
            _launch_validation_fixture()
        )
        _validate_densespark_launch_command(
            command,
            image_id=DENSESPARK_LOCAL_IMAGE_ID,
            run_identity="run-1",
            repository=repository,
            pq_artifact=pq_artifact,
            compile_cache=compile_cache,
        )
        mutations = (
            ("--pull=never", "--pull=always"),
            ("bridge", "none"),
            ("all", "device=0"),
            ("host", "private"),
            ("memlock=-1", "memlock=65536"),
            ("stack=67108864", "stack=8388608"),
            ("127.0.0.1:8000:8000", "0.0.0.0:8000:8000"),
            ("ai.sparkbench.managed=true", "ai.sparkbench.managed=false"),
            ("ai.sparkbench.run=run-1", "ai.sparkbench.run=other"),
            ("ai.sparkbench.backend=vllm", "ai.sparkbench.backend=other"),
            ("HF_HUB_OFFLINE=1", "HF_HUB_OFFLINE=0"),
            ("VLLM_NO_USAGE_STATS=1", "VLLM_NO_USAGE_STATS=0"),
            (
                f"{repository}:/root/.cache/huggingface/hub/"
                f"{repository.name}:ro",
                f"{repository}:/models:rw",
            ),
            (
                f"{pq_artifact}:{DENSESPARK_PQ_CONTAINER_PATH}:ro",
                f"{pq_artifact}:{DENSESPARK_PQ_CONTAINER_PATH}:rw",
            ),
            (
                f"{compile_cache}:/root/.cache/vllm:rw",
                f"{compile_cache}:/root/.cache/vllm:ro",
            ),
            (DENSESPARK_LOCAL_IMAGE_ID, "sha256:" + "0" * 64),
            ("serve", "chat"),
            (DENSESPARK_CONTAINER_SNAPSHOT, "/synthetic/different-snapshot"),
            (DENSESPARK_SERVED_NAME, "different-served-name"),
            ("65536", "32768"),
        )
        for old, new in mutations:
            with self.subTest(old=old):
                changed = list(command)
                changed[changed.index(old)] = new
                with self.assertRaises(DenseSparkContractError):
                    _validate_densespark_launch_command(
                        changed,
                        image_id=DENSESPARK_LOCAL_IMAGE_ID,
                        run_identity="run-1",
                        repository=repository,
                        pq_artifact=pq_artifact,
                        compile_cache=compile_cache,
                    )

    def test_launch_command_validator_rejects_arbitrary_extra_options(self) -> None:
        command, repository, pq_artifact, compile_cache = (
            _launch_validation_fixture()
        )
        image_index = command.index(DENSESPARK_LOCAL_IMAGE_ID)
        extras = (
            ("--privileged",),
            ("--ipc", "private"),
            ("--volume", "/synthetic/host:/synthetic/container:rw"),
            ("--mount", "type=bind,src=/synthetic,dst=/synthetic"),
        )
        for extra in extras:
            with self.subTest(extra=extra):
                changed = list(command)
                changed[image_index:image_index] = extra
                with self.assertRaises(DenseSparkContractError):
                    _validate_densespark_launch_command(
                        changed,
                        image_id=DENSESPARK_LOCAL_IMAGE_ID,
                        run_identity="run-1",
                        repository=repository,
                        pq_artifact=pq_artifact,
                        compile_cache=compile_cache,
                    )

        changed = [*command, "--trust-remote-code"]
        with self.assertRaises(DenseSparkContractError):
            _validate_densespark_launch_command(
                changed,
                image_id=DENSESPARK_LOCAL_IMAGE_ID,
                run_identity="run-1",
                repository=repository,
                pq_artifact=pq_artifact,
                compile_cache=compile_cache,
            )

    def test_cold_compile_safety_trip_immediately_stops_exact_owned_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self.prepare_home(home)
            commands: list[list[str]] = []

            def run(
                command: list[str],
                *,
                check: bool = True,
                timeout: float | None = None,
            ) -> subprocess.CompletedProcess[str]:
                del check, timeout
                commands.append(command)
                if command[:3] == ["docker", "image", "inspect"]:
                    return _completed(stdout=DENSESPARK_LOCAL_IMAGE_ID + "\n")
                if command[:2] == ["docker", "run"]:
                    return _completed(stdout=CONTAINER_ID + "\n")
                if command[:2] == ["docker", "inspect"]:
                    return _completed(stdout="true run-1\n")
                if command[:2] in (["docker", "stop"], ["docker", "rm"]):
                    return _completed()
                raise AssertionError(f"unexpected command: {command}")

            watchdog = _TrippedStartupWatchdog()
            with self.runtime_patches(home) as stack:
                stack.enter_context(
                    patch(
                        "bench.runtime._densespark_startup_watchdog",
                        return_value=watchdog,
                    )
                )
                wait = stack.enter_context(patch("bench.runtime.wait_for_endpoint"))
                stack.enter_context(patch("bench.runtime._run", side_effect=run))
                with self.assertRaisesRegex(
                    HostSafetyError, "synthetic DenseSpark swap-growth breach"
                ):
                    start_vllm(self.model(), workspace=Path(directory))

            wait.assert_not_called()
            self.assertTrue(watchdog.started)
            self.assertTrue(watchdog.stopped)
            self.assertIn(
                ["docker", "stop", "--time", "0", CONTAINER_ID],
                commands,
            )
            self.assertIn(["docker", "rm", CONTAINER_ID], commands)

    def test_pq_or_image_mismatch_fails_before_container_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self.prepare_home(home)
            with self.runtime_patches(home) as stack:
                stack.enter_context(
                    patch(
                        "bench.runtime.validate_densespark_pq_artifact",
                        side_effect=DenseSparkContractError("wrong"),
                    )
                )
                run = stack.enter_context(patch("bench.runtime._run"))
                with self.assertRaisesRegex(RuntimeErrorWithContext, "provenance"):
                    start_vllm(self.model(), workspace=Path(directory))
                run.assert_not_called()

            commands: list[list[str]] = []

            def wrong_image(
                command: list[str],
                *,
                check: bool = True,
                timeout: float | None = None,
            ) -> subprocess.CompletedProcess[str]:
                del check, timeout
                commands.append(command)
                return _completed(stdout=f"sha256:{'0' * 64}\n")

            with self.runtime_patches(home), patch(
                "bench.runtime._run", side_effect=wrong_image
            ):
                with self.assertRaisesRegex(RuntimeErrorWithContext, "provenance"):
                    start_vllm(self.model(), workspace=Path(directory))
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][:3], ["docker", "image", "inspect"])

    def test_startup_failure_saves_logs_and_invokes_owned_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self.prepare_home(home)

            def run(
                command: list[str],
                *,
                check: bool = True,
                timeout: float | None = None,
            ) -> subprocess.CompletedProcess[str]:
                del check, timeout
                if command[:3] == ["docker", "image", "inspect"]:
                    return _completed(stdout=DENSESPARK_LOCAL_IMAGE_ID + "\n")
                if command[:2] == ["docker", "run"]:
                    return _completed(stdout=CONTAINER_ID + "\n")
                raise AssertionError(f"unexpected command: {command}")

            with self.runtime_patches(home) as stack:
                stack.enter_context(
                    patch(
                        "bench.runtime.wait_for_endpoint",
                        side_effect=RuntimeErrorWithContext("not ready"),
                    )
                )
                save_logs = stack.enter_context(
                    patch("bench.runtime.save_server_logs")
                )
                stop = stack.enter_context(patch("bench.runtime.ManagedServer.stop"))
                stack.enter_context(patch("bench.runtime._run", side_effect=run))
                with self.assertRaisesRegex(RuntimeErrorWithContext, "not ready"):
                    start_vllm(
                        self.model(),
                        workspace=Path(directory),
                        server_log_path=Path(directory) / "server.log",
                    )

            save_logs.assert_called_once()
            stop.assert_called_once()

    def test_docker_run_failure_or_empty_id_recovers_only_owned_container(self) -> None:
        outcomes = (
            _completed(returncode=1),
            _completed(stdout="\n"),
        )
        for outcome in outcomes:
            with self.subTest(returncode=outcome.returncode, stdout=outcome.stdout):
                with tempfile.TemporaryDirectory() as directory:
                    home = Path(directory)
                    self.prepare_home(home)
                    commands: list[list[str]] = []

                    def run(
                        command: list[str],
                        *,
                        check: bool = True,
                        timeout: float | None = None,
                    ) -> subprocess.CompletedProcess[str]:
                        del check, timeout
                        commands.append(command)
                        if command[:3] == ["docker", "image", "inspect"]:
                            return _completed(
                                stdout=DENSESPARK_LOCAL_IMAGE_ID + "\n"
                            )
                        if command[:2] == ["docker", "run"]:
                            return outcome
                        if command[:2] == ["docker", "inspect"]:
                            return _completed(stdout="true run-1\n")
                        if command[:2] in (["docker", "stop"], ["docker", "rm"]):
                            return _completed()
                        raise AssertionError(f"unexpected command: {command}")

                    with self.runtime_patches(home) as stack:
                        stack.enter_context(
                            patch(
                                "bench.runtime._existing_container",
                                side_effect=[
                                    None,
                                    (CONTAINER_ID, True, "run-1"),
                                ],
                            )
                        )
                        stack.enter_context(
                            patch("bench.runtime._run", side_effect=run)
                        )
                        with self.assertRaisesRegex(
                            RuntimeErrorWithContext,
                            "docker run|container ID",
                        ):
                            start_vllm(
                                self.model(),
                                workspace=Path(directory),
                            )

                    self.assertIn(
                        ["docker", "stop", "--time", "30", CONTAINER_ID],
                        commands,
                    )
                    self.assertIn(["docker", "rm", CONTAINER_ID], commands)

    def test_download_mode_is_rejected_before_provenance_or_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self.prepare_home(home)
            with (
                patch("bench.densespark.Path.home", return_value=home),
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime._run") as run,
            ):
                with self.assertRaisesRegex(
                    RuntimeErrorWithContext, "model-artifact downloads"
                ):
                    start_vllm(
                        self.model(),
                        workspace=Path(directory),
                        allow_download=True,
                    )
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
