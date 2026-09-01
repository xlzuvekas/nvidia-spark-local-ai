from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.densespark import (
    DENSESPARK_C1_ARGS,
    DENSESPARK_MODEL_REVISION,
    DENSESPARK_FAST_CODING_SUITE_ID,
    DENSESPARK_FAST_REQUEST_BODY_JSON,
    DENSESPARK_NATIVE_262K_FAST_PROFILE_BY_DEPTH,
    DENSESPARK_NATIVE_262K_FAST_PROFILE_IDS,
    DENSESPARK_NATIVE_262K_ARGS,
    DENSESPARK_NATIVE_262K_GPU_MEMORY_UTILIZATION,
    DENSESPARK_NATIVE_262K_PROFILE_ID,
    DENSESPARK_NATIVE_CONTEXT,
    DENSESPARK_PQ_SHA256,
    DENSESPARK_PQ_SIZE_BYTES,
    DENSESPARK_PROFILE_ID,
    DENSESPARK_WARMUP_SYNC_LOCAL_IMAGE_ID,
    DENSESPARK_WARMUP_SYNC_MODE,
    DENSESPARK_WARMUP_SYNC_PROFILE_ID,
    DENSESPARK_WEIGHT_FILE_COUNT,
    DENSESPARK_WEIGHT_SIZE_BYTES,
    DenseSparkContractError,
    DenseSparkPQArtifact,
    DenseSparkSnapshotReceipt,
    densespark_args_for_profile,
    densespark_c1_cache_config,
    densespark_cache_namespace,
    densespark_compile_cache_path,
    densespark_configuration_digest,
    densespark_expected_launch_policy,
    densespark_expected_resolved_provenance,
    densespark_fast_coding_prompt,
    densespark_max_context_for_profile,
    densespark_speculative_depth_for_profile,
    is_densespark_warmup_sync_profile,
    validate_densespark_profile,
)
from bench.manifest import (
    ManifestError,
    load_models,
    load_suite,
    model_spec_to_dict,
    validate_benchmark_selection,
    validate_model,
)
from bench.runtime import _densespark_gpu_memory_provenance, start_vllm
from bench.runner import (
    PreflightError,
    _request_arguments,
    _require_frozen_densespark_suite,
    _validate_capability,
)


ROOT = Path(__file__).resolve().parents[1]
CONTAINER_ID = "c" * 64
BASE_CONFIG_SHA256 = (
    "sha256:e6ac07581881aa589dfeebca7ca034d9"
    "9858ab333166bc5648cbfa944543fda6"
)
SYNC_64K_CONFIG_SHA256 = (
    "sha256:a74a143c5d2cd26976d859cc29949efb"
    "915ea414ff1bdbeaef332ad3ebbe1a44"
)
NATIVE_262K_CONFIG_SHA256 = (
    "sha256:87eee0e35d5377d9d9f4067932fe2749"
    "6c3625e3071cd80ba119497ce039c4d7"
)


def _model() -> object:
    return load_models(ROOT / "manifests" / "models.toml")[
        DENSESPARK_NATIVE_262K_PROFILE_ID
    ]


def _frozen_suite(path: Path) -> dict[str, object]:
    suite = load_suite(path)
    return {
        "id": suite.id,
        "description": suite.description,
        "schema_version": suite.schema_version,
        "protocol_digest": suite.protocol_digest,
        "cases": [asdict(case) for case in suite.cases],
    }


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


class DenseSparkNative262KManifestTests(unittest.TestCase):
    def test_profile_is_exact_native_context_bf16_auto_kv(self) -> None:
        model = _model()
        validate_model(model)
        validate_densespark_profile(model)

        self.assertEqual(DENSESPARK_NATIVE_CONTEXT, model.max_context)
        self.assertEqual(DENSESPARK_NATIVE_CONTEXT, model.native_context)
        self.assertEqual("exploratory", model.support_status)
        self.assertEqual(
            DENSESPARK_WARMUP_SYNC_LOCAL_IMAGE_ID,
            model.local_image_id,
        )
        self.assertEqual(DENSESPARK_NATIVE_262K_ARGS, model.args)
        self.assertEqual(
            "auto",
            model.args[model.args.index("--kv-cache-dtype") + 1],
        )
        self.assertEqual(
            "bfloat16",
            model.args[model.args.index("--mamba-ssm-cache-dtype") + 1],
        )
        self.assertEqual(
            DENSESPARK_NATIVE_262K_GPU_MEMORY_UTILIZATION,
            model.args[model.args.index("--gpu-memory-utilization") + 1],
        )
        self.assertNotIn("--hf-overrides", model.args)
        self.assertNotIn("--enforce-eager", model.args)
        self.assertFalse(any("rope" in argument.lower() for argument in model.args))
        self.assertEqual(
            '{"method":"mtp","num_speculative_tokens":8,'
            '"draft_sample_method":"probabilistic"}',
            model.args[model.args.index("--speculative-config") + 1],
        )
        self.assertTrue(is_densespark_warmup_sync_profile(model))

    def test_profile_and_suite_mutations_fail_closed(self) -> None:
        model = _model()
        suite = load_suite(
            ROOT / "manifests" / "suites" / "qwen38_27b_densespark_c1.toml"
        )
        validate_benchmark_selection(model, suite)

        for mutation in (
            {"max_context": 65_536},
            {"local_image_id": "sha256:" + "0" * 64},
            {"args": DENSESPARK_C1_ARGS},
            {
                "args": tuple(
                    "fp8" if value == "auto" else value for value in model.args
                )
            },
            {
                "args": tuple(
                    "0.86"
                    if value == DENSESPARK_NATIVE_262K_GPU_MEMORY_UTILIZATION
                    else value
                    for value in model.args
                )
            },
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises(ManifestError):
                    validate_model(replace(model, **mutation))

    def test_fast_profiles_are_exact_no_thinking_depth_candidates(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        suite = load_suite(
            ROOT
            / "manifests"
            / "suites"
            / "qwen38_27b_densespark_fast_coding.toml"
        )
        self.assertEqual(DENSESPARK_FAST_CODING_SUITE_ID, suite.id)
        self.assertIn("synthetic repetitive-code continuation", suite.description)
        self.assertIn("not a representative coding-agent workload", suite.description)
        self.assertEqual(
            DENSESPARK_NATIVE_262K_FAST_PROFILE_IDS,
            frozenset(DENSESPARK_NATIVE_262K_FAST_PROFILE_BY_DEPTH.values()),
        )
        for depth, profile_id in DENSESPARK_NATIVE_262K_FAST_PROFILE_BY_DEPTH.items():
            with self.subTest(depth=depth):
                model = models[profile_id]
                validate_model(model)
                validate_densespark_profile(model)
                validate_benchmark_selection(model, suite)
                self.assertEqual(DENSESPARK_NATIVE_CONTEXT, model.max_context)
                self.assertEqual(
                    DENSESPARK_FAST_REQUEST_BODY_JSON,
                    model.request_body_json,
                )
                self.assertEqual(
                    depth,
                    densespark_speculative_depth_for_profile(profile_id),
                )
                self.assertIn(
                    f'"num_speculative_tokens":{depth}',
                    model.args[model.args.index("--speculative-config") + 1],
                )

        with self.assertRaises(ManifestError):
            validate_benchmark_selection(_model(), suite)
        with self.assertRaises(ManifestError):
            validate_benchmark_selection(
                models[DENSESPARK_NATIVE_262K_FAST_PROFILE_BY_DEPTH[8]],
                load_suite(
                    ROOT
                    / "manifests"
                    / "suites"
                    / "qwen38_27b_densespark_c1.toml"
                ),
            )

    def test_fast_coding_prompt_is_deterministic_unique_and_bounded(self) -> None:
        first = densespark_fast_coding_prompt("d256-r0-w0")
        self.assertEqual(first, densespark_fast_coding_prompt("d256-r0-w0"))
        self.assertNotEqual(first, densespark_fast_coding_prompt("d256-r1-w0"))
        self.assertIn("test_merge_000", first)
        self.assertIn("until the output limit", first)
        for invalid in ("", "contains spaces", "x" * 257):
            with self.subTest(invalid=invalid):
                with self.assertRaises(DenseSparkContractError):
                    densespark_fast_coding_prompt(invalid)

    def test_fast_prompt_and_oracle_require_both_profile_and_case(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        suite = load_suite(
            ROOT
            / "manifests"
            / "suites"
            / "qwen38_27b_densespark_fast_coding.toml"
        )
        case = suite.cases[0]
        fast_model = models[DENSESPARK_NATIVE_262K_FAST_PROFILE_BY_DEPTH[8]]
        server = SimpleNamespace(
            backend="vllm",
            base_url="http://127.0.0.1:8000/v1",
            authorization=None,
        )
        fast_arguments = _request_arguments(
            server=server,
            model=fast_model,
            case=case,
            request_id="fast-r0-w0",
        )
        invalid_cross_pair_arguments = _request_arguments(
            server=server,
            model=_model(),
            case=case,
            request_id="fast-r0-w0",
        )
        self.assertIn("test_merge_000", fast_arguments["prompt"])
        self.assertNotIn(
            "test_merge_000", invalid_cross_pair_arguments["prompt"]
        )

        valid_result = SimpleNamespace(
            finish_reason="length",
            completion_tokens=case.max_output_tokens,
            content="def merge_sort(values): pass\ndef test_merge_000(): pass",
            reasoning="",
            reasoning_tokens=0,
            output_tps=60.0,
        )
        self.assertTrue(
            _validate_capability(case, valid_result, model=fast_model)["passed"]
        )
        for mutation in (
            {"content": "unrelated output"},
            {"reasoning": "hidden reasoning"},
            {"reasoning_tokens": 1},
        ):
            with self.subTest(mutation=mutation):
                result = SimpleNamespace(**{**vars(valid_result), **mutation})
                self.assertFalse(
                    _validate_capability(case, result, model=fast_model)["passed"]
                )
        self.assertTrue(
            _validate_capability(case, valid_result, model=_model())["passed"]
        )

    def test_frozen_boundary_rejects_fast_profile_suite_cross_pairs(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        fast_model = SimpleNamespace(
            **model_spec_to_dict(
                models[DENSESPARK_NATIVE_262K_FAST_PROFILE_BY_DEPTH[8]]
            )
        )
        native_model = SimpleNamespace(**model_spec_to_dict(_model()))
        fast_suite = _frozen_suite(
            ROOT
            / "manifests"
            / "suites"
            / "qwen38_27b_densespark_fast_coding.toml"
        )
        base_suite = _frozen_suite(
            ROOT / "manifests" / "suites" / "qwen38_27b_densespark_c1.toml"
        )

        _require_frozen_densespark_suite(fast_model, fast_suite)
        _require_frozen_densespark_suite(native_model, base_suite)
        for model, suite in (
            (native_model, fast_suite),
            (fast_model, base_suite),
        ):
            with self.subTest(model=model.id, suite=suite["id"]):
                with self.assertRaisesRegex(PreflightError, "profile/suite"):
                    _require_frozen_densespark_suite(model, suite)


class DenseSparkNative262KContractTests(unittest.TestCase):
    def test_gpu_memory_provenance_is_profile_exact_and_fails_closed(self) -> None:
        base = _densespark_gpu_memory_provenance(
            DENSESPARK_PROFILE_ID,
            DENSESPARK_C1_ARGS,
        )
        native = _densespark_gpu_memory_provenance(
            DENSESPARK_NATIVE_262K_PROFILE_ID,
            DENSESPARK_NATIVE_262K_ARGS,
        )

        self.assertEqual(0.86, base["gpu_memory_utilization"])
        self.assertEqual(
            "upstream_0.90_to_0.86_memavailable_margin",
            base["gpu_memory_configuration_reduction"],
        )
        self.assertEqual(0.70, native["gpu_memory_utilization"])
        self.assertEqual(
            "upstream_0.90_to_0.70_native_context_headroom",
            native["gpu_memory_configuration_reduction"],
        )

        with self.assertRaises(DenseSparkContractError):
            _densespark_gpu_memory_provenance("unknown", DENSESPARK_C1_ARGS)
        drifted = tuple(
            "0.71"
            if argument == DENSESPARK_NATIVE_262K_GPU_MEMORY_UTILIZATION
            else argument
            for argument in DENSESPARK_NATIVE_262K_ARGS
        )
        with self.assertRaises(DenseSparkContractError):
            _densespark_gpu_memory_provenance(
                DENSESPARK_NATIVE_262K_PROFILE_ID,
                drifted,
            )

    def test_profile_helpers_and_cache_namespace_are_context_bound(self) -> None:
        native = densespark_c1_cache_config(DENSESPARK_NATIVE_262K_PROFILE_ID)
        base = densespark_c1_cache_config(DENSESPARK_PROFILE_ID)
        sync_64k = densespark_c1_cache_config(DENSESPARK_WARMUP_SYNC_PROFILE_ID)

        self.assertEqual(
            DENSESPARK_NATIVE_CONTEXT,
            densespark_max_context_for_profile(DENSESPARK_NATIVE_262K_PROFILE_ID),
        )
        self.assertEqual(
            DENSESPARK_NATIVE_262K_ARGS,
            densespark_args_for_profile(DENSESPARK_NATIVE_262K_PROFILE_ID),
        )
        self.assertEqual(DENSESPARK_NATIVE_CONTEXT, native["DENSESPARK_MAX_MODEL_LEN"])
        self.assertEqual("auto", native["DENSESPARK_KV_CACHE_DTYPE"])
        self.assertEqual(
            DENSESPARK_NATIVE_262K_GPU_MEMORY_UTILIZATION,
            native["DENSESPARK_GPU_MEMORY_UTILIZATION"],
        )
        self.assertEqual("0.86", base["DENSESPARK_GPU_MEMORY_UTILIZATION"])
        self.assertEqual("0.86", sync_64k["DENSESPARK_GPU_MEMORY_UTILIZATION"])
        self.assertEqual(
            "bfloat16", native["DENSESPARK_MAMBA_SSM_CACHE_DTYPE"]
        )
        self.assertEqual(DENSESPARK_WARMUP_SYNC_MODE, native["DENSESPARK_WARMUP_MODE"])
        self.assertNotIn("DENSESPARK_KV_CACHE_DTYPE", base)
        self.assertNotIn("DENSESPARK_KV_CACHE_DTYPE", sync_64k)

        namespaces = {
            densespark_cache_namespace(config)
            for config in (base, sync_64k, native)
        }
        self.assertEqual(3, len(namespaces))
        self.assertEqual(
            BASE_CONFIG_SHA256,
            densespark_configuration_digest(base),
        )
        self.assertEqual(
            SYNC_64K_CONFIG_SHA256,
            densespark_configuration_digest(sync_64k),
        )
        self.assertEqual(
            NATIVE_262K_CONFIG_SHA256,
            densespark_configuration_digest(native),
        )
        receipt = densespark_expected_resolved_provenance(
            DENSESPARK_NATIVE_262K_PROFILE_ID
        )
        self.assertEqual(
            densespark_configuration_digest(native),
            receipt["configuration_sha256"],
        )
        self.assertEqual(
            densespark_cache_namespace(native),
            receipt["cache_namespace"],
        )
        self.assertEqual(DENSESPARK_WARMUP_SYNC_MODE, receipt["mode"])

        with self.assertRaises(DenseSparkContractError):
            densespark_args_for_profile("unknown")
        with self.assertRaises(DenseSparkContractError):
            densespark_max_context_for_profile("unknown")

    def test_fast_depths_isolate_compile_cache_except_identical_mtp8(self) -> None:
        native = densespark_c1_cache_config(DENSESPARK_NATIVE_262K_PROFILE_ID)
        configs = {
            depth: densespark_c1_cache_config(profile_id)
            for depth, profile_id in DENSESPARK_NATIVE_262K_FAST_PROFILE_BY_DEPTH.items()
        }
        self.assertEqual(8, native["DENSESPARK_SPEC_TOKENS"])
        self.assertEqual(
            densespark_cache_namespace(native),
            densespark_cache_namespace(configs[8]),
        )
        self.assertEqual(
            {4, 6, 7, 8},
            {int(config["DENSESPARK_SPEC_TOKENS"]) for config in configs.values()},
        )
        self.assertEqual(
            4,
            len(
                {
                    densespark_cache_namespace(config)
                    for config in configs.values()
                }
            ),
        )


class DenseSparkNative262KRuntimeTests(unittest.TestCase):
    def runtime_model(self) -> SimpleNamespace:
        return SimpleNamespace(
            **model_spec_to_dict(_model()),
            run_identity="run-native-262k",
            resolved_densespark_launch_policy=densespark_expected_launch_policy(),
        )

    def patches(self, home: Path, watchdog: _Watchdog) -> ExitStack:
        repository = (
            home
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--Frozenlock--Qwen3.8-27B-int4-AutoRound"
        )
        (repository / "snapshots" / DENSESPARK_MODEL_REVISION).mkdir(parents=True)
        (home / ".cache").chmod(0o700)

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
            patch("bench.runtime.validate_densespark_warmup_sync_sources")
        )
        stack.enter_context(
            patch("bench.runtime._densespark_startup_watchdog", return_value=watchdog)
        )
        stack.enter_context(patch("bench.runtime.wait_for_endpoint", return_value=0.25))
        return stack

    def test_launch_is_exact_loopback_native_context_and_separate_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            watchdog = _Watchdog()
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
                        stdout=DENSESPARK_WARMUP_SYNC_LOCAL_IMAGE_ID + "\n"
                    )
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
        self.assertIn("127.0.0.1:8000:8000", launch)
        self.assertIn(DENSESPARK_WARMUP_SYNC_LOCAL_IMAGE_ID, launch)
        self.assertEqual(
            list(DENSESPARK_NATIVE_262K_ARGS),
            launch[-len(DENSESPARK_NATIVE_262K_ARGS) :],
        )
        mounts = [
            launch[index + 1]
            for index, argument in enumerate(launch[:-1])
            if argument == "--volume"
        ]
        native_namespace = densespark_cache_namespace(
            densespark_c1_cache_config(DENSESPARK_NATIVE_262K_PROFILE_ID)
        )
        self.assertIn(native_namespace, mounts[-1])
        self.assertNotEqual(
            densespark_compile_cache_path(home=home),
            densespark_compile_cache_path(
                home=home,
                profile_id=DENSESPARK_NATIVE_262K_PROFILE_ID,
            ),
        )
        assert server.native_provenance is not None
        self.assertEqual(
            0.70,
            server.native_provenance["gpu_memory_utilization"],
        )
        self.assertEqual(
            "upstream_0.90_to_0.70_native_context_headroom",
            server.native_provenance["gpu_memory_configuration_reduction"],
        )
        self.assertEqual("auto", server.native_provenance["kv_cache_dtype"])
        self.assertEqual(
            DENSESPARK_NATIVE_CONTEXT,
            server.native_provenance["max_model_len"],
        )
        self.assertEqual(
            "bfloat16",
            server.native_provenance["mamba_ssm_cache_dtype"],
        )


if __name__ == "__main__":
    unittest.main()
