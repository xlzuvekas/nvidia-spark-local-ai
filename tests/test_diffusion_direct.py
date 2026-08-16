from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import types
import tempfile
import unittest
from unittest.mock import Mock, patch

from bench.diffusion_direct import (
    DIRECT_ARTIFACT_SHA256,
    DIRECT_RUNTIME,
    DIRECT_RUNTIME_CONFIG_SHA256,
    DIRECT_RUNTIME_EXECUTABLE_SHA256,
    DIRECT_RUNTIME_LOCK_SHA256,
    DIRECT_RUNTIME_VERSIONS,
    WorkerOutcome,
    _invoke_worker,
    _offline_worker_environment,
    _patch_causal_mask_kwargs,
    _validate_worker_result,
    run_direct_diffusion,
    verify_direct_profile,
)
from bench.manifest import ManifestError, load_models, load_suite
from bench.runner import create_plan, results_lock_path
from sparkbench import command_benchmark, command_plan


ROOT = Path(__file__).resolve().parents[1]


class DirectDiffusionTests(unittest.TestCase):
    def test_manifests_fail_closed_vllm_and_expose_one_direct_profile(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        direct = models["nemotron-labs-diffusion-14b-transformers-direct"]

        for model_id in (
            "nemotron-labs-diffusion-8b-bf16",
            "nemotron-labs-diffusion-14b-bf16",
        ):
            model = models[model_id]
            self.assertEqual(model.support_status, "incompatible")
            self.assertEqual(model.tasks, ("diffusion",))
            self.assertEqual(model.backend, "vllm")
        self.assertEqual(direct.backend, "transformers")
        self.assertEqual(direct.lifecycle, "subprocess")
        self.assertEqual(direct.tasks, ("diffusion",))
        self.assertEqual(direct.runtime_python, DIRECT_RUNTIME)
        self.assertEqual(direct.revision, "b69aaebbcfa95a7e5f1de36d6134e4c858ddbc97")
        self.assertIn("model.safetensors", DIRECT_ARTIFACT_SHA256)
        self.assertIn(
            "modeling_nemotron_labs_diffusion.py", DIRECT_ARTIFACT_SHA256
        )

        suite = load_suite(ROOT / "manifests" / "suites" / "diffusion_direct.toml")
        self.assertTrue(all(case.kind == "diffusion" for case in suite.cases))
        self.assertTrue(
            all(case.max_output_tokens % 32 == 0 for case in suite.cases)
        )

    def test_diffusion_suite_rejects_non_block_aligned_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            path.write_text(
                """
schema_version = 1
[suite]
id = "bad"
[[cases]]
id = "bad-block"
kind = "diffusion"
requires = ["diffusion"]
max_output_tokens = 33
temperature = 0.0
concurrency = 1
"""
            )
            with self.assertRaisesRegex(ManifestError, "divisible"):
                load_suite(path)

    def test_generic_plan_path_rejects_direct_and_incompatible_before_probes(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        suite = load_suite(ROOT / "manifests" / "suites" / "diffusion_direct.toml")
        with tempfile.TemporaryDirectory() as directory:
            with patch("bench.runner._image_digest") as image_digest:
                for model_id in (
                    "nemotron-labs-diffusion-8b-bf16",
                    "nemotron-labs-diffusion-14b-bf16",
                    "nemotron-labs-diffusion-14b-transformers-direct",
                ):
                    with self.subTest(model=model_id):
                        with self.assertRaisesRegex(
                            RuntimeError, "Incompatible|diffusion-direct"
                        ):
                            create_plan(
                                model=models[model_id],
                                suite=suite,
                                results_root=Path(directory),
                                models_path=ROOT / "manifests" / "models.toml",
                                suite_path=(
                                    ROOT
                                    / "manifests"
                                    / "suites"
                                    / "diffusion_direct.toml"
                                ),
                            )
                image_digest.assert_not_called()

    def test_cli_plan_and_benchmark_reject_non_runnable_profiles(self) -> None:
        common = {
            "models": ROOT / "manifests" / "models.toml",
            "suite": ROOT / "manifests" / "suites" / "diffusion_direct.toml",
            "results": ROOT / "results",
        }
        with (
            patch("sparkbench.create_plan") as create_plan_mock,
            patch("sparkbench.execute_plan") as execute_plan_mock,
        ):
            for command in (command_plan, command_benchmark):
                for model_id in (
                    "nemotron-labs-diffusion-8b-bf16",
                    "nemotron-labs-diffusion-14b-bf16",
                    "nemotron-labs-diffusion-14b-transformers-direct",
                ):
                    with self.subTest(command=command.__name__, model=model_id):
                        args = types.SimpleNamespace(model=model_id, **common)
                        with self.assertRaisesRegex(
                            ManifestError, "Incompatible|diffusion-direct"
                        ):
                            command(args)
            create_plan_mock.assert_not_called()
            execute_plan_mock.assert_not_called()

    def test_offline_environment_removes_credentials_and_proxies(self) -> None:
        runtime = Path(DIRECT_RUNTIME)
        with patch.dict(
            "bench.diffusion_direct.os.environ",
            {
                "HF_TOKEN": "secret",
                "HTTPS_PROXY": "http://proxy",
                "SAFE_VALUE": "kept",
            },
            clear=True,
        ):
            environment = _offline_worker_environment(runtime)

        self.assertNotIn("HF_TOKEN", environment)
        self.assertNotIn("HTTPS_PROXY", environment)
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(environment["SAFE_VALUE"], "kept")
        self.assertEqual(environment["SPARKBENCH_PARENT_PID"], str(os.getpid()))

    def test_worker_result_must_match_the_frozen_measurement_shape(self) -> None:
        case = {
            "case_id": "block-generation-32--fixture",
            "repetitions": 1,
            "max_output_tokens": 32,
        }
        result = {
            "prompt_tokens": 12,
            "output_tokens": 32,
            "completion_tokens": 32,
            "output_blocks": 1,
            "max_output_tokens": 32,
            "wall_time_s": 2.0,
            "block_generation_output_tps": 16.0,
            "block_generation_blocks_per_s": 0.5,
            "mean_block_generation_latency_s": 2.0,
            "block_generation_metric_source": (
                "completion_tokens_per_end_to_end_block_generation_wall_time"
            ),
            "nfe": 8,
            "nfe_per_block": 8.0,
            "nfe_per_output_token": 0.25,
            "output_tokens_per_nfe": 4.0,
            "output_sha256": "a" * 64,
            "seed": 3407,
            "temperature": 0.0,
            "finish_reason": "length",
        }
        worker = {
            "status": "complete",
            "logic_sha256": "b" * 64,
            "load_time_s": 1.0,
            "runtime": dict(DIRECT_RUNTIME_VERSIONS),
            "cleanup": {
                "allocated_bytes_after_model_delete": 0,
                "reserved_bytes_after_empty_cache": 0,
            },
            "cases": [
                {
                    "case_id": case["case_id"],
                    "measured_wall_time_s": 2.0,
                    "requests": [result],
                }
            ],
        }
        _validate_worker_result(worker, cases=[case], logic_hash="b" * 64)
        result["completion_tokens"] = 31
        with self.assertRaisesRegex(RuntimeError, "exact requested length"):
            _validate_worker_result(worker, cases=[case], logic_hash="b" * 64)

    def test_artifact_symlink_escape_is_rejected_before_hashing(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        model = models["nemotron-labs-diffusion-14b-transformers-direct"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "repository" / "snapshots" / "revision"
            snapshot.mkdir(parents=True)
            outside = root / "outside"
            outside.write_text("untrusted")
            for name in DIRECT_ARTIFACT_SHA256:
                (snapshot / name).write_text("fixture")
            escaped = snapshot / "modeling_nemotron_labs_diffusion.py"
            escaped.unlink()
            escaped.symlink_to(outside)

            def digest(path: Path) -> str:
                if path.name == "pyvenv.cfg":
                    return DIRECT_RUNTIME_CONFIG_SHA256
                if path.name == "uv.lock":
                    return DIRECT_RUNTIME_LOCK_SHA256
                if path == Path(DIRECT_RUNTIME).resolve():
                    return DIRECT_RUNTIME_EXECUTABLE_SHA256
                return DIRECT_ARTIFACT_SHA256.get(path.name, "a" * 64)

            with (
                patch(
                    "bench.diffusion_direct._snapshot_path",
                    return_value=snapshot,
                ),
                patch("bench.diffusion_direct._sha256_file", side_effect=digest),
            ):
                with self.assertRaisesRegex(RuntimeError, "escapes"):
                    verify_direct_profile(model)

    def test_transformers_five_mask_compatibility_shim(self) -> None:
        masking_utils = types.SimpleNamespace()

        def original(**kwargs: object) -> dict[str, object]:
            return kwargs

        masking_utils.create_causal_mask = original
        masking_utils.create_sliding_window_causal_mask = original
        transformers = types.ModuleType("transformers")
        transformers.masking_utils = masking_utils
        with patch.dict("sys.modules", {"transformers": transformers}):
            _patch_causal_mask_kwargs()
            result = masking_utils.create_causal_mask(
                input_embeds="tensor", cache_position="old"
            )

        self.assertEqual(result, {"inputs_embeds": "tensor"})

    def test_worker_timeout_terminates_and_reaps_process_group(self) -> None:
        class FakeProcess:
            pid = 987654
            returncode: int | None = None

            def __init__(self) -> None:
                self.calls = 0

            def communicate(self, timeout: float | None = None):
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired("worker", timeout)
                self.returncode = -15
                return "", ""

            def poll(self) -> int | None:
                return self.returncode

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            result = root / "result.json"
            config.write_text("{}")
            process = FakeProcess()
            with (
                patch("bench.diffusion_direct.subprocess.Popen", return_value=process),
                patch("bench.diffusion_direct.os.killpg") as killpg,
            ):
                outcome = _invoke_worker(
                    runtime=Path("/runtime/python"),
                    config_path=config,
                    result_path=result,
                    log_dir=root / "logs",
                    timeout_s=1,
                )

        self.assertTrue(outcome.cleanup["timed_out"])
        self.assertTrue(outcome.cleanup["process_reaped"])
        self.assertTrue(outcome.cleanup["terminate_requested"])
        killpg.assert_called_once_with(process.pid, 15)

    def test_orchestrator_journals_metrics_and_process_cleanup(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        model = models["nemotron-labs-diffusion-14b-transformers-direct"]
        suite = load_suite(ROOT / "manifests" / "suites" / "diffusion_direct.toml")
        verification = {
            "snapshot": "/offline/snapshot",
            "artifacts_sha256": dict(DIRECT_ARTIFACT_SHA256),
            "runtime_python": DIRECT_RUNTIME,
            "runtime_config_sha256": "b" * 64,
            "worker_logic": "/repo/bench/diffusion_direct.py",
            "worker_logic_sha256": "a" * 64,
        }

        def invoke(**kwargs: object) -> WorkerOutcome:
            config = json.loads(Path(kwargs["config_path"]).read_text())
            cases = []
            for case in config["cases"]:
                requests = [
                    {
                        "prompt_tokens": 20,
                        "output_tokens": case["max_output_tokens"],
                        "completion_tokens": case["max_output_tokens"],
                        "output_blocks": case["max_output_tokens"] // 32,
                        "wall_time_s": 4.0,
                        "elapsed_s": 4.0,
                        "block_generation_output_tps": (
                            case["max_output_tokens"] / 4.0
                        ),
                        "block_generation_blocks_per_s": (
                            (case["max_output_tokens"] // 32) / 4.0
                        ),
                        "mean_block_generation_latency_s": (
                            4.0 / (case["max_output_tokens"] // 32)
                        ),
                        "block_generation_metric_source": (
                            "completion_tokens_per_end_to_end_block_generation_wall_time"
                        ),
                        "nfe": 16,
                        "nfe_per_block": (
                            16 / (case["max_output_tokens"] // 32)
                        ),
                        "nfe_per_output_token": (
                            16 / case["max_output_tokens"]
                        ),
                        "output_tokens_per_nfe": (
                            case["max_output_tokens"] / 16
                        ),
                        "nfe_per_s": 4.0,
                        "output_sha256": "c" * 64,
                        "max_output_tokens": case["max_output_tokens"],
                        "seed": 3407,
                        "temperature": 0.0,
                        "finish_reason": "length",
                    }
                    for _ in range(case["repetitions"])
                ]
                cases.append(
                    {
                        "case_id": case["case_id"],
                        "requests": requests,
                        "measured_wall_time_s": 12.0,
                    }
                )
            return WorkerOutcome(
                result={
                    "status": "complete",
                    "logic_sha256": "a" * 64,
                    "load_time_s": 10.0,
                    "runtime": dict(DIRECT_RUNTIME_VERSIONS),
                    "memory": {"generation_peak_allocated_bytes": 123},
                    "cases": cases,
                    "cleanup": {
                        "allocated_bytes_after_model_delete": 0,
                        "reserved_bytes_after_empty_cache": 0,
                    },
                },
                cleanup={
                    "pid": 1234,
                    "returncode": 0,
                    "timed_out": False,
                    "terminate_requested": False,
                    "kill_requested": False,
                    "process_reaped": True,
                    "cuda_context_cleanup": "worker_process_reaped",
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            results = workspace / "results"
            workspace.mkdir()
            telemetry = Mock()
            with (
                patch(
                    "bench.diffusion_direct.verify_direct_profile",
                    return_value=verification,
                ),
                patch("bench.diffusion_direct._preflight"),
                patch("bench.diffusion_direct._invoke_worker", side_effect=invoke),
                patch(
                    "bench.diffusion_direct.TelemetrySampler",
                    return_value=telemetry,
                ),
            ):
                summary = run_direct_diffusion(
                    model=model,
                    suite=suite,
                    workspace=workspace,
                    results_root=results,
                    timeout_s=60,
                )

            events = [
                json.loads(line)
                for line in (Path(summary["run_dir"]) / "events.jsonl")
                .read_text()
                .splitlines()
            ]

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["load_time_s"], 10.0)
        self.assertTrue(summary["cleanup_proof"]["process_reaped"])
        self.assertEqual(summary["cases"][0]["nfe"], 48)
        self.assertEqual(summary["cases"][0]["nfe_per_block"], 4.0)
        self.assertIn("median_block_generation_output_tps", summary["cases"][0])
        self.assertIn(
            "median_mean_block_generation_latency_s", summary["cases"][0]
        )
        self.assertNotIn("decode_tps", summary["cases"][0])
        self.assertIn("worker_cleanup", [event["event"] for event in events])
        self.assertIn("run_complete", [event["event"] for event in events])
        telemetry.start.assert_called_once()
        telemetry.stop.assert_called_once()

    def test_global_lock_blocks_worker_before_verification(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        model = models["nemotron-labs-diffusion-14b-transformers-direct"]
        suite = load_suite(ROOT / "manifests" / "suites" / "diffusion_direct.toml")
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            lock_path = results_lock_path(workspace)
            lock_path.parent.mkdir(parents=True)
            with lock_path.open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch(
                    "bench.diffusion_direct.verify_direct_profile"
                ) as verify:
                    with self.assertRaisesRegex(RuntimeError, "holds"):
                        run_direct_diffusion(
                            model=model,
                            suite=suite,
                            workspace=workspace,
                            results_root=workspace / "results",
                        )
                verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
