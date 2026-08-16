from __future__ import annotations

import copy
from dataclasses import replace
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock, call, patch

from bench.manifest import ManifestError, load_models, load_suite
from bench.runner import results_lock_path
from bench.trtllm_direct import (
    DIRECT_ARGS,
    DIRECT_AUDIO_PATH,
    DIRECT_AUDIO_SHA256,
    DIRECT_EXPECTED_TRANSCRIPTION,
    DIRECT_IMAGE,
    DIRECT_IMAGE_DIGEST,
    DIRECT_KV_CACHE_OPTIONS,
    DIRECT_MODEL_ID,
    DIRECT_PROMPT,
    DIRECT_REVISION,
    DIRECT_SOURCE,
    DirectTrtllmError,
    WorkerOutcome,
    _character_edit_distance,
    _cleanup_owned_container,
    _docker_command,
    _invoke_worker,
    _normalize_transcription,
    _validate_suite,
    _validate_worker_result,
    _worker_environment,
    run_direct_trtllm,
    verify_direct_profile,
)
from sparkbench import (
    DEFAULT_AUDIO_SUITE,
    build_parser,
    command_benchmark,
    command_plan,
    command_trtllm_direct,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "manifests" / "models.toml"
SUITE_PATH = ROOT / "manifests" / "suites" / "audio_asr.toml"
LOGIC_HASH = "b" * 64


def _completed(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _case(case_id: str = "spark-tts-chinese-exact--fixture") -> dict[str, object]:
    return {
        "id": "spark-tts-chinese-exact",
        "case_id": case_id,
        "kind": "capability",
        "requires": ["chat", "audio"],
        "warmups": 1,
        "repetitions": 3,
        "concurrency": 1,
        "prompt_repetitions": 0,
        "max_output_tokens": 128,
        "temperature": 0.0,
    }


def _valid_worker_result(
    *, case_id: str = "spark-tts-chinese-exact--fixture"
) -> dict[str, object]:
    expected = _normalize_transcription(DIRECT_EXPECTED_TRANSCRIPTION)
    normalized_hash = hashlib.sha256(expected.encode()).hexdigest()
    requests = []
    for repetition in range(3):
        elapsed_s = 2.0 + repetition
        completion_tokens = 24 + repetition
        requests.append(
            {
                "repetition": repetition,
                "elapsed_s": elapsed_s,
                "completion_tokens": completion_tokens,
                "output_tps": completion_tokens / elapsed_s,
                "finish_reason": "stop",
                "transcription_exact": True,
                "normalized_output_length": len(expected),
                "character_edit_distance": 0,
                "character_error_rate": 0.0,
                "output_sha256": hashlib.sha256(
                    f"output-{repetition}".encode()
                ).hexdigest(),
                "normalized_transcription_sha256": normalized_hash,
                "token_ids_sha256": hashlib.sha256(
                    f"tokens-{repetition}".encode()
                ).hexdigest(),
                "expected_transcription_sha256": normalized_hash,
            }
        )
    warmup = copy.deepcopy(requests[0])
    warmup.update(
        {
            "repetition": -1,
            "output_sha256": hashlib.sha256(b"warmup-output").hexdigest(),
            "token_ids_sha256": hashlib.sha256(b"warmup-tokens").hexdigest(),
        }
    )
    return {
        "status": "complete",
        "logic_sha256": LOGIC_HASH,
        "audio_sha256": DIRECT_AUDIO_SHA256,
        "case_id": case_id,
        "load_time_s": 12.0,
        "input_prepare_time_s": 0.5,
        "runtime": {
            "python": "3.12.3",
            "tensorrt_llm": "1.3.0rc13",
            "torch": "fixture",
        },
        "memory": {
            "load_peak_allocated_bytes": None,
            "availability": "unavailable",
        },
        "warmups_completed": 1,
        "warmup": warmup,
        "requests": requests,
        "measured_wall_s": sum(item["elapsed_s"] for item in requests) + 0.25,
    }


def _inexact_worker_result(
    *, case_id: str = "spark-tts-chinese-exact--fixture"
) -> dict[str, object]:
    worker = _valid_worker_result(case_id=case_id)
    expected = _normalize_transcription(DIRECT_EXPECTED_TRANSCRIPTION)
    actual = "deterministicbutincorrecttranscription"
    actual_hash = hashlib.sha256(actual.encode()).hexdigest()
    distance = _character_edit_distance(actual, expected)
    for trial in [worker["warmup"], *worker["requests"]]:
        trial.update(
            {
                "transcription_exact": False,
                "normalized_output_length": len(actual),
                "character_edit_distance": distance,
                "character_error_rate": distance / len(expected),
                "normalized_transcription_sha256": actual_hash,
            }
        )
    return worker


class DirectTrtllmTests(unittest.TestCase):
    def setUp(self) -> None:
        popen_guard = patch(
            "bench.trtllm_direct.subprocess.Popen",
            side_effect=AssertionError("Docker execution must be explicitly mocked"),
        )
        control_guard = patch(
            "bench.trtllm_direct._run_docker_control",
            side_effect=AssertionError("Docker control must be explicitly mocked"),
        )
        popen_guard.start()
        control_guard.start()
        self.addCleanup(popen_guard.stop)
        self.addCleanup(control_guard.stop)

    def test_manifest_suite_and_cli_are_gated_to_the_direct_adapter(self) -> None:
        models = load_models(MODEL_PATH)
        model = models[DIRECT_MODEL_ID]
        suite = load_suite(SUITE_PATH)

        self.assertEqual(model.backend, "trtllm")
        self.assertEqual(model.support_status, "spark_trtllm_direct")
        self.assertEqual(model.lifecycle, "docker")
        self.assertEqual(model.source, DIRECT_SOURCE)
        self.assertEqual(model.revision, DIRECT_REVISION)
        self.assertEqual(model.image, DIRECT_IMAGE)
        self.assertEqual(model.image_digest, DIRECT_IMAGE_DIGEST)
        self.assertEqual(model.endpoint, "offline://trtllm")
        self.assertEqual(model.tasks, ("chat", "audio"))
        self.assertEqual(model.args, DIRECT_ARGS)
        self.assertEqual(
            DIRECT_PROMPT,
            "Transcribe the audio clip into text, please don't add other text.",
        )
        self.assertNotIn("<|", DIRECT_PROMPT)
        self.assertEqual(
            dict(DIRECT_KV_CACHE_OPTIONS),
            {
                "free_gpu_memory_fraction": 0.6,
                "enable_block_reuse": False,
            },
        )

        _validate_suite(suite)
        case = suite.cases[0]
        self.assertEqual(case.warmups, 1)
        self.assertEqual(case.repetitions, 3)
        self.assertEqual(case.concurrency, 1)
        self.assertEqual(case.temperature, 0.0)

        parsed = build_parser().parse_args(["trtllm-direct", DIRECT_MODEL_ID])
        self.assertIs(parsed.function, command_trtllm_direct)
        self.assertEqual(parsed.suite, DEFAULT_AUDIO_SUITE)

        common = {
            "model": DIRECT_MODEL_ID,
            "models": MODEL_PATH,
            "suite": SUITE_PATH,
            "results": ROOT / "results",
        }
        with (
            patch("sparkbench.create_plan") as create_plan,
            patch("sparkbench.execute_plan") as execute_plan,
        ):
            for command in (command_plan, command_benchmark):
                with self.subTest(command=command.__name__):
                    with self.assertRaisesRegex(ManifestError, "trtllm-direct"):
                        command(types.SimpleNamespace(**common))
            create_plan.assert_not_called()
            execute_plan.assert_not_called()

        with patch("bench.trtllm_direct._repository_path") as repository:
            with self.assertRaisesRegex(DirectTrtllmError, "certified shape"):
                verify_direct_profile(replace(model, image_digest="sha256:" + "0" * 64))
        repository.assert_not_called()
        with self.assertRaisesRegex(DirectTrtllmError, "certified workload"):
            _validate_suite(
                replace(suite, cases=(replace(case, repetitions=2),))
            )

    def test_docker_argv_is_digest_pinned_offline_and_minimally_mounted(self) -> None:
        model = load_models(MODEL_PATH)[DIRECT_MODEL_ID]
        run_id = "run-1234567890abcdef"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker_dir = root / "worker"
            repository = root / "model-repository"
            command, cidfile, labels = _docker_command(
                model=model,
                run_id=run_id,
                worker_dir=worker_dir,
                repository=repository,
            )

        self.assertEqual(command[:2], ["docker", "run"])
        self.assertEqual(command[command.index("--pull") + 1], "never")
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("--read-only", command)
        self.assertIn("no-new-privileges", command)
        self.assertEqual(command[command.index("--cap-drop") + 1], "ALL")
        self.assertNotIn("--publish", command)
        self.assertNotIn("--privileged", command)
        self.assertFalse(any("docker.sock" in argument for argument in command))

        image_reference = f"{DIRECT_IMAGE}@{DIRECT_IMAGE_DIGEST}"
        self.assertIn(image_reference, command)
        self.assertRegex(image_reference, r"@sha256:[0-9a-f]{64}$")
        self.assertNotIn("--entrypoint", command)
        self.assertEqual(command[command.index(image_reference) + 1], "python3")
        self.assertEqual(command[command.index("--cidfile") + 1], str(cidfile))
        self.assertEqual(cidfile, worker_dir / "container.cid")

        mounted = [
            command[index + 1]
            for index, argument in enumerate(command)
            if argument == "--volume"
        ]
        self.assertEqual(
            mounted,
            [
                f"{repository}:/model_repo:ro",
                f"{DIRECT_AUDIO_PATH}:/inputs/prompt_audio.wav:ro",
                f"{ROOT}:/sparkbench:ro",
                f"{worker_dir}:/output:rw",
            ],
        )
        self.assertTrue(all(value.endswith(":ro") for value in mounted[:3]))
        self.assertEqual([value for value in mounted if value.endswith(":rw")], [mounted[3]])

        self.assertEqual(
            labels,
            {
                "io.sparkbench.managed": "true",
                "io.sparkbench.run_id": run_id,
                "io.sparkbench.profile": DIRECT_MODEL_ID,
            },
        )
        label_arguments = [
            command[index + 1]
            for index, argument in enumerate(command)
            if argument == "--label"
        ]
        self.assertEqual(
            set(label_arguments),
            {f"{key}={value}" for key, value in labels.items()},
        )
        environment_arguments = [
            command[index + 1]
            for index, argument in enumerate(command)
            if argument == "--env"
        ]
        self.assertIn("HF_HUB_OFFLINE=1", environment_arguments)
        self.assertIn("TRANSFORMERS_OFFLINE=1", environment_arguments)
        self.assertIn("TRITON_CACHE_DIR=/tmp/triton-cache", environment_arguments)
        self.assertIn("CUDA_CACHE_PATH=/tmp/cuda-cache", environment_arguments)
        self.assertIn(
            "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-cache",
            environment_arguments,
        )
        self.assertNotIn("HF_TOKEN", environment_arguments)
        self.assertNotIn("HUGGING_FACE_HUB_TOKEN", environment_arguments)
        self.assertNotIn("base64", " ".join(command).lower())

        with patch.dict(
            "bench.trtllm_direct.os.environ",
            {
                "HF_TOKEN": "secret",
                "HTTPS_PROXY": "http://proxy",
                "SAFE_VALUE": "kept",
            },
            clear=True,
        ):
            environment = _worker_environment()
        self.assertNotIn("HF_TOKEN", environment)
        self.assertNotIn("HTTPS_PROXY", environment)
        self.assertEqual(environment["SAFE_VALUE"], "kept")

    def test_worker_result_accepts_one_warmup_and_three_exact_trials(self) -> None:
        _validate_worker_result(
            _valid_worker_result(), case=_case(), logic_hash=LOGIC_HASH
        )

    def test_worker_result_accepts_well_formed_inexact_trials(self) -> None:
        _validate_worker_result(
            _inexact_worker_result(), case=_case(), logic_hash=LOGIC_HASH
        )

    def test_worker_result_rejects_shape_and_semantic_mutations(self) -> None:
        mutations = {
            "failed status": lambda value: value.update(
                {"status": "failed", "error": "fixture failure"}
            ),
            "logic drift": lambda value: value.update({"logic_sha256": "c" * 64}),
            "zero load time": lambda value: value.update({"load_time_s": 0.0}),
            "nonfinite preparation": lambda value: value.update(
                {"input_prepare_time_s": float("nan")}
            ),
            "wrong audio": lambda value: value.update({"audio_sha256": "0" * 64}),
            "wrong case": lambda value: value.update({"case_id": "other-case"}),
            "wrong warmup count": lambda value: value.update({"warmups_completed": 0}),
            "inconsistent warmup accuracy": lambda value: value["warmup"].update(
                {"transcription_exact": False}
            ),
            "two measured trials": lambda value: value.update(
                {"requests": value["requests"][:2]}
            ),
            "wrong repetition": lambda value: value["requests"][1].update(
                {"repetition": 2}
            ),
            "zero latency": lambda value: value["requests"][0].update(
                {"elapsed_s": 0.0}
            ),
            "too many tokens": lambda value: value["requests"][0].update(
                {"completion_tokens": 129}
            ),
            "inconsistent rate": lambda value: value["requests"][0].update(
                {"output_tps": 999.0}
            ),
            "inconsistent transcription flag": lambda value: value["requests"][0].update(
                {"transcription_exact": False}
            ),
            "wrong normalized hash": lambda value: value["requests"][0].update(
                {"normalized_transcription_sha256": "0" * 64}
            ),
            "negative normalized length": lambda value: value["requests"][0].update(
                {"normalized_output_length": -1}
            ),
            "impossible edit distance": lambda value: value["requests"][0].update(
                {"character_edit_distance": 999}
            ),
            "inconsistent character error rate": lambda value: value["requests"][0].update(
                {"character_error_rate": 0.5}
            ),
            "nonfinite character error rate": lambda value: value["requests"][0].update(
                {"character_error_rate": float("inf")}
            ),
            "invalid output hash": lambda value: value["requests"][0].update(
                {"output_sha256": "not-a-hash"}
            ),
            "invalid token hash": lambda value: value["requests"][0].update(
                {"token_ids_sha256": "g" * 64}
            ),
            "truncated": lambda value: value["requests"][0].update(
                {"finish_reason": "length"}
            ),
            "short measured wall": lambda value: value.update(
                {"measured_wall_s": 1.0}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                worker = copy.deepcopy(_valid_worker_result())
                mutate(worker)
                with self.assertRaises(DirectTrtllmError):
                    _validate_worker_result(
                        worker, case=_case(), logic_hash=LOGIC_HASH
                    )

    def test_cleanup_stops_only_the_exact_labeled_container(self) -> None:
        cid = "a" * 64
        labels = {
            "io.sparkbench.managed": "true",
            "io.sparkbench.run_id": "run-1",
            "io.sparkbench.profile": DIRECT_MODEL_ID,
        }
        with tempfile.TemporaryDirectory() as directory:
            cidfile = Path(directory) / "container.cid"
            cidfile.write_text(cid)
            with patch(
                "bench.trtllm_direct._run_docker_control",
                side_effect=[
                    _completed(stdout=json.dumps(labels)),
                    _completed(),
                    _completed(),
                    _completed(),
                    _completed(returncode=1),
                ],
            ) as control:
                cleanup = _cleanup_owned_container(cidfile, labels)

            self.assertTrue(cleanup["ownership_verified"])
            self.assertTrue(cleanup["stop_requested"])
            self.assertTrue(cleanup["remove_requested"])
            self.assertEqual(
                control.call_args_list,
                [
                    call(
                        [
                            "docker",
                            "inspect",
                            "--format",
                            "{{json .Config.Labels}}",
                            cid,
                        ]
                    ),
                    call(["docker", "stop", "--time", "10", cid]),
                    call(["docker", "inspect", cid]),
                    call(["docker", "rm", "--force", cid]),
                    call(["docker", "inspect", cid]),
                ],
            )

            mismatched = {**labels, "io.sparkbench.run_id": "someone-else"}
            with patch(
                "bench.trtllm_direct._run_docker_control",
                return_value=_completed(stdout=json.dumps(mismatched)),
            ) as control:
                cleanup = _cleanup_owned_container(cidfile, labels)
            self.assertTrue(cleanup["ownership_mismatch"])
            self.assertFalse(cleanup["stop_requested"])
            self.assertFalse(cleanup["remove_requested"])
            self.assertEqual(control.call_count, 1)

            cidfile.write_text("../../not-a-container")
            with patch("bench.trtllm_direct._run_docker_control") as control:
                cleanup = _cleanup_owned_container(cidfile, labels)
            self.assertTrue(cleanup["cidfile_invalid"])
            control.assert_not_called()

    def test_worker_timeout_cleans_container_then_terminates_and_reaps_group(self) -> None:
        class FakeProcess:
            pid = 987654
            returncode: int | None = None

            def __init__(self) -> None:
                self.calls = 0

            def communicate(self, timeout: float | None = None):
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired("worker", timeout)
                self.returncode = -signal.SIGTERM
                return (
                    "",
                    f"{DIRECT_AUDIO_PATH} data:audio/wav;base64,SECRET-PAYLOAD",
                )

            def poll(self) -> int | None:
                return self.returncode

        cleanup_proof = {
            "container_found": True,
            "ownership_verified": True,
            "stop_requested": True,
            "remove_requested": True,
            "cleanup_verified": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = FakeProcess()
            with (
                patch(
                    "bench.trtllm_direct.subprocess.Popen", return_value=process
                ) as popen,
                patch(
                    "bench.trtllm_direct._cleanup_owned_container",
                    return_value=cleanup_proof,
                ) as cleanup,
                patch("bench.trtllm_direct.os.killpg") as killpg,
            ):
                outcome = _invoke_worker(
                    command=["docker", "run", "fixture"],
                    cidfile=root / "container.cid",
                    labels={"io.sparkbench.run_id": "run-1"},
                    result_path=root / "result.json",
                    log_dir=root / "logs",
                    timeout_s=1.0,
                )
            stderr = (root / "logs" / "stderr.log").read_text()

        self.assertTrue(outcome.cleanup["timed_out"])
        self.assertTrue(outcome.cleanup["terminate_requested"])
        self.assertFalse(outcome.cleanup["kill_requested"])
        self.assertTrue(outcome.cleanup["process_reaped"])
        self.assertEqual(outcome.cleanup["container"], cleanup_proof)
        self.assertIn("deadline", str(outcome.error))
        cleanup.assert_called()
        killpg.assert_called_once_with(process.pid, signal.SIGTERM)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertNotIn(str(DIRECT_AUDIO_PATH), stderr)
        self.assertNotIn("SECRET-PAYLOAD", stderr)
        self.assertNotIn("data:audio", stderr)

    def test_worker_timeout_escalates_to_sigkill_and_reaps(self) -> None:
        class FakeProcess:
            pid = 876543
            returncode: int | None = None

            def __init__(self) -> None:
                self.calls = 0

            def communicate(self, timeout: float | None = None):
                self.calls += 1
                if self.calls <= 2:
                    raise subprocess.TimeoutExpired("worker", timeout)
                self.returncode = -signal.SIGKILL
                return "", ""

            def poll(self) -> int | None:
                return self.returncode

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = FakeProcess()
            with (
                patch("bench.trtllm_direct.subprocess.Popen", return_value=process),
                patch(
                    "bench.trtllm_direct._cleanup_owned_container",
                    return_value={"container_found": False},
                ),
                patch("bench.trtllm_direct.os.killpg") as killpg,
            ):
                outcome = _invoke_worker(
                    command=["docker", "run", "fixture"],
                    cidfile=root / "container.cid",
                    labels={"io.sparkbench.run_id": "run-1"},
                    result_path=root / "result.json",
                    log_dir=root / "logs",
                    timeout_s=1.0,
                )

        self.assertTrue(outcome.cleanup["timed_out"])
        self.assertTrue(outcome.cleanup["terminate_requested"])
        self.assertTrue(outcome.cleanup["kill_requested"])
        self.assertTrue(outcome.cleanup["process_reaped"])
        self.assertEqual(
            killpg.call_args_list,
            [
                call(process.pid, signal.SIGTERM),
                call(process.pid, signal.SIGKILL),
            ],
        )

    def test_global_lock_blocks_before_profile_verification_or_docker(self) -> None:
        model = load_models(MODEL_PATH)[DIRECT_MODEL_ID]
        suite = load_suite(SUITE_PATH)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            lock_path = results_lock_path(workspace)
            lock_path.parent.mkdir(parents=True)
            with lock_path.open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with (
                    patch("bench.trtllm_direct.verify_direct_profile") as verify,
                    patch("bench.trtllm_direct._invoke_worker") as invoke,
                ):
                    with self.assertRaisesRegex(DirectTrtllmError, "holds"):
                        run_direct_trtllm(
                            model=model,
                            suite=suite,
                            workspace=workspace,
                            results_root=workspace / "results",
                        )
                verify.assert_not_called()
                invoke.assert_not_called()

    def test_success_artifacts_contain_metrics_and_hashes_but_no_audio_payload(self) -> None:
        model = load_models(MODEL_PATH)[DIRECT_MODEL_ID]
        suite = load_suite(SUITE_PATH)
        verification = {
            "model_revision": DIRECT_REVISION,
            "image": DIRECT_IMAGE,
            "image_digest": DIRECT_IMAGE_DIGEST,
            "audio_sha256": DIRECT_AUDIO_SHA256,
            "worker_logic_sha256": LOGIC_HASH,
        }

        def invoke(**kwargs: object) -> WorkerOutcome:
            config_path = Path(kwargs["result_path"]).with_name("config.json")
            config = json.loads(config_path.read_text())
            return WorkerOutcome(
                result=_valid_worker_result(case_id=config["case"]["case_id"]),
                cleanup={
                    "pid": 1234,
                    "returncode": 0,
                    "timed_out": False,
                    "terminate_requested": False,
                    "kill_requested": False,
                    "process_reaped": True,
                    "container": {
                        "container_found": False,
                        "container_absent": True,
                    },
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            telemetry = Mock()
            with (
                patch(
                    "bench.trtllm_direct.verify_direct_profile",
                    return_value=verification,
                ),
                patch("bench.trtllm_direct._preflight"),
                patch("bench.trtllm_direct._repository_path", return_value=workspace),
                patch("bench.trtllm_direct._invoke_worker", side_effect=invoke),
                patch(
                    "bench.trtllm_direct.TelemetrySampler",
                    return_value=telemetry,
                ),
                patch(
                    "bench.trtllm_direct._telemetry_summaries", return_value={}
                ),
            ):
                summary = run_direct_trtllm(
                    model=model,
                    suite=suite,
                    workspace=workspace,
                    results_root=workspace / "results",
                    timeout_s=60,
                )
            run_dir = Path(summary["run_dir"])
            canonical = "\n".join(
                (run_dir / name).read_text()
                for name in ("plan.json", "events.jsonl", "summary.json")
            )
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["cases"][0]["requests"], 3)
        self.assertEqual(summary["cases"][0]["warmups"], 1)
        self.assertEqual(
            len([event for event in events if event["event"] == "request_complete"]),
            3,
        )
        self.assertNotIn(str(DIRECT_AUDIO_PATH), canonical)
        self.assertNotIn("data:audio", canonical)
        self.assertNotIn("base64,", canonical.lower())
        self.assertNotIn('"audio_path"', canonical)
        self.assertNotIn(DIRECT_EXPECTED_TRANSCRIPTION, canonical)
        telemetry.start.assert_called_once()
        telemetry.stop.assert_called_once()

    def test_inexact_artifacts_are_partial_terminal_and_privacy_safe(self) -> None:
        model = load_models(MODEL_PATH)[DIRECT_MODEL_ID]
        suite = load_suite(SUITE_PATH)
        verification = {
            "model_revision": DIRECT_REVISION,
            "image": DIRECT_IMAGE,
            "image_digest": DIRECT_IMAGE_DIGEST,
            "audio_sha256": DIRECT_AUDIO_SHA256,
            "worker_logic_sha256": LOGIC_HASH,
        }

        def invoke(**kwargs: object) -> WorkerOutcome:
            config_path = Path(kwargs["result_path"]).with_name("config.json")
            config = json.loads(config_path.read_text())
            return WorkerOutcome(
                result=_inexact_worker_result(
                    case_id=config["case"]["case_id"]
                ),
                cleanup={
                    "pid": 1234,
                    "returncode": 0,
                    "timed_out": False,
                    "terminate_requested": False,
                    "kill_requested": False,
                    "process_reaped": True,
                    "container": {
                        "container_found": False,
                        "container_absent": True,
                    },
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            telemetry = Mock()
            with (
                patch(
                    "bench.trtllm_direct.verify_direct_profile",
                    return_value=verification,
                ),
                patch("bench.trtllm_direct._preflight"),
                patch("bench.trtllm_direct._repository_path", return_value=workspace),
                patch("bench.trtllm_direct._invoke_worker", side_effect=invoke),
                patch(
                    "bench.trtllm_direct.TelemetrySampler",
                    return_value=telemetry,
                ),
                patch(
                    "bench.trtllm_direct._telemetry_summaries", return_value={}
                ),
            ):
                summary = run_direct_trtllm(
                    model=model,
                    suite=suite,
                    workspace=workspace,
                    results_root=workspace / "results",
                    timeout_s=60,
                )
            run_dir = Path(summary["run_dir"])
            canonical = "\n".join(
                (run_dir / name).read_text()
                for name in ("plan.json", "events.jsonl", "summary.json")
            )
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]

        request_events = [
            event for event in events if event["event"] == "request_complete"
        ]
        self.assertEqual(summary["status"], "partial")
        self.assertIsNone(summary["error"])
        self.assertFalse(summary["cases"][0]["validation_passed"])
        self.assertEqual(summary["cases"][0]["exact_matches"], 0)
        self.assertGreater(summary["cases"][0]["median_character_edit_distance"], 0)
        self.assertGreater(summary["cases"][0]["median_character_error_rate"], 0)
        self.assertEqual(len(request_events), 3)
        self.assertTrue(
            all(event["validation"]["passed"] is False for event in request_events)
        )
        self.assertTrue(
            all(
                event["validation"]["reason"] == "transcription_mismatch"
                for event in request_events
            )
        )
        self.assertEqual(
            [event for event in events if event["event"] == "run_complete"][-1][
                "status"
            ],
            "partial",
        )
        self.assertFalse(any(event["event"] == "run_aborted" for event in events))
        self.assertIn('"normalized_output_length"', canonical)
        self.assertIn('"character_edit_distance"', canonical)
        self.assertIn('"character_error_rate"', canonical)
        self.assertNotIn(str(DIRECT_AUDIO_PATH), canonical)
        self.assertNotIn(DIRECT_EXPECTED_TRANSCRIPTION, canonical)
        self.assertNotIn("deterministicbutincorrecttranscription", canonical)


if __name__ == "__main__":
    unittest.main()
