from __future__ import annotations

from pathlib import Path
import io
import json
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.client import BenchmarkRequestError
from bench.host_safety import HostMemorySample, HostSafetyError
from bench.journal import Journal, content_hash
from bench.report import summarize_run
from bench.runner import (
    PreflightError,
    _estimated_context_tokens,
    _preflight,
    _prompt,
    _record_host_safety_breach,
    execute_plan,
    results_lock_path,
)
from bench.runtime import (
    ManagedServer,
    RuntimeErrorWithContext,
    connect_ollama,
    start_vllm,
)


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class _JSONResponse(io.BytesIO):
    def __init__(self, payload: dict[str, object]):
        super().__init__(json.dumps(payload).encode())

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class _FakeHostSafetyWatchdog:
    def __init__(self) -> None:
        self.failure: HostSafetyError | None = None
        self.abort_callback_error: BaseException | None = None
        self.abort_callback = None
        self.start_calls = 0
        self.stop_calls = 0

    @property
    def tripped(self) -> bool:
        return self.failure is not None

    def start(self) -> _FakeHostSafetyWatchdog:
        self.start_calls += 1
        if self.failure is not None:
            raise self.failure
        return self

    def stop(self) -> None:
        self.stop_calls += 1

    def register_abort_callback(self, callback: object) -> None:
        if not callable(callback):
            raise TypeError("abort callback must be callable")
        self.abort_callback = callback
        if self.failure is not None:
            try:
                callback()
            except BaseException as error:
                self.abort_callback_error = error

    def raise_if_tripped(self) -> None:
        if self.failure is not None:
            raise self.failure

    def trip(self, error: HostSafetyError) -> None:
        if self.failure is not None:
            return
        self.failure = error
        if self.abort_callback is not None:
            try:
                self.abort_callback()
            except BaseException as callback_error:
                self.abort_callback_error = callback_error


class RuntimeSafetyTests(unittest.TestCase):
    def test_managed_server_refuses_to_stop_unowned_container(self) -> None:
        server = ManagedServer(
            backend="vllm",
            base_url="http://127.0.0.1:8000/v1",
            container_id="container-id",
            run_identity="run-1",
        )
        with patch("bench.runtime._run", return_value=_completed(stdout="false\n")) as run:
            with self.assertRaisesRegex(RuntimeErrorWithContext, "Refusing to stop"):
                server.stop()

        self.assertEqual(run.call_count, 1)
        self.assertEqual(server.container_id, "container-id")

    def test_managed_server_stops_only_after_ownership_check(self) -> None:
        server = ManagedServer(
            backend="vllm",
            base_url="http://127.0.0.1:8000/v1",
            container_id="container-id",
            run_identity="run-1",
        )
        with patch(
            "bench.runtime._run",
            side_effect=[_completed(stdout="true run-1\n"), _completed(), _completed()],
        ) as run:
            server.stop()

        self.assertEqual(
            [item.args[0][:2] for item in run.call_args_list],
            [["docker", "inspect"], ["docker", "stop"], ["docker", "rm"]],
        )
        self.assertIsNone(server.container_id)

    def test_sglang_host_safety_interrupt_checks_ownership_and_preserves_cleanup_state(
        self,
    ) -> None:
        server = ManagedServer(
            backend="sglang",
            base_url="http://127.0.0.1:30000/v1",
            container_id="container-id",
            run_identity="run-1",
            authorization="Bearer fixture-key",
            api_key="fixture-key",
            api_key_path=Path("/mock/api-key"),
        )
        with patch(
            "bench.runtime._run",
            side_effect=[_completed(stdout="true run-1\n"), _completed()],
        ) as run:
            server.interrupt_owned()

        self.assertEqual(
            [item.args[0][:2] for item in run.call_args_list],
            [["docker", "inspect"], ["docker", "stop"]],
        )
        self.assertEqual(
            run.call_args_list[1].args[0],
            ["docker", "stop", "--time", "0", "container-id"],
        )
        self.assertEqual(server.container_id, "container-id")
        self.assertEqual(server.authorization, "Bearer fixture-key")
        self.assertEqual(server.api_key, "fixture-key")
        self.assertEqual(server.api_key_path, Path("/mock/api-key"))

    def test_sglang_host_safety_interrupt_refuses_an_unowned_container(
        self,
    ) -> None:
        server = ManagedServer(
            backend="sglang",
            base_url="http://127.0.0.1:30000/v1",
            container_id="container-id",
            run_identity="run-1",
        )
        with patch(
            "bench.runtime._run", return_value=_completed(stdout="true other-run\n")
        ) as run:
            with self.assertRaisesRegex(RuntimeErrorWithContext, "Refusing to stop"):
                server.interrupt_owned()

        run.assert_called_once()
        self.assertEqual(server.container_id, "container-id")

    def test_sglang_interrupt_leaves_the_container_for_normal_owned_cleanup(
        self,
    ) -> None:
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
                _completed(stdout="true run-1\n"),
                _completed(),
                _completed(),
            ],
        ) as run:
            server.interrupt_owned()
            server.stop()

        self.assertEqual(
            [item.args[0][:2] for item in run.call_args_list],
            [
                ["docker", "inspect"],
                ["docker", "stop"],
                ["docker", "inspect"],
                ["docker", "stop"],
                ["docker", "rm"],
            ],
        )
        self.assertEqual(run.call_args_list[1].args[0][3], "0")
        self.assertEqual(run.call_args_list[3].args[0][3], "30")
        self.assertIsNone(server.container_id)

    def test_keep_server_performs_no_lifecycle_command(self) -> None:
        server = ManagedServer(
            backend="vllm",
            base_url="http://127.0.0.1:8000/v1",
            container_id="container-id",
            run_identity="run-1",
        )
        with patch("bench.runtime._run") as run:
            server.stop(keep_server=True)
        run.assert_not_called()
        self.assertEqual(server.container_id, "container-id")

    def test_start_refuses_to_replace_unmanaged_named_container(self) -> None:
        model = SimpleNamespace(source="example/model", image="example/image", args=[])
        with patch("bench.runtime._existing_container", return_value=("abc", False, "")):
            with self.assertRaisesRegex(RuntimeErrorWithContext, "unmanaged container"):
                start_vllm(model, workspace=Path("/unused"))

    def test_no_download_mode_requires_exact_cached_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            repository = workspace / "data" / "huggingface" / "hub" / "models--example--model"
            (repository / "snapshots" / "different-revision").mkdir(parents=True)
            model = SimpleNamespace(
                source="example/model",
                revision="required-revision",
                image="example/image",
                args=[],
                served_name="example/model",
                startup_timeout_s=1,
                cache_dir="project",
            )
            with (
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch(
                    "bench.runtime._run",
                    return_value=_completed(stderr="docker launch should not run", returncode=1),
                ) as run,
            ):
                with self.assertRaisesRegex(RuntimeErrorWithContext, "not marked cached"):
                    start_vllm(model, workspace=workspace, allow_download=False)

            run.assert_not_called()

    def test_no_download_container_launch_forbids_image_pulls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            snapshot = (
                workspace
                / "data"
                / "huggingface"
                / "hub"
                / "models--example--model"
                / "snapshots"
                / "required-revision"
            )
            snapshot.mkdir(parents=True)
            model = SimpleNamespace(
                source="example/model",
                revision="required-revision",
                image="example/image@sha256:" + "a" * 64,
                resolved_image="example/image@sha256:" + "a" * 64,
                args=[],
                served_name="example/model",
                startup_timeout_s=1,
                cache_dir="project",
                run_identity="run-1",
            )
            with (
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime.wait_for_endpoint", return_value=0.1),
                patch("bench.runtime._run", return_value=_completed(stdout="container-id\n")) as run,
            ):
                server = start_vllm(model, workspace=workspace, allow_download=False)

            launch = run.call_args_list[0].args[0]
            self.assertIn("--pull=never", launch)
            self.assertIn(model.resolved_image, launch)
            self.assertEqual(server.container_id, "container-id")

    def test_connect_ollama_marks_only_newly_loaded_model_for_unload(self) -> None:
        model = SimpleNamespace(
            endpoint="http://127.0.0.1:11434/v1",
            source="target:latest",
            revision="abc123",
        )
        tags = {
            "models": [
                {
                    "name": "target:latest",
                    "model": "target:latest",
                    "digest": "sha256:abc123456",
                }
            ]
        }
        for loaded, expected_unload in (([], True), ([{"name": "target:latest"}], False)):
            with self.subTest(already_loaded=bool(loaded)):
                with (
                    patch("bench.runtime.endpoint_ready", return_value=True),
                    patch(
                        "bench.runtime.urllib.request.urlopen",
                        side_effect=[_JSONResponse(tags), _JSONResponse({"models": loaded})],
                    ),
                ):
                    server = connect_ollama(model)

                self.assertEqual(server.unload_ollama, expected_unload)

    def test_connect_ollama_fails_closed_when_initial_load_state_is_unknown(self) -> None:
        model = SimpleNamespace(
            endpoint="http://127.0.0.1:11434/v1",
            source="target:latest",
            revision="abc123",
        )
        tags = {
            "models": [
                {
                    "name": "target:latest",
                    "digest": "sha256:abc123456",
                }
            ]
        }
        with (
            patch("bench.runtime.endpoint_ready", return_value=True),
            patch(
                "bench.runtime.urllib.request.urlopen",
                side_effect=[_JSONResponse(tags), ConnectionResetError("ps unavailable")],
            ),
        ):
            with self.assertRaisesRegex(RuntimeErrorWithContext, "loaded|ownership|state"):
                connect_ollama(model)

    def test_ollama_stop_waits_until_owned_model_is_confirmed_unloaded(self) -> None:
        server = ManagedServer(
            backend="ollama",
            base_url="http://127.0.0.1:11434/v1",
            ollama_model="target:latest",
            unload_ollama=True,
        )

        cli_ps_calls = 0

        def command_result(command: list[str], **kwargs: object):
            nonlocal cli_ps_calls
            if command[:2] == ["ollama", "ps"]:
                cli_ps_calls += 1
                return _completed(
                    stdout=(
                        "NAME ID SIZE PROCESSOR CONTEXT UNTIL\n"
                        + (
                            "target:latest abc 1 GB 100% GPU 4096 4 minutes\n"
                            if cli_ps_calls == 1
                            else ""
                        )
                    )
                )
            return _completed()

        with (
            patch("bench.runtime._run", side_effect=command_result) as run,
            patch(
                "bench.runtime.urllib.request.urlopen",
                side_effect=[
                    _JSONResponse({"models": [{"name": "target:latest"}]}),
                    _JSONResponse({"models": []}),
                ],
            ) as urlopen,
            patch("bench.runtime.time.sleep"),
        ):
            server.stop()

        commands = [item.args[0] for item in run.call_args_list]
        self.assertIn(["ollama", "stop", "target:latest"], commands)
        verified_via_cli = sum(
            command[:2] == ["ollama", "ps"] for command in commands
        ) >= 2
        self.assertTrue(
            verified_via_cli or urlopen.call_count >= 2,
            "Ollama teardown returned before observing loaded then unloaded states",
        )

    def test_ollama_model_not_loaded_by_sparkbench_is_left_running(self) -> None:
        server = ManagedServer(
            backend="ollama",
            base_url="http://127.0.0.1:11434/v1",
            ollama_model="target:latest",
            unload_ollama=False,
        )
        with (
            patch("bench.runtime._run") as run,
            patch("bench.runtime.urllib.request.urlopen") as urlopen,
        ):
            server.stop()

        run.assert_not_called()
        urlopen.assert_not_called()


class PlanSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        network_guard = patch(
            "urllib.request.urlopen",
            side_effect=AssertionError(
                "plan safety tests must explicitly mock every HTTP request"
            ),
        )
        network_guard.start()
        self.addCleanup(network_guard.stop)

    def _write_runnable_plan(
        self,
        root: Path,
        *,
        backend: str = "ollama",
        case_ids: tuple[str, ...] = ("decode",),
        tasks: tuple[str, ...] = ("chat",),
        kind: str = "decode",
        requires: tuple[str, ...] = ("chat",),
        prompt_repetitions: int = 0,
        host_safety: bool = False,
    ) -> Path:
        model = {
            "id": f"{backend}-target",
            "backend": backend,
            "source": "target:latest" if backend == "ollama" else "example/model",
            "served_name": "target:latest" if backend == "ollama" else "example/model",
            "tasks": list(tasks),
            "max_context": 8192,
            "endpoint": (
                "http://127.0.0.1:11434/v1"
                if backend == "ollama"
                else "http://127.0.0.1:8000/v1"
            ),
        }
        if backend == "vllm":
            model.update(
                {
                    "image": "example/image@sha256:" + "a" * 64,
                    "args": [],
                    "startup_timeout_s": 1,
                    "cache_dir": "project",
                }
            )
        if backend == "sglang":
            model.update(
                {
                    "image": "example/image@sha256:" + "a" * 64,
                    "args": [],
                    "startup_timeout_s": 1,
                    "cache_dir": "project",
                }
            )
        if host_safety:
            model.update(
                {
                    "host_safety_min_memavailable_gib": 14,
                    "host_safety_max_swap_growth_mib": 512,
                    "host_safety_max_starting_swap_mib": 64,
                }
            )
        case_template = {
            "kind": kind,
            "requires": list(requires),
            "warmups": 0,
            "repetitions": 1,
            "max_output_tokens": 8,
            "temperature": 0.0,
            "concurrency": 1,
            "prompt_repetitions": prompt_repetitions,
        }
        cases = [{"id": case_id, **case_template} for case_id in case_ids]
        suite = {
            "id": "suite",
            "description": "",
            "schema_version": 1,
            "cases": cases,
        }
        fingerprint = content_hash({"model": model, "suite": suite})
        frozen_suite = {
            **suite,
            "cases": [
                {
                    **case,
                    "case_id": f"{case['id']}--{content_hash({'model': model, 'case': case}, 12)}",
                }
                for case in cases
            ],
        }
        plan = {
            "fingerprint": fingerprint,
            "model": model,
            "suite": frozen_suite,
            "resolved": {},
        }
        run_dir = root / "run"
        run_dir.mkdir()
        (run_dir / "plan.json").write_text(json.dumps(plan))
        return run_dir

    def test_preflight_rejects_unrelated_containers_before_gpu_check(self) -> None:
        model = SimpleNamespace(estimated_ram_gib=1)
        with patch("bench.runner._command_output", return_value="database\nweb") as output:
            with self.assertRaisesRegex(PreflightError, "database, web"):
                _preflight(model)
        output.assert_called_once()

    def test_results_lock_is_scoped_below_workspace(self) -> None:
        workspace = Path("/tmp/example-workspace")
        self.assertEqual(
            results_lock_path(workspace),
            workspace / "results" / ".sparkbench.lock",
        )

    def test_host_safety_breach_evidence_is_deduplicated_per_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "events.jsonl")
            errors = (
                HostSafetyError(
                    "memavailable_below_minimum",
                    "first synthetic breach",
                    observed_kib=1,
                    limit_kib=2,
                ),
                HostSafetyError(
                    "swap_growth_above_maximum",
                    "second synthetic breach",
                    observed_kib=3,
                    limit_kib=2,
                ),
            )
            for attempt, error in enumerate(errors, start=1):
                journal.append(
                    {"event": "run_start", "attempt": attempt}
                )
                _record_host_safety_breach(
                    journal, error, stage="case_execution"
                )
                _record_host_safety_breach(
                    journal, error, stage="case_execution"
                )

            breaches = [
                event
                for event in journal.events()
                if event.get("event") == "host_safety_breach"
            ]
            self.assertEqual(
                [event["code"] for event in breaches],
                [
                    "memavailable_below_minimum",
                    "swap_growth_above_maximum",
                ],
            )

    def test_ollama_rerank_remains_explicitly_adapter_unimplemented(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_runnable_plan(
                root,
                backend="ollama",
                case_ids=("rerank",),
                tasks=("rerank",),
                kind="capability",
                requires=("rerank",),
                prompt_repetitions=4,
            )
            with (
                patch("bench.runner._preflight") as preflight,
                patch("bench.runner.start_server") as start_server,
            ):
                summary = execute_plan(run_dir, workspace=root)

            self.assertEqual(summary["status"], "no_work")
            self.assertEqual(len(summary["unimplemented_cases"]), 1)
            preflight.assert_not_called()
            start_server.assert_not_called()
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            skipped = [
                event
                for event in events
                if event.get("event") == "case_skipped_adapter_unimplemented"
            ]
            self.assertEqual(skipped[0]["backend"], "ollama")
            self.assertIn("vLLM /score", skipped[0]["reason"])

    def test_ollama_multimodal_embeddings_skip_the_vllm_chat_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_runnable_plan(
                root,
                backend="ollama",
                case_ids=("multimodal-embedding",),
                tasks=("embeddings", "vision"),
                kind="capability",
                requires=("embeddings", "vision"),
                prompt_repetitions=64,
            )
            with (
                patch("bench.runner._preflight") as preflight,
                patch("bench.runner.start_server") as start_server,
            ):
                summary = execute_plan(run_dir, workspace=root)

            self.assertEqual(summary["status"], "no_work")
            self.assertEqual(len(summary["unimplemented_cases"]), 1)
            preflight.assert_not_called()
            start_server.assert_not_called()
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            skipped = next(
                event
                for event in events
                if event.get("event") == "case_skipped_adapter_unimplemented"
            )
            self.assertEqual(skipped["capabilities"], ["embeddings", "vision"])
            self.assertIn("Chat Embeddings", skipped["reason"])

    def test_tampered_plan_is_rejected_before_preflight_or_server_start(self) -> None:
        model = {
            "id": "model",
            "backend": "external",
            "source": "model",
            "served_name": "model",
            "tasks": ["chat"],
            "max_context": 1024,
            "endpoint": "http://127.0.0.1:8000/v1",
        }
        suite = {
            "id": "suite",
            "description": "",
            "schema_version": 1,
            "cases": [
                {
                    "id": "decode",
                    "case_id": "decode--original",
                    "kind": "decode",
                    "requires": ["chat"],
                    "warmups": 0,
                    "repetitions": 1,
                    "max_output_tokens": 1,
                    "temperature": 0.0,
                    "concurrency": 1,
                    "prompt_repetitions": 0,
                }
            ],
        }
        fingerprint_suite = {
            **suite,
            "cases": [{key: value for key, value in suite["cases"][0].items() if key != "case_id"}],
        }
        plan = {
            "fingerprint": content_hash({"model": model, "suite": fingerprint_suite}),
            "model": model,
            "suite": suite,
        }
        plan["model"]["served_name"] = "tampered"

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "plan.json").write_text(json.dumps(plan))
            with (
                patch("bench.runner._preflight") as preflight,
                patch("bench.runner.start_server") as start_server,
            ):
                with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                    execute_plan(run_dir, workspace=Path(directory))

        preflight.assert_not_called()
        start_server.assert_not_called()

    def test_prime_request_failure_journals_abort_and_still_tears_down_ollama(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_runnable_plan(root)
            server = SimpleNamespace(
                backend="ollama",
                base_url="http://127.0.0.1:11434/v1",
                startup_s=0.01,
                container_id=None,
                ollama_model="target:latest",
                unload_ollama=True,
                stop=Mock(),
            )
            telemetry = Mock()
            with (
                patch("bench.runner._preflight"),
                patch("bench.runner.TelemetrySampler", return_value=telemetry),
                patch("bench.runner.start_server", return_value=server),
                patch(
                    "bench.runner._prime_model",
                    side_effect=BenchmarkRequestError("chat request prime-1 disconnected"),
                ),
            ):
                with self.assertRaisesRegex(BenchmarkRequestError, "disconnected"):
                    execute_plan(run_dir, workspace=root)

            server.stop.assert_called_once_with(keep_server=False)
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            aborted = [event for event in events if event.get("event") == "run_aborted"]
            self.assertEqual(len(aborted), 1)
            self.assertEqual(aborted[0]["error_type"], "BenchmarkRequestError")
            self.assertEqual(aborted[0]["stage"], "first_request")
            self.assertIn("disconnected", aborted[0]["error"])
            summary = summarize_run(run_dir)
            self.assertEqual(summary["status"], "aborted")
            self.assertEqual(summary["run_error"]["stage"], "first_request")
            self.assertEqual(summary["completed_cases"], 0)

    def test_preflight_failure_persists_abort_and_truthful_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_runnable_plan(root)
            telemetry = Mock()
            error = PreflightError("fixture GPU process is active")
            with (
                patch("bench.runner._preflight", side_effect=error),
                patch("bench.runner.TelemetrySampler", return_value=telemetry),
                patch("bench.runner.start_server") as start_server,
            ):
                with self.assertRaisesRegex(PreflightError, "GPU process"):
                    execute_plan(run_dir, workspace=root)

            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            aborted = [
                event for event in events if event.get("event") == "run_aborted"
            ]
            summary = json.loads((run_dir / "summary.json").read_text())

        self.assertEqual(len(aborted), 1)
        self.assertEqual(aborted[0]["stage"], "preflight")
        self.assertEqual(aborted[0]["error_type"], "PreflightError")
        self.assertNotIn("run_complete", {event["event"] for event in events})
        self.assertEqual(summary["status"], "aborted")
        self.assertEqual(summary["run_error"]["stage"], "preflight")
        self.assertEqual(summary["completed_cases"], 0)
        start_server.assert_not_called()
        telemetry.start.assert_not_called()

    def test_vllm_startup_failure_persists_abort_and_truthful_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_runnable_plan(root, backend="vllm")
            telemetry = Mock()
            startup_error = RuntimeErrorWithContext(
                "Server exited during startup: unsupported architecture"
            )
            with (
                patch("bench.runner._preflight"),
                patch("bench.runner.TelemetrySampler", return_value=telemetry),
                patch("bench.runner.start_server", side_effect=startup_error),
            ):
                with self.assertRaisesRegex(
                    RuntimeErrorWithContext, "unsupported architecture"
                ):
                    execute_plan(run_dir, workspace=root)

            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            aborted = [
                event for event in events if event.get("event") == "run_aborted"
            ]
            summary = json.loads((run_dir / "summary.json").read_text())

        self.assertEqual(len(aborted), 1)
        self.assertEqual(aborted[0]["stage"], "server_start")
        self.assertEqual(aborted[0]["error_type"], "RuntimeErrorWithContext")
        self.assertIn("unsupported architecture", aborted[0]["error"])
        self.assertNotIn("run_complete", {event["event"] for event in events})
        self.assertEqual(summary["status"], "aborted")
        self.assertEqual(summary["run_error"]["stage"], "server_start")
        self.assertEqual(summary["completed_cases"], 0)
        telemetry.start.assert_called_once()
        telemetry.stop.assert_called_once()

    def test_telemetry_start_failure_is_a_durable_startup_abort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_runnable_plan(root)
            telemetry = Mock()
            telemetry.start.side_effect = RuntimeError("telemetry thread failed")
            with (
                patch("bench.runner._preflight"),
                patch("bench.runner.TelemetrySampler", return_value=telemetry),
                patch("bench.runner.start_server") as start_server,
            ):
                with self.assertRaisesRegex(RuntimeError, "telemetry thread"):
                    execute_plan(run_dir, workspace=root)

            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            aborted = [
                event for event in events if event.get("event") == "run_aborted"
            ]
            summary = json.loads((run_dir / "summary.json").read_text())

        self.assertEqual(len(aborted), 1)
        self.assertEqual(aborted[0]["stage"], "telemetry_start")
        self.assertEqual(summary["status"], "aborted")
        start_server.assert_not_called()
        telemetry.stop.assert_not_called()

    def test_sglang_host_safety_start_failure_aborts_before_server_launch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_runnable_plan(
                root, backend="sglang", host_safety=True
            )
            telemetry = Mock()
            watchdog = _FakeHostSafetyWatchdog()
            sample = HostMemorySample(
                mem_total_kib=128 * 1024**2,
                memavailable_kib=14 * 1024**2 - 1,
                swap_total_kib=8 * 1024**2,
                swap_free_kib=8 * 1024**2,
            )
            watchdog.failure = HostSafetyError(
                "memavailable_below_minimum",
                "synthetic MemAvailable breach",
                sample=sample,
                observed_kib=sample.memavailable_kib,
                limit_kib=14 * 1024**2,
                starting_swap_used_kib=0,
            )
            with (
                patch("bench.runner._preflight"),
                patch("bench.runner.TelemetrySampler", return_value=telemetry),
                patch(
                    "bench.runner.HostSafetyWatchdog", return_value=watchdog
                ) as constructor,
                patch("bench.runner.start_server") as start_server,
            ):
                with self.assertRaises(HostSafetyError) as raised:
                    execute_plan(run_dir, workspace=root)

            self.assertIs(raised.exception, watchdog.failure)
            constructor.assert_called_once_with(
                min_memavailable_gib=14,
                max_swap_growth_mib=512,
                max_starting_swap_mib=64,
            )
            start_server.assert_not_called()
            self.assertEqual(watchdog.start_calls, 1)
            self.assertGreaterEqual(watchdog.stop_calls, 1)
            telemetry.stop.assert_called_once()
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            breach = [
                event
                for event in events
                if event.get("event") == "host_safety_breach"
            ]
            aborted = [
                event for event in events if event.get("event") == "run_aborted"
            ]
            self.assertEqual(len(breach), 1)
            self.assertEqual(breach[0]["stage"], "host_safety_start")
            self.assertEqual(breach[0]["observed_kib"], sample.memavailable_kib)
            self.assertEqual(len(aborted), 1)
            self.assertEqual(aborted[0]["error_type"], "HostSafetyError")

    def test_sglang_host_safety_rejects_keep_server_before_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_runnable_plan(
                root, backend="sglang", host_safety=True
            )
            telemetry = Mock()
            watchdog = _FakeHostSafetyWatchdog()
            with (
                patch("bench.runner._preflight") as preflight,
                patch("bench.runner.TelemetrySampler", return_value=telemetry),
                patch("bench.runner.HostSafetyWatchdog", return_value=watchdog),
                patch("bench.runner.start_server") as start_server,
            ):
                with self.assertRaisesRegex(
                    PreflightError, "keep-server.*host-safety"
                ):
                    execute_plan(
                        run_dir,
                        workspace=root,
                        keep_server=True,
                    )

            preflight.assert_not_called()
            telemetry.start.assert_not_called()
            start_server.assert_not_called()
            self.assertEqual(watchdog.start_calls, 0)
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            aborted = [
                event for event in events if event.get("event") == "run_aborted"
            ]
            self.assertEqual(len(aborted), 1)
            self.assertEqual(aborted[0]["stage"], "host_safety_policy")

    def test_live_safety_breach_wins_over_socket_error_and_forces_cleanup(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_runnable_plan(
                root,
                backend="sglang",
                case_ids=("first", "must-not-run"),
                host_safety=True,
            )
            telemetry = Mock()
            watchdog = _FakeHostSafetyWatchdog()
            server = SimpleNamespace(
                backend="sglang",
                base_url="http://127.0.0.1:30000/v1",
                startup_s=0.1,
                container_id="container-id",
                run_identity=None,
                ollama_model=None,
                unload_ollama=False,
                authorization="Bearer fixture",
                interrupt_owned=Mock(
                    side_effect=[
                        RuntimeError("synthetic immediate stop failure"),
                        None,
                    ]
                ),
                stop=Mock(),
            )
            first_request = Mock()
            first_request.to_dict.return_value = {}
            safety_error = HostSafetyError(
                "swap_growth_above_maximum",
                "synthetic swap growth breach",
                sample=HostMemorySample(
                    mem_total_kib=128 * 1024**2,
                    memavailable_kib=20 * 1024**2,
                    swap_total_kib=8 * 1024**2,
                    swap_free_kib=8 * 1024**2 - (512 * 1024 + 1),
                ),
                observed_kib=512 * 1024 + 1,
                limit_kib=512 * 1024,
                starting_swap_used_kib=0,
            )

            def start(*args: object, **kwargs: object) -> object:
                model = args[0]
                server.run_identity = model.run_identity
                kwargs["on_server_created"](server)
                kwargs["abort_check"]()
                return server

            def fail_first_case(**kwargs: object) -> None:
                watchdog.trip(safety_error)
                raise BenchmarkRequestError("synthetic socket reset")

            with (
                patch("bench.runner._preflight"),
                patch("bench.runner.TelemetrySampler", return_value=telemetry),
                patch("bench.runner.HostSafetyWatchdog", return_value=watchdog),
                patch("bench.runner.start_server", side_effect=start),
                patch("bench.runner.capture_server_provenance", return_value={}),
                patch("bench.runner._prime_model", return_value=first_request),
                patch(
                    "bench.runner._execute_case", side_effect=fail_first_case
                ) as execute_case,
                patch(
                    "bench.runner.save_server_logs",
                    side_effect=lambda *args, **kwargs: self.assertEqual(
                        server.interrupt_owned.call_count, 2
                    ),
                ),
            ):
                with self.assertRaises(HostSafetyError) as raised:
                    execute_plan(
                        run_dir,
                        workspace=root,
                        continue_on_error=True,
                    )

            self.assertIs(raised.exception, safety_error)
            self.assertEqual(execute_case.call_count, 1)
            self.assertEqual(server.interrupt_owned.call_count, 2)
            server.stop.assert_called_once_with(keep_server=False)
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            names = [event["event"] for event in events]
            self.assertIn("server_stopped", names)
            self.assertNotIn("server_kept", names)
            self.assertNotIn("run_complete", names)
            aborted = next(
                event for event in events if event["event"] == "run_aborted"
            )
            self.assertEqual(aborted["error_type"], "HostSafetyError")
            self.assertEqual(aborted["stage"], "case_execution")
            interrupt_failure = next(
                event
                for event in events
                if event["event"] == "host_safety_interrupt_failed"
            )
            self.assertEqual(interrupt_failure["error_type"], "RuntimeError")
            self.assertNotIn("error", interrupt_failure)

    def test_cleanup_time_safety_trip_supersedes_request_error_durably(
        self,
    ) -> None:
        for continue_on_error in (False, True):
            with self.subTest(continue_on_error=continue_on_error):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    run_dir = self._write_runnable_plan(
                        root, backend="sglang", host_safety=True
                    )
                    telemetry = Mock()
                    watchdog = _FakeHostSafetyWatchdog()
                    server = SimpleNamespace(
                        backend="sglang",
                        base_url="http://127.0.0.1:30000/v1",
                        startup_s=0.1,
                        container_id="container-id",
                        run_identity=None,
                        ollama_model=None,
                        unload_ollama=False,
                        authorization="Bearer fixture",
                        interrupt_owned=Mock(),
                        stop=Mock(),
                    )
                    first_request = Mock()
                    first_request.to_dict.return_value = {}
                    safety_error = HostSafetyError(
                        "memavailable_below_minimum",
                        "synthetic cleanup-time memory breach",
                        sample=HostMemorySample(
                            mem_total_kib=128 * 1024**2,
                            memavailable_kib=14 * 1024**2 - 1,
                            swap_total_kib=8 * 1024**2,
                            swap_free_kib=8 * 1024**2,
                        ),
                        observed_kib=14 * 1024**2 - 1,
                        limit_kib=14 * 1024**2,
                        starting_swap_used_kib=0,
                    )

                    def start(*args: object, **kwargs: object) -> object:
                        model = args[0]
                        server.run_identity = model.run_identity
                        kwargs["on_server_created"](server)
                        kwargs["abort_check"]()
                        return server

                    def trip_during_cleanup(*args: object, **kwargs: object) -> None:
                        watchdog.trip(safety_error)

                    with (
                        patch("bench.runner._preflight"),
                        patch(
                            "bench.runner.TelemetrySampler",
                            return_value=telemetry,
                        ),
                        patch(
                            "bench.runner.HostSafetyWatchdog",
                            return_value=watchdog,
                        ),
                        patch("bench.runner.start_server", side_effect=start),
                        patch(
                            "bench.runner.capture_server_provenance",
                            return_value={},
                        ),
                        patch(
                            "bench.runner._prime_model",
                            return_value=first_request,
                        ),
                        patch(
                            "bench.runner._execute_case",
                            side_effect=BenchmarkRequestError(
                                "synthetic request socket failure"
                            ),
                        ),
                        patch(
                            "bench.runner.save_server_logs",
                            side_effect=trip_during_cleanup,
                        ),
                    ):
                        with self.assertRaises(HostSafetyError) as raised:
                            execute_plan(
                                run_dir,
                                workspace=root,
                                continue_on_error=continue_on_error,
                            )

                    self.assertIs(raised.exception, safety_error)
                    server.interrupt_owned.assert_called_once_with()
                    server.stop.assert_called_once_with(keep_server=False)
                    events = [
                        json.loads(line)
                        for line in (
                            run_dir / "events.jsonl"
                        ).read_text().splitlines()
                    ]
                    aborted = [
                        event
                        for event in events
                        if event.get("event") == "run_aborted"
                    ]
                    self.assertEqual(len(aborted), 1)
                    self.assertEqual(aborted[0]["error_type"], "HostSafetyError")
                    self.assertEqual(aborted[0]["stage"], "server_cleanup")
                    summary = json.loads((run_dir / "summary.json").read_text())
                    self.assertEqual(
                        summary["run_error"]["error_type"], "HostSafetyError"
                    )

    def test_completed_plan_resume_is_idempotent_and_starts_no_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_runnable_plan(root)
            plan = json.loads((run_dir / "plan.json").read_text())
            case_id = plan["suite"]["cases"][0]["case_id"]
            journal = Journal(run_dir / "events.jsonl")
            journal.append({"event": "run_start", "completed_cases_at_resume": []})
            journal.append(
                {
                    "event": "case_complete",
                    "case_id": case_id,
                    "attempt_id": "done",
                    "kind": "decode",
                    "elapsed_s": 1.0,
                }
            )
            journal.append({"event": "run_complete", "status": "completed"})
            with (
                patch("bench.runner._preflight") as preflight,
                patch("bench.runner.start_server") as start_server,
            ):
                summary = execute_plan(run_dir, workspace=root)

            self.assertEqual(summary["status"], "complete")
            preflight.assert_not_called()
            start_server.assert_not_called()

    def test_resume_recovers_external_run_when_all_cases_finished_before_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_runnable_plan(root, backend="external")
            plan = json.loads((run_dir / "plan.json").read_text())
            case_id = plan["suite"]["cases"][0]["case_id"]
            journal = Journal(run_dir / "events.jsonl")
            journal.append({"event": "run_start", "completed_cases_at_resume": []})
            journal.append(
                {
                    "event": "case_complete",
                    "case_id": case_id,
                    "attempt_id": "done-before-crash",
                    "kind": "decode",
                    "elapsed_s": 1.0,
                }
            )
            with (
                patch("bench.runner._preflight") as preflight,
                patch("bench.runner.start_server") as start_server,
            ):
                summary = execute_plan(run_dir, workspace=root)

            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["run_completion_status"], "recovered_all_terminal")
            preflight.assert_not_called()
            start_server.assert_not_called()

    def test_mid_suite_resume_removes_owned_vllm_before_preflight_and_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_runnable_plan(
                root,
                backend="vllm",
                case_ids=("decode-complete", "decode-pending"),
            )
            plan = json.loads((run_dir / "plan.json").read_text())
            completed_case, pending_case = plan["suite"]["cases"]
            run_identity = f"{plan['fingerprint']}-{run_dir.name}"
            journal = Journal(run_dir / "events.jsonl")
            journal.append({"event": "run_start", "completed_cases_at_resume": []})
            journal.append({"event": "server_ready", "backend": "vllm"})
            journal.append(
                {
                    "event": "case_complete",
                    "case_id": completed_case["case_id"],
                    "attempt_id": "completed-before-crash",
                    "kind": "decode",
                    "elapsed_s": 1.0,
                }
            )
            order: list[str] = []

            def runtime_command(command: list[str], **kwargs: object):
                if command[:3] == ["docker", "ps", "-a"]:
                    return _completed(stdout=f"container-id true {run_identity}\n")
                if command[:2] == ["docker", "inspect"]:
                    return _completed(stdout=f"true {run_identity}\n")
                if command[:2] == ["docker", "stop"]:
                    order.append("old_server_stopped")
                elif command[:2] == ["docker", "rm"]:
                    order.append("old_server_removed")
                return _completed()

            def preflight(model: object) -> None:
                self.assertIn("old_server_removed", order)
                order.append("preflight")

            fresh_server = SimpleNamespace(
                backend="vllm",
                base_url="http://127.0.0.1:8000/v1",
                startup_s=0.1,
                container_id=None,
                ollama_model=None,
                unload_ollama=False,
                stop=Mock(),
            )

            def start_server(*args: object, **kwargs: object) -> object:
                self.assertIn("preflight", order)
                order.append("new_server_started")
                return fresh_server

            def complete_case(**kwargs: object) -> None:
                case = kwargs["case"]
                case_journal = kwargs["journal"]
                self.assertEqual(case.case_id, pending_case["case_id"])
                case_journal.append(
                    {
                        "event": "case_complete",
                        "case_id": case.case_id,
                        "attempt_id": "completed-after-resume",
                        "kind": case.kind,
                        "elapsed_s": 1.0,
                    }
                )

            telemetry = Mock()
            first_request = Mock()
            first_request.to_dict.return_value = {}
            with (
                patch("bench.runtime._run", side_effect=runtime_command),
                patch("bench.runner._preflight", side_effect=preflight),
                patch("bench.runner.start_server", side_effect=start_server),
                patch("bench.runner.TelemetrySampler", return_value=telemetry),
                patch("bench.runner._prime_model", return_value=first_request),
                patch("bench.runner._execute_case", side_effect=complete_case),
                patch("bench.runner.snapshot_vllm_spec_decode_metrics") as scrape,
            ):
                summary = execute_plan(run_dir, workspace=root)

            self.assertLess(order.index("old_server_removed"), order.index("preflight"))
            self.assertLess(order.index("preflight"), order.index("new_server_started"))
            self.assertEqual(summary["status"], "complete")
            fresh_server.stop.assert_called_once_with(keep_server=False)
            scrape.assert_not_called()
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            names = [event["event"] for event in events]
            measurement_index = names.index("measurement_complete")
            stopped_index = max(
                index
                for index, name in enumerate(names)
                if name == "server_stopped"
            )
            self.assertLess(
                measurement_index,
                stopped_index,
            )
            self.assertLess(
                stopped_index, names.index("run_complete")
            )
            measurement = events[measurement_index]
            stopped = events[stopped_index]
            self.assertGreaterEqual(measurement["elapsed_s"], 0.0)
            self.assertGreaterEqual(stopped["cleanup_elapsed_s"], 0.0)

    def test_resume_stops_exact_owned_vllm_before_marking_run_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_runnable_plan(root, backend="vllm")
            plan = json.loads((run_dir / "plan.json").read_text())
            case_id = plan["suite"]["cases"][0]["case_id"]
            run_identity = f"{plan['fingerprint']}-{run_dir.name}"
            journal = Journal(run_dir / "events.jsonl")
            journal.append({"event": "run_start", "completed_cases_at_resume": []})
            journal.append({"event": "server_ready", "backend": "vllm"})
            journal.append(
                {
                    "event": "case_complete",
                    "case_id": case_id,
                    "attempt_id": "done-before-crash",
                    "kind": "decode",
                    "elapsed_s": 1.0,
                }
            )
            provenance = run_dir / "server" / "provenance.json"
            provenance.parent.mkdir(parents=True)
            provenance.write_text(
                json.dumps(
                    {
                        "backend": "vllm",
                        "base_url": "http://127.0.0.1:8000/v1",
                        "container_id": "container-id",
                    }
                )
            )
            commands: list[list[str]] = []

            def runtime_command(command: list[str], **kwargs: object):
                commands.append(command)
                if command[:3] == ["docker", "ps", "-a"]:
                    return _completed(stdout=f"container-id true {run_identity}\n")
                if command[:2] == ["docker", "inspect"]:
                    if "--format" in command:
                        return _completed(stdout=f"true {run_identity}\n")
                    return _completed(
                        stdout=json.dumps(
                            [
                                {
                                    "Id": "container-id",
                                    "Image": "sha256:image",
                                    "Config": {"Entrypoint": [], "Cmd": []},
                                }
                            ]
                        )
                    )
                if command[:2] in (["docker", "stop"], ["docker", "rm"]):
                    events = [
                        json.loads(line)
                        for line in (run_dir / "events.jsonl").read_text().splitlines()
                    ]
                    self.assertFalse(
                        any(event.get("event") == "run_complete" for event in events),
                        "run_complete was persisted before owned-server cleanup",
                    )
                return _completed()

            with (
                patch("bench.runner._preflight") as preflight,
                patch("bench.runner.start_server") as start_server,
                patch("bench.runtime._run", side_effect=runtime_command),
            ):
                summary = execute_plan(run_dir, workspace=root)

            command_prefixes = [command[:2] for command in commands]
            self.assertIn(["docker", "stop"], command_prefixes)
            self.assertIn(["docker", "rm"], command_prefixes)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["run_completion_status"], "recovered_all_terminal")
            preflight.assert_not_called()
            start_server.assert_not_called()

    def test_resume_does_not_complete_when_owned_vllm_cleanup_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_runnable_plan(root, backend="vllm")
            plan = json.loads((run_dir / "plan.json").read_text())
            case_id = plan["suite"]["cases"][0]["case_id"]
            run_identity = f"{plan['fingerprint']}-{run_dir.name}"
            journal = Journal(run_dir / "events.jsonl")
            journal.append({"event": "run_start", "completed_cases_at_resume": []})
            journal.append({"event": "server_ready", "backend": "vllm"})
            journal.append(
                {
                    "event": "case_complete",
                    "case_id": case_id,
                    "attempt_id": "done-before-crash",
                    "kind": "decode",
                    "elapsed_s": 1.0,
                }
            )

            def runtime_command(command: list[str], **kwargs: object):
                if command[:3] == ["docker", "ps", "-a"]:
                    return _completed(stdout=f"container-id true {run_identity}\n")
                if command[:2] == ["docker", "inspect"]:
                    return _completed(stdout=f"true {run_identity}\n")
                if command[:2] == ["docker", "stop"]:
                    raise RuntimeErrorWithContext("owned container cleanup failed")
                return _completed()

            with (
                patch("bench.runner._preflight"),
                patch("bench.runner.start_server"),
                patch("bench.runtime._run", side_effect=runtime_command),
            ):
                try:
                    execute_plan(run_dir, workspace=root)
                except RuntimeErrorWithContext:
                    pass

            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            self.assertFalse(
                any(event.get("event") == "run_complete" for event in events),
                "failed cleanup must leave the run resumable rather than certified complete",
            )
            aborted = [
                event for event in events if event.get("event") == "run_aborted"
            ]
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(len(aborted), 1)
            self.assertEqual(aborted[0]["stage"], "lifecycle_recovery")
            self.assertEqual(summary["status"], "aborted")

    def test_resume_does_not_silently_complete_ollama_with_unknown_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_runnable_plan(root, backend="ollama")
            plan = json.loads((run_dir / "plan.json").read_text())
            case_id = plan["suite"]["cases"][0]["case_id"]
            journal = Journal(run_dir / "events.jsonl")
            journal.append({"event": "run_start", "completed_cases_at_resume": []})
            journal.append({"event": "server_ready", "backend": "ollama"})
            journal.append(
                {
                    "event": "case_complete",
                    "case_id": case_id,
                    "attempt_id": "done-before-crash",
                    "kind": "decode",
                    "elapsed_s": 1.0,
                }
            )

            with (
                patch("bench.runner._preflight") as preflight,
                patch("bench.runner.start_server") as start_server,
                patch("bench.runtime._ollama_model_loaded", return_value=True),
            ):
                try:
                    summary = execute_plan(run_dir, workspace=root)
                except RuntimeError:
                    summary = None

            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            self.assertFalse(
                any(event.get("event") == "run_complete" for event in events),
                "unknown Ollama lifecycle ownership must not be silently finalized",
            )
            if summary is not None:
                self.assertNotEqual(summary["status"], "complete")
            preflight.assert_not_called()
            start_server.assert_not_called()

    def test_text_context_estimate_keeps_headroom_for_chat_template(self) -> None:
        case = SimpleNamespace(
            id="prefill-near-limit",
            kind="prefill",
            requires=["chat"],
            prompt_repetitions=32691,
            max_output_tokens=1,
        )

        estimated, basis = _estimated_context_tokens(case)

        self.assertGreater(estimated, 32768)
        self.assertEqual(basis, "prompt_words_plus_request_margin")

    def test_vision_context_estimate_uses_the_generated_image_size_clamp(self) -> None:
        def estimate(image_size: int) -> tuple[int, str]:
            case = SimpleNamespace(
                id="vision",
                kind="capability",
                requires=["vision"],
                prompt_repetitions=image_size,
                max_output_tokens=16,
            )
            return _estimated_context_tokens(case)

        self.assertEqual(estimate(1), estimate(16))
        self.assertEqual(estimate(10_000), estimate(2048))

    def test_tool_context_estimate_includes_serialized_schema_overhead(self) -> None:
        case = SimpleNamespace(
            id="tool-call",
            kind="capability",
            requires=["tools"],
            prompt_repetitions=0,
            max_output_tokens=64,
        )

        estimated, basis = _estimated_context_tokens(case)
        prompt_only_estimate = (
            len(_prompt(case, "context-estimate").split())
            + case.max_output_tokens
            + 128
        )

        self.assertGreater(estimated, prompt_only_estimate)
        self.assertIn("tool_schema", basis)

    def test_legacy_plan_rejects_tampered_case_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._write_runnable_plan(root)
            plan_path = run_dir / "plan.json"
            plan = json.loads(plan_path.read_text())
            plan["suite"]["cases"][0]["case_id"] = "decode--tampered"
            plan_path.write_text(json.dumps(plan))

            with self.assertRaisesRegex(RuntimeError, "case identity"):
                execute_plan(run_dir, workspace=root)

    def test_journal_append_separates_a_torn_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text('{"event":"partial"')
            journal = Journal(path)
            journal.append(
                {"event": "case_complete", "case_id": "case-1", "attempt_id": "one"}
            )

            self.assertEqual(journal.completed_cases(), {"case-1"})
if __name__ == "__main__":
    unittest.main()
