"""Regression contracts for the dedicated SM121 storage canary executor."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.host_safety import HostSafetyError
from bench.journal import Journal
from bench.manifest import load_models, load_suite
from bench.runner import (
    PreflightError,
    SM121StorageQualityGateError,
    create_plan,
    create_sm121_storage_canary_plan,
    execute_plan,
    execute_sm121_storage_canary,
)
from bench.sglang_sm121_storage import (
    SM121_STORAGE_BUILD_CONTRACT_SHA256,
    SM121_STORAGE_CACHE_PAGES,
    SM121_STORAGE_CANDIDATE_ID,
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_MAX_BATCH_PAGES,
    SM121_STORAGE_MODE,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_QUEUE_DEPTH,
    SM121_STORAGE_RUNTIME_PROVENANCE_EVENT,
    SM121_STORAGE_RUNTIME_PROVENANCE_FIELDS,
    SM121_STORAGE_SOURCE_TREE,
)
from bench.seccomp_profile_contract import DERIVED_SHA256


def _native_provenance() -> dict[str, object]:
    return {
        "candidate_id": SM121_STORAGE_CANDIDATE_ID,
        "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
        "build_contract_sha256": SM121_STORAGE_BUILD_CONTRACT_SHA256,
        "docker_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
        "sglang_storage_mode": SM121_STORAGE_MODE,
        "sglang_ple_nvme_backend": "io_uring",
        "sglang_ple_nvme_queue_depth": SM121_STORAGE_QUEUE_DEPTH,
        "sglang_ple_nvme_max_batch_pages": SM121_STORAGE_MAX_BATCH_PAGES,
        "sglang_ple_nvme_cache_pages": SM121_STORAGE_CACHE_PAGES,
        "sglang_rust_build_mode": "never",
        "seccomp_profile_sha256": "sha256:" + DERIVED_SHA256,
        "container_rootfs": "readonly_tmpfs_writable_cache",
        "container_capabilities": "dropped_all",
        "container_no_new_privileges": True,
        "hf_network_policy": "offline",
        "network_topology": "loopback_published_bridge",
        "benchmark_scope": "sm121_storage_pre_admission_canary",
        "model_acquisition": "disabled_exact_read_only_snapshot",
        "api_authentication": "ephemeral_bearer",
        "api_key_file_mode": "0600",
    }


class _TrippedWatchdog:
    def __init__(self) -> None:
        self.failure: HostSafetyError | None = None
        self.abort_callback_error: BaseException | None = None
        self._abort_callback: object | None = None
        self.start_calls = 0
        self.stop_calls = 0

    @property
    def tripped(self) -> bool:
        return self.failure is not None

    def start(self) -> _TrippedWatchdog:
        self.start_calls += 1
        return self

    def stop(self) -> None:
        self.stop_calls += 1

    def register_abort_callback(self, callback: object) -> None:
        self._abort_callback = callback

    def trip_after_registration(self) -> None:
        assert callable(self._abort_callback)
        self.failure = HostSafetyError("synthetic", "synthetic watchdog trip")
        try:
            self._abort_callback()
        except BaseException as error:
            self.abort_callback_error = error

    def raise_if_tripped(self) -> None:
        if self.failure is not None:
            raise self.failure


class SGLangSm121StorageCanaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Path(__file__).resolve().parents[1]
        self.models = load_models(self.workspace / "manifests" / "models.toml")
        self.model = self.models[
            "qwen38-flash-next-nvfp4-sm121-triton-storage-target-only-sglang"
        ]
        self.suite = load_suite(
            self.workspace
            / "manifests"
            / "suites"
            / "qwen38_flash_next_sm121_triton_storage_canary.toml"
        )

    def _freeze(self, results: Path) -> Path:
        with (
            patch("bench.runner._image_digest", return_value=None),
            patch(
                "bench.runner._sm121_storage_image_identity",
                return_value={
                    "docker_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
                    "platform": SM121_STORAGE_PLATFORM,
                    "source_tree": SM121_STORAGE_SOURCE_TREE,
                },
            ),
            patch("bench.runner._host_snapshot", return_value={"host": "test"}),
        ):
            return create_sm121_storage_canary_plan(
                model=self.model,
                suite=self.suite,
                results_root=results,
                models_path=self.workspace / "manifests" / "models.toml",
                suite_path=(
                    self.workspace
                    / "manifests"
                    / "suites"
                    / "qwen38_flash_next_sm121_triton_storage_canary.toml"
                ),
            )

    def test_ordinary_plan_and_execute_paths_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "dedicated"):
                create_plan(
                    model=self.model,
                    suite=self.suite,
                    results_root=results,
                    models_path=self.workspace / "manifests" / "models.toml",
                    suite_path=(
                        self.workspace
                        / "manifests"
                        / "suites"
                        / "qwen38_flash_next_sm121_triton_storage_canary.toml"
                    ),
                )
            run_dir = self._freeze(results)
            with self.assertRaisesRegex(PreflightError, "dedicated"):
                execute_plan(run_dir, workspace=self.workspace)

    def test_canary_uses_two_fresh_servers_without_a_prime_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            run_dir = self._freeze(results)
            servers = [
                SimpleNamespace(
                    backend="sglang",
                    startup_s=1.0,
                    container_id="one",
                    native_provenance=_native_provenance(),
                    stop=Mock(),
                ),
                SimpleNamespace(
                    backend="sglang",
                    startup_s=2.0,
                    container_id="two",
                    native_provenance=_native_provenance(),
                    stop=Mock(),
                ),
            ]
            telemetry = Mock()

            def execute_case(*, case: SimpleNamespace, journal: object, **_: object) -> None:
                getattr(journal, "append")(
                    {
                        "event": "case_complete",
                        "case_id": case.case_id,
                        "kind": case.kind,
                        "concurrency": case.concurrency,
                        "elapsed_s": 0.1,
                        "validation_passed": True,
                    }
                )

            with (
                patch("bench.runner._preflight") as preflight,
                patch("bench.runner._host_safety_watchdog", return_value=None),
                patch("bench.runner.TelemetrySampler", return_value=telemetry),
                patch("bench.runner.start_server", side_effect=servers) as start,
                patch(
                    "bench.runner.capture_server_provenance",
                    return_value={"candidate_id": "synthetic"},
                ),
                patch("bench.runner.save_server_logs"),
                patch("bench.runner._execute_case", side_effect=execute_case),
                patch("bench.runner._prime_model") as prime,
                patch("bench.runner.summarize_run", return_value={"status": "complete"}),
            ):
                summary = execute_sm121_storage_canary(
                    run_dir, workspace=self.workspace
                )

            self.assertEqual(summary["status"], "complete")
            self.assertEqual(preflight.call_count, 2)
            self.assertEqual(start.call_count, 2)
            self.assertEqual(servers[0].stop.call_count, 1)
            self.assertEqual(servers[1].stop.call_count, 1)
            prime.assert_not_called()
            events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
            self.assertEqual(events.count('"event":"server_ready"'), 2)
            self.assertEqual(events.count('"first_inference_is_case":true'), 2)
            journal_events = Journal(run_dir / "events.jsonl").events()
            provenance_events = [
                event
                for event in journal_events
                if event["event"] == SM121_STORAGE_RUNTIME_PROVENANCE_EVENT
            ]
            self.assertEqual(len(provenance_events), 2)
            expected_event_fields = {
                "timestamp",
                "event",
                "fresh_server_lifetime",
                *SM121_STORAGE_RUNTIME_PROVENANCE_FIELDS,
            }
            for lifetime, event in enumerate(provenance_events, start=1):
                self.assertEqual(set(event), expected_event_fields)
                self.assertEqual(event["fresh_server_lifetime"], lifetime)
                self.assertEqual(
                    event,
                    {
                        "timestamp": event["timestamp"],
                        "event": SM121_STORAGE_RUNTIME_PROVENANCE_EVENT,
                        "fresh_server_lifetime": lifetime,
                        **_native_provenance(),
                    },
                )

    def test_quality_failure_does_not_start_the_long_context_lifetime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            run_dir = self._freeze(results)
            server = SimpleNamespace(
                backend="sglang",
                startup_s=1.0,
                container_id="one",
                native_provenance=_native_provenance(),
                stop=Mock(),
            )

            def execute_case(*, case: SimpleNamespace, journal: Journal, **_: object) -> None:
                journal.append(
                    {
                        "event": "case_complete",
                        "case_id": case.case_id,
                        "kind": case.kind,
                        "concurrency": case.concurrency,
                        "elapsed_s": 0.1,
                        "validation_passed": False,
                    }
                )

            with (
                patch("bench.runner._preflight"),
                patch("bench.runner._host_safety_watchdog", return_value=None),
                patch("bench.runner.TelemetrySampler", return_value=Mock()),
                patch("bench.runner.start_server", return_value=server) as start,
                patch("bench.runner.capture_server_provenance", return_value={}),
                patch("bench.runner.save_server_logs"),
                patch("bench.runner._execute_case", side_effect=execute_case),
                patch("bench.runner.summarize_run", return_value={"status": "partial"}),
            ):
                with self.assertRaises(SM121StorageQualityGateError):
                    execute_sm121_storage_canary(run_dir, workspace=self.workspace)

            self.assertEqual(start.call_count, 1)
            self.assertEqual(server.stop.call_count, 1)
            events = Journal(run_dir / "events.jsonl").events()
            ready_case_ids = [
                event["case_id"] for event in events if event["event"] == "server_ready"
            ]
            self.assertEqual(len(ready_case_ids), 1)
            self.assertTrue(ready_case_ids[0].startswith("synthetic-exact-answer-v2--"))
            self.assertTrue(any(event["event"] == "run_aborted" for event in events))

    def test_watchdog_callback_is_not_interrupted_twice_during_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            run_dir = self._freeze(results)
            watchdog = _TrippedWatchdog()
            server = SimpleNamespace(
                backend="sglang",
                startup_s=1.0,
                container_id="one",
                native_provenance=_native_provenance(),
                interrupt_owned=Mock(),
                stop=Mock(),
            )

            def start_with_watchdog(*_: object, **kwargs: object) -> SimpleNamespace:
                on_server_created = kwargs["on_server_created"]
                assert callable(on_server_created)
                on_server_created(server)
                watchdog.trip_after_registration()
                return server

            with (
                patch("bench.runner._preflight"),
                patch("bench.runner._host_safety_watchdog", return_value=watchdog),
                patch("bench.runner.TelemetrySampler", return_value=Mock()),
                patch("bench.runner.start_server", side_effect=start_with_watchdog) as start,
                patch("bench.runner.save_server_logs"),
                patch("bench.runner.summarize_run", return_value={"status": "aborted"}),
            ):
                with self.assertRaises(HostSafetyError):
                    execute_sm121_storage_canary(run_dir, workspace=self.workspace)

            self.assertEqual(start.call_count, 1)
            self.assertEqual(server.interrupt_owned.call_count, 1)
            self.assertEqual(server.stop.call_count, 1)
            self.assertGreaterEqual(watchdog.stop_calls, 1)

    def test_canary_cannot_resume_a_partial_lifetime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            run_dir = self._freeze(results)
            (run_dir / "events.jsonl").write_text(
                '{"event":"run_start"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(PreflightError, "non-resumable"):
                execute_sm121_storage_canary(run_dir, workspace=self.workspace)


if __name__ == "__main__":
    unittest.main()
