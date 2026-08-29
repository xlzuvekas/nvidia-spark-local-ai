"""Offline tests for the isolated prospective-SM121 8K admission gate."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.execution_admission import model_execution_blocker
from bench.manifest import load_models, load_suite
from bench.sglang_sm121_cache_observability import (
    SM121_CACHE_OBSERVABILITY_CACHED_SERIES,
    SM121_CACHE_SOURCE_DIGESTS,
)
from bench.sglang_sm121_cache_semantic import SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS
from bench.sglang_sm121_chunked_prefill_admission import (
    SM121_CHUNKED_PREFILL_8K_ADMISSION_RUNTIME_EXPECTED,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_TIMED_CASE_ID,
    SM121ChunkedPrefill8KAdmissionError,
    derive_sm121_chunked_prefill_8k_admission_t0,
    validate_sm121_chunked_prefill_8k_admission_profile,
    validate_sm121_chunked_prefill_8k_admission_suite,
    validate_sm121_chunked_prefill_8k_admission_t0_event,
)
from bench.sglang_sm121_chunked_prefill_performance import (
    SM121_CHUNKED_PREFILL_PERFORMANCE_METRIC_FIELDS,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CANDIDATE_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CONTROL_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY,
    validate_sm121_chunked_prefill_performance_pair,
)
from bench.sglang_sm121_storage import (
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_SOURCE_TREE,
)
from bench.sm121_chunked_prefill_admission_runner import (
    _t0_event,
    audit_sm121_chunked_prefill_8k_admission,
    create_sm121_chunked_prefill_8k_admission_plan,
    execute_sm121_chunked_prefill_8k_admission,
)
from bench.sm121_chunked_prefill_runner import (
    create_sm121_chunked_prefill_performance_campaign,
)
from sparkbench import (
    DEFAULT_SM121_CHUNKED_PREFILL_8K_ADMISSION_OUTPUT_ROOT,
    DEFAULT_SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE,
    build_parser,
    command_audit_sm121_chunked_prefill_8k_preflight,
    command_sm121_chunked_prefill_8k_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "manifests" / "models.toml"
SUITE_PATH = (
    ROOT
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sm121_triton_storage_chunked_prefill_8k_preflight.toml"
)
V3_SUITE_PATH = (
    ROOT
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sm121_triton_storage_chunked_prefill_performance_v3.toml"
)


def _metrics(*, input_tokens: int) -> dict[str, object]:
    values: dict[str, object] = {
        "available": True,
        "guardrail_metrics_available": True,
    }
    for metric in SM121_CHUNKED_PREFILL_PERFORMANCE_METRIC_FIELDS:
        values[metric] = input_tokens if metric == "prefill_input_tokens" else 0
    for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
        values[f"cached_{source}_series_present"] = False
    return values


def _result() -> dict[str, object]:
    return {
        "private_prompt_token_ids": (1,),
        "prompt_tokens": 58_000,
        "completion_tokens": 1,
        "reasoning_tokens": 0,
        "response_detail_state": "omitted",
        "response_device_cached_tokens": None,
        "response_host_cached_tokens": None,
        "response_storage_cached_tokens": None,
        "usage_detail_state": "null",
        "usage_cached_tokens": None,
    }


class SM121ChunkedPrefill8KAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        models = load_models(MODELS)
        cls.model = models[SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CANDIDATE_PROFILE_ID]
        cls.control = models[SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CONTROL_PROFILE_ID]
        cls.suite = load_suite(SUITE_PATH)
        cls.v3_suite = load_suite(V3_SUITE_PATH)

    def _freeze(self, output_root: Path) -> Path:
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
            patch("bench.runner._host_snapshot", return_value={"host": "fixture"}),
        ):
            return create_sm121_chunked_prefill_8k_admission_plan(
                model=self.model,
                suite=self.suite,
                output_root=output_root,
                models_path=MODELS,
                suite_path=SUITE_PATH,
            )

    @staticmethod
    def _runtime() -> dict[str, object]:
        return {
            "mamba_radix_cache_strategy": "extra_buffer_lazy",
            "max_mamba_cache_size": 4,
            "chunked_prefill_size": 8192,
            **SM121_CHUNKED_PREFILL_8K_ADMISSION_RUNTIME_EXPECTED,
        }

    def _run_with_mocks(
        self,
        run_dir: Path,
        *,
        quality_passed: bool = True,
        cold_result: dict[str, object] | None = None,
        preflight_side_effect: object = None,
    ) -> tuple[dict[str, object], int, int]:
        servers = [
            SimpleNamespace(backend="sglang", stop=Mock(), interrupt_owned=Mock()),
            SimpleNamespace(backend="sglang", stop=Mock(), interrupt_owned=Mock()),
        ]
        with (
            patch(
                "bench.sm121_chunked_prefill_admission_runner.base_runner._preflight",
                side_effect=preflight_side_effect,
            ) as preflight,
            patch("bench.sm121_chunked_prefill_admission_runner.TelemetrySampler"),
            patch(
                "bench.sm121_chunked_prefill_admission_runner.base_runner._host_safety_watchdog",
                return_value=None,
            ),
            patch(
                "bench.sm121_chunked_prefill_admission_runner.base_runner.start_server",
                side_effect=servers,
            ) as start_server,
            patch("bench.sm121_chunked_prefill_admission_runner.save_server_logs"),
            patch(
                "bench.sm121_chunked_prefill_admission_runner.inspect_sm121_cache_source_digests",
                return_value=dict(SM121_CACHE_SOURCE_DIGESTS),
            ),
            patch(
                "bench.sm121_chunked_prefill_admission_runner.inspect_sm121_chunked_prefill_runtime_identity",
                return_value=self._runtime(),
            ),
            patch(
                "bench.sm121_chunked_prefill_admission_runner.base_runner._quality_request_arguments",
                return_value={},
            ),
            patch(
                "bench.sm121_chunked_prefill_admission_runner.base_runner.stream_chat_request",
                return_value={},
            ),
            patch(
                "bench.sm121_chunked_prefill_admission_runner.base_runner._validate_quality_item",
                return_value={"passed": quality_passed},
            ),
            patch(
                "bench.sm121_chunked_prefill_admission_runner.settle_sm121_cache_observability_metrics",
                side_effect=[
                    (_metrics(input_tokens=0), 0.0, 2, True),
                    (_metrics(input_tokens=1), 0.0, 2, True),
                ],
            ),
            patch(
                "bench.sm121_chunked_prefill_admission_runner.request_sm121_cache_semantic_turn",
                return_value=cold_result or _result(),
            ),
            patch(
                "bench.sm121_chunked_prefill_runner._messages",
                return_value=([{"role": "user", "content": "synthetic"}],),
            ),
            patch(
                "bench.sm121_chunked_prefill_runner._EXPECTED_RESPONSES",
                ("SYNTHETIC",),
            ),
        ):
            summary = execute_sm121_chunked_prefill_8k_admission(
                run_dir, workspace=ROOT
            )
        return summary, start_server.call_count, preflight.call_count

    def test_exact_prospective_pair_and_admission_suite_validate(self) -> None:
        validate_sm121_chunked_prefill_performance_pair(self.control, self.model)
        validate_sm121_chunked_prefill_8k_admission_profile(self.model)
        validate_sm121_chunked_prefill_8k_admission_suite(self.suite)
        self.assertEqual(4096, SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY.control_chunk_size)
        self.assertEqual(8192, SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY.candidate_chunk_size)

    def test_profile_and_suite_drift_are_rejected(self) -> None:
        arguments = list(self.model.args)
        arguments[arguments.index("--chunked-prefill-size") + 1] = "4096"
        with self.assertRaises(SM121ChunkedPrefill8KAdmissionError):
            validate_sm121_chunked_prefill_8k_admission_profile(
                replace(self.model, args=tuple(arguments))
            )
        drifted_suite = replace(
            self.suite,
            cases=(
                self.suite.cases[0],
                replace(self.suite.cases[1], max_output_tokens=64),
            ),
        )
        with self.assertRaises(SM121ChunkedPrefill8KAdmissionError):
            validate_sm121_chunked_prefill_8k_admission_suite(drifted_suite)

    def test_t0_event_is_scalar_only_and_rejects_a_cache_hit(self) -> None:
        case = SimpleNamespace(
            case_id=SM121_CHUNKED_PREFILL_8K_ADMISSION_TIMED_CASE_ID + "--0123456789ab"
        )
        event = _t0_event(
            case=case,
            result=_result(),
            before=_metrics(input_tokens=0),
            before_polls=2,
            before_settled=True,
            after=_metrics(input_tokens=1),
            after_polls=2,
            after_settled=True,
        )
        validate_sm121_chunked_prefill_8k_admission_t0_event(event)
        self.assertNotIn("request_wall_s", event)
        self.assertEqual((True, "admitted"), derive_sm121_chunked_prefill_8k_admission_t0(event))
        event["after_cached_device_tokens"] = 1
        event["delta_cached_device_tokens"] = 1
        event["cold_t0_admitted"] = False
        event["cold_t0_basis"] = "cold_hit"
        self.assertEqual((False, "cold_hit"), derive_sm121_chunked_prefill_8k_admission_t0(event))
        validate_sm121_chunked_prefill_8k_admission_t0_event(event)

    def test_t0_rejects_unexpected_cache_detail_states(self) -> None:
        case = SimpleNamespace(
            case_id=SM121_CHUNKED_PREFILL_8K_ADMISSION_TIMED_CASE_ID + "--0123456789ab"
        )
        for field in ("response_detail_state", "usage_detail_state"):
            with self.subTest(field=field):
                result = _result()
                result[field] = "unexpected"
                event = _t0_event(
                    case=case,
                    result=result,
                    before=_metrics(input_tokens=0),
                    before_polls=2,
                    before_settled=True,
                    after=_metrics(input_tokens=1),
                    after_polls=2,
                    after_settled=True,
                )
                self.assertEqual(
                    (False, field), derive_sm121_chunked_prefill_8k_admission_t0(event)
                )
                validate_sm121_chunked_prefill_8k_admission_t0_event(event)

    def test_freeze_and_successful_execution_are_non_evidence_and_auditable(self) -> None:
        (ROOT / "logs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "logs") as directory:
            run_dir = self._freeze(Path(directory))
            summary, starts, preflights = self._run_with_mocks(run_dir)
            self.assertEqual("complete", summary["status"])
            self.assertEqual("admitted", summary["decision"])
            self.assertEqual(2, summary["static_attestations"])
            self.assertEqual(2, summary["runtime_attestations"])
            self.assertEqual(2, starts)
            self.assertEqual(2, preflights)
            self.assertTrue((run_dir / "admission.json").is_file())
            self.assertFalse((run_dir / "summary.json").exists())
            before = {
                path.name: path.read_bytes()
                for path in run_dir.iterdir()
                if path.is_file()
            }
            report = audit_sm121_chunked_prefill_8k_admission(run_dir)
            self.assertTrue(report["read_only"])
            self.assertTrue(report["ok"])
            self.assertEqual(
                before,
                {
                    path.name: path.read_bytes()
                    for path in run_dir.iterdir()
                    if path.is_file()
                },
            )

    def test_quality_failure_blocks_cold_t0_and_terminal_audit(self) -> None:
        (ROOT / "logs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "logs") as directory:
            run_dir = self._freeze(Path(directory))
            summary, starts, preflights = self._run_with_mocks(
                run_dir, quality_passed=False
            )
            self.assertEqual("partial", summary["status"])
            self.assertEqual("blocked", summary["decision"])
            self.assertFalse(summary["cold_t0_admitted"])
            self.assertEqual(1, starts)
            self.assertEqual(1, preflights)
            self.assertFalse(audit_sm121_chunked_prefill_8k_admission(run_dir)["ok"])

    def test_unexpected_cache_detail_state_blocks_cold_t0_execution(self) -> None:
        (ROOT / "logs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "logs") as directory:
            for field in ("response_detail_state", "usage_detail_state"):
                with self.subTest(field=field):
                    result = _result()
                    result[field] = "unexpected"
                    run_dir = self._freeze(Path(directory) / field)
                    summary, starts, preflights = self._run_with_mocks(
                        run_dir, cold_result=result
                    )
                    self.assertEqual("partial", summary["status"])
                    self.assertEqual("blocked", summary["decision"])
                    self.assertFalse(summary["cold_t0_admitted"])
                    self.assertEqual(2, starts)
                    self.assertEqual(2, preflights)

    def test_second_lifetime_rechecks_preflight(self) -> None:
        (ROOT / "logs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "logs") as directory:
            run_dir = self._freeze(Path(directory))
            summary, starts, preflights = self._run_with_mocks(
                run_dir,
                preflight_side_effect=[None, RuntimeError("synthetic preflight")],
            )
            self.assertEqual("partial", summary["status"])
            self.assertEqual("blocked", summary["decision"])
            self.assertTrue(summary["quality_admitted"])
            self.assertFalse(summary["cold_t0_admitted"])
            self.assertEqual(1, starts)
            self.assertEqual(2, preflights)

    def test_profile_stays_blocked_from_generic_execution(self) -> None:
        self.assertIn(
            "chunked-prefill performance profile",
            model_execution_blocker(self.model) or "",
        )
        self.assertIsNone(
            model_execution_blocker(
                self.model, allow_sm121_chunked_prefill_performance=True
            )
        )

    def test_v3_performance_campaign_remains_blocked_without_a_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "verified 8K admission receipt"):
                create_sm121_chunked_prefill_performance_campaign(
                    control_model=self.control,
                    candidate_model=self.model,
                    suite=self.v3_suite,
                    results_root=output_root,
                    models_path=MODELS,
                    suite_path=V3_SUITE_PATH,
                )
            self.assertEqual([], list(output_root.iterdir()))

    def test_preflight_refuses_to_pollute_tracked_trees(self) -> None:
        for output_root in (ROOT, ROOT / "results", ROOT / "evidence"):
            with self.subTest(output_root=output_root):
                with self.assertRaisesRegex(RuntimeError, "ignored logs root"):
                    create_sm121_chunked_prefill_8k_admission_plan(
                        model=self.model,
                        suite=self.suite,
                        output_root=output_root,
                        models_path=MODELS,
                        suite_path=SUITE_PATH,
                    )

    def test_preflight_rejects_a_logs_symlink_escape(self) -> None:
        (ROOT / "logs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "logs") as directory:
            escape = Path(directory) / "escape"
            escape.symlink_to(ROOT / "evidence", target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "logs topology"):
                create_sm121_chunked_prefill_8k_admission_plan(
                    model=self.model,
                    suite=self.suite,
                    output_root=escape,
                    models_path=MODELS,
                    suite_path=SUITE_PATH,
                )

    def test_preflight_requires_a_private_output_directory(self) -> None:
        (ROOT / "logs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "logs") as directory:
            private_target = Path(directory) / "private-run"
            run_dir = self._freeze(private_target)
            self.assertEqual(0o700, run_dir.stat().st_mode & 0o777)

            public_target = Path(directory) / "public-run"
            public_target.mkdir(mode=0o755)
            os.chmod(public_target, 0o755)
            with self.assertRaisesRegex(RuntimeError, "not private"):
                create_sm121_chunked_prefill_8k_admission_plan(
                    model=self.model,
                    suite=self.suite,
                    output_root=public_target,
                    models_path=MODELS,
                    suite_path=SUITE_PATH,
                )

    def test_execution_rejects_a_copied_plan_outside_logs_before_writes(self) -> None:
        (ROOT / "logs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "logs") as logs_directory:
            run_dir = self._freeze(Path(logs_directory))
            with tempfile.TemporaryDirectory() as external_directory:
                external_run = Path(external_directory) / "copied-plan"
                external_run.mkdir()
                for name in ("plan.json", "inventory.json"):
                    shutil.copy2(run_dir / name, external_run / name)
                with (
                    patch(
                        "bench.sm121_chunked_prefill_admission_runner.base_runner.start_server"
                    ) as start_server,
                    patch(
                        "bench.sm121_chunked_prefill_admission_runner.TelemetrySampler"
                    ) as telemetry,
                    self.assertRaisesRegex(RuntimeError, "ignored logs root"),
                ):
                    execute_sm121_chunked_prefill_8k_admission(
                        external_run, workspace=ROOT
                    )
                start_server.assert_not_called()
                telemetry.assert_not_called()

    def test_execution_rejects_a_copied_plan_under_public_logs_before_writes(self) -> None:
        (ROOT / "logs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "logs") as directory:
            source_run = self._freeze(Path(directory) / "source")
            public_root = Path(directory) / "public"
            public_root.mkdir(mode=0o755)
            os.chmod(public_root, 0o755)
            copied_run = public_root / "copied-plan"
            copied_run.mkdir()
            for name in ("plan.json", "inventory.json"):
                shutil.copy2(source_run / name, copied_run / name)
            with (
                patch(
                    "bench.sm121_chunked_prefill_admission_runner.base_runner.start_server"
                ) as start_server,
                patch(
                    "bench.sm121_chunked_prefill_admission_runner.TelemetrySampler"
                ) as telemetry,
                self.assertRaisesRegex(RuntimeError, "not private"),
            ):
                execute_sm121_chunked_prefill_8k_admission(copied_run, workspace=ROOT)
            start_server.assert_not_called()
            telemetry.assert_not_called()

    def test_execution_rejects_precreated_write_targets_before_serving(self) -> None:
        (ROOT / "logs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "logs") as directory:
            run_dir = self._freeze(Path(directory))
            (run_dir / "server").symlink_to(ROOT / "evidence", target_is_directory=True)
            with (
                patch(
                    "bench.sm121_chunked_prefill_admission_runner.base_runner.start_server"
                ) as start_server,
                patch(
                    "bench.sm121_chunked_prefill_admission_runner.TelemetrySampler"
                ) as telemetry,
                self.assertRaisesRegex(RuntimeError, "server topology is invalid"),
            ):
                execute_sm121_chunked_prefill_8k_admission(run_dir, workspace=ROOT)
            start_server.assert_not_called()
            telemetry.assert_not_called()

    def test_unsafe_incomplete_topology_cannot_trigger_recovery(self) -> None:
        (ROOT / "logs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "logs") as directory:
            run_dir = self._freeze(Path(directory))
            (run_dir / "events.jsonl").write_text('{"event":"synthetic"}\n')
            (run_dir / "server").symlink_to(ROOT / "evidence", target_is_directory=True)
            with (
                patch(
                    "bench.sm121_chunked_prefill_admission_runner.recover_owned_sglang"
                ) as recover,
                self.assertRaisesRegex(RuntimeError, "server topology is invalid"),
            ):
                execute_sm121_chunked_prefill_8k_admission(run_dir, workspace=ROOT)
            recover.assert_not_called()

    def test_incomplete_run_recovers_only_its_own_lifetimes(self) -> None:
        (ROOT / "logs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "logs") as directory:
            run_dir = self._freeze(Path(directory))
            (run_dir / "events.jsonl").write_text('{"event":"synthetic"}\n')
            with (
                patch(
                    "bench.sm121_chunked_prefill_admission_runner.recover_owned_sglang",
                    return_value="already_absent",
                ) as recover,
                patch(
                    "bench.sm121_chunked_prefill_admission_runner.base_runner.start_server"
                ) as start_server,
                self.assertRaisesRegex(RuntimeError, "non-resumable"),
            ):
                execute_sm121_chunked_prefill_8k_admission(run_dir, workspace=ROOT)
            self.assertEqual(2, recover.call_count)
            start_server.assert_not_called()

    def test_lock_contention_prevents_incomplete_run_recovery(self) -> None:
        (ROOT / "logs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "logs") as directory:
            run_dir = self._freeze(Path(directory))
            (run_dir / "events.jsonl").write_text('{"event":"synthetic"}\n')
            with (
                patch(
                    "bench.sm121_chunked_prefill_admission_runner.fcntl.flock",
                    side_effect=BlockingIOError,
                ),
                patch(
                    "bench.sm121_chunked_prefill_admission_runner.recover_owned_sglang"
                ) as recover,
                self.assertRaisesRegex(RuntimeError, "benchmark lock"),
            ):
                execute_sm121_chunked_prefill_8k_admission(run_dir, workspace=ROOT)
            recover.assert_not_called()

    def test_audit_rejects_reordered_or_field_smuggled_records_and_key_residue(self) -> None:
        (ROOT / "logs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "logs") as directory:
            for mutation in ("reordered", "smuggled", "lifetime_swap", "api_key"):
                with self.subTest(mutation=mutation):
                    run_dir = self._freeze(Path(directory) / mutation)
                    self._run_with_mocks(run_dir)
                    if mutation == "api_key":
                        key_path = run_dir / "server" / "lifetime-1" / "api-key"
                        key_path.parent.mkdir(parents=True)
                        key_path.write_text("synthetic")
                    else:
                        events_path = run_dir / "events.jsonl"
                        events = [
                            json.loads(line) for line in events_path.read_text().splitlines()
                        ]
                        if mutation == "reordered":
                            events[8], events[9] = events[9], events[8]
                        elif mutation == "lifetime_swap":
                            events[2], events[9] = events[9], events[2]
                        else:
                            events[5]["synthetic_extra"] = True
                        events_path.write_text(
                            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
                        )
                    self.assertFalse(audit_sm121_chunked_prefill_8k_admission(run_dir)["ok"])

    def test_cli_exposes_only_non_resumable_preflight_and_read_only_audit(self) -> None:
        parser = build_parser()
        preflight = parser.parse_args(["sm121-chunked-prefill-8k-preflight"])
        self.assertEqual(
            DEFAULT_SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE, preflight.suite
        )
        self.assertEqual(
            DEFAULT_SM121_CHUNKED_PREFILL_8K_ADMISSION_OUTPUT_ROOT,
            preflight.output_root,
        )
        self.assertIs(command_sm121_chunked_prefill_8k_preflight, preflight.function)
        self.assertFalse(hasattr(preflight, "allow_download"))
        audit = parser.parse_args(
            ["audit-sm121-chunked-prefill-8k-preflight", "synthetic-run"]
        )
        self.assertEqual(Path("synthetic-run"), audit.run_dir)
        self.assertIs(command_audit_sm121_chunked_prefill_8k_preflight, audit.function)


if __name__ == "__main__":
    unittest.main()
