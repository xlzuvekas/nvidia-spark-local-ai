"""Offline contract tests for the private SM121 C1 plan/audit scaffold."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench import sm121_agent_admission_runner as admission
from bench.manifest import load_models, load_suite
from bench.sglang_sm121_agent_admission import (
    SM121_AGENT_ADMISSION_PROFILE_ID,
    SM121_AGENT_ADMISSION_STATIC_PROBE_ID,
    SM121_AGENT_ADMISSION_STATIC_PROBE_SCHEMA_VERSION,
)
from bench.sglang_sm121_storage import (
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_SOURCE_TREE,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "manifests" / "models.toml"
SUITE_PATH = (
    ROOT
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sm121_triton_storage_agent_admission.toml"
)


def _complete_events() -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        {
            "event": "run_start",
            "execution_mode": admission.SM121_AGENT_ADMISSION_EXECUTION_MODE,
            "admission_id": admission.SM121_AGENT_ADMISSION_ID,
            "profile_id": admission.SM121_AGENT_ADMISSION_PROFILE_ID,
            "suite_id": admission.SM121_AGENT_ADMISSION_SUITE_ID,
        },
        {"event": "measurement_started"},
        {
            "event": "sm121_agent_parser_static_attestation",
            "schema_version": SM121_AGENT_ADMISSION_STATIC_PROBE_SCHEMA_VERSION,
            "probe_id": SM121_AGENT_ADMISSION_STATIC_PROBE_ID,
            "docker_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
            "source_tree": SM121_STORAGE_SOURCE_TREE,
            "reasoning_parser_qwen3": True,
            "tool_call_parser_qwen3_coder": True,
        },
    ]
    gates = {
        "quality": {
            "event": "sm121_agent_quality_gate",
            "quality_item_count": 4,
            "quality_admitted": True,
            "payload_contract_verified": True,
        },
        "tools": {
            "event": "sm121_agent_tool_gate",
            "variant": 0,
            "scenario_count": 4,
            "scenario_passes": {
                case_id: True for case_id in admission.SM121_AGENT_ADMISSION_TOOL_CASE_IDS
            },
            "tools_admitted": True,
            "payload_contract_verified": True,
        },
        "long_context": {
            "event": "sm121_agent_long_context_gate",
            "input_tokenization_verified": True,
            "context_fit": True,
            "zero_metric_cache_hits": True,
            "zero_response_cache_hits": True,
            "guardrails_clean": True,
            "long_context_admitted": True,
            "payload_contract_verified": True,
        },
    }
    first_cases = (
        admission.SM121_AGENT_ADMISSION_QUALITY_CASE_ID,
        admission.SM121_AGENT_ADMISSION_TOOL_CASE_IDS[0],
        admission.SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID,
    )
    for lifetime, (phase, first_case) in enumerate(
        zip(("quality", "tools", "long_context"), first_cases, strict=True),
        start=1,
    ):
        events.extend(
            (
                {
                    "event": "sm121_agent_static_attestation",
                    "fresh_lifetime": lifetime,
                    "phase": phase,
                    "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
                    **admission.SM121_CACHE_SOURCE_DIGESTS,
                    **admission._SCALAR_STATIC_ASSERTIONS,
                },
                {
                    "event": "sm121_agent_runtime_attestation",
                    "fresh_lifetime": lifetime,
                    "phase": phase,
                    **admission._runtime_expected(),
                },
                {
                    "event": "server_ready",
                    "backend": "sglang",
                    "fresh_lifetime": lifetime,
                    "phase": phase,
                    "first_inference_is_admission_gate": True,
                    "first_protocol_case": first_case,
                },
                {**gates[phase], "fresh_lifetime": lifetime},
                {
                    "event": "server_stopped",
                    "backend": "sglang",
                    "fresh_lifetime": lifetime,
                },
                {
                    "event": "sm121_agent_lifetime_complete",
                    "fresh_lifetime": lifetime,
                    "phase": phase,
                    "within_timeout": True,
                    "admitted": True,
                },
            )
        )
    events.extend(
        (
            {"event": "measurement_complete", "status": "complete"},
            {"event": "run_complete", "status": "admitted"},
        )
    )
    return events


class SM121AgentAdmissionRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_models(MODELS)[SM121_AGENT_ADMISSION_PROFILE_ID]
        cls.suite = load_suite(SUITE_PATH)

    def _freeze(self, temporary: Path) -> Path:
        logs_root = temporary / "logs"
        with (
            patch.object(admission, "_LOGS_ROOT", logs_root),
            patch("bench.runner._image_digest", return_value=None),
            patch(
                "bench.runner._sm121_storage_image_identity",
                return_value={
                    "docker_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
                    "platform": SM121_STORAGE_PLATFORM,
                    "source_tree": SM121_STORAGE_SOURCE_TREE,
                },
            ),
            patch("bench.runner._host_snapshot", return_value={"fixture": True}),
        ):
            return admission.create_sm121_agent_admission_plan(
                model=self.model,
                suite=self.suite,
                output_root=logs_root / "agent-admissions",
                models_path=MODELS,
                suite_path=SUITE_PATH,
            )

    def test_freeze_is_private_and_binds_the_prospective_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            self.assertEqual(0o700, run_dir.stat().st_mode & 0o777)
            with patch.object(admission, "_LOGS_ROOT", temporary / "logs"):
                _plan, model, suite = admission._load_plan(run_dir)
            self.assertFalse(hasattr(model, "agent_admission_authorized"))
            self.assertEqual(self.model.id, model.id)
            self.assertEqual(self.suite.id, suite.id)

    def test_execution_refuses_without_concrete_in_repository_adapters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            with patch(
                "bench.sm121_agent_admission_runner.base_runner.start_server"
            ) as start_server:
                with self.assertRaisesRegex(RuntimeError, "live adapters are not implemented"):
                    admission.execute_sm121_agent_admission(
                        run_dir, workspace=temporary
                    )
            start_server.assert_not_called()

    def test_caller_supplied_hooks_are_not_an_execution_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            with patch(
                "bench.sm121_agent_admission_runner.base_runner.start_server"
            ) as start_server:
                with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'hooks'"):
                    admission.execute_sm121_agent_admission(
                        run_dir, workspace=temporary, hooks=object()
                    )
            start_server.assert_not_called()

    def test_audit_rejects_a_planned_but_unexecuted_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            with patch.object(admission, "_LOGS_ROOT", temporary / "logs"):
                report = admission.audit_sm121_agent_admission(run_dir)
            self.assertFalse(report["ok"])
            self.assertIn("missing_record", {item["code"] for item in report["errors"]})

    def test_summary_integrity_rejects_boolean_mutation(self) -> None:
        summary = admission._summary(
            parser_static_admitted=True,
            quality_admitted=True,
            tools_admitted=True,
            long_context_admitted=True,
            source_static_attestations=3,
            runtime_attestations=3,
            completed_lifetimes=3,
            terminal_stage="complete",
            failure_code="generic",
        )
        summary["tools_admitted"] = False
        with self.assertRaises(admission.SM121AgentAdmissionError):
            admission._validate_summary(summary)
        summary = admission._summary(
            parser_static_admitted=True,
            quality_admitted=True,
            tools_admitted=True,
            long_context_admitted=True,
            source_static_attestations=3,
            runtime_attestations=3,
            completed_lifetimes=3,
            terminal_stage="complete",
            failure_code="generic",
        )
        summary["schema_version"] = True
        summary["integrity_hash"] = admission.content_hash(
            {key: value for key, value in summary.items() if key != "integrity_hash"},
            64,
        )
        with self.assertRaises(admission.SM121AgentAdmissionError):
            admission._validate_summary(summary)

    def test_plan_and_summary_require_full_sha256_integrity_hashes(self) -> None:
        summary = admission._summary(
            parser_static_admitted=True,
            quality_admitted=True,
            tools_admitted=True,
            long_context_admitted=True,
            source_static_attestations=3,
            runtime_attestations=3,
            completed_lifetimes=3,
            terminal_stage="complete",
            failure_code="generic",
        )
        summary["integrity_hash"] = ""
        with self.assertRaises(admission.SM121AgentAdmissionError):
            admission._validate_summary(summary)
        summary = admission._summary(
            parser_static_admitted=True,
            quality_admitted=True,
            tools_admitted=True,
            long_context_admitted=True,
            source_static_attestations=3,
            runtime_attestations=3,
            completed_lifetimes=3,
            terminal_stage="complete",
            failure_code="generic",
        )
        summary["terminal_stage"] = []
        summary["integrity_hash"] = admission.content_hash(
            {key: value for key, value in summary.items() if key != "integrity_hash"},
            64,
        )
        with self.assertRaises(admission.SM121AgentAdmissionError):
            admission._validate_summary(summary)

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            plan_path = run_dir / "plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["integrity_hash"] = ""
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with patch.object(admission, "_LOGS_ROOT", temporary / "logs"):
                with self.assertRaises(admission.base_runner.PreflightError):
                    admission._load_plan(run_dir)
            plan["integrity_hash"] = admission.content_hash(
                {key: value for key, value in plan.items() if key != "integrity_hash"},
                64,
            )
            plan["schema_version"] = True
            plan["integrity_hash"] = admission.content_hash(
                {key: value for key, value in plan.items() if key != "integrity_hash"},
                64,
            )
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with patch.object(admission, "_LOGS_ROOT", temporary / "logs"):
                with self.assertRaises(admission.base_runner.PreflightError):
                    admission._load_plan(run_dir)

    def test_short_complete_journal_fails_closed_without_indexing(self) -> None:
        summary = admission._summary(
            parser_static_admitted=True,
            quality_admitted=True,
            tools_admitted=True,
            long_context_admitted=True,
            source_static_attestations=3,
            runtime_attestations=3,
            completed_lifetimes=3,
            terminal_stage="complete",
            failure_code="generic",
        )
        errors = admission._complete_errors([], summary, ())
        self.assertEqual(errors[0]["code"], "event_topology")

    def test_structurally_complete_record_is_not_accepted_before_controller_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            journal = admission.Journal(run_dir / "events.jsonl")
            for event in _complete_events():
                journal.append(event)
            summary = admission._summary(
                parser_static_admitted=True,
                quality_admitted=True,
                tools_admitted=True,
                long_context_admitted=True,
                source_static_attestations=3,
                runtime_attestations=3,
                completed_lifetimes=3,
                terminal_stage="complete",
                failure_code="generic",
            )
            (run_dir / "admission.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            with patch.object(admission, "_LOGS_ROOT", temporary / "logs"):
                report = admission.audit_sm121_agent_admission(run_dir)
        self.assertFalse(report["ok"])
        self.assertEqual(
            {item["code"] for item in report["errors"]},
            {"controller_unimplemented"},
        )

    def test_partial_record_rejects_unknown_freeform_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            admission.Journal(run_dir / "events.jsonl").append(
                {"event": "unexpected", "details": "synthetic-only"}
            )
            summary = admission._summary(
                parser_static_admitted=False,
                quality_admitted=False,
                tools_admitted=False,
                long_context_admitted=False,
                source_static_attestations=0,
                runtime_attestations=0,
                completed_lifetimes=0,
                terminal_stage="parser_static",
                failure_code="static_parser",
            )
            (run_dir / "admission.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            with patch.object(admission, "_LOGS_ROOT", temporary / "logs"):
                report = admission.audit_sm121_agent_admission(run_dir)
        self.assertFalse(report["ok"])
        self.assertTrue(
            {"scalar_safety", "not_admitted"}
            <= {item["code"] for item in report["errors"]}
        )


if __name__ == "__main__":
    unittest.main()
