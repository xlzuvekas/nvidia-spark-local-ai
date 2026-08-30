"""Offline contract tests for the private SM121 C1 plan/audit scaffold."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench import sm121_agent_admission_runner as admission
from bench.manifest import load_models, load_suite
from bench.sglang_sm121_agent_admission import (
    SM121_AGENT_ADMISSION_LONG_CONTEXT_BUDGET_PROBE_ID,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_BUDGET_PROBE_SCHEMA_VERSION,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_BUDGET_TOKENS,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_PROMPT_TOKENS,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_TEMPLATE_SHA256,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_OUTPUT_TOKENS,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_RAW_PROMPT_SHA256,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_RENDERED_PROMPT_SHA256,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_TOKENIZER_SHA256,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_TOOLS_SHA256,
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
            "reasoning_parser_instantiated": True,
            "tool_call_parser_instantiated": True,
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_coder",
            "chunked_prefill_size": 4096,
            "max_running_requests": 1,
            "max_total_tokens": 65536,
            "context_length": 65536,
        },
    ]
    events.append(_long_context_budget_probe())
    gates = {
        "quality": admission._quality_gate_expected(),
        "tools": admission._tool_gate_expected(),
        "long_context": admission._long_context_gate_expected(),
    }
    metrics_before = {
        field: 0
        for field in admission.SM121_AGENT_ADMISSION_NATIVE_CACHE_METRIC_FIELDS
    }
    metrics_after = dict(metrics_before)
    metrics_after["prefill_input_tokens"] = 60_000
    native_receipt = {
        "event": "sm121_agent_native_cache_metrics_receipt",
        "schema_version": 1,
        "fresh_lifetime": 3,
        "same_owned_generation": True,
        "metrics_available": True,
        "guardrail_metrics_available": True,
        "metrics_before_settled": True,
        "metrics_after_settled": True,
        "metrics_before_polls": 2,
        "metrics_after_polls": 2,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "native_input_observed": True,
        "zero_metric_cache_hits": True,
        "guardrails_clean": True,
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
                *(
                    (native_receipt,)
                    if phase == "long_context"
                    else ()
                ),
                gates[phase],
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


def _long_context_budget_probe() -> dict[str, object]:
    return {
        "event": "sm121_agent_long_context_budget_preflight",
        "schema_version": SM121_AGENT_ADMISSION_LONG_CONTEXT_BUDGET_PROBE_SCHEMA_VERSION,
        "probe_id": SM121_AGENT_ADMISSION_LONG_CONTEXT_BUDGET_PROBE_ID,
        "raw_prompt_sha256": SM121_AGENT_ADMISSION_LONG_CONTEXT_RAW_PROMPT_SHA256,
        "tools_sha256": SM121_AGENT_ADMISSION_LONG_CONTEXT_TOOLS_SHA256,
        "tokenizer_sha256": SM121_AGENT_ADMISSION_LONG_CONTEXT_TOKENIZER_SHA256,
        "chat_template_sha256": SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_TEMPLATE_SHA256,
        "rendered_prompt_sha256": SM121_AGENT_ADMISSION_LONG_CONTEXT_RENDERED_PROMPT_SHA256,
        "chat_prompt_tokens": SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_PROMPT_TOKENS,
        "output_tokens": SM121_AGENT_ADMISSION_LONG_CONTEXT_OUTPUT_TOKENS,
        "budget_tokens": SM121_AGENT_ADMISSION_LONG_CONTEXT_BUDGET_TOKENS,
        "context_length": 65_536,
        "within_context": True,
    }


def _native_receipt() -> dict[str, object]:
    return next(
        event
        for event in _complete_events()
        if event["event"] == "sm121_agent_native_cache_metrics_receipt"
    )


class _FakeServer:
    backend = "sglang"

    def __init__(self) -> None:
        self.stopped = False
        self.interrupted = False

    def stop(self) -> None:
        self.stopped = True

    def interrupt_owned(self) -> None:
        self.interrupted = True


class _QualityClient:
    """One in-memory exact-answer client fixture for the C1 prompt test."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self._answers = [
            item.expected_answer
            for _ in range(2)
            for item in admission.base_runner._QUALITY_ITEMS
        ]

    def _bind_controller_deadline(self, deadline: float) -> None:
        self.deadline = deadline

    def run_quality_turn(self, *, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content=f"FINAL: {self._answers.pop(0)}")

    def diagnostics(self) -> SimpleNamespace:
        return SimpleNamespace(
            to_dict=lambda: admission._payload_diagnostics(
                outbound=8, tools=0, cache_zero=0
            )
        )


class _HostSafetyFailure:
    """Synchronous host-gate double with no server-side effects."""

    def __init__(self) -> None:
        self.stopped = False

    def start(self) -> None:
        return None

    def raise_if_tripped(self) -> None:
        raise admission.HostSafetyError("synthetic", "synthetic host gate")

    def stop(self) -> None:
        self.stopped = True


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

    def test_freeze_is_private_and_returns_no_launch_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            self.assertEqual(0o700, run_dir.stat().st_mode & 0o777)
            with patch.object(admission, "_LOGS_ROOT", temporary / "logs"):
                _plan, model, suite = admission._load_plan(run_dir)
            self.assertFalse(hasattr(model, "agent_admission_authorized"))
            self.assertEqual(self.model.id, model.id)
            self.assertEqual(self.suite.id, suite.id)
            self.assertFalse(
                hasattr(admission, "_register_sm121_agent_admission_controller")
            )
            with self.assertRaisesRegex(
                admission.SM121AgentAdmissionError,
                "launch is not authorized",
            ):
                admission._require_sm121_agent_admission_plan_binding(
                    model, admission._SM121AgentAdmissionLaunchLease()
                )

    def test_unregistered_session_cannot_issue_a_lease(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            with patch.object(admission, "_LOGS_ROOT", temporary / "logs"):
                _plan, model, _suite = admission._load_plan(run_dir)
            with self.assertRaises(admission.SM121AgentAdmissionError):
                admission._issue_sm121_agent_admission_launch_lease(
                    model,
                    admission._SM121AgentAdmissionControllerSession(),
                    fresh_lifetime=1,
                )
            self.assertFalse(admission._SM121_AGENT_ADMISSION_CONTROLLERS)
            self.assertFalse(admission._SM121_AGENT_ADMISSION_LEASES)

    def test_execution_requires_the_private_logs_topology_before_any_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            with patch(
                "bench.sm121_agent_admission_runner.start_sm121_agent_admission_server"
            ) as start_server:
                with patch.object(admission, "_LOGS_ROOT", temporary / "other-logs"):
                    with self.assertRaisesRegex(
                        admission.base_runner.PreflightError,
                        "logs are unavailable",
                    ):
                        admission.execute_sm121_agent_admission(
                            run_dir, workspace=temporary
                        )
            start_server.assert_not_called()

    def test_caller_supplied_hooks_are_not_an_execution_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            with patch(
                "bench.sm121_agent_admission_runner.start_sm121_agent_admission_server"
            ) as start_server:
                with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'hooks'"):
                    admission.execute_sm121_agent_admission(
                        run_dir, workspace=temporary, hooks=object()
                    )
            start_server.assert_not_called()

    def test_controller_runs_three_private_lifetimes_and_audits_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            servers = [_FakeServer(), _FakeServer(), _FakeServer()]
            parser_event = _complete_events()[2]

            def long_gate(**kwargs: object) -> dict[str, object]:
                journal = kwargs["journal"]
                assert isinstance(journal, admission.Journal)
                journal.append(_native_receipt())
                return admission._long_context_gate_expected()

            with (
                patch.object(admission, "_LOGS_ROOT", temporary / "logs"),
                patch.object(admission.base_runner, "_preflight"),
                patch.object(admission, "require_sm121_agent_admission_clean_start"),
                patch.object(admission.base_runner, "_host_safety_watchdog", return_value=None),
                patch.object(
                    admission,
                    "probe_sm121_agent_parser_static_preflight",
                    return_value=parser_event,
                ),
                patch.object(
                    admission,
                    "sm121_agent_admission_target_snapshot",
                    return_value=temporary,
                ),
                patch.object(
                    admission,
                    "probe_sm121_agent_long_context_budget_preflight",
                    return_value={
                        key: value
                        for key, value in _long_context_budget_probe().items()
                        if key != "event"
                    },
                ),
                patch.object(
                    admission,
                    "inspect_sm121_cache_source_digests",
                    return_value=dict(admission.SM121_CACHE_SOURCE_DIGESTS),
                ),
                patch.object(
                    admission,
                    "inspect_sm121_agent_admission_runtime_identity",
                    return_value=admission._runtime_expected(),
                ),
                patch.object(
                    admission,
                    "start_sm121_agent_admission_server",
                    side_effect=servers,
                ) as start_server,
                patch.object(admission, "_quality_gate", return_value=admission._quality_gate_expected()),
                patch.object(admission, "_tool_gate", return_value=admission._tool_gate_expected()),
                patch.object(admission, "_long_context_gate", side_effect=long_gate),
            ):
                summary = admission.execute_sm121_agent_admission(
                    run_dir, workspace=temporary
                )
                report = admission.audit_sm121_agent_admission(run_dir)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["decision"], "admitted")
        self.assertEqual(start_server.call_count, 3)
        leases = [
            call.kwargs["_launch_capability"]
            for call in start_server.call_args_list
        ]
        self.assertEqual(3, len({id(lease) for lease in leases}))
        self.assertFalse(admission._SM121_AGENT_ADMISSION_CONTROLLERS)
        self.assertFalse(admission._SM121_AGENT_ADMISSION_LEASES)
        self.assertTrue(all(server.stopped for server in servers))
        self.assertTrue(report["ok"], report)

    def test_controller_stops_after_first_gate_failure_and_writes_scalar_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            server = _FakeServer()
            parser_event = _complete_events()[2]
            with (
                patch.object(admission, "_LOGS_ROOT", temporary / "logs"),
                patch.object(admission.base_runner, "_preflight"),
                patch.object(admission, "require_sm121_agent_admission_clean_start"),
                patch.object(admission.base_runner, "_host_safety_watchdog", return_value=None),
                patch.object(
                    admission,
                    "probe_sm121_agent_parser_static_preflight",
                    return_value=parser_event,
                ),
                patch.object(
                    admission,
                    "sm121_agent_admission_target_snapshot",
                    return_value=temporary,
                ),
                patch.object(
                    admission,
                    "probe_sm121_agent_long_context_budget_preflight",
                    return_value={
                        key: value
                        for key, value in _long_context_budget_probe().items()
                        if key != "event"
                    },
                ),
                patch.object(
                    admission,
                    "inspect_sm121_cache_source_digests",
                    return_value=dict(admission.SM121_CACHE_SOURCE_DIGESTS),
                ),
                patch.object(
                    admission,
                    "inspect_sm121_agent_admission_runtime_identity",
                    return_value=admission._runtime_expected(),
                ),
                patch.object(
                    admission,
                    "start_sm121_agent_admission_server",
                    return_value=server,
                ) as start_server,
                patch.object(
                    admission,
                    "_quality_gate",
                    side_effect=admission.SM121AgentAdmissionExecutionError("quality"),
                ),
            ):
                summary = admission.execute_sm121_agent_admission(
                    run_dir, workspace=temporary
                )
            events = admission.Journal(run_dir / "events.jsonl").strict_events()
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["failure_code"], "quality")
        self.assertEqual(start_server.call_count, 1)
        self.assertTrue(server.stopped)
        self.assertFalse(admission._SM121_AGENT_ADMISSION_CONTROLLERS)
        self.assertFalse(admission._SM121_AGENT_ADMISSION_LEASES)
        self.assertEqual(
            events[-3]["event"],
            "sm121_agent_blocked",
        )
        self.assertEqual(events[-3]["failure_code"], "quality")

    def test_complete_gate_booleans_downgrade_when_the_exact_record_audit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            servers = [_FakeServer(), _FakeServer(), _FakeServer()]
            parser_event = _complete_events()[2]

            def long_gate(**kwargs: object) -> dict[str, object]:
                journal = kwargs["journal"]
                assert isinstance(journal, admission.Journal)
                journal.append(_native_receipt())
                return admission._long_context_gate_expected()

            malformed_quality = dict(admission._quality_gate_expected())
            malformed_quality["quality_admitted"] = False
            with (
                patch.object(admission, "_LOGS_ROOT", temporary / "logs"),
                patch.object(admission.base_runner, "_preflight"),
                patch.object(admission, "require_sm121_agent_admission_clean_start"),
                patch.object(admission.base_runner, "_host_safety_watchdog", return_value=None),
                patch.object(
                    admission,
                    "probe_sm121_agent_parser_static_preflight",
                    return_value=parser_event,
                ),
                patch.object(
                    admission,
                    "sm121_agent_admission_target_snapshot",
                    return_value=temporary,
                ),
                patch.object(
                    admission,
                    "probe_sm121_agent_long_context_budget_preflight",
                    return_value={
                        key: value
                        for key, value in _long_context_budget_probe().items()
                        if key != "event"
                    },
                ),
                patch.object(
                    admission,
                    "inspect_sm121_cache_source_digests",
                    return_value=dict(admission.SM121_CACHE_SOURCE_DIGESTS),
                ),
                patch.object(
                    admission,
                    "inspect_sm121_agent_admission_runtime_identity",
                    return_value=admission._runtime_expected(),
                ),
                patch.object(
                    admission,
                    "start_sm121_agent_admission_server",
                    side_effect=servers,
                ),
                patch.object(admission, "_quality_gate", return_value=malformed_quality),
                patch.object(admission, "_tool_gate", return_value=admission._tool_gate_expected()),
                patch.object(admission, "_long_context_gate", side_effect=long_gate),
            ):
                summary = admission.execute_sm121_agent_admission(
                    run_dir, workspace=temporary
                )
                report = admission.audit_sm121_agent_admission(run_dir)
            events = admission.Journal(run_dir / "events.jsonl").strict_events()
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["decision"], "blocked")
        self.assertFalse(summary["record_valid"])
        self.assertFalse(report["ok"])
        self.assertEqual(events[-3]["event"], "sm121_agent_blocked")
        self.assertEqual(events[-3]["terminal_stage"], "record_audit")

    def test_quality_gate_uses_the_explicit_exact_answer_grammar(self) -> None:
        client = _QualityClient()
        case = SimpleNamespace(repetitions=2)
        with patch.object(
            admission, "create_sm121_agent_admission_client", return_value=client
        ):
            gate = admission._quality_gate(
                server=object(),
                model=SimpleNamespace(),
                case=case,
                deadline=admission.time.monotonic() + 10.0,
                watchdog=None,
            )
        self.assertEqual(gate, admission._quality_gate_expected())
        self.assertEqual(len(client.prompts), 8)
        for prompt, item in zip(
            client.prompts,
            admission.base_runner._QUALITY_ITEMS * 2,
            strict=True,
        ):
            self.assertIn(item.question, prompt)
            self.assertIn("`FINAL: <answer>`", prompt)
            self.assertIn("Do not include an explanation.", prompt)

    def test_long_context_wires_settled_before_request_after_and_receipt_in_order(self) -> None:
        trace: list[str] = []
        before = {
            field: 0
            for field in admission.SM121_AGENT_ADMISSION_NATIVE_CACHE_METRIC_FIELDS
        }
        before["available"] = True
        before["guardrail_metrics_available"] = True
        after = dict(before)
        after["prefill_input_tokens"] = 60_000
        lease = object()

        class Client:
            def _bind_controller_deadline(self, deadline: float) -> None:
                self.deadline = deadline

            def run_long_context_turn(self) -> None:
                trace.append("request")

            def long_context_receipt(self) -> dict[str, bool]:
                return {
                    "input_tokenization_verified": True,
                    "context_fit": True,
                    "zero_response_cache_hits": True,
                    "response_semantics_verified": True,
                    "first_turn_only": True,
                }

            def diagnostics(self) -> SimpleNamespace:
                return SimpleNamespace(
                    to_dict=lambda: admission._payload_diagnostics(
                        outbound=1, tools=1, cache_zero=1
                    )
                )

        client = Client()

        def settle(*_args: object, **kwargs: object):
            if kwargs.get("expected_lease") is None:
                trace.append("before")
                return before, lease, 2, True
            self.assertIs(kwargs["expected_lease"], lease)
            trace.append("after")
            return after, lease, 2, True

        with tempfile.TemporaryDirectory() as directory:
            journal = admission.Journal(Path(directory) / "events.jsonl")
            with (
                patch.object(
                    admission,
                    "create_sm121_agent_admission_client",
                    return_value=client,
                ),
                patch.object(
                    admission,
                    "settle_sm121_agent_admission_metrics",
                    side_effect=settle,
                ),
            ):
                gate = admission._long_context_gate(
                    server=object(),
                    model=SimpleNamespace(),
                    deadline=admission.time.monotonic() + 10.0,
                    watchdog=None,
                    journal=journal,
                )
            events = journal.strict_events()
        self.assertEqual(gate, admission._long_context_gate_expected())
        self.assertEqual(trace, ["before", "request", "after"])
        self.assertEqual(
            [event["event"] for event in events],
            ["sm121_agent_native_cache_metrics_receipt"],
        )

    def test_long_context_does_not_write_a_receipt_when_the_before_read_fails(self) -> None:
        client = SimpleNamespace(
            _bind_controller_deadline=lambda _deadline: None,
            run_long_context_turn=lambda: self.fail("request must not run"),
        )
        with tempfile.TemporaryDirectory() as directory:
            journal = admission.Journal(Path(directory) / "events.jsonl")
            with (
                patch.object(
                    admission,
                    "create_sm121_agent_admission_client",
                    return_value=client,
                ),
                patch.object(
                    admission,
                    "settle_sm121_agent_admission_metrics",
                    side_effect=admission.RuntimeErrorWithContext("metrics failed"),
                ),
                self.assertRaises(admission.SM121AgentAdmissionExecutionError),
            ):
                admission._long_context_gate(
                    server=object(),
                    model=SimpleNamespace(),
                    deadline=admission.time.monotonic() + 10.0,
                    watchdog=None,
                    journal=journal,
                )
            self.assertEqual(journal.events(), [])

    def test_host_gate_blocks_before_parser_or_server_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            watchdog = _HostSafetyFailure()
            with (
                patch.object(admission, "_LOGS_ROOT", temporary / "logs"),
                patch.object(admission.base_runner, "_preflight"),
                patch.object(admission, "require_sm121_agent_admission_clean_start"),
                patch.object(
                    admission.base_runner,
                    "_host_safety_watchdog",
                    return_value=watchdog,
                ),
                patch.object(
                    admission,
                    "probe_sm121_agent_parser_static_preflight",
                ) as parser,
                patch.object(
                    admission,
                    "start_sm121_agent_admission_server",
                ) as start_server,
            ):
                summary = admission.execute_sm121_agent_admission(
                    run_dir, workspace=temporary
                )
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["failure_code"], "host_safety")
        self.assertTrue(watchdog.stopped)
        parser.assert_not_called()
        start_server.assert_not_called()

    def test_clean_start_blocks_before_parser_or_server_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            with (
                patch.object(admission, "_LOGS_ROOT", temporary / "logs"),
                patch.object(admission.base_runner, "_preflight"),
                patch.object(
                    admission,
                    "require_sm121_agent_admission_clean_start",
                    side_effect=admission.RuntimeErrorWithContext("occupied"),
                ) as clean_start,
                patch.object(
                    admission,
                    "probe_sm121_agent_parser_static_preflight",
                ) as parser,
                patch.object(
                    admission,
                    "start_sm121_agent_admission_server",
                ) as start_server,
                patch.object(
                    admission,
                    "_issue_sm121_agent_admission_launch_lease",
                ) as issue,
            ):
                summary = admission.execute_sm121_agent_admission(
                    run_dir, workspace=temporary
                )
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["failure_code"], "preflight")
        clean_start.assert_called_once_with()
        parser.assert_not_called()
        start_server.assert_not_called()
        issue.assert_not_called()

    def test_each_lifetime_rechecks_clean_start_before_static_or_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            journal = admission.Journal(temporary / "events.jsonl")
            with (
                patch.object(admission.base_runner, "_preflight"),
                patch.object(
                    admission,
                    "require_sm121_agent_admission_clean_start",
                    side_effect=admission.RuntimeErrorWithContext("occupied"),
                ) as clean_start,
                patch.object(admission, "_static_event") as static,
                patch.object(
                    admission, "start_sm121_agent_admission_server"
                ) as start_server,
                self.assertRaises(admission.SM121AgentAdmissionExecutionError) as caught,
            ):
                admission._run_lifetime(
                    run_dir=temporary,
                    workspace=temporary,
                    model=SimpleNamespace(),
                    fresh_lifetime=1,
                    phase="quality",
                    first_case_id=admission.SM121_AGENT_ADMISSION_QUALITY_CASE_ID,
                    controller_session=object(),
                    journal=journal,
                    operation=lambda *_args: {},
                )
        self.assertEqual(caught.exception.failure_code, "preflight")
        clean_start.assert_called_once_with()
        static.assert_not_called()
        start_server.assert_not_called()

    def test_launch_prerequisite_failure_is_not_mislabeled_as_runtime_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            journal = admission.Journal(temporary / "events.jsonl")
            with (
                patch.object(admission.base_runner, "_preflight"),
                patch.object(admission, "require_sm121_agent_admission_clean_start"),
                patch.object(admission.base_runner, "_host_safety_watchdog", return_value=None),
                patch.object(
                    admission,
                    "_static_event",
                    return_value={
                        "event": "sm121_agent_static_attestation",
                        "fresh_lifetime": 1,
                        "phase": "quality",
                    },
                ),
                patch.object(
                    admission,
                    "start_sm121_agent_admission_server",
                    side_effect=admission.RuntimeErrorWithContext("docker unavailable"),
                ),
                patch.object(
                    admission,
                    "_issue_sm121_agent_admission_launch_lease",
                    return_value=object(),
                ),
                patch.object(
                    admission,
                    "recover_owned_sglang",
                    return_value="already_absent",
                ) as recover,
                self.assertRaises(admission.SM121AgentAdmissionExecutionError) as caught,
            ):
                admission._run_lifetime(
                    run_dir=temporary,
                    workspace=temporary,
                    model=SimpleNamespace(run_identity="synthetic-run"),
                    fresh_lifetime=1,
                    phase="quality",
                    first_case_id=admission.SM121_AGENT_ADMISSION_QUALITY_CASE_ID,
                    controller_session=object(),
                    journal=journal,
                    operation=lambda *_args: {},
                )
        self.assertEqual(caught.exception.failure_code, "dependency_unavailable")
        recover.assert_called_once_with(
            "synthetic-run",
            api_key_path=temporary / "server" / "lifetime-1" / "api-key",
        )

    def test_prior_partial_summary_recovers_owned_lifetimes_before_refusing_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_dir = self._freeze(temporary)
            admission.write_json(run_dir / "admission.json", {"partial": True})
            with (
                patch.object(admission, "_LOGS_ROOT", temporary / "logs"),
                patch.object(
                    admission,
                    "recover_owned_sglang",
                    return_value="already_absent",
                ) as recover,
                self.assertRaisesRegex(
                    admission.base_runner.PreflightError,
                    "non-resumable",
                ),
            ):
                admission.execute_sm121_agent_admission(
                    run_dir, workspace=temporary
                )
        self.assertEqual(recover.call_count, 3)

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

    def test_structurally_complete_record_is_accepted_by_the_read_only_auditor(self) -> None:
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
        self.assertTrue(report["ok"])
        self.assertEqual(report["errors"], [])

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
