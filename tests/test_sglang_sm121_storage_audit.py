"""Read-only lifecycle contracts for the direct SM121 storage canary audit."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.audit import audit_sm121_storage_canary_run
from bench.manifest import load_models, load_suite
from bench.report import summarize_run
from bench.runner import (
    create_sm121_storage_canary_plan,
    execute_sm121_storage_canary,
)
from bench.sglang_sm121_storage import (
    SM121_STORAGE_EXECUTION_MODE,
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_RUNTIME_PROVENANCE_EVENT,
    SM121_STORAGE_SOURCE_TREE,
    _SM121_STORAGE_RUNTIME_PROVENANCE_EXPECTED,
)
from sparkbench import (
    build_parser,
    command_audit_sm121_storage_canary,
)


class SM121StorageCanaryAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Path(__file__).resolve().parents[1]
        self.model = load_models(self.workspace / "manifests" / "models.toml")[
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

    @staticmethod
    def _request_result() -> dict[str, object]:
        return {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "elapsed_s": 0.2,
            "ttft_s": 0.05,
            "decode_tps": 20.0,
            "decode_metric_source": "client_estimate",
        }

    def _events(self, run_dir: Path) -> list[dict[str, object]]:
        plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
        cases = plan["suite"]["cases"]
        first_id = cases[0]["case_id"]
        second_id = cases[1]["case_id"]

        def case_events(
            *, case_id: str, lifetime: int, kind: str
        ) -> list[dict[str, object]]:
            attempt_id = f"synthetic-attempt-{lifetime}"
            validation = (
                {"passed": True, "quality_category": "synthetic"}
                if kind == "quality"
                else {"passed": True}
            )
            return [
                {
                    "event": "case_start",
                    "case_id": case_id,
                    "attempt_id": attempt_id,
                    "kind": kind,
                    "concurrency": 1,
                },
                {
                    "event": "request_complete",
                    "case_id": case_id,
                    "attempt_id": attempt_id,
                    "kind": kind,
                    "repetition": 0,
                    "result": self._request_result(),
                    "validation": validation,
                },
                {
                    "event": "case_complete",
                    "case_id": case_id,
                    "attempt_id": attempt_id,
                    "kind": kind,
                    "concurrency": 1,
                    "elapsed_s": 0.2,
                    "validation_passed": True,
                },
            ]

        return [
            {
                "event": "run_start",
                "execution_mode": SM121_STORAGE_EXECUTION_MODE,
            },
            {"event": "measurement_started"},
            {
                "event": SM121_STORAGE_RUNTIME_PROVENANCE_EVENT,
                "fresh_server_lifetime": 1,
                **_SM121_STORAGE_RUNTIME_PROVENANCE_EXPECTED,
            },
            {
                "event": "server_ready",
                "backend": "sglang",
                "fresh_server_lifetime": 1,
                "first_inference_is_case": True,
                "case_id": first_id,
            },
            *case_events(case_id=first_id, lifetime=1, kind="quality"),
            {
                "event": "server_stopped",
                "backend": "sglang",
                "fresh_server_lifetime": 1,
            },
            {
                "event": SM121_STORAGE_RUNTIME_PROVENANCE_EVENT,
                "fresh_server_lifetime": 2,
                **_SM121_STORAGE_RUNTIME_PROVENANCE_EXPECTED,
            },
            {
                "event": "server_ready",
                "backend": "sglang",
                "fresh_server_lifetime": 2,
                "first_inference_is_case": True,
                "case_id": second_id,
            },
            *case_events(case_id=second_id, lifetime=2, kind="capability"),
            {
                "event": "server_stopped",
                "backend": "sglang",
                "fresh_server_lifetime": 2,
            },
            {"event": "measurement_complete", "elapsed_s": 0.4},
            {"event": "run_complete", "status": "completed"},
        ]

    @staticmethod
    def _write_events(run_dir: Path, events: list[dict[str, object]]) -> None:
        (run_dir / "events.jsonl").write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
            encoding="utf-8",
        )

    def test_valid_fresh_lifetime_run_audits_and_reports_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._freeze(Path(directory))
            self._write_events(run_dir, self._events(run_dir))

            report = audit_sm121_storage_canary_run(run_dir)
            summary = summarize_run(run_dir)

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(summary["status"], "complete")

    def test_executor_journal_matches_the_fresh_lifetime_audit_contract(self) -> None:
        """Exercise the actual executor's event order without a model load."""

        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._freeze(Path(directory))
            servers = [
                SimpleNamespace(
                    backend="sglang",
                    startup_s=float(lifetime),
                    native_provenance=dict(
                        _SM121_STORAGE_RUNTIME_PROVENANCE_EXPECTED
                    ),
                    stop=Mock(),
                )
                for lifetime in (1, 2)
            ]

            def synthetic_case(
                *, case: SimpleNamespace, journal: object, **_: object
            ) -> None:
                attempt_id = f"synthetic-{case.id}"
                validation = (
                    {"passed": True, "quality_category": "synthetic"}
                    if case.kind == "quality"
                    else {"passed": True}
                )
                append = getattr(journal, "append")
                append(
                    {
                        "event": "case_start",
                        "case_id": case.case_id,
                        "attempt_id": attempt_id,
                        "kind": case.kind,
                        "concurrency": case.concurrency,
                    }
                )
                append(
                    {
                        "event": "request_complete",
                        "case_id": case.case_id,
                        "attempt_id": attempt_id,
                        "kind": case.kind,
                        "repetition": 0,
                        "result": self._request_result(),
                        "validation": validation,
                    }
                )
                append(
                    {
                        "event": "case_complete",
                        "case_id": case.case_id,
                        "attempt_id": attempt_id,
                        "kind": case.kind,
                        "concurrency": case.concurrency,
                        "elapsed_s": 0.2,
                        "validation_passed": True,
                    }
                )

            with (
                patch("bench.runner._preflight"),
                patch("bench.runner._host_safety_watchdog", return_value=None),
                patch("bench.runner.TelemetrySampler", return_value=Mock()),
                patch("bench.runner.start_server", side_effect=servers),
                patch("bench.runner.capture_server_provenance", return_value={}),
                patch("bench.runner.save_server_logs"),
                patch("bench.runner._execute_case", side_effect=synthetic_case),
            ):
                summary = execute_sm121_storage_canary(
                    run_dir, workspace=self.workspace
                )
            report = audit_sm121_storage_canary_run(run_dir)
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(summary["status"], "complete")
        self.assertTrue(report["ok"], report["errors"])
        self.assertFalse(
            any(event["event"] == "first_request_complete" for event in events)
        )
        provenance_indexes = [
            index
            for index, event in enumerate(events)
            if event["event"] == SM121_STORAGE_RUNTIME_PROVENANCE_EVENT
        ]
        self.assertEqual(len(provenance_indexes), 2)
        for index in provenance_indexes:
            self.assertEqual(events[index + 1]["event"], "server_ready")

    def test_rejects_reordered_lifetime_case_missing_case_and_primer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            cases = {
                "reordered": "sm121_storage_case_order_mismatch",
                "missing": "sm121_storage_case_complete_count",
                "primer": "sm121_storage_unexpected_primer",
                "provenance": "sm121_storage_invalid_runtime_provenance",
            }
            for name, expected_code in cases.items():
                with self.subTest(name=name):
                    run_dir = self._freeze(results / name)
                    events = self._events(run_dir)
                    plan = json.loads(
                        (run_dir / "plan.json").read_text(encoding="utf-8")
                    )
                    first_id = plan["suite"]["cases"][0]["case_id"]
                    second_id = plan["suite"]["cases"][1]["case_id"]
                    if name == "reordered":
                        for event in events:
                            if event.get("case_id") == first_id:
                                event["case_id"] = second_id
                            elif event.get("case_id") == second_id:
                                event["case_id"] = first_id
                    elif name == "missing":
                        events = [
                            event
                            for event in events
                            if not (
                                event.get("event")
                                in {
                                    "case_start",
                                    "request_complete",
                                    "case_complete",
                                }
                                and event.get("case_id") == second_id
                            )
                        ]
                    elif name == "provenance":
                        next(
                            event
                            for event in events
                            if event.get("event")
                            == SM121_STORAGE_RUNTIME_PROVENANCE_EVENT
                            and event.get("fresh_server_lifetime") == 2
                        )["candidate_id"] = "wrong"
                    else:
                        events.insert(
                            4,
                            {
                                "event": "first_request_complete",
                                "result": {"completion_tokens": 1},
                            },
                        )
                    self._write_events(run_dir, events)

                    report = audit_sm121_storage_canary_run(run_dir)
                    summary = summarize_run(run_dir)

                    self.assertFalse(report["ok"])
                    self.assertIn(
                        expected_code,
                        {issue["code"] for issue in report["errors"]},
                    )
                    self.assertEqual(summary["status"], "partial")

    def test_cli_is_read_only_and_returns_nonzero_for_invalid_topology(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._freeze(Path(directory))
            events = self._events(run_dir)
            events.insert(4, {"event": "first_request_complete"})
            self._write_events(run_dir, events)
            before = (run_dir / "events.jsonl").read_bytes()
            output = io.StringIO()
            with redirect_stdout(output):
                status = command_audit_sm121_storage_canary(
                    argparse.Namespace(run_dir=run_dir)
                )
            payload = json.loads(output.getvalue())

            self.assertEqual(before, (run_dir / "events.jsonl").read_bytes())

        self.assertEqual(status, 1)
        self.assertFalse(payload["ok"])

    def test_parser_exposes_direct_storage_canary_audit(self) -> None:
        args = build_parser().parse_args(
            ["audit-sm121-storage-canary", "synthetic-run"]
        )

        self.assertIs(args.function, command_audit_sm121_storage_canary)
        self.assertEqual(args.run_dir, Path("synthetic-run"))


if __name__ == "__main__":
    unittest.main()
