from __future__ import annotations

from contextlib import redirect_stdout
import argparse
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.audit import audit_matrix
from bench.journal import content_hash
from sparkbench import build_parser, command_audit_matrix, command_matrix


class MatrixFixture:
    def __init__(self, root: Path, *, max_num_seqs: int = 1) -> None:
        self.matrix_dir = root / "20260816T000000Z-quick"
        self.matrix_dir.mkdir()
        self.model = {
            "id": "fixture-model",
            "backend": "vllm",
            "source": "example/model",
            "revision": "fixed-revision",
            "image_digest": "sha256:abc123",
            "args": ["--max-num-seqs", str(max_num_seqs)],
        }
        self.case = {
            "id": "decode-40",
            "kind": "decode",
            "max_output_tokens": 40,
            "prompt_repetitions": 0,
            "repetitions": 2,
            "concurrency": 1,
            "temperature": 0.0,
            "warmups": 0,
            "requires": ["chat"],
        }
        self.suite = {
            "schema_version": 1,
            "id": "quick",
            "description": "fixture",
            "cases": [self.case],
        }
        self.resolved = {"image_digest": "example/image@sha256:abc123"}
        self.fingerprint = content_hash(
            {"model": self.model, "suite": self.suite, "resolved": self.resolved}
        )
        self.case_id = (
            f"{self.case['id']}--"
            f"{content_hash({'model': self.model, 'case': self.case}, 12)}"
        )
        self.run_dir = (
            self.matrix_dir
            / f"20260816T000000Z-fixture-model-quick-{self.fingerprint[:8]}"
        )
        self.run_dir.mkdir()
        self.plan = {
            "schema_version": 2,
            "created_at": "2026-08-16T00:00:00.000+00:00",
            "fingerprint": self.fingerprint,
            "models_manifest": "manifests/models.toml",
            "suite_manifest": "manifests/suites/quick.toml",
            "model": self.model,
            "suite": {**self.suite, "cases": [{**self.case, "case_id": self.case_id}]},
            "resolved": self.resolved,
            "host_at_plan": {},
        }
        self.plan["integrity_hash"] = content_hash(self.plan, 64)
        self.events = self._measured_events()
        self.summary = self._measured_summary()
        self.index = {
            "created_at": "2026-08-16T00:00:00.000+00:00",
            "suite": "quick",
            "models": ["fixture-model"],
            "runs": [
                {
                    "model": "fixture-model",
                    "run_dir": str(self.run_dir),
                    "status": "complete",
                    "completed_cases": 1,
                }
            ],
        }
        self.write()

    @staticmethod
    def _timestamp(index: int) -> str:
        return f"2026-08-16T00:00:{index:02d}.000+00:00"

    def _measured_events(self) -> list[dict[str, object]]:
        attempt = "attempt-1"
        raw = [
            {"event": "run_start", "completed_cases_at_resume": []},
            {"event": "server_ready", "backend": "vllm", "startup_s": 1.0},
            {"event": "first_request_complete", "result": {"completion_tokens": 1}},
            {
                "event": "case_start",
                "case_id": self.case_id,
                "attempt_id": attempt,
                "kind": "decode",
                "concurrency": 1,
            },
            {
                "event": "request_complete",
                "case_id": self.case_id,
                "attempt_id": attempt,
                "kind": "decode",
                "result": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "elapsed_s": 1.0,
                },
            },
            {
                "event": "request_complete",
                "case_id": self.case_id,
                "attempt_id": attempt,
                "kind": "decode",
                "result": {
                    "prompt_tokens": 12,
                    "completion_tokens": 20,
                    "elapsed_s": 1.5,
                },
            },
            {
                "event": "case_complete",
                "case_id": self.case_id,
                "attempt_id": attempt,
                "kind": "decode",
                "concurrency": 1,
                "elapsed_s": 4.0,
                "validation_passed": True,
            },
            {"event": "server_stopped", "backend": "vllm"},
            {"event": "run_complete", "status": "completed"},
        ]
        return [
            {"timestamp": self._timestamp(index), **event}
            for index, event in enumerate(raw)
        ]

    def _measured_summary(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "model": {"id": "fixture-model"},
            "suite": "quick",
            "status": "complete",
            "run_completion_status": "completed",
            "completed_cases": 1,
            "failed_cases": [],
            "validation_failed_cases": [],
            "unimplemented_cases": [],
            "unsupported_cases": [],
            "context_limited_cases": [],
            "cases": [
                {
                    "case_id": self.case_id,
                    "attempt_id": "attempt-1",
                    "kind": "decode",
                    "requests": 2,
                    "concurrency": 1,
                    "prompt_tokens": 22,
                    "completion_tokens": 40,
                    "elapsed_s": 4.0,
                    "aggregate_output_tps": 10.0,
                    "validation_passed": True,
                }
            ],
        }

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def write(self) -> None:
        self._write_json(self.matrix_dir / "matrix.json", self.index)
        self._write_json(self.run_dir / "plan.json", self.plan)
        self._write_json(self.run_dir / "summary.json", self.summary)
        (self.run_dir / "events.jsonl").write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in self.events)
        )


class MatrixAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = MatrixFixture(Path(self.temporary.name))

    def test_valid_matrix_recomputes_aggregate_and_flags_serialized_queue(self) -> None:
        report = audit_matrix(self.fixture.matrix_dir)

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["error_count"], 0)
        self.assertTrue(report["read_only"])
        self.assertEqual(
            report["matrix"]["serialized_queue_runs"], ["fixture-model"]
        )
        run = report["runs"][0]
        self.assertEqual(run["scheduling_label"], "serialized_queue")
        self.assertEqual(run["max_num_seqs"], 1)
        recomputed = run["recomputed_cases"][0]
        self.assertEqual(recomputed["completion_tokens"], 40)
        self.assertEqual(recomputed["wall_s"], 4.0)
        self.assertEqual(recomputed["aggregate_output_tps"], 10.0)
        self.assertTrue(recomputed["summary_matches"])

    def test_audit_does_not_modify_matrix_files(self) -> None:
        paths = sorted(path for path in self.fixture.matrix_dir.rglob("*") if path.is_file())
        before = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in paths
        }

        report = audit_matrix(self.fixture.matrix_dir)

        after_paths = sorted(
            path for path in self.fixture.matrix_dir.rglob("*") if path.is_file()
        )
        after = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in after_paths
        }
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(before, after)

    def test_discrepancies_emit_json_and_nonzero(self) -> None:
        self.fixture.summary["cases"][0]["aggregate_output_tps"] = 9.5  # type: ignore[index]
        self.fixture.events = [
            event for event in self.fixture.events if event["event"] != "server_stopped"
        ]
        self.fixture.index["runs"][0]["status"] = "running"  # type: ignore[index]
        self.fixture.write()

        output = io.StringIO()
        with redirect_stdout(output):
            status = command_audit_matrix(
                argparse.Namespace(matrix_dir=self.fixture.matrix_dir)
            )
        payload = json.loads(output.getvalue())
        codes = {issue["code"] for issue in payload["errors"]}

        self.assertEqual(status, 1)
        self.assertFalse(payload["ok"])
        self.assertIn("matrix_run_not_finished", codes)
        self.assertIn("missing_server_cleanup", codes)
        self.assertIn("aggregate_output_tps_mismatch", codes)

    def test_tampered_plan_fails_fingerprint_and_integrity(self) -> None:
        self.fixture.plan["model"]["revision"] = "tampered"  # type: ignore[index]
        self.fixture.write()

        report = audit_matrix(self.fixture.matrix_dir)
        codes = {issue["code"] for issue in report["errors"]}

        self.assertFalse(report["ok"])
        self.assertIn("plan_integrity_mismatch", codes)
        self.assertIn("plan_fingerprint_mismatch", codes)

    def test_no_work_skip_is_a_complete_lifecycle_without_server(self) -> None:
        self.fixture.events = [
            {
                "timestamp": self.fixture._timestamp(0),
                "event": "run_start",
                "completed_cases_at_resume": [],
            },
            {
                "timestamp": self.fixture._timestamp(1),
                "event": "case_skipped_unsupported",
                "case_id": self.fixture.case_id,
            },
            {
                "timestamp": self.fixture._timestamp(2),
                "event": "run_complete",
                "status": "no_work",
            },
        ]
        self.fixture.summary.update(
            {
                "status": "no_work",
                "run_completion_status": "no_work",
                "completed_cases": 0,
                "unsupported_cases": [self.fixture.case_id],
                "cases": [],
            }
        )
        self.fixture.index["runs"][0].update(  # type: ignore[index]
            {"status": "no_work", "completed_cases": 0}
        )
        self.fixture.write()

        report = audit_matrix(self.fixture.matrix_dir)

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["runs"][0]["derived_status"], "no_work")
        self.assertEqual(report["runs"][0]["recomputed_cases"], [])

    def test_startup_abort_is_terminal_without_unstarted_case_outcomes(self) -> None:
        run_error = {
            "stage": "server_start",
            "error_type": "RuntimeErrorWithContext",
            "error": "Server exited during startup: unsupported architecture",
        }
        self.fixture.events = [
            {
                "timestamp": self.fixture._timestamp(0),
                "event": "run_start",
                "completed_cases_at_resume": [],
            },
            {
                "timestamp": self.fixture._timestamp(1),
                "event": "run_aborted",
                **run_error,
            },
        ]
        self.fixture.summary.update(
            {
                "status": "aborted",
                "run_completion_status": None,
                "run_error": run_error,
                "completed_cases": 0,
                "cases": [],
            }
        )
        index_entry = self.fixture.index["runs"][0]  # type: ignore[index]
        index_entry.update(
            {
                "status": "failed",
                "error_type": run_error["error_type"],
                "error": run_error["error"],
            }
        )
        index_entry.pop("completed_cases")
        self.fixture.write()

        report = audit_matrix(self.fixture.matrix_dir)

        self.assertTrue(report["ok"], report["errors"])
        run = report["runs"][0]
        self.assertEqual(run["terminal_event"], "run_aborted")
        self.assertEqual(run["derived_status"], "aborted")
        self.assertNotIn("missing_case_outcome", run["error_codes"])
        self.assertNotIn("missing_run_complete", run["error_codes"])

    def test_matrix_startup_exception_remains_failed_and_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            model = SimpleNamespace(
                id="fixture-model",
                backend="vllm",
                support_status="spark_vllm_matrix",
                tasks=("chat",),
            )
            suite = SimpleNamespace(id="quick")
            availability = {
                model.id: SimpleNamespace(available=True),
            }

            def create_plan(*, results_root: Path, **kwargs: object) -> Path:
                run_dir = results_root / "fixture-run"
                run_dir.mkdir()
                return run_dir

            args = argparse.Namespace(
                models=Path("models.toml"),
                suite=Path("suite.toml"),
                results=results,
                backend=None,
                task=None,
                match="*",
                limit=None,
                plan_only=False,
                allow_download=False,
                fail_fast=False,
            )
            with (
                patch("sparkbench.load_models", return_value={model.id: model}),
                patch("sparkbench.load_suite", return_value=suite),
                patch("sparkbench._inventory"),
                patch(
                    "sparkbench.assess_model_availability",
                    return_value=availability,
                ),
                patch("sparkbench.create_plan", side_effect=create_plan),
                patch(
                    "sparkbench.execute_plan",
                    side_effect=RuntimeError("startup failed"),
                ),
                redirect_stdout(io.StringIO()),
            ):
                status = command_matrix(args)

            index_path = next((results / "matrices").glob("*/matrix.json"))
            index = json.loads(index_path.read_text())

        self.assertEqual(status, 1)
        self.assertEqual(index["runs"][0]["status"], "failed")
        self.assertEqual(index["runs"][0]["error_type"], "RuntimeError")

    def test_matrix_serializes_run_directory_relative_to_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            model = SimpleNamespace(
                id="fixture-model",
                backend="vllm",
                support_status="spark_vllm_matrix",
                tasks=("chat",),
            )
            suite = SimpleNamespace(id="quick")
            availability = {model.id: SimpleNamespace(available=True)}

            def create_plan(*, results_root: Path, **kwargs: object) -> Path:
                run_dir = results_root / "fixture-run"
                run_dir.mkdir()
                return run_dir

            args = argparse.Namespace(
                models=Path("models.toml"),
                suite=Path("suite.toml"),
                results=results,
                backend=None,
                task=None,
                match="*",
                limit=None,
                plan_only=True,
                allow_download=False,
                fail_fast=False,
            )
            with (
                patch("sparkbench.load_models", return_value={model.id: model}),
                patch("sparkbench.load_suite", return_value=suite),
                patch("sparkbench._inventory"),
                patch(
                    "sparkbench.assess_model_availability",
                    return_value=availability,
                ),
                patch("sparkbench.create_plan", side_effect=create_plan),
                redirect_stdout(io.StringIO()),
            ):
                status = command_matrix(args)

            index_path = next((results / "matrices").glob("*/matrix.json"))
            index = json.loads(index_path.read_text())

        self.assertEqual(status, 0)
        self.assertEqual(index["runs"][0]["run_dir"], "fixture-run")

    def test_audit_accepts_legacy_workspace_relative_run_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix_parent = Path(directory) / "results" / "matrices"
            matrix_parent.mkdir(parents=True)
            fixture = MatrixFixture(matrix_parent)
            legacy_run_dir = (
                Path("results")
                / "matrices"
                / fixture.matrix_dir.name
                / fixture.run_dir.name
            )
            fixture.index["runs"][0]["run_dir"] = str(legacy_run_dir)  # type: ignore[index]
            fixture.summary["run_dir"] = str(legacy_run_dir)
            fixture.write()

            report = audit_matrix(fixture.matrix_dir)

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["runs"][0]["run_dir"], str(fixture.run_dir.resolve()))

    def test_audit_rejects_run_directory_outside_matrix(self) -> None:
        outside_run_dir = self.fixture.matrix_dir.parent / "outside-run"
        outside_run_dir.mkdir()
        self.fixture.index["runs"][0]["run_dir"] = str(outside_run_dir)  # type: ignore[index]
        self.fixture.write()

        report = audit_matrix(self.fixture.matrix_dir)
        codes = {issue["code"] for issue in report["errors"]}

        self.assertFalse(report["ok"])
        self.assertIn("run_directory_outside_matrix", codes)

    def test_malformed_jsonl_record_and_torn_tail_are_structural_errors(self) -> None:
        with (self.fixture.run_dir / "events.jsonl").open("a") as stream:
            stream.write("not-json")

        report = audit_matrix(self.fixture.matrix_dir)
        codes = {issue["code"] for issue in report["errors"]}

        self.assertIn("invalid_jsonl_record", codes)
        self.assertIn("unterminated_events_jsonl", codes)

    def test_parallel_configuration_is_not_labeled_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = MatrixFixture(Path(temporary), max_num_seqs=8)

            report = audit_matrix(fixture.matrix_dir)

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["runs"][0]["max_num_seqs"], 8)
        self.assertEqual(report["runs"][0]["scheduling_label"], "parallel_configured")
        self.assertEqual(report["matrix"]["serialized_queue_runs"], [])

    def test_parser_exposes_audit_matrix_command(self) -> None:
        args = build_parser().parse_args(
            ["audit-matrix", str(self.fixture.matrix_dir)]
        )

        self.assertIs(args.function, command_audit_matrix)
        self.assertEqual(args.matrix_dir, self.fixture.matrix_dir)


if __name__ == "__main__":
    unittest.main()
