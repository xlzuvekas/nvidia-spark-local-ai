from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from bench.annotations import (
    append_measurement_annotation,
    measurement_annotations,
)
from bench.report import summarize_run
from sparkbench import build_parser, command_annotate


class MeasurementAnnotationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_dir = Path(temporary.name) / "run"
        self.run_dir.mkdir()
        self.case_id = "decode-128--abc123"
        (self.run_dir / "plan.json").write_text(
            json.dumps(
                {
                    "model": {"id": "fixture"},
                    "suite": {
                        "id": "quick",
                        "cases": [
                            {"id": "decode-128", "case_id": self.case_id}
                        ],
                    },
                }
            )
        )
        (self.run_dir / "events.jsonl").write_text(
            json.dumps({"timestamp": "before", "event": "run_complete"}) + "\n"
        )

    def test_case_annotation_is_append_only_and_machine_readable(self) -> None:
        events_path = self.run_dir / "events.jsonl"
        before = events_path.read_bytes()

        annotation = append_measurement_annotation(
            self.run_dir,
            scope="case",
            case_id=self.case_id,
            reason="host metrics request overlapped a measured repetition",
            evidence=("server/server.log:96", "20:09:58 MST"),
        )

        after = events_path.read_bytes()
        self.assertTrue(after.startswith(before))
        self.assertGreater(len(after), len(before))
        self.assertEqual(annotation["event"], "measurement_annotation")
        self.assertEqual(annotation["schema_version"], 1)
        self.assertEqual(annotation["scope"], "case")
        self.assertEqual(annotation["case_id"], self.case_id)
        self.assertFalse(annotation["measurement_valid"])
        self.assertEqual(
            annotation["evidence"],
            ["server/server.log:96", "20:09:58 MST"],
        )
        normalized = measurement_annotations(
            [json.loads(line) for line in after.decode().splitlines()]
        )
        self.assertEqual(normalized[0]["case_id"], self.case_id)
        self.assertFalse(normalized[0]["measurement_valid"])

    def test_startup_annotation_rejects_case_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not include"):
            append_measurement_annotation(
                self.run_dir,
                scope="startup",
                case_id=self.case_id,
                reason="probe overlapped startup",
            )

    def test_case_annotation_requires_known_frozen_case_and_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "require --case-id"):
            append_measurement_annotation(
                self.run_dir,
                scope="case",
                reason="probe overlapped measurement",
            )
        with self.assertRaisesRegex(ValueError, "not present in the frozen plan"):
            append_measurement_annotation(
                self.run_dir,
                scope="case",
                case_id="unknown",
                reason="probe overlapped measurement",
            )
        with self.assertRaisesRegex(ValueError, "reason must be non-empty"):
            append_measurement_annotation(
                self.run_dir,
                scope="startup",
                reason="   ",
            )

    def test_annotation_requires_existing_plan_and_journal(self) -> None:
        missing = self.run_dir.parent / "missing"
        missing.mkdir()
        with self.assertRaisesRegex(ValueError, "frozen plan is missing"):
            append_measurement_annotation(
                missing,
                scope="startup",
                reason="probe overlapped startup",
            )

        (self.run_dir / "events.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "event journal is missing"):
            append_measurement_annotation(
                self.run_dir,
                scope="startup",
                reason="probe overlapped startup",
            )

    def test_report_retains_metrics_and_marks_only_affected_measurements(self) -> None:
        (self.run_dir / "events.jsonl").write_text(
            "".join(
                json.dumps(event) + "\n"
                for event in (
                    {"event": "run_start"},
                    {
                        "event": "request_complete",
                        "case_id": self.case_id,
                        "attempt_id": "attempt",
                        "kind": "decode",
                        "result": {
                            "ttft_s": 0.1,
                            "elapsed_s": 3.0,
                            "decode_tps": 42.0,
                            "prompt_tokens": 32,
                            "completion_tokens": 128,
                        },
                    },
                    {
                        "event": "case_complete",
                        "case_id": self.case_id,
                        "attempt_id": "attempt",
                        "kind": "decode",
                        "elapsed_s": 4.0,
                        "validation_passed": True,
                    },
                    {"event": "run_complete", "status": "completed"},
                )
            )
        )
        append_measurement_annotation(
            self.run_dir,
            scope="startup",
            reason="metadata probe overlapped startup",
            evidence=("probe ended before server_ready",),
        )

        startup_only = summarize_run(self.run_dir)

        self.assertFalse(startup_only["startup_measurement_valid"])
        self.assertEqual(len(startup_only["startup_measurement_annotations"]), 1)
        self.assertTrue(startup_only["cases"][0]["measurement_valid"])
        self.assertEqual(startup_only["measurement_invalid_cases"], [])

        append_measurement_annotation(
            self.run_dir,
            scope="case",
            case_id=self.case_id,
            reason="metrics scrape overlapped a measured repetition",
        )
        summary = summarize_run(self.run_dir)
        row = summary["cases"][0]

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(len(summary["measurement_annotations"]), 2)
        self.assertEqual(summary["measurement_invalid_cases"], [self.case_id])
        self.assertFalse(row["measurement_valid"])
        self.assertEqual(len(row["measurement_annotations"]), 1)
        self.assertEqual(row["completion_tokens"], 128)
        self.assertEqual(row["aggregate_output_tps"], 32.0)

    def test_cli_parser_and_command_append_then_regenerate_summary(self) -> None:
        args = build_parser().parse_args(
            [
                "annotate",
                str(self.run_dir),
                "--scope",
                "startup",
                "--reason",
                "inventory probe overlapped startup",
                "--evidence",
                "probe interval ended before server_ready",
            ]
        )
        self.assertIs(args.function, command_annotate)

        output = io.StringIO()
        with redirect_stdout(output):
            status = args.function(args)
        payload = json.loads(output.getvalue())

        self.assertEqual(status, 0)
        self.assertFalse(payload["startup_measurement_valid"])
        self.assertEqual(payload["measurement_invalid_cases"], [])
        self.assertTrue((self.run_dir / "summary.json").is_file())
        annotations = measurement_annotations(
            [
                json.loads(line)
                for line in (self.run_dir / "events.jsonl").read_text().splitlines()
            ]
        )
        self.assertEqual(len(annotations), 1)


if __name__ == "__main__":
    unittest.main()
