from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import bench.autoresearch_campaign as campaign_module
from bench.autoresearch_checkpoint import CheckpointError
import sparkbench


SHA_A = "a" * 64
SHA_B = "b" * 64
OID_A = "a" * 40


def _acknowledgement(private_marker: str) -> SimpleNamespace:
    return SimpleNamespace(
        campaign=SimpleNamespace(campaign_id="synthetic-autoresearch"),
        completion=SimpleNamespace(sequence=2, pair_kind="screen"),
        evidence=SimpleNamespace(index_sha256=SHA_A),
        repository=SimpleNamespace(head_commit=OID_A),
        to_mapping=lambda: {
            "integrity_hash": SHA_B,
            "private_raw_path": private_marker,
            "raw_command": "synthetic-private-command",
        },
    )


class AutoresearchCheckpointCliTests(unittest.TestCase):
    def _parse_checkpoint(self, campaign_dir: str = "relative/campaign"):
        return sparkbench.build_parser().parse_args(
            ["autoresearch-checkpoint", campaign_dir]
        )

    def test_parser_dispatches_checkpoint_with_a_path(self) -> None:
        args = self._parse_checkpoint()

        self.assertEqual(args.campaign_dir, Path("relative/campaign"))
        self.assertIs(args.function, sparkbench.command_autoresearch_checkpoint)

    def test_success_dispatch_prints_only_sanitized_scalar_fields(self) -> None:
        private_marker = "/private/results/raw-campaign-marker"
        acknowledgement = _acknowledgement(private_marker)
        args = self._parse_checkpoint(private_marker)
        output = io.StringIO()

        with (
            patch.object(
                campaign_module,
                "acknowledge_campaign_checkpoint",
                return_value=acknowledgement,
                create=True,
            ) as acknowledge,
            redirect_stdout(output),
        ):
            exit_code = args.function(args)

        self.assertEqual(exit_code, 0)
        acknowledge.assert_called_once_with(Path(private_marker), sparkbench.WORKSPACE)
        payload = json.loads(output.getvalue())
        self.assertEqual(
            payload,
            {
                "acknowledgement_integrity_sha256": SHA_B,
                "campaign_id": "synthetic-autoresearch",
                "checkpoint_sequence": 2,
                "evidence_index_sha256": SHA_A,
                "pair_kind": "screen",
                "repository_commit": OID_A,
                "status": "checkpoint_acknowledged",
            },
        )
        serialized = output.getvalue()
        self.assertNotIn(private_marker, serialized)
        self.assertNotIn("synthetic-private-command", serialized)

    def test_readiness_errors_are_sanitized_and_exit_one(self) -> None:
        private_marker = "/private/results/not-ready-marker"
        args = self._parse_checkpoint()

        for code in sorted(sparkbench.AUTORESEARCH_CHECKPOINT_READINESS_CODES):
            with self.subTest(code=code):
                output = io.StringIO()
                with (
                    patch.object(
                        campaign_module,
                        "acknowledge_campaign_checkpoint",
                        side_effect=CheckpointError(code, private_marker),
                        create=True,
                    ),
                    redirect_stdout(output),
                ):
                    exit_code = args.function(args)

                self.assertEqual(exit_code, 1)
                self.assertEqual(
                    json.loads(output.getvalue()),
                    {"reason": code, "status": "checkpoint_required"},
                )
                self.assertNotIn(private_marker, output.getvalue())

    def test_structural_and_journal_errors_re_raise(self) -> None:
        args = self._parse_checkpoint()

        for code in (
            "checkpoint_state_invalid",
            "journal_prefix_changed",
            "journal_prefix_invalid",
            "synthetic_campaign_corruption",
        ):
            with self.subTest(code=code):
                failure = CheckpointError(code, "synthetic structural failure")
                with (
                    patch.object(
                        campaign_module,
                        "acknowledge_campaign_checkpoint",
                        side_effect=failure,
                        create=True,
                    ),
                    self.assertRaises(CheckpointError) as raised,
                ):
                    args.function(args)
                self.assertIs(raised.exception, failure)

    def test_main_maps_structural_checkpoint_error_to_exit_two(self) -> None:
        error_output = io.StringIO()
        with (
            patch.object(
                campaign_module,
                "acknowledge_campaign_checkpoint",
                side_effect=CheckpointError(
                    "journal_prefix_invalid", "campaign journal is malformed"
                ),
                create=True,
            ),
            patch.object(
                sys,
                "argv",
                ["sparkbench.py", "autoresearch-checkpoint", "relative/campaign"],
            ),
            redirect_stderr(error_output),
            self.assertRaises(SystemExit) as exited,
        ):
            sparkbench.main()

        self.assertEqual(exited.exception.code, 2)
        self.assertIn("campaign journal is malformed", error_output.getvalue())

    def test_autoresearch_run_uses_distinct_checkpoint_exit_status(self) -> None:
        args = sparkbench.build_parser().parse_args(
            ["autoresearch-run", "relative/campaign"]
        )
        expected = {
            "active": 0,
            "complete": 0,
            "checkpoint_required": 3,
            "blocked_environment": 1,
            "expired": 1,
            "terminated": 1,
        }

        for status, exit_code in expected.items():
            with self.subTest(status=status):
                output = io.StringIO()
                with (
                    patch("sparkbench.run_campaign", return_value={"status": status}),
                    redirect_stdout(output),
                ):
                    observed = args.function(args)
                self.assertEqual(observed, exit_code)
                self.assertEqual(json.loads(output.getvalue()), {"status": status})


if __name__ == "__main__":
    unittest.main()
