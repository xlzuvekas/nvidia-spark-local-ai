"""Offline contract tests for the registered current-runtime autoresearch loop."""

from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import bench.autoresearch_v2 as v2
from bench.journal import content_hash, write_json
from sparkbench import (
    DEFAULT_AUTORESEARCH_V2_CACHE_POLICY_CAMPAIGN,
    build_parser,
    command_autoresearch_v2_plan,
    command_autoresearch_v2_run,
)


_PAIR_BINDING = "sha256:" + "b" * 64
_INTEGRITY = "c" * 64
_CUTOFF = "2026-08-29T10:00:00-07:00"
_NOW = datetime(2026, 8, 29, 3, 0, tzinfo=timezone.utc)


def _child_campaign(root: Path) -> Path:
    child = root / "cache-policy-campaigns" / "20260829T030000Z-cache-policy"
    child.mkdir(parents=True)
    write_json(
        child / "campaign.json",
        {
            "campaign_id": v2.SM121_CACHE_PERFORMANCE_CAMPAIGN_ID,
            "integrity_hash": _INTEGRITY,
            "pair_binding": {"pair_binding_sha256": _PAIR_BINDING},
            "prerequisite_bundle_sha256s": list(
                v2.SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S
            ),
        },
    )
    return child


def _round(root: Path) -> tuple[Path, Path]:
    child = _child_campaign(root)
    round_dir = root / v2.AUTORESEARCH_V2_RESULT_ROOT / "20260829t030000z-round"
    round_dir.mkdir(parents=True)
    payload: dict[str, object] = {
        "schema_version": v2.AUTORESEARCH_V2_SCHEMA_VERSION,
        "campaign_id": v2.AUTORESEARCH_V2_CAMPAIGN_ID,
        "created_at": "2026-08-29T10:00:00.000+00:00",
        "cutoff": _CUTOFF,
        "execution_mode": v2.AUTORESEARCH_V2_EXECUTION_MODE,
        "definition_sha256": v2._expected_definition_sha256(),
        "runner": v2.AUTORESEARCH_V2_RUNNER_CACHE_POLICY,
        "axis": v2.AUTORESEARCH_V2_AXIS,
        "control_profile_id": v2.SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID,
        "candidate_profile_id": v2.SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID,
        "control_arm": "A",
        "candidate_arm": "B",
        "suite_id": v2.SM121_CACHE_PERFORMANCE_SUITE_ID,
        "cell_timeout_s": v2.SM121_CACHE_PERFORMANCE_CELL_TIMEOUT_S,
        "required_lifetimes": v2.AUTORESEARCH_V2_REQUIRED_LIFETIMES,
        "primary": v2.AUTORESEARCH_V2_PRIMARY,
        "promotion_ratio": v2.AUTORESEARCH_V2_PROMOTION_RATIO,
        "full_wall_guardrail_ratio": v2.AUTORESEARCH_V2_FULL_WALL_GUARDRAIL_RATIO,
        "min_cutoff_remaining_s": v2.AUTORESEARCH_V2_MIN_CUTOFF_REMAINING_S,
        "child_campaign_id": v2.SM121_CACHE_PERFORMANCE_CAMPAIGN_ID,
        "child_campaign_directory": child.name,
        "child_campaign_integrity_hash": _INTEGRITY,
        "child_pair_binding_sha256": _PAIR_BINDING,
        "child_prerequisite_bundle_sha256s": list(
            v2.SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S
        ),
    }
    payload["integrity_hash"] = content_hash(payload, 64)
    write_json(round_dir / "round.json", payload)
    return round_dir, child


def _child_summary(*, status: str = "complete", decision: str = "retain_a") -> dict[str, object]:
    return {
        "status": status,
        "decision": decision,
        "completed_arms": 4 if status == "complete" else 1,
        "score": {
            "status": status,
            "decision": decision,
            "a_later_wall_s": 2.0 if status == "complete" else None,
            "b_later_wall_s": 40.0 if status == "complete" else None,
            "a_full_wall_s": 30.0 if status == "complete" else None,
            "b_full_wall_s": 80.0 if status == "complete" else None,
            "winner_later_wall_ratio": 0.05 if status == "complete" else None,
            "winner_full_wall_ratio": 0.4 if status == "complete" else None,
        },
    }


class AutoresearchV2DefinitionTests(unittest.TestCase):
    def test_preview_is_fixed_to_the_current_cache_runner(self) -> None:
        preview = v2.preview_autoresearch_v2(
            DEFAULT_AUTORESEARCH_V2_CACHE_POLICY_CAMPAIGN
        )
        self.assertEqual(v2.AUTORESEARCH_V2_RUNNER_CACHE_POLICY, preview["runner"])
        self.assertEqual("A", preview["control_arm"])
        self.assertEqual("B", preview["candidate_arm"])
        self.assertEqual(
            v2.SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID,
            preview["control_profile_id"],
        )
        self.assertEqual(
            v2.SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID,
            preview["candidate_profile_id"],
        )

    def test_definition_rejects_one_axis_or_timeout_rewrite(self) -> None:
        source = DEFAULT_AUTORESEARCH_V2_CACHE_POLICY_CAMPAIGN.read_text()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = root / "manifests"
            campaigns = manifests / "campaigns"
            suites = manifests / "suites"
            campaigns.mkdir(parents=True)
            suites.mkdir()
            (manifests / "models.toml").write_text("# fixture\n")
            (suites / "suite.toml").write_text("# fixture\n")
            path = campaigns / "campaign.toml"
            path.write_text(
                source.replace("../models.toml", "../models.toml")
                .replace(
                    "../suites/qwen38_flash_next_sm121_triton_storage_cache_policy_performance_v1.toml",
                    "../suites/suite.toml",
                )
                .replace("cell_timeout_s = 1200", "cell_timeout_s = 900")
            )
            with self.assertRaisesRegex(v2.AutoresearchV2Error, "cell_timeout_s"):
                v2.load_autoresearch_v2_definition(path)

    def test_freeze_binds_child_before_any_execution(self) -> None:
        definition = DEFAULT_AUTORESEARCH_V2_CACHE_POLICY_CAMPAIGN
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = _child_campaign(root)
            captured: dict[str, object] = {}

            def fake_create(**kwargs: object) -> Path:
                captured.update(kwargs)
                return child

            models = {
                v2.SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID: object(),
                v2.SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID: object(),
            }
            suite = type("Suite", (), {"id": v2.SM121_CACHE_PERFORMANCE_SUITE_ID})()
            with (
                patch("bench.autoresearch_v2.create_sm121_cache_performance_campaign", fake_create),
                patch("bench.autoresearch_v2.load_models", return_value=models),
                patch("bench.autoresearch_v2.load_suite", return_value=suite),
            ):
                round_dir = v2.freeze_autoresearch_v2(
                    definition,
                    results_root=root,
                    evidence_root=root / "evidence",
                    cutoff=_CUTOFF,
                    now=_NOW,
                )
            self.assertIs(
                models[v2.SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID],
                captured["cache_on_model"],
            )
            self.assertIs(
                models[v2.SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID],
                captured["cache_off_model"],
            )
            payload = v2._load_round(round_dir)
            self.assertEqual(child.name, payload["child_campaign_directory"])
            self.assertEqual(_PAIR_BINDING, payload["child_pair_binding_sha256"])
            self.assertFalse((round_dir / "events.jsonl").exists())

    def test_freeze_rejects_insufficient_explicit_cutoff(self) -> None:
        with self.assertRaisesRegex(v2.AutoresearchV2Error, "insufficient time"):
            v2.freeze_autoresearch_v2(
                DEFAULT_AUTORESEARCH_V2_CACHE_POLICY_CAMPAIGN,
                results_root=Path("results"),
                evidence_root=Path("evidence"),
                cutoff="2026-08-29T05:00:00+00:00",
                now=_NOW,
            )


class AutoresearchV2RoundTests(unittest.TestCase):
    def test_child_binding_rejects_prerequisite_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            round_dir, child = _round(root)
            campaign = json.loads((child / "campaign.json").read_text())
            campaign["prerequisite_bundle_sha256s"] = []
            write_json(child / "campaign.json", campaign)
            payload = v2._load_round(round_dir)
            with self.assertRaisesRegex(v2.AutoresearchV2Error, "binding changed"):
                v2._validate_child_binding(round_dir, payload)

    def test_decision_translation_is_candidate_centric_and_fail_closed(self) -> None:
        audit = {"ok": True}
        self.assertEqual(
            ("complete", "retain"),
            v2._decision_from_child(_child_summary(decision="retain_b"), audit),
        )
        self.assertEqual(
            ("complete", "reject"),
            v2._decision_from_child(_child_summary(decision="retain_a"), audit),
        )
        for decision in ("no_retention", "guardrail_reject"):
            self.assertEqual(
                ("complete", "inconclusive"),
                v2._decision_from_child(_child_summary(decision=decision), audit),
            )
        self.assertEqual(
            ("partial", "inconclusive"),
            v2._decision_from_child(
                _child_summary(status="partial", decision="not_evaluated"), audit
            ),
        )
        self.assertEqual(
            ("partial", "inconclusive"),
            v2._decision_from_child(_child_summary(), {"ok": False}),
        )

    def test_run_is_non_resumable_and_emits_only_scalar_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            round_dir, _child = _round(root)
            source = _child_summary(decision="retain_a")
            execute = Mock(return_value=source)
            with (
                patch("bench.autoresearch_v2.execute_sm121_cache_performance_campaign", execute),
                patch(
                    "bench.autoresearch_v2.audit_sm121_cache_performance_campaign",
                    return_value={"ok": True},
                ),
            ):
                summary = v2.run_autoresearch_v2(
                    round_dir,
                    workspace=root,
                    evidence_root=root / "evidence",
                    now=_NOW,
                )
                self.assertEqual("complete", summary["status"])
                self.assertEqual("reject", summary["decision"])
                execute.assert_called_once()
                with self.assertRaisesRegex(v2.AutoresearchV2Error, "terminal"):
                    v2.run_autoresearch_v2(
                        round_dir,
                        workspace=root,
                        evidence_root=root / "evidence",
                        now=_NOW,
                    )
            public = json.loads((round_dir / "summary.json").read_text())
            self.assertEqual(v2._SUMMARY_FIELDS, frozenset(public))
            rendered = json.dumps(public, sort_keys=True)
            for sentinel in ("prompt", "completion", "token_ids", "request_id", "path"):
                self.assertNotIn(sentinel, rendered)

    def test_executor_failure_is_terminal_scalar_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            round_dir, _child = _round(root)
            cause = RuntimeError("synthetic child request details")
            with (
                patch(
                    "bench.autoresearch_v2.execute_sm121_cache_performance_campaign",
                    side_effect=cause,
                ),
                patch("bench.autoresearch_v2._validate_failure_child_source"),
            ):
                with self.assertRaises(v2.AutoresearchV2ExecutionFailure) as caught:
                    v2.run_autoresearch_v2(
                        round_dir,
                        workspace=root,
                        evidence_root=root / "evidence",
                        now=_NOW,
                    )
            self.assertEqual("child_execution", caught.exception.stage)
            self.assertIs(cause, caught.exception.__cause__)
            payload = v2._load_round(round_dir)
            expected = v2._failure_summary_payload(
                payload, stage="child_execution"
            )
            self.assertEqual(expected, caught.exception.summary)
            saved = json.loads((round_dir / "summary.json").read_text())
            self.assertEqual(expected, saved)
            rendered = json.dumps(saved, sort_keys=True)
            self.assertNotIn("synthetic child request details", rendered)
            self.assertNotIn("request", rendered)
            self.assertEqual(
                "failed", v2._validate_events(round_dir, payload, terminal=True)
            )
            with self.assertRaisesRegex(v2.AutoresearchV2Error, "failure child"):
                v2.summarize_autoresearch_v2(
                    round_dir, evidence_root=root / "evidence"
                )
            with self.assertRaisesRegex(v2.AutoresearchV2Error, "terminal"):
                v2.run_autoresearch_v2(
                    round_dir,
                    workspace=root,
                    evidence_root=root / "evidence",
                    now=_NOW,
                )

    def test_signal_like_interrupt_does_not_synthesize_a_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            round_dir, _child = _round(root)
            with patch(
                "bench.autoresearch_v2.execute_sm121_cache_performance_campaign",
                side_effect=KeyboardInterrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    v2.run_autoresearch_v2(
                        round_dir,
                        workspace=root,
                        evidence_root=root / "evidence",
                        now=_NOW,
                    )
            self.assertFalse((round_dir / "summary.json").exists())
            self.assertEqual(
                ["autoresearch_v2_round_started"],
                [
                    event["event"]
                    for event in v2.Journal(round_dir / "events.jsonl").strict_events()
                ],
            )
            with self.assertRaisesRegex(v2.AutoresearchV2Error, "cannot be resumed"):
                v2.run_autoresearch_v2(
                    round_dir,
                    workspace=root,
                    evidence_root=root / "evidence",
                    now=_NOW,
                )

    def test_failure_receipt_persistence_failure_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            round_dir, _child = _round(root)
            with (
                patch(
                    "bench.autoresearch_v2.execute_sm121_cache_performance_campaign",
                    side_effect=RuntimeError("synthetic child failure"),
                ),
                patch("bench.autoresearch_v2._validate_failure_child_source"),
                patch(
                    "bench.autoresearch_v2.write_json",
                    side_effect=OSError("synthetic receipt write failure"),
                ),
            ):
                with self.assertRaisesRegex(
                    v2.AutoresearchV2Error, "could not be persisted"
                ) as caught:
                    v2.run_autoresearch_v2(
                        round_dir,
                        workspace=root,
                        evidence_root=root / "evidence",
                        now=_NOW,
                    )
            self.assertIsInstance(caught.exception.__cause__, OSError)
            self.assertFalse((round_dir / "summary.json").exists())
            self.assertEqual(
                ["autoresearch_v2_round_started", "autoresearch_v2_round_failed"],
                [
                    event["event"]
                    for event in v2.Journal(round_dir / "events.jsonl").strict_events()
                ],
            )

    def test_audit_failure_is_terminal_scalar_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            round_dir, _child = _round(root)
            cause = OSError("synthetic audit payload")
            with (
                patch(
                    "bench.autoresearch_v2.execute_sm121_cache_performance_campaign",
                    return_value=_child_summary(),
                ),
                patch(
                    "bench.autoresearch_v2.audit_sm121_cache_performance_campaign",
                    side_effect=cause,
                ),
                patch("bench.autoresearch_v2._validate_failure_child_source"),
            ):
                with self.assertRaises(v2.AutoresearchV2ExecutionFailure) as caught:
                    v2.run_autoresearch_v2(
                        round_dir,
                        workspace=root,
                        evidence_root=root / "evidence",
                        now=_NOW,
                    )
            self.assertEqual("child_audit", caught.exception.stage)
            self.assertIs(cause, caught.exception.__cause__)
            with self.assertRaisesRegex(v2.AutoresearchV2Error, "failure child"):
                v2.summarize_autoresearch_v2(
                    round_dir, evidence_root=root / "evidence"
                )

    def test_projection_failure_is_terminal_scalar_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            round_dir, _child = _round(root)
            cause = RuntimeError("synthetic projection payload")
            with (
                patch(
                    "bench.autoresearch_v2.execute_sm121_cache_performance_campaign",
                    return_value=_child_summary(),
                ),
                patch(
                    "bench.autoresearch_v2.audit_sm121_cache_performance_campaign",
                    return_value={"ok": True},
                ),
                patch("bench.autoresearch_v2._summary_payload", side_effect=cause),
                patch("bench.autoresearch_v2._validate_failure_child_source"),
            ):
                with self.assertRaises(v2.AutoresearchV2ExecutionFailure) as caught:
                    v2.run_autoresearch_v2(
                        round_dir,
                        workspace=root,
                        evidence_root=root / "evidence",
                        now=_NOW,
                    )
            self.assertEqual("projection", caught.exception.stage)
            self.assertIs(cause, caught.exception.__cause__)

    def test_failure_stage_is_bound_to_its_terminal_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            round_dir, _child = _round(root)
            with (
                patch(
                    "bench.autoresearch_v2.execute_sm121_cache_performance_campaign",
                    side_effect=RuntimeError("synthetic child request details"),
                ),
                patch("bench.autoresearch_v2._validate_failure_child_source"),
            ):
                with self.assertRaises(v2.AutoresearchV2ExecutionFailure):
                    v2.run_autoresearch_v2(
                        round_dir,
                        workspace=root,
                        evidence_root=root / "evidence",
                        now=_NOW,
                    )
            saved = json.loads((round_dir / "summary.json").read_text())
            saved["failure_stage"] = "child_audit"
            saved["integrity_hash"] = content_hash(
                {key: value for key, value in saved.items() if key != "integrity_hash"},
                64,
            )
            write_json(round_dir / "summary.json", saved)
            with self.assertRaisesRegex(
                v2.AutoresearchV2Error, "failure stage changed"
            ):
                v2.summarize_autoresearch_v2(
                    round_dir, evidence_root=root / "evidence"
                )

    def test_failure_started_event_is_bound_to_its_round(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            round_dir, _child = _round(root)
            with (
                patch(
                    "bench.autoresearch_v2.execute_sm121_cache_performance_campaign",
                    side_effect=RuntimeError("synthetic child request details"),
                ),
                patch("bench.autoresearch_v2._validate_failure_child_source"),
            ):
                with self.assertRaises(v2.AutoresearchV2ExecutionFailure):
                    v2.run_autoresearch_v2(
                        round_dir,
                        workspace=root,
                        evidence_root=root / "evidence",
                        now=_NOW,
                    )
            journal_path = round_dir / "events.jsonl"
            events = v2.Journal(journal_path).strict_events()
            events[0]["definition_sha256"] = "0" * 64
            journal_path.write_text(
                "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                v2.AutoresearchV2Error, "started event definition binding changed"
            ):
                v2.summarize_autoresearch_v2(
                    round_dir, evidence_root=root / "evidence"
                )

    def test_summarize_recomputes_the_child_projection_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            round_dir, child = _round(root)
            source = _child_summary(decision="retain_b")
            write_json(child / "summary.json", source)
            payload = v2._load_round(round_dir)
            summary = v2._summary_payload(payload, child_summary=source, audit={"ok": True})
            write_json(round_dir / "summary.json", summary)
            journal = v2.Journal(round_dir / "events.jsonl")
            journal.append(
                {
                    "event": "autoresearch_v2_round_started",
                    "campaign_id": payload["campaign_id"],
                    "execution_mode": payload["execution_mode"],
                    "definition_sha256": payload["definition_sha256"],
                    "child_pair_binding_sha256": payload["child_pair_binding_sha256"],
                }
            )
            journal.append(
                {
                    "event": "autoresearch_v2_round_scored",
                    "campaign_id": payload["campaign_id"],
                    "child_pair_binding_sha256": payload["child_pair_binding_sha256"],
                    "status": summary["status"],
                    "decision": summary["decision"],
                    "child_status": summary["child_status"],
                    "child_decision": summary["child_decision"],
                    "audit_ok": summary["audit_ok"],
                    "completed_arms": summary["completed_arms"],
                }
            )
            journal.append(
                {
                    "event": "autoresearch_v2_round_complete",
                    "campaign_id": payload["campaign_id"],
                    "status": summary["status"],
                    "decision": summary["decision"],
                }
            )
            with patch(
                "bench.autoresearch_v2.audit_sm121_cache_performance_campaign",
                return_value={"ok": True},
            ):
                observed = v2.summarize_autoresearch_v2(
                    round_dir, evidence_root=root / "evidence"
                )
            self.assertEqual("retain", observed["decision"])


class AutoresearchV2CliTests(unittest.TestCase):
    def test_cli_has_separate_non_resumable_commands(self) -> None:
        parser = build_parser()
        plan = parser.parse_args(["autoresearch-v2-plan", "--dry-run"])
        self.assertEqual(command_autoresearch_v2_plan, plan.function)
        self.assertEqual(DEFAULT_AUTORESEARCH_V2_CACHE_POLICY_CAMPAIGN, plan.campaign)
        self.assertFalse(hasattr(plan, "allow_download"))
        run = parser.parse_args(["autoresearch-v2-run", "synthetic-round"])
        self.assertEqual(command_autoresearch_v2_run, run.function)
        self.assertEqual(Path("synthetic-round"), run.round_dir)

    def test_cli_failure_prints_only_the_scalar_receipt(self) -> None:
        args = build_parser().parse_args(["autoresearch-v2-run", "synthetic-round"])
        summary = {
            "schema_version": v2.AUTORESEARCH_V2_SCHEMA_VERSION,
            "campaign_id": v2.AUTORESEARCH_V2_CAMPAIGN_ID,
            "execution_mode": v2.AUTORESEARCH_V2_EXECUTION_MODE,
            "child_campaign_id": v2.SM121_CACHE_PERFORMANCE_CAMPAIGN_ID,
            "child_pair_binding_sha256": _PAIR_BINDING,
            "status": "partial",
            "decision": "inconclusive",
            "failure_stage": "child_execution",
            "integrity_hash": "d" * 64,
        }
        failure = v2.AutoresearchV2ExecutionFailure(
            stage="child_execution", summary=summary
        )
        output = io.StringIO()
        with (
            patch("sparkbench.run_autoresearch_v2", side_effect=failure),
            redirect_stdout(output),
        ):
            self.assertEqual(1, command_autoresearch_v2_run(args))
        self.assertEqual(summary, json.loads(output.getvalue()))
        self.assertNotIn("synthetic-round", output.getvalue())
