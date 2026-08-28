from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import bench.autoresearch_campaign as campaign_module
from bench.autoresearch import pair_order
from bench.autoresearch_campaign import (
    CampaignPlanningError,
    CellProjectionError,
    acknowledge_campaign_checkpoint,
    load_frozen_campaign,
    run_campaign,
)
from bench.autoresearch_checkpoint import CheckpointError, CheckpointGate
from bench.evidence import _autoresearch_published_run_id
from bench.journal import Journal, content_hash
from tests.test_autoresearch_campaign import (
    ROOT,
    _admission_meminfo,
    _freeze_campaign_fixture,
)
from tests.test_autoresearch_replay_hardening import (
    _Clock,
    _CompleteCellHarness,
    _events,
)


def _ready_gate(_campaign: object, _workspace: Path) -> CheckpointGate:
    return CheckpointGate("ready", None, 0)


def _append_candidate_started(campaign_dir: Path) -> None:
    campaign = load_frozen_campaign(campaign_dir)
    proposal = campaign.proposals[0]
    campaign_module._append_frozen_transition(
        Journal(campaign_dir / "events.jsonl"),
        campaign,
        {
            "event": "autoresearch_candidate_started",
            "candidate_id": proposal.candidate_id,
            "axis": proposal.axis,
            "delta_digest": proposal.delta.digest,
        },
    )


def _append_pair_started(campaign_dir: Path) -> None:
    campaign = load_frozen_campaign(campaign_dir)
    state = campaign_module._replay_frozen_campaign(
        campaign,
        campaign_module._controller_events(Journal(campaign_dir / "events.jsonl")),
    )
    assert state.candidate_id is not None
    campaign_module._append_frozen_transition(
        Journal(campaign_dir / "events.jsonl"),
        campaign,
        {
            "event": "autoresearch_pair_started",
            "candidate_id": state.candidate_id,
            "pair_index": state.next_pair_index,
            "order": list(pair_order(state.next_pair_index)),
        },
    )


def _make_failed_calibration(campaign_dir: Path) -> tuple[_Clock, _CompleteCellHarness]:
    campaign = load_frozen_campaign(campaign_dir)
    cells = campaign.cells_for(candidate_id="control", stage="calibration")
    clock = _Clock()
    harness = _CompleteCellHarness(clock)
    harness(cells["control_a"])
    harness(cells["control_b"])
    plan = json.loads((cells["control_b"].run_dir / "plan.json").read_text())
    case_id = next(
        case["case_id"]
        for case in plan["suite"]["cases"]
        if case["id"] == "agentic-select-and-call"
    )
    summary_path = cells["control_b"].run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    case = next(item for item in summary["cases"] if item["case_id"] == case_id)
    case["median_agentic_task_wall_s"] = float(
        case["median_agentic_task_wall_s"]
    ) * 2.0
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
    return clock, harness


class AutoresearchCheckpointIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        recovery = patch.object(
            campaign_module, "_recover_cell", return_value="already_absent"
        )
        recovery.start()
        self.addCleanup(recovery.stop)

    def test_created_at_is_exact_timezone_aware_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            root = Path(directory)
            campaign_dir = _freeze_campaign_fixture(root)
            frozen_path = campaign_dir / "campaign.json"
            frozen = json.loads(frozen_path.read_text())
            exact = frozen["created_at"]
            self.assertEqual(load_frozen_campaign(campaign_dir).created_at, exact)

            for invalid in ("2026-08-28T00:00:00", "2" * 65):
                with self.subTest(invalid=invalid):
                    changed = dict(frozen)
                    changed["created_at"] = invalid
                    changed.pop("integrity_hash")
                    changed["integrity_hash"] = content_hash(changed, 64)
                    frozen_path.write_text(json.dumps(changed, sort_keys=True) + "\n")
                    with self.assertRaisesRegex(
                        CampaignPlanningError, "creation time"
                    ):
                        load_frozen_campaign(campaign_dir)

    def test_calibration_completion_matches_exporter_public_ids(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=harness,
                    checkpoint_gate_reader=_ready_gate,
                )
            campaign = load_frozen_campaign(campaign_dir)
            completion = campaign_module._latest_checkpoint_completion(campaign)
            cells = campaign.cells_for(candidate_id="control", stage="calibration")
            ordered = (cells["control_a"], cells["control_b"])

        self.assertEqual(completion.sequence, 1)
        self.assertEqual(completion.pair_kind, "calibration")
        self.assertEqual(
            completion.ordered_evidence_run_ids,
            tuple(
                _autoresearch_published_run_id(
                    campaign_id=campaign.campaign_id,
                    cell_id=cell.cell_id,
                    ordinal=cell.ordinal,
                    created_at=campaign.created_at,
                )
                for cell in ordered
            ),
        )
        self.assertEqual(
            completion.cell_plan_integrity_sha256s,
            tuple(cell.plan_integrity_hash for cell in ordered),
        )

    def test_latest_completion_survives_decisions_and_reverse_confirmation(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            gate_calls: list[str] = []

            def ready(campaign: object, workspace: Path) -> CheckpointGate:
                gate_calls.append(str(getattr(campaign, "campaign_id")))
                self.assertEqual(workspace, ROOT)
                return CheckpointGate("ready", None, 0)

            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=harness,
                    checkpoint_gate_reader=ready,
                )
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=harness,
                    checkpoint_gate_reader=ready,
                )
                screen = campaign_module._latest_checkpoint_completion(
                    load_frozen_campaign(campaign_dir)
                )
                final = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=harness,
                    checkpoint_gate_reader=ready,
                )
            campaign = load_frozen_campaign(campaign_dir)
            confirmation = campaign_module._latest_checkpoint_completion(campaign)
            candidate_id = campaign.proposals[0].candidate_id
            confirmation_cells = campaign.cells_for(
                candidate_id=candidate_id, stage="confirmation"
            )

        self.assertEqual(screen.sequence, 2)
        self.assertEqual(screen.pair_kind, "screen")
        self.assertEqual(confirmation.sequence, 3)
        self.assertEqual(confirmation.pair_kind, "confirmation")
        self.assertEqual(confirmation.search_pair_index, 1)
        self.assertEqual(
            confirmation.ordered_cell_ids,
            tuple(confirmation_cells[arm].cell_id for arm in pair_order(1)),
        )
        self.assertEqual(final["status"], "complete")
        self.assertEqual(len(gate_calls), 2)

    def test_second_candidate_screen_uses_occurrence_not_global_pair_index(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            clock = _Clock()
            harness = _CompleteCellHarness(clock, improvement=0.9)
            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                for _ in range(3):
                    run_campaign(
                        campaign_dir,
                        workspace=ROOT,
                        now=clock,
                        meminfo_reader=_admission_meminfo,
                        cell_runner=harness,
                        checkpoint_gate_reader=_ready_gate,
                    )
            campaign = load_frozen_campaign(campaign_dir)
            completion = campaign_module._latest_checkpoint_completion(campaign)
            candidate_id = campaign.proposals[1].candidate_id
            cells = campaign.cells_for(candidate_id=candidate_id, stage="screen")

        self.assertEqual(completion.sequence, 3)
        self.assertEqual(completion.search_pair_index, 1)
        self.assertEqual(completion.candidate_id, candidate_id)
        self.assertEqual(completion.pair_kind, "screen")
        self.assertEqual(
            completion.ordered_cell_ids,
            tuple(cells[arm].cell_id for arm in pair_order(1)),
        )

    def test_missing_checkpoint_pauses_before_admission_and_mutation(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=harness,
                    checkpoint_gate_reader=_ready_gate,
                )
                events_before = (campaign_dir / "events.jsonl").read_bytes()
                summary_before = (campaign_dir / "summary.json").read_bytes()
                calls_before = tuple(harness.calls)

                def forbidden(*_args: object, **_kwargs: object) -> object:
                    self.fail("checkpoint pause crossed into local admission or launch")

                paused = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=forbidden,  # type: ignore[arg-type]
                    harness_identity_reader=forbidden,  # type: ignore[arg-type]
                    cell_runner=forbidden,  # type: ignore[arg-type]
                )

            self.assertEqual(paused["status"], "checkpoint_required")
            self.assertEqual(paused["checkpoint_reason"], "missing")
            self.assertEqual(paused["checkpoint_sequence"], 1)
            self.assertEqual((campaign_dir / "events.jsonl").read_bytes(), events_before)
            self.assertEqual((campaign_dir / "summary.json").read_bytes(), summary_before)
            self.assertEqual(tuple(harness.calls), calls_before)

    def test_gate_is_not_rechecked_between_cells_or_during_scored_settlement(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            gate_calls: list[str] = []

            def gate(_campaign: object, _workspace: Path) -> CheckpointGate:
                gate_calls.append("gate")
                return CheckpointGate("ready", None, 1)

            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=harness,
                    checkpoint_gate_reader=gate,
                )
                calls_after_calibration = len(harness.calls)
                with (
                    patch.object(
                        campaign_module,
                        "_require_score_eligible",
                        side_effect=KeyboardInterrupt,
                    ),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    run_campaign(
                        campaign_dir,
                        workspace=ROOT,
                        now=clock,
                        meminfo_reader=_admission_meminfo,
                        cell_runner=harness,
                        checkpoint_gate_reader=gate,
                    )
                self.assertEqual(gate_calls, ["gate"])
                self.assertEqual(len(harness.calls) - calls_after_calibration, 2)

                settled = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=lambda _cell: self.fail("scored pair relaunched a cell"),
                    checkpoint_gate_reader=lambda *_args: self.fail(
                        "scored settlement performed a checkpoint proof"
                    ),
                )

            self.assertEqual(settled["status"], "active")
            self.assertEqual(gate_calls, ["gate"])

    def test_raw_complete_calibration_reconciliation_writes_controller_start(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            campaign = load_frozen_campaign(campaign_dir)
            cells = campaign.cells_for(candidate_id="control", stage="calibration")
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            harness(cells["control_a"])
            harness(cells["control_b"])
            calls_before = tuple(harness.calls)
            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=lambda: self.fail("reconcile entered admission"),
                    cell_runner=lambda _cell: self.fail("reconcile launched a cell"),
                    checkpoint_gate_reader=lambda *_args: self.fail(
                        "reconcile checked a checkpoint in the same invocation"
                    ),
                )
            events = _events(campaign_dir / "events.jsonl")

        self.assertEqual(summary["status"], "active")
        self.assertEqual(tuple(harness.calls), calls_before)
        self.assertEqual(events[0]["event"], "autoresearch_campaign_started")

    def test_preexisting_calibration_record_without_journal_stops_at_boundary(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            campaign = load_frozen_campaign(campaign_dir)
            cells = campaign.cells_for(candidate_id="control", stage="calibration")
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            harness(cells["control_a"])
            harness(cells["control_b"])
            self.assertTrue(campaign_module._reconcile_raw_calibration(campaign))
            self.assertFalse((campaign_dir / "events.jsonl").exists())
            calls_before = tuple(harness.calls)

            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=lambda: self.fail("recovery entered admission"),
                    cell_runner=lambda _cell: self.fail("recovery launched search"),
                    checkpoint_gate_reader=lambda *_args: self.fail(
                        "recovered boundary checked the checkpoint in the same call"
                    ),
                )
            events = _events(campaign_dir / "events.jsonl")

        self.assertEqual(summary["status"], "active")
        self.assertEqual(tuple(harness.calls), calls_before)
        self.assertEqual(events[0]["event"], "autoresearch_campaign_started")

    def test_explicit_checkpoint_rejects_unsettled_boundaries_before_core(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            root = Path(directory)
            for boundary in ("partial_calibration", "candidate", "pair", "scored"):
                with self.subTest(boundary=boundary):
                    boundary_root = root / boundary
                    boundary_root.mkdir()
                    campaign_dir = _freeze_campaign_fixture(boundary_root)
                    campaign = load_frozen_campaign(campaign_dir)
                    clock = _Clock()
                    harness = _CompleteCellHarness(clock)
                    cells = campaign.cells_for(
                        candidate_id="control", stage="calibration"
                    )
                    if boundary == "partial_calibration":
                        harness(cells["control_a"])
                    else:
                        with patch.object(
                            campaign_module, "_recover_cell", return_value="absent"
                        ):
                            run_campaign(
                                campaign_dir,
                                workspace=ROOT,
                                now=clock,
                                meminfo_reader=_admission_meminfo,
                                cell_runner=harness,
                                checkpoint_gate_reader=_ready_gate,
                            )
                        if boundary in {"candidate", "pair"}:
                            _append_candidate_started(campaign_dir)
                            if boundary == "pair":
                                _append_pair_started(campaign_dir)
                        else:
                            with (
                                patch.object(
                                    campaign_module,
                                    "_recover_cell",
                                    return_value="absent",
                                ),
                                patch.object(
                                    campaign_module,
                                    "_require_score_eligible",
                                    side_effect=KeyboardInterrupt,
                                ),
                                self.assertRaises(KeyboardInterrupt),
                            ):
                                run_campaign(
                                    campaign_dir,
                                    workspace=ROOT,
                                    now=clock,
                                    meminfo_reader=_admission_meminfo,
                                    cell_runner=harness,
                                    checkpoint_gate_reader=_ready_gate,
                                )

                    with patch(
                        "bench.autoresearch_checkpoint.acknowledge_checkpoint"
                    ) as core:
                        with self.assertRaises(CheckpointError) as raised:
                            acknowledge_campaign_checkpoint(campaign_dir, ROOT)
                    self.assertEqual(
                        raised.exception.code, "checkpoint_boundary_unsettled"
                    )
                    core.assert_not_called()

    def test_failed_calibration_idle_crash_rejects_checkpoint_before_core(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            _clock, _harness = _make_failed_calibration(campaign_dir)
            campaign = load_frozen_campaign(campaign_dir)
            campaign_module._append_campaign_started(
                Journal(campaign_dir / "events.jsonl"), campaign
            )
            with self.assertRaises(CampaignPlanningError):
                campaign_module._reconcile_raw_calibration(campaign)

            with patch(
                "bench.autoresearch_checkpoint.acknowledge_checkpoint"
            ) as core:
                with self.assertRaises(CheckpointError) as raised:
                    acknowledge_campaign_checkpoint(campaign_dir, ROOT)

        self.assertEqual(raised.exception.code, "checkpoint_boundary_unsettled")
        core.assert_not_called()

    def test_idle_terminal_decision_crashes_reject_checkpoint_before_core(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            root = Path(directory)
            for mode in ("promotion", "all_rejected"):
                with self.subTest(mode=mode):
                    mode_root = root / mode
                    mode_root.mkdir()
                    campaign_dir = _freeze_campaign_fixture(mode_root)
                    clock = _Clock()
                    harness = _CompleteCellHarness(
                        clock, improvement=1.05 if mode == "promotion" else 0.9
                    )
                    runs_before_tail = 2 if mode == "promotion" else 3
                    with patch.object(
                        campaign_module, "_recover_cell", return_value="absent"
                    ):
                        for _ in range(runs_before_tail):
                            run_campaign(
                                campaign_dir,
                                workspace=ROOT,
                                now=clock,
                                meminfo_reader=_admission_meminfo,
                                cell_runner=harness,
                                checkpoint_gate_reader=_ready_gate,
                            )

                        original_append = campaign_module._append_frozen_transition

                        def interrupt_completion(
                            journal: Journal,
                            campaign: object,
                            event: object,
                        ) -> object:
                            assert isinstance(event, dict)
                            if event.get("event") == "autoresearch_campaign_completed":
                                raise KeyboardInterrupt
                            return original_append(  # type: ignore[arg-type]
                                journal, campaign, event
                            )

                        with (
                            patch.object(
                                campaign_module,
                                "_append_frozen_transition",
                                side_effect=interrupt_completion,
                            ),
                            self.assertRaises(KeyboardInterrupt),
                        ):
                            run_campaign(
                                campaign_dir,
                                workspace=ROOT,
                                now=clock,
                                meminfo_reader=_admission_meminfo,
                                cell_runner=harness,
                                checkpoint_gate_reader=_ready_gate,
                            )

                    with patch(
                        "bench.autoresearch_checkpoint.acknowledge_checkpoint"
                    ) as core:
                        with self.assertRaises(CheckpointError) as raised:
                            acknowledge_campaign_checkpoint(campaign_dir, ROOT)
                    self.assertEqual(
                        raised.exception.code, "checkpoint_boundary_unsettled"
                    )
                    core.assert_not_called()

    def test_failed_calibration_terminal_replays_and_is_acknowledgeable(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            clock, harness = _make_failed_calibration(campaign_dir)
            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                terminal = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=harness,
                    checkpoint_gate_reader=_ready_gate,
                )
                replayed = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=lambda: self.fail("terminal replay entered admission"),
                    cell_runner=lambda _cell: self.fail("terminal replay launched a cell"),
                    checkpoint_gate_reader=lambda *_args: self.fail(
                        "terminal replay checked a checkpoint"
                    ),
                )

            sentinel = SimpleNamespace(name="acknowledgement")

            def core(**kwargs: object) -> object:
                reader = kwargs["completion_reader"]
                self.assertTrue(callable(reader))
                first = reader()
                second = reader()
                self.assertEqual(first, second)
                self.assertEqual(first.sequence, 1)
                self.assertEqual(first.pair_kind, "calibration")
                return sentinel

            with patch(
                "bench.autoresearch_checkpoint.acknowledge_checkpoint",
                side_effect=core,
            ):
                acknowledgement = acknowledge_campaign_checkpoint(campaign_dir, ROOT)

        self.assertEqual(terminal["status"], "terminated")
        self.assertEqual(replayed, terminal)
        self.assertIs(acknowledgement, sentinel)

    def test_terminal_without_a_completed_pair_reaches_core_no_pair(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            campaign = load_frozen_campaign(campaign_dir)
            journal = Journal(campaign_dir / "events.jsonl")
            campaign_module._append_campaign_started(journal, campaign)
            campaign_module._append_terminal(
                journal,
                campaign,
                failure_kind="measurement",
                cleanup_verified=True,
            )
            with self.assertRaises(CheckpointError) as raised:
                acknowledge_campaign_checkpoint(
                    campaign_dir,
                    ROOT,
                    evidence_verifier=lambda *_args: self.fail("unexpected evidence"),
                    repository_verifier=lambda *_args: self.fail("unexpected Git"),
                )

        self.assertEqual(raised.exception.code, "no_completed_pair")

    def test_ack_cleanup_precedes_boundary_and_core_proof(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            calls: list[str] = []
            sentinel = SimpleNamespace(name="acknowledgement")

            with (
                patch.object(
                    campaign_module,
                    "_recover_interrupted_cells",
                    side_effect=lambda _campaign: calls.append("cleanup"),
                ),
                patch.object(
                    campaign_module,
                    "_require_checkpoint_boundary",
                    side_effect=lambda _campaign: calls.append("boundary"),
                ),
                patch(
                    "bench.autoresearch_checkpoint.acknowledge_checkpoint",
                    side_effect=lambda **_kwargs: calls.append("core") or sentinel,
                ),
            ):
                acknowledgement = acknowledge_campaign_checkpoint(campaign_dir, ROOT)

        self.assertIs(acknowledgement, sentinel)
        self.assertEqual(calls, ["cleanup", "boundary", "core"])

    def test_ack_recovery_failure_prevents_boundary_and_proof(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            with (
                patch.object(
                    campaign_module,
                    "_recover_interrupted_cells",
                    side_effect=CellProjectionError(
                        "synthetic cleanup failure", failure_kind="cleanup_breach"
                    ),
                ),
                patch.object(campaign_module, "_require_checkpoint_boundary") as boundary,
                patch(
                    "bench.autoresearch_checkpoint.acknowledge_checkpoint"
                ) as core,
                self.assertRaises(CheckpointError) as raised,
            ):
                acknowledge_campaign_checkpoint(
                    campaign_dir,
                    ROOT,
                    evidence_verifier=lambda *_args: self.fail("unexpected evidence"),
                    repository_verifier=lambda *_args: self.fail("unexpected Git"),
                )

        self.assertEqual(raised.exception.code, "checkpoint_boundary_unsettled")
        boundary.assert_not_called()
        core.assert_not_called()

    def test_exact_container_without_journal_terminalizes_after_full_cleanup(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            campaign = load_frozen_campaign(campaign_dir)
            recovered: list[str] = []

            def recover(cell: object) -> str:
                cell_id = str(getattr(cell, "cell_id"))
                recovered.append(cell_id)
                return (
                    "stopped_owned_container"
                    if len(recovered) == 1
                    else "already_absent"
                )

            with (
                patch.object(
                    campaign_module,
                    "recover_owned_worker",
                    return_value=SimpleNamespace(outcome="no_state"),
                ),
                patch.object(campaign_module, "_recover_cell", side_effect=recover),
            ):
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=_Clock(),
                    meminfo_reader=lambda: self.fail("recovery entered admission"),
                    cell_runner=lambda _cell: self.fail("recovery relaunched a cell"),
                    checkpoint_gate_reader=lambda *_args: self.fail(
                        "recovery checked a checkpoint"
                    ),
                )

        self.assertEqual(recovered, [cell.cell_id for cell in campaign.cells])
        self.assertEqual(summary["status"], "terminated")
        self.assertEqual(summary["terminal_reason"], "measurement")


if __name__ == "__main__":
    unittest.main()
