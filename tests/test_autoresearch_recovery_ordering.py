from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import bench.autoresearch_campaign as campaign_module
from bench.autoresearch_campaign import (
    CampaignPlanningError,
    CellProjectionError,
    CellProjection,
    FrozenCell,
    _recover_interrupted_cells,
    load_frozen_campaign,
    run_campaign,
)
from bench.autoresearch_worker import WorkerCleanupResult
from bench.journal import Journal
from tests.test_autoresearch_campaign import (
    ROOT,
    _admission_meminfo,
    _freeze_campaign_fixture,
    _synthetic_projection,
    _synthetic_projection_boundary,
)
from tests.test_autoresearch_replay_hardening import (
    _Clock,
    _CompleteCellHarness,
    _events,
    _set_event_timestamp,
)


_execution_admission_patcher = None


def setUpModule() -> None:
    global _execution_admission_patcher
    _execution_admission_patcher = patch(
        "bench.execution_admission.RETIRED_SGLANG_SOURCE_OVERLAY_DIGESTS",
        frozenset(),
    )
    _execution_admission_patcher.start()


def tearDownModule() -> None:
    assert _execution_admission_patcher is not None
    _execution_admission_patcher.stop()


def _no_worker() -> WorkerCleanupResult:
    return WorkerCleanupResult(
        outcome="no_state",
        identity=None,
        return_code=None,
        sigint_sent=False,
        sigkill_sent=False,
        process_lookup_race=False,
        state_removed=False,
    )


def _interrupt(_cell: FrozenCell) -> CellProjection:
    raise KeyboardInterrupt


class AutoresearchRecoveryOrderingTests(unittest.TestCase):
    def setUp(self) -> None:
        gate = patch.object(
            campaign_module,
            "_checkpoint_gate_for_campaign",
            return_value=SimpleNamespace(ready=True),
        )
        gate.start()
        self.addCleanup(gate.stop)

    def _start_pair_without_raw(
        self, campaign_dir: Path, clock: _Clock, harness: _CompleteCellHarness
    ) -> tuple[object, dict[str, FrozenCell]]:
        run_campaign(
            campaign_dir,
            workspace=ROOT,
            now=clock,
            meminfo_reader=_admission_meminfo,
            cell_runner=harness,
        )
        with self.assertRaises(KeyboardInterrupt):
            run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=clock,
                meminfo_reader=_admission_meminfo,
                cell_runner=_interrupt,
            )
        campaign = load_frozen_campaign(campaign_dir)
        candidate_id = campaign.proposals[0].candidate_id
        return campaign, campaign.cells_for(
            candidate_id=candidate_id, stage="screen"
        )

    def test_all_workers_are_recovered_before_indexed_raw_gap_replay(self) -> None:
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
                )
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=harness,
                )
            campaign = load_frozen_campaign(campaign_dir)
            cells = campaign.cells_for(
                candidate_id=campaign.proposals[0].candidate_id, stage="screen"
            )
            first, second = cells["champion"], cells["candidate"]
            stopped = next(
                event
                for event in _events(first.run_dir / "events.jsonl")
                if event.get("event") == "server_stopped"
            )
            _set_event_timestamp(
                second,
                "run_start",
                datetime.fromisoformat(str(stopped["timestamp"]))
                + timedelta(seconds=120.001),
            )
            before = (campaign_dir / "events.jsonl").read_bytes()
            recovered: list[Path] = []
            containers: list[str] = []

            def recover(run_dir: Path, **_kwargs: object) -> WorkerCleanupResult:
                recovered.append(run_dir)
                return _no_worker()

            def recover_container(cell: FrozenCell) -> str:
                containers.append(cell.cell_id)
                return "already_absent"

            with (
                patch.object(campaign_module, "recover_owned_worker", side_effect=recover),
                patch.object(
                    campaign_module, "_recover_cell", side_effect=recover_container
                ),
            ):
                with self.assertRaisesRegex(
                    CampaignPlanningError, "controller-bound raw"
                ):
                    run_campaign(
                        campaign_dir,
                        workspace=ROOT,
                        now=clock,
                        meminfo_reader=_admission_meminfo,
                        cell_runner=lambda _cell: self.fail("tampered replay launched"),
                    )

            self.assertEqual(recovered, [cell.run_dir for cell in campaign.cells])
            self.assertEqual(containers, [cell.cell_id for cell in campaign.cells])
            self.assertEqual((campaign_dir / "events.jsonl").read_bytes(), before)

    def test_later_exact_container_owner_clears_prior_probe_mismatches(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign = load_frozen_campaign(
                _freeze_campaign_fixture(Path(directory))
            )
            owner_index = len(campaign.cells) // 2
            (campaign.cells[owner_index].run_dir / "events.jsonl").write_text("{}\n")
            attempted: list[str] = []

            def recover_container(cell: FrozenCell) -> str:
                attempted.append(cell.cell_id)
                cell_index = len(attempted) - 1
                if cell_index < owner_index:
                    raise CellProjectionError(
                        "different exact container owner",
                        failure_kind="ownership_ambiguity",
                    )
                if cell_index == owner_index:
                    return "stopped_owned_container"
                return "already_absent"

            with (
                patch.object(
                    campaign_module, "recover_owned_worker", return_value=_no_worker()
                ),
                patch.object(
                    campaign_module, "_recover_cell", side_effect=recover_container
                ),
            ):
                _recover_interrupted_cells(campaign)

            self.assertEqual(attempted, [cell.cell_id for cell in campaign.cells])

    def test_container_mismatch_after_exact_stop_remains_terminal(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign = load_frozen_campaign(
                _freeze_campaign_fixture(Path(directory))
            )
            attempted: list[str] = []

            def recover_container(cell: FrozenCell) -> str:
                attempted.append(cell.cell_id)
                if len(attempted) == 1:
                    return "stopped_owned_container"
                if len(attempted) == 2:
                    raise CellProjectionError(
                        "replacement container appeared",
                        failure_kind="ownership_ambiguity",
                    )
                return "already_absent"

            with (
                patch.object(
                    campaign_module, "recover_owned_worker", return_value=_no_worker()
                ),
                patch.object(
                    campaign_module, "_recover_cell", side_effect=recover_container
                ),
            ):
                with self.assertRaises(CellProjectionError) as raised:
                    _recover_interrupted_cells(campaign)

            self.assertEqual(raised.exception.failure_kind, "ownership_ambiguity")
            self.assertEqual(attempted, [cell.cell_id for cell in campaign.cells])

    def test_valid_first_raw_prefix_persists_before_incomplete_second(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                campaign, cells = self._start_pair_without_raw(
                    campaign_dir, clock, harness
                )
                order = ("champion", "candidate")
                harness(cells[order[0]])
                harness(cells[order[1]])
                second_events = cells[order[1]].run_dir / "events.jsonl"
                incomplete = [
                    event
                    for event in _events(second_events)
                    if event.get("event") != "run_complete"
                ]
                second_events.write_text(
                    "".join(json.dumps(event, sort_keys=True) + "\n" for event in incomplete)
                )
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=lambda: self.fail("invalid prefix reached admission"),
                    cell_runner=lambda _cell: self.fail("invalid prefix launched"),
                )

            events = _events(campaign_dir / "events.jsonl")
            completions = [
                event for event in events if event.get("event") == "autoresearch_cell_completed"
            ]
            self.assertEqual(summary["terminal_reason"], "measurement")
            self.assertEqual([event["arm"] for event in completions], [order[0]])
            self.assertFalse(
                any(event.get("event") == "autoresearch_pair_scored" for event in events)
            )

    def test_missing_admission_precedes_raw_prefix_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                _campaign, cells = self._start_pair_without_raw(
                    campaign_dir, clock, harness
                )
                harness(cells["champion"])
                (campaign_dir / "admissions.jsonl").unlink()
                controller_before = (campaign_dir / "events.jsonl").read_bytes()

                with self.assertRaisesRegex(
                    CampaignPlanningError,
                    "calibration execution has no admitted provenance",
                ):
                    run_campaign(
                        campaign_dir,
                        workspace=ROOT,
                        now=clock,
                        meminfo_reader=lambda: self.fail(
                            "missing admission reached live preflight"
                        ),
                        cell_runner=lambda _cell: self.fail(
                            "missing admission launched inference"
                        ),
                    )

            self.assertEqual(
                (campaign_dir / "events.jsonl").read_bytes(),
                controller_before,
            )

    def test_incomplete_unadmitted_raw_is_audit_before_projection(self) -> None:
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
                )
                campaign = load_frozen_campaign(campaign_dir)
                orphan = campaign.cells_for(
                    candidate_id=campaign.proposals[0].candidate_id,
                    stage="confirmation",
                )["candidate"]
                (orphan.run_dir / "events.jsonl").write_text("{}\n")
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=lambda: self.fail("orphan reached admission"),
                    cell_runner=lambda _cell: self.fail("orphan launched"),
                )

            self.assertEqual(summary["terminal_reason"], "audit")

    def test_raw_appearance_between_prefix_and_admission_is_conservative(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                _campaign, cells = self._start_pair_without_raw(
                    campaign_dir, clock, harness
                )
                first_path = cells["champion"].run_dir / "events.jsonl"
                original_prefix = campaign_module._raw_search_prefix
                injected = False

                def inject(campaign: object, state: object) -> tuple[str, ...]:
                    nonlocal injected
                    prefix = original_prefix(campaign, state)  # type: ignore[arg-type]
                    if not injected:
                        first_path.write_text("{}\n")
                        injected = True
                    return prefix

                admission_calls: list[str] = []

                def dirty_meminfo() -> str:
                    admission_calls.append("memory")
                    first_path.unlink()
                    return _admission_meminfo(swap_used_mib=65)

                with patch.object(
                    campaign_module, "_raw_search_prefix", side_effect=inject
                ):
                    summary = run_campaign(
                        campaign_dir,
                        workspace=ROOT,
                        now=clock,
                        meminfo_reader=dirty_meminfo,
                        cell_runner=lambda _cell: self.fail("race launched inference"),
                    )

            self.assertEqual(admission_calls, ["memory"])
            self.assertEqual(summary["terminal_reason"], "swap_pressure")

    def test_promotion_completion_tail_skips_admission(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            stopped_at = datetime.fromisoformat("2026-08-28T00:00:00-07:00")

            def projection_for(cell: object) -> CellProjection:
                projection = _synthetic_projection(cell, improvement=1.05)
                if "agent64k-none" in str(getattr(cell, "profile_id")):
                    return replace(projection, normalized_flags=())
                return projection

            original_append = Journal.append

            def interrupt_completion(journal: Journal, event: dict[str, object]) -> None:
                if event.get("event") == "autoresearch_campaign_completed":
                    raise KeyboardInterrupt
                original_append(journal, event)

            with _synthetic_projection_boundary(
                projection_for,
                stopped_at=stopped_at,
                audit_reserve_s=25_200.0,
            ):
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: stopped_at,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=projection_for,
                )
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: stopped_at,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=projection_for,
                )
                with patch.object(Journal, "append", new=interrupt_completion):
                    with self.assertRaises(KeyboardInterrupt):
                        run_campaign(
                            campaign_dir,
                            workspace=ROOT,
                            now=lambda: stopped_at,
                            meminfo_reader=_admission_meminfo,
                            cell_runner=projection_for,
                        )
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: datetime.fromisoformat("2026-08-28T08:00:00-07:00"),
                    meminfo_reader=lambda: self.fail("promotion tail used memory admission"),
                    harness_identity_reader=lambda _workspace: self.fail(
                        "promotion tail used harness admission"
                    ),
                    cell_runner=lambda _cell: self.fail("promotion tail launched"),
                )

            self.assertEqual(summary["status"], "complete")

    def test_final_rejection_completion_tail_skips_admission(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            stopped_at = datetime.fromisoformat("2026-08-28T00:00:00-07:00")

            def projection_for(cell: object) -> CellProjection:
                return _synthetic_projection(cell, improvement=1.0)

            original_append = Journal.append

            def interrupt_completion(journal: Journal, event: dict[str, object]) -> None:
                if event.get("event") == "autoresearch_campaign_completed":
                    raise KeyboardInterrupt
                original_append(journal, event)

            with _synthetic_projection_boundary(
                projection_for,
                stopped_at=stopped_at,
                audit_reserve_s=25_200.0,
            ):
                for _ in range(3):
                    run_campaign(
                        campaign_dir,
                        workspace=ROOT,
                        now=lambda: stopped_at,
                        meminfo_reader=_admission_meminfo,
                        cell_runner=projection_for,
                    )
                with patch.object(Journal, "append", new=interrupt_completion):
                    with self.assertRaises(KeyboardInterrupt):
                        run_campaign(
                            campaign_dir,
                            workspace=ROOT,
                            now=lambda: stopped_at,
                            meminfo_reader=_admission_meminfo,
                            cell_runner=projection_for,
                        )
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: datetime.fromisoformat("2026-08-28T08:00:00-07:00"),
                    meminfo_reader=lambda: self.fail(
                        "final rejection tail used memory admission"
                    ),
                    harness_identity_reader=lambda _workspace: self.fail(
                        "final rejection tail used harness admission"
                    ),
                    cell_runner=lambda _cell: self.fail(
                        "final rejection tail launched"
                    ),
                )

            self.assertEqual(summary["status"], "complete")
            self.assertEqual(set(summary["candidate_decisions"].values()), {"reject"})


if __name__ == "__main__":
    unittest.main()
