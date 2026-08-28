from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bench.autoresearch_campaign as campaign_module
from bench.autoresearch_campaign import (
    CampaignPlanningError,
    FrozenCell,
    load_frozen_campaign,
    run_campaign,
)
from bench.journal import content_hash
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


def _artifact_snapshot(cell: FrozenCell) -> dict[str, bytes]:
    return {
        name: (cell.run_dir / name).read_bytes()
        for name in (
            "plan.json",
            "summary.json",
            "events.jsonl",
            "telemetry.jsonl",
        )
    }


def _mutate_agentic_raw_summary(
    cell: FrozenCell, *, factor: float = 1.0001
) -> None:
    plan = json.loads((cell.run_dir / "plan.json").read_text())
    frozen_case_id = next(
        case["case_id"]
        for case in plan["suite"]["cases"]
        if case["id"] == "agentic-select-and-call"
    )
    summary_path = cell.run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    case = next(
        item for item in summary["cases"] if item["case_id"] == frozen_case_id
    )
    case["median_agentic_task_wall_s"] = (
        float(case["median_agentic_task_wall_s"]) * factor
    )
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")


def _mutate_and_rehash_calibration(path: Path) -> None:
    record = json.loads(path.read_text())
    ratios = record["observation"]["primary_speed_ratios"]
    ratios[0] = float(ratios[0]) * 1.0001
    payload = {key: value for key, value in record.items() if key != "integrity_hash"}
    record["integrity_hash"] = content_hash(payload, 64)
    path.write_text(json.dumps(record, sort_keys=True) + "\n")


class AutoresearchCalibrationReconciliationTests(unittest.TestCase):
    def test_unrehashed_calibration_record_tamper_already_fails_closed(self) -> None:
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

                calibration_path = campaign_dir / "calibration.json"
                record = json.loads(calibration_path.read_text())
                ratios = record["observation"]["primary_speed_ratios"]
                ratios[0] = float(ratios[0]) * 1.0001
                calibration_path.write_text(
                    json.dumps(record, sort_keys=True) + "\n"
                )
                calls_before = tuple(harness.calls)
                with self.assertRaises(CampaignPlanningError):
                    run_campaign(
                        campaign_dir,
                        workspace=ROOT,
                        now=clock,
                        meminfo_reader=_admission_meminfo,
                        cell_runner=harness,
                    )

            self.assertEqual(tuple(harness.calls), calls_before)

    def test_two_raw_calibration_cells_reconcile_before_cutoff_or_admission(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            campaign = load_frozen_campaign(campaign_dir)
            cells = campaign.cells_for(candidate_id="control", stage="calibration")
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            harness(cells["control_a"])
            harness(cells["control_b"])
            calls_before = tuple(harness.calls)
            clock.value = datetime.fromisoformat("2026-08-28T08:00:00-07:00")
            admission_calls: list[str] = []
            identity_calls: list[Path] = []

            def dirty_meminfo() -> str:
                admission_calls.append("meminfo")
                return _admission_meminfo(swap_used_mib=65)

            def harness_identity(workspace: Path) -> tuple[str, int]:
                identity_calls.append(workspace)
                return campaign.harness_tree_sha256, campaign.harness_file_count

            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=dirty_meminfo,
                    harness_identity_reader=harness_identity,
                    cell_runner=harness,
                )

            self.assertTrue(summary["calibration_recorded"])
            self.assertIsNotNone(campaign_module._load_calibration(campaign))
            self.assertEqual(tuple(harness.calls), calls_before)
            self.assertEqual(admission_calls, [])
            self.assertEqual(identity_calls, [])

    def test_calibration_reserve_uses_later_durable_completion(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            campaign = load_frozen_campaign(campaign_dir)
            cells = campaign.cells_for(candidate_id="control", stage="calibration")
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            harness.finalize_after_cell = cells["control_b"].cell_id
            harness.finalization_s = 7.0

            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=harness,
                )

            record = json.loads((campaign_dir / "calibration.json").read_text())
            completed = next(
                event
                for event in _events(cells["control_b"].run_dir / "events.jsonl")
                if event.get("event") == "run_complete"
            )
            expected = (
                campaign.cutoff
                - datetime.fromisoformat(str(completed["timestamp"]))
            ).total_seconds()
            self.assertEqual(
                record["observation"]["timing"]["audit_reserve_remaining_s"],
                expected,
            )

    def test_valid_failed_calibration_is_terminal_and_summarizable(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            campaign = load_frozen_campaign(campaign_dir)
            cells = campaign.cells_for(candidate_id="control", stage="calibration")
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            harness(cells["control_a"])
            harness(cells["control_b"])
            _mutate_agentic_raw_summary(cells["control_b"], factor=0.5)
            calls_before = tuple(harness.calls)

            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=harness,
                )

            record = json.loads((campaign_dir / "calibration.json").read_text())
            self.assertEqual(summary["status"], "terminated")
            self.assertTrue(summary["calibration_recorded"])
            self.assertFalse(record["passed"])
            self.assertEqual(tuple(harness.calls), calls_before)

    def test_incomplete_control_a_is_terminal_and_never_relaunched(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            campaign = load_frozen_campaign(campaign_dir)
            cells = campaign.cells_for(candidate_id="control", stage="calibration")
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            harness(cells["control_a"])
            events_path = cells["control_a"].run_dir / "events.jsonl"
            incomplete = [
                event
                for event in _events(events_path)
                if event.get("event") != "run_complete"
            ]
            events_path.write_text(
                "".join(json.dumps(event, sort_keys=True) + "\n" for event in incomplete)
            )
            raw_before = events_path.read_bytes()
            calls_before = tuple(harness.calls)

            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=harness,
                )

            self.assertEqual(summary["status"], "terminated")
            self.assertEqual(events_path.read_bytes(), raw_before)
            self.assertEqual(tuple(harness.calls), calls_before)
            self.assertFalse((campaign_dir / "calibration.json").exists())

    def test_existing_calibration_rejects_raw_or_rehashed_record_tamper(self) -> None:
        for mutation in ("raw", "record"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                dir=ROOT / "results"
            ) as directory:
                campaign_dir = _freeze_campaign_fixture(Path(directory))
                clock = _Clock()
                harness = _CompleteCellHarness(clock)
                with patch.object(
                    campaign_module, "_recover_cell", return_value="absent"
                ):
                    run_campaign(
                        campaign_dir,
                        workspace=ROOT,
                        now=clock,
                        meminfo_reader=_admission_meminfo,
                        cell_runner=harness,
                    )
                    campaign = load_frozen_campaign(campaign_dir)
                    cells = campaign.cells_for(
                        candidate_id="control", stage="calibration"
                    )
                    if mutation == "raw":
                        _mutate_agentic_raw_summary(cells["control_b"])
                    else:
                        _mutate_and_rehash_calibration(
                            campaign_dir / "calibration.json"
                        )
                    calls_before = tuple(harness.calls)
                    with self.assertRaises(CampaignPlanningError):
                        run_campaign(
                            campaign_dir,
                            workspace=ROOT,
                            now=clock,
                            meminfo_reader=_admission_meminfo,
                            cell_runner=harness,
                        )

                self.assertEqual(tuple(harness.calls), calls_before)

    def test_control_b_without_control_a_is_rejected_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            campaign = load_frozen_campaign(campaign_dir)
            cells = campaign.cells_for(candidate_id="control", stage="calibration")
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            harness(cells["control_b"])
            b_snapshot = _artifact_snapshot(cells["control_b"])
            calls_before = tuple(harness.calls)

            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                with self.assertRaises(CampaignPlanningError):
                    run_campaign(
                        campaign_dir,
                        workspace=ROOT,
                        now=clock,
                        meminfo_reader=_admission_meminfo,
                        cell_runner=harness,
                    )

            self.assertEqual(tuple(harness.calls), calls_before)
            self.assertFalse((cells["control_a"].run_dir / "events.jsonl").exists())
            self.assertEqual(_artifact_snapshot(cells["control_b"]), b_snapshot)
            self.assertFalse((campaign_dir / "calibration.json").exists())

    def test_control_a_is_preserved_and_only_b_launches_at_inclusive_gap(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            campaign = load_frozen_campaign(campaign_dir)
            cells = campaign.cells_for(candidate_id="control", stage="calibration")
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            harness(cells["control_a"])
            a_snapshot = _artifact_snapshot(cells["control_a"])
            clock.advance(120.0)

            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=harness,
                )

            self.assertTrue(summary["calibration_recorded"])
            self.assertEqual(
                tuple(harness.calls),
                (cells["control_a"].cell_id, cells["control_b"].cell_id),
            )
            self.assertEqual(_artifact_snapshot(cells["control_a"]), a_snapshot)
            a_stop = next(
                event
                for event in _events(cells["control_a"].run_dir / "events.jsonl")
                if event.get("event") == "server_stopped"
            )
            b_start = next(
                event
                for event in _events(cells["control_b"].run_dir / "events.jsonl")
                if event.get("event") == "run_start"
            )
            gap_s = (
                datetime.fromisoformat(str(b_start["timestamp"]))
                - datetime.fromisoformat(str(a_stop["timestamp"]))
            ).total_seconds()
            self.assertEqual(gap_s, 120.0)


if __name__ == "__main__":
    unittest.main()
