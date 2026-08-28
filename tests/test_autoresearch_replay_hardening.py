from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import bench.autoresearch_campaign as campaign_module
from bench.autoresearch_campaign import (
    CampaignPlanningError,
    CellProjection,
    FrozenCell,
    load_frozen_campaign,
    project_completed_cell,
    run_campaign,
    summarize_campaign,
)
from tests.test_autoresearch_campaign import (
    ROOT,
    _admission_meminfo,
    _freeze_campaign_fixture,
)


class _Clock:
    def __init__(self) -> None:
        self.value = datetime.fromisoformat("2026-08-28T00:00:00-07:00")

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class _CompleteCellHarness:
    """Write strict, synthetic raw cells while preserving each frozen plan."""

    def __init__(self, clock: _Clock, *, improvement: float = 1.05) -> None:
        self.clock = clock
        self.improvement = improvement
        self.calls: list[str] = []
        self.crash_after_cell: str | None = None
        self.delay_after_cell: str | None = None
        self.delay_after_s = 0.0
        self.finalize_after_cell: str | None = None
        self.finalization_s = 0.0

    @staticmethod
    def _arg_value(model: dict[str, object], option: str) -> int:
        arguments = model["args"]
        assert isinstance(arguments, list)
        index = arguments.index(option)
        return int(arguments[index + 1])

    def __call__(self, cell: FrozenCell) -> CellProjection:
        self.calls.append(cell.cell_id)
        started = self.clock()
        self.clock.advance(10.0)
        stopped = self.clock()
        run_completed = (
            stopped + timedelta(seconds=self.finalization_s)
            if self.finalize_after_cell == cell.cell_id
            else stopped
        )

        plan = json.loads((cell.run_dir / "plan.json").read_text())
        model = plan["model"]
        suite = plan["suite"]
        assert isinstance(model, dict)
        assert isinstance(suite, dict)
        cases = suite["cases"]
        assert isinstance(cases, list)

        is_baseline = cell.profile_id.endswith(
            "agent64k-low-ple-mapped-sglang"
        )
        factor = 1.0 if is_baseline else self.improvement
        case_summaries: list[dict[str, object]] = []
        for raw_case in cases:
            assert isinstance(raw_case, dict)
            stable_id = str(raw_case["id"])
            summary: dict[str, object] = {
                "case_id": raw_case["case_id"],
                "kind": raw_case["kind"],
                "measurement_valid": True,
                "validation_passed": True,
                "aggregate_output_tps": 30.0,
                "median_e2e_s": 20.0,
                "median_ttft_s": 2.0,
                "telemetry": {"minimum_memavailable_gib": 20.0},
            }
            if stable_id.startswith("agentic-"):
                summary.update(
                    {
                        "median_agentic_task_wall_s": 10.0 / factor,
                        "median_agentic_first_turn_ttft_s": 2.0,
                    }
                )
            elif stable_id == "long-context-needle-60000-agent-c1":
                summary.update(
                    {
                        "median_e2e_s": 20.0 / factor,
                        "median_ttft_s": 2.0,
                    }
                )
            elif stable_id == "agent64k-decode-256-c1-v1":
                summary.update(
                    {
                        "aggregate_output_tps": 30.0 * factor,
                        "median_ttft_s": 2.0,
                    }
                )
            case_summaries.append(summary)

        summary = {
            "status": "complete",
            "completed_cases": len(cases),
            "failed_cases": [],
            "validation_failed_cases": [],
            "measurement_invalid_cases": [],
            "unsupported_cases": [],
            "unimplemented_cases": [],
            "context_limited_cases": [],
            "measurement_annotations": [],
            "startup_measurement_annotations": [],
            "startup_measurement_valid": True,
            "cases": case_summaries,
        }
        (cell.run_dir / "summary.json").write_text(
            json.dumps(summary, sort_keys=True) + "\n"
        )

        depth = self._arg_value(model, "--speculative-num-steps")
        accepted = {str(index): 1 for index in range(depth)}
        measurement_started_ns = 1_000_000_000
        measurement_complete_ns = 101_000_000_000
        server_stopped_ns = 111_000_000_000
        events: list[dict[str, object]] = [
            {
                "timestamp": started.isoformat(),
                "event": "run_start",
                "completed_cases_at_resume": [],
                "plan_fingerprint": cell.plan_fingerprint,
                "run_nonce": cell.run_nonce,
            },
            {
                "timestamp": started.isoformat(),
                "event": "measurement_started",
                "monotonic_ns": measurement_started_ns,
                "plan_fingerprint": cell.plan_fingerprint,
                "run_nonce": cell.run_nonce,
            },
            {"timestamp": started.isoformat(), "event": "server_ready"},
        ]
        events.extend(
            {
                "timestamp": started.isoformat(),
                "event": "case_complete",
                "case_id": raw_case["case_id"],
                "validation_passed": True,
            }
            for raw_case in cases
            if isinstance(raw_case, dict)
        )
        events.extend(
            (
                {
                    "timestamp": stopped.isoformat(),
                    "event": "sglang_spec_decode_metrics_snapshot",
                    "metrics": {
                        "requested": True,
                        "method": "NEXTN",
                        "configured_max_draft_tokens": depth,
                        "accepted_tokens_per_position": accepted,
                        "num_drafts": 10,
                        "num_accepted_tokens": depth,
                    },
                },
                {
                    "timestamp": stopped.isoformat(),
                    "event": "measurement_complete",
                    "elapsed_s": 100.0,
                    "monotonic_ns": measurement_complete_ns,
                },
                {
                    "timestamp": stopped.isoformat(),
                    "event": "server_stopped",
                    "backend": "sglang",
                    "cleanup_elapsed_s": 10.0,
                    "monotonic_ns": server_stopped_ns,
                },
                {
                    "timestamp": run_completed.isoformat(),
                    "event": "run_complete",
                    "status": "completed",
                },
            )
        )
        (cell.run_dir / "events.jsonl").write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
        )
        swap_total_kib = 16 * 1024 * 1024
        telemetry = (
            {
                "memavailable_kib": 20 * 1024 * 1024,
                "swaptotal_kib": swap_total_kib,
                "swapfree_kib": swap_total_kib,
            },
            {
                "memavailable_kib": 19 * 1024 * 1024,
                "swaptotal_kib": swap_total_kib,
                "swapfree_kib": swap_total_kib,
            },
        )
        (cell.run_dir / "telemetry.jsonl").write_text(
            "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in telemetry)
        )

        projection = project_completed_cell(cell.run_dir)
        if self.crash_after_cell == cell.cell_id:
            raise KeyboardInterrupt
        if self.delay_after_cell == cell.cell_id:
            self.clock.advance(self.delay_after_s)
        return projection


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _set_event_timestamp(
    cell: FrozenCell, event_name: str, timestamp: datetime
) -> None:
    events_path = cell.run_dir / "events.jsonl"
    events = _events(events_path)
    event = next(item for item in events if item.get("event") == event_name)
    event["timestamp"] = timestamp.isoformat()
    events_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in events)
    )


class AutoresearchReplayHardeningTests(unittest.TestCase):
    def _write_raw_complete_screen_pair(
        self,
        campaign_dir: Path,
        clock: _Clock,
        harness: _CompleteCellHarness,
    ) -> tuple[object, dict[str, FrozenCell]]:
        run_campaign(
            campaign_dir,
            workspace=ROOT,
            now=clock,
            meminfo_reader=_admission_meminfo,
            cell_runner=harness,
        )
        campaign = load_frozen_campaign(campaign_dir)
        candidate_id = campaign.proposals[0].candidate_id
        cells = campaign.cells_for(candidate_id=candidate_id, stage="screen")
        harness.crash_after_cell = cells["candidate"].cell_id
        with self.assertRaises(KeyboardInterrupt):
            run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=clock,
                meminfo_reader=_admission_meminfo,
                cell_runner=harness,
            )
        harness.crash_after_cell = None
        return campaign, cells

    def test_copied_journal_is_rejected_by_exact_frozen_instance(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            root = Path(directory)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            with patch.object(
                campaign_module,
                "utc_now",
                return_value="2026-08-28T00:00:00.000+00:00",
            ):
                first = _freeze_campaign_fixture(first_root)
            with patch.object(
                campaign_module,
                "utc_now",
                return_value="2026-08-28T00:00:00.001+00:00",
            ):
                second = _freeze_campaign_fixture(second_root)

            first_frozen = json.loads((first / "campaign.json").read_text())
            second_frozen = json.loads((second / "campaign.json").read_text())
            self.assertNotEqual(
                first_frozen["integrity_hash"], second_frozen["integrity_hash"]
            )

            def interrupt(_cell: FrozenCell) -> CellProjection:
                raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                run_campaign(
                    first,
                    workspace=ROOT,
                    now=lambda: datetime.fromisoformat(
                        "2026-08-28T00:00:00-07:00"
                    ),
                    meminfo_reader=_admission_meminfo,
                    cell_runner=interrupt,
                )
            shutil.copyfile(first / "events.jsonl", second / "events.jsonl")

            launched: list[str] = []
            with self.assertRaises(CampaignPlanningError):
                run_campaign(
                    second,
                    workspace=ROOT,
                    now=lambda: datetime.fromisoformat(
                        "2026-08-28T00:00:01-07:00"
                    ),
                    meminfo_reader=_admission_meminfo,
                    cell_runner=lambda cell: launched.append(cell.cell_id),  # type: ignore[arg-type,return-value]
                )
            self.assertEqual(launched, [])

    def test_summary_recomputes_and_rejects_a_tampered_pair_score(self) -> None:
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
            candidate_id = campaign.proposals[0].candidate_id
            candidate_cell = campaign.cells_for(
                candidate_id=candidate_id, stage="screen"
            )["candidate"]
            plan = json.loads((candidate_cell.run_dir / "plan.json").read_text())
            frozen_case_id = next(
                case["case_id"]
                for case in plan["suite"]["cases"]
                if case["id"] == "agentic-select-and-call"
            )
            summary_path = candidate_cell.run_dir / "summary.json"
            raw_summary = json.loads(summary_path.read_text())
            raw_case = next(
                case
                for case in raw_summary["cases"]
                if case["case_id"] == frozen_case_id
            )
            raw_case["median_agentic_task_wall_s"] = (
                float(raw_case["median_agentic_task_wall_s"]) * 1.0001
            )
            summary_path.write_text(json.dumps(raw_summary, sort_keys=True) + "\n")

            with self.assertRaises(CampaignPlanningError):
                summarize_campaign(campaign_dir)

    def test_replay_projects_each_completed_frozen_cell_with_nonce_binding(self) -> None:
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
                candidate_id = campaign.proposals[0].candidate_id
                cells = campaign.cells_for(
                    candidate_id=candidate_id, stage="screen"
                )
                harness.crash_after_cell = cells["candidate"].cell_id
                with self.assertRaises(KeyboardInterrupt):
                    run_campaign(
                        campaign_dir,
                        workspace=ROOT,
                        now=clock,
                        meminfo_reader=_admission_meminfo,
                        cell_runner=harness,
                    )

            controller_events = _events(campaign_dir / "events.jsonl")
            self.assertEqual(
                sum(
                    event.get("event") == "autoresearch_cell_completed"
                    for event in controller_events
                ),
                1,
            )
            events_path = cells["champion"].run_dir / "events.jsonl"
            raw_events = _events(events_path)
            run_start = next(
                event for event in raw_events if event.get("event") == "run_start"
            )
            run_start["run_nonce"] = "f" * 32
            events_path.write_text(
                "".join(json.dumps(event, sort_keys=True) + "\n" for event in raw_events)
            )

            with self.assertRaises(CampaignPlanningError):
                summarize_campaign(campaign_dir)

    def test_pair_score_reserve_replays_from_durable_run_completion(self) -> None:
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
                candidate_id = campaign.proposals[0].candidate_id
                second = campaign.cells_for(
                    candidate_id=candidate_id, stage="screen"
                )["candidate"]
                harness.delay_after_cell = second.cell_id
                harness.delay_after_s = 600.0
                harness.finalize_after_cell = second.cell_id
                harness.finalization_s = 7.0
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=harness,
                )

            score = next(
                event
                for event in _events(campaign_dir / "events.jsonl")
                if event.get("event") == "autoresearch_pair_scored"
            )
            observation = score["observation"]
            assert isinstance(observation, dict)
            timing = observation["timing"]
            assert isinstance(timing, dict)
            completed = next(
                event
                for event in _events(second.run_dir / "events.jsonl")
                if event.get("event") == "run_complete"
            )
            completed_at = datetime.fromisoformat(str(completed["timestamp"]))
            stopped = next(
                event
                for event in _events(second.run_dir / "events.jsonl")
                if event.get("event") == "server_stopped"
            )
            stopped_at = datetime.fromisoformat(str(stopped["timestamp"]))
            durable_reserve = (campaign.cutoff - completed_at).total_seconds()
            current_reserve = (campaign.cutoff - clock()).total_seconds()
            self.assertEqual(
                timing["audit_reserve_remaining_s"], durable_reserve
            )
            self.assertEqual(completed_at - stopped_at, timedelta(seconds=7.0))
            self.assertEqual(durable_reserve - current_reserve, 593.0)

    def test_replay_scores_both_raw_complete_cells_without_new_inference(self) -> None:
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
                campaign = campaign_module.load_frozen_campaign(campaign_dir)
                candidate_id = campaign.proposals[0].candidate_id
                second = campaign.cells_for(
                    candidate_id=candidate_id, stage="screen"
                )["candidate"]
                harness.crash_after_cell = second.cell_id
                with self.assertRaises(KeyboardInterrupt):
                    run_campaign(
                        campaign_dir,
                        workspace=ROOT,
                        now=clock,
                        meminfo_reader=_admission_meminfo,
                        cell_runner=harness,
                )

                calls_at_crash = tuple(harness.calls)
                admission_calls: list[str] = []
                before = _events(campaign_dir / "events.jsonl")
                self.assertEqual(
                    sum(
                        event.get("event") == "autoresearch_cell_completed"
                        for event in before
                    ),
                    1,
                )
                self.assertFalse(
                    any(
                        event.get("event") == "autoresearch_pair_scored"
                        for event in before
                    )
                )

                harness.crash_after_cell = None
                clock.value = datetime.fromisoformat("2026-08-28T08:00:00-07:00")

                def dirty_meminfo() -> str:
                    admission_calls.append("memory")
                    return _admission_meminfo(swap_used_mib=65)

                def changed_harness(_workspace: Path) -> tuple[str, int]:
                    admission_calls.append("harness")
                    return "0" * 64, 1

                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=dirty_meminfo,
                    harness_identity_reader=changed_harness,
                    cell_runner=lambda _cell: self.fail(
                        "raw-complete reconciliation launched inference"
                    ),
                )

            after = _events(campaign_dir / "events.jsonl")
            self.assertEqual(tuple(harness.calls), calls_at_crash)
            self.assertEqual(admission_calls, [])
            self.assertEqual(
                sum(
                    event.get("event") == "autoresearch_cell_completed"
                    for event in after
                ),
                2,
            )
            self.assertEqual(
                sum(
                    event.get("event") == "autoresearch_pair_scored"
                    for event in after
                ),
                1,
            )
            self.assertEqual(
                sum(
                    event.get("event") == "autoresearch_candidate_decided"
                    for event in after
                ),
                1,
            )
            self.assertNotEqual(summary["status"], "terminated")
            self.assertEqual(summary["next_pair_index"], 1)

    def test_raw_complete_prefix_is_preserved_before_dirty_launch_admission(self) -> None:
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
                candidate_id = campaign.proposals[0].candidate_id
                cells = campaign.cells_for(
                    candidate_id=candidate_id, stage="screen"
                )
                harness.crash_after_cell = cells["champion"].cell_id
                with self.assertRaises(KeyboardInterrupt):
                    run_campaign(
                        campaign_dir,
                        workspace=ROOT,
                        now=clock,
                        meminfo_reader=_admission_meminfo,
                        cell_runner=harness,
                    )
                harness.crash_after_cell = None
                calls_before_resume = tuple(harness.calls)
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=lambda: _admission_meminfo(swap_used_mib=65),
                    cell_runner=lambda _cell: self.fail(
                        "dirty admission launched the pristine second arm"
                    ),
                )

            events = _events(campaign_dir / "events.jsonl")

        self.assertEqual(tuple(harness.calls), calls_before_resume)
        self.assertEqual(summary["status"], "terminated")
        self.assertEqual(summary["terminal_reason"], "swap_pressure")
        self.assertEqual(
            sum(
                event.get("event") == "autoresearch_cell_completed"
                for event in events
            ),
            1,
        )

    def test_later_raw_arm_without_ordered_prefix_fails_before_admission(self) -> None:
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
                candidate_id = campaign.proposals[0].candidate_id
                cells = campaign.cells_for(
                    candidate_id=candidate_id, stage="screen"
                )

                def interrupt_before_raw(_cell: FrozenCell) -> CellProjection:
                    raise KeyboardInterrupt

                with self.assertRaises(KeyboardInterrupt):
                    run_campaign(
                        campaign_dir,
                        workspace=ROOT,
                        now=clock,
                        meminfo_reader=_admission_meminfo,
                        cell_runner=interrupt_before_raw,
                    )
                harness(cells["candidate"])
                calls_before_resume = tuple(harness.calls)
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=lambda: self.fail(
                        "out-of-order raw artifacts consulted memory admission"
                    ),
                    harness_identity_reader=lambda _workspace: self.fail(
                        "out-of-order raw artifacts consulted harness admission"
                    ),
                    cell_runner=lambda _cell: self.fail(
                        "out-of-order raw artifacts launched inference"
                    ),
                )

            events = _events(campaign_dir / "events.jsonl")

        self.assertEqual(tuple(harness.calls), calls_before_resume)
        self.assertEqual(summary["status"], "terminated")
        self.assertEqual(summary["terminal_reason"], "audit")
        self.assertFalse(
            any(
                event.get("event") == "autoresearch_cell_completed"
                for event in events
            )
        )

    def test_raw_complete_pair_accepts_inclusive_durable_inter_cell_gap(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                _campaign, cells = self._write_raw_complete_screen_pair(
                    campaign_dir, clock, harness
                )
                stopped = next(
                    event
                    for event in _events(cells["champion"].run_dir / "events.jsonl")
                    if event.get("event") == "server_stopped"
                )
                _set_event_timestamp(
                    cells["candidate"],
                    "run_start",
                    datetime.fromisoformat(str(stopped["timestamp"]))
                    + timedelta(seconds=120),
                )
                clock.value = datetime.fromisoformat("2026-08-28T08:00:00-07:00")
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=lambda: self.fail(
                        "durable reconciliation consulted memory admission"
                    ),
                    harness_identity_reader=lambda _workspace: self.fail(
                        "durable reconciliation consulted harness admission"
                    ),
                    cell_runner=lambda _cell: self.fail(
                        "durable reconciliation launched inference"
                    ),
                )

        self.assertNotEqual(summary["status"], "terminated")
        self.assertEqual(summary["next_pair_index"], 1)

    def test_raw_complete_pair_rejects_durable_inter_cell_gap_over_limit(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            clock = _Clock()
            harness = _CompleteCellHarness(clock)
            with patch.object(campaign_module, "_recover_cell", return_value="absent"):
                _campaign, cells = self._write_raw_complete_screen_pair(
                    campaign_dir, clock, harness
                )
                stopped = next(
                    event
                    for event in _events(cells["champion"].run_dir / "events.jsonl")
                    if event.get("event") == "server_stopped"
                )
                _set_event_timestamp(
                    cells["candidate"],
                    "run_start",
                    datetime.fromisoformat(str(stopped["timestamp"]))
                    + timedelta(seconds=120.001),
                )
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=lambda: self.fail(
                        "invalid durable pair consulted memory admission"
                    ),
                    harness_identity_reader=lambda _workspace: self.fail(
                        "invalid durable pair consulted harness admission"
                    ),
                    cell_runner=lambda _cell: self.fail(
                        "invalid durable pair launched inference"
                    ),
                )

            events = _events(campaign_dir / "events.jsonl")

        self.assertEqual(summary["status"], "terminated")
        self.assertEqual(summary["terminal_reason"], "measurement")
        self.assertEqual(
            sum(
                event.get("event") == "autoresearch_cell_completed"
                for event in events
            ),
            1,
        )
        self.assertFalse(
            any(event.get("event") == "autoresearch_pair_scored" for event in events)
        )

    def test_future_raw_search_cell_is_rejected_without_inference(self) -> None:
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
                candidate_id = campaign.proposals[0].candidate_id
                future = campaign.cells_for(
                    candidate_id=candidate_id, stage="confirmation"
                )["candidate"]
                harness(future)
                calls_before_resume = tuple(harness.calls)
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=lambda: self.fail(
                        "orphan rejection consulted memory admission"
                    ),
                    harness_identity_reader=lambda _workspace: self.fail(
                        "orphan rejection consulted harness admission"
                    ),
                    cell_runner=lambda _cell: self.fail(
                        "orphan rejection launched inference"
                    ),
                )

        self.assertEqual(tuple(harness.calls), calls_before_resume)
        self.assertEqual(summary["status"], "terminated")
        self.assertEqual(summary["terminal_reason"], "audit")


if __name__ == "__main__":
    unittest.main()
