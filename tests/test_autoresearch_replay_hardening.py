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
                    "timestamp": stopped.isoformat(),
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
        return projection


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class AutoresearchReplayHardeningTests(unittest.TestCase):
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

    @unittest.skip("awaiting frozen score recomputation source")
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

            events_path = campaign_dir / "events.jsonl"
            records = _events(events_path)
            scored = next(
                event
                for event in records
                if event.get("event") == "autoresearch_pair_scored"
            )
            observation = scored["observation"]
            assert isinstance(observation, dict)
            ratios = observation["primary_speed_ratios"]
            assert isinstance(ratios, list)
            ratios[0] = float(ratios[0]) * 1.0001
            events_path.write_text(
                "".join(json.dumps(event, sort_keys=True) + "\n" for event in records)
            )

            with self.assertRaises(CampaignPlanningError):
                summarize_campaign(campaign_dir)

    @unittest.skip("awaiting raw-complete pair reconciliation source")
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
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=clock,
                    meminfo_reader=lambda: _admission_meminfo(swap_used_mib=65),
                    cell_runner=harness,
                )

            after = _events(campaign_dir / "events.jsonl")
            self.assertEqual(tuple(harness.calls), calls_at_crash)
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
            self.assertNotEqual(summary["status"], "terminated")
            self.assertEqual(summary["next_pair_index"], 1)


if __name__ == "__main__":
    unittest.main()
