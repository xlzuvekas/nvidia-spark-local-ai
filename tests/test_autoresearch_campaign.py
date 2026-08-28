from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path
import tempfile
import unittest

from bench.autoresearch_campaign import (
    CampaignPlanningError,
    CaseMeasurement,
    CellProjection,
    CellProjectionError,
    EXPECTED_AXES,
    EXPECTED_CASE_IDS,
    EXPECTED_PRIMARY_CASE_IDS,
    _cell_specs,
    freeze_campaign,
    campaign_admission,
    load_frozen_campaign,
    load_campaign_definition,
    observation_from_cells,
    project_completed_cell,
    run_campaign,
    semantic_config,
    validate_campaign,
)
from bench.autoresearch import CampaignPolicy
from bench.journal import content_hash, write_json
from bench.runner import _canonical_case


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PATH = (
    ROOT
    / "manifests"
    / "campaigns"
    / "qwen38_flash_next_single_user_autoresearch.toml"
)


class AutoresearchCampaignPlanningTests(unittest.TestCase):
    def test_definition_and_three_semantic_deltas_are_exact(self) -> None:
        definition = load_campaign_definition(CAMPAIGN_PATH, workspace=ROOT)
        preview, models, suite = validate_campaign(definition)

        self.assertEqual(preview.suite_id, suite.id)
        self.assertEqual(preview.policy.primary_case_ids, EXPECTED_PRIMARY_CASE_IDS)
        self.assertEqual(preview.policy.allowed_axes, EXPECTED_AXES)
        self.assertEqual(
            tuple(proposal.axis for proposal in preview.proposals),
            EXPECTED_AXES,
        )
        self.assertEqual(preview.to_mapping()["planned_cell_count"], 14)
        self.assertFalse(preview.to_mapping()["execution_started"])

        baseline = semantic_config(models[definition.baseline_id])
        self.assertEqual(baseline["chunked_prefill_size"], 1024)
        self.assertEqual(
            baseline["nextn_bundle"], {"steps": 2, "draft_tokens": 3}
        )
        self.assertEqual(
            baseline["reasoning_policy"],
            {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "reasoning_effort": "low",
                }
            },
        )

    def test_candidate_queue_freezes_calibration_and_two_fresh_pairs_each(self) -> None:
        definition = load_campaign_definition(CAMPAIGN_PATH, workspace=ROOT)
        preview, _models, _suite = validate_campaign(definition)
        cells = _cell_specs(preview)

        self.assertEqual(len(cells), 14)
        self.assertEqual(
            tuple(cell["arm"] for cell in cells[:2]),
            ("control_a", "control_b"),
        )
        for offset, proposal in enumerate(preview.proposals):
            block = cells[2 + 4 * offset : 6 + 4 * offset]
            self.assertEqual(
                tuple((cell["stage"], cell["arm"]) for cell in block),
                (
                    ("screen", "champion"),
                    ("screen", "candidate"),
                    ("confirmation", "candidate"),
                    ("confirmation", "champion"),
                ),
            )
            self.assertTrue(
                all(cell["candidate_id"] == proposal.candidate_id for cell in block)
            )

    def test_non_axis_model_change_is_rejected(self) -> None:
        definition = load_campaign_definition(CAMPAIGN_PATH, workspace=ROOT)
        preview, models, suite = validate_campaign(definition)
        candidate = models[preview.proposals[0].candidate_id]
        models[candidate.id] = replace(candidate, estimated_ram_gib=103.0)

        # Exercise the invariant directly through a temporary models manifest is
        # unnecessary: changing a non-axis field must make the dataclasses differ
        # after the three semantic axes and bookkeeping fields are normalized.
        from bench.autoresearch_campaign import _invariant_model_projection

        baseline = models[definition.baseline_id]
        self.assertNotEqual(
            _invariant_model_projection(baseline),
            _invariant_model_projection(models[candidate.id]),
        )
        self.assertEqual(suite.id, preview.suite_id)

    def test_loader_rejects_unknown_keys_and_paths_outside_workspace(self) -> None:
        source = CAMPAIGN_PATH.read_text()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            unknown = root / "unknown.toml"
            unknown.write_text(source + "\nunknown = true\n")
            with self.assertRaisesRegex(CampaignPlanningError, "unknown keys"):
                load_campaign_definition(unknown, workspace=ROOT)

            escaped = root / "escaped.toml"
            escaped.write_text(source.replace('../models.toml', '../../../../etc/passwd'))
            with self.assertRaisesRegex(CampaignPlanningError, "inside the workspace"):
                load_campaign_definition(escaped, workspace=ROOT)

    def test_freeze_uses_unique_cell_roots_and_never_executes(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_create_plan(**kwargs: object) -> Path:
            calls.append(kwargs)
            results_root = kwargs["results_root"]
            assert isinstance(results_root, Path)
            run_dir = results_root / "frozen-run"
            run_dir.mkdir()
            model = kwargs["model"]
            suite = kwargs["suite"]
            fingerprint = content_hash(
                {"model": getattr(model, "id"), "suite": getattr(suite, "id")},
                64,
            )
            write_json(
                run_dir / "plan.json",
                {
                    "fingerprint": fingerprint,
                    "integrity_hash": "a" * 64,
                },
            )
            return run_dir

        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = freeze_campaign(
                CAMPAIGN_PATH,
                workspace=ROOT,
                results_root=Path(directory),
                create_plan_fn=fake_create_plan,
            )
            frozen = json.loads((campaign_dir / "campaign.json").read_text())

        self.assertEqual(len(calls), 14)
        self.assertEqual(len(frozen["cells"]), 14)
        roots = [Path(call["results_root"]) for call in calls]
        self.assertEqual(len(set(roots)), 14)
        self.assertFalse(frozen["execution_started"])
        integrity = frozen.pop("integrity_hash")
        self.assertEqual(integrity, content_hash(frozen, 64))


def _freeze_campaign_fixture(root: Path) -> Path:
    def fake_create_plan(**kwargs: object) -> Path:
        results_root = kwargs["results_root"]
        model = kwargs["model"]
        suite = kwargs["suite"]
        assert isinstance(results_root, Path)
        run_dir = results_root / "frozen-run"
        run_dir.mkdir()
        model_data = asdict(model)
        suite_data = asdict(suite)
        suite_data.pop("protocol_digest", None)
        cases = [_canonical_case(model_data, case) for case in suite_data["cases"]]
        frozen_suite = {**suite_data, "cases": cases}
        resolved = {"image_digest": "image@sha256:" + "a" * 64}
        fingerprint = content_hash(
            {"model": model_data, "suite": suite_data, "resolved": resolved}
        )
        plan = {
            "schema_version": 2,
            "created_at": "2026-08-28T00:00:00+00:00",
            "fingerprint": fingerprint,
            "models_manifest": "ignored",
            "suite_manifest": "ignored",
            "model": model_data,
            "suite": frozen_suite,
            "resolved": resolved,
            "host_at_plan": {},
        }
        plan["integrity_hash"] = content_hash(plan, 64)
        write_json(run_dir / "plan.json", plan)
        return run_dir

    return freeze_campaign(
        CAMPAIGN_PATH,
        workspace=ROOT,
        results_root=root,
        create_plan_fn=fake_create_plan,
    )


def _admission_meminfo(*, available_gib: int = 120, swap_used_mib: int = 0) -> str:
    total_kib = 128 * 1024**2
    swap_total_kib = 16 * 1024**2
    return "\n".join(
        (
            f"MemTotal: {total_kib} kB",
            f"MemAvailable: {available_gib * 1024**2} kB",
            f"SwapTotal: {swap_total_kib} kB",
            f"SwapFree: {swap_total_kib - swap_used_mib * 1024} kB",
        )
    )


def _synthetic_projection(cell: object, *, improvement: float) -> CellProjection:
    profile_id = str(getattr(cell, "profile_id"))
    fingerprint = str(getattr(cell, "plan_fingerprint"))
    baseline = profile_id.endswith("agent64k-low-ple-mapped-sglang")
    factor = 1.0 if baseline else improvement
    measurements = tuple(
        CaseMeasurement(
            case_id,
            (
                20.0 * factor
                if case_id == "agent64k-decode-256-c1-v1"
                else 10.0 / factor
            ),
            "higher" if case_id == "agent64k-decode-256-c1-v1" else "lower",
            2.0,
        )
        for case_id in EXPECTED_PRIMARY_CASE_IDS
    )
    return CellProjection(
        profile_id,
        fingerprint,
        measurements,
        100.0,
        10.0,
        20.0,
        0.0,
        ("--same",),
    )


class AutoresearchCampaignControllerTests(unittest.TestCase):
    def test_loader_rejects_rehashed_schedule_and_safety_tampering(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            schedule_root = Path(directory) / "schedule"
            schedule_root.mkdir()
            campaign_dir = _freeze_campaign_fixture(schedule_root)
            campaign_path = campaign_dir / "campaign.json"
            frozen = json.loads(campaign_path.read_text())
            frozen["cells"][2]["stage"] = "confirmation"
            frozen.pop("integrity_hash")
            frozen["integrity_hash"] = content_hash(frozen, 64)
            write_json(campaign_path, frozen)
            with self.assertRaisesRegex(
                CampaignPlanningError, "schedule or profile binding"
            ):
                load_frozen_campaign(campaign_dir)

            safety_root = Path(directory) / "safety"
            safety_root.mkdir()
            campaign_dir = _freeze_campaign_fixture(safety_root)
            frozen = json.loads((campaign_dir / "campaign.json").read_text())
            cell = frozen["cells"][0]
            plan_path = campaign_dir / cell["run_dir"] / "plan.json"
            plan = json.loads(plan_path.read_text())
            plan["model"]["host_safety_min_memavailable_gib"] = 1
            suite_without_case_ids = {
                **plan["suite"],
                "cases": [
                    {key: value for key, value in case.items() if key != "case_id"}
                    for case in plan["suite"]["cases"]
                ],
            }
            plan["fingerprint"] = content_hash(
                {
                    "model": plan["model"],
                    "suite": suite_without_case_ids,
                    "resolved": plan["resolved"],
                }
            )
            plan.pop("integrity_hash")
            plan["integrity_hash"] = content_hash(plan, 64)
            write_json(plan_path, plan)
            cell["plan_fingerprint"] = plan["fingerprint"]
            cell["plan_integrity_hash"] = plan["integrity_hash"]
            frozen.pop("integrity_hash")
            frozen["integrity_hash"] = content_hash(frozen, 64)
            write_json(campaign_dir / "campaign.json", frozen)
            with self.assertRaisesRegex(CampaignPlanningError, "host-safety gates"):
                load_frozen_campaign(campaign_dir)

    def test_admission_reports_time_swap_and_memory_without_execution(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign = load_frozen_campaign(
                _freeze_campaign_fixture(Path(directory))
            )
            blockers = campaign_admission(
                campaign,
                now=datetime.fromisoformat("2026-08-28T06:00:00-07:00"),
                meminfo_reader=lambda: _admission_meminfo(
                    available_gib=100, swap_used_mib=65
                ),
            )

        self.assertEqual(
            blockers,
            (
                "insufficient_time_for_pair",
                "starting_swap_above_clean_limit",
                "insufficient_preflight_memavailable",
            ),
        )

    def test_run_blocks_before_journal_or_cell_when_starting_swap_is_dirty(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            cell_calls: list[str] = []
            summary = run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=lambda: datetime.fromisoformat("2026-08-28T00:00:00-07:00"),
                meminfo_reader=lambda: _admission_meminfo(swap_used_mib=65),
                cell_runner=lambda cell: cell_calls.append(cell.cell_id),  # type: ignore[arg-type,return-value]
            )
            events_exists = (campaign_dir / "events.jsonl").exists()

        self.assertEqual(summary["status"], "blocked_environment")
        self.assertEqual(summary["blockers"], ["starting_swap_above_clean_limit"])
        self.assertEqual(cell_calls, [])
        self.assertFalse(events_exists)

    def test_controller_calibrates_confirms_promotes_and_stops_fixed_queue(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            calls: list[str] = []

            def runner(cell: object) -> CellProjection:
                calls.append(str(getattr(cell, "cell_id")))
                return _synthetic_projection(cell, improvement=1.05)

            summary = run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=lambda: datetime.fromisoformat("2026-08-28T00:00:00-07:00"),
                meminfo_reader=_admission_meminfo,
                cell_runner=runner,  # type: ignore[arg-type]
            )
            screened = run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=lambda: datetime.fromisoformat("2026-08-28T00:00:00-07:00"),
                meminfo_reader=_admission_meminfo,
                cell_runner=runner,  # type: ignore[arg-type]
            )
            promoted = run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=lambda: datetime.fromisoformat("2026-08-28T00:00:00-07:00"),
                meminfo_reader=_admission_meminfo,
                cell_runner=runner,  # type: ignore[arg-type]
            )
            replayed = run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=lambda: datetime.fromisoformat("2026-08-28T00:00:00-07:00"),
                meminfo_reader=_admission_meminfo,
                cell_runner=runner,  # type: ignore[arg-type]
            )

        self.assertEqual(summary["status"], "active")
        self.assertTrue(summary["calibration_recorded"])
        self.assertEqual(screened["status"], "active")
        self.assertEqual(promoted["status"], "complete")
        self.assertEqual(replayed, promoted)
        self.assertEqual(len(calls), 6)
        first_candidate = (
            "qwen38-flash-next-nvfp4-mtp2-agent64k-none-ple-mapped-sglang"
        )
        self.assertEqual(promoted["candidate_decisions"][first_candidate], "promote")
        self.assertEqual(promoted["next_pair_index"], 2)

    def test_equal_candidates_are_rejected_and_queue_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            calls: list[str] = []

            def runner(cell: object) -> CellProjection:
                calls.append(str(getattr(cell, "cell_id")))
                return _synthetic_projection(cell, improvement=1.0)

            summaries = [
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: datetime.fromisoformat("2026-08-28T00:00:00-07:00"),
                    meminfo_reader=_admission_meminfo,
                    cell_runner=runner,  # type: ignore[arg-type]
                )
                for _ in range(4)
            ]
            summary = summaries[-1]

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(
            [item["status"] for item in summaries],
            ["active", "active", "active", "complete"],
        )
        self.assertEqual(set(summary["candidate_decisions"].values()), {"reject"})
        self.assertEqual(summary["next_pair_index"], 3)
        self.assertEqual(len(calls), 8)

    def test_hard_ineligible_score_terminates_instead_of_deciding(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))

            def runner(cell: object) -> CellProjection:
                projection = _synthetic_projection(cell, improvement=1.05)
                if getattr(cell, "stage") == "screen" and getattr(cell, "arm") == "candidate":
                    return replace(projection, measurement_elapsed_s=1800.001)
                return projection

            calibrated = run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=lambda: datetime.fromisoformat("2026-08-28T00:00:00-07:00"),
                meminfo_reader=_admission_meminfo,
                cell_runner=runner,  # type: ignore[arg-type]
            )
            summary = run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=lambda: datetime.fromisoformat("2026-08-28T00:00:00-07:00"),
                meminfo_reader=_admission_meminfo,
                cell_runner=runner,  # type: ignore[arg-type]
            )

        self.assertEqual(calibrated["status"], "active")
        self.assertEqual(summary["status"], "terminated")
        self.assertEqual(summary["terminal_reason"], "measurement")
        self.assertEqual(summary["candidate_decisions"], {})
        self.assertEqual(summary["next_pair_index"], 1)


class AutoresearchCellProjectionTests(unittest.TestCase):
    def _write_cell(
        self,
        root: Path,
        *,
        minimum_memavailable_kib: int = 14 * 1024 * 1024,
        baseline_swap_kib: int = 64 * 1024,
        maximum_swap_growth_kib: int = 512 * 1024,
    ) -> Path:
        definition = load_campaign_definition(CAMPAIGN_PATH, workspace=ROOT)
        preview, models, suite = validate_campaign(definition)
        model_data = asdict(models[definition.baseline_id])
        suite_data = asdict(suite)
        suite_data.pop("protocol_digest", None)
        cases = [
            _canonical_case(model_data, case)
            for case in suite_data["cases"]
        ]
        frozen_suite = {**suite_data, "cases": cases}
        resolved = {"image_digest": "image@sha256:" + "a" * 64}
        fingerprint = content_hash(
            {"model": model_data, "suite": suite_data, "resolved": resolved}
        )
        plan = {
            "schema_version": 2,
            "created_at": "2026-08-28T00:00:00+00:00",
            "fingerprint": fingerprint,
            "models_manifest": "ignored",
            "suite_manifest": "ignored",
            "model": model_data,
            "suite": frozen_suite,
            "resolved": resolved,
            "host_at_plan": {},
        }
        plan["integrity_hash"] = content_hash(plan, 64)
        write_json(root / "plan.json", plan)

        case_summaries = []
        for case in cases:
            stable_id = case["id"]
            summary = {
                "case_id": case["case_id"],
                "kind": case["kind"],
                "measurement_valid": True,
                "validation_passed": True,
                "aggregate_output_tps": 30.0,
                "median_e2e_s": 20.0,
                "median_ttft_s": 0.5,
                "telemetry": {"minimum_memavailable_gib": 20.0},
            }
            if stable_id.startswith("agentic-"):
                summary.update(
                    {
                        "median_agentic_task_wall_s": 10.0,
                        "median_agentic_first_turn_ttft_s": 2.0,
                    }
                )
            if stable_id == "long-context-needle-60000-agent-c1":
                summary.update({"median_e2e_s": 20.0, "median_ttft_s": 15.0})
            case_summaries.append(summary)
        summary = {
            "status": "complete",
            "completed_cases": 9,
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
        write_json(root / "summary.json", summary)

        depth = 2
        events = [
            {"event": "run_start", "completed_cases_at_resume": []},
            {"event": "server_ready"},
            *(
                {
                    "event": "case_complete",
                    "case_id": case["case_id"],
                    "attempt_id": f"attempt-{index}",
                }
                for index, case in enumerate(cases)
            ),
            {
                "event": "sglang_spec_decode_metrics_snapshot",
                "metrics": {
                    "requested": True,
                    "method": "NEXTN",
                    "configured_max_draft_tokens": depth,
                    "accepted_tokens_per_position": {"0": 8, "1": 7},
                    "num_drafts": 10,
                    "num_accepted_tokens": 15,
                },
            },
            {"event": "measurement_complete", "elapsed_s": 100.0},
            {
                "event": "server_stopped",
                "cleanup_elapsed_s": 10.0,
            },
            {"event": "run_complete"},
        ]
        (root / "events.jsonl").write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
        )
        swap_total = 16 * 1024 * 1024
        telemetry = [
            {
                "memavailable_kib": 20 * 1024 * 1024,
                "swaptotal_kib": swap_total,
                "swapfree_kib": swap_total - baseline_swap_kib,
            },
            {
                "memavailable_kib": minimum_memavailable_kib,
                "swaptotal_kib": swap_total,
                "swapfree_kib": (
                    swap_total - baseline_swap_kib - maximum_swap_growth_kib
                ),
            },
        ]
        (root / "telemetry.jsonl").write_text(
            "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in telemetry)
        )
        self.assertEqual(preview.suite_id, suite.id)
        return root

    def test_projection_accepts_inclusive_memory_swap_and_start_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            projection = project_completed_cell(
                self._write_cell(Path(directory))
            )

        self.assertEqual(projection.minimum_memavailable_gib, 14.0)
        self.assertEqual(projection.maximum_swap_growth_mib, 512.0)
        self.assertEqual(
            tuple(item.case_id for item in projection.measurements),
            EXPECTED_PRIMARY_CASE_IDS,
        )
        self.assertEqual(projection.measurement_elapsed_s, 100.0)
        self.assertEqual(projection.cleanup_elapsed_s, 10.0)

    def test_projection_rejects_each_safety_boundary_breach(self) -> None:
        fixtures = (
            (
                {"minimum_memavailable_kib": 14 * 1024 * 1024 - 1},
                "memory_pressure",
            ),
            (
                {"maximum_swap_growth_kib": 512 * 1024 + 1},
                "swap_pressure",
            ),
            ({"baseline_swap_kib": 64 * 1024 + 1}, "swap_pressure"),
        )
        for kwargs, failure_kind in fixtures:
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                run_dir = self._write_cell(Path(directory), **kwargs)
                with self.assertRaises(CellProjectionError) as raised:
                    project_completed_cell(run_dir)
                self.assertEqual(raised.exception.failure_kind, failure_kind)

    def test_projection_classifies_durable_terminal_artifacts(self) -> None:
        fixtures = (
            (
                {
                    "event": "host_safety_breach",
                    "code": "memavailable_below_minimum",
                },
                "memory_pressure",
            ),
            (
                {
                    "event": "host_safety_breach",
                    "code": "swap_growth_above_maximum",
                },
                "swap_pressure",
            ),
            ({"event": "cleanup_failed"}, "cleanup_breach"),
        )
        for artifact, expected in fixtures:
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                run_dir = self._write_cell(Path(directory))
                events_path = run_dir / "events.jsonl"
                events = [
                    json.loads(line) for line in events_path.read_text().splitlines()
                ]
                events.insert(-1, artifact)
                events_path.write_text(
                    "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
                )
                summary = json.loads((run_dir / "summary.json").read_text())
                summary["status"] = "aborted"
                write_json(run_dir / "summary.json", summary)

                with self.assertRaises(CellProjectionError) as raised:
                    project_completed_cell(run_dir)
                self.assertEqual(raised.exception.failure_kind, expected)

    def test_projection_rejects_resume_bad_audit_and_tamper(self) -> None:
        mutations = ("resume", "audit", "plan")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                run_dir = self._write_cell(Path(directory))
                if mutation == "plan":
                    plan = json.loads((run_dir / "plan.json").read_text())
                    plan["model"]["served_name"] = "tampered"
                    write_json(run_dir / "plan.json", plan)
                else:
                    events = [
                        json.loads(line)
                        for line in (run_dir / "events.jsonl").read_text().splitlines()
                    ]
                    if mutation == "resume":
                        events[0]["completed_cases_at_resume"] = ["old-case"]
                    else:
                        audit = next(
                            event
                            for event in events
                            if event["event"] == "sglang_spec_decode_metrics_snapshot"
                        )
                        audit["metrics"]["num_accepted_tokens"] = 0
                    (run_dir / "events.jsonl").write_text(
                        "".join(
                            json.dumps(event, sort_keys=True) + "\n"
                            for event in events
                        )
                    )
                with self.assertRaises(CellProjectionError):
                    project_completed_cell(run_dir)

    def test_observation_uses_task_wall_inverse_decode_rate_and_ttft_guardrail(
        self,
    ) -> None:
        policy = CampaignPolicy(
            primary_case_ids=EXPECTED_PRIMARY_CASE_IDS,
            allowed_axes=EXPECTED_AXES,
        )
        champion_measurements = tuple(
            CaseMeasurement(
                case_id,
                10.0 if case_id != "agent64k-decode-256-c1-v1" else 20.0,
                "higher" if case_id == "agent64k-decode-256-c1-v1" else "lower",
                2.0,
            )
            for case_id in EXPECTED_PRIMARY_CASE_IDS
        )
        candidate_measurements = tuple(
            CaseMeasurement(
                case_id,
                8.0 if case_id != "agent64k-decode-256-c1-v1" else 25.0,
                "higher" if case_id == "agent64k-decode-256-c1-v1" else "lower",
                2.1,
            )
            for case_id in EXPECTED_PRIMARY_CASE_IDS
        )
        champion = CellProjection(
            "champion",
            "a" * 16,
            champion_measurements,
            100.0,
            10.0,
            18.0,
            10.0,
            ("--a",),
        )
        candidate = CellProjection(
            "candidate",
            "b" * 16,
            candidate_measurements,
            90.0,
            8.0,
            19.5,
            9.0,
            ("--a",),
        )

        observation = observation_from_cells(
            policy,
            pair_index=1,
            champion=champion,
            candidate=candidate,
            audit_reserve_remaining_s=900.0,
        )

        self.assertTrue(
            all(math.isclose(ratio, 1.25) for ratio in observation.primary_speed_ratios)
        )
        self.assertTrue(math.isclose(observation.median_ttft_ratio, 1.05))
        self.assertEqual(observation.timing.cell_elapsed_s, (90.0, 100.0))
        self.assertEqual(observation.timing.pair_elapsed_s, 190.0)
        self.assertEqual(observation.timing.cleanup_elapsed_s, 10.0)
        self.assertEqual(
            observation.simplification.minimum_memavailable_gain_gib, 1.5
        )


if __name__ == "__main__":
    unittest.main()
