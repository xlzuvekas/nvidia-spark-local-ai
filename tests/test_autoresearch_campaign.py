from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

from bench.autoresearch_campaign import (
    CampaignPlanningError,
    CaseMeasurement,
    CellProjection,
    CellProjectionError,
    EXPECTED_AXES,
    EXPECTED_CASE_IDS,
    EXPECTED_PRIMARY_CASE_IDS,
    _cell_specs,
    _normalized_flags,
    freeze_campaign,
    campaign_admission,
    load_frozen_campaign,
    load_campaign_definition,
    observation_from_cells,
    project_completed_cell,
    run_frozen_cell,
    run_campaign,
    semantic_config,
    validate_campaign,
)
from bench.autoresearch import CampaignPolicy
from bench.autoresearch_worker import (
    WorkerCleanupResult,
    WorkerLifecycleError,
    WorkerRunResult,
)
from bench.journal import content_hash, write_json
from bench.manifest import model_spec_to_dict
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
        reasoning_candidate = models[preview.proposals[0].candidate_id]
        self.assertLess(
            set(_normalized_flags(model_spec_to_dict(reasoning_candidate))),
            set(
                _normalized_flags(
                    model_spec_to_dict(models[definition.baseline_id])
                )
            ),
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

    def test_definition_rejects_manifest_only_protocol_rewrites(self) -> None:
        source = (
            CAMPAIGN_PATH.read_text()
            .replace('../models.toml', '../../models.toml')
            .replace('../suites/', '../../suites/')
        )
        with tempfile.TemporaryDirectory(dir=CAMPAIGN_PATH.parent) as directory:
            root = Path(directory)
            changed_id = root / "changed-id.toml"
            changed_id.write_text(
                source.replace(
                    "qwen38-flash-next-single-user-autoresearch-2026-08-28",
                    "qwen38-flash-next-single-user-autoresearch-rewritten",
                )
            )
            with self.assertRaisesRegex(CampaignPlanningError, "campaign ID"):
                load_campaign_definition(changed_id, workspace=ROOT)

            changed_candidate = root / "changed-candidate.toml"
            changed_candidate.write_text(
                source.replace(
                    "qwen38-flash-next-nvfp4-mtp3-agent64k-low-ple-mapped-sglang",
                    "qwen38-flash-next-nvfp4-mtp2-agent64k-low-ple-mapped-sglang",
                )
            )
            with self.assertRaisesRegex(CampaignPlanningError, "candidate queue"):
                load_campaign_definition(changed_candidate, workspace=ROOT)

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
                    "schema_version": 2,
                    "fingerprint": fingerprint,
                    "integrity_hash": "a" * 64,
                    "run_nonce": f"{len(calls):032x}",
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
    plan_ordinal = 0

    def fake_create_plan(**kwargs: object) -> Path:
        nonlocal plan_ordinal
        plan_ordinal += 1
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
            "run_nonce": f"{plan_ordinal:032x}",
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


def _worker_result(*, outcome: str, timed_out: bool) -> WorkerRunResult:
    cleanup = WorkerCleanupResult(
        outcome=outcome,  # type: ignore[arg-type]
        identity=None,
        return_code=-2 if timed_out else 0,
        sigint_sent=timed_out,
        sigkill_sent=outcome == "killed",
        process_lookup_race=False,
        state_removed=True,
    )
    return WorkerRunResult(
        return_code=cleanup.return_code or 0,
        timed_out=timed_out,
        cleanup=cleanup,
    )


class AutoresearchFrozenCellWorkerTests(unittest.TestCase):
    def test_timeout_uses_owned_worker_and_exact_sglang_recovery(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign = load_frozen_campaign(
                _freeze_campaign_fixture(Path(directory))
            )
            cell = campaign.cells[0]
            captured: dict[str, object] = {}

            def worker(command: list[str], **kwargs: object) -> WorkerRunResult:
                captured["command"] = command
                captured.update(kwargs)
                return _worker_result(outcome="interrupted", timed_out=True)

            ticks = iter((10.0, 11.0))
            with patch(
                "bench.autoresearch_campaign._recover_cell",
                return_value="already_absent",
            ) as recover:
                with self.assertRaises(CellProjectionError) as raised:
                    run_frozen_cell(
                        cell,
                        workspace=ROOT,
                        cell_timeout_s=1_800,
                        cleanup_timeout_s=120,
                        worker_runner=worker,
                        monotonic=lambda: next(ticks),
                    )

            log_path = cell.run_dir / "controller.log"
            self.assertEqual(stat.S_IMODE(log_path.stat().st_mode), 0o600)

        self.assertEqual(raised.exception.failure_kind, "measurement")
        self.assertEqual(captured["cell_run_dir"], cell.run_dir)
        self.assertEqual(captured["run_nonce"], cell.run_nonce)
        self.assertEqual(captured["timeout_s"], 1_800)
        self.assertEqual(captured["interrupt_grace_s"], 120)
        self.assertEqual(
            captured["command"],
            [
                sys.executable,
                str((ROOT / "sparkbench.py").resolve()),
                "run",
                str(cell.run_dir),
                "--fail-fast",
            ],
        )
        recover.assert_called_once_with(cell)

    def test_forced_worker_kill_is_a_cleanup_breach(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign = load_frozen_campaign(
                _freeze_campaign_fixture(Path(directory))
            )
            cell = campaign.cells[0]
            with patch(
                "bench.autoresearch_campaign._recover_cell",
                return_value="already_absent",
            ):
                with self.assertRaises(CellProjectionError) as raised:
                    run_frozen_cell(
                        cell,
                        workspace=ROOT,
                        cell_timeout_s=1_800,
                        cleanup_timeout_s=120,
                        worker_runner=lambda *_args, **_kwargs: _worker_result(
                            outcome="killed", timed_out=True
                        ),
                    )

        self.assertEqual(raised.exception.failure_kind, "cleanup_breach")

    def test_worker_identity_failure_is_ownership_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign = load_frozen_campaign(
                _freeze_campaign_fixture(Path(directory))
            )
            cell = campaign.cells[0]

            def fail_worker(*_args: object, **_kwargs: object) -> WorkerRunResult:
                raise WorkerLifecycleError(
                    "identity_mismatch", "synthetic worker identity mismatch"
                )

            with patch("bench.autoresearch_campaign._recover_cell") as recover:
                with self.assertRaises(CellProjectionError) as raised:
                    run_frozen_cell(
                        cell,
                        workspace=ROOT,
                        cell_timeout_s=1_800,
                        cleanup_timeout_s=120,
                        worker_runner=fail_worker,
                    )

        self.assertEqual(raised.exception.failure_kind, "ownership_ambiguity")
        recover.assert_not_called()


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

    def test_loader_rejects_rehashed_cutoff_and_profile_content(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            cutoff_root = Path(directory) / "cutoff"
            cutoff_root.mkdir()
            campaign_dir = _freeze_campaign_fixture(cutoff_root)
            campaign_path = campaign_dir / "campaign.json"
            frozen = json.loads(campaign_path.read_text())
            frozen["preview"]["cutoff"] = "2099-01-01T07:00:00-07:00"
            frozen["preview_digest"] = content_hash(frozen["preview"], 64)
            frozen.pop("integrity_hash")
            frozen["integrity_hash"] = content_hash(frozen, 64)
            write_json(campaign_path, frozen)
            with self.assertRaisesRegex(CampaignPlanningError, "cutoff changed"):
                load_frozen_campaign(campaign_dir)

            profile_root = Path(directory) / "profile"
            profile_root.mkdir()
            campaign_dir = _freeze_campaign_fixture(profile_root)
            campaign_path = campaign_dir / "campaign.json"
            frozen = json.loads(campaign_path.read_text())
            candidate_id = frozen["preview"]["proposals"][0]["candidate_id"]
            for cell in frozen["cells"]:
                if cell["profile_id"] != candidate_id:
                    continue
                plan_path = campaign_dir / cell["run_dir"] / "plan.json"
                plan = json.loads(plan_path.read_text())
                plan["model"]["description"] += " rewritten"
                plan["suite"]["cases"] = [
                    _canonical_case(
                        plan["model"],
                        {
                            key: value
                            for key, value in case.items()
                            if key != "case_id"
                        },
                    )
                    for case in plan["suite"]["cases"]
                ]
                suite_basis = {
                    **plan["suite"],
                    "cases": [
                        {key: value for key, value in case.items() if key != "case_id"}
                        for case in plan["suite"]["cases"]
                    ],
                }
                plan["fingerprint"] = content_hash(
                    {
                        "model": plan["model"],
                        "suite": suite_basis,
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
            write_json(campaign_path, frozen)
            with self.assertRaisesRegex(CampaignPlanningError, "model profile"):
                load_frozen_campaign(campaign_dir)

            safety_root = Path(directory) / "safety"
            safety_root.mkdir()
            campaign_dir = _freeze_campaign_fixture(safety_root)
            frozen = json.loads((campaign_dir / "campaign.json").read_text())
            cell = frozen["cells"][0]
            plan_path = campaign_dir / cell["run_dir"] / "plan.json"
            plan = json.loads(plan_path.read_text())
            plan["model"]["host_safety_min_memavailable_gib"] = 1
            plan["suite"]["cases"] = [
                _canonical_case(
                    plan["model"],
                    {
                        key: value
                        for key, value in case.items()
                        if key != "case_id"
                    },
                )
                for case in plan["suite"]["cases"]
            ]
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

    def test_run_blocks_before_journal_when_executable_harness_changed(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            cell_calls: list[str] = []
            summary = run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=lambda: datetime.fromisoformat("2026-08-28T00:00:00-07:00"),
                meminfo_reader=_admission_meminfo,
                harness_identity_reader=lambda _workspace: ("0" * 64, 1),
                cell_runner=lambda cell: cell_calls.append(cell.cell_id),  # type: ignore[arg-type,return-value]
            )

        self.assertEqual(summary["status"], "blocked_environment")
        self.assertEqual(summary["blockers"], ["harness_code_changed"])
        self.assertEqual(cell_calls, [])

    def test_active_campaign_admission_blocker_is_durably_terminal(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))

            def interrupt(_cell: object) -> CellProjection:
                raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: datetime.fromisoformat(
                        "2026-08-28T00:00:00-07:00"
                    ),
                    meminfo_reader=_admission_meminfo,
                    cell_runner=interrupt,
                )
            summary = run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=lambda: datetime.fromisoformat("2026-08-28T00:01:00-07:00"),
                meminfo_reader=lambda: _admission_meminfo(swap_used_mib=65),
                cell_runner=lambda _cell: self.fail("blocked resume launched a cell"),
            )

        self.assertEqual(summary["status"], "terminated")
        self.assertEqual(summary["terminal_reason"], "swap_pressure")

    def test_harness_identity_is_rechecked_immediately_before_cell(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            frozen = load_frozen_campaign(campaign_dir)
            identities = iter(
                [
                    (frozen.harness_tree_sha256, frozen.harness_file_count),
                    ("0" * 64, frozen.harness_file_count),
                ]
            )
            calls: list[str] = []
            summary = run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=lambda: datetime.fromisoformat("2026-08-28T00:00:00-07:00"),
                meminfo_reader=_admission_meminfo,
                harness_identity_reader=lambda _workspace: next(identities),
                cell_runner=lambda cell: calls.append(cell.cell_id),  # type: ignore[arg-type,return-value]
            )

        self.assertEqual(summary["status"], "terminated")
        self.assertEqual(summary["terminal_reason"], "audit")
        self.assertEqual(calls, [])

    def test_campaign_lock_rejects_links_and_an_active_controller(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            lock_path = campaign_dir / ".autoresearch.lock"
            protected = Path(directory) / "protected"
            protected.write_text("do-not-touch")
            lock_path.symlink_to(protected)
            with self.assertRaisesRegex(CampaignPlanningError, "lock path is unsafe"):
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: datetime.fromisoformat("2026-08-28T00:00:00-07:00"),
                    meminfo_reader=_admission_meminfo,
                )
            self.assertEqual(protected.read_text(), "do-not-touch")

            lock_path.unlink()
            os.link(protected, lock_path)
            with self.assertRaisesRegex(CampaignPlanningError, "single-link"):
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: datetime.fromisoformat("2026-08-28T00:00:00-07:00"),
                    meminfo_reader=_admission_meminfo,
                )
            self.assertEqual(protected.read_text(), "do-not-touch")

            lock_path.unlink()
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(
                    CampaignPlanningError, "another autoresearch controller"
                ):
                    run_campaign(
                        campaign_dir,
                        workspace=ROOT,
                        now=lambda: datetime.fromisoformat(
                            "2026-08-28T00:00:00-07:00"
                        ),
                        meminfo_reader=_admission_meminfo,
                    )
            finally:
                os.close(descriptor)
            self.assertFalse((campaign_dir / "events.jsonl").exists())

    def test_controller_calibrates_confirms_promotes_and_stops_fixed_queue(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            calls: list[str] = []

            def runner(cell: object) -> CellProjection:
                calls.append(str(getattr(cell, "cell_id")))
                projection = _synthetic_projection(cell, improvement=1.05)
                if "agent64k-none" in str(getattr(cell, "profile_id")):
                    return replace(projection, normalized_flags=())
                return projection

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
        self.assertEqual(
            promoted["candidate_decisions"][first_candidate],
            "promote_simplification",
        )
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
