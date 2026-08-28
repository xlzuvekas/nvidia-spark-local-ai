from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from typing import Callable, Iterator

from bench.autoresearch_campaign import (
    CampaignPlanningError,
    CaseMeasurement,
    CellProjection,
    CellProjectionError,
    EXPECTED_AXES,
    EXPECTED_CASE_IDS,
    EXPECTED_PRIMARY_CASE_IDS,
    _CellLifecycleProgress,
    _cell_specs,
    _normalized_flags,
    campaign_evidence_snapshot,
    freeze_campaign,
    campaign_admission,
    load_frozen_campaign,
    load_campaign_definition,
    observation_from_cells,
    project_completed_cell,
    run_frozen_cell,
    run_campaign,
    semantic_config,
    summarize_campaign,
    validate_campaign,
)
from bench.autoresearch import CampaignPolicy
from bench.autoresearch_admission import AdmissionJournalError
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


_execution_admission_patcher = None


def setUpModule() -> None:
    """Keep historical controller mechanics independent of live retirement."""

    global _execution_admission_patcher
    _execution_admission_patcher = patch(
        "bench.execution_admission.RETIRED_SGLANG_SOURCE_OVERLAY_DIGESTS",
        frozenset(),
    )
    _execution_admission_patcher.start()


def tearDownModule() -> None:
    assert _execution_admission_patcher is not None
    _execution_admission_patcher.stop()


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

    def test_schema_three_requires_admission_journal_and_schema_two_loads(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            root = Path(directory)
            invalid_root = root / "invalid"
            invalid_root.mkdir()
            invalid_dir = _freeze_campaign_fixture(invalid_root)
            frozen_path = invalid_dir / "campaign.json"
            frozen = json.loads(frozen_path.read_text())
            self.assertEqual(frozen["schema_version"], 3)
            self.assertIs(frozen["admission_journal_required"], True)
            frozen["admission_journal_required"] = False
            frozen.pop("integrity_hash")
            frozen["integrity_hash"] = content_hash(frozen, 64)
            write_json(frozen_path, frozen)
            with self.assertRaisesRegex(
                CampaignPlanningError,
                "admission-journal requirement changed",
            ):
                load_frozen_campaign(invalid_dir)

            legacy_root = root / "legacy"
            legacy_root.mkdir()
            legacy_dir = _freeze_campaign_fixture(legacy_root)
            _downgrade_pre_admission_campaign_fixture(legacy_dir)
            legacy = load_frozen_campaign(legacy_dir)

        self.assertFalse(legacy.admission_journal_required)


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


def _downgrade_pre_admission_campaign_fixture(campaign_dir: Path) -> None:
    """Model a schema-2 campaign whose raw work predates admission journals."""

    frozen_path = campaign_dir / "campaign.json"
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("schema_version") != 3:
        raise AssertionError("fixture is not the current frozen campaign schema")
    frozen["schema_version"] = 2
    frozen.pop("admission_journal_required")
    frozen.pop("integrity_hash")
    frozen["integrity_hash"] = content_hash(frozen, 64)
    write_json(frozen_path, frozen)


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


def _admission_records(campaign_dir: Path) -> tuple[dict[str, object], ...]:
    path = campaign_dir / "admissions.jsonl"
    return tuple(json.loads(line) for line in path.read_text().splitlines())


def _campaign_tree_snapshot(root: Path) -> tuple[tuple[object, ...], ...]:
    """Capture topology, stable metadata, and content without recording atime."""

    rows: list[tuple[object, ...]] = []
    for path in (root, *sorted(root.rglob("*"))):
        metadata = os.lstat(path)
        relative = "." if path == root else path.relative_to(root).as_posix()
        if stat.S_ISREG(metadata.st_mode):
            kind = "file"
            content_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        elif stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
            content_sha256 = ""
        elif stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
            content_sha256 = os.readlink(path)
        else:
            kind = "other"
            content_sha256 = ""
        rows.append(
            (
                relative,
                kind,
                stat.S_IMODE(metadata.st_mode),
                metadata.st_ino,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                content_sha256,
            )
        )
    return tuple(rows)


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


@contextmanager
def _synthetic_projection_boundary(
    projector: Callable[[object], CellProjection],
    *,
    stopped_at: datetime,
    audit_reserve_s: float,
) -> Iterator[None]:
    """Isolate controller-policy tests from the raw replay boundary."""

    def calibration_raw_topology(
        campaign: object,
    ) -> tuple[dict[str, object], dict[str, CellProjection]]:
        cells = getattr(campaign, "cells_for")(
            candidate_id="control", stage="calibration"
        )
        campaign_dir = Path(getattr(campaign, "campaign_dir"))
        if not (campaign_dir / "calibration.json").exists():
            return cells, {}
        return cells, {
            arm: projector(cells[arm]) for arm in ("control_a", "control_b")
        }

    def durable_audit_reserve(
        campaign: object, projections: dict[str, CellProjection]
    ) -> float:
        profile_ids = {projection.profile_id for projection in projections.values()}
        if len(profile_ids) == 1:
            policy = getattr(campaign, "policy")
            return max(audit_reserve_s, float(getattr(policy, "audit_reserve_s")))
        return audit_reserve_s

    with (
        patch(
            "bench.autoresearch_campaign._project_frozen_cell",
            side_effect=projector,
        ),
        patch(
            "bench.autoresearch_campaign._calibration_raw_topology",
            side_effect=calibration_raw_topology,
        ),
        patch(
            "bench.autoresearch_campaign._server_stopped_at",
            return_value=stopped_at,
        ),
        patch(
            "bench.autoresearch_campaign._run_started_at",
            return_value=stopped_at,
        ),
        patch(
            "bench.autoresearch_campaign._durable_pair_audit_reserve_s",
            side_effect=durable_audit_reserve,
        ),
        patch(
            "bench.autoresearch_campaign._validate_completed_pair_gap",
            return_value=0.0,
        ),
        patch(
            "bench.autoresearch_campaign._checkpoint_gate_for_campaign",
            return_value=SimpleNamespace(ready=True),
        ),
    ):
        yield


def _worker_result(
    *, outcome: str, timed_out: bool, timeout_phase: str | None = None
) -> WorkerRunResult:
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
        timeout_phase=timeout_phase,  # type: ignore[arg-type]
    )


class AutoresearchCampaignEvidenceSnapshotTests(unittest.TestCase):
    def _cutoff_campaign(self, root: Path) -> tuple[Path, object]:
        campaign_dir = _freeze_campaign_fixture(root)
        frozen = load_frozen_campaign(campaign_dir)
        summary = run_campaign(
            campaign_dir,
            workspace=ROOT,
            now=lambda: frozen.cutoff + timedelta(seconds=1),
            meminfo_reader=lambda: _admission_meminfo(),
            harness_identity_reader=lambda _root: (
                frozen.harness_tree_sha256,
                frozen.harness_file_count,
            ),
            cell_runner=lambda _cell: self.fail("cutoff campaign launched a cell"),
        )
        self.assertEqual("expired", summary["status"])
        return campaign_dir, load_frozen_campaign(campaign_dir)

    def test_schema_three_snapshot_is_deterministic_and_never_writes(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir, frozen = self._cutoff_campaign(Path(directory))
            before = _campaign_tree_snapshot(campaign_dir)
            with patch(
                "bench.autoresearch_campaign.write_json",
                side_effect=AssertionError("evidence snapshot must be read-only"),
            ):
                first = campaign_evidence_snapshot(campaign_dir)
                middle = _campaign_tree_snapshot(campaign_dir)
                second = campaign_evidence_snapshot(campaign_dir)
            after = _campaign_tree_snapshot(campaign_dir)

        self.assertEqual(first, second)
        self.assertEqual(before, middle)
        self.assertEqual(before, after)
        self.assertEqual(1, first["snapshot_schema_version"])
        self.assertEqual(3, first["frozen_campaign_schema_version"])
        self.assertTrue(first["admission_journal_required"])
        self.assertEqual("required", first["provenance_mode"])
        self.assertEqual(frozen.campaign_id, first["campaign_id"])
        self.assertEqual(
            frozen.integrity_hash, first["campaign_integrity_sha256"]
        )
        self.assertEqual(frozen.preview_digest, first["preview_sha256"])
        self.assertEqual(frozen.policy_digest, first["policy_sha256"])
        self.assertEqual(frozen.harness_tree_sha256, first["harness_tree_sha256"])
        self.assertEqual(frozen.harness_file_count, first["harness_file_count"])
        self.assertEqual(14, first["planned_cell_count"])
        self.assertEqual(
            [proposal.candidate_id for proposal in frozen.proposals],
            [proposal["candidate_id"] for proposal in first["proposals"]],
        )
        self.assertEqual({}, first["controller_event_counts"])
        self.assertEqual("expired", first["summary"]["status"])
        self.assertEqual(1, first["summary"]["admission_count"])
        self.assertEqual(1, len(first["admissions"]))
        admission = first["admissions"][0]
        self.assertEqual("calibration", admission["target_kind"])
        self.assertEqual("cutoff", admission["outcome"])
        self.assertEqual(["insufficient_time_for_pair"], admission["blockers"])
        self.assertEqual(
            admission["record_sha256"],
            first["summary"]["last_admission_sha256"],
        )

    def test_snapshot_fails_closed_while_controller_lock_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir, _frozen = self._cutoff_campaign(Path(directory))
            lock_path = campaign_dir / ".autoresearch.lock"
            descriptor = os.open(
                lock_path,
                os.O_RDWR | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(
                    CampaignPlanningError,
                    "another autoresearch controller holds the campaign lock",
                ):
                    campaign_evidence_snapshot(campaign_dir)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)


class AutoresearchFrozenCellWorkerTests(unittest.TestCase):
    def test_failure_cleanup_stop_marker_advances_to_finalization(self) -> None:
        timestamp = "2026-08-28T00:00:00+00:00"
        fingerprint = "0123456789abcdef"
        nonce = "1234567890abcdef1234567890abcdef"
        with tempfile.TemporaryDirectory() as directory:
            events_path = Path(directory) / "events.jsonl"
            events = (
                {
                    "event": "run_start",
                    "timestamp": timestamp,
                    "completed_cases_at_resume": [],
                    "plan_fingerprint": fingerprint,
                    "run_nonce": nonce,
                },
                {
                    "event": "measurement_started",
                    "timestamp": timestamp,
                    "monotonic_ns": 1_000_000_000,
                    "plan_fingerprint": fingerprint,
                    "run_nonce": nonce,
                },
                {
                    "event": "server_stopped",
                    "timestamp": timestamp,
                    "backend": "sglang",
                    "cleanup_elapsed_s": 10.0,
                    "monotonic_ns": 11_000_000_000,
                },
                {
                    "event": "run_aborted",
                    "timestamp": timestamp,
                    "stage": "host_safety",
                },
            )
            events_path.write_text(
                "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
            )
            probe = _CellLifecycleProgress(
                events_path=events_path,
                plan_fingerprint=fingerprint,
                run_nonce=nonce,
                measurement_timeout_s=1_800,
                cleanup_timeout_s=120,
            )

            progress = probe()

        self.assertIsNotNone(progress)
        self.assertEqual(progress.phase, "finalization")  # type: ignore[union-attr]

    def test_progress_probe_detects_in_place_marker_rewrite(self) -> None:
        timestamp = "2026-08-28T00:00:00+00:00"
        fingerprint = "0123456789abcdef"
        nonce = "1234567890abcdef1234567890abcdef"
        with tempfile.TemporaryDirectory() as directory:
            events_path = Path(directory) / "events.jsonl"
            events = [
                {
                    "event": "run_start",
                    "timestamp": timestamp,
                    "completed_cases_at_resume": [],
                    "plan_fingerprint": fingerprint,
                    "run_nonce": nonce,
                },
                {
                    "event": "measurement_started",
                    "timestamp": timestamp,
                    "monotonic_ns": 1_000_000_000,
                    "plan_fingerprint": fingerprint,
                    "run_nonce": nonce,
                },
            ]
            events_path.write_text(
                "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
            )
            probe = _CellLifecycleProgress(
                events_path=events_path,
                plan_fingerprint=fingerprint,
                run_nonce=nonce,
                measurement_timeout_s=1_800,
                cleanup_timeout_s=120,
            )
            probe()
            events[1]["monotonic_ns"] = 2_000_000_000
            events_path.write_text(
                "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
            )

            with self.assertRaisesRegex(
                WorkerLifecycleError, "marker changed in place"
            ):
                probe()

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
        self.assertTrue(callable(captured["progress_probe"]))
        self.assertEqual(captured["start_marker_timeout_s"], 30)
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

    def test_cleanup_phase_timeout_is_a_breach_without_forced_kill(self) -> None:
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
                            outcome="interrupted",
                            timed_out=True,
                            timeout_phase="cleanup",
                        ),
                    )

        self.assertEqual(raised.exception.failure_kind, "cleanup_breach")

    def test_zero_exit_without_terminal_markers_recovers_exact_server(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign = load_frozen_campaign(
                _freeze_campaign_fixture(Path(directory))
            )
            cell = campaign.cells[0]
            with patch(
                "bench.autoresearch_campaign._recover_cell",
                return_value="already_absent",
            ) as recover:
                with self.assertRaises(CellProjectionError):
                    run_frozen_cell(
                        cell,
                        workspace=ROOT,
                        cell_timeout_s=1_800,
                        cleanup_timeout_s=120,
                        worker_runner=lambda *_args, **_kwargs: _worker_result(
                            outcome="completed", timed_out=False
                        ),
                    )

        recover.assert_called_once_with(cell)

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
        recover.assert_called_once_with(cell)


class AutoresearchCampaignControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        container_recovery = patch(
            "bench.autoresearch_campaign._recover_cell",
            return_value="already_absent",
        )
        container_recovery.start()
        self.addCleanup(container_recovery.stop)

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
            admissions = _admission_records(campaign_dir)
            admission_mode = stat.S_IMODE(
                (campaign_dir / "admissions.jsonl").stat().st_mode
            )

        self.assertEqual(summary["status"], "blocked_environment")
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["controller_status"], "planned")
        self.assertEqual(summary["blockers"], ["starting_swap_above_clean_limit"])
        self.assertEqual(summary["admission_count"], 1)
        self.assertEqual(admission_mode, 0o600)
        self.assertEqual(admissions[0]["target_kind"], "calibration")
        self.assertEqual(admissions[0]["outcome"], "blocked_environment")
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
            admissions = _admission_records(campaign_dir)

        self.assertEqual(summary["status"], "blocked_environment")
        self.assertEqual(summary["blockers"], ["harness_code_changed"])
        self.assertEqual(admissions[0]["blockers"], ["harness_code_changed"])
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
            admissions = _admission_records(campaign_dir)

        self.assertEqual(summary["status"], "terminated")
        self.assertEqual(summary["terminal_reason"], "swap_pressure")
        self.assertEqual(summary["controller_status"], "terminated")
        self.assertEqual(summary["admission_count"], 2)
        self.assertEqual(
            tuple(record["outcome"] for record in admissions),
            ("admitted", "blocked_environment"),
        )

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

    def test_fresh_cutoff_is_expired_but_mixed_safety_denial_is_blocked(self) -> None:
        cutoff_time = datetime.fromisoformat("2026-08-28T05:37:50.001-07:00")
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            root = Path(directory)
            expired_root = root / "expired"
            expired_root.mkdir()
            expired_dir = _freeze_campaign_fixture(expired_root)
            expired = run_campaign(
                expired_dir,
                workspace=ROOT,
                now=lambda: cutoff_time,
                meminfo_reader=_admission_meminfo,
                cell_runner=lambda _cell: self.fail("cutoff launched a cell"),
            )
            expired_admissions = _admission_records(expired_dir)

            mixed_root = root / "mixed"
            mixed_root.mkdir()
            mixed_dir = _freeze_campaign_fixture(mixed_root)
            mixed = run_campaign(
                mixed_dir,
                workspace=ROOT,
                now=lambda: cutoff_time,
                meminfo_reader=lambda: _admission_meminfo(swap_used_mib=65),
                cell_runner=lambda _cell: self.fail("mixed denial launched a cell"),
            )
            mixed_admissions = _admission_records(mixed_dir)

        self.assertEqual(expired["status"], "expired")
        self.assertEqual(expired["controller_status"], "planned")
        self.assertEqual(expired["terminal_reason"], None)
        self.assertEqual(expired_admissions[0]["outcome"], "cutoff")
        self.assertEqual(
            expired_admissions[0]["blockers"], ["insufficient_time_for_pair"]
        )
        self.assertEqual(mixed["status"], "blocked_environment")
        self.assertEqual(
            mixed_admissions[0]["blockers"],
            ["insufficient_time_for_pair", "starting_swap_above_clean_limit"],
        )

    def test_active_cutoff_is_terminal_and_effectively_expired(self) -> None:
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
                now=lambda: datetime.fromisoformat(
                    "2026-08-28T05:37:50.001-07:00"
                ),
                meminfo_reader=_admission_meminfo,
                cell_runner=lambda _cell: self.fail("active cutoff launched a cell"),
            )

        self.assertEqual(summary["status"], "expired")
        self.assertEqual(summary["controller_status"], "terminated")
        self.assertEqual(summary["terminal_reason"], "cutoff")
        self.assertEqual(summary["admission_count"], 2)

    def test_public_summary_preserves_denial_without_a_new_observation(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            denied = run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=lambda: datetime.fromisoformat("2026-08-28T00:00:00-07:00"),
                meminfo_reader=lambda: _admission_meminfo(swap_used_mib=65),
                cell_runner=lambda _cell: self.fail("denial launched a cell"),
            )
            before = (campaign_dir / "admissions.jsonl").read_bytes()
            summarized = summarize_campaign(campaign_dir)
            after = (campaign_dir / "admissions.jsonl").read_bytes()

        self.assertEqual(summarized, denied)
        self.assertEqual(after, before)

    def test_malformed_admission_chain_prevents_resume_and_cell_launch(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=lambda: datetime.fromisoformat("2026-08-28T00:00:00-07:00"),
                meminfo_reader=lambda: _admission_meminfo(swap_used_mib=65),
                cell_runner=lambda _cell: self.fail("denial launched a cell"),
            )
            journal_path = campaign_dir / "admissions.jsonl"
            journal_path.write_bytes(journal_path.read_bytes() + b"torn")
            calls: list[str] = []
            with self.assertRaisesRegex(
                CampaignPlanningError, "admission journal is unsafe or malformed"
            ):
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: datetime.fromisoformat(
                        "2026-08-28T00:01:00-07:00"
                    ),
                    meminfo_reader=_admission_meminfo,
                    cell_runner=lambda cell: calls.append(str(cell.cell_id)),  # type: ignore[union-attr,return-value]
                )

        self.assertEqual(calls, [])

    def test_rehashed_wrong_admission_target_fails_history_validation(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            campaign = load_frozen_campaign(campaign_dir)
            run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=lambda: datetime.fromisoformat("2026-08-28T00:00:00-07:00"),
                meminfo_reader=lambda: _admission_meminfo(swap_used_mib=65),
                cell_runner=lambda _cell: self.fail("denial launched a cell"),
            )
            admission_path = campaign_dir / "admissions.jsonl"
            record = json.loads(admission_path.read_text())
            record["target_kind"] = "screen"
            record["candidate_id"] = campaign.proposals[0].candidate_id
            unsigned = {
                key: value
                for key, value in record.items()
                if key != "record_sha256"
            }
            record["record_sha256"] = content_hash(unsigned, 64)
            admission_path.write_text(json.dumps(record, sort_keys=True) + "\n")
            os.chmod(admission_path, 0o600)

            with self.assertRaisesRegex(
                CampaignPlanningError,
                "target does not match its controller prefix",
            ):
                summarize_campaign(campaign_dir)

    def test_schema_three_summary_requires_calibration_and_pair_admissions(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            root = Path(directory)
            missing_calibration_root = root / "missing-calibration"
            missing_calibration_root.mkdir()
            missing_calibration = _freeze_campaign_fixture(
                missing_calibration_root
            )
            stopped_at = datetime.fromisoformat("2026-08-28T00:00:00-07:00")

            def projection_for(cell: object) -> CellProjection:
                return _synthetic_projection(cell, improvement=1.0)

            with _synthetic_projection_boundary(
                projection_for,
                stopped_at=stopped_at,
                audit_reserve_s=25_200.0,
            ):
                run_campaign(
                    missing_calibration,
                    workspace=ROOT,
                    now=lambda: stopped_at,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=projection_for,  # type: ignore[arg-type]
                )
                (missing_calibration / "admissions.jsonl").unlink()
                with self.assertRaisesRegex(
                    CampaignPlanningError,
                    "calibration execution has no admitted provenance",
                ):
                    summarize_campaign(missing_calibration)

            missing_pair_root = root / "missing-pair"
            missing_pair_root.mkdir()
            missing_pair = _freeze_campaign_fixture(missing_pair_root)
            with _synthetic_projection_boundary(
                projection_for,
                stopped_at=stopped_at,
                audit_reserve_s=25_200.0,
            ):
                run_campaign(
                    missing_pair,
                    workspace=ROOT,
                    now=lambda: stopped_at,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=projection_for,  # type: ignore[arg-type]
                )
                run_campaign(
                    missing_pair,
                    workspace=ROOT,
                    now=lambda: stopped_at,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=projection_for,  # type: ignore[arg-type]
                )
                admission_path = missing_pair / "admissions.jsonl"
                calibration_line = admission_path.read_text().splitlines()[0]
                admission_path.write_text(calibration_line + "\n")
                with self.assertRaisesRegex(
                    CampaignPlanningError,
                    "search pair execution has no admitted provenance",
                ):
                    summarize_campaign(missing_pair)

    def test_admission_append_failure_precedes_controller_and_cell_mutation(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            calls: list[str] = []
            with (
                patch(
                    "bench.autoresearch_campaign.append_admission_record",
                    side_effect=AdmissionJournalError("synthetic append failure"),
                ),
                self.assertRaisesRegex(
                    CampaignPlanningError,
                    "admission journal is unsafe or malformed",
                ),
            ):
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: datetime.fromisoformat(
                        "2026-08-28T00:00:00-07:00"
                    ),
                    meminfo_reader=_admission_meminfo,
                    cell_runner=lambda cell: calls.append(str(cell.cell_id)),  # type: ignore[union-attr,return-value]
                )

            self.assertEqual(calls, [])
            self.assertFalse((campaign_dir / "events.jsonl").exists())
            self.assertFalse((campaign_dir / "summary.json").exists())
            self.assertFalse((campaign_dir / "admissions.jsonl").exists())

    def test_malformed_admission_precedes_recovery_terminal_mutation(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=lambda: datetime.fromisoformat("2026-08-28T00:00:00-07:00"),
                meminfo_reader=lambda: _admission_meminfo(swap_used_mib=65),
                cell_runner=lambda _cell: self.fail("denial launched a cell"),
            )
            admission_path = campaign_dir / "admissions.jsonl"
            admission_path.write_bytes(admission_path.read_bytes() + b"torn")
            with (
                patch(
                    "bench.autoresearch_campaign._recover_interrupted_cells",
                    side_effect=CellProjectionError(
                        "synthetic recovery failure",
                        failure_kind="cleanup_breach",
                    ),
                ),
                self.assertRaisesRegex(
                    CampaignPlanningError,
                    "admission journal is unsafe or malformed",
                ),
            ):
                run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: datetime.fromisoformat(
                        "2026-08-28T00:01:00-07:00"
                    ),
                    meminfo_reader=_admission_meminfo,
                    cell_runner=lambda _cell: self.fail("recovery launched a cell"),
                )

            self.assertFalse((campaign_dir / "events.jsonl").exists())

    def test_admission_samples_clock_after_memory_and_harness(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            campaign = load_frozen_campaign(campaign_dir)
            order: list[str] = []
            clock = [datetime.fromisoformat("2026-08-28T05:37:50-07:00")]

            def meminfo() -> str:
                order.append("meminfo")
                return _admission_meminfo()

            def harness(_workspace: Path) -> tuple[str, int]:
                order.append("harness")
                clock[0] += timedelta(milliseconds=1)
                return campaign.harness_tree_sha256, campaign.harness_file_count

            def now() -> datetime:
                order.append("clock")
                return clock[0]

            summary = run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=now,
                meminfo_reader=meminfo,
                harness_identity_reader=harness,
                cell_runner=lambda _cell: self.fail("cutoff launched a cell"),
            )

        self.assertEqual(order, ["meminfo", "harness", "clock"])
        self.assertEqual(summary["status"], "expired")

    def test_campaign_lock_rejects_links_and_an_active_controller(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_campaign_fixture(Path(directory))
            lock_path = campaign_dir / ".autoresearch.lock"
            protected = Path(directory) / "protected"
            protected.write_text("do-not-touch")
            lock_path.unlink()
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

            def projection_for(cell: object) -> CellProjection:
                projection = _synthetic_projection(cell, improvement=1.05)
                if "agent64k-none" in str(getattr(cell, "profile_id")):
                    return replace(projection, normalized_flags=())
                return projection

            def runner(cell: object) -> CellProjection:
                calls.append(str(getattr(cell, "cell_id")))
                return projection_for(cell)

            stopped_at = datetime.fromisoformat("2026-08-28T00:00:00-07:00")
            with _synthetic_projection_boundary(
                projection_for,
                stopped_at=stopped_at,
                audit_reserve_s=25_200.0,
            ):
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: stopped_at,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=runner,  # type: ignore[arg-type]
                )
                screened = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: stopped_at,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=runner,  # type: ignore[arg-type]
                )
                promoted = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: stopped_at,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=runner,  # type: ignore[arg-type]
                )
                replayed = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: stopped_at,
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

            def projection_for(cell: object) -> CellProjection:
                return _synthetic_projection(cell, improvement=1.0)

            def runner(cell: object) -> CellProjection:
                calls.append(str(getattr(cell, "cell_id")))
                return projection_for(cell)

            stopped_at = datetime.fromisoformat("2026-08-28T00:00:00-07:00")
            with _synthetic_projection_boundary(
                projection_for,
                stopped_at=stopped_at,
                audit_reserve_s=25_200.0,
            ):
                summaries = [
                    run_campaign(
                        campaign_dir,
                        workspace=ROOT,
                        now=lambda: stopped_at,
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

            def projection_for(cell: object) -> CellProjection:
                projection = _synthetic_projection(cell, improvement=1.05)
                if (
                    getattr(cell, "stage") == "screen"
                    and getattr(cell, "arm") == "candidate"
                ):
                    return replace(projection, measurement_elapsed_s=1800.001)
                return projection

            stopped_at = datetime.fromisoformat("2026-08-28T00:00:00-07:00")
            with _synthetic_projection_boundary(
                projection_for,
                stopped_at=stopped_at,
                audit_reserve_s=800.0,
            ):
                calibrated = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: stopped_at,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=projection_for,  # type: ignore[arg-type]
                )
                summary = run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: stopped_at,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=projection_for,  # type: ignore[arg-type]
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
        run_nonce = "1234567890abcdef1234567890abcdef"
        plan = {
            "schema_version": 2,
            "created_at": "2026-08-28T00:00:00+00:00",
            "fingerprint": fingerprint,
            "run_nonce": run_nonce,
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
        timestamp = "2026-08-28T00:00:00+00:00"
        measurement_started_ns = 1_000_000_000
        measurement_complete_ns = 101_000_000_000
        server_stopped_ns = 111_000_000_000
        events = [
            {
                "event": "run_start",
                "timestamp": timestamp,
                "completed_cases_at_resume": [],
                "plan_fingerprint": fingerprint,
                "run_nonce": run_nonce,
            },
            {
                "event": "measurement_started",
                "timestamp": timestamp,
                "monotonic_ns": measurement_started_ns,
                "plan_fingerprint": fingerprint,
                "run_nonce": run_nonce,
            },
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
            {
                "event": "measurement_complete",
                "timestamp": timestamp,
                "elapsed_s": 100.0,
                "monotonic_ns": measurement_complete_ns,
            },
            {
                "event": "server_stopped",
                "timestamp": timestamp,
                "backend": "sglang",
                "cleanup_elapsed_s": 10.0,
                "monotonic_ns": server_stopped_ns,
            },
            {
                "event": "run_complete",
                "timestamp": timestamp,
                "status": "completed",
            },
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
        self.assertEqual(
            len(projection.normalized_flags), len(set(projection.normalized_flags))
        )

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

    def test_projection_enforces_inclusive_causal_and_cleanup_deadlines(self) -> None:
        fixtures = (
            (1_800_000_000_000, 10_000_000_000, True, None),
            (1_800_000_000_001, 10_000_000_000, False, "measurement"),
            (100_000_000_000, 120_000_000_000, True, None),
            (100_000_000_000, 120_000_000_001, False, "cleanup_breach"),
        )
        for elapsed_ns, cleanup_ns, passes, failure_kind in fixtures:
            with self.subTest(
                elapsed_ns=elapsed_ns, cleanup_ns=cleanup_ns
            ), tempfile.TemporaryDirectory() as directory:
                run_dir = self._write_cell(Path(directory))
                events_path = run_dir / "events.jsonl"
                events = [json.loads(line) for line in events_path.read_text().splitlines()]
                started = next(
                    event for event in events if event["event"] == "measurement_started"
                )
                completed = next(
                    event for event in events if event["event"] == "measurement_complete"
                )
                stopped = next(
                    event for event in events if event["event"] == "server_stopped"
                )
                completed["monotonic_ns"] = started["monotonic_ns"] + elapsed_ns
                completed["elapsed_s"] = elapsed_ns / 1_000_000_000
                stopped["monotonic_ns"] = completed["monotonic_ns"] + cleanup_ns
                stopped["cleanup_elapsed_s"] = cleanup_ns / 1_000_000_000
                events_path.write_text(
                    "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
                )

                if passes:
                    project_completed_cell(run_dir)
                else:
                    with self.assertRaises(CellProjectionError) as raised:
                        project_completed_cell(run_dir)
                    self.assertEqual(raised.exception.failure_kind, failure_kind)

    def test_projection_enforces_durable_finalization_timestamp(self) -> None:
        fixtures = (
            ("2026-08-28T00:00:10+00:00", True),
            ("2026-08-28T00:00:10.000001+00:00", False),
            ("2026-08-27T23:59:59.999999+00:00", False),
        )
        for completed_at, passes in fixtures:
            with self.subTest(
                completed_at=completed_at
            ), tempfile.TemporaryDirectory() as directory:
                run_dir = self._write_cell(Path(directory))
                events_path = run_dir / "events.jsonl"
                events = [
                    json.loads(line) for line in events_path.read_text().splitlines()
                ]
                run_complete = next(
                    event for event in events if event["event"] == "run_complete"
                )
                run_complete["timestamp"] = completed_at
                events_path.write_text(
                    "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
                )

                if passes:
                    project_completed_cell(run_dir)
                else:
                    with self.assertRaises(CellProjectionError) as raised:
                        project_completed_cell(run_dir)
                    self.assertEqual(raised.exception.failure_kind, "cleanup_breach")

    def test_projection_rejects_duplicate_and_wrong_bound_markers(self) -> None:
        for mutation in ("duplicate", "wrong_nonce"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                run_dir = self._write_cell(Path(directory))
                events_path = run_dir / "events.jsonl"
                events = [json.loads(line) for line in events_path.read_text().splitlines()]
                if mutation == "duplicate":
                    index = next(
                        index
                        for index, event in enumerate(events)
                        if event["event"] == "measurement_complete"
                    )
                    events.insert(index + 1, dict(events[index]))
                else:
                    marker = next(
                        event
                        for event in events
                        if event["event"] == "measurement_started"
                    )
                    marker["run_nonce"] = "f" * 32
                events_path.write_text(
                    "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
                )

                with self.assertRaises(CellProjectionError) as raised:
                    project_completed_cell(run_dir)
                self.assertEqual(raised.exception.failure_kind, "audit")

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
