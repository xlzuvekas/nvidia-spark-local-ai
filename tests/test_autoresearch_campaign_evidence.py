from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import bench.autoresearch_campaign as campaign_module
import bench.evidence as evidence_module
from bench.evidence import (
    EvidenceError,
    SCHEMA_VERSION,
    _autoresearch_admission_sha256,
    _write_bundle,
    export_evidence,
    verify_evidence,
)
from bench.journal import content_hash
from tests.test_autoresearch_campaign import (
    ROOT,
    _admission_records,
    _admission_meminfo,
    _downgrade_pre_admission_campaign_fixture,
    _freeze_campaign_fixture,
    _synthetic_projection,
    _synthetic_projection_boundary,
)
from tests.test_autoresearch_legacy_summary import _create_sealed_fixture
from tests.test_evidence import (
    EvidenceFixture,
    RAW_COMPLETION,
    RAW_HOST_PATH,
    RAW_REASONING,
    RAW_REQUEST_ID,
    RAW_SECRET,
)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_campaign_export(
    campaign: Path,
    _results_root: Path,
    output_root: Path,
) -> dict[str, object]:
    relative = Path("campaigns") / campaign.name
    bundle_sha256, _ = _write_bundle(
        output_root,
        relative,
        {
            "manifest.json": {
                "campaign_id": campaign.name,
                "evidence_kind": "synthetic_campaign",
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
            },
            "measurements.json": {
                "measurement_count": 0,
                "measurements": [],
                "schema_version": SCHEMA_VERSION,
            },
            "telemetry.json": {
                "capture_count": 0,
                "captures": [],
                "schema_version": SCHEMA_VERSION,
            },
        },
    )
    return {
        "bundle_sha256": bundle_sha256,
        "campaign_id": campaign.name,
        "evidence_kind": "synthetic_campaign",
        "file": f"campaigns/{campaign.name}/manifest.json",
        "status": "complete",
    }


def _export(fixture: EvidenceFixture, output: Path) -> dict[str, object]:
    with (
        patch(
            "bench.evidence._export_campaign",
            side_effect=_fake_campaign_export,
        ),
        patch(
            "bench.evidence._validate_runtime_overlay_tree",
            return_value=False,
        ),
    ):
        return export_evidence(
            results_root=fixture.results,
            output_root=output,
        )


def _campaign_bundle(output: Path) -> tuple[dict[str, object], Path]:
    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in index["campaigns"]
        if entry.get("evidence_kind") == "autoresearch_campaign"
    ]
    if len(entries) != 1:
        raise AssertionError("expected exactly one autoresearch campaign bundle")
    entry = entries[0]
    return entry, (output / Path(str(entry["file"]))).parent


def _remove_default_run(fixture: EvidenceFixture) -> None:
    shutil.rmtree(fixture.run_dir)


def _freeze_export_campaign(autoresearch_root: Path) -> Path:
    with patch(
        "bench.execution_admission.RETIRED_SGLANG_SOURCE_OVERLAY_DIGESTS",
        frozenset(),
    ):
        campaign_dir = _freeze_campaign_fixture(autoresearch_root)
    campaign_path = campaign_dir / "campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    for cell in campaign["cells"]:
        ordinal = int(cell["ordinal"])
        old_run = campaign_dir / str(cell["run_dir"])
        new_run = old_run.with_name(
            f"20260828T0000{ordinal:02d}Z-frozen-run"
        )
        old_run.rename(new_run)
        cell["run_dir"] = new_run.relative_to(campaign_dir).as_posix()
    campaign.pop("integrity_hash")
    campaign["integrity_hash"] = content_hash(campaign, 64)
    _write_json(campaign_path, campaign)
    return campaign_dir


def _create_export_sealed_fixture(
    autoresearch_root: Path,
) -> tuple[Path, object, dict[str, object]]:
    with patch(
        "tests.test_autoresearch_legacy_summary._freeze_campaign_fixture",
        side_effect=_freeze_export_campaign,
    ):
        return _create_sealed_fixture(autoresearch_root)


def _write_schema_three_cutoff(fixture: EvidenceFixture) -> Path:
    _remove_default_run(fixture)
    autoresearch_root = fixture.results / "autoresearch"
    autoresearch_root.mkdir()
    campaign_dir = _freeze_export_campaign(autoresearch_root)
    campaign = campaign_module.load_frozen_campaign(campaign_dir)

    def no_cell(_cell: object) -> object:
        raise AssertionError("cutoff fixture launched a cell")

    with (
        patch.object(
            campaign_module,
            "model_execution_blocker",
            return_value=None,
        ),
        patch.object(
            campaign_module,
            "_recover_cell",
            return_value="already_absent",
        ),
    ):
        summary = campaign_module.run_campaign(
            campaign_dir,
            workspace=ROOT,
            now=lambda: datetime.fromisoformat(
                "2026-08-28T05:37:50.001-07:00"
            ),
            meminfo_reader=_admission_meminfo,
            harness_identity_reader=lambda _workspace: (
                campaign.harness_tree_sha256,
                campaign.harness_file_count,
            ),
            cell_runner=no_cell,  # type: ignore[arg-type]
        )
    if summary["status"] != "expired":
        raise AssertionError("cutoff fixture did not reach expired status")
    return campaign_dir


_PROGRESS_STOPPED_AT = datetime.fromisoformat("2026-08-28T00:00:00-07:00")


def _progress_projection(cell: object) -> object:
    projection = _synthetic_projection(cell, improvement=1.05)
    if "agent64k-none" in str(getattr(cell, "profile_id")):
        return replace(projection, normalized_flags=())
    return projection


def _write_schema_three_progress(
    fixture: EvidenceFixture, *, invocations: int
) -> Path:
    _remove_default_run(fixture)
    autoresearch_root = fixture.results / "autoresearch"
    autoresearch_root.mkdir()
    campaign_dir = _freeze_export_campaign(autoresearch_root)

    with (
        patch.object(campaign_module, "model_execution_blocker", return_value=None),
        patch.object(
            campaign_module, "_recover_cell", return_value="already_absent"
        ),
        _synthetic_projection_boundary(
            _progress_projection,  # type: ignore[arg-type]
            stopped_at=_PROGRESS_STOPPED_AT,
            audit_reserve_s=25_200.0,
        ),
    ):
        summary: dict[str, object] | None = None
        for _index in range(invocations):
            summary = campaign_module.run_campaign(
                campaign_dir,
                workspace=ROOT,
                now=lambda: _PROGRESS_STOPPED_AT,
                meminfo_reader=_admission_meminfo,
                cell_runner=_progress_projection,  # type: ignore[arg-type]
            )
    if summary is None:
        raise AssertionError("progress fixture requires at least one invocation")
    return campaign_dir


def _export_progress(fixture: EvidenceFixture, output: Path) -> dict[str, object]:
    with _synthetic_projection_boundary(
        _progress_projection,  # type: ignore[arg-type]
        stopped_at=_PROGRESS_STOPPED_AT,
        audit_reserve_s=25_200.0,
    ):
        return _export(fixture, output)


def _write_schema_two_optional(fixture: EvidenceFixture) -> Path:
    _remove_default_run(fixture)
    autoresearch_root = fixture.results / "autoresearch"
    autoresearch_root.mkdir()
    campaign_dir = _freeze_export_campaign(autoresearch_root)
    _downgrade_pre_admission_campaign_fixture(campaign_dir)
    return campaign_dir


def _refresh_all_checksums(output: Path, bundle: Path) -> None:
    bundle_files = {
        path.name: _sha256(path)
        for path in sorted(bundle.iterdir())
        if path.is_file() and path.name != "checksums.json"
    }
    _write_json(
        bundle / "checksums.json",
        {"files": bundle_files, "schema_version": SCHEMA_VERSION},
    )

    index_path = output / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for entry in index["campaigns"]:
        if entry.get("evidence_kind") == "autoresearch_campaign":
            entry["bundle_sha256"] = _sha256(bundle / "checksums.json")
    _write_json(index_path, index)

    root_files = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != output / "checksums.json"
    }
    _write_json(
        output / "checksums.json",
        {"files": root_files, "schema_version": SCHEMA_VERSION},
    )


def _exclusive_lock_available(lock_path: Path) -> bool:
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return True
    finally:
        os.close(descriptor)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for child in value.values() for key in _all_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _all_keys(child)}
    return set()


class AutoresearchCampaignEvidenceTests(unittest.TestCase):
    def test_schema_three_cutoff_exports_deterministically_with_nested_runs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            campaign_dir = _write_schema_three_cutoff(fixture)
            first_output = root / "evidence-first"
            second_output = root / "evidence-second"

            first = _export(fixture, first_output)
            repeated = _export(fixture, first_output)
            independent = _export(fixture, second_output)

            self.assertTrue(first["changed"])
            self.assertFalse(repeated["changed"])
            self.assertTrue(independent["changed"])
            self.assertEqual(14, first["runs"])
            self.assertEqual(_tree_bytes(first_output), _tree_bytes(second_output))
            self.assertEqual("verified", verify_evidence(first_output)["status"])

            index = json.loads(
                (first_output / "index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(14, len(index["runs"]))
            self.assertEqual(14, len({entry["run_id"] for entry in index["runs"]}))
            entry, bundle = _campaign_bundle(first_output)
            self.assertEqual("expired", entry["status"])
            self.assertEqual(
                {"manifest.json", "controller.json", "admissions.json", "checksums.json"},
                {path.name for path in bundle.iterdir()},
            )
            manifest = json.loads(
                (bundle / "manifest.json").read_text(encoding="utf-8")
            )
            admissions = json.loads(
                (bundle / "admissions.json").read_text(encoding="utf-8")
            )
            self.assertEqual("autoresearch_campaign", manifest["evidence_kind"])
            self.assertEqual(3, manifest["frozen_campaign_schema_version"])
            self.assertEqual(14, manifest["planned_cell_count"])
            self.assertEqual("required", manifest["provenance_mode"])
            self.assertEqual(1, admissions["record_count"])
            self.assertEqual("cutoff", admissions["records"][0]["outcome"])
            self.assertNotIn(
                campaign_dir.name,
                b"\n".join(_tree_bytes(first_output).values()).decode("utf-8"),
            )

    def test_fresh_environment_denial_clean_resume_exports_admission_chain(
        self,
    ) -> None:
        blocked_at = datetime.fromisoformat("2026-08-28T00:00:00-07:00")
        admitted_at = datetime.fromisoformat("2026-08-28T00:01:00-07:00")
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            _remove_default_run(fixture)
            autoresearch_root = fixture.results / "autoresearch"
            autoresearch_root.mkdir()
            campaign_dir = _freeze_export_campaign(autoresearch_root)
            campaign = campaign_module.load_frozen_campaign(campaign_dir)
            expected_cells = campaign.cells_for(
                candidate_id="control", stage="calibration"
            )
            cell_calls: list[str] = []

            def projection(cell: object) -> object:
                cell_calls.append(str(getattr(cell, "cell_id")))
                return _progress_projection(cell)

            with (
                patch.object(
                    campaign_module, "model_execution_blocker", return_value=None
                ),
                patch.object(
                    campaign_module, "_recover_cell", return_value="already_absent"
                ),
                _synthetic_projection_boundary(
                    _progress_projection,  # type: ignore[arg-type]
                    stopped_at=admitted_at,
                    audit_reserve_s=25_200.0,
                ),
            ):
                blocked = campaign_module.run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: blocked_at,
                    meminfo_reader=lambda: _admission_meminfo(swap_used_mib=65),
                    cell_runner=lambda _cell: self.fail(
                        "blocked admission launched a cell"
                    ),
                )
                resumed = campaign_module.run_campaign(
                    campaign_dir,
                    workspace=ROOT,
                    now=lambda: admitted_at,
                    meminfo_reader=_admission_meminfo,
                    cell_runner=projection,  # type: ignore[arg-type]
                )
                output = root / "evidence"
                exported = _export(fixture, output)
                verification = verify_evidence(output)

            records = _admission_records(campaign_dir)
            controller_events = [
                json.loads(line)
                for line in (campaign_dir / "events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            entry, bundle = _campaign_bundle(output)
            controller = json.loads(
                (bundle / "controller.json").read_text(encoding="utf-8")
            )
            published = json.loads(
                (bundle / "admissions.json").read_text(encoding="utf-8")
            )

        self.assertEqual("blocked_environment", blocked["status"])
        self.assertEqual("planned", blocked["controller_status"])
        self.assertEqual(1, blocked["admission_count"])
        self.assertEqual("active", resumed["status"])
        self.assertEqual("active", resumed["controller_status"])
        self.assertTrue(resumed["calibration_recorded"])
        self.assertEqual(2, resumed["admission_count"])
        self.assertEqual(
            ("blocked_environment", "admitted"),
            tuple(record["outcome"] for record in records),
        )
        self.assertEqual((1, 2), tuple(record["sequence"] for record in records))
        self.assertEqual(
            ("calibration", "calibration"),
            tuple(record["target_kind"] for record in records),
        )
        self.assertEqual(
            (0, 0), tuple(record["controller_event_count"] for record in records)
        )
        self.assertEqual(
            records[0]["controller_prefix_sha256"],
            records[1]["controller_prefix_sha256"],
        )
        self.assertEqual(
            records[0]["record_sha256"], records[1]["previous_record_sha256"]
        )
        self.assertEqual(
            [expected_cells[arm].cell_id for arm in ("control_a", "control_b")],
            cell_calls,
        )
        self.assertEqual(
            ["autoresearch_campaign_started"],
            [event["event"] for event in controller_events],
        )
        self.assertTrue(exported["changed"])
        self.assertEqual("active", entry["status"])
        self.assertEqual("active", controller["controller_status"])
        self.assertTrue(controller["calibration_recorded"])
        self.assertEqual(2, published["record_count"])
        self.assertEqual(
            ["blocked_environment", "admitted"],
            [record["outcome"] for record in published["records"]],
        )
        self.assertEqual("admitted", published["effective_outcome"])
        self.assertEqual([], published["effective_blockers"])
        self.assertEqual("verified", verification["status"])

    def test_schema_two_campaign_exports_optional_empty_provenance(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            _write_schema_two_optional(fixture)
            output = root / "evidence"

            _export(fixture, output)
            entry, bundle = _campaign_bundle(output)

            manifest = json.loads(
                (bundle / "manifest.json").read_text(encoding="utf-8")
            )
            controller = json.loads(
                (bundle / "controller.json").read_text(encoding="utf-8")
            )
            admissions = json.loads(
                (bundle / "admissions.json").read_text(encoding="utf-8")
            )
            self.assertEqual("planned", entry["status"])
            self.assertEqual(2, manifest["frozen_campaign_schema_version"])
            self.assertEqual("optional", manifest["provenance_mode"])
            self.assertEqual("planned", controller["status"])
            self.assertEqual(0, admissions["record_count"])
            self.assertEqual([], admissions["records"])
            self.assertEqual("unobserved", admissions["effective_outcome"])
            self.assertEqual([], admissions["effective_blockers"])
            self.assertEqual("verified", verify_evidence(output)["status"])

    def test_exact_sealed_legacy_exports_unjournaled_provenance(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            _remove_default_run(fixture)
            autoresearch_root = fixture.results / "autoresearch"
            autoresearch_root.mkdir()
            _campaign_dir, seal, expected_summary = _create_export_sealed_fixture(
                autoresearch_root
            )
            output = root / "evidence"

            with patch.object(
                campaign_module,
                "_LEGACY_BLOCKED_CAMPAIGN_SEAL",
                seal,
            ):
                _export(fixture, output)
                entry, bundle = _campaign_bundle(output)

                manifest = json.loads(
                    (bundle / "manifest.json").read_text(encoding="utf-8")
                )
                controller = json.loads(
                    (bundle / "controller.json").read_text(encoding="utf-8")
                )
                admissions = json.loads(
                    (bundle / "admissions.json").read_text(encoding="utf-8")
                )
                self.assertEqual(expected_summary["status"], entry["status"])
                self.assertEqual(
                    "sealed_legacy_unjournaled", manifest["provenance_mode"]
                )
                self.assertEqual("planned", controller["controller_status"])
                self.assertEqual(0, admissions["record_count"])
                self.assertEqual([], admissions["records"])
                self.assertEqual(
                    "blocked_environment", admissions["effective_outcome"]
                )
                self.assertEqual(
                    expected_summary["blockers"],
                    admissions["effective_blockers"],
                )
                self.assertEqual("verified", verify_evidence(output)["status"])

    def test_source_admission_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            campaign_dir = _write_schema_three_cutoff(fixture)
            admission_path = campaign_dir / "admissions.jsonl"
            record = json.loads(admission_path.read_text(encoding="utf-8"))
            record["memavailable_kib"] += 1
            _write_json(admission_path, record)

            with self.assertRaises(EvidenceError):
                _export(fixture, root / "evidence")

    def test_published_controller_tamper_fails_after_checksum_refresh(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            _write_schema_three_cutoff(fixture)
            output = root / "evidence"
            _export(fixture, output)
            _entry, bundle = _campaign_bundle(output)
            controller_path = bundle / "controller.json"
            controller = json.loads(controller_path.read_text(encoding="utf-8"))
            controller["controller_event_count"] = 1
            _write_json(controller_path, controller)
            _refresh_all_checksums(output, bundle)

            with self.assertRaises(EvidenceError):
                verify_evidence(output)

    def test_confirmation_history_exports_and_rejects_premature_promotion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            _write_schema_three_progress(fixture, invocations=2)
            output = root / "evidence"
            _export_progress(fixture, output)
            _entry, bundle = _campaign_bundle(output)
            controller_path = bundle / "controller.json"
            controller = json.loads(controller_path.read_text(encoding="utf-8"))

            self.assertEqual("active", controller["status"])
            self.assertEqual("candidate", controller["controller_phase"])
            self.assertEqual(
                ["confirm_simplification"],
                [row["decision"] for row in controller["candidate_decisions"]],
            )
            self.assertEqual("verified", verify_evidence(output)["status"])

            controller["candidate_decisions"][0][
                "decision"
            ] = "promote_simplification"
            _write_json(controller_path, controller)
            _refresh_all_checksums(output, bundle)
            with self.assertRaises(EvidenceError):
                verify_evidence(output)

    def test_completed_confirmation_history_rejects_mode_change(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            _write_schema_three_progress(fixture, invocations=3)
            output = root / "evidence"
            _export_progress(fixture, output)
            _entry, bundle = _campaign_bundle(output)
            controller_path = bundle / "controller.json"
            controller = json.loads(controller_path.read_text(encoding="utf-8"))

            self.assertEqual("complete", controller["status"])
            self.assertEqual(
                ["confirm_simplification", "promote_simplification"],
                [row["decision"] for row in controller["candidate_decisions"]],
            )
            self.assertEqual(
                2,
                controller["controller_event_counts"][
                    "autoresearch_candidate_decided"
                ],
            )
            self.assertEqual("verified", verify_evidence(output)["status"])

            controller["candidate_decisions"][1]["decision"] = "promote"
            _write_json(controller_path, controller)
            _refresh_all_checksums(output, bundle)
            with self.assertRaises(EvidenceError):
                verify_evidence(output)

    def test_invented_controller_phase_and_event_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            _write_schema_three_progress(fixture, invocations=1)
            output = root / "evidence"
            _export_progress(fixture, output)
            _entry, bundle = _campaign_bundle(output)
            controller_path = bundle / "controller.json"
            original = json.loads(controller_path.read_text(encoding="utf-8"))

            mutations = (
                {**original, "controller_phase": "invented_phase"},
                {
                    **original,
                    "controller_event_counts": {"invented_event": 1},
                },
                {**original, "controller_phase": "scored"},
            )
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    _write_json(controller_path, mutation)
                    _refresh_all_checksums(output, bundle)
                    with self.assertRaises(EvidenceError):
                        verify_evidence(output)

    def test_active_controller_rejects_cutoff_admission_tail(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            _write_schema_three_progress(fixture, invocations=1)
            output = root / "evidence"
            _export_progress(fixture, output)
            _entry, bundle = _campaign_bundle(output)
            admissions_path = bundle / "admissions.json"
            admissions = json.loads(admissions_path.read_text(encoding="utf-8"))
            record = admissions["records"][-1]
            record["observed_at"] = "2026-08-28T05:37:50.001-07:00"
            record["remaining_s"] = 4929.999
            record["blockers"] = ["insufficient_time_for_pair"]
            record["outcome"] = "cutoff"
            record["record_sha256"] = _autoresearch_admission_sha256(record)
            admissions["effective_outcome"] = "cutoff"
            admissions["effective_blockers"] = ["insufficient_time_for_pair"]
            _write_json(admissions_path, admissions)
            _refresh_all_checksums(output, bundle)

            with self.assertRaises(EvidenceError):
                verify_evidence(output)

    def test_search_target_without_calibration_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            _write_schema_three_cutoff(fixture)
            output = root / "evidence"
            _export(fixture, output)
            _entry, bundle = _campaign_bundle(output)
            manifest = json.loads(
                (bundle / "manifest.json").read_text(encoding="utf-8")
            )
            admissions_path = bundle / "admissions.json"
            admissions = json.loads(admissions_path.read_text(encoding="utf-8"))
            record = admissions["records"][0]
            record["target_kind"] = "screen"
            record["candidate_id"] = manifest["candidates"][0]["candidate_id"]
            record["record_sha256"] = _autoresearch_admission_sha256(record)
            _write_json(admissions_path, admissions)
            _refresh_all_checksums(output, bundle)

            with self.assertRaises(EvidenceError):
                verify_evidence(output)

    def test_export_holds_campaign_lock_through_nested_runs_and_verify(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            campaign_dir = _write_schema_two_optional(fixture)
            lock_path = campaign_dir / ".autoresearch.lock"
            output = root / "evidence"
            original_export_run = evidence_module._export_run
            original_verify = evidence_module.verify_evidence
            nested_probes = 0
            verification_probes = 0

            def locked_export_run(*args: object, **kwargs: object) -> object:
                nonlocal nested_probes
                run_dir = Path(args[0])
                if campaign_dir in run_dir.parents:
                    nested_probes += 1
                    self.assertFalse(_exclusive_lock_available(lock_path))
                return original_export_run(*args, **kwargs)  # type: ignore[arg-type]

            def locked_verify(*args: object, **kwargs: object) -> object:
                nonlocal verification_probes
                verification_probes += 1
                self.assertFalse(_exclusive_lock_available(lock_path))
                return original_verify(*args, **kwargs)  # type: ignore[arg-type]

            with (
                patch.object(
                    evidence_module, "_export_run", side_effect=locked_export_run
                ),
                patch.object(
                    evidence_module, "verify_evidence", side_effect=locked_verify
                ),
            ):
                _export(fixture, output)

            self.assertEqual(14, nested_probes)
            self.assertGreaterEqual(verification_probes, 1)
            self.assertTrue(_exclusive_lock_available(lock_path))

    def test_export_releases_campaign_lock_after_failure(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            campaign_dir = _write_schema_two_optional(fixture)
            lock_path = campaign_dir / ".autoresearch.lock"
            with patch.object(
                evidence_module,
                "_export_run",
                side_effect=EvidenceError("injected nested export failure"),
            ):
                with self.assertRaises(EvidenceError):
                    _export(fixture, root / "evidence")
            self.assertTrue(_exclusive_lock_available(lock_path))

    def test_export_rejects_preheld_lock_and_campaign_set_race(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            campaign_dir = _write_schema_two_optional(fixture)
            lock_path = campaign_dir / ".autoresearch.lock"
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CLOEXEC)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(EvidenceError):
                    _export(fixture, root / "evidence-locked")
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            self.assertFalse((root / "evidence-locked").exists())

            original_sources = evidence_module._autoresearch_sources

            def add_campaign(*args: object, **kwargs: object) -> object:
                (fixture.results / "autoresearch" / "new-campaign").mkdir()
                return original_sources(*args, **kwargs)  # type: ignore[arg-type]

            with patch.object(
                evidence_module,
                "_autoresearch_sources",
                side_effect=add_campaign,
            ):
                with self.assertRaises(EvidenceError):
                    _export(fixture, root / "evidence-race")
            self.assertFalse((root / "evidence-race").exists())

    def test_campaign_bundle_contains_no_sensitive_or_raw_fields(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            campaign_dir = _write_schema_three_cutoff(fixture)
            source_nonces = [
                json.loads(path.read_text(encoding="utf-8"))["run_nonce"]
                for path in campaign_dir.rglob("plan.json")
            ]
            output = root / "evidence"
            _export(fixture, output)
            _entry, bundle = _campaign_bundle(output)
            bundle_bytes = {
                path.name: path.read_bytes()
                for path in bundle.iterdir()
                if path.is_file()
            }
            serialized = b"\n".join(bundle_bytes.values()).decode("utf-8")
            for private_value in (
                str(campaign_dir),
                str(fixture.results),
                campaign_dir.name,
                RAW_COMPLETION,
                RAW_REASONING,
                RAW_REQUEST_ID,
                RAW_HOST_PATH,
                RAW_SECRET,
                *source_nonces,
            ):
                with self.subTest(private_value=private_value):
                    self.assertNotIn(private_value, serialized)

            keys: set[str] = set()
            for name, payload in bundle_bytes.items():
                if name.endswith(".json"):
                    keys.update(_all_keys(json.loads(payload)))
            self.assertTrue(
                {
                    "command",
                    "completion",
                    "environment",
                    "local_path",
                    "prompt",
                    "reasoning",
                    "request_id",
                    "run_dir",
                    "run_nonce",
                    "tool_payload",
                }.isdisjoint(keys)
            )


if __name__ == "__main__":
    unittest.main()
