from __future__ import annotations

from dataclasses import replace
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch

import bench.autoresearch_campaign as campaign_module
from bench.autoresearch_campaign import CampaignPlanningError
from bench.journal import write_json
from tests.test_autoresearch_campaign import (
    ROOT,
    _downgrade_pre_admission_campaign_fixture,
    _freeze_campaign_fixture,
)


def _freeze_synthetic_campaign(root: Path) -> Path:
    with patch(
        "bench.execution_admission.RETIRED_SGLANG_SOURCE_OVERLAY_DIGESTS",
        frozenset(),
    ):
        return _freeze_campaign_fixture(root)


def _tree_snapshot(root: Path) -> tuple[tuple[object, ...], ...]:
    paths = (root, *sorted(root.rglob("*")))
    rows: list[tuple[object, ...]] = []
    for path in paths:
        metadata = os.lstat(path)
        relative = "." if path == root else path.relative_to(root).as_posix()
        if stat.S_ISREG(metadata.st_mode):
            kind = "file"
            content = hashlib.sha256(path.read_bytes()).hexdigest()
        elif stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
            content = ""
        elif stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
            content = os.readlink(path)
        else:
            kind = "other"
            content = ""
        rows.append(
            (
                relative,
                kind,
                stat.S_IMODE(metadata.st_mode),
                metadata.st_ino,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                content,
            )
        )
    return tuple(rows)


def _blocked_summary(campaign: object) -> dict[str, object]:
    return {
        "blockers": [
            "insufficient_time_for_pair",
            "starting_swap_above_clean_limit",
            "insufficient_preflight_memavailable",
        ],
        "calibration_recorded": False,
        "campaign_id": getattr(campaign, "campaign_id"),
        "candidate_decisions": {},
        "next_pair_index": 0,
        "policy_digest": getattr(campaign, "policy_digest"),
        "schema_version": 1,
        "status": "blocked_environment",
        "terminal_reason": None,
    }


def _create_sealed_fixture(
    root: Path,
) -> tuple[Path, campaign_module._LegacyBlockedCampaignSeal, dict[str, object]]:
    campaign_dir = _freeze_synthetic_campaign(root)
    _downgrade_pre_admission_campaign_fixture(campaign_dir)
    campaign = campaign_module.load_frozen_campaign(campaign_dir)
    for cell in campaign.cells:
        write_json(cell.run_dir / "inventory.json", {"synthetic": True})
    summary = _blocked_summary(campaign)
    write_json(campaign_dir / "summary.json", summary)
    lock_metadata = os.lstat(campaign_dir / ".autoresearch.lock")
    if (
        not stat.S_ISREG(lock_metadata.st_mode)
        or lock_metadata.st_nlink != 1
        or stat.S_IMODE(lock_metadata.st_mode) != 0o600
        or lock_metadata.st_size != 0
    ):
        raise AssertionError("frozen campaign lock does not have sealed topology")
    campaign_bytes = (campaign_dir / "campaign.json").read_bytes()
    summary_bytes = (campaign_dir / "summary.json").read_bytes()
    seal = campaign_module._LegacyBlockedCampaignSeal(
        campaign_id=campaign.campaign_id,
        created_at=campaign.created_at,
        cutoff=campaign.cutoff.isoformat(),
        integrity_sha256=campaign.integrity_hash,
        preview_sha256=campaign.preview_digest,
        policy_sha256=campaign.policy_digest,
        harness_tree_sha256=campaign.harness_tree_sha256,
        harness_file_count=campaign.harness_file_count,
        campaign_json_sha256=hashlib.sha256(campaign_bytes).hexdigest(),
        campaign_json_size=len(campaign_bytes),
        summary_json_sha256=hashlib.sha256(summary_bytes).hexdigest(),
        summary_json_size=len(summary_bytes),
        tree_sha256="",
        tree_size=0,
    )
    with patch.object(campaign_module, "_LEGACY_BLOCKED_CAMPAIGN_SEAL", seal):
        tree_sha256, tree_size = campaign_module._legacy_tree_identity(campaign)
    seal = replace(
        seal,
        tree_sha256=tree_sha256,
        tree_size=tree_size,
    )
    return campaign_dir, seal, summary


class ImmutableLegacySummaryTests(unittest.TestCase):
    def test_production_seal_matches_the_audited_constants(self) -> None:
        self.assertEqual(
            campaign_module._LEGACY_BLOCKED_CAMPAIGN_SEAL,
            campaign_module._LegacyBlockedCampaignSeal(
                campaign_id=(
                    "qwen38-flash-next-single-user-autoresearch-2026-08-28"
                ),
                created_at="2026-08-28T08:09:19.396+00:00",
                cutoff="2026-08-28T07:00:00-07:00",
                integrity_sha256=(
                    "ea576eaf6540bd842e956bbaec719227b60389df52505390ad0d26025bdf7d92"
                ),
                preview_sha256=(
                    "be2b70f0c0415d258e7d43979566e59606c55bc3adc87534d93945a067d8d1cb"
                ),
                policy_sha256=(
                    "ff3237c4106ebafbc50710e9a2222007611fc972ccc001b28a7100a6c01a50e7"
                ),
                harness_tree_sha256=(
                    "33170881721d0dce0f4466495110b336a7451fcd1635c5667f7fc5f722f7599f"
                ),
                harness_file_count=84,
                campaign_json_sha256=(
                    "523112428589a338faab93bf2eaa94a474bc0722c9ff396ca0fff4f34269f421"
                ),
                campaign_json_size=14_874,
                summary_json_sha256=(
                    "8c197a3f0dbda0bb06c3fca1e04940f54cb8f918754aa164acd8603f4afa847d"
                ),
                summary_json_size=471,
                tree_sha256=(
                    "5b121756b6a95644cb99969f309adaf43ea9d9b5d153749d869cd8bac420e988"
                ),
                tree_size=156_253,
            ),
        )

    def test_summary_is_copy_safe_idempotent_and_byte_preserving(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            root = Path(directory)
            source, seal, expected = _create_sealed_fixture(root)
            copied = root / "renamed-copy"
            shutil.copytree(source, copied)
            self.assertNotEqual(
                (source / "summary.json").stat().st_ino,
                (copied / "summary.json").stat().st_ino,
            )
            before = _tree_snapshot(copied)
            with (
                patch.object(
                    campaign_module, "_LEGACY_BLOCKED_CAMPAIGN_SEAL", seal
                ),
                patch.object(
                    campaign_module,
                    "write_json",
                    side_effect=AssertionError("legacy summary must not be written"),
                ),
            ):
                self.assertEqual(
                    campaign_module.summarize_campaign(copied), expected
                )
                self.assertEqual(
                    campaign_module.summarize_campaign(copied), expected
                )
            self.assertEqual(_tree_snapshot(copied), before)

    def test_evidence_snapshot_reads_exact_sealed_summary_without_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir, seal, expected = _create_sealed_fixture(Path(directory))
            before = _tree_snapshot(campaign_dir)
            with (
                patch.object(
                    campaign_module, "_LEGACY_BLOCKED_CAMPAIGN_SEAL", seal
                ),
                patch.object(
                    campaign_module,
                    "write_json",
                    side_effect=AssertionError("legacy evidence must not be written"),
                ),
            ):
                first = campaign_module.campaign_evidence_snapshot(campaign_dir)
                second = campaign_module.campaign_evidence_snapshot(campaign_dir)
            after = _tree_snapshot(campaign_dir)

        self.assertEqual(first, second)
        self.assertEqual(before, after)
        self.assertEqual(expected, first["summary"])
        self.assertEqual(1, first["snapshot_schema_version"])
        self.assertEqual(2, first["frozen_campaign_schema_version"])
        self.assertFalse(first["admission_journal_required"])
        self.assertEqual("sealed_legacy_unjournaled", first["provenance_mode"])
        self.assertEqual(seal.campaign_id, first["campaign_id"])
        self.assertEqual(seal.integrity_sha256, first["campaign_integrity_sha256"])
        self.assertEqual(seal.preview_sha256, first["preview_sha256"])
        self.assertEqual(seal.policy_sha256, first["policy_sha256"])
        self.assertEqual([], first["admissions"])
        self.assertEqual({}, first["controller_event_counts"])

    def test_run_refuses_sealed_campaign_before_execution_or_mutation(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir, seal, _expected = _create_sealed_fixture(Path(directory))
            before = _tree_snapshot(campaign_dir)
            with (
                patch.object(
                    campaign_module, "_LEGACY_BLOCKED_CAMPAIGN_SEAL", seal
                ),
                patch.object(
                    campaign_module,
                    "model_execution_blocker",
                    side_effect=AssertionError("model admission was consulted"),
                ),
            ):
                with self.assertRaisesRegex(CampaignPlanningError, "immutable"):
                    campaign_module.run_campaign(
                        campaign_dir,
                        workspace=ROOT,
                        meminfo_reader=lambda: self.fail("memory was consulted"),
                        harness_identity_reader=lambda _root: self.fail(
                            "harness was consulted"
                        ),
                        cell_runner=lambda _cell: self.fail("cell was launched"),
                    )
            self.assertEqual(_tree_snapshot(campaign_dir), before)

    def test_checkpoint_refuses_sealed_campaign_without_creating_lock(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir, seal, _expected = _create_sealed_fixture(Path(directory))
            (campaign_dir / ".autoresearch.lock").unlink()
            before = _tree_snapshot(campaign_dir)
            with (
                patch.object(
                    campaign_module, "_LEGACY_BLOCKED_CAMPAIGN_SEAL", seal
                ),
                patch.object(
                    campaign_module,
                    "_recover_interrupted_cells",
                    side_effect=AssertionError("recovery was entered"),
                ),
            ):
                with self.assertRaisesRegex(CampaignPlanningError, "immutable"):
                    campaign_module.acknowledge_campaign_checkpoint(
                        campaign_dir, ROOT
                    )
            self.assertEqual(_tree_snapshot(campaign_dir), before)

    def test_summary_rejects_state_artifacts_without_rewriting(self) -> None:
        mutations = (
            ("controller journal", lambda campaign: campaign / "events.jsonl"),
            ("admission journal", lambda campaign: campaign / "admissions.jsonl"),
            ("calibration", lambda campaign: campaign / "calibration.json"),
            (
                "cell journal",
                lambda campaign: next(
                    path.parent
                    for path in campaign.rglob("plan.json")
                )
                / "events.jsonl",
            ),
        )
        for label, target in mutations:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory(
                    dir=ROOT / "results"
                ) as directory:
                    campaign_dir, seal, _expected = _create_sealed_fixture(
                        Path(directory)
                    )
                    target(campaign_dir).write_text("{}\n", encoding="utf-8")
                    before = _tree_snapshot(campaign_dir)
                    with (
                        patch.object(
                            campaign_module,
                            "_LEGACY_BLOCKED_CAMPAIGN_SEAL",
                            seal,
                        ),
                        patch.object(
                            campaign_module,
                            "write_json",
                            side_effect=AssertionError("summary was rewritten"),
                        ),
                    ):
                        with self.assertRaises(CampaignPlanningError):
                            campaign_module.summarize_campaign(campaign_dir)
                    self.assertEqual(_tree_snapshot(campaign_dir), before)

    def test_summary_rejects_missing_or_changed_bytes_without_repair(self) -> None:
        for label in ("missing", "changed"):
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory(
                    dir=ROOT / "results"
                ) as directory:
                    campaign_dir, seal, _expected = _create_sealed_fixture(
                        Path(directory)
                    )
                    summary_path = campaign_dir / "summary.json"
                    if label == "missing":
                        summary_path.unlink()
                    else:
                        payload = bytearray(summary_path.read_bytes())
                        payload[-1] = ord(" ")
                        summary_path.write_bytes(payload)
                    before = _tree_snapshot(campaign_dir)
                    with (
                        patch.object(
                            campaign_module,
                            "_LEGACY_BLOCKED_CAMPAIGN_SEAL",
                            seal,
                        ),
                        patch.object(
                            campaign_module,
                            "write_json",
                            side_effect=AssertionError("summary was repaired"),
                        ),
                    ):
                        with self.assertRaises(CampaignPlanningError):
                            campaign_module.summarize_campaign(campaign_dir)
                    self.assertEqual(_tree_snapshot(campaign_dir), before)

    def test_summary_rejects_mutated_planning_artifact(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir, seal, _expected = _create_sealed_fixture(Path(directory))
            inventory = next(campaign_dir.rglob("inventory.json"))
            inventory.write_text('{"mutated":true}\n', encoding="utf-8")
            before = _tree_snapshot(campaign_dir)
            with (
                patch.object(
                    campaign_module, "_LEGACY_BLOCKED_CAMPAIGN_SEAL", seal
                ),
                patch.object(
                    campaign_module,
                    "write_json",
                    side_effect=AssertionError("summary was rewritten"),
                ),
            ):
                with self.assertRaisesRegex(CampaignPlanningError, "tree content"):
                    campaign_module.summarize_campaign(campaign_dir)
            self.assertEqual(_tree_snapshot(campaign_dir), before)

    def test_summary_detects_replacement_after_its_sealed_read(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir, seal, _expected = _create_sealed_fixture(Path(directory))
            summary_path = campaign_dir / "summary.json"
            original_reader = campaign_module._read_sealed_regular_file

            def racing_reader(path: Path, **kwargs: object) -> bytes:
                payload = original_reader(path, **kwargs)  # type: ignore[arg-type]
                if path == summary_path:
                    changed = bytearray(payload)
                    changed[-1] = ord(" ")
                    path.write_bytes(changed)
                return payload

            with (
                patch.object(
                    campaign_module, "_LEGACY_BLOCKED_CAMPAIGN_SEAL", seal
                ),
                patch.object(
                    campaign_module,
                    "_read_sealed_regular_file",
                    side_effect=racing_reader,
                ),
                patch.object(
                    campaign_module,
                    "write_json",
                    side_effect=AssertionError("summary was rewritten"),
                ),
            ):
                with self.assertRaisesRegex(CampaignPlanningError, "while reading"):
                    campaign_module.summarize_campaign(campaign_dir)
            self.assertEqual(summary_path.read_bytes()[-1:], b" ")
            self.assertFalse((campaign_dir / "summary.json.tmp").exists())

    def test_summary_rejects_linked_manifest_and_summary(self) -> None:
        for artifact in ("campaign.json", "summary.json"):
            for link_kind in ("symlink", "hardlink"):
                with self.subTest(artifact=artifact, link_kind=link_kind):
                    with tempfile.TemporaryDirectory(
                        dir=ROOT / "results"
                    ) as directory:
                        root = Path(directory)
                        campaign_dir, seal, _expected = _create_sealed_fixture(root)
                        path = campaign_dir / artifact
                        protected = root / f"protected-{artifact}"
                        protected.write_bytes(path.read_bytes())
                        expected_protected = protected.read_bytes()
                        path.unlink()
                        if link_kind == "symlink":
                            path.symlink_to(protected)
                        else:
                            os.link(protected, path)
                        before = _tree_snapshot(campaign_dir)
                        with patch.object(
                            campaign_module,
                            "_LEGACY_BLOCKED_CAMPAIGN_SEAL",
                            seal,
                        ):
                            with self.assertRaises(CampaignPlanningError):
                                campaign_module.summarize_campaign(campaign_dir)
                        self.assertEqual(_tree_snapshot(campaign_dir), before)
                        self.assertEqual(protected.read_bytes(), expected_protected)

    def test_summary_requires_existing_safe_unheld_lock(self) -> None:
        for label in ("missing", "wrong-mode", "symlink", "hardlink", "held"):
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory(
                    dir=ROOT / "results"
                ) as directory:
                    root = Path(directory)
                    campaign_dir, seal, _expected = _create_sealed_fixture(root)
                    lock_path = campaign_dir / ".autoresearch.lock"
                    held_descriptor: int | None = None
                    if label == "missing":
                        lock_path.unlink()
                    elif label == "wrong-mode":
                        lock_path.chmod(0o644)
                    elif label in {"symlink", "hardlink"}:
                        lock_path.unlink()
                        protected = root / f"protected-lock-{label}"
                        descriptor = os.open(
                            protected, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
                        )
                        os.close(descriptor)
                        if label == "symlink":
                            lock_path.symlink_to(protected)
                        else:
                            os.link(protected, lock_path)
                    else:
                        held_descriptor = os.open(lock_path, os.O_RDWR)
                        fcntl.flock(
                            held_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                    before = _tree_snapshot(campaign_dir)
                    try:
                        with patch.object(
                            campaign_module,
                            "_LEGACY_BLOCKED_CAMPAIGN_SEAL",
                            seal,
                        ):
                            with self.assertRaises(CampaignPlanningError):
                                campaign_module.summarize_campaign(campaign_dir)
                    finally:
                        if held_descriptor is not None:
                            os.close(held_descriptor)
                    self.assertEqual(_tree_snapshot(campaign_dir), before)

    def test_different_identity_uses_locked_normal_summary_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_synthetic_campaign(Path(directory))
            campaign = campaign_module.load_frozen_campaign(campaign_dir)
            self.assertFalse(campaign_module._is_legacy_blocked_campaign(campaign))
            summary = campaign_module.summarize_campaign(campaign_dir)

            self.assertEqual(summary["status"], "planned")
            self.assertEqual(
                campaign_module._read_json_object(
                    campaign_dir / "summary.json", context="test summary"
                ),
                summary,
            )
            lock_path = campaign_dir / ".autoresearch.lock"
            self.assertTrue(lock_path.is_file())
            summary_bytes = (campaign_dir / "summary.json").read_bytes()
            descriptor = os.open(lock_path, os.O_RDWR)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(
                    CampaignPlanningError, "another autoresearch controller"
                ):
                    campaign_module.summarize_campaign(campaign_dir)
            finally:
                os.close(descriptor)
            self.assertEqual(
                (campaign_dir / "summary.json").read_bytes(), summary_bytes
            )

    def test_normal_summary_rejects_lock_path_rebinding(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = _freeze_synthetic_campaign(Path(directory))
            lock_path = campaign_dir / ".autoresearch.lock"
            original_flock = fcntl.flock

            def rebind_lock(descriptor: int, operation: int) -> None:
                original_flock(descriptor, operation)
                lock_path.unlink()
                replacement = os.open(
                    lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
                )
                os.close(replacement)

            with patch.object(
                campaign_module.fcntl, "flock", side_effect=rebind_lock
            ):
                with self.assertRaisesRegex(CampaignPlanningError, "path changed"):
                    campaign_module.summarize_campaign(campaign_dir)
            self.assertFalse((campaign_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
