from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from bench.autoresearch_admission import (
    ADMISSION_BLOCKER_ORDER,
    AdmissionBinding,
    AdmissionJournalError,
    AdmissionTarget,
    append_admission_record,
    controller_prefix_sha256,
    observe_admission,
    read_admission_journal,
)
from bench.journal import canonical_json


def _binding() -> AdmissionBinding:
    return AdmissionBinding(
        campaign_id="synthetic-autoresearch",
        campaign_integrity_sha256="1" * 64,
        preview_sha256="2" * 64,
        policy_sha256="3" * 64,
        cutoff_at="2026-08-28T07:00:00-07:00",
        required_remaining_s=4930.0,
        required_memavailable_kib=22 * 1024 * 1024,
        max_starting_swap_kib=64 * 1024,
        harness_sha256="4" * 64,
        harness_file_count=84,
    )


def _target() -> AdmissionTarget:
    return AdmissionTarget(
        kind="screen",
        candidate_id="synthetic-candidate",
        pair_index=0,
    )


def _observation(
    binding: AdmissionBinding,
    *,
    remaining_s: float = 4930.0,
    memavailable_kib: int | None = None,
    swap_used_kib: int = 0,
    harness_sha256: str | None = None,
    harness_file_count: int | None = None,
):
    cutoff = datetime.fromisoformat(binding.cutoff_at)
    return observe_admission(
        binding,
        observed_at=cutoff - timedelta(seconds=remaining_s),
        memavailable_kib=(
            binding.required_memavailable_kib
            if memavailable_kib is None
            else memavailable_kib
        ),
        swap_used_kib=swap_used_kib,
        observed_harness_sha256=harness_sha256 or binding.harness_sha256,
        observed_harness_file_count=(
            binding.harness_file_count
            if harness_file_count is None
            else harness_file_count
        ),
    )


class AutoresearchAdmissionDecisionTests(unittest.TestCase):
    def test_exact_pair_boundary_is_inclusive(self) -> None:
        binding = _binding()

        exact = _observation(binding, remaining_s=4930.0)
        below = _observation(binding, remaining_s=4929.999)

        self.assertEqual(exact.blockers, ())
        self.assertEqual(exact.outcome, "admitted")
        self.assertAlmostEqual(exact.remaining_s, 4930.0)
        self.assertEqual(below.blockers, ("insufficient_time_for_pair",))
        self.assertEqual(below.outcome, "cutoff")
        self.assertAlmostEqual(below.remaining_s, 4929.999, places=6)

    def test_simultaneous_blockers_keep_frozen_order_and_safety_outcome(self) -> None:
        binding = _binding()

        observation = _observation(
            binding,
            remaining_s=-1.0,
            memavailable_kib=binding.required_memavailable_kib - 1,
            swap_used_kib=binding.max_starting_swap_kib + 1,
            harness_sha256="5" * 64,
        )

        self.assertEqual(observation.blockers, ADMISSION_BLOCKER_ORDER)
        self.assertEqual(observation.outcome, "blocked_environment")

    def test_targets_reject_ambiguous_control_and_search_bindings(self) -> None:
        with self.assertRaisesRegex(AdmissionJournalError, "control pair zero"):
            AdmissionTarget(kind="calibration", candidate_id="candidate", pair_index=0)
        with self.assertRaisesRegex(AdmissionJournalError, "cannot use the control"):
            AdmissionTarget(kind="screen", candidate_id="control", pair_index=0)
        with self.assertRaisesRegex(AdmissionJournalError, "kind is invalid"):
            AdmissionTarget(kind="other", candidate_id="candidate", pair_index=0)


class AutoresearchAdmissionJournalTests(unittest.TestCase):
    def test_records_are_mode_0600_chained_and_controller_prefix_bound(self) -> None:
        binding = _binding()
        events: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "admissions.jsonl"
            first = append_admission_record(
                path,
                binding=binding,
                target=AdmissionTarget(
                    kind="calibration", candidate_id="control", pair_index=0
                ),
                observation=_observation(binding),
                controller_events=events,
            )
            events.append(
                {
                    "event": "autoresearch_campaign_started",
                    "transition_id": "campaign-started",
                }
            )
            second = append_admission_record(
                path,
                binding=binding,
                target=_target(),
                observation=_observation(binding, remaining_s=4929.999),
                controller_events=events,
            )
            records = read_admission_journal(
                path, binding=binding, controller_events=events
            )
            mode = stat.S_IMODE(path.stat().st_mode)

        self.assertEqual(mode, 0o600)
        self.assertEqual(tuple(item["sequence"] for item in records), (1, 2))
        self.assertEqual(first["previous_record_sha256"], "0" * 64)
        self.assertEqual(
            second["previous_record_sha256"], first["record_sha256"]
        )
        self.assertEqual(first["controller_event_count"], 0)
        self.assertEqual(first["controller_prefix_sha256"], controller_prefix_sha256([]))
        self.assertEqual(second["controller_event_count"], 1)
        self.assertEqual(second["controller_prefix_sha256"], controller_prefix_sha256(events))
        self.assertEqual(second["outcome"], "cutoff")
        self.assertEqual(records, (first, second))

    def test_changed_controller_prefix_invalidates_history_and_prevents_append(self) -> None:
        binding = _binding()
        original_events = [{"event": "started", "transition_id": "one"}]
        changed_events = [{"event": "started", "transition_id": "two"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "admissions.jsonl"
            append_admission_record(
                path,
                binding=binding,
                target=_target(),
                observation=_observation(binding),
                controller_events=original_events,
            )
            before = path.read_bytes()
            with self.assertRaisesRegex(
                AdmissionJournalError, "controller prefix changed"
            ):
                append_admission_record(
                    path,
                    binding=binding,
                    target=replace(_target(), pair_index=1),
                    observation=_observation(binding),
                    controller_events=changed_events,
                )
            after = path.read_bytes()

        self.assertEqual(after, before)

    def test_digest_tamper_is_rejected_before_an_append(self) -> None:
        binding = _binding()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "admissions.jsonl"
            record = append_admission_record(
                path,
                binding=binding,
                target=_target(),
                observation=_observation(binding),
                controller_events=[],
            )
            record["memavailable_kib"] += 1
            path.write_text(canonical_json(record) + "\n", encoding="utf-8")
            os.chmod(path, 0o600)
            before = path.read_bytes()
            with self.assertRaisesRegex(AdmissionJournalError, "digest changed"):
                append_admission_record(
                    path,
                    binding=binding,
                    target=replace(_target(), pair_index=1),
                    observation=_observation(binding),
                    controller_events=[],
                )
            after = path.read_bytes()

        self.assertEqual(after, before)

    def test_unknown_fields_torn_tail_and_duplicate_keys_fail_closed(self) -> None:
        binding = _binding()
        fixtures = {
            "unknown": b'{"unknown":true}\n',
            "torn": b'{"schema_version":1}',
            "duplicate": b'{"schema_version":1,"schema_version":1}\n',
        }
        for label, payload in fixtures.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "admissions.jsonl"
                path.write_bytes(payload)
                os.chmod(path, 0o600)
                with self.assertRaises(AdmissionJournalError):
                    read_admission_journal(
                        path, binding=binding, controller_events=[]
                    )

    def test_links_and_permissive_modes_are_rejected(self) -> None:
        binding = _binding()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = root / "protected"
            protected.write_text("private", encoding="utf-8")
            os.chmod(protected, 0o600)
            linked = root / "admissions.jsonl"
            linked.symlink_to(protected)
            with self.assertRaises(AdmissionJournalError):
                read_admission_journal(
                    linked, binding=binding, controller_events=[]
                )
            self.assertEqual(protected.read_text(encoding="utf-8"), "private")

            linked.unlink()
            linked.write_bytes(b"")
            os.chmod(linked, 0o644)
            with self.assertRaisesRegex(AdmissionJournalError, "mode-0600"):
                append_admission_record(
                    linked,
                    binding=binding,
                    target=_target(),
                    observation=_observation(binding),
                    controller_events=[],
                )

    def test_reader_rejects_record_rebound_to_another_campaign(self) -> None:
        binding = _binding()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "admissions.jsonl"
            append_admission_record(
                path,
                binding=binding,
                target=_target(),
                observation=_observation(binding),
                controller_events=[],
            )
            other = replace(binding, campaign_integrity_sha256="9" * 64)
            with self.assertRaisesRegex(
                AdmissionJournalError, "campaign_integrity_sha256 binding changed"
            ):
                read_admission_journal(path, binding=other, controller_events=[])

    def test_record_contains_only_scalar_allowlist_and_hash_is_reproducible(self) -> None:
        binding = _binding()
        with tempfile.TemporaryDirectory() as directory:
            record = append_admission_record(
                Path(directory) / "admissions.jsonl",
                binding=binding,
                target=_target(),
                observation=_observation(binding),
                controller_events=[],
            )

        unsigned = {
            key: value for key, value in record.items() if key != "record_sha256"
        }
        self.assertEqual(
            record["record_sha256"],
            hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest(),
        )
        self.assertFalse(
            any(
                forbidden in canonical_json(record).lower()
                for forbidden in (
                    "path",
                    "command",
                    "prompt",
                    "completion",
                    "request_id",
                    "api_key",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
