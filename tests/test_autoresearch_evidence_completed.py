from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench.evidence import SCHEMA_VERSION, _write_bundle, export_evidence, verify_evidence
from tests.test_evidence import (
    EvidenceFixture,
    RAW_COMPLETION,
    RAW_HOST_PATH,
    RAW_REASONING,
    RAW_REQUEST_ID,
    RAW_SECRET,
)


MARKER_RAW_SENTINEL = "AUTORESEARCH_MARKER_RAW_SENTINEL"
MARKER_MONOTONIC_BASE = 987_654_321_012_345_678


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _json_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key
            for child in value.values()
            for key in _json_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _json_keys(child)}
    return set()


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


class CompletedAutoresearchEvidenceTests(unittest.TestCase):
    def test_completed_nested_cells_export_deterministically_without_raw_markers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            _campaign_dir, run_dirs = fixture.write_autoresearch_campaign()

            marker_nonces: list[str] = []
            marker_monotonic_values: list[int] = []
            for ordinal, run_dir in enumerate(run_dirs, start=1):
                plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
                nonce = str(plan["run_nonce"])
                fingerprint = str(plan["fingerprint"])
                marker_nonces.append(nonce)

                events_path = run_dir / "events.jsonl"
                events = [
                    json.loads(line)
                    for line in events_path.read_text(encoding="utf-8").splitlines()
                ]
                start_ns = MARKER_MONOTONIC_BASE + ordinal * 1_000
                complete_ns = start_ns + 500
                stopped_ns = complete_ns + 100
                marker_monotonic_values.extend((start_ns, complete_ns, stopped_ns))

                events[0].update(
                    {
                        "plan_fingerprint": fingerprint,
                        "run_nonce": nonce,
                    }
                )
                events.insert(
                    1,
                    {
                        "event": "measurement_started",
                        "monotonic_ns": start_ns,
                        "plan_fingerprint": fingerprint,
                        "raw": MARKER_RAW_SENTINEL,
                        "run_nonce": nonce,
                        "timestamp": "2026-08-17T00:00:00.500Z",
                    },
                )
                events.insert(
                    -1,
                    {
                        "elapsed_s": 0.0000005,
                        "event": "measurement_complete",
                        "monotonic_ns": complete_ns,
                        "timestamp": "2026-08-17T00:00:06.500Z",
                    },
                )
                events.insert(
                    -1,
                    {
                        "backend": "synthetic",
                        "cleanup_elapsed_s": 0.0000001,
                        "event": "server_stopped",
                        "monotonic_ns": stopped_ns,
                        "timestamp": "2026-08-17T00:00:06.750Z",
                    },
                )
                fixture.write_jsonl(events_path, events)

            first_output = root / "evidence-first"
            second_output = root / "evidence-second"
            with patch(
                "bench.evidence._export_campaign",
                side_effect=_fake_campaign_export,
            ):
                first = export_evidence(
                    results_root=fixture.results,
                    output_root=first_output,
                )
                repeated = export_evidence(
                    results_root=fixture.results,
                    output_root=first_output,
                )
                independent = export_evidence(
                    results_root=fixture.results,
                    output_root=second_output,
                )

            first_bytes = _tree_bytes(first_output)
            self.assertTrue(first["changed"])
            self.assertFalse(repeated["changed"])
            self.assertTrue(independent["changed"])
            self.assertEqual(14, first["runs"])
            self.assertEqual(first_bytes, _tree_bytes(second_output))
            self.assertEqual("verified", verify_evidence(first_output)["status"])
            self.assertEqual("verified", verify_evidence(second_output)["status"])

            index = json.loads((first_output / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(14, len(index["runs"]))
            for entry in index["runs"]:
                self.assertTrue(entry["measurement_terminal"])
                manifest = json.loads(
                    (first_output / entry["file"]).read_text(encoding="utf-8")
                )
                self.assertEqual("run_complete", manifest["lifecycle"]["terminal_event"])
                self.assertEqual(
                    1,
                    manifest["lifecycle"]["event_counts"]["measurement_started"],
                )
                self.assertEqual(
                    1,
                    manifest["lifecycle"]["event_counts"]["measurement_complete"],
                )
                self.assertEqual(
                    1,
                    manifest["lifecycle"]["event_counts"]["server_stopped"],
                )

            serialized = b"\n".join(first_bytes.values()).decode("utf-8")
            for private_value in (
                MARKER_RAW_SENTINEL,
                RAW_COMPLETION,
                RAW_REASONING,
                RAW_REQUEST_ID,
                RAW_HOST_PATH,
                RAW_SECRET,
                *marker_nonces,
                *(str(value) for value in marker_monotonic_values),
            ):
                with self.subTest(private_value=private_value):
                    self.assertNotIn(private_value, serialized)

            projected_keys: set[str] = set()
            for relative_path, payload in first_bytes.items():
                if relative_path.endswith(".json"):
                    projected_keys.update(_json_keys(json.loads(payload)))
            self.assertTrue({"completion_tokens", "decode_tps"} <= projected_keys)
            self.assertTrue(
                {
                    "monotonic_ns",
                    "plan_fingerprint",
                    "raw",
                    "run_nonce",
                }.isdisjoint(projected_keys)
            )


if __name__ == "__main__":
    unittest.main()
