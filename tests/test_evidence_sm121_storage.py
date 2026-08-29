"""Scalar evidence contracts for the singleton SM121 storage canary."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench.evidence import EvidenceError, SCHEMA_VERSION, export_evidence, verify_evidence
from bench.manifest import load_models, load_suite
from bench.report import summarize_run
from bench.runner import create_sm121_storage_canary_plan
from bench.seccomp_profile_contract import DERIVED_SHA256
from bench.sglang_sm121_storage import (
    SM121_STORAGE_BUILD_CONTRACT_SHA256,
    SM121_STORAGE_CACHE_PAGES,
    SM121_STORAGE_CANDIDATE_ID,
    SM121_STORAGE_EXECUTION_MODE,
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_MAX_BATCH_PAGES,
    SM121_STORAGE_MODE,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_PROFILE_ID,
    SM121_STORAGE_QUEUE_DEPTH,
    SM121_STORAGE_RUNTIME_PROVENANCE_EVENT,
    SM121_STORAGE_SOURCE_TREE,
    SM121_STORAGE_VARIED_CONTEXT_CHAT_PROMPT_TOKENS,
)
import tests.test_evidence as evidence_test_support


PRIVATE_COMPLETION = "SM121_PRIVATE_COMPLETION_SENTINEL"
PRIVATE_REASON = "SM121_PRIVATE_VALIDATION_REASON_SENTINEL"
PRIVATE_REQUEST_ID = "SM121_PRIVATE_REQUEST_ID_SENTINEL"


def _native_provenance() -> dict[str, object]:
    return {
        "candidate_id": SM121_STORAGE_CANDIDATE_ID,
        "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
        "build_contract_sha256": SM121_STORAGE_BUILD_CONTRACT_SHA256,
        "docker_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
        "sglang_storage_mode": SM121_STORAGE_MODE,
        "sglang_ple_nvme_backend": "io_uring",
        "sglang_ple_nvme_queue_depth": SM121_STORAGE_QUEUE_DEPTH,
        "sglang_ple_nvme_max_batch_pages": SM121_STORAGE_MAX_BATCH_PAGES,
        "sglang_ple_nvme_cache_pages": SM121_STORAGE_CACHE_PAGES,
        "sglang_rust_build_mode": "never",
        "seccomp_profile_sha256": "sha256:" + DERIVED_SHA256,
        "container_rootfs": "readonly_tmpfs_writable_cache",
        "container_capabilities": "dropped_all",
        "container_no_new_privileges": True,
        "hf_network_policy": "offline",
        "network_topology": "loopback_published_bridge",
        "benchmark_scope": "sm121_storage_pre_admission_canary",
        "model_acquisition": "disabled_exact_read_only_snapshot",
        "api_authentication": "ephemeral_bearer",
        "api_key_file_mode": "0600",
    }


class SM121StorageEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = Path(__file__).resolve().parents[1]
        self.model = load_models(self.repository / "manifests" / "models.toml")[
            SM121_STORAGE_PROFILE_ID
        ]
        self.suite_path = (
            self.repository
            / "manifests"
            / "suites"
            / "qwen38_flash_next_sm121_triton_storage_canary.toml"
        )
        self.suite = load_suite(self.suite_path)

    def _write_canary(
        self,
        fixture: evidence_test_support.EvidenceFixture,
        *,
        varied_context_passed: bool,
    ) -> Path:
        with (
            patch("bench.runner._image_digest", return_value=None),
            patch(
                "bench.runner._sm121_storage_image_identity",
                return_value={
                    "docker_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
                    "platform": SM121_STORAGE_PLATFORM,
                    "source_tree": SM121_STORAGE_SOURCE_TREE,
                },
            ),
            patch("bench.runner._host_snapshot", return_value={"host": "fixture"}),
        ):
            run_dir = create_sm121_storage_canary_plan(
                model=self.model,
                suite=self.suite,
                results_root=fixture.results,
                models_path=self.repository / "manifests" / "models.toml",
                suite_path=self.suite_path,
            )
        plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
        case_ids = [case["case_id"] for case in plan["suite"]["cases"]]
        events: list[dict[str, object]] = []
        start = datetime(2026, 8, 28, 1, 0, tzinfo=timezone.utc)

        def append(event: dict[str, object]) -> None:
            events.append(
                {
                    "timestamp": (start + timedelta(seconds=len(events))).isoformat(),
                    **event,
                }
            )

        append(
            {
                "event": "run_start",
                "execution_mode": SM121_STORAGE_EXECUTION_MODE,
                "plan_fingerprint": plan["fingerprint"],
                "run_nonce": plan["run_nonce"],
            }
        )
        append(
            {
                "event": "measurement_started",
                "monotonic_ns": 1,
                "plan_fingerprint": plan["fingerprint"],
                "run_nonce": plan["run_nonce"],
            }
        )
        quality_categories = (
            "arithmetic",
            "logic",
            "instruction_following",
            "code_reasoning",
        )
        quality_ids = (
            "arithmetic-01",
            "logic-01",
            "instruction-01",
            "code-01",
        )
        for lifetime, case_id in enumerate(case_ids, start=1):
            append(
                {
                    "event": SM121_STORAGE_RUNTIME_PROVENANCE_EVENT,
                    "fresh_server_lifetime": lifetime,
                    **_native_provenance(),
                }
            )
            append(
                {
                    "event": "server_ready",
                    "backend": "sglang",
                    "startup_s": float(lifetime),
                    "fresh_server_lifetime": lifetime,
                    "first_inference_is_case": True,
                    "case_id": case_id,
                }
            )
            attempt_id = str(lifetime) * 32
            quality = lifetime == 1
            kind = "quality" if quality else "capability"
            append(
                {
                    "event": "case_start",
                    "case_id": case_id,
                    "attempt_id": attempt_id,
                    "kind": kind,
                    "concurrency": 1,
                }
            )
            request_count = 8 if quality else 1
            for request_index in range(request_count):
                passed = True if quality else varied_context_passed
                result: dict[str, object] = {
                    "prompt_tokens": (
                        20
                        if quality
                        else SM121_STORAGE_VARIED_CONTEXT_CHAT_PROMPT_TOKENS
                    ),
                    "completion_tokens": 2,
                    "reasoning_tokens": None,
                    "ttft_s": 0.1,
                    "elapsed_s": 1.0,
                    "decode_s": 0.9,
                    "decode_tps": 1.1,
                    "output_tps": 2.0,
                    "emission_events": 2,
                    "finish_reason": "stop",
                    "response_model": "synthetic",
                    "decode_metric_source": "client_estimate",
                    "load_s": None,
                    "server_prompt_s": 0.2,
                }
                if quality:
                    result.update(
                        {
                            "request_id": PRIVATE_REQUEST_ID,
                            "started_at_ns": 1,
                            "content": PRIVATE_COMPLETION,
                            "reasoning": PRIVATE_REASON,
                            "tool_calls": [],
                            "cached_prompt_tokens": None,
                            "server_prompt_tokens": None,
                            "server_cached_prompt_tokens": None,
                            "server_decode_tokens": None,
                            "server_decode_s": None,
                        }
                    )
                validation: dict[str, object] = {
                    "passed": passed,
                    "reason": None if passed else PRIVATE_REASON,
                }
                if quality:
                    item = request_index % 4
                    validation.update(
                        {
                            "expected_answer": "synthetic",
                            "extracted_answer": PRIVATE_COMPLETION,
                            "quality_category": quality_categories[item],
                            "quality_item_id": quality_ids[item],
                        }
                    )
                append(
                    {
                        "event": "request_complete",
                        "case_id": case_id,
                        "attempt_id": attempt_id,
                        "kind": kind,
                        "repetition": request_index // 4 if quality else 0,
                        "burst_elapsed_s": 1.0,
                        "result": result,
                        "validation": validation,
                    }
                )
            append(
                {
                    "event": "case_complete",
                    "case_id": case_id,
                    "attempt_id": attempt_id,
                    "kind": kind,
                    "concurrency": 1,
                    "elapsed_s": float(request_count),
                    "validation_passed": True if quality else varied_context_passed,
                }
            )
            append(
                {
                    "event": "server_stopped",
                    "backend": "sglang",
                    "fresh_server_lifetime": lifetime,
                }
            )
        append({"event": "measurement_complete", "elapsed_s": 20.0, "monotonic_ns": 2})
        append({"event": "run_complete", "status": "completed"})
        fixture.write_jsonl(run_dir / "events.jsonl", events)
        summarize_run(run_dir)
        return run_dir

    @staticmethod
    def _export(
        fixture: evidence_test_support.EvidenceFixture,
    ) -> dict[str, object]:
        with patch(
            "bench.evidence._export_campaign",
            side_effect=evidence_test_support.EvidenceExportTests.fake_campaign_export,
        ):
            return export_evidence(
                results_root=fixture.results,
                output_root=fixture.output,
            )

    @staticmethod
    def _refresh_checksums(
        fixture: evidence_test_support.EvidenceFixture, run_id: str
    ) -> None:
        root = fixture.output
        bundle = root / "runs" / run_id
        files = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(bundle.iterdir())
            if path.name != "checksums.json"
        }
        fixture.write_json(
            bundle / "checksums.json",
            {"files": files, "schema_version": SCHEMA_VERSION},
        )
        index_path = root / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        entry = next(item for item in index["runs"] if item["run_id"] == run_id)
        entry["bundle_sha256"] = hashlib.sha256(
            (bundle / "checksums.json").read_bytes()
        ).hexdigest()
        fixture.write_json(index_path, index)
        top = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and path != root / "checksums.json"
        }
        fixture.write_json(
            root / "checksums.json",
            {"files": top, "schema_version": SCHEMA_VERSION},
        )

    def test_complete_and_validation_partial_export_deterministically_as_scalars(
        self,
    ) -> None:
        for passed, expected_status in ((True, "complete"), (False, "partial")):
            with self.subTest(status=expected_status), tempfile.TemporaryDirectory() as directory:
                fixture = evidence_test_support.EvidenceFixture(Path(directory))
                run_dir = self._write_canary(
                    fixture, varied_context_passed=passed
                )
                first = self._export(fixture)
                self.assertTrue(first["changed"])
                original = {
                    str(path.relative_to(fixture.output)): path.read_bytes()
                    for path in fixture.output.rglob("*")
                    if path.is_file()
                }
                second = self._export(fixture)
                self.assertFalse(second["changed"])
                self.assertEqual(
                    original,
                    {
                        str(path.relative_to(fixture.output)): path.read_bytes()
                        for path in fixture.output.rglob("*")
                        if path.is_file()
                    },
                )
                self.assertEqual("verified", verify_evidence(fixture.output)["status"])
                bundle = fixture.output / "runs" / run_dir.name
                manifest = json.loads((bundle / "manifest.json").read_text())
                samples = json.loads((bundle / "samples.json").read_text())
                self.assertEqual(expected_status, manifest["status"])
                self.assertEqual(9, samples["sample_count"])
                self.assertEqual(
                    SM121_STORAGE_LOCAL_IMAGE_ID.removeprefix("sha256:"),
                    manifest["runtime"]["sm121_storage_canary"][
                        "docker_image_sha256"
                    ],
                )
                serialized = b"\n".join(original.values()).decode(
                    "utf-8", errors="ignore"
                )
                for private in (
                    PRIVATE_COMPLETION,
                    PRIVATE_REASON,
                    PRIVATE_REQUEST_ID,
                ):
                    self.assertNotIn(private, serialized)
                self.assertFalse(
                    {
                        "content",
                        "reasoning",
                        "request_id",
                        "prompt",
                        "messages",
                        "tool_calls",
                    }
                    & evidence_test_support.json_keys(samples)
                )

    def test_export_rejects_tampered_runtime_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            run_dir = self._write_canary(fixture, varied_context_passed=True)
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            provenance = next(
                event
                for event in events
                if event.get("event") == SM121_STORAGE_RUNTIME_PROVENANCE_EVENT
            )
            provenance["sglang_ple_nvme_queue_depth"] = 1
            fixture.write_jsonl(run_dir / "events.jsonl", events)
            with self.assertRaisesRegex(EvidenceError, "lifecycle audit failed"):
                self._export(fixture)

    def test_verifier_rejects_checksum_refreshed_runtime_downgrade(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            run_dir = self._write_canary(fixture, varied_context_passed=True)
            self._export(fixture)
            manifest_path = fixture.output / "runs" / run_dir.name / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runtime"]["sm121_storage_canary"][
                "docker_image_sha256"
            ] = "0" * 64
            fixture.write_json(manifest_path, manifest)
            self._refresh_checksums(fixture, run_dir.name)
            with self.assertRaisesRegex(
                EvidenceError, "SM121 storage runtime identity changed"
            ):
                verify_evidence(fixture.output)


if __name__ == "__main__":
    unittest.main()
