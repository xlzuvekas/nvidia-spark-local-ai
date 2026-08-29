"""Scalar evidence contracts for the paired SM121 cache-semantic canary.

The source fixture intentionally carries private response text, validation
text, and request IDs.  The B/A evidence bundles may retain only typed scalar
cache observations and correctness counters; they make no timing, TPS, or
energy claim.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from bench.audit import (
    audit_sm121_cache_semantic_arm_run,
    audit_sm121_cache_semantic_pair,
)
from bench.evidence import EvidenceError, SCHEMA_VERSION, export_evidence, verify_evidence
from bench.journal import content_hash
from bench.manifest import load_models, load_suite
from bench.report import summarize_run
from bench.runner import create_sm121_cache_semantic_pair_plans
from bench.sglang_sm121_cache_observability import (
    SM121_CACHE_OBSERVABILITY_CACHED_SERIES,
    SM121_CACHE_SOURCE_DIGESTS,
)
from bench.sglang_sm121_cache_semantic import (
    SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
    SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID,
    SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
    SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID,
    SM121_CACHE_SEMANTIC_CASE_ID,
    SM121_CACHE_SEMANTIC_EXECUTION_MODE,
    SM121_CACHE_SEMANTIC_MAX_MAMBA_CACHE_SIZE,
    SM121_CACHE_SEMANTIC_METRIC_FIELDS,
    SM121_CACHE_SEMANTIC_QUALITY_CASE_ID,
    SM121_CACHE_SEMANTIC_RUNTIME_ATTESTATION_EVENT,
    SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED,
    SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
    SM121_CACHE_SEMANTIC_STATIC_ATTESTATION_EVENT,
    SM121_CACHE_SEMANTIC_SUITE_ID,
    SM121_CACHE_SEMANTIC_TURN_OBSERVATION_EVENT,
    SM121_CACHE_SEMANTIC_TURN_ORDER,
    derive_sm121_cache_semantic_turn_admission,
    sm121_cache_semantic_cache_off_receipt_sha256,
    sm121_cache_semantic_pair_binding_sha256,
    sm121_cache_semantic_pair_instance_sha256,
)
from bench.sglang_sm121_storage import (
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_SOURCE_TREE,
)
import tests.test_evidence as evidence_test_support


PRIVATE_COMPLETION = "SEMANTIC_PRIVATE_COMPLETION_SENTINEL"
PRIVATE_REASON = "SEMANTIC_PRIVATE_VALIDATION_REASON_SENTINEL"
PRIVATE_REQUEST_ID = "SEMANTIC_PRIVATE_REQUEST_ID_SENTINEL"


class SM121CacheSemanticEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = Path(__file__).resolve().parents[1]
        models = load_models(self.repository / "manifests" / "models.toml")
        self.cache_off_model = models[SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID]
        self.cache_on_model = models[SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID]
        self.suite_path = (
            self.repository
            / "manifests"
            / "suites"
            / "qwen38_flash_next_sm121_triton_storage_cache_policy_semantic_canary.toml"
        )
        self.suite = load_suite(self.suite_path)

    def _freeze_pair(
        self, fixture: evidence_test_support.EvidenceFixture
    ) -> tuple[Path, Path]:
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
            return create_sm121_cache_semantic_pair_plans(
                cache_off_model=self.cache_off_model,
                cache_on_model=self.cache_on_model,
                suite=self.suite,
                results_root=fixture.results,
                models_path=self.repository / "manifests" / "models.toml",
                suite_path=self.suite_path,
            )

    @staticmethod
    def _rewrite_plan_integrity(
        fixture: evidence_test_support.EvidenceFixture,
        run_dir: Path,
        plan: dict[str, object],
    ) -> None:
        plan["integrity_hash"] = content_hash(
            {key: value for key, value in plan.items() if key != "integrity_hash"},
            64,
        )
        fixture.write_json(run_dir / "plan.json", plan)

    @staticmethod
    def _request_result(
        *, prompt_tokens: int, completion_tokens: int
    ) -> dict[str, object]:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": 0,
            "ttft_s": 0.1,
            "elapsed_s": 1.0,
            "decode_s": 0.9,
            "decode_tps": float(max(completion_tokens - 1, 0)) / 0.9,
            "output_tps": float(completion_tokens),
            "emission_events": completion_tokens,
            "finish_reason": "stop",
            "response_model": "synthetic",
            "decode_metric_source": "client_estimate",
            "load_s": None,
            "server_prompt_s": None,
            "cached_prompt_tokens": None,
            "server_cached_prompt_tokens": None,
            "server_decode_tokens": None,
            "server_decode_s": None,
            "server_prompt_tokens": None,
        }

    @staticmethod
    def _turn_event(
        *, arm: str, turn: str, case_id: str, attempt_id: str, admitted: bool
    ) -> dict[str, object]:
        """Build a complete scalar turn, including a valid partial state."""

        turn_index = SM121_CACHE_SEMANTIC_TURN_ORDER.index(turn)
        prompt_tokens = (32_768, 33_024, 33_280)[turn_index]
        shared_prefix_tokens = (0, 32_768, 33_024)[turn_index]
        positive_device = arm == SM121_CACHE_SEMANTIC_CACHE_ON_ARM and turn != "T0"
        response_device = shared_prefix_tokens if positive_device else 0
        before = {metric: 0 for metric in SM121_CACHE_SEMANTIC_METRIC_FIELDS}
        after = dict(before)
        after["prefill_input_tokens"] = prompt_tokens
        if positive_device:
            after["prefill_device_hit_tokens"] = response_device
            after["cached_total_tokens"] = response_device
            after["cached_device_tokens"] = response_device
        metrics_available = admitted or turn != "T1"
        event: dict[str, object] = {
            "event": SM121_CACHE_SEMANTIC_TURN_OBSERVATION_EVENT,
            "case_id": case_id,
            "protocol_case_id": SM121_CACHE_SEMANTIC_CASE_ID,
            "attempt_id": attempt_id,
            "turn": turn,
            "arm": arm,
            "cache_details_requested": True,
            "prompt_token_ids_requested": True,
            "streaming": False,
            "thinking_disabled": True,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 1,
            "reasoning_tokens": 0,
            "append_only_prompt_identity_verified": True,
            "cross_arm_prompt_identity_verified": (
                None if arm == SM121_CACHE_SEMANTIC_CACHE_OFF_ARM else True
            ),
            "shared_prefix_tokens": shared_prefix_tokens,
            "response_detail_state": (
                "nonzero_details" if positive_device else "zero_details"
            ),
            "usage_detail_state": "nonzero_details" if positive_device else "zero_details",
            "response_device_cached_tokens": response_device,
            "response_host_cached_tokens": 0,
            "response_storage_cached_tokens": 0,
            "usage_cached_tokens": response_device,
            "metrics_available": metrics_available,
            "guardrail_metrics_available": metrics_available,
            "metrics_before_polls": 2 if metrics_available else 1,
            "metrics_after_polls": 2 if metrics_available else 1,
            "metrics_before_settled": metrics_available,
            "metrics_after_settled": metrics_available,
        }
        for metric in SM121_CACHE_SEMANTIC_METRIC_FIELDS:
            event[f"before_{metric}"] = before[metric]
            event[f"after_{metric}"] = after[metric]
            event[f"delta_{metric}"] = after[metric] - before[metric]
        for prefix in ("before", "after"):
            for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
                event[f"{prefix}_cached_{source}_series_present"] = False
        turn_admitted, basis = derive_sm121_cache_semantic_turn_admission(event)
        event["semantic_turn_admitted"] = turn_admitted
        event["semantic_turn_basis"] = basis
        return event

    @staticmethod
    def _static_event(arm: str, lifetime: int) -> dict[str, object]:
        return {
            "event": SM121_CACHE_SEMANTIC_STATIC_ATTESTATION_EVENT,
            "arm": arm,
            "fresh_server_lifetime": lifetime,
            "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
            **SM121_CACHE_SOURCE_DIGESTS,
            **SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
        }

    @staticmethod
    def _runtime_event(arm: str, lifetime: int) -> dict[str, object]:
        return {
            "event": SM121_CACHE_SEMANTIC_RUNTIME_ATTESTATION_EVENT,
            "arm": arm,
            "fresh_server_lifetime": lifetime,
            **SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED[arm],
            "mamba_radix_cache_strategy": "extra_buffer_lazy",
            "max_mamba_cache_size": SM121_CACHE_SEMANTIC_MAX_MAMBA_CACHE_SIZE,
        }

    def _write_arm(
        self,
        run_dir: Path,
        *,
        arm: str,
        cache_off_fingerprint: str | None,
        cache_off_pair_binding_sha256: str | None = None,
        partial: bool,
    ) -> None:
        plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
        quality_case_id, semantic_case_id = [
            case["case_id"] for case in plan["suite"]["cases"]
        ]
        binding = plan["semantic_pair"]
        if arm == SM121_CACHE_SEMANTIC_CACHE_OFF_ARM:
            cache_off_terminal_receipt_sha256: str | None = None
        else:
            assert isinstance(cache_off_fingerprint, str)
            assert isinstance(cache_off_pair_binding_sha256, str)
            cache_off_terminal_receipt_sha256 = (
                sm121_cache_semantic_cache_off_receipt_sha256(
                    binding["pair_instance_sha256"],
                    cache_off_fingerprint,
                    cache_off_pair_binding_sha256,
                )
            )
        events: list[dict[str, object]] = []
        start = datetime(2026, 8, 28, 2, 0, tzinfo=timezone.utc)

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
                "execution_mode": SM121_CACHE_SEMANTIC_EXECUTION_MODE,
                "arm": arm,
                "plan_fingerprint": plan["fingerprint"],
                "semantic_pair_binding_sha256": binding["pair_binding_sha256"],
                "cache_off_plan_fingerprint": cache_off_fingerprint,
                "cache_off_audit_passed": None if arm == "B" else True,
                "cache_off_terminal_receipt_sha256": (
                    cache_off_terminal_receipt_sha256
                ),
            }
        )
        append({"event": "measurement_started"})
        append(self._static_event(arm, 1))
        append(self._runtime_event(arm, 1))
        append(
            {
                "event": "server_ready",
                "backend": "sglang",
                "fresh_server_lifetime": 1,
                "first_inference_is_case": True,
                "case_id": quality_case_id,
            }
        )
        quality_attempt = "1" * 32
        append(
            {
                "event": "case_start",
                "case_id": quality_case_id,
                "attempt_id": quality_attempt,
                "kind": "quality",
                "concurrency": 1,
            }
        )
        categories = (
            "arithmetic",
            "logic",
            "instruction_following",
            "code_reasoning",
        )
        for index, category in enumerate(categories):
            result = self._request_result(prompt_tokens=64 + index, completion_tokens=2)
            result.update(
                {
                    "request_id": PRIVATE_REQUEST_ID,
                    "content": PRIVATE_COMPLETION,
                    "reasoning": PRIVATE_REASON,
                    "tool_calls": [],
                }
            )
            append(
                {
                    "event": "request_complete",
                    "case_id": quality_case_id,
                    "attempt_id": quality_attempt,
                    "kind": "quality",
                    "repetition": 0,
                    "burst_elapsed_s": 1.0,
                    "result": result,
                    "validation": {
                        "passed": True,
                        "expected_answer": "synthetic",
                        "extracted_answer": PRIVATE_COMPLETION,
                        "quality_category": category,
                        "quality_item_id": f"{category}-01",
                    },
                }
            )
        append(
            {
                "event": "case_complete",
                "case_id": quality_case_id,
                "attempt_id": quality_attempt,
                "kind": "quality",
                "concurrency": 1,
                "elapsed_s": 4.0,
                "validation_passed": True,
            }
        )
        append(
            {
                "event": "server_stopped",
                "backend": "sglang",
                "fresh_server_lifetime": 1,
            }
        )
        append(self._static_event(arm, 2))
        append(self._runtime_event(arm, 2))
        append(
            {
                "event": "server_ready",
                "backend": "sglang",
                "fresh_server_lifetime": 2,
                "first_inference_is_case": True,
                "case_id": semantic_case_id,
            }
        )
        semantic_attempt = "2" * 32
        append(
            {
                "event": "case_start",
                "case_id": semantic_case_id,
                "attempt_id": semantic_attempt,
                "kind": "capability",
                "concurrency": 1,
            }
        )
        turns: list[dict[str, object]] = []
        for index, turn in enumerate(SM121_CACHE_SEMANTIC_TURN_ORDER):
            observation = self._turn_event(
                arm=arm,
                turn=turn,
                case_id=semantic_case_id,
                attempt_id=semantic_attempt,
                admitted=not partial,
            )
            turns.append(observation)
            append(observation)
            result = self._request_result(
                prompt_tokens=int(observation["prompt_tokens"]), completion_tokens=1
            )
            result.update(
                {
                    "request_id": PRIVATE_REQUEST_ID,
                    "content": PRIVATE_COMPLETION,
                    "reasoning": PRIVATE_REASON,
                    "tool_calls": [],
                }
            )
            append(
                {
                    "event": "request_complete",
                    "case_id": semantic_case_id,
                    "attempt_id": semantic_attempt,
                    "kind": "capability",
                    "repetition": index,
                    "burst_elapsed_s": 1.0,
                    "result": result,
                    "validation": {"passed": observation["semantic_turn_admitted"]},
                }
            )
        append(
            {
                "event": "case_complete",
                "case_id": semantic_case_id,
                "attempt_id": semantic_attempt,
                "kind": "capability",
                "concurrency": 1,
                "elapsed_s": 3.0,
                "validation_passed": all(
                    event["semantic_turn_admitted"] is True for event in turns
                ),
            }
        )
        append(
            {
                "event": "server_stopped",
                "backend": "sglang",
                "fresh_server_lifetime": 2,
            }
        )
        append({"event": "measurement_complete"})
        append({"event": "run_complete", "status": "completed"})
        with (run_dir / "events.jsonl").open("w", encoding="utf-8") as stream:
            for event in events:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
        summarize_run(run_dir)

    @staticmethod
    def _export(fixture: evidence_test_support.EvidenceFixture) -> dict[str, object]:
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
        bundle = fixture.output / "runs" / run_id
        files = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(bundle.iterdir())
            if path.name != "checksums.json"
        }
        fixture.write_json(
            bundle / "checksums.json",
            {"files": files, "schema_version": SCHEMA_VERSION},
        )
        index_path = fixture.output / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        entry = next(item for item in index["runs"] if item["run_id"] == run_id)
        entry["bundle_sha256"] = hashlib.sha256(
            (bundle / "checksums.json").read_bytes()
        ).hexdigest()
        fixture.write_json(index_path, index)
        top = {
            str(path.relative_to(fixture.output)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(fixture.output.rglob("*"))
            if path.is_file() and path != fixture.output / "checksums.json"
        }
        fixture.write_json(
            fixture.output / "checksums.json",
            {"files": top, "schema_version": SCHEMA_VERSION},
        )

    def test_pair_export_is_scalar_only_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            cache_off, cache_on = self._freeze_pair(fixture)
            cache_off_plan = json.loads((cache_off / "plan.json").read_text())
            self._write_arm(
                cache_off,
                arm=SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
                cache_off_fingerprint=None,
                partial=False,
            )
            self._write_arm(
                cache_on,
                arm=SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
                cache_off_fingerprint=cache_off_plan["fingerprint"],
                cache_off_pair_binding_sha256=cache_off_plan["semantic_pair"][
                    "pair_binding_sha256"
                ],
                partial=False,
            )
            self.assertTrue(audit_sm121_cache_semantic_arm_run(cache_off)["ok"])
            self.assertTrue(audit_sm121_cache_semantic_arm_run(cache_on)["ok"])
            self.assertTrue(audit_sm121_cache_semantic_pair(cache_off, cache_on)["ok"])

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

            for run_dir, arm in (
                (cache_off, SM121_CACHE_SEMANTIC_CACHE_OFF_ARM),
                (cache_on, SM121_CACHE_SEMANTIC_CACHE_ON_ARM),
            ):
                bundle = fixture.output / "runs" / run_dir.name
                manifest = json.loads((bundle / "manifest.json").read_text())
                samples = json.loads((bundle / "samples.json").read_text())
                aggregates = json.loads((bundle / "summary.json").read_text())["aggregates"]
                self.assertEqual("complete", manifest["status"])
                self.assertNotIn("journal_elapsed_s", manifest["lifecycle"])
                self.assertEqual(7, samples["sample_count"])
                self.assertEqual(arm, manifest["runtime"]["sm121_cache_semantic"]["arm"])
                self.assertEqual([], json.loads((bundle / "telemetry.json").read_text())["chunks"])
                self.assertNotIn("elapsed_s", samples["samples"][0])
                self.assertNotIn("aggregate_output_tps", aggregates["cases"][0])
                self.assertTrue(aggregates["cases"][0]["observability_only"])
                raw_summary = json.loads((run_dir / "summary.json").read_text())
                self.assertIsNone(raw_summary["artifact_validation"])
                self.assertIsNone(raw_summary["startup_telemetry"])
                self.assertIsNone(raw_summary["first_request_after_start"])
                self.assertIsNone(raw_summary["shutdown_telemetry"])
                for case in raw_summary["cases"]:
                    self.assertIsNone(case["elapsed_s"])
                    self.assertIsNone(case["aggregate_output_tps"])
                    self.assertIsNone(case["request_tps"])

            serialized = b"\n".join(original.values()).decode("utf-8", errors="ignore")
            for private in (PRIVATE_COMPLETION, PRIVATE_REASON, PRIVATE_REQUEST_ID):
                self.assertNotIn(private, serialized)
            published_keys = set().union(
                *(
                    evidence_test_support.json_keys(json.loads(path.read_text()))
                    for run_dir in (cache_off, cache_on)
                    for path in (fixture.output / "runs" / run_dir.name).glob("*.json")
                )
            )
            self.assertFalse(
                {
                    "content",
                    "reasoning",
                    "request_id",
                    "prompt",
                    "messages",
                    "tool_calls",
                    "elapsed_s",
                    "decode_tps",
                    "output_tps",
                    "energy_j",
                }
                & published_keys
            )

    def test_partial_semantic_arm_exports_without_a_timing_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            cache_off, _cache_on = self._freeze_pair(fixture)
            self._write_arm(
                cache_off,
                arm=SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
                cache_off_fingerprint=None,
                partial=True,
            )
            self._export(fixture)
            self.assertEqual("verified", verify_evidence(fixture.output)["status"])
            bundle = fixture.output / "runs" / cache_off.name
            manifest = json.loads((bundle / "manifest.json").read_text())
            aggregates = json.loads((bundle / "summary.json").read_text())["aggregates"]
            self.assertEqual("partial", manifest["status"])
            self.assertEqual("partial", aggregates["status"])
            self.assertEqual(
                [aggregates["cases"][1]["case_id"]],
                aggregates["validation_failed_cases"],
            )

    def test_partial_pair_audit_authorizes_only_a_truly_unstarted_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            cache_off, cache_on = self._freeze_pair(fixture)
            self._write_arm(
                cache_off,
                arm=SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
                cache_off_fingerprint=None,
                partial=True,
            )
            report = audit_sm121_cache_semantic_pair(cache_off, cache_on)
            self.assertTrue(report["ok"])
            self.assertTrue(report["authorized_terminal_partial"])
            self.assertEqual(
                "cache_off_terminal_partial_cache_on_unstarted",
                report["topology"],
            )
            self.assertEqual(
                "unstarted",
                report["arms"][SM121_CACHE_SEMANTIC_CACHE_ON_ARM][
                    "execution_state"
                ],
            )
            serialized = json.dumps(report, sort_keys=True)
            self.assertNotIn(str(cache_off.resolve()), serialized)
            self.assertNotIn(str(cache_on.resolve()), serialized)
            self.assertNotIn("run_dir", serialized)

    def test_partial_export_rejects_touched_a_execution_artifact(self) -> None:
        for artifact in ("events.jsonl", "summary.json"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                fixture = evidence_test_support.EvidenceFixture(Path(directory))
                cache_off, cache_on = self._freeze_pair(fixture)
                self._write_arm(
                    cache_off,
                    arm=SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
                    cache_off_fingerprint=None,
                    partial=True,
                )
                if artifact == "events.jsonl":
                    fixture.write_jsonl(cache_on / artifact, [])
                else:
                    fixture.write_json(cache_on / artifact, {})
                with self.assertRaises(EvidenceError):
                    self._export(fixture)
                audit = audit_sm121_cache_semantic_pair(cache_off, cache_on)
                self.assertFalse(audit["ok"])
                serialized = json.dumps(audit, sort_keys=True)
                self.assertNotIn(str(cache_off.resolve()), serialized)
                self.assertNotIn(str(cache_on.resolve()), serialized)

    def test_partial_export_rejects_dangling_a_event_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            cache_off, cache_on = self._freeze_pair(fixture)
            self._write_arm(
                cache_off,
                arm=SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
                cache_off_fingerprint=None,
                partial=True,
            )
            os.symlink("missing-events.jsonl", cache_on / "events.jsonl")
            with self.assertRaises(EvidenceError):
                self._export(fixture)
            self.assertFalse(
                audit_sm121_cache_semantic_pair(cache_off, cache_on)["ok"]
            )

    def test_partial_export_authenticates_unstarted_a_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            cache_off, cache_on = self._freeze_pair(fixture)
            self._write_arm(
                cache_off,
                arm=SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
                cache_off_fingerprint=None,
                partial=True,
            )
            cache_on_plan = json.loads((cache_on / "plan.json").read_text())
            cache_on_plan["created_at"] = "tampered-static-plan"
            fixture.write_json(cache_on / "plan.json", cache_on_plan)
            with self.assertRaises(EvidenceError):
                self._export(fixture)

    def test_pair_rejects_instance_mismatch_after_plan_integrity_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            cache_off, cache_on = self._freeze_pair(fixture)
            replacement_instance = sm121_cache_semantic_pair_instance_sha256(
                "0" * 32, "1" * 32
            )
            for run_dir in (cache_off, cache_on):
                plan = json.loads((run_dir / "plan.json").read_text())
                binding = plan["semantic_pair"]
                binding["pair_instance_sha256"] = replacement_instance
                binding["pair_binding_sha256"] = (
                    sm121_cache_semantic_pair_binding_sha256(binding)
                )
                self._rewrite_plan_integrity(fixture, run_dir, plan)
            self._write_arm(
                cache_off,
                arm=SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
                cache_off_fingerprint=None,
                partial=True,
            )
            self.assertFalse(
                audit_sm121_cache_semantic_pair(cache_off, cache_on)["ok"]
            )
            with self.assertRaises(EvidenceError):
                self._export(fixture)

    def test_pair_rejects_changed_a_cache_off_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            cache_off, cache_on = self._freeze_pair(fixture)
            cache_off_plan = json.loads((cache_off / "plan.json").read_text())
            self._write_arm(
                cache_off,
                arm=SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
                cache_off_fingerprint=None,
                partial=False,
            )
            self._write_arm(
                cache_on,
                arm=SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
                cache_off_fingerprint=cache_off_plan["fingerprint"],
                cache_off_pair_binding_sha256=cache_off_plan["semantic_pair"][
                    "pair_binding_sha256"
                ],
                partial=False,
            )
            events = [
                json.loads(line)
                for line in (cache_on / "events.jsonl").read_text().splitlines()
            ]
            events[0]["cache_off_terminal_receipt_sha256"] = "sha256:" + "0" * 64
            fixture.write_jsonl(cache_on / "events.jsonl", events)
            self.assertFalse(
                audit_sm121_cache_semantic_arm_run(cache_on)["ok"]
            )
            self.assertFalse(
                audit_sm121_cache_semantic_pair(cache_off, cache_on)["ok"]
            )
            with self.assertRaises(EvidenceError):
                self._export(fixture)

    def test_verifier_rejects_checksum_refreshed_timing_injection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            cache_off, _cache_on = self._freeze_pair(fixture)
            self._write_arm(
                cache_off,
                arm=SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
                cache_off_fingerprint=None,
                partial=True,
            )
            self._export(fixture)
            samples_path = fixture.output / "runs" / cache_off.name / "samples.json"
            samples = json.loads(samples_path.read_text(encoding="utf-8"))
            samples["samples"][0]["elapsed_s"] = 1.0
            fixture.write_json(samples_path, samples)
            self._refresh_checksums(fixture, cache_off.name)
            with self.assertRaises(EvidenceError):
                verify_evidence(fixture.output)

    def test_verifier_rejects_checksum_refreshed_orphan_a_arm(self) -> None:
        """A refreshed archive cannot turn a completed pair into an A-only claim."""

        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            cache_off, cache_on = self._freeze_pair(fixture)
            cache_off_plan = json.loads((cache_off / "plan.json").read_text())
            self._write_arm(
                cache_off,
                arm=SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
                cache_off_fingerprint=None,
                partial=False,
            )
            self._write_arm(
                cache_on,
                arm=SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
                cache_off_fingerprint=cache_off_plan["fingerprint"],
                cache_off_pair_binding_sha256=cache_off_plan["semantic_pair"][
                    "pair_binding_sha256"
                ],
                partial=False,
            )
            self._export(fixture)

            shutil.rmtree(fixture.output / "runs" / cache_off.name)
            index_path = fixture.output / "index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["runs"] = [
                entry
                for entry in index["runs"]
                if entry["run_id"] != cache_off.name
            ]
            index["run_count"] = len(index["runs"])
            index["run_status_counts"] = {"complete": 1}
            fixture.write_json(index_path, index)
            top = {
                str(path.relative_to(fixture.output)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(fixture.output.rglob("*"))
                if path.is_file() and path != fixture.output / "checksums.json"
            }
            fixture.write_json(
                fixture.output / "checksums.json",
                {"files": top, "schema_version": SCHEMA_VERSION},
            )
            with self.assertRaisesRegex(EvidenceError, "lacks its B control"):
                verify_evidence(fixture.output)

    def test_verifier_rejects_public_pair_instance_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            cache_off, cache_on = self._freeze_pair(fixture)
            cache_off_plan = json.loads((cache_off / "plan.json").read_text())
            self._write_arm(
                cache_off,
                arm=SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
                cache_off_fingerprint=None,
                partial=False,
            )
            self._write_arm(
                cache_on,
                arm=SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
                cache_off_fingerprint=cache_off_plan["fingerprint"],
                cache_off_pair_binding_sha256=cache_off_plan["semantic_pair"][
                    "pair_binding_sha256"
                ],
                partial=False,
            )
            self._export(fixture)
            manifest_path = fixture.output / "runs" / cache_on.name / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            binding = manifest["runtime"]["sm121_cache_semantic"]["pair_binding"]
            replacement_instance = "sha256:" + "f" * 64
            self.assertNotEqual(replacement_instance, binding["pair_instance_sha256"])
            binding["pair_instance_sha256"] = replacement_instance
            binding["pair_binding_sha256"] = sm121_cache_semantic_pair_binding_sha256(
                binding
            )
            fixture.write_json(manifest_path, manifest)
            self._refresh_checksums(fixture, cache_on.name)
            with self.assertRaisesRegex(EvidenceError, "not reciprocal"):
                verify_evidence(fixture.output)

    def test_export_rejects_raw_prompt_token_ids_in_turn_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            cache_off, _cache_on = self._freeze_pair(fixture)
            self._write_arm(
                cache_off,
                arm=SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
                cache_off_fingerprint=None,
                partial=True,
            )
            events = [
                json.loads(line)
                for line in (cache_off / "events.jsonl").read_text().splitlines()
            ]
            turn = next(
                event
                for event in events
                if event.get("event") == SM121_CACHE_SEMANTIC_TURN_OBSERVATION_EVENT
            )
            turn["prompt_token_ids"] = [1, 2, 3]
            fixture.write_jsonl(cache_off / "events.jsonl", events)
            summarize_run(cache_off)
            with self.assertRaises(EvidenceError):
                self._export(fixture)

    def test_completed_b_requires_a_started_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            cache_off, cache_on = self._freeze_pair(fixture)
            self._write_arm(
                cache_off,
                arm=SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
                cache_off_fingerprint=None,
                partial=False,
            )
            audit = audit_sm121_cache_semantic_pair(cache_off, cache_on)
            self.assertFalse(audit["ok"])
            self.assertIn(
                "semantic_pair_unstarted_candidate",
                {issue["code"] for issue in audit["errors"]},
            )
            with self.assertRaises(EvidenceError):
                self._export(fixture)


if __name__ == "__main__":
    unittest.main()
