"""Scalar-only evidence contracts for SM121 chunk-size prefill campaigns."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench.audit import audit_sm121_chunked_prefill_performance_campaign
from bench.evidence import EvidenceError, export_evidence, verify_evidence
from bench.journal import content_hash
from bench.manifest import load_models, load_suite
from bench.sglang_sm121_cache_observability import SM121_CACHE_SOURCE_DIGESTS
from bench.sglang_sm121_cache_semantic import SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS
from bench.sglang_sm121_chunked_prefill_performance import (
    SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CASE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_MAX_MAMBA_CACHE_SIZE,
    SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT,
    SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EVENT,
    SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EXPECTED,
    SM121_CHUNKED_PREFILL_PERFORMANCE_STATIC_EVENT,
    SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS,
    SM121_CHUNKED_PREFILL_PERFORMANCE_TURN_EVENT,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CANDIDATE_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CONTROL_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V2_STUDY,
    score_sm121_chunked_prefill_performance_campaign,
    sm121_chunked_prefill_performance_study,
)
from bench.sglang_sm121_storage import (
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_SOURCE_TREE,
)
from bench.sm121_chunked_prefill_runner import (
    create_sm121_chunked_prefill_performance_campaign,
)
from tests.test_sglang_sm121_chunked_prefill_performance import _turn
import tests.test_evidence as evidence_test_support


PRIVATE_PROMPT = "CHUNKED_PREFILL_PRIVATE_PROMPT_SENTINEL"
PRIVATE_COMPLETION = "CHUNKED_PREFILL_PRIVATE_COMPLETION_SENTINEL"
PRIVATE_REQUEST_ID = "CHUNKED_PREFILL_PRIVATE_REQUEST_ID_SENTINEL"


class SM121ChunkedPrefillEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = Path(__file__).resolve().parents[1]
        models = load_models(self.repository / "manifests" / "models.toml")
        self.control = models[SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID]
        self.candidate = models[SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID]
        self.suite_path = (
            self.repository
            / "manifests"
            / "suites"
            / "qwen38_flash_next_sm121_triton_storage_chunked_prefill_performance_v1.toml"
        )
        self.suite = load_suite(self.suite_path)
        self.v2_control = models[
            SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CONTROL_PROFILE_ID
        ]
        self.v2_candidate = models[
            SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CANDIDATE_PROFILE_ID
        ]
        self.v2_suite_path = (
            self.repository
            / "manifests"
            / "suites"
            / "qwen38_flash_next_sm121_triton_storage_chunked_prefill_performance_v2.toml"
        )
        self.v2_suite = load_suite(self.v2_suite_path)

    @staticmethod
    def _export(
        fixture: evidence_test_support.EvidenceFixture, *, replace: bool = False
    ) -> dict[str, object]:
        with patch(
            "bench.evidence._export_campaign",
            side_effect=evidence_test_support.EvidenceExportTests.fake_campaign_export,
        ):
            return export_evidence(
                results_root=fixture.results, output_root=fixture.output, replace=replace
            )

    def _freeze(
        self,
        fixture: evidence_test_support.EvidenceFixture,
        *,
        control: object | None = None,
        candidate: object | None = None,
        suite: object | None = None,
        suite_path: Path | None = None,
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
            return create_sm121_chunked_prefill_performance_campaign(
                control_model=self.control if control is None else control,
                candidate_model=self.candidate if candidate is None else candidate,
                suite=self.suite if suite is None else suite,
                results_root=fixture.results / "chunked-prefill-campaigns",
                models_path=self.repository / "manifests" / "models.toml",
                suite_path=self.suite_path if suite_path is None else suite_path,
            )

    @staticmethod
    def _static_event(
        arm: str, lifetime_ordinal: int, *, control_chunk_size: int, candidate_chunk_size: int
    ) -> dict[str, object]:
        return {
            "event": SM121_CHUNKED_PREFILL_PERFORMANCE_STATIC_EVENT,
            "arm": arm,
            "lifetime_ordinal": lifetime_ordinal,
            "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
            "chunked_prefill_size": control_chunk_size if arm == "A" else candidate_chunk_size,
            **SM121_CACHE_SOURCE_DIGESTS,
            **SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
        }

    @staticmethod
    def _runtime_event(
        arm: str, lifetime_ordinal: int, *, control_chunk_size: int, candidate_chunk_size: int
    ) -> dict[str, object]:
        return {
            "event": SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EVENT,
            "arm": arm,
            "lifetime_ordinal": lifetime_ordinal,
            "mamba_radix_cache_strategy": "extra_buffer_lazy",
            "max_mamba_cache_size": SM121_CHUNKED_PREFILL_PERFORMANCE_MAX_MAMBA_CACHE_SIZE,
            "chunked_prefill_size": control_chunk_size if arm == "A" else candidate_chunk_size,
            **SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EXPECTED,
        }

    @staticmethod
    def _lifetime(
        *, ordinal: int, arm: str, timed_case_id: str
    ) -> dict[str, object]:
        turns: list[dict[str, object]] = []
        for index, turn in enumerate(SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS):
            observation = _turn(
                ordinal=ordinal * 2,
                arm=arm,
                turn=turn,
                wall_s=100.0 if turn == "T0" else 10.0 + index,
                timed_case_id=timed_case_id.rsplit("--", 1)[0],
            )
            observation["case_id"] = timed_case_id
            turns.append(observation)
        return {
            "ordinal": ordinal,
            "arm": arm,
            "quality_admitted": True,
            "timed_admitted": True,
            "within_timeout": True,
            "turns": turns,
        }

    def _write_completed_campaign(self, campaign_dir: Path) -> None:
        campaign = json.loads((campaign_dir / "campaign.json").read_text())
        study = sm121_chunked_prefill_performance_study(campaign["campaign_id"])
        lifetimes: list[dict[str, object]] = []
        start = datetime(2026, 8, 29, 2, 0, tzinfo=timezone.utc)
        for ordinal, (name, arm) in enumerate(
            zip(campaign["run_directories"], SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER),
            start=1,
        ):
            run_dir = campaign_dir / "runs" / name
            plan = json.loads((run_dir / "plan.json").read_text())
            case_ids = {case["id"]: case["case_id"] for case in plan["suite"]["cases"]}
            quality_case_id = case_ids["synthetic-exact-answer-v2"]
            timed_case_id = case_ids[study.timed_case_id]
            quality_ordinal, timed_ordinal = ordinal * 2 - 1, ordinal * 2
            events: list[dict[str, object]] = []

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
                    "execution_mode": study.execution_mode,
                    "arm": arm,
                    "campaign_ordinal": ordinal,
                    "plan_fingerprint": plan["fingerprint"],
                    "chunked_prefill_performance_pair_binding_sha256": plan[
                        "chunked_prefill_performance_pair"
                    ]["pair_binding_sha256"],
                    "private_prompt": PRIVATE_PROMPT,
                    "private_completion": PRIVATE_COMPLETION,
                    "private_request_id": PRIVATE_REQUEST_ID,
                }
            )
            append({"event": "measurement_started"})
            append(
                self._static_event(
                    arm,
                    quality_ordinal,
                    control_chunk_size=study.control_chunk_size,
                    candidate_chunk_size=study.candidate_chunk_size,
                )
            )
            append(
                self._runtime_event(
                    arm,
                    quality_ordinal,
                    control_chunk_size=study.control_chunk_size,
                    candidate_chunk_size=study.candidate_chunk_size,
                )
            )
            append(
                {
                    "event": "server_ready",
                    "backend": "sglang",
                    "lifetime_ordinal": quality_ordinal,
                    "phase": "quality",
                    "first_inference_is_case": True,
                    "case_id": quality_case_id,
                }
            )
            append(
                {
                    "event": "sm121_chunked_prefill_performance_quality_case_start",
                    "arm": arm,
                    "lifetime_ordinal": quality_ordinal,
                    "case_id": quality_case_id,
                }
            )
            append(
                {
                    "event": "sm121_chunked_prefill_performance_quality_case_complete",
                    "arm": arm,
                    "lifetime_ordinal": quality_ordinal,
                    "case_id": quality_case_id,
                    "quality_admitted": True,
                    "item_count": SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT,
                }
            )
            append(
                {
                    "event": "server_stopped",
                    "backend": "sglang",
                    "lifetime_ordinal": quality_ordinal,
                }
            )
            append(
                {
                    "event": "sm121_chunked_prefill_performance_lifetime_complete",
                    "arm": arm,
                    "lifetime_ordinal": quality_ordinal,
                    "phase": "quality",
                    "lifetime_wall_s": 1.0,
                    "within_timeout": True,
                    "admitted": True,
                }
            )
            append(
                self._static_event(
                    arm,
                    timed_ordinal,
                    control_chunk_size=study.control_chunk_size,
                    candidate_chunk_size=study.candidate_chunk_size,
                )
            )
            append(
                self._runtime_event(
                    arm,
                    timed_ordinal,
                    control_chunk_size=study.control_chunk_size,
                    candidate_chunk_size=study.candidate_chunk_size,
                )
            )
            append(
                {
                    "event": "server_ready",
                    "backend": "sglang",
                    "lifetime_ordinal": timed_ordinal,
                    "phase": "timed",
                    "first_inference_is_case": True,
                    "case_id": timed_case_id,
                }
            )
            append(
                {
                    "event": "sm121_chunked_prefill_performance_timed_case_start",
                    "arm": arm,
                    "lifetime_ordinal": timed_ordinal,
                    "case_id": timed_case_id,
                }
            )
            lifetime = self._lifetime(
                ordinal=ordinal, arm=arm, timed_case_id=timed_case_id
            )
            for observation in lifetime["turns"]:
                append(observation)
            append(
                {
                    "event": "sm121_chunked_prefill_performance_timed_case_complete",
                    "arm": arm,
                    "lifetime_ordinal": timed_ordinal,
                    "case_id": timed_case_id,
                    "timed_admitted": True,
                }
            )
            append(
                {
                    "event": "server_stopped",
                    "backend": "sglang",
                    "lifetime_ordinal": timed_ordinal,
                }
            )
            append(
                {
                    "event": "sm121_chunked_prefill_performance_lifetime_complete",
                    "arm": arm,
                    "lifetime_ordinal": timed_ordinal,
                    "phase": "timed",
                    "lifetime_wall_s": 2.0,
                    "within_timeout": True,
                    "admitted": True,
                }
            )
            append({"event": "measurement_complete"})
            append({"event": "run_complete", "status": "completed"})
            evidence_test_support.EvidenceFixture.write_jsonl(
                run_dir / "events.jsonl", events
            )
            lifetimes.append(lifetime)
        score = score_sm121_chunked_prefill_performance_campaign(
            lifetimes, study=study
        )
        summary: dict[str, object] = {
            "schema_version": 1,
            "campaign_id": campaign["campaign_id"],
            "execution_mode": campaign["execution_mode"],
            "pair_binding_sha256": campaign["pair_binding"]["pair_binding_sha256"],
            "status": score.status,
            "decision": score.decision,
            "completed_arms": 4,
            "lifetimes": lifetimes,
            "score": score.to_mapping(),
        }
        summary["integrity_hash"] = content_hash(summary, 64)
        evidence_test_support.EvidenceFixture.write_json(
            campaign_dir / "summary.json", summary
        )

    def test_export_is_scalar_only_deterministic_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            self.assertTrue(self._export(fixture)["changed"])
            campaign = self._freeze(fixture)
            self._write_completed_campaign(campaign)
            self.assertTrue(audit_sm121_chunked_prefill_performance_campaign(campaign)["ok"])
            first = self._export(fixture, replace=True)
            self.assertTrue(first["changed"])
            original = {
                str(path.relative_to(fixture.output)): path.read_bytes()
                for path in fixture.output.rglob("*")
                if path.is_file()
            }
            second = self._export(fixture, replace=True)
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
            index = json.loads((fixture.output / "index.json").read_text())
            entry = next(
                item
                for item in index["campaigns"]
                if item["evidence_kind"] == "sm121_chunked_prefill_performance"
            )
            manifest = json.loads((fixture.output / entry["file"]).read_text())
            self.assertEqual(4, manifest["completed_arms"])
            self.assertEqual(4, len(manifest["quality_attestations"]))
            serialized = b"\n".join(original.values()).decode("utf-8", errors="ignore")
            for private in (PRIVATE_PROMPT, PRIVATE_COMPLETION, PRIVATE_REQUEST_ID):
                self.assertNotIn(private, serialized)

    def test_v2_export_uses_a_distinct_protocol_and_rejects_cross_study_mixups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            self._export(fixture)
            campaign = self._freeze(
                fixture,
                control=self.v2_control,
                candidate=self.v2_candidate,
                suite=self.v2_suite,
                suite_path=self.v2_suite_path,
            )
            self._write_completed_campaign(campaign)
            report = audit_sm121_chunked_prefill_performance_campaign(campaign)
            self.assertTrue(report["ok"])
            self.assertEqual(
                SM121_CHUNKED_PREFILL_PERFORMANCE_V2_STUDY.campaign_id,
                report["campaign_id"],
            )
            self.assertTrue(self._export(fixture, replace=True)["changed"])
            self.assertEqual("verified", verify_evidence(fixture.output)["status"])
            index = json.loads((fixture.output / "index.json").read_text())
            entry = next(
                item
                for item in index["campaigns"]
                if item["evidence_kind"] == "sm121_chunked_prefill_performance"
                and item["campaign_id"].startswith(
                    SM121_CHUNKED_PREFILL_PERFORMANCE_V2_STUDY.campaign_id + "-"
                )
            )
            manifest = json.loads((fixture.output / entry["file"]).read_text())
            self.assertEqual(
                SM121_CHUNKED_PREFILL_PERFORMANCE_V2_STUDY.campaign_id,
                manifest["protocol"]["campaign_id"],
            )
            self.assertEqual([2048, 4096], manifest["protocol"]["chunked_prefill_sizes"])
            campaign_data = json.loads((campaign / "campaign.json").read_text())
            events_path = (
                campaign / "runs" / campaign_data["run_directories"][0] / "events.jsonl"
            )
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            next(
                event
                for event in events
                if event["event"] == SM121_CHUNKED_PREFILL_PERFORMANCE_STATIC_EVENT
            )["chunked_prefill_size"] = 1024
            evidence_test_support.EvidenceFixture.write_jsonl(events_path, events)
            self.assertFalse(audit_sm121_chunked_prefill_performance_campaign(campaign)["ok"])

    def test_export_retains_the_audited_bootstrap_counter_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            self._export(fixture)
            campaign = self._freeze(fixture)
            self._write_completed_campaign(campaign)
            campaign_data = json.loads((campaign / "campaign.json").read_text())
            names = campaign_data["run_directories"]
            first_events_path = campaign / "runs" / names[0] / "events.jsonl"
            events = [json.loads(line) for line in first_events_path.read_text().splitlines()]
            t0_index = next(
                index
                for index, event in enumerate(events)
                if event["event"] == SM121_CHUNKED_PREFILL_PERFORMANCE_TURN_EVENT
                and event["turn"] == "T0"
            )
            legacy_turn = dict(events[t0_index])
            legacy_turn.update(
                {
                    "before_prefill_input_tokens": 64,
                    "after_prefill_input_tokens": 65,
                    "delta_prefill_input_tokens": 1,
                    "timed_turn_admitted": False,
                    "timed_turn_basis": "cold_lifetime",
                }
            )
            timed_stop = next(
                dict(event)
                for event in events
                if event["event"] == "server_stopped"
                and event["lifetime_ordinal"] == 2
            )
            timed_complete = next(
                dict(event)
                for event in events
                if event["event"]
                == "sm121_chunked_prefill_performance_lifetime_complete"
                and event["lifetime_ordinal"] == 2
            )
            timed_complete["admitted"] = False
            partial_events = [
                *events[:t0_index],
                legacy_turn,
                timed_stop,
                timed_complete,
                {
                    "timestamp": "2026-08-29T02:01:00+00:00",
                    "event": "run_aborted",
                    "error_type": "SM121ChunkedPrefillPerformanceRequestError",
                    "error": "SM121 chunked-prefill performance request failed; details omitted",
                },
            ]
            evidence_test_support.EvidenceFixture.write_jsonl(
                first_events_path, partial_events
            )
            for name in names[1:]:
                evidence_test_support.EvidenceFixture.write_jsonl(
                    campaign / "runs" / name / "events.jsonl", []
                )
            summary_path = campaign / "summary.json"
            summary = json.loads(summary_path.read_text())
            first_lifetime = dict(summary["lifetimes"][0])
            first_lifetime.update(
                timed_admitted=False,
                within_timeout=False,
                turns=[{key: value for key, value in legacy_turn.items() if key != "timestamp"}],
            )
            summary["lifetimes"] = [
                first_lifetime,
                *[
                    {
                        "ordinal": ordinal,
                        "arm": arm,
                        "quality_admitted": False,
                        "timed_admitted": False,
                        "within_timeout": False,
                        "turns": [],
                    }
                    for ordinal, arm in enumerate(
                        SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER[1:], start=2
                    )
                ],
            ]
            score = score_sm121_chunked_prefill_performance_campaign(summary["lifetimes"])
            summary.update(
                status=score.status,
                decision=score.decision,
                completed_arms=0,
                score=score.to_mapping(),
            )
            summary["integrity_hash"] = content_hash(
                {key: value for key, value in summary.items() if key != "integrity_hash"},
                64,
            )
            evidence_test_support.EvidenceFixture.write_json(summary_path, summary)

            self.assertTrue(audit_sm121_chunked_prefill_performance_campaign(campaign)["ok"])
            self.assertTrue(self._export(fixture, replace=True)["changed"])
            self.assertEqual("verified", verify_evidence(fixture.output)["status"])
            index = json.loads((fixture.output / "index.json").read_text())
            entry = next(
                item
                for item in index["campaigns"]
                if item["evidence_kind"] == "sm121_chunked_prefill_performance"
            )
            manifest = json.loads((fixture.output / entry["file"]).read_text())
            self.assertEqual("partial", manifest["status"])
            self.assertEqual(0, manifest["completed_arms"])

    def test_audit_rejects_cross_arm_attestation_ordinal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            self._export(fixture)
            campaign = self._freeze(fixture)
            self._write_completed_campaign(campaign)
            first_name = json.loads((campaign / "campaign.json").read_text())["run_directories"][0]
            events_path = campaign / "runs" / first_name / "events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            static = next(
                event
                for event in events
                if event["event"] == SM121_CHUNKED_PREFILL_PERFORMANCE_STATIC_EVENT
            )
            static["lifetime_ordinal"] = 7
            evidence_test_support.EvidenceFixture.write_jsonl(events_path, events)
            report = audit_sm121_chunked_prefill_performance_campaign(campaign)
            self.assertFalse(report["ok"])
            self.assertEqual("campaign_contract_invalid", report["errors"][0]["code"])


if __name__ == "__main__":
    unittest.main()
