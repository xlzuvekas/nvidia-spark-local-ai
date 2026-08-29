"""Scalar-only evidence contracts for SM121 chunk-size prefill campaigns."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from bench.audit import audit_sm121_chunked_prefill_performance_campaign
from bench.evidence import (
    EvidenceError,
    SANITIZATION_POLICY,
    SCHEMA_VERSION,
    export_evidence,
    verify_evidence,
    verify_staged_evidence,
)
from bench.journal import content_hash
from bench.manifest import load_models, load_suite
from bench.runner import create_plan
from bench.sglang_sm121_cache_observability import SM121_CACHE_SOURCE_DIGESTS
from bench.sglang_sm121_cache_semantic import SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS
from bench.sm121_chunked_prefill_evidence import (
    ChunkedPrefillEvidenceError,
    EVIDENCE_KIND,
    manifest_from_source,
    verify_manifest,
)
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
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CANDIDATE_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CONTROL_PROFILE_ID,
    score_sm121_chunked_prefill_performance_campaign,
    sm121_chunked_prefill_performance_study,
    sm121_chunked_prefill_performance_pair_binding_sha256,
)
from bench.sglang_sm121_chunked_prefill_admission import (
    SM121_CHUNKED_PREFILL_8K_ADMISSION_EXECUTION_MODE,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_ID,
    SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE_ID,
    sm121_chunked_prefill_8k_admission_receipt,
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
        self.v3_control = models[
            SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CONTROL_PROFILE_ID
        ]
        self.v3_candidate = models[
            SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CANDIDATE_PROFILE_ID
        ]
        self.v3_suite_path = (
            self.repository
            / "manifests"
            / "suites"
            / "qwen38_flash_next_sm121_triton_storage_chunked_prefill_performance_v3.toml"
        )
        self.v3_suite = load_suite(self.v3_suite_path)

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

    def _v3_receipt(
        self, fixture: evidence_test_support.EvidenceFixture
    ) -> dict[str, object]:
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
            plan_dir = create_plan(
                model=self.v3_candidate,
                suite=self.v3_suite,
                results_root=fixture.results / "receipt-fixture",
                models_path=self.repository / "manifests" / "models.toml",
                suite_path=self.v3_suite_path,
                allow_sm121_chunked_prefill_performance=True,
                run_label="receipt-fixture",
            )
        plan = json.loads((plan_dir / "plan.json").read_text())
        summary: dict[str, object] = {
            "schema_version": 1,
            "admission_id": SM121_CHUNKED_PREFILL_8K_ADMISSION_ID,
            "execution_mode": SM121_CHUNKED_PREFILL_8K_ADMISSION_EXECUTION_MODE,
            "status": "complete",
            "decision": "admitted",
            "terminal_stage": "complete",
            "failure_code": None,
            "profile_id": SM121_CHUNKED_PREFILL_PERFORMANCE_V3_CANDIDATE_PROFILE_ID,
            "suite_id": SM121_CHUNKED_PREFILL_8K_ADMISSION_SUITE_ID,
            "quality_admitted": True,
            "cold_t0_admitted": True,
            "quality_within_timeout": True,
            "cold_t0_within_timeout": True,
            "static_attestations": 2,
            "runtime_attestations": 2,
        }
        summary["integrity_hash"] = content_hash(summary, 64)
        return sm121_chunked_prefill_8k_admission_receipt(
            summary,
            admission_plan_integrity_hash=str(plan["integrity_hash"]),
            admission_model_contract_sha256=content_hash(
                {
                    "domain": "sm121-chunked-prefill-v3-candidate-model-v1",
                    "value": plan["model"],
                },
                64,
            ),
            admission_local_image_contract_sha256=content_hash(
                {
                    "domain": "sm121-chunked-prefill-v3-local-image-v1",
                    "value": plan["resolved"]["local_image"],
                },
                64,
            ),
            admission_audit_sha256="a" * 64,
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

    def _valid_v3_scalar_bundle(self) -> tuple[dict[str, object], dict[str, object]]:
        """Build a fully shaped V3 publication without an admission receipt."""

        study = SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY
        timed_case_id = study.timed_case_id + "--0123456789ab"
        lifetimes = [
            self._lifetime(ordinal=ordinal, arm=arm, timed_case_id=timed_case_id)
            for ordinal, arm in enumerate(
                SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER, start=1
            )
        ]
        static_attestations: list[dict[str, object]] = []
        runtime_attestations: list[dict[str, object]] = []
        for ordinal, arm in enumerate(
            SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER, start=1
        ):
            for lifetime_ordinal in (ordinal * 2 - 1, ordinal * 2):
                static_attestations.append(
                    self._static_event(
                        arm,
                        lifetime_ordinal,
                        control_chunk_size=study.control_chunk_size,
                        candidate_chunk_size=study.candidate_chunk_size,
                    )
                )
                runtime_attestations.append(
                    self._runtime_event(
                        arm,
                        lifetime_ordinal,
                        control_chunk_size=study.control_chunk_size,
                        candidate_chunk_size=study.candidate_chunk_size,
                    )
                )
        score = score_sm121_chunked_prefill_performance_campaign(
            lifetimes, study=study
        )
        instance = "a" * 64
        campaign_id = study.campaign_id + "-" + instance[:12]
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "evidence_kind": EVIDENCE_KIND,
            "campaign_id": campaign_id,
            "protocol": {
                "campaign_id": study.campaign_id,
                "suite_id": study.suite_id,
                "execution_mode": study.execution_mode,
                "arm_order": list(SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER),
                "chunked_prefill_sizes": [
                    study.control_chunk_size,
                    study.candidate_chunk_size,
                ],
                "cell_timeout_s": 1_200,
                "quality_item_count": SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT,
                "timed_turns": list(SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS),
                "measurement": "non_streaming_request_wall_s_only",
                "primary": "cache_cold_t0_request_wall_s",
                "ttft": None,
            },
            "binding": {
                "campaign_instance_sha256": "sha256:" + instance,
                "pair_binding_sha256": "sha256:" + "b" * 64,
            },
            "status": score.status,
            "decision": score.decision,
            "completed_arms": 4,
            "lifetimes": lifetimes,
            "score": score.to_mapping(),
            "static_attestations": static_attestations,
            "runtime_attestations": runtime_attestations,
            "quality_attestations": [
                {
                    "arm": arm,
                    "quality_lifetime_ordinal": ordinal * 2 - 1,
                    "case_id": "synthetic-exact-answer-v2--0123456789ab",
                    "quality_admitted": True,
                    "item_count": SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_ITEM_COUNT,
                }
                for ordinal, arm in enumerate(
                    SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER, start=1
                )
            ],
            "sanitization": {
                "free_form_text_included": False,
                "payloads_included": False,
                "policy": SANITIZATION_POLICY,
                "raw_identifiers_included": False,
            },
        }
        entry = {
            "bundle_sha256": "c" * 64,
            "campaign_id": campaign_id,
            "evidence_kind": EVIDENCE_KIND,
            "file": f"campaigns/{campaign_id}/manifest.json",
            "status": score.status,
        }
        return manifest, entry

    @staticmethod
    def _refresh_v3_bundle_checksums(root: Path, campaign_id: str) -> None:
        bundle = root / "campaigns" / campaign_id
        bundle_checksums = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(bundle.iterdir())
            if path.name != "checksums.json"
        }
        evidence_test_support.EvidenceFixture.write_json(
            bundle / "checksums.json",
            {"files": bundle_checksums, "schema_version": SCHEMA_VERSION},
        )
        index_path = root / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        entry = next(
            value
            for value in index["campaigns"]
            if value["campaign_id"] == campaign_id
        )
        entry["bundle_sha256"] = hashlib.sha256(
            (bundle / "checksums.json").read_bytes()
        ).hexdigest()
        evidence_test_support.EvidenceFixture.write_json(index_path, index)
        checksums_path = root / "checksums.json"
        root_checksums = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and path != checksums_path
        }
        evidence_test_support.EvidenceFixture.write_json(
            checksums_path,
            {"files": root_checksums, "schema_version": SCHEMA_VERSION},
        )

    def _forge_valid_v3_bundle(
        self, fixture: evidence_test_support.EvidenceFixture
    ) -> tuple[dict[str, object], dict[str, object]]:
        self._export(fixture)
        campaign = self._freeze(
            fixture,
            control=self.v2_control,
            candidate=self.v2_candidate,
            suite=self.v2_suite,
            suite_path=self.v2_suite_path,
        )
        self._write_completed_campaign(campaign)
        self.assertTrue(self._export(fixture, replace=True)["changed"])
        index_path = fixture.output / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        v2_entry = next(
            entry
            for entry in index["campaigns"]
            if entry["evidence_kind"] == EVIDENCE_KIND
            and entry["campaign_id"].startswith(
                SM121_CHUNKED_PREFILL_PERFORMANCE_V2_STUDY.campaign_id + "-"
            )
        )
        manifest, entry = self._valid_v3_scalar_bundle()
        (fixture.output / "campaigns" / v2_entry["campaign_id"]).rename(
            fixture.output / "campaigns" / entry["campaign_id"]
        )
        evidence_test_support.EvidenceFixture.write_json(
            fixture.output / entry["file"], manifest
        )
        v2_entry.update(entry)
        index["campaigns"].sort(key=lambda value: value["campaign_id"])
        evidence_test_support.EvidenceFixture.write_json(index_path, index)
        self._refresh_v3_bundle_checksums(fixture.output, entry["campaign_id"])
        return manifest, entry

    def test_v3_projection_and_manifest_are_blocked_without_admission_receipt(self) -> None:
        manifest, entry = self._valid_v3_scalar_bundle()
        source = {
            "study": SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY,
            "binding": manifest["binding"],
            "summary": {
                "status": manifest["status"],
                "decision": manifest["decision"],
                "completed_arms": manifest["completed_arms"],
                "lifetimes": manifest["lifetimes"],
                "score": manifest["score"],
            },
            "static_events": manifest["static_attestations"],
            "runtime_events": manifest["runtime_attestations"],
            "quality_attestations": manifest["quality_attestations"],
        }
        with patch(
            "bench.sm121_chunked_prefill_evidence."
            "SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY",
            object(),
        ):
            self.assertEqual(
                manifest,
                manifest_from_source(
                    source,
                    schema_version=SCHEMA_VERSION,
                    sanitization_policy=SANITIZATION_POLICY,
                ),
            )
            verify_manifest(
                manifest,
                entry,
                schema_version=SCHEMA_VERSION,
                sanitization_policy=SANITIZATION_POLICY,
            )
        with self.assertRaisesRegex(
            ChunkedPrefillEvidenceError, "requires a verified 8K admission receipt"
        ):
            manifest_from_source(
                source,
                schema_version=SCHEMA_VERSION,
                sanitization_policy=SANITIZATION_POLICY,
            )
        with self.assertRaisesRegex(
            ChunkedPrefillEvidenceError, "requires a verified 8K admission receipt"
        ):
            verify_manifest(
                manifest,
                entry,
                schema_version=SCHEMA_VERSION,
                sanitization_policy=SANITIZATION_POLICY,
            )

    def test_v3_local_audit_requires_the_fresh_matching_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            receipt = self._v3_receipt(fixture)
            with (
                patch(
                    "bench.sm121_chunked_prefill_runner."
                    "load_verified_sm121_chunked_prefill_8k_admission_receipt",
                    return_value=receipt,
                ),
                patch(
                    "bench.sm121_chunked_prefill_runner._V3_LOGS_ROOT",
                    fixture.results,
                ),
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
                campaign = create_sm121_chunked_prefill_performance_campaign(
                    control_model=self.v3_control,
                    candidate_model=self.v3_candidate,
                    suite=self.v3_suite,
                    results_root=fixture.results / "chunked-prefill-campaigns",
                    models_path=self.repository / "manifests" / "models.toml",
                    suite_path=self.v3_suite_path,
                    admission_run_dir=Path("private-admission"),
                )
            self._write_completed_campaign(campaign)
            self.assertFalse(audit_sm121_chunked_prefill_performance_campaign(campaign)["ok"])
            with patch(
                "bench.audit.load_verified_sm121_chunked_prefill_8k_admission_receipt",
                return_value=receipt,
            ):
                self.assertFalse(
                    audit_sm121_chunked_prefill_performance_campaign(
                        campaign, admission_run_dir=Path("private-admission")
                    )["ok"]
                )
            campaign_path = campaign / "campaign.json"
            before_audit = (
                campaign_path.read_bytes(),
                campaign_path.stat().st_mode,
                campaign_path.stat().st_ctime_ns,
            )
            with patch(
                "bench.audit.load_verified_sm121_chunked_prefill_8k_admission_receipt",
                return_value=receipt,
            ), patch(
                "bench.sm121_chunked_prefill_runner._V3_LOGS_ROOT",
                fixture.results,
            ):
                report = audit_sm121_chunked_prefill_performance_campaign(
                    campaign, admission_run_dir=Path("private-admission")
                )
            self.assertTrue(report["ok"])
            self.assertEqual(
                before_audit,
                (
                    campaign_path.read_bytes(),
                    campaign_path.stat().st_mode,
                    campaign_path.stat().st_ctime_ns,
                ),
            )
            self.assertEqual(
                SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY.campaign_id,
                report["campaign_id"],
            )
            campaign_data = json.loads((campaign / "campaign.json").read_text())
            binding = campaign_data["pair_binding"]
            binding["admission_receipt_sha256"] = "b" * 64
            binding["pair_binding_sha256"] = (
                sm121_chunked_prefill_performance_pair_binding_sha256(binding)
            )
            campaign_data["integrity_hash"] = content_hash(
                {
                    key: value
                    for key, value in campaign_data.items()
                    if key != "integrity_hash"
                },
                64,
            )
            (campaign / "campaign.json").write_text(
                json.dumps(campaign_data, sort_keys=True) + "\n"
            )
            for name in campaign_data["run_directories"]:
                plan_path = campaign / "runs" / name / "plan.json"
                plan = json.loads(plan_path.read_text())
                plan["chunked_prefill_performance_pair"] = binding
                plan["integrity_hash"] = content_hash(
                    {
                        key: value
                        for key, value in plan.items()
                        if key != "integrity_hash"
                    },
                    64,
                )
                plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n")
            summary_path = campaign / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["pair_binding_sha256"] = binding["pair_binding_sha256"]
            summary["integrity_hash"] = content_hash(
                {key: value for key, value in summary.items() if key != "integrity_hash"},
                64,
            )
            summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
            with patch(
                "bench.audit.load_verified_sm121_chunked_prefill_8k_admission_receipt",
                return_value=receipt,
            ), patch(
                "bench.sm121_chunked_prefill_runner._V3_LOGS_ROOT",
                fixture.results,
            ):
                self.assertFalse(
                    audit_sm121_chunked_prefill_performance_campaign(
                        campaign, admission_run_dir=Path("private-admission")
                    )["ok"]
                )

    def test_v3_bundle_cannot_pass_evidence_or_staged_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            matrix = fixture.results / "matrices" / "synthetic-matrix"
            matrix.mkdir()
            fixture.write_json(
                matrix / "matrix.json",
                {
                    "models": ["synthetic-model"],
                    "runs": [],
                    "suite": "synthetic-suite",
                },
            )
            _manifest, _entry = self._forge_valid_v3_bundle(fixture)
            with patch(
                "bench.sm121_chunked_prefill_evidence."
                "SM121_CHUNKED_PREFILL_PERFORMANCE_V3_STUDY",
                object(),
            ):
                self.assertEqual("verified", verify_evidence(fixture.output)["status"])
            with self.assertRaisesRegex(
                EvidenceError, "SM121 chunked-prefill evidence changed"
            ):
                verify_evidence(fixture.output)

            subprocess.run(
                ["git", "init", "--quiet"],
                cwd=directory,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                ["git", "add", "--", fixture.output.name],
                cwd=directory,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            with self.assertRaisesRegex(
                EvidenceError, "SM121 chunked-prefill evidence changed"
            ):
                verify_staged_evidence(
                    repo_root=Path(directory), evidence_root=Path(fixture.output.name)
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
