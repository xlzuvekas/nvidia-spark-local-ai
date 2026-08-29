"""Scalar-only evidence contracts for the SM121 A/B/B/A timing campaign."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bench.autoresearch_v2 as autoresearch_v2
from bench.audit import audit_sm121_cache_performance_campaign
from bench.evidence import (
    EvidenceError,
    export_evidence,
    verify_evidence,
    verify_sm121_cache_performance_prerequisites,
)
from bench.journal import content_hash
from bench.manifest import load_models, load_suite
from bench.runner import create_sm121_cache_performance_campaign
from bench.sglang_sm121_cache_observability import SM121_CACHE_SOURCE_DIGESTS
from bench.sglang_sm121_cache_performance import (
    SM121_CACHE_PERFORMANCE_ARM_ORDER,
    SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID,
    SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID,
    SM121_CACHE_PERFORMANCE_MAX_MAMBA_CACHE_SIZE,
    SM121_CACHE_PERFORMANCE_QUALITY_ITEM_COUNT,
    SM121_CACHE_PERFORMANCE_RUNTIME_EVENT,
    SM121_CACHE_PERFORMANCE_RUNTIME_EXPECTED,
    SM121_CACHE_PERFORMANCE_STATIC_EVENT,
    SM121_CACHE_PERFORMANCE_TIMED_TURNS,
    SM121_CACHE_PERFORMANCE_TURN_EVENT,
    score_sm121_cache_performance_campaign,
)
from bench.sglang_sm121_cache_semantic import SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS
from bench.sglang_sm121_storage import (
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_SOURCE_TREE,
)
from tests.test_sglang_sm121_cache_performance import _turn
import tests.test_evidence as evidence_test_support


PRIVATE_PROMPT = "CACHE_PERFORMANCE_PRIVATE_PROMPT_SENTINEL"
PRIVATE_COMPLETION = "CACHE_PERFORMANCE_PRIVATE_COMPLETION_SENTINEL"
PRIVATE_REQUEST_ID = "CACHE_PERFORMANCE_PRIVATE_REQUEST_ID_SENTINEL"


class SM121CachePerformanceEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = Path(__file__).resolve().parents[1]
        models = load_models(self.repository / "manifests" / "models.toml")
        self.cache_on_model = models[SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID]
        self.cache_off_model = models[SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID]
        self.suite_path = (
            self.repository
            / "manifests"
            / "suites"
            / "qwen38_flash_next_sm121_triton_storage_cache_policy_performance_v1.toml"
        )
        self.suite = load_suite(self.suite_path)

    @staticmethod
    def _export(
        fixture: evidence_test_support.EvidenceFixture, *, replace: bool = False
    ) -> dict[str, object]:
        with patch(
            "bench.evidence._export_campaign",
            side_effect=evidence_test_support.EvidenceExportTests.fake_campaign_export,
        ):
            return export_evidence(
                results_root=fixture.results,
                output_root=fixture.output,
                replace=replace,
            )

    def _freeze(
        self, fixture: evidence_test_support.EvidenceFixture
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
            return create_sm121_cache_performance_campaign(
                cache_on_model=self.cache_on_model,
                cache_off_model=self.cache_off_model,
                suite=self.suite,
                results_root=fixture.results / "cache-policy-campaigns",
                models_path=self.repository / "manifests" / "models.toml",
                suite_path=self.suite_path,
                evidence_root=fixture.output,
            )

    @staticmethod
    def _static_event(arm: str, lifetime_ordinal: int) -> dict[str, object]:
        return {
            "event": SM121_CACHE_PERFORMANCE_STATIC_EVENT,
            "arm": arm,
            "lifetime_ordinal": lifetime_ordinal,
            "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
            **SM121_CACHE_SOURCE_DIGESTS,
            **SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
        }

    @staticmethod
    def _runtime_event(arm: str, lifetime_ordinal: int) -> dict[str, object]:
        return {
            "event": SM121_CACHE_PERFORMANCE_RUNTIME_EVENT,
            "arm": arm,
            "lifetime_ordinal": lifetime_ordinal,
            "mamba_radix_cache_strategy": "extra_buffer_lazy",
            "max_mamba_cache_size": SM121_CACHE_PERFORMANCE_MAX_MAMBA_CACHE_SIZE,
            **SM121_CACHE_PERFORMANCE_RUNTIME_EXPECTED[arm],
        }

    @staticmethod
    def _lifetime(
        *, ordinal: int, arm: str, timed_case_id: str
    ) -> dict[str, object]:
        turns: list[dict[str, object]] = []
        for index, turn in enumerate(SM121_CACHE_PERFORMANCE_TIMED_TURNS):
            observation = _turn(
                ordinal=ordinal * 2,
                arm=arm,
                turn=turn,
                wall_s=20.0 + index,
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
        lifetimes: list[dict[str, object]] = []
        start = datetime(2026, 8, 29, 2, 0, tzinfo=timezone.utc)
        for ordinal, (name, arm) in enumerate(
            zip(campaign["run_directories"], SM121_CACHE_PERFORMANCE_ARM_ORDER),
            start=1,
        ):
            run_dir = campaign_dir / "runs" / name
            plan = json.loads((run_dir / "plan.json").read_text())
            cases = {case["id"]: case["case_id"] for case in plan["suite"]["cases"]}
            quality_case_id = cases["synthetic-exact-answer-v2"]
            timed_case_id = cases[
                "sm121-cache-policy-shared-prefix-performance-v1"
            ]
            quality_ordinal = ordinal * 2 - 1
            timed_ordinal = ordinal * 2
            events: list[dict[str, object]] = []

            def append(event: dict[str, object]) -> None:
                events.append(
                    {
                        "timestamp": (
                            start + timedelta(seconds=len(events))
                        ).isoformat(),
                        **event,
                    }
                )

            append(
                {
                    "event": "run_start",
                    "execution_mode": "sm121_storage_cache_policy_performance_abba_fresh_lifetimes",
                    "arm": arm,
                    "campaign_ordinal": ordinal,
                    "plan_fingerprint": plan["fingerprint"],
                    "cache_performance_pair_binding_sha256": plan[
                        "cache_performance_pair"
                    ]["pair_binding_sha256"],
                    "private_prompt": PRIVATE_PROMPT,
                    "private_completion": PRIVATE_COMPLETION,
                    "private_request_id": PRIVATE_REQUEST_ID,
                }
            )
            append({"event": "measurement_started"})
            append(self._static_event(arm, quality_ordinal))
            append(self._runtime_event(arm, quality_ordinal))
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
                    "event": "sm121_cache_performance_quality_case_start",
                    "arm": arm,
                    "lifetime_ordinal": quality_ordinal,
                    "case_id": quality_case_id,
                }
            )
            append(
                {
                    "event": "sm121_cache_performance_quality_case_complete",
                    "arm": arm,
                    "lifetime_ordinal": quality_ordinal,
                    "case_id": quality_case_id,
                    "quality_admitted": True,
                    "item_count": SM121_CACHE_PERFORMANCE_QUALITY_ITEM_COUNT,
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
                    "event": "sm121_cache_performance_lifetime_complete",
                    "arm": arm,
                    "lifetime_ordinal": quality_ordinal,
                    "phase": "quality",
                    "lifetime_wall_s": 1.0,
                    "within_timeout": True,
                    "admitted": True,
                }
            )
            append(self._static_event(arm, timed_ordinal))
            append(self._runtime_event(arm, timed_ordinal))
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
                    "event": "sm121_cache_performance_timed_case_start",
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
                    "event": "sm121_cache_performance_timed_case_complete",
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
                    "event": "sm121_cache_performance_lifetime_complete",
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
        score = score_sm121_cache_performance_campaign(lifetimes)
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

    def test_export_is_deterministic_scalar_only_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            self.assertTrue(self._export(fixture)["changed"])
            prerequisite = json.loads((fixture.output / "index.json").read_text())["runs"][0][
                "bundle_sha256"
            ]
            with (
                patch(
                    "bench.runner.SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S",
                    (prerequisite,),
                ),
                patch(
                    "bench.evidence.SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S",
                    (prerequisite,),
                ),
            ):
                campaign = self._freeze(fixture)
                self._write_completed_campaign(campaign)
                self.assertTrue(
                    audit_sm121_cache_performance_campaign(
                        campaign, evidence_root=fixture.output
                    )["ok"]
                )
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
                published = json.loads((fixture.output / "index.json").read_text())
                entry = next(
                    item
                    for item in published["campaigns"]
                    if item["evidence_kind"] == "sm121_cache_policy_performance"
                )
                manifest = json.loads(
                    (fixture.output / entry["file"]).read_text(encoding="utf-8")
                )
                self.assertEqual(4, manifest["completed_arms"])
                self.assertEqual(4, len(manifest["quality_attestations"]))
                self.assertTrue(
                    all(
                        item["item_count"] == SM121_CACHE_PERFORMANCE_QUALITY_ITEM_COUNT
                        for item in manifest["quality_attestations"]
                    )
                )
                serialized = b"\n".join(original.values()).decode(
                    "utf-8", errors="ignore"
                )
                for private in (PRIVATE_PROMPT, PRIVATE_COMPLETION, PRIVATE_REQUEST_ID):
                    self.assertNotIn(private, serialized)
                with patch(
                    "bench.evidence.SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S",
                    ("0" * 64,),
                ):
                    with self.assertRaisesRegex(EvidenceError, "prerequisite evidence"):
                        verify_sm121_cache_performance_prerequisites(
                            fixture.output, already_verified=True
                        )

    def test_audit_rejects_cross_arm_attestation_ordinal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            self._export(fixture)
            prerequisite = json.loads((fixture.output / "index.json").read_text())["runs"][0][
                "bundle_sha256"
            ]
            with (
                patch(
                    "bench.runner.SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S",
                    (prerequisite,),
                ),
                patch(
                    "bench.evidence.SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S",
                    (prerequisite,),
                ),
            ):
                campaign = self._freeze(fixture)
                self._write_completed_campaign(campaign)
                first_name = json.loads((campaign / "campaign.json").read_text())["run_directories"][0]
                events_path = campaign / "runs" / first_name / "events.jsonl"
                events = [json.loads(line) for line in events_path.read_text().splitlines()]
                first_static = next(
                    event
                    for event in events
                    if event["event"] == SM121_CACHE_PERFORMANCE_STATIC_EVENT
                )
                first_static["lifetime_ordinal"] = 7
                evidence_test_support.EvidenceFixture.write_jsonl(events_path, events)
                report = audit_sm121_cache_performance_campaign(
                    campaign, evidence_root=fixture.output
                )
                self.assertFalse(report["ok"])
                self.assertIn("campaign_contract_invalid", report["errors"][0]["code"])

    def test_audit_rejects_partial_turn_absent_from_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            self._export(fixture)
            prerequisite = json.loads((fixture.output / "index.json").read_text())["runs"][0][
                "bundle_sha256"
            ]
            with (
                patch(
                    "bench.runner.SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S",
                    (prerequisite,),
                ),
                patch(
                    "bench.evidence.SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S",
                    (prerequisite,),
                ),
            ):
                campaign = self._freeze(fixture)
                self._write_completed_campaign(campaign)
                summary_path = campaign / "summary.json"
                summary = json.loads(summary_path.read_text())
                first = summary["lifetimes"][0]
                first["timed_admitted"] = False
                first["turns"] = first["turns"][:1]
                for lifetime in summary["lifetimes"][1:]:
                    lifetime["quality_admitted"] = False
                    lifetime["timed_admitted"] = False
                    lifetime["within_timeout"] = False
                    lifetime["turns"] = []
                score = score_sm121_cache_performance_campaign(summary["lifetimes"])
                summary["status"] = score.status
                summary["decision"] = score.decision
                summary["completed_arms"] = 0
                summary["score"] = score.to_mapping()
                summary["integrity_hash"] = content_hash(
                    {key: value for key, value in summary.items() if key != "integrity_hash"},
                    64,
                )
                evidence_test_support.EvidenceFixture.write_json(summary_path, summary)
                report = audit_sm121_cache_performance_campaign(
                    campaign, evidence_root=fixture.output
                )
                self.assertFalse(report["ok"])
                self.assertIn("campaign_contract_invalid", report["errors"][0]["code"])

    def test_export_accepts_honest_terminal_timed_prefix(self) -> None:
        """A request failure may retain only the scalar turns it journaled."""

        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            self._export(fixture)
            prerequisite = json.loads((fixture.output / "index.json").read_text())["runs"][0][
                "bundle_sha256"
            ]
            with (
                patch(
                    "bench.runner.SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S",
                    (prerequisite,),
                ),
                patch(
                    "bench.evidence.SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S",
                    (prerequisite,),
                ),
            ):
                campaign = self._freeze(fixture)
                self._write_completed_campaign(campaign)
                names = json.loads((campaign / "campaign.json").read_text())["run_directories"]
                first_events_path = campaign / "runs" / names[0] / "events.jsonl"
                events = [
                    json.loads(line)
                    for line in first_events_path.read_text(encoding="utf-8").splitlines()
                ]
                third_turn = next(
                    index
                    for index, event in enumerate(events)
                    if event.get("event") == SM121_CACHE_PERFORMANCE_TURN_EVENT
                    and event.get("turn") == "T2"
                )
                partial_events = events[:third_turn]
                partial_events.extend(
                    [
                        {
                            "timestamp": "2026-08-29T02:30:00+00:00",
                            "event": "server_stopped",
                            "backend": "sglang",
                            "lifetime_ordinal": 2,
                        },
                        {
                            "timestamp": "2026-08-29T02:30:01+00:00",
                            "event": "sm121_cache_performance_lifetime_complete",
                            "arm": "A",
                            "lifetime_ordinal": 2,
                            "phase": "timed",
                            "lifetime_wall_s": 2.0,
                            "within_timeout": True,
                            "admitted": False,
                        },
                        {
                            "timestamp": "2026-08-29T02:30:02+00:00",
                            "event": "run_aborted",
                            "stage": "timed_lifetime",
                            "error_type": "SM121CachePerformanceRequestError",
                            "error": "request admission failed",
                        },
                    ]
                )
                evidence_test_support.EvidenceFixture.write_jsonl(
                    first_events_path, partial_events
                )
                for name in names[1:]:
                    evidence_test_support.EvidenceFixture.write_jsonl(
                        campaign / "runs" / name / "events.jsonl", []
                    )
                summary_path = campaign / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                first = summary["lifetimes"][0]
                first["timed_admitted"] = False
                first["within_timeout"] = False
                first["turns"] = first["turns"][:2]
                for lifetime in summary["lifetimes"][1:]:
                    lifetime["quality_admitted"] = False
                    lifetime["timed_admitted"] = False
                    lifetime["within_timeout"] = False
                    lifetime["turns"] = []
                score = score_sm121_cache_performance_campaign(summary["lifetimes"])
                summary["status"] = score.status
                summary["decision"] = score.decision
                summary["completed_arms"] = 0
                summary["score"] = score.to_mapping()
                summary["integrity_hash"] = content_hash(
                    {
                        key: value
                        for key, value in summary.items()
                        if key != "integrity_hash"
                    },
                    64,
                )
                evidence_test_support.EvidenceFixture.write_json(summary_path, summary)

                self.assertTrue(
                    audit_sm121_cache_performance_campaign(
                        campaign, evidence_root=fixture.output
                    )["ok"]
                )
                self.assertTrue(self._export(fixture, replace=True)["changed"])
                self.assertEqual("verified", verify_evidence(fixture.output)["status"])
                entry = next(
                    item
                    for item in json.loads((fixture.output / "index.json").read_text())["campaigns"]
                    if item["evidence_kind"] == "sm121_cache_policy_performance"
                )
                manifest = json.loads(
                    (fixture.output / entry["file"]).read_text(encoding="utf-8")
                )
                self.assertEqual("partial", manifest["status"])
                self.assertEqual(2, len(manifest["lifetimes"][0]["turns"]))
                self.assertEqual(1, len(manifest["quality_attestations"]))

    def test_export_validates_but_never_publishes_autoresearch_v2_wrapper(self) -> None:
        """A valid v2 controller remains scalar-free provenance around its child."""

        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            self._export(fixture)
            prerequisite = json.loads((fixture.output / "index.json").read_text())["runs"][0][
                "bundle_sha256"
            ]
            now = datetime(2026, 8, 29, 2, 0, tzinfo=timezone.utc)
            cutoff = "2026-08-29T08:00:00+00:00"
            with (
                patch(
                    "bench.runner.SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S",
                    (prerequisite,),
                ),
                patch(
                    "bench.evidence.SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S",
                    (prerequisite,),
                ),
                patch(
                    "bench.autoresearch_v2.SM121_CACHE_PERFORMANCE_PREREQUISITE_BUNDLE_SHA256S",
                    (prerequisite,),
                ),
                patch("bench.runner.datetime") as runner_datetime,
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
                runner_datetime.now.return_value = now
                round_dir = autoresearch_v2.freeze_autoresearch_v2(
                    self.repository
                    / "manifests"
                    / "campaigns"
                    / "qwen38_flash_next_sm121_autoresearch_v2_cache_policy.toml",
                    results_root=fixture.results,
                    evidence_root=fixture.output,
                    cutoff=cutoff,
                    now=now,
                )
                runner_datetime.now.return_value = now + timedelta(seconds=1)
                frozen_round = autoresearch_v2.freeze_autoresearch_v2(
                    self.repository
                    / "manifests"
                    / "campaigns"
                    / "qwen38_flash_next_sm121_autoresearch_v2_cache_policy.toml",
                    results_root=fixture.results,
                    evidence_root=fixture.output,
                    cutoff=cutoff,
                    now=now + timedelta(seconds=1),
                )
                self.assertEqual(
                    {"round.json"}, {path.name for path in frozen_round.iterdir()}
                )
                round_payload = autoresearch_v2._load_round(round_dir)
                child = (
                    fixture.results
                    / "cache-policy-campaigns"
                    / str(round_payload["child_campaign_directory"])
                )
                self._write_completed_campaign(child)
                child_summary = json.loads((child / "summary.json").read_text())
                with (
                    patch(
                        "bench.autoresearch_v2.execute_sm121_cache_performance_campaign",
                        return_value=child_summary,
                    ),
                    patch(
                        "bench.autoresearch_v2.audit_sm121_cache_performance_campaign",
                        return_value={"ok": True},
                    ),
                ):
                    summary = autoresearch_v2.run_autoresearch_v2(
                        round_dir,
                        workspace=fixture.results.parent,
                        evidence_root=fixture.output,
                        now=now,
                    )
                self.assertEqual("inconclusive", summary["decision"])
                failed_plan_root = (
                    fixture.results
                    / autoresearch_v2.AUTORESEARCH_V2_RESULT_ROOT
                    / (
                        "20260829T020002Z-"
                        f"{autoresearch_v2.AUTORESEARCH_V2_CAMPAIGN_ID}"
                    )
                )
                failed_plan_root.mkdir()
                self.assertTrue(self._export(fixture, replace=True)["changed"])
            published = json.loads((fixture.output / "index.json").read_text())
            self.assertEqual(
                1,
                sum(
                    entry["evidence_kind"] == "sm121_cache_policy_performance"
                    for entry in published["campaigns"]
                ),
            )
            serialized = b"\n".join(
                path.read_bytes() for path in fixture.output.rglob("*") if path.is_file()
            ).decode("utf-8", errors="ignore")
            self.assertNotIn("autoresearch-v2", serialized)

    def test_export_rejects_unknown_autoresearch_v2_wrapper_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            self._export(fixture)
            round_dir = (
                fixture.results
                / autoresearch_v2.AUTORESEARCH_V2_RESULT_ROOT
                / (
                    "20260829T020000Z-"
                    f"{autoresearch_v2.AUTORESEARCH_V2_CAMPAIGN_ID}"
                )
            )
            round_dir.mkdir(parents=True)
            (round_dir / "unexpected.txt").write_text("synthetic\n", encoding="utf-8")
            with self.assertRaisesRegex(EvidenceError, "autoresearch-v2 round topology"):
                self._export(fixture, replace=True)


if __name__ == "__main__":
    unittest.main()
