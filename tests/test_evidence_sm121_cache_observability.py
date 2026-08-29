"""Scalar evidence contracts for the SM121 cache-off B0 observability lane.

The fixture deliberately contains private response text, reasoning, and request
identifiers in the ignored source journal.  The asserted evidence bundle must
retain only the fixed scalar B0 observation contract.
"""

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
from bench.runner import create_sm121_cache_observability_plan
from bench.sglang_sm121_cache_observability import (
    SM121_CACHE_OBSERVABILITY_CACHED_SERIES,
    SM121_CACHE_OBSERVABILITY_EXECUTION_MODE,
    SM121_CACHE_OBSERVABILITY_METRIC_FIELDS,
    SM121_CACHE_OBSERVABILITY_SUITE_ID,
    SM121_CACHE_RUNTIME_ATTESTATION_EVENT,
    SM121_CACHE_RUNTIME_EXPECTED,
    SM121_CACHE_SOURCE_DIGESTS,
    SM121_CACHE_STATIC_ASSERTIONS,
    SM121_CACHE_STATIC_ATTESTATION_EVENT,
    SM121_CACHE_ZERO_HIT_CASE_ID,
    SM121_CACHE_ZERO_HIT_EVENT,
    SM121_CACHE_ZERO_HIT_REQUEST_CONTRACT_SHA256,
)
from bench.sglang_sm121_storage import (
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_PROFILE_ID,
    SM121_STORAGE_SOURCE_TREE,
)
import tests.test_evidence as evidence_test_support


PRIVATE_COMPLETION = "B0_PRIVATE_COMPLETION_SENTINEL"
PRIVATE_REASON = "B0_PRIVATE_VALIDATION_REASON_SENTINEL"
PRIVATE_REQUEST_ID = "B0_PRIVATE_REQUEST_ID_SENTINEL"


class SM121CacheObservabilityEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = Path(__file__).resolve().parents[1]
        self.model = load_models(self.repository / "manifests" / "models.toml")[
            SM121_STORAGE_PROFILE_ID
        ]
        self.suite_path = (
            self.repository
            / "manifests"
            / "suites"
            / "qwen38_flash_next_sm121_triton_storage_cache_observability_canary.toml"
        )
        self.suite = load_suite(self.suite_path)

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
            return create_sm121_cache_observability_plan(
                model=self.model,
                suite=self.suite,
                results_root=fixture.results,
                models_path=self.repository / "manifests" / "models.toml",
                suite_path=self.suite_path,
            )

    @staticmethod
    def _zero_hit_event(
        *, case_id: str, attempt_id: str, admitted: bool
    ) -> dict[str, object]:
        """Build a fully scalar B0 observation with no cache hit.

        The admitted fixture records a positive input delta and zero cache-hit
        deltas.  The partial fixture records unavailable metrics, making the
        zero observation intentionally nonadmissible without emitting a raw
        response or metrics payload.
        """

        event: dict[str, object] = {
            "event": SM121_CACHE_ZERO_HIT_EVENT,
            "case_id": case_id,
            "protocol_case_id": SM121_CACHE_ZERO_HIT_CASE_ID,
            "attempt_id": attempt_id,
            "request_contract_sha256": SM121_CACHE_ZERO_HIT_REQUEST_CONTRACT_SHA256,
            "cache_details_requested": True,
            "streaming": False,
            "thinking_disabled": True,
            "response_detail_state": "omitted",
            "usage_detail_state": "omitted",
            "response_device_cached_tokens": None,
            "response_host_cached_tokens": None,
            "response_storage_cached_tokens": None,
            "usage_cached_tokens": None,
            "metrics_available": admitted,
            "metrics_before_polls": 2 if admitted else 1,
            "metrics_after_polls": 2 if admitted else 1,
            "metrics_before_settle_s": 1.0 if admitted else 0.0,
            "metrics_after_settle_s": 1.0 if admitted else 0.0,
            "metrics_before_settled": admitted,
            "metrics_after_settled": admitted,
            "zero_hit_basis": (
                "omitted_or_null_with_native_counters"
                if admitted
                else "not_admitted"
            ),
            "zero_hit_admitted": admitted,
        }
        for prefix in ("before", "after", "delta"):
            for field in SM121_CACHE_OBSERVABILITY_METRIC_FIELDS:
                event[f"{prefix}_{field}"] = 0
        if admitted:
            event["before_prefill_input_tokens"] = 100
            event["after_prefill_input_tokens"] = 117
            event["delta_prefill_input_tokens"] = 17
            for prefix in ("before", "after"):
                event[f"{prefix}_kv_available_tokens"] = 65_536
                event[f"{prefix}_mamba_available_tokens"] = 65_536
        for prefix in ("before", "after"):
            for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
                event[f"{prefix}_cached_{source}_series_present"] = False
        return event

    @staticmethod
    def _request_result(*, prompt_tokens: int, completion_tokens: int) -> dict[str, object]:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": 0,
            "ttft_s": 0.1,
            "elapsed_s": 1.0,
            "decode_s": 0.9,
            "decode_tps": float(completion_tokens - 1) / 0.9,
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

    def _write_b0(
        self,
        fixture: evidence_test_support.EvidenceFixture,
        *,
        admitted: bool,
    ) -> Path:
        run_dir = self._freeze(fixture)
        plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
        cases = plan["suite"]["cases"]
        self.assertEqual(SM121_CACHE_OBSERVABILITY_SUITE_ID, plan["suite"]["id"])
        self.assertEqual(2, len(cases))
        quality_case_id = cases[0]["case_id"]
        observation_case_id = cases[1]["case_id"]
        self.assertTrue(observation_case_id.startswith(SM121_CACHE_ZERO_HIT_CASE_ID + "--"))

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
                "execution_mode": SM121_CACHE_OBSERVABILITY_EXECUTION_MODE,
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
        append(
            {
                "event": SM121_CACHE_STATIC_ATTESTATION_EVENT,
                "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
                **SM121_CACHE_SOURCE_DIGESTS,
                **SM121_CACHE_STATIC_ASSERTIONS,
            }
        )
        append({"event": SM121_CACHE_RUNTIME_ATTESTATION_EVENT, **SM121_CACHE_RUNTIME_EXPECTED})
        append(
            {
                "event": "server_ready",
                "backend": "sglang",
                "startup_s": 1.0,
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
        items = ("arithmetic-01", "logic-01", "instruction-01", "code-01")
        for index, (category, item) in enumerate(zip(categories, items, strict=True)):
            result = self._request_result(prompt_tokens=64 + index, completion_tokens=2)
            result.update(
                {
                    "request_id": PRIVATE_REQUEST_ID,
                    "started_at_ns": 1,
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
                        "quality_item_id": item,
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

        observation_attempt = "2" * 32
        append(
            {
                "event": "case_start",
                "case_id": observation_case_id,
                "attempt_id": observation_attempt,
                "kind": "capability",
                "concurrency": 1,
            }
        )
        append(
            self._zero_hit_event(
                case_id=observation_case_id,
                attempt_id=observation_attempt,
                admitted=admitted,
            )
        )
        append(
            {
                "event": "request_complete",
                "case_id": observation_case_id,
                "attempt_id": observation_attempt,
                "kind": "capability",
                "repetition": 0,
                "burst_elapsed_s": 1.0,
                "result": self._request_result(prompt_tokens=17, completion_tokens=2),
                "validation": {"passed": admitted},
            }
        )
        append(
            {
                "event": "case_complete",
                "case_id": observation_case_id,
                "attempt_id": observation_attempt,
                "kind": "capability",
                "concurrency": 1,
                "elapsed_s": 1.0,
                "validation_passed": admitted,
            }
        )
        append(
            {
                "event": "server_stopped",
                "backend": "sglang",
                "fresh_server_lifetime": 1,
            }
        )
        append({"event": "measurement_complete", "elapsed_s": 8.0, "monotonic_ns": 2})
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

    def test_admitted_and_nonadmitted_b0_export_deterministically_as_scalars(self) -> None:
        for admitted, expected_status in ((True, "complete"), (False, "partial")):
            with self.subTest(status=expected_status), tempfile.TemporaryDirectory() as directory:
                fixture = evidence_test_support.EvidenceFixture(Path(directory))
                run_dir = self._write_b0(fixture, admitted=admitted)
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
                self.assertEqual(5, samples["sample_count"])
                runtime = manifest["runtime"]["sm121_cache_observability"]
                self.assertEqual(
                    admitted,
                    runtime["zero_hit_observation"]["zero_hit_admitted"],
                )
                self.assertEqual(
                    SM121_STORAGE_LOCAL_IMAGE_ID.removeprefix("sha256:"),
                    runtime["docker_image_sha256"],
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

    def test_export_rejects_tampered_b0_static_proof_and_event_order(self) -> None:
        mutations = ("proof", "order")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = evidence_test_support.EvidenceFixture(Path(directory))
                run_dir = self._write_b0(fixture, admitted=True)
                events = [
                    json.loads(line)
                    for line in (run_dir / "events.jsonl").read_text().splitlines()
                ]
                if mutation == "proof":
                    static = next(
                        event
                        for event in events
                        if event.get("event") == SM121_CACHE_STATIC_ATTESTATION_EVENT
                    )
                    static["cache_registry_sha256"] = "sha256:" + "0" * 64
                else:
                    runtime_index = next(
                        index
                        for index, event in enumerate(events)
                        if event.get("event") == SM121_CACHE_RUNTIME_ATTESTATION_EVENT
                    )
                    ready_index = next(
                        index
                        for index, event in enumerate(events)
                        if event.get("event") == "server_ready"
                    )
                    events[runtime_index], events[ready_index] = (
                        events[ready_index],
                        events[runtime_index],
                    )
                fixture.write_jsonl(run_dir / "events.jsonl", events)
                summarize_run(run_dir)
                with self.assertRaises(EvidenceError):
                    self._export(fixture)

    def test_verifier_rejects_checksum_refreshed_b0_runtime_downgrade(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = evidence_test_support.EvidenceFixture(Path(directory))
            run_dir = self._write_b0(fixture, admitted=True)
            self._export(fixture)
            manifest_path = fixture.output / "runs" / run_dir.name / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runtime"]["sm121_cache_observability"]["cache_impl"] = (
                "RadixCache"
            )
            fixture.write_json(manifest_path, manifest)
            self._refresh_checksums(fixture, run_dir.name)
            with self.assertRaises(EvidenceError):
                verify_evidence(fixture.output)


if __name__ == "__main__":
    unittest.main()
