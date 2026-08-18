from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from bench.evidence import (
    EvidenceError,
    _NINFER_TOP_FIELDS,
    SCHEMA_VERSION,
    _assert_source_tree,
    _load_json,
    _project_case,
    _project_ninfer_report,
    _project_request_result,
    _project_requests,
    _project_suite,
    _project_summary,
    _validate_agentic_aggregates,
    _validate_output_value,
    _write_bundle,
    export_evidence,
    verify_evidence,
    verify_staged_evidence,
)


RAW_COMPLETION = "RAW_COMPLETION_SENTINEL"
RAW_REASONING = "RAW_REASONING_SENTINEL"
RAW_REQUEST_ID = "RAW_REQUEST_ID_SENTINEL"
RAW_HOST_PATH = "/home/private-user/benchmark-cache/model.gguf"
RAW_SECRET = "hf" + "_" + "0123456789abcdefghijklmnop"


def _agentic_suite() -> dict[str, object]:
    scenarios = (
        ("agentic-select-and-call", 1),
        ("agentic-no-tool", 2),
        ("agentic-two-hop", 0),
        ("agentic-tool-error-recovery", 3),
    )
    return {
        "id": "agentic-tools",
        "description": (
            "Deterministic multi-turn tool selection, abstention, dependency, "
            "and recovery checks with scalar-only results."
        ),
        "schema_version": 1,
        "cases": [
            {
                "case_id": f"{scenario}--{suffix:012x}",
                "concurrency": 1,
                "id": scenario,
                "kind": "agentic",
                "max_output_tokens": 4096,
                "max_turns": 6,
                "prompt_repetitions": 0,
                "repetitions": 3,
                "requires": ["chat", "tools"],
                "temperature": 0.0,
                "warmups": 0,
            }
            for scenario, suffix in scenarios
        ],
    }


class EvidenceFixture:
    def __init__(self, root: Path) -> None:
        self.results = root / "results"
        self.output = root / "evidence"
        self.results.mkdir()
        (self.results / ".sparkbench.lock").touch()
        (self.results / "matrices").mkdir()
        self.run_id = "20260817T000000Z-synthetic"
        self.run_dir = self.results / self.run_id
        self.run_dir.mkdir()
        self._write_required_campaigns()
        self._write_standalone_results(aggregate_tps=12.5)
        self._write_run()

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
            encoding="utf-8",
        )

    def _write_required_campaigns(self) -> None:
        for name in (
            "moe-bandwidth-20260817T1539Z",
            "ninfer-experimental-sm121a-20260817T181134Z",
            "ninfer-qwen38-nvfp4-sm121a-20260817T200147Z",
        ):
            (self.results / name).mkdir()
        admission = self.results / "ninfer-gb10-20260817"
        admission.mkdir()
        (admission / "stock-cmake-sm121.log").write_text(
            "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got '121a'\n",
            encoding="utf-8",
        )
        (admission / "stock-cmake-default.log").write_text(
            "NInfer requires CUDA 13.1 or newer; found CUDA compiler 13.0.88\n",
            encoding="utf-8",
        )

    def _write_standalone_results(self, *, aggregate_tps: float) -> None:
        probe_ids = (
            ("math-en-eval-style", "math", "en"),
            ("code-en", "code", "en"),
            ("code-de", "code", "de"),
            ("technical-explanation-fr", "technical_explanation", "fr"),
            ("reasoning-fr", "reasoning", "fr"),
            ("free-prose-en", "free_prose", "en"),
            ("free-prose-fr", "free_prose", "fr"),
            ("free-prose-de", "free_prose", "de"),
        )

        def summary() -> dict[str, float | int]:
            return {
                "aggregate_decode_tps": aggregate_tps,
                "aggregate_output_tps": aggregate_tps,
                "completion_tokens": 30,
                "maximum_decode_tps": aggregate_tps,
                "median_decode_tps": aggregate_tps,
                "median_e2e_s": 1.0,
                "median_ttft_s": 0.1,
                "minimum_decode_tps": aggregate_tps,
                "prompt_tokens": 30,
                "requests": 3,
            }

        probes = []
        for probe_id, category, language in probe_ids:
            samples = []
            for repetition in range(1, 4):
                samples.append(
                    {
                        "completion_tokens": 10,
                        "decode_s": 0.9,
                        "decode_tps": aggregate_tps,
                        "e2e_s": 1.0,
                        "emission_events": 10,
                        "measured_order": repetition,
                        "output_tps": aggregate_tps,
                        "prompt_tokens": 10,
                        "repetition": repetition,
                        "sample_id": f"{RAW_REQUEST_ID}-{repetition}",
                        "ttft_s": 0.1,
                    }
                )
            probes.append(
                {
                    "category": category,
                    "id": probe_id,
                    "language": language,
                    "samples": samples,
                    "summary": summary(),
                }
            )
        self.write_json(
            self.results / "content-battery-dspark-sglang-20260817.json",
            {
                "schema_version": 1,
                "battery": {
                    "id": "dgx-spark-qwen38-content",
                    "prompt_set_version": 1,
                    "protocol_version": 1,
                },
                "endpoint": "loopback",
                "model": "qwen3.8-27b",
                "probes": probes,
                "protocol": {
                    "aggregate_decode_tps": "sum_completion_tokens_minus_first_over_sum_post_ttft_seconds",
                    "aggregate_output_tps": "sum_completion_tokens_over_sum_e2e_seconds",
                    "fresh_prompt_tags": RAW_REQUEST_ID,
                    "max_output_tokens": 680,
                    "minimum_output_tokens": 50,
                    "repetitions_per_prompt": 3,
                    "temperature": 0.0,
                    "transport": "openai_chat_completions_sse",
                    "warmups": 1,
                },
                "summary": summary(),
                "warmup": {
                    "completion_tokens": 10,
                    "decode_s": 0.9,
                    "decode_tps": aggregate_tps,
                    "e2e_s": 1.0,
                    "emission_events": 10,
                    "id": RAW_REQUEST_ID,
                    "output_tps": aggregate_tps,
                    "prompt_tokens": 10,
                    "ttft_s": 0.1,
                },
            },
        )
        self.write_json(
            self.results / "upstream-bench-matrix-dspark-sglang-20260817.json",
            {
                "battery_version": 1,
                "endpoint": "loopback",
                "method": "synthetic",
                "model_id": "qwen3.8-27b",
                "results": [
                    {"probe": probe, "tok_s": aggregate_tps}
                    for probe in (
                        "math (EN, eval-style)",
                        "code (EN)",
                        "code (DE)",
                        "technical explain (FR)",
                        "reasoning (FR)",
                        "free prose (EN)",
                        "free prose (FR)",
                        "free prose (DE)",
                    )
                ],
            },
        )

    @staticmethod
    def _request_result(*, completion_tokens: int) -> dict[str, object]:
        return {
            "prompt_tokens": 10,
            "completion_tokens": completion_tokens,
            "elapsed_s": 1.25,
            "ttft_s": 0.2,
            "decode_tps": 20.0,
            "content": f"{RAW_COMPLETION} {RAW_SECRET}",
            "reasoning": f"{RAW_REASONING} {RAW_HOST_PATH}",
            "request_id": RAW_REQUEST_ID,
            "tool_calls": [{"arguments": RAW_SECRET}],
            "output_sha256": "a" * 64,
        }

    def _write_run(self) -> None:
        self.write_json(
            self.run_dir / "plan.json",
            {
                "schema_version": 1,
                "model": {
                    "id": "synthetic-model",
                    "backend": "synthetic",
                    "architecture": "synthetic",
                    "quantization": "fp8",
                    "source": "example/synthetic-model",
                    "revision": "a" * 40,
                    "tasks": ["chat"],
                    "lifecycle": "managed",
                },
                "suite": {
                    "id": "synthetic-suite",
                    "schema_version": 1,
                    "cases": [
                        {
                            "id": "chat-case",
                            "kind": "chat",
                            "repetitions": 3,
                            "warmups": 0,
                            "max_output_tokens": 32,
                            "temperature": 0.0,
                        }
                    ],
                },
            },
        )
        events = [
            {
                "timestamp": "2026-08-17T00:00:00Z",
                "event": "run_start",
            },
            {
                "timestamp": "2026-08-17T00:00:01Z",
                "event": "first_request_complete",
                "result": self._request_result(completion_tokens=1),
            },
            {
                "timestamp": "2026-08-17T00:00:02Z",
                "event": "case_start",
                "case_id": "chat-case",
                "attempt_id": "private-attempt-a",
            },
            {
                "timestamp": "2026-08-17T00:00:03Z",
                "event": "request_complete",
                "case_id": "chat-case",
                "kind": "chat",
                "attempt_id": "private-attempt-a",
                "request_tag": RAW_REQUEST_ID,
                "result": self._request_result(completion_tokens=10),
            },
            {
                "timestamp": "2026-08-17T00:00:04Z",
                "event": "request_complete",
                "case_id": "chat-case",
                "kind": "chat",
                "attempt_id": "private-attempt-a",
                "result": self._request_result(completion_tokens=11),
            },
            {
                "timestamp": "2026-08-17T00:00:05Z",
                "event": "case_start",
                "case_id": "chat-case",
                "attempt_id": "private-attempt-b",
            },
            {
                "timestamp": "2026-08-17T00:00:06Z",
                "event": "request_complete",
                "case_id": "chat-case",
                "kind": "chat",
                "attempt_id": "private-attempt-b",
                "result": self._request_result(completion_tokens=12),
            },
            {
                "timestamp": "2026-08-17T00:00:07Z",
                "event": "run_complete",
                "status": "completed",
                "diagnostic": RAW_HOST_PATH,
            },
        ]
        self.write_jsonl(self.run_dir / "events.jsonl", events)
        self.write_json(
            self.run_dir / "summary.json",
            {
                "schema_version": "2",
                "status": "complete",
                "run_completion_status": "completed",
                "completed_cases": 1,
                "failed_cases": [],
                "validation_failed_cases": [],
                "unimplemented_cases": [],
                "unsupported_cases": [],
                "context_limited_cases": [],
                "first_request_after_start": self._request_result(
                    completion_tokens=1
                ),
                "measurement_annotations": [
                    {
                        "reason": RAW_REASONING,
                        "request_id": RAW_REQUEST_ID,
                        "path": RAW_HOST_PATH,
                    }
                ],
                "cases": [
                    {
                        "case_id": "chat-case",
                        "attempt_id": "private-attempt-b",
                        "kind": "chat",
                        "requests": 1,
                        "prompt_tokens": 10,
                        "completion_tokens": 12,
                        "elapsed_s": 1.25,
                        "aggregate_output_tps": 20.0,
                        "measurement_valid": True,
                        "validation_passed": True,
                        "measurement_annotations": [
                            {"reason": RAW_COMPLETION, "secret": RAW_SECRET}
                        ],
                        "output_sha256": "b" * 64,
                    }
                ],
            },
        )
        self.write_jsonl(
            self.run_dir / "telemetry.jsonl",
            [
                {
                    "timestamp": "2026-08-17T00:00:02.000Z",
                    "gpu_timestamp": "raw-device-time-a",
                    "phase": "case:chat-case:measure",
                    "gpu_util_pct": 80.0,
                    "power_w": 100.0,
                    "memfree_kib": 10,
                    "gpu_error": "",
                },
                {
                    "timestamp": "2026-08-17T00:00:02.500Z",
                    "gpu_timestamp": "raw-device-time-b",
                    "phase": "case:chat-case:measure",
                    "gpu_util_pct": 90.0,
                    "power_w": 110.0,
                    "memfree_kib": 9,
                    "gpu_error": f"{RAW_SECRET} {RAW_HOST_PATH}",
                },
                {
                    "timestamp": "2026-08-17T00:00:07.000Z",
                    "gpu_timestamp": "raw-device-time-c",
                    "phase": "idle",
                    "gpu_util_pct": 0.0,
                    "power_w": 40.0,
                    "memfree_kib": 12,
                    "gpu_error": None,
                },
            ],
        )

    def change_aggregate(self, value: float) -> None:
        self._write_standalone_results(aggregate_tps=value)


def json_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key
            for child in value.values()
            for key in json_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in json_keys(child)}
    return set()


class EvidenceExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = EvidenceFixture(Path(self.temporary.name))

    def exported_bytes(self) -> dict[str, bytes]:
        return {
            str(path.relative_to(self.fixture.output)): path.read_bytes()
            for path in self.fixture.output.rglob("*")
            if path.is_file()
        }

    def git(self, *arguments: str) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=self.temporary.name,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def add_matrix_source(self) -> None:
        matrix = self.fixture.results / "matrices" / "synthetic-matrix"
        matrix.mkdir()
        self.fixture.write_json(
            matrix / "matrix.json",
            {
                "models": ["synthetic-model"],
                "runs": [],
                "suite": "synthetic-suite",
            },
        )

    @staticmethod
    def fake_campaign_export(
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

    def export(self, *, replace: bool = False) -> dict[str, object]:
        # Production campaigns have deliberately exact file-set contracts. Patch
        # that independent adapter so this fixture can remain a small run corpus.
        with patch(
            "bench.evidence._export_campaign",
            side_effect=self.fake_campaign_export,
        ):
            return export_evidence(
                results_root=self.fixture.results,
                output_root=self.fixture.output,
                replace=replace,
            )

    def test_export_is_deterministic_and_excludes_raw_values(self) -> None:
        first = self.export()
        self.assertTrue(first["changed"])
        original = self.exported_bytes()

        second = self.export()

        self.assertFalse(second["changed"])
        self.assertEqual(original, self.exported_bytes())
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

        serialized_json = b"\n".join(
            data for name, data in original.items() if name.endswith(".json")
        ).decode("utf-8")
        for sentinel in (
            RAW_COMPLETION,
            RAW_REASONING,
            RAW_REQUEST_ID,
            RAW_HOST_PATH,
            RAW_SECRET,
            "private-attempt-a",
            "private-attempt-b",
            "raw-device-time-a",
        ):
            with self.subTest(sentinel=sentinel):
                self.assertNotIn(sentinel, serialized_json)

        all_keys: set[str] = set()
        for name, data in original.items():
            if name.endswith(".json"):
                all_keys.update(json_keys(json.loads(data)))
        self.assertTrue({"completion_tokens", "decode_tps"} <= all_keys)
        self.assertIn('"gpu_error_present"', serialized_json)
        for forbidden in (
            "content",
            "reasoning",
            "request_id",
            "request_tag",
            "tool_calls",
            "timestamp",
            "gpu_timestamp",
        ):
            self.assertNotIn(forbidden, all_keys)
        self.assertFalse(any(key.endswith("_path") for key in all_keys))

    def test_source_ordinals_and_columnar_telemetry_are_preserved(self) -> None:
        self.export()
        run = self.fixture.output / "runs" / self.fixture.run_id
        samples = json.loads((run / "samples.json").read_text(encoding="utf-8"))[
            "samples"
        ]
        self.assertEqual([1, 2, 3, 4], [sample["sample_index"] for sample in samples])
        self.assertEqual(
            ["first_request", "measured_request", "measured_request", "measured_request"],
            [sample["sample_type"] for sample in samples],
        )
        self.assertEqual([1, 1, 2], [sample["case_attempt"] for sample in samples[1:]])
        self.assertEqual(
            [1, 2, 1], [sample["case_sample_index"] for sample in samples[1:]]
        )
        self.assertEqual(
            [False, False, True], [sample["selected_attempt"] for sample in samples[1:]]
        )

        metadata = json.loads((run / "telemetry.json").read_text(encoding="utf-8"))
        self.assertEqual(3, metadata["sample_count"])
        self.assertEqual(2, metadata["segment_count"])
        self.assertEqual(["telemetry-0001.json"], metadata["chunks"])
        chunk = json.loads(
            (run / metadata["chunks"][0]).read_text(encoding="utf-8")
        )
        self.assertEqual(["case:chat-case", "idle"], [
            segment["phase"] for segment in chunk["segments"]
        ])
        self.assertEqual([2, 1], [len(segment["rows"]) for segment in chunk["segments"]])
        elapsed_index = metadata["columns"].index("elapsed_s")
        error_index = metadata["columns"].index("gpu_error_present")
        memfree_index = metadata["columns"].index("memfree_bytes")
        first_rows = chunk["segments"][0]["rows"]
        self.assertEqual([0.0, 0.5], [row[elapsed_index] for row in first_rows])
        self.assertEqual([False, True], [row[error_index] for row in first_rows])
        self.assertEqual([10 * 1024, 9 * 1024], [row[memfree_index] for row in first_rows])
        self.assertEqual(0.0, chunk["segments"][1]["rows"][0][elapsed_index])

    def test_changed_export_requires_replace(self) -> None:
        self.export()
        self.fixture.change_aggregate(13.5)

        with self.assertRaisesRegex(EvidenceError, "rerun with --replace"):
            self.export()

        result = self.export(replace=True)
        self.assertTrue(result["changed"])
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

    def test_tampered_file_fails_checksum_verification(self) -> None:
        self.export()
        index_path = self.fixture.output / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["source_file_count"] += 1
        self.fixture.write_json(index_path, index)

        with self.assertRaisesRegex(EvidenceError, "checksums do not match"):
            verify_evidence(self.fixture.output)

    def test_output_target_cannot_be_results_ancestor(self) -> None:
        with self.assertRaisesRegex(EvidenceError, "unsafe evidence output target"):
            export_evidence(
                results_root=self.fixture.results,
                output_root=Path(self.temporary.name),
            )

    def test_output_target_cannot_be_final_component_symlink(self) -> None:
        real_output = Path(self.temporary.name) / "real-evidence"
        real_output.mkdir()
        self.fixture.output.symlink_to(real_output, target_is_directory=True)

        with self.assertRaisesRegex(EvidenceError, "must not be a symlink"):
            export_evidence(
                results_root=self.fixture.results,
                output_root=self.fixture.output,
            )

    def test_foreign_file_fails_topology_even_with_updated_checksums(self) -> None:
        self.export()
        foreign = self.fixture.output / "foreign.json"
        self.fixture.write_json(foreign, {"schema_version": SCHEMA_VERSION})
        checksums_path = self.fixture.output / "checksums.json"
        checksums = {
            str(path.relative_to(self.fixture.output)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(self.fixture.output.rglob("*"))
            if path.is_file() and path != checksums_path
        }
        self.fixture.write_json(
            checksums_path,
            {"files": checksums, "schema_version": SCHEMA_VERSION},
        )

        with self.assertRaisesRegex(EvidenceError, "top-level layout changed"):
            verify_evidence(self.fixture.output)

    def test_staged_evidence_verifies_with_an_unrelated_text_file(self) -> None:
        self.add_matrix_source()
        self.export()
        repository = Path(self.temporary.name)
        notes = repository / "benchmark-notes.txt"
        notes.write_text("scalar benchmark notes only\n", encoding="utf-8")
        self.git("init", "--quiet")
        self.git("add", "--", self.fixture.output.name, notes.name)

        result = verify_staged_evidence(
            repo_root=repository,
            evidence_root=Path(self.fixture.output.name),
        )

        evidence_files = sum(
            path.is_file() for path in self.fixture.output.rglob("*")
        )
        self.assertEqual("staged_verified", result["status"])
        self.assertEqual(evidence_files, result["files"])
        self.assertEqual(evidence_files + 1, result["staged_file_count"])
        self.assertRegex(result["tree_sha256"], r"^[0-9a-f]{64}$")

    def test_staged_blob_secret_is_detected_after_worktree_overwrite(self) -> None:
        self.add_matrix_source()
        self.export()
        repository = Path(self.temporary.name)
        staged_only = repository / "staged-only.txt"
        staged_value = RAW_SECRET
        staged_only.write_text(f"credential={staged_value}\n", encoding="utf-8")
        self.git("init", "--quiet")
        self.git("add", "--", self.fixture.output.name, staged_only.name)
        staged_only.write_text("safe worktree replacement\n", encoding="utf-8")

        with self.assertRaises(EvidenceError) as caught:
            verify_staged_evidence(
                repo_root=repository,
                evidence_root=Path(self.fixture.output.name),
            )

        message = str(caught.exception)
        self.assertIn("huggingface-token detector matched staged file", message)
        self.assertIn(staged_only.name, message)
        self.assertNotIn(staged_value, message)
        self.assertNotIn(staged_value, staged_only.read_text(encoding="utf-8"))


class EvidenceValidationTests(unittest.TestCase):
    def test_duplicate_keys_nonfinite_and_unknown_request_fields_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"metric":1,"metric":2}\n', encoding="utf-8")
            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"metric":NaN}\n', encoding="utf-8")

            with self.assertRaisesRegex(EvidenceError, "duplicate JSON key"):
                _load_json(duplicate, root)
            with self.assertRaisesRegex(EvidenceError, "non-finite JSON constant"):
                _load_json(nonfinite, root)

        with self.assertRaisesRegex(EvidenceError, "unknown request result fields"):
            _project_request_result({"completion_tokens": 1, "new_raw_field": "x"})

    def test_agentic_suite_projection_is_an_exact_four_case_contract(self) -> None:
        suite = _agentic_suite()
        scenarios = tuple(case["id"] for case in suite["cases"])

        projected = _project_suite({"suite": suite})
        self.assertEqual("agentic-tools", projected["id"])
        self.assertNotIn("description", projected)

        invalid_suites = []
        unknown_root = json.loads(json.dumps(suite))
        unknown_root["extra"] = 1
        invalid_suites.append(unknown_root)
        wrong_budget = json.loads(json.dumps(suite))
        wrong_budget["cases"][0]["max_turns"] = 5
        invalid_suites.append(wrong_budget)
        duplicate_scenario = json.loads(json.dumps(suite))
        duplicate_scenario["cases"][1]["id"] = scenarios[0]
        duplicate_scenario["cases"][1]["case_id"] = f"{scenarios[0]}--ffffffffffff"
        invalid_suites.append(duplicate_scenario)
        unknown_case_field = json.loads(json.dumps(suite))
        unknown_case_field["cases"][0]["payload"] = "hidden"
        invalid_suites.append(unknown_case_field)
        for invalid in invalid_suites:
            with self.subTest(invalid=invalid), self.assertRaises(EvidenceError):
                _project_suite({"suite": invalid})

    def test_agentic_request_and_case_metrics_are_scalar_allowlisted(self) -> None:
        agentic_payload = {
            "schema_version": 1,
            "scenario_id": "agentic-two-hop",
            "variant": 2,
            "passed": True,
            "failure_code": None,
            "max_turns": 6,
            "max_output_tokens": 4096,
            "turns_used": 3,
            "expected_tool_calls": 2,
            "tool_calls_requested": 2,
            "tool_calls_executed": 2,
            "tool_calls_succeeded": 2,
            "tool_errors": 0,
            "malformed_tool_calls": 0,
            "unknown_tool_calls": 0,
            "final_answer_emitted": True,
            "final_answer_correct": True,
            "tool_sequence_correct": True,
            "recovery_required": False,
            "recovery_succeeded": False,
            "turn_limit_reached": False,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "emission_events": 4,
            "first_turn_ttft_s": 0.25,
            "request_elapsed_s": 1.5,
            "wall_s": 1.6,
            "length_terminated_turns": 0,
            "elapsed_s": 1.6,
            "ttft_s": 0.25,
            "finish_reason": "stop",
            "output_tps": 12.5,
            "decode_s": None,
            "decode_tps": None,
            "decode_metric_source": None,
        }
        projected_request = _project_request_result(
            agentic_payload,
            kind="agentic",
        )
        self.assertEqual(projected_request["scenario_id"], "agentic-two-hop")
        self.assertEqual(projected_request["tool_calls_executed"], 2)
        self.assertTrue(projected_request["final_answer_correct"])
        self.assertNotIn("content", projected_request)
        self.assertNotIn("tool_calls", projected_request)

        projected_case = _project_case(
            {
                "case_id": "agentic-two-hop--000000000000",
                "kind": "agentic",
                "requests": 3,
                "concurrency": 1,
                "prompt_tokens": 300,
                "completion_tokens": 60,
                "elapsed_s": 9.0,
                "measurement_valid": True,
                "measurement_annotations": [],
                "validation_passed": False,
                "agentic_tasks": 3,
                "agentic_tasks_succeeded": 2,
                "agentic_task_success_rate": 2 / 3,
                "agentic_tasks_per_s": 3 / 9,
                "agentic_max_turns": 6,
                "agentic_max_output_tokens_per_turn": 4096,
                "agentic_model_requests": 12,
                "agentic_model_requests_per_s": 12 / 9,
                "agentic_expected_tool_calls": 6,
                "agentic_tool_calls_requested": 7,
                "agentic_tool_calls_executed": 6,
                "agentic_tool_calls_succeeded": 6,
                "agentic_tool_errors": 0,
                "agentic_malformed_tool_calls": 1,
                "agentic_unknown_tool_calls": 0,
                "agentic_final_answers_emitted": 3,
                "agentic_final_answers_correct": 2,
                "agentic_tool_sequences_correct": 2,
                "agentic_recoveries_required": 0,
                "agentic_recoveries_succeeded": 0,
                "agentic_turn_limit_hits": 0,
                "agentic_length_terminated_turns": 0,
                "median_agentic_turns_used": 4,
                "median_agentic_task_wall_s": 3.0,
                "median_agentic_model_request_sum_s": 2.5,
                "median_agentic_first_turn_ttft_s": 0.2,
                "aggregate_output_tps": None,
                "decode_estimate_one_token_chunks": None,
                "decode_metric_source": None,
                "median_decode_tps": None,
                "median_e2e_s": 3.0,
                "median_estimated_decode_tps": None,
                "median_ttft_s": None,
                "p95_e2e_s": None,
                "p95_ttft_s": None,
                "request_tps": None,
                "telemetry": {},
            }
        )
        self.assertEqual(projected_case["kind"], "agentic")
        self.assertEqual(projected_case["agentic_tasks_succeeded"], 2)
        self.assertIsNone(projected_case["aggregate_output_tps"])

        for mutation in (
            {"schema_version": 2},
            {"variant": 3},
            {"completion_tokens": -1},
            {"tool_calls_executed": 3},
            {"failure_code": "missing_final"},
        ):
            with self.subTest(mutation=mutation):
                invalid = {**agentic_payload, **mutation}
                with self.assertRaises(EvidenceError):
                    _project_request_result(invalid, kind="agentic")

        with self.assertRaises(EvidenceError):
            _project_case(
                {
                    "kind": "decode",
                    "agentic_tasks": 3,
                }
            )

        base_event = {
            "event": "request_complete",
            "case_id": "agentic-two-hop--000000000000",
            "attempt_id": "attempt",
            "kind": "agentic",
            "repetition": 2,
            "burst_elapsed_s": 1.6,
            "result": agentic_payload,
            "validation": {"passed": True},
        }
        case_start = {
            "event": "case_start",
            "case_id": "agentic-two-hop--000000000000",
            "attempt_id": "attempt",
        }
        for missing in ("repetition", "validation"):
            malformed_event = dict(base_event)
            malformed_event.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(EvidenceError):
                _project_requests(
                    [case_start, malformed_event],
                    None,
                    evidence_kind="serving",
                )

        events = [case_start]
        for variant in range(3):
            events.append(
                {
                    **base_event,
                    "repetition": variant,
                    "result": {**agentic_payload, "variant": variant},
                }
            )
        selected_summary = {
            "cases": [
                {
                    "case_id": "agentic-two-hop--000000000000",
                    "attempt_id": "attempt",
                }
            ]
        }
        samples = _project_requests(
            events, selected_summary, evidence_kind="serving"
        )
        with self.assertRaisesRegex(EvidenceError, "exact agentic-tools suite"):
            _validate_agentic_aggregates(samples, {"cases": [projected_case]})
        with self.assertRaisesRegex(EvidenceError, "aggregate disagrees"):
            _validate_agentic_aggregates(
                samples,
                {"cases": [projected_case]},
                suite=_project_suite({"suite": _agentic_suite()}),
            )

    def test_source_tree_rejects_symlinks_hardlinks_and_fifos(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            (root / "linked.json").symlink_to(target)
            with self.assertRaisesRegex(EvidenceError, "special or linked file"):
                _assert_source_tree(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            os.link(target, root / "hardlink.json")
            with self.assertRaisesRegex(EvidenceError, "special or linked file"):
                _assert_source_tree(root)

        if hasattr(os, "mkfifo"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                os.mkfifo(root / "telemetry.pipe")
                with self.assertRaisesRegex(EvidenceError, "special or linked file"):
                    _assert_source_tree(root)

    def test_output_validator_rejects_forbidden_keys_paths_and_credentials(self) -> None:
        invalid_values = (
            {"request-id": "opaque"},
            {"nested": {"prompt": "opaque"}},
            {"artifact_path": "relative/model.gguf"},
            {"safe_metric": RAW_HOST_PATH},
            {"safe_metric": RAW_SECRET},
            {"safe_metric": "sk-" + "proj-0123456789abcdefghijklmnop"},
            {
                "safe_metric": (
                    "eyJ0123456789abcd.eyJ0123456789abcd."
                    "eyJ0123456789abcd"
                )
            },
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(EvidenceError):
                    _validate_output_value(value)

    def test_checksums_only_tree_fails_topology_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            EvidenceFixture.write_json(
                root / "checksums.json",
                {"files": {}, "schema_version": SCHEMA_VERSION},
            )

            with self.assertRaisesRegex(EvidenceError, "top-level layout changed"):
                verify_evidence(root)

    def test_serving_request_missing_attempt_id_fails_closed(self) -> None:
        events = [
            {
                "event": "request_complete",
                "case_id": "chat-case",
                "kind": "chat",
                "result": {"completion_tokens": 1},
            }
        ]

        with self.assertRaisesRegex(EvidenceError, "attempt identifier is missing"):
            _project_requests(events, None, evidence_kind="serving")

    def test_ninfer_report_rejects_non_v11_schema(self) -> None:
        report = dict.fromkeys(_NINFER_TOP_FIELDS)
        report.update(
            {
                "artifact_type": "ninfer_bench_report",
                "schema_version": 10,
                "tool": "ninfer_bench",
            }
        )

        with self.assertRaisesRegex(EvidenceError, "unrecognized NInfer report"):
            _project_ninfer_report(report)

    def test_summary_identity_fields_require_text(self) -> None:
        for field in ("status", "schema_version", "suite"):
            with self.subTest(field=field):
                with self.assertRaises(EvidenceError):
                    _project_summary({field: 42})

    def test_required_summary_numeric_fields_reject_bool_and_null(self) -> None:
        for value in (True, None):
            with self.subTest(scope="summary", value=value):
                with self.assertRaises(EvidenceError):
                    _project_summary({"completed_cases": value})
            with self.subTest(scope="case", value=value):
                with self.assertRaises(EvidenceError):
                    _project_case({"requests": value})

    def test_summary_aggregate_fields_reject_wrong_types(self) -> None:
        invalid = (
            ({"metrics": {"metric_source": 42}}, "metrics.metric_source"),
            ({"runtime": {"python": False}}, "runtime.python"),
            ({"metrics": {"perplexity": {"value": 2.0}}}, "metrics.perplexity"),
        )
        for summary, field in invalid:
            with self.subTest(field=field):
                with self.assertRaises(EvidenceError):
                    _project_summary(summary)

    def test_summary_aggregate_roots_require_objects(self) -> None:
        invalid = (
            {"metrics": 42},
            {"runtime": False},
            {"memory": []},
        )
        for summary in invalid:
            with self.subTest(root=next(iter(summary))):
                with self.assertRaises(EvidenceError):
                    _project_summary(summary)

    def test_nested_aggregate_keys_and_digest_lists_are_strict(self) -> None:
        invalid = (
            {
                "speculative_decoding": {
                    "accepted_tokens_per_position": {"not-an-index": 1}
                }
            },
            {
                "artifact_validation": {
                    "model_shard_sha256s": [["a" * 64]]
                }
            },
        )
        for summary in invalid:
            with self.subTest(root=next(iter(summary))):
                with self.assertRaises(EvidenceError):
                    _project_summary(summary)

    def test_summary_case_preserves_allowlisted_nullables(self) -> None:
        projected = _project_case(
            {
                "aggregate_output_tps": None,
                "case_id": None,
                "median_ttft_s": None,
                "validation_passed": None,
            }
        )

        self.assertEqual(
            {
                "aggregate_output_tps": None,
                "case_id": None,
                "median_ttft_s": None,
                "validation_passed": None,
            },
            projected,
        )

    def test_single_file_artifact_validation_omits_null_shard_fields(self) -> None:
        projected = _project_summary(
            {
                "artifact_validation": {
                    "model_shard_count": None,
                    "model_shard_sha256s": None,
                    "model_total_size_bytes": None,
                }
            }
        )

        self.assertEqual(
            {"artifact_validation": {}},
            projected,
        )

    def test_artifact_validation_requires_atomic_shard_metadata(self) -> None:
        invalid = (
            {"model_shard_count": 3},
            {
                "model_shard_count": 3,
                "model_shard_sha256s": ["a" * 64, "b" * 64],
                "model_total_size_bytes": 42,
            },
            {
                "model_shard_count": True,
                "model_shard_sha256s": ["a" * 64],
                "model_total_size_bytes": 42,
            },
        )
        for artifact_validation in invalid:
            with self.subTest(artifact_validation=artifact_validation):
                with self.assertRaises(EvidenceError):
                    _project_summary({"artifact_validation": artifact_validation})

    def test_quality_accuracy_category_rejects_null(self) -> None:
        with self.assertRaises(EvidenceError):
            _project_case(
                {"quality_accuracy_by_category": {"code": None}}
            )


if __name__ == "__main__":
    unittest.main()
