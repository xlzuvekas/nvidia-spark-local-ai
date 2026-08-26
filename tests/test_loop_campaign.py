from __future__ import annotations

from collections import Counter
import asyncio
import copy
import json
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import bench.loop_campaign as loop_campaign
from bench.loop_campaign import (
    BABILONG_REVISION,
    BABILONG_SOURCE,
    HALO_REVISION,
    HALO_SOURCE,
    HALO_SUBAGENT_ARGUMENT_ERROR,
    HALO_VERSION,
    OPENAI_AGENTS_VERSION,
    OPENAI_VERSION,
    PLAN_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    PYARROW_VERSION,
    RLM_REVISION,
    RLM_SOURCE,
    RLM_VERSION,
    RLM_ADMISSION_ADMITTED,
    RLM_COMPACTION_CONTEXT_TOKENS,
    RLM_COMPACTION_THRESHOLD_PCT,
    RLM_DEPTH2_HOLD,
    RLM_FORCED_COMPACTION_THRESHOLD_PCT,
    Journal,
    LoopCampaignError,
    _build_parser,
    _case_id,
    _bounded_halo_chat_create,
    _content_hash,
    _docker_worker_command,
    _halo_profile_candidates,
    _halo_subagent_builder_with_validation_recovery,
    _journal_held_rlm_cases,
    _needs_server_restart,
    _recover_plan_servers,
    _recover_plan_workers,
    _rlm_compaction_admission,
    _safe_error_result,
    _scalar_result,
    _validate_reasoning_profiles,
    build_cases,
    cleanup_campaign,
    compare_babilong_answer,
    generate_halo_trace_fixture,
    load_campaign_manifest,
    load_campaign_plan,
    parse_prometheus_counters,
    prometheus_delta,
    run_worker,
    score_halo_answer,
    summarize_campaign,
    _worker_rlm_babilong,
)


def _minimal_campaign_toml() -> str:
    """Return a complete, strictly valid campaign with a small case matrix."""

    return f'''schema_version = {PLAN_SCHEMA_VERSION}
id = "unit-loop"
description = "Synthetic unit-test campaign"

[window]
rlm_stop_at = "2035-01-01T00:00:00+00:00"
measurement_stop_at = "2035-01-01T01:00:00+00:00"
hard_stop_at = "2035-01-01T01:02:00+00:00"
cleanup_reserve_s = 120

[upstreams]
rlm_source = "{RLM_SOURCE}"
rlm_revision = "{RLM_REVISION}"
rlm_version = "{RLM_VERSION}"
halo_source = "{HALO_SOURCE}"
halo_revision = "{HALO_REVISION}"
halo_version = "{HALO_VERSION}"
openai_agents_version = "{OPENAI_AGENTS_VERSION}"
openai_version = "{OPENAI_VERSION}"
pyarrow_version = "{PYARROW_VERSION}"
babilong_source = "{BABILONG_SOURCE}"
babilong_revision = "{BABILONG_REVISION}"

[rlm]
model_profile = "synthetic-rlm"
reasoning_control = "fixed_unsupported"
lengths = ["4k", "64k"]
direct_lengths = ["4k"]
tasks = ["qa1", "qa2"]
row_indices = [0, 1]
max_iterations = 2
max_concurrent_subcalls = 2
max_total_tokens = 4096
max_output_tokens = 128
compaction = true
compaction_threshold_pct = 0.85
direct_timeout_s = 30
episode_timeout_s = 60
recursive_depth2_tasks = ["qa1"]
recursive_depth2_lengths = ["64k"]
recursive_depth2_rows = [0]
worker_isolation = "docker"

[halo]
model_profiles = ["synthetic-halo", "synthetic-halo-fallback"]
reasoning_effort = "none"
trace_counts = [32, 64]
seeds = [0, 1]
depths = [0, 1]
max_parallel = 2
max_turns = 4
max_output_tokens = 256
episode_timeout_s = 90
depth2_trace_counts = [32]
depth2_seeds = [0]
'''


def _load_minimal_config() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "campaign.toml"
        path.write_text(_minimal_campaign_toml(), encoding="utf-8")
        return load_campaign_manifest(path)


def _write_plan(path: Path, base: dict[str, object]) -> dict[str, object]:
    plan = copy.deepcopy(base)
    plan["fingerprint"] = _content_hash(plan)
    plan["integrity_hash"] = _content_hash(plan)
    path.write_text(
        json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return plan


class LoopCaseConstructionTests(unittest.TestCase):
    def test_minimal_valid_manifest_builds_deterministic_counterbalanced_cases(
        self,
    ) -> None:
        config = _load_minimal_config()
        original = copy.deepcopy(config)

        first = build_cases(config)
        second = build_cases(config)

        self.assertEqual(first, second)
        self.assertEqual(config, original)
        self.assertEqual(len(first), 22)
        self.assertEqual(len({case["case_id"] for case in first}), 22)
        self.assertTrue(
            all(
                re.fullmatch(r"(?:rlm|halo)-[0-9a-f]{16}", str(case["case_id"]))
                for case in first
            )
        )
        for case in first:
            raw = {key: value for key, value in case.items() if key != "case_id"}
            self.assertEqual(case["case_id"], _case_id(raw))

        rlm_cases = [case for case in first if case["phase"] == "rlm"]
        halo_cases = [case for case in first if case["phase"] == "halo"]
        self.assertEqual(len(rlm_cases), 13)
        self.assertEqual(len(halo_cases), 9)
        self.assertEqual(
            {case["reasoning_control"] for case in rlm_cases},
            {"fixed_unsupported"},
        )
        recursive = [case for case in rlm_cases if case["treatment"] != "direct"]
        direct = [case for case in rlm_cases if case["treatment"] == "direct"]
        self.assertTrue(all(case["compaction"] is True for case in recursive))
        self.assertTrue(
            all(
                case["compaction_threshold_pct"]
                == RLM_COMPACTION_THRESHOLD_PCT
                for case in recursive
            )
        )
        self.assertTrue(all(case["compaction"] is False for case in direct))
        self.assertTrue(
            all(case["compaction_threshold_pct"] is None for case in direct)
        )
        self.assertEqual(
            {
                case["admission_status"]
                for case in rlm_cases
                if case["treatment"] != "rlm_depth2"
            },
            {RLM_ADMISSION_ADMITTED},
        )
        self.assertEqual(
            {case["reasoning_effort"] for case in halo_cases}, {"none"}
        )
        self.assertEqual(
            [case["treatment"] for case in rlm_cases[:4]],
            ["rlm_depth1", "direct", "direct", "rlm_depth1"],
        )
        self.assertTrue(
            all(
                case["context_length"] == "4k"
                for case in rlm_cases
                if case["treatment"] == "direct"
            )
        )
        self.assertEqual(
            [case["max_depth"] for case in halo_cases[:4]],
            [0, 1, 1, 0],
        )
        self.assertEqual(rlm_cases[-1]["treatment"], "rlm_depth2")
        self.assertEqual(rlm_cases[-1]["admission_status"], RLM_DEPTH2_HOLD)
        self.assertEqual(halo_cases[-1]["treatment"], "halo_depth2")

        changed = copy.deepcopy(config)
        changed["rlm"]["compaction_threshold_pct"] = 0.8
        changed_case = next(
            case
            for case in build_cases(changed)
            if case["treatment"] == "rlm_depth1"
        )
        original_case = next(
            case for case in first if case["treatment"] == "rlm_depth1"
        )
        self.assertNotEqual(changed_case["case_id"], original_case["case_id"])

    def test_halo_profile_selection_locks_after_first_success(self) -> None:
        plan = {"halo": {"model_profiles": ["primary", "fallback"]}}
        self.assertEqual(
            _halo_profile_candidates(plan, []), ["primary", "fallback"]
        )
        primary_failure = [
            {
                "event": "case_failed",
                "phase": "halo",
                "profile_id": "primary",
            }
        ]
        self.assertEqual(
            _halo_profile_candidates(plan, primary_failure), ["fallback"]
        )
        primary_success_then_failure = [
            {
                "event": "case_complete",
                "phase": "halo",
                "profile_id": "primary",
            },
            *primary_failure,
        ]
        self.assertEqual(
            _halo_profile_candidates(plan, primary_success_then_failure), ["primary"]
        )

        single_plan = {"halo": {"model_profiles": ["only"]}}
        self.assertEqual(_halo_profile_candidates(single_plan, []), ["only"])
        self.assertEqual(
            _halo_profile_candidates(
                single_plan,
                [
                    {
                        "event": "case_failed",
                        "phase": "halo",
                        "profile_id": "only",
                    }
                ],
            ),
            ["only"],
        )

    def test_manifest_accepts_one_or_two_halo_profiles_and_rejects_other_counts(
        self,
    ) -> None:
        original = 'model_profiles = ["synthetic-halo", "synthetic-halo-fallback"]'
        accepted = (
            ["synthetic-halo"],
            ["synthetic-halo", "synthetic-halo-fallback"],
        )
        for profiles in accepted:
            with self.subTest(accepted=profiles):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "campaign.toml"
                    path.write_text(
                        _minimal_campaign_toml().replace(
                            original, f"model_profiles = {json.dumps(profiles)}"
                        ),
                        encoding="utf-8",
                    )
                    config = load_campaign_manifest(path)
                    self.assertEqual(config["halo"]["model_profiles"], profiles)

        rejected = (
            ([], "non-empty unique string list"),
            (["one", "two", "three"], "at most one fallback"),
            (["duplicate", "duplicate"], "non-empty unique string list"),
        )
        for profiles, message in rejected:
            with self.subTest(rejected=profiles):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "campaign.toml"
                    path.write_text(
                        _minimal_campaign_toml().replace(
                            original, f"model_profiles = {json.dumps(profiles)}"
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(LoopCampaignError, message):
                        load_campaign_manifest(path)

    def test_manifest_rejects_unfrozen_reasoning_controls(self) -> None:
        for old, new, message in (
            (
                'reasoning_control = "fixed_unsupported"',
                'reasoning_control = "none"',
                "fixed_unsupported",
            ),
            (
                'reasoning_effort = "none"',
                'reasoning_effort = "high"',
                "reasoning_effort",
            ),
        ):
            with self.subTest(new=new), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "campaign.toml"
                path.write_text(_minimal_campaign_toml().replace(old, new), encoding="utf-8")
                with self.assertRaisesRegex(LoopCampaignError, message):
                    load_campaign_manifest(path)

    def test_manifest_requires_exact_compaction_controls(self) -> None:
        mutations = (
            ("compaction = true", "compaction = false", "must be true"),
            (
                "compaction_threshold_pct = 0.85",
                "compaction_threshold_pct = 1",
                "exactly 0.85",
            ),
            ("compaction = true\n", "", "keys do not match"),
        )
        for old, new, message in mutations:
            with self.subTest(new=new), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "campaign.toml"
                path.write_text(
                    _minimal_campaign_toml().replace(old, new), encoding="utf-8"
                )
                with self.assertRaisesRegex(LoopCampaignError, message):
                    load_campaign_manifest(path)

    def test_optional_forced_compaction_diagnostic_is_strict_and_precedes_core(
        self,
    ) -> None:
        table = '''[rlm.compaction_diagnostic]
threshold_pct = 0.20
tasks = ["qa1"]
lengths = ["64k"]
row_indices = [0]

[halo]'''
        document = _minimal_campaign_toml().replace("[halo]", table)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "campaign.toml"
            path.write_text(document, encoding="utf-8")
            config = load_campaign_manifest(path)

        cases = build_cases(config)
        self.assertEqual(len(cases), 23)
        forced = cases[0]
        self.assertEqual(forced["treatment"], "rlm_depth1_forced_compaction")
        self.assertEqual(
            forced["compaction_threshold_pct"],
            RLM_FORCED_COMPACTION_THRESHOLD_PCT,
        )
        self.assertEqual(forced["context_length"], "64k")
        self.assertEqual(forced["task"], "qa1")
        self.assertEqual(forced["row_index"], 0)
        self.assertEqual(forced["max_depth"], 1)
        self.assertEqual(forced["max_iterations"], config["rlm"]["max_iterations"])
        self.assertEqual(forced["timeout_s"], config["rlm"]["episode_timeout_s"])
        self.assertEqual(forced["admission_status"], RLM_ADMISSION_ADMITTED)

        mutations = (
            ("threshold_pct = 0.20", "threshold_pct = 0.25", "exactly 0.20"),
            ("row_indices = [0]", "row_indices = [99]", "core subsets"),
            (
                'row_indices = [0]\n',
                'row_indices = [0]\nunknown = true\n',
                "keys do not match",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(new=new), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "campaign.toml"
                path.write_text(document.replace(old, new), encoding="utf-8")
                with self.assertRaisesRegex(LoopCampaignError, message):
                    load_campaign_manifest(path)

    def test_compaction_admission_preserves_served_context_headroom(self) -> None:
        config = _load_minimal_config()
        rlm = copy.deepcopy(config["rlm"])
        rlm["max_output_tokens"] = 768
        admission = _rlm_compaction_admission(
            rlm, served_context_tokens=40_960
        )
        self.assertEqual(admission["package_context_tokens"], 32_768)
        self.assertEqual(admission["package_context_tokens"], RLM_COMPACTION_CONTEXT_TOKENS)
        self.assertEqual(admission["threshold_tokens"], 27_852)
        self.assertEqual(admission["output_reserve_tokens"], 768)
        self.assertEqual(admission["headroom_tokens"], 12_340)
        self.assertTrue(admission["depth1_admitted"])
        self.assertFalse(admission["depth2_admitted"])
        with self.assertRaisesRegex(LoopCampaignError, "output headroom"):
            _rlm_compaction_admission(rlm, served_context_tokens=28_000)

    def test_depth2_hold_is_journaled_once(self) -> None:
        config = _load_minimal_config()
        cases = build_cases(config)
        held = [
            case
            for case in cases
            if case.get("admission_status") == RLM_DEPTH2_HOLD
        ]
        self.assertEqual(len(held), 1)
        with tempfile.TemporaryDirectory() as temporary:
            journal = Journal(Path(temporary) / "journal.jsonl")
            plan = {"cases": cases}
            _journal_held_rlm_cases(plan=plan, journal=journal)
            _journal_held_rlm_cases(plan=plan, journal=journal)
            events = [
                event
                for event in journal.events()
                if event["event"] == "case_skipped_held"
            ]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["case_id"], held[0]["case_id"])
        self.assertEqual(events[0]["admission_status"], RLM_DEPTH2_HOLD)

    def test_final_exhausted_case_does_not_require_a_cold_restart(self) -> None:
        self.assertTrue(_needs_server_restart(pending_count=1, current_attempt=1))
        self.assertTrue(_needs_server_restart(pending_count=2, current_attempt=2))
        self.assertFalse(_needs_server_restart(pending_count=1, current_attempt=2))
        with self.assertRaises(ValueError):
            _needs_server_restart(pending_count=0, current_attempt=1)

    def test_reasoning_profile_admission_rejects_drift(self) -> None:
        config = {
            "rlm": {"model_profile": "rlm"},
            "halo": {"model_profiles": ["primary", "fallback"]},
        }
        thinking_off = '{"chat_template_kwargs":{"enable_thinking":false}}'
        server_off = '{"enable_thinking":false}'
        models = {
            "rlm": SimpleNamespace(request_body_json=None),
            "primary": SimpleNamespace(
                request_body_json=thinking_off,
                args=("--default-chat-template-kwargs", server_off),
            ),
            "fallback": SimpleNamespace(
                request_body_json=thinking_off,
                args=("--default-chat-template-kwargs", server_off),
            ),
        }
        _validate_reasoning_profiles(config, models)

        drifted = dict(models)
        drifted["primary"] = SimpleNamespace(
            request_body_json=thinking_off,
            args=("--default-chat-template-kwargs", '{"enable_thinking":true}'),
        )
        with self.assertRaisesRegex(LoopCampaignError, "enable_thinking=false"):
            _validate_reasoning_profiles(config, drifted)

        drifted = dict(models)
        drifted["rlm"] = SimpleNamespace(request_body_json=thinking_off)
        with self.assertRaisesRegex(LoopCampaignError, "reasoning request knob"):
            _validate_reasoning_profiles(config, drifted)


class BabiLongScoringTests(unittest.TestCase):
    def test_requires_exactly_one_non_question_location_label(self) -> None:
        self.assertTrue(
            compare_babilong_answer(
                target="kitchen",
                output="Kitchen.",
                question="Where is Mary?",
                task="qa1",
            )
        )
        self.assertTrue(
            compare_babilong_answer(
                target="kitchen",
                output="kitchen. bedroom appears in later explanation",
                question="Where is Mary?",
                task="qa2",
            )
        )
        self.assertFalse(
            compare_babilong_answer(
                target="kitchen",
                output="kitchen and bedroom",
                question="Where is Mary?",
                task="qa3",
            )
        )
        self.assertFalse(
            compare_babilong_answer(
                target="kitchen",
                output="kitchen",
                question="Did Mary leave the kitchen?",
                task="qa4",
            )
        )
        self.assertFalse(
            compare_babilong_answer(
                target="kitchen",
                output="office",
                question="Where is Mary?",
                task="qa1",
            )
        )

    def test_rejects_tasks_outside_the_frozen_scorer(self) -> None:
        with self.assertRaisesRegex(LoopCampaignError, "qa1-qa4"):
            compare_babilong_answer(
                target="kitchen",
                output="kitchen",
                question="Where is Mary?",
                task="qa5",
            )


class HaloFixtureAndScoringTests(unittest.TestCase):
    def test_fixture_is_byte_and_truth_deterministic_with_two_spans_per_trace(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path = root / "first.jsonl"
            second_path = root / "second.jsonl"

            first_truth = generate_halo_trace_fixture(
                first_path, trace_count=32, seed=7
            )
            first_bytes = first_path.read_bytes()
            repeated_truth = generate_halo_trace_fixture(
                first_path, trace_count=32, seed=7
            )
            second_truth = generate_halo_trace_fixture(
                second_path, trace_count=32, seed=7
            )

            self.assertEqual(first_truth, repeated_truth)
            self.assertEqual(first_truth, second_truth)
            self.assertEqual(first_bytes, first_path.read_bytes())
            self.assertEqual(first_bytes, second_path.read_bytes())
            self.assertEqual(first_truth["trace_count"], 32)
            self.assertEqual(first_truth["span_count"], 64)

            rows = [json.loads(line) for line in first_bytes.splitlines()]
            self.assertEqual(len(rows), 64)
            trace_counts = Counter(row["trace_id"] for row in rows)
            self.assertEqual(len(trace_counts), 32)
            self.assertEqual(set(trace_counts.values()), {2})
            kinds = Counter(row["attributes"]["openinference.span.kind"] for row in rows)
            self.assertEqual(kinds, Counter({"AGENT": 32, "TOOL": 32}))

    def test_perfect_halo_answer_scores_one_and_invalid_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            truth = generate_halo_trace_fixture(
                Path(temporary) / "fixture.jsonl", trace_count=40, seed=3
            )

        families = [
            {
                "id": family,
                "count": truth["family_counts"][family],
                "example_trace_ids": [truth["family_trace_ids"][family][0]],
            }
            for family in truth["active_families"]
        ]
        perfect = score_halo_answer(
            "result: " + json.dumps({"families": families}), truth
        )
        self.assertTrue(perfect["json_valid"])
        self.assertEqual(perfect["predicted_family_count"], 4)
        for metric in (
            "family_precision",
            "family_recall",
            "family_f1",
            "mean_count_accuracy",
            "exact_count_rate",
            "citation_precision",
            "citation_family_coverage",
        ):
            self.assertEqual(perfect[metric], 1.0)

        family = truth["active_families"][0]
        invalid = score_halo_answer(
            json.dumps(
                {
                    "families": [
                        {
                            "id": family,
                            "count": "2",
                            "example_trace_ids": [],
                        }
                    ]
                }
            ),
            truth,
        )
        self.assertFalse(invalid["json_valid"])
        self.assertEqual(invalid["predicted_family_count"], 0)
        self.assertEqual(invalid["family_f1"], 0.0)

        extra_top_level = score_halo_answer(
            json.dumps({"families": families, "explanation": "synthetic"}), truth
        )
        self.assertFalse(extra_top_level["json_valid"])
        self.assertEqual(extra_top_level["predicted_family_count"], 0)

        malformed = score_halo_answer("not json", truth)
        self.assertFalse(malformed["json_valid"])
        self.assertTrue(
            all(
                value in {False, 0, 0.0}
                for value in malformed.values()
            )
        )


class PrometheusCounterTests(unittest.TestCase):
    def test_parser_aggregates_labeled_series_and_ignores_invalid_samples(self) -> None:
        exposition = """# HELP vllm:prompt_tokens_total prompt tokens
vllm:prompt_tokens_total{model_name="one"} 10
vllm:prompt_tokens_total{model_name="two"} 2
vllm:generation_tokens_total 5 123456
vllm:request_success_total{finished_reason="stop"} 2
vllm:request_success_total{finished_reason="length"} 3
vllm:prefix_cache_hits_total -1
vllm:prefix_cache_queries_total NaN
vllm:prompt_tokens_cached_total +Inf
unrelated_counter 999
malformed
"""

        self.assertEqual(
            parse_prometheus_counters(exposition),
            {
                "vllm:prompt_tokens_total": 12.0,
                "vllm:generation_tokens_total": 5.0,
                "vllm:request_success_total": 5.0,
            },
        )

    def test_delta_uses_common_counters_and_rejects_reset_or_nonfinite_delta(
        self,
    ) -> None:
        before = {
            "vllm:prompt_tokens_total": 10.0,
            "vllm:generation_tokens_total": 4.0,
            "vllm:prefix_cache_hits_total": 1.0,
        }
        after = {
            "vllm:prompt_tokens_total": 16.0,
            "vllm:generation_tokens_total": 9.0,
            "vllm:request_success_total": 2.0,
        }
        self.assertEqual(
            prometheus_delta(before, after),
            {
                "vllm:prompt_tokens_total": 6.0,
                "vllm:generation_tokens_total": 5.0,
            },
        )

        with self.assertRaisesRegex(LoopCampaignError, "counter reset"):
            prometheus_delta(
                {"vllm:prompt_tokens_total": 10.0},
                {"vllm:prompt_tokens_total": 9.0},
            )
        with self.assertRaisesRegex(LoopCampaignError, "counter reset"):
            prometheus_delta(
                {"vllm:prompt_tokens_total": 10.0},
                {"vllm:prompt_tokens_total": float("nan")},
            )


class ScalarResultTests(unittest.TestCase):
    def test_safe_error_result_keeps_only_scalar_failure_fingerprints(self) -> None:
        class ProviderError(Exception):
            code = "invalid_function_parameters"
            status_code = 400

        try:
            try:
                raise ProviderError("private provider response")
            except ProviderError as cause:
                raise RuntimeError("private outer message") from cause
        except RuntimeError as error:
            result = _safe_error_result(error)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(result["error_cause_type"], "ProviderError")
        self.assertEqual(result["error_code"], "invalid_function_parameters")
        self.assertEqual(result["error_http_status"], 400)
        self.assertEqual(result["error_frame_file"], "test_loop_campaign.py")
        self.assertEqual(result["error_cause_frame_file"], "test_loop_campaign.py")
        self.assertNotIn("message", result)
        self.assertNotIn("private", json.dumps(result))
        self.assertEqual(_scalar_result(result)["error_type"], "RuntimeError")

        ProviderError.code = "req_identifier_123"
        identifier_result = _safe_error_result(ProviderError("private"))
        self.assertNotIn("error_code", identifier_result)

        token_error = RuntimeError("private")
        token_error.tokens_used = 140_027  # type: ignore[attr-defined]
        token_error.token_limit = 131_072  # type: ignore[attr-defined]
        token_result = _safe_error_result(token_error)
        self.assertEqual(token_result["error_tokens_used"], 140_027)
        self.assertEqual(token_result["error_token_limit"], 131_072)

        response_error = RuntimeError("private response error")
        response_error.response = SimpleNamespace(  # type: ignore[attr-defined]
            status_code=503,
            text="private provider response",
        )
        response_result = _safe_error_result(response_error)
        self.assertEqual(response_result["error_http_status"], 503)
        self.assertNotIn("response", json.dumps(response_result))
        self.assertNotIn("private", json.dumps(response_result))

    def test_safe_error_result_reads_allowlisted_scalars_from_intermediate_cause(
        self,
    ) -> None:
        class BadRequestError(Exception):
            code = "invalid_function_parameters"
            status_code = 400
            tokens_used = 257
            token_limit = 256

        try:
            try:
                try:
                    raise ValueError("private root response")
                except ValueError as cause:
                    raise BadRequestError("private provider response") from cause
            except BadRequestError as cause:
                raise RuntimeError("private outer message") from cause
        except RuntimeError as error:
            result = _safe_error_result(error)

        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(result["error_cause_type"], "ValueError")
        self.assertEqual(result["error_code"], "invalid_function_parameters")
        self.assertEqual(result["error_http_status"], 400)
        self.assertEqual(result["error_tokens_used"], 257)
        self.assertEqual(result["error_token_limit"], 256)
        self.assertEqual(result["error_frame_file"], "test_loop_campaign.py")
        self.assertEqual(result["error_cause_frame_file"], "test_loop_campaign.py")
        serialized = json.dumps(result)
        self.assertNotIn("private", serialized)
        self.assertNotIn("message", serialized)
        self.assertNotIn("response", serialized)

    def test_accepts_only_allowlisted_scalar_shapes(self) -> None:
        self.assertEqual(
            _scalar_result(
                {
                    "status": "ok",
                    "error_type": "SyntheticError",
                    "passed": True,
                    "count": 3,
                    "ratio": 0.5,
                    "optional": None,
                }
            ),
            {
                "error_type": "SyntheticError",
                "passed": True,
                "count": 3,
                "ratio": 0.5,
                "optional": None,
            },
        )

    def test_rejects_text_content_nested_values_bad_keys_and_nonfinite_numbers(
        self,
    ) -> None:
        invalid_results = (
            {"status": "ok", "answer": "private completion"},
            {"status": "ok", "items": [1, 2]},
            {"status": "ok", "nested": {"count": 1}},
            {"status": "ok", "BadKey": 1},
            {"status": "ok", "ratio": float("inf")},
        )
        for result in invalid_results:
            with self.subTest(result=result):
                with self.assertRaises(LoopCampaignError):
                    _scalar_result(result)

    def test_isolated_worker_attaches_stdin_and_uses_frozen_network(self) -> None:
        plan_id = "a" * 64
        command, container_name = _docker_worker_command(
            Path("synthetic-worker-source"),
            worker_kind="rlm",
            isolation_plan=plan_id,
            isolation_case="rlm-" + "b" * 16,
        )
        self.assertIn("--interactive", command)
        self.assertEqual(
            command[command.index("--network") + 1],
            "sparkbench-loop-" + plan_id[:12],
        )
        self.assertTrue(container_name.startswith("sparkbench-loop-worker-"))

    def test_rlm_worker_wires_and_counts_root_compaction(self) -> None:
        captured: dict[str, object] = {}

        class FakeRLM:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            def _completion_turn(self, *_args: object, **_kwargs: object) -> None:
                return None

            def _compact_history(self, *_args: object, **_kwargs: object) -> list[object]:
                return []

            def completion(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
                self._completion_turn(None, None, None)
                self._compact_history(None, None, [], 1)
                usage = SimpleNamespace(
                    model_usage_summaries={
                        "synthetic": SimpleNamespace(total_calls=1)
                    },
                    total_input_tokens=100,
                    total_output_tokens=20,
                )
                return SimpleNamespace(
                    metadata=None,
                    response="kitchen",
                    usage_summary=usage,
                    execution_time=0.25,
                )

        payload = {
            "reasoning_control": "fixed_unsupported",
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "synthetic",
            "request_timeout_s": 30,
            "max_depth": 1,
            "max_iterations": 8,
            "max_total_tokens": 262_144,
            "max_concurrent_subcalls": 2,
            "max_output_tokens": 768,
            "engine_timeout_s": 60,
            "compaction": True,
            "compaction_threshold_pct": 0.85,
            "context": "synthetic context",
            "question": "Where is Mary?",
            "target": "kitchen",
            "task": "qa1",
        }
        with patch.dict("sys.modules", {"rlm": SimpleNamespace(RLM=FakeRLM)}):
            result = _worker_rlm_babilong(payload)

        self.assertEqual(captured["max_tokens"], 262_144)
        self.assertEqual(captured["sampling_args"]["max_tokens"], 768)  # type: ignore[index]
        self.assertEqual(captured["sub_sampling_args"]["max_tokens"], 768)  # type: ignore[index]
        self.assertIs(captured["compaction"], True)
        self.assertEqual(captured["compaction_threshold_pct"], 0.85)
        self.assertEqual(result["compaction_enabled"], True)
        self.assertEqual(result["compaction_threshold_pct"], 0.85)
        self.assertEqual(result["compaction_count"], 1)
        self.assertEqual(result["iterations"], 1)

        forced_payload = {
            **payload,
            "compaction_threshold_pct": RLM_FORCED_COMPACTION_THRESHOLD_PCT,
        }
        with patch.dict("sys.modules", {"rlm": SimpleNamespace(RLM=FakeRLM)}):
            forced_result = _worker_rlm_babilong(forced_payload)
        self.assertEqual(
            captured["compaction_threshold_pct"],
            RLM_FORCED_COMPACTION_THRESHOLD_PCT,
        )
        self.assertEqual(
            forced_result["compaction_threshold_pct"],
            RLM_FORCED_COMPACTION_THRESHOLD_PCT,
        )
        self.assertEqual(forced_result["compaction_count"], 1)
        self.assertEqual(forced_result["iterations"], 1)

    def test_halo_direct_chat_calls_are_deterministic_and_output_bounded(self) -> None:
        async def original(
            _resource: object, *_args: object, **kwargs: object
        ) -> dict[str, object]:
            return kwargs

        bounded = _bounded_halo_chat_create(original, 1024)
        defaults = asyncio.run(
            bounded(object(), model="synthetic", messages=[])
        )
        self.assertEqual(defaults["temperature"], 0.0)
        self.assertEqual(defaults["max_tokens"], 1024)

        clamped = asyncio.run(
            bounded(
                object(),
                model="synthetic",
                messages=[],
                temperature=0.8,
                max_completion_tokens=4096,
                max_tokens=2048,
            )
        )
        self.assertEqual(clamped["temperature"], 0.0)
        self.assertEqual(clamped["max_completion_tokens"], 1024)
        self.assertNotIn("max_tokens", clamped)


class HaloSubagentValidationRecoveryTests(unittest.TestCase):
    class SyntheticValidationError(Exception):
        pass

    class SyntheticInfrastructureError(Exception):
        pass

    @classmethod
    def _tool(cls) -> SimpleNamespace:
        async def invoke(_context: object, raw_arguments: str) -> str:
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as error:
                raise cls.SyntheticValidationError from error
            if not isinstance(arguments, dict) or not isinstance(
                arguments.get("input"), str
            ):
                raise cls.SyntheticValidationError
            return "valid-child-result"

        def build() -> SimpleNamespace:
            return SimpleNamespace(on_invoke_tool=invoke)

        guarded_build = _halo_subagent_builder_with_validation_recovery(
            build,
            cls.SyntheticValidationError,
        )
        return guarded_build()

    def test_valid_call_preserves_normal_result(self) -> None:
        result = asyncio.run(
            self._tool().on_invoke_tool(object(), '{"input":"question"}')
        )
        self.assertEqual(result, "valid-child-result")

    def test_malformed_json_returns_constant_retry_error(self) -> None:
        result = asyncio.run(self._tool().on_invoke_tool(object(), "{"))
        self.assertEqual(result, HALO_SUBAGENT_ARGUMENT_ERROR)

    def test_missing_input_returns_constant_retry_error(self) -> None:
        result = asyncio.run(self._tool().on_invoke_tool(object(), "{}"))
        self.assertEqual(result, HALO_SUBAGENT_ARGUMENT_ERROR)

    def test_wrong_type_input_returns_constant_retry_error(self) -> None:
        result = asyncio.run(
            self._tool().on_invoke_tool(object(), '{"input":123}')
        )
        self.assertEqual(result, HALO_SUBAGENT_ARGUMENT_ERROR)

    def test_validation_error_does_not_echo_input(self) -> None:
        private_marker = "synthetic-private-tool-input"
        raw_arguments = json.dumps({"input": {"value": private_marker}})
        result = asyncio.run(
            self._tool().on_invoke_tool(object(), raw_arguments)
        )
        self.assertEqual(result, HALO_SUBAGENT_ARGUMENT_ERROR)
        self.assertNotIn(private_marker, result)

    def test_non_validation_error_propagates(self) -> None:
        async def invoke(_context: object, _raw_arguments: str) -> str:
            raise self.SyntheticInfrastructureError

        def build() -> SimpleNamespace:
            return SimpleNamespace(on_invoke_tool=invoke)

        guarded_build = _halo_subagent_builder_with_validation_recovery(
            build,
            self.SyntheticValidationError,
        )
        tool = guarded_build()
        with self.assertRaises(self.SyntheticInfrastructureError):
            asyncio.run(tool.on_invoke_tool(object(), '{"input":"question"}'))


class WorkerCancellationTests(unittest.TestCase):
    def test_stop_request_interrupts_blocking_worker_communication(self) -> None:
        class BlockingProcess:
            pid = 4321

            def __init__(self) -> None:
                self.calls = 0

            def communicate(
                self, *, input: bytes | None, timeout: float
            ) -> tuple[bytes, bytes]:
                self.calls += 1
                self.asserted_input = input
                self.asserted_timeout = timeout
                loop_campaign._STOP_REQUESTED = True
                raise subprocess.TimeoutExpired("synthetic-worker", timeout)

        process = BlockingProcess()
        with (
            patch.object(loop_campaign, "_STOP_REQUESTED", False),
            patch.object(loop_campaign.subprocess, "Popen", return_value=process),
            patch.object(
                loop_campaign,
                "_terminate_worker_process",
                return_value=True,
            ) as terminate,
        ):
            result = run_worker(
                worker_kind="direct",
                payload={"synthetic": True},
                timeout_s=30,
                workspace=Path("synthetic-workspace"),
                isolated=False,
            )

        self.assertEqual(result, {"status": "stopped"})
        self.assertEqual(process.calls, 1)
        self.assertLessEqual(process.asserted_timeout, 1.0)
        terminate.assert_called_once_with(process, container_name=None)

    def test_worker_termination_targets_exact_container_and_reaps_client(self) -> None:
        class RunningProcess:
            pid = 9876

            def __init__(self) -> None:
                self.wait_timeouts: list[float] = []

            def poll(self) -> None:
                return None

            def wait(self, *, timeout: float) -> int:
                self.wait_timeouts.append(timeout)
                return 0

        process = RunningProcess()
        removed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with (
            patch.object(loop_campaign.subprocess, "run", return_value=removed) as run,
            patch.object(loop_campaign.os, "killpg") as killpg,
        ):
            cleanup_ok = loop_campaign._terminate_worker_process(
                process, container_name="sparkbench-loop-worker-0123456789abcdef"
            )

        self.assertTrue(cleanup_ok)
        self.assertEqual(
            run.call_args.args[0],
            [
                "/usr/bin/docker",
                "rm",
                "-f",
                "sparkbench-loop-worker-0123456789abcdef",
            ],
        )
        killpg.assert_called_once_with(process.pid, loop_campaign.signal.SIGTERM)
        self.assertEqual(process.wait_timeouts, [5])


class ExactPlanCleanupTests(unittest.TestCase):
    @staticmethod
    def _write_legacy_plan(run_dir: Path) -> dict[str, object]:
        raw_case: dict[str, object] = {
            "phase": "halo",
            "treatment": "halo_depth0",
            "reasoning_effort": "none",
            "trace_count": 32,
            "seed": 0,
            "max_depth": 0,
        }
        case = {**raw_case, "case_id": _case_id(raw_case)}
        return _write_plan(
            run_dir / "plan.json",
            {
                "schema_version": 1,
                "protocol_version": 1,
                "campaign_id": "synthetic-cleanup",
                "rlm": {
                    "model_profile": "synthetic-rlm",
                    "reasoning_control": "fixed_unsupported",
                },
                "halo": {
                    "model_profiles": ["synthetic-halo"],
                    "reasoning_effort": "none",
                },
                "models": {
                    "synthetic-halo": {},
                    "synthetic-rlm": {},
                },
                "cases": [case],
            },
        )

    def test_cleanup_is_idempotent_and_orders_exact_resource_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            self._write_legacy_plan(run_dir)
            operations: list[str] = []
            with (
                patch.object(
                    loop_campaign,
                    "_recover_plan_workers",
                    side_effect=lambda *_: operations.append("workers"),
                ),
                patch.object(
                    loop_campaign,
                    "_recover_plan_servers",
                    side_effect=lambda *_: operations.append("server"),
                ),
                patch.object(
                    loop_campaign,
                    "_remove_worker_network",
                    side_effect=lambda *_: operations.append("network") or True,
                ),
            ):
                first = cleanup_campaign(run_dir=run_dir, workspace=root)
                second = cleanup_campaign(run_dir=run_dir, workspace=root)

            self.assertEqual(first, {"status": "cleanup_verified"})
            self.assertEqual(second, first)
            self.assertEqual(
                operations,
                ["workers", "server", "network"] * 2,
            )
            cleanup_events = [
                event["event"]
                for event in Journal(run_dir / "journal.jsonl").events()
                if event["event"].startswith("campaign_cleanup_")
            ]
            self.assertEqual(
                cleanup_events,
                ["campaign_cleanup_verified", "campaign_cleanup_verified"],
            )

    def test_cleanup_validates_plan_before_touching_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            plan = self._write_legacy_plan(run_dir)
            plan["campaign_id"] = "tampered"
            (run_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            with patch.object(loop_campaign, "_recover_plan_workers") as workers:
                with self.assertRaisesRegex(LoopCampaignError, "integrity check"):
                    cleanup_campaign(run_dir=run_dir, workspace=root)
            workers.assert_not_called()

    def test_cleanup_attempts_every_class_but_refuses_foreign_server(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            self._write_legacy_plan(run_dir)
            operations: list[str] = []

            def reject_server(*_: object) -> None:
                operations.append("server")
                raise LoopCampaignError("foreign server")

            with (
                patch.object(
                    loop_campaign,
                    "_recover_plan_workers",
                    side_effect=lambda *_: operations.append("workers"),
                ),
                patch.object(
                    loop_campaign,
                    "_recover_plan_servers",
                    side_effect=reject_server,
                ),
                patch.object(
                    loop_campaign,
                    "_remove_worker_network",
                    side_effect=lambda *_: operations.append("network") or True,
                ),
            ):
                with self.assertRaisesRegex(
                    LoopCampaignError, "cleanup was not verified"
                ):
                    cleanup_campaign(run_dir=run_dir, workspace=root)

            self.assertEqual(operations, ["workers", "server", "network"])
            events = Journal(run_dir / "journal.jsonl").events()
            self.assertEqual(events[-1]["event"], "campaign_cleanup_failed")
            self.assertEqual(events[-1]["failure_count"], 1)

    def test_worker_recovery_uses_both_exact_ownership_labels(self) -> None:
        plan = {"fingerprint": "a" * 64}
        listed = subprocess.CompletedProcess(
            [], 0, stdout="worker-container-id\n", stderr=""
        )
        removed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as temporary:
            journal = Journal(Path(temporary) / "journal.jsonl")
            with patch.object(
                loop_campaign.subprocess,
                "run",
                side_effect=[listed, removed],
            ) as run:
                _recover_plan_workers(plan, journal)

        inspect_command = run.call_args_list[0].args[0]
        self.assertIn("label=ai.sparkbench.loop-worker=true", inspect_command)
        self.assertIn(
            "label=ai.sparkbench.loop-plan=" + "a" * 64,
            inspect_command,
        )
        self.assertEqual(
            run.call_args_list[1].args[0],
            ["/usr/bin/docker", "rm", "-f", "worker-container-id"],
        )

    def test_server_recovery_refuses_identity_outside_frozen_profiles(self) -> None:
        plan = {
            "fingerprint": "b" * 64,
            "models": {"synthetic-halo": {}, "synthetic-rlm": {}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            journal = Journal(Path(temporary) / "journal.jsonl")
            with patch.object(
                loop_campaign,
                "recover_owned_vllm",
                return_value="different_container_present",
            ) as recover:
                with self.assertRaisesRegex(LoopCampaignError, "differently owned"):
                    _recover_plan_servers(plan, journal)

        self.assertEqual(recover.call_count, 2)
        expected = {
            "loop-" + "b" * 12 + "-synthetic-halo",
            "loop-" + "b" * 12 + "-synthetic-rlm",
        }
        self.assertEqual({call.args[0] for call in recover.call_args_list}, expected)

    def test_cleanup_parser_exposes_exec_stop_post_safe_absolute_inputs(self) -> None:
        options = _build_parser().parse_args(
            [
                "cleanup",
                "/var/lib/sparkbench/exact-run",
                "--workspace",
                "/opt/sparkbench/local-llm",
            ]
        )
        self.assertEqual(options.command, "cleanup")
        self.assertTrue(options.run_dir.is_absolute())
        self.assertTrue(options.workspace.is_absolute())


class FrozenPlanTests(unittest.TestCase):
    def test_forced_compaction_plan_is_fingerprinted_and_semantically_strict(
        self,
    ) -> None:
        diagnostic = '''[rlm.compaction_diagnostic]
threshold_pct = 0.20
tasks = ["qa1"]
lengths = ["64k"]
row_indices = [0]

[halo]'''
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest_path = run_dir / "campaign.toml"
            manifest_path.write_text(
                _minimal_campaign_toml().replace("[halo]", diagnostic),
                encoding="utf-8",
            )
            config = load_campaign_manifest(manifest_path)
            forced_case = build_cases(config)[0]
            rlm_config = copy.deepcopy(config["rlm"])
            base: dict[str, object] = {
                "schema_version": PLAN_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "campaign_id": "unit-loop",
                "rlm": rlm_config,
                "rlm_compaction_admission": _rlm_compaction_admission(
                    rlm_config, served_context_tokens=40_960
                ),
                "models": {"synthetic-rlm": {"max_context": 40_960}},
                "halo": {"reasoning_effort": "none"},
                "cases": [forced_case],
            }
            plan = _write_plan(run_dir / "plan.json", base)
            loaded = load_campaign_plan(run_dir)
            self.assertEqual(loaded, plan)
            self.assertEqual(
                loaded["cases"][0]["compaction_threshold_pct"],
                RLM_FORCED_COMPACTION_THRESHOLD_PCT,
            )

            drifted = copy.deepcopy(base)
            drifted_case = drifted["cases"][0]
            assert isinstance(drifted_case, dict)
            drifted_case["compaction_threshold_pct"] = RLM_COMPACTION_THRESHOLD_PCT
            drifted_case["case_id"] = _case_id(
                {
                    key: value
                    for key, value in drifted_case.items()
                    if key != "case_id"
                }
            )
            _write_plan(run_dir / "plan.json", drifted)
            with self.assertRaisesRegex(LoopCampaignError, "case compaction"):
                load_campaign_plan(run_dir)

    def test_round_trip_and_integrity_and_case_identity_rejection(self) -> None:
        raw_case: dict[str, object] = {
            "phase": "halo",
            "treatment": "halo_depth0",
            "reasoning_effort": "none",
            "trace_count": 32,
            "seed": 0,
            "max_depth": 0,
            "max_parallel": 1,
            "max_turns": 4,
            "max_output_tokens": 128,
            "timeout_s": 30,
        }
        case = {**raw_case, "case_id": _case_id(raw_case)}
        rlm_config = {
            "model_profile": "synthetic-rlm",
            "reasoning_control": "fixed_unsupported",
            "compaction": True,
            "compaction_threshold_pct": 0.85,
            "max_output_tokens": 128,
        }
        base: dict[str, object] = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "campaign_id": "unit-loop",
            "rlm": rlm_config,
            "rlm_compaction_admission": _rlm_compaction_admission(
                rlm_config, served_context_tokens=40_960
            ),
            "models": {"synthetic-rlm": {"max_context": 40_960}},
            "halo": {"reasoning_effort": "none"},
            "cases": [case],
        }

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            plan_path = run_dir / "plan.json"
            plan = _write_plan(plan_path, base)
            self.assertEqual(load_campaign_plan(run_dir), plan)

            legacy_base = copy.deepcopy(base)
            legacy_base["schema_version"] = 1
            legacy_base["protocol_version"] = 1
            legacy_plan = _write_plan(plan_path, legacy_base)
            self.assertEqual(load_campaign_plan(run_dir), legacy_plan)

            plan = _write_plan(plan_path, base)
            protocol_tamper = copy.deepcopy(base)
            protocol_tamper["protocol_version"] = 1
            _write_plan(plan_path, protocol_tamper)
            with self.assertRaisesRegex(LoopCampaignError, "protocol version"):
                load_campaign_plan(run_dir)

            plan = _write_plan(plan_path, base)
            integrity_tamper = copy.deepcopy(plan)
            integrity_tamper["campaign_id"] = "tampered"
            plan_path.write_text(json.dumps(integrity_tamper), encoding="utf-8")
            with self.assertRaisesRegex(LoopCampaignError, "integrity check"):
                load_campaign_plan(run_dir)

            case_tamper = copy.deepcopy(base)
            tampered_case = case_tamper["cases"][0]
            assert isinstance(tampered_case, dict)
            tampered_case["seed"] = 99
            _write_plan(plan_path, case_tamper)
            with self.assertRaisesRegex(LoopCampaignError, "case identity"):
                load_campaign_plan(run_dir)

    def test_summary_requires_latest_run_cleanup_and_complete_counters(self) -> None:
        raw_case: dict[str, object] = {
            "phase": "halo",
            "treatment": "halo_depth0",
            "reasoning_effort": "none",
            "trace_count": 32,
            "seed": 0,
            "max_depth": 0,
        }
        case = {**raw_case, "case_id": _case_id(raw_case)}
        rlm_config = {
            "model_profile": "synthetic-rlm",
            "reasoning_control": "fixed_unsupported",
            "compaction": True,
            "compaction_threshold_pct": 0.85,
            "max_output_tokens": 128,
        }
        base: dict[str, object] = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "campaign_id": "unit-loop",
            "rlm": rlm_config,
            "rlm_compaction_admission": _rlm_compaction_admission(
                rlm_config, served_context_tokens=40_960
            ),
            "models": {"synthetic-rlm": {"max_context": 40_960}},
            "halo": {
                "model_profiles": ["synthetic-halo"],
                "reasoning_effort": "none",
            },
            "cases": [case],
        }
        complete = {
            "event": "case_complete",
            "case_id": case["case_id"],
            "phase": "halo",
            "treatment": "halo_depth0",
            "profile_id": "synthetic-halo",
            "wall_s": 2.0,
            "vllm_prompt_tokens": 10.0,
            "vllm_generation_tokens": 4.0,
        }

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            _write_plan(run_dir / "plan.json", base)
            journal_path = run_dir / "journal.jsonl"
            events = [
                {"event": "campaign_started"},
                complete,
                {"event": "campaign_cleanup_failed"},
                {"event": "campaign_resumed"},
            ]
            journal_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

            pending = summarize_campaign(run_dir)
            self.assertEqual(
                pending["status"], "measurements_complete_cleanup_pending"
            )
            group = pending["groups"][0]
            self.assertEqual(group["reasoning_effort"], "none")
            self.assertIsNone(group["reasoning_control"])
            self.assertIsNone(group["cached_prompt_tokens"])
            self.assertIsNone(group["cache_fraction"])

            with journal_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": "campaign_cleanup_verified"}) + "\n")
            verified = summarize_campaign(run_dir)
            self.assertEqual(verified["status"], "complete")

            with journal_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": "campaign_cleanup_failed"}) + "\n")
            failed = summarize_campaign(run_dir)
            self.assertEqual(failed["status"], "cleanup_failed")


if __name__ == "__main__":
    unittest.main()
