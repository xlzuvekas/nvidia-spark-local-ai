"""Offline contract tests for SM121 chunked-prefill studies."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import unittest

from bench.manifest import ManifestError, load_models, load_suite, validate_benchmark_selection
from bench.sglang_sm121_cache_observability import (
    SM121_CACHE_OBSERVABILITY_CACHED_SERIES,
)
from bench.sglang_sm121_chunked_prefill_performance import (
    SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_CHUNK_SIZE,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CASE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_CHUNK_SIZE,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_METRIC_FIELDS,
    SM121_CHUNKED_PREFILL_PERFORMANCE_PAIR_BINDING_SCHEMA_VERSION,
    SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_SUITE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS,
    SM121_CHUNKED_PREFILL_PERFORMANCE_TURN_EVENT,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V1_STUDY,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CANDIDATE_CHUNK_SIZE,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CANDIDATE_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CASE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CONTROL_CHUNK_SIZE,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CONTROL_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V2_STUDY,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V2_SUITE_ID,
    SM121ChunkedPrefillPerformanceError,
    derive_sm121_chunked_prefill_performance_turn_admission,
    score_sm121_chunked_prefill_performance_campaign,
    sm121_chunked_prefill_performance_arm,
    sm121_chunked_prefill_performance_study,
    sm121_chunked_prefill_performance_pair_binding_sha256,
    sm121_chunked_prefill_performance_pair_instance_sha256,
    validate_sm121_chunked_prefill_performance_candidate,
    validate_sm121_chunked_prefill_performance_recorded_turn_event,
    validate_sm121_chunked_prefill_performance_pair,
    validate_sm121_chunked_prefill_performance_pair_binding,
    validate_sm121_chunked_prefill_performance_suite,
    validate_sm121_chunked_prefill_performance_turn_event,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "manifests" / "models.toml"
SUITE = (
    ROOT
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sm121_triton_storage_chunked_prefill_performance_v1.toml"
)
V2_SUITE = (
    ROOT
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sm121_triton_storage_chunked_prefill_performance_v2.toml"
)


def _argument_value(arguments: tuple[str, ...], flag: str) -> str:
    indexes = [index for index, value in enumerate(arguments) if value == flag]
    if len(indexes) != 1 or indexes[0] + 1 >= len(arguments):
        raise AssertionError(f"invalid {flag} placement")
    return arguments[indexes[0] + 1]


def _nonces() -> list[str]:
    return [
        hashlib.sha256(f"prefill-nonce-{index}".encode()).hexdigest()[:32]
        for index in range(4)
    ]


def _turn(
    *,
    ordinal: int,
    arm: str,
    turn: str,
    wall_s: float,
    timed_case_id: str = SM121_CHUNKED_PREFILL_PERFORMANCE_CASE_ID,
) -> dict[str, object]:
    later = turn != "T0"
    prompt_tokens = 58_000 + SM121_CHUNKED_PREFILL_PERFORMANCE_TIMED_TURNS.index(
        turn
    ) * 256
    shared = 0 if not later else prompt_tokens - 256
    event: dict[str, object] = {
        "event": SM121_CHUNKED_PREFILL_PERFORMANCE_TURN_EVENT,
        "arm": arm,
        "lifetime_ordinal": ordinal,
        "case_id": timed_case_id + "--0123456789ab",
        "protocol_case_id": timed_case_id,
        "turn": turn,
        "cache_details_requested": True,
        "prompt_token_ids_requested": True,
        "streaming": False,
        "thinking_disabled": True,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 1,
        "reasoning_tokens": 0,
        "shared_prefix_tokens": shared,
        "append_only_prompt_identity_verified": True,
        "cross_lifetime_prompt_identity_verified": True,
        "response_detail_state": "nonzero_details" if later else "omitted",
        "usage_detail_state": "null",
        "response_device_cached_tokens": shared if later else None,
        "response_host_cached_tokens": 0 if later else None,
        "response_storage_cached_tokens": 0 if later else None,
        "usage_cached_tokens": None,
        "metrics_available": True,
        "guardrail_metrics_available": True,
        "metrics_before_polls": 2,
        "metrics_after_polls": 2,
        "metrics_before_settled": True,
        "metrics_after_settled": True,
        "request_wall_s": wall_s,
        "timed_turn_admitted": True,
        "timed_turn_basis": "admitted",
    }
    for metric in SM121_CHUNKED_PREFILL_PERFORMANCE_METRIC_FIELDS:
        after = 0
        if metric == "prefill_input_tokens":
            after = 1
        if later and metric in {"prefill_device_hit_tokens", "cached_device_tokens"}:
            after = shared
        event[f"before_{metric}"] = 0
        event[f"after_{metric}"] = after
        event[f"delta_{metric}"] = after
    for prefix in ("before", "after"):
        for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
            event[f"{prefix}_cached_{source}_series_present"] = (
                later and prefix == "after" and source == "device"
            )
    return event


def _lifetime(
    ordinal: int,
    arm: str,
    *,
    t0: float,
    later: float,
    timed_case_id: str = SM121_CHUNKED_PREFILL_PERFORMANCE_CASE_ID,
) -> dict[str, object]:
    return {
        "ordinal": ordinal,
        "arm": arm,
        "quality_admitted": True,
        "timed_admitted": True,
        "within_timeout": True,
        "turns": [
            _turn(
                ordinal=ordinal * 2,
                arm=arm,
                turn="T0",
                wall_s=t0,
                timed_case_id=timed_case_id,
            ),
            _turn(
                ordinal=ordinal * 2,
                arm=arm,
                turn="T1",
                wall_s=later / 2,
                timed_case_id=timed_case_id,
            ),
            _turn(
                ordinal=ordinal * 2,
                arm=arm,
                turn="T2",
                wall_s=later / 2,
                timed_case_id=timed_case_id,
            ),
        ],
    }


class SM121ChunkedPrefillPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        models = load_models(MODELS)
        cls.control = models[SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID]
        cls.candidate = models[SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID]
        cls.suite = load_suite(SUITE)
        cls.v2_control = models[
            SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CONTROL_PROFILE_ID
        ]
        cls.v2_candidate = models[
            SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CANDIDATE_PROFILE_ID
        ]
        cls.v2_suite = load_suite(V2_SUITE)

    def test_pair_is_current_cache_on_and_differs_only_by_chunk_size(self) -> None:
        validate_sm121_chunked_prefill_performance_pair(
            self.control, self.candidate
        )
        self.assertEqual("A", sm121_chunked_prefill_performance_arm(self.control))
        self.assertEqual("B", sm121_chunked_prefill_performance_arm(self.candidate))
        self.assertEqual(
            str(SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_CHUNK_SIZE),
            _argument_value(self.control.args, "--chunked-prefill-size"),
        )
        self.assertEqual(
            str(SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_CHUNK_SIZE),
            _argument_value(self.candidate.args, "--chunked-prefill-size"),
        )
        self.assertEqual(("chat",), self.control.tasks)
        self.assertEqual(("chat",), self.candidate.tasks)
        self.assertNotIn("--tool-call-parser", self.control.args)
        self.assertNotIn("--tool-call-parser", self.candidate.args)

    def test_v2_is_a_distinct_2k_4k_study_with_no_cross_study_pairing(self) -> None:
        validate_sm121_chunked_prefill_performance_pair(
            self.v2_control, self.v2_candidate
        )
        validate_sm121_chunked_prefill_performance_suite(self.v2_suite)
        validate_benchmark_selection(self.v2_control, self.v2_suite)
        self.assertEqual(
            SM121_CHUNKED_PREFILL_PERFORMANCE_V2_STUDY,
            sm121_chunked_prefill_performance_study(self.v2_control),
        )
        self.assertEqual(
            SM121_CHUNKED_PREFILL_PERFORMANCE_V2_STUDY,
            sm121_chunked_prefill_performance_study(self.v2_suite.id),
        )
        self.assertEqual(
            SM121_CHUNKED_PREFILL_PERFORMANCE_V2_SUITE_ID, self.v2_suite.id
        )
        self.assertEqual(
            str(SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CONTROL_CHUNK_SIZE),
            _argument_value(self.v2_control.args, "--chunked-prefill-size"),
        )
        self.assertEqual(
            str(SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CANDIDATE_CHUNK_SIZE),
            _argument_value(self.v2_candidate.args, "--chunked-prefill-size"),
        )
        with self.assertRaisesRegex(
            SM121ChunkedPrefillPerformanceError, "candidate profile"
        ):
            validate_sm121_chunked_prefill_performance_pair(
                self.control, self.v2_candidate
            )
        with self.assertRaisesRegex(ManifestError, "requires"):
            validate_benchmark_selection(self.control, self.v2_suite)

    def test_candidate_rejects_any_other_serving_delta(self) -> None:
        arguments = list(self.candidate.args)
        index = arguments.index("--max-running-requests")
        arguments[index + 1] = "2"
        drifted = replace(self.candidate, args=tuple(arguments))
        with self.assertRaisesRegex(
            SM121ChunkedPrefillPerformanceError, "profile args changed"
        ):
            validate_sm121_chunked_prefill_performance_candidate(drifted)

    def test_pair_rejects_same_chunk_size(self) -> None:
        arguments = list(self.candidate.args)
        index = arguments.index("--chunked-prefill-size")
        arguments[index + 1] = str(
            SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_CHUNK_SIZE
        )
        drifted = replace(self.candidate, args=tuple(arguments))
        with self.assertRaisesRegex(
            SM121ChunkedPrefillPerformanceError, "profile args changed"
        ):
            validate_sm121_chunked_prefill_performance_pair(self.control, drifted)

    def test_suite_is_exact_and_profiles_cannot_select_other_suites(self) -> None:
        validate_sm121_chunked_prefill_performance_suite(self.suite)
        validate_benchmark_selection(self.control, self.suite)
        self.assertEqual(
            SM121_CHUNKED_PREFILL_PERFORMANCE_SUITE_ID, self.suite.id
        )
        self.assertEqual(
            (
                SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID,
                SM121_CHUNKED_PREFILL_PERFORMANCE_CASE_ID,
            ),
            tuple(case.id for case in self.suite.cases),
        )
        unrelated = load_suite(ROOT / "manifests" / "suites" / "smoke.toml")
        with self.assertRaisesRegex(ManifestError, "requires"):
            validate_benchmark_selection(self.control, unrelated)

    def test_suite_rejects_a_shape_rewrite(self) -> None:
        changed = replace(
            self.suite,
            cases=(
                self.suite.cases[0],
                replace(self.suite.cases[1], max_output_tokens=64),
            ),
        )
        with self.assertRaisesRegex(
            SM121ChunkedPrefillPerformanceError, "max_output_tokens"
        ):
            validate_sm121_chunked_prefill_performance_suite(changed)

    def test_turn_contract_requires_a_cold_t0_and_cached_appends(self) -> None:
        t0 = _turn(ordinal=2, arm="A", turn="T0", wall_s=20.0)
        later = _turn(ordinal=2, arm="A", turn="T1", wall_s=2.0)
        validate_sm121_chunked_prefill_performance_turn_event(t0)
        self.assertEqual(
            (True, "admitted"),
            derive_sm121_chunked_prefill_performance_turn_admission(t0),
        )
        validate_sm121_chunked_prefill_performance_turn_event(later)
        self.assertEqual(
            (True, "admitted"),
            derive_sm121_chunked_prefill_performance_turn_admission(later),
        )
        t0["after_cached_total_tokens"] = 1
        t0["delta_cached_total_tokens"] = 1
        with self.assertRaisesRegex(
            SM121ChunkedPrefillPerformanceError, "admission changed"
        ):
            validate_sm121_chunked_prefill_performance_turn_event(t0)

    def test_t0_allows_bootstrap_prefill_but_not_prior_cache_residency(self) -> None:
        t0 = _turn(ordinal=2, arm="A", turn="T0", wall_s=20.0)
        # A fresh SGLang lifetime can increment its global input counter while
        # becoming ready.  That does not make the controller's first request
        # cache-warm when all cache counters remain zero.
        t0["before_prefill_input_tokens"] = 64
        t0["after_prefill_input_tokens"] = 65
        t0["delta_prefill_input_tokens"] = 1
        self.assertEqual(
            (True, "admitted"),
            derive_sm121_chunked_prefill_performance_turn_admission(t0),
        )
        validate_sm121_chunked_prefill_performance_turn_event(t0)

        t0["before_cached_total_tokens"] = 1
        t0["after_cached_total_tokens"] = 1
        t0["delta_cached_total_tokens"] = 0
        with self.assertRaisesRegex(
            SM121ChunkedPrefillPerformanceError, "admission changed"
        ):
            validate_sm121_chunked_prefill_performance_turn_event(t0)

    def test_only_the_audited_legacy_bootstrap_partial_is_readable(self) -> None:
        legacy = _turn(ordinal=2, arm="A", turn="T0", wall_s=20.0)
        legacy["before_prefill_input_tokens"] = 64
        legacy["after_prefill_input_tokens"] = 65
        legacy["delta_prefill_input_tokens"] = 1
        legacy["timed_turn_admitted"] = False
        legacy["timed_turn_basis"] = "cold_lifetime"
        with self.assertRaisesRegex(
            SM121ChunkedPrefillPerformanceError, "admission changed"
        ):
            validate_sm121_chunked_prefill_performance_turn_event(legacy)
        validate_sm121_chunked_prefill_performance_recorded_turn_event(legacy)

        legacy["before_cached_total_tokens"] = 1
        legacy["after_cached_total_tokens"] = 1
        legacy["delta_cached_total_tokens"] = 0
        legacy["timed_turn_admitted"] = True
        legacy["timed_turn_basis"] = "admitted"
        with self.assertRaisesRegex(
            SM121ChunkedPrefillPerformanceError, "admission changed"
        ):
            validate_sm121_chunked_prefill_performance_recorded_turn_event(legacy)

    def test_score_retains_2k_only_when_t0_and_guardrails_pass(self) -> None:
        lifetimes = [
            _lifetime(1, "A", t0=100.0, later=20.0),
            _lifetime(2, "B", t0=95.0, later=21.0),
            _lifetime(3, "B", t0=95.0, later=21.0),
            _lifetime(4, "A", t0=100.0, later=20.0),
        ]
        result = score_sm121_chunked_prefill_performance_campaign(lifetimes)
        self.assertEqual("complete", result.status)
        self.assertEqual("retain_b", result.decision)
        self.assertEqual(0.95, result.candidate_t0_wall_ratio)
        self.assertEqual(1.05, result.candidate_later_wall_ratio)

        guardrail = [
            _lifetime(1, "A", t0=100.0, later=20.0),
            _lifetime(2, "B", t0=95.0, later=21.1),
            _lifetime(3, "B", t0=95.0, later=21.1),
            _lifetime(4, "A", t0=100.0, later=20.0),
        ]
        self.assertEqual(
            "guardrail_reject",
            score_sm121_chunked_prefill_performance_campaign(guardrail).decision,
        )

    def test_score_does_not_promote_an_incomplete_campaign(self) -> None:
        lifetimes = [
            _lifetime(1, "A", t0=100.0, later=20.0),
            _lifetime(2, "B", t0=95.0, later=21.0),
            _lifetime(3, "B", t0=95.0, later=21.0),
            _lifetime(4, "A", t0=100.0, later=20.0),
        ]
        for lifetime in lifetimes[2:]:
            lifetime.update(
                quality_admitted=False,
                timed_admitted=False,
                within_timeout=False,
                turns=[],
            )
        result = score_sm121_chunked_prefill_performance_campaign(lifetimes)
        self.assertEqual("partial", result.status)
        self.assertEqual("not_evaluated", result.decision)

    def test_v2_score_requires_its_own_timed_case(self) -> None:
        lifetimes = [
            _lifetime(
                1,
                "A",
                t0=100.0,
                later=20.0,
                timed_case_id=SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CASE_ID,
            ),
            _lifetime(
                2,
                "B",
                t0=90.0,
                later=20.0,
                timed_case_id=SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CASE_ID,
            ),
            _lifetime(
                3,
                "B",
                t0=90.0,
                later=20.0,
                timed_case_id=SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CASE_ID,
            ),
            _lifetime(
                4,
                "A",
                t0=100.0,
                later=20.0,
                timed_case_id=SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CASE_ID,
            ),
        ]
        self.assertEqual(
            "retain_b",
            score_sm121_chunked_prefill_performance_campaign(
                lifetimes, study=SM121_CHUNKED_PREFILL_PERFORMANCE_V2_STUDY
            ).decision,
        )
        with self.assertRaisesRegex(
            SM121ChunkedPrefillPerformanceError, "turn topology"
        ):
            score_sm121_chunked_prefill_performance_campaign(
                lifetimes, study=SM121_CHUNKED_PREFILL_PERFORMANCE_V1_STUDY
            )

    def test_pair_binding_commits_the_only_two_chunk_sizes(self) -> None:
        binding: dict[str, object] = {
            "schema_version": SM121_CHUNKED_PREFILL_PERFORMANCE_PAIR_BINDING_SCHEMA_VERSION,
            "suite_id": SM121_CHUNKED_PREFILL_PERFORMANCE_SUITE_ID,
            "execution_mode": "sm121_storage_chunked_prefill_performance_abba_fresh_lifetimes",
            "arm_order": list(SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER),
            "profile_ids": [
                SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID,
                SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID,
            ],
            "chunked_prefill_sizes": [1024, 2048],
            "quality_case_id": SM121_CHUNKED_PREFILL_PERFORMANCE_QUALITY_CASE_ID,
            "timed_case_id": SM121_CHUNKED_PREFILL_PERFORMANCE_CASE_ID,
            "cell_timeout_s": 1200,
            "campaign_instance_sha256": sm121_chunked_prefill_performance_pair_instance_sha256(
                _nonces()
            ),
            "plan_fingerprints": ["0" * 16, "1" * 16, "2" * 16, "3" * 16],
        }
        binding["pair_binding_sha256"] = (
            sm121_chunked_prefill_performance_pair_binding_sha256(binding)
        )
        validate_sm121_chunked_prefill_performance_pair_binding(binding)
        binding["chunked_prefill_sizes"] = [1024, 4096]
        with self.assertRaisesRegex(
            SM121ChunkedPrefillPerformanceError, "size binding"
        ):
            validate_sm121_chunked_prefill_performance_pair_binding(binding)


if __name__ == "__main__":
    unittest.main()
