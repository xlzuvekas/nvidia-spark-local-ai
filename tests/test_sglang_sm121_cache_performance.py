"""Offline contracts for the SM121 cache-policy wall-time campaign."""

from __future__ import annotations

import hashlib
import unittest

from bench.sglang_sm121_cache_observability import (
    SM121_CACHE_OBSERVABILITY_CACHED_SERIES,
)
from bench.sglang_sm121_cache_performance import (
    SM121_CACHE_PERFORMANCE_ARM_ORDER,
    SM121_CACHE_PERFORMANCE_CACHE_ON_ARM,
    SM121_CACHE_PERFORMANCE_CASE_ID,
    SM121_CACHE_PERFORMANCE_METRIC_FIELDS,
    SM121_CACHE_PERFORMANCE_PAIR_BINDING_SCHEMA_VERSION,
    SM121_CACHE_PERFORMANCE_SCHEMA_VERSION,
    SM121_CACHE_PERFORMANCE_SUITE_ID,
    SM121_CACHE_PERFORMANCE_TIMED_TURNS,
    SM121_CACHE_PERFORMANCE_TURN_EVENT,
    SM121CachePerformanceError,
    derive_sm121_cache_performance_turn_admission,
    score_sm121_cache_performance_campaign,
    sm121_cache_performance_pair_binding_sha256,
    sm121_cache_performance_pair_instance_sha256,
    validate_sm121_cache_performance_pair_binding,
    validate_sm121_cache_performance_turn_event,
)


def _nonces() -> list[str]:
    return [hashlib.sha256(f"nonce-{index}".encode()).hexdigest()[:32] for index in range(4)]


def _turn(
    *, ordinal: int, arm: str, turn: str, wall_s: float
) -> dict[str, object]:
    later = turn != "T0"
    positive = arm == SM121_CACHE_PERFORMANCE_CACHE_ON_ARM and later
    prompt_tokens = 32_768 + SM121_CACHE_PERFORMANCE_TIMED_TURNS.index(turn) * 256
    event: dict[str, object] = {
        "event": SM121_CACHE_PERFORMANCE_TURN_EVENT,
        "arm": arm,
        "lifetime_ordinal": ordinal,
        "case_id": SM121_CACHE_PERFORMANCE_CASE_ID + "--0123456789ab",
        "protocol_case_id": SM121_CACHE_PERFORMANCE_CASE_ID,
        "turn": turn,
        "cache_details_requested": True,
        "prompt_token_ids_requested": True,
        "streaming": False,
        "thinking_disabled": True,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 1,
        "reasoning_tokens": 0,
        "shared_prefix_tokens": 0 if not later else prompt_tokens - 256,
        "append_only_prompt_identity_verified": True,
        "cross_lifetime_prompt_identity_verified": True,
        "response_detail_state": "nonzero_details" if positive else "omitted",
        "usage_detail_state": "null",
        "response_device_cached_tokens": prompt_tokens - 256 if positive else None,
        "response_host_cached_tokens": 0 if positive else None,
        "response_storage_cached_tokens": 0 if positive else None,
        "usage_cached_tokens": None,
        "metrics_available": True,
        "guardrail_metrics_available": True,
        "metrics_before_polls": 2,
        "metrics_after_polls": 2,
        "metrics_before_settled": True,
        "metrics_after_settled": True,
        "request_wall_s": wall_s,
    }
    for metric in SM121_CACHE_PERFORMANCE_METRIC_FIELDS:
        before = 0
        after = 0
        if metric == "prefill_input_tokens":
            after = 1
        if positive and metric in {"prefill_device_hit_tokens", "cached_device_tokens"}:
            after = prompt_tokens - 256
        event[f"before_{metric}"] = before
        event[f"after_{metric}"] = after
        event[f"delta_{metric}"] = after - before
    for prefix in ("before", "after"):
        for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
            event[f"{prefix}_cached_{source}_series_present"] = (
                positive and prefix == "after" and source == "device"
            )
    event["timed_turn_admitted"] = True
    event["timed_turn_basis"] = "admitted"
    return event


def _lifetime(
    ordinal: int, arm: str, *, t0: float, later: float
) -> dict[str, object]:
    return {
        "ordinal": ordinal,
        "arm": arm,
        "quality_admitted": True,
        "timed_admitted": True,
        "within_timeout": True,
        "turns": [
            _turn(ordinal=ordinal * 2, arm=arm, turn="T0", wall_s=t0),
            _turn(ordinal=ordinal * 2, arm=arm, turn="T1", wall_s=later / 2),
            _turn(ordinal=ordinal * 2, arm=arm, turn="T2", wall_s=later / 2),
        ],
    }


class SM121CachePerformanceContractTests(unittest.TestCase):
    def test_turn_admission_accepts_current_omitted_zero_details(self) -> None:
        event = _turn(ordinal=4, arm="B", turn="T0", wall_s=30.0)
        validate_sm121_cache_performance_turn_event(event)

    def test_turn_rejects_synthetic_ttft_field(self) -> None:
        event = _turn(ordinal=4, arm="B", turn="T0", wall_s=30.0)
        event["ttft_s"] = 1.0
        with self.assertRaisesRegex(SM121CachePerformanceError, "fields are invalid"):
            validate_sm121_cache_performance_turn_event(event)

    def test_turn_accepts_unexpected_null_cache_detail_shape(self) -> None:
        event = _turn(ordinal=2, arm="A", turn="T0", wall_s=30.0)
        event.update(
            {
                "response_detail_state": "unexpected",
                "response_device_cached_tokens": None,
                "response_host_cached_tokens": None,
                "response_storage_cached_tokens": None,
                "usage_detail_state": "unexpected",
                "usage_cached_tokens": None,
            }
        )
        validate_sm121_cache_performance_turn_event(event)
        self.assertEqual(
            (True, "admitted"),
            derive_sm121_cache_performance_turn_admission(event),
        )

    def test_turn_rejects_nonzero_cached_total_for_cache_off(self) -> None:
        event = _turn(ordinal=4, arm="B", turn="T1", wall_s=30.0)
        event["after_cached_total_tokens"] = 1
        event["delta_cached_total_tokens"] = 1
        with self.assertRaisesRegex(SM121CachePerformanceError, "admission changed"):
            validate_sm121_cache_performance_turn_event(event)

    def test_turn_rejects_mismatched_positive_usage_cache_detail(self) -> None:
        event = _turn(ordinal=2, arm="A", turn="T1", wall_s=30.0)
        event["usage_detail_state"] = "nonzero_details"
        event["usage_cached_tokens"] = 1
        with self.assertRaisesRegex(SM121CachePerformanceError, "admission changed"):
            validate_sm121_cache_performance_turn_event(event)

    def test_score_retains_b_at_inclusive_threshold(self) -> None:
        lifetimes = [
            _lifetime(1, "A", t0=20.0, later=100.0),
            _lifetime(2, "B", t0=20.0, later=95.0),
            _lifetime(3, "B", t0=20.0, later=95.0),
            _lifetime(4, "A", t0=20.0, later=100.0),
        ]
        result = score_sm121_cache_performance_campaign(lifetimes)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.decision, "retain_b")
        self.assertEqual(result.winner_later_wall_ratio, 0.95)

    def test_score_preserves_partial_without_a_decision(self) -> None:
        lifetimes = [
            _lifetime(1, "A", t0=20.0, later=100.0),
            _lifetime(2, "B", t0=20.0, later=95.0),
            _lifetime(3, "B", t0=20.0, later=95.0),
            _lifetime(4, "A", t0=20.0, later=100.0),
        ]
        for lifetime in lifetimes[2:]:
            lifetime["quality_admitted"] = False
            lifetime["timed_admitted"] = False
            lifetime["within_timeout"] = False
            lifetime["turns"] = []
        result = score_sm121_cache_performance_campaign(lifetimes)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.decision, "not_evaluated")

    def test_score_rejects_swapped_matching_arm_lifetime(self) -> None:
        lifetimes = [
            _lifetime(1, "A", t0=20.0, later=100.0),
            _lifetime(2, "B", t0=20.0, later=95.0),
            _lifetime(3, "B", t0=20.0, later=95.0),
            _lifetime(4, "A", t0=20.0, later=100.0),
        ]
        for event in lifetimes[3]["turns"]:
            event["lifetime_ordinal"] = 2
        with self.assertRaisesRegex(SM121CachePerformanceError, "turn topology"):
            score_sm121_cache_performance_campaign(lifetimes)

    def test_partial_summary_rejects_untyped_turn_payload(self) -> None:
        lifetimes = [
            _lifetime(1, "A", t0=20.0, later=100.0),
            _lifetime(2, "B", t0=20.0, later=95.0),
            _lifetime(3, "B", t0=20.0, later=95.0),
            _lifetime(4, "A", t0=20.0, later=100.0),
        ]
        lifetimes[1]["timed_admitted"] = False
        lifetimes[1]["turns"] = [{"private": "must-not-publish"}]
        for lifetime in lifetimes[2:]:
            lifetime["quality_admitted"] = False
            lifetime["timed_admitted"] = False
            lifetime["within_timeout"] = False
            lifetime["turns"] = []
        with self.assertRaisesRegex(SM121CachePerformanceError, "turn event fields"):
            score_sm121_cache_performance_campaign(lifetimes)

    def test_pair_binding_is_opaque_and_exact(self) -> None:
        instance = sm121_cache_performance_pair_instance_sha256(_nonces())
        binding: dict[str, object] = {
            "schema_version": SM121_CACHE_PERFORMANCE_PAIR_BINDING_SCHEMA_VERSION,
            "suite_id": SM121_CACHE_PERFORMANCE_SUITE_ID,
            "execution_mode": "sm121_storage_cache_policy_performance_abba_fresh_lifetimes",
            "arm_order": list(SM121_CACHE_PERFORMANCE_ARM_ORDER),
            "profile_ids": [
                "qwen38-flash-next-nvfp4-sm121-triton-storage-cache-performance-on-sglang",
                "qwen38-flash-next-nvfp4-sm121-triton-storage-cache-performance-off-sglang",
            ],
            "quality_case_id": "synthetic-exact-answer-v2",
            "timed_case_id": SM121_CACHE_PERFORMANCE_CASE_ID,
            "cell_timeout_s": 1200,
            "campaign_instance_sha256": instance,
            "plan_fingerprints": ["0" * 16, "1" * 16, "2" * 16, "3" * 16],
        }
        binding["pair_binding_sha256"] = sm121_cache_performance_pair_binding_sha256(binding)
        validate_sm121_cache_performance_pair_binding(binding)
        binding["arm_order"] = ["B", "A", "B", "A"]
        with self.assertRaisesRegex(SM121CachePerformanceError, "arm order"):
            validate_sm121_cache_performance_pair_binding(binding)


if __name__ == "__main__":
    unittest.main()
