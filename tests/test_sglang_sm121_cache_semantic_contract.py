"""Offline manifest contracts for the paired SM121 cache-policy semantic lane."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

from bench.manifest import ManifestError, load_models, load_suite, validate_benchmark_selection
from bench.sglang_sm121_cache_semantic import (
    SM121_CACHE_SEMANTIC_ARM_ORDER,
    SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
    SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID,
    SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
    SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID,
    SM121_CACHE_SEMANTIC_CASE_ID,
    SM121_CACHE_SEMANTIC_COLD_INPUT_MAX_TOKENS,
    SM121_CACHE_SEMANTIC_COLD_INPUT_MIN_TOKENS,
    SM121_CACHE_SEMANTIC_EXECUTION_MODE,
    SM121_CACHE_SEMANTIC_LOCAL_LIFETIME_ORDER,
    SM121_CACHE_SEMANTIC_MAX_MAMBA_CACHE_SIZE,
    SM121_CACHE_SEMANTIC_METRIC_FIELDS,
    SM121_CACHE_SEMANTIC_PAIR_BINDING_FIELDS,
    SM121_CACHE_SEMANTIC_PAIR_BINDING_SCHEMA_VERSION,
    SM121_CACHE_SEMANTIC_PROFILE_ORDER,
    SM121_CACHE_SEMANTIC_QUALITY_CASE_ID,
    SM121_CACHE_SEMANTIC_RUNTIME_ATTESTATION_EVENT,
    SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED,
    SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
    SM121_CACHE_SEMANTIC_STATIC_ATTESTATION_EVENT,
    SM121_CACHE_SEMANTIC_SUITE_ID,
    SM121_CACHE_SEMANTIC_TURN_OBSERVATION_EVENT,
    SM121_CACHE_SEMANTIC_TURN_ORDER,
    SM121CacheSemanticError,
    derive_sm121_cache_semantic_turn_admission,
    expected_sm121_cache_semantic_event_counts,
    is_sm121_cache_semantic_plan,
    sm121_cache_semantic_arm,
    sm121_cache_semantic_cache_off_receipt_sha256,
    sm121_cache_semantic_case_metadata,
    sm121_cache_semantic_lifecycle_issues,
    sm121_cache_semantic_pair_binding_sha256,
    sm121_cache_semantic_pair_instance_sha256,
    sm121_cache_semantic_runtime_expected,
    sm121_cache_semantic_turn_issues,
    validate_sm121_cache_semantic_candidate,
    validate_sm121_cache_semantic_pair,
    validate_sm121_cache_semantic_pair_binding,
    validate_sm121_cache_semantic_runtime_attestation_event,
    validate_sm121_cache_semantic_static_attestation_event,
    validate_sm121_cache_semantic_suite,
    validate_sm121_cache_semantic_turn_event,
)
from bench.sglang_sm121_cache_observability import (
    SM121_CACHE_OBSERVABILITY_CACHED_SERIES,
    SM121_CACHE_SOURCE_DIGESTS,
)
from bench.sglang_sm121_storage import SM121_STORAGE_PROFILE_ID
from bench.sglang_sm121_storage import SM121_STORAGE_SOURCE_TREE


ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = ROOT / "manifests" / "models.toml"
SUITE_PATH = (
    ROOT
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sm121_triton_storage_cache_policy_semantic_canary.toml"
)


def _static_event(arm: str, lifetime: int) -> dict[str, object]:
    return {
        "event": SM121_CACHE_SEMANTIC_STATIC_ATTESTATION_EVENT,
        "arm": arm,
        "fresh_server_lifetime": lifetime,
        "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
        **SM121_CACHE_SOURCE_DIGESTS,
        **SM121_CACHE_SEMANTIC_STATIC_ASSERTIONS,
    }


def _runtime_event(arm: str, lifetime: int) -> dict[str, object]:
    return {
        "event": SM121_CACHE_SEMANTIC_RUNTIME_ATTESTATION_EVENT,
        "arm": arm,
        "fresh_server_lifetime": lifetime,
        **SM121_CACHE_SEMANTIC_RUNTIME_EXPECTED[arm],
        "mamba_radix_cache_strategy": "extra_buffer_lazy",
        "max_mamba_cache_size": SM121_CACHE_SEMANTIC_MAX_MAMBA_CACHE_SIZE,
    }


def _turn_event(
    arm: str,
    turn: str,
    *,
    case_id: str = f"{SM121_CACHE_SEMANTIC_CASE_ID}--abcdef123456",
    attempt_id: str = "semantic-attempt",
) -> dict[str, object]:
    """Build one complete scalar event and derive its persisted admission."""

    turn_index = SM121_CACHE_SEMANTIC_TURN_ORDER.index(turn)
    prompt_tokens = (32_768, 33_024, 33_280)[turn_index]
    shared_prefix_tokens = (0, 32_768, 33_024)[turn_index]
    positive_device = arm == SM121_CACHE_SEMANTIC_CACHE_ON_ARM and turn != "T0"
    response_state = "nonzero_details" if positive_device else "zero_details"
    response_device = shared_prefix_tokens if positive_device else 0
    before = {metric: 0 for metric in SM121_CACHE_SEMANTIC_METRIC_FIELDS}
    after = dict(before)
    after["prefill_input_tokens"] = prompt_tokens
    if positive_device:
        after["prefill_device_hit_tokens"] = response_device
        after["cached_total_tokens"] = response_device
        after["cached_device_tokens"] = response_device
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
        "response_detail_state": response_state,
        "usage_detail_state": response_state,
        "response_device_cached_tokens": response_device,
        "response_host_cached_tokens": 0,
        "response_storage_cached_tokens": 0,
        "usage_cached_tokens": response_device,
        "metrics_available": True,
        "guardrail_metrics_available": True,
        "metrics_before_polls": 2,
        "metrics_after_polls": 2,
        "metrics_before_settled": True,
        "metrics_after_settled": True,
    }
    for metric in SM121_CACHE_SEMANTIC_METRIC_FIELDS:
        event[f"before_{metric}"] = before[metric]
        event[f"after_{metric}"] = after[metric]
        event[f"delta_{metric}"] = after[metric] - before[metric]
    for prefix in ("before", "after"):
        for source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
            event[f"{prefix}_cached_{source}_series_present"] = False
    admitted, basis = derive_sm121_cache_semantic_turn_admission(event)
    event["semantic_turn_admitted"] = admitted
    event["semantic_turn_basis"] = basis
    return event


def _lifecycle_events(
    arm: str, *, partial_semantic_case: bool = False
) -> tuple[list[dict[str, object]], tuple[str, str]]:
    quality_case_id = f"{SM121_CACHE_SEMANTIC_QUALITY_CASE_ID}--0123456789ab"
    semantic_case_id = f"{SM121_CACHE_SEMANTIC_CASE_ID}--abcdef123456"
    turns = [
        _turn_event(arm, turn, case_id=semantic_case_id)
        for turn in SM121_CACHE_SEMANTIC_TURN_ORDER
    ]
    if partial_semantic_case:
        partial = turns[1]
        partial.update(
            {
                "response_detail_state": "unexpected",
                "usage_detail_state": "unexpected",
                "response_device_cached_tokens": None,
                "response_host_cached_tokens": None,
                "response_storage_cached_tokens": None,
                "usage_cached_tokens": None,
            }
        )
        admitted, basis = derive_sm121_cache_semantic_turn_admission(partial)
        partial["semantic_turn_admitted"] = admitted
        partial["semantic_turn_basis"] = basis
    events: list[dict[str, object]] = [
        {
            "event": "run_start",
            "execution_mode": SM121_CACHE_SEMANTIC_EXECUTION_MODE,
        },
        {"event": "measurement_started"},
        _static_event(arm, 1),
        _runtime_event(arm, 1),
        {
            "event": "server_ready",
            "backend": "sglang",
            "fresh_server_lifetime": 1,
            "first_inference_is_case": True,
            "case_id": quality_case_id,
        },
        {
            "event": "case_start",
            "case_id": quality_case_id,
            "attempt_id": "quality-attempt",
            "kind": "quality",
        },
    ]
    events.extend(
        {
            "event": "request_complete",
            "case_id": quality_case_id,
            "attempt_id": "quality-attempt",
            "kind": "quality",
            "result": {},
        }
        for _ in range(4)
    )
    events.extend(
        [
            {
                "event": "case_complete",
                "case_id": quality_case_id,
                "attempt_id": "quality-attempt",
                "kind": "quality",
                "validation_passed": True,
            },
            {"event": "server_stopped", "backend": "sglang", "fresh_server_lifetime": 1},
            _static_event(arm, 2),
            _runtime_event(arm, 2),
            {
                "event": "server_ready",
                "backend": "sglang",
                "fresh_server_lifetime": 2,
                "first_inference_is_case": True,
                "case_id": semantic_case_id,
            },
            {
                "event": "case_start",
                "case_id": semantic_case_id,
                "attempt_id": "semantic-attempt",
                "kind": "capability",
            },
        ]
    )
    for turn in turns:
        events.append(turn)
        events.append(
            {
                "event": "request_complete",
                "case_id": semantic_case_id,
                "attempt_id": "semantic-attempt",
                "kind": "capability",
                "result": {},
            }
        )
    events.extend(
        [
            {
                "event": "case_complete",
                "case_id": semantic_case_id,
                "attempt_id": "semantic-attempt",
                "kind": "capability",
                "validation_passed": all(
                    turn["semantic_turn_admitted"] for turn in turns
                ),
            },
            {"event": "server_stopped", "backend": "sglang", "fresh_server_lifetime": 2},
            {"event": "measurement_complete"},
            {"event": "run_complete", "status": "completed"},
        ]
    )
    return events, (quality_case_id, semantic_case_id)


def _pair_binding(
    arm: str, peer_plan_fingerprint: str, pair_instance_sha256: str
) -> dict[str, object]:
    binding: dict[str, object] = {
        "schema_version": SM121_CACHE_SEMANTIC_PAIR_BINDING_SCHEMA_VERSION,
        "suite_id": SM121_CACHE_SEMANTIC_SUITE_ID,
        "execution_mode": SM121_CACHE_SEMANTIC_EXECUTION_MODE,
        "arm": arm,
        "profile_id": (
            SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID
            if arm == SM121_CACHE_SEMANTIC_CACHE_OFF_ARM
            else SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID
        ),
        "arm_order": list(SM121_CACHE_SEMANTIC_ARM_ORDER),
        "local_lifetime_order": list(SM121_CACHE_SEMANTIC_LOCAL_LIFETIME_ORDER),
        "quality_case_id": SM121_CACHE_SEMANTIC_QUALITY_CASE_ID,
        "semantic_case_id": SM121_CACHE_SEMANTIC_CASE_ID,
        "semantic_case_metadata": sm121_cache_semantic_case_metadata(),
        "peer_plan_fingerprint": peer_plan_fingerprint,
        "pair_instance_sha256": pair_instance_sha256,
    }
    binding["pair_binding_sha256"] = sm121_cache_semantic_pair_binding_sha256(
        binding
    )
    return binding


class SM121CacheSemanticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = load_models(MODELS_PATH)
        cls.suite = load_suite(SUITE_PATH)
        cls.cache_off = cls.models[SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID]
        cls.cache_on = cls.models[SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID]

    def test_pair_is_exact_and_selection_is_scoped(self) -> None:
        validate_sm121_cache_semantic_candidate(self.cache_off)
        validate_sm121_cache_semantic_candidate(self.cache_on)
        validate_sm121_cache_semantic_pair(self.cache_off, self.cache_on)
        validate_benchmark_selection(self.cache_off, self.suite)
        validate_benchmark_selection(self.cache_on, self.suite)
        self.assertTrue(is_sm121_cache_semantic_plan(self.cache_off, self.suite))
        self.assertTrue(is_sm121_cache_semantic_plan(self.cache_on, self.suite))
        self.assertEqual(
            SM121_CACHE_SEMANTIC_PROFILE_ORDER,
            (
                SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID,
                SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID,
            ),
        )
        self.assertEqual(
            SM121_CACHE_SEMANTIC_ARM_ORDER,
            (SM121_CACHE_SEMANTIC_CACHE_OFF_ARM, SM121_CACHE_SEMANTIC_CACHE_ON_ARM),
        )
        with self.assertRaisesRegex(ManifestError, "requires"):
            validate_benchmark_selection(
                self.models[SM121_STORAGE_PROFILE_ID], self.suite
            )

    def test_cache_off_adds_only_the_disable_radix_flag(self) -> None:
        off_args = self.cache_off.args
        on_args = self.cache_on.args
        self.assertEqual(off_args.count("--disable-radix-cache"), 1)
        self.assertNotIn("--disable-radix-cache", on_args)
        self.assertEqual(
            tuple(argument for argument in off_args if argument != "--disable-radix-cache"),
            on_args,
        )
        self.assertIn("--mamba-radix-cache-strategy", on_args)
        strategy_index = on_args.index("--mamba-radix-cache-strategy")
        self.assertEqual(on_args[strategy_index + 1], "extra_buffer_lazy")
        cache_size_index = on_args.index("--max-mamba-cache-size")
        self.assertEqual(on_args[cache_size_index + 1], "4")

    def test_runtime_identity_is_arm_specific_and_detached(self) -> None:
        self.assertEqual(sm121_cache_semantic_arm(self.cache_off), "B")
        self.assertEqual(sm121_cache_semantic_arm(self.cache_on), "A")
        cache_off = sm121_cache_semantic_runtime_expected(self.cache_off)
        cache_on = sm121_cache_semantic_runtime_expected(self.cache_on)
        self.assertEqual(cache_off["cache_impl"], "ChunkCache")
        self.assertTrue(cache_off["disable_radix_cache"])
        self.assertFalse(cache_off["mamba_extra_buffer_enabled"])
        self.assertFalse(cache_off["mamba_extra_buffer_lazy_enabled"])
        self.assertEqual(cache_on["cache_impl"], "UnifiedRadixCache")
        self.assertFalse(cache_on["disable_radix_cache"])
        self.assertTrue(cache_on["hybrid_ssm"])
        self.assertTrue(cache_on["mamba_extra_buffer_enabled"])
        self.assertTrue(cache_on["mamba_extra_buffer_lazy_enabled"])
        cache_off["cache_impl"] = "mutated"
        self.assertEqual(
            sm121_cache_semantic_runtime_expected(self.cache_off)["cache_impl"],
            "ChunkCache",
        )

    def test_semantic_metadata_has_no_timing_claim(self) -> None:
        metadata = sm121_cache_semantic_case_metadata()
        self.assertEqual(metadata["case_id"], SM121_CACHE_SEMANTIC_CASE_ID)
        self.assertEqual(metadata["turn_order"], list(SM121_CACHE_SEMANTIC_TURN_ORDER))
        self.assertEqual(
            metadata["cold_input_min_tokens"], SM121_CACHE_SEMANTIC_COLD_INPUT_MIN_TOKENS
        )
        self.assertEqual(
            metadata["cold_input_max_tokens"], SM121_CACHE_SEMANTIC_COLD_INPUT_MAX_TOKENS
        )
        self.assertEqual(metadata["measurement"], "semantic_only")
        self.assertEqual(metadata["timing_claims"], "forbidden")
        self.assertTrue(metadata["later_turns_require_shared_prefix"])
        self.assertNotIn("wall_s", metadata)
        self.assertNotIn("tps", metadata)

    def test_tampering_fails_closed(self) -> None:
        with self.assertRaisesRegex(SM121CacheSemanticError, "cache-off profile ID"):
            validate_sm121_cache_semantic_pair(self.cache_on, self.cache_off)
        wrong_cache_size = list(self.cache_on.args)
        wrong_cache_size[wrong_cache_size.index("4")] = "5"
        with self.assertRaisesRegex(SM121CacheSemanticError, "args"):
            validate_sm121_cache_semantic_candidate(
                replace(self.cache_on, args=tuple(wrong_cache_size))
            )
        reordered = replace(self.suite, cases=tuple(reversed(self.suite.cases)))
        with self.assertRaisesRegex(SM121CacheSemanticError, "suite case 0"):
            validate_sm121_cache_semantic_suite(reordered)
        with self.assertRaisesRegex(ManifestError, "requires"):
            validate_benchmark_selection(
                self.cache_off,
                replace(self.suite, id="unrelated-suite"),
            )

    def test_loaded_suite_identity_is_exact(self) -> None:
        self.assertEqual(self.suite.id, SM121_CACHE_SEMANTIC_SUITE_ID)
        validate_sm121_cache_semantic_suite(self.suite)

    def test_scalar_attestations_and_turn_admission_are_arm_specific(self) -> None:
        for arm, model in (
            (SM121_CACHE_SEMANTIC_CACHE_OFF_ARM, self.cache_off),
            (SM121_CACHE_SEMANTIC_CACHE_ON_ARM, self.cache_on),
        ):
            with self.subTest(arm=arm):
                static = _static_event(arm, 1)
                runtime = _runtime_event(arm, 1)
                validate_sm121_cache_semantic_static_attestation_event(static)
                validate_sm121_cache_semantic_runtime_attestation_event(runtime)
                for turn in SM121_CACHE_SEMANTIC_TURN_ORDER:
                    event = _turn_event(arm, turn)
                    self.assertEqual(
                        derive_sm121_cache_semantic_turn_admission(event),
                        (True, "admitted"),
                    )
                    validate_sm121_cache_semantic_turn_event(event)
                    self.assertEqual(sm121_cache_semantic_turn_issues(event), ())
                self.assertEqual(sm121_cache_semantic_arm(model), arm)

    def test_zero_hit_unexpected_details_are_partial_not_admitted(self) -> None:
        event = _turn_event(SM121_CACHE_SEMANTIC_CACHE_OFF_ARM, "T1")
        event.update(
            {
                "response_detail_state": "unexpected",
                "usage_detail_state": "unexpected",
                "response_device_cached_tokens": None,
                "response_host_cached_tokens": None,
                "response_storage_cached_tokens": None,
                "usage_cached_tokens": None,
            }
        )
        admitted, basis = derive_sm121_cache_semantic_turn_admission(event)
        self.assertEqual((admitted, basis), (False, "zero_hit_not_reconciled"))
        event["semantic_turn_admitted"] = admitted
        event["semantic_turn_basis"] = basis
        validate_sm121_cache_semantic_turn_event(event)
        self.assertIn(
            "semantic_zero_hit_details",
            {issue["code"] for issue in sm121_cache_semantic_turn_issues(event)},
        )

        tampered = deepcopy(event)
        tampered["semantic_turn_admitted"] = True
        tampered["semantic_turn_basis"] = "admitted"
        with self.assertRaises(SM121CacheSemanticError):
            validate_sm121_cache_semantic_turn_event(tampered)

    def test_nonzero_guardrail_baseline_is_not_admitted(self) -> None:
        for metric in ("evicted_tokens", "retracted_requests"):
            with self.subTest(metric=metric):
                event = _turn_event(SM121_CACHE_SEMANTIC_CACHE_ON_ARM, "T1")
                event[f"before_{metric}"] = 3
                event[f"after_{metric}"] = 3
                event[f"delta_{metric}"] = 0
                admitted, basis = derive_sm121_cache_semantic_turn_admission(event)
                self.assertEqual((admitted, basis), (False, "guardrail_activity"))
                event["semantic_turn_admitted"] = admitted
                event["semantic_turn_basis"] = basis
                validate_sm121_cache_semantic_turn_event(event)
                self.assertIn(
                    "semantic_cache_guardrail",
                    {issue["code"] for issue in sm121_cache_semantic_turn_issues(event)},
                )

    def test_lifecycle_accepts_b_and_a_and_preserves_partial_semantics(self) -> None:
        self.assertEqual(
            expected_sm121_cache_semantic_event_counts()["request_complete"], 7
        )
        for arm in SM121_CACHE_SEMANTIC_ARM_ORDER:
            with self.subTest(arm=arm):
                events, case_ids = _lifecycle_events(arm)
                self.assertEqual(
                    sm121_cache_semantic_lifecycle_issues(
                        events, planned_case_ids=case_ids, arm=arm
                    ),
                    (),
                )

        partial_events, partial_case_ids = _lifecycle_events(
            SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
            partial_semantic_case=True,
        )
        partial_issues = sm121_cache_semantic_lifecycle_issues(
            partial_events,
            planned_case_ids=partial_case_ids,
            arm=SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
        )
        partial_codes = {issue["code"] for issue in partial_issues}
        self.assertIn("semantic_zero_hit_details", partial_codes)
        self.assertNotIn("semantic_case_validation", partial_codes)

    def test_pair_binding_is_canonical_and_reciprocal(self) -> None:
        off_fingerprint = "0123456789abcdef"
        on_fingerprint = "fedcba9876543210"
        pair_instance = sm121_cache_semantic_pair_instance_sha256(
            "0" * 32, "1" * 32
        )
        with self.assertRaises(SM121CacheSemanticError):
            sm121_cache_semantic_pair_instance_sha256("0" * 32, "0" * 32)
        self.assertNotEqual(
            pair_instance,
            sm121_cache_semantic_pair_instance_sha256("1" * 32, "0" * 32),
        )
        cache_off_binding = _pair_binding(
            SM121_CACHE_SEMANTIC_CACHE_OFF_ARM, on_fingerprint, pair_instance
        )
        cache_on_binding = _pair_binding(
            SM121_CACHE_SEMANTIC_CACHE_ON_ARM, off_fingerprint, pair_instance
        )
        self.assertEqual(
            frozenset(cache_off_binding), SM121_CACHE_SEMANTIC_PAIR_BINDING_FIELDS
        )
        self.assertNotEqual(
            cache_off_binding["pair_binding_sha256"],
            cache_on_binding["pair_binding_sha256"],
        )
        validate_sm121_cache_semantic_pair_binding(
            cache_off_binding,
            self.cache_off,
            self.suite,
            peer_plan_fingerprint=on_fingerprint,
            peer_binding=cache_on_binding,
        )
        validate_sm121_cache_semantic_pair_binding(
            cache_on_binding,
            self.cache_on,
            self.suite,
            peer_plan_fingerprint=off_fingerprint,
            peer_binding=cache_off_binding,
        )

        tampered = deepcopy(cache_off_binding)
        tampered["semantic_case_metadata"] = {"schema_version": 1}
        with self.assertRaises(SM121CacheSemanticError):
            validate_sm121_cache_semantic_pair_binding(
                tampered,
                self.cache_off,
                self.suite,
                peer_plan_fingerprint=on_fingerprint,
            )

        mismatched_instance = deepcopy(cache_on_binding)
        mismatched_instance["pair_instance_sha256"] = (
            sm121_cache_semantic_pair_instance_sha256("2" * 32, "3" * 32)
        )
        mismatched_instance["pair_binding_sha256"] = (
            sm121_cache_semantic_pair_binding_sha256(mismatched_instance)
        )
        with self.assertRaises(SM121CacheSemanticError):
            validate_sm121_cache_semantic_pair_binding(
                cache_off_binding,
                self.cache_off,
                self.suite,
                peer_plan_fingerprint=on_fingerprint,
                peer_binding=mismatched_instance,
            )

        receipt = sm121_cache_semantic_cache_off_receipt_sha256(
            pair_instance,
            off_fingerprint,
            cache_off_binding["pair_binding_sha256"],
        )
        self.assertRegex(receipt, r"^sha256:[0-9a-f]{64}$")
        self.assertNotEqual(
            receipt,
            sm121_cache_semantic_cache_off_receipt_sha256(
                sm121_cache_semantic_pair_instance_sha256("4" * 32, "5" * 32),
                off_fingerprint,
                cache_off_binding["pair_binding_sha256"],
            ),
        )


if __name__ == "__main__":
    unittest.main()
