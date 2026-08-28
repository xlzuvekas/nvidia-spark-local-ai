from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import unittest

from bench.sglang_cache_policy_protocol import (
    ARM_A,
    ARM_A_CACHE_IMPL,
    ARM_B,
    ARM_B_CACHE_IMPL,
    DRAFT_BLOCKERS,
    FINISHED_REQUEST_DEVICE_HIT_METRIC,
    GUARDRAIL_RATIO,
    LIFETIME_ORDER,
    PREFILL_DEVICE_HIT_METRIC,
    PROMOTION_RATIO,
    PROTOCOL_PHASE,
    PROTOCOL_SCHEMA_VERSION,
    PROTOCOL_STATUS,
    SGLangCachePolicyProtocolError,
    VALIDATOR_ID,
    normalize_cache_observation,
    protocol_descriptor,
    protocol_sha256,
    summarize_cache_policy_campaign,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("ascii")).hexdigest()


PROVISIONAL_RUNTIME_DIGEST = _digest("synthetic provisional source contract")
PROVISIONAL_SOURCE_TREE = hashlib.sha256(
    b"synthetic composed source tree"
).hexdigest()[:40]


def _descriptor(
    *,
    t1_minimum: int = 32_000,
    t1_maximum: int = 32_768,
    t2_minimum: int = 32_256,
    t2_maximum: int = 33_024,
) -> dict[str, object]:
    return protocol_descriptor(
        provisional_runtime_source_contract_sha256=(
            PROVISIONAL_RUNTIME_DIGEST
        ),
        provisional_source_tree=PROVISIONAL_SOURCE_TREE,
        provisional_arm_a_t1_device_hit_minimum=t1_minimum,
        provisional_arm_a_t1_device_hit_maximum=t1_maximum,
        provisional_arm_a_t2_device_hit_minimum=t2_minimum,
        provisional_arm_a_t2_device_hit_maximum=t2_maximum,
    )


def _cache_observation(
    device_tokens: int,
    *,
    host_tokens: int = 0,
    storage_tokens: int = 0,
) -> dict[str, object]:
    return {
        "request_detail_state": "reported",
        "request_device_tokens": device_tokens,
        "request_host_tokens": host_tokens,
        "request_storage_tokens": storage_tokens,
        "provisional_settled_sglang_prefill_device_hit_tokens_delta": (
            device_tokens
        ),
        "provisional_settled_sglang_finished_request_device_hit_tokens_delta": (
            device_tokens
        ),
    }


def _turn(
    turn: str,
    arm: str,
    *,
    wall_s: float,
    ttft_s: float,
) -> dict[str, object]:
    if turn == "T0":
        input_tokens = 32_768
        common_prefix_tokens = 0
        cached_device_tokens = 0
    elif turn == "T1":
        input_tokens = 33_024
        common_prefix_tokens = 32_768
        cached_device_tokens = 32_512 if arm == ARM_A else 0
    elif turn == "T2":
        input_tokens = 33_280
        common_prefix_tokens = 33_024
        cached_device_tokens = 32_768 if arm == ARM_A else 0
    else:
        raise AssertionError(f"test helper received unknown turn {turn}")
    return {
        "turn": turn,
        "input_tokens": input_tokens,
        "common_prefix_tokens": common_prefix_tokens,
        "cache_observation": _cache_observation(cached_device_tokens),
        "provisional_prompt_identity_match": True,
        "provisional_correctness_passed": True,
        "eviction_count": 0,
        "retraction_count": 0,
        "other_request_count": 0,
        "pressure_breach": False,
        "ttft_s": ttft_s,
        "wall_s": wall_s,
    }


def _lifetime(
    ordinal: int,
    arm: str,
    *,
    later_wall_s: float,
    later_ttft_s: float,
    cold_wall_s: float,
) -> dict[str, object]:
    return {
        "lifetime_ordinal": ordinal,
        "arm": arm,
        "cache_impl": ARM_A_CACHE_IMPL if arm == ARM_A else ARM_B_CACHE_IMPL,
        "mamba_extra_buffer_of": arm == ARM_A,
        "mamba_extra_buffer_lazy_of": arm == ARM_A,
        "provisional_fresh_server_observed": True,
        "pre_t0_request_count": 0,
        "pre_t0_warmup_count": 0,
        "provisional_startup_identity_match": True,
        "turns": [
            _turn("T0", arm, wall_s=cold_wall_s, ttft_s=3.0),
            _turn(
                "T1",
                arm,
                wall_s=later_wall_s / 2,
                ttft_s=later_ttft_s,
            ),
            _turn(
                "T2",
                arm,
                wall_s=later_wall_s / 2,
                ttft_s=later_ttft_s,
            ),
        ],
    }


def _campaign(
    *,
    a_later_wall_s: tuple[float, float] = (100.0, 100.0),
    b_later_wall_s: tuple[float, float] = (95.0, 95.0),
    a_later_ttft_s: tuple[float, float] = (2.0, 2.0),
    b_later_ttft_s: tuple[float, float] = (2.1, 2.1),
    a_cold_wall_s: tuple[float, float] = (20.0, 20.0),
    b_cold_wall_s: tuple[float, float] = (25.0, 25.0),
) -> list[dict[str, object]]:
    arm_indexes = {ARM_A: 0, ARM_B: 0}
    result: list[dict[str, object]] = []
    for ordinal, arm in enumerate(LIFETIME_ORDER, start=1):
        index = arm_indexes[arm]
        arm_indexes[arm] += 1
        if arm == ARM_A:
            later_wall_s = a_later_wall_s[index]
            later_ttft_s = a_later_ttft_s[index]
            cold_wall_s = a_cold_wall_s[index]
        else:
            later_wall_s = b_later_wall_s[index]
            later_ttft_s = b_later_ttft_s[index]
            cold_wall_s = b_cold_wall_s[index]
        result.append(
            _lifetime(
                ordinal,
                arm,
                later_wall_s=later_wall_s,
                later_ttft_s=later_ttft_s,
                cold_wall_s=cold_wall_s,
            )
        )
    return result


def _envelope(
    *,
    descriptor: dict[str, object] | None = None,
    campaign: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    frozen_descriptor = _descriptor() if descriptor is None else descriptor
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_descriptor": frozen_descriptor,
        "protocol_sha256": protocol_sha256(frozen_descriptor),
        "provisional_runtime_source_contract_sha256": (
            PROVISIONAL_RUNTIME_DIGEST
        ),
        "validator_id": VALIDATOR_ID,
        "lifetimes": _campaign() if campaign is None else campaign,
    }


def _lifetime_row(
    envelope: dict[str, object], index: int
) -> dict[str, object]:
    lifetimes = envelope["lifetimes"]
    if not isinstance(lifetimes, list):
        raise AssertionError("test fixture lifetimes must be a list")
    row = lifetimes[index]
    if not isinstance(row, dict):
        raise AssertionError("test fixture lifetime must be an object")
    return row


def _turn_row(
    envelope: dict[str, object], lifetime_index: int, turn_index: int
) -> dict[str, object]:
    lifetime = _lifetime_row(envelope, lifetime_index)
    turns = lifetime["turns"]
    if not isinstance(turns, list):
        raise AssertionError("test fixture turns must be a list")
    row = turns[turn_index]
    if not isinstance(row, dict):
        raise AssertionError("test fixture turn must be an object")
    return row


def _cache_row(
    envelope: dict[str, object], lifetime_index: int, turn_index: int
) -> dict[str, object]:
    turn = _turn_row(envelope, lifetime_index, turn_index)
    cache = turn["cache_observation"]
    if not isinstance(cache, dict):
        raise AssertionError("test fixture cache observation must be an object")
    return cache


def _assert_scalar_leaves(test: unittest.TestCase, value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            test.assertIsInstance(key, str)
            _assert_scalar_leaves(test, child)
        return
    if isinstance(value, list):
        for child in value:
            _assert_scalar_leaves(test, child)
        return
    test.assertTrue(
        value is None or isinstance(value, (str, int, float, bool)),
        msg=f"non-scalar output leaf: {value!r}",
    )


class SGLangCachePolicyProtocolTests(unittest.TestCase):
    def test_descriptor_is_hash_bound_and_explicitly_draft_only(self) -> None:
        first = _descriptor()
        second = _descriptor()

        self.assertEqual(first, second)
        self.assertEqual(first["lifecycle"]["phase"], PROTOCOL_PHASE)
        self.assertEqual(first["lifecycle"]["status"], PROTOCOL_STATUS)
        self.assertFalse(first["lifecycle"]["protocol_runnable"])
        self.assertFalse(first["lifecycle"]["measurement_admissible"])
        self.assertEqual(first["lifecycle"]["selected_arm"], "none")
        self.assertEqual(first["lifecycle"]["blockers"], list(DRAFT_BLOCKERS))
        self.assertTrue(first["lifecycle"]["caller_cannot_override"])
        self.assertEqual(protocol_sha256(first), protocol_sha256(second))

    def test_descriptor_states_future_workload_and_native_requirements(self) -> None:
        descriptor = _descriptor()
        requirements = descriptor["future_admission_requirements"]
        workload = requirements["workload_identity"]

        self.assertEqual(
            set(workload),
            {
                "tokenizer",
                "model",
                "template",
                "tools",
                "turn_plan",
                "reasoning",
                "sampling",
                "output_cap",
                "correctness_validator",
            },
        )
        self.assertIn("pinned", workload["correctness_validator"])
        self.assertEqual(
            descriptor["unfrozen_native_contracts"],
            {
                "device_cache": (
                    "absent; request/native equality check is provisional only"
                ),
                "residency": "absent; no native gauges accepted",
                "state_pool": "absent; no native gauges accepted",
                "host_cache": "absent; request value is not native proof",
                "storage_cache": "absent; request value is not native proof",
            },
        )
        cache_contract = descriptor["provisional_cache_observation_contract"]
        self.assertEqual(
            cache_contract["prefill_device_hit_metric"],
            PREFILL_DEVICE_HIT_METRIC,
        )
        self.assertEqual(
            cache_contract["finished_request_device_hit_metric"],
            FINISHED_REQUEST_DEVICE_HIT_METRIC,
        )
        self.assertIn("does not prove", cache_contract["host_storage_rule"])

    def test_descriptor_requires_nonplaceholder_pin_tree_and_intervals(self) -> None:
        kwargs = {
            "provisional_runtime_source_contract_sha256": (
                "sha256:" + "0" * 64
            ),
            "provisional_source_tree": PROVISIONAL_SOURCE_TREE,
            "provisional_arm_a_t1_device_hit_minimum": 1,
            "provisional_arm_a_t1_device_hit_maximum": 2,
            "provisional_arm_a_t2_device_hit_minimum": 3,
            "provisional_arm_a_t2_device_hit_maximum": 4,
        }
        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError, "non-placeholder pinned digest"
        ):
            protocol_descriptor(**kwargs)

        kwargs["provisional_runtime_source_contract_sha256"] = (
            PROVISIONAL_RUNTIME_DIGEST
        )
        kwargs["provisional_source_tree"] = "66712e"
        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError, "full lowercase 40-hex Git tree"
        ):
            protocol_descriptor(**kwargs)

        kwargs["provisional_source_tree"] = PROVISIONAL_SOURCE_TREE
        kwargs["provisional_arm_a_t1_device_hit_minimum"] = 3
        kwargs["provisional_arm_a_t1_device_hit_maximum"] = 2
        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError, "interval is reversed"
        ):
            protocol_descriptor(**kwargs)

    def test_envelope_rejects_caller_lifecycle_and_canary_assertions(self) -> None:
        for field, value in (
            ("protocol_phase", "measurement"),
            ("protocol_status", "admitted"),
            ("zero_hit_canary_admitted", True),
            ("measurement_admissible", True),
        ):
            with self.subTest(field=field):
                envelope = _envelope()
                envelope[field] = value
                with self.assertRaisesRegex(
                    SGLangCachePolicyProtocolError, f"unknown {field}"
                ):
                    summarize_cache_policy_campaign(envelope)

    def test_envelope_types_hash_runtime_and_validator_are_strict(self) -> None:
        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError, "exact JSON object"
        ):
            summarize_cache_policy_campaign([])

        tuple_lifetimes = _envelope()
        tuple_lifetimes["lifetimes"] = tuple(tuple_lifetimes["lifetimes"])
        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError, "exact four-item JSON list"
        ):
            summarize_cache_policy_campaign(tuple_lifetimes)

        bad_hash = _envelope()
        bad_hash["protocol_sha256"] = _digest("different protocol")
        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError, "canonical descriptor"
        ):
            summarize_cache_policy_campaign(bad_hash)

        bad_runtime = _envelope()
        bad_runtime["provisional_runtime_source_contract_sha256"] = _digest(
            "different source"
        )
        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError, "does not match descriptor"
        ):
            summarize_cache_policy_campaign(bad_runtime)

        bad_validator = _envelope()
        bad_validator["validator_id"] = "self-admitting-validator"
        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError, "validator identity changed"
        ):
            summarize_cache_policy_campaign(bad_validator)

    def test_descriptor_lifecycle_tamper_fails_with_recomputed_hash(self) -> None:
        envelope = _envelope()
        descriptor = deepcopy(envelope["protocol_descriptor"])
        descriptor["lifecycle"]["measurement_admissible"] = True
        descriptor["lifecycle"]["status"] = "admitted"
        envelope["protocol_descriptor"] = descriptor
        envelope["protocol_sha256"] = protocol_sha256(descriptor)

        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError, "descriptor semantics"
        ):
            summarize_cache_policy_campaign(envelope)

    def test_clean_diagnostic_can_never_admit_or_select(self) -> None:
        summary = summarize_cache_policy_campaign(_envelope())

        self.assertEqual(summary["protocol_phase"], "draft")
        self.assertEqual(summary["protocol_status"], "not_admitted")
        self.assertFalse(summary["protocol_runnable"])
        self.assertFalse(summary["measurement_admissible"])
        self.assertEqual(summary["decision"], "not_admitted")
        self.assertEqual(summary["selected_arm"], "none")
        self.assertEqual(summary["draft_blockers"], list(DRAFT_BLOCKERS))
        self.assertEqual(summary["diagnostic_speed_result"], "b_faster")
        self.assertEqual(summary["diagnostic_candidate_arm"], ARM_B)
        self.assertNotIn("promote_b", json.dumps(summary, sort_keys=True))
        self.assertNotIn("retain_a", json.dumps(summary, sort_keys=True))

    def test_cache_implementation_cannot_echo_arbitrary_caller_text(self) -> None:
        for value in (
            "synthetic-secret-sentinel",
            "admitted",
            "promote_b",
            "retain_a",
        ):
            with self.subTest(value=value):
                envelope = _envelope()
                _lifetime_row(envelope, 0)["cache_impl"] = value
                with self.assertRaisesRegex(
                    SGLangCachePolicyProtocolError,
                    "cache implementation must be one of",
                ):
                    summarize_cache_policy_campaign(envelope)

    def test_reported_cache_reconciles_and_publishes_both_native_deltas(self) -> None:
        observation = _cache_observation(32_512)
        normalized = normalize_cache_observation(observation)

        self.assertEqual(
            normalized,
            {
                "request_detail_state": "reported",
                "reported_request_device_tokens": 32_512,
                "reported_request_host_tokens": 0,
                "reported_request_storage_tokens": 0,
                "provisional_native_prefill_device_hit_tokens_delta": 32_512,
                "provisional_native_finished_request_device_hit_tokens_delta": (
                    32_512
                ),
                "provisional_device_reconciliation_matched": True,
            },
        )
        summary = summarize_cache_policy_campaign(_envelope())
        turn = summary["lifetimes"][0]["turns"][1]
        self.assertEqual(
            turn["provisional_native_prefill_device_hit_tokens_delta"], 32_512
        )
        self.assertEqual(
            turn[
                "provisional_native_finished_request_device_hit_tokens_delta"
            ],
            32_512,
        )
        self.assertFalse(turn["native_host_storage_reconciliation_available"])

    def test_omitted_and_null_cache_details_are_rejected(self) -> None:
        for state in ("omitted", "null"):
            with self.subTest(state=state):
                observation = _cache_observation(0)
                observation["request_detail_state"] = state
                observation["request_device_tokens"] = None
                observation["request_host_tokens"] = None
                observation["request_storage_tokens"] = None
                with self.assertRaisesRegex(
                    SGLangCachePolicyProtocolError,
                    "no admitted zero-hit canary record exists",
                ):
                    normalize_cache_observation(observation)

    def test_request_and_native_device_mismatches_fail_closed(self) -> None:
        request_mismatch = _envelope()
        cache = _cache_row(request_mismatch, 0, 1)
        cache["request_device_tokens"] = 32_511
        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError, "disagrees with provisional native"
        ):
            summarize_cache_policy_campaign(request_mismatch)

        native_mismatch = _envelope()
        cache = _cache_row(native_mismatch, 0, 1)
        cache[
            "provisional_settled_sglang_finished_request_device_hit_tokens_delta"
        ] = 32_511
        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError, "native device-hit deltas disagree"
        ):
            summarize_cache_policy_campaign(native_mismatch)

    def test_host_storage_zero_is_never_published_as_native_proof(self) -> None:
        summary = summarize_cache_policy_campaign(_envelope())
        self.assertFalse(summary["native_host_storage_reconciliation_available"])
        self.assertIn(
            "native_host_storage_cache_reconciliation_contract_absent",
            summary["draft_blockers"],
        )

        nonzero = _envelope()
        _cache_row(nonzero, 0, 1)["request_host_tokens"] = 1
        diagnostic = summarize_cache_policy_campaign(nonzero)
        self.assertIn("cache_hit", diagnostic["diagnostic_invalid_gates"])
        self.assertEqual(
            diagnostic["diagnostic_speed_result"], "not_evaluated"
        )
        self.assertEqual(diagnostic["decision"], "not_admitted")

    def test_rows_cannot_choose_provisional_hit_intervals(self) -> None:
        envelope = _envelope()
        _turn_row(envelope, 0, 1)["expected_device_hit_minimum"] = 1
        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError,
            "unknown expected_device_hit_minimum",
        ):
            summarize_cache_policy_campaign(envelope)

        below_interval = _envelope()
        cache = _cache_row(below_interval, 0, 1)
        cache["request_device_tokens"] = 31_999
        cache[
            "provisional_settled_sglang_prefill_device_hit_tokens_delta"
        ] = 31_999
        cache[
            "provisional_settled_sglang_finished_request_device_hit_tokens_delta"
        ] = 31_999
        summary = summarize_cache_policy_campaign(below_interval)
        self.assertEqual(summary["diagnostic_invalid_gates"], ["cache_hit"])
        self.assertEqual(summary["diagnostic_speed_result"], "not_evaluated")

    def test_diagnostic_gates_never_change_draft_decision(self) -> None:
        envelope = _envelope()
        first = _lifetime_row(envelope, 0)
        first["provisional_fresh_server_observed"] = False
        first["pre_t0_request_count"] = 1
        first["pre_t0_warmup_count"] = 1
        _turn_row(envelope, 0, 1)["provisional_correctness_passed"] = False

        summary = summarize_cache_policy_campaign(envelope)

        self.assertEqual(
            summary["diagnostic_invalid_gates"],
            [
                "fresh_server",
                "pre_t0_request_count",
                "pre_t0_warmup_count",
                "correctness",
            ],
        )
        self.assertFalse(summary["diagnostic_observations_valid"])
        self.assertEqual(summary["diagnostic_speed_result"], "not_evaluated")
        self.assertFalse(summary["measurement_admissible"])
        self.assertEqual(summary["decision"], "not_admitted")

    def test_abba_and_t0_t2_topology_are_exact(self) -> None:
        wrong_arm = _envelope()
        _lifetime_row(wrong_arm, 1)["arm"] = ARM_A
        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError, "ordered exactly A, B, B, A"
        ):
            summarize_cache_policy_campaign(wrong_arm)

        wrong_turn = _envelope()
        turns = _lifetime_row(wrong_turn, 0)["turns"]
        if not isinstance(turns, list):
            raise AssertionError("test fixture turns must be a list")
        turns[1], turns[2] = turns[2], turns[1]
        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError, "ordered exactly T0, T1, T2"
        ):
            summarize_cache_policy_campaign(wrong_turn)

    def test_b_diagnostic_boundaries_are_inclusive(self) -> None:
        summary = summarize_cache_policy_campaign(_envelope())

        self.assertEqual(summary["b_over_a_later_wall_ratio"], 0.95)
        self.assertEqual(summary["diagnostic_speed_result"], "b_faster")
        self.assertEqual(summary["diagnostic_later_ttft_ratio"], 1.05)
        self.assertEqual(summary["diagnostic_full_wall_ratio"], 1.0)
        self.assertTrue(summary["diagnostic_guardrails_evaluated"])
        self.assertTrue(summary["diagnostic_guardrails_passed"])
        self.assertEqual(summary["decision"], "not_admitted")

    def test_a_diagnostic_boundaries_are_inclusive(self) -> None:
        campaign = _campaign(
            a_later_wall_s=(95.0, 95.0),
            b_later_wall_s=(100.0, 100.0),
            a_later_ttft_s=(2.1, 2.1),
            b_later_ttft_s=(2.0, 2.0),
            a_cold_wall_s=(25.0, 25.0),
            b_cold_wall_s=(20.0, 20.0),
        )
        summary = summarize_cache_policy_campaign(_envelope(campaign=campaign))

        self.assertEqual(summary["a_over_b_later_wall_ratio"], 0.95)
        self.assertEqual(summary["diagnostic_speed_result"], "a_faster")
        self.assertEqual(summary["diagnostic_later_ttft_ratio"], 1.05)
        self.assertEqual(summary["diagnostic_full_wall_ratio"], 1.0)
        self.assertTrue(summary["diagnostic_guardrails_passed"])
        self.assertEqual(summary["selected_arm"], "none")

    def test_speed_just_beyond_point_95_is_inconclusive_both_directions(
        self,
    ) -> None:
        campaigns = {
            "B": _campaign(b_later_wall_s=(95.0001, 95.0001)),
            "A": _campaign(
                a_later_wall_s=(95.0001, 95.0001),
                b_later_wall_s=(100.0, 100.0),
            ),
        }
        for direction, campaign in campaigns.items():
            with self.subTest(direction=direction):
                summary = summarize_cache_policy_campaign(
                    _envelope(campaign=campaign)
                )
                self.assertEqual(
                    summary["diagnostic_speed_result"], "inconclusive"
                )
                self.assertFalse(summary["diagnostic_guardrails_evaluated"])
                self.assertIsNone(summary["diagnostic_later_ttft_ratio"])
                self.assertIsNone(summary["diagnostic_full_wall_ratio"])
                self.assertEqual(summary["decision"], "not_admitted")

    def test_full_wall_at_and_beyond_1_05_is_symmetric(self) -> None:
        cases = {
            "B_at": (_campaign(
                a_later_ttft_s=(2.0, 2.0),
                b_later_ttft_s=(2.0, 2.0),
                a_cold_wall_s=(20.0, 20.0),
                b_cold_wall_s=(31.0, 31.0),
            ), True),
            "B_beyond": (_campaign(
                a_later_ttft_s=(2.0, 2.0),
                b_later_ttft_s=(2.0, 2.0),
                a_cold_wall_s=(20.0, 20.0),
                b_cold_wall_s=(31.0001, 31.0001),
            ), False),
            "A_at": (_campaign(
                a_later_wall_s=(95.0, 95.0),
                b_later_wall_s=(100.0, 100.0),
                a_later_ttft_s=(2.0, 2.0),
                b_later_ttft_s=(2.0, 2.0),
                a_cold_wall_s=(31.0, 31.0),
                b_cold_wall_s=(20.0, 20.0),
            ), True),
            "A_beyond": (_campaign(
                a_later_wall_s=(95.0, 95.0),
                b_later_wall_s=(100.0, 100.0),
                a_later_ttft_s=(2.0, 2.0),
                b_later_ttft_s=(2.0, 2.0),
                a_cold_wall_s=(31.0001, 31.0001),
                b_cold_wall_s=(20.0, 20.0),
            ), False),
        }
        for name, (campaign, expected_pass) in cases.items():
            with self.subTest(name=name):
                summary = summarize_cache_policy_campaign(
                    _envelope(campaign=campaign)
                )
                self.assertTrue(summary["diagnostic_guardrails_evaluated"])
                self.assertEqual(
                    summary["diagnostic_guardrails_passed"], expected_pass
                )
                if expected_pass:
                    self.assertEqual(
                        summary["diagnostic_full_wall_ratio"], 1.05
                    )
                else:
                    self.assertGreater(
                        summary["diagnostic_full_wall_ratio"], 1.05
                    )
                self.assertEqual(summary["decision"], "not_admitted")

    def test_selected_ttft_just_beyond_1_05_is_symmetric(self) -> None:
        campaigns = {
            "B": _campaign(b_later_ttft_s=(2.1001, 2.1001)),
            "A": _campaign(
                a_later_wall_s=(95.0, 95.0),
                b_later_wall_s=(100.0, 100.0),
                a_later_ttft_s=(2.1001, 2.1001),
                b_later_ttft_s=(2.0, 2.0),
                a_cold_wall_s=(25.0, 25.0),
                b_cold_wall_s=(20.0, 20.0),
            ),
        }
        for direction, campaign in campaigns.items():
            with self.subTest(direction=direction):
                summary = summarize_cache_policy_campaign(
                    _envelope(campaign=campaign)
                )
                self.assertEqual(
                    summary["diagnostic_candidate_arm"], direction
                )
                self.assertGreater(
                    summary["diagnostic_later_ttft_ratio"], 1.05
                )
                self.assertFalse(summary["diagnostic_guardrails_passed"])
                self.assertEqual(summary["selected_arm"], "none")
                self.assertEqual(summary["decision"], "not_admitted")

    def test_nonfinite_inputs_aggregates_and_ratios_fail_closed(self) -> None:
        nonfinite_input = _envelope()
        _turn_row(nonfinite_input, 0, 0)["wall_s"] = float("nan")
        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError, "positive finite JSON number"
        ):
            summarize_cache_policy_campaign(nonfinite_input)

        aggregate = _envelope()
        for lifetime_index in range(4):
            for turn_index in (1, 2):
                _turn_row(aggregate, lifetime_index, turn_index)["wall_s"] = 1e308
        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError, "finite representable JSON number"
        ):
            summarize_cache_policy_campaign(aggregate)

        ratio_campaign = _campaign(
            a_later_wall_s=(1e-323, 1e-323),
            b_later_wall_s=(1e100, 1e100),
            a_later_ttft_s=(5e-324, 5e-324),
            b_later_ttft_s=(2.0, 2.0),
        )
        with self.assertRaisesRegex(
            SGLangCachePolicyProtocolError, "finite representable JSON number"
        ):
            summarize_cache_policy_campaign(
                _envelope(campaign=ratio_campaign)
            )

    def test_scalar_proof_is_deterministic_and_payload_free(self) -> None:
        envelope = _envelope(campaign=_campaign(a_later_wall_s=(80.0, 120.0)))
        first = summarize_cache_policy_campaign(envelope)
        second = summarize_cache_policy_campaign(deepcopy(envelope))

        self.assertEqual(first, second)
        self.assertEqual(first["arm_a_mean_later_wall_s"], 100.0)
        self.assertEqual(first["arm_b_mean_later_wall_s"], 95.0)
        first_a = first["lifetimes"][0]
        self.assertEqual(
            [turn["turn"] for turn in first_a["turns"]],
            ["T0", "T1", "T2"],
        )
        self.assertEqual(
            first_a["turns"][0]["reported_request_device_tokens"], 0
        )
        self.assertEqual(
            first_a["turns"][1]["reported_request_device_tokens"], 32_512
        )
        _assert_scalar_leaves(self, first)
        serialized = json.dumps(first, sort_keys=True, allow_nan=False)
        for forbidden in (
            "request_id",
            "prompt_token_ids",
            "completion",
            "reasoning_payload",
            "tool_payload",
        ):
            self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
