from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from pathlib import Path
import tempfile
import unittest

from bench.autoresearch import (
    AUDIT_RESERVE_S,
    CALIBRATION_GEOMEAN_MAX,
    CALIBRATION_GEOMEAN_MIN,
    CALIBRATION_PRIMARY_RATIO_MAX,
    CALIBRATION_PRIMARY_RATIO_MIN,
    CALIBRATION_TTFT_RATIO_MAX,
    CALIBRATION_TTFT_RATIO_MIN,
    CELL_TIMEOUT_S,
    CLEANUP_TIMEOUT_S,
    PAIR_TIMEOUT_S,
    POLICY_SCHEMA_VERSION,
    PROMOTION_COMBINED_GEOMEAN_MIN,
    PROMOTION_PAIR_GEOMEAN_STRICT_MIN,
    SCREEN_GEOMEAN_MIN,
    SCREEN_PRIMARY_RATIO_MIN,
    SCREEN_TTFT_RATIO_MAX,
    SIMPLIFICATION_COMBINED_GEOMEAN_MIN,
    SIMPLIFICATION_MEMAVAILABLE_GAIN_GIB,
    AutoresearchError,
    CampaignPolicy,
    EligibilityInputs,
    FailureKind,
    PairObservation,
    SimplificationEvidence,
    TimingInputs,
    TransitionError,
    append_transition,
    evaluate_calibration,
    evaluate_promotion,
    evaluate_screen,
    evaluate_simplification_promotion,
    evaluate_simplification_screen,
    failure_disposition,
    geometric_mean,
    pair_order,
    replay_transitions,
    strictly_simpler_flag_bundle,
    validate_one_axis_delta,
)
from bench.journal import Journal


PRIMARY_CASES = ("decode-256-c1", "fresh-short-c1")
ALLOWED_AXES = ("mtp-depth", "buffer-strategy", "ple-placement")


def policy() -> CampaignPolicy:
    return CampaignPolicy(
        primary_case_ids=PRIMARY_CASES,
        allowed_axes=ALLOWED_AXES,
    )


def eligible(**overrides: bool) -> EligibilityInputs:
    values = {
        "cells_completed": True,
        "measurement_valid": True,
        "validation_passed": True,
        "workload_matched": True,
        "artifact_identity_verified": True,
        "audit_requirement_passed": True,
        "cleanup_verified": True,
        "memory_pressure": False,
        "swap_pressure": False,
        "oom": False,
        "ownership_ambiguous": False,
        "cleanup_breach": False,
    }
    values.update(overrides)
    return EligibilityInputs(**values)


def observation(
    pair_index: int,
    ratios: tuple[float, ...] = (1.04, 1.04),
    *,
    ttft: float = 1.0,
    gates: EligibilityInputs | None = None,
    memory_gain_gib: float = 0.0,
    champion_flags: tuple[str, ...] = ("--base",),
    candidate_flags: tuple[str, ...] = ("--base",),
    case_ids: tuple[str, ...] = PRIMARY_CASES,
    timing: TimingInputs | None = None,
) -> PairObservation:
    return PairObservation(
        pair_index=pair_index,
        primary_case_ids=case_ids,
        primary_speed_ratios=ratios,
        median_ttft_ratio=ttft,
        timing=timing
        or TimingInputs(
            cell_elapsed_s=(600.0, 600.0),
            pair_elapsed_s=1_200.0,
            cleanup_elapsed_s=30.0,
            audit_reserve_remaining_s=1_200.0,
        ),
        eligibility=gates or eligible(),
        simplification=SimplificationEvidence(
            minimum_memavailable_gain_gib=memory_gain_gib,
            champion_flags=champion_flags,
            candidate_flags=candidate_flags,
        ),
    )


def transition(event: str, index: int, **fields: object) -> dict[str, object]:
    return {"event": event, "transition_id": f"t{index}", **fields}


def campaign_started(contract: CampaignPolicy, index: int = 0) -> dict[str, object]:
    return transition(
        "autoresearch_campaign_started",
        index,
        campaign_id="campaign-1",
        policy_digest=contract.digest,
    )


def candidate_started(index: int, candidate_id: str = "candidate-1") -> dict[str, object]:
    return transition(
        "autoresearch_candidate_started",
        index,
        candidate_id=candidate_id,
        axis="mtp-depth",
        delta_digest="a" * 64,
    )


def pair_events(
    pair: PairObservation,
    *,
    first_transition: int,
    candidate_id: str = "candidate-1",
) -> list[dict[str, object]]:
    order = pair_order(pair.pair_index)
    return [
        transition(
            "autoresearch_pair_started",
            first_transition,
            candidate_id=candidate_id,
            pair_index=pair.pair_index,
            order=list(order),
        ),
        transition(
            "autoresearch_cell_completed",
            first_transition + 1,
            candidate_id=candidate_id,
            pair_index=pair.pair_index,
            arm=order[0],
        ),
        transition(
            "autoresearch_cell_completed",
            first_transition + 2,
            candidate_id=candidate_id,
            pair_index=pair.pair_index,
            arm=order[1],
        ),
        transition(
            "autoresearch_pair_scored",
            first_transition + 3,
            candidate_id=candidate_id,
            pair_index=pair.pair_index,
            observation=pair.to_mapping(),
        ),
    ]


class CampaignPolicyTests(unittest.TestCase):
    def test_policy_is_immutable_exact_and_digest_stable(self) -> None:
        contract = policy()

        self.assertEqual(contract.schema_version, POLICY_SCHEMA_VERSION)
        self.assertEqual(contract.cell_timeout_s, CELL_TIMEOUT_S)
        self.assertEqual(contract.pair_timeout_s, PAIR_TIMEOUT_S)
        self.assertEqual(contract.cleanup_timeout_s, CLEANUP_TIMEOUT_S)
        self.assertEqual(contract.audit_reserve_s, AUDIT_RESERVE_S)
        self.assertEqual(contract.screen_geomean_min, SCREEN_GEOMEAN_MIN)
        self.assertEqual(
            contract.screen_primary_ratio_min, SCREEN_PRIMARY_RATIO_MIN
        )
        self.assertEqual(contract.screen_ttft_ratio_max, SCREEN_TTFT_RATIO_MAX)
        self.assertEqual(
            contract.promotion_pair_geomean_strict_min,
            PROMOTION_PAIR_GEOMEAN_STRICT_MIN,
        )
        self.assertEqual(
            contract.promotion_combined_geomean_min,
            PROMOTION_COMBINED_GEOMEAN_MIN,
        )
        self.assertEqual(
            contract.simplification_combined_geomean_min,
            SIMPLIFICATION_COMBINED_GEOMEAN_MIN,
        )
        self.assertEqual(
            contract.simplification_memavailable_gain_gib,
            SIMPLIFICATION_MEMAVAILABLE_GAIN_GIB,
        )
        self.assertRegex(contract.digest, r"^[0-9a-f]{64}$")
        self.assertEqual(
            CampaignPolicy.from_mapping(contract.to_mapping()).digest,
            contract.digest,
        )
        with self.assertRaises(FrozenInstanceError):
            contract.cell_timeout_s = 1  # type: ignore[misc]

    def test_policy_rejects_unknown_and_missing_keys(self) -> None:
        raw = policy().to_mapping()
        with self.assertRaisesRegex(AutoresearchError, "unknown keys"):
            CampaignPolicy.from_mapping({**raw, "surprise": 1})
        missing = dict(raw)
        missing.pop("audit_reserve_s")
        with self.assertRaisesRegex(AutoresearchError, "missing keys"):
            CampaignPolicy.from_mapping(missing)

    def test_every_fixed_policy_value_rejects_mutation(self) -> None:
        raw = policy().to_mapping()
        fixed_fields = set(raw) - {"primary_case_ids", "allowed_axes"}
        for field in fixed_fields:
            with self.subTest(field=field):
                changed = dict(raw)
                value = changed[field]
                changed[field] = value + 1 if isinstance(value, int) else value + 0.01
                with self.assertRaisesRegex(AutoresearchError, "must equal"):
                    CampaignPolicy.from_mapping(changed)

    def test_policy_rejects_bool_numbers_and_bad_id_sets(self) -> None:
        raw = policy().to_mapping()
        for field in ("cell_timeout_s", "screen_geomean_min"):
            with self.subTest(field=field):
                malformed = dict(raw)
                malformed[field] = True
                with self.assertRaises(AutoresearchError):
                    CampaignPolicy.from_mapping(malformed)
        for field, value in (
            ("primary_case_ids", []),
            ("primary_case_ids", ["same", "same"]),
            ("allowed_axes", ["UpperCase"]),
        ):
            with self.subTest(field=field, value=value):
                malformed = dict(raw)
                malformed[field] = value
                with self.assertRaises(AutoresearchError):
                    CampaignPolicy.from_mapping(malformed)
        with self.assertRaisesRegex(AutoresearchError, "immutable tuple"):
            CampaignPolicy(
                primary_case_ids=list(PRIMARY_CASES),  # type: ignore[arg-type]
                allowed_axes=ALLOWED_AXES,
            )


class CandidateDeltaTests(unittest.TestCase):
    def test_exactly_one_allowed_axis_is_digest_bound(self) -> None:
        champion = {
            "mtp-depth": 2,
            "buffer-strategy": "lazy",
            "nested": {"graphs": [1, 2, 4]},
        }
        candidate = {
            "nested": {"graphs": [1, 2, 4]},
            "buffer-strategy": "lazy",
            "mtp-depth": 3,
        }

        delta = validate_one_axis_delta(
            champion,
            candidate,
            allowed_axes=("mtp-depth",),
        )

        self.assertEqual(delta.axis, "mtp-depth")
        self.assertEqual(delta.champion_value_json, "2")
        self.assertEqual(delta.candidate_value_json, "3")
        self.assertNotEqual(delta.champion_config_digest, delta.candidate_config_digest)
        self.assertRegex(delta.digest, r"^[0-9a-f]{64}$")

    def test_bool_and_integer_are_distinct_json_values(self) -> None:
        delta = validate_one_axis_delta(
            {"mtp-depth": True},
            {"mtp-depth": 1},
            allowed_axes=("mtp-depth",),
        )
        self.assertEqual(delta.champion_value_json, "true")
        self.assertEqual(delta.candidate_value_json, "1")

    def test_delta_rejects_no_change_multiple_changes_and_topology_change(self) -> None:
        invalid = (
            ({"a": 1}, {"a": 1}, ("a",)),
            ({"a": 1, "b": 2}, {"a": 2, "b": 3}, ("a", "b")),
            ({"a": 1}, {"a": 2, "b": 3}, ("a", "b")),
            ({}, {}, ("a",)),
        )
        for champion, candidate, axes in invalid:
            with self.subTest(champion=champion, candidate=candidate):
                with self.assertRaises(AutoresearchError):
                    validate_one_axis_delta(
                        champion, candidate, allowed_axes=axes
                    )

    def test_delta_rejects_disallowed_duplicate_and_invalid_axes(self) -> None:
        with self.assertRaisesRegex(AutoresearchError, "disallowed"):
            validate_one_axis_delta(
                {"a": 1}, {"a": 2}, allowed_axes=("other",)
            )
        with self.assertRaisesRegex(AutoresearchError, "duplicates"):
            validate_one_axis_delta(
                {"a": 1}, {"a": 2}, allowed_axes=("a", "a")
            )
        with self.assertRaises(AutoresearchError):
            validate_one_axis_delta(
                {"Upper": 1}, {"Upper": 2}, allowed_axes=("Upper",)
            )

    def test_delta_rejects_nonfinite_or_non_json_values(self) -> None:
        for value in (math.nan, math.inf, {1, 2}):
            with self.subTest(value=value), self.assertRaises(AutoresearchError):
                validate_one_axis_delta(
                    {"a": 1}, {"a": value}, allowed_axes=("a",)
                )

    def test_strict_flag_bundle_requires_a_proper_subset(self) -> None:
        self.assertTrue(
            strictly_simpler_flag_bundle(("--a", "--b"), ("--a",))
        )
        self.assertTrue(strictly_simpler_flag_bundle(("--a",), ()))
        self.assertFalse(
            strictly_simpler_flag_bundle(("--a", "--b"), ("--b", "--a"))
        )
        self.assertFalse(
            strictly_simpler_flag_bundle(("--a",), ("--different",))
        )
        with self.assertRaisesRegex(AutoresearchError, "duplicates"):
            strictly_simpler_flag_bundle(("--a", "--a"), ("--a",))
        with self.assertRaises(AutoresearchError):
            strictly_simpler_flag_bundle(("--a\nsecret",), ())


class EligibilityAndObservationTests(unittest.TestCase):
    def test_eligibility_is_explicit_roundtrippable_and_immutable(self) -> None:
        gates = eligible()
        self.assertTrue(gates.eligible)
        self.assertEqual(gates.failed_gates, ())
        self.assertFalse(gates.campaign_terminal_pressure)
        self.assertEqual(
            EligibilityInputs.from_mapping(gates.to_mapping()), gates
        )
        with self.assertRaises(FrozenInstanceError):
            gates.oom = True  # type: ignore[misc]

    def test_each_positive_and_hazard_gate_fails_closed(self) -> None:
        positives = (
            "cells_completed",
            "measurement_valid",
            "validation_passed",
            "workload_matched",
            "artifact_identity_verified",
            "audit_requirement_passed",
            "cleanup_verified",
        )
        hazards = (
            "memory_pressure",
            "swap_pressure",
            "oom",
            "ownership_ambiguous",
            "cleanup_breach",
        )
        for field in positives:
            with self.subTest(field=field):
                gates = eligible(**{field: False})
                self.assertFalse(gates.eligible)
                self.assertIn(field, gates.failed_gates)
                self.assertFalse(gates.campaign_terminal_pressure)
        for field in hazards:
            with self.subTest(field=field):
                gates = eligible(**{field: True})
                self.assertFalse(gates.eligible)
                self.assertIn(field, gates.failed_gates)
                self.assertTrue(gates.campaign_terminal_pressure)

    def test_eligibility_rejects_unknown_missing_and_nonboolean_values(self) -> None:
        raw = eligible().to_mapping()
        with self.assertRaisesRegex(AutoresearchError, "unknown keys"):
            EligibilityInputs.from_mapping({**raw, "unknown": False})
        missing = dict(raw)
        missing.pop("oom")
        with self.assertRaisesRegex(AutoresearchError, "missing keys"):
            EligibilityInputs.from_mapping(missing)
        malformed = dict(raw)
        malformed["oom"] = 0
        with self.assertRaisesRegex(AutoresearchError, "boolean"):
            EligibilityInputs.from_mapping(malformed)

    def test_observation_roundtrip_and_alignment_validation(self) -> None:
        value = observation(2)
        self.assertEqual(PairObservation.from_mapping(value.to_mapping()), value)
        self.assertAlmostEqual(value.speed_geomean, 1.04)
        raw = value.to_mapping()
        with self.assertRaisesRegex(AutoresearchError, "unknown keys"):
            PairObservation.from_mapping({**raw, "raw_result": "forbidden"})
        for ratios in ([], [1.0], [0.0, 1.0], [math.nan, 1.0]):
            malformed = dict(raw)
            malformed["primary_speed_ratios"] = ratios
            with self.subTest(ratios=ratios), self.assertRaises(AutoresearchError):
                PairObservation.from_mapping(malformed)
        malformed = dict(raw)
        malformed["median_ttft_ratio"] = True
        with self.assertRaises(AutoresearchError):
            PairObservation.from_mapping(malformed)

    def test_simplification_requires_immutable_valid_flag_bundles(self) -> None:
        with self.assertRaisesRegex(AutoresearchError, "tuples"):
            SimplificationEvidence(
                minimum_memavailable_gain_gib=1.0,
                champion_flags=["--a"],  # type: ignore[arg-type]
                candidate_flags=(),
            )
        raw = observation(0).simplification.to_mapping()
        with self.assertRaisesRegex(AutoresearchError, "unknown keys"):
            SimplificationEvidence.from_mapping({**raw, "claim": True})

    def test_timing_inputs_roundtrip_and_reject_malformed_values(self) -> None:
        timing = observation(0).timing
        self.assertEqual(TimingInputs.from_mapping(timing.to_mapping()), timing)
        raw = timing.to_mapping()
        with self.assertRaisesRegex(AutoresearchError, "unknown keys"):
            TimingInputs.from_mapping({**raw, "sleep_s": 1})
        for field, value in (
            ("cell_elapsed_s", [1.0]),
            ("cell_elapsed_s", [1.0, -1.0]),
            ("pair_elapsed_s", math.inf),
            ("cleanup_elapsed_s", True),
            ("audit_reserve_remaining_s", -0.1),
        ):
            with self.subTest(field=field, value=value):
                malformed = dict(raw)
                malformed[field] = value
                with self.assertRaises(AutoresearchError):
                    TimingInputs.from_mapping(malformed)

    def test_timing_budget_boundaries_are_inclusive_and_each_overrun_fails(self) -> None:
        contract = policy()
        boundary = TimingInputs(
            cell_elapsed_s=(1_800.0, 1_800.0),
            pair_elapsed_s=3_600.0,
            cleanup_elapsed_s=120.0,
            audit_reserve_remaining_s=900.0,
        )
        self.assertEqual(boundary.failed_budgets(contract), ())
        failures = (
            (
                TimingInputs((1_800.001, 1.0), 2.0, 1.0, 900.0),
                "cell_timeout",
            ),
            (
                TimingInputs((1.0, 1.0), 3_600.001, 1.0, 900.0),
                "pair_timeout",
            ),
            (
                TimingInputs((1.0, 1.0), 2.0, 120.001, 900.0),
                "cleanup_timeout",
            ),
            (
                TimingInputs((1.0, 1.0), 2.0, 1.0, 899.999),
                "audit_reserve",
            ),
        )
        for timing, reason in failures:
            with self.subTest(reason=reason):
                self.assertEqual(timing.failed_budgets(contract), (reason,))


class ScorePolicyTests(unittest.TestCase):
    def test_geometric_mean_is_stable_and_strictly_positive(self) -> None:
        self.assertAlmostEqual(geometric_mean((1.0, 4.0)), 2.0)
        self.assertAlmostEqual(geometric_mean((1e-200, 1e200)), 1.0)
        for values in ((), (0.0,), (-1.0,), (math.nan,), (math.inf,), (True,)):
            with self.subTest(values=values), self.assertRaises(AutoresearchError):
                geometric_mean(values)

    def test_pair_order_alternates_forever_and_rejects_bad_indexes(self) -> None:
        self.assertEqual(pair_order(0), ("champion", "candidate"))
        self.assertEqual(pair_order(1), ("candidate", "champion"))
        self.assertEqual(pair_order(8), ("champion", "candidate"))
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(AutoresearchError):
                pair_order(value)  # type: ignore[arg-type]

    def test_screen_boundaries_are_inclusive(self) -> None:
        contract = policy()
        at_geomean = observation(0, (1.03, 1.03), ttft=1.10)
        at_primary_floor = observation(0, (0.95, 1.20), ttft=1.10)
        self.assertTrue(evaluate_screen(contract, at_geomean).passed)
        self.assertTrue(evaluate_screen(contract, at_primary_floor).passed)

    def test_screen_enforces_geomean_every_primary_ttft_and_eligibility(self) -> None:
        contract = policy()
        failures = (
            (observation(0, (1.029, 1.029)), "geomean_below_screen_min"),
            (
                observation(0, (0.949, 1.30)),
                "primary_ratio_below_screen_floor",
            ),
            (
                observation(0, (1.04, 1.04), ttft=1.101),
                "ttft_ratio_above_screen_ceiling",
            ),
            (
                observation(0, gates=eligible(validation_passed=False)),
                "ineligible:validation_passed",
            ),
            (
                observation(0, case_ids=("other", "fresh-short-c1")),
                "primary_cases_mismatch",
            ),
            (
                observation(
                    0,
                    timing=TimingInputs(
                        (1_800.001, 1.0), 2.0, 1.0, 900.0
                    ),
                ),
                "timing:cell_timeout",
            ),
            (
                observation(
                    0,
                    timing=TimingInputs(
                        (1.0, 1.0), 3_600.001, 1.0, 900.0
                    ),
                ),
                "timing:pair_timeout",
            ),
            (
                observation(
                    0,
                    timing=TimingInputs(
                        (1.0, 1.0), 2.0, 120.001, 900.0
                    ),
                ),
                "timing:cleanup_timeout",
            ),
            (
                observation(
                    0,
                    timing=TimingInputs(
                        (1.0, 1.0), 2.0, 1.0, 899.999
                    ),
                ),
                "timing:audit_reserve",
            ),
        )
        for value, reason in failures:
            with self.subTest(reason=reason):
                decision = evaluate_screen(contract, value)
                self.assertFalse(decision.passed)
                self.assertIn(reason, decision.reasons)

    def test_standard_promotion_boundaries_and_reverse_order(self) -> None:
        contract = policy()
        first = observation(0, (1.03, 1.03))
        reverse = observation(1, (1.03, 1.03))
        decision = evaluate_promotion(contract, first, reverse)
        self.assertTrue(decision.passed, decision.reasons)
        self.assertAlmostEqual(decision.geometric_mean_ratio, 1.03)

    def test_standard_promotion_requires_each_pair_strictly_above_one(self) -> None:
        contract = policy()
        first = observation(0, (1.07, 1.07))
        reverse = observation(1, (1.0, 1.0))
        decision = evaluate_promotion(contract, first, reverse)
        self.assertFalse(decision.passed)
        self.assertIn(
            "reverse_pair_geomean_not_strictly_above_one", decision.reasons
        )

    def test_standard_promotion_applies_guardrails_to_reverse_pair(self) -> None:
        contract = policy()
        primary_failure = evaluate_promotion(
            contract,
            observation(0, (1.20, 1.20)),
            observation(1, (0.949, 1.30)),
        )
        self.assertFalse(primary_failure.passed)
        self.assertIn(
            "reverse_primary_ratio_below_screen_floor",
            primary_failure.reasons,
        )

        ttft_failure = evaluate_promotion(
            contract,
            observation(0, (1.04, 1.04)),
            observation(1, (1.04, 1.04), ttft=1.101),
        )
        self.assertFalse(ttft_failure.passed)
        self.assertIn(
            "reverse_ttft_ratio_above_screen_ceiling",
            ttft_failure.reasons,
        )

    def test_standard_promotion_requires_both_pairs_hard_eligible(self) -> None:
        contract = policy()
        for first_gates, reverse_gates in (
            (eligible(measurement_valid=False), eligible()),
            (eligible(), eligible(ownership_ambiguous=True)),
        ):
            with self.subTest(
                first_gates=first_gates,
                reverse_gates=reverse_gates,
            ):
                decision = evaluate_promotion(
                    contract,
                    observation(0, gates=first_gates),
                    observation(1, gates=reverse_gates),
                )
                self.assertFalse(decision.passed)

    def test_standard_promotion_rejects_low_combined_or_bad_confirmation(self) -> None:
        contract = policy()
        first = observation(0, (1.031, 1.031))
        low = observation(1, (1.001, 1.001))
        decision = evaluate_promotion(contract, first, low)
        self.assertFalse(decision.passed)
        self.assertIn("combined_geomean_below_promotion_min", decision.reasons)

        nonconsecutive = observation(2, (1.04, 1.04))
        decision = evaluate_promotion(contract, observation(0), nonconsecutive)
        self.assertFalse(decision.passed)
        self.assertIn("confirmation_pair_index_not_consecutive", decision.reasons)

        mismatched_flags = observation(
            1,
            champion_flags=("--other",),
            candidate_flags=("--other",),
        )
        decision = evaluate_promotion(contract, observation(0), mismatched_flags)
        self.assertFalse(decision.passed)
        self.assertIn("confirmation_flag_bundles_mismatch", decision.reasons)

    def test_simplification_boundary_passes_with_memory_confirmed_twice(self) -> None:
        contract = policy()
        first = observation(0, (0.99, 0.99), memory_gain_gib=1.0)
        reverse = observation(1, (0.99, 0.99), memory_gain_gib=1.0)
        decision = evaluate_simplification_promotion(contract, first, reverse)
        self.assertTrue(decision.passed, decision.reasons)
        self.assertAlmostEqual(decision.geometric_mean_ratio, 0.99)

    def test_simplification_screen_accepts_one_proven_benefit(self) -> None:
        contract = policy()
        decision = evaluate_simplification_screen(
            contract,
            observation(
                0,
                (0.99, 0.99),
                champion_flags=("--a", "--b"),
                candidate_flags=("--a",),
            ),
        )

        self.assertTrue(decision.passed, decision.reasons)
        missing = evaluate_simplification_screen(
            contract, observation(0, (1.0, 1.0))
        )
        self.assertFalse(missing.passed)
        self.assertIn("simplification_benefit_missing", missing.reasons)

    def test_simplification_passes_with_strict_flags_confirmed_twice(self) -> None:
        contract = policy()
        kwargs = {
            "champion_flags": ("--a", "--b"),
            "candidate_flags": ("--a",),
        }
        decision = evaluate_simplification_promotion(
            contract,
            observation(0, (1.0, 1.0), **kwargs),
            observation(1, (1.0, 1.0), **kwargs),
        )
        self.assertTrue(decision.passed, decision.reasons)

    def test_simplification_requires_combined_speed_and_same_benefit_twice(self) -> None:
        contract = policy()
        below = evaluate_simplification_promotion(
            contract,
            observation(0, (0.98, 0.98), memory_gain_gib=1.2),
            observation(1, (0.98, 0.98), memory_gain_gib=1.2),
        )
        self.assertFalse(below.passed)
        self.assertIn(
            "combined_geomean_below_simplification_floor", below.reasons
        )
        one_memory = evaluate_simplification_promotion(
            contract,
            observation(0, (1.0, 1.0), memory_gain_gib=1.2),
            observation(1, (1.0, 1.0), memory_gain_gib=0.9),
        )
        self.assertFalse(one_memory.passed)
        self.assertIn(
            "simplification_benefit_not_confirmed_twice", one_memory.reasons
        )

    def test_simplification_applies_guardrails_to_both_pairs(self) -> None:
        contract = policy()
        first_primary_failure = evaluate_simplification_promotion(
            contract,
            observation(0, (0.949, 1.04), memory_gain_gib=1.2),
            observation(1, (1.0, 1.0), memory_gain_gib=1.2),
        )
        self.assertFalse(first_primary_failure.passed)
        self.assertIn(
            "first_primary_ratio_below_screen_floor",
            first_primary_failure.reasons,
        )

        reverse_ttft_failure = evaluate_simplification_promotion(
            contract,
            observation(0, (1.0, 1.0), memory_gain_gib=1.2),
            observation(
                1,
                (1.0, 1.0),
                ttft=1.101,
                memory_gain_gib=1.2,
            ),
        )
        self.assertFalse(reverse_ttft_failure.passed)
        self.assertIn(
            "reverse_ttft_ratio_above_screen_ceiling",
            reverse_ttft_failure.reasons,
        )

    def test_simplification_requires_both_pairs_hard_eligible(self) -> None:
        contract = policy()
        for first_gates, reverse_gates in (
            (eligible(audit_requirement_passed=False), eligible()),
            (eligible(), eligible(swap_pressure=True)),
        ):
            with self.subTest(
                first_gates=first_gates,
                reverse_gates=reverse_gates,
            ):
                decision = evaluate_simplification_promotion(
                    contract,
                    observation(
                        0,
                        (1.0, 1.0),
                        gates=first_gates,
                        memory_gain_gib=1.2,
                    ),
                    observation(
                        1,
                        (1.0, 1.0),
                        gates=reverse_gates,
                        memory_gain_gib=1.2,
                    ),
                )
                self.assertFalse(decision.passed)

    def test_calibration_inclusive_boundaries(self) -> None:
        contract = policy()
        for ratios, ttft in (
            ((0.97, 0.97), 0.90),
            ((1.03, 1.03), 1.10),
            ((0.95, 1.05), 1.0),
        ):
            with self.subTest(ratios=ratios, ttft=ttft):
                decision = evaluate_calibration(
                    contract, observation(0, ratios, ttft=ttft)
                )
                self.assertTrue(decision.passed, decision.reasons)

    def test_calibration_rejects_each_out_of_range_dimension(self) -> None:
        contract = policy()
        failures = (
            (
                observation(0, (0.969, 0.969)),
                "calibration_geomean_out_of_range",
            ),
            (
                observation(0, (1.031, 1.031)),
                "calibration_geomean_out_of_range",
            ),
            (
                observation(0, (0.949, 1.05)),
                "calibration_primary_ratio_out_of_range",
            ),
            (
                observation(0, (0.95, 1.051)),
                "calibration_primary_ratio_out_of_range",
            ),
            (
                observation(0, (1.0, 1.0), ttft=0.899),
                "calibration_ttft_ratio_out_of_range",
            ),
            (
                observation(0, (1.0, 1.0), ttft=1.101),
                "calibration_ttft_ratio_out_of_range",
            ),
        )
        for value, reason in failures:
            with self.subTest(reason=reason):
                decision = evaluate_calibration(contract, value)
                self.assertFalse(decision.passed)
                self.assertIn(reason, decision.reasons)

    def test_constants_match_audited_thresholds(self) -> None:
        self.assertEqual((SCREEN_GEOMEAN_MIN, SCREEN_PRIMARY_RATIO_MIN), (1.03, 0.95))
        self.assertEqual(SCREEN_TTFT_RATIO_MAX, 1.10)
        self.assertEqual(PROMOTION_PAIR_GEOMEAN_STRICT_MIN, 1.00)
        self.assertEqual(PROMOTION_COMBINED_GEOMEAN_MIN, 1.03)
        self.assertEqual(SIMPLIFICATION_COMBINED_GEOMEAN_MIN, 0.99)
        self.assertEqual(SIMPLIFICATION_MEMAVAILABLE_GAIN_GIB, 1.0)
        self.assertEqual(
            (CALIBRATION_GEOMEAN_MIN, CALIBRATION_GEOMEAN_MAX),
            (0.97, 1.03),
        )
        self.assertEqual(
            (CALIBRATION_PRIMARY_RATIO_MIN, CALIBRATION_PRIMARY_RATIO_MAX),
            (0.95, 1.05),
        )
        self.assertEqual(
            (CALIBRATION_TTFT_RATIO_MIN, CALIBRATION_TTFT_RATIO_MAX),
            (0.90, 1.10),
        )


class FailureDispositionTests(unittest.TestCase):
    def test_only_recovered_syntax_and_startup_errors_are_discardable(self) -> None:
        for kind in (FailureKind.CANDIDATE_SYNTAX, FailureKind.CANDIDATE_STARTUP):
            with self.subTest(kind=kind):
                self.assertEqual(
                    failure_disposition(
                        kind,
                        cleanup_verified=True,
                        restored_preflight=True,
                    ),
                    "discard_candidate",
                )
                for cleanup, preflight in ((False, True), (True, False), (False, False)):
                    self.assertEqual(
                        failure_disposition(
                            kind,
                            cleanup_verified=cleanup,
                            restored_preflight=preflight,
                        ),
                        "terminate_campaign",
                    )

    def test_every_other_failure_is_campaign_terminal(self) -> None:
        for kind in FailureKind:
            if kind in {
                FailureKind.CANDIDATE_SYNTAX,
                FailureKind.CANDIDATE_STARTUP,
            }:
                continue
            with self.subTest(kind=kind):
                self.assertEqual(
                    failure_disposition(
                        kind,
                        cleanup_verified=True,
                        restored_preflight=True,
                    ),
                    "terminate_campaign",
                )
        with self.assertRaisesRegex(AutoresearchError, "unknown failure"):
            failure_disposition(
                "unknown", cleanup_verified=True, restored_preflight=True
            )


class ReplayTests(unittest.TestCase):
    def test_empty_replay_is_new_and_immutable(self) -> None:
        state = replay_transitions(policy(), [])
        self.assertEqual(state.phase, "new")
        self.assertIsNone(state.campaign_id)
        with self.assertRaises(FrozenInstanceError):
            state.phase = "idle"  # type: ignore[misc]

    def test_standard_screen_reverse_promotion_and_completion(self) -> None:
        contract = policy()
        first = observation(0, (1.04, 1.04))
        reverse = observation(1, (1.03, 1.03))
        events = [
            campaign_started(contract),
            candidate_started(1),
            *pair_events(first, first_transition=2),
            transition(
                "autoresearch_candidate_decided",
                6,
                candidate_id="candidate-1",
                decision="confirm",
            ),
            *pair_events(reverse, first_transition=7),
            transition(
                "autoresearch_candidate_decided",
                11,
                candidate_id="candidate-1",
                decision="promote",
            ),
            transition("autoresearch_campaign_completed", 12),
        ]

        state = replay_transitions(contract, events)

        self.assertEqual(state.phase, "terminal")
        self.assertEqual(state.terminal_reason, "completed")
        self.assertEqual(state.next_pair_index, 2)
        self.assertEqual(len(state.seen_transition_ids), 13)

    def test_simplification_confirmation_and_promotion(self) -> None:
        contract = policy()
        first = observation(0, (0.99, 0.99), memory_gain_gib=1.1)
        reverse = observation(1, (0.99, 0.99), memory_gain_gib=1.2)
        events = [
            campaign_started(contract),
            candidate_started(1),
            *pair_events(first, first_transition=2),
            transition(
                "autoresearch_candidate_decided",
                6,
                candidate_id="candidate-1",
                decision="confirm_simplification",
            ),
            *pair_events(reverse, first_transition=7),
            transition(
                "autoresearch_candidate_decided",
                11,
                candidate_id="candidate-1",
                decision="promote_simplification",
            ),
        ]
        state = replay_transitions(contract, events)
        self.assertEqual(state.phase, "idle")
        self.assertEqual(state.next_pair_index, 2)

    def test_rejection_allows_next_candidate_at_alternating_global_pair(self) -> None:
        contract = policy()
        first = observation(0, (1.0, 1.0))
        second_candidate_pair = observation(1, (1.04, 1.04))
        events = [
            campaign_started(contract),
            candidate_started(1),
            *pair_events(first, first_transition=2),
            transition(
                "autoresearch_candidate_decided",
                6,
                candidate_id="candidate-1",
                decision="reject",
            ),
            candidate_started(7, "candidate-2"),
            *pair_events(
                second_candidate_pair,
                first_transition=8,
                candidate_id="candidate-2",
            ),
        ]
        state = replay_transitions(contract, events)
        self.assertEqual(state.phase, "scored")
        self.assertEqual(state.candidate_id, "candidate-2")
        self.assertEqual(state.next_pair_index, 2)

    def test_recovered_syntax_or_startup_can_discard_without_consuming_pair(self) -> None:
        contract = policy()
        for kind in ("candidate_syntax", "candidate_startup"):
            with self.subTest(kind=kind):
                events = [
                    campaign_started(contract),
                    candidate_started(1),
                    transition(
                        "autoresearch_candidate_discarded",
                        2,
                        candidate_id="candidate-1",
                        failure_kind=kind,
                        cleanup_verified=True,
                        restored_preflight=True,
                    ),
                ]
                state = replay_transitions(contract, events)
                self.assertEqual(state.phase, "idle")
                self.assertEqual(state.next_pair_index, 0)

    def test_pressure_ownership_and_cleanup_failures_are_terminal(self) -> None:
        contract = policy()
        for kind in (
            "memory_pressure",
            "swap_pressure",
            "oom",
            "ownership_ambiguity",
            "cleanup_breach",
        ):
            with self.subTest(kind=kind):
                events = [
                    campaign_started(contract),
                    candidate_started(1),
                    transition(
                        "autoresearch_campaign_terminated",
                        2,
                        failure_kind=kind,
                        cleanup_verified=kind != "cleanup_breach",
                        restored_preflight=False,
                    ),
                ]
                state = replay_transitions(contract, events)
                self.assertEqual(state.phase, "terminal")
                self.assertEqual(state.terminal_reason, kind)

    def test_cutoff_terminates_an_idle_campaign(self) -> None:
        contract = policy()
        events = [
            campaign_started(contract),
            transition(
                "autoresearch_campaign_terminated",
                1,
                failure_kind="cutoff",
                cleanup_verified=True,
                restored_preflight=True,
            ),
        ]

        state = replay_transitions(contract, events)

        self.assertEqual(state.phase, "terminal")
        self.assertEqual(state.terminal_reason, FailureKind.CUTOFF.value)
        self.assertEqual(state.next_pair_index, 0)

    def test_cutoff_cannot_discard_an_active_candidate(self) -> None:
        contract = policy()
        events = [
            campaign_started(contract),
            candidate_started(1),
            transition(
                "autoresearch_candidate_discarded",
                2,
                candidate_id="candidate-1",
                failure_kind="cutoff",
                cleanup_verified=True,
                restored_preflight=True,
            ),
        ]

        with self.assertRaisesRegex(TransitionError, "campaign-terminal"):
            replay_transitions(contract, events)

    def test_recovered_startup_must_discard_not_terminate(self) -> None:
        contract = policy()
        events = [
            campaign_started(contract),
            candidate_started(1),
            transition(
                "autoresearch_campaign_terminated",
                2,
                failure_kind="candidate_startup",
                cleanup_verified=True,
                restored_preflight=True,
            ),
        ]
        with self.assertRaisesRegex(TransitionError, "must discard"):
            replay_transitions(contract, events)

    def test_terminal_failure_cannot_be_discarded(self) -> None:
        contract = policy()
        events = [
            campaign_started(contract),
            candidate_started(1),
            transition(
                "autoresearch_candidate_discarded",
                2,
                candidate_id="candidate-1",
                failure_kind="oom",
                cleanup_verified=True,
                restored_preflight=True,
            ),
        ]
        with self.assertRaisesRegex(TransitionError, "campaign-terminal"):
            replay_transitions(contract, events)

    def test_state_machine_rejects_wrong_order_and_premature_score(self) -> None:
        contract = policy()
        base = [campaign_started(contract), candidate_started(1)]
        wrong_order = transition(
            "autoresearch_pair_started",
            2,
            candidate_id="candidate-1",
            pair_index=0,
            order=["candidate", "champion"],
        )
        with self.assertRaisesRegex(TransitionError, "alternating"):
            replay_transitions(contract, [*base, wrong_order])

        started = pair_events(observation(0), first_transition=2)[0]
        score = pair_events(observation(0), first_transition=2)[-1]
        with self.assertRaisesRegex(TransitionError, "both ordered cells"):
            replay_transitions(contract, [*base, started, score])

    def test_state_machine_rejects_wrong_cell_and_duplicate_third_cell(self) -> None:
        contract = policy()
        pair = observation(0)
        pair_parts = pair_events(pair, first_transition=2)
        wrong_cell = dict(pair_parts[1])
        wrong_cell["arm"] = "candidate"
        with self.assertRaisesRegex(TransitionError, "frozen pair order"):
            replay_transitions(
                contract,
                [campaign_started(contract), candidate_started(1), pair_parts[0], wrong_cell],
            )

        extra = transition(
            "autoresearch_cell_completed",
            5,
            candidate_id="candidate-1",
            pair_index=0,
            arm="champion",
        )
        with self.assertRaises(TransitionError):
            replay_transitions(
                contract,
                [campaign_started(contract), candidate_started(1), *pair_parts[:3], extra],
            )

    def test_state_machine_rejects_candidate_pair_and_observation_mismatches(self) -> None:
        contract = policy()
        base = [campaign_started(contract), candidate_started(1)]
        malformed = pair_events(observation(0), first_transition=2)
        malformed[0]["candidate_id"] = "other"
        with self.assertRaisesRegex(TransitionError, "does not match"):
            replay_transitions(contract, [*base, malformed[0]])

        malformed = pair_events(observation(0), first_transition=2)
        malformed[-1]["observation"] = observation(1).to_mapping()
        with self.assertRaisesRegex(TransitionError, "observation pair index"):
            replay_transitions(contract, [*base, *malformed])

        malformed = pair_events(observation(0), first_transition=2)
        malformed[-1]["observation"] = {
            **observation(0).to_mapping(),
            "raw_prompt": "must reject",
        }
        with self.assertRaisesRegex(TransitionError, "unknown keys"):
            replay_transitions(contract, [*base, *malformed])

    def test_confirmation_and_promotion_are_threshold_guarded(self) -> None:
        contract = policy()
        failed = observation(0, (1.0, 1.0))
        events = [
            campaign_started(contract),
            candidate_started(1),
            *pair_events(failed, first_transition=2),
            transition(
                "autoresearch_candidate_decided",
                6,
                candidate_id="candidate-1",
                decision="confirm",
            ),
        ]
        with self.assertRaisesRegex(TransitionError, "passing screen"):
            replay_transitions(contract, events)

        first = observation(0, (1.04, 1.04))
        reverse = observation(1, (1.0, 1.0))
        events = [
            campaign_started(contract),
            candidate_started(1),
            *pair_events(first, first_transition=2),
            transition(
                "autoresearch_candidate_decided",
                6,
                candidate_id="candidate-1",
                decision="confirm",
            ),
            *pair_events(reverse, first_transition=7),
            transition(
                "autoresearch_candidate_decided",
                11,
                candidate_id="candidate-1",
                decision="promote",
            ),
        ]
        with self.assertRaisesRegex(TransitionError, "reverse-order"):
            replay_transitions(contract, events)

    def test_ineligible_scored_pair_must_terminate_not_reject(self) -> None:
        contract = policy()
        invalid_pairs = (
            observation(0, gates=eligible(validation_passed=False)),
            observation(
                0,
                timing=TimingInputs((1_800.001, 1.0), 2.0, 1.0, 900.0),
            ),
            observation(0, case_ids=("wrong-case", "fresh-short-c1")),
        )
        for invalid in invalid_pairs:
            with self.subTest(invalid=invalid):
                events = [
                    campaign_started(contract),
                    candidate_started(1),
                    *pair_events(invalid, first_transition=2),
                    transition(
                        "autoresearch_candidate_decided",
                        6,
                        candidate_id="candidate-1",
                        decision="reject",
                    ),
                ]
                with self.assertRaisesRegex(TransitionError, "must terminate"):
                    replay_transitions(contract, events)

    def test_campaign_cannot_complete_with_active_work_or_continue_after_terminal(self) -> None:
        contract = policy()
        with self.assertRaisesRegex(TransitionError, "while idle"):
            replay_transitions(
                contract,
                [
                    campaign_started(contract),
                    candidate_started(1),
                    transition("autoresearch_campaign_completed", 2),
                ],
            )
        with self.assertRaisesRegex(TransitionError, "after campaign terminal"):
            replay_transitions(
                contract,
                [
                    campaign_started(contract),
                    transition("autoresearch_campaign_completed", 1),
                    candidate_started(2),
                ],
            )

    def test_event_registry_rejects_unknown_missing_extra_and_bad_timestamp(self) -> None:
        contract = policy()
        malformed_events = (
            {"event": "unknown", "transition_id": "t0"},
            {
                "event": "autoresearch_campaign_started",
                "transition_id": "t0",
                "campaign_id": "campaign-1",
            },
            {**campaign_started(contract), "raw": "payload"},
            {**campaign_started(contract), "timestamp": "2026-08-27T01:00:00"},
            {**campaign_started(contract), "transition_id": "Upper"},
        )
        for event in malformed_events:
            with self.subTest(event=event), self.assertRaises(TransitionError):
                replay_transitions(contract, [event])

    def test_policy_digest_and_candidate_axis_are_bound(self) -> None:
        contract = policy()
        bad_digest = dict(campaign_started(contract))
        bad_digest["policy_digest"] = "b" * 64
        with self.assertRaisesRegex(TransitionError, "digest"):
            replay_transitions(contract, [bad_digest])

        bad_axis = candidate_started(1)
        bad_axis["axis"] = "unapproved"
        with self.assertRaisesRegex(TransitionError, "not allowed"):
            replay_transitions(contract, [campaign_started(contract), bad_axis])

    def test_identical_transition_retry_is_idempotent_but_collision_fails(self) -> None:
        contract = policy()
        started = campaign_started(contract)
        duplicate = {**started, "timestamp": "2026-08-27T01:00:00+00:00"}
        state = replay_transitions(contract, [started, duplicate])
        self.assertEqual(state.phase, "idle")
        self.assertEqual(state.seen_transition_ids, ("t0",))

        collision = dict(started)
        collision["campaign_id"] = "campaign-2"
        with self.assertRaisesRegex(TransitionError, "collision"):
            replay_transitions(contract, [started, collision])

    def test_append_transition_uses_journal_fsync_shape_and_is_idempotent(self) -> None:
        contract = policy()
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "events.jsonl")
            event = campaign_started(contract)

            first = append_transition(journal, contract, event)
            second = append_transition(journal, contract, event)

            self.assertEqual(first.phase, "idle")
            self.assertEqual(second.phase, "idle")
            events = journal.events()
            self.assertEqual(len(events), 1)
            self.assertIn("timestamp", events[0])
            with self.assertRaisesRegex(TransitionError, "must not provide"):
                append_transition(
                    journal,
                    contract,
                    {**candidate_started(1), "timestamp": "2026-08-27T01:00:00+00:00"},
                )
            self.assertEqual(len(journal.events()), 1)

    def test_invalid_append_does_not_mutate_journal(self) -> None:
        contract = policy()
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "events.jsonl")
            append_transition(journal, contract, campaign_started(contract))
            before = journal.path.read_bytes()
            invalid = transition(
                "autoresearch_pair_started",
                1,
                candidate_id="candidate-1",
                pair_index=0,
                order=["champion", "candidate"],
            )
            with self.assertRaises(TransitionError):
                append_transition(journal, contract, invalid)
            self.assertEqual(journal.path.read_bytes(), before)

    def test_append_replays_past_a_torn_json_tail(self) -> None:
        contract = policy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            journal = Journal(path)
            append_transition(journal, contract, campaign_started(contract))
            with path.open("a", encoding="utf-8") as stream:
                stream.write('{"event":"torn"')

            state = append_transition(journal, contract, candidate_started(1))

            self.assertEqual(state.phase, "candidate")
            intact = journal.events()
            self.assertEqual(
                [event["event"] for event in intact],
                [
                    "autoresearch_campaign_started",
                    "autoresearch_candidate_started",
                ],
            )


if __name__ == "__main__":
    unittest.main()
