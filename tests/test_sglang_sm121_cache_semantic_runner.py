"""Offline execution seams for the paired SM121 semantic-cache canary."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from bench import runner, runtime
from bench.journal import Journal
from bench.manifest import load_models, load_suite
from bench.sglang_sm121_cache_semantic import (
    SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID,
    SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID,
    SM121_CACHE_SEMANTIC_CASE_ID,
    SM121_CACHE_SEMANTIC_COLD_INPUT_MIN_TOKENS,
    SM121_CACHE_SEMANTIC_TURN_OBSERVATION_EVENT,
    SM121_CACHE_SEMANTIC_TURN_ORDER,
    sm121_cache_semantic_cache_off_receipt_sha256,
    sm121_cache_semantic_pair_instance_sha256,
    validate_sm121_cache_semantic_pair_binding,
)
from bench.sglang_sm121_storage import (
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_SOURCE_TREE,
)
from sparkbench import (
    build_parser,
    command_audit_sm121_cache_policy_semantic,
    command_sm121_cache_policy_semantic_canary,
)
from bench.telemetry import TelemetrySampler


def _snapshot(
    *, input_tokens: int, device_hits: int = 0, cached_device: int = 0
) -> dict[str, object]:
    snapshot = runtime._sm121_cache_metric_defaults()
    snapshot.update(
        {
            "available": True,
            "guardrail_metrics_available": True,
            "prefill_input_tokens": input_tokens,
            "prefill_device_hit_tokens": device_hits,
            "cached_device_tokens": cached_device,
            "cached_device_series_present": cached_device > 0,
        }
    )
    return snapshot


def _result(prompt_token_ids: tuple[int, ...], *, device_hits: int = 0) -> dict[str, object]:
    detail_state = "nonzero_details" if device_hits else "zero_details"
    return {
        "private_prompt_token_ids": prompt_token_ids,
        "prompt_tokens": len(prompt_token_ids),
        "completion_tokens": 1,
        "reasoning_tokens": 0,
        "response_detail_state": detail_state,
        "response_device_cached_tokens": device_hits,
        "response_host_cached_tokens": 0,
        "response_storage_cached_tokens": 0,
        "usage_detail_state": detail_state,
        "usage_cached_tokens": device_hits,
    }


class SM121CacheSemanticRunnerTests(unittest.TestCase):
    def _case(self) -> SimpleNamespace:
        return SimpleNamespace(
            id=SM121_CACHE_SEMANTIC_CASE_ID,
            case_id=SM121_CACHE_SEMANTIC_CASE_ID + "--0123456789ab",
            kind="capability",
            concurrency=1,
            max_output_tokens=32,
        )

    def test_cache_off_turns_are_scalar_only_and_adjacent_to_requests(self) -> None:
        base = tuple(range(SM121_CACHE_SEMANTIC_COLD_INPUT_MIN_TOKENS + 8))
        token_ids = (
            base,
            base + (100_000,),
            base + (100_000, 100_001),
        )
        snapshots = []
        total_input = 0
        for ids in token_ids:
            snapshots.extend(
                (
                    _snapshot(input_tokens=total_input),
                    _snapshot(input_tokens=total_input + len(ids)),
                )
            )
            total_input += len(ids)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = Journal(root / "events.jsonl")
            telemetry = TelemetrySampler(root / "telemetry.jsonl")
            with (
                mock.patch(
                    "bench.runner.settle_sm121_cache_observability_metrics",
                    side_effect=[(value, 0.0, 2, True) for value in snapshots],
                ) as settle,
                mock.patch(
                    "bench.runner.request_sm121_cache_semantic_turn",
                    side_effect=[_result(ids) for ids in token_ids],
                ),
            ):
                observed = runner._execute_sm121_cache_semantic_case(
                    server=SimpleNamespace(),
                    model=SimpleNamespace(served_name="synthetic-model"),
                    case=self._case(),
                    arm="B",
                    control_prompt_token_ids=None,
                    journal=journal,
                    telemetry=telemetry,
                )

            events = journal.events()
            turns = [
                event
                for event in events
                if event.get("event") == SM121_CACHE_SEMANTIC_TURN_OBSERVATION_EVENT
            ]
            self.assertEqual(observed, token_ids)
            self.assertEqual(
                [call.kwargs for call in settle.call_args_list],
                [{"semantic_arm": "B"}] * (2 * len(token_ids)),
            )
            self.assertEqual([event["turn"] for event in turns], list(SM121_CACHE_SEMANTIC_TURN_ORDER))
            self.assertTrue(all(event["semantic_turn_admitted"] for event in turns))
            self.assertEqual([event["shared_prefix_tokens"] for event in turns], [0, len(base), len(base) + 1])
            for index, event in enumerate(events):
                if event.get("event") == SM121_CACHE_SEMANTIC_TURN_OBSERVATION_EVENT:
                    self.assertEqual(events[index + 1].get("event"), "request_complete")
            serialized = (root / "events.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("private_prompt_token_ids", serialized)
            self.assertNotIn("shared-ledger-entry", serialized)
            self.assertNotIn("SEMANTIC-CACHE", serialized)

    def test_cache_on_rejects_private_cross_arm_token_mismatch_without_journaling_ids(self) -> None:
        base = tuple(range(SM121_CACHE_SEMANTIC_COLD_INPUT_MIN_TOKENS + 8))
        observed_ids = (base, base + (100_000,), base + (100_000, 100_001))
        control_ids = (base + (7,), observed_ids[1], observed_ids[2])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = Journal(root / "events.jsonl")
            with (
                mock.patch(
                    "bench.runner.settle_sm121_cache_observability_metrics",
                    return_value=(_snapshot(input_tokens=0), 0.0, 2, True),
                ),
                mock.patch(
                    "bench.runner.request_sm121_cache_semantic_turn",
                    return_value=_result(observed_ids[0]),
                ),
                self.assertRaises(runner.SM121CacheSemanticRequestError),
            ):
                runner._execute_sm121_cache_semantic_case(
                    server=SimpleNamespace(),
                    model=SimpleNamespace(served_name="synthetic-model"),
                    case=self._case(),
                    arm="A",
                    control_prompt_token_ids=control_ids,
                    journal=journal,
                    telemetry=TelemetrySampler(root / "telemetry.jsonl"),
                )
            serialized = (root / "events.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("100000", serialized)
            self.assertNotIn("private_prompt_token_ids", serialized)

    def test_pair_plans_have_reciprocal_scalar_binding(self) -> None:
        workspace = Path(__file__).resolve().parents[1]
        models = load_models(workspace / "manifests" / "models.toml")
        suite_path = (
            workspace
            / "manifests"
            / "suites"
            / "qwen38_flash_next_sm121_triton_storage_cache_policy_semantic_canary.toml"
        )
        suite = load_suite(suite_path)
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch("bench.runner._image_digest", return_value=None),
                mock.patch(
                    "bench.runner._sm121_storage_image_identity",
                    return_value={
                        "docker_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
                        "platform": SM121_STORAGE_PLATFORM,
                        "source_tree": SM121_STORAGE_SOURCE_TREE,
                    },
                ),
                mock.patch("bench.runner._host_snapshot", return_value={"host": "test"}),
            ):
                cache_off_run, cache_on_run = runner.create_sm121_cache_semantic_pair_plans(
                    cache_off_model=models[SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID],
                    cache_on_model=models[SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID],
                    suite=suite,
                    results_root=Path(directory),
                    models_path=workspace / "manifests" / "models.toml",
                    suite_path=suite_path,
                )
            cache_off_plan, cache_off_model, cache_off_suite = (
                runner._load_sm121_cache_semantic_plan(cache_off_run)
            )
            cache_on_plan, cache_on_model, cache_on_suite = (
                runner._load_sm121_cache_semantic_plan(cache_on_run)
            )
            runner._validate_sm121_cache_semantic_pair_plans(
                cache_off_plan,
                cache_off_model,
                cache_off_suite,
                cache_on_plan,
                cache_on_model,
                cache_on_suite,
            )
            self.assertEqual(
                cache_off_plan["semantic_pair"]["peer_plan_fingerprint"],
                cache_on_plan["fingerprint"],
            )
            self.assertEqual(
                cache_on_plan["semantic_pair"]["peer_plan_fingerprint"],
                cache_off_plan["fingerprint"],
            )
            expected_instance = sm121_cache_semantic_pair_instance_sha256(
                cache_off_plan["run_nonce"], cache_on_plan["run_nonce"]
            )
            self.assertEqual(
                cache_off_plan["semantic_pair"]["pair_instance_sha256"],
                expected_instance,
            )
            self.assertEqual(
                cache_on_plan["semantic_pair"]["pair_instance_sha256"],
                expected_instance,
            )
            validate_sm121_cache_semantic_pair_binding(
                cache_off_plan["semantic_pair"],
                cache_off_model,
                cache_off_suite,
                peer_plan_fingerprint=cache_on_plan["fingerprint"],
                peer_binding=cache_on_plan["semantic_pair"],
            )

    def test_cli_exposes_only_dedicated_paired_entrypoints(self) -> None:
        parser = build_parser()
        run_args = parser.parse_args(["sm121-cache-policy-semantic-canary"])
        audit_args = parser.parse_args(
            ["audit-sm121-cache-policy-semantic", "cache-off", "cache-on"]
        )
        self.assertIs(run_args.function, command_sm121_cache_policy_semantic_canary)
        self.assertIs(audit_args.function, command_audit_sm121_cache_policy_semantic)

    def test_pair_controller_blocks_a_until_b_lifecycle_is_complete(self) -> None:
        cache_off_plan = {"fingerprint": "b" * 16}
        cache_on_plan = {"fingerprint": "a" * 16}
        control_ids = ((1,), (1, 2), (1, 2, 3))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch(
                    "bench.runner._load_sm121_cache_semantic_plan",
                    side_effect=[
                        (cache_off_plan, SimpleNamespace(), SimpleNamespace()),
                        (cache_on_plan, SimpleNamespace(), SimpleNamespace()),
                    ],
                ),
                mock.patch("bench.runner._validate_sm121_cache_semantic_pair_plans"),
                mock.patch(
                    "bench.runner._execute_sm121_cache_semantic_arm",
                    return_value=({"status": "partial"}, control_ids),
                ) as execute_arm,
                mock.patch("bench.runner._sm121_cache_semantic_arm_complete", return_value=False),
            ):
                result = runner.execute_sm121_cache_semantic_canary(
                    root / "cache-off",
                    root / "cache-on",
                    workspace=root,
                )
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["cache_on_started"])
        execute_arm.assert_called_once()
        self.assertIsNone(execute_arm.call_args.kwargs["control_prompt_token_ids"])

    def test_pair_controller_passes_private_b_identity_only_to_a(self) -> None:
        cache_off_binding = {"pair_binding_sha256": "sha256:" + "b" * 64}
        cache_on_binding = {"pair_instance_sha256": "sha256:" + "a" * 64}
        cache_off_plan = {
            "fingerprint": "b" * 16,
            "semantic_pair": cache_off_binding,
        }
        cache_on_plan = {
            "fingerprint": "a" * 16,
            "semantic_pair": cache_on_binding,
        }
        control_ids = ((1,), (1, 2), (1, 2, 3))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch(
                    "bench.runner._load_sm121_cache_semantic_plan",
                    side_effect=[
                        (cache_off_plan, SimpleNamespace(), SimpleNamespace()),
                        (cache_on_plan, SimpleNamespace(), SimpleNamespace()),
                    ],
                ),
                mock.patch("bench.runner._validate_sm121_cache_semantic_pair_plans"),
                mock.patch(
                    "bench.runner._execute_sm121_cache_semantic_arm",
                    side_effect=[
                        ({"status": "complete"}, control_ids),
                        ({"status": "complete"}, control_ids),
                    ],
                ) as execute_arm,
                mock.patch(
                    "bench.runner._sm121_cache_semantic_arm_complete",
                    side_effect=[True, True],
                ),
            ):
                result = runner.execute_sm121_cache_semantic_canary(
                    root / "cache-off",
                    root / "cache-on",
                    workspace=root,
                )
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["cache_on_started"])
        self.assertEqual(execute_arm.call_count, 2)
        cache_off_call, cache_on_call = execute_arm.call_args_list
        self.assertIsNone(cache_off_call.kwargs["control_prompt_token_ids"])
        self.assertEqual(cache_on_call.kwargs["control_prompt_token_ids"], control_ids)
        self.assertEqual(cache_on_call.kwargs["cache_off_plan_fingerprint"], "b" * 16)
        self.assertIs(cache_on_call.kwargs["cache_off_audit_passed"], True)
        self.assertEqual(
            cache_on_call.kwargs["cache_off_terminal_receipt_sha256"],
            sm121_cache_semantic_cache_off_receipt_sha256(
                cache_on_binding["pair_instance_sha256"],
                cache_off_plan["fingerprint"],
                cache_off_binding["pair_binding_sha256"],
            ),
        )


if __name__ == "__main__":
    unittest.main()
