"""Offline freeze and controller tests for SM121 cache-performance ABBA runs."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.journal import Journal, content_hash
from bench.manifest import load_models, load_suite
from bench.runner import (
    PreflightError,
    SM121CachePerformanceRequestError,
    _execute_sm121_cache_performance_arm,
    _execute_sm121_cache_performance_quality_case,
    _load_sm121_cache_performance_campaign,
    _sm121_cache_performance_turn_event,
    _sm121_cache_performance_interrupt_terminal_server,
    create_sm121_cache_performance_campaign,
    execute_sm121_cache_performance_campaign,
)
from bench.sglang_sm121_cache_observability import (
    SM121_CACHE_OBSERVABILITY_CACHED_SERIES,
)
from bench.sglang_sm121_cache_performance import (
    SM121_CACHE_PERFORMANCE_ARM_ORDER,
    SM121_CACHE_PERFORMANCE_CASE_ID,
    SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID,
    SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID,
    SM121_CACHE_PERFORMANCE_METRIC_FIELDS,
    SM121_CACHE_PERFORMANCE_QUALITY_CASE_ID,
    SM121_CACHE_PERFORMANCE_TIMED_TURNS,
)
from bench.sglang_sm121_storage import (
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_SOURCE_TREE,
)
from sparkbench import (
    DEFAULT_SM121_CACHE_PERFORMANCE_SUITE,
    build_parser,
    command_audit_sm121_cache_policy_performance,
    command_sm121_cache_policy_performance,
)
from tests.test_sglang_sm121_cache_performance import _lifetime


class SM121CachePerformanceRunnerTests(unittest.TestCase):
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

    def test_cli_exposes_dedicated_non_resumable_campaign_and_audit(self) -> None:
        parser = build_parser()
        run_args = parser.parse_args(["sm121-cache-policy-performance"])
        self.assertEqual(DEFAULT_SM121_CACHE_PERFORMANCE_SUITE, run_args.suite)
        self.assertEqual(command_sm121_cache_policy_performance, run_args.function)
        self.assertFalse(hasattr(run_args, "allow_download"))
        audit_args = parser.parse_args(
            ["audit-sm121-cache-policy-performance", "synthetic-campaign"]
        )
        self.assertEqual(Path("synthetic-campaign"), audit_args.campaign_dir)
        self.assertEqual(
            command_audit_sm121_cache_policy_performance, audit_args.function
        )

    def _freeze(self, root: Path) -> Path:
        with (
            patch("bench.runner._validate_sm121_cache_performance_prerequisites"),
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
                results_root=root / "cache-policy-campaigns",
                models_path=self.repository / "manifests" / "models.toml",
                suite_path=self.suite_path,
                evidence_root=root / "evidence",
            )

    def test_freeze_binds_four_unique_labeled_plans_and_rejects_nonce_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign_dir = self._freeze(root)
            campaign = json.loads((campaign_dir / "campaign.json").read_text())
            names = campaign["run_directories"]
            self.assertEqual(4, len(names))
            self.assertEqual(4, len(set(names)))
            self.assertTrue(
                all(
                    name.endswith(label)
                    for name, label in zip(
                        names,
                        (
                            "-performance-1-a",
                            "-performance-2-b",
                            "-performance-3-b",
                            "-performance-4-a",
                        ),
                        strict=True,
                    )
                )
            )
            with patch("bench.runner._validate_sm121_cache_performance_prerequisites"):
                _campaign, loaded = _load_sm121_cache_performance_campaign(
                    campaign_dir, evidence_root=root / "evidence"
                )
            self.assertEqual(
                list(SM121_CACHE_PERFORMANCE_ARM_ORDER),
                [
                    "A" if model.id == SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID else "B"
                    for _run_dir, _plan, model, _suite in loaded
                ],
            )
            plan_path = campaign_dir / "runs" / names[0] / "plan.json"
            plan = json.loads(plan_path.read_text())
            plan["run_nonce"] = "f" * 32
            plan["integrity_hash"] = content_hash(
                {key: value for key, value in plan.items() if key != "integrity_hash"},
                64,
            )
            plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n")
            with patch("bench.runner._validate_sm121_cache_performance_prerequisites"):
                with self.assertRaisesRegex(PreflightError, "binding is invalid"):
                    _load_sm121_cache_performance_campaign(
                        campaign_dir, evidence_root=root / "evidence"
                    )

    @staticmethod
    def _loaded_campaign(root: Path) -> tuple[dict[str, object], list[tuple[Path, dict[str, object], SimpleNamespace, SimpleNamespace]]]:
        loaded: list[tuple[Path, dict[str, object], SimpleNamespace, SimpleNamespace]] = []
        for ordinal, arm in enumerate(SM121_CACHE_PERFORMANCE_ARM_ORDER, start=1):
            run_dir = root / f"run-{ordinal}-{arm.lower()}"
            run_dir.mkdir()
            profile_id = (
                SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID
                if arm == "A"
                else SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID
            )
            loaded.append(
                (
                    run_dir,
                    {"fingerprint": f"{ordinal:x}" * 16},
                    SimpleNamespace(id=profile_id),
                    SimpleNamespace(),
                )
            )
        return (
            {"pair_binding": {"pair_binding_sha256": "sha256:" + "a" * 64}},
            loaded,
        )

    def test_controller_runs_abba_and_stops_after_first_terminal_arm(self) -> None:
        for fail_ordinal in (None, 2):
            with self.subTest(fail_ordinal=fail_ordinal), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                campaign_dir = root / "campaign"
                campaign_dir.mkdir()
                campaign, loaded = self._loaded_campaign(root)
                calls: list[tuple[int, str, tuple[tuple[int, ...], ...] | None]] = []
                private_ids = ((101,), (202,), (303,))

                def execute_arm(**kwargs: object):
                    ordinal = int(kwargs["campaign_ordinal"])
                    model = kwargs["model"]
                    reference = kwargs["reference_prompt_token_ids"]
                    arm = "A" if model.id == SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID else "B"
                    calls.append((ordinal, arm, reference))
                    if ordinal == fail_ordinal:
                        return (
                            {
                                "ordinal": ordinal,
                                "arm": arm,
                                "quality_admitted": False,
                                "timed_admitted": False,
                                "within_timeout": False,
                                "turns": [],
                            },
                            reference,
                            False,
                        )
                    return (
                        _lifetime(ordinal, arm, t0=20.0, later=100.0),
                        private_ids,
                        True,
                    )

                with (
                    patch(
                        "bench.runner._load_sm121_cache_performance_campaign",
                        return_value=(campaign, loaded),
                    ),
                    patch(
                        "bench.runner._execute_sm121_cache_performance_arm",
                        side_effect=execute_arm,
                    ),
                ):
                    summary = execute_sm121_cache_performance_campaign(
                        campaign_dir,
                        workspace=root / "workspace",
                        evidence_root=root / "evidence",
                    )
                expected_ordinals = [1, 2, 3, 4] if fail_ordinal is None else [1, 2]
                self.assertEqual(expected_ordinals, [ordinal for ordinal, _arm, _ref in calls])
                self.assertEqual(
                    list(SM121_CACHE_PERFORMANCE_ARM_ORDER[: len(calls)]),
                    [arm for _ordinal, arm, _ref in calls],
                )
                self.assertIsNone(calls[0][2])
                if len(calls) > 1:
                    self.assertEqual(private_ids, calls[1][2])
                self.assertEqual("complete" if fail_ordinal is None else "partial", summary["status"])
                self.assertTrue((campaign_dir / "summary.json").is_file())
                self.assertFalse(
                    any(
                        (run_dir / "events.jsonl").exists()
                        for run_dir, _plan, _model, _suite in loaded
                    )
                )

    def test_terminal_cleanup_interrupts_only_failed_or_expired_server(self) -> None:
        server = Mock()
        with patch("bench.runner.time.monotonic", return_value=10.0):
            _sm121_cache_performance_interrupt_terminal_server(
                server=server, deadline=20.0, terminal_error=None
            )
        server.interrupt_owned.assert_not_called()
        with patch("bench.runner.time.monotonic", return_value=20.0):
            _sm121_cache_performance_interrupt_terminal_server(
                server=server, deadline=20.0, terminal_error=None
            )
        _sm121_cache_performance_interrupt_terminal_server(
            server=server,
            deadline=20.0,
            terminal_error=SM121CachePerformanceRequestError(),
        )
        self.assertEqual(2, server.interrupt_owned.call_count)

    def test_turn_builder_seeds_derived_fields_before_admission(self) -> None:
        source = _lifetime(1, "A", t0=20.0, later=100.0)["turns"][0]
        result = {
            field: source[field]
            for field in (
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "response_detail_state",
                "usage_detail_state",
                "response_device_cached_tokens",
                "response_host_cached_tokens",
                "response_storage_cached_tokens",
                "usage_cached_tokens",
            )
        }
        before = {
            "available": True,
            "guardrail_metrics_available": True,
        }
        after = dict(before)
        for metric in SM121_CACHE_PERFORMANCE_METRIC_FIELDS:
            before[metric] = source[f"before_{metric}"]
            after[metric] = source[f"after_{metric}"]
        for cache_source in SM121_CACHE_OBSERVABILITY_CACHED_SERIES:
            before[f"cached_{cache_source}_series_present"] = source[
                f"before_cached_{cache_source}_series_present"
            ]
            after[f"cached_{cache_source}_series_present"] = source[
                f"after_cached_{cache_source}_series_present"
            ]
        event = _sm121_cache_performance_turn_event(
            case=SimpleNamespace(case_id=source["case_id"]),
            arm="A",
            lifetime_ordinal=2,
            turn="T0",
            result=result,
            request_wall_s=20.0,
            before=before,
            before_polls=2,
            before_settled=True,
            after=after,
            after_polls=2,
            after_settled=True,
            append_only_prompt_identity_verified=True,
            cross_lifetime_prompt_identity_verified=True,
            shared_prefix_tokens=0,
        )
        self.assertTrue(event["timed_turn_admitted"])
        self.assertEqual("admitted", event["timed_turn_basis"])

    def test_quality_gate_reuses_the_admitted_v2_prompt_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompt_tags: list[str] = []
            answers = iter(("83", "no", "silver", "9"))

            def request_arguments(**kwargs: object) -> dict[str, object]:
                prompt_tags.append(str(kwargs["prompt_tag"]))
                return {}

            with (
                patch(
                    "bench.runner._quality_request_arguments",
                    side_effect=request_arguments,
                ),
                patch(
                    "bench.runner.stream_chat_request",
                    side_effect=lambda **_kwargs: SimpleNamespace(
                        content="FINAL: " + next(answers)
                    ),
                ),
            ):
                _execute_sm121_cache_performance_quality_case(
                    server=SimpleNamespace(),
                    model=SimpleNamespace(),
                    case=SimpleNamespace(case_id="quality-case"),
                    journal=Journal(Path(directory) / "events.jsonl"),
                    arm="A",
                    lifetime_ordinal=1,
                    watchdog=None,
                    deadline=1_000_000_000_000.0,
                )
            self.assertEqual(["r0"] * 4, prompt_tags)

    def test_terminal_timed_prefix_is_recovered_without_private_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            quality_case = SimpleNamespace(
                id=SM121_CACHE_PERFORMANCE_QUALITY_CASE_ID,
                case_id=SM121_CACHE_PERFORMANCE_QUALITY_CASE_ID + "--0123456789ab",
                requires=("chat",),
                max_output_tokens=512,
            )
            timed_case = SimpleNamespace(
                id=SM121_CACHE_PERFORMANCE_CASE_ID,
                case_id=SM121_CACHE_PERFORMANCE_CASE_ID + "--0123456789ab",
                requires=("chat",),
                max_output_tokens=32,
            )
            model = SimpleNamespace(
                id=SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID,
                tasks=("chat",),
                max_context=65_536,
                cache_performance_pair={"pair_binding_sha256": "sha256:" + "a" * 64},
            )
            expected_turns = []

            def timed_failure(**kwargs: object):
                journal = kwargs["journal"]
                case = kwargs["case"]
                for turn in ("T0", "T1"):
                    event = _lifetime(1, "A", t0=20.0, later=100.0)["turns"][
                        SM121_CACHE_PERFORMANCE_TIMED_TURNS.index(turn)
                    ]
                    event["case_id"] = case.case_id
                    expected_turns.append(event)
                    journal.append(event)
                raise SM121CachePerformanceRequestError()

            with (
                patch("bench.runner._preflight"),
                patch(
                    "bench.runner._execute_sm121_cache_performance_quality_lifetime",
                    return_value=1.0,
                ),
                patch(
                    "bench.runner._execute_sm121_cache_performance_timed_lifetime",
                    side_effect=timed_failure,
                ),
            ):
                lifetime, private_ids, completed = _execute_sm121_cache_performance_arm(
                    run_dir=run_dir,
                    plan={"fingerprint": "0" * 16},
                    model=model,
                    suite=SimpleNamespace(cases=(quality_case, timed_case)),
                    campaign_ordinal=1,
                    workspace=run_dir / "workspace",
                    reference_prompt_token_ids=None,
                )
            self.assertFalse(completed)
            self.assertIsNone(private_ids)
            self.assertTrue(lifetime["quality_admitted"])
            self.assertFalse(lifetime["timed_admitted"])
            self.assertEqual(expected_turns, lifetime["turns"])
            self.assertTrue(
                all("timestamp" not in event for event in lifetime["turns"])
            )
            raw = (run_dir / "events.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("private_prompt_token_ids", raw)


if __name__ == "__main__":
    unittest.main()
