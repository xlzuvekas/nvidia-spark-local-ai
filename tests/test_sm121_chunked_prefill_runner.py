"""Offline controller tests for SM121 chunk-size A/B/B/A campaigns."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.journal import content_hash
from bench.manifest import load_models, load_suite
from bench.runner import PreflightError
from bench.sglang_sm121_chunked_prefill_performance import (
    SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_MAX_MAMBA_CACHE_SIZE,
    SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EXPECTED,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V1_STUDY,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CAMPAIGN_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CANDIDATE_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CONTROL_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_V2_STUDY,
)
from bench.sglang_sm121_storage import (
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_SOURCE_TREE,
)
from bench.sm121_chunked_prefill_runner import (
    _load_campaign,
    _runtime_event,
    SM121ChunkedPrefillPerformanceRequestError,
    create_sm121_chunked_prefill_performance_campaign,
    execute_sm121_chunked_prefill_performance_campaign,
)
from sparkbench import (
    DEFAULT_SM121_CHUNKED_PREFILL_PERFORMANCE_SUITE,
    DEFAULT_SM121_CHUNKED_PREFILL_PERFORMANCE_V2_SUITE,
    build_parser,
    command_audit_sm121_chunked_prefill_performance,
    command_sm121_chunked_prefill_performance,
    command_sm121_chunked_prefill_performance_v2,
)
from tests.test_sglang_sm121_chunked_prefill_performance import _lifetime


class SM121ChunkedPrefillRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = Path(__file__).resolve().parents[1]
        models = load_models(self.repository / "manifests" / "models.toml")
        self.control_model = models[
            SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID
        ]
        self.candidate_model = models[
            SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID
        ]
        self.suite_path = (
            self.repository
            / "manifests"
            / "suites"
            / "qwen38_flash_next_sm121_triton_storage_chunked_prefill_performance_v1.toml"
        )
        self.suite = load_suite(self.suite_path)
        self.v2_control_model = models[
            SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CONTROL_PROFILE_ID
        ]
        self.v2_candidate_model = models[
            SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CANDIDATE_PROFILE_ID
        ]
        self.v2_suite_path = (
            self.repository
            / "manifests"
            / "suites"
            / "qwen38_flash_next_sm121_triton_storage_chunked_prefill_performance_v2.toml"
        )
        self.v2_suite = load_suite(self.v2_suite_path)

    def test_cli_exposes_dedicated_non_resumable_campaign_and_audit(self) -> None:
        parser = build_parser()
        run_args = parser.parse_args(["sm121-chunked-prefill-performance"])
        self.assertEqual(DEFAULT_SM121_CHUNKED_PREFILL_PERFORMANCE_SUITE, run_args.suite)
        self.assertEqual(command_sm121_chunked_prefill_performance, run_args.function)
        self.assertFalse(hasattr(run_args, "allow_download"))
        v2_args = parser.parse_args(["sm121-chunked-prefill-performance-v2"])
        self.assertEqual(DEFAULT_SM121_CHUNKED_PREFILL_PERFORMANCE_V2_SUITE, v2_args.suite)
        self.assertEqual(command_sm121_chunked_prefill_performance_v2, v2_args.function)
        self.assertFalse(hasattr(v2_args, "allow_download"))
        audit_args = parser.parse_args(
            ["audit-sm121-chunked-prefill-performance", "synthetic-campaign"]
        )
        self.assertEqual(Path("synthetic-campaign"), audit_args.campaign_dir)
        self.assertEqual(
            command_audit_sm121_chunked_prefill_performance, audit_args.function
        )

    def _freeze(self, root: Path) -> Path:
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
            return create_sm121_chunked_prefill_performance_campaign(
                control_model=self.control_model,
                candidate_model=self.candidate_model,
                suite=self.suite,
                results_root=root / "chunked-prefill-campaigns",
                models_path=self.repository / "manifests" / "models.toml",
                suite_path=self.suite_path,
            )

    def test_freeze_binds_four_unique_plans_and_rejects_nonce_tamper(self) -> None:
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
                        ("-prefill-1-a", "-prefill-2-b", "-prefill-3-b", "-prefill-4-a"),
                        strict=True,
                    )
                )
            )
            _campaign, study, loaded = _load_campaign(campaign_dir)
            self.assertEqual(SM121_CHUNKED_PREFILL_PERFORMANCE_V1_STUDY, study)
            self.assertEqual(
                list(SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER),
                [
                    "A"
                    if model.id
                    == SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID
                    else "B"
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
            with self.assertRaisesRegex(PreflightError, "binding is invalid"):
                _load_campaign(campaign_dir)

    def test_v2_freeze_is_separate_and_binds_2k_4k(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
                campaign_dir = create_sm121_chunked_prefill_performance_campaign(
                    control_model=self.v2_control_model,
                    candidate_model=self.v2_candidate_model,
                    suite=self.v2_suite,
                    results_root=root / "chunked-prefill-campaigns",
                    models_path=self.repository / "manifests" / "models.toml",
                    suite_path=self.v2_suite_path,
                )
            campaign = json.loads((campaign_dir / "campaign.json").read_text())
            self.assertEqual(SM121_CHUNKED_PREFILL_PERFORMANCE_V2_CAMPAIGN_ID, campaign["campaign_id"])
            self.assertEqual([2048, 4096], campaign["pair_binding"]["chunked_prefill_sizes"])
            _campaign, study, _loaded = _load_campaign(campaign_dir)
            self.assertEqual(SM121_CHUNKED_PREFILL_PERFORMANCE_V2_STUDY, study)

    def test_v2_runtime_attestation_rejects_a_valid_but_wrong_study_chunk(self) -> None:
        observed = {
            "mamba_radix_cache_strategy": "extra_buffer_lazy",
            "max_mamba_cache_size": SM121_CHUNKED_PREFILL_PERFORMANCE_MAX_MAMBA_CACHE_SIZE,
            **SM121_CHUNKED_PREFILL_PERFORMANCE_RUNTIME_EXPECTED,
        }
        with patch(
            "bench.sm121_chunked_prefill_runner.inspect_sm121_chunked_prefill_runtime_identity",
            return_value={**observed, "chunked_prefill_size": 2048},
        ):
            event = _runtime_event(
                server=object(),
                study=SM121_CHUNKED_PREFILL_PERFORMANCE_V2_STUDY,
                arm="A",
                lifetime_ordinal=1,
            )
        self.assertEqual(2048, event["chunked_prefill_size"])
        with patch(
            "bench.sm121_chunked_prefill_runner.inspect_sm121_chunked_prefill_runtime_identity",
            return_value={**observed, "chunked_prefill_size": 1024},
        ):
            with self.assertRaises(SM121ChunkedPrefillPerformanceRequestError):
                _runtime_event(
                    server=object(),
                    study=SM121_CHUNKED_PREFILL_PERFORMANCE_V2_STUDY,
                    arm="A",
                    lifetime_ordinal=1,
                )

    @staticmethod
    def _loaded_campaign(
        root: Path,
    ) -> tuple[
        dict[str, object],
        object,
        list[tuple[Path, dict[str, object], SimpleNamespace, SimpleNamespace]],
    ]:
        loaded: list[
            tuple[Path, dict[str, object], SimpleNamespace, SimpleNamespace]
        ] = []
        for ordinal, arm in enumerate(
            SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER, start=1
        ):
            run_dir = root / f"run-{ordinal}-{arm.lower()}"
            run_dir.mkdir()
            profile_id = (
                SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID
                if arm == "A"
                else SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID
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
            SM121_CHUNKED_PREFILL_PERFORMANCE_V1_STUDY,
            loaded,
        )

    def test_controller_runs_abba_and_stops_after_first_terminal_arm(self) -> None:
        for fail_ordinal in (None, 2):
            with self.subTest(fail_ordinal=fail_ordinal), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                campaign_dir = root / "campaign"
                campaign_dir.mkdir()
                campaign, study, loaded = self._loaded_campaign(root)
                calls: list[tuple[int, str, tuple[tuple[int, ...], ...] | None]] = []
                private_ids = ((101,), (202,), (303,))

                def execute_arm(**kwargs: object):
                    ordinal = int(kwargs["campaign_ordinal"])
                    model = kwargs["model"]
                    reference = kwargs["reference_prompt_token_ids"]
                    arm = (
                        "A"
                        if model.id
                        == SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID
                        else "B"
                    )
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
                        _lifetime(ordinal, arm, t0=100.0, later=20.0),
                        private_ids,
                        True,
                    )

                with (
                    patch(
                        "bench.sm121_chunked_prefill_runner._load_campaign",
                        return_value=(campaign, study, loaded),
                    ),
                    patch(
                        "bench.sm121_chunked_prefill_runner._execute_arm",
                        side_effect=execute_arm,
                    ),
                ):
                    summary = execute_sm121_chunked_prefill_performance_campaign(
                        campaign_dir, workspace=root / "workspace"
                    )
                expected = [1, 2, 3, 4] if fail_ordinal is None else [1, 2]
                self.assertEqual(expected, [ordinal for ordinal, _arm, _ref in calls])
                self.assertEqual(
                    list(SM121_CHUNKED_PREFILL_PERFORMANCE_ARM_ORDER[: len(calls)]),
                    [arm for _ordinal, arm, _ref in calls],
                )
                self.assertIsNone(calls[0][2])
                if len(calls) > 1:
                    self.assertEqual(private_ids, calls[1][2])
                self.assertEqual(
                    "complete" if fail_ordinal is None else "partial",
                    summary["status"],
                )
                self.assertTrue((campaign_dir / "summary.json").is_file())
                self.assertFalse(
                    any(
                        (run_dir / "events.jsonl").exists()
                        for run_dir, _plan, _model, _suite in loaded
                    )
                )


if __name__ == "__main__":
    unittest.main()
