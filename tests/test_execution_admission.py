from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.autoresearch_campaign import (
    CampaignPlanningError,
    CellProjectionError,
    freeze_campaign,
    load_frozen_campaign,
    run_campaign,
    run_frozen_cell,
)
from bench.execution_admission import (
    RETIRED_SGLANG_SOURCE_OVERLAY_DIGESTS,
    model_execution_blocker,
)
from bench.journal import content_hash
from bench.loop_campaign import LoopCampaignError, create_campaign_plan, execute_campaign
from bench.manifest import (
    ManifestError,
    SGLangSourceOverlay,
    load_models,
    load_suite,
    model_spec_to_dict,
)
from bench.runner import PreflightError, _canonical_case, create_plan, execute_plan
from bench.runtime import RuntimeErrorWithContext, start_sglang
from sparkbench import command_matrix
from tests.test_autoresearch_campaign import (
    CAMPAIGN_PATH,
    ROOT,
    _freeze_campaign_fixture,
)
from tests.test_loop_campaign import _load_minimal_config


RETIRED_DIGEST = (
    "sha256:e30566492e1502f94a4c7fed42d90b5"
    "23bbb662580c628459e6e63c7b5263c75"
)
SAFE_DIGEST = "sha256:" + "1" * 64
RETIRED_MESSAGE = "retired SGLang source overlay"
RETIRED_PROFILE_STATUSES = {
    "qwen38-flash-next-nvfp4-mtp-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp-depth0-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp-depth1-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp-depth2-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp-depth3-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp-c8-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp2-c8-sglang": "incompatible",
    "qwen38-flash-next-nvfp4-mtp2-c8-lazy-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp1-c8-lazy-ple-mapped-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp2-c8-lazy-ple-mapped-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp3-c8-lazy-ple-mapped-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp2-agent64k-low-ple-mapped-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp3-agent64k-low-ple-mapped-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp2-agent64k-low-chunk2k-ple-mapped-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp2-agent64k-none-ple-mapped-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp2-c6-extra-ple-mapped-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp3-c6-extra-ple-mapped-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp3-c8-lazy-ple-omitted-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp3-quality-v2-ple-mapped-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-mtp3-quality-v2-ple-omitted-sglang": "exploratory",
    "qwen38-flash-next-nvfp4-long-sglang": "incompatible",
}


def _overlay(
    digest: str = RETIRED_DIGEST,
    *,
    host_path: str = "renamed/overlay.py",
    container_path: str = "/renamed/overlay.py",
) -> dict[str, str]:
    return {
        "host_path": host_path,
        "container_path": container_path,
        "digest": digest,
    }


def _blocked_namespace(identifier: str = "renamed-retired") -> SimpleNamespace:
    return SimpleNamespace(
        id=identifier,
        backend="sglang",
        support_status="spark_vllm_matrix",
        tasks=("chat",),
        prefix_cache_mode=None,
        sglang_source_overlays=(SimpleNamespace(**_overlay()),),
    )


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, bytes], ...]:
    if not root.exists():
        return ()
    rows: list[tuple[str, str, bytes]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            rows.append((relative, "directory", b""))
        elif path.is_file():
            rows.append((relative, "file", path.read_bytes()))
        else:
            rows.append((relative, "other", b""))
    return tuple(rows)


def _write_frozen_runner_plan(run_dir: Path) -> Path:
    model = {
        "id": "renamed-frozen-retired",
        "backend": "sglang",
        "source": "example/model",
        "served_name": "example/model",
        "tasks": ["chat"],
        "max_context": 8192,
        "endpoint": "http://127.0.0.1:30000/v1",
        "support_status": "spark_vllm_matrix",
        "sglang_source_overlays": [_overlay()],
    }
    case = {
        "id": "decode",
        "kind": "decode",
        "requires": ["chat"],
        "warmups": 0,
        "repetitions": 1,
        "max_output_tokens": 8,
        "temperature": 0.0,
        "concurrency": 1,
        "prompt_repetitions": 0,
    }
    suite = {
        "id": "synthetic-suite",
        "description": "",
        "schema_version": 1,
        "cases": [case],
    }
    plan = {
        "schema_version": 2,
        "created_at": "2026-08-28T00:00:00+00:00",
        "fingerprint": content_hash(
            {"model": model, "suite": suite, "resolved": {}}
        ),
        "run_nonce": "a" * 32,
        "models_manifest": "synthetic-models.toml",
        "suite_manifest": "synthetic-suite.toml",
        "model": model,
        "suite": {**suite, "cases": [_canonical_case(model, case)]},
        "resolved": {},
        "host_at_plan": {},
    }
    plan["integrity_hash"] = content_hash(plan, 64)
    run_dir.mkdir(parents=True)
    plan_path = run_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    return plan_path


class ExecutionBlockerTests(unittest.TestCase):
    def test_chunked_prefill_profiles_require_their_dedicated_executor(self) -> None:
        profiles = load_models(ROOT / "manifests" / "models.toml")
        for profile_id in (
            "qwen38-flash-next-nvfp4-sm121-triton-storage-chunked-prefill-performance-1k-sglang",
            "qwen38-flash-next-nvfp4-sm121-triton-storage-chunked-prefill-performance-2k-sglang",
            "qwen38-flash-next-nvfp4-sm121-triton-storage-chunked-prefill-performance-2k-v2-sglang",
            "qwen38-flash-next-nvfp4-sm121-triton-storage-chunked-prefill-performance-4k-v2-sglang",
            "qwen38-flash-next-nvfp4-sm121-triton-storage-chunked-prefill-performance-4k-v3-sglang",
            "qwen38-flash-next-nvfp4-sm121-triton-storage-chunked-prefill-performance-8k-v3-sglang",
        ):
            with self.subTest(profile_id=profile_id):
                profile = profiles[profile_id]
                self.assertIn(
                    "chunked-prefill performance profile",
                    model_execution_blocker(profile) or "",
                )
                self.assertIsNone(
                    model_execution_blocker(
                        profile,
                        allow_sm121_chunked_prefill_performance=True,
                    )
                )

    def test_cache_performance_profiles_require_their_dedicated_executor(self) -> None:
        profiles = load_models(ROOT / "manifests" / "models.toml")
        for profile_id in (
            "qwen38-flash-next-nvfp4-sm121-triton-storage-cache-performance-on-sglang",
            "qwen38-flash-next-nvfp4-sm121-triton-storage-cache-performance-off-sglang",
        ):
            with self.subTest(profile_id=profile_id):
                profile = profiles[profile_id]
                self.assertIn(
                    "cache-policy performance profile",
                    model_execution_blocker(profile) or "",
                )
                self.assertIsNone(
                    model_execution_blocker(
                        profile,
                        allow_sm121_cache_performance=True,
                    )
                )

    def test_semantic_cache_profiles_require_their_dedicated_executor(self) -> None:
        profiles = load_models(ROOT / "manifests" / "models.toml")
        for profile_id in (
            "qwen38-flash-next-nvfp4-sm121-triton-storage-cache-policy-off-sglang",
            "qwen38-flash-next-nvfp4-sm121-triton-storage-cache-policy-on-sglang",
        ):
            with self.subTest(profile_id=profile_id):
                profile = profiles[profile_id]
                self.assertIn(
                    "cache-policy semantic profile",
                    model_execution_blocker(profile) or "",
                )
                self.assertIsNone(
                    model_execution_blocker(
                        profile,
                        allow_sm121_cache_semantic_canary=True,
                    )
                )

    def test_retired_manifest_inventory_and_statuses_are_exact(self) -> None:
        self.assertEqual(
            RETIRED_SGLANG_SOURCE_OVERLAY_DIGESTS,
            frozenset({RETIRED_DIGEST}),
        )
        models = load_models(ROOT / "manifests" / "models.toml")
        actual = {
            model.id: model.support_status
            for model in models.values()
            if any(
                overlay.digest == RETIRED_DIGEST
                for overlay in model.sglang_source_overlays
            )
        }
        self.assertEqual(actual, RETIRED_PROFILE_STATUSES)
        self.assertEqual(
            Counter(actual.values()),
            Counter({"exploratory": 19, "incompatible": 2}),
        )

    def test_blocker_resists_representation_and_metadata_bypasses(self) -> None:
        profile = load_models(ROOT / "manifests" / "models.toml")[
            "qwen38-flash-next-nvfp4-mtp-sglang"
        ]
        unsafe = next(
            overlay
            for overlay in profile.sglang_source_overlays
            if overlay.digest == RETIRED_DIGEST
        )
        safe = SGLangSourceOverlay("safe.py", "/safe.py", SAFE_DIGEST)
        typed = replace(
            profile,
            id="renamed",
            backend="vllm",
            support_status="spark_vllm_matrix",
            recipe_revision="0" * 40,
            sglang_source_overlays=(replace(unsafe, host_path="moved.py"), safe),
        )
        mapping = model_spec_to_dict(profile)
        mapping.update(
            {
                "id": "mapping-alias",
                "backend": "external",
                "support_status": "incompatible",
                "recipe_revision": None,
                "sglang_source_overlays": [
                    _overlay(SAFE_DIGEST),
                    _overlay(host_path="elsewhere.py", container_path="/elsewhere.py"),
                ],
            }
        )
        namespace = SimpleNamespace(
            id="namespace-alias",
            backend="ollama",
            support_status="exploratory",
            sglang_source_overlays=(
                SimpleNamespace(**_overlay()),
                SimpleNamespace(**_overlay(SAFE_DIGEST)),
            ),
        )
        mixed = {
            "sglang_source_overlays": (
                SimpleNamespace(**_overlay(SAFE_DIGEST)),
                SimpleNamespace(**_overlay()),
            )
        }
        for index, value in enumerate((profile, typed, mapping, namespace, mixed)):
            with self.subTest(index=index):
                self.assertIn(RETIRED_MESSAGE, model_execution_blocker(value) or "")

        path_only = {
            "sglang_source_overlays": [
                _overlay(
                    SAFE_DIGEST,
                    host_path=f"path/{RETIRED_DIGEST}.py",
                    container_path=f"/{RETIRED_DIGEST}.py",
                )
            ]
        }
        self.assertIsNone(model_execution_blocker(path_only))
        self.assertIsNone(model_execution_blocker(SimpleNamespace()))


class GenericExecutionAdmissionTests(unittest.TestCase):
    def test_create_plan_rejects_before_validation_or_filesystem_mutation(self) -> None:
        profile = load_models(ROOT / "manifests" / "models.toml")[
            "qwen38-flash-next-nvfp4-mtp-sglang"
        ]
        model = replace(
            profile,
            id="renamed-create-plan",
            support_status="spark_vllm_matrix",
        )
        suite = load_suite(ROOT / "manifests" / "suites" / "smoke.toml")
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            with (
                patch("bench.runner.validate_benchmark_selection") as validate,
                patch("bench.runner._image_digest") as image_digest,
                patch("bench.runner._host_snapshot") as host_snapshot,
                self.assertRaisesRegex(RuntimeError, RETIRED_MESSAGE),
            ):
                create_plan(
                    model=model,
                    suite=suite,
                    results_root=results,
                    models_path=Path("models.toml"),
                    suite_path=Path("suite.toml"),
                )
            self.assertFalse(results.exists())
        validate.assert_not_called()
        image_digest.assert_not_called()
        host_snapshot.assert_not_called()

    def test_frozen_plan_checks_integrity_then_rejects_without_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            plan_path = _write_frozen_runner_plan(run_dir)
            original = _tree_snapshot(root)
            with (
                patch("bench.runner.Journal") as journal,
                patch("bench.runner.results_lock_path") as lock_path,
                patch("bench.runner._recover_pending_lifecycle") as recover,
                patch("bench.runner._preflight") as preflight,
                patch("bench.runner.TelemetrySampler") as telemetry,
                patch("bench.runner.start_server") as start_server,
                self.assertRaisesRegex(PreflightError, RETIRED_MESSAGE),
            ):
                execute_plan(run_dir, workspace=root / "workspace")
            self.assertEqual(_tree_snapshot(root), original)
            for probe in (journal, lock_path, recover, preflight, telemetry, start_server):
                probe.assert_not_called()

            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["model"]["served_name"] = "tampered"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with (
                patch("bench.runner._preflight") as tampered_preflight,
                patch("bench.runner.start_server") as tampered_start,
                self.assertRaisesRegex(RuntimeError, "fingerprint"),
            ):
                execute_plan(run_dir, workspace=root / "workspace")
            tampered_preflight.assert_not_called()
            tampered_start.assert_not_called()

    def test_start_sglang_rejects_before_callbacks_or_runtime_probes(self) -> None:
        model = _blocked_namespace()
        abort_check = Mock()
        on_server_created = Mock()
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "uncreated-workspace"
            with (
                patch("bench.runtime._existing_container") as existing,
                patch("bench.runtime._port_is_free") as port_free,
                patch("bench.runtime._exact_sglang_snapshot") as snapshot,
                patch("bench.runtime._resolve_sglang_source_overlays") as overlays,
                patch("bench.runtime.secrets.token_urlsafe") as token,
                patch("bench.runtime._run") as run,
                self.assertRaisesRegex(RuntimeErrorWithContext, RETIRED_MESSAGE),
            ):
                start_sglang(
                    model,
                    workspace=workspace,
                    abort_check=abort_check,
                    on_server_created=on_server_created,
                )
            self.assertFalse(workspace.exists())
        for probe in (
            abort_check,
            on_server_created,
            existing,
            port_free,
            snapshot,
            overlays,
            token,
            run,
        ):
            probe.assert_not_called()


class MatrixExecutionAdmissionTests(unittest.TestCase):
    @staticmethod
    def _args(results: Path, *, match: str) -> argparse.Namespace:
        return argparse.Namespace(
            models=Path("models.toml"),
            suite=Path("suite.toml"),
            results=results,
            backend=None,
            task=None,
            match=match,
            limit=None,
            plan_only=True,
            allow_download=True,
            fail_fast=False,
        )

    def test_matrix_excludes_retired_and_blocked_only_selection_is_atomic(self) -> None:
        safe = SimpleNamespace(
            id="safe-model",
            backend="ollama",
            support_status="exploratory",
            tasks=("chat",),
            prefix_cache_mode=None,
            sglang_source_overlays=(),
        )
        blocked = _blocked_namespace("blocked-model")
        models = {blocked.id: blocked, safe.id: safe}
        availability = {
            identifier: SimpleNamespace(available=True) for identifier in models
        }

        def fake_create_plan(*, model: SimpleNamespace, results_root: Path, **_: object) -> Path:
            run_dir = results_root / f"run-{model.id}"
            run_dir.mkdir()
            return run_dir

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            planned: list[str] = []

            def record_plan(**kwargs: object) -> Path:
                model = kwargs["model"]
                assert isinstance(model, SimpleNamespace)
                planned.append(model.id)
                return fake_create_plan(**kwargs)  # type: ignore[arg-type]

            with (
                patch("sparkbench.load_models", return_value=models),
                patch("sparkbench.load_suite", return_value=SimpleNamespace(id="smoke")),
                patch("sparkbench._inventory"),
                patch("sparkbench.assess_model_availability", return_value=availability),
                patch("sparkbench.validate_benchmark_selection"),
                patch("sparkbench.create_plan", side_effect=record_plan),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(command_matrix(self._args(root / "mixed", match="*")), 0)
            self.assertEqual(planned, [safe.id])

            blocked_results = root / "blocked-only"
            with (
                patch("sparkbench.load_models", return_value=models),
                patch("sparkbench.load_suite", return_value=SimpleNamespace(id="smoke")),
                patch("sparkbench._inventory"),
                patch("sparkbench.assess_model_availability", return_value=availability),
                patch("sparkbench.create_plan") as create,
                self.assertRaisesRegex(ManifestError, "no runnable"),
            ):
                command_matrix(self._args(blocked_results, match=blocked.id))
            create.assert_not_called()
            self.assertFalse((blocked_results / "matrices").exists())


class AutoresearchExecutionAdmissionTests(unittest.TestCase):
    def _frozen_campaign(self, root: Path) -> Path:
        with patch(
            "bench.execution_admission.RETIRED_SGLANG_SOURCE_OVERLAY_DIGESTS",
            frozenset(),
        ):
            return _freeze_campaign_fixture(root)

    def test_freeze_rejects_before_campaign_topology_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "autoresearch"
            with self.assertRaisesRegex(CampaignPlanningError, RETIRED_MESSAGE):
                freeze_campaign(
                    CAMPAIGN_PATH,
                    workspace=ROOT,
                    results_root=results,
                )
            self.assertFalse(results.exists())

    def test_run_rejects_without_mutating_frozen_campaign(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = self._frozen_campaign(Path(directory))
            load_frozen_campaign(campaign_dir)
            original = _tree_snapshot(campaign_dir)
            with (
                patch("bench.autoresearch_campaign._campaign_lock") as lock,
                patch("bench.autoresearch_campaign._recover_interrupted_cells") as recover,
                self.assertRaisesRegex(CampaignPlanningError, RETIRED_MESSAGE),
            ):
                run_campaign(campaign_dir, workspace=ROOT)
            self.assertEqual(_tree_snapshot(campaign_dir), original)
        lock.assert_not_called()
        recover.assert_not_called()

    def test_direct_cell_rejects_before_log_worker_or_recovery(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = self._frozen_campaign(Path(directory))
            cell = load_frozen_campaign(campaign_dir).cells[0]
            original = _tree_snapshot(campaign_dir)
            worker = Mock()
            with (
                patch("bench.autoresearch_campaign._private_append_log") as private_log,
                patch("bench.autoresearch_campaign._recover_cell") as recover,
                self.assertRaisesRegex(CellProjectionError, RETIRED_MESSAGE),
            ):
                run_frozen_cell(
                    cell,
                    workspace=ROOT,
                    cell_timeout_s=1800,
                    cleanup_timeout_s=120,
                    worker_runner=worker,
                )
            self.assertEqual(_tree_snapshot(campaign_dir), original)
        private_log.assert_not_called()
        recover.assert_not_called()
        worker.assert_not_called()


class LoopCampaignExecutionAdmissionTests(unittest.TestCase):
    def test_create_rejects_all_profiles_before_environment_or_results(self) -> None:
        config = _load_minimal_config()
        safe = SimpleNamespace(
            tasks=("chat", "tools"),
            max_context=40_960,
            sglang_source_overlays=(),
        )
        blocked = SimpleNamespace(
            tasks=("chat", "tools"),
            max_context=40_960,
            sglang_source_overlays=(SimpleNamespace(**_overlay()),),
        )
        models = {
            "synthetic-rlm": safe,
            "synthetic-halo": safe,
            "synthetic-halo-fallback": blocked,
        }
        loop_python = SimpleNamespace(is_file=Mock())
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "loop-results"
            with (
                patch("bench.loop_campaign.load_campaign_manifest", return_value=config),
                patch("bench.loop_campaign.load_models", return_value=models),
                patch("bench.loop_campaign._validate_reasoning_profiles") as reasoning,
                patch("bench.loop_campaign.DEFAULT_LOOP_PYTHON", loop_python),
                patch("bench.loop_campaign._verify_loop_environment") as environment,
                patch("bench.loop_campaign._verify_worker_image") as worker_image,
                patch("bench.loop_campaign._dataset_inventory") as dataset,
                patch("bench.loop_campaign._repository_provenance") as repository,
                self.assertRaisesRegex(LoopCampaignError, RETIRED_MESSAGE),
            ):
                create_campaign_plan(
                    campaign_path=Path("campaign.toml"),
                    models_path=Path("models.toml"),
                    results_root=results,
                )
            self.assertFalse(results.exists())
        for probe in (
            reasoning,
            loop_python.is_file,
            environment,
            worker_image,
            dataset,
            repository,
        ):
            probe.assert_not_called()

    def test_execute_checks_loaded_models_before_repository_or_runtime(self) -> None:
        plan = {
            "models": {
                "safe": {"sglang_source_overlays": []},
                "retired-last": {"sglang_source_overlays": [_overlay()]},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "uncreated-run"
            workspace = root / "uncreated-workspace"
            with (
                patch("bench.loop_campaign.load_campaign_plan", return_value=plan) as load,
                patch("bench.loop_campaign._repository_provenance") as repository,
                patch("bench.loop_campaign._seconds_until") as deadline,
                patch("bench.loop_campaign._verify_frozen_admission") as admission,
                patch("bench.loop_campaign.Journal") as journal,
                patch("bench.loop_campaign.results_lock_path") as lock_path,
                patch("bench.loop_campaign.TelemetrySampler") as telemetry,
                self.assertRaisesRegex(LoopCampaignError, RETIRED_MESSAGE),
            ):
                execute_campaign(run_dir=run_dir, workspace=workspace)
            self.assertFalse(run_dir.exists())
            self.assertFalse(workspace.exists())
        load.assert_called_once_with(run_dir)
        for probe in (
            repository,
            deadline,
            admission,
            journal,
            lock_path,
            telemetry,
        ):
            probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
