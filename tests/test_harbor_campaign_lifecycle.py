from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from bench.harbor_campaign_lifecycle import (
    CampaignLifecycleError,
    CleanupStatus,
    DEFAULT_RAW_ROOT,
    ModelAdmission,
    RuntimeAdmission,
    _EXPECTED_MODEL,
    _attempt_gate,
    _cross_validate_model,
    _create_run_directory,
    _deadline_limited_timeout,
    _harbor_cleanup_certified,
    _is_exact_admission_payload,
    _parse_bounded_json_object,
    _record_status_then_load,
    _raw_descendant_owner_is_private,
    _scalar_output_path,
    _trial_timeout_s,
    admit_native_platform,
    build_lifecycle_envelope,
    build_parser,
    certify_private_raw_jobs,
    create_ephemeral_key,
    prepare_external_raw_root,
    remove_ephemeral_key,
    stage_head_agent_source,
    validate_lifecycle_envelope,
    write_scalar_result,
)
from bench.harbor_terminal import (
    HarborAttempt,
    HarborRunStatus,
    NpmArtifactAdmission,
    NpmArtifactRecord,
    iter_trials,
    load_campaign,
    summarize_campaign_results,
)
from bench.manifest import load_models


REPOSITORY = Path(__file__).resolve().parents[1]
CAMPAIGN_PATH = (
    REPOSITORY / "manifests" / "campaigns" / "harbor_terminal_coder_next.toml"
)


def _npm_admission(campaign: object) -> NpmArtifactAdmission:
    declared: list[tuple[str, str, str, str]] = []
    for agent in campaign.agents:
        declared.append(
            (agent.npm_package, agent.version, agent.npm_shasum, agent.npm_integrity)
        )
        if agent.platform_package is not None:
            declared.append(
                (
                    agent.platform_package,
                    agent.version,
                    agent.platform_shasum,
                    agent.platform_integrity,
                )
            )
    records = tuple(
        NpmArtifactRecord(
            package=package,
            version=version,
            size_bytes=index,
            shasum=shasum,
            integrity=integrity,
        )
        for index, (package, version, shasum, integrity) in enumerate(
            sorted(declared), start=1
        )
    )
    projection = [
        {
            "package": record.package,
            "version": record.version,
            "size_bytes": record.size_bytes,
            "shasum": record.shasum,
            "integrity": record.integrity,
        }
        for record in records
    ]
    digest = "sha256:" + hashlib.sha256(
        json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return NpmArtifactAdmission(digest=digest, artifacts=records)


def _cleanup(**changes: bool) -> CleanupStatus:
    values = {
        "harbor_resources_removed": True,
        "bridge_stopped": True,
        "bridge_socket_removed": True,
        "server_stopped": True,
        "telemetry_stopped": True,
        "key_removed": True,
        "raw_jobs_private_retained": True,
        "raw_jobs_key_free": True,
        "derived_dataset_removed": True,
        "runtime_overlays_removed": True,
        "staged_assets_removed": True,
        "socket_directory_removed": True,
        "npm_scratch_removed": True,
        "agent_source_removed": True,
        "python_pycache_removed": True,
    }
    values.update(changes)
    return CleanupStatus(**values)


def _runtime_admission(*, model: ModelAdmission | None = None) -> RuntimeAdmission:
    return RuntimeAdmission(
        artifact_validation=True,
        harbor_runtime_verified=True,
        node_tree_verified=True,
        agent_trees_verified=2,
        npm_artifacts_verified=3,
        runtime_assets_verified=2,
        agent_source_files_verified=1,
        python_bytecode_cache_empty=True,
        host_arm64=True,
        docker_server_arm64=True,
        model=model,
        unix_bridge_verified=model is not None,
    )


def _status(trial: object, **changes: object) -> HarborRunStatus:
    values: dict[str, object] = {
        "trial": trial,
        "exit_code": 0,
        "timed_out": False,
        "wall_s": 1.0,
        "main_image_id": "sha256:" + "1" * 64,
        "main_image_fingerprint": "sha256:" + "2" * 64,
        "main_image_arm64": True,
        "relay_image_arm64": True,
        "built_image_cleanup_succeeded": True,
        "setup_relay_rejected": True,
        "agent_relay_passed": True,
        "wrong_auth_rejected": True,
        "other_loopback_rejected": True,
        "gost_rejected": True,
        "dns_rejected": True,
        "gateway_rejected": True,
        "public_rejected": True,
        "capabilities_dropped": True,
        "cleanup_succeeded": True,
        "containers_found": 1,
        "containers_removed": 1,
        "networks_found": 1,
        "networks_removed": 1,
        "volumes_found": 1,
        "volumes_removed": 1,
    }
    values.update(changes)
    return HarborRunStatus(**values)


class LifecycleSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.campaign = load_campaign(CAMPAIGN_PATH)
        self.summary = summarize_campaign_results(
            self.campaign,
            (),
            network_policy_patch_digest="sha256:" + "a" * 64,
            npm_artifact_admission=_npm_admission(self.campaign),
        )
        self.payload = build_lifecycle_envelope(
            campaign=self.campaign,
            campaign_summary=self.summary,
            model_provenance=dict(_EXPECTED_MODEL),
            git_revision="b" * 40,
            git_clean=True,
            admission=_runtime_admission(),
            cleanup=_cleanup(),
            status="aborted",
            stop_reason="preflight",
            started_at="2026-08-17T00:00:00+00:00",
            finished_at="2026-08-17T00:00:01+00:00",
            elapsed_s=1.0,
            trials_started=0,
            trials_completed=0,
            cutoff_reached=False,
        )

    def test_zero_attempt_schema_is_exact_and_has_null_pass_rate(self) -> None:
        self.assertEqual(self.payload["schema_version"], 2)
        self.assertEqual(self.payload["campaign"]["trials"], [])
        self.assertIsNone(self.payload["campaign"]["summary"]["pass_rate"])
        validate_lifecycle_envelope(self.payload, campaign=self.campaign)

        old_schema = deepcopy(self.payload)
        old_schema["schema_version"] = 1
        with self.assertRaises(CampaignLifecycleError):
            validate_lifecycle_envelope(old_schema, campaign=self.campaign)

    def test_unknown_fields_and_innocuous_raw_strings_fail_closed(self) -> None:
        cases = []
        top = deepcopy(self.payload)
        top["note"] = "captured completion"
        cases.append(top)
        admission = deepcopy(self.payload)
        admission["admission"]["note"] = "captured completion"
        cases.append(admission)
        campaign = deepcopy(self.payload)
        campaign["campaign"]["summary"]["note"] = "captured completion"
        cases.append(campaign)
        trial = deepcopy(self.payload)
        trial["campaign"]["trials"] = [{"note": "captured prompt"}]
        trial["campaign"]["summary"]["attempts"] = 1
        cases.append(trial)
        for payload in cases:
            with self.subTest(payload=tuple(payload)):
                with self.assertRaises(CampaignLifecycleError):
                    validate_lifecycle_envelope(payload, campaign=self.campaign)

    def test_writer_revalidates_the_same_recursive_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            output = root / "result.json"
            write_scalar_result(output, self.payload, campaign=self.campaign)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

            unsafe = deepcopy(self.payload)
            unsafe["cleanup"]["completion"] = "raw"
            with self.assertRaises(CampaignLifecycleError):
                write_scalar_result(root / "unsafe.json", unsafe, campaign=self.campaign)

    def test_sensitive_value_canary_is_checked_after_exact_validation(self) -> None:
        with self.assertRaisesRegex(CampaignLifecycleError, "key entered"):
            build_lifecycle_envelope(
                campaign=self.campaign,
                campaign_summary=self.summary,
                model_provenance=dict(_EXPECTED_MODEL),
                git_revision="b" * 40,
                git_clean=True,
                admission=_runtime_admission(),
                cleanup=_cleanup(),
                status="aborted",
                stop_reason="preflight",
                started_at="2026-08-17T00:00:00+00:00",
                finished_at="2026-08-17T00:00:01+00:00",
                elapsed_s=1.0,
                trials_started=0,
                trials_completed=0,
                cutoff_reached=False,
                sensitive_values=("b" * 40,),
            )


class LifecycleGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.campaign = load_campaign(CAMPAIGN_PATH)
        self.trials = iter_trials(self.campaign)

    def test_reward_zero_is_a_valid_canary(self) -> None:
        attempt = HarborAttempt(
            trial=self.trials[0], status=_status(self.trials[0]), job_result=object()
        )
        projection = {"reward": 0, "exception_class": None, "paired_image_match": None}
        self.assertIsNone(_attempt_gate((attempt,), {"trials": [projection]}))

    def test_native_platform_normalizes_arm64_spellings(self) -> None:
        for spelling in ("aarch64", "arm64"):
            with self.subTest(spelling=spelling):
                result = subprocess.CompletedProcess(
                    args=("docker", "info"), returncode=0, stdout=spelling + "\n"
                )
                self.assertEqual(
                    admit_native_platform(
                        runner=lambda *args, **kwargs: result,
                        machine=lambda: "aarch64",
                    ),
                    (True, True),
                )

        wrong = subprocess.CompletedProcess(
            args=("docker", "info"), returncode=0, stdout="x86_64\n"
        )
        with self.assertRaises(CampaignLifecycleError):
            admit_native_platform(
                runner=lambda *args, **kwargs: wrong,
                machine=lambda: "aarch64",
            )

    def test_structured_admission_accepts_only_one_bounded_json_envelope(self) -> None:
        expected = {"ok": True, "marker": "MARKER"}
        bare = '{"ok":true,"marker":"MARKER"}'
        fenced = '```json\n{"ok":true,"marker":"MARKER"}\n```'
        self.assertEqual(_parse_bounded_json_object(bare), expected)
        self.assertEqual(_parse_bounded_json_object(fenced), expected)

        invalid = (
            "prose " + bare,
            fenced + " trailing",
            '```json\n```json\n{}\n```\n```',
            '{"ok":true,"ok":false}',
            '{"value":NaN}',
            '{"value":1e9999}',
            "[]",
            "",
        )
        for value in invalid:
            with self.subTest(value_length=len(value)):
                with self.assertRaises(CampaignLifecycleError):
                    _parse_bounded_json_object(value)

        self.assertTrue(_is_exact_admission_payload(expected, "MARKER"))
        self.assertFalse(
            _is_exact_admission_payload({"ok": 1, "marker": "MARKER"}, "MARKER")
        )
        self.assertFalse(
            _is_exact_admission_payload(
                {"ok": 1.0, "marker": "MARKER"}, "MARKER"
            )
        )
        self.assertFalse(
            _is_exact_admission_payload(
                {"ok": True, "marker": "MARKER", "extra": 1}, "MARKER"
            )
        )

    def test_canary_requires_final_result_cleanup_native_images_and_probes(self) -> None:
        projection = {"reward": 1, "exception_class": None, "paired_image_match": None}
        missing = HarborAttempt(
            trial=self.trials[0], status=_status(self.trials[0]), job_result=None
        )
        self.assertEqual(
            _attempt_gate((missing,), {"trials": [projection]}), "canary_gate"
        )
        bad_probe = HarborAttempt(
            trial=self.trials[0],
            status=_status(self.trials[0], gost_rejected=False),
            job_result=object(),
        )
        self.assertEqual(
            _attempt_gate((bad_probe,), {"trials": [projection]}), "canary_gate"
        )

        timed_out = HarborAttempt(
            trial=self.trials[0],
            status=_status(self.trials[0]),
            job_result=object(),
        )
        timeout_projection = {
            "reward": None,
            "exception_class": "AgentTimeoutError",
            "paired_image_match": None,
        }
        self.assertIsNone(
            _attempt_gate((timed_out,), {"trials": [timeout_projection]})
        )
        timeout_bad_probe = HarborAttempt(
            trial=self.trials[0],
            status=_status(self.trials[0], dns_rejected=False),
            job_result=object(),
        )
        self.assertEqual(
            _attempt_gate(
                (timeout_bad_probe,), {"trials": [timeout_projection]}
            ),
            "canary_gate",
        )
        nonterminal_timeout = deepcopy(timeout_projection)
        nonterminal_timeout["exception_class"] = "ApiError"
        self.assertEqual(
            _attempt_gate((timed_out,), {"trials": [nonterminal_timeout]}),
            "canary_gate",
        )

    def test_auth_is_immediate_and_endpoint_gate_is_narrow_and_consecutive(self) -> None:
        first = HarborAttempt(
            trial=self.trials[0], status=_status(self.trials[0]), job_result=object()
        )
        auth = {"reward": None, "exception_class": "AuthenticationError", "paired_image_match": None}
        self.assertEqual(
            _attempt_gate((first,), {"trials": [auth]}), "auth_failure_gate"
        )

        second = HarborAttempt(
            trial=self.trials[1], status=_status(self.trials[1]), job_result=object()
        )
        endpoints = [
            {"reward": None, "exception_class": "ApiConnectionError", "paired_image_match": True},
            {"reward": None, "exception_class": "UnknownApiError", "paired_image_match": True},
        ]
        self.assertEqual(
            _attempt_gate((first, second), {"trials": endpoints}),
            "endpoint_failure_gate",
        )
        generic = deepcopy(endpoints)
        generic[0]["exception_class"] = "ApiError"
        self.assertIsNone(_attempt_gate((first, second), {"trials": generic}))

    def test_image_pair_mismatch_stops_before_a_third_trial(self) -> None:
        attempts = tuple(
            HarborAttempt(trial=trial, status=_status(trial), job_result=object())
            for trial in self.trials[:2]
        )
        projections = [
            {"reward": 0, "exception_class": None, "paired_image_match": False},
            {"reward": 0, "exception_class": None, "paired_image_match": False},
        ]
        self.assertEqual(
            _attempt_gate(attempts, {"trials": projections}), "image_identity_gate"
        )

    def test_trial_timeout_preserves_the_audit_reserve(self) -> None:
        self.assertEqual(
            _trial_timeout_s(self.campaign, deadline=10_000, now=5_000), 3_600
        )
        self.assertEqual(
            _trial_timeout_s(self.campaign, deadline=10_000, now=7_000), 3_000
        )
        self.assertIsNone(
            _trial_timeout_s(self.campaign, deadline=10_000, now=10_000)
        )
        deadline_status = _status(
            self.trials[0], exit_code=None, timed_out=True
        )
        self.assertTrue(
            _deadline_limited_timeout(
                deadline_status, remaining_s=3_000, per_trial_cap_s=3_600
            )
        )
        self.assertFalse(
            _deadline_limited_timeout(
                deadline_status, remaining_s=4_000, per_trial_cap_s=3_600
            )
        )

    def test_raw_loader_failure_retains_failed_cleanup_status(self) -> None:
        status = _status(self.trials[0], cleanup_succeeded=False)
        attempts: list[HarborAttempt] = []

        def fail() -> object:
            raise CampaignLifecycleError("synthetic raw parser failure")

        with self.assertRaises(CampaignLifecycleError):
            _record_status_then_load(
                trial=self.trials[0], status=status, attempts=attempts, loader=fail
            )
        self.assertEqual(len(attempts), 1)
        self.assertIsNone(attempts[0].job_result)
        self.assertFalse(_harbor_cleanup_certified(1, (status,)))
        self.assertFalse(_harbor_cleanup_certified(1, ()))

    def test_full_model_spec_rejects_runtime_argument_drift(self) -> None:
        model = load_models(REPOSITORY / "manifests" / "models.toml")[
            self.campaign.model.profile
        ]
        _cross_validate_model(self.campaign, model)
        changed = replace(model, args=(*model.args[:-2], "8191", model.args[-1]))
        with self.assertRaises(CampaignLifecycleError):
            _cross_validate_model(self.campaign, changed)


class LifecycleFilesystemTests(unittest.TestCase):
    def test_private_run_name_preserves_the_unix_socket_path_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            raw.chmod(0o700)
            run = _create_run_directory(
                raw,
                "qwen3-coder-next-harbor-terminal-offline-2026-08-18",
            )
            self.assertTrue(run.name.startswith("hc-"))
            candidate = DEFAULT_RAW_ROOT / run.name / "relay-private" / "model.sock"
            self.assertLess(len(str(candidate).encode()), 100)

    def test_external_output_and_key_are_owner_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            workspace = base / "repo"
            workspace.mkdir(mode=0o700)
            raw = prepare_external_raw_root(base / "private", workspace=workspace)
            run = raw / "run"
            run.mkdir(mode=0o700)
            output = _scalar_output_path(run, "campaign-result.json")
            self.assertEqual(output.parent, run)
            with self.assertRaises(CampaignLifecycleError):
                _scalar_output_path(run, "../escape.json")

            secret_dir = run / "relay"
            secret_dir.mkdir(mode=0o700)
            key_path = secret_dir / "internal-api-key"
            key = create_ephemeral_key(key_path)
            self.assertTrue(key)
            self.assertEqual(stat.S_IMODE(key_path.stat().st_mode), 0o600)
            self.assertTrue(remove_ephemeral_key(key_path))

    def test_retained_raw_tree_is_scanned_for_the_exact_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory)
            external.chmod(0o700)
            run = external / "run"
            run.mkdir(mode=0o700)
            jobs = run / "jobs"
            jobs.mkdir(mode=0o755)
            record = jobs / "result.json"
            record.write_text('{"scalar":1}\n', encoding="utf-8")
            record.chmod(0o644)
            self.assertEqual(
                certify_private_raw_jobs(
                    run, owner=external, relay_credential="exact-secret"
                ),
                (True, True),
            )
            record.write_text('{"value":"exact-secret"}\n', encoding="utf-8")
            self.assertEqual(
                certify_private_raw_jobs(
                    run, owner=external, relay_credential="exact-secret"
                ),
                (True, False),
            )

    def test_retained_raw_descendants_allow_only_host_or_container_root(self) -> None:
        self.assertTrue(_raw_descendant_owner_is_private(0))
        self.assertTrue(_raw_descendant_owner_is_private(os.geteuid()))
        rejected_uid = next(
            uid for uid in (1, 2, 65534) if uid not in {0, os.geteuid()}
        )
        self.assertFalse(_raw_descendant_owner_is_private(rejected_uid))
        self.assertFalse(_raw_descendant_owner_is_private(True))

    def test_retained_raw_tree_fails_closed_on_walk_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory)
            external.chmod(0o700)
            run = external / "run"
            run.mkdir(mode=0o700)

            def failed_walk(
                _top: Path, *, followlinks: bool, onerror: object
            ) -> object:
                self.assertFalse(followlinks)
                onerror(PermissionError("unreadable descendant"))
                return iter(())

            with patch(
                "bench.harbor_campaign_lifecycle.os.walk", side_effect=failed_walk
            ):
                self.assertEqual(
                    certify_private_raw_jobs(
                        run, owner=external, relay_credential="exact-secret"
                    ),
                    (False, False),
                )

    def test_cli_has_no_checkout_or_bridge_host_escape_hatches(self) -> None:
        options = {option for action in build_parser()._actions for option in action.option_strings}
        self.assertIn("--harbor-runtime-root", options)
        self.assertIn("--tool-prefix-root", options)
        self.assertNotIn("--harbor-checkout", options)
        self.assertNotIn("--bridge-host", options)
        self.assertNotIn("--models", options)

    def test_agent_source_is_staged_from_exact_head_not_worktree(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repository = base / "repo"
            bench = repository / "bench"
            bench.mkdir(parents=True)
            for name in ("harbor_pinned_agents.py",):
                (bench / name).write_bytes((REPOSITORY / "bench" / name).read_bytes())
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(
                ["git", "config", "user.email", "fixture@example.invalid"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Fixture"],
                cwd=repository,
                check=True,
            )
            subprocess.run(["git", "add", "bench"], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "fixture"], cwd=repository, check=True
            )
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            # A post-commit working-tree payload must never enter the staged package.
            (bench / "harbor_pinned_agents.py").write_text(
                "captured_completion = 'must not execute'\n", encoding="utf-8"
            )
            run_root = base / "run"
            run_root.mkdir(mode=0o700)
            source, cache = stage_head_agent_source(
                campaign,
                workspace=repository,
                run_root=run_root,
                revision=revision,
            )
            self.assertEqual(
                (source / "bench" / "harbor_pinned_agents.py").read_bytes(),
                (REPOSITORY / "bench" / "harbor_pinned_agents.py").read_bytes(),
            )
            self.assertEqual(tuple(cache.iterdir()), ())
            self.assertEqual(
                stat.S_IMODE((source / "bench" / "harbor_pinned_agents.py").stat().st_mode),
                0o444,
            )
            admitted_source = (
                source / "bench" / "harbor_pinned_agents.py"
            ).read_text(encoding="utf-8")
            self.assertIn(
                '"/logs/agent/opencode/xdg-data",', admitted_source
            )
            self.assertIn(
                '"/logs/agent/opencode/xdg-state",', admitted_source
            )
            self.assertIn("rm -rf --one-file-system --", admitted_source)
            self.assertNotIn("find {trees}", admitted_source)


if __name__ == "__main__":
    unittest.main()
