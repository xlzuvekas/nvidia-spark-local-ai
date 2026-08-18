from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from bench.harbor_terminal import (
    CampaignLock,
    HarborAttempt,
    HarborCampaignError,
    HarborInvocation,
    HarborRawResult,
    HarborRuntimeAdmission,
    HarborRunStatus,
    NpmArtifactAdmission,
    NpmArtifactRecord,
    RuntimeOverlayAdmission,
    _harbor_runtime_digest,
    _inspect_image,
    _run_process_group,
    _runtime_admission_digest,
    build_harbor_invocation,
    canonical_bridge_base_url,
    canonical_summary_bytes,
    cleanup_harbor_containers,
    derive_private_task_dataset,
    hold_campaign_lock,
    iter_trials,
    load_campaign,
    load_network_admission,
    load_trial_job_result,
    read_api_key,
    run_harbor_invocation,
    snapshot_harbor_resources,
    summarize_campaign_results,
    verify_npm_artifact_admission,
    verify_harbor_runtime,
    verify_private_task_dataset,
)
from bench.harbor_runtime_assets import TreeAdmission


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PATH = ROOT / "manifests" / "campaigns" / "harbor_terminal_coder_next.toml"
BRIDGE_URL = "http://127.0.0.1:18080/v1"
CANARY = "CANARY-DO-NOT-PERSIST-7e36c9"


def _campaign():
    return load_campaign(CAMPAIGN_PATH)


def _fake_npm_admission(campaign) -> NpmArtifactAdmission:
    records = []
    for agent in campaign.agents:
        records.append(
            NpmArtifactRecord(
                package=agent.npm_package,
                version=agent.version,
                size_bytes=100 + len(records),
                shasum=agent.npm_shasum,
                integrity=agent.npm_integrity,
            )
        )
        if agent.platform_package is not None:
            records.append(
                NpmArtifactRecord(
                    package=agent.platform_package,
                    version=agent.version,
                    size_bytes=100 + len(records),
                    shasum=agent.platform_shasum,
                    integrity=agent.platform_integrity,
                )
            )
    records.sort(key=lambda record: record.package)
    payload = [
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
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return NpmArtifactAdmission(digest=digest, artifacts=tuple(records))


def _fake_runtime_admission(campaign, trial, root: Path) -> RuntimeOverlayAdmission:
    node_root = root / "fake-node"
    agent_root = root / "fake-agent"
    node_root.mkdir(exist_ok=True)
    agent_root.mkdir(exist_ok=True)
    node = TreeAdmission(
        protocol="sparkbench-readonly-tree-v1",
        digest=campaign.toolchain.node_tree_sha256.removeprefix("sha256:"),
        entries=2,
        files=1,
        links=0,
        size_bytes=campaign.toolchain.node_tree_size_bytes,
        resolved_path=node_root,
    )
    agent_pin = campaign.agent(trial.agent_id)
    agent = TreeAdmission(
        protocol="sparkbench-readonly-tree-v1",
        digest=agent_pin.install_tree_sha256.removeprefix("sha256:"),
        entries=1,
        files=1,
        links=0,
        size_bytes=agent_pin.install_tree_size_bytes,
        resolved_path=agent_root,
    )
    compose = root / "synthetic-overlay.json"
    encoded = b'{"services":{}}\n'
    compose.write_bytes(encoded)
    compose.chmod(0o600)
    return RuntimeOverlayAdmission(
        trial=trial,
        digest=_runtime_admission_digest(campaign, trial, node, agent),
        compose_sha256="sha256:" + hashlib.sha256(encoded).hexdigest(),
        node_tree=node,
        agent_tree=agent,
        compose_path=compose,
    )


def _fake_harbor_runtime(campaign, root: Path) -> HarborRuntimeAdmission:
    runtime = root / "fake-harbor-runtime"
    executable = runtime / campaign.harbor.executable_path
    python = runtime / campaign.harbor.python_path
    launcher = runtime / campaign.harbor.python_launcher_path
    executable.parent.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    executable.write_bytes(b"synthetic admitted Harbor entry point\n")
    python.write_bytes(b"synthetic admitted Python runtime\n")
    executable.chmod(0o555)
    python.chmod(0o555)
    launcher.symlink_to(campaign.harbor.python_launcher_target)
    tree = TreeAdmission(
        protocol="sparkbench-readonly-tree-v1",
        digest=campaign.harbor.runtime_tree_sha256.removeprefix("sha256:"),
        entries=campaign.harbor.runtime_tree_entries,
        files=campaign.harbor.runtime_tree_files,
        links=campaign.harbor.runtime_tree_links,
        size_bytes=campaign.harbor.runtime_tree_size_bytes,
        resolved_path=runtime,
    )
    return HarborRuntimeAdmission(
        digest=_harbor_runtime_digest(campaign, tree),
        tree=tree,
        executable_path=executable,
        python_launcher_path=launcher,
        python_path=python,
    )


def _fake_agent_source(root: Path) -> tuple[Path, Path]:
    source_root = root / "fake-agent-source"
    bench_root = source_root / "bench"
    bench_root.mkdir(parents=True)
    source_root.chmod(0o700)
    bench_root.chmod(0o700)
    for name in ("harbor_pinned_agents.py",):
        destination = bench_root / name
        destination.write_bytes((ROOT / "bench" / name).read_bytes())
        destination.chmod(0o444)
    pycache_root = root / "fake-python-pycache"
    pycache_root.mkdir(mode=0o700)
    return source_root, pycache_root


def _write_private_key(path: Path, value: str = "ephemeral-test-key") -> None:
    path.write_text(value + "\n", encoding="utf-8")
    path.chmod(0o600)


def _write_network_marker(path: Path) -> None:
    marker = {
        "schema_version": 1,
        "setup_relay_rejected": True,
        "agent_relay_passed": True,
        "wrong_auth_rejected": True,
        "other_loopback_rejected": True,
        "gost_rejected": True,
        "dns_rejected": True,
        "gateway_rejected": True,
        "public_rejected": True,
        "capabilities_dropped": True,
    }
    path.write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    path.chmod(0o600)


def _task_toml(task_id: str) -> str:
    return f'''schema_version = "1.1"
artifacts = []

[task]
name = "terminal-bench/{task_id}"
description = "Synthetic fixture"
keywords = ["fixture"]

[verifier]
timeout_sec = 900.0

[agent]
timeout_sec = 900.0

[environment]
build_timeout_sec = 600.0
docker_image = "fixture/{task_id}:pinned"
cpus = 1
memory_mb = 1024
storage_mb = 1024
gpus = 0
allow_internet = true
mcp_servers = []
'''


def _make_source_checkout(root: Path) -> Path:
    checkout = root / "terminal-bench"
    tasks_root = checkout / "tasks"
    tasks_root.mkdir(parents=True)
    campaign = _campaign()
    for task_id in (*campaign.dataset.tasks, "not-selected"):
        task = tasks_root / task_id
        (task / "tests").mkdir(parents=True)
        (task / "environment").mkdir()
        (task / "task.toml").write_text(_task_toml(task_id), encoding="utf-8")
        (task / "instruction.md").write_text(
            f"Synthetic instruction for {task_id}\n", encoding="utf-8"
        )
        (task / "tests" / "test.sh").write_text("exit 0\n", encoding="utf-8")
        (task / "environment" / "Dockerfile").write_text(
            "FROM scratch\n", encoding="utf-8"
        )
    subprocess.run(
        ["git", "init", "--quiet", str(checkout)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "add", "--", "tasks"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return checkout


def _derive_fixture(root: Path):
    workspace = root / "workspace"
    workspace.mkdir()
    cache = root / "cache"
    cache.mkdir()
    source = _make_source_checkout(cache)
    destination = cache / "derived"
    campaign = _campaign()
    source_snapshot = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    policy = derive_private_task_dataset(
        campaign,
        source_checkout=source,
        destination=destination,
        repo_root=workspace,
        checkout_verifier=lambda *_: None,
    )
    return campaign, workspace, cache, source, source_snapshot, policy


def _status(
    trial,
    *,
    exit_code: int | None = 0,
    timed_out: bool = False,
    cleanup_succeeded: bool = True,
) -> HarborRunStatus:
    return HarborRunStatus(
        trial=trial,
        exit_code=exit_code,
        timed_out=timed_out,
        wall_s=20.0,
        main_image_id="sha256:"
        + hashlib.sha256(trial.task_id.encode("ascii")).hexdigest(),
        main_image_fingerprint="sha256:"
        + hashlib.sha256((trial.task_id + ":runtime").encode("ascii")).hexdigest(),
        main_image_arm64=True,
        relay_image_arm64=True,
        built_image_cleanup_succeeded=cleanup_succeeded,
        setup_relay_rejected=True,
        agent_relay_passed=True,
        wrong_auth_rejected=True,
        other_loopback_rejected=True,
        gost_rejected=True,
        dns_rejected=True,
        gateway_rejected=True,
        public_rejected=True,
        capabilities_dropped=True,
        cleanup_succeeded=cleanup_succeeded,
        containers_found=1,
        containers_removed=1,
        networks_found=1,
        networks_removed=1,
        volumes_found=0,
        volumes_removed=0,
    )


def _timing(start: str, finish: str) -> dict[str, str]:
    return {"started_at": start, "finished_at": finish}


def _job_result(campaign, trial, *, reward: int = 1) -> HarborRawResult:
    agent = campaign.agent(trial.agent_id)
    raw_trial = {
        "id": "raw-id-must-not-survive",
        "trial_name": "raw-trial-name-must-not-survive",
        "trial_uri": "/local/private/path",
        "task_name": f"terminal-bench/{trial.task_id}",
        "agent_info": {
            "name": trial.agent_id,
            "version": agent.version,
            "model_info": {
                "provider": "openai",
                "name": campaign.model.served_name,
            },
        },
        "agent_result": {
            "n_input_tokens": 100 + trial.index,
            "n_cache_tokens": 10,
            "n_output_tokens": 20 + trial.index,
            "metadata": {
                "prompt": CANARY,
                "tool_arguments": {"secret": CANARY},
            },
        },
        "verifier_result": {"rewards": {"reward": reward}},
        "started_at": "2026-08-17T01:00:00+00:00",
        "finished_at": "2026-08-17T01:00:20+00:00",
        "environment_setup": _timing(
            "2026-08-17T01:00:00+00:00", "2026-08-17T01:00:02+00:00"
        ),
        "agent_setup": _timing(
            "2026-08-17T01:00:02+00:00", "2026-08-17T01:00:05+00:00"
        ),
        "agent_execution": _timing(
            "2026-08-17T01:00:05+00:00", "2026-08-17T01:00:17+00:00"
        ),
        "verifier": _timing(
            "2026-08-17T01:00:17+00:00", "2026-08-17T01:00:20+00:00"
        ),
        "exception_info": None,
        "config": {
            "instruction": CANARY,
            "api_key": CANARY,
            "command": CANARY,
        },
    }
    return HarborRawResult(
        job={
            "id": "raw-job-id-must-not-survive",
            "n_total_trials": 1,
            "stats": {"n_retries": 0, "n_completed_trials": 1},
            "private": CANARY,
        },
        trial=raw_trial,
    )


class _FakeDocker:
    def __init__(self) -> None:
        self.resources: dict[str, dict[str, str]] = {
            "container": {},
            "network": {},
            "volume": {},
        }
        self.commands: list[list[str]] = []
        self.inspect_failures: set[tuple[str, str]] = set()
        self.image_architecture: dict[str, object] = {}
        self.image_ids: dict[str, str] = {}
        self.image_layers: dict[str, list[str]] = {}
        self.image_variants: dict[str, object] = {}
        self.image_configs: dict[str, object] = {}
        self.image_rootfs: dict[str, object] = {}
        self.baseline_image_ids: set[str] = set()
        self.removed_images: set[str] = set()

    def __call__(self, command, **_):
        argv = list(command)
        self.commands.append(argv)
        if argv[:5] == ["docker", "image", "ls", "--no-trunc", "--quiet"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout="\n".join(sorted(self.baseline_image_ids)) + "\n"
            )
        if argv[:4] == ["docker", "image", "inspect", "--format"]:
            if argv[-1] in self.removed_images:
                return subprocess.CompletedProcess(argv, 1, stdout="")
            architecture = self.image_architecture.get(argv[-1], "arm64")
            image_id = self.image_ids.get(argv[-1], "sha256:" + "a" * 64)
            if argv[4] == "{{.Id}}":
                return subprocess.CompletedProcess(argv, 0, stdout=image_id + "\n")
            layers = self.image_layers.get(
                argv[-1], ["sha256:" + "b" * 64]
            )
            rootfs = self.image_rootfs.get(
                argv[-1], {"Type": "layers", "Layers": layers}
            )
            config = self.image_configs.get(
                argv[-1],
                {
                    "Image": "dynamic-parent-id",
                    "Labels": {"com.docker.compose.project": "dynamic"},
                    "Env": ["PATH=/usr/bin"],
                    "Cmd": ["/bin/sh"],
                },
            )
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps(
                    {
                        "Id": image_id,
                        "Architecture": architecture,
                        "Os": "linux",
                        "Variant": self.image_variants.get(argv[-1]),
                        "RootFS": rootfs,
                        "Config": config,
                    },
                    sort_keys=True,
                )
                + "\n",
            )
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(
                argv, 1 if argv[-1] in self.removed_images else 0, stdout=""
            )
        if argv[:4] == ["docker", "image", "rm", "--force"]:
            self.removed_images.add(argv[-1])
            return subprocess.CompletedProcess(argv, 0, stdout="")
        if argv[:3] == ["docker", "ps", "-aq"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout="\n".join(self.resources["container"]) + "\n"
            )
        for resource in ("network", "volume"):
            if argv[:4] == ["docker", resource, "ls", "-q"]:
                return subprocess.CompletedProcess(
                    argv, 0, stdout="\n".join(self.resources[resource]) + "\n"
                )
        if argv[:2] == ["docker", "inspect"]:
            identifier = argv[-1]
            if ("container", identifier) in self.inspect_failures:
                return subprocess.CompletedProcess(argv, 1, stdout="")
            project = self.resources["container"].get(identifier)
            return subprocess.CompletedProcess(
                argv, 0 if project else 1, stdout=(project or "") + "\n"
            )
        for resource in ("network", "volume"):
            if argv[:3] == ["docker", resource, "inspect"]:
                identifier = argv[-1]
                if (resource, identifier) in self.inspect_failures:
                    return subprocess.CompletedProcess(argv, 1, stdout="")
                project = self.resources[resource].get(identifier)
                return subprocess.CompletedProcess(
                    argv, 0 if project else 1, stdout=(project or "") + "\n"
                )
        if argv[:3] == ["docker", "rm", "--force"]:
            existed = self.resources["container"].pop(argv[-1], None) is not None
            return subprocess.CompletedProcess(argv, 0 if existed else 1, stdout="")
        for resource in ("network", "volume"):
            if argv[:3] == ["docker", resource, "rm"]:
                existed = self.resources[resource].pop(argv[-1], None) is not None
                return subprocess.CompletedProcess(argv, 0 if existed else 1, stdout="")
        raise AssertionError(f"Unexpected Docker command: {argv!r}")


class HarborManifestTests(unittest.TestCase):
    def test_manifest_loads_exact_pins_platform_artifact_and_trial_order(self) -> None:
        campaign = _campaign()

        self.assertEqual(campaign.harbor.version, "0.21.0")
        self.assertEqual(campaign.dataset.version, "2.1")
        self.assertEqual(campaign.execution.trial_wall_timeout_s, 3_600)
        self.assertEqual(len(iter_trials(campaign)), 12)
        self.assertEqual(iter_trials(campaign)[0].task_id, "build-cython-ext")
        self.assertEqual(iter_trials(campaign)[0].agent_id, "qwen-coder")
        self.assertEqual(
            campaign.agent("opencode").platform_package, "opencode-linux-arm64"
        )

    def test_manifest_rejects_unknown_keys_wrong_types_and_changed_pins(self) -> None:
        source = CAMPAIGN_PATH.read_text(encoding="utf-8")
        mutations = (
            "unexpected = true\n" + source,
            source.replace('version = "0.21.0"', 'version = "0.21.1"', 1),
            source.replace("n_attempts = 1", "n_attempts = true", 1),
            source.replace(
                '  "build-cython-ext:opencode",',
                '  "build-cython-ext:qwen-coder",',
                1,
            ),
            source.replace(
                'platform_shasum = "5d4952bb8c1c3bbcccc52bcd07a540a845e31408"\n',
                "",
                1,
            ),
            source.replace(
                'python_launcher_path = ".venv/bin/python"\n', "", 1
            ),
            source.replace(
                'agent_source_sha256 = "sha256:cc898eea',
                'agent_source_sha256 = "sha256:dc898eea',
                1,
            ),
            source.replace(
                "runtime_tree_entries = 66729", "runtime_tree_entries = true", 1
            ),
            source.replace(
                "install_tree_size_bytes = 132176032",
                "install_tree_size_bytes = true",
                1,
            ),
            source.replace(
                'relay_script_sha256 = "sha256:ebf7e377',
                'relay_script_sha256 = "sha256:abf7e377',
                1,
            ),
            source.replace("agent_timeout_s = 900", "agent_timeout_s = 901", 1),
            source.replace(
                "trial_wall_timeout_s = 3600", "trial_wall_timeout_s = 3601", 1
            ),
            source.replace(
                "hard_campaign_cutoff_s = 23400",
                "hard_campaign_cutoff_s = 23399",
                1,
            ),
            source.replace(
                "reserve_for_audit_s = 5400", "reserve_for_audit_s = 5399", 1
            ),
            source.replace(
                "server_default_temperature = 1.0",
                "server_default_temperature = 0.9",
                1,
            ),
            source.replace(
                "server_default_top_p = 0.95", "server_default_top_p = 0.9", 1
            ),
            source.replace(
                "server_default_top_k = 40", "server_default_top_k = 41", 1
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, content in enumerate(mutations):
                with self.subTest(index=index):
                    path = Path(directory) / f"bad-{index}.toml"
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(HarborCampaignError):
                        load_campaign(path)

    def test_bridge_url_requires_the_fixed_loopback_relay(self) -> None:
        self.assertEqual(
            canonical_bridge_base_url("http://127.0.0.1:18080/v1/"), BRIDGE_URL
        )
        for value in (
            "http://127.0.0.1:38421/v1",
            "http://localhost:18080/v1",
            "https://127.0.0.1:18080/v1",
            "http://127.0.0.1/v1",
            "http://127.0.0.1:18080/v1/models",
            "http://user@127.0.0.1:18080/v1",
            "http://172.31.0.1:18080/v1",
            "http://8.8.8.8:38421/v1",
        ):
            with self.subTest(value=value):
                with self.assertRaises(HarborCampaignError):
                    canonical_bridge_base_url(value)


class HarborSecretFileTests(unittest.TestCase):
    def test_api_key_requires_owner_only_single_link_regular_bounded_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid"
            _write_private_key(valid)
            self.assertEqual(read_api_key(valid), "ephemeral-test-key")

            symlink = root / "symlink"
            symlink.symlink_to(valid)
            with self.assertRaisesRegex(HarborCampaignError, "regular file"):
                read_api_key(symlink)

            first_link = root / "first-link"
            second_link = root / "second-link"
            _write_private_key(first_link)
            os.link(first_link, second_link)
            with self.assertRaisesRegex(HarborCampaignError, "single-link"):
                read_api_key(first_link)

            fifo = root / "fifo"
            os.mkfifo(fifo, 0o600)
            with self.assertRaisesRegex(HarborCampaignError, "regular file"):
                read_api_key(fifo)

            public = root / "public"
            _write_private_key(public)
            public.chmod(0o640)
            with self.assertRaisesRegex(HarborCampaignError, "owner-only"):
                read_api_key(public)

            oversized = root / "oversized"
            oversized.write_bytes(b"x" * (16 * 1024 + 1))
            oversized.chmod(0o600)
            with self.assertRaisesRegex(HarborCampaignError, "bounded"):
                read_api_key(oversized)


class HarborNpmAdmissionTests(unittest.TestCase):
    def _fixture(self, root: Path):
        workspace = root / "workspace"
        workspace.mkdir(parents=True)
        artifact_root = root / "artifacts"
        artifact_root.mkdir()
        payloads = {
            "@qwen-code/qwen-code": b"synthetic qwen wrapper tarball\n",
            "opencode-ai": b"synthetic opencode wrapper tarball\n",
            "opencode-linux-arm64": b"synthetic opencode arm64 tarball\n",
        }

        def hashes(payload: bytes) -> tuple[str, str]:
            shasum = hashlib.sha1(payload).hexdigest()
            integrity = "sha512-" + base64.b64encode(
                hashlib.sha512(payload).digest()
            ).decode("ascii")
            return shasum, integrity

        agents = []
        for agent in _campaign().agents:
            shasum, integrity = hashes(payloads[agent.npm_package])
            changes = {"npm_shasum": shasum, "npm_integrity": integrity}
            if agent.platform_package is not None:
                platform_shasum, platform_integrity = hashes(
                    payloads[agent.platform_package]
                )
                changes.update(
                    {
                        "platform_shasum": platform_shasum,
                        "platform_integrity": platform_integrity,
                    }
                )
            agents.append(replace(agent, **changes))
        campaign = replace(_campaign(), agents=tuple(agents))
        paths = {}
        for index, (package, payload) in enumerate(payloads.items(), start=1):
            path = artifact_root / f"artifact-{index}.tgz"
            path.write_bytes(payload)
            paths[package] = path
        return campaign, workspace, paths

    def test_exact_tarballs_govern_admission_and_unsafe_files_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign, workspace, paths = self._fixture(root)
            admission = verify_npm_artifact_admission(
                campaign, paths, repo_root=workspace
            )
            self.assertEqual(len(admission.artifacts), 3)
            self.assertRegex(admission.digest, r"^sha256:[0-9a-f]{64}$")

            corrupt = dict(paths)
            corrupt["opencode-ai"].write_bytes(b"changed tarball\n")
            with self.assertRaisesRegex(HarborCampaignError, "digest"):
                verify_npm_artifact_admission(campaign, corrupt, repo_root=workspace)

            campaign, workspace, paths = self._fixture(root / "second")
            target = paths["opencode-ai"]
            symlink = target.with_name("symlink.tgz")
            symlink.symlink_to(target)
            symlink_paths = dict(paths)
            symlink_paths["opencode-ai"] = symlink
            with self.assertRaises(HarborCampaignError):
                verify_npm_artifact_admission(
                    campaign, symlink_paths, repo_root=workspace
                )

            hardlink = target.with_name("hardlink.tgz")
            os.link(target, hardlink)
            with self.assertRaisesRegex(HarborCampaignError, "unsafe"):
                verify_npm_artifact_admission(campaign, paths, repo_root=workspace)
            hardlink.unlink()

            with patch("bench.harbor_terminal.MAX_NPM_TARBALL_BYTES", 1):
                with self.assertRaisesRegex(HarborCampaignError, "oversized"):
                    verify_npm_artifact_admission(
                        campaign, paths, repo_root=workspace
                    )


class HarborNetworkPolicyTests(unittest.TestCase):
    def test_derivation_changes_only_policy_and_verifier_mode_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign, workspace, cache, source, before, policy = _derive_fixture(root)

            after = {
                path.relative_to(source).as_posix(): path.read_bytes()
                for path in source.rglob("*")
                if path.is_file()
            }
            self.assertEqual(before, after)
            self.assertRegex(policy.digest, r"^sha256:[0-9a-f]{64}$")
            self.assertFalse((policy.dataset_dir / "not-selected").exists())
            for task_id in campaign.dataset.tasks:
                source_task = source / "tasks" / task_id
                derived_task = policy.dataset_dir / task_id
                for relative in (
                    "instruction.md",
                    "tests/test.sh",
                    "environment/Dockerfile",
                ):
                    self.assertEqual(
                        (source_task / relative).read_bytes(),
                        (derived_task / relative).read_bytes(),
                    )
                self.assertEqual(
                    os.stat(source_task / "tests/test.sh").st_mode & 0o111,
                    0,
                )
                self.assertEqual(
                    os.stat(derived_task / "tests/test.sh").st_mode & 0o777,
                    0o555,
                )
                task_text = (derived_task / "task.toml").read_text(encoding="utf-8")
                self.assertIn('network_mode = "allowlist"', task_text)
                self.assertIn(
                    'allowed_hosts = ["sparkbench-relay.invalid"]', task_text
                )
                self.assertEqual(task_text.count('network_mode = "no-network"'), 2)
                self.assertIn("allow_internet = true", task_text)
                self.assertEqual(
                    os.stat(derived_task / "task.toml").st_mode & 0o777,
                    0o644,
                )

            second = derive_private_task_dataset(
                campaign,
                source_checkout=source,
                destination=cache / "derived-second",
                repo_root=workspace,
                checkout_verifier=lambda *_: None,
            )
            self.assertEqual(policy.digest, second.digest)
            verify_private_task_dataset(
                campaign, policy, repo_root=workspace
            )

            verifier = policy.dataset_dir / campaign.dataset.tasks[0] / "tests/test.sh"
            verifier.chmod(0o755)
            with self.assertRaisesRegex(HarborCampaignError, "verifier launcher mode"):
                verify_private_task_dataset(campaign, policy, repo_root=workspace)

    def test_existing_agent_network_policy_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            cache = root / "cache"
            cache.mkdir()
            source = _make_source_checkout(cache)
            task = source / "tasks" / "build-cython-ext" / "task.toml"
            task.write_text(
                task.read_text().replace(
                    "[agent]\n", '[agent]\nnetwork_mode = "public"\n', 1
                )
            )
            subprocess.run(
                ["git", "-C", str(source), "add", "--", str(task)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self.assertRaisesRegex(HarborCampaignError, "already has"):
                derive_private_task_dataset(
                    _campaign(),
                    source_checkout=source,
                    destination=cache / "derived",
                    repo_root=workspace,
                    checkout_verifier=lambda *_: None,
                )

    def test_linked_source_file_is_rejected_and_partial_destination_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            cache = root / "cache"
            cache.mkdir()
            source = _make_source_checkout(cache)
            linked = source / "tasks" / "build-cython-ext" / "instruction.md"
            second = cache / "instruction-hardlink"
            os.link(linked, second)
            with self.assertRaisesRegex(HarborCampaignError, "linked"):
                derive_private_task_dataset(
                    _campaign(),
                    source_checkout=source,
                    destination=cache / "derived-linked",
                    repo_root=workspace,
                    checkout_verifier=lambda *_: None,
                )
            second.unlink()

            destination = cache / "derived-interrupted"
            with patch(
                "bench.harbor_terminal._copy_task_tree",
                side_effect=HarborCampaignError("synthetic race"),
            ):
                with self.assertRaisesRegex(HarborCampaignError, "synthetic race"):
                    derive_private_task_dataset(
                        _campaign(),
                        source_checkout=source,
                        destination=destination,
                        repo_root=workspace,
                        checkout_verifier=lambda *_: None,
                    )
            self.assertFalse(destination.exists())

    def test_untracked_ignored_and_executable_bit_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in ("untracked", "ignored", "mode"):
                with self.subTest(case=case):
                    case_root = root / case
                    workspace = case_root / "workspace"
                    cache = case_root / "cache"
                    workspace.mkdir(parents=True)
                    cache.mkdir()
                    source = _make_source_checkout(cache)
                    task = source / "tasks" / "build-cython-ext"
                    if case == "mode":
                        (task / "instruction.md").chmod(0o755)
                        pattern = "blob or mode"
                    else:
                        extra = task / f"{case}.secret"
                        if case == "ignored":
                            exclude = source / ".git" / "info" / "exclude"
                            with exclude.open("a", encoding="utf-8") as handle:
                                handle.write("tasks/build-cython-ext/ignored.secret\n")
                        extra.write_text(CANARY, encoding="utf-8")
                        pattern = "ignored or untracked"
                    with self.assertRaisesRegex(HarborCampaignError, pattern):
                        derive_private_task_dataset(
                            _campaign(),
                            source_checkout=source,
                            destination=cache / "derived",
                            repo_root=workspace,
                            checkout_verifier=lambda *_: None,
                        )


class HarborInvocationTests(unittest.TestCase):
    def _invocation_fixture(self, root: Path):
        campaign, workspace, cache, _, _, policy = _derive_fixture(root)
        jobs = cache / "raw-jobs"
        jobs.mkdir()
        trial = iter_trials(campaign)[0]
        harbor_runtime = _fake_harbor_runtime(campaign, cache)
        agent_source, pycache = _fake_agent_source(cache)
        invocation = build_harbor_invocation(
            campaign,
            trial=trial,
            npm_artifact_admission=_fake_npm_admission(campaign),
            runtime_overlay_admission=_fake_runtime_admission(
                campaign, trial, cache
            ),
            harbor_runtime_admission=harbor_runtime,
            agent_source_root=agent_source,
            python_pycache_root=pycache,
            derived_dataset=policy,
            jobs_dir=jobs,
            base_url=BRIDGE_URL,
            repo_root=workspace,
            ambient_env={
                "PATH": "/usr/bin",
                "HOME": "/tmp/synthetic-home",
                "LANG": "C.UTF-8",
                "DOCKER_HOST": "unix:///synthetic.sock",
                "HF_TOKEN": "must-not-cross-boundary",
                "GITHUB_TOKEN": "must-not-cross-boundary",
            },
            runtime_admission_validator=lambda *_: None,
        )
        trial_dir = invocation.raw_job_dir / "build-cython-ext__AbC1234"
        trial_dir.mkdir(parents=True)
        _write_network_marker(trial_dir / "sparkbench-network-admission.json")
        return campaign, workspace, cache, jobs, policy, invocation

    def test_command_is_one_task_pinned_serial_and_secret_is_env_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign, _, _, _, _, invocation = self._invocation_fixture(
                Path(directory)
            )
            argv = list(invocation.argv)
            serialized_argv = json.dumps(argv)

            self.assertNotIn(CANARY, serialized_argv)
            self.assertIn("OPENAI_API_KEY=${OPENAI_API_KEY}", argv)
            self.assertIn("OPENAI_BASE_URL=${OPENAI_BASE_URL}", argv)
            self.assertEqual(argv.count("--include-task-name"), 1)
            self.assertIn("--force-build", argv)
            self.assertIn("--no-delete", argv)
            self.assertNotIn("--delete", argv)
            self.assertIn("--yes", argv)
            self.assertNotIn("--allow-agent-host", argv)
            self.assertEqual(argv[argv.index("--n-concurrent") + 1], "1")
            self.assertEqual(argv[argv.index("--max-retries") + 1], "0")
            self.assertEqual(
                argv[argv.index("--agent-kwarg") + 1],
                f"version={campaign.agent('qwen-coder').version}",
            )
            self.assertEqual(
                argv[argv.index("--model") + 1],
                f"openai/{campaign.model.served_name}",
            )
            self.assertEqual(
                argv[argv.index("--agent-import-path") + 1],
                "bench.harbor_pinned_agents:PinnedQwenCode",
            )
            self.assertEqual(invocation.timeout_s, 3_600)
            self.assertEqual(
                invocation.env["OPENAI_API_KEY"],
                "sparkbench-relay-placeholder-v1",
            )
            self.assertEqual(invocation.env["OPENAI_BASE_URL"], BRIDGE_URL)
            self.assertEqual(invocation.env["HARBOR_TELEMETRY"], "off")
            self.assertNotIn("HF_TOKEN", invocation.env)
            self.assertNotIn("GITHUB_TOKEN", invocation.env)
            self.assertEqual(invocation.env["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertEqual(
                invocation.env["PYTHONPYCACHEPREFIX"],
                str(invocation.python_pycache_root),
            )
            self.assertEqual(
                invocation.env["PYTHONPATH"], str(invocation.agent_source_root)
            )
            self.assertNotIn(CANARY, repr(invocation))
            self.assertTrue(argv[0].endswith("/.venv/bin/python"))
            self.assertEqual(argv[1], "-B")
            self.assertTrue(argv[2].endswith("/.venv/bin/harbor"))

    def test_harbor_runtime_binds_exact_python_launcher_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            campaign = _campaign()
            fixture = _fake_harbor_runtime(campaign, root)
            executable_bytes = fixture.executable_path.read_bytes()
            python_bytes = fixture.python_path.read_bytes()
            campaign = replace(
                campaign,
                harbor=replace(
                    campaign.harbor,
                    executable_size_bytes=len(executable_bytes),
                    executable_sha256="sha256:"
                    + hashlib.sha256(executable_bytes).hexdigest(),
                    python_size_bytes=len(python_bytes),
                    python_sha256="sha256:"
                    + hashlib.sha256(python_bytes).hexdigest(),
                ),
            )
            with patch(
                "bench.harbor_terminal.verify_normalized_tree",
                return_value=fixture.tree,
            ):
                admitted = verify_harbor_runtime(
                    campaign, fixture.tree.resolved_path, repo_root=workspace
                )
                self.assertEqual(
                    os.readlink(admitted.python_launcher_path),
                    "../../.python-runtime/bin/python3.13",
                )
                admitted.python_launcher_path.unlink()
                admitted.python_launcher_path.symlink_to("../../wrong-python")
                with self.assertRaisesRegex(HarborCampaignError, "launcher"):
                    verify_harbor_runtime(
                        campaign, fixture.tree.resolved_path, repo_root=workspace
                    )

    def test_raw_jobs_are_rejected_inside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign, workspace, cache, _, _, policy = _derive_fixture(root)
            inside_jobs = workspace / "results"
            inside_jobs.mkdir()
            agent_source, pycache = _fake_agent_source(cache)
            with self.assertRaisesRegex(HarborCampaignError, "outside"):
                build_harbor_invocation(
                    campaign,
                    trial=iter_trials(campaign)[0],
                    npm_artifact_admission=_fake_npm_admission(campaign),
                    runtime_overlay_admission=_fake_runtime_admission(
                        campaign, iter_trials(campaign)[0], cache
                    ),
                    harbor_runtime_admission=_fake_harbor_runtime(
                        campaign, cache
                    ),
                    agent_source_root=agent_source,
                    python_pycache_root=pycache,
                    derived_dataset=policy,
                    jobs_dir=inside_jobs,
                    base_url=BRIDGE_URL,
                    repo_root=workspace,
                    runtime_admission_validator=lambda *_: None,
                )

    def test_run_requires_one_live_lock_and_cleans_only_owned_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, _, invocation = self._invocation_fixture(Path(directory))
            docker = _FakeDocker()
            unrelated_old = "a" * 64
            docker.resources["container"][unrelated_old] = "other__abc1234__env"

            def process_runner(_argv, env, _timeout):
                self.assertEqual(
                    env["OPENAI_API_KEY"], "sparkbench-relay-placeholder-v1"
                )
                project = "build-cython-ext__abc1234__env"
                docker.resources["container"]["b" * 64] = project
                docker.resources["network"]["c" * 64] = project
                docker.resources["volume"]["owned-volume"] = project
                docker.resources["container"]["d" * 64] = "other__def5678__env"
                return 0, False

            clock = iter((10.0, 30.0)).__next__
            lock = CampaignLock(descriptor=9)
            status = run_harbor_invocation(
                invocation,
                lock=lock,
                timeout_s=100,
                process_runner=process_runner,
                docker_runner=docker,
                clock=clock,
            )
            self.assertTrue(status.cleanup_succeeded)
            self.assertEqual(status.main_image_id, "sha256:" + "a" * 64)
            self.assertRegex(
                status.main_image_fingerprint or "", r"^sha256:[0-9a-f]{64}$"
            )
            self.assertTrue(status.built_image_cleanup_succeeded)
            self.assertTrue(status.agent_relay_passed)
            self.assertTrue(status.gost_rejected)
            self.assertEqual(status.containers_found, 1)
            self.assertEqual(status.containers_removed, 1)
            self.assertEqual(status.networks_removed, 1)
            self.assertEqual(status.volumes_removed, 1)
            self.assertIn(unrelated_old, docker.resources["container"])
            self.assertIn("d" * 64, docker.resources["container"])
            self.assertFalse(
                any(command[:2] == ["docker", "system"] for command in docker.commands)
            )
            self.assertIn(
                [
                    "docker",
                    "image",
                    "rm",
                    "--force",
                    "build-cython-ext__abc1234__env-main",
                ],
                docker.commands,
            )

            lock.active = False
            with self.assertRaisesRegex(HarborCampaignError, "continuously held"):
                run_harbor_invocation(
                    invocation,
                    lock=lock,
                    timeout_s=100,
                    process_runner=process_runner,
                    docker_runner=docker,
                )

    def test_image_fingerprint_ignores_metadata_but_binds_runtime_shape(self) -> None:
        docker = _FakeDocker()
        docker.image_ids["first"] = "sha256:" + "1" * 64
        docker.image_ids["second"] = "sha256:" + "2" * 64
        docker.image_variants["second"] = "v8"
        docker.image_configs["first"] = {
            "Image": "first-parent",
            "Labels": {"build": "first"},
            "Env": ["PATH=/usr/bin"],
            "Cmd": ["/bin/sh"],
        }
        docker.image_configs["second"] = {
            "Image": "second-parent",
            "Labels": {"build": "second"},
            "Env": ["PATH=/usr/bin"],
            "Cmd": ["/bin/sh"],
        }
        first_id, first_fingerprint, first_arm64 = _inspect_image(
            "first", runner=docker
        )
        second_id, second_fingerprint, second_arm64 = _inspect_image(
            "second", runner=docker
        )
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(first_fingerprint, second_fingerprint)
        self.assertTrue(first_arm64 and second_arm64)

        docker.image_layers["second"] = ["sha256:" + "3" * 64]
        _, changed_fingerprint, _ = _inspect_image("second", runner=docker)
        self.assertNotEqual(first_fingerprint, changed_fingerprint)

        docker.image_layers.pop("second")
        docker.image_configs["second"]["Env"] = ["PATH=/different"]
        _, changed_fingerprint, _ = _inspect_image("second", runner=docker)
        self.assertNotEqual(first_fingerprint, changed_fingerprint)

        docker.image_variants["second"] = "v7"
        with self.assertRaisesRegex(HarborCampaignError, "variant"):
            _inspect_image("second", runner=docker)

    def test_image_fingerprint_rejects_malformed_or_unbounded_metadata(self) -> None:
        docker = _FakeDocker()
        docker.image_architecture["bad-architecture"] = []
        with self.assertRaisesRegex(HarborCampaignError, "identity"):
            _inspect_image("bad-architecture", runner=docker)

        docker.image_variants["bad-variant"] = {}
        with self.assertRaisesRegex(HarborCampaignError, "variant"):
            _inspect_image("bad-variant", runner=docker)

        docker.image_rootfs["extra-rootfs"] = {
            "Type": "layers",
            "Layers": ["sha256:" + "b" * 64],
            "unexpected": True,
        }
        with self.assertRaisesRegex(HarborCampaignError, "identity"):
            _inspect_image("extra-rootfs", runner=docker)

        docker.image_configs["bad-label"] = {"Labels": {"key": 7}}
        with self.assertRaisesRegex(HarborCampaignError, "label"):
            _inspect_image("bad-label", runner=docker)

        docker.image_configs["nonfinite"] = {"Memory": float("nan")}
        with self.assertRaisesRegex(HarborCampaignError, "not JSON"):
            _inspect_image("nonfinite", runner=docker)

        with patch("bench.harbor_terminal.MAX_IMAGE_INSPECT_BYTES", 1):
            with self.assertRaisesRegex(HarborCampaignError, "Could not inspect"):
                _inspect_image("oversized", runner=docker)

    def test_invalid_fingerprint_still_removes_the_exact_built_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, _, invocation = self._invocation_fixture(Path(directory))
            docker = _FakeDocker()
            reference = "build-cython-ext__abc1234__env-main"
            image_id = "sha256:" + "a" * 64
            docker.image_variants[reference] = {}
            status = run_harbor_invocation(
                invocation,
                lock=CampaignLock(descriptor=9),
                timeout_s=100,
                process_runner=lambda *_: (0, False),
                docker_runner=docker,
                clock=iter((10.0, 20.0)).__next__,
            )

            self.assertIsNone(status.main_image_id)
            self.assertIsNone(status.main_image_fingerprint)
            self.assertFalse(status.main_image_arm64)
            self.assertTrue(status.built_image_cleanup_succeeded)
            self.assertTrue(status.cleanup_succeeded)
            self.assertIn(reference, docker.removed_images)
            self.assertIn(image_id, docker.removed_images)

    def test_network_marker_is_exact_owner_only_scalar_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, _, invocation = self._invocation_fixture(Path(directory))
            admitted = load_network_admission(invocation)
            self.assertTrue(all(admitted.values()))
            marker = (
                invocation.raw_job_dir
                / "build-cython-ext__AbC1234"
                / "sparkbench-network-admission.json"
            )
            payload = json.loads(marker.read_text(encoding="ascii"))
            payload["raw_error"] = CANARY
            marker.write_text(json.dumps(payload), encoding="ascii")
            marker.chmod(0o600)
            with self.assertRaisesRegex(HarborCampaignError, "marker"):
                load_network_admission(invocation)

            marker.unlink()
            target = marker.with_name("marker-target")
            _write_network_marker(target)
            marker.symlink_to(target)
            with self.assertRaises(HarborCampaignError):
                load_network_admission(invocation)

    def test_staged_agent_source_and_empty_pycache_are_rechecked_before_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, _, invocation = self._invocation_fixture(Path(directory))
            source = invocation.agent_source_root / "bench" / "harbor_pinned_agents.py"
            source.chmod(0o644)
            with self.assertRaisesRegex(HarborCampaignError, "source mode"):
                run_harbor_invocation(
                    invocation,
                    lock=CampaignLock(descriptor=9),
                    timeout_s=10,
                    process_runner=lambda *_: (0, False),
                    docker_runner=_FakeDocker(),
                )

        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, _, invocation = self._invocation_fixture(Path(directory))
            (invocation.python_pycache_root / "ignored.pyc").write_bytes(CANARY.encode())
            with self.assertRaisesRegex(HarborCampaignError, "must be empty"):
                run_harbor_invocation(
                    invocation,
                    lock=CampaignLock(descriptor=9),
                    timeout_s=10,
                    process_runner=lambda *_: (0, False),
                    docker_runner=_FakeDocker(),
                )

    def test_timeout_is_scalar_and_cleanup_is_certified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, _, invocation = self._invocation_fixture(Path(directory))
            docker = _FakeDocker()
            clock = iter((5.0, 15.0)).__next__
            status = run_harbor_invocation(
                invocation,
                lock=CampaignLock(descriptor=9),
                timeout_s=10,
                process_runner=lambda *_: (None, True),
                docker_runner=docker,
                clock=clock,
            )
            self.assertTrue(status.timed_out)
            self.assertIsNone(status.exit_code)
            self.assertTrue(status.cleanup_succeeded)
            self.assertEqual(status.wall_s, 10.0)

    def test_interrupt_reaps_process_group_and_outer_finally_cleans_resources(self) -> None:
        class InterruptedProcess:
            pid = 24680

            def __init__(self) -> None:
                self.wait_calls = 0

            def wait(self, *, timeout):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise KeyboardInterrupt
                if self.wait_calls == 2:
                    raise subprocess.TimeoutExpired("harbor", timeout)
                return -signal.SIGKILL

        process = InterruptedProcess()
        def interrupt_killpg(_pid, action):
            if action == 0:
                raise ProcessLookupError

        with patch(
            "bench.harbor_terminal.subprocess.Popen", return_value=process
        ), patch(
            "bench.harbor_terminal.os.killpg", side_effect=interrupt_killpg
        ) as killpg:
            with self.assertRaises(KeyboardInterrupt):
                _run_process_group(("harbor",), {"PATH": "/usr/bin"}, 10)
        self.assertEqual(
            [call.args for call in killpg.call_args_list if call.args[1] != 0],
            [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)],
        )
        self.assertEqual(process.wait_calls, 3)

        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, _, invocation = self._invocation_fixture(Path(directory))
            docker = _FakeDocker()
            owned = "b" * 64

            def interrupted_runner(*_):
                docker.resources["container"][owned] = (
                    "build-cython-ext__abc1234__env"
                )
                raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                run_harbor_invocation(
                    invocation,
                    lock=CampaignLock(descriptor=9),
                    timeout_s=10,
                    process_runner=interrupted_runner,
                    docker_runner=docker,
                )
            self.assertNotIn(owned, docker.resources["container"])

    def test_unreaped_process_group_fails_closed_after_sigkill(self) -> None:
        class UnreapedProcess:
            pid = 13579

            def __init__(self) -> None:
                self.wait_calls = 0

            def wait(self, *, timeout):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise KeyboardInterrupt
                raise subprocess.TimeoutExpired("harbor", timeout)

        process = UnreapedProcess()
        with patch(
            "bench.harbor_terminal.subprocess.Popen", return_value=process
        ), patch("bench.harbor_terminal.os.killpg") as killpg:
            with self.assertRaisesRegex(HarborCampaignError, "could not be reaped"):
                _run_process_group(("harbor",), {"PATH": "/usr/bin"}, 10)
        self.assertEqual(process.wait_calls, 3)
        self.assertEqual(
            [call.args[1] for call in killpg.call_args_list],
            [signal.SIGTERM, signal.SIGKILL],
        )

    def test_successful_leader_exit_drains_surviving_process_group(self) -> None:
        class ExitedLeader:
            pid = 97531

            def wait(self, *, timeout):
                return 0

        process = ExitedLeader()
        state = {"alive": True}

        def process_group(_pid, action):
            if action == 0:
                if state["alive"]:
                    return None
                raise ProcessLookupError
            if action == signal.SIGTERM:
                state["alive"] = False
                return None
            raise AssertionError("SIGKILL should not be needed")

        with patch(
            "bench.harbor_terminal.subprocess.Popen", return_value=process
        ) as popen, patch(
            "bench.harbor_terminal.os.killpg", side_effect=process_group
        ) as killpg:
            self.assertEqual(
                _run_process_group(("harbor",), {"PATH": "/usr/bin"}, 10),
                (0, False),
            )
        self.assertEqual(popen.call_args.kwargs["umask"], 0o077)
        self.assertIn(
            (process.pid, signal.SIGTERM),
            [call.args for call in killpg.call_args_list],
        )

    def test_cleanup_fails_closed_when_labeled_resource_cannot_be_inspected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, _, invocation = self._invocation_fixture(Path(directory))
            docker = _FakeDocker()
            identifier = "e" * 64
            docker.resources["container"][identifier] = (
                "build-cython-ext__abc1234__env"
            )
            docker.inspect_failures.add(("container", identifier))
            status = cleanup_harbor_containers(
                invocation,
                baseline=snapshot_harbor_resources(runner=_FakeDocker()),
                lock=CampaignLock(descriptor=9),
                runner=docker,
            )
            self.assertFalse(status.succeeded)
            self.assertIn(identifier, docker.resources["container"])

    def test_global_lock_is_continuous_and_non_reentrant_by_new_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            lock_path = workspace / "results" / ".sparkbench.lock"
            with hold_campaign_lock(workspace) as token:
                self.assertTrue(token.active)
                second = os.open(lock_path, os.O_RDWR)
                try:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    os.close(second)
            self.assertFalse(token.active)


class HarborSummaryTests(unittest.TestCase):
    def _attempts(self, count: int = 12):
        campaign = _campaign()
        attempts = []
        for trial in iter_trials(campaign)[:count]:
            attempts.append(
                HarborAttempt(
                    trial=trial,
                    status=_status(trial),
                    job_result=_job_result(
                        campaign, trial, reward=1 if trial.index % 2 else 0
                    ),
                )
            )
        return campaign, attempts

    def test_full_summary_is_canonical_scalar_only_and_golden(self) -> None:
        campaign, attempts = self._attempts()
        summary = summarize_campaign_results(
            campaign,
            attempts,
            network_policy_patch_digest="sha256:" + "1" * 64,
            npm_artifact_admission=_fake_npm_admission(campaign),
        )
        encoded = canonical_summary_bytes(summary)

        self.assertNotIn(CANARY, encoded.decode())
        self.assertNotIn("raw-id", encoded.decode())
        self.assertNotIn("/local/private/path", encoded.decode())
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["protocol"], "harbor-terminal-scalar-v2")
        self.assertEqual(summary["summary"]["planned_attempts"], 12)
        self.assertEqual(summary["summary"]["attempts"], 12)
        self.assertEqual(summary["summary"]["passed"], 6)
        self.assertEqual(summary["summary"]["completed_results"], 12)
        self.assertEqual(summary["summary"]["unstarted_attempts"], 0)
        self.assertEqual(summary["summary"]["output_tokens"], 318)
        self.assertEqual(len(summary["trials"]), 12)
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            "87e8bd6599952929bd1ef5a21be2fe8cc3616bf0a902a4f2d51006acee02d180",
        )

    def test_partial_and_zero_summary_accept_only_an_exact_prefix(self) -> None:
        campaign, attempts = self._attempts(3)
        summary = summarize_campaign_results(
            campaign,
            attempts,
            network_policy_patch_digest="sha256:" + "2" * 64,
            npm_artifact_admission=_fake_npm_admission(campaign),
            campaign_cutoff_reached=True,
        )
        self.assertEqual(summary["summary"]["attempts"], 3)
        self.assertEqual(summary["summary"]["unstarted_attempts"], 9)
        self.assertTrue(summary["summary"]["campaign_cutoff_reached"])
        self.assertFalse(summary["summary"]["campaign_complete"])

        empty = summarize_campaign_results(
            campaign,
            [],
            network_policy_patch_digest="sha256:" + "2" * 64,
            npm_artifact_admission=_fake_npm_admission(campaign),
            campaign_cutoff_reached=True,
        )
        self.assertEqual(empty["trials"], [])
        self.assertEqual(empty["summary"]["attempts"], 0)
        self.assertEqual(empty["summary"]["unstarted_attempts"], 12)
        self.assertIsNone(empty["summary"]["pass_rate"])
        canonical_summary_bytes(empty)
        attempts[0], attempts[1] = attempts[1], attempts[0]
        with self.assertRaisesRegex(HarborCampaignError, "exact trial_order prefix"):
            summarize_campaign_results(
                campaign,
                attempts,
                network_policy_patch_digest="sha256:" + "2" * 64,
                npm_artifact_admission=_fake_npm_admission(campaign),
            )

    def test_missing_timed_out_result_is_recorded_without_fabrication(self) -> None:
        campaign = _campaign()
        trial = iter_trials(campaign)[0]
        attempt = HarborAttempt(
            trial=trial,
            status=_status(trial, exit_code=None, timed_out=True),
            job_result=None,
        )
        summary = summarize_campaign_results(
            campaign,
            [attempt],
            network_policy_patch_digest="sha256:" + "3" * 64,
            npm_artifact_admission=_fake_npm_admission(campaign),
            campaign_cutoff_reached=True,
        )
        item = summary["trials"][0]
        self.assertEqual(item["exception_class"], "CampaignCutoffError")
        self.assertIsNone(item["reward"])
        self.assertIsNone(item["output_tokens"])
        self.assertEqual(summary["summary"]["missing_results"], 1)

        partial = _job_result(campaign, trial)
        partial.job["stats"]["n_completed_trials"] = 0
        ignored = HarborAttempt(
            trial=trial,
            status=_status(trial, exit_code=None, timed_out=True),
            job_result=partial,
        )
        ignored_summary = summarize_campaign_results(
            campaign,
            [ignored],
            network_policy_patch_digest="sha256:" + "3" * 64,
            npm_artifact_admission=_fake_npm_admission(campaign),
            campaign_cutoff_reached=True,
        )
        self.assertEqual(
            ignored_summary["trials"][0]["exception_class"],
            "CampaignCutoffError",
        )

    def test_paired_agents_compare_runtime_fingerprints_not_build_ids(self) -> None:
        campaign, attempts = self._attempts(2)
        attempts[1] = replace(
            attempts[1],
            status=replace(
                attempts[1].status,
                main_image_id="sha256:" + "f" * 64,
            ),
        )
        summary = summarize_campaign_results(
            campaign,
            attempts,
            network_policy_patch_digest="sha256:" + "6" * 64,
            npm_artifact_admission=_fake_npm_admission(campaign),
        )
        self.assertEqual(summary["summary"]["image_pair_mismatches"], 0)
        self.assertTrue(
            all(item["paired_image_match"] is True for item in summary["trials"])
        )

        attempts[1] = replace(
            attempts[1],
            status=replace(
                attempts[1].status,
                main_image_fingerprint="sha256:" + "e" * 64,
            ),
        )
        summary = summarize_campaign_results(
            campaign,
            attempts,
            network_policy_patch_digest="sha256:" + "6" * 64,
            npm_artifact_admission=_fake_npm_admission(campaign),
        )
        self.assertEqual(summary["summary"]["image_pair_mismatches"], 1)
        self.assertTrue(
            all(item["paired_image_match"] is False for item in summary["trials"])
        )
        self.assertTrue(all(not item["passed"] for item in summary["trials"]))
        self.assertTrue(
            all(
                item["exception_class"] == "HarborProcessError"
                for item in summary["trials"]
            )
        )

    def test_canonical_encoder_rejects_unknown_keys_and_arbitrary_strings(self) -> None:
        campaign, attempts = self._attempts(1)
        valid = summarize_campaign_results(
            campaign,
            attempts,
            network_policy_patch_digest="sha256:" + "5" * 64,
            npm_artifact_admission=_fake_npm_admission(campaign),
        )
        mutations = []
        unknown_root = deepcopy(valid)
        unknown_root["unknown"] = 1
        mutations.append(unknown_root)
        old_schema = deepcopy(valid)
        old_schema["schema_version"] = 1
        old_schema["protocol"] = "harbor-terminal-scalar-v1"
        mutations.append(old_schema)
        unknown_nested = deepcopy(valid)
        unknown_nested["trials"][0]["prompt"] = CANARY
        mutations.append(unknown_nested)
        string_identity = deepcopy(valid)
        string_identity["campaign_id"] = CANARY
        mutations.append(string_identity)
        string_task = deepcopy(valid)
        string_task["trials"][0]["task"] = CANARY
        mutations.append(string_task)
        string_artifact = deepcopy(valid)
        string_artifact["pins"]["npm_artifact_admission"]["artifacts"][0][
            "package"
        ] = CANARY
        mutations.append(string_artifact)
        bool_reward = deepcopy(valid)
        bool_reward["trials"][0]["reward"] = True
        mutations.append(bool_reward)
        bool_total = deepcopy(valid)
        bool_total["summary"]["passed"] = True
        mutations.append(bool_total)
        bool_rate = deepcopy(valid)
        bool_rate["summary"]["pass_rate"] = True
        mutations.append(bool_rate)
        bool_complete = deepcopy(valid)
        bool_complete["summary"]["campaign_complete"] = 1
        mutations.append(bool_complete)
        for mutation in mutations:
            with self.assertRaises(HarborCampaignError):
                canonical_summary_bytes(mutation)

    def test_wrong_identity_version_and_secret_exception_class_are_rejected(self) -> None:
        campaign, attempts = self._attempts(1)
        cases = []
        wrong_task = deepcopy(attempts[0].job_result)
        wrong_task.trial["task_name"] = "build-cython-ext"
        cases.append(wrong_task)
        wrong_version = deepcopy(attempts[0].job_result)
        wrong_version.trial["agent_info"]["version"] = "latest"
        cases.append(wrong_version)
        secret_exception = deepcopy(attempts[0].job_result)
        secret_exception.trial["exception_info"] = {
            "exception_type": CANARY,
            "exception_message": CANARY,
            "exception_traceback": CANARY,
        }
        cases.append(secret_exception)
        bool_total = deepcopy(attempts[0].job_result)
        bool_total.job["n_total_trials"] = True
        cases.append(bool_total)
        bool_retries = deepcopy(attempts[0].job_result)
        bool_retries.job["stats"]["n_retries"] = False
        cases.append(bool_retries)

        for raw in cases:
            with self.subTest(raw=raw.trial.get("task_name")):
                bad = HarborAttempt(
                    trial=attempts[0].trial,
                    status=attempts[0].status,
                    job_result=raw,
                )
                with self.assertRaises(HarborCampaignError):
                    summarize_campaign_results(
                        campaign,
                        [bad],
                        network_policy_patch_digest="sha256:" + "4" * 64,
                        npm_artifact_admission=_fake_npm_admission(campaign),
                    )

    def test_raw_result_loader_rejects_symlink_hardlink_and_oversize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            jobs = root / "jobs"
            jobs.mkdir()
            trial = iter_trials(_campaign())[0]
            invocation = HarborInvocation(
                trial=trial,
                job_name="synthetic-job",
                timeout_s=3_600,
                npm_artifact_admission_digest="sha256:" + "0" * 64,
                runtime_overlay_admission_digest="sha256:" + "9" * 64,
                harbor_runtime_admission_digest="sha256:" + "8" * 64,
                agent_source_admission_digest="sha256:" + "7" * 64,
                task_image="fixture/task:pinned",
                relay_image="node@sha256:" + "1" * 64,
                workspace_root=workspace,
                agent_source_root=root / "synthetic-agent-source",
                python_pycache_root=root / "synthetic-pycache",
                raw_job_dir=jobs / "synthetic-job",
                argv=("harbor",),
                env={},
            )
            real_job = root / "real-job"
            real_job.mkdir()
            (real_job / "result.json").write_text("{}", encoding="utf-8")
            (jobs / "synthetic-job").symlink_to(real_job, target_is_directory=True)
            with self.assertRaisesRegex(HarborCampaignError, "real directory"):
                load_trial_job_result(invocation, jobs_dir=jobs, repo_root=workspace)

            (jobs / "synthetic-job").unlink()
            (jobs / "synthetic-job").mkdir()
            raw = jobs / "synthetic-job" / "result.json"
            raw.write_text("{}", encoding="utf-8")
            hardlink = root / "hardlink"
            os.link(raw, hardlink)
            with self.assertRaisesRegex(HarborCampaignError, "unsafe"):
                load_trial_job_result(invocation, jobs_dir=jobs, repo_root=workspace)
            hardlink.unlink()
            with patch("bench.harbor_terminal.MAX_RAW_JSON_BYTES", 1):
                with self.assertRaisesRegex(HarborCampaignError, "oversized"):
                    load_trial_job_result(
                        invocation, jobs_dir=jobs, repo_root=workspace
                    )

    def test_raw_result_loader_matches_pinned_harbor_job_child_topology(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            jobs = root / "jobs"
            jobs.mkdir()
            campaign = _campaign()
            trial = iter_trials(campaign)[0]
            invocation = HarborInvocation(
                trial=trial,
                job_name="synthetic-live-job",
                timeout_s=3_600,
                npm_artifact_admission_digest="sha256:" + "0" * 64,
                runtime_overlay_admission_digest="sha256:" + "9" * 64,
                harbor_runtime_admission_digest="sha256:" + "8" * 64,
                agent_source_admission_digest="sha256:" + "7" * 64,
                task_image="fixture/task:pinned",
                relay_image="node@sha256:" + "1" * 64,
                workspace_root=workspace,
                agent_source_root=root / "synthetic-agent-source",
                python_pycache_root=root / "synthetic-pycache",
                raw_job_dir=jobs / "synthetic-live-job",
                argv=("harbor",),
                env={},
            )
            job_dir = jobs / invocation.job_name
            child = job_dir / "build-cython-ext__AbC1234"
            child.mkdir(parents=True)
            raw = _job_result(campaign, trial)
            raw.trial["trial_name"] = child.name
            (job_dir / "result.json").write_text(
                json.dumps(raw.job), encoding="utf-8"
            )
            (child / "result.json").write_text(
                json.dumps(raw.trial), encoding="utf-8"
            )
            loaded = load_trial_job_result(
                invocation, jobs_dir=jobs, repo_root=workspace
            )
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.trial["task_name"], "terminal-bench/build-cython-ext")


if __name__ == "__main__":
    unittest.main()
