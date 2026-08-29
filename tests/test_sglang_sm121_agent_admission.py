"""Contracts for the non-inference current-SM121 agent parser preflight."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.execution_admission import model_execution_blocker
from bench.manifest import (
    ManifestError,
    load_models,
    load_suite,
    validate_benchmark_selection,
    validate_model,
)
from bench.runner import create_plan
from bench.runtime import RuntimeErrorWithContext, start_sglang
from bench.sglang_sm121_agent_admission import (
    SM121_AGENT_ADMISSION_ARGS,
    SM121_AGENT_ADMISSION_PROFILE_ID,
    SM121_AGENT_ADMISSION_STATIC_PROBE_ID,
    SM121_AGENT_ADMISSION_STATIC_PROBE_SCHEMA_VERSION,
    SM121AgentAdmissionError,
    _STATIC_PROBE_SCRIPT,
    is_sm121_agent_admission_candidate,
    probe_sm121_agent_parser_static_preflight,
    validate_sm121_agent_admission_profile,
    validate_sm121_agent_parser_static_probe,
)
from bench.sglang_sm121_storage import (
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_LOCAL_IMAGE_TAG,
    SM121_STORAGE_SOURCE_TREE,
)
from sparkbench import build_parser, command_sm121_agent_parser_preflight


ROOT = Path(__file__).resolve().parents[1]


def _completed(
    *, stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


def _image_inspection() -> dict[str, object]:
    return {
        "Id": SM121_STORAGE_LOCAL_IMAGE_ID,
        "RepoTags": [SM121_STORAGE_LOCAL_IMAGE_TAG],
        "Os": "linux",
        "Architecture": "arm64",
        "Config": {
            "Labels": {
                "ai.sglang.build.commit": SM121_STORAGE_SOURCE_TREE,
                "org.opencontainers.image.revision": SM121_STORAGE_SOURCE_TREE,
            }
        },
    }


class SM121AgentAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_models(ROOT / "manifests" / "models.toml")[
            SM121_AGENT_ADMISSION_PROFILE_ID
        ]

    def test_manifest_profile_is_exact_and_tombstoned(self) -> None:
        self.assertTrue(is_sm121_agent_admission_candidate(self.model))
        validate_sm121_agent_admission_profile(self.model)
        self.assertEqual(self.model.args, SM121_AGENT_ADMISSION_ARGS)
        self.assertIn(
            "dedicated parser/tool admission controller",
            model_execution_blocker(self.model) or "",
        )
        smoke = load_suite(ROOT / "manifests" / "suites" / "smoke.toml")
        with self.assertRaisesRegex(ManifestError, "tombstoned"):
            validate_benchmark_selection(self.model, smoke)

    def test_profile_rejects_parser_and_policy_drift(self) -> None:
        changed_request = replace(
            self.model,
            request_body_json='{"chat_template_kwargs":{"enable_thinking":false}}',
        )
        with self.assertRaises(SM121AgentAdmissionError):
            validate_sm121_agent_admission_profile(changed_request)

        args = list(self.model.args)
        args[args.index("--tool-call-parser") + 1] = "hermes"
        changed_parser = replace(self.model, args=tuple(args))
        with self.assertRaises(SM121AgentAdmissionError):
            validate_sm121_agent_admission_profile(changed_parser)

        forbidden = replace(
            self.model,
            args=(*self.model.args, "--enable-auto-tool-choice"),
        )
        with self.assertRaises(SM121AgentAdmissionError):
            validate_sm121_agent_admission_profile(forbidden)

        for field, value in (
            ("fetch_allow_patterns", ("surprise",)),
            ("runtime_binary", "unexpected-runtime"),
            ("prefix_cache_mode", "unadmitted-cache"),
            ("sglang_allow_hf_metadata_probe", 1),
        ):
            with self.subTest(field=field):
                changed = replace(self.model, **{field: value})
                with self.assertRaises(SM121AgentAdmissionError):
                    validate_sm121_agent_admission_profile(changed)
                with self.assertRaises(ManifestError):
                    validate_model(changed)

    def test_static_probe_is_image_bound_and_never_requests_gpu_or_network(self) -> None:
        runner = Mock(
            side_effect=(
                _completed(stdout=json.dumps(_image_inspection())),
                _completed(
                    stdout=json.dumps(
                        {
                            "reasoning_parser_qwen3": True,
                            "tool_call_parser_qwen3_coder": True,
                        }
                    )
                ),
            )
        )

        with patch(
            "bench.sglang_sm121_agent_admission.secrets.token_hex",
            return_value="a" * 32,
        ):
            probe = probe_sm121_agent_parser_static_preflight(
                self.model, runner=runner
            )

        self.assertEqual(
            probe,
            {
                "schema_version": SM121_AGENT_ADMISSION_STATIC_PROBE_SCHEMA_VERSION,
                "probe_id": SM121_AGENT_ADMISSION_STATIC_PROBE_ID,
                "docker_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
                "source_tree": SM121_STORAGE_SOURCE_TREE,
                "reasoning_parser_qwen3": True,
                "tool_call_parser_qwen3_coder": True,
            },
        )
        self.assertEqual(runner.call_count, 2)
        image_inspect = runner.call_args_list[0].args[0]
        self.assertEqual(
            image_inspect,
            [
                "docker",
                "image",
                "inspect",
                SM121_STORAGE_LOCAL_IMAGE_TAG,
                "--format",
                "{{json .}}",
            ],
        )
        launch = runner.call_args_list[1].args[0]
        self.assertEqual(
            launch,
            [
                "docker",
                "run",
                "--rm",
                "--pull=never",
                "--name",
                "sparkbench-sm121-agent-parser-" + "a" * 32,
                "--label",
                "io.sparkbench.sm121-agent-parser-preflight=" + "a" * 32,
                "--runtime",
                "runc",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "64",
                "--memory",
                "2g",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,noexec,size=64m",
                "--env",
                "HOME=/tmp",
                "--env",
                "XDG_CACHE_HOME=/tmp",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--entrypoint",
                "python3",
                SM121_STORAGE_LOCAL_IMAGE_ID,
                "-c",
                _STATIC_PROBE_SCRIPT,
            ],
        )
        for forbidden_flag in (
            "--gpus",
            "--device",
            "--volume",
            "-v",
            "--mount",
            "--volumes-from",
            "--publish",
            "-p",
            "-P",
            "--privileged",
            "--cap-add",
            "--ipc",
            "--pid",
            "--uts",
            "--userns",
            "--cgroupns",
            "--host",
        ):
            with self.subTest(forbidden_flag=forbidden_flag):
                self.assertNotIn(forbidden_flag, launch)

    def test_static_probe_rejects_absent_registry_and_timing_like_fields(self) -> None:
        runner = Mock(
            side_effect=(
                _completed(stdout=json.dumps(_image_inspection())),
                _completed(
                    stdout=json.dumps(
                        {
                            "reasoning_parser_qwen3": False,
                            "tool_call_parser_qwen3_coder": True,
                        }
                    )
                ),
            )
        )
        with self.assertRaises(SM121AgentAdmissionError):
            probe_sm121_agent_parser_static_preflight(self.model, runner=runner)

        valid = {
            "schema_version": SM121_AGENT_ADMISSION_STATIC_PROBE_SCHEMA_VERSION,
            "probe_id": SM121_AGENT_ADMISSION_STATIC_PROBE_ID,
            "docker_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
            "source_tree": SM121_STORAGE_SOURCE_TREE,
            "reasoning_parser_qwen3": True,
            "tool_call_parser_qwen3_coder": True,
        }
        valid["wall_s"] = 1.0
        with self.assertRaises(SM121AgentAdmissionError):
            validate_sm121_agent_parser_static_probe(valid)

    def test_static_probe_fails_closed_on_image_or_json_drift(self) -> None:
        malformed_image = Mock(return_value=_completed(stdout="not-json"))
        with self.assertRaises(SM121AgentAdmissionError):
            probe_sm121_agent_parser_static_preflight(
                self.model, runner=malformed_image
            )

        label_drift = _image_inspection()
        label_drift["Config"] = {
            "Labels": {
                "ai.sglang.build.commit": "0" * 40,
                "org.opencontainers.image.revision": SM121_STORAGE_SOURCE_TREE,
            }
        }
        changed_image = Mock(return_value=_completed(stdout=json.dumps(label_drift)))
        with self.assertRaises(SM121AgentAdmissionError):
            probe_sm121_agent_parser_static_preflight(
                self.model, runner=changed_image
            )

        malformed_result = Mock(
            side_effect=(
                _completed(stdout=json.dumps(_image_inspection())),
                _completed(stdout='{"reasoning_parser_qwen3":true,'),
            )
        )
        with self.assertRaises(SM121AgentAdmissionError):
            probe_sm121_agent_parser_static_preflight(
                self.model, runner=malformed_result
            )

        duplicate_result = Mock(
            side_effect=(
                _completed(stdout=json.dumps(_image_inspection())),
                _completed(
                    stdout=(
                        '{"reasoning_parser_qwen3":true,'
                        '"reasoning_parser_qwen3":true,'
                        '"tool_call_parser_qwen3_coder":true}'
                    )
                ),
            )
        )
        with self.assertRaises(SM121AgentAdmissionError):
            probe_sm121_agent_parser_static_preflight(
                self.model, runner=duplicate_result
            )

    def test_timeout_cleanup_requires_the_owned_random_label(self) -> None:
        nonce = "b" * 32
        timeout = subprocess.TimeoutExpired(cmd=["docker", "run"], timeout=30)
        runner = Mock(
            side_effect=(
                _completed(stdout=json.dumps(_image_inspection())),
                timeout,
                _completed(stdout=nonce + "\n"),
                _completed(),
            )
        )
        with patch(
            "bench.sglang_sm121_agent_admission.secrets.token_hex",
            return_value=nonce,
        ), self.assertRaises(SM121AgentAdmissionError):
            probe_sm121_agent_parser_static_preflight(self.model, runner=runner)

        name = "sparkbench-sm121-agent-parser-" + nonce
        self.assertEqual(
            runner.call_args_list[2].args[0],
            [
                "docker",
                "container",
                "inspect",
                name,
                "--format",
                "{{ index .Config.Labels \"io.sparkbench.sm121-agent-parser-preflight\" }}",
            ],
        )
        self.assertEqual(
            runner.call_args_list[3].args[0],
            ["docker", "container", "rm", "--force", name],
        )

    def test_timeout_cleanup_does_not_remove_an_unverified_container(self) -> None:
        nonce = "c" * 32
        runner = Mock(
            side_effect=(
                _completed(stdout=json.dumps(_image_inspection())),
                subprocess.TimeoutExpired(cmd=["docker", "run"], timeout=30),
                _completed(stdout="other-run\n"),
            )
        )
        with patch(
            "bench.sglang_sm121_agent_admission.secrets.token_hex",
            return_value=nonce,
        ), self.assertRaises(SM121AgentAdmissionError):
            probe_sm121_agent_parser_static_preflight(self.model, runner=runner)
        self.assertEqual(runner.call_count, 3)

    def test_tombstone_rejects_plan_and_runtime_before_side_effects(self) -> None:
        smoke = load_suite(ROOT / "manifests" / "suites" / "smoke.toml")
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            with (
                patch("bench.runner.validate_benchmark_selection") as selection,
                self.assertRaisesRegex(
                    RuntimeError, "dedicated parser/tool admission controller"
                ),
            ):
                create_plan(
                    model=self.model,
                    suite=smoke,
                    results_root=results,
                    models_path=Path("models.toml"),
                    suite_path=Path("smoke.toml"),
                )
            self.assertFalse(results.exists())
            selection.assert_not_called()

        abort_check = Mock()
        on_server_created = Mock()
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "uncreated-workspace"
            with (
                patch("bench.runtime._existing_container") as existing,
                patch("bench.runtime._port_is_free") as port_free,
                patch("bench.runtime._exact_sglang_snapshot") as snapshot,
                patch("bench.runtime._resolve_sglang_source_overlays") as overlays,
                patch("bench.runtime._run") as run,
                self.assertRaisesRegex(
                    RuntimeErrorWithContext,
                    "dedicated parser/tool admission controller",
                ),
            ):
                start_sglang(
                    self.model,
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
            run,
        ):
            probe.assert_not_called()

    def test_command_loads_the_exact_profile_and_prints_only_validated_scalars(self) -> None:
        expected = {
            "schema_version": SM121_AGENT_ADMISSION_STATIC_PROBE_SCHEMA_VERSION,
            "probe_id": SM121_AGENT_ADMISSION_STATIC_PROBE_ID,
            "docker_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
            "source_tree": SM121_STORAGE_SOURCE_TREE,
            "reasoning_parser_qwen3": True,
            "tool_call_parser_qwen3_coder": True,
        }
        output = io.StringIO()
        with (
            patch("sparkbench.load_models", return_value={
                SM121_AGENT_ADMISSION_PROFILE_ID: self.model
            }) as load,
            patch("sparkbench.validate_sm121_agent_admission_profile") as validate,
            patch(
                "sparkbench.probe_sm121_agent_parser_static_preflight",
                return_value=expected,
            ) as probe,
            patch(
                "sparkbench.validate_sm121_agent_parser_static_probe",
                return_value=expected,
            ) as validate_probe,
            redirect_stdout(output),
        ):
            status = command_sm121_agent_parser_preflight(
                SimpleNamespace(models=Path("ignored.toml"))
            )
        self.assertEqual(status, 0)
        load.assert_called_once_with(Path("ignored.toml"))
        validate.assert_called_once_with(self.model)
        probe.assert_called_once_with(self.model)
        validate_probe.assert_called_once_with(expected)
        self.assertEqual(json.loads(output.getvalue()), expected)

    def test_cli_registers_a_no_selection_parser_preflight(self) -> None:
        args = build_parser().parse_args(["sm121-agent-parser-preflight"])
        self.assertIs(args.function, command_sm121_agent_parser_preflight)
        self.assertEqual(args.models, ROOT / "manifests" / "models.toml")
