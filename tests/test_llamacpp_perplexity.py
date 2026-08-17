from __future__ import annotations

import hashlib
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from bench.llamacpp_perplexity import (
    LlamaCppPerplexityError,
    PINNED_DATASET_SHA256,
    PINNED_PERPLEXITY_BINARY,
    PINNED_PERPLEXITY_BINARY_SHA256,
    PINNED_PERPLEXITY_BINARY_SIZE_BYTES,
    PINNED_RUNTIME_REVISION,
    PINNED_RUNTIME_SOURCE,
    ProcessOutcome,
    _invoke_perplexity,
    parse_final_perplexity,
    run_llamacpp_perplexity,
)
from sparkbench import build_parser, command_perplexity


MODEL_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _model(**changes: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": "qwen-fixture-llamacpp",
        "backend": "llamacpp",
        "source": "example/model",
        "revision": MODEL_REVISION,
        "model_file": "model.gguf",
        "model_digest": "sha256:" + "1" * 64,
        "model_size_bytes": 16,
        "runtime_source_dir": str(PINNED_RUNTIME_SOURCE),
        "runtime_revision": PINNED_RUNTIME_REVISION,
        "native_context": 262_144,
        "max_context": 32_768,
        "cache_dir": "project",
        "quantization": "fixture",
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _cleanup(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "pid": 321,
        "returncode": 0,
        "timed_out": False,
        "terminate_requested": False,
        "kill_requested": False,
        "process_reaped": True,
        "process_group_isolated": True,
    }
    values.update(changes)
    return values


class LlamaCppPerplexityTests(unittest.TestCase):
    def test_certified_runtime_and_dataset_pins_are_exact(self) -> None:
        self.assertEqual(
            PINNED_RUNTIME_SOURCE,
            Path("/home/xlz/.cache/sparkbench/llama.cpp-b10453"),
        )
        self.assertEqual(
            PINNED_RUNTIME_REVISION,
            "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70",
        )
        self.assertEqual(
            PINNED_PERPLEXITY_BINARY,
            PINNED_RUNTIME_SOURCE / "build/bin/llama-perplexity",
        )
        self.assertEqual(PINNED_PERPLEXITY_BINARY_SIZE_BYTES, 51_551_520)
        self.assertEqual(
            PINNED_PERPLEXITY_BINARY_SHA256,
            "sha256:31ec19f4d8c071d691f7f4dde4a432771a50872eb29d87e9408a39f366ed5972",
        )
        self.assertEqual(
            PINNED_DATASET_SHA256,
            "sha256:173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08",
        )

    def test_cli_requires_explicit_dataset_chunks_and_timeout(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "perplexity",
                "qwen-model",
                "--dataset",
                "/datasets/wikitext.txt",
                "--chunks",
                "7",
                "--timeout",
                "900",
            ]
        )
        self.assertIs(args.function, command_perplexity)
        self.assertEqual(args.ctx_size, 512)
        self.assertEqual(args.chunks, 7)
        self.assertEqual(args.timeout, 900.0)

    def test_split_gguf_profile_fails_closed_before_worker_setup(self) -> None:
        model = _model(
            model_file=None,
            model_digest=None,
            model_size_bytes=None,
            model_shards=(
                SimpleNamespace(
                    path="model-00001-of-00002.gguf",
                    digest="sha256:" + "1" * 64,
                    size_bytes=8,
                ),
                SimpleNamespace(
                    path="model-00002-of-00002.gguf",
                    digest="sha256:" + "2" * 64,
                    size_bytes=8,
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                LlamaCppPerplexityError,
                "does not yet support split GGUF profiles",
            ):
                run_llamacpp_perplexity(
                    model=model,
                    workspace=root,
                    results_root=root / "results",
                    dataset=root / "dataset.txt",
                    chunks=1,
                    timeout_s=60,
                )
            self.assertFalse((root / "results").exists())

    def test_parser_accepts_terminal_ppl_and_rejects_malformed_output(self) -> None:
        self.assertEqual(
            parse_final_perplexity(
                "[1]12.0000,\n",
                "\x1b[32mFinal estimate: PPL = 7.1250 +/- 0.03125\x1b[0m\n",
            ),
            (7.125, 0.03125),
        )
        with self.assertRaisesRegex(
            LlamaCppPerplexityError, "final PPL estimate"
        ):
            parse_final_perplexity("perplexity initialized", "complete")
        with self.assertRaisesRegex(
            LlamaCppPerplexityError, "invalid uncertainty"
        ):
            parse_final_perplexity(
                "", "Final estimate: PPL = 7.1250 +/- -0.1"
            )

    def test_gguf_hash_mismatch_aborts_without_launching_process(self) -> None:
        model_payload = b"fixture GGUF"
        binary_payload = b"fixture binary"
        dataset_payload = b"fixture Wikitext"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            runtime_source = root / "llama.cpp-b10453"
            binary = runtime_source / "build/bin/llama-perplexity"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(binary_payload)
            binary.chmod(0o755)
            model_path = (
                workspace
                / "data/huggingface/hub/models--example--model/snapshots"
                / MODEL_REVISION
                / "model.gguf"
            )
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(model_payload)
            dataset = root / "wikitext.txt"
            dataset.write_bytes(dataset_payload)
            model = _model(
                runtime_source_dir=str(runtime_source),
                model_size_bytes=len(model_payload),
                model_digest="sha256:" + "0" * 64,
            )
            telemetry = Mock()
            revision = subprocess.CompletedProcess(
                [], 0, stdout=PINNED_RUNTIME_REVISION + "\n", stderr=""
            )
            with (
                patch(
                    "bench.llamacpp_perplexity.PINNED_RUNTIME_SOURCE",
                    runtime_source,
                ),
                patch(
                    "bench.llamacpp_perplexity.PINNED_PERPLEXITY_BINARY",
                    binary,
                ),
                patch(
                    "bench.llamacpp_perplexity.PINNED_PERPLEXITY_BINARY_SIZE_BYTES",
                    len(binary_payload),
                ),
                patch(
                    "bench.llamacpp_perplexity.PINNED_PERPLEXITY_BINARY_SHA256",
                    _digest(binary_payload),
                ),
                patch(
                    "bench.llamacpp_perplexity.PINNED_DATASET_SHA256",
                    _digest(dataset_payload),
                ),
                patch(
                    "bench.llamacpp_perplexity.subprocess.run",
                    return_value=revision,
                ),
                patch("bench.llamacpp_perplexity.subprocess.Popen") as popen,
                patch("bench.llamacpp_perplexity._preflight") as preflight,
                patch(
                    "bench.llamacpp_perplexity.TelemetrySampler",
                    return_value=telemetry,
                ),
                patch(
                    "bench.llamacpp_perplexity._telemetry_summaries",
                    return_value={},
                ),
            ):
                summary = run_llamacpp_perplexity(
                    model=model,
                    workspace=workspace,
                    results_root=workspace / "results",
                    dataset=dataset,
                    chunks=1,
                    timeout_s=60,
                )

            run_dir = Path(summary["run_dir"])
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(summary["status"], "aborted")
            self.assertIn("GGUF SHA-256 mismatch", str(summary["error"]))
            self.assertTrue((run_dir / "logs/stdout.log").is_file())
            self.assertTrue((run_dir / "logs/stderr.log").is_file())
            self.assertEqual(events[-1]["event"], "run_aborted")
            popen.assert_not_called()
            preflight.assert_not_called()
            telemetry.start.assert_called_once()
            telemetry.stop.assert_called_once()

    def test_timeout_terminates_and_reaps_private_process_group(self) -> None:
        class FakeProcess:
            pid = 4321

            def __init__(self) -> None:
                self.returncode: int | None = None
                self.calls = 0

            def communicate(
                self, timeout: float | None = None
            ) -> tuple[str, str]:
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired("llama-perplexity", timeout)
                self.returncode = -signal.SIGTERM
                return "partial stdout", "timeout stderr"

            def poll(self) -> int | None:
                return self.returncode

        process = FakeProcess()
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch(
                    "bench.llamacpp_perplexity.subprocess.Popen",
                    return_value=process,
                ) as popen,
                patch("bench.llamacpp_perplexity.os.killpg") as killpg,
            ):
                outcome = _invoke_perplexity(
                    command=("/pinned/llama-perplexity", "--offline"),
                    cwd=Path(temporary),
                    timeout_s=1,
                )

        self.assertIn("deadline", str(outcome.error))
        self.assertTrue(outcome.cleanup["timed_out"])
        self.assertTrue(outcome.cleanup["terminate_requested"])
        self.assertTrue(outcome.cleanup["process_reaped"])
        self.assertFalse(outcome.cleanup["kill_requested"])
        killpg.assert_called_once_with(process.pid, signal.SIGTERM)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertIs(popen.call_args.kwargs["stdin"], subprocess.DEVNULL)
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(environment["LLAMA_ARG_OFFLINE"], "1")
        self.assertNotIn("HF_TOKEN", environment)
        self.assertNotIn("HTTPS_PROXY", environment)

    def test_success_persists_metrics_provenance_journal_logs_and_telemetry(
        self,
    ) -> None:
        verification = {
            "runtime_source_revision": PINNED_RUNTIME_REVISION,
            "runtime_binary_path": str(PINNED_PERPLEXITY_BINARY),
            "runtime_binary_sha256": PINNED_PERPLEXITY_BINARY_SHA256,
            "runtime_binary_size_bytes": PINNED_PERPLEXITY_BINARY_SIZE_BYTES,
            "model_source": "example/model",
            "model_revision": MODEL_REVISION,
            "model_file": "model.gguf",
            "model_path": "/cache/model.gguf",
            "model_size_bytes": 16,
            "model_sha256": "sha256:" + "1" * 64,
            "dataset_path": "/datasets/wikitext.txt",
            "dataset_size_bytes": 100,
            "dataset_sha256": PINNED_DATASET_SHA256,
        }
        outcome = ProcessOutcome(
            stdout="[1]8.0,\n",
            stderr="Final estimate: PPL = 7.5000 +/- 0.02500\n",
            wall_time_s=12.5,
            cleanup=_cleanup(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            dataset = Path(temporary) / "wikitext.txt"
            dataset.write_text("private fixture content", encoding="utf-8")
            telemetry = Mock()
            with (
                patch(
                    "bench.llamacpp_perplexity.verify_perplexity_inputs",
                    return_value=verification,
                ),
                patch("bench.llamacpp_perplexity._preflight") as preflight,
                patch(
                    "bench.llamacpp_perplexity._invoke_perplexity",
                    return_value=outcome,
                ) as invoke,
                patch(
                    "bench.llamacpp_perplexity.TelemetrySampler",
                    return_value=telemetry,
                ),
                patch(
                    "bench.llamacpp_perplexity._telemetry_summaries",
                    return_value={"llamacpp_perplexity": {"samples": 2}},
                ),
            ):
                summary = run_llamacpp_perplexity(
                    model=_model(),
                    workspace=workspace,
                    results_root=workspace / "results",
                    dataset=dataset,
                    chunks=3,
                    ctx_size=512,
                    timeout_s=900,
                )

            run_dir = Path(summary["run_dir"])
            plan = json.loads((run_dir / "plan.json").read_text())
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            persisted = "\n".join(
                (run_dir / name).read_text()
                for name in (
                    "plan.json",
                    "events.jsonl",
                    "result.json",
                    "summary.json",
                )
            )
            stderr_log = (run_dir / "logs/stderr.log").read_text()

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["metrics"]["perplexity"], 7.5)
        self.assertEqual(summary["metrics"]["uncertainty"], 0.025)
        self.assertEqual(summary["metrics"]["wall_time_s"], 12.5)
        self.assertEqual(events[-1]["event"], "run_complete")
        self.assertIn("artifact_verified", [event["event"] for event in events])
        self.assertEqual(
            plan["argv"][-7:],
            [
                "--ctx-size",
                "512",
                "--n-gpu-layers",
                "all",
                "--flash-attn",
                "on",
                "--offline",
            ],
        )
        self.assertNotIn("--save-all-logits", plan["argv"])
        self.assertNotIn("private fixture content", persisted)
        self.assertNotIn("HF_TOKEN", persisted)
        self.assertEqual(stderr_log, outcome.stderr)
        preflight.assert_called_once()
        invoke.assert_called_once()
        telemetry.start.assert_called_once()
        telemetry.stop.assert_called_once()

    def test_parse_failure_is_terminal_aborted_with_reaped_cleanup(self) -> None:
        verification = {
            "dataset_sha256": PINNED_DATASET_SHA256,
            "model_sha256": "sha256:" + "1" * 64,
        }
        outcome = ProcessOutcome(
            stdout="perplexity progress without terminal estimate",
            stderr="",
            wall_time_s=1.0,
            cleanup=_cleanup(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            telemetry = Mock()
            with (
                patch(
                    "bench.llamacpp_perplexity.verify_perplexity_inputs",
                    return_value=verification,
                ),
                patch("bench.llamacpp_perplexity._preflight"),
                patch(
                    "bench.llamacpp_perplexity._invoke_perplexity",
                    return_value=outcome,
                ),
                patch(
                    "bench.llamacpp_perplexity.TelemetrySampler",
                    return_value=telemetry,
                ),
                patch(
                    "bench.llamacpp_perplexity._telemetry_summaries",
                    return_value={},
                ),
            ):
                summary = run_llamacpp_perplexity(
                    model=_model(),
                    workspace=workspace,
                    results_root=workspace / "results",
                    dataset=Path(temporary) / "wikitext.txt",
                    chunks=1,
                    timeout_s=60,
                )

            events = [
                json.loads(line)
                for line in (
                    Path(summary["run_dir"]) / "events.jsonl"
                ).read_text().splitlines()
            ]

        self.assertEqual(summary["status"], "aborted")
        self.assertIn("final PPL estimate", str(summary["error"]))
        self.assertTrue(summary["cleanup_proof"]["process_reaped"])
        self.assertEqual(events[-1]["event"], "run_aborted")


if __name__ == "__main__":
    unittest.main()
