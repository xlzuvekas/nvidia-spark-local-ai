from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.runtime import (
    ManagedServer,
    RuntimeErrorWithContext,
    start_server,
    start_sglang,
)


def _completed(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


class SGLangRuntimeTests(unittest.TestCase):
    def _model(self) -> SimpleNamespace:
        digest = (
            "sha256:3f51e3b127bd0fe8f261a84c6ad54ce"
            "42bdb65eb2e57e228a9f6359e89bd08ec"
        )
        return SimpleNamespace(
            backend="sglang",
            id="phi-4-multimodal-instruct-nvfp4",
            source="nvidia/Phi-4-multimodal-instruct-NVFP4",
            revision="617cfabb9ad6c2c6e318fd21c1961536b84f65a1",
            served_name="nvidia/Phi-4-multimodal-instruct-NVFP4",
            image="scitrera/dgx-spark-sglang:0.5.10rc0",
            image_digest=digest,
            resolved_image=f"scitrera/dgx-spark-sglang@{digest}",
            cache_dir="project",
            startup_timeout_s=1200,
            run_identity="run-1",
            args=[
                "--served-model-name",
                "nvidia/Phi-4-multimodal-instruct-NVFP4",
                "--host",
                "0.0.0.0",
                "--port",
                "30000",
                "--quantization",
                "modelopt_fp4",
            ],
        )

    def test_launch_is_offline_digest_pinned_and_snapshot_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            model = self._model()
            snapshot = (
                workspace
                / "data"
                / "huggingface"
                / "hub"
                / "models--nvidia--Phi-4-multimodal-instruct-NVFP4"
                / "snapshots"
                / model.revision
            )
            snapshot.mkdir(parents=True)
            with (
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime.wait_for_endpoint", return_value=2.5) as wait,
                patch(
                    "bench.runtime._run",
                    return_value=_completed(stdout="container-id\n"),
                ) as run,
            ):
                server = start_sglang(
                    model, workspace=workspace, allow_download=True
                )

        launch = run.call_args.args[0]
        self.assertEqual(launch[:4], ["docker", "run", "--detach", "--pull=never"])
        self.assertIn("127.0.0.1:30000:30000", launch)
        self.assertIn("ai.sparkbench.managed=true", launch)
        self.assertIn("ai.sparkbench.run=run-1", launch)
        self.assertIn("ai.sparkbench.backend=sglang", launch)
        self.assertIn("HF_HUB_OFFLINE=1", launch)
        self.assertIn("TRANSFORMERS_OFFLINE=1", launch)
        self.assertNotIn("HF_TOKEN", launch)
        self.assertIn(
            f"{workspace / 'data' / 'huggingface'}:/root/.cache/huggingface:ro",
            launch,
        )
        image_index = launch.index(model.resolved_image)
        self.assertEqual(
            launch[image_index + 1 : image_index + 4],
            ["serve", "--model-path", (
                "/root/.cache/huggingface/hub/"
                "models--nvidia--Phi-4-multimodal-instruct-NVFP4/"
                f"snapshots/{model.revision}"
            )],
        )
        self.assertEqual(
            launch[launch.index("--quantization") + 1], "modelopt_fp4"
        )
        self.assertEqual(server.backend, "sglang")
        self.assertEqual(server.base_url, "http://127.0.0.1:30000/v1")
        wait.assert_called_once_with(server.base_url, 1200.0, "container-id")

    def test_unmanaged_named_container_is_never_replaced(self) -> None:
        with patch(
            "bench.runtime._existing_container", return_value=("abc", False, "")
        ):
            with self.assertRaisesRegex(
                RuntimeErrorWithContext, "unmanaged container sparkbench-sglang"
            ):
                start_sglang(self._model(), workspace=Path("/unused"))

    def test_dispatch_forwards_append_safe_log_path(self) -> None:
        model = self._model()
        log_path = Path("/mock/run/server/server.log")
        expected = object()
        with patch("bench.runtime.start_sglang", return_value=expected) as start:
            actual = start_server(
                model,
                workspace=Path("/mock/workspace"),
                allow_download=True,
                server_log_path=log_path,
            )
        self.assertIs(actual, expected)
        start.assert_called_once_with(
            model,
            workspace=Path("/mock/workspace"),
            allow_download=True,
            server_log_path=log_path,
        )

    def test_owned_sglang_stop_checks_exact_run_label(self) -> None:
        server = ManagedServer(
            backend="sglang",
            base_url="http://127.0.0.1:30000/v1",
            container_id="container-id",
            run_identity="run-1",
        )
        with patch(
            "bench.runtime._run",
            side_effect=[
                _completed(stdout="true run-1\n"),
                _completed(),
                _completed(),
            ],
        ) as run:
            server.stop()

        self.assertEqual(
            [call.args[0][:2] for call in run.call_args_list],
            [["docker", "inspect"], ["docker", "stop"], ["docker", "rm"]],
        )
        self.assertIsNone(server.container_id)


if __name__ == "__main__":
    unittest.main()
