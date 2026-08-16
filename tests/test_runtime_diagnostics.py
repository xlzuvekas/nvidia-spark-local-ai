from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.runtime import (
    RuntimeErrorWithContext,
    start_server,
    start_vllm,
)


def _completed(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


class VllmStartupDiagnosticsTests(unittest.TestCase):
    def test_failed_start_appends_full_logs_before_owned_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server_log = root / "run" / "server" / "server.log"
            server_log.parent.mkdir(parents=True)
            server_log.write_text("prior attempt without newline")
            model = SimpleNamespace(
                source="example/model",
                image="example/image",
                args=[],
                served_name="example/model",
                startup_timeout_s=30,
                cache_dir="project",
                cached=True,
                run_identity="run-1",
            )
            full_logs = "".join(
                f"startup-line-{index:03d}\n" for index in range(150)
            )
            commands: list[list[str]] = []

            def run(
                command: list[str],
                *,
                check: bool = True,
                timeout: float | None = None,
            ) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                operation = command[1]
                if operation == "run":
                    return _completed(stdout="container-id\n")
                if operation == "logs":
                    return _completed(stdout=full_logs)
                if operation == "inspect":
                    return _completed(stdout="true run-1\n")
                if operation in {"stop", "rm"}:
                    return _completed()
                raise AssertionError(f"unexpected mocked command: {command}")

            startup_error = RuntimeErrorWithContext(
                "Server exited during startup: truncated tail"
            )
            with (
                patch("bench.runtime._existing_container", return_value=None),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime._run", side_effect=run),
                patch(
                    "bench.runtime.wait_for_endpoint", side_effect=startup_error
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeErrorWithContext, "truncated tail"
                ):
                    start_vllm(
                        model,
                        workspace=root,
                        server_log_path=server_log,
                    )

            persisted = server_log.read_text()

        operations = [command[1] for command in commands]
        self.assertEqual(operations, ["run", "logs", "inspect", "stop", "rm"])
        self.assertNotIn("--tail", commands[1])
        self.assertTrue(persisted.startswith("prior attempt without newline\n"))
        self.assertIn(
            "--- SparkBench docker logs (container-id) ---\n", persisted
        )
        self.assertIn("startup-line-000\n", persisted)
        self.assertIn("startup-line-149\n", persisted)

    def test_start_server_forwards_run_log_path_only_to_vllm(self) -> None:
        model = SimpleNamespace(backend="vllm")
        expected = object()
        workspace = Path("/mock/workspace")
        server_log = Path("/mock/run/server/server.log")

        with patch("bench.runtime.start_vllm", return_value=expected) as start:
            actual = start_server(
                model,
                workspace=workspace,
                allow_download=True,
                server_log_path=server_log,
            )

        self.assertIs(actual, expected)
        start.assert_called_once_with(
            model,
            workspace=workspace,
            allow_download=True,
            server_log_path=server_log,
        )


if __name__ == "__main__":
    unittest.main()
