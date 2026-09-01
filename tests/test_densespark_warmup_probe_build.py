from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from bench import densespark_warmup_probe_build as build


ROOT = Path(__file__).resolve().parents[1]


class DenseSparkWarmupProbeBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repository_root = Path(self.temporary_directory.name) / "repository"
        for relative in (
            build.DOCKERFILE_RELATIVE_PATH,
            build.DOCKERIGNORE_RELATIVE_PATH,
            build.PROBE_RELATIVE_PATH,
        ):
            source = ROOT / Path(*relative.parts)
            destination = self.repository_root / Path(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    @staticmethod
    def _result(
        command: Sequence[str],
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    def test_fixed_build_has_no_pull_or_build_argument_escape_hatch(self) -> None:
        commands: list[tuple[str, ...]] = []

        def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
            frozen = tuple(command)
            commands.append(frozen)
            if frozen[:3] == ("docker", "image", "inspect"):
                image = frozen[-1]
                image_id = (
                    build.EXPECTED_BASE_IMAGE_ID
                    if image == build.BASE_IMAGE
                    else build.EXPECTED_DERIVED_IMAGE_ID
                )
                return self._result(command, stdout=f"{image_id}\n")
            return self._result(command)

        receipt = build.build_densespark_warmup_probe(
            repository_root=self.repository_root,
            runner=runner,
        )

        expected_dockerfile = (
            self.repository_root
            / Path(*build.DOCKERFILE_RELATIVE_PATH.parts)
        ).resolve()
        expected_root = self.repository_root.resolve()
        self.assertEqual(
            commands,
            [
                (
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    build.BASE_IMAGE,
                ),
                (
                    "docker",
                    "build",
                    "--pull=false",
                    "--network=none",
                    "--file",
                    str(expected_dockerfile),
                    "--tag",
                    build.DERIVED_IMAGE,
                    str(expected_root),
                ),
                (
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    build.BASE_IMAGE,
                ),
                (
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    build.DERIVED_IMAGE,
                ),
            ],
        )
        build_command = commands[1]
        self.assertNotIn("--pull", build_command)
        self.assertNotIn("--build-arg", build_command)
        self.assertNotIn("--build-context", build_command)
        self.assertEqual(receipt.base_image_id, build.EXPECTED_BASE_IMAGE_ID)
        self.assertEqual(receipt.derived_image_id, build.EXPECTED_DERIVED_IMAGE_ID)
        self.assertEqual(
            receipt.dockerfile_sha256,
            build.EXPECTED_DOCKERFILE_SHA256,
        )
        self.assertEqual(
            receipt.dockerignore_sha256,
            build.EXPECTED_DOCKERIGNORE_SHA256,
        )
        self.assertEqual(receipt.probe_sha256, build.EXPECTED_PROBE_SHA256)

    def test_base_id_drift_stops_before_build(self) -> None:
        commands: list[tuple[str, ...]] = []

        def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
            commands.append(tuple(command))
            return self._result(command, stdout=f"sha256:{'0' * 64}\n")

        with self.assertRaisesRegex(
            build.DenseSparkWarmupProbeBuildError,
            "image ID drifted",
        ):
            build.build_densespark_warmup_probe(
                repository_root=self.repository_root,
                runner=runner,
            )
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][-1], build.BASE_IMAGE)

    def test_derived_id_drift_is_rejected_without_repinning(self) -> None:
        inspect_count = 0

        def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
            nonlocal inspect_count
            if tuple(command)[:3] == ("docker", "image", "inspect"):
                inspect_count += 1
                image_id = (
                    f"sha256:{'1' * 64}"
                    if inspect_count == 3
                    else build.EXPECTED_BASE_IMAGE_ID
                )
                return self._result(command, stdout=f"{image_id}\n")
            return self._result(command)

        with self.assertRaisesRegex(
            build.DenseSparkWarmupProbeBuildError,
            "image ID drifted",
        ):
            build.build_densespark_warmup_probe(
                repository_root=self.repository_root,
                runner=runner,
            )
        self.assertEqual(build.EXPECTED_DERIVED_IMAGE_ID, (
            "sha256:c7adf2163f7dd04b52eb5ec91f373bf8"
            "fcd1cc63a51f61c2d457ad2976564153"
        ))

    def test_base_retag_during_build_is_rejected_before_derived_inspect(self) -> None:
        commands: list[tuple[str, ...]] = []
        base_inspects = 0

        def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
            nonlocal base_inspects
            frozen = tuple(command)
            commands.append(frozen)
            if frozen[:3] == ("docker", "image", "inspect"):
                self.assertEqual(frozen[-1], build.BASE_IMAGE)
                base_inspects += 1
                image_id = (
                    build.EXPECTED_BASE_IMAGE_ID
                    if base_inspects == 1
                    else f"sha256:{'2' * 64}"
                )
                return self._result(command, stdout=f"{image_id}\n")
            return self._result(command)

        with self.assertRaisesRegex(
            build.DenseSparkWarmupProbeBuildError,
            "image ID drifted",
        ):
            build.build_densespark_warmup_probe(
                repository_root=self.repository_root,
                runner=runner,
            )
        self.assertEqual(len(commands), 3)
        self.assertEqual(commands[-1][-1], build.BASE_IMAGE)

    def test_source_drift_and_symlinks_stop_before_docker(self) -> None:
        probe = self.repository_root / Path(*build.PROBE_RELATIVE_PATH.parts)
        probe.write_bytes(b"drifted probe\n")
        calls = 0

        def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
            nonlocal calls
            calls += 1
            return self._result(command)

        with self.assertRaisesRegex(
            build.DenseSparkWarmupProbeBuildError,
            "probe_sha256 does not match",
        ):
            build.build_densespark_warmup_probe(
                repository_root=self.repository_root,
                runner=runner,
            )
        self.assertEqual(calls, 0)

        probe.unlink()
        target = self.repository_root / "synthetic-target.py"
        target.write_bytes((ROOT / Path(*build.PROBE_RELATIVE_PATH.parts)).read_bytes())
        probe.symlink_to(target)
        with self.assertRaisesRegex(
            build.DenseSparkWarmupProbeBuildError,
            "must not be a symlink",
        ):
            build.validate_checked_in_sources(repository_root=self.repository_root)

    def test_source_change_during_build_fails_closed(self) -> None:
        dockerignore = self.repository_root / Path(
            *build.DOCKERIGNORE_RELATIVE_PATH.parts
        )

        def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
            frozen = tuple(command)
            if frozen[:3] == ("docker", "image", "inspect"):
                return self._result(
                    command,
                    stdout=f"{build.EXPECTED_BASE_IMAGE_ID}\n",
                )
            dockerignore.write_text("**\n", encoding="utf-8")
            return self._result(command)

        with self.assertRaisesRegex(
            build.DenseSparkWarmupProbeBuildError,
            "dockerignore_sha256 does not match",
        ):
            build.build_densespark_warmup_probe(
                repository_root=self.repository_root,
                runner=runner,
            )

    def test_docker_failures_and_malformed_inspection_are_sanitized(self) -> None:
        results = (
            self._result((), returncode=1, stderr="synthetic-secret"),
            self._result((), stdout="not-an-image-id\n"),
            self._result(
                (),
                stdout=(
                    f"{build.EXPECTED_BASE_IMAGE_ID}\n"
                    f"{build.EXPECTED_BASE_IMAGE_ID}\n"
                ),
            ),
        )
        for result in results:
            with self.subTest(result=result):
                with self.assertRaises(build.DenseSparkWarmupProbeBuildError) as raised:
                    build.build_densespark_warmup_probe(
                        repository_root=self.repository_root,
                        runner=lambda command, result=result: result,
                    )
                self.assertNotIn("synthetic-secret", str(raised.exception))

        def missing_docker(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
            raise FileNotFoundError("synthetic-secret")

        with self.assertRaises(build.DenseSparkWarmupProbeBuildError) as raised:
            build.build_densespark_warmup_probe(
                repository_root=self.repository_root,
                runner=missing_docker,
            )
        self.assertNotIn("synthetic-secret", str(raised.exception))

    def test_unexpected_cli_arguments_are_rejected_before_any_build(self) -> None:
        for arguments in (
            ("--build-arg", "BASE_IMAGE=remote/image:latest"),
            ("--pull",),
            ("--tag", "untrusted"),
            ("unexpected",),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(
                    build.DenseSparkWarmupProbeBuildError,
                    "accepts no arguments",
                ):
                    build.main(arguments)

    def test_dockerignore_is_an_exact_single_payload_allowlist(self) -> None:
        path = ROOT / Path(*build.DOCKERIGNORE_RELATIVE_PATH.parts)
        self.assertEqual(
            path.read_text(encoding="utf-8").splitlines(),
            [
                "**",
                "!bench/",
                "!bench/assets/",
                "!bench/assets/densespark_qwen_warmup_probe.py",
            ],
        )
        observed = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
        self.assertEqual(observed, build.EXPECTED_DOCKERIGNORE_SHA256)

        fixture = self.repository_root / Path(*build.DOCKERIGNORE_RELATIVE_PATH.parts)
        fixture.write_text(
            fixture.read_text(encoding="utf-8") + "!unexpected/**\n",
            encoding="utf-8",
        )
        calls = 0

        def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
            nonlocal calls
            calls += 1
            return self._result(command)

        with self.assertRaisesRegex(
            build.DenseSparkWarmupProbeBuildError,
            "dockerignore_sha256 does not match",
        ):
            build.build_densespark_warmup_probe(
                repository_root=self.repository_root,
                runner=runner,
            )
        self.assertEqual(calls, 0)

    def test_dockerfile_contract_is_literal_and_digest_pinned(self) -> None:
        path = ROOT / Path(*build.DOCKERFILE_RELATIVE_PATH.parts)
        payload = path.read_bytes()
        text = payload.decode("utf-8")
        instructions = tuple(
            line.strip() for line in text.splitlines() if line.strip()
        )
        self.assertEqual(instructions[0], f"FROM {build.BASE_IMAGE}")
        self.assertFalse(any(line.startswith("ARG ") for line in instructions))
        self.assertNotIn("${", text)
        self.assertIn('ENTRYPOINT ["vllm"]', text)
        for digest in (
            build.EXPECTED_BASE_IMAGE_ID,
            build.EXPECTED_PROBE_SHA256,
            "sha256:2b08d94662e7b04ce61c0f7a818e0cd1768fe7602a89df04ec6148f62fe3acdb",
            "sha256:452ae5db905110df8eb7aac90a93ac80863d166f8ea7d52b8cec02c477477aed",
            "sha256:d42cdc95d8d221b49693a46119c714fee3f290282bdfefa63f92f9725f1b20ea",
            "sha256:53eaae681b5a0327465b28b7b1983303335db852ac9667ae05faa3682d8c6b8c",
            "sha256:000ab8996af9788fdb8843a6a3b91833e7a14c8acc0e1ea073a536330f64cb6f",
            "sha256:6f6395c128e80861f7f7d21b8e1e4547261ab9e928390aa7a7a89ce0d701ff36",
        ):
            self.assertIn(digest.removeprefix("sha256:"), text)
        observed = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        self.assertEqual(observed, build.EXPECTED_DOCKERFILE_SHA256)
        self.assertEqual(
            build.validate_checked_in_sources(repository_root=ROOT),
            {
                "dockerfile_sha256": build.EXPECTED_DOCKERFILE_SHA256,
                "dockerignore_sha256": build.EXPECTED_DOCKERIGNORE_SHA256,
                "probe_sha256": build.EXPECTED_PROBE_SHA256,
            },
        )


if __name__ == "__main__":
    unittest.main()
