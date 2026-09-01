"""Fail-closed local builder for the DenseSpark Qwen warmup probe image.

This helper intentionally exposes no build options.  The base tag, source
files, Dockerfile, output tag, and both image IDs are one immutable contract.
It never asks Docker to pull and never forwards build arguments.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import subprocess
import sys
from typing import Protocol


BASE_IMAGE = "local/densespark:qwen38-27b-v1.2-0abecc3"
EXPECTED_BASE_IMAGE_ID = (
    "sha256:d8d02859a49ebf452d9e20b5fbc0790c"
    "d4c38fe9a1f5184096b06e3cc6a751d1"
)
DERIVED_IMAGE = (
    "local/densespark:qwen38-27b-v1.2-warmup-probe-hardened-572e66d5"
)
EXPECTED_DERIVED_IMAGE_ID = (
    "sha256:c7adf2163f7dd04b52eb5ec91f373bf8"
    "fcd1cc63a51f61c2d457ad2976564153"
)
DOCKERFILE_RELATIVE_PATH = PurePosixPath(
    "patches/vllm/Dockerfile.densespark-qwen-warmup-probe"
)
EXPECTED_DOCKERFILE_SHA256 = (
    "sha256:572e66d585ed74a5f0b278e2feb2cf7d"
    "ba260ca84c53ba93be66d7c2e69c571a"
)
DOCKERIGNORE_RELATIVE_PATH = PurePosixPath(
    "patches/vllm/Dockerfile.densespark-qwen-warmup-probe.dockerignore"
)
EXPECTED_DOCKERIGNORE_SHA256 = (
    "sha256:100ee126af6ef26dd45e85b9e90f5cc0"
    "adb8d6b0c51d391c37117fc7168627ea"
)
PROBE_RELATIVE_PATH = PurePosixPath(
    "bench/assets/densespark_qwen_warmup_probe.py"
)
EXPECTED_PROBE_SHA256 = (
    "sha256:95089265e60f67da8d8f33d6fb249e4c"
    "79300f0891c38f2f15d4e125001821d3"
)

_SHA256_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SOURCE_PINS = (
    ("dockerfile_sha256", DOCKERFILE_RELATIVE_PATH, EXPECTED_DOCKERFILE_SHA256),
    (
        "dockerignore_sha256",
        DOCKERIGNORE_RELATIVE_PATH,
        EXPECTED_DOCKERIGNORE_SHA256,
    ),
    ("probe_sha256", PROBE_RELATIVE_PATH, EXPECTED_PROBE_SHA256),
)


class DenseSparkWarmupProbeBuildError(RuntimeError):
    """The immutable local build contract was not satisfied."""


class CommandResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str]], CommandResult]


@dataclass(frozen=True)
class DenseSparkWarmupProbeBuildReceipt:
    base_image: str
    base_image_id: str
    derived_image: str
    derived_image_id: str
    dockerfile_sha256: str
    dockerignore_sha256: str
    probe_sha256: str


def _subprocess_runner(command: Sequence[str]) -> CommandResult:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )


def _metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _repository_root(repository_root: Path | None) -> Path:
    supplied = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root)
    )
    try:
        if supplied.is_symlink():
            raise DenseSparkWarmupProbeBuildError(
                "DenseSpark warmup-probe repository root must not be a symlink"
            )
        resolved = supplied.resolve(strict=True)
    except DenseSparkWarmupProbeBuildError:
        raise
    except OSError as exc:
        raise DenseSparkWarmupProbeBuildError(
            "DenseSpark warmup-probe repository root is unavailable"
        ) from exc
    if not resolved.is_dir():
        raise DenseSparkWarmupProbeBuildError(
            "DenseSpark warmup-probe repository root must be a directory"
        )
    return resolved


def _checked_in_file(root: Path, relative: PurePosixPath) -> Path:
    candidate = root
    try:
        for component in relative.parts:
            candidate /= component
            if candidate.is_symlink():
                raise DenseSparkWarmupProbeBuildError(
                    "DenseSpark warmup-probe source must not be a symlink"
                )
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except DenseSparkWarmupProbeBuildError:
        raise
    except (OSError, ValueError) as exc:
        raise DenseSparkWarmupProbeBuildError(
            "DenseSpark warmup-probe source is unavailable or escapes the repository"
        ) from exc
    return resolved


def _regular_file_sha256(root: Path, relative: PurePosixPath) -> str:
    source = _checked_in_file(root, relative)
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise DenseSparkWarmupProbeBuildError(
            "DenseSpark warmup-probe source could not be opened safely"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DenseSparkWarmupProbeBuildError(
                "DenseSpark warmup-probe source must be a regular file"
            )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = source.stat(follow_symlinks=False)
    except OSError as exc:
        raise DenseSparkWarmupProbeBuildError(
            "DenseSpark warmup-probe source changed while being hashed"
        ) from exc
    if (
        not stat.S_ISREG(path_after.st_mode)
        or _metadata(before) != _metadata(after)
        or _metadata(after) != _metadata(path_after)
    ):
        raise DenseSparkWarmupProbeBuildError(
            "DenseSpark warmup-probe source changed while being hashed"
        )
    return f"sha256:{digest.hexdigest()}"


def validate_checked_in_sources(
    *, repository_root: Path | None = None
) -> dict[str, str]:
    """Validate the exact Dockerfile and probe payload used as build inputs."""

    root = _repository_root(repository_root)
    receipt: dict[str, str] = {}
    for name, relative, expected in _SOURCE_PINS:
        observed = _regular_file_sha256(root, relative)
        if not secrets.compare_digest(observed, expected):
            raise DenseSparkWarmupProbeBuildError(
                f"DenseSpark warmup-probe {name} does not match its pin"
            )
        receipt[name] = observed
    return receipt


def _run(command: Sequence[str], *, runner: CommandRunner, purpose: str) -> CommandResult:
    try:
        result = runner(tuple(command))
    except (OSError, subprocess.SubprocessError) as exc:
        raise DenseSparkWarmupProbeBuildError(
            f"Docker failed while {purpose}"
        ) from exc
    returncode = getattr(result, "returncode", None)
    if type(returncode) is not int or returncode != 0:
        raise DenseSparkWarmupProbeBuildError(
            f"Docker failed while {purpose}"
        )
    return result


def _inspect_exact_image(
    image: str,
    expected_image_id: str,
    *,
    runner: CommandRunner,
    purpose: str,
) -> str:
    if _SHA256_ID_RE.fullmatch(expected_image_id) is None:
        raise DenseSparkWarmupProbeBuildError("expected Docker image ID is invalid")
    command = ("docker", "image", "inspect", "--format", "{{.Id}}", image)
    result = _run(command, runner=runner, purpose=purpose)
    stdout = getattr(result, "stdout", None)
    if not isinstance(stdout, str):
        raise DenseSparkWarmupProbeBuildError(
            f"Docker returned invalid output while {purpose}"
        )
    lines = stdout.splitlines()
    if len(lines) != 1 or _SHA256_ID_RE.fullmatch(lines[0]) is None:
        raise DenseSparkWarmupProbeBuildError(
            f"Docker returned invalid output while {purpose}"
        )
    if not secrets.compare_digest(lines[0], expected_image_id):
        raise DenseSparkWarmupProbeBuildError(
            f"Docker image ID drifted while {purpose}"
        )
    return lines[0]


def build_densespark_warmup_probe(
    *,
    repository_root: Path | None = None,
    runner: CommandRunner = _subprocess_runner,
) -> DenseSparkWarmupProbeBuildReceipt:
    """Build and verify the one pinned local diagnostic image.

    There is deliberately no caller-provided image, tag, environment, Dockerfile,
    build argument, or extra command option.  Source and base identity are checked
    before the build and checked again afterward to detect mid-build drift.
    """

    root = _repository_root(repository_root)
    sources = validate_checked_in_sources(repository_root=root)
    base_id = _inspect_exact_image(
        BASE_IMAGE,
        EXPECTED_BASE_IMAGE_ID,
        runner=runner,
        purpose="validating the local DenseSpark base image",
    )
    dockerfile = root / Path(*DOCKERFILE_RELATIVE_PATH.parts)
    build_command = (
        "docker",
        "build",
        "--pull=false",
        "--network=none",
        "--file",
        str(dockerfile),
        "--tag",
        DERIVED_IMAGE,
        str(root),
    )
    _run(build_command, runner=runner, purpose="building the warmup-probe image")

    sources_after = validate_checked_in_sources(repository_root=root)
    if sources_after != sources:
        raise DenseSparkWarmupProbeBuildError(
            "DenseSpark warmup-probe sources changed during the build"
        )
    _inspect_exact_image(
        BASE_IMAGE,
        EXPECTED_BASE_IMAGE_ID,
        runner=runner,
        purpose="revalidating the local DenseSpark base image",
    )
    derived_id = _inspect_exact_image(
        DERIVED_IMAGE,
        EXPECTED_DERIVED_IMAGE_ID,
        runner=runner,
        purpose="validating the derived warmup-probe image",
    )
    return DenseSparkWarmupProbeBuildReceipt(
        base_image=BASE_IMAGE,
        base_image_id=base_id,
        derived_image=DERIVED_IMAGE,
        derived_image_id=derived_id,
        dockerfile_sha256=sources["dockerfile_sha256"],
        dockerignore_sha256=sources["dockerignore_sha256"],
        probe_sha256=sources["probe_sha256"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed build contract; all command-line arguments are rejected."""

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise DenseSparkWarmupProbeBuildError(
            "the DenseSpark warmup-probe builder accepts no arguments"
        )
    receipt = build_densespark_warmup_probe()
    print(json.dumps(asdict(receipt), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DenseSparkWarmupProbeBuildError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
