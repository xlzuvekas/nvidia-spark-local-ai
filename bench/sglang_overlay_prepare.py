"""Prepare pinned SGLang source overlays for Qwen3.8-Flash-Next.

The generated files are intentionally ignored runtime inputs.  This module
extracts their exact bases from an already-cached digest-pinned image, applies
the pinned public patchers, and admits only the expected final byte digests.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_IMAGE = (
    "lmsysorg/sglang@sha256:"
    "14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4"
)
OUTPUT_RELATIVE = Path(
    "results/runtime-overlays/qwen38-flash-next-bf2b7c75"
)
PATCHER_ROOT = REPOSITORY_ROOT / "patches" / "sglang"
MODULE_PATH_MARKER = "SPARKBENCH_SGLANG_MODULE_PATHS="
CONTAINER_ID = re.compile(r"[0-9a-f]{12,64}")


class OverlayPreparationError(RuntimeError):
    """Raised when a pinned overlay cannot be reproduced safely."""


@dataclass(frozen=True)
class OverlaySpec:
    module: str
    container_path: str
    output_name: str
    patcher_name: str
    patcher_sha256: str
    output_sha256: str


MODULE_OVERLAYS = (
    OverlaySpec(
        module="sglang.srt.models.qwen4_exp",
        container_path=(
            "/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py"
        ),
        output_name="qwen4_exp.py",
        patcher_name="bf2b7c75-ple_mmap.py",
        patcher_sha256=(
            "eeabdde061631c9b606d4ccc7371ff8f"
            "b01c6cc034dfe6bad1e4f29a8aa21555"
        ),
        output_sha256=(
            "c687bf96b8adb980eaf3a1db2ad4a7c"
            "00b558537865d91674c0e1b43f4ae1d71"
        ),
    ),
    OverlaySpec(
        module=(
            "sglang.srt.layers.attention.qwen_sparse_attn_backend"
        ),
        container_path=(
            "/sgl-workspace/sglang/python/sglang/srt/layers/attention/"
            "qwen_sparse_attn_backend.py"
        ),
        output_name="qwen_sparse_attn_backend.py",
        patcher_name="bf2b7c75-qsa_trtllm_sm120.py",
        patcher_sha256=(
            "f60ccb9f9e350a43155a1a7a20d154b"
            "e0b7e93c29dacb3db95d397ba910090b2"
        ),
        output_sha256=(
            "e30566492e1502f94a4c7fed42d90b5"
            "23bbb662580c628459e6e63c7b5263c75"
        ),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _docker(
    arguments: Sequence[str], *, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["docker", *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as error:
        raise OverlayPreparationError("Docker is not installed") from error
    except subprocess.TimeoutExpired as error:
        raise OverlayPreparationError(
            "Docker timed out while preparing SGLang overlays"
        ) from error


def _require_success(
    result: subprocess.CompletedProcess[str], purpose: str
) -> str:
    if result.returncode != 0:
        raise OverlayPreparationError(
            f"Docker failed to {purpose} (exit {result.returncode})"
        )
    return result.stdout.strip()


def _verify_cached_image() -> None:
    output = _require_success(
        _docker(
            [
                "image",
                "inspect",
                "--format={{json .RepoDigests}}",
                PINNED_IMAGE,
            ],
            timeout=60,
        ),
        "inspect the exact cached image; refusing to pull it",
    )
    try:
        repo_digests = json.loads(output)
    except json.JSONDecodeError as error:
        raise OverlayPreparationError(
            "Docker returned invalid RepoDigests for the cached image"
        ) from error
    accepted = {PINNED_IMAGE, f"docker.io/{PINNED_IMAGE}"}
    if not isinstance(repo_digests, list) or not accepted.intersection(
        repo_digests
    ):
        raise OverlayPreparationError(
            "the cached image does not expose the required immutable RepoDigest"
        )


def _verify_patchers(specs: Sequence[OverlaySpec]) -> None:
    for spec in specs:
        patcher = PATCHER_ROOT / spec.patcher_name
        if patcher.is_symlink() or not patcher.is_file():
            raise OverlayPreparationError(
                f"missing regular vendored patcher: {spec.patcher_name}"
            )
        actual = _sha256(patcher)
        if actual != spec.patcher_sha256:
            raise OverlayPreparationError(
                f"vendored patcher digest mismatch for {spec.patcher_name}: "
                f"expected {spec.patcher_sha256}, got {actual}"
            )


def _checked_output_parent(workspace: Path) -> Path:
    try:
        root = workspace.resolve(strict=True)
    except FileNotFoundError as error:
        raise OverlayPreparationError("workspace does not exist") from error
    if not root.is_dir():
        raise OverlayPreparationError("workspace must be a directory")

    cursor = root
    for component in ("results", "runtime-overlays"):
        cursor = cursor / component
        if cursor.is_symlink():
            raise OverlayPreparationError(
                "runtime overlay output must not traverse a symbolic link"
            )
        if cursor.exists() and not cursor.is_dir():
            raise OverlayPreparationError(
                "runtime overlay output parent must be a directory"
            )
        cursor.mkdir(mode=0o700, exist_ok=True)
    try:
        cursor.resolve(strict=True).relative_to(root)
    except ValueError as error:
        raise OverlayPreparationError(
            "runtime overlay output escapes the workspace"
        ) from error
    return cursor


def _existing_output_is_complete(
    target: Path, specs: Sequence[OverlaySpec]
) -> bool:
    if target.is_symlink():
        raise OverlayPreparationError(
            "runtime overlay target must not be a symbolic link"
        )
    if not target.exists():
        return False
    if not target.is_dir():
        raise OverlayPreparationError(
            "runtime overlay target exists but is not a directory"
        )

    expected_names = {spec.output_name for spec in specs}
    entries = tuple(target.iterdir())
    if {entry.name for entry in entries} != expected_names:
        raise OverlayPreparationError(
            "runtime overlay target is partial or contains unexpected files; "
            "refusing to overwrite it"
        )
    expected_by_name = {spec.output_name: spec for spec in specs}
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise OverlayPreparationError(
                "runtime overlay target contains a non-regular file; "
                "refusing to overwrite it"
            )
        actual = _sha256(entry)
        expected = expected_by_name[entry.name].output_sha256
        if actual != expected:
            raise OverlayPreparationError(
                f"runtime overlay digest mismatch for {entry.name}; "
                "refusing to overwrite it"
            )
    return True


def _discovery_program(specs: Sequence[OverlaySpec]) -> str:
    module_names = tuple(spec.module for spec in specs)
    return (
        "import importlib, json\n"
        f"names = {module_names!r}\n"
        "paths = {}\n"
        "for name in names:\n"
        "    module = importlib.import_module(name)\n"
        "    paths[name] = module.__file__\n"
        f"print({MODULE_PATH_MARKER!r} + json.dumps(paths, sort_keys=True))\n"
    )


def _discover_module_paths(
    specs: Sequence[OverlaySpec],
) -> dict[str, str]:
    output = _require_success(
        _docker(
            [
                "run",
                "--rm",
                "--pull=never",
                "--network=none",
                "--env",
                "NVIDIA_VISIBLE_DEVICES=void",
                "--env",
                "CUDA_VISIBLE_DEVICES=",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--entrypoint",
                "python3",
                PINNED_IMAGE,
                "-c",
                _discovery_program(specs),
            ]
        ),
        "import the pinned SGLang modules",
    )
    marked = [
        line[len(MODULE_PATH_MARKER) :]
        for line in output.splitlines()
        if line.startswith(MODULE_PATH_MARKER)
    ]
    if len(marked) != 1:
        raise OverlayPreparationError(
            "module discovery did not return exactly one marked result"
        )
    try:
        paths = json.loads(marked[0])
    except json.JSONDecodeError as error:
        raise OverlayPreparationError(
            "module discovery returned invalid JSON"
        ) from error
    expected_modules = {spec.module for spec in specs}
    if not isinstance(paths, dict) or set(paths) != expected_modules:
        raise OverlayPreparationError(
            "module discovery returned an unexpected module set"
        )
    for spec in specs:
        discovered = paths[spec.module]
        if not isinstance(discovered, str):
            raise OverlayPreparationError(
                f"module discovery returned no file for {spec.module}"
            )
        parsed = PurePosixPath(discovered)
        if (
            not parsed.is_absolute()
            or ".." in parsed.parts
            or discovered != spec.container_path
        ):
            raise OverlayPreparationError(
                f"module path mismatch for {spec.module}: {discovered!r}"
            )
    return {str(key): str(value) for key, value in paths.items()}


def _extract_sources(
    destination: Path,
    paths: dict[str, str],
    specs: Sequence[OverlaySpec],
) -> None:
    create = _docker(
        [
            "create",
            "--pull=never",
            "--network=none",
            "--label",
            "ai.sparkbench.purpose=sglang-overlay-preparation",
            "--entrypoint",
            "/bin/true",
            PINNED_IMAGE,
        ],
        timeout=60,
    )
    container_id = _require_success(
        create, "create an inert extraction container"
    )
    if CONTAINER_ID.fullmatch(container_id) is None:
        raise OverlayPreparationError(
            "Docker returned an invalid extraction-container ID"
        )

    extraction_error: Exception | None = None
    try:
        for spec in specs:
            output = destination / spec.output_name
            _require_success(
                _docker(
                    [
                        "cp",
                        f"{container_id}:{paths[spec.module]}",
                        str(output),
                    ],
                    timeout=60,
                ),
                f"extract {spec.module}",
            )
            if output.is_symlink() or not output.is_file():
                raise OverlayPreparationError(
                    f"Docker did not extract a regular {spec.output_name}"
                )
    except Exception as error:  # cleanup is still limited to our inert container
        extraction_error = error
    cleanup = _docker(
        ["container", "rm", container_id],
        timeout=60,
    )
    if extraction_error is not None:
        if cleanup.returncode != 0:
            raise OverlayPreparationError(
                "source extraction failed and its inert container could not "
                "be removed"
            ) from extraction_error
        raise extraction_error
    _require_success(cleanup, "remove the inert extraction container")


def _apply_patcher(source: Path, spec: OverlaySpec) -> None:
    patcher = PATCHER_ROOT / spec.patcher_name
    try:
        result = subprocess.run(
            [sys.executable, str(patcher), str(source)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        raise OverlayPreparationError(
            f"vendored patcher timed out: {spec.patcher_name}"
        ) from error
    if result.returncode != 0:
        raise OverlayPreparationError(
            f"vendored patcher rejected the image source: {spec.patcher_name}"
        )


def _verify_ast(source: Path, spec: OverlaySpec) -> None:
    try:
        text = source.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=spec.output_name)
    except (OSError, UnicodeError, SyntaxError) as error:
        raise OverlayPreparationError(
            f"patched overlay failed AST parsing: {spec.output_name}"
        ) from error

    if spec.output_name == "qwen4_exp.py":
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_alloc_ple_table"
        ]
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "Qwen4ExpPinnedHostEmbedding"
        ]
        if len(helpers) != 1 or len(classes) != 1:
            raise OverlayPreparationError(
                "PLE overlay is missing its unique helper or embedding class"
            )
        class_source = ast.get_source_segment(text, classes[0]) or ""
        if (
            helpers[0].lineno >= classes[0].lineno
            or "_alloc_ple_table(source_weight.shape" not in class_source
            or "pin_memory" in class_source
            or "SGLANG_QWEN4_PLE_MMAP_DIR" not in text
            or "torch.from_file(path, shared=True" not in text
        ):
            raise OverlayPreparationError(
                "PLE overlay failed structural verification"
            )
    elif spec.output_name == "qwen_sparse_attn_backend.py":
        if (
            "from sglang.srt.utils import "
            "is_sm100_supported, is_sm120_supported" not in text
            or "if not (is_sm100_supported() or is_sm120_supported()):"
            not in text
        ):
            raise OverlayPreparationError(
                "QSA overlay failed structural verification"
            )
    else:
        raise OverlayPreparationError(
            f"no AST policy exists for {spec.output_name}"
        )


def _verify_patched_sources(
    directory: Path, specs: Sequence[OverlaySpec]
) -> None:
    for spec in specs:
        source = directory / spec.output_name
        _verify_ast(source, spec)
        actual = _sha256(source)
        if actual != spec.output_sha256:
            raise OverlayPreparationError(
                f"patched overlay digest mismatch for {spec.output_name}: "
                f"expected {spec.output_sha256}, got {actual}"
            )


def _write_outputs(
    source_directory: Path,
    target: Path,
    specs: Sequence[OverlaySpec],
) -> None:
    try:
        target.mkdir(mode=0o700)
    except FileExistsError as error:
        raise OverlayPreparationError(
            "runtime overlay target appeared during preparation; "
            "refusing to overwrite it"
        ) from error
    for spec in specs:
        source = source_directory / spec.output_name
        output = target / spec.output_name
        try:
            with output.open("xb") as stream:
                stream.write(source.read_bytes())
            output.chmod(0o600)
        except FileExistsError as error:
            raise OverlayPreparationError(
                f"runtime overlay appeared during preparation: "
                f"{spec.output_name}"
            ) from error
    _verify_patched_sources(target, specs)


def prepare_overlays(workspace: Path = REPOSITORY_ROOT) -> Path:
    """Reproduce the two ignored overlays, or verify an exact existing pair."""

    specs = MODULE_OVERLAYS
    _verify_cached_image()
    _verify_patchers(specs)
    output_parent = _checked_output_parent(workspace)
    target = output_parent / OUTPUT_RELATIVE.name
    if _existing_output_is_complete(target, specs):
        return target

    with tempfile.TemporaryDirectory(
        prefix="sparkbench-sglang-overlays-"
    ) as directory:
        staging = Path(directory)
        paths = _discover_module_paths(specs)
        _extract_sources(staging, paths, specs)
        for spec in specs:
            _apply_patcher(staging / spec.output_name, spec)
        _verify_patched_sources(staging, specs)
        _write_outputs(staging, target, specs)
    return target


def main() -> int:
    try:
        target = prepare_overlays()
    except OverlayPreparationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"prepared: {target.relative_to(REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
