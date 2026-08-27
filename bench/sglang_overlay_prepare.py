"""Prepare pinned SGLang source overlays for Qwen3.8-Flash-Next.

The generated files are intentionally ignored runtime inputs.  This module
extracts their exact bases from an already-cached digest-pinned image, applies
the pinned public patchers, and admits only the expected final byte digests.
"""

from __future__ import annotations

import argparse
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

from bench.qwen38_ple_cache import (
    PLECacheError,
    default_cache_path,
    materialize_ple_cache,
    validate_ple_cache,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_IMAGE = (
    "lmsysorg/sglang@sha256:"
    "14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4"
)
OUTPUT_RELATIVE = Path(
    "results/runtime-overlays/qwen38-flash-next-bf2b7c75-persistent-ple-v1"
)
PLE_ABLATION_OUTPUT_RELATIVE = Path(
    "results/runtime-overlays/"
    "qwen38-flash-next-bf2b7c75-persistent-ple-ablation-v1"
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
    post_patcher_name: str | None = None
    post_patcher_sha256: str | None = None
    final_patcher_name: str | None = None
    final_patcher_sha256: str | None = None


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
            "0b513b4dc4f2394f6b1733bb0b74fa40"
            "ab59f4a04f6b33601350b2a606c67804"
        ),
        post_patcher_name="qwen38-persistent-ple-cache.py",
        post_patcher_sha256=(
            "bf47f244406e149a3c7fe51d42d326d6"
            "3a008733d55868b51a73112052e3bcdf"
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

PLE_ABLATION_OVERLAYS = (
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
            "bcdc2c86aa59784ffe27d53c8d214e56"
            "b6aa45c02b1d5841fd956d1f006d6030"
        ),
        post_patcher_name="qwen38-persistent-ple-cache.py",
        post_patcher_sha256=(
            "bf47f244406e149a3c7fe51d42d326d6"
            "3a008733d55868b51a73112052e3bcdf"
        ),
        final_patcher_name="qwen38-ple-omission-ablation.py",
        final_patcher_sha256=(
            "cf4a28f2ca7cfc87acdb01602993367e"
            "b214a19caa66b8c9ca3bfd2a4e227fdd"
        ),
    ),
    MODULE_OVERLAYS[1],
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
        patchers = ((spec.patcher_name, spec.patcher_sha256),)
        if spec.post_patcher_name is not None:
            if spec.post_patcher_sha256 is None:
                raise OverlayPreparationError(
                    "post-patcher name is missing its pinned digest"
                )
            patchers += (
                (spec.post_patcher_name, spec.post_patcher_sha256),
            )
        if spec.final_patcher_name is not None:
            if spec.final_patcher_sha256 is None:
                raise OverlayPreparationError(
                    "final patcher name is missing its pinned digest"
                )
            patchers += (
                (spec.final_patcher_name, spec.final_patcher_sha256),
            )
        for patcher_name, expected_digest in patchers:
            patcher = PATCHER_ROOT / patcher_name
            if patcher.is_symlink() or not patcher.is_file():
                raise OverlayPreparationError(
                    f"missing regular pinned patcher: {patcher_name}"
                )
            actual = _sha256(patcher)
            if actual != expected_digest:
                raise OverlayPreparationError(
                    f"pinned patcher digest mismatch for {patcher_name}: "
                    f"expected {expected_digest}, got {actual}"
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
    patcher_names = [spec.patcher_name]
    if spec.post_patcher_name is not None:
        patcher_names.append(spec.post_patcher_name)
    if spec.final_patcher_name is not None:
        patcher_names.append(spec.final_patcher_name)
    for patcher_name in patcher_names:
        patcher = PATCHER_ROOT / patcher_name
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
                f"pinned patcher timed out: {patcher_name}"
            ) from error
        if result.returncode != 0:
            raise OverlayPreparationError(
                f"pinned patcher rejected the image source: {patcher_name}"
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
            or "SGLANG_QWEN4_PLE_CACHE_MODE" not in text
            or "_validate_readonly_ple_cache" not in text
            or "shared=not readonly" not in text
            or "if _ple_cache_is_readonly():" not in text
            or 'tensor.get("shard_count") != 128' not in text
            or r're.search(r"\.ngram_embedding\.shard_(\d+)\.weight$", name)'
            not in text
            or "ple_cache_seen_shards != expected_ple_shards" not in text
            or "loaded_weight.dtype != torch.float8_e4m3fn" not in text
            or "tuple(loaded_weight.shape) != (2500012, 160)" not in text
            or 'f"{prefix}.ngram_embedding.weight_scale"' not in text
        ):
            raise OverlayPreparationError(
                "PLE overlay failed structural verification"
            )
        ablation_markers = (
            "sparkbench_omit_ple",
            "ple_omitted_checkpoint_weights",
            "expected_ple_weights",
            "omitted PLE checkpoint tensor set mismatch",
        )
        if spec.final_patcher_name is not None and any(
            marker not in text for marker in ablation_markers
        ):
            raise OverlayPreparationError(
                "PLE omission ablation overlay failed structural verification"
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


def _prepare_overlay_variant(
    *, workspace: Path, specs: Sequence[OverlaySpec], output_relative: Path
) -> Path:
    _verify_cached_image()
    _verify_patchers(specs)
    output_parent = _checked_output_parent(workspace)
    target = output_parent / output_relative.name
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


def prepare_overlays(workspace: Path = REPOSITORY_ROOT) -> Path:
    """Reproduce the two mapped-PLE overlays, or verify an exact existing pair."""

    return _prepare_overlay_variant(
        workspace=workspace,
        specs=MODULE_OVERLAYS,
        output_relative=OUTPUT_RELATIVE,
    )


def prepare_ple_ablation_overlays(workspace: Path = REPOSITORY_ROOT) -> Path:
    """Prepare overlays that support explicit mapped and omitted PLE arms."""

    return _prepare_overlay_variant(
        workspace=workspace,
        specs=PLE_ABLATION_OVERLAYS,
        output_relative=PLE_ABLATION_OUTPUT_RELATIVE,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare pinned SGLang overlays and persistent PLE cache"
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--materialize-ple",
        action="store_true",
        help=(
            "offline-build or verify/adopt the exact 47.7 GiB PLE cache; "
            "does not construct the model"
        ),
    )
    actions.add_argument(
        "--verify-ple-cache",
        action="store_true",
        help="fully hash and verify an existing completed PLE cache",
    )
    actions.add_argument(
        "--prepare-ple-ablation",
        action="store_true",
        help=(
            "prepare the pinned overlay pair supporting matched mapped/omitted "
            "PLE experiments"
        ),
    )
    args = parser.parse_args(argv)
    if args.materialize_ple:
        try:
            record = materialize_ple_cache(progress=print)
        except PLECacheError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(
            "ready: persistent PLE cache "
            f"sha256:{record.payload_sha256}"
        )
        return 0
    if args.verify_ple_cache:
        try:
            record = validate_ple_cache(
                default_cache_path(), verify_payload=True
            )
        except PLECacheError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(
            "verified: persistent PLE cache "
            f"sha256:{record.payload_sha256}"
        )
        return 0
    if args.prepare_ple_ablation:
        try:
            target = prepare_ple_ablation_overlays()
        except OverlayPreparationError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(f"prepared: {target.relative_to(REPOSITORY_ROOT)}")
        return 0
    try:
        target = prepare_overlays()
    except OverlayPreparationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"prepared: {target.relative_to(REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
