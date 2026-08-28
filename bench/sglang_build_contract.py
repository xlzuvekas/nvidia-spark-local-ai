"""Offline verification for a pinned SGLang source/build candidate.

This module validates source composition and an exact Docker build invocation.
It deliberately does not fetch, modify either inspected repository, build,
inspect images, or admit a runtime. Source transitions are replayed only in a
disposable isolated Git index and object directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


CONTRACT_KIND = "sparkbench-sglang-build-candidate"
CONTRACT_SCHEMA_VERSION = 1
CONTRACT_CANDIDATE_ID = "sglang-sm121-triton-storage-v1"
CONTRACT_STATUS = "source_and_build_invocation_only"
EXCLUDED_QSA_COMMIT = "8ef3b3fee34a3b5543b65393dd217ed0362a9273"
REQUIRED_PROTECTED_PATHS = frozenset(
    {
        "python/sglang/srt/layers/attention/qsa/sm121_varlen.py",
        "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py",
        "python/sglang/srt/utils/common.py",
        "test/registered/kernels/test_qsa.py",
    }
)
CONTEXT_MODE = "tracked-tree-export"
AUTOMATIC_BUILD_ARGUMENTS = frozenset({"TARGETARCH"})
INERT_DOCKERFILE_ARGUMENTS = frozenset({"MOONCAKE_COMPILE_ARG"})
EXPLICIT_EMPTY_BUILD_ARGUMENTS = frozenset(
    {"PIP_DEFAULT_INDEX", "SGLANG_BUILD_URL", "SGL_VERSION", "UBUNTU_MIRROR"}
)
EXPECTED_BUILD_ARGUMENTS = frozenset(
    {
        "BRANCH_TYPE",
        "BUILD_TYPE",
        "CUDA_BASE_IMAGE",
        "CUDA_VERSION",
        "FLASHINFER_VERSION",
        "GDRCOPY_VERSION",
        "GITHUB_ARTIFACTORY",
        "HPC_OPS_COMMIT",
        "INSTALL_FLASHINFER_JIT_CACHE",
        "MOONCAKE_VERSION",
        "MSCCLPP_VERSION",
        "PIP_DEFAULT_INDEX",
        "SGLANG_BUILD_COMMIT",
        "SGLANG_BUILD_URL",
        "SGLANG_IMAGE_TAG",
        "SGL_DEEP_GEMM_VERSION",
        "SGL_KERNEL_VERSION",
        "SGL_VERSION",
        "UBUNTU_MIRROR",
        "USE_LATEST_SGLANG",
    }
)
_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "kind", "candidate_id", "status", "source", "build"}
)
_SOURCE_KEYS = frozenset(
    {
        "upstream_repository",
        "storage_repository",
        "base_commit",
        "base_tree",
        "final_tree",
        "excluded_commits",
        "protected_files",
        "steps",
    }
)
_COMMIT_STEP_KEYS = frozenset(
    {"kind", "repository", "commit", "input_tree", "output_tree"}
)
_PATCH_STEP_KEYS = frozenset(
    {"kind", "path", "sha256", "patch_id", "input_tree", "output_tree"}
)
_PROTECTED_FILE_KEYS = frozenset({"path", "sha256"})
_BUILD_KEYS = frozenset(
    {
        "dockerfile",
        "dockerfile_sha256",
        "dockerignore",
        "dockerignore_sha256",
        "target",
        "platform",
        "automatic_targetarch",
        "context_mode",
        "external_base_argument",
        "external_base_reference",
        "external_base_index_digest",
        "external_base_manifest_digest",
        "external_base_config_digest",
        "external_base_stages",
        "stage_names",
        "args",
    }
)
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LOWER_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_ARGUMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_FROM_VARIABLE = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)"
)
_BRACED_VARIABLE_AT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[^}]*\}")
_SIMPLE_VARIABLE_AT = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_SHELL_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MAX_GIT_OUTPUT_BYTES = 4 * 1024 * 1024


class SGLangBuildContractError(RuntimeError):
    """Raised when the local source/build candidate is not exactly pinned."""


@dataclass(frozen=True, slots=True)
class SourceStep:
    kind: str
    input_tree: str
    output_tree: str
    repository: str | None = None
    commit: str | None = None
    path: str | None = None
    sha256: str | None = None
    patch_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProtectedFile:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SGLangBuildContract:
    candidate_id: str
    status: str
    base_commit: str
    base_tree: str
    final_tree: str
    excluded_commits: tuple[str, ...]
    protected_files: tuple[ProtectedFile, ...]
    source_steps: tuple[SourceStep, ...]
    dockerfile: str
    dockerfile_sha256: str
    dockerignore: str
    dockerignore_sha256: str
    target: str
    platform: str
    automatic_targetarch: str
    context_mode: str
    external_base_argument: str
    external_base_reference: str
    external_base_index_digest: str
    external_base_manifest_digest: str
    external_base_config_digest: str
    external_base_stages: tuple[str, ...]
    stage_names: tuple[str, ...]
    build_args: tuple[tuple[str, str], ...]

    def build_arg_map(self) -> dict[str, str]:
        """Return the immutable contract pairs as a fresh mapping."""

        return dict(self.build_args)


@dataclass(frozen=True, slots=True)
class SGLangBuildVerification:
    candidate_id: str
    status: str
    source_tree: str
    dockerfile_sha256: str
    target: str
    platform: str
    external_base_reference: str
    build_arg_count: int

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic, path-free verification summary."""

        return {
            "build_arg_count": self.build_arg_count,
            "candidate_id": self.candidate_id,
            "dockerfile_sha256": self.dockerfile_sha256,
            "external_base_reference": self.external_base_reference,
            "platform": self.platform,
            "source_tree": self.source_tree,
            "status": self.status,
            "target": self.target,
            "verified": True,
        }


@dataclass(frozen=True, slots=True)
class _FromInstruction:
    image: str
    stage: str
    line_number: int
    flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ArgInstruction:
    name: str
    default: str | None
    line_number: int
    global_scope: bool


@dataclass(frozen=True, slots=True)
class _GitBlob:
    mode: str
    data: bytes


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise SGLangBuildContractError(
            f"{label} keys are invalid: {'; '.join(details)}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SGLangBuildContractError(f"{label} must be a table")
    return value


def _string(
    value: Mapping[str, Any], key: str, label: str, *, allow_empty: bool = False
) -> str:
    item = value.get(key)
    if not isinstance(item, str) or (not allow_empty and not item):
        suffix = "a string" if allow_empty else "a non-empty string"
        raise SGLangBuildContractError(f"{label}.{key} must be {suffix}")
    if "\x00" in item or "\n" in item or "\r" in item:
        raise SGLangBuildContractError(
            f"{label}.{key} contains a forbidden control character"
        )
    return item


def _string_list(value: Mapping[str, Any], key: str, label: str) -> tuple[str, ...]:
    items = value.get(key)
    if not isinstance(items, list) or not items:
        raise SGLangBuildContractError(f"{label}.{key} must be a non-empty array")
    output = []
    for item in items:
        if not isinstance(item, str) or not item:
            raise SGLangBuildContractError(
                f"{label}.{key} must contain only non-empty strings"
            )
        output.append(item)
    if len(output) != len(set(output)):
        raise SGLangBuildContractError(f"{label}.{key} contains duplicates")
    return tuple(output)


def _require_hex40(value: str, label: str) -> None:
    if _HEX40.fullmatch(value) is None:
        raise SGLangBuildContractError(f"{label} must be 40 lowercase hex digits")


def _require_hex64(value: str, label: str) -> None:
    if _HEX64.fullmatch(value) is None:
        raise SGLangBuildContractError(f"{label} must be 64 lowercase hex digits")


def _require_digest(value: str, label: str) -> None:
    if _DIGEST.fullmatch(value) is None:
        raise SGLangBuildContractError(
            f"{label} must be a lowercase sha256 OCI digest"
        )


def _require_https_repository(value: str, label: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SGLangBuildContractError(
            f"{label} must be an HTTPS repository URL without credentials"
        )


def _safe_relative_path(value: str, label: str) -> PurePosixPath:
    if "\\" in value:
        raise SGLangBuildContractError(f"{label} must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise SGLangBuildContractError(f"{label} must be a normalized relative path")
    return path


def load_sglang_build_contract(path: Path) -> SGLangBuildContract:
    """Load and strictly validate a schema-1 SGLang build contract."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise SGLangBuildContractError(
            "could not read a valid SGLang build contract"
        ) from error
    return parse_sglang_build_contract(text)


def parse_sglang_build_contract(text: str) -> SGLangBuildContract:
    """Parse contract text already bound to a trusted local Git blob."""

    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise SGLangBuildContractError(
            "could not read a valid SGLang build contract"
        ) from error
    root = _mapping(document, "contract")
    _exact_keys(root, _TOP_LEVEL_KEYS, "contract")
    if type(root["schema_version"]) is not int or root["schema_version"] != (
        CONTRACT_SCHEMA_VERSION
    ):
        raise SGLangBuildContractError("unsupported contract.schema_version")
    if root["kind"] != CONTRACT_KIND:
        raise SGLangBuildContractError("contract.kind is not recognized")
    candidate_id = _string(root, "candidate_id", "contract")
    if (
        _LOWER_ID.fullmatch(candidate_id) is None
        or candidate_id != CONTRACT_CANDIDATE_ID
    ):
        raise SGLangBuildContractError(
            "contract.candidate_id is not the schema-1 candidate"
        )
    status = _string(root, "status", "contract")
    if status != CONTRACT_STATUS:
        raise SGLangBuildContractError(
            "contract.status must remain source_and_build_invocation_only"
        )

    source = _mapping(root["source"], "source")
    _exact_keys(source, _SOURCE_KEYS, "source")
    upstream_repository = _string(source, "upstream_repository", "source")
    storage_repository = _string(source, "storage_repository", "source")
    _require_https_repository(upstream_repository, "source.upstream_repository")
    _require_https_repository(storage_repository, "source.storage_repository")
    base_commit = _string(source, "base_commit", "source")
    base_tree = _string(source, "base_tree", "source")
    final_tree = _string(source, "final_tree", "source")
    _require_hex40(base_commit, "source.base_commit")
    _require_hex40(base_tree, "source.base_tree")
    _require_hex40(final_tree, "source.final_tree")
    excluded_commits = _string_list(source, "excluded_commits", "source")
    for index, commit in enumerate(excluded_commits):
        _require_hex40(commit, f"source.excluded_commits[{index}]")
    if excluded_commits != (EXCLUDED_QSA_COMMIT,):
        raise SGLangBuildContractError(
            "source.excluded_commits must contain only the competing QSA commit"
        )
    if base_commit in excluded_commits:
        raise SGLangBuildContractError(
            "source.base_commit cannot also be an excluded commit"
        )

    raw_protected_files = source.get("protected_files")
    if not isinstance(raw_protected_files, list) or not raw_protected_files:
        raise SGLangBuildContractError(
            "source.protected_files must be a non-empty array"
        )
    protected_files: list[ProtectedFile] = []
    protected_paths: set[str] = set()
    for index, raw_protected in enumerate(raw_protected_files):
        label = f"source.protected_files[{index}]"
        protected = _mapping(raw_protected, label)
        _exact_keys(protected, _PROTECTED_FILE_KEYS, label)
        protected_path = _string(protected, "path", label)
        _safe_relative_path(protected_path, f"{label}.path")
        protected_sha256 = _string(protected, "sha256", label)
        _require_hex64(protected_sha256, f"{label}.sha256")
        if protected_path in protected_paths:
            raise SGLangBuildContractError(
                "source.protected_files contains duplicate paths"
            )
        protected_paths.add(protected_path)
        protected_files.append(
            ProtectedFile(path=protected_path, sha256=protected_sha256)
        )
    if protected_paths != REQUIRED_PROTECTED_PATHS:
        raise SGLangBuildContractError(
            "source.protected_files must contain the complete QSA safety set"
        )

    raw_steps = source.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise SGLangBuildContractError("source.steps must be a non-empty array")
    steps: list[SourceStep] = []
    prior_tree = base_tree
    commits: set[str] = set()
    patch_paths: set[str] = set()
    for index, raw_step in enumerate(raw_steps):
        label = f"source.steps[{index}]"
        step = _mapping(raw_step, label)
        kind = _string(step, "kind", label)
        expected_keys = _COMMIT_STEP_KEYS if kind == "commit" else _PATCH_STEP_KEYS
        if kind not in {"commit", "patch"}:
            raise SGLangBuildContractError(f"{label}.kind is not recognized")
        _exact_keys(step, expected_keys, label)
        input_tree = _string(step, "input_tree", label)
        output_tree = _string(step, "output_tree", label)
        _require_hex40(input_tree, f"{label}.input_tree")
        _require_hex40(output_tree, f"{label}.output_tree")
        if input_tree != prior_tree:
            raise SGLangBuildContractError(
                f"{label}.input_tree does not continue the ordered source chain"
            )
        if output_tree == input_tree:
            raise SGLangBuildContractError(f"{label} does not change the source tree")
        if kind == "commit":
            repository = _string(step, "repository", label)
            if repository not in {"upstream", "storage"}:
                raise SGLangBuildContractError(
                    f"{label}.repository is not declared by the schema"
                )
            commit = _string(step, "commit", label)
            _require_hex40(commit, f"{label}.commit")
            if commit in commits:
                raise SGLangBuildContractError("source.steps repeats a commit")
            if commit in excluded_commits:
                raise SGLangBuildContractError(
                    "source.steps includes an explicitly excluded commit"
                )
            commits.add(commit)
            steps.append(
                SourceStep(
                    kind=kind,
                    repository=repository,
                    commit=commit,
                    input_tree=input_tree,
                    output_tree=output_tree,
                )
            )
        else:
            patch_path = _string(step, "path", label)
            _safe_relative_path(patch_path, f"{label}.path")
            sha256 = _string(step, "sha256", label)
            patch_id = _string(step, "patch_id", label)
            _require_hex64(sha256, f"{label}.sha256")
            _require_hex40(patch_id, f"{label}.patch_id")
            if patch_path in patch_paths:
                raise SGLangBuildContractError("source.steps repeats a patch path")
            patch_paths.add(patch_path)
            steps.append(
                SourceStep(
                    kind=kind,
                    path=patch_path,
                    sha256=sha256,
                    patch_id=patch_id,
                    input_tree=input_tree,
                    output_tree=output_tree,
                )
            )
        prior_tree = output_tree
    if prior_tree != final_tree:
        raise SGLangBuildContractError(
            "source.final_tree does not match the ordered source chain"
        )

    build = _mapping(root["build"], "build")
    _exact_keys(build, _BUILD_KEYS, "build")
    dockerfile = _string(build, "dockerfile", "build")
    dockerignore = _string(build, "dockerignore", "build")
    _safe_relative_path(dockerfile, "build.dockerfile")
    _safe_relative_path(dockerignore, "build.dockerignore")
    dockerfile_sha256 = _string(build, "dockerfile_sha256", "build")
    dockerignore_sha256 = _string(build, "dockerignore_sha256", "build")
    _require_hex64(dockerfile_sha256, "build.dockerfile_sha256")
    _require_hex64(dockerignore_sha256, "build.dockerignore_sha256")
    target = _string(build, "target", "build")
    platform = _string(build, "platform", "build")
    automatic_targetarch = _string(build, "automatic_targetarch", "build")
    context_mode = _string(build, "context_mode", "build")
    if target != "runtime" or platform != "linux/arm64":
        raise SGLangBuildContractError(
            "schema 1 is restricted to target runtime on linux/arm64"
        )
    if automatic_targetarch != "arm64":
        raise SGLangBuildContractError(
            "build.automatic_targetarch must be arm64"
        )
    if context_mode != CONTEXT_MODE:
        raise SGLangBuildContractError(
            "build.context_mode must be tracked-tree-export"
        )

    external_base_argument = _string(build, "external_base_argument", "build")
    if external_base_argument not in EXPECTED_BUILD_ARGUMENTS:
        raise SGLangBuildContractError(
            "build.external_base_argument is not an allowed build argument"
        )
    external_base_reference = _string(build, "external_base_reference", "build")
    external_base_index_digest = _string(
        build, "external_base_index_digest", "build"
    )
    external_base_manifest_digest = _string(
        build, "external_base_manifest_digest", "build"
    )
    external_base_config_digest = _string(
        build, "external_base_config_digest", "build"
    )
    for name, digest in (
        ("external_base_index_digest", external_base_index_digest),
        ("external_base_manifest_digest", external_base_manifest_digest),
        ("external_base_config_digest", external_base_config_digest),
    ):
        _require_digest(digest, f"build.{name}")
    if not external_base_reference.endswith(
        "@" + external_base_manifest_digest
    ):
        raise SGLangBuildContractError(
            "build.external_base_reference is not pinned to its ARM64 manifest"
        )
    external_base_stages = _string_list(build, "external_base_stages", "build")
    stage_names = _string_list(build, "stage_names", "build")
    if target not in stage_names or not set(external_base_stages).issubset(stage_names):
        raise SGLangBuildContractError(
            "build stage topology does not contain its target and external stages"
        )

    raw_build_args = _mapping(build["args"], "build.args")
    actual_arg_names = frozenset(raw_build_args)
    missing_args = sorted(EXPECTED_BUILD_ARGUMENTS - actual_arg_names)
    unknown_args = sorted(actual_arg_names - EXPECTED_BUILD_ARGUMENTS)
    if missing_args or unknown_args:
        details = []
        if missing_args:
            details.append("missing " + ", ".join(missing_args))
        if unknown_args:
            details.append("unknown " + ", ".join(unknown_args))
        raise SGLangBuildContractError(
            "build.args keys are invalid: " + "; ".join(details)
        )
    build_args: list[tuple[str, str]] = []
    for name, raw_value in raw_build_args.items():
        if not isinstance(raw_value, str):
            raise SGLangBuildContractError(f"build.args.{name} must be a string")
        if "\x00" in raw_value or "\n" in raw_value or "\r" in raw_value:
            raise SGLangBuildContractError(
                f"build.args.{name} contains a forbidden control character"
            )
        build_args.append((name, raw_value))
    build_arg_map = dict(build_args)
    if build_arg_map[external_base_argument] != external_base_reference:
        raise SGLangBuildContractError(
            "the external base argument does not equal its pinned reference"
        )
    if build_arg_map["BRANCH_TYPE"] != "local":
        raise SGLangBuildContractError("BRANCH_TYPE must select local source")
    if build_arg_map["USE_LATEST_SGLANG"] != "0":
        raise SGLangBuildContractError("USE_LATEST_SGLANG must remain disabled")
    if build_arg_map["SGLANG_BUILD_COMMIT"] != final_tree:
        raise SGLangBuildContractError(
            "SGLANG_BUILD_COMMIT must bind the composite final Git tree"
        )
    if build_arg_map["SGL_VERSION"] != "":
        raise SGLangBuildContractError(
            "SGL_VERSION must remain empty for BRANCH_TYPE=local"
        )
    for name in EXPLICIT_EMPTY_BUILD_ARGUMENTS:
        if build_arg_map[name] != "":
            raise SGLangBuildContractError(
                f"{name} must remain explicitly empty in schema 1"
            )

    return SGLangBuildContract(
        candidate_id=candidate_id,
        status=status,
        base_commit=base_commit,
        base_tree=base_tree,
        final_tree=final_tree,
        excluded_commits=excluded_commits,
        protected_files=tuple(protected_files),
        source_steps=tuple(steps),
        dockerfile=dockerfile,
        dockerfile_sha256=dockerfile_sha256,
        dockerignore=dockerignore,
        dockerignore_sha256=dockerignore_sha256,
        target=target,
        platform=platform,
        automatic_targetarch=automatic_targetarch,
        context_mode=context_mode,
        external_base_argument=external_base_argument,
        external_base_reference=external_base_reference,
        external_base_index_digest=external_base_index_digest,
        external_base_manifest_digest=external_base_manifest_digest,
        external_base_config_digest=external_base_config_digest,
        external_base_stages=external_base_stages,
        stage_names=stage_names,
        build_args=tuple(build_args),
    )


def _git_environment() -> dict[str, str]:
    return {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", ""),
    }


def _git(
    root: Path,
    arguments: Sequence[str],
    purpose: str,
    *,
    input_bytes: bytes | None = None,
    extra_environment: Mapping[str, str] | None = None,
) -> bytes:
    command = [
        "git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        f"core.hooksPath={os.devnull}",
        "-C",
        os.fspath(root),
        *arguments,
    ]
    environment = _git_environment()
    if extra_environment is not None:
        environment.update(extra_environment)
    try:
        result = subprocess.run(
            command,
            check=False,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            env=environment,
        )
    except FileNotFoundError as error:
        raise SGLangBuildContractError("Git is required for offline verification") from error
    except subprocess.TimeoutExpired as error:
        raise SGLangBuildContractError(f"Git timed out while {purpose}") from error
    if result.returncode != 0:
        raise SGLangBuildContractError(f"Git failed while {purpose}")
    if len(result.stdout) > _MAX_GIT_OUTPUT_BYTES:
        raise SGLangBuildContractError(f"Git returned excessive output while {purpose}")
    return result.stdout


def _require_git_root(root: Path, label: str) -> Path:
    if root.is_symlink():
        raise SGLangBuildContractError(f"{label} must not be a symlink")
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise SGLangBuildContractError(f"{label} does not exist") from error
    if not resolved.is_dir():
        raise SGLangBuildContractError(f"{label} must be a directory")
    output = _git(resolved, ["rev-parse", "--show-toplevel"], f"locating {label}")
    try:
        reported = Path(output.decode("utf-8").strip()).resolve(strict=True)
    except (OSError, UnicodeError) as error:
        raise SGLangBuildContractError(f"Git returned an invalid {label}") from error
    if reported != resolved:
        raise SGLangBuildContractError(f"{label} must be the exact Git repository root")
    return resolved


def _ascii_git_output(output: bytes, label: str) -> str:
    try:
        return output.decode("ascii", errors="strict").strip()
    except UnicodeError as error:
        raise SGLangBuildContractError(f"Git returned malformed {label}") from error


def _head_commit(root: Path, label: str) -> str:
    commit = _ascii_git_output(
        _git(root, ["rev-parse", "--verify", "HEAD^{commit}"], f"reading {label} HEAD"),
        f"{label} HEAD",
    )
    _require_hex40(commit, f"{label} HEAD")
    return commit


def _require_head_unchanged(root: Path, expected: str, label: str) -> None:
    if _head_commit(root, label) != expected:
        raise SGLangBuildContractError(f"{label} HEAD changed during verification")


def _head_regular_blob(
    root: Path, head: str, relative: str, label: str
) -> bytes:
    return _head_regular_blob_entry(root, head, relative, label).data


def _head_regular_blob_entry(
    root: Path, head: str, relative: str, label: str
) -> _GitBlob:
    path = _safe_relative_path(relative, label).as_posix()
    listing = _git(
        root,
        ["ls-tree", "-z", head, "--", path],
        f"locating pinned {label}",
    )
    records = [record for record in listing.split(b"\0") if record]
    if len(records) != 1 or b"\t" not in records[0]:
        raise SGLangBuildContractError(f"{label} is not one pinned Git blob")
    metadata, raw_path = records[0].split(b"\t", 1)
    try:
        mode, object_type, object_id = metadata.decode("ascii").split()
        listed_path = raw_path.decode("utf-8")
    except (UnicodeError, ValueError) as error:
        raise SGLangBuildContractError(f"Git returned malformed {label} metadata") from error
    if (
        mode not in {"100644", "100755"}
        or object_type != "blob"
        or _HEX40.fullmatch(object_id) is None
        or listed_path != path
    ):
        raise SGLangBuildContractError(f"{label} must be one regular tracked blob")
    return _GitBlob(
        mode=mode,
        data=_git(
            root,
            ["cat-file", "blob", object_id],
            f"reading pinned {label}",
        ),
    )


def _require_safe_index_flags(
    root: Path, label: str, paths: Sequence[str] | None = None
) -> None:
    arguments = ["ls-files", "-v", "-z"]
    if paths:
        arguments.extend(["--", *paths])
    output = _git(root, arguments, f"checking {label} index flags")
    records = [record for record in output.split(b"\0") if record]
    if not records:
        raise SGLangBuildContractError(f"{label} has no tracked files")
    for record in records:
        if not record.startswith(b"H "):
            raise SGLangBuildContractError(
                f"{label} contains assume-unchanged, skip-worktree, or unsafe index flags"
            )


def _require_safe_attributes(
    root: Path,
    label: str,
    paths: Sequence[str] | None = None,
    *,
    require_archive_safe: bool = False,
) -> None:
    if paths is None:
        tracked = [
            record
            for record in _git(
                root,
                ["ls-files", "-z"],
                f"listing tracked {label} paths for filter checks",
            ).split(b"\0")
            if record
        ]
    else:
        tracked = [path.encode("utf-8") for path in paths]
    if not tracked:
        raise SGLangBuildContractError(f"{label} has no tracked files")
    if require_archive_safe:
        archive_paths = list(tracked)
        for path in tracked:
            parts = path.split(b"/")
            archive_paths.extend(
                b"/".join(parts[:depth]) for depth in range(1, len(parts))
            )
        tracked = list(dict.fromkeys(archive_paths))
    attribute_names = [b"filter"]
    if require_archive_safe:
        attribute_names.extend((b"export-ignore", b"export-subst"))
    for offset in range(0, len(tracked), 256):
        chunk = tracked[offset : offset + 256]
        output = _git(
            root,
            [
                "check-attr",
                "-z",
                "--stdin",
                *(name.decode("ascii") for name in attribute_names),
            ],
            f"checking {label} safety attributes",
            input_bytes=b"\0".join(chunk) + b"\0",
        )
        fields = output.split(b"\0")
        if fields and fields[-1] == b"":
            fields.pop()
        if len(fields) != len(chunk) * len(attribute_names) * 3:
            raise SGLangBuildContractError(
                f"Git returned malformed {label} safety attributes"
            )
        field_index = 0
        for path in chunk:
            for expected_attribute in attribute_names:
                returned_path, attribute, value = fields[
                    field_index : field_index + 3
                ]
                field_index += 3
                if returned_path != path or attribute != expected_attribute:
                    raise SGLangBuildContractError(
                        f"Git returned malformed {label} safety attributes"
                    )
                if value in {b"unspecified", b"unset"}:
                    continue
                if attribute == b"filter":
                    raise SGLangBuildContractError(
                        f"{label} must not apply Git clean-filter attributes"
                    )
                raise SGLangBuildContractError(
                    f"{label} must not apply Git archive-control attributes"
                )


def _require_no_gitlinks(root: Path, label: str) -> None:
    records = _git(
        root,
        ["ls-files", "--stage", "-z"],
        f"checking {label} gitlinks",
    ).split(b"\0")
    if any(record.startswith(b"160000 ") for record in records if record):
        raise SGLangBuildContractError(
            f"{label} must not contain Git submodules or gitlinks"
        )


def _require_complete_history_controls(root: Path) -> None:
    shallow = _ascii_git_output(
        _git(
            root,
            ["rev-parse", "--is-shallow-repository"],
            "checking source history completeness",
        ),
        "source shallow state",
    )
    if shallow != "false":
        raise SGLangBuildContractError(
            "source repository must have non-shallow history"
        )
    replacements = _git(
        root,
        ["replace", "-l"],
        "checking source replacement objects",
    )
    if replacements:
        raise SGLangBuildContractError("source repository contains replacement objects")
    graft_path_text = _ascii_git_output(
        _git(
            root,
            ["rev-parse", "--git-path", "info/grafts"],
            "locating source grafts",
        ),
        "source graft path",
    )
    graft_path = Path(graft_path_text)
    if not graft_path.is_absolute():
        graft_path = root / graft_path
    try:
        if graft_path.exists() and graft_path.stat().st_size:
            raise SGLangBuildContractError("source repository contains grafted history")
    except OSError as error:
        raise SGLangBuildContractError("could not inspect source grafts") from error


def _require_object_integrity(root: Path, label: str) -> None:
    config_names = _git(
        root,
        ["config", "--includes", "--name-only", "--null", "--list"],
        f"reading {label} integrity configuration",
    )
    for raw_name in config_names.split(b"\0"):
        name = raw_name.lower()
        if name.startswith(b"tar."):
            raise SGLangBuildContractError(
                f"{label} contains archive-affecting Git configuration"
            )
        if (
            name.startswith(b"fsck.")
            or name.startswith(b"fetch.fsck.")
            or name.startswith(b"receive.fsck.")
            or name == b"core.alternaterefscommand"
            or name == b"extensions.partialclone"
            or (
                name.startswith(b"remote.")
                and (
                    name.endswith(b".promisor")
                    or name.endswith(b".partialclonefilter")
                )
            )
        ):
            raise SGLangBuildContractError(
                f"{label} contains integrity-bypass Git configuration"
            )
    alternates_text = _ascii_git_output(
        _git(
            root,
            ["rev-parse", "--git-path", "objects/info/alternates"],
            f"locating {label} alternate objects",
        ),
        f"{label} alternate-object path",
    )
    alternates = Path(alternates_text)
    if not alternates.is_absolute():
        alternates = root / alternates
    try:
        if alternates.is_symlink() or (
            alternates.exists()
            and (not alternates.is_file() or alternates.stat().st_size != 0)
        ):
            raise SGLangBuildContractError(
                f"{label} must not use alternate Git object databases"
            )
    except OSError as error:
        raise SGLangBuildContractError(
            f"could not inspect {label} alternate objects"
        ) from error
    objects_text = _ascii_git_output(
        _git(
            root,
            ["rev-parse", "--git-path", "objects"],
            f"locating {label} object database",
        ),
        f"{label} object-database path",
    )
    objects = Path(objects_text)
    if not objects.is_absolute():
        objects = root / objects
    try:
        if any((objects / "pack").glob("*.promisor")):
            raise SGLangBuildContractError(
                f"{label} must not use a partial-clone object database"
            )
    except OSError as error:
        raise SGLangBuildContractError(
            f"could not inspect {label} object database"
        ) from error
    _git(
        root,
        ["fsck", "--strict", "--full", "--no-dangling"],
        f"verifying {label} object integrity",
    )


def _repository_file(
    root: Path, relative: str, label: str, *, require_tracked: bool = True
) -> Path:
    posix_path = _safe_relative_path(relative, label)
    candidate = root.joinpath(*posix_path.parts)
    if candidate.is_symlink():
        raise SGLangBuildContractError(f"{label} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise SGLangBuildContractError(f"{label} is not a regular repository file") from error
    if not resolved.is_file():
        raise SGLangBuildContractError(f"{label} is not a regular repository file")
    if require_tracked:
        _git(
            root,
            ["ls-files", "--error-unmatch", "--", posix_path.as_posix()],
            f"checking tracked {label}",
        )
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise SGLangBuildContractError("could not hash a pinned file") from error
    return digest.hexdigest()


def _require_worktree_blob(
    root: Path, relative: str, expected: bytes, label: str
) -> Path:
    path = _repository_file(root, relative, label)
    try:
        actual = path.read_bytes()
    except OSError as error:
        raise SGLangBuildContractError(f"could not read {label}") from error
    if actual != expected:
        raise SGLangBuildContractError(f"{label} worktree bytes differ from HEAD")
    return path


def _patch_id(repository_root: Path, patch: bytes) -> str:
    output = _git(
        repository_root,
        ["patch-id", "--stable"],
        "calculating a tracked patch ID",
        input_bytes=patch,
    )
    try:
        lines = output.decode("ascii").splitlines()
    except UnicodeError as error:
        raise SGLangBuildContractError("Git returned a malformed patch ID") from error
    if len(lines) != 1:
        raise SGLangBuildContractError("a tracked patch must contain one logical patch")
    fields = lines[0].split()
    if len(fields) != 2 or _HEX40.fullmatch(fields[0]) is None:
        raise SGLangBuildContractError("Git returned a malformed patch ID")
    return fields[0]


def _contract_relative_path(repository_root: Path, contract_path: Path) -> str:
    try:
        candidate = (
            contract_path
            if contract_path.is_absolute()
            else repository_root / contract_path
        )
        if candidate.is_symlink():
            raise SGLangBuildContractError("contract path must not be a symlink")
        if contract_path.is_absolute():
            resolved = contract_path.resolve(strict=True)
        else:
            relative = _safe_relative_path(contract_path.as_posix(), "contract path")
            resolved = repository_root.joinpath(*relative.parts).resolve(strict=True)
        relative_path = resolved.relative_to(repository_root)
    except (OSError, ValueError) as error:
        raise SGLangBuildContractError(
            "contract path must resolve inside the repository root"
        ) from error
    return relative_path.as_posix()


def _verify_contract_files(
    repository_root: Path,
    repository_head: str,
    contract_relative: str,
    contract_blob: bytes,
    contract: SGLangBuildContract,
) -> dict[str, bytes]:
    tracked_paths = [
        contract_relative,
        *(
            step.path
            for step in contract.source_steps
            if step.kind == "patch" and step.path is not None
        ),
    ]
    _require_safe_attributes(
        repository_root, "contract repository", tracked_paths
    )
    _require_worktree_blob(
        repository_root, contract_relative, contract_blob, "contract path"
    )
    patches: dict[str, bytes] = {}
    for index, step in enumerate(contract.source_steps):
        if step.kind != "patch":
            continue
        assert step.path is not None
        assert step.sha256 is not None
        assert step.patch_id is not None
        patch = _head_regular_blob(
            repository_root,
            repository_head,
            step.path,
            f"source.steps[{index}].path",
        )
        _require_worktree_blob(
            repository_root,
            step.path,
            patch,
            f"source.steps[{index}].path",
        )
        if hashlib.sha256(patch).hexdigest() != step.sha256:
            raise SGLangBuildContractError(
                f"source.steps[{index}] patch SHA-256 does not match"
            )
        if _patch_id(repository_root, patch) != step.patch_id:
            raise SGLangBuildContractError(
                f"source.steps[{index}] stable patch ID does not match"
            )
        patches[step.path] = patch
    _require_safe_index_flags(
        repository_root, "contract repository", tracked_paths
    )
    dirty = _git(
        repository_root,
        ["status", "--porcelain=v1", "--untracked-files=all", "--", *tracked_paths],
        "checking contract and patch cleanliness",
    )
    if dirty:
        raise SGLangBuildContractError(
            "the contract and its tracked patches must be clean"
        )
    return patches


def parse_explicit_build_args(
    values: Sequence[str], expected: Mapping[str, str]
) -> dict[str, str]:
    """Parse explicit KEY=VALUE arguments without consulting the environment."""

    parsed: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            name = raw if _ARGUMENT_NAME.fullmatch(raw) else "build argument"
            raise SGLangBuildContractError(
                f"{name} is inherited; every build argument requires KEY=VALUE"
            )
        name, value = raw.split("=", 1)
        if _ARGUMENT_NAME.fullmatch(name) is None:
            raise SGLangBuildContractError("a build argument name is invalid")
        if name in parsed:
            raise SGLangBuildContractError(f"duplicate build argument: {name}")
        if "\x00" in value or "\n" in value or "\r" in value:
            raise SGLangBuildContractError(
                f"build argument {name} contains a forbidden control character"
            )
        parsed[name] = value
    expected_names = set(expected)
    actual_names = set(parsed)
    missing = sorted(expected_names - actual_names)
    unknown = sorted(actual_names - expected_names)
    mismatched = sorted(
        name
        for name in expected_names & actual_names
        if parsed[name] != expected[name]
    )
    if missing or unknown or mismatched:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        if mismatched:
            details.append("mismatched " + ", ".join(mismatched))
        raise SGLangBuildContractError(
            "explicit build arguments do not match the contract: "
            + "; ".join(details)
        )
    return parsed


def _logical_dockerfile_lines(text: str) -> list[tuple[int, str]]:
    output: list[tuple[int, str]] = []
    pending = ""
    start_line = 0
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not pending:
            start_line = line_number
        if stripped.endswith("\\"):
            pending += stripped[:-1].rstrip() + " "
            continue
        logical = (pending + stripped).strip()
        pending = ""
        if logical:
            output.append((start_line, logical))
    if pending:
        raise SGLangBuildContractError("Dockerfile ends in a continued instruction")
    return output


def _reject_unsupported_dockerfile_heredocs(text: str) -> None:
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.lstrip()
        if stripped and not stripped.startswith("#") and "<<" in stripped:
            raise SGLangBuildContractError(
                f"Dockerfile heredoc syntax at line {line_number} is unsupported"
            )


def _strip_unquoted_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, character in enumerate(line):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote != "'":
            escaped = True
            continue
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            continue
        if (
            character == "#"
            and quote is None
            and (index == 0 or line[index - 1].isspace())
        ):
            return line[:index]
    return line


def _dockerfile_topology(
    text: str,
) -> tuple[
    tuple[str, ...],
    tuple[_FromInstruction, ...],
    tuple[_ArgInstruction, ...],
]:
    stages: list[str] = []
    from_instructions: list[_FromInstruction] = []
    arg_instructions: list[_ArgInstruction] = []
    saw_from = False
    for line_number, line in _logical_dockerfile_lines(text):
        instruction = line.split(None, 1)[0].upper()
        if instruction not in {"ARG", "FROM"}:
            continue
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError as error:
            raise SGLangBuildContractError(
                f"Dockerfile instruction at line {line_number} is malformed"
            ) from error
        if instruction == "ARG":
            if len(tokens) != 2:
                raise SGLangBuildContractError(
                    f"Dockerfile ARG at line {line_number} is malformed"
                )
            name_value = tokens[1]
            name, separator, default = name_value.partition("=")
            if _ARGUMENT_NAME.fullmatch(name) is None:
                raise SGLangBuildContractError(
                    f"Dockerfile ARG at line {line_number} has an invalid name"
                )
            arg_instructions.append(
                _ArgInstruction(
                    name=name,
                    default=default if separator else None,
                    line_number=line_number,
                    global_scope=not saw_from,
                )
            )
            continue
        position = 1
        flags: list[str] = []
        while position < len(tokens) and tokens[position].startswith("--"):
            flags.append(tokens[position])
            position += 1
        remaining = tokens[position:]
        if len(remaining) != 3 or remaining[1].upper() != "AS":
            raise SGLangBuildContractError(
                f"Dockerfile FROM at line {line_number} must name one stage"
            )
        image, _, stage = remaining
        if not stage or stage in stages:
            raise SGLangBuildContractError(
                f"Dockerfile FROM at line {line_number} has a duplicate stage"
            )
        stages.append(stage)
        saw_from = True
        from_instructions.append(
            _FromInstruction(
                image=image,
                stage=stage,
                line_number=line_number,
                flags=tuple(flags),
            )
        )
    return tuple(stages), tuple(from_instructions), tuple(arg_instructions)


def _expanded_shell_references(text: str) -> set[str]:
    references: set[str] = set()

    def scan(index: int, *, stop_at_parenthesis: bool) -> int:
        quote: str | None = None
        while index < len(text):
            character = text[index]
            if character == "\\" and quote != "'":
                index += 2
                continue
            if quote == "'":
                if character == "'":
                    quote = None
                index += 1
                continue
            if character == "'" and quote is None:
                quote = "'"
                index += 1
                continue
            if character == '"':
                quote = None if quote == '"' else '"'
                index += 1
                continue
            if character == "`":
                raise SGLangBuildContractError(
                    "Dockerfile shell reference uses unsupported backtick substitution"
                )
            if quote is None and character == "#":
                raise SGLangBuildContractError(
                    "Dockerfile shell reference contains an unsupported shell comment"
                )
            if quote is None and stop_at_parenthesis:
                if (
                    text.startswith("case", index)
                    and (index == 0 or not (text[index - 1].isalnum() or text[index - 1] == "_"))
                    and (
                        index + 4 == len(text)
                        or not (text[index + 4].isalnum() or text[index + 4] == "_")
                    )
                ):
                    raise SGLangBuildContractError(
                        "Dockerfile command substitution contains an unsupported case construct"
                    )
                if character == "(":
                    raise SGLangBuildContractError(
                        "Dockerfile command substitution contains an unsupported raw parenthesis"
                    )
                if character == ")":
                    return index + 1
            if character != "$":
                index += 1
                continue
            if text.startswith("$$", index):
                index += 2
                continue
            if text.startswith("$(", index):
                index = scan(index + 2, stop_at_parenthesis=True)
                continue
            match = _BRACED_VARIABLE_AT.match(text, index)
            if match is None:
                match = _SIMPLE_VARIABLE_AT.match(text, index)
            if match is None:
                index += 1
                continue
            references.add(match.group(1))
            index = match.end()
        if quote is not None:
            raise SGLangBuildContractError(
                "Dockerfile shell reference contains an unterminated quote"
            )
        if stop_at_parenthesis:
            raise SGLangBuildContractError(
                "Dockerfile shell reference contains an unterminated command substitution"
            )
        return index

    scan(0, stop_at_parenthesis=False)
    return references


def _reachable_dockerfile_stages(
    text: str,
    from_instructions: Sequence[_FromInstruction],
    target: str,
) -> frozenset[str]:
    dependencies: dict[str, set[str]] = {}
    prior_stages: set[str] = set()
    current_stage: str | None = None
    from_index = 0
    for line_number, line in _logical_dockerfile_lines(text):
        instruction = line.split(None, 1)[0].upper()
        if instruction == "FROM":
            if from_index >= len(from_instructions):
                raise SGLangBuildContractError(
                    "Dockerfile stage dependency scan is inconsistent"
                )
            parsed = from_instructions[from_index]
            from_index += 1
            current_stage = parsed.stage.lower()
            dependencies[current_stage] = set()
            parent = parsed.image.lower()
            if parent in prior_stages:
                dependencies[current_stage].add(parent)
            prior_stages.add(current_stage)
            continue
        if current_stage is None or instruction not in {"ADD", "COPY", "RUN"}:
            continue
        _, separator, body = line.partition(" ")
        if not separator:
            raise SGLangBuildContractError(
                f"Dockerfile instruction at line {line_number} is malformed"
            )
        flag_tokens: list[str] = []
        remainder = body.lstrip()
        while remainder.startswith("--"):
            match = re.match(r"([^\s]+)(?:\s+|$)", remainder)
            if match is None:
                raise SGLangBuildContractError(
                    f"Dockerfile instruction at line {line_number} is malformed"
                )
            flag_tokens.append(match.group(1))
            remainder = remainder[match.end() :]
        source_stages: list[str] = []
        for token in flag_tokens:
            if instruction in {"ADD", "COPY"} and token.startswith("--from="):
                source_stages.append(token.partition("=")[2].lower())
            if instruction == "RUN" and token.startswith("--mount="):
                fields = token.partition("=")[2].split(",")
                source_stages.extend(
                    field.partition("=")[2].lower()
                    for field in fields
                    if field.startswith("from=")
                )
        for source_stage in source_stages:
            if source_stage not in prior_stages:
                raise SGLangBuildContractError(
                    "Dockerfile uses an unapproved external or forward stage source"
                )
            dependencies[current_stage].add(source_stage)
    if from_index != len(from_instructions):
        raise SGLangBuildContractError(
            "Dockerfile stage dependency scan is inconsistent"
        )
    target_stage = target.lower()
    if target_stage not in dependencies:
        raise SGLangBuildContractError("Dockerfile target stage is unavailable")
    reachable: set[str] = set()
    pending = [target_stage]
    while pending:
        stage = pending.pop()
        if stage in reachable:
            continue
        reachable.add(stage)
        pending.extend(dependencies[stage] - reachable)
    return frozenset(reachable)


def _reachable_build_arg_references(
    text: str, reachable_stages: frozenset[str]
) -> set[str]:
    references: set[str] = set()
    global_args: set[str] = set()
    stage_args: dict[str, set[str]] = {}
    current_args: set[str] | None = None
    current_stage: str | None = None
    for line_number, line in _logical_dockerfile_lines(text):
        instruction, separator, body = line.partition(" ")
        instruction = instruction.upper()
        if instruction == "ARG":
            try:
                tokens = shlex.split(line, comments=True, posix=True)
            except ValueError as error:
                raise SGLangBuildContractError(
                    f"Dockerfile ARG at line {line_number} is malformed"
                ) from error
            if len(tokens) != 2:
                raise SGLangBuildContractError(
                    f"Dockerfile ARG at line {line_number} is malformed"
                )
            name = tokens[1].partition("=")[0]
            if current_args is None:
                global_args.add(name)
            else:
                current_args.add(name)
            continue
        if instruction == "FROM":
            try:
                tokens = shlex.split(line, comments=True, posix=True)
            except ValueError as error:
                raise SGLangBuildContractError(
                    f"Dockerfile FROM at line {line_number} is malformed"
                ) from error
            position = 1
            while position < len(tokens) and tokens[position].startswith("--"):
                position += 1
            remaining = tokens[position:]
            if len(remaining) != 3 or remaining[1].upper() != "AS":
                raise SGLangBuildContractError(
                    f"Dockerfile FROM at line {line_number} must name one stage"
                )
            image, _, stage = remaining
            current_stage = stage.lower()
            if current_stage in reachable_stages:
                references.update(_expanded_shell_references(image) & global_args)
            current_args = set(stage_args.get(image.lower(), set()))
            stage_args[current_stage] = current_args
            continue
        if (
            current_args is None
            or current_stage not in reachable_stages
            or not separator
        ):
            continue
        semantic_body = _strip_unquoted_comment(body)
        if instruction == "RUN":
            command = re.sub(r"^(?:--[^\s]+\s+)*", "", semantic_body).lstrip()
            if command.startswith("["):
                continue
            expanded = _expanded_shell_references(command)
        elif instruction in {"ENV", "LABEL"}:
            expanded = _expanded_shell_references(semantic_body)
        else:
            continue
        references.update(expanded & current_args)
    return references


def _lexical_non_arg_references(text: str) -> set[str]:
    references: set[str] = set()
    for _, line in _logical_dockerfile_lines(text):
        instruction, separator, body = line.partition(" ")
        if instruction.upper() == "ARG" or not separator:
            continue
        references.update(_SHELL_IDENTIFIER.findall(body))
    return references


def _resolve_from_image(image: str, build_args: Mapping[str, str]) -> str:
    missing: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        value = build_args.get(name)
        if value is None:
            missing.add(name)
            return ""
        return value

    resolved = _FROM_VARIABLE.sub(replace, image)
    if missing or "$" in resolved:
        raise SGLangBuildContractError(
            "Dockerfile FROM contains an unresolved build argument"
        )
    return resolved


def _verify_dockerfile(
    dockerfile_text: str,
    contract: SGLangBuildContract,
    build_args: Mapping[str, str],
) -> None:
    _reject_unsupported_dockerfile_heredocs(dockerfile_text)
    stages, instructions, arg_instructions = _dockerfile_topology(
        dockerfile_text
    )
    if stages != contract.stage_names:
        raise SGLangBuildContractError("Dockerfile stage topology does not match")
    declared_args = {instruction.name for instruction in arg_instructions}
    expected_declared = set(
        EXPECTED_BUILD_ARGUMENTS
        | AUTOMATIC_BUILD_ARGUMENTS
        | INERT_DOCKERFILE_ARGUMENTS
    )
    missing_declared = sorted(expected_declared - declared_args)
    unknown_declared = sorted(declared_args - expected_declared)
    if missing_declared or unknown_declared:
        details = []
        if missing_declared:
            details.append("missing " + ", ".join(missing_declared))
        if unknown_declared:
            details.append("unknown " + ", ".join(unknown_declared))
        raise SGLangBuildContractError(
            "Dockerfile ARG declarations do not match: " + "; ".join(details)
        )
    external_declarations = [
        instruction
        for instruction in arg_instructions
        if instruction.name == contract.external_base_argument
    ]
    if (
        len(external_declarations) != 1
        or not external_declarations[0].global_scope
        or external_declarations[0].default != contract.external_base_reference
    ):
        raise SGLangBuildContractError(
            "Dockerfile external base ARG must be declared once with its pinned "
            "default in global scope before the first FROM"
        )
    nonempty_defaults = sorted(
        name
        for name in EXPLICIT_EMPTY_BUILD_ARGUMENTS
        if any(
            instruction.name == name
            and instruction.default not in {None, ""}
            for instruction in arg_instructions
        )
    )
    if nonempty_defaults:
        raise SGLangBuildContractError(
            "an explicit empty build argument would override a non-empty default: "
            + ", ".join(nonempty_defaults)
        )

    reachable_stages = _reachable_dockerfile_stages(
        dockerfile_text, instructions, contract.target
    )
    referenced_arguments = _reachable_build_arg_references(
        dockerfile_text, reachable_stages
    )
    unreferenced = sorted(EXPECTED_BUILD_ARGUMENTS - referenced_arguments)
    if unreferenced:
        raise SGLangBuildContractError(
            "Dockerfile build arguments lack a recognized reachable reference: "
            + ", ".join(unreferenced)
        )
    unexpectedly_consumed = sorted(
        INERT_DOCKERFILE_ARGUMENTS & _lexical_non_arg_references(dockerfile_text)
    )
    if unexpectedly_consumed:
        raise SGLangBuildContractError(
            "an intentionally omitted Dockerfile ARG became lexically referenced: "
            + ", ".join(unexpectedly_consumed)
        )

    prior_stages: set[str] = set()
    external_stages: list[str] = []
    for instruction in instructions:
        if instruction.flags:
            raise SGLangBuildContractError(
                f"Dockerfile stage {instruction.stage} has unapproved FROM flags"
            )
        image_lower = instruction.image.lower()
        if image_lower == "scratch" or image_lower in prior_stages:
            prior_stages.add(instruction.stage.lower())
            continue
        expected_expression = "${" + contract.external_base_argument + "}"
        if instruction.image != expected_expression:
            raise SGLangBuildContractError(
                f"Dockerfile stage {instruction.stage} has an unapproved external FROM"
            )
        resolved = _resolve_from_image(instruction.image, build_args)
        if resolved != contract.external_base_reference:
            raise SGLangBuildContractError(
                f"Dockerfile stage {instruction.stage} resolves the wrong external base"
            )
        if not resolved.endswith("@" + contract.external_base_manifest_digest):
            raise SGLangBuildContractError(
                f"Dockerfile stage {instruction.stage} is not pinned to the ARM64 manifest"
            )
        external_stages.append(instruction.stage)
        prior_stages.add(instruction.stage.lower())
    if tuple(external_stages) != contract.external_base_stages:
        raise SGLangBuildContractError(
            "Dockerfile external base stages do not match the contract"
        )
    if contract.target not in stages:
        raise SGLangBuildContractError("Dockerfile does not contain the pinned target")


def _require_commit_object(root: Path, commit: str, label: str) -> None:
    _git(
        root,
        ["cat-file", "-e", f"{commit}^{{commit}}"],
        f"checking {label} commit object",
    )


def _commit_patch(root: Path, commit: str, label: str) -> bytes:
    _require_commit_object(root, commit, label)
    parents = _ascii_git_output(
        _git(
            root,
            ["rev-list", "--parents", "-n", "1", commit],
            f"reading {label} parents",
        ),
        f"{label} parents",
    ).split()
    if len(parents) != 2 or parents[0] != commit:
        raise SGLangBuildContractError(
            f"{label} must be one non-merge commit with one parent"
        )
    return _git(
        root,
        [
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "-p",
            "--binary",
            "--full-index",
            "--no-renames",
            parents[1],
            commit,
            "--",
        ],
        f"reading {label} patch",
    )


def _replay_temp_parent(forbidden_roots: Sequence[Path]) -> Path:
    for candidate in (Path("/tmp"), Path("/var/tmp")):
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if not resolved.is_dir() or any(
            resolved == root or resolved.is_relative_to(root)
            for root in forbidden_roots
        ):
            continue
        return resolved
    raise SGLangBuildContractError(
        "no replay scratch parent is available outside the inspected repositories"
    )


def _replay_source_chain(
    source_root: Path,
    contract_repository_root: Path,
    contract: SGLangBuildContract,
    tracked_patches: Mapping[str, bytes],
) -> None:
    objects_text = _ascii_git_output(
        _git(
            source_root,
            ["rev-parse", "--git-path", "objects"],
            "locating source Git objects",
        ),
        "source Git object path",
    )
    source_objects = Path(objects_text)
    if not source_objects.is_absolute():
        source_objects = source_root / source_objects
    try:
        source_objects = source_objects.resolve(strict=True)
    except OSError as error:
        raise SGLangBuildContractError("source Git object directory is unavailable") from error

    forbidden_roots = (source_root, contract_repository_root)
    replay_parent = _replay_temp_parent(forbidden_roots)
    with tempfile.TemporaryDirectory(
        prefix="sparkbench-sglang-contract-",
        dir=replay_parent,
    ) as directory:
        scratch = Path(directory).resolve()
        if any(
            scratch == root or scratch.is_relative_to(root)
            for root in forbidden_roots
        ):
            raise SGLangBuildContractError(
                "replay scratch resolved inside an inspected repository"
            )
        scratch_objects = scratch / "objects"
        scratch_objects.mkdir()
        replay_environment = {
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": os.fspath(source_objects),
            "GIT_INDEX_FILE": os.fspath(scratch / "index"),
            "GIT_OBJECT_DIRECTORY": os.fspath(scratch_objects),
        }
        _git(
            source_root,
            ["read-tree", contract.base_tree],
            "initializing the isolated source replay",
            extra_environment=replay_environment,
        )
        current_tree = contract.base_tree
        for index, step in enumerate(contract.source_steps):
            if step.input_tree != current_tree:
                raise SGLangBuildContractError(
                    f"source.steps[{index}] replay input tree is out of order"
                )
            if step.kind == "commit":
                assert step.commit is not None
                patch = _commit_patch(
                    source_root,
                    step.commit,
                    f"source.steps[{index}]",
                )
            else:
                assert step.path is not None
                patch = tracked_patches.get(step.path)
                if patch is None:
                    raise SGLangBuildContractError(
                        f"source.steps[{index}] tracked patch is unavailable"
                    )
            if not patch:
                raise SGLangBuildContractError(
                    f"source.steps[{index}] has an empty patch"
                )
            _git(
                source_root,
                [
                    "apply",
                    "--cached",
                    "--binary",
                    "--unidiff-zero",
                    "--whitespace=nowarn",
                    "-",
                ],
                f"replaying source.steps[{index}] in isolated scratch state",
                input_bytes=patch,
                extra_environment=replay_environment,
            )
            current_tree = _ascii_git_output(
                _git(
                    source_root,
                    ["write-tree"],
                    f"calculating source.steps[{index}] output tree",
                    extra_environment=replay_environment,
                ),
                f"source.steps[{index}] output tree",
            )
            if current_tree != step.output_tree:
                raise SGLangBuildContractError(
                    f"source.steps[{index}] does not produce its pinned output tree"
                )
        if current_tree != contract.final_tree:
            raise SGLangBuildContractError(
                "isolated source replay does not produce source.final_tree"
            )


def _verify_source_repository(
    source_root: Path,
    contract_repository_root: Path,
    contract: SGLangBuildContract,
    build_args: Mapping[str, str],
    tracked_patches: Mapping[str, bytes],
) -> None:
    root = _require_git_root(source_root, "source root")
    _require_object_integrity(root, "source repository")
    source_head = _head_commit(root, "source")
    _require_complete_history_controls(root)
    _require_safe_index_flags(root, "source repository")
    _require_no_gitlinks(root, "source repository")
    _require_safe_attributes(
        root, "source repository", require_archive_safe=True
    )
    dirty = _git(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"],
        "checking source cleanliness",
    )
    if dirty:
        raise SGLangBuildContractError("source root must have a clean working tree")
    ignored = _git(
        root,
        ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
        "checking ignored source files",
    )
    if ignored:
        raise SGLangBuildContractError(
            "source root contains ignored files; use a pristine tracked-tree export"
        )
    head_tree = _ascii_git_output(
        _git(
            root,
            ["rev-parse", "--verify", f"{source_head}^{{tree}}"],
            "reading the source HEAD tree",
        ),
        "source HEAD tree",
    )
    if head_tree != contract.final_tree:
        raise SGLangBuildContractError("source HEAD tree does not match the contract")

    _require_commit_object(root, contract.base_commit, "source base")
    base_tree = _ascii_git_output(
        _git(
            root,
            ["rev-parse", "--verify", f"{contract.base_commit}^{{tree}}"],
            "reading the source base tree",
        ),
        "source base tree",
    )
    if base_tree != contract.base_tree:
        raise SGLangBuildContractError(
            "source.base_commit does not resolve to source.base_tree"
        )
    for excluded in contract.excluded_commits:
        _require_commit_object(root, excluded, "excluded QSA")
    ancestry = _ascii_git_output(
        _git(
            root,
            ["rev-list", source_head],
            "checking excluded source ancestry",
        ),
        "source ancestry",
    ).splitlines()
    excluded_ancestors = sorted(set(ancestry) & set(contract.excluded_commits))
    if excluded_ancestors:
        raise SGLangBuildContractError(
            "source HEAD descends from an explicitly excluded commit"
        )
    if contract.base_commit not in ancestry:
        raise SGLangBuildContractError(
            "source HEAD ancestry does not contain the pinned base commit"
        )
    _replay_source_chain(
        root,
        contract_repository_root,
        contract,
        tracked_patches,
    )

    for index, protected in enumerate(contract.protected_files):
        base_protected_blob = _head_regular_blob_entry(
            root,
            contract.base_commit,
            protected.path,
            f"source.protected_files[{index}].base_path",
        )
        protected_blob = _head_regular_blob_entry(
            root,
            source_head,
            protected.path,
            f"source.protected_files[{index}].path",
        )
        _require_worktree_blob(
            root,
            protected.path,
            protected_blob.data,
            f"source.protected_files[{index}].path",
        )
        if (
            protected_blob.mode != base_protected_blob.mode
            or protected_blob.data != base_protected_blob.data
        ):
            raise SGLangBuildContractError(
                f"source.protected_files[{index}] differs from source.base_commit"
            )
        if hashlib.sha256(protected_blob.data).hexdigest() != protected.sha256:
            raise SGLangBuildContractError(
                f"source.protected_files[{index}] SHA-256 does not match"
            )

    dockerfile_blob = _head_regular_blob(
        root, source_head, contract.dockerfile, "build.dockerfile"
    )
    dockerignore_blob = _head_regular_blob(
        root, source_head, contract.dockerignore, "build.dockerignore"
    )
    _require_worktree_blob(
        root, contract.dockerfile, dockerfile_blob, "build.dockerfile"
    )
    _require_worktree_blob(
        root, contract.dockerignore, dockerignore_blob, "build.dockerignore"
    )
    if hashlib.sha256(dockerfile_blob).hexdigest() != contract.dockerfile_sha256:
        raise SGLangBuildContractError("Dockerfile SHA-256 does not match the contract")
    if hashlib.sha256(dockerignore_blob).hexdigest() != contract.dockerignore_sha256:
        raise SGLangBuildContractError(".dockerignore SHA-256 does not match the contract")
    try:
        dockerfile_text = dockerfile_blob.decode("utf-8")
    except UnicodeError as error:
        raise SGLangBuildContractError("Dockerfile must be valid UTF-8") from error
    _verify_dockerfile(dockerfile_text, contract, build_args)
    _require_head_unchanged(root, source_head, "source")


def verify_sglang_build_contract(
    *,
    repository_root: Path,
    contract_path: Path,
    source_root: Path,
    target: str,
    platform: str,
    build_args: Sequence[str],
) -> SGLangBuildVerification:
    """Verify one local source tree and exact build invocation, offline."""

    root = _require_git_root(repository_root, "contract repository root")
    _require_object_integrity(root, "contract repository")
    repository_head = _head_commit(root, "contract repository")
    contract_relative = _contract_relative_path(root, contract_path)
    contract_blob = _head_regular_blob(
        root, repository_head, contract_relative, "contract path"
    )
    try:
        contract_text = contract_blob.decode("utf-8")
    except UnicodeError as error:
        raise SGLangBuildContractError("contract must be valid UTF-8") from error
    contract = parse_sglang_build_contract(contract_text)
    tracked_patches = _verify_contract_files(
        root,
        repository_head,
        contract_relative,
        contract_blob,
        contract,
    )
    if target != contract.target:
        raise SGLangBuildContractError("requested target does not match the contract")
    if platform != contract.platform:
        raise SGLangBuildContractError("requested platform does not match the contract")
    explicit_args = parse_explicit_build_args(build_args, contract.build_arg_map())
    _verify_source_repository(
        source_root,
        root,
        contract,
        explicit_args,
        tracked_patches,
    )
    _require_head_unchanged(root, repository_head, "contract repository")
    return SGLangBuildVerification(
        candidate_id=contract.candidate_id,
        status=contract.status,
        source_tree=contract.final_tree,
        dockerfile_sha256=contract.dockerfile_sha256,
        target=contract.target,
        platform=contract.platform,
        external_base_reference=contract.external_base_reference,
        build_arg_count=len(explicit_args),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a pinned SGLang source tree and explicit Docker build "
            "invocation without network, Docker, or inspected-repository mutation."
        )
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument(
        "--build-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "one explicit build argument; repeat for the complete contract map "
            "(KEY without '=' is rejected as inherited)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        verification = verify_sglang_build_contract(
            repository_root=args.repository_root,
            contract_path=args.contract,
            source_root=args.source_root,
            target=args.target,
            platform=args.platform,
            build_args=args.build_arg,
        )
    except SGLangBuildContractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(verification.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
