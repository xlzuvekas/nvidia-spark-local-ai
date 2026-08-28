from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zlib
from unittest.mock import patch

from bench import sglang_build_contract as contract


BASE_MANIFEST = "sha256:" + "a" * 64
BASE_INDEX = "sha256:" + "b" * 64
BASE_CONFIG = "sha256:" + "c" * 64
BASE_REFERENCE = (
    "nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04@" + BASE_MANIFEST
)
STAGES = (
    "base",
    "torch_deps",
    "hpc_ops_builder",
    "flashinfer_cache",
    "devtools_builder",
    "gateway_builder",
    "framework",
    "local_src",
    "framework_final",
    "runtime",
)

EXCLUDED_QSA_COMMIT_OBJECT = b"""tree 0b52b53b8dacd09c0aa8f292e54445f79ed322de
parent 9f101e39ff09b356355e6a11183eaa3f7bb15f8c
author John Zinno <62895131+jzinno@users.noreply.github.com> 1787776650 -0400
committer John Zinno <62895131+jzinno@users.noreply.github.com> 1787844904 -0400

feat(qsa): add SM121 sparse GQA decode kernel
"""


def _run_git(root: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def _initialize_repository(root: Path) -> None:
    root.mkdir(parents=True)
    _run_git(root, "init", "--quiet")
    _run_git(root, "config", "user.name", "Synthetic Fixture")
    _run_git(root, "config", "user.email", "fixture@example.invalid")


def _commit(root: Path, message: str) -> str:
    _run_git(root, "add", "-A")
    _run_git(root, "commit", "--quiet", "-m", message)
    return _run_git(root, "rev-parse", "HEAD").decode("ascii").strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patch_id(root: Path, path: Path) -> str:
    output = _run_git(root, "patch-id", "--stable", input_bytes=path.read_bytes())
    return output.decode("ascii").split()[0]


def _corrupt_loose_blob(
    root: Path,
    revision_path: str,
    original: bytes,
    replacement: bytes,
) -> None:
    if len(original) != len(replacement):
        raise AssertionError("corruption replacement must preserve blob length")
    object_id = _run_git(root, "rev-parse", revision_path).decode("ascii").strip()
    object_path = root / ".git" / "objects" / object_id[:2] / object_id[2:]
    inflated = zlib.decompress(object_path.read_bytes())
    header, separator, data = inflated.partition(b"\0")
    if not separator or data.count(original) != 1:
        raise AssertionError("synthetic loose blob does not contain expected bytes")
    object_path.chmod(0o644)
    object_path.write_bytes(
        zlib.compress(header + b"\0" + data.replace(original, replacement, 1))
    )


def _dockerfile(build_args: dict[str, str]) -> str:
    declarations = []
    for name, value in build_args.items():
        if name in {"CUDA_BASE_IMAGE", "CUDA_VERSION"}:
            continue
        if name == "SGLANG_BUILD_COMMIT":
            value = "unknown"
        elif name == "SGLANG_IMAGE_TAG":
            value = "local/sglang:dev"
        declarations.append(f'ARG {name}="{value}"' if " " in value else f"ARG {name}={value}")
    references = " ".join("${" + name + "}" for name in build_args)
    return "\n".join(
        [
            "ARG CUDA_VERSION=13.0.3",
            f"ARG CUDA_BASE_IMAGE={BASE_REFERENCE}",
            "FROM ${CUDA_BASE_IMAGE} AS base",
            "ARG CUDA_VERSION",
            "ARG TARGETARCH",
            *declarations,
            'ARG MOONCAKE_COMPILE_ARG="-DUSE_HTTP=ON -DUSE_CUDA=ON"',
            f"RUN echo {references} ${{TARGETARCH}} >/dev/null",
            "FROM base AS torch_deps",
            "FROM torch_deps AS hpc_ops_builder",
            "FROM torch_deps AS flashinfer_cache",
            "FROM base AS devtools_builder",
            "FROM base AS gateway_builder",
            "FROM torch_deps AS framework",
            "FROM scratch AS local_src",
            "FROM framework AS framework_final",
            "FROM ${CUDA_BASE_IMAGE} AS runtime",
            "COPY --from=framework_final /sgl-workspace /sgl-workspace",
            "",
        ]
    )


class SyntheticContract:
    def __init__(self, *, protected_drift: str | None = None) -> None:
        if protected_drift not in {None, "added", "changed", "deleted", "mode"}:
            raise AssertionError("unknown synthetic protected-file drift")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository_root = self.root / "contract-repository"
        self.source_root = self.root / "source"
        _initialize_repository(self.source_root)
        (self.source_root / "base.txt").write_text("base\n", encoding="utf-8")
        self.protected_relative = (
            "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
        )
        self.protected_paths: dict[str, Path] = {}
        for relative in sorted(contract.REQUIRED_PROTECTED_PATHS):
            path = self.source_root.joinpath(*Path(relative).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            if protected_drift != "added" or relative != self.protected_relative:
                path.write_text(
                    f"safe protected content for {relative}\n",
                    encoding="utf-8",
                )
            self.protected_paths[relative] = path
        self.protected_path = self.protected_paths[self.protected_relative]
        base_protected_hashes = {
            relative: _sha256(path)
            for relative, path in self.protected_paths.items()
            if path.exists()
        }
        self.base_commit = _commit(self.source_root, "synthetic base")
        self.base_tree = self.head_tree()
        self.build_args = {
            "CUDA_BASE_IMAGE": BASE_REFERENCE,
            "CUDA_VERSION": "13.0.3",
            "BUILD_TYPE": "all",
            "BRANCH_TYPE": "local",
            "SGL_KERNEL_VERSION": "0.4.6.post1",
            "SGL_VERSION": "",
            "SGL_DEEP_GEMM_VERSION": "0.1.5.post2",
            "USE_LATEST_SGLANG": "0",
            "GDRCOPY_VERSION": "2.5.1",
            "PIP_DEFAULT_INDEX": "",
            "UBUNTU_MIRROR": "",
            "GITHUB_ARTIFACTORY": "github.com",
            "INSTALL_FLASHINFER_JIT_CACHE": "1",
            "FLASHINFER_VERSION": "0.6.17",
            "MOONCAKE_VERSION": "0.3.12.post1",
            "MSCCLPP_VERSION": "sglang-v0.9.1",
            "HPC_OPS_COMMIT": "d" * 40,
            "SGLANG_BUILD_COMMIT": "",
            "SGLANG_BUILD_URL": "",
            "SGLANG_IMAGE_TAG": "local/sglang:synthetic-runtime",
        }
        (self.source_root / "docker").mkdir()
        if protected_drift == "added":
            self.protected_path.write_text(
                f"safe protected content for {self.protected_relative}\n",
                encoding="utf-8",
            )
        elif protected_drift == "changed":
            self.protected_path.write_text(
                "changed competing QSA content\n",
                encoding="utf-8",
            )
        elif protected_drift == "deleted":
            self.protected_path.unlink()
        elif protected_drift == "mode":
            self.protected_path.chmod(0o755)
        self.protected_hashes = {
            relative: (
                _sha256(path)
                if path.exists()
                else base_protected_hashes[relative]
            )
            for relative, path in self.protected_paths.items()
        }
        (self.source_root / ".gitignore").write_text("ignored.tmp\n", encoding="utf-8")
        (self.source_root / ".dockerignore").write_text(".git\n", encoding="utf-8")
        dockerfile = self.source_root / "docker" / "Dockerfile"
        dockerfile.write_text(_dockerfile(self.build_args), encoding="utf-8")
        (self.source_root / "value.txt").write_text("old\n", encoding="utf-8")
        self.source_change_commit = _commit(self.source_root, "synthetic source")
        self.source_change_tree = self.head_tree()
        (self.source_root / "value.txt").write_text("new\n", encoding="utf-8")
        self.source_patch = _run_git(
            self.source_root,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "--full-index",
            "--no-renames",
            "--",
            "value.txt",
        )
        _commit(self.source_root, "synthetic tracked patch result")
        self.source_tree = self.head_tree()
        self.build_args["SGLANG_BUILD_COMMIT"] = self.source_tree
        excluded_object = _run_git(
            self.source_root,
            "hash-object",
            "-t",
            "commit",
            "-w",
            "--stdin",
            input_bytes=EXCLUDED_QSA_COMMIT_OBJECT,
        ).decode("ascii").strip()
        if excluded_object != contract.EXCLUDED_QSA_COMMIT:
            raise AssertionError("the embedded excluded-QSA commit object drifted")

        _initialize_repository(self.repository_root)
        patch_root = self.repository_root / "patches" / "sglang"
        patch_root.mkdir(parents=True)
        self.patch_relative = "patches/sglang/synthetic.patch"
        self.patch_path = self.repository_root / self.patch_relative
        self.patch_path.write_bytes(self.source_patch)
        self.contract_relative = "patches/sglang/synthetic-v1.toml"
        self.contract_path = self.repository_root / self.contract_relative
        self.excluded_commits = [contract.EXCLUDED_QSA_COMMIT]
        self.write_contract()
        _commit(self.repository_root, "synthetic contract")

    def cleanup(self) -> None:
        self.temporary.cleanup()

    def head_tree(self) -> str:
        return _run_git(
            self.source_root, "rev-parse", "HEAD^{tree}"
        ).decode("ascii").strip()

    def explicit_args(self) -> list[str]:
        values = dict(self.build_args)
        values["SGLANG_BUILD_COMMIT"] = self.source_tree
        return [f"{name}={value}" for name, value in values.items()]

    def write_contract(self, *, extra_top_level: str = "") -> None:
        dockerfile = self.source_root / "docker" / "Dockerfile"
        build_args = dict(self.build_args)
        build_args["SGLANG_BUILD_COMMIT"] = self.source_tree
        args_text = "\n".join(
            f"{name} = {json.dumps(value)}" for name, value in build_args.items()
        )
        stage_text = ", ".join(json.dumps(stage) for stage in STAGES)
        excluded_text = ", ".join(
            json.dumps(commit) for commit in self.excluded_commits
        )
        text = f'''schema_version = 1
kind = "sparkbench-sglang-build-candidate"
candidate_id = "sglang-sm121-triton-storage-v1"
status = "source_and_build_invocation_only"
{extra_top_level}
[source]
upstream_repository = "https://github.com/sgl-project/sglang.git"
storage_repository = "https://github.com/example/sglang.git"
base_commit = "{self.base_commit}"
base_tree = "{self.base_tree}"
final_tree = "{self.source_tree}"
excluded_commits = [{excluded_text}]

[[source.steps]]
kind = "commit"
repository = "storage"
commit = "{self.source_change_commit}"
input_tree = "{self.base_tree}"
output_tree = "{self.source_change_tree}"

[[source.steps]]
kind = "patch"
path = "{self.patch_relative}"
sha256 = "{_sha256(self.patch_path)}"
patch_id = "{_patch_id(self.repository_root, self.patch_path)}"
input_tree = "{self.source_change_tree}"
output_tree = "{self.source_tree}"

[build]
dockerfile = "docker/Dockerfile"
dockerfile_sha256 = "{_sha256(dockerfile)}"
dockerignore = ".dockerignore"
dockerignore_sha256 = "{_sha256(self.source_root / '.dockerignore')}"
target = "runtime"
platform = "linux/arm64"
automatic_targetarch = "arm64"
context_mode = "tracked-tree-export"
external_base_argument = "CUDA_BASE_IMAGE"
external_base_reference = "{BASE_REFERENCE}"
external_base_index_digest = "{BASE_INDEX}"
external_base_manifest_digest = "{BASE_MANIFEST}"
external_base_config_digest = "{BASE_CONFIG}"
external_base_stages = ["base", "runtime"]
stage_names = [{stage_text}]

[build.args]
{args_text}
'''
        protected_text = "\n".join(
            "\n".join(
                (
                    "[[source.protected_files]]",
                    f"path = {json.dumps(relative)}",
                    f"sha256 = {json.dumps(self.protected_hashes[relative])}",
                    "",
                )
            )
            for relative in sorted(self.protected_paths)
        )
        text = text.replace(
            "\n[[source.steps]]",
            "\n" + protected_text + "[[source.steps]]",
            1,
        )
        self.contract_path.write_text(text, encoding="utf-8")

    def commit_contract(self, message: str = "update synthetic contract") -> None:
        _commit(self.repository_root, message)

    def verify(
        self,
        *,
        build_args: list[str] | None = None,
        target: str = "runtime",
        platform: str = "linux/arm64",
        repository_root: Path | None = None,
    ) -> contract.SGLangBuildVerification:
        return contract.verify_sglang_build_contract(
            repository_root=repository_root or self.repository_root,
            contract_path=Path(self.contract_relative),
            source_root=self.source_root,
            target=target,
            platform=platform,
            build_args=build_args if build_args is not None else self.explicit_args(),
        )


class SGLangBuildContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = SyntheticContract()

    def tearDown(self) -> None:
        self.fixture.cleanup()

    def test_exact_contract_verifies_with_explicit_empty_values(self) -> None:
        self.assertEqual(contract._git_environment()["GIT_ATTR_NOSYSTEM"], "1")
        self.assertEqual(contract._git_environment()["GIT_NO_LAZY_FETCH"], "1")
        self.assertEqual(contract._git_environment()["GIT_PAGER"], "")
        result = self.fixture.verify()
        self.assertEqual(result.candidate_id, contract.CONTRACT_CANDIDATE_ID)
        self.assertEqual(result.status, contract.CONTRACT_STATUS)
        self.assertEqual(result.build_arg_count, len(contract.EXPECTED_BUILD_ARGUMENTS))
        parsed = contract.parse_explicit_build_args(
            self.fixture.explicit_args(), self.fixture.build_args | {
                "SGLANG_BUILD_COMMIT": self.fixture.source_tree
            }
        )
        self.assertEqual(parsed["SGL_VERSION"], "")
        self.assertEqual(parsed["PIP_DEFAULT_INDEX"], "")
        self.assertEqual(parsed["UBUNTU_MIRROR"], "")
        self.assertEqual(parsed["SGLANG_BUILD_URL"], "")

    def test_rejects_unknown_contract_keys(self) -> None:
        self.fixture.write_contract(extra_top_level='surprise = "value"\n')
        self.fixture.commit_contract()
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "unknown surprise"):
            self.fixture.verify()

    def test_rejects_duplicate_toml_keys(self) -> None:
        self.fixture.write_contract(
            extra_top_level='candidate_id = "duplicate-candidate"\n'
        )
        self.fixture.commit_contract()
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "valid SGLang build contract"
        ):
            self.fixture.verify()

    def test_rejects_changed_status_and_broken_source_chain(self) -> None:
        text = self.fixture.contract_path.read_text(encoding="utf-8")
        self.fixture.contract_path.write_text(
            text.replace(
                'status = "source_and_build_invocation_only"',
                'status = "runtime_admitted"',
            ),
            encoding="utf-8",
        )
        self.fixture.commit_contract()
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "status"):
            self.fixture.verify()

        self.fixture.cleanup()
        self.fixture = SyntheticContract()
        text = self.fixture.contract_path.read_text(encoding="utf-8")
        self.fixture.contract_path.write_text(
            text.replace(
                f'output_tree = "{self.fixture.source_change_tree}"',
                f'output_tree = "{"f" * 40}"',
                1,
            ),
            encoding="utf-8",
        )
        self.fixture.commit_contract()
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "input_tree"):
            self.fixture.verify()

    def test_rejects_unknown_missing_duplicate_and_inherited_build_args(self) -> None:
        values = self.fixture.explicit_args()
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "missing CUDA_VERSION"):
            self.fixture.verify(
                build_args=[item for item in values if not item.startswith("CUDA_VERSION=")]
            )
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "unknown EXTRA"):
            self.fixture.verify(build_args=[*values, "EXTRA=value"])
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "duplicate"):
            self.fixture.verify(build_args=[*values, "CUDA_VERSION=13.0.3"])
        inherited = [
            "CUDA_VERSION" if item.startswith("CUDA_VERSION=") else item
            for item in values
        ]
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "inherited"):
            self.fixture.verify(build_args=inherited)

    def test_rejects_mismatched_target_platform_and_argument_value(self) -> None:
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "target"):
            self.fixture.verify(target="framework_final")
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "platform"):
            self.fixture.verify(platform="linux/amd64")
        values = [
            "BRANCH_TYPE=remote" if item == "BRANCH_TYPE=local" else item
            for item in self.fixture.explicit_args()
        ]
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "mismatched BRANCH_TYPE"):
            self.fixture.verify(build_args=values)

    def test_rejects_dirty_untracked_and_ignored_source_files(self) -> None:
        (self.fixture.source_root / "untracked.txt").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "clean working tree"):
            self.fixture.verify()
        (self.fixture.source_root / "untracked.txt").unlink()
        (self.fixture.source_root / "ignored.tmp").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "ignored files"):
            self.fixture.verify()

    def test_rejects_wrong_head_tree(self) -> None:
        (self.fixture.source_root / "tracked.txt").write_text("changed", encoding="utf-8")
        _commit(self.fixture.source_root, "change source tree")
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "HEAD tree"):
            self.fixture.verify()

    def test_rejects_base_commit_tree_mismatch(self) -> None:
        text = self.fixture.contract_path.read_text(encoding="utf-8")
        text = text.replace(
            f'base_tree = "{self.fixture.base_tree}"',
            f'base_tree = "{self.fixture.source_tree}"',
        ).replace(
            f'input_tree = "{self.fixture.base_tree}"',
            f'input_tree = "{self.fixture.source_tree}"',
            1,
        )
        self.fixture.contract_path.write_text(text, encoding="utf-8")
        self.fixture.commit_contract()
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "base_commit"):
            self.fixture.verify()

    def test_rejects_source_step_that_does_not_replay(self) -> None:
        final_commit = _run_git(
            self.fixture.source_root, "rev-parse", "HEAD"
        ).decode("ascii").strip()
        text = self.fixture.contract_path.read_text(encoding="utf-8")
        self.fixture.contract_path.write_text(
            text.replace(self.fixture.source_change_commit, final_commit, 1),
            encoding="utf-8",
        )
        self.fixture.commit_contract()
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError,
            r"replaying source.steps\[0\]",
        ):
            self.fixture.verify()

    def test_rejects_unsafe_source_index_flags(self) -> None:
        _run_git(
            self.fixture.source_root,
            "update-index",
            "--assume-unchanged",
            "base.txt",
        )
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "index flags"):
            self.fixture.verify()

    def test_rejects_source_gitlinks_before_status(self) -> None:
        commit = _run_git(
            self.fixture.source_root, "rev-parse", "HEAD"
        ).decode("ascii").strip()
        _run_git(
            self.fixture.source_root,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{commit},synthetic-submodule",
        )
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "submodules or gitlinks"
        ):
            self.fixture.verify()

    def test_rejects_clean_filters_before_source_status(self) -> None:
        attributes = self.fixture.source_root / ".git" / "info" / "attributes"
        attributes.write_text("base.txt filter=auditnormalize\n", encoding="utf-8")
        calls: list[tuple[Path, tuple[str, ...]]] = []
        original = contract._git

        def recording_git(
            root: Path,
            arguments: list[str] | tuple[str, ...],
            purpose: str,
            *,
            input_bytes: bytes | None = None,
            extra_environment: dict[str, str] | None = None,
        ) -> bytes:
            calls.append((root.resolve(), tuple(arguments)))
            return original(
                root,
                arguments,
                purpose,
                input_bytes=input_bytes,
                extra_environment=extra_environment,
            )

        with patch.object(contract, "_git", side_effect=recording_git):
            with self.assertRaisesRegex(
                contract.SGLangBuildContractError, "clean-filter attributes"
            ):
                self.fixture.verify()
        self.assertFalse(
            any(
                root == self.fixture.source_root.resolve()
                and arguments[0] == "status"
                for root, arguments in calls
            )
        )

    def test_rejects_archive_attributes_including_directories(self) -> None:
        attributes = self.fixture.source_root / ".git" / "info" / "attributes"
        attributes.write_text("python export-ignore\n", encoding="utf-8")
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "archive-control attributes"
        ):
            self.fixture.verify()

    def test_rejects_clean_filters_before_contract_status(self) -> None:
        attributes = (
            self.fixture.repository_root / ".git" / "info" / "attributes"
        )
        attributes.write_text(
            f"{self.fixture.contract_relative} filter=auditnormalize\n",
            encoding="utf-8",
        )
        calls: list[tuple[Path, tuple[str, ...]]] = []
        original = contract._git

        def recording_git(
            root: Path,
            arguments: list[str] | tuple[str, ...],
            purpose: str,
            *,
            input_bytes: bytes | None = None,
            extra_environment: dict[str, str] | None = None,
        ) -> bytes:
            calls.append((root.resolve(), tuple(arguments)))
            return original(
                root,
                arguments,
                purpose,
                input_bytes=input_bytes,
                extra_environment=extra_environment,
            )

        with patch.object(contract, "_git", side_effect=recording_git):
            with self.assertRaisesRegex(
                contract.SGLangBuildContractError, "clean-filter attributes"
            ):
                self.fixture.verify()
        self.assertFalse(any(arguments[0] == "status" for _, arguments in calls))

    def test_rejects_corrupt_source_object_bytes(self) -> None:
        _corrupt_loose_blob(
            self.fixture.source_root,
            "HEAD:base.txt",
            b"base\n",
            b"evil\n",
        )
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "source repository object integrity"
        ):
            self.fixture.verify()

    def test_rejects_corrupt_contract_object_bytes(self) -> None:
        _corrupt_loose_blob(
            self.fixture.repository_root,
            f"HEAD:{self.fixture.contract_relative}",
            b"schema_version",
            b"xchema_version",
        )
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "contract repository object integrity"
        ):
            self.fixture.verify()

    def test_rejects_worktree_scoped_fsck_bypass(self) -> None:
        _run_git(
            self.fixture.source_root,
            "config",
            "extensions.worktreeConfig",
            "true",
        )
        _run_git(
            self.fixture.source_root,
            "config",
            "--worktree",
            "fsck.missingEmail",
            "ignore",
        )
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "integrity-bypass Git configuration"
        ):
            self.fixture.verify()

    def test_rejects_archive_affecting_git_config(self) -> None:
        _run_git(self.fixture.source_root, "config", "tar.umask", "0777")
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "archive-affecting Git configuration"
        ):
            self.fixture.verify()

    def test_rejects_missing_excluded_qsa_object(self) -> None:
        object_path = (
            self.fixture.source_root
            / ".git"
            / "objects"
            / contract.EXCLUDED_QSA_COMMIT[:2]
            / contract.EXCLUDED_QSA_COMMIT[2:]
        )
        object_path.unlink()
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError,
            "checking excluded QSA commit object",
        ):
            self.fixture.verify()

    def test_rejects_protected_qsa_content_drift(self) -> None:
        expected_sha256 = _sha256(self.fixture.protected_path)
        text = self.fixture.contract_path.read_text(encoding="utf-8")
        self.fixture.contract_path.write_text(
            text.replace(expected_sha256, "e" * 64),
            encoding="utf-8",
        )
        self.fixture.commit_contract()
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "protected_files"):
            self.fixture.verify()

    def test_rejects_protected_qsa_change_from_base(self) -> None:
        for drift, expected in (
            ("added", "base_path"),
            ("changed", "differs from source.base_commit"),
            ("deleted", "protected_files.*path"),
            ("mode", "differs from source.base_commit"),
        ):
            with self.subTest(drift=drift):
                self.fixture.cleanup()
                self.fixture = SyntheticContract(protected_drift=drift)
                with self.assertRaisesRegex(
                    contract.SGLangBuildContractError,
                    expected,
                ):
                    self.fixture.verify()

    def test_rejects_explicitly_excluded_ancestry(self) -> None:
        original = contract._git

        def qsa_ancestry(
            root: Path,
            arguments: list[str] | tuple[str, ...],
            purpose: str,
            *,
            input_bytes: bytes | None = None,
            extra_environment: dict[str, str] | None = None,
        ) -> bytes:
            output = original(
                root,
                arguments,
                purpose,
                input_bytes=input_bytes,
                extra_environment=extra_environment,
            )
            if len(arguments) == 2 and arguments[0] == "rev-list":
                output += (contract.EXCLUDED_QSA_COMMIT + "\n").encode("ascii")
            return output

        with patch.object(contract, "_git", side_effect=qsa_ancestry):
            with self.assertRaisesRegex(contract.SGLangBuildContractError, "excluded commit"):
                self.fixture.verify()

    def test_rejects_changed_excluded_commit_pin(self) -> None:
        text = self.fixture.contract_path.read_text(encoding="utf-8")
        self.fixture.contract_path.write_text(
            text.replace(contract.EXCLUDED_QSA_COMMIT, "e" * 40),
            encoding="utf-8",
        )
        self.fixture.commit_contract()
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "competing QSA"):
            self.fixture.verify()

    def test_rejects_tampered_and_symlinked_patch(self) -> None:
        self.fixture.patch_path.write_text(
            self.fixture.patch_path.read_text(encoding="utf-8") + "# tamper\n",
            encoding="utf-8",
        )
        self.fixture.commit_contract("tamper patch")
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "SHA-256"):
            self.fixture.verify()

        self.fixture.cleanup()
        self.fixture = SyntheticContract()
        target = self.fixture.repository_root / "replacement.patch"
        target.write_text(self.fixture.patch_path.read_text(encoding="utf-8"), encoding="utf-8")
        self.fixture.patch_path.unlink()
        self.fixture.patch_path.symlink_to(target)
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "symlink"):
            self.fixture.verify()

    def test_rejects_nested_contract_repository_root(self) -> None:
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "exact Git repository root"):
            self.fixture.verify(repository_root=self.fixture.repository_root / "patches")

    def test_dockerfile_requires_reachable_references_and_pinned_external_from(self) -> None:
        loaded = contract.load_sglang_build_contract(self.fixture.contract_path)
        source = (self.fixture.source_root / "docker" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        contract._verify_dockerfile(source, loaded, loaded.build_arg_map())
        self.assertIn(
            "SGL_DEEP_GEMM_VERSION",
            contract._expanded_shell_references(
                '"$(sed -n \'s/^[^"]*"value"/x/p\' file)" '
                "${SGL_DEEP_GEMM_VERSION}"
            ),
        )
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "unsupported raw parenthesis"
        ):
            contract._expanded_shell_references(
                '"$( (true); echo \'$HPC_OPS_COMMIT\'; echo ok )"'
            )
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "unsupported shell comment"
        ):
            contract._expanded_shell_references(
                '"$(echo # ${HPC_OPS_COMMIT})"'
            )
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "unsupported case construct"
        ):
            contract._expanded_shell_references(
                '"$(case x in x) echo \'$HPC_OPS_COMMIT\';; esac)"'
            )
        self.assertNotIn(
            "HPC_OPS_COMMIT",
            contract._expanded_shell_references("$$HPC_OPS_COMMIT"),
        )
        self.assertNotIn(
            "HPC_OPS_COMMIT",
            contract._expanded_shell_references("$${HPC_OPS_COMMIT}"),
        )
        self.assertIn(
            "HPC_OPS_COMMIT",
            contract._expanded_shell_references("$$$HPC_OPS_COMMIT"),
        )

        with self.assertRaisesRegex(contract.SGLangBuildContractError, "unapproved external FROM"):
            contract._verify_dockerfile(
                source.replace(
                    "FROM ${CUDA_BASE_IMAGE} AS runtime",
                    "FROM nvidia/cuda:latest AS runtime",
                ),
                loaded,
                loaded.build_arg_map(),
            )
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "recognized reachable reference"
        ):
            contract._verify_dockerfile(
                source.replace("${HPC_OPS_COMMIT}", "unused-hpc-commit"),
                loaded,
                loaded.build_arg_map(),
            )
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "heredoc syntax"
        ):
            contract._verify_dockerfile(
                source + "RUN <<'EOF'\nRUN echo ${HPC_OPS_COMMIT}\nEOF\n",
                loaded,
                loaded.build_arg_map(),
            )
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "non-empty default"):
            contract._verify_dockerfile(
                source.replace(
                    "ARG PIP_DEFAULT_INDEX=",
                    "ARG PIP_DEFAULT_INDEX=https://example.invalid/simple",
                ),
                loaded,
                loaded.build_arg_map(),
            )
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "FROM flags"):
            contract._verify_dockerfile(
                source.replace(
                    "FROM ${CUDA_BASE_IMAGE} AS runtime",
                    "FROM --platform=linux/amd64 ${CUDA_BASE_IMAGE} AS runtime",
                ),
                loaded,
                loaded.build_arg_map(),
            )
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "recognized reachable reference"
        ):
            contract._verify_dockerfile(
                source.replace("${HPC_OPS_COMMIT}", "unused-hpc-commit")
                + "\n# ${HPC_OPS_COMMIT}\n",
                loaded,
                loaded.build_arg_map(),
            )
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "unsupported shell comment"
        ):
            contract._verify_dockerfile(
                source.replace("${HPC_OPS_COMMIT}", "unused-hpc-commit")
                + "\nRUN echo ok;# ${HPC_OPS_COMMIT}\n",
                loaded,
                loaded.build_arg_map(),
            )
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "intentionally omitted"
        ):
            contract._verify_dockerfile(
                source.replace(
                    'ARG MOONCAKE_COMPILE_ARG="-DUSE_HTTP=ON -DUSE_CUDA=ON"',
                    'ARG MOONCAKE_COMPILE_ARG="-DUSE_HTTP=ON -DUSE_CUDA=ON"\n'
                    "WORKDIR ${MOONCAKE_COMPILE_ARG}",
                    1,
                ),
                loaded,
                loaded.build_arg_map(),
            )
        for inert_expression in (
            "${#MOONCAKE_COMPILE_ARG}",
            "${!MOONCAKE_COMPILE_ARG}",
            "MOONCAKE_COMPILE_ARG",
        ):
            with self.subTest(inert_expression=inert_expression):
                with self.assertRaisesRegex(
                    contract.SGLangBuildContractError, "intentionally omitted"
                ):
                    contract._verify_dockerfile(
                        source.replace(
                            'ARG MOONCAKE_COMPILE_ARG="-DUSE_HTTP=ON -DUSE_CUDA=ON"',
                            'ARG MOONCAKE_COMPILE_ARG="-DUSE_HTTP=ON -DUSE_CUDA=ON"\n'
                            f"WORKDIR {inert_expression}",
                            1,
                        ),
                        loaded,
                        loaded.build_arg_map(),
                    )
        self.assertIn(
            "MOONCAKE_COMPILE_ARG",
            contract._lexical_non_arg_references(
                "RUN echo $((MOONCAKE_COMPILE_ARG+1))"
            ),
        )
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "intentionally omitted"
        ):
            contract._verify_dockerfile(
                source.replace(
                    'ARG MOONCAKE_COMPILE_ARG="-DUSE_HTTP=ON -DUSE_CUDA=ON"',
                    'ARG MOONCAKE_COMPILE_ARG="-DUSE_HTTP=ON -DUSE_CUDA=ON"\n'
                    "WORKDIR ${OTHER:-${MOONCAKE_COMPILE_ARG}}",
                    1,
                ),
                loaded,
                loaded.build_arg_map(),
            )
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "recognized reachable reference"
        ):
            contract._verify_dockerfile(
                source.replace("${HPC_OPS_COMMIT}", "'${HPC_OPS_COMMIT}'"),
                loaded,
                loaded.build_arg_map(),
            )
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "recognized reachable reference"
        ):
            contract._verify_dockerfile(
                source.replace("${HPC_OPS_COMMIT}", r"\${HPC_OPS_COMMIT}"),
                loaded,
                loaded.build_arg_map(),
            )
        with self.assertRaisesRegex(
            contract.SGLangBuildContractError, "recognized reachable reference"
        ):
            contract._verify_dockerfile(
                source.replace(
                    "COPY --from=framework_final /sgl-workspace /sgl-workspace",
                    "COPY local.txt /local.txt",
                    1,
                ),
                loaded,
                loaded.build_arg_map(),
            )
        global_base = f"ARG CUDA_BASE_IMAGE={BASE_REFERENCE}"
        with self.assertRaisesRegex(contract.SGLangBuildContractError, "global scope"):
            contract._verify_dockerfile(
                source.replace(
                    global_base + "\nFROM ${CUDA_BASE_IMAGE} AS base",
                    "FROM ${CUDA_BASE_IMAGE} AS base\n" + global_base,
                ),
                loaded,
                loaded.build_arg_map(),
            )

    def test_verifier_mutates_only_isolated_replay_scratch(self) -> None:
        before_source = _run_git(
            self.fixture.source_root, "status", "--porcelain=v1", "--untracked-files=all"
        )
        before_contract = _run_git(
            self.fixture.repository_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        calls: list[tuple[tuple[str, ...], dict[str, str] | None]] = []
        original = contract._git

        def recording_git(
            root: Path,
            arguments: list[str] | tuple[str, ...],
            purpose: str,
            *,
            input_bytes: bytes | None = None,
            extra_environment: dict[str, str] | None = None,
        ) -> bytes:
            calls.append((tuple(arguments), extra_environment))
            return original(
                root,
                arguments,
                purpose,
                input_bytes=input_bytes,
                extra_environment=extra_environment,
            )

        with patch.object(contract, "_git", side_effect=recording_git):
            for ambient_root in (
                self.fixture.source_root,
                self.fixture.repository_root,
            ):
                calls.clear()
                with patch.dict(os.environ, {"TMPDIR": str(ambient_root)}):
                    self.fixture.verify()
                self.assertTrue(calls)
                self.assertLessEqual(
                    {arguments[0] for arguments, _ in calls},
                    {
                        "apply",
                        "cat-file",
                        "check-attr",
                        "config",
                        "diff-tree",
                        "fsck",
                        "ls-files",
                        "ls-tree",
                        "patch-id",
                        "read-tree",
                        "replace",
                        "rev-list",
                        "rev-parse",
                        "status",
                        "write-tree",
                    },
                )
                scratch_commands = {"apply", "read-tree", "write-tree"}
                for arguments, extra_environment in calls:
                    if arguments[0] not in scratch_commands:
                        continue
                    self.assertIsNotNone(extra_environment)
                    assert extra_environment is not None
                    self.assertIn("GIT_INDEX_FILE", extra_environment)
                    self.assertIn("GIT_OBJECT_DIRECTORY", extra_environment)
                    for environment_name in (
                        "GIT_INDEX_FILE",
                        "GIT_OBJECT_DIRECTORY",
                    ):
                        scratch_path = Path(extra_environment[environment_name])
                        self.assertFalse(
                            scratch_path.is_relative_to(self.fixture.source_root)
                        )
                        self.assertFalse(
                            scratch_path.is_relative_to(self.fixture.repository_root)
                        )
                        self.assertFalse(scratch_path.parent.exists())
        self.assertEqual(
            before_source,
            _run_git(
                self.fixture.source_root,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
        )
        self.assertEqual(
            before_contract,
            _run_git(
                self.fixture.repository_root,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
        )

    def test_cli_summary_is_deterministic_and_path_free(self) -> None:
        argv = [
            "--repository-root",
            str(self.fixture.repository_root),
            "--contract",
            self.fixture.contract_relative,
            "--source-root",
            str(self.fixture.source_root),
            "--target",
            "runtime",
            "--platform",
            "linux/arm64",
        ]
        for value in self.fixture.explicit_args():
            argv.extend(["--build-arg", value])
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = contract.main(argv)
        self.assertEqual(result, 0, stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["verified"])
        self.assertNotIn(str(self.fixture.root), stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_tracked_production_contract_and_patch_digests(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        contract_path = (
            repository_root
            / "patches"
            / "sglang"
            / "sm121-storage-build-candidate-v1.toml"
        )
        loaded = contract.load_sglang_build_contract(contract_path)
        self.assertEqual(
            loaded.excluded_commits,
            ("8ef3b3fee34a3b5543b65393dd217ed0362a9273",),
        )
        patch_steps = [step for step in loaded.source_steps if step.kind == "patch"]
        for step in patch_steps:
            assert step.path is not None
            assert step.sha256 is not None
            path = repository_root.joinpath(*Path(step.path).parts)
            self.assertEqual(_sha256(path), step.sha256)
        self.assertEqual(
            _sha256(
                repository_root
                / "patches"
                / "sglang"
                / "bdb62e9f-cuda130-arm64-base-pin.patch"
            ),
            "7de1f4f3ed468c1a4b7ac9e98abbe73b6908095321289d0b5328334608f6df11",
        )


if __name__ == "__main__":
    unittest.main()
