"""Offline verification for the future-only Qwen io_uring seccomp profile.

The verifier only reads pinned repository artifacts.  It neither invokes Docker
nor changes daemon, container, kernel, or filesystem state outside its inputs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tomllib
from typing import Any, Mapping, Sequence


CONTRACT_SCHEMA_VERSION = 1
CONTRACT_KIND = "sparkbench-seccomp-profile-contract"
CONTRACT_CANDIDATE_ID = "qwen38-io-uring-docker-v29.2.1"
CONTRACT_STATUS = "future_only_not_admitted"
DEFAULT_CONTRACT_PATH = (
    "patches/sglang/seccomp/qwen38-io-uring-docker-v29.2.1.toml"
)
BASELINE_PATH = "patches/sglang/seccomp/moby-profiles-seccomp-v0.1.0-default.json"
DERIVED_PATH = "patches/sglang/seccomp/qwen38-io-uring-docker-v29.2.1.json"
BASELINE_SHA256 = "01536f1d1df938ae611eba20d6349e0de7a99b6ecdee1549427a0b01b8301e28"
DERIVED_SHA256 = "1c9c9ffc77260ddc8361f0443bac881348324b00b732d5cfabde61a239ff5b62"
MOBY_PROFILES_COMMIT = "c936cc7b4074219137bc0bee45670f5e4618d462"
DOCKER_ENGINE_COMMIT = "6bc6209b88a7a834c91f77d848e025c79e0227a1"
IO_URING_NAMES = (
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
)
IO_URING_ALLOW_GROUP: dict[str, object] = {
    "names": list(IO_URING_NAMES),
    "action": "SCMP_ACT_ALLOW",
}
CANONICAL_FORMAT = (
    "json.dumps(ensure_ascii=True,sort_keys=True,separators=(',',':'))+newline"
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "candidate_id",
        "status",
        "baseline",
        "engine",
        "derived",
        "extension",
    }
)
_BASELINE_KEYS = frozenset({"repository", "module", "tag", "commit", "path", "sha256"})
_ENGINE_KEYS = frozenset(
    {
        "product",
        "version",
        "source_tag",
        "source_commit",
        "seccomp_module",
        "seccomp_module_version",
    }
)
_DERIVED_KEYS = frozenset({"path", "sha256", "canonical_format"})
_EXTENSION_KEYS = frozenset({"action", "names"})


class SeccompProfileContractError(ValueError):
    """Raised when a seccomp profile contract is unsafe or not exactly pinned."""


@dataclass(frozen=True, slots=True)
class SeccompProfileContract:
    """Pinned source and derivation identity for the future-only profile."""

    candidate_id: str
    status: str
    baseline_sha256: str
    derived_sha256: str
    baseline_path: str
    derived_path: str
    engine_version: str
    engine_source_commit: str
    profiles_source_commit: str


@dataclass(frozen=True, slots=True)
class SeccompProfileVerification:
    """Path-free scalar result of an offline profile integrity check."""

    candidate_id: str
    status: str
    baseline_sha256: str
    derived_sha256: str
    engine_version: str
    engine_source_commit: str
    profiles_source_commit: str

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic, path-free verification summary."""

        return {
            "baseline_sha256": self.baseline_sha256,
            "candidate_id": self.candidate_id,
            "derived_sha256": self.derived_sha256,
            "engine_source_commit": self.engine_source_commit,
            "engine_version": self.engine_version,
            "profiles_source_commit": self.profiles_source_commit,
            "status": self.status,
            "verified": True,
        }


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    if actual == expected:
        return
    if expected - actual:
        raise SeccompProfileContractError(f"{label} has missing required fields")
    raise SeccompProfileContractError(f"{label} has unknown fields")


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise SeccompProfileContractError(f"{label} must be a table")
    return value


def _require_string(value: Mapping[str, Any], key: str, label: str) -> str:
    item = value.get(key)
    if type(item) is not str or not item or "\x00" in item:
        raise SeccompProfileContractError(f"{label}.{key} must be a non-empty string")
    return item


def _require_exact(
    value: Mapping[str, Any], key: str, expected: str | int, label: str
) -> str | int:
    item = value.get(key)
    if type(item) is not type(expected) or item != expected:
        raise SeccompProfileContractError(f"{label}.{key} does not match the pinned contract")
    return item


def _require_relative_path(value: str, label: str) -> str:
    if "\\" in value or "\x00" in value:
        raise SeccompProfileContractError(f"{label} must be a safe relative path")
    pieces = value.split("/")
    if not pieces or any(piece in {"", ".", ".."} for piece in pieces):
        raise SeccompProfileContractError(f"{label} must be a safe relative path")
    if value.startswith("/"):
        raise SeccompProfileContractError(f"{label} must be a safe relative path")
    return value


def _require_names(value: Mapping[str, Any]) -> tuple[str, ...]:
    names = value.get("names")
    if type(names) is not list or tuple(names) != IO_URING_NAMES:
        raise SeccompProfileContractError("extension.names does not match the pinned allowlist")
    if any(type(name) is not str for name in names):
        raise SeccompProfileContractError("extension.names does not match the pinned allowlist")
    return tuple(names)


def _parse_toml(data: bytes) -> Mapping[str, Any]:
    try:
        text = data.decode("utf-8")
        value = tomllib.loads(text)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise SeccompProfileContractError("contract TOML is invalid") from error
    return _require_mapping(value, "contract")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SeccompProfileContractError("JSON object has duplicate keys")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise SeccompProfileContractError("JSON profile contains a non-finite constant")


def strict_json_loads(data: bytes, label: str = "JSON profile") -> Any:
    """Load UTF-8 JSON, rejecting duplicate keys and non-finite constants."""

    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except UnicodeDecodeError as error:
        raise SeccompProfileContractError("JSON profile is not UTF-8 JSON") from error
    except (json.JSONDecodeError, RecursionError) as error:
        raise SeccompProfileContractError("JSON profile is invalid") from error


def _repository_root(path: Path) -> Path:
    try:
        information = path.lstat()
    except OSError as error:
        raise SeccompProfileContractError("repository root is unavailable") from error
    if stat.S_ISLNK(information.st_mode) or not stat.S_ISDIR(information.st_mode):
        raise SeccompProfileContractError("repository root must be a real directory")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise SeccompProfileContractError("repository root is unavailable") from error


def _read_regular_repository_file(root: Path, relative: str, label: str) -> bytes:
    _require_relative_path(relative, label)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise SeccompProfileContractError("secure descriptor traversal is unavailable")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    descriptors: list[int] = []
    try:
        descriptor = os.open(root, flags | directory)
        descriptors.append(descriptor)
        pieces = relative.split("/")
        for index, piece in enumerate(pieces):
            child_flags = flags
            if index != len(pieces) - 1:
                child_flags |= directory
            child = os.open(piece, child_flags, dir_fd=descriptor)
            descriptors.append(child)
            descriptor = child
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode):
            raise SeccompProfileContractError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    except SeccompProfileContractError:
        raise
    except OSError as error:
        raise SeccompProfileContractError(f"{label} is unavailable or unsafe") from error
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _load_contract(root: Path, contract_path: str) -> SeccompProfileContract:
    document = _parse_toml(
        _read_regular_repository_file(root, contract_path, "contract TOML")
    )
    _exact_keys(document, _TOP_LEVEL_KEYS, "contract")
    _require_exact(document, "schema_version", CONTRACT_SCHEMA_VERSION, "contract")
    _require_exact(document, "kind", CONTRACT_KIND, "contract")
    _require_exact(document, "candidate_id", CONTRACT_CANDIDATE_ID, "contract")
    _require_exact(document, "status", CONTRACT_STATUS, "contract")

    baseline = _require_mapping(document.get("baseline"), "baseline")
    engine = _require_mapping(document.get("engine"), "engine")
    derived = _require_mapping(document.get("derived"), "derived")
    extension = _require_mapping(document.get("extension"), "extension")
    _exact_keys(baseline, _BASELINE_KEYS, "baseline")
    _exact_keys(engine, _ENGINE_KEYS, "engine")
    _exact_keys(derived, _DERIVED_KEYS, "derived")
    _exact_keys(extension, _EXTENSION_KEYS, "extension")

    for table, values in (
        (
            baseline,
            {
                "repository": "https://github.com/moby/profiles",
                "module": "github.com/moby/profiles/seccomp",
                "tag": "seccomp/v0.1.0",
                "commit": MOBY_PROFILES_COMMIT,
                "path": BASELINE_PATH,
                "sha256": BASELINE_SHA256,
            },
        ),
        (
            engine,
            {
                "product": "Docker Engine",
                "version": "29.2.1",
                "source_tag": "docker-v29.2.1",
                "source_commit": DOCKER_ENGINE_COMMIT,
                "seccomp_module": "github.com/moby/profiles/seccomp",
                "seccomp_module_version": "v0.1.0",
            },
        ),
        (
            derived,
            {
                "path": DERIVED_PATH,
                "sha256": DERIVED_SHA256,
                "canonical_format": CANONICAL_FORMAT,
            },
        ),
    ):
        for key, expected in values.items():
            _require_exact(table, key, expected, "contract")
    _require_exact(extension, "action", "SCMP_ACT_ALLOW", "extension")
    _require_names(extension)

    return SeccompProfileContract(
        candidate_id=CONTRACT_CANDIDATE_ID,
        status=CONTRACT_STATUS,
        baseline_sha256=BASELINE_SHA256,
        derived_sha256=DERIVED_SHA256,
        baseline_path=_require_relative_path(
            _require_string(baseline, "path", "baseline"), "baseline.path"
        ),
        derived_path=_require_relative_path(
            _require_string(derived, "path", "derived"), "derived.path"
        ),
        engine_version="29.2.1",
        engine_source_commit=DOCKER_ENGINE_COMMIT,
        profiles_source_commit=MOBY_PROFILES_COMMIT,
    )


def load_seccomp_profile_contract(
    repository_root: Path | str,
    contract_path: Path | str = DEFAULT_CONTRACT_PATH,
) -> SeccompProfileContract:
    """Load the exact future-only contract without mutating local state."""

    root = _repository_root(Path(repository_root))
    return _load_contract(root, _require_relative_path(str(contract_path), "contract path"))


def _expected_derived_profile(baseline: object) -> dict[str, object]:
    if type(baseline) is not dict:
        raise SeccompProfileContractError("baseline profile must be an object")
    syscalls = baseline.get("syscalls")
    if type(syscalls) is not list:
        raise SeccompProfileContractError("baseline profile must contain a syscall list")
    for group in syscalls:
        if type(group) is not dict:
            raise SeccompProfileContractError("baseline profile syscall groups are invalid")
        names = group.get("names")
        if type(names) is not list or any(type(name) is not str for name in names):
            raise SeccompProfileContractError("baseline profile syscall groups are invalid")
        if set(names).intersection(IO_URING_NAMES):
            raise SeccompProfileContractError("baseline profile already grants io_uring")
    expected = deepcopy(baseline)
    expected["syscalls"].append(deepcopy(IO_URING_ALLOW_GROUP))
    return expected


def _canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SeccompProfileContractError("profile cannot be canonically encoded") from error
    return (text + "\n").encode("utf-8")


def verify_seccomp_profile_contract(
    repository_root: Path | str,
    contract_path: Path | str = DEFAULT_CONTRACT_PATH,
) -> SeccompProfileVerification:
    """Verify profile bytes and derivation offline, without Docker interaction."""

    root = _repository_root(Path(repository_root))
    contract = _load_contract(
        root, _require_relative_path(str(contract_path), "contract path")
    )
    baseline_bytes = _read_regular_repository_file(
        root, contract.baseline_path, "baseline profile"
    )
    derived_bytes = _read_regular_repository_file(
        root, contract.derived_path, "derived profile"
    )
    baseline = strict_json_loads(baseline_bytes, "baseline profile")
    derived = strict_json_loads(derived_bytes, "derived profile")
    if hashlib.sha256(baseline_bytes).hexdigest() != contract.baseline_sha256:
        raise SeccompProfileContractError("baseline profile digest does not match")
    expected = _expected_derived_profile(baseline)
    if derived != expected:
        raise SeccompProfileContractError(
            "derived profile differs by more than the pinned io_uring allow group"
        )
    expected_bytes = _canonical_json_bytes(expected)
    if derived_bytes != expected_bytes:
        raise SeccompProfileContractError("derived profile is not canonical")
    if hashlib.sha256(derived_bytes).hexdigest() != contract.derived_sha256:
        raise SeccompProfileContractError("derived profile digest does not match")
    return SeccompProfileVerification(
        candidate_id=contract.candidate_id,
        status=contract.status,
        baseline_sha256=contract.baseline_sha256,
        derived_sha256=contract.derived_sha256,
        engine_version=contract.engine_version,
        engine_source_commit=contract.engine_source_commit,
        profiles_source_commit=contract.profiles_source_commit,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only verifier and print a scalar-only result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path(DEFAULT_CONTRACT_PATH))
    arguments = parser.parse_args(argv)
    try:
        result = verify_seccomp_profile_contract(
            arguments.repository_root, arguments.contract
        )
    except SeccompProfileContractError as error:
        print(f"seccomp profile verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result.as_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
