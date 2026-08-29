"""Freeze and audit the minimal offline Pi core source closure.

This module deliberately stops before package installation or Pi execution.
It turns the audited, larger candidate lock into a deterministic *closure
lock* whose only root dependency is @mariozechner/pi-agent-core@0.57.1.  That
closure lock is input for a future tarball-direct normalizer, not evidence
that ``npm ci`` can install it.  This module does not invoke Node, npm, a Pi
CLI, a model server, or the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import base64
import binascii
from collections import deque
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import tarfile
from typing import Any, Mapping
from urllib.parse import urlsplit

from bench.harbor_runtime_assets import RuntimeAssetError, inspect_normalized_tree


PI_CORE_PREFIX_PROTOCOL = "sparkbench-pi-core-source-v1"
PI_CORE_MATERIALIZED_PREFIX_PROTOCOL = "sparkbench-pi-core-prefix-v1"
PI_CORE_PREFIX_NAME = "sparkbench-pi-core-prefix"
PI_CORE_VERSION = "0.57.1"
PI_AGENT_CORE_PACKAGE = "@mariozechner/pi-agent-core"
PI_AI_PACKAGE = "@mariozechner/pi-ai"
PI_AGENT_CORE_INTEGRITY = (
    "sha512-WXsBbkNWOObFGHkhixaT8GXJpHDd3+fn8QntYF+4R8Sa9WB90ENXWidO6b7vc"
    "KX+JX0jjO5dIsQxmzosARJKlg=="
)
PI_AI_INTEGRITY = (
    "sha512-Bd/J4a3YpdzJVyHLih0vDSdB0QPL4ti0XsAwtHOK/8eVhB0fHM1CpcgIrcBFJ2"
    "3TMcKXMi0qamz18ERfp8tmgg=="
)
PI_CORE_CANDIDATE_LOCK_SHA256 = (
    "sha256:efc060b934aab3243b4f06e013b783db70e81f80f4dc36685fde65c09fdc1016"
)
PI_CORE_FROZEN_LOCK_SHA256 = (
    "sha256:1002f0f898df7217500b7bfa838e6ae2e5e9d6b122c8f9dcac79c56c6737323d"
)
PI_CORE_LOCKFILE_VERSION = 3
PI_CORE_CLOSURE_PACKAGE_COUNT = 222
PI_CORE_UNIQUE_ARTIFACT_COUNT = 207
PI_CORE_INSTALL_SCRIPT_PACKAGE_COUNT = 1
PI_CORE_OPTIONAL_PACKAGE_COUNT = 1
PI_CORE_PREFIX_DIRECTORY_NAME = "pi-core-0.57.1-1002f0f898df"
PI_CORE_PATH_LIST_SHA256 = (
    "sha256:ebe369bc873ebab868810aad5b728b3f3a900bad215d13d7f6f2cc93c8c660f0"
)
PI_CORE_PATH_VERSION_INTEGRITY_SHA256 = (
    "sha256:ea4489fd826db97b06da628392900a210c90a44d9e4efc92f7d30019c3cab2fc"
)
MAX_LOCK_BYTES = 32 * 1024 * 1024
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_MATERIALIZATION_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_TOTAL_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
MAX_CLOSURE_PACKAGES = 1_024
MAX_ARCHIVE_MEMBERS = 100_000
MAX_ARCHIVE_PATH_METADATA_BYTES = 16 * 1024 * 1024
MAX_PREFIX_ENTRIES = 100_000
MAX_PREFIX_FILE_BYTES = 512 * 1024 * 1024
MAX_PREFIX_BYTES = 4 * 1024 * 1024 * 1024
MAX_PACKAGE_MANIFEST_BYTES = 2 * 1024 * 1024

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PACKAGE_COMPONENT_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_OUTPUT_ENTRY_KEYS = frozenset(
    {
        "dependencies",
        "hasInstallScript",
        "integrity",
        "optional",
        "optionalDependencies",
        "peerDependencies",
        "peerDependenciesMeta",
        "resolved",
        "version",
    }
)
_SOURCE_ENTRY_KEYS = _OUTPUT_ENTRY_KEYS | {
    "bin",
    "deprecated",
    "engines",
    "funding",
    "license",
    "name",
}


class PiCorePrefixError(RuntimeError):
    """Raised when the frozen Pi core source boundary is unsafe or incomplete."""


@dataclass(frozen=True, slots=True)
class PiCoreLockSummary:
    """Scalar identity for the minimal Pi core package-lock closure."""

    protocol: str
    frozen_lock_sha256: str
    package_count: int
    integrity_count: int
    unique_artifact_count: int
    install_script_package_count: int
    optional_package_count: int
    document: dict[str, Any] = field(repr=False, compare=False)

    def scalar(
        self, *, status: str, origin_lock_sha256: str | None = None
    ) -> dict[str, object]:
        """Return the public-safe scalar projection without paths or package names."""

        result: dict[str, object] = {
            "protocol": self.protocol,
            "status": status,
            "frozen_lock_sha256": self.frozen_lock_sha256,
            "package_count": self.package_count,
            "integrity_count": self.integrity_count,
            "unique_artifact_count": self.unique_artifact_count,
            "install_script_package_count": self.install_script_package_count,
            "optional_package_count": self.optional_package_count,
        }
        if origin_lock_sha256 is not None:
            result["origin_lock_sha256"] = origin_lock_sha256
        return result


@dataclass(frozen=True, slots=True)
class PiCoreCacheAudit:
    """Scalar-only read-only validation of all package tarball blobs."""

    protocol: str
    frozen_lock_sha256: str
    package_count: int
    artifact_count: int
    artifact_size_bytes: int

    def scalar(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "status": "cache_complete",
            "frozen_lock_sha256": self.frozen_lock_sha256,
            "package_count": self.package_count,
            "artifact_count": self.artifact_count,
            "artifact_size_bytes": self.artifact_size_bytes,
        }


@dataclass(frozen=True, slots=True)
class PiCorePrefixMaterialization:
    """Scalar identity for a newly created, scripts-disabled Pi prefix."""

    protocol: str
    frozen_lock_sha256: str
    package_count: int
    artifact_count: int
    artifact_size_bytes: int
    tree_digest: str
    tree_entries: int
    tree_files: int
    tree_size_bytes: int
    prefix_directory_name: str
    resolved_path: Path = field(repr=False, compare=False)

    def scalar(self) -> dict[str, object]:
        """Return the public-safe result without a local filesystem path."""

        return {
            "protocol": self.protocol,
            "status": "materialized",
            "frozen_lock_sha256": self.frozen_lock_sha256,
            "package_count": self.package_count,
            "artifact_count": self.artifact_count,
            "artifact_size_bytes": self.artifact_size_bytes,
            "tree_digest": self.tree_digest,
            "tree_entries": self.tree_entries,
            "tree_files": self.tree_files,
            "tree_size_bytes": self.tree_size_bytes,
            "prefix_directory_name": self.prefix_directory_name,
        }


def _same_file(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_uid == after.st_uid
        and before.st_mode == after.st_mode
        and before.st_nlink == after.st_nlink
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
    )


def _secure_open_flags(base: int) -> int:
    """Return file-open flags that fail closed if final-symlink protection lacks."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise PiCorePrefixError("Pi core closure requires O_NOFOLLOW support")
    return base | getattr(os, "O_CLOEXEC", 0) | no_follow


def _read_regular_file(
    path: Path,
    *,
    maximum: int,
    context: str,
    require_owner_private: bool,
) -> bytes:
    """Read one bounded regular file without following its final symlink.

    The externally supplied candidate lock is authenticated by its pinned
    digest, so it need not be owned or mode-private.  A tracked frozen lock is
    a local policy object and is therefore held to the stronger ownership and
    permissions rule.
    """

    try:
        before = os.lstat(path)
    except OSError as error:
        raise PiCorePrefixError(f"{context} cannot be inspected") from error
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
        raise PiCorePrefixError(f"{context} is unsafe")
    if require_owner_private and (
        before.st_nlink != 1
        or before.st_uid != os.geteuid()
        or before.st_mode & 0o022
    ):
        raise PiCorePrefixError(f"{context} is unsafe")
    flags = _secure_open_flags(os.O_RDONLY)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PiCorePrefixError(f"{context} cannot be opened") from error
    blocks: list[bytes] = []
    remaining = maximum + 1
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file(before, opened):
            raise PiCorePrefixError(f"{context} changed while opening")
        while remaining > 0:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                break
            blocks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    payload = b"".join(blocks)
    if (
        len(payload) > maximum
        or len(payload) != opened.st_size
        or not _same_file(opened, after)
    ):
        raise PiCorePrefixError(f"{context} changed while reading")
    return payload


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> object:
    raise ValueError("non-finite JSON value")


def _load_json_object(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise PiCorePrefixError(f"{context} is not strict JSON") from error
    if type(value) is not dict:
        raise PiCorePrefixError(f"{context} is not an object")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise PiCorePrefixError("Pi core source lock is not serializable") from error
    return (encoded + "\n").encode("ascii")


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _package_name_parts(value: object, *, context: str) -> tuple[str, ...]:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or "\x00" in value
    ):
        raise PiCorePrefixError(f"{context} is invalid")
    parts = value.split("/")
    if value.startswith("@"):
        valid = (
            len(parts) == 2
            and parts[0].startswith("@")
            and _PACKAGE_COMPONENT_PATTERN.fullmatch(parts[0][1:]) is not None
            and _PACKAGE_COMPONENT_PATTERN.fullmatch(parts[1]) is not None
        )
    else:
        valid = (
            len(parts) == 1
            and _PACKAGE_COMPONENT_PATTERN.fullmatch(parts[0]) is not None
        )
    if not valid:
        raise PiCorePrefixError(f"{context} is invalid")
    if "node_modules" in parts:
        raise PiCorePrefixError(f"{context} is invalid")
    return tuple(parts)


def _package_name_from_path(path: object) -> str:
    if not isinstance(path, str) or not path.startswith("node_modules/"):
        raise PiCorePrefixError("Pi core lock package path is invalid")
    parts = path.split("/")
    index = 0
    last_name: tuple[str, ...] | None = None
    while index < len(parts):
        if parts[index] != "node_modules":
            raise PiCorePrefixError("Pi core lock package path is invalid")
        index += 1
        if index >= len(parts):
            raise PiCorePrefixError("Pi core lock package path is invalid")
        if parts[index].startswith("@"):
            if index + 1 >= len(parts):
                raise PiCorePrefixError("Pi core lock package path is invalid")
            candidate = "/".join(parts[index : index + 2])
            index += 2
        else:
            candidate = parts[index]
            index += 1
        last_name = _package_name_parts(candidate, context="Pi core lock package name")
    if last_name is None:
        raise PiCorePrefixError("Pi core lock package path is invalid")
    return "/".join(last_name)


def _dependency_map(
    entry: Mapping[str, object], key: str, *, context: str
) -> dict[str, str]:
    value = entry.get(key, {})
    if value is None:
        return {}
    if type(value) is not dict:
        raise PiCorePrefixError(f"{context}.{key} is invalid")
    result: dict[str, str] = {}
    for name, specification in value.items():
        _package_name_parts(name, context=f"{context}.{key} package")
        if not isinstance(specification, str):
            raise PiCorePrefixError(f"{context}.{key} specification is invalid")
        normalized_specification = specification.casefold()
        if (
            not specification
            or specification.strip() != specification
            or any(
                ord(character) < 0x20
                or 0x7F <= ord(character) <= 0x9F
                or 0xD800 <= ord(character) <= 0xDFFF
                for character in specification
            )
            or normalized_specification.startswith(
                (
                    "/",
                    "./",
                    "../",
                    "file:",
                    "git:",
                    "git+",
                    "github:",
                    "http:",
                    "https:",
                    "link:",
                    "workspace:",
                    "ssh:",
                )
            )
        ):
            raise PiCorePrefixError(f"{context}.{key} specification is invalid")
        result[name] = specification
    return result


def _optional_peer_names(entry: Mapping[str, object], *, context: str) -> frozenset[str]:
    peer_dependencies = _dependency_map(entry, "peerDependencies", context=context)
    metadata = entry.get("peerDependenciesMeta", {})
    if metadata is None:
        return frozenset()
    if type(metadata) is not dict:
        raise PiCorePrefixError(f"{context}.peerDependenciesMeta is invalid")
    optional: set[str] = set()
    for name, value in metadata.items():
        _package_name_parts(name, context=f"{context}.peerDependenciesMeta package")
        if type(value) is not dict:
            raise PiCorePrefixError(f"{context}.peerDependenciesMeta is invalid")
        if frozenset(value) != {"optional"} or type(value["optional"]) is not bool:
            raise PiCorePrefixError(f"{context}.peerDependenciesMeta is invalid")
        # npm lockfiles can retain optional-peer metadata after the matching
        # peer declaration disappears.  It carries no resolution edge, but
        # remains part of the lock entry and must be retained verbatim.
        if value["optional"] and name in peer_dependencies:
            optional.add(name)
    return frozenset(optional)


def _resolve_dependency(
    packages: Mapping[str, object], requester: str, dependency: str
) -> str | None:
    _package_name_parts(dependency, context="Pi core lock dependency")
    current = requester
    while current:
        candidate = current + "/node_modules/" + dependency
        if candidate in packages:
            return candidate
        components = current.split("/")
        try:
            package_marker = len(components) - 1 - components[::-1].index(
                "node_modules"
            )
        except ValueError as error:
            raise PiCorePrefixError("Pi core lock package path is invalid") from error
        parent_components = components[:package_marker]
        current = "/".join(parent_components)
    candidate = "node_modules/" + dependency
    return candidate if candidate in packages else None


def _parse_integrity(value: object, *, context: str) -> bytes:
    if not isinstance(value, str) or not value.startswith("sha512-"):
        raise PiCorePrefixError(f"{context} is invalid")
    try:
        digest = base64.b64decode(value.removeprefix("sha512-"), validate=True)
    except (ValueError, binascii.Error) as error:
        raise PiCorePrefixError(f"{context} is invalid") from error
    if len(digest) != hashlib.sha512().digest_size:
        raise PiCorePrefixError(f"{context} is invalid")
    return digest


def _validate_resolved_url(value: object, *, context: str) -> str:
    if not isinstance(value, str) or len(value) > 2_048 or "\x00" in value:
        raise PiCorePrefixError(f"{context} is invalid")
    try:
        parsed = urlsplit(value)
    except ValueError as error:
        raise PiCorePrefixError(f"{context} is invalid") from error
    if (
        parsed.scheme != "https"
        or parsed.hostname != "registry.npmjs.org"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or not parsed.path.startswith("/")
        or not parsed.path.endswith(".tgz")
        or parsed.query
        or parsed.fragment
    ):
        raise PiCorePrefixError(f"{context} is invalid")
    return value


def _validate_package_entry(
    path: str, value: object, *, allow_source_metadata: bool
) -> dict[str, Any]:
    _package_name_from_path(path)
    allowed_keys = _SOURCE_ENTRY_KEYS if allow_source_metadata else _OUTPUT_ENTRY_KEYS
    if type(value) is not dict or not set(value).issubset(allowed_keys):
        raise PiCorePrefixError("Pi core lock package entry is invalid")
    version = value.get("version")
    if (
        not isinstance(version, str)
        or not version
        or len(version) > 128
        or "\x00" in version
        or "\n" in version
        or "\r" in version
    ):
        raise PiCorePrefixError("Pi core lock package entry is invalid")
    _validate_resolved_url(value.get("resolved"), context="Pi core lock resolved URL")
    _parse_integrity(value.get("integrity"), context="Pi core lock integrity")
    for key in ("dependencies", "optionalDependencies", "peerDependencies"):
        _dependency_map(value, key, context="Pi core lock package entry")
    _optional_peer_names(value, context="Pi core lock package entry")
    for key in ("optional", "hasInstallScript"):
        if key in value and type(value[key]) is not bool:
            raise PiCorePrefixError("Pi core lock package entry is invalid")
    return dict(value)


def _closure_paths(
    packages: Mapping[str, object], *, allow_source_metadata: bool
) -> frozenset[str]:
    start = "node_modules/" + PI_AGENT_CORE_PACKAGE
    if start not in packages:
        raise PiCorePrefixError("Pi core lock has no pinned agent-core package")
    pending = deque([start])
    selected: set[str] = set()
    while pending:
        path = pending.popleft()
        if path in selected:
            continue
        if len(selected) >= MAX_CLOSURE_PACKAGES:
            raise PiCorePrefixError("Pi core lock closure exceeds its bound")
        entry = _validate_package_entry(
            path, packages[path], allow_source_metadata=allow_source_metadata
        )
        selected.add(path)
        optional_peers = _optional_peer_names(
            entry, context="Pi core lock package entry"
        )
        seen_dependencies: set[str] = set()
        for key in ("dependencies", "optionalDependencies", "peerDependencies"):
            dependency_map = _dependency_map(
                entry, key, context="Pi core lock package entry"
            )
            for dependency in sorted(dependency_map):
                if dependency in seen_dependencies:
                    continue
                seen_dependencies.add(dependency)
                if key == "peerDependencies" and dependency in optional_peers:
                    continue
                target = _resolve_dependency(packages, path, dependency)
                if target is None:
                    raise PiCorePrefixError("Pi core lock has an unresolved dependency")
                pending.append(target)
    return frozenset(selected)


def _normalized_package_entry(value: Mapping[str, object]) -> dict[str, object]:
    return {key: value[key] for key in sorted(_OUTPUT_ENTRY_KEYS) if key in value}


def _root_package_entry() -> dict[str, object]:
    return {
        "dependencies": {PI_AGENT_CORE_PACKAGE: PI_CORE_VERSION},
        "name": PI_CORE_PREFIX_NAME,
        "version": PI_CORE_VERSION,
    }


def _frozen_document_from_packages(
    packages: Mapping[str, object],
) -> dict[str, Any]:
    selected = _closure_paths(packages, allow_source_metadata=True)
    result_packages: dict[str, object] = {"": _root_package_entry()}
    for path in sorted(selected):
        entry = _validate_package_entry(
            path, packages[path], allow_source_metadata=True
        )
        result_packages[path] = _normalized_package_entry(entry)
    return {
        "lockfileVersion": PI_CORE_LOCKFILE_VERSION,
        "name": PI_CORE_PREFIX_NAME,
        "packages": result_packages,
        "requires": True,
        "version": PI_CORE_VERSION,
    }


def _validate_pinned_core_and_ai(
    packages: Mapping[str, object], *, context: str
) -> None:
    core_path = "node_modules/" + PI_AGENT_CORE_PACKAGE
    ai_path = "node_modules/" + PI_AI_PACKAGE
    core = packages.get(core_path)
    ai = packages.get(ai_path)
    if type(core) is not dict or type(ai) is not dict:
        raise PiCorePrefixError(f"{context} is incomplete")
    if (
        core.get("version") != PI_CORE_VERSION
        or core.get("integrity") != PI_AGENT_CORE_INTEGRITY
        or ai.get("version") != PI_CORE_VERSION
        or ai.get("integrity") != PI_AI_INTEGRITY
    ):
        raise PiCorePrefixError(f"{context} changed")
    core_dependencies = _dependency_map(core, "dependencies", context=context)
    if core_dependencies != {PI_AI_PACKAGE: "^0.57.1"}:
        raise PiCorePrefixError(f"{context} changed")


def _validate_selection_anchors(
    packages: Mapping[str, object], closure: frozenset[str]
) -> None:
    paths = sorted(closure)
    path_payload = ("\n".join(paths) + "\n").encode("utf-8")
    records: list[str] = []
    for path in paths:
        entry = packages[path]
        if type(entry) is not dict:
            raise PiCorePrefixError("Pi core lock package entry is invalid")
        version = entry.get("version")
        integrity = entry.get("integrity")
        if not isinstance(version, str) or not isinstance(integrity, str):
            raise PiCorePrefixError("Pi core lock package entry is invalid")
        records.append(f"{path}\t{version}\t{integrity}")
    record_payload = ("\n".join(records) + "\n").encode("utf-8")
    if (
        _sha256(path_payload) != PI_CORE_PATH_LIST_SHA256
        or _sha256(record_payload) != PI_CORE_PATH_VERSION_INTEGRITY_SHA256
    ):
        raise PiCorePrefixError("Pi core lock selection changed")


def _summary(document: dict[str, Any]) -> PiCoreLockSummary:
    packages_value = document.get("packages")
    if type(packages_value) is not dict:
        raise PiCorePrefixError("Pi core lock packages are invalid")
    packages: dict[str, object] = dict(packages_value)
    root = packages.pop("", None)
    if root != _root_package_entry():
        raise PiCorePrefixError("Pi core lock root package changed")
    _validate_pinned_core_and_ai(packages, context="Pi core lock")
    closure = _closure_paths(packages, allow_source_metadata=False)
    if closure != frozenset(packages):
        raise PiCorePrefixError("Pi core lock has unexpected package entries")
    _validate_selection_anchors(packages, closure)
    package_count = len(closure)
    integrities: set[bytes] = set()
    install_scripts = 0
    optional_packages = 0
    for path in closure:
        entry = _validate_package_entry(
            path, packages[path], allow_source_metadata=False
        )
        integrities.add(
            _parse_integrity(entry["integrity"], context="Pi core lock integrity")
        )
        install_scripts += int(entry.get("hasInstallScript") is True)
        optional_packages += int(entry.get("optional") is True)
    if (
        package_count != PI_CORE_CLOSURE_PACKAGE_COUNT
        or len(integrities) != PI_CORE_UNIQUE_ARTIFACT_COUNT
        or install_scripts != PI_CORE_INSTALL_SCRIPT_PACKAGE_COUNT
        or optional_packages != PI_CORE_OPTIONAL_PACKAGE_COUNT
    ):
        raise PiCorePrefixError("Pi core lock closure changed")
    encoded = _canonical_json_bytes(document)
    frozen_lock_sha256 = _sha256(encoded)
    if frozen_lock_sha256 != PI_CORE_FROZEN_LOCK_SHA256:
        raise PiCorePrefixError("Pi core frozen lock identity changed")
    return PiCoreLockSummary(
        protocol=PI_CORE_PREFIX_PROTOCOL,
        frozen_lock_sha256=frozen_lock_sha256,
        package_count=package_count,
        integrity_count=package_count,
        unique_artifact_count=len(integrities),
        install_script_package_count=install_scripts,
        optional_package_count=optional_packages,
        document=document,
    )


def freeze_pinned_pi_core_lock(source_path: Path) -> PiCoreLockSummary:
    """Derive the exact 222-package Pi core lock from the audited source lock."""

    source_bytes = _read_regular_file(
        source_path,
        maximum=MAX_LOCK_BYTES,
        context="Pi core candidate lock",
        require_owner_private=False,
    )
    if _sha256(source_bytes) != PI_CORE_CANDIDATE_LOCK_SHA256:
        raise PiCorePrefixError("Pi core candidate lock identity changed")
    source = _load_json_object(source_bytes, context="Pi core candidate lock")
    if (
        source.get("lockfileVersion") != PI_CORE_LOCKFILE_VERSION
        or source.get("requires") is not True
        or type(source.get("packages")) is not dict
    ):
        raise PiCorePrefixError("Pi core candidate lock is not npm lockfile v3")
    document = _frozen_document_from_packages(source["packages"])
    return _summary(document)


def load_frozen_pi_core_lock(path: Path) -> PiCoreLockSummary:
    """Load and validate a tracked, minimal Pi core package-lock."""

    raw = _read_regular_file(
        path,
        maximum=MAX_LOCK_BYTES,
        context="Pi core frozen lock",
        require_owner_private=True,
    )
    document = _load_json_object(raw, context="Pi core frozen lock")
    if frozenset(document) != {
        "lockfileVersion",
        "name",
        "packages",
        "requires",
        "version",
    }:
        raise PiCorePrefixError("Pi core frozen lock fields changed")
    if (
        document.get("lockfileVersion") != PI_CORE_LOCKFILE_VERSION
        or document.get("name") != PI_CORE_PREFIX_NAME
        or document.get("version") != PI_CORE_VERSION
        or document.get("requires") is not True
    ):
        raise PiCorePrefixError("Pi core frozen lock identity changed")
    summary = _summary(document)
    if raw != _canonical_json_bytes(document):
        raise PiCorePrefixError("Pi core frozen lock is not canonical")
    return summary


def _validated_summary(summary: PiCoreLockSummary) -> PiCoreLockSummary:
    """Reject a forged or subsequently mutated public summary object."""

    if type(summary) is not PiCoreLockSummary or type(summary.document) is not dict:
        raise PiCorePrefixError("Pi core frozen lock summary is invalid")
    canonical = _canonical_json_bytes(summary.document)
    document = _load_json_object(canonical, context="Pi core frozen lock summary")
    revalidated = _summary(document)
    if revalidated != summary:
        raise PiCorePrefixError("Pi core frozen lock summary changed")
    return revalidated


def _open_owned_private_directory(path: Path, *, context: str) -> int:
    """Open a verified output directory without following its final symlink."""

    try:
        before = os.lstat(path)
    except OSError as error:
        raise PiCorePrefixError(f"{context} is unavailable") from error
    if (
        not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_mode & 0o022
    ):
        raise PiCorePrefixError(f"{context} is unsafe")
    flags = _secure_open_flags(os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PiCorePrefixError(f"{context} cannot be opened") from error
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode) or not _same_file(before, opened):
        os.close(descriptor)
        raise PiCorePrefixError(f"{context} changed while opening")
    return descriptor


def _new_private_temp_directory(parent_descriptor: int) -> tuple[str, int]:
    flags = _secure_open_flags(os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    for _attempt in range(32):
        name = ".sparkbench-pi-core-" + secrets.token_hex(16)
        try:
            os.mkdir(name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        except OSError as error:
            raise PiCorePrefixError("Pi core frozen lock cannot stage") from error
        try:
            descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        except OSError as error:
            try:
                os.rmdir(name, dir_fd=parent_descriptor)
            except OSError:
                pass
            raise PiCorePrefixError("Pi core frozen lock cannot stage") from error
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
        ):
            os.close(descriptor)
            try:
                os.rmdir(name, dir_fd=parent_descriptor)
            except OSError:
                pass
            raise PiCorePrefixError("Pi core frozen lock staging directory is unsafe")
        return name, descriptor
    raise PiCorePrefixError("Pi core frozen lock staging name collision")


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write one in-memory block fully without changing durability semantics."""

    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise PiCorePrefixError("Pi core write made no progress")
        offset += written


def _write_descriptor(descriptor: int, payload: bytes) -> None:
    _write_all(descriptor, payload)
    os.fsync(descriptor)


def _cleanup_temp_directory(
    parent_descriptor: int, directory_name: str, directory_descriptor: int
) -> None:
    """Best-effort cleanup after all file handles within a private dir close."""

    os.close(directory_descriptor)
    try:
        os.rmdir(directory_name, dir_fd=parent_descriptor)
    except OSError:
        # A stale, inaccessible staging directory is safer than removing an
        # object after an unexpected filesystem race.
        pass


def write_new_frozen_pi_core_lock(path: Path, summary: PiCoreLockSummary) -> None:
    """Atomically publish one canonical lock without replacing any target."""

    summary = _validated_summary(summary)
    if path.name in {"", ".", ".."}:
        raise PiCorePrefixError("Pi core frozen lock destination is invalid")
    parent_descriptor = _open_owned_private_directory(
        path.parent, context="Pi core frozen lock destination"
    )
    staging_name: str | None = None
    staging_descriptor: int | None = None
    file_descriptor: int | None = None
    file_name = "lock.json"
    try:
        staging_name, staging_descriptor = _new_private_temp_directory(parent_descriptor)
        flags = _secure_open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            file_descriptor = os.open(file_name, flags, 0o644, dir_fd=staging_descriptor)
        except OSError as error:
            raise PiCorePrefixError("Pi core frozen lock cannot stage") from error
        os.fchmod(file_descriptor, 0o644)
        _write_descriptor(file_descriptor, _canonical_json_bytes(summary.document))
        written = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(written.st_mode)
            or written.st_uid != os.geteuid()
            or written.st_nlink != 1
            or written.st_mode & 0o022
        ):
            raise PiCorePrefixError("Pi core frozen lock staging file is unsafe")
        os.close(file_descriptor)
        file_descriptor = None
        try:
            os.link(
                file_name,
                path.name,
                src_dir_fd=staging_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise PiCorePrefixError(
                "Pi core frozen lock destination already exists"
            ) from error
        except OSError as error:
            raise PiCorePrefixError("Pi core frozen lock cannot be published") from error
        os.fsync(parent_descriptor)
        os.unlink(file_name, dir_fd=staging_descriptor)
        os.fsync(staging_descriptor)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if staging_name is not None and staging_descriptor is not None:
            try:
                os.unlink(file_name, dir_fd=staging_descriptor)
            except FileNotFoundError:
                pass
            except OSError:
                pass
            _cleanup_temp_directory(
                parent_descriptor, staging_name, staging_descriptor
            )
        os.close(parent_descriptor)


def _cache_blob_path(cache_sha512_root: Path, integrity: object) -> Path:
    digest = _parse_integrity(integrity, context="Pi core cache integrity").hex()
    return cache_sha512_root / digest[:2] / digest[2:4] / digest[4:]


def _hash_cache_blob(path: Path) -> tuple[int, bytes]:
    """Hash one content-addressed npm blob through a no-follow descriptor."""

    try:
        before = os.lstat(path)
    except OSError as error:
        raise PiCorePrefixError("Pi core cache blob is unavailable") from error
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_ARTIFACT_BYTES:
        raise PiCorePrefixError("Pi core cache blob is unsafe")
    flags = _secure_open_flags(os.O_RDONLY)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PiCorePrefixError("Pi core cache blob cannot be opened") from error
    digest = hashlib.sha512()
    total = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file(before, opened):
            raise PiCorePrefixError("Pi core cache blob changed while opening")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > MAX_ARTIFACT_BYTES:
                raise PiCorePrefixError("Pi core cache blob exceeds its bound")
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if total != opened.st_size or not _same_file(opened, after):
        raise PiCorePrefixError("Pi core cache blob changed while hashing")
    return total, digest.digest()


def audit_pi_core_cache(
    summary: PiCoreLockSummary, *, cache_sha512_root: Path
) -> PiCoreCacheAudit:
    """Read-only validate every minimal-closure tarball against its SRI digest."""

    summary = _validated_summary(summary)
    try:
        root_metadata = os.lstat(cache_sha512_root)
    except OSError as error:
        raise PiCorePrefixError("Pi core npm content store is unavailable") from error
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise PiCorePrefixError("Pi core npm content store is unsafe")
    packages = summary.document["packages"]
    if type(packages) is not dict:
        raise PiCorePrefixError("Pi core lock packages are invalid")
    artifacts: dict[bytes, Path] = {}
    for path, entry in packages.items():
        if path == "":
            continue
        if type(entry) is not dict:
            raise PiCorePrefixError("Pi core lock package entry is invalid")
        integrity = entry.get("integrity")
        digest = _parse_integrity(integrity, context="Pi core lock integrity")
        artifact = _cache_blob_path(cache_sha512_root, integrity)
        existing = artifacts.get(digest)
        if existing is not None and existing != artifact:
            raise PiCorePrefixError("Pi core cache identity is ambiguous")
        artifacts[digest] = artifact
    if len(artifacts) != summary.unique_artifact_count:
        raise PiCorePrefixError("Pi core cache artifact count changed")
    total = 0
    for expected_digest, artifact in sorted(artifacts.items(), key=lambda item: item[0]):
        size, observed_digest = _hash_cache_blob(artifact)
        if observed_digest != expected_digest:
            raise PiCorePrefixError("Pi core cache blob digest changed")
        total += size
        if total > MAX_TOTAL_ARTIFACT_BYTES:
            raise PiCorePrefixError("Pi core cache exceeds its aggregate bound")
    return PiCoreCacheAudit(
        protocol=PI_CORE_PREFIX_PROTOCOL,
        frozen_lock_sha256=summary.frozen_lock_sha256,
        package_count=summary.package_count,
        artifact_count=len(artifacts),
        artifact_size_bytes=total,
    )


@dataclass(slots=True)
class _PrefixStagingTotals:
    """Bound the not-yet-published regular-file tree."""

    entries: int = 0
    files: int = 0
    size_bytes: int = 0

    def add_directory(self) -> None:
        self.entries += 1
        if self.entries > MAX_PREFIX_ENTRIES:
            raise PiCorePrefixError("Pi core prefix exceeds its entry bound")

    def add_file(self, size_bytes: int) -> None:
        if not 0 <= size_bytes <= MAX_PREFIX_FILE_BYTES:
            raise PiCorePrefixError("Pi core archive file exceeds its bound")
        self.entries += 1
        self.files += 1
        self.size_bytes += size_bytes
        if (
            self.entries > MAX_PREFIX_ENTRIES
            or self.size_bytes > MAX_PREFIX_BYTES
        ):
            raise PiCorePrefixError("Pi core prefix exceeds its bound")


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _open_external_private_prefix_parent(
    path: Path, *, repo_root: Path
) -> tuple[Path, int]:
    """Open one exact owner-private materialization parent outside the repo."""

    try:
        repository = repo_root.resolve(strict=True)
    except OSError as error:
        raise PiCorePrefixError("Pi core repository root is unavailable") from error
    absolute = Path(os.path.abspath(path))
    if absolute == repository or _path_is_within(absolute, repository):
        raise PiCorePrefixError("Pi core prefix parent must stay outside the repository")
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise PiCorePrefixError("Pi core prefix parent is unavailable") from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise PiCorePrefixError("Pi core prefix parent is unsafe")
        if stat.S_ISLNK(metadata.st_mode):
            raise PiCorePrefixError("Pi core prefix parent contains a symbolic link")
        writable_by_peer = bool(metadata.st_mode & 0o022)
        sticky_root_directory = (
            metadata.st_uid == 0 and bool(metadata.st_mode & stat.S_ISVTX)
        )
        if writable_by_peer and not sticky_root_directory:
            raise PiCorePrefixError("Pi core prefix parent is writable by another user")
        if metadata.st_uid not in {0, os.geteuid()}:
            raise PiCorePrefixError("Pi core prefix parent has an unexpected owner")
    try:
        final_metadata = os.lstat(absolute)
    except OSError as error:
        raise PiCorePrefixError("Pi core prefix parent is unavailable") from error
    if (
        not stat.S_ISDIR(final_metadata.st_mode)
        or final_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(final_metadata.st_mode) != 0o700
    ):
        raise PiCorePrefixError("Pi core prefix parent must be owner-private mode 0700")
    flags = _secure_open_flags(os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise PiCorePrefixError("Pi core prefix parent cannot be opened") from error
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode) or not _same_file(final_metadata, opened):
        os.close(descriptor)
        raise PiCorePrefixError("Pi core prefix parent changed while opening")
    return absolute, descriptor


def _new_private_prefix_staging_directory(parent_descriptor: int) -> tuple[str, int]:
    """Create one unguessable, owner-private staging directory by descriptor."""

    flags = _secure_open_flags(os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    for _attempt in range(32):
        name = ".sparkbench-pi-prefix-" + secrets.token_hex(16)
        try:
            os.mkdir(name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        except OSError as error:
            raise PiCorePrefixError("Pi core prefix cannot stage") from error
        try:
            descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        except OSError as error:
            try:
                os.rmdir(name, dir_fd=parent_descriptor)
            except OSError:
                pass
            raise PiCorePrefixError("Pi core prefix cannot stage") from error
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            os.close(descriptor)
            try:
                os.rmdir(name, dir_fd=parent_descriptor)
            except OSError:
                pass
            raise PiCorePrefixError("Pi core prefix staging directory is unsafe")
        return name, descriptor
    raise PiCorePrefixError("Pi core prefix staging name collision")


def _staging_directory(
    parent_descriptor: int,
    name: str,
    totals: _PrefixStagingTotals,
    *,
    count_in_prefix: bool = True,
) -> int:
    """Open an existing staging directory or create it with exact mode 0700."""

    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise PiCorePrefixError("Pi core prefix staging path is invalid")
    created = False
    try:
        os.mkdir(name, 0o700, dir_fd=parent_descriptor)
        created = True
    except FileExistsError:
        pass
    except OSError as error:
        raise PiCorePrefixError("Pi core prefix staging directory cannot be created") from error
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise PiCorePrefixError("Pi core prefix staging directory cannot be inspected") from error
    if (
        not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) != 0o700
    ):
        raise PiCorePrefixError("Pi core prefix staging directory is unsafe")
    flags = _secure_open_flags(os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise PiCorePrefixError("Pi core prefix staging directory cannot be opened") from error
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode) or not _same_file(before, opened):
        os.close(descriptor)
        raise PiCorePrefixError("Pi core prefix staging directory changed while opening")
    if created and count_in_prefix:
        totals.add_directory()
    return descriptor


def _ensure_staging_directories(
    root_descriptor: int,
    components: tuple[str, ...],
    totals: _PrefixStagingTotals,
) -> int:
    """Create a safe relative directory chain and return its final descriptor."""

    if not components:
        raise PiCorePrefixError("Pi core prefix staging path is empty")
    descriptor = os.dup(root_descriptor)
    try:
        for component in components:
            child = _staging_directory(descriptor, component, totals)
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _safe_archive_components(value: object) -> tuple[str, ...]:
    """Validate one effective tar member path before it reaches the filesystem."""

    if not isinstance(value, str):
        raise PiCorePrefixError("Pi core archive path is invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as error:
        raise PiCorePrefixError("Pi core archive path is not UTF-8") from error
    normalized = value.rstrip("/")
    if (
        not normalized
        or len(encoded) > 4_096
        or "\x00" in value
        or "\r" in value
        or "\n" in value
        or value.startswith("/")
        or "\\" in value
    ):
        raise PiCorePrefixError("Pi core archive path is invalid")
    raw_components = tuple(normalized.split("/"))
    if (
        len(raw_components) > 64
        or any(
            not component
            or component == ".."
            or len(component.encode("utf-8")) > 255
            for component in raw_components
        )
    ):
        raise PiCorePrefixError("Pi core archive path is invalid")
    components = tuple(component for component in raw_components if component != ".")
    if not components:
        raise PiCorePrefixError("Pi core archive path is invalid")
    return components


def _write_archive_regular_file(
    directory_descriptor: int,
    name: str,
    source: object,
    *,
    size_bytes: int,
    capture: bool,
    totals: _PrefixStagingTotals,
) -> bytes | None:
    """Write one bounded tar member once, without allowing replacement."""

    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise PiCorePrefixError("Pi core archive file path is invalid")
    if not 0 <= size_bytes <= MAX_PREFIX_FILE_BYTES:
        raise PiCorePrefixError("Pi core archive file exceeds its bound")
    if capture and size_bytes > MAX_PACKAGE_MANIFEST_BYTES:
        raise PiCorePrefixError("Pi core package manifest exceeds its bound")
    totals.add_file(size_bytes)
    flags = _secure_open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_descriptor)
    except OSError as error:
        raise PiCorePrefixError("Pi core archive file cannot be staged") from error
    chunks: list[bytes] | None = [] if capture else None
    remaining = size_bytes
    try:
        os.fchmod(descriptor, 0o600)
        while remaining:
            try:
                block = source.read(min(remaining, 1024 * 1024))
            except (OSError, tarfile.TarError) as error:
                raise PiCorePrefixError("Pi core archive file cannot be read") from error
            if not isinstance(block, bytes) or not block:
                raise PiCorePrefixError("Pi core archive file is truncated")
            _write_all(descriptor, block)
            if chunks is not None:
                chunks.append(block)
            remaining -= len(block)
        try:
            trailing = source.read(1)
        except (OSError, tarfile.TarError) as error:
            raise PiCorePrefixError("Pi core archive file cannot be read") from error
        if trailing not in {b"", None}:
            raise PiCorePrefixError("Pi core archive file size is inconsistent")
        os.fsync(descriptor)
        written = os.fstat(descriptor)
        if (
            not stat.S_ISREG(written.st_mode)
            or written.st_uid != os.geteuid()
            or written.st_nlink != 1
            or stat.S_IMODE(written.st_mode) != 0o600
            or written.st_size != size_bytes
        ):
            raise PiCorePrefixError("Pi core archive staging file is unsafe")
    finally:
        os.close(descriptor)
    return b"".join(chunks) if chunks is not None else None


def _assert_duplicate_archive_file_matches_staging(
    directory_descriptor: int,
    name: str,
    source: object,
    *,
    size_bytes: int,
) -> None:
    """Permit only byte-identical effective duplicate regular tar members."""

    if not 0 <= size_bytes <= MAX_PREFIX_FILE_BYTES:
        raise PiCorePrefixError("Pi core archive file exceeds its bound")
    try:
        before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except OSError as error:
        raise PiCorePrefixError("Pi core duplicate archive file is unavailable") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_size != size_bytes
    ):
        raise PiCorePrefixError("Pi core duplicate archive file conflicts")
    flags = _secure_open_flags(os.O_RDONLY)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as error:
        raise PiCorePrefixError("Pi core duplicate archive file cannot be opened") from error
    remaining = size_bytes
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file(before, opened):
            raise PiCorePrefixError("Pi core duplicate archive file changed while opening")
        while remaining:
            try:
                archive_block = source.read(min(remaining, 1024 * 1024))
            except (OSError, tarfile.TarError) as error:
                raise PiCorePrefixError("Pi core duplicate archive file cannot be read") from error
            if not isinstance(archive_block, bytes) or not archive_block:
                raise PiCorePrefixError("Pi core duplicate archive file conflicts")
            staged_block = os.read(descriptor, len(archive_block))
            if archive_block != staged_block:
                raise PiCorePrefixError("Pi core duplicate archive file conflicts")
            remaining -= len(archive_block)
        try:
            trailing = source.read(1)
        except (OSError, tarfile.TarError) as error:
            raise PiCorePrefixError("Pi core duplicate archive file cannot be read") from error
        if trailing not in {b"", None} or os.read(descriptor, 1):
            raise PiCorePrefixError("Pi core duplicate archive file conflicts")
        after = os.fstat(descriptor)
        if not _same_file(opened, after):
            raise PiCorePrefixError("Pi core duplicate archive file changed while reading")
    finally:
        os.close(descriptor)


def _copy_cache_blob_to_private_staging(
    cache_sha512_root: Path,
    integrity: object,
    *,
    staging_descriptor: int,
    name: str,
) -> None:
    """Copy-and-hash one mutable cache blob before any archive processing."""

    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise PiCorePrefixError("Pi core cache staging name is invalid")
    expected_digest = _parse_integrity(integrity, context="Pi core cache integrity")
    artifact = _cache_blob_path(cache_sha512_root, integrity)
    try:
        before = os.lstat(artifact)
    except OSError as error:
        raise PiCorePrefixError("Pi core cache blob is unavailable") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or not 0 < before.st_size <= MAX_MATERIALIZATION_ARTIFACT_BYTES
    ):
        raise PiCorePrefixError("Pi core cache blob is unsafe for materialization")
    read_flags = _secure_open_flags(os.O_RDONLY)
    write_flags = _secure_open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        source_descriptor = os.open(artifact, read_flags)
    except OSError as error:
        raise PiCorePrefixError("Pi core cache blob cannot be opened") from error
    destination_descriptor: int | None = None
    try:
        opened = os.fstat(source_descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file(before, opened):
            raise PiCorePrefixError("Pi core cache blob changed while opening")
        try:
            destination_descriptor = os.open(
                name, write_flags, 0o600, dir_fd=staging_descriptor
            )
        except OSError as error:
            raise PiCorePrefixError("Pi core cache staging file cannot be created") from error
        digest = hashlib.sha512()
        total = 0
        while True:
            block = os.read(source_descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > MAX_MATERIALIZATION_ARTIFACT_BYTES:
                raise PiCorePrefixError("Pi core cache blob exceeds materialization bound")
            digest.update(block)
            _write_all(destination_descriptor, block)
        after = os.fstat(source_descriptor)
        if total != opened.st_size or not _same_file(opened, after):
            raise PiCorePrefixError("Pi core cache blob changed while copying")
        if digest.digest() != expected_digest:
            raise PiCorePrefixError("Pi core cache blob digest changed")
        os.fchmod(destination_descriptor, 0o400)
        os.fsync(destination_descriptor)
        copied = os.fstat(destination_descriptor)
        if (
            not stat.S_ISREG(copied.st_mode)
            or copied.st_uid != os.geteuid()
            or copied.st_nlink != 1
            or stat.S_IMODE(copied.st_mode) != 0o400
            or copied.st_size != total
        ):
            raise PiCorePrefixError("Pi core cache staging file is unsafe")
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        os.close(source_descriptor)


def _expected_archive_package_names(summary: PiCoreLockSummary) -> dict[str, str]:
    """Bind npm alias locations to their actual package manifest names."""

    packages_value = summary.document.get("packages")
    if type(packages_value) is not dict:
        raise PiCorePrefixError("Pi core lock packages are invalid")
    packages = {path: entry for path, entry in packages_value.items() if path != ""}
    aliases: dict[str, str] = {}
    for requester in sorted(packages):
        entry = packages[requester]
        if type(entry) is not dict:
            raise PiCorePrefixError("Pi core lock package entry is invalid")
        for key in ("dependencies", "optionalDependencies", "peerDependencies"):
            dependencies = _dependency_map(
                entry, key, context="Pi core lock package entry"
            )
            for dependency, specification in dependencies.items():
                if not specification.startswith("npm:"):
                    continue
                target = _resolve_dependency(packages, requester, dependency)
                if target is None:
                    continue
                alias_specification = specification.removeprefix("npm:")
                if alias_specification.startswith("@"):
                    separator = alias_specification.find("@", alias_specification.find("/") + 1)
                else:
                    separator = alias_specification.find("@")
                if separator <= 0 or separator == len(alias_specification) - 1:
                    raise PiCorePrefixError("Pi core npm alias specification is invalid")
                package_name = alias_specification[:separator]
                _package_name_parts(
                    package_name, context="Pi core npm alias package name"
                )
                existing = aliases.setdefault(target, package_name)
                if existing != package_name:
                    raise PiCorePrefixError("Pi core npm alias resolution is ambiguous")
    result: dict[str, str] = {}
    for path in sorted(packages):
        result[path] = aliases.get(path, _package_name_from_path(path))
    return result


def _validate_extracted_package_manifest(
    payload: bytes | None, *, expected_name: str, expected_version: object
) -> None:
    """Check the installed package identity without executing its scripts."""

    if payload is None:
        raise PiCorePrefixError("Pi core archive has no package manifest")
    manifest = _load_json_object(payload, context="Pi core package manifest")
    name = manifest.get("name")
    _package_name_parts(name, context="Pi core package manifest name")
    if name != expected_name or manifest.get("version") != expected_version:
        raise PiCorePrefixError("Pi core package manifest identity changed")


def _extract_private_archive(
    artifact_directory_descriptor: int,
    artifact_name: str,
    *,
    package_path: str,
    package_entry: Mapping[str, object],
    expected_name: str,
    prefix_descriptor: int,
    totals: _PrefixStagingTotals,
) -> None:
    """Extract one verified private tarball with no links or overwrite paths."""

    try:
        before = os.stat(
            artifact_name,
            dir_fd=artifact_directory_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise PiCorePrefixError("Pi core private artifact is unavailable") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o400
        or not 0 < before.st_size <= MAX_MATERIALIZATION_ARTIFACT_BYTES
    ):
        raise PiCorePrefixError("Pi core private artifact is unsafe")
    flags = _secure_open_flags(os.O_RDONLY)
    try:
        descriptor = os.open(artifact_name, flags, dir_fd=artifact_directory_descriptor)
    except OSError as error:
        raise PiCorePrefixError("Pi core private artifact cannot be opened") from error
    file_object = os.fdopen(descriptor, "rb", closefd=False)
    archive: tarfile.TarFile | None = None
    target_components = tuple(package_path.split("/"))
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file(before, opened):
            raise PiCorePrefixError("Pi core private artifact changed while opening")
        try:
            archive = tarfile.open(fileobj=file_object, mode="r|gz")
        except (OSError, tarfile.TarError) as error:
            raise PiCorePrefixError("Pi core private artifact is not a gzip tarball") from error
        archive_root: str | None = None
        seen_paths: dict[tuple[str, ...], str] = {}
        manifest_payload: bytes | None = None
        member_count = 0
        path_metadata_bytes = 0
        for member in archive:
            member_count += 1
            if member_count > MAX_ARCHIVE_MEMBERS:
                raise PiCorePrefixError("Pi core archive exceeds its member bound")
            components = _safe_archive_components(member.name)
            path_metadata_bytes += len(member.name.encode("utf-8"))
            if path_metadata_bytes > MAX_ARCHIVE_PATH_METADATA_BYTES:
                raise PiCorePrefixError("Pi core archive path metadata exceeds its bound")
            if archive_root is None:
                archive_root = components[0]
            if components[0] != archive_root:
                raise PiCorePrefixError("Pi core archive has multiple top-level roots")
            relative_components = components[1:]
            if "node_modules" in relative_components:
                raise PiCorePrefixError("Pi core archive embeds node_modules")
            if getattr(member, "issparse", lambda: False)():
                raise PiCorePrefixError("Pi core archive has an unsupported sparse file")
            member_kind = (
                "directory"
                if member.isdir()
                else "file"
                if member.isreg()
                else "unsupported"
            )
            previous_kind = seen_paths.get(relative_components)
            if previous_kind is not None:
                if previous_kind != member_kind:
                    raise PiCorePrefixError("Pi core archive member paths conflict")
                if member_kind == "directory":
                    if member.size != 0:
                        raise PiCorePrefixError("Pi core archive directory has content")
                    continue
                if member_kind != "file" or not relative_components:
                    raise PiCorePrefixError("Pi core archive has an unsupported member")
                if not isinstance(member.size, int) or isinstance(member.size, bool):
                    raise PiCorePrefixError("Pi core archive member size is invalid")
                parent = _ensure_staging_directories(
                    prefix_descriptor,
                    target_components + relative_components[:-1],
                    totals,
                )
                try:
                    source = archive.extractfile(member)
                    if source is None:
                        raise PiCorePrefixError("Pi core archive file cannot be extracted")
                    try:
                        _assert_duplicate_archive_file_matches_staging(
                            parent,
                            relative_components[-1],
                            source,
                            size_bytes=member.size,
                        )
                    finally:
                        source.close()
                finally:
                    os.close(parent)
                continue
            seen_paths[relative_components] = member_kind
            if member.isdir():
                if member.size != 0:
                    raise PiCorePrefixError("Pi core archive directory has content")
                if relative_components:
                    directory = _ensure_staging_directories(
                        prefix_descriptor,
                        target_components + relative_components,
                        totals,
                    )
                    os.close(directory)
                continue
            if not member.isreg() or not relative_components:
                raise PiCorePrefixError("Pi core archive has an unsupported member")
            if not isinstance(member.size, int) or isinstance(member.size, bool):
                raise PiCorePrefixError("Pi core archive member size is invalid")
            parent = _ensure_staging_directories(
                prefix_descriptor,
                target_components + relative_components[:-1],
                totals,
            )
            try:
                source = archive.extractfile(member)
                if source is None:
                    raise PiCorePrefixError("Pi core archive file cannot be extracted")
                try:
                    payload = _write_archive_regular_file(
                        parent,
                        relative_components[-1],
                        source,
                        size_bytes=member.size,
                        capture=relative_components == ("package.json",),
                        totals=totals,
                    )
                finally:
                    source.close()
            finally:
                os.close(parent)
            if relative_components == ("package.json",):
                manifest_payload = payload
        if archive_root is None:
            raise PiCorePrefixError("Pi core archive is empty")
        _validate_extracted_package_manifest(
            manifest_payload,
            expected_name=expected_name,
            expected_version=package_entry.get("version"),
        )
        after = os.fstat(descriptor)
        if not _same_file(opened, after):
            raise PiCorePrefixError("Pi core private artifact changed while extracting")
    except (OSError, EOFError, tarfile.TarError) as error:
        raise PiCorePrefixError("Pi core private artifact cannot be extracted") from error
    finally:
        if archive is not None:
            archive.close()
        file_object.close()
        os.close(descriptor)


def _normalize_staging_prefix(root_descriptor: int) -> None:
    """Normalize a freshly built prefix before it is visible at its final name."""

    def visit(descriptor: int) -> None:
        try:
            names = sorted(os.listdir(descriptor), key=lambda value: value.encode("utf-8"))
        except (OSError, UnicodeError) as error:
            raise PiCorePrefixError("Pi core prefix cannot be listed safely") from error
        for name in names:
            if not name or "/" in name or "\\" in name or name in {".", ".."}:
                raise PiCorePrefixError("Pi core prefix entry name is invalid")
            try:
                before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as error:
                raise PiCorePrefixError("Pi core prefix entry cannot be inspected") from error
            if before.st_uid != os.geteuid():
                raise PiCorePrefixError("Pi core prefix entry has an unexpected owner")
            if stat.S_ISDIR(before.st_mode):
                flags = _secure_open_flags(
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                )
                try:
                    child = os.open(name, flags, dir_fd=descriptor)
                except OSError as error:
                    raise PiCorePrefixError("Pi core prefix directory cannot be opened") from error
                try:
                    opened = os.fstat(child)
                    if not stat.S_ISDIR(opened.st_mode) or not _same_file(before, opened):
                        raise PiCorePrefixError("Pi core prefix directory changed while opening")
                    if stat.S_IMODE(opened.st_mode) != 0o700:
                        raise PiCorePrefixError("Pi core prefix directory mode is unsafe")
                    visit(child)
                    os.fchmod(child, 0o555)
                    os.fsync(child)
                finally:
                    os.close(child)
            elif stat.S_ISREG(before.st_mode):
                if (
                    before.st_nlink != 1
                    or stat.S_IMODE(before.st_mode) != 0o600
                ):
                    raise PiCorePrefixError("Pi core prefix file is unsafe")
                flags = _secure_open_flags(os.O_RDONLY)
                try:
                    file_descriptor = os.open(name, flags, dir_fd=descriptor)
                except OSError as error:
                    raise PiCorePrefixError("Pi core prefix file cannot be opened") from error
                try:
                    opened = os.fstat(file_descriptor)
                    if not stat.S_ISREG(opened.st_mode) or not _same_file(before, opened):
                        raise PiCorePrefixError("Pi core prefix file changed while opening")
                    os.fchmod(file_descriptor, 0o444)
                    os.fsync(file_descriptor)
                finally:
                    os.close(file_descriptor)
            else:
                raise PiCorePrefixError("Pi core prefix contains a special file")
        try:
            after_names = sorted(
                os.listdir(descriptor), key=lambda value: value.encode("utf-8")
            )
        except (OSError, UnicodeError) as error:
            raise PiCorePrefixError("Pi core prefix changed while normalizing") from error
        if names != after_names:
            raise PiCorePrefixError("Pi core prefix changed while normalizing")

    visit(root_descriptor)
    os.fchmod(root_descriptor, 0o555)
    os.fsync(root_descriptor)


def _ensure_no_existing_prefix(parent_descriptor: int) -> None:
    try:
        os.lstat(PI_CORE_PREFIX_DIRECTORY_NAME, dir_fd=parent_descriptor)
    except FileNotFoundError:
        return
    except OSError as error:
        raise PiCorePrefixError("Pi core prefix destination cannot be inspected") from error
    raise PiCorePrefixError("Pi core prefix destination already exists")


def _best_effort_remove_prefix_staging(
    parent_descriptor: int, name: str, root_descriptor: int
) -> None:
    """Remove only the still-bound private staging tree after a failed build.

    Cleanup is deliberately best effort.  Every action is descriptor-relative;
    if a same-owner process substitutes any directory name, inode comparison
    fails and the unexpected object is left in place rather than removed.
    """

    root_identity = os.fstat(root_descriptor)

    def remove_children(descriptor: int) -> None:
        os.fchmod(descriptor, 0o700)
        names = os.listdir(descriptor)
        for child_name in names:
            before = os.stat(
                child_name, dir_fd=descriptor, follow_symlinks=False
            )
            if stat.S_ISDIR(before.st_mode):
                flags = _secure_open_flags(
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                )
                child = os.open(child_name, flags, dir_fd=descriptor)
                try:
                    opened = os.fstat(child)
                    if (
                        not stat.S_ISDIR(opened.st_mode)
                        or (before.st_dev, before.st_ino)
                        != (opened.st_dev, opened.st_ino)
                    ):
                        raise PiCorePrefixError(
                            "Pi core prefix staging directory changed during cleanup"
                        )
                    remove_children(child)
                finally:
                    os.close(child)
                after = os.stat(
                    child_name, dir_fd=descriptor, follow_symlinks=False
                )
                if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                    raise PiCorePrefixError(
                        "Pi core prefix staging directory changed during cleanup"
                    )
                os.rmdir(child_name, dir_fd=descriptor)
            else:
                os.unlink(child_name, dir_fd=descriptor)
        os.fsync(descriptor)

    try:
        remove_children(root_descriptor)
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            stat.S_ISDIR(named.st_mode)
            and (named.st_dev, named.st_ino)
            == (root_identity.st_dev, root_identity.st_ino)
        ):
            os.rmdir(name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
    except (OSError, PiCorePrefixError):
        # A private stale directory is safer than any broader pathname cleanup.
        pass


def _rename_directory_no_replace(
    parent_descriptor: int, source_name: str, destination_name: str
) -> None:
    """Publish a staged directory atomically without POSIX-rename replacement.

    Python does not expose ``renameat2``.  Falling back to ``os.rename`` would
    let a concurrent creator place an empty destination between the preflight
    check and the publish step, which POSIX rename may replace.  Require Linux
    ``RENAME_NOREPLACE`` instead and fail closed when the host libc lacks it.
    """

    if (
        not source_name
        or not destination_name
        or "/" in source_name
        or "/" in destination_name
        or "\\" in source_name
        or "\\" in destination_name
        or source_name in {".", ".."}
        or destination_name in {".", ".."}
    ):
        raise PiCorePrefixError("Pi core prefix publish name is invalid")
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as error:
        raise PiCorePrefixError("Pi core prefix requires renameat2 no-replace support") from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_code = ctypes.get_errno()
    if error_code == errno.EEXIST:
        raise PiCorePrefixError("Pi core prefix destination already exists")
    raise PiCorePrefixError("Pi core prefix cannot be published") from OSError(error_code, "renameat2")


def materialize_pi_core_prefix(
    summary: PiCoreLockSummary,
    *,
    cache_sha512_root: Path,
    prefix_parent: Path,
    repo_root: Path,
) -> PiCorePrefixMaterialization:
    """Build one immutable Pi-core prefix directly from verified cached tarballs.

    This deliberately invokes no package manager, Node process, Pi runtime,
    network client, container, or model server.  Every archive is copied and
    SHA-512 verified from the mutable cache into a private staging file before
    the tar parser reads it.  A fully normalized prefix is then published only
    at an absent deterministic child name of an explicitly supplied external
    owner-private parent.
    """

    summary = _validated_summary(summary)
    cache_audit = audit_pi_core_cache(summary, cache_sha512_root=cache_sha512_root)
    expected_names = _expected_archive_package_names(summary)
    packages_value = summary.document.get("packages")
    if type(packages_value) is not dict:
        raise PiCorePrefixError("Pi core lock packages are invalid")
    parent_path, parent_descriptor = _open_external_private_prefix_parent(
        prefix_parent, repo_root=repo_root
    )
    staging_name: str | None = None
    staging_descriptor: int | None = None
    artifact_descriptor: int | None = None
    published = False
    try:
        _ensure_no_existing_prefix(parent_descriptor)
        staging_name, staging_descriptor = _new_private_prefix_staging_directory(
            parent_descriptor
        )
        totals = _PrefixStagingTotals()
        artifact_descriptor = _staging_directory(
            staging_descriptor, ".artifacts", totals, count_in_prefix=False
        )
        for package_path in sorted(expected_names):
            entry = packages_value.get(package_path)
            if type(entry) is not dict:
                raise PiCorePrefixError("Pi core lock package entry is invalid")
            artifact_name = "artifact.tgz"
            _copy_cache_blob_to_private_staging(
                cache_sha512_root,
                entry.get("integrity"),
                staging_descriptor=artifact_descriptor,
                name=artifact_name,
            )
            try:
                _extract_private_archive(
                    artifact_descriptor,
                    artifact_name,
                    package_path=package_path,
                    package_entry=entry,
                    expected_name=expected_names[package_path],
                    prefix_descriptor=staging_descriptor,
                    totals=totals,
                )
            finally:
                try:
                    os.unlink(artifact_name, dir_fd=artifact_descriptor)
                except FileNotFoundError:
                    pass
                except OSError as error:
                    raise PiCorePrefixError("Pi core private artifact cannot be removed") from error
        os.fsync(artifact_descriptor)
        os.close(artifact_descriptor)
        artifact_descriptor = None
        try:
            os.rmdir(".artifacts", dir_fd=staging_descriptor)
        except OSError as error:
            raise PiCorePrefixError("Pi core private artifact directory cannot be removed") from error
        _normalize_staging_prefix(staging_descriptor)
        staging_path = parent_path / staging_name
        try:
            tree = inspect_normalized_tree(staging_path, repo_root=repo_root)
        except RuntimeAssetError as error:
            raise PiCorePrefixError("Pi core normalized prefix is unsafe") from error
        if (
            tree.links != 0
            or tree.entries != totals.entries
            or tree.files != totals.files
            or tree.size_bytes != totals.size_bytes
        ):
            raise PiCorePrefixError("Pi core normalized prefix inventory changed")
        _ensure_no_existing_prefix(parent_descriptor)
        _rename_directory_no_replace(
            parent_descriptor, staging_name, PI_CORE_PREFIX_DIRECTORY_NAME
        )
        # Once renameat2 succeeds, the deterministic destination is live even
        # if its durability fsync reports an error.  Never treat it as staging
        # again or cleanup could remove a successfully published prefix.
        published = True
        try:
            os.fsync(parent_descriptor)
        except OSError as error:
            raise PiCorePrefixError(
                "Pi core prefix was published but its parent directory was not synced"
            ) from error
        final_path = parent_path / PI_CORE_PREFIX_DIRECTORY_NAME
        return PiCorePrefixMaterialization(
            protocol=PI_CORE_MATERIALIZED_PREFIX_PROTOCOL,
            frozen_lock_sha256=summary.frozen_lock_sha256,
            package_count=summary.package_count,
            artifact_count=cache_audit.artifact_count,
            artifact_size_bytes=cache_audit.artifact_size_bytes,
            tree_digest="sha256:" + tree.digest,
            tree_entries=tree.entries,
            tree_files=tree.files,
            tree_size_bytes=tree.size_bytes,
            prefix_directory_name=PI_CORE_PREFIX_DIRECTORY_NAME,
            resolved_path=final_path,
        )
    finally:
        if artifact_descriptor is not None:
            os.close(artifact_descriptor)
        if staging_descriptor is not None:
            if not published and staging_name is not None:
                _best_effort_remove_prefix_staging(
                    parent_descriptor, staging_name, staging_descriptor
                )
            os.close(staging_descriptor)
        os.close(parent_descriptor)
