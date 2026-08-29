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
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Mapping
from urllib.parse import urlsplit


PI_CORE_PREFIX_PROTOCOL = "sparkbench-pi-core-source-v1"
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
PI_CORE_PATH_LIST_SHA256 = (
    "sha256:ebe369bc873ebab868810aad5b728b3f3a900bad215d13d7f6f2cc93c8c660f0"
)
PI_CORE_PATH_VERSION_INTEGRITY_SHA256 = (
    "sha256:ea4489fd826db97b06da628392900a210c90a44d9e4efc92f7d30019c3cab2fc"
)
MAX_LOCK_BYTES = 32 * 1024 * 1024
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_TOTAL_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
MAX_CLOSURE_PACKAGES = 1_024

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


def _write_descriptor(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise PiCorePrefixError("Pi core frozen lock write made no progress")
        offset += written
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
