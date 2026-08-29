"""Read-only static admission for one materialized Pi-core prefix.

The frozen source closure and scripts-disabled materializer deliberately do not
make a prefix executable.  This module binds a separately retained external
prefix to its complete immutable-tree and Pi-core entrypoint pins without
invoking Node, npm, Pi, Docker, a model server, or the network.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

from bench.harbor_runtime_assets import (
    RuntimeAssetError,
    TREE_PROTOCOL,
    verify_admitted_file,
    verify_normalized_tree,
)
from bench.pi_core_prefix import (
    PI_CORE_FROZEN_LOCK_SHA256,
    PI_CORE_PREFIX_DIRECTORY_NAME,
)


PI_CORE_PREFIX_ADMISSION_PROTOCOL = "sparkbench-pi-core-prefix-admission-v1"
PI_CORE_PREFIX_ADMISSION_SCHEMA_VERSION = 1
PI_CORE_PREFIX_ENTRYPOINT = (
    "node_modules/@mariozechner/pi-agent-core/dist/index.js"
)
MAX_PREFIX_ADMISSION_PIN_BYTES = 16 * 1024

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PIN_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "frozen_lock_sha256",
        "prefix_directory_name",
        "tree",
        "entrypoint",
    }
)
_TREE_FIELDS = frozenset(
    {
        "protocol",
        "digest",
        "entries",
        "files",
        "links",
        "size_bytes",
    }
)
_ENTRYPOINT_FIELDS = frozenset(
    {
        "relative_path",
        "digest",
        "size_bytes",
        "mode",
    }
)


class PiCorePrefixAdmissionError(RuntimeError):
    """Raised when a Pi-core prefix cannot receive the frozen static admission."""


@dataclass(frozen=True, slots=True)
class PiCorePrefixAdmissionPin:
    """Schema-locked immutable identity expected from a retained prefix."""

    protocol: str
    frozen_lock_sha256: str
    prefix_directory_name: str
    tree_protocol: str
    tree_digest: str
    tree_entries: int
    tree_files: int
    tree_links: int
    tree_size_bytes: int
    entrypoint_relative_path: str
    entrypoint_digest: str
    entrypoint_size_bytes: int
    entrypoint_mode: int


@dataclass(frozen=True, slots=True)
class PiCorePrefixAdmission:
    """Scalar-only result from validating an external retained Pi-core prefix."""

    protocol: str
    frozen_lock_sha256: str
    tree_protocol: str
    tree_digest: str
    tree_entries: int
    tree_files: int
    tree_links: int
    tree_size_bytes: int
    entrypoint_digest: str
    entrypoint_size_bytes: int
    entrypoint_mode: int

    def scalar(self) -> dict[str, object]:
        """Return the admission result without local paths or package metadata."""

        return {
            "protocol": self.protocol,
            "status": "admitted",
            "frozen_lock_sha256": self.frozen_lock_sha256,
            "tree_protocol": self.tree_protocol,
            "tree_digest": self.tree_digest,
            "tree_entries": self.tree_entries,
            "tree_files": self.tree_files,
            "tree_links": self.tree_links,
            "tree_size_bytes": self.tree_size_bytes,
            "entrypoint_digest": self.entrypoint_digest,
            "entrypoint_size_bytes": self.entrypoint_size_bytes,
            "entrypoint_mode": f"{self.entrypoint_mode:04o}",
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


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise PiCorePrefixAdmissionError("Pi core prefix admission pin is invalid") from error
    return (encoded + "\n").encode("ascii")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError("non-finite JSON constant: " + value)


def _read_frozen_pin(path: Path) -> bytes:
    """Read one small tracked policy file without symlink or mutation races."""

    candidate = Path(os.path.abspath(path))
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise PiCorePrefixAdmissionError(
                "Pi core prefix admission pin cannot be inspected"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise PiCorePrefixAdmissionError(
                "Pi core prefix admission pin path contains a symbolic link"
            )
    try:
        before = os.lstat(candidate)
    except OSError as error:
        raise PiCorePrefixAdmissionError(
            "Pi core prefix admission pin cannot be inspected"
        ) from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.geteuid()
        or before.st_mode & 0o022
        or not 0 < before.st_size <= MAX_PREFIX_ADMISSION_PIN_BYTES
    ):
        raise PiCorePrefixAdmissionError("Pi core prefix admission pin is unsafe")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise PiCorePrefixAdmissionError("Pi core prefix admission requires O_NOFOLLOW")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise PiCorePrefixAdmissionError(
            "Pi core prefix admission pin cannot be opened"
        ) from error
    chunks: list[bytes] = []
    remaining = MAX_PREFIX_ADMISSION_PIN_BYTES + 1
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file(before, opened):
            raise PiCorePrefixAdmissionError(
                "Pi core prefix admission pin changed while opening"
            )
        while remaining > 0:
            block = os.read(descriptor, min(remaining, 64 * 1024))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            not _same_file(opened, after)
            or len(payload) != opened.st_size
            or len(payload) > MAX_PREFIX_ADMISSION_PIN_BYTES
        ):
            raise PiCorePrefixAdmissionError(
                "Pi core prefix admission pin changed while reading"
            )
        return payload
    finally:
        os.close(descriptor)


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise PiCorePrefixAdmissionError(f"Pi core prefix admission {field} is invalid")
    return value


def _require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PiCorePrefixAdmissionError(f"Pi core prefix admission {field} is invalid")
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PiCorePrefixAdmissionError(f"Pi core prefix admission {field} is invalid")
    return value


def _valid_direct_pin(pin: object, *, frozen_lock_sha256: str) -> bool:
    """Reject malformed caller-built pins before opening an external prefix."""

    if not isinstance(pin, PiCorePrefixAdmissionPin):
        return False
    if (
        frozen_lock_sha256 != PI_CORE_FROZEN_LOCK_SHA256
        or pin.protocol != PI_CORE_PREFIX_ADMISSION_PROTOCOL
        or pin.frozen_lock_sha256 != frozen_lock_sha256
        or pin.prefix_directory_name != PI_CORE_PREFIX_DIRECTORY_NAME
        or pin.tree_protocol != TREE_PROTOCOL
        or pin.entrypoint_relative_path != PI_CORE_PREFIX_ENTRYPOINT
        or pin.entrypoint_mode != 0o444
        or not isinstance(pin.tree_digest, str)
        or _SHA256_PATTERN.fullmatch(pin.tree_digest) is None
        or not isinstance(pin.entrypoint_digest, str)
        or _SHA256_PATTERN.fullmatch(pin.entrypoint_digest) is None
        or isinstance(pin.tree_entries, bool)
        or not isinstance(pin.tree_entries, int)
        or pin.tree_entries <= 0
        or isinstance(pin.tree_files, bool)
        or not isinstance(pin.tree_files, int)
        or not 0 < pin.tree_files <= pin.tree_entries
        or type(pin.tree_links) is not int
        or pin.tree_links != 0
        or isinstance(pin.tree_size_bytes, bool)
        or not isinstance(pin.tree_size_bytes, int)
        or pin.tree_size_bytes <= 0
        or isinstance(pin.entrypoint_size_bytes, bool)
        or not isinstance(pin.entrypoint_size_bytes, int)
        or pin.entrypoint_size_bytes <= 0
    ):
        return False
    return True


def load_pi_core_prefix_admission_pin(path: Path) -> PiCorePrefixAdmissionPin:
    """Load the fixed canonical static-prefix admission policy."""

    payload = _read_frozen_pin(path)
    try:
        document = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_reject_json_constant,
        )
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise PiCorePrefixAdmissionError("Pi core prefix admission pin is invalid") from error
    if (
        type(document) is not dict
        or frozenset(document) != _PIN_FIELDS
        or document.get("schema_version") != PI_CORE_PREFIX_ADMISSION_SCHEMA_VERSION
        or document.get("protocol") != PI_CORE_PREFIX_ADMISSION_PROTOCOL
        or document.get("frozen_lock_sha256") != PI_CORE_FROZEN_LOCK_SHA256
        or document.get("prefix_directory_name") != PI_CORE_PREFIX_DIRECTORY_NAME
    ):
        raise PiCorePrefixAdmissionError("Pi core prefix admission pin is invalid")
    if _canonical_json_bytes(document) != payload:
        raise PiCorePrefixAdmissionError("Pi core prefix admission pin is not canonical")
    tree = document["tree"]
    entrypoint = document["entrypoint"]
    if type(tree) is not dict or frozenset(tree) != _TREE_FIELDS:
        raise PiCorePrefixAdmissionError("Pi core prefix admission tree pin is invalid")
    if type(entrypoint) is not dict or frozenset(entrypoint) != _ENTRYPOINT_FIELDS:
        raise PiCorePrefixAdmissionError(
            "Pi core prefix admission entrypoint pin is invalid"
        )
    if tree.get("protocol") != TREE_PROTOCOL or tree.get("links") != 0:
        raise PiCorePrefixAdmissionError("Pi core prefix admission tree pin is invalid")
    tree_digest = _require_digest(tree.get("digest"), field="tree digest")
    tree_entries = _require_positive_int(tree.get("entries"), field="tree entries")
    tree_files = _require_positive_int(tree.get("files"), field="tree files")
    tree_links = _require_nonnegative_int(tree.get("links"), field="tree links")
    tree_size_bytes = _require_positive_int(
        tree.get("size_bytes"), field="tree size"
    )
    if tree_files + tree_links > tree_entries:
        raise PiCorePrefixAdmissionError("Pi core prefix admission tree pin is invalid")
    if (
        entrypoint.get("relative_path") != PI_CORE_PREFIX_ENTRYPOINT
        or entrypoint.get("mode") != 0o444
    ):
        raise PiCorePrefixAdmissionError(
            "Pi core prefix admission entrypoint pin is invalid"
        )
    relative_path = entrypoint["relative_path"]
    if (
        not isinstance(relative_path, str)
        or PurePosixPath(relative_path).is_absolute()
        or any(part in {"", ".", ".."} for part in PurePosixPath(relative_path).parts)
    ):
        raise PiCorePrefixAdmissionError(
            "Pi core prefix admission entrypoint pin is invalid"
        )
    return PiCorePrefixAdmissionPin(
        protocol=PI_CORE_PREFIX_ADMISSION_PROTOCOL,
        frozen_lock_sha256=PI_CORE_FROZEN_LOCK_SHA256,
        prefix_directory_name=PI_CORE_PREFIX_DIRECTORY_NAME,
        tree_protocol=TREE_PROTOCOL,
        tree_digest=tree_digest,
        tree_entries=tree_entries,
        tree_files=tree_files,
        tree_links=tree_links,
        tree_size_bytes=tree_size_bytes,
        entrypoint_relative_path=relative_path,
        entrypoint_digest=_require_digest(
            entrypoint.get("digest"), field="entrypoint digest"
        ),
        entrypoint_size_bytes=_require_positive_int(
            entrypoint.get("size_bytes"), field="entrypoint size"
        ),
        entrypoint_mode=0o444,
    )


def admit_pi_core_prefix(
    prefix: Path,
    *,
    repo_root: Path,
    frozen_lock_sha256: str,
    pin: PiCorePrefixAdmissionPin,
) -> PiCorePrefixAdmission:
    """Verify one externally retained normalized prefix against its fixed pin."""

    if not _valid_direct_pin(pin, frozen_lock_sha256=frozen_lock_sha256):
        raise PiCorePrefixAdmissionError("Pi core prefix admission policy is invalid")
    try:
        tree = verify_normalized_tree(
            prefix,
            repo_root=repo_root,
            expected_digest=pin.tree_digest,
            expected_size_bytes=pin.tree_size_bytes,
            expected_entries=pin.tree_entries,
            expected_files=pin.tree_files,
            expected_links=pin.tree_links,
        )
        if tree.resolved_path.name != pin.prefix_directory_name:
            raise PiCorePrefixAdmissionError("Pi core prefix directory name is invalid")
        entrypoint = verify_admitted_file(
            tree,
            pin.entrypoint_relative_path,
            expected_digest=pin.entrypoint_digest,
            expected_size_bytes=pin.entrypoint_size_bytes,
            expected_mode=pin.entrypoint_mode,
        )
    except RuntimeAssetError as error:
        raise PiCorePrefixAdmissionError(
            "Pi core prefix does not match its frozen admission"
        ) from error
    return PiCorePrefixAdmission(
        protocol=pin.protocol,
        frozen_lock_sha256=pin.frozen_lock_sha256,
        tree_protocol=tree.protocol,
        tree_digest="sha256:" + tree.digest,
        tree_entries=tree.entries,
        tree_files=tree.files,
        tree_links=tree.links,
        tree_size_bytes=tree.size_bytes,
        entrypoint_digest="sha256:" + entrypoint.digest,
        entrypoint_size_bytes=entrypoint.size_bytes,
        entrypoint_mode=entrypoint.mode,
    )
