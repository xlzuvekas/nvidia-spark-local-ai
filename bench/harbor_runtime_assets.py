"""Immutable runtime-tree admission for the pinned Harbor campaign.

The campaign's Node and agent prefixes are built once outside the repository,
made read-only, and pinned by a canonical full-tree digest.  This verifier is
fd-relative and never follows a path while hashing it, so a bind mount cannot
silently substitute a different file after ordinary pathname validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path, PurePosixPath
import posixpath
import stat
from typing import Iterator


TREE_PROTOCOL = "sparkbench-readonly-tree-v1"
MAX_TREE_ENTRIES = 100_000
MAX_TREE_BYTES = 4 * 1024 * 1024 * 1024
MAX_STAGED_ASSET_BYTES = 1024 * 1024


class RuntimeAssetError(RuntimeError):
    """Raised when an external runtime artifact differs from its frozen pin."""


@dataclass(frozen=True, slots=True)
class TreeAdmission:
    """Scalar-only admission for one complete immutable runtime tree."""

    protocol: str
    digest: str
    entries: int
    files: int
    links: int
    size_bytes: int
    resolved_path: Path = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class FileAdmission:
    """Scalar-only binding of one regular file inside an admitted tree."""

    digest: str
    size_bytes: int
    mode: int
    resolved_path: Path = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class LinkAdmission:
    """Exact internal relative symlink used by an admitted runtime."""

    target: str
    resolved_path: Path = field(repr=False, compare=False)
    resolved_target_path: Path = field(repr=False, compare=False)


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_external_components(path: Path, *, repo_root: Path) -> Path:
    """Resolve a real, immutable, owner-controlled tree outside the repo."""

    absolute = Path(os.path.abspath(path))
    repository = repo_root.resolve(strict=True)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            original_metadata = os.lstat(current)
        except OSError as error:
            raise RuntimeAssetError("Runtime tree component cannot be inspected") from error
        if stat.S_ISLNK(original_metadata.st_mode):
            raise RuntimeAssetError("Runtime tree path contains a symbolic link")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise RuntimeAssetError("Runtime tree does not resolve") from error
    if resolved == repository or _within(resolved, repository):
        raise RuntimeAssetError("Runtime tree must stay outside the repository")
    current = Path(resolved.anchor)
    for component in resolved.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise RuntimeAssetError("Runtime tree component cannot be inspected") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeAssetError("Runtime tree path contains a symbolic link")
        writable_by_peer = bool(metadata.st_mode & 0o022)
        sticky_root_directory = (
            stat.S_ISDIR(metadata.st_mode)
            and metadata.st_uid == 0
            and bool(metadata.st_mode & stat.S_ISVTX)
        )
        if writable_by_peer and not sticky_root_directory:
            raise RuntimeAssetError("Runtime tree path is writable by another user")
        if metadata.st_uid not in {0, os.geteuid()}:
            raise RuntimeAssetError("Runtime tree path has an unexpected owner")
    metadata = os.lstat(resolved)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise RuntimeAssetError("Runtime tree root must be an owned real directory")
    if stat.S_IMODE(metadata.st_mode) != 0o555:
        raise RuntimeAssetError("Runtime tree root must be normalized mode 0555")
    return resolved


def _safe_relative_text(value: str, *, context: str) -> str:
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as error:
        raise RuntimeAssetError(f"{context} is not UTF-8") from error
    if (
        not encoded
        or b"\0" in encoded
        or b"\n" in encoded
        or b"\r" in encoded
        or value.startswith("/")
    ):
        raise RuntimeAssetError(f"{context} is not a canonical relative path")
    return value


def _link_stays_inside(relative_path: str, target: str) -> None:
    _safe_relative_text(target, context="Runtime symlink target")
    if PurePosixPath(target).is_absolute():
        raise RuntimeAssetError("Runtime symlink target must be relative")
    combined = posixpath.normpath(
        posixpath.join(posixpath.dirname(relative_path), target)
    )
    if combined == ".." or combined.startswith("../") or combined.startswith("/"):
        raise RuntimeAssetError("Runtime symlink escapes its admitted tree")


def _same_inode(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
        and before.st_uid == after.st_uid
        and before.st_mode == after.st_mode
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
        and before.st_nlink == after.st_nlink
    )


def _same_opened_file(before: os.stat_result, opened: os.stat_result) -> bool:
    return stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1 and _same_inode(
        before, opened
    )


def _hash_regular_file(
    directory_fd: int, name: str, before: os.stat_result
) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise RuntimeAssetError("Runtime file cannot be opened safely") from error
    digest = hashlib.sha256()
    total = 0
    try:
        opened = os.fstat(descriptor)
        if not _same_opened_file(before, opened):
            raise RuntimeAssetError("Runtime file changed while opening")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > MAX_TREE_BYTES:
                raise RuntimeAssetError("Runtime file exceeds the campaign bound")
            digest.update(block)
        after = os.fstat(descriptor)
        if not _same_opened_file(opened, after) or total != opened.st_size:
            raise RuntimeAssetError("Runtime file changed while hashing")
    finally:
        os.close(descriptor)
    return total, digest.hexdigest()


def _open_child_directory(
    directory_fd: int, name: str, before: os.stat_result
) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise RuntimeAssetError("Runtime directory cannot be opened safely") from error
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not _same_inode(before, opened)
    ):
        os.close(descriptor)
        raise RuntimeAssetError("Runtime directory changed while opening")
    return descriptor


def _tree_records(
    root_fd: int, root_identity: os.stat_result
) -> Iterator[tuple[bytes, str, int, int, int]]:
    """Yield canonical records and their scalar contribution in path order."""

    def visit(
        directory_fd: int,
        parent: PurePosixPath,
        identity: os.stat_result,
    ) -> Iterator[tuple[bytes, str, int, int, int]]:
        try:
            names = os.listdir(directory_fd)
        except OSError as error:
            raise RuntimeAssetError("Runtime directory cannot be listed safely") from error
        try:
            names.sort(key=lambda item: item.encode("utf-8"))
        except UnicodeError as error:
            raise RuntimeAssetError("Runtime entry name is not UTF-8") from error
        for name in names:
            if name in {"", ".", ".."} or "/" in name:
                raise RuntimeAssetError("Runtime entry name is invalid")
            relative = (parent / name).as_posix()
            _safe_relative_text(relative, context="Runtime entry path")
            try:
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as error:
                raise RuntimeAssetError("Runtime entry cannot be inspected safely") from error
            if metadata.st_uid != os.geteuid():
                raise RuntimeAssetError("Runtime entry has an unexpected owner")
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISDIR(metadata.st_mode):
                if mode != 0o555:
                    raise RuntimeAssetError("Runtime directory mode is not normalized")
                record = f"D\0{relative}\0{mode:04o}\n".encode("utf-8")
                yield record, relative, 0, 0, 0
                child_fd = _open_child_directory(directory_fd, name, metadata)
                try:
                    yield from visit(child_fd, parent / name, os.fstat(child_fd))
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(metadata.st_mode):
                if mode not in {0o444, 0o555} or metadata.st_nlink != 1:
                    raise RuntimeAssetError("Runtime file mode or link count is unsafe")
                size, content_digest = _hash_regular_file(directory_fd, name, metadata)
                record = (
                    f"F\0{relative}\0{mode:04o}\0{size}\0{content_digest}\n"
                ).encode("utf-8")
                yield record, relative, 1, 0, size
            elif stat.S_ISLNK(metadata.st_mode):
                try:
                    target = os.readlink(name, dir_fd=directory_fd)
                except OSError as error:
                    raise RuntimeAssetError("Runtime link cannot be read safely") from error
                try:
                    after_link = os.stat(
                        name, dir_fd=directory_fd, follow_symlinks=False
                    )
                except OSError as error:
                    raise RuntimeAssetError("Runtime link changed while reading") from error
                if not _same_inode(metadata, after_link):
                    raise RuntimeAssetError("Runtime link changed while reading")
                _link_stays_inside(relative, target)
                record = f"L\0{relative}\0{target}\n".encode("utf-8")
                yield record, relative, 0, 1, 0
            else:
                raise RuntimeAssetError("Runtime tree contains a special file")

        try:
            names_after = os.listdir(directory_fd)
            after = os.fstat(directory_fd)
        except (OSError, UnicodeError) as error:
            raise RuntimeAssetError("Runtime directory changed while reading") from error
        if sorted(names_after, key=lambda item: item.encode("utf-8")) != names or not _same_inode(
            identity, after
        ):
            raise RuntimeAssetError("Runtime directory changed while reading")

    yield from visit(root_fd, PurePosixPath(), root_identity)


def inspect_normalized_tree(path: Path, *, repo_root: Path) -> TreeAdmission:
    """Hash one normalized external tree without treating it as admitted.

    Callers that create a new immutable tree need a safe way to obtain its
    scalar identity before a later policy pins it.  This remains the same
    fd-relative, no-follow traversal used by ``verify_normalized_tree``; it
    deliberately does not loosen any ownership, mode, size, or containment
    rule.
    """

    resolved = _validate_external_components(path, repo_root=repo_root)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        root_before = os.lstat(resolved)
        root_fd = os.open(resolved, flags)
    except OSError as error:
        raise RuntimeAssetError("Runtime tree root cannot be opened safely") from error
    try:
        root_opened = os.fstat(root_fd)
    except OSError as error:
        os.close(root_fd)
        raise RuntimeAssetError("Runtime tree root cannot be inspected safely") from error
    if not stat.S_ISDIR(root_opened.st_mode) or not _same_inode(
        root_before, root_opened
    ):
        os.close(root_fd)
        raise RuntimeAssetError("Runtime tree root changed while opening")
    digest = hashlib.sha256()
    entries = files = links = size_bytes = 0
    records: list[tuple[bytes, bytes, int, int, int]] = []
    try:
        for record, relative, file_count, link_count, size in _tree_records(
            root_fd, root_opened
        ):
            path_bytes = relative.encode("utf-8")
            entries += 1
            files += file_count
            links += link_count
            size_bytes += size
            if entries > MAX_TREE_ENTRIES or size_bytes > MAX_TREE_BYTES:
                raise RuntimeAssetError("Runtime tree exceeds campaign bounds")
            records.append((path_bytes, record, file_count, link_count, size))
    finally:
        os.close(root_fd)
    try:
        root_after = os.lstat(resolved)
    except OSError as error:
        raise RuntimeAssetError("Runtime tree root changed after hashing") from error
    if not _same_inode(root_opened, root_after):
        raise RuntimeAssetError("Runtime tree root changed after hashing")
    records.sort(key=lambda item: item[0])
    if len({item[0] for item in records}) != len(records):
        raise RuntimeAssetError("Runtime tree contains duplicate canonical paths")
    for _path_bytes, record, _file_count, _link_count, _size in records:
        digest.update(record)
    return TreeAdmission(
        protocol=TREE_PROTOCOL,
        digest=digest.hexdigest(),
        entries=entries,
        files=files,
        links=links,
        size_bytes=size_bytes,
        resolved_path=resolved,
    )


def verify_normalized_tree(
    path: Path,
    *,
    repo_root: Path,
    expected_digest: str,
    expected_size_bytes: int,
    expected_entries: int | None = None,
    expected_files: int | None = None,
    expected_links: int | None = None,
) -> TreeAdmission:
    """Hash one exact external tree through fd-relative, no-follow reads."""

    digest_pin = expected_digest.removeprefix("sha256:")
    count_pins = (expected_entries, expected_files, expected_links)
    if (
        len(digest_pin) != 64
        or any(character not in "0123456789abcdef" for character in digest_pin)
        or not 0 < expected_size_bytes <= MAX_TREE_BYTES
        or any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
            for value in count_pins
        )
        or expected_entries is not None
        and not 0 < expected_entries <= MAX_TREE_ENTRIES
        or expected_files is not None
        and not 0 < expected_files <= (expected_entries or MAX_TREE_ENTRIES)
        or expected_links is not None
        and not 0 <= expected_links <= (expected_entries or MAX_TREE_ENTRIES)
        or expected_entries is not None
        and expected_files is not None
        and expected_links is not None
        and expected_files + expected_links > expected_entries
    ):
        raise RuntimeAssetError("Runtime tree pin is invalid")
    admission = inspect_normalized_tree(path, repo_root=repo_root)
    if (
        admission.digest != digest_pin
        or admission.size_bytes != expected_size_bytes
        or expected_entries is not None
        and admission.entries != expected_entries
        or expected_files is not None
        and admission.files != expected_files
        or expected_links is not None
        and admission.links != expected_links
    ):
        raise RuntimeAssetError("Runtime tree does not match its frozen admission")
    return admission


def verify_admitted_file(
    tree: TreeAdmission,
    relative_path: str,
    *,
    expected_digest: str,
    expected_size_bytes: int,
    expected_mode: int = 0o555,
) -> FileAdmission:
    """Bind and hash one no-follow regular file below an admitted tree root."""

    digest_pin = expected_digest.removeprefix("sha256:")
    candidate = PurePosixPath(relative_path)
    if (
        not relative_path
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or len(digest_pin) != 64
        or any(character not in "0123456789abcdef" for character in digest_pin)
        or isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or not 0 < expected_size_bytes <= MAX_TREE_BYTES
        or expected_mode not in {0o444, 0o555}
    ):
        raise RuntimeAssetError("Admitted runtime file pin is invalid")
    root_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(tree.resolved_path, root_flags)
    except OSError as error:
        raise RuntimeAssetError("Admitted runtime root cannot be reopened") from error
    try:
        for component in candidate.parts[:-1]:
            try:
                metadata = os.stat(
                    component, dir_fd=descriptor, follow_symlinks=False
                )
            except OSError as error:
                raise RuntimeAssetError("Admitted runtime path cannot be inspected") from error
            child = _open_child_directory(descriptor, component, metadata)
            os.close(descriptor)
            descriptor = child
        try:
            before = os.stat(
                candidate.parts[-1], dir_fd=descriptor, follow_symlinks=False
            )
        except OSError as error:
            raise RuntimeAssetError("Admitted runtime file cannot be inspected") from error
        size_bytes, digest = _hash_regular_file(
            descriptor, candidate.parts[-1], before
        )
    finally:
        os.close(descriptor)
    if (
        stat.S_IMODE(before.st_mode) != expected_mode
        or size_bytes != expected_size_bytes
        or digest != digest_pin
    ):
        raise RuntimeAssetError("Admitted runtime file differs from its frozen pin")
    return FileAdmission(
        digest=digest,
        size_bytes=size_bytes,
        mode=expected_mode,
        resolved_path=tree.resolved_path.joinpath(*candidate.parts),
    )


def verify_admitted_symlink(
    tree: TreeAdmission, relative_path: str, *, expected_target: str
) -> LinkAdmission:
    """Bind one exact relative, internal symlink below an admitted tree."""

    candidate = PurePosixPath(relative_path)
    if (
        not relative_path
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise RuntimeAssetError("Admitted runtime link path is invalid")
    _link_stays_inside(relative_path, expected_target)
    combined = posixpath.normpath(
        posixpath.join(posixpath.dirname(relative_path), expected_target)
    )
    root_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(tree.resolved_path, root_flags)
    except OSError as error:
        raise RuntimeAssetError("Admitted runtime root cannot be reopened") from error
    try:
        for component in candidate.parts[:-1]:
            try:
                metadata = os.stat(
                    component, dir_fd=descriptor, follow_symlinks=False
                )
            except OSError as error:
                raise RuntimeAssetError("Admitted runtime link path cannot be inspected") from error
            child = _open_child_directory(descriptor, component, metadata)
            os.close(descriptor)
            descriptor = child
        name = candidate.parts[-1]
        try:
            before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            actual_target = os.readlink(name, dir_fd=descriptor)
            after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as error:
            raise RuntimeAssetError("Admitted runtime link cannot be read safely") from error
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISLNK(before.st_mode)
        or before.st_uid != os.geteuid()
        or not _same_inode(before, after)
        or actual_target != expected_target
    ):
        raise RuntimeAssetError("Admitted runtime link differs from its frozen pin")
    return LinkAdmission(
        target=actual_target,
        resolved_path=tree.resolved_path.joinpath(*candidate.parts),
        resolved_target_path=tree.resolved_path.joinpath(*PurePosixPath(combined).parts),
    )


def stage_immutable_asset(
    source: Path,
    destination: Path,
    *,
    repo_root: Path,
    expected_digest: str,
    expected_source_mode: int,
    output_mode: int,
) -> Path:
    """Copy one pinned tracked asset into a new external immutable file."""

    digest_pin = expected_digest.removeprefix("sha256:")
    if (
        len(digest_pin) != 64
        or any(character not in "0123456789abcdef" for character in digest_pin)
        or expected_source_mode not in {0o644, 0o755}
        or output_mode not in {0o444, 0o555}
    ):
        raise RuntimeAssetError("Runtime asset pin is invalid")
    repository = repo_root.resolve(strict=True)
    source_absolute = Path(os.path.abspath(source))
    try:
        source_resolved = source_absolute.resolve(strict=True)
    except OSError as error:
        raise RuntimeAssetError("Runtime asset source does not resolve") from error
    if source_absolute != source_resolved or not _within(source_resolved, repository):
        raise RuntimeAssetError("Runtime asset source must be a canonical repo file")
    try:
        before = os.lstat(source_resolved)
    except OSError as error:
        raise RuntimeAssetError("Runtime asset source cannot be inspected") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != expected_source_mode
        or not 0 < before.st_size <= MAX_STAGED_ASSET_BYTES
    ):
        raise RuntimeAssetError("Runtime asset source is not a pinned regular file")
    source_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        source_fd = os.open(source_resolved, source_flags)
    except OSError as error:
        raise RuntimeAssetError("Runtime asset source cannot be opened safely") from error
    try:
        opened = os.fstat(source_fd)
        if not _same_opened_file(before, opened):
            raise RuntimeAssetError("Runtime asset source changed while opening")
        payload = _read_fd_bounded(source_fd, MAX_STAGED_ASSET_BYTES)
        after = os.fstat(source_fd)
        if (
            not _same_opened_file(opened, after)
            or len(payload) != opened.st_size
            or hashlib.sha256(payload).hexdigest() != digest_pin
        ):
            raise RuntimeAssetError("Runtime asset source differs from its pin")
    finally:
        os.close(source_fd)

    candidate = Path(os.path.abspath(destination))
    parent = candidate.parent
    try:
        parent_resolved = parent.resolve(strict=True)
        parent_metadata = os.lstat(parent_resolved)
    except OSError as error:
        raise RuntimeAssetError("Runtime asset destination parent is unavailable") from error
    if (
        parent != parent_resolved
        or parent_resolved == repository
        or _within(parent_resolved, repository)
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) != 0o700
        or candidate.exists()
        or candidate.is_symlink()
    ):
        raise RuntimeAssetError("Runtime asset destination is not isolated")
    output_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        output_fd = os.open(candidate, output_flags, output_mode)
    except OSError as error:
        raise RuntimeAssetError("Runtime asset destination cannot be created") from error
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(output_fd, view[written:])
            if count <= 0:
                raise RuntimeAssetError("Runtime asset destination short write")
            written += count
        os.fchmod(output_fd, output_mode)
        os.fsync(output_fd)
        output_metadata = os.fstat(output_fd)
        if (
            not stat.S_ISREG(output_metadata.st_mode)
            or output_metadata.st_uid != os.geteuid()
            or output_metadata.st_nlink != 1
            or stat.S_IMODE(output_metadata.st_mode) != output_mode
            or output_metadata.st_size != len(payload)
        ):
            raise RuntimeAssetError("Runtime asset destination admission failed")
    except BaseException:
        os.close(output_fd)
        candidate.unlink(missing_ok=True)
        raise
    else:
        os.close(output_fd)
    return candidate


def _read_fd_bounded(descriptor: int, limit: int) -> bytes:
    blocks: list[bytes] = []
    total = 0
    while True:
        block = os.read(descriptor, min(64 * 1024, limit + 1 - total))
        if not block:
            return b"".join(blocks)
        blocks.append(block)
        total += len(block)
        if total > limit:
            raise RuntimeAssetError("Runtime asset exceeds its size bound")
