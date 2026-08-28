"""Append-only benchmark event journal and frozen-plan helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


STRICT_JOURNAL_MAX_BYTES = 64 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def canonical_json(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()[:length]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


class Journal:
    """Durable JSONL writer. Each append is flushed before returning."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        record = {"timestamp": utc_now(), **event}
        flags = os.O_RDWR | os.O_APPEND | os.O_CLOEXEC
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        if not nofollow:
            raise OSError("secure journal append requires no-follow support")
        created = False
        try:
            descriptor = os.open(
                self.path,
                flags | nofollow | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            created = True
        except FileExistsError:
            descriptor = os.open(self.path, flags | nofollow)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.geteuid()
            ):
                raise OSError("journal must be an owned single-link regular file")
            if stat.S_IMODE(metadata.st_mode) != 0o600:
                os.fchmod(descriptor, 0o600)
            needs_separator = (
                metadata.st_size > 0
                and os.pread(descriptor, 1, metadata.st_size - 1) != b"\n"
            )
            payload = (
                (b"\n" if needs_separator else b"")
                + canonical_json(record).encode("utf-8")
                + b"\n"
            )
            written = 0
            while written < len(payload):
                count = os.write(descriptor, payload[written:])
                if count <= 0:
                    raise OSError("short journal append")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if created:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            descriptor = os.open(self.path.parent, flags)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def events(self) -> list[dict[str, Any]]:
        """Return every intact journal record, ignoring a torn crash tail."""

        events: list[dict[str, Any]] = []
        if not self.path.exists():
            return events
        for line in self.path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def strict_events(
        self, *, max_bytes: int = STRICT_JOURNAL_MAX_BYTES
    ) -> list[dict[str, Any]]:
        """Read a security-sensitive journal without filtering corruption.

        Unlike :meth:`events`, this rejects links, topology changes, duplicate
        JSON keys, empty records, malformed records, and a torn final record.
        It is intended for controller state whose filtered replay could launch
        the wrong frozen work.
        """

        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("strict journal byte bound must be a positive integer")
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        if not nofollow:
            raise OSError("strict journal read requires no-follow support")
        try:
            descriptor = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC | nofollow)
        except FileNotFoundError:
            return []
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_uid != os.geteuid()
            ):
                raise OSError("journal must be an owned single-link regular file")
            if before.st_size > max_bytes:
                raise OSError("strict journal exceeds its byte bound")
            payload = bytearray()
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(
                    descriptor,
                    min(65_536, before.st_size - offset),
                    offset,
                )
                if not chunk:
                    raise OSError("strict journal changed while being read")
                payload.extend(chunk)
                offset += len(chunk)
            after = os.fstat(descriptor)
            if (
                after.st_dev != before.st_dev
                or after.st_ino != before.st_ino
                or after.st_size != before.st_size
                or after.st_mtime_ns != before.st_mtime_ns
                or after.st_ctime_ns != before.st_ctime_ns
            ):
                raise OSError("strict journal changed while being read")
        finally:
            os.close(descriptor)

        if not payload:
            return []
        if not payload.endswith(b"\n"):
            raise ValueError("strict journal has a torn final record")

        def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(f"strict journal record repeats key {key!r}")
                value[key] = item
            return value

        events: list[dict[str, Any]] = []
        for line in bytes(payload).splitlines():
            if not line:
                raise ValueError("strict journal contains an empty record")
            try:
                event = json.loads(
                    line.decode("utf-8"), object_pairs_hook=reject_duplicate_pairs
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("strict journal contains a malformed record") from error
            if not isinstance(event, dict):
                raise ValueError("strict journal record is not an object")
            events.append(event)
        return events

    def completed_cases(self) -> set[str]:
        completed: set[str] = set()
        for event in self.events():
            if event.get("event") == "case_complete":
                completed.add(str(event["case_id"]))
        return completed

    def terminal_cases(self) -> set[str]:
        terminal = self.completed_cases()
        terminal_events = {
            "case_skipped_unsupported",
            "case_skipped_adapter_unimplemented",
            "case_skipped_context_limit",
        }
        for event in self.events():
            if event.get("event") in terminal_events:
                terminal.add(str(event["case_id"]))
        return terminal
