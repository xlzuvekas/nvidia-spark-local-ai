"""Append-only benchmark event journal and frozen-plan helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


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
        needs_separator = False
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as existing:
                existing.seek(-1, os.SEEK_END)
                needs_separator = existing.read(1) != b"\n"
        with self.path.open("a", encoding="utf-8") as stream:
            if needs_separator:
                stream.write("\n")
            stream.write(canonical_json(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

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
