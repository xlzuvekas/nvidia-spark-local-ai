"""Durable evidence and Git acknowledgements between autoresearch pairs.

The execution controller owns campaign replay and measurement auditing.  This
module owns the narrower boundary proof: a completed pair may be followed by
another pair only after its scalar evidence is unchanged, verified, committed,
and visible on the configured upstream.  The acknowledgement itself lives in
ignored ``logs/`` state and contains hashes and public identifiers only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
from typing import Any, Callable, Mapping, Sequence


ACK_SCHEMA_VERSION = "sparkbench-autoresearch-checkpoint-v1"
ACK_DIRECTORY = Path("logs") / "autoresearch-checkpoints"
MAX_ACK_BYTES = 128 * 1024
MAX_JOURNAL_BYTES = 4 * 1024 * 1024
MAX_EVIDENCE_CONTROL_BYTES = 4 * 1024 * 1024
MAX_CHECKPOINT_SEQUENCE = 7

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_OBJECT_ID_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,255}\Z")
_PUBLISHED_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}\Z")
_REMOTE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_ACK_KEYS = frozenset(
    {
        "schema_version",
        "campaign",
        "completion",
        "pair_state_sha256",
        "journal_prefix",
        "evidence",
        "repository",
        "acknowledged_at",
        "integrity_hash",
    }
)


class CheckpointError(RuntimeError):
    """A typed, non-secret checkpoint admission failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    if set(value) != expected:
        raise CheckpointError("checkpoint_state_invalid", f"{name} schema changed")


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CheckpointError("checkpoint_state_invalid", f"{name} is not SHA-256")
    return value


def _object_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or _OBJECT_ID_RE.fullmatch(value) is None:
        raise CheckpointError("checkpoint_state_invalid", f"{name} is not a Git object ID")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise CheckpointError("checkpoint_state_invalid", f"{name} is unsafe")
    return value


def _published_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _PUBLISHED_ID_RE.fullmatch(value) is None:
        raise CheckpointError("checkpoint_state_invalid", f"{name} is unsafe")
    return value


def _integer(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise CheckpointError("checkpoint_state_invalid", f"{name} is invalid")
    return value


def _git_ref(value: Any, name: str, *, prefix: str) -> str:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise CheckpointError("checkpoint_state_invalid", f"{name} is invalid")
    if (
        len(value) > 255
        or value.endswith(("/", ".", ".lock"))
        or "//" in value
        or ".." in value
        or "@{" in value
        or "\\" in value
        or any(ord(character) < 32 or character in " ~^:?*[" for character in value)
    ):
        raise CheckpointError("checkpoint_state_invalid", f"{name} is unsafe")
    return value


def _timestamp(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 64:
        raise CheckpointError("checkpoint_state_invalid", "acknowledgement time is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise CheckpointError(
            "checkpoint_state_invalid", "acknowledgement time is invalid"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CheckpointError(
            "checkpoint_state_invalid", "acknowledgement time is not timezone-aware"
        )
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(
                "checkpoint_state_invalid", f"duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _parse_object(
    data: bytes, *, name: str, code: str = "checkpoint_state_invalid"
) -> dict[str, Any]:
    try:
        value = json.loads(data, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError(code, f"{name} is invalid JSON") from error
    if not isinstance(value, dict):
        raise CheckpointError(code, f"{name} must be an object")
    return value


@dataclass(frozen=True, slots=True)
class CampaignBinding:
    campaign_id: str
    campaign_integrity_sha256: str
    preview_sha256: str
    policy_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, "campaign ID")
        _sha256(self.campaign_integrity_sha256, "campaign integrity")
        _sha256(self.preview_sha256, "campaign preview")
        _sha256(self.policy_sha256, "campaign policy")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "campaign_integrity_sha256": self.campaign_integrity_sha256,
            "policy_sha256": self.policy_sha256,
            "preview_sha256": self.preview_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "CampaignBinding":
        if not isinstance(value, dict):
            raise CheckpointError("checkpoint_state_invalid", "campaign binding is invalid")
        _exact_keys(
            value,
            frozenset(
                {
                    "campaign_id",
                    "campaign_integrity_sha256",
                    "preview_sha256",
                    "policy_sha256",
                }
            ),
            "campaign binding",
        )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class PairCompletion:
    """Scalar identity of the latest pair, after controller-side re-audit."""

    sequence: int
    pair_kind: str
    candidate_id: str
    search_pair_index: int | None
    ordered_cell_ids: tuple[str, str]
    ordered_evidence_run_ids: tuple[str, str]
    cell_plan_integrity_sha256s: tuple[str, str]
    observation_sha256: str

    def __post_init__(self) -> None:
        _integer(
            self.sequence,
            "checkpoint sequence",
            minimum=1,
            maximum=MAX_CHECKPOINT_SEQUENCE,
        )
        if self.pair_kind not in {"calibration", "screen", "confirmation"}:
            raise CheckpointError("checkpoint_state_invalid", "pair kind is invalid")
        _identifier(self.candidate_id, "candidate ID")
        if len(self.ordered_cell_ids) != 2 or len(set(self.ordered_cell_ids)) != 2:
            raise CheckpointError("checkpoint_state_invalid", "pair cell IDs must be unique")
        if len(self.ordered_evidence_run_ids) != 2 or len(
            set(self.ordered_evidence_run_ids)
        ) != 2:
            raise CheckpointError(
                "checkpoint_state_invalid", "pair evidence run IDs must be unique"
            )
        for value in self.ordered_cell_ids:
            _identifier(value, "cell ID")
        for value in self.ordered_evidence_run_ids:
            _published_identifier(value, "published evidence run ID")
        if len(self.cell_plan_integrity_sha256s) != 2:
            raise CheckpointError(
                "checkpoint_state_invalid", "pair plan integrities are incomplete"
            )
        for value in self.cell_plan_integrity_sha256s:
            _sha256(value, "cell plan integrity")
        _sha256(self.observation_sha256, "pair observation")
        if self.pair_kind == "calibration":
            if (
                self.sequence != 1
                or self.candidate_id != "control"
                or self.search_pair_index is not None
            ):
                raise CheckpointError(
                    "checkpoint_state_invalid", "calibration completion is inconsistent"
                )
        else:
            expected_index = self.sequence - 2
            if (
                self.candidate_id == "control"
                or isinstance(self.search_pair_index, bool)
                or self.search_pair_index != expected_index
            ):
                raise CheckpointError(
                    "checkpoint_state_invalid", "search completion index is inconsistent"
                )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "cell_plan_integrity_sha256s": list(
                self.cell_plan_integrity_sha256s
            ),
            "observation_sha256": self.observation_sha256,
            "ordered_cell_ids": list(self.ordered_cell_ids),
            "ordered_evidence_run_ids": list(self.ordered_evidence_run_ids),
            "pair_kind": self.pair_kind,
            "search_pair_index": self.search_pair_index,
            "sequence": self.sequence,
        }

    @property
    def digest(self) -> str:
        return _content_sha256(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: Any) -> "PairCompletion":
        if not isinstance(value, dict):
            raise CheckpointError("checkpoint_state_invalid", "pair completion is invalid")
        _exact_keys(
            value,
            frozenset(
                {
                    "candidate_id",
                    "cell_plan_integrity_sha256s",
                    "observation_sha256",
                    "ordered_cell_ids",
                    "ordered_evidence_run_ids",
                    "pair_kind",
                    "search_pair_index",
                    "sequence",
                }
            ),
            "pair completion",
        )
        for key in (
            "ordered_cell_ids",
            "ordered_evidence_run_ids",
            "cell_plan_integrity_sha256s",
        ):
            raw = value[key]
            if not isinstance(raw, list) or len(raw) != 2:
                raise CheckpointError(
                    "checkpoint_state_invalid", f"pair completion {key} is invalid"
                )
        return cls(
            sequence=value["sequence"],
            pair_kind=value["pair_kind"],
            candidate_id=value["candidate_id"],
            search_pair_index=value["search_pair_index"],
            ordered_cell_ids=tuple(value["ordered_cell_ids"]),
            ordered_evidence_run_ids=tuple(value["ordered_evidence_run_ids"]),
            cell_plan_integrity_sha256s=tuple(
                value["cell_plan_integrity_sha256s"]
            ),
            observation_sha256=value["observation_sha256"],
        )


@dataclass(frozen=True, slots=True)
class JournalPrefix:
    event_count: int
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _integer(self.event_count, "journal event count", minimum=1, maximum=10_000)
        _integer(
            self.size_bytes,
            "journal prefix size",
            minimum=1,
            maximum=MAX_JOURNAL_BYTES,
        )
        _sha256(self.sha256, "journal prefix")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "event_count": self.event_count,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "JournalPrefix":
        if not isinstance(value, dict):
            raise CheckpointError("checkpoint_state_invalid", "journal prefix is invalid")
        _exact_keys(
            value,
            frozenset({"event_count", "sha256", "size_bytes"}),
            "journal prefix",
        )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class EvidenceCell:
    cell_id: str
    published_run_id: str
    bundle_sha256: str
    status: str
    measurement_terminal: bool

    def __post_init__(self) -> None:
        _identifier(self.cell_id, "evidence cell ID")
        _published_identifier(self.published_run_id, "published evidence run ID")
        _sha256(self.bundle_sha256, "evidence bundle")
        if self.status != "complete" or self.measurement_terminal is not True:
            raise CheckpointError(
                "evidence_incomplete", "pair evidence is not measurement-terminal"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "bundle_sha256": self.bundle_sha256,
            "cell_id": self.cell_id,
            "measurement_terminal": self.measurement_terminal,
            "published_run_id": self.published_run_id,
            "status": self.status,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "EvidenceCell":
        if not isinstance(value, dict):
            raise CheckpointError("checkpoint_state_invalid", "evidence cell is invalid")
        _exact_keys(
            value,
            frozenset(
                {
                    "cell_id",
                    "published_run_id",
                    "bundle_sha256",
                    "status",
                    "measurement_terminal",
                }
            ),
            "evidence cell",
        )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class EvidenceProof:
    index_sha256: str
    checksums_sha256: str
    cells: tuple[EvidenceCell, EvidenceCell]

    def __post_init__(self) -> None:
        _sha256(self.index_sha256, "evidence index")
        _sha256(self.checksums_sha256, "evidence checksums")
        if len({cell.cell_id for cell in self.cells}) != 2 or len(
            {cell.published_run_id for cell in self.cells}
        ) != 2:
            raise CheckpointError(
                "checkpoint_state_invalid", "evidence cells must be unique"
            )

    def require_completion(self, completion: PairCompletion) -> None:
        if tuple(cell.cell_id for cell in self.cells) != completion.ordered_cell_ids:
            raise CheckpointError(
                "evidence_pair_binding_mismatch", "evidence cell order changed"
            )
        if tuple(
            cell.published_run_id for cell in self.cells
        ) != completion.ordered_evidence_run_ids:
            raise CheckpointError(
                "evidence_pair_binding_mismatch", "published evidence identity changed"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "cells": [cell.to_mapping() for cell in self.cells],
            "checksums_sha256": self.checksums_sha256,
            "index_sha256": self.index_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "EvidenceProof":
        if not isinstance(value, dict):
            raise CheckpointError("checkpoint_state_invalid", "evidence proof is invalid")
        _exact_keys(
            value,
            frozenset({"cells", "checksums_sha256", "index_sha256"}),
            "evidence proof",
        )
        raw_cells = value["cells"]
        if not isinstance(raw_cells, list) or len(raw_cells) != 2:
            raise CheckpointError("checkpoint_state_invalid", "evidence cells are invalid")
        return cls(
            index_sha256=value["index_sha256"],
            checksums_sha256=value["checksums_sha256"],
            cells=tuple(EvidenceCell.from_mapping(item) for item in raw_cells),
        )


@dataclass(frozen=True, slots=True)
class RepositoryProof:
    head_commit: str
    local_branch_ref: str
    upstream_ref: str
    upstream_commit: str
    remote_name: str
    remote_ref: str
    remote_commit: str

    def __post_init__(self) -> None:
        _object_id(self.head_commit, "HEAD commit")
        _git_ref(self.local_branch_ref, "local branch", prefix="refs/heads/")
        _git_ref(self.upstream_ref, "upstream ref", prefix="refs/remotes/")
        _object_id(self.upstream_commit, "upstream commit")
        if (
            not isinstance(self.remote_name, str)
            or _REMOTE_RE.fullmatch(self.remote_name) is None
            or self.remote_name in {".", ".."}
        ):
            raise CheckpointError("checkpoint_state_invalid", "remote name is unsafe")
        _git_ref(self.remote_ref, "remote ref", prefix="refs/heads/")
        _object_id(self.remote_commit, "remote commit")
        if not (
            self.head_commit == self.upstream_commit == self.remote_commit
        ):
            raise CheckpointError(
                "repository_not_pushed", "HEAD is not identical to its live upstream"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "head_commit": self.head_commit,
            "local_branch_ref": self.local_branch_ref,
            "remote_commit": self.remote_commit,
            "remote_name": self.remote_name,
            "remote_ref": self.remote_ref,
            "upstream_commit": self.upstream_commit,
            "upstream_ref": self.upstream_ref,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "RepositoryProof":
        if not isinstance(value, dict):
            raise CheckpointError("checkpoint_state_invalid", "repository proof is invalid")
        _exact_keys(
            value,
            frozenset(
                {
                    "head_commit",
                    "local_branch_ref",
                    "remote_commit",
                    "remote_name",
                    "remote_ref",
                    "upstream_commit",
                    "upstream_ref",
                }
            ),
            "repository proof",
        )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class CheckpointAcknowledgement:
    campaign: CampaignBinding
    completion: PairCompletion
    journal_prefix: JournalPrefix
    evidence: EvidenceProof
    repository: RepositoryProof
    acknowledged_at: str

    def __post_init__(self) -> None:
        _timestamp(self.acknowledged_at)
        self.evidence.require_completion(self.completion)

    def to_mapping(self, *, include_integrity: bool = True) -> dict[str, Any]:
        result = {
            "acknowledged_at": self.acknowledged_at,
            "campaign": self.campaign.to_mapping(),
            "completion": self.completion.to_mapping(),
            "evidence": self.evidence.to_mapping(),
            "journal_prefix": self.journal_prefix.to_mapping(),
            "pair_state_sha256": self.completion.digest,
            "repository": self.repository.to_mapping(),
            "schema_version": ACK_SCHEMA_VERSION,
        }
        if include_integrity:
            result["integrity_hash"] = _content_sha256(result)
        return result

    @classmethod
    def from_mapping(cls, value: Any) -> "CheckpointAcknowledgement":
        if not isinstance(value, dict):
            raise CheckpointError("checkpoint_state_invalid", "checkpoint must be an object")
        _exact_keys(value, _ACK_KEYS, "checkpoint")
        if value["schema_version"] != ACK_SCHEMA_VERSION:
            raise CheckpointError(
                "checkpoint_state_invalid", "checkpoint schema version changed"
            )
        integrity = _sha256(value["integrity_hash"], "checkpoint integrity")
        payload = {key: item for key, item in value.items() if key != "integrity_hash"}
        if _content_sha256(payload) != integrity:
            raise CheckpointError(
                "checkpoint_state_invalid", "checkpoint integrity does not match"
            )
        completion = PairCompletion.from_mapping(value["completion"])
        if value["pair_state_sha256"] != completion.digest:
            raise CheckpointError(
                "checkpoint_state_invalid", "pair-state digest does not match"
            )
        try:
            return cls(
                campaign=CampaignBinding.from_mapping(value["campaign"]),
                completion=completion,
                journal_prefix=JournalPrefix.from_mapping(value["journal_prefix"]),
                evidence=EvidenceProof.from_mapping(value["evidence"]),
                repository=RepositoryProof.from_mapping(value["repository"]),
                acknowledged_at=value["acknowledged_at"],
            )
        except CheckpointError as error:
            if error.code == "checkpoint_state_invalid":
                raise
            raise CheckpointError(
                "checkpoint_state_invalid", "checkpoint contains an invalid proof"
            ) from error


@dataclass(frozen=True, slots=True)
class CheckpointGate:
    status: str
    reason: str | None
    sequence: int

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "checkpoint_reason": self.reason,
            "checkpoint_sequence": self.sequence,
            "status": self.status,
        }


def _secure_read(
    path: Path,
    *,
    maximum: int,
    code: str,
    required_mode: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise CheckpointError(code, "no-follow file access is unavailable")
    directory = _open_directory_nofollow(path.parent, code=code)
    try:
        descriptor = os.open(path.name, flags | nofollow, dir_fd=directory)
    except OSError as error:
        os.close(directory)
        raise CheckpointError(code, "checkpoint input is unreadable") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
            or metadata.st_size > maximum
            or (
                required_mode is not None
                and stat.S_IMODE(metadata.st_mode) != required_mode
            )
        ):
            raise CheckpointError(code, "checkpoint input is unsafe")
        data = bytearray()
        while True:
            chunk = os.read(descriptor, min(65_536, maximum + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > maximum:
                raise CheckpointError(code, "checkpoint input is too large")
        return bytes(data)
    finally:
        os.close(descriptor)
        os.close(directory)


def _open_directory_nofollow(path: Path, *, code: str) -> int:
    """Open every directory component without following a symlink."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise CheckpointError(code, "no-follow directory access is unavailable")
    absolute = Path(path).absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | nofollow
    try:
        descriptor = os.open(absolute.anchor or os.sep, flags)
        for component in absolute.parts[1:]:
            next_descriptor = os.open(
                component, flags, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise OSError("path component is not a directory")
        return descriptor
    except OSError as error:
        try:
            os.close(descriptor)
        except (NameError, OSError):
            pass
        raise CheckpointError(code, "directory path is unsafe") from error


def _journal_events(data: bytes) -> tuple[dict[str, Any], ...]:
    if not data or not data.endswith(b"\n"):
        raise CheckpointError(
            "journal_prefix_invalid", "campaign journal is empty or has a torn tail"
        )
    lines = data.splitlines()
    events: list[dict[str, Any]] = []
    for line in lines:
        if not line:
            raise CheckpointError("journal_prefix_invalid", "campaign journal has a blank line")
        try:
            event = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CheckpointError(
                "journal_prefix_invalid", "campaign journal contains invalid JSON"
            ) from error
        if not isinstance(event, dict):
            raise CheckpointError(
                "journal_prefix_invalid", "campaign journal event is not an object"
            )
        events.append(event)
    return tuple(events)


def capture_journal_prefix(path: Path) -> JournalPrefix:
    data = _secure_read(
        Path(path), maximum=MAX_JOURNAL_BYTES, code="journal_prefix_invalid"
    )
    events = _journal_events(data)
    return JournalPrefix(len(events), len(data), _bytes_sha256(data))


def require_journal_prefix(path: Path, prefix: JournalPrefix) -> None:
    data = _secure_read(
        Path(path), maximum=MAX_JOURNAL_BYTES, code="journal_prefix_invalid"
    )
    events = _journal_events(data)
    if (
        len(data) < prefix.size_bytes
        or _bytes_sha256(data[: prefix.size_bytes]) != prefix.sha256
        or len(_journal_events(data[: prefix.size_bytes])) != prefix.event_count
        or len(events) < prefix.event_count
    ):
        raise CheckpointError(
            "journal_prefix_changed", "campaign journal no longer has the acknowledged prefix"
        )


def checkpoint_state_path(
    workspace: Path,
    campaign: CampaignBinding,
    *,
    state_root: Path | None = None,
) -> Path:
    root = (
        Path(state_root)
        if state_root is not None
        else Path(workspace) / ACK_DIRECTORY
    )
    return root / f"{campaign.campaign_integrity_sha256}.json"


def autoresearch_published_run_id(
    *,
    campaign_id: str,
    cell_id: str,
    ordinal: int,
    created_at: str,
) -> str:
    """Return the evidence exporter's public ID for one frozen campaign cell."""

    _identifier(campaign_id, "campaign ID")
    _identifier(cell_id, "cell ID")
    _integer(ordinal, "cell ordinal", minimum=1, maximum=14)
    _timestamp(created_at)
    from .evidence import _autoresearch_published_run_id

    value = _autoresearch_published_run_id(
        campaign_id=campaign_id,
        cell_id=cell_id,
        ordinal=ordinal,
        created_at=created_at,
    )
    return _published_identifier(value, "published evidence run ID")


def _prepare_state_directory(path: Path) -> None:
    parent = path.parent
    grandparent = parent.parent
    try:
        grandparent_descriptor = _open_directory_nofollow(
            grandparent, code="checkpoint_state_invalid"
        )
    except CheckpointError:
        ancestor = _open_directory_nofollow(
            grandparent.parent, code="checkpoint_state_invalid"
        )
        grandparent_created = False
        try:
            os.mkdir(grandparent.name, 0o700, dir_fd=ancestor)
            grandparent_created = True
        except FileExistsError:
            pass
        finally:
            if grandparent_created:
                os.fsync(ancestor)
            os.close(ancestor)
        grandparent_descriptor = _open_directory_nofollow(
            grandparent, code="checkpoint_state_invalid"
        )
    try:
        grandparent_metadata = os.fstat(grandparent_descriptor)
        if grandparent_metadata.st_uid != os.geteuid():
            raise CheckpointError(
                "checkpoint_state_invalid", "checkpoint parent directory is unsafe"
            )
        parent_created = False
        try:
            os.mkdir(parent.name, 0o700, dir_fd=grandparent_descriptor)
            parent_created = True
        except FileExistsError:
            pass
        if parent_created:
            os.fsync(grandparent_descriptor)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            parent.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | nofollow,
            dir_fd=grandparent_descriptor,
        )
    except (OSError, CheckpointError) as error:
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError(
            "checkpoint_state_invalid", "checkpoint directory is unsafe"
        ) from error
    finally:
        os.close(grandparent_descriptor)
    try:
        final_metadata = os.fstat(descriptor)
        if (
            final_metadata.st_uid != os.geteuid()
            or not stat.S_ISDIR(final_metadata.st_mode)
        ):
            raise CheckpointError(
                "checkpoint_state_invalid", "checkpoint directory is not private"
            )
        if stat.S_IMODE(final_metadata.st_mode) != 0o700:
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_acknowledgement(path: Path) -> CheckpointAcknowledgement | None:
    path = Path(path)
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    data = _secure_read(
        path,
        maximum=MAX_ACK_BYTES,
        code="checkpoint_state_invalid",
        required_mode=0o600,
    )
    return CheckpointAcknowledgement.from_mapping(
        _parse_object(data, name="checkpoint")
    )


def _write_acknowledgement(path: Path, acknowledgement: CheckpointAcknowledgement) -> None:
    _prepare_state_directory(path)
    parent = path.parent
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise CheckpointError(
            "checkpoint_state_invalid", "no-follow file access is unavailable"
        )
    directory = _open_directory_nofollow(
        parent, code="checkpoint_state_invalid"
    )
    temporary_name = f".{path.name}.{secrets.token_hex(8)}.tmp"
    descriptor = -1
    try:
        try:
            current = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or current.st_uid != os.geteuid()
            or stat.S_IMODE(current.st_mode) != 0o600
        ):
            raise CheckpointError(
                "checkpoint_state_invalid", "existing checkpoint file is unsafe"
            )
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | nofollow,
            0o600,
            dir_fd=directory,
        )
        payload = json.dumps(
            acknowledgement.to_mapping(), indent=2, sort_keys=True
        ).encode("utf-8") + b"\n"
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
        )
        os.fsync(directory)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(directory)


GitRunner = Callable[[Path, tuple[str, ...]], bytes]


def _run_git(workspace: Path, arguments: tuple[str, ...]) -> bytes:
    try:
        process = subprocess.run(
            ["git", *arguments],
            cwd=workspace,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CheckpointError("remote_unverified", "Git proof could not be obtained") from error
    if process.returncode != 0:
        raise CheckpointError("remote_unverified", "Git proof could not be obtained")
    return process.stdout


def _git_text(
    workspace: Path,
    arguments: Sequence[str],
    *,
    runner: GitRunner,
    code: str,
) -> str:
    try:
        data = runner(workspace, tuple(arguments))
        value = data.decode("utf-8").strip()
    except (CheckpointError, UnicodeDecodeError) as error:
        if isinstance(error, CheckpointError) and error.code == code:
            raise
        raise CheckpointError(code, "Git proof could not be obtained") from error
    if not value:
        raise CheckpointError(code, "Git proof is empty")
    return value


def prove_repository(
    workspace: Path,
    *,
    git_runner: GitRunner = _run_git,
) -> RepositoryProof:
    """Prove an attached, clean HEAD is identical to its live remote branch."""

    workspace = Path(workspace).resolve(strict=True)
    try:
        status = git_runner(
            workspace,
            ("status", "--porcelain=v2", "-z", "--untracked-files=all"),
        )
    except CheckpointError as error:
        raise CheckpointError(
            "repository_unverified", "repository status is unavailable"
        ) from error
    if status:
        raise CheckpointError("repository_dirty", "repository is not clean")
    local_ref = _git_text(
        workspace,
        ("symbolic-ref", "-q", "HEAD"),
        runner=git_runner,
        code="repository_detached",
    )
    _git_ref(local_ref, "local branch", prefix="refs/heads/")
    head = _object_id(
        _git_text(
            workspace,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            runner=git_runner,
            code="repository_unverified",
        ),
        "HEAD commit",
    )
    branch_name = local_ref.removeprefix("refs/heads/")
    remote_name = _git_text(
        workspace,
        ("config", "--get", f"branch.{branch_name}.remote"),
        runner=git_runner,
        code="upstream_missing",
    )
    if _REMOTE_RE.fullmatch(remote_name) is None or remote_name in {".", ".."}:
        raise CheckpointError("upstream_missing", "configured upstream remote is unsafe")
    remote_ref = _git_text(
        workspace,
        ("config", "--get", f"branch.{branch_name}.merge"),
        runner=git_runner,
        code="upstream_missing",
    )
    _git_ref(remote_ref, "remote ref", prefix="refs/heads/")
    upstream_ref = _git_text(
        workspace,
        ("rev-parse", "--symbolic-full-name", "@{upstream}"),
        runner=git_runner,
        code="upstream_missing",
    )
    _git_ref(upstream_ref, "upstream ref", prefix="refs/remotes/")
    upstream = _object_id(
        _git_text(
            workspace,
            ("rev-parse", "--verify", "@{upstream}^{commit}"),
            runner=git_runner,
            code="upstream_missing",
        ),
        "upstream commit",
    )
    if head != upstream:
        raise CheckpointError(
            "repository_not_pushed", "HEAD differs from its local upstream"
        )
    remote_output = _git_text(
        workspace,
        ("ls-remote", "--exit-code", remote_name, remote_ref),
        runner=git_runner,
        code="remote_unverified",
    )
    remote_records = remote_output.splitlines()
    if len(remote_records) != 1:
        raise CheckpointError("remote_unverified", "live remote proof is ambiguous")
    try:
        remote_commit, returned_ref = remote_records[0].split("\t", 1)
    except ValueError as error:
        raise CheckpointError("remote_unverified", "live remote proof is malformed") from error
    _object_id(remote_commit, "remote commit")
    if returned_ref != remote_ref or remote_commit != head:
        raise CheckpointError(
            "repository_not_pushed", "HEAD differs from the live remote branch"
        )
    try:
        final_status = git_runner(
            workspace,
            ("status", "--porcelain=v2", "-z", "--untracked-files=all"),
        )
    except CheckpointError as error:
        raise CheckpointError(
            "repository_unverified", "repository status is unavailable"
        ) from error
    final_head = _object_id(
        _git_text(
            workspace,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            runner=git_runner,
            code="repository_unverified",
        ),
        "final HEAD commit",
    )
    if final_status or final_head != head:
        raise CheckpointError(
            "checkpoint_race", "repository changed while its proof was collected"
        )
    return RepositoryProof(
        head_commit=head,
        local_branch_ref=local_ref,
        upstream_ref=upstream_ref,
        upstream_commit=upstream,
        remote_name=remote_name,
        remote_ref=remote_ref,
        remote_commit=remote_commit,
    )


def _read_evidence_proof(
    completion: PairCompletion,
    *,
    evidence_root: Path,
) -> EvidenceProof:
    index_bytes = _secure_read(
        Path(evidence_root) / "index.json",
        maximum=MAX_EVIDENCE_CONTROL_BYTES,
        code="evidence_invalid",
    )
    checksums_bytes = _secure_read(
        Path(evidence_root) / "checksums.json",
        maximum=MAX_EVIDENCE_CONTROL_BYTES,
        code="evidence_invalid",
    )
    index = _parse_object(
        index_bytes, name="evidence index", code="evidence_invalid"
    )
    runs = index.get("runs")
    if not isinstance(runs, list):
        raise CheckpointError("evidence_invalid", "evidence run index is missing")
    by_id: dict[str, dict[str, Any]] = {}
    for raw in runs:
        if not isinstance(raw, dict):
            raise CheckpointError("evidence_invalid", "evidence run entry is invalid")
        run_id = raw.get("run_id")
        if not isinstance(run_id, str) or run_id in by_id:
            raise CheckpointError("evidence_invalid", "evidence run IDs are invalid")
        by_id[run_id] = raw
    cells: list[EvidenceCell] = []
    for cell_id, run_id in zip(
        completion.ordered_cell_ids,
        completion.ordered_evidence_run_ids,
        strict=True,
    ):
        entry = by_id.get(run_id)
        if entry is None:
            raise CheckpointError(
                "evidence_pair_binding_mismatch", "completed pair is absent from evidence"
            )
        cells.append(
            EvidenceCell(
                cell_id=cell_id,
                published_run_id=run_id,
                bundle_sha256=entry.get("bundle_sha256"),
                status=entry.get("status"),
                measurement_terminal=entry.get("measurement_terminal"),
            )
        )
    proof = EvidenceProof(
        index_sha256=_bytes_sha256(index_bytes),
        checksums_sha256=_bytes_sha256(checksums_bytes),
        cells=tuple(cells),
    )
    proof.require_completion(completion)
    return proof


def prove_evidence(
    completion: PairCompletion,
    *,
    workspace: Path,
    results_root: Path | None = None,
    evidence_root: Path | None = None,
    exporter: Callable[..., Mapping[str, Any]] | None = None,
    verifier: Callable[[Path], Mapping[str, Any]] | None = None,
    staged_verifier: Callable[..., Mapping[str, Any]] | None = None,
) -> EvidenceProof:
    """Re-export without replacement, then verify working and Git-index evidence."""

    if exporter is None or verifier is None or staged_verifier is None:
        from .evidence import export_evidence, verify_evidence, verify_staged_evidence

        exporter = exporter or export_evidence
        verifier = verifier or verify_evidence
        staged_verifier = staged_verifier or verify_staged_evidence
    workspace = Path(workspace).resolve(strict=True)
    results = Path(results_root) if results_root is not None else workspace / "results"
    evidence = Path(evidence_root) if evidence_root is not None else workspace / "evidence"
    try:
        export_report = exporter(
            results_root=results,
            output_root=evidence,
            replace=False,
        )
    except CheckpointError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise CheckpointError(
            "evidence_not_current", "evidence does not match current results"
        ) from error
    if export_report.get("changed") is not False:
        raise CheckpointError(
            "evidence_not_current", "tracked evidence does not match current results"
        )
    try:
        verification = verifier(evidence)
    except CheckpointError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise CheckpointError("evidence_invalid", "working evidence is invalid") from error
    if verification.get("status") != "verified":
        raise CheckpointError("evidence_invalid", "working evidence did not verify")
    try:
        staged = staged_verifier(repo_root=workspace, evidence_root=evidence)
    except CheckpointError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise CheckpointError("evidence_invalid", "Git-index evidence is invalid") from error
    if staged.get("status") != "staged_verified":
        raise CheckpointError("evidence_invalid", "Git-index evidence did not verify")
    return _read_evidence_proof(completion, evidence_root=evidence)


CompletionReader = Callable[[], PairCompletion | None]
EvidenceReader = Callable[[PairCompletion], EvidenceProof]
RepositoryReader = Callable[[], RepositoryProof]


def acknowledge_checkpoint(
    *,
    workspace: Path,
    campaign: CampaignBinding,
    journal_path: Path,
    completion_reader: CompletionReader,
    evidence_verifier: EvidenceReader | None = None,
    repository_verifier: RepositoryReader | None = None,
    evidence_snapshot_reader: EvidenceReader | None = None,
    repository_snapshot_reader: RepositoryReader | None = None,
    state_root: Path | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> CheckpointAcknowledgement:
    """Create one acknowledgement only if every bound proof stays unchanged.

    The caller must hold the campaign controller lock so ignored measurement
    state cannot advance between the two snapshots.
    """

    workspace = Path(workspace).resolve(strict=True)
    evidence_verify = evidence_verifier or (
        lambda completion: prove_evidence(completion, workspace=workspace)
    )
    evidence_snapshot = evidence_snapshot_reader or (
        lambda completion: _read_evidence_proof(
            completion, evidence_root=workspace / "evidence"
        )
    )
    repository_verify = repository_verifier or (
        lambda: prove_repository(workspace)
    )
    repository_snapshot = repository_snapshot_reader or repository_verify

    completion_before = completion_reader()
    if completion_before is None:
        raise CheckpointError("no_completed_pair", "no completed pair needs acknowledgement")
    journal_before = capture_journal_prefix(journal_path)
    repository_before = repository_verify()
    evidence_before = evidence_verify(completion_before)
    evidence_before.require_completion(completion_before)

    completion_after = completion_reader()
    journal_after = capture_journal_prefix(journal_path)
    if completion_after is None:
        raise CheckpointError("checkpoint_race", "completed pair changed during verification")
    evidence_after = evidence_snapshot(completion_after)
    repository_after = repository_snapshot()
    if (
        completion_before != completion_after
        or journal_before != journal_after
        or evidence_before != evidence_after
        or repository_before != repository_after
    ):
        raise CheckpointError("checkpoint_race", "checkpoint inputs changed during verification")

    timestamp = now()
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise CheckpointError("checkpoint_state_invalid", "checkpoint clock is naive")
    acknowledgement = CheckpointAcknowledgement(
        campaign=campaign,
        completion=completion_before,
        journal_prefix=journal_before,
        evidence=evidence_before,
        repository=repository_before,
        acknowledged_at=timestamp.isoformat(timespec="milliseconds"),
    )
    path = checkpoint_state_path(workspace, campaign, state_root=state_root)
    existing = load_acknowledgement(path)
    if existing is not None and (
        existing.campaign == acknowledgement.campaign
        and existing.completion == acknowledgement.completion
        and existing.journal_prefix == acknowledgement.journal_prefix
        and existing.evidence == acknowledgement.evidence
        and existing.repository == acknowledgement.repository
    ):
        return existing
    _write_acknowledgement(path, acknowledgement)
    loaded = load_acknowledgement(path)
    if loaded != acknowledgement:
        raise CheckpointError("checkpoint_state_invalid", "checkpoint write did not verify")
    return acknowledgement


def checkpoint_gate(
    *,
    workspace: Path,
    campaign: CampaignBinding,
    completion: PairCompletion | None,
    journal_path: Path,
    evidence_reader: EvidenceReader | None = None,
    repository_reader: RepositoryReader | None = None,
    state_root: Path | None = None,
) -> CheckpointGate:
    """Return a non-mutating gate decision for the next pair admission."""

    if completion is None:
        return CheckpointGate("ready", None, 0)
    path = checkpoint_state_path(workspace, campaign, state_root=state_root)
    acknowledgement = load_acknowledgement(path)
    if acknowledgement is None:
        return CheckpointGate("checkpoint_required", "missing", completion.sequence)
    if acknowledgement.campaign != campaign:
        raise CheckpointError(
            "checkpoint_state_invalid", "checkpoint is bound to another campaign"
        )
    if acknowledgement.completion != completion:
        return CheckpointGate("checkpoint_required", "new_pair", completion.sequence)
    require_journal_prefix(journal_path, acknowledgement.journal_prefix)

    evidence_current_reader = evidence_reader or (
        lambda item: prove_evidence(item, workspace=Path(workspace))
    )
    repository_current_reader = repository_reader or (
        lambda: prove_repository(Path(workspace))
    )
    try:
        evidence = evidence_current_reader(completion)
        evidence.require_completion(completion)
    except CheckpointError as error:
        if error.code == "checkpoint_state_invalid":
            raise
        return CheckpointGate(
            "checkpoint_required", "evidence_changed", completion.sequence
        )
    if evidence != acknowledgement.evidence:
        return CheckpointGate(
            "checkpoint_required", "evidence_changed", completion.sequence
        )
    try:
        repository = repository_current_reader()
    except CheckpointError as error:
        if error.code == "checkpoint_state_invalid":
            raise
        reason = (
            "remote_unverified"
            if error.code in {"remote_unverified", "repository_unverified"}
            else "repository_changed"
        )
        return CheckpointGate("checkpoint_required", reason, completion.sequence)
    if repository != acknowledgement.repository:
        return CheckpointGate(
            "checkpoint_required", "repository_changed", completion.sequence
        )
    return CheckpointGate("ready", None, completion.sequence)
