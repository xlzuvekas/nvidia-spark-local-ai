"""Strict scalar admission provenance for autoresearch campaigns.

Admission records are observational: callers must always recompute a live
decision before launching work.  The journal is append-only, campaign-bound,
controller-prefix-bound, and deliberately excludes paths and request data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

ADMISSION_SCHEMA_VERSION = 1
ADMISSION_JOURNAL_MAX_BYTES = 4 * 1024 * 1024
GENESIS_RECORD_SHA256 = "0" * 64
ADMISSION_BLOCKER_ORDER = (
    "harness_code_changed",
    "insufficient_time_for_pair",
    "starting_swap_above_clean_limit",
    "insufficient_preflight_memavailable",
)
ADMISSION_OUTCOMES = frozenset({"admitted", "blocked_environment", "cutoff"})
ADMISSION_TARGET_KINDS = frozenset({"calibration", "screen", "confirmation"})
_STABLE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,255}\Z")

_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "sequence",
        "previous_record_sha256",
        "record_sha256",
        "campaign_id",
        "campaign_integrity_sha256",
        "preview_sha256",
        "policy_sha256",
        "controller_event_count",
        "controller_prefix_sha256",
        "target_kind",
        "candidate_id",
        "pair_index",
        "observed_at",
        "cutoff_at",
        "remaining_s",
        "required_remaining_s",
        "memavailable_kib",
        "required_memavailable_kib",
        "swap_used_kib",
        "max_starting_swap_kib",
        "observed_harness_sha256",
        "observed_harness_file_count",
        "harness_matches",
        "blockers",
        "outcome",
    }
)


class AdmissionJournalError(ValueError):
    """Raised when admission provenance is invalid or cannot be secured."""


@dataclass(frozen=True, slots=True)
class AdmissionBinding:
    """Exact frozen inputs needed to construct and verify admission records."""

    campaign_id: str
    campaign_integrity_sha256: str
    preview_sha256: str
    policy_sha256: str
    cutoff_at: str
    required_remaining_s: float
    required_memavailable_kib: int
    max_starting_swap_kib: int
    harness_sha256: str
    harness_file_count: int

    def __post_init__(self) -> None:
        _require_stable_id(self.campaign_id, name="campaign_id")
        for name in (
            "campaign_integrity_sha256",
            "preview_sha256",
            "policy_sha256",
            "harness_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        _require_aware_timestamp(self.cutoff_at, name="cutoff_at")
        _require_positive_float(
            self.required_remaining_s, name="required_remaining_s"
        )
        _require_nonnegative_int(
            self.required_memavailable_kib, name="required_memavailable_kib"
        )
        _require_nonnegative_int(
            self.max_starting_swap_kib, name="max_starting_swap_kib"
        )
        _require_positive_int(self.harness_file_count, name="harness_file_count")


@dataclass(frozen=True, slots=True)
class AdmissionTarget:
    """One calibration, screen, or confirmation pair launch boundary."""

    kind: str
    candidate_id: str
    pair_index: int

    def __post_init__(self) -> None:
        if self.kind not in ADMISSION_TARGET_KINDS:
            raise AdmissionJournalError("admission target kind is invalid")
        _require_stable_id(self.candidate_id, name="candidate_id")
        _require_nonnegative_int(self.pair_index, name="pair_index")
        if self.kind == "calibration":
            if self.candidate_id != "control" or self.pair_index != 0:
                raise AdmissionJournalError(
                    "calibration admission target must be control pair zero"
                )
        elif self.candidate_id == "control":
            raise AdmissionJournalError(
                "search admission target cannot use the control candidate"
            )


@dataclass(frozen=True, slots=True)
class AdmissionObservation:
    """A live scalar preflight observation before journal sequencing."""

    observed_at: str
    remaining_s: float
    memavailable_kib: int
    swap_used_kib: int
    observed_harness_sha256: str
    observed_harness_file_count: int
    harness_matches: bool
    blockers: tuple[str, ...]
    outcome: str


def controller_prefix_sha256(events: Sequence[Mapping[str, Any]]) -> str:
    """Return the canonical hash of one exact controller-event prefix."""

    normalized: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            raise AdmissionJournalError("controller prefix event is not an object")
        normalized.append(_strict_json_object(event, context="controller event"))
    return hashlib.sha256(_strict_canonical_json(normalized)).hexdigest()


def observe_admission(
    binding: AdmissionBinding,
    *,
    observed_at: datetime,
    memavailable_kib: int,
    swap_used_kib: int,
    observed_harness_sha256: str,
    observed_harness_file_count: int,
) -> AdmissionObservation:
    """Compute one complete live decision in frozen blocker order."""

    if not isinstance(observed_at, datetime):
        raise AdmissionJournalError("observed_at must be a datetime")
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise AdmissionJournalError("observed_at must be timezone-aware")
    _require_nonnegative_int(memavailable_kib, name="memavailable_kib")
    _require_nonnegative_int(swap_used_kib, name="swap_used_kib")
    _require_sha256(
        observed_harness_sha256, name="observed_harness_sha256"
    )
    _require_positive_int(
        observed_harness_file_count, name="observed_harness_file_count"
    )
    cutoff = _require_aware_timestamp(binding.cutoff_at, name="cutoff_at")
    remaining_s = (cutoff - observed_at).total_seconds()
    harness_matches = (
        observed_harness_sha256 == binding.harness_sha256
        and observed_harness_file_count == binding.harness_file_count
    )
    blockers: list[str] = []
    if not harness_matches:
        blockers.append("harness_code_changed")
    if remaining_s < binding.required_remaining_s:
        blockers.append("insufficient_time_for_pair")
    if swap_used_kib > binding.max_starting_swap_kib:
        blockers.append("starting_swap_above_clean_limit")
    if memavailable_kib < binding.required_memavailable_kib:
        blockers.append("insufficient_preflight_memavailable")
    outcome = (
        "admitted"
        if not blockers
        else "cutoff"
        if blockers == ["insufficient_time_for_pair"]
        else "blocked_environment"
    )
    return AdmissionObservation(
        observed_at=observed_at.isoformat(),
        remaining_s=remaining_s,
        memavailable_kib=memavailable_kib,
        swap_used_kib=swap_used_kib,
        observed_harness_sha256=observed_harness_sha256,
        observed_harness_file_count=observed_harness_file_count,
        harness_matches=harness_matches,
        blockers=tuple(blockers),
        outcome=outcome,
    )


def _validate_observation(
    binding: AdmissionBinding, observation: AdmissionObservation
) -> None:
    if not isinstance(observation, AdmissionObservation):
        raise AdmissionJournalError(
            "admission observation must be AdmissionObservation"
        )
    observed_at = _require_aware_timestamp(
        observation.observed_at, name="observed_at"
    )
    cutoff = _require_aware_timestamp(binding.cutoff_at, name="cutoff_at")
    _require_float(observation.remaining_s, name="remaining_s")
    if observation.remaining_s != (cutoff - observed_at).total_seconds():
        raise AdmissionJournalError("admission remaining seconds changed")
    _require_nonnegative_int(
        observation.memavailable_kib, name="memavailable_kib"
    )
    _require_nonnegative_int(observation.swap_used_kib, name="swap_used_kib")
    _require_sha256(
        observation.observed_harness_sha256,
        name="observed_harness_sha256",
    )
    _require_positive_int(
        observation.observed_harness_file_count,
        name="observed_harness_file_count",
    )
    expected_match = (
        observation.observed_harness_sha256 == binding.harness_sha256
        and observation.observed_harness_file_count == binding.harness_file_count
    )
    if type(observation.harness_matches) is not bool:
        raise AdmissionJournalError("admission harness_matches must be boolean")
    if observation.harness_matches != expected_match:
        raise AdmissionJournalError("admission harness match changed")
    blockers: list[str] = []
    if not expected_match:
        blockers.append("harness_code_changed")
    if observation.remaining_s < binding.required_remaining_s:
        blockers.append("insufficient_time_for_pair")
    if observation.swap_used_kib > binding.max_starting_swap_kib:
        blockers.append("starting_swap_above_clean_limit")
    if observation.memavailable_kib < binding.required_memavailable_kib:
        blockers.append("insufficient_preflight_memavailable")
    if type(observation.blockers) is not tuple or observation.blockers != tuple(
        blockers
    ):
        raise AdmissionJournalError("admission blocker order or values changed")
    expected_outcome = (
        "admitted"
        if not blockers
        else "cutoff"
        if blockers == ["insufficient_time_for_pair"]
        else "blocked_environment"
    )
    if observation.outcome != expected_outcome:
        raise AdmissionJournalError("admission outcome does not match blockers")


def read_admission_journal(
    path: Path,
    *,
    binding: AdmissionBinding,
    controller_events: Sequence[Mapping[str, Any]],
    max_bytes: int = ADMISSION_JOURNAL_MAX_BYTES,
) -> tuple[dict[str, Any], ...]:
    """Strictly read and verify the complete admission chain."""

    _require_max_bytes(max_bytes)
    events = _normalize_controller_events(controller_events)
    parent_descriptor = _open_parent_directory(path)
    try:
        try:
            descriptor = _open_existing_journal(
                parent_descriptor, path.name, read_write=False
            )
        except FileNotFoundError:
            return ()
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            payload = _read_descriptor(
                descriptor,
                parent_descriptor=parent_descriptor,
                name=path.name,
                max_bytes=max_bytes,
            )
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)
    records = _decode_records(payload)
    _validate_chain(records, binding=binding, controller_events=events)
    return tuple(records)


def append_admission_record(
    path: Path,
    *,
    binding: AdmissionBinding,
    target: AdmissionTarget,
    observation: AdmissionObservation,
    controller_events: Sequence[Mapping[str, Any]],
    max_bytes: int = ADMISSION_JOURNAL_MAX_BYTES,
) -> dict[str, Any]:
    """Verify the old chain, append one record, and durably flush it.

    The campaign lock remains the authority that binds this observation to a
    launch.  A journal-local lock additionally prevents accidental concurrent
    writers from producing duplicate sequence numbers.
    """

    _require_max_bytes(max_bytes)
    events = _normalize_controller_events(controller_events)
    _validate_observation(binding, observation)
    parent_descriptor = _open_parent_directory(path)
    descriptor: int | None = None
    created = False
    try:
        try:
            descriptor = _open_existing_journal(
                parent_descriptor, path.name, read_write=True, append=True
            )
        except FileNotFoundError:
            first = _record_payload(
                binding=binding,
                target=target,
                observation=observation,
                controller_events=events,
                sequence=1,
                previous_record_sha256=GENESIS_RECORD_SHA256,
            )
            first_encoded = _strict_canonical_json(first) + b"\n"
            if len(first_encoded) > max_bytes:
                raise AdmissionJournalError(
                    "admission journal exceeds its byte bound"
                )
            descriptor, created = _create_or_open_append_journal(
                parent_descriptor, path.name
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        payload = _read_descriptor(
            descriptor,
            parent_descriptor=parent_descriptor,
            name=path.name,
            max_bytes=max_bytes,
        )
        records = _decode_records(payload)
        _validate_chain(records, binding=binding, controller_events=events)
        record = _record_payload(
            binding=binding,
            target=target,
            observation=observation,
            controller_events=events,
            sequence=len(records) + 1,
            previous_record_sha256=(
                records[-1]["record_sha256"] if records else GENESIS_RECORD_SHA256
            ),
        )
        encoded = _strict_canonical_json(record) + b"\n"
        expected_payload = payload + encoded
        if len(expected_payload) > max_bytes:
            raise AdmissionJournalError("admission journal exceeds its byte bound")
        written = 0
        while written < len(encoded):
            count = os.write(descriptor, encoded[written:])
            if count <= 0:
                raise OSError("short admission journal append")
            written += count
        os.fsync(descriptor)
        observed_payload = _read_descriptor(
            descriptor,
            parent_descriptor=parent_descriptor,
            name=path.name,
            max_bytes=max_bytes,
        )
        if observed_payload != expected_payload:
            raise AdmissionJournalError(
                "admission journal changed during append"
            )
        observed_records = _decode_records(observed_payload)
        _validate_chain(
            observed_records, binding=binding, controller_events=events
        )
        if observed_records[-1] != record:
            raise AdmissionJournalError("admission journal tail changed")
        if created:
            os.fsync(parent_descriptor)
            _require_descriptor_entry_identity(
                descriptor, parent_descriptor, path.name
            )
    except (OSError, TypeError, ValueError) as error:
        if isinstance(error, AdmissionJournalError):
            raise
        raise AdmissionJournalError("admission journal append failed") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
    return record


def _record_payload(
    *,
    binding: AdmissionBinding,
    target: AdmissionTarget,
    observation: AdmissionObservation,
    controller_events: Sequence[Mapping[str, Any]],
    sequence: int,
    previous_record_sha256: str,
) -> dict[str, Any]:
    prefix = tuple(controller_events)
    payload: dict[str, Any] = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "sequence": sequence,
        "previous_record_sha256": previous_record_sha256,
        "campaign_id": binding.campaign_id,
        "campaign_integrity_sha256": binding.campaign_integrity_sha256,
        "preview_sha256": binding.preview_sha256,
        "policy_sha256": binding.policy_sha256,
        "controller_event_count": len(prefix),
        "controller_prefix_sha256": controller_prefix_sha256(prefix),
        "target_kind": target.kind,
        "candidate_id": target.candidate_id,
        "pair_index": target.pair_index,
        "observed_at": observation.observed_at,
        "cutoff_at": binding.cutoff_at,
        "remaining_s": observation.remaining_s,
        "required_remaining_s": binding.required_remaining_s,
        "memavailable_kib": observation.memavailable_kib,
        "required_memavailable_kib": binding.required_memavailable_kib,
        "swap_used_kib": observation.swap_used_kib,
        "max_starting_swap_kib": binding.max_starting_swap_kib,
        "observed_harness_sha256": observation.observed_harness_sha256,
        "observed_harness_file_count": observation.observed_harness_file_count,
        "harness_matches": observation.harness_matches,
        "blockers": list(observation.blockers),
        "outcome": observation.outcome,
    }
    payload["record_sha256"] = _record_sha256(payload)
    _validate_record(
        payload,
        binding=binding,
        controller_events=prefix,
        expected_sequence=sequence,
        expected_previous=previous_record_sha256,
    )
    return payload


def _validate_chain(
    records: Sequence[Mapping[str, Any]],
    *,
    binding: AdmissionBinding,
    controller_events: Sequence[Mapping[str, Any]],
) -> None:
    previous = GENESIS_RECORD_SHA256
    for index, record in enumerate(records, start=1):
        _validate_record(
            record,
            binding=binding,
            controller_events=controller_events,
            expected_sequence=index,
            expected_previous=previous,
        )
        previous = str(record["record_sha256"])


def _validate_record(
    record: Mapping[str, Any],
    *,
    binding: AdmissionBinding,
    controller_events: Sequence[Mapping[str, Any]],
    expected_sequence: int,
    expected_previous: str,
) -> None:
    if set(record) != _RECORD_KEYS:
        raise AdmissionJournalError("admission record schema changed")
    schema_version = record.get("schema_version")
    if type(schema_version) is not int or schema_version != ADMISSION_SCHEMA_VERSION:
        raise AdmissionJournalError("admission record schema version changed")
    sequence = record.get("sequence")
    _require_positive_int(sequence, name="sequence")
    if sequence != expected_sequence:
        raise AdmissionJournalError("admission sequence is not contiguous")
    if record.get("previous_record_sha256") != expected_previous:
        raise AdmissionJournalError("admission digest chain is broken")
    _require_sha256(record.get("record_sha256"), name="record_sha256")
    if record["record_sha256"] != _record_sha256(record):
        raise AdmissionJournalError("admission record digest changed")
    for key, expected in (
        ("campaign_id", binding.campaign_id),
        ("campaign_integrity_sha256", binding.campaign_integrity_sha256),
        ("preview_sha256", binding.preview_sha256),
        ("policy_sha256", binding.policy_sha256),
        ("cutoff_at", binding.cutoff_at),
    ):
        if record.get(key) != expected:
            raise AdmissionJournalError(f"admission {key} binding changed")
    required_remaining = record.get("required_remaining_s")
    _require_positive_float(required_remaining, name="required_remaining_s")
    if required_remaining != binding.required_remaining_s:
        raise AdmissionJournalError(
            "admission required_remaining_s binding changed"
        )
    required_memory = record.get("required_memavailable_kib")
    maximum_swap = record.get("max_starting_swap_kib")
    _require_nonnegative_int(
        required_memory, name="required_memavailable_kib"
    )
    _require_nonnegative_int(maximum_swap, name="max_starting_swap_kib")
    if required_memory != binding.required_memavailable_kib:
        raise AdmissionJournalError(
            "admission required_memavailable_kib binding changed"
        )
    if maximum_swap != binding.max_starting_swap_kib:
        raise AdmissionJournalError(
            "admission max_starting_swap_kib binding changed"
        )
    count = record.get("controller_event_count")
    _require_nonnegative_int(count, name="controller_event_count")
    if count > len(controller_events):
        raise AdmissionJournalError("admission controller prefix is unavailable")
    prefix_hash = record.get("controller_prefix_sha256")
    _require_sha256(prefix_hash, name="controller_prefix_sha256")
    if prefix_hash != controller_prefix_sha256(controller_events[:count]):
        raise AdmissionJournalError("admission controller prefix changed")
    target = AdmissionTarget(
        kind=record.get("target_kind"),
        candidate_id=record.get("candidate_id"),
        pair_index=record.get("pair_index"),
    )
    del target
    observed_at = _require_aware_timestamp(
        record.get("observed_at"), name="observed_at"
    )
    cutoff = _require_aware_timestamp(record.get("cutoff_at"), name="cutoff_at")
    remaining_s = record.get("remaining_s")
    _require_float(remaining_s, name="remaining_s")
    if remaining_s != (cutoff - observed_at).total_seconds():
        raise AdmissionJournalError("admission remaining seconds changed")
    memavailable = record.get("memavailable_kib")
    swap_used = record.get("swap_used_kib")
    harness_count = record.get("observed_harness_file_count")
    _require_nonnegative_int(memavailable, name="memavailable_kib")
    _require_nonnegative_int(swap_used, name="swap_used_kib")
    _require_positive_int(
        harness_count, name="observed_harness_file_count"
    )
    harness_hash = record.get("observed_harness_sha256")
    _require_sha256(harness_hash, name="observed_harness_sha256")
    harness_matches = record.get("harness_matches")
    if not isinstance(harness_matches, bool):
        raise AdmissionJournalError("admission harness_matches must be boolean")
    expected_match = (
        harness_hash == binding.harness_sha256
        and harness_count == binding.harness_file_count
    )
    if harness_matches != expected_match:
        raise AdmissionJournalError("admission harness match changed")
    blockers: list[str] = []
    if not expected_match:
        blockers.append("harness_code_changed")
    if remaining_s < binding.required_remaining_s:
        blockers.append("insufficient_time_for_pair")
    if swap_used > binding.max_starting_swap_kib:
        blockers.append("starting_swap_above_clean_limit")
    if memavailable < binding.required_memavailable_kib:
        blockers.append("insufficient_preflight_memavailable")
    if record.get("blockers") != blockers:
        raise AdmissionJournalError("admission blocker order or values changed")
    outcome = record.get("outcome")
    if outcome not in ADMISSION_OUTCOMES:
        raise AdmissionJournalError("admission outcome is invalid")
    expected_outcome = (
        "admitted"
        if not blockers
        else "cutoff"
        if blockers == ["insufficient_time_for_pair"]
        else "blocked_environment"
    )
    if outcome != expected_outcome:
        raise AdmissionJournalError("admission outcome does not match blockers")


def _record_sha256(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(_strict_canonical_json(payload)).hexdigest()


def _decode_records(payload: bytes) -> list[dict[str, Any]]:
    if not payload:
        return []
    if not payload.endswith(b"\n"):
        raise AdmissionJournalError("admission journal has a torn final record")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise AdmissionJournalError(
                    f"admission record repeats key {key!r}"
                )
            value[key] = item
        return value

    records: list[dict[str, Any]] = []
    for line in payload.splitlines():
        if not line:
            raise AdmissionJournalError("admission journal contains an empty record")
        try:
            record = json.loads(
                line.decode("utf-8"), object_pairs_hook=reject_duplicates
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AdmissionJournalError(
                "admission journal contains malformed JSON"
            ) from error
        if not isinstance(record, dict):
            raise AdmissionJournalError("admission record is not an object")
        records.append(record)
    return records


def _open_parent_directory(path: Path) -> int:
    if not isinstance(path, Path) or path.name in {"", ".", ".."}:
        raise AdmissionJournalError("admission journal path is invalid")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise AdmissionJournalError("admission journal requires no-follow support")
    flags = os.O_RDONLY | os.O_CLOEXEC | nofollow | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path.parent, flags)
    except OSError as error:
        raise AdmissionJournalError(
            "admission journal parent is missing or unsafe"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        parent_metadata = os.lstat(path.parent)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_dev != parent_metadata.st_dev
            or metadata.st_ino != parent_metadata.st_ino
        ):
            raise AdmissionJournalError(
                "admission journal parent identity is unsafe"
            )
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _open_existing_journal(
    parent_descriptor: int,
    name: str,
    *,
    read_write: bool,
    append: bool = False,
) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise AdmissionJournalError("admission journal requires no-follow support")
    flags = (os.O_RDWR if read_write else os.O_RDONLY) | os.O_CLOEXEC | nofollow
    if append:
        flags |= os.O_APPEND
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise AdmissionJournalError("admission journal path is unsafe") from error
    try:
        _require_descriptor_entry_identity(descriptor, parent_descriptor, name)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _create_or_open_append_journal(
    parent_descriptor: int, name: str
) -> tuple[int, bool]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags = os.O_RDWR | os.O_APPEND | os.O_CLOEXEC | nofollow
    try:
        descriptor = os.open(
            name,
            flags | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_descriptor,
        )
        created = True
    except FileExistsError:
        descriptor = _open_existing_journal(
            parent_descriptor, name, read_write=True, append=True
        )
        created = False
    except OSError as error:
        raise AdmissionJournalError("admission journal path is unsafe") from error
    try:
        _require_descriptor_entry_identity(descriptor, parent_descriptor, name)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, created


def _read_descriptor(
    descriptor: int,
    *,
    parent_descriptor: int,
    name: str,
    max_bytes: int,
) -> bytes:
    before = _require_descriptor_entry_identity(
        descriptor, parent_descriptor, name
    )
    if before.st_size > max_bytes:
        raise AdmissionJournalError("admission journal exceeds its byte bound")
    payload = bytearray()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(descriptor, min(65_536, before.st_size - offset), offset)
        if not chunk:
            raise AdmissionJournalError("admission journal changed while being read")
        payload.extend(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    if _metadata_identity(after) != _metadata_identity(before):
        raise AdmissionJournalError("admission journal changed while being read")
    _require_descriptor_entry_identity(descriptor, parent_descriptor, name)
    return bytes(payload)


def _require_descriptor_entry_identity(
    descriptor: int, parent_descriptor: int, name: str
) -> os.stat_result:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise AdmissionJournalError(
            "admission journal must be an owned mode-0600 single-link file"
        )
    try:
        path_metadata = os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False
        )
    except OSError as error:
        raise AdmissionJournalError("admission journal path changed") from error
    if _metadata_identity(path_metadata) != _metadata_identity(metadata):
        raise AdmissionJournalError("admission journal path changed")
    return metadata


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_max_bytes(value: Any) -> None:
    _require_positive_int(value, name="max_bytes")


def _require_stable_id(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _STABLE_ID_PATTERN.fullmatch(value) is None:
        raise AdmissionJournalError(f"{name} must be a stable lowercase ID")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AdmissionJournalError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_aware_timestamp(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str):
        raise AdmissionJournalError(f"{name} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise AdmissionJournalError(f"{name} must be an ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AdmissionJournalError(f"{name} must be timezone-aware")
    return parsed


def _require_float(value: Any, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise AdmissionJournalError(f"{name} must be a finite float")
    return value


def _require_positive_float(value: Any, *, name: str) -> float:
    parsed = _require_float(value, name=name)
    if parsed <= 0:
        raise AdmissionJournalError(f"{name} must be positive")
    return parsed


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AdmissionJournalError(f"{name} must be a nonnegative integer")
    return value


def _require_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AdmissionJournalError(f"{name} must be a positive integer")
    return value


def _normalize_controller_events(
    events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(events, (str, bytes, bytearray)) or not isinstance(
        events, Sequence
    ):
        raise AdmissionJournalError("controller events must be a sequence")
    return tuple(
        _strict_json_object(event, context="controller event") for event in events
    )


def _strict_json_object(
    value: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AdmissionJournalError(f"{context} must be an object")
    normalized = _strict_json_value(dict(value), context=context)
    if not isinstance(normalized, dict):  # pragma: no cover - fixed above.
        raise AdmissionJournalError(f"{context} must be an object")
    return normalized


def _strict_json_value(value: Any, *, context: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise AdmissionJournalError(f"{context} contains a nonfinite number")
        return value
    if type(value) is list:
        return [
            _strict_json_value(item, context=context)
            for item in value
        ]
    if type(value) is dict:
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise AdmissionJournalError(f"{context} contains a non-string key")
            normalized[key] = _strict_json_value(item, context=context)
        return normalized
    raise AdmissionJournalError(f"{context} contains a non-JSON value")


def _strict_canonical_json(value: Any) -> bytes:
    normalized = _strict_json_value(value, context="admission payload")
    try:
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:  # pragma: no cover - normalized above.
        raise AdmissionJournalError(
            "admission payload is not strict JSON"
        ) from error
