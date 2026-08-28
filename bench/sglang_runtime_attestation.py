"""Strict scalar-only admission records for the future SM121 Triton runtime.

This module validates an already-collected admission record.  It does not read
files, inspect Docker, launch a server, or make an admission decision from
unverified live state.  The fixed source/build identity prevents the retired
d91 image and its source-overlay route from being represented as this future
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final


RUNTIME_ATTESTATION_SCHEMA_VERSION: Final = 1
RUNTIME_ATTESTATION_KIND: Final = "sparkbench-sglang-runtime-attestation"
RUNTIME_ATTESTATION_STATUS: Final = "admitted"

SM121_TRITON_CANDIDATE_ID: Final = "sglang-sm121-triton-storage-v1"
SM121_TRITON_SOURCE_TREE: Final = "274ee330db7ea9653807b868c0fb8693d50ed7b2"
SM121_TRITON_BUILD_CONTRACT_SHA256: Final = (
    "sha256:c9c7c5bb958a8cf4c0fbc904b40c5e51fac82ef97c6e1fc391e2b67b5c9d9975"
)
SM121_TRITON_PLATFORM: Final = "linux/arm64"

_RECORD_FIELDS: Final = frozenset(
    {
        "schema_version",
        "kind",
        "candidate_id",
        "status",
        "source_tree",
        "build_contract_sha256",
        "oci_image_digest",
        "platform",
        "model_sha256",
        "tokenizer_sha256",
        "revision_sha256",
        "profile_sha256",
        "retired_overlay_rejected",
        "storage_import_passed",
        "io_uring_passed",
        "ple_rows_passed",
        "sm121_triton_passed",
        "quality_passed",
        "long_context_passed",
    }
)
_REQUIRED_TRUE_FIELDS: Final = (
    "retired_overlay_rejected",
    "storage_import_passed",
    "io_uring_passed",
    "ple_rows_passed",
    "sm121_triton_passed",
    "quality_passed",
    "long_context_passed",
)
_SHA256_PATTERN: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_TREE_PATTERN: Final = re.compile(r"[0-9a-f]{40}\Z")
_PLACEHOLDER_SHA256_BODIES: Final = frozenset(
    {
        "0" * 64,
        "f" * 64,
        "deadbeef" * 8,
        "0123456789abcdef" * 4,
    }
)


class SGLangRuntimeAttestationError(ValueError):
    """Raised when a future runtime admission record is malformed or unsafe."""


def _require_record(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise SGLangRuntimeAttestationError("runtime attestation must be an object")
    if any(type(key) is not str for key in value):
        raise SGLangRuntimeAttestationError(
            "runtime attestation field names must be strings"
        )
    return value


def _require_exact_fields(record: dict[str, object]) -> None:
    actual = frozenset(record)
    if actual == _RECORD_FIELDS:
        return
    if _RECORD_FIELDS - actual:
        raise SGLangRuntimeAttestationError(
            "runtime attestation has missing required fields"
        )
    raise SGLangRuntimeAttestationError("runtime attestation has unknown fields")


def _require_exact_int(value: object, expected: int, *, field: str) -> int:
    if type(value) is not int or value != expected:
        raise SGLangRuntimeAttestationError(
            f"runtime attestation {field} is not the supported value"
        )
    return value


def _require_exact_string(value: object, expected: str, *, field: str) -> str:
    if type(value) is not str or value != expected:
        raise SGLangRuntimeAttestationError(
            f"runtime attestation {field} does not match the pinned candidate"
        )
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise SGLangRuntimeAttestationError(
            f"runtime attestation {field} must be a lowercase sha256 digest"
        )
    body = value.removeprefix("sha256:")
    if body in _PLACEHOLDER_SHA256_BODIES or len(set(body)) < 8:
        raise SGLangRuntimeAttestationError(
            f"runtime attestation {field} must not be a placeholder digest"
        )
    return value


def _require_git_tree(value: object) -> str:
    if type(value) is not str or _GIT_TREE_PATTERN.fullmatch(value) is None:
        raise SGLangRuntimeAttestationError(
            "runtime attestation source_tree must be a lowercase Git tree"
        )
    if value != SM121_TRITON_SOURCE_TREE:
        raise SGLangRuntimeAttestationError(
            "runtime attestation source_tree does not match the pinned candidate"
        )
    return value


def _require_true(value: object, *, field: str) -> bool:
    if type(value) is not bool or value is not True:
        raise SGLangRuntimeAttestationError(
            f"runtime attestation {field} must be true"
        )
    return value


@dataclass(frozen=True, slots=True)
class SGLangRuntimeAttestation:
    """Typed, scalar-only record for one admitted SM121 Triton runtime."""

    schema_version: int
    kind: str
    candidate_id: str
    status: str
    source_tree: str
    build_contract_sha256: str
    oci_image_digest: str
    platform: str
    model_sha256: str
    tokenizer_sha256: str
    revision_sha256: str
    profile_sha256: str
    retired_overlay_rejected: bool
    storage_import_passed: bool
    io_uring_passed: bool
    ple_rows_passed: bool
    sm121_triton_passed: bool
    quality_passed: bool
    long_context_passed: bool

    def __post_init__(self) -> None:
        _require_exact_int(
            self.schema_version,
            RUNTIME_ATTESTATION_SCHEMA_VERSION,
            field="schema_version",
        )
        _require_exact_string(self.kind, RUNTIME_ATTESTATION_KIND, field="kind")
        _require_exact_string(
            self.candidate_id,
            SM121_TRITON_CANDIDATE_ID,
            field="candidate_id",
        )
        _require_exact_string(
            self.status,
            RUNTIME_ATTESTATION_STATUS,
            field="status",
        )
        _require_git_tree(self.source_tree)
        _require_exact_string(
            self.build_contract_sha256,
            SM121_TRITON_BUILD_CONTRACT_SHA256,
            field="build_contract_sha256",
        )
        _require_sha256(self.oci_image_digest, field="oci_image_digest")
        _require_exact_string(
            self.platform,
            SM121_TRITON_PLATFORM,
            field="platform",
        )
        _require_sha256(self.model_sha256, field="model_sha256")
        _require_sha256(self.tokenizer_sha256, field="tokenizer_sha256")
        _require_sha256(self.revision_sha256, field="revision_sha256")
        _require_sha256(self.profile_sha256, field="profile_sha256")
        for field in _REQUIRED_TRUE_FIELDS:
            _require_true(getattr(self, field), field=field)

    def to_mapping(self) -> dict[str, object]:
        """Return a deterministic scalar-only representation of this record."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "source_tree": self.source_tree,
            "build_contract_sha256": self.build_contract_sha256,
            "oci_image_digest": self.oci_image_digest,
            "platform": self.platform,
            "model_sha256": self.model_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "revision_sha256": self.revision_sha256,
            "profile_sha256": self.profile_sha256,
            "retired_overlay_rejected": self.retired_overlay_rejected,
            "storage_import_passed": self.storage_import_passed,
            "io_uring_passed": self.io_uring_passed,
            "ple_rows_passed": self.ple_rows_passed,
            "sm121_triton_passed": self.sm121_triton_passed,
            "quality_passed": self.quality_passed,
            "long_context_passed": self.long_context_passed,
        }


def validate_sglang_runtime_attestation(
    value: object,
) -> SGLangRuntimeAttestation:
    """Fail closed unless ``value`` is one complete admitted future record."""

    record = _require_record(value)
    _require_exact_fields(record)
    return SGLangRuntimeAttestation(
        schema_version=record["schema_version"],
        kind=record["kind"],
        candidate_id=record["candidate_id"],
        status=record["status"],
        source_tree=record["source_tree"],
        build_contract_sha256=record["build_contract_sha256"],
        oci_image_digest=record["oci_image_digest"],
        platform=record["platform"],
        model_sha256=record["model_sha256"],
        tokenizer_sha256=record["tokenizer_sha256"],
        revision_sha256=record["revision_sha256"],
        profile_sha256=record["profile_sha256"],
        retired_overlay_rejected=record["retired_overlay_rejected"],
        storage_import_passed=record["storage_import_passed"],
        io_uring_passed=record["io_uring_passed"],
        ple_rows_passed=record["ple_rows_passed"],
        sm121_triton_passed=record["sm121_triton_passed"],
        quality_passed=record["quality_passed"],
        long_context_passed=record["long_context_passed"],
    )
