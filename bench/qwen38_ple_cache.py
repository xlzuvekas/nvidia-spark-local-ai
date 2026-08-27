"""Offline materialization and validation for the pinned Flash-Next PLE table."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
from typing import Any


_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
_MIN_FREE_RESERVE_BYTES = 64 * 1024 * 1024


class PLECacheError(RuntimeError):
    """Raised when the persistent PLE cache cannot be trusted or produced."""


@dataclass(frozen=True, slots=True)
class PLESourceFile:
    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PLELayout:
    model_id: str
    source: str
    revision: str
    recipe_source: str
    recipe_revision: str
    config_sha256: str
    index_sha256: str
    tensor_prefix: str
    dtype: str
    shard_count: int
    rows_per_shard: int
    embedding_dim: int
    cache_file: str
    marker_file: str
    payload_sha256: str
    source_files: tuple[PLESourceFile, ...]

    @property
    def total_rows(self) -> int:
        return self.shard_count * self.rows_per_shard

    @property
    def payload_size_bytes(self) -> int:
        return self.total_rows * self.embedding_dim


PINNED_LAYOUT = PLELayout(
    model_id="qwen38-flash-next-nvfp4-mtp-sglang",
    source="RadixArk/Qwen3.8-Flash-Next-NVFP4",
    revision="7b719225242aacd3dbd3f9407468c2ee9a9d2594",
    recipe_source="hashd1ve/qwen38-flash-next-one-dgx-spark",
    recipe_revision="bf2b7c75870d3703730b6bd8f3bb93dc622c278d",
    config_sha256=(
        "e765305daba0951974308f4d32c075b5"
        "2a6a45974730d273f2216718a994d624"
    ),
    index_sha256=(
        "da5ca9c3b65e48e151329e64e141c2f"
        "a700bf2f99aec53cc014e4b52a6ff7a84"
    ),
    tensor_prefix=(
        "model.language_model.layers.1.ple.ple_embedding."
        "ngram_embedding.shard_"
    ),
    dtype="F8_E4M3",
    shard_count=128,
    rows_per_shard=2_500_012,
    embedding_dim=160,
    cache_file="ple_table_51200245760_51200245760.bin",
    marker_file="ple-cache-v1.json",
    payload_sha256=(
        "b070f9644adf93794d8a1030584ab705"
        "809387e64396a9327a68fa3a3a6666b3"
    ),
    source_files=(
        PLESourceFile(
            "model-plefp8-00000.safetensors",
            5_200_027_199,
            "dc2e845b7edd35bda92834fba3626bf7d199e28d6aceac99fee654aade390cfd",
        ),
        PLESourceFile(
            "model-plefp8-00001.safetensors",
            5_200_027_202,
            "899eaa0716e28594468a1389ee58cb065c23907f1270de3831f5ecb0a4f82d56",
        ),
        PLESourceFile(
            "model-plefp8-00002.safetensors",
            5_200_027_198,
            "06fd8a11abf0419a669f89397b8d70dd6ff42d401e6b2a037c65e49704faaf71",
        ),
        PLESourceFile(
            "model-plefp8-00003.safetensors",
            5_200_027_189,
            "c6aaa1fc08e84eced3c8151ac8679ed943888eba0fcef2556963693430f95bd9",
        ),
        PLESourceFile(
            "model-plefp8-00004.safetensors",
            5_200_027_190,
            "d94e97c96d3ea09208614da016960f3f4b429f47a044c898a51b493f42f74ba2",
        ),
        PLESourceFile(
            "model-plefp8-00005.safetensors",
            5_200_027_190,
            "03d5d4792e14a4ab55bae50bb459624b01786f818bbcfcaed1a5d0235af484c6",
        ),
        PLESourceFile(
            "model-plefp8-00006.safetensors",
            5_200_027_190,
            "586cccbc12383021bc9bc02f206d9b19fbd1672373ed9a5d91e3b0ce34c2418f",
        ),
        PLESourceFile(
            "model-plefp8-00007.safetensors",
            5_200_027_190,
            "c4a23bcc10f3cde6b633e82c282a9a518e3a64d433628283e1bc592c94cf3d6c",
        ),
        PLESourceFile(
            "model-plefp8-00008.safetensors",
            5_200_027_190,
            "2dc8098c0d020bff277c9cf499a6b908e17836b70a5f949dfa24793371c9a87e",
        ),
        PLESourceFile(
            "model-plefp8-00009.safetensors",
            4_400_023_163,
            "61de98b89bb79f386795787d7a76827a26f1e292c26edbb0a1b613da455f5a9c",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class _TensorSlice:
    index: int
    path: Path
    offset: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class PLECacheRecord:
    marker_sha256: str
    payload_sha256: str
    payload_size_bytes: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PLECacheError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _decode_json(raw: bytes, description: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PLECacheError(f"invalid {description} JSON") from error


def _canonical_json(document: Any) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _marker_document(layout: PLELayout) -> dict[str, Any]:
    return {
        "artifact": {
            "revision": layout.revision,
            "source": layout.source,
        },
        "config_sha256": f"sha256:{layout.config_sha256}",
        "index_sha256": f"sha256:{layout.index_sha256}",
        "kind": "sparkbench-qwen4-ple-cache",
        "recipe": {
            "revision": layout.recipe_revision,
            "source": layout.recipe_source,
        },
        "schema_version": 1,
        "source_files": [
            {
                "name": source.name,
                "sha256": f"sha256:{source.sha256}",
                "size_bytes": source.size_bytes,
            }
            for source in layout.source_files
        ],
        "tensor": {
            "dtype": layout.dtype,
            "file": layout.cache_file,
            "sha256": f"sha256:{layout.payload_sha256}",
            "shape": [layout.total_rows, layout.embedding_dim],
            "shard_count": layout.shard_count,
            "size_bytes": layout.payload_size_bytes,
            "source_name_pattern": f"{layout.tensor_prefix}N.weight",
        },
    }


def expected_marker_bytes(layout: PLELayout = PINNED_LAYOUT) -> bytes:
    """Return the deterministic completion marker for an exact PLE layout."""

    return _canonical_json(_marker_document(layout))


def expected_marker_sha256(layout: PLELayout = PINNED_LAYOUT) -> str:
    return hashlib.sha256(expected_marker_bytes(layout)).hexdigest()


def default_snapshot_path(
    *, home: Path | None = None, layout: PLELayout = PINNED_LAYOUT
) -> Path:
    base = home if home is not None else Path.home()
    repository = "models--" + layout.source.replace("/", "--")
    return (
        base
        / ".cache"
        / "huggingface"
        / "hub"
        / repository
        / "snapshots"
        / layout.revision
    )


def default_cache_path(
    *, home: Path | None = None, layout: PLELayout = PINNED_LAYOUT
) -> Path:
    base = home if home is not None else Path.home()
    return (
        base
        / ".cache"
        / "sparkbench"
        / "sglang"
        / layout.model_id
        / "ple"
    )


def _require_regular(path: Path, description: str) -> os.stat_result:
    if path.is_symlink():
        raise PLECacheError(f"{description} must not be a symbolic link")
    try:
        status = path.stat()
    except FileNotFoundError as error:
        raise PLECacheError(f"missing {description}") from error
    if not stat.S_ISREG(status.st_mode):
        raise PLECacheError(f"{description} must be a regular file")
    return status


def _validate_source_file(
    path: Path,
    source: PLESourceFile,
    *,
    verify_digest: bool,
) -> None:
    if not path.is_file():
        raise PLECacheError(f"missing pinned source file {source.name}")
    try:
        size = path.stat().st_size
    except OSError as error:
        raise PLECacheError(
            f"cannot inspect pinned source file {source.name}"
        ) from error
    if size != source.size_bytes:
        raise PLECacheError(
            f"pinned source size mismatch for {source.name}: "
            f"expected {source.size_bytes}, got {size}"
        )

    # Hugging Face LFS snapshot links name their immutable blob by SHA-256.
    # A regular test/export copy has no such identity and must be hashed.
    linked_digest: str | None = None
    if path.is_symlink():
        try:
            linked_digest = path.resolve(strict=True).name
        except FileNotFoundError as error:
            raise PLECacheError(
                f"broken pinned source link {source.name}"
            ) from error
        if linked_digest != source.sha256:
            raise PLECacheError(
                f"pinned source blob mismatch for {source.name}"
            )
    if verify_digest or linked_digest is None:
        actual = _sha256_file(path)
        if actual != source.sha256:
            raise PLECacheError(
                f"pinned source digest mismatch for {source.name}"
            )


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    try:
        with path.open("rb") as stream:
            encoded_length = stream.read(8)
            if len(encoded_length) != 8:
                raise PLECacheError(f"truncated safetensors header in {path.name}")
            header_length = struct.unpack("<Q", encoded_length)[0]
            if not 1 <= header_length <= _MAX_SAFETENSORS_HEADER_BYTES:
                raise PLECacheError(
                    f"unsafe safetensors header length in {path.name}"
                )
            raw_header = stream.read(header_length)
            if len(raw_header) != header_length:
                raise PLECacheError(f"truncated safetensors header in {path.name}")
    except OSError as error:
        raise PLECacheError(f"cannot read {path.name}") from error
    header = _decode_json(raw_header, f"{path.name} safetensors header")
    if not isinstance(header, dict):
        raise PLECacheError(f"invalid safetensors header in {path.name}")
    return 8 + header_length, header


def _validate_snapshot(
    snapshot: Path,
    layout: PLELayout,
    *,
    verify_source_digests: bool,
    progress: Callable[[str], None] | None,
) -> tuple[_TensorSlice, ...]:
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise PLECacheError("exact pinned snapshot directory is unavailable")

    config = snapshot / "config.json"
    index = snapshot / "model.safetensors.index.json"
    if not config.is_file() or _sha256_file(config) != layout.config_sha256:
        raise PLECacheError("pinned config.json digest mismatch")
    if not index.is_file() or _sha256_file(index) != layout.index_sha256:
        raise PLECacheError("pinned model index digest mismatch")

    config_document = _decode_json(config.read_bytes(), "config.json")
    if not isinstance(config_document, dict):
        raise PLECacheError("config.json must be an object")
    text_config = config_document.get("text_config")
    if not isinstance(text_config, dict):
        raise PLECacheError("config.json is missing text_config")
    if text_config.get("split_ngram_parts") != layout.shard_count:
        raise PLECacheError("config split_ngram_parts does not match the layout")
    if text_config.get("ple_embedding_dtype") != "float8_e4m3fn":
        raise PLECacheError("config PLE dtype does not match the FP8 layout")

    index_document = _decode_json(index.read_bytes(), "model index")
    if not isinstance(index_document, dict):
        raise PLECacheError("model index must be an object")
    weight_map = index_document.get("weight_map")
    if not isinstance(weight_map, dict):
        raise PLECacheError("model index is missing weight_map")

    expected_sources = {source.name: source for source in layout.source_files}
    for source in layout.source_files:
        _validate_source_file(
            snapshot / source.name,
            source,
            verify_digest=verify_source_digests,
        )
        if progress is not None and verify_source_digests:
            progress(f"verified source {source.name}")

    suffix = ".weight"
    indexed_names = {
        name
        for name in weight_map
        if isinstance(name, str)
        and name.startswith(layout.tensor_prefix)
        and name.endswith(suffix)
    }
    expected_names = {
        f"{layout.tensor_prefix}{index}{suffix}"
        for index in range(layout.shard_count)
    }
    if indexed_names != expected_names:
        raise PLECacheError("model index has missing or unexpected PLE shards")

    headers: dict[str, tuple[int, dict[str, Any]]] = {}
    slices: list[_TensorSlice] = []
    for shard_index in range(layout.shard_count):
        name = f"{layout.tensor_prefix}{shard_index}{suffix}"
        filename = weight_map.get(name)
        if not isinstance(filename, str) or filename not in expected_sources:
            raise PLECacheError(f"unexpected source mapping for PLE shard {shard_index}")
        path = snapshot / filename
        if filename not in headers:
            headers[filename] = _read_safetensors_header(path)
        data_start, header = headers[filename]
        tensor = header.get(name)
        if not isinstance(tensor, dict) or set(tensor) != {
            "data_offsets",
            "dtype",
            "shape",
        }:
            raise PLECacheError(f"invalid metadata for PLE shard {shard_index}")
        if tensor.get("dtype") != layout.dtype or tensor.get("shape") != [
            layout.rows_per_shard,
            layout.embedding_dim,
        ]:
            raise PLECacheError(f"shape or dtype mismatch for PLE shard {shard_index}")
        offsets = tensor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
        ):
            raise PLECacheError(f"invalid offsets for PLE shard {shard_index}")
        start, end = offsets
        expected_size = layout.rows_per_shard * layout.embedding_dim
        if start < 0 or end - start != expected_size:
            raise PLECacheError(f"byte-size mismatch for PLE shard {shard_index}")
        if data_start + end > expected_sources[filename].size_bytes:
            raise PLECacheError(f"out-of-range PLE shard {shard_index}")
        slices.append(
            _TensorSlice(
                index=shard_index,
                path=path,
                offset=data_start + start,
                size_bytes=expected_size,
            )
        )
    return tuple(slices)


def _iter_tensor_bytes(
    slices: tuple[_TensorSlice, ...],
) -> Iterator[bytes]:
    for tensor in slices:
        remaining = tensor.size_bytes
        try:
            with tensor.path.open("rb") as stream:
                stream.seek(tensor.offset)
                while remaining:
                    chunk = stream.read(min(_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise PLECacheError(
                            f"truncated PLE tensor shard {tensor.index}"
                        )
                    remaining -= len(chunk)
                    yield chunk
        except OSError as error:
            raise PLECacheError(
                f"cannot read PLE tensor shard {tensor.index}"
            ) from error


def _validate_cache_directory(cache: Path) -> None:
    if cache.is_symlink():
        raise PLECacheError("PLE cache directory must not be a symbolic link")
    if cache.exists() and not cache.is_dir():
        raise PLECacheError("PLE cache path must be a directory")


def _create_cache_directory(cache: Path) -> None:
    cursor = cache.anchor and Path(cache.anchor) or Path()
    for component in cache.parts[1:] if cache.is_absolute() else cache.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise PLECacheError("PLE cache path must not traverse a symbolic link")
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache.chmod(0o700)


def _write_marker(cache: Path, layout: PLELayout) -> None:
    marker = cache / layout.marker_file
    temporary = cache / f".{layout.marker_file}.partial"
    if marker.exists() or marker.is_symlink() or temporary.exists():
        raise PLECacheError("PLE completion marker already exists or is partial")
    raw = expected_marker_bytes(layout)
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
        marker.chmod(0o400)
        directory_fd = os.open(cache, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise PLECacheError("could not commit the PLE completion marker") from error


def validate_ple_cache(
    cache: Path,
    *,
    layout: PLELayout = PINNED_LAYOUT,
    verify_payload: bool = False,
) -> PLECacheRecord:
    """Validate an exact completed cache without changing it."""

    _validate_cache_directory(cache)
    if not cache.is_dir():
        raise PLECacheError("completed PLE cache is unavailable")
    try:
        cache_status = cache.stat()
    except OSError as error:
        raise PLECacheError("cannot inspect PLE cache directory") from error
    if stat.S_IMODE(cache_status.st_mode) != 0o500:
        raise PLECacheError(
            "completed PLE cache directory must have mode 0500"
        )
    expected_names = {layout.cache_file, layout.marker_file}
    try:
        entries = tuple(cache.iterdir())
    except OSError as error:
        raise PLECacheError("cannot inspect PLE cache directory") from error
    if {entry.name for entry in entries} != expected_names:
        raise PLECacheError("PLE cache is stale, partial, or has unexpected files")

    payload = cache / layout.cache_file
    marker = cache / layout.marker_file
    payload_status = _require_regular(payload, "PLE cache payload")
    marker_status = _require_regular(marker, "PLE completion marker")
    if payload_status.st_size != layout.payload_size_bytes:
        raise PLECacheError("PLE cache payload size mismatch")
    if payload_status.st_mode & 0o222 or marker_status.st_mode & 0o222:
        raise PLECacheError("completed PLE cache files must be read-only")
    try:
        raw_marker = marker.read_bytes()
    except OSError as error:
        raise PLECacheError("cannot read PLE completion marker") from error
    if raw_marker != expected_marker_bytes(layout):
        raise PLECacheError("PLE completion marker is stale or non-deterministic")
    if verify_payload and _sha256_file(payload) != layout.payload_sha256:
        raise PLECacheError("PLE cache payload digest mismatch")
    return PLECacheRecord(
        marker_sha256=hashlib.sha256(raw_marker).hexdigest(),
        payload_sha256=layout.payload_sha256,
        payload_size_bytes=layout.payload_size_bytes,
    )


def materialize_ple_cache(
    *,
    snapshot: Path | None = None,
    cache: Path | None = None,
    layout: PLELayout = PINNED_LAYOUT,
    progress: Callable[[str], None] | None = None,
) -> PLECacheRecord:
    """Build or adopt the exact PLE cache without constructing the model."""

    source_root = snapshot if snapshot is not None else default_snapshot_path(layout=layout)
    target = cache if cache is not None else default_cache_path(layout=layout)
    _validate_cache_directory(target)

    # Existing complete caches are admitted only through the exact marker and a
    # full payload hash in this explicit offline command. Runtime admission uses
    # the same marker but avoids faulting 47.7 GiB into the page cache.
    if target.is_dir() and (target / layout.marker_file).exists():
        record = validate_ple_cache(target, layout=layout, verify_payload=True)
        if progress is not None:
            progress("verified existing completed PLE cache")
        return record

    slices = _validate_snapshot(
        source_root,
        layout,
        verify_source_digests=not (
            target.is_dir() and (target / layout.cache_file).exists()
        ),
        progress=progress,
    )
    _create_cache_directory(target)
    entries = tuple(target.iterdir())
    allowed_unmarked = {layout.cache_file}
    names = {entry.name for entry in entries}
    if names and names != allowed_unmarked:
        raise PLECacheError("PLE cache is stale, partial, or has unexpected files")

    payload = target / layout.cache_file
    if payload.exists() or payload.is_symlink():
        status = _require_regular(payload, "unmarked PLE cache payload")
        if status.st_size != layout.payload_size_bytes:
            raise PLECacheError("unmarked PLE cache payload has the wrong size")
        if progress is not None:
            progress("validating unmarked PLE cache payload for adoption")
        if _sha256_file(payload) != layout.payload_sha256:
            raise PLECacheError("unmarked PLE cache payload digest mismatch")
    else:
        free_bytes = shutil.disk_usage(target).free
        if free_bytes < layout.payload_size_bytes + _MIN_FREE_RESERVE_BYTES:
            raise PLECacheError("insufficient free space for the PLE cache payload")
        partial = target / f".{layout.cache_file}.partial"
        if partial.exists() or partial.is_symlink():
            raise PLECacheError("partial PLE cache payload already exists")
        digest = hashlib.sha256()
        try:
            descriptor = os.open(
                partial,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as output:
                for chunk in _iter_tensor_bytes(slices):
                    output.write(chunk)
                    digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
            if partial.stat().st_size != layout.payload_size_bytes:
                raise PLECacheError("materialized PLE payload has the wrong size")
            if digest.hexdigest() != layout.payload_sha256:
                raise PLECacheError("materialized PLE payload digest mismatch")
            os.replace(partial, payload)
        except OSError as error:
            raise PLECacheError("could not materialize the PLE cache payload") from error
        if progress is not None:
            progress("materialized exact PLE payload")

    try:
        payload.chmod(0o400)
    except OSError as error:
        raise PLECacheError(
            "the exact verified PLE payload cannot be made read-only; "
            "change ownership of this one payload to the invoking user and "
            f"rerun (no ownership change was attempted): {payload}"
        ) from error
    _write_marker(target, layout)
    try:
        target.chmod(0o500)
    except OSError as error:
        raise PLECacheError(
            "could not make the completed PLE cache directory read-only"
        ) from error
    record = validate_ple_cache(target, layout=layout, verify_payload=False)
    if progress is not None:
        progress("committed deterministic PLE completion marker")
    return record
