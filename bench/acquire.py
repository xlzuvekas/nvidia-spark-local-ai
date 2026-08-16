"""Pinned, manifest-driven Hugging Face snapshot acquisition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from typing import Any

from .manifest import ModelSpec, validate_model


_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_HF_REPOSITORY_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)
_WEIGHT_SUFFIXES = (".bin", ".gguf", ".h5", ".pt", ".pth", ".safetensors")
_WEIGHT_INDEX_SUFFIXES = tuple(
    f"{suffix}.index.json" for suffix in _WEIGHT_SUFFIXES
)
_SHARDED_WEIGHT_PATTERN = re.compile(
    r"^(?P<prefix>.+)-(?P<ordinal>[0-9]+)-of-(?P<total>[0-9]+)"
    r"(?P<suffix>\.bin|\.gguf|\.h5|\.pt|\.pth|\.safetensors)$"
)
_MAX_INDEX_BYTES = 64 * 1024 * 1024


class AcquisitionError(RuntimeError):
    """Raised when a profile cannot be fetched or its snapshot is incomplete."""


@dataclass(frozen=True, slots=True)
class SnapshotVerification:
    """Evidence that one exact cached snapshot is locally usable."""

    snapshot_path: str
    snapshot_bytes: int
    file_count: int
    weight_file_count: int
    weight_index_count: int
    referenced_shard_count: int
    exact_model_file: str | None = None
    exact_model_size_bytes: int | None = None
    exact_model_sha256: str | None = None
    exact_mmproj_file: str | None = None
    exact_mmproj_size_bytes: int | None = None
    exact_mmproj_sha256: str | None = None


def resolve_huggingface_hub_root(
    cache_dir: str,
    *,
    workspace: str | Path,
    home: str | Path | None = None,
) -> Path:
    """Resolve a manifest cache selector exactly as inventory and runtime do."""

    workspace_path = Path(workspace).expanduser().resolve()
    if cache_dir == "project":
        return workspace_path / "data" / "huggingface" / "hub"
    if cache_dir == "user":
        home_path = Path.home() if home is None else Path(home)
        return home_path.expanduser().resolve() / ".cache" / "huggingface" / "hub"
    raise AcquisitionError(
        f"Unsupported Hugging Face cache selector {cache_dir!r}; expected "
        "'project' or 'user'"
    )


def fetch_model_snapshot(
    model: ModelSpec,
    *,
    workspace: str | Path,
    home: str | Path | None = None,
    executable: str = "hf",
) -> dict[str, Any]:
    """Download and verify one exact manifest-pinned Hugging Face snapshot.

    The subprocess is strictly the local ``hf download`` client. It receives no
    token argument, and captured client output is never included in successful
    provenance.
    """

    _validate_fetch_profile(model)
    hub_root = resolve_huggingface_hub_root(
        model.cache_dir, workspace=workspace, home=home
    )
    hub_root.mkdir(parents=True, exist_ok=True)

    executable_path = shutil.which(executable)
    if executable_path is None:
        raise AcquisitionError(
            f"Hugging Face CLI {executable!r} is not installed or executable"
        )

    command = _download_command(executable_path, model, hub_root)
    environment = os.environ.copy()
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(workspace).resolve(),
        env=environment,
    )
    if completed.returncode != 0:
        raise AcquisitionError(
            f"hf download failed for {model.source}@{model.revision} "
            f"with exit code {completed.returncode}; client output was suppressed"
        )

    verification = verify_exact_snapshot(model, hub_root=hub_root)
    return {
        "schema_version": 1,
        "status": "verified",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model.id,
        "source": model.source,
        "revision": model.revision,
        "cache_dir": model.cache_dir,
        "cache_root": str(hub_root),
        "fetch_allow_patterns": list(model.fetch_allow_patterns),
        "fetch_ignore_patterns": list(model.fetch_ignore_patterns),
        **asdict(verification),
    }


def verify_exact_snapshot(
    model: ModelSpec, *, hub_root: str | Path
) -> SnapshotVerification:
    """Verify the exact snapshot and every shard named by a weight index."""

    _validate_fetch_profile(model)
    root = Path(hub_root).expanduser().resolve()
    repository_root = root / ("models--" + model.source.replace("/", "--"))
    snapshot = repository_root / "snapshots" / str(model.revision)
    if not snapshot.is_dir():
        raise AcquisitionError(f"Exact downloaded snapshot is missing: {snapshot}")

    try:
        repository_resolved = repository_root.resolve(strict=True)
        repository_resolved.relative_to(root)
        snapshot.resolve(strict=True).relative_to(repository_resolved)
    except (OSError, ValueError) as error:
        raise AcquisitionError(
            f"Exact snapshot escapes the selected cache root: {snapshot}"
        ) from error

    files = _snapshot_files(snapshot)
    if not files:
        raise AcquisitionError(f"Exact downloaded snapshot is empty: {snapshot}")

    total_bytes = 0
    unique_files: set[tuple[int, int]] = set()
    weight_files: set[Path] = set()
    for path in files:
        target = _verified_cache_file(path, repository_resolved, label="snapshot file")
        stat = target.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in unique_files:
            unique_files.add(identity)
            total_bytes += stat.st_size
        if path.name.endswith(_WEIGHT_SUFFIXES):
            if stat.st_size <= 0:
                raise AcquisitionError(f"Model weight file is empty: {path}")
            weight_files.add(path)

    index_paths = tuple(
        path for path in files if path.name.endswith(_WEIGHT_INDEX_SUFFIXES)
    )
    referenced_shards: set[Path] = set()
    weight_index_count = 0
    for index_path in index_paths:
        index_target = _verified_cache_file(
            index_path, repository_resolved, label="weight index"
        )
        if index_target.stat().st_size > _MAX_INDEX_BYTES:
            raise AcquisitionError(f"Weight index is unexpectedly large: {index_path}")
        try:
            payload = json.loads(index_target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise AcquisitionError(f"Cannot parse weight index {index_path}: {error}") from error
        if not isinstance(payload, dict) or "weight_map" not in payload:
            raise AcquisitionError(
                f"Weight index has no weight_map object: {index_path}"
            )
        weight_map = payload["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise AcquisitionError(
                f"Weight index has no non-empty object weight_map: {index_path}"
            )
        weight_index_count += 1
        for shard_name in weight_map.values():
            shard_path = _index_shard_path(index_path, shard_name)
            shard_target = _verified_cache_file(
                shard_path, repository_resolved, label="index-referenced shard"
            )
            if shard_target.stat().st_size <= 0:
                raise AcquisitionError(f"Index-referenced shard is empty: {shard_path}")
            referenced_shards.add(shard_path)
            weight_files.add(shard_path)

    if not weight_files:
        raise AcquisitionError(
            f"Exact snapshot contains no recognized model weight files: {snapshot}"
        )
    _verify_sharded_layouts(weight_files, referenced_shards, weight_index_count)
    exact_model_file: str | None = None
    exact_model_size_bytes: int | None = None
    exact_model_sha256: str | None = None
    if model.model_file is not None:
        exact_path = snapshot / model.model_file
        exact_target = _verified_cache_file(
            exact_path, repository_resolved, label="manifest-pinned model file"
        )
        exact_model_file = model.model_file
        exact_model_size_bytes = exact_target.stat().st_size
        if (
            model.model_size_bytes is not None
            and exact_model_size_bytes != model.model_size_bytes
        ):
            raise AcquisitionError(
                "Manifest-pinned model size mismatch: "
                f"expected {model.model_size_bytes}, got {exact_model_size_bytes}"
            )
        exact_model_sha256 = _sha256_file(exact_target)
        if (
            model.model_digest is not None
            and exact_model_sha256 != model.model_digest
        ):
            raise AcquisitionError(
                "Manifest-pinned model SHA-256 mismatch: "
                f"expected {model.model_digest}, got {exact_model_sha256}"
            )
    exact_mmproj_file: str | None = None
    exact_mmproj_size_bytes: int | None = None
    exact_mmproj_sha256: str | None = None
    if model.mmproj_file is not None:
        mmproj_path = snapshot / model.mmproj_file
        mmproj_target = _verified_cache_file(
            mmproj_path,
            repository_resolved,
            label="manifest-pinned multimodal projector",
        )
        exact_mmproj_file = model.mmproj_file
        exact_mmproj_size_bytes = mmproj_target.stat().st_size
        if (
            model.mmproj_size_bytes is not None
            and exact_mmproj_size_bytes != model.mmproj_size_bytes
        ):
            raise AcquisitionError(
                "Manifest-pinned mmproj size mismatch: "
                f"expected {model.mmproj_size_bytes}, got {exact_mmproj_size_bytes}"
            )
        exact_mmproj_sha256 = _sha256_file(mmproj_target)
        if (
            model.mmproj_digest is not None
            and exact_mmproj_sha256 != model.mmproj_digest
        ):
            raise AcquisitionError(
                "Manifest-pinned mmproj SHA-256 mismatch: "
                f"expected {model.mmproj_digest}, got {exact_mmproj_sha256}"
            )
    return SnapshotVerification(
        snapshot_path=str(snapshot),
        snapshot_bytes=total_bytes,
        file_count=len(files),
        weight_file_count=len(weight_files),
        weight_index_count=weight_index_count,
        referenced_shard_count=len(referenced_shards),
        exact_model_file=exact_model_file,
        exact_model_size_bytes=exact_model_size_bytes,
        exact_model_sha256=exact_model_sha256,
        exact_mmproj_file=exact_mmproj_file,
        exact_mmproj_size_bytes=exact_mmproj_size_bytes,
        exact_mmproj_sha256=exact_mmproj_sha256,
    )


def _validate_fetch_profile(model: ModelSpec) -> None:
    validate_model(model, context=f"model {model.id!r}")
    if model.support_status == "incompatible":
        raise AcquisitionError(f"Incompatible profile {model.id!r} cannot be fetched")
    if model.backend not in {
        "llamacpp",
        "sglang",
        "transformers",
        "trtllm",
        "vllm",
    }:
        raise AcquisitionError(
            f"Profile {model.id!r} is not a Hugging Face-backed profile"
        )
    if not _HF_REPOSITORY_PATTERN.fullmatch(model.source):
        raise AcquisitionError(
            f"Profile {model.id!r} does not use a Hugging Face repository ID"
        )
    if not isinstance(model.revision, str) or not _COMMIT_PATTERN.fullmatch(
        model.revision
    ):
        raise AcquisitionError(
            f"Profile {model.id!r} must pin a full 40-character lowercase commit SHA"
        )


def _download_command(
    executable: str, model: ModelSpec, hub_root: Path
) -> list[str]:
    command = [
        executable,
        "download",
        model.source,
        "--repo-type",
        "model",
        "--revision",
        str(model.revision),
        "--cache-dir",
        str(hub_root),
        "--quiet",
    ]
    if model.fetch_allow_patterns:
        command.extend(("--include", *model.fetch_allow_patterns))
    if model.fetch_ignore_patterns:
        command.extend(("--exclude", *model.fetch_ignore_patterns))
    return command


def _snapshot_files(snapshot: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(snapshot, followlinks=False):
        current = Path(directory)
        for name in directory_names:
            candidate = current / name
            if candidate.is_symlink():
                raise AcquisitionError(
                    f"Snapshot contains an unsupported directory symlink: {candidate}"
                )
        files.extend(current / name for name in file_names)
    return tuple(sorted(files))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _verified_cache_file(path: Path, repository_root: Path, *, label: str) -> Path:
    try:
        target = path.resolve(strict=True)
    except OSError as error:
        raise AcquisitionError(f"Missing or unreadable {label}: {path}") from error
    try:
        target.relative_to(repository_root)
    except ValueError as error:
        raise AcquisitionError(f"{label.capitalize()} escapes its cache repository: {path}") from error
    if not target.is_file():
        raise AcquisitionError(f"Missing or invalid {label}: {path}")
    return target


def _index_shard_path(index_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise AcquisitionError(
            f"Weight index contains a non-string or empty shard path: {index_path}"
        )
    if "\\" in value or any(ord(character) < 32 for character in value):
        raise AcquisitionError(f"Weight index contains an unsafe shard path: {index_path}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise AcquisitionError(f"Weight index contains an unsafe shard path: {index_path}")
    return index_path.parent.joinpath(*relative.parts)


def _verify_sharded_layouts(
    weight_files: set[Path],
    referenced_shards: set[Path],
    weight_index_count: int,
) -> None:
    groups: dict[tuple[Path, str, int, str], set[int]] = {}
    sharded_files: set[Path] = set()
    for path in weight_files:
        match = _SHARDED_WEIGHT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        total = int(match.group("total"))
        ordinal = int(match.group("ordinal"))
        if total <= 0:
            raise AcquisitionError(f"Sharded model weight has invalid total: {path}")
        key = (path.parent, match.group("prefix"), total, match.group("suffix"))
        groups.setdefault(key, set()).add(ordinal)
        sharded_files.add(path)

    if not sharded_files:
        return
    if weight_index_count == 0:
        raise AcquisitionError("Sharded model weights require a verified weight index")
    unreferenced = sorted(sharded_files - referenced_shards)
    if unreferenced:
        raise AcquisitionError(
            f"Sharded model weight is not referenced by an index: {unreferenced[0]}"
        )
    for (parent, prefix, total, suffix), ordinals in groups.items():
        one_based = set(range(1, total + 1))
        # Most Transformers checkpoints use one-based ``1..N`` shard names.
        # GPT-OSS uses ``of-N`` as the terminal zero-based index, hence
        # ``0..N`` and N + 1 physical shards.
        zero_based = set(range(total + 1))
        if ordinals != one_based and ordinals != zero_based:
            label = parent / f"{prefix}-<shard>-of-{total:05d}{suffix}"
            raise AcquisitionError(f"Sharded model layout is incomplete: {label}")
