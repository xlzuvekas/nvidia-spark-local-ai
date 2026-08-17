"""Read-only discovery of local model artifacts and serving runtimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Any, Mapping
import urllib.error
import urllib.request

from .manifest import ModelSpec


@dataclass(frozen=True, slots=True)
class HuggingFaceSnapshot:
    source: str
    revision: str
    path: Path
    size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class DockerImage:
    repository: str
    tag: str | None
    digest: str | None
    image_id: str
    size: str | None = None

    @property
    def reference(self) -> str:
        return f"{self.repository}:{self.tag}" if self.tag else self.repository


@dataclass(frozen=True, slots=True)
class OllamaModel:
    name: str
    revision: str
    size_bytes: int | None
    modified: str
    architecture: str | None = None
    parameters: str | None = None
    max_context: int | None = None
    embedding_length: int | None = None
    quantization: str | None = None
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Inventory:
    collected_at: str
    python_version: str
    platform: str
    machine: str
    huggingface_snapshots: tuple[HuggingFaceSnapshot, ...]
    docker_images: tuple[DockerImage, ...]
    ollama_models: tuple[OllamaModel, ...]
    ollama_version: str | None = None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelAvailability:
    model_id: str
    source_available: bool
    runtime_available: bool
    details: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.source_available and self.runtime_available


class InventoryError(RuntimeError):
    """Raised when an installed inventory provider cannot be queried."""


def discover_huggingface_snapshots(
    cache_root: str | Path = "data/huggingface/hub",
    *,
    calculate_sizes: bool = False,
) -> tuple[HuggingFaceSnapshot, ...]:
    """Discover complete-looking Hugging Face snapshot directories.

    Size calculation follows the snapshot symlinks into the blob store and is
    optional because it can be slow for very large registries.
    """

    root = Path(cache_root)
    if not root.is_dir():
        return ()
    snapshots: list[HuggingFaceSnapshot] = []
    for repository in sorted(root.glob("models--*")):
        snapshot_root = repository / "snapshots"
        if not snapshot_root.is_dir():
            continue
        source = repository.name.removeprefix("models--").replace("--", "/")
        for snapshot in sorted(snapshot_root.iterdir()):
            if not snapshot.is_dir():
                continue
            snapshots.append(
                HuggingFaceSnapshot(
                    source=source,
                    revision=snapshot.name,
                    path=snapshot,
                    size_bytes=_tree_size(snapshot) if calculate_sizes else None,
                )
            )
    return tuple(snapshots)


def discover_docker_images(
    *, executable: str = "docker", timeout_s: float = 15.0
) -> tuple[DockerImage, ...]:
    """Read the local Docker image table without pulling or modifying images."""

    output = _run(
        executable,
        [
            "image",
            "ls",
            "--digests",
            "--no-trunc",
            "--format",
            "{{json .}}",
        ],
        timeout_s=timeout_s,
    )
    images: list[DockerImage] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise InventoryError(
                f"docker image output line {line_number} was not JSON: {error}"
            ) from error
        repository = _nonempty(row.get("Repository"))
        image_id = _nonempty(row.get("ID"))
        if repository is None or image_id is None:
            continue
        images.append(
            DockerImage(
                repository=repository,
                tag=_none_marker(row.get("Tag")),
                digest=_none_marker(row.get("Digest")),
                image_id=image_id,
                size=_none_marker(row.get("Size")),
            )
        )
    return tuple(images)


def discover_ollama_models(
    *,
    executable: str = "ollama",
    include_metadata: bool = True,
    timeout_s: float = 15.0,
) -> tuple[OllamaModel, ...]:
    """Read Ollama's local registry and, optionally, each model's metadata."""

    output = _run(executable, ["list"], timeout_s=timeout_s)
    models: list[OllamaModel] = []
    for line in output.splitlines()[1:]:
        columns = re.split(r"\s{2,}", line.strip(), maxsplit=3)
        if len(columns) < 2:
            continue
        name, revision = columns[:2]
        size = _parse_size(columns[2]) if len(columns) >= 3 else None
        modified = columns[3] if len(columns) >= 4 else ""
        metadata: Mapping[str, Any] = {}
        if include_metadata:
            shown = _run(executable, ["show", name], timeout_s=timeout_s)
            metadata = _parse_ollama_show(shown)
        models.append(
            OllamaModel(
                name=name,
                revision=revision,
                size_bytes=size,
                modified=modified,
                architecture=metadata.get("architecture"),
                parameters=metadata.get("parameters"),
                max_context=metadata.get("max_context"),
                embedding_length=metadata.get("embedding_length"),
                quantization=metadata.get("quantization"),
                capabilities=metadata.get("capabilities", ()),
            )
        )
    return tuple(models)


def discover_ollama_version(
    endpoint: str = "http://127.0.0.1:11434", *, timeout_s: float = 3.0
) -> str:
    """Return the running local Ollama server version."""

    url = f"{endpoint.rstrip('/')}/api/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise InventoryError(f"cannot query Ollama at {url}: {error}") from error
    version = payload.get("version") if isinstance(payload, dict) else None
    if not isinstance(version, str) or not version:
        raise InventoryError(f"Ollama at {url} returned no version")
    return version


def collect_inventory(
    *,
    huggingface_root: str | Path = "data/huggingface/hub",
    calculate_sizes: bool = False,
    include_docker: bool = True,
    include_ollama: bool = True,
) -> Inventory:
    """Collect all available providers, retaining provider failures as data."""

    errors: list[str] = []
    snapshots = discover_huggingface_snapshots(
        huggingface_root, calculate_sizes=calculate_sizes
    )

    images: tuple[DockerImage, ...] = ()
    if include_docker:
        try:
            images = discover_docker_images()
        except InventoryError as error:
            errors.append(str(error))

    ollama_models: tuple[OllamaModel, ...] = ()
    ollama_version: str | None = None
    if include_ollama:
        try:
            ollama_models = discover_ollama_models()
        except InventoryError as error:
            errors.append(str(error))
        try:
            ollama_version = discover_ollama_version()
        except InventoryError as error:
            errors.append(str(error))

    return Inventory(
        collected_at=datetime.now(timezone.utc).isoformat(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        huggingface_snapshots=snapshots,
        docker_images=images,
        ollama_models=ollama_models,
        ollama_version=ollama_version,
        errors=tuple(errors),
    )


def assess_model_availability(
    models: Mapping[str, ModelSpec], inventory: Inventory
) -> dict[str, ModelAvailability]:
    """Match manifest entries to exact local checkpoints and runtimes."""

    hf_keys = {
        (_snapshot_cache_dir(snapshot), snapshot.source, snapshot.revision)
        for snapshot in inventory.huggingface_snapshots
    }
    hf_sources = {
        (_snapshot_cache_dir(snapshot), snapshot.source)
        for snapshot in inventory.huggingface_snapshots
    }
    hf_snapshots = {
        (_snapshot_cache_dir(snapshot), snapshot.source, snapshot.revision): snapshot.path
        for snapshot in inventory.huggingface_snapshots
    }
    ollama = {model.name: model for model in inventory.ollama_models}
    image_references = {image.reference for image in inventory.docker_images}
    availability: dict[str, ModelAvailability] = {}
    for model in models.values():
        details: list[str] = []
        if model.backend == "ollama":
            local = ollama.get(model.source)
            source_available = local is not None and (
                model.revision is None or local.revision.startswith(model.revision)
            )
            runtime_available = inventory.ollama_version is not None
            if local is None:
                details.append("Ollama model is not registered locally")
            elif model.revision and not local.revision.startswith(model.revision):
                details.append("Ollama model revision differs from the manifest")
            if not runtime_available:
                details.append("Ollama endpoint is unavailable")
        elif model.backend == "llamacpp":
            snapshot = hf_snapshots.get(
                (model.cache_dir, model.source, str(model.revision))
            )
            model_file = snapshot / str(model.model_file) if snapshot else None
            model_file_available = bool(model_file and model_file.is_file())
            mmproj_file = (
                snapshot / str(model.mmproj_file)
                if snapshot and model.mmproj_file
                else None
            )
            mmproj_available = model.mmproj_file is None or bool(
                mmproj_file and mmproj_file.is_file()
            )
            draft_snapshot = (
                hf_snapshots.get(
                    (
                        model.cache_dir,
                        str(model.draft_source),
                        str(model.draft_revision),
                    )
                )
                if model.draft_source is not None
                else None
            )
            draft_model_file = (
                draft_snapshot / str(model.draft_model_file)
                if draft_snapshot and model.draft_model_file
                else None
            )
            draft_model_available = model.draft_model_file is None or bool(
                draft_model_file and draft_model_file.is_file()
            )
            source_available = (
                model_file_available
                and mmproj_available
                and draft_model_available
            )
            runtime_available = bool(
                model.runtime_binary and Path(model.runtime_binary).is_file()
            )
            if not model_file_available:
                details.append("exact GGUF file is not cached")
            if not mmproj_available:
                details.append("exact multimodal projector is not cached")
            if not draft_model_available:
                details.append("exact draft GGUF file is not cached")
            if not runtime_available:
                details.append("pinned llama-server binary is unavailable")
        else:
            target_available = (
                (model.cache_dir, model.source, model.revision) in hf_keys
                if model.revision
                else (model.cache_dir, model.source) in hf_sources
            )
            draft_available = model.draft_source is None or (
                model.cache_dir,
                model.draft_source,
                model.draft_revision,
            ) in hf_keys
            source_available = target_available and draft_available
            if model.backend == "transformers":
                runtime_available = bool(
                    model.runtime_python and Path(model.runtime_python).is_file()
                )
                if not runtime_available:
                    details.append("certified Transformers runtime is unavailable")
            elif model.image and model.image_digest:
                image_name = model.image.split("@sha256:", 1)[0]
                runtime_available = any(
                    image.digest == model.image_digest
                    and image_name in {image.repository, image.reference}
                    for image in inventory.docker_images
                )
            else:
                runtime_available = bool(
                    model.image and model.image in image_references
                )
            if not target_available:
                details.append("checkpoint revision is not cached")
            if not draft_available:
                details.append("draft checkpoint revision is not cached")
            if not runtime_available and model.backend != "transformers":
                details.append("container image or digest is not cached")
        availability[model.id] = ModelAvailability(
            model_id=model.id,
            source_available=source_available,
            runtime_available=runtime_available,
            details=tuple(details),
        )
    return availability


def inventory_to_dict(inventory: Inventory) -> dict[str, Any]:
    """Return a JSON-serializable representation suitable for a frozen plan."""

    result = asdict(inventory)
    for snapshot in result["huggingface_snapshots"]:
        snapshot["path"] = str(snapshot["path"])
    return result


def _run(
    executable: str, arguments: list[str], *, timeout_s: float
) -> str:
    path = shutil.which(executable)
    if path is None:
        raise InventoryError(f"{executable!r} is not installed")
    try:
        completed = subprocess.run(
            [path, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        raise InventoryError(
            f"{executable} {' '.join(arguments)} timed out after {timeout_s:g}s"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise InventoryError(f"{executable} {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _parse_ollama_show(output: str) -> dict[str, Any]:
    model_fields: dict[str, str] = {}
    capabilities: list[str] = []
    section = ""
    for raw_line in output.splitlines():
        stripped = raw_line.strip()
        indentation = len(raw_line) - len(raw_line.lstrip())
        if indentation == 2 and stripped:
            section = stripped
            continue
        if section == "Model" and indentation >= 4 and stripped:
            columns = re.split(r"\s{2,}", stripped, maxsplit=1)
            if len(columns) == 2:
                model_fields[columns[0]] = columns[1]
        elif section == "Capabilities" and indentation >= 4 and stripped:
            capabilities.append(stripped.split()[0])
    return {
        "architecture": model_fields.get("architecture"),
        "parameters": model_fields.get("parameters"),
        "max_context": _parse_int(model_fields.get("context length")),
        "embedding_length": _parse_int(model_fields.get("embedding length")),
        "quantization": model_fields.get("quantization"),
        "capabilities": tuple(capabilities),
    }


def _tree_size(root: Path) -> int:
    size = 0
    seen: set[tuple[int, int]] = set()
    for directory, _, files in os.walk(root):
        for filename in files:
            path = Path(directory) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity not in seen:
                seen.add(identity)
                size += stat.st_size
    return size


def _snapshot_cache_dir(snapshot: HuggingFaceSnapshot) -> str:
    user_root = (Path.home() / ".cache" / "huggingface" / "hub").resolve()
    try:
        snapshot.path.resolve().relative_to(user_root)
    except ValueError:
        return "project"
    return "user"


def _parse_size(value: str) -> int | None:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?B)", value.strip())
    if not match:
        return None
    multipliers = {
        "B": 1,
        "KB": 1_000,
        "MB": 1_000_000,
        "GB": 1_000_000_000,
        "TB": 1_000_000_000_000,
    }
    return int(float(match.group(1)) * multipliers[match.group(2)])


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.replace(",", ""))
    except ValueError:
        return None


def _nonempty(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _none_marker(value: Any) -> str | None:
    normalized = _nonempty(value)
    return None if normalized in {None, "<none>"} else normalized
