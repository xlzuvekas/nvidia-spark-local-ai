"""Typed, dependency-free loaders for Spark benchmark manifests."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import tomllib
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit


SCHEMA_VERSION = 1
KNOWN_TASKS = frozenset(
    {"chat", "embeddings", "json", "rerank", "thinking", "tools", "vision"}
)
KNOWN_BACKENDS = frozenset({"external", "ollama", "vllm"})
KNOWN_SUPPORT_STATUSES = frozenset(
    {
        "exploratory",
        "spark_other_backend",
        "spark_vllm_matrix",
        "spark_vllm_recipe",
    }
)
KNOWN_CASE_KINDS = frozenset({"capability", "concurrency", "decode", "prefill"})
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_MODEL_KEYS = frozenset(
    {
        "id",
        "description",
        "backend",
        "source",
        "revision",
        "served_name",
        "tasks",
        "architecture",
        "quantization",
        "lifecycle",
        "image",
        "image_digest",
        "max_context",
        "native_context",
        "startup_timeout_s",
        "estimated_ram_gib",
        "endpoint",
        "args",
        "cache_dir",
        "request_body_json",
        "support_status",
    }
)
_CASE_KEYS = frozenset(
    {
        "id",
        "kind",
        "requires",
        "warmups",
        "repetitions",
        "concurrency",
        "prompt_repetitions",
        "max_output_tokens",
        "temperature",
    }
)


class ManifestError(ValueError):
    """Raised when a manifest is malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One reproducible model-serving configuration."""

    id: str
    backend: str
    source: str
    served_name: str
    tasks: tuple[str, ...]
    image: str | None = None
    max_context: int = 0
    native_context: int | None = None
    startup_timeout_s: int = 1_800
    args: tuple[str, ...] = ()
    endpoint: str = "http://127.0.0.1:8000/v1"
    estimated_ram_gib: float | None = None
    revision: str | None = None
    image_digest: str | None = None
    architecture: str = "unknown"
    quantization: str | None = None
    lifecycle: str = "docker"
    description: str = ""
    cache_dir: str = "project"
    request_body_json: str | None = None
    support_status: str = "exploratory"


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """A workload case expanded across compatible model configurations."""

    id: str
    kind: str
    requires: tuple[str, ...]
    warmups: int = 0
    repetitions: int = 1
    max_output_tokens: int = 1
    temperature: float = 0.0
    concurrency: int = 1
    prompt_repetitions: int = 0


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    """A named collection of benchmark cases."""

    id: str
    cases: tuple[CaseSpec, ...]
    description: str = ""
    schema_version: int = SCHEMA_VERSION


def load_models(path: str | Path) -> dict[str, ModelSpec]:
    """Load and validate a model registry keyed by stable configuration ID."""

    manifest_path = Path(path)
    document = _read_toml(manifest_path)
    _reject_unknown(document, {"schema_version", "models"}, str(manifest_path))
    _check_schema_version(document, manifest_path)
    rows = document.get("models")
    if not isinstance(rows, list) or not rows:
        raise ManifestError(f"{manifest_path}: 'models' must be a non-empty array")

    models: dict[str, ModelSpec] = {}
    for index, row in enumerate(rows):
        context = f"{manifest_path}: models[{index}]"
        if not isinstance(row, dict):
            raise ManifestError(f"{context} must be a table")
        _reject_unknown(row, _MODEL_KEYS, context)
        backend = _required_string(row, "backend", context).lower()
        model = ModelSpec(
            id=_required_string(row, "id", context),
            backend=backend,
            source=_required_string(row, "source", context),
            served_name=_required_string(row, "served_name", context),
            tasks=_string_tuple(row, "tasks", context, required=True),
            image=_optional_string(row, "image", context),
            max_context=_required_int(row, "max_context", context),
            native_context=_optional_int(
                row, "native_context", context, default=None
            ),
            startup_timeout_s=_optional_int(
                row, "startup_timeout_s", context, default=1_800
            ),
            args=_string_tuple(row, "args", context),
            endpoint=_optional_string(row, "endpoint", context)
            or _default_endpoint(backend),
            estimated_ram_gib=_optional_number(
                row, "estimated_ram_gib", context
            ),
            revision=_optional_string(row, "revision", context),
            image_digest=_optional_string(row, "image_digest", context),
            architecture=_optional_string(row, "architecture", context)
            or "unknown",
            quantization=_optional_string(row, "quantization", context),
            lifecycle=_optional_string(row, "lifecycle", context) or "docker",
            description=_optional_string(row, "description", context) or "",
            cache_dir=_optional_string(row, "cache_dir", context) or "project",
            request_body_json=_optional_string(
                row, "request_body_json", context
            ),
            support_status=(
                _optional_string(row, "support_status", context) or "exploratory"
            ),
        )
        validate_model(model, context=context)
        if model.id in models:
            raise ManifestError(f"{context}: duplicate model id {model.id!r}")
        models[model.id] = model
    return models


def load_suite(path: str | Path) -> SuiteSpec:
    """Load and validate one benchmark suite."""

    manifest_path = Path(path)
    document = _read_toml(manifest_path)
    _reject_unknown(
        document, {"schema_version", "suite", "cases"}, str(manifest_path)
    )
    version = _check_schema_version(document, manifest_path)
    suite_row = document.get("suite")
    if not isinstance(suite_row, dict):
        raise ManifestError(f"{manifest_path}: 'suite' must be a table")
    _reject_unknown(suite_row, {"id", "description"}, f"{manifest_path}: suite")

    rows = document.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ManifestError(f"{manifest_path}: 'cases' must be a non-empty array")
    cases: list[CaseSpec] = []
    for index, row in enumerate(rows):
        context = f"{manifest_path}: cases[{index}]"
        if not isinstance(row, dict):
            raise ManifestError(f"{context} must be a table")
        _reject_unknown(row, _CASE_KEYS, context)
        case = CaseSpec(
            id=_required_string(row, "id", context),
            kind=_required_string(row, "kind", context).lower(),
            requires=_string_tuple(row, "requires", context, required=True),
            warmups=_optional_int(row, "warmups", context, default=0),
            repetitions=_optional_int(row, "repetitions", context, default=1),
            max_output_tokens=_optional_int(
                row, "max_output_tokens", context, default=1
            ),
            temperature=_optional_number(row, "temperature", context, default=0.0),
            concurrency=_optional_int(row, "concurrency", context, default=1),
            prompt_repetitions=_optional_int(
                row, "prompt_repetitions", context, default=0
            ),
        )
        validate_case(case, context=context)
        cases.append(case)

    suite = SuiteSpec(
        id=_required_string(suite_row, "id", f"{manifest_path}: suite"),
        description=_optional_string(
            suite_row, "description", f"{manifest_path}: suite"
        )
        or "",
        cases=tuple(cases),
        schema_version=version,
    )
    validate_suite(suite, context=str(manifest_path))
    return suite


def validate_model(model: ModelSpec, *, context: str = "model") -> None:
    """Validate a model dataclass, including instances built outside the loader."""

    _validate_id(model.id, f"{context}.id")
    for name, value in (
        ("backend", model.backend),
        ("source", model.source),
        ("served_name", model.served_name),
        ("architecture", model.architecture),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ManifestError(f"{context}.{name} must be a non-empty string")
    if model.backend not in KNOWN_BACKENDS:
        raise ManifestError(
            f"{context}.backend must be one of {sorted(KNOWN_BACKENDS)}"
        )
    _validate_unique_values(model.tasks, f"{context}.tasks", KNOWN_TASKS)
    if model.max_context <= 0:
        raise ManifestError(f"{context}.max_context must be positive")
    if model.native_context is not None and model.native_context < model.max_context:
        raise ManifestError(
            f"{context}.native_context must be at least max_context"
        )
    if model.startup_timeout_s <= 0:
        raise ManifestError(f"{context}.startup_timeout_s must be positive")
    if model.estimated_ram_gib is not None and model.estimated_ram_gib <= 0:
        raise ManifestError(f"{context}.estimated_ram_gib must be positive")
    if model.lifecycle not in {"docker", "existing"}:
        raise ManifestError(f"{context}.lifecycle must be 'docker' or 'existing'")
    if model.cache_dir not in {"project", "user"}:
        raise ManifestError(f"{context}.cache_dir must be 'project' or 'user'")
    if model.support_status not in KNOWN_SUPPORT_STATUSES:
        raise ManifestError(
            f"{context}.support_status must be one of {sorted(KNOWN_SUPPORT_STATUSES)}"
        )
    if model.lifecycle == "docker" and not model.image:
        raise ManifestError(f"{context}.image is required for Docker lifecycle")
    if model.image_digest and not _DIGEST_PATTERN.fullmatch(model.image_digest):
        raise ManifestError(f"{context}.image_digest must be a sha256 digest")
    _validate_endpoint(model.endpoint, f"{context}.endpoint")
    if any(not isinstance(argument, str) or not argument for argument in model.args):
        raise ManifestError(f"{context}.args must contain non-empty strings")
    if model.request_body_json is not None:
        try:
            request_body = json.loads(model.request_body_json)
        except json.JSONDecodeError as error:
            raise ManifestError(
                f"{context}.request_body_json must contain valid JSON"
            ) from error
        if not isinstance(request_body, dict):
            raise ManifestError(
                f"{context}.request_body_json must decode to a JSON object"
            )


def validate_case(case: CaseSpec, *, context: str = "case") -> None:
    """Validate a case dataclass."""

    _validate_id(case.id, f"{context}.id")
    if case.kind not in KNOWN_CASE_KINDS:
        raise ManifestError(
            f"{context}.kind must be one of {sorted(KNOWN_CASE_KINDS)}"
        )
    _validate_unique_values(case.requires, f"{context}.requires", KNOWN_TASKS)
    if case.warmups < 0:
        raise ManifestError(f"{context}.warmups must not be negative")
    if case.repetitions <= 0:
        raise ManifestError(f"{context}.repetitions must be positive")
    if case.concurrency <= 0:
        raise ManifestError(f"{context}.concurrency must be positive")
    if case.prompt_repetitions < 0:
        raise ManifestError(f"{context}.prompt_repetitions must not be negative")
    if case.max_output_tokens <= 0:
        raise ManifestError(f"{context}.max_output_tokens must be positive")
    if not 0 <= case.temperature <= 2:
        raise ManifestError(f"{context}.temperature must be between 0 and 2")
    if case.kind == "decode" and case.prompt_repetitions != 0:
        raise ManifestError(
            f"{context}.prompt_repetitions must be 0 for decode cases"
        )
    if case.kind == "prefill" and case.prompt_repetitions <= 0:
        raise ManifestError(
            f"{context}.prompt_repetitions must be positive for prefill cases"
        )


def validate_suite(suite: SuiteSpec, *, context: str = "suite") -> None:
    """Validate suite-level identity and case uniqueness."""

    _validate_id(suite.id, f"{context}.id")
    if suite.schema_version != SCHEMA_VERSION:
        raise ManifestError(
            f"{context}.schema_version must be {SCHEMA_VERSION}, "
            f"got {suite.schema_version}"
        )
    if not suite.cases:
        raise ManifestError(f"{context}.cases must not be empty")
    seen: set[str] = set()
    for index, case in enumerate(suite.cases):
        validate_case(case, context=f"{context}.cases[{index}]")
        if case.id in seen:
            raise ManifestError(f"{context}: duplicate case id {case.id!r}")
        seen.add(case.id)


def validate_models(models: Mapping[str, ModelSpec] | Iterable[ModelSpec]) -> None:
    """Validate a registry or iterable and reject duplicate or mismatched IDs."""

    items = models.items() if isinstance(models, Mapping) else None
    values = models.values() if isinstance(models, Mapping) else models
    seen: set[str] = set()
    for index, model in enumerate(values):
        validate_model(model, context=f"models[{index}]")
        if model.id in seen:
            raise ManifestError(f"duplicate model id {model.id!r}")
        seen.add(model.id)
    if items is not None:
        for key, model in items:
            if key != model.id:
                raise ManifestError(
                    f"model registry key {key!r} does not match id {model.id!r}"
                )


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except OSError as error:
        raise ManifestError(f"cannot read {path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ManifestError(f"invalid TOML in {path}: {error}") from error


def _check_schema_version(document: Mapping[str, Any], path: Path) -> int:
    version = document.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ManifestError(f"{path}: schema_version must be an integer")
    if version != SCHEMA_VERSION:
        raise ManifestError(
            f"{path}: unsupported schema_version {version}; expected {SCHEMA_VERSION}"
        )
    return version


def _reject_unknown(
    table: Mapping[str, Any], allowed: set[str] | frozenset[str], context: str
) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise ManifestError(f"{context}: unknown field(s): {', '.join(unknown)}")


def _required_string(table: Mapping[str, Any], key: str, context: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _optional_string(
    table: Mapping[str, Any], key: str, context: str
) -> str | None:
    if key not in table:
        return None
    return _required_string(table, key, context)


def _required_int(table: Mapping[str, Any], key: str, context: str) -> int:
    if key not in table:
        raise ManifestError(f"{context}.{key} is required")
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{context}.{key} must be an integer")
    return value


def _optional_int(
    table: Mapping[str, Any], key: str, context: str, *, default: int | None
) -> int | None:
    if key not in table:
        return default
    return _required_int(table, key, context)


def _optional_number(
    table: Mapping[str, Any],
    key: str,
    context: str,
    *,
    default: float | None = None,
) -> float | None:
    if key not in table:
        return default
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{context}.{key} must be a number")
    return float(value)


def _string_tuple(
    table: Mapping[str, Any],
    key: str,
    context: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    if key not in table:
        if required:
            raise ManifestError(f"{context}.{key} is required")
        return ()
    value = table[key]
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ManifestError(f"{context}.{key} must be an array of strings")
    if required and not value:
        raise ManifestError(f"{context}.{key} must not be empty")
    return tuple(item.strip() for item in value)


def _validate_id(value: str, context: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ManifestError(
            f"{context} must match {_ID_PATTERN.pattern!r}; got {value!r}"
        )


def _validate_unique_values(
    values: tuple[str, ...], context: str, allowed: frozenset[str]
) -> None:
    if not values:
        raise ManifestError(f"{context} must not be empty")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ManifestError(f"{context} contains unknown values: {', '.join(unknown)}")
    if len(values) != len(set(values)):
        raise ManifestError(f"{context} must not contain duplicates")


def _validate_endpoint(endpoint: str, context: str) -> None:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ManifestError(f"{context} must be an HTTP(S) URL")
    if parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ManifestError(f"{context} must use a loopback host")


def _default_endpoint(backend: str) -> str:
    if backend == "ollama":
        return "http://127.0.0.1:11434/v1"
    return "http://127.0.0.1:8000/v1"
