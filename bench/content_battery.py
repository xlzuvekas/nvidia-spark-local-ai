"""Fresh-prompt content battery for an already-managed loopback server.

The prompt set is copied verbatim from ``bench-matrix.sh`` in
``hasso5703/dgx-spark-qwen38`` at commit
``3590fb29296b1babd85405daad1eef1c4a3ebe0f``.  Unlike that script, this
protocol never repeats a complete prompt and never persists generated text.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request


BATTERY_ID = "dgx-spark-qwen38-content"
PROMPT_SET_VERSION = 1
PROTOCOL_VERSION = 1
REPETITIONS_PER_PROMPT = 3
MAX_OUTPUT_TOKENS = 680
MIN_OUTPUT_TOKENS = 50
TEMPERATURE = 0.0
WARMUP_MAX_OUTPUT_TOKENS = 24
MAX_API_KEY_BYTES = 16 * 1024


class ContentBatteryError(RuntimeError):
    """Raised when the endpoint or a measurement violates the protocol."""


@dataclass(frozen=True, slots=True)
class Probe:
    """One frozen upstream prompt plus non-content reporting metadata."""

    id: str
    language: str
    category: str
    prompt: str


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    """Scalar-only metrics retained for one streamed request."""

    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    e2e_s: float
    decode_s: float
    decode_tps: float
    output_tps: float
    emission_events: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


# Probe battery v1 from upstream bench-matrix.sh. Do not edit these strings in
# place: create a new prompt-set version if the workload changes.
PROBES = (
    Probe(
        id="math-en-eval-style",
        language="en",
        category="math",
        prompt=(
            "Natalia sold clips to 48 friends in April, half as many in May, "
            "and 3x May's amount in June. How many clips did she sell in total? "
            "Think step by step."
        ),
    ),
    Probe(
        id="code-en",
        language="en",
        category="code",
        prompt=(
            "Write a Python class implementing an LRU cache with O(1) get and "
            "put, with docstrings and a small usage example."
        ),
    ),
    Probe(
        id="code-de",
        language="de",
        category="code",
        prompt=(
            "Schreibe eine Python-Klasse RingBuffer mit push/pop in O(1), "
            "Docstrings und einem kleinen Test."
        ),
    ),
    Probe(
        id="technical-explanation-fr",
        language="fr",
        category="technical_explanation",
        prompt=(
            "Explique en d\u00e9tail comment fonctionne un index B-tree dans une base "
            "de donn\u00e9es relationnelle, avec un exemple concret d'insertion."
        ),
    ),
    Probe(
        id="reasoning-fr",
        language="fr",
        category="reasoning",
        prompt=(
            "Un train part de Paris \u00e0 14h00 \u00e0 160 km/h vers Lyon (465 km). Un "
            "autre part de Lyon \u00e0 14h30 \u00e0 120 km/h vers Paris. \u00c0 quelle heure "
            "se croisent-ils ? Montre ton raisonnement."
        ),
    ),
    Probe(
        id="free-prose-en",
        language="en",
        category="free_prose",
        prompt=(
            "Describe in detail a walk through an autumn forest: the colors of "
            "the leaves, the sounds, the smell after rain, and the thoughts that "
            "cross your mind."
        ),
    ),
    Probe(
        id="free-prose-fr",
        language="fr",
        category="free_prose",
        prompt=(
            "D\u00e9cris en d\u00e9tail une promenade dans une for\u00eat d'automne : les "
            "couleurs des feuilles, les sons, l'odeur apr\u00e8s la pluie, et les "
            "pens\u00e9es qui traversent l'esprit."
        ),
    ),
    Probe(
        id="free-prose-de",
        language="de",
        category="free_prose",
        prompt=(
            "Beschreibe ausf\u00fchrlich einen Spaziergang durch einen herbstlichen "
            "Wald: die Farben der Bl\u00e4tter, die Ger\u00e4usche, den Geruch nach Regen, "
            "und die Gedanken, die einem dabei durch den Kopf gehen."
        ),
    ),
)


# Every measured tag is a rotation of one of three permutations of the same
# eight single-digit fields. The textual tag length and field multiset are
# therefore identical. Observed prompt-token equality is also enforced for the
# three repetitions of every probe, so tokenizer-specific differences fail the
# run instead of silently biasing it.
_TAG_BASES = (
    (0, 1, 2, 3, 4, 5, 6, 7),
    (0, 2, 4, 6, 1, 3, 5, 7),
    (0, 3, 6, 1, 4, 7, 2, 5),
)
_ROUND_ORDERS = (
    tuple(range(len(PROBES))),
    tuple(reversed(range(len(PROBES)))),
    (3, 5, 0, 7, 1, 4, 2, 6),
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Prevent a loopback server from redirecting the client off-device."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        raise ContentBatteryError("The loopback endpoint attempted an HTTP redirect")


def _default_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirect(),
    )


def canonical_loopback_base_url(value: str) -> str:
    """Validate and canonicalize a literal loopback OpenAI ``/v1`` URL."""

    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "http":
        raise ContentBatteryError("Base URL must use http on the loopback interface")
    if parsed.username is not None or parsed.password is not None:
        raise ContentBatteryError("Base URL must not contain credentials")
    if parsed.hostname not in {"127.0.0.1", "::1"}:
        raise ContentBatteryError("Base URL host must be the literal loopback address")
    try:
        port = parsed.port
    except ValueError as error:
        raise ContentBatteryError("Base URL contains an invalid port") from error
    if port is None:
        raise ContentBatteryError("Base URL must include an explicit port")
    if parsed.path.rstrip("/") != "/v1" or parsed.query or parsed.fragment:
        raise ContentBatteryError("Base URL must end at /v1 without query or fragment")
    host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
    return f"http://{host}:{port}/v1"


def read_api_key(path: Path | None) -> str | None:
    """Read one bounded, non-symlinked bearer token without logging it."""

    if path is None:
        return None
    if path.is_symlink():
        raise ContentBatteryError("API key file must not be a symbolic link")
    try:
        metadata = path.stat()
    except OSError as error:
        raise ContentBatteryError("Could not read API key file") from error
    if not path.is_file():
        raise ContentBatteryError("API key path must be a regular file")
    if metadata.st_size <= 0 or metadata.st_size > MAX_API_KEY_BYTES:
        raise ContentBatteryError("API key file has an invalid size")
    try:
        key = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as error:
        raise ContentBatteryError("Could not read API key file") from error
    if not key or any(character in key for character in "\r\n\0"):
        raise ContentBatteryError("API key file does not contain one valid token")
    return key


def _headers(api_key: str | None, *, json_content: bool = False) -> dict[str, str]:
    headers = {"Content-Type": "application/json"} if json_content else {}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _tag(probe_index: int, repetition: int) -> str:
    base = _TAG_BASES[repetition]
    rotated = base[probe_index:] + base[:probe_index]
    return " ".join(str(value) for value in rotated)


def tagged_prompt(probe_index: int, repetition: int) -> str:
    """Return one fixed-format prompt that differs before the upstream text."""

    tag = _tag(probe_index, repetition)
    return (
        f"Benchmark tag {tag}. Ignore this tag and do not mention it.\n\n"
        f"{PROBES[probe_index].prompt}"
    )


def _usage_integer(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContentBatteryError(f"Streaming response lacked exact usage.{key}")
    return value


def verify_served_model(
    *,
    base_url: str,
    model: str,
    timeout_s: float,
    api_key: str | None = None,
    opener: Any | None = None,
) -> None:
    """Require the requested model ID to be advertised by the local endpoint."""

    client = opener or _default_opener()
    request = urllib.request.Request(
        f"{base_url}/models",
        headers=_headers(api_key),
        method="GET",
    )
    try:
        with client.open(request, timeout=timeout_s) as response:
            payload = json.load(response)
    except ContentBatteryError:
        raise
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise ContentBatteryError("Could not verify the loopback model endpoint") from error
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ContentBatteryError("Model endpoint did not return an OpenAI data list")
    model_ids = {
        item.get("id")
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if model not in model_ids:
        raise ContentBatteryError(f"Requested model is not advertised: {model}")


def stream_request(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
    api_key: str | None = None,
    opener: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RequestMetrics:
    """Measure one SSE request while discarding all generated text."""

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers=_headers(api_key, json_content=True),
    )
    client = opener or _default_opener()
    started = clock()
    first_output_at: float | None = None
    usage: dict[str, Any] | None = None
    emission_events = 0
    try:
        with client.open(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].lstrip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as error:
                    raise ContentBatteryError("Streaming response contained invalid JSON") from error
                if not isinstance(event, dict):
                    raise ContentBatteryError("Streaming response event was not an object")
                if event.get("usage") is not None:
                    if not isinstance(event["usage"], dict):
                        raise ContentBatteryError("Streaming response usage was not an object")
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not isinstance(choices, list):
                    raise ContentBatteryError("Streaming response choices was not a list")
                emitted = False
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta") or {}
                    if not isinstance(delta, dict):
                        continue
                    for field in ("content", "reasoning", "reasoning_content"):
                        value = delta.get(field)
                        if isinstance(value, str) and value:
                            emitted = True
                            break
                    if emitted:
                        break
                if emitted:
                    emission_events += 1
                    if first_output_at is None:
                        first_output_at = clock()
    except ContentBatteryError:
        raise
    except urllib.error.HTTPError as error:
        error.close()
        raise ContentBatteryError(
            f"Loopback chat endpoint returned HTTP {error.code}"
        ) from error
    except (OSError, urllib.error.URLError) as error:
        raise ContentBatteryError("Loopback streaming request failed") from error
    finished = clock()
    if usage is None:
        raise ContentBatteryError("Streaming response omitted exact token usage")
    if first_output_at is None:
        raise ContentBatteryError("Streaming response emitted no content or reasoning")
    prompt_tokens = _usage_integer(usage, "prompt_tokens")
    completion_tokens = _usage_integer(usage, "completion_tokens")
    ttft_s = first_output_at - started
    e2e_s = finished - started
    decode_s = e2e_s - ttft_s
    if (
        not math.isfinite(ttft_s)
        or not math.isfinite(e2e_s)
        or not math.isfinite(decode_s)
        or ttft_s < 0
        or e2e_s <= 0
        or decode_s <= 0
    ):
        raise ContentBatteryError("Streaming request produced invalid timing values")
    return RequestMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_s=ttft_s,
        e2e_s=e2e_s,
        decode_s=decode_s,
        decode_tps=max(completion_tokens - 1, 0) / decode_s,
        output_tps=completion_tokens / e2e_s,
        emission_events=emission_events,
    )


def _round(value: float) -> float:
    return round(value, 9)


def _metrics_payload(metrics: RequestMetrics) -> dict[str, int | float]:
    return {
        "prompt_tokens": metrics.prompt_tokens,
        "completion_tokens": metrics.completion_tokens,
        "ttft_s": _round(metrics.ttft_s),
        "e2e_s": _round(metrics.e2e_s),
        "decode_s": _round(metrics.decode_s),
        "decode_tps": _round(metrics.decode_tps),
        "output_tps": _round(metrics.output_tps),
        "emission_events": metrics.emission_events,
    }


def _summary(metrics: list[RequestMetrics]) -> dict[str, int | float]:
    completion_tokens = sum(item.completion_tokens for item in metrics)
    prompt_tokens = sum(item.prompt_tokens for item in metrics)
    e2e_s = sum(item.e2e_s for item in metrics)
    decode_s = sum(item.decode_s for item in metrics)
    decode_tokens = sum(max(item.completion_tokens - 1, 0) for item in metrics)
    return {
        "requests": len(metrics),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "aggregate_output_tps": _round(completion_tokens / e2e_s),
        "aggregate_decode_tps": _round(decode_tokens / decode_s),
        "median_ttft_s": _round(statistics.median(item.ttft_s for item in metrics)),
        "median_e2e_s": _round(statistics.median(item.e2e_s for item in metrics)),
        "median_decode_tps": _round(
            statistics.median(item.decode_tps for item in metrics)
        ),
        "minimum_decode_tps": _round(min(item.decode_tps for item in metrics)),
        "maximum_decode_tps": _round(max(item.decode_tps for item in metrics)),
    }


def run_battery(
    *,
    base_url: str,
    model: str,
    timeout_s: float = 900.0,
    api_key: str | None = None,
    opener: Any | None = None,
    verify_function: Callable[..., None] = verify_served_model,
    request_function: Callable[..., RequestMetrics] = stream_request,
) -> dict[str, Any]:
    """Run the frozen battery against one already-running loopback endpoint."""

    canonical = canonical_loopback_base_url(base_url)
    if not model or model.strip() != model:
        raise ContentBatteryError("Model ID must be a non-empty exact string")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ContentBatteryError("Timeout must be a positive finite number")
    client = opener or _default_opener()
    verify_function(
        base_url=canonical,
        model=model,
        timeout_s=timeout_s,
        api_key=api_key,
        opener=client,
    )
    warmup = request_function(
        base_url=canonical,
        model=model,
        prompt="Battery warmup only. Reply with OK.",
        max_tokens=WARMUP_MAX_OUTPUT_TOKENS,
        timeout_s=timeout_s,
        api_key=api_key,
        opener=client,
    )
    if warmup.completion_tokens < 1:
        raise ContentBatteryError("Warmup returned no completion tokens")

    grouped: dict[int, list[tuple[int, int, RequestMetrics]]] = {
        index: [] for index in range(len(PROBES))
    }
    all_metrics: list[RequestMetrics] = []
    seen_prompts: set[str] = set()
    measured_order = 0
    for repetition, order in enumerate(_ROUND_ORDERS):
        for probe_index in order:
            prompt = tagged_prompt(probe_index, repetition)
            if prompt in seen_prompts:
                raise ContentBatteryError("Fresh-prompt protocol produced a duplicate")
            seen_prompts.add(prompt)
            measured_order += 1
            metrics = request_function(
                base_url=canonical,
                model=model,
                prompt=prompt,
                max_tokens=MAX_OUTPUT_TOKENS,
                timeout_s=timeout_s,
                api_key=api_key,
                opener=client,
            )
            if metrics.completion_tokens < MIN_OUTPUT_TOKENS:
                raise ContentBatteryError(
                    f"Short output for {PROBES[probe_index].id} repetition "
                    f"{repetition + 1}: {metrics.completion_tokens} < "
                    f"{MIN_OUTPUT_TOKENS} tokens"
                )
            grouped[probe_index].append((repetition, measured_order, metrics))
            all_metrics.append(metrics)

    probes_payload: list[dict[str, Any]] = []
    for probe_index, probe in enumerate(PROBES):
        samples = sorted(grouped[probe_index], key=lambda item: item[0])
        if len(samples) != REPETITIONS_PER_PROMPT:
            raise ContentBatteryError(f"Incomplete repetitions for {probe.id}")
        observed_prompt_lengths = {item[2].prompt_tokens for item in samples}
        if len(observed_prompt_lengths) != 1:
            raise ContentBatteryError(
                f"Unique tags did not preserve prompt-token length for {probe.id}"
            )
        sample_payload = []
        sample_metrics = []
        for repetition, order, metrics in samples:
            sample_metrics.append(metrics)
            sample_payload.append(
                {
                    "sample_id": f"{probe.id}-r{repetition + 1}",
                    "repetition": repetition + 1,
                    "measured_order": order,
                    **_metrics_payload(metrics),
                }
            )
        probes_payload.append(
            {
                "id": probe.id,
                "language": probe.language,
                "category": probe.category,
                "samples": sample_payload,
                "summary": _summary(sample_metrics),
            }
        )

    return {
        "schema_version": 1,
        "battery": {
            "id": BATTERY_ID,
            "prompt_set_version": PROMPT_SET_VERSION,
            "protocol_version": PROTOCOL_VERSION,
        },
        "endpoint": canonical,
        "model": model,
        "protocol": {
            "transport": "openai_chat_completions_sse",
            "warmups": 1,
            "repetitions_per_prompt": REPETITIONS_PER_PROMPT,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "minimum_output_tokens": MIN_OUTPUT_TOKENS,
            "fresh_prompt_tags": "eight-single-digit-permutations-v1",
            "aggregate_output_tps": "sum_completion_tokens_over_sum_e2e_seconds",
            "aggregate_decode_tps": "sum_completion_tokens_minus_first_over_sum_post_ttft_seconds",
        },
        "warmup": {"id": "warmup", **_metrics_payload(warmup)},
        "probes": probes_payload,
        "summary": _summary(all_metrics),
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
    """Atomically create one result file without overwriting prior evidence."""

    if path.exists() or path.is_symlink():
        raise ContentBatteryError(f"Output path already exists: {path}")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ContentBatteryError(f"Output path already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen fresh-prompt content battery against an already-managed "
            "loopback OpenAI-compatible server."
        )
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="literal loopback /v1 URL, for example http://127.0.0.1:30000/v1",
    )
    parser.add_argument("--model", required=True, help="exact served model ID")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="optional file containing one bearer token; its value is never persisted",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new JSON result path; existing files are never overwritten",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=900.0,
        help="per-request timeout (default: 900)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ContentBatteryError(f"Output path already exists: {args.output}")
        api_key = read_api_key(args.api_key_file)
        result = run_battery(
            base_url=args.base_url,
            model=args.model,
            timeout_s=args.timeout_seconds,
            api_key=api_key,
        )
        write_result(args.output, result)
    except ContentBatteryError as error:
        raise SystemExit(str(error)) from error
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
