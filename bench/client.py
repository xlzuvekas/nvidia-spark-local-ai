"""Dependency-free OpenAI-compatible streaming benchmark client."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request


@dataclass
class RequestResult:
    request_id: str
    started_at_ns: int
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    elapsed_s: float
    decode_s: float
    decode_tps: float
    output_tps: float
    emission_events: int
    finish_reason: str | None
    response_model: str | None
    content: str
    reasoning: str
    tool_calls: list[dict[str, Any]]
    decode_metric_source: str = "client_estimate"
    load_s: float | None = None
    server_prompt_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkRequestError(RuntimeError):
    pass


@dataclass
class EmbeddingResult:
    request_id: str
    started_at_ns: int
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    elapsed_s: float
    decode_s: float
    decode_tps: float
    output_tps: float
    emission_events: int
    finish_reason: str | None
    response_model: str | None
    dimension: int
    batch_size: int
    items_per_s: float
    finite: bool
    norms: list[float]
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tool_calls"] = value["tool_calls"] or []
        return value


def embedding_request(
    *,
    base_url: str,
    model: str,
    inputs: list[str],
    request_id: str,
    timeout_s: float = 900,
) -> EmbeddingResult:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/embeddings",
        data=json.dumps({"model": model, "input": inputs}).encode(),
        headers={"Content-Type": "application/json"},
    )
    started_wall_ns = time.time_ns()
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        try:
            body = error.read().decode(errors="replace")[:2000]
        finally:
            error.close()
        raise BenchmarkRequestError(f"HTTP {error.code}: {body}") from error
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        raise BenchmarkRequestError(
            f"Embedding request {request_id} failed: {error}"
        ) from error
    elapsed_s = time.perf_counter() - started
    vectors = [item.get("embedding", []) for item in payload.get("data", [])]
    if len(vectors) != len(inputs) or not vectors:
        raise BenchmarkRequestError("Embedding response returned the wrong number of vectors")
    dimension = len(vectors[0])
    if dimension == 0 or any(len(vector) != dimension for vector in vectors):
        raise BenchmarkRequestError("Embedding vectors had missing or inconsistent dimensions")
    finite = all(math.isfinite(float(value)) for vector in vectors for value in vector)
    norms = [math.sqrt(sum(float(value) ** 2 for value in vector)) for vector in vectors]
    usage = payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    return EmbeddingResult(
        request_id=request_id,
        started_at_ns=started_wall_ns,
        prompt_tokens=prompt_tokens,
        completion_tokens=0,
        ttft_s=elapsed_s,
        elapsed_s=elapsed_s,
        decode_s=0.0,
        decode_tps=0.0,
        output_tps=0.0,
        emission_events=1,
        finish_reason="stop",
        response_model=payload.get("model", model),
        dimension=dimension,
        batch_size=len(inputs),
        items_per_s=len(inputs) / max(elapsed_s, 1e-9),
        finite=finite,
        norms=norms,
    )


def _has_output(delta: dict[str, Any]) -> bool:
    return bool(
        delta.get("content")
        or delta.get("reasoning")
        or delta.get("reasoning_content")
        or delta.get("tool_calls")
    )


def stream_chat_request(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    request_id: str,
    extra_body: dict[str, Any] | None = None,
    timeout_s: float = 900,
    start_barrier: threading.Barrier | None = None,
) -> RequestResult:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra_body:
        protected = {"model", "max_tokens", "temperature", "stream", "stream_options"}
        overlap = protected & set(extra_body)
        if overlap:
            raise BenchmarkRequestError(
                "Extra request body cannot override benchmark fields: "
                + ", ".join(sorted(overlap))
            )
        payload.update(extra_body)
    if start_barrier:
        start_barrier.wait(timeout=30)
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started_wall_ns = time.time_ns()
    started = time.perf_counter()
    first_output_at: float | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    response_model: str | None = None
    emission_events = 0
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].lstrip()
                if data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as error:
                    raise BenchmarkRequestError(f"Invalid SSE JSON: {line[:200]}") from error
                response_model = event.get("model", response_model)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    if _has_output(delta):
                        emission_events += 1
                        if first_output_at is None:
                            first_output_at = time.perf_counter()
                    if isinstance(delta.get("content"), str):
                        content_parts.append(delta["content"])
                    reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content")
                    if isinstance(reasoning_delta, str):
                        reasoning_parts.append(reasoning_delta)
                    for tool_delta in delta.get("tool_calls") or []:
                        index = int(tool_delta.get("index", 0))
                        current = tool_calls.setdefault(
                            index,
                            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                        )
                        if tool_delta.get("id"):
                            current["id"] += str(tool_delta["id"])
                        function = tool_delta.get("function") or {}
                        if function.get("name"):
                            current["function"]["name"] += str(function["name"])
                        if function.get("arguments"):
                            current["function"]["arguments"] += str(function["arguments"])
    except urllib.error.HTTPError as error:
        try:
            body = error.read().decode(errors="replace")[:2000]
        finally:
            error.close()
        raise BenchmarkRequestError(f"HTTP {error.code}: {body}") from error
    except (urllib.error.URLError, OSError) as error:
        detail = getattr(error, "reason", error)
        raise BenchmarkRequestError(
            f"Chat request {request_id} failed: {detail}"
        ) from error
    finished = time.perf_counter()
    if usage is None:
        raise BenchmarkRequestError("Streaming response did not include token usage")
    if first_output_at is None:
        raise BenchmarkRequestError("Streaming response did not emit content or reasoning")
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    decode_s = max(finished - first_output_at, 1e-9)
    return RequestResult(
        request_id=request_id,
        started_at_ns=started_wall_ns,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_s=first_output_at - started,
        elapsed_s=finished - started,
        decode_s=decode_s,
        decode_tps=max(completion_tokens - 1, 0) / decode_s,
        output_tps=completion_tokens / max(finished - started, 1e-9),
        emission_events=emission_events,
        finish_reason=finish_reason,
        response_model=response_model,
        content="".join(content_parts),
        reasoning="".join(reasoning_parts),
        tool_calls=[tool_calls[index] for index in sorted(tool_calls)],
    )


def _ollama_messages(prompt: str, extra_body: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate the small OpenAI message subset used by suites to Ollama."""

    supplied = extra_body.pop("messages", None)
    if supplied is None:
        return [{"role": "user", "content": prompt}]
    messages: list[dict[str, Any]] = []
    for message in supplied:
        content = message.get("content", "")
        if isinstance(content, str):
            messages.append({"role": message.get("role", "user"), "content": content})
            continue
        text_parts: list[str] = []
        images: list[str] = []
        for part in content or []:
            if part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
            elif part.get("type") == "image_url":
                image_url = (part.get("image_url") or {}).get("url", "")
                if isinstance(image_url, str) and image_url.startswith("data:"):
                    image_url = image_url.partition(",")[2]
                if image_url:
                    images.append(str(image_url))
        translated: dict[str, Any] = {
            "role": message.get("role", "user"),
            "content": "\n".join(text_parts),
        }
        if images:
            translated["images"] = images
        messages.append(translated)
    return messages


def stream_ollama_chat_request(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    request_id: str,
    context_size: int,
    extra_body: dict[str, Any] | None = None,
    timeout_s: float = 900,
    start_barrier: threading.Barrier | None = None,
) -> RequestResult:
    """Stream Ollama's native API with an explicit, reproducible context size."""

    additions = dict(extra_body or {})
    protected = {"model", "stream", "options", "keep_alive"}
    overlap = protected & set(additions)
    if overlap:
        raise BenchmarkRequestError(
            "Extra request body cannot override Ollama benchmark fields: "
            + ", ".join(sorted(overlap))
        )
    payload: dict[str, Any] = {
        "model": model,
        "messages": _ollama_messages(prompt, additions),
        "stream": True,
        "keep_alive": "5m",
        "options": {
            "num_ctx": context_size,
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    response_format = additions.pop("response_format", None)
    if response_format:
        payload["format"] = "json"
    reasoning_effort = additions.pop("reasoning_effort", None)
    if reasoning_effort is not None:
        payload["think"] = False if reasoning_effort == "none" else reasoning_effort
    if "tool_choice" in additions:
        additions.pop("tool_choice")
    payload.update(additions)

    if start_barrier:
        start_barrier.wait(timeout=30)
    root = base_url.removesuffix("/v1").rstrip("/")
    request = urllib.request.Request(
        f"{root}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started_wall_ns = time.time_ns()
    started = time.perf_counter()
    first_output_at: float | None = None
    finished_payload: dict[str, Any] | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    emission_events = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise BenchmarkRequestError(
                        f"Invalid Ollama JSON for {request_id}: {raw_line[:200]!r}"
                    ) from error
                if event.get("error"):
                    raise BenchmarkRequestError(
                        f"Ollama chat request {request_id} failed: {event['error']}"
                    )
                message = event.get("message") or {}
                content = message.get("content")
                reasoning = message.get("thinking") or message.get("reasoning")
                native_tools = message.get("tool_calls") or []
                if content or reasoning or native_tools:
                    emission_events += 1
                    if first_output_at is None:
                        first_output_at = time.perf_counter()
                if isinstance(content, str):
                    content_parts.append(content)
                if isinstance(reasoning, str):
                    reasoning_parts.append(reasoning)
                for index, call in enumerate(native_tools):
                    function = call.get("function") or {}
                    arguments = function.get("arguments", {})
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, separators=(",", ":"), sort_keys=True)
                    tool_calls.append(
                        {
                            "id": str(call.get("id") or f"ollama-{index}"),
                            "type": "function",
                            "function": {
                                "name": str(function.get("name", "")),
                                "arguments": arguments,
                            },
                        }
                    )
                if event.get("done"):
                    finished_payload = event
    except urllib.error.HTTPError as error:
        try:
            body = error.read().decode(errors="replace")[:2000]
        finally:
            error.close()
        raise BenchmarkRequestError(
            f"Ollama chat request {request_id} returned HTTP {error.code}: {body}"
        ) from error
    except (urllib.error.URLError, OSError) as error:
        detail = getattr(error, "reason", error)
        raise BenchmarkRequestError(
            f"Ollama chat request {request_id} failed: {detail}"
        ) from error

    finished = time.perf_counter()
    if finished_payload is None:
        raise BenchmarkRequestError(
            f"Ollama chat request {request_id} ended without a final record"
        )
    if first_output_at is None:
        first_output_at = finished
    required_counters = (
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    )
    missing_counters = [key for key in required_counters if key not in finished_payload]
    if missing_counters:
        raise BenchmarkRequestError(
            f"Ollama chat request {request_id} omitted native timing counters: "
            + ", ".join(missing_counters)
        )
    try:
        prompt_tokens = int(finished_payload["prompt_eval_count"])
        prompt_duration_ns = float(finished_payload["prompt_eval_duration"])
        completion_tokens = int(finished_payload["eval_count"])
        eval_duration_ns = float(finished_payload["eval_duration"])
    except (TypeError, ValueError) as error:
        raise BenchmarkRequestError(
            f"Ollama chat request {request_id} returned invalid native timing counters"
        ) from error
    if (
        prompt_tokens < 0
        or prompt_duration_ns < 0
        or completion_tokens <= 0
        or eval_duration_ns <= 0
    ):
        raise BenchmarkRequestError(
            f"Ollama chat request {request_id} returned non-positive generation counters"
        )
    server_decode_s = eval_duration_ns / 1e9
    client_decode_s = max(finished - first_output_at, 1e-9)
    decode_s = server_decode_s if server_decode_s > 0 else client_decode_s
    decode_source = "server_reported_eval_duration" if server_decode_s > 0 else "client_estimate"
    decode_tokens = (
        completion_tokens
        if server_decode_s > 0
        else max(completion_tokens - 1, 0)
    )
    return RequestResult(
        request_id=request_id,
        started_at_ns=started_wall_ns,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_s=first_output_at - started,
        elapsed_s=finished - started,
        decode_s=decode_s,
        decode_tps=decode_tokens / max(decode_s, 1e-9),
        output_tps=completion_tokens / max(finished - started, 1e-9),
        emission_events=emission_events,
        finish_reason=finished_payload.get("done_reason"),
        response_model=finished_payload.get("model", model),
        content="".join(content_parts),
        reasoning="".join(reasoning_parts),
        tool_calls=tool_calls,
        decode_metric_source=decode_source,
        load_s=float(finished_payload.get("load_duration", 0)) / 1e9,
        server_prompt_s=prompt_duration_ns / 1e9,
    )


def concurrent_chat_requests(
    *,
    requests: list[dict[str, Any]],
    concurrency: int,
    request_function: Callable[..., RequestResult] = stream_chat_request,
) -> tuple[list[RequestResult], float]:
    barrier = threading.Barrier(len(requests)) if len(requests) > 1 else None
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(request_function, **request, start_barrier=barrier)
            for request in requests
        ]
        results = [future.result() for future in futures]
    return results, time.perf_counter() - started
