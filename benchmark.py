#!/usr/bin/env python3
"""Small, dependency-free benchmark for the local Qwen3.8 vLLM endpoint."""

from __future__ import annotations

import json
import argparse
import statistics
import time
import urllib.request


URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.8-27B"


def stream_request(prompt: str, max_tokens: int, temperature: float) -> dict[str, float]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at = None
    usage = None
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if first_token_at is None and (
                    delta.get("content") or delta.get("reasoning")
                ):
                    first_token_at = time.perf_counter()
    finished = time.perf_counter()
    if usage is None or first_token_at is None:
        raise RuntimeError("Streaming response did not include timing or usage data")
    completion_tokens = usage["completion_tokens"]
    decode_seconds = max(finished - first_token_at, 1e-9)
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": completion_tokens,
        "ttft_s": first_token_at - started,
        "elapsed_s": finished - started,
        "decode_tps": max(completion_tokens - 1, 0) / decode_seconds,
        "overall_tps": completion_tokens / (finished - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    print("Decode benchmark: 3 runs, up to 256 output tokens")
    decode_runs = []
    for trial in range(1, 4):
        result = stream_request(
            "Write an unbroken numbered list of distinct two-word phrases. "
            "Continue until the output limit; do not conclude or summarize.",
            256,
            args.temperature,
        )
        decode_runs.append(result)
        print(
            f"  run {trial}: prompt={result['prompt_tokens']:.0f}, "
            f"output={result['completion_tokens']:.0f}, "
            f"TTFT={result['ttft_s']:.3f}s, elapsed={result['elapsed_s']:.2f}s, "
            f"decode={result['decode_tps']:.2f} tok/s",
            flush=True,
        )

    print("Prefill probes: one output token, unique uncached prompts")
    prefill_runs = []
    for repeat in (128, 1024, 4096):
        nonce = time.time_ns()
        prompt = (
            f"Benchmark nonce {nonce}. Read the following text and reply with one word. "
            + ("measurement " * repeat)
        )
        result = stream_request(prompt, 1, args.temperature)
        prefill_runs.append(result)
        approximate_rate = result["prompt_tokens"] / result["ttft_s"]
        print(
            f"  prompt={result['prompt_tokens']:.0f} tokens: "
            f"TTFT={result['ttft_s']:.3f}s, approx prefill={approximate_rate:.0f} tok/s",
            flush=True,
        )

    print("Summary")
    print(
        f"  median TTFT: {statistics.median(r['ttft_s'] for r in decode_runs):.3f}s"
    )
    print(
        f"  median decode: "
        f"{statistics.median(r['decode_tps'] for r in decode_runs):.2f} tok/s"
    )
    print(
        f"  median end-to-end output rate: "
        f"{statistics.median(r['overall_tps'] for r in decode_runs):.2f} tok/s"
    )


if __name__ == "__main__":
    main()
