# Cached Model Smoke Results — 2026-08-14

These are one-request capability screens, not steady-state rankings. All Ollama
chat models used their cached GGUF weights, the native `/api/chat` endpoint, a
32K served context (8K for DeepSeek OCR), temperature zero, and 32 requested
output tokens. Decode rate comes from Ollama's reported evaluation duration;
cold load is recorded separately. SparkBench unloaded every model between runs.

| Cached model | Quantization | Cold load | Decode | TTFT | Smoke outcome |
| --- | --- | ---: | ---: | ---: | --- |
| Mistral Medium 3.5 128B | Q4_K_M | 13.39 s | 2.72 tok/s | 0.89 s | Chat, JSON, tools, vision passed |
| Nemotron 3 Super 123.6B | Q4_K_M | 15.02 s | 20.36 tok/s | 0.56 s | Chat, JSON, tools passed |
| Llama 3.3 70B | Q4_K_M | 29.86 s | 4.57 tok/s | 0.41 s | Chat, JSON, tools passed |
| Nemotron Cascade 2 31.6B | Q4_K_M | 17.16 s | 72.58 tok/s | 0.20 s | Chat, JSON, tools passed |
| Qwen 3.5 35B-A3B | Q4_K_M | 6.84 s | 55.70 tok/s | 0.26 s | Chat, JSON, tools, vision passed |
| Gemma 4 31B | Q4_K_M | 5.87 s | 10.49 tok/s | 0.33 s | Chat, tools, vision passed; JSON was fenced |
| GLM 4.7 Flash | Q4_K_M | 16.52 s | 59.21 tok/s | 0.22 s | Chat and tools passed; JSON was fenced |
| Qwen 3 30B-A3B | Q4_K_M | 4.29 s | 76.01 tok/s | 0.20 s | Chat, JSON, tools passed |
| Gemma 3 12B | Q4_K_M | 3.67 s | 25.30 tok/s | 0.23 s | Chat, JSON, vision passed |
| DeepSeek OCR 3.3B | F16 | 4.79 s | — | — | Vision/JSON probes passed; generic decode stopped empty |

The first Mistral attempt used Ollama's automatic 262K context. Its log showed
75 GiB of weights, 88 GiB of KV cache, and roughly 165 GiB total allocation;
the operating system killed it for OOM. The explicit 32K profile above fits and
is the correct single-Spark deployment boundary for this quantization/runtime.

Embedding screens covered both engines:

- `all-MiniLM-L6-v2` through vLLM: 384 dimensions, about 186 items/s at batch 1
  and 968 items/s at batch 32 for the 32-repeat input. Its 256-token serving
  profile correctly marked the 512-repeat case context-limited.
- `nomic-embed-text` through Ollama: 768 dimensions, about 57 items/s at batch 1
  and 202 items/s at batch 32; the longer-input batch-1 case reached 36.6 items/s.

The cached `openai/gpt-oss-120b` MXFP4 checkpoint also fit through NVIDIA's
vLLM 26.07 image at a controlled 32K context. Startup took 499.6 seconds,
including 399.9 seconds to load 66.14 GiB of model memory; the KV cache used
33.42 GiB and minimum startup-phase host-available memory was 9.05 GiB. Chat,
JSON, and tool probes passed. Its 32-token screen delivered an aggregate 29.30
output tok/s. The 38.95 tok/s client decode estimate is retained but excluded
from exact ranking because the stream bundled multiple tokens per emission.

Qwen3.8-27B NVFP4+MTP also completed the repeated `quick.toml` suite at a 32K
served context. Single-stream aggregate output was 17.27 tok/s; concurrency 2
reached 27.90 tok/s and concurrency 4 reached 53.93 tok/s. Client-TTFT prefill
approximations were 1,111 tok/s at 324 prompt tokens, 2,134 tok/s at 2,117
tokens, and 1,916 tok/s at 8,261 tokens. The 8,284-token retrieval probe passed.
See [BENCHMARK.md](../BENCHMARK.md) for exact protocol, latency, capacity, and
telemetry details.

Use `smoke.toml` for compatibility, `quick.toml` for a short performance screen,
and `core.toml` for repeated decode, prefill, concurrency, and 16K correctness.
The repository's focused Qwen3.8 study remains the controlled precision/MTP
comparison; do not mix its longer-run numbers with this smoke table.
