# NVIDIA Spark Local AI Experiments

Reproducible local inference experiments on NVIDIA DGX Spark / GB10. This
repository records the configurations that worked, the ones that did not, and
the measurements behind each conclusion.

**Start with the [full 2026-08-16 capability campaign](docs/benchmark-results-2026-08-16.md).**
It reports successful, partial, incompatible, and repaired paths with exact
model revisions, image digests, telemetry, and retained local evidence IDs.

## Campaign coverage

SparkBench exercised six runtime paths on one GB10, one configuration at a
time:

- vLLM for dense, MoE, reasoning, vision-language, embedding, and reranking
  profiles;
- Ollama for quantized chat, native prefill timing, vision, OCR, and embedding
  or reranking endpoints;
- SGLang for Phi-4 FP8 text and vision, while retaining incompatible attempts;
- an offline TensorRT-LLM direct adapter for Phi-4 audio transcription; and
- an offline Transformers direct adapter for Nemotron block-diffusion text;
  and
- a pinned native llama.cpp build for Unsloth Qwen3.8 GGUF, including its
  embedded NextN/MTP head.

Suites cover decode, prefill, concurrency, long-context retrieval, JSON, tool
calling, small exact-answer quality checks, embeddings, reranking, vision, OCR,
ASR, and diffusion. These are targeted capability and performance probes, not
broad model-quality scores. Read the campaign report before comparing unlike
backends or treating a transport pass as semantic correctness.

## Original Qwen3.8 result

The first experiment serves Qwen3.8-27B through an OpenAI-compatible vLLM API.
Moving from BF16 to NVFP4 weights and enabling the model's built-in MTP head
increased sustained decode from **3.91 to 16.04 tokens/second** on one Spark.

| Configuration | Weight memory | Median decode | Relative speed |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B BF16 | 51.1 GiB | 3.91 tok/s | 1.0x |
| Qwen3.8-27B NVFP4 | 24.18 GiB | 8.41 tok/s | 2.15x |
| Qwen3.8-27B NVFP4 + MTP | 24.97 GiB | **16.04 tok/s** | **4.10x** |

A separate deterministic SparkBench profile tested serving throughput at a
32K context with eight sequence slots:

| Workload | Median E2E | Aggregate output throughput |
| --- | ---: | ---: |
| Single stream | 7.556 s | 17.27 tok/s |
| Concurrency 2 | 4.474 s | 27.90 tok/s |
| Concurrency 4 | 4.473 s | **53.93 tok/s** |

Concurrency 4 delivered 1.93 times the concurrency-2 throughput with essentially
unchanged median latency. These are case-wall aggregate rates; per-request decode
rates remain client estimates because MTP bundled tokens in streamed events. The
same run passed an exact-key retrieval probe with an 8,284-token prompt.

Read the [full Qwen3.8-27B write-up](docs/qwen38-27b.md) for the architecture,
benchmark methodology, tuning sweep, failed experiments, and why MTP matters
so much on Spark. Protocol details and measurements live in
[BENCHMARK.md](BENCHMARK.md).

## Original Qwen3.8 tested system

The broader campaign used multiple pinned runtime images; each run records its
exact environment in `plan.json` and server provenance. The original Compose
experiment used:

- NVIDIA GB10, compute capability 12.1
- 128 GB unified LPDDR5X memory
- Ubuntu 24.04 on ARM64
- CUDA 13 and NVIDIA driver 580.142
- vLLM `0.1.dev19754+g3a0914114`
- `vllm/vllm-openai:qwen38` ARM64 image

The tested image is publicly available on Docker Hub. Its ARM64 manifest at
the time of testing was
`sha256:541e0e475418de6178b45c0d9ef420fb6be79bf43130a4d552cb668e425f4d27`.
The Compose files use the moving `qwen38` tag so that new clones receive fixes;
pin the digest if exact historical reproduction is more important.

## Quick start: fastest configuration

Prerequisites are Docker Engine, Docker Compose, and the NVIDIA Container
Toolkit configured for Docker. Model weights are downloaded from Hugging Face
on first start. Set `HF_TOKEN` if the repository or your environment requires
authentication.

```bash
git clone https://github.com/xlzuvekas/nvidia-spark-local-ai.git
cd nvidia-spark-local-ai
docker compose -f compose.nvfp4.yaml -f compose.nvfp4-mtp.yaml up -d
docker compose -f compose.nvfp4.yaml -f compose.nvfp4-mtp.yaml logs -f
```

The API listens only on the local machine at `http://127.0.0.1:8000/v1`.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.8-27B",
    "messages": [{"role": "user", "content": "Explain unified memory briefly."}],
    "max_tokens": 128,
    "temperature": 0.7,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Model weights are retained under `data/huggingface/`. Compiled vLLM and
FlashInfer kernels are retained under `data/vllm/`, avoiding repeated
compilation and autotuning after container recreation.

## Other configurations

BF16 baseline:

```bash
docker compose -f compose.yaml up -d
```

NVFP4 weights without speculative decoding:

```bash
docker compose -f compose.nvfp4.yaml up -d
```

Override MTP depth or the scheduler budget for experiments:

```bash
MTP_TOKENS=4 MAX_BATCHED_TOKENS=8192 \
  docker compose -f compose.nvfp4.yaml -f compose.nvfp4-mtp.yaml up -d
```

Only one configuration should own port 8000 at a time. Compose will recreate
the shared `qwen38-27b` container when switching configurations.

## Benchmarking

The original dependency-free client reproduces the focused Qwen3.8 experiment
against a server already listening on port 8000:

```bash
python3 benchmark.py
python3 benchmark.py --temperature 0
```

The benchmark uses unique prompts for prefill probes so prefix caching cannot
inflate the result. Decode throughput is reported separately from TTFT and
end-to-end throughput.

For repeatable multi-model work, SparkBench adds manifest-driven inventory,
planning, lifecycle checks, telemetry, resumable execution, and JSON/CSV reports:

```bash
python3 sparkbench.py inventory --sizes
python3 sparkbench.py list --verbose
python3 sparkbench.py fetch phi-4-reasoning-plus-fp8
python3 sparkbench.py benchmark qwen38-27b-nvfp4-mtp3
python3 sparkbench.py benchmark qwen38-27b-nvfp4-mtp3-throughput \
  --suite manifests/suites/core.toml
python3 sparkbench.py benchmark gpt-oss-120b-mxfp4 \
  --suite manifests/suites/reasoning_quick.toml
python3 sparkbench.py matrix --backend ollama --task chat \
  --suite manifests/suites/quick.toml
python3 sparkbench.py diffusion-direct \
  nemotron-labs-diffusion-14b-transformers-direct
python3 sparkbench.py trtllm-direct \
  phi-4-multimodal-instruct-fp8-trtllm-audio --timeout 7200
python3 sparkbench.py benchmark qwen38-27b-ud-q4-k-xl-llamacpp \
  --suite manifests/suites/quick.toml
python3 sparkbench.py benchmark qwen38-27b-ud-q4-k-xl-llamacpp-mtp3 \
  --suite manifests/suites/core.toml
```

The default `smoke.toml` suite performs quick endpoint and capability checks;
`quick.toml` is a short performance screen; `core.toml` runs the repeated decode,
prefill, concurrency, long-context, and structured-output matrix.
`reasoning_quick.toml` and `reasoning_core.toml` give reasoning models enough
completion budget to reach visible answers. Focused suites include
`capabilities.toml`, `chat_quality.toml`, `embeddings.toml`,
`multimodal_embeddings.toml`, `multimodal_rerank.toml`, `vision.toml`,
`ocr.toml`, `audio_asr.toml`, and `diffusion_direct.toml`. The last two are
executed through their dedicated direct commands above.
`llamacpp_mtp_depth.toml` isolates the native GGUF draft-depth sweep.

Reasoning prefill throughput uses TTFT to the first visible content or reasoning
delta. Needle correctness checks final content only: a key found solely in hidden
reasoning is not an answer received by the caller.
Hugging Face and Docker image downloads are disabled unless `--allow-download`
is supplied; `fetch` is the explicit pinned-snapshot acquisition path.
SparkBench also refuses to benchmark while unrelated containers or GPU compute
processes are active; it does not stop them automatically.

Ollama chat runs use its native API so the frozen plan controls context size and
records server-reported prompt/decode durations. Cached large-model profiles use
32K context to fit safely in unified memory. OpenAI-compatible SSE results are
explicitly labeled as client estimates when chunks cannot be mapped one-to-one
to tokens.

SparkBench addresses Ollama over loopback but does not reconfigure the host
service. Verify its listener with `ss -ltnp 'sport = :11434'` and restrict
`OLLAMA_HOST` separately if the daemon is exposed beyond the machine.

Each run freezes a `plan.json` under `results/` and records raw events, telemetry,
server provenance/logs, and generated `summary.json`/`summary.csv`. A stopped run
can be resumed or summarized without changing its plan:

```bash
python3 sparkbench.py run results/<run-directory>
python3 sparkbench.py summarize results/<run-directory>
python3 sparkbench.py audit-matrix results/matrices/<matrix-directory>
```

`audit-matrix` reads only frozen plans, event journals, and existing summaries.
It emits JSON and exits nonzero for structural or aggregate-math discrepancies;
it never contacts Docker, the GPU, or the network.

Model configurations live in `manifests/models.toml`, suites in
`manifests/suites/`, and pipeline modules in `bench/`. See the
[benchmark strategy](docs/benchmark-strategy.md) and
[local model inventory](docs/local-model-inventory.md) for scope and current
cache coverage. The dated [campaign plan](docs/benchmark-campaign-2026-08-15.md),
[cached training guide](docs/cached-training-capability-2026-08-15.md), and
[Nemotron direct-run guide](docs/nemotron-diffusion-direct.md) preserve the
specialized protocols. The consolidated evidence is in the
[2026-08-16 campaign results](docs/benchmark-results-2026-08-16.md); the earlier
[2026-08-14 smoke results](docs/benchmark-results-2026-08-14.md) remain as a
historical snapshot.

Run the dependency-free unit suite before changing the pipeline:

```bash
python3 -m unittest discover -s tests -v
```

## Native GGUF highlights

The pinned Unsloth Qwen3.8 GGUF track now covers Q8, UD-Q4_K_XL, and
UD-IQ2_XXS on the same llama.cpp build and eight-slot 32K-per-slot layout.
The Q8 checkpoint was roughly one-third slower than Q4 without improving the
small exact-answer screen. IQ2 was much faster and about half the Q4 size, but
failed the JSON validator in both smoke and core, so it is a capacity floor—not
a quality-equivalent replacement.

For Q4's embedded NextN head, bracket controls drifted only 0.33%. A six-depth
sweep followed by 20-request confirmation measured **23.80 tok/s at maximum
draft depth 5** versus **23.27 tok/s at depth 4**; depth 5 is the tested leader,
but the 2.28% margin is modest. The F16 vision projector also passed all nine
solid-color transport/recognition probes across 64, 512, and 1024 pixels. These
targeted checks do not establish broad language or vision quality.

## Original Qwen3.8 conclusions

- NVFP4 weights are the largest straightforward win on GB10.
- MTP depth 3 is the best tested single-stream setting for this model.
- The eight-sequence profile reached 53.93 aggregate output tok/s at concurrency
  4, although larger samples are still needed for tail-latency claims.
- FP8 remains the supported KV-cache format on SM121 in the tested vLLM build.
- NVFP4 KV-cache code exists, but this build gates its FlashInfer path to SM100.
- The 8,284-token retrieval probe passed; it does not validate the full 32K
  served context or the checkpoint's 262K native limit.
- FP4 KV would mostly benefit long-context capacity, not short-context decode,
  where repeatedly reading model weights dominates.
- Spark's CPU and GPU share memory bandwidth. Unrelated memory-heavy CPU work
  can therefore compete directly with inference.

## Status and next work

The campaign now includes TensorRT-LLM and SGLang anchors, concurrent serving,
text and multimodal embeddings/reranking, vision/OCR, ASR, diffusion, power and
thermal sampling, and explicit quality checks. Coverage varies by model, and
failed or partial cases remain part of the result.

The managed llama.cpp path validates pinned binaries, GGUF/projector hashes,
offline loopback isolation, native speculative-decoding counters, and cleanup
by process identity. Matched Qwen3.8 runs now cover the quantization ladder,
vision, MTP depth, smoke, quick, and core suites; see the campaign report for
the measurements and validity limits. Broader calibrated quality, native
32K/128K/262K correctness, and agent workloads remain future work.

## License

MIT. Model weights and container images retain their own licenses.
