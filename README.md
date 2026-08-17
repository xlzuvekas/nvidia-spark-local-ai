# NVIDIA Spark Local AI Experiments

Reproducible local inference experiments on NVIDIA DGX Spark / GB10. This
repository records the configurations that worked, the ones that did not, and
the measurements behind each conclusion.

**Start with the [latest overnight report](docs/benchmark-results-2026-08-17.md).**
It records the completed Unsloth Qwen3.6 and Qwen3.8 GGUF, SGLang/DSpark, and
Muse-Glimmer/DFlash experiments with their semantic gates and exact evidence
IDs. The preceding
[full 2026-08-16 capability campaign](docs/benchmark-results-2026-08-16.md)
reports successful, partial, incompatible, and repaired paths with exact model
revisions, image digests, telemetry, and retained local evidence IDs.

## Overnight findings

- **Qwen3.6 Q4 + MTP2:** embedded MTP2 was active and raised matched
  fixed-budget aggregate decode by 24–29% in the short screens and roughly
  50–56% in the core screen. Both profiles passed chat, JSON, tools, and
  exact-key retrieval through 245K-token prompts; MTP2 was 4.2–7.0% slower to
  first token in the prompt-dominated long-context sweep.
- **Qwen3.8 NVFP4 + DSpark:** the pinned community SGLang recipe served at
  26.369 tok/s for D256 and scaled to 127.185 aggregate tok/s at C8. JSON and
  tools passed, but core output was reasoning-only: retrieval was 0/3 and the
  small exact-answer screen was 0/4. Fresh prompt throughput ranged from
  14.499 to 42.739 tok/s, so the headline 34–38 tok/s is workload-dependent.
- **Qwen3.8 Q5, perplexity, and 262K context:** Q5 reached 9.229 tok/s at D256
  and 45.301 aggregate tok/s at C8 while passing the bounded capability cells.
  Matched WikiText-2 did not resolve Q4/Q5/Q8 (their reported intervals
  overlap); IQ2 was 12.77% worse than Q4. Separate Q4 baseline/MTP5 profiles
  both passed all 10 retrieval probes through 245K-token prompts, although
  MTP5 TTFT was 3.2–5.1% slower.
- **Muse-Glimmer Q4 + DFlash15:** llama.cpp admitted the pinned sidecar; the
  quick lifetime recorded 155 drafts, 1,992 proposals, and 1,271 acceptances.
  DFlash raised matched D128 aggregate emission from 11.707 to 59.259 tok/s,
  but all 30 fixed-token quick responses still had empty visible content and
  reasoning-side prompt echo; both 8K needles failed. These are accelerated
  invalid emissions, not usable-answer throughput, so core was deliberately
  stopped at the semantic gate.

## Campaign coverage

SparkBench exercised six runtime paths on one GB10, one configuration at a
time:

- vLLM for dense, MoE, reasoning, vision-language, embedding, and reranking
  profiles;
- Ollama for quantized chat, native prefill timing, vision, OCR, and embedding
  or reranking endpoints;
- SGLang for Phi-4 FP8 text and vision, plus a managed Qwen3.8 NVFP4/DSpark
  target-and-draft profile with measured throughput and retained semantic
  failures;
- an offline TensorRT-LLM direct adapter for Phi-4 audio transcription;
- an offline Transformers direct adapter for Nemotron block-diffusion text;
  and
- a pinned native llama.cpp build for Unsloth Qwen3.6 and Qwen3.8 GGUF,
  including their embedded MTP/NextN heads.

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
python3 sparkbench.py benchmark qwen36-35b-a3b-ud-q4-k-xl-llamacpp-mtp2 \
  --suite manifests/suites/llamacpp_long_context.toml
python3 sparkbench.py benchmark qwen38-27b-nvfp4-dspark-sglang \
  --suite manifests/suites/core.toml
python3 sparkbench.py benchmark qwen38-27b-ud-q5-k-xl-llamacpp \
  --suite manifests/suites/core.toml
python3 sparkbench.py benchmark muse-glimmer-30b-ud-q4-k-xl-llamacpp-dflash15 \
  --suite manifests/suites/smoke.toml
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
`llamacpp_mtp_depth.toml` isolates the native GGUF draft-depth sweep, while
`llamacpp_long_context.toml` probes exact-key retrieval from 32K through 245K.

The direct perplexity command records pinned model, runtime, and dataset
hashes. Compare only the same base model with identical dataset and runtime
settings:

```bash
python3 sparkbench.py perplexity qwen38-27b-ud-q5-k-xl-llamacpp \
  --dataset /absolute/path/to/wiki.test.raw --chunks 64 --ctx-size 512 \
  --timeout 3600
```

`content_battery.py` measures an already-running OpenAI-compatible server and
is lifecycle-neutral: it neither launches nor stops the server. Pass its
Bearer credential by file; the saved JSON contains scalar timing and usage
evidence, not prompts, completions, request tags, or the key.

```bash
RUN_DIR=/absolute/path/to/kept-sglang-run
python3 content_battery.py \
  --base-url http://127.0.0.1:30000/v1 --model qwen3.8-27b \
  --api-key-file "$RUN_DIR/server/api-key" \
  --output results/content-battery-dspark.json --timeout-seconds 900
```

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
[2026-08-17 overnight results](docs/benchmark-results-2026-08-17.md) and
[2026-08-16 campaign results](docs/benchmark-results-2026-08-16.md); the earlier
[2026-08-14 smoke results](docs/benchmark-results-2026-08-14.md) remain as a
historical snapshot.

Run the dependency-free unit suite before changing the pipeline:

```bash
python3 -m unittest discover -s tests -v
```

## Native GGUF highlights

The managed Unsloth track now covers Qwen3.6 35B-A3B UD-Q4_K_XL as well as
Qwen3.8. On Qwen3.6, embedded MTP2 improved matched D256 aggregate decode by
29.0% in the five-request screen and 53.7% in core. It passed chat, JSON,
tools, and exact-key retrieval through 245K tokens, but did not improve
prefill: long-context TTFT was 4.2–7.0% slower.
See the [overnight report](docs/benchmark-results-2026-08-17.md) for sample-size,
queueing, telemetry, and comparison limits.

The pinned Unsloth Qwen3.8 GGUF track now covers Q8, UD-Q4_K_XL, and
UD-IQ2_XXS on the same llama.cpp build and eight-slot 32K-per-slot layout.
The Q8 checkpoint was roughly one-third slower than Q4 without improving the
small exact-answer screen. IQ2 was much faster and about half the Q4 size, but
failed the JSON validator in both smoke and core, so it is a capacity floor—not
a quality-equivalent replacement. Matched WikiText-2 perplexity likewise left
Q4, Q5, and Q8 unresolved while separating IQ2 as 12.77% worse than Q4.

For Q4's embedded NextN head, bracket controls drifted only 0.33%. A six-depth
sweep followed by 20-request confirmation measured **23.80 tok/s at maximum
draft depth 5** versus **23.27 tok/s at depth 4**; depth 5 is the tested leader,
but the 2.28% margin is modest. The F16 vision projector also passed all nine
solid-color transport/recognition probes across 64, 512, and 1024 pixels. These
targeted checks do not establish broad language or vision quality.

The Q5 middle quantization passed smoke and bounded core capability checks,
with true eight-slot aggregate throughput reaching 45.301 tok/s at C8. A
separate one-slot Q4 baseline/MTP5 comparison passed 20/20 combined retrieval
requests through 245K-token prompts; MTP5 was active but slower on every
prompt-dominated tier. Muse-Glimmer's DFlash15 sidecar was also active through
the fifteenth draft position, but both Muse smoke profiles emitted
reasoning-only prompt echo rather than visible answers and failed
structured-output contracts.

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
by process identity. Matched Qwen3.6 and Qwen3.8 evidence now covers MTP,
long-context, quantization, vision, smoke, quick, and core probes; see the dated
reports for the exact coverage and validity limits. Broader calibrated quality,
Muse prompt/template repair, and agent workloads remain future work.

The `qwen38-27b-nvfp4-dspark-sglang` profile encodes the pinned target and
draft snapshots, container digest, recipe flags, isolated compile cache,
loopback API, ephemeral per-run authentication, and redacted provenance needed
to evaluate the community
[DGX Spark Qwen3.8 recipe](https://github.com/hasso5703/dgx-spark-qwen38).
The managed run completed after a compile-cache permission repair. It measured
26.369 tok/s at D256 and 127.185 aggregate tok/s at C8, while fresh content
prompts ranged from 14.499 to 42.739 tok/s. Those rates are execution evidence,
not a correctness result: output remained reasoning-only, and the frozen core
budgets failed every retrieval and exact-answer check.

## License

MIT. Model weights and container images retain their own licenses.
