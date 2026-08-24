# NVIDIA Spark Local AI Experiments

Reproducible local-inference experiments on one NVIDIA DGX Spark / GB10. The
repository preserves working configurations, failed admissions, bounded
quality checks, and the provenance needed to interpret each number.

## Start here

| Question | Best entry point |
| --- | --- |
| Which MoE models are useful on Spark now? | [MoE landscape, including Laguna XS/S, G9v3, NInfer, and the dense Muse control](docs/moe-landscape-2026-08-17.md) |
| Which models complete deterministic multi-turn tool tasks? | [Agentic tool-use results: strict success, trace correctness, MTP, and Laguna](docs/agentic-tools-results-2026-08-17.md) |
| How does Qwen3.6 perform on a strict two-hop long-context needle? | [Two-hop retrieval: Qwen3.6 baseline versus MTP2 through the 245,760-target tier](docs/multihop-long-context-results-2026-08-18.md) |
| What does native llama.cpp prompt-KV reuse change for Qwen3.6? | [Prefix-cache controls: 8K and 32K shared-prefix cold/warm observations](docs/qwen36-prefix-cache-results-2026-08-18.md) |
| How did Qwen3-Coder-Next fare on terminal coding tasks? | [Harbor/Terminal-Bench-derived results: Qwen Code versus OpenCode](docs/harbor-terminal-results-2026-08-18.md) |
| How are offline coding-agent harnesses compared? | [Qwen3-Coder-Next Harbor campaign protocol](BENCHMARK.md#harbor-terminal-coding-agent-campaign) |
| What ran in the 2026-08-17 overnight campaign? | [Unsloth Qwen3.6/Qwen3.8, DSpark, perplexity, long context, and Muse DFlash](docs/benchmark-results-2026-08-17.md) |
| What is the broad cross-runtime baseline? | [2026-08-16 campaign: vLLM, Ollama, SGLang, llama.cpp, TensorRT-LLM, and Transformers](docs/benchmark-results-2026-08-16.md) |
| How was the original Qwen3.8 result produced? | [Focused Qwen3.8 study](docs/qwen38-27b.md) and [exact benchmark record](BENCHMARK.md) |
| How should results be compared or published? | [Benchmark protocol and evidence policy](BENCHMARK.md#sparkbench-protocol-and-evidence-publication) |
| Where are the machine-readable measurements? | [Sanitized evidence guide](evidence/README.md) and [complete evidence index](evidence/index.json) |
| Which profiles and cached artifacts exist? | [Local model inventory](docs/local-model-inventory.md) and [candidate survey](docs/model-candidates-2026-08-15.md) |
| How did local models handle bounded memory decisions? | [Graphiti-style resolver and synthetic transaction component results](docs/memory-operations-results-2026-08-24.md) |
| How should Laguna, Graphiti, and MemFS-style reflection be tested locally? | [Article audit and bounded memory benchmark plan](docs/laguna-graphiti-memory-plan-2026-08-24.md) |

The generated public evidence archive is intentionally separate from raw run
data. Its [human-readable map](evidence/README.md) and
[machine-readable index](evidence/index.json) account for complete, partial,
aborted, and nonterminal attempts without publishing raw payloads. The
[evidence publication section](BENCHMARK.md#publishing-sanitized-evidence)
explains how to create and verify both files.

## What the results say

- In the first memory-operation component panel, all four no-thinking profiles
  tied at 33/33 exact operations: 9/9 Graphiti-style resolver decisions and
  24/24 explicitly synthetic transaction decisions. Laguna XS had the smallest
  descriptive summed request wall time at 42.05625 seconds. The matched Ornith
  reasoning-on run was exploratory and fell to 27/33, failing all six bounded
  refusal variants while the narrow canary recorded six protected-value
  emissions. This single-run, mixed-quantization component panel is not an
  end-to-end Graphiti, MemFS, safety, or statistically significant result; see
  the [memory-operation report](docs/memory-operations-results-2026-08-24.md).
- In the two-replicate Harbor/Terminal-Bench-derived coding-agent panel,
  Qwen3-Coder-Next produced **1/24 strict passes overall**: Qwen Code passed
  1/12 and OpenCode passed 0/12. The one `fix-git` pass did not repeat. All 24
  trials finalized and the network, native-image, image-pair, and cleanup gates
  passed, separating task failure from harness infrastructure. This fixed
  six-task subset is not an official Terminal-Bench score; see the
  [Harbor result report](docs/harbor-terminal-results-2026-08-18.md) for the
  shared-verifier, quantization, telemetry, and replication limits.
- In the clean-revision agentic campaign, Laguna XS 2.1 was the fastest
  configuration to pass all 12 deterministic multi-turn tool episodes; Laguna
  S 2.1 and both Qwen3.8 configurations also passed 12/12. All eight
  configurations produced correct tool traces, but Qwen3.6 and Nemotron missed
  the strict final-answer envelope. Qwen3.8 MTP4 preserved 12/12 success while
  reducing matched summed episode wall time by 44.7%. The preceding dirty-tree
  pilot had the same 62/96 strict and 96/96 trace outcomes and remains
  separately labeled exploratory. See the
  [agentic report](docs/agentic-tools-results-2026-08-17.md) for the
  trace/answer distinction, clean provenance, and comparison limits.
- Sparse models are the most promising way around Spark's shared-memory
  bandwidth ceiling. In the matched native kernel panel, the measured 30--35B
  MoE artifacts decoded far faster than the dense Qwen3.8 and Muse controls.
  Serving geometry, quantization, and semantic gates still matter, so the
  [MoE report](docs/moe-landscape-2026-08-17.md) keeps unlike results separate.
- Laguna XS 2.1 is a validated one-slot MoE candidate. Laguna S 2.1 proves that
  a quantized 118B-A8B model can fit and execute
  on one Spark. Its earlier core quality result is partial, while the newer
  deterministic agentic tool battery passed 12/12; neither result is a broad
  quality score. Its one-slot concurrency remains queued.
- Speculative heads can materially improve sustained decode: Qwen3.6 MTP,
  Qwen3.8 MTP, Nemotron MTP, and Muse DFlash all exercised their draft paths.
  They do not improve every prefill workload, and accelerated invalid output
  is not counted as usable throughput.
- In native llama.cpp's serial Qwen3.6 prompt-KV controls, a warm shared prefix
  reported 98.4--99.6% cached prompt tokens and an observed 13.7--71.9-second
  within-run median first-token-time difference versus the preceding forced
  cold request. Decode rate remained effectively unchanged; the result is not
  a fresh-prefill or cross-run causal comparison.
- NInfer reached Qwen3.6 and Qwen3.8 GPU execution only through the repository's
  bounded [experimental SM121a patch](patches/ninfer/README.md). These eager
  engine measurements are not evidence of upstream support or production
  readiness; no NInfer branch was pushed upstream.
- G9v3 39B-A5B remains admission-only because its custom architecture lacks a
  validated path in the pinned optimized runtimes. Muse-Glimmer is retained as
  a dense bandwidth control, but its tested DFlash serving path failed the
  semantic-output gate.

These are targeted capability and performance probes, not broad model-quality
scores. Read the linked report before comparing different runtimes, context
geometries, slot counts, or validation states.

## Results and research index

### Measured results

- [Memory-operation component results — 2026-08-24](docs/memory-operations-results-2026-08-24.md):
  exact Graphiti-style resolver and synthetic transaction decisions across
  Laguna XS/S, Ornith, and Qwen3.6, with the Ornith thinking run kept
  exploratory and scalar-only limits stated explicitly.
- [Qwen3.6 native llama.cpp prefix-cache controls — 2026-08-18](docs/qwen36-prefix-cache-results-2026-08-18.md):
  serial 8K and 32K shared-prefix cold/warm controls with request-scoped cache,
  first-token, wall-time, prompt-work, and decode-rate measurements kept
  distinct.
- [Qwen3.6 two-hop long-context retrieval — 2026-08-18](docs/multihop-long-context-results-2026-08-18.md):
  matched baseline and MTP2 measurements through the 245,760-target tier,
  with exact final-key validation, timing, and lifetime draft-token counters.
- [Qwen3-Coder-Next Harbor terminal results — 2026-08-18](docs/harbor-terminal-results-2026-08-18.md):
  two complete replicates of a fixed six-task Terminal-Bench 2.1 subset through
  Qwen Code and OpenCode, with strict rewards, failure labels, infrastructure
  gates, and scalar-only evidence kept distinct.
- [Agentic tool-use results — 2026-08-17](docs/agentic-tools-results-2026-08-17.md):
  four deterministic multi-turn scenarios across Laguna, Qwen3.6, Qwen3.8,
  Nemotron, and matched MTP controls, with strict and trace outcomes separated.
- [MoE landscape — 2026-08-17](docs/moe-landscape-2026-08-17.md): current
  sparse-model options, same-binary kernel panel, Laguna 33B-A3B and 118B-A8B,
  G9v3 admission, NInfer's experimental port, and Muse as a dense control.
- [Overnight results — 2026-08-17](docs/benchmark-results-2026-08-17.md):
  Unsloth Qwen3.6 MTP2, Qwen3.8 Q5 and DSpark, matched perplexity, 262K-context
  probes, and Muse-Glimmer DFlash admission.
- [Full campaign results — 2026-08-16](docs/benchmark-results-2026-08-16.md):
  cross-runtime performance, capability, media, embedding, reranking, ASR,
  diffusion, power, and thermal coverage.
- [Initial campaign results — 2026-08-14](docs/benchmark-results-2026-08-14.md):
  the historical smoke snapshot and the original Qwen3.8 anchor.
- [Original Qwen3.8 analysis](docs/qwen38-27b.md): the BF16-to-NVFP4-to-MTP
  progression and the architectural explanation.

### Method, scope, and specialized guides

- [Benchmark protocol and evidence policy](BENCHMARK.md)
- [Offline-derived Harbor terminal coding-agent protocol](BENCHMARK.md#harbor-terminal-coding-agent-campaign)
- [Benchmark strategy](docs/benchmark-strategy.md)
- [Dated campaign plan](docs/benchmark-campaign-2026-08-15.md)
- [Local model inventory](docs/local-model-inventory.md)
- [Model candidate survey](docs/model-candidates-2026-08-15.md)
- [Cached media capabilities](docs/cached-media-capabilities-2026-08-15.md)
- [Cached training capability](docs/cached-training-capability-2026-08-15.md)
- [Laguna, Graphiti, and memory-reflection benchmark plan](docs/laguna-graphiti-memory-plan-2026-08-24.md)
- [Nemotron diffusion direct-run guide](docs/nemotron-diffusion-direct.md)
- [Experimental NInfer SM121a patch and reproduction notes](patches/ninfer/README.md)

## Reproduce the original Qwen3.8 path

Prerequisites are Docker Engine, Docker Compose, and the NVIDIA Container
Toolkit configured for Docker. Model weights are downloaded from Hugging Face
on first start; provide `HF_TOKEN` through the environment if access requires
it.

```bash
git clone https://github.com/xlzuvekas/nvidia-spark-local-ai.git
cd nvidia-spark-local-ai
docker compose -f compose.nvfp4.yaml -f compose.nvfp4-mtp.yaml up -d
docker compose -f compose.nvfp4.yaml -f compose.nvfp4-mtp.yaml logs -f
```

The OpenAI-compatible API binds to `127.0.0.1:8000`. Run the dependency-free
client once the server is ready:

```bash
python3 benchmark.py
python3 benchmark.py --temperature 0
```

The original result moved Qwen3.8-27B from 3.91 tok/s with BF16 weights to
8.41 tok/s with NVFP4 and 16.04 tok/s with the built-in MTP head on the same
Spark. See [BENCHMARK.md](BENCHMARK.md) for the configuration, repetitions,
prefill measurements, and comparison limits.

Use `compose.yaml` for BF16 or `compose.nvfp4.yaml` for NVFP4 without MTP. Only
one configuration should own port 8000 at a time. Cached weights and compiled
artifacts remain under ignored `data/` paths.

## Run SparkBench

SparkBench is the manifest-driven harness for repeatable multi-model work. It
adds pinned artifact resolution, frozen plans, lifecycle checks, telemetry,
resumable execution, capability validators, and deterministic summaries.

```bash
# Inspect profiles and local caches.
python3 sparkbench.py inventory --sizes
python3 sparkbench.py list --verbose

# Fetch an explicitly pinned snapshot, then run one profile.
python3 sparkbench.py fetch qwen38-27b-ud-q5-k-xl-llamacpp
python3 sparkbench.py benchmark qwen38-27b-ud-q5-k-xl-llamacpp \
  --suite manifests/suites/core.toml

# Run a filtered matrix or an offline direct adapter.
python3 sparkbench.py matrix --backend ollama --task chat
python3 sparkbench.py diffusion-direct \
  nemotron-labs-diffusion-14b-transformers-direct
python3 sparkbench.py trtllm-direct \
  phi-4-multimodal-instruct-fp8-trtllm-audio --timeout 7200

# Resume and audit frozen work without changing its plan.
python3 sparkbench.py run results/<run-directory>
python3 sparkbench.py resume results/<run-directory>
python3 sparkbench.py summarize results/<run-directory>
python3 sparkbench.py audit-matrix results/matrices/<matrix-directory>
```

Run one inference configuration at a time. SparkBench refuses unrelated GPU or
container workloads and blocks implicit downloads unless `--allow-download` is
explicit. Managed services stay on loopback, and cleanup is checked by process
identity.

The default `smoke.toml` suite checks admission and basic capabilities;
`quick.toml` is a short performance screen; `core.toml` repeats decode,
prefill, concurrency, long-context, structured-output, tool, and small
exact-answer cases. Specialized suites under `manifests/suites/` cover
agentic tool loops, reasoning, embeddings, reranking, vision, OCR, ASR,
diffusion, speculative depth, and long context.

For matched perplexity, hold the base model, dataset hash, runtime, chunk count,
and context size constant:

```bash
python3 sparkbench.py perplexity qwen38-27b-ud-q5-k-xl-llamacpp \
  --dataset /absolute/path/to/wiki.test.raw --chunks 64 --ctx-size 512 \
  --timeout 3600
```

`content_battery.py` measures an existing OpenAI-compatible server without
managing its lifecycle. Supply credentials only through `--api-key-file`; its
output is scalar-only and excludes prompts, completions, request tags, and the
key.

## Publish sanitized evidence

Raw `results/`, `data/`, and `logs/` stay local and ignored. To create the
reviewable scalar-only archive described above:

```bash
python3 sparkbench.py export-evidence \
  --results results --output evidence --replace
python3 sparkbench.py verify-evidence evidence
# After staging the intended commit:
python3 sparkbench.py verify-evidence evidence --staged
```

The exporter fails closed on unknown schemas and unsafe source files. It keeps
measurements, validation outcomes, telemetry, terminal state, and reproducible
artifact/runtime pins while excluding captured inputs or outputs, reasoning,
tool payloads, request identifiers, local paths, logs, media, and credentials.
See the [publication policy](BENCHMARK.md#publishing-sanitized-evidence) before
committing a refreshed archive.

## Repository map

| Path | Purpose | Git policy |
| --- | --- | --- |
| `sparkbench.py`, `bench/` | CLI and harness modules | Tracked |
| `manifests/models.toml` | Pinned model/runtime profiles | Tracked |
| `manifests/suites/` | Schema-versioned workload suites | Tracked |
| `docs/`, `BENCHMARK.md` | Analysis and protocol | Tracked |
| `evidence/` | Deterministic, sanitized scalar evidence after export | Tracked |
| `patches/ninfer/` | Bounded local NInfer experiment; not an upstream fork | Tracked |
| `results/`, `data/`, `logs/` | Raw runs, weights, caches, media, and logs | Ignored |
| `tests/` | Offline unit and schema tests | Tracked |

Validate code changes with:

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile sparkbench.py benchmark.py content_battery.py bench/*.py tests/*.py
```

## License

MIT. Model weights, datasets, and container images retain their own licenses.
