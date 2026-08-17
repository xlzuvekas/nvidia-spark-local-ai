# Qwen3.8-27B DGX Spark benchmark

This file preserves the exact original Qwen3.8 experiment first. The
repository-wide SparkBench measurement and publication contract follows in
[SparkBench protocol and evidence publication](#sparkbench-protocol-and-evidence-publication).

Measured on 2026-08-14 with the configuration in `compose.yaml`:

- NVIDIA GB10, one GPU
- Official BF16 `Qwen/Qwen3.8-27B` weights
- vLLM `0.1.dev19754+g3a0914114`
- 65,536-token maximum context
- FP8 KV cache
- One concurrent sequence
- 52% GPU/unified-memory utilization
- Thinking disabled

## Decode

Three warm runs, each capped at 256 generated tokens:

| Run | TTFT | Elapsed | Decode rate |
| --- | ---: | ---: | ---: |
| 1 | 0.423 s | 65.38 s | 3.93 tok/s |
| 2 | 0.353 s | 66.16 s | 3.87 tok/s |
| 3 | 0.314 s | 65.46 s | 3.91 tok/s |

Median TTFT was **0.353 seconds** and median decode throughput was
**3.91 tokens/second**. GPU SM utilization held at 96% throughout the sampled
decode window.

## Prefill

These are end-to-end approximations calculated as prompt tokens divided by
time to first output token. Each prompt was unique to avoid prefix-cache hits.

| Prompt tokens | TTFT | Approximate prefill rate |
| ---: | ---: | ---: |
| 173 | 0.389 s | 445 tok/s |
| 1,069 | 0.983 s | 1,088 tok/s |
| 4,141 | 3.615 s | 1,146 tok/s |

Run the benchmark again with:

```bash
python3 benchmark.py
```

## Quantization and MTP comparison

The same benchmark was repeated with `Inferact/Qwen3.8-27B-NVFP4`, first
without speculative decoding and then with the model's built-in MTP head using
three draft tokens.

| Configuration | Resident weights | Median TTFT | Median decode | 4,141-token prefill |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 51.1 GiB | 0.353 s | 3.91 tok/s | 1,146 tok/s |
| NVFP4 | 24.18 GiB | **0.163 s** | 8.41 tok/s | **1,995 tok/s** |
| NVFP4 + MTP (3 tokens) | 24.97 GiB | 0.398 s | **15.14 tok/s** | 1,834 tok/s |

NVFP4 was 2.15 times faster than BF16 for decode. Adding MTP made NVFP4 1.80
times faster again, or 3.87 times faster than BF16 overall. During the MTP
benchmark, 495 of 828 drafted tokens were accepted (59.8%). MTP therefore
clearly benefits sustained generation on this GB10, while plain NVFP4 has the
best latency for short responses and the best measured prefill throughput.

NVFP4 used the native `FlashInferCutlassNvFp4LinearKernel`; it did not fall
back to BF16 execution.

## DGX Spark tuning sweep

A subsequent single-sequence sweep tested the main speculative-decoding and
scheduler controls. All decode results are medians of three 256-token runs at
temperature 0.7 unless noted otherwise.

| Configuration | Median decode | Result |
| --- | ---: | --- |
| MTP 2, 4,096 scheduled tokens | 15.45 tok/s | Slightly slower |
| MTP 3, 4,096 scheduled tokens | **16.04 tok/s** | Best validated setting |
| MTP 4, 4,096 scheduled tokens | 14.13 tok/s | Verification overhead wins |
| MTP 3, 8,192 scheduled tokens | 15.24 tok/s | No single-sequence benefit |
| MTP 3, 8,192 tokens, temperature 0 | 15.41 tok/s | More consistent, not faster |
| MTP 3, text-only mode | 15.38 tok/s | No decode benefit; slower prefill |

The winning run measured 16.39, 16.04, and 15.73 tok/s. Disabling chunked
prefill was rejected by vLLM because a non-chunked scheduler must set
`max_num_batched_tokens` at least as high as the 65,536-token maximum context.
NVFP4 KV cache is present in this vLLM build but its FlashInfer backend is
restricted to the SM100 family; the GB10 is SM121, so FP8 remains the supported
KV-cache format here.

## SparkBench quick-suite concurrency profile

A later cached-only run exercised the reproducible `quick.toml` suite with the
`qwen38-27b-nvfp4-mtp3-throughput` profile. It used the exact NVFP4 revision in
`manifests/models.toml`, MTP depth 3, FP8 KV cache, a 32,768-token served
context, eight sequence slots, 8,192 scheduled tokens, and temperature zero.
Thinking was disabled. Each case followed one warm-up request; decode and
prefill used three measured repetitions, while each concurrency level used two
measured bursts. The sample counts are too small for p95 claims.

| Workload | Measured requests | Tokens/request (prompt → output) | Median TTFT | Median E2E | Aggregate output | Median client decode estimate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single stream | 3 | 79 → 128 | 0.366 s | 7.556 s | 17.27 tok/s | 17.68 tok/s |
| Concurrency 2 | 4 | 79 → 64 | 0.609 s | 4.474 s | 27.90 tok/s | 16.58 tok/s |
| Concurrency 4 | 8 | 79 → 64 | 0.561 s | 4.473 s | **53.93 tok/s** | 16.47 tok/s |

The concurrency figures use total completed output tokens divided by measured
case wall time, including minor harness time between bursts. Four-way serving
delivered 1.93 times the concurrency-two aggregate throughput with essentially
unchanged median E2E latency (4.4731 versus 4.4738 seconds). The streaming server
bundled multiple tokens in some SSE emissions, so the per-request decode column
is explicitly an estimate; use aggregate output throughput as the primary
concurrency result.

| Actual prompt tokens | Median TTFT | Client-TTFT prefill approximation |
| ---: | ---: | ---: |
| 324 | 0.292 s | 1,111 tok/s |
| 2,117 | 0.992 s | 2,134 tok/s |
| 8,261 | 4.311 s | 1,916 tok/s |

The separate long-context check admitted 8,284 prompt tokens, returned the
hidden key correctly, and reached first output in 4.444 seconds. All seven suite
cases completed without validation failures. This single probe does not validate
the full 32K served context or the checkpoint's 262K native limit.

Cached-only process startup took 453.44 seconds, including 173.21 seconds for
model loading plus compilation, graph setup, and FP4 autotuning. vLLM reported
24.97 GiB of model memory and 64.97 GiB available for KV cache: 1,139,598 tokens,
or a theoretical 34.78 concurrent 32K requests before scheduler and workload
limits. Minimum system-available memory during measured cases was 13.02 GiB.
Startup peaked at 100.64 W and 81 °C in sampled GPU telemetry. The compiled and
autotuned artifacts were persisted, but this run did not measure a subsequent
warm-start time. This remains an exploratory serving profile. vLLM also warned
that FP8 KV-cache scales defaulted to 1.0 without calibration, so the run
validates performance and its one retrieval probe—not broad model accuracy.

## SparkBench Protocol and Evidence Publication

SparkBench extends the focused experiment above to multiple models and runtime
families. Every managed run freezes its model profile, suite, artifact pins,
runtime configuration, hardware identity, and harness revision before serving
starts. The frozen plan is immutable for the life of the run; interrupted work
is resumed from that plan instead of being silently regenerated.

### Execution rules

- Run one inference configuration at a time. Refuse unrelated GPU compute and
  container workloads rather than stopping them implicitly.
- Resolve pinned cached artifacts before measurement. Network acquisition is a
  separate `fetch` step or requires an explicit `--allow-download` flag.
- Bind managed inference endpoints to loopback. Preserve per-run authentication
  and verify cleanup using process or container identity.
- Warm up before measured repetitions and use unique prefill prompts so prefix
  caching cannot inflate fresh-prompt results.
- Record prompt tokens, output tokens, TTFT, end-to-end time, aggregate output
  throughput, runtime-native counters when available, validation state, and
  sampled telemetry with explicit units.
- Preserve failures, partial results, early stops, and unsupported admissions.
  Do not turn successful transport or accelerated invalid emissions into a
  semantic-quality claim.

Aggregate output throughput is completed output tokens divided by measured case
wall time and is the primary cross-request decode metric. Per-request client
decode rates are secondary when a server can bundle multiple tokens into one
stream event. Client-TTFT prefill is an approximation unless the runtime reports
an isolated prompt-evaluation duration.

Concurrency results are comparable only when the serving-slot geometry is the
same. A one-slot profile receiving C2, C4, or C8 requests measures queued
aggregate service, not parallel-sequence scaling. Similarly, compare
perplexity only for the same base model, tokenizer, dataset hash, runtime,
chunk count, and context size. Exact revisions, image digests, artifact hashes,
hardware, date, and validation state accompany publishable conclusions.

### Raw run records

The complete local source of truth lives under ignored `results/` paths. A
managed run can include `plan.json`, an append-only event journal, telemetry,
server provenance and logs, generated summaries, and cleanup evidence. Matrix,
perplexity, direct-adapter, llama-bench, NInfer, and content-battery campaigns
have their own bounded source layouts.

Raw records are intentionally not committed. They can contain captured prompts
or completions, reasoning, tool calls, request identifiers, process details,
host paths, raw media, logs, or ephemeral credentials. `data/` and `logs/` are
also local-only because they hold weights, caches, media, and runtime output.
Exact raw run IDs may be cited in a report for local traceability, but the path
name is not itself public evidence.

### Publishing sanitized evidence

An evidence export creates a deterministic tracked archive without copying raw
records:

```bash
python3 sparkbench.py export-evidence \
  --results results --output evidence --replace
python3 sparkbench.py verify-evidence evidence
# After staging the intended commit:
python3 sparkbench.py verify-evidence evidence --staged
```

The intended archive entry points are `evidence/README.md` for people and
`evidence/index.json` for tools. Run bundles retain scalar request measurements,
case aggregates, validation booleans and bounded categories, lifecycle state,
compact numeric telemetry, and reproducibility pins such as artifact hashes,
runtime revisions, image digests, hardware, and harness revision. Campaign and
matrix bundles retain only their explicitly supported scalar schemas.

The exporter must fail closed. Unknown fields or schema versions, malformed or
non-finite numbers, duplicate JSON keys, unsafe file types or links, unexpected
source files, unsafe output placement, and configured size limits are errors.
Every bundle and the archive root carry checksums, and verification recomputes
those checksums while cross-checking index counts and references.

The archive excludes all captured input and output text, reasoning text, tool
arguments or responses, transcriptions, request or sample tags, raw identifiers,
local paths, commands, environment variables, logs, media, model weights,
caches, and credentials. String fields that remain are allowlisted bounded
labels, public model/runtime identifiers, status values, units, hashes, and
other non-content provenance.

Before committing a refresh:

1. Stop writes to the selected raw run corpus and let the exporter acquire the
   benchmark lock.
2. Export twice and confirm the second pass is unchanged.
3. Run `verify-evidence` against the finished archive.
4. Inspect the Git diff and staged file list; confirm that only documentation,
   code, tests, manifests, patches, and sanitized `evidence/` files are staged.
5. Run `python3 sparkbench.py verify-evidence evidence --staged` to reconstruct
   and validate the exact Git-index evidence tree and secret-scan every staged
   text blob.
6. Run the repository tests before committing or pushing.

Never hand-copy a raw result into `evidence/`, loosen an allowlist merely to make
an export pass, or publish a number without its status and comparison geometry.
When a legitimate schema evolves, update the exporter, add an offline regression
fixture, regenerate the archive, and document any conclusion that changed.
