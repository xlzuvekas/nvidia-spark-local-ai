# Qwen3.8-27B DGX Spark benchmark

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
