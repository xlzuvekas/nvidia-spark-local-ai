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
