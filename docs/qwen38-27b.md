# Getting Qwen3.8-27B past 15 tok/s on NVIDIA DGX Spark

The NVIDIA DGX Spark is an unusual inference machine. Its GB10 combines an
ARM CPU and Blackwell GPU around 128 GB of unified LPDDR5X memory. That capacity
makes models such as Qwen3.8-27B pleasantly easy to fit, but the shared 273 GB/s
memory interface also dictates how fast dense autoregressive decoding can run.

This experiment started with the official BF16 Qwen3.8-27B checkpoint and a
simple goal: serve it locally through an OpenAI-compatible API, then find the
highest repeatable single-request generation speed without compromising the
65,536-token context window.

## Baseline

The BF16 checkpoint occupied 51.1 GiB and decoded at a 3.91 tok/s median. GPU
utilization was approximately 96%, so the system was not waiting on the CPU or
request scheduler. The obvious next step was reducing the amount of model data
that must move for every autoregressive step.

The `Inferact/Qwen3.8-27B-NVFP4` checkpoint reduced resident weight memory to
24.18 GiB and raised decode to 8.41 tok/s. Logs confirmed execution through
`FlashInferCutlassNvFp4LinearKernel`; this was native NVFP4 execution rather
than a dequantized BF16 fallback.

That result is close to what Spark's memory system predicts. Streaming roughly
25 GB of weights over a theoretical 273 GB/s interface permits about eleven
complete model passes per second before any kernel, cache, or system overhead.
The observed 8.41 tok/s is consistent with a workload dominated by moving
weights through shared memory.

## Why MTP changes the equation

Multi-token prediction does not make Spark's memory faster. Instead, it lets
one expensive target-model pass verify multiple candidate tokens. Every
additional accepted token amortizes the cost of reading the model weights.

Qwen3.8-27B includes an MTP head, so vLLM can use it without loading a second
draft model. With three speculative tokens, decode initially reached 15.14
tok/s and accepted 495 of 828 drafted tokens, a 59.8% acceptance rate. A later
validation run produced 16.39, 16.04, and 15.73 tok/s, for a 16.04 tok/s median.

This is why effective token throughput can exceed the rough one-pass-per-token
bandwidth ceiling: some passes yield more than one accepted output token.

## Tuning sweep

We changed one major serving control at a time and measured three 256-token
runs after startup and warm-up.

| Configuration | Median decode | Observation |
| --- | ---: | --- |
| MTP depth 2, 4,096-token scheduler | 15.45 tok/s | Less useful speculation |
| MTP depth 3, 4,096-token scheduler | **16.04 tok/s** | Best validated setting |
| MTP depth 4, 4,096-token scheduler | 14.13 tok/s | Extra verification lost |
| MTP depth 3, 8,192-token scheduler | 15.24 tok/s | No single-stream benefit |
| Same, deterministic sampling | 15.41 tok/s | Consistent but not faster |
| MTP depth 3, language-model-only | 15.38 tok/s | No decode improvement |

Depth four was an important negative result. More draft tokens do not
automatically mean more output throughput: later draft positions have lower
acceptance, while the target still pays to verify them.

Increasing `max_num_batched_tokens` addresses vLLM's general speculative-slot
warning but did not help one active sequence. It may still matter for concurrent
throughput. Likewise, deterministic sampling made the run-to-run results more
stable but did not beat the normal chat sampling median.

Language-model-only mode removed the unused vision reservation, but this did
not improve decode and reduced measured prefill in this particular run. It is
still reasonable when minimizing memory use or startup work is more important
than peak prefill.

Disabling chunked prefill was not a valid comparison with a 65K maximum context
and 4,096 scheduled tokens. vLLM requires a non-chunked scheduler to admit the
entire maximum sequence, meaning `max_num_batched_tokens` would need to rise to
at least 65,536. That is a different memory and compilation tradeoff, so the
working configuration retains chunked prefill.

## Why not FP4 KV cache?

Weight quantization and KV-cache quantization are separate kernel paths. The
tested vLLM package recognizes `nvfp4` as a cache type and contains FlashInfer
NVFP4 cache code, but its backend checks for the SM100 capability family. GB10
reports SM121, so NVFP4 KV is not supported by this image on Spark.

FP8 KV is also a sensible performance choice for this workload. At short and
moderate contexts, reading approximately 25 GB of weights dominates the much
smaller KV-cache read. Halving KV storage would mostly increase long-context
capacity and concurrency. It is worth revisiting once SM121 routing is
officially supported, especially for 64K contexts, but it is unlikely to double
short-context decode.

## Prefill behaves differently

The 4,141-token prefill probes reached roughly 1,500–2,000 tok/s with NVFP4.
Prefill processes many tokens in parallel and turns the matrix operations into
larger, more efficient GPU work. Decode processes one new position at a time
and repeatedly streams weights, so it cannot exploit the GPU in the same way.

That contrast—very fast prefill but bandwidth-bound decode—is expected on
Spark, not evidence that the decode configuration is broken.

## Operational lessons

Persist `/root/.cache/vllm`. A container recreation otherwise discards compiled
graphs and FlashInfer autotuning results, adding minutes to every experimental
restart. The Compose configuration mounts this at `data/vllm/`.

Remember that Spark uses unified memory. CPU and GPU capacity is convenient,
but CPU-side memory traffic can contend with GPU inference. Keep unrelated
memory-heavy jobs quiet during latency-sensitive serving, leave enough capacity
to avoid active swapping, use the supplied power adapter, and maintain airflow
for sustained clocks.

Finally, record cold and warm behavior separately. The first request following
startup often carries graph or kernel warm-up cost and should not be presented
as steady-state throughput.

## Reproduce it

Start the winning service:

```bash
docker compose -f compose.nvfp4.yaml -f compose.nvfp4-mtp.yaml up -d
```

Wait for `/v1/models` to respond, then run:

```bash
python3 benchmark.py
```

The exact measurements and software versions are preserved in
[`BENCHMARK.md`](../BENCHMARK.md).
