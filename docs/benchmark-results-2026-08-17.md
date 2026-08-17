# DGX Spark Overnight Benchmark Results — 2026-08-17

This report continues the
[2026-08-16 campaign](benchmark-results-2026-08-16.md) with terminal overnight
runs on one NVIDIA GB10. The Qwen3.6 evidence now covers twelve completed native
llama.cpp runs for the Unsloth Qwen3.6 35B-A3B Q4_K_XL GGUF: baseline and
MTP2 smoke, matched decode, quick, long-context, core, and standalone
chat-quality screens. Nonterminal runs are deliberately excluded. Generated
`results/` artifacts remain local and uncommitted by repository policy; the
exact evidence directories are listed below. A second terminal block covers
the pinned SGLang NVFP4 + DSpark recipe,
its core suite, and two content-dependent throughput batteries. A third block
adds native Qwen3.8 Q5 smoke, quick, and core screens plus a matched
four-quantization WikiText-2 perplexity screen, followed by a matched Q4
baseline/MTP5 262K-context retrieval comparison. A final bounded block records
the Muse-Glimmer 30B Q4_K_XL smoke and quick diagnostics with and without its
DFlash15 sidecar; all four reached terminal state, but neither profile passed
the semantic-output gate.

## Measurement Rules

Aggregate output throughput is completed output tokens divided by measured
case wall time and is the primary decode metric. Median per-request client
decode is secondary. Client-TTFT prefill is an approximation computed as
observed prompt tokens per request divided by median time to first token; it
is not an isolated llama.cpp prompt-evaluation counter.

For the Qwen3.6 pair below, the quick suite has only three decode repetitions,
three prefill repetitions, and two concurrency bursts. The matched decode sweep
has five repetitions. These samples support medians and directional comparisons,
not p95 or broad variance claims. Both Qwen3.6 profiles used
`runtime_parallel = 1`, so C2/C4 requests were queued through one slot; those
cells compare aggregate queued service throughput, not parallel-sequence
scaling. The smoke chat and 8K needle cells have one measured request each and
are primarily protocol/correctness checks.

For Qwen3.6 MTP profiles, llama.cpp speculative-decoding counters cover the
complete persisted server lifetime, including the prime request, warmups, and
measured requests. They prove that the configured MTP depth was exercised, but
cannot be assigned to an individual measured case. Later sections state their
own slot geometry and speculative-decoding method.

## Frozen Artifact and Runtime

All twelve Qwen3.6 plans resolved the same cached
[`unsloth/Qwen3.6-35B-A3B-MTP-GGUF` snapshot](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/tree/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d),
revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`. The exact file was
`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, 22,853,663,008 bytes (21.284 GiB), at
`sha256:55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1`.
Every run repeated artifact validation and recorded that digest.

The native runtime was
[`llama.cpp` b10453](https://github.com/ggml-org/llama.cpp/releases/tag/b10453),
source commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`. Its 58,085,600-byte
`llama-server` binary was pinned at
`sha256:ae1bd49f869ff3397b2a5d757fcf010c6eaaf16c4e3071a15861312defcd4e40`.
This is the same pinned native runtime family used for the prior
[Qwen3.8 GGUF campaign](benchmark-results-2026-08-16.md#qwen38-27b-unsloth-q4-gguf-on-managed-llamacpp)
and follows NVIDIA's
[DGX Spark llama.cpp playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/1fb66f059ee427c5a3678b3117ef73aab042b458/nvidia/llama-cpp/README.md).

Plans recorded an NVIDIA GB10 with driver 580.142 and CUDA 12.1, Linux
6.17.0-1014-nvidia on aarch64, and 125,508,472 KiB host memory. Both profiles
offloaded all layers, enabled flash attention, disabled fit heuristics and
reasoning, used 8,192/512 batch/ubatch sizes, Q8_0 KV caches, one runtime slot,
and a 262,144-token configured/native context. MTP2 changed only the native
speculation settings: `draft-mtp`, maximum draft length two, Q8_0 draft KV,
and backend sampling. This is therefore a same-artifact, same-runtime
comparison rather than a comparison with the prior
[official Qwen3.6 NVFP4 vLLM run](benchmark-results-2026-08-16.md#qwen36-35b-a3b-nvfp4-mtp3).

## Terminal Evidence

| Purpose | Exact run directory | Status |
| --- | --- | --- |
| Baseline smoke | `results/20260817T061055Z-qwen36-35b-a3b-ud-q4-k-xl-llamacpp-smoke-2595cdaf` | Complete |
| MTP2 smoke | `results/20260817T061240Z-qwen36-35b-a3b-ud-q4-k-xl-llamacpp-mtp2-smoke-23b59b0a` | Complete |
| Baseline D256 | `results/20260817T061319Z-qwen36-35b-a3b-ud-q4-k-xl-llamacpp-llamacpp-mtp-depth-ff758044` | Complete |
| MTP2 D256 | `results/20260817T061434Z-qwen36-35b-a3b-ud-q4-k-xl-llamacpp-mtp2-llamacpp-mtp-depth-1d874877` | Complete |
| Baseline quick | `results/20260817T061534Z-qwen36-35b-a3b-ud-q4-k-xl-llamacpp-quick-e79b299c` | Complete |
| MTP2 quick | `results/20260817T061707Z-qwen36-35b-a3b-ud-q4-k-xl-llamacpp-mtp2-quick-598db72e` | Complete |
| Baseline long context | `results/20260817T061841Z-qwen36-35b-a3b-ud-q4-k-xl-llamacpp-llamacpp-long-context-dd5ba2c1` | Complete |
| MTP2 long context | `results/20260817T062812Z-qwen36-35b-a3b-ud-q4-k-xl-llamacpp-mtp2-llamacpp-long-context-e63f7a09` | Complete |

The eight pre-core summaries have terminal `completed` status, valid startup
measurements, no measurement annotations, and no invalid or failed measured
cases. Server readiness took 4.046–6.057 seconds after artifact validation;
these were sequential cached-artifact launches, not controlled cold-start
replicates.

## Smoke Capability Screen

Both profiles completed all three declared/supported smoke cases: chat, strict
JSON, and tool calling. The chat request produced 32 tokens at 45.265880
aggregate tok/s for baseline and 53.647639 for MTP2; median E2E was
0.690944/0.580550 seconds. Those are single requests with 79/75 prompt tokens,
so the apparent 18.5% throughput difference is not treated as a performance
estimate. Vision, embeddings, and reranking were profile-declared unsupported
and skipped rather than attempted failures.

## Matched Five-Request Decode

The D256 micro-sweep provides the strongest small-sample MTP comparison. Every
request reached its 256-token budget and passed validation.

| Metric | Baseline | MTP2 | MTP2 change |
| --- | ---: | ---: | ---: |
| Requests / output tokens | 5 / 1,280 | 5 / 1,280 | Matched |
| Observed prompt tokens | 400 | 395 | -1/request |
| Measured case wall time | 21.778777 s | 16.886653 s | -22.463% |
| Aggregate output | 58.772815 tok/s | **75.799507 tok/s** | **+28.970%** |
| Median client decode | 61.417796 tok/s | **79.601125 tok/s** | **+29.606%** |
| Median TTFT | 0.169841 s | 0.186347 s | +9.719% |
| Median E2E | 4.323832 s | **3.389820 s** | -21.601% |

MTP2 therefore improved the primary aggregate metric by 29.0% while adding
16.5 ms to median first-token latency. The lower E2E reflects the faster
post-first-token generation. The one-token-per-request prompt-count difference
is retained explicitly; no claim beyond this frozen prompt family and five
sequential repetitions is warranted.

## Quick Decode, Queue, and Prefill Screen

The independently recomputed quick generation cells all completed their
output budgets and passed validation:

| Case | Requests / output tokens | Baseline aggregate | MTP2 aggregate | MTP2 change | Baseline / MTP2 median client decode | Baseline / MTP2 median TTFT | Baseline / MTP2 median E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D128 | 3 / 384 each | 57.664841 | **72.566673** | **+25.842%** | 62.380146 / 80.363125 | 0.167721 / 0.188481 s | 2.203626 / 1.764676 s |
| C2 queue | 4 / 256 each | 51.776517 | **64.129045** | **+23.857%** | 59.173711 / 78.648093 | 0.777208 / 0.672502 s | 1.844721 / 1.467736 s |
| C4 queue | 8 / 512 each | 52.108395 | **65.151767** | **+25.031%** | 59.711220 / 78.964317 | 2.001237 / 1.649808 s | 3.053649 / 2.437393 s |

Aggregate and client-decode columns are tokens/s. Baseline/MTP2 prompt-token
totals were 243/240 for D128, 316/312 for C2, and 608/616 for C4. The frozen
requests thus differed by one observed prompt token per request while retaining
the same output budgets and request shapes. C2 and C4 stayed nearly flat
within each profile because one runtime slot serialized the queue. Their TTFT
and E2E medians include queue waiting and must not be read as independent
per-sequence latency or batching gains.

Prefill did not improve with MTP2:

| Target | Requests | Baseline / MTP2 total prompt tokens | Baseline / MTP2 median TTFT | Baseline / MTP2 approximate prompt tok/s | MTP2 change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 3 each | 969 / 975 | 0.235058 / 0.258918 s | 1,374.131 / 1,255.224 | -8.653% |
| 2,048 | 3 each | 6,351 / 6,351 | 0.993209 / 1.041892 s | 2,131.475 / 2,031.880 | -4.673% |
| 8,192 | 3 each | 24,786 / 24,780 | 3.298797 / 3.426211 s | 2,504.549 / 2,410.826 | -3.742% |

Each prefill request emitted only one measured token and had no content-quality
validator. These values include client/request/first-emission overhead and
should not be compared with server-native prompt-evaluation counters.

Both one-request 8K exact-key cases passed. Baseline used 8,285 prompt tokens
with 3.505188-second TTFT and 3.748970-second E2E; MTP2 used 8,284 with
3.649520-second TTFT and 3.792822-second E2E. That is 4.118% slower TTFT and
1.170% slower E2E for MTP2 in a single trial. By itself that is not evidence
of a general long-context penalty; the matched sweep below tests the direction
at four larger tiers.

## MTP2 Execution Evidence

The MTP counters demonstrate that the embedded depth-two path was active, not
merely configured:

| MTP2 run | Drafts | Draft tokens | Accepted draft tokens | Acceptance | Accepted at positions 0 / 1 | Average proposal length | Depth-two proof |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Smoke | 31 | 62 | 53 | 85.484% | 29 / 24 | 2.000 | Pass |
| D256 | 699 | 1,395 | 1,089 | 78.065% | 603 / 486 | 1.996 | Pass |
| Quick | 554 | 1,094 | 855 | 78.154% | 494 / 361 | 1.975 | Pass |
| Long context | 47 | 94 | 92 | **97.872%** | 47 / 45 | 2.000 | Pass |

Acceptance is accepted draft tokens divided by proposed draft tokens. Each
snapshot accepted tokens at draft position one, so the deepest accepted draft
depth was two. The matched baseline snapshots requested no speculation and
recorded zero draft activity.

## Matched Long-Context Retrieval

Both single-slot profiles completed every exact-key retrieval through actual
245,857/245,854-token baseline/MTP2 prompts. There were no warmups,
context-limited cases, skips, failures, or annotations; all 20 responses
stopped naturally.

| Suite target | Requests/profile | Baseline / MTP2 actual prompt tokens/request | Baseline / MTP2 median TTFT | MTP2 TTFT change | Baseline / MTP2 median E2E | MTP2 E2E change | Exact-key validation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 3 | 32,862–32,863 / 32,861–32,862 | 13.682551 / 14.274537 s | +4.327% | 13.942985 / 14.441513 s | +3.575% | 3/3 each |
| 65,536 | 3 | 65,626–65,629 / 65,630–65,631 | 29.795965 / 31.049617 s | +4.207% | 30.068577 / 31.313528 s | +4.140% | 3/3 each |
| 131,072 | 3 | 131,168–131,169 / 131,167–131,169 | 70.976926 / 75.024521 s | +5.703% | 71.347721 / 75.386113 s | +5.660% | 3/3 each |
| 245,760 | 1 | 245,857 / 245,854 | 173.787910 / 185.898422 s | +6.969% | 174.335260 / 186.335244 s | +6.883% | 1/1 each |

Both final prompts occupied 93.79% of the configured 262,144-token context,
establishing near-limit fit and exact retrieval for this one generated-key
fixture. MTP2 median TTFT was 4.2–7.0% slower and E2E was 3.6–6.9% slower at
all four tiers. The 245K tier has only one request per profile, and the suite
tests exact-key retrieval rather than broad long-context comprehension.
Output counts varied from 10 to 14 tokens, so these prefill-dominated rows are
reported as TTFT/E2E rather than ranked by output-token throughput. With such
short completions, the MTP2 path had little decode work over which to recover
its prompt-to-first-token overhead; this does not contradict the longer D128
and D256 decode gains.

## Capacity and Telemetry Bounds

| Run | Minimum sampled host `MemAvailable` | Peak sampled temperature | Peak sampled power |
| --- | ---: | ---: | ---: |
| Baseline D256 | 86.586 GiB | 58 °C | 41.58 W |
| MTP2 D256 | 84.464 GiB | 59 °C | 43.09 W |
| Baseline quick | 82.722 GiB | 72 °C | 76.00 W |
| MTP2 quick | 79.353 GiB | 71 °C | 76.15 W |
| Baseline long context | 79.248 GiB | 86 °C | 91.21 W |
| MTP2 long context | 76.197 GiB | 87 °C | 90.93 W |

Sampled swap-free did not fall from the first to final sample in any of these
runs. Across the matched long-context runs, host-wide `MemAvailable` stayed at
or above 79.248/76.197 GiB for baseline/MTP2, while sampled temperature peaked
at 86/87 °C. These measurements show that the frozen profiles fit this host
without observed swap growth; host-wide memory is not a model-resident
footprint, sequential cross-run differences are not isolated model allocation,
and one-second power samples are not board-total energy measurements.

## Qwen3.6 Bounded Conclusions

- The Unsloth Qwen3.6 35B-A3B UD-Q4_K_XL artifact is operational with and
  without MTP2 on the pinned native llama.cpp stack for chat, strict JSON,
  tool calling, and exact-key retrieval through 245,857/245,854-token prompts.
- Embedded MTP2 was measurably active and improved matched small-sample decode
  aggregate throughput by 23.9–29.0% across D128, D256, and the serialized
  C2/C4 queues.
- MTP2 did not improve the client-TTFT prefill approximations; the three
  observed targets were 3.7–8.7% lower. The matched D256 median TTFT was 9.7%
  higher even as E2E fell 21.6%.
- In the prefill-dominated long-context retrieval sweep, MTP2 was 4.2–7.0%
  slower to first token and 3.6–6.9% slower end to end while retaining 10/10
  exact-key correctness and demonstrably exercising depth two.
- These results do not compare GGUF quality with NVFP4, do not establish
  multi-slot scaling, and do not justify p95, cold-start, general quality, or
  general long-context-comprehension claims.

## Qwen3.8 NVFP4 + DSpark on Managed SGLang

This block evaluates the configuration from the
[NVIDIA forum reproduction](https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257)
and the
[`hasso5703/dgx-spark-qwen38` repository](https://github.com/hasso5703/dgx-spark-qwen38).
The exact upstream
[benchmark methodology](https://github.com/hasso5703/dgx-spark-qwen38/blob/main/BENCHMARKS.md)
is reproduced separately from the SparkBench core suite and the fresh-prompt
streaming battery; their metrics are not silently mixed.

### Exact Evidence and Recipe Pins

| Artifact | Exact path | Terminal state / SHA256 |
| --- | --- | --- |
| Initial cache-permission attempt | `results/20260817T065426Z-qwen38-27b-nvfp4-dspark-sglang-core-6b79826a` | Aborted before serving; summary `22a1272ac060531ee170d6dc352e4c326f3e58fb83a0fc7e3e4f33d3e8696f28` |
| Managed core run | `results/20260817T065612Z-qwen38-27b-nvfp4-dspark-sglang-core-6b79826a` | `completed_server_kept`; summary `de468346bd9810aa56a651d2047aede4cf601b31de2cdf367e7318d6e74cef80` |
| Fresh streaming battery | `results/content-battery-dspark-sglang-20260817.json` | Complete scalar artifact; `e0e88dacc86a7e5bff960f6fe4f3456b56a7cc29d325b3f22bf987a7f8339fa6` |
| Upstream two-call battery | `/home/xlz/.cache/sparkbench/external/dgx-spark-qwen38/bench-matrix-dspark-sglang-20260817.json` | Complete upstream artifact; `f1515f717befb5645d795c06cf8e1e304a957665724d3ed1362c02df264b08da` |

The managed profile pinned the upstream recipe at
[`3590fb29296b1babd85405daad1eef1c4a3ebe0f`](https://github.com/hasso5703/dgx-spark-qwen38/tree/3590fb29296b1babd85405daad1eef1c4a3ebe0f)
and the container at
`lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1`
(local image ID
`sha256:0076dffa60b76b7bf033c04d05e0cc69d46f2b8cd60aa2468827782afe9bc38f`).
The exact target was
[`RadixArk/Qwen3.8-27B-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4/tree/52d1adc5f38aa5ebf099c29ed7025ba34cfbb854)
at revision `52d1adc5f38aa5ebf099c29ed7025ba34cfbb854`: three weight files totaling
21,921,697,280 bytes. The unquantized draft was
[`RadixArk/Qwen3.8-27B-DSpark`](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/tree/923ed3a8572615643f0137e424e4ce4edd7f1cda)
at revision `923ed3a8572615643f0137e424e4ce4edd7f1cda`, 2,718,576,122 bytes. Combined
weight bytes were 24,640,273,402 (22.948 GiB). The target declares
Apache-2.0; the draft card reports license `other`, so redistribution and
deployment must account for that distinct draft-license status.

The serving flags match the pinned recipe configuration: TP1, 0.50 static
memory fraction, FlashInfer, 8,192-token chunked prefill, prefill CUDA graph
disabled, CUDA graph maximum batch four, DSPARK block size seven with an
unquantized draft, lazy extra-buffer Mamba radix caching, BF16 SSM state,
Mamba cache 96, eight maximum running requests, Torch compile through batch
four, and two continuous decode steps. The Qwen3 reasoning and Qwen3 Coder
tool parsers were enabled. SparkBench reproduced the serving configuration,
but did not apply the upstream installer's separate `minimal` chat-template
patch. It is therefore a pinned recipe-configuration reproduction, not an
exact reproduction of every upstream request-formatting/install step.

### Cache, Authentication, and Network Boundaries

The first attempt stopped at `server_start` with
`PermissionError: [Errno 13] Permission denied: .../data/sglang`. It produced
no measured cases and no usable server. The successful attempt moved the
persistent TorchInductor cache to the user-private path
`~/.cache/sparkbench/sglang/qwen38-27b-nvfp4-dspark-sglang/compile`; the
533.763869-second startup below is its first measured compile/load observation,
not a warm-cache startup distribution.

The successful launch deliberately narrowed several security boundaries:

- Docker published port 30000 only on host `127.0.0.1`, even though SGLang
  listened on `0.0.0.0` inside the container.
- Only the exact target and draft repositories were mounted, read-only. The
  whole Hugging Face cache and its possible credential file were not mounted;
  container HF caches pointed at `/tmp`, implicit-token discovery was disabled,
  and model acquisition was disabled.
- Each run generated an ephemeral Bearer credential, persisted it at mode
  `0600` for authorized local clients, and used it for readiness and OpenAI
  requests. Saved Docker argv and logs redact the key; provenance confirms a
  redacted marker rather than the credential.
- Exact local weights remained pinned, but strict HF offline flags were
  intentionally omitted because this image performs a LongCat model-existence
  lookup during startup. Provenance records
  `documented_longcat_metadata_probe`. This is an explicit metadata-egress
  exception, not a claim that the container was network-sandboxed.

The core run used `--keep-server` so the same authenticated process could
serve both follow-up batteries. Consequently its summary has no shutdown
telemetry and records `completed_server_kept`; the credential was retained for
that bounded handoff. Normal owned cleanup is designed to remove both the
container and key, but any later cleanup is outside these artifacts and is not
claimed here.

### SparkBench Core Throughput

The core summary is terminal `partial`, not failed: 15 cases executed, two of
them failed semantic validation, and three profile-declared capabilities were
skipped. Startup and every measured case remained annotation-free and valid.
All decode and concurrency requests reached their frozen output budgets.
Metrics below were recomputed from `request_complete` token counts and
`case_complete` wall times.

| Case | Requests / output tokens | Aggregate output | Median client decode | Median TTFT | Median E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| D256 | 5 / 1,280 | **26.369420 tok/s** | 26.792914 tok/s | 0.206334 s | 9.728128 s |
| D1024 | 5 / 5,120 | 24.042949 tok/s | 23.212328 tok/s | 0.214313 s | 44.286037 s |
| C1 | 5 / 1,280 | 24.431196 tok/s | 25.076969 tok/s | 0.201149 s | 10.369843 s |
| C2 | 10 / 2,560 | 45.547253 tok/s | 24.129914 tok/s | 0.243492 s | 10.807154 s |
| C4 | 20 / 5,120 | 81.503702 tok/s | 23.735274 tok/s | 0.230372 s | 10.970267 s |
| C8 | 40 / 10,240 | **127.184830 tok/s** | 18.656844 tok/s | 0.541110 s | 14.205846 s |

For example, D256 is exactly `1,280 / 48.541074599 = 26.369420`
tok/s, and C8 is `10,240 / 80.512746652 = 127.184830` tok/s. D1024
aggregate was 8.823% below D256. C2/C4/C8 reached 1.864/3.336/5.206 times C1
aggregate throughput; scaling was material but sublinear, and per-request
client decode fell at higher concurrency. SGLang emitted multi-token SSE
chunks, so aggregate usage-token throughput is primary and these client decode
estimates are not per-token ITL measurements.

Client-TTFT prefill approximations were:

| Suite target | Observed prompt tokens/request | Median TTFT | Approximate prompt tok/s |
| ---: | ---: | ---: | ---: |
| 128 | 236 | 0.246582 s | 957.085 |
| 1,024 | 1,133 | 0.549676 s | 2,061.215 |
| 4,096 | 4,206 | 2.010882 s | **2,091.619** |
| 16,384 | 16,494 | 9.336991 s | 1,766.522 |
| 32,768 | 32,879 | 20.950625 s | 1,569.357 |

Each prefill request emitted one token and carried no semantic validator.
These values include client, request, and first-emission overhead; they are not
server-native prompt-evaluation counters.

### Reasoning-Only Output and Semantic Failures

The performance rows are valid token-generation measurements, but they are
not usable-answer latency measurements. Every one of the 85 D256/D1024 and
C1–C8 requests exhausted its budget with `finish_reason=length`, nonempty
reasoning, and empty visible content. Their validators checked completion
length, not answer semantics.

The semantic cases expose the distinction:

- Strict JSON and tool calling passed 5/5 each. JSON responses stopped
  naturally with visible content; all tool requests ended with a parsed tool
  call.
- The 16K needle used 16,514–16,516 actual prompt tokens and 32 output tokens.
  All three responses ended at the length cap with empty visible content; one
  reasoning trace began reproducing the hidden key but did not return it in
  final content. Exact validation correctly scored 0/3.
- Exact-answer quality scored 0/4. All four requests used the complete
  64-token budget and ended by length. The reasoning traces explicitly
  computed the correct arithmetic, instruction, and code answers, while the
  logic trace was still working through the premises; visible content was
  empty or truncated to an incomplete `FINAL` marker. The result is a
  reasoning-budget/parser-output failure under this frozen request contract,
  not evidence that all four underlying answers were unknown.
- Vision, embeddings, and reranking were profile-declared unsupported and
  skipped rather than attempted failures.

The missing upstream `minimal` template patch is a protocol difference, but
this campaign did not isolate it as the cause. A larger reasoning budget or a
template change might alter visible-answer completion; neither repair is
credited without a matched run.

### Upstream Battery Versus Fresh Streaming Battery

The exact upstream `bench-matrix.sh` artifact used one discarded warmup and,
for each probe, two nonstreaming calls with the same prompt at maximum token
budgets 80 then 680. Its reported rate is
`(completion_680 - completion_80) / (wall_680 - wall_80)`. This attempts to
cancel prefill without streaming, but repeats the exact prompt within each
pair, provides only one delta sample per workload, and persists only a
one-decimal rate—not the four raw token/time values. The upstream documentation
reports that SGLang+DSpark repeats were stable, but this protocol remains
different from a cache-resistant streaming measurement and can be unsafe on
other speculative stacks.

The fresh battery instead used three token-length-matched uniquely tagged
versions of each semantic prompt, interleaved 24 measured requests after one
warmup, greedy sampling, and OpenAI chat-completions SSE. It directly measured
TTFT and computed sample decode as `(completion_tokens - 1) / post-TTFT time`.
Only scalar timing/token data were retained—no prompts, tags, or generated
text—so speed is independently auditable while semantic quality is not. Its
client also required a literal loopback URL, disabled proxy use and redirects,
and read the bounded nonsymlinked key without logging it.

| Workload | Upstream two-call delta | Fresh aggregate decode | Fresh aggregate output | Fresh median TTFT |
| --- | ---: | ---: | ---: | ---: |
| Math, English eval-style | 36.8 tok/s | **42.739118 tok/s** | 40.583749 tok/s | 0.235057 s |
| Code, English | 25.3 tok/s | **29.616215 tok/s** | 29.359674 tok/s | 0.231122 s |
| Code, German | **27.7 tok/s** | 27.239047 tok/s | 27.025891 tok/s | 0.233141 s |
| Technical explanation, French | 18.7 tok/s | **22.859938 tok/s** | 22.700160 tok/s | 0.236108 s |
| Reasoning, French | 29.2 tok/s | **34.995132 tok/s** | 34.618242 tok/s | 0.240094 s |
| Free prose, English | 16.7 tok/s | **17.886150 tok/s** | 17.795146 tok/s | 0.233771 s |
| Free prose, French | 13.8 tok/s | **14.906475 tok/s** | 14.852522 tok/s | 0.232180 s |
| Free prose, German | 12.2 tok/s | **14.498595 tok/s** | 14.447651 tok/s | 0.232152 s |

Against the rounded upstream values, the fresh aggregate-decode differences
are +16.1%, +17.1%, -1.7%, +22.2%, +19.8%, +7.1%, +8.0%, and +18.8% in table
order. Those are protocol deltas, not same-method speedups. The fresh artifact
does support the content-dependence claim: math was 2.948 times German free
prose, English code was 1.656 times English prose, and French reasoning was
2.348 times French prose.

Across all 24 fresh requests, 14,600 adjusted decode tokens over
684.495327032 post-TTFT seconds yielded **21.329583 aggregate decode tok/s**;
14,624 completion tokens over 690.187974445 E2E seconds yielded 21.188431
aggregate output tok/s. The overall median sample decode was 25.904718 tok/s
and median TTFT was 0.234736 seconds. This overall rate is token/time weighted,
not an equal-weight average of eight workloads. The streaming method also
handled the short math completions (166–176 tokens) directly rather than
depending on a 680-minus-80 delta.

### Startup, Capacity, and Lifecycle Bounds

The successful server reached readiness in 533.763869 seconds (8.896 minutes).
The retained log attributes 298.89 seconds to target verify CUDA-graph capture
and another 53.98 seconds to draft verify capture.
Its first post-readiness request then had 14.428483-second TTFT and
15.074570-second E2E for eight reasoning tokens, versus roughly 0.20–0.24
second steady-state TTFT in most core generation rows. Startup and first
inference therefore must be treated separately; this single first-compile run
does not establish warm restart time.

Startup telemetry retained at least 44.657 GiB host-wide `MemAvailable`. Across
the complete core journal, minimum sampled `MemAvailable` was 42.583 GiB,
sampled temperature peaked at 87 °C, and sampled power peaked at 79.40 W.
Swap-free fell by 161,612 KiB (0.154 GiB) from first to final telemetry sample,
so this run does not support a zero-swap-growth claim. Docker used matching
100g memory and memory-swap limits plus 16g shared memory. These are host-wide
capacity observations and sampled module power, not isolated model footprint
or board-total energy. The two battery artifacts contain no resource telemetry,
and the kept-server summary contains no shutdown measurement.

### DSpark Bounded Conclusions

- The exact pinned target, draft, image, and recipe configuration served
  successfully after the compile-cache permission path was repaired. Strict
  JSON and tool calling worked, and C1–C8 aggregate throughput scaled to
  127.185 tok/s at eight concurrent requests.
- Single-stream SparkBench aggregate was 26.369 tok/s at D256 and 24.043 at
  D1024. Fresh streaming content throughput ranged from 42.739 tok/s for the
  math prompt to 14.499 for German prose, confirming that a single
  "34–38 tok/s" headline does not describe every workload.
- Core decode/concurrency tokens were reasoning-only under this unpatched
  request contract. The 0/3 needle and 0/4 exact-answer results remain real
  semantic failures at the frozen budgets even though several reasoning traces
  contained the answer internally. Neither external speed battery scores
  correctness.
- The upstream two-call artifact is useful recipe-method reproduction, while
  the fresh unique-tag streaming artifact is the stronger cache-resistant
  timing evidence. Their numerical difference must not be interpreted as a
  same-protocol regression or gain.
- This is one GB10, one cold/compile launch, one core run, and three fresh
  samples per content prompt. It does not isolate DSpark from SGLang, NVFP4,
  checkpoint, template, or engine effects and does not establish tail latency
  or general model quality.

## Qwen3.8 Q5 and Matched Quantization Perplexity

### Frozen Q5 Artifact, Runtime, and Evidence

The new serving runs used the cached
[`unsloth/Qwen3.8-27B-GGUF` snapshot](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe),
revision `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`. The exact Q5 file was
`Qwen3.8-27B-UD-Q5_K_XL.gguf`, 20,218,178,624 bytes (18.830 GiB), at
`sha256:176a6a3f034e9cdc447c10cd00329fc9b31002e6589b9295f2ad4f1eefe0f6ab`.
That is 2,294,784,000 bytes (2.137 GiB, 12.803%) larger than the Q4 artifact.

All three serving plans used the same pinned
[`llama.cpp` b10453](https://github.com/ggml-org/llama.cpp/releases/tag/b10453)
source commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`. The 58,085,600-byte
`llama-server` binary was
`sha256:ae1bd49f869ff3397b2a5d757fcf010c6eaaf16c4e3071a15861312defcd4e40`.
It offloaded all layers, enabled flash attention, used 8,192/512
batch/ubatch sizes and Q8_0 KV caches, disabled reasoning and fit heuristics,
and served eight 32,768-token slots from a 262,144-token total context.
The API was offline and loopback-only with its UI and credentialed CORS
disabled. Speculative decoding was not requested.

| Purpose | Exact run directory | Status |
| --- | --- | --- |
| Q5 smoke | `results/20260817T074113Z-qwen38-27b-ud-q5-k-xl-llamacpp-smoke-79f2e221` | Complete |
| Q5 quick | `results/20260817T074157Z-qwen38-27b-ud-q5-k-xl-llamacpp-quick-d61e803a` | Complete |
| Q5 core | `results/20260817T092025Z-qwen38-27b-ud-q5-k-xl-llamacpp-core-285b2215` | `completed` / `partial` |
| Q4 perplexity | `results/20260817T074529603755Z-qwen38-27b-ud-q4-k-xl-llamacpp-perplexity-26edeed3` | Complete |
| Q5 perplexity | `results/20260817T074635246349Z-qwen38-27b-ud-q5-k-xl-llamacpp-perplexity-487a5ffb` | Complete |
| Q8 perplexity | `results/20260817T074750563275Z-qwen38-27b-q8-0-llamacpp-perplexity-144de135` | Complete |
| IQ2 perplexity | `results/20260817T074932213246Z-qwen38-27b-ud-iq2-xxs-llamacpp-perplexity-615e19e7` | Complete |

All three Q5 serving journals end in terminal `completed` run status with no
runtime-failed cases or measurement annotations. Smoke and quick summaries are
complete; core is partial for semantic/context reasons detailed below.
Artifact verification took 9.392946/9.654702/9.748527 seconds and process
startup took 6.049640/6.056980/6.059774 seconds for smoke/quick/core. These are
three sequential cached launches, not cold-start replicates.

### Q5 Smoke and Quick Results

The Q5 smoke run passed every declared serving capability. Vision,
embeddings, and reranking were explicitly unsupported and skipped rather than
attempted failures.

| Case | Requests | Prompt / output tokens | Aggregate output | Median TTFT | Median E2E | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Chat | 1 | 79 / 32 | 8.594589 tok/s | 0.413000 s | 3.708401 s | Pass |
| Strict JSON | 1 | 79 / 14 | 7.677519 tok/s | 0.411178 s | 1.807554 s | Pass |
| Tool call | 1 | 339 / 36 | 7.845970 tok/s | 1.577170 s | 4.573831 s | Pass |

These are one-request protocol checks. Their output counts and prompt shapes
differ, so neither their ordering nor small differences from another smoke
run are a throughput ranking.

The quick journal independently sums to 16/16 passing semantic requests in
the decode, concurrency, and needle cases. Each decode/concurrency request
reached its fixed output budget.

| Case | Requests | Total prompt / output | Aggregate output | Median client decode | Median TTFT | Median E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D128 | 3 | 243 / 384 | 9.096305 tok/s | 9.290270 tok/s | 0.401402 s | 14.056656 s |
| C2 | 4 | 308 / 256 | 16.112933 tok/s | 8.645338 tok/s | 0.637019 s | 7.924216 s |
| C4 | 8 | 624 / 512 | 27.655410 tok/s | 7.707756 tok/s | 1.056330 s | 9.229649 s |
| 8K needle | 1 | 8,283 / 11 | 0.820435 tok/s | — | 12.300016 s | 13.402241 s |

Unlike the serialized Qwen3.6 profiles above, Q5 had eight runtime slots, so
C2 and C4 exercised parallel sequences. C4 aggregate output was 1.716 times
C2, while median per-request decode and latency worsened. There were only two
bursts at each concurrency and no C1 case of the same 64-token shape; this is
not a saturation curve, p95 estimate, or tail-latency result.

| Suite target | Observed prompt tokens/request | Requests | Client-TTFT approximation | Median TTFT | Median E2E |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 324 | 3 | 474.449338 prompt tok/s | 0.682897 s | 0.683059 s |
| 2,048 | 2,117 | 3 | 645.385423 prompt tok/s | 3.280210 s | 3.280373 s |
| 8,192 | 8,259 | 3 | 685.302780 prompt tok/s | 12.051607 s | 12.051786 s |

The observed counts include the request template around each synthetic target.
Each estimate is observed prompt tokens divided by client TTFT with one output
token; it is not a server prompt-evaluation counter and does not isolate
network, scheduling, or first-token decode time. All prefill cases were run
serially. The quick suite has only three decode and three prefill repetitions,
so it supports medians and directional screening rather than variance claims.

Against the same-suite Q4/Q8 anchors already reconciled in the
[2026-08-16 report](benchmark-results-2026-08-16.md#post-hardening-gguf-quantization-and-vision-sweep),
Q5 aggregate D128/C2/C4 was 11.6–12.4% below Q4 and 28.5–29.3% above Q8.
Its three client-TTFT prefill estimates were 6.8–7.1% below Q4 and ranged from
8.3% above to 0.3% below Q8. Those anchors were serialized launches rather
than interleaved replicates, so the result is a useful middle-quant direction,
not a precise causal effect of quantization.

Across the 134 measured-case telemetry samples in Q5 quick, minimum host-wide
`MemAvailable` was 73.227 GiB, sampled power peaked at 90.01 W, and sampled
temperature peaked at 84 °C. These are pressure and module-sensor bounds, not
isolated model allocation, board-total power, or energy-per-token estimates.

### Q5 Core Results

The core journal completed 14 cases without runtime failure or annotations.
It contains 122 measured requests: 20 metric-only prefill requests and 102
requests with semantic/fixed-budget validation. Of those, 97 passed and five
failed; all five failures belong to D1024. One 32K prefill case was skipped by
the declared context guard, while vision, embeddings, and reranking were
profile-declared unsupported. The terminal summary is therefore `partial`,
not an interrupted run.

#### Fixed-Budget Decode and True Eight-Slot Concurrency

Every D256 and C1–C8 request reached 256 output tokens and passed: 80/80
requests and 20,480 output tokens in total.

| Case | Requests | Aggregate output | Median client decode | Median TTFT | Median E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| D256 | 5 | 9.229134 tok/s | 9.324194 tok/s | 0.410830 s | 27.759120 s |
| C1 | 5 | 9.258192 tok/s | 9.368420 tok/s | 0.411309 s | 27.630729 s |
| C2 | 10 | **9.168093 tok/s** | 4.639366 tok/s | 0.833124 s | 55.749886 s |
| C4 | 20 | 29.862739 tok/s | 7.720527 tok/s | 1.255337 s | 34.271414 s |
| C8 | 40 | 45.300536 tok/s | 5.940877 tok/s | 2.436416 s | 45.351250 s |

This profile genuinely had eight llama.cpp slots, unlike the serialized
Qwen3.6 core pair. C4 and C8 aggregate throughput reached 3.226x and 4.893x
C1 respectively, and C8 was 1.517x C4. C2 is the conspicuous exception: it
was 0.973% below C1 while median request decode halved and E2E doubled.
The plateau is not one bad burst—its five burst-level aggregates were
9.183750, 9.204047, 9.170475, 9.215547, and 9.082612 tok/s, versus
9.231264–9.288773 for C1. One run does not identify whether scheduler,
batching, graph, or another runtime behavior caused this non-monotonic shape;
the report preserves it rather than interpolating expected scaling.

#### D1024 Suppression

All five D1024 requests ended with normal `stop` before their fixed 1,024-token
budgets, at **311/621/606/606/594 tokens**. They total **2,738 of 5,120
requested tokens (53.477%)**, for 0/5 validation. Median TTFT and E2E were
0.405639 and 64.972221 seconds. Both aggregate and median-client-decode rates
are suppressed because every request violated the fixed-budget contract. The
retained latency is diagnostic only.

#### Prefill and the Per-Slot Context Limit

Each completed tier had five serial requests and one output token:

| Target | Actual prompt tokens/request | Client-TTFT proxy | Median TTFT | Median E2E |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 196 | 377.033770 prompt tok/s | 0.519847 s | 0.520034 s |
| 1,024 | 1,091 | 585.721551 prompt tok/s | 1.862660 s | 1.862832 s |
| 4,096 | 4,166 | 673.742356 prompt tok/s | 6.183373 s | 6.183557 s |
| 16,384 | 16,455 | 673.848431 prompt tok/s | 24.419438 s | 24.419572 s |

The 32,768 target was not attempted: its conservative estimate was 32,909
tokens against the 32,768-token per-slot cap. Although llama.cpp received a
262,144-token total context, `parallel = 8` divided the served maximum into
eight 32,768-token slots. As elsewhere, these are client-TTFT approximations,
not isolated server prompt-evaluation counters, and five samples per cell do
not establish p95 or arbitrary-prompt behavior.

#### Capability and Quality Screen

| Case | Validation | Total prompt / output | Aggregate output | Median TTFT | Median E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16K needle | 3/3 | 49,436 / 41 | 0.524751 tok/s | 24.486990 s | 25.947659 s |
| Strict JSON | 5/5 | 400 / 70 | 7.650800 tok/s | 0.423138 s | 1.817945 s |
| Tool call | 5/5 | 1,695 / 180 | 8.762551 tok/s | 1.114774 s | 4.093160 s |
| Four exact answers | 4/4 | 479 / 19 | 5.451669 tok/s | 0.461847 s | 0.840239 s |

Arithmetic, logic, instruction-following, and code-reasoning exact answers all
passed. These tiny capability cells validate the frozen fixtures and API
contracts, not general model quality. Their differing prompt/output shapes
also make cross-row throughput ranking invalid. Speculation was not requested,
and the terminal counter snapshot correctly retained zero drafts.

#### Quantization Context and Resource Bounds

Against the same-suite Q4/Q8 core anchors in the
[2026-08-16 report](benchmark-results-2026-08-16.md#post-hardening-gguf-quantization-and-vision-sweep),
Q5 D256/C1/C2/C4/C8 aggregate throughput was 11.1–15.4% below Q4 and
28.8–32.6% above Q8. Its four available prefill proxies were 4.6–7.4% below
Q4; relative to Q8 they ranged from 16.4% above at the shortest tier to 0.5%
below at 16K. The runs were sequential rather than interleaved, and the
published anchors have their own run-order caveats, so this supports a
middle-quant direction rather than precise quantization causality.

Across 1,459 measured-case telemetry samples, minimum host-wide
`MemAvailable` was 72.985 GiB, sampled power peaked at 91.30 W, and sampled
temperature peaked at 85 °C. Measured-case swap-free fell by 12.430 MiB, so
this run does not support a zero-swap-growth claim. These are host-wide and
sampled module-sensor observations, not isolated allocation, board-total
power, or energy-efficiency measurements.

### Matched WikiText-2 Perplexity Screen

All four direct runs used the same Qwen3.8 source revision and tokenizer, the
same 1,290,590-byte
`/home/xlz/.cache/sparkbench/wikitext-2-raw/wiki.test.raw` file at
`sha256:173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`,
and the same 51,551,520-byte `llama-perplexity` binary at
`sha256:31ec19f4d8c071d691f7f4dde4a432771a50872eb29d87e9408a39f366ed5972`.
The runtime source was the same b10453 commit. Each offline command used
`--chunks 64 --ctx-size 512 --n-gpu-layers all --flash-attn on`; all returned
zero and their isolated process groups were reaped without timeout or signals.

The exact model artifacts were:

- Q4 `Qwen3.8-27B-UD-Q4_K_XL.gguf`, 17,923,394,624 bytes (16.692 GiB),
  `sha256:bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`;
- Q5 `Qwen3.8-27B-UD-Q5_K_XL.gguf`, 20,218,178,624 bytes (18.830 GiB),
  `sha256:176a6a3f034e9cdc447c10cd00329fc9b31002e6589b9295f2ad4f1eefe0f6ab`;
- Q8 `Qwen3.8-27B-Q8_0.gguf`, 29,047,086,048 bytes (27.052 GiB),
  `sha256:a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348`;
- IQ2 `Qwen3.8-27B-UD-IQ2_XXS.gguf`, 9,010,048,064 bytes (8.391 GiB),
  `sha256:8d1b37297d6cf98303cd396896f35e01089ddcc904053a9c6997f7a1c35b8524`.

Lower perplexity is better. The `±` values below are llama.cpp's reported
uncertainty terms; the interval is simple estimate-minus/plus-term arithmetic,
not a newly inferred confidence interval.

| Quant | Final PPL | Reported interval | Delta from Q4 | Process wall time | Minimum `MemAvailable` | Peak power / temperature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4 | 6.5315 ± 0.12407 | 6.40743–6.65557 | — | 48.859029 s | 94.102 GiB | 90.23 W / 83 °C |
| Q5 | 6.5275 ± 0.12403 | 6.40347–6.65153 | -0.0040 (-0.061%) | 52.109141 s | 91.952 GiB | 90.64 W / 85 °C |
| Q8 | 6.5205 ± 0.12385 | 6.39665–6.64435 | -0.0110 (-0.168%) | 56.781982 s | 83.552 GiB | 80.83 W / 81 °C |
| IQ2 | 7.3655 ± 0.14230 | 7.22320–7.50780 | +0.8340 (+12.769%) | 50.015229 s | 102.353 GiB | 94.03 W / 84 °C |

The final-estimate lines in all four retained stderr logs match their scalar
results. Q4, Q5, and Q8 span only 0.0110 PPL and their reported intervals
overlap almost completely, so this screen does not resolve a quality ordering
among them. In particular, Q5's 12.803% file-size cost over Q4 did not produce
a resolved WikiText-2 improvement. IQ2 was 12.769% worse than Q4, 12.838%
worse than Q5, and 12.959% worse than Q8; its reported interval does not
overlap any of the other three. That is clear within-screen evidence of
degradation on this matched likelihood task, not a measurement of chat
quality.

This remains one 64-chunk estimate per quantization on one corpus, run
sequentially. The uncertainty terms are large enough to dominate the small
Q4/Q5/Q8 deltas. Process wall time also includes setup and was exposed to run
order, page-cache, and thermal effects, so the wall-time and sampled-resource
columns do not rank inference speed, model footprint, or efficiency.
Perplexity comparisons are meaningful here because model revision, tokenizer,
dataset, runtime, and settings match. They must not be compared numerically
with Qwen3.6, DSpark, or another model/tokenizer, and they do not test strict
JSON, tool use, instruction following, factuality, or reasoning.

## Qwen3.8 Q4 262K Long Context: Baseline Versus MTP5

### Matched Configuration and Terminal Evidence

Both profiles used the exact Q4 artifact and native runtime pinned above:
`Qwen3.8-27B-UD-Q4_K_XL.gguf`, 17,923,394,624 bytes, revision
`f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`, at
`sha256:bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`,
served by llama.cpp b10453 commit
`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70` and the same
`sha256:ae1bd49f869ff3397b2a5d757fcf010c6eaaf16c4e3071a15861312defcd4e40`
server binary. Each had one runtime slot and the full 262,144-token context,
with all layers offloaded, flash attention, 8,192/512 batch/ubatch sizes,
Q8_0 target KV, reasoning off, and the same offline loopback-only API.

MTP5 changed only the native speculative settings: embedded `draft-mtp`, a
maximum five draft tokens, Q8_0 draft KV, and backend sampling. Baseline did
not request speculation.

| Profile | Exact run directory | Terminal status |
| --- | --- | --- |
| Baseline | `results/20260817T075117Z-qwen38-27b-ud-q4-k-xl-llamacpp-long-context-llamacpp-long-context-a6821df4` | Complete |
| MTP5 | `results/20260817T082157Z-qwen38-27b-ud-q4-k-xl-llamacpp-mtp5-long-context-llamacpp-long-context-adbd7ab0` | Complete |

Both summaries are terminal `completed`, with four completed cases, no
failures, skips, context-limit exclusions, invalid measurements, or
annotations. Baseline/MTP5 artifact verification took 8.346309/8.538663
seconds and process startup took 6.052577/6.049325 seconds. These were
sequential cached launches, so those small differences are not cold-start
effects.

### Exact-Key Validation and Actual Prompt Lengths

Each profile passed all ten measured exact-key requests: three repetitions at
the nominal 32K, 64K, and 128K tiers, plus one at the 245,760-token target.
Thus validation was 10/10 per profile and 20/20 combined. All requests ended
normally with 11–14 output tokens rather than exhausting the 32-token budget.

| Suite target | Requests/profile | Baseline actual prompt tokens | MTP5 actual prompt tokens | Baseline / MTP5 validation |
| ---: | ---: | ---: | ---: | ---: |
| 32,768 | 3 | 32,860–32,861 | 32,861–32,862 | 3/3 / 3/3 |
| 65,536 | 3 | 65,629–65,630 | 65,629–65,630 | 3/3 / 3/3 |
| 131,072 | 3 | 131,165–131,167 | 131,166–131,167 | 3/3 / 3/3 |
| 245,760 | 1 | 245,854 | 245,856 | 1/1 / 1/1 |

The actual token counts include the chat template, needle, and per-request
tag around each synthetic target. They differ by at most two tokens between
profiles, and the table reports measured counts rather than treating suite IDs
as token counts.

### TTFT and End-to-End Comparison

Request-level journal recomputation gives the following medians. Deltas are
MTP5 minus baseline, so a positive value is slower.

| Target | Baseline TTFT | MTP5 TTFT | TTFT delta | Baseline E2E | MTP5 E2E | E2E delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 46.938498 s | 48.445709 s | +1.507211 s (+3.211%) | 48.070891 s | 48.810394 s | +0.739502 s (+1.538%) |
| 65,536 | 103.151212 s | 107.430889 s | +4.279677 s (+4.149%) | 104.593153 s | 108.074861 s | +3.481707 s (+3.329%) |
| 131,072 | 246.045093 s | 256.693322 s | +10.648229 s (+4.328%) | 247.363953 s | 257.551576 s | +10.187622 s (+4.118%) |
| 245,760 | 588.263535 s | 618.031116 s | +29.767582 s (+5.060%) | 590.163685 s | 618.824751 s | +28.661066 s (+4.856%) |

MTP5 therefore improved neither TTFT nor E2E at any measured tier. This suite
is dominated by prompt evaluation and generates only a short exact key. The
small, differing output counts also make aggregate output tok/s and inferred
decode-only comparisons misleading. The three lower tiers have only three
requests each, and the 245,760 tier has one; the monotonic direction is useful
for this fixture, but there is no p95, variance, or broad long-context quality
claim.

### MTP5 Activity and Depth Proof

The retained lifetime counter snapshot independently reconciles to 27 drafts,
132 proposed draft tokens, and 119 accepted draft tokens, or **90.151515%**
acceptance. Accepted-token position counts were **26/25/25/24/19** for
zero-based positions 0–4 and sum exactly to 119. The proposal width was
132 / 27 = **4.888889 tokens/draft**; position 4 was exercised, so the deepest
accepted draft depth was five and the configured-depth proof passed. The
reported mean accepted length was **5.407407**, defined here as the mandatory
target token plus 119 / 27 accepted draft tokens. Baseline correctly retained
zero drafts and reported speculation as not requested.

These Prometheus counters cover the complete persisted server lifetime,
including the first post-start request and measured requests. The suite had no
warmups, but the counters still cannot be attributed to a particular context
tier or measured request. High generation-stage acceptance therefore does not
contradict the slower prompt-dominated TTFT and E2E results.

### Capacity and Resource Bounds

At the largest case, the 245,854/245,856-token prompts left 16,290/16,288
input-token slots in the 262,144-token context. Reserving the planned maximum
32 output tokens still left 16,258/16,256 tokens, about 6.20% context headroom,
for baseline/MTP5. This establishes successful operation near 240K input on
this fixture; it does not prove every prompt shape up to the native limit.

Across baseline's 1,729 measured-case telemetry samples, minimum host-wide
`MemAvailable` was 77.115 GiB, sampled power peaked at 91.81 W, and sampled
temperature peaked at 86 °C. Across MTP5's 1,797 samples, the corresponding
bounds were 73.701 GiB, 90.90 W, and 86 °C. MTP5's observed minimum memory
headroom was 3.414 GiB lower, but these are sequential host-wide samples, not
isolated allocations or a causal MTP memory measurement. Sampled power is
module power rather than board-total power, and neither run establishes
energy efficiency.

## Qwen3.6 Matched Core: Baseline Versus MTP2

### Frozen Pair and Terminal Evidence

The core pair returned to the exact Qwen3.6 artifact pinned at the start of
this report: 22,853,663,008-byte
`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, snapshot revision
`5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`, at
`sha256:55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1`.
Both used the same llama.cpp b10453 commit and 58,085,600-byte server binary
at `sha256:ae1bd49f869ff3397b2a5d757fcf010c6eaaf16c4e3071a15861312defcd4e40`.
The profiles matched on one 262,144-token slot, all-layer offload, flash
attention, 8,192/512 batch/ubatch sizes, Q8_0 target KV, reasoning disabled,
and the offline loopback-only API. MTP2 added only native `draft-mtp`, maximum
draft depth two, Q8_0 draft KV, and backend sampling.

| Profile | Exact run directory | Run / summary status |
| --- | --- | --- |
| Baseline | `results/20260817T085410Z-qwen36-35b-a3b-ud-q4-k-xl-llamacpp-core-ff006c90` | `completed` / `partial` |
| MTP2 | `results/20260817T090707Z-qwen36-35b-a3b-ud-q4-k-xl-llamacpp-mtp2-core-90d8fae5` | `completed` / `partial` |

Both journals reached terminal `run_complete` with 15 completed cases, no
runtime failures, invalid measurements, or annotations. Each had 127 measured
requests: 25 metric-only prefill requests and 102 requests with validation.
Baseline recorded 101 passes and one failure; MTP2 recorded 100 passes and two
failures. Vision, embeddings, and reranking were declared unsupported and
skipped in both profiles. The partial summaries are therefore semantic, not
process-lifecycle failures.

Baseline/MTP2 artifact verification took 10.718040/10.733986 seconds and
process startup took 6.048634/6.053120 seconds. These were sequential cached
launches and do not establish cold-start variance.

### Valid Fixed-Budget Decode and Serialized Queueing

Every request in D256 and C1–C8 reached its 256-token budget and passed:
80/80 requests and 20,480 output tokens per profile. Journal-derived aggregate
throughput and per-request client decode were:

| Case | Requests | Baseline aggregate | MTP2 aggregate | MTP2 change | Baseline / MTP2 median client decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| D256 | 5 | 49.253005 tok/s | 75.714966 tok/s | +53.727% | 50.923116 / 80.635517 tok/s |
| C1 | 5 | 48.293743 tok/s | 75.552703 tok/s | +56.444% | 49.955665 / 80.023432 tok/s |
| C2 | 10 | 49.179305 tok/s | 74.723039 tok/s | +51.940% | 50.827320 / 78.221365 tok/s |
| C4 | 20 | 49.254241 tok/s | 73.852054 tok/s | +49.940% | 50.695232 / 78.660910 tok/s |
| C8 | 40 | 49.214338 tok/s | 74.959078 tok/s | +52.311% | 50.667002 / 79.018539 tok/s |

| Case | Baseline / MTP2 median TTFT | Baseline / MTP2 median E2E |
| --- | ---: | ---: |
| D256 | 0.170668 / 0.186364 s | 5.177107 / 3.352188 s |
| C1 | 0.182795 / 0.203081 s | 5.290100 / 3.384636 s |
| C2 | 2.743488 / 1.879795 s | 7.786313 / 5.013980 s |
| C4 | 7.971434 / 5.423666 s | 12.981083 / 8.756496 s |
| C8 | 18.348184 / 11.887587 s | 23.370816 / 15.146393 s |

The prominent result is the D256 increase from 49.253005 to 75.714966
aggregate tok/s. However, both servers had `parallel = 1`: C2, C4, and C8
requests queued behind one sequence slot. Aggregate output stayed within a
1.99% band for baseline and a 2.30% band for MTP2 from C1 through C8, while
median TTFT/E2E grew with queue depth. These rows demonstrate stable
serialized service and MTP decode acceleration, not multi-sequence scaling.

### D1024 Is Suppressed in Both Profiles

Neither D1024 case is a valid fixed-budget rate. One of five requests in each
profile ended semantically early while the other four reached 1,024 tokens:

| Profile | Requested total | Observed total | Early-stop request | Validation | Median TTFT / E2E | Ranked rate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 5,120 | 4,691 | 595 tokens | 4/5 | 0.173085 / 20.270312 s | Suppressed |
| MTP2 | 5,120 | 4,690 | 594 tokens | 4/5 | 0.194865 / 13.115646 s | Suppressed |

The totals are only 91.621%/91.602% of the requested budgets. Dividing those
unequal totals by case wall time would yield attractive-looking raw numbers,
but the report deliberately does not rank them or publish a median decode rate.
The retained TTFT/E2E medians are diagnostic latency only and do not repair
the failed fixed-budget contract.

### Client-TTFT Prefill Proxies

Each prefill tier had five serial requests and one output token. The actual
prompt counts include the template around the synthetic target.

| Target | Actual prompt tokens/request, baseline / MTP2 | Baseline proxy | MTP2 proxy | MTP2 change | Baseline / MTP2 median TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 197 / 196 | 927.686835 tok/s | 869.140607 tok/s | -6.311% | 0.212356 / 0.225510 s |
| 1,024 | 1,093 / 1,093 | 1,734.726266 tok/s | 1,670.419698 tok/s | -3.707% | 0.630071 / 0.654327 s |
| 4,096 | 4,163 / 4,164 | 2,329.639406 tok/s | 2,238.997338 tok/s | -3.891% | 1.786972 / 1.859761 s |
| 16,384 | 16,454 / 16,455 | 2,494.503457 tok/s | 2,417.864450 tok/s | -3.072% | 6.596102 / 6.805592 s |
| 32,768 | 32,836 / 32,838 | 2,496.204040 tok/s | 2,394.666615 tok/s | -4.068% | 13.154373 / 13.712974 s |

MTP2 was lower at every prefill tier. These are prompt tokens divided by
client time to first token, not isolated llama.cpp prompt-evaluation counters;
they include request, scheduling, and first-token work. Five serial samples
per cell support a matched directional result, not tail latency or a claim
about arbitrary prompt structure.

### Capability and Exact-Answer Checks

| Case | Baseline / MTP2 validation | Prompt / output totals, baseline / MTP2 | Baseline / MTP2 aggregate output | Baseline / MTP2 median TTFT | Baseline / MTP2 median E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16K needle | 3/3 / 3/3 | 49,435 / 37; 49,434 / 36 | 1.718269 / 1.653424 tok/s | 6.938729 / 7.099183 s | 7.136827 / 7.247406 s |
| Strict JSON | 5/5 / 5/5 | 405 / 100; 400 / 100 | 35.651547 / 46.500530 tok/s | 0.169027 / 0.186191 s | 0.548343 / 0.417165 s |
| Tool call | 5/5 / 5/5 | 1,700 / 180; 1,695 / 180 | 42.090908 / 63.142649 tok/s | 0.297799 / 0.266663 s | 0.841800 / 0.557739 s |
| Four exact answers | 4/4 / **3/4** | 467 / 19; 479 / 20 | 16.289526 / 17.246410 tok/s | 0.197455 / 0.220320 s | 0.271653 / 0.272527 s |

Both profiles passed JSON, tools, and the 16K exact-key needle completely.
Baseline passed arithmetic, logic, instruction-following, and code-reasoning
exact answers. MTP2 passed the first three but returned extracted answer `10`
instead of expected `9` for the code-reasoning item. The exact-answer aggregate
is retained as raw timing but is not a quality-adjusted performance comparison.
The core miss is real in that journal, but the standalone repeat below did not
reproduce it. It is therefore not stable across the two matched MTP2 launches
and cannot establish a general MTP quality regression. Likewise, differing
short completion counts make the needle aggregate rates unsuitable for a
speedup claim.

### Standalone Exact-Answer Repeat

Two terminal standalone launches reran the same four fixed, temperature-zero
items and corresponding case identities as the core profiles:

| Profile | Exact run directory | Status |
| --- | --- | --- |
| Baseline | `results/20260817T091737Z-qwen36-35b-a3b-ud-q4-k-xl-llamacpp-chat-quality-6ec4205f` | Complete |
| MTP2 | `results/20260817T091817Z-qwen36-35b-a3b-ud-q4-k-xl-llamacpp-mtp2-chat-quality-38ea794e` | Complete |

Both used the same pinned artifact, runtime, one-slot serving geometry, and
profile-specific arguments as their core counterparts. Both summaries are
terminal `complete`, with no failures, annotations, or invalid measurements.
Request-level reconciliation was:

| Profile | Arithmetic / logic / instruction / code | Total prompt / output | Aggregate output | Median TTFT | Median E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | Pass / pass / pass / pass | 467 / 19 | 17.893823 tok/s | 0.185037 s | 0.257893 s |
| MTP2 | Pass / pass / pass / pass | 479 / 19 | 17.931735 tok/s | 0.200693 s | 0.252402 s |

Baseline was therefore 4/4 in core and 4/4 standalone, or 8/8 across two
exposures. MTP2 was 3/4 in core and 4/4 standalone, or 7/8: its code item
changed from extracted `10` in core to the expected `9` in the repeat. The
matched repeat corrects the interpretation—the miss was not deterministic or
stable across launches. Eight tiny exact-answer exposures still establish
neither broad quality equivalence nor a general regression. The nearly equal
raw aggregate timings are also not a meaningful quality-adjusted speed
benchmark.

The repeat's MTP lifetime snapshot recorded 9 drafts, 18 proposed draft
tokens, and 16 accepted tokens, or **88.888889%** acceptance, with position
counts **9/7**. Average proposal width was exactly two, position one was
accepted, and the depth-two proof passed. The reported mean accepted length
was 2.777778 including the mandatory target token. Those counters include the
prime request and all four measured items, so they cannot show whether
speculation affected the repeated code answer. Baseline again recorded zero
drafts and speculation not requested.

### Core MTP2 Activity and Depth Proof

The final cumulative snapshot reconciles to **11,137 drafts**, **22,221
proposed draft tokens**, and **17,048 accepted draft tokens**: **76.720220%**
draft-token acceptance. Accepted position counts were **9,488/7,560** and sum
exactly to 17,048. Average proposal width was 22,221 / 11,137 = **1.995241**;
position one was accepted, establishing deepest draft depth two and passing
the configured-depth proof. The reported mean accepted length was
**2.530753**, the mandatory target token plus 17,048 / 11,137 accepted draft
tokens. Baseline correctly reported zero drafts and speculation not requested.

Counters cover the complete persisted server lifetime, including the prime
request, warmups, and all measured requests. They prove MTP2 execution and
depth but cannot be assigned to D256, a concurrency row, or any individual
quality/capability request.

### Core Telemetry and Bounded Takeaway

Across baseline's 629 measured-case samples, minimum host-wide
`MemAvailable` was 79.094 GiB, sampled power peaked at 82.53 W, and sampled
temperature peaked at 85 °C. Across MTP2's 458 samples, the corresponding
bounds were 76.243 GiB, 82.88 W, and 85 °C. MTP2's minimum observed memory
headroom was 2.851 GiB lower, but the profiles were sequential and these are
host-wide readings rather than isolated model allocations. The different
sample counts mostly reflect different case durations.

Measured-case swap-free rose by 2.613 MiB in baseline and fell by 21.480 MiB
in MTP2. The latter is small relative to total swap but means this core run
does not support a zero-swap-growth claim. Power is sampled module power, not
board-total power; these observations do not establish energy efficiency.

Within the valid fixed-budget rows, MTP2 delivered roughly 50–56% higher
aggregate decode throughput while its client-prefill proxies were 3–6% lower.
Both D1024 rows remain unranked, the apparent C1–C8 plateau is serialized
single-slot service rather than scaling. The standalone quality repeat shows
that the MTP2 code-reasoning miss was not stable, while the combined 7/8 MTP2
and 8/8 baseline exact-answer results remain far too small for a blanket claim
of equivalence or regression.

## Muse-Glimmer 30B Q4: Baseline Versus DFlash15 Smoke and Quick

This is deliberately bounded smoke and quick evidence, not a core benchmark.
All four persisted journals reached terminal `completed` lifecycle state:

| Profile / suite | Exact run directory | Terminal case evidence | Summary status |
| --- | --- | --- | --- |
| Baseline smoke | `results/20260817T100237Z-muse-glimmer-30b-ud-q4-k-xl-llamacpp-smoke-2659185a` | 3/3 completed; 1/3 validation | `partial` |
| DFlash15 smoke | `results/20260817T100524Z-muse-glimmer-30b-ud-q4-k-xl-llamacpp-dflash15-smoke-9e931867` | 3/3 completed; 1/3 validation | `partial` |
| Baseline quick | `results/20260817T100612Z-muse-glimmer-30b-ud-q4-k-xl-llamacpp-quick-efc4a720` | 4/7 completed; 3 request failures; 3/4 validation | `partial` |
| DFlash15 quick | `results/20260817T100823Z-muse-glimmer-30b-ud-q4-k-xl-llamacpp-dflash15-quick-1edc5e0e` | 4/7 completed; 3 request failures; 3/4 validation | `partial` |

The smoke runs had no request/runtime failure; their `partial` labels come
from the failed JSON and tool validators. Vision, embeddings, and reranking
were declared unsupported and skipped. Each quick run completed D128, C2, C4,
and its 8K needle, failed all three prefill paths at request time, and failed
the needle validator. None of the four runs recorded an invalid measurement,
context-limit event, or measurement annotation.

### Frozen Artifacts and Independent Lifecycles

Both profiles loaded the same cached
[`unsloth/Muse-Glimmer-30B-GGUF` revision](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF/tree/faa5b025c584459c13febfa5c59883516710ae39),
`faa5b025c584459c13febfa5c59883516710ae39`. The exact main file was
`Muse-Glimmer-30B-UD-Q4_K_XL.gguf`, 15,878,222,368 bytes (14.788 GiB), at
`sha256:82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38`.
The DFlash profile additionally loaded
[`meta-models/Muse-Glimmer-30B-GGUF` revision](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/tree/43c7eadd41352a299ea8e0a36b3157978dd63596),
`43c7eadd41352a299ea8e0a36b3157978dd63596`: the
1,631,208,128-byte (1.519 GiB) `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` sidecar at
`sha256:b2e808bf656086fe86bd0d0bd990f01d33e377537a07c02d45371517c8b264ef`.

Both used the pinned llama.cpp b10453 binary and commit documented above,
including the identical binary digest. They offloaded all layers, enabled
flash attention, used 8,192/512 batch/ubatch sizes and Q8_0 KV caches, disabled
fit heuristics and reasoning, exposed only loopback in offline mode, and
allocated four 32,768-token slots within a 131,072-token total context.
DFlash added only its pinned sidecar plus `draft-dflash` and maximum draft
length 15.

Each run used a fresh, independent subprocess lifetime and was stopped after
its one cumulative metrics snapshot. For smoke, artifact validation took
7.356672/8.322857 seconds and subsequent server readiness took
4.056930/4.050525 seconds for baseline/DFlash. For quick, the corresponding
pairs were 7.527491/8.164044 seconds and 4.048477/4.055724 seconds. These are
cached starts with one launch per profile and suite, not cold-start
measurements.

### Exact Smoke Results

Each row is one request with no warmup. Aggregate output is independently
recomputed as completion tokens divided by case wall time; the TTFT and E2E
values are therefore single observations despite the median labels in the
summaries.

| Case | Prompt / completion tokens, baseline / DFlash | Baseline / DFlash case wall | Baseline / DFlash aggregate output | Baseline / DFlash TTFT | Baseline / DFlash E2E | Validation, baseline / DFlash |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Chat | 107 / 32; 108 / 32 | 2.891056 / 1.174425 s | 11.068618 / **27.247367 tok/s** | 0.549887 / 0.907044 s | 2.874048 / 1.160158 s | Pass / pass, protocol only |
| Strict JSON | 104 / 64; 106 / 64 | 5.539480 / 0.921950 s | 11.553432 / 69.418075 tok/s | 0.538761 / 0.430009 s | 5.526208 / 0.907778 s | Fail / fail |
| Tool call | 446 / 64; 446 / 64 | 5.937983 / 1.834526 s | 10.778071 / 34.886395 tok/s | 0.931801 / 1.340391 s | 5.924383 / 1.821078 s | Fail / fail |

For the only decode-kind row, DFlash raised primary chat aggregate output from
11.068618 to 27.247367 tok/s, **2.461677x or +146.167741%**. Its secondary
client decode estimate rose from 13.338146 to 122.474635 tok/s, **9.182283x**,
while TTFT increased by 0.357157 seconds (**+64.950958%**) and E2E fell by
1.713891 seconds (**-59.633329%**). This is one 32-token request, and the
prompt differed by one token; it is an execution signal, not a stable speed
estimate. The larger raw JSON/tool rate changes are not ranked because both
outputs failed their contracts.

### Matched Quick Fixed-Token Generation

D128 had three measured serial requests after one warmup. C2 and C4 each had
two measured bursts, producing four and eight requests respectively. All 15
decode/concurrency requests per profile reached their exact 128- or 64-token
budgets, so the aggregate arithmetic itself is valid:

| Case | Requests / output tokens per profile | Prompt totals, baseline / DFlash | Case wall, baseline / DFlash | Aggregate output, baseline / DFlash | DFlash change | Median client decode, baseline / DFlash | Median TTFT, baseline / DFlash | Median E2E, baseline / DFlash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D128 | 3 / 384 | 327 / 321 | 32.800277 / 6.480036 s | 11.707218 / **59.258934 tok/s** | **+406.174328%** | 12.241592 / 76.029160 tok/s | 0.556388 / 0.435489 s | 10.936115 / 2.103200 s |
| C2 | 4 / 256 | 424 / 424 | 11.527458 / 4.098326 s | 22.207845 / **62.464521 tok/s** | **+181.272319%** | 12.270768 / 88.223495 tok/s | 0.611089 / 0.867840 s | 5.745170 / 1.972563 s |
| C4 | 8 / 512 | 864 / 840 | 11.997674 / 3.494318 s | 42.674938 / **146.523580 tok/s** | **+243.348076%** | 12.009692 / 81.645995 tok/s | 0.732553 / 0.943276 s | 5.978249 / 1.718195 s |

DFlash's aggregate ratios were **5.061743x/2.812723x/3.433481x** at
D128/C2/C4. Median E2E fell 80.768309%/65.665722%/71.259232%, while TTFT
changed -21.729280%/+42.015406%/+28.765555%. The four-slot runtime made C2 and
C4 true simultaneous-slot bursts rather than a one-slot queue. C4 aggregate
was 1.921615x C2 for baseline and 2.345709x for DFlash, but two bursts per cell
are too few for tail latency or robust scaling claims; DFlash's two C2 burst
times alone ranged from 1.124182 to 2.937242 seconds.

These fixed-token rates quantify **accelerated prompt echo, not usable answer
throughput**. Every one of the 30 decode/concurrency responses had empty
`content`, nonempty `reasoning` beginning with its benchmark nonce, and
`finish_reason = length`. Their generic decode validators checked completed
generation rather than answer semantics. The matched output budgets make the
token-rate comparison arithmetically sound, but do not repair the served
content contract.

### Failed Prefill Paths and 8K Needle

All six planned prefill cases failed with the same
`BenchmarkRequestError`: `Streaming response did not emit content or
reasoning`.

| Target | Baseline failure elapsed | DFlash failure elapsed | Measured repetitions persisted |
| ---: | ---: | ---: | ---: |
| 256 | 0.519685 s | 0.870473 s | 0 / 0 |
| 2,048 | 2.480955 s | 3.441958 s | 0 / 0 |
| 8,192 | 9.081540 s | 9.794512 s | 0 / 0 |

Each case configured one warmup followed by three measured repetitions. No
`request_complete` event was journaled, so each path failed before a measured
sample existed. The elapsed failure times are diagnostics, not TTFT or prefill
throughput, and no prefill rate can be recovered from them.

Both one-request 8K exact-key cases also failed semantically. Baseline used an
8,310-token prompt, emitted 32 tokens, and recorded 9.262660-second TTFT and
11.683715-second E2E; DFlash used 8,311/32 tokens and recorded
9.874888/11.831346 seconds. DFlash was 6.609634% slower to first token and
1.263561% slower end to end. Both visible contents were empty, both reasoning
fields contained only a truncated nonce echo, both ended at `length`, and
neither returned the needle. Their raw 2.734817/2.700958 tok/s aggregates are
failed-capability diagnostics, not ranked long-context throughput.

### Reasoning-Only Prompt Echo and the Core Stop

All six smoke responses ended at the configured `length` limit. Every
visible `content` field was empty; all 32 or 64 completion tokens were instead
returned in `reasoning`, despite the common `--reasoning off` launch argument.
Each trace began by echoing its unique benchmark nonce and instruction. The
baseline reasoning strings were 76/211/207 characters for chat/JSON/tools;
DFlash produced 73/206/207. Thus this is not a difference in useful answer
content.

The permissive chat smoke validator passed transport and token generation even
though neither profile emitted a visible assistant answer. Both JSON traces
ran out of budget while repeating the JSON instruction and produced no
parsable object. Both tool traces recognized that `multiply` should be used
but emitted no tool call; the persisted `tool_calls` arrays were empty. These
are semantic output-contract failures, not merely formatting differences.

The quick journals confirmed rather than repaired the smoke failure: all 32
completed quick responses across both profiles were reasoning-only nonce
echoes, the six prefill paths emitted nothing journalable, and both needles
failed. A core run would therefore benchmark accelerated, truncated prompt
echo rather than a usable served contract. No Muse-Glimmer core result is
warranted from this configuration. The prompt/template/reasoning-output
contract must first pass a bounded semantic retest; only then would a matched
core comparison be interpretable.

### Positive DFlash Evidence and Resource Bounds

The terminal DFlash lifetime snapshot recorded **14 drafts, 174 proposed draft
tokens, and 150 accepted draft tokens**, or **86.206897%** acceptance. Accepted
counts by zero-based draft position were
**14/14/14/14/13/12/11/11/10/9/8/6/6/4/4**, summing exactly to 150; positive
position-14 acceptance shows that the configured fifteenth draft position was
actually exercised. Average proposal width was **12.428571** tokens. The
reported mean accepted length was **11.714286**, which includes the mandatory
target token in addition to 150 / 14 accepted draft tokens. Baseline requested
no speculation and recorded zero drafts or draft tokens in both lifetimes.

The quick DFlash snapshot separately recorded **155 drafts, 1,992 proposed
draft tokens, and 1,271 accepted draft tokens**, or **63.805221%** acceptance.
Its position counts were
**134/114/109/105/99/93/89/87/83/72/66/62/59/50/49**, summing exactly to
1,271 and again reaching position 14. Average proposal width was **12.851613**;
reported mean accepted length was **9.200000**, the mandatory target token
plus 1,271 / 155 accepted draft tokens.

Each counter set covers its complete one-server lifetime, including the prime
request, warmups, failed paths, and measured requests. They prove DFlash
execution but cannot be assigned to an individual case, and they do not make
the invalid semantic outputs useful.

Across measured cases, baseline's 14 telemetry samples retained at least
96.777946 GiB host-wide `MemAvailable` and observed peaks of 53.15 W and
56 °C. DFlash retained at least 91.564610 GiB and observed peaks of 46.03 W
and 51 °C, but it had only one sample per case. The sidecar profile therefore
fit with substantial headroom, while the unequal, extremely sparse sampling
cannot establish its incremental allocation, power, thermal, or energy cost.

Across the four completed quick cases, baseline's 65 telemetry samples
retained at least 96.185677 GiB host-wide `MemAvailable`, with sampled peaks of
90.29 W and 80 °C. DFlash's 26 samples retained at least 90.133774 GiB, with
sampled peaks of 90.07 W and 78 °C. The quicker DFlash cases naturally
produced fewer one-second samples. Both profiles fit and shut down cleanly;
sequential host-wide memory readings and unequal sample counts cannot isolate
sidecar memory, power, thermal, or energy deltas.
