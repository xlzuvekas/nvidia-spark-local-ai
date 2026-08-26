# Qwen3.6 versus Qwen3.8 cache-off long-context throughput — 2026-08-25

## Result

On one NVIDIA DGX Spark / GB10, the Qwen3.6 35B-A3B NVFP4+MTP3 MoE was faster
than dense Qwen3.8 27B NVFP4+MTP3 in every matched valid generation cell. At
short-prompt offered C64 it delivered 893.674 aggregate output tok/s versus
359.081 tok/s, a 2.49x advantage. In the longest matched valid generation cell,
approximately 246K repeated words at offered C2, Qwen3.6 delivered 1.951
aggregate tok/s versus 0.728 tok/s, with median TTFT of 195.222 seconds versus
517.665 seconds.

Qwen3.8 finalized and validated all 25 planned cases. Qwen3.6 finalized all 25,
but only 22 passed validation: one response stopped early in each of the
8K/C32 and 30K/C32 generation cells, and two of three responses stopped early
in the 246K/C1 generation cell. The reporter correctly suppresses aggregate
output and median decode rates for those three cells. Both models passed every
key-presence retrieval probe: 39/39 requests across seven cases per model,
including approximately 246K at offered C2 and 30K at offered C32.

These are exploratory dirty-worktree runs and cache-off baselines, not a
prefix-cache savings comparison. They measure two deployment artifacts and
their distinct runtimes, architectures and serving geometries; the speed
difference cannot be assigned to MoE alone.

## Short-prompt saturation

Each row contains three synchronized measured bursts. `TPS` is aggregate
completion tokens divided by full measured case wall; `stream` is the median
client estimate after the first SSE emission and is secondary because all
cells bundled tokens in stream events. TTFT includes queueing.

| Offered C | Qwen3.6 TPS / stream | Qwen3.6 wall | Qwen3.6 TTFT | Qwen3.8 TPS / stream | Qwen3.8 wall | Qwen3.8 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 96.906 / 100.945 | 7.925 s | 0.104 s | 17.184 / 17.544 | 44.692 s | 0.381 s |
| 8 | 357.754 / 52.258 | 17.174 s | 0.287 s | 111.947 / 15.600 | 54.883 s | 0.799 s |
| 16 | 552.975 / 38.926 | 22.222 s | 0.459 s | 183.482 / 12.722 | 66.971 s | 1.131 s |
| 32 | 753.886 / 26.294 | 32.599 s | 0.694 s | 272.494 / 9.501 | 90.189 s | 1.934 s |
| 64 | **893.674 / 15.521** | 55.000 s | 1.258 s | **359.081 / 6.309** | 136.883 s | 3.666 s |

Both servers were sampled with 64 running requests in the C64 short-prompt
cell. Aggregate throughput continued to rise through C64, while median
per-request stream rate fell by about 6.5x for Qwen3.6 and 2.8x for Qwen3.8
from C1. C64 is therefore a batch-throughput point, not the best interactive
latency point.

## Long-input generation

The target column is a repeated-word count, not an exact token count. Each
valid row again contains three measured bursts and exactly 256 completion
tokens per request. Aggregate TPS includes prefill, queueing and journal
overhead; it is not pure decode TPS.

| Target / offered C | Qwen3.6 TPS / stream | Qwen3.6 wall | Qwen3.6 TTFT | Qwen3.8 TPS / stream | Qwen3.8 wall | Qwen3.8 TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K / C1 | 65.507 / 99.555 | 11.724 s | 1.341 s | 15.420 / 22.364 | 49.804 s | 4.756 s |
| 8K / C8 | 128.989 / 33.859 | 47.632 s | 8.195 s | 40.047 / 11.832 | 153.421 s | 28.158 s |
| 8K / C32 | invalid | 158.904 s | 23.623 s | 48.415 / 3.051 | 507.611 s | 81.118 s |
| 8K / C64 | 160.624 / 4.727 | 306.007 s | 44.729 s | 50.192 / 1.576 | 979.283 s | 151.949 s |
| 30K / C1 | 31.531 / 97.599 | 24.357 s | 5.662 s | 8.235 / 21.501 | 93.262 s | 19.218 s |
| 30K / C8 | 39.390 / 11.031 | 155.980 s | 27.907 s | 11.929 / 3.529 | 515.055 s | 94.248 s |
| 30K / C32 | invalid | 595.268 s | 99.105 s | 13.301 / 0.876 | 1,847.655 s | 311.353 s |
| 61K / C1 | 15.211 / 121.882 | 50.491 s | 14.754 s | 4.602 / 20.251 | 166.898 s | 42.883 s |
| 61K / C2 | 15.744 / 50.819 | 97.560 s | 23.131 s | 5.211 / 12.810 | 294.762 s | 66.222 s |
| 123K / C1 | 5.793 / 80.164 | 132.566 s | 41.164 s | 1.975 / 18.265 | 388.834 s | 115.269 s |
| 123K / C2 | 5.947 / 33.856 | 258.290 s | 62.921 s | 2.120 / 10.643 | 724.666 s | 173.479 s |
| 246K / C1 | invalid | 393.533 s | 129.313 s | 0.718 / 17.145 | 1,070.333 s | 341.549 s |
| 246K / C2 | **1.951 / 33.084** | 787.180 s | 195.222 s | **0.728 / 8.552** | 2,110.517 s | 517.665 s |

Qwen3.8's C1 post-first-emission stream estimate stayed between 17.145 and
22.364 tok/s from the short cell through the 246K target. Its apparent
slowdown in aggregate output rate is predominantly the growing prefill wall:
median TTFT rose from 0.381 seconds short to 341.549 seconds at 246K. Qwen3.6
showed the same shape at a much faster level, with valid C1 stream estimates of
80.164–121.882 tok/s from 61K through 123K.

Additional long-context concurrency bought little aggregate throughput. For
Qwen3.8 at 246K, C2 improved aggregate rate by only 1.4% over C1 while raising
median TTFT by 51.6% and halving the per-request stream rate. At 30K, its C32
cell improved aggregate rate by 11.5% over C8 while raising median TTFT by
3.30x. Sampled live logs showed roughly 20–22 Qwen3.8 requests running at once
in that offered-C32 cell; this is observational occupancy, not an exact
summary metric. For Qwen3.8, C1 is the practical 246K geometry and C8 is the
more useful 30K latency/throughput tradeoff.

The invalid Qwen3.6 rows are not zero-throughput results. At 8K/C32, 95/96
responses reached 256 tokens and one stopped at 143. At 30K/C32, 95/96 reached
256 and one stopped at 111. At 246K/C1, one response reached 256 and two
stopped at 84 and 94. Because a single short response invalidates the full
cell, their raw diagnostic rates are deliberately excluded from the table.

## Retrieval and runtime stability

The retrieval probes require the nonce-derived key to appear in the response;
they do not require an exact key-only response and are not general
long-document comprehension tests. Both models passed all conditions. At the
246K/C2 probe, Qwen3.6 recorded 195.103-second median TTFT and 258.299-second
case wall, versus 512.897-second median TTFT and 683.142-second wall for
Qwen3.8. At 30K/C32, the corresponding observations were 97.632/185.984
seconds and 308.922/588.280 seconds.

Both saturation runs exposed a retry risk. In the first Qwen3.8 server
lifetime, the 30K/C32 generation case failed and the following retrieval case
lost its connection. In the first Qwen3.6 lifetime, the server failed at short
C16 and the remaining 12 cases could not run. Live server logs showed a CUDA
illegal-memory-access failure in each first lifetime. Resuming the same frozen
plans on fresh server lifetimes completed every previously failed case; the
final combined Qwen3.8 saturation summary was 15/15 valid, and the Qwen3.6
saturation summary had no failed cases but two generation-validation failures.
The separate Qwen3.6 native-context run completed without a request failure and
had the 246K/C1 validation failure described above. The two successful resumes
argue against treating the crashes as deterministic admission boundaries, but
they do not establish production stability; first-use kernel or other runtime
state is only a possible cause, not an established diagnosis.

## Configuration and measurement boundary

All four profiles used FP8 KV cache, chunked prefill, MTP depth three,
temperature zero and explicit `--no-enable-prefix-caching`. The 32K profiles
used `--max-num-seqs 64`, an 8,192-token batch ceiling and 50%/80% GPU-memory
utilization for Qwen3.6/Qwen3.8 respectively. Native-context profiles used
`--max-num-seqs 2`; Qwen3.6 used 40% GPU-memory utilization and an 8,192-token
ceiling, while Qwen3.8 used 52% and a 4,096-token ceiling.

The model pins were:

- `nvidia/Qwen3.6-35B-A3B-NVFP4@491c2f1ea524c639598bf8fa787a93fed5a6fbce`,
  using `nvcr.io/nvidia/vllm:26.07-py3` at image digest
  `sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268`;
- `Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462`,
  using the pinned Qwen3.8 vLLM image at digest
  `sha256:4a2f33a884222f7049b983263ad9976f89452bb81affecf5b67d89ad35c1bc31`.

Periodic live server logs reported a 0.0% prefix-cache hit rate, and startup
configuration showed caching disabled. Request-scoped cache counters were
unavailable, so these runs are profile-and-log cache-off, not counter-verified.
They do not measure how much caching saves. A matched cache experiment needs
an intentionally shared prefix, a unique suffix after that prefix, identical
arrival schedules and separate cold/warm accounting; the nonce-first prompts
used here deliberately do not provide that comparison.

Native-context run-lifetime MTP counters were active. Qwen3.6 reported 81.26%
draft-token acceptance and 3.438 mean acceptance length including the verifier
bonus token; Qwen3.8 reported 86.24% and 3.587. These cumulative counters
include priming and all measured cases and are not per-cell causal estimates.

All six managed server lifecycles completed teardown successfully with zero
cleanup failures, and case telemetry summaries are present for all 50 cases.
Final handoff checks found no remaining benchmark container or GPU compute
process.

Each schema-2 plan fingerprint and integrity hash bind its frozen model, full
suite geometry and resolved image. The host snapshot also recorded repository
HEAD `e8a392def52c312cfb0b25e97aab55c2b99e3357` plus a dirty status, but not a
digest of the working patch. A later commit cannot turn these into exact
clean-revision measurements, so every number in this note remains exploratory.
Raw prompts, completions, reasoning, request identifiers, telemetry streams
and local run locations remain untracked and are intentionally excluded here.

## Publication boundary

The complete publication set preserves all six attempt directories rather than
selecting only the four terminal measured runs:

- [Qwen3.6 initial plan](../evidence/runs/20260825T053811Z-qwen36-35b-a3b-nvfp4-mtp3-long-tps-long-context-tps-b2a243cf/manifest.json);
- [Qwen3.8 aborted initial lifecycle](../evidence/runs/20260825T053811Z-qwen38-27b-nvfp4-mtp3-long-tps-long-context-tps-6fbfc156/manifest.json);
- [Qwen3.6 native-context run](../evidence/runs/20260825T054429Z-qwen36-35b-a3b-nvfp4-mtp3-long-tps-long-context-tps-1f2a29c4/manifest.json);
- [Qwen3.6 C64 saturation run](../evidence/runs/20260825T054429Z-qwen36-35b-a3b-nvfp4-mtp3-tps64-throughput-saturation-88e3e7cd/manifest.json);
- [Qwen3.8 native-context run](../evidence/runs/20260825T054429Z-qwen38-27b-nvfp4-mtp3-long-tps-long-context-tps-a1931174/manifest.json); and
- [Qwen3.8 C64 saturation run](../evidence/runs/20260825T054429Z-qwen38-27b-nvfp4-mtp3-tps64-throughput-saturation-9d51810a/manifest.json).

The exporter operates over the complete raw-results topology. The two exact
private Harbor lifecycle records required to preserve the historical Harbor
campaign are available locally and supplied explicitly to a full refresh; they
remain outside Git. Hand-selecting or hand-merging only the successful runs
would still violate the evidence protocol. The published projections exclude
captured content and local details, and publication does not upgrade these
dirty-worktree measurements into clean-revision evidence. A clean-revision
rerun remains necessary for that stronger provenance claim.
