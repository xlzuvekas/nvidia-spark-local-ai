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
| How fast are Qwen3.6 and Qwen3.8 under concurrency and 61K–246K inputs? | [Cache-off NVFP4+MTP3 throughput, wall time, TTFT, retries, and validation](docs/qwen36-qwen38-long-context-tps-2026-08-25.md) |
| How has Qwen3.8-Flash-Next run on one Spark? | [Historical native Radix SGLang and IQ4_XS llama.cpp throughput, validation, and memory boundaries](docs/qwen38-flash-next-gb10-2026-08-26.md) |
| What did the historical native SGLang Flash-Next path show on GB10? | [Measured NVFP4+MTP result, read-only NVMe PLE mechanics, exact pins, and long-context limit](docs/qwen38-flash-next-native-mtp-optimization-2026-08-26.md) |
| What did the first day of GB10 community work add? | [Primary-source review of the one-Spark vLLM mmap patch, upstream status, and dual-Spark SGLang report](docs/qwen38-flash-next-gb10-day-one-2026-08-27.md) |
| What changed on day two for Flash-Next on GB10? | [SGLang's SM121 safety reversal, vLLM's new mmap PR and profiler evidence, and ranked reproduction targets](docs/qwen38-flash-next-gb10-day-two-delta-2026-08-28.md) |
| What has the replacement SGLang candidate actually passed? | [SM121 Triton/storage prerequisites, target-only quality and varied-context admission, and the remaining scope limits](docs/qwen38-flash-next-sm121-storage-pre-admission-2026-08-28.md) |
| What did the admitted SM121 cache-policy timing show without leaking prompts or claiming TTFT? | [Two independent audited A/B/B/A fresh-lifetime results: retain cache-on A](docs/qwen38-flash-next-sm121-cache-performance-protocol-2026-08-29.md) |
| How is the live SM121 product track now using an autoresearch-style loop safely? | [Current-runtime v2 registry: freeze one reviewed candidate, execute once, and derive retain/reject/inconclusive from audited scalar evidence](docs/qwen38-flash-next-sm121-autoresearch-v2-2026-08-29.md) |
| What did the current-SM121 chunked-prefill studies show? | [Audited fresh-lifetime A/B/B/A results: retain 2K over 1K, then 4K over 2K; 8K passed private safety admission but remains unmeasured](docs/qwen38-flash-next-sm121-chunked-prefill-protocol-2026-08-29.md) |
| How should the vLLM direct-PLE-mmap path be reproduced? | [Exact stacked source boundary, inferred Radix checkpoint, local readiness, admission, ABBA, long-context, and profiler plan](docs/qwen38-flash-next-vllm-mmap-reproduction-2026-08-28.md) |
| What do matched PLE mapping/omission and NEXTN depths show? | [Replicated lazy-C8 depth results, semantic-ablation failures, and quality-clean exact-answer v2](docs/qwen38-flash-next-ple-depth-study-2026-08-27.md) |
| What happened to the single-user Qwen3.8-Flash-Next serving search? | [Frozen 14-cell, nine-case 64K protocol; admission expired without a measurement, and its retired runtime is now blocked at every execution ingress](docs/qwen38-flash-next-single-user-autoresearch-2026-08-28.md) |
| What should the next single-user Flash-Next experiments test? | [Ranked cache/state, fan-out, MTP, CUDA-graph, chunk-size, and streaming protocols, plus source rejection of continuous decode and adaptive NEXTN](docs/qwen38-flash-next-single-user-next-experiments-2026-08-28.md) |
| What is limiting single-user Flash-Next TPS on Spark? | [Evidence-backed target-pass, batching, MTP, PLE, startup, and profiler analysis](docs/qwen38-flash-next-tps-bottleneck-2026-08-28.md) |
| How should Pi and richer cowork tasks be benchmarked locally? | [Pinned Pi-in-Harbor adapter and deterministic cowork-core-v1 plan](docs/pi-cowork-harness-plan-2026-08-28.md) |
| Can the current SM121 image even support the needed Qwen reasoning/tool parsers? | [Static image-local parser preflight: `qwen3` and `qwen3_coder` are present; live agent admission remains pending](docs/qwen38-flash-next-sm121-agent-admission-preflight-2026-08-29.md) |
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
| How will the overnight RLM and HALO loops be compared? | [Frozen paired BABILong-derived and synthetic Graphiti-like campaign protocol](docs/rlm-halo-overnight-2026-08-25.md) |
| What did the RLM/HALO continuation show? | [Audited compaction, breadth, 20-turn reliability, and 65K-trace continuation results](docs/rlm-halo-continuation-2026-08-26.md) |

The generated public evidence archive is intentionally separate from raw run
data. Its [human-readable map](evidence/README.md) and
[machine-readable index](evidence/index.json) account for complete, partial,
aborted, and nonterminal attempts without publishing raw payloads. The
[evidence publication section](BENCHMARK.md#publishing-sanitized-evidence)
explains how to create and verify both files. The current refresh contains
2,031 files covering 318 run bundles and 29 campaign bundles.

## What the results say

- On the current SM121 cache-on Qwen3.8-Flash-Next stack, the audited
  fresh-lifetime 1K/2K chunked-prefill A/B/B/A panel retained 2K for the 60K
  static-history proxy. Mean cache-cold `T0` wall was 38.635 s at 2K versus
  50.022 s at 1K (`0.772x`); append proxy wall remained within guardrail at
  `1.010x`, and full `T0`–`T2` wall was `0.787x`. This is request-wall evidence
  at C1 with thinking disabled—not TTFT, decode TPS, concurrency, or an
  agentic-coding result. See the
  [chunked-prefill protocol and result](docs/qwen38-flash-next-sm121-chunked-prefill-protocol-2026-08-29.md).
- The independent 2K/4K fresh-lifetime A/B/B/A follow-up also retained 4K for
  that same proxy. Mean cache-cold `T0` wall was 30.026 s at 4K versus
  38.634 s at 2K (`0.777x`); append proxy wall was `1.006x`, and full
  `T0`–`T2` wall was `0.795x`. Four quality gates, eight static/runtime
  attestations, and the read-only audit all passed. This successive setting
  change remains request-wall evidence at C1 with thinking disabled—not TTFT,
  decode TPS, concurrency, tool calling, or an agentic-coding result.
- The 8K chunked-prefill candidate has passed a separate private
  quality-plus-cold-`T0` safety admission. That timing-free admission is not
  evidence; the receipt-bound 4K/8K comparison remains unmeasured and is
  excluded from the ratios above.
- In the historical, safety-superseded SM121 TRT-LLM mapped-PLE ABBA panel,
  NEXTN depth two beat depth one in all
  five cells across two independent lifetimes per arm. Mean D256 and fresh
  C1/C2/C4/C8 throughput moved from 28.304 and
  27.217/46.077/73.713/109.351 tok/s at D1 to 29.402 and
  29.594/51.870/75.471/117.140 tok/s at D2, gains of 2.4--12.6%. The single D3
  mapped lifetime reached 118.454 tok/s at C8, only 1.1% above the D2 mean, and
  crossed the 14 GiB startup memory floor. PLE omission was valid through C2
  (-2.9%/+7.8%/+3.4% at D256/C1/C2) but ended one C4 and one C8 request early,
  so no official high-concurrency aggregate is published. Separate stable-prompt
  quality-v2 lifetimes completed at strict 8/8 for mapped and omitted arms under
  the unchanged validator. See the
  [matched PLE/depth result](docs/qwen38-flash-next-ple-depth-study-2026-08-27.md).
- The full Radix Qwen3.8-Flash-Next checkpoint completed a historical native
  SGLang panel on one Spark. These measurements used the SM121 TRT-LLM route
  later restricted after varied-token corruption; they are evidence, not
  current deployment guidance. SparkBench now rejects the exact retired overlay
  digest across new and frozen execution paths without rewriting historical
  profiles or evidence. New work requires a newly built and admitted SM121
  Triton rebaseline; see the [day-two safety review](docs/qwen38-flash-next-gb10-day-two-delta-2026-08-28.md).
  The [primary run](evidence/runs/20260827T032027Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-20e1283b/summary.json)
  used ModelOpt NVFP4 main weights, the source FP8 PLE through a digest-pinned
  read-only NVMe mmap, and the trained BF16 `NEXTN` head. It reached 28.504
  aggregate output tok/s at D256 and 27.413/50.330/72.821 tok/s at fresh
  C1/C2/C4. Its repeated-word 8K/32K prefill proxies were
  2,103.468/2,179.588 prompt tok/s, and both exact-key needles passed 3/3.
  Startup took 581.652 s, the first measured request had 14.552 s TTFT, no swap
  growth was observed, and minimum available memory across cases was 16.564
  GiB. The lifecycle completed; the scalar summary is `partial` only because
  the synthetic quality case scored 3/4 after one code-reasoning miss. Thirty
  periodic server-log samples had mean accepted length 2.956 and mean
  acceptance rate 0.653, but they are neither a lifetime nor case aggregate.
  The later clean, near-matched D256/C1 pair gives the bounded MTP estimate:
  MTP3 reached 30.123639 tok/s versus 16.663713 off (`1.807739x`), saved
  137.288 seconds/44.682% of case wall time, and measured `1.821397x` output
  tokens per sampled joule. Its separate native audit accepted 175/243
  proposals (72.0165%), scoped only to that explicit audit request. The MTP3
  arm encoded 1,610 prompt tokens versus 1,590 off, a disclosed 1.26% input
  mismatch across otherwise nominally identical 20-request shapes. The clean
  bounded lazy-state MTP2 profile reached 114.5755 tok/s at offered C8,
  +48.069% over C4 while holding median E2E to `1.930421x` C1; operator-log
  inspection observed all eight requests running. See the
  [native result and exact pins](docs/qwen38-flash-next-native-mtp-optimization-2026-08-26.md).
- The same historical native MTP route passed the repeated-word 131K exact-key case in two
  runs, including [one with 72.285 s TTFT](evidence/runs/20260827T024144Z-qwen38-flash-next-nvfp4-mtp-long-sglang-qwen38-flash-next-sglang-long-context-7c25f743/summary.json).
  It did not safely admit the 245K case: the MTP pool rejected the request at
  the `0.85` allocation, `0.87` was pressure-unsafe, and the capped target-only
  BF16-state profile safety-aborted at 0.046 GiB available. That last operator
  observation also saw about 6.1 GiB of new swap and memory-PSI full `avg10`
  19.84; it is not a sanitized aggregate. The retained 245K profile is marked
  incompatible. These repeated-word cases measure serving capacity and
  exact-key retention, not natural-document understanding or cold/varied-token
  NVMe-PLE cost.
- On clean revision `efabab7`, the Qwen3.8-Flash-Next IQ4_XS llama.cpp core run
  completed its terminal lifecycle and reported 20.193 aggregate output tok/s
  at D256 and 19.860/19.782/51.927/71.709 tok/s at C1/C2/C4/C8. The 16K
  needle passed 3/3, structured JSON 5/5, tool calling 5/5, and exact answers
  4/4. Its summary is partial only because D1024 returned 4,327 of 5,120
  requested completion tokens and failed validation. Runwide minimum available
  memory was 4.011 GiB; the live server showed at least 3.85 GiB `VmSwap`
  during 16K prefill and recovered after teardown. The shorter quick run had no
  new swap use observed, reached 19.601/31.240/49.363 tok/s at D128/C2/C4, and
  bottomed at 4.270 GiB available. These are different suite shapes, not a
  matched swap comparison. See the
  [day-zero report](docs/qwen38-flash-next-gb10-2026-08-26.md).
- In the clean RLM/HALO continuation, all 106 planned cases reached an explicit
  terminal state, but only 40 completed and scored before the frozen cutoffs.
  Normal depth-1 RLM scored 2/24 and recorded no recursive subcall in any
  completed case. HALO completed 2/9 attempted 20-turn cells, including one
  65,536-trace depth-0 case, while no attempted depth-1 cell completed. Across
  seven overlapping cell dimensions, the prior 10-turn campaign completed 7/7
  versus 1/7 here; this sequential cross-run regression is diagnostic, not a
  causal estimate. Prefix reuse was high, but no cache-off arm measured its
  benefit. See the [continuation report](docs/rlm-halo-continuation-2026-08-26.md).
- In the exploratory cache-off NVFP4+MTP3 panel, Qwen3.6 35B-A3B reached
  893.674 aggregate output tok/s at short-prompt offered C64 versus 359.081 for
  dense Qwen3.8 27B. At the approximately 246K/C2 target, the matched valid
  rates were 1.951 versus 0.728 tok/s and median TTFT was 195.222 versus
  517.665 seconds. Qwen3.8 finalized and validated all 25 planned cases;
  Qwen3.6 finalized all 25 and validated 22. Both passed 39/39 retrieval
  requests. Both saturation plans encountered a CUDA failure in their first
  server lifetime and completed the remaining cases after resume, so this
  uncached baseline is neither a cache-savings nor production-stability claim.
  See the [long-context throughput report](docs/qwen36-qwen38-long-context-tps-2026-08-25.md).
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

- [Qwen3.8-Flash-Next matched PLE/depth result — 2026-08-27](docs/qwen38-flash-next-ple-depth-study-2026-08-27.md):
  mapped-PLE depth-one/depth-two ABBA replication, matched depth-three omission
  ablation, lazy C8 throughput and safety, and strict 8/8 quality-v2 lifetimes.
- [Qwen3.8-Flash-Next historical native SGLang result and optimization record — 2026-08-26](docs/qwen38-flash-next-native-mtp-optimization-2026-08-26.md):
  full Radix NVFP4 admission with read-only NVMe PLE and trained `NEXTN`, native
  C1-C4 and 8K/32K measurements, clean MTP3/off confirmation, bounded MTP2 C8,
  131K admission, the incompatible 245K boundary, and earlier kernel controls;
  the measured SM121 TRT-LLM route is safety-superseded.
- [Qwen3.8-Flash-Next day-zero GB10 results — 2026-08-26](docs/qwen38-flash-next-gb10-2026-08-26.md):
  pinned IQ4_XS llama.cpp admission, quick and core throughput, bounded text
  capability checks, and the observed unified-memory and swap boundary.
- [RLM and HALO continuation results — 2026-08-26](docs/rlm-halo-continuation-2026-08-26.md):
  forced RLM compaction, BABILong-derived breadth, 20-turn HALO reliability,
  65K synthetic-trace retrieval, lifecycle overhead, and explicit limits on
  caching and recursion claims.
- [Qwen3.6 versus Qwen3.8 cache-off long-context throughput — 2026-08-25](docs/qwen36-qwen38-long-context-tps-2026-08-25.md):
  short saturation, 8K/30K concurrency and native 61K–246K generation and
  retrieval, with wall time, validation failures and retry instability kept
  explicit.
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
- [RLM and HALO overnight campaign protocol](docs/rlm-halo-overnight-2026-08-25.md)
- [Qwen3.8-Flash-Next GB10 day-one literature review](docs/qwen38-flash-next-gb10-day-one-2026-08-27.md)
- [Qwen3.8-Flash-Next GB10 day-two literature delta](docs/qwen38-flash-next-gb10-day-two-delta-2026-08-28.md)
- [Qwen3.8-Flash-Next SM121 native-storage pre-admission and first-run canary](docs/qwen38-flash-next-sm121-storage-pre-admission-2026-08-28.md)
- [Qwen3.8-Flash-Next SM121 cache-policy timing protocol](docs/qwen38-flash-next-sm121-cache-performance-protocol-2026-08-29.md)
- [Qwen3.8-Flash-Next SM121 autoresearch v2 registry](docs/qwen38-flash-next-sm121-autoresearch-v2-2026-08-29.md)
- [Qwen3.8-Flash-Next SM121 1K/2K result, 2K/4K follow-up, and prospective 8K admission](docs/qwen38-flash-next-sm121-chunked-prefill-protocol-2026-08-29.md)
- [Qwen3.8-Flash-Next vLLM direct-mmap reproduction plan](docs/qwen38-flash-next-vllm-mmap-reproduction-2026-08-28.md)
- [Qwen3.8-Flash-Next single-user autoresearch protocol](docs/qwen38-flash-next-single-user-autoresearch-2026-08-28.md)
- [Qwen3.8-Flash-Next single-user serving backlog](docs/qwen38-flash-next-single-user-next-experiments-2026-08-28.md)
- [Qwen3.8-Flash-Next TPS bottleneck analysis](docs/qwen38-flash-next-tps-bottleneck-2026-08-28.md)
- [Pinned Pi and cowork harness plan](docs/pi-cowork-harness-plan-2026-08-28.md)
- [Qwen3.8-Flash-Next current-SM121 Pi/cowork agent-admission parser preflight](docs/qwen38-flash-next-sm121-agent-admission-preflight-2026-08-29.md)
- [Autoresearch controller hardening and campaign-evidence record](docs/autoresearch-controller-hardening-2026-08-28.md)
- [Nemotron diffusion direct-run guide](docs/nemotron-diffusion-direct.md)
- [Experimental NInfer SM121a patch and reproduction notes](patches/ninfer/README.md)
- [SGLang SM121 QSA and persistent read-only PLE patch guide](patches/sglang/README.md)
- [vLLM direct-mmap live-token-width semantic backport](patches/vllm/README.md)

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
measurements, validation outcomes, telemetry, terminal state, reproducible
artifact/runtime pins, typed safety annotations, and path-free source-overlay
and read-only PLE-cache digests while excluding captured inputs or outputs,
reasoning, tool payloads, request identifiers, local paths, logs, media, and
credentials.
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
