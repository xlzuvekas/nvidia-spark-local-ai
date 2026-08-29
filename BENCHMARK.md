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

### Qwen3.8-Flash-Next day-zero GB10 protocol

The 2026-08-26 local route pins `unsloth/Qwen3.8-Flash-Next-GGUF` revision
`2c41bd2a0b3f51c503c11f1c7ed2e6bb34036beb`: three UD-IQ4_XS shards totaling
93,682,584,224 bytes. It uses the unmerged `qwen4exp` llama.cpp revision
`035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`, binary digest
`sha256:6b0e09f19768e1424eac29b27d6d7f5ca661a9f73b5b7a2ecba5e768af8a366a`,
and profile `qwen38-flash-next-ud-iq4-xs-llamacpp-p8`. The profile allocates
eight 32,768-token slots, offloads all layers, enables flash attention, uses
F16 key/value cache, and disables thinking at both the server and request
levels. This panel did not exercise MTP, multimodal input, a reasoning-effort
sweep, prompt-cache controls, or context beyond the served 32K per slot.

The first startup attempt with Q8_0 key/value cache aborted at the exact
runtime's `qwen4exp` graph assertion
`inp->self_k_rot == nullptr && inp->self_v_rot == nullptr`. Changing both cache
types to F16 admitted the same pinned weights and runtime. This is a bounded
workaround for that exact combination, not a general Q8_0 compatibility claim.

The core run
`20260826T165913Z-qwen38-flash-next-ud-iq4-xs-llamacpp-p8-core-b5a0f9ad`
used `manifests/suites/core.toml` at clean harness revision `efabab7`. It reached
terminal `completed`; its summary is `partial` only because the five D1024
requests emitted 4,327 of 5,120 requested completion tokens, so that case's
validation is false. All rates below are aggregate completed output tokens per
case wall time; the quick and core rows retain different budgets and repetitions
and are not matched estimates.

| Run | Decode | C1 | C2 | C4 | C8 | Minimum `MemAvailable` | Live server swap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Quick | D128 19.601 tok/s | — | 31.240 tok/s | 49.363 tok/s | — | 4.270 GiB | No new use observed |
| Core | D256 20.193 tok/s | 19.860 tok/s | 19.782 tok/s | 51.927 tok/s | 71.709 tok/s | 4.011 GiB | At least 3.85 GiB during 16K prefill |

The core validators passed the 16K needle 3/3, structured JSON 5/5, tool calls
5/5, and exact answers 4/4. The live swap observation recovered after managed
teardown; it does not isolate a performance effect. Reproduce the frozen path
with:

```bash
python3 sparkbench.py fetch qwen38-flash-next-ud-iq4-xs-llamacpp-p8
python3 sparkbench.py benchmark qwen38-flash-next-ud-iq4-xs-llamacpp-p8 \
  --suite manifests/suites/core.toml
```

The evidence exporter treats each `loop-*` attempt as its own strict scalar
campaign bundle and excludes its prompts, completions, traces, reasoning,
request identifiers, local paths, and raw logs. The two exact private Harbor
lifecycle records needed to reproduce the historical Harbor bundle are retained
locally and are inputs, not tracked artifacts. A first export requires both;
subsequent refreshes may carry the existing canonical Harbor bundle forward only
after its schema and checksums verify. The current tracked refresh contains
1,969 files, 309 run bundles, and 21 campaign bundles. It publishes the four day-zero llama.cpp Flash
Next attempts under
[`evidence/runs/`](evidence/runs/), including the
[core bundle](evidence/runs/20260826T165913Z-qwen38-flash-next-ud-iq4-xs-llamacpp-p8-core-b5a0f9ad/manifest.json),
as well as the later native SGLang attempts described below. The complete
refresh verifies deterministically without reopening private inputs. See the
[day-zero GB10 report](docs/qwen38-flash-next-gb10-2026-08-26.md) for artifact
hashes, smoke history, results, and comparison limits.

### Qwen3.8-Flash-Next native SGLang protocol and results

The native runtime probe pins the Linux aarch64 image
`lmsysorg/sglang:qwen38flashnext@sha256:14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4`.
Its labels report SGLang base commit
`d91c3682b0b429e4c70df63cd57f819588ce29b0`; open PR #36497 commit
`73a255206f916366c8d26d4022f82ddfb0ab558d` is support lineage, not the image
build commit. The import control measured SGLang `0.0.0.dev1+gd91c3682b`,
PyTorch `2.13.0+cu130`, CUDA runtime `13.0`, FlashInfer `0.6.17`, and an NVIDIA
GB10 at compute capability `[12, 1]`. Qwen4 target, `NEXTN`, and ModelOpt NVFP4
embedding symbols imported successfully.

#### Full Radix NVFP4 + read-only PLE result

**Safety supersession, 2026-08-28:** the SGLang measurements and commands in
this section bind the historical SM121 TRT-LLM overlay later restricted after
varied-token corruption. Preserve them as provenance only; do not run the
commands or use these profiles for new inference. New work requires a newly
built, pinned, and admitted SM121 Triton runtime. The former bounded exception,
the frozen fourteen-cell campaign, exhausted its admission window without a
measurement and must not re-enter planning, execution, checkpoint, or direct
startup paths; see its
[full protocol](docs/qwen38-flash-next-single-user-autoresearch-2026-08-28.md#current-status-sealed-and-time-inadmissible-no-campaign-measurements).
SparkBench now enforces that retirement by the exact QSA overlay digest before
fresh planning, frozen execution, campaign side effects, or direct SGLang
startup. Historical loading, evidence, and cleanup remain readable.
The exact measurement-free closeout is also content-sealed: summarization can
only read its pinned blocked summary without mutation, while run and checkpoint
entry points refuse it.

The completed native profile pins `RadixArk/Qwen3.8-Flash-Next-NVFP4` revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`: 206 weight files totaling
135,195,303,851 bytes. It uses ModelOpt NVFP4 for the main MoE weights, keeps
the source PLE in FP8, and loads the trained `NEXTN` head in BF16. The
51,200,245,760-byte PLE payload is materialized once on NVMe, mounted read-only,
and mapped without copying it into anonymous memory; this is not an NVFP4 PLE
claim.

The exact runtime and derived-artifact pins are:

- image
  `lmsysorg/sglang:qwen38flashnext@sha256:14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4`;
- public recipe revision
  `bf2b7c75870d3703730b6bd8f3bb93dc622c278d`;
- `qwen4_exp.py` overlay SHA-256
  `0b513b4dc4f2394f6b1733bb0b74fa40ab59f4a04f6b33601350b2a606c67804`
  and `qwen_sparse_attn_backend.py` overlay SHA-256
  `e30566492e1502f94a4c7fed42d90b523bbb662580c628459e6e63c7b5263c75`;
- read-only PLE payload SHA-256
  `b070f9644adf93794d8a1030584ab705809387e64396a9327a68fa3a3a6666b3`
  and completion-marker SHA-256
  `f0ef55e4e4dec9b6b936a42af4ca2eb9b2f24ced373b1e216f7a6d507b171665`.

Run
[`20260827T032027Z-...-20e1283b`](evidence/runs/20260827T032027Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-20e1283b/manifest.json)
ran on the DGX Spark/GB10 at clean harness revision
`717b17c3150072f6cbc8d0cc5861c489af92d8bd` and completed all 12 cases in
one managed server lifecycle. Its summary status is `partial` only because the
synthetic exact-answer validator scored 3/4: the `code_reasoning` item failed
while arithmetic, instruction following, and logic passed. JSON and tool-call
smokes passed, as did all three 8K and all three 32K exact-key needle requests.

All decode rates below are aggregate completed output tokens divided by case
wall time. Prefill is the client-TTFT approximation, not a runtime-isolated
kernel measurement. The llama.cpp values come from the clean day-zero core run,
except its 8K prefill proxy, which comes from the clean quick run; the C2 value
is a visible discontinuity/outlier. This is a descriptive cross-runtime view,
not an MTP-off causal arm: runtime, quantization, PLE placement, suite shape,
and speculative decoding all differ.

| Metric | Native SGLang NVFP4 + MTP | IQ4_XS llama.cpp |
| --- | ---: | ---: |
| D256 aggregate output | 28.504 tok/s | 20.193 tok/s |
| Fresh C1 aggregate output | 27.413 tok/s | 19.860 tok/s |
| Fresh C2 aggregate output | 50.330 tok/s | 19.782 tok/s (outlier) |
| Fresh C4 aggregate output | 72.821 tok/s | 51.927 tok/s |
| Repeated-word 8K prefill proxy | 2,103.468 prompt tok/s | 674.500 prompt tok/s |
| Repeated-word 32K prefill proxy | 2,179.588 prompt tok/s | — |

Native startup took 581.652 seconds. Server-log timing attributed 420.36 seconds
to target-weight loading and 83.86 seconds to MTP loading; those components do
not account for every startup phase. The first measured request had 14.552
seconds TTFT. No swap growth was observed during the completed run, and the
minimum available memory sampled across measured cases was 16.564 GiB.

Thirty periodic server-log samples had mean accepted length 2.956 and mean
acceptance rate 0.653. They are sparse observations, not an authoritative
lifetime or case aggregate, and they are not used to identify MTP's throughput
contribution.

#### Clean MTP confirmation and bounded C8

The clean, near-matched confirmation at harness revision `2ce8b292` ran MTP3
and MTP off in separate lifetimes with the same nominal D256/C1 geometry, two
warmups and 20 measured requests. All 40 measured requests completed and
validated. MTP3 encoded 1,610 prompt tokens versus 1,590 off, a 1.26% aggregate
input-token mismatch; each arm produced 5,120 output tokens. Aggregate output
throughput is primary because speculative streaming may bundle multiple tokens
in one event and bias the client-timed decode estimate.

| Arm | Output tokens | Aggregate output | Case wall time | Sampled output tok/J |
| --- | ---: | ---: | ---: | ---: |
| MTP3 | 5,120 | **30.123639 tok/s** | **169.966 s** | **0.785612** |
| MTP off | 5,120 | 16.663713 tok/s | 307.254 s | 0.431324 |

MTP3 measured `1.807739x` throughput (+80.7739%), saved 137.288 seconds
(44.682%) and measured `1.821397x` sampled output tokens per joule. This is a
bounded single-stream estimate from a near-matched control; one pair of server
lifetimes does not estimate between-lifetime variance. The preceding
fixed-order depth-0/1/2/3
screen observed 15.9384/26.7568/29.5341/30.7661 tok/s, but it ran from a dirty
worktree and its depth-one/depth-two startup was swap-contaminated. It is
exploratory and does not resolve depth two versus depth three.

After the measured MTP3 case, the runner performed one authenticated,
scalar-only audit: `/v1/tokenize`, then native non-streaming `/generate`. The
pinned build rejects `return_meta_info` and supplies `meta_info` automatically.
The audit accepted 175/243 proposed draft tokens across 81 verifies (72.0165%),
with position counts 72/55/48 and mean accepted length 3.16049. Its scope is
`explicit_sglang_native_audit_requests_only`; no counter is attributed
retroactively to the streaming measurement. Generated text, output IDs,
request identifiers and unallowlisted metadata are discarded.

The clean MTP2 `extra_buffer_lazy` profile then ran the short-context C1-C8
suite with a 4,096 context cap, 32,768 total-token pool, eight decode graphs and
32 recurrent states. Fresh C1/C2/C4/C8 aggregate rates were 28.7930, 48.2511,
77.3798 and **114.5755 tok/s**. All 24 C8 requests completed 256 validated
tokens each. C8 was 48.069% above C4 while median end-to-end latency was
`1.930421x` C1, passing the frozen `>=10%` retention and `<2x` latency gates.
Operator-log inspection observed eight running, zero queued and graph
execution; occupancy from logs is not promoted to machine evidence. The
separate MTP2 audit accepted 159/192 proposals and applies only to that audit.

Ordinary `extra_buffer` with 32 states completed offered C8 at 80.5772 tok/s,
but operator-log inspection showed six running and two queued, so it is not a
simultaneous-eight result. The 40-state MTP2 allocation was safety-stopped at
602.48 MiB swap growth against a 512 MiB gate. An MTP3/40-state attempt
separately crossed below the 14 GiB host-availability floor during graph
capture. These are
capacity rejections, not crashes; the lazy MTP2/32-state profile was retained
only within the historical measured runtime.

The repeated-word 131K exact-key case passed twice: once in
[`a06b138a`](evidence/runs/20260827T015017Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-a06b138a/summary.json)
and once in [`7c25f743`](evidence/runs/20260827T024144Z-qwen38-flash-next-nvfp4-mtp-long-sglang-qwen38-flash-next-sglang-long-context-7c25f743/summary.json),
where its TTFT was 72.285 seconds. The 245K case is not compatible with this
one-Spark profile. The MTP route could not reserve its request pool at the
pressure-safe `0.85` allocation, while `0.87` was pressure-unsafe. A capped
target-only/BF16-state attempt
[`7b88e52c`](evidence/runs/20260827T030636Z-qwen38-flash-next-nvfp4-long-sglang-qwen38-flash-next-sglang-long-context-7b88e52c/manifest.json)
was safety-aborted at 0.046 GiB available. The operator also observed about 6.1
GiB of new swap and memory-PSI full `avg10` 19.84 in that attempt; those two
values are operational observations, not sanitized case aggregates. The exact
245K diagnostic is retained with support status `incompatible` and must not be
served. All 8K-131K prompts repeat one synthetic word, so they test serving
mechanics and exact-key retention rather than natural-document understanding or
cold/varied-token NVMe-PLE cost.

The following commands record how the historical offline overlays, PLE payload,
and native suite were prepared; do not execute them:

```bash
python3 prepare_sglang_overlays.py
python3 prepare_sglang_overlays.py --materialize-ple
python3 prepare_sglang_overlays.py --verify-ple-cache
python3 sparkbench.py fetch qwen38-flash-next-nvfp4-mtp-sglang
python3 sparkbench.py benchmark qwen38-flash-next-nvfp4-mtp-sglang \
  --suite manifests/suites/qwen38_flash_next_sglang_native.toml
```

#### Matched PLE/depth replication and quality v2

The follow-up protocol uses the separate ablation-capable overlay pair produced
by `prepare_sglang_overlays.py --prepare-ple-ablation`. Mapped controls retain
the exact read-only FP8 PLE payload. Omitted arms use a canonical sentinel that
removes the pinned PLE layer and skips exactly its 138 checkpoint tensors; they
have different model semantics and must remain labeled as an ablation.

Performance arms hold `extra_buffer_lazy`, 32 recurrent states, a 32,768-token
pool, 4,096-token context, decode graphs 1-8, thinking off, and C1/C2/C4/C8
constant. Depths one and two run in ABBA lifetime order to two replicates each;
mapped depth three and omitted depth three then form the PLE comparison. The
dedicated `ple-study-*` cases use model-independent prompt tags so profile arms
receive identical prompt text without changing any frozen legacy case.

The quality-v2 suite is separate: mapped and omitted depth-three profiles use
thinking on, `reasoning_effort=low`, temperature zero, C1, a 512-token cap, and
two repetitions of all four exact-answer items. Protocol v2 keeps timestamped
request IDs out of the prompt. Passing requires 8/8 under the original strict
validator; answer keys, retry policy, and partial-run semantics are unchanged.

All eight planned lifetimes reached terminal `run_complete`. The D1/D2 ABBA
replicates completed every case and give these unweighted lifetime means:

| Case | D1 mapped | D2 mapped | D2 vs D1 | D3 mapped | D3 omitted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warmed D256/C1 | 28.304 tok/s | 29.402 tok/s | +3.881% | 32.221 tok/s | 31.286 tok/s |
| Fresh C1 | 27.217 tok/s | 29.594 tok/s | +8.734% | 30.762 tok/s | 33.171 tok/s |
| Fresh C2 | 46.077 tok/s | 51.870 tok/s | +12.572% | 53.230 tok/s | 55.050 tok/s |
| Fresh C4 | 73.713 tok/s | 75.471 tok/s | +2.385% | 80.143 tok/s | — |
| Fresh C8 | 109.351 tok/s | 117.140 tok/s | +7.123% | 118.454 tok/s | — |

The D3 mapped points come from one lifetime and its startup measurement is
invalid: the first request crossed the 14 GiB floor at 13.9045677185 GiB. All
five serving cases still completed and remain case-valid. The omitted D3 arm
completed D256/C1/C2 but stopped one C4 request at 172/256 tokens and one C8
request at 232/256; its run is honestly partial and those two official
aggregates are null. Descriptive incomplete-work rates must not fill the table.
PLE omission therefore has no valid high-concurrency speedup result and remains
a semantic ablation, not a deployable optimization.

Both separate quality-v2 lifetimes completed at strict 8/8. This resolves the
old 3/4 synthetic result through stable prompt text and a frozen thinking-low
quality configuration, without changing the validator or partial-run policy.
See the
[PLE/depth result](docs/qwen38-flash-next-ple-depth-study-2026-08-27.md) for
individual lifetimes, exact evidence links, wall time, memory, safety gates,
artifact pins, and interpretation limits.

The 2026-08-28 follow-up freezes a fresh mapped-PLE 2 × 2 block to measure the
ordinary/lazy buffer-strategy effect at NEXTN depths two and three, and to give
lazy D3 its second independent lifetime. New ordinary D2/D3 profiles differ
from their clean lazy controls only in public identity and
`--mamba-radix-cache-strategy`; all use 32 recurrent states and the same 4K,
32K-token-pool, graph-1-through-8 geometry. The fixed lifetime order is
ordinary D2, ordinary D3, lazy D2, lazy D3. Primary per-case interaction is the
additive difference-in-differences `(L3 - L2) - (O3 - O2)` from that one block.

The dedicated interaction suite preserves the prior D256/C1/C2/C4/C8 cases in
the same order, then appends C6. Under the 32-state pool, offered C6 is
state-capacity-feasible for ordinary and lazy while offered C8 is feasible only
for lazy. Treat C1/C2/C4 as the primary matched-strategy cells, C6 as a secondary
shared offered-load/state-capacity point, and C8 as a scheduling/capacity
outcome rather than matched-concurrency TPS.
The first ordinary-D2 lifetime completed that five-case prefix, but its appended
C6 tail exceeded the 512 MiB runtime swap-growth limit by reaching a
2,473.8359375 MiB increase. The C6 case is measurement-invalid and must not be
retried. The frozen `interaction-core-v2` recovery suite contained exactly the
five completed prefix cases, but the following ordinary-D3 startup increased
swap by 3,173.1484375 MiB and was stopped before any case. That second breach
terminated the block: lazy D2/D3 were not started, the interaction is not
estimable, and lazy D3 remains unreplicated. Do not splice the earlier lazy
panel into the missing fresh cells.
The earlier 40-state ordinary D2/D3 safety
rejections remain incompatible and must not be retried. The full frozen design
and commands are in the
[follow-up protocol](docs/qwen38-flash-next-ple-depth-study-2026-08-27.md#lazy-buffer--depth-follow-up--frozen-2026-08-28).

#### Single-user 64K autoresearch campaign (frozen; admission-expired)

This was a prospective C1 serving search, not a reported measurement result.
Its sole bounded historical exception is exhausted: no pair may now run, and
the plan is not a deployment-safety baseline.
The campaign-schema-2 freeze created fourteen pristine cells from clean,
pushed revision `aa9cca8` at 01:09 MST on 2026-08-28. Its exact sealed summary
remains schema 1. The first legacy pre-journal preflight invocation returned
`blocked_environment` with `starting_swap_above_clean_limit`: used host swap was
868.414 MiB against the frozen 64 MiB start cap. No durable admission record,
controller event, calibration record, cell summary, worker state, container,
or model request was created.

At 05:38 MST, after the inclusive pair-admission boundary closed at 05:37:51,
a second pristine invocation returned exit status 1 with
`insufficient_time_for_pair`, `starting_swap_above_clean_limit`, and
`insufficient_preflight_memavailable`. The legacy schema-1 summary still calls
this `blocked_environment`, but the window cannot reopen before the fixed
cutoff. Its post-invocation audit found no controller or cell journal,
calibration, worker state, server, container, GPU compute process, or
measurement. The 30-directory/31-file campaign topology and frozen identities
remained exact. Do not invoke its execution controller again. The public
summarizer can only validate and return the exact sealed schema-1 summary
without mutation.

The scalar archive publishes all fourteen plans as `nonterminal` with
`measurement_terminal=false` plus one campaign-level
[`autoresearch_campaign` bundle](evidence/campaigns/qwen38-flash-next-single-user-autoresearch-2026-08-28/manifest.json).
The latter reports `blocked_environment`, controller `planned`/phase `new`,
frozen schema 2, `sealed_legacy_unjournaled`, zero admissions, events, or
decisions, and the three sealed blockers. These are provenance projections,
not benchmark observations. The frozen profile queue is, in order:

1. `qwen38-flash-next-nvfp4-mtp2-agent64k-low-ple-mapped-sglang`
   (baseline: mapped PLE, lazy recurrent state, NEXTN depth two, 1,024-token
   chunks, low reasoning);
2. `qwen38-flash-next-nvfp4-mtp2-agent64k-none-ple-mapped-sglang`
   (explicit no-thinking candidate);
3. `qwen38-flash-next-nvfp4-mtp2-agent64k-low-chunk2k-ple-mapped-sglang`
   (2,048-token chunk candidate); and
4. `qwen38-flash-next-nvfp4-mtp3-agent64k-low-ple-mapped-sglang`
   (depth-three candidate, deliberately last because prior depth-three
   geometries crossed the memory or swap safety gate).

All four profiles bind only to
`manifests/suites/qwen38_flash_next_sglang_agent64k_autoresearch.toml`.
That immutable nine-case suite runs JSON and tool-call smokes, exact-answer v2,
the four three-variant agentic scenarios, a 60,000-repetition long-context
needle, and a warmed five-repetition D256/C1 decode cell. Every measured
request is C1 at temperature zero. The profile queue is fixed; it does not
authorize composing candidate axes or editing a live profile.

One fresh server lifetime has an inclusive 1,800-second causal measurement
envelope, opened by its durable `measurement_started` marker and closed by
`measurement_complete`. Each cell gets 30 seconds to reach that start marker,
then 120 seconds of separately attributed owned cleanup through
`server_stopped`; `run_complete` must follow the stop within 10 seconds before
the cell is scoreable. A pair is admitted only with at least 4,930 seconds
remaining: `2 * 1,800` measurement + `2 * 120` cleanup + `2 * 30` start-marker
allowance + `120` inter-cell gap + `10` final-cell finalization + `900` audit
reserve. The first finalization fits inside the inter-cell allowance.
Search-pair scores derive the durable audit reserve from the later cell's
`run_complete` wall timestamp, not a later replay clock. A reset cannot restore
the elapsed admission window. Do not resume, refreeze, copy, or shorten this
campaign. The historical commands below are preserved as provenance only; do
not run them against this campaign:

```bash
python3 sparkbench.py autoresearch-plan \
  --campaign manifests/campaigns/qwen38_flash_next_single_user_autoresearch.toml \
  --dry-run

python3 sparkbench.py autoresearch-plan \
  --campaign manifests/campaigns/qwen38_flash_next_single_user_autoresearch.toml \
  --results results/autoresearch

python3 sparkbench.py autoresearch-run results/autoresearch/FROZEN_CAMPAIGN_DIR
```

The one supported controller closeout command is read-only for this exact
sealed identity:

```bash
python3 sparkbench.py autoresearch-summarize \
  results/autoresearch/FROZEN_CAMPAIGN_DIR
```

It validates the seal and topology and returns the preserved schema-1
`blocked_environment` summary without writing. It cannot create admission
records, change status, or authorize execution. No admission record or
`expired` status is reconstructed. Fresh schema-3 campaigns instead classify
a pure time-only denial as `cutoff`/`expired`; mixed time plus safety denials
preserve every blocker and retain the stronger safety outcome.

Fresh freezes use campaign schema 3 with `admission_journal_required=true`.
On an invocation that can reach a launch, the controller verifies the chained
admission journal before reconciliation or controller mutation. It appends a
current, target-bound live admission immediately before a calibration or
search-pair launch decision. Prior records never authorize execution. The
schema-2 legacy campaign remains frozen and unjournaled.

In the historical procedure, the exact directory printed by the freeze command
was substituted above. Each run invocation would execute at most one
calibration, screen, or confirmation pair and then return. Frozen cells are
one-use: an incomplete started cell or an
inter-cell gap over 120 seconds invalidates the pair and terminates the
campaign; neither arm is restarted. A raw-complete cell may instead be
reprojected and reconciled without inference only when its fingerprint, plan
integrity, nonce, lifecycle markers, validations, telemetry, and frozen order
all replay exactly. The four profiles also freeze a fail-closed 250 ms host
watchdog: at least 14 GiB
available memory, at most 64 MiB starting swap, and at most 512 MiB additional
swap. It interrupts only the exact owned SGLang container; safety profiles
reject `--keep-server`. After every audited pair, and after any terminal safety
stop, export only allowlisted scalar evidence, verify the staged projection,
commit, and push before explicitly resuming. Raw prompts, completions,
reasoning, tool payloads, logs, identifiers, and commands remain ignored.

Read-only evidence closeout remains valid for the sealed campaign:

```bash
python3 sparkbench.py autoresearch-summarize \
  results/autoresearch/FROZEN_CAMPAIGN_DIR
python3 sparkbench.py export-evidence \
  --results results --output evidence --replace
python3 sparkbench.py verify-evidence evidence
git diff --exit-code -- evidence
git add evidence
python3 sparkbench.py verify-evidence evidence --staged
```

The controller enforces the remote checkpoint boundary before it admits a new
pair. The following tail applies only to an admitted fresh schema-3 campaign
after a completed pair, never to this sealed directory:

```bash
git commit -m "Record autoresearch pair evidence"
git push
python3 sparkbench.py autoresearch-checkpoint \
  results/autoresearch/FROZEN_CAMPAIGN_DIR
python3 sparkbench.py autoresearch-run \
  results/autoresearch/FROZEN_CAMPAIGN_DIR
```

The checkpoint command writes a private mode-0600 acknowledgement under ignored
`logs/autoresearch-checkpoints/`, outside both raw `results/` and tracked
`evidence/`, so it cannot become an input to the evidence checksum it binds.
It proves the completed pair, journal prefix, evidence tree, clean commit, and
identical live upstream. No Git, evidence export, remote proof, or other
network operation is permitted between pair cells. `checkpoint_required` is a
resumable nonterminal pause that starts no cell, does not write or rewrite
`summary.json`, and appends no failure transition; `autoresearch-run` still
prints the in-memory status and uses exit status 3 so automation can distinguish
that pause from a completed run or an environmental blocker. Exact-owned
cleanup and any deterministic raw reconciliation happen before the gate, so a
cleanup, ownership, or one-use failure takes precedence over a checkpoint
pause.

No campaign-admitted Pi harness is available. The nine cases are deterministic
coding/cowork proxies; they do not measure Pi, repository editing, document
work, or end-to-end agent productivity. See the
[full campaign protocol](docs/qwen38-flash-next-single-user-autoresearch-2026-08-28.md)
for the queue, scoring, current stop, and interpretation limits.

#### Retained day-zero diagnostics

The isolated CUDA NVFP4 embedding primitive matched an independent
E2M1/block-scale reference with maximum absolute error `0`. That is a primitive
correctness result, not proof that Qwen4 loads a sharded NVFP4 PLE table. The
BF16 FlashInfer GDN probe failed on SM121 and required FP32, while native QSA
failed during CuTe MLIR compilation. Preserve those failures: disabling QSA and
retaining FP32 GDN defines a diagnostic control, not a representative optimized
configuration.

The server fixture is
`inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B` revision
`5fbd297b1529cfa7db2510896d1ad77d1bf41e44`. A temporary runtime copy corrected
its stale `qwen_sparse_attention` layer label to the official `full_attention`
value and set root `language_model_only` to `true` for the text-only loader.
With QSA disabled, the real tiny checkpoint loaded in `1.71 s` and returned
HTTP `200` for a 4-prompt-token to 8-completion-token request in `1.759583 s`
client wall time.

The fixture omits actual MTP weights. A separate `--load-format dummy` control
therefore instantiated synthetic target and MTP tensors solely to exercise the
native `NEXTN` path. It returned HTTP `200` with 16 completion tokens in
`0.338387 s` and reported 15 proposed, 15 verified, and 0 accepted draft
tokens. This proves draft execution and counter visibility, but zero accepted
drafts provide no acceleration evidence. Synthetic and tiny-fixture timing is
not representative TPS, model quality, or a comparison between MTP and
non-MTP serving.

The controls were subsequently frozen as manifest profiles and rerun from
clean harness revision `d50c75799dd00122c39f0d26b28f7344f67828c4`. The
[real-weight QSA-disabled smoke](evidence/runs/20260826T190843Z-qwen38-flash-next-tiny-qsa-disabled-sglang-smoke-30d30d00/summary.json)
and [dummy `NEXTN` smoke](evidence/runs/20260826T190953Z-qwen38-flash-next-tiny-dummy-nextn-sglang-smoke-931e5c58/summary.json)
both completed and stopped their owned container. Their tracked timing remains
admission-only: the controls have different weights and token counts, and the
SGLang OpenAI path exported no speculative-acceptance aggregate.

A later manual diagnostic admitted SM121 to the existing FlashInfer paged
decode resolver. The [exact two-line patch](patches/sglang/README.md) selected
the installed XQA-capable wrapper and completed native-QSA 4-to-8 and warm
6-to-32-token requests on the tiny fixture. That observation remains a bounded
kernel-routing result, not representative throughput. The same overlay lineage,
extended with the persistent read-only PLE loader, underlies the full run above;
it is digest-pinned local derived support, not evidence of upstream SGLang
support. See the
[native result and optimization record](docs/qwen38-flash-next-native-mtp-optimization-2026-08-26.md)
for the fit mechanism, integration history, and comparison limits.

### Agentic tool-use protocol

The `agentic-tools` suite is a bounded admission gate for multi-turn function
calling. It covers tool selection with distractors, correct no-tool abstention,
two dependent calls, and recovery from one typed transient tool error. Each
scenario has three deterministic variants. Tool ordering varies by variant,
tools execute only through an in-process allowlist, and model-provided calls are
schema-checked before dispatch.

An episode runs at temperature zero with automatic tool choice, one active
request, a maximum of six model turns, and up to 4,096 completion tokens per
turn. Server slot geometry remains profile-specific and must match for paired
performance claims.
The frozen context admission estimate includes all six output budgets plus tool
history overhead. Agentic cases do not run concurrently and do not use a
per-case warm-up.

Report two outcomes separately:

- **strict task success** requires the declared call sequence and exact
  argument values, successful dependency or error-recovery behavior, and a
  final answer accepted by the bounded `FINAL:` envelope grammar before either
  limit;
- **tool-trace correctness** validates selection, abstention, arguments,
  ordering, dependency, and recovery without waiving the final envelope-format
  requirement.

Strict success is the primary deployment result. Trace correctness is a
diagnostic, not an alternate pass criterion. Rank matched configurations by
success first, then turns, malformed or unknown calls, recovery rate, episode
wall time, and sampled energy per strict solve. MTP comparisons additionally
require runtime-native proof of draft activity and the same scenario variants,
budgets, runtime, main artifact, and serving geometry.

Only scalar episode outcomes may enter journals or exported evidence. Scenario
text, response content, reasoning, tool arguments and responses, call
identifiers, and per-request tags remain excluded. The first complete campaign
and its comparison limits are recorded in
[the 2026-08-17 agentic tool-use report](docs/agentic-tools-results-2026-08-17.md).

### Memory-operation component protocol

The [`memory-operations` suite](manifests/suites/memory_operations.toml) is a
bounded component test for models proposed as offline memory reflectors. It
does not launch Graphiti, mutate a MemFS tree, retrieve memories, or measure
downstream question answering. Instead, it isolates the model decision that a
deterministic memory service could validate and apply.

The first family contains three Graphiti-inspired edge-resolution cases:
semantic reuse, accepting a new fact while invalidating a contradiction, and
accepting an unrelated new fact. Their output
contract matches Graphiti's two-array resolver shape: `duplicate_facts` and
`contradicted_facts` contain zero-based integer indexes into continuously
numbered candidate lists. Duplicate indexes are limited to existing facts;
contradiction indexes may refer to existing facts or the subsequent
invalidation-candidate range. The prompts and graph state are synthetic and
smaller than Graphiti's
production extraction and candidate-resolution pipeline, so these results must
not be called a Graphiti end-to-end score.

The second family is explicitly a synthetic memory-transaction extension. Its
eight cases cover add, supersede, explicit forget/delete, duplicate no-op,
temporal invalidation, profile/project/session placement, secret refusal, and
untrusted-transcript instruction refusal. The response is one exact
eleven-field JSON object with a fixed action vocabulary and evidence indexes.
It is intended to test a future design in which the model proposes a bounded
transaction while deterministic code validates paths, applies edits, commits,
and advances the reflection cursor. It is not the current Letta Code MemFS
tool-and-git contract.

Version 1 admits only the frozen single-slot llama.cpp geometry with 32,768
allocated context tokens, Q8 key/value cache types, and prompt-cache reuse
forced off for every measured request. Every case has three fixed,
byte-replayable nonce-derived variants, no per-case warm-up, one serial request
per variant, temperature zero, and a 1,536-token completion cap. The ordinary
one-request server-startup probe remains outside the measured memory battery.
The cap is intended to leave visible-JSON headroom after the matched Ornith
profile's 1,024-token
reasoning-budget setting, but the sampler may also generate delimiters or
additional reasoning blocks; neither 512 visible tokens nor an observed
reasoning count is implied. Both families request strict JSON Schema with
unknown properties rejected, and the
harness independently parses duplicate keys, non-standard constants, types,
sets, dates, paths, evidence indexes, and the exact oracle.

The version 1 suite content is bound to protocol digest
`sha256:96df2d5d742c6f4863c77ec3c6cc980845d43900e25607d37fe0be361f0808f1`.
The digest covers the response schemas, prompts, deterministic variant
construction, limits, oracles, protected-value set, and grading contract. It
is incorporated into the suite, plan fingerprint, model-bound case IDs, and
published evidence. A semantic change to any covered input requires a new
digest and new plans; matching case names or a plan created under an earlier
digest is not equivalent provenance.

Report exact operation accuracy, schema-valid emission rate, protected-value
emissions, refusal success, field-level transaction accuracy, and the
Graphiti resolver confusion matrix. Compare models only with the same suite,
variant count, context/slot geometry, runtime controls, and thinking policy.
Pin and report each exact quantization; because the initial panel mixes Q4_K_M
and UD-Q4_K_XL artifacts, its cross-model differences cannot be attributed to
architecture alone. Protected-value detection is deliberately narrow: it
catches contiguous verbatim synthetic values after NFKC normalization and
case folding across visible output, reasoning, and tool payloads. It is not a
general claim of resistance to split, encoded, or confusable-transformed
exfiltration.
llama.cpp b10453 reports all decoded tokens in
`completion_tokens` but does not report an exact reasoning-token partition;
therefore its `reasoning_tokens` value remains null. A thinking run may compare
total completion and wall-time overhead, but it is not evidence of observed
reasoning usage until a pinned runtime supplies that counter. Its client TTFT
is time to the first emitted reasoning-or-visible delta, not time to the first
visible JSON token.

Prompts, generated nonces, expected transactions, model output, reasoning
text, paths, values, identifiers, and transport errors remain raw local data.
Only the fixed scalar result schema and recomputed aggregates may enter tracked
evidence. Publication additionally requires an offline exporter fixture,
deterministic re-export, semantic tamper tests, and staged verification.

The first five-profile component campaign and its exact scalar results are
recorded in the
[2026-08-24 memory-operation report](docs/memory-operations-results-2026-08-24.md).
Its four no-thinking profiles tied at 33/33 operations; the matched Ornith
reasoning-on profile is separately labeled exploratory and passed 27/33. Each
profile was executed once over three deterministic variants per case, so these
bounded outcomes are not estimates of statistical significance. Diagnostic
runs made before the protocol digest bound all semantic inputs were excluded.

### Multi-hop long-context needle protocol

The [`llamacpp-multihop-long-context` suite](manifests/suites/llamacpp_multihop_long_context.toml)
is a single-slot, synthetic two-hop retrieval protocol for profiles explicitly
configured with a 262,144-token context. It is intended for the paired
Qwen3.6 base/MTP2 and Qwen3.8 base/MTP5 long-context profiles, not their
short-context or multi-slot throughput variants.

Each nonce derives one fixed target chain, `source -> relay -> final`, and two
independent two-link decoy chains. The six relation records are separated by
distributed filler, then the query supplies the target source and asks for its
final key. This requires selecting the target source-to-relay relation and
then its relay-to-final relation; a decoy final key is not a valid answer.

The suite has no warm-ups and runs one request at a time at temperature zero.
Its filler targets are 32,768, 65,536, 131,072, and 245,760 repetitions, with
three measured repetitions at each of the first three targets and one at the
last. Each request has a 32-token output budget. These are controlled prompt
construction targets rather than a claim of exact tokenizer token counts.

The oracle accepts only the visible final response after surrounding whitespace
is removed when it exactly equals the nonce-derived target final key. An
explanation, an intermediate relay, a decoy key, or additional visible text
fails. Generated prompts, nonce-derived relation values and identifiers,
visible completions, reasoning, and tool payloads are excluded from the
multi-hop journal payload; it retains only a fixed allowlist of scalar request
measurements and validation state. Failures use a fixed public-safe message
rather than exposing generated values or transport details.

This is a bounded synthetic two-hop retrieval check. It does not establish
general reasoning, long-document comprehension, multi-document question
answering, or a broad long-context quality claim.

### Concurrent long-context throughput protocol

The [`throughput-saturation` suite](manifests/suites/throughput_saturation.toml)
and [`long-context-tps` suite](manifests/suites/long_context_tps.toml) compare
the pinned Qwen3.6 35B-A3B and Qwen3.8 27B NVFP4+MTP3 vLLM recipes without
changing their historical profiles. Separate suites keep 32K offered-load
saturation distinct from conservative native-context C1/C2 measurements.

The dedicated `*-tps64` profiles serve a 32,768-token maximum context and send
barrier-synchronized client bursts through offered C64. The Qwen3.6/Qwen3.8
profiles use 50%/80% GPU-memory utilization respectively, FP8 KV, an
8,192-token batch ceiling, chunked prefill and MTP depth three. The suite
measures 256-token short-prompt generation at C1, C8, C16, C32 and C64, then fresh
8,192- and 30,720-word synthetic inputs through C64 and C32 respectively.
Configured C is offered load, not proof that every request is simultaneously
resident. Sampled server running/waiting logs may describe observed occupancy,
but the generic summary does not parse them into an exact concurrency metric.
The tokenizer-free planning estimate places the 30,720-word/C32 cells below a
previously observed Qwen3.8 KV capacity; actual tokenization and server
admission remain authoritative.

The conservative `*-long-tps` profiles serve 262,144 tokens and set
`--max-num-seqs 2`. Qwen3.6 uses 40% memory and an 8,192-token batch ceiling;
Qwen3.8 uses 52% memory and a 4,096-token batch ceiling. The long suite sends
C1 and C2 fixed-output generation at 61,440, 122,880 and 245,760 repeated-word
targets. Pre-run KV-capacity arithmetic is an admission estimate, not proof
that two requests execute concurrently or that their tokenized contexts have
those exact lengths.

Both suites include key-presence retrieval probes; the saturation suite also
checks 30,720 words at offered C32, while the native-context suite checks
245,760 words at offered C2. Each request has a unique nonce, but chat-template
tokens may precede it and the validator requires key presence rather than
exact-response equality. Every dedicated profile explicitly passes
`--no-enable-prefix-caching`, and selection validation restricts each suite to
its exact profile family. The generic TPS path does not parse the startup
log, require request-scoped cache counters, or emit a cache-verification label.
A run may therefore be described as profile-and-startup-log cache-off only
after separate log inspection; it must not be called counter-verified when
request counters are unavailable. Context rejections, request errors and
failed retrieval remain failures.

The five short-prompt cells have one excluded serial warmup request. Long-input
generation and retrieval cells have no case warmup, so first-use effects are
inside their measurements. Generation validation is strict: every response
must finish by length with exactly 256 completion tokens. One short response
invalidates the whole cell, and the summary suppresses its aggregate and median
decode rates. A completed validation-failed cell is terminal and is not retried
by resume; resume only retries cases recorded as failed.

Report total prompt and completion tokens, case wall time, aggregate output
tokens per measured case wall second, median TTFT, the median client-estimated
post-first-emission decode rate, validation state, configured offered
concurrency, any separately sampled running/waiting occupancy, run-level MTP
metrics when available, telemetry and cleanup state. Case wall excludes warmup
but includes prefill, queueing, client setup and between-burst journal overhead.
Long-prompt aggregate output TPS is therefore not pure decode TPS. SSE events
may bundle tokens, making the client decode estimate secondary; report its
chunking indicator. MTP metrics are cumulative server-lifetime counters that
also cover priming and warmup and may combine resumed lifetimes, not per-case
acceptance measurements.

Three repetitions are three synchronized bursts in one server lifecycle. They
support descriptive medians and explicit burst ranges only, not p95 or
significance claims; pooled request p95 fields emitted by the generic reporter
are not used for this protocol. Disclose every failed attempt and restart even
when a later completed attempt makes the final summary look clean. The frozen
schema-2 plan fingerprint binds the model, full suite geometry and resolved
image, and profile/suite selection is paired fail-closed. It does not bind the
dirty working-tree patch; a dirty-worktree run without a patch digest remains
exploratory. The workload is a synthetic capacity and serving-throughput
surface, not a general long-context quality evaluation.

### Native llama.cpp prefix-cache protocol

The [`llamacpp-prefix-cache` suite](manifests/suites/llamacpp_prefix_cache.toml)
is a dedicated, serial prompt-KV reuse experiment for native llama.cpp. It is
not a replacement for the normal unique-prompt prefill protocol: the shared
prefix is intentional here, and cache-specific profiles cannot be combined
with a general benchmark suite or matrix selection. Conversely, this suite
requires a dedicated cache profile. The admitted profiles are the single-slot,
262,144-context Qwen3.6 and Qwen3.8 pairs:

- `qwen36-35b-a3b-ud-q4-k-xl-llamacpp-prefix-cache-{off,on}`;
- `qwen38-27b-ud-q4-k-xl-llamacpp-long-context-prefix-cache-{off,on}`.

Each profile pair preserves the matching artifact, llama.cpp runtime pin,
single-slot geometry, Q8 KV types, context allocation, generation controls,
and all non-cache server arguments. Mode selection is the literal profile-level
`--no-cache-prompt` or `--cache-prompt` flag; the cache-on schedule's documented
forced-cold requests are the only per-request override. Plan creation and
execution validate that profile/suite pairing and reject an altered cache flag,
multiple slots, or a cache-profile run outside this suite.

The suite contains exactly two synthetic cases: an 8,192-repetition shared-prefix
target and a 32,768-repetition shared-prefix target. Both use temperature zero, one
slot (`id_slot=0`), concurrency one, no warm-ups, five paired blocks, and a
fixed 128-token output limit. A block begins with a deterministic pair key
before the repeated synthetic filler, so it cannot reuse the preceding block's
prefix. All three requests in that block send the same long prefix to the same
slot; only a short request-specific suffix is appended after it. The suffix
therefore cannot prevent the intended long-prefix reuse while keeping requests
distinct.

The frozen request order differs only by cache mode:

- **off:** `forced-cold-a`, `forced-cold-b`, `forced-cold-c`, each using the
  `--no-cache-prompt` profile default and required to report zero cached prompt
  tokens;
- **on:** `forced-cold-a` and `forced-cold-b` explicitly send
  `cache_prompt: false`, then `warm-prefix-hit` uses the `--cache-prompt`
  profile default. The first two are required to be cold; the third must report
  at least 90% cached prompt tokens.

Thus each case contains 15 measured requests (three per block times five), and
each profile run contains 30. The matching pair key is based on stable suite
case metadata rather than a profile-specific frozen case ID, so an off/on pair
uses the same long prefixes without persisting prompt text.

Every request reconciles logical prompt tokens, cached prompt tokens, physical
uncached prompt tokens, and server timings against the final llama.cpp SSE
usage and timing payload. The before/after `/metrics` delta is retained only
as a non-negative scalar Prometheus diagnostic: it is server-global and
batch-scoped, so it is not required to match a request or used in per-request
rates, aggregates, or paired timing claims. Reports keep condition totals and
medians from the request-scoped server fields, including cache-hit fraction,
physical prompt rate, cache-assisted logical prompt rate, server decode rate,
and output rate. A cache-assisted logical input rate is explicitly not a
fresh-prefill rate.

The five retained diagnostic fields are explicitly named
`prometheus_global_prompt_tokens`, `prometheus_global_cached_prompt_tokens`,
`prometheus_global_decode_tokens`, `prometheus_global_prompt_s`, and
`prometheus_global_decode_s`. They are never request-native measurements;
the corresponding `server_*` fields are the final request-scoped SSE values.

For each block, the report also retains `forced-cold-b` minus the third request
for TTFT, end-to-end wall time, and server-reported prompt-processing time.
This is an observed within-run, order-position delta: in the off profile it is the cold
order control, and in the on profile it is the cold-to-warm observation. It is
not a causal difference-in-differences estimate, a cross-run causal claim, or
evidence that prefix caching changes decode TPS. Any cross-run comparison must
state its time/order controls and matching provenance separately.

The cache journal and exported evidence are scalar-only. They exclude the
synthetic prefix and suffix text, request identifiers, completions, reasoning,
tool payloads, raw Prometheus metrics payloads, commands, paths, and credentials.
Only allowlisted counts, durations, rates, fixed condition labels, validation
state, and public pinned provenance are retained. A failed cache control uses a
fixed safe failure message rather than publishing transport or content details.

The completed Qwen3.6 cache-off/cache-on controls are summarized in the
[2026-08-18 prefix-cache result note](docs/qwen36-prefix-cache-results-2026-08-18.md).
It reports only request-scoped scalar outcomes and keeps prompt-cache effects,
fresh-prefill rates, and decode TPS as separate concepts.

### Harbor terminal coding-agent campaign

The Qwen3-Coder-Next Harbor campaign defines a paired comparison of two
coding-agent clients against one locally served model. Its normative definition
is
[`manifests/campaigns/harbor_terminal_coder_next.toml`](manifests/campaigns/harbor_terminal_coder_next.toml).
The completed two-replicate outcome is reported separately in
[Qwen3-Coder-Next Harbor terminal results](docs/harbor-terminal-results-2026-08-18.md);
do not infer or replace that result from the presence of the manifest alone.
The corrected campaign ID is
`qwen3-coder-next-harbor-terminal-offline-2026-08-18`. Earlier trials under the
2026-08-17 ID are diagnostic only: their verifier upload was not traversable
under the capability-dropped container, so they must not be repaired, regraded,
or combined with a fresh run.

The fixed serving profile is
`qwen3-coder-next-80b-a3b-ud-q4-k-xl-llamacpp`: the exact 49,608,478,720-byte
Unsloth `Qwen3-Coder-Next-UD-Q4_K_XL.gguf` artifact, llama.cpp b10453 source
revision and server-binary digest recorded in `manifests/models.toml`, one
sequence, 65,536 allocated context tokens, an 8,192-token server output cap,
Q8 key/value cache, full GPU offload, flash attention, and no speculative
decoder. The server defaults to temperature 1.0, top-p 0.95, and top-k 40.
Agent clients may send their own generation settings; the bridge does not
rewrite requests. Results therefore describe the complete model-plus-client
stack, not a sampling-controlled comparison of agent prompts alone.

The remaining inputs are pinned in the campaign manifest:

- Harbor 0.21.0 at revision
  `64afbbcb62165950301e1a6407c729aa26d844ff`, executed from the manifest-pinned
  read-only runtime tree that includes its CPython interpreter, virtual
  environment, installed packages, and source;
- Terminal-Bench 2.1 at revision
  `7131e4375048a0e408a8fb404b5f499d726b695b`;
- Qwen Code 0.21.13 and OpenCode 1.18.18, including their npm integrity and
  shasum values, the actual `opencode-linux-arm64` executable package integrity,
  and upstream source revisions; and
- six tasks whose agent and verifier phases are runtime-offline:
  `fix-git`, `cancel-async-tasks`, `fix-code-vulnerability`, `regex-log`,
  `polyglot-c-py`, and `query-optimize`.

The measured lifecycle performs no npm or NVM installation. A separate,
credential-free bootstrap produced normalized read-only Node, Qwen Code, and
OpenCode trees from the exact published distributions. The manifest pins the
complete tree digests and byte counts, the Node executable digest, package
integrities, source revisions, and the ARM64 OpenCode package. Before every
trial, the lifecycle hashes every mounted entry through no-follow file
descriptors and rejects path substitution, unsafe links, hardlinks, special
files, ownership or mode drift, or a changed byte. Custom Harbor agent classes
replace the stock network installers with the admitted read-only prefixes and
verify their versions; they never invoke a downloader or package manager. After
OpenCode agent execution, its custom cleanup is limited to deleting the
ephemeral `xdg-data` and `xdg-state` trees. The retained `opencode.txt` remains
the OpenCode trajectory and metric source. The complete Harbor runtime is
admitted the same way, and commands execute only its verified entry point.

Within each replicate, every task-agent pair runs once, with one active trial,
no retry, and a 900-second agent timeout. The containing Harbor invocation has a
separate 3,600-second wall ceiling covering native image build, agent setup,
agent execution, and verification. The twelve trials use the manifest's fixed
counterbalanced order:
the starting client alternates across tasks so simple warmup or time-order drift
does not consistently favor one client. Harbor must build each task image for
the native ARM64 host instead of pulling an AMD64-only prebuilt image. The
adapter retains each exact built image ID. Pair equivalence is not inferred from
ID equality; it is defined by the campaign's bounded semantic runtime
fingerprint over Linux/ARM64, RootFS layer digests, and runtime `Config`,
excluding the non-runtime `Image` and `Labels` fields. A Qwen Code/OpenCode task
pair is not a valid comparison unless those fingerprints match. A failure or
timeout remains a measured failed attempt; it is not silently retried or
replaced. This small, selected task panel is an exploratory admission screen,
not a broad coding-quality claim.

#### Measured result

Two corrected-ID replicates completed on 2026-08-18. The fixed six-task by
two-client panel produced **1/24 strict passes (4.1667%)**: Qwen Code passed
**1/12 (8.3333%)**, while OpenCode passed **0/12**. Qwen Code earned the sole
reward `1` on `fix-git` in the first replicate; that result did not repeat.
The other five tasks were 0/4 across two clients and two replicates.

All 24 trials finalized. Both campaign envelopes completed their 12 planned
attempts, and the network-admission, native-image, paired-image, and cleanup
failure counters remained zero. The summaries also record zero containing
Harbor wrapper timeouts. One Qwen Code trial separately reached its 900-second
agent timeout and remains a reward-zero `AgentTimeoutError`; do not conflate
that agent outcome with the 3,600-second wrapper counter.

Token telemetry is complete for 12/12 Qwen Code trials but only 5/12 OpenCode
trials because seven OpenCode early exits have no token counts. Token or wall
totals therefore do not support a fair efficiency ranking. The
[full report](docs/harbor-terminal-results-2026-08-18.md) preserves the
per-task, replicate, exception-label, and interpretation detail. Its
[tracked scalar bundle](evidence/campaigns/qwen3-coder-next-harbor-terminal-offline-2026-08-18/manifest.json)
uses schema `sparkbench-harbor-evidence-v1`, pins clean harness commit
`26600d4abe48c082ce6764a61618516837069b9c` and derived-policy digest
`sha256:4749be56af707f6d7615ac5cdb0fb7fa8d50fcdd49e5d4c9a9bfebb71677b4ef`,
and declares that payloads are excluded.

This remains a derived harness-stack result, not an official Terminal-Bench
2.1 score. The six-task selection, one UD-Q4_K_XL model on one machine,
temperature 1.0 defaults, two replicates, client-controlled requests,
transformed offline verifier, and shared-verifier trust boundary all remain
attached to the result. The scalar exception classes are outcomes, not causal
diagnoses; raw payload evidence is neither published nor used to infer one.

#### Execution lifecycle

The outer orchestrator must acquire `hold_campaign_lock(workspace)` before it
starts llama.cpp and retain that single repository lock across the model,
authenticated Unix-socket bridge, every Harbor invocation, and teardown. While
holding the lock, it creates the verified derived dataset and runtime overlay in
an external owner-private cache, follows the manifest's exact `trial_order`,
builds each command with `build_harbor_invocation(...)`, and executes it through
`run_harbor_invocation(...)`. The generated Harbor command fixes Docker
execution, native image building and deletion, one attempt, one concurrent
agent, one trial, zero retries, the selected task, exact custom agent class,
served model, frozen tool prefixes, and phase-specific network policy. Do not
hand-edit that command.

After each invocation, project the external raw job with the adapter's strict
loader and canonical JSON serialization. The fingerprint-bearing campaign
summary and its outer lifecycle envelope both use schema version 2; version 1
records are intentionally incompatible. Derived tasks and raw Harbor jobs must
resolve outside the repository; only the later allowlisted scalar projection is
eligible for the evidence exporter. Cleanup of Harbor containers, bridge,
server, sampler, the derived task copy, and the key file belongs in the
lock-owning `finally` path. The owner-private raw job tree remains ignored local
evidence and must pass an exact ephemeral-key residue scan before cleanup is
certified. An outer convenience CLI must preserve this lifecycle and the frozen
manifest rather than creating a second execution contract.

After the harness commit is clean and every admission input is present, run the
frozen lifecycle with:

```bash
python3 harbor_campaign.py
```

Its defaults resolve the pinned Harbor runtime, tool prefixes, Terminal-Bench
checkout, and owner-private raw/output root outside the repository. Optional
path flags relocate only those exact-verified inputs; they do not change the
model, task order, bridge endpoint, sampling geometry, or lifecycle contract.

#### Isolation and credential boundary

The inference server remains bound to `127.0.0.1`. Its authenticated host bridge
listens only on an owner-private mode-0600 Unix socket and forwards only to that
loopback server. A dedicated, read-only Node relay shares Harbor's egress
sidecar network namespace, listens only on container loopback, and is the sole
container with the socket/key directory mounted. The untrusted task receives a
fixed non-secret placeholder; the relay validates it, substitutes a per-run
internal bearer, and the host bridge validates and strips that bearer before
connecting upstream. The real credential never enters Harbor arguments,
environment, task files, or published evidence. Neither boundary logs headers
or payloads, and both enforce bounded connections, headers, buffers, and
timeouts. Delete the owner-only key and prove the Unix socket is absent after
cleanup, including interruption paths.

Task setup begins with `no-network` because the admitted clients require no
installation. During the agent phase, one atomically updated, permanent
default-drop nftables chain permits only IPv4 loopback TCP to the fixed relay
port; DNS, ICMP, IPv6, raw sockets, the Docker gateway, public addresses, and
all other loopback ports remain blocked. The verifier phase atomically returns
to deny-all, preventing surviving agent children from regaining egress.
Embedded probes certify these transitions in every invocation. The adapter
verifies every byte not deliberately transformed against the pinned source.
Derivation applies the fixed phase policy to task metadata, pins each mutable
base-image tag to one Linux/ARM64 digest, and appends a dedicated Python verifier
environment. It narrowly removes the upstream online `apt`/`curl`/`pip`/`uvx`
bootstrap from each `tests/test.sh`, points the same pytest invocation at the
preinstalled environment, and retains the task assertions and reward logic.
Both the derived `tests/` directory and `test.sh` are mode `0555`: Harbor
directly executes that verifier under `cap_drop: ALL`, so the uploaded copy must
be traversable and executable without adding capabilities. Each final task
image reserves `/tests` as UID/GID 65532 mode `0555`; admission requires the
pinned Compose copy path to populate that foreign-owned directory, and a
fallback upload failure stops the canary. The deterministic
patch digest binds the source and derived Dockerfile, task metadata, test
launcher bytes, and every source/derived mode.

The verifier packages are available without runtime networking, but the task
images are still built through the ordinary Docker builder. Direct verifier
package versions and base-image digests are pinned; transitive Python artifacts
are not hash-locked. Exact semantic image fingerprints therefore establish a
matched pair within this run, not byte-for-byte rebuild determinism. Harbor's
shared verifier also remains a task-harness trust model, not a tamper-resistant
anti-cheat boundary: the verifier runtime is visible to the root agent before
the tests are uploaded. Keep that limitation attached to any result. The model
is never bound to a wildcard or LAN address, and neither task nor inference uses
Docker host networking.

The network-policy, Dockerfile, verifier-bootstrap, and mode transformations
intentionally differ from upstream Terminal-Bench 2.1. Report the result as a
Harbor/Terminal-Bench-derived harness-stack outcome, not an official
Terminal-Bench 2.1 score.

#### Admission and stop gates

Before the measured matrix, require all of the following:

1. exact model, runtime, Harbor, dataset, and agent pins resolve, and the model
   file and runtime binary match their recorded sizes and digests;
2. no unrelated GPU process or running container is present, port 8000 is free,
   and available unified memory is at least the profile's 96 GiB estimate plus
   an 8 GiB reserve;
3. the loopback model passes basic chat, structured-output, and tool-call
   admission, including one valid tool call;
4. the full Harbor runtime and all Node/agent prefix trees match their complete
   immutable admissions, and the relay image is native ARM64;
5. the exact derived dataset is deterministic and failure injection leaves no
   partial tree; one Python 3.13, one Python 3.11, and one Ubuntu-derived image
   certify their pinned verifier runtime under `cap_drop: ALL` and verifier
   deny-all;
6. the public oracle solution earns reward `1` once for every selected task
   through the exact derived Harbor path, with `query-optimize` repeated once
   to screen its timing threshold, and every canary image/container cleans up;
   and
7. authenticated relay/bridge access succeeds, invalid access never reaches
   the model, every forbidden network probe fails, the phase-policy and relay
   assets match their digests, and no raw-payload publication path is enabled.

Abort rather than reinterpret a run when model readiness exceeds its
1,200-second profile timeout, the canary cannot finish within its 3,600-second
containing invocation ceiling and clean up, available memory falls below the
admission reserve, swap grows without
recovering, or bridge/network isolation fails. Stop the campaign after two
consecutive endpoint or chat-template failures. Do not start a new trial at the
23,400-second campaign cutoff; preserve the remaining 5,400 seconds of the
eight-hour window for cleanup, reconciliation, deterministic evidence export,
verification, and documentation. Preserve all completed and failed attempts
when a stop gate fires.

#### Records and publication

Harbor job results, task workspaces, trajectories, prompts, completions,
reasoning, tool payloads, logs, identifiers, commands, environment state, local
paths, and the ephemeral credential remain raw local records under ignored
storage. They must never be copied into Git or quoted in a report. The campaign
adapter may project only its strict allowlist of scalar outcomes and public,
bounded provenance: task and agent labels, terminal status, verifier reward,
token counts, durations, timestamps, version/digest pins, policy-patch digest,
and cleanup/admission booleans. Unknown fields fail closed.

Publication follows [the sanitized evidence workflow](#publishing-sanitized-evidence).
Add campaign evidence only after the exporter supports its exact scalar schema,
an offline synthetic fixture passes, two exports are byte-identical, the archive
verifies, and `verify-evidence --staged` validates and secret-scans the exact Git
index. A report must retain failures and partial states and must label the
one-attempt design, task subset, client-controlled requests, derived network
policy and verifier transformation, shared-verifier trust, non-hash-locked
build dependencies, serving geometry, and absence of a broad quality or
official leaderboard claim.

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

### SM121 cache-policy performance lane

The semantic cache pair and timing comparison are separate claims. The
dedicated `sm121-cache-policy-performance` command is the only execution path
for the admitted SM121 Triton/storage cache-policy profiles. It freezes a
non-resumable A/B/B/A campaign beneath `<results>/cache-policy-campaigns/` and
requires verified target, B0, and paired-semantic scalar evidence before both
freezing and execution. Each arm gets an isolated four-item exact-answer
quality server and a separate timed server, for eight fresh lifetimes with
1,200-second admission deadlines. An observed expiry is rejected and triggers
owned-server interruption before diagnostic cleanup.

Timed observations contain only non-streaming request wall time for cold T0 and
append-only T1/T2. They do not contain or claim TTFT, decode TPS, aggregate
throughput, energy, or agent speed. A result may retain a policy only with an
unrounded at-least-5% later-turn wall-time improvement and a no-more-than-5%
full-sequence wall-time regression. Any failure is terminal `partial` and
`not_evaluated`; it cannot be resumed. See the
[operational protocol](docs/qwen38-flash-next-sm121-cache-performance-protocol-2026-08-29.md)
for the exact profiles, source/evidence binding, audit command, and privacy
boundary. One complete read-only-audited campaign admitted all 12 timing rows
and returned `retain_a`: the cache-on A policy's two-replica mean later-turn
request wall time was 2.808 seconds versus 45.017 seconds for cache-off B, and
its full three-turn mean was 37.151 seconds versus 85.718 seconds. This is a
bounded request-wall result for the exact lane, not a TTFT, TPS, throughput,
energy, or agent-speed claim.

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

Startup capacity stops may also receive an append-only
`annotate-safety-gate` record. The v2 record is deliberately closed and
prose-free: `host_memavailable` uses GiB with a strict `observed < limit`
breach, while `startup_swap_growth` uses MiB with a strict
`observed > limit` breach. Values must be finite and bounded, only one record
per metric is allowed, and the summary publishes the gates in canonical metric
order. Export requires the timestamped journal record, both summary annotation
mirrors, `startup_measurement_valid=false`, and any legacy swap-gate projection
to agree. Tracked evidence retains only metric, observed value, limit, unit and
comparison; it drops the timestamp and all free-form annotation fields.

The first Harbor campaign export requires its two owner-private lifecycle
inputs. Later full refreshes may omit them: the exporter carries the existing
sanitized Harbor bundle forward only after its exact schema and checksums pass,
and fails rather than preserving a malformed or noncanonical bundle.

The intended archive entry points are `evidence/README.md` for people and
`evidence/index.json` for tools. Run bundles retain scalar request measurements,
case aggregates, validation booleans and bounded categories, lifecycle state,
compact numeric telemetry, and reproducibility pins such as artifact hashes,
runtime revisions, image digests, hardware, and harness revision. For supported
SGLang runs, path-free source-overlay basenames and hashes plus the all-or-none
read-only PLE mmap mode, marker digest, and payload digest are retained. PLE-study
bundles additionally retain a versioned boolean omission dimension, with mapped
controls bound to `false` and semantic-ablation arms bound to `true`. The two
known startup-safety failure annotations are projected only through their
strictly typed scalar schemas. Campaign and matrix bundles retain only their
explicitly supported scalar schemas.

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
