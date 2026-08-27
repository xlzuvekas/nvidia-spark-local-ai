# Qwen3.8-Flash-Next on one DGX Spark / GB10 — 2026-08-26

Updated 2026-08-27 with clean MTP controls, the bounded C8 result, and the
day-one upstream/community boundary.

## Result

The full released Qwen3.8-Flash-Next language stack now runs on one 128 GB DGX
Spark through SGLang with its Radix main model in NVFP4, its 51B-parameter PLE
table retained exactly in FP8 but mapped read-only from NVMe, and its trained
NEXTN module retained in BF16. The terminal native run completed all 12 planned
cases and delivered 28.504 aggregate output tok/s at D256, 27.413 at C1,
50.330 at C2, and 72.821 at C4. Its 8K/32K client-TTFT prefill proxies were
2,103.468/2,179.588 tok/s, and both exact-key needles passed 3/3. The lifecycle
was `completed` but the publication status was `partial` solely because the
synthetic exact-answer battery passed 3/4.

This is not an all-NVFP4 deployment. In particular, the PLE is the released
FP8 table, not an NVFP4 conversion, and the NEXTN module is BF16. The PLE mmap
avoids making the 47.684 GiB table permanently resident, but its pages still
consume unified-memory page cache when touched. The measured synthetic prompts
repeat one word and therefore do not establish cold or varied-token PLE cost.
The later clean D256/C1 confirmation provides a bounded near-matched MTP estimate:
MTP3 reached 30.123639 aggregate output tok/s versus 16.663713 with MTP off,
or `1.807739x`, while saving 137.288 seconds (44.682%) and measuring
`1.821397x` output tokens per sampled joule. A separate authenticated native
audit accepted 175 of 243 proposed tokens (72.0165%). Its counters apply only
to that explicit audit request, not to the preceding streaming workload.

At bounded 4K context, a clean MTP2 profile using SGLang's lazy extra-buffer
recurrent-state strategy measured 114.5755 aggregate output tok/s at offered
C8. That was 48.069% above its matched C4 case while C8 median end-to-end
latency was `1.930421x` its C1 value, clearing the frozen `>=10%`
throughput-retention and `<2x` latency gates. Operator-log inspection observed
eight running requests; the tracked machine evidence is the scalar client
result and telemetry, not an occupancy counter.

The primary native lifecycle started in 581.652 seconds and retained at least
16.564 GiB sampled host `MemAvailable` with no sampled swap growth. That safe
operating claim ends at the measured 32K tier. Two independent 131K C1 needles
passed, but 245K was unsafe: the `.85` C4/MTP allocation rejected it at a
179,514-token pool boundary; `.87` entered severe memory pressure; and the
bounded target-only 245K diagnostic was stopped at 0.046 GiB sampled
`MemAvailable` after the operator observed about 6.1 GiB of swap use and PSI
memory `avg10` of 19.84. The target-only long profile is retained as
`incompatible`, not as a serving recipe.

The earlier provisional llama.cpp path remains a useful target-only comparator.
It runs the 87.249 GiB Unsloth `UD-IQ4_XS` GGUF with F16 K/V, but the converter
omitted MTP. The same artifact and runtime aborted during graph construction
with Q8_0 K/V. This is an exact-commit compatibility result, not a general claim
that every Q8 cache implementation is incompatible with the architecture.

The clean eight-slot quick run completed all seven cases. It delivered 19.601
aggregate output tok/s at D128/C1, 31.240 tok/s at C2, and 49.363 tok/s at C4;
the 8K needle passed. The longer core run reached 71.709 aggregate tok/s at C8
and passed all bounded 16K retrieval, JSON, tool-call, and exact-answer checks.
It was terminal but `partial` because the D1024 case produced 4,327 of the
required 5,120 completion tokens, so SparkBench suppressed that case's rate.

The llama.cpp deployment is close to the memory ceiling. The quick run reached 4.270
GiB minimum sampled `MemAvailable` without new swap use. During the core run,
minimum sampled `MemAvailable` reached 4.011 GiB and a live process diagnostic
observed at least 3.85 GiB of `llama-server` `VmSwap` after the 16K prefill
stage. Swap stopped growing and recovered after teardown, but the core numbers
remain memory-pressured exploratory results. The quick run is the cleaner
bounded-admission result.

## Tested configuration

The measured host was one aarch64 DGX Spark / GB10 with 125,508,244 KiB of
unified system memory. The native SGLang and provisional llama.cpp deployments
were text-only, used temperature zero, and disabled thinking in the request
template. They otherwise differed materially.

| Deployment | Main model | PLE | MTP | Serving geometry | Measured scope |
| --- | --- | --- | --- | --- | --- |
| Native SGLang baseline | released Radix ModelOpt NVFP4 | released FP8, 51,200,245,760-byte read-only NVMe mmap | released BF16 NEXTN, steps 3 / top-k 1 / four total speculative tokens | max running 4; 20 recurrent-state slots; `.85` static fraction; 262,144 declared context | D256, C1/C2/C4, 8K/32K prefill and needles |
| Native SGLang MTP controls | same | same | near-matched MTP3 and off lifetimes | max running 4; 20 recurrent-state slots for MTP3; D256/C1 | clean 20-request estimate plus one separate MTP3 counter audit |
| Native SGLang bounded C8 | same | same | BF16 NEXTN, steps 2 / three total speculative tokens | max running 8; 32 lazy recurrent-state slots; 4,096 context; 32,768 total-token cap | fresh C1/C2/C4/C8 and one separate MTP2 counter audit |
| Provisional llama.cpp | Unsloth `UD-IQ4_XS` GGUF | GGUF-converted representation | omitted by converter | eight slots, each 32,768 context; F16 K/V | target-only D256, C1/C2/C4/C8 and bounded context |

The native path used Triton prefill attention, the pinned SM121 TRT-LLM/XQA
decode overlay, `modelopt_fp4`, 1,024-token chunked prefill, and the exact
released Radix snapshot. The completed PLE file and marker were mounted
read-only. The MTP3/off pair is a near-matched same-runtime control; comparisons to
llama.cpp remain descriptive rather than a backend A/B experiment.

Both successful llama.cpp profiles used full GPU offload, CUDA flash attention,
an 8,192-token batch, a 512-token microbatch, Jinja chat templating, F16 K/V,
and no automatic fit adjustment.

| Profile | Slots | Context per slot | Aggregate allocation | Purpose |
| --- | ---: | ---: | ---: | --- |
| `qwen38-flash-next-ud-iq4-xs-llamacpp` | 1 | 32,768 | 32,768 | admission, chat, JSON, tools |
| `qwen38-flash-next-ud-iq4-xs-llamacpp-p8` | 8 | 32,768 | 262,144 | true parallel-sequence throughput |

The selected GGUF is target-only: it has no exported MTP head and no bundled
vision projector. The measured server reported zero draft tokens. These are
text-only, no-thinking, no-speculation results and are not comparable to an
MTP-enabled serving recipe as if only the backend changed. The P8 aggregate
allocation also does not establish a successful 262K single request.

## Throughput and latency

Aggregate output throughput divides all completion tokens by full measured
case wall time. The per-request decode rate is a client streaming estimate
after first emission; TTFT includes prompt work and queueing.

### Native SGLang full-model suite

| Case | Requests | Aggregate output | Median per-request decode | Median TTFT | Median E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| D256 / C1 | 3 | **28.504 tok/s** | 29.764 tok/s | 0.336 s | 8.904 s |
| C1 | 3 | **27.413 tok/s** | 28.898 tok/s | 0.420 s | 9.257 s |
| C2 | 6 | **50.330 tok/s** | 27.481 tok/s | 0.442 s | 9.820 s |
| C4 | 12 | **72.821 tok/s** | 20.112 tok/s | 0.432 s | 13.414 s |

C4 delivered 2.66 times C1 aggregate throughput while median per-request
decode fell by 30.4%. All four throughput cases were measurement-valid. The
run's 874.043-second managed journal includes the 581.652-second server startup,
the first-use request, all 12 cases, and clean shutdown.

The native prefill figures are client-observed TTFT proxies, not engine prompt
counters:

| Repetition target | Actual prompt tokens/request | Median TTFT | Client-TTFT proxy |
| ---: | ---: | ---: | ---: |
| 8,192 | 8,261 | 3.927 s | **2,103.468 tok/s** |
| 32,768 | 32,835 | 15.065 s | **2,179.588 tok/s** |

### Clean MTP3 versus MTP off

The final near-matched confirmation ran at clean harness revision `2ce8b292`,
with two warmups and 20 measured D256/C1 requests in each separate server
lifetime. All 40 requests completed and each arm produced 5,120 validated
output tokens. MTP3 encoded 1,610 prompt tokens versus 1,590 off, a 1.26%
aggregate input-token mismatch despite the identical nominal shape.
Whole-case throughput is primary because speculative streaming can put
multiple tokens in one event and bias client-estimated per-request decode.

| Arm | Aggregate output | Case wall time | Median E2E | P95 E2E | Sampled output tok/J |
| --- | ---: | ---: | ---: | ---: | ---: |
| [MTP3](../evidence/runs/20260827T194940Z-qwen38-flash-next-nvfp4-mtp-depth3-sglang-qwen38-flash-next-sglang-mtp-depth-confirm-af30d00f/summary.json) | **30.123639 tok/s** | **169.966 s** | 8.344 s | 9.509 s | **0.785612** |
| [MTP off](../evidence/runs/20260827T200256Z-qwen38-flash-next-nvfp4-mtp-depth0-sglang-qwen38-flash-next-sglang-mtp-depth-confirm-aa26aac9/summary.json) | 16.663713 tok/s | 307.254 s | 15.294 s | 15.788 s | 0.431324 |

The resulting `1.807739x` throughput and `1.821397x` energy-efficiency ratios
are bounded to this no-thinking single-stream workload. MTP3 ran first, the two
arms are independent server lifetimes, and this one pair does not estimate
between-lifetime variance.

SparkBench ran a dedicated acceptance audit after the measured MTP3 case. It
used authenticated `/v1/tokenize` followed by native non-streaming `/generate`;
the pinned server rejects `return_meta_info`, but returns the required
`meta_info` automatically. The audit recorded 175 accepted of 243 proposed
draft tokens across 81 verifies, 72.0165% acceptance, position counts 72/55/48,
and mean accepted length 3.16049. Text, token IDs, request identifiers and
unrelated metadata were discarded. The exported scope is explicitly the audit
request only; these are not retroactive counters for the 20 streaming requests.

An earlier five-request forward screen observed 15.9384/26.7568/29.5341/
30.7661 tok/s at depths 0/1/2/3. It ran from a dirty worktree in fixed order,
and depth-one/depth-two startup was swap-contaminated, so it is exploratory.
It shows why off/MTP3 was confirmed; it does not resolve the 4.2% observed
depth-three edge over depth two.

### Bounded MTP2 C8 ladder

The clean lazy-state run
[`9597ea2a`](../evidence/runs/20260827T193218Z-qwen38-flash-next-nvfp4-mtp2-c8-lazy-sglang-qwen38-flash-next-sglang-c8-9597ea2a/summary.json)
used depth two, a 4,096-token context cap, a 32,768-token total pool, eight
decode graphs and 32 lazy extra-buffer recurrent states. It measured:

| Fresh shape | Requests | Aggregate output | Median E2E | Sampled output tok/J |
| --- | ---: | ---: | ---: | ---: |
| C1 | 3 | 28.7930 tok/s | 9.010 s | 0.8061 |
| C2 | 6 | 48.2511 tok/s | 10.819 s | 1.3110 |
| C4 | 12 | 77.3798 tok/s | 13.180 s | 2.0950 |
| C8 | 24 | **114.5755 tok/s** | 17.393 s | **3.0928** |

All 24 C8 requests completed 256 validated tokens each. C8 retained +48.069%
aggregate throughput over C4 and held median end-to-end latency to
`1.930421x` C1, passing the frozen `>=10%` and `<2x` gates. During the C8 case,
minimum sampled `MemAvailable` was 16.450 GiB. Operator inspection of the server
log observed eight running requests, no queue and CUDA-graph execution; those
occupancy fields are not promoted to machine evidence. The separate audit
accepted 159/192 proposed tokens (82.8125%), scoped only to its audit request.

The ordinary 32-state `extra_buffer` arm completed offered C8 at 80.5772 tok/s,
but operator-log inspection showed a six-running/two-queued split. It is not a
simultaneous-eight execution result. Increasing that strategy to 40 states was
safety-stopped before measurement when swap growth reached 602.48 MiB, above
the frozen 512 MiB limit. An MTP3/40-state attempt independently crossed the
14 GiB host-availability floor during graph capture. Both are capacity
rejections, not model crashes; the lazy 32-state MTP2 profile is the retained
bounded C8 route.

### Fair local GGUF comparison

The closest descriptive comparator uses the same nominal D256 and C1/C2/C4
shapes from the llama.cpp core run. The native run used three repetitions per
shape while the llama.cpp core used five. Both requested 256 output tokens per
request, but runtime, artifact, quantization, cache, PLE, MTP, and memory state
all differ.

| Shape | Native SGLang | llama.cpp GGUF | Descriptive ratio |
| --- | ---: | ---: | ---: |
| D256 | 28.504 tok/s | 20.193 tok/s | 1.41x |
| C1 | 27.413 tok/s | 19.860 tok/s | 1.38x |
| C2 | 50.330 tok/s | 19.782 tok/s | 2.54x* |
| C4 | 72.821 tok/s | 51.927 tok/s | 1.40x |
| 8K prefill proxy | 2,103.468 tok/s | 674.500 tok/s | 3.12x |

The 8K prefill prompts were especially close in realized size: 24,783 total
native prompt tokens across three requests versus 24,786 for llama.cpp. The
llama.cpp C2 value is an outlier that failed to improve on C1 in its
memory-pressured core lifecycle, so the 2.54x ratio is not a stable scaling
claim. Excluding that discontinuity, native D256/C1/C4 aggregate throughput was
1.38-1.41 times the GGUF comparator. None of these ratios isolates SGLang,
NVFP4, FP8 PLE, or NEXTN, and none is authoritative evidence of MTP speedup.

### Clean quick suite

| Case | Requests | Aggregate output | Median per-request decode | Median TTFT | Median E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| D128 / C1 | 3 | 19.601 tok/s | 20.910 tok/s | 0.447 s | 6.507 s |
| C2 | 4 | 31.240 tok/s | 18.265 tok/s | 0.633 s | 4.083 s |
| C4 | 8 | 49.363 tok/s | 14.973 tok/s | 0.950 s | 5.160 s |

The quick run also passed its one 8,284-token needle request, with 13.335-second
TTFT and 13.933-second E2E. Its managed journal wall was 271.166 seconds,
including 43.801 seconds of artifact validation and 90.149 seconds of server
startup, but excluding the CLI's preceding plan/fingerprint phase.

### Memory-pressured core suite

| Case | Requests | Aggregate output | Median per-request decode | Median TTFT | Median E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| D256 / C1 | 5 | 20.193 tok/s | 20.928 tok/s | 0.438 s | 12.605 s |
| C1 | 5 | 19.860 tok/s | 20.818 tok/s | 0.475 s | 12.712 s |
| C2 | 10 | 19.782 tok/s | 10.244 tok/s | 0.903 s | 25.794 s |
| C4 | 20 | 51.927 tok/s | 13.627 tok/s | 1.104 s | 19.773 s |
| C8 | 40 | **71.709 tok/s** | 9.764 tok/s | 2.153 s | 28.337 s |

C8 delivered 3.61 times C1 aggregate throughput while reducing median
per-request decode by 53.1%. C2 did not improve aggregate throughput in this
longer-output suite, whereas it did in the quick suite. Output length, runtime
state, and the observed swap pressure all differ, so the run does not isolate
the cause of that discontinuity.

The core journal wall was 1,341.781 seconds, including 43.842 seconds of
artifact validation and 96.143 seconds of startup. All managed requests
finalized, the server stopped cleanly, and memory recovered after teardown.

### Prefill proxy

SparkBench estimates prefill rate from client-observed TTFT. This is not an
engine-native prompt-evaluation counter and includes request and scheduler
overhead.

| Repetition target | Actual prompt tokens/request | Median TTFT | Client-TTFT proxy |
| ---: | ---: | ---: | ---: |
| 128 | 194 | 0.585 s | 331.688 tok/s |
| 1,024 | 1,090 | 2.054 s | 530.584 tok/s |
| 4,096 | 4,164 | 6.850 s | 607.909 tok/s |
| 16,384 | 16,452 | 26.606 s | 618.363 tok/s |

The planned 32K prefill was skipped because its estimated 32,909-token request
exceeded the 32,768-token per-slot limit.

## Bounded validation

### Native SGLang

| Check | Outcome | Boundary |
| --- | --- | --- |
| Chat smoke | 1/1 pass | 32-token bounded generation |
| Strict JSON smoke | 1/1 pass | one fixed structured-output fixture |
| Tool-call smoke | 1/1 pass | one fixed tool fixture |
| Synthetic exact answers | 3/4 pass | caused the otherwise completed run to publish as `partial` |
| 8K needle | 3/3 pass | repeated-word exact-key fixture |
| 32K needle | 3/3 pass | repeated-word exact-key fixture; safe primary-run boundary |
| 131K needle, C4/MTP profile | 1/1 pass | 131,171 prompt tokens; 67.419-second TTFT |
| 131K needle, C1/MTP profile | 1/1 pass | 131,169 prompt tokens; 72.285-second TTFT |
| 245K needle | no publishable pass | rejected or operator-stopped at the memory/swap safety boundary |

The two 131K successes are bounded retrieval observations, not a supported
131K serving envelope. They came from aborted lifecycles whose subsequent 245K
case either could not enter the KV pool or was stopped. The declared 262,144
model context is therefore a model/runtime maximum, not a safe one-Spark
allocation. The 245K target-only diagnostic is negative safety evidence even
though it removed NEXTN and capped the requested KV allocation.

### Provisional llama.cpp

| Check | Outcome | Boundary |
| --- | --- | --- |
| P1 chat smoke | 1/1 pass | 32-token bounded generation |
| P1 strict JSON smoke | 0/1 | valid object was Markdown-fenced; formatting-contract failure |
| P1 tool-call smoke | 1/1 pass | one fixed tool fixture |
| P8 8K needle | 1/1 pass | exact key-presence fixture |
| P8 16K needle | 3/3 pass | exact key-presence fixture |
| P8 core JSON | 5/5 pass | fixed structured-output fixture |
| P8 core tool call | 5/5 pass | fixed tool fixture |
| P8 exact answers | 4/4 pass | four synthetic deterministic prompts |
| P8 D1024 | invalid | 4,327/5,120 requested completion tokens; rate suppressed |

The smoke and core cases used the same generic `json_object` response format.
The pinned llama.cpp parser did not convert that empty format into a grammar:
the one smoke response was fenced, while all five measured core responses
were bare valid JSON. No validator was weakened and no fenced response was
reinterpreted as a pass. This difference is evidence that the formatting
contract is not fully reliable, not evidence that the core path used stronger
schema enforcement. These fixtures are capability gates, not a broad quality
score.

The llama.cpp PR author reports that its QSA approximation can diverge above
the sparse 2,048-token budget. Passing three 16K needle probes is useful but
does not establish general long-context equivalence or quality.

## Historical deployment anchors

The table uses the same core-suite shapes, but it is descriptive rather than a
causal architecture comparison. Historical runs used llama.cpp b10453, Q8_0
K/V, different GGUF quantizations and dirty repository states. Flash Next used
the provisional `qwen4exp` runtime, F16 K/V, and a clean harness. Laguna S had
only one serving slot, so its C2/C4/C8 cells measured queued service rather
than parallel sequence scaling.

| Deployment | D256 | C1 | C2 | C4 | C8 | P128 | P1K | P4K | P16K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B Q4, P8 | 56.937 | 56.475 | 54.488 | 140.182 | 184.899 | 933.9 | 1,777.5 | 2,292.1 | 2,472.5 |
| **Qwen3.8-Flash-Next IQ4_XS, P8** | **20.193** | **19.860** | **19.782** | **51.927** | **71.709** | **331.7** | **530.6** | **607.9** | **618.4** |
| Dense Qwen3.8 27B Q4, P8 | 10.413 | 10.423 | 10.318 | 34.169 | 53.567 | 395.1 | 632.6 | 723.4 | 721.7 |
| Laguna S 118B-A8B Q4, P1 | 22.817 | 22.736 | 22.770 | 22.742 | 22.713 | 413.4 | 786.7 | 1,083.6 | 1,140.0 |

Flash Next decoded 1.34–1.94 times faster than the historical dense Qwen3.8
27B control across the valid generation cells, while its prefill proxy was
14.3–16.1% slower. Qwen3.6 remained 2.58–4.00 times faster across these
generation and prefill observations. Flash Next single-stream decode was also
slower than Laguna S, but its real P8 geometry scaled aggregate throughput
while Laguna's one-slot requests queued. Model scale, active width, runtime,
quantization, cache type, and memory pressure all prevent attributing these
differences to MoE alone.

The historical source records are the [MoE landscape](moe-landscape-2026-08-17.md)
and the [2026-08-16 benchmark report](benchmark-results-2026-08-16.md).

## Q8_0 failure and F16 workaround

The first clean smoke attempt used Q8_0 K/V. Artifact validation passed, but
the server exited before readiness while building the QSA graph:

```text
qwen4exp.cpp:544: GGML_ASSERT(inp->self_k_rot == nullptr && inp->self_v_rot == nullptr) failed
```

The failed run was bound to repository commit `c52212f`, exact runtime commit
`035e2273`, and the same three model shards later used successfully. Commit
`efabab7` changed the two Flash Next profiles to F16 K/V. The next P1 smoke and
both P8 runs loaded successfully. This demonstrates a working workaround for
the pinned stack; it does not prove a universal root cause or validate a
different llama.cpp revision.

## Artifact and runtime pins

### Native SGLang

The native model is the released
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594)
snapshot at revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`: 206
weight files and 135,195,303,851 indexed weight bytes (125.910 GiB). No model
tensor was requantized for the measured run. The main routed experts use the
checkpoint's ModelOpt NVFP4 representation, the complete PLE table remains
FP8, and the embedded NEXTN tensors remain BF16.

The 51,200,245,760-byte PLE table was materialized offline into one exact
file-backed mmap, then admitted read-only. Its payload SHA-256 is
`b070f9644adf93794d8a1030584ab705809387e64396a9327a68fa3a3a6666b3`; the
completed-layout marker SHA-256 is
`f0ef55e4e4dec9b6b936a42af4ca2eb9b2f24ced373b1e216f7a6d507b171665`.
Startup checks the immutable marker, payload size and pinned payload identity;
the full payload was hashed when it was built and during explicit verification,
not on every server start.

The aarch64 runtime image is
`lmsysorg/sglang@sha256:14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4`.
The public recipe is
[`hashd1ve/qwen38-flash-next-one-dgx-spark`](https://huggingface.co/hashd1ve/qwen38-flash-next-one-dgx-spark/tree/bf2b7c75870d3703730b6bd8f3bb93dc622c278d)
at `bf2b7c75870d3703730b6bd8f3bb93dc622c278d`. Two generated, read-only
runtime overlays are pinned independently:

| Overlay | SHA-256 | Purpose |
| --- | --- | --- |
| `qwen4_exp.py` | `0b513b4dc4f2394f6b1733bb0b74fa40ab59f4a04f6b33601350b2a606c67804` | persistent read-only PLE mmap loader |
| `qwen_sparse_attn_backend.py` | `e30566492e1502f94a4c7fed42d90b523bbb662580c628459e6e63c7b5263c75` | SM121 QSA decode routing |

The primary run used clean harness revision
`717b17c3150072f6cbc8d0cc5861c489af92d8bd`. These pins reproduce the exact
measured stack; they do not claim tagged upstream GB10 support.

### Provisional llama.cpp

The measured artifact is the immutable
[Unsloth UD-IQ4_XS listing](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/2c41bd2a0b3f51c503c11f1c7ed2e6bb34036beb/UD-IQ4_XS),
revision `2c41bd2a0b3f51c503c11f1c7ed2e6bb34036beb`.

| Shard | Bytes | SHA-256 |
| --- | ---: | --- |
| `00001-of-00003` | 10,946,624 | `5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4` |
| `00002-of-00003` | 49,835,229,856 | `577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7` |
| `00003-of-00003` | 43,836,407,744 | `d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833` |

The total is 93,682,584,224 bytes, or 87.248706 GiB. Artifact validation
recomputed all three hashes before each measured server lifetime.

The runtime is the open, unmerged [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)
at commit [`035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`](https://github.com/ggml-org/llama.cpp/commit/035e22731a7fd70b9854b3a2d64ec68e9b1a45d3).
The measured `llama-server` SHA-256 is
`6b0e09f19768e1424eac29b27d6d7f5ca661a9f73b5b7a2ecba5e768af8a366a`.
The branch's [converter](https://github.com/ggml-org/llama.cpp/blob/035e22731a7fd70b9854b3a2d64ec68e9b1a45d3/conversion/qwen4exp.py)
explicitly disables MTP export. This is provisional support, not an upstream
release claim.

## Official recipe fit on one Spark

The [official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/f5d08274bafd880402bd16f5e3e6c514136ec06c)
describes a 125B main model with 6B active parameters per token, plus 51B of
n-gram/PLE embeddings and a 4B MTP component. The official BF16 checkpoint is
335.276 GiB and the [official FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/bcd9f01ddc9cff2316eb84281bebcd5b058bddce)
is 172.782 GiB, so neither admits on one Spark.

- The [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
  has no GB10 lane. Its NVIDIA BF16/FP8 cells use TP4, its AMD BF16/FP8 cells
  use TP8, and NVFP4 TP1 is limited to B200/B300/GB300. The recipe selects the
  same 125.91 GiB community `RadixArk/Qwen3.8-Flash-Next-NVFP4` artifact tested
  here. It cannot be resident in Spark's approximately 119.7 GiB OS-visible
  memory as published. The local run admitted it by retaining the exact
  47.684 GiB FP8 PLE on NVMe and demand-paging it through the pinned overlay;
  that is a measured local extension, not a cookbook GB10 lane.
- The [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) requires a
  dedicated image and currently documents the official FP8 checkpoint on four
  GB300 GPUs or eight H200 GPUs. Its host-memory PLE offload does not by itself
  solve Spark's unified-memory capacity bound. Model support
  [PR #53896](https://github.com/vllm-project/vllm/pull/53896) and PLE-offload
  [PR #53899](https://github.com/vllm-project/vllm/pull/53899) remained open at
  the 2026-08-27 review cutoff, and the recipe has no validated one-GB10 NVFP4
  lane. A community NVMe-mmap extension is reviewed separately in the
  [day-one literature report](qwen38-flash-next-gb10-day-one-2026-08-27.md).
- The [TokenSpeed recipe](https://lightseek.org/tokenspeed/recipes/models#qwen38-flash-next)
  publishes a TP4 FP8 + MTP3 launch. It has no single-Spark recipe.

Those recipes are useful datacenter references, but they are not alternative
measurements in this report. The file-backed-PLE SGLang result is the only
measured full-model/MTP single-Spark path here; the smaller target-only GGUF is
a different deployment. The earlier
[native NVFP4/I4 + MTP plan](qwen38-flash-next-native-mtp-optimization-2026-08-26.md)
identified PLE capacity as the fit lever and proposed quantization. The final
experiment instead preserved the released FP8 PLE exactly on NVMe. It closes
bounded admission and throughput through 32K, but the 245K pressure result
still rules out treating the native context maximum as a safe Spark envelope.

## GLM-5.3-Flash disposition

GLM-5.3-Flash was deferred, not benchmarked. Z.ai describes it as a
[320B-total/18B-active multimodal MoE](https://z.ai/blog/glm-5.3-flash).
Its [official FP8 checkpoint](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a)
contains 305.788 GiB of safetensors, and the
[official vLLM recipe](https://github.com/vllm-project/recipes/blob/8bb447dc1f6e937afae0af777e53b3e452977ee5/models/zai-org/GLM-5.3-Flash.yaml)
lists a 386 GB minimum VRAM requirement. No GGUF weights were present in the
[Unsloth WIP repository](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/49599e06c57b68347ac9f1034df254bb0aa8030b)
at the 2026-08-26 16:55 UTC cutoff.

The available [llama.cpp PR #27752](https://github.com/ggml-org/llama.cpp/pull/27752)
was still open and unmerged, text-only, without an MTP graph, and had not been
tested on real weights or numerically validated against the Hugging Face
implementation. A same-day update wired its DSA indexer after the initial
audit, but did not clear those validation or artifact blockers. Neither the
artifact nor runtime path was therefore suitable for a memory-admissible,
valid long-context Spark benchmark.

## Reproduce

Fetch once, then run without implicit downloads. For the native path, prepare
the two digest-pinned overlays from the cached image and materialize and verify
the exact PLE file before launching the server:

```bash
python3 sparkbench.py inventory --sizes
python3 sparkbench.py fetch qwen38-flash-next-nvfp4-mtp-sglang
python3 -m bench.sglang_overlay_prepare
python3 -m bench.sglang_overlay_prepare --materialize-ple
python3 -m bench.sglang_overlay_prepare --verify-ple-cache
python3 sparkbench.py benchmark qwen38-flash-next-nvfp4-mtp-sglang \
  --suite manifests/suites/qwen38_flash_next_sglang_native.toml
```

Do not reproduce the incompatible 245K target-only profile as a normal
benchmark. It crossed the stop boundary and is retained only so the negative
configuration remains auditable.

For llama.cpp, the runtime binary must exist at the profile path and match its
recorded SHA-256:

```bash
python3 sparkbench.py fetch qwen38-flash-next-ud-iq4-xs-llamacpp
python3 sparkbench.py benchmark qwen38-flash-next-ud-iq4-xs-llamacpp \
  --suite manifests/suites/smoke.toml
python3 sparkbench.py benchmark qwen38-flash-next-ud-iq4-xs-llamacpp-p8 \
  --suite manifests/suites/quick.toml
python3 sparkbench.py benchmark qwen38-flash-next-ud-iq4-xs-llamacpp-p8 \
  --suite manifests/suites/core.toml
```

Pass a Hugging Face credential only through `HF_TOKEN` if the source requires
one. Run one inference configuration at a time and preserve loopback serving.

## Run and publication ledger

### Native persistent-PLE attempts

This sequence includes the initial writable mmap prototypes and every
subsequent verified-read-only attempt. Every row came from a clean harness
worktree; aborted rows remain negative or diagnostic evidence, not measurements
silently merged into the final run.

| Run ID | Harness | Geometry | Terminal state | Evidence interpretation |
| --- | --- | --- | --- | --- |
| [`20260827T002718Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-b879c795`](../evidence/runs/20260827T002718Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-b879c795/manifest.json) | `ef764bc3` | writable PLE mmap, `.79`, MTP3/C4 | startup operator-abort; 0 cases | first persistent-mmap load attempt only |
| [`20260827T003820Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-245ea9e3`](../evidence/runs/20260827T003820Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-245ea9e3/manifest.json) | `a46b7004` | writable PLE mmap plus checkpoint cache drop, `.79`, MTP3/C4 | startup operator-abort; 0 cases | load/cache diagnostic only |
| [`20260827T012359Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-005c4fd6`](../evidence/runs/20260827T012359Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-005c4fd6/manifest.json) | `cc687c66` | verified read-only PLE, `.79`, MTP3/C4 | startup abort; 0 cases | API-key argument admission failure |
| [`20260827T012647Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-005c4fd6`](../evidence/runs/20260827T012647Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-005c4fd6/manifest.json) | `20bbd957` | verified read-only PLE, `.79`, MTP3/C4 | startup abort; 0 cases | default recurrent cache admitted zero requests; five state slots are required per request |
| [`20260827T013808Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-fa899717`](../evidence/runs/20260827T013808Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-fa899717/manifest.json) | `d386b478` | verified read-only PLE, `.85`, MTP3/C4 | operator-abort after 5/14 cases | first full-model bounded request passes; incomplete throughput run |
| [`20260827T015017Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-a06b138a`](../evidence/runs/20260827T015017Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-a06b138a/manifest.json) | `67e00627` | `.85`, 20 recurrent slots, MTP3/C4 | abort after 13/14 cases | 131K pass; 245K rejected at 179,514-token pool limit |
| [`20260827T020849Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-8f0a58d7`](../evidence/runs/20260827T020849Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-8f0a58d7/manifest.json) | `8ed2df2c` | planned `.87`, 20 recurrent slots, MTP3/C4 | preflight abort; 0 cases | 116.0 GiB model-plus-reserve estimate exceeded 114.9 GiB available |
| [`20260827T020950Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-c80757e7`](../evidence/runs/20260827T020950Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-c80757e7/manifest.json) | `08fc7430` | `.87`, 20 recurrent slots, MTP3/C4 | operator-abort after 13/14 cases | 131K pass; 245K entered unsafe memory/swap pressure |
| [`20260827T024144Z-qwen38-flash-next-nvfp4-mtp-long-sglang-qwen38-flash-next-sglang-long-context-7c25f743`](../evidence/runs/20260827T024144Z-qwen38-flash-next-nvfp4-mtp-long-sglang-qwen38-flash-next-sglang-long-context-7c25f743/manifest.json) | `d2f7aca7` | `.85`, five recurrent slots, MTP3/C1 | operator-abort after 1/2 cases | independent 131K pass, 72.285-second TTFT; no 245K result |
| [`20260827T025734Z-qwen38-flash-next-nvfp4-long-sglang-qwen38-flash-next-sglang-long-context-b8e9080e`](../evidence/runs/20260827T025734Z-qwen38-flash-next-nvfp4-long-sglang-qwen38-flash-next-sglang-long-context-b8e9080e/manifest.json) | `a14b3aee` | target-only `.85`, one recurrent slot, C1 | startup abort; 0 cases | one slot was below the five-slot request floor |
| [`20260827T030636Z-qwen38-flash-next-nvfp4-long-sglang-qwen38-flash-next-sglang-long-context-7b88e52c`](../evidence/runs/20260827T030636Z-qwen38-flash-next-nvfp4-long-sglang-qwen38-flash-next-sglang-long-context-7b88e52c/manifest.json) | `d0e53f45` | target-only `.85`, five BF16 recurrent slots, 246,272-token cap, C1 | safety abort during only case | 0.046 GiB sampled minimum; about 6.1 GiB observed swap and PSI memory `avg10` 19.84; profile retired |
| [`20260827T032027Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-20e1283b`](../evidence/runs/20260827T032027Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-20e1283b/manifest.json) | `717b17c3` | `.85`, 20 recurrent slots, MTP3/C4; 8K/32K cap | `completed` / `partial`; 12/12 cases | primary safe bounded run; only publication failure was exact answers 3/4 |

### Native MTP and C8 optimization attempts

The first four-arm depth matrix was an explicitly exploratory dirty-worktree
screen at commit `6778586`, in fixed 0/1/2/3 order. Its 15.9384, 26.7568,
29.5341 and 30.7661 tok/s observations are retained, but the screen is not a
publication-quality depth ranking and its depth-one/depth-two startups were
swap-contaminated. The final off/MTP3 pair and C8 ladder below ran from clean
revisions.

| Run | Harness | Geometry | Terminal state | Evidence interpretation |
| --- | --- | --- | --- | --- |
| [`a4336a0f`](../evidence/runs/20260827T183826Z-qwen38-flash-next-nvfp4-mtp-c8-sglang-qwen38-flash-next-sglang-c8-a4336a0f/manifest.json) | `35337d6` clean | MTP3, 40 ordinary states, planned C8 | preflight abort; 0 cases | conservative model-plus-reserve estimate rejected before Docker |
| [`617007f4`](../evidence/runs/20260827T183912Z-qwen38-flash-next-nvfp4-mtp-c8-sglang-qwen38-flash-next-sglang-c8-617007f4/manifest.json) | `efceae4` clean | MTP3, 40 ordinary states, planned C8 | startup safety abort; 0 cases | host availability crossed below 14 GiB during graph capture; capacity rejection |
| [`a8f54d30`](../evidence/runs/20260827T185155Z-qwen38-flash-next-nvfp4-mtp2-c8-sglang-qwen38-flash-next-sglang-c8-a8f54d30/manifest.json) | `5948de1` clean | MTP2, 32 ordinary states, offered C8 | completed; 6/6 cases | 80.5772 tok/s offered C8; operator-log observation showed six running/two queued |
| [`85e3ddfb`](../evidence/runs/20260827T192011Z-qwen38-flash-next-nvfp4-mtp2-c8-sglang-qwen38-flash-next-sglang-c8-85e3ddfb/manifest.json) | `e3e719b` clean | MTP2, 40 ordinary states, planned C8 | startup safety abort; 0 cases | 602.48 MiB swap growth exceeded frozen 512 MiB gate |
| [`9597ea2a`](../evidence/runs/20260827T193218Z-qwen38-flash-next-nvfp4-mtp2-c8-lazy-sglang-qwen38-flash-next-sglang-c8-9597ea2a/manifest.json) | `2ce8b29` clean | MTP2, 32 lazy states, C8 | completed; 6/6 cases | retained C8 arm, 114.5755 tok/s; scalar gates passed and all-eight-running was observed only in operator logs |
| [`af30d00f`](../evidence/runs/20260827T194940Z-qwen38-flash-next-nvfp4-mtp-depth3-sglang-qwen38-flash-next-sglang-mtp-depth-confirm-af30d00f/manifest.json) | `2ce8b29` clean | MTP3, D256/C1, 20 requests | completed | 30.123639 tok/s plus separate 175/243 native audit |
| [`aa26aac9`](../evidence/runs/20260827T200256Z-qwen38-flash-next-nvfp4-mtp-depth0-sglang-qwen38-flash-next-sglang-mtp-depth-confirm-aa26aac9/manifest.json) | `2ce8b29` clean | MTP off, D256/C1, 20 requests | completed | near-matched 16.663713 tok/s control |

### Provisional llama.cpp attempts

| Run ID | Revision | K/V | Terminal state | Published interpretation |
| --- | --- | --- | --- | --- |
| `20260826T163638Z-qwen38-flash-next-ud-iq4-xs-llamacpp-smoke-b76517fb` | `c52212f` clean | Q8_0 | aborted before readiness | exact negative compatibility result |
| `20260826T164557Z-qwen38-flash-next-ud-iq4-xs-llamacpp-smoke-92c5cd3c` | `efabab7` clean | F16 | completed / partial | P1 admission and bounded chat/tool result |
| `20260826T165220Z-qwen38-flash-next-ud-iq4-xs-llamacpp-p8-quick-37477295` | `efabab7` clean | F16 | completed / complete | clean bounded P8 throughput result |
| `20260826T165913Z-qwen38-flash-next-ud-iq4-xs-llamacpp-p8-core-b5a0f9ad` | `efabab7` clean | F16 | completed / partial | memory-pressured core stress result |

Raw run records remain ignored. They contain captured content and local
runtime details and must not be committed.

The final native [scalar summary](../evidence/runs/20260827T032027Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-20e1283b/summary.json)
and the target-only [245K safety summary](../evidence/runs/20260827T030636Z-qwen38-flash-next-nvfp4-long-sglang-qwen38-flash-next-sglang-long-context-7b88e52c/summary.json)
contain no captured prompts or completions. The four published llama.cpp
attempt-scoped scalar bundles are the
[Q8_0 startup failure](../evidence/runs/20260826T163638Z-qwen38-flash-next-ud-iq4-xs-llamacpp-smoke-b76517fb/manifest.json),
[F16 P1 smoke](../evidence/runs/20260826T164557Z-qwen38-flash-next-ud-iq4-xs-llamacpp-smoke-92c5cd3c/manifest.json),
[F16 P8 quick](../evidence/runs/20260826T165220Z-qwen38-flash-next-ud-iq4-xs-llamacpp-p8-quick-37477295/manifest.json),
and [F16 P8 core](../evidence/runs/20260826T165913Z-qwen38-flash-next-ud-iq4-xs-llamacpp-p8-core-b5a0f9ad/manifest.json)
bundles. The full exporter recognizes the prior `loop-*` topology, and the
two exact private Harbor lifecycle inputs remain outside Git. After validating
its canonical schema and checksums, the exporter carried the already-sanitized
Harbor campaign forward without reopening those inputs. At this publication
cutoff the deterministic archive contains 1,835 files and indexes 285 run
bundles. Normal archive verification passed, and a second complete export
reported no change; no hand-selected or hand-merged archive is valid.
