# Qwen3.8-Flash-Next on one DGX Spark / GB10 — 2026-08-26

## Result

Qwen3.8-Flash-Next runs on one 128 GB DGX Spark through the provisional
llama.cpp path when the 87.249 GiB Unsloth `UD-IQ4_XS` GGUF uses F16 K/V
cache. The same artifact and runtime aborted during graph construction with
Q8_0 K/V. This is an exact-commit compatibility result, not a general claim
that every Q8 cache implementation is incompatible with the architecture.

The clean eight-slot quick run completed all seven cases. It delivered 19.601
aggregate output tok/s at D128/C1, 31.240 tok/s at C2, and 49.363 tok/s at C4;
the 8K needle passed. The longer core run reached 71.709 aggregate tok/s at C8
and passed all bounded 16K retrieval, JSON, tool-call, and exact-answer checks.
It was terminal but `partial` because the D1024 case produced 4,327 of the
required 5,120 completion tokens, so SparkBench suppressed that case's rate.

The deployment is close to the memory ceiling. The quick run reached 4.270
GiB minimum sampled `MemAvailable` without new swap use. During the core run,
minimum sampled `MemAvailable` reached 4.011 GiB and a live process diagnostic
observed at least 3.85 GiB of `llama-server` `VmSwap` after the 16K prefill
stage. Swap stopped growing and recovered after teardown, but the core numbers
remain memory-pressured exploratory results. The quick run is the cleaner
bounded-admission result.

## Tested configuration

The measured host was one aarch64 DGX Spark / GB10 with 125,508,244 KiB of
unified system memory. Both successful profiles used full GPU offload, CUDA
flash attention, an 8,192-token batch, a 512-token microbatch, Jinja chat
templating, F16 K/V, and no automatic fit adjustment. Thinking was disabled in
both the server and request template. Temperature was zero.

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
  community `RadixArk/Qwen3.8-Flash-Next-NVFP4` artifact, which is 125.91 GiB
  before runtime and K/V—still beyond Spark's approximately 119.7 GiB
  OS-visible memory.
- The [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) requires a
  dedicated image and selects the community
  `Inferact/Qwen3.8-Flash-Next-NVFP4` artifact for its NVFP4 lane. It declares
  130 GB minimum VRAM for that variant, 265 GB for FP8, and 423 GB for BF16. It
  has no memory-safe single-Spark admission.
- The [TokenSpeed recipe](https://lightseek.org/tokenspeed/recipes/models#qwen38-flash-next)
  publishes a TP4 FP8 + MTP3 launch. It has no single-Spark recipe.

Those recipes are useful datacenter references, but they are not alternative
measurements in this report. The smaller target-only GGUF is the only tested
single-Spark path, and its lack of MTP makes its TPS a different deployment.

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

Fetch once, then run without implicit downloads. The runtime binary must exist
at the profile path and match its recorded SHA-256.

```bash
python3 sparkbench.py inventory --sizes
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

| Run ID | Revision | K/V | Terminal state | Published interpretation |
| --- | --- | --- | --- | --- |
| `20260826T163638Z-qwen38-flash-next-ud-iq4-xs-llamacpp-smoke-b76517fb` | `c52212f` clean | Q8_0 | aborted before readiness | exact negative compatibility result |
| `20260826T164557Z-qwen38-flash-next-ud-iq4-xs-llamacpp-smoke-92c5cd3c` | `efabab7` clean | F16 | completed / partial | P1 admission and bounded chat/tool result |
| `20260826T165220Z-qwen38-flash-next-ud-iq4-xs-llamacpp-p8-quick-37477295` | `efabab7` clean | F16 | completed / complete | clean bounded P8 throughput result |
| `20260826T165913Z-qwen38-flash-next-ud-iq4-xs-llamacpp-p8-core-b5a0f9ad` | `efabab7` clean | F16 | completed / partial | memory-pressured core stress result |

Raw run records remain ignored. They contain captured content and local
runtime details and must not be committed.

The four published attempt-scoped scalar bundles are the
[Q8_0 startup failure](../evidence/runs/20260826T163638Z-qwen38-flash-next-ud-iq4-xs-llamacpp-smoke-b76517fb/manifest.json),
[F16 P1 smoke](../evidence/runs/20260826T164557Z-qwen38-flash-next-ud-iq4-xs-llamacpp-smoke-92c5cd3c/manifest.json),
[F16 P8 quick](../evidence/runs/20260826T165220Z-qwen38-flash-next-ud-iq4-xs-llamacpp-p8-quick-37477295/manifest.json),
and [F16 P8 core](../evidence/runs/20260826T165913Z-qwen38-flash-next-ud-iq4-xs-llamacpp-p8-core-b5a0f9ad/manifest.json)
bundles. The full exporter recognizes the prior `loop-*` topology, and the
two exact private Harbor lifecycle inputs needed to preserve the historical
campaign are available locally for explicit use during refresh. Neither raw
source is copied into Git. A complete deterministic re-export with both inputs
and normal archive verification passed, and a second export reported no change;
no hand-selected or hand-merged archive is valid.
