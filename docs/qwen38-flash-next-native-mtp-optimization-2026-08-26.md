# Qwen3.8-Flash-Next native SGLang result and MTP/PLE optimization — 2026-08-26

Updated 2026-08-27 with the clean MTP3/off confirmation, native scalar counter
audit, and bounded MTP2 C8 state-cache result.

**Safety supersession, 2026-08-28:** every SGLang measurement in this report
used the digest-pinned SM121 TRT-LLM overlay later restricted after
varied-token corruption. The checkpoint was admitted only within that measured
historical runtime; neither it nor these profiles is admitted for new
inference. Preserve the results as within-runtime evidence. New work requires a
newly built, pinned, and admitted SM121 Triton route and fresh rebaseline; see
the [day-two safety review](qwen38-flash-next-gb10-day-two-delta-2026-08-28.md).

## Decision

Keep the measured 87.249 GiB `UD-IQ4_XS` GGUF as a target-only local
comparator. Its converter omitted MTP, so it cannot isolate native SGLang,
ModelOpt NVFP4, PLE placement, or `NEXTN` effects.

The released
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594)
checkpoint historically admitted and ran on one DGX Spark. The measured route did not
make its approximately 125.91 GiB tensor payload resident: it keeps the main
routed experts in NVFP4, keeps the trained `NEXTN` module in BF16, and maps the
exact 51,200,245,760-byte FP8 PLE table read-only from NVMe. The PLE is FP8, not
NVFP4, and the file mapping is private/read-only rather than a pinned host copy.

The primary run
[`20e1283b`](../evidence/runs/20260827T032027Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-20e1283b/summary.json)
completed all 12 planned cases. Its evidence status is `partial` only because
the bounded quality battery scored 3/4; JSON, tools, 8K and 32K needle cases,
and all throughput cases completed. Aggregate output throughput was 28.504
tok/s for D256, then 27.413, 50.330, and 72.821 tok/s at fresh C1, C2, and C4.
The client-TTFT prefill approximations were 2,103.468 tok/s at 8K and 2,179.588
tok/s at 32K.

This closed full-checkpoint admission only within the pinned historical
runtime, and the clean follow-up gives a bounded single-stream MTP estimate.
Across separate near-matched D256/C1
lifetimes, MTP3 delivered 30.123639 aggregate output tok/s versus 16.663713
with MTP off: `1.807739x` throughput, 137.288 seconds or 44.682% less measured
case wall time, and `1.821397x` sampled output tokens per joule. The MTP3
lifetime's separate native audit accepted 175 of 243 proposed draft tokens
(72.0165%). Those counters apply only to that explicit audit request, not
retroactively to the 20 measured requests.

A separate 131K repeated-word needle case passed, while 245K was
pressure-unsafe. NVFP4/I4 PLE packing is therefore future memory/headroom
optimization rather than the prerequisite for native admission. At bounded
4K context, the clean MTP2 lazy-state profile also reached 114.5755 aggregate
output tok/s at offered C8; operator-log inspection observed all eight requests
running with no queue. That was the historically retained short-context
throughput profile;
it is not evidence that the long-context MTP3 allocation can support eight
simultaneous requests.

## Why PLE is the fit lever

The official configuration has `ngram_size = 3`, eight heads per n-gram, and a
2,560-element PLE embedding. The
[reference implementation](https://github.com/huggingface/transformers/blob/36deb0b53ed0863f4b4dfdea23dcaec7f3df3701/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1018-L1114)
constructs 16 hashed lookup heads. Each head returns 160 values, so one token
logically gathers 16 rows and 2,560 table values. The table contains roughly
320 million rows but only about 2.5 KiB of FP8 values are selected per token.
It is therefore an unusually good quantization or tiered-storage target: its
capacity is large while its per-token read set is small.

The Radix checkpoint quantizes the 48 main-model routed-expert layers to NVFP4,
but retains the PLE table in FP8 and all 31 MTP tensors in BF16. Safetensors
metadata gives the following approximate storage split:

| Component | Stored precision | Payload |
| --- | --- | ---: |
| Main routed experts | NVFP4 weights and scales | 63.318 GiB |
| PLE n-gram table | FP8 plus a scalar scale | 47.684 GiB |
| Complete MTP module | BF16 | 4.856 GiB |
| Main non-expert text | mostly BF16 | 9.155 GiB |
| Vision | BF16 | 0.836 GiB |
| Other PLE tensors | BF16 | 0.061 GiB |

The repository is larger than the sum of tensor payloads because it also
contains safetensors headers and qualification files. These figures remain fit
estimates; the measured route separately binds the exact checkpoint revision,
PLE byte count and hashes. Any future packed derivative must publish its own
exact file sizes and hashes.

### Candidate PLE representations

| PLE strategy | Estimated PLE storage/residency | Estimated text-only weight residency | Gross Spark headroom | Disposition |
| --- | ---: | ---: | ---: | --- |
| Fully resident FP8 table | 47.684 GiB | about 125.1 GiB | negative | Reject as a fully resident one-Spark layout |
| Exact file-backed FP8 | 47.684 GiB on NVMe, demand-paged | about 77.4 GiB plus runtime/cache | about 42.3 GiB before runtime/cache | Implemented and measured native baseline |
| NVFP4, group 16 | about 26.82 GiB | about 104.2 GiB if resident | about 15.5 GiB | Future capacity/locality optimization; existing reference gather to fuse and wire |
| Row-scaled I4 | about 24.44 GiB | about 101.8 GiB if resident | about 17.9 GiB | Secondary future experiment; needs a new loader/kernel |
| PLE omitted | none | about 77.4 GiB | about 42.3 GiB | Runtime/MTP smoke only; changes model semantics |

The NVFP4 estimate includes packed 4-bit values and one FP8 block scale per 16
values. It is approximately 2.4 GiB larger than a one-scale-per-row I4 design.
SGLang already has a correctness-oriented ModelOpt unpack/dequant method for
the NVFP4 representation, although Qwen4 does not wire it and the method is not
yet a fused production PLE kernel. That is now an optimization path beyond the
measured exact-FP8 baseline, not an admission dependency. Extending the known
layout remains a better first packed-PLE experiment than defining a second I4
format for a small additional saving.

The estimated 15.5 GiB gross headroom in a fully resident
PLE-NVFP4/BF16-MTP form would still be marginal.
The MTP routed experts alone contain about 2.517B BF16 parameters, or 4.688
GiB. Packing only those experts to group-16 NVFP4 reduces them to approximately
1.318 GiB and puts the text-only weight estimate near 100.8 GiB, leaving about
18.9 GiB gross. With its remaining BF16 tensors, the complete optimized MTP
module is about 1.487 GiB. MTP projections, norms, attention, and head remain
BF16. A weight-only W4A16 form is the preferred first MTP quantization if the
serialized loader can express it because it avoids inventing activation
calibration for the draft layer; otherwise the Inferact-style W4A4 layout needs
calibration and acceptance validation.

### K/V budget

The target has 12 full-attention layers, two K/V heads, and a 256-element head
dimension. A BF16 target K/V estimate is 24,576 bytes per cached token: about
0.75 GiB at 32K, 1.5 GiB at 64K, 3 GiB at 128K, and 6 GiB at 262K for one
sequence. This excludes the MTP draft cache, GDN state, QSA index state,
allocator slack, and runtime memory. FP8 can roughly halve the target K/V
payload, but it is an optimization after a BF16 correctness baseline, not part
of the first admission claim.

NVFP4 K/V is explicitly out of the initial matrix. An open
[SGLang issue](https://github.com/sgl-project/sglang/issues/36010) reports a
native-MTP failure in the NVFP4 K/V extend path on Qwen3.8-27B. That report is
not proof of a Flash-Next defect, but it is sufficient reason not to combine
two experimental dimensions in the first correctness run.

## Native artifact audit

| Artifact | Immutable revision | Repository bytes | MTP/PLE disposition | One-Spark result |
| --- | --- | ---: | --- | --- |
| Qwen BF16 | [`f5d0827`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/f5d08274bafd880402bd16f5e3e6c514136ec06c) | 335.28 GiB | BF16 MTP; BF16 PLE | Reject |
| Qwen FP8 | [`bcd9f01`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/bcd9f01ddc9cff2316eb84281bebcd5b058bddce) | 172.76 GiB indexed tensors | MTP retained; FP8 PLE | Reject |
| Radix NVFP4 | [`7b71922`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594) | 125.96 GiB | Main routed experts NVFP4; BF16 MTP; exact FP8 PLE mapped read-only | Admitted and measured through persistent NVMe PLE tiering |
| Inferact NVFP4 | [`103a760`](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4/tree/103a7608316173ca6edd49929544244de7ffda70) | 170.19 GiB indexed tensors | Main and MTP routed experts NVFP4; 95.368 GiB BF16 PLE | Reject |
| Unsloth GGUF | [`2c41bd2`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/2c41bd2a0b3f51c503c11f1c7ed2e6bb34036beb/UD-IQ4_XS) | 87.249 GiB selected quant | Converter omitted MTP | Target-only comparator |

Two newly indexed repositories named as NVFP4 contained only a few kilobytes of
metadata at the 2026-08-26 18:18 UTC cutoff. They are placeholders, not weight
alternatives. MLX and GGUF MTP conversions are useful implementation references
but are not native CUDA/ModelOpt serving artifacts for this experiment.

## Runtime choice

### SGLang first

The official
[SGLang Flash-Next cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
selects the Radix checkpoint for NVFP4. Its dedicated image is multi-architecture;
the exact Linux aarch64 platform image pulled and tested on the GB10 is:

```text
lmsysorg/sglang:qwen38flashnext@sha256:14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4
```

The image identifies its SGLang base as commit
[`d91c368`](https://github.com/sgl-project/sglang/commit/d91c3682b0b429e4c70df63cd57f819588ce29b0),
with additional image overlays. Open
[SGLang PR #36497](https://github.com/sgl-project/sglang/pull/36497) at
[`73a2552`](https://github.com/sgl-project/sglang/commit/73a255206f916366c8d26d4022f82ddfb0ab558d)
is useful support lineage, but it is not the image's reported build commit.
No tagged or PyPI release contains that support. The cookbook validates NVFP4
on B200/B300/GB300, not GB10, so the image digest and local diagnostics below
are reproducibility evidence rather than a general GB10 support claim.

#### Measured GB10 diagnostics

The immutable image passed a package, model-registration, and hardware import
probe with SGLang `0.0.0.dev1+gd91c3682b`, PyTorch `2.13.0+cu130`, CUDA runtime
`13.0`, FlashInfer `0.6.17`, and an NVIDIA GB10 reported as compute capability
`[12, 1]`. The Qwen4 target, Qwen4 `NEXTN` draft, and ModelOpt NVFP4 embedding
classes all imported. This probe proved only that the aarch64 package surface
was present; full-checkpoint admission was established later and still does not
prove every native kernel on SM121.

| Diagnostic | Measured result | Boundary |
| --- | --- | --- |
| ModelOpt NVFP4 embedding primitive | Pass; CUDA output matched an independent E2M1/block-scale reference with maximum absolute error `0` | Isolated embedding method, not the Qwen4 PLE loader or a fused production kernel |
| FlashInfer GDN | BF16 failed on SM121; retaining FP32 for the GDN path was required | Conditional runtime pass, not a BF16 GDN support claim |
| Stock native QSA resolver | Fail during CuTe MLIR compilation on SM121 | The pinned SM121/XQA overlay below is required for the measured route |
| Real tiny checkpoint, QSA-disabled semantic control | HTTP `200`; 4 prompt tokens to 8 completion tokens in `1.759583 s`; model load `1.71 s` | Bounded request/response and real-weight execution control only |
| Dummy target plus `NEXTN` MTP | Both target and MTP loaded; HTTP `200` with 16 completion tokens in `0.338387 s`; 15 proposed, 15 verified, 0 accepted | Synthetic weights; proves draft execution and counters, not useful speculation |
| Full Radix NVFP4 checkpoint | Admitted through exact read-only FP8 PLE tiering; all 12 primary cases completed | Main model NVFP4 + BF16 `NEXTN` + FP8 PLE, not an all-NVFP4 or fully resident result |

Both server controls used the pinned
[`inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B`](https://huggingface.co/inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B/tree/5fbd297b1529cfa7db2510896d1ad77d1bf41e44)
fixture at revision `5fbd297b1529cfa7db2510896d1ad77d1bf41e44`.
It is a small development architecture fixture, not a quality model. Its
`layer_types` used the stale `qwen_sparse_attention` label; the local runtime
copy corrected that value to the official `full_attention` label. The same
copy set the root `language_model_only` field to `true` for SGLang's text-only
loader, and the real-weight control disabled QSA after the native QSA kernel
failed. Those are fixture/runtime workarounds, not modifications to the pinned
cached snapshot or evidence that the full artifact works unchanged.

The fixture intentionally omits actual MTP tensors. Consequently the
real-weight control did not exercise `NEXTN`; the separate `--load-format
dummy` control instantiated synthetic target and MTP tensors from the fixture
shapes. Its zero accepted drafts are expected diagnostic evidence, not a model
quality or speed result. Neither request timing is representative TPS, and the
two timings must not be compared as a performance experiment.

#### Clean SparkBench controls

The two admissible controls were then rerun sequentially through SparkBench
from clean harness revision
`d50c75799dd00122c39f0d26b28f7344f67828c4`. Both runs used the exact image and
fixture pins above, completed their terminal lifecycle, stopped the owned
container, and exported only sanitized scalar evidence:

| Control | Run | Applicable smoke result | First-request boundary |
| --- | --- | --- | --- |
| Real fixture, QSA disabled, no MTP | [`30d30d00`](../evidence/runs/20260826T190843Z-qwen38-flash-next-tiny-qsa-disabled-sglang-smoke-30d30d00/summary.json) | 118 prompt and 32 completion tokens; median TTFT `0.018570 s`, median end-to-end `0.157423 s`, client-estimated decode `223.258 tok/s` | TTFT `22.231822 s`, dominated by first-use compilation |
| Dummy target and dummy `NEXTN`, QSA disabled | [`931e5c58`](../evidence/runs/20260826T190953Z-qwen38-flash-next-tiny-dummy-nextn-sglang-smoke-931e5c58/summary.json) | 117 prompt and 32 completion tokens; median TTFT `0.671786 s`, median end-to-end `0.897854 s`, client-estimated decode `137.127 tok/s` | TTFT `16.331661 s`, also a first-use path |

These are one-request admission smokes, not a matched throughput comparison:
the weights differ, the prompt-token counts differ, the model is a development
fixture, and the speculative stream was emitted in one-token chunks. The
OpenAI-compatible response and current SparkBench SGLang adapter did not expose
native acceptance counters, so the exported `speculative_decoding` aggregate is
`null`. The earlier direct `/generate` diagnostic exposed the synthetic
15-proposed, 15-verified, 0-accepted counters; those counters must not be
silently attributed to the clean smoke runs.

#### Full-checkpoint measured result

The successful profile is
`qwen38-flash-next-nvfp4-mtp-sglang` from `manifests/models.toml`. It binds:

- Radix checkpoint revision
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594` and public recipe revision
  `bf2b7c75870d3703730b6bd8f3bb93dc622c278d`;
- image digest
  `14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4`;
- 51,200,245,760-byte PLE payload SHA-256
  `b070f9644adf93794d8a1030584ab705809387e64396a9327a68fa3a3a6666b3`
  and deterministic marker SHA-256
  `f0ef55e4e4dec9b6b936a42af4ca2eb9b2f24ced373b1e216f7a6d507b171665`;
- `qwen4_exp.py` overlay SHA-256
  `0b513b4dc4f2394f6b1733bb0b74fa40ab59f4a04f6b33601350b2a606c67804`
  and QSA overlay SHA-256
  `e30566492e1502f94a4c7fed42d90b523bbb662580c628459e6e63c7b5263c75`.

The measured server used TP1, ModelOpt FP4, Triton prefill, TRT-LLM MHA
decode, 1,024-token chunked prefill, `mem-fraction-static=0.85`, four running
requests, a 20-slot recurrent cache, and the official `NEXTN` recipe with depth
three, top-k one, and four total speculative tokens. PLE mmap was explicit
`readonly`; startup rejected
nondefault checkpoint-loader modes and the runtime mount was read-only.

Startup took 581.652 seconds: target loading accounted for 420.36 seconds and
trained MTP loading for 83.86 seconds. Sampled available memory never fell
below 16.564 GiB during the primary run and swap did not grow. The result table
uses completed, validated output tokens over full case wall time:

| Case | Result |
| --- | ---: |
| D256, C1 | 28.504 tok/s |
| Fresh C1 | 27.413 tok/s |
| Fresh C2 | 50.330 tok/s |
| Fresh C4 | 72.821 tok/s |
| Repeated-word 8K prefill | 2,103.468 tok/s, client-TTFT approximation |
| Repeated-word 32K prefill | 2,179.588 tok/s, client-TTFT approximation |
| 8K / 32K exact needle | Pass / pass |
| Bounded quality | 3/4; code-reasoning item failed |

The 8K and 32K prompts repeat one synthetic word. They demonstrate serving
mechanics and exact-key retention, not natural-document quality, adversarial
retrieval, or worst-case cold/varied-token NVMe PLE behavior.

Periodic server logs contained 30 acceptance samples with mean accepted length
2.956 and mean acceptance rate 0.653. They show that the trained draft was
active, but they are operator-log observations rather than authoritative
machine-readable lifetime or case aggregates. They are not used for the
speedup estimate or retroactively converted into counters.

#### Clean MTP-off confirmation and scoped native counters

The final confirmation ran from clean harness revision `2ce8b292` in two
separate server lifetimes, MTP3 first and MTP off second. Both arms used the
same pinned target, image, overlays, read-only PLE payload, no-thinking request,
D256/C1 case, two warmups and 20 measured repetitions. Every measured request
completed and each arm produced 5,120 validated output tokens. MTP3 encoded
1,610 prompt tokens versus 1,590 off, a 1.26% aggregate input-token mismatch,
so this is near-matched rather than token-identical. Whole-case aggregate
throughput is primary because SGLang may bundle multiple speculative tokens in
one stream event, biasing the client's event-timed per-request decode estimate.

| Clean arm | Requests / output | Aggregate output | Case wall time | Sampled output tok/J |
| --- | ---: | ---: | ---: | ---: |
| [MTP3](../evidence/runs/20260827T194940Z-qwen38-flash-next-nvfp4-mtp-depth3-sglang-qwen38-flash-next-sglang-mtp-depth-confirm-af30d00f/summary.json) | 20 / 5,120 | **30.123639 tok/s** | **169.966 s** | **0.785612** |
| [MTP off](../evidence/runs/20260827T200256Z-qwen38-flash-next-nvfp4-mtp-depth0-sglang-qwen38-flash-next-sglang-mtp-depth-confirm-aa26aac9/summary.json) | 20 / 5,120 | 16.663713 tok/s | 307.254 s | 0.431324 |

MTP3 therefore measured `1.807739x` the MTP-off aggregate rate, or +80.7739%.
It saved 137.288 seconds, or 44.682% of case wall time, and measured
`1.821397x` the sampled output tokens per joule. This is a bounded
single-stream estimate from a near-matched control, not a guarantee for other
prompts, context lengths,
sampling policies, or concurrency levels. The two independent server
lifetimes also leave between-lifetime variance unestimated.

After the measured MTP3 case, SparkBench ran one dedicated scalar-only
acceptance audit. It authenticated an explicit `/v1/tokenize` request, sent the
result to native non-streaming `/generate`, validated the returned counters,
and discarded generated text, output token IDs, request identifiers and all
unallowlisted metadata. The pinned build returns `meta_info` automatically and
rejects a `return_meta_info` request field, so the audit deliberately omits
that field. Its 81 verify calls proposed exactly 243 tokens at configured depth
three; 175 were accepted, for 72.0165% draft acceptance, position counts
72/55/48, and mean accepted length 3.16049 including the verified target token.
The scope is `explicit_sglang_native_audit_requests_only`: it proves that the
trained draft executed and its counters were internally consistent for that
audit request, but supplies no counters for the preceding streaming workload.

The earlier forward depth screen remains exploratory because all four arms ran
from dirty commit `6778586` in fixed order 0, 1, 2, 3, with five D256 requests
per lifetime. Its observed aggregate rates were 15.9384, 26.7568, 29.5341 and
30.7661 tok/s. Depth one captured most of the gain, depth two added a smaller
increment, and depth three was only 4.2% above depth two. Startup swap changed
during the depth-one/depth-two sequence, including roughly 2.2 GiB by the
depth-two startup, so the sweep is not suitable for memory or fine-grained
depth-two-versus-three claims. It motivated the clean off/MTP3 confirmation;
it does not replace one, and it does not establish depth three as better than
depth two.

#### Bounded C8 state-cache result

The concurrency ladder used MTP2 because the dirty sweep left the small
MTP3-over-MTP2 difference unresolved and because MTP2 requires fewer recurrent
states. All retained C8 measurements are 4K-context, no-thinking, fresh-short
D256 requests in one lifecycle; they do not expand the 131K single-request
boundary.

| Recurrent-state strategy | Configured cache | Admission/result | Interpretation |
| --- | ---: | --- | --- |
| `extra_buffer` | 32 states | completed; offered-C8 80.5772 tok/s | Operator-log observation showed at most six running and two queued, so this is not a true-eight-running result |
| `extra_buffer` | 40 states | safety-rejected before measurement | Swap grew 602.48 MiB, above the frozen 512 MiB ceiling, despite 18.19 GB engine reserve; sampled host availability remained above the separate 14 GiB floor |
| `extra_buffer_lazy` | 32 states | completed; **offered C8 114.5755 tok/s** | Operator-log observation showed eight running, queue zero and graph execution; scalar client result passed the retention gates |

The clean lazy-state run
[`9597ea2a`](../evidence/runs/20260827T193218Z-qwen38-flash-next-nvfp4-mtp2-c8-lazy-sglang-qwen38-flash-next-sglang-c8-9597ea2a/summary.json)
measured 28.7930, 48.2511, 77.3798 and 114.5755 aggregate output tok/s at
fresh C1/C2/C4/C8. C8 was 48.069% above C4, clearing the prespecified 10%
retention gate, while median end-to-end latency was 17.393 seconds versus
9.010 seconds at C1, a `1.930421x` ratio below the `2x` ceiling. All 24 C8
requests completed 256 tokens each and validated. During the C8 case, minimum
sampled `MemAvailable` was 16.450 GiB. Its separate MTP2 audit accepted 159 of 192
proposals (82.8125%), but that value again applies only to the audit request.

The non-lazy 32-state run is useful negative geometry evidence: its client
offered eight requests and recorded 80.5772 tok/s, but the server log was
observed to execute six while queuing two. Log-derived occupancy is not part of
the machine evidence archive. Increasing the same strategy to 40 states was
stopped at the frozen swap-growth gate, while an earlier MTP3/40-state attempt
was stopped when host `MemAvailable` crossed below 14 GiB during graph capture.
These are pressure-gated capacity rejections, not model crashes. The lazy
32-state profile was therefore the historically retained C8 arm; its
all-eight-running designation is
an operator-log observation, not a tracked occupancy counter.

Long-context escalation established a narrower boundary. The 131K
repeated-word needle completed and validated in
[`7c25f743`](../evidence/runs/20260827T024144Z-qwen38-flash-next-nvfp4-mtp-long-sglang-qwen38-flash-next-sglang-long-context-7c25f743/summary.json),
although that diagnostic was stopped before a terminal success. At 245K, the
0.85 pool was insufficient; a 0.87 experiment created unacceptable pressure.
The target-only BF16-state profile with a 246,272-token cap still aborted in
[`7b88e52c`](../evidence/runs/20260827T030636Z-qwen38-flash-next-nvfp4-long-sglang-qwen38-flash-next-sglang-long-context-7b88e52c/summary.json):
sampled `MemAvailable` reached 0.046 GiB, while the operator observed roughly
6.1 GiB of swap growth and memory PSI full `avg10` of 19.84. That profile is
retained as `incompatible`; 245K is not a supported one-Spark result.

#### Historical SM121 QSA admission, superseded for new runs

Read-only inspection of the exact image isolates the SM121 fallback. QSA's
`_resolve_trtllm_sparse_decode()` admits only `is_sm100_supported()`. GB10 fails
that guard; classic FlashAttention 2 is absent, so the resolver selects the
installed FlashAttention 4 CuTe path that failed compilation. The generic
`--attention-backend` option cannot override this because SGLang replaces it
with `QwenSparseAttnBackend` for the hybrid GDN/QSA architecture.

The installed FlashInfer `0.6.17` XQA implementation already admits compute
capability major 12, page size 64, and QSA's supported head geometry. The
historical local experiment therefore admitted
`is_sm120_supported()` alongside the SM100 check and let FlashInfer's existing
auto dispatch select XQA on SM120/SM121.

That exact two-line change was then mounted read-only over the pinned image.
On compute capability 12.1, the resolver selected
`flashinfer.decode.trtllm_batch_decode_with_kv_cache`. With native QSA retained,
the real tiny fixture completed a 4-to-8-token request in `7.234597 s` including
first-use compilation, then a warm 6-to-32-token request in `0.278661 s`; both
returned HTTP `200`. This closes the observed CuTe blocker for this fixture,
and the same resolver route subsequently served the measured full checkpoint.
That is local runtime evidence, not upstream SM121 support. An independent
SM12x packed-XQA parity test and pinned derived-image rerun remain prerequisites
for an upstream support claim.
The exact source hashes, patch, and boundaries are preserved in the
[experimental SGLang SM121 XQA guide](../patches/sglang/README.md).

Subsequent multi-trial varied-token testing found corruption on the SM121
TRT-LLM route, and upstream restriction `99c9362` returned it to exact SM120.
Open [SGLang PR #36845](https://github.com/sgl-project/sglang/pull/36845) now
provides an explicit SM121 Triton packed-varlen fallback directly atop that
restriction. Therefore the two-line local patch is retained only to reproduce
the measured historical image; do not use it for a new build or support claim.
New work must preserve the exact SM120 TRT-LLM restriction, use an explicitly
attested SM121 Triton fallback, and pass varied-token long-context validation.
See the
[day-two review](qwen38-flash-next-gb10-day-two-delta-2026-08-28.md) for the
exact ancestry and evidence boundary.

The remaining packed-PLE path is promising because the measured image imported
the relevant class,
and the support-lineage source at PR commit `73a2552` contains a
[`ModelOptNvFp4EmbeddingMethod`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/quantization/modelopt_quant.py#L633-L728)
that gathers packed NVFP4 embedding rows, expands their E2M1 values and group
scales, and emits BF16. It currently uses PyTorch indexing and unpacking rather
than a fused PLE kernel. The measured overlay privately maps the exact FP8
table and retains the existing overlapped gather schedule; the packed NVFP4
path must integrate with that schedule rather than be assumed to work already.

Four integration gaps remain before replacing the measured exact-FP8 PLE with
a packed NVFP4 PLE:

1. Qwen4 constructs the PLE `VocabParallelEmbedding` without a quantization
   configuration or quantization prefix, so the existing ModelOpt NVFP4 method
   is never selected for this table.
2. The Qwen4 loader currently accepts `shard_N.weight` PLE tensors in FP8/BF16,
   while an NVFP4 PLE needs packed `weight`, row/block `weight_scale`, and a
   global `weight_scale_2`. The shard loader must copy all three without ever
   materializing the full table in BF16.
3. The unpack/dequant reference should become a fused packed-U8 gather that
   emits BF16 and is validated first at TP1 on SM121.
4. SGLang intentionally disables quantization for an embedded MTP module in a
   serialized ModelOpt checkpoint. A derived checkpoint needs an explicit,
   fail-closed MTP quantization declaration before the MTP expert tensors may be
   constructed as W4A16/NVFP4. The BF16-MTP variant does not need this fourth
   change.

Do not treat `--ple-offload-embedding` alone as the fit mechanism. The measured
route depends on the digest-pinned persistent mmap overlay and read-only mount;
the stock pinned-host wrapper would consume the unified capacity that tiering
is meant to preserve. A future NVFP4 PLE requires its own packed loader/kernel.

Native MTP uses the bundled head; no sidecar draft model is needed. The measured
SGLang baseline used:

```text
--speculative-algorithm NEXTN
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

The checkpoint contains one physical MTP layer. Steps 1/2/3 repeatedly apply
that layer during speculation; they are not three separately trained heads.
The exploratory forward screen tested off and steps 1/2/3; the clean
confirmation then established the MTP3-versus-off effect. It did not resolve
the small depth-three-versus-depth-two difference, so depth three remains the
recipe baseline rather than a proven optimum. The bounded C8 arm therefore
uses depth two to reduce recurrent-state demand.

### vLLM second

The official [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)
has a clean built-in MTP interface and an aarch64 image:

```text
vllm/vllm-openai:qwen38-flash-next@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Its `VLLM_PLE_CPU_OFFLOAD=1` path helps a discrete GPU with separate host RAM,
but not a single Spark UMA capacity bound. The recipe's published NVFP4
checkpoint is also larger than the Radix source. vLLM remains valuable after a
fit artifact exists because SparkBench already parses its proposed/accepted
MTP counters; it is not the first checkpoint-construction target. Flash-Next
support remains in open [vLLM PR #53896](https://github.com/vllm-project/vllm/pull/53896),
and PLE offload remains a separate open
[PR #53899](https://github.com/vllm-project/vllm/pull/53899). The inspected
image labels do not identify an exact vLLM source commit, so the image would
need to be rebuilt from a pinned revision before it met this repository's
provenance contract.

A day-one [community one-Spark vLLM mmap patch](https://github.com/blazux/qwen3.8-Flash-DGX/tree/82ed48d373d8a2c03d142d203f07bce0a6b69125)
reports roughly 17 tok/s off and 27 tok/s with MTP2, plus 2,400-2,660 tok/s
prefill. Its code repository and base-image digest are pinned, but its model
download is not revision-pinned and it publishes no persisted repeated
measurement or power/memory bundle. It is a valuable next A/B target, not a
measurement in this report. The complete source audit and safe reproduction
requirements are in the
[GB10 day-one literature review](qwen38-flash-next-gb10-day-one-2026-08-27.md).

The current [TokenSpeed recipe](https://lightseek.org/tokenspeed/recipes/models#qwen38-flash-next)
publishes FP8 TP4 plus MTP3 and has no native single-Spark NVFP4/I4 route.

## Historical build sequence

The completed admission path was:

1. pin and acquire the exact Radix revision and aarch64 SGLang image;
2. reproduce the SM121 QSA resolver overlay from the pinned public recipe;
3. concatenate and verify all 128 FP8 PLE shards without constructing the
   model, then bind the immutable payload and deterministic completion marker;
4. add the audited persistent-cache loader overlay, mount the cache read-only,
   and retain PLE `weight_scale` loading while skipping only the 128 table
   copies;
5. load the NVFP4 target and trained BF16 MTP under the bounded 0.85/C4 profile;
6. complete the 12-case native suite, then escalate repeated-word context to
   the validated 131K boundary;
7. add the authenticated scalar-only native acceptance audit and run the MTP
   depth screen; and
8. confirm MTP3 against MTP off from a clean revision, then complete offered C8
   at bounded 4K context with the MTP2 lazy recurrent-state strategy.

The pre-supersession optimization sequence was deliberately narrower. It is
retained as historical planning context, not authorization to execute the old
SGLang profiles:

1. **Depth-two versus depth-three replication.** The clean off/MTP3 result
   establishes a bounded MTP gain, but the dirty five-request sweep cannot
   resolve its approximately 4% depth-three edge over depth two. Use clean,
   counter-audited replicated lifetimes before choosing between them outside
   the current recipe/C8 roles.
2. **Quality confirmation.** Investigate the failed code-reasoning item and run
   a larger fixed battery without changing runtime settings.
3. **Memory-pressure boundary.** Treat 131K as the historical route's upper
   bounded retrieval observation.
   Do not rerun the incompatible 245K profile until the projected and observed
   reserve close without swap or PSI pressure.
4. **Packed PLE optimization.** Extend the zero-error group-16 primitive through
   the Qwen4 shard loader and a fused packed gather. Compare resident NVFP4 PLE
   and exact file-backed FP8 PLE for correctness, cold/warm locality, startup,
   and headroom.
5. **Only if justified, quantize MTP experts.** Preserve the non-expert MTP path
   in BF16 and require acceptance/quality gains to exceed conversion risk.

## Optimization matrix after admission

Change one axis at a time and restart the server between context tiers.
The seed is the successful TP1, exact-FP8-PLE, BF16-MTP, depth-3, 0.85/C4
profile above; it is a measured baseline, not a proven optimum.

| Axis | Screening values | Rule |
| --- | --- | --- |
| MTP | off; steps 1, 2, 3 | Compare target-verified effective tok/s and acceptance by position |
| FP4 backend | `auto`, then available cuDNN and CUTLASS candidates | Prove SM121 execution in eager mode first; pin the winner |
| CUDA graph | eager, batch 1, then batch 2/4 | Do not diagnose a graph failure and a model failure together |
| Chunked prefill | 1,024; 2,048; 4,096 | Official recipe's 4,096 is a candidate, not a GB10 optimum |
| Context | 8K, 32K, 64K, 128K; 245K held | Fresh lifetime per tier; do not retry 245K until the memory equation closes |
| Concurrency | 1, 2, 4, 8 | Stop when memory, queueing, or per-request latency defeats aggregate gain |
| K/V | BF16, then calibrated/validated FP8 | No NVFP4 K/V in the first matrix |
| Prefix cache | off, then on for the winning MTP depth | Measure cold establishment and exact warm reuse separately |
| Reasoning | fixed no-thinking speed screen; one fixed reasoning-effort quality arm | Never mix reasoning effort inside a TPS comparison |

Primary performance is completed, validated output tokens divided by full case
wall time. Also retain TTFT, TPOT, aggregate output throughput, startup time,
minimum available memory, swap delta, power, proposed/accepted MTP tokens,
acceptance by position, and cache-hit counters. Engine-emitted tokens that fail
the validator are not usable throughput.

The completed first depth screen used D256/C1 with two warmups and five
repetitions, but its dirty state and fixed order make it exploratory. The clean
20-repetition confirmation established MTP3 versus off, and the clean lazy
MTP2 ladder completed bounded offered C8 with all-eight-running observed in the
operator log. A future depth-two/depth-three
confirmation should use replicated or counterbalanced lifetimes. Streaming and
non-streaming parity must still be checked below and above 32K because
speculative output corruption can otherwise look like a speedup.

## Acceptance gates

A native result is publishable only when all applicable gates pass:

- the exact artifact and aarch64 image revisions are immutable and hashed;
- the served model loads without implicit download, swap growth, or unrelated
  GPU/container work;
- PLE is present for quality runs; the exact-FP8 route binds payload and marker
  digests, while any future packed route must also match the reference gather;
- an MTP-configured result may be published as counter-unverified, but any MTP
  speedup claim requires an MTP-off control plus authoritative proposed,
  verified, and accepted-token aggregates;
- where acceptance is claimed, `0 <= accepted_tokens <= proposed_tokens`, with
  finite per-position rates;
- deterministic chat, exact-answer, JSON, tool-call, retrieval, and streaming
  parity checks pass at the tested boundary;
- the same reasoning, sampling, cache, context, and concurrency settings are
  used inside each causal comparison; and
- conclusions distinguish measured admission/performance from causal backend,
  MTP, precision, or PLE-placement comparisons.

The GGUF comparison remains descriptive: quantization, runtime, PLE precision,
and MTP all differ. It is a product target, not a backend-only causal control.

## Repository state after full-checkpoint admission

SparkBench now supports a separate authenticated SGLang acceptance audit. It
uses `/v1/tokenize` plus native `/generate`, retains only validated scalar
counts/rates/histograms, and fails the lifecycle closed when required counters
are absent or inconsistent. The pinned build rejects `return_meta_info`; its
native final response includes the counters automatically. The implementation
requires `accepted <= proposed`, the configured number of proposed tokens per
verify call, a histogram whose accepted-token sum matches the total, and finite
rates. It discards response text, token IDs, request IDs, reasoning and
unallowlisted metadata.

That audit remains deliberately separate from the OpenAI streaming workload.
Its exported scope says `explicit_sglang_native_audit_requests_only`, so the
repository does not manufacture retroactive counters for measured cases.
Periodic `/metrics` or server-log acceptance gauges remain activity
cross-checks only; they are not cumulative proposed/accepted-token evidence.

The remaining protocol work is to add clean replicated depth-two/depth-three
confirmation, isolated 16K/32K/64K/128K lifetimes, streaming/non-streaming
parity at longer context, and a separate prefix-cache cold/warm protocol.
Persist only sanitized scalar evidence—never prompts, completions, reasoning,
tool payloads, request identifiers, local paths, commands or raw logs.

The manifest retains the explicitly labeled tiny/dummy diagnostics, the
measured full-checkpoint profiles, the MTP controls, and the bounded lazy C8
arm. Keep every route pinned to the exact FP8 PLE payload/marker and overlay
digests. None may be relabeled as tagged upstream GB10 support; the 245K
profile remains incompatible.
