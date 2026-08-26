# Qwen3.8-Flash-Next native NVFP4/I4 + MTP diagnostics and plan — 2026-08-26

## Decision

Keep the measured 87.249 GiB `UD-IQ4_XS` GGUF as a target-only local
comparator. It is useful for bounded quality, wall-time, and throughput targets,
but its converter omitted the MTP module, so it cannot answer whether native
NVFP4/I4 plus MTP is faster.

No released native CUDA checkpoint currently admits on one DGX Spark. The
smallest complete candidate is the 125.96 GiB repository / approximately
125.91 GiB tensor payload in
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594).
The measured host exposes only 128,520,441,856 bytes, or 119.694 GiB, of total
unified system memory. The checkpoint therefore exceeds physical memory before
Python, CUDA, serving state, graphs, K/V cache, or the operating system are
counted. Moving its PLE table from device to pinned host memory does not change
that capacity result on Spark because both placements consume the same physical
memory.

The first credible native route is a derived, text-only ModelOpt checkpoint:

1. retain the Radix routed experts in NVFP4 W4A4;
2. pack the 51B-parameter PLE table as NVFP4 and dequantize only gathered rows;
3. retain the MTP non-expert path in BF16;
4. try the MTP routed experts in BF16 first, then W4A16/NVFP4 only if the first
   form cannot admit; and
5. serve through the provisional SGLang Qwen4 path with native `NEXTN` MTP.

This is still a build and benchmark plan for the full model, not a measured
native Flash-Next performance result. The exact SGLang day-zero image and a
pinned 0.2B development fixture were subsequently downloaded and exercised.
The full Qwen, Radix, and Inferact native checkpoints were neither downloaded
nor attempted because their published forms do not satisfy the one-Spark fit
gate.

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
contains safetensors headers and qualification files. These figures are for
fit planning; the eventual derived artifact must publish exact file sizes and
hashes.

### Candidate PLE representations

| PLE strategy | Estimated PLE storage/residency | Estimated text-only weight residency | Gross Spark headroom | Disposition |
| --- | ---: | ---: | ---: | --- |
| Current FP8 table | 47.684 GiB | about 125.1 GiB | negative | Reject before download as a serving target |
| NVFP4, group 16 | about 26.82 GiB | about 104.2 GiB | about 15.5 GiB | First native build; existing reference gather to fuse and wire |
| Row-scaled I4 | about 24.44 GiB | about 101.8 GiB | about 17.9 GiB | Secondary experiment; needs a new loader/kernel |
| Exact file-backed FP8 | 47.684 GiB on NVMe, bounded rows resident | about 77.4 GiB plus staging/cache | about 42.3 GiB before staging | Best exact-memory design, more runtime engineering |
| PLE omitted | none | about 77.4 GiB | about 42.3 GiB | Runtime/MTP smoke only; changes model semantics |

The NVFP4 estimate includes packed 4-bit values and one FP8 block scale per 16
values. It is approximately 2.4 GiB larger than a one-scale-per-row I4 design.
SGLang already has a correctness-oriented ModelOpt unpack/dequant method for
this representation, although Qwen4 does not wire it and the method is not yet
a fused production PLE kernel. Extending that known layout is still a better
first engineering trade than defining a second I4 format for a small additional
saving.

The 15.5 GiB gross headroom in the PLE-NVFP4/BF16-MTP form is still marginal.
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
| Radix NVFP4 | [`7b71922`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594) | 125.96 GiB | Main routed experts NVFP4; BF16 MTP; FP8 PLE | Best derivation source, not directly admissible |
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
classes all imported. This proves the aarch64 package surface is present; it
does not prove that the full checkpoint fits or that every native kernel works
on SM121.

| Diagnostic | Measured result | Boundary |
| --- | --- | --- |
| ModelOpt NVFP4 embedding primitive | Pass; CUDA output matched an independent E2M1/block-scale reference with maximum absolute error `0` | Isolated embedding method, not the Qwen4 PLE loader or a fused production kernel |
| FlashInfer GDN | BF16 failed on SM121; retaining FP32 for the GDN path was required | Conditional runtime pass, not a BF16 GDN support claim |
| Native QSA | Fail during CuTe MLIR compilation on SM121 | QSA must be disabled or fixed before a representative native run |
| Real tiny checkpoint, QSA-disabled semantic control | HTTP `200`; 4 prompt tokens to 8 completion tokens in `1.759583 s`; model load `1.71 s` | Bounded request/response and real-weight execution control only |
| Dummy target plus `NEXTN` MTP | Both target and MTP loaded; HTTP `200` with 16 completion tokens in `0.338387 s`; 15 proposed, 15 verified, 0 accepted | Synthetic weights; proves draft execution and counters, not useful speculation |
| Full Radix NVFP4 checkpoint | Not downloaded or attempted | Published payload already exceeds total physical unified memory |

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

#### Smallest credible QSA fix

Read-only inspection of the exact image isolates the SM121 fallback. QSA's
`_resolve_trtllm_sparse_decode()` admits only `is_sm100_supported()`. GB10 fails
that guard; classic FlashAttention 2 is absent, so the resolver selects the
installed FlashAttention 4 CuTe path that failed compilation. The generic
`--attention-backend` option cannot override this because SGLang replaces it
with `QwenSparseAttnBackend` for the hybrid GDN/QSA architecture.

The installed FlashInfer `0.6.17` XQA implementation already admits compute
capability major 12, page size 64, and QSA's supported head geometry. The
smallest credible upstream experiment is therefore to admit
`is_sm120_supported()` alongside the SM100 check and let FlashInfer's existing
auto dispatch select XQA on SM120/SM121. This is a source-supported hypothesis,
not a measured fix. It needs a resolver unit test, an SM12x packed-XQA parity
test, and the native tiny QSA server smoke before the blocker can be closed.

This path is promising because the measured image imported the relevant class,
and the support-lineage source at PR commit `73a2552` contains a
[`ModelOptNvFp4EmbeddingMethod`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/quantization/modelopt_quant.py#L633-L728)
that gathers packed NVFP4 embedding rows, expands their E2M1 values and group
scales, and emits BF16. It currently uses PyTorch indexing and unpacking rather
than a fused PLE kernel. The Qwen4 model separately overlaps its existing
BF16/FP8 pinned-host PLE gather with the preceding layer; the packed NVFP4 path
must be integrated with that scheduling rather than assumed to work already.

Four integration gaps must be fixed and tested before building a 100+ GiB
artifact:

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

Do not combine `--ple-offload-embedding` with the NVFP4 PLE build. The current
pinned-host wrapper accepts only unquantized BF16 or FP8 tables, and on Spark a
full pinned copy would consume the unified capacity the quantization is meant
to save.

Native MTP uses the bundled head; no sidecar draft model is needed. The initial
SGLang settings are:

```text
--speculative-algorithm NEXTN
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

The checkpoint contains one physical MTP layer. Steps 1/2/3 repeatedly apply
that layer during speculation; they are not three separately trained heads.
The measured sweep must also test steps 1 and 2 with the corresponding bounded
draft tree, plus MTP off. Depth 3 is a recipe default, not an assumed winner.

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

The current [TokenSpeed recipe](https://lightseek.org/tokenspeed/recipes/models#qwen38-flash-next)
publishes FP8 TP4 plus MTP3 and has no native single-Spark NVFP4/I4 route.

## Build sequence

1. **Complete the tiny format fixture.** The isolated group-16 NVFP4 embedding
   primitive already matched the independent E2M1/block-scale reference with
   zero maximum absolute error on SM121. Extend that control to dimension 160
   and multiple PLE-style shards, then test duplicate IDs, first/last rows,
   shard boundaries, and malformed/missing scales through the Qwen4 loader.
2. **Qwen4 loader and kernel patch.** Pass the quantization configuration and
   prefix into the PLE embedding, add sharded packed-weight and scale loading,
   and fuse packed-U8 gather/dequant to BF16. Keep the existing BF16/FP8 path
   unchanged. Add a full-shape metadata-only allocation check so a wrong dtype
   cannot allocate a 95 GiB BF16 table.
3. **Harden the runtime-only smoke.** The pinned tiny fixture proved text-only
   real-weight execution with QSA disabled, and dummy weights proved Qwen4
   `NEXTN` draft activity on SM121. Fix or explicitly route around the native
   QSA CuTe MLIR failure, retain FP32 GDN, and automate both controls. These are
   still architecture diagnostics, not model-quality or throughput results.
4. **Stream the derivative.** Start from the pinned Radix checkpoint and
   transform PLE shards one at a time; never load or dequantize the complete
   table. Preserve all unmodified tensor bytes and record the source revision,
   converter revision, per-file sizes, and SHA-256 digests. First retain BF16
   MTP.
5. **Bounded admission.** Load text-only, eager, cache-off, one request, and a
   4K served context. Refuse swap and retain at least a measured runtime reserve
   before increasing K/V allocation. The planning gate is at least 12 GiB of
   projected reserve; after readiness require at least 12 GiB `MemAvailable`
   and stop below 8 GiB. Then test 32K with BF16 K/V.
6. **MTP proof.** Run MTP off and steps 1/2/3 in separate server lifetimes.
   Require nonzero proposed tokens, internally consistent accepted-token
   counters, and no content corruption before comparing throughput.
7. **Only if needed, quantize MTP experts.** Add explicit serialized metadata
   and W4A16/NVFP4 routed-expert loading. Preserve the remainder of MTP in BF16.
   Reject the form if its accepted length or bounded quality loses more than
   its wall-time gain.
8. **Escalate context and concurrency.** Move through 16K, 32K, 64K, and 128K
   in fresh lifetimes; attempt 262K only after the memory equation closes.

The full 125.96 GiB artifact acquisition starts only after steps 1–3 pass. The
exact image is already local, but the failed native QSA probe and incomplete
Qwen4 PLE-loader fixture still justify deferring the oversized weight source.

## Optimization matrix after admission

Change one axis at a time and restart the server between context tiers.

| Axis | Screening values | Rule |
| --- | --- | --- |
| MTP | off; steps 1, 2, 3 | Compare target-verified effective tok/s and acceptance by position |
| FP4 backend | `auto`, then available cuDNN and CUTLASS candidates | Prove SM121 execution in eager mode first; pin the winner |
| CUDA graph | eager, batch 1, then batch 2/4 | Do not diagnose a graph failure and a model failure together |
| Chunked prefill | 1,024; 2,048; 4,096 | Official recipe's 4,096 is a candidate, not a GB10 optimum |
| Context | 4K, 16K, 32K, 64K, 128K, then 262K | Fresh lifetime per tier; cross 32K explicitly |
| Concurrency | 1, 2, 4, 8 | Stop when memory, queueing, or per-request latency defeats aggregate gain |
| K/V | BF16, then calibrated/validated FP8 | No NVFP4 K/V in the first matrix |
| Prefix cache | off, then on for the winning MTP depth | Measure cold establishment and exact warm reuse separately |
| Reasoning | fixed no-thinking speed screen; one fixed reasoning-effort quality arm | Never mix reasoning effort inside a TPS comparison |

Primary performance is completed, validated output tokens divided by full case
wall time. Also retain TTFT, TPOT, aggregate output throughput, startup time,
minimum available memory, swap delta, power, proposed/accepted MTP tokens,
acceptance by position, and cache-hit counters. Engine-emitted tokens that fail
the validator are not usable throughput.

The first depth screen uses D256/C1 with two warmups and five repetitions. The
winner gets a 20-repetition confirmation with reversed/interleaved order before
the C1/C2/C4/C8 and long-context ladders. Streaming and non-streaming parity
must be checked below and above 32K because speculative output corruption can
otherwise look like a speedup.

## Acceptance gates

A native result is publishable only when all applicable gates pass:

- the exact artifact and aarch64 image revisions are immutable and hashed;
- the served model loads without implicit download, swap growth, or unrelated
  GPU/container work;
- PLE is present for quality runs and its packed gather matches the reference;
- requested MTP produces nonzero draft activity and counter topology matches
  the configured depth;
- `0 <= accepted_tokens <= proposed_tokens`, with finite per-position rates;
- deterministic chat, exact-answer, JSON, tool-call, retrieval, and streaming
  parity checks pass at the tested boundary;
- the same reasoning, sampling, cache, context, and concurrency settings are
  used inside each causal comparison; and
- the native form beats the GGUF target on effective TPS or offers a clearly
  measured quality/context benefit that justifies lower speed.

The GGUF comparison remains descriptive: quantization, runtime, PLE precision,
and MTP all differ. It is a product target, not a backend-only causal control.

## Repository work after the runtime fixture passes

SparkBench already supports vLLM concurrency and parses vLLM lifetime MTP
counters, but the SGLang OpenAI path currently exports no corresponding
acceptance aggregate and neither backend fails closed per case. Before a long
campaign:

- add a separate authenticated, non-streaming SGLang acceptance audit using
  `return_meta_info=true`; validate and retain only proposed/accepted draft
  counts, verify count, acceptance scalars, and the bounded histogram;
- require `accepted <= proposed`, histogram steps equal verify count, histogram
  accepted-token sum equal the reported accepted count, and proposed count
  equal `verify_count * (configured_draft_tokens - 1)`;
- optionally enable `/metrics` and retain `sglang:spec_verify_calls_total` only
  as an activity cross-check. Its acceptance fields are interval gauges and it
  has no cumulative proposed/accepted-token counters, so it cannot replace the
  per-request audit;
- keep streaming TPS measurements explicitly counter-unverified until exact
  same-request streaming counters are available; SGLang rejects
  `return_meta_info=true` for OpenAI streaming requests;

- snapshot speculative counters after warmup and after each measured case;
- derive bounded per-case deltas and reject missing/reset/inconsistent MTP
  evidence;
- add backend-neutral MTP depth screening and confirmation suites;
- add isolated Flash-Next 16K/32K/64K/128K suites rather than reusing one
  ascending lifetime;
- add a separate vLLM/SGLang prefix-cache cold/warm protocol; and
- persist only sanitized scalar evidence—never prompts, completions, reasoning,
  tool payloads, request identifiers, local paths, or raw logs.

The manifest may retain the explicitly labeled tiny/dummy diagnostic controls,
but no production or representative native Flash-Next profile should be added
until a pinned full-quality artifact passes the packed loader, native QSA,
SM121 runtime, and memory-admission gates above.
