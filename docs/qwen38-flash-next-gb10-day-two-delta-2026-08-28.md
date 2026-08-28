# Qwen3.8-Flash-Next on GB10: day-two literature delta — 2026-08-28

## Conclusion

Public implementation work accelerated, but official one-Spark GB10 support
did not arrive. The most important day-two change is negative: SGLang briefly
enabled its TRT-LLM sparse-decode route on SM121, then restricted that route to
exact SM120 after multi-trial long-context corruption was reported on GB10.
That makes the repository's digest-pinned local SGLang overlay useful measured
evidence, but not a generally safe upstream recipe. Its two repeated-word 131K
passes do not establish varied-token correctness at long context.

The highest-value new reproduction target is an open vLLM pull request that
maps the PLE safetensors directly instead of materializing the full table. Its
author reports about 25.1 tok/s warm on one Spark with MTP2, exact random-row
checks, and no GPU-resident PLE allocation. It is still an unmerged community
branch stacked on another open model-support pull request, so it is an
experiment candidate rather than supported deployment guidance.

One external vLLM profile also sharpens the performance hypothesis. It
attributes most per-token wall time to BF16 dense GEMV and only a small share
to the NVFP4 MoE experts. That is consistent with this repository's inference
that the target pass, dense weight traffic, or low-batch kernel path limits
single-user decode more than the page-backed PLE lookup does. It is not local
profiler evidence and does not distinguish memory bandwidth, occupancy,
launch overhead, or scheduler cost on this exact stack.

The operational decision is:

- do not change the frozen SGLang autoresearch campaign or its 84-file harness
  before its cutoff;
- keep the local clean MTP3/off and replicated mapped-PLE MTP2 results as the
  measured anchors;
- treat SGLang's general SM121 TRT-LLM route as unsafe until upstream has a
  validated replacement with natural long-context coverage; and
- after the frozen campaign closes, reproduce the vLLM mmap branch under the
  same immutable, offline, loopback-authenticated, scalar-evidence protocol.

## Review window and evidence classes

This is a strict delta from the day-one review's repository cutoff. The prior
report was committed at `2026-08-27T21:58:59Z`; this review covers public
changes visible through `2026-08-28T09:30:00Z`. Materially relevant changes
that began before the cutoff but were absent from the prior report are labeled
**supplemental** or explicitly described as a sequence that straddles the
cutoff; they are not silently counted as day-two publications.

The source classes remain deliberately separate:

| Class | Interpretation in this report |
| --- | --- |
| Merged upstream commit | Authoritative for what code was merged, not for GB10 performance or broad correctness |
| Open upstream pull request | Inspectable experiment target; not release support |
| Pinned community repository or checkpoint | Reproducible starting point if all dependencies and model revisions are also pinned |
| Forum or repository benchmark claim | Discovery evidence until independently reproduced with a persisted protocol and artifacts |
| Tracked SparkBench evidence | The only class treated as a measurement from this machine |

No external number below is added to the repository's measured result table.

## SGLang: SM121 enablement was narrowed after corruption

This sequence straddles the cutoff: the initial SM121 enablement preceded it;
the safety restriction that defines the present upstream boundary followed it.

[SGLang PR #36649](https://github.com/sgl-project/sglang/pull/36649)
merged as
[`7c66045d71f067c1c5da2b85baad3c47d9a19cb7`](https://github.com/sgl-project/sglang/commit/7c66045d71f067c1c5da2b85baad3c47d9a19cb7).
It was merged to `qwen4-main-squashed` at `2026-08-27T09:44:04Z`, before this
review's delta cutoff; it was not a release or a merge to SGLang `main`. It
admitted SM121 to the FlashInfer TRT-LLM sparse-decode route. Subsequent testing
on open
[PR #36556](https://github.com/sgl-project/sglang/pull/36556), at test head
[`dac5523d1e5d2f4297fec40ef02fc76fb0f662d1`](https://github.com/poorpaper/sglang/commit/dac5523d1e5d2f4297fec40ef02fc76fb0f662d1),
reported stochastic corruption that increased with context length:

| Input tier | TRT-LLM corrupt trials | Community repaired-fallback result |
| ---: | ---: | --- |
| 120K | 1/4 | 6/6 clean |
| 160K | 1/4 | not tested |
| 190K | 2/4 | 6/6 clean |
| 210K | 4/4 | 6/6 clean |
| 240K | not reported | 6/6 clean only at chunked prefill 1,024 and memory fraction 0.82 |

At 240K, the same repaired fallback at chunked prefill 4,096 and memory
fraction 0.90 produced two corrupt trials out of six under workspace pressure.
The test used two Sparks, TP2, real weights, and MiaAI-Lab's repaired Triton
varlen fallback—not PR #36556's exact SGLang FlashAttention-4 implementation.
It supports the fallback direction but does not validate that PR's fallback.
The source is the
[multi-trial PR report](https://github.com/sgl-project/sglang/pull/36556#issuecomment-5448079828).
Non-corrupt TRT-LLM runs were also reported at 43.5 tok/s versus 56.6 tok/s for
the repaired fallback, attributed by the tester to lower MTP acceptance. Those
rates are external branch-specific measurements, not a one-Spark result from
this repository.

[PR #36806](https://github.com/sgl-project/sglang/pull/36806) then merged as
[`99c9362e6685db579c469f6e0e566b08827b3477`](https://github.com/sgl-project/sglang/commit/99c9362e6685db579c469f6e0e566b08827b3477),
at `2026-08-28T09:08:22Z`, also to `qwen4-main-squashed`. It restricts the path
to exact SM120 and excludes SM121/GB10 on that branch; it is not release
support. The original
[GB10 issue #36558](https://github.com/sgl-project/sglang/issues/36558) remains
open. A later
[dummy-weight A/B report](https://github.com/sgl-project/sglang/issues/36558#issuecomment-5446719556)
showed that PR #36556 fixes the initialization crash, but explicitly did not
validate the real checkpoint or long-KV output. An earlier corrected
[real-output report](https://github.com/sgl-project/sglang/issues/36558#issuecomment-5432980916)
described token-zero `!` streams beyond a few thousand tokens and an engine
that remained poisoned until restart. That corrected report predates the
delta cutoff and is included as supplemental context.

This changes the confidence boundary, not the historical local measurements:

- the local overlay is image-digest- and payload-digest-pinned and has passed
  the repository's repeated-word 131K exact-key case twice;
- that local path is not the same claim as "merged SGLang supports GB10"; and
- repeated-word exact-key retention is too narrow to rule out stochastic or
  varied-token corruption at 120K and above.

The frozen campaign's 60K repeated-word exact-key case remains a mandatory
synthetic capacity gate. It must not be weakened or replaced to accommodate
the new literature, but it cannot address varied-token corruption. Natural
varied-token validation belongs in a separate post-campaign protocol.

## vLLM: direct PLE mmap is the leading reproduction target

[vLLM PR #54129](https://github.com/vllm-project/vllm/pull/54129) opened on
2026-08-28 and remains open. Its functional commit is
[`eae5aa8fb15c3af1a8ebc23b0d027f465c6c57f3`](https://github.com/Trosfy/vllm/commit/eae5aa8fb15c3af1a8ebc23b0d027f465c6c57f3);
the reviewed rebased head was
[`8e4e036a311604800334989485b4ee23925956da`](https://github.com/Trosfy/vllm/commit/8e4e036a311604800334989485b4ee23925956da).
It is stacked on model-support PR #53896 rather than the CPU-offload PR
#53899.

With `VLLM_PLE_MMAP=1`, the branch maps the PLE tensors directly from their
safetensors storage and avoids allocating the complete table on the GPU. The
author reports the following on one Spark at 32K with MTP2:

| Author-reported check | Result |
| --- | ---: |
| Warm output rate | 25.1 tok/s |
| Cold output rate | about 18 tok/s |
| MTP acceptance | 76% |
| GPU PLE bytes | 0 |
| Random-row oracle matches | 1,000/1,000 |
| GSM8K spot check | 19/20 |
| Targeted model tests | 205 passing; one unrelated AMD suite ignored |

This is stronger implementation evidence than a throughput-only forum post,
but it still lacks a merged release, a locally frozen container lineage, a
matched SGLang protocol, and tracked request-level SparkBench evidence. The PR
calls the input an "FP8 checkpoint" without publishing its model ID or
revision. Its roughly 78 GiB backbone plus 47.68 GiB FP8 PLE footprint strongly
suggests the Radix NVFP4-backbone/FP8-PLE artifact, but that is an inference,
not author-confirmed provenance. The reviewed head resolved the earlier merge
conflict; the PR remains open without upstream code-test CI.

Model-support [PR #53896](https://github.com/vllm-project/vllm/pull/53896)
also remains open. Its reviewed head was
[`89d0bb71aeb2f3e15c16efc69d33c3fbe223a765`](https://github.com/peakcrosser7/vllm/commit/89d0bb71aeb2f3e15c16efc69d33c3fbe223a765).
A new correctness commit,
[`0e0802f4637c73589c9943d420758177df454d9a`](https://github.com/peakcrosser7/vllm/commit/0e0802f4637c73589c9943d420758177df454d9a),
moves request-layout-dependent PLE n-gram ID computation into the splitting
custom operation used with piecewise CUDA graphs. That commit belongs to the
newer #53896 head and is not contained in reviewed PR #54129 head `8e4e036`;
combining them would be a new integration arm rather than exact reproduction.

**Supplemental:** the pull request also contains useful one-GB10 community
data that was not included in the day-one report. One
[MTP/concurrency table](https://github.com/vllm-project/vllm/pull/53896#issuecomment-5444210155)
reports:

| Offered concurrency | No speculation | MTP2 | MTP3 |
| ---: | ---: | ---: | ---: |
| 1 | 17.1 | 28.5 | 27.4 |
| 4 | 44.1 | 50.7 | 60.6 |
| 8 | 87.5 | 89.0 | 93.4 |

The table used one GB10/TP1, contributor-local changes to admit NVFP4 with PLE
CPU offload, and a 64 GiB swapfile carrying cold PLE pages. It does not prove
that the reviewed heads of #53896 or #53899 support NVFP4 PLE offload; their
stated validation boundary remains BF16/FP8. It is also not matched to this
repository's prompts, server lifetime, PLE mechanism, or validation gates.
The author reported about a fourfold C8 throughput difference between
`max-num-seqs` caps of 2 and 16, so that cap, queue time, and running/queued
counters must be frozen before interpreting concurrency scaling. The table
does independently reinforce two local observations: MTP is a large C1 lever,
while batching raises aggregate throughput much more than it improves one
user's serial decode.

A separate, supplemental
[profile in the same PR](https://github.com/vllm-project/vllm/pull/53896#issuecomment-5445670384)
attributes 69.4% of token wall time to BF16 cuBLAS GEMV. It estimates 4.84B
dense BF16 parameters moving 9.68 GB of 10.98 GB per token, while the NVFP4 MoE
experts consumed 0.47 ms per token. Treat that as external diagnostic evidence,
not as a local hardware counter or proof of a single root cause.

## CPU offload has a security and operability cost

[vLLM PR #53899](https://github.com/vllm-project/vllm/pull/53899) remains open.
There was no post-cutoff code change; the reviewed head remained
[`95dc96d1d012a25ff5c3823a1e77197c8dae4654`](https://github.com/peakcrosser7/vllm/commit/95dc96d1d012a25ff5c3823a1e77197c8dae4654),
dated `2026-08-27T06:00:29Z`. Its CPU-offload route uses `pidfd_getfd`.
Contributor testing first identified the default Docker seccomp profile; a
[later correction](https://github.com/vllm-project/vllm/issues/53960#issuecomment-5443452996)
identified Linux Yama `ptrace_scope=1` as another sibling-process block,
including on bare metal. `CAP_SYS_PTRACE` bypasses that Yama gate. The branch
does not yet provide a clear preflight or actionable error for these
conditions.

That capability expansion conflicts with this repository's preference for a
strictly scoped inference container. It makes the direct mmap branch the
preferable local target if its pins and correctness gates hold. A supplemental,
corrected
[real-weight report](https://github.com/vllm-project/vllm/pull/53899#issuecomment-5439230565)
gave 26.0–26.1 tok/s cold and 27.3–27.7 tok/s warm; the earlier higher figure
should not be used.

## New one-Spark community artifacts

### NVFP4-compressed PLE checkpoint

A new
[NVIDIA forum post](https://forums.developer.nvidia.com/t/qwen-3-8-flash-next-single-node-vllm-24-tok-s/381551)
links a repository pinned at
[`d46cd51`](https://github.com/provsalt/qwen3.8-flash-ple-nvfp4/tree/d46cd51eb3503b4c0019a74e0fb9262a912d5951)
and weights pinned at
[`48d9819`](https://huggingface.co/provsalt/Qwen3.8-Flash-Next-NVFP4-PLE-NVFP4/tree/48d98195ac8da8ff10d9ee497b5d52e7817f058d).
The author reports one GB10, TP1, vLLM development code, MTP3, BF16 KV cache,
262K context, and PLE CPU offload:

| Author-reported property | Value |
| --- | ---: |
| C1 output rate | 24.1 tok/s |
| Prompt rate | about 1,410 tok/s |
| C10 aggregate output rate | 45.2 tok/s |
| BF16 PLE size | 95.368 GiB |
| NVFP4 PLE size | 26.822 GiB |
| Complete converted checkpoint | 101.643 GiB |

The capacity reduction is interesting, but quality is unestablished. The
repository has only two commits, no persisted benchmark bundle, and no
completed findings document for the claimed multimodal checks. Quantizing the
PLE changes model semantics, so this artifact needs row-level, language,
tool-use, long-context, and multimodal validation before performance matters.

### Task-shaped vLLM and llama.cpp comparison

A one-GX10 report became visible in an
[NVIDIA forum reply](https://forums.developer.nvidia.com/t/qwen3-8-flash-next/381228/166).
Its pinned
[measurement document](https://github.com/0xBakeer/qwen38-flash-next-spark/blob/1611340c1a69b2d3d6c144dd5155716027ef85d1/docs/measurements.md)
uses temperature zero, medians of three, and unique prompt tokens per repeat.
It reports:

| Task shape | vLLM NVFP4 + MTP3 | llama.cpp Q4 + prompt lookup |
| --- | ---: | ---: |
| File reproduction | 39.1 tok/s | 88.5 tok/s |
| Bug fix | 35.0 tok/s | 46.1 tok/s |
| Function addition | 33.6 tok/s | 32.2 tok/s |
| Prose | 32.2 tok/s | 27.8 tok/s |

It additionally reports vLLM prefill at 2,231 tok/s for 2,542 input tokens and
2,183 tok/s for 195,458 tokens, with decode at 31.7/33.5/31.7 tok/s around
1K/32K/128K. The document is pinned, but its setup follows mutable upstream
`main`, leaves the model revision unpinned, and selects the first cached
snapshot. It motivates a task-shaped local battery; it is not a controlled
cross-runtime result for this repository.

### Supplemental: updated community repository

The previously reviewed `blazux` repository advanced to
[`d2854bf`](https://github.com/blazux/qwen3.8-Flash-DGX/commit/d2854bfff0a0b6f46984b0941ed1db6010031295).
It now defaults to 262K context, eight sequences, memory utilization 0.85,
MTP2, and YaRN. It self-reports 25–28 tok/s for typical MTP2 single-stream
decode, about 36 tok/s on predictable text, about 17 tok/s without MTP, and
successful 500K needle checks. Its no-spec native-offload concurrency report
rises from 17.1 tok/s at C1 to 266.8 tok/s at C48. That concurrency sweep used
a separate native PLE-offload path, no MTP, and 8K context; it is not a result
from PR #54129's direct-safetensors mmap implementation. These are useful
hypothesis inputs, but neither the model revision nor a full repeated evidence
bundle is frozen.

### HashK PLE: reject as a reproduction target

The
[HashK forum topic](https://forums.developer.nvidia.com/t/qwen3-8-flash-next-180b-single-solo-dgx-spark-with-hashk-ple-nvfp4/381519)
contains high throughput and tool-evaluation claims, including a later
[87/100 hard-mode report](https://forums.developer.nvidia.com/t/381519/16).
However, the linked
[repository](https://github.com/Death-By-Tokens/Qwen3.8-Flash-Next-180B-on-ONE-DGX-Spark)
is unavailable, there is no source revision to pin, the claimed PLE
reconstruction cosine was only about 0.50, and users reported persistent `!`
failure modes. It is not an admissible reproduction target.

## Official support still has not moved

The [official vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)
remains at recipe commit
[`7997f1d`](https://github.com/vllm-project/recipes/commit/7997f1d1bf1b7785a0367f19d2614cc3043c5948)
for this model and still has no validated one-Spark GB10 NVFP4 lane. Open
[recipe PR #870](https://github.com/vllm-project/recipes/pull/870) targets four
RTX 5090 cards, not GB10. The official model-card and Qwen repository updates
visible in this window—the Hugging Face head
[`de4b8e4`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/commit/de4b8e4d43b917e7706784d8bb445c9af86a3540)
and Qwen repository head
[`6988587`](https://github.com/QwenLM/Qwen3.8-Flash-Next/commit/69885871a64393807d988b27b1b5e380e8f28526)—were
documentation changes, not a new Spark runtime path.

The support matrix at this review is therefore:

| Route | One-Spark GB10 status | Local action |
| --- | --- | --- |
| SGLang TRT-LLM sparse decode | SM121 excluded by the reviewed `qwen4-main-squashed` gate after corruption reports; no release support | Keep the existing pinned overlay experimental; strengthen varied-token validation |
| vLLM direct PLE mmap | Open PR stacked on open model support; promising row and spot checks | Highest-priority post-cutoff reproduction target |
| vLLM CPU PLE offload | Open PR; default container isolation blocks `pidfd_getfd` | Do not add `SYS_PTRACE` merely to benchmark it; prefer mmap |
| Compressed NVFP4 PLE | Pinned community checkpoint, minimal quality evidence | Quality-first research arm, not champion candidate |
| HashK PLE | Source unavailable and reconstruction/output warnings | Reject |
| Official recipe | No one-Spark GB10 NVFP4 lane | Continue to label local paths experimental |

## What this changes in the local optimization plan

The day-two evidence narrows the next work rather than changing the frozen
campaign:

1. Resume the existing SGLang campaign only if its original preflight passes
   before cutoff. Otherwise preserve its typed blocked/cutoff state; do not
   mutate or re-summarize it into a synthetic completion.
2. Freeze the exact stacked PR #54129 tree rather than combining it with the
   newer #53896 head, then build an immutable Linux/aarch64 image. Use the
   pinned local Radix NVFP4-backbone/FP8-PLE checkpoint as the best-supported
   reconstruction target while clearly labeling the checkpoint identity as
   inferred until the author confirms it.
3. Use direct read-only mmap without `SYS_PTRACE`, implicit downloads, mutable
   branches, wildcard cache selection, or public API exposure.
4. Reuse the clean local D256 prompts and client geometry while freezing a new
   matched vLLM off/MTP2 pair; retain the local MTP3/off pair as the separate
   SGLang anchor. Then run C1/C2/C4/C8 and the deterministic coding/cowork
   battery under the same scalar telemetry, fixed `max-num-seqs`, and explicit
   queue counters.
5. Add natural varied-token long-context trials around 60K, 120K, 160K, 190K,
   and 210K only as fresh-lifetime ascending admissions. Precompute token, KV,
   and workspace budgets; retain the existing MemAvailable, swap, and PSI
   gates; and admit the next tier only after the prior tier is correct and
   pressure-clean. On corruption, preserve a typed failure and restart rather
   than retrying inside a poisoned process.
6. Profile a fixed C1 decode span with kernel timing and hardware counters to
   distinguish BF16 weight traffic from occupancy, launch, scheduler, and PLE
   page-fault costs.
7. Publish external claims and local measurements in separate tables; promote
   no configuration without semantic, lifecycle, memory, swap, and cleanup
   gates.

The likely single-user win remains reducing target verification passes per
useful output token through a well-accepted speculative head. The best newly
actionable systems hypothesis is that direct PLE mmap can preserve fit without
broad container capability, while the strongest new caution is that short or
repeated-word success cannot certify the GB10 long-context kernel path.

The exact source/checkpoint inference, local readiness audit, build boundary
and staged comparison are frozen in the
[direct-mmap reproduction plan](qwen38-flash-next-vllm-mmap-reproduction-2026-08-28.md).
