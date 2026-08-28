# Qwen3.8-Flash-Next on GB10: day-two literature delta — 2026-08-28

## Conclusion

Public implementation work accelerated, but official one-Spark GB10 support
did not arrive. The most important day-two change is negative: SGLang briefly
enabled its TRT-LLM sparse-decode route on SM121, then restricted that route to
exact SM120 after multi-trial long-context corruption was reported on GB10.
That makes the repository's digest-pinned local SGLang overlay useful measured
evidence, but not a generally safe upstream recipe. Its two repeated-word 131K
passes do not establish varied-token correctness at long context.

An open corrective SGLang pull request now supplies the first explicit SM121
Triton packed-varlen fallback directly atop that safety restriction. Its author
reports clean one-Spark short, concurrent, tool and varied-token 120K--210K
checks, while forced TRT-LLM produced token-zero output. This is meaningful
corrective evidence, but the pull request is minutes old, unmerged, publicly
red/gated, and contains no NVMe PLE reader or comparable end-to-end TPS result.

The highest-value new systems reproduction target is an open vLLM pull request
that maps the PLE safetensors directly instead of materializing the full table.
Its author reports about 25.1 tok/s warm on one Spark with MTP2, exact random-row
checks, and no GPU-resident PLE allocation. It is still an unmerged community
branch stacked on another open model-support pull request, so it is an
experiment candidate rather than supported deployment guidance.

A new GB10 vLLM issue adds a workload-specific safety gate: a growing
multi-turn prefix can crash the GDN path when prefix caching is enabled even
though repeated identical prompts appear healthy and fast. That report used a
different third-party mmap patch, so it does not establish a defect in the
reviewed direct-mmap branch. It does establish that identical-prompt cache
replay is an inadequate admission test for this model family.

A second open SGLang branch now supplies a concrete `io_uring` PLE reader and
an exact Radix checkpoint pin. It reports 24.23 output tok/s across forty
512-token requests on one Spark. The storage design is a serious reproduction
candidate, but its current stack still contains the SM121 TRT-LLM enablement
that upstream subsequently restricted after corruption reports. Its new
correctness-first Triton fallback runs only when the preferred TRT-LLM kernel
is unavailable. The reader can be evaluated only after separating that storage
change from the stale QSA dispatch boundary.

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
- the former bounded exception for the frozen fourteen-cell campaign exhausted
  its admission window without a measurement, so keep TRT-LLM excluded on SM121
  and treat the new explicit Triton fallback as an unmerged corrective
  candidate, not release support;
- treat SGLang's new `io_uring` reader as a component candidate, not permission
  to run its exact stale SM121 QSA stack; and
- after the frozen campaign closes, reproduce the vLLM mmap branch under the
  same immutable, offline, loopback-authenticated, scalar-evidence protocol,
  starting cache-off and requiring a growing-prefix canary before any cache-on
  result.

## Review window and evidence classes

This is a strict delta from the day-one review's repository cutoff. The prior
report was committed at `2026-08-27T21:58:59Z`; this review covers public
changes visible through `2026-08-28T10:43:36Z`. Materially relevant changes
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

The expired campaign planned a 60K repeated-word exact-key synthetic capacity
gate but never measured it. A future admitted runtime must freeze that gate
again without weakening or replacing it; the case still cannot address
varied-token corruption. Natural varied-token validation belongs in a separate
future protocol.

## SGLang: an explicit SM121 Triton fallback is now inspectable

[SGLang PR #36845](https://github.com/sgl-project/sglang/pull/36845) opened at
`2026-08-28T10:22:31Z`. Its single reviewed commit,
[`3681c4e03f6848dff82972b3f572602d3b8394cc`](https://github.com/sgl-project/sglang/commit/3681c4e03f6848dff82972b3f572602d3b8394cc),
sits directly atop restriction `99c9362`. It retains TRT-LLM sparse decode for
exact SM120 and routes SM121 to a narrow
[Triton packed-varlen QSA kernel](https://github.com/sgl-project/sglang/blob/3681c4e03f6848dff82972b3f572602d3b8394cc/python/sglang/srt/layers/attention/qsa/sm121_varlen.py).
This is the first source boundary in this review that both preserves the SM121
restriction and supplies an explicit GB10 fallback.

The author reports
[68 QSA tests](https://github.com/sgl-project/sglang/blob/3681c4e03f6848dff82972b3f572602d3b8394cc/test/registered/kernels/test_qsa.py),
including 30 BF16/FP16 differential cases
with worst absolute error `0.001953125` and relative RMS `6.994461e-05`.
Reported kernel times at batch one/four/seven are 0.112843/0.155732/0.210166
ms. A separate one-Spark, TP1, 262K, NVFP4-expert, FP8-PLE, NEXTN run reported:

| Author-reported gate | Result |
| --- | ---: |
| Cold startup | 554.32 s |
| Sequential / concurrent exact-output requests | 20/20 / 4/4 |
| Structured tool call | Passed |
| Exact varied-token NIAH at 120K / 190K / 210K | Passed in 50.46 / 58.13 / 53.30 s |
| Token-ID-zero outputs / final health | 0 / HTTP 200 |
| Forced SM121 TRT-LLM control | Token ID 0 for all 32 generated tokens at all three lengths |

These are author-reported validation results, not local measurements. The PR
is open and unmerged; at the audit its public checks were gated/red. It does
not pin the checkpoint revision, image digest, full launch or prompt bundle,
and it adds no comparable end-to-end TPS measurement. Its one commit contains
no NVMe reader, so the end-to-end run used a separate capacity mechanism not
defined by the PR. Varied-token NIAH is much stronger than repeated-word
retention but is still synthetic rather than broad natural-context coverage.

PR #36567 and PR #36845 diverge from the earlier SM121 enablement; neither
contains the other. The concrete storage-plus-restriction-preserving-kernel
candidate is a new integration: rebase only #36567's Rust reader
[`04648a7`](https://github.com/jzinno/sglang/commit/04648a701501e473081ebabd2c110474f915e924)
and PLE streaming
[`9f101e3`](https://github.com/jzinno/sglang/commit/9f101e39ff09b356355e6a11183eaa3f7bb15f8c)
onto `3681c4e`, excluding its competing QSA commit `8ef3b3`. That result needs
its own source identity, row oracle, dispatch attestation and long-context
validation; it is not a reproduction of either published head.

A local read-only static replay now establishes that the two storage commits
apply to `3681c4e` without content conflicts. The tree after the Rust reader is
`cb9b2dffb10ae70bc91915c3eade4957fa649eaa`; after PLE streaming it is
`ddda8dde3b6655c4e0c0ff094d87ef1f5cc71a92`. No added line references QSA,
TRT-LLM or SM121, and the resolver, fallback, architecture detector and QSA
test blobs remain byte-identical to `3681c4e`. This is static composition
evidence, not a build or runtime result.

The replay also found a packaging blocker. The new `_storage` Rust extension
is auto-discovered by the build, but prebuilt-module staging, required-module
checking and import smoke tests still enumerate only `server`, `grpc` and
`multimodal`. Before building, add `storage` to the explicit lists in
`stage_rust_ext_modules.sh` and `ci_install_dependency.sh`, produce a new tree
identity, and prove an ARM64 `_storage` import with runtime building disabled.
An `EPERM` or `ENOSYS` skip on the target is a failed `io_uring` admission, not
a passing test.

## SGLang: the new `io_uring` reader is useful, but its QSA stack is stale

Open [SGLang PR #36567](https://github.com/sgl-project/sglang/pull/36567), at
reviewed head
[`d866243006f5dcb073223cfa4fe90a7a3f740c45`](https://github.com/sgl-project/sglang/commit/d866243006f5dcb073223cfa4fe90a7a3f740c45),
adds a substantially different NVMe PLE route. It keeps the table in the
original indexed safetensors, parses exact row offsets, and uses a bundled Rust
reader with persistent `io_uring`, `O_DIRECT`, page-aligned bounded reads and
an optional application LRU. A background thread begins the read before the
preceding decoder layer, then a persistent pinned stage supports asynchronous
H2D and FP8-to-BF16 conversion. The implementation also retains an mmap reader
as a correctness fallback
([source](https://github.com/sgl-project/sglang/blob/d866243006f5dcb073223cfa4fe90a7a3f740c45/python/sglang/srt/models/qwen4_ple_nvme.py#L266-L418),
[staging path](https://github.com/sgl-project/sglang/blob/d866243006f5dcb073223cfa4fe90a7a3f740c45/python/sglang/srt/models/qwen4_ple_nvme.py#L452-L606)).

The PR publishes unusually specific one-Spark data for the exact local Radix
revision `7b71922`:

| Author-reported measure | Result |
| --- | ---: |
| Sixteen-row Rust-reader microbenchmark, 1,000 iterations | 0.208 ms p50 / 0.627 ms p95 / 0.944 ms p99 |
| Sparse GQA SM121 kernel, batch one, 100 iterations | 0.0978 ms mean |
| Forty requests: chat/STEM/math/code, 512 output tokens each | 20,480/20,480 output tokens; zero API errors, restarts or OOMs |
| Aggregate output | 24.23 tok/s over 845.30 s |
| Per-domain output | 23.52 / 23.78 / 26.97 / 23.01 tok/s |
| Mean acceptance length | 2.60 |
| Overall median TTFT / reported TPOT aggregate | 469.02 ms / 39.06 ms |
| Mixed gather log after 7,000 calls | 703,568 selected rows; 4.475 ms mean read time |
| Target / NEXTN loading | 466.10 s / 90.20 s |
| Post-pool available memory | 18.31 GiB; 447,040 KV-cache tokens allocated |

The run used TP1, concurrency one, BF16 KV, 32K context, eager execution and
MTP3 with top-k one/four draft tokens. Its prompt set, 512-token outputs,
reasoning policy, eager graphs and MTP depth do not match either the local
SGLang anchors or vLLM's MTP2 report. The rates are therefore not a backend
ranking. The row checks—first, boundary, final and sixteen random rows against
`safe_open`—are useful admission evidence but much smaller than this
repository's planned 1,000-row oracle.

The decisive caveat is branch ancestry. Head `d866243` contains the earlier
SM121 enablement `7c66045` and does not contain the later safety restriction
`99c9362`. Its resolver still selects TRT-LLM on SM121 whenever FlashInfer's
decode function imports
([resolver](https://github.com/sgl-project/sglang/blob/d866243006f5dcb073223cfa4fe90a7a3f740c45/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py#L52-L67)).
Only when that preferred path is unavailable does the new SM121 Triton sparse
GQA fallback run
([dispatch](https://github.com/sgl-project/sglang/blob/d866243006f5dcb073223cfa4fe90a7a3f740c45/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py#L1645-L1665)).
The fallback's model-shape unit comparison and microbenchmark do not establish
that the published end-to-end run selected it or that either route is correct
at varied 120K context. At this audit, the PR remained open and its public
base, extra and AMD check summaries were red.

Reproduce the storage component only as a clearly named integration arm:

- rebase storage commits `04648a7` and `9f101e3` onto `3681c4e`, excluding
  #36567's competing QSA commit; never silently carry the stale TRT-LLM
  resolver;
- pin the Rust/Cargo tree, wheel and image digest, then run the full row oracle,
  compile/dispatch attestation and ascending natural long-context gate;
- use a narrow seccomp profile that admits only `io_uring_setup`,
  `io_uring_enter` and `io_uring_register`; never use `seccomp=unconfined` or a
  privileged container merely to benchmark it; and
- keep `io_uring` and any application page-cache size as explicit axes. The
  local host reports `kernel.io_uring_disabled=0`, but container syscall
  admission and the internal NVMe's `O_DIRECT` behavior remain untested.

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
A separately accepted performance fix,
[`4df2ce22d086007a81930d93b3b657a1d197aecc`](https://github.com/peakcrosser7/vllm/commit/4df2ce22d086007a81930d93b3b657a1d197aecc),
limits the packed PLE hash view to `:num_tokens`. Without it, the implementation
fills and hashes the full configured batched-token width even for a one-token
decode call. That fix is present in the newer model-support lineage but absent
from PR #54129 head `8e4e036`.

Preserve `8e4e036` as the author-shape control, then test only the live-width
slice as a separately hashed integration. Do not use scheduler ceilings as the
first workaround for that wasted decode work. The live-width change preserves
the mmap branch's existing whole-operation graph boundary and should precede
mmap worker or chunk tuning.

A new correctness commit,
[`0e0802f4637c73589c9943d420758177df454d9a`](https://github.com/peakcrosser7/vllm/commit/0e0802f4637c73589c9943d420758177df454d9a),
moves request-layout-dependent PLE n-gram ID computation into the splitting
custom operation used with piecewise CUDA graphs. That commit belongs to the
newer #53896 head and is not contained in reviewed PR #54129 head `8e4e036`;
it is needed only if a later experiment narrows the mmap boundary to
gather-only. Combining it prematurely would be a new integration arm rather
than exact reproduction.

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

### Growing-prefix cache crash is a new admission gate

[vLLM issue #54173](https://github.com/vllm-project/vllm/issues/54173), opened
on 2026-08-28, reports an illegal memory access on one GB10 when a cached long
prefix is resumed at successively greater lengths. The report pins vLLM
`8e685d198`, image digest
`sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`,
and the same Radix checkpoint revision used in this repository's reconstruction.
The reporter labeled failed cache-on attempts `mamba-cache-mode=align` and
`all`, and disabling async scheduling did not prevent the crash. Those labels
must not be interpreted as two resolved cache modes on the reviewed Qwen4Exp
heads: the
[model class](https://github.com/Trosfy/vllm/blob/8e4e036a311604800334989485b4ee23925956da/vllm/models/qwen4_exp/nvidia/model.py#L587-L603)
does not declare `SupportsMambaPrefixCaching`, the interface needed for all-mode
prefix caching, so shared
[configuration](https://github.com/Trosfy/vllm/blob/8e4e036a311604800334989485b4ee23925956da/vllm/model_executor/models/config.py#L622-L657)
normalizes `all` to `align`, while model initialization
[rejects](https://github.com/Trosfy/vllm/blob/8e4e036a311604800334989485b4ee23925956da/vllm/models/qwen4_exp/nvidia/model.py#L612-L624)
a surviving `all` value. Disabling prefix caching was the only reported stable
configuration. A future comparison must persist the resolved cache
configuration rather than infer it from the launch argument.

The trigger is easy for a benchmark to miss. Nine byte-identical roughly 50K
prompts completed while cached TTFT fell from 24.38 to 1.40 seconds, but a
shared prefix extended across turns failed within roughly five to ten
requests. The reporter's cache-on and cache-off deep-context checks remained
correct before the crash, so a short semantic smoke also would not screen it.

This is discovery evidence, not attribution to vLLM, GDN, or PLE alone. The
image carries the separate `blazux/qwen3.8-Flash-DGX` mmap patch, and the
unpatched checkpoint cannot fit on that one-Spark host. The first PR #54129
admission must therefore remain explicitly cache-off: use
`--no-enable-prefix-caching`, require resolved `enable_prefix_caching=false`
and `mamba_cache_mode=none`, attest the
[exported cache-configuration labels](https://github.com/Trosfy/vllm/blob/8e4e036a311604800334989485b4ee23925956da/vllm/config/cache.py#L294-L297),
and require zero local prefix-query and hit-counter deltas. Prefix caching is a separate
quarantined candidate requiring a fresh-process, varied-length growing-prefix
sequence, exact cleanup, and immediate typed failure on the first device fault;
an identical-prompt replay cannot promote it.

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

### Mixed-FP8 dense-projection checkpoint

The current
[`lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8`](https://huggingface.co/lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8)
head is `344f3a6820275dfcbb07d9c2a9d8b7ca1f37b3af`. It retains NVFP4 routed
experts while quantizing QSA attention and GDN input/output projections to
128-by-128 blockwise FP8 weight-only form. That directly targets the dense
BF16 GEMV/weight-traffic hypothesis rather than changing only PLE storage.

[vLLM issue #54125](https://github.com/vllm-project/vllm/issues/54125) reports
that the preview runtime incorrectly declares DeepGEMM supported for the whole
SM120 capability family, including GB10's SM121. Selecting DeepGEMM for these
FP8 projections hard-faulted during startup. With `VLLM_USE_DEEP_GEMM=0`, the
author attested the CUTLASS fallback and reported 23.7 tok/s single-stream,
156 tok/s at C16, and a 39% single-stream improvement over an
unquantized-dense build of the same model. Reported NLL/token was 0.7610 versus
0.7748 for that comparison.

Those figures omit a pinned model revision, MTP state, request count, prompt
shape and context, so they cannot be compared with this repository's 29.4
tok/s mapped-MTP2 anchor. The Hugging Face card labels the artifact a private
candidate and warns that stock SGLang silently mishandles its `FP8_PB_WO`
layers; its documented one-command route was verified on SM120, not GB10.

This is nevertheless the strongest new artifact-level performance hypothesis.
Test it only after the exact vLLM mmap baseline is admitted, as a separate
quality-first checkpoint arm with an immutable revision. On GB10, force and
attest `VLLM_USE_DEEP_GEMM=0` before model load and reject any resolved
DeepGEMM kernel rather than recovering after a device fault. Require matched
perplexity, the strict coding/cowork gates, varied-token long context and the
same MTP/runtime geometry before making a speed or quality claim.

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
| SGLang TRT-LLM sparse decode | SM121 excluded by the reviewed `qwen4-main-squashed` gate after corruption reports; no release support | Preserve only for historical provenance; the sole frozen-campaign exception is exhausted, and this route must not be used for new integration |
| SGLang SM121 Triton QSA | Open corrective PR directly atop the restriction; useful one-Spark correctness evidence, but unmerged and publicly red/gated | Leading SM121 Triton component candidate; build and admit before combining with storage |
| SGLang `io_uring` PLE streaming | Open stacked PR with exact-checkpoint data, but its head predates the SM121 safety restriction and its public checks are red | Rebase/force the SM121 Triton QSA fallback before testing the reader; use narrow syscall admission |
| vLLM direct PLE mmap | Open PR stacked on open model support; promising row and spot checks; a different mmap stack has a growing-prefix cache-on crash | Highest-priority post-cutoff reproduction target, cache-off first |
| Mixed-FP8 dense projections | Private-candidate checkpoint directly targets dense GEMV traffic; external speed claim lacks matched protocol and DeepGEMM hard-faults on SM121 | Quality-first arm after mmap baseline; force and attest CUTLASS |
| vLLM CPU PLE offload | Open PR; default container isolation blocks `pidfd_getfd` | Do not add `SYS_PTRACE` merely to benchmark it; prefer mmap |
| Compressed NVFP4 PLE | Pinned community checkpoint, minimal quality evidence | Quality-first research arm, not champion candidate |
| HashK PLE | Source unavailable and reconstruction/output warnings | Reject |
| Official recipe | No one-Spark GB10 NVFP4 lane | Continue to label local paths experimental |

## What this changes in the local optimization plan

The day-two evidence narrows the next work rather than changing the frozen
campaign. Step 1 preserves the frozen SGLang product campaign; step 2 retires
its historical runtime after cutoff; steps 3--11 are a separate post-cutoff
systems/reproduction sequence and do not reorder the future SGLang
serving-flag backlog:

1. Preserve the existing SGLang campaign's admission-expired, unmeasured state.
   Its bounded historical exception ended when less than one full pair budget
   remained; do not rerun or re-summarize it into a synthetic completion, and
   do not reuse its frozen plan as a deployment-safety claim.
2. After cutoff, make every historical profile bound to QSA overlay digest
   `e30566492e1502f94a4c7fed42d90b523bbb662580c628459e6e63c7b5263c75`
   non-runnable. The current manifest contains 21 such bindings. Add a
   fail-closed execution-admission tombstone keyed to that exact digest; do not
   rewrite their historical `support_status`, frozen raw manifests, fingerprints,
   or scalar evidence. Require distinct IDs for any SM121 Triton successor.
   Those 21 are also exactly the profiles using recipe revision
   `bf2b7c75870d3703730b6bd8f3bb93dc622c278d` with explicit
   `trtllm_mha` decode; no other overlay targets that QSA source under a
   different digest. Enforce the digest retirement both when selecting a new
   manifest profile and when loading an already-frozen plan; its embedded
   historical `support_status` must not bypass the post-cutoff gate.
3. Freeze the exact stacked PR #54129 tree rather than combining it with the
   newer #53896 head, then build an immutable Linux/aarch64 image. Use the
   pinned local Radix NVFP4-backbone/FP8-PLE checkpoint as the best-supported
   reconstruction target while clearly labeling the checkpoint identity as
   inferred until the author confirms it.
4. After the untouched author-shape control, apply only the `4df2ce2`
   live-width slice as a new source identity and require hash-ID/output parity
   across decode, MTP, prefill and same-token-count/different-layout replay.
5. Use direct read-only mmap without `SYS_PTRACE`, implicit downloads, mutable
   branches, wildcard cache selection, public API exposure, or prefix caching
   in the initial admission.
6. Build the SGLang storage candidate only from `04648a7` and `9f101e3` atop
   `3681c4e`; exclude `8ef3b3`. Add only the three `io_uring` syscalls to a
   pinned seccomp profile and test direct reads before any server launch.
7. Reuse the clean local D256 prompts and client geometry while freezing a new
   matched vLLM off/MTP2 pair. Retain the local MTP3/off pair as a historical
   numeric anchor only; any SGLang comparator must be newly built and admitted
   on SM121 Triton. Then run C1/C2/C4/C8 and the deterministic coding/cowork
   battery under the same scalar telemetry, fixed `max-num-seqs`, and explicit
   queue counters.
8. Add natural varied-token long-context trials around 60K, 120K, 160K, 190K,
   and 210K only as fresh-lifetime ascending admissions. Precompute token, KV,
   and workspace budgets; retain the existing MemAvailable, swap, and PSI
   gates; and admit the next tier only after the prior tier is correct and
   pressure-clean. On corruption, preserve a typed failure and restart rather
   than retrying inside a poisoned process.
9. Only after cache-off admission, run the resolved-`align` growing-prefix
   canary defined in the reproduction plan in a separate fresh lifetime. Do
   not add an `all` arm. Repeating a fixed prompt is not evidence that
   cache-resume state is safe.
10. Profile a fixed C1 decode span with CUDA/NVTX timing and process counters to
   partition target, launch, scheduler, and PLE page-fault costs. GB10 hardware
   counters require separate authorization and expose only LPDDR-facing
   proxies, not direct DRAM bytes.
11. Publish external claims and local measurements in separate tables; promote
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
