# Qwen3.8-Flash-Next vLLM direct-mmap reproduction plan — 2026-08-28

## Decision

The new vLLM direct-PLE-mmap branch is the highest-value post-campaign runtime
target, but the published 25.1 tok/s result is not exactly reproducible from
the pull request alone. Its code tree can be pinned; its checkpoint revision,
image/toolchain, complete launch, prompts, cache-state procedure, and timing
script were not published.

The best-supported checkpoint interpretation is the already-cached
[`RadixArk/Qwen3.8-Flash-Next-NVFP4` revision `7b71922`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594):
an NVFP4 backbone with an E4M3-FP8 PLE table. The pull request calls its input
an "FP8 checkpoint," but its reported roughly 78 GiB resident backbone plus
47.68 GiB PLE footprint does not fit the official all-FP8 checkpoint. The PR
also directly addresses a single-Spark issue whose target is the Radix model.
This identification remains an inference until the author confirms the model
ID.

Therefore:

- do not download the 172.82 GiB official all-FP8 repository merely to follow
  an ambiguous phrase in the PR;
- use the complete local Radix revision as the explicitly inferred first
  candidate, while preserving the provenance limitation;
- build the exact PR source into a new digest-pinned arm64 image rather than
  overlaying its Python files on an older cached runtime; and
- do not start that work until the current frozen SGLang campaign has reached
  its cutoff or a real terminal state and the host passes the unchanged clean
  swap/memory admission gates.

This plan is separate from the frozen 84-file campaign harness. It does not
authorize changing that harness before cutoff.

## What is and is not pinned upstream

[vLLM PR #54129](https://github.com/vllm-project/vllm/pull/54129) is open,
non-draft, and unmerged. The reviewed source boundary is:

| Component | Exact revision | Role |
| --- | --- | --- |
| Full PR head | [`8e4e036a311604800334989485b4ee23925956da`](https://github.com/Trosfy/vllm/commit/8e4e036a311604800334989485b4ee23925956da) | Complete tree to build |
| Direct-mmap change | [`eae5aa8fb15c3af1a8ebc23b0d027f465c6c57f3`](https://github.com/Trosfy/vllm/commit/eae5aa8fb15c3af1a8ebc23b0d027f465c6c57f3) | Six-file functional change |
| Model-support snapshot in the stack | [`2a4cd640ff1a61b66124ddbaaf02a73781f7295a`](https://github.com/peakcrosser7/vllm/commit/2a4cd640ff1a61b66124ddbaaf02a73781f7295a) | Older #53896 state actually included by this PR |
| First stack merge | [`7bf07f0ab59b3bad389c7ea232e9c02f60d7186f`](https://github.com/Trosfy/vllm/commit/7bf07f0ab59b3bad389c7ea232e9c02f60d7186f) | Merges `2a4cd640` with upstream base `b2a6e9dc` |
| First upstream base | [`b2a6e9dca0c2e62142b96adafceb99fbeb3e60c3`](https://github.com/vllm-project/vllm/commit/b2a6e9dca0c2e62142b96adafceb99fbeb3e60c3) | `main` parent of the first stack merge |
| Final upstream base | [`6f7df92a8e6cdc74a725b8f10b4d0b48ba2b37ef`](https://github.com/vllm-project/vllm/commit/6f7df92a8e6cdc74a725b8f10b4d0b48ba2b37ef) | `main` merged with the mmap series to create the reviewed head |

The exact ancestry is
`2a4cd640 + b2a6e9dc → 7bf07f0a → eae5aa8f`, followed by
`eae5aa8f + 6f7df92a → 8e4e036a`.

The head resolved the conflict reported earlier in the PR and removed the
`needs-rebase` label. It remains open and had no upstream code-test CI result
at this cutoff: the public Actions job stopped at contributor/label
eligibility, not at a model test. The author's reported tests are local.

Do not replace the included model-support snapshot with current
[PR #53896](https://github.com/vllm-project/vllm/pull/53896) head
`89d0bb71aeb2f3e15c16efc69d33c3fbe223a765` in a purported reproduction.
That would create a new integration arm. PR #54129 explicitly does not depend
on CPU-offload [PR #53899](https://github.com/vllm-project/vllm/pull/53899),
so it needs neither its worker process nor `pidfd_getfd`/`SYS_PTRACE`.

The functional change touches:

- the Qwen4Exp NVIDIA model and PLE layer;
- a new direct-mmap implementation;
- vLLM environment and compilation configuration; and
- a synthetic mmap test module.

It is stacked on the much larger model-support series. Copying only those six
files into a cached pre-support image is not a faithful build.

## Checkpoint inference

The PR body discloses only "FP8 checkpoint," without a repository ID or
revision. Three facts make the local Radix snapshot the strongest candidate:

1. The PR describes an approximately 78 GiB resident backbone plus a 47.68 GiB
   FP8 PLE table on one roughly 120 GiB unified-memory host.
2. Its motivating [single-Spark issue #53960](https://github.com/vllm-project/vllm/issues/53960)
   uses `RadixArk/Qwen3.8-Flash-Next-NVFP4`.
3. Radix revision `7b71922` contains ModelOpt NVFP4 packed backbone weights but
   excludes the PLE tensors from NVFP4 conversion; its 51.2B E4M3 parameters
   account for the reported 47.68 GiB PLE.

By contrast, official
[`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/bcd9f01ddc9cff2316eb84281bebcd5b058bddce)
contains 172.78 GiB of safetensor shards. Removing a 47.68 GiB PLE still leaves
more than 120 GiB, inconsistent with the reported backbone footprint and
one-Spark admission. The audit-time
[`970c569`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/970c569adaca6b35532111fd6b27351b2baefe50)
head changes README content only and has the same tensor manifest as the
weight pin `bcd9f01`.

The direct-mmap implementation itself accepts E4M3-FP8 PLE shards and rejects
E5M2 and non-FP8 PLE storage. It does not prove that every other tensor in the
checkpoint is FP8. The local run must describe its arm as "Radix NVFP4
backbone with exact FP8 PLE mmap," not "official FP8," unless the author
supplies contrary provenance.

## Local readiness audit

The read-only audit at `2026-08-28T09:52:07Z` found:

| Requirement | Local state | Decision |
| --- | --- | --- |
| Radix candidate | Complete `7b71922` snapshot; 125.91 GiB tensor bytes | Ready and already pinned |
| Official all-FP8 alternative | Absent; remote repository is 172.82 GiB total | Do not acquire without clarification |
| PR source/runtime | No checkout, wheel, package, profile, or image at `8e4e036` | Must build after cutoff |
| Official Flash-Next image | Digest recorded in prior research but not cached; labels did not identify an exact source commit | Not a provenance-complete substitute |
| Cached `vllm-openai:qwen38` | Older arm64 Qwen3.8-27B build at `3a0914`; predates Flash-Next PR | Reject for this experiment |
| NVMe capacity | 1,356.00 GiB available | Build/acquisition capacity is sufficient |
| Unified memory | 119.69 GiB total; 111.79 GiB available at audit | Capacity looks plausible, not an admission pass |
| Swap | 0.85 GiB used versus the frozen 64 MiB clean-start cap | Blocks benchmarking |
| Containers | None running | Clean on this dimension |

The source tree is small enough to fetch, but a correct arm64/CUDA build is not
already present. The repository defaults at this head include Python 3.12 in
Docker, PyTorch 2.13.0, FlashInfer 0.6.17, CUDA 13.0.3, Ubuntu 24.04 and NCCL
2.30.7. Those are source defaults, not the author's environment. Some base
images are referenced by mutable tags and several Python requirements are
ranges, so the build must resolve and record actual content digests and a full
package freeze.

## Branch behavior that must be preserved

The branch's default-off path is activated with the following exact settings:

| Setting | Frozen baseline | Reason |
| --- | ---: | --- |
| `VLLM_PLE_MMAP` | `1` | Enable direct safetensors mapping |
| `VLLM_PLE_MMAP_WORKERS` | `32` | Code default; gather thread count |
| `VLLM_PLE_MMAP_CHUNK` | `2048` | Code default; rows per gather task |
| `VLLM_PLE_MMAP_PREWARM` | `0` | Code default; keep cold/resident behavior explicit |
| Tensor parallelism | `1` | One-Spark route |
| Maximum context | `32768` for author-shape admission | Protocol resolution of the published approximate "32K" geometry |
| Speculation | MTP, two speculative tokens | Published smoke geometry |
| Modality | Text only | Published smoke geometry |
| CUDA graph mode | `PIECEWISE` | Author-recommended reconstruction setting |
| Maximum batched tokens | `8192`, subject to resolved-config confirmation | Source-derived OpenAI-server default on a non-A100 device reporting at least 70 GiB; not published by the author |
| Maximum sequences | `1024`, subject to resolved-config confirmation | Same source-derived default; a feasibility control, not the eventual single-user setting |

The source refuses full and full-and-piecewise graph modes and requires
`CompilationMode.VLLM_COMPILE` when mmap is on. It also refuses an
operator-supplied split list that omits
`vllm::qwen4_exp_ple_mmap_forward`. The guard does not reject graph mode
`NONE`; `PIECEWISE` is the frozen author-shape reconstruction, not the only
mode accepted by code. The run must fail admission if the server silently
falls back, ignores an option, or cannot prove that the op is split outside
capture.

For a repository ID, the mmap resolver calls an offline local-only snapshot
lookup. Serving an absolute verified snapshot directory is simpler: retain
original shard filenames, tensor names, layout and offsets; mount the snapshot
read-only; and forbid reload to a different checkpoint inside one process.
The tracked record contains only model/revision/file hashes, never the local
path.

The author also reported a 2,100 MHz GPU clock cap. Do not change the host's
clock merely to imitate that observation. Record the actual clock policy and
keep any external 25.1 tok/s comparison descriptive.

## Pre-build compatibility and integration audit

A read-only audit of the complete local Radix snapshot against the exact
`8e4e036` mmap implementation found no static blocker for a fresh-process,
immutable-local-directory, TP1, text-only launch. The header and geometry gate
passes:

| Check | Pinned Radix result |
| --- | --- |
| Safetensors set | 206 files; every header parses, every tensor offset is in bounds, and tensor names are unique |
| Header cap | Largest header is 203,584 bytes, below the branch's 100 MiB limit |
| PLE topology | Exactly one PLE layer: zero-based layer 1, matching `ple_layer_ids=[2]` |
| PLE shards | All 128 indices `0..127`; each is `[2,500,012, 160]` E4M3 FP8 |
| PLE total | 320,001,536 rows and 51,200,245,760 bytes, or 47.683945 GiB |
| Scale | One exact-name `[1]` BF16 scale tensor, a dtype accepted by the branch |

The 160-byte row width follows independently from `ple_embed_dim=2560` divided
by `(ngram_size - 1) * heads_per_ngram = 16`; the model's 256-wide attention
heads are unrelated. This establishes compatibility with
[`discover_shards`](https://github.com/Trosfy/vllm/blob/8e4e036a311604800334989485b4ee23925956da/vllm/models/qwen4_exp/nvidia/ple_mmap.py#L183-L268)
and the branch's header/shape predicates. It does **not** prove that the real
loader intercepts every Radix shard and scale correctly, that the
streamed-versus-direct scale values match, that the ModelOpt NVFP4 backbone
loads, or that the compiled custom op executes on GB10. No large tensor payload
or scale value was read during this audit.

The same review found two concrete integration defects outside the frozen
fresh-process protocol:

- **Same-path hot reload is unsafe.** The layer discards streamed PLE shard
  tensors and may replace its scale, but
  [`build_tables`](https://github.com/Trosfy/vllm/blob/8e4e036a311604800334989485b4ee23925956da/vllm/models/qwen4_exp/nvidia/ple_mmap.py#L790-L815)
  skips an already attached table when the path string is unchanged, while
  header discovery is cached only by path. An iterator-based reload or an
  in-place snapshot replacement can therefore mix a new backbone/scale with
  old mapped rows. Disable weight reload for this arm, keep the mounted
  snapshot immutable, and require a full process restart for any checkpoint
  change.
- **Repository-ID resolution can disagree with the real loader.** The mmap
  [`resolver`](https://github.com/Trosfy/vllm/blob/8e4e036a311604800334989485b4ee23925956da/vllm/models/qwen4_exp/nvidia/ple_mmap.py#L641-L659)
  does not carry the loader's custom download directory or non-default weight
  source through its offline lookup. Pass the exact completed snapshot
  directory as the model. Repository IDs, `model_weights` indirection,
  RunAI/object sources, and custom download-directory resolution are outside
  this reproduction.

There is also a fail-closed validation gap: the mmap discovery glob silently
uses the last copy of a duplicate `(layer, shard)` or scale tensor, independent
of the default loader's index-based duplicate filtering. The pinned Radix
manifest has no duplicate tensor names, but unique PLE tensor ownership is now
an explicit admission check.

Finally, the branch's large synthetic test module is not an end-to-end model
test. Its custom-op tests cover CPU dispatch, schema/fake behavior and graph
guards, but do not execute the op under real CUDA compilation and piecewise
capture; its model-load hook tests stub the loader and table builder. The first
GPU gate must therefore prove compile, capture and replay at two request shapes
and across a second request, then run the mmap-versus-`safe_open` oracle. The
claim remains text-only; multimodal execution is not established by this PR.

## Static performance model and tuning order

The mmap path is a capacity-enabling design, not a zero-copy path. Each PLE
invocation sits on the token critical path as:

```text
eager GPU n-gram hash
  -> blocking GPU-to-CPU ids
  -> CPU sort/dedupe and task plan
  -> blocking thread-pool mmap gather/page faults
  -> CPU inverse-order copy
  -> effectively synchronous pageable CPU-to-GPU rows
  -> device output copy
  -> FP8 dequantization
```

The source makes the whole hash-plus-gather operation opaque to compilation
and graph capture, then performs an explicit blocking
[`ids.to("cpu", non_blocking=False)`](https://github.com/Trosfy/vllm/blob/8e4e036a311604800334989485b4ee23925956da/vllm/models/qwen4_exp/nvidia/ple_mmap.py#L545-L573).
The CPU gather uses `np.unique`, advanced-index copies, a blocking pool map and
a final `out[inverse]` copy before the pageable H2D transfer
([implementation](https://github.com/Trosfy/vllm/blob/8e4e036a311604800334989485b4ee23925956da/vllm/models/qwen4_exp/nvidia/ple_mmap.py#L343-L409)).
The custom op then copies the device result into its traced output buffer
([implementation](https://github.com/Trosfy/vllm/blob/8e4e036a311604800334989485b4ee23925956da/vllm/models/qwen4_exp/nvidia/ple_mmap.py#L941-L968)).

For this checkpoint's one PLE layer, each logical token produces 16 int64 ids:
128 bytes copied D2H, at most 2,560 bytes of useful FP8 rows gathered and copied
H2D, another 2,560-byte device output copy, and a 5,120-byte BF16 dequantized
result. These payloads are tiny. Warm cost is more likely to come from the
eager boundary, synchronization, allocations and small-task dispatch than raw
row bandwidth. On a cold random lookup, sixteen distinct 160-byte rows require
at least sixteen 4 KiB pages, a 25.6x page-granularity amplification before
boundary crossings or readahead.

The highest-value static finding is separate from mmap I/O: PLE hashing fills
the full second dimension of a `[num_reqs, max_num_batched_tokens]` workspace,
builds shifted contexts and hashes that workspace, then selects the currently
scheduled token positions only at the end
([workspace](https://github.com/Trosfy/vllm/blob/8e4e036a311604800334989485b4ee23925956da/vllm/models/qwen4_exp/nvidia/ple_layer.py#L303-L315),
[hash](https://github.com/Trosfy/vllm/blob/8e4e036a311604800334989485b4ee23925956da/vllm/models/qwen4_exp/nvidia/ple_layer.py#L352-L420)).
Decode hash work therefore scales with actual request count times the configured
batched-token ceiling, not merely the current decode-token count. At this head,
an OpenAI server on a non-A100 device reporting at least 70 GiB defaults to
`8192` batched tokens and `1024` sequences
([defaults](https://github.com/Trosfy/vllm/blob/8e4e036a311604800334989485b4ee23925956da/vllm/engine/arg_utils.py#L2627-L2645)).
The runtime must print and verify the resolved values; relying on automatic
hardware classification is inadmissible after the author-shape control.

This waste has since received an upstream fix, but the mmap branch does not
contain it. Commit
[`4df2ce2`](https://github.com/peakcrosser7/vllm/commit/4df2ce22d086007a81930d93b3b657a1d197aecc)
changes the packed view from all columns to
`self.padded_buffer[:num_reqs, :num_tokens]` in the newer model-support branch.
That leaves the maximum-size allocation in place but makes fill and hash work
scale with the live physical token width. Relative to a ceiling `M` and a live
width `T`, fill work falls by `M/T` and the leading trigram-hash dimensions by
approximately `(M + 2)/(T + 2)`. Record the actual `T` on every call: ordinary
C1 decode is expected to be near one, while MTP verification may be wider.

The safest first optimization is therefore the exact mmap head plus only that
one-line semantic change. The mmap head already places the whole hash-and-gather
custom op outside compilation and graph capture, so this candidate removes
wasted work without changing the capture boundary. It is a new integration
identity, not the unmodified author branch, and must receive its own source-tree
hash. The author-shape control remains the untouched `8e4e036` head.

Do not fold the newer branch's
[`0e0802f`](https://github.com/peakcrosser7/vllm/commit/0e0802f4637c73589c9943d420758177df454d9a)
hash-only split into this first candidate. That change protects a later
gather-only graph boundary from same-token-count/different-request-layout
capture reuse; it adds no benefit while mmap still splits the whole operation.
If that boundary is narrowed later, rebase at or after `0e0802f`, retain both
the hash-only and mmap-gather operators in the split list, and compare the
two-op design independently.

This review cannot establish that PLE is the overall warm-TPS limiter. The
backbone and active-expert LPDDR path may still dominate, consistent with the
independent evidence summarized in the
[TPS bottleneck audit](qwen38-flash-next-tps-bottleneck-2026-08-28.md).
Instrument PLE phases and the whole token concurrently; do not infer total-TPS
causality from a faster microphase.

After two unchanged baseline lifetimes agree, change one axis at a time:

1. reproduce the unmodified `8e4e036` control, then apply only the live-width
   slice from `4df2ce2`; require identical IDs and outputs across C1, C2, MTP
   verification, unequal request lengths, prefill, and same-`T`/different-layout
   replay, with no new compile or capture;
2. retain the winning hash implementation and freeze `max-num-seqs=2`, the
   actual C2 product ceiling, while retaining the `8192` batched-token control;
3. with sequences fixed at two, compare `max-num-batched-tokens=2048` against
   `8192`; bracket with `4096`, then `1024`, only when the first contrast is
   material, scoring both resident agent-task wall and 8K/32K prefill wall.
   After the live-width patch this is a scheduler and prefill axis, not a decode
   PLE-hash workaround;
4. at the winning scheduler geometry, compare mmap workers 16 versus 32;
   changing the 2,048-row chunk cannot alter ordinary C1/C2 decode topology,
   while MTP2 can make scheduled tokens per forward exceed request count;
5. add a measured small-call serial/thread-pool-bypass arm only if pool
   enqueue/wait is at least 10% of PLE wall;
6. compare the existing whole-forward split against the later hash-only plus
   gather-only design, stopping on any recompile, constraint violation or
   output mismatch;
7. test persistent pinned staging only if H2D is at least 5% of PLE wall, and
   test chunk size only on prefill calls whose unique rows per shard exceed the
   current chunk; and
8. keep prewarm as a cold-start experiment, never a resident-TPS arm. Its
   built-in 8 GiB headroom is below this repository's 14 GiB safety floor, so
   it requires a safer bound before admission.

Per-call instrumentation must include actual requests and scheduled/MTP
tokens, raw and unique ids, shards/tasks/active workers, hash CUDA time,
blocking D2H wall, CPU plan/pool/page-copy/inverse-copy wall, H2D and output-copy
wall, dequant time, total custom-op wall, minor/major faults and process NVMe
bytes. Pair that with graph launches, device idle gaps, LPDDR traffic, CPU run
queue, requested-page hit rate, MTP acceptance and end-to-end TTFT/ITL/task
wall. The existing `pending` log field is task count, not observed worker
concurrency.

Stop a worker search when adjacent settings differ by less than 3%. Promote a
candidate only when two fresh lifetimes move in the same direction, improve
correct task wall or PLE wall by at least 5% (or decode TPS by at least 3%
outside measured drift), retain exact oracle/greedy correctness, and preserve
the memory, swap, pressure and latency gates.

## Acquisition and build phase

This phase may use the network, but it must be separate from measurement:

1. fetch the exact Trosfy source head and verify the commit graph above;
2. resolve every base image to its Linux/aarch64 child digest before build;
3. build without overlaying files into an older vLLM image;
4. record the resulting OCI digest, source tree digest, compiler/toolchain,
   CUDA, PyTorch, FlashInfer, vLLM and complete installed-package freeze;
5. run the branch's checkpoint- and GPU-free synthetic mmap tests, the
   header-only validator against the pinned snapshot, and a CPU loader-seam
   test that proves shard/scale interception without recording their values;
   and
6. disable network access for admission and all measured lifetimes.

The author reported:

```text
pytest tests/models/qwen4_exp --ignore=tests/models/qwen4_exp/test_qsa_amd.py
205 passed
```

The AMD exclusion was attributed to a pre-existing duplicate-registration
collection problem. Reproduce the same targeted command and separately record
the exclusion; do not relabel it as a complete upstream suite. Also run the
branch's pre-commit checks, but do not treat either local result as upstream CI.

## Phase 1: branch admission on the inferred Radix checkpoint

The first managed lifetime answers only whether the pinned branch safely runs
the inferred checkpoint. It is not yet a backend speed comparison.

Admission requires:

- the original clean-start gates: at least 14 GiB `MemAvailable`, used swap at
  most 64 MiB, no unrelated GPU/container workload, loopback plus ephemeral
  authentication, and an owned-container cleanup boundary;
- exact source/image/checkpoint/tokenizer hashes and zero implicit downloads;
- read-only direct mapping of the checkpoint shards, no offload worker and no
  added container capability;
- piecewise compiled graphs with the mmap op present in the split set;
- a 1,000-row mmap-versus-`safe_open` oracle whose deterministic seed,
  sampling protocol and digest are frozen before launch, with 1,000/1,000
  exact matches; and
- scalar confirmation of zero GPU-resident PLE bytes, with cold and resident
  page counts recorded separately.

Use no optional prewarm. Capture start-to-ready, target/MTP load components,
first correct request, minimum available memory, swap delta, PSI, PLE residency,
page faults, NVMe reads, native proposed/accepted counts, and exact cleanup.

Reconstruct the published shape as a clearly derived protocol: TP1, 32K,
text-only, greedy, MTP2, three unique unscored warmups, then three unique
900-token requests. The author did not publish prompts, request JSON, maximum
sequence count, KV dtype, cache policy, batched-token limit, backend choices,
or timing script. Freeze all of those locally and do not claim numerical
reproduction of 25.1 tok/s.

Run the deterministic structured, tool, exact-answer and code-reasoning gates
before promoting the branch to a comparative panel. Any wrong-output,
token-zero/repetition degeneration, missing native counter, pressure breach,
or lifecycle failure rejects admission even if output TPS is high.

## Phase 2: matched single-user comparison

Only after Phase 1 passes should vLLM and SGLang be compared. Both bundles use
the same Radix revision, tokenizer, rendered prompts, MTP2 depth, temperature,
request limits and client. This is a runtime-bundle comparison, not a causal
isolation of mmap from every other engine difference.

Freeze two C2-capable 64K bundles and use fresh-lifetime ABBA order:

1. SGLang A;
2. vLLM A;
3. vLLM B; and
4. SGLang B.

The independent lifetime is the replicate (`n=2` per arm). Reverse serial and
parallel block order in B. Never pool individual requests into a false sample
count.

Here 64K is a shared server cap/pool, not permission for two 64K requests.
Freeze maximum running/sequences at two, graph batches one/two, and 65,536
tokens as the context/pool cap. The SGLang bundle retains eight lazy recurrent
slots; the vLLM bundle must precompute and freeze equivalent KV, recurrent,
QSA, workspace and batched-token capacity. For every C2 pair, the two fully
rendered inputs plus reserved outputs and MTP allowance must total no more than
61,440 tokens, leaving 4,096 tokens unused. Two near-60K requests are
prohibited. Moving vLLM from its public 32K smoke to this 64K/C2 geometry is a
new admission, not inherited support.

Before SGLang A starts, freeze exactly the four existing strict agentic cases
(`agentic-select-and-call`, `agentic-no-tool`, `agentic-two-hop`, and
`agentic-tool-error-recovery`) with three variants each for all four lifetimes.
These task requests use thinking enabled at low effort. Do not add cowork cases
after the ABBA panel begins; admit and freeze `cowork-core-v1` as a later,
separate campaign. D256 remains explicitly no-thinking.

The smallest scored panel keeps three outcomes separate:

| Outcome | Shape | Promotion rule |
| --- | --- | --- |
| Task latency | Frozen 12-episode strict agent/tool suite at C1, thinking enabled/low effort | Correctness first; both vLLM lifetimes must be directionally faster and the unweighted arm mean must improve at least 5% |
| Decode TPS | Six unique no-thinking D256 requests at C1 | Full requested output and valid finish reasons required; report full case wall, TTFT and E2E |
| Service fan-out | Two byte-distinct, token-matched six-request D256 sets, one serial and one as three C2 pairs | Report time-to-all, queue/preemption and fairness; make no decomposed coding/cowork claim |

The two D256 sets have no shared prefix. A assigns set one to serial and set
two to C2; B swaps the sets and reverses block order. Each mode must complete
exactly six 256-token outputs. This prevents the second block from measuring
prefix reuse as well as concurrency. Synthetic D256 fan-out is only a service
capacity/time-to-all result, not proof that one coding or cowork task benefits
from parallel decomposition.

Pin `max-num-seqs` and prove running/queued occupancy. Public vLLM data changed
about fourfold at C8 when only that cap changed, so an offered-concurrency
label without scheduler counters is insufficient.

Pi/Harbor is intentionally not part of the first runtime comparison. It is a
new client-stack variable. Run the pinned Pi/cowork battery only after a
runtime winner survives the common native client and correctness gates.

Three common unscored warmups define resident state for each core lifetime.
Report start-to-ready and ready-to-first-correct separately; never add them to
or subtract them from resident task wall or D256 TPS. Optional mmap prewarm
remains off. Only fresh SGLang A/B lifetimes are the comparator. Historical
mapped-lazy D2 values—29.402 D256, 29.594 C1 and 51.870 C2 tok/s—used 4K
geometry and are descriptive sanity checks, not a 64K baseline or samples to
pool.

## Long-context and profiler gates

Long-context correctness uses separate C1 profiles/lifetimes with a 131,072
maximum context and token pool so memory pressure does not contaminate the
scored resident panel. Before launch, budget the 120K input, 128-token output,
MTP allowance, QSA/workspace and KV allocations; target BF16 KV alone is about
2.75 GiB at 120K, excluding MTP, GDN, QSA and allocator overhead.

- two varied-token 60K exact-key trials at different seeds/depths;
- only after both are correct and pressure-clean, four varied-token 120K
  trials, the minimum panel that directly addresses the public 1/4 stochastic
  corruption report; and
- stop on the first corruption, token-zero stream, non-finite output signal,
  OOM, preemption, 14 GiB memory-floor breach, or 512 MiB swap growth.

Preserve the typed failure, restart the process, and do not retry inside a
possibly poisoned engine. Do not escalate to 160K/190K/210K in the minimum
panel. A branch needs 2/2 at 60K and 4/4 at 120K for any bounded
long-context claim. Four trials detect a true independent 25% per-request
corruption rate with only `1 - 0.75^4 = 68.36%` probability, so 4/4 is a
minimum reproduction screen, not certification that the failure is absent.

Profiler lifetimes are also separate and unscored: SGLang-P and vLLM-P, both
MTP-off. After two profiler warmups, capture three C1 D256 requests, three C2
pairs of D256 requests, and one separately pressure-admitted varied-token 60K
prefill window. Run an identical unprofiled calibration span and publish the
profiler overhead. Raw profiled TPS must not be compared with the MTP2 product
score. Resolve GB10 counter names before freezing the plan and retain only
allowlisted, process- and phase-attributed scalar aggregates for:

- LPDDR read/write bandwidth and L2 traffic;
- kernel union time, launch count/duration and inter-kernel host gaps;
- SM activity, achieved occupancy, eligible warps and tensor-pipe activity;
- graph replays versus ordinary launches;
- process-attributed page faults, UVM movement, NVMe reads and PLE residency;
- CPU submit/scheduler time and context switches;
- scheduler running/queued/preemption/queue time; and
- high-rate power, clocks, temperature and throttle state.

Raw profiler traces remain private. Coarse GPU utilization alone cannot decide
between bandwidth, occupancy, launch/scheduler, or PLE-I/O limits.

## Time and promotion boundary

The source build and any acquisition are a separately capped preparation phase
and must not consume a frozen measurement window. Once all artifacts are
local, the plan contains nine managed lifetimes: one Phase-1 admission, four
core ABBA lifetimes, two long-context lifetimes and two profiler lifetimes.
Budget about 4–5.5 hours plus any separately admitted cowork campaign; enforce
a six-hour hard cap including validation, export, checkpointing and cleanup.
Freeze 1,800 seconds per startup, 2,700 seconds per admission/core/long
lifetime, and 1,800 seconds per profiler lifetime.

Do not begin an ABBA pair or long/profiler phase unless the complete next unit
plus cleanup/evidence reserve fits. At the cap, preserve partial and failed
outcomes rather than relaxing pressure or correctness gates.

A local win remains **pinned experimental** because both upstream PRs are
open. It may be called the task-latency winner only when both independent
vLLM lifetimes match SGLang's strict success, both are directionally faster
than their paired SGLang lifetimes, and the unweighted arm mean correct-task
wall improves by at least 5%. Within ±5% at `n=2` is no winner. D256 TPS,
fan-out capacity, and long-context safety remain separate conclusions.
