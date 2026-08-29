# Qwen3.8-Flash-Next SM121 Triton/storage pre-admission — 2026-08-28

## Outcome

The newly composed SM121 Triton plus read-only `io_uring` PLE candidate has
passed its build, storage, and focused GPU-kernel prerequisites on the local
GB10. It is **not admitted to serve the model**. A dedicated first-run canary
is now implemented, but it has not loaded the checkpoint or started a server.
No throughput, quality, or long-context result is claimed here.

This is deliberately a mechanical pre-admission record. It narrows the next
work from source/build uncertainty to an end-to-end correctness and performance
protocol, while preserving the permanent exclusion of the historical SM121
TRT-LLM sparse-decode route.

## Fixed candidate identity

| Field | Value |
| --- | --- |
| Candidate | `sglang-sm121-triton-storage-v1` |
| Source tree | `274ee330db7ea9653807b868c0fb8693d50ed7b2` |
| Build contract SHA-256 | `c9c7c5bb958a8cf4c0fbc904b40c5e51fac82ef97c6e1fc391e2b67b5c9d9975` |
| Local Docker image ID (not a registry digest) | `sha256:b14c39fb7cb2e0b82f2f8cae1e115a55f2bb69b5ec6fd7ccc4099b219d1096b0` |
| Platform | `linux/arm64` |
| Device | NVIDIA GB10, compute capability 12.1 |

The source preserves the exact-SM120 restriction for TRT-LLM sparse decode.
On SM121, the resolver selected `qsa_sm121_varlen_attention` and reported no
TRT-LLM sparse-decode implementation. This confirms the intended dispatch
boundary only; it is not a full-model output check.

The image was built from the fixed source-tree archive using the pinned
Linux/ARM64 CUDA base. Its build is reproducible as a source and invocation
boundary, not as a complete dependency closure: external package resolution
during the image build remains mutable. There is no registry manifest digest
for this local image; its Docker config ID, platform, and source-tree labels
form the execution identity.

## Narrow storage and syscall result

The default Docker seccomp policy rejected an `io_uring` setup probe with
`EPERM`, as expected. The candidate uses a per-container derived policy, never
a daemon-wide or unconfined policy. It is the Docker 29.2.1 baseline plus one
unconditional allow group containing exactly:

- `io_uring_setup`
- `io_uring_enter`
- `io_uring_register`

The baseline SHA-256 is
`01536f1d1df938ae611eba20d6349e0de7a99b6ecdee1549427a0b01b8301e28`; the
derived profile SHA-256 is
`1c9c9ffc77260ddc8361f0443bac881348324b00b732d5cfabde61a239ff5b62`.
The tracked profile contract verifies that the derived JSON is exactly that
baseline plus the three-syscall group. The target container remained
network-disabled, read-only, capability-dropped, and `no-new-privileges`; only
ephemeral compiler/cache directories were writable.

With runtime Rust building disabled, the compiled `_storage` extension imported
and exposed `IoUringReader`; its artifact SHA-256 was
`213c79d463a941cd65ee5cb9a93977b510d1f647fd2c0db690cc0f65d3799bf4`.
The upstream io_uring unit test passed without an `EPERM`/`ENOSYS` skip. A
synthetic direct-I/O safetensors gather agreed with both memory-mapped and
`safe_open` reads for four rows.

The cached Radix PLE manifest also parsed without a model load:

| PLE property | Result |
| --- | ---: |
| Shards | 128 |
| Files | 10 |
| Embedding dimension / row bytes | 160 / 160 |
| Total rows | 320,001,536 |
| Tensor bytes | 51,200,245,760 |
| Dtype | F8_E4M3 |

Under the same restricted container, a 1,000-row, zero-cache, queued-direct-I/O
oracle produced 1,000 exact matches against both reference readers. The sampled
row identifiers are represented only by their set digest
`f684acfd6103ee339a3d6cace1b22315cd70bc1d8c1258cb6eae90a89cf0a2c6`; row data
and model payloads were discarded.

## SM121 Triton kernel result

Eight real-GPU QSA tests passed under the candidate image:

| Gate | Passing cases |
| --- | ---: |
| Packed-varlen numerical comparison to Torch | 4 parameterizations |
| CUDA-graph replay with dynamic lengths | 1 |
| Compaction plus sparse-attention reference comparison | 1 |
| Compressed-QSA page-64 graph metadata equivalence | 1 |
| Speculative rows plus padded-tail graph layout | 1 |

These are source-kernel tests with CUDA visible and isolated temporary compiler
caches. They cover the new SM121 Triton fallback, CUDA graph replay, and the
full-page-64 compressed-QSA bookkeeping used by this PLE path. They do not
exercise model loading, sustained serving, MTP acceptance, request scheduling,
or varied-token long-context behavior.

## Implemented first execution gate — not yet run

The repository now has one deliberately narrow, target-only canary lane for
this candidate. Ordinary planning, matrix, benchmark, and resume paths reject
the profile; the dedicated lane accepts only its fixed image, checkpoint,
profile, and two-case suite. It freezes the local Docker ID, platform, and
source-tree label in the plan, then rechecks the mutable tag and launches by
the immutable ID. The container remains offline, loopback-published,
read-only, capability-dropped, and subject to the derived `io_uring` seccomp
profile.

The lane runs two non-resumable server lifetimes, each with no warmup or primer:

1. `synthetic-exact-answer-v2` runs the existing strict quality validator twice
   with a 512-token output cap.
2. `sm121-varied-context-needle-19000-mid-s20260828-c1-v1` is one deterministic
   retrieval request with 19,000 two-word filler records and a unique 12-word
   recovery phrase at the midpoint. It accepts only the phrase in order,
   normalizing whitespace and case but not punctuation, and has a 64-token
   output cap.

The varied request is the first inference after its own server is ready; it
does not reuse the quality server, a cache, a partial journal, MTP, a draft
overlay, CUDA graphs, or a repeated-word-only context. Its configured context
limit is 65,536. Offline verification against the pinned target tokenizer and
the `enable_thinking=false` chat template counted 62,336 input tokens; the
fixed admission budget is 62,400 after the 64-token output allowance. The
generator's prompt SHA-256 is pinned in the regression contract without
retaining its text. A first run must still record its actual scalar prompt
accounting before it can support a long-context claim.

Prompts, completions, request identifiers, and credentials remain outside the
tracked record. A completed canary may publish only exporter-approved scalar
evidence after teardown and verification. The quality lifetime is a hard
admission gate: any failed synthetic item tears down that server and prevents
the long-context lifetime from starting. The read-only canary auditor requires
the two ordered fresh lifetimes, scalar runtime provenance immediately before
each ready event, no primer, and terminal cleanup before scalar evidence can
be exported.

## What remains blocked

The runtime-attestation schema intentionally accepts only an `admitted` record
when all of the following are true: retired-overlay rejection, storage import,
io_uring, PLE row comparison, SM121 Triton, quality, and long-context. The
first five prerequisites are now demonstrated; the last two are false until
the implemented target-only, cache-off end-to-end canary completes.
Implementing the lane is not an admission result.

Before any performance comparison or cache experiment, the next protocol must:

1. pin the immutable image, model, tokenizer, revision, and serving profile;
2. start a new loopback-only target-only server with the read-only PLE mount;
3. complete a quality-clean synthetic exact-answer canary;
4. complete a varied-token long-context canary at matched prompt/output limits;
5. verify teardown and scalar-only evidence export; then
6. run a cold A/B/B/A cache policy experiment in separate process lifetimes.

The retired overlay and all historical TRT-LLM measurements remain excluded
from this candidate. Passing this pre-admission does not license a speed claim
or an overnight cache campaign.

## Repository verification

The new seccomp contract and runtime-attestation validators are fail-closed and
scalar-only. Their focused suite has 21 tests; the complete repository suite
passed 968 tests, and module compilation passed. No raw result bundle or
evidence archive was written because these are infrastructure prerequisites,
not a completed SparkBench measurement.
