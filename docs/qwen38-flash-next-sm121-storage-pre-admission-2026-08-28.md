# Qwen3.8-Flash-Next SM121 Triton/storage pre-admission — 2026-08-28

## Outcome

The newly composed SM121 Triton plus read-only `io_uring` PLE candidate has
passed its build, storage, and focused GPU-kernel prerequisites on the local
GB10. Its target-only first-run canary has now completed quality-clean and
varied-token 64K retrieval checks in two fresh server lifetimes. It is still
**not generally admitted for serving, cache policy, or speed claims**: this is
one cache-disabled correctness/capability result, not a throughput benchmark.

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

## First execution gate — completed

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
retaining its text.

The completed canary recorded the following scalar result:

| Gate | Result |
| --- | --- |
| Quality lifetime startup | 493.27 s to ready |
| Synthetic quality | 8/8 exact items across two repetitions; 100% accuracy |
| Long-context lifetime startup | 483.00 s to ready |
| Varied-context retrieval | Passed; 62,336 prompt tokens and 24 completion tokens |
| Long-context latency | 54.23 s TTFT; 55.68 s end-to-end |
| Lifecycle / safety | 1,068.53 s total; clean two-lifetime audit; no startup safety gate |

The long-context request produced one measured completion event, so it does
not yield a defensible decode-TPS estimate. Its 0.43 aggregate output tokens/s
is end-to-end and dominated by the 54-second prefill/first-token interval; it
must not be compared with steady-state decode benchmarks. The lowest observed
available host memory was 18.36 GiB, with no safety gate recorded.

Prompts, completions, request identifiers, and credentials remain outside the
tracked record. A completed canary may publish only exporter-approved scalar
evidence after teardown and verification. The quality lifetime is a hard
admission gate: any failed synthetic item tears down that server and prevents
the long-context lifetime from starting. The read-only canary auditor requires
the two ordered fresh lifetimes, scalar runtime provenance immediately before
each ready event, no primer, and terminal cleanup before scalar evidence can
be exported.

## Cache-off B0 observability — completed

The next lane is deliberately smaller than a cache A/B: it does **not** enable
Radix, select a cache policy, reuse a prefix, compare wall time, or report TPS.
It asks one question first: on the exact cache-disabled candidate, do the
server's response-detail extension and native counters agree that one fresh,
non-streaming request had no cache hit?

The dedicated `qwen38-flash-next-sm121-triton-storage-cache-observability-canary`
suite uses the same immutable image, target snapshot, offline/read-only
container boundary, and one fresh lifetime. Its fixed order is:

1. Four clean `synthetic-exact-answer-v2` quality items, with thinking disabled.
2. One separately named synthetic request, also thinking-disabled and
   non-streaming, with `return_cached_tokens_details=true` and `n=1`.
3. A scalar-only cache observation immediately before that request's ordinary
   scalar completion record.

Before starting the server, the runner hashes the reviewed cache-selection,
OpenAI-extension, usage-accounting, and Prometheus source roles inside the
immutable local image without retaining source text. At startup it requires the
cache-off `ChunkCache` log record and `disable_radix_cache=true`. Around the
single observation request it waits for two identical Prometheus snapshots
before and after; the required input counter must advance while device, host,
storage, and fallback-total cache-hit counters remain unchanged. The six KV and
Mamba residency gauges must also be present, although their signed changes are
diagnostic rather than cache-hit evidence.

The result is `complete` only when those settled native counters agree with
explicit zero details or the reviewed omitted/null zero-detail form. A clean
but non-admitted observation is terminal `partial`, not an aborted execution;
it blocks cache-on work. Any malformed detail, nonzero hit, missing metric,
unsettled metrics, failed quality item, extra lifecycle event, or cleanup
failure fails closed. The runner always attempts to stop its owned server even
if log capture or watchdog cleanup itself fails.

The first B0 execution reached that clean terminal `partial` state because the
metric reader did not initially account for SGLang's stable scheduler label
vector. It was a measurement-parser diagnostic, not a cache-hit observation:
the response detail was omitted, usage detail was null, and no native counter
could yet be admitted. That scalar-only partial record is retained so the
correction is auditable.

After normalizing the shared scheduler labels and rerunning from a new server
lifetime, the B0 canary completed. All four quality items passed. The server
returned the reviewed omitted response detail and null usage detail; two
identical native metric snapshots were observed on each side of the direct
request. The input-prefill counter advanced by 64 tokens while every device,
host, storage, and fallback-total cache-hit counter changed by zero. The
runtime attestation selected `ChunkCache` with Radix disabled, and the scalar
record was admitted as a zero-hit observation.

This establishes only the cache-disabled baseline: the direct request did not
use a recorded cache tier under the reviewed runtime. It does **not** measure a
cache-on policy, cache benefit, wall time, throughput, or a reusable-prefix
effect.

Reproduce the B0 check only through its dedicated entry point:

```bash
python3 sparkbench.py sm121-cache-observability-canary \
  qwen38-flash-next-nvfp4-sm121-triton-storage-target-only-sglang
python3 sparkbench.py audit-sm121-cache-observability results/<b0-run>
```

The raw journal may contain only ignored local runtime artifacts. The tracked
evidence exporter retains ordinary scalar request counts plus a fixed
cache-attestation object; it excludes prompts, completions, reasoning, request
IDs, response bodies, raw metrics, source text, and timings presented as a
performance result. It also rejects a checksum-refreshed generic downgrade of
the B0 bundle.

## Paired cache-policy semantic canary — B partial, A correctly withheld

The next admission-only step is a paired B-then-A semantic probe, not a
benchmark. It uses two newly isolated profiles against the same pinned local
image, snapshot, C1 geometry, disabled CUDA graphs, and
`extra_buffer_lazy`/four-state setting. The only serving-argument difference
is `--disable-radix-cache` in B. Neither arm can be selected by generic
planning, matrix, or serving entry points.

Its order is deliberately four independent server lifetimes:

1. Cache-off B quality gate.
2. Cache-off B semantic probe.
3. Cache-on A quality gate, only after B is complete.
4. Cache-on A semantic probe, only after B's scalar lifecycle and private
   prompt-identity controls pass.

The quality lifetime cannot populate the semantic cache. In its own fresh
second lifetime, each arm makes a cold T0 request followed by two deterministic
append-only turns with a 32K–48K synthetic shared prefix. Prompt text,
completions, and token IDs never leave process memory. The runner compares the
three A token-ID sequences to B in memory, then persists only counts and
booleans.

B requires settled input-counter progress and zero device, host, and storage
hits on every turn. A requires the same zero result on T0; T1 and T2 must each
report explicit positive device-only details that reconcile exactly with native
device-hit and residency deltas. Both arms reject host/storage hits, eviction,
retraction, missing guardrail counters, unsettled snapshots, or a failed exact
answer. A non-admitted semantic result is terminal `partial`; it is never
silently promoted to a speed result.

The first frozen B/A execution took that authorized terminal-partial path. B
completed its isolated quality lifetime and all three semantic turns, then
teardown and lifecycle audit completed cleanly. Its cache/detail observations
and ordinary native counters were settled, but the runtime did not materialize
labeled samples for the two native eviction/retraction counters. The audit
therefore marked each turn non-admitted for unavailable guardrails and the
controller left A entirely untouched. This is an observability finding only:
it establishes neither cache behavior nor a cache-on comparison.

The pinned metrics collector explains the gap. Its labeled Prometheus counters
are materialized only when an increment path calls `labels(...).inc(...)`; on a
fresh no-event server, Prometheus still declares the counter type but emits no
labeled sample. The initial parser required a sample, so it correctly failed
closed under its then-current contract. The scalar B-only partial is retained
as evidence. A subsequent fresh pair may accept a declared-but-unmaterialized
zero only after the parser pins that exact source behavior and requires the
counter declaration itself; it may not infer zero from a missing metric alone.

Run and inspect the pair only through the dedicated commands:

```bash
python3 sparkbench.py sm121-cache-policy-semantic-canary
python3 sparkbench.py audit-sm121-cache-policy-semantic \
  results/<cache-off-b-run> results/<cache-on-a-run>
```

The semantic evidence lane emits no wall time, latency, TPS, energy, telemetry,
prompt, completion, token-ID, request-ID, or source-text field. A successful
pair establishes semantic cache behavior for this exact candidate only. A
separate matched cold A/B/B/A protocol remains necessary before making a cache
benefit or serving-performance claim.

Each frozen pair carries an opaque pair-instance SHA-256 commitment derived
from its two raw plan nonces; the raw nonces never leave the ignored plans. A
also records a scalar receipt for the completed B control before it can start.
Those commitments prevent static profile fingerprints from accidentally pairing
separate attempts, without exposing a path, prompt, token sequence, or timing
field. The exporter still accepts one B/A pair per results root, so a retry
uses a separate results root/archive rather than mixing campaigns.

## What remains blocked

The runtime-attestation schema intentionally accepts only an `admitted` record
when all of the following are true: retired-overlay rejection, storage import,
io_uring, PLE row comparison, SM121 Triton, quality, and long-context. The
target-only canary and its B0 cache-off observation now demonstrate those
gates. That narrow admission result does not remove the ordinary-entrypoint
tombstone or establish a general serving configuration.

Before any performance comparison or cache experiment, the next protocol must:

1. pin the immutable image, model, tokenizer, revision, and serving profile;
2. retain the fresh-process quality, varied-token, and B0 zero-hit checks as
   admission gates;
3. verify teardown and scalar-only evidence export; then
4. run a cold A/B/B/A cache-policy experiment in separate process lifetimes.

The retired overlay and all historical TRT-LLM measurements remain excluded
from this candidate. Passing this pre-admission does not license a speed claim
or an overnight cache campaign.

## Repository verification

The new seccomp contract and runtime-attestation validators are fail-closed and
scalar-only. The complete repository suite passed 993 tests, module
compilation and lint passed, and the committed scalar archive was regenerated
and verified from the completed canary. Raw result bundles, logs, prompts,
completions, and credentials remain ignored.
