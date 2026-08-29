# Qwen3.8-Flash-Next SM121 cache-policy timing protocol — 2026-08-29

Status: frozen and audited offline; no timing campaign has been executed yet.

The replacement SM121 Triton/storage candidate now has scalar evidence for its
target admission, cache-off B0 behavior, and a complete cache-off/cache-on
semantic pair. That establishes the narrow fact needed to measure cache policy;
it does not establish a cache speedup. This document records the separate,
non-resumable performance lane that may make that comparison.

## Exact comparison

The lane uses only these two sibling profiles against the same pinned local
image, weights, serving geometry, and 64K served context:

- A: `UnifiedRadixCache`, `extra_buffer_lazy`, four Mamba cache states.
- B: the same arguments plus `--disable-radix-cache`.

They differ only in cache policy. CUDA graphs remain disabled, concurrency is
one, native read-only NVMe PLE remains enabled, and the request disables
thinking. Generic plan, matrix, and serving entry points reject both profiles;
only the dedicated controller can start them.

The controller freezes four opaque-nonce plans in this order: A, B, B, A. Each
arm receives two newly started and torn-down servers: an isolated four-item
exact-answer quality lifetime followed by a separate timed lifetime. In each
timed lifetime, a cold 32K–48K T0 request is followed by append-only T1 and T2
requests. The process retains prompt token IDs only long enough to verify the
fixed cross-lifetime identities; it never writes them to the journal.

## Measurement and decision rule

The only timing field is client-measured wall time around each non-streaming
request. It excludes metrics settling and does not claim TTFT, decode TPS,
throughput, energy, or an end-to-end agent speedup. Every quality or timed
lifetime has a 1,200-second admission deadline. An observed expiry is rejected
and immediately interrupts the owned server before diagnostic cleanup; cleanup
itself may finish after that deadline and is not part of a timing claim.

For each policy, the reducer averages the two replicas' `T1 + T2` wall time and
the full `T0 + T1 + T2` wall time. A policy is eligible only if its later-turn
mean is at least 5% lower than the other policy's mean, using exact unrounded
comparisons. It is retained only when its full-sequence mean is no more than 5%
worse than the control. Otherwise the result is `no_retention`; any failed
admission, timeout, identity mismatch, quality failure, or malformed cache
observation makes the campaign terminal `partial` and `not_evaluated`.

## Evidence boundary

Before planning and again before execution, the controller verifies the four
exact prerequisite scalar bundles in the tracked evidence index. The campaign
binds the prerequisite hashes, four frozen plan fingerprints, and an opaque
campaign-instance digest. Export and read-only audit recheck that binding,
fresh-lifetime topology, quality completion, cache/static/runtime attestations,
turn-to-journal reconciliation, and the scalar reducer.

The published bundle may contain only status, decision, scalar wall times,
typed cache counters, boolean admissions, fixed IDs, source/image hashes, and
the four small quality attestations. It excludes prompts, completions,
reasoning, token IDs, request IDs, tool payloads, raw metrics, logs, paths, and
wall-clock timestamps. A partially completed lifetime can retain only a
validated prefix of T0/T1/T2 that exactly matches its raw journal; it cannot
become a payload side channel.

## Authorized commands

`--results` is the parent results directory; the command creates its ignored
`cache-policy-campaigns/` child. There is intentionally no resume command: a
started campaign is terminal and a retry requires a newly frozen A/B/B/A set.

```bash
python3 sparkbench.py sm121-cache-policy-performance \
  --results results --evidence evidence

python3 sparkbench.py audit-sm121-cache-policy-performance \
  results/cache-policy-campaigns/<campaign> --evidence evidence
```

After a terminal campaign, export only through the normal scalar exporter, run
it twice to confirm the second output is unchanged, verify the archive, then
stage and verify the exact evidence index. Until those steps complete, this is
a protocol rather than a performance result.
