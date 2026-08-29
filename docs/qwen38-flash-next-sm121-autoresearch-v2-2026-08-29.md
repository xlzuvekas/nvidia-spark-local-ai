# Qwen3.8-Flash-Next SM121 autoresearch v2 — 2026-08-29

Status: implementation, offline contract tests, and the first fresh round are
complete. Its audited child retained cache-on A, so v2 rejected cache-off B.

## Why there is a v2 controller

The original single-user autoresearch campaign is preserved as historical
provenance. Its fixed cutoff elapsed before any model request, and the
superseded TRT-LLM overlay is execution-blocked. Reopening it, shortening its
time budget, or copying its profiles would invalidate the original frozen
protocol.

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch/tree/228791fb499afffb54b46200aca536f79142f117)
separates a fixed evaluator, one controlled change surface, and a persistent
research instruction. Its fixed five-minute training window and mutable
`train.py` are not appropriate for a Spark serving lifetime: cold start alone
can take minutes, unified-memory pressure contaminates subsequent work, and
correctness cannot be reduced to one unconstrained scalar.

The new v2 path keeps the useful pattern while constraining it for serving:

1. propose exactly one reviewed control/candidate axis;
2. freeze its manifest, explicit offset-aware cutoff, four nonce-bound plans,
   and child binding before inference;
3. run only one non-resumable, loopback/offline, one-configuration-at-a-time
   child campaign;
4. score only the child's audited scalar reducer; and
5. retain, reject, or call the candidate inconclusive without modifying a
   champion in place.

It is intentionally a registry, not an agent permission to invent flags. A
new hypothesis needs its own runner, profiles, suite, static/runtime
attestations, scalar projection, tests, and documentation before it can enter
the registry.

## First registered proposal

The first v2 proposal turns the strongest current product finding into an
independent, candidate-centric confirmation:

| Role | Arm | Profile | Difference |
| --- | --- | --- | --- |
| Control | A | `UnifiedRadixCache` with `extra_buffer_lazy` | Current retained cache-on policy |
| Candidate | B | `ChunkCache` with `--disable-radix-cache` | Cache-policy bundle only |

The v2 candidate is deliberately the cache-off bundle. The prior audited
campaign retained cache-on A by a large margin, so the correct first v2 action
is a rigorously bounded rejection/confirmation rather than pretending a known
bad control is a new optimization. It still measures a fresh independent
A/B/B/A campaign; no prior timing row enters its result.

Each arm owns a fresh exact-answer quality lifetime and a separate fresh timed
T0/T1/T2 lifetime. Both lifetime types have a 1,200-second admission deadline.
The v2 planner requires 10,200 seconds before the explicit cutoff: all eight
child lifetime budgets plus a ten-minute terminal-audit reserve. It does not
pad work to that duration.

The registered child is the existing SM121 cache-policy controller, so its
strict prompt-identity, quality, cache-state, safety, ownership, and scalar
evidence rules remain authoritative. The v2 wrapper binds the child campaign
basename, integrity digest, pair-binding digest, and prerequisite-bundle
digests. It writes no prompts, completions, reasoning, token IDs, request IDs,
tool payloads, logs, or local paths.

## Frozen decision mapping

The child reducer is the only component that performs wall-time threshold
arithmetic. It already uses exact decimal comparisons for the 5% later-turn
improvement and 5% full-sequence guardrail. V2 maps only its audited outcome:

| Child outcome | V2 result |
| --- | --- |
| `retain_b` | `retain` candidate B |
| `retain_a` | `reject` candidate B |
| `no_retention`, `guardrail_reject`, or `not_evaluated` | `inconclusive` |
| failed child audit or other malformed terminal state | `partial` / `inconclusive` |

No rounded public float is re-reduced to decide a winner. The result remains a
request-wall result for this synthetic shared-prefix lane; it is not TTFT,
decode TPS, aggregate throughput, energy, or end-to-end agent productivity.

## First completed round

The first fresh v2 child completed all four A/B/B/A arms and passed its
read-only cache-policy audit. Its child reducer returned `retain_a`, which the
wrapper correctly mapped to `complete` / `reject` for candidate B. The two
replicas measured mean later-turn request wall time of 2.801 seconds for A
versus 45.188 seconds for B (A/B = 0.0620), and mean full three-turn request
wall time of 36.979 seconds for A versus 79.115 seconds for B (A/B = 0.4674).

This is a fresh independent confirmation of the earlier cache-policy result;
the two campaigns are not pooled or re-reduced into a new measurement. The
published scalar child bundle is
[`e578d510b0fc`](../evidence/campaigns/qwen38-flash-next-sm121-cache-policy-performance-v1-e578d510b0fc/manifest.json).

## Commands

Previewing is read-only:

```bash
python3 sparkbench.py autoresearch-v2-plan --dry-run
```

Freezing requires an explicit future cutoff with enough room for the entire
non-resumable round. The following is illustrative; choose the actual frozen
cutoff at launch:

```bash
python3 sparkbench.py autoresearch-v2-plan \
  --cutoff '2026-08-29T09:00:00-07:00'
python3 sparkbench.py autoresearch-v2-run \
  results/autoresearch-v2/FROZEN_ROUND
python3 sparkbench.py autoresearch-v2-summarize \
  results/autoresearch-v2/FROZEN_ROUND
```

There is no resume command. If a started round is interrupted or fails, it is
terminal provenance and the next attempt must freeze new plans. The child
campaign stays beneath `results/cache-policy-campaigns/`, so the existing
read-only audit and scalar-only evidence exporter remain the sole publication
path for its actual measurements. The wrapper is ignored controller
provenance, not a second measurement archive. The exporter validates only its
empty pre-plan, frozen-unstarted, or terminal shapes and emits evidence only
for its independently audited child.

## Boundaries and next proposal

This does not complete the historical O2→O3→L2→L3 lazy-buffer interaction.
That frozen block remains non-estimable: O2's C6 tail and O3 startup crossed
the swap safety gate, and fresh L2/L3 never started. The old lazy panel cannot
be substituted into its missing cells.

With the cache-off candidate rejected twice in independent campaigns, the next
product question should be a newly admitted one-axis long-context prefill
candidate such as 1,024 versus 2,048 token chunked prefill. It is not
registered yet: current SM121 source/runtime attestation, exact profiles, a
quality/workload contract, and a dedicated scalar reducer are still required
before a GPU lifetime is authorized.
