# SM121 1K/2K chunked-prefill protocol — 2026-08-29

## Status

The frozen contract, dedicated fresh-lifetime controller, fail-closed runtime
chunk-size attestation, read-only audit, and scalar-only evidence path are
implemented. No chunked-prefill campaign has been frozen or run. The profiles
remain blocked from generic execution; the dedicated
`sm121-chunked-prefill-performance` command is the only path that can admit a
live A/B/B/A campaign.

## Question

On the current, cache-on SM121 native-NVMe Qwen3.8 Flash-Next stack, does
raising `--chunked-prefill-size` from 1,024 to 2,048 reduce correct cold 60K
request wall time enough to justify the setting for a single user's long
context?

This is not a decode-TPS study. It does not test concurrency or claim an
agentic coding benchmark. The present admitted baseline is chat-only with
thinking disabled and no configured tool parser. A genuine agentic-coding
comparison needs a separately admitted current-runtime tool/parser and
reasoning-policy lane before it can reuse this axis.

## Frozen siblings

| Arm | Profile | Chunked prefill | Held constant |
| --- | --- | ---: | --- |
| A | `...chunked-prefill-performance-1k-sglang` | 1,024 | Current local SM121 image, NVFP4 target, C1, UnifiedRadixCache, `extra_buffer_lazy`, four Mamba slots, 64K pool, no-thinking, graphs disabled |
| B | `...chunked-prefill-performance-2k-sglang` | 2,048 | Identical to A |

Both profiles pin the current local image and the same RadixArk target
revision. They have no source overlays, mapped PLE, speculative decoding,
tool parser, or hidden profile-specific request policy. The profile contract
normalizes the served name and verifies that the only command-line difference
is the chunk-size value.

## Planned measurement

The dedicated controller will use a controller-private deterministic 60K
static-history generator. It issues a cold `T0` request, then two fixed
append-only history turns (`T1`, `T2`) without feeding model output back into
the next prompt. Prompt text, completions, token IDs, request identifiers, and
rendered request bodies stay in memory or ignored raw provenance.

- A/B/B/A order, with two independent fresh lifetimes per arm.
- Each arm receives one isolated four-item exact-answer quality lifetime and
  one timed lifetime: eight fresh server lifetimes total.
- Every timed request must be correct, no-thinking, non-streaming, C1, and
  must preserve its private cross-lifetime prompt-token identity.
- `T0` is the sole prefill primary. `T1`/`T2` are a static-history proxy for
  subsequent long-context turns, not an agent-tool trajectory.
- Current cache residency/guardrail observations remain mandatory: cold `T0`,
  device-only append hits, no eviction, no retraction, and no pressure breach.

The first version will use the existing non-streaming, exact-response adapter,
so it measures request wall only. TTFT requires a separately admitted
privacy-safe streaming adapter and is not inferred from request wall.

## Decision rule

The controller will freeze arithmetic means before execution:

- promote B only when correct cold-`T0` mean wall is at most `0.95 × A`;
- reject B on an append-turn or full-`T0`–`T2` wall guardrail above
  `1.05 × A`;
- otherwise retain A or report the speed result inconclusive, according to the
  frozen reducer;
- keep decode throughput, cache counters, memory, swap, and startup separate
  diagnostics rather than pooling them into the decision.

Each lifetime keeps the existing 1,200-second bound, 10 GiB MemAvailable
floor, 512 MiB starting-swap ceiling, 512 MiB swap-growth ceiling, loopback
endpoint, no-download rule, ownership cleanup, and benchmark lock.

## Before execution

The controller requires an exact `chunked_prefill_size` server-info attestation
for both arms and validates the current UnifiedRadix/lazy runtime identity.
Run `audit-sm121-chunked-prefill-performance` before exporting the completed
scalar result with `export-evidence`, then verify it. A generic `plan`, `run`,
or matrix invocation remains rejected by execution admission.
