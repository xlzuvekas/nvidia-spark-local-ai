# SM121 1K/2K chunked-prefill protocol — 2026-08-29

## Status

The frozen contract, dedicated fresh-lifetime controller, fail-closed runtime
chunk-size attestation, read-only audit, and scalar-only evidence path are
implemented. The first frozen campaign reached the quality gate and a 60K
cache-cold `T0`, then correctly terminalized as partial: an over-strict
controller check treated SGLang's 64-token ready-state bootstrap increment of
the global input counter as cache reuse even though every cache-hit and
residency counter was zero. That non-admitted partial is retained as scalar
evidence; it does not inform the A/B decision.

The corrected newly frozen campaign completed all four A/B/B/A arms, passed
all four quality lifetimes, and its read-only audit reported zero errors. It
retains 2,048-token chunked prefill for this specific 60K static-history
proxy. The [completed scalar bundle](../evidence/campaigns/qwen38-flash-next-sm121-chunked-prefill-performance-v1-d87fb7c61722/manifest.json)
and the [initial non-decisive partial bundle](../evidence/campaigns/qwen38-flash-next-sm121-chunked-prefill-performance-v1-8232b4449e14/manifest.json)
contain no request content. Profiles remain blocked from generic execution;
the dedicated `sm121-chunked-prefill-performance` command is the only path that
can admit another v1 live A/B/B/A campaign. The separately named v2 2K/4K
follow-up is frozen but not yet measured; it uses its own command, campaign
identity, profiles, and evidence bundle namespace.

## Question

On the current, cache-on SM121 native-NVMe Qwen3.8 Flash-Next stack, does
raising `--chunked-prefill-size` from 1,024 to 2,048 reduce correct cache-cold
60K request wall time enough to justify the setting for a single user's long
context?

This is not a decode-TPS study. It does not test concurrency or claim an
agentic coding benchmark. The present admitted baseline is chat-only with
thinking disabled and no configured tool parser. A genuine agentic-coding
comparison needs a separately admitted current-runtime tool/parser and
reasoning-policy lane before it can reuse this axis.

## Completed result

The frozen reducer uses the unweighted mean of two fresh lifetimes per arm.
All timed requests were non-streaming and quality-admitted; the table reports
request wall only.

| Metric | 1K A mean (s) | 2K B mean (s) | B/A | Frozen rule |
| --- | ---: | ---: | ---: | --- |
| Cache-cold `T0` 60K request | 50.022 | 38.635 | 0.772 | retain B at `<= 0.95` |
| Append proxy `T1 + T2` | 3.168 | 3.199 | 1.010 | guardrail `<= 1.05` |
| Full `T0`–`T2` | 53.190 | 41.834 | 0.787 | guardrail `<= 1.05` |

The 2K setting clears the cache-cold primary by 22.8%, stays within the
append guardrail, and lowers the complete three-turn wall by 21.3%; the frozen
decision is therefore `retain_b`. This result promotes only the current
SM121/cache-on/C1/no-thinking static-history prefill setting. It makes no
claim about TTFT, decode TPS, concurrency, tool calling, or agentic coding.

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

## Registered v2 follow-up — not yet measured

The retained 2K setting is the control for a separate 2K-versus-4K panel. It
is deliberately not an extension of the completed v1 campaign: its frozen
campaign ID is `qwen38-flash-next-sm121-chunked-prefill-performance-v2`, its
timed case has a v2 identity, and its evidence projection rejects v1/v2
profile, attestation, turn, or run-binding mixtures.

| Arm | Profile | Chunked prefill | Status |
| --- | --- | ---: | --- |
| A | `...chunked-prefill-performance-2k-v2-sglang` | 2,048 | Retained v1 setting, new v2 control |
| B | `...chunked-prefill-performance-4k-v2-sglang` | 4,096 | Candidate; no live result yet |

It keeps the same current cache-on/C1/no-thinking 60K static-history request
wall protocol, ABBA order, fresh lifetimes, quality gate, cache-cold and
append checks, 1,200-second lifetime bound, and reducer thresholds. The only
serving delta within v2 is `--chunked-prefill-size`. Use
`sm121-chunked-prefill-performance-v2` to freeze and execute it. Until its
audit and scalar evidence are complete, it makes no performance claim.

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
- Cache-cold `T0` requires zero device/host/storage hits and zero cached-token
  residency before and after the measured request. SGLang's ready-state
  bootstrap may leave a nonzero global input-counter baseline; the controller
  requires a positive request-local input delta but never mistakes that
  baseline for a cache hit.
- Current cache residency/guardrail observations remain mandatory: cache-cold
  `T0`, device-only append hits, no eviction, no retraction, and no pressure
  breach.

The first version will use the existing non-streaming, exact-response adapter,
so it measures request wall only. TTFT requires a separately admitted
privacy-safe streaming adapter and is not inferred from request wall.

## Decision rule

The controller will freeze arithmetic means before execution:

- promote B only when correct cache-cold-`T0` mean wall is at most `0.95 × A`;
- reject B on an append-turn or full-`T0`–`T2` wall guardrail above
  `1.05 × A`;
- otherwise retain A or report the speed result inconclusive, according to the
  frozen reducer;
- keep decode throughput, cache counters, memory, swap, and startup separate
  diagnostics rather than pooling them into the decision.

Each lifetime keeps the existing 1,200-second bound, 10 GiB MemAvailable
floor, 512 MiB starting-swap ceiling, 512 MiB swap-growth ceiling, loopback
endpoint, no-download rule, ownership cleanup, and benchmark lock.

## Admission for a future campaign

Each controller requires an exact `chunked_prefill_size` server-info
attestation for both arms and validates the current UnifiedRadix/lazy runtime
identity. Run `audit-sm121-chunked-prefill-performance` before exporting a
completed scalar result with `export-evidence`, then verify it. A generic
`plan`, `run`, or matrix invocation remains rejected by execution admission.
