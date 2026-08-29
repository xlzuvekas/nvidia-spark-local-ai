# SM121 chunked-prefill studies — 2026-08-29

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
follow-up also completed, passed all four quality lifetimes, and its
read-only audit reported zero errors. Its separate campaign identity,
profiles, and evidence namespace keep it distinct from the completed 1K/2K
evidence.

The next 8,192-token candidate is only a prospective admission target. Its
separate two-lifetime quality-plus-cold-`T0` gate is implemented and
fail-closed, but has not produced an admission or timing record. It remains
outside evidence and cannot select 8K or enable a v3 A/B/B/A campaign.

## Question

The completed v1 and v2 panels retained 2,048 then 4,096 tokens for the exact
cache-on/C1/no-thinking 60K static-history proxy. The open question is whether
the exact 8,192-token profile can first pass a separate safety admission; only
then could a newly frozen 4K/8K A/B/B/A request-wall comparison be considered.

This is not a decode-TPS study. It does not test concurrency or claim an
agentic coding benchmark. The present admitted baseline is chat-only with
thinking disabled and no configured tool parser. A genuine agentic-coding
comparison needs a separately admitted current-runtime tool/parser and
reasoning-policy lane before it can reuse this axis.

## Completed v1 result (1K/2K)

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

## Frozen v1 siblings

| Arm | Profile | Chunked prefill | Held constant |
| --- | --- | ---: | --- |
| A | `...chunked-prefill-performance-1k-sglang` | 1,024 | Current local SM121 image, NVFP4 target, C1, UnifiedRadixCache, `extra_buffer_lazy`, four Mamba slots, 64K pool, no-thinking, graphs disabled |
| B | `...chunked-prefill-performance-2k-sglang` | 2,048 | Identical to A |

Both profiles pin the current local image and the same RadixArk target
revision. They have no source overlays, mapped PLE, speculative decoding,
tool parser, or hidden profile-specific request policy. The profile contract
normalizes the served name and verifies that the only command-line difference
is the chunk-size value.

## Completed v2 follow-up (2K/4K)

The retained 2K setting is the control for a separate 2K-versus-4K panel. It
is deliberately not an extension of the completed v1 campaign: its frozen
campaign ID is `qwen38-flash-next-sm121-chunked-prefill-performance-v2`, its
timed case has a v2 identity, and its evidence projection rejects v1/v2
profile, attestation, turn, or run-binding mixtures.

| Arm | Profile | Chunked prefill | Status |
| --- | --- | ---: | --- |
| A | `...chunked-prefill-performance-2k-v2-sglang` | 2,048 | Retained v1 setting, new v2 control |
| B | `...chunked-prefill-performance-4k-v2-sglang` | 4,096 | Retained by the audited v2 panel |

It keeps the same current cache-on/C1/no-thinking 60K static-history request
wall protocol, ABBA order, fresh lifetimes, quality gate, cache-cold and
append checks, 1,200-second lifetime bound, and reducer thresholds. The only
serving delta within v2 is `--chunked-prefill-size`. The completed A/B/B/A
panel had two fresh timed lifetimes per arm, eight static and eight runtime
attestations, and zero audit errors. Its [completed scalar bundle](../evidence/campaigns/qwen38-flash-next-sm121-chunked-prefill-performance-v2-d569ae86eb6d/manifest.json)
contains no request content.

| Metric | 2K A mean (s) | 4K B mean (s) | B/A | Frozen rule |
| --- | ---: | ---: | ---: | --- |
| Cache-cold `T0` 60K request | 38.634 | 30.026 | 0.777 | retain B at `<= 0.95` |
| Append proxy `T1 + T2` | 3.245 | 3.263 | 1.006 | guardrail `<= 1.05` |
| Full `T0`–`T2` | 41.879 | 33.290 | 0.795 | guardrail `<= 1.05` |

The 4K setting lowers cache-cold request wall by 22.3%, remains within the
append guardrail, and lowers full three-turn wall by 20.5%; the frozen
decision is `retain_b`. Together with v1, this promotes 4,096-token chunked
prefill only for the current SM121/cache-on/C1/no-thinking 60K static-history
proxy. It does not establish TTFT, decode TPS, concurrency, tool-calling, or
agentic-coding performance.

## Completed v1/v2 measurement protocol

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

Both versions use the existing non-streaming, exact-response adapter, so they
measure request wall only. TTFT requires a separately admitted
privacy-safe streaming adapter and is not inferred from request wall.

## Completed v1/v2 decision rule

Both completed panels froze arithmetic means before execution:

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

## Prospective v3 8K admission — no performance result

The 4,096-token setting remains the retained setting for this exact
cache-on/C1/no-thinking 60K static-history proxy. The 8,192-token candidate
has not been compared, retained, or promoted.

Before a future 4K/8K A/B/B/A campaign can be frozen, the exact 8K profile
must complete a standalone, non-resumable admission check: one fresh four-item
exact-answer quality lifetime and one separate fresh cache-cold 60K `T0`
lifetime. Each lifetime rechecks host preflight, retains the image/source and
runtime chunk-size attestations, uses the existing timeout and cleanup rules,
and is audited for strict lifecycle order and removal of ephemeral API-key
files.

The `T0` gate requires a correct no-thinking, non-streaming response, settled
metrics, zero device/host/storage cache reuse, no eviction or retraction, and
an exact private prompt-token identity check. It records no request-wall time,
TPS, ratio, or performance decision and is not benchmark evidence. A later
performance controller remains hard-blocked until a verified admission receipt
is designed and bound; passing the gate would not select 8K, replace 4K, or
establish TTFT, concurrency, tool, agentic-coding, or general-serving
performance.
