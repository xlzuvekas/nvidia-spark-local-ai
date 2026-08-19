# Qwen3.6 native llama.cpp prefix-cache results — 2026-08-18

## Result

The dedicated single-slot Qwen3.6 prefix-cache controls completed **30/30**
validated requests with prompt caching disabled and **30/30** with it enabled.
In the enabled schedule, the deliberate warm-prefix request reported **98.4285%**
cached prompt tokens at the 8,192-repetition target and **99.6061%** at the
32,768-repetition target. Its observed median first-token time was respectively
**13.687 s** and **71.893 s** below the immediately preceding forced-cold
request in the same five-block run; the corresponding end-to-end differences
were **13.694 s** and **71.881 s**.

This is a serial same-slot prompt-KV reuse result, not a fresh-prefill
benchmark, a general serving-throughput score, or a causal
difference-in-differences estimate. The cache-off order controls remained near
zero, but an off/on comparison still has separate process and run order. The
result shows the observed cold-to-warm behavior of this exact stack only.

## Cache controls and paired observations

Each target contains five three-request blocks. In cache-off mode all requests
are forced cold. In cache-on mode the first two are forced cold and the third
is the warm-prefix request. The values below are medians of the five within-run
`forced-cold-b` minus third-request pairs; positive values mean the third
request was faster. The same deterministic long prefix is shared within a
block, while the short suffix remains distinct.

| Shared-prefix target | Cache-off B minus C: TTFT / E2E / server prompt | Cache-on B minus warm: TTFT / E2E / server prompt | Warm cached tokens | Server decode rate, cold B → third |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 repetitions | +0.002 / -0.003 / -0.008 s | **+13.687 / +13.694 / +13.687 s** | 161,657 / 164,238 (**98.4285%**) | 52.547 → 52.571 tok/s |
| 32,768 repetitions | +0.064 / +0.053 / +0.069 s | **+71.893 / +71.881 / +71.894 s** | 653,185 / 655,768 (**99.6061%**) | 34.956 → 34.789 tok/s |

The cache-off B-to-C controls are 2–69 ms in magnitude, whereas the
cache-on B-to-warm observations are 13.7 s and 71.9 s. That supports the
intended cache-control distinction within each run; it does not identify a
separate causal estimate across independently started server processes.

Each five-request condition emitted 640 output tokens. Server decode rate was
effectively unchanged within the precision and sample size here: 52.547 to
52.571 tok/s at the smaller target and 34.956 to 34.789 tok/s at the larger
target. The warm condition reports much less physical prompt work and lower
first-output/wall-time measurements; these measurements do not show that
prefix caching increases decoder TPS.

## Request-scoped prompt work and wall time

The table retains condition totals so cache-assisted logical input rates cannot
be confused with fresh-prefill rates. Prompt-processing seconds are server
reported totals across five requests; condition wall includes those five
serial requests.

| Target | Condition | Logical prompt tokens | Cached prompt tokens | Uncached prompt tokens | Condition wall | Server prompt time | Physical prompt rate | Cache-assisted logical input rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | cold B | 164,237 | 0 | 164,237 | 82.620 s | 70.200 s | 2,339.557 tok/s | 2,339.557 tok/s |
| 8,192 | warm prefix hit | 164,238 | 161,657 | 2,581 | 14.102 s | 1.738 s | 1,484.714 tok/s | 94,477.508 tok/s |
| 32,768 | cold B | 655,765 | 0 | 655,765 | 381.539 s | 362.488 s | 1,809.068 tok/s | 1,809.068 tok/s |
| 32,768 | warm prefix hit | 655,768 | 653,185 | 2,583 | 21.712 s | 2.554 s | 1,011.173 tok/s | 256,715.123 tok/s |

The final column divides all logical input tokens by server prompt-processing
time after most prefix tokens have been reused. It is a cache-assisted logical
rate, explicitly **not** a fresh prompt-evaluation rate. The physical prompt
rate instead divides only uncached prompt tokens by that same server time.

## Frozen scope and measurement boundary

Both profiles used the same Unsloth Qwen3.6 35B-A3B UD-Q4_K_XL artifact at
revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`, one llama.cpp slot, a
262,144-token context, Q8_0 K/V cache types, temperature zero, a 128-token
output cap, and reasoning disabled. The native server pin was llama.cpp
revision `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`. The only profile-level
mode difference was `--no-cache-prompt` versus `--cache-prompt`; the cache-on
schedule explicitly forces its first two requests cold.

The title and filename use the benchmark host's MST date, 2026-08-18. The
sanitized evidence manifests use their UTC capture date, 2026-08-19, for these
same two runs.

Logical prompt, cached prompt, uncached prompt, completion, and timing counts
are reconciled from final request-scoped llama.cpp SSE fields. Server-global
Prometheus deltas are retained only as non-negative diagnostics and are not
used in this report's request rates, totals, or paired timing claims.

The suite tests two synthetic shared-prefix sizes, five paired blocks per size,
and one model/runtime/quantization/slot geometry. It does not measure cache
capacity under concurrent users, cache eviction, realistic document behavior,
general latency distributions, p95, semantic quality, or other models. Prompt
text, outputs, reasoning, identifiers, raw metric payloads, and local run
locations are intentionally excluded.

## Publication boundary

This note uses only fixed condition labels and scalar summaries from completed
controls. The normal evidence exporter and verifier gate publishes any tracked
evidence separately; raw measurement artifacts remain untracked.
