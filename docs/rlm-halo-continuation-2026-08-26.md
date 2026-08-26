# RLM and HALO continuation — 2026-08-26

## Verified starting point

The clean-revision campaign with plan fingerprint `51cfeed7fa75` is
operationally complete. Independent reconstruction verified its plan integrity,
all 78 journal records, all 2,149 telemetry records, monotonic timestamps,
case dimensions, terminal topology, group arithmetic, and cleanup. Its 26
planned cases reconcile to 25 completed cases plus one predeclared held RLM
depth-2 case. The one failed HALO attempt stopped and restarted the server and
completed on attempt two. No exact-plan worker, serving container, network, or
GPU process remained.

The recorded `partial` status is therefore intentional-hold partiality, not a
deadline or cleanup failure. The current summary schema does not distinguish
those meanings in its top-level status; `held_cases = 1`, zero deadline skips,
and zero exhausted cases are required context. Journals and telemetry are
internally consistent scalar streams but are not cryptographically hash-chained.

The first campaign supplied two strong experimental signals:

- normal-threshold RLM completed all four 8K–128K cases, but only the 128K case
  was correct and every `compaction_count` remained zero; and
- HALO depth 0 returned valid JSON in 9/9 cases with mean family F1 0.578,
  while depth 1 returned valid JSON in only 1/9 with mean F1 0.044. Seven of
  the eight invalid depth-1 completions made exactly ten root tool calls,
  created no subagent, and never finalized, implicating the ten-turn boundary
  rather than parallel child execution.

## Completed continuation result

The clean-revision continuation with plan fingerprint `8b52bc5c6bc5` ran from
23:55:25 MST on August 25 through 05:48:47 MST on August 26. Final cleanup was
verified at 05:48:48, before both independent stop safeguards. All 106 planned
cases have exactly one terminal outcome:

| Phase | Completed | Exhausted | Deadline-skipped | Held |
| --- | ---: | ---: | ---: | ---: |
| RLM | 38 | 2 | 16 | 1 |
| HALO | 2 | 7 | 40 | 0 |
| **Total** | **40** | **9** | **56** | **1** |

The summary status is `partial` because only completed cases are scored; it is
not a controller, accounting, or cleanup failure. All 60 started attempts have
one outcome, all 20 server lifecycles have matched start/ready/stop events, and
no exact-plan worker, serving container, network, or GPU process remained.

### RLM findings

The completed BABILong-derived panel scored 2/12 direct cases, 2/24 normal
depth-1 cases, and 0/2 forced-compaction cases as correct. Treatment aggregates
mix different task and context selections, so they are descriptive rather than
a causal quality comparison. In the 12 fully attempted direct/depth-1 pairs at
8K and 32K, direct produced two correct answers; depth 1 produced one correct
answer, ten incorrect answers, and one deterministic token-limit exhaustion.
Among the 11 pairs where both treatments completed, each produced one correct
answer and depth 1 had 11.64 times the median direct wall time.

None of the 24 completed normal depth-1 or two forced-compaction cases recorded
a recursive subcall. The scored treatments therefore measured iterative RLM
scaffolds with recursion permitted, not an exercised recursion-quality effect.
The completed normal arm reused 82.553% of prompt tokens and reported 12.260
effective generation tokens per end-to-end second, compared with 0.125% and
3.814 for the much shorter direct completions. There is no cache-disabled arm,
and the two treatments generate very different amounts of work, so these
values do not establish a caching speedup or make depth 1 faster per task.

Forced 0.20-threshold compaction actually ran in only one of its two cases. In
that matched trajectory it compacted twice, reduced cumulative prompt traffic
45.65%, increased accepted requests from 9 to 15, increased wall time 11.93%,
and changed the result from correct to incorrect. The other forced case never
crossed its threshold and matched the normal case's calls, generation, and
incorrect outcome. One effective trajectory proves the mechanism can run but
cannot estimate its latency or quality effect. Normal 0.85-threshold compaction
ran once among the 24 completed normal cases.

The two exhausted RLM cases repeated the same explicit aggregate-token failure
on both attempts: 265,760 versus a 262,144 cap in one cell and 278,877 versus
262,144 in the other. Their identical retries and intervening cold starts used
about 20 minutes without adding information. Explicit token-limit failures
should be non-retryable in the next controller revision.

### HALO findings

HALO attempted nine of 49 planned cells before the frozen cutoff. Two of four
depth-0 cells completed; zero of five attempted depth-1 cells completed; the
depth-2 sentinel was not attempted. Across 17 attempts there were two
completions, eleven HTTP 400 failures, and four timeouts. The persisted scalar
error chain identifies `EngineAgentExhaustedError`, `HTTPStatusError`, and HTTP
400, but does not retain a structured parameter or failed-attempt token delta.
Context growth is therefore a leading hypothesis, not a demonstrated cause.

The two scored depth-0 completions used 2,048 and 65,536 synthetic traces. In
aggregate they accepted 39 model requests, processed 554,149 prompt tokens,
reused 482,944 of them (87.151%), generated 6,608 tokens, and reported 8.647
effective generation tokens per end-to-end second. Both finalized as valid
JSON with family F1 0.667 and citation precision 1.0; their mean count accuracy
was 0.403. These are two cutoff-selected survivors, not a scale comparison.
The 65,536-trace completion used fewer cumulative prompt tokens and less wall
time than the 2,048-trace completion, while the identical 2,048-trace case
failed once and then completed, showing that failure was path-dependent rather
than monotonically determined by corpus size.

Seven cell dimensions overlap the preceding 10-turn campaign. All seven
eventually completed with `max_turns = 10`, and four returned valid JSON. With
the same pinned model, image, and upstream revisions but `max_turns = 20`, one
of seven completed and returned valid JSON while six exhausted retries. The
runs were sequential, inference is stochastic, and their repository and
campaign revisions differ, so this is a strong cross-run regression associated
with the longer trajectory budget rather than a causal estimate. Doubling the
turn budget did not repair depth-1 finalization in this continuation.

### Schedule, resources, and interpretation

The campaign ran for 5 hours 53 minutes 22 seconds. Model attempts occupied
4 hours 42 minutes 5 seconds, or 79.8% of wall time. Twenty server launches
occupied 1 hour 9 minutes 53 seconds, or 19.8%; HALO's 14 cold restarts alone
used 55 minutes 40 seconds. The failure path therefore invalidated the plan's
optimistic fit estimate and left 56 cases explicitly deadline-skipped. A
roughly 75-second admission guard flushed each phase before its nominal cutoff
and left the cleanup reserve intact.

The 10,359-row scalar telemetry stream covers the full campaign. Trapezoidal
integration records approximately 204.28 Wh of GPU-board energy; that excludes
the rest of the system and facility. Observed power ranged from 5.11 to 100.07
W, temperature peaked at 87 C, and no out-of-memory or cleanup failure was
recorded.

This continuation is an external-context experiment: BABILong documents stay
in the RLM environment and HALO searches an indexed trace corpus. It is not a
native model-context test beyond 128K. HALO held `reasoning_effort = "none"`;
the pinned RLM recipe does not support a comparable runtime effort control.
No reasoning-effort, cache-off, agent-concurrency, native-context, or exercised
recursion effect can be inferred from this run.

The next discriminating sequence is:

1. retain scalar-safe failed-attempt counters and structured error classes;
2. make deterministic token caps non-retryable and test health-checked server
   reuse after request-level HALO failures;
3. interleave independent 10-turn and 20-turn depth-0/2,048-trace replicates;
4. admit a separate treatment that guarantees one child invocation before
   comparing recursive quality; and
5. measure caching with an explicit matched cache-off arm.

## Continuation questions

The continuation is an adaptive queue with two phase cutoffs. It asks:

1. Does deliberately early root-history compaction execute safely, and does it
   change the result relative to the normal 0.85 threshold?
2. Does RLM's apparent 128K advantage survive three tasks and three fixed rows,
   with paired direct controls where the server context permits them?
3. Does doubling HALO's root turn budget repair finalization while holding the
   model, reasoning control, parallel allowance, prompt, fixtures, and scorer
   fixed?
4. Does the same trace-navigation treatment remain usable at 65,536 synthetic
   Graphiti-like traces?

## Frozen matrix

RLM uses the same pinned BF16 `mit-oasys/rlm-qwen3-8b-v0.1` profile and normal
0.85 compaction control. The forced diagnostic is ordered first and changes
only its fingerprinted threshold to 0.20, which maps to 6,553 tokens under the
pinned package's 32,768-token context lookup.

| RLM treatment | Selection | Cases |
| --- | --- | ---: |
| Forced compaction | `qa1`, 128K, rows 11 and 47, threshold 0.20 | 2 |
| Normal depth 1 | `qa1`–`qa3`, 8K/32K/64K/128K, rows 11/47/73 | 36 |
| Direct control | `qa1`–`qa3`, 8K/32K, rows 11/47/73 | 18 |
| Held depth-2 sentinel | `qa1`, 128K, row 11 | 1 |

HALO retains `reasoning_effort = "none"`, two parallel subagents, the no-MTP
Qwen3.8 NVFP4 profile, and the prior scorer. It changes `max_turns` from 10 to
20. Counts 256/2,048/8,192 with seeds 0–2 are exact paired intervention cells;
seeds 3–5 add replication, and 65,536 adds the scale tier. A local admission
canary generated and indexed one 65,536-trace, 92,273,018-byte fixture in 2.16
seconds without a model server or download.

| HALO treatment | Counts | Seeds | Cases |
| --- | --- | --- | ---: |
| Depth 0 | 256, 2,048, 8,192, 65,536 | 0–5 | 24 |
| Depth 1 | 256, 2,048, 8,192, 65,536 | 0–5 | 24 |
| Depth-2 scale sentinel | 65,536 | 0 | 1 |

The plan therefore contains 106 cases: 105 executable and one explicit hold.
At launch, the matrix was intentionally oversubscribed against pathological
double-failure caps and was estimated to fit the remaining window under the
then-observed runtimes. The completed results above show that repeated HALO
failures and cold restarts exhausted that margin. Ordering prioritizes forced
compaction, then the breadth panel, then matched HALO intervention cells. A
phase cutoff produces explicit terminal skips and never converts incomplete
work into zero quality or throughput.

## Deadlines and supervision

All timestamps use explicit MST (`-07:00`):

| Boundary | Timestamp |
| --- | --- |
| RLM admission cutoff | `2026-08-26T02:40:00-07:00` |
| HALO measurement cutoff | `2026-08-26T05:50:00-07:00` |
| Frozen hard stop | `2026-08-26T06:50:00-07:00` |
| Cleanup reserve | 3,600 seconds |

The unattended user service must retain the exact-plan `ExecStopPost` cleanup,
restart only unexpected failures, and have absolute stop timers at 06:00 and
06:20 MST. Raw plans, fixtures, indexes, journals, telemetry, prompts, model
outputs, and server logs remain ignored. The clean overnight baseline is the
the [51cfeed7 scalar bundle](../evidence/campaigns/20260826T012139Z-rlm-halo-overnight-2026-08-25-51cfeed7/manifest.json),
and this continuation is the
[8b52bc5c scalar bundle](../evidence/campaigns/20260826T065438Z-rlm-halo-continuation-2026-08-26-8b52bc5c/manifest.json).
The [overnight publication ledger](rlm-halo-overnight-2026-08-25.md#privacy-and-publication-boundary)
accounts for the preceding plans, smoke attempts, and canaries. These projected
bundles contain only allowlisted scalar outcomes, aggregates, safe provenance,
and telemetry. The complete archive passed deterministic full export and normal
verification, and its immediate re-export reported no change.
