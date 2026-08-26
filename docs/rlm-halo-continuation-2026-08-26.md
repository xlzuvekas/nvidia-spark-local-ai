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
The matrix is intentionally oversubscribed against pathological double-failure
caps but fits the remaining window under the observed runtimes with margin.
Ordering prioritizes forced compaction, then the breadth panel, then matched
HALO intervention cells. A phase cutoff produces explicit terminal skips and
never converts incomplete work into zero quality or throughput.

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
outputs, and server logs remain ignored. Any tracked evidence must use the
scalar allowlist exporter and staged verification.
