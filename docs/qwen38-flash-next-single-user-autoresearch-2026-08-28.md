# Qwen3.8-Flash-Next single-user autoresearch protocol — 2026-08-28

## Objective

Optimize one pinned Qwen3.8-Flash-Next NVFP4 deployment on one DGX Spark for a
single interactive user doing coding-agent and cowork-style work. The primary
product outcome is less wall time for a correct task, not maximum aggregate
throughput under concurrent users.

This is a serving-configuration search. It does not train or modify model
weights. Model, tokenizer, PLE payload, SGLang image, source overlays, hardware,
prompts, validators, sampling policy, and evaluator versions remain immutable.
One declared serving axis or coupled NEXTN bundle may change per candidate.

## Current status: safety-stopped, no campaign measurements

This campaign has not started and none of its four profiles has produced a
measurement under this protocol. The host is currently safety-stopped because
the preceding buffer-strategy interaction block crossed the frozen 512 MiB
swap-growth limit twice: the ordinary depth-two C6 tail reached
2,473.8359375 MiB of growth, and the following ordinary depth-three startup
reached 3,173.1484375 MiB before any case began. Those observations belong to
the earlier interaction study, not to this campaign.

Do not run any inference command in this document until an operator resets the
Spark and a fresh preflight establishes unambiguous ownership, no unrelated
GPU or container workload, at least 14 GiB `MemAvailable`, and a new recorded
swap baseline no greater than 64 MiB. A reset does not make either rejected
interaction cell valid and does not authorize resuming it. If the reset and
preflight leave less than 4,620 seconds before the frozen cutoff, record the
campaign as stopped without starting a pair; do not move the cutoff or silently
shorten a cell.

## Autoresearch adaptation

The controller design is adapted from Karpathy's
[`autoresearch`](https://github.com/karpathy/autoresearch/tree/228791fb499afffb54b46200aca536f79142f117).
Upstream separates an immutable evaluator in
[`prepare.py`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/prepare.py),
one agent-editable implementation in
[`train.py`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/train.py),
and the persistent experiment policy in
[`program.md`](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md).
Its five-minute training window, single scalar, one-run promotion, destructive
Git reset, and overwritten log are not copied.

Spark serving has about a ten-minute cold start, correctness is lexicographic,
and unified-memory pressure can contaminate later measurements. The local
adaptation therefore uses:

- a 1,800-second inclusive envelope for one fresh server lifetime;
- a 3,600-second envelope for one frozen two-cell pair;
- 120 seconds of owned cleanup grace outside the causal measurement;
- a 900-second final audit reserve;
- an append-only, replayable journal rather than an overwritten log;
- immutable frozen plans for both pair cells before either cell starts; and
- a champion pointer that advances only after reverse-order confirmation.

Do not start a pair unless at least 4,620 seconds remain before the hard
campaign cutoff. The overnight cutoff is 07:00 MST on 2026-08-28. A workload
may finish before its envelope; do not pad it. Startup, evaluator wall time,
TTFT, prefill, and decode remain separate metrics.

## Exact four-profile queue

The queue is finite and immutable. Each candidate below is a one-axis change
from the first profile; public identity and served name are bookkeeping, not
experimental axes.

| Queue | Profile | Role and only intended delta from baseline |
| ---: | --- | --- |
| 1 | `qwen38-flash-next-nvfp4-mtp2-agent64k-low-ple-mapped-sglang` | Baseline: mapped FP8 PLE, `extra_buffer_lazy`, NEXTN depth 2, 1,024-token chunked prefill, low reasoning |
| 2 | `qwen38-flash-next-nvfp4-mtp2-agent64k-none-ple-mapped-sglang` | Request policy changes from thinking/low to explicit no-thinking |
| 3 | `qwen38-flash-next-nvfp4-mtp2-agent64k-low-chunk2k-ple-mapped-sglang` | Chunked-prefill size 2,048 |
| 4 | `qwen38-flash-next-nvfp4-mtp3-agent64k-low-ple-mapped-sglang` | NEXTN depth 3 bundle: three speculative steps and four draft tokens |

All profiles retain the same pinned checkpoint, mapped PLE payload and marker,
SGLang image and overlays, 65,536-token context and token pool, C1 request
geometry, four lazy recurrent states, 0.85 static-memory fraction, tool and
reasoning parsers, and temperature-zero suite. The queue does not permit a PLE
omission arm, ordinary buffers, extra concurrency, a context change, or an
unlisted reasoning effort.

The first matched pair is baseline-to-baseline calibration. Candidate proposals
then follow queue order 2, 3, 4. The request-policy and chunk candidates retain
the baseline server geometry and run before depth three. Depth three is last
because earlier, different depth-three geometries crossed the memory or swap
safety gate. Every candidate is defined against the frozen baseline. If a
candidate is promoted before later proposals are reached, stop this queue: a
later fixed profile may then differ from the champion on more than one axis.
Combining a winning axis with another candidate requires a new profile, new
frozen plan, and new protocol revision rather than an in-place edit.
Calibration is recorded in its own admission record and does not consume the
candidate journal's global pair index; search pair index zero therefore still
runs champion then candidate.

## Exact nine-case suite

Every profile is bound to
`manifests/suites/qwen38_flash_next_sglang_agent64k_autoresearch.toml`; using a
different suite is an admission failure. Cases run in this exact order:

| Order | Case ID | Shape |
| ---: | --- | --- |
| 1 | `json-smoke` | JSON capability; 1 repetition; 64 output tokens |
| 2 | `tools-smoke` | tool-call capability; 1 repetition; 64 output tokens |
| 3 | `synthetic-exact-answer-v2` | strict quality; 2 repetitions; 512 output tokens |
| 4 | `agentic-select-and-call` | 3 variants; at most 6 turns and 4,096 output tokens per turn |
| 5 | `agentic-no-tool` | 3 variants; at most 6 turns and 4,096 output tokens per turn |
| 6 | `agentic-two-hop` | 3 variants; at most 6 turns and 4,096 output tokens per turn |
| 7 | `agentic-tool-error-recovery` | 3 variants; at most 6 turns and 4,096 output tokens per turn |
| 8 | `long-context-needle-60000-agent-c1` | 60,000 prompt repetitions; 1 repetition; 128 output tokens |
| 9 | `agent64k-decode-256-c1-v1` | 1 warm-up, then 5 repetitions of 256 output tokens |

All nine cases use concurrency one and temperature zero. The first eight have
no warm-up. Agentic ordering, arguments, dependencies, recovery, and final
answers are validated; the 60K case is a synthetic exact-key capacity probe,
not natural-document comprehension. Warm-up work from the final decode case is
not scored.

## Controller commands and time budgets

The current safety stop makes the execution command non-executable until the
reset and preflight above. Verification and planning do not launch inference.
First verify the pinned overlay and PLE payload without downloading anything,
then inspect the exact scalar proposal:

```bash
python3 prepare_sglang_overlays.py --prepare-ple-ablation
python3 prepare_sglang_overlays.py --verify-ple-cache
python3 sparkbench.py inventory --sizes
python3 sparkbench.py list --verbose
python3 sparkbench.py autoresearch-plan \
  --campaign manifests/campaigns/qwen38_flash_next_single_user_autoresearch.toml \
  --dry-run
```

Freeze all fourteen pristine cell plans once. The command prints the immutable
campaign directory; substitute that exact printed path in the following
commands:

```bash
python3 sparkbench.py autoresearch-plan \
  --campaign manifests/campaigns/qwen38_flash_next_single_user_autoresearch.toml \
  --results results/autoresearch

python3 sparkbench.py autoresearch-run results/autoresearch/FROZEN_CAMPAIGN_DIR
python3 sparkbench.py autoresearch-summarize results/autoresearch/FROZEN_CAMPAIGN_DIR
```

One `autoresearch-run` invocation crosses at most one complete pair boundary:
first calibration, then one screen or confirmation pair. This deliberate stop
lets the operator export, verify, commit, and push the scalar checkpoint before
explicitly invoking the same command again. The controller counterbalances
cell order, starts each frozen cell in a new process group, sends `SIGINT` at
the 1,800-second causal boundary, allows 120 seconds for owned cleanup, and
uses `SIGKILL` only as a last resort. The two causal envelopes sum to the frozen
3,600-second pair budget; the 900-second audit reserve remains outside it. A
timeout, interruption, or inter-cell gap over 120 seconds invalidates the pair;
never resume that cell.

## Evaluator hierarchy

Eligibility is decided before speed:

1. Both frozen plan fingerprints and exact artifact pins verify.
2. Both cells are fresh, non-resumed server lifetimes.
3. Every planned case is terminal and every strict validator passes.
4. Tool selection, tool arguments, ordering, dependency, recovery, and final
   answers pass the frozen scalar-only agentic battery.
5. Completion counts and finish policy match; invalid or truncated output has
   no throughput score.
6. No unrelated GPU/container workload, implicit download, ownership
   ambiguity, cleanup failure, or raw payload publication occurs.
7. Sampled `MemAvailable` stays at or above 14 GiB and swap growth stays at or
   below 512 MiB.

No pinned, campaign-admitted Pi harness is available. A transitive Pi package
elsewhere on the host is not a frozen offline dependency and is excluded. The
SparkBench multi-turn tool battery is therefore a proxy admission gate. The
pinned Harbor/Qwen Code harness and fresh-prompt content battery may be used
only as separately labeled diagnostics through their existing offline,
loopback, scalar-only contracts; neither is part of the nine-case suite.
Harbor task reward is a correctness gate and agent wall time is an end-to-end
diagnostic. The content battery covers code, technical explanation, reasoning,
and prose, but its minimum-length check is not semantic correctness; its rates
cannot promote a candidate that fails the agentic gate.

Consequently, this campaign can compare deterministic JSON, tool use,
exact-answer behavior, synthetic 60K retention, and C1 decode under the four
serving profiles. It cannot support a claim about Pi, autonomous repository
editing, cowork document handling, or end-to-end single-user productivity.

Primary speed is the geometric mean of six fixed-task speed factors. For the
four agentic cases, each factor is control/candidate median task wall time. For
the 60K needle it is control/candidate median end-to-end time. For the fixed
D256 decode case it is candidate/control aggregate output TPS. First-turn TTFT
is a separate guardrail. Any missing validation, truncation, or mismatched work
removes the entire pair from scoring; token-rate gains cannot compensate for a
different or incorrect task. Concurrency greater than one is outside the
promotion score.

## Calibration, screening, and promotion

Begin with one control-to-control pair. Stop the search if its overall
geometric-mean ratio is outside `[0.97, 1.03]`, any primary case is outside
`[0.95, 1.05]`, or its median-TTFT ratio is outside `[0.90, 1.10]`. Those bounds
are a calibration gate, not a confidence interval.

Pair order alternates. Odd search pairs run champion then candidate; even pairs
run candidate then champion. A candidate is provisionally screen-positive only
when all hard gates pass and:

- its pair geometric-mean speed ratio is at least `1.03`;
- no primary-case ratio is below `0.95`; and
- its median-TTFT ratio is at most `1.10`.

Before promotion, run a second pair in the opposite order. Both pair-level
geometric means must be strictly above `1.00`, and their combined geometric
mean must be at least `1.03`. A simplicity candidate may instead be retained
only when the confirmed combined ratio is at least `0.99` and it either adds at
least 1 GiB of minimum available-memory headroom or removes a declared serving
flag bundle. Equal/noisy results do not advance the champion.

## Safety and recovery

A candidate syntax or startup error may be discarded after exact owned cleanup
and a restored preflight. A real memory-floor breach, swap-growth breach, OOM
under pressure, ownership ambiguity, or cleanup failure is campaign-terminal;
later cells would inherit contaminated state. Never relax a gate, retry a
safety rejection, resume a measured cell, use implicit downloads, or stop an
unowned process.

Each admitted campaign profile also enables a separate fail-closed 250 ms host
watchdog. Its synchronous starting sample requires at least 14 GiB
`MemAvailable` and no more than 64 MiB used swap; later samples enforce the same
memory floor and at most 512 MiB additional swap. Missing, malformed, or
internally inconsistent `/proc/meminfo`, a changed `SwapTotal`, or a dead
monitor thread terminates the run. A breach interrupts only the exact owned
SGLang container, remains authoritative over a concurrent request error, and
forces removal before optional log collection. Safety-enabled profiles reject
`--keep-server`.

The controller replays a smaller append-only state machine: campaign start,
candidate start, pair start with frozen order, two ordered cell completions,
pair score, and candidate decision. A passing screen returns the candidate to
the pair-start state for its reverse-order confirmation. Rejection returns to
the fixed queue; promotion or queue exhaustion completes the campaign. The
control-to-control calibration is integrity-bound in a separate admission
record and never consumes a search pair index. Pressure, OOM, ownership,
cleanup, audit, validation, or measurement failures terminate the campaign.
`blocked_environment` is pre-journal admission state, not a measured result.

If interrupted before a cell starts, replay may continue. If interrupted while
a cell owns a server, recover only that exact lifecycle and invalidate the
pair. If the inter-cell gap exceeds 120 seconds, restart the entire pair with
new lifetimes. If both cells completed but no decision was written, audit and
score idempotently. Never combine resumed or pre-interruption measurements for
promotion.

## Checkpoint, commit, and push policy

Push the frozen clean protocol before any campaign preflight. Within a pair,
append a private local checkpoint after each terminal cell and verify owned
cleanup immediately, but do not edit, commit, pull, or push between the two
cells; that would risk plan drift and consume the 120-second inter-cell bound.

After every complete audited pair—calibration, screen, or confirmation—export
the deterministic scalar projection, review only the new allowlisted files,
stage those exact paths, and run both normal and staged evidence verification.
Commit and push that checkpoint before proposing the next pair. Apply the same
export/verify/commit/push boundary immediately after any campaign-terminal
safety event or the final cutoff. A failed push stops further pairs until it is
resolved; local raw state alone is not a substitute for the requested remote
checkpoint.

Never stage `results/`, raw journals, server logs, telemetry streams, prompts,
completions, reasoning, tool payloads, request identifiers, local paths, or
commands. Frequent pushes do not relax the scalar-only publication contract,
and Git work never enters a measured 30-minute cell or 60-minute pair.

The controller enforces that boundary by returning after at most one pair. A
status of `active` means the just-finished pair is locally replayable and the
same frozen campaign directory may be passed to `autoresearch-run` after the
checkpoint is pushed. `blocked_environment` starts no cell and writes no
campaign transition; `terminated` and `complete` must not be resumed.

## Evidence and interpretation

Raw plans, journals, server logs, telemetry, prompts, completions, reasoning,
tool payloads, request identifiers, and agent trajectories remain under
ignored `results/`. Only an allowlisted deterministic scalar projection may be
tracked. Each candidate record must retain its exact profile delta, cell order,
run status, gate outcomes, aggregate C1 metrics, agent success and wall time,
TTFT diagnostics, memory/swap extrema, and keep/discard decision.

This one-night adaptive search is local and sequential. Even a confirmed local
winner is not a general Qwen3.8 recommendation, an official agent benchmark,
or evidence about a different context length, reasoning policy, runtime,
quantization, checkpoint, workload, or machine.
