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

The schema-2 campaign froze all fourteen pristine cell plans from clean, pushed
revision `aa9cca8` at 01:09 MST on 2026-08-28. Its first admission returned
`blocked_environment` with `starting_swap_above_clean_limit`. At that preflight,
used host swap was 889,256 kB (868.414 MiB), above the frozen 64 MiB start cap;
`MemAvailable` was 118,269,252 kB and no container was running. The controller
created no event journal, calibration record, cell summary, worker state,
container, or model request. None of the four profiles has produced a
measurement under this protocol.

The verified [scalar evidence index](../evidence/index.json) now contains all
fourteen frozen cells as `nonterminal` with `measurement_terminal=false`.
Those entries publish the plan topology and artifact bindings only; they are
not throughput, wall-time, quality, or memory observations. The prior schema-1
freeze was measurement-free and was moved intact into the ignored private
archive before this schema-2 export.

The host stop originated when the preceding buffer-strategy interaction block
crossed the frozen 512 MiB swap-growth limit twice: the ordinary depth-two C6
tail reached 2,473.8359375 MiB of growth, and the following ordinary
depth-three startup reached 3,173.1484375 MiB before any case began. Those
observations belong to the earlier interaction study, not to this campaign.

Do not run any inference command in this document until an operator resets the
Spark and a fresh preflight establishes unambiguous ownership, no unrelated
GPU or container workload, at least 14 GiB `MemAvailable`, and a new recorded
swap baseline no greater than 64 MiB. A reset does not make either rejected
interaction cell valid and does not authorize resuming it. If the reset and
preflight leave less than 4,930 seconds before the frozen cutoff, record the
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

- a 30-second cap for each fresh worker to publish its bound
  `measurement_started` marker;
- a 1,800-second inclusive causal envelope from `measurement_started` through
  `measurement_complete` for one fresh server lifetime;
- a 3,600-second causal measurement envelope for one frozen two-cell pair;
- 120 seconds of owned cleanup grace per cell, ending at `server_stopped`;
- a 120-second inter-cell allowance measured from the first
  `server_stopped` marker;
- a 10-second finalization bound from the final cell's `server_stopped` to
  `run_complete`;
- a 900-second final audit reserve;
- an append-only, replayable journal rather than an overwritten log;
- immutable frozen plans for both pair cells before either cell starts; and
- a champion pointer that advances only after reverse-order confirmation.

Do not start a pair unless at least 4,930 seconds remain before the hard
campaign cutoff. The exact admission arithmetic is `2 * 1,800` seconds of
causal measurement + `2 * 120` seconds of owned cleanup + `2 * 30` seconds to
reach the bound start markers + `120` seconds for the inter-cell gap + `10`
seconds for final-cell finalization + `900` seconds of audit reserve = `4,930`
seconds. The first cell's at-most-10-second finalization is inside the
120-second inter-cell allowance, which begins at its `server_stopped` marker.
The overnight cutoff is 07:00 MST on 2026-08-28. A workload may finish before
its envelope; do not pad it. Startup, evaluator wall time, TTFT, prefill, and
decode remain separate metrics.

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

Do not invoke `autoresearch-summarize` after a fresh pre-journal
`blocked_environment` return. The current summarizer derives controller state
only; with no journal it would rewrite the preserved blocker summary as
`planned` and drop its blocker codes. The frozen campaign remains safely
resumable without that command. Use the controller's run output and existing
summary for the preflight outcome; use the summarizer only after a controller
journal exists.

The controller and `autoresearch-checkpoint` command enforce a remote evidence
boundary between settled pairs. A run returns after each calibration, screen,
or confirmation pair. The next pair is not admitted until the latest completed
pair has matching tracked evidence in a clean commit that is identical to the
live upstream and the private acknowledgement has been written.

Every run invocation first exhaustively recovers the exact frozen worker and
container identities, validates controller structure, and replays or reconciles
any ordered raw-complete prefix. The checkpoint command performs the same
exhaustive recovery before its strict raw replay and evidence/Git proof. Those
local safety steps precede checkpoint evaluation, so a cleanup, ownership,
topology, or incomplete-one-use failure takes precedence over
`checkpoint_required`. The remote gate runs only at a stable boundary before
new work: after calibration, after a nonfinal rejection, or after a passing
screen before its confirmation. It never runs between pair arms, for an already
admitted `candidate_started` or `pair` state, while a scored pair still needs
its deterministic decision, or while a terminal tail still needs its
deterministic completion transition.

One `autoresearch-run` invocation crosses at most one complete pair boundary:
first calibration, then one screen or confirmation pair. This deliberate stop
lets the operator export, verify, commit, and push the scalar checkpoint before
explicitly invoking the same command again. The controller counterbalances
cell order and starts each frozen cell in a new process group. `run_start`
binds the plan fingerprint and one-use nonce. The durable
`measurement_started` monotonic marker opens the 1,800-second causal clock;
`measurement_complete` closes it and opens the separate 120-second owned
cleanup phase; `server_stopped` closes cleanup and opens a 10-second
finalization phase; and `run_complete` closes that phase. Only a cell whose
remaining raw audit also passes is scoreable. At an expired phase the
supervisor interrupts the exact owned group with `SIGINT` and uses `SIGKILL`
only as a bounded last resort. A search-pair score records its audit reserve
from the later cell's durable `run_complete` wall timestamp, not from a later
controller replay or scoring clock. A timeout, incomplete lifecycle, or
inter-cell gap over 120 seconds invalidates the pair; never resume or rerun
that frozen cell.

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
a cell owns a server, recover only that exact lifecycle. A frozen cell is
one-use: if it started but its exact artifacts do not reach bound
`measurement_started`, `measurement_complete`, `server_stopped`, and
`run_complete` terminal state, the cell and pair are invalid and the campaign
terminates; neither arm is launched again. An inter-cell gap over 120 seconds
likewise invalidates the pair rather than authorizing a replacement lifetime.

A raw-complete cell is different from an incomplete started cell. Its existing
artifacts may be reprojected and reconciled into a missing controller
completion or score only when the frozen fingerprint, plan integrity, nonce,
terminal markers, validations, telemetry, and pair order all replay exactly.
Reconciliation reuses those durable artifacts and performs no inference. If a
raw-complete pair lacks a decision, recompute the observation from both raw
cells and score or decide idempotently; never combine a partial cell with a new
lifetime or relabel pre-interruption measurements.

## Checkpoint, commit, and push policy

Push the frozen clean protocol before any campaign preflight. Within a pair,
fsync each cell's private lifecycle markers and verify exact owned cleanup, but
do not run any Git operation, evidence export, live-remote check, or other
network operation between the two cells. In particular, never run the
`autoresearch-checkpoint` command after only one arm: its evidence
and live-upstream proofs belong strictly after a complete pair, outside the
120-second inter-cell bound.

After every complete audited pair—calibration, screen, or confirmation—export
the deterministic scalar projection, review only the new allowlisted files,
stage those exact paths, and run both normal and staged evidence verification.
Commit and push that checkpoint before proposing the next pair. Apply the same
export/verify/commit/push boundary immediately after any campaign-terminal
safety event or the final cutoff. A failed push stops further pairs until it is
resolved; local raw state alone is not a substitute for the requested remote
checkpoint.

The explicit workflow is:

```bash
python3 sparkbench.py export-evidence \
  --results results --output evidence --replace
python3 sparkbench.py verify-evidence evidence
git add evidence
python3 sparkbench.py verify-evidence evidence --staged
git commit -m "Record autoresearch pair evidence"
git push
python3 sparkbench.py autoresearch-checkpoint \
  results/autoresearch/FROZEN_CAMPAIGN_DIR
python3 sparkbench.py autoresearch-run \
  results/autoresearch/FROZEN_CAMPAIGN_DIR
```

The explicit acknowledgement is a private, scalar-only, mode-0600 record under
ignored `logs/autoresearch-checkpoints/`, keyed by the frozen campaign
integrity. It binds the latest completed pair, the controller-journal prefix,
the verified working and Git-index evidence, clean `HEAD`, and the identical
live upstream. Keeping it out of both `results/` and `evidence/` prevents the
acknowledgement from becoming an input to the corpus whose checksum it proves.
It is never a substitute for the tracked scalar evidence or the pushed commit.
An explicit checkpoint may bind the latest completed pair after the campaign
has become terminal. That includes a fully measured but failed calibration, or
the last completed pair when a later pair terminated partway through. The
partial pair itself is never acknowledged. A terminal campaign with no
completed pair has nothing to acknowledge and returns `no_completed_pair`;
`autoresearch-run` never gates or resumes a terminal campaign.

Never stage `results/`, raw journals, server logs, telemetry streams, prompts,
completions, reasoning, tool payloads, request identifiers, local paths, or
commands. Frequent pushes do not relax the scalar-only publication contract,
and Git work never enters a causal measurement or an admitted pair lifecycle.

The controller returns after at most one pair. A missing, newer, or changed
acknowledgement returns `checkpoint_required` before local host admission or
another pair start. That status is a resumable, nonterminal pause: it starts no
cell, does not write or rewrite `summary.json`, and appends no failure
transition; `autoresearch-run` prints that in-memory status and uses exit status
3. It otherwise exits 0 for `active`/`complete` and 1 for blocked or terminated
state. `autoresearch-checkpoint` exits 0 after a verified acknowledgement, 1
when proof is not ready, and 2 for structural corruption. After the exact
checkpoint is acknowledged, the same frozen campaign may resume. `active`
means the just-finished pair is locally replayable; it does not by itself prove
that the remote checkpoint exists.
`blocked_environment` starts no cell and writes no campaign transition;
`terminated` and `complete` must not be resumed.

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
