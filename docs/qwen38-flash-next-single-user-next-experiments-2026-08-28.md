# Qwen3.8-Flash-Next single-user serving backlog — 2026-08-28

## Scope

This is the ranked backlog after the frozen 64K autoresearch campaign. It is
not part of that immutable campaign and does not authorize changing its plans,
cutoff, suite, or profile queue. Freeze a new protocol only after the current
campaign is terminal and its scalar evidence is published.

The product target remains one person using a coding agent or cowork-style
assistant. Optimize correct end-to-end task wall time, later-turn latency, and
safe residency on one DGX Spark. Aggregate multi-user throughput is secondary.

## Evidence-backed prior

The retained default is mapped FP8 PLE, `extra_buffer_lazy`, and NEXTN depth
two. Its two independent fresh-C1 lifetimes averaged 29.594 output tok/s; its
two warmed D256 lifetimes averaged 29.402 tok/s. The single mapped depth-three
D256 point reached 32.221 tok/s, but it was unreplicated and its first request
fell to 13.905 GiB available memory, below the frozen 14 GiB floor. See the
[matched depth study](qwen38-flash-next-ple-depth-study-2026-08-27.md).

MTP is the strongest established decode lever. In a separate clean bounded
comparison, MTP3 reached 30.123639 tok/s versus 16.663713 off, a `1.807739x`
gain that saved 137.288 seconds, or 44.682%, over 5,120 output tokens. That is
not a matched MTP2/MTP3 result. See the
[native MTP study](qwen38-flash-next-native-mtp-optimization-2026-08-26.md).

Two independent mapped-PLE MTP2 lifetimes also give a replicated fan-out
signal. Their C2 cases performed exactly twice the fixed 256-token request work
in only `1.155x` and `1.128x` the corresponding three-request C1 case wall.
Projecting each C1 wall linearly to six serial requests, the observed C2 wall
would be 42.243% and 43.582% lower; a six-request serial arm was not measured.
Median per-request E2E increased 11.113% and 11.360%, while median TTFT
increased by 0.171 and 0.161 seconds. These are 4K synthetic fixed-output
observations, not proof for 64K agent tasks, but they justify an
admission-first single-user fan-out test.

Cold start is still the largest managed-lifetime cost. The comparable
mapped-PLE depth lifetimes spent roughly 578–601 seconds in startup before
useful work. For an interactive deployment, retaining a healthy resident
server can be a larger product win than a few percent of decode throughput
when requests would otherwise pay cold start. Residency also consumes the
Spark's memory and device availability and must remain an explicit operator
choice.

The current [frozen campaign](qwen38-flash-next-single-user-autoresearch-2026-08-28.md)
already tests three one-axis questions in order:

1. low reasoning versus explicit no-thinking;
2. 1,024- versus 2,048-token chunked prefill; and
3. NEXTN depth two (`steps=2`, `draft_tokens=3`) versus depth three
   (`steps=3`, `draft_tokens=4`).

Do not duplicate those cells or combine a winning axis with another change in
place. Promotion ends the current fixed queue; continuing with another axis or
a combined champion requires a new frozen profile and campaign.

## Ranked next experiments

### 1. Long shared-prefix Radix-cache pair

This has the highest direct coding/cowork value. The current 60K case is a
one-shot prompt, while the agent cases have small histories; neither measures
repeated long-prefix work across tool turns.

- Control: the promoted profile with the runtime's default Radix cache.
- Candidate: add only `--disable-radix-cache`.
- Suite: one cold 32K–48K coding or document prefix followed by two or three
  appended, deterministic tool turns with byte-identical shared prefix and
  unique suffixes. Every turn's maximum tokenized transcript plus output and
  draft allowance must fit the frozen context and token pool.
- Design: fresh-lifetime ABBA with two independent lifetimes per arm. Treat the
  first turn as cold calibration and score later turns separately.
- Primary outcomes: strict task correctness, later-turn TTFT, and complete task
  wall time. Decode TPS is secondary.
- Required controls: identical tokenized prefixes after chat-template and tool
  serialization, cache residency/eviction bounds, no cross-arm warm state, and
  scalar native cache-hit accounting if a privacy review admits it. Use a
  prescribed transcript, or reject a pair when prior model output or reasoning
  makes the two histories differ.

This experiment asks whether the default cache avoids repeated prefill and
hybrid-state work. It must not infer a cache hit from latency alone.

### 2. Two-way single-user fan-out, admission first

This is an application-scheduling experiment for one person decomposing one
task into two independent subtasks. It is not a multi-user throughput claim.
The retained 64K profile admits only one running request, so first freeze a
separate C2-capable profile and prove its C1 behavior and safety before scoring
parallel work.

- Admission bundle: keep the 65,536-token pool, raise running requests from one
  to two, provide eight lazy recurrent-state slots for two four-slot sequences,
  and capture decode graph batches one and two. Treat this as one inseparable
  concurrency-geometry bundle.
- Capacity gate: precompute the sum of both requests' fully rendered inputs,
  reserved outputs, and draft allowances. It must fit the shared pool with an
  explicit safety margin; full-60K requests are therefore outside this arm.
- Geometry control: compare the C2-capable profile at C1 against the retained
  C1 profile before using it. Reject it if the extra graph/cache geometry
  breaches pressure gates or materially slows one-request work.
- Scheduling control: inside the same admitted C2 profile, run two fixed,
  independent, strictly validated native cowork/agent subtasks serially and
  then in parallel. Counterbalance serial/parallel order across fresh lifetimes
  and keep total prompts, output budgets, tools, and oracle work identical. Add
  Pi only after its separate client-stack admission.
- Primary outcome: time until both correct artifacts are complete. Retain
  per-task E2E, TTFT, output-limit hits, memory, swap, and fairness as
  guardrails.

Promote fan-out only for genuinely decomposable work with disjoint virtual
workspaces or an exact deterministic merge. Sequential tool chains and one
long-context task remain on the C1 path.

### 3. Short-output MTP break-even

The clean MTP3/off result proves the decode benefit at D256, but the primary
native run's 20-token JSON, 36-token tool, and 32-token chat outputs spent
69.4%, 78.3%, and 53.7% of E2E in TTFT. Trained-MTP loading also accounted for
83.86 seconds of the 581.652-second cold start. Treating the clean 20-request
MTP3/off D256 case-wall difference as linear in total output at fixed request
count, then combining it with that separately attributed startup component,
projects a coarse crossover near 3,127 total emitted tokens for that batch
shape. It is not a break-even for one request or an arbitrary short-turn mix,
an MTP2 prediction, or an end-to-end measurement.

- Control: the promoted fixed-depth MTP profile.
- Candidate: an otherwise identical MTP-off profile that removes only the
  complete speculative-decoding bundle.
- Suite: strict short JSON/tool cases, the complete multi-turn agent battery,
  and D256 as a decode anchor. Hold reasoning, prompt, output, parser, cache,
  and sampling policy fixed.
- Design: fresh-lifetime ABBA with two independent lifetimes per arm. Report
  cold start-to-first-correct-task wall and resident task wall separately;
  never subtract startup after the fact to create a synthetic winner.
- Interpretation: an MTP-off arm may justify a specialized ephemeral profile
  if it wins cold managed wall without losing correctness. It should not
  replace the resident default unless it also wins matched resident task wall.

### 4. Continuous decode steps

- Control: the runtime default, `num_continuous_decode_steps=1`.
- Candidate: add `--num-continuous-decode-steps 2`.
- Follow-up: test four only if two wins and remains responsive.
- Design: ABBA with two independent lifetimes per arm against the current
  champion; use D256 and the complete agent battery.
- Primary outcomes: task wall time and output tok/s.
- Guardrails: first visible emission, cancellation latency, tool-call stop
  behavior, finish counts, exact validation, memory, and swap.

The expected mechanism is fewer scheduler and CPU round trips at C1. This is a
low-memory-risk serving flag, but a throughput gain cannot excuse worse
interactive stop or cancellation behavior.

### 5. Decode CUDA-graph causal control

- Control bundle: `--cuda-graph-backend-decode full` with
  `--cuda-graph-bs-decode 1`.
- Candidate bundle: `--cuda-graph-backend-decode disabled`, removing the batch
  argument.
- Design: ABBA with two independent lifetimes per arm; keep startup/capture
  time separate from D256 and agent task wall.
- Interpretation: the expected result is a control speed win. A disabled-graph
  simplification may promote only at `>=0.99x` combined speed if it also
  removes the declared graph-serving bundle or proves at least 1 GiB more
  available memory twice.

Do not capture graph batches above one for a C1 objective. They are unexercised
at one running request and add capture, startup, and headroom cost.

### 6. Conditional next chunk size

Propose another value only if the current queue reaches and completes a valid
1,024/2,048 pair. An earlier promotion ends that queue without a chunk result.

- If 2,048 wins, compare the promoted champion with 4,096.
- If 2,048 loses, compare 1,024 with 512.
- Clone each new profile from the then-current champion, change only
  `--chunked-prefill-size`, and retain ABBA replication.
- Score 60K TTFT/E2E, later-turn agent wall time, D256 throughput, memory, and
  swap separately.

At concurrency one, larger chunks have no scheduling-fairness justification.
They must win on actual prefill or task wall time without harming interactive
latency or pressure safety.

### 7. Adaptive NEXTN, admission first

The fixed-depth evidence shows declining marginal acceptance at deeper
positions, so adaptive proposal length is plausible for mixed code and tool
turns. Do not freeze this treatment until the exact pinned runtime confirms the
accepted adaptive configuration and Qwen3.8-Flash-Next compatibility.

- Control: the promoted fixed-depth champion, top-k one.
- Candidate: one exact adaptive-speculation bundle; do not combine a different
  top-k, chunk size, or reasoning policy.
- Required evidence: per-case proposed and accepted token counts plus position
  histograms, exact output/finish validation, task wall time, memory, and swap.
- Failure rule: an unsupported flag, schema ambiguity, or missing native
  counters rejects the candidate before scored inference. After source and CLI
  audit, at most one explicit unscored counter-admission smoke may establish
  that the required native surface exists.

### 8. Stream coalescing

This is lower priority than compute and cache axes.

- Control: `stream_interval=1`.
- Candidate: `--stream-interval 2`; test four only after a clean win.
- Score complete task wall together with user-visible first emission,
  inter-emission delay, cancellation, and tool-stop latency.

Reduced client/SSE overhead is not a model decode gain. Do not promote on an
apparent client TPS change while responsiveness regresses.

## Diagnostics, not immediate candidates

A separate unscored lifetime with one D256 request and one deterministic
varied-token long prompt should retain allowlisted scalar SM occupancy,
memory-bandwidth, CPU-fault, and NVMe-read counters. The collection method and
attribution window must be pinned; profiling overhead, system-wide faults, and
page-cache state can otherwise confound timing. The present coarse GPU
utilization and power samples cannot distinguish target-weight bandwidth, PLE
page traffic, kernel overhead, or another resource. This diagnostic must use
the retained champion, remain outside a causal candidate pair, and must not
silently change its serving flags.

Overlap scheduling enabled versus `--disable-overlap-schedule` is a lower-value
C1 diagnostic. `mamba_track_interval` should remain frozen until the long
prefix-cache pair proves that hybrid-state reuse is material.

## Explicit depriorities

- Scheduling policy changes are effectively inert at one running request.
- Extra CUDA-graph batch sizes are unexercised at C1 and consume memory.
- Do not lower the 65,536-token pool until exact tokenized input, output
  reserve, and runtime/draft allowance prove that every case still fits;
  increasing it primarily buys capacity rather than C1 decode speed.
- Medium or xhigh reasoning is a correctness-rescue treatment, not a speed
  axis. Low versus no-thinking is already frozen.
- PLE omission changes semantics and previously produced incomplete C4/C8
  requests; it is not a deployment speed candidate.
- Ordinary-buffer depth three has an unsafe pressure history.
- Packed NVFP4 PLE requires a separate artifact/kernel protocol, not a serving
  flag ablation.
- NGRAM speculation, speculative top-k other than one, and two-batch overlap
  remain incompatible with the pinned Qwen PLE path.

## Promotion and publication rules

Every causal comparison changes one declared axis or one inseparable flag
bundle. Use fresh server lifetimes, counterbalanced order, baseline
calibration, hard correctness and lifecycle gates, and at least two independent
lifetimes per arm. Observed lifetime drift is comparable to a 3% speed
threshold, so a single forward run cannot promote.

Publish only the deterministic scalar projection. Raw prompts, completions,
reasoning, tool payloads, agent trajectories, logs, paths, commands, request
identifiers, and native trace bodies remain ignored. Stop immediately on the
existing memory, swap, ownership, and cleanup thresholds. Every future
campaign must freeze its own explicit cutoff; never inherit or extend the
expired cutoff from this campaign.
