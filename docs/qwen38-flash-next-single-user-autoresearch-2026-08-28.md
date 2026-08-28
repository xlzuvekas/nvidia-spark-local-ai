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

The local Pi executable is not installed. The frozen SparkBench multi-turn tool
battery is the first admission gate. The pinned Harbor/Qwen Code harness and
fresh-prompt content battery may be used only through their existing offline,
loopback, scalar-only contracts. Harbor task reward is a correctness gate and
agent wall time is an end-to-end diagnostic. The content battery covers code,
technical explanation, reasoning, and prose, but its minimum-length check is
not semantic correctness; its rates cannot promote a candidate that fails the
agentic or coding gate.

Primary speed is the geometric mean of candidate/control
`aggregate_output_tps` ratios over the frozen C1 generation cells. Task wall
time and first-turn TTFT are product-facing diagnostics and become the primary
ordering whenever two configurations produce different valid token counts.
Concurrency greater than one is outside the promotion score.

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

The controller replays these states:

`prepared -> proposed -> pair_frozen -> admitted -> cell_started ->
cell_terminal -> second_cell_started -> pair_complete -> audited ->
provisional|discarded`

A provisional candidate continues through `confirmation_frozen` and the same
cell states to `promoted|discarded`. Separate terminal classifications are
`candidate_crash`, `safety_rejected`, `pair_invalid`, `blocked_environment`,
`blocked_cleanup`, and `stopped`.

If interrupted before a cell starts, replay may continue. If interrupted while
a cell owns a server, recover only that exact lifecycle and invalidate the
pair. If the inter-cell gap exceeds 120 seconds, restart the entire pair with
new lifetimes. If both cells completed but no decision was written, audit and
score idempotently. Never combine resumed or pre-interruption measurements for
promotion.

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
