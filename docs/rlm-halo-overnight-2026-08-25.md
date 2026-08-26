# RLM and HALO overnight campaign — 2026-08-25

## Purpose

This is the frozen protocol for one deadline-bounded, local-only comparison of
direct long-context prompting, recursive language-model inference (RLM), and
HALO trace analysis on one NVIDIA DGX Spark. It is a campaign plan, not a result
report.

The run asks two narrow questions:

1. On the same fixed BABILong-derived rows, when does a bounded depth-1 RLM
   loop improve exact answers enough to justify its extra calls, tokens, and
   wall time relative to one direct request?
2. On deterministic synthetic Graphiti-like traces, how do HALO depth 0, 1,
   and a small depth-2 supplement change failure-family identification,
   counting, citations, tool use, token use, cache reuse, and wall time?

The campaign manifest is
[`manifests/campaigns/rlm_halo_overnight.toml`](../manifests/campaigns/rlm_halo_overnight.toml).
Planning freezes that manifest, the selected model profiles, dataset inventory,
case order, deadlines, and content hashes into an ignored run directory before
any model server starts.

## Frozen time window

All timestamps carry the explicit `-07:00` MST offset.

| Boundary | Frozen timestamp | Meaning |
| --- | --- | --- |
| RLM stop | `2026-08-25T20:38:20-07:00` | Start no further RLM or direct episodes; journal the remainder as deadline-skipped. |
| Measurement stop | `2026-08-26T05:38:20-07:00` | Start no further HALO episode. |
| Hard stop | `2026-08-26T06:38:20-07:00` | Absolute campaign deadline. |
| Cleanup reserve | 3,600 seconds | The full final hour is reserved for server teardown, recovery checks, and summary generation. |

Each episode is clipped to the remaining phase budget with additional teardown
margin. The runner will not start a case that cannot retain its minimum margin.

## RLM matrix

RLM uses the dedicated `rlm-qwen3-8b-bf16` profile:
`mit-oasys/rlm-qwen3-8b-v0.1` in BF16 with a 40,960-token serving context.
The long BABILong document remains in the RLM Python context variable; bounded
model subcalls inspect it without placing the entire 128K document into one
server request. This checkpoint has fixed model-recipe reasoning behavior:
the runtime effort interface is unsupported and has not been validated, so the
campaign records `reasoning_control = "fixed_unsupported"` and does not claim a
thinking-off RLM treatment.

| Treatment | Lengths | Tasks | Rows | Cases |
| --- | --- | --- | --- | ---: |
| RLM depth 1 | 8K, 32K, 64K, 128K | `qa1` | 11 | 4 |
| Direct paired control | 8K, 32K | `qa1` | 11 | 2 |
| Held RLM depth-2 sentinel | 128K | `qa1` | 11 | 1 |

That produces 7 planned RLM-phase cases: 6 executable cases and one explicit
held depth-2 sentinel. Length and treatment order rotate by replicate so the
two direct/RLM pairs are not always exposed to the same server age. The 64K and
128K cells are recursive-only coverage. This one-row QA1 panel preserves all
four context lengths while retaining retry margin in the remaining fixed RLM
window; it is not a three-task BABILong panel.

The recursive bounds are fixed at:

- eight iterations;
- two concurrent recursive subcalls;
- a 262,144-token upstream `max_tokens` guard for each RLM instance (root or
  recursive child), not a global cross-recursion token cap;
- 768 output tokens per RLM model call;
- 900 seconds per RLM episode; and
- required digest-pinned Docker isolation for recursive workers.

Depth-1 cases explicitly enable pinned RLM 0.1.3 root-history compaction at
`0.85`. The package maps the served model name to a 32,768-token context for
compaction, so the frozen threshold is 27,852 tokens. Depth-2 remains held
because the pinned package does not propagate compaction into child RLMs.

Direct controls have a 180-second episode timeout and request only the one
location label. Accuracy uses the fixed location-label uniqueness rule. The
campaign also records scalar wall time, RLM iterations, recursive subcalls,
vLLM prompt/cached/generation-token and successful-request deltas, prefix-cache
counters, and effective generation tokens per wall second.

RLM 0.1.3 reports `UsageSummary` for the current RLM instance but does not fold
recursive-child usage into the parent's returned summary. The journal therefore
labels those SDK-reported counts separately and marks their recursive coverage;
the exclusive vLLM counter deltas are the campaign-wide token and successful-call
accounting for an episode. The 900-second killable-worker timeout remains the
global resource bound for recursive work.

### Smoke-test boundary

The pre-campaign smoke and exact canaries exercised serving, worker isolation,
timeout, retry, compaction admission, and cleanup paths. The final compacted
128K RLM gate and matched no-MTP HALO canary completed and scored. Their scalar
outcomes were:

| Probe | Outcome |
| --- | --- |
| 8K direct control | Completed in 6.522 seconds, used 7,694 prompt and 64 generation tokens (9.812 effective generation tokens/s), and answered incorrectly. |
| 128K depth-1 RLM, 24,576-token guard, 300-second timeout | Reached `TokenLimitExceededError` after 178.865 seconds. |
| 128K depth-1 RLM, 131,072-token guard, 300-second timeout | Reached the 300-second episode timeout before producing an answer. |
| 128K depth-1 RLM, 131,072-token guard, 600-second timeout | Reached `TokenLimitExceededError` after 458.422 seconds. |
| 128K depth-1 RLM, 262,144-token guard, 900-second timeout | The first attempt cleared the earlier aggregate-token boundary, accepted eight model requests, and then returned HTTP 400 on the ninth/default-answer request after 558.709 seconds. Accepted root prompts had grown through 39,802 tokens; reserving 768 output tokens leaves a 40,192-token input envelope under the served 40,960-token context. The retry received only the campaign's remaining 361.453-second allowance and timed out. No scored result was produced; exact container and network cleanup was verified. |
| Exact 128K depth-1 RLM with compaction enabled at 0.85 | Completed correctly on attempt one in 504.926 seconds over 9 successful model requests. It processed 107,755 prompt tokens, of which 95,728 were cached (88.839%), and 6,150 generation tokens at 12.180 effective generation tokens/s. The trajectory finished before the 27,852-token root-history trigger, so `compaction_count = 0`: this admits the exact treatment but does not exercise the compaction branch. Normal cleanup and the independent exact-plan post-stop cleanup both verified no owned resources remained. |
| Pre-fix 256-trace depth-1 HALO on Qwen3.8 with MTP3 | Server startup took 261.759 seconds. All 8 model requests returned HTTP 200, then the episode failed after 111.263 seconds with `UserError` caused by `ValidationError`. Source isolation identified the pinned HALO `call_subagent` `AgentAsToolInput` parse, which overwrote the SDK error handler. The local protocol workaround catches only that validation class and returns a constant retry result without retaining arguments or validation text. |
| Post-fix 256-trace depth-1 HALO on Qwen3.8 with MTP3 | Server startup took 267.789 seconds. The canary cleared the old validation boundary with 13 HTTP 200 model requests, then encountered 20 repeated HTTP 400 responses and failed after 285.775 seconds with `EngineAgentExhaustedError`; no scored result was produced. |
| Matched post-fix HALO canary on Qwen3.8 without MTP | Server startup took 245.576 seconds and the episode completed in 312.132 seconds over 19 model requests. It used 151,790 prompt tokens, of which 108,192 were cached (71.277%), and 2,656 generation tokens at 8.509 effective generation tokens/s. The response was valid JSON with family precision 1.0, recall 0.25, F1 0.4, count accuracy 0.041667, citation coverage 0.25, and citation precision 1.0. One subagent completed and the episode made 32 tool calls. |
| Qwen3.6 diagnostic canary | Server startup took 118.787 seconds. All 30 chat requests returned HTTP 500 because xgrammar could not import `normalize_tool_choice`; HALO exhausted the agent after 67.113 seconds with `EngineAgentExhaustedError`. |

The RLM failures show that upstream `max_tokens` counts cumulative input and
output usage as repeated root history grows; it is not merely a per-response
output allowance and cannot protect one oversized request. The 262,144-token
guard cleared the aggregate boundary and exposed the next one: this integration
leaves pinned RLM 0.1.3 compaction disabled, so the root history grows until a
request no longer fits the server. The smallest depth-1 repair is an explicit,
fingerprinted `compaction = true` treatment at threshold `0.85`. The pinned RLM
model lookup maps this Qwen3-8B name to 32,768 tokens, yielding a 27,852-token
threshold and 12,340 tokens of headroom after the 768-token output reserve.
That materially changes the treatment, so it was admitted through the exact
128K canary above. The canary completed but did not cross the trigger; overnight
results must therefore report compaction counts and treat zero as unexercised,
not as proof that compaction ran. Pinned RLM does not propagate these settings
into recursive children, so the depth-2 supplement remains held pending a
narrow propagation patch and test.

All HALO canaries used the `none` reasoning controls. The exact canary campaign
completed cleanly and verified cleanup. At `n=1`, the no-MTP success validates
transport and control plumbing, but its quality was weak despite valid
structure and precise reported citations. Treat it as admission evidence, not
a model-quality conclusion. The MTP3 and Qwen3.6 failures remain integration
defects and produced no quality or TPS result.

### BABILong comparison boundary

The context, question, and target come from exact cached Arrow files at
`RMT-team/babilong@ee0d588794c7ac098062ee0d247c733d62e94fe2`.
The planner admits only the expected three-column, 100-row files and the
selected fixed row indices. Context and answer text never enter the journal.

These are BABILong-derived paired probes, not an official BABILong leaderboard
submission. They use one task and one fixed row, a local model,
campaign-specific direct/RLM prompts, bounded recursive inference, and a narrow
local scorer. Report within-campaign paired differences; do not compare the
resulting accuracy directly with published full-suite scores.

## HALO matrix

HALO has 19 logical episodes:

| Treatment | Trace counts | Seeds | Cases |
| --- | --- | --- | ---: |
| Depth 0 | 256, 2,048, 8,192 | 0, 1, 2 | 9 |
| Depth 1 | 256, 2,048, 8,192 | 0, 1, 2 | 9 |
| Depth-2 supplement | 8,192 | 0 | 1 |

Depth-1 and depth-2 runs permit at most two parallel subagents. Every agent has
a ten-turn root budget, subagents receive the runner's bounded smaller budget,
each model response is capped at 1,024 tokens, and each episode is capped at
600 seconds. Depth 0 forces one effective parallel slot and provides the
non-recursive HALO control.

The sole admitted overnight serving profile is `qwen38-27b-nvfp4-halo`. It
uses the Qwen3.8 NVFP4 artifact with eight-sequence serving geometry and MTP
disabled. The profile makes `{"enable_thinking":false}` a server-side chat-template
default as well as a request-body default. This matters because HALO's root and
subagent calls use the OpenAI Agents SDK while its synthesis and compaction
tools call Chat Completions directly. The campaign forces the Agents SDK to
Chat Completions so all four paths use the locally validated vLLM surface. The
plan records `reasoning_effort = "none"`; the pinned HALO model config omits the
top-level API field, while the serving profile enforces the equivalent Qwen
control with the server-side `enable_thinking=false` default.

Qwen3.8 supports the graded effort values `low`, `medium`, and `xhigh`; `high`
is not supported. The current overnight control remains `none` and does not mix
effort levels. A separate later Qwen3.8 supplement may pair `low` versus
`xhigh`, but only after the selected effort is verified on root, subagent,
synthesis, and compaction requests; in particular, the direct compaction path
must no longer be an open propagation gap.

There is no overnight failover or model-panel comparison. The matched Qwen3.8
MTP3 profile is diagnostic-only until its repeated HTTP 400 boundary is fixed;
its no-MTP pairing isolates serving-path and MTP sensitivity while holding
model revision, quantization, and concurrency geometry fixed. Qwen3.6 is also
diagnostic-only because its pinned xgrammar stack cannot serve tool requests.
Every overnight HALO event therefore uses the sole admitted no-MTP profile.

### Synthetic Graphiti-like traces

Each fixture is deterministically generated from `(trace_count, seed)` and has
exactly two OTel-shaped spans per trace: a `memory.reflect` agent span and one
Graphiti-like tool span. Four of these six failure families are active for a
given seed: search timeout, write conflict, duplicate edge, entity split,
invalid arguments, and empty retrieval. Some failures have OTel error status;
others are semantic anomalies inside otherwise successful tool results.

HALO must return a constrained JSON family list containing counts and verified
example trace IDs. The transient answer is graded for JSON validity,
family-level precision/recall/F1, count accuracy, exact-count rate, citation
precision, and citation family coverage. The journal retains only those
scalars plus durable-item, tool-call, subagent, token, cache, and timing counts.

This fixture tests bounded trace exploration and delegation. It is not a run
against the user's 72K Graphiti traces, a Graphiti integration benchmark, or a
claim about production failure prevalence.

## Local artifacts and admission

The campaign performs no implicit downloads. Planning and startup require the
exact cached artifacts and fail closed when any pin is absent:

- `alexzhang13/rlm@0b45df99c43fb3844a3b796a15d13c0f9d07afd8`
  (`rlm==0.1.3`);
- `context-labs/HALO@b7f8509745d67b499b4e80efe20ea37c03426a74`
  (`halo-engine==0.3.5`);
- `openai-agents==0.14.7`, `openai==2.32.0`, and `pyarrow==21.0.0` in the
  ignored external loop environment;
- `mit-oasys/rlm-qwen3-8b-v0.1` revision
  `399e50b54e59248c7e79476fe4a7f1772bb7c75b` in BF16;
- Qwen3.8 NVFP4 revision
  `6128240ebaf4eaa7bad2b3d1c72c37d677c5f462` and image digest
  `sha256:4a2f33a884222f7049b983263ad9976f89452bb81affecf5b67d89ad35c1bc31`;
- the sole admitted HALO profile uses that exact Qwen3.8 artifact and image;
  and
- the exact cached BABILong Arrow files described above.

SparkBench's repository-wide lock and unrelated-workload preflight still
apply. Exactly one managed model server runs at a time and binds only to
loopback. Every run or resume revalidates the pinned loop environment, isolated
worker image, and the size and SHA-256 of every selected BABILong Arrow file
against the frozen plan before a server starts.

### Isolation canary and Docker substitution

The intended unprivileged bubblewrap worker was tested before freezing this
protocol. Although `/usr/bin/bwrap` is installed, this host reports
`kernel.apparmor_restrict_unprivileged_userns=1`, and a minimal user-namespace
canary failed at UID-map setup with `Permission denied`. Treating the binary's
mere presence as isolation would therefore be false admission.

The frozen manifest instead requires `worker_isolation = "docker"`. Recursive
workers use the already cached worker image
`nvcr.io/nvidia/vllm:26.07-py3@sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268`
with `--pull=never` and no GPU devices. The worker has a read-only root,
read-only exact-HEAD source export, read-only Python runtime and read-only loop
environment; ignored files, untracked files, raw results, and the worktree
itself are not mounted. Only a 1 GiB `noexec,nosuid,nodev` `/tmp` is writable. It runs as the invoking
UID/GID with all capabilities dropped, `no-new-privileges`, a 128-PID limit,
an 8 GiB memory limit, and no inherited host environment or credentials; the
command supplies only `HOME`, `PYTHONNOUSERSITE`, and `PYTHONPATH` alongside
the pinned image defaults. It joins a plan-labeled, Docker-internal network
with the managed vLLM container, so the worker can reach only local containers
and has no Internet route. The host-facing vLLM endpoint remains literal
loopback. The worker is named and labeled from the frozen plan and case
identities, removed after every episode, and force-removed on timeout; the
internal network is ownership-checked and removed at phase cleanup.

HALO 0.3.5 normally prepares a Deno/Pyodide `run_code` sandbox on first use;
that preparation would download npm and wheel assets absent from the local
cache. This offline campaign explicitly forces HALO's sandbox resolver to
return unavailable, so `run_code` is not registered. Trace query, count,
search, view, synthesis, context, and bounded subagent tools remain enabled.
This is part of the frozen treatment and prevents implicit, unpinned downloads.

## Privacy and publication boundary

This is a scalar-only campaign by construction:

- BABILong context, questions, targets, direct completions, RLM responses and
  trajectories remain transient inside killable workers;
- RLM persistence and trajectory metadata are disabled;
- recursive RLM workers use the Docker boundary above with bounded stdin/stdout
  and a temporary home; no host credential environment is forwarded;
- HALO `run_code`, telemetry, and sensitive Agents SDK tracing are disabled;
  text deltas are not consumed, and tool or subagent answers are discarded
  after transient grading;
- generated Graphiti-like JSONL fixtures, indexes, raw server logs, telemetry,
  journals, and summaries remain under ignored `results/`; and
- the append-only journal accepts only allowlisted scalar values and the
  frozen case/profile dimensions.

Do not commit the raw run directory. Any later tracked evidence must go through
the repository's scalar allowlist exporter and staged verification. The
protocol document contains pins and matrix dimensions, never captured prompts,
completions, reasoning, tool payloads, request identifiers, credentials, or
local run paths.

## Failure and resume rules

- A timed-out or failed case is journaled, its owned server is stopped, and the
  case may be attempted once more. Two failed attempts make it exhausted.
- RLM and HALO retry only their same frozen serving profiles; HALO has no
  overnight profile failover.
- Resume reloads the immutable plan and append-only journal, recovers only
  containers bearing the plan's exact run identity, skips completed or other
  terminal cases, and continues the first pending case.
- The v2 reader retains v1 plan compatibility for offline summaries. Execution
  still requires the plan's exact repository revision, so changing protocol
  code never silently resumes an older plan under new semantics.
- An unexpected process failure can therefore be resumed safely. A graceful
  `SIGINT` or `SIGTERM` is an intentional campaign stop: remaining cases are
  marked terminal `case_skipped_campaign_stop` and are not re-opened by
  `resume`.
- Phase-deadline skips are terminal and remain explicit in the summary. The
  runner never borrows the cleanup reserve for additional measurements.
- The unattended launch is wrapped in a user-service watchdog configured to
  signal the controller before the hard stop, bound its shutdown grace, and
  restart only unexpected nonzero exits. This outer guard does not move any
  frozen phase deadline or reopen terminal cases. Its stop grace must cover the
  longest frozen startup timeout plus cleanup; a 90-second grace is invalid
  because the observed cold starts took as long as 279.914 seconds.
- Docker owns the model and worker container processes outside the user-service
  cgroup. A normal controller return performs exact-identity cleanup, but the
  watchdog alone cannot guarantee cleanup after controller `SIGKILL`. Every
  unattended unit therefore invokes the exact-plan, idempotent `cleanup`
  command as `ExecStopPost`; it validates the frozen plan before removing only
  workers, a server, and a network carrying that plan's exact ownership.
- Server startup rejection, case failure, timeout, exhaustion, deadline skip,
  recovery, and cleanup failure are distinct scalar
  journal events. Failed or partial work must not be reported as zero quality
  or zero throughput.

## Commands

From the repository root, freeze the plan and note the printed run directory:

```bash
python3 loop_campaign.py plan
```

Optional paths are explicit:

```bash
python3 loop_campaign.py plan \
  --campaign manifests/campaigns/rlm_halo_overnight.toml \
  --models manifests/models.toml \
  --results results/loop-campaigns
```

Run the frozen plan, substituting the directory printed by `plan`:

```bash
python3 loop_campaign.py run results/loop-campaigns/<run-directory>
```

After an unexpected interruption, resume that same frozen directory:

```bash
python3 loop_campaign.py resume results/loop-campaigns/<run-directory>
```

Regenerate the deterministic scalar summary without starting a server:

```bash
python3 loop_campaign.py summarize results/loop-campaigns/<run-directory>
```

Idempotently verify and remove only that frozen plan's owned Docker resources:

```bash
python3 loop_campaign.py cleanup results/loop-campaigns/<run-directory> \
  --workspace /absolute/path/to/local-llm
```

Use `--workspace /absolute/path/to/local-llm` with `run`, `resume`, or `cleanup`
when the current working directory is not the repository root. Planning after
the hard deadline is rejected rather than silently shifting the frozen window.
