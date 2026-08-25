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
server request. Thinking is disabled in every request.

| Treatment | Lengths | Tasks | Rows | Cases |
| --- | --- | --- | --- | ---: |
| RLM depth 1 | 8K, 32K, 64K, 128K | `qa1`, `qa2`, `qa3` | 11, 47, 73 | 36 |
| Direct paired control | 8K, 32K | `qa1`, `qa2`, `qa3` | 11, 47, 73 | 18 |
| RLM depth-2 supplement | 32K, 128K | `qa2`, `qa3` | 11, 47 | 8 |

That produces 62 RLM-phase episodes. Length, task, and treatment order rotate
by replicate so the direct/RLM pairs are not always exposed to the same server
age. Only the 18 shared 8K/32K cells are paired direct-versus-RLM comparisons;
the 64K/128K and depth-2 cells are recursive-only coverage.

The recursive bounds are fixed at:

- eight iterations;
- two concurrent recursive subcalls;
- a 24,576-token upstream `max_tokens` guard for each RLM instance (root or
  recursive child), not a global cross-recursion token cap;
- 768 output tokens per RLM model call;
- 300 seconds per RLM episode; and
- required digest-pinned Docker isolation for recursive workers.

Direct controls have a 180-second episode timeout and request only the one
location label. Accuracy uses the fixed location-label uniqueness rule. The
campaign also records scalar wall time, RLM iterations, recursive subcalls,
vLLM prompt/cached/generation-token and successful-request deltas, prefix-cache
counters, and effective generation tokens per wall second.

RLM 0.1.3 reports `UsageSummary` for the current RLM instance but does not fold
recursive-child usage into the parent's returned summary. The journal therefore
labels those SDK-reported counts separately and marks their recursive coverage;
the exclusive vLLM counter deltas are the campaign-wide token and successful-call
accounting for an episode. The 300-second killable-worker timeout remains the
global resource bound for recursive work.

### BABILong comparison boundary

The context, question, and target come from exact cached Arrow files at
`RMT-team/babilong@ee0d588794c7ac098062ee0d247c733d62e94fe2`.
The planner admits only the expected three-column, 100-row files and the
selected fixed row indices. Context and answer text never enter the journal.

These are BABILong-derived paired probes, not an official BABILong leaderboard
submission. They use only three tasks and three fixed rows, a local quantized
model, campaign-specific direct/RLM prompts, bounded recursive inference, and a
narrow local scorer. Report within-campaign paired differences; do not compare
the resulting accuracy directly with published full-suite scores.

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

The ordered serving profiles are:

1. primary: `qwen38-27b-nvfp4-mtp3-halo`, derived from the eight-sequence
   Qwen3.8 NVFP4+MTP3 throughput profile; and
2. fallback: `qwen36-35b-a3b-nvfp4-mtp3-halo`, derived from the conservative
   Qwen3.6 35B-A3B NVFP4+MTP3 profile.

Both profiles make `{"enable_thinking":false}` a server-side chat-template
default as well as a request-body default. This matters because HALO's root and
subagent calls use the OpenAI Agents SDK while its synthesis and compaction
tools call Chat Completions directly. The campaign forces the Agents SDK to
Chat Completions so all four paths use the locally validated vLLM surface.

The profiles are failover choices, not a matched model panel. The runner tries
Qwen3.8 first. If it cannot admit, or its first HALO case fails before any HALO
case completes, it locks the campaign to Qwen3.6 and retries that case. After
the first successful HALO case, the selected profile is fixed across failures
and resumes. Every event retains its actual `profile_id`, and summaries group
by both treatment and profile rather than combining mixed-profile observations.

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
- Qwen3.6 NVFP4 revision
  `491c2f1ea524c639598bf8fa787a93fed5a6fbce` and image digest
  `sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268`;
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

## Failure, fallback, and resume rules

- A timed-out or failed case is journaled, its owned server is stopped, and the
  case may be attempted once more. Two failed attempts make it exhausted.
- RLM restarts the same frozen serving profile. HALO may switch once from the
  ordered primary to the frozen fallback before retrying.
- Resume reloads the immutable plan and append-only journal, recovers only
  containers bearing the plan's exact run identity, skips completed or other
  terminal cases, and continues the first pending case.
- An unexpected process failure can therefore be resumed safely. A graceful
  `SIGINT` or `SIGTERM` is an intentional campaign stop: remaining cases are
  marked terminal `case_skipped_campaign_stop` and are not re-opened by
  `resume`.
- Phase-deadline skips are terminal and remain explicit in the summary. The
  runner never borrows the cleanup reserve for additional measurements.
- The unattended launch is wrapped in a user-service watchdog configured to
  signal the controller before the hard stop, bound its shutdown grace, and
  restart only unexpected nonzero exits. This outer guard does not move any
  frozen phase deadline or reopen terminal cases.
- Server startup rejection, case failure, timeout, fallback selection,
  exhaustion, deadline skip, recovery, and cleanup failure are distinct scalar
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

Use `--workspace /absolute/path/to/local-llm` with `run` or `resume` only when
the current working directory is not the repository root. Planning after the
hard deadline is rejected rather than silently shifting the frozen window.
