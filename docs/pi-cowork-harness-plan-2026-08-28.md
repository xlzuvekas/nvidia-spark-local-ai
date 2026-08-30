# Pinned Pi and cowork harness plan — 2026-08-28

## Decision

Use Pi only after it has been normalized into an immutable offline dependency,
and run it inside a newly admitted Pi-specific adaptation of the existing
ephemeral Harbor isolation rather than as an ambient host CLI. Keep
coding-agent and cowork-style results separate:

- Pi + the six existing Harbor tasks measure an end-to-end coding client stack;
- a new deterministic `cowork-core-v1` suite measures structured multi-file,
  retrieval, table, and revision-conflict work through bounded virtual tools;
- the Qwen3.8-Flash-Next 64K autoresearch suite is admission-expired and cannot
  be relabeled as either benchmark.

This plan is prospective. Do not edit the frozen campaign's harness, profiles,
suite, cutoff, or plans before its fixed 07:00 cutoff.

Freeze three records rather than one overloaded campaign: C1 Pi coding on the
six Harbor tasks, C1 `cowork-core-v1` on its twelve episodes, and a separate
C1/C2 cowork fan-out campaign and profile. They may share admitted code and
pins, but require distinct campaign IDs and evidence bundles. Never pool coding
capability, cowork capability, and scheduling results.

At the frozen 84-file executable boundary
`33170881721d0dce0f4466495110b336a7451fcd1635c5667f7fc5f722f7599f`, Pi was
design-only. The 2026-08-29 follow-up adds one narrow prerequisite, not a Pi
benchmark: the admission-only current-SM121 low-thinking/tool profile, its
private controller, and an image-local static parser preflight. The exact image
imports `qwen3` reasoning and `qwen3_coder` tool parsers without starting a
server, mounting host
weights, loading a model, or using a GPU. See the [preflight record](qwen38-flash-next-sm121-agent-admission-preflight-2026-08-29.md).
Before a future controller lifetime can start, its separate offline target
tokenizer probe must also prove the exact low-thinking/tool request budget
against the read-only cached snapshot; this remains a prerequisite, not a Pi
benchmark or agent result.

There is now a checked-in minimal frozen source closure, a tested
scripts-disabled direct-materialization command, and one retained
owner-private normalized prefix that passed the static admission pin. There is
still no static core wrapper, Pi/cowork manifest, runner, C2 scheduler,
validator, or evidence exporter. The current generic agentic and historical
Harbor paths deliberately enforce
different C1-only topologies; reuse their isolation and oracle patterns, but do
not widen their schemas in place to admit Pi.

The temporary tree and the `pi-agent-core` root entrypoint now also have a
tracked, schema-locked static-admission pin. `pi-core-prefix-admit` can verify
an explicitly supplied external retained prefix read-only against it; it does
not create a prefix, import JavaScript, alter the admission-only model profile, or
authorize a Pi/cowork run.

## Local Pi availability

For planning, a read-only local audit independently observed a complete
installation of `@mariozechner/pi-coding-agent@0.57.1` and its matching
`pi-agent-core`, `pi-ai`, and `pi-tui` packages; its dependency closure
validates under Node 22. The exact npm artifact is cached locally with
integrity:

`sha512-u5MQEduj68rwVIsRsqrWkJYiJCyPph/a6bMoJAQKo1sb+Pc17Y/ojwa+wGssnUMjEB38AQKofWTVe8NFEpSWNw==`

That is planning evidence of local availability, not a tracked benchmark
admission. The only complete installation belongs to another dirty checkout;
it is not on `PATH`, is not a normalized tree, and has no campaign-owned
full-tree fingerprint. An older 0.49.3 artifact is also cached, while a
lockfile reference to 0.72.1 has no installed offline tree. Neither is a
substitute for admitting 0.57.1 exactly.

Do not invoke the ambient Pi CLI even for discovery in a measured lifecycle.
It loads user resources and extensions before normal execution and would make
the dependency surface ambiguous.

## Immutable Pi prefix

After the fixed cutoff and retirement of the superseded runtime:

1. Create an owner-private, normalized, read-only Pi 0.57.1 prefix from the
   separately admitted, hash-locked artifact set. The full 222-record closure
   and one temporary scripts-disabled smoke have been verified, but no retained
   campaign prefix exists.
2. Re-verify every exact lockfile integrity against locally staged content and
   perform a credential-free offline materialization for the retained prefix.
   Stage any missing artifacts outside the measured lifecycle; never install or
   resolve packages during a measured run.
3. Freeze every transitive lock entry, the full tree digest, regular-file count
   and bytes, entrypoint digest, Node binary/tree digests, wrapper digest,
   literal system-prompt digest, tool-schema digest, and payload-policy digest.
4. Reject symlinks outside the prefix, unexpected files, mutable ownership,
   ambient packages, extensions, themes, skills, sessions, or config.
5. Verify all pins before task admission and again before every agent trial.

The first implementation should use a small Node wrapper around
`pi-agent-core`, not the full TUI/CLI. Give it in-memory messages, a fixed model
object, a literal prompt, no retries, no compaction, no session/config
discovery, and no current-directory/date/user-file injection.

The native Python runner may reuse deterministic schemas and exact oracles for
suite admission. In the Pi arm, however, pinned `pi-agent-core` owns the
message/tool loop and the wrapper validates its event stream, tool schema, call
IDs, limits, and terminal state. Do not double-run the Python tool parser, and
do not route Pi or cowork results through the existing `kind="agentic"`
manifest or evidence schema.

## Isolation and endpoint path

The new Pi image and adapter should run Pi as a verified non-root task user
inside each ephemeral Harbor task container. The current Harbor main container
runs the agent as root, so non-root UID, ownership, and capability checks are a
new admission requirement rather than an existing property. A root-agent
`fix-git` run may be an unscored plumbing canary only; no coding correctness or
latency result is publishable until non-root admission passes. Pi's coding
tools accept absolute paths and include an unrestricted shell, so a host
process is not an acceptable boundary.

Adapt the existing isolation and authenticated-relay design through a new
Pi-specific campaign; do not reuse the old campaign manifest or record:

```text
Pi in task container
  -> task-loopback relay
  -> owner-private Unix socket
  -> managed host bridge
  -> SGLang on host loopback
```

Keep default-drop networking with no DNS, public, or gateway route. A
non-secret placeholder may identify the relay inside the task. The real server
credential is available only to relay/bridge infrastructure through the
owner-private mounted socket/key directory; it never enters Pi arguments,
environment, task state, or evidence.

For the coding campaign, expose exactly `read`, `bash`, `edit`, and `write`
inside the task container. Freeze a 900-second task wall ceiling, a lower
wrapper timeout with TERM/KILL grace, maximum model turns, concurrency one, and
no automatic retry. One `fix-git` episode is an unscored plumbing canary only.
Any scored coding claim requires at least two complete six-task blocks in
independent fresh server lifetimes and a frozen aggregation rule. That
standalone canary is outside every block estimator; after admission, each block
runs a new `fix-git` episode as its scored first task.

Start with the existing pinned six-task Harbor set. The historical
`qwen38-flash-next-nvfp4-mtp2-agent64k-low-ple-mapped-sglang` profile supplies
only measured geometry and rate priors; its SM121 TRT-LLM overlay is
superseded and must not serve a new Pi campaign. First build and admit a new
64K PLE-capacity/lazy/MTP2 profile on an explicitly pinned SM121 Triton runtime,
including native ARM64 build, import, runtime, and varied-token correctness
gates; then rebaseline it and give the Pi manifest a new identity. The current
candidate composition uses the `io_uring` PLE reader, not the historical
persistent-mmap overlay, and remains unadmitted until those gates pass. A future
mmap port is a separate integration and admission gate. Freeze a new execution
order rather than importing the old Qwen Code/OpenCode counterbalance:

- `fix-git` as the isolation/correctness canary;
- `cancel-async-tasks`;
- `fix-code-vulnerability`;
- `regex-log`;
- `polyglot-c-py`; and
- `query-optimize`.

These tasks measure coding only. Do not claim that they cover cowork-style
document, retrieval, or structured-data work.

## Episode, block, and replicate

An **episode** is one task executed by one fresh Pi instance in one fresh task
container, private home, in-memory history, and workspace. A C1 **block** is
the fixed ordered six coding episodes or the fixed ordered twelve cowork
episodes, all served by one newly started model-server lifetime. Clean every
C1 episode's Pi, task, relay, home, and workspace before starting the next one;
keep only the admitted server, bridge, and private endpoint infrastructure
alive until the block ends. The serial C2 pair uses the pair-level teardown
defined below so cleanup cannot enter its makespan.

A scored **replicate** is one complete identical block, not one task or model
request. Every coding or C1 cowork claim requires at least two complete blocks
on independent fresh server lifetimes. Freeze the episode order and
lifetime-level estimator before launch; never pool per-request ratios or call
multiple episodes within one server lifetime independent replicates. The C2
scheduling record uses its separately defined counterbalanced lifetime panel
below.

## Exact request policy

Pi 0.57.1's Qwen compatibility collapses every enabled thinking level to a
boolean. It does not preserve the current deployment's nested low-effort
policy. The first Pi/cowork campaign pins exactly one `thinking_low` mapping,
and a pinned payload hook must assert the final outgoing request after every
Pi/provider transform:

```json
{"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"low"}}
```

It must also force temperature zero and the frozen output limit. Explicit
no-thinking is a separate frozen arm, never a fallback inside this campaign.
Do not rely on Pi's CLI thinking label. Public evidence retains only
`reasoning_policy="thinking_low"` and the payload-policy digest, never the
request JSON.

Unit fixtures must prove that the payload hook rejects top-level reasoning
fields, unknown nested fields, missing effort, a changed temperature or token
limit, and any model/endpoint drift. Hold this wrapper constant across serving
arms so the experiment changes server configuration rather than client policy.

## Episode history and cache boundary

Start every episode with a fresh in-memory Pi instance, empty history, private
home, and disjoint workspace. Disable saved sessions, discovery, compaction,
retries, and cross-episode history. History grows normally only within an
episode; later turns are intended to exercise that episode's own growing
prefix.

Prevent accidental reuse between episodes with an admitted non-prompt cache
namespace: generate a private per-episode `cache_salt`, keep it stable across
that episode's turns, and assert it after every provider transform. For C1
blocks only, an admitted reset between episodes may replace the namespace. C2
requires two simultaneously valid independent namespaces; a parallel pair
cannot reset between tasks, and a serial in-window reset would contaminate its
makespan. If the pinned runtime cannot prove namespace consumption, C2 is
non-runnable and must be refrozen rather than measured. Never inject a sentinel
into the measured prompt. Require unique salts across episodes and zero
device/host/storage native hits on every episode's first request. Keep salts
private. Publish request-scoped cached-token counts only after a zero-hit
semantics canary reconciles them with native counters. Normalize an absent/null
detail object to zero only for that admitted behavior; malformed or unsupported
telemetry remains invalid. Otherwise omit the counts, and never infer a cache
hit from TTFT.

## C1 and C2 scheduling admission

The newly built, admitted, and rebaselined 64K profile is C1-only. Freeze a separate C2 bundle
with the same 65,536-token context/pool, `--max-running-requests 2`,
`--max-mamba-cache-size 8`, and `--cuda-graph-bs-decode 1 2`. Before fan-out,
compare this eight-lazy-slot profile at C1 with the newly admitted C1
profile. Use the same exact serial cowork pair in fresh ABBA lifetimes with two
independent lifetimes per profile. The primary is the ratio of arithmetic-mean
lifetime resident wall; require `C2-profile/C1-profile <= 1.0102`, every exact
oracle, sampled `MemAvailable >= 14 GiB`, starting used swap `<= 64 MiB`, and
swap growth `<= 64 MiB`. Keep D256 separate. Do not define the workload,
replicate count, estimator, or threshold after seeing results.

For every C2 pair, the maximum combined fully rendered histories, reserved
outputs, and MTP allowance over every legal simultaneous path must total at
most 61,440 tokens, leaving 4,096 tokens unallocated.
Run two independent Pi instances and process groups with private homes,
histories, and workspaces behind one common release barrier. In both serial and
parallel modes, prepare and admit both task containers, Pi processes, homes,
and workspaces before the timed release. Parallel releases both tasks; serial
releases the second only after the first terminal result. C2 means two
independent subtasks for one user; it is not a multi-user result or permission
to parallelize a sequential tool chain.

For serial C2, when task one reaches an ordinary model terminal state eligible
for continuation, the common orchestrator timestamps it and releases task two
in the same transition. A campaign-stopping safety, provenance, or cleanup fault
instead skips task two and enters pair cleanup. Task one's terminal state must
already prohibit further model or tool work, but do not run its verifier, remove
its container, delete its workspace, or perform other heavy cleanup before that
release. After task two terminates, verify and clean both tasks before ending
the server lifetime. Parallel C2 uses the same pair-level teardown after both
terminal results. This keeps cleanup out of both makespans without allowing an
active first task to contaminate the second.

The smallest admission-only sequence is one C1 `fix-git` trial, one C1
`cowork-write-conflict-recovery` variant, then one C2 pair containing
byte-distinct `cowork-table-reconcile` and conflict-recovery variants. This
checks isolation, low-effort payload attestation, growing history, mutation and
conflict recovery, dual-client cleanup, and C2 capacity. It makes no speed
claim. The pair remains conditional on exact tokenizer/path admission: if both
worst-case legal rendered histories, output reservations, and MTP allowance do
not fit the 61,440-token budget, freeze shorter C2-specific variants. Never
truncate dynamically or pair the 40K--48K retrieval case.

For a first descriptive scheduling panel, define workload set A as table and
conflict variant zero and set B as the corresponding variant one. Fresh
lifetime A runs set A serially and set B in parallel. Fresh lifetime B runs set
A in parallel and set B serially, with the serial task order reversed. Score
time-to-both only when both exact oracles pass. Repeat that counterbalanced
two-lifetime panel once, yielding two independent panels, four fresh server
lifetimes, and two observations for every set/mode. For each panel, divide the
sum of its two parallel makespans by the sum of its two serial makespans;
require both ratios to be at most `0.90`. For every task variant, require its
parallel per-task resident wall to be at most `1.25x` its matched serial
per-task resident wall and its common-bridge-measured parallel first-turn TTFT no more
than `0.50` seconds above its matched serial TTFT. Every oracle and the exact
memory/swap gates above must pass. Keep common-bridge model E2E separate. Keep the
six-task coding result C1-only and separate from cowork; never pool their
scores.

Require common-release skew at most `10 ms`. Within every scored parallel task
pair, at least one pair of outbound model-request intervals must overlap and an
admitted scheduler metric must show maximum running-request count two during
that pair. The common host bridge/orchestrator timestamps release, both outbound
request intervals, and both terminal results on one monotonic clock; clocks in
separate task relays or wrappers cannot establish overlap or makespan. If any
of those attestations is absent, label the result dual-client offered load and
make it non-promotable. Retain per-task wall/fairness, running/queued maxima,
memory, swap, and cleanup as guardrails.

## Pi timing and scalar evidence

Record cold server start-to-ready, task-container/Pi setup, resident task wall,
external verifier wall, and certified cleanup wall separately. The trusted
wrapper owns task-local monotonic timestamps:

- resident task wall: instruction release through `agent_end` or terminal
  abort, including Pi orchestration, model requests, and tools;
- turn wall: Pi `turn_start` through `turn_end`;
- tool wall: tool execution start through end.

The common host bridge/orchestrator owns model timing after the final payload
is serialized:

- request start: bridge dispatch toward the loopback model endpoint;
- TTFT: first model response content/thinking/tool event; and
- model E2E: terminal model response.

Publish bridge request E2E separately from Pi turn wall. Pi `turn_start` is not
evidence that an HTTP request has been dispatched. Task relays deliver terminal
notices to the common orchestrator, which timestamps them. For C2, only that
clock can establish request overlap, makespan, and release skew; wrapper clocks
remain task-local.

For C2, makespan is the common release barrier through the later terminal
result. For serial work, it is release of the first task through the second
terminal result. Never substitute summed per-task walls for makespan or fold
setup, verifier, or cleanup into resident task wall.

Publish first-turn and median later-turn TTFT, summed and maximum request/tool
wall, total resident task wall or makespan, model-request count, turn count,
tool counts by category, tool-error count, bounded overlap/release-skew fields,
timeout and stop enums, external verifier reward, and cleanup/admission
booleans. Per-turn timing arrays and traces remain private. Pi-reported input,
cache-read, output, and total tokens may be published only after a canary
validates their semantics against the admitted native source. Label them
`pi_usage`; they are not native server decode counters, and client-stack task
wall is not server TPS.

Wrapper stdout is one exact scalar JSON object. Keep prompts, completions,
reasoning, tool arguments/results, commands, diffs, paths, sessions, request
identifiers, environment, logs, and workspaces in ignored owner-private state.
Reject unknown output keys and secret-scan the exact staged evidence tree.

## Deterministic cowork-core-v1

The cowork suite should use an in-memory synthetic virtual workspace, not
shell or host filesystem access. The native admission runner may reuse the
bounded Python parser. The Pi arm reuses only canonical scenarios, tool
schemas, limits, exact oracles, protocol-digest construction, and scalar
validators; `pi-agent-core` remains the sole message/tool-loop owner. Add four
three-variant cases:

| Case | Synthetic workload | Exact pass condition | Target live context |
| --- | --- | --- | ---: |
| `cowork-multifile-brief` | Read six project files containing updates, an erratum, constraints, and distractors; submit a structured brief | Required facts and citations exact; stale facts absent; exact section schema | <=28K |
| `cowork-table-reconcile` | Join four integer-valued CSV-like tables, deduplicate revisions, and produce a canonical table | Exact rows, order, cells, and integer-cent totals | <=22K |
| `cowork-retrieve-revisions` | Search a fixed 96-document corpus with revision chains, decoys, and one prompt-injection string | Four exact facts; latest revisions; citation precision/recall 1.0; injection ignored | 40K–48K |
| `cowork-write-conflict-recovery` | Commit a document update; receive one deterministic revision conflict; reread and merge | Conflict observed; reread; remote edit retained; second commit and final artifact exact | <=26K |

Use scenario-specific virtual tools such as `read_source`, `search_sources`,
`submit_table`, and `commit_document`. Deterministic lexical retrieval replaces
embeddings, and exact oracles replace an LLM judge.

Freeze temperature zero, client concurrency one per Pi instance/episode, at
most eight turns and sixteen tool calls, 1,536 output tokens per turn, and a
4 KiB canonical argument limit. The C2 campaign runs two such independent
concurrency-one Pi instances at once; no global client limiter may serialize
their model requests. For
the selected profile, precompute every request's maximum tokenizer-rendered
input plus its reserved output and speculative allowance across every legal
path. Require that total to remain at or below 61,440 tokens, leaving 4,096
tokens unallocated in the 65,536-token server pool for C1. For C2, evaluate
every legal simultaneous pair of paths and require the sum across both requests
to remain at or below 61,440; individual-request compliance is insufficient.

The twelve episodes should target roughly 6,500–8,000 assistant-emitted tokens,
including reasoning and serialized tool calls. The historical superseded
runtime's ~29.4 tok/s C1 result is a planning prior only; refreeze timing and
the suite ceiling after the newly built and admitted runtime baseline. A provisional seven-to-ten
minute resident target and 15-minute (900-second) suite cap must not be carried
forward as measured expectations without that rebaseline. Cold server startup
remains separate.

## Cowork scoring and publication

Correctness is lexicographically ahead of speed. A sample passes only when its
final virtual artifact, citations, workspace mutation set, tool behavior, and
recovery state all match the deterministic oracle.

The scalar projection may contain:

- pass/failure enum;
- fact, citation, row, cell, and retrieval counts;
- stale/unsupported fact counts;
- recovery booleans and conflict count;
- turns and tool-call counts by category;
- prompt, completion, cached-prompt, and reasoning token counts when their
  source semantics are validated;
- first-turn and aggregate later-turn TTFT;
- summed request time, task wall time, emission count, and output-limit hits;
- task success rate and solved tasks per second; and
- pinned public version/digest identifiers and lifecycle booleans.

Do not publish filenames, source IDs, queries, table values, synthetic
workspace or artifact hashes, prompts, completions, reasoning, arguments, tool
results, trajectories, request IDs, commands, or paths. Public pinned package,
model, and protocol digests remain allowed provenance. Put unique private leak
canaries only in ignored fixture/raw-state surfaces and exception paths, never
in measured prompts, then prove they are absent from journals, reports,
exports, and staged blobs.

## Cleanup and failure rules

Every C1 episode and every terminating C2 pair ends through one bounded task-level
`finally` path. Apply each step to both tasks in a pair:

1. abort Pi;
2. TERM/KILL the Pi wrapper's exact owned process group within grace;
3. restore network deny-all;
4. run the external verifier;
5. remove each exact task/relay container and ephemeral task image;
6. delete each isolated Pi home/session/temp state, workspace, and shell spill
   files;
7. prove no episode-owned descendant, container, home, workspace, or task state
   remains.

At the end of each block or scheduling lifetime--normal, failed, invalid,
interrupted, campaign-stopped, or aborted--an unconditional second bounded
`finally` path stops the exact bridge and model server under the campaign lock,
deletes the private key/socket, and proves no lifetime-owned descendant,
container, socket, key, or state remains. If an episode cleanup is uncertain,
do not reuse the server for the next episode; invalidate the lifetime and run
the lifetime cleanup immediately.

A wrong artifact, oracle miss, verifier reward zero, output-limit finish, or
ordinary bounded timeout is a valid failed model trial. Preserve its scalar
outcome and continue only after certified cleanup; stopping on those failures
would selection-bias success evidence. Pin or payload drift, network escape,
malformed telemetry, secret-policy failure, or uncertain/escalated cleanup
invalidates the trial and stops the campaign. Never resume an interrupted task
or reinterpret a Pi task-wall result as server-only throughput.

## Implementation order after cutoff

1. Build privacy and forced-timeout fixtures for the Pi-core wrapper against
   the retained admitted prefix.
2. Run the `fix-git` canary through the new Pi-specific adaptation of the
   Harbor isolation path.
3. Freeze a coding-only Pi campaign against the six current tasks.
4. Implement and fixture-test `cowork-core-v1` separately.
5. Add Pi to cowork only after the virtual-tool oracle passes with the native
   bounded runner; label Pi results as client-stack measurements.

Pi is viable locally, but the adapter, payload policy, and immutable prefix are
part of the measured system. Their hashes and effects must be explicit rather
than treated as a transparent harness.
