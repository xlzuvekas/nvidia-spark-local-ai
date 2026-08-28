# Pinned Pi and cowork harness plan — 2026-08-28

## Decision

Use Pi only after it has been normalized into an immutable offline dependency,
and run it inside a newly admitted Pi-specific adaptation of the existing
ephemeral Harbor isolation rather than as an ambient host CLI. Keep
coding-agent and cowork-style results separate:

- Pi + the six existing Harbor tasks measure an end-to-end coding client stack;
- a new deterministic `cowork-core-v1` suite measures structured multi-file,
  retrieval, table, and revision-conflict work through bounded virtual tools;
- the current Qwen3.8-Flash-Next 64K autoresearch suite remains unchanged and
  cannot be relabeled as either benchmark.

This plan is prospective. Do not edit the frozen campaign's harness, profiles,
suite, cutoff, or plans.

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

After the current frozen campaign is terminal:

1. Create an owner-private, normalized, read-only Pi 0.57.1 prefix from a
   separately admitted, hash-locked artifact set. The top-level tarball and a
   dependency-complete installed tree are available, but the full transitive
   tarball cache closure has not been proven.
2. Verify every exact lockfile integrity against locally staged content and
   perform a credential-free offline materialization. Stage any missing
   artifacts outside the measured lifecycle; never install or resolve packages
   during a measured run.
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

## Isolation and endpoint path

The new Pi image and adapter should run Pi as a verified non-root task user
inside each ephemeral Harbor task container. The current Harbor main container
runs the agent as root, so non-root UID, ownership, and capability checks are a
new admission requirement rather than an existing property. If the canary
retains the root agent, preserve the shared-verifier trust limitation in every
result. Pi's coding tools accept absolute paths and include an unrestricted
shell, so a host process is not an acceptable boundary.

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
wrapper timeout with TERM/KILL grace, maximum model turns, concurrency one, one
trial, and no automatic retry.

Start with the existing pinned six-task Harbor set. A new Pi-only manifest must
pin the starting server profile as
`qwen38-flash-next-nvfp4-mtp2-agent64k-low-ple-mapped-sglang` and freeze a new
execution order rather than importing the old Qwen Code/OpenCode
counterbalance:

- `fix-git` as the isolation/correctness canary;
- `cancel-async-tasks`;
- `fix-code-vulnerability`;
- `regex-log`;
- `polyglot-c-py`; and
- `query-optimize`.

These tasks measure coding only. Do not claim that they cover cowork-style
document, retrieval, or structured-data work.

## Exact request policy

Pi 0.57.1's Qwen compatibility collapses every enabled thinking level to a
boolean. It does not preserve the current deployment's nested low-effort
policy. A pinned payload hook must assert the outgoing request schema and force
one of exactly two mappings:

```json
{"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"low"}}
```

or the frozen explicit no-thinking mapping. It must also force temperature
zero and the frozen output limit. Do not rely on Pi's CLI thinking label.

Unit fixtures must prove that the payload hook rejects top-level reasoning
fields, unknown nested fields, missing effort, a changed temperature or token
limit, and any model/endpoint drift. Hold this wrapper constant across serving
arms so the experiment changes server configuration rather than client policy.

## Pi timing and scalar evidence

The trusted wrapper owns monotonic timestamps:

- task wall: immediately before the prompt through `agent_end`;
- request start: `turn_start`;
- TTFT: first text, thinking, or tool-call start;
- model E2E: assistant message end;
- tool wall: tool execution start through end; and
- turn wall: turn start through end.

Publish first-turn and median later-turn TTFT, total task wall, model-request
count, turn count, tool counts by category, tool-error count, timeout and stop
enums, external verifier reward, and cleanup/admission booleans. Pi-reported
input, cache-read, output, and total tokens may be published only after a
canary validates their semantics. Label them `pi_usage`; they are not native
server decode counters.

Wrapper stdout is one exact scalar JSON object. Keep prompts, completions,
reasoning, tool arguments/results, commands, diffs, paths, sessions, request
identifiers, environment, logs, and workspaces in ignored owner-private state.
Reject unknown output keys and secret-scan the exact staged evidence tree.

## Deterministic cowork-core-v1

The cowork suite should use an in-memory synthetic virtual workspace, not
shell or host filesystem access. Reuse the existing bounded tool-loop parser,
strict call-ID handling, lifecycle, telemetry, journals, protocol digests, and
exact JSON oracles. Add four three-variant cases:

| Case | Synthetic workload | Exact pass condition | Target live context |
| --- | --- | --- | ---: |
| `cowork-multifile-brief` | Read six project files containing updates, an erratum, constraints, and distractors; submit a structured brief | Required facts and citations exact; stale facts absent; exact section schema | <=28K |
| `cowork-table-reconcile` | Join four integer-valued CSV-like tables, deduplicate revisions, and produce a canonical table | Exact rows, order, cells, and integer-cent totals | <=22K |
| `cowork-retrieve-revisions` | Search a fixed 96-document corpus with revision chains, decoys, and one prompt-injection string | Four exact facts; latest revisions; citation precision/recall 1.0; injection ignored | 40K–48K |
| `cowork-write-conflict-recovery` | Commit a document update; receive one deterministic revision conflict; reread and merge | Conflict observed; reread; remote edit retained; second commit and final artifact exact | <=26K |

Use scenario-specific virtual tools such as `read_source`, `search_sources`,
`submit_table`, and `commit_document`. Deterministic lexical retrieval replaces
embeddings, and exact oracles replace an LLM judge.

Freeze temperature zero, concurrency one, at most eight turns and sixteen tool
calls, 1,536 output tokens per turn, and a 4 KiB canonical argument limit. For
the selected profile, precompute every request's maximum tokenizer-rendered
input plus its reserved output and speculative allowance across every legal
path. Require that total to remain at or below 61,440 tokens, leaving 4,096
tokens unallocated in the 65,536-token server pool.

The twelve episodes should target roughly 6,500–8,000 assistant-emitted tokens,
including reasoning and serialized tool calls. At the retained ~29.4 tok/s C1
anchor, decode is approximately four to five minutes. The provisional target
is seven to ten minutes including prefill and tools, with a 15-minute
(900-second) suite cap. Cold server startup remains separate.

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
model, and protocol digests remain allowed provenance. Inject unique private
sentinels into every raw surface and exception path, then prove they are absent
from journals, reports, exports, and staged blobs.

## Cleanup and failure rules

Every Pi trial ends through one bounded `finally` path:

1. abort Pi;
2. TERM/KILL the Pi wrapper's exact owned process group within grace;
3. restore network deny-all;
4. run the external verifier;
5. remove the exact task/relay containers and ephemeral task image;
6. delete the isolated Pi home/session/temp state, including shell spill files;
7. stop the exact bridge and model server under the campaign lock;
8. delete the private key/socket; and
9. prove no owned descendant, container, socket, key, or task state remains.

A cleanup ambiguity, timeout escalation, network escape, pin mismatch, payload
drift, oracle failure, or secret-policy violation invalidates the trial and
stops the campaign. Never resume an interrupted task or reinterpret a Pi task
wall result as server-only throughput.

## Implementation order after cutoff

1. Normalize and verify the Pi prefix without invoking the ambient CLI.
2. Build privacy and forced-timeout fixtures for the Pi-core wrapper.
3. Run the `fix-git` canary through the new Pi-specific adaptation of the
   Harbor isolation path.
4. Freeze a coding-only Pi campaign against the six current tasks.
5. Implement and fixture-test `cowork-core-v1` separately.
6. Add Pi to cowork only after the virtual-tool oracle passes with the native
   bounded runner; label Pi results as client-stack measurements.

Pi is viable locally, but the adapter, payload policy, and immutable prefix are
part of the measured system. Their hashes and effects must be explicit rather
than treated as a transparent harness.
