# Qwen3.8-27B DGX Spark benchmark

This file preserves the exact original Qwen3.8 experiment first. The
repository-wide SparkBench measurement and publication contract follows in
[SparkBench protocol and evidence publication](#sparkbench-protocol-and-evidence-publication).

Measured on 2026-08-14 with the configuration in `compose.yaml`:

- NVIDIA GB10, one GPU
- Official BF16 `Qwen/Qwen3.8-27B` weights
- vLLM `0.1.dev19754+g3a0914114`
- 65,536-token maximum context
- FP8 KV cache
- One concurrent sequence
- 52% GPU/unified-memory utilization
- Thinking disabled

## Decode

Three warm runs, each capped at 256 generated tokens:

| Run | TTFT | Elapsed | Decode rate |
| --- | ---: | ---: | ---: |
| 1 | 0.423 s | 65.38 s | 3.93 tok/s |
| 2 | 0.353 s | 66.16 s | 3.87 tok/s |
| 3 | 0.314 s | 65.46 s | 3.91 tok/s |

Median TTFT was **0.353 seconds** and median decode throughput was
**3.91 tokens/second**. GPU SM utilization held at 96% throughout the sampled
decode window.

## Prefill

These are end-to-end approximations calculated as prompt tokens divided by
time to first output token. Each prompt was unique to avoid prefix-cache hits.

| Prompt tokens | TTFT | Approximate prefill rate |
| ---: | ---: | ---: |
| 173 | 0.389 s | 445 tok/s |
| 1,069 | 0.983 s | 1,088 tok/s |
| 4,141 | 3.615 s | 1,146 tok/s |

Run the benchmark again with:

```bash
python3 benchmark.py
```

## Quantization and MTP comparison

The same benchmark was repeated with `Inferact/Qwen3.8-27B-NVFP4`, first
without speculative decoding and then with the model's built-in MTP head using
three draft tokens.

| Configuration | Resident weights | Median TTFT | Median decode | 4,141-token prefill |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 51.1 GiB | 0.353 s | 3.91 tok/s | 1,146 tok/s |
| NVFP4 | 24.18 GiB | **0.163 s** | 8.41 tok/s | **1,995 tok/s** |
| NVFP4 + MTP (3 tokens) | 24.97 GiB | 0.398 s | **15.14 tok/s** | 1,834 tok/s |

NVFP4 was 2.15 times faster than BF16 for decode. Adding MTP made NVFP4 1.80
times faster again, or 3.87 times faster than BF16 overall. During the MTP
benchmark, 495 of 828 drafted tokens were accepted (59.8%). MTP therefore
clearly benefits sustained generation on this GB10, while plain NVFP4 has the
best latency for short responses and the best measured prefill throughput.

NVFP4 used the native `FlashInferCutlassNvFp4LinearKernel`; it did not fall
back to BF16 execution.

## DGX Spark tuning sweep

A subsequent single-sequence sweep tested the main speculative-decoding and
scheduler controls. All decode results are medians of three 256-token runs at
temperature 0.7 unless noted otherwise.

| Configuration | Median decode | Result |
| --- | ---: | --- |
| MTP 2, 4,096 scheduled tokens | 15.45 tok/s | Slightly slower |
| MTP 3, 4,096 scheduled tokens | **16.04 tok/s** | Best validated setting |
| MTP 4, 4,096 scheduled tokens | 14.13 tok/s | Verification overhead wins |
| MTP 3, 8,192 scheduled tokens | 15.24 tok/s | No single-sequence benefit |
| MTP 3, 8,192 tokens, temperature 0 | 15.41 tok/s | More consistent, not faster |
| MTP 3, text-only mode | 15.38 tok/s | No decode benefit; slower prefill |

The winning run measured 16.39, 16.04, and 15.73 tok/s. Disabling chunked
prefill was rejected by vLLM because a non-chunked scheduler must set
`max_num_batched_tokens` at least as high as the 65,536-token maximum context.
NVFP4 KV cache is present in this vLLM build but its FlashInfer backend is
restricted to the SM100 family; the GB10 is SM121, so FP8 remains the supported
KV-cache format here.

## SparkBench quick-suite concurrency profile

A later cached-only run exercised the reproducible `quick.toml` suite with the
`qwen38-27b-nvfp4-mtp3-throughput` profile. It used the exact NVFP4 revision in
`manifests/models.toml`, MTP depth 3, FP8 KV cache, a 32,768-token served
context, eight sequence slots, 8,192 scheduled tokens, and temperature zero.
Thinking was disabled. Each case followed one warm-up request; decode and
prefill used three measured repetitions, while each concurrency level used two
measured bursts. The sample counts are too small for p95 claims.

| Workload | Measured requests | Tokens/request (prompt → output) | Median TTFT | Median E2E | Aggregate output | Median client decode estimate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single stream | 3 | 79 → 128 | 0.366 s | 7.556 s | 17.27 tok/s | 17.68 tok/s |
| Concurrency 2 | 4 | 79 → 64 | 0.609 s | 4.474 s | 27.90 tok/s | 16.58 tok/s |
| Concurrency 4 | 8 | 79 → 64 | 0.561 s | 4.473 s | **53.93 tok/s** | 16.47 tok/s |

The concurrency figures use total completed output tokens divided by measured
case wall time, including minor harness time between bursts. Four-way serving
delivered 1.93 times the concurrency-two aggregate throughput with essentially
unchanged median E2E latency (4.4731 versus 4.4738 seconds). The streaming server
bundled multiple tokens in some SSE emissions, so the per-request decode column
is explicitly an estimate; use aggregate output throughput as the primary
concurrency result.

| Actual prompt tokens | Median TTFT | Client-TTFT prefill approximation |
| ---: | ---: | ---: |
| 324 | 0.292 s | 1,111 tok/s |
| 2,117 | 0.992 s | 2,134 tok/s |
| 8,261 | 4.311 s | 1,916 tok/s |

The separate long-context check admitted 8,284 prompt tokens, returned the
hidden key correctly, and reached first output in 4.444 seconds. All seven suite
cases completed without validation failures. This single probe does not validate
the full 32K served context or the checkpoint's 262K native limit.

Cached-only process startup took 453.44 seconds, including 173.21 seconds for
model loading plus compilation, graph setup, and FP4 autotuning. vLLM reported
24.97 GiB of model memory and 64.97 GiB available for KV cache: 1,139,598 tokens,
or a theoretical 34.78 concurrent 32K requests before scheduler and workload
limits. Minimum system-available memory during measured cases was 13.02 GiB.
Startup peaked at 100.64 W and 81 °C in sampled GPU telemetry. The compiled and
autotuned artifacts were persisted, but this run did not measure a subsequent
warm-start time. This remains an exploratory serving profile. vLLM also warned
that FP8 KV-cache scales defaulted to 1.0 without calibration, so the run
validates performance and its one retrieval probe—not broad model accuracy.

## SparkBench Protocol and Evidence Publication

SparkBench extends the focused experiment above to multiple models and runtime
families. Every managed run freezes its model profile, suite, artifact pins,
runtime configuration, hardware identity, and harness revision before serving
starts. The frozen plan is immutable for the life of the run; interrupted work
is resumed from that plan instead of being silently regenerated.

### Execution rules

- Run one inference configuration at a time. Refuse unrelated GPU compute and
  container workloads rather than stopping them implicitly.
- Resolve pinned cached artifacts before measurement. Network acquisition is a
  separate `fetch` step or requires an explicit `--allow-download` flag.
- Bind managed inference endpoints to loopback. Preserve per-run authentication
  and verify cleanup using process or container identity.
- Warm up before measured repetitions and use unique prefill prompts so prefix
  caching cannot inflate fresh-prompt results.
- Record prompt tokens, output tokens, TTFT, end-to-end time, aggregate output
  throughput, runtime-native counters when available, validation state, and
  sampled telemetry with explicit units.
- Preserve failures, partial results, early stops, and unsupported admissions.
  Do not turn successful transport or accelerated invalid emissions into a
  semantic-quality claim.

Aggregate output throughput is completed output tokens divided by measured case
wall time and is the primary cross-request decode metric. Per-request client
decode rates are secondary when a server can bundle multiple tokens into one
stream event. Client-TTFT prefill is an approximation unless the runtime reports
an isolated prompt-evaluation duration.

### Agentic tool-use protocol

The `agentic-tools` suite is a bounded admission gate for multi-turn function
calling. It covers tool selection with distractors, correct no-tool abstention,
two dependent calls, and recovery from one typed transient tool error. Each
scenario has three deterministic variants. Tool ordering varies by variant,
tools execute only through an in-process allowlist, and model-provided calls are
schema-checked before dispatch.

An episode runs at temperature zero with automatic tool choice, one active
request, a maximum of six model turns, and up to 4,096 completion tokens per
turn. Server slot geometry remains profile-specific and must match for paired
performance claims.
The frozen context admission estimate includes all six output budgets plus tool
history overhead. Agentic cases do not run concurrently and do not use a
per-case warm-up.

Report two outcomes separately:

- **strict task success** requires the declared call sequence and exact
  argument values, successful dependency or error-recovery behavior, and a
  final answer accepted by the bounded `FINAL:` envelope grammar before either
  limit;
- **tool-trace correctness** validates selection, abstention, arguments,
  ordering, dependency, and recovery without waiving the final envelope-format
  requirement.

Strict success is the primary deployment result. Trace correctness is a
diagnostic, not an alternate pass criterion. Rank matched configurations by
success first, then turns, malformed or unknown calls, recovery rate, episode
wall time, and sampled energy per strict solve. MTP comparisons additionally
require runtime-native proof of draft activity and the same scenario variants,
budgets, runtime, main artifact, and serving geometry.

Only scalar episode outcomes may enter journals or exported evidence. Scenario
text, response content, reasoning, tool arguments and responses, call
identifiers, and per-request tags remain excluded. The first complete campaign
and its comparison limits are recorded in
[the 2026-08-17 agentic tool-use report](docs/agentic-tools-results-2026-08-17.md).

### Harbor terminal coding-agent campaign

The Qwen3-Coder-Next Harbor campaign is a pre-run, paired comparison of two
coding-agent clients against one locally served model. Its normative definition
is
[`manifests/campaigns/harbor_terminal_coder_next.toml`](manifests/campaigns/harbor_terminal_coder_next.toml).
Do not infer a result from the presence of the manifest or this protocol.
The corrected campaign ID is
`qwen3-coder-next-harbor-terminal-offline-2026-08-18`. Earlier trials under the
2026-08-17 ID are diagnostic only: their verifier upload was not traversable
under the capability-dropped container, so they must not be repaired, regraded,
or combined with a fresh run.

The fixed serving profile is
`qwen3-coder-next-80b-a3b-ud-q4-k-xl-llamacpp`: the exact 49,608,478,720-byte
Unsloth `Qwen3-Coder-Next-UD-Q4_K_XL.gguf` artifact, llama.cpp b10453 source
revision and server-binary digest recorded in `manifests/models.toml`, one
sequence, 65,536 allocated context tokens, an 8,192-token server output cap,
Q8 key/value cache, full GPU offload, flash attention, and no speculative
decoder. The server defaults to temperature 1.0, top-p 0.95, and top-k 40.
Agent clients may send their own generation settings; the bridge does not
rewrite requests. Results therefore describe the complete model-plus-client
stack, not a sampling-controlled comparison of agent prompts alone.

The remaining inputs are pinned in the campaign manifest:

- Harbor 0.21.0 at revision
  `64afbbcb62165950301e1a6407c729aa26d844ff`, executed from the manifest-pinned
  read-only runtime tree that includes its CPython interpreter, virtual
  environment, installed packages, and source;
- Terminal-Bench 2.1 at revision
  `7131e4375048a0e408a8fb404b5f499d726b695b`;
- Qwen Code 0.21.13 and OpenCode 1.18.18, including their npm integrity and
  shasum values, the actual `opencode-linux-arm64` executable package integrity,
  and upstream source revisions; and
- six tasks whose agent and verifier phases are runtime-offline:
  `fix-git`, `cancel-async-tasks`, `fix-code-vulnerability`, `regex-log`,
  `polyglot-c-py`, and `query-optimize`.

The measured lifecycle performs no npm or NVM installation. A separate,
credential-free bootstrap produced normalized read-only Node, Qwen Code, and
OpenCode trees from the exact published distributions. The manifest pins the
complete tree digests and byte counts, the Node executable digest, package
integrities, source revisions, and the ARM64 OpenCode package. Before every
trial, the lifecycle hashes every mounted entry through no-follow file
descriptors and rejects path substitution, unsafe links, hardlinks, special
files, ownership or mode drift, or a changed byte. Custom Harbor agent classes
replace the stock network installers with the admitted read-only prefixes and
verify their versions; they never invoke a downloader or package manager. After
OpenCode agent execution, its custom cleanup is limited to deleting the
ephemeral `xdg-data` and `xdg-state` trees. The retained `opencode.txt` remains
the OpenCode trajectory and metric source. The complete Harbor runtime is
admitted the same way, and commands execute only its verified entry point.

Each task-agent pair runs once, with one active trial, no retry, and a 900-second
agent timeout. The containing Harbor invocation has a separate 3,600-second
wall ceiling covering native image build, agent setup, agent execution, and
verification. The twelve trials use the manifest's fixed counterbalanced order:
the starting client alternates across tasks so simple warmup or time-order drift
does not consistently favor one client. Harbor must build each task image for
the native ARM64 host instead of pulling an AMD64-only prebuilt image. The
adapter retains each exact built image ID. Pair equivalence is not inferred from
ID equality; it is defined by the campaign's bounded semantic runtime
fingerprint over Linux/ARM64, RootFS layer digests, and runtime `Config`,
excluding the non-runtime `Image` and `Labels` fields. A Qwen Code/OpenCode task
pair is not a valid comparison unless those fingerprints match. A failure or
timeout remains a measured failed attempt; it is not silently retried or
replaced. This small, selected task panel is an exploratory admission screen,
not a broad coding-quality claim.

#### Execution lifecycle

The outer orchestrator must acquire `hold_campaign_lock(workspace)` before it
starts llama.cpp and retain that single repository lock across the model,
authenticated Unix-socket bridge, every Harbor invocation, and teardown. While
holding the lock, it creates the verified derived dataset and runtime overlay in
an external owner-private cache, follows the manifest's exact `trial_order`,
builds each command with `build_harbor_invocation(...)`, and executes it through
`run_harbor_invocation(...)`. The generated Harbor command fixes Docker
execution, native image building and deletion, one attempt, one concurrent
agent, one trial, zero retries, the selected task, exact custom agent class,
served model, frozen tool prefixes, and phase-specific network policy. Do not
hand-edit that command.

After each invocation, project the external raw job with the adapter's strict
loader and canonical JSON serialization. The fingerprint-bearing campaign
summary and its outer lifecycle envelope both use schema version 2; version 1
records are intentionally incompatible. Derived tasks and raw Harbor jobs must
resolve outside the repository; only the later allowlisted scalar projection is
eligible for the evidence exporter. Cleanup of Harbor containers, bridge,
server, sampler, the derived task copy, and the key file belongs in the
lock-owning `finally` path. The owner-private raw job tree remains ignored local
evidence and must pass an exact ephemeral-key residue scan before cleanup is
certified. An outer convenience CLI must preserve this lifecycle and the frozen
manifest rather than creating a second execution contract.

After the harness commit is clean and every admission input is present, run the
frozen lifecycle with:

```bash
python3 harbor_campaign.py
```

Its defaults resolve the pinned Harbor runtime, tool prefixes, Terminal-Bench
checkout, and owner-private raw/output root outside the repository. Optional
path flags relocate only those exact-verified inputs; they do not change the
model, task order, bridge endpoint, sampling geometry, or lifecycle contract.

#### Isolation and credential boundary

The inference server remains bound to `127.0.0.1`. Its authenticated host bridge
listens only on an owner-private mode-0600 Unix socket and forwards only to that
loopback server. A dedicated, read-only Node relay shares Harbor's egress
sidecar network namespace, listens only on container loopback, and is the sole
container with the socket/key directory mounted. The untrusted task receives a
fixed non-secret placeholder; the relay validates it, substitutes a per-run
internal bearer, and the host bridge validates and strips that bearer before
connecting upstream. The real credential never enters Harbor arguments,
environment, task files, or published evidence. Neither boundary logs headers
or payloads, and both enforce bounded connections, headers, buffers, and
timeouts. Delete the owner-only key and prove the Unix socket is absent after
cleanup, including interruption paths.

Task setup begins with `no-network` because the admitted clients require no
installation. During the agent phase, one atomically updated, permanent
default-drop nftables chain permits only IPv4 loopback TCP to the fixed relay
port; DNS, ICMP, IPv6, raw sockets, the Docker gateway, public addresses, and
all other loopback ports remain blocked. The verifier phase atomically returns
to deny-all, preventing surviving agent children from regaining egress.
Embedded probes certify these transitions in every invocation. The adapter
verifies every byte not deliberately transformed against the pinned source.
Derivation applies the fixed phase policy to task metadata, pins each mutable
base-image tag to one Linux/ARM64 digest, and appends a dedicated Python verifier
environment. It narrowly removes the upstream online `apt`/`curl`/`pip`/`uvx`
bootstrap from each `tests/test.sh`, points the same pytest invocation at the
preinstalled environment, and retains the task assertions and reward logic.
Both the derived `tests/` directory and `test.sh` are mode `0555`: Harbor
directly executes that verifier under `cap_drop: ALL`, so the uploaded copy must
be traversable and executable without adding capabilities. Each final task
image reserves `/tests` as UID/GID 65532 mode `0555`; admission requires the
pinned Compose copy path to populate that foreign-owned directory, and a
fallback upload failure stops the canary. The deterministic
patch digest binds the source and derived Dockerfile, task metadata, test
launcher bytes, and every source/derived mode.

The verifier packages are available without runtime networking, but the task
images are still built through the ordinary Docker builder. Direct verifier
package versions and base-image digests are pinned; transitive Python artifacts
are not hash-locked. Exact semantic image fingerprints therefore establish a
matched pair within this run, not byte-for-byte rebuild determinism. Harbor's
shared verifier also remains a task-harness trust model, not a tamper-resistant
anti-cheat boundary: the verifier runtime is visible to the root agent before
the tests are uploaded. Keep that limitation attached to any result. The model
is never bound to a wildcard or LAN address, and neither task nor inference uses
Docker host networking.

The network-policy, Dockerfile, verifier-bootstrap, and mode transformations
intentionally differ from upstream Terminal-Bench 2.1. Report the result as a
Harbor/Terminal-Bench-derived harness-stack outcome, not an official
Terminal-Bench 2.1 score.

#### Admission and stop gates

Before the measured matrix, require all of the following:

1. exact model, runtime, Harbor, dataset, and agent pins resolve, and the model
   file and runtime binary match their recorded sizes and digests;
2. no unrelated GPU process or running container is present, port 8000 is free,
   and available unified memory is at least the profile's 96 GiB estimate plus
   an 8 GiB reserve;
3. the loopback model passes basic chat, structured-output, and tool-call
   admission, including one valid tool call;
4. the full Harbor runtime and all Node/agent prefix trees match their complete
   immutable admissions, and the relay image is native ARM64;
5. the exact derived dataset is deterministic and failure injection leaves no
   partial tree; one Python 3.13, one Python 3.11, and one Ubuntu-derived image
   certify their pinned verifier runtime under `cap_drop: ALL` and verifier
   deny-all;
6. the public oracle solution earns reward `1` once for every selected task
   through the exact derived Harbor path, with `query-optimize` repeated once
   to screen its timing threshold, and every canary image/container cleans up;
   and
7. authenticated relay/bridge access succeeds, invalid access never reaches
   the model, every forbidden network probe fails, the phase-policy and relay
   assets match their digests, and no raw-payload publication path is enabled.

Abort rather than reinterpret a run when model readiness exceeds its
1,200-second profile timeout, the canary cannot finish within its 3,600-second
containing invocation ceiling and clean up, available memory falls below the
admission reserve, swap grows without
recovering, or bridge/network isolation fails. Stop the campaign after two
consecutive endpoint or chat-template failures. Do not start a new trial at the
23,400-second campaign cutoff; preserve the remaining 5,400 seconds of the
eight-hour window for cleanup, reconciliation, deterministic evidence export,
verification, and documentation. Preserve all completed and failed attempts
when a stop gate fires.

#### Records and publication

Harbor job results, task workspaces, trajectories, prompts, completions,
reasoning, tool payloads, logs, identifiers, commands, environment state, local
paths, and the ephemeral credential remain raw local records under ignored
storage. They must never be copied into Git or quoted in a report. The campaign
adapter may project only its strict allowlist of scalar outcomes and public,
bounded provenance: task and agent labels, terminal status, verifier reward,
token counts, durations, timestamps, version/digest pins, policy-patch digest,
and cleanup/admission booleans. Unknown fields fail closed.

Publication follows [the sanitized evidence workflow](#publishing-sanitized-evidence).
Add campaign evidence only after the exporter supports its exact scalar schema,
an offline synthetic fixture passes, two exports are byte-identical, the archive
verifies, and `verify-evidence --staged` validates and secret-scans the exact Git
index. A report must retain failures and partial states and must label the
one-attempt design, task subset, client-controlled requests, derived network
policy and verifier transformation, shared-verifier trust, non-hash-locked
build dependencies, serving geometry, and absence of a broad quality or
official leaderboard claim.

Concurrency results are comparable only when the serving-slot geometry is the
same. A one-slot profile receiving C2, C4, or C8 requests measures queued
aggregate service, not parallel-sequence scaling. Similarly, compare
perplexity only for the same base model, tokenizer, dataset hash, runtime,
chunk count, and context size. Exact revisions, image digests, artifact hashes,
hardware, date, and validation state accompany publishable conclusions.

### Raw run records

The complete local source of truth lives under ignored `results/` paths. A
managed run can include `plan.json`, an append-only event journal, telemetry,
server provenance and logs, generated summaries, and cleanup evidence. Matrix,
perplexity, direct-adapter, llama-bench, NInfer, and content-battery campaigns
have their own bounded source layouts.

Raw records are intentionally not committed. They can contain captured prompts
or completions, reasoning, tool calls, request identifiers, process details,
host paths, raw media, logs, or ephemeral credentials. `data/` and `logs/` are
also local-only because they hold weights, caches, media, and runtime output.
Exact raw run IDs may be cited in a report for local traceability, but the path
name is not itself public evidence.

### Publishing sanitized evidence

An evidence export creates a deterministic tracked archive without copying raw
records:

```bash
python3 sparkbench.py export-evidence \
  --results results --output evidence --replace
python3 sparkbench.py verify-evidence evidence
# After staging the intended commit:
python3 sparkbench.py verify-evidence evidence --staged
```

The intended archive entry points are `evidence/README.md` for people and
`evidence/index.json` for tools. Run bundles retain scalar request measurements,
case aggregates, validation booleans and bounded categories, lifecycle state,
compact numeric telemetry, and reproducibility pins such as artifact hashes,
runtime revisions, image digests, hardware, and harness revision. Campaign and
matrix bundles retain only their explicitly supported scalar schemas.

The exporter must fail closed. Unknown fields or schema versions, malformed or
non-finite numbers, duplicate JSON keys, unsafe file types or links, unexpected
source files, unsafe output placement, and configured size limits are errors.
Every bundle and the archive root carry checksums, and verification recomputes
those checksums while cross-checking index counts and references.

The archive excludes all captured input and output text, reasoning text, tool
arguments or responses, transcriptions, request or sample tags, raw identifiers,
local paths, commands, environment variables, logs, media, model weights,
caches, and credentials. String fields that remain are allowlisted bounded
labels, public model/runtime identifiers, status values, units, hashes, and
other non-content provenance.

Before committing a refresh:

1. Stop writes to the selected raw run corpus and let the exporter acquire the
   benchmark lock.
2. Export twice and confirm the second pass is unchanged.
3. Run `verify-evidence` against the finished archive.
4. Inspect the Git diff and staged file list; confirm that only documentation,
   code, tests, manifests, patches, and sanitized `evidence/` files are staged.
5. Run `python3 sparkbench.py verify-evidence evidence --staged` to reconstruct
   and validate the exact Git-index evidence tree and secret-scan every staged
   text blob.
6. Run the repository tests before committing or pushing.

Never hand-copy a raw result into `evidence/`, loosen an allowlist merely to make
an export pass, or publish a number without its status and comparison geometry.
When a legitimate schema evolves, update the exporter, add an offline regression
fixture, regenerate the archive, and document any conclusion that changed.
