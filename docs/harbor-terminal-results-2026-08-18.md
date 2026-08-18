# Qwen3-Coder-Next Harbor terminal results — 2026-08-18

## Result summary

The corrected two-replicate campaign produced **1 strict pass in 24 trials
(4.1667%)** on one NVIDIA DGX Spark. Qwen Code passed **1/12 (8.3333%)** and
OpenCode passed **0/12**. The sole pass was Qwen Code on `fix-git` in the first
replicate; the same model-client-task combination scored zero in the second
replicate, so the success did not replicate.

The panel was fixed in advance: six selected Terminal-Bench 2.1 tasks, two
agent clients, and two complete replicates. Every task-agent pair ran once per
replicate, with no retry. This is a Harbor/Terminal-Bench-derived harness-stack
result, **not an official Terminal-Bench 2.1 score** and not a broad coding or
model-quality score.

| Agent client | Strict passes | Failed trials | Token telemetry present |
| --- | ---: | ---: | ---: |
| Qwen Code 0.21.13 | **1/12** | 11/12 | 12/12 |
| OpenCode 1.18.18 | **0/12** | 12/12 | 5/12 |
| **Combined** | **1/24** | **23/24** | **17/24** |

All 24 trials finalized. The poor task score is separate from the campaign's
infrastructure result: both replicates completed, and every recorded
network-isolation, native-image, image-pair, and cleanup gate passed.

## Outcomes by task and replicate

Only `fix-git` earned reward `1`, once. Each of the remaining 23 trial cells
earned reward `0`.

| Selected task | Qwen Code, two replicates | OpenCode, two replicates | Combined |
| --- | ---: | ---: | ---: |
| `fix-git` | **1/2** | 0/2 | **1/4** |
| `cancel-async-tasks` | 0/2 | 0/2 | 0/4 |
| `fix-code-vulnerability` | 0/2 | 0/2 | 0/4 |
| `regex-log` | 0/2 | 0/2 | 0/4 |
| `polyglot-c-py` | 0/2 | 0/2 | 0/4 |
| `query-optimize` | 0/2 | 0/2 | 0/4 |

| Replicate | Window (UTC) | Qwen Code | OpenCode | Combined | Summed trial wall | Lifecycle elapsed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 10:17:06–10:59:04 | 1/6 | 0/6 | **1/12** | 2,394.699 s | 2,518.201 s |
| 2 | 11:01:08–11:34:36 | 0/6 | 0/6 | **0/12** | 1,883.898 s | 2,007.910 s |
| **Combined** | — | **1/12** | **0/12** | **1/24** | **4,278.597 s** | **4,526.110 s** |

The scalar exception labels do not establish causes. Qwen Code recorded two
`UnknownApiError` trials, one `AgentTimeoutError`, and one
`NonZeroAgentExitCodeError`; seven other Qwen Code failures had no exception
label. OpenCode recorded seven `NonZeroAgentExitCodeError` trials and five
reward-zero trials without an exception label. Raw trajectories, prompts,
completions, reasoning, tool payloads, and free-form errors were deliberately
excluded from the tracked evidence and were not used to retrofit a diagnosis.

## Task failure versus infrastructure

Both lifecycle envelopes report `status=completed` and `stop_reason=completed`,
with 12/12 planned results present. Across the 24 trial projections:

- 24/24 admitted the agent relay, native ARM64 main image, native ARM64 relay
  image, capability drop, and matched task image semantic fingerprint;
- 24/24 rejected the recorded forbidden network paths, including public,
  gateway, DNS, and non-relay loopback probes;
- 24/24 report successful trial and built-image cleanup; and
- the campaign summaries record zero Harbor process failures, Harbor wrapper
  timeouts, network-admission failures, native-image admission failures,
  image-pair mismatches, cleanup failures, or built-image cleanup failures.

`harbor_timeouts=0` refers to the 3,600-second containing Harbor ceiling. One
Qwen Code `polyglot-c-py` trial in replicate 1 reached the separate 900-second
agent limit and is correctly retained as an `AgentTimeoutError`, reward-zero
attempt. That is an agent-level terminal outcome, not a contradiction of the
wrapper-timeout count.

The second `polyglot-c-py` Qwen Code attempt did not reproduce that timeout: it
ended after 560.734 seconds of agent execution with
`NonZeroAgentExitCodeError` and reward `0`. This observation is not enough to
attribute either failure to the model, client, task, or runtime.

## Token telemetry

Token counts are descriptive telemetry, not a fair client-efficiency
comparison. All 12 Qwen Code trials reported counts, while seven OpenCode
early-exit trials did not. Treating those missing values as zeros would make
OpenCode appear artificially cheap.

| Agent client | Measured trials | Input tokens including cache | Cache tokens | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| Qwen Code | 12/12 | 12,148,796 | 11,809,551 | 40,326 |
| OpenCode | 5/12 | 36,847 | 0 | 1,036 |
| **Observed total** | **17/24** | **12,185,643** | **11,809,551** | **41,362** |

The same selection effect makes the summed client wall times unsuitable as a
speed ranking: many OpenCode trials terminated early rather than solving the
task.

## Frozen stack and execution geometry

The normative protocol and ordering are in the
[campaign manifest](../manifests/campaigns/harbor_terminal_coder_next.toml) and
[Harbor campaign section](../BENCHMARK.md#harbor-terminal-coding-agent-campaign).
Both measured replicates recorded a clean harness revision
`26600d4abe48c082ce6764a61618516837069b9c`.

- Model profile:
  `qwen3-coder-next-80b-a3b-ud-q4-k-xl-llamacpp`, served one sequence at a
  65,536-token context with an 8,192-token output cap. The server defaults were
  temperature 1.0, top-p 0.95, and top-k 40; agent clients could submit their
  own generation settings.
- Model artifact:
  `unsloth/Qwen3-Coder-Next-GGUF@ce09c67b53bc8739eef83fe67b2f5d293c270632`,
  49,608,478,720 bytes, SHA-256
  `4bb93f0a0221ef4ff963ca9094df629c8dfdfabc3b4fdd85c1a2e4c0624fce36`.
- Runtime: llama.cpp revision
  `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`; server-binary SHA-256
  `ae1bd49f869ff3397b2a5d757fcf010c6eaaf16c4e3071a15861312defcd4e40`.
- Harness and task source: Harbor 0.21.0 at
  `64afbbcb62165950301e1a6407c729aa26d844ff`; Terminal-Bench 2.1 at
  `7131e4375048a0e408a8fb404b5f499d726b695b`.
- Agent clients: Qwen Code 0.21.13 at
  `d959015974302fb60ebd99adb81a68c2f482eaa3`; OpenCode 1.18.18 at
  `31406ccc51b4bd2a4e1e086b2bcaa5f7f804f26d`, including its pinned ARM64
  executable package.
- Derived-task/network-policy digest:
  `sha256:4749be56af707f6d7615ac5cdb0fb7fa8d50fcdd49e5d4c9a9bfebb71677b4ef`.

The model and both clients ran sequentially on one machine. There was no
speculative decoder, no concurrent trial, and no second quantization or model
control. Results therefore describe this one UD-Q4_K_XL
model-plus-client-plus-derived-harness configuration at temperature 1.0, not
the unquantized checkpoint or either client in general.

## Separate serving smoke result

A separate SparkBench smoke run of the same model artifact and llama.cpp
binary is tracked under the
[serving smoke bundle](../evidence/runs/20260818T070118Z-qwen3-coder-next-80b-a3b-ud-q4-k-xl-llamacpp-smoke-c05ac5fb/manifest.json).
It is not one of the 24 Harbor trials and used a different clean harness
revision, `a3b1a2ac5d39114d4b38b05639fd430344650ad9`.

That run is terminal `partial`: three supported cases completed, the chat and
tool validators passed, and the JSON validator failed. Vision, embeddings, and
reranking were declared unsupported and skipped. The first request and chat
case both ended at their configured output limits (`finish_reason=length`).
These scalars are an admission snapshot, not a coding-agent score. The Harbor
campaign later passed its own distinct chat, JSON, and tool-call admission
checks in both replicates; one validator's result should not be substituted for
the other.

## Evidence and limits

The tracked
[campaign manifest](../evidence/campaigns/qwen3-coder-next-harbor-terminal-offline-2026-08-18/manifest.json),
[replicate projections](../evidence/campaigns/qwen3-coder-next-harbor-terminal-offline-2026-08-18/replicates.json),
and
[checksums](../evidence/campaigns/qwen3-coder-next-harbor-terminal-offline-2026-08-18/checksums.json)
use schema `sparkbench-harbor-evidence-v1` and sanitization policy
`strict-scalar-allowlist-v1`. The manifest declares two UTC-ordered replicates,
`payloads_included=false`, and campaign status `complete`. The complete tracked
campaign bundle is indexed with SHA-256
`303cbc72ab8fea7505d6bc15e3ba32dc04fc32a17c2028fc45b76ac06fe0cd23`.

Interpretation remains deliberately narrow:

- six selected tasks and two replicates are too small for a broad coding-agent
  ranking or an estimate of either client's stable pass rate;
- the network policy, image pinning, and offline verifier transformation differ
  from upstream Terminal-Bench 2.1;
- client-controlled prompts and requests mean this is a complete stack
  comparison, not a sampling-matched comparison of prompt wrappers;
- the verifier environment is visible to the root agent before test upload, so
  the shared verifier is a task-harness trust model, not a tamper-resistant
  anti-cheat boundary;
- transitive verifier dependencies are not hash-locked, although paired task
  image semantic fingerprints matched within every trial pair; and
- one quantization, one machine, temperature 1.0, and two replicates do not
  establish behavior for other runtimes, sampling settings, checkpoints, or
  hardware.

The defensible conclusion is limited but useful: this exact Qwen3-Coder-Next
stack admitted and completed the isolated offline-derived campaign lifecycle,
but neither client was reliable on the selected tasks, and OpenCode frequently
exited before token telemetry was available.
