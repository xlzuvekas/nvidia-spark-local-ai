# Agentic tool-use results — 2026-08-17

## Result summary

The exploratory deterministic multi-turn tool campaign completed **96
episodes** on one DGX Spark: four scenarios, three variants per scenario, and
eight matched or candidate-serving configurations.

The frozen plans recorded an uncommitted implementation, so these measurements
are preserved as exploratory evidence rather than a fully reproducible release
result. The harness is being committed before a clean-revision replication; the
replication, not this pilot alone, is the publication-grade provenance gate.

- **All 96/96 tool traces were correct.** This includes correct tool
  abstention, selection and arguments, dependent two-hop execution, and all
  24/24 injected transient-error recoveries.
- **62/96 episodes passed the stricter end-to-end oracle.** The other 34 had a
  correct tool trace but did not emit the required one-line `FINAL:` envelope.
  There were no wrong-envelope answers, malformed calls, unknown calls, turn
  limit hits, or output-length terminations.
- **Laguna XS 2.1 was the fastest fully passing configuration:** 12/12 strict
  passes in 13.54 seconds of summed episode wall time. Its energy coverage was
  incomplete for the shortest no-tool case, so no whole-panel joules-per-solve
  value is reported.
- **Laguna S 2.1 also passed 12/12.** Its 34.66-second task total was 1.90× as
  fast as dense Qwen3.8 without MTP and 1.07× as fast as Qwen3.8 with
  MTP4, despite the Laguna artifact containing 118B total / 8B active
  parameters and 68.35 GiB of verified weights.
- **Qwen3.8 MTP4 preserved 12/12 strict success** while cutting matched summed
  episode wall time by 43.6% and sampled case energy by 33.9%.
- **Qwen3.6 and Nemotron expose an envelope-format distinction.** Their tool
  traces were 12/12 both with and without MTP, but Qwen3.6 passed only 7/12
  strict envelopes and Nemotron passed 0/12. Nemotron MTP3 had the lowest raw
  task total, 12.48 seconds, but it is not a passing end-to-end configuration.

These are small synthetic admission tasks, not broad agent benchmarks. The
strict result remains the primary deployment result; trace correctness is a
diagnostic that explains why a strict episode failed.

## Overall outcomes

The table ranks fully passing configurations first. “Task wall” is the sum of
the 12 measured episode wall times and excludes artifact verification, model
loading, the post-start prime request, and shutdown. Energy is the sum of
case-phase sampled GPU energy; it is secondary because sampling is coarse for
these short episodes.

| Configuration | Strict success | Correct traces | Missing `FINAL:` | Task wall | Median episode | Sampled case energy (coverage) | Sampled J / strict solve |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Laguna XS 2.1 33B-A3B Q4_K_M | **12/12** | 12/12 | 0 | **13.54 s** | **1.064 s** | 466 J (3/4 cases) | — |
| Laguna S 2.1 118B-A8B UD-Q4_K_XL | **12/12** | 12/12 | 0 | 34.66 s | 3.157 s | 1,608 J (4/4) | 134.0 J |
| Qwen3.8 27B UD-Q4_K_XL + MTP4 | **12/12** | 12/12 | 0 | 37.18 s | 3.542 s | 2,057 J (4/4) | 171.5 J |
| Qwen3.8 27B UD-Q4_K_XL | **12/12** | 12/12 | 0 | 65.95 s | 6.175 s | 3,112 J (4/4) | 259.3 J |
| Qwen3.6 35B-A3B UD-Q4_K_XL + MTP2 | 7/12 | 12/12 | 5 | 15.77 s | 1.513 s | 525 J (4/4) | 75.0 J |
| Qwen3.6 35B-A3B UD-Q4_K_XL | 7/12 | 12/12 | 5 | 17.73 s | 1.645 s | 580 J (3/4) | — |
| Nemotron 3.5 Lightning 30B-A3B Q4_0 + MTP3 | 0/12 | 12/12 | 12 | 12.48 s | 1.155 s | 412 J (3/4) | — |
| Nemotron 3.5 Lightning 30B-A3B Q4_0 | 0/12 | 12/12 | 12 | 14.48 s | 1.370 s | 490 J (4/4) | — |

Nemotron baseline's complete four-case energy coverage corresponds to 40.8 J
per correct trace. The MTP3 run lacks energy for one case, so it has no
comparable whole-panel value. Trace energy describes tool mechanics, not
strictly solved episodes.

## Strict envelope versus tool trace

Each cell below is the strict pass count out of three variants. Every
corresponding scenario/configuration trace was 3/3, including no-tool
abstention. Therefore every non-pass in this table is specifically a missing
strict final envelope rather than a bad tool choice, bad argument, failed
dependency, or failed retry.

| Configuration | Select and call | No tool | Two hop | Error recovery | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laguna XS 2.1 | 3/3 | 3/3 | 3/3 | 3/3 | **12/12** |
| Laguna S 2.1 | 3/3 | 3/3 | 3/3 | 3/3 | **12/12** |
| Qwen3.8 | 3/3 | 3/3 | 3/3 | 3/3 | **12/12** |
| Qwen3.8 + MTP4 | 3/3 | 3/3 | 3/3 | 3/3 | **12/12** |
| Qwen3.6 | 0/3 | 3/3 | 3/3 | 1/3 | 7/12 |
| Qwen3.6 + MTP2 | 0/3 | 3/3 | 3/3 | 1/3 | 7/12 |
| Nemotron 3.5 Lightning | 0/3 | 0/3 | 0/3 | 0/3 | 0/12 |
| Nemotron 3.5 Lightning + MTP3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/12 |

The strict validator recognizes one bounded final-answer form. It intentionally
does not inspect or publish response text. A `missing_final` scalar therefore
does not establish whether prose outside the envelope was semantically
equivalent; it establishes that the serving response did not satisfy the
declared API contract. Formatter or chat-template work should be retested
against the same oracle rather than retroactively treating these episodes as
passes.

## MTP: quality preservation and task latency

All three speculative pairs used the same main artifact and suite geometry as
their non-MTP control. MTP preserved the exact strict/trace outcome pattern in
every pair. Runtime-native cumulative counters proved draft activity for the
one server lifetime in each accelerated run.

| Matched pair | Strict / trace, base → MTP | Accepted draft tokens | Mean accepted length | Task wall, base → MTP | Wall reduction | Task-rate gain | Common-case sampled energy reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6, MTP2 | 7/12 / 12/12 → same | 417/424 (98.35%) | 2.967 | 17.73 → 15.77 s | 11.0% | 12.4% | 18.0% (3 matched cases) |
| Nemotron, MTP3 | 0/12 / 12/12 → same | 401/437 (91.76%) | 3.747 | 14.48 → 12.48 s | 13.9% | 16.1% | 5.8% (3 matched cases) |
| Qwen3.8, MTP4 | 12/12 / 12/12 → same | 439/483 (90.89%) | 4.628 | 65.95 → 37.18 s | **43.6%** | **77.4%** | **33.9%** |

The acceptance counters cover the complete measured server lifetime, including
the prime request, while task wall covers only the 12 episodes. The large
Qwen3.8 gain is consistent with speculative decoding reducing the dense
model's decode cost. The smaller sparse-model gains are still positive, but
short tool episodes retain fixed TTFT and round-trip overhead.

## Protocol

The schema-versioned `agentic-tools` suite contains four deterministic
scenarios:

1. Select the correct tool among distractors and call it with exact arguments.
2. Abstain when no tool is needed.
3. Perform two sequential calls where the second depends on the first result.
4. Recover from one injected typed transient error by retrying the same call.

Each scenario has three data variants and rotates the tool ordering. Runs use
temperature zero, `tool_choice=auto`, one active episode, no case warm-up, and
no parallel episode concurrency. Each model turn has a 4,096-token completion
budget and each episode is capped at six model turns. The conservative context
admission estimate is 26,624 tokens, below every tested served context.

The in-process executor recognizes only the suite's bounded scalar tools.
Model-provided call identifiers are replaced with episode-local identifiers,
and arguments are parsed and schema-checked before dispatch. The journal keeps
only aggregate scalar results. It does not retain the scenario text, response
content, reasoning, arguments, tool responses, or per-request identifiers.

Strict success requires all of the following:

- the exact scenario-specific tool sequence and validated arguments;
- the expected dependency or injected-error recovery behavior;
- one final answer accepted by the declared bounded `FINAL:` envelope grammar; and
- completion before the turn and output limits.

Across this campaign the maximum episode used three turns and 130 total
completion tokens. There were zero length-terminated turns and zero turn-limit
hits. The larger budget was therefore sufficient, but this campaign does not
identify the minimum safe budget for harder agentic work.

## Hardware, artifacts, and lifecycle

- Platform: one NVIDIA DGX Spark / GB10, driver 580.142, compute capability
  12.1, and 119.694 GiB system-visible unified memory.
- Backend: native llama.cpp subprocess and loopback-only endpoint. Qwen3.8 base
  and MTP4 retained their matched P8 server geometry; the other six profiles
  used P1. Only one episode was active in every measurement.
- Runtime revision: `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`.
- Runtime binary SHA-256:
  `ae1bd49f869ff3397b2a5d757fcf010c6eaaf16c4e3071a15861312defcd4e40`.
- Harness base revision recorded by every frozen plan:
  `c756d457732740f6b689053528fee4fbc5eee3ad`; every plan also recorded a dirty
  worktree because the agentic implementation had not yet been committed. The
  exact pilot-time source tree therefore cannot be reconstructed from that
  revision alone, even though the scalar artifacts and later hardened harness
  are retained here.
- Measurement window: 2026-08-17 23:14:49 through 23:28:29 UTC.

| Model artifact | Pinned source and revision | Quantization | Verified weight size |
| --- | --- | --- | ---: |
| Qwen3.6 35B-A3B | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF@5bc3e238d916f48a861bac2f8a1990a0e9b7e98d` | UD-Q4_K_XL | 22,853,663,008 B |
| Laguna XS 2.1 33B-A3B | `poolside/Laguna-XS-2.1-GGUF@1a37c0a5fb8c7a18e6106decb6be6327d1b63fa6` | Q4_K_M | 20,274,300,032 B |
| Nemotron 3.5 Lightning 30B-A3B | `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF@9d425fe18d84ab04da6aabb757d2e2807083d054` | Q4_0 | 18,898,091,584 B |
| Qwen3.8 27B | `unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe` | UD-Q4_K_XL | 17,923,394,624 B |
| Laguna S 2.1 118B-A8B | `unsloth/Laguna-S-2.1-GGUF@750f92f90cf54159c4d7a610cb7b3e74498e75c6` | UD-Q4_K_XL, three shards | 73,395,172,000 B |

Main-artifact SHA-256 values were, respectively,
`55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1`,
`1ac7079101fca5a6df8c5a7523a3c30ea7d1c0e4b1258090e7d6d4039287f6cb`,
`61f87e75974e4b535dcdf9aad056541a9514f1dfa4538b463b081d19b7a00e3c`,
and `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`.
Laguna S's three shard hashes were
`0cfaf46917260d253773e5e2fab64329fa5c9c60fdf0db0f59f31205b5f5dd32`,
`2296102462b02edca70163121ac62bacf7a82078c0eafc91625c8822850769bf`,
and `9150e2338f7690af29685b6a2ca621a8fda7ecf9724678266c4b04b7c6dd0ef3`.
Nemotron MTP3 also used the pinned 1,155,907,520-byte draft artifact with
SHA-256
`19f964207d5236dc88662686f00604a5494974c23fb04dd16a5ad7b2eebbd5b4`.

The configurations ran sequentially. Every run had one server-ready event,
one verified server stop, four completed cases, no case exception, no run
abort, valid startup measurement state, and zero measurement annotations.
Runs with strict validation failures are terminal `partial` results rather than
infrastructure failures.

## Exact local result directories

These ignored raw directories are the source set used for this report. Their
strict scalar projections are also tracked under matching directories in
`evidence/runs/`; the exporter excludes prompts, completions, reasoning, tool
arguments and responses, request identifiers, paths, logs, and credentials.

- `results/20260817T231449Z-qwen36-35b-a3b-ud-q4-k-xl-llamacpp-agentic-tools-02b3617b`
- `results/20260817T231614Z-laguna-xs21-33b-a3b-q4-k-m-llamacpp-agentic-tools-c5ca5827`
- `results/20260817T231714Z-nemotron35-lightning-30b-a3b-q4-0-llamacpp-agentic-tools-07355fb3`
- `results/20260817T231833Z-qwen38-27b-ud-q4-k-xl-llamacpp-agentic-tools-0013f56f`
- `results/20260817T232029Z-qwen36-35b-a3b-ud-q4-k-xl-llamacpp-mtp2-agentic-tools-9c909ffe`
- `results/20260817T232123Z-nemotron35-lightning-30b-a3b-q4-0-llamacpp-mtp3-agentic-tools-465072aa`
- `results/20260817T232209Z-qwen38-27b-ud-q4-k-xl-llamacpp-mtp4-agentic-tools-38449200`
- `results/20260817T232458Z-laguna-s21-118b-a8b-ud-q4-k-xl-llamacpp-agentic-tools-24322be1`

## Limits and next step

- The battery is deliberately small and synthetic. It validates the declared
  tool-loop contract, not general API conformance, coding, browsing, planning,
  long-horizon
  memory, or real environment interaction.
- Cross-model wall and energy comparisons combine different model output
  lengths and quantizations. Matched MTP deltas are stronger evidence than
  unlike-model rankings.
- Case-energy values are sampled intervals, not whole-run energy. The shortest
  configurations have only 12–17 total case-phase samples.
- Baseline/MTP ordering was not counterbalanced. MTP activity is proven by
  native counters, but the smaller Qwen3.6 and Nemotron deltas should be
  confirmed with reversed ordering if used as precise performance claims.
- All results used one active episode, but Qwen3.8 retained P8 server geometry
  while the other profiles used P1. Unlike-model latency rankings therefore
  mix slot geometry; matched Qwen3.8 base/MTP4 comparisons do not. These runs
  do not establish equivalent parser behavior or performance in vLLM, SGLang,
  Ollama, NInfer, or concurrent serving.
- Because the frozen plans recorded a dirty worktree, this pilot is explicitly
  exploratory. Committing the later hardened harness does not retroactively
  make those plans clean; a same-matrix clean-revision replication is required
  for the reproducible result.

The next quality tier should keep this battery as a fast admission gate, then
run the fully passing configurations through a pinned offline function-calling
benchmark and a small containerized terminal task set. Qwen3.6 and Nemotron
should first receive a bounded formatter/chat-template investigation using the
same strict oracle.
