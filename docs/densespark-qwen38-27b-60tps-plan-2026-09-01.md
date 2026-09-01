# DenseSpark Qwen3.8 27B: 60 tok/s and agent-turn plan

Date: 2026-09-01
Hardware: one NVIDIA DGX Spark / GB10
Primary objective: the fastest safe single-user coding and cowork service that
retains the checkpoint-native 262,144-token context capability.

## Outcome definition

`60 tok/s` means all of the following, not one short peak:

- concurrency one;
- the pinned Frozenlock AutoRound INT4 checkpoint, DenseSpark PQ head, image,
  and synchronization-only warmup instrumentation;
- a server launched with `--max-model-len 262144` and enough admitted KV
  capacity for one complete native-context request;
- at least 60.0 median client-estimated decode tok/s and 60.0 aggregate output
  tok/s in the explicitly labeled no-thinking synthetic repetitive-code
  continuation ceiling;
- a warm D256 screen followed by a D1024 confirmation;
- two fresh server lifetimes for the retained configuration;
- every request length-terminated at the requested output count;
- exact tool, structured-output, and quality gates passing; and
- the existing startup, swap-growth, memory, device-error, and cleanup gates
  remaining clean.

The reasoning-enabled lane is reported separately. It is not required to reach
60 tok/s because reasoning-token entropy changes MTP acceptance, but its rate,
quality, and agent wall time remain first-class results.

The ceiling is not a claim about representative coding-agent throughput. A
separate divergent agent/tool lane must report end-to-end turn wall time,
usable answer tokens, tool validity, and task success before a configuration is
recommended for Codex or cowork use.

For long conversations, decode tok/s alone is insufficient. At 8K, 64K, 128K,
and a bounded near-native context, the report must also retain TTFT, complete
turn wall time, prompt-token count, cache-hit tokens, and request-scoped MTP
acceptance. Prefix caching may improve prefill and turn wall time; it cannot be
credited as a decode-speed improvement.

## The clue and the controlled diagnosis

The historical synchronization-only diagnostic reached a 61.814716 tok/s
median on three D256 numbered-phrase continuation requests from the preserved
original `benchmark.py` path. It was not a coding workload. Its server-lifetime
MTP counters showed 58.59% draft-token acceptance and a 5.6875 mean accepted
length. It was not an admitted result because the lifetime crossed the swap
policy and its receipt was manual. See
[the DenseSpark analysis](densespark-qwen38-27b-2026-08-31.md#sync-only-manual-diagnostic-non-publishable).

The first managed native-262K run instead measured 34.463345 median client
decode tok/s and 33.866536 aggregate output tok/s. Its counters showed 24.6758%
acceptance and a 2.9741 mean accepted length. All three 256-token completions
contained reasoning and no answer content. That made it a reasoning-trace
throughput test, not a like-for-like reproduction of the coding diagnostic.

A same-process controlled mode/workload matrix on the safe 0.70/262K server
then separated mode and workload. These values are local diagnostic
observations, not tracked evidence:

| Request lane | Median client decode | Median output | MTP acceptance | Mean accepted length |
|---|---:|---:|---:|---:|
| Unique numbered phrases, default thinking | 32.0140 tok/s | 31.3588 tok/s | 24.00% | 2.9202 |
| Unique numbered phrases, thinking off | 45.0083 tok/s | 44.3108 tok/s | 36.74% | 3.9394 |
| Synthetic code, thinking off | **70.5156 tok/s** | **67.1747 tok/s** | **67.42%** | **6.3934** |
| Synthetic code with textual `/no_think` suffix | 59.1230 tok/s | 56.9588 tok/s | 57.36% | 5.5887 |

A later fixed-length D1024 continuation made the mechanism clearer: all three
requests emitted exactly 1,024 visible tokens, median client decode reached
**79.3277 tok/s**, aggregate output reached **77.5127 tok/s**, draft acceptance
was **72.8056%**, and mean accepted length was **6.8244**. A fresh streaming
D1024 check on the same server reached **81.1029 decode tok/s** at 75.1712%
acceptance and 7.0137 mean accepted length. The fixture repeats nearly templated
pytest definitions, so these are deliberate MTP-friendly throughput ceilings,
not representative agent-task results.

The exact Responses path used by Codex was also verified. On three interleaved
D256 pairs, `reasoning.effort="none"` produced only visible answer content and
had median 69.8718% acceptance / 6.5897 mean accepted length. `low` retained
thinking, mixed reasoning with content, and measured 52.5000% / 5.2000. The
installed Codex and vLLM versions therefore expose a real no-thinking control;
the change is not a prompt-suffix artifact.

Four longer interleaved Responses pairs exposed the more important caveat.
On the repetitive D1024 continuation, `none` and `low` both decoded at roughly
77 wall tok/s and both reached about 73.5% acceptance / 6.88 mean accepted
length. The repeated code template eventually dominates the small thinking
prefix. `none` still moved median first visible code from roughly 1.97 seconds
to 0.25 seconds and emitted zero reasoning characters, but it did not make the
already predictable continuation's raw token execution faster.

The approximate causal accounting is unusually clean. At fixed MTP depth
eight, mean output tokens per speculative cycle are
`L = 1 + 8 * draft_acceptance`; draft acceptance and mean accepted length are
therefore two views of the same counters, not independent signals. The slow
managed lane implied `2.9741 / 34.4633 = 86.30 ms` per complete speculative
cycle. The repetitive D1024 code lane implied
`6.8244 / 79.3277 = 86.03 ms` per cycle. This cycle includes drafting, target
verification, and serving overhead; it is not a target-kernel timing.

Accepted length rose 2.2946x and TPS rose 2.3018x while the inferred cycle
cost differed by only 0.31%. Holding the slow cycle cost fixed predicts
`6.8244 / 0.08630 = 79.08 tok/s`, within 0.31% of the observed 79.33 tok/s. In
this fixed-K8 diagnostic, the measured speedup is therefore almost entirely
more accepted MTP tokens per expensive cycle, not a hidden CUDA-graph,
quantization, memory-utilization, or endpoint change. The slow counters are
lifetime-scoped while its TPS is a case median, so this is strong causal
diagnosis rather than request-matched publishable evidence.

The unchanged server produced both the slow and fast regimes. This rules out
0.70 GPU-memory utilization, native 262K capacity, or a generally degraded
server as the sole explanation. The dominant observed mechanism is workload-
and-mode-dependent MTP acceptance. An explicit request-layer thinking control
also beat a prompt suffix and is the configuration mechanism to retain.

## Brainstorm: divergent hypotheses

The initial pass deliberately considered independent levers before ranking
them:

1. hidden reasoning versus visible code changes token predictability;
2. the synthetic phrase fixture is adversarial to speculative acceptance;
3. cumulative Prometheus counters mix warmups and unrelated Codex requests;
4. MTP depth eight may be too deep in the low-acceptance lane or too shallow in
   the high-acceptance lane;
5. the PQ proposal head may miss important code-token candidates;
6. probabilistic and deterministic draft sampling may have different speed and
   acceptance tradeoffs;
7. native-262K scheduler geometry or a lower KV allocation may change decode
   graphs even for short prompts;
8. full CUDA graph replay may cover different target/verification shapes;
9. Humming/Marlin dispatch thresholds may select a poor batch-one kernel;
10. GDN, dense BF16 projections, the INT8 LM head, or target verification may
    dominate per-step wall time;
11. host memory pressure, thermal state, or background unified-memory traffic
    may cause apparent regressions;
12. automatic prefix caching may remove repeated prefills in agent loops; and
13. a client-side tool catalog may dominate prompt size even when the model is
    fast.

## Convergent ranking

| Rank | Hypothesis | Evidence now | Expected value | Cost/risk | Decision |
|---:|---|---|---|---|---|
| 1 | Thinking mode and output entropy control MTP acceptance | Same server moved from 32.0 to 70.5 tok/s and 24.0% to 67.4% acceptance | Very high | Low | Build explicit lanes and freeze them |
| 2 | MTP depth is lane-specific | Historical depth sweeps and acceptance-by-position make later proposals plausibly useful only in the fast lane | High | Medium; fresh startup per arm | Sweep after measurement repair |
| 3 | Cumulative counters are contaminating comparisons | Current summaries scope counters to the whole lifetime | High confidence, measurement value | Low | Add before/after deltas |
| 4 | Prefix caching dominates repeated agent-turn prefill | Consecutive Codex prompts were 143,814 and 147,884 tokens | Very high agent-wall value | High correctness risk on hybrid GDN | Quarantine behind semantic gates |
| 5 | CUDA graphs improve decode | FULL graphs are active; a different stack showed a small positive screen | Moderate | Medium startup cost | Verify state, then deprioritize unless coverage is wrong |
| 6 | 64K versus 262K or 0.86 versus 0.70 explains the old result | Both slow and fast lanes ran on the same 0.70/262K process | Low | Medium startup cost | Reject as primary cause; retain only confirmation |
| 7 | Kernel/quantization changes can add the remaining margin | Dense/GDN and projection work remains measurable | Moderate | High engineering and quality risk | Profile only after mode/MTP/cache work |
| 8 | Concurrency can create a 60 tok/s headline | It changes aggregate service throughput, not one user's serial rate | Irrelevant to objective | Low | Do not use it to claim success |

## Frozen controls

Every comparative decode arm keeps these fixed unless its name explicitly says
otherwise:

- one GB10 and one managed inference configuration at a time;
- loopback-only OpenAI-compatible endpoint;
- exact image ID, checkpoint revision, weight file inventory, PQ artifact hash,
  and compile-cache namespace;
- `--max-model-len 262144`, `--kv-cache-dtype auto`, BF16 Mamba state, 0.70 GPU
  memory utilization, 8,192 batched tokens, Humming linear backend, and
  FlashInfer GDN prefill;
- full-and-piecewise CUDA graphs with the hybrid-prefill guard;
- FCFS scheduling and concurrency one;
- prefix caching off for decode/MTP comparisons;
- temperature zero, stream enabled, exact requested output length, and unique
  prefill identities;
- the same synthetic fixture within each matched comparison; and
- no downloads, unrelated GPU/container work, or public network serving.

The no-thinking lane must resolve to answer content with zero reasoning
characters. The reasoning lane must resolve to the requested reasoning policy.
Mode mismatch invalidates the request instead of merely annotating it.

## Measurement repair

Before another configuration comparison, the harness must:

1. snapshot native MTP counters immediately before each scored request;
2. snapshot them immediately after that same request;
3. subtract drafts, proposed tokens, accepted tokens, and each accepted position;
4. reject negative/reset/inconsistent deltas;
5. preserve one exact delta per repetition and derive the case aggregate only
   from the complete repetition set;
6. record content-token and reasoning-token counts without retaining text;
7. require aggregate accepted positions to equal the accepted-token delta;
8. distinguish client decode, aggregate output, TTFT, and full E2E wall time; and
9. derive GPU-memory utilization and every policy label from the selected
   profile rather than a hard-coded 64K value.

Raw prompts, completions, reasoning, request IDs, commands, and server logs stay
outside tracked evidence. Only deterministic sanitized scalar projections may
enter `evidence/`.

## Experiment ladder

### E0 — clean native-262K control

Purpose: replace the current provenance-mislabeled run.

- Start from zero unrelated GPU/container work and an admitted swap state.
- Launch the exact 0.70 native-262K profile.
- Prove at least 262,144 usable sequence tokens and persist actual KV capacity.
- Run one unscored warmup followed by five reasoning-enabled D256 requests.
- Run the short exact-answer, JSON, and tool gates.
- Stop cleanly unless the process is deliberately retained for the next
  same-configuration request-layer arm.

### E1 — mode/workload matrix in one process

Purpose: reproduce the causal diagnosis without startup drift.

Run five requests per cell in balanced order:

1. default-thinking phrase fixture;
2. no-thinking phrase fixture;
3. default/low-reasoning synthetic coding fixture; and
4. explicit no-thinking synthetic coding fixture.

Each cell gets request-scoped MTP deltas. The fast lane advances only if all
five responses contain answer content, contain zero reasoning characters,
length-terminate, and have a median decode of at least 60 tok/s. The reasoning
lane advances on correctness even if it is slower.

### E2 — MTP depth screen

Purpose: maximize accepted outputs per expensive target verification.

Candidate depths are 4, 6, 7, and 8. Depth eight is the control. The depth-seven
arm was added after source review showed that current upstream fused GDN
verification admits total verification width at most eight; K=8 produces width
nine and falls back, while K=7 fits that future port's fused boundary. The
pinned v0.27.1 runtime does not receive credit for an upstream kernel it lacks,
but this sweep establishes the matching behavior before a costly port. Each
depth gets one fresh lifetime, one warmup, and five D256 coding requests.
Alternate around the control when time permits: 8, 4, 8, 6, 8, 7.

Early reject an arm when any of these occurs:

- mode mismatch, wrong output count, validation failure, device error, or
  safety-gate violation;
- p50 decode below 55 tok/s after five requests;
- no improvement over the nearest control despite higher verification work; or
- accepted positions show that the added tail is effectively unused.

Retain a candidate only when its median improves by at least 3% over the
matched control without worsening p25 by more than 5%. Otherwise retain depth
eight. A smaller reasoning-lane screen compares depths 4 and 8 because low
acceptance makes long speculative tails especially suspect.

### E3 — sustained confirmation

Purpose: turn a screen into a claim.

- Run the retained arm in two new lifetimes.
- In each lifetime run one warmup, five D256 requests, then three D1024
  requests.
- Require both lifetime medians and their pooled median to meet the 60 tok/s
  target in the fast lane.
- Record p25/p50/p75, aggregate output, TTFT, E2E, acceptance by position,
  temperature, power, clocks, minimum MemAvailable, and swap growth.
- Run exact answer, JSON, tool selection, tool execution, and a bounded coding
  response structural gate. Never execute generated model code.

### E4 — context ladder

Purpose: measure the service the user will actually feel.

At 8K, 64K, 128K, and the largest safety-admitted near-native prompt, run one
retrieval oracle followed by a D256 coding response. Keep prefix caching off.
Report TTFT, approximate prefill throughput, decode, E2E, MTP acceptance, KV
usage, and memory pressure. Passing native capacity does not imply 60 tok/s at
262K; the short-context headline and context-dependent curve remain separate.

### E5 — quarantined hybrid prefix-cache candidate

Purpose: reduce repeated Codex/tool-loop prefills without corrupting state.

The only real cache-on arm for this Qwen3.8 hybrid path is
`--enable-prefix-caching --mamba-cache-mode align`. It receives a separate
profile, cache namespace, fresh process, and resolved-configuration receipt.
It is never mixed into E0–E4.

Gates, in order:

1. reproduce the recipe's raw-token boundary oracle at lengths 1, 8, 9, and 10
   against a fresh cache-off golden;
2. repeat each boundary enough to catch the reported nondeterminism;
3. test cold plus at least two identical long-prefix replays;
4. run twelve monotonically growing 40K-to-160K synthetic prefixes with unique
   suffixes and an exact semantic oracle;
5. require zero hits on the cold request and positive, plausible native hits on
   later requests;
6. abort immediately on the first token mismatch, CUDA/device fault, engine
   death, impossible counter delta, eviction anomaly, or pressure failure; and
7. only after all synthetic gates, run a short scalar-only Codex tool loop.

If the production cache-on arm fails, use fresh diagnostic processes to test
MTP off first and graphs disabled second. Those isolation arms cannot be
promoted as the production result without their own throughput and quality
qualification.

### E6 — real harness check

Purpose: translate microbenchmarks to coding/cowork wall time.

- Run one bounded Codex or Pi task with a small, explicit tool catalog.
- Compare low reasoning with the fast no-thinking lane where the client
  protocol permits it.
- Measure first-turn and later-turn prompt tokens, TTFT, decode, tool-call
  validity, retries, total task wall, and prefix-cache tokens.
- Repeat with the normal tool catalog only after the small-catalog path works.
- Do not attribute tool-schema prompt reduction to model/kernel speed.

## Eight-hour execution budget

| Elapsed window | Work |
|---|---|
| 0:00–0:45 | Freeze this protocol, repair provenance and request-scoped metrics, unit-test contracts |
| 0:45–1:30 | E0 clean control and E1 managed mode/workload reproduction |
| 1:30–4:15 | E2 depth screen, including startup and cleanup time |
| 4:15–5:30 | E3 two-lifetime D256/D1024 confirmation |
| 5:30–6:30 | E4 bounded context ladder or, if decode confirmation fails, profiler-guided isolation |
| 6:30–7:20 | E5 cache boundary/growing-prefix gate; stop early on any correctness fault |
| 7:20–8:00 | E6 short harness check, scalar evidence export, tests, documentation, selective commit and push |

Correctness and cleanup take priority over filling the clock. A failed cache
gate returns immediately to the admitted cache-off winner. If a startup consumes
the remaining confirmation budget, skip the next speculative arm rather than
publish an unreplicated winner.

## Promotion and stop rules

Promote a configuration only when:

- the exact launch receipt agrees with argv and resolved runtime state;
- the fast lane clears 60 tok/s in both fresh lifetimes at D256 and in the
  sustained D1024 confirmation;
- all scalar mode, token-count, MTP, quality, tool, memory, and device gates
  pass;
- no comparison relies on cumulative metrics from another workload;
- long-context behavior and prefix-cache state are labeled separately; and
- deterministic evidence verification and the staged secret scan pass.

Do not claim success from a single request, a short client-decode peak,
concurrency, repeated-token output, a cache-assisted prefill rate, an unsafe
swap-crossing run, or a completion that spent the measured window only in
hidden reasoning.
