# Qwen3.8-Flash-Next single-user serving backlog — 2026-08-28

## Scope

This is the ranked backlog for after the frozen 64K autoresearch campaign. The
campaign remains pre-measurement and safety-stopped; this file is not part of
that immutable campaign and does not authorize changing its plans, cutoff,
suite, or profile queue. Freeze a new protocol only after the campaign is
terminal and its scalar evidence is published.

The product target remains one person using a coding agent or cowork-style
assistant. Optimize correct end-to-end task wall time, later-turn latency, and
safe residency on one DGX Spark. Aggregate multi-user throughput is secondary.

This file owns the future SGLang **product track**: serving flags and workload
geometry are ranked only after a new admitted baseline reproduces the historical
mapped-FP8-PLE, lazy-state, MTP2 geometry. The
parallel **systems track** first admits exact vLLM direct mmap, then isolates
its live-token-width patch and the mixed-FP8 artifact; see the
[vLLM reproduction plan](qwen38-flash-next-vllm-mmap-reproduction-2026-08-28.md)
and [bottleneck analysis](qwen38-flash-next-tps-bottleneck-2026-08-28.md).
The tracks have different runtime and artifact baselines and are not one
interleaved ranking. A systems candidate does not displace the newly admitted
product baseline until it passes its own admission and matched task gates.

That baseline is a historical measured prior and the immutable campaign's
frozen identity, not a runtime for new work: it uses the superseded SM121
TRT-LLM overlay. Before any backlog experiment, build and admit a newly pinned
SM121 Triton runtime and PLE-capacity mechanism, reproduce the baseline geometry
and correctness, and establish fresh C1 rates. The current statically composed
`3681c4e`-derived candidate uses the `io_uring` PLE reader; it does not
by itself preserve the historical persistent-mmap overlay. Treat a future mmap
port as a separate integration and admission gate. The ranking below orders
product questions, not permission to reuse the historical image. Never pool
performance across the d91-image composition and a `3681c4e`-derived or later
admitted runtime.

## Evidence-backed prior

The historical retained prior is mapped FP8 PLE, `extra_buffer_lazy`, and NEXTN
depth two. Its two independent fresh-C1 lifetimes averaged 29.594 output tok/s;
its two warmed D256 lifetimes averaged 29.402 tok/s. The single mapped depth-three
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
planned three one-axis questions in order:

1. low reasoning versus explicit no-thinking;
2. 1,024- versus 2,048-token chunked prefill; and
3. NEXTN depth two (`steps=2`, `draft_tokens=3`) versus depth three
   (`steps=3`, `draft_tokens=4`).

None of those cells started, so all three questions remain unanswered. Do not
duplicate or alter the cells before terminal disposition. Afterward, re-freeze
any still-valued axis against the newly built and admitted runtime; there is no
winner to combine or promote from this campaign.

## Ranked next experiments

### 1. Long shared-prefix SGLang cache-policy bundle

This has the highest direct coding/cowork value. The current 60K case is a
one-shot prompt, while the agent cases have small histories; neither measures
repeated long-prefix work across tool turns.

- Bundle A: the newly admitted C1 baseline, default Radix behavior, and requested
  `extra_buffer_lazy`; require startup `impl=UnifiedRadixCache` with hybrid SSM.
- Bundle B: add `--disable-radix-cache`; the baseline chunked prefill then selects
  `ChunkCache`, and the runtime disables the lazy Mamba-state predicates even
  though their argument remains present. Require `impl=ChunkCache` at startup,
  pin the source assertion that both lazy predicates are false, and corroborate
  one-state runtime use with scalar pool/state evidence.
- Suite: in each lifetime run T0 as a cold 32K--48K deterministic coding or
  document prefix, then T1 and T2 after appending prescribed assistant tool
  calls, tool results, and user suffixes. Do not feed arm-specific generated
  text back into the next prompt.
- Design: fresh-lifetime A/B/B/A with two independent lifetimes per bundle.
  Use no inference warmup before T0. Treat T0 as cold calibration and score
  T1/T2 separately.
- Promotion contract: strict correctness is mandatory. The sole speed primary
  is the ratio of unweighted arm means for lifetime-level T1+T2 resident wall.
  Promote B only if `B/A <=0.95`; retain A if `A/B <=0.95`; otherwise call the
  speed result inconclusive. Require the selected bundle's unweighted-mean
  later-turn TTFT and full T0--T2 wall each to remain `<=1.05x` the other
  bundle. Decode TPS and T0 cold wall are separate diagnostics, never averaged
  into the primary.
- Identity gate: render with the same pinned tokenizer, template, tools,
  serialization, reasoning policy, sampling and output cap. Compute
  domain-separated token-ID digests from volatile
  `return_prompt_token_ids=true` responses, compare them privately, then
  discard IDs and digests. Publish only scalar token/common-prefix counts and
  `prompt_identity_verified`; reject mismatched histories.
- Hit gate: make non-streaming `return_cached_tokens_details=true` responses
  primary. Require T0 device/host/storage cached counts to be zero in both
  bundles. First admit a zero-hit canary: on the reviewed implementation an
  omitted or null detail object means zero only when native counters also
  prove no hit. Normalize that admitted response to zero; otherwise reject
  missing or malformed request details. Before freezing, derive a numeric
  expected interval for every A T1/T2 device hit from the rendered common
  prefix, `input_len-1` cap, page alignment, and admitted Mamba checkpoint
  grid; require B to remain zero.
  Reconcile request values after quiescent metric settle with deltas for
  `sglang:prefill_effective_tokens_total{mode="device_hit"}` and
  `sglang:cached_tokens_total{cache_source="device"}`; do not treat an
  instantaneous scrape as request attribution. `sglang:cache_hit_rate` alone
  is insufficient.
- Residency gate: record KV/Mamba available, used, and evictable token gauges
  around every turn. Any other request, eviction, retraction, pressure breach,
  or missing counter invalidates the lifetime.

This is deliberately a product-bundle comparison, not an isolated trie
ablation. In the exact
[`d91c3682` overrides](https://github.com/sgl-project/sglang/blob/d91c3682b0b429e4c70df63cd57f819588ce29b0/python/sglang/srt/arg_groups/overrides.py),
disabling Radix also suppresses `extra_buffer_lazy`; the cache
[registry](https://github.com/sgl-project/sglang/blob/d91c3682b0b429e4c70df63cd57f819588ce29b0/python/sglang/srt/mem_cache/registry.py)
selects `ChunkCache`. Bundle A has a four-slot lazy peak at C1 while B needs one
live recurrent-state slot, but the explicit four-slot pool likely remains
allocated in both, so do not claim reclaimed model memory. The public d91 tree
predates public Qwen4/QSA support; these mechanics are a historical source
prior only. Re-audit them on the newly built and admitted source and attest its exact
runtime before claiming that a matched hybrid prefix restores full/QSA KV or
that matched Mamba-slot copy-on-write restores PLE side state.
The result estimates whether the admitted cache/state bundle avoids repeated
prefill for iterative work. It must not infer a hit from latency or attribute
the delta to Radix bookkeeping alone.
It does not exercise vLLM `enable_prefix_caching` or `mamba_cache_mode`; those
belong to the separately quarantined systems track.

### 2. Two-way single-user fan-out, admission first

This is an application-scheduling experiment for one person decomposing one
task into two independent subtasks. It is not a multi-user throughput claim.
The historical 64K prior admitted only one running request. First establish a
new C1 baseline, then freeze a separate C2-capable profile and prove its C1
behavior and safety before scoring parallel work.

- Admission bundle: keep the 65,536-token pool, raise running requests from one
  to two, provide eight lazy recurrent-state slots for two four-slot sequences,
  and capture decode graph batches one and two. Treat this as one inseparable
  concurrency-geometry bundle.
- Capacity gate: precompute the maximum combined fully rendered histories,
  reserved outputs, and draft allowances over every legal simultaneous
  multi-turn path. It must fit the shared pool with an explicit safety margin;
  full-60K requests are therefore outside this arm.
- Geometry control: compare the C2-capable profile at C1 against the newly
  admitted C1 profile before using it. Before launch, freeze an exact
  fixed-agent workload, an unscored warmup, a lifetime-level resident-wall
  estimator, at least two fresh lifetimes per profile, and a numeric
  non-inferiority bound. Require the C2-profile/C1-profile ratio of unweighted
  mean lifetime wall to be `<=1.0102`, every oracle, and every pressure gate;
  keep D256 as a separate secondary rather than averaging it into task wall.
- Scheduling control: inside the same admitted C2 profile, run two fixed,
  independent, strictly validated native cowork/agent subtasks serially and
  then in parallel. Counterbalance serial/parallel order across fresh lifetimes
  and keep total prompts, output budgets, tools, and oracle work identical.
  Prepare both task environments before the timed release in both modes;
  parallel releases both, while serial releases the second only after the first
  terminal result. Add Pi only after its separate client-stack admission.
- Primary outcome: time until both correct artifacts are complete. Retain
  per-task E2E, TTFT, output-limit hits, memory, swap, and fairness as
  guardrails. Before launch, freeze at least two fresh lifetime pairs, the
  lifetime-level time-to-both estimator, a numeric minimum speedup, and numeric
  per-task E2E/TTFT fairness bounds. Do not pool request ratios or choose a
  threshold after observing the results.

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

- Control: the newly admitted fixed-depth MTP2 C1 baseline.
- Candidate: an otherwise identical MTP-off profile that removes only the
  complete speculative-decoding bundle.
- Suite: strict short JSON/tool cases, the complete multi-turn agent battery,
  and D256 as a decode anchor. Hold reasoning, prompt, output, parser, cache,
  and sampling policy fixed.
- Design: fresh-lifetime ABBA with two independent lifetimes per arm. Report
  cold start-to-first-correct-task wall and resident task wall separately;
  never subtract startup after the fact to create a synthetic winner.
- Promotion contract: every exact oracle must pass. For the ephemeral decision,
  the sole primary is the ratio of unweighted arm means for cold
  start-to-first-correct short-battery wall; MTP off must be `<=0.95x` MTP2.
  For the separate resident decision, the sole primary is unweighted-mean
  multi-turn agent resident wall and the same `<=0.95x` rule applies. Neither
  decision may use the other's outcome, D256, or startup subtraction. Preserve
  TTFT and D256 as diagnostics.

### 4. Rejected at source gate: continuous decode steps

Do not run the planned `num_continuous_decode_steps=1` versus `2` ABBA on the
reviewed public sources. On both the reported-base
[`d91c3682` tree](https://github.com/sgl-project/sglang/blob/d91c3682b0b429e4c70df63cd57f819588ce29b0/python/sglang/srt/server_args.py#L966-L970)
and the corrective-source candidate's public
[`3681c4e` source](https://github.com/sgl-project/sglang/blob/3681c4e03f6848dff82972b3f572602d3b8394cc/python/sglang/srt/server_args.py#L972-L976),
the field is declared with default one but no scheduler or worker reads it. The
generated scalar CLI accepts any integer without a choices/range gate, while
the exact
[scheduler loops](https://github.com/sgl-project/sglang/blob/d91c3682b0b429e4c70df63cd57f819588ce29b0/python/sglang/srt/managers/scheduler.py#L1721-L1828)
retain one plan/run/process cycle per iteration.

Values one and two are therefore equivalent on those public trees. Pristine
d91 predates Qwen4/QSA, while the measured image contains baked changes plus
tracked overlays. Before considering its exact historical runtime, statically
attest that the image's server/scheduler/worker files also have no consumer; if
they match, reject without a GPU lifetime. Reconsider only on a new admitted
source identity that actually reads the field, with a unit or instrumented
cadence test proving the branch is exercised before any GPU ABBA.

### 5. Decode CUDA-graph causal control

- Control bundle: `--cuda-graph-backend-decode full` with
  `--cuda-graph-bs-decode 1`.
- Candidate: change only the backend to `disabled`; retain
  `--cuda-graph-bs-decode 1` and the disabled prefill-graph setting.
- Admission: require resolved decode backend `full`, batch list `[1]`, and
  `max_bs=1` versus backend `disabled` with the same list and maximum. The
  control must
  increment `sglang:cuda_graph_passes_total{mode="decode_cuda_graph"}` and the
  candidate only `mode="decode_none"`; this attests target replay. Separately
  require exact pinned-runtime state/log evidence for nonzero versus zero
  target-verify, draft-decode, and draft-extend capture construction.
- Canary: before ABBA, run one exact capability/tool/stream/cancel sequence and
  D256. Require semantic terminal/tool invariants, bounded cancellation, and no
  emission after a stop; graph/eager token identity and cancel latency are not
  assumed.
- Design: ABBA with two independent lifetimes per arm. In every lifetime use
  the same unscored warmup, five D256 repetitions, and one fixed agent/tool
  fixture. Flush and attest the request cache before each timed repetition. If
  that is unavailable, use an admitted private per-request salt in the first
  cacheable block, share its schedule across arms, prove its pre-tokenized LCP
  is below the reusable unit, and require request-scoped cached-token count
  zero. Keep cold ready time, per-phase capture time/memory, D256, NEXTN
  acceptance, agent task wall, MemAvailable, and swap separate.
- Interpretation: the expected result is a control speed win. A disabled-graph
  simplification uses correct fixed-agent resident wall as its sole promotion
  primary. Require the exact oracle in every replicate and a candidate/control
  lifetime-level wall ratio at most `1.0102` (equivalent to at least `0.99x`
  speed), plus removal of the graph bundle or at least 1 GiB more available
  memory twice. Keep D256 as a separate secondary; never combine heterogeneous
  speed measures.

This is a full speculative-decode graph treatment, not target verify alone.
Exact source returns no target capture when decode graphs are
[disabled](https://github.com/sgl-project/sglang/blob/d91c3682b0b429e4c70df63cd57f819588ce29b0/python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py#L410-L490),
and NEXTN returns before draft-decode/draft-extend
[capture](https://github.com/sgl-project/sglang/blob/d91c3682b0b429e4c70df63cd57f819588ce29b0/python/sglang/srt/speculative/eagle_worker_v2.py#L343-L392).
Do not remove the explicit batch-one list in this causal pair. When absent, the
exact GB10 path
[resolves](https://github.com/sgl-project/sglang/blob/d91c3682b0b429e4c70df63cd57f819588ce29b0/python/sglang/srt/server_args.py#L4791-L4865)
`max_bs=256` even with the backend disabled, and NEXTN consumes that maximum
for chain-buffer allocation. List removal is a later configuration-cleanup
bundle. Do not capture graph batches above one for a C1 objective; they are
unexercised and add capture, startup, and headroom cost.

Keep graph results source-pin-local. Safe candidate `3681c4e` adds Qwen
QSA-specific draft backend and index-sharing wiring, so identical flags need
not imply identical graph topology or overlap behavior across d91 and 368.
Never pool those lifetimes. Public d91 also predates native Qwen3.8
Flash-Next/QSA support; the historical serving identity is the digest-pinned
image plus baked patches and tracked overlays. Treat specialized QSA capture as
a runtime-attested fact, not an inference from pristine d91 source.
Per-phase memory/startup fields attest capture construction and footprint, not
draft replay counts; without a native draft-replay counter, attribute speed
only to the complete graph bundle. Bracket fixed-width verify/accept counters
after the unscored warmup and separately around D256 and the agent fixture;
never publish a whole-lifetime rate that mixes those blocks.

### 6. Conditional next chunk size

The stopped campaign produced no 1,024/2,048 result. First refreeze that pair on
the newly admitted runtime. Propose another value only after the new pair
completes validly:

- If 2,048 wins, compare the promoted champion with 4,096.
- If 2,048 loses, compare 1,024 with 512.
- If the 1,024/2,048 pair is inconclusive, stop; do not propose another value.
- Clone each new profile from the then-current admitted winner, change only
  `--chunked-prefill-size`, and use ABBA with two fresh lifetimes per arm.
- Promotion contract: every exact oracle must pass. The sole primary is the
  candidate/control ratio of unweighted arm means for lifetime-level correct
  60K E2E; promote only at `<=0.95`. Require candidate/control
  unweighted-mean later-turn agent wall and 60K TTFT each to remain `<=1.05`.
  Keep D256 throughput, memory, and swap separate.

At concurrency one, larger chunks have no scheduling-fairness justification.
They must win on actual prefill or task wall time without harming interactive
latency or pressure safety.

### 7. Rejected at source gate: adaptive NEXTN

The generic adaptive EAGLE machinery exists at both reviewed public pins, and
corrective-source candidate `3681c4e` makes the Qwen4Exp draft load path
inspectable. Public d91 lacks
native Qwen4Exp draft support, so the historical Qwen path still depends on the
digest-pinned image, baked patches, and overlays; pristine d91 alone cannot
attest its Qwen-specific consumption. This does not admit a benchmark arm. The
[default candidate union](https://github.com/sgl-project/sglang/blob/d91c3682b0b429e4c70df63cd57f819588ce29b0/python/sglang/srt/speculative/adaptive_spec_params.py#L22-L47)
is `{0, 1, 3, 7}`, so the retained depth-two setting fails the exact
[membership gate](https://github.com/sgl-project/sglang/blob/d91c3682b0b429e4c70df63cd57f819588ce29b0/python/sglang/srt/arg_groups/speculative_hook.py#L788-L810)
rather than becoming the baseline tier. A custom adaptive config is therefore
a coupled bundle, not a one-flag treatment. On that reviewed source, adaptive mode
also
[disables Qwen QSA index sharing](https://github.com/sgl-project/sglang/blob/3681c4e03f6848dff82972b3f572602d3b8394cc/python/sglang/srt/speculative/eagle_worker_v2.py#L382-L389).
After alias resolution, adaptive also requires EAGLE/EAGLE3 with top-k one;
DP attention, multi-layer EAGLE, two-batch overlap, or PDMux can warn and fall
back to static parameters. A future gate must attest the resolved algorithm,
top-k, and `speculative_adaptive=true`, not argv alone.

The decisive blocker is measurement validity. The scheduler exposes
point-in-time active-step/draft-token gauges and records verify/accepted totals,
but no per-tier residence or per-verify proposed count. The tokenizer then
[computes proposed drafts](https://github.com/sgl-project/sglang/blob/d91c3682b0b429e4c70df63cd57f819588ce29b0/python/sglang/srt/managers/tokenizer_manager.py#L2780-L2803)
as verify count times the startup fixed draft width. That formula becomes wrong
after any adaptive tier switch. A request or observation window spanning a
switch therefore has no attestable aggregate proposed count or acceptance rate.

The default union also expands the maximum from retained step two to step seven
(three versus eight draft-token slots), and each candidate owns step-shaped
draft/verify/extend resources and graphs. That is a serving-geometry bundle,
not a free controller toggle.

Do not spend a GPU lifetime or publish native acceptance metrics for adaptive
NEXTN on these pins. Reconsider only after an admitted source change records
exact proposed tokens per verify, active-tier residence, and accepted position;
then freeze and hash a matched custom policy. The smallest future pair is
adaptive `[2]` as a no-switch control versus `[1, 2]`, both initialized at two:
both retain a three-slot maximum and disable candidate-source QSA index sharing,
while the candidate adds only step-one runtime state. Run an unscored counter
oracle before any ABBA.

### 8. Stream endpoint suppression, profile-specific

This remains lower priority than compute and cache axes. For OpenAI chat,
control `stream_interval=1` versus a candidate changing only
`--stream-interval 2`; reject any raw `/generate` request-level override. The
field is a real output-side axis on both reviewed pins, but neither server nor
request parsing validates a positive value. Those pins provide historical
mechanics only: re-audit consumption and parser behavior on a newly built and
admitted SM121 Triton runtime, pin its source identity, and statically require
exactly one or two before launch.

Do not describe the candidate as a two-token buffer. The exact
[output streamer](https://github.com/sgl-project/sglang/blob/d91c3682b0b429e4c70df63cd57f819588ce29b0/python/sglang/srt/managers/scheduler_components/output_streamer.py#L362-L390)
emits an unfinished request at interval `N>1` only when cumulative output
length modulo `N` equals one. NEXTN first appends an entire accepted run and
calls the streamer once, so it can jump over an emission endpoint. At two,
scalar decode emits at lengths 1, 3, 5, and so on, while odd/even speculative
acceptance can suppress several verify results or make the setting effectively
inert. Finished requests always flush. Prefill produces one sampled token, so
the first scheduler output batch is unchanged; measure the first non-empty
semantic wire delta because reasoning/tool parsers may hold that batch.

- Design: fresh-lifetime A/B/B/A at C1 with identical request bytes, sampling,
  reasoning, tools, and limits. In each lifetime run one deterministic streamed
  256--512-token coding/text case and one forced tool-call case; add one short
  cancel probe only if cancellation is part of promotion.
- Wire attestation: timestamp loopback SSE privately and group parser-created
  frames by cumulative completion-token value. Send identical
  `stream_options={"include_usage":true,"continuous_usage_stats":true}` in both
  arms and admit its semantics first. Before launch, define a semantic frame as
  a choice delta with non-empty content, reasoning, or tool-call material;
  count role-only, finish-only, usage-only, extension, error, and `[DONE]`
  frames separately, excluding them from semantic-gap statistics. Freeze the
  zero/one-semantic-frame behavior. Publish only semantic frame and
  distinct-usage counts, first semantic delta, median/p95/maximum wire gap,
  first tool-name time, valid-tool-JSON time, finish time, total wall, final
  token count, `final_output_exact_match`, and exact-tool/oracle booleans. Event
  arrays and content-derived hashes stay private.
- Server controls: bracket requests with TTFT/E2E, generation/request, verify,
  and abort counters plus acceptance and active-width gauges/log snapshots.
  Server inter-token latency is token-normalized at output batches, not a
  wire-gap metric.
- Gates: exact final text/tool hash, token count, tool name/arguments, finish
  reason, complete terminal flush, no retraction/error, and comparable NEXTN
  evidence in each arm independently. Freeze the lifetime-level estimator and
  numeric first-semantic, tool-ready, p95/maximum-gap, and minimum wall-gain
  thresholds before launch; never pool request ratios or define
  non-inferiority after seeing results.

Reduced IPC, detokenizer/parser work, or SSE frames is not a model-forward gain.
Promote only as an empirically retained profile-specific setting; do not make
it a general default from frame reduction, call it two-token coalescing, or
extrapolate automatically to interval four.

## Retained-SGLang product diagnostics

Within the newly admitted SGLang product track, a separate unscored lifetime should
bracket three profiled C1 D256 requests with two unprofiled three-request
blocks. Start with CUDA/NVTX timelines,
source timers, process `/proc` deltas, PSI, NVMe disk deltas and one-hertz
device telemetry; add one deterministic varied-token long prompt only after
the minimum trace is interpretable. Pin the collection window and report U-P-U
overhead because profiling, system-wide faults and page-cache state can
otherwise confound timing.

Current GB10 hardware counters require separate privilege authorization, and
the CC 12.1 metric surface exposes no direct LPDDR-byte counter. If a later
disposable privileged lifetime is explicitly approved, treat L2 sysmem-fill
sectors only as an LPDDR-facing proxy, never measured LPDDR GB/s. The present
coarse utilization and power samples cannot distinguish target-weight traffic,
PLE page traffic, kernel overhead or another resource. This diagnostic must
use the then-current newly admitted or promoted product profile, remain outside
a causal candidate pair, and must not
silently change its serving flags. The separately specified vLLM direct-mmap
profile is a systems-attribution diagnostic chosen for instrumentability; it is
not this product-track lifetime and the two diagnostics are not an A/B pair.

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
  axis. Low versus no-thinking was frozen but remains unmeasured; re-freeze it
  only after the stopped campaign is terminal.
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
threshold, so a single forward run cannot promote. Freeze one lifetime-level
primary and its direction before launch. Unless a section declares an explicit
alternative contract, require candidate/control primary wall `<=0.95` and
every latency guardrail `<=1.05`; if no numeric estimator and thresholds are
frozen, do not launch.

Publish only the deterministic scalar projection. Raw prompts, completions,
reasoning, tool payloads, agent trajectories, logs, paths, commands, request
identifiers, and native trace bodies remain ignored. Stop immediately on the
existing memory, swap, ownership, and cleanup thresholds. Every future
campaign must freeze its own explicit cutoff; never inherit or extend the
expired cutoff from this campaign.
