# Qwen3.8-Flash-Next TPS bottleneck analysis — 2026-08-28

## Conclusion

For a healthy resident server at concurrency one, the strongest current
hypothesis is the cost of each small-batch target-model verification pass:
routed-weight traffic and low arithmetic intensity, small-kernel efficiency,
and possibly launch/scheduler overhead. MTP and request batching produced the
largest measured gains and are consistent with amortizing that cost.

The data do not yet separate LPDDR bandwidth, kernel occupancy, kernel-launch
gaps, or CPU scheduling well enough to name one physical resource as the sole
bottleneck. Coarse GPU utilization near 95–96% is not proof of saturation: two
replicated C8 lifetimes delivered about four times C1 aggregate throughput at
nearly the same utilization reading and only about 10–11% more sampled power.

The exact read-only NVMe mapping of the trained PLE is primarily an
admission/capacity mechanism. Nothing measured so far shows that storage tactic
dominating warm short-prompt decode. Memory headroom and swap are hard
configuration/safety limits, but they are not themselves a demonstrated
per-token rate limiter.

For the user's actual wall time, the hierarchy changes by workload:

1. a cold managed lifetime is dominated by roughly ten minutes of startup;
2. a resident long generation is dominated by target/draft decode work;
3. a short JSON or tool turn is often dominated by TTFT rather than emission;
4. a long iterative agent may be dominated by repeated prefill unless prefix
   reuse works; and
5. the fixed synthetic ladder motivates testing whether two independent
   subtasks can trade higher per-task latency for lower combined makespan
   through C2 fan-out.

## What the scalar evidence establishes

| Signal | Observation | Defensible implication | Boundary |
| --- | --- | --- | --- |
| MTP3 versus off | [Clean MTP3](../evidence/runs/20260827T194940Z-qwen38-flash-next-nvfp4-mtp-depth3-sglang-qwen38-flash-next-sglang-mtp-depth-confirm-af30d00f/summary.json) reached 30.123639 tok/s versus [off](../evidence/runs/20260827T200256Z-qwen38-flash-next-nvfp4-mtp-depth0-sglang-qwen38-flash-next-sglang-mtp-depth-confirm-aa26aac9/summary.json) at 16.663713 tok/s, `1.807739x` over 5,120 output tokens | Enabling trained MTP3 is the largest established C1 decode lever; its separate audit confirms the intended multi-token verification path | One lifetime per arm, 1.26% aggregate prompt-token mismatch, D256 only |
| Replicated MTP2 batching | [Lifetime A](../evidence/runs/20260827T234248Z-qwen38-flash-next-nvfp4-mtp2-c8-lazy-ple-mapped-sglang-qwen38-flash-next-sglang-ple-depth-c8-0c745ab5/summary.json) scaled from 28.441 C1 to 113.108 C8 tok/s (`3.9769x`); [lifetime B](../evidence/runs/20260827T235628Z-qwen38-flash-next-nvfp4-mtp2-c8-lazy-ple-mapped-sglang-qwen38-flash-next-sglang-ple-depth-c8-0c745ab5/summary.json) scaled from 30.746 to 121.173 (`3.9411x`) | One sequence does not exhaust aggregate service capacity; batching raises capacity and is consistent with weight/kernel amortization | 4K synthetic 256-token requests, not one user's sequential response rate |
| Per-job C8 cost | Median E2E was `1.9741x` and `1.9479x` C1 while aggregate output was about `3.96x` | Parallel independent work can reduce total makespan even though each task slows | Requires decomposable work; TTFT also rose `2.58x`/`2.42x` |
| Coarse utilization and power | C1/C8 utilization was 95.85%/95.40% and 96.00%/96.00%; sampled power was 34.16/38.01 W and 34.69/38.24 W | The utilization percentage records busy time, not useful occupancy, arithmetic intensity, or bandwidth saturation | Sensor scope and one-second sampling cannot attribute kernels or memory traffic |
| PLE omission | The matched D3 ablation changed valid D256/C1/C2 rates by -2.9%/+7.8%/+3.4%, then produced incomplete C4/C8 work | No large, stable short-decode gain has been established by removing PLE | Omission changes model semantics and invalid high-concurrency work has no aggregate |
| Prefill versus decode | The native run measured 2,103–2,180 prompt tok/s on repeated-word 8K/32K, versus 28.504 output tok/s at D256 | Decode follows a very different, serial recurrent/target path from batched synthetic prefill | Prompt locality and metric definitions differ; this is not a direct ratio of one kernel |
| Short-output TTFT | In the primary run, 20-token JSON, 36-token tool, and 32-token chat cases spent 69.4%, 78.3%, and 53.7% of E2E in TTFT; D256 spent 3.8% | Decode TPS matters for sustained output, while prompt/request latency dominates many short tool turns | One request per short case; these are not a complete agent-task distribution |
| Long-prompt TTFT | The synthetic 8K/32K needles spent 92.3%/98.2% of E2E before first output | At long context, prefill and locality can dominate wall time even when prompt tok/s is high | Repeated-word prompts and client-TTFT approximations, not natural documents or native prompt counters |
| Startup | The primary native lifetime took 581.652 s; logs attributed 420.36 s to target loading and 83.86 s to MTP loading | Residency dominates small serving-flag gains when every task otherwise starts cold | Log components do not cover every startup phase |
| Headroom | Safe mapped MTP2 cases retained about 16 GiB available, while the single mapped D3 first request fell to 13.905 GiB and ordinary-buffer arms later grew swap | Depth, graph, recurrent-state, and context geometry are constrained by unified-memory headroom | A pressure boundary is not proof of the steady-state decode bottleneck |

The MTP2 native audits also show that speculation was real, not a client timing
artifact. They accepted 157/196 and 158/198 drafted tokens, with mean accepted
lengths 2.602 and 2.596 including the verified target token. The separate MTP3
audit accepted 175/243 draft tokens across 81 drafts and reached mean accepted
length 3.160. These counters apply only to the explicit audit requests, not
retroactively to the streaming cases.

The clean MTP3/off pair also reported almost unchanged coarse utilization and
power: 95.772%/38.596 W with MTP3 versus 95.939%/38.755 W off. Together with
the separate acceptance audit, the speed and energy-per-output gains are
consistent with avoiding serial target work rather than raising the sampled
power envelope, but the measurements do not apportion the gain. MTP also had a
higher median TTFT, 0.408 seconds versus 0.257 off, even though D256 median E2E
was 8.344 versus 15.294 seconds. Neither result identifies the target kernel as
bandwidth- or compute-bound or guarantees a win for very short output.

The primary lifetime gives a stronger internal anti-power-limit observation:
D256 averaged 34.821 W and peaked at 37.8 W, while 8K/32K prefill in that same
lifetime averaged 56.195/63.989 W and peaked at 64.61/65.86 W. Short decode was
below the highest observed power draw, although the samples contain no
high-rate cap or throttle-reason counters.

Across the replicated mapped panel, depth two beat depth one in every cell by
2.385–12.572%. The single depth-three lifetime was only 1.122% above the D2
mean at C8, up to 9.586% above at D256, and violated the first-request memory
floor. This supports MTP2 as the safe default while leaving the exact
depth-two/depth-three task-wall comparison to the frozen campaign.

## Likely resident-C1 mechanism

The MTP and concurrency signals point in the same direction. A plain token
step pays for a target-model pass to emit one token. Speculation verifies
multiple candidate positions in one target pass; batching processes positions
from several sequences together. Both can improve matrix shapes, reuse loaded
weights, and reduce per-output launch/scheduling cost.

That makes target routed-weight movement and small-batch kernel efficiency the
leading combined hypothesis. It is intentionally broader than “memory
bandwidth bound.” The same throughput pattern can arise from:

- LPDDR traffic for target/shared/expert weights;
- poor occupancy or small GEMMs at batch one;
- dispatch and synchronization around routed experts, QSA, or GDN;
- kernel-launch and scheduler round trips; or
- a mixture that changes with accepted draft length and batch size.

The existing samples cannot apportion those mechanisms. High reported GPU
utilization only says some GPU work was active during most sample windows.
Likewise, the close community and local no-spec rates do not prove a common
hardware bottleneck.

One new external result raises the priority of dense weight traffic without
settling it. A
[mixed-FP8 checkpoint](https://huggingface.co/lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8)
quantizes QSA and GDN input/output projections that remain BF16 in the Radix
artifact. Its author reports a 39% single-stream improvement over an
unquantized-dense build after forcing vLLM away from a broken SM121 DeepGEMM
route to CUTLASS
([issue #54125](https://github.com/vllm-project/vllm/issues/54125)). That is
directionally consistent with the BF16 GEMV/weight-traffic hypothesis.

It is not local causal evidence: the artifact changed, the published request
and MTP geometry is incomplete, and the model card labels the checkpoint a
private candidate. Test it only after the exact mmap runtime baseline, with a
pinned revision, matched perplexity and strict task/long-context gates.

## Why PLE and NVMe are not the current headline

The exact 47.684 GiB FP8 PLE payload is read-only and demand-mapped from NVMe,
which is what lets the full checkpoint fit. Warm short prompts may reuse PLE
rows or page-cache state, but no tracked row-set, residency, page-fault, or NVMe
counter establishes that working set. The semantic omission arm did not
produce a large consistent speedup before its high-concurrency failures. That
is weak evidence against PLE lookup being the dominant warm short-decode cost,
not an explanation of why.

It does not settle cold or varied-token behavior. A long natural prompt may
fault many different PLE rows, and iterative agents may experience a different
page-cache working set. The current repeated-word prefixes are explicitly poor
proxies for that question. Native page-fault and NVMe-byte attribution is still
needed.

## Measurements that would narrow the physical bottleneck

This section defines the parallel vLLM **systems-attribution track**, not the
ranking of serving flags against the retained SGLang product baseline. Start
with one separate, unscored vLLM direct-mmap, MTP-off, C1 diagnostic
lifetime after that branch passes build and admission. Its direct-mmap boundary
can be instrumented more precisely than the current SGLang path. Bracket three
profiled D256 requests with two unprofiled three-request blocks, then expand to
the existing 4K C8-capable mapped-MTP2 profile and C1/C2/C8 only if the minimum
trace is interpretable. Use a separately admitted 64K profile for one
deterministic varied-token long prompt. None of these diagnostics belongs in a
promotion score; publish only allowlisted aggregates.

The local Nsight gate materially limits the claim. CUDA/NVTX timelines work,
but hardware counters currently require profiling privilege, and the stock
CC 12.1/GB20B metric sets expose no direct DRAM-byte counter. Even with a
separately authorized privileged lifetime, L2 sysmem-fill sectors are only an
LPDDR-facing proxy. Model-implied dense bytes divided by a clipped GEMV-kernel
union can be reported as an effective lower-bound proxy, not measured LPDDR
bandwidth or proof of saturation.

| Diagnostic | Bandwidth/weight-traffic signature | Kernel/scheduler signature |
| --- | --- | --- |
| C1 fixed-depth timeline and optional counters | High target GEMV union plus rising L2 sysmem-fill proxy and stable model-implied bytes per step | Material host/launch gaps, low eligible warps, or low achieved occupancy; no direct LPDDR-ceiling claim is available |
| Matched C1/C2/C8 timelines | Similar target-kernel time or proxy traffic per pass but more useful tokens per pass as concurrency rises | Smaller host gaps or better kernel occupancy as concurrency rises |
| Continuous decode steps 1 versus 2 | Little change if target kernels/traffic dominate | Wall/TPS gain with unchanged output and no memory change implicates scheduler round trips |
| Decode CUDA graph on versus off | Small effect if launches are already amortized | Large resident-C1 regression when disabled implicates launch/capture benefits |
| Warm versus varied PLE rows | Warm page faults/NVMe reads near zero; varied rows add attributable reads and stalls | Similar I/O but unchanged decode bottleneck points back to target kernels |
| MTP off versus fixed depth | Target bytes/verification calls per useful token fall with accepted length | Gains instead track fewer launches or larger verification kernels |

Pin counter names, collection windows, profiler versions, and U-P-U overhead.
Use timestamp unions: blocking D2H overlaps preceding hash work and pageable
H2D API wall can overlap its GPU copy, so summing phase durations would
double-count the critical path. System page faults and disk counters need exact
process/phase attribution; raw traces, commands, paths, prompts, completions,
kernel strings and request identifiers remain private. Do not treat a profiler
run as a promotion pair.

## Practical optimization order

Until that attribution exists, use two parallel queues rather than reading the
following prerequisites as one total ranking. Items 1--3 and 6--9 are the
retained-SGLang product track; items 4--5 are the vLLM systems track:

1. keep a healthy server resident when interactive latency matters;
2. retain mapped PLE, lazy recurrent state, and MTP2 as the safe default;
3. finish the frozen low/no-thinking, chunk-size, and MTP2/MTP3 task-wall
   comparisons only after a clean host preflight;
4. on the direct-mmap vLLM branch, test the isolated live-token hash slice
   before scheduler ceilings or mmap worker tuning;
5. after that runtime baseline, compare the mixed-FP8 dense-projection
   checkpoint as a quality-first artifact arm with DeepGEMM disabled and the
   resolved CUTLASS path attested;
6. measure repeated long-prefix Radix reuse for multi-turn agents;
7. test continuous decode steps and the decode-graph causal control;
8. admit a separate C2-capable geometry for independent single-user fan-out;
   and
9. evaluate MTP off only as a cold short-task specialization, not as the
   resident decode default.

The ranked product-track protocols are in the
[single-user experiment backlog](qwen38-flash-next-single-user-next-experiments-2026-08-28.md).
The systems track is ordered exact vLLM mmap control, isolated live-token-width
patch, then matched mixed-FP8 artifact; its profiling lifetime remains
diagnostic rather than promotional.
No optimization may trade away exact task correctness, lifecycle integrity, or
the frozen memory/swap gates for a higher token rate.
