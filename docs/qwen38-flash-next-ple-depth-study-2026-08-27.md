# Qwen3.8-Flash-Next matched PLE and NEXTN-depth study — 2026-08-27

## Question and result

This experiment asks two narrower questions on one DGX Spark / GB10:

1. how much short-context throughput changes when the trained PLE layer is
   explicitly omitted rather than served from the exact read-only FP8 NVMe
   mapping; and
2. whether the earlier NEXTN depth-one and depth-two ordering repeats in clean,
   counterbalanced lifetimes when recurrent-state geometry and concurrency are
   held constant.

The protocol and harness were frozen before measurement. All eight planned
lifetimes reached a terminal state. The two exact-answer quality arms completed
at strict 8/8, resolving the earlier timestamp-sensitive 3/4 result without
changing its validator. All four mapped depth-one/depth-two performance
lifetimes completed and validated. The mapped depth-three lifetime completed
all cases but crossed the frozen 14 GiB host-availability floor by 0.095 GiB
during its first measured request. The omitted depth-three lifetime is partial
because one C4 request and one C8 request ended early; their official case
aggregates are correctly withheld.

PLE omission changes model semantics. Its narrow quality pass and descriptive
speedups do not make it an interchangeable deployment.

## Results

### Replicated mapped-PLE depth one versus depth two

The ABBA order was depth 1A, depth 2A, depth 2B, depth 1B. Every case below
completed its requested tokens and passed validation. Values are aggregate
output tokens divided by full case wall time in tok/s; the two-lifetime mean is
the unit used for the depth comparison.

| Case | D1 A | D1 B | D1 mean | D2 A | D2 B | D2 mean | D2 vs D1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Warmed D256/C1 | 27.459 | 29.148 | 28.304 | 28.758 | 30.046 | 29.402 | +3.881% |
| Fresh C1 | 26.500 | 27.933 | 27.217 | 28.441 | 30.746 | 29.594 | +8.734% |
| Fresh C2 | 44.969 | 47.185 | 46.077 | 49.243 | 54.497 | 51.870 | +12.572% |
| Fresh C4 | 70.458 | 76.969 | 73.713 | 73.762 | 77.181 | 75.471 | +2.385% |
| Fresh C8 | 106.894 | 111.808 | 109.351 | 113.108 | 121.173 | 117.140 | +7.123% |

Depth two was faster in every measured cell. With only two independent
lifetimes per arm, this establishes a replicated local ordering, not a precise
population effect. The C4 gain is also smaller than the observed D1 replicate
range, while the largest mean separation was at C2. The scalar-only native
audit requests reported draft-acceptance rates of 0.882/0.889 for D1 and
0.801/0.798 for D2. Those one-request audits confirm the configured proposal
depth; they are not case or lifetime aggregates, and their denominators change
with maximum draft depth.

### Matched depth-three PLE mapping versus omission

This is one mapped lifetime followed by one omitted lifetime, not a replicated
causal estimate. The valid mapped cases all completed. Omission completed and
validated D256, C1, and C2, but one of 12 C4 requests stopped at 172/256 tokens
and one of 24 C8 requests stopped at 232/256. The exporter therefore publishes
no official aggregate for those two cases. Parenthesized values divide the
actually completed token count by case wall time only to diagnose the partial
run; they must not be compared as successful throughput measurements.

| Case | PLE mapped | PLE omitted | Omitted vs mapped | Status |
| --- | ---: | ---: | ---: | --- |
| Warmed D256/C1 | 32.221 tok/s | 31.286 tok/s | -2.900% | Both valid |
| Fresh C1 | 30.762 tok/s | 33.171 tok/s | +7.833% | Both valid |
| Fresh C2 | 53.230 tok/s | 55.050 tok/s | +3.419% | Both valid |
| Fresh C4 | 80.143 tok/s | — (89.870 descriptive) | — (+12.137% descriptive) | Omitted invalid: 2,988/3,072 tokens |
| Fresh C8 | 118.454 tok/s | — (139.508 descriptive) | — (+17.773% descriptive) | Omitted invalid: 6,120/6,144 tokens |

The two high-concurrency descriptive rates are inflated by doing less work and
cannot support an omission speedup claim. Even among valid cases, the sign
changes: omission was 2.9% slower on warmed D256 and 3.4--7.8% faster on fresh
C1/C2. Operationally, the startup-telemetry minima were 14.623 GiB mapped and
19.055 GiB omitted; first-request minima were 13.905 and 19.199 GiB. The mapped
startup measurement is safety-invalid, and this is one sequential pair, so
these observations are not a valid causal memory estimate. They also do not
imply that the 50 GB mmap payload is fully resident.

The mapped D3 point was above the D2 replicate mean in all five cells, but only
by 1.122% at C8. It is unreplicated and its first request breached the frozen
memory floor, so D2 remains the fastest replicated, safety-clean configuration
from this panel. The native audit acceptance rates were 0.671 mapped and 0.622
omitted, again scoped to one explicit audit request per lifetime.

### Quality-clean exact answers

Both quality lifetimes used thinking, low reasoning effort, temperature zero,
and the stable v2 prompt tags. The original strict exact-answer validator was
unchanged.

| Arm | Strict score | Status | Output tokens | Reasoning tokens | Case wall time |
| --- | ---: | --- | ---: | ---: | ---: |
| D3 PLE mapped | 8/8 | Complete | 682 | 636 | 22.247 s |
| D3 PLE omitted | 8/8 | Complete | 650 | 604 | 18.968 s |

This resolves the earlier synthetic exact-answer failure as a reproducible
quality configuration rather than a validator exception. The wall times are
not a clean speed comparison because the two arms generated different token
counts and reasoning trajectories. Passing four synthetic items twice also
does not establish broad semantic equivalence after PLE removal; the sustained
generation failures above are direct counterevidence to treating it as such.

### Safety and retained evidence

Seven lifetimes stayed above the 14 GiB host-availability floor. The mapped D3
performance run recorded 13.9045677185 GiB during its first request, so its
typed safety annotation sets `startup_measurement_valid=false`; its five case
measurements remain visible and valid. No owned server or container survived
between lifetimes.

| Lifetime | Server startup | Managed wall time | Minimum available | Swap growth |
| --- | ---: | ---: | ---: | ---: |
| Quality mapped | 594.750 s | 653.696 s | 14.322 GiB | +0.414 MiB |
| Quality omitted | 582.289 s | 634.916 s | 19.005 GiB | 0 MiB |
| D1 A mapped | 586.952 s | 823.595 s | 17.062 GiB | +0.039 MiB |
| D2 A mapped | 580.372 s | 803.846 s | 15.590 GiB | +20.953 MiB |
| D2 B mapped | 600.722 s | 812.236 s | 15.066 GiB | +14.398 MiB |
| D1 B mapped | 587.101 s | 814.498 s | 16.642 GiB | +9.500 MiB |
| D3 mapped | 578.296 s | 822.767 s | **13.905 GiB** | +6.738 MiB |
| D3 omitted | 576.373 s | 768.691 s | 19.055 GiB | 0 MiB |

Managed journal time totaled 6,134.245 seconds (1.704 hours), including startup,
first-request warmup, cases, the scalar native audit, and shutdown. Swap stayed
far below the 512 MiB growth gate, but it carried between lifetimes; the row
increments are order-confounded and are not independent configuration effects.
The omitted D3 wall-time reduction also includes its faster valid cells and two
early stops, so it is not a startup or end-to-end speedup estimate.

Sanitized scalar evidence:

- quality: [mapped](../evidence/runs/20260827T230652Z-qwen38-flash-next-nvfp4-mtp3-quality-v2-ple-mapped-sglang-qwen38-flash-next-sglang-quality-v2-b7248f63/summary.json) and [omitted](../evidence/runs/20260827T231756Z-qwen38-flash-next-nvfp4-mtp3-quality-v2-ple-omitted-sglang-qwen38-flash-next-sglang-quality-v2-02c1c626/summary.json);
- ABBA replication: [D1 A](../evidence/runs/20260827T232851Z-qwen38-flash-next-nvfp4-mtp1-c8-lazy-ple-mapped-sglang-qwen38-flash-next-sglang-ple-depth-c8-87836e9b/summary.json), [D2 A](../evidence/runs/20260827T234248Z-qwen38-flash-next-nvfp4-mtp2-c8-lazy-ple-mapped-sglang-qwen38-flash-next-sglang-ple-depth-c8-0c745ab5/summary.json), [D2 B](../evidence/runs/20260827T235628Z-qwen38-flash-next-nvfp4-mtp2-c8-lazy-ple-mapped-sglang-qwen38-flash-next-sglang-ple-depth-c8-0c745ab5/summary.json), and [D1 B](../evidence/runs/20260828T001017Z-qwen38-flash-next-nvfp4-mtp1-c8-lazy-ple-mapped-sglang-qwen38-flash-next-sglang-ple-depth-c8-87836e9b/summary.json); and
- depth-three comparison: [mapped](../evidence/runs/20260828T002411Z-qwen38-flash-next-nvfp4-mtp3-c8-lazy-ple-mapped-sglang-qwen38-flash-next-sglang-ple-depth-c8-c98f473f/summary.json) and [omitted](../evidence/runs/20260828T003801Z-qwen38-flash-next-nvfp4-mtp3-c8-lazy-ple-omitted-sglang-qwen38-flash-next-sglang-ple-depth-c8-5b30b3b6/summary.json).

## Frozen artifacts and omission contract

Every arm pins:

- `RadixArk/Qwen3.8-Flash-Next-NVFP4` revision
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`;
- recipe `hashd1ve/qwen38-flash-next-one-dgx-spark` revision
  `bf2b7c75870d3703730b6bd8f3bb93dc622c278d`;
- SGLang image digest
  `14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4`;
- QSA overlay SHA-256
  `e30566492e1502f94a4c7fed42d90b523bbb662580c628459e6e63c7b5263c75`;
  and
- the ablation-capable `qwen4_exp.py` SHA-256
  `bcdc2c86aa59784ffe27d53c8d214e56b6aa45c02b1d5841fd956d1f006d6030`.

Mapped arms use the exact 51,200,245,760-byte FP8 PLE payload and completion
marker already admitted by the native route. Omitted arms have no PLE mount,
cache, or offload flag. A fail-closed overlay accepts only the canonical
`sparkbench_omit_ple=true` sentinel, verifies the pinned PLE layout, constructs
no PLE module, and skips exactly the checkpoint's 138 PLE tensors. The same
overlay is mounted in the mapped control with the sentinel absent.

## Matched performance geometry

All performance profiles disable thinking and use temperature zero. The only
intended model-axis differences are NEXTN depth and the explicitly labeled PLE
semantic ablation. Each server lifetime uses:

| Setting | Frozen value |
| --- | ---: |
| Recurrent-state strategy | `extra_buffer_lazy` |
| Recurrent states | 32 |
| Static memory fraction | 0.85 |
| Total-token pool | 32,768 |
| Context cap | 4,096 |
| Chunked prefill / maximum prefill | 1,024 / 4,096 |
| Maximum running requests | 8 |
| Decode CUDA graphs | batches 1 through 8 |
| Prefill CUDA graphs | disabled |

The suite runs one warmed D256/C1 case and fresh C1, C2, C4, and C8 cases.
Every case requests 256 output tokens for three repetitions. Request prompts
derive from the public `ple-study-*` case ID rather than the model-derived
case ID, so mapped/omitted and depth arms receive byte-identical prompts while
their request IDs remain unique within a lifetime.

The lifetime order is counterbalanced for the replicated depths:

1. mapped PLE, NEXTN depth 1;
2. mapped PLE, NEXTN depth 2;
3. mapped PLE, NEXTN depth 2;
4. mapped PLE, NEXTN depth 1;
5. mapped PLE, NEXTN depth 3; and
6. omitted PLE, NEXTN depth 3.

Depths one and two therefore reach two independent lifetimes each in ABBA
order. Depth three supplies the matched PLE comparison under the same lazy C8
geometry; it is not a second depth-three replicate. Primary throughput is
completed output tokens divided by full case wall time. Retain per-request
latency, TTFT, token counts, validation, startup, memory, swap, and power, and
report replicate means plus individual lifetimes rather than pooling requests
as independent server replicates.

## Quality-clean exact-answer protocol

The earlier native run was partial because one code-reasoning answer was wrong,
not because of truncation or parsing. Its supposedly deterministic prompt also
embedded a timestamped request ID. Protocol v2 removes request IDs from prompt
text and uses stable repetition tags; frozen v1 plans retain their original
wording and nonce behavior.

Mapped and omitted depth-three quality profiles run separately with thinking
enabled, `reasoning_effort=low`, temperature zero, C1, and a 512-token limit.
The four embedded exact-answer items each run twice. A quality-clean arm must
pass all 8/8 responses under the unchanged exact validator. There is no retry,
majority vote, relaxed answer key, or status override. The mapped arm is the
completion gate for resolving the old synthetic failure; the omitted arm
measures whether the semantic ablation retains this bounded behavior.

## Safety and execution

Run only one inference configuration at a time from a committed clean harness.
Reject unrelated GPU/container work, implicit downloads, available-memory
descent below 14 GiB, or swap growth above 512 MiB. Every lifecycle must stop
its owned container before the next starts.

```bash
python3 prepare_sglang_overlays.py --prepare-ple-ablation
python3 prepare_sglang_overlays.py --verify-ple-cache

python3 sparkbench.py benchmark \
  qwen38-flash-next-nvfp4-mtp3-quality-v2-ple-mapped-sglang \
  --suite manifests/suites/qwen38_flash_next_sglang_quality_v2.toml

python3 sparkbench.py benchmark \
  qwen38-flash-next-nvfp4-mtp3-quality-v2-ple-omitted-sglang \
  --suite manifests/suites/qwen38_flash_next_sglang_quality_v2.toml

python3 sparkbench.py benchmark \
  qwen38-flash-next-nvfp4-mtp1-c8-lazy-ple-mapped-sglang \
  --suite manifests/suites/qwen38_flash_next_sglang_ple_depth_c8.toml
```

Run the mapped quality gate first and the omitted quality arm second; the latter
also supplies the first real cold-start proof for the omission loader before the
six performance lifetimes. Repeat the performance command with the exact
profile order above. Raw prompts, completions, reasoning, request IDs, logs,
paths, and commands remain ignored. Only deterministic scalar evidence from the
allowlisted exporter may be committed.
