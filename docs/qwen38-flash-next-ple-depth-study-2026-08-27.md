# Qwen3.8-Flash-Next matched PLE and NEXTN-depth study — 2026-08-27

## Question and status

This experiment asks two narrower questions on one DGX Spark / GB10:

1. how much short-context throughput changes when the trained PLE layer is
   explicitly omitted rather than served from the exact read-only FP8 NVMe
   mapping; and
2. whether the earlier NEXTN depth-one and depth-two ordering repeats in clean,
   counterbalanced lifetimes when recurrent-state geometry and concurrency are
   held constant.

The protocol and harness are frozen below before measurement. Results will be
added only from terminal, sanitized run evidence. PLE omission changes model
semantics; even a faster omitted arm is not an interchangeable deployment.

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
