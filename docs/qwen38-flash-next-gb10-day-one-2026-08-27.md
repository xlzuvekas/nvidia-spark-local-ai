# Qwen3.8-Flash-Next on GB10: day-one literature review — 2026-08-27

> Update: the [day-two delta](qwen38-flash-next-gb10-day-two-delta-2026-08-28.md)
> records SGLang's subsequent SM121 safety restriction, vLLM's open direct-mmap
> implementation, and newer one-Spark community evidence. This report remains
> frozen at its original cutoff.

## Conclusion

The first day of public GB10 work converged on the same fit mechanism measured
in this repository: keep the large FP8 PLE lookup on NVMe and demand-page only
the selected rows. It also converged on MTP as the main single-stream decode
lever. The strongest new one-Spark report is a pinned community vLLM patch that
claims about 17 tok/s with MTP off and 27 tok/s with MTP2; those values are
numerically close to this repository's clean SGLang 16.663713-off and
30.123639-MTP3 measurements. The local control is near-matched rather than
token-identical: its MTP3/off arms encoded 1,610/1,590 aggregate prompt tokens.

That agreement is corroboration, not a backend comparison. The public vLLM
report uses a different engine, MTP depth, prompt, context and measurement
script and publishes no persisted per-run artifacts, repetitions, or
power/memory telemetry. vLLM's model-support and PLE-offload pull requests also
remain open. There is still no official, documented, validated one-GB10 NVFP4
lane in the vLLM recipe.

The practical decision is therefore:

- keep the clean SGLang MTP3/off and lazy-MTP2 C8 results as the local measured
  anchors;
- treat the community vLLM path as a high-value, literature-derived A/B target,
  not evidence already measured here; and
- if it is run locally, rebuild it under SparkBench's immutable-model,
  loopback/authentication, scalar-evidence and exact-owned-cleanup controls.

## Scope and source quality

This review was frozen on 2026-08-27. It distinguishes official recipe text
and upstream merge state from community code and self-reported forum numbers.
Only the local SparkBench results elsewhere in this repository are tracked
machine evidence.

| Source | What it establishes | What it does not establish |
| --- | --- | --- |
| [Official vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) | Dedicated image, vLLM 0.29+, MTP interface, PLE CPU-offload interface, recommended FP8 deployments on 4x GB300 and 8x H200 | A validated one-Spark NVFP4 or NVMe-PLE configuration |
| [vLLM model-support PR #53896](https://github.com/vllm-project/vllm/pull/53896) | Active upstream implementation and review history | Merged/tagged support; it remained open at the cutoff |
| [vLLM PLE-offload PR #53899](https://github.com/vllm-project/vllm/pull/53899) | Active PLE-offload implementation; its PR validation described BF16/FP8 offload, not NVFP4 | Merged/tagged support or a validated single-GB10 NVFP4 lane; it remained open |
| [`blazux/qwen3.8-Flash-DGX` at `82ed48d`](https://github.com/blazux/qwen3.8-Flash-DGX/tree/82ed48d373d8a2c03d142d203f07bce0a6b69125) | Inspectable one-Spark vLLM mmap patch, pinned repository and base-image digest, launch/smoke scripts, and author-reported single-GX10 numbers | A frozen model revision or persisted repeated benchmark/telemetry bundle |
| [SGLang issue #36558](https://github.com/sgl-project/sglang/issues/36558) | An open stock-image SM121 QSA-resolver failure report from a TP2 dual-Spark attempt | Failure of the pinned local XQA overlay, or proof that all SGLang GB10 routes fail |
| [Two-Spark SGLang forum report](https://forums.developer.nvidia.com/t/qwen3-8-flash-next-nvfp4-on-2x-dgx-spark-full-multimodal-70-tok-s-peak-47-typical/381428) | A detailed community recipe and claimed no-spec/MTP/graph observations on 2x GB10 plus ConnectX | A controlled repeated benchmark, a one-Spark result, or completed multimodal soak validation |

The NVIDIA forum is useful discovery material, but forum claims remain
self-reported unless their underlying artifacts and protocol are independently
reproduced. GitHub pull-request state is authoritative for whether a change is
merged, not for the performance of an unmerged branch on this machine.

## One-Spark vLLM mmap path

The community repository is frozen at commit
[`82ed48d373d8a2c03d142d203f07bce0a6b69125`](https://github.com/blazux/qwen3.8-Flash-DGX/tree/82ed48d373d8a2c03d142d203f07bce0a6b69125).
Its Dockerfile pins the upstream multi-architecture image index
`sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`;
the inspected Linux/aarch64 child is
`sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e`.
The image tag and repository are pinned, but the source commit represented by
the image labels was not identified.

The 478-line
[`vllm_ple_mmap.py`](https://github.com/blazux/qwen3.8-Flash-DGX/blob/82ed48d373d8a2c03d142d203f07bce0a6b69125/src/vllm_ple_mmap.py)
has SHA-256
`2bca73dd0f77e72937cdfc43312c3fc4d217847d4bb126cf3665bd8caa3108c8`.
It replaces the large PLE embedding parameter with a placeholder, opens the
FP8 safetensors regions through read-only NumPy memmaps, copies input IDs from
GPU to CPU, deduplicates and sorts row IDs, gathers through a CPU thread pool,
and copies the gathered FP8 rows back to the GPU. The custom operation is kept
outside piecewise CUDA graphs. This is a comprehensible monkeypatch, not a
fused GPU/NVMe kernel; the synchronous ID transfer and pageable return copy are
plausible remaining decode bottlenecks.

The repository reports one ASUS GX10, one request and 32K context:

| Community-reported metric | Claimed result |
| --- | ---: |
| Prefill | about 2,400-2,660 tok/s |
| Decode, MTP off | about 17 tok/s |
| Decode, MTP2 | about 27 tok/s |
| MTP2 acceptance | about 67% |

Its smoke script is a real measurement script, but the repository contains no
persisted request-level result bundle, repeated trials, confidence interval,
power trace or memory-pressure trace. The numbers should be reproduced, not
copied into the local benchmark table as if they came from SparkBench.

The model provenance is weaker than the code/image provenance. The downloader
runs `hf download "$MODEL"` without `--revision`, and the serving script selects
a cached `snapshots/*` entry rather than checking an immutable revision. A
future repository update or cache layout can therefore change the checkpoint
under the same command.

The launch script is also literature-only under this repository's operational
protocol: it force-removes a container by a fixed name, publishes on all
interfaces, provides no API key, and chooses the first matching cached
snapshot. It must not be run unchanged. A local test needs the exact Radix
revision, a verified read-only snapshot, canonical loopback, an ephemeral API
key, no implicit download, and cleanup tied to the exact owned container.

## Official and upstream vLLM boundary

The [official recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) specifies
vLLM 0.29+ and a dedicated mutable image tag. Its documented production lanes
use the official FP8 checkpoint on four GB300 GPUs or eight H200 GPUs, and its
MTP example uses three speculative tokens. PLE CPU offload keeps the lookup
table in host memory; that is useful on discrete-GPU systems but does not by
itself solve a Spark unified-memory capacity bound. The page-backed community
NVMe patch is a separate extension.

At the cutoff, [PR #53896](https://github.com/vllm-project/vllm/pull/53896)
for model support and [PR #53899](https://github.com/vllm-project/vllm/pull/53899)
for PLE offload were both open. The model-support PR's own table said offload
validation covered BF16/FP8 on GB200, while NVFP4 offload was not currently
supported. Day-zero images and PR branches can be useful experiment bases, but
they are not equivalent to a merged release or a supported one-Spark lane.

## SGLang reports and the local result

The open [SGLang issue #36558](https://github.com/sgl-project/sglang/issues/36558)
reports that the stock resolver rejected SM121 for its TRT-LLM sparse-decode
path and fell through to a CuTe route that failed compilation. Its reported
scope was a stock-image TP2 dual-Spark attempt. This repository independently
observed that resolver problem and uses a digest-pinned local overlay that
admits the installed XQA-capable path. The issue validates the existence of a
stock-routing gap; it does not invalidate the measured overlay result or prove
upstream support for it.

The [two-Spark forum recipe](https://forums.developer.nvidia.com/t/qwen3-8-flash-next-nvfp4-on-2x-dgx-spark-full-multimodal-70-tok-s-peak-47-typical/381428)
reports 2x GB10 with ConnectX, patched SGLang, TP2, MTP4 and CUDA graphs. For
one 200-token decode stream it claims 20 tok/s without speculation, 33 typical
and 55 warmed peak with MTP, then 47 typical and 70.2 peak with decode graphs.
Reported cold/code acceptance ranged from 0.36 to 0.56. The author said the
multimodal graph configuration was still soaking.

Those values are not comparable to the local one-Spark numbers as a scale-out
ratio: hardware count, interconnect, tensor parallelism, MTP depth, graph
configuration, prompt and warm state all differ, and the post publishes no
controlled repetitions. In particular, its single-stream 70.2 peak must not be
compared directly with this repository's 114.5755 aggregate C8 rate.

## Cross-source interpretation

| Observation | Local measured evidence | External corroboration | Boundary |
| --- | --- | --- | --- |
| One-Spark no-spec decode | 16.663713 tok/s, clean D256/C1 | community vLLM about 17 tok/s | Different engine, prompt/context and measurement protocol |
| MTP decode gain | clean SGLang MTP3 30.123639 tok/s, `1.807739x` off | community vLLM MTP2 about 27 tok/s at about 67% acceptance | Local inputs differ by 1.26%; external MTP depth differs and the public number has no repeated artifact bundle |
| Prefill | local SGLang 2,103-2,180 prompt tok/s on repeated-word 8K/32K | community vLLM about 2,400-2,660 tok/s at reported 32K | Synthetic prompt/locality and client timing differ |
| Eight-way service | local lazy MTP2 114.5755 aggregate tok/s; scalar gates passed | no matched one-Spark external C8 result | Local occupancy is an operator-log observation, not tracked counter evidence |
| PLE fit mechanism | exact 47.684 GiB FP8 payload in a verified read-only NVMe mmap | community vLLM maps source safetensors regions read-only | Different implementations; neither proves cold/varied-token quality or locality |

The close off/MTP rates do not show PLE disk bandwidth dominating these short
decode prompts and are consistent with target verification/weight traffic
being important. That is a weak inference, not a profiler result: both routes
use page-backed PLE and could share a common bottleneck. The superseding direct
study should measure a pinned vLLM arm against a newly built and admitted SM121
Triton SGLang comparator under the same prompt, token, cache, MTP, and telemetry
protocol. The existing SM121 TRT-LLM arm is historical only and must not be
rerun.

## Historical reproduction requirements

A local vLLM reproduction should be a new derived profile, not the community
script executed directly:

1. pin the Linux/aarch64 image child, community patch digest and exact Radix
   model revision;
2. resolve the cached snapshot explicitly, mount it read-only and forbid
   network acquisition during measurement;
3. retain loopback binding, ephemeral authentication and exact owned-container
   teardown;
4. start with MTP off to prove admission and capture memory/swap gates, then run
   a matched MTP2 lifetime;
5. use the same D256 prompt/repetitions as the clean SGLang confirmation and
   capture native cumulative proposed/accepted counters; and
6. publish only deterministic sanitized scalar evidence, keeping raw server
   logs, prompts, completions, request identifiers and local paths out of Git.

Until that experiment passes, the community vLLM numbers remain a useful
external reference and implementation lead, not a repository benchmark.
