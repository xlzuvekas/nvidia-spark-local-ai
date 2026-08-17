# Mixture-of-Experts Models on DGX Spark — 2026-08-17

This note surveys current sparse Mixture-of-Experts (MoE) models that can
plausibly run on one 128 GB NVIDIA DGX Spark, then separates models that are
ready for controlled measurement from previews that merely fit in memory.
Model-card claims and local measurements are identified separately. Exact
benchmark evidence remains under ignored `results/` paths; no weights,
prompts, completions, or credentials are committed.

## Executive Conclusion

Five new pinned profiles now have terminal local evidence. At the API layer,
Qwen3.6 35B-A3B reached 184.899118 aggregate tok/s at eight true serving
slots, Laguna XS 2.1 33B-A3B reached 77.532993 tok/s for validated P1 D256,
and Laguna S 2.1 118B-A8B reached 22.817433 tok/s for the same validated P1
D256 shape. Nemotron 3.5 Lightning's validated MTP3 D128 rate was 95.646009
tok/s versus 58.416972 tok/s without the sidecar. In the separate same-binary
kernel panel, four roughly 30--35B sparse artifacts produced 61--83 tok/s at
TG1024 versus 10.964899 tok/s for the dense Qwen3.8 27B control. These are
bounded results from different profile geometries, not one interchangeable
leaderboard.

A separate, explicitly experimental NInfer port also reached GB10 GPU
execution for Qwen3.6 35B-A3B and dense Qwen3.8 27B NVFP4. In matched eager
Engine matrices, Qwen3.6 MTP3 raised mean TG1024 decode from 67.691870 to
140.190259 tok/s (+107.100585%); Qwen3.8 MTP3 raised it from 11.681340 to
21.221587 tok/s (+81.670823%). Pure-prefill means fell by roughly 1–2% in
both comparisons. This does not make upstream NInfer a supported Spark
runtime: stock source rejects `sm_121a`, the Qwen3.6 CUDA Graph attempt
failed before measurement, both measured profiles force 128-token prefill
chunks and eager decode, and no external numerical or semantic-quality
oracle was run. These are bounded experimental port measurements, not
stock-support or production-readiness claims.

The new 118B-A8B result is a successful memory and execution admission. The
exact three-shard Unsloth UD-Q4_K_XL quant totals 73,395,172,000 bytes
(68.354581 GiB), served in one 32,768-token slot, and retained at least
32.516830 GiB `MemAvailable` during the terminal core journal. Smoke and quick
were validation-clean. Core completed normally but is `partial`: D1024 ended
early in all five requests, strict JSON passed only 1/5, and exact-answer
quality was 3/4; D256, every queued C1/C2/C4/C8 request, tools, and the 16K
needle validated. The Poolside model card advertises a 1,048,576-token native
context, but that native limit was not tested here.

The recently released 39B-A5B preview is
[`ai9stars/G9v3-39A5B`](https://huggingface.co/ai9stars/G9v3-39A5B), pinned here
at revision `a3463bfc030a4824dd7c381d1571fdc2de0bd24c`. It is genuinely public,
Apache-2.0, and physically fits Spark: 39B total parameters, about 5B active
per token, 131,072-token context, and 77,939,453,856 bytes of BF16 weight
shards. It is not yet a defensible performance baseline. Its custom G9v3
architecture has no native implementation in the pinned optimized runtimes,
and the available community GGUF requires a non-upstream pinned llama.cpp
fork. It therefore remains an admission candidate rather than a measured row.
Gemma 4 26B-A4B is now the most valuable unmeasured multimodal and
quantization comparison.

The architectural hypothesis is plausible but narrower than “active
parameters determine speed.” Sparse routing reduces expert weight traffic for
a single token only when the runtime uses fused routing and expert kernels.
All weights still need residency; attention, shared experts, routing, KV
cache, and vision encoders remain dense. As concurrency rises, requests touch
a larger union of experts, so total-model bandwidth can reappear as the
bottleneck. The useful experiment is therefore a matched C1/C2/C4/C8 sweep,
not a comparison of parameter labels from model cards.

## The 39B-A5B Preview: G9v3

The pinned G9v3 configuration contains 38 transformer layers: one dense layer
and 37 MoE layers with 320 routed experts, 32 selected experts per token, and
one shared expert. The hidden size is 2,048 and each routed expert has a
512-wide intermediate projection. This is unusually high fan-out: “5B
active” does not mean that only a compact contiguous 5B-weight working set is
read under batching.

The official card claims recent Transformers, vLLM, and SGLang paths. The
current checkpoint uses remote custom code, however, and its pinned
[`modeling_g9v3.py`](https://huggingface.co/ai9stars/G9v3-39A5B/blob/a3463bfc030a4824dd7c381d1571fdc2de0bd24c/modeling_g9v3.py)
stores experts in an `nn.ModuleList`, calls `len()` and indexes that list, and
invokes experts individually. vLLM's documented Transformers MoE path replaces
such expert lists with a fused callable that accepts hidden states, top-k
indices, and weights. Because G9v3 continues to use list semantics after the
replacement point, incompatibility is a concrete static risk rather than a
mere missing registry entry. This has not been claimed as a live failure: it
needs an isolated construction/generation admission test.

The only currently reproducible GGUF route found was the community
[`linuxid10t/G9v3-39A5B-GGUF`](https://huggingface.co/linuxid10t/G9v3-39A5B-GGUF)
Q4_K_M artifact, 23,559,222,240 bytes, which requires a non-upstream llama.cpp
commit. That combination should remain quarantined: pin and inspect the fork,
then require coherent raw completion, think/no-think behavior, stop behavior,
JSON, and tool calls before recording any throughput headline.

## Spark-Fit Candidate Matrix

“Ready” below means a pinned practical quant plus an optimized runtime path
exists; the artifact column distinguishes first-party from community builds.
It does not mean every advertised context length or modality has been
validated on one Spark.

| Priority | Model | Total / active | Practical Spark artifact | License | Current verdict |
| ---: | --- | ---: | --- | --- | --- |
| 1 | [Nemotron 3.5 Lightning 30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) | 30B / 3B | Official NVFP4 21.56 GB + 1.35 GB [DSpark draft](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark), or official GGUF Q4_0 18.90 GB + 1.16 GB MTP | OpenMDW-1.1 | Measured matched GGUF baseline/MTP3; NVIDIA also validates its NVFP4 route on one Spark. |
| 2 | [Poolside Laguna XS 2.1](https://huggingface.co/poolside/Laguna-XS-2.1) | 33B / 3B | Official [Q4_K_M GGUF](https://huggingface.co/poolside/Laguna-XS-2.1-GGUF) 20.27 GB; official NVFP4 and DFlash also exist | OpenMDW-1.1 | Measured P1 at 32K: validation-clean fixed decode through D1024 and 77.533 tok/s at D256. |
| 3 | [Poolside Laguna S 2.1](https://huggingface.co/poolside/Laguna-S-2.1) | 117.6B / 8.5B (118B-A8B) | Tested Unsloth three-shard UD-Q4_K_XL, 73.395 GB | OpenMDW-1.1 | Measured P1 at 32K: D256 passed at 22.817 tok/s with 32.517 GiB minimum available memory; 1M native context remains untested. |
| 4 | [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) | 25.2B / 3.8B | NVIDIA NVFP4 18.8 GB; Google QAT Q4 14.4 GB; [Unsloth GGUF/MTP](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | Apache-2.0 | Highest-value text, vision, and quantization cross-check. |
| 5 | [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) | 30B / 3B | [Unsloth GGUF](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF) MXFP4_MOE 16.97 GB or Q4_K_M 18.31 GB | MIT | Mature text/agent speed control; use current post-loop-fix artifacts only. |
| 6 | [Liquid LFM2-24B-A2B](https://huggingface.co/LiquidAI/LFM2-24B-A2B-GGUF) | 24B / 2.3B | First-party Q4_K_M 14.42 GB | LFM Open License 1.0 | Low-cost speed floor with day-one llama.cpp/vLLM/SGLang support. |
| 7 | [Arcee Trinity Mini](https://huggingface.co/arcee-ai/Trinity-Mini-NVFP4) | 26B / 3B | First-party NVFP4 17.16 GB or Q4_K_M 15.94 GB | OpenMDW-1.1 | Compact, low-risk control. |
| 8 | [Mistral Small 4 119B-A6B](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) | 119B / 6.5B | Official NVFP4 70.8 GB or Unsloth GGUF 58–74 GB | Apache-2.0 | Fits, but official serving recipes use more than one GPU; TP1 is admission-first. |
| 9 | [Leanstral 1.5 119B-A6B](https://huggingface.co/mistralai/Leanstral-1.5-119B-A6B) | 119B / 6.5B | Community 67.14 GB NVFP4 GGUF | Apache-2.0 | Useful code/Lean/math reproduction target, not a general-chat control. |
| 10 | [Step 3.7 Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF) | 198B / about 11B | Official 102–112 GB GGUF plus optional projector | Apache-2.0 | Scientifically compelling but leaves little headroom and needs StepFun's runtime branch. |
| 11 | [Nemotron Labs Puzzle 75B-A9B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4) | 75B / 9B | About 47.6 GB NVFP4 + 5.9 GB MTP | OpenMDW-1.1 | Fits, but NVIDIA recommends TP2/TP4; single-Spark TP1 is an admission experiment. |

Kimi Linear 48B-A3B remains interesting for 1M-token hybrid-attention work,
but its first-party BF16 weights are roughly 98 GB and practical Q4 artifacts
are community-produced. Existing Qwen3.6 35B-A3B and GPT-OSS 120B-A5.1 results
are better local controls than another immediate download.

NInfer is not counted as a stock-ready Spark runtime in this matrix. Its
pinned upstream source is deliberately RTX 5090/`sm_120a`-only. The local
`sm_121a` experiment below proves a constrained eager path can execute on
GB10, but it remains a three-file, non-upstream port with an unresolved CUDA
Graph admission failure.

## Benchmark Admission and Comparison Rules

1. Run a non-speculative semantic smoke first: ordinary chat, strict JSON, and
   one tool call. Empty visible content, prompt echo, malformed stops, or
   parser failure blocks performance promotion.
2. Freeze the same target artifact, runtime, sampling, context, KV type, slot
   count, and prompts for baseline versus MTP/DSpark/DFlash.
3. Require persisted, positive drafted/proposed/accepted counters for every
   speculative server lifetime. An enabled flag is not evidence of activity.
4. Measure unique-request C1/C2/C4/C8 bursts. Repeated prompts can activate
   prefix or speculative replay caches and exaggerate throughput.
5. Report aggregate output tokens per case wall time as primary. Label
   llama.cpp `llama-bench` token-generation results as model-forward/kernel
   measurements, not API serving or answer-quality results.
6. Separate visible final content from reasoning. A fixed-token length pass is
   not instruction-following evidence.
7. Keep context scaling in the matrix. Sparse expert traffic does not remove
   dense attention/KV cost, and long prompts can reverse short-decode rankings.

## Pinned Reproducible Profiles

The repository pins five new profiles used by the terminal evidence below:

- `nemotron35-lightning-30b-a3b-q4-0-llamacpp`
- `nemotron35-lightning-30b-a3b-q4-0-llamacpp-mtp3`
- `laguna-xs21-33b-a3b-q4-k-m-llamacpp`
- `laguna-s21-118b-a8b-ud-q4-k-xl-llamacpp`
- `qwen36-35b-a3b-ud-q4-k-xl-llamacpp-p8`

The Nemotron pair uses the official
[`ggml-org` snapshot](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/tree/9d425fe18d84ab04da6aabb757d2e2807083d054),
including the exact Q4_0 MTP sidecar. The repository currently exposes Q4_0
files even though an NVIDIA recipe mentions a Q4_K_M alias; the manifest pins
what actually exists rather than relying on that alias. Both profiles use the
same llama.cpp b10453 binary and one 40,960-token slot. The MTP profile changes
only sidecar identity and maximum draft length three.

The original matched Nemotron reproduction starts with:

```bash
python3 sparkbench.py fetch nemotron35-lightning-30b-a3b-q4-0-llamacpp-mtp3
python3 sparkbench.py benchmark nemotron35-lightning-30b-a3b-q4-0-llamacpp \
  --suite manifests/suites/smoke.toml --fail-fast
python3 sparkbench.py benchmark nemotron35-lightning-30b-a3b-q4-0-llamacpp-mtp3 \
  --suite manifests/suites/smoke.toml --fail-fast
```

Those smoke gates and the subsequent measured runs are now terminal. The same
smoke-first rule still applies when reproducing them on another host; the
Laguna and Qwen profiles have their own exact pins and geometry documented in
their measured-result sections.

<!-- BEGIN LOCAL MEASURED RESULTS 2026-08-17 -->
## Local Measured Results: Nemotron 3.5 and the MoE Kernel Panel

The admission run changed the Nemotron recommendation from a paper candidate
to a measured one. On this Spark, the official Q4_0 target served coherent
chat, strict JSON, and tool calls with and without its Q4_0 MTP sidecar. MTP3
substantially accelerated short fixed-length decode, but made all three
repeat-prefill measurements slower. A separate same-binary `llama-bench`
panel then showed a much larger generation-rate gap between one dense 27B
control and four roughly 30--35B sparse models. The bracket control drift was
only -0.328136%, so that gap is not explained by machine drift during the
sweep.

### Exact Pins and Geometry

All server and kernel measurements used llama.cpp b10453 at commit
`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`. The `llama-server` binary was
`sha256:ae1bd49f869ff3397b2a5d757fcf010c6eaaf16c4e3071a15861312defcd4e40`;
the separate 51,559,272-byte `llama-bench` binary was
`sha256:cc16b06acc899a8fa4f1231c341abec5eb27b7f96a18a57ec75a8703e46ff3fc`.
The host inventory recorded one NVIDIA GB10, driver 580.142, compute
capability 12.1, and 128 GB unified memory. The kernel panel recorded 20 host
threads.

The Nemotron server pair pinned
`ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` revision
`9d425fe18d84ab04da6aabb757d2e2807083d054`. The target
`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` is exactly
18,898,091,584 bytes with
`sha256:61f87e75974e4b535dcdf9aad056541a9514f1dfa4538b463b081d19b7a00e3c`.
The MTP file is exactly 1,155,907,520 bytes with
`sha256:19f964207d5236dc88662686f00604a5494974c23fb04dd16a5ad7b2eebbd5b4`.
Both used full GPU offload, flash attention, fit disabled, batch/microbatch
8,192/512, Q8_0 K/V caches, one 40,960-token slot, Jinja, and reasoning off.
The MTP profile added only `draft-mtp`, maximum draft length three, and full
draft offload.

### Nemotron Endpoint Results

The smoke pair is terminal `complete`: ordinary chat, strict JSON, and a tool
call all passed in both runs. The quick and quality processes also reached
terminal `run_completion_status = "completed"`, but their summaries are
correctly `partial` because some output validations failed. The following are
the valid, suite-matched cases; rates are aggregate output tokens per case
wall time unless the metric says otherwise.

| Case | Baseline | MTP3 | MTP3 delta | Validation |
| --- | ---: | ---: | ---: | --- |
| Smoke D32 aggregate | 46.504561 tok/s | 61.769419 tok/s | +32.824434% | 1/1 and 1/1 |
| Smoke D32 median client decode | 63.810108 tok/s | 108.202021 tok/s | +69.568777% | 1/1 and 1/1 |
| Quick D128 aggregate | 58.416972 tok/s | 95.646009 tok/s | **+63.729829%** | 3/3 and 3/3 |
| Quick D128 median client decode | 64.776607 tok/s | 115.679572 tok/s | **+78.582327%** | 3/3 and 3/3 |
| Quick D128 median TTFT | 0.214388 s | 0.235025 s | +9.625874% | 3/3 and 3/3 |
| Repeat-prefill P256, median approximate | 1,248.054241 tok/s | 1,158.714936 tok/s | **-7.158287%** | metric-valid |
| Repeat-prefill P2048, median approximate | 3,026.065562 tok/s | 2,634.805307 tok/s | **-12.929669%** | metric-valid |
| Repeat-prefill P8192, median approximate | 3,847.724217 tok/s | 3,401.557290 tok/s | **-11.595606%** | metric-valid |
| 8K needle, output rate | 5.020662 tok/s | 4.624289 tok/s | -7.894834% | 1/1 and 1/1 |
| Synthetic exact-answer quality | 3/4, 75% | 3/4, 75% | no change | both below gate |

MTP also improved the smoke JSON case from 36.939639 to 44.144483 aggregate
tok/s (+19.504371%) and the smoke tool case from 35.306306 to 47.507684
tok/s (+34.558639%); both capability validations passed. Those are single
short requests, not stable throughput estimates. The 8K needle was recovered
by both profiles, but MTP's median TTFT was 2.490122 seconds versus 2.194456
seconds (+13.473335%).

The speculative evidence is native, positive, and depth-valid in every MTP
server lifetime:

| Suite | Drafts | Draft tokens | Accepted | Acceptance | Mean accepted length | Accepted positions 1/2/3 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Smoke | 26 | 77 | 63 | 81.818182% | 3.423077 | 23 / 21 / 19 |
| Quick | 398 | 1,174 | 903 | 76.916525% | 3.268844 | 347 / 295 / 261 |
| Quality | 11 | 32 | 24 | 75.000000% | 3.181818 | 11 / 8 / 5 |

Every row reached and accepted position three. These Prometheus counters are
cumulative across the whole server lifetime, including the fixed first
request and any suite warmups; they are not measured-case-only acceptance
rates. `mean accepted length` is `1 + accepted tokens / drafts`, so it
includes the target-verified token and can exceed the configured number of
draft tokens.

Results that failed validation are deliberately not promoted. In the quick
baseline, C2 passed 2/4 requests and C4 passed 4/8; MTP3 passed 3/4 and 6/8.
The other requests stopped early, so summary aggregate rates are null and raw
completion-token/wall-time quotients would compare unequal work. The separate
baseline D256 run is likewise only an admission trace: one of five requests
reached 256 tokens, while four stopped at 208, 52, 189, and 52 tokens. Its
per-request client decode estimates clustered from 63.423809 to 64.210985
tok/s, but the case has no valid aggregate headline. Both quality runs missed
the same code-reasoning item and passed arithmetic, instruction-following,
and logic.

The decode comparisons are preliminary rather than a confidence interval.
The suites use unique request identifiers, producing small prompt-token
differences between profiles: 79 versus 82 tokens in smoke D32 and 249 versus
255 total tokens in quick D128. D32 has one request and D128 has three. The
large decode gain, slower repeat-prefill path, and validated speculative
counters are clear enough to justify a longer matched confirmation; they do
not yet establish an all-workload MTP speedup.

The exact ignored evidence directories are:

- `results/20260817T153330Z-nemotron35-lightning-30b-a3b-q4-0-llamacpp-smoke-54ac8917`
- `results/20260817T153401Z-nemotron35-lightning-30b-a3b-q4-0-llamacpp-mtp3-smoke-3dc41213`
- `results/20260817T153543Z-nemotron35-lightning-30b-a3b-q4-0-llamacpp-quick-8070a80d`
- `results/20260817T153652Z-nemotron35-lightning-30b-a3b-q4-0-llamacpp-mtp3-quick-8e76c3f2`
- `results/20260817T153757Z-nemotron35-lightning-30b-a3b-q4-0-llamacpp-chat-quality-ba988a47`
- `results/20260817T153826Z-nemotron35-lightning-30b-a3b-q4-0-llamacpp-mtp3-chat-quality-ec2fb937`
- `results/20260817T153432Z-nemotron35-lightning-30b-a3b-q4-0-llamacpp-llamacpp-mtp-depth-e5459c8f`

### Bracketed Same-Binary Bandwidth Panel

The kernel panel used the same `llama-bench` binary and, for every admitted
artifact, `--offline`, full GPU offload, flash attention, batch/microbatch
8,192/512, Q8_0 K/V, and five repetitions. It measured prompt processing at
128, 1,024, 4,096, and 16,384 tokens, then fixed generation at 256 and 1,024
tokens. Values below are arithmetic mean tokens/second recomputed from the
persisted samples; no server, chat template, sampler, speculative decoder, or
semantic validator is involved. File GB below is decimal bytes divided by
one billion; exact byte counts and hashes follow the table.

| Artifact | File GB | P128 | P1024 | P4096 | P16384 | TG256 | TG1024 | TG1024 / dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unsloth Qwen3.8 dense 27B UD-Q4_K_XL | 17.923 | 687.948 | 778.083 | 775.069 | 734.378 | 11.003 | 10.965 | 1.000x |
| Qwen3 30B-A3B Q4_K_M | 18.557 | 1,201.413 | **2,519.778** | 2,437.042 | 2,032.064 | **83.775** | **82.559** | **7.529x** |
| Unsloth Qwen3.6 35B-A3B UD-Q4_K_XL | 22.854 | 1,181.712 | 2,104.189 | 2,071.894 | 1,987.249 | 62.153 | 61.563 | 5.615x |
| Nemotron Cascade 2 31B-A3.5B Q4_K_M | 24.272 | 866.084 | 2,061.495 | 2,101.354 | 2,029.808 | 72.082 | 71.727 | 6.542x |
| Nemotron 3.5 Lightning Q4_0 | 18.898 | 1,053.799 | 2,392.019 | 2,414.303 | **2,360.614** | 66.773 | 66.945 | 6.105x |

The opening dense-Qwen D256 control was 11.003968 +/- 0.043641 tok/s; the
closing control was 10.967860 +/- 0.040222 tok/s. Closing/opening drift was
**-0.328136%**, well inside the 5% gate. The same dense row inside the panel
was 11.002524 tok/s at D256 and 10.964899 tok/s at D1024. Because the control
bound is tiny relative to the 5.6--7.5x sparse/dense generation gaps, the
table reports measured rates directly rather than applying a drift
normalization.

This is strong local evidence for the bandwidth hypothesis: at similar
resident artifact sizes, the sparse models execute generation much faster
than the dense control. It is not an architecture-only causal estimate. The
models differ in active width, layer composition, quantization family
(Q4_K_M, UD-Q4_K_XL, or Q4_0), and kernel path, and `llama-bench` says nothing
about instruction following or answer quality. Its reported metadata also
calls the two Nemotron artifacts 31B-A3.5B even though NVIDIA names Lightning
30B-A3B; the table preserves each product identity and uses the benchmark's
metadata only to explain that discrepancy.

Two cached candidates failed direct b10453 admission before producing a row:
the 19,019,269,280-byte GLM-4.7-Flash Q4_K_M blob
`sha256:9eba2761cf0b88b8bc11a065a7b5b47f1b13ce820e8e492cb1010b450f9ec950`
and the 23,869,179,840-byte Qwen3.5 35B-A3B Q4_K_M blob
`sha256:900dde62fb7ebe8a5a25e35d5b7633f403f226a310965fed51d50f5238ba145a`.
The logs only establish `failed to load model`; they do not diagnose the
cause, and both models remain candidates for their supported runtimes.

The admitted artifact pins were:

- Unsloth Qwen3.8 revision `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`,
  17,923,394,624 bytes,
  `sha256:bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`.
- Ollama `qwen3:30b-a3b-instruct-2507-q4_K_M` revision `19e422b02313`,
  18,556,685,856 bytes,
  `sha256:78b329e716e7e9775973d392cd132b1f1ff1c8287a992887caeb6fd6c56ba9cc`.
- Unsloth Qwen3.6 revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`,
  22,853,663,008 bytes,
  `sha256:55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1`.
- Ollama `nemotron-cascade-2:latest` revision `e0705e3fe8f7`,
  24,272,433,056 bytes,
  `sha256:9e0c827cfd6a6d000032be3da3d0914668b0c1112977e927186d29c4487466c4`.
- The exact Nemotron 3.5 Lightning target pin documented above.

Panel evidence is under `results/moe-bandwidth-20260817T1539Z/`:
`control-pre.json`, `panel.json`, `panel-part2.json`, `cascade.json`,
`nemotron35.json`, and `control-post.json`, with corresponding logs. The
first panel file contains 12 complete objects before the GLM load failure;
the second contains six before the Qwen3.5 failure. Their outer arrays are
therefore intentionally unterminated. This audit parsed those completed
objects by closing the arrays in memory only and left the raw evidence
unchanged.

### True Concurrent Serving: Qwen3.6 P8

The terminal core run at
`results/20260817T161005Z-qwen36-35b-a3b-ud-q4-k-xl-llamacpp-p8-core-1b6ef809`
adds an API-serving concurrency result to the forward-only panel. It used the
22,853,663,008-byte Unsloth Qwen3.6 35B-A3B UD-Q4_K_XL artifact at revision
`5bc3e238d916f48a861bac2f8a1990a0e9b7e98d` and
`sha256:55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1`.
The pinned b10453 server allocated 262,144 total context tokens across eight
parallel slots, exposing 32,768 tokens per request. It retained full GPU
offload, flash attention, fit disabled, 8,192/512 batch/microbatch, Q8_0 K/V,
Jinja, reasoning off, loopback-only offline serving, and no speculation.

All 75 concurrent D256 requests reached the token limit and validated. The
primary rate is total output tokens divided by case wall time; scaling and
linear efficiency are relative to the measured C1 case.

| Concurrency | Passed (tokens) | Aggregate tok/s | Scaling vs C1 | Linear efficiency | Median client decode / request | Median TTFT / E2E | Sampled average W | Output tok/sampled J |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 5/5 (1,280) | 56.474661 | 1.000x | 100.000% | 58.575010 tok/s | 0.175031 / 4.528423 s | 46.149 | 1.263286 |
| C2 | 10/10 (2,560) | 54.488351 | 0.964828x | 48.241% | 27.995190 tok/s | 0.357570 / 9.449834 s | 44.394 | 1.256409 |
| C4 | 20/20 (5,120) | 140.182241 | 2.482215x | 62.055% | 37.433976 tok/s | 0.410146 / 7.222377 s | 50.614 | 2.850036 |
| C8 | 40/40 (10,240) | **184.899118** | **3.274019x** | 40.925% | 24.309732 tok/s | 0.753942 / 11.125120 s | 55.523 | **3.395497** |

Scaling was not monotonic: C2 was 3.517171% below C1 aggregate, C4 was
157.270111% above C2, and C8 added 31.899102% over C4. The C8 result is a real
3.274x throughput gain, not an 8x gain; its per-request latency and decode
rate also show the batching tradeoff. A repeated, reordered sweep is needed
before treating the C2 discontinuity as a stable scheduler property.

The independent D256 case passed 5/5 with 56.936634 aggregate tok/s, 59.094379
median client-decode tok/s, and 0.167656/4.481339-second median TTFT/E2E. D1024
is not a valid aggregate: four requests reached 1,024 tokens, while one
stopped at 592, leaving 4,688 output tokens total. Its five per-request client
decode estimates were tightly bounded from 58.415094 to 59.499264 tok/s, but
the summary correctly suppresses aggregate and median decode headlines.

Five-request repeat-prefill measurements were metric-valid client-TTFT
approximations:

| Nominal prefill | Median approximate tok/s | Median TTFT |
| ---: | ---: | ---: |
| P128 | 933.930429 | 0.210936 s |
| P1024 | 1,777.503576 | 0.613782 s |
| P4096 | 2,292.083520 | 1.816688 s |
| P16384 | 2,472.527237 | 6.655134 s |

P32768 was correctly skipped before launch: the estimated 32,909 required
tokens exceeded the 32,768-token per-slot limit. The 16K needle passed 3/3
with 7.117562-second median TTFT. Structured JSON and tool calls each passed
5/5 at 39.028991 and 46.781328 aggregate tok/s. Exact-answer quality passed
4/4 (100%) across arithmetic, code reasoning, instruction following, and
logic. Vision, embeddings, and reranking were unsupported by this text
profile rather than attempted failures.

The process reached terminal completion with no run error, measurement
annotations, or invalid measurements. Its top-level status is `partial`:
D1024 failed output validation and P32768 is separately context-limited, but
the process itself completed normally.

Across 413 telemetry samples, minimum `MemAvailable` was 77.382736 GiB, swap
free remained 16,776,876 KiB, peak power was 80.30 W, peak temperature 83 °C,
peak GPU utilization 95%, and peak SM clock 2,515 MHz. The C8 case itself
averaged 55.523 W, peaked at 59.86 W, and retained 77.676559 GiB minimum
`MemAvailable`; the run therefore preserved substantial reported memory
headroom at eight slots.

### NInfer on GB10: Experimental `sm_121a` Eager Port

[NInfer](https://github.com/Neroued/ninfer) is **not stock-supported on DGX
Spark** at the pinned upstream commit
`5f45a26f81b6a15805a3d4d09d5c3d60f420b210`. Upstream deliberately accepts
only `CMAKE_CUDA_ARCHITECTURES=120a`, requires compute capability 12.0 at
runtime, and documents an RTX 5090 plus CUDA 13.1 or newer. Two unmodified
configure probes establish distinct local blockers. Supplying `121a` was
rejected by the exact architecture gate before compiler detection; leaving
the default at `120a` detected host CUDA 13.0.88 and was then rejected by the
minimum-version gate. The 155-byte and 942-byte logs are respectively
`sha256:a5fb0838122b597e1506906ed3a4b727050eeec80a234c3392c613b61a2546b2`
and
`sha256:cce35daba28cd6163247a064a9a5ffbe9a2e62df22d2170887d254eb3abb44ae`.
These are configure-time admission failures, not zero-throughput benchmark
rows.

The measured route was an explicitly experimental working-tree port. Relative
to that upstream commit it modified exactly three files: the CMake compile
gate changed from `120a` to `121a`, the Qwen3.6 runtime device gate changed
from compute capability 12.0 to 12.1, and the Dockerfile pinned arm64 CUDA
13.1.2 base digests and packaged `ninfer_bench`. The complete binary diff is
represented by the SHA-256 of `git diff --binary` over those files:
`6903090db8a04784147f858f0e29444579032a2da8a3f4a4737d86bd3563f6be`;
the Dockerfile itself is
`sha256:5989230b3c3eb52b18c6b2891f6283e4aa1bf08aae5d05b6d919121fb904b761`.
No kernel schedule, memory planner, or numerical implementation was retuned
by this patch.

The container build used CUDA compiler 13.1.115 and completed both the apps
and the separate product benchmark. The resulting local image is
`linux/arm64`, 3,080,276,171 bytes, and has image ID
`sha256:d916d04747b1098df1953cf4ce5065a2947b7f445c815af16693172c58bea83f`.
Its labels preserve the upstream revision and
`io.sparkbench.port=experimental-sm121a`. The devel and runtime bases were
pinned to
`sha256:369b61cfc40a36830d246f0a14004a716aee72f4c520ee719b73d2b47e8bd9ce`
and
`sha256:6c8bcccb947a781668a7bfed5357a316456131cf5273a72db7fd014db0a0f81a`.
The apt repositories still floated at build time, so a complete runtime
package inventory was retained. A successful build and packaged executable
establish build/package admission only; they do not establish GPU execution.

The exact native artifact was
[`neroued/Qwen3.6-35B-A3B-NInfer`](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer)
revision `3d960f7b670ea706105571a822334a1f09759271`, file
`qwen3_6_35b_a3b.ninfer`, 22,783,246,080 bytes (21.218551 GiB), and
`sha256:1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2`.
Independent artifact inspection resolved identity
`qwen3.6-35b-a3b/groupwise-int` with 940 objects: 934 tensors and six
resources. This is NInfer's groupwise-integer artifact, not its NVFP4 path
and not the GGUF measured in the llama.cpp sections.

The raw-token corpus contained exactly 65,536 whitespace-delimited token IDs
in 333,620 bytes with
`sha256:27e4f63c17efe3f89b5cf278b3b1a42a737316ed4044d7d0d1d52437059d1002`.
Its schema-v1 manifest records local `Qwen/Qwen3.6-27B` tokenization, no added
special tokens, no chat template, and a curated bank tiled, rotated, and
truncated to the requested length. It gives deterministic token lengths; it
is not a chat, instruction-following, or semantic-quality corpus.

#### GPU Execution Gate and CUDA Graph Blocker

The first GPU claim comes from a schema-11 `ninfer_bench` report, not from the
image build or a `--help` invocation. The combined gate ran `pp96+tg1` and
`pp128+tg1` with one repetition, no warmup, maximum context 256,
128-token prefill chunks, INT8 group-64 KV, MTP disabled, and CUDA Graphs
disabled. Both cases generated exactly the two requested output tokens: one
from begin/prefill and one timed decode token. Both persisted finite,
positive prefill and decode durations while loading target
`qwen3_6_35b_a3b`, weights `groupwise-int`, on `NVIDIA GB10` with CUDA
runtime and driver API 13.1. This exercises the audited 96-token, 128-token,
and one-token decode routes. Its one cold repetition is an execution gate,
not a throughput estimate; its rates are deliberately not promoted.

The gate report is
`experimental-sm121a-admission-mtp0.json`,
`sha256:77cdf34f4fead57aeaf5d3e8766f2b24f07bc9019adba2105b5d7fe861385caa`.
It establishes that these exact kernels complete operationally on GB10. It
does not compare their values against a reference implementation.

The first 8K MTP0 matrix then failed during CUDA Graph preparation, before a
test or report was produced. NInfer recorded 1,107,603,456 bytes consumed
versus a 12,582,912-byte planned allowance. This is an admission failure of
the tested CUDA Graph configuration, not a zero-throughput result. The
matched performance pair therefore used eager decode and must not be mixed
with NInfer's upstream RTX 5090 CUDA Graph numbers.

#### Matched Eager MTP0 versus MTP3

Both terminal reports used the same artifact, corpus, `max_context=8192`,
128-token prefill chunks, INT8 group-64 KV, eager decode, five measured
repetitions after one discarded warmup, and exact requested-output checks.
MTP3 changed only the speculative window to three and selected the optimized
proposal head. Values are arithmetic means plus sample standard deviation;
the delta is MTP3 relative to MTP0. TG tests use an untimed one-token seed
prefill and report the requested `G` committed decode tokens.

| Engine metric | MTP0 mean +/- SD tok/s | MTP3 mean +/- SD tok/s | MTP3 delta | MTP3 accepted / drafted |
| --- | ---: | ---: | ---: | ---: |
| PP128 prefill | 1,309.040131 +/- 12.019478 | 1,290.612605 +/- 13.985590 | -1.407713% | n/a |
| PP1024 prefill | 1,146.373396 +/- 6.987050 | 1,127.810318 +/- 7.003477 | -1.619287% | n/a |
| PP4096 prefill | 1,032.871289 +/- 1.825106 | 1,022.595179 +/- 1.896370 | -0.994907% | n/a |
| TG256 decode | 67.821291 +/- 0.144205 | **123.231194 +/- 0.241697** | **+81.699866%** | 875 / 1,200 (72.916667%) |
| TG1024 decode | 67.691870 +/- 0.185437 | **140.190259 +/- 0.158602** | **+107.100585%** | 3,740 / 4,135 (90.447400%) |
| PP2048+TG256 prefill | 1,055.012144 +/- 3.238117 | 1,054.265485 +/- 3.209199 | -0.070773% | n/a |
| PP2048+TG256 decode | 66.353294 +/- 0.209378 | **137.685571 +/- 0.461116** | **+107.503746%** | 940 / 1,010 (93.069307%) |

Every pure-prefill repetition produced its one required begin token. Every
TG256 and combined repetition produced 257 total output tokens, including
256 timed decode tokens; every TG1024 repetition produced 1,025 total,
including 1,024 timed decode tokens. Across the five measured MTP3
repetitions, TG256 recorded 400 speculative rounds and five fallback steps;
TG1024 recorded 1,380 rounds and no fallback; the combined case recorded 340
rounds and no fallback. Pure-prefill cases correctly drafted no tokens.

The baseline and MTP3 reports came from separate process lifetimes in fixed
order rather than an alternated bracket. The roughly 1% prefill differences
are therefore descriptive, not a causal estimate. The much larger decode
differences coincide with positive, depth-three speculative counters and
exact output counts, but remain single-request raw-Engine throughput rather
than OpenAI API serving or answer-quality measurements.

NInfer reported 29.154518 GiB available device memory after MTP0 startup and
28.365582 GiB after MTP3 startup. The process-level telemetry sidecars held
183 MTP0 samples and 114 MTP3 samples. MTP0 averaged 47.516448 W, peaked at
55.99 W, and peaked at 75 °C; MTP3 averaged 54.321228 W, peaked at 58.40 W,
and peaked at 78 °C. These samples cover the complete process lifetimes, not
case-scoped energy, so no joules/token comparison is inferred.

One separate greedy CLI parity probe produced six tokens under both MTP0 and
MTP3, and both raw stdout files had the same
`sha256:70c411c2d585fe0a111a53bd29c21f46dccd2034988f0a89ed7dd62007d00ede`.
The raw completion and prompt remain ignored and uncommitted. This is useful
internal speculative parity for one sample; it is not an external numerical
oracle, a semantic-quality screen, or proof that other prompts match.

The exact ignored evidence root is
`results/ninfer-experimental-sm121a-20260817T181134Z/`. Important evidence
hashes are:

| Evidence file | SHA-256 |
| --- | --- |
| `build.log` | `f730b029d63756dbc1221bf9f4efd9b564a1846684f57b86dc34dca0212b6dce` |
| `image-inspect.json` | `c5688cd9f9c98f68afd236ed4cd0892de47e0ca6f302b6af5aba1879eccb3fee` |
| `packages.tsv` | `24f9d9ed04a206c4a8254d4cec7386b5823a5beab28849c0662a27d43e07fe8a` |
| `artifact-inspect.json` | `864f2da3b264581c523c852c06ba2b584502fd15340b11ef179b7791310a379f` |
| `native-mtp0.log` | `6e79105f2c6fdc248a9008607ffc516a5a357cad046fd9681fcbd26bc84606ee` |
| `experimental-sm121a-native-eager-mtp0.json` | `20223a882aa3f410c51e2cf64ee909b003b0ca393ae9718cfebcb56214101bf9` |
| `experimental-sm121a-native-eager-mtp3.json` | `94642711e937c185a376a047621bc227330021354183a8d1119aec3f8066340c` |
| `native-eager-mtp0-telemetry.csv` | `9cf23b6507f316cf5ece9e9f66ef6d2ac13a50cfd0d74e1d1e867c8f956b79a0` |
| `native-eager-mtp3-telemetry.csv` | `79cb0804e26f2dcb55d146b5e231c3d9d586882ad7de12a40702754be0865465` |

The two stock configure logs remain separately under
`results/ninfer-gb10-20260817/`. Until the architecture port is upstreamed,
the CUDA Graph planner is fixed for GB10, and a reference/semantic validation
campaign passes, the appropriate status is **experimental eager port**, not
stock NInfer support.

#### Qwen3.8 27B NVFP4 on the Same Experimental Port

The same image and three-file source patch were then tested with NInfer's
official
[`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/tree/d6d0b3b61a38262e57217e64e7f44cf4ce98bda1)
artifact at revision
`d6d0b3b61a38262e57217e64e7f44cf4ce98bda1`. The single version-2 artifact
is 21,492,695,040 bytes (20.016632 GiB) with
`sha256:bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32`.
NInfer's independent inspector resolved identity `qwen3.8-27b/nvfp4`, target
`qwen3_8_27b`, and 1,124 objects: 1,118 tensors plus six resources. This is a
mixed profile rather than universal NVFP4: the inspector reports 112 NVFP4
tensors and 146 row-scaled FP8 tensors among its registered formats.

Two one-repetition execution gates preceded measurement. MTP0 completed
exact `pp96+tg1` and `pp128+tg1` routes; MTP3 completed exact
`pp128+tg32`, produced positive counters at all three draft positions, and
accounted for 32 committed decode tokens. Both loaded the expected artifact
on NVIDIA GB10 under CUDA runtime/driver API 13.1. These cold gates establish
the audited prefill and eager-decode routes only; their rates are not promoted
as performance measurements.

The terminal matched pair used `max_context=8192`, 128-token prefill chunks,
INT8 group-64 KV, eager decode, five measured repetitions after one discarded
warmup, and exact output-length checks. MTP3 changed only the draft window to
three and selected the embedded optimized proposal head. Values are arithmetic
means plus sample standard deviation; deltas are MTP3 relative to MTP0.

| Engine metric | MTP0 mean +/- SD tok/s | MTP3 mean +/- SD tok/s | MTP3 delta | MTP3 accepted / drafted |
| --- | ---: | ---: | ---: | ---: |
| PP128 prefill | 1,008.948447 +/- 8.785989 | 994.713142 +/- 5.377957 | -1.410905% | n/a |
| PP1024 prefill | 1,052.919599 +/- 3.684916 | 1,036.428662 +/- 1.921179 | -1.566210% | n/a |
| PP4096 prefill | 1,044.019498 +/- 1.570377 | 1,030.375858 +/- 2.076544 | -1.306838% | n/a |
| TG256 decode | 11.715064 +/- 0.062220 | **18.683347 +/- 0.073374** | **+59.481388%** | 610 / 1,995 (30.576441%) |
| TG1024 decode | 11.681340 +/- 0.012898 | **21.221587 +/- 0.042279** | **+81.670823%** | 2,765 / 7,050 (39.219858%) |
| PP2048 prefill | 1,036.622403 +/- 5.429863 | 1,024.551994 +/- 7.258133 | -1.164398% | n/a |
| PP2048+TG256 decode | 11.610403 +/- 0.025611 | **36.614015 +/- 0.178323** | **+215.355257%** | 940 / 1,010 (93.069307%) |

All 60 measured repetitions across the pair produced their exact requested
output counts. Across the 15 measured MTP3 decode-bearing repetitions, the
runtime recorded 3,355 speculative rounds, 10,055 drafted tokens, 4,315
accepted tokens, and ten fallback steps: 42.913973% overall acceptance, with
accepted-position totals 2,105/1,370/840. Acceptance was highly
workload-dependent. In particular, the deterministic corpus continuation in
the combined case accepted 93.069307% of proposals, so its 3.15x speedup is
not a general chat-throughput estimate.

The matched reports came from separate process lifetimes in fixed MTP0 then
MTP3 order, not an alternated thermal bracket. NInfer reported 67.579258 and
66.833233 GiB free after startup respectively; loaded GPU weights were
18.976276 and 19.729072 GiB. The extra optimized draft head accounts for 14
additional loaded tensors and approximately 0.753 GiB of weights.

The fixed corpus supplies raw token IDs and exact lengths without a chat
template, sampler-quality screen, or semantic validator. A separate greedy
CLI probe therefore checked only internal speculative parity: MTP0 and MTP3
produced the same eight token IDs and byte-identical stdout with
`sha256:8bde74ff31522c083c52ecf8faaab10280142d15dd991b4d59b773d4a1d64d10`.
The raw prompt and completion remain ignored and uncommitted. This one-sample
match is not an external numerical oracle or a broad quality result.

The MTP0 process lifetime had 854 telemetry samples, averaged 47.276897 W,
peaked at 64.67 W and 83 °C; MTP3 had 480 samples, averaged 59.407958 W,
peaked at 63.16 W and 80 °C. Sampling spans load, warmup, and measurement, so
these are lifecycle resource bounds rather than case-scoped energy results.
The exact ignored evidence root is
`results/ninfer-qwen38-nvfp4-sm121a-20260817T200147Z/`; its `SHA256SUMS`
ledger has
`sha256:ee97458816895a5b6d625bde4201c68265251e95e95614b9d496c4ecb1003c21`
and verifies the artifact inspection, gates, matched reports, logs, telemetry,
and private parity artifacts. No NInfer source or branch was pushed upstream.
The reusable exact port diff is preserved in this repository as
[`patches/ninfer/5f45a26f-sm121a.patch`](../patches/ninfer/5f45a26f-sm121a.patch).

### Poolside Laguna XS 2.1: Official GGUF P1

Laguna XS 2.1 also cleared admission. All three runs pinned Poolside's
20,274,300,032-byte `Laguna-XS-2.1-Q4_K_M.gguf` at revision
`1a37c0a5fb8c7a18e6106decb6be6327d1b63fa6` and
`sha256:1ac7079101fca5a6df8c5a7523a3c30ea7d1c0e4b1258090e7d6d4039287f6cb`.
They used the same b10453 server pin documented above, full GPU offload,
flash attention, fit disabled, batch/microbatch 8,192/512, Q8_0 K/V, Jinja,
reasoning off, no speculation, and one 32,768-token slot.

This profile is `--parallel 1`. Every C2/C4/C8 case below is therefore a
burst queued through one serving slot, not simultaneous model execution and
not evidence of multi-slot throughput scaling. Its near-flat aggregate rate
and rising TTFT are the expected serialized shape.

The fixed-length decode cases all reached their limits and validated:

| Suite / case | Passed (tokens) | Aggregate tok/s | Median client decode | Median TTFT / E2E |
| --- | ---: | ---: | ---: | ---: |
| Smoke D32 | 1/1 (32) | 59.363118 | 81.808757 tok/s | 0.145783 / 0.524716 s |
| Quick D128 | 3/3 (384) | 75.226305 | 82.487642 tok/s | 0.149243 / 1.688868 s |
| Core D256 | 5/5 (1,280) | **77.532993** | 81.344412 tok/s | 0.154527 / 3.288373 s |
| Core D1024 | 5/5 (5,120) | 77.057868 | 78.030312 tok/s | 0.162860 / 13.272572 s |

The queued burst cases were also validation-clean. Quick used 64 output
tokens per request; core used 256:

| Suite / queued burst | Passed (tokens) | Aggregate tok/s | Capacity vs core C1 | Median TTFT / E2E |
| --- | ---: | ---: | ---: | ---: |
| Quick C2 | 4/4 (256) | 69.349106 | n/a | 0.597395 / 1.374107 s |
| Quick C4 | 8/8 (512) | 68.701867 | n/a | 1.532251 / 2.312291 s |
| Core C1 | 5/5 (1,280) | 76.549890 | 1.000000x | 0.154349 / 3.336737 s |
| Core C2 | 10/10 (2,560) | 76.713544 | 1.002138x | 1.791240 / 4.966544 s |
| Core C4 | 20/20 (5,120) | 76.090327 | 0.993997x | 5.187993 / 8.414750 s |
| Core C8 | 40/40 (10,240) | 76.282433 | 0.996506x | 11.885095 / 15.079883 s |

These values describe single-slot queue capacity. In particular, labeling
the C8 row “eight-way concurrent throughput” would be incorrect even though
all 40 client requests passed.

All repeat-prefill rows were measurement-valid client-TTFT approximations:

| Suite / nominal prefill | Requests | Median approximate tok/s | Median TTFT |
| --- | ---: | ---: | ---: |
| Quick P256 | 3 | 1,624.748231 | 0.219111 s |
| Quick P2048 | 3 | 2,577.757775 | 0.832894 s |
| Quick P8192 | 3 | 2,863.483606 | 2.895075 s |
| Core P128 | 5 | 1,265.042773 | 0.180231 s |
| Core P1024 | 5 | 2,182.474586 | 0.515470 s |
| Core P4096 | 5 | **2,796.337383** | 1.500177 s |
| Core P16384 | 5 | 2,654.349619 | 6.210561 s |

Semantic capability results were consistent across depths. Smoke JSON and
tool calls passed at 43.065154 and 51.705768 aggregate tok/s. The quick 8K
needle passed 1/1 at 3.327080 output tok/s with 2.855263-second TTFT. In core,
structured JSON and tools passed 5/5 at 43.628098 and 61.571172 aggregate
tok/s; the 16K needle passed 3/3 at 1.839677 output tok/s with 6.173637-second
median TTFT.

Smoke and quick are terminal `complete` with no validation failure. The core
process also completed normally with no run error or invalid measurement. Its
terminal `partial` status comes solely from the exact-answer screen: Laguna
scored 3/4 (75%), missing code reasoning while passing arithmetic,
instruction following, and logic. That invalid quality case has no promoted
throughput claim. Core P32768 was separately skipped before launch because
the estimated 32,909 required tokens exceeded the served 32,768-token limit;
vision, embeddings, and reranking were unsupported rather than attempted
failures.

Across the smoke, quick, and core journals, 522 telemetry samples retained a
minimum `MemAvailable` of 82.603195 GiB and constant 16,776,876 KiB swap free.
Peak sampled power was 78.98 W, temperature 83 °C, GPU utilization 95%, and
SM clock 2,535 MHz. The respective smoke/quick/core minimum
`MemAvailable` readings were 91.096222, 88.584187, and 82.603195 GiB. This is
a comfortable memory admission at P1, not evidence that an untested P8
Laguna geometry will behave the same way.

Exact ignored evidence directories are:

- `results/20260817T162539Z-laguna-xs21-33b-a3b-q4-k-m-llamacpp-smoke-2203e543`
- `results/20260817T162611Z-laguna-xs21-33b-a3b-q4-k-m-llamacpp-quick-7d4fb10f`
- `results/20260817T162725Z-laguna-xs21-33b-a3b-q4-k-m-llamacpp-core-fb5a1800`

### Poolside Laguna S 2.1: 118B-A8B Split GGUF P1

Laguna S 2.1 also fits and executes on one Spark. Poolside describes the
model as 117.6B total and 8.5B active parameters, commonly shortened to
118B-A8B, with a 1,048,576-token native context. The measured profile did
**not** exercise that context: both its benchmark ceiling and one served slot
were fixed at 32,768 tokens. The profile's benchmark-facing `native_context`
field is also conservatively pinned to 32,768, so it must not be read as a
restatement of the model-card limit.

The tested target is Unsloth's Dynamic 2.0 UD-Q4_K_XL split GGUF at revision
`750f92f90cf54159c4d7a610cb7b3e74498e75c6`. All three ordered shards were
verified before every server launch:

| Ordered shard | Exact bytes | SHA-256 |
| --- | ---: | --- |
| `Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003.gguf` | 3,683,648 | `0cfaf46917260d253773e5e2fab64329fa5c9c60fdf0db0f59f31205b5f5dd32` |
| `Laguna-S-2.1-UD-Q4_K_XL-00002-of-00003.gguf` | 49,971,821,312 | `2296102462b02edca70163121ac62bacf7a82078c0eafc91625c8822850769bf` |
| `Laguna-S-2.1-UD-Q4_K_XL-00003-of-00003.gguf` | 23,419,667,040 | `9150e2338f7690af29685b6a2ca621a8fda7ecf9724678266c4b04b7c6dd0ef3` |

The exact total is 73,395,172,000 bytes, or 68.354581 GiB. The same pinned
llama.cpp b10453 server and binary hash documented above loaded shard one and
resolved the complete split. It used full GPU offload, flash attention, fit
disabled, batch/microbatch 8,192/512, Q8_0 K/V, Jinja, reasoning off, no
speculation, and `--parallel 1`.

Smoke and quick both reached terminal `complete` with every attempted case
validation-clean. The following fixed-length cases recompute from raw request
token totals divided by case wall time:

| Suite / case | Passed (tokens) | Aggregate tok/s | Median client decode | Median TTFT / E2E |
| --- | ---: | ---: | ---: | ---: |
| Smoke D32 | 1/1 (32) | 16.867742 | 22.247475 tok/s | 0.470635 / 1.864052 s |
| Quick D128 | 3/3 (384) | 21.886615 | 23.652401 tok/s | 0.468032 / 5.837466 s |
| Core D256 | 5/5 (1,280) | **22.817433** | 23.755136 tok/s | 0.467098 / 11.199717 s |

Core D1024 is deliberately excluded from that table. All five requests ended
with `stop` after only 661--711 tokens, for 3,450 rather than 5,120 requested
tokens. Their raw median client decode estimate was 23.492089 tok/s, but the
case correctly suppresses aggregate and median throughput because it did not
perform equal fixed-length work.

As with Laguna XS, this is a one-slot profile. C2/C4/C8 are queued client
bursts, not concurrent model execution. Every one of the 75 core burst
requests reached 256 tokens and validated; the near-flat aggregate rate and
rising TTFT show serialized capacity rather than scaling:

| Queued burst | Passed (tokens) | Aggregate tok/s | Capacity vs C1 | Median client decode | Median TTFT / E2E |
| ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 5/5 (1,280) | 22.736010 | 1.000000x | 23.671708 tok/s | 0.469567 / 11.240098 s |
| C2 | 10/10 (2,560) | 22.770312 | 1.001509x | 23.658999 tok/s | 6.042229 / 16.841803 s |
| C4 | 20/20 (5,120) | 22.741801 | 1.000255x | 23.595871 tok/s | 17.308098 / 28.071041 s |
| C8 | 40/40 (10,240) | 22.712520 | 0.998967x | 23.590745 tok/s | 39.882534 / 50.673374 s |

The shorter quick queued bursts also passed: C2 produced 256 tokens at
20.334930 tok/s and C4 produced 512 at 20.423930 tok/s. They used only 64
output tokens per request and are not mixed into the core capacity ratios.

Every repeat-prefill case was measurement-valid. These remain client-TTFT
approximations, not native server prompt timings:

| Suite / nominal prefill | Requests | Median approximate tok/s | Median TTFT |
| --- | ---: | ---: | ---: |
| Quick P256 | 3 | 527.591468 | 0.670974 s |
| Quick P2048 | 3 | 962.945864 | 2.230655 s |
| Quick P8192 | 3 | **1,161.402395** | 7.138783 s |
| Core P128 | 5 | 413.364677 | 0.546733 s |
| Core P1024 | 5 | 786.743880 | 1.427402 s |
| Core P4096 | 5 | 1,083.547846 | 3.873387 s |
| Core P16384 | 5 | 1,140.017181 | 14.460308 s |

Capability results require the same qualification as throughput. Smoke
strict JSON and tools each passed 1/1, and the quick 8K needle passed 1/1
with 7.228690-second TTFT. In core, the 16K needle passed 3/3 with
14.241116-second median TTFT and tool calls passed 5/5. Strict JSON was not
stable: only 1/5 core requests emitted bare valid JSON; four wrapped the
otherwise correct object in Markdown fences. Exact-answer quality was 3/4
(75%), missing code reasoning while passing arithmetic, instruction
following, and logic.

The core process reached terminal completion with no run error or invalid
measurement. Its top-level `partial` status comes from the early-stop D1024,
the 1/5 JSON gate, and the 3/4 quality gate. P32768 was separately skipped
before launch because the estimated 32,909 required tokens exceeded the
served 32,768-token slot. Vision, embeddings, and reranking were unsupported
rather than attempted failures.

Across the three raw journals, 1,798 telemetry samples retained a minimum
`MemAvailable` of 32.516830 GiB. The smoke, quick, and core minima were
40.659718, 37.721359, and 32.516830 GiB respectively. Peak sampled power was
77.02 W, temperature 84 °C, GPU utilization 96%, and SM clock 2,535 MHz.
This establishes comfortable P1/32K admission for this quant, not P8 or
million-token feasibility.

Exact ignored evidence directories are:

- `results/20260817T173310Z-laguna-s21-118b-a8b-ud-q4-k-xl-llamacpp-smoke-f57b8b05`
- `results/20260817T173730Z-laguna-s21-118b-a8b-ud-q4-k-xl-llamacpp-quick-d341ef58`
- `results/20260817T174246Z-laguna-s21-118b-a8b-ud-q4-k-xl-llamacpp-core-f2ac29d0`

### Muse-Glimmer Dense 28B Target-Only Kernel Bracket

Muse-Glimmer is a dense control, not an MoE result. Although the artifact and
product use the `30B` name, llama.cpp reports 27,854,794,240 parameters, so
this section labels it dense 28B. The run loaded only the target GGUF: there
was no DFlash sidecar, speculative decoder, server, chat template, sampler,
or semantic output path.

The exact target was Unsloth
`Muse-Glimmer-30B-UD-Q4_K_XL.gguf`, revision
`faa5b025c584459c13febfa5c59883516710ae39`, 15,878,222,368 file bytes, and
`sha256:82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38`.
Its benchmark metadata reports 15,865,108,480 model-payload bytes. The same
pinned b10453 `llama-bench` binary used CUDA, 20 host threads, full GPU
offload, flash attention, batch/microbatch 8,192/512, Q8_0 K/V, and five
samples per row.

The six target rows recompute as follows; the dispersion column is the
persisted sample standard deviation, not a confidence interval:

| Kernel test | Mean tok/s | Sample SD | Sample range |
| --- | ---: | ---: | ---: |
| P128 | 831.516939 | 16.577321 | 802.320--841.963 |
| P1024 | **957.115259** | 10.055002 | 942.556--970.370 |
| P4096 | 821.678457 | 45.034499 | 771.737--868.814 |
| P16384 | 874.677155 | 46.686463 | 818.104--912.385 |
| TG256 | **11.851645** | 0.037742 | 11.8095--11.9084 |
| TG1024 | 11.762560 | 0.006378 | 11.7542--11.7695 |

TG1024 was 0.751668% below TG256, showing that the roughly 11.8 tok/s dense
generation rate persisted across the longer fixed budget. P4096 and P16384
had visibly larger run-to-run dispersion than P1024, so their apparent
ordering should not be treated as a context-scaling law.

The bracket used the same Unsloth Qwen3.8 dense 27B UD-Q4_K_XL control and
identical kernel geometry:

| Dense-Qwen D256 control | Mean tok/s | Sample SD |
| --- | ---: | ---: |
| Opening | 10.599094 | 0.042775 |
| Closing | 10.458164 | 0.043643 |

Closing/opening drift was **-1.329642%**, inside the 5% gate. Muse TG256 was
11.817529% above the opening control and 13.324337% above the closing control;
relative to their arithmetic midpoint of 10.528629 tok/s, the descriptive
difference was +12.565891%. No drift normalization is applied. The artifacts
have similar parameter counts and the same quantization family, but differ in
architecture and file size, so this is a bracketed observation rather than
an architecture-only effect.

This target-only kernel result belongs with the dense bandwidth controls. Its
approximately 11.8 tok/s generation rate remains in the same regime as the
dense Qwen control, not the 61--83 tok/s sparse rows above. Cross-model answer
quality cannot be inferred because `llama-bench` performs no answer emission
or validation.

The earlier Muse baseline/DFlash serving journals remain separate
**invalid-emission diagnostics only**. They emitted empty visible content and
nonce/prompt echoes in reasoning, failed JSON/tool or needle contracts, and
did not yield valid prefill measurements. Positive DFlash counters prove that
the draft mechanism ran, but neither those counters nor the accelerated
invalid emissions are usable-answer throughput. No historical DFlash serving
rate is merged into this target-only table.

Exact ignored evidence files are
`results/moe-bandwidth-20260817T1539Z/muse-control-pre.json`,
`results/moe-bandwidth-20260817T1539Z/muse-glimmer.json`, and
`results/moe-bandwidth-20260817T1539Z/muse-control-post.json`, plus their
same-directory logs. All three logs contain the expected single-device CUDA
initialization and no recorded load error.

<!-- END LOCAL MEASURED RESULTS 2026-08-17 -->
