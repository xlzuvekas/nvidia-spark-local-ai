# DenseSpark v1.2 on Qwen3.8 27B: local evidence and transferable lessons

Date: 2026-08-31

## Scope and verdict

Labels in this report are deliberate:

- **Source fact** — read from the exact pinned recipe or local launch contract.
- **Measured** — present in a persisted local run receipt.
- **Inference** — a bounded interpretation of those facts or measurements.
- **Planned** — implemented or proposed work with no completed result yet.

**Measured:** neither exact DenseSpark v1.2 managed attempt reached a ready API;
both recorded zero benchmark cases. The first stopped on the host swap-growth
gate. The second reached the generic Qwen Triton warmup line and was interrupted
196 seconds later without a subsequent ready message.

**Inference:** this is an artifact-pinned startup evaluation with explicit
dirty-tree and legacy launch-policy caveats, not a local managed v1.2 throughput
evaluation. The unresolved managed blocker is readiness after the logged warmup
stage; the receipts do not identify a specific warmup helper or kernel as the
root cause. A later sync-only manual diagnostic reached readiness, but it ran
outside the managed watchdog and failed the host swap-admission boundary at
shutdown, so it does not change that verdict.

**Withheld:** no persisted receipt proves that a roughly 45 tok/s manual run had
CUDA graphs disabled. The nearby measurement is a v1.1 exploratory run whose
launch command does not record a graph-disable control. It must not be reported
as “v1.2 graph-disabled ~45 TPS.”

**Measured, separate SGLang stack:** a newly completed matched-prompt,
one-lifetime-per-arm screen found FULL decode graphs 9.80% higher in aggregate
output rate and 3.50% higher in median client-estimated decode rate than the
disabled arm. This is an n=1 directional result for the pinned RadixArk
Qwen3.8-27B plus DSpark stack, not DenseSpark v1.2, and requires replication.

## Immutable identity and reproducibility boundary

| Layer | Exact identity | Evidence boundary |
|---|---|---|
| DenseSpark recipe | `albond/DenseSpark-Qwen3.8-27B` commit `0abecc3005cebe6f5e1e0c0e1f16552f95fe0228`, tree `347468a41f8431b0c5a94a56e316566a06489a43`, tag/version v1.2 | **Source fact.** The v1.2 delta from v1.1 commit `0c96f23ad0b7683bcbb75ac0bf421f27189c7941` adds automatic tool choice with `qwen3_xml` and corrects version labels; it does not change the performance stack. |
| Target checkpoint | `Frozenlock/Qwen3.8-27B-int4-AutoRound` revision `b4c61732c4f2d8af323d75ba5702b5c7f3361539`; 8 weight files, 18,996,706,072 bytes | **Source fact.** [The adapter](../bench/densespark.py) descriptor-hashes the complete exact 18-file snapshot, including every weight and tokenizer/config artifact, not only the revision label or filenames. |
| Local image | `local/densespark:qwen38-27b-v1.2-0abecc3`; image ID `sha256:d8d02859a49ebf452d9e20b5fbc0790cd4c38fe9a1f5184096b06e3cc6a751d1` | **Measured plan provenance.** The image contains the vLLM 0.27.1-based recipe. |
| PQ draft artifact | 34,906,281 bytes; `sha256:4e794c398d700002479b914e2c5d530ead57ca5861862ab4230bc470cf95cea9` | **Measured plan provenance.** The managed adapter validates size and digest before launch. |
| Launch contract | profile `qwen38-27b-int4-autoround-densespark-c1`; configuration `sha256:e6ac07581881aa589dfeebca7ca034d99858ab333166bc5648cbfa944543fda6`; plan fingerprint `7ded7b696f4a42d7` | **Measured plan provenance.** Both managed attempts resolved the same image, model, PQ artifact, and configuration. |
| Host | `NVIDIA GB10, 580.142, 12.1`; aarch64; Docker 29.2.1 | **Measured plan provenance.** Both plans captured the same hardware class. |

The managed plans recorded harness Git HEAD
`645407473391eeee6c0c30c9753b4602c566d443` **and a dirty worktree**. The model,
image, PQ artifact, frozen plan, and configuration digest are immutable; the
working source tree as a whole is not. This caveat prevents treating the attempt
as a clean-checkout reproduction even though its served-artifact identity is
exact.

The two persisted plans predate the separately digested launch-policy receipt.
The evidence exporter authenticates their exact plan hashes as
`legacy-unbound`; it does not retroactively claim that their launch controls
were frozen. For future plans, the current managed Docker contract is
artifact-download-disabled and HF-offline: it uses the exact local snapshot,
disables image pulls, and sets `HF_HUB_OFFLINE=1`. It nevertheless retains
Docker's bridge network and has no egress-denial control. It is therefore
explicitly **not egress-isolated**; offline artifact resolution must not be
described as network isolation.

The writable compiler cache is isolated by the full configuration/image digest.
Its user-owned anchors are mode `0700`, and the adapter descriptor-walks the
selected namespace before mounting it: symlinks, hard-linked files, special
files, untrusted owners, group/world-writable nodes, excessive nesting, and
concurrent topology changes fail closed. Root-owned descendants created by the
container remain admissible under those user-owned anchors.

The upstream `--gpu-memory-utilization` value was reduced from 0.90 to 0.86 to
preserve more host `MemAvailable` on this unified-memory system. This is a
MemAvailable-focused configuration reduction, not a broad or measured safety
repair. In particular, the first managed 0.86 attempt still tripped the separate
swap-growth watchdog.

## What the recipe does

### AutoRound W4A16 body

**Source fact:** Qwen3.8 27B is a dense `Qwen3_5ForConditionalGeneration` model
with 64 layers (48 Gated DeltaNet/linear-attention and 16 full-attention), hidden
size 5,120, dense MLP size 17,408, vocabulary 248,320, an untied LM head, and one
built-in MTP layer. The pinned checkpoint stores symmetric AutoRound INT4,
group-size-128 weights and uses BF16 activations: W4A16. The primary mechanism is
lower per-token weight traffic; it is not a claim that every Qwen checkpoint can
reuse these quantized tensors.

### MTP plus a product-quantized proposal head

**Source fact:** the C1 profile uses probabilistic MTP with eight speculative
tokens. Its proposal-only PQ head divides the 5,120-dimensional head into 128
subspaces of 40 dimensions with 256 centroids each. It scans a 31.8 MB code
array instead of the full 1.271 GB INT8 head, keeps 2,048 candidates, and then
gathers and reranks those candidates with the INT8 head.

Only proposals use this approximation; the target model verifies them. A PQ
miss can reduce acceptance and performance, but it does not bypass target
verification. This is a structural correctness boundary from the pinned source,
not a locally measured v1.2 acceptance or quality result.

### Shape- and chip-specific kernels

**Source fact:** the recipe combines several GB10/SM121-specific choices:

- a per-channel INT8, Triton-batched LM head to reduce traffic through the large
  untied output projection;
- Marlin W4A16 below 256 rows and Humming W4A8/INT8 at 256 rows and above;
- retained FP8/CUTLASS layouts for selected projection families at exact prefill
  row counts 8,000, 8,192, and 16,000, with Humming elsewhere;
- Humming 0.1.13 for its SM121 selector and GB10 fallback; and
- a version-pinned FlashInfer GDN prefill eligibility patch for SM12.x.

These are dispatch policies tied to exact shapes, checkpoint geometry, vLLM
0.27.1, and GB10. The transferable idea is to route by measured shape and bytes,
not to copy the binaries or thresholds to another architecture.

### Full decode CUDA graphs, guarded prefills

**Source fact:** the recipe preserves FULL CUDA graphs for genuine decode while
patching hybrid-model dispatch so a short context batch cannot be mistaken for
a speculative verification batch before GDN recurrent state is initialized.
The context-bearing prefill is forced away from the decode graph; actual decode
keeps it.

**Measured startup evidence:** the second managed log configured
`FULL_AND_PIECEWISE`, profiled 44 PIECEWISE and 44 FULL sizes (largest 504), and
estimated 2.10 GiB of graph memory. Because the API never became ready, this is
not a throughput result and does not establish a graph speedup.

### Prefix caching is deliberately off

**Source fact:** the profile uses `--no-enable-prefix-caching` with
`--mamba-cache-mode none`. In the pinned vLLM 0.27.1 hybrid path, the recipe
authors observed nondeterminism for a repeated one-token prompt with prefix
caching plus `align`, even with the short-prefill guard. This is a correctness
choice for this runtime/model combination, not a general claim that prefix
caching is undesirable.

## What actually ran

| Attempt | Persisted observation | Publishable conclusion |
|---|---|---|
| Manual v1.1 exploratory C1, untracked local receipt (`sha256:be2e674794b8b1c8a5dd3deb737cc5b9f03c3e7e6a0c71f7884329e040048319`) | Image `local/densespark:qwen38-27b-v1.1-0c96f23`; three 1,000-input/1,000-output single-request runs measured 43.787858, 37.243805, and 56.329056 output tok/s; median 43.787858. | **Measured, v1.1 only.** The recorded server command has no `--enforce-eager` or equivalent graph-disable marker, so graph state is unresolved. |
| Exact v1.2 managed run at 22:59:50Z: [tracked scalar evidence](../evidence/runs/20260831T225950Z-qwen38-27b-int4-autoround-densespark-c1-qwen38-27b-densespark-c1-7ded7b69/manifest.json) | At server start, swap grew 582,000 KiB from 259,564 to 841,564 KiB, exceeding the 524,288 KiB growth ceiling. Status `aborted`; zero cases. | **Measured safety stop.** No endpoint, throughput, quality, or tool result. |
| Exact v1.2 managed run at 23:06:49Z: [tracked scalar evidence](../evidence/runs/20260831T230649Z-qwen38-27b-int4-autoround-densespark-c1-qwen38-27b-densespark-c1-7ded7b69/manifest.json) | Cached compile and graph profiling completed. The last persisted server line at 23:10:24Z was `Warming up Qwen Triton kernels for model_type=qwen3_5_text`; operator interruption was recorded at 23:13:40Z. Status `aborted`; zero cases. | **Measured readiness blocker, root cause unresolved.** No endpoint or benchmark result. |

### Sync-only manual diagnostic: non-publishable

This diagnostic is deliberately outside the managed result table. There is no
persisted raw receipt, it did not run under the managed watchdog, and its
shutdown crossed the host swap policy. The observations below are therefore
**operator-recorded manual diagnostics**, not publishable benchmark evidence or
an admission result.

**Source fact:** the preserved
[historical diagnostic Dockerfile](../patches/vllm/Dockerfile.densespark-qwen-warmup-probe-manual-20260831)
names the v1.2 base tag, keeps the normal `vllm` entrypoint, and installs
[a probe](../bench/assets/densespark_qwen_warmup_probe.py) as `sitecustomize.py`
in both the API and EngineCore processes. The sync-only mode wraps each unchanged
helper with an accelerator synchronization before and after the call; it skips
no helper and does not substitute the optional rank-four-state diagnostic. The
locally derived image was operator-recorded as
`sha256:8c75eff2e41c35ca29af6f5af47ddaa4c2bd0c8ef0da34498b0ffe00e0ad034e`.
That historical recipe hashes to
`f1bb07060061ec5bcabac38b3119406e9e69f42602b106d6d89c25e6ee530afb`.
Its mutable `FROM` indirection and overridable digest arguments did not by
themselves enforce the recorded base identity, so this remains a manual
diagnostic identity rather than a reproducible rebuild claim.

The hardened
[rebuild Dockerfile](../patches/vllm/Dockerfile.densespark-qwen-warmup-probe)
hashes to
`572e66d585ed74a5f0b278e2feb2cf7dba260ca84c53ba93be66d7c2e69c571a`.
It hardcodes the base tag and every source/entrypoint digest and contains no
Dockerfile `ARG`. The standard-library
[local build helper](../bench/densespark_warmup_probe_build.py) validates that
the base tag resolves locally to
`sha256:d8d02859a49ebf452d9e20b5fbc0790cd4c38fe9a1f5184096b06e3cc6a751d1`,
validates all three checked-in input hashes, invokes the fixed build with
`--pull=false`, `--network=none`, and no build arguments, revalidates the inputs
and base tag after the build, and then requires the derived tag to resolve to an
exact pinned ID. It accepts no command-line options. A different derived ID is a
hard failure, not permission to repin it. A Dockerfile-specific context policy,
`sha256:100ee126af6ef26dd45e85b9e90f5cc0adb8d6b0c51d391c37117fc7168627ea`,
excludes the rest of the 83 GB worktree and admits only the probe payload needed
by `COPY`.

The hardened rebuild completed locally through that exact helper. It produced
the distinct tag
`local/densespark:qwen38-27b-v1.2-warmup-probe-hardened-572e66d5` with image ID
`sha256:c7adf2163f7dd04b52eb5ec91f373bf8fcd1cc63a51f61c2d457ad2976564153`.
The historical tag and `sha256:8c75eff…` image were left untouched. The managed
warmup-sync profile now points only at the hardened identity; neither image has
been promoted into managed throughput evidence. The reproducible invocation is:

```bash
python3 -m bench.densespark_warmup_probe_build
```

The hardened recipe fail-closes on the fixed inputs below; the historical
operator build recorded the same digest values as overridable defaults:

| Diagnostic input | Required SHA-256 |
|---|---|
| Probe payload | `95089265e60f67da8d8f33d6fb249e4c79300f0891c38f2f15d4e125001821d3` |
| `qwen_triton_warmup.py` | `2b08d94662e7b04ce61c0f7a818e0cd1768fe7602a89df04ec6148f62fe3acdb` |
| `kernel_warmup.py` | `452ae5db905110df8eb7aac90a93ac80863d166f8ea7d52b8cec02c477477aed` |
| `qwen_gdn_linear_attn.py` | `d42cdc95d8d221b49693a46119c714fee3f290282bdfefa63f92f9725f1b20ea` |
| `mamba_utils.py` | `53eaae681b5a0327465b28b7b1983303335db852ac9667ae05faa3682d8c6b8c` |
| `fused_sigmoid_gating.py` | `000ab8996af9788fdb8843a6a3b91833e7a14c8acc0e1ea073a536330f64cb6f` |
| vLLM entrypoint | `6f6395c128e80861f7f7d21b8e1e4547261ab9e928390aa7a7a89ce0d701ff36` |

**Manual observation:** all three EngineCore helper calls returned across their
explicit synchronization boundaries: causal-convolution warmup in 0.528629 s,
fused post-convolution warmup in 0.113467 s, and fused-sigmoid/gating delta-rule
warmup in 0.164382 s. Full graph capture then completed in about 30 seconds; its
actual graph pool was 1.28 GiB versus a 1.70 GiB estimate, and the server reached
ready.

After one unscored JIT warm pass, three identical temperature-zero D256
numbered-phrase continuation requests each recorded 43 prompt and 256
completion tokens. This is the low-entropy synthetic prompt preserved by
[`benchmark.py`](../benchmark.py), not a coding task:

| Request | Elapsed | Output tok/s |
|---|---:|---:|
| 1 | 4.358850 s | 58.731092 |
| 2 | 4.133956 s | 61.926148 |
| 3 | 4.141409 s | 61.814716 |

The manual synthetic-continuation ceiling was 61.814716 output tok/s. A
separate synthetic `add_numbers` request returned one valid tool call in
0.831975 s. These are
functional and repeatability diagnostics, not a managed throughput or agentic
quality result.

After seven length-terminated requests, the manually read metrics snapshot
reported 2,176 drafted tokens, 1,275 accepted tokens, 272 draft operations, and
1,552 generated tokens. That is 58.59% draft-token acceptance and mean accepted
length `1 + 1275 / 272 = 5.6875`, including the verified target token.

**Safety boundary:** shutdown observation saw swap use rise to approximately
1,580,000 KiB before settling near 590,000 KiB. The settled value alone exceeds
the managed 512 MiB (524,288 KiB) starting-swap gate. Consequently this launch
is not admissible or production evidence despite reaching ready and serving
requests.

**Inference:** explicit synchronization was sufficient for this one manual
launch to pass the previously last-seen warmup stage. It does not prove a race,
causally explain the earlier managed failure, or assign fault to a helper,
process, or rank. A controlled base-versus-sync comparison is required.

## Why the old SGLang graph pair is not causal

The earlier pair used a different stack: SGLang with
`RadixArk/Qwen3.8-27B-NVFP4` revision
`52d1adc5f38aa5ebf099c29ed7025ba34cfbb854` and DSpark draft revision
`923ed3a8572615643f0137e424e4ce4edd7f1cda`. Its completed observations were:

| Decode graph arm | Run | Median client-estimated decode tok/s | Total prompt tokens / output tokens |
|---|---|---:|---:|
| disabled | 22:30:36Z, run suffix `610c0100` | 26.809591 | 600 / 1,280 |
| disabled | 22:54:34Z, run suffix `610c0100` | 26.576118 | 600 / 1,280 |
| full | 22:35:29Z, run suffix `801b10c9` | 27.907974 | 605 / 1,280 |
| full | 22:42:59Z, run suffix `801b10c9` | 27.106197 | 605 / 1,280 |

**Fact:** the frozen case IDs differed by arm
(`decode-256-c1--66ff0eba5a3d` versus
`decode-256-c1--78d6365e7eea`). The generic runner incorporated that arm-bound
case identity and a clock value into the request ID, and the request ID into the
`Benchmark nonce ...` prompt prefix. The 120-versus-121 prompt tokens per
request corroborate that the prompt bytes were not matched.

**Inference:** the numerical ordering is descriptive only. It cannot identify a
CUDA-graph effect because prompt construction changed with the arm.

## Replacement matched-prompt protocol

**Implemented fact:** [the new suite](../manifests/suites/qwen38_27b_dspark_c1_cuda_graph.toml)
reserves the exact case ID
`matched-prompt-qwen38-27b-dspark-cuda-graph-d256-c1-v1` under protocol
`matched-request-unique-v1`. It freezes one D256/C1 case at temperature zero,
one warmup, and five measured repetitions. The runner derives one warmup tag
and five repetition-indexed measured tags from the common case contract without
publishing them. This creates byte-identical prompts across arms while keeping
each request unique within an arm. The two frozen launch profiles are source clones whose serving
controls differ only in decode graph backend (`full` versus `disabled`), aside
from profile identity/description. Manifest and frozen-plan validation reject
unreviewed matched IDs or suite drift; [the regression test](../tests/test_qwen38_27b_dspark_cuda_graph_profiles.py)
captures request IDs and prompt bytes and asserts cross-arm equality.

Evidence export authenticates the exact per-arm model/runtime contract and all
five measured request IDs before redacting those IDs. Published verification
then requires the exact arm-bound case/model digests, sample schemas, and
recomputed aggregate relationships; it does not publish commands, prompts, or
request identifiers.

### First matched result

Both arms used SGLang image
`lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1`,
the model and draft revisions named above, batch-one decode, eager prefill, and
the same five deterministic measured request IDs. Each arm recorded 535 prompt
tokens, 1,280 completion tokens, and five completed requests marked
measurement-valid. This generic decode case has no semantic or quality
validator. Exported
reasoning-token totals also differed: 1,289 disabled versus 1,285 FULL, so this
screen establishes neither output nor quality equivalence. The ignored local
server logs showed `cuda graph: False` and `cuda graph: True`, respectively,
during decode; prefill remained false in both. Those raw logs are not versioned
or linked. The pinned profiles and tracked scalar summaries below are the
reproducible public boundary.

| Metric | [Disabled](../evidence/runs/20260831T232106Z-qwen38-27b-nvfp4-dspark-c1-cuda-graph-disabled-sglang-qwen38-27b-dspark-c1-cuda-graph-2d35736f/summary.json) | [FULL](../evidence/runs/20260831T232549Z-qwen38-27b-nvfp4-dspark-c1-cuda-graph-full-sglang-qwen38-27b-dspark-c1-cuda-graph-a07ce0e8/summary.json) | FULL relative to disabled |
|---|---:|---:|---:|
| Aggregate output tok/s | 23.380815 | 25.672424 | **+9.80%** |
| Median client-estimated decode tok/s | 24.852022 | 25.720896 | **+3.50%** |
| Median TTFT | 0.180641 s | 0.200548 s | **+11.02% (worse)** |
| Median E2E | 10.441375 s | 10.147109 s | **-2.82%** |
| Five-request case elapsed | 54.745739 s | 49.858945 s | **-8.93%** |
| Output tokens per sampled joule | 0.675626 | 0.740987 | **+9.68%** |
| Server startup | 181.188 s / 177 samples | 273.778 s / 266 samples | +51.10% wall time |

**Measured:** FULL graph startup included 96.83 seconds of target-verify capture
and 12.37 seconds of draft-verify capture, explaining most of its larger startup
cost. **Inference:** FULL decode graphs are a promising direction for this exact
D256/C1 SGLang configuration, but the evidence is only one disabled lifetime
followed by one FULL lifetime with five requests each. The modest median-decode
margin, fixed arm order, and lack of an alternated replicate preclude a stable
effect-size or broader Qwen claim.

## Bounded transfer to Qwen3.8 27B and Flash-Next

| Lesson | Bounded conclusion | Do not transfer |
|---|---|---|
| Quantize bytes first | For dense Qwen3.8 27B, W4A16 is the recipe's foundational traffic reduction. For Flash-Next, weight-only W4A16 is a reasonable first experiment for its remaining BF16 MTP experts or dense projections, behind an exact quality gate. | The DenseSpark checkpoint tensors, quality result, or numeric speedup. Flash-Next's main routed experts are already NVFP4 in the admitted local artifact. |
| Optimize proposal cost, then measure acceptance | DenseSpark's PQ head demonstrates a useful pattern: compress proposal-only vocabulary scoring and let the target preserve correctness. The historical Flash-Next C1 receipts (30.123639 tok/s with native MTP3 versus 16.663713 off) support prioritizing native speculation, but their 1.26% prompt-token mismatch means the `1.807739x` ratio is not a transferable exact effect size. | The 5,120-by-248,320 PQ artifact, its 2,048-candidate width, MTP depth eight, or its acceptance behavior. Flash-Next uses different MTP, MoE, QSA, and PLE geometry. |
| Dispatch on real shapes and the target chip | Profile byte traffic and route exact GB10 shapes to the kernel that wins; keep fail-closed shape and version audits. | DenseSpark's Marlin/Humming/CUTLASS binaries, 256-row crossover, or selected projection families without new measurements. |
| Graph only graph-safe regions | DenseSpark's hybrid-prefill guard is relevant to any GDN model: distinguish context work from speculative verification before replay. The matched 27B SGLang screen is positive but n=1. For Flash-Next's file-backed PLE path, [the local graph contract](../bench/sglang_sm121_cuda_graph.py) allows only `breakable` decode graphs with eager prefill; host `tolist()`, thread/Future, `io_uring`, `ctypes` copy, and transfer/event boundaries prohibit full capture. | A DenseSpark v1.2 or Flash-Next full-graph benefit inferred from the separate 27B SGLang result. The old unmatched graph delta remains non-causal. |
| Treat caches as separate correctness domains | DenseSpark prefix caching remains off because of a pinned vLLM hybrid correctness failure. Flash-Next must evaluate its SGLang radix/prefix cache independently from its file-backed PLE page cache. | A universal “cache off” rule, or an assumption that PLE cache behavior proves prompt-prefix cache behavior. |

The defensible transfer is therefore architectural: minimize bytes in the
target and draft paths, dispatch exact shapes, preserve graph capture only
across graph-safe regions, and gate every change on matched prompts, target
verification/acceptance counters, output validation, memory safety, and quality.
It is not a transfer of DenseSpark artifacts or performance numbers.

## Next evidence gates

1. Wait until starting swap is within the frozen safety gate, then run a managed,
   persisted base-versus-sync comparison with identical artifacts and launch
   controls. Keep every helper enabled, alternate arm order on replication, and
   record per-helper timing, readiness, graph memory, cleanup, and swap.
2. Require that controlled comparison, a ready endpoint, and clean canaries
   before reporting v1.2 throughput, tool handling, quality, or agent
   suitability. The manual 61.814716 tok/s median is not such a result.
3. Replicate the matched SGLang graph protocol in alternated arm order before
   treating its first positive graph delta as stable.
4. For Flash-Next, test only the safe `breakable`-versus-`disabled` graph pair,
   then separately gate any W4A16 MTP experiment on quality and native
   speculation counters.
