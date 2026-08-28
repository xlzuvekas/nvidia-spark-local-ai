# Experimental SGLang SM121 XQA + persistent FP8 PLE route

This directory preserves the audited SGLang source changes used to serve the
full Qwen3.8-Flash-Next Radix checkpoint on DGX Spark. The original two-line QSA
change was first isolated with a tiny fixture; the full route additionally
requires the persistent read-only FP8 PLE loader. Together they produced a
measured full-checkpoint result. They remain an experimental local route, not
upstream SGLang or general SM121 support, and the two-line QSA patch alone is
not a full-model admission recipe.

**Safety supersession, 2026-08-28:** later varied-token testing found
long-context corruption on the SM121 TRT-LLM route, and SGLang restriction
`99c9362` returned that kernel to exact SM120. Open
[PR #36845](https://github.com/sgl-project/sglang/pull/36845) adds an explicit
SM121 Triton packed-varlen fallback. Preserve this directory for historical
measurement provenance only; do not use `d91c3682-qsa-sm121-xqa.patch` in a new
build. New integration work must preserve the exact SM120 TRT-LLM restriction,
use an explicitly attested SM121 Triton fallback, and pass varied-token
long-context validation. The
[day-two review](../../docs/qwen38-flash-next-gb10-day-two-delta-2026-08-28.md)
records the exact ancestry, caveats and current component plan.

## Current storage-reader integration candidate

A read-only static replay applied storage-only commits `04648a7` and `9f101e3`
in that order to SM121 Triton base `3681c4e`, excluding competing QSA commit
`8ef3b3`. Both apply without content conflicts. The deterministic tree after
the reader is `cb9b2dffb10ae70bc91915c3eade4957fa649eaa`; the tree after PLE
streaming is `ddda8dde3b6655c4e0c0ff094d87ef1f5cc71a92`.

The integration changes eleven files. No added line references QSA, TRT-LLM or
SM121, and the SM120-restricted resolver, Triton fallback, architecture detector and QSA
test blobs remain byte-identical to `3681c4e`. Static diff, Python AST, Ruff,
Rustfmt and locked/offline Cargo-metadata checks pass. These facts establish
source composition only; no extension, image, server or model was built or
run.

The source composition exposed one build-admission gap: the new `_storage`
extension is auto-discovered and built, but the prebuilt artifact staging,
required-module check and import-smoke lists omit it. Apply tracked patch
`ddda8dde-storage-extension-packaging.patch` only after the two storage commits
above. It adds `storage` to all three lists. The patch ID is
`afb1c5aad878841df67bd96d57a1075b5167cc00`, its file SHA-256 is
`f7f7a7f7231c3b893bb868a0919cea5c71dbeb1d2ca5c0dff0ebb982dd56fbc7`, the
input tree is
`ddda8dde3b6655c4e0c0ff094d87ef1f5cc71a92`, and the resulting tree is
`bdb62e9fbc76f6e206cb0136576b88b7e1517a51`. Bash syntax, embedded-Python AST,
exact module-list assertions, `git apply --check`, and `git diff --check` pass.
No fourth hard-coded extension list was found.

That resolves the static packaging omission, not build or runtime admission.
The resulting source must still require:

```text
cargo clippy -p sglang-storage --all-targets -- -D warnings
cargo test -p sglang-storage
pytest -q test/registered/unit/models/test_qwen4_ple_nvme.py
pytest -q test/registered/unit/storage/test_io_uring_reader.py
pytest -q test/registered/kernels/test_qsa.py -k sm121
```

Also prove an ARM64 `_storage` build/import with runtime building disabled,
verify artifact contents, and compare synthetic resident versus NVMe gathers.
On the target, an `EPERM` or `ENOSYS` skip is an admission failure so the narrow
three-syscall `io_uring` policy is tested rather than assumed.

### Pinned ARM64 build candidate

`sm121-storage-build-candidate-v1.toml` is the strict composition and build
invocation contract for candidate `sglang-sm121-triton-storage-v1`. Starting
from tree `bdb62e9fbc76f6e206cb0136576b88b7e1517a51`, apply
`bdb62e9f-cuda130-arm64-base-pin.patch`. Its SHA-256 is
`7de1f4f3ed468c1a4b7ac9e98abbe73b6908095321289d0b5328334608f6df11`,
its stable patch ID is `9d143b6c2fd01a26e906a077dd4e917c675faefb`,
and the resulting tree is
`274ee330db7ea9653807b868c0fb8693d50ed7b2`. The patch materializes
`.dockerignore` as a regular file and makes both external `FROM` stages use the
Linux/ARM64 child manifest
`sha256:689c34b3fdc12303418e90828359fcdb3c7053bc8aa8d7681823186dde1fe572`
for CUDA 13.0.3. The competing QSA commit
`8ef3b3fee34a3b5543b65393dd217ed0362a9273` is explicitly excluded from the
source chain and ancestry.

Verify a clean, non-shallow reconstructed source checkout and the intended
invocation before building. The checkout must contain the pinned base and step
commit objects plus the explicitly excluded QSA commit object:

```text
python3 -m bench.sglang_build_contract \
  --repository-root . \
  --contract patches/sglang/sm121-storage-build-candidate-v1.toml \
  --source-root /path/to/reconstructed/sglang \
  --target runtime \
  --platform linux/arm64 \
  --build-arg KEY=VALUE  # repeat for every exact entry in [build.args]
```

Every build argument must be supplied as `KEY=VALUE`; missing, extra,
duplicate, value-mismatched, or environment-inherited `KEY` arguments fail.
The empty `SGL_VERSION`, `PIP_DEFAULT_INDEX`, `UBUNTU_MIRROR`, and
`SGLANG_BUILD_URL` values are intentional. `TARGETARCH` is automatic BuildKit
metadata rather than a passed build argument; architecture selection is bound
by `--platform linux/arm64`. `MOONCAKE_COMPILE_ARG` is declared but unused by
this Dockerfile and is deliberately not passed; the verifier fails if it is
lexically referenced outside its `ARG` declaration without a schema change.

The argument-reference check is restricted to stages reachable from the
`runtime` target through `FROM`, `COPY --from`, and `RUN --mount ... from=`
edges. It proves that each passed argument has a recognized reference in that
build graph; it does not prove that every conditional shell branch executes.
For this ARM64, CUDA 13.0.3, local-source invocation, `HPC_OPS_COMMIT` is in an
x86-only branch, `SGL_DEEP_GEMM_VERSION` is in a CUDA-12 branch, and
`USE_LATEST_SGLANG` plus `SGL_VERSION` are in inactive non-local source
branches. Their frozen values are invocation provenance, not evidence that
they affect the resulting image.

The verifier uses only local Git and file operations and never modifies either
inspected repository. Before trusting tree IDs or cleanliness, it runs strict
full object-integrity checks, rejects alternate object databases and effective
clean-filter or archive-control attributes, and rejects integrity-bypass Git
configuration. It replays the source transitions in a disposable, isolated Git
index and object directory, then removes that scratch state. Run the verifier
against the clean Git checkout first; it intentionally requires a Git
repository. After it succeeds, create the build context directly from the
literal `source_tree` returned by the verifier. For this schema-1 contract,
the config- and attribute-isolated handoff is:

```text
env -i PATH="$PATH" LANG=C LC_ALL=C \
  GIT_ATTR_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 \
  GIT_NO_LAZY_FETCH=1 GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 \
  GIT_PAGER= GIT_TERMINAL_PROMPT=0 \
  git -C /path/to/reconstructed/sglang \
    -c core.fsmonitor=false -c core.hooksPath=/dev/null -c tar.umask=0000 \
    archive --format=tar 274ee330db7ea9653807b868c0fb8693d50ed7b2 \
    > sglang-sm121-storage-274ee330.tar
```

Build from exactly that tar stream. Do not substitute a later `HEAD`, export
the mutable worktree, or run the verifier against the export. A tracked-tree
export is required because
this Dockerfile's `.dockerignore` does not exclude `.git`. This contract's
status is `source_and_build_invocation_only`: it describes a build candidate,
not an admitted runtime, and it is not a reproducible dependency closure.
Mutable apt, pip, rustup, and other remote dependency resolution still require
recording the newly built OCI digest and completing the ARM64 `_storage`, SM121
Triton, long-context, and quality admission checks separately.

## Historical measured route identity

The `gd91c3682b` package version is a reported base, not a reconstructible
Qwen3.8 source identity. Pristine upstream d91 lacks the native Qwen4 model,
QSA backend, and associated draft/glue code. The historical runtime identity is
the exact image digest and its baked contents plus the tracked overlays below;
source claims from public d91 must be re-attested against that image.

- Container image: `lmsysorg/sglang@sha256:14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4`
- Reported SGLang base: `d91c3682b0b429e4c70df63cd57f819588ce29b0`
- SGLang package: `0.0.0.dev1+gd91c3682b`
- FlashInfer: `0.6.17`
- Target: NVIDIA GB10, compute capability 12.1
- Source file: `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`
- Original source-file SHA-256: `c959835d05d0f395ad7eae4330cf264af9f6f7c1bff3d45a39bb953d2536f5f2`
- Patched source-file SHA-256: `a6b003ed21b3be8ba763e8627aee39baee3d84184f5bf0fc650a1a6b853119d3`
- Patch: `d91c3682-qsa-sm121-xqa.patch`
- Patch SHA-256: `ca3fe1e1e8ebb0c1c606d4786a9b6b4c29bd9aff59baccee7e432136d6450768`

The stock resolver admits FlashInfer paged sparse decode only when
`is_sm100_supported()` is true. On SM121 it therefore falls through to the
FlashAttention 4 CuTe varlen implementation, which failed MLIR compilation in
the native QSA smoke. The patch also admits `is_sm120_supported()`. In the exact
image, that helper covers compute-capability major 12 with the installed CUDA,
and FlashInfer dispatches the existing decode API to XQA.

## Pinned full-checkpoint overlays

`python3 prepare_sglang_overlays.py` reproduces the two ignored runtime
overlays required by the full Flash-Next profile. It uses only the exact cached
image above, imports the two SGLang modules in a network-disabled container
without a GPU, extracts them through an inert container, applies the pinned
patchers, and admits only AST-verified files with the profile's expected
digests. It never pulls an image, and cleanup removes only the inert container
ID that it created. Existing partial or mismatched output is not overwritten.

The two vendored patchers are byte-for-byte copies from
`hashd1ve/qwen38-flash-next-one-dgx-spark` at commit
`bf2b7c75870d3703730b6bd8f3bb93dc622c278d` (MIT):

- [`ple_mmap.py`](https://github.com/hashd1ve/qwen38-flash-next-one-dgx-spark/blob/bf2b7c75870d3703730b6bd8f3bb93dc622c278d/patches/ple_mmap.py), vendored as `bf2b7c75-ple_mmap.py`, SHA-256 `eeabdde061631c9b606d4ccc7371ff8fb01c6cc034dfe6bad1e4f29a8aa21555`;
- [`qsa_trtllm_sm120.py`](https://github.com/hashd1ve/qwen38-flash-next-one-dgx-spark/blob/bf2b7c75870d3703730b6bd8f3bb93dc622c278d/patches/qsa_trtllm_sm120.py), vendored as `bf2b7c75-qsa_trtllm_sm120.py`, SHA-256 `f60ccb9f9e350a43155a1a7a20d154be0b7e93c29dacb3db95d397ba910090b2`.

The retained legacy generated files are
`results/runtime-overlays/qwen38-flash-next-bf2b7c75/qwen4_exp.py`
(`c687bf96b8adb980eaf3a1db2ad4a7c00b558537865d91674c0e1b43f4ae1d71`)
and `qwen_sparse_attn_backend.py`
(`e30566492e1502f94a4c7fed42d90b523bbb662580c628459e6e63c7b5263c75`).
That exact pair is retained for already-frozen writable-mmap plans. The
historical measured profile used the separate
`results/runtime-overlays/qwen38-flash-next-bf2b7c75-persistent-ple-v1`
target. Its `qwen4_exp.py` additionally applies
`qwen38-persistent-ple-cache.py` (SHA-256
`bf47f244406e149a3c7fe51d42d326d63a008733d55868b51a73112052e3bcdf`)
and is pinned as
`0b513b4dc4f2394f6b1733bb0b74fa40ab59f4a04f6b33601350b2a606c67804`;
the QSA file is byte-identical to the old pair.

`python3 prepare_sglang_overlays.py --materialize-ple` validates the exact
cached Radix revision and its 128 FP8 shards, concatenates them in numeric
`shard_0` through `shard_127` order without constructing the model, and commits
an immutable payload plus deterministic completion marker. It can adopt an
unmarked payload only after hashing all 51,200,245,760 bytes to
`b070f9644adf93794d8a1030584ab705809387e64396a9327a68fa3a3a6666b3`.
The deterministic completion marker is pinned by SHA-256
`f0ef55e4e4dec9b6b936a42af4ca2eb9b2f24ced373b1e216f7a6d507b171665`.
It never downloads or invokes Docker. If an exact payload belongs to another
user, ownership of that one file must be corrected explicitly before rerunning;
the command never changes ownership. `--verify-ple-cache` performs a full
offline recheck. The explicit `readonly` profile mode then validates the marker,
rejects incompatible SGLang weight-loader modes, mounts the cache read-only,
and maps it with `shared=False` while still loading the PLE `weight_scale`.
The upstream [MIT license](https://github.com/hashd1ve/qwen38-flash-next-one-dgx-spark/blob/bf2b7c75870d3703730b6bd8f3bb93dc622c278d/LICENSE)
applies to the vendored patchers.

### Explicit PLE omission ablation

`python3 prepare_sglang_overlays.py --prepare-ple-ablation` builds a separate
overlay pair for matched mapped-versus-omitted experiments. Its `qwen4_exp.py`
applies `qwen38-ple-omission-ablation.py` after the mmap and persistent-cache
patchers and is pinned as
`bcdc2c86aa59784ffe27d53c8d214e56b6aa45c02b1d5841fd956d1f006d6030`.
The QSA overlay remains
`e30566492e1502f94a4c7fed42d90b523bbb662580c628459e6e63c7b5263c75`.

The mapped control uses this same ablation-capable source with the sentinel
absent, so its model graph and exact read-only FP8 PLE payload are unchanged.
The omitted arm requires the single canonical model override
`{"sparkbench_omit_ple":true}`. It verifies the pinned `[2]` PLE-layer layout,
constructs no PLE module, and skips exactly the 138 checkpoint tensors under
`model.language_model.layers.1.ple`: 128 table shards and ten auxiliary
tensors. Any missing, duplicate, or unexpected PLE tensor fails startup.

This is a semantic ablation, not a storage or serving optimization. It removes
trained model parameters and therefore cannot be treated as equivalent to
mapped PLE without a separate quality result. Manifest, runtime, and evidence
admission bind the omitted label to the exact model revision, recipe revision,
two-file overlay pair, embedded draft, and absence of a PLE cache or mount.

## Bounded result

### Historical tiny QSA control

With the QSA file mounted read-only over the exact image, the resolver reported
`flashinfer.decode.trtllm_batch_decode_with_kv_cache` on compute capability
12.1. The pinned real-weight development fixture
`inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B` at revision
`5fbd297b1529cfa7db2510896d1ad77d1bf41e44` kept native QSA enabled and
completed:

- 4 prompt tokens to 8 completion tokens: HTTP `200`, client wall time
  `7.234597 s` including first-use compilation;
- 6 prompt tokens to 32 completion tokens: HTTP `200`, client wall time
  `0.278661 s` after warmup.

The fixture remained BF16, GDN state remained FP32, linear attention used
Triton, CUDA graphs were disabled, and tokenizer initialization was skipped.
The output token IDs were discarded. These are manual admission diagnostics,
not representative TPS, quality, kernel parity, MTP, or full-checkpoint
evidence.

### Full checkpoint

The persistent overlay pair subsequently admitted the exact 125.96 GiB Radix
repository on one DGX Spark by leaving its 51,200,245,760-byte FP8 PLE payload
on NVMe and mapping it read-only. The served composition was main-model NVFP4,
trained BF16 `NEXTN`, and exact FP8 PLE; the PLE was not NVFP4.

Run
[`20e1283b`](../../evidence/runs/20260827T032027Z-qwen38-flash-next-nvfp4-mtp-sglang-qwen38-flash-next-sglang-native-20e1283b/summary.json)
completed all 12 planned cases with evidence status `partial` because the
bounded quality battery scored 3/4. Startup was 581.652 seconds, including
420.36 seconds for the target and 83.86 seconds for trained MTP. Sampled
`MemAvailable` stayed at or above 16.564 GiB and swap did not grow.

| Case | Measured result |
| --- | ---: |
| D256 / fresh C1 | 28.504 / 27.413 tok/s |
| Fresh C2 / C4 | 50.330 / 72.821 tok/s |
| 8K / 32K prefill | 2,103.468 / 2,179.588 tok/s, client-TTFT approximation |
| 8K / 32K repeated-word needle | Pass / pass |

Periodic logs provided 30 draft-acceptance samples with mean accepted length
2.956 and mean acceptance rate 0.653. Those observations prove draft activity,
not MTP acceleration: there is no MTP-off control or authoritative per-run
accepted/proposed-token aggregate.

A separate repeated-word 131K needle case passed. The 245K lane did not: the
0.85 pool was insufficient, 0.87 caused pressure, and the target-only/BF16-state
246,272-token-cap run
[`7b88e52c`](../../evidence/runs/20260827T030636Z-qwen38-flash-next-nvfp4-long-sglang-qwen38-flash-next-sglang-long-context-7b88e52c/summary.json)
was aborted after sampled `MemAvailable` reached 0.046 GiB; the operator
observed roughly 6.1 GiB swap growth and memory PSI full `avg10` 19.84. That
profile is incompatible. All long prompts repeat one synthetic word, so the
passes demonstrate serving mechanics and exact-key retention, not
natural-document quality or worst-case cold/varied-token NVMe PLE cost.

## Validation still required

Before proposing this as supported behavior:

1. Do not extend this historical TRT-LLM overlay. Reproduce the explicit SM121
   Triton fallback at `3681c4e` or a later reviewed descendant, with dispatch
   attestation and varied-token long-context coverage.
2. Add an SM12x GPU parity test for page size 64, BF16, GQA, and sequence
   lengths crossing a page boundary against an independent reference.
3. Repeat the full-checkpoint route through a pinned derived image rather than
   source-file bind mounts.
4. Add an authoritative SGLang accepted/proposed-token aggregate and matched
   MTP-off/depth controls before attributing throughput to `NEXTN`.
5. Treat NVFP4/I4 PLE as a future memory optimization: validate its packed
   loader and fused gather against the measured exact-FP8 baseline.
6. Do not retry 245K until projected and observed reserve close without swap or
   PSI pressure.

See the
[native diagnostics and optimization plan](../../docs/qwen38-flash-next-native-mtp-optimization-2026-08-26.md)
for the stock failure, clean controls, measured full-model result, and remaining
optimization boundaries.
