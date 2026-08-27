# Experimental SGLang QSA SM121 XQA route

This directory preserves the exact two-line SGLang source patch used for the
Qwen3.8-Flash-Next tiny-fixture diagnostic on DGX Spark. It is an experimental
local route, not upstream SGLang support and not proof that the full model fits.

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
That exact pair is retained for already-frozen writable-mmap plans. The current
profile uses the separate
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
It never downloads or invokes Docker. If an exact payload belongs to another
user, ownership of that one file must be corrected explicitly before rerunning;
the command never changes ownership. `--verify-ple-cache` performs a full
offline recheck. The explicit `readonly` profile mode then validates the marker,
rejects incompatible SGLang weight-loader modes, mounts the cache read-only,
and maps it with `shared=False` while still loading the PLE `weight_scale`.
The upstream [MIT license](https://github.com/hashd1ve/qwen38-flash-next-one-dgx-spark/blob/bf2b7c75870d3703730b6bd8f3bb93dc622c278d/LICENSE)
applies to the vendored patchers.

## Bounded result

With the patched file mounted read-only over the exact image, the resolver
reported `flashinfer.decode.trtllm_batch_decode_with_kv_cache` on compute
capability 12.1. The pinned real-weight development fixture
`inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B` at revision
`5fbd297b1529cfa7db2510896d1ad77d1bf41e44` then kept native QSA enabled and
completed:

- 4 prompt tokens to 8 completion tokens: HTTP `200`, client wall time
  `7.234597 s` including first-use compilation;
- 6 prompt tokens to 32 completion tokens: HTTP `200`, client wall time
  `0.278661 s` after warmup.

The fixture remained BF16, GDN state remained FP32, linear attention used
Triton, CUDA graphs were disabled, and tokenizer initialization was skipped.
The output token IDs were discarded. These are manual admission diagnostics,
not representative TPS, quality, kernel parity, MTP, packed-PLE loading, or
full-checkpoint evidence. The full 125.96 GiB Radix checkpoint was not acquired.

## Validation still required

Before proposing this as supported behavior:

1. Add a resolver unit test for SM100 false / SM120 true and the all-false and
   missing-FlashInfer cases.
2. Add an SM12x GPU parity test for page size 64, BF16, GQA, and sequence
   lengths crossing a page boundary against an independent reference.
3. Repeat the native tiny QSA smoke through a pinned derived image rather than
   a source-file bind mount.
4. Run the packed PLE loader and real trained MTP acceptance gates before any
   performance comparison.

See the
[native diagnostics and optimization plan](../../docs/qwen38-flash-next-native-mtp-optimization-2026-08-26.md)
for the stock failure, clean controls, fit gate, and full-model boundaries.
