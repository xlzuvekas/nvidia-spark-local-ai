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
