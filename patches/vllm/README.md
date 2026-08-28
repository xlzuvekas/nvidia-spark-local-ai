# Experimental vLLM PLE live-token-width backport

This directory preserves one semantic backport for the direct-PLE-mmap vLLM
experiment. It limits Qwen3.8-Flash-Next PLE hash work to the live physical
token width. It is not a cherry-pick, an unmodified reproduction of either
upstream branch, or a measured result.

- mmap base commit: `8e4e036a311604800334989485b4ee23925956da`
- mmap base tree: `fb926ea1d0b897caa4f26a5885d6867f72b67905`
- semantic source commit: `4df2ce22d086007a81930d93b3b657a1d197aecc`
- semantic source parent: `4ab6e99d246478a0f0a1f694b0b19d2c649eaf1b`
- target: `vllm/models/qwen4_exp/nvidia/ple_layer.py`
- base blob: `2663a60bf4f077ef70111269164b51f2fe99c32a`
- base file SHA-256: `956239e7754098b480734a0b8e0a32447333c41fc22645e2035bc858781e6738`
- patch: `8e4e036-4df2ce2-ple-live-token-width.patch`
- patch SHA-256: `ab6804086965c89cab7018abaa2de61445e43c5879ceb1ace2eb2ce0a7ea93bd`
- patched blob: `c6f3a811cd4e558aa75d96d289dce6c05ba4eff1`
- patched file SHA-256: `f887c4925d021c5a0db4c059ac1520065a47bce804ed7902f8bd8630b1a4f688`
- patched tree: `3735794739795044e792066854d601d429ba0836`

The newer model-support branch accepted the same one-line semantic change in
[`4df2ce2`](https://github.com/peakcrosser7/vllm/commit/4df2ce22d086007a81930d93b3b657a1d197aecc),
but that commit is absent from the mmap head. Its original base blob differs,
so this repository pins a minimal patch constructed against `8e4e036` instead
of vendoring the upstream format-patch. The later hash-only graph split in
`0e0802f` is intentionally excluded.

## Fail-closed application

Set `VLLM_SOURCE` to a clean detached checkout and `VLLM_PATCH` to the tracked
patch, then verify and apply without fuzzy or three-way fallback:

```bash
test "$(git -C "$VLLM_SOURCE" rev-parse HEAD)" = \
  8e4e036a311604800334989485b4ee23925956da
test "$(git -C "$VLLM_SOURCE" rev-parse HEAD^{tree})" = \
  fb926ea1d0b897caa4f26a5885d6867f72b67905
test -z "$(git -C "$VLLM_SOURCE" status --porcelain=v1)"
test "$(git -C "$VLLM_SOURCE" rev-parse \
  HEAD:vllm/models/qwen4_exp/nvidia/ple_layer.py)" = \
  2663a60bf4f077ef70111269164b51f2fe99c32a
test "$(sha256sum "$VLLM_PATCH" | cut -d' ' -f1)" = \
  ab6804086965c89cab7018abaa2de61445e43c5879ceb1ace2eb2ce0a7ea93bd

git -C "$VLLM_SOURCE" apply --check --index "$VLLM_PATCH"
git -C "$VLLM_SOURCE" apply --index "$VLLM_PATCH"
git -C "$VLLM_SOURCE" diff --cached --check

test "$(git -C "$VLLM_SOURCE" hash-object \
  vllm/models/qwen4_exp/nvidia/ple_layer.py)" = \
  c6f3a811cd4e558aa75d96d289dce6c05ba4eff1
test "$(git -C "$VLLM_SOURCE" write-tree)" = \
  3735794739795044e792066854d601d429ba0836
git -C "$VLLM_SOURCE" apply --reverse --check "$VLLM_PATCH"
```

Do not use `--3way`, `patch -p1`, or apply this artifact to another source
identity. Build and measurement remain post-campaign work. Before promotion,
run the complete upstream PLE-mmap tests, a poisoned-tail/live-width property
test, same-token-count/different-request-layout replay, real GPU
compile/capture/replay, the mmap-versus-`safe_open` row oracle, and the frozen
C1/task ABBA protocol in the
[direct-mmap reproduction plan](../../docs/qwen38-flash-next-vllm-mmap-reproduction-2026-08-28.md).
