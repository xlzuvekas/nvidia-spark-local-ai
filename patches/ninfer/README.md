# Experimental NInfer SM121a Port

This directory preserves the exact, bounded NInfer source patch used for the
DGX Spark measurements. It is an experimental local port, not upstream NInfer
support.

- Upstream repository: `https://github.com/Neroued/ninfer`
- Upstream commit: `5f45a26f81b6a15805a3d4d09d5c3d60f420b210`
- Patch: `5f45a26f-sm121a.patch`
- Patch SHA-256: `6903090db8a04784147f858f0e29444579032a2da8a3f4a4737d86bd3563f6be`
- Target: DGX Spark GB10, compute capability 12.1 (`sm_121a`)

The patch changes only three upstream files. It pins the ARM64 CUDA 13.1.2
build/runtime image manifests, builds `ninfer_bench`, changes the CMake target
from `120a` to `121a`, and changes the Qwen3.6-family runtime device gate from
SM120 to SM121. Qwen3.8 uses that same execution package.

Apply it only to the exact commit:

```bash
STACK_REPO=/path/to/nvidia-spark-local-ai
NINFER_SOURCE=/path/to/ninfer
git clone https://github.com/Neroued/ninfer.git "$NINFER_SOURCE"
git -C "$NINFER_SOURCE" checkout 5f45a26f81b6a15805a3d4d09d5c3d60f420b210
git -C "$NINFER_SOURCE" apply --check "$STACK_REPO/patches/ninfer/5f45a26f-sm121a.patch"
git -C "$NINFER_SOURCE" apply "$STACK_REPO/patches/ninfer/5f45a26f-sm121a.patch"
```

The GB10 experiments also force eager execution and
`--prefill-chunk 128`. Those runtime constraints are not encoded in this
source patch and are not claims of stock support. See
[`docs/moe-landscape-2026-08-17.md`](../../docs/moe-landscape-2026-08-17.md)
for the admission boundary, exact image identity, and measured results.
