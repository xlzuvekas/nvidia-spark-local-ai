# Nemotron Diffusion Direct Benchmark

The cached Nemotron Labs Diffusion checkpoints are not valid vLLM chat
profiles. The 8B snapshot is incomplete because its configured remote modeling
module is absent. The 14B custom architecture is resolved by the pinned vLLM
image as pooling/embedding, not diffusion-language generation. Both legacy
vLLM rows are therefore marked `incompatible`, expose only the `diffusion`
capability, and fail closed before server startup.

## Supported Offline Path

Run the dedicated 14B Transformers adapter after other matrix work finishes:

```bash
python3 sparkbench.py diffusion-direct \
  nemotron-labs-diffusion-14b-transformers-direct
```

The command takes the same non-blocking `results/.sparkbench.lock` as a normal
run, rejects active containers/GPU work in preflight, and launches one
foreground worker with a 7,200-second default deadline. It uses the existing
`/home/xlz/AGENTIC-RESEARCH/diffusion-stuff/.venv` only as a Python/CUDA
runtime; all executed benchmark logic is versioned and SHA-256 frozen in this
repository. Hugging Face and Transformers offline modes are mandatory, proxy
and token variables are removed, and the worker loads the exact cached 14B
revision by snapshot path.

Before CUDA starts, the adapter recomputes SHA-256 for the 25.3 GiB weight,
configuration, tokenizer, chat template, and all three custom Python modules.
It also verifies the runtime configuration, worker code hash, package versions,
fixed prompt hash, seed `3407`, block length `32`, threshold `0.9`, BF16, and
greedy temperature `0`.

## Results and Interpretation

Each run writes an immutable plan, `events.jsonl`, telemetry, worker logs,
worker result, and `summary.json`. Reported measurements include model load
time, output tokens, end-to-end block-generation wall time and output tokens/s,
NFE, NFE/output-token, output-tokens/NFE, peak CUDA memory, sampled host/GPU
telemetry when available, deterministic output hashes, and proof that the
timeout-bounded worker process was reaped. Block-specific fields include
completed blocks, blocks/s, mean end-to-end seconds/block, and NFE/block.

These are block-generation metrics. Do not interpret them as autoregressive
decode TPS, TPOT, ITL, or conventional TTFT. Direct generation returns only a
completed token tensor and NFE, so prefill timing and time-to-first-token are
not available.
