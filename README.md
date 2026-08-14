# NVIDIA Spark Local AI Experiments

Reproducible local inference experiments on NVIDIA DGX Spark / GB10. This
repository records the configurations that worked, the ones that did not, and
the measurements behind each conclusion.

The first experiment serves Qwen3.8-27B through an OpenAI-compatible vLLM API.
Moving from BF16 to NVFP4 weights and enabling the model's built-in MTP head
increased sustained decode from **3.91 to 16.04 tokens/second** on one Spark.

| Configuration | Weight memory | Median decode | Relative speed |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B BF16 | 51.1 GiB | 3.91 tok/s | 1.0x |
| Qwen3.8-27B NVFP4 | 24.18 GiB | 8.41 tok/s | 2.15x |
| Qwen3.8-27B NVFP4 + MTP | 24.97 GiB | **16.04 tok/s** | **4.10x** |

Read the [full Qwen3.8-27B write-up](docs/qwen38-27b.md) for the architecture,
benchmark methodology, tuning sweep, failed experiments, and why MTP matters
so much on Spark. Raw measurements live in [BENCHMARK.md](BENCHMARK.md).

## Tested system

- NVIDIA GB10, compute capability 12.1
- 128 GB unified LPDDR5X memory
- Ubuntu 24.04 on ARM64
- CUDA 13 and NVIDIA driver 580.142
- vLLM `0.1.dev19754+g3a0914114`
- `vllm/vllm-openai:qwen38` ARM64 image

The tested image is publicly available on Docker Hub. Its ARM64 manifest at
the time of testing was
`sha256:541e0e475418de6178b45c0d9ef420fb6be79bf43130a4d552cb668e425f4d27`.
The Compose files use the moving `qwen38` tag so that new clones receive fixes;
pin the digest if exact historical reproduction is more important.

## Quick start: fastest configuration

Prerequisites are Docker Engine, Docker Compose, and the NVIDIA Container
Toolkit configured for Docker. Model weights are downloaded from Hugging Face
on first start. Set `HF_TOKEN` if the repository or your environment requires
authentication.

```bash
git clone https://github.com/xlzuvekas/nvidia-spark-local-ai.git
cd nvidia-spark-local-ai
docker compose -f compose.nvfp4.yaml -f compose.nvfp4-mtp.yaml up -d
docker compose -f compose.nvfp4.yaml -f compose.nvfp4-mtp.yaml logs -f
```

The API listens only on the local machine at `http://127.0.0.1:8000/v1`.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.8-27B",
    "messages": [{"role": "user", "content": "Explain unified memory briefly."}],
    "max_tokens": 128,
    "temperature": 0.7,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Model weights are retained under `data/huggingface/`. Compiled vLLM and
FlashInfer kernels are retained under `data/vllm/`, avoiding repeated
compilation and autotuning after container recreation.

## Other configurations

BF16 baseline:

```bash
docker compose -f compose.yaml up -d
```

NVFP4 weights without speculative decoding:

```bash
docker compose -f compose.nvfp4.yaml up -d
```

Override MTP depth or the scheduler budget for experiments:

```bash
MTP_TOKENS=4 MAX_BATCHED_TOKENS=8192 \
  docker compose -f compose.nvfp4.yaml -f compose.nvfp4-mtp.yaml up -d
```

Only one configuration should own port 8000 at a time. Compose will recreate
the shared `qwen38-27b` container when switching configurations.

## Benchmarking

The original dependency-free client reproduces the focused Qwen3.8 experiment
against a server already listening on port 8000:

```bash
python3 benchmark.py
python3 benchmark.py --temperature 0
```

The benchmark uses unique prompts for prefill probes so prefix caching cannot
inflate the result. Decode throughput is reported separately from TTFT and
end-to-end throughput.

For repeatable multi-model work, SparkBench adds manifest-driven inventory,
planning, lifecycle checks, telemetry, resumable execution, and JSON/CSV reports:

```bash
python3 sparkbench.py inventory --sizes
python3 sparkbench.py list --verbose
python3 sparkbench.py benchmark qwen38-27b-nvfp4-mtp3
python3 sparkbench.py benchmark qwen38-27b-nvfp4-mtp3 \
  --suite manifests/suites/quick.toml
python3 sparkbench.py matrix --backend ollama --task chat
```

The default `smoke.toml` suite performs quick endpoint and capability checks;
`quick.toml` is a short performance screen; `core.toml` runs the repeated decode,
prefill, concurrency, long-context, and structured-output matrix. Dedicated
`embeddings.toml` and `vision.toml` suites keep unlike throughput units separate.
Hugging Face and Docker image downloads are disabled unless `--allow-download`
is supplied. SparkBench also refuses to benchmark while unrelated containers or
GPU compute processes are active; it does not stop them automatically.

Ollama chat runs use its native API so the frozen plan controls context size and
records server-reported prompt/decode durations. Cached large-model profiles use
32K context to fit safely in unified memory. OpenAI-compatible SSE results are
explicitly labeled as client estimates when chunks cannot be mapped one-to-one
to tokens.

SparkBench addresses Ollama over loopback but does not reconfigure the host
service. Verify its listener with `ss -ltnp 'sport = :11434'` and restrict
`OLLAMA_HOST` separately if the daemon is exposed beyond the machine.

Each run freezes a `plan.json` under `results/` and records raw events, telemetry,
server provenance/logs, and generated `summary.json`/`summary.csv`. A stopped run
can be resumed or summarized without changing its plan:

```bash
python3 sparkbench.py run results/<run-directory>
python3 sparkbench.py summarize results/<run-directory>
```

Model configurations live in `manifests/models.toml`, suites in
`manifests/suites/`, and pipeline modules in `bench/`. See the
[benchmark strategy](docs/benchmark-strategy.md) and
[local model inventory](docs/local-model-inventory.md) for scope and current
cache coverage. The first broad cached sweep is summarized in
[the 2026-08-14 smoke results](docs/benchmark-results-2026-08-14.md).

Run the dependency-free unit suite before changing the pipeline:

```bash
python3 -m unittest discover -s tests -v
```

## Current conclusions

- NVFP4 weights are the largest straightforward win on GB10.
- MTP depth 3 is the best tested single-stream setting for this model.
- FP8 remains the supported KV-cache format on SM121 in the tested vLLM build.
- NVFP4 KV-cache code exists, but this build gates its FlashInfer path to SM100.
- FP4 KV would mostly benefit long-context capacity, not short-context decode,
  where repeatedly reading model weights dominates.
- Spark's CPU and GPU share memory bandwidth. Unrelated memory-heavy CPU work
  can therefore compete directly with inference.

## Repository roadmap

Future experiments may cover newer vLLM builds, TensorRT-LLM, long-context KV
cache formats, concurrent serving, embeddings, multimodal models, power and
thermal behavior, and agent workloads. Results should include exact versions,
commands, warm-up behavior, and unsuccessful attempts—not just peak numbers.

## License

MIT. Model weights and container images retain their own licenses.
