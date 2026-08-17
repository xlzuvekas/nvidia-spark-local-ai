# Repository Guidelines

## Structure

`sparkbench.py` is the CLI; modules live in `bench/`, profiles in `manifests/models.toml`, suites in `manifests/suites/`, and tests in `tests/`. `benchmark.py` and Compose preserve the original experiment. Keep protocols in `BENCHMARK.md`, analysis in `docs/`, and generated `data/`, `logs/`, and `results/` out of Git. `docs/benchmark-results-2026-08-17.md` covers Qwen3.6 MTP2, Qwen3.8 DSpark/Q5/262K, perplexity, and Muse DFlash admission.

## Commands

- `python3 sparkbench.py inventory --sizes` and `python3 sparkbench.py list --verbose` inspect caches and profiles.
- `python3 sparkbench.py fetch <model>` acquires pinned model and optional draft snapshots.
- `python3 sparkbench.py benchmark <model> --suite manifests/suites/core.toml` runs one profile; `python3 sparkbench.py matrix --backend ollama --task chat` runs a filtered sequence.
- `python3 sparkbench.py diffusion-direct nemotron-labs-diffusion-14b-transformers-direct` and `python3 sparkbench.py trtllm-direct phi-4-multimodal-instruct-fp8-trtllm-audio` run offline direct adapters.
- `python3 sparkbench.py perplexity <model> --dataset <path> --chunks 64 --ctx-size 512 --timeout 3600` records matched perplexity.
- `python3 sparkbench.py plan`, `python3 sparkbench.py run`, `python3 sparkbench.py resume`, and `python3 sparkbench.py summarize` manage frozen runs; `python3 sparkbench.py audit-matrix results/matrices/<id>` audits them.
- `python3 content_battery.py --base-url <loopback-url> --model <served-id> --api-key-file <path> --output <path>` measures an existing server without managing it.
- `python3 -m unittest discover -s tests -v` and `python3 -m py_compile sparkbench.py benchmark.py bench/*.py tests/*.py` validate code.
- `docker compose -f compose.nvfp4.yaml -f compose.nvfp4-mtp.yaml up -d` plus `python3 benchmark.py` preserves the original path.

Run one inference configuration at a time. SparkBench refuses unrelated GPU/container workloads and blocks implicit downloads unless `--allow-download` is explicit. Retain pinned artifact hashes and offline loopback cleanup.

## Style

Use four-space indentation, standard-library Python, `snake_case`, `UPPER_CASE` constants, and public-API type annotations. Keep manifests schema-versioned, reject unknown keys, and use stable lowercase IDs. Record units, hardware, revision, image digest, and date.

## Testing Guidelines

Run unittest discovery and compile Python. Load changed manifests, validate Compose with `docker compose ... config --quiet`, and smoke-test serving changes. Unit tests must not download, stop services, use Docker, or require a GPU. Use unique prefill prompts. Perplexity comparisons require the same base model, dataset hash, runtime, chunks, and context.

## Commits

Use concise, imperative commit subjects. Pull requests identify configuration, hardware, exact versions, reproduction commands, impact, and failed or partial cases. Update `BENCHMARK.md` or `docs/` when conclusions change.

## Security

Pass Hugging Face credentials through `HF_TOKEN`; never commit tokens, weights, caches, raw media, captured prompts or completions, request tags, API keys, or secrets. Versioned benchmark fixtures allowed. Preserve loopback. Content-battery evidence must remain scalar-only and use `--api-key-file`; redact credentials from logs and provenance.
