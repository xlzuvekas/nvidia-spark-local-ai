# Repository Guidelines

## Project Structure & Module Organization

`sparkbench.py` is the CLI; code lives in `bench/`, profiles in `manifests/models.toml`, suites in `manifests/suites/`, and tests in `tests/`. `benchmark.py` and Compose files preserve the original Qwen3.8 experiment. Keep protocol measurements in `BENCHMARK.md` and analysis in `docs/`; the campaign report is `docs/benchmark-results-2026-08-16.md`. Never commit generated `data/`, `logs/`, or `results/` content.

## Build, Test, and Development Commands

- `python3 sparkbench.py inventory --sizes` and `python3 sparkbench.py list --verbose` inspect caches and profiles.
- `python3 sparkbench.py fetch <model>` explicitly acquires one pinned Hugging Face snapshot.
- `python3 sparkbench.py benchmark <model> --suite manifests/suites/core.toml` runs one profile; `python3 sparkbench.py matrix --backend ollama --task chat` runs a filtered sequence.
- `python3 sparkbench.py diffusion-direct nemotron-labs-diffusion-14b-transformers-direct` runs native block diffusion; `python3 sparkbench.py trtllm-direct phi-4-multimodal-instruct-fp8-trtllm-audio` runs pinned offline ASR.
- `python3 sparkbench.py benchmark qwen38-27b-ud-q4-k-xl-llamacpp-mtp3 --suite manifests/suites/core.toml` runs the pinned native GGUF/MTP profile.
- The `python3 sparkbench.py` subcommands `plan`, `run`, `resume`, and `summarize` manage frozen runs; `python3 sparkbench.py audit-matrix results/matrices/<id>` verifies journals read-only.
- `python3 -m unittest discover -s tests -v` and `python3 -m py_compile sparkbench.py benchmark.py bench/*.py tests/*.py` validate code.
- `docker compose -f compose.nvfp4.yaml -f compose.nvfp4-mtp.yaml up -d` plus `python3 benchmark.py` preserves the original path.

Run one inference configuration at a time. SparkBench refuses unrelated GPU/container workloads and blocks implicit downloads unless `--allow-download` is explicit. Native llama.cpp runs must retain pinned artifact hashes and offline loopback cleanup.

## Coding Style & Naming Conventions

Use four-space indentation, standard-library Python, `snake_case`, `UPPER_CASE` constants, and type annotations for public APIs. Keep manifests schema-versioned, reject unknown keys, use stable lowercase IDs such as `decode-256`, and align Compose overlays. Include units and exact hardware, model revision, image digest, and date with results.

## Testing Guidelines

There is no coverage threshold. Run unittest discovery and compile Python modules. Load changed manifests, validate Compose with `docker compose ... config --quiet`, and smoke-test serving changes. Unit tests must not download, stop services, use Docker, or require a GPU. Separate warm/cold behavior and use unique prefill prompts.

## Commit & Pull Request Guidelines

Use concise, imperative commit subjects. Keep commits focused. Pull requests should identify the configuration, hardware and exact versions, reproduction commands, measured impact, and failed/partial cases. Link issues and update `BENCHMARK.md` or `docs/` when conclusions change.

## Security & Configuration

Pass Hugging Face credentials through `HF_TOKEN`; never commit tokens, weights, caches, raw media payloads, or machine secrets. Preserve loopback-only APIs unless broader exposure is explicitly secured.
