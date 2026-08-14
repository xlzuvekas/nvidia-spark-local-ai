# Repository Guidelines

## Project Structure & Module Organization

`sparkbench.py` is the manifest-driven CLI, with implementation in `bench/`, model definitions in `manifests/models.toml`, and suites in `manifests/suites/`. `benchmark.py` and root Compose files preserve the Qwen3.8 experiment. Keep tests in `tests/`, measurements in `BENCHMARK.md`, and analysis under `docs/`. Do not commit generated `data/`, `logs/`, or `results/` content.

## Build, Test, and Development Commands

- `python3 sparkbench.py inventory --sizes` inventories Hugging Face, Ollama, and Docker artifacts without downloading.
- `python3 sparkbench.py list --verbose` shows manifest configurations and cache availability.
- `python3 sparkbench.py benchmark qwen38-27b-nvfp4-mtp3` runs smoke; select `quick.toml` or `core.toml` with `--suite` for performance work.
- `python3 sparkbench.py matrix --backend ollama --task chat` runs cached chat profiles sequentially.
- `python3 sparkbench.py plan <model>` freezes a plan; `run`/`resume` executes it and `summarize` rebuilds reports.
- `python3 -m unittest discover -s tests -v` runs the unit suite.
- `python3 -m py_compile sparkbench.py benchmark.py bench/*.py` checks syntax.
- `docker compose -f compose.nvfp4.yaml -f compose.nvfp4-mtp.yaml up -d` starts the fastest original configuration; `python3 benchmark.py` exercises that live endpoint.

Run one inference configuration at a time. Managed vLLM uses port 8000; SparkBench refuses unrelated workloads and blocks uncached downloads unless `--allow-download` is explicit.

## Coding Style & Naming Conventions

Use four-space indentation, standard-library Python, `snake_case` names, `UPPER_CASE` constants, and type annotations for public APIs. Keep manifests schema-versioned, reject unknown keys, and use stable lowercase IDs such as `decode-256`. Align Compose options across overlays. Include units, hardware, software versions, and dates with results.

## Testing Guidelines

There is no coverage threshold. Run unittest discovery and compile all Python modules. Load modified manifests through SparkBench, validate changed Compose combinations with `docker compose ... config --quiet`, and run smoke for serving changes. Unit tests must not pull models, stop services, or require a GPU. Separate warm and cold behavior and use unique uncached-prefill prompts.

## Commit & Pull Request Guidelines

The history uses concise, imperative commit subjects (for example, `Publish DGX Spark local AI experiments`). Keep each commit focused. Pull requests should explain the configuration changed, tested hardware and image/version, exact reproduction commands, and measured impact. Link relevant issues and update `BENCHMARK.md` or `docs/` when conclusions change; include log excerpts instead of screenshots unless visual evidence is necessary.

## Security & Configuration

Pass Hugging Face credentials through `HF_TOKEN`; never commit tokens, downloaded weights, caches, or machine-specific secrets. Preserve the loopback-only API binding unless broader network exposure is explicitly required and secured.
