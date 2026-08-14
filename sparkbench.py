#!/usr/bin/env python3
"""Command-line entry point for the DGX Spark benchmark pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import fnmatch
import json
from pathlib import Path
import sys

from bench.inventory import (
    assess_model_availability,
    collect_inventory,
    discover_huggingface_snapshots,
    inventory_to_dict,
)
from bench.manifest import ManifestError, load_models, load_suite
from bench.journal import utc_now, write_json
from bench.report import summarize_run
from bench.runner import create_plan, execute_plan


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_MODELS = WORKSPACE / "manifests" / "models.toml"
DEFAULT_SUITE = WORKSPACE / "manifests" / "suites" / "smoke.toml"


def _inventory(*, sizes: bool = False):
    inventory = collect_inventory(
        huggingface_root=WORKSPACE / "data" / "huggingface" / "hub",
        calculate_sizes=sizes,
    )
    user_snapshots = discover_huggingface_snapshots(
        Path.home() / ".cache" / "huggingface" / "hub",
        calculate_sizes=sizes,
    )
    known = {(item.source, item.revision, item.path) for item in inventory.huggingface_snapshots}
    combined = inventory.huggingface_snapshots + tuple(
        item
        for item in user_snapshots
        if (item.source, item.revision, item.path) not in known
    )
    return replace(inventory, huggingface_snapshots=combined)


def command_inventory(args: argparse.Namespace) -> int:
    inventory = _inventory(sizes=args.sizes)
    payload = inventory_to_dict(inventory)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Hugging Face snapshots: {len(inventory.huggingface_snapshots)}")
    print(f"Ollama models: {len(inventory.ollama_models)} (server {inventory.ollama_version or 'offline'})")
    print(f"Docker images: {len(inventory.docker_images)}")
    for snapshot in inventory.huggingface_snapshots:
        size = f" ({snapshot.size_bytes / 1024**3:.2f} GiB)" if snapshot.size_bytes else ""
        print(f"  hf     {snapshot.source}@{snapshot.revision[:12]}{size}")
    for model in inventory.ollama_models:
        size = f" ({model.size_bytes / 1024**3:.2f} GiB)" if model.size_bytes else ""
        print(f"  ollama {model.name}@{model.revision[:12]}{size}")
    for error in inventory.errors:
        print(f"warning: {error}", file=sys.stderr)
    return 0


def command_list(args: argparse.Namespace) -> int:
    models = load_models(args.models)
    inventory = _inventory()
    availability = assess_model_availability(models, inventory)
    for model in models.values():
        local = availability[model.id]
        marker = "cached" if local.available else "unavailable"
        tasks = ",".join(model.tasks)
        print(f"{model.id:42} {model.backend:8} {marker:11} {tasks:24} {model.source}")
        if args.verbose:
            native = f"/{model.native_context}" if model.native_context else ""
            metadata = (
                f"support={model.support_status}; served/native context="
                f"{model.max_context}{native}"
            )
            details = "; ".join((metadata, *local.details))
            print("  " + details)
    return 0


def _select(args: argparse.Namespace):
    models = load_models(args.models)
    try:
        model = models[args.model]
    except KeyError as error:
        choices = ", ".join(sorted(models))
        raise ManifestError(f"unknown model {args.model!r}; choices: {choices}") from error
    suite = load_suite(args.suite)
    return model, suite


def command_plan(args: argparse.Namespace) -> int:
    model, suite = _select(args)
    run_dir = create_plan(
        model=model,
        suite=suite,
        results_root=args.results,
        models_path=args.models,
        suite_path=args.suite,
    )
    print(run_dir)
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    model, suite = _select(args)
    run_dir = create_plan(
        model=model,
        suite=suite,
        results_root=args.results,
        models_path=args.models,
        suite_path=args.suite,
    )
    print(f"Plan: {run_dir}")
    summary = execute_plan(
        run_dir,
        workspace=WORKSPACE,
        allow_download=args.allow_download,
        keep_server=args.keep_server,
        continue_on_error=not args.fail_fast,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 1


def command_run(args: argparse.Namespace) -> int:
    summary = execute_plan(
        args.run_dir,
        workspace=WORKSPACE,
        allow_download=args.allow_download,
        keep_server=args.keep_server,
        continue_on_error=not args.fail_fast,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 1


def command_matrix(args: argparse.Namespace) -> int:
    models = load_models(args.models)
    suite = load_suite(args.suite)
    availability = assess_model_availability(models, _inventory())
    selected = []
    for model in models.values():
        if args.backend and model.backend not in args.backend:
            continue
        if args.task and not set(args.task).issubset(model.tasks):
            continue
        if args.match and not fnmatch.fnmatch(model.id, args.match):
            continue
        if not args.allow_download and not availability[model.id].available:
            continue
        selected.append(model)
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise ManifestError("matrix filters selected no runnable model configurations")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    matrix_dir = args.results / "matrices" / f"{stamp}-{suite.id}"
    matrix_dir.mkdir(parents=True, exist_ok=False)
    index: dict[str, object] = {
        "created_at": utc_now(),
        "suite": suite.id,
        "models": [model.id for model in selected],
        "runs": [],
    }
    index_path = matrix_dir / "matrix.json"
    write_json(index_path, index)
    failures = 0
    for model in selected:
        run_dir = create_plan(
            model=model,
            suite=suite,
            results_root=matrix_dir,
            models_path=args.models,
            suite_path=args.suite,
        )
        entry: dict[str, object] = {
            "model": model.id,
            "run_dir": str(run_dir),
            "status": "planned" if args.plan_only else "running",
        }
        index["runs"].append(entry)  # type: ignore[union-attr]
        write_json(index_path, index)
        if args.plan_only:
            print(f"planned {model.id}: {run_dir}")
            continue
        print(f"benchmarking {model.id}: {run_dir}", flush=True)
        try:
            summary = execute_plan(
                run_dir,
                workspace=WORKSPACE,
                allow_download=args.allow_download,
                keep_server=False,
                continue_on_error=not args.fail_fast,
            )
        except Exception as error:
            failures += 1
            entry.update(
                {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
            )
            write_json(index_path, index)
            if args.fail_fast:
                raise
        else:
            entry.update(
                {"status": summary["status"], "completed_cases": summary["completed_cases"]}
            )
            if summary["status"] != "complete":
                failures += 1
            write_json(index_path, index)
    print(matrix_dir)
    return 1 if failures else 0


def command_summarize(args: argparse.Namespace) -> int:
    print(json.dumps(summarize_run(args.run_dir), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="inspect local model and runtime caches")
    inventory.add_argument("--sizes", action="store_true", help="calculate snapshot disk usage")
    inventory.add_argument("--output", type=Path, help="also write inventory JSON")
    inventory.set_defaults(function=command_inventory)

    list_parser = subparsers.add_parser("list", help="list manifest model configurations")
    list_parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    list_parser.add_argument("--verbose", action="store_true")
    list_parser.set_defaults(function=command_list)

    def add_selection(command: argparse.ArgumentParser) -> None:
        command.add_argument("model", help="model configuration ID")
        command.add_argument("--models", type=Path, default=DEFAULT_MODELS)
        command.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
        command.add_argument("--results", type=Path, default=WORKSPACE / "results")

    plan = subparsers.add_parser("plan", help="freeze an immutable single-model plan")
    add_selection(plan)
    plan.set_defaults(function=command_plan)

    benchmark = subparsers.add_parser("benchmark", help="create and immediately execute a plan")
    add_selection(benchmark)
    benchmark.add_argument("--allow-download", action="store_true")
    benchmark.add_argument("--keep-server", action="store_true")
    benchmark.add_argument("--fail-fast", action="store_true")
    benchmark.set_defaults(function=command_benchmark)

    run = subparsers.add_parser("run", aliases=["resume"], help="execute or resume a frozen plan")
    run.add_argument("run_dir", type=Path)
    run.add_argument("--allow-download", action="store_true")
    run.add_argument("--keep-server", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.set_defaults(function=command_run)

    matrix = subparsers.add_parser("matrix", help="plan or run a filtered model matrix sequentially")
    matrix.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    matrix.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    matrix.add_argument("--results", type=Path, default=WORKSPACE / "results")
    matrix.add_argument("--backend", action="append", choices=["vllm", "ollama", "external"])
    matrix.add_argument("--task", action="append", help="require a declared task (repeatable)")
    matrix.add_argument("--match", default="*", help="model-ID glob")
    matrix.add_argument("--limit", type=int)
    matrix.add_argument("--plan-only", action="store_true")
    matrix.add_argument("--allow-download", action="store_true")
    matrix.add_argument("--fail-fast", action="store_true")
    matrix.set_defaults(function=command_matrix)

    summarize = subparsers.add_parser("summarize", help="regenerate JSON and CSV summaries")
    summarize.add_argument("run_dir", type=Path)
    summarize.set_defaults(function=command_summarize)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.function(args))
    except (ManifestError, RuntimeError, OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
