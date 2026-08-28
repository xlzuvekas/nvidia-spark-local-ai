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

from bench.annotations import (
    STARTUP_SAFETY_GATE_REGISTRY,
    append_measurement_annotation,
    append_startup_safety_gate,
)
from bench.acquire import fetch_model_snapshot
from bench.audit import audit_matrix
from bench.autoresearch_campaign import (
    freeze_campaign,
    preview_campaign,
    run_campaign,
    summarize_campaign,
)
from bench.diffusion_direct import run_direct_diffusion
from bench.evidence import export_evidence, verify_evidence, verify_staged_evidence
from bench.inventory import (
    assess_model_availability,
    collect_inventory,
    discover_huggingface_snapshots,
    inventory_to_dict,
)
from bench.llamacpp_perplexity import run_llamacpp_perplexity
from bench.manifest import (
    ManifestError,
    load_models,
    load_suite,
    validate_benchmark_selection,
)
from bench.prefix_cache_protocol import PREFIX_CACHE_SUITE_ID
from bench.journal import utc_now, write_json
from bench.report import summarize_run
from bench.runner import create_plan, execute_plan
from bench.trtllm_direct import run_direct_trtllm


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_MODELS = WORKSPACE / "manifests" / "models.toml"
DEFAULT_SUITE = WORKSPACE / "manifests" / "suites" / "smoke.toml"
DEFAULT_DIFFUSION_SUITE = (
    WORKSPACE / "manifests" / "suites" / "diffusion_direct.toml"
)
DEFAULT_AUDIO_SUITE = WORKSPACE / "manifests" / "suites" / "audio_asr.toml"
DEFAULT_EVIDENCE = WORKSPACE / "evidence"
DEFAULT_RESULTS = WORKSPACE / "results"


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


def command_fetch(args: argparse.Namespace) -> int:
    models = load_models(args.models)
    try:
        model = models[args.model]
    except KeyError as error:
        choices = ", ".join(sorted(models))
        raise ManifestError(
            f"unknown model {args.model!r}; choices: {choices}"
        ) from error
    result = fetch_model_snapshot(model, workspace=WORKSPACE)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _select(args: argparse.Namespace):
    models = load_models(args.models)
    try:
        model = models[args.model]
    except KeyError as error:
        choices = ", ".join(sorted(models))
        raise ManifestError(f"unknown model {args.model!r}; choices: {choices}") from error
    suite = load_suite(args.suite)
    validate_benchmark_selection(model, suite)
    return model, suite


def command_plan(args: argparse.Namespace) -> int:
    model, suite = _select(args)
    if model.support_status == "incompatible":
        raise ManifestError("Incompatible model profiles cannot be planned")
    if model.backend in {"transformers", "trtllm"}:
        command = (
            "diffusion-direct"
            if model.backend == "transformers"
            else "trtllm-direct"
        )
        raise ManifestError(f"Use {command} for direct profiles")
    run_dir = create_plan(
        model=model,
        suite=suite,
        results_root=args.results,
        models_path=args.models,
        suite_path=args.suite,
    )
    print(run_dir)
    return 0


def command_autoresearch_plan(args: argparse.Namespace) -> int:
    preview = preview_campaign(args.campaign, workspace=WORKSPACE)
    if args.dry_run:
        print(json.dumps(preview.to_mapping(), indent=2, sort_keys=True))
        return 0
    campaign_dir = freeze_campaign(
        args.campaign,
        workspace=WORKSPACE,
        results_root=args.results,
    )
    print(campaign_dir)
    return 0


def command_autoresearch_run(args: argparse.Namespace) -> int:
    summary = run_campaign(args.campaign_dir, workspace=WORKSPACE)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] in {"active", "complete"} else 1


def command_autoresearch_summarize(args: argparse.Namespace) -> int:
    summary = summarize_campaign(args.campaign_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    model, suite = _select(args)
    if model.support_status == "incompatible":
        raise ManifestError("Incompatible model profiles cannot be benchmarked")
    if model.backend in {"transformers", "trtllm"}:
        command = (
            "diffusion-direct"
            if model.backend == "transformers"
            else "trtllm-direct"
        )
        raise ManifestError(f"Use {command} for direct profiles")
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


def command_diffusion_direct(args: argparse.Namespace) -> int:
    model, suite = _select(args)
    summary = run_direct_diffusion(
        model=model,
        suite=suite,
        workspace=WORKSPACE,
        results_root=args.results,
        timeout_s=args.timeout,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 1


def command_trtllm_direct(args: argparse.Namespace) -> int:
    model, suite = _select(args)
    summary = run_direct_trtllm(
        model=model,
        suite=suite,
        workspace=WORKSPACE,
        results_root=args.results,
        timeout_s=args.timeout,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 1


def command_perplexity(args: argparse.Namespace) -> int:
    models = load_models(args.models)
    try:
        model = models[args.model]
    except KeyError as error:
        choices = ", ".join(sorted(models))
        raise ManifestError(
            f"unknown model {args.model!r}; choices: {choices}"
        ) from error
    summary = run_llamacpp_perplexity(
        model=model,
        workspace=WORKSPACE,
        results_root=args.results,
        dataset=args.dataset,
        chunks=args.chunks,
        ctx_size=args.ctx_size,
        timeout_s=args.timeout,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 1


def command_matrix(args: argparse.Namespace) -> int:
    models = load_models(args.models)
    suite = load_suite(args.suite)
    availability = assess_model_availability(models, _inventory())
    selected = []
    for model in models.values():
        if model.support_status == "incompatible" or model.backend in {
            "transformers",
            "trtllm",
        }:
            continue
        if args.backend and model.backend not in args.backend:
            continue
        if args.task and not set(args.task).issubset(model.tasks):
            continue
        if args.match and not fnmatch.fnmatch(model.id, args.match):
            continue
        prefix_cache_mode = getattr(model, "prefix_cache_mode", None)
        if suite.id == PREFIX_CACHE_SUITE_ID:
            if prefix_cache_mode is None:
                continue
        elif prefix_cache_mode is not None:
            continue
        if not args.allow_download and not availability[model.id].available:
            continue
        selected.append(model)
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise ManifestError("matrix filters selected no runnable model configurations")
    for model in selected:
        validate_benchmark_selection(model, suite, context="matrix selection")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    matrix_dir = (args.results / "matrices" / f"{stamp}-{suite.id}").resolve()
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
        created_run_dir = create_plan(
            model=model,
            suite=suite,
            results_root=matrix_dir,
            models_path=args.models,
            suite_path=args.suite,
        )
        run_dir = created_run_dir.resolve()
        try:
            relative_run_dir = run_dir.relative_to(matrix_dir)
        except ValueError as error:
            raise RuntimeError(
                f"matrix run directory resolves outside {matrix_dir}: {run_dir}"
            ) from error
        if len(relative_run_dir.parts) != 1:
            raise RuntimeError(
                "matrix run directory must be an immediate child of "
                f"{matrix_dir}: {run_dir}"
            )
        entry: dict[str, object] = {
            "model": model.id,
            "run_dir": relative_run_dir.as_posix(),
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


def command_annotate(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    annotation = append_measurement_annotation(
        run_dir,
        scope=args.scope,
        reason=args.reason,
        case_id=args.case_id,
        evidence=args.evidence,
    )
    summary = summarize_run(run_dir)
    print(
        json.dumps(
            {
                "annotation": annotation,
                "summary_path": str(run_dir / "summary.json"),
                "startup_measurement_valid": summary[
                    "startup_measurement_valid"
                ],
                "measurement_invalid_cases": summary[
                    "measurement_invalid_cases"
                ],
            },
            indent=2,
        )
    )
    return 0


def command_annotate_safety_gate(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    annotation = append_startup_safety_gate(
        run_dir,
        metric=args.metric,
        observed=args.observed,
        limit=args.limit,
        unit=args.unit,
        comparison=args.comparison,
    )
    summary = summarize_run(run_dir)
    print(
        json.dumps(
            {
                "annotation": annotation,
                "summary_path": str(run_dir / "summary.json"),
                "startup_measurement_valid": summary[
                    "startup_measurement_valid"
                ],
                "startup_safety_gates": summary["startup_safety_gates"],
            },
            indent=2,
        )
    )
    return 0


def command_audit_matrix(args: argparse.Namespace) -> int:
    report = audit_matrix(args.matrix_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def command_export_evidence(args: argparse.Namespace) -> int:
    report = export_evidence(
        results_root=args.results,
        output_root=args.output,
        harbor_results=args.harbor_result,
        replace=args.replace,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_verify_evidence(args: argparse.Namespace) -> int:
    report = (
        verify_staged_evidence(repo_root=WORKSPACE, evidence_root=args.evidence)
        if args.staged
        else verify_evidence(args.evidence)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
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

    fetch = subparsers.add_parser(
        "fetch", help="download and verify one pinned Hugging Face model snapshot"
    )
    fetch.add_argument("model", help="model configuration ID")
    fetch.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    fetch.set_defaults(function=command_fetch)

    def add_selection(command: argparse.ArgumentParser) -> None:
        command.add_argument("model", help="model configuration ID")
        command.add_argument("--models", type=Path, default=DEFAULT_MODELS)
        command.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
        command.add_argument("--results", type=Path, default=WORKSPACE / "results")

    plan = subparsers.add_parser("plan", help="freeze an immutable single-model plan")
    add_selection(plan)
    plan.set_defaults(function=command_plan)

    autoresearch_plan = subparsers.add_parser(
        "autoresearch-plan",
        help="validate or freeze the bounded single-user autoresearch campaign",
    )
    autoresearch_plan.add_argument("--campaign", type=Path, required=True)
    autoresearch_plan.add_argument(
        "--results", type=Path, default=WORKSPACE / "results" / "autoresearch"
    )
    autoresearch_plan.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the scalar proposal without writing plans",
    )
    autoresearch_plan.set_defaults(function=command_autoresearch_plan)

    autoresearch_run = subparsers.add_parser(
        "autoresearch-run",
        aliases=["autoresearch-resume"],
        help="execute or replay a frozen autoresearch campaign",
    )
    autoresearch_run.add_argument("campaign_dir", type=Path)
    autoresearch_run.set_defaults(function=command_autoresearch_run)

    autoresearch_summarize = subparsers.add_parser(
        "autoresearch-summarize",
        help="replay and summarize a frozen autoresearch campaign",
    )
    autoresearch_summarize.add_argument("campaign_dir", type=Path)
    autoresearch_summarize.set_defaults(function=command_autoresearch_summarize)

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

    diffusion = subparsers.add_parser(
        "diffusion-direct",
        help="run the certified offline Transformers block-generation adapter",
    )
    add_selection(diffusion)
    diffusion.set_defaults(suite=DEFAULT_DIFFUSION_SUITE)
    diffusion.add_argument(
        "--timeout",
        type=float,
        help="hard worker deadline in seconds (defaults to the model profile)",
    )
    diffusion.set_defaults(function=command_diffusion_direct)

    trtllm = subparsers.add_parser(
        "trtllm-direct",
        help="run the pinned offline TensorRT-LLM Phi audio adapter",
    )
    add_selection(trtllm)
    trtllm.set_defaults(suite=DEFAULT_AUDIO_SUITE)
    trtllm.add_argument(
        "--timeout",
        type=float,
        help="hard container deadline in seconds (defaults to the model profile)",
    )
    trtllm.set_defaults(function=command_trtllm_direct)

    perplexity = subparsers.add_parser(
        "perplexity",
        help="run pinned offline llama.cpp perplexity",
    )
    perplexity.add_argument("model", help="llama.cpp model configuration ID")
    perplexity.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    perplexity.add_argument("--results", type=Path, default=WORKSPACE / "results")
    perplexity.add_argument("--dataset", type=Path, required=True)
    perplexity.add_argument("--chunks", type=int, required=True)
    perplexity.add_argument("--ctx-size", type=int, default=512)
    perplexity.add_argument("--timeout", type=float, required=True)
    perplexity.set_defaults(function=command_perplexity)

    matrix = subparsers.add_parser("matrix", help="plan or run a filtered model matrix sequentially")
    matrix.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    matrix.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    matrix.add_argument("--results", type=Path, default=WORKSPACE / "results")
    matrix.add_argument(
        "--backend",
        action="append",
        choices=["vllm", "llamacpp", "ollama", "sglang", "trtllm", "external"],
    )
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

    annotate = subparsers.add_parser(
        "annotate", help="append a measurement-validity annotation to a frozen run"
    )
    annotate.add_argument("run_dir", type=Path)
    annotate.add_argument(
        "--scope", required=True, choices=("startup", "case")
    )
    annotate.add_argument("--reason", required=True)
    annotate.add_argument("--case-id")
    annotate.add_argument(
        "--evidence",
        action="append",
        default=[],
        help="supporting fact or local artifact reference (repeatable)",
    )
    annotate.set_defaults(function=command_annotate)

    annotate_safety_gate = subparsers.add_parser(
        "annotate-safety-gate",
        help="append a typed startup safety-gate breach to a frozen run",
    )
    annotate_safety_gate.add_argument("run_dir", type=Path)
    annotate_safety_gate.add_argument(
        "--metric", required=True, choices=tuple(STARTUP_SAFETY_GATE_REGISTRY)
    )
    annotate_safety_gate.add_argument("--observed", required=True, type=float)
    annotate_safety_gate.add_argument("--limit", required=True, type=float)
    annotate_safety_gate.add_argument(
        "--unit", required=True, choices=("gib", "mib")
    )
    annotate_safety_gate.add_argument(
        "--comparison", required=True, choices=("gt", "lt")
    )
    annotate_safety_gate.set_defaults(function=command_annotate_safety_gate)

    audit = subparsers.add_parser(
        "audit-matrix", help="read-only verification of a completed matrix"
    )
    audit.add_argument("matrix_dir", type=Path)
    audit.set_defaults(function=command_audit_matrix)

    export = subparsers.add_parser(
        "export-evidence",
        help="publish deterministic scalar-only evidence from ignored raw results",
    )
    export.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    export.add_argument("--output", type=Path, default=DEFAULT_EVIDENCE)
    export.add_argument(
        "--harbor-result",
        action="append",
        default=[],
        type=Path,
        help=(
            "explicit owner-private canonical Harbor lifecycle result; "
            "provide exactly two when publishing Harbor evidence"
        ),
    )
    export.add_argument(
        "--replace",
        action="store_true",
        help="atomically replace an existing evidence tree when measurements changed",
    )
    export.set_defaults(function=command_export_evidence)

    verify = subparsers.add_parser(
        "verify-evidence",
        help="validate the sanitized evidence schema, checksums, and secret policy",
    )
    verify.add_argument("evidence", type=Path, nargs="?", default=DEFAULT_EVIDENCE)
    verify.add_argument(
        "--staged",
        action="store_true",
        help="verify the exact staged evidence tree and scan every staged text blob",
    )
    verify.set_defaults(function=command_verify_evidence)
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
