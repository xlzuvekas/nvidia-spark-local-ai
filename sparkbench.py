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
from bench.audit import (
    audit_matrix,
    audit_sm121_cache_observability_run,
    audit_sm121_cache_semantic_pair,
    audit_sm121_chunked_prefill_performance_campaign,
    audit_sm121_cache_performance_campaign,
    audit_sm121_storage_canary_run,
)
from bench.autoresearch_campaign import (
    freeze_campaign,
    preview_campaign,
    run_campaign,
    summarize_campaign,
)
from bench.autoresearch_v2 import (
    freeze_autoresearch_v2,
    preview_autoresearch_v2,
    run_autoresearch_v2,
    summarize_autoresearch_v2,
)
from bench.diffusion_direct import run_direct_diffusion
from bench.evidence import export_evidence, verify_evidence, verify_staged_evidence
from bench.execution_admission import model_execution_blocker
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
from bench.runner import (
    create_plan,
    create_sm121_cache_observability_plan,
    create_sm121_cache_semantic_pair_plans,
    create_sm121_cache_performance_campaign,
    create_sm121_storage_canary_plan,
    execute_plan,
    execute_sm121_cache_observability_canary,
    execute_sm121_cache_semantic_canary,
    execute_sm121_cache_performance_campaign,
    execute_sm121_storage_canary,
)
from bench.sm121_chunked_prefill_runner import (
    create_sm121_chunked_prefill_performance_campaign,
    execute_sm121_chunked_prefill_performance_campaign,
)
from bench.sglang_sm121_cache_semantic import (
    SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID,
    SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID,
)
from bench.sglang_sm121_cache_performance import (
    SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID,
    SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID,
)
from bench.sglang_sm121_chunked_prefill_performance import (
    SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID,
    SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID,
)
from bench.trtllm_direct import run_direct_trtllm


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_MODELS = WORKSPACE / "manifests" / "models.toml"
DEFAULT_SUITE = WORKSPACE / "manifests" / "suites" / "smoke.toml"
DEFAULT_DIFFUSION_SUITE = (
    WORKSPACE / "manifests" / "suites" / "diffusion_direct.toml"
)
DEFAULT_AUDIO_SUITE = WORKSPACE / "manifests" / "suites" / "audio_asr.toml"
DEFAULT_SM121_STORAGE_CANARY_SUITE = (
    WORKSPACE / "manifests" / "suites" / "qwen38_flash_next_sm121_triton_storage_canary.toml"
)
DEFAULT_SM121_CACHE_OBSERVABILITY_SUITE = (
    WORKSPACE
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sm121_triton_storage_cache_observability_canary.toml"
)
DEFAULT_SM121_CACHE_SEMANTIC_SUITE = (
    WORKSPACE
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sm121_triton_storage_cache_policy_semantic_canary.toml"
)
DEFAULT_SM121_CACHE_PERFORMANCE_SUITE = (
    WORKSPACE
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sm121_triton_storage_cache_policy_performance_v1.toml"
)
DEFAULT_SM121_CHUNKED_PREFILL_PERFORMANCE_SUITE = (
    WORKSPACE
    / "manifests"
    / "suites"
    / "qwen38_flash_next_sm121_triton_storage_chunked_prefill_performance_v1.toml"
)
DEFAULT_AUTORESEARCH_V2_CACHE_POLICY_CAMPAIGN = (
    WORKSPACE
    / "manifests"
    / "campaigns"
    / "qwen38_flash_next_sm121_autoresearch_v2_cache_policy.toml"
)
DEFAULT_EVIDENCE = WORKSPACE / "evidence"
DEFAULT_RESULTS = WORKSPACE / "results"
AUTORESEARCH_CHECKPOINT_READINESS_CODES = frozenset(
    {
        "checkpoint_boundary_unsettled",
        "checkpoint_race",
        "evidence_incomplete",
        "evidence_invalid",
        "evidence_not_current",
        "evidence_pair_binding_mismatch",
        "no_completed_pair",
        "remote_unverified",
        "repository_detached",
        "repository_dirty",
        "repository_not_pushed",
        "repository_unverified",
        "upstream_missing",
    }
)


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
    if summary["status"] == "checkpoint_required":
        return 3
    return 0 if summary["status"] in {"active", "complete"} else 1


def command_autoresearch_checkpoint(args: argparse.Namespace) -> int:
    from bench.autoresearch_campaign import acknowledge_campaign_checkpoint
    from bench.autoresearch_checkpoint import CheckpointError

    try:
        acknowledgement = acknowledge_campaign_checkpoint(
            args.campaign_dir,
            WORKSPACE,
        )
    except CheckpointError as error:
        if error.code not in AUTORESEARCH_CHECKPOINT_READINESS_CODES:
            raise
        print(
            json.dumps(
                {
                    "reason": error.code,
                    "status": "checkpoint_required",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    integrity = acknowledgement.to_mapping()["integrity_hash"]
    print(
        json.dumps(
            {
                "acknowledgement_integrity_sha256": integrity,
                "campaign_id": acknowledgement.campaign.campaign_id,
                "checkpoint_sequence": acknowledgement.completion.sequence,
                "evidence_index_sha256": acknowledgement.evidence.index_sha256,
                "pair_kind": acknowledgement.completion.pair_kind,
                "repository_commit": acknowledgement.repository.head_commit,
                "status": "checkpoint_acknowledged",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_autoresearch_summarize(args: argparse.Namespace) -> int:
    summary = summarize_campaign(args.campaign_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_autoresearch_v2_plan(args: argparse.Namespace) -> int:
    """Preview or freeze one current-runtime autoresearch-v2 round."""

    preview = preview_autoresearch_v2(args.campaign)
    if args.dry_run:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0
    if not args.cutoff:
        raise ValueError("autoresearch-v2 planning requires --cutoff")
    round_dir = freeze_autoresearch_v2(
        args.campaign,
        results_root=args.results,
        evidence_root=args.evidence,
        cutoff=args.cutoff,
    )
    print(round_dir)
    return 0


def command_autoresearch_v2_run(args: argparse.Namespace) -> int:
    """Execute the only authorized current-runtime v2 round once."""

    summary = run_autoresearch_v2(
        args.round_dir,
        workspace=WORKSPACE,
        evidence_root=args.evidence,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete" else 1


def command_autoresearch_v2_summarize(args: argparse.Namespace) -> int:
    """Read-only verify a terminal current-runtime v2 round."""

    summary = summarize_autoresearch_v2(args.round_dir, evidence_root=args.evidence)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete" else 1


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


def command_sm121_storage_canary(args: argparse.Namespace) -> int:
    """Run the isolated pre-admission SM121 native-storage canary."""

    model, suite = _select(args)
    run_dir = create_sm121_storage_canary_plan(
        model=model,
        suite=suite,
        results_root=args.results,
        models_path=args.models,
        suite_path=args.suite,
    )
    print(f"Plan: {run_dir}")
    summary = execute_sm121_storage_canary(run_dir, workspace=WORKSPACE)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 1


def command_sm121_cache_observability_canary(args: argparse.Namespace) -> int:
    """Run the isolated cache-off B0 observation canary."""

    model, suite = _select(args)
    run_dir = create_sm121_cache_observability_plan(
        model=model,
        suite=suite,
        results_root=args.results,
        models_path=args.models,
        suite_path=args.suite,
    )
    print(f"Plan: {run_dir}")
    summary = execute_sm121_cache_observability_canary(run_dir, workspace=WORKSPACE)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 1


def command_sm121_cache_policy_semantic_canary(args: argparse.Namespace) -> int:
    """Run the dedicated cache-off B then cache-on A semantic canary."""

    models = load_models(args.models)
    try:
        cache_off_model = models[SM121_CACHE_SEMANTIC_CACHE_OFF_PROFILE_ID]
        cache_on_model = models[SM121_CACHE_SEMANTIC_CACHE_ON_PROFILE_ID]
    except KeyError as error:
        raise ManifestError(
            "SM121 semantic canary requires both exact paired cache-policy profiles"
        ) from error
    suite = load_suite(args.suite)
    cache_off_run, cache_on_run = create_sm121_cache_semantic_pair_plans(
        cache_off_model=cache_off_model,
        cache_on_model=cache_on_model,
        suite=suite,
        results_root=args.results,
        models_path=args.models,
        suite_path=args.suite,
    )
    print(f"Cache-off B plan: {cache_off_run}")
    print(f"Cache-on A plan: {cache_on_run}")
    summary = execute_sm121_cache_semantic_canary(
        cache_off_run,
        cache_on_run,
        workspace=WORKSPACE,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 1


def command_sm121_cache_policy_performance(args: argparse.Namespace) -> int:
    """Freeze and execute the only authorized SM121 cache A/B/B/A timing lane."""

    models = load_models(args.models)
    try:
        cache_on_model = models[SM121_CACHE_PERFORMANCE_CACHE_ON_PROFILE_ID]
        cache_off_model = models[SM121_CACHE_PERFORMANCE_CACHE_OFF_PROFILE_ID]
    except KeyError as error:
        raise ManifestError(
            "SM121 cache-performance campaign requires both exact A/B profiles"
        ) from error
    suite = load_suite(args.suite)
    campaign_dir = create_sm121_cache_performance_campaign(
        cache_on_model=cache_on_model,
        cache_off_model=cache_off_model,
        suite=suite,
        results_root=args.results / "cache-policy-campaigns",
        models_path=args.models,
        suite_path=args.suite,
        evidence_root=args.evidence,
    )
    print(f"Campaign: {campaign_dir}")
    summary = execute_sm121_cache_performance_campaign(
        campaign_dir,
        workspace=WORKSPACE,
        evidence_root=args.evidence,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete" else 1


def command_sm121_chunked_prefill_performance(args: argparse.Namespace) -> int:
    """Freeze and execute the only authorized current-SM121 1K/2K study."""

    models = load_models(args.models)
    try:
        control_model = models[
            SM121_CHUNKED_PREFILL_PERFORMANCE_CONTROL_PROFILE_ID
        ]
        candidate_model = models[
            SM121_CHUNKED_PREFILL_PERFORMANCE_CANDIDATE_PROFILE_ID
        ]
    except KeyError as error:
        raise ManifestError(
            "SM121 chunked-prefill campaign requires both exact 1K/2K profiles"
        ) from error
    suite = load_suite(args.suite)
    campaign_dir = create_sm121_chunked_prefill_performance_campaign(
        control_model=control_model,
        candidate_model=candidate_model,
        suite=suite,
        results_root=args.results / "chunked-prefill-campaigns",
        models_path=args.models,
        suite_path=args.suite,
    )
    print(f"Campaign: {campaign_dir}")
    summary = execute_sm121_chunked_prefill_performance_campaign(
        campaign_dir, workspace=WORKSPACE
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
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
        if model_execution_blocker(model) is not None:
            continue
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


def command_audit_sm121_storage_canary(args: argparse.Namespace) -> int:
    report = audit_sm121_storage_canary_run(args.run_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def command_audit_sm121_cache_observability(args: argparse.Namespace) -> int:
    report = audit_sm121_cache_observability_run(args.run_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def command_audit_sm121_cache_policy_semantic(args: argparse.Namespace) -> int:
    report = audit_sm121_cache_semantic_pair(
        args.cache_off_run_dir,
        args.cache_on_run_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def command_audit_sm121_cache_policy_performance(args: argparse.Namespace) -> int:
    report = audit_sm121_cache_performance_campaign(
        args.campaign_dir, evidence_root=args.evidence
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def command_audit_sm121_chunked_prefill_performance(
    args: argparse.Namespace,
) -> int:
    report = audit_sm121_chunked_prefill_performance_campaign(args.campaign_dir)
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

    autoresearch_checkpoint = subparsers.add_parser(
        "autoresearch-checkpoint",
        help="acknowledge verified evidence and Git state for a completed pair",
    )
    autoresearch_checkpoint.add_argument("campaign_dir", type=Path)
    autoresearch_checkpoint.set_defaults(function=command_autoresearch_checkpoint)

    autoresearch_summarize = subparsers.add_parser(
        "autoresearch-summarize",
        help="replay and summarize a frozen autoresearch campaign",
    )
    autoresearch_summarize.add_argument("campaign_dir", type=Path)
    autoresearch_summarize.set_defaults(function=command_autoresearch_summarize)

    autoresearch_v2_plan = subparsers.add_parser(
        "autoresearch-v2-plan",
        help="preview or freeze the registered current-runtime autoresearch round",
    )
    autoresearch_v2_plan.add_argument(
        "--campaign",
        type=Path,
        default=DEFAULT_AUTORESEARCH_V2_CACHE_POLICY_CAMPAIGN,
    )
    autoresearch_v2_plan.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    autoresearch_v2_plan.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    autoresearch_v2_plan.add_argument(
        "--cutoff",
        help="required offset-aware ISO-8601 admission cutoff for a frozen round",
    )
    autoresearch_v2_plan.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the scalar proposal without writing plans",
    )
    autoresearch_v2_plan.set_defaults(function=command_autoresearch_v2_plan)

    autoresearch_v2_run = subparsers.add_parser(
        "autoresearch-v2-run",
        help="execute one frozen non-resumable current-runtime autoresearch round",
    )
    autoresearch_v2_run.add_argument("round_dir", type=Path)
    autoresearch_v2_run.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    autoresearch_v2_run.set_defaults(function=command_autoresearch_v2_run)

    autoresearch_v2_summarize = subparsers.add_parser(
        "autoresearch-v2-summarize",
        help="read-only verify a terminal current-runtime autoresearch round",
    )
    autoresearch_v2_summarize.add_argument("round_dir", type=Path)
    autoresearch_v2_summarize.add_argument(
        "--evidence", type=Path, default=DEFAULT_EVIDENCE
    )
    autoresearch_v2_summarize.set_defaults(function=command_autoresearch_v2_summarize)

    benchmark = subparsers.add_parser("benchmark", help="create and immediately execute a plan")
    add_selection(benchmark)
    benchmark.add_argument("--allow-download", action="store_true")
    benchmark.add_argument("--keep-server", action="store_true")
    benchmark.add_argument("--fail-fast", action="store_true")
    benchmark.set_defaults(function=command_benchmark)

    storage_canary = subparsers.add_parser(
        "sm121-storage-canary",
        help="run the pre-admission fresh-process SM121 native-storage canary",
    )
    add_selection(storage_canary)
    storage_canary.set_defaults(suite=DEFAULT_SM121_STORAGE_CANARY_SUITE)
    storage_canary.set_defaults(function=command_sm121_storage_canary)

    cache_observability = subparsers.add_parser(
        "sm121-cache-observability-canary",
        help="run the cache-off SM121 B0 fresh-server observation canary",
    )
    add_selection(cache_observability)
    cache_observability.set_defaults(suite=DEFAULT_SM121_CACHE_OBSERVABILITY_SUITE)
    cache_observability.set_defaults(function=command_sm121_cache_observability_canary)

    cache_semantic = subparsers.add_parser(
        "sm121-cache-policy-semantic-canary",
        help="run the cache-off B then cache-on A SM121 semantic canary",
    )
    cache_semantic.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    cache_semantic.add_argument(
        "--suite", type=Path, default=DEFAULT_SM121_CACHE_SEMANTIC_SUITE
    )
    cache_semantic.add_argument(
        "--results", type=Path, default=WORKSPACE / "results"
    )
    cache_semantic.set_defaults(function=command_sm121_cache_policy_semantic_canary)

    cache_performance = subparsers.add_parser(
        "sm121-cache-policy-performance",
        help="run the frozen fresh-lifetime SM121 cache A/B/B/A wall-time campaign",
    )
    cache_performance.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    cache_performance.add_argument(
        "--suite", type=Path, default=DEFAULT_SM121_CACHE_PERFORMANCE_SUITE
    )
    cache_performance.add_argument(
        "--results", type=Path, default=WORKSPACE / "results"
    )
    cache_performance.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    cache_performance.set_defaults(function=command_sm121_cache_policy_performance)

    chunked_prefill_performance = subparsers.add_parser(
        "sm121-chunked-prefill-performance",
        help="run the frozen fresh-lifetime SM121 1K/2K prefill A/B/B/A campaign",
    )
    chunked_prefill_performance.add_argument(
        "--models", type=Path, default=DEFAULT_MODELS
    )
    chunked_prefill_performance.add_argument(
        "--suite", type=Path, default=DEFAULT_SM121_CHUNKED_PREFILL_PERFORMANCE_SUITE
    )
    chunked_prefill_performance.add_argument(
        "--results", type=Path, default=WORKSPACE / "results"
    )
    chunked_prefill_performance.set_defaults(
        function=command_sm121_chunked_prefill_performance
    )

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

    storage_audit = subparsers.add_parser(
        "audit-sm121-storage-canary",
        help="read-only topology verification of one SM121 storage canary run",
    )
    storage_audit.add_argument("run_dir", type=Path)
    storage_audit.set_defaults(function=command_audit_sm121_storage_canary)

    cache_observability_audit = subparsers.add_parser(
        "audit-sm121-cache-observability",
        help="read-only topology verification of one SM121 B0 observation run",
    )
    cache_observability_audit.add_argument("run_dir", type=Path)
    cache_observability_audit.set_defaults(
        function=command_audit_sm121_cache_observability
    )

    cache_semantic_audit = subparsers.add_parser(
        "audit-sm121-cache-policy-semantic",
        help="read-only verification of a completed SM121 B-then-A semantic pair",
    )
    cache_semantic_audit.add_argument("cache_off_run_dir", type=Path)
    cache_semantic_audit.add_argument("cache_on_run_dir", type=Path)
    cache_semantic_audit.set_defaults(
        function=command_audit_sm121_cache_policy_semantic
    )

    cache_performance_audit = subparsers.add_parser(
        "audit-sm121-cache-policy-performance",
        help="read-only verification of one SM121 cache A/B/B/A campaign",
    )
    cache_performance_audit.add_argument("campaign_dir", type=Path)
    cache_performance_audit.add_argument(
        "--evidence", type=Path, default=DEFAULT_EVIDENCE
    )
    cache_performance_audit.set_defaults(
        function=command_audit_sm121_cache_policy_performance
    )

    chunked_prefill_performance_audit = subparsers.add_parser(
        "audit-sm121-chunked-prefill-performance",
        help="read-only verification of one SM121 1K/2K prefill campaign",
    )
    chunked_prefill_performance_audit.add_argument("campaign_dir", type=Path)
    chunked_prefill_performance_audit.set_defaults(
        function=command_audit_sm121_chunked_prefill_performance
    )

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
