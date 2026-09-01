from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
import shutil
import statistics
from collections.abc import Callable
from pathlib import Path
import subprocess
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import bench.memory_ops as memory_ops
from bench.evidence import (
    EvidenceError,
    HARBOR_CAMPAIGN_ID,
    HARBOR_EXPECTED_DERIVATION_DIGEST,
    HARBOR_EXPECTED_GIT_REVISION,
    HARBOR_SCHEMA_VERSION,
    _MEMORY_PANEL_MODEL_ARTIFACTS,
    _MEMORY_PANEL_MODELS,
    _NINFER_TOP_FIELDS,
    SCHEMA_VERSION,
    _assert_source_tree,
    _export_run,
    _load_json,
    _project_model,
    _project_case,
    _project_ninfer_report,
    _project_request_result,
    _project_requests,
    _project_suite,
    _project_summary,
    _validate_agentic_aggregates,
    _validate_output_value,
    _write_bundle,
    export_evidence,
    verify_evidence,
    verify_staged_evidence,
)
from bench.densespark import (
    DENSESPARK_LOCAL_IMAGE_ID,
    DENSESPARK_MODEL_REVISION,
    DENSESPARK_PQ_SHA256,
    DENSESPARK_PROFILE_ID,
    DENSESPARK_WARMUP_SYNC_LOCAL_IMAGE_ID,
    DENSESPARK_WARMUP_SYNC_MODE,
    DENSESPARK_WARMUP_SYNC_PROFILE_ID,
    densespark_expected_launch_policy,
    densespark_expected_resolved_provenance,
)
from bench.memory_ops import (
    MEMORY_OPERATION_CONTEXT_TOKENS,
    MEMORY_OPERATION_LLAMACPP_DIGEST,
    MEMORY_OPERATION_LLAMACPP_REVISION,
    MEMORY_OPERATION_OUTPUT_TOKENS,
    MEMORY_OPERATION_PROTOCOL_DIGEST,
    MEMORY_OPERATION_SCENARIO_IDS,
    MEMORY_OPERATION_SERVER_TIMING_TOLERANCE_S,
    memory_operation_llamacpp_args,
)
from bench.manifest import (
    MATCHED_PROMPT_GRAPH_STUDIES,
    MATCHED_REQUEST_UNIQUE_PROTOCOL,
    QWEN38_27B_DSPARK_CUDA_GRAPH_DISABLED_PROFILE_ID,
    QWEN38_27B_DSPARK_CUDA_GRAPH_FULL_PROFILE_ID,
    load_models,
    load_suite,
    model_spec_to_dict,
)
from bench.sglang_sm121_cuda_graph import SM121_CUDA_GRAPH_BREAKABLE_PROFILE_ID
from bench.report import summarize_run
from bench.harbor_campaign_lifecycle import (
    CleanupStatus,
    ModelAdmission,
    RuntimeAdmission,
    _EXPECTED_MODEL,
    build_lifecycle_envelope,
)
from bench.harbor_terminal import (
    HarborAttempt,
    HarborRawResult,
    HarborRunStatus,
    NpmArtifactAdmission,
    NpmArtifactRecord,
    iter_trials,
    load_campaign,
    summarize_campaign_results,
)
from sparkbench import build_parser


RAW_COMPLETION = "RAW_COMPLETION_SENTINEL"
RAW_REASONING = "RAW_REASONING_SENTINEL"
RAW_REQUEST_ID = "RAW_REQUEST_ID_SENTINEL"
RAW_HOST_PATH = "/home/private-user/benchmark-cache/model.gguf"
RAW_SECRET = "hf" + "_" + "0123456789abcdefghijklmnop"

REPOSITORY = Path(__file__).resolve().parents[1]
HARBOR_CAMPAIGN_PATH = (
    REPOSITORY / "manifests" / "campaigns" / "harbor_terminal_coder_next.toml"
)


def _synthetic_content_hash(value: object, *, length: int = 64) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()[:length]


def _autoresearch_suite(model: dict[str, object]) -> dict[str, object]:
    suite = asdict(
        load_suite(
            REPOSITORY
            / "manifests"
            / "suites"
            / "qwen38_flash_next_sglang_agent64k_autoresearch.toml"
        )
    )
    suite.pop("protocol_digest", None)
    suite["cases"] = [
        {
            **case,
            "case_id": (
                f"{case['id']}--"
                f"{_synthetic_content_hash({'model': model, 'case': case}, length=12)}"
            ),
        }
        for case in suite["cases"]
    ]
    return suite


def _harbor_npm_admission(
    campaign: object, *, size_offset: int = 0
) -> NpmArtifactAdmission:
    declared: list[tuple[str, str, str, str]] = []
    for agent in campaign.agents:
        declared.append(
            (agent.npm_package, agent.version, agent.npm_shasum, agent.npm_integrity)
        )
        if agent.platform_package is not None:
            declared.append(
                (
                    agent.platform_package,
                    agent.version,
                    agent.platform_shasum,
                    agent.platform_integrity,
                )
            )
    records = tuple(
        NpmArtifactRecord(
            package=package,
            version=version,
            size_bytes=index + size_offset,
            shasum=shasum,
            integrity=integrity,
        )
        for index, (package, version, shasum, integrity) in enumerate(
            sorted(declared), start=1
        )
    )
    projection = [
        {
            "package": record.package,
            "version": record.version,
            "size_bytes": record.size_bytes,
            "shasum": record.shasum,
            "integrity": record.integrity,
        }
        for record in records
    ]
    digest = "sha256:" + hashlib.sha256(
        json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return NpmArtifactAdmission(digest=digest, artifacts=records)


def _harbor_status(trial: object) -> HarborRunStatus:
    return HarborRunStatus(
        trial=trial,
        exit_code=0,
        timed_out=False,
        wall_s=1.0,
        main_image_id="sha256:" + "1" * 64,
        main_image_fingerprint="sha256:" + "2" * 64,
        main_image_arm64=True,
        relay_image_arm64=True,
        built_image_cleanup_succeeded=True,
        setup_relay_rejected=True,
        agent_relay_passed=True,
        wrong_auth_rejected=True,
        other_loopback_rejected=True,
        gost_rejected=True,
        dns_rejected=True,
        gateway_rejected=True,
        public_rejected=True,
        capabilities_dropped=True,
        cleanup_succeeded=True,
        containers_found=1,
        containers_removed=1,
        networks_found=1,
        networks_removed=1,
        volumes_found=1,
        volumes_removed=1,
    )


def _harbor_job_result(campaign: object, trial: object, *, reward: int) -> HarborRawResult:
    agent = campaign.agent(trial.agent_id)

    def timing(start: str, finish: str) -> dict[str, str]:
        return {"started_at": start, "finished_at": finish}

    return HarborRawResult(
        job={
            "id": "synthetic-private-job-id",
            "n_total_trials": 1,
            "stats": {"n_retries": 0, "n_completed_trials": 1},
            "synthetic_private_value": "must-not-survive",
        },
        trial={
            "id": "synthetic-private-trial-id",
            "trial_name": "synthetic-private-trial-name",
            "trial_uri": "synthetic-private-uri",
            "task_name": f"terminal-bench/{trial.task_id}",
            "agent_info": {
                "name": trial.agent_id,
                "version": agent.version,
                "model_info": {
                    "provider": "openai",
                    "name": campaign.model.served_name,
                },
            },
            "agent_result": {
                "n_input_tokens": 100 + trial.index,
                "n_cache_tokens": 10,
                "n_output_tokens": 20 + trial.index,
                "metadata": {"synthetic_private_value": "must-not-survive"},
            },
            "verifier_result": {"rewards": {"reward": reward}},
            "started_at": "2026-08-18T01:00:00+00:00",
            "finished_at": "2026-08-18T01:00:20+00:00",
            "environment_setup": timing(
                "2026-08-18T01:00:00+00:00", "2026-08-18T01:00:02+00:00"
            ),
            "agent_setup": timing(
                "2026-08-18T01:00:02+00:00", "2026-08-18T01:00:05+00:00"
            ),
            "agent_execution": timing(
                "2026-08-18T01:00:05+00:00", "2026-08-18T01:00:17+00:00"
            ),
            "verifier": timing(
                "2026-08-18T01:00:17+00:00", "2026-08-18T01:00:20+00:00"
            ),
            "exception_info": None,
            "config": {"synthetic_private_value": "must-not-survive"},
        },
    )


def _harbor_envelope(
    *,
    started_at: str,
    finished_at: str,
    size_offset: int = 0,
) -> dict[str, object]:
    campaign = load_campaign(HARBOR_CAMPAIGN_PATH)
    attempts = tuple(
        HarborAttempt(
            trial=trial,
            status=_harbor_status(trial),
            job_result=_harbor_job_result(
                campaign, trial, reward=1 if trial.index % 3 else 0
            ),
        )
        for trial in iter_trials(campaign)
    )
    summary = summarize_campaign_results(
        campaign,
        attempts,
        network_policy_patch_digest=HARBOR_EXPECTED_DERIVATION_DIGEST,
        npm_artifact_admission=_harbor_npm_admission(
            campaign, size_offset=size_offset
        ),
    )
    admission = RuntimeAdmission(
        artifact_validation=True,
        harbor_runtime_verified=True,
        node_tree_verified=True,
        agent_trees_verified=2,
        npm_artifacts_verified=3,
        runtime_assets_verified=2,
        agent_source_files_verified=1,
        python_bytecode_cache_empty=True,
        host_arm64=True,
        docker_server_arm64=True,
        model=ModelAdmission(
            chat_passed=True,
            json_passed=True,
            tool_call_passed=True,
            prompt_tokens=30,
            completion_tokens=15,
            wall_s=2.0,
        ),
        unix_bridge_verified=True,
    )
    cleanup = CleanupStatus(
        harbor_resources_removed=True,
        bridge_stopped=True,
        bridge_socket_removed=True,
        server_stopped=True,
        telemetry_stopped=True,
        key_removed=True,
        raw_jobs_private_retained=True,
        raw_jobs_key_free=True,
        derived_dataset_removed=True,
        runtime_overlays_removed=True,
        staged_assets_removed=True,
        socket_directory_removed=True,
        npm_scratch_removed=True,
        agent_source_removed=True,
        python_pycache_removed=True,
    )
    return build_lifecycle_envelope(
        campaign=campaign,
        campaign_summary=summary,
        model_provenance=dict(_EXPECTED_MODEL),
        git_revision=HARBOR_EXPECTED_GIT_REVISION,
        git_clean=True,
        admission=admission,
        cleanup=cleanup,
        status="completed",
        stop_reason="completed",
        started_at=started_at,
        finished_at=finished_at,
        elapsed_s=300.0,
        trials_started=12,
        trials_completed=12,
        cutoff_reached=False,
    )


def _write_private_json(path: Path, value: object, *, canonical: bool = True) -> None:
    if canonical:
        text = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ) + "\n"
    else:
        text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="ascii")
    path.chmod(0o600)


def _agentic_suite() -> dict[str, object]:
    scenarios = (
        ("agentic-select-and-call", 1),
        ("agentic-no-tool", 2),
        ("agentic-two-hop", 0),
        ("agentic-tool-error-recovery", 3),
    )
    return {
        "id": "agentic-tools",
        "description": (
            "Deterministic multi-turn tool selection, abstention, dependency, "
            "and recovery checks with scalar-only results."
        ),
        "schema_version": 1,
        "cases": [
            {
                "case_id": f"{scenario}--{suffix:012x}",
                "concurrency": 1,
                "id": scenario,
                "kind": "agentic",
                "max_output_tokens": 4096,
                "max_turns": 6,
                "prompt_repetitions": 0,
                "repetitions": 3,
                "requires": ["chat", "tools"],
                "temperature": 0.0,
                "warmups": 0,
            }
            for scenario, suffix in scenarios
        ],
    }


def _memory_model() -> dict[str, object]:
    return {
        "id": "laguna-xs21-33b-a3b-q4-k-m-llamacpp",
        "backend": "llamacpp",
        "architecture": "laguna",
        "quantization": "q4_k_m",
        "source": "poolside/Laguna-XS-2.1-GGUF",
        "revision": "1a37c0a5fb8c7a18e6106decb6be6327d1b63fa6",
        "support_status": "spark_other_backend",
        "tasks": ["chat", "json", "tools"],
        "lifecycle": "subprocess",
        "model_file": "Laguna-XS-2.1-Q4_K_M.gguf",
        "model_digest": "sha256:1ac7079101fca5a6df8c5a7523a3c30ea7d1c0e4b1258090e7d6d4039287f6cb",
        "model_size_bytes": 20274300032,
        "runtime_binary": "/synthetic/llama.cpp-b10453/bin/llama-server",
        "runtime_digest": MEMORY_OPERATION_LLAMACPP_DIGEST,
        "runtime_revision": MEMORY_OPERATION_LLAMACPP_REVISION,
        "runtime_parallel": 1,
        "max_context": MEMORY_OPERATION_CONTEXT_TOKENS,
        "native_context": MEMORY_OPERATION_CONTEXT_TOKENS,
        "startup_timeout_s": 600,
        "estimated_ram_gib": 48.0,
        "request_body_json": '{"chat_template_kwargs":{"enable_thinking":false}}',
        "args": list(memory_operation_llamacpp_args(enable_thinking=False)),
    }


def _memory_suite(model: dict[str, object]) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for scenario_id in MEMORY_OPERATION_SCENARIO_IDS:
        case: dict[str, object] = {
            "concurrency": 1,
            "id": scenario_id,
            "kind": "memory",
            "max_output_tokens": MEMORY_OPERATION_OUTPUT_TOKENS,
            "max_turns": 1,
            "prompt_repetitions": 0,
            "repetitions": 3,
            "requires": ["chat", "json"],
            "temperature": 0.0,
            "warmups": 0,
        }
        digest = hashlib.sha256(
            json.dumps(
                {
                    "model": model,
                    "case": case,
                    "protocol_digest": MEMORY_OPERATION_PROTOCOL_DIGEST,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:12]
        cases.append({**case, "case_id": f"{scenario_id}--{digest}"})
    return {
        "id": "memory-operations",
        "description": (
            "Graphiti-style edge resolution followed by explicitly synthetic "
            "MemFS/transaction extension cases; exact JSON grading and "
            "scalar-only results."
        ),
        "protocol_digest": MEMORY_OPERATION_PROTOCOL_DIGEST,
        "schema_version": 1,
        "cases": cases,
    }


def _memory_result(scenario_id: str, variant: int) -> dict[str, object]:
    graphiti = scenario_id.startswith("graphiti-")
    expected_resolver = {
        "graphiti-reuse-fact": "REUSE_FACT",
        "graphiti-invalidate-fact": "CREATE_AND_INVALIDATE",
        "graphiti-create-fact": "CREATE_FACT",
    }.get(scenario_id)
    mutation_expected = scenario_id in {
        "memory-add",
        "memory-delete",
        "memory-supersede",
        "memory-temporal-invalidate",
        "memory-tier-placement",
    }
    completion_tokens = 20 + variant
    ttft_s = 0.4
    decode_s = 1.6 + variant / 10
    elapsed_s = ttft_s + decode_s
    return {
        "schema_version": 1,
        "scenario_id": scenario_id,
        "variant": variant,
        "passed": True,
        "failure_code": None,
        "json_object_emitted": True,
        "schema_valid": True,
        "action_correct": True,
        "target_correct": None if graphiti else True,
        "path_correct": None if graphiti else True,
        "tier_correct": None if graphiti else True,
        "value_correct": None if graphiti else True,
        "valid_from_correct": None if graphiti else True,
        "valid_to_correct": None if graphiti else True,
        "evidence_correct": None if graphiti else True,
        "reason_correct": None if graphiti else True,
        "duplicate_facts_correct": True if graphiti else None,
        "contradicted_facts_correct": True if graphiti else None,
        "protected_value_emitted": False,
        "mutation_expected": mutation_expected,
        "mutation_selected": mutation_expected,
        "secret_refusal_required": scenario_id == "memory-secret-refusal",
        "secret_refusal_succeeded": scenario_id == "memory-secret-refusal",
        "injection_refusal_required": scenario_id == "memory-injection-refusal",
        "injection_refusal_succeeded": scenario_id == "memory-injection-refusal",
        "graphiti_resolver_case": graphiti,
        "synthetic_extension_case": not graphiti,
        "resolver_decision_correct": True if graphiti else None,
        "expected_resolver_action": expected_resolver,
        "selected_resolver_action": expected_resolver,
        "unexpected_field_count": 0,
        "unexpected_tool_call_count": 0,
        "max_output_tokens": MEMORY_OPERATION_OUTPUT_TOKENS,
        "prompt_cache_disabled": True,
        "prompt_tokens": 100 + variant,
        "cached_prompt_tokens": 0,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": None,
        "emission_events": completion_tokens,
        "ttft_s": ttft_s,
        "elapsed_s": elapsed_s,
        "decode_s": decode_s,
        "decode_tps": (completion_tokens - 1) / decode_s,
        "output_tps": completion_tokens / elapsed_s,
        "server_prompt_tokens": 100 + variant,
        "server_cached_prompt_tokens": 0,
        "server_decode_tokens": completion_tokens,
        "server_prompt_s": 0.2,
        "server_decode_s": 1.0,
        "finish_reason": "stop",
        "decode_metric_source": "client_estimate",
    }


class EvidenceFixture:
    def __init__(self, root: Path) -> None:
        self.results = root / "results"
        self.output = root / "evidence"
        self.results.mkdir()
        (self.results / ".sparkbench.lock").touch()
        (self.results / "matrices").mkdir()
        self.run_id = "20260817T000000Z-synthetic"
        self.run_dir = self.results / self.run_id
        self.run_dir.mkdir()
        self._write_required_campaigns()
        self._write_standalone_results(aggregate_tps=12.5)
        self._write_run()

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
            encoding="utf-8",
        )

    def _write_required_campaigns(self) -> None:
        for name in (
            "moe-bandwidth-20260817T1539Z",
            "ninfer-experimental-sm121a-20260817T181134Z",
            "ninfer-qwen38-nvfp4-sm121a-20260817T200147Z",
        ):
            (self.results / name).mkdir()
        admission = self.results / "ninfer-gb10-20260817"
        admission.mkdir()
        (admission / "stock-cmake-sm121.log").write_text(
            "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got '121a'\n",
            encoding="utf-8",
        )
        (admission / "stock-cmake-default.log").write_text(
            "NInfer requires CUDA 13.1 or newer; found CUDA compiler 13.0.88\n",
            encoding="utf-8",
        )

    def _write_standalone_results(self, *, aggregate_tps: float) -> None:
        probe_ids = (
            ("math-en-eval-style", "math", "en"),
            ("code-en", "code", "en"),
            ("code-de", "code", "de"),
            ("technical-explanation-fr", "technical_explanation", "fr"),
            ("reasoning-fr", "reasoning", "fr"),
            ("free-prose-en", "free_prose", "en"),
            ("free-prose-fr", "free_prose", "fr"),
            ("free-prose-de", "free_prose", "de"),
        )

        def summary() -> dict[str, float | int]:
            return {
                "aggregate_decode_tps": aggregate_tps,
                "aggregate_output_tps": aggregate_tps,
                "completion_tokens": 30,
                "maximum_decode_tps": aggregate_tps,
                "median_decode_tps": aggregate_tps,
                "median_e2e_s": 1.0,
                "median_ttft_s": 0.1,
                "minimum_decode_tps": aggregate_tps,
                "prompt_tokens": 30,
                "requests": 3,
            }

        probes = []
        for probe_id, category, language in probe_ids:
            samples = []
            for repetition in range(1, 4):
                samples.append(
                    {
                        "completion_tokens": 10,
                        "decode_s": 0.9,
                        "decode_tps": aggregate_tps,
                        "e2e_s": 1.0,
                        "emission_events": 10,
                        "measured_order": repetition,
                        "output_tps": aggregate_tps,
                        "prompt_tokens": 10,
                        "repetition": repetition,
                        "sample_id": f"{RAW_REQUEST_ID}-{repetition}",
                        "ttft_s": 0.1,
                    }
                )
            probes.append(
                {
                    "category": category,
                    "id": probe_id,
                    "language": language,
                    "samples": samples,
                    "summary": summary(),
                }
            )
        self.write_json(
            self.results / "content-battery-dspark-sglang-20260817.json",
            {
                "schema_version": 1,
                "battery": {
                    "id": "dgx-spark-qwen38-content",
                    "prompt_set_version": 1,
                    "protocol_version": 1,
                },
                "endpoint": "loopback",
                "model": "qwen3.8-27b",
                "probes": probes,
                "protocol": {
                    "aggregate_decode_tps": "sum_completion_tokens_minus_first_over_sum_post_ttft_seconds",
                    "aggregate_output_tps": "sum_completion_tokens_over_sum_e2e_seconds",
                    "fresh_prompt_tags": RAW_REQUEST_ID,
                    "max_output_tokens": 680,
                    "minimum_output_tokens": 50,
                    "repetitions_per_prompt": 3,
                    "temperature": 0.0,
                    "transport": "openai_chat_completions_sse",
                    "warmups": 1,
                },
                "summary": summary(),
                "warmup": {
                    "completion_tokens": 10,
                    "decode_s": 0.9,
                    "decode_tps": aggregate_tps,
                    "e2e_s": 1.0,
                    "emission_events": 10,
                    "id": RAW_REQUEST_ID,
                    "output_tps": aggregate_tps,
                    "prompt_tokens": 10,
                    "ttft_s": 0.1,
                },
            },
        )
        self.write_json(
            self.results / "upstream-bench-matrix-dspark-sglang-20260817.json",
            {
                "battery_version": 1,
                "endpoint": "loopback",
                "method": "synthetic",
                "model_id": "qwen3.8-27b",
                "results": [
                    {"probe": probe, "tok_s": aggregate_tps}
                    for probe in (
                        "math (EN, eval-style)",
                        "code (EN)",
                        "code (DE)",
                        "technical explain (FR)",
                        "reasoning (FR)",
                        "free prose (EN)",
                        "free prose (FR)",
                        "free prose (DE)",
                    )
                ],
            },
        )

    @staticmethod
    def _request_result(*, completion_tokens: int) -> dict[str, object]:
        return {
            "prompt_tokens": 10,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": None,
            "elapsed_s": 1.25,
            "ttft_s": 0.2,
            "decode_tps": 20.0,
            "emission_events": completion_tokens,
            "finish_reason": "stop",
            "content": f"{RAW_COMPLETION} {RAW_SECRET}",
            "reasoning": f"{RAW_REASONING} {RAW_HOST_PATH}",
            "request_id": RAW_REQUEST_ID,
            "tool_calls": [],
            "output_sha256": "a" * 64,
        }

    def _write_run(self) -> None:
        self.write_json(
            self.run_dir / "plan.json",
            {
                "schema_version": 1,
                "model": {
                    "id": "synthetic-model",
                    "backend": "synthetic",
                    "architecture": "synthetic",
                    "quantization": "fp8",
                    "source": "example/synthetic-model",
                    "revision": "a" * 40,
                    "tasks": ["chat"],
                    "lifecycle": "managed",
                },
                "suite": {
                    "id": "synthetic-suite",
                    "schema_version": 1,
                    "cases": [
                        {
                            "id": "chat-case",
                            "kind": "chat",
                            "repetitions": 3,
                            "warmups": 0,
                            "max_output_tokens": 32,
                            "temperature": 0.0,
                        }
                    ],
                },
            },
        )
        events = [
            {
                "timestamp": "2026-08-17T00:00:00Z",
                "event": "run_start",
            },
            {
                "timestamp": "2026-08-17T00:00:01Z",
                "event": "first_request_complete",
                "result": self._request_result(completion_tokens=1),
            },
            {
                "timestamp": "2026-08-17T00:00:02Z",
                "event": "case_start",
                "case_id": "chat-case",
                "attempt_id": "private-attempt-a",
            },
            {
                "timestamp": "2026-08-17T00:00:03Z",
                "event": "request_complete",
                "case_id": "chat-case",
                "kind": "chat",
                "attempt_id": "private-attempt-a",
                "request_tag": RAW_REQUEST_ID,
                "result": self._request_result(completion_tokens=10),
            },
            {
                "timestamp": "2026-08-17T00:00:04Z",
                "event": "request_complete",
                "case_id": "chat-case",
                "kind": "chat",
                "attempt_id": "private-attempt-a",
                "result": self._request_result(completion_tokens=11),
            },
            {
                "timestamp": "2026-08-17T00:00:05Z",
                "event": "case_start",
                "case_id": "chat-case",
                "attempt_id": "private-attempt-b",
            },
            {
                "timestamp": "2026-08-17T00:00:06Z",
                "event": "request_complete",
                "case_id": "chat-case",
                "kind": "chat",
                "attempt_id": "private-attempt-b",
                "result": self._request_result(completion_tokens=12),
            },
            {
                "timestamp": "2026-08-17T00:00:07Z",
                "event": "run_complete",
                "status": "completed",
                "diagnostic": RAW_HOST_PATH,
            },
        ]
        self.write_jsonl(self.run_dir / "events.jsonl", events)
        self.write_json(
            self.run_dir / "summary.json",
            {
                "schema_version": "2",
                "status": "complete",
                "run_completion_status": "completed",
                "completed_cases": 1,
                "failed_cases": [],
                "validation_failed_cases": [],
                "unimplemented_cases": [],
                "unsupported_cases": [],
                "context_limited_cases": [],
                "first_request_after_start": self._request_result(
                    completion_tokens=1
                ),
                "measurement_annotations": [
                    {
                        "reason": RAW_REASONING,
                        "request_id": RAW_REQUEST_ID,
                        "path": RAW_HOST_PATH,
                    }
                ],
                "cases": [
                    {
                        "case_id": "chat-case",
                        "attempt_id": "private-attempt-b",
                        "kind": "chat",
                        "requests": 1,
                        "prompt_tokens": 10,
                        "completion_tokens": 12,
                        "elapsed_s": 1.25,
                        "aggregate_output_tps": 20.0,
                        "measurement_valid": True,
                        "validation_passed": True,
                        "measurement_annotations": [
                            {"reason": RAW_COMPLETION, "secret": RAW_SECRET}
                        ],
                        "output_sha256": "b" * 64,
                    }
                ],
            },
        )
        self.write_jsonl(
            self.run_dir / "telemetry.jsonl",
            [
                {
                    "timestamp": "2026-08-17T00:00:02.000Z",
                    "gpu_timestamp": "raw-device-time-a",
                    "phase": "case:chat-case:measure",
                    "gpu_util_pct": 80.0,
                    "power_w": 100.0,
                    "memfree_kib": 10,
                    "gpu_error": "",
                },
                {
                    "timestamp": "2026-08-17T00:00:02.500Z",
                    "gpu_timestamp": "raw-device-time-b",
                    "phase": "case:chat-case:measure",
                    "gpu_util_pct": 90.0,
                    "power_w": 110.0,
                    "memfree_kib": 9,
                    "gpu_error": f"{RAW_SECRET} {RAW_HOST_PATH}",
                },
                {
                    "timestamp": "2026-08-17T00:00:07.000Z",
                    "gpu_timestamp": "raw-device-time-c",
                    "phase": "idle",
                    "gpu_util_pct": 0.0,
                    "power_w": 40.0,
                    "memfree_kib": 12,
                    "gpu_error": None,
                },
            ],
        )

    def write_matched_graph_run(self, profile_id: str) -> None:
        """Replace the generic run with one complete synthetic graph screen."""

        repository = Path(__file__).resolve().parents[1]
        models = load_models(repository / "manifests" / "models.toml")
        study = next(
            item
            for item in MATCHED_PROMPT_GRAPH_STUDIES
            if profile_id in item.profile_ids
        )
        suite_filename = {
            "qwen38-27b-dspark-c1-cuda-graph": (
                "qwen38_27b_dspark_c1_cuda_graph.toml"
            ),
            "qwen38-flash-next-sm121-triton-storage-c1-cuda-graph": (
                "qwen38_flash_next_sm121_triton_storage_c1_cuda_graph.toml"
            ),
        }[study.suite_id]
        model = json.loads(json.dumps(model_spec_to_dict(models[profile_id])))
        suite_spec = load_suite(
            repository / "manifests" / "suites" / suite_filename
        )
        suite = json.loads(json.dumps(asdict(suite_spec)))
        suite.pop("protocol_digest", None)
        case = suite["cases"][0]
        identity_case = {key: value for key, value in case.items() if key != "case_id"}
        case_digest = _synthetic_content_hash(
            {"model": model, "case": identity_case}, length=12
        )
        case_id = f"{study.case_id}--{case_digest}"
        case["case_id"] = case_id

        shutil.rmtree(self.run_dir)
        self.run_id = (
            f"20260817T000000Z-{profile_id}-{study.suite_id}-00000000"
        )
        self.run_dir = self.results / self.run_id
        self.run_dir.mkdir()
        self.write_json(
            self.run_dir / "plan.json",
            {
                "schema_version": 2,
                "host_at_plan": {
                    "git_commit": "d" * 40,
                    "git_status": "",
                    "memtotal_kib": 128 * 1024 * 1024,
                    "nvidia_smi": "NVIDIA GB10, 580.126.09, 12.1",
                },
                "model": model,
                "suite": suite,
            },
        )

        def result(
            request_id: str,
            *,
            completion_tokens: int,
            prompt_tokens: int,
            ttft_s: float,
            decode_s: float,
        ) -> dict[str, object]:
            elapsed_s = ttft_s + decode_s
            return {
                "cached_prompt_tokens": None,
                "completion_tokens": completion_tokens,
                "content": "synthetic",
                "decode_metric_source": "client_estimate",
                "decode_s": decode_s,
                "decode_tps": max(completion_tokens - 1, 0) / decode_s,
                "elapsed_s": elapsed_s,
                "emission_events": min(completion_tokens, 100),
                "finish_reason": "length",
                "load_s": None,
                "output_tps": completion_tokens / elapsed_s,
                "prompt_tokens": prompt_tokens,
                "reasoning": "",
                "reasoning_tokens": 0,
                "request_id": request_id,
                "response_model": model["served_name"],
                "server_cached_prompt_tokens": None,
                "server_decode_s": None,
                "server_decode_tokens": None,
                "server_prompt_s": None,
                "server_prompt_tokens": None,
                "started_at_ns": 1,
                "tool_calls": [],
                "ttft_s": ttft_s,
            }

        first_result = result(
            "first-request-after-start-1000",
            completion_tokens=8,
            prompt_tokens=10,
            ttft_s=0.2,
            decode_s=0.8,
        )
        measured_results = [
            result(
                f"{study.case_id}-r{repetition}-w0",
                completion_tokens=256,
                prompt_tokens=100,
                ttft_s=0.2,
                decode_s=9.8,
            )
            for repetition in range(5)
        ]
        attempt_id = "private-matched-attempt"
        events: list[dict[str, object]] = []

        def append(event: dict[str, object]) -> None:
            ordinal = len(events)
            events.append(
                {
                    "timestamp": f"2026-08-17T00:00:{ordinal:02d}Z",
                    **event,
                }
            )

        append({"event": "run_start"})
        append({"event": "measurement_started"})
        append({"event": "server_ready", "backend": "sglang"})
        append(
            {
                "event": "first_request_complete",
                "backend": "sglang",
                "result": first_result,
            }
        )
        append(
            {
                "event": "case_start",
                "case_id": case_id,
                "attempt_id": attempt_id,
                "kind": "decode",
                "concurrency": 1,
            }
        )
        for repetition, measured in enumerate(measured_results):
            append(
                {
                    "event": "request_complete",
                    "case_id": case_id,
                    "attempt_id": attempt_id,
                    "kind": "decode",
                    "repetition": repetition,
                    "burst_elapsed_s": float(measured["elapsed_s"]) + 0.01,
                    "result": measured,
                    "validation": {"passed": True, "reason": None},
                }
            )
        case_elapsed_s = sum(
            float(item["elapsed_s"]) + 0.01 for item in measured_results
        ) + 0.1
        append(
            {
                "event": "case_complete",
                "case_id": case_id,
                "attempt_id": attempt_id,
                "kind": "decode",
                "concurrency": 1,
                "elapsed_s": case_elapsed_s,
                "validation_passed": True,
            }
        )
        append({"event": "measurement_complete"})
        append({"event": "server_stopped", "backend": "sglang"})
        append({"event": "run_complete", "status": "completed"})
        self.write_jsonl(self.run_dir / "events.jsonl", events)

        decode_rates = [float(item["decode_tps"]) for item in measured_results]
        elapsed = [float(item["elapsed_s"]) for item in measured_results]
        ttfts = [float(item["ttft_s"]) for item in measured_results]
        case_telemetry = {"sampled_energy_j": 100.0}
        self.write_json(
            self.run_dir / "summary.json",
            {
                "artifact_validation": None,
                "artifact_validation_telemetry": None,
                "cases": [
                    {
                        "aggregate_output_tps": 1280 / case_elapsed_s,
                        "attempt_id": attempt_id,
                        "case_id": case_id,
                        "completion_tokens": 1280,
                        "concurrency": 1,
                        "decode_estimate_one_token_chunks": False,
                        "decode_metric_source": "client_estimate",
                        "elapsed_s": case_elapsed_s,
                        "kind": "decode",
                        "measurement_annotations": [],
                        "measurement_valid": True,
                        "median_decode_tps": statistics.median(decode_rates),
                        "median_e2e_s": statistics.median(elapsed),
                        "median_estimated_decode_tps": statistics.median(
                            decode_rates
                        ),
                        "median_ttft_s": statistics.median(ttfts),
                        "output_tokens_per_sampled_joule": 12.8,
                        "p95_e2e_s": None,
                        "p95_ttft_s": None,
                        "prompt_tokens": 500,
                        "reasoning_tokens": 0,
                        "request_tps": 5 / case_elapsed_s,
                        "requests": 5,
                        "telemetry": case_telemetry,
                        "validation_passed": True,
                    }
                ],
                "completed_cases": 1,
                "context_limited_cases": [],
                "failed_cases": [],
                "first_request_after_start": first_result,
                "first_request_telemetry": {},
                "llamacpp_dflash_evidence": None,
                "llamacpp_mtp_evidence": None,
                "measurement_annotations": [],
                "measurement_invalid_cases": [],
                "run_completion_status": "completed",
                "shutdown_telemetry": {},
                "speculative_decoding": None,
                "startup_measurement_annotations": [],
                "startup_measurement_valid": True,
                "startup_safety_gates": [],
                "startup_telemetry": {},
                "status": "complete",
                "suite": study.suite_id,
                "unimplemented_cases": [],
                "unsupported_cases": [],
                "validation_failed_cases": [],
            },
        )

    def write_autoresearch_campaign(
        self, *, mixed_suite: bool = False
    ) -> tuple[Path, list[Path]]:
        """Nest fourteen bound cells whose private run basenames all collide."""

        source_files = (
            {}
            if mixed_suite
            else {
                path.name: path.read_bytes()
                for path in self.run_dir.iterdir()
                if path.is_file() and path.name != "plan.json"
            }
        )
        source_plan = json.loads(
            (self.run_dir / "plan.json").read_text(encoding="utf-8")
        )
        source_plan.update(
            {
                "schema_version": 2,
                "created_at": "2026-08-17T00:00:00.000+00:00",
                "resolved": {},
            }
        )
        if mixed_suite:
            source_plan["suite"] = _autoresearch_suite(source_plan["model"])
        source_suite = source_plan["suite"]
        suite_without_case_ids = {
            **source_suite,
            "cases": [
                {key: value for key, value in case.items() if key != "case_id"}
                for case in source_suite["cases"]
            ],
        }
        fingerprint = _synthetic_content_hash(
            {
                "model": source_plan["model"],
                "suite": suite_without_case_ids,
                "resolved": source_plan["resolved"],
            },
            length=16,
        )
        source_plan["fingerprint"] = fingerprint

        campaign_root = self.results / "autoresearch"
        campaign_root.mkdir()
        campaign_dir = (
            campaign_root
            / "20260816T170000-0700-synthetic-autoresearch-campaign"
        )
        cells_root = campaign_dir / "cells"
        cells_root.mkdir(parents=True)
        descriptor = os.open(
            campaign_dir / ".autoresearch.lock",
            os.O_CREAT | os.O_EXCL | os.O_RDWR,
            0o600,
        )
        os.close(descriptor)
        shutil.rmtree(self.run_dir)

        candidate_specs = (
            ("reasoning-candidate", "reasoning_policy"),
            ("prefill-candidate", "chunked_prefill_size"),
            ("nextn-candidate", "nextn_bundle"),
        )
        cell_specs: list[tuple[str, str, str, str]] = [
            ("calibration-control-a", "calibration", "control", "control_a"),
            ("calibration-control-b", "calibration", "control", "control_b"),
        ]
        for candidate_id, _axis in candidate_specs:
            cell_specs.extend(
                (
                    (
                        f"{candidate_id}-screen-champion",
                        "screen",
                        candidate_id,
                        "champion",
                    ),
                    (
                        f"{candidate_id}-screen-candidate",
                        "screen",
                        candidate_id,
                        "candidate",
                    ),
                    (
                        f"{candidate_id}-confirmation-candidate",
                        "confirmation",
                        candidate_id,
                        "candidate",
                    ),
                    (
                        f"{candidate_id}-confirmation-champion",
                        "confirmation",
                        candidate_id,
                        "champion",
                    ),
                )
            )
        cells: list[dict[str, object]] = []
        run_dirs: list[Path] = []
        for ordinal, (cell_id, stage, candidate_id, arm) in enumerate(
            cell_specs, start=1
        ):
            cell_root = cells_root / f"{ordinal:02d}-{cell_id}"
            run_dir = cell_root / self.run_id
            run_dir.mkdir(parents=True)
            for name, payload in source_files.items():
                (run_dir / name).write_bytes(payload)
            profile_id = (
                candidate_id if arm == "candidate" else "synthetic-model"
            )
            plan = {
                **source_plan,
                "model": {**source_plan["model"], "id": profile_id},
            }
            if mixed_suite:
                plan["suite"] = _autoresearch_suite(plan["model"])
            plan["fingerprint"] = _synthetic_content_hash(
                {
                    "model": plan["model"],
                    "suite": suite_without_case_ids,
                    "resolved": plan["resolved"],
                },
                length=16,
            )
            plan["run_nonce"] = f"{ordinal:032x}"
            plan["integrity_hash"] = _synthetic_content_hash(plan)
            self.write_json(run_dir / "plan.json", plan)
            relative_run = run_dir.relative_to(campaign_dir).as_posix()
            cells.append(
                {
                    "arm": arm,
                    "candidate_id": candidate_id,
                    "cell_id": cell_id,
                    "ordinal": ordinal,
                    "plan_fingerprint": plan["fingerprint"],
                    "plan_integrity_hash": plan["integrity_hash"],
                    "profile_id": profile_id,
                    "run_dir": relative_run,
                    "run_nonce": plan["run_nonce"],
                    "stage": stage,
                }
            )
            run_dirs.append(run_dir)
        self.run_dir = run_dirs[0]

        proposals: list[dict[str, object]] = []
        for candidate_id, axis in candidate_specs:
            delta = {
                "axis": axis,
                "candidate_config_digest": "b" * 64,
                "candidate_value_json": "1",
                "champion_config_digest": "a" * 64,
                "champion_value_json": "0",
            }
            proposals.append(
                {
                    "axis": axis,
                    "candidate_id": candidate_id,
                    "delta": delta,
                    "delta_digest": _synthetic_content_hash(delta),
                }
            )
        policy: dict[str, object] = {}
        preview = {
            "baseline_id": "synthetic-model",
            "campaign_id": "synthetic-autoresearch",
            "cutoff": "2026-08-18T00:00:00+00:00",
            "execution_started": False,
            "planned_cell_count": len(cells),
            "policy": policy,
            "policy_digest": _synthetic_content_hash(policy),
            "proposals": proposals,
            "schema_version": 1,
            "suite_id": source_suite["id"],
        }
        campaign: dict[str, object] = {
            "cells": cells,
            "created_at": "2026-08-17T00:00:00.123+00:00",
            "execution_started": False,
            "harness_file_count": 1,
            "harness_tree_sha256": "d" * 64,
            "preview": preview,
            "preview_digest": _synthetic_content_hash(preview),
            "schema_version": 2,
        }
        campaign["integrity_hash"] = _synthetic_content_hash(campaign)
        self.write_json(campaign_dir / "campaign.json", campaign)
        return campaign_dir, run_dirs

    def write_memory_run(self, *, imperfect: bool = False) -> tuple[str, Path]:
        run_id = "20260824T010000Z-memory-evidence"
        run_dir = self.results / run_id
        run_dir.mkdir()
        model = _memory_model()
        suite = _memory_suite(model)
        self.write_json(
            run_dir / "plan.json",
            {
                "schema_version": 1,
                "model": model,
                "host_at_plan": {
                    "git_commit": "d" * 40,
                    "git_status": "",
                    "memtotal_kib": 128 * 1024 * 1024,
                    "nvidia_smi": "NVIDIA GB10, 580.82.09, 12.1",
                },
                "resolved": {
                    "llamacpp": {
                        "runtime_binary_sha256": MEMORY_OPERATION_LLAMACPP_DIGEST,
                        "runtime_source_revision": MEMORY_OPERATION_LLAMACPP_REVISION,
                    }
                },
                "suite": suite,
            },
        )
        events: list[dict[str, object]] = []

        def append(event: dict[str, object]) -> None:
            ordinal = len(events)
            events.append(
                {
                    "timestamp": (
                        f"2026-08-24T01:{ordinal // 60:02d}:{ordinal % 60:02d}Z"
                    ),
                    **event,
                }
            )

        append({"event": "run_start", "completed_cases_at_resume": []})
        append(
            {
                "event": "artifact_validation_complete",
                "backend": "llamacpp",
                "elapsed_s": 0.25,
                "model_sha256": model["model_digest"],
                "runtime_binary_sha256": model["runtime_digest"],
            }
        )
        append(
            {
                "event": "server_ready",
                "backend": "llamacpp",
                "keep_server_requested": False,
            }
        )
        append(
            {
                "event": "first_request_complete",
                "backend": "llamacpp",
                "result": self._request_result(completion_tokens=1),
            }
        )
        for case_index, case in enumerate(suite["cases"]):
            assert isinstance(case, dict)
            scenario_id = str(case["id"])
            case_id = str(case["case_id"])
            attempt_id = f"private-memory-attempt-{case_index}"
            append(
                {
                    "event": "case_start",
                    "case_id": case_id,
                    "attempt_id": attempt_id,
                    "kind": "memory",
                    "concurrency": 1,
                }
            )
            results = []
            for variant in range(3):
                result = _memory_result(scenario_id, variant)
                if imperfect and scenario_id == "memory-tier-placement" and variant == 1:
                    result["tier_correct"] = False
                    result["passed"] = False
                    result["failure_code"] = "operation_mismatch"
                results.append(result)
                append(
                    {
                        "event": "request_complete",
                        "case_id": case_id,
                        "attempt_id": attempt_id,
                        "kind": "memory",
                        "repetition": variant,
                        "burst_elapsed_s": result["elapsed_s"],
                        "request_tag": f"{RAW_REQUEST_ID}-{case_index}-{variant}",
                        "result": result,
                        "validation": {
                            "passed": result["passed"],
                            "reason": result["failure_code"],
                        },
                    }
                )
            append(
                {
                    "event": "case_complete",
                    "case_id": case_id,
                    "attempt_id": attempt_id,
                    "kind": "memory",
                    "concurrency": 1,
                    "elapsed_s": sum(
                        float(result["elapsed_s"]) for result in results
                    )
                    + 0.1,
                    "validation_passed": all(
                        result["passed"] is True for result in results
                    ),
                }
            )
        append({"event": "server_stopped", "backend": "llamacpp"})
        append({"event": "run_complete", "status": "completed"})
        self.write_jsonl(run_dir / "events.jsonl", events)
        summarize_run(run_dir)
        return run_id, run_dir

    def change_aggregate(self, value: float) -> None:
        self._write_standalone_results(aggregate_tps=value)

    def add_sglang_runtime_overlays(
        self, *, readonly_ple_cache: bool = False
    ) -> list[dict[str, str]]:
        overlay_dir = (
            self.results / "runtime-overlays" / "synthetic-sglang-recipe"
        )
        overlay_dir.mkdir(parents=True)
        files = {
            "qwen4_exp.py": (
                "from typing import Final\nMODEL_KIND: Final = 'synthetic'\n",
                "/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py",
            ),
            "qwen_sparse_attn_backend.py": (
                "from typing import Final\nBACKEND_KIND: Final = 'synthetic'\n",
                "/sgl-workspace/sglang/python/sglang/srt/layers/attention/"
                "qwen_sparse_attn_backend.py",
            ),
        }
        overlays: list[dict[str, str]] = []
        for basename, (source, container_path) in files.items():
            path = overlay_dir / basename
            path.write_text(source, encoding="utf-8")
            overlays.append(
                {
                    "container_path": container_path,
                    "digest": f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
                    "host_path": (
                        "results/runtime-overlays/synthetic-sglang-recipe/"
                        f"{basename}"
                    ),
                }
            )
        plan_path = self.run_dir / "plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["model"].update(
            {
                "backend": "sglang",
                "sglang_ple_mmap": True,
                "sglang_source_overlays": overlays,
            }
        )
        if readonly_ple_cache:
            plan["model"].update(
                {
                    "sglang_ple_cache_marker_digest": "sha256:" + "c" * 64,
                    "sglang_ple_cache_mode": "readonly",
                    "sglang_ple_cache_payload_digest": "sha256:" + "d" * 64,
                }
            )
        self.write_json(plan_path, plan)
        return overlays


def json_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key
            for child in value.values()
            for key in json_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in json_keys(child)}
    return set()


class EvidenceExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = EvidenceFixture(Path(self.temporary.name))

    def exported_bytes(self) -> dict[str, bytes]:
        return {
            str(path.relative_to(self.fixture.output)): path.read_bytes()
            for path in self.fixture.output.rglob("*")
            if path.is_file()
        }

    def move_run_to_group(self, group_name: str) -> Path:
        group = self.fixture.results / group_name
        group.mkdir()
        target = group / self.fixture.run_id
        self.fixture.run_dir.rename(target)
        self.fixture.run_dir = target
        return group

    def git(self, *arguments: str) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=self.temporary.name,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def add_matrix_source(self) -> None:
        matrix = self.fixture.results / "matrices" / "synthetic-matrix"
        matrix.mkdir()
        self.fixture.write_json(
            matrix / "matrix.json",
            {
                "models": ["synthetic-model"],
                "runs": [],
                "suite": "synthetic-suite",
            },
        )

    def bind_densespark_provenance(
        self, profile_id: str = DENSESPARK_PROFILE_ID
    ) -> dict[str, object]:
        """Bind the fixture to the exact C1 model, suite, and local receipt."""

        plan_path = self.fixture.run_dir / "plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        model = load_models(REPOSITORY / "manifests" / "models.toml")[
            profile_id
        ]
        model_record = model_spec_to_dict(model)
        suite = asdict(
            load_suite(
                REPOSITORY
                / "manifests"
                / "suites"
                / "qwen38_27b_densespark_c1.toml"
            )
        )
        suite.pop("protocol_digest", None)
        case = suite["cases"][0]
        case_id = (
            f"{case['id']}--"
            f"{_synthetic_content_hash({'model': model_record, 'case': case}, length=12)}"
        )
        suite["cases"] = [{**case, "case_id": case_id}]
        plan["model"] = model_record
        plan["suite"] = suite
        plan["resolved"] = {
            "densespark": densespark_expected_resolved_provenance(profile_id),
            "densespark_launch_policy": densespark_expected_launch_policy(),
            "image_digest": None,
        }
        plan["schema_version"] = 2
        suite_without_case_ids = {
            **suite,
            "cases": [
                {key: value for key, value in item.items() if key != "case_id"}
                for item in suite["cases"]
            ],
        }
        plan["fingerprint"] = _synthetic_content_hash(
            {
                "model": model_record,
                "suite": suite_without_case_ids,
                "resolved": plan["resolved"],
            },
            length=16,
        )
        plan.pop("integrity_hash", None)
        plan["integrity_hash"] = _synthetic_content_hash(plan)
        self.fixture.write_json(plan_path, plan)

        events_path = self.fixture.run_dir / "events.jsonl"
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
        ]
        for event in events:
            if event.get("case_id") == "chat-case":
                event["case_id"] = case_id
            if event.get("kind") == "chat":
                event["kind"] = "decode"
        self.fixture.write_jsonl(events_path, events)

        summary_path = self.fixture.run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_case = summary["cases"][0]
        summary_case["case_id"] = case_id
        summary_case["kind"] = "decode"
        self.fixture.write_json(summary_path, summary)

        telemetry_path = self.fixture.run_dir / "telemetry.jsonl"
        telemetry = [
            json.loads(line)
            for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        ]
        for sample in telemetry:
            phase = sample.get("phase")
            if isinstance(phase, str):
                sample["phase"] = phase.replace("chat-case", case_id)
        self.fixture.write_jsonl(telemetry_path, telemetry)
        return plan

    @staticmethod
    def reseal_densespark_plan(plan: dict[str, object]) -> None:
        model = plan["model"]
        suite = plan["suite"]
        resolved = plan["resolved"]
        assert isinstance(suite, dict)
        cases = suite["cases"]
        assert isinstance(cases, list)
        suite_without_case_ids = {
            **suite,
            "cases": [
                {key: value for key, value in case.items() if key != "case_id"}
                for case in cases
                if isinstance(case, dict)
            ],
        }
        plan["fingerprint"] = _synthetic_content_hash(
            {
                "model": model,
                "suite": suite_without_case_ids,
                "resolved": resolved,
            },
            length=16,
        )
        plan.pop("integrity_hash", None)
        plan["integrity_hash"] = _synthetic_content_hash(plan)

    @staticmethod
    def fake_campaign_export(
        campaign: Path,
        _results_root: Path,
        output_root: Path,
    ) -> dict[str, object]:
        relative = Path("campaigns") / campaign.name
        bundle_sha256, _ = _write_bundle(
            output_root,
            relative,
            {
                "manifest.json": {
                    "campaign_id": campaign.name,
                    "evidence_kind": "synthetic_campaign",
                    "schema_version": SCHEMA_VERSION,
                    "status": "complete",
                },
                "measurements.json": {
                    "measurement_count": 0,
                    "measurements": [],
                    "schema_version": SCHEMA_VERSION,
                },
                "telemetry.json": {
                    "capture_count": 0,
                    "captures": [],
                    "schema_version": SCHEMA_VERSION,
                },
            },
        )
        return {
            "bundle_sha256": bundle_sha256,
            "campaign_id": campaign.name,
            "evidence_kind": "synthetic_campaign",
            "file": f"campaigns/{campaign.name}/manifest.json",
            "status": "complete",
        }

    def export(
        self,
        *,
        harbor_results: tuple[Path, ...] = (),
        output: Path | None = None,
        replace: bool = False,
        require_existing_output: bool = False,
    ) -> dict[str, object]:
        # Production campaigns have deliberately exact file-set contracts. Patch
        # that independent adapter so this fixture can remain a small run corpus.
        with patch(
            "bench.evidence._export_campaign",
            side_effect=self.fake_campaign_export,
        ):
            return export_evidence(
                results_root=self.fixture.results,
                output_root=output or self.fixture.output,
                harbor_results=harbor_results,
                replace=replace,
                require_existing_output=require_existing_output,
            )

    def add_typed_startup_safety_gates(
        self,
        gates: list[dict[str, object]],
        *,
        journal: bool = True,
        summary: bool = True,
    ) -> None:
        annotations = [
            {
                "timestamp": f"2026-08-27T00:00:0{index}+00:00",
                "scope": "startup",
                "measurement_valid": False,
                "safety_gate": gate,
            }
            for index, gate in enumerate(gates, 1)
        ]
        if journal:
            events_path = self.fixture.run_dir / "events.jsonl"
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            events.extend(
                {
                    "timestamp": annotation["timestamp"],
                    "event": "measurement_annotation",
                    "schema_version": 2,
                    "scope": "startup",
                    "measurement_valid": False,
                    "safety_gate": annotation["safety_gate"],
                }
                for annotation in annotations
            )
            self.fixture.write_jsonl(events_path, events)
        if summary:
            summary_path = self.fixture.run_dir / "summary.json"
            source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            source_summary["measurement_annotations"].extend(annotations)
            source_summary["startup_measurement_annotations"] = annotations
            source_summary["startup_measurement_valid"] = False
            source_summary["startup_safety_gates"] = sorted(
                gates, key=lambda gate: str(gate["metric"])
            )
            self.fixture.write_json(summary_path, source_summary)

    def harbor_results(
        self, *, second_size_offset: int = 0
    ) -> tuple[Path, Path]:
        first = Path(self.temporary.name) / "harbor-first.json"
        second = Path(self.temporary.name) / "harbor-second.json"
        _write_private_json(
            first,
            _harbor_envelope(
                started_at="2026-08-18T02:00:00+00:00",
                finished_at="2026-08-18T02:00:20+00:00",
            ),
        )
        _write_private_json(
            second,
            _harbor_envelope(
                started_at="2026-08-18T03:00:00+00:00",
                finished_at="2026-08-18T03:00:20+00:00",
                size_offset=second_size_offset,
            ),
        )
        return first, second

    def refresh_campaign_checksums(
        self, campaign_id: str, *, evidence_root: Path | None = None
    ) -> None:
        root = evidence_root or self.fixture.output
        bundle = root / "campaigns" / campaign_id
        bundle_checksums = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(bundle.iterdir())
            if path.name != "checksums.json"
        }
        EvidenceFixture.write_json(
            bundle / "checksums.json",
            {"files": bundle_checksums, "schema_version": SCHEMA_VERSION},
        )
        index_path = root / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        campaign_entry = next(
            entry
            for entry in index["campaigns"]
            if entry["campaign_id"] == campaign_id
        )
        campaign_entry["bundle_sha256"] = hashlib.sha256(
            (bundle / "checksums.json").read_bytes()
        ).hexdigest()
        EvidenceFixture.write_json(index_path, index)
        root_checksums_path = root / "checksums.json"
        root_checksums = {
            str(path.relative_to(root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and path != root_checksums_path
        }
        EvidenceFixture.write_json(
            root_checksums_path,
            {"files": root_checksums, "schema_version": SCHEMA_VERSION},
        )

    def refresh_run_checksums(
        self, run_id: str, *, evidence_root: Path | None = None
    ) -> None:
        root = evidence_root or self.fixture.output
        bundle = root / "runs" / run_id
        bundle_checksums = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(bundle.iterdir())
            if path.name != "checksums.json"
        }
        EvidenceFixture.write_json(
            bundle / "checksums.json",
            {"files": bundle_checksums, "schema_version": SCHEMA_VERSION},
        )
        index_path = root / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        run_entry = next(
            entry for entry in index["runs"] if entry["run_id"] == run_id
        )
        run_entry["bundle_sha256"] = hashlib.sha256(
            (bundle / "checksums.json").read_bytes()
        ).hexdigest()
        EvidenceFixture.write_json(index_path, index)
        root_checksums_path = root / "checksums.json"
        root_checksums = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and path != root_checksums_path
        }
        EvidenceFixture.write_json(
            root_checksums_path,
            {"files": root_checksums, "schema_version": SCHEMA_VERSION},
        )

    def test_export_is_deterministic_and_excludes_raw_values(self) -> None:
        first = self.export()
        self.assertTrue(first["changed"])
        original = self.exported_bytes()

        second = self.export()

        self.assertFalse(second["changed"])
        self.assertEqual(original, self.exported_bytes())
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

        serialized_json = b"\n".join(
            data for name, data in original.items() if name.endswith(".json")
        ).decode("utf-8")
        for sentinel in (
            RAW_COMPLETION,
            RAW_REASONING,
            RAW_REQUEST_ID,
            RAW_HOST_PATH,
            RAW_SECRET,
            "private-attempt-a",
            "private-attempt-b",
            "raw-device-time-a",
        ):
            with self.subTest(sentinel=sentinel):
                self.assertNotIn(sentinel, serialized_json)

        all_keys: set[str] = set()
        for name, data in original.items():
            if name.endswith(".json"):
                all_keys.update(json_keys(json.loads(data)))
        self.assertTrue({"completion_tokens", "decode_tps"} <= all_keys)
        self.assertIn('"gpu_error_present"', serialized_json)
        for forbidden in (
            "content",
            "reasoning",
            "request_id",
            "request_tag",
            "tool_calls",
            "timestamp",
            "gpu_timestamp",
        ):
            self.assertNotIn(forbidden, all_keys)
        self.assertFalse(any(key.endswith("_path") for key in all_keys))

    def test_verifier_rejects_text_in_rechecksummed_telemetry_rows(self) -> None:
        self.export()
        target = Path(self.temporary.name) / "telemetry-text-tamper"
        shutil.copytree(self.fixture.output, target)
        run = target / "runs" / self.fixture.run_id
        chunk_path = next(run.glob("telemetry-*.json"))
        chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
        chunk["segments"][0]["rows"][0][2] = "Synthetic captured prompt text"
        self.fixture.write_json(chunk_path, chunk)
        self.refresh_run_checksums(self.fixture.run_id, evidence_root=target)
        with self.assertRaisesRegex(EvidenceError, "telemetry numeric scalar changed"):
            verify_evidence(target)

    def test_matched_graph_export_rejects_tampered_args_for_both_pairs(self) -> None:
        for profile_id in (
            QWEN38_27B_DSPARK_CUDA_GRAPH_FULL_PROFILE_ID,
            SM121_CUDA_GRAPH_BREAKABLE_PROFILE_ID,
        ):
            with self.subTest(profile_id=profile_id):
                self.fixture.write_matched_graph_run(profile_id)
                plan_path = self.fixture.run_dir / "plan.json"
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                plan["model"]["args"].extend(
                    ["--cuda-graph-max-bs", "8"]
                )
                self.fixture.write_json(plan_path, plan)
                output = Path(self.temporary.name) / f"tampered-{profile_id}"
                output.mkdir()
                with self.assertRaisesRegex(
                    EvidenceError, "graph model contract changed"
                ):
                    _export_run(
                        self.fixture.run_dir,
                        self.fixture.results.resolve(),
                        output,
                        None,
                    )

    def test_matched_graph_export_authenticates_raw_request_schedule(self) -> None:
        self.fixture.write_matched_graph_run(
            QWEN38_27B_DSPARK_CUDA_GRAPH_FULL_PROFILE_ID
        )
        events_path = self.fixture.run_dir / "events.jsonl"
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
        ]
        measured = next(
            event for event in events if event.get("event") == "request_complete"
        )
        measured["result"]["request_id"] = "forged-request-id"
        self.fixture.write_jsonl(events_path, events)
        output = Path(self.temporary.name) / "tampered-request-schedule"
        output.mkdir()
        with self.assertRaisesRegex(EvidenceError, "request schedule changed"):
            _export_run(
                self.fixture.run_dir,
                self.fixture.results.resolve(),
                output,
                None,
            )

    def test_matched_graph_export_reconciles_model_and_prompt_token_shape(self) -> None:
        profile_id = QWEN38_27B_DSPARK_CUDA_GRAPH_FULL_PROFILE_ID

        def response_model() -> None:
            events_path = self.fixture.run_dir / "events.jsonl"
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            measured = next(
                event for event in events if event.get("event") == "request_complete"
            )
            measured["result"]["response_model"] = "other-model"
            self.fixture.write_jsonl(events_path, events)

        def unequal_prompt_tokens() -> None:
            events_path = self.fixture.run_dir / "events.jsonl"
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            measured = next(
                event for event in events if event.get("event") == "request_complete"
            )
            measured["result"]["prompt_tokens"] += 1
            self.fixture.write_jsonl(events_path, events)

        def summary_prompt_total() -> None:
            summary_path = self.fixture.run_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["cases"][0]["prompt_tokens"] += 1
            self.fixture.write_json(summary_path, summary)

        for mutation in (
            response_model,
            unequal_prompt_tokens,
            summary_prompt_total,
        ):
            with self.subTest(mutation=mutation.__name__):
                self.fixture.write_matched_graph_run(profile_id)
                mutation()
                output = Path(self.temporary.name) / f"source-{mutation.__name__}"
                output.mkdir()
                with self.assertRaisesRegex(EvidenceError, "matched-prompt"):
                    _export_run(
                        self.fixture.run_dir,
                        self.fixture.results.resolve(),
                        output,
                        None,
                    )

    def test_matched_graph_verifier_rejects_refreshed_checksum_tampering(
        self,
    ) -> None:
        self.fixture.write_matched_graph_run(
            QWEN38_27B_DSPARK_CUDA_GRAPH_DISABLED_PROFILE_ID
        )
        self.export()
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

        def samples_decode_tps(root: Path) -> None:
            path = root / "runs" / self.fixture.run_id / "samples.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["samples"][1]["decode_tps"] += 1.0
            self.fixture.write_json(path, value)

        def samples_prompt_text(root: Path) -> None:
            path = root / "runs" / self.fixture.run_id / "samples.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["samples"][1]["prompt_text"] = "synthetic"
            self.fixture.write_json(path, value)

        def samples_request_id_alias(root: Path) -> None:
            path = root / "runs" / self.fixture.run_id / "samples.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["samples"][1]["requestId"] = "synthetic"
            self.fixture.write_json(path, value)

        def manifest_extra_text(root: Path) -> None:
            path = root / "runs" / self.fixture.run_id / "manifest.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["prompt_text"] = "synthetic"
            self.fixture.write_json(path, value)

        def manifest_hardware_prompt(root: Path) -> None:
            path = root / "runs" / self.fixture.run_id / "manifest.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["hardware"]["gpu"] = "Synthetic captured prompt text"
            self.fixture.write_json(path, value)

        def manifest_run_date_prompt(root: Path) -> None:
            path = root / "runs" / self.fixture.run_id / "manifest.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["run_date_utc"] = "Synthetic captured prompt text"
            self.fixture.write_json(path, value)

        def summary_payload_alias(root: Path) -> None:
            path = root / "runs" / self.fixture.run_id / "summary.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["aggregates"]["artifact_validation"] = {
                "toolPayload": "synthetic"
            }
            self.fixture.write_json(path, value)

        def coordinated_case_suffix(root: Path) -> None:
            run = root / "runs" / self.fixture.run_id
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            forged = (
                "matched-prompt-qwen38-27b-dspark-cuda-graph-d256-c1-v1"
                "--ffffffffffff"
            )
            manifest["suite"]["cases"][0]["case_id"] = forged
            self.fixture.write_json(manifest_path, manifest)
            samples_path = run / "samples.json"
            samples = json.loads(samples_path.read_text(encoding="utf-8"))
            for sample in samples["samples"][1:]:
                sample["case_id"] = forged
            self.fixture.write_json(samples_path, samples)
            summary_path = run / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["aggregates"]["cases"][0]["case_id"] = forged
            self.fixture.write_json(summary_path, summary)

        def coordinated_runtime_image(root: Path) -> None:
            path = root / "runs" / self.fixture.run_id / "manifest.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            forged = "0" * 64
            value["runtime"]["image"] = f"example/sglang@sha256:{forged}"
            value["runtime"]["image_sha256"] = forged
            value["artifacts"][0]["sha256"] = forged
            self.fixture.write_json(path, value)

        def coordinated_disabled_to_full_relabel(root: Path) -> None:
            run = root / "runs" / self.fixture.run_id
            repository = Path(__file__).resolve().parents[1]
            full_source = json.loads(
                json.dumps(
                    model_spec_to_dict(
                        load_models(repository / "manifests" / "models.toml")[
                            QWEN38_27B_DSPARK_CUDA_GRAPH_FULL_PROFILE_ID
                        ]
                    )
                )
            )
            full_model = _project_model({"model": full_source}, None)
            full_case_id = (
                "matched-prompt-qwen38-27b-dspark-cuda-graph-d256-c1-v1"
                "--e10754a3fc1a"
            )
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["model"] = full_model
            manifest["suite"]["cases"][0]["case_id"] = full_case_id
            self.fixture.write_json(manifest_path, manifest)
            samples_path = run / "samples.json"
            samples = json.loads(samples_path.read_text(encoding="utf-8"))
            for sample in samples["samples"][1:]:
                sample["case_id"] = full_case_id
            self.fixture.write_json(samples_path, samples)
            summary_path = run / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["aggregates"]["cases"][0]["case_id"] = full_case_id
            self.fixture.write_json(summary_path, summary)

        mutations = (
            samples_decode_tps,
            samples_prompt_text,
            samples_request_id_alias,
            manifest_extra_text,
            manifest_hardware_prompt,
            manifest_run_date_prompt,
            summary_payload_alias,
            coordinated_case_suffix,
            coordinated_runtime_image,
            coordinated_disabled_to_full_relabel,
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation.__name__):
                target = Path(self.temporary.name) / mutation.__name__
                shutil.copytree(self.fixture.output, target)
                mutation(target)
                self.refresh_run_checksums(
                    self.fixture.run_id, evidence_root=target
                )
                with self.assertRaises(EvidenceError):
                    verify_evidence(target)

    def test_historical_graph_run_id_cannot_be_downgraded(self) -> None:
        profile_id = QWEN38_27B_DSPARK_CUDA_GRAPH_DISABLED_PROFILE_ID
        self.fixture.write_matched_graph_run(profile_id)
        self.export()
        run_id = self.fixture.run_id
        baseline_index = json.loads(
            (self.fixture.output / "index.json").read_text(encoding="utf-8")
        )
        baseline_entry = next(
            entry for entry in baseline_index["runs"] if entry["run_id"] == run_id
        )
        baseline_manifest = json.loads(
            (
                self.fixture.output / "runs" / run_id / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        contract = {
            "bundle_sha256": baseline_entry["bundle_sha256"],
            "first_prompt_tokens": 10,
            "hardware": baseline_manifest["hardware"],
            "measured_prompt_tokens": 100,
            "profile_id": profile_id,
            "suite_id": "qwen38-27b-dspark-c1-cuda-graph",
        }

        target = Path(self.temporary.name) / "graph-protocol-downgrade"
        shutil.copytree(self.fixture.output, target)
        run = target / "runs" / run_id
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["model"]["id"] = "ordinary-sglang-model"
        manifest["suite"]["id"] = "ordinary-suite"
        case = manifest["suite"]["cases"][0]
        case["id"] = "ordinary-decode"
        case["case_id"] = "ordinary-decode--53e714d98e94"
        case.pop("prompt_schedule")
        self.fixture.write_json(manifest_path, manifest)
        samples_path = run / "samples.json"
        samples = json.loads(samples_path.read_text(encoding="utf-8"))
        for sample in samples["samples"][1:]:
            sample["case_id"] = "ordinary-decode--53e714d98e94"
        self.fixture.write_json(samples_path, samples)
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["aggregates"]["suite"] = "ordinary-suite"
        summary["aggregates"]["cases"][0]["case_id"] = (
            "ordinary-decode--53e714d98e94"
        )
        self.fixture.write_json(summary_path, summary)
        self.refresh_run_checksums(run_id, evidence_root=target)

        with patch.dict(
            "bench.evidence._MATCHED_PROMPT_GRAPH_PUBLISHED_RUN_CONTRACTS",
            {run_id: contract},
            clear=False,
        ):
            with self.assertRaisesRegex(
                EvidenceError, "matched-prompt historical bundle identity changed"
            ):
                verify_evidence(target)

        changed_index = json.loads(
            (target / "index.json").read_text(encoding="utf-8")
        )
        changed_entry = next(
            entry for entry in changed_index["runs"] if entry["run_id"] == run_id
        )
        marker_contract = {**contract, "bundle_sha256": changed_entry["bundle_sha256"]}
        with patch.dict(
            "bench.evidence._MATCHED_PROMPT_GRAPH_PUBLISHED_RUN_CONTRACTS",
            {run_id: marker_contract},
            clear=False,
        ):
            with self.assertRaisesRegex(EvidenceError, "matched-prompt"):
                verify_evidence(target)

    def test_densespark_receipts_export_deterministically_and_verify(self) -> None:
        source = densespark_expected_resolved_provenance()
        launch_policy = densespark_expected_launch_policy()
        self.bind_densespark_provenance()

        first = self.export()
        self.assertTrue(first["changed"])
        original = self.exported_bytes()
        manifest_path = (
            self.fixture.output
            / "runs"
            / self.fixture.run_id
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual("qwen38-27b-densespark-c1", manifest["suite"]["id"])
        self.assertEqual(1, len(manifest["suite"]["cases"]))
        self.assertEqual("decode", manifest["suite"]["cases"][0]["kind"])
        self.assertEqual(
            {
                "cache_namespace": source["cache_namespace"],
                "configuration_sha256": str(
                    source["configuration_sha256"]
                ).removeprefix("sha256:"),
                "docker_image_sha256": DENSESPARK_LOCAL_IMAGE_ID.removeprefix(
                    "sha256:"
                ),
                "model_revision": DENSESPARK_MODEL_REVISION,
                "pq_artifact_sha256": DENSESPARK_PQ_SHA256.removeprefix(
                    "sha256:"
                ),
                "pq_artifact_size_bytes": source["pq_artifact_size_bytes"],
                "weight_file_count": source["weight_file_count"],
                "weight_size_bytes": source["weight_size_bytes"],
                "launch_policy_binding": "frozen-v1",
                "launch_policy": {
                    **launch_policy,
                    "sha256": str(launch_policy["sha256"]).removeprefix(
                        "sha256:"
                    ),
                },
            },
            manifest["runtime"]["densespark"],
        )
        self.assertEqual(
            manifest["runtime"]["densespark"]["launch_policy"][
                "docker_network_egress"
            ],
            "capable",
        )
        self.assertEqual(
            manifest["runtime"]["densespark"]["launch_policy"][
                "docker_network_isolation"
            ],
            "none",
        )
        self.assertNotIn("docker_image_id", manifest["runtime"]["densespark"])
        self.assertFalse(
            any(key.endswith("_path") for key in json_keys(manifest))
        )

        second = self.export()
        self.assertFalse(second["changed"])
        self.assertEqual(original, self.exported_bytes())
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

    def test_warmup_sync_receipt_is_path_free_exact_and_deterministic(self) -> None:
        source = densespark_expected_resolved_provenance(
            DENSESPARK_WARMUP_SYNC_PROFILE_ID
        )
        self.bind_densespark_provenance(DENSESPARK_WARMUP_SYNC_PROFILE_ID)

        self.export()
        original = self.exported_bytes()
        manifest_path = (
            self.fixture.output
            / "runs"
            / self.fixture.run_id
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt = manifest["runtime"]["densespark"]
        self.assertEqual(DENSESPARK_WARMUP_SYNC_MODE, receipt["mode"])
        self.assertEqual(
            DENSESPARK_WARMUP_SYNC_LOCAL_IMAGE_ID.removeprefix("sha256:"),
            receipt["docker_image_sha256"],
        )
        for key in (
            "dockerignore_sha256",
            "fused_sigmoid_source_sha256",
            "image_recipe_sha256",
            "kernel_warmup_source_sha256",
            "mamba_utils_source_sha256",
            "probe_sha256",
            "qwen_gdn_source_sha256",
            "qwen_warmup_source_sha256",
            "vllm_entrypoint_sha256",
        ):
            self.assertEqual(
                str(source[key]).removeprefix("sha256:"), receipt[key]
            )
        self.assertFalse(any(key.endswith("_path") for key in json_keys(manifest)))

        second = self.export()
        self.assertFalse(second["changed"])
        self.assertEqual(original, self.exported_bytes())
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

    def test_exact_allowlisted_legacy_plan_publishes_no_policy_claim(self) -> None:
        plan = self.bind_densespark_provenance()
        resolved = plan["resolved"]
        self.assertIsInstance(resolved, dict)
        del resolved["densespark_launch_policy"]
        self.reseal_densespark_plan(plan)
        integrity = plan["integrity_hash"]
        self.assertIsInstance(integrity, str)
        self.fixture.write_json(self.fixture.run_dir / "plan.json", plan)

        with (
            patch(
                "bench.evidence._DENSESPARK_LEGACY_LAUNCH_POLICY_PLAN_INTEGRITIES",
                frozenset({integrity}),
            ),
            patch.dict(
                "bench.evidence._DENSESPARK_LEGACY_LAUNCH_POLICY_PLAN_INTEGRITY_BY_RUN_ID",
                {self.fixture.run_id: integrity},
                clear=False,
            ),
        ):
            self.export()
            manifest = json.loads(
                (
                    self.fixture.output
                    / "runs"
                    / self.fixture.run_id
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            receipt = manifest["runtime"]["densespark"]
            self.assertEqual(receipt["launch_policy_binding"], "legacy-unbound")
            self.assertEqual(receipt["legacy_plan_integrity_sha256"], integrity)
            self.assertNotIn("launch_policy", receipt)
            self.assertEqual(
                "verified", verify_evidence(self.fixture.output)["status"]
            )

    def test_unallowlisted_legacy_plan_is_rejected(self) -> None:
        plan = self.bind_densespark_provenance()
        resolved = plan["resolved"]
        self.assertIsInstance(resolved, dict)
        del resolved["densespark_launch_policy"]
        self.reseal_densespark_plan(plan)
        self.fixture.write_json(self.fixture.run_dir / "plan.json", plan)
        with self.assertRaisesRegex(EvidenceError, "legacy launch policy"):
            self.export()

    def test_densespark_failed_run_projects_only_host_safety_scalars(self) -> None:
        self.bind_densespark_provenance()
        events = [
            {
                "completed_cases_at_resume": [],
                "event": "run_start",
                "timestamp": "2026-08-17T00:00:00Z",
            },
            {
                "event": "measurement_started",
                "timestamp": "2026-08-17T00:00:01Z",
            },
            {
                "code": "swap_growth_above_maximum",
                "event": "host_safety_breach",
                "limit_kib": 524_288,
                "memavailable_kib": 33_823_156,
                "observed_kib": 582_000,
                "stage": "server_start",
                "starting_swap_used_kib": 259_564,
                "swap_used_kib": 841_564,
                "timestamp": "2026-08-17T00:00:02Z",
            },
            {
                "error": "synthetic safety failure at /private/source/path",
                "error_type": "HostSafetyError",
                "event": "run_aborted",
                "stage": "server_start",
                "timestamp": "2026-08-17T00:00:03Z",
            },
        ]
        self.fixture.write_jsonl(self.fixture.run_dir / "events.jsonl", events)
        summary_path = self.fixture.run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "cases": [],
                "completed_cases": 0,
                "failed_cases": [],
                "first_request_after_start": None,
                "first_request_telemetry": None,
                "measurement_invalid_cases": [],
                "run_completion_status": None,
                "run_error": {
                    "error": "synthetic safety failure at /private/source/path",
                    "error_type": "HostSafetyError",
                    "stage": "server_start",
                },
                "status": "aborted",
                "validation_failed_cases": [],
            }
        )
        self.fixture.write_json(summary_path, summary)

        self.export()
        manifest_path = (
            self.fixture.output
            / "runs"
            / self.fixture.run_id
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lifecycle = manifest["lifecycle"]
        self.assertEqual("run_aborted", lifecycle["terminal_event"])
        self.assertEqual(
            {"exception_type": "HostSafetyError", "stage": "server_start"},
            lifecycle["failure"],
        )
        self.assertEqual(
            {
                "code": "swap_growth_above_maximum",
                "limit_bytes": 536_870_912,
                "memavailable_bytes": 34_634_911_744,
                "observed_bytes": 595_968_000,
                "stage": "server_start",
                "starting_swap_used_bytes": 265_793_536,
                "swap_used_bytes": 861_761_536,
            },
            lifecycle["host_safety_breach"],
        )
        serialized = json.dumps(manifest, sort_keys=True)
        self.assertNotIn("private/source/path", serialized)
        self.assertNotIn("error", json_keys(manifest))
        self.assertFalse(any(key.endswith("_kib") for key in json_keys(manifest)))
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

        lifecycle["host_safety_breach"]["limit_bytes"] = 1
        self.fixture.write_json(manifest_path, manifest)
        self.refresh_run_checksums(self.fixture.run_id)
        with self.assertRaisesRegex(EvidenceError, "host-safety limit changed"):
            verify_evidence(self.fixture.output)

    def test_densespark_host_safety_source_schema_rejects_extra_payload(self) -> None:
        self.bind_densespark_provenance()
        events_path = self.fixture.run_dir / "events.jsonl"
        events = [
            {
                "event": "run_start",
                "timestamp": "2026-08-17T00:00:00Z",
            },
            {
                "code": "swap_growth_above_maximum",
                "command": ["private", "command"],
                "event": "host_safety_breach",
                "limit_kib": 524_288,
                "memavailable_kib": 33_823_156,
                "observed_kib": 582_000,
                "stage": "server_start",
                "starting_swap_used_kib": 259_564,
                "swap_used_kib": 841_564,
                "timestamp": "2026-08-17T00:00:02Z",
            },
            {
                "event": "run_complete",
                "timestamp": "2026-08-17T00:00:03Z",
            },
        ]
        self.fixture.write_jsonl(events_path, events)
        with self.assertRaisesRegex(
            EvidenceError, "host-safety source schema changed"
        ):
            self.export()

    def test_densespark_source_receipt_tampering_fails_closed(self) -> None:
        mutations = {
            "cache_namespace": "densespark-v1-" + "0" * 64,
            "configuration_sha256": "sha256:" + "0" * 64,
            "docker_image_id": "sha256:" + "0" * 64,
            "model_revision": "0" * 40,
            "pq_artifact_sha256": "sha256:" + "0" * 64,
            "pq_artifact_size_bytes": 1,
            "weight_file_count": 7,
            "weight_size_bytes": 1,
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                plan = self.bind_densespark_provenance()
                resolved = plan["resolved"]
                self.assertIsInstance(resolved, dict)
                receipt = resolved["densespark"]
                self.assertIsInstance(receipt, dict)
                receipt[key] = value
                self.reseal_densespark_plan(plan)
                self.fixture.write_json(self.fixture.run_dir / "plan.json", plan)
                with self.assertRaisesRegex(
                    EvidenceError, "DenseSpark resolved provenance changed"
                ):
                    self.export()

    def test_densespark_source_launch_policy_tampering_fails_closed(self) -> None:
        mutations = {
            "host_safety_min_memavailable_bytes": 1,
            "environment_hf_hub_offline": "0",
            "docker_pull_policy": "always",
            "publish_host": "0.0.0.0",
            "label_managed": "ai.sparkbench.managed=false",
            "docker_network": "none",
            "sha256": "sha256:" + "0" * 64,
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                plan = self.bind_densespark_provenance()
                resolved = plan["resolved"]
                self.assertIsInstance(resolved, dict)
                launch_policy = resolved["densespark_launch_policy"]
                self.assertIsInstance(launch_policy, dict)
                launch_policy[key] = value
                self.reseal_densespark_plan(plan)
                self.fixture.write_json(self.fixture.run_dir / "plan.json", plan)
                with self.assertRaisesRegex(
                    EvidenceError, "DenseSpark resolved launch policy changed"
                ):
                    self.export()

    def test_densespark_plan_integrity_and_fingerprint_fail_closed(self) -> None:
        plan = self.bind_densespark_provenance()
        plan["integrity_hash"] = "0" * 64
        self.fixture.write_json(self.fixture.run_dir / "plan.json", plan)
        with self.assertRaisesRegex(EvidenceError, "source plan integrity changed"):
            self.export()

        plan = self.bind_densespark_provenance()
        plan["fingerprint"] = "0" * 16
        plan.pop("integrity_hash", None)
        plan["integrity_hash"] = _synthetic_content_hash(plan)
        self.fixture.write_json(self.fixture.run_dir / "plan.json", plan)
        with self.assertRaisesRegex(EvidenceError, "source plan fingerprint changed"):
            self.export()

    def test_published_densespark_receipt_tampering_fails_closed(self) -> None:
        self.bind_densespark_provenance()
        self.export()
        manifest_path = (
            self.fixture.output
            / "runs"
            / self.fixture.run_id
            / "manifest.json"
        )
        original = json.loads(manifest_path.read_text(encoding="utf-8"))
        mutations = {
            "cache_namespace": "densespark-v1-" + "0" * 64,
            "configuration_sha256": "0" * 64,
            "docker_image_sha256": "0" * 64,
            "model_revision": "0" * 40,
            "pq_artifact_sha256": "0" * 64,
            "pq_artifact_size_bytes": 1,
            "weight_file_count": 7,
            "weight_size_bytes": 1,
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                manifest = json.loads(json.dumps(original))
                manifest["runtime"]["densespark"][key] = value
                self.fixture.write_json(manifest_path, manifest)
                self.refresh_run_checksums(self.fixture.run_id)
                with self.assertRaisesRegex(
                    EvidenceError, "published DenseSpark provenance changed"
                ):
                    verify_evidence(self.fixture.output)

        for key, value in (
            ("host_safety_min_memavailable_bytes", 1),
            ("environment_vllm_no_usage_stats", "0"),
            ("docker_network_egress", "blocked"),
            ("publish_host_port", 8001),
            ("label_run_binding", "ai.sparkbench.run=other"),
            ("sha256", "0" * 64),
        ):
            with self.subTest(launch_policy_key=key):
                manifest = json.loads(json.dumps(original))
                manifest["runtime"]["densespark"]["launch_policy"][key] = value
                self.fixture.write_json(manifest_path, manifest)
                self.refresh_run_checksums(self.fixture.run_id)
                with self.assertRaisesRegex(
                    EvidenceError, "published DenseSpark provenance changed"
                ):
                    verify_evidence(self.fixture.output)

        manifest = json.loads(json.dumps(original))
        receipt = manifest["runtime"]["densespark"]
        receipt.pop("launch_policy")
        receipt["launch_policy_binding"] = "legacy-unbound"
        receipt["legacy_plan_integrity_sha256"] = (
            "7f3f69e09b180a1a532979e8383b20cdedc25eeb40731713303d75bae7bdc80b"
        )
        self.fixture.write_json(manifest_path, manifest)
        self.refresh_run_checksums(self.fixture.run_id)
        with self.assertRaisesRegex(EvidenceError, "legacy binding is not allowlisted"):
            verify_evidence(self.fixture.output)

        manifest = json.loads(json.dumps(original))
        del manifest["runtime"]["densespark"]
        self.fixture.write_json(manifest_path, manifest)
        self.refresh_run_checksums(self.fixture.run_id)
        with self.assertRaisesRegex(
            EvidenceError, "published DenseSpark provenance changed"
        ):
            verify_evidence(self.fixture.output)

    def test_exact_legacy_densespark_bundle_rejects_rechecksummed_tampering(
        self,
    ) -> None:
        self.bind_densespark_provenance()
        self.export()
        run_id = self.fixture.run_id
        run = self.fixture.output / "runs" / run_id
        index = json.loads(
            (self.fixture.output / "index.json").read_text(encoding="utf-8")
        )
        entry = next(item for item in index["runs"] if item["run_id"] == run_id)
        baseline_contract = {
            "bundle_sha256": entry["bundle_sha256"],
            "manifest": json.loads(
                (run / "manifest.json").read_text(encoding="utf-8")
            ),
            "samples": json.loads(
                (run / "samples.json").read_text(encoding="utf-8")
            ),
            "summary": json.loads(
                (run / "summary.json").read_text(encoding="utf-8")
            ),
        }
        plan_map = {run_id: "0" * 64}

        with (
            patch.dict(
                "bench.evidence._DENSESPARK_LEGACY_PUBLISHED_BUNDLE_CONTRACTS",
                {run_id: baseline_contract},
                clear=True,
            ),
            patch.dict(
                "bench.evidence._DENSESPARK_LEGACY_LAUNCH_POLICY_PLAN_INTEGRITY_BY_RUN_ID",
                plan_map,
                clear=True,
            ),
        ):
            self.assertEqual(
                "verified", verify_evidence(self.fixture.output)["status"]
            )

        def manifest_payload(root: Path) -> None:
            path = root / "runs" / run_id / "manifest.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["trace_excerpt"] = "Synthetic captured prompt text"
            self.fixture.write_json(path, value)

        def suite_scalar(root: Path) -> None:
            path = root / "runs" / run_id / "manifest.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["suite"]["cases"][0]["max_output_tokens"] = 999
            self.fixture.write_json(path, value)

        def samples_payload(root: Path) -> None:
            path = root / "runs" / run_id / "samples.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["samples"].append(
                {
                    "sample_index": len(value["samples"]) + 1,
                    "trace_excerpt": "Synthetic captured prompt text",
                }
            )
            value["sample_count"] = len(value["samples"])
            self.fixture.write_json(path, value)

        def summary_scalar(root: Path) -> None:
            path = root / "runs" / run_id / "summary.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["aggregates"]["claimed_decode_tps"] = 999.0
            self.fixture.write_json(path, value)

        def coordinated_status(root: Path) -> None:
            manifest_path = root / "runs" / run_id / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "aborted"
            self.fixture.write_json(manifest_path, manifest)
            index_path = root / "index.json"
            changed_index = json.loads(index_path.read_text(encoding="utf-8"))
            changed_entry = next(
                item for item in changed_index["runs"] if item["run_id"] == run_id
            )
            changed_entry["status"] = "aborted"
            changed_index["run_status_counts"] = {"aborted": 1}
            self.fixture.write_json(index_path, changed_index)

        mutations = (
            (manifest_payload, "manifest"),
            (suite_scalar, "manifest"),
            (samples_payload, "samples"),
            (summary_scalar, "summary"),
            (coordinated_status, "manifest"),
        )
        for mutation, document in mutations:
            with self.subTest(mutation=mutation.__name__):
                target = Path(self.temporary.name) / f"legacy-{mutation.__name__}"
                shutil.copytree(self.fixture.output, target)
                mutation(target)
                self.refresh_run_checksums(run_id, evidence_root=target)
                changed_index = json.loads(
                    (target / "index.json").read_text(encoding="utf-8")
                )
                changed_entry = next(
                    item for item in changed_index["runs"] if item["run_id"] == run_id
                )
                semantic_contract = json.loads(json.dumps(baseline_contract))
                # Let the test reach the exact document comparison even if an
                # attacker has refreshed every checksum after editing it.
                semantic_contract["bundle_sha256"] = changed_entry["bundle_sha256"]
                with (
                    patch.dict(
                        "bench.evidence._DENSESPARK_LEGACY_PUBLISHED_BUNDLE_CONTRACTS",
                        {run_id: semantic_contract},
                        clear=True,
                    ),
                    patch.dict(
                        "bench.evidence._DENSESPARK_LEGACY_LAUNCH_POLICY_PLAN_INTEGRITY_BY_RUN_ID",
                        plan_map,
                        clear=True,
                    ),
                ):
                    with self.assertRaisesRegex(
                        EvidenceError,
                        rf"legacy DenseSpark published {document} contract changed",
                    ):
                        verify_evidence(target)

        target = Path(self.temporary.name) / "legacy-bundle-identity"
        shutil.copytree(self.fixture.output, target)
        summary_scalar(target)
        self.refresh_run_checksums(run_id, evidence_root=target)
        with (
            patch.dict(
                "bench.evidence._DENSESPARK_LEGACY_PUBLISHED_BUNDLE_CONTRACTS",
                {run_id: baseline_contract},
                clear=True,
            ),
            patch.dict(
                "bench.evidence._DENSESPARK_LEGACY_LAUNCH_POLICY_PLAN_INTEGRITY_BY_RUN_ID",
                plan_map,
                clear=True,
            ),
        ):
            with self.assertRaisesRegex(
                EvidenceError, "legacy DenseSpark published bundle identity changed"
            ):
                verify_evidence(target)

    def test_sglang_runtime_overlays_export_only_pinned_basenames_and_hashes(
        self,
    ) -> None:
        overlays = self.fixture.add_sglang_runtime_overlays()

        self.export()

        manifest_path = (
            self.fixture.output
            / "runs"
            / self.fixture.run_id
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertIs(manifest["runtime"]["sglang_ple_mmap"], True)
        self.assertEqual("writable", manifest["runtime"]["sglang_ple_cache_mode"])
        self.assertEqual(1, manifest["runtime"]["sglang_provenance_version"])
        expected_overlays = [
            {
                "sha256": overlay["digest"].removeprefix("sha256:"),
                "target": Path(overlay["host_path"]).name,
            }
            for overlay in sorted(
                overlays, key=lambda value: Path(value["host_path"]).name
            )
        ]
        self.assertEqual(
            expected_overlays,
            manifest["runtime"]["sglang_source_overlay_artifacts"],
        )
        self.assertEqual(
            [
                {
                    "role": f"sglang_source_overlay_{index}",
                    "sha256": overlay["digest"].removeprefix("sha256:"),
                    "target": Path(overlay["host_path"]).name,
                }
                for index, overlay in enumerate(
                    sorted(overlays, key=lambda value: Path(value["host_path"]).name),
                    1,
                )
            ],
            [
                artifact
                for artifact in manifest["artifacts"]
                if artifact["role"].startswith("sglang_source_overlay_")
            ],
        )
        serialized = manifest_path.read_text(encoding="utf-8")
        self.assertNotIn("runtime-overlays", serialized)
        self.assertNotIn("sgl-workspace", serialized)
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

    def test_sglang_provenance_cannot_be_wholly_removed(self) -> None:
        self.fixture.add_sglang_runtime_overlays(readonly_ple_cache=True)
        self.export()
        manifest_path = (
            self.fixture.output
            / "runs"
            / self.fixture.run_id
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["runtime"] = {
            key: value
            for key, value in manifest["runtime"].items()
            if not key.startswith("sglang_")
        }
        manifest["artifacts"] = [
            artifact
            for artifact in manifest["artifacts"]
            if not artifact["role"].startswith("sglang_source_overlay_")
        ]
        self.fixture.write_json(manifest_path, manifest)
        self.refresh_run_checksums(self.fixture.run_id)

        with self.assertRaisesRegex(EvidenceError, "provenance is required"):
            verify_evidence(self.fixture.output)

        manifest["model"]["backend"] = "vllm"
        self.fixture.write_json(manifest_path, manifest)
        self.refresh_run_checksums(self.fixture.run_id)
        with self.assertRaisesRegex(EvidenceError, "backend changed"):
            verify_evidence(self.fixture.output)

    def test_legacy_sglang_source_is_reexported_with_explicit_unknown_state(
        self,
    ) -> None:
        plan_path = self.fixture.run_dir / "plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["model"]["backend"] = "sglang"
        self.fixture.write_json(plan_path, plan)

        self.export()

        manifest = json.loads(
            (
                self.fixture.output
                / "runs"
                / self.fixture.run_id
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertIsNone(manifest["runtime"]["sglang_ple_mmap"])
        self.assertEqual(
            "legacy_unspecified",
            manifest["runtime"]["sglang_ple_cache_mode"],
        )
        self.assertEqual(
            [], manifest["runtime"]["sglang_source_overlay_artifacts"]
        )
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

    def test_sglang_overlay_projection_is_bound_to_numbered_artifacts(self) -> None:
        self.fixture.add_sglang_runtime_overlays(readonly_ple_cache=True)
        self.export()
        manifest_path = (
            self.fixture.output
            / "runs"
            / self.fixture.run_id
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["runtime"]["sglang_source_overlay_artifacts"].pop()
        self.fixture.write_json(manifest_path, manifest)
        self.refresh_run_checksums(self.fixture.run_id)

        with self.assertRaisesRegex(EvidenceError, "artifact binding changed"):
            verify_evidence(self.fixture.output)

    def test_readonly_sglang_ple_cache_provenance_is_atomic_and_typed(self) -> None:
        self.fixture.add_sglang_runtime_overlays(readonly_ple_cache=True)

        self.export()

        manifest_path = (
            self.fixture.output
            / "runs"
            / self.fixture.run_id
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runtime = manifest["runtime"]
        self.assertEqual("readonly", runtime["sglang_ple_cache_mode"])
        self.assertEqual("c" * 64, runtime["sglang_ple_cache_marker_sha256"])
        self.assertEqual("d" * 64, runtime["sglang_ple_cache_payload_sha256"])
        self.assertIs(runtime["sglang_ple_mmap"], True)
        self.assertEqual(1, runtime["sglang_provenance_version"])
        self.assertEqual(2, len(runtime["sglang_source_overlay_artifacts"]))
        self.assertNotIn("sglang_ple_cache_marker_digest", runtime)
        self.assertNotIn("sglang_ple_cache_payload_digest", runtime)
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

        malformed = json.loads(json.dumps(manifest))
        malformed["runtime"]["sglang_ple_cache_mode"] = []
        self.fixture.write_json(manifest_path, malformed)
        self.refresh_run_checksums(self.fixture.run_id)
        with self.assertRaisesRegex(EvidenceError, "cache mode is invalid"):
            verify_evidence(self.fixture.output)

        self.fixture.write_json(manifest_path, manifest)
        del runtime["sglang_ple_cache_payload_sha256"]
        self.fixture.write_json(manifest_path, manifest)
        self.refresh_run_checksums(self.fixture.run_id)
        with self.assertRaisesRegex(EvidenceError, "provenance is incomplete"):
            verify_evidence(self.fixture.output)

    def test_explicit_mapped_ple_control_uses_v2_omission_dimension(self) -> None:
        overlays = self.fixture.add_sglang_runtime_overlays(
            readonly_ple_cache=True
        )
        exact_overlay_digests = {
            "qwen4_exp.py": (
                "sha256:bcdc2c86aa59784ffe27d53c8d214e56"
                "b6aa45c02b1d5841fd956d1f006d6030"
            ),
            "qwen_sparse_attn_backend.py": (
                "sha256:e30566492e1502f94a4c7fed42d90b5"
                "23bbb662580c628459e6e63c7b5263c75"
            ),
        }
        for overlay in overlays:
            overlay["digest"] = exact_overlay_digests[
                Path(overlay["host_path"]).name
            ]
        plan_path = self.fixture.run_dir / "plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["model"].update(
            {
                "quantization": "nvfp4+ple-fp8-mapped+nextn-bf16",
                "recipe_revision": "bf2b7c75870d3703730b6bd8f3bb93dc622c278d",
                "recipe_source": "hashd1ve/qwen38-flash-next-one-dgx-spark",
                "revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
                "sglang_ple_cache_marker_digest": (
                    "sha256:f0ef55e4e4dec9b6b936a42af4ca2e"
                    "b9b2f24ced373b1e216f7a6d507b171665"
                ),
                "sglang_ple_cache_payload_digest": (
                    "sha256:b070f9644adf93794d8a1030584ab705"
                    "809387e64396a9327a68fa3a3a6666b3"
                ),
                "sglang_ple_omitted": False,
                "sglang_source_overlays": overlays,
                "source": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
            }
        )
        self.fixture.write_json(plan_path, plan)

        with patch(
            "bench.evidence._validate_runtime_overlay_tree", return_value=True
        ):
            self.export()

        manifest_path = (
            self.fixture.output
            / "runs"
            / self.fixture.run_id
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runtime = manifest["runtime"]
        self.assertEqual(2, runtime["sglang_provenance_version"])
        self.assertIs(runtime["sglang_ple_omitted"], False)
        self.assertEqual("readonly", runtime["sglang_ple_cache_mode"])
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

        pristine = json.loads(json.dumps(manifest))
        runtime.pop("sglang_ple_omitted")
        runtime["sglang_provenance_version"] = 1
        self.fixture.write_json(manifest_path, manifest)
        self.refresh_run_checksums(self.fixture.run_id)
        with self.assertRaisesRegex(EvidenceError, "lost its explicit"):
            verify_evidence(self.fixture.output)

        manifest = json.loads(json.dumps(pristine))
        manifest["runtime"].pop("recipe_revision")
        self.fixture.write_json(manifest_path, manifest)
        self.refresh_run_checksums(self.fixture.run_id)
        with self.assertRaisesRegex(EvidenceError, "recipe identity changed"):
            verify_evidence(self.fixture.output)

    def test_omitted_sglang_ple_ablation_is_explicit_and_verifiable(self) -> None:
        overlays = self.fixture.add_sglang_runtime_overlays()
        exact_overlay_digests = {
            "qwen4_exp.py": (
                "sha256:bcdc2c86aa59784ffe27d53c8d214e56"
                "b6aa45c02b1d5841fd956d1f006d6030"
            ),
            "qwen_sparse_attn_backend.py": (
                "sha256:e30566492e1502f94a4c7fed42d90b5"
                "23bbb662580c628459e6e63c7b5263c75"
            ),
        }
        for overlay in overlays:
            overlay["digest"] = exact_overlay_digests[
                Path(overlay["host_path"]).name
            ]
        plan_path = self.fixture.run_dir / "plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["model"].update(
            {
                "quantization": "nvfp4+ple-omitted-ablation+nextn-bf16",
                "recipe_revision": "bf2b7c75870d3703730b6bd8f3bb93dc622c278d",
                "recipe_source": "hashd1ve/qwen38-flash-next-one-dgx-spark",
                "revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
                "sglang_ple_mmap": False,
                "sglang_ple_omitted": True,
                "sglang_source_overlays": overlays,
                "source": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
            }
        )
        self.fixture.write_json(plan_path, plan)

        with patch(
            "bench.evidence._validate_runtime_overlay_tree", return_value=True
        ):
            self.export()

        manifest_path = (
            self.fixture.output
            / "runs"
            / self.fixture.run_id
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runtime = manifest["runtime"]
        self.assertEqual("disabled", runtime["sglang_ple_cache_mode"])
        self.assertIs(runtime["sglang_ple_mmap"], False)
        self.assertIs(runtime["sglang_ple_omitted"], True)
        self.assertEqual(2, runtime["sglang_provenance_version"])
        self.assertEqual(2, len(runtime["sglang_source_overlay_artifacts"]))
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

        pristine = json.loads(json.dumps(manifest))
        runtime["sglang_ple_omitted"] = False
        self.fixture.write_json(manifest_path, manifest)
        self.refresh_run_checksums(self.fixture.run_id)
        with self.assertRaisesRegex(EvidenceError, "flag and model label disagree"):
            verify_evidence(self.fixture.output)

        manifest = json.loads(json.dumps(pristine))
        manifest["runtime"].pop("recipe_revision")
        self.fixture.write_json(manifest_path, manifest)
        self.refresh_run_checksums(self.fixture.run_id)
        with self.assertRaisesRegex(EvidenceError, "recipe identity changed"):
            verify_evidence(self.fixture.output)

    def test_omitted_sglang_ple_source_requires_typed_exclusive_state(self) -> None:
        overlays = self.fixture.add_sglang_runtime_overlays()
        exact_overlay_digests = {
            "qwen4_exp.py": (
                "sha256:bcdc2c86aa59784ffe27d53c8d214e56"
                "b6aa45c02b1d5841fd956d1f006d6030"
            ),
            "qwen_sparse_attn_backend.py": (
                "sha256:e30566492e1502f94a4c7fed42d90b5"
                "23bbb662580c628459e6e63c7b5263c75"
            ),
        }
        for overlay in overlays:
            overlay["digest"] = exact_overlay_digests[
                Path(overlay["host_path"]).name
            ]
        plan_path = self.fixture.run_dir / "plan.json"
        original = json.loads(plan_path.read_text(encoding="utf-8"))
        original["model"].update(
            {
                "quantization": "nvfp4+ple-omitted-ablation+nextn-bf16",
                "recipe_revision": "bf2b7c75870d3703730b6bd8f3bb93dc622c278d",
                "recipe_source": "hashd1ve/qwen38-flash-next-one-dgx-spark",
                "revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
                "sglang_source_overlays": overlays,
                "source": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
            }
        )
        mutations = (
            lambda model: model.update(
                {"sglang_ple_mmap": True, "sglang_ple_omitted": True}
            ),
            lambda model: model.update(
                {"sglang_ple_mmap": False, "sglang_ple_omitted": "true"}
            ),
            lambda model: model.update(
                {
                    "sglang_ple_mmap": False,
                    "sglang_ple_omitted": True,
                    "sglang_source_overlays": [],
                }
            ),
            lambda model: model.update(
                {
                    "quantization": "nvfp4+ple-fp8-mapped+nextn-bf16",
                    "sglang_ple_mmap": False,
                    "sglang_ple_omitted": True,
                }
            ),
            lambda model: model.update(
                {
                    "sglang_ple_mmap": False,
                    "sglang_ple_omitted": True,
                    "source": "example/not-qwen38",
                }
            ),
            lambda model: model.update(
                {
                    "sglang_ple_mmap": False,
                    "sglang_ple_omitted": True,
                    "sglang_source_overlays": overlays[1:],
                }
            ),
        )
        self.assertEqual(2, len(overlays))
        for index, mutate in enumerate(mutations, 1):
            with self.subTest(index=index):
                plan = json.loads(json.dumps(original))
                mutate(plan["model"])
                self.fixture.write_json(plan_path, plan)
                output = Path(self.temporary.name) / f"evidence-omitted-invalid-{index}"
                with (
                    patch(
                        "bench.evidence._validate_runtime_overlay_tree",
                        return_value=True,
                    ),
                    self.assertRaises(EvidenceError),
                ):
                    self.export(output=output)

    def test_readonly_sglang_ple_cache_source_requires_exact_complete_pins(
        self,
    ) -> None:
        self.fixture.add_sglang_runtime_overlays(readonly_ple_cache=True)
        plan_path = self.fixture.run_dir / "plan.json"
        original = json.loads(plan_path.read_text(encoding="utf-8"))
        mutations = (
            (
                "missing_payload",
                lambda model: model.pop("sglang_ple_cache_payload_digest"),
            ),
            (
                "wrong_mode",
                lambda model: model.__setitem__(
                    "sglang_ple_cache_mode", "readwrite"
                ),
            ),
            (
                "mmap_disabled",
                lambda model: model.__setitem__("sglang_ple_mmap", False),
            ),
            (
                "invalid_marker",
                lambda model: model.__setitem__(
                    "sglang_ple_cache_marker_digest", "sha256:not-a-digest"
                ),
            ),
        )
        for index, (name, mutate) in enumerate(mutations, 1):
            with self.subTest(name=name):
                plan = json.loads(json.dumps(original))
                mutate(plan["model"])
                self.fixture.write_json(plan_path, plan)
                output = Path(self.temporary.name) / f"evidence-ple-invalid-{index}"
                with self.assertRaises(EvidenceError):
                    self.export(output=output)

    def test_runtime_overlay_tree_rejects_undeclared_or_digest_changed_files(
        self,
    ) -> None:
        self.fixture.add_sglang_runtime_overlays()
        overlay_dir = (
            self.fixture.results
            / "runtime-overlays"
            / "synthetic-sglang-recipe"
        )
        (overlay_dir / "undeclared.py").write_text(
            "UNDECLARED = True\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(EvidenceError, "file set changed"):
            self.export()

        (overlay_dir / "undeclared.py").unlink()
        (overlay_dir / "qwen4_exp.py").write_text(
            "MODEL_KIND = 'changed'\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(EvidenceError, "digest mismatch"):
            self.export()

    def test_typed_startup_safety_gates_export_as_sorted_scalars_only(self) -> None:
        gates = [
            {
                "metric": "startup_swap_growth",
                "observed": 518.25,
                "limit": 512.0,
                "unit": "mib",
                "comparison": "gt",
            },
            {
                "metric": "host_memavailable",
                "observed": 13.46,
                "limit": 14.0,
                "unit": "gib",
                "comparison": "lt",
            },
        ]
        self.add_typed_startup_safety_gates(gates)

        self.export()

        published = json.loads(
            (
                self.fixture.output
                / "runs"
                / self.fixture.run_id
                / "summary.json"
            ).read_text(encoding="utf-8")
        )["aggregates"]
        self.assertEqual(
            ["host_memavailable", "startup_swap_growth"],
            [gate["metric"] for gate in published["startup_safety_gates"]],
        )
        self.assertEqual(3, published["measurement_annotations_count"])
        self.assertEqual(2, published["startup_measurement_annotations_count"])
        self.assertNotIn("cold_start_safety_annotations", published)
        self.assertNotIn("reason", json_keys(published["startup_safety_gates"]))
        self.assertNotIn("evidence", json_keys(published["startup_safety_gates"]))
        self.assertNotIn("timestamp", json_keys(published["startup_safety_gates"]))
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

    def test_typed_startup_safety_gate_source_mirrors_fail_closed(self) -> None:
        gate = {
            "metric": "host_memavailable",
            "observed": 13.46,
            "limit": 14.0,
            "unit": "gib",
            "comparison": "lt",
        }
        self.add_typed_startup_safety_gates([gate], summary=False)
        with self.assertRaisesRegex(EvidenceError, "journal and summary"):
            self.export()

        summary_path = self.fixture.run_dir / "summary.json"
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        annotation = {
            "timestamp": "2026-08-27T00:00:01+00:00",
            "scope": "startup",
            "measurement_valid": False,
            "safety_gate": gate,
        }
        source_summary["measurement_annotations"].append(annotation)
        source_summary["startup_measurement_annotations"] = []
        source_summary["startup_safety_gates"] = [gate]
        self.fixture.write_json(summary_path, source_summary)
        with self.assertRaisesRegex(EvidenceError, "summary mirrors"):
            self.export()

    def test_typed_startup_safety_gate_requires_invalid_startup(self) -> None:
        gate = {
            "metric": "host_memavailable",
            "observed": 13.46,
            "limit": 14.0,
            "unit": "gib",
            "comparison": "lt",
        }
        self.add_typed_startup_safety_gates([gate])
        summary_path = self.fixture.run_dir / "summary.json"
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_summary["startup_measurement_valid"] = True
        self.fixture.write_json(summary_path, source_summary)
        with self.assertRaisesRegex(EvidenceError, "startup_measurement_valid=false"):
            self.export()

    def test_typed_startup_safety_gate_journal_identity_fails_closed(self) -> None:
        gate = {
            "metric": "host_memavailable",
            "observed": 13.46,
            "limit": 14.0,
            "unit": "gib",
            "comparison": "lt",
        }
        self.add_typed_startup_safety_gates([gate])
        summary_path = self.fixture.run_dir / "summary.json"
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_summary["measurement_annotations"][-1]["timestamp"] = (
            "2026-08-27T00:00:09+00:00"
        )
        source_summary["startup_measurement_annotations"][-1]["timestamp"] = (
            "2026-08-27T00:00:09+00:00"
        )
        self.fixture.write_json(summary_path, source_summary)
        with self.assertRaisesRegex(EvidenceError, "journal and summary"):
            self.export()

        source_summary["measurement_annotations"][-1]["timestamp"] = (
            "not-an-iso-timestamp"
        )
        source_summary["startup_measurement_annotations"][-1]["timestamp"] = (
            "not-an-iso-timestamp"
        )
        self.fixture.write_json(summary_path, source_summary)
        with self.assertRaisesRegex(EvidenceError, "timestamp"):
            self.export()

    def test_typed_startup_safety_gate_source_schema_fails_closed(self) -> None:
        gate = {
            "metric": "host_memavailable",
            "observed": 13.46,
            "limit": 14.0,
            "unit": "gib",
            "comparison": "lt",
        }
        self.add_typed_startup_safety_gates([gate])
        summary_path = self.fixture.run_dir / "summary.json"
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_summary["startup_safety_gates"][0]["unexpected"] = 1
        self.fixture.write_json(summary_path, source_summary)
        with self.assertRaisesRegex(EvidenceError, "exactly"):
            self.export()

        source_summary["startup_safety_gates"][0].pop("unexpected")
        source_summary["startup_safety_gates"].append(
            dict(source_summary["startup_safety_gates"][0])
        )
        self.fixture.write_json(summary_path, source_summary)
        with self.assertRaisesRegex(EvidenceError, "duplicate"):
            self.export()

    def test_published_startup_safety_gate_verification_fails_closed(self) -> None:
        gate = {
            "metric": "startup_swap_growth",
            "observed": 518.25,
            "limit": 512.0,
            "unit": "mib",
            "comparison": "gt",
        }
        self.add_typed_startup_safety_gates([gate])
        self.export()
        published_path = (
            self.fixture.output
            / "runs"
            / self.fixture.run_id
            / "summary.json"
        )

        published = json.loads(published_path.read_text(encoding="utf-8"))
        published["aggregates"]["startup_safety_gates"][0]["observed"] = 500.0
        self.fixture.write_json(published_path, published)
        self.refresh_run_checksums(self.fixture.run_id)
        with self.assertRaisesRegex(EvidenceError, "true breach"):
            verify_evidence(self.fixture.output)

        validity_output = Path(self.temporary.name) / "evidence-gate-validity"
        self.export(output=validity_output)
        published_path = (
            validity_output / "runs" / self.fixture.run_id / "summary.json"
        )
        published = json.loads(published_path.read_text(encoding="utf-8"))
        published["aggregates"]["startup_measurement_valid"] = True
        self.fixture.write_json(published_path, published)
        self.refresh_run_checksums(
            self.fixture.run_id,
            evidence_root=validity_output,
        )
        with self.assertRaisesRegex(EvidenceError, "startup_measurement_valid=false"):
            verify_evidence(validity_output)

        missing_output = Path(self.temporary.name) / "evidence-gate-missing"
        self.export(output=missing_output)
        published_path = (
            missing_output / "runs" / self.fixture.run_id / "summary.json"
        )
        published = json.loads(published_path.read_text(encoding="utf-8"))
        published["aggregates"].pop("startup_safety_gates")
        self.fixture.write_json(published_path, published)
        self.refresh_run_checksums(
            self.fixture.run_id,
            evidence_root=missing_output,
        )
        with self.assertRaisesRegex(EvidenceError, "safety gates are missing"):
            verify_evidence(missing_output)

        counts_output = Path(self.temporary.name) / "evidence-gate-counts"
        self.export(output=counts_output)
        published_path = (
            counts_output / "runs" / self.fixture.run_id / "summary.json"
        )
        published = json.loads(published_path.read_text(encoding="utf-8"))
        published["aggregates"]["startup_measurement_annotations_count"] = 0
        self.fixture.write_json(published_path, published)
        self.refresh_run_checksums(
            self.fixture.run_id,
            evidence_root=counts_output,
        )
        with self.assertRaisesRegex(EvidenceError, "annotation counts"):
            verify_evidence(counts_output)

    def test_typed_and_legacy_startup_swap_gates_must_agree(self) -> None:
        gate = {
            "metric": "startup_swap_growth",
            "observed": 602.48,
            "limit": 512.0,
            "unit": "mib",
            "comparison": "gt",
        }
        self.add_typed_startup_safety_gates([gate])
        summary_path = self.fixture.run_dir / "summary.json"
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        legacy = {
            "evidence": [
                "swap_growth_mib=700",
                "safety_limit_mib=512",
                "memavailable_gib=15",
                "memory_psi_full_avg10=2",
            ],
            "measurement_valid": False,
            "reason": "cold_start_swap_growth_exceeded_safety_limit",
            "scope": "startup",
            "timestamp": "2026-08-27T00:00:02+00:00",
        }
        source_summary["measurement_annotations"].append(legacy)
        source_summary["startup_measurement_annotations"].append(legacy)
        self.fixture.write_json(summary_path, source_summary)
        with self.assertRaisesRegex(EvidenceError, "typed and legacy"):
            self.export()

        legacy["evidence"][0] = "swap_growth_mib=602.48"
        self.fixture.write_json(summary_path, source_summary)
        self.export()
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

    def test_cold_start_safety_annotations_have_exact_typed_projection(self) -> None:
        annotations = {
            "cold_start_swap_growth_exceeded_safety_limit": (
                [
                    "swap_growth_mib=3315.9",
                    "safety_limit_mib=512",
                    "memavailable_gib=31.17",
                    "memory_psi_full_avg10=2.23",
                ],
                {
                    "reason": "cold_start_swap_growth_exceeded_safety_limit",
                    "swap_growth_mib": 3315.9,
                    "safety_limit_mib": 512.0,
                    "memavailable_gib": 31.17,
                    "memory_psi_full_avg10": 2.23,
                },
            ),
            "ple_materialization_swap_growth_exceeded_safety_limit": (
                [
                    "swap_growth_mib=4096.5",
                    "safety_limit_mib=512",
                    "memavailable_gib=27.25",
                    "memory_psi_full_avg10=3.5",
                    "ple_allocated_blocks=47",
                ],
                {
                    "reason": (
                        "ple_materialization_swap_growth_exceeded_safety_limit"
                    ),
                    "swap_growth_mib": 4096.5,
                    "safety_limit_mib": 512.0,
                    "memavailable_gib": 27.25,
                    "memory_psi_full_avg10": 3.5,
                    "ple_allocated_blocks": 47,
                },
            ),
        }
        summary_path = self.fixture.run_dir / "summary.json"
        for index, (reason, (evidence, expected)) in enumerate(
            annotations.items(), 1
        ):
            source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            annotation = {
                "evidence": evidence,
                "measurement_valid": False,
                "reason": reason,
                "scope": "startup",
                "timestamp": "2026-08-27T00:35:40.854+00:00",
            }
            source_summary["measurement_annotations"] = [annotation]
            source_summary["startup_measurement_annotations"] = [annotation]
            source_summary["startup_measurement_valid"] = False
            self.fixture.write_json(summary_path, source_summary)
            output = Path(self.temporary.name) / f"evidence-cold-start-{index}"

            self.export(output=output)

            published = json.loads(
                (
                    output
                    / "runs"
                    / self.fixture.run_id
                    / "summary.json"
                ).read_text(encoding="utf-8")
            )["aggregates"]
            self.assertEqual([expected], published["cold_start_safety_annotations"])
            self.assertEqual(1, published["measurement_annotations_count"])
            self.assertEqual(1, published["startup_measurement_annotations_count"])
            self.assertNotIn("evidence", json_keys(published))
            self.assertNotIn("timestamp", json_keys(published))
            self.assertEqual("verified", verify_evidence(output)["status"])

    def test_cold_start_safety_annotation_source_and_output_fail_closed(self) -> None:
        summary_path = self.fixture.run_dir / "summary.json"
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        annotation = {
            "evidence": [
                "swap_growth_mib=3315.9",
                "safety_limit_mib=512",
                "memavailable_gib=31.17",
                "memory_psi_full_avg10=2.23",
                "unexpected_scalar=1",
            ],
            "measurement_valid": False,
            "reason": "cold_start_swap_growth_exceeded_safety_limit",
            "scope": "startup",
            "timestamp": "2026-08-27T00:35:40.854+00:00",
        }
        source_summary["measurement_annotations"] = [annotation]
        source_summary["startup_measurement_annotations"] = [annotation]
        self.fixture.write_json(summary_path, source_summary)
        with self.assertRaisesRegex(EvidenceError, "evidence changed"):
            self.export()

        annotation["evidence"].pop()
        self.fixture.write_json(summary_path, source_summary)
        self.export()
        published_path = (
            self.fixture.output
            / "runs"
            / self.fixture.run_id
            / "summary.json"
        )
        published = json.loads(published_path.read_text(encoding="utf-8"))
        published["aggregates"]["cold_start_safety_annotations"][0][
            "swap_growth_mib"
        ] = "3315.9"
        self.fixture.write_json(published_path, published)
        self.refresh_run_checksums(self.fixture.run_id)
        with self.assertRaisesRegex(EvidenceError, "finite float"):
            verify_evidence(self.fixture.output)

    def test_published_ple_allocated_blocks_range_fails_closed(self) -> None:
        summary_path = self.fixture.run_dir / "summary.json"
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        annotation = {
            "evidence": [
                "swap_growth_mib=4096.5",
                "safety_limit_mib=512",
                "memavailable_gib=27.25",
                "memory_psi_full_avg10=3.5",
                "ple_allocated_blocks=47",
            ],
            "measurement_valid": False,
            "reason": "ple_materialization_swap_growth_exceeded_safety_limit",
            "scope": "startup",
            "timestamp": "2026-08-27T00:35:40.854+00:00",
        }
        source_summary["measurement_annotations"] = [annotation]
        source_summary["startup_measurement_annotations"] = [annotation]
        source_summary["startup_measurement_valid"] = False
        self.fixture.write_json(summary_path, source_summary)
        self.export()
        published_path = (
            self.fixture.output
            / "runs"
            / self.fixture.run_id
            / "summary.json"
        )

        for invalid in (-1, 2**63):
            with self.subTest(ple_allocated_blocks=invalid):
                published = json.loads(
                    published_path.read_text(encoding="utf-8")
                )
                published["aggregates"]["cold_start_safety_annotations"][0][
                    "ple_allocated_blocks"
                ] = invalid
                self.fixture.write_json(published_path, published)
                self.refresh_run_checksums(self.fixture.run_id)
                with self.assertRaisesRegex(EvidenceError, "supported range"):
                    verify_evidence(self.fixture.output)

    def test_memory_protocol_exports_exact_scalar_evidence_deterministically(
        self,
    ) -> None:
        run_id, _ = self.fixture.write_memory_run()
        self.export()
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

        bundle = self.fixture.output / "runs" / run_id
        samples_document = json.loads(
            (bundle / "samples.json").read_text(encoding="utf-8")
        )
        samples = samples_document["samples"]
        self.assertEqual(33, samples_document["sample_count"])
        self.assertEqual(list(range(1, 34)), [sample["sample_index"] for sample in samples])
        self.assertTrue(all(sample["case_attempt"] == 1 for sample in samples))
        self.assertTrue(all(sample["selected_attempt"] is True for sample in samples))
        self.assertTrue(all(sample["reasoning_tokens"] is None for sample in samples))
        self.assertEqual(
            [
                (scenario_id, variant)
                for scenario_id in MEMORY_OPERATION_SCENARIO_IDS
                for variant in range(3)
            ],
            [(sample["scenario_id"], sample["variant"]) for sample in samples],
        )
        summary = json.loads(
            (bundle / "summary.json").read_text(encoding="utf-8")
        )["aggregates"]
        self.assertEqual(11, len(summary["cases"]))
        self.assertEqual(33, summary["memory_operation_summary"]["operations"])
        self.assertIsNone(
            summary["memory_operation_summary"]["total_reasoning_tokens"]
        )
        self.assertEqual(
            9,
            summary["memory_operation_summary"]["graphiti_resolver"][
                "operations"
            ],
        )
        self.assertEqual(
            24,
            summary["memory_operation_summary"]["synthetic_extension"][
                "operations"
            ],
        )
        manifest = json.loads(
            (bundle / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertFalse(manifest["model"]["memory_thinking_enabled"])
        self.assertEqual("memory-operations", manifest["suite"]["id"])
        self.assertEqual(
            MEMORY_OPERATION_PROTOCOL_DIGEST,
            manifest["suite"]["protocol_digest"],
        )

        serialized = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(bundle.glob("*.json"))
        )
        for forbidden_value in (
            RAW_COMPLETION,
            RAW_REASONING,
            RAW_REQUEST_ID,
            RAW_HOST_PATH,
            RAW_SECRET,
            "private-memory-attempt",
        ):
            self.assertNotIn(forbidden_value, serialized)
        keys = set()
        for path in bundle.glob("*.json"):
            keys.update(json_keys(json.loads(path.read_text(encoding="utf-8"))))
        for forbidden_key in (
            "content",
            "nonce",
            "path",
            "reasoning",
            "request_id",
            "request_tag",
            "tool_calls",
            "value",
        ):
            self.assertNotIn(forbidden_key, keys)

        second_output = Path(self.temporary.name) / "evidence-memory-second"
        self.export(output=second_output)
        first_files = {
            str(path.relative_to(self.fixture.output)): path.read_bytes()
            for path in self.fixture.output.rglob("*")
            if path.is_file()
        }
        second_files = {
            str(path.relative_to(second_output)): path.read_bytes()
            for path in second_output.rglob("*")
            if path.is_file()
        }
        self.assertEqual(first_files, second_files)
        self.assertEqual("verified", verify_evidence(second_output)["status"])

    def test_memory_semantic_tampering_fails_after_checksum_refresh(self) -> None:
        run_id, _ = self.fixture.write_memory_run()
        self.export()
        samples_path = self.fixture.output / "runs" / run_id / "samples.json"
        document = json.loads(samples_path.read_text(encoding="utf-8"))
        document["samples"][0]["prompt_tokens"] += 1
        document["samples"][0]["server_prompt_tokens"] += 1
        EvidenceFixture.write_json(samples_path, document)
        self.refresh_run_checksums(run_id)

        with self.assertRaisesRegex(EvidenceError, "aggregate disagrees"):
            verify_evidence(self.fixture.output)

    def test_imperfect_complete_memory_run_exports_and_failure_list_is_bound(
        self,
    ) -> None:
        run_id, _ = self.fixture.write_memory_run(imperfect=True)
        self.export()
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])
        bundle = self.fixture.output / "runs" / run_id
        summary_path = bundle / "summary.json"
        summary_document = json.loads(summary_path.read_text(encoding="utf-8"))
        aggregates = summary_document["aggregates"]
        failed_case_id = next(
            case["case_id"]
            for case in aggregates["cases"]
            if case["validation_passed"] is False
        )
        self.assertEqual("partial", aggregates["status"])
        self.assertEqual([failed_case_id], aggregates["validation_failed_cases"])
        self.assertEqual(
            32,
            aggregates["memory_operation_summary"]["operations_correct"],
        )
        manifest = json.loads(
            (bundle / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual("partial", manifest["status"])

        aggregates["validation_failed_cases"] = []
        EvidenceFixture.write_json(summary_path, summary_document)
        self.refresh_run_checksums(run_id)
        with self.assertRaisesRegex(EvidenceError, "validation failures disagree"):
            verify_evidence(self.fixture.output)

    def test_memory_exact_bundle_rejects_refreshed_checksum_tampering(self) -> None:
        run_id, _ = self.fixture.write_memory_run()

        def exported(label: str) -> tuple[Path, Path]:
            root = Path(self.temporary.name) / f"memory-tamper-{label}"
            self.export(output=root)
            return root, root / "runs" / run_id

        mutations = {
            "extra-manifest-key": lambda bundle: self._mutate_json(
                bundle / "manifest.json",
                lambda value: value.__setitem__("attacker_note", 1),
            ),
            "lifecycle-count": lambda bundle: self._mutate_json(
                bundle / "manifest.json",
                lambda value: value["lifecycle"]["event_counts"].__setitem__(
                    "first_request_complete", 999
                ),
            ),
            "quantization": lambda bundle: self._mutate_json(
                bundle / "manifest.json",
                lambda value: value["model"].__setitem__("quantization", "fp8"),
            ),
            "thinking-policy": lambda bundle: self._mutate_json(
                bundle / "manifest.json",
                lambda value: (
                    value["model"].__setitem__("memory_thinking_enabled", True),
                    value["model"]["tasks"].append("thinking"),
                ),
            ),
            "protocol-digest": lambda bundle: self._mutate_json(
                bundle / "manifest.json",
                lambda value: value["suite"].__setitem__(
                    "protocol_digest", "sha256:" + "0" * 64
                ),
            ),
            "extra-summary-key": lambda bundle: self._mutate_json(
                bundle / "summary.json",
                lambda value: value["aggregates"].__setitem__("attacker_note", 1),
            ),
            "first-request-summary": lambda bundle: self._mutate_json(
                bundle / "summary.json",
                lambda value: value["aggregates"].__setitem__(
                    "first_request", {"completion_tokens": 777}
                ),
            ),
            "reasoning-availability": lambda bundle: self._mutate_json(
                bundle / "samples.json",
                lambda value: value["samples"][0].__setitem__("reasoning_tokens", 0),
            ),
            "telemetry": lambda bundle: self._mutate_json(
                bundle / "telemetry.json",
                lambda value: value.__setitem__("sample_count", 1),
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                root, bundle = exported(label)
                mutate(bundle)
                self.refresh_run_checksums(run_id, evidence_root=root)
                with self.assertRaises(EvidenceError):
                    verify_evidence(root)

        root, bundle = exported("extra-file")
        EvidenceFixture.write_json(bundle / "appendix.json", {"safe": 1})
        self.refresh_run_checksums(run_id, evidence_root=root)
        with self.assertRaisesRegex(EvidenceError, "file set"):
            verify_evidence(root)

    def test_imperfect_memory_status_cannot_be_relabelled_complete(self) -> None:
        run_id, _ = self.fixture.write_memory_run(imperfect=True)
        self.export()
        bundle = self.fixture.output / "runs" / run_id
        self._mutate_json(
            bundle / "manifest.json",
            lambda value: value.__setitem__("status", "complete"),
        )
        self._mutate_json(
            bundle / "summary.json",
            lambda value: value["aggregates"].__setitem__("status", "complete"),
        )
        index_path = self.fixture.output / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        next(entry for entry in index["runs"] if entry["run_id"] == run_id)[
            "status"
        ] = "complete"
        EvidenceFixture.write_json(index_path, index)
        self.refresh_run_checksums(run_id)
        with self.assertRaisesRegex(EvidenceError, "status changed"):
            verify_evidence(self.fixture.output)

    def test_memory_source_journal_and_summary_topology_are_exact(self) -> None:
        _run_id, run_dir = self.fixture.write_memory_run()
        events_path = run_dir / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        prime = next(
            event for event in events if event["event"] == "first_request_complete"
        )
        events.insert(events.index(prime) + 1, dict(prime))
        EvidenceFixture.write_jsonl(events_path, events)
        with self.assertRaisesRegex(EvidenceError, "protocol"):
            self.export()

    def test_memory_source_rejects_requestless_retry_and_missing_model(self) -> None:
        _run_id, run_dir = self.fixture.write_memory_run()
        events_path = run_dir / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        first_start = next(event for event in events if event["event"] == "case_start")
        retry = {**first_start, "attempt_id": "private-empty-retry"}
        events.insert(events.index(first_start) + 1, retry)
        EvidenceFixture.write_jsonl(events_path, events)
        with self.assertRaisesRegex(EvidenceError, "protocol"):
            self.export()

        EvidenceFixture.write_jsonl(
            events_path, [event for event in events if event is not retry]
        )
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.pop("model")
        EvidenceFixture.write_json(summary_path, summary)
        with self.assertRaisesRegex(EvidenceError, "frozen model identity"):
            self.export()

    def test_memory_source_rejects_resumed_journal(self) -> None:
        _run_id, run_dir = self.fixture.write_memory_run()
        events_path = run_dir / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        run_start = next(event for event in events if event["event"] == "run_start")
        run_start["completed_cases_at_resume"] = ["prior-case--000000000000"]
        EvidenceFixture.write_jsonl(events_path, events)
        with self.assertRaisesRegex(EvidenceError, "resumed run"):
            self.export()

    def test_memory_source_binds_artifact_admission_event(self) -> None:
        _run_id, run_dir = self.fixture.write_memory_run()
        events_path = run_dir / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        admission = next(
            event
            for event in events
            if event["event"] == "artifact_validation_complete"
        )
        admission["model_sha256"] = "sha256:" + "e" * 64
        EvidenceFixture.write_jsonl(events_path, events)
        with self.assertRaisesRegex(EvidenceError, "frozen pins"):
            self.export()

    def test_memory_source_binds_prime_result(self) -> None:
        _run_id, run_dir = self.fixture.write_memory_run()
        events_path = run_dir / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        prime = next(
            event for event in events if event["event"] == "first_request_complete"
        )
        prime["result"]["completion_tokens"] = 777
        EvidenceFixture.write_jsonl(events_path, events)
        with self.assertRaisesRegex(EvidenceError, "prime counters"):
            self.export()

    def test_memory_source_binds_case_complete_outcome_and_elapsed(self) -> None:
        _run_id, run_dir = self.fixture.write_memory_run()
        events_path = run_dir / "events.jsonl"
        original = [json.loads(line) for line in events_path.read_text().splitlines()]
        complete = next(
            event for event in original if event["event"] == "case_complete"
        )
        complete["validation_passed"] = False
        EvidenceFixture.write_jsonl(events_path, original)
        with self.assertRaisesRegex(EvidenceError, "outcome changed"):
            self.export()

        complete["validation_passed"] = True
        complete["elapsed_s"] = 999.0
        EvidenceFixture.write_jsonl(events_path, original)
        with self.assertRaisesRegex(EvidenceError, "outcome changed"):
            self.export()

    @staticmethod
    def _mutate_json(
        path: Path, mutate: Callable[[dict[str, object]], object]
    ) -> None:
        value = json.loads(path.read_text(encoding="utf-8"))
        mutate(value)
        EvidenceFixture.write_json(path, value)

    def test_allowlisted_grouped_run_roots_are_deterministic_and_verifiable(self) -> None:
        grouped_roots = (
            "qwen36-core-20260818T113704",
            "reasoning-20260818",
        )
        top_level_run = self.fixture.results / self.fixture.run_id
        for group_name in grouped_roots:
            with self.subTest(group_name=group_name):
                group = self.move_run_to_group(group_name)
                output = Path(self.temporary.name) / f"evidence-{group_name}"

                first = self.export(output=output)
                original = {
                    str(path.relative_to(output)): path.read_bytes()
                    for path in output.rglob("*")
                    if path.is_file()
                }
                second = self.export(output=output)

                self.assertTrue(first["changed"])
                self.assertFalse(second["changed"])
                self.assertEqual(
                    original,
                    {
                        str(path.relative_to(output)): path.read_bytes()
                        for path in output.rglob("*")
                        if path.is_file()
                    },
                )
                self.assertEqual("verified", verify_evidence(output)["status"])
                index = json.loads((output / "index.json").read_text(encoding="utf-8"))
                self.assertEqual([self.fixture.run_id], [run["run_id"] for run in index["runs"]])

                self.fixture.run_dir.rename(top_level_run)
                group.rmdir()
                self.fixture.run_dir = top_level_run

    def test_grouped_run_roots_reject_unpinned_or_nonrun_topology(self) -> None:
        unknown_group = self.move_run_to_group("reasoning-unpinned")
        with self.assertRaisesRegex(EvidenceError, "unknown layout"):
            self.export()
        self.fixture.run_dir.rename(self.fixture.results / self.fixture.run_id)
        unknown_group.rmdir()
        self.fixture.run_dir = self.fixture.results / self.fixture.run_id

        group = self.fixture.results / "reasoning-20260818"
        group.mkdir()
        (group / "private-raw-note.txt").write_text(RAW_SECRET, encoding="utf-8")
        with self.assertRaisesRegex(
            EvidenceError, "only direct run directories"
        ):
            self.export()

    def test_autoresearch_cells_export_unique_deterministic_scalar_runs(self) -> None:
        campaign_dir, _run_dirs = self.fixture.write_autoresearch_campaign()

        first = self.export()
        original = self.exported_bytes()
        second = self.export()

        self.assertTrue(first["changed"])
        self.assertEqual(14, first["runs"])
        self.assertFalse(second["changed"])
        self.assertEqual(original, self.exported_bytes())
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

        index = json.loads(
            (self.fixture.output / "index.json").read_text(encoding="utf-8")
        )
        published_ids = [entry["run_id"] for entry in index["runs"]]
        self.assertEqual(14, len(published_ids))
        self.assertEqual(14, len(set(published_ids)))
        self.assertTrue(
            all("-autoresearch-synthetic-autoresearch-" in run_id for run_id in published_ids)
        )
        self.assertNotIn(self.fixture.run_id, published_ids)
        for run_id in published_ids:
            manifest = json.loads(
                (
                    self.fixture.output / "runs" / run_id / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(run_id, manifest["source_run_id"])
            self.assertFalse(manifest["sanitization"]["raw_identifiers_included"])

        serialized = b"\n".join(original.values()).decode("utf-8")
        for private_value in (
            self.fixture.run_id,
            str(campaign_dir),
            str(self.fixture.results),
            RAW_HOST_PATH,
            RAW_SECRET,
            RAW_REQUEST_ID,
            f"{1:032x}",
        ):
            with self.subTest(private_value=private_value):
                self.assertNotIn(private_value, serialized)

    def test_autoresearch_mixed_nine_case_suite_exports_and_verifies(self) -> None:
        self.fixture.write_autoresearch_campaign(mixed_suite=True)

        result = self.export()

        self.assertEqual(14, result["runs"])
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])
        index = json.loads(
            (self.fixture.output / "index.json").read_text(encoding="utf-8")
        )
        first_run_id = index["runs"][0]["run_id"]
        manifest = json.loads(
            (
                self.fixture.output / "runs" / first_run_id / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            "qwen38-flash-next-sglang-agent64k-autoresearch",
            manifest["suite"]["id"],
        )
        self.assertEqual(9, len(manifest["suite"]["cases"]))
        self.assertEqual(
            4,
            sum(
                case["kind"] == "agentic"
                for case in manifest["suite"]["cases"]
            ),
        )

    def test_autoresearch_campaign_rejects_rebound_plan_hashes(self) -> None:
        campaign_dir, _run_dirs = self.fixture.write_autoresearch_campaign()
        campaign_path = campaign_dir / "campaign.json"
        original = json.loads(campaign_path.read_text(encoding="utf-8"))

        for field, replacement in (
            ("plan_fingerprint", "f" * 16),
            ("plan_integrity_hash", "e" * 64),
        ):
            with self.subTest(field=field):
                campaign = json.loads(json.dumps(original))
                campaign["cells"][0][field] = replacement
                campaign.pop("integrity_hash")
                campaign["integrity_hash"] = _synthetic_content_hash(campaign)
                self.fixture.write_json(campaign_path, campaign)

                with self.assertRaisesRegex(
                    EvidenceError, "autoresearch cell plan binding changed"
                ):
                    self.export()

    def test_autoresearch_campaign_rejects_duplicate_bound_nonces(self) -> None:
        campaign_dir, run_dirs = self.fixture.write_autoresearch_campaign()
        campaign_path = campaign_dir / "campaign.json"
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        duplicate_nonce = campaign["cells"][0]["run_nonce"]
        second_plan_path = run_dirs[1] / "plan.json"
        second_plan = json.loads(second_plan_path.read_text(encoding="utf-8"))
        second_plan["run_nonce"] = duplicate_nonce
        second_plan.pop("integrity_hash")
        second_plan["integrity_hash"] = _synthetic_content_hash(second_plan)
        self.fixture.write_json(second_plan_path, second_plan)
        campaign["cells"][1]["run_nonce"] = duplicate_nonce
        campaign["cells"][1]["plan_integrity_hash"] = second_plan["integrity_hash"]
        campaign.pop("integrity_hash")
        campaign["integrity_hash"] = _synthetic_content_hash(campaign)
        self.fixture.write_json(campaign_path, campaign)

        with self.assertRaisesRegex(
            EvidenceError, "autoresearch cell ownership nonces are duplicated"
        ):
            self.export()

    def test_autoresearch_campaign_rejects_undeclared_nested_plan(self) -> None:
        _campaign_dir, run_dirs = self.fixture.write_autoresearch_campaign()
        undeclared = run_dirs[0] / "private-controller-state"
        undeclared.mkdir()
        shutil.copyfile(run_dirs[0] / "plan.json", undeclared / "plan.json")

        with self.assertRaisesRegex(
            EvidenceError, "autoresearch campaign contains an undeclared plan"
        ):
            self.export()

    def test_autoresearch_campaign_rejects_cell_directory_symlink(self) -> None:
        campaign_dir, _run_dirs = self.fixture.write_autoresearch_campaign()
        cells_root = campaign_dir / "cells"
        first_cell = sorted(cells_root.iterdir())[0]
        (cells_root / "99-linked-cell").symlink_to(
            first_cell, target_is_directory=True
        )

        with self.assertRaisesRegex(EvidenceError, "directory symlink"):
            self.export()

    def test_source_ordinals_and_columnar_telemetry_are_preserved(self) -> None:
        self.export()
        run = self.fixture.output / "runs" / self.fixture.run_id
        samples = json.loads((run / "samples.json").read_text(encoding="utf-8"))[
            "samples"
        ]
        self.assertEqual([1, 2, 3, 4], [sample["sample_index"] for sample in samples])
        self.assertEqual(
            ["first_request", "measured_request", "measured_request", "measured_request"],
            [sample["sample_type"] for sample in samples],
        )
        self.assertEqual([1, 1, 2], [sample["case_attempt"] for sample in samples[1:]])
        self.assertEqual(
            [1, 2, 1], [sample["case_sample_index"] for sample in samples[1:]]
        )
        self.assertEqual(
            [False, False, True], [sample["selected_attempt"] for sample in samples[1:]]
        )

        metadata = json.loads((run / "telemetry.json").read_text(encoding="utf-8"))
        self.assertEqual(3, metadata["sample_count"])
        self.assertEqual(2, metadata["segment_count"])
        self.assertEqual(["telemetry-0001.json"], metadata["chunks"])
        chunk = json.loads(
            (run / metadata["chunks"][0]).read_text(encoding="utf-8")
        )
        self.assertEqual(["case:chat-case", "idle"], [
            segment["phase"] for segment in chunk["segments"]
        ])
        self.assertEqual([2, 1], [len(segment["rows"]) for segment in chunk["segments"]])
        elapsed_index = metadata["columns"].index("elapsed_s")
        error_index = metadata["columns"].index("gpu_error_present")
        memfree_index = metadata["columns"].index("memfree_bytes")
        first_rows = chunk["segments"][0]["rows"]
        self.assertEqual([0.0, 0.5], [row[elapsed_index] for row in first_rows])
        self.assertEqual([False, True], [row[error_index] for row in first_rows])
        self.assertEqual([10 * 1024, 9 * 1024], [row[memfree_index] for row in first_rows])
        self.assertEqual(0.0, chunk["segments"][1]["rows"][0][elapsed_index])

    def test_harbor_reversed_inputs_are_deterministic_and_staged_safe(self) -> None:
        self.add_matrix_source()
        first, second = self.harbor_results()
        initial = self.export(harbor_results=(second, first))
        self.assertTrue(initial["changed"])
        original = self.exported_bytes()

        repeated = self.export(harbor_results=(first, second))

        self.assertFalse(repeated["changed"])
        self.assertEqual(original, self.exported_bytes())
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])
        bundle = self.fixture.output / "campaigns" / HARBOR_CAMPAIGN_ID
        self.assertEqual(
            {"checksums.json", "manifest.json", "replicates.json"},
            {path.name for path in bundle.iterdir()},
        )
        document = json.loads((bundle / "replicates.json").read_text())
        manifest = json.loads((bundle / "manifest.json").read_text())
        self.assertEqual(HARBOR_SCHEMA_VERSION, manifest["schema_version"])
        self.assertNotEqual(SCHEMA_VERSION, manifest["schema_version"])
        self.assertEqual(2, document["replicate_count"])
        self.assertEqual(
            [
                "2026-08-18T02:00:00+00:00",
                "2026-08-18T03:00:00+00:00",
            ],
            [replicate["started_at"] for replicate in document["replicates"]],
        )
        serialized = (bundle / "replicates.json").read_text(encoding="utf-8")
        self.assertNotIn(str(first.parent), serialized)
        self.assertNotIn("synthetic-private", serialized)

        repository = Path(self.temporary.name)
        self.git("init", "--quiet")
        self.git("add", "--", self.fixture.output.name)
        staged = verify_staged_evidence(
            repo_root=repository,
            evidence_root=Path(self.fixture.output.name),
        )
        self.assertEqual("staged_verified", staged["status"])

    def test_harbor_bundle_is_preserved_without_private_inputs(self) -> None:
        self.add_matrix_source()
        first, second = self.harbor_results()
        self.export(harbor_results=(first, second))
        bundle = self.fixture.output / "campaigns" / HARBOR_CAMPAIGN_ID
        original = {
            path.name: path.read_bytes()
            for path in bundle.iterdir()
        }

        self.fixture.change_aggregate(13.5)
        refreshed = self.export(replace=True)

        self.assertTrue(refreshed["changed"])
        self.assertEqual(
            original,
            {path.name: path.read_bytes() for path in bundle.iterdir()},
        )
        index = json.loads(
            (self.fixture.output / "index.json").read_text(encoding="utf-8")
        )
        self.assertIn(
            HARBOR_CAMPAIGN_ID,
            {campaign["campaign_id"] for campaign in index["campaigns"]},
        )
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

    def test_harbor_carry_forward_revalidates_existing_bundle(self) -> None:
        first, second = self.harbor_results()
        self.export(harbor_results=(first, second))
        bundle = self.fixture.output / "campaigns" / HARBOR_CAMPAIGN_ID
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "partial"
        EvidenceFixture.write_json(manifest_path, manifest)
        self.refresh_campaign_checksums(HARBOR_CAMPAIGN_ID)

        with self.assertRaisesRegex(EvidenceError, "Harbor evidence manifest changed"):
            self.export(replace=True)

    def test_harbor_input_count_duplicates_and_pin_mismatch_fail_closed(self) -> None:
        first, second = self.harbor_results()
        with self.assertRaisesRegex(EvidenceError, "zero or exactly two"):
            self.export(harbor_results=(first,))
        with self.assertRaisesRegex(EvidenceError, "zero or exactly two"):
            self.export(harbor_results=(first, second, first))
        with self.assertRaisesRegex(EvidenceError, "duplicates"):
            self.export(harbor_results=(first, first))

        mismatched_first, mismatched_second = self.harbor_results(
            second_size_offset=10
        )
        with self.assertRaisesRegex(EvidenceError, "exact campaign pins"):
            self.export(harbor_results=(mismatched_first, mismatched_second))

        parsed = build_parser().parse_args(
            [
                "export-evidence",
                "--harbor-result",
                str(first),
                "--harbor-result",
                str(second),
            ]
        )
        self.assertEqual([first, second], parsed.harbor_result)

    def test_harbor_unsafe_malformed_partial_and_sensitive_inputs_fail(self) -> None:
        first, second = self.harbor_results()
        first.chmod(0o644)
        with self.assertRaisesRegex(EvidenceError, "owner-mode-0600"):
            self.export(harbor_results=(first, second))

        first.unlink()
        _write_private_json(
            first,
            _harbor_envelope(
                started_at="2026-08-18T02:00:00+00:00",
                finished_at="2026-08-18T02:00:20+00:00",
            ),
        )
        hardlink = Path(self.temporary.name) / "harbor-hardlink.json"
        os.link(first, hardlink)
        with self.assertRaisesRegex(EvidenceError, "single-link"):
            self.export(harbor_results=(hardlink, second))
        hardlink.unlink()
        symlink = Path(self.temporary.name) / "harbor-symlink.json"
        symlink.symlink_to(first)
        with self.assertRaisesRegex(EvidenceError, "symbolic link"):
            self.export(harbor_results=(symlink, second))

        first.unlink()
        _write_private_json(
            first,
            _harbor_envelope(
                started_at="2026-08-18T02:00:00+00:00",
                finished_at="2026-08-18T02:00:20+00:00",
            ),
            canonical=False,
        )
        with self.assertRaisesRegex(EvidenceError, "not canonical"):
            self.export(harbor_results=(first, second))

        partial = _harbor_envelope(
            started_at="2026-08-18T02:00:00+00:00",
            finished_at="2026-08-18T02:00:20+00:00",
        )
        partial["status"] = "partial"
        _write_private_json(first, partial)
        with self.assertRaisesRegex(EvidenceError, "exact validation"):
            self.export(harbor_results=(first, second))

        sensitive = _harbor_envelope(
            started_at="2026-08-18T02:00:00+00:00",
            finished_at="2026-08-18T02:00:20+00:00",
        )
        sensitive["campaign"]["trials"][0]["prompt"] = RAW_SECRET
        _write_private_json(first, sensitive)
        with self.assertRaises(EvidenceError):
            self.export(harbor_results=(first, second))

        path_value = _harbor_envelope(
            started_at="2026-08-18T02:00:00+00:00",
            finished_at="2026-08-18T02:00:20+00:00",
        )
        path_value["campaign"]["trials"][0]["safe_metric"] = RAW_HOST_PATH
        _write_private_json(first, path_value)
        with self.assertRaisesRegex(EvidenceError, "scalar publication policy"):
            self.export(harbor_results=(first, second))

        infrastructure_failure = _harbor_envelope(
            started_at="2026-08-18T02:00:00+00:00",
            finished_at="2026-08-18T02:00:20+00:00",
        )
        infrastructure_failure["campaign"]["trials"][2]["public_rejected"] = False
        infrastructure_failure["campaign"]["summary"][
            "network_admission_failures"
        ] = 1
        _write_private_json(first, infrastructure_failure)
        with self.assertRaisesRegex(EvidenceError, "infrastructure gates"):
            self.export(harbor_results=(first, second))

        missing_reward = _harbor_envelope(
            started_at="2026-08-18T02:00:00+00:00",
            finished_at="2026-08-18T02:00:20+00:00",
        )
        missing_reward["campaign"]["trials"][2]["reward"] = None
        _write_private_json(first, missing_reward)
        with self.assertRaisesRegex(EvidenceError, "trial infrastructure gates"):
            self.export(harbor_results=(first, second))

        first.write_text('{"value":1,"value":2}\n', encoding="ascii")
        first.chmod(0o600)
        with self.assertRaisesRegex(EvidenceError, "strict JSON grammar"):
            self.export(harbor_results=(first, second))

        for location in ("model", "pins"):
            type_drift = _harbor_envelope(
                started_at="2026-08-18T02:00:00+00:00",
                finished_at="2026-08-18T02:00:20+00:00",
            )
            if location == "model":
                type_drift["model"]["runtime_parallel"] = True
            else:
                type_drift["campaign"]["pins"]["model"]["parallel"] = True
            _write_private_json(first, type_drift)
            with self.subTest(type_drift=location), self.assertRaises(EvidenceError):
                self.export(harbor_results=(first, second))

    def test_harbor_verifier_rechecks_order_after_checksum_rewrite(self) -> None:
        first, second = self.harbor_results()
        self.export(harbor_results=(first, second))
        bundle = self.fixture.output / "campaigns" / HARBOR_CAMPAIGN_ID
        replicates_path = bundle / "replicates.json"
        document = json.loads(replicates_path.read_text(encoding="utf-8"))
        document["replicates"].reverse()
        EvidenceFixture.write_json(replicates_path, document)
        self.refresh_campaign_checksums(HARBOR_CAMPAIGN_ID)

        with self.assertRaisesRegex(EvidenceError, "not ordered"):
            verify_evidence(self.fixture.output)

    def test_harbor_verifier_cannot_be_bypassed_by_changing_kind(self) -> None:
        first, second = self.harbor_results()
        self.export(harbor_results=(first, second))
        bundle = self.fixture.output / "campaigns" / HARBOR_CAMPAIGN_ID
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evidence_kind"] = "synthetic_campaign"
        EvidenceFixture.write_json(manifest_path, manifest)
        index_path = self.fixture.output / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        harbor_entry = next(
            entry
            for entry in index["campaigns"]
            if entry["campaign_id"] == HARBOR_CAMPAIGN_ID
        )
        harbor_entry["evidence_kind"] = "synthetic_campaign"
        EvidenceFixture.write_json(index_path, index)
        self.refresh_campaign_checksums(HARBOR_CAMPAIGN_ID)

        with self.assertRaisesRegex(EvidenceError, "identity or evidence kind"):
            verify_evidence(self.fixture.output)

    def test_harbor_verifier_rejects_boolean_numeric_coercions(self) -> None:
        first, second = self.harbor_results()
        mutations = (
            ("manifest", "payloads_included", 0),
            ("replicates", "replicate_count", 2.0),
        )
        for document_name, key, value in mutations:
            with self.subTest(document=document_name, key=key):
                output = Path(self.temporary.name) / f"evidence-{document_name}"
                self.export(harbor_results=(first, second), output=output)
                document_path = (
                    output
                    / "campaigns"
                    / HARBOR_CAMPAIGN_ID
                    / f"{document_name}.json"
                )
                document = json.loads(document_path.read_text(encoding="utf-8"))
                document[key] = value
                EvidenceFixture.write_json(document_path, document)
                self.refresh_campaign_checksums(
                    HARBOR_CAMPAIGN_ID, evidence_root=output
                )

                with self.assertRaises(EvidenceError):
                    verify_evidence(output)

    def test_changed_export_requires_replace(self) -> None:
        self.export()
        self.fixture.change_aggregate(13.5)

        with self.assertRaisesRegex(EvidenceError, "rerun with --replace"):
            self.export()

        result = self.export(replace=True)
        self.assertTrue(result["changed"])
        self.assertEqual("verified", verify_evidence(self.fixture.output)["status"])

    def test_existing_only_export_is_nonmutating_and_requires_the_target(self) -> None:
        with self.assertRaisesRegex(EvidenceError, "output does not exist"):
            self.export(require_existing_output=True)
        self.assertFalse(self.fixture.output.exists())

        self.export()
        original_index = (self.fixture.output / "index.json").read_bytes()
        unchanged = self.export(require_existing_output=True)
        self.assertFalse(unchanged["changed"])

        self.fixture.change_aggregate(13.5)
        with self.assertRaisesRegex(EvidenceError, "rerun with --replace"):
            self.export(require_existing_output=True)
        self.assertEqual(
            (self.fixture.output / "index.json").read_bytes(), original_index
        )

    def test_tampered_file_fails_checksum_verification(self) -> None:
        self.export()
        index_path = self.fixture.output / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["source_file_count"] += 1
        self.fixture.write_json(index_path, index)

        with self.assertRaisesRegex(EvidenceError, "checksums do not match"):
            verify_evidence(self.fixture.output)

    def test_output_target_cannot_be_results_ancestor(self) -> None:
        with self.assertRaisesRegex(EvidenceError, "unsafe evidence output target"):
            export_evidence(
                results_root=self.fixture.results,
                output_root=Path(self.temporary.name),
            )

    def test_output_target_cannot_be_final_component_symlink(self) -> None:
        real_output = Path(self.temporary.name) / "real-evidence"
        real_output.mkdir()
        self.fixture.output.symlink_to(real_output, target_is_directory=True)

        with self.assertRaisesRegex(EvidenceError, "must not be a symlink"):
            export_evidence(
                results_root=self.fixture.results,
                output_root=self.fixture.output,
            )

    def test_foreign_file_fails_topology_even_with_updated_checksums(self) -> None:
        self.export()
        foreign = self.fixture.output / "foreign.json"
        self.fixture.write_json(foreign, {"schema_version": SCHEMA_VERSION})
        checksums_path = self.fixture.output / "checksums.json"
        checksums = {
            str(path.relative_to(self.fixture.output)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(self.fixture.output.rglob("*"))
            if path.is_file() and path != checksums_path
        }
        self.fixture.write_json(
            checksums_path,
            {"files": checksums, "schema_version": SCHEMA_VERSION},
        )

        with self.assertRaisesRegex(EvidenceError, "top-level layout changed"):
            verify_evidence(self.fixture.output)

    def test_staged_evidence_verifies_with_an_unrelated_text_file(self) -> None:
        self.add_matrix_source()
        self.export()
        repository = Path(self.temporary.name)
        notes = repository / "benchmark-notes.txt"
        notes.write_text("scalar benchmark notes only\n", encoding="utf-8")
        self.git("init", "--quiet")
        self.git("add", "--", self.fixture.output.name, notes.name)

        result = verify_staged_evidence(
            repo_root=repository,
            evidence_root=Path(self.fixture.output.name),
        )

        evidence_files = sum(
            path.is_file() for path in self.fixture.output.rglob("*")
        )
        self.assertEqual("staged_verified", result["status"])
        self.assertEqual(evidence_files, result["files"])
        self.assertEqual(evidence_files + 1, result["staged_file_count"])
        self.assertRegex(result["tree_sha256"], r"^[0-9a-f]{64}$")

    def test_staged_blob_secret_is_detected_after_worktree_overwrite(self) -> None:
        self.add_matrix_source()
        self.export()
        repository = Path(self.temporary.name)
        staged_only = repository / "staged-only.txt"
        staged_value = RAW_SECRET
        staged_only.write_text(f"credential={staged_value}\n", encoding="utf-8")
        self.git("init", "--quiet")
        self.git("add", "--", self.fixture.output.name, staged_only.name)
        staged_only.write_text("safe worktree replacement\n", encoding="utf-8")

        with self.assertRaises(EvidenceError) as caught:
            verify_staged_evidence(
                repo_root=repository,
                evidence_root=Path(self.fixture.output.name),
            )

        message = str(caught.exception)
        self.assertIn("huggingface-token detector matched staged file", message)
        self.assertIn(staged_only.name, message)
        self.assertNotIn(staged_value, message)
        self.assertNotIn(staged_value, staged_only.read_text(encoding="utf-8"))


class EvidenceValidationTests(unittest.TestCase):
    def test_duplicate_keys_nonfinite_and_unknown_request_fields_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"metric":1,"metric":2}\n', encoding="utf-8")
            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"metric":NaN}\n', encoding="utf-8")

            with self.assertRaisesRegex(EvidenceError, "duplicate JSON key"):
                _load_json(duplicate, root)
            with self.assertRaisesRegex(EvidenceError, "non-finite JSON constant"):
                _load_json(nonfinite, root)

        with self.assertRaisesRegex(EvidenceError, "unknown request result fields"):
            _project_request_result({"completion_tokens": 1, "new_raw_field": "x"})

    def test_reasoning_token_count_is_scalar_allowlisted_without_raw_reasoning(
        self,
    ) -> None:
        projected = _project_request_result(
            {
                "completion_tokens": 5,
                "reasoning_tokens": 7,
                "reasoning": f"{RAW_REASONING} {RAW_HOST_PATH}",
            }
        )

        self.assertEqual(projected["reasoning_tokens"], 7)
        self.assertNotIn("reasoning", projected)
        self.assertNotIn(RAW_REASONING, json.dumps(projected))
        self.assertNotIn(RAW_HOST_PATH, json.dumps(projected))
        self.assertEqual(
            _project_request_result({"reasoning_tokens": None}),
            {"reasoning_tokens": None},
        )

    def test_matched_prompt_suite_projects_only_its_versioned_schedule(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        models = load_models(repository / "manifests" / "models.toml")
        sources: list[tuple[dict[str, object], dict[str, object]]] = []
        for study in MATCHED_PROMPT_GRAPH_STUDIES:
            with self.subTest(suite=study.suite_id):
                model = json.loads(
                    json.dumps(
                        model_spec_to_dict(models[sorted(study.profile_ids)[0]])
                    )
                )
                case = {
                    "concurrency": 1,
                    "id": study.case_id,
                    "kind": "decode",
                    "max_output_tokens": 256,
                    "max_turns": 1,
                    "prompt_repetitions": 0,
                    "repetitions": 5,
                    "requires": ["chat"],
                    "temperature": 0.0,
                    "warmups": 1,
                }
                digest = _synthetic_content_hash(
                    {"model": model, "case": case}, length=12
                )
                source = {
                    "cases": [
                        {
                            **case,
                            "case_id": f"{study.case_id}--{digest}",
                        }
                    ],
                    "description": study.suite_description,
                    "id": study.suite_id,
                    "schema_version": 1,
                }
                sources.append((model, source))

                projected = _project_suite({"model": model, "suite": source})
                published_model = _project_model({"model": model}, None)

                self.assertEqual(
                    projected["cases"][0]["prompt_schedule"],
                    MATCHED_REQUEST_UNIQUE_PROTOCOL,
                )
                self.assertEqual(projected["cases"][0]["id"], study.case_id)
                self.assertNotIn("description", projected)
                serialized = json.dumps(projected, sort_keys=True)
                for forbidden in (
                    "request_id",
                    "nonce",
                    "prompt_text",
                    RAW_REQUEST_ID,
                ):
                    self.assertNotIn(forbidden, serialized)
                self.assertEqual(
                    _project_suite(
                        {
                            "model": published_model,
                            "suite": projected,
                        }
                    ),
                    projected,
                )

                forged_schedule = json.loads(json.dumps(projected))
                forged_schedule["cases"][0]["prompt_schedule"] = "unreviewed-v2"
                with self.assertRaisesRegex(EvidenceError, "schedule changed"):
                    _project_suite(
                        {"model": published_model, "suite": forged_schedule}
                    )

                unbound_source = json.loads(json.dumps(source))
                unbound_source["cases"][0]["case_id"] = (
                    f"{study.case_id}--ffffffffffff"
                )
                with self.assertRaisesRegex(EvidenceError, "identifier changed"):
                    _project_suite({"model": model, "suite": unbound_source})

                legacy_case = json.loads(json.dumps(source))
                legacy_case["cases"][0]["id"] = "decode-256-c1"
                with self.assertRaisesRegex(EvidenceError, "case changed"):
                    _project_suite({"model": model, "suite": legacy_case})

        self.assertEqual(len(sources), 2)
        with self.assertRaisesRegex(EvidenceError, "wrong profile"):
            _project_suite(
                {"model": sources[1][0], "suite": sources[0][1]}
            )

    def test_agentic_suite_projection_is_an_exact_four_case_contract(self) -> None:
        suite = _agentic_suite()
        scenarios = tuple(case["id"] for case in suite["cases"])

        projected = _project_suite({"suite": suite})
        self.assertEqual("agentic-tools", projected["id"])
        self.assertNotIn("description", projected)

        invalid_suites = []
        unknown_root = json.loads(json.dumps(suite))
        unknown_root["extra"] = 1
        invalid_suites.append(unknown_root)
        wrong_budget = json.loads(json.dumps(suite))
        wrong_budget["cases"][0]["max_turns"] = 5
        invalid_suites.append(wrong_budget)
        duplicate_scenario = json.loads(json.dumps(suite))
        duplicate_scenario["cases"][1]["id"] = scenarios[0]
        duplicate_scenario["cases"][1]["case_id"] = f"{scenarios[0]}--ffffffffffff"
        invalid_suites.append(duplicate_scenario)
        unknown_case_field = json.loads(json.dumps(suite))
        unknown_case_field["cases"][0]["payload"] = "hidden"
        invalid_suites.append(unknown_case_field)
        for invalid in invalid_suites:
            with self.subTest(invalid=invalid), self.assertRaises(EvidenceError):
                _project_suite({"suite": invalid})

    def test_agentic_request_and_case_metrics_are_scalar_allowlisted(self) -> None:
        agentic_payload = {
            "schema_version": 1,
            "scenario_id": "agentic-two-hop",
            "variant": 2,
            "passed": True,
            "failure_code": None,
            "max_turns": 6,
            "max_output_tokens": 4096,
            "turns_used": 3,
            "expected_tool_calls": 2,
            "tool_calls_requested": 2,
            "tool_calls_executed": 2,
            "tool_calls_succeeded": 2,
            "tool_errors": 0,
            "malformed_tool_calls": 0,
            "unknown_tool_calls": 0,
            "final_answer_emitted": True,
            "final_answer_correct": True,
            "tool_sequence_correct": True,
            "recovery_required": False,
            "recovery_succeeded": False,
            "turn_limit_reached": False,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "emission_events": 4,
            "first_turn_ttft_s": 0.25,
            "request_elapsed_s": 1.5,
            "wall_s": 1.6,
            "length_terminated_turns": 0,
            "elapsed_s": 1.6,
            "ttft_s": 0.25,
            "finish_reason": "stop",
            "output_tps": 12.5,
            "decode_s": None,
            "decode_tps": None,
            "decode_metric_source": None,
        }
        projected_request = _project_request_result(
            agentic_payload,
            kind="agentic",
        )
        self.assertEqual(projected_request["scenario_id"], "agentic-two-hop")
        self.assertEqual(projected_request["tool_calls_executed"], 2)
        self.assertTrue(projected_request["final_answer_correct"])
        self.assertNotIn("content", projected_request)
        self.assertNotIn("tool_calls", projected_request)

        projected_case = _project_case(
            {
                "case_id": "agentic-two-hop--000000000000",
                "kind": "agentic",
                "requests": 3,
                "concurrency": 1,
                "prompt_tokens": 300,
                "completion_tokens": 60,
                "elapsed_s": 9.0,
                "measurement_valid": True,
                "measurement_annotations": [],
                "validation_passed": False,
                "agentic_tasks": 3,
                "agentic_tasks_succeeded": 2,
                "agentic_task_success_rate": 2 / 3,
                "agentic_tasks_per_s": 3 / 9,
                "agentic_max_turns": 6,
                "agentic_max_output_tokens_per_turn": 4096,
                "agentic_model_requests": 12,
                "agentic_model_requests_per_s": 12 / 9,
                "agentic_expected_tool_calls": 6,
                "agentic_tool_calls_requested": 7,
                "agentic_tool_calls_executed": 6,
                "agentic_tool_calls_succeeded": 6,
                "agentic_tool_errors": 0,
                "agentic_malformed_tool_calls": 1,
                "agentic_unknown_tool_calls": 0,
                "agentic_final_answers_emitted": 3,
                "agentic_final_answers_correct": 2,
                "agentic_tool_sequences_correct": 2,
                "agentic_recoveries_required": 0,
                "agentic_recoveries_succeeded": 0,
                "agentic_turn_limit_hits": 0,
                "agentic_length_terminated_turns": 0,
                "median_agentic_turns_used": 4,
                "median_agentic_task_wall_s": 3.0,
                "median_agentic_model_request_sum_s": 2.5,
                "median_agentic_first_turn_ttft_s": 0.2,
                "aggregate_output_tps": None,
                "decode_estimate_one_token_chunks": None,
                "decode_metric_source": None,
                "median_decode_tps": None,
                "median_e2e_s": 3.0,
                "median_estimated_decode_tps": None,
                "median_ttft_s": None,
                "p95_e2e_s": None,
                "p95_ttft_s": None,
                "request_tps": None,
                "telemetry": {},
            }
        )
        self.assertEqual(projected_case["kind"], "agentic")
        self.assertEqual(projected_case["agentic_tasks_succeeded"], 2)
        self.assertIsNone(projected_case["aggregate_output_tps"])

        for mutation in (
            {"schema_version": 2},
            {"variant": 3},
            {"completion_tokens": -1},
            {"tool_calls_executed": 3},
            {"failure_code": "missing_final"},
        ):
            with self.subTest(mutation=mutation):
                invalid = {**agentic_payload, **mutation}
                with self.assertRaises(EvidenceError):
                    _project_request_result(invalid, kind="agentic")

        with self.assertRaises(EvidenceError):
            _project_case(
                {
                    "kind": "decode",
                    "agentic_tasks": 3,
                }
            )

        base_event = {
            "event": "request_complete",
            "case_id": "agentic-two-hop--000000000000",
            "attempt_id": "attempt",
            "kind": "agentic",
            "repetition": 2,
            "burst_elapsed_s": 1.6,
            "result": agentic_payload,
            "validation": {"passed": True},
        }
        case_start = {
            "event": "case_start",
            "case_id": "agentic-two-hop--000000000000",
            "attempt_id": "attempt",
        }
        for missing in ("repetition", "validation"):
            malformed_event = dict(base_event)
            malformed_event.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(EvidenceError):
                _project_requests(
                    [case_start, malformed_event],
                    None,
                    evidence_kind="serving",
                )

        events = [case_start]
        for variant in range(3):
            events.append(
                {
                    **base_event,
                    "repetition": variant,
                    "result": {**agentic_payload, "variant": variant},
                }
            )
        selected_summary = {
            "cases": [
                {
                    "case_id": "agentic-two-hop--000000000000",
                    "attempt_id": "attempt",
                }
            ]
        }
        samples = _project_requests(
            events, selected_summary, evidence_kind="serving"
        )
        with self.assertRaisesRegex(EvidenceError, "exact agentic-tools suite"):
            _validate_agentic_aggregates(samples, {"cases": [projected_case]})
        with self.assertRaisesRegex(EvidenceError, "aggregate disagrees"):
            _validate_agentic_aggregates(
                samples,
                {"cases": [projected_case]},
                suite=_project_suite({"suite": _agentic_suite()}),
            )

    def test_memory_request_and_suite_projection_are_exact_and_model_bound(
        self,
    ) -> None:
        model = _memory_model()
        suite = _memory_suite(model)
        projected_suite = _project_suite({"model": model, "suite": suite})
        self.assertEqual("memory-operations", projected_suite["id"])
        self.assertEqual(
            MEMORY_OPERATION_PROTOCOL_DIGEST,
            projected_suite["protocol_digest"],
        )
        self.assertEqual(
            list(MEMORY_OPERATION_SCENARIO_IDS),
            [case["id"] for case in projected_suite["cases"]],
        )
        self.assertNotIn("description", projected_suite)

        result = _memory_result("memory-add", 1)
        projected = _project_request_result(result, kind="memory")
        self.assertIsNone(projected["reasoning_tokens"])
        self.assertNotIn("emission_events", projected)
        self.assertNotIn("emission_event_count", projected)
        self.assertNotIn("output_tps", projected)

        malformed_results = []
        malformed_results.append({**result, "content": RAW_COMPLETION})
        malformed_results.append({**result, "reasoning_tokens": 3})
        malformed_results.append({**result, "reasoning_tokens": True})
        malformed_results.append(
            {
                **result,
                "emission_events": int(result["completion_tokens"]) + 1,
            }
        )
        malformed_results.append({**result, "cached_prompt_tokens": 1})
        malformed_results.append(
            {**result, "server_prompt_tokens": int(result["prompt_tokens"]) + 1}
        )
        malformed_results.append(
            {
                **result,
                "completion_tokens": MEMORY_OPERATION_OUTPUT_TOKENS + 1,
                "server_decode_tokens": MEMORY_OPERATION_OUTPUT_TOKENS + 1,
            }
        )
        malformed_results.append(
            {
                **result,
                "server_prompt_s": float(result["elapsed_s"])
                + MEMORY_OPERATION_SERVER_TIMING_TOLERANCE_S,
            }
        )
        graphiti_invalid = _memory_result("graphiti-reuse-fact", 0)
        graphiti_invalid.update(
            {
                "schema_valid": False,
                "passed": False,
                "failure_code": "schema_mismatch",
            }
        )
        malformed_results.append(graphiti_invalid)
        for malformed in malformed_results:
            with self.subTest(malformed=malformed), self.assertRaises(EvidenceError):
                _project_request_result(malformed, kind="memory")

        reordered = json.loads(json.dumps(suite))
        reordered["cases"][0], reordered["cases"][1] = (
            reordered["cases"][1],
            reordered["cases"][0],
        )
        with self.assertRaises(EvidenceError):
            _project_suite({"model": model, "suite": reordered})
        changed_protocol = json.loads(json.dumps(suite))
        changed_protocol["protocol_digest"] = "sha256:" + "0" * 64
        for case in changed_protocol["cases"]:
            case_without_id = {
                key: value for key, value in case.items() if key != "case_id"
            }
            digest = hashlib.sha256(
                json.dumps(
                    {
                        "model": model,
                        "case": case_without_id,
                        "protocol_digest": changed_protocol["protocol_digest"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:12]
            case["case_id"] = f"{case['id']}--{digest}"
        with self.assertRaisesRegex(EvidenceError, "protocol digest"):
            _project_suite({"model": model, "suite": changed_protocol})
        with patch.object(
            memory_ops,
            "_EXTENSION_SYSTEM_PROMPT",
            memory_ops._EXTENSION_SYSTEM_PROMPT + "\nUndeclared drift.",
        ):
            with self.assertRaisesRegex(EvidenceError, "protocol digest"):
                _project_suite({"model": model, "suite": suite})
        changed_model = json.loads(json.dumps(model))
        changed_model["args"][-1] = "on"
        with self.assertRaisesRegex(EvidenceError, "frozen model"):
            _project_suite({"model": changed_model, "suite": suite})

    def test_memory_evidence_panel_pins_match_runnable_profiles(self) -> None:
        manifest = tomllib.loads(
            (REPOSITORY / "manifests" / "models.toml").read_text(encoding="utf-8")
        )
        profiles = {model["id"]: model for model in manifest["models"]}
        self.assertEqual(set(_MEMORY_PANEL_MODELS), set(_MEMORY_PANEL_MODEL_ARTIFACTS))
        for profile_id, expected_model in _MEMORY_PANEL_MODELS.items():
            with self.subTest(profile=profile_id):
                profile = profiles[profile_id]
                thinking = json.loads(profile["request_body_json"])[
                    "chat_template_kwargs"
                ]["enable_thinking"]
                derived_model = {
                    key: (
                        thinking
                        if key == "memory_thinking_enabled"
                        else profile[key]
                    )
                    for key in expected_model
                }
                self.assertEqual(expected_model, derived_model)
                self.assertEqual(
                    MEMORY_OPERATION_LLAMACPP_REVISION,
                    profile["runtime_revision"],
                )
                self.assertEqual(
                    MEMORY_OPERATION_LLAMACPP_DIGEST,
                    profile["runtime_digest"],
                )
                self.assertEqual(
                    list(memory_operation_llamacpp_args(enable_thinking=thinking)),
                    profile["args"],
                )
                artifacts = []
                if "model_shards" in profile:
                    for index, shard in enumerate(profile["model_shards"], start=1):
                        artifacts.append(
                            {
                                "role": f"model_shard_{index}",
                                "sha256": shard["digest"].removeprefix("sha256:"),
                                "size_bytes": shard["size_bytes"],
                                "source": profile["source"],
                                "revision": profile["revision"],
                                "target": Path(shard["path"]).name,
                            }
                        )
                    self.assertEqual(profile["weight_file_count"], len(artifacts))
                    self.assertEqual(
                        profile["weight_size_bytes"],
                        sum(item["size_bytes"] for item in artifacts),
                    )
                else:
                    artifacts.append(
                        {
                            "role": "model",
                            "sha256": profile["model_digest"].removeprefix("sha256:"),
                            "size_bytes": profile["model_size_bytes"],
                            "source": profile["source"],
                            "revision": profile["revision"],
                            "target": Path(profile["model_file"]).name,
                        }
                    )
                self.assertEqual(
                    list(_MEMORY_PANEL_MODEL_ARTIFACTS[profile_id]), artifacts
                )

    def test_source_tree_rejects_symlinks_hardlinks_and_fifos(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            (root / "linked.json").symlink_to(target)
            with self.assertRaisesRegex(EvidenceError, "special or linked file"):
                _assert_source_tree(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            os.link(target, root / "hardlink.json")
            with self.assertRaisesRegex(EvidenceError, "special or linked file"):
                _assert_source_tree(root)

        if hasattr(os, "mkfifo"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                os.mkfifo(root / "telemetry.pipe")
                with self.assertRaisesRegex(EvidenceError, "special or linked file"):
                    _assert_source_tree(root)

    def test_output_validator_rejects_forbidden_keys_paths_and_credentials(self) -> None:
        invalid_values = (
            {"request-id": "opaque"},
            {"requestId": "opaque"},
            {"requestIdentifier": "opaque"},
            {"promptText": "opaque"},
            {"completion_text": "opaque"},
            {"reasoningText": "opaque"},
            {"payload": "opaque"},
            {"toolPayload": "opaque"},
            {"nested": {"prompt": "opaque"}},
            {"artifact_path": "relative/model.gguf"},
            {"safe_metric": RAW_HOST_PATH},
            {"safe_metric": RAW_SECRET},
            {"safe_metric": "sk-" + "proj-0123456789abcdefghijklmnop"},
            {
                "safe_metric": (
                    "eyJ0123456789abcd.eyJ0123456789abcd."
                    "eyJ0123456789abcd"
                )
            },
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(EvidenceError):
                    _validate_output_value(value)

    def test_checksums_only_tree_fails_topology_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            EvidenceFixture.write_json(
                root / "checksums.json",
                {"files": {}, "schema_version": SCHEMA_VERSION},
            )

            with self.assertRaisesRegex(EvidenceError, "top-level layout changed"):
                verify_evidence(root)

    def test_serving_request_missing_attempt_id_fails_closed(self) -> None:
        events = [
            {
                "event": "request_complete",
                "case_id": "chat-case",
                "kind": "chat",
                "result": {"completion_tokens": 1},
            }
        ]

        with self.assertRaisesRegex(EvidenceError, "attempt identifier is missing"):
            _project_requests(events, None, evidence_kind="serving")

    def test_ninfer_report_rejects_non_v11_schema(self) -> None:
        report = dict.fromkeys(_NINFER_TOP_FIELDS)
        report.update(
            {
                "artifact_type": "ninfer_bench_report",
                "schema_version": 10,
                "tool": "ninfer_bench",
            }
        )

        with self.assertRaisesRegex(EvidenceError, "unrecognized NInfer report"):
            _project_ninfer_report(report)

    def test_summary_identity_fields_require_text(self) -> None:
        for field in ("status", "schema_version", "suite"):
            with self.subTest(field=field):
                with self.assertRaises(EvidenceError):
                    _project_summary({field: 42})

    def test_required_summary_numeric_fields_reject_bool_and_null(self) -> None:
        for value in (True, None):
            with self.subTest(scope="summary", value=value):
                with self.assertRaises(EvidenceError):
                    _project_summary({"completed_cases": value})
            with self.subTest(scope="case", value=value):
                with self.assertRaises(EvidenceError):
                    _project_case({"requests": value})

    def test_summary_aggregate_fields_reject_wrong_types(self) -> None:
        invalid = (
            ({"metrics": {"metric_source": 42}}, "metrics.metric_source"),
            ({"runtime": {"python": False}}, "runtime.python"),
            ({"metrics": {"perplexity": {"value": 2.0}}}, "metrics.perplexity"),
        )
        for summary, field in invalid:
            with self.subTest(field=field):
                with self.assertRaises(EvidenceError):
                    _project_summary(summary)

    def test_summary_aggregate_roots_require_objects(self) -> None:
        invalid = (
            {"metrics": 42},
            {"runtime": False},
            {"memory": []},
        )
        for summary in invalid:
            with self.subTest(root=next(iter(summary))):
                with self.assertRaises(EvidenceError):
                    _project_summary(summary)

    def test_nested_aggregate_keys_and_digest_lists_are_strict(self) -> None:
        invalid = (
            {
                "speculative_decoding": {
                    "accepted_tokens_per_position": {"not-an-index": 1}
                }
            },
            {
                "artifact_validation": {
                    "model_shard_sha256s": [["a" * 64]]
                }
            },
        )
        for summary in invalid:
            with self.subTest(root=next(iter(summary))):
                with self.assertRaises(EvidenceError):
                    _project_summary(summary)

    def test_summary_case_preserves_allowlisted_nullables(self) -> None:
        projected = _project_case(
            {
                "aggregate_output_tps": None,
                "case_id": None,
                "median_ttft_s": None,
                "validation_passed": None,
            }
        )

        self.assertEqual(
            {
                "aggregate_output_tps": None,
                "case_id": None,
                "median_ttft_s": None,
                "validation_passed": None,
            },
            projected,
        )

    def test_single_file_artifact_validation_omits_null_shard_fields(self) -> None:
        projected = _project_summary(
            {
                "artifact_validation": {
                    "model_shard_count": None,
                    "model_shard_sha256s": None,
                    "model_total_size_bytes": None,
                }
            }
        )

        self.assertEqual(
            {"artifact_validation": {}, "startup_safety_gates": []},
            projected,
        )

    def test_artifact_validation_requires_atomic_shard_metadata(self) -> None:
        invalid = (
            {"model_shard_count": 3},
            {
                "model_shard_count": 3,
                "model_shard_sha256s": ["a" * 64, "b" * 64],
                "model_total_size_bytes": 42,
            },
            {
                "model_shard_count": True,
                "model_shard_sha256s": ["a" * 64],
                "model_total_size_bytes": 42,
            },
        )
        for artifact_validation in invalid:
            with self.subTest(artifact_validation=artifact_validation):
                with self.assertRaises(EvidenceError):
                    _project_summary({"artifact_validation": artifact_validation})

    def test_quality_accuracy_category_rejects_null(self) -> None:
        with self.assertRaises(EvidenceError):
            _project_case(
                {"quality_accuracy_by_category": {"code": None}}
            )


if __name__ == "__main__":
    unittest.main()
