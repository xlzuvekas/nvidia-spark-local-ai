from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import bench.memory_ops as memory_ops
from bench.journal import Journal
from bench.journal import content_hash
from bench.manifest import (
    CaseSpec,
    KNOWN_MEMORY_OPERATION_CASE_IDS,
    ManifestError,
    load_suite,
    validate_case,
)
from bench.memory_ops import (
    MEMORY_OPERATION_CONTEXT_TOKENS,
    MEMORY_OPERATION_OUTPUT_TOKENS,
    MEMORY_OPERATION_PROTOCOL_DIGEST,
    MEMORY_OPERATION_SCENARIO_IDS,
    MEMORY_OPERATION_SUITE_DESCRIPTION,
    MEMORY_OPERATION_SUITE_ID,
    MEMORY_OPERATION_VARIANT_COUNT,
    MemoryOperationError,
    MemoryOperationRunResult,
    _scenario,
    compute_memory_operation_protocol_digest,
    estimate_memory_operation_context_tokens,
    memory_operation_llamacpp_args,
    run_memory_operation_scenario,
)
from bench.report import summarize_run
from bench.runner import (
    PreflightError,
    _canonical_case,
    _estimated_context_tokens,
    _execute_case,
    _validate_memory_operation_plan_selection,
    create_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def _chat_result(
    content: str,
    *,
    reasoning: str = "",
    reasoning_tokens: int | None = 7,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=101,
        completion_tokens=23,
        reasoning_tokens=reasoning_tokens,
        ttft_s=0.02,
        elapsed_s=0.2,
        decode_s=0.18,
        decode_tps=22 / 0.18,
        output_tps=115.0,
        emission_events=4,
        finish_reason=finish_reason,
        decode_metric_source="client_estimate",
        cached_prompt_tokens=0,
        server_prompt_tokens=101,
        server_cached_prompt_tokens=0,
        server_decode_tokens=23,
        server_prompt_s=0.05,
        server_decode_s=0.15,
        content=content,
        reasoning=reasoning,
        tool_calls=[],
    )


class _ScriptedRequest:
    def __init__(self, results: list[SimpleNamespace]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if not self.results:
            raise AssertionError("memory benchmark made an unexpected request")
        return self.results.pop(0)


def _run(
    scenario_id: str,
    variant: int,
    response: SimpleNamespace,
    *,
    seed: str = "ephemeral-test-1700000000000000000",
) -> tuple[MemoryOperationRunResult, _ScriptedRequest]:
    request = _ScriptedRequest([response])
    result = run_memory_operation_scenario(
        scenario_id=scenario_id,
        variant=variant,
        request_function=request,
        request_kwargs={
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "synthetic-model",
        },
        request_id_prefix=seed,
        max_output_tokens=MEMORY_OPERATION_OUTPUT_TOKENS,
        temperature=0.0,
        extra_body={
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return result, request


def _oracle_response(
    scenario_id: str,
    variant: int,
    *,
    seed: str = "ephemeral-test-1700000000000000000",
    reasoning_tokens: int | None = 7,
) -> SimpleNamespace:
    scenario = _scenario(scenario_id, variant)
    return _chat_result(
        json.dumps(scenario.expected, separators=(",", ":"), sort_keys=True),
        reasoning_tokens=reasoning_tokens,
    )


def _runner_case(case_id: str = "graphiti-reuse-fact") -> SimpleNamespace:
    return SimpleNamespace(
        id=case_id,
        case_id=f"{case_id}--synthetic",
        kind="memory",
        requires=["chat", "json"],
        warmups=0,
        repetitions=3,
        max_output_tokens=1536,
        max_turns=1,
        temperature=0.0,
        concurrency=1,
        prompt_repetitions=0,
    )


def _frozen_memory_selection(
    *, enable_thinking: bool = False
) -> tuple[SimpleNamespace, SimpleNamespace]:
    tasks = ["chat", "json"]
    if enable_thinking:
        tasks.append("thinking")
    model = SimpleNamespace(
        backend="llamacpp",
        lifecycle="subprocess",
        runtime_revision=(
            "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70"
        ),
        runtime_digest=(
            "sha256:ae1bd49f869ff3397b2a5d757fcf010c6eaaf16c4e3071a15861312defcd4e40"
        ),
        runtime_parallel=1,
        max_context=MEMORY_OPERATION_CONTEXT_TOKENS,
        native_context=MEMORY_OPERATION_CONTEXT_TOKENS,
        tasks=tasks,
        request_body_json=json.dumps(
            {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            separators=(",", ":"),
        ),
        args=list(
            memory_operation_llamacpp_args(enable_thinking=enable_thinking)
        ),
    )
    cases = [
        SimpleNamespace(
            case_id=f"{scenario_id}--{index:012x}",
            concurrency=1,
            id=scenario_id,
            kind="memory",
            max_output_tokens=MEMORY_OPERATION_OUTPUT_TOKENS,
            max_turns=1,
            prompt_repetitions=0,
            repetitions=MEMORY_OPERATION_VARIANT_COUNT,
            requires=["chat", "json"],
            temperature=0.0,
            warmups=0,
        )
        for index, scenario_id in enumerate(MEMORY_OPERATION_SCENARIO_IDS)
    ]
    return model, SimpleNamespace(
        id=MEMORY_OPERATION_SUITE_ID,
        description=MEMORY_OPERATION_SUITE_DESCRIPTION,
        protocol_digest=MEMORY_OPERATION_PROTOCOL_DIGEST,
        schema_version=1,
        cases=cases,
    )


def _scalar_result(
    *,
    scenario_id: str = "graphiti-reuse-fact",
    variant: int = 0,
    reasoning_tokens: int | None = 7,
) -> MemoryOperationRunResult:
    graphiti = scenario_id.startswith("graphiti-")
    expected_actions = {
        "graphiti-reuse-fact": "REUSE_FACT",
        "graphiti-invalidate-fact": "CREATE_AND_INVALIDATE",
        "graphiti-create-fact": "CREATE_FACT",
    }
    expected_action = expected_actions.get(scenario_id)
    observed_extension_action = (
        None if graphiti else str(_scenario(scenario_id, variant).expected["action"])
    )
    mutation_selected = observed_extension_action in {
        "ADD",
        "SUPERSEDE",
        "DELETE",
        "INVALIDATE",
    }
    secret_required = scenario_id == "memory-secret-refusal"
    injection_required = scenario_id == "memory-injection-refusal"
    return MemoryOperationRunResult(
        schema_version=1,
        scenario_id=scenario_id,
        variant=variant,
        passed=True,
        failure_code=None,
        json_object_emitted=True,
        schema_valid=True,
        action_correct=True,
        target_correct=None if graphiti else True,
        path_correct=None if graphiti else True,
        tier_correct=None if graphiti else True,
        value_correct=None if graphiti else True,
        valid_from_correct=None if graphiti else True,
        valid_to_correct=None if graphiti else True,
        evidence_correct=None if graphiti else True,
        reason_correct=None if graphiti else True,
        duplicate_facts_correct=True if graphiti else None,
        contradicted_facts_correct=True if graphiti else None,
        protected_value_emitted=False,
        mutation_expected=mutation_selected,
        mutation_selected=mutation_selected,
        secret_refusal_required=secret_required,
        secret_refusal_succeeded=secret_required,
        injection_refusal_required=injection_required,
        injection_refusal_succeeded=injection_required,
        graphiti_resolver_case=graphiti,
        synthetic_extension_case=not graphiti,
        resolver_decision_correct=True if graphiti else None,
        expected_resolver_action=expected_action,
        selected_resolver_action=expected_action,
        unexpected_field_count=0,
        unexpected_tool_call_count=0,
        max_output_tokens=1536,
        prompt_cache_disabled=True,
        prompt_tokens=101,
        cached_prompt_tokens=0,
        completion_tokens=23,
        reasoning_tokens=reasoning_tokens,
        emission_events=4,
        ttft_s=0.02,
        elapsed_s=0.2,
        decode_s=0.18,
        decode_tps=22 / 0.18,
        output_tps=115.0,
        server_prompt_tokens=101,
        server_cached_prompt_tokens=0,
        server_decode_tokens=23,
        server_prompt_s=0.05,
        server_decode_s=0.15,
        finish_reason="stop",
        decode_metric_source="client_estimate",
    )


class MemoryOperationTests(unittest.TestCase):
    def test_manifest_constants_and_family_order_do_not_drift(self) -> None:
        self.assertEqual(
            KNOWN_MEMORY_OPERATION_CASE_IDS,
            frozenset(MEMORY_OPERATION_SCENARIO_IDS),
        )
        suite = load_suite(
            ROOT / "manifests" / "suites" / "memory_operations.toml"
        )

        self.assertEqual(suite.id, "memory-operations")
        self.assertEqual(suite.description, MEMORY_OPERATION_SUITE_DESCRIPTION)
        self.assertEqual(
            suite.protocol_digest, MEMORY_OPERATION_PROTOCOL_DIGEST
        )
        self.assertEqual(
            tuple(case.id for case in suite.cases),
            MEMORY_OPERATION_SCENARIO_IDS,
        )
        self.assertEqual(
            tuple(case.id for case in suite.cases[:3]),
            (
                "graphiti-reuse-fact",
                "graphiti-invalidate-fact",
                "graphiti-create-fact",
            ),
        )
        for case in suite.cases:
            with self.subTest(case=case.id):
                self.assertEqual(case.kind, "memory")
                self.assertEqual(case.requires, ("chat", "json"))
                self.assertEqual(case.repetitions, MEMORY_OPERATION_VARIANT_COUNT)
                self.assertEqual(case.warmups, 0)
                self.assertEqual(case.max_output_tokens, 1536)
                self.assertEqual(case.max_turns, 1)
                self.assertEqual(case.temperature, 0.0)
                self.assertEqual(case.concurrency, 1)

    def test_manifest_rejects_noncanonical_memory_cases(self) -> None:
        valid = CaseSpec(
            id="memory-add",
            kind="memory",
            requires=("chat", "json"),
            repetitions=3,
            max_output_tokens=1536,
        )
        validate_case(valid)
        invalid = (
            replace(valid, id="memory-unknown"),
            replace(valid, requires=("chat",)),
            replace(valid, repetitions=1),
            replace(valid, warmups=1),
            replace(valid, concurrency=2),
            replace(valid, temperature=0.1),
            replace(valid, max_output_tokens=1535),
            replace(valid, max_turns=2),
        )
        for case in invalid:
            with self.subTest(case=case), self.assertRaises(ManifestError):
                validate_case(case)

    def test_protocol_digest_recomputes_from_prompts_schemas_and_oracles(
        self,
    ) -> None:
        self.assertEqual(
            MEMORY_OPERATION_PROTOCOL_DIGEST,
            compute_memory_operation_protocol_digest(),
        )
        payload = memory_ops.memory_operation_protocol_payload()
        scenarios = payload["oracle"]["scenarios"]
        self.assertEqual(
            len(MEMORY_OPERATION_SCENARIO_IDS) * MEMORY_OPERATION_VARIANT_COUNT,
            len(scenarios),
        )
        self.assertEqual(
            [
                (scenario_id, variant)
                for scenario_id in MEMORY_OPERATION_SCENARIO_IDS
                for variant in range(MEMORY_OPERATION_VARIANT_COUNT)
            ],
            [(row["id"], row["variant"]) for row in scenarios],
        )

        baseline_case = asdict(
            load_suite(
                ROOT / "manifests" / "suites" / "memory_operations.toml"
            ).cases[0]
        )
        model = {"id": "synthetic-model", "tasks": ["chat", "json"]}
        baseline_case_id = _canonical_case(
            model,
            baseline_case,
            protocol_digest=MEMORY_OPERATION_PROTOCOL_DIGEST,
        )["case_id"]
        baseline_fingerprint = content_hash(
            {
                "model": model,
                "suite": {
                    "id": MEMORY_OPERATION_SUITE_ID,
                    "protocol_digest": MEMORY_OPERATION_PROTOCOL_DIGEST,
                    "cases": [baseline_case],
                },
                "resolved": {},
            }
        )

        original_scenario = memory_ops._scenario

        def oracle_drift(scenario_id: str, variant: int) -> object:
            scenario = original_scenario(scenario_id, variant)
            if scenario_id == MEMORY_OPERATION_SCENARIO_IDS[0] and variant == 0:
                expected = dict(scenario.expected)
                expected["duplicate_facts"] = []
                return replace(scenario, expected=expected)
            return scenario

        drift_contexts = (
            patch.object(
                memory_ops,
                "_GRAPHITI_SYSTEM_PROMPT",
                memory_ops._GRAPHITI_SYSTEM_PROMPT + "\nProtocol drift.",
            ),
            patch.object(
                memory_ops,
                "_GRAPHITI_RESPONSE_FORMAT",
                {
                    **memory_ops._GRAPHITI_RESPONSE_FORMAT,
                    "json_schema": {
                        **memory_ops._GRAPHITI_RESPONSE_FORMAT["json_schema"],
                        "name": "drifted_graphiti_schema",
                    },
                },
            ),
            patch.object(memory_ops, "_scenario", side_effect=oracle_drift),
        )
        for label, context in zip(
            ("system_prompt", "response_schema", "oracle"),
            drift_contexts,
            strict=True,
        ):
            with self.subTest(drift=label), context:
                drift_digest = compute_memory_operation_protocol_digest()
                self.assertNotEqual(MEMORY_OPERATION_PROTOCOL_DIGEST, drift_digest)
                drift_case_id = _canonical_case(
                    model,
                    baseline_case,
                    protocol_digest=drift_digest,
                )["case_id"]
                drift_fingerprint = content_hash(
                    {
                        "model": model,
                        "suite": {
                            "id": MEMORY_OPERATION_SUITE_ID,
                            "protocol_digest": drift_digest,
                            "cases": [baseline_case],
                        },
                        "resolved": {},
                    }
                )
                self.assertNotEqual(baseline_case_id, drift_case_id)
                self.assertNotEqual(baseline_fingerprint, drift_fingerprint)

        legacy_case = _canonical_case(model, baseline_case)
        self.assertEqual(
            legacy_case["case_id"],
            f"{baseline_case['id']}--"
            f"{content_hash({'model': model, 'case': baseline_case}, 12)}",
        )
        production_drift_contexts = (
            patch.object(
                memory_ops,
                "_GRAPHITI_SYSTEM_PROMPT",
                memory_ops._GRAPHITI_SYSTEM_PROMPT + "\nUndeclared drift.",
            ),
            patch.object(memory_ops, "_scenario", side_effect=oracle_drift),
        )
        for label, context in zip(
            ("prompt", "oracle"), production_drift_contexts, strict=True
        ):
            with self.subTest(stale_runtime_digest=label), context:
                frozen_model, frozen_suite = _frozen_memory_selection()
                with self.assertRaisesRegex(PreflightError, "digest"):
                    _validate_memory_operation_plan_selection(
                        frozen_model, frozen_suite
                    )
                with self.assertRaisesRegex(ManifestError, "digest"):
                    load_suite(
                        ROOT
                        / "manifests"
                        / "suites"
                        / "memory_operations.toml"
                    )
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaisesRegex(ManifestError, "digest"):
                        create_plan(
                            model=frozen_model,
                            suite=frozen_suite,
                            results_root=Path(directory),
                            models_path=ROOT / "manifests" / "models.toml",
                            suite_path=(
                                ROOT
                                / "manifests"
                                / "suites"
                                / "memory_operations.toml"
                            ),
                        )

    def test_frozen_memory_selection_accepts_only_the_exact_protocol(self) -> None:
        for enable_thinking in (False, True):
            with self.subTest(enable_thinking=enable_thinking):
                model, suite = _frozen_memory_selection(
                    enable_thinking=enable_thinking
                )
                _validate_memory_operation_plan_selection(model, suite)

        def permute_cases(model: SimpleNamespace, suite: SimpleNamespace) -> None:
            suite.cases[0], suite.cases[1] = suite.cases[1], suite.cases[0]

        def change_backend(model: SimpleNamespace, suite: SimpleNamespace) -> None:
            model.backend = "ollama"

        def change_lifecycle(model: SimpleNamespace, suite: SimpleNamespace) -> None:
            model.lifecycle = "container"

        def change_runtime_revision(
            model: SimpleNamespace, suite: SimpleNamespace
        ) -> None:
            model.runtime_revision = "0" * 40

        def change_runtime_digest(
            model: SimpleNamespace, suite: SimpleNamespace
        ) -> None:
            model.runtime_digest = "sha256:" + "0" * 64

        def change_description(
            model: SimpleNamespace, suite: SimpleNamespace
        ) -> None:
            suite.description = "Altered memory-operation suite"

        def change_protocol_digest(
            model: SimpleNamespace, suite: SimpleNamespace
        ) -> None:
            suite.protocol_digest = "sha256:" + "0" * 64

        def change_context(model: SimpleNamespace, suite: SimpleNamespace) -> None:
            model.max_context = MEMORY_OPERATION_CONTEXT_TOKENS // 2

        def change_arguments(model: SimpleNamespace, suite: SimpleNamespace) -> None:
            model.args.append("--cache-prompt")

        def change_tasks(model: SimpleNamespace, suite: SimpleNamespace) -> None:
            model.tasks = ["chat"]

        def mismatch_thinking(model: SimpleNamespace, suite: SimpleNamespace) -> None:
            model.request_body_json = (
                '{"chat_template_kwargs":{"enable_thinking":true}}'
            )

        def invalidate_thinking(model: SimpleNamespace, suite: SimpleNamespace) -> None:
            model.request_body_json = (
                '{"chat_template_kwargs":{"enable_thinking":"true"}}'
            )

        mutations = {
            "permuted_cases": permute_cases,
            "backend": change_backend,
            "lifecycle": change_lifecycle,
            "runtime_revision": change_runtime_revision,
            "runtime_digest": change_runtime_digest,
            "suite_description": change_description,
            "suite_protocol_digest": change_protocol_digest,
            "context": change_context,
            "arguments": change_arguments,
            "tasks": change_tasks,
            "thinking_capability": mismatch_thinking,
            "thinking_policy": invalidate_thinking,
        }
        for label, mutate in mutations.items():
            with self.subTest(mutation=label):
                model, suite = _frozen_memory_selection()
                mutate(model, suite)
                with self.assertRaises(PreflightError):
                    _validate_memory_operation_plan_selection(model, suite)

    def test_every_scenario_and_nonce_variant_has_an_exact_oracle(self) -> None:
        for scenario_id in MEMORY_OPERATION_SCENARIO_IDS:
            for variant in range(MEMORY_OPERATION_VARIANT_COUNT):
                with self.subTest(scenario=scenario_id, variant=variant):
                    result, request = _run(
                        scenario_id,
                        variant,
                        _oracle_response(scenario_id, variant),
                    )

                    self.assertTrue(result.passed)
                    self.assertIsNone(result.failure_code)
                    self.assertTrue(result.schema_valid)
                    self.assertTrue(result.action_correct)
                    self.assertEqual(result.reasoning_tokens, 7)
                    self.assertEqual(len(request.calls), 1)
                    body = request.calls[0]["extra_body"]
                    self.assertIs(body["cache_prompt"], False)
                    response_format = body["response_format"]
                    self.assertEqual(response_format["type"], "json_schema")
                    self.assertTrue(response_format["json_schema"]["strict"])
                    response_schema = response_format["json_schema"]["schema"]
                    self.assertFalse(response_schema["additionalProperties"])
                    self.assertEqual(
                        set(response_schema["properties"]),
                        set(response_schema["required"]),
                    )
                    self.assertEqual(
                        set(response_schema["required"]),
                        set(
                            _scenario(scenario_id, variant).expected
                        ),
                    )
                    self.assertEqual(body["messages"][0]["role"], "system")
                    self.assertEqual(body["messages"][1]["role"], "user")

                    if scenario_id.startswith("graphiti-"):
                        self.assertTrue(result.graphiti_resolver_case)
                        self.assertFalse(result.synthetic_extension_case)
                        prompt = _scenario(scenario_id, variant).prompt
                        self.assertIn("Existing facts:", prompt)
                        self.assertIn("Fact invalidation candidates:", prompt)
                        self.assertIn("New fact:", prompt)
                        self.assertNotIn("Transcript:", prompt)
                        if scenario_id == "graphiti-invalidate-fact":
                            self.assertIn(
                                "workspace_label is single-valued", prompt
                            )
                            self.assertIn("supersedes the existing one", prompt)
                        self.assertTrue(result.resolver_decision_correct)
                        self.assertIsNone(result.target_correct)
                        self.assertIsNone(result.evidence_correct)
                        self.assertTrue(result.duplicate_facts_correct)
                        self.assertTrue(result.contradicted_facts_correct)
                        for property_schema in response_schema["properties"].values():
                            self.assertEqual(property_schema["type"], "array")
                            self.assertEqual(
                                property_schema["items"]["type"], "integer"
                            )
                        self.assertIn(
                            result.expected_resolver_action,
                            {
                                "REUSE_FACT",
                                "CREATE_AND_INVALIDATE",
                                "CREATE_FACT",
                            },
                        )
                    else:
                        self.assertFalse(result.graphiti_resolver_case)
                        self.assertTrue(result.synthetic_extension_case)
                        expected_reason = _scenario(
                            scenario_id, variant
                        ).expected["reason"]
                        self.assertIn(
                            f"Canonical reason: {expected_reason}.",
                            body["messages"][1]["content"],
                        )
                        self.assertIsNone(result.resolver_decision_correct)
                        self.assertTrue(result.target_correct)
                        self.assertTrue(result.evidence_correct)
                        self.assertIsNone(result.duplicate_facts_correct)
                        self.assertIsNone(result.contradicted_facts_correct)
                        properties = response_schema["properties"]
                        self.assertEqual(properties["action"]["type"], "string")
                        self.assertEqual(
                            {item["type"] for item in properties["target"]["anyOf"]},
                            {"string", "null"},
                        )
                        self.assertEqual(properties["evidence"]["type"], "array")
                        self.assertEqual(
                            properties["evidence"]["items"]["type"], "integer"
                        )
                        for date_field in ("valid_from", "valid_to"):
                            date_pattern = properties[date_field]["anyOf"][0][
                                "pattern"
                            ]
                            self.assertEqual(
                                date_pattern,
                                r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
                            )
                            self.assertNotIn(r"\d", date_pattern)
                        self.assertEqual(properties["duplicate_facts"]["maxItems"], 0)
                        self.assertEqual(
                            properties["contradicted_facts"]["maxItems"], 0
                        )

    def test_request_controls_require_cache_disable(self) -> None:
        request = _ScriptedRequest([_oracle_response("memory-add", 0)])
        base_arguments = {
            "scenario_id": "memory-add",
            "variant": 0,
            "request_function": request,
            "request_kwargs": {
                "base_url": "http://127.0.0.1:8000/v1",
                "model": "synthetic-model",
            },
            "request_id_prefix": "ephemeral-test",
            "max_output_tokens": MEMORY_OPERATION_OUTPUT_TOKENS,
        }
        for extra_body in ({}, {"cache_prompt": True}):
            with self.subTest(extra_body=extra_body), self.assertRaisesRegex(
                ValueError, "cache_prompt false"
            ):
                run_memory_operation_scenario(
                    **base_arguments,
                    extra_body=extra_body,
                )
        self.assertEqual(request.calls, [])

    def test_oracles_are_replayable_and_variants_remain_distinct(self) -> None:
        first = _scenario("memory-add", 0)
        replay = _scenario("memory-add", 0)
        second = _scenario("memory-add", 1)

        self.assertEqual(first, replay)
        self.assertNotEqual(first.expected, second.expected)
        self.assertNotEqual(first.prompt, second.prompt)

    def test_persistable_result_is_scalar_only_and_omits_ephemeral_values(self) -> None:
        seed = "ephemeral-test-1700000000000000000"
        scenario = _scenario("memory-add", 0)
        result, _ = _run(
            "memory-add", 0, _oracle_response("memory-add", 0, seed=seed), seed=seed
        )
        payload = result.to_dict()
        serialized = json.dumps(payload, sort_keys=True)

        banned_keys = {
            "content",
            "expected",
            "messages",
            "model",
            "path",
            "prompt",
            "reasoning",
            "request_id",
            "target",
            "tool_calls",
            "value",
        }
        self.assertFalse(banned_keys & set(payload))
        self.assertTrue(
            all(
                isinstance(value, (bool, float, int, str, type(None)))
                for value in payload.values()
            )
        )
        for ephemeral in scenario.expected.values():
            if isinstance(ephemeral, str) and ephemeral not in {
                "ADD",
                "profile",
                "new_fact",
            }:
                self.assertNotIn(ephemeral, serialized)

    def test_invalid_json_extra_fields_and_wrong_operations_fail_exactly(self) -> None:
        invalid, _ = _run("memory-add", 0, _chat_result("not json"))
        self.assertFalse(invalid.passed)
        self.assertEqual(invalid.failure_code, "invalid_json")

        scenario = _scenario("memory-add", 0)
        extra = dict(scenario.expected)
        extra["comment"] = "should not be accepted"
        schema_failure, _ = _run(
            "memory-add", 0, _chat_result(json.dumps(extra))
        )
        self.assertEqual(schema_failure.failure_code, "schema_mismatch")
        self.assertEqual(schema_failure.unexpected_field_count, 1)

        wrong = dict(scenario.expected)
        wrong["action"] = "DELETE"
        operation_failure, _ = _run(
            "memory-add", 0, _chat_result(json.dumps(wrong))
        )
        self.assertEqual(operation_failure.failure_code, "operation_mismatch")
        self.assertFalse(operation_failure.action_correct)

        resolver = _scenario("graphiti-reuse-fact", 0)
        wrong_set = dict(resolver.expected)
        wrong_set["duplicate_facts"] = [1]
        resolver_failure, _ = _run(
            "graphiti-reuse-fact", 0, _chat_result(json.dumps(wrong_set))
        )
        self.assertEqual(resolver_failure.failure_code, "operation_mismatch")
        self.assertTrue(resolver_failure.action_correct)
        self.assertFalse(resolver_failure.duplicate_facts_correct)
        self.assertFalse(resolver_failure.resolver_decision_correct)

    def test_exact_json_rejects_duplicate_constants_and_oversize_payloads(self) -> None:
        scenario = _scenario("memory-add", 0)
        canonical = json.dumps(
            scenario.expected, separators=(",", ":"), sort_keys=True
        )
        duplicate_key = canonical[:-1] + ',"action":"ADD"}'
        payloads = {
            "duplicate_key": duplicate_key,
            "nan": '{"value":NaN}',
            "positive_infinity": '{"value":Infinity}',
            "negative_infinity": '{"value":-Infinity}',
            "oversize": json.dumps("x" * 20_000),
        }
        for label, payload in payloads.items():
            with self.subTest(payload=label):
                result, _ = _run("memory-add", 0, _chat_result(payload))
                self.assertFalse(result.passed)
                self.assertEqual(result.failure_code, "invalid_json")
                self.assertFalse(result.json_object_emitted)

    def test_index_arrays_reject_bool_range_and_duplicate_values(self) -> None:
        graphiti = _scenario("graphiti-reuse-fact", 0)
        extension = _scenario("memory-add", 0)
        invalid_indexes = {
            "bool": [True],
            "out_of_range": [2],
            "duplicate": [0, 0],
        }
        for label, indexes in invalid_indexes.items():
            with self.subTest(family="graphiti", invalid=label):
                payload = dict(graphiti.expected)
                payload["duplicate_facts"] = indexes
                result, _ = _run(
                    "graphiti-reuse-fact", 0, _chat_result(json.dumps(payload))
                )
                self.assertEqual(result.failure_code, "schema_mismatch")
                self.assertFalse(result.schema_valid)

        extension_indexes = {
            "bool": [True],
            "out_of_range": [1_000_001],
            "duplicate": [0, 0],
        }
        for label, indexes in extension_indexes.items():
            with self.subTest(family="extension", invalid=label):
                payload = dict(extension.expected)
                payload["evidence"] = indexes
                result, _ = _run(
                    "memory-add", 0, _chat_result(json.dumps(payload))
                )
                self.assertEqual(result.failure_code, "schema_mismatch")
                self.assertFalse(result.schema_valid)

    def test_protected_values_in_visible_or_hidden_output_fail_without_persistence(self) -> None:
        seed = "ephemeral-test-1700000000000000000"
        scenario = _scenario("memory-secret-refusal", 0)
        protected_value = scenario.protected_values[0]
        response = _chat_result(
            json.dumps(scenario.expected),
            reasoning=f"Do not retain {protected_value}",
        )

        result, _ = _run("memory-secret-refusal", 0, response, seed=seed)

        self.assertFalse(result.passed)
        self.assertEqual(result.failure_code, "protected_value")
        self.assertTrue(result.protected_value_emitted)
        self.assertNotIn(protected_value, json.dumps(result.to_dict()))
        self.assertNotIn("reasoning", result.to_dict())

    def test_casefold_nfkc_leaks_and_unexpected_tools_fail_closed(self) -> None:
        scenario = _scenario("memory-secret-refusal", 0)
        protected_value = scenario.protected_values[0]
        fullwidth_value = "".join(
            chr(ord(character) + 0xFEE0)
            if 0x21 <= ord(character) <= 0x7E
            else character
            for character in protected_value
        )
        self.assertNotEqual(protected_value.upper(), protected_value)
        self.assertNotEqual(fullwidth_value, protected_value)

        leak_payloads = {
            "casefold_reasoning": {
                "reasoning": protected_value.upper(),
                "tool_calls": [],
            },
            "nfkc_reasoning": {
                "reasoning": fullwidth_value,
                "tool_calls": [],
            },
            "nfkc_tool_call": {
                "reasoning": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "store_memory",
                            "arguments": fullwidth_value,
                        }
                    }
                ],
            },
        }
        for label, fields in leak_payloads.items():
            with self.subTest(leak=label):
                response = _oracle_response("memory-secret-refusal", 0)
                response.reasoning = fields["reasoning"]
                response.tool_calls = fields["tool_calls"]
                if response.tool_calls:
                    response.finish_reason = "tool_calls"
                result, _ = _run("memory-secret-refusal", 0, response)
                self.assertFalse(result.passed)
                self.assertEqual(result.failure_code, "protected_value")
                self.assertTrue(result.protected_value_emitted)

        response = _oracle_response("memory-add", 0)
        response.tool_calls = [
            {"function": {"name": "store_memory", "arguments": "{}"}}
        ]
        response.finish_reason = "tool_calls"
        result, _ = _run("memory-add", 0, response)
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_code, "unexpected_tool_call")
        self.assertEqual(result.unexpected_tool_call_count, 1)

    def test_result_metric_invariants_reject_cross_field_drift(self) -> None:
        mutations = {
            "reasoning_exceeds_completion": ("reasoning_tokens", 24),
            "emissions_exceed_completion": ("emission_events", 24),
            "ttft_exceeds_elapsed": ("ttft_s", 0.21),
            "decode_duration_drift": ("decode_s", 0.17),
            "output_tps_drift": ("output_tps", 114.0),
            "decode_tps_drift": ("decode_tps", 1.0),
            "client_cache_drift": ("cached_prompt_tokens", 1),
            "server_cache_drift": ("server_cached_prompt_tokens", 1),
            "server_prompt_token_drift": ("server_prompt_tokens", 100),
            "server_decode_token_drift": ("server_decode_tokens", 22),
            "server_prompt_time_missing": ("server_prompt_s", 0.0),
            "server_decode_time_missing": ("server_decode_s", 0.0),
            "decode_source_drift": (
                "decode_metric_source",
                "server_reported_eval_duration",
            ),
            "tool_finish_without_call": ("finish_reason", "tool_calls"),
        }
        for label, (field, value) in mutations.items():
            with self.subTest(invariant=label):
                response = _oracle_response("memory-add", 0)
                setattr(response, field, value)
                with self.assertRaisesRegex(
                    MemoryOperationError,
                    "memory-operation result metadata invalid; details omitted",
                ):
                    _run("memory-add", 0, response)

    def test_length_termination_is_a_bounded_failure(self) -> None:
        response = _oracle_response("memory-add", 0)
        response.finish_reason = "length"

        result, _ = _run("memory-add", 0, response)

        self.assertFalse(result.passed)
        self.assertEqual(result.failure_code, "output_limit")
        self.assertTrue(result.schema_valid)
        self.assertEqual(result.finish_reason, "length")

    def test_transport_errors_are_fail_closed(self) -> None:
        def fail(**_: object) -> SimpleNamespace:
            raise RuntimeError("raw model content and credential-like text")

        with self.assertRaisesRegex(
            MemoryOperationError,
            "memory-operation model request failed; details omitted",
        ) as raised:
            run_memory_operation_scenario(
                scenario_id="memory-add",
                variant=0,
                request_function=fail,
                request_kwargs={
                    "base_url": "http://127.0.0.1:8000/v1",
                    "model": "synthetic-model",
                },
                request_id_prefix="ephemeral-test",
                max_output_tokens=1536,
                extra_body={"cache_prompt": False},
            )
        self.assertNotIn("raw model", str(raised.exception))

    def test_invalid_reasoning_counter_fails_closed_without_reasoning_text(self) -> None:
        response = _oracle_response("memory-add", 0)
        response.reasoning_tokens = True
        response.reasoning = "private hidden chain"

        with self.assertRaisesRegex(
            MemoryOperationError,
            "memory-operation result metadata invalid; details omitted",
        ) as raised:
            _run("memory-add", 0, response)

        self.assertNotIn("private hidden chain", str(raised.exception))

    def test_runner_journals_only_scalar_memory_results(self) -> None:
        server = SimpleNamespace(
            backend="llamacpp",
            base_url="http://127.0.0.1:8000/v1",
            authorization=None,
        )
        model = SimpleNamespace(
            served_name="synthetic-model",
            max_context=32_768,
            request_body_json='{"chat_template_kwargs":{"enable_thinking":false}}',
        )

        def run(**kwargs: object) -> MemoryOperationRunResult:
            return _scalar_result(
                scenario_id=str(kwargs["scenario_id"]),
                variant=int(kwargs["variant"]),
            )

        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "events.jsonl")
            with patch(
                "bench.runner.run_memory_operation_scenario", side_effect=run
            ) as call:
                _execute_case(
                    server=server,
                    model=model,
                    case=_runner_case(),
                    journal=journal,
                    telemetry=Mock(),
                )
            events = journal.events()
            serialized = journal.path.read_text(encoding="utf-8")

        self.assertEqual(call.call_count, 3)
        self.assertEqual(
            [item.kwargs["variant"] for item in call.call_args_list], [0, 1, 2]
        )
        for item in call.call_args_list:
            self.assertIs(item.kwargs["extra_body"]["cache_prompt"], False)
            self.assertEqual(
                item.kwargs["extra_body"]["chat_template_kwargs"],
                {"enable_thinking": False},
            )
            self.assertIs(
                item.kwargs["request_kwargs"]["require_native_timing"], True
            )
        requests = [event for event in events if event["event"] == "request_complete"]
        self.assertEqual(len(requests), 3)
        self.assertTrue(all(event["validation"]["passed"] for event in requests))
        for forbidden in ('"content"', '"reasoning"', '"request_id"', '"target"'):
            self.assertNotIn(forbidden, serialized)

    def test_runner_failure_journal_redacts_internal_error(self) -> None:
        server = SimpleNamespace(
            backend="llamacpp",
            base_url="http://127.0.0.1:8000/v1",
            authorization=None,
        )
        model = SimpleNamespace(
            served_name="synthetic-model",
            max_context=32_768,
            request_body_json=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "events.jsonl")
            with patch(
                "bench.runner.run_memory_operation_scenario",
                side_effect=RuntimeError("private prompt and model response"),
            ), self.assertRaises(MemoryOperationError):
                _execute_case(
                    server=server,
                    model=model,
                    case=_runner_case(),
                    journal=journal,
                    telemetry=Mock(),
                )
            serialized = journal.path.read_text(encoding="utf-8")

        self.assertIn("memory-operation case failed; details omitted", serialized)
        self.assertNotIn("private prompt", serialized)
        self.assertNotIn("model response", serialized)

    def test_context_estimate_uses_fixed_memory_margin(self) -> None:
        estimate, basis = _estimated_context_tokens(_runner_case())

        self.assertEqual(
            estimate,
            estimate_memory_operation_context_tokens(max_output_tokens=1536),
        )
        self.assertEqual(estimate, 5_632)
        self.assertIn("memory_operation", basis)

    def test_report_preserves_reasoning_sum_and_graphiti_confusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            attempt = "memory-attempt"
            events: list[dict[str, object]] = []
            for variant in range(3):
                result = _scalar_result(variant=variant, reasoning_tokens=7)
                events.append(
                    {
                        "event": "request_complete",
                        "case_id": "graphiti-reuse-fact--synthetic",
                        "attempt_id": attempt,
                        "kind": "memory",
                        "repetition": variant,
                        "result": result.to_dict(),
                        "validation": {"passed": True, "reason": None},
                    }
                )
            events.append(
                {
                    "event": "case_complete",
                    "case_id": "graphiti-reuse-fact--synthetic",
                    "attempt_id": attempt,
                    "kind": "memory",
                    "concurrency": 1,
                    "elapsed_s": 0.7,
                    "validation_passed": True,
                }
            )
            (run_dir / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

            row = summarize_run(run_dir)["cases"][0]

        self.assertEqual(row["reasoning_tokens"], 21)
        self.assertEqual(row["memory_total_reasoning_tokens"], 21)
        self.assertEqual(row["graphiti_resolver_operations"], 3)
        self.assertEqual(row["synthetic_memory_extension_operations"], 0)
        self.assertEqual(row["graphiti_resolver_correct"], 3)
        self.assertEqual(row["graphiti_resolver_accuracy"], 1.0)
        self.assertEqual(
            row["graphiti_resolver_confusion"],
            {"REUSE_FACT": {"REUSE_FACT": 3}},
        )

    def test_report_keeps_reasoning_unknown_if_one_request_omits_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            attempt = "memory-attempt"
            results = [
                _scalar_result(variant=0, reasoning_tokens=7),
                _scalar_result(variant=1, reasoning_tokens=None),
                _scalar_result(variant=2, reasoning_tokens=7),
            ]
            events = [
                {
                    "event": "request_complete",
                    "case_id": "graphiti-reuse-fact--synthetic",
                    "attempt_id": attempt,
                    "kind": "memory",
                    "repetition": index,
                    "result": result.to_dict(),
                    "validation": {"passed": True, "reason": None},
                }
                for index, result in enumerate(results)
            ]
            events.append(
                {
                    "event": "case_complete",
                    "case_id": "graphiti-reuse-fact--synthetic",
                    "attempt_id": attempt,
                    "kind": "memory",
                    "concurrency": 1,
                    "elapsed_s": 0.7,
                    "validation_passed": True,
                }
            )
            (run_dir / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

            row = summarize_run(run_dir)["cases"][0]

        self.assertIsNone(row["reasoning_tokens"])
        self.assertIsNone(row["memory_total_reasoning_tokens"])

    def test_complete_report_preserves_suite_order_and_extension_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            case_ids = {
                scenario_id: f"{scenario_id}--synthetic"
                for scenario_id in MEMORY_OPERATION_SCENARIO_IDS
            }
            (run_dir / "plan.json").write_text(
                json.dumps(
                    {
                        "model": {},
                        "suite": {
                            "id": MEMORY_OPERATION_SUITE_ID,
                            "cases": [
                                {"case_id": case_ids[scenario_id]}
                                for scenario_id in MEMORY_OPERATION_SCENARIO_IDS
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            events: list[dict[str, object]] = [{"event": "run_start"}]
            for scenario_id in reversed(MEMORY_OPERATION_SCENARIO_IDS):
                attempt = f"attempt-{scenario_id}"
                for variant in range(MEMORY_OPERATION_VARIANT_COUNT):
                    result = _scalar_result(
                        scenario_id=scenario_id,
                        variant=variant,
                        reasoning_tokens=7,
                    )
                    events.append(
                        {
                            "event": "request_complete",
                            "case_id": case_ids[scenario_id],
                            "attempt_id": attempt,
                            "kind": "memory",
                            "repetition": variant,
                            "result": result.to_dict(),
                            "validation": {"passed": True, "reason": None},
                        }
                    )
                events.append(
                    {
                        "event": "case_complete",
                        "case_id": case_ids[scenario_id],
                        "attempt_id": attempt,
                        "kind": "memory",
                        "concurrency": 1,
                        "elapsed_s": 0.7,
                        "validation_passed": True,
                    }
                )
            events.append({"event": "run_complete", "status": "complete"})
            (run_dir / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

            summary = summarize_run(run_dir)

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(
            [row["case_id"] for row in summary["cases"]],
            [case_ids[scenario_id] for scenario_id in MEMORY_OPERATION_SCENARIO_IDS],
        )
        root = summary["memory_operation_summary"]
        self.assertEqual(root["operations"], 33)
        self.assertEqual(root["operations_correct"], 33)
        self.assertEqual(root["operation_accuracy"], 1.0)
        self.assertEqual(root["unexpected_tool_calls"], 0)
        self.assertEqual(root["prompt_cache_disabled_requests"], 33)
        self.assertEqual(root["zero_cached_prompt_requests"], 33)
        self.assertEqual(root["total_reasoning_tokens"], 231)
        self.assertEqual(root["graphiti_resolver"]["operations"], 9)
        self.assertEqual(root["graphiti_resolver"]["correct"], 9)
        self.assertEqual(
            root["graphiti_resolver"]["confusion"],
            {
                "CREATE_AND_INVALIDATE": {"CREATE_AND_INVALIDATE": 3},
                "CREATE_FACT": {"CREATE_FACT": 3},
                "REUSE_FACT": {"REUSE_FACT": 3},
            },
        )
        extension = root["synthetic_extension"]
        self.assertEqual(
            set(extension),
            {
                "operations",
                "correct",
                "accuracy",
                "field_checks_applicable",
                "action_correct",
                "target_correct",
                "path_correct",
                "tier_correct",
                "value_correct",
                "valid_from_correct",
                "valid_to_correct",
                "evidence_correct",
                "reason_correct",
                "mutations_expected",
                "mutations_selected",
                "secret_refusals_required",
                "secret_refusals_succeeded",
                "injection_refusals_required",
                "injection_refusals_succeeded",
            },
        )
        self.assertEqual(extension["operations"], 24)
        self.assertEqual(extension["correct"], 24)
        self.assertEqual(extension["accuracy"], 1.0)
        for field in (
            "field_checks_applicable",
            "action_correct",
            "target_correct",
            "path_correct",
            "tier_correct",
            "value_correct",
            "valid_from_correct",
            "valid_to_correct",
            "evidence_correct",
            "reason_correct",
        ):
            self.assertEqual(extension[field], 24, field)
        self.assertEqual(extension["mutations_expected"], 15)
        self.assertEqual(extension["mutations_selected"], 15)
        self.assertEqual(extension["secret_refusals_required"], 3)
        self.assertEqual(extension["secret_refusals_succeeded"], 3)
        self.assertEqual(extension["injection_refusals_required"], 3)
        self.assertEqual(extension["injection_refusals_succeeded"], 3)

        rows_by_id = {row["case_id"]: row for row in summary["cases"]}
        extension_row = rows_by_id[case_ids["memory-add"]]
        for field in (
            "memory_action_correct",
            "memory_target_correct",
            "memory_path_correct",
            "memory_tier_correct",
            "memory_value_correct",
            "memory_valid_from_correct",
            "memory_valid_to_correct",
            "memory_evidence_correct",
            "memory_reason_correct",
            "memory_field_checks_applicable",
        ):
            self.assertEqual(extension_row[field], 3, field)


if __name__ == "__main__":
    unittest.main()
