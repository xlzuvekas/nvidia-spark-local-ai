from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.client import BenchmarkRequestError, RequestResult, stream_chat_request
from bench.evidence import (
    EvidenceError,
    SCHEMA_VERSION,
    _project_case,
    _project_request_result,
    _project_requests,
    _project_summary,
    _validate_prefix_cache_aggregates,
    _write_bundle,
    export_evidence,
    verify_evidence,
)
from bench.journal import Journal
from bench.llamacpp_cache_metrics import (
    LlamaCppCacheMetricsError,
    delta_llamacpp_cache_metrics,
    parse_llamacpp_cache_metrics,
    require_llamacpp_cache_delta,
)
from bench.manifest import (
    CaseSpec,
    ManifestError,
    SuiteSpec,
    load_models,
    load_suite,
    validate_benchmark_selection,
    validate_model,
    validate_suite,
)
from bench.prefix_cache_protocol import (
    PREFIX_CACHE_CONTEXT_TOKENS,
    PREFIX_CACHE_PREFIX_TARGETS,
    prefix_cache_llamacpp_args,
    prefix_cache_steps,
)
from bench.report import summarize_run
from bench.runner import (
    PreflightError,
    PrefixCacheError,
    _execute_case,
    _estimated_context_tokens,
    _prefix_cache_prompt,
    _prefix_cache_shared_prefix,
    _prefix_cache_steps,
    _validate_prefix_cache_plan_selection,
    _validate_prefix_cache_result,
)
from sparkbench import command_matrix
from tests.test_evidence import EvidenceFixture


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "manifests" / "suites" / "llamacpp_prefix_cache.toml"


class _MockResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self) -> _MockResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):
        return iter(self._lines)


def _sse(payload: dict[str, object]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _cache_suite_document() -> dict[str, object]:
    """Return the frozen two-target cache suite as it appears in a plan."""

    cases: list[dict[str, object]] = []
    for case_id, target in (
        ("llamacpp-prefix-cache-8192--a1b2c3d4e5f6", 8192),
        ("llamacpp-prefix-cache-32768--b1c2d3e4f5a6", 32768),
    ):
        cases.append(
            {
                "case_id": case_id,
                "id": f"llamacpp-prefix-cache-{target}",
                "kind": "cache",
                "requires": ["chat"],
                "warmups": 0,
                "repetitions": 5,
                "max_output_tokens": 128,
                "max_turns": 1,
                "temperature": 0.0,
                "concurrency": 1,
                "prompt_repetitions": target,
            }
        )
    return {
        "id": "llamacpp-prefix-cache",
        "schema_version": 1,
        "cases": cases,
    }


def _cache_model_document(mode: str) -> dict[str, object]:
    return {
        "id": f"synthetic-prefix-cache-{mode}",
        "backend": "llamacpp",
        "architecture": "synthetic",
        "quantization": "q4_k_m",
        "source": "synthetic/prefix-cache",
        "revision": "a" * 40,
        "tasks": ["chat"],
        "lifecycle": "managed",
        "prefix_cache_mode": mode,
        "runtime_parallel": 1,
        "max_context": 262144,
        "native_context": 262144,
        "startup_timeout_s": 600,
        "support_status": "supported",
    }


def _disabled_llamacpp_speculative_summary() -> dict[str, object]:
    """The only generic draft rollup a cache source may discard."""

    return {
        "accepted_tokens_per_position": {},
        "configured_max_draft_tokens": None,
        "draft_acceptance_rate": None,
        "mean_accepted_length": None,
        "method": None,
        "num_accepted_tokens": 0,
        "num_draft_tokens": 0,
        "num_drafts": 0,
        "proposal_depth": None,
        "requested": False,
        "scope": "all_persisted_llamacpp_server_lifetimes",
        "snapshot_count": 1,
        "source": "llamacpp_prometheus_cumulative_counters",
    }


def _cache_scalar_result(
    *, mode: str, pair_index: int, step_ordinal: int, prefix_target: int
) -> dict[str, object]:
    condition, _, control = prefix_cache_steps(mode)[step_ordinal - 1]
    cached_tokens = 90 if condition == "warm-prefix-hit" else 0
    server_prompt_tokens = 100 - cached_tokens
    # Prometheus prompt counters are server-global.  This deliberate +2
    # diagnostic offset must not weaken final SSE request reconciliation.
    prometheus_global_prompt_tokens = server_prompt_tokens + 2
    prompt_s = 0.1 if cached_tokens else 1.0
    elapsed_s = 0.25 if cached_tokens else 2.0
    return {
        "cache_condition": condition,
        "cache_pair_index": pair_index,
        "cache_prompt_control": control,
        "cache_prefix_target_words": prefix_target,
        "cache_profile_mode": mode,
        "cache_step_ordinal": step_ordinal,
        "cached_prompt_tokens": cached_tokens,
        "completion_tokens": 128,
        "decode_metric_source": "client_estimate",
        "decode_s": 1.0,
        "decode_tps": 127.0,
        "elapsed_s": elapsed_s,
        "emission_events": 128,
        "finish_reason": "length",
        "prometheus_global_cached_prompt_tokens": cached_tokens,
        "prometheus_global_decode_s": 1.0,
        "prometheus_global_decode_tokens": 128,
        "prometheus_global_prompt_s": prompt_s,
        "prometheus_global_prompt_tokens": prometheus_global_prompt_tokens,
        "output_tps": 128.0 / elapsed_s,
        "prompt_tokens": 100,
        "reasoning_tokens": None,
        "server_cached_prompt_tokens": cached_tokens,
        "server_decode_s": 1.0,
        "server_decode_tokens": 128,
        "server_prompt_s": prompt_s,
        "server_prompt_tokens": server_prompt_tokens,
        "ttft_s": 0.1 if cached_tokens else 1.0,
    }


def _cache_events(mode: str, suite: dict[str, object]) -> list[dict[str, object]]:
    """Build synthetic scalar-only events for an offline evidence fixture."""

    events: list[dict[str, object]] = []
    cases = suite["cases"]
    assert isinstance(cases, list)
    for case in cases:
        assert isinstance(case, dict)
        case_id = str(case["case_id"])
        target = int(case["prompt_repetitions"])
        attempt_id = f"synthetic-cache-attempt-{target}"
        events.append(
            {
                "event": "case_start",
                "case_id": case_id,
                "attempt_id": attempt_id,
                "kind": "cache",
                "concurrency": 1,
            }
        )
        case_elapsed_s = 0.1
        for pair_index in range(1, 6):
            for step_ordinal in range(1, 4):
                result = _cache_scalar_result(
                    mode=mode,
                    pair_index=pair_index,
                    step_ordinal=step_ordinal,
                    prefix_target=target,
                )
                case_elapsed_s += float(result["elapsed_s"])
                events.append(
                    {
                        "event": "request_complete",
                        "case_id": case_id,
                        "attempt_id": attempt_id,
                        "kind": "cache",
                        "repetition": pair_index - 1,
                        "burst_elapsed_s": result["elapsed_s"],
                        "result": result,
                        "validation": {"passed": True, "reason": None},
                    }
                )
        events.append(
            {
                "event": "case_complete",
                "case_id": case_id,
                "attempt_id": attempt_id,
                "kind": "cache",
                "concurrency": 1,
                "elapsed_s": case_elapsed_s,
                "validation_passed": True,
            }
        )
    return events


def _fake_campaign_export(
    campaign: Path,
    _results_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Keep the cache fixture focused on run evidence, not campaign fixtures."""

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


def _write_test_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _refresh_run_and_root_checksums(output_root: Path, run_id: str) -> None:
    """Refresh checksums after a deliberate semantic verifier tamper."""

    bundle = output_root / "runs" / run_id
    bundle_checksums = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(bundle.iterdir())
        if path.name != "checksums.json"
    }
    _write_test_json(
        bundle / "checksums.json",
        {"files": bundle_checksums, "schema_version": SCHEMA_VERSION},
    )
    index_path = output_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = next(item for item in index["runs"] if item["run_id"] == run_id)
    entry["bundle_sha256"] = hashlib.sha256(
        (bundle / "checksums.json").read_bytes()
    ).hexdigest()
    _write_test_json(index_path, index)
    # The root checksum file is the only excluded file; child bundle checksum
    # files must be included, so filter by the resolved top-level path.
    root_checksums = {
        str(path.relative_to(output_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path != output_root / "checksums.json"
    }
    _write_test_json(
        output_root / "checksums.json",
        {"files": root_checksums, "schema_version": SCHEMA_VERSION},
    )


class LlamaCppPrefixCacheMetricsTests(unittest.TestCase):
    def test_parser_and_delta_accept_global_prometheus_offset(self) -> None:
        before = parse_llamacpp_cache_metrics(
            """
llamacpp:prompt_tokens_total 100
llamacpp:prompt_tokens_cached_total 0
llamacpp:prompt_seconds_total 1.25
llamacpp:tokens_predicted_total 128
llamacpp:tokens_predicted_seconds_total 2.5
"""
        )
        after = parse_llamacpp_cache_metrics(
            """
llamacpp:prompt_tokens_total 112
llamacpp:prompt_tokens_cached_total 90
llamacpp:prompt_seconds_total 1.35
llamacpp:tokens_predicted_total 256
llamacpp:tokens_predicted_seconds_total 3.5
"""
        )
        delta = delta_llamacpp_cache_metrics(before, after)
        self.assertEqual(delta["prompt_tokens"], 12)
        self.assertEqual(delta["cached_prompt_tokens"], 90)
        self.assertAlmostEqual(float(delta["prompt_s"]), 0.1)
        self.assertEqual(delta["decode_tokens"], 128)
        self.assertEqual(delta["decode_s"], 1)
        require_llamacpp_cache_delta(delta)

    def test_parser_and_delta_fail_closed(self) -> None:
        self.assertIsNone(
            parse_llamacpp_cache_metrics(
                "llamacpp:prompt_tokens_total 1\n"
            )
        )
        self.assertIsNone(
            parse_llamacpp_cache_metrics(
                """
llamacpp:prompt_tokens_total 1
llamacpp:prompt_tokens_cached_total -1
llamacpp:prompt_seconds_total 1
llamacpp:tokens_predicted_total 1
llamacpp:tokens_predicted_seconds_total 1
"""
            )
        )
        with self.assertRaises(LlamaCppCacheMetricsError):
            delta_llamacpp_cache_metrics(
                {
                    "prompt_tokens": 10,
                    "cached_prompt_tokens": 0,
                    "prompt_s": 1,
                    "decode_tokens": 1,
                    "decode_s": 1,
                },
                {
                    "prompt_tokens": 9,
                    "cached_prompt_tokens": 0,
                    "prompt_s": 1,
                    "decode_tokens": 1,
                    "decode_s": 1,
                },
            )


class PrefixCacheClientTests(unittest.TestCase):
    def test_stream_client_parses_usage_cache_and_native_timings(self) -> None:
        response = _MockResponse(
            [
                _sse({"choices": [{"delta": {"content": "output"}}]}),
                _sse(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 128,
                            "prompt_tokens_details": {"cached_tokens": 90},
                        },
                        "timings": {
                            "cache_n": 90,
                            "prompt_n": 10,
                            "prompt_ms": 25.0,
                            "predicted_n": 128,
                            "predicted_ms": 500.0,
                        },
                    }
                ),
                b"data: [DONE]\n\n",
            ]
        )
        with patch("bench.client.urllib.request.urlopen", return_value=response):
            result = stream_chat_request(
                base_url="http://127.0.0.1:9999/v1",
                model="synthetic",
                prompt="synthetic private prompt",
                max_tokens=128,
                temperature=0.0,
                request_id="synthetic-request-id",
                require_native_cache_metrics=True,
                require_native_timing=True,
            )
        self.assertEqual(result.cached_prompt_tokens, 90)
        self.assertEqual(result.server_prompt_tokens, 10)
        self.assertEqual(result.server_cached_prompt_tokens, 90)
        self.assertEqual(result.server_decode_tokens, 128)
        self.assertAlmostEqual(result.server_prompt_s or 0, 0.025)
        self.assertAlmostEqual(result.server_decode_s or 0, 0.5)

    def test_stream_client_rejects_mismatched_or_missing_native_cache_counters(self) -> None:
        invalid_events = [
            _sse({"choices": [{"delta": {"content": "output"}}]}),
            _sse(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 128,
                        "prompt_tokens_details": {"cached_tokens": 90},
                    },
                    "timings": {
                        "cache_n": 89,
                        "prompt_n": 11,
                        "prompt_ms": 25.0,
                        "predicted_n": 128,
                        "predicted_ms": 500.0,
                    },
                }
            ),
            b"data: [DONE]\n\n",
        ]
        with patch(
            "bench.client.urllib.request.urlopen",
            return_value=_MockResponse(invalid_events),
        ), self.assertRaisesRegex(BenchmarkRequestError, "cache counters disagreed"):
            stream_chat_request(
                base_url="http://127.0.0.1:9999/v1",
                model="synthetic",
                prompt="synthetic private prompt",
                max_tokens=128,
                temperature=0.0,
                request_id="synthetic-request-id",
                require_native_cache_metrics=True,
                require_native_timing=True,
            )

    def test_stream_client_rejects_coerced_native_usage_counters(self) -> None:
        for invalid_prompt_tokens in ("100", True, 100.0, 100.5):
            with self.subTest(prompt_tokens=invalid_prompt_tokens):
                response = _MockResponse(
                    [
                        _sse({"choices": [{"delta": {"content": "output"}}]}),
                        _sse(
                            {
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": invalid_prompt_tokens,
                                    "completion_tokens": 128,
                                    "prompt_tokens_details": {
                                        "cached_tokens": 0
                                    },
                                },
                                "timings": {
                                    "cache_n": 0,
                                    "prompt_n": 100,
                                    "prompt_ms": 25.0,
                                    "predicted_n": 128,
                                    "predicted_ms": 500.0,
                                },
                            }
                        ),
                        b"data: [DONE]\n\n",
                    ]
                )
                with patch(
                    "bench.client.urllib.request.urlopen", return_value=response
                ), self.assertRaisesRegex(BenchmarkRequestError, "exact prompt_tokens"):
                    stream_chat_request(
                        base_url="http://127.0.0.1:9999/v1",
                        model="synthetic",
                        prompt="synthetic private prompt",
                        max_tokens=128,
                        temperature=0.0,
                        request_id="synthetic-request-id",
                        require_native_cache_metrics=True,
                        require_native_timing=True,
                    )

    def test_stream_client_rejects_integral_float_cache_and_timing_counts(self) -> None:
        for field, value in (
            ("cached_tokens", 90.0),
            ("cache_n", 90.0),
            ("prompt_n", 10.0),
            ("predicted_n", 128.0),
        ):
            with self.subTest(field=field):
                details: dict[str, object] = {"cached_tokens": 90}
                timings: dict[str, object] = {
                    "cache_n": 90,
                    "prompt_n": 10,
                    "prompt_ms": 25.0,
                    "predicted_n": 128,
                    "predicted_ms": 500.0,
                }
                if field == "cached_tokens":
                    details[field] = value
                else:
                    timings[field] = value
                response = _MockResponse(
                    [
                        _sse({"choices": [{"delta": {"content": "output"}}]}),
                        _sse(
                            {
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": 100,
                                    "completion_tokens": 128,
                                    "prompt_tokens_details": details,
                                },
                                "timings": timings,
                            }
                        ),
                        b"data: [DONE]\n\n",
                    ]
                )
                with patch(
                    "bench.client.urllib.request.urlopen", return_value=response
                ), self.assertRaises(BenchmarkRequestError):
                    stream_chat_request(
                        base_url="http://127.0.0.1:9999/v1",
                        model="synthetic",
                        prompt="synthetic private prompt",
                        max_tokens=128,
                        temperature=0.0,
                        request_id="synthetic-request-id",
                        require_native_cache_metrics=True,
                        require_native_timing=True,
                    )


class PrefixCacheProtocolTests(unittest.TestCase):
    def _case(self) -> SimpleNamespace:
        return SimpleNamespace(
            id="llamacpp-prefix-cache-8192",
            case_id="llamacpp-prefix-cache-8192--a1b2c3d4e5f6",
            kind="cache",
            requires=("chat",),
            warmups=0,
            repetitions=5,
            max_output_tokens=128,
            temperature=0.0,
            concurrency=1,
            prompt_repetitions=8192,
        )

    def test_manifest_profiles_are_explicit_matched_single_slot_controls(self) -> None:
        suite = load_suite(SUITE_PATH)
        self.assertEqual(suite.id, "llamacpp-prefix-cache")
        self.assertEqual(
            [(case.kind, case.prompt_repetitions, case.repetitions) for case in suite.cases],
            [("cache", 8192, 5), ("cache", 32768, 5)],
        )
        models = load_models(ROOT / "manifests" / "models.toml")
        pairs = (
            (
                "qwen36-35b-a3b-ud-q4-k-xl-llamacpp-prefix-cache-off",
                "qwen36-35b-a3b-ud-q4-k-xl-llamacpp-prefix-cache-on",
            ),
            (
                "qwen38-27b-ud-q4-k-xl-llamacpp-long-context-prefix-cache-off",
                "qwen38-27b-ud-q4-k-xl-llamacpp-long-context-prefix-cache-on",
            ),
        )
        for off_id, on_id in pairs:
            with self.subTest(model=on_id):
                off = models[off_id]
                on = models[on_id]
                self.assertEqual(off.runtime_parallel, 1)
                self.assertEqual(on.runtime_parallel, 1)
                self.assertEqual(off.max_context, 262144)
                self.assertEqual(on.max_context, 262144)
                self.assertEqual(off.native_context, 262144)
                self.assertEqual(on.native_context, 262144)
                self.assertEqual(off.prefix_cache_mode, "off")
                self.assertEqual(on.prefix_cache_mode, "on")
                self.assertEqual(tuple(off.args), prefix_cache_llamacpp_args("off"))
                self.assertEqual(tuple(on.args), prefix_cache_llamacpp_args("on"))
                self.assertIn("--no-cache-prompt", off.args)
                self.assertIn("--cache-prompt", on.args)
                self.assertNotIn("--cache-reuse", off.args)
                self.assertNotIn("--cache-reuse", on.args)
                self.assertEqual(off.source, on.source)
                self.assertEqual(off.revision, on.revision)
                self.assertEqual(off.model_digest, on.model_digest)
                self.assertEqual(off.runtime_digest, on.runtime_digest)
                self.assertEqual(off.runtime_revision, on.runtime_revision)
                with self.assertRaises(ManifestError):
                    validate_model(replace(on, prefix_cache_mode="off"))
                with self.assertRaises(ManifestError):
                    validate_model(replace(on, args=(*on.args, "--cache-reuse", "64")))
                with self.assertRaises(ManifestError):
                    validate_model(
                        replace(on, args=(*on.args[:-1], "--cache-prompt=true"))
                    )
                validate_benchmark_selection(on, suite)
                with self.assertRaises(ManifestError):
                    validate_benchmark_selection(
                        off,
                        load_suite(ROOT / "manifests" / "suites" / "smoke.toml"),
                    )
                with self.assertRaises(ManifestError):
                    validate_benchmark_selection(
                        models["qwen36-35b-a3b-ud-q4-k-xl-llamacpp"], suite
                    )

        with self.assertRaises(ManifestError):
            validate_suite(
                SuiteSpec(
                    id="llamacpp-prefix-cache",
                    cases=(
                        CaseSpec(
                            id="ordinary",
                            kind="decode",
                            requires=("chat",),
                            max_output_tokens=8,
                        ),
                    ),
                )
            )

    def test_prompt_is_shared_before_a_suffix_only_nonce(self) -> None:
        case = self._case()
        shared = _prefix_cache_shared_prefix(case, 3)
        first = _prefix_cache_prompt(case, 3, "private-request-a")
        second = _prefix_cache_prompt(case, 3, "private-request-b")
        self.assertTrue(first.startswith(shared))
        self.assertTrue(second.startswith(shared))
        self.assertNotEqual(first, second)
        self.assertNotIn("private-request-a", shared)
        self.assertNotIn("private-request-b", shared)
        self.assertNotEqual(
            _prefix_cache_shared_prefix(case, 2),
            _prefix_cache_shared_prefix(case, 3),
        )
        other_profile_case = SimpleNamespace(
            **{**vars(case), "case_id": "other-profile-frozen-case"}
        )
        self.assertEqual(
            _prefix_cache_shared_prefix(case, 3),
            _prefix_cache_shared_prefix(other_profile_case, 3),
        )

    def test_false_cold_seed_contract_allows_a_following_profile_default_hit(self) -> None:
        steps = _prefix_cache_steps("on")
        self.assertEqual(steps, prefix_cache_steps("on"))
        self.assertEqual(
            steps,
            (
                ("forced-cold-a", False, "force-off"),
                ("forced-cold-b", False, "force-off"),
                ("warm-prefix-hit", None, "profile-default"),
            ),
        )
        # b10453's false path sets n_past=0, processes the full prompt, and
        # retains that slot state; the following profile-default true request
        # can therefore reuse B's shared prefix.  This mock state contract
        # guards against accidentally changing B or warm's control mode.
        seeded = False
        cached_counts: list[int] = []
        for _, cache_prompt, _ in steps:
            effective_cache_prompt = cache_prompt is not False
            cached_counts.append(90 if effective_cache_prompt and seeded else 0)
            seeded = True
        self.assertEqual(cached_counts, [0, 0, 90])

    def test_cache_context_estimate_is_conservative(self) -> None:
        estimated, basis = _estimated_context_tokens(self._case())
        self.assertEqual(
            basis,
            "prefix_cache_words_times_six_plus_output_and_template_margin",
        )
        self.assertEqual(estimated, 8192 * 6 + 128 + 1024)
        self.assertGreater(estimated, self._case().prompt_repetitions)

    def test_runner_rejects_boolean_server_timing_scalars(self) -> None:
        result = RequestResult(
            request_id="synthetic-request-id",
            started_at_ns=1,
            prompt_tokens=100,
            completion_tokens=128,
            reasoning_tokens=None,
            ttft_s=0.1,
            elapsed_s=1.0,
            decode_s=0.9,
            decode_tps=127.0,
            output_tps=128.0,
            emission_events=128,
            finish_reason="length",
            response_model="synthetic",
            content="",
            reasoning="",
            tool_calls=[],
            cached_prompt_tokens=0,
            server_prompt_tokens=100,
            server_cached_prompt_tokens=0,
            server_decode_tokens=128,
            server_prompt_s=True,
            server_decode_s=1.0,
        )
        with self.assertRaises(PrefixCacheError):
            _validate_prefix_cache_result(
                result=result,
                prometheus_metrics={
                    "prompt_tokens": 100,
                    "cached_prompt_tokens": 0,
                    "prompt_s": 1.0,
                    "decode_tokens": 128,
                    "decode_s": 1.0,
                },
                case=self._case(),
                condition="forced-cold-a",
            )

    def test_runner_uses_reconciled_sse_counters_not_global_prometheus_tokens(
        self,
    ) -> None:
        result = RequestResult(
            request_id="synthetic-request-id",
            started_at_ns=1,
            prompt_tokens=100,
            completion_tokens=128,
            reasoning_tokens=None,
            ttft_s=0.1,
            elapsed_s=1.0,
            decode_s=0.9,
            decode_tps=127 / 0.9,
            output_tps=128.0,
            emission_events=128,
            finish_reason="length",
            response_model="synthetic",
            content="",
            reasoning="",
            tool_calls=[],
            cached_prompt_tokens=0,
            server_prompt_tokens=100,
            server_cached_prompt_tokens=0,
            server_decode_tokens=128,
            server_prompt_s=1.0,
            server_decode_s=1.0,
        )
        prometheus_metrics = {
            # A global/batch counter can differ from the per-request final SSE
            # prompt counter even while the exact SSE identities hold.
            "prompt_tokens": 102,
            "cached_prompt_tokens": 0,
            "prompt_s": 1.0,
            "decode_tokens": 128,
            "decode_s": 1.0,
        }
        validation = _validate_prefix_cache_result(
            result=result,
            prometheus_metrics=prometheus_metrics,
            case=self._case(),
            condition="forced-cold-a",
        )
        self.assertTrue(validation["passed"])
        self.assertFalse(
            _validate_prefix_cache_result(
                result=replace(result, server_prompt_tokens=102),
                prometheus_metrics=prometheus_metrics,
                case=self._case(),
                condition="forced-cold-a",
            )["passed"]
        )

    def test_runner_journals_scalar_only_cache_measurements_and_reports_pair_metrics(self) -> None:
        case = self._case()
        request_ids: list[str] = []
        prompts: list[str] = []
        request_controls: list[dict[str, object]] = []

        def run(mode: str) -> tuple[list[dict[str, object]], dict[str, object]]:
            counters: dict[str, float] = {
                "prompt_tokens": 0.0,
                "cached_prompt_tokens": 0.0,
                "prompt_s": 0.0,
                "decode_tokens": 0.0,
                "decode_s": 0.0,
            }

            def snapshot(_: str) -> dict[str, float]:
                return dict(counters)

            def request(**kwargs: object) -> RequestResult:
                extra_body = kwargs["extra_body"]
                assert isinstance(extra_body, dict)
                self.assertEqual(extra_body.get("id_slot"), 0)
                request_controls.append(dict(extra_body))
                cache_override = extra_body.get("cache_prompt")
                self.assertTrue(
                    cache_override is None or isinstance(cache_override, bool)
                )
                cache_prompt = (
                    cache_override is not False if mode == "on" else False
                )
                request_id = str(kwargs["request_id"])
                prompt = str(kwargs["prompt"])
                request_ids.append(request_id)
                prompts.append(prompt)
                cached = 90 if cache_prompt else 0
                physical = 100 - cached
                prompt_s = 0.1 if cache_prompt else 1.0
                elapsed_s = 0.3 if cache_prompt else 2.2
                # Prometheus counters are global batch diagnostics, not
                # request-scoped accounting.  Model a realistic +2 drift.
                counters["prompt_tokens"] += physical + 2
                counters["cached_prompt_tokens"] += cached
                counters["prompt_s"] += prompt_s
                counters["decode_tokens"] += 128
                counters["decode_s"] += 1.0
                return RequestResult(
                    request_id=request_id,
                    started_at_ns=1,
                    prompt_tokens=100,
                    completion_tokens=128,
                    reasoning_tokens=None,
                    ttft_s=0.1 if cache_prompt else 2.0,
                    elapsed_s=elapsed_s,
                    decode_s=1.0,
                    decode_tps=127.0,
                    output_tps=128 / elapsed_s,
                    emission_events=128,
                    finish_reason="length",
                    response_model="synthetic-private-model",
                    content="prefix-cache-sensitive-completion",
                    reasoning="prefix-cache-sensitive-reasoning",
                    tool_calls=[],
                    cached_prompt_tokens=cached,
                    server_prompt_tokens=physical,
                    server_cached_prompt_tokens=cached,
                    server_decode_tokens=128,
                    server_prompt_s=prompt_s,
                    server_decode_s=1.0,
                )

            server = SimpleNamespace(
                backend="llamacpp", base_url="http://127.0.0.1:8080/v1"
            )
            model = SimpleNamespace(
                prefix_cache_mode=mode,
                runtime_parallel=1,
                args=("--cache-prompt" if mode == "on" else "--no-cache-prompt",),
                served_name="synthetic",
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "events.jsonl"
                journal = Journal(path)
                with (
                    patch("bench.runner.snapshot_llamacpp_cache_metrics", side_effect=snapshot),
                    patch("bench.runner.stream_chat_request", side_effect=request),
                    patch("bench.runner.time.time_ns", return_value=777),
                ):
                    _execute_case(
                        server=server,
                        model=model,
                        case=case,
                        journal=journal,
                        telemetry=Mock(),
                    )
                events = journal.events()
                serialized = path.read_text(encoding="utf-8")
                summary = summarize_run(Path(directory))
            return events, {"serialized": serialized, "summary": summary}

        summaries: dict[str, dict[str, object]] = {}
        for mode, expected_requests in (("off", 15), ("on", 15)):
            with self.subTest(mode=mode):
                events, artifacts = run(mode)
                measured = [event for event in events if event["event"] == "request_complete"]
                self.assertEqual(len(measured), expected_requests)
                self.assertTrue(all(event["validation"]["passed"] for event in measured))
                for event in measured:
                    result = event["result"]
                    self.assertNotIn("request_id", result)
                    self.assertNotIn("content", result)
                    self.assertNotIn("reasoning", result)
                    self.assertNotIn("tool_calls", result)
                    self.assertNotIn("response_model", result)
                    self.assertEqual(
                        result["prometheus_global_prompt_tokens"],
                        result["server_prompt_tokens"] + 2,
                    )
                    _project_request_result(result, kind="cache")
                if mode == "off":
                    self.assertTrue(
                        all("cache_prompt" not in control for control in request_controls)
                    )
                else:
                    controls = [
                        event["result"]["cache_prompt_control"] for event in measured
                    ]
                    self.assertEqual(
                        controls,
                        ["force-off", "force-off", "profile-default"] * 5,
                    )
                serialized = str(artifacts["serialized"])
                for sensitive in (
                    "prefix-cache-sensitive-completion",
                    "prefix-cache-sensitive-reasoning",
                    *request_ids,
                ):
                    self.assertNotIn(sensitive, serialized)
                self.assertTrue(all("Request suffix nonce" in prompt for prompt in prompts))
                row = artifacts["summary"]["cases"][0]  # type: ignore[index]
                cache_summary = row["prefix_cache"]  # type: ignore[index]
                self.assertEqual(cache_summary["profile_mode"], mode)
                self.assertGreater(cache_summary["case_session_wall_s"], 0)
                self.assertNotIn("median_ttft_s", row)
                self.assertNotIn("aggregate_output_tps", row)
                third = cache_summary["conditions"][-1]
                if mode == "on":
                    self.assertEqual(third["cache_condition"], "warm-prefix-hit")
                    self.assertGreaterEqual(third["cache_hit_fraction"], 0.90)
                else:
                    self.assertEqual(third["cache_condition"], "forced-cold-c")
                    self.assertEqual(third["cache_hit_fraction"], 0)
                self.assertEqual(
                    cache_summary["paired_second_to_third"]["paired_blocks"], 5
                )
                projected_case = _project_case(row)
                self.assertIn("prefix_cache", projected_case)
                summaries[mode] = row

class PrefixCacheEvidenceTests(unittest.TestCase):
    def _write_cache_run(
        self, run_dir: Path, *, mode: str, with_startup_provenance: bool = False
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        list[dict[str, object]],
        dict[str, object],
    ]:
        suite = _cache_suite_document()
        model = _cache_model_document(mode)
        plan = {
            "schema_version": 1,
            "model": {
                **model,
                "model_file": "synthetic-prefix-cache.gguf",
                "model_digest": "b" * 64,
                "model_size_bytes": 1024,
                "runtime_binary": "/synthetic/llama-server",
                "runtime_digest": "c" * 64,
                "runtime_revision": "d" * 40,
            },
            "suite": suite,
        }
        events: list[dict[str, object]] = [{"event": "run_start"}]
        if with_startup_provenance:
            events.extend(
                (
                    {
                        "event": "artifact_validation_complete",
                        "elapsed_s": 0.1,
                        "runtime_binary_sha256": "a" * 64,
                        "model_sha256": "b" * 64,
                    },
                    {
                        "event": "first_request_complete",
                        "result": {
                            "completion_tokens": 1,
                            "decode_metric_source": "client_estimate",
                            "decode_s": 0.1,
                            "decode_tps": 0.0,
                            "elapsed_s": 0.2,
                            "emission_events": 1,
                            "finish_reason": "length",
                            "output_tps": 5.0,
                            "prompt_tokens": 1,
                            "reasoning_tokens": None,
                            "ttft_s": 0.1,
                        },
                    },
                )
            )
        events.extend((*_cache_events(mode, suite), {"event": "run_complete", "status": "completed"}))
        _write_test_json(run_dir / "plan.json", plan)
        (run_dir / "events.jsonl").write_text(
            "".join(
                json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                for event in events
            ),
            encoding="utf-8",
        )
        telemetry_path = run_dir / "telemetry.jsonl"
        if telemetry_path.exists():
            telemetry_path.unlink()
        return suite, model, events, summarize_run(run_dir)

    def test_source_cache_case_telemetry_and_energy_metric_are_discarded(self) -> None:
        telemetry = {
            "sampled_energy_j": 4.0,
            "samples": 5,
        }
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            _, _, _, source_summary = self._write_cache_run(
                fixture.run_dir, mode="on", with_startup_provenance=True
            )
            source_summary["cases"][0]["telemetry"] = telemetry
            source_summary["cases"][0]["output_tokens_per_sampled_joule"] = 480.0
            _write_test_json(fixture.run_dir / "summary.json", source_summary)

            projected = _project_summary(source_summary)
            self.assertNotIn("telemetry", projected["cases"][0])
            self.assertNotIn(
                "output_tokens_per_sampled_joule", projected["cases"][0]
            )

            with patch(
                "bench.evidence._export_campaign", side_effect=_fake_campaign_export
            ):
                export_evidence(
                    results_root=fixture.results,
                    output_root=fixture.output,
                )
            self.assertEqual("verified", verify_evidence(fixture.output)["status"])
            exported_summary = json.loads(
                (
                    fixture.output / "runs" / fixture.run_id / "summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertNotIn(
                "telemetry", exported_summary["aggregates"]["cases"][0]
            )
            self.assertNotIn(
                "output_tokens_per_sampled_joule",
                exported_summary["aggregates"]["cases"][0],
            )

            malformed = copy.deepcopy(source_summary)
            malformed["cases"][0]["telemetry"] = "synthetic-cache-telemetry"
            with self.assertRaisesRegex(
                EvidenceError, "prefix-cache case.telemetry must be an object"
            ):
                _project_summary(malformed)

            unknown = copy.deepcopy(source_summary)
            unknown["cases"][0]["telemetry"] = {"synthetic_trace": 1}
            with self.assertRaisesRegex(
                EvidenceError, "unknown prefix-cache case.telemetry fields"
            ):
                _project_summary(unknown)

            negative_energy_metric = copy.deepcopy(source_summary)
            negative_energy_metric["cases"][0]["output_tokens_per_sampled_joule"] = -1.0
            with self.assertRaisesRegex(
                EvidenceError,
                "prefix-cache case.output_tokens_per_sampled_joule must be finite and nonnegative",
            ):
                _project_summary(negative_energy_metric)

    def test_source_disabled_speculative_summary_is_discarded(self) -> None:
        """Cache export accepts only the known no-draft llama.cpp rollup."""

        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            _, _, _, source_summary = self._write_cache_run(
                fixture.run_dir, mode="on", with_startup_provenance=True
            )
            source_summary["speculative_decoding"] = (
                _disabled_llamacpp_speculative_summary()
            )
            _write_test_json(fixture.run_dir / "summary.json", source_summary)

            projected = _project_summary(source_summary)
            self.assertNotIn("speculative_decoding", projected)

            with patch(
                "bench.evidence._export_campaign", side_effect=_fake_campaign_export
            ):
                export_evidence(
                    results_root=fixture.results,
                    output_root=fixture.output,
                )
            self.assertEqual("verified", verify_evidence(fixture.output)["status"])
            exported_summary = json.loads(
                (
                    fixture.output / "runs" / fixture.run_id / "summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertNotIn(
                "speculative_decoding", exported_summary["aggregates"]
            )

            retained = copy.deepcopy(exported_summary)
            retained["aggregates"]["speculative_decoding"] = (
                _disabled_llamacpp_speculative_summary()
            )
            exported_summary_path = (
                fixture.output / "runs" / fixture.run_id / "summary.json"
            )
            _write_test_json(exported_summary_path, retained)
            _refresh_run_and_root_checksums(fixture.output, fixture.run_id)
            with self.assertRaisesRegex(
                EvidenceError, "prefix-cache aggregate summary projection changed"
            ):
                verify_evidence(fixture.output)

            mutations = (
                ("requested", {"requested": True}),
                ("nonzero-drafts", {"num_draft_tokens": 1}),
                (
                    "accepted-position",
                    {"accepted_tokens_per_position": {"1": 1}},
                ),
                ("scope", {"scope": "other_scope"}),
                ("integer-type", {"num_drafts": 0.0}),
                ("unknown", {"unexpected": 0}),
            )
            for name, changed_values in mutations:
                with self.subTest(mutation=name):
                    malformed = copy.deepcopy(source_summary)
                    speculative = malformed["speculative_decoding"]
                    self.assertIsInstance(speculative, dict)
                    speculative.update(changed_values)
                    with self.assertRaisesRegex(
                        EvidenceError,
                        "speculative_decoding|unknown aggregate field",
                    ):
                        _project_summary(malformed)

    def test_exact_cache_validator_binds_model_suite_samples_and_summary(self) -> None:
        raw_result = _cache_scalar_result(
            mode="on", pair_index=1, step_ordinal=1, prefix_target=8192
        )
        raw_result["scores"] = [0.0]
        with self.assertRaisesRegex(
            EvidenceError, "prefix-cache request result does not match its exact schema"
        ):
            _project_request_result(raw_result, kind="cache")

        legacy_result = _cache_scalar_result(
            mode="on", pair_index=1, step_ordinal=1, prefix_target=8192
        )
        for old_key, new_key in (
            ("native_cached_prompt_tokens", "prometheus_global_cached_prompt_tokens"),
            ("native_decode_s", "prometheus_global_decode_s"),
            ("native_decode_tokens", "prometheus_global_decode_tokens"),
            ("native_prompt_s", "prometheus_global_prompt_s"),
            ("native_prompt_tokens", "prometheus_global_prompt_tokens"),
        ):
            legacy_result[old_key] = legacy_result.pop(new_key)
        with self.assertRaisesRegex(
            EvidenceError, "prefix-cache request result does not match its exact schema"
        ):
            _project_request_result(legacy_result, kind="cache")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("off", "on"):
                with self.subTest(mode=mode):
                    run_dir = root / mode
                    run_dir.mkdir()
                    suite, model, events, raw_summary = self._write_cache_run(
                        run_dir, mode=mode
                    )
                    samples = _project_requests(
                        events, raw_summary, evidence_kind="serving"
                    )
                    summary = _project_summary(raw_summary)
                    _validate_prefix_cache_aggregates(
                        samples,
                        summary,
                        model=model,
                        suite=suite,
                    )
                    cache_samples = [
                        sample for sample in samples if sample.get("kind") == "cache"
                    ]
                    self.assertEqual(len(cache_samples), 30)
                    self.assertTrue(
                        all(sample["selected_attempt"] is True for sample in cache_samples)
                    )
                    if mode == "off":
                        self.assertTrue(
                            all(sample["cached_prompt_tokens"] == 0 for sample in cache_samples)
                        )
                    else:
                        warm_samples = [
                            sample
                            for sample in cache_samples
                            if sample["cache_condition"] == "warm-prefix-hit"
                        ]
                        self.assertEqual(len(warm_samples), 10)
                        self.assertTrue(
                            all(
                                sample["cached_prompt_tokens"]
                                / sample["prompt_tokens"]
                                >= 0.90
                                for sample in warm_samples
                            )
                        )

            tamper_run_dir = root / "tamper"
            tamper_run_dir.mkdir()
            suite, model, events, raw_summary = self._write_cache_run(
                tamper_run_dir, mode="on"
            )
            samples = _project_requests(events, raw_summary, evidence_kind="serving")
            summary = _project_summary(raw_summary)
            altered_summary = copy.deepcopy(summary)
            altered_summary["cases"][0]["prefix_cache"]["conditions"][2][
                "median_e2e_s"
            ] += 0.01
            with self.assertRaisesRegex(
                EvidenceError, "prefix-cache summary aggregates disagree"
            ):
                _validate_prefix_cache_aggregates(
                    samples,
                    altered_summary,
                    model=model,
                    suite=suite,
                )

            altered_samples = copy.deepcopy(samples)
            warm = next(
                sample
                for sample in altered_samples
                if sample.get("cache_condition") == "warm-prefix-hit"
            )
            warm["cached_prompt_tokens"] = 91
            with self.assertRaisesRegex(EvidenceError, "server counters do not reconcile"):
                _validate_prefix_cache_aggregates(
                    altered_samples,
                    summary,
                    model=model,
                    suite=suite,
                )

            malformed_sample = copy.deepcopy(samples)
            malformed_sample[0]["cache_note"] = 0
            with self.assertRaisesRegex(
                EvidenceError, "prefix-cache evidence sample does not match its exact schema"
            ):
                _validate_prefix_cache_aggregates(
                    malformed_sample,
                    summary,
                    model=model,
                    suite=suite,
                )

            generic_sample = copy.deepcopy(samples)
            generic_sample[0]["scores"] = [0.0]
            with self.assertRaisesRegex(
                EvidenceError, "prefix-cache evidence sample does not match its exact schema"
            ):
                _validate_prefix_cache_aggregates(
                    generic_sample,
                    summary,
                    model=model,
                    suite=suite,
                )

            swapped_samples = copy.deepcopy(samples)
            first_case_id = str(swapped_samples[0]["case_id"])
            cold_index = next(
                index
                for index, sample in enumerate(swapped_samples)
                if sample.get("case_id") == first_case_id
                and sample.get("cache_pair_index") == 1
                and sample.get("cache_condition") == "forced-cold-a"
            )
            warm_index = next(
                index
                for index, sample in enumerate(swapped_samples)
                if sample.get("case_id") == first_case_id
                and sample.get("cache_pair_index") == 1
                and sample.get("cache_condition") == "warm-prefix-hit"
            )
            swapped_samples[cold_index], swapped_samples[warm_index] = (
                swapped_samples[warm_index],
                swapped_samples[cold_index],
            )
            with self.assertRaisesRegex(EvidenceError, "fixed run order"):
                _validate_prefix_cache_aggregates(
                    swapped_samples,
                    summary,
                    model=model,
                    suite=suite,
                )

            generic_case = copy.deepcopy(summary)
            generic_case["cases"][0]["audio_duration_s"] = 1.0
            with self.assertRaisesRegex(
                EvidenceError, "prefix-cache summary case does not match its exact schema"
            ):
                _validate_prefix_cache_aggregates(
                    samples,
                    generic_case,
                    model=model,
                    suite=suite,
                )

            float_model = copy.deepcopy(model)
            float_model["runtime_parallel"] = 1.0
            with self.assertRaisesRegex(EvidenceError, "runtime_parallel must be an integer"):
                _validate_prefix_cache_aggregates(
                    samples,
                    summary,
                    model=float_model,
                    suite=suite,
                )

            for field, altered in (
                ("max_context", 1),
                ("max_context", 197_759),
                ("max_context", 262_143),
                ("max_context", 262_145),
                ("native_context", 1),
                ("native_context", 262_143),
                ("native_context", 262_145),
                ("native_context", 262_144.0),
            ):
                with self.subTest(model_context_field=field, altered=altered):
                    altered_model = copy.deepcopy(model)
                    altered_model[field] = altered
                    with self.assertRaisesRegex(
                        EvidenceError,
                        "contexts must equal the fixed protocol context|must be an integer",
                    ):
                        _validate_prefix_cache_aggregates(
                            samples,
                            summary,
                            model=altered_model,
                            suite=suite,
                        )

            for field, expected in (
                ("warmups", 0),
                ("repetitions", 5),
                ("max_output_tokens", 128),
                ("concurrency", 1),
            ):
                with self.subTest(float_suite_field=field):
                    float_suite = copy.deepcopy(suite)
                    float_suite["cases"][0][field] = float(expected)
                    with self.assertRaisesRegex(
                        EvidenceError, f"suite case.{field} must be an integer"
                    ):
                        _validate_prefix_cache_aggregates(
                            samples,
                            summary,
                            model=model,
                            suite=float_suite,
                        )

            invalid_rate_samples = copy.deepcopy(samples)
            invalid_rate_samples[0]["decode_tps"] = 1.0
            with self.assertRaisesRegex(EvidenceError, "client timing metrics are inconsistent"):
                _validate_prefix_cache_aggregates(
                    invalid_rate_samples,
                    summary,
                    model=model,
                    suite=suite,
                )

            short_session_summary = copy.deepcopy(summary)
            short_session_summary["cases"][0]["elapsed_s"] = 1.0
            short_session_summary["cases"][0]["prefix_cache"][
                "case_session_wall_s"
            ] = 1.0
            with self.assertRaisesRegex(EvidenceError, "session wall is shorter"):
                _validate_prefix_cache_aggregates(
                    samples,
                    short_session_summary,
                    model=model,
                    suite=suite,
                )

            non_cache_summary = copy.deepcopy(summary)
            non_cache_summary["cases"][0]["kind"] = "decode"
            with self.assertRaisesRegex(EvidenceError, "non-cache case"):
                _validate_prefix_cache_aggregates(
                    samples,
                    non_cache_summary,
                    model=model,
                    suite=suite,
                )

    def test_export_reexport_and_verify_recompute_cache_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            self._write_cache_run(
                fixture.run_dir, mode="on", with_startup_provenance=True
            )

            with patch(
                "bench.evidence._export_campaign", side_effect=_fake_campaign_export
            ):
                first = export_evidence(
                    results_root=fixture.results,
                    output_root=fixture.output,
                )
                exported = {
                    str(path.relative_to(fixture.output)): path.read_bytes()
                    for path in fixture.output.rglob("*")
                    if path.is_file()
                }
                second = export_evidence(
                    results_root=fixture.results,
                    output_root=fixture.output,
                )

            self.assertTrue(first["changed"])
            self.assertFalse(second["changed"])
            self.assertEqual(
                exported,
                {
                    str(path.relative_to(fixture.output)): path.read_bytes()
                    for path in fixture.output.rglob("*")
                    if path.is_file()
                },
            )
            self.assertEqual("verified", verify_evidence(fixture.output)["status"])

            exported_samples = json.loads(
                (fixture.output / "runs" / fixture.run_id / "samples.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(exported_samples["sample_count"], 30)
            self.assertTrue(
                all(sample["kind"] == "cache" for sample in exported_samples["samples"])
            )
            exported_manifest = json.loads(
                (fixture.output / "runs" / fixture.run_id / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                set(exported_manifest),
                {
                    "artifacts",
                    "evidence_kind",
                    "lifecycle",
                    "model",
                    "run_date_utc",
                    "runtime",
                    "sanitization",
                    "schema_version",
                    "source_run_id",
                    "status",
                    "suite",
                },
            )
            self.assertEqual(
                {artifact["role"] for artifact in exported_manifest["artifacts"]},
                {"model", "runtime_binary"},
            )
            self.assertNotIn("versions", exported_manifest["runtime"])

            exported_summary_path = (
                fixture.output / "runs" / fixture.run_id / "summary.json"
            )
            exported_summary = json.loads(
                exported_summary_path.read_text(encoding="utf-8")
            )
            exported_summary["aggregates"]["cases"][0]["prefix_cache"][
                "conditions"
            ][2]["median_e2e_s"] += 0.01
            _write_test_json(exported_summary_path, exported_summary)
            _refresh_run_and_root_checksums(fixture.output, fixture.run_id)
            with self.assertRaisesRegex(
                EvidenceError, "prefix-cache summary aggregates disagree"
            ):
                verify_evidence(fixture.output)

            exported_samples_path = (
                fixture.output / "runs" / fixture.run_id / "samples.json"
            )
            exported_manifest_path = (
                fixture.output / "runs" / fixture.run_id / "manifest.json"
            )
            exported_summary_path.write_bytes(
                exported[f"runs/{fixture.run_id}/summary.json"]
            )
            exported_samples = json.loads(
                exported_samples_path.read_text(encoding="utf-8")
            )
            exported_samples["samples"][0]["cache_note"] = 0
            _write_test_json(exported_samples_path, exported_samples)
            _refresh_run_and_root_checksums(fixture.output, fixture.run_id)
            with self.assertRaisesRegex(
                EvidenceError, "prefix-cache evidence sample does not match its exact schema"
            ):
                verify_evidence(fixture.output)

            exported_samples_path.write_bytes(
                exported[f"runs/{fixture.run_id}/samples.json"]
            )
            exported_manifest = json.loads(
                exported_manifest_path.read_text(encoding="utf-8")
            )
            exported_manifest["model"]["runtime_parallel"] = 1.0
            _write_test_json(exported_manifest_path, exported_manifest)
            _refresh_run_and_root_checksums(fixture.output, fixture.run_id)
            with self.assertRaisesRegex(EvidenceError, "runtime_parallel must be an integer"):
                verify_evidence(fixture.output)

            exported_manifest_path.write_bytes(
                exported[f"runs/{fixture.run_id}/manifest.json"]
            )
            exported_manifest = json.loads(
                exported_manifest_path.read_text(encoding="utf-8")
            )
            exported_manifest["model"]["max_context"] = 262_143
            _write_test_json(exported_manifest_path, exported_manifest)
            _refresh_run_and_root_checksums(fixture.output, fixture.run_id)
            with self.assertRaisesRegex(
                EvidenceError, "contexts must equal the fixed protocol context"
            ):
                verify_evidence(fixture.output)

            exported_manifest_path.write_bytes(
                exported[f"runs/{fixture.run_id}/manifest.json"]
            )
            exported_manifest = json.loads(
                exported_manifest_path.read_text(encoding="utf-8")
            )
            exported_manifest["trace"] = "synthetic-cache-manifest-trace"
            _write_test_json(exported_manifest_path, exported_manifest)
            _refresh_run_and_root_checksums(fixture.output, fixture.run_id)
            with self.assertRaisesRegex(
                EvidenceError, "prefix-cache manifest does not match its exact schema"
            ):
                verify_evidence(fixture.output)

            exported_manifest_path.write_bytes(
                exported[f"runs/{fixture.run_id}/manifest.json"]
            )
            exported_samples = json.loads(
                exported_samples_path.read_text(encoding="utf-8")
            )
            exported_samples["samples"].append(
                {
                    "kind": "decode",
                    "sample_index": 31,
                    "sample_type": "measured_request",
                    "trace": "synthetic-nonprotocol-cache-sample",
                }
            )
            exported_samples["sample_count"] = 31
            _write_test_json(exported_samples_path, exported_samples)
            _refresh_run_and_root_checksums(fixture.output, fixture.run_id)
            with self.assertRaisesRegex(
                EvidenceError, "prefix-cache samples document does not match its exact schema"
            ):
                verify_evidence(fixture.output)

            exported_samples_path.write_bytes(
                exported[f"runs/{fixture.run_id}/samples.json"]
            )
            exported_samples = json.loads(
                exported_samples_path.read_text(encoding="utf-8")
            )
            exported_samples["sample_count"] = 30.0
            _write_test_json(exported_samples_path, exported_samples)
            _refresh_run_and_root_checksums(fixture.output, fixture.run_id)
            with self.assertRaisesRegex(
                EvidenceError, "prefix-cache samples document does not match its exact schema"
            ):
                verify_evidence(fixture.output)

            exported_samples_path.write_bytes(
                exported[f"runs/{fixture.run_id}/samples.json"]
            )
            telemetry_index_path = (
                fixture.output / "runs" / fixture.run_id / "telemetry.json"
            )
            telemetry_chunk_path = (
                fixture.output / "runs" / fixture.run_id / "telemetry-0001.json"
            )
            telemetry_index = json.loads(telemetry_index_path.read_text(encoding="utf-8"))
            telemetry_index.update(
                {
                    "chunk_count": 1,
                    "chunks": ["telemetry-0001.json"],
                    "sample_count": 1,
                    "segment_count": 1,
                }
            )
            telemetry_chunk = {
                "sample_count": 1,
                "schema_version": SCHEMA_VERSION,
                "segments": [
                    {
                        "first_phase_sample_index": 1,
                        "first_sample_index": 1,
                        "phase": "idle",
                        "phase_segment": 1,
                        "rows": [[0.0, False, None, None, None, None, None, None, None, None, None, None]],
                    }
                ],
            }
            _write_test_json(telemetry_index_path, telemetry_index)
            _write_test_json(telemetry_chunk_path, telemetry_chunk)
            _refresh_run_and_root_checksums(fixture.output, fixture.run_id)
            self.assertEqual("verified", verify_evidence(fixture.output)["status"])

            telemetry_chunk["segments"][0]["rows"][0][2] = (
                "synthetic-cache-telemetry-trace"
            )
            _write_test_json(telemetry_chunk_path, telemetry_chunk)
            _refresh_run_and_root_checksums(fixture.output, fixture.run_id)
            with self.assertRaisesRegex(
                EvidenceError,
                (
                    r"(?:prefix-cache telemetry\.gpu_util_pct must be a finite "
                    r"JSON float|telemetry numeric scalar changed)"
                ),
            ):
                verify_evidence(fixture.output)

            telemetry_chunk["segments"][0]["rows"][0][2] = None
            _write_test_json(telemetry_chunk_path, telemetry_chunk)
            _write_test_json(
                fixture.output / "runs" / fixture.run_id / "trace.json",
                {"trace": "synthetic-unreferenced-cache-trace"},
            )
            _refresh_run_and_root_checksums(fixture.output, fixture.run_id)
            with self.assertRaisesRegex(
                EvidenceError, "prefix-cache bundle file set does not match its protocol"
            ):
                verify_evidence(fixture.output)


class PrefixCacheSelectionTests(unittest.TestCase):
    def test_matrix_filters_cache_profiles_before_plan_creation(self) -> None:
        normal = SimpleNamespace(
            id="normal-llamacpp",
            backend="llamacpp",
            support_status="supported",
            tasks=("chat",),
            prefix_cache_mode=None,
        )
        cache = SimpleNamespace(
            id="cache-llamacpp",
            backend="llamacpp",
            support_status="supported",
            tasks=("chat",),
            prefix_cache_mode="on",
        )

        def planned_for(suite_id: str, results: Path) -> list[str]:
            plans: list[str] = []

            def create_plan(*, model: SimpleNamespace, results_root: Path, **_: object) -> Path:
                plans.append(model.id)
                run_dir = results_root / f"run-{model.id}"
                run_dir.mkdir()
                return run_dir

            args = SimpleNamespace(
                models=Path("synthetic-models.toml"),
                suite=Path("synthetic-suite.toml"),
                results=results,
                backend=None,
                task=None,
                match="*",
                limit=None,
                plan_only=True,
                allow_download=True,
                fail_fast=False,
            )
            with (
                patch("sparkbench.load_models", return_value={normal.id: normal, cache.id: cache}),
                patch("sparkbench.load_suite", return_value=SimpleNamespace(id=suite_id)),
                patch("sparkbench._inventory"),
                patch("sparkbench.assess_model_availability", return_value={}),
                patch("sparkbench.create_plan", side_effect=create_plan),
            ):
                self.assertEqual(command_matrix(args), 0)
            return plans

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(planned_for("smoke", root / "normal"), [normal.id])
            self.assertEqual(
                planned_for("llamacpp-prefix-cache", root / "cache"), [cache.id]
            )

    def test_legacy_plan_preflight_rejects_profile_and_suite_mismatches(self) -> None:
        cache_case = SimpleNamespace(
            id="llamacpp-prefix-cache-8192",
            kind="cache",
            prompt_repetitions=8192,
        )
        cache_model = SimpleNamespace(
            prefix_cache_mode="on",
            runtime_parallel=1,
            args=("--cache-prompt",),
        )
        with self.assertRaises(PreflightError):
            _validate_prefix_cache_plan_selection(
                cache_model,
                SimpleNamespace(id="smoke", cases=[cache_case]),
            )
        with self.assertRaises(PreflightError):
            _validate_prefix_cache_plan_selection(
                SimpleNamespace(prefix_cache_mode=None, runtime_parallel=1, args=()),
                SimpleNamespace(id="llamacpp-prefix-cache", cases=[cache_case]),
            )
        with self.assertRaises(PreflightError):
            _validate_prefix_cache_plan_selection(
                SimpleNamespace(
                    prefix_cache_mode="on",
                    runtime_parallel=1,
                    args=("--cache-prompt=true",),
                ),
                SimpleNamespace(id="llamacpp-prefix-cache", cases=[cache_case]),
            )
        with self.assertRaises(PreflightError):
            _validate_prefix_cache_plan_selection(
                SimpleNamespace(prefix_cache_mode=None, runtime_parallel=1, args=()),
                SimpleNamespace(
                    id="llamacpp-prefix-cache",
                    cases=[SimpleNamespace(id="ordinary", kind="decode")],
                ),
            )

    def test_legacy_plan_preflight_requires_exact_cache_geometry_and_context(self) -> None:
        def frozen_suite() -> SimpleNamespace:
            return SimpleNamespace(
                id="llamacpp-prefix-cache",
                cases=[
                    SimpleNamespace(
                        case_id=f"{case_id}--{'a' * 12}",
                        id=case_id,
                        kind="cache",
                        requires=["chat"],
                        warmups=0,
                        repetitions=5,
                        max_output_tokens=128,
                        max_turns=1,
                        temperature=0.0,
                        concurrency=1,
                        prompt_repetitions=target,
                    )
                    for case_id, target in PREFIX_CACHE_PREFIX_TARGETS.items()
                ],
            )

        def frozen_model() -> SimpleNamespace:
            return SimpleNamespace(
                backend="llamacpp",
                prefix_cache_mode="on",
                runtime_parallel=1,
                max_context=PREFIX_CACHE_CONTEXT_TOKENS,
                native_context=PREFIX_CACHE_CONTEXT_TOKENS,
                args=list(prefix_cache_llamacpp_args("on")),
            )

        _validate_prefix_cache_plan_selection(frozen_model(), frozen_suite())
        for field, altered in (
            ("requires", ["chat", "tools"]),
            ("warmups", 1),
            ("repetitions", 1),
            ("max_output_tokens", 127),
            ("max_turns", 2),
            ("temperature", 0),
            ("concurrency", 2),
            ("concurrency", 1.0),
            ("prompt_repetitions", 8_193),
        ):
            with self.subTest(case_field=field):
                suite = frozen_suite()
                setattr(suite.cases[0], field, altered)
                with self.assertRaises(PreflightError):
                    _validate_prefix_cache_plan_selection(frozen_model(), suite)
        for field, altered in (
            ("backend", "ollama"),
            ("runtime_parallel", 1.0),
            ("max_context", PREFIX_CACHE_CONTEXT_TOKENS - 1),
            ("native_context", PREFIX_CACHE_CONTEXT_TOKENS - 1),
            ("args", tuple(prefix_cache_llamacpp_args("on"))),
        ):
            with self.subTest(model_field=field):
                model = frozen_model()
                setattr(model, field, altered)
                with self.assertRaises(PreflightError):
                    _validate_prefix_cache_plan_selection(model, frozen_suite())
        for appended in (
            ["--parallel=2"],
            ["--parallel", "2"],
            ["--ctx-size=1"],
            ["--ctx-size", "1"],
            ["--metrics"],
            ["--metrics=false"],
        ):
            with self.subTest(appended_arguments=appended):
                model = frozen_model()
                model.args.extend(appended)
                with self.assertRaisesRegex(
                    PreflightError, "arguments do not match the exact protocol"
                ):
                    _validate_prefix_cache_plan_selection(model, frozen_suite())


if __name__ == "__main__":
    unittest.main()
