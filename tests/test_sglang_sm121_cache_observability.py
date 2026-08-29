"""Offline contracts for the pinned SM121 B0 cache-observability lane."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from bench import runner, runtime
from bench.journal import Journal
from bench.manifest import ManifestError, load_models, load_suite, validate_benchmark_selection
from bench.sglang_sm121_cache_observability import (
    SM121_CACHE_OBSERVABILITY_EXECUTION_MODE,
    SM121_CACHE_OBSERVABILITY_METRIC_FIELDS,
    SM121_CACHE_OBSERVABILITY_SUITE_ID,
    SM121_CACHE_RUNTIME_ATTESTATION_EVENT,
    SM121_CACHE_RUNTIME_EXPECTED,
    SM121_CACHE_SOURCE_DIGESTS,
    SM121_CACHE_STATIC_ASSERTIONS,
    SM121_CACHE_STATIC_ATTESTATION_EVENT,
    SM121_CACHE_ZERO_HIT_CASE_ID,
    SM121_CACHE_ZERO_HIT_EVENT,
    SM121_CACHE_ZERO_HIT_EXPECTED_RESPONSE,
    SM121_CACHE_ZERO_HIT_MAX_OUTPUT_TOKENS,
    SM121_CACHE_ZERO_HIT_PROMPT,
    SM121_CACHE_ZERO_HIT_PROMPT_SHA256,
    SM121_CACHE_ZERO_HIT_REQUEST_CONTRACT_SHA256,
    SM121CacheObservabilityError,
    derive_sm121_cache_zero_hit_admission,
    is_sm121_cache_observability_plan,
    sm121_cache_observability_lifecycle_issues,
    sm121_cache_zero_hit_request_body,
    sm121_cache_zero_hit_request_contract,
    validate_sm121_cache_observability_suite,
    validate_sm121_cache_runtime_attestation_event,
    validate_sm121_cache_static_attestation_event,
    validate_sm121_cache_zero_hit_event,
    validate_sm121_cache_zero_hit_request_contract,
)
from bench.sglang_sm121_cache_semantic import (
    SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
    SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
)
from bench.sglang_sm121_storage import (
    SM121_STORAGE_LOCAL_IMAGE_ID,
    SM121_STORAGE_PLATFORM,
    SM121_STORAGE_PROFILE_ID,
    SM121_STORAGE_SOURCE_TREE,
)


class _Response:
    """Tiny offline ``urlopen`` response double."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.read_sizes: list[int | None] = []

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def read(self, size: int | None = None) -> bytes:
        self.read_sizes.append(size)
        return self.payload if size is None else self.payload[:size]


def _static_event() -> dict[str, object]:
    return {
        "event": SM121_CACHE_STATIC_ATTESTATION_EVENT,
        "candidate_source_tree": SM121_STORAGE_SOURCE_TREE,
        **SM121_CACHE_SOURCE_DIGESTS,
        **SM121_CACHE_STATIC_ASSERTIONS,
    }


def _runtime_event() -> dict[str, object]:
    return {
        "event": SM121_CACHE_RUNTIME_ATTESTATION_EVENT,
        **SM121_CACHE_RUNTIME_EXPECTED,
    }


def _zero_event(
    *,
    case_id: str | None = None,
    attempt_id: str = "capability-attempt",
    metrics_available: bool = True,
    settled: bool = True,
    cached_total_delta: int = 0,
    detail_state: str = "omitted",
) -> dict[str, object]:
    """Return a complete scalar B0 event, recomputing its admission bit."""

    event: dict[str, object] = {
        "event": SM121_CACHE_ZERO_HIT_EVENT,
        "case_id": case_id
        or f"{SM121_CACHE_ZERO_HIT_CASE_ID}--abcdef123456",
        "protocol_case_id": SM121_CACHE_ZERO_HIT_CASE_ID,
        "attempt_id": attempt_id,
        "request_contract_sha256": SM121_CACHE_ZERO_HIT_REQUEST_CONTRACT_SHA256,
        "cache_details_requested": True,
        "streaming": False,
        "thinking_disabled": True,
        "response_detail_state": detail_state,
        "usage_detail_state": detail_state,
        "response_device_cached_tokens": None,
        "response_host_cached_tokens": None,
        "response_storage_cached_tokens": None,
        "usage_cached_tokens": None,
        "metrics_available": metrics_available,
        "metrics_before_polls": 2 if settled else 1,
        "metrics_after_polls": 2 if settled else 1,
        "metrics_before_settle_s": 0.0,
        "metrics_after_settle_s": 0.0,
        "metrics_before_settled": settled if metrics_available else False,
        "metrics_after_settled": settled if metrics_available else False,
        "zero_hit_basis": "not_admitted",
        "zero_hit_admitted": False,
    }
    before = {
        "prefill_input_tokens": 100,
        "prefill_device_hit_tokens": 0,
        "prefill_host_hit_tokens": 0,
        "prefill_storage_hit_tokens": 0,
        "cached_total_tokens": 0,
        "cached_device_tokens": 0,
        "cached_host_tokens": 0,
        "cached_storage_tokens": 0,
        "kv_available_tokens": 100,
        "kv_evictable_tokens": 0,
        "kv_used_tokens": 0,
        "mamba_available_tokens": 100,
        "mamba_evictable_tokens": 0,
        "mamba_used_tokens": 0,
    }
    after = dict(before)
    after["prefill_input_tokens"] = 108 if metrics_available else 100
    after["kv_available_tokens"] = 99
    after["kv_used_tokens"] = 1
    after["mamba_available_tokens"] = 99
    after["mamba_used_tokens"] = 1
    after["cached_total_tokens"] = cached_total_delta
    for metric in SM121_CACHE_OBSERVABILITY_METRIC_FIELDS:
        event[f"before_{metric}"] = before[metric]
        event[f"after_{metric}"] = after[metric]
        event[f"delta_{metric}"] = after[metric] - before[metric]
    for prefix in ("before", "after"):
        event[f"{prefix}_cached_total_series_present"] = (
            prefix == "after" and cached_total_delta > 0
        )
        event[f"{prefix}_cached_device_series_present"] = False
        event[f"{prefix}_cached_host_series_present"] = False
        event[f"{prefix}_cached_storage_series_present"] = False
    if detail_state == "zero_details":
        event.update(
            {
                "response_device_cached_tokens": 0,
                "response_host_cached_tokens": 0,
                "response_storage_cached_tokens": 0,
                "usage_cached_tokens": 0,
            }
        )
    elif detail_state == "nonzero_details":
        event.update(
            {
                "response_device_cached_tokens": 3,
                "response_host_cached_tokens": 0,
                "response_storage_cached_tokens": 0,
                "usage_cached_tokens": 3,
            }
        )
    admitted, basis = derive_sm121_cache_zero_hit_admission(event)
    event["zero_hit_admitted"] = admitted
    event["zero_hit_basis"] = basis
    return event


def _valid_lifecycle_events(*, admitted: bool = True) -> tuple[
    list[dict[str, object]], tuple[str, str]
]:
    quality_case_id = "synthetic-exact-answer-v2--0123456789ab"
    capability_case_id = f"{SM121_CACHE_ZERO_HIT_CASE_ID}--abcdef123456"
    zero = _zero_event(
        case_id=capability_case_id,
        metrics_available=admitted,
        settled=admitted,
    )
    if not admitted:
        # Make the partial case a valid well-formed observation, rather than
        # merely falsifying a derived flag.
        zero = _zero_event(
            case_id=capability_case_id,
            metrics_available=False,
            settled=False,
        )
    events: list[dict[str, object]] = [
        {
            "event": "run_start",
            "execution_mode": SM121_CACHE_OBSERVABILITY_EXECUTION_MODE,
        },
        {"event": "measurement_started"},
        _static_event(),
        _runtime_event(),
        {
            "event": "server_ready",
            "backend": "sglang",
            "fresh_server_lifetime": 1,
            "first_inference_is_case": True,
            "case_id": quality_case_id,
        },
        {
            "event": "case_start",
            "case_id": quality_case_id,
            "attempt_id": "quality-attempt",
            "kind": "quality",
        },
    ]
    events.extend(
        {
            "event": "request_complete",
            "case_id": quality_case_id,
            "attempt_id": "quality-attempt",
            "kind": "quality",
            "result": {},
        }
        for _ in range(4)
    )
    events.extend(
        [
            {
                "event": "case_complete",
                "case_id": quality_case_id,
                "attempt_id": "quality-attempt",
                "kind": "quality",
                "validation_passed": True,
            },
            {
                "event": "case_start",
                "case_id": capability_case_id,
                "attempt_id": "capability-attempt",
                "kind": "capability",
            },
            zero,
            {
                "event": "request_complete",
                "case_id": capability_case_id,
                "attempt_id": "capability-attempt",
                "kind": "capability",
                "result": {},
            },
            {
                "event": "case_complete",
                "case_id": capability_case_id,
                "attempt_id": "capability-attempt",
                "kind": "capability",
                "validation_passed": zero["zero_hit_admitted"],
            },
            {
                "event": "server_stopped",
                "backend": "sglang",
                "fresh_server_lifetime": 1,
            },
            {"event": "measurement_complete"},
            {"event": "run_complete", "status": "completed"},
        ]
    )
    return events, (quality_case_id, capability_case_id)


def _metric_exposition(
    *,
    cached_total: int | None = None,
    cached_source: str = "total",
    bad_source: bool = False,
    guardrails: bool = False,
    scheduler_labels: bool = False,
    guardrail_eviction_type: str | None = None,
    guardrail_retraction_type: str | None = None,
    guardrail_eviction_sample: int | None = None,
    guardrail_retraction_sample: int | None = None,
    guardrail_eviction_bad_labels: bool = False,
    guardrail_eviction_legacy_labels: bool = False,
) -> str:
    base = (
        'engine_type="prefill",model_name="synthetic",moe_ep_rank="0",'
        'pp_rank="0",tp_rank="0"'
        if scheduler_labels
        else ""
    )

    def labels(selector: str = "") -> str:
        joined = ",".join(item for item in (base, selector) if item)
        return "{" + joined + "}" if joined else ""

    lines = [
        f'sglang:prefill_effective_tokens_total{labels("mode=\"input\"")} 17',
        f'sglang:prefill_effective_tokens_total{labels("mode=\"device_hit\"")} 0',
        f'sglang:prefill_effective_tokens_total{labels("mode=\"host_hit\"")} 0',
        f'sglang:prefill_effective_tokens_total{labels("mode=\"storage_hit\"")} 0',
        f"sglang:kv_available_tokens{labels()} 90",
        f"sglang:kv_evictable_tokens{labels()} 0",
        f"sglang:kv_used_tokens{labels()} 10",
        f"sglang:mamba_available_tokens{labels()} 80",
        f"sglang:mamba_evictable_tokens{labels()} 0",
        f"sglang:mamba_used_tokens{labels()} 20",
    ]
    if cached_total is not None:
        source = "unknown" if bad_source else cached_source
        lines.append(
            f'sglang:cached_tokens_total{labels(f"cache_source=\"{source}\"")} {cached_total}'
        )
    if guardrails:
        guardrail_eviction_type = "counter"
        guardrail_retraction_type = "counter"
        guardrail_eviction_sample = 0
        guardrail_retraction_sample = 0
    if guardrail_eviction_type is not None:
        lines.append(f"# TYPE sglang:evicted_tokens_total {guardrail_eviction_type}")
    if guardrail_retraction_type is not None:
        lines.append(
            "# TYPE sglang:num_retracted_requests_total "
            f"{guardrail_retraction_type}"
        )
    if guardrail_eviction_sample is not None:
        eviction_labels = (
            labels()
            if guardrail_eviction_legacy_labels
            else labels('cache_type="UnifiedRadixCache"')
            if guardrail_eviction_bad_labels
            else '{cache_type="UnifiedRadixCache"}'
        )
        lines.append(
            "sglang:evicted_tokens_total"
            f"{eviction_labels} {guardrail_eviction_sample}"
        )
    if guardrail_retraction_sample is not None:
        lines.extend(
            (
                "sglang:num_retracted_requests_total"
                f"{labels()} {guardrail_retraction_sample}",
            )
        )
    return "\n".join(lines) + "\n"


class SM121CacheObservabilityContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Path(__file__).resolve().parents[1]
        self.model = load_models(self.workspace / "manifests" / "models.toml")[
            SM121_STORAGE_PROFILE_ID
        ]
        self.suite = load_suite(
            self.workspace
            / "manifests"
            / "suites"
            / "qwen38_flash_next_sm121_triton_storage_cache_observability_canary.toml"
        )

    def test_real_manifest_loads_and_binds_only_to_the_storage_profile(self) -> None:
        self.assertEqual(self.suite.id, SM121_CACHE_OBSERVABILITY_SUITE_ID)
        validate_sm121_cache_observability_suite(self.suite)
        validate_benchmark_selection(self.model, self.suite)
        self.assertTrue(is_sm121_cache_observability_plan(self.model, self.suite))

        with self.assertRaises(SM121CacheObservabilityError):
            validate_sm121_cache_observability_suite(
                replace(self.suite, protocol_digest="sha256:" + "a" * 64)
            )
        with self.assertRaises(ManifestError):
            validate_benchmark_selection(
                replace(self.model, id="not-the-sm121-storage-profile"), self.suite
            )

    def test_request_contract_hash_and_body_are_pinned(self) -> None:
        validate_sm121_cache_zero_hit_request_contract()
        self.assertEqual(
            sm121_cache_zero_hit_request_contract(),
            {
                "schema_version": 1,
                "endpoint": "/v1/chat/completions",
                "prompt_sha256": SM121_CACHE_ZERO_HIT_PROMPT_SHA256,
                "max_tokens": SM121_CACHE_ZERO_HIT_MAX_OUTPUT_TOKENS,
                "temperature": 0.0,
                "n": 1,
                "stream": False,
                "return_cached_tokens_details": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        body = sm121_cache_zero_hit_request_body("synthetic-served-model")
        self.assertEqual(
            body,
            {
                "model": "synthetic-served-model",
                "messages": [{"role": "user", "content": SM121_CACHE_ZERO_HIT_PROMPT}],
                "max_tokens": 16,
                "temperature": 0.0,
                "n": 1,
                "stream": False,
                "return_cached_tokens_details": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        self.assertEqual(
            SM121_CACHE_ZERO_HIT_REQUEST_CONTRACT_SHA256,
            "sha256:aebacb6a7767f6bcd82caff7294d08e6d323281f463770c2c9187457af6f2c8a",
        )
        with self.assertRaises(SM121CacheObservabilityError):
            sm121_cache_zero_hit_request_body("")

    def test_static_and_runtime_attestations_reject_boolean_coercion(self) -> None:
        static = _static_event()
        runtime_event = _runtime_event()
        validate_sm121_cache_static_attestation_event(static)
        validate_sm121_cache_runtime_attestation_event(runtime_event)

        static["cache_off_selects_chunk_cache"] = 1
        runtime_event["disable_radix_cache"] = 1
        with self.assertRaises(SM121CacheObservabilityError):
            validate_sm121_cache_static_attestation_event(static)
        with self.assertRaises(SM121CacheObservabilityError):
            validate_sm121_cache_runtime_attestation_event(runtime_event)

    def test_zero_event_derives_admission_and_records_nonadmitted_variants(self) -> None:
        admitted = _zero_event()
        self.assertEqual(
            derive_sm121_cache_zero_hit_admission(admitted),
            (True, "omitted_or_null_with_native_counters"),
        )
        validate_sm121_cache_zero_hit_event(admitted)

        explicit = _zero_event(detail_state="zero_details")
        self.assertEqual(
            derive_sm121_cache_zero_hit_admission(explicit),
            (True, "explicit_details"),
        )
        validate_sm121_cache_zero_hit_event(explicit)

        fallback_total = _zero_event(cached_total_delta=3)
        self.assertEqual(
            derive_sm121_cache_zero_hit_admission(fallback_total),
            (False, "not_admitted"),
        )
        validate_sm121_cache_zero_hit_event(fallback_total)

        unavailable = _zero_event(metrics_available=False, settled=False)
        self.assertEqual(
            derive_sm121_cache_zero_hit_admission(unavailable),
            (False, "not_admitted"),
        )
        validate_sm121_cache_zero_hit_event(unavailable)

        tampered = deepcopy(fallback_total)
        tampered["zero_hit_admitted"] = True
        tampered["zero_hit_basis"] = "explicit_details"
        with self.assertRaises(SM121CacheObservabilityError):
            validate_sm121_cache_zero_hit_event(tampered)

    def test_zero_event_reconciles_signed_pool_deltas_and_rejects_bad_binding(self) -> None:
        event = _zero_event()
        self.assertEqual(event["delta_kv_available_tokens"], -1)
        validate_sm121_cache_zero_hit_event(event)

        bad_delta = deepcopy(event)
        bad_delta["delta_kv_available_tokens"] = 0
        with self.assertRaises(SM121CacheObservabilityError):
            validate_sm121_cache_zero_hit_event(bad_delta)

        bad_case = deepcopy(event)
        bad_case["case_id"] = SM121_CACHE_ZERO_HIT_CASE_ID
        with self.assertRaises(SM121CacheObservabilityError):
            validate_sm121_cache_zero_hit_event(bad_case)

    def test_lifecycle_accepts_complete_and_well_formed_partial_runs(self) -> None:
        for admitted in (True, False):
            with self.subTest(admitted=admitted):
                events, case_ids = _valid_lifecycle_events(admitted=admitted)
                self.assertEqual(
                    sm121_cache_observability_lifecycle_issues(
                        events, planned_case_ids=case_ids
                    ),
                    (),
                )

    def test_lifecycle_rejects_terminal_topology_and_binding_tampering(self) -> None:
        events, case_ids = _valid_lifecycle_events()
        mutations = (
            ("execution_mode", "b0_execution_mode"),
            ("abort", "b0_unexpected_event"),
            ("request_binding", "b0_request_binding"),
            ("zero_binding", "b0_zero_binding"),
            ("terminal_status", "b0_run_complete_status"),
        )
        for name, expected_code in mutations:
            with self.subTest(name=name):
                changed = deepcopy(events)

                def changed_find(event_name: str) -> dict[str, object]:
                    return next(event for event in changed if event["event"] == event_name)

                # The compact mutation table captures the intended operation;
                # rebuild its closures against this independent journal copy.
                if name == "execution_mode":
                    changed_find("run_start")["execution_mode"] = "wrong"
                elif name == "abort":
                    changed.insert(-1, {"event": "run_aborted"})
                elif name == "request_binding":
                    next(
                        event
                        for event in changed
                        if event["event"] == "request_complete" and event["kind"] == "quality"
                    )["case_id"] = "wrong-case"
                elif name == "zero_binding":
                    next(
                        event
                        for event in changed
                        if event["event"] == SM121_CACHE_ZERO_HIT_EVENT
                    )["attempt_id"] = "wrong-attempt"
                else:
                    changed_find("run_complete")["status"] = "partial"
                codes = {
                    str(issue["code"])
                    for issue in sm121_cache_observability_lifecycle_issues(
                        changed, planned_case_ids=case_ids
                    )
                }
                self.assertIn(expected_code, codes)


class SM121CacheObservabilityRuntimeTests(unittest.TestCase):
    @staticmethod
    def _server(base_url: str = "http://127.0.0.1:30000/v1") -> SimpleNamespace:
        return SimpleNamespace(
            **{
                "backend": "sglang",
                "base_url": base_url,
                "author" + "ization": None,
                "container_id": "synthetic-container",
            }
        )

    def test_metrics_snapshot_tracks_no_series_and_fallback_total(self) -> None:
        server = self._server()
        with patch(
            "bench.runtime.urllib.request.urlopen",
            return_value=_Response(_metric_exposition().encode("utf-8")),
        ):
            empty = runtime.snapshot_sm121_cache_observability_metrics(server)
        self.assertTrue(empty["available"])
        self.assertFalse(empty["guardrail_metrics_available"])
        self.assertEqual(empty["prefill_input_tokens"], 17)
        self.assertEqual(empty["cached_total_tokens"], 0)
        self.assertFalse(empty["cached_total_series_present"])

        with patch(
            "bench.runtime.urllib.request.urlopen",
            return_value=_Response(
                _metric_exposition(scheduler_labels=True).encode("utf-8")
            ),
        ):
            labeled = runtime.snapshot_sm121_cache_observability_metrics(server)
        self.assertTrue(labeled["available"])
        self.assertEqual(labeled["prefill_input_tokens"], 17)

        with patch(
            "bench.runtime.urllib.request.urlopen",
            return_value=_Response(_metric_exposition(cached_total=3).encode("utf-8")),
        ):
            fallback = runtime.snapshot_sm121_cache_observability_metrics(server)
        self.assertTrue(fallback["available"])
        self.assertEqual(fallback["cached_total_tokens"], 3)
        self.assertTrue(fallback["cached_total_series_present"])

        with patch(
            "bench.runtime.urllib.request.urlopen",
            return_value=_Response(
                _metric_exposition(cached_total=3, bad_source=True).encode("utf-8")
            ),
        ):
            malformed = runtime.snapshot_sm121_cache_observability_metrics(server)
        self.assertFalse(malformed["available"])

    def test_metrics_snapshot_tracks_device_hits_and_guardrail_counters(self) -> None:
        server = self._server()
        with patch(
            "bench.runtime.urllib.request.urlopen",
            return_value=_Response(
                _metric_exposition(
                    cached_total=32_768,
                    cached_source="device",
                    guardrails=True,
                    scheduler_labels=True,
                ).encode("utf-8")
            ),
        ):
            snapshot = runtime.snapshot_sm121_cache_observability_metrics(
                server, semantic_arm=SM121_CACHE_SEMANTIC_CACHE_ON_ARM
            )
        self.assertTrue(snapshot["available"])
        self.assertTrue(snapshot["guardrail_metrics_available"])
        self.assertEqual(snapshot["cached_device_tokens"], 32_768)
        self.assertTrue(snapshot["cached_device_series_present"])
        self.assertEqual(snapshot["evicted_tokens"], 0)
        self.assertEqual(snapshot["retracted_requests"], 0)

        with patch(
            "bench.runtime.urllib.request.urlopen",
            return_value=_Response(
                _metric_exposition(
                    guardrails=True,
                    scheduler_labels=True,
                    guardrail_eviction_legacy_labels=True,
                ).encode("utf-8")
            ),
        ):
            legacy = runtime.snapshot_sm121_cache_observability_metrics(server)
        self.assertTrue(legacy["available"])
        self.assertTrue(legacy["guardrail_metrics_available"])

    def test_semantic_guardrail_zero_omission_is_arm_and_schema_bound(self) -> None:
        server = self._server()

        def snapshot(
            exposition: str, arm: str
        ) -> dict[str, object]:
            with patch(
                "bench.runtime.urllib.request.urlopen",
                return_value=_Response(exposition.encode("utf-8")),
            ):
                return runtime.snapshot_sm121_cache_observability_metrics(
                    server, semantic_arm=arm
                )

        cache_off_zero = snapshot(
            _metric_exposition(
                scheduler_labels=True,
                guardrail_retraction_type="counter",
            ),
            SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
        )
        self.assertTrue(cache_off_zero["available"])
        self.assertTrue(cache_off_zero["guardrail_metrics_available"])
        self.assertEqual(cache_off_zero["evicted_tokens"], 0)
        self.assertEqual(cache_off_zero["retracted_requests"], 0)

        cache_on_zero = snapshot(
            _metric_exposition(
                scheduler_labels=True,
                guardrail_eviction_type="counter",
                guardrail_retraction_type="counter",
            ),
            SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
        )
        self.assertTrue(cache_on_zero["available"])
        self.assertTrue(cache_on_zero["guardrail_metrics_available"])

        cache_on_materialized = snapshot(
            _metric_exposition(
                scheduler_labels=True,
                guardrail_eviction_type="counter",
                guardrail_retraction_type="counter",
                guardrail_eviction_sample=7,
                guardrail_retraction_sample=3,
            ),
            SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
        )
        self.assertTrue(cache_on_materialized["guardrail_metrics_available"])
        self.assertEqual(cache_on_materialized["evicted_tokens"], 7)
        self.assertEqual(cache_on_materialized["retracted_requests"], 3)

        bad_cases = (
            (
                "cache_off_eviction_family_present",
                _metric_exposition(
                    scheduler_labels=True,
                    guardrail_eviction_type="counter",
                    guardrail_retraction_type="counter",
                ),
                SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
            ),
            (
                "cache_on_eviction_family_missing",
                _metric_exposition(
                    scheduler_labels=True,
                    guardrail_retraction_type="counter",
                ),
                SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
            ),
            (
                "cache_off_truncated_eviction_family",
                _metric_exposition(
                    scheduler_labels=True,
                    guardrail_retraction_type="counter",
                )
                + "# HELP sglang:evicted_tokens_total eviction counter\n",
                SM121_CACHE_SEMANTIC_CACHE_OFF_ARM,
            ),
            (
                "wrong_counter_type",
                _metric_exposition(
                    scheduler_labels=True,
                    guardrail_eviction_type="gauge",
                    guardrail_retraction_type="counter",
                ),
                SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
            ),
            (
                "wrong_eviction_labels",
                _metric_exposition(
                    scheduler_labels=True,
                    guardrail_eviction_type="counter",
                    guardrail_retraction_type="counter",
                    guardrail_eviction_sample=0,
                    guardrail_eviction_bad_labels=True,
                ),
                SM121_CACHE_SEMANTIC_CACHE_ON_ARM,
            ),
        )
        for name, exposition, arm in bad_cases:
            with self.subTest(name=name):
                self.assertFalse(
                    snapshot(exposition, arm)["guardrail_metrics_available"]
                )

        duplicate_type = (
            _metric_exposition(
                scheduler_labels=True,
                guardrail_eviction_type="counter",
                guardrail_retraction_type="counter",
            )
            + "# TYPE sglang:evicted_tokens_total counter\n"
        )
        self.assertFalse(
            snapshot(duplicate_type, SM121_CACHE_SEMANTIC_CACHE_ON_ARM)[
                "guardrail_metrics_available"
            ]
        )
        stem_alias = (
            _metric_exposition(
                scheduler_labels=True,
                guardrail_retraction_type="counter",
            )
            + "# TYPE sglang:evicted_tokens counter\n"
        )
        self.assertFalse(
            snapshot(stem_alias, SM121_CACHE_SEMANTIC_CACHE_ON_ARM)[
                "guardrail_metrics_available"
            ]
        )

    def test_metrics_settlement_requires_two_identical_available_snapshots(self) -> None:
        server = self._server()
        stable = runtime._sm121_cache_metric_defaults()
        stable["available"] = True
        with patch(
            "bench.runtime.snapshot_sm121_cache_observability_metrics",
            side_effect=[dict(stable), dict(stable)],
        ):
            snapshot, elapsed, polls, settled = (
                runtime.settle_sm121_cache_observability_metrics(server)
            )
        self.assertEqual(snapshot, stable)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(polls, 2)
        self.assertTrue(settled)

        unavailable = runtime._sm121_cache_metric_defaults()
        with patch(
            "bench.runtime.snapshot_sm121_cache_observability_metrics",
            return_value=unavailable,
        ):
            _, _, polls, settled = runtime.settle_sm121_cache_observability_metrics(
                server
            )
        self.assertEqual(polls, 1)
        self.assertFalse(settled)

        with patch(
            "bench.runtime.snapshot_sm121_cache_observability_metrics",
            side_effect=[dict(stable), dict(stable)],
        ) as snapshot:
            runtime.settle_sm121_cache_observability_metrics(
                server, semantic_arm=SM121_CACHE_SEMANTIC_CACHE_ON_ARM
            )
        self.assertEqual(
            snapshot.call_args_list,
            [
                call(
                    server, semantic_arm=SM121_CACHE_SEMANTIC_CACHE_ON_ARM
                ),
                call(
                    server, semantic_arm=SM121_CACHE_SEMANTIC_CACHE_ON_ARM
                ),
            ],
        )

    def test_static_source_and_startup_attestations_are_scalar_only(self) -> None:
        workspace = Path(__file__).resolve().parents[1]
        model = load_models(workspace / "manifests" / "models.toml")[
            SM121_STORAGE_PROFILE_ID
        ]
        source_output = "\n".join(
            f"{SM121_CACHE_SOURCE_DIGESTS[field].removeprefix('sha256:')}  {path}"
            for field, path in runtime._SM121_CACHE_SOURCE_FILES.items()
        )
        command_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=source_output, stderr=""
        )
        with (
            patch(
                "bench.runtime._sm121_storage_image_id",
                return_value=SM121_STORAGE_LOCAL_IMAGE_ID,
            ),
            patch("bench.runtime._run", return_value=command_result) as run,
        ):
            static = runtime.attest_sm121_cache_observability_static_source(model)
        validate_sm121_cache_static_attestation_event(static)
        command = run.call_args.args[0]
        self.assertIn("--network", command)
        self.assertIn("none", command)
        self.assertIn("--read-only", command)
        self.assertNotIn("--gpus", command)

        startup = (
            "Tree cache initialized: source=default impl=ChunkCache "
            "hybrid_swa=False hybrid_ssm=True hicache_attached=False "
            "streaming_wrapped=False"
        )
        logs_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=startup, stderr=""
        )
        with (
            patch("bench.runtime._run", return_value=logs_result),
            patch(
                "bench.runtime._sm121_cache_server_info_disable_radix",
                return_value=True,
            ),
        ):
            observed = runtime.attest_sm121_cache_observability_runtime(self._server())
        validate_sm121_cache_runtime_attestation_event(observed)

    def test_b0_runtime_flag_reader_does_not_require_semantic_fields(self) -> None:
        """B0 only binds cache-off; the paired lane owns richer server-info checks."""

        with patch(
            "bench.runtime.urllib.request.urlopen",
            return_value=_Response(
                json.dumps({"server_args": {"disable_radix_cache": True}}).encode(
                    "utf-8"
                )
            ),
        ):
            self.assertIs(
                runtime._sm121_cache_server_info_disable_radix(self._server()),
                True,
            )

    def test_zero_hit_request_is_loopback_and_returns_only_scalars(self) -> None:
        server = self._server()
        private_response_id = "SYNTHETIC_PRIVATE_REQUEST_ID"
        payload = {
            "id": private_response_id,
            "choices": [
                {"message": {"content": SM121_CACHE_ZERO_HIT_EXPECTED_RESPONSE + " "}}
            ],
            "sglext": None,
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 1,
                "reasoning_tokens": 0,
                "prompt_tokens_details": None,
            },
        }
        response = _Response(json.dumps(payload).encode("utf-8"))
        with (
            patch("bench.runtime.urllib.request.urlopen", return_value=response) as urlopen,
            patch("bench.runtime.time.monotonic", side_effect=[10.0, 10.25]),
        ):
            result = runtime.request_sm121_cache_observability_zero_hit(
                server, served_name="synthetic-served-model"
            )
        self.assertEqual(
            result,
            {
                "prompt_tokens": 7,
                "completion_tokens": 1,
                "reasoning_tokens": 0,
                "elapsed_s": 0.25,
                "response_detail_state": "null",
                "response_device_cached_tokens": None,
                "response_host_cached_tokens": None,
                "response_storage_cached_tokens": None,
                "usage_detail_state": "null",
                "usage_cached_tokens": None,
            },
        )
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(json.loads(request.data), sm121_cache_zero_hit_request_body("synthetic-served-model"))
        self.assertEqual(response.read_sizes, [1_048_577])
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn(private_response_id, serialized)

        non_loopback = self._server("http://localhost:30000/v1")
        with patch("bench.runtime.urllib.request.urlopen") as urlopen:
            with self.assertRaises(runtime.RuntimeErrorWithContext):
                runtime.request_sm121_cache_observability_zero_hit(
                    non_loopback, served_name="synthetic-served-model"
                )
        urlopen.assert_not_called()

    def test_zero_hit_request_rejects_invalid_choice_topology_or_content(self) -> None:
        server = self._server()
        valid_usage = {
            "prompt_tokens": 7,
            "completion_tokens": 1,
            "reasoning_tokens": 0,
        }
        invalid_cases = (
            (
                "multiple choices",
                [
                    {"message": {"content": SM121_CACHE_ZERO_HIT_EXPECTED_RESPONSE}},
                    {"message": {"content": SM121_CACHE_ZERO_HIT_EXPECTED_RESPONSE}},
                ],
                "lacks required scalars",
            ),
            (
                "wrong content",
                [{"message": {"content": "CACHE-OBS-42"}}],
                "failed validation",
            ),
        )
        for name, choices, error_pattern in invalid_cases:
            with self.subTest(name=name):
                response = _Response(
                    json.dumps({"choices": choices, "usage": valid_usage}).encode("utf-8")
                )
                with patch("bench.runtime.urllib.request.urlopen", return_value=response):
                    with self.assertRaisesRegex(
                        runtime.RuntimeErrorWithContext, error_pattern
                    ):
                        runtime.request_sm121_cache_observability_zero_hit(
                            server, served_name="synthetic-served-model"
                        )


class SM121CacheObservabilityExecutorTests(unittest.TestCase):
    def test_log_capture_failure_still_stops_the_owned_server(self) -> None:
        """B0 cleanup must not leak a running server when logs cannot be saved."""

        workspace = Path(__file__).resolve().parents[1]
        model = load_models(workspace / "manifests" / "models.toml")[
            SM121_STORAGE_PROFILE_ID
        ]
        suite = load_suite(
            workspace
            / "manifests"
            / "suites"
            / "qwen38_flash_next_sm121_triton_storage_cache_observability_canary.toml"
        )
        server = SimpleNamespace(
            backend="sglang",
            startup_s=0.0,
            container_id="synthetic-container",
            stop=Mock(),
        )
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            with (
                patch("bench.runner._image_digest", return_value=None),
                patch(
                    "bench.runner._sm121_storage_image_identity",
                    return_value={
                        "docker_image_id": SM121_STORAGE_LOCAL_IMAGE_ID,
                        "platform": SM121_STORAGE_PLATFORM,
                        "source_tree": SM121_STORAGE_SOURCE_TREE,
                    },
                ),
                patch("bench.runner._host_snapshot", return_value={"host": "test"}),
            ):
                run_dir = runner.create_sm121_cache_observability_plan(
                    model=model,
                    suite=suite,
                    results_root=results,
                    models_path=workspace / "manifests" / "models.toml",
                    suite_path=(
                        workspace
                        / "manifests"
                        / "suites"
                        / "qwen38_flash_next_sm121_triton_storage_cache_observability_canary.toml"
                    ),
                )

            with (
                patch("bench.runner._preflight"),
                patch("bench.runner._host_safety_watchdog", return_value=None),
                patch("bench.runner.TelemetrySampler", return_value=Mock()),
                patch("bench.runner.start_server", return_value=server),
                patch("bench.runner.capture_server_provenance", return_value={}),
                patch(
                    "bench.runner.attest_sm121_cache_observability_static_source",
                    return_value={"event": SM121_CACHE_STATIC_ATTESTATION_EVENT},
                ),
                patch(
                    "bench.runner.attest_sm121_cache_observability_runtime",
                    return_value={"event": SM121_CACHE_RUNTIME_ATTESTATION_EVENT},
                ),
                patch("bench.runner._execute_case"),
                patch("bench.runner._require_sm121_storage_quality_gate"),
                patch("bench.runner._execute_sm121_cache_observability_case"),
                patch(
                    "bench.runner.save_server_logs",
                    side_effect=OSError("synthetic log capture failure"),
                ),
                patch("bench.runner.summarize_run", return_value={"status": "aborted"}),
            ):
                with self.assertRaisesRegex(OSError, "synthetic log capture failure"):
                    runner.execute_sm121_cache_observability_canary(
                        run_dir, workspace=results
                    )

            server.stop.assert_called_once_with()
            events = Journal(run_dir / "events.jsonl").events()
            self.assertTrue(any(event["event"] == "server_stopped" for event in events))
            self.assertTrue(any(event["event"] == "cleanup_failed" for event in events))
            self.assertTrue(any(event["event"] == "run_aborted" for event in events))


if __name__ == "__main__":
    unittest.main()
