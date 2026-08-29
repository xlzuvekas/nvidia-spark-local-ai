from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest import mock

from bench import runtime


class _Response:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False

    def read(self, limit: int = -1) -> bytes:
        del limit
        return self.payload


def _server() -> SimpleNamespace:
    return SimpleNamespace(
        backend="sglang",
        container_id="synthetic-container",
        base_url="http://127.0.0.1:30000/v1",
        authorization="Bearer synthetic",
    )


def _payload(*, prompt_token_ids: object = None) -> dict[str, object]:
    if prompt_token_ids is None:
        prompt_token_ids = [1, 2, 3]
    return {
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "reasoning_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
        "choices": [
            {
                "index": 0,
                "message": {"content": "SYNTHETIC-ANSWER"},
                "prompt_token_ids": prompt_token_ids,
            }
        ],
        "sglext": {"cached_tokens_details": {"device": 2, "host": 0, "storage": 0}},
    }


class SM121CacheSemanticRuntimeTests(unittest.TestCase):
    def test_semantic_turn_returns_private_ids_and_scalar_observations(self) -> None:
        with mock.patch.object(
            runtime.urllib.request,
            "urlopen",
            return_value=_Response(_payload()),
        ) as urlopen:
            result = runtime.request_sm121_cache_semantic_turn(
                _server(),
                served_name="synthetic-model",
                messages=[{"role": "user", "content": "synthetic prompt"}],
                expected_response="SYNTHETIC-ANSWER",
                max_tokens=16,
            )

        self.assertEqual(result["private_prompt_token_ids"], (1, 2, 3))
        self.assertEqual(result["prompt_tokens"], 3)
        self.assertEqual(result["response_detail_state"], "nonzero_details")
        self.assertEqual(result["response_device_cached_tokens"], 2)
        self.assertEqual(result["usage_cached_tokens"], 2)
        body = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertTrue(body["return_cached_tokens_details"])
        self.assertTrue(body["return_prompt_token_ids"])
        self.assertFalse(body["stream"])

    def test_semantic_turn_rejects_invalid_private_prompt_ids(self) -> None:
        with mock.patch.object(
            runtime.urllib.request,
            "urlopen",
            return_value=_Response(_payload(prompt_token_ids=[1, True, 3])),
        ):
            with self.assertRaisesRegex(
                runtime.RuntimeErrorWithContext, "prompt IDs are invalid"
            ):
                runtime.request_sm121_cache_semantic_turn(
                    _server(),
                    served_name="synthetic-model",
                    messages=[{"role": "user", "content": "synthetic prompt"}],
                    expected_response="SYNTHETIC-ANSWER",
                    max_tokens=16,
                )

    def test_semantic_turn_accepts_source_omitted_storage_detail(self) -> None:
        payload = _payload()
        payload["sglext"] = {"cached_tokens_details": {"device": 2, "host": 0}}
        with mock.patch.object(
            runtime.urllib.request,
            "urlopen",
            return_value=_Response(payload),
        ):
            result = runtime.request_sm121_cache_semantic_turn(
                _server(),
                served_name="synthetic-model",
                messages=[{"role": "user", "content": "synthetic prompt"}],
                expected_response="SYNTHETIC-ANSWER",
                max_tokens=16,
            )

        self.assertEqual(result["response_detail_state"], "nonzero_details")
        self.assertEqual(result["response_device_cached_tokens"], 2)
        self.assertEqual(result["response_host_cached_tokens"], 0)
        self.assertEqual(result["response_storage_cached_tokens"], 0)

    def test_runtime_identity_uses_resolved_cache_fields(self) -> None:
        startup = (
            "Tree cache initialized: source=default impl=UnifiedRadixCache "
            "hybrid_swa=False hybrid_ssm=True hicache_attached=False "
            "streaming_wrapped=False"
        )
        with (
            mock.patch.object(
                runtime,
                "_run",
                return_value=SimpleNamespace(returncode=0, stdout=startup, stderr=""),
            ),
            mock.patch.object(
                runtime,
                "_sm121_cache_server_info_fields",
                return_value={
                    "disable_radix_cache": False,
                    "mamba_radix_cache_strategy": "extra_buffer_lazy",
                    "max_mamba_cache_size": 4,
                },
            ),
        ):
            observed = runtime.inspect_sm121_cache_runtime_identity(_server())

        self.assertEqual(observed["cache_impl"], "UnifiedRadixCache")
        self.assertFalse(observed["disable_radix_cache"])
        self.assertTrue(observed["mamba_extra_buffer_enabled"])
        self.assertTrue(observed["mamba_extra_buffer_lazy_enabled"])
        self.assertEqual(observed["max_mamba_cache_size"], 4)

    def test_chunked_prefill_runtime_identity_attests_one_size(self) -> None:
        startup = (
            "Tree cache initialized: source=default impl=UnifiedRadixCache "
            "hybrid_swa=False hybrid_ssm=True hicache_attached=False "
            "streaming_wrapped=False"
        )
        with (
            mock.patch.object(
                runtime,
                "_run",
                return_value=SimpleNamespace(returncode=0, stdout=startup, stderr=""),
            ),
            mock.patch.object(
                runtime,
                "_sm121_cache_server_info_fields",
                return_value={
                    "disable_radix_cache": False,
                    "mamba_radix_cache_strategy": "extra_buffer_lazy",
                    "max_mamba_cache_size": 4,
                },
            ),
            mock.patch.object(
                runtime.urllib.request,
                "urlopen",
                return_value=_Response(
                    {"scheduler": {"chunked_prefill_size": 2048}}
                ),
            ),
        ):
            observed = runtime.inspect_sm121_chunked_prefill_runtime_identity(
                _server()
            )

        self.assertEqual(2048, observed["chunked_prefill_size"])
        self.assertEqual("UnifiedRadixCache", observed["cache_impl"])

    def test_chunked_prefill_runtime_identity_rejects_conflicting_sizes(self) -> None:
        with mock.patch.object(
            runtime,
            "inspect_sm121_cache_runtime_identity",
            return_value={"cache_impl": "UnifiedRadixCache"},
        ), mock.patch.object(
            runtime.urllib.request,
            "urlopen",
            return_value=_Response(
                {
                    "scheduler": {"chunked_prefill_size": 1024},
                    "model": {"chunked_prefill_size": 2048},
                }
            ),
        ):
            with self.assertRaisesRegex(
                runtime.RuntimeErrorWithContext, "chunked-prefill runtime field"
            ):
                runtime.inspect_sm121_chunked_prefill_runtime_identity(_server())


if __name__ == "__main__":
    unittest.main()
