"""Offline contracts for the private byte-bound SM121 C1 client."""

from __future__ import annotations

import io
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error

from bench import sm121_agent_admission_client as client_module
from bench.manifest import load_models
from bench.sglang_sm121_agent_admission import (
    SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_PROMPT_TOKENS,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_PROMPT_REPETITIONS,
    SM121_AGENT_ADMISSION_PROFILE_ID,
    SM121_AGENT_ADMISSION_QUALITY_CASE_ID,
)


ROOT = Path(__file__).resolve().parents[1]


def _event(value: dict[str, object]) -> bytes:
    return ("data: " + json.dumps(value, separators=(",", ":")) + "\n\n").encode(
        "utf-8"
    )


class _Response:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):
        return iter(self._lines)


class _DeadlineResponse(_Response):
    """Synthetic streaming response that records socket deadline updates."""

    def __init__(self, lines: list[bytes]) -> None:
        super().__init__(lines)
        self.socket_timeouts: list[float] = []
        self.fp = SimpleNamespace(
            raw=SimpleNamespace(
                _sock=SimpleNamespace(settimeout=self.socket_timeouts.append)
            )
        )


def _response(
    *,
    content: str = "ok",
    finish_reason: str = "stop",
    tool_calls: list[dict[str, object]] | None = None,
    cached_tokens: int | None = None,
    prompt_tokens: int = 11,
) -> _Response:
    delta: dict[str, object] = {}
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    else:
        delta["content"] = content
    usage: dict[str, object] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 3,
    }
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return _Response(
        [
            _event(
                {
                    "model": client_module.SM121_AGENT_ADMISSION_SERVED_NAME,
                    "choices": [
                        {"delta": delta, "finish_reason": finish_reason}
                    ]
                }
            ),
            _event(
                {
                    "model": client_module.SM121_AGENT_ADMISSION_SERVED_NAME,
                    "usage": usage,
                    "choices": [],
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )


class SM121AgentAdmissionClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_models(ROOT / "manifests" / "models.toml")[
            SM121_AGENT_ADMISSION_PROFILE_ID
        ]

    def _server(self, *, auth: str = "Bearer fixture"):
        return SimpleNamespace(
            backend="sglang",
            base_url=client_module.SM121_AGENT_ADMISSION_LOOPBACK_ENDPOINT,
            authorization=auth,
        )

    def _client(self, case_id: str):
        return client_module.create_sm121_agent_admission_client(
            server=self._server(),
            model=self.model,
            case_id=case_id,
        )

    def test_factory_rejects_a_non_c1_model_even_if_its_served_name_matches(self) -> None:
        lookalike = SimpleNamespace(
            id="unrelated-model",
            served_name=client_module.SM121_AGENT_ADMISSION_SERVED_NAME,
        )
        with self.assertRaises(client_module.SM121AgentAdmissionRequestError):
            client_module.create_sm121_agent_admission_client(
                server=self._server(),
                model=lookalike,
                case_id=SM121_AGENT_ADMISSION_QUALITY_CASE_ID,
            )

    def test_quality_sends_the_exact_prevalidated_bytes_once(self) -> None:
        client = self._client(SM121_AGENT_ADMISSION_QUALITY_CASE_ID)
        prompt = "synthetic quality marker"
        expected = client._serialized_body(
            messages=[{"role": "user", "content": prompt}]
        )
        opener = Mock(return_value=_response(content="FINAL: 7"))

        with patch(
            "bench.sm121_agent_admission_client._open_loopback_request", opener
        ):
            result = client.run_quality_turn(prompt=prompt)

        self.assertEqual(result.content, "FINAL: 7")
        self.assertEqual(opener.call_count, 1)
        request = opener.call_args.args[0]
        self.assertEqual(request.data, expected)
        self.assertEqual(
            opener.call_args.kwargs,
            {"timeout_s": client_module.SM121_AGENT_ADMISSION_REQUEST_TIMEOUT_S},
        )
        body = json.loads(expected)
        self.assertEqual(
            body,
            {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "reasoning_effort": "low",
                },
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
                "model": self.model.served_name,
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": 0.0,
            },
        )
        diagnostics = client.diagnostics().to_dict()
        self.assertEqual(
            diagnostics,
            {
                "outbound_body_count": 1,
                "validated_low_thinking_body_count": 1,
                "validated_tool_body_count": 0,
                "validated_cache_zero_body_count": 0,
                "transport_attempt_count": 1,
                "transport_retry_count": 0,
                "payload_contract_verified": True,
            },
        )
        self.assertEqual(
            client_module.validate_c1_payload_diagnostics(diagnostics), diagnostics
        )

    def test_controller_deadline_caps_open_and_every_streaming_read(self) -> None:
        client = self._client(SM121_AGENT_ADMISSION_QUALITY_CASE_ID)
        client._bind_controller_deadline(client_module.time.monotonic() + 10.0)
        response = _DeadlineResponse(_response(content="FINAL: 83")._lines)
        opener = Mock(return_value=response)
        with patch(
            "bench.sm121_agent_admission_client._open_loopback_request", opener
        ):
            client.run_quality_turn(prompt="synthetic quality prompt")
        timeout = opener.call_args.kwargs["timeout_s"]
        self.assertGreater(timeout, 0)
        self.assertLessEqual(timeout, 10.0)
        self.assertTrue(response.socket_timeouts)
        self.assertTrue(all(0 < value <= 10.0 for value in response.socket_timeouts))

        expired = self._client(SM121_AGENT_ADMISSION_QUALITY_CASE_ID)
        with patch.object(client_module.time, "monotonic", return_value=10.0):
            expired._bind_controller_deadline(11.0)
        with (
            patch.object(client_module.time, "monotonic", return_value=11.0),
            patch(
                "bench.sm121_agent_admission_client._open_loopback_request"
            ) as blocked_open,
            self.assertRaises(client_module.SM121AgentAdmissionRequestError),
        ):
            expired.run_quality_turn(prompt="synthetic quality prompt")
        blocked_open.assert_not_called()

    def test_long_context_adds_exact_tools_auto_and_cache_zero_receipt(self) -> None:
        client = self._client(SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID)
        expected = client._serialized_body(
            messages=[
                {"role": "user", "content": client_module._long_context_prompt()}
            ]
        )
        opener = Mock(
            return_value=_response(
                content=client_module._LONG_CONTEXT_EXPECTED_CONTENT,
                cached_tokens=0,
                prompt_tokens=SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_PROMPT_TOKENS,
            )
        )

        with patch(
            "bench.sm121_agent_admission_client._open_loopback_request", opener
        ):
            result = client.run_long_context_turn()

        self.assertEqual(opener.call_args.args[0].data, expected)
        body = json.loads(expected)
        self.assertEqual(body["cache_prompt"], False)
        self.assertEqual(body["return_cached_tokens_details"], True)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(
            body["messages"][0]["content"].count(client_module._LONG_CONTEXT_FILLER),
            SM121_AGENT_ADMISSION_LONG_CONTEXT_PROMPT_REPETITIONS,
        )
        self.assertEqual(body["tools"], client_module._expected_tools(
            SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID, 0
        ))
        self.assertEqual(
            client.long_context_receipt(),
            {
                "input_tokenization_verified": True,
                "context_fit": True,
                "zero_response_cache_hits": True,
                "response_semantics_verified": True,
                "first_turn_only": True,
            },
        )
        self.assertEqual(
            client.diagnostics().to_dict(),
            {
                "outbound_body_count": 1,
                "validated_low_thinking_body_count": 1,
                "validated_tool_body_count": 1,
                "validated_cache_zero_body_count": 1,
                "transport_attempt_count": 1,
                "transport_retry_count": 0,
                "payload_contract_verified": True,
            },
        )
        with self.assertRaises(client_module.SM121AgentAdmissionRequestError):
            client.run_long_context_turn()

    def test_long_context_zero_requires_an_exact_first_turn_receipt(self) -> None:
        for cached, expected in ((0, True), (1, False), (None, False), (False, False)):
            with self.subTest(cached=cached):
                client = self._client(SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID)
                opener = Mock(
                    return_value=_response(
                        content=client_module._LONG_CONTEXT_EXPECTED_CONTENT,
                        cached_tokens=cached,
                        prompt_tokens=(
                            SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_PROMPT_TOKENS
                        ),
                    )
                )
                with patch(
                    "bench.sm121_agent_admission_client._open_loopback_request",
                    opener,
                ):
                    client.run_long_context_turn()
                self.assertIs(
                    client.long_context_receipt()["zero_response_cache_hits"],
                    expected,
                )

    def test_long_context_receipt_rejects_short_input_tokenization(self) -> None:
        client = self._client(SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID)
        opener = Mock(
            return_value=_response(
                content=client_module._LONG_CONTEXT_EXPECTED_CONTENT,
                cached_tokens=0,
                prompt_tokens=1,
            )
        )

        with patch(
            "bench.sm121_agent_admission_client._open_loopback_request", opener
        ):
            client.run_long_context_turn()

        receipt = client.long_context_receipt()
        self.assertFalse(receipt["input_tokenization_verified"])
        self.assertFalse(receipt["context_fit"])
        self.assertTrue(receipt["zero_response_cache_hits"])
        self.assertTrue(receipt["response_semantics_verified"])
        self.assertTrue(receipt["first_turn_only"])

    def test_long_context_receipt_rejects_a_noncanonical_template_token_count(self) -> None:
        client = self._client(SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID)
        opener = Mock(
            return_value=_response(
                content=client_module._LONG_CONTEXT_EXPECTED_CONTENT,
                cached_tokens=0,
                prompt_tokens=SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_PROMPT_TOKENS
                - 1,
            )
        )
        with patch(
            "bench.sm121_agent_admission_client._open_loopback_request", opener
        ):
            client.run_long_context_turn()
        receipt = client.long_context_receipt()
        self.assertFalse(receipt["input_tokenization_verified"])
        self.assertFalse(receipt["context_fit"])

    def test_long_context_receipt_rejects_tool_or_answer_drift(self) -> None:
        cases = (
            _response(
                content="WRONG-ANSWER",
                cached_tokens=0,
                prompt_tokens=SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_PROMPT_TOKENS,
            ),
            _response(
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "multiply", "arguments": "{}"},
                    }
                ],
                cached_tokens=0,
                prompt_tokens=SM121_AGENT_ADMISSION_LONG_CONTEXT_CHAT_PROMPT_TOKENS,
            ),
        )
        for response in cases:
            with self.subTest(response=response):
                client = self._client(SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID)
                opener = Mock(return_value=response)
                with patch(
                    "bench.sm121_agent_admission_client._open_loopback_request",
                    opener,
                ):
                    client.run_long_context_turn()
                self.assertFalse(
                    client.long_context_receipt()["response_semantics_verified"]
                )

    def test_final_body_rejects_truthy_json_type_drift(self) -> None:
        client = self._client(SM121_AGENT_ADMISSION_QUALITY_CASE_ID)
        body = json.loads(
            client._serialized_body(
                messages=[{"role": "user", "content": "synthetic marker"}]
            )
        )
        body["chat_template_kwargs"]["enable_thinking"] = 1
        with self.assertRaises(client_module.SM121AgentAdmissionRequestError):
            client._validate_final_body(body)
        body["chat_template_kwargs"]["enable_thinking"] = True
        body["stream_options"]["include_usage"] = 1
        with self.assertRaises(client_module.SM121AgentAdmissionRequestError):
            client._validate_final_body(body)

    def test_agentic_loop_owns_two_no_retry_tool_turns(self) -> None:
        client = self._client("agentic-select-and-call")
        opener = Mock(
            side_effect=(
                _response(
                    finish_reason="tool_calls",
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {
                                "name": "multiply",
                                "arguments": '{"a":6,"b":7}',
                            },
                        }
                    ],
                ),
                _response(content="FINAL: 42"),
            )
        )

        with patch(
            "bench.sm121_agent_admission_client._open_loopback_request", opener
        ):
            result = client.run_agentic()

        self.assertTrue(result.passed)
        self.assertEqual(opener.call_count, 2)
        first = json.loads(opener.call_args_list[0].args[0].data)
        second = json.loads(opener.call_args_list[1].args[0].data)
        self.assertEqual(first["tool_choice"], "auto")
        self.assertEqual(first["chat_template_kwargs"]["reasoning_effort"], "low")
        self.assertNotIn("cache_prompt", first)
        self.assertEqual(second["messages"][-1]["role"], "tool")
        self.assertEqual(
            client.diagnostics().to_dict(),
            {
                "outbound_body_count": 2,
                "validated_low_thinking_body_count": 2,
                "validated_tool_body_count": 2,
                "validated_cache_zero_body_count": 0,
                "transport_attempt_count": 2,
                "transport_retry_count": 0,
                "payload_contract_verified": True,
            },
        )

    def test_agentic_body_drift_fails_before_a_transport_attempt(self) -> None:
        client = self._client("agentic-select-and-call")
        scenario = client_module.agentic_tools._scenario("agentic-select-and-call", 0)
        opener = Mock()
        with patch(
            "bench.sm121_agent_admission_client._open_loopback_request", opener
        ), self.assertRaises(client_module.SM121AgentAdmissionRequestError):
            client._agentic_turn(
                prompt=scenario.prompt,
                max_tokens=4_096,
                temperature=0.0,
                request_id="synthetic-id",
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "reasoning_effort": "medium",
                    },
                    "messages": [
                        {"role": "system", "content": "synthetic system"},
                        {"role": "user", "content": scenario.prompt},
                    ],
                    "tools": client_module._expected_tools(
                        "agentic-select-and-call", 0
                    ),
                    "tool_choice": "auto",
                },
            )
        opener.assert_not_called()
        self.assertEqual(client.diagnostics().outbound_body_count, 0)

    def test_agentic_truthy_low_thinking_drift_fails_before_transport(self) -> None:
        client = self._client("agentic-select-and-call")
        scenario = client_module.agentic_tools._scenario("agentic-select-and-call", 0)
        opener = Mock()
        with patch(
            "bench.sm121_agent_admission_client._open_loopback_request", opener
        ), self.assertRaises(client_module.SM121AgentAdmissionRequestError):
            client._agentic_turn(
                prompt=scenario.prompt,
                max_tokens=4_096,
                temperature=False,
                request_id="synthetic-id",
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": 1,
                        "reasoning_effort": "low",
                    },
                    "messages": [
                        {"role": "system", "content": "synthetic system"},
                        {"role": "user", "content": scenario.prompt},
                    ],
                    "tools": client_module._expected_tools(
                        "agentic-select-and-call", 0
                    ),
                    "tool_choice": "auto",
                },
            )
        opener.assert_not_called()

    def test_transport_and_response_failures_do_not_expose_payload_markers(self) -> None:
        prompt = "PROMPT-SECRET-MARKER"
        auth = "Bearer " + "AUTH-SECRET-MARKER"
        client = client_module.create_sm121_agent_admission_client(
            server=self._server(auth=auth),
            model=self.model,
            case_id=SM121_AGENT_ADMISSION_QUALITY_CASE_ID,
        )
        http_error = urllib.error.HTTPError(
            "http://127.0.0.1:30000/HTTP-SECRET-MARKER",
            503,
            "synthetic",
            None,
            io.BytesIO(b"RESPONSE-SECRET-MARKER"),
        )
        opener = Mock(side_effect=http_error)
        with patch(
            "bench.sm121_agent_admission_client._open_loopback_request", opener
        ), self.assertRaises(client_module.SM121AgentAdmissionRequestError) as caught:
            client.run_quality_turn(prompt=prompt)
        rendered = str(caught.exception) + repr(client.diagnostics().to_dict())
        for marker in (
            "PROMPT-SECRET-MARKER",
            "AUTH-SECRET-MARKER",
            "HTTP-SECRET-MARKER",
            "RESPONSE-SECRET-MARKER",
        ):
            self.assertNotIn(marker, rendered)
        self.assertEqual(caught.exception.code, "transport")
        self.assertEqual(opener.call_count, 1)
        self.assertFalse(client.diagnostics().payload_contract_verified)

        malformed = self._client(SM121_AGENT_ADMISSION_QUALITY_CASE_ID)
        response_opener = Mock(
            return_value=_Response([b"data: MALFORMED-RESPONSE-MARKER\n\n"])
        )
        with patch(
            "bench.sm121_agent_admission_client._open_loopback_request", response_opener
        ), self.assertRaises(client_module.SM121AgentAdmissionRequestError) as caught:
            malformed.run_quality_turn(prompt=prompt)
        self.assertEqual(caught.exception.code, "response")
        self.assertNotIn("MALFORMED-RESPONSE-MARKER", str(caught.exception))
        self.assertEqual(response_opener.call_count, 1)

    def test_response_budget_fails_closed_before_unbounded_accumulation(self) -> None:
        client = self._client(SM121_AGENT_ADMISSION_QUALITY_CASE_ID)
        opener = Mock(return_value=_Response([b"data: {}\n\n"]))
        with patch(
            "bench.sm121_agent_admission_client._open_loopback_request", opener
        ), patch.object(client_module, "_MAX_RESPONSE_BYTES", 1), self.assertRaises(
            client_module.SM121AgentAdmissionRequestError
        ) as caught:
            client.run_quality_turn(prompt="synthetic prompt")
        self.assertEqual(caught.exception.code, "response")
        self.assertEqual(opener.call_count, 1)

    def test_factory_has_no_hook_or_transport_override_surface(self) -> None:
        parameters = set(inspect.signature(
            client_module.create_sm121_agent_admission_client
        ).parameters)
        self.assertEqual(parameters, {"server", "model", "case_id", "variant"})
        with self.assertRaises(TypeError):
            client_module.create_sm121_agent_admission_client(
                server=self._server(),
                model=self.model,
                case_id=SM121_AGENT_ADMISSION_QUALITY_CASE_ID,
                transport=object(),
            )
        wrong_endpoint = SimpleNamespace(
            backend="sglang",
            base_url="http://127.0.0.1:30001/v1",
            authorization="Bearer fixture",
        )
        with self.assertRaises(client_module.SM121AgentAdmissionRequestError):
            client_module.create_sm121_agent_admission_client(
                server=wrong_endpoint,
                model=self.model,
                case_id=SM121_AGENT_ADMISSION_QUALITY_CASE_ID,
            )

    def test_diagnostics_reject_freeform_and_inconsistent_values(self) -> None:
        with self.assertRaises(client_module.SM121AgentAdmissionRequestError):
            client_module.validate_c1_payload_diagnostics(
                {
                    "outbound_body_count": 1,
                    "validated_low_thinking_body_count": 1,
                    "validated_tool_body_count": 0,
                    "validated_cache_zero_body_count": 0,
                    "transport_attempt_count": 1,
                    "transport_retry_count": 0,
                    "payload_contract_verified": True,
                    "message": "not scalar-schema safe",
                }
            )
        with self.assertRaises(client_module.SM121AgentAdmissionRequestError):
            client_module.validate_c1_payload_diagnostics(
                {
                    "outbound_body_count": 1,
                    "validated_low_thinking_body_count": 1,
                    "validated_tool_body_count": 0,
                    "validated_cache_zero_body_count": 0,
                    "transport_attempt_count": 2,
                    "transport_retry_count": 0,
                    "payload_contract_verified": True,
                }
            )


if __name__ == "__main__":
    unittest.main()
