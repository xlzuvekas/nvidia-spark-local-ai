"""Private byte-bound client for the prospective SM121 C1 agent admission.

This module is deliberately separate from the generic benchmark client.  It
constructs each final OpenAI-compatible body itself, validates the strict
low-thinking/tools/cache-zero contract from the exact serialized bytes, and
sends those same bytes once through a loopback-only, no-proxy, no-redirect
transport.  Request and response text remain in memory only; diagnostics are
scalar-only and are intended for the private admission controller.

Importing this module does not start a server.  It is not an execution surface:
the C1 runner remains tombstoned until its dedicated lifetime controller is
implemented and reviewed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import secrets
import time
from typing import Any, Mapping
import urllib.error
import urllib.request

from . import agentic_tools
from .agentic_tools import AgenticRunResult
from .client import RequestResult
from .sglang_sm121_agent_admission import (
    SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID,
    SM121_AGENT_ADMISSION_ENDPOINT,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_MIN_PROMPT_TOKENS,
    SM121_AGENT_ADMISSION_LONG_CONTEXT_PROMPT_REPETITIONS,
    SM121_AGENT_ADMISSION_QUALITY_CASE_ID,
    SM121_AGENT_ADMISSION_SERVED_NAME,
    SM121_AGENT_ADMISSION_TOOL_CASE_IDS,
    SM121AgentAdmissionError,
    validate_sm121_agent_admission_profile,
)
from .sglang_sm121_storage import SM121_STORAGE_CONTEXT_LENGTH


SM121_AGENT_ADMISSION_LOOPBACK_ENDPOINT = SM121_AGENT_ADMISSION_ENDPOINT
SM121_AGENT_ADMISSION_REQUEST_TIMEOUT_S = 900.0
_MAX_BODY_BYTES = 8 * 1024 * 1024
_MAX_SSE_LINE_BYTES = 2 * 1024 * 1024
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_RESPONSE_EVENTS = 1_024
_MAX_TOOL_FIELD_BYTES = 64 * 1024
_LOW_THINKING = {"enable_thinking": True, "reasoning_effort": "low"}
_LONG_CONTEXT_FILLER = "archive "
_LONG_CONTEXT_PROMPT_PREFIX = (
    "Read the complete synthetic context before answering. "
)
_LONG_CONTEXT_PROMPT_SUFFIX = (
    "Do not call a tool. Reply with exactly LONG-CONTEXT-READY."
)
_LONG_CONTEXT_EXPECTED_CONTENT = "LONG-CONTEXT-READY"
_CASE_OUTPUT_TOKENS = {
    SM121_AGENT_ADMISSION_QUALITY_CASE_ID: 512,
    **{case_id: 4_096 for case_id in SM121_AGENT_ADMISSION_TOOL_CASE_IDS},
    SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID: 128,
}
_AGENTIC_CASE_IDS = frozenset(SM121_AGENT_ADMISSION_TOOL_CASE_IDS)
_PUBLIC_DIAGNOSTIC_FIELDS = frozenset(
    {
        "outbound_body_count",
        "validated_low_thinking_body_count",
        "validated_tool_body_count",
        "validated_cache_zero_body_count",
        "transport_attempt_count",
        "transport_retry_count",
        "payload_contract_verified",
    }
)


class SM121AgentAdmissionRequestError(RuntimeError):
    """A fixed-message C1 transport or body-contract failure.

    The stable ``code`` is intentionally coarse.  It never includes an
    endpoint, authorization value, request identifier, request body, or server
    response data.
    """

    _CODES = frozenset({"body", "response", "transport"})

    def __init__(self, code: str) -> None:
        self.code = code if code in self._CODES else "transport"
        super().__init__("SM121 agent admission request failed")


@dataclass(frozen=True, slots=True)
class C1PayloadDiagnostics:
    """The sole scalar payload/transport projection for the C1 controller."""

    outbound_body_count: int
    validated_low_thinking_body_count: int
    validated_tool_body_count: int
    validated_cache_zero_body_count: int
    transport_attempt_count: int
    transport_retry_count: int
    payload_contract_verified: bool

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "outbound_body_count": self.outbound_body_count,
            "validated_low_thinking_body_count": (
                self.validated_low_thinking_body_count
            ),
            "validated_tool_body_count": self.validated_tool_body_count,
            "validated_cache_zero_body_count": (
                self.validated_cache_zero_body_count
            ),
            "transport_attempt_count": self.transport_attempt_count,
            "transport_retry_count": self.transport_retry_count,
            "payload_contract_verified": self.payload_contract_verified,
        }


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-standard JSON constant")


def _exact_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _authorization_headers(authorization: object) -> dict[str, str]:
    if (
        not isinstance(authorization, str)
        or not authorization.startswith("Bearer ")
        or len(authorization) <= len("Bearer ")
        or "\r" in authorization
        or "\n" in authorization
    ):
        raise SM121AgentAdmissionRequestError("transport")
    return {"Content-Type": "application/json", "Authorization": authorization}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        return None


def _open_loopback_request(
    request: urllib.request.Request, *, timeout_s: float
) -> Any:
    """Open the exact C1 request once with proxies and redirects disabled."""

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirect(),
    )
    return opener.open(request, timeout=timeout_s)


def _expected_tools(case_id: str, variant: int) -> list[dict[str, Any]]:
    if case_id in _AGENTIC_CASE_IDS:
        scenario = agentic_tools._scenario(case_id, variant)
    elif case_id == SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID:
        # The long-context first-turn gate needs the same canonical tool schema
        # shape as the deterministic select/call episode, but it does not run a
        # tool loop.  Keeping this derived from the shared source avoids a
        # second hand-maintained tool-schema copy.
        scenario = agentic_tools._scenario("agentic-select-and-call", 0)
    else:
        raise SM121AgentAdmissionRequestError("body")
    return agentic_tools._rotated_tool_schemas(scenario)


def _validate_string(value: object, *, allow_empty: bool = False) -> bool:
    return isinstance(value, str) and (allow_empty or bool(value))


def _is_low_thinking(value: object) -> bool:
    return (
        type(value) is dict
        and set(value) == {"enable_thinking", "reasoning_effort"}
        and value.get("enable_thinking") is True
        and value.get("reasoning_effort") == "low"
    )


def _is_zero_temperature(value: object) -> bool:
    return (
        type(value) in {int, float}
        and type(value) is not bool
        and float(value) == 0.0
    )


def _validate_tool_call(value: object) -> bool:
    if type(value) is not dict or set(value) != {"id", "type", "function"}:
        return False
    function = value.get("function")
    return (
        _validate_string(value.get("id"))
        and value.get("type") == "function"
        and type(function) is dict
        and set(function) == {"name", "arguments"}
        and _validate_string(function.get("name"))
        and _validate_string(function.get("arguments"), allow_empty=True)
    )


def _validate_messages(messages: object, *, kind: str) -> bool:
    if type(messages) is not list or not messages or len(messages) > 40:
        return False
    if kind in {"quality", "long_context"}:
        return (
            len(messages) == 1
            and type(messages[0]) is dict
            and set(messages[0]) == {"role", "content"}
            and messages[0].get("role") == "user"
            and _validate_string(messages[0].get("content"))
        )
    if kind != "agentic" or len(messages) < 2:
        return False
    for index, message in enumerate(messages):
        if type(message) is not dict:
            return False
        role = message.get("role")
        if index == 0:
            if (
                set(message) != {"role", "content"}
                or role != "system"
                or not _validate_string(message.get("content"))
            ):
                return False
            continue
        if index == 1:
            if (
                set(message) != {"role", "content"}
                or role != "user"
                or not _validate_string(message.get("content"))
            ):
                return False
            continue
        if role == "assistant":
            if set(message) != {"role", "content", "tool_calls"}:
                return False
            if message.get("content") is not None and not isinstance(
                message.get("content"), str
            ):
                return False
            calls = message.get("tool_calls")
            if type(calls) is not list or not calls or not all(
                _validate_tool_call(call) for call in calls
            ):
                return False
        elif role == "tool":
            if set(message) != {"role", "tool_call_id", "name", "content"}:
                return False
            if not all(
                _validate_string(message.get(field))
                for field in ("tool_call_id", "name", "content")
            ):
                return False
        else:
            return False
    return True


def _cached_prompt_tokens(usage: Mapping[str, Any]) -> int | None:
    details = usage.get("prompt_tokens_details")
    if type(details) is not dict:
        return None
    return _exact_nonnegative_int(details.get("cached_tokens"))


def _long_context_prompt() -> str:
    """Render the exact 60K first-turn C1 context entirely in memory."""

    return (
        _LONG_CONTEXT_PROMPT_PREFIX
        + _LONG_CONTEXT_FILLER * SM121_AGENT_ADMISSION_LONG_CONTEXT_PROMPT_REPETITIONS
        + _LONG_CONTEXT_PROMPT_SUFFIX
    )


class _SM121AgentAdmissionClient:
    """One case-bound private request client with no caller-selected transport."""

    def __init__(
        self,
        *,
        server: object,
        model: object,
        case_id: str,
        variant: int,
    ) -> None:
        try:
            validate_sm121_agent_admission_profile(model)
        except SM121AgentAdmissionError as error:
            raise SM121AgentAdmissionRequestError("body") from error
        if case_id not in _CASE_OUTPUT_TOKENS:
            raise SM121AgentAdmissionRequestError("body")
        if (
            isinstance(variant, bool)
            or not isinstance(variant, int)
            or variant != 0
        ):
            raise SM121AgentAdmissionRequestError("body")
        if (
            getattr(server, "backend", None) != "sglang"
            or getattr(server, "base_url", None)
            != SM121_AGENT_ADMISSION_LOOPBACK_ENDPOINT
        ):
            raise SM121AgentAdmissionRequestError("transport")
        served_name = getattr(model, "served_name", None)
        if served_name != SM121_AGENT_ADMISSION_SERVED_NAME:
            raise SM121AgentAdmissionRequestError("body")
        self._headers = _authorization_headers(getattr(server, "authorization", None))
        self._case_id = case_id
        self._variant = variant
        self._outbound_body_count = 0
        self._validated_low_thinking_body_count = 0
        self._validated_tool_body_count = 0
        self._validated_cache_zero_body_count = 0
        self._transport_attempt_count = 0
        self._successful_response_count = 0
        self._long_context_prompt_tokens: int | None = None
        self._long_context_cached_prompt_tokens: int | None = None
        self._long_context_response_semantics_verified = False

    def diagnostics(self) -> C1PayloadDiagnostics:
        verified = (
            self._outbound_body_count > 0
            and self._outbound_body_count
            == self._validated_low_thinking_body_count
            and self._outbound_body_count == self._transport_attempt_count
            and self._outbound_body_count == self._successful_response_count
            and (
                self._case_id == SM121_AGENT_ADMISSION_QUALITY_CASE_ID
                or self._outbound_body_count == self._validated_tool_body_count
            )
            and (
                self._case_id != SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID
                or self._outbound_body_count
                == self._validated_cache_zero_body_count
            )
        )
        return C1PayloadDiagnostics(
            outbound_body_count=self._outbound_body_count,
            validated_low_thinking_body_count=(
                self._validated_low_thinking_body_count
            ),
            validated_tool_body_count=self._validated_tool_body_count,
            validated_cache_zero_body_count=(
                self._validated_cache_zero_body_count
            ),
            transport_attempt_count=self._transport_attempt_count,
            transport_retry_count=0,
            payload_contract_verified=verified,
        )

    def _kind(self) -> str:
        if self._case_id == SM121_AGENT_ADMISSION_QUALITY_CASE_ID:
            return "quality"
        if self._case_id == SM121_AGENT_ADMISSION_LONG_CONTEXT_CASE_ID:
            return "long_context"
        return "agentic"

    def _body_object(self, *, messages: list[dict[str, Any]]) -> dict[str, Any]:
        kind = self._kind()
        if not _validate_messages(messages, kind=kind):
            raise SM121AgentAdmissionRequestError("body")
        body: dict[str, Any] = {
            "model": SM121_AGENT_ADMISSION_SERVED_NAME,
            "messages": deepcopy(messages),
            "max_tokens": _CASE_OUTPUT_TOKENS[self._case_id],
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": dict(_LOW_THINKING),
        }
        if kind in {"agentic", "long_context"}:
            body["tools"] = _expected_tools(self._case_id, self._variant)
            body["tool_choice"] = "auto"
        if kind == "long_context":
            body["cache_prompt"] = False
            body["return_cached_tokens_details"] = True
        return body

    def _validate_final_body(self, body: object) -> None:
        kind = self._kind()
        if type(body) is not dict:
            raise SM121AgentAdmissionRequestError("body")
        expected_fields = {
            "model",
            "messages",
            "max_tokens",
            "temperature",
            "stream",
            "stream_options",
            "chat_template_kwargs",
        }
        if kind in {"agentic", "long_context"}:
            expected_fields |= {"tools", "tool_choice"}
        if kind == "long_context":
            expected_fields |= {"cache_prompt", "return_cached_tokens_details"}
        if set(body) != expected_fields:
            raise SM121AgentAdmissionRequestError("body")
        stream_options = body.get("stream_options")
        template = body.get("chat_template_kwargs")
        valid_stream_options = (
            type(stream_options) is dict
            and set(stream_options) == {"include_usage"}
            and stream_options.get("include_usage") is True
        )
        if (
            body.get("model") != SM121_AGENT_ADMISSION_SERVED_NAME
            or body.get("max_tokens") != _CASE_OUTPUT_TOKENS[self._case_id]
            or not _is_zero_temperature(body.get("temperature"))
            or body.get("stream") is not True
            or not valid_stream_options
            or not _is_low_thinking(template)
            or not _validate_messages(body.get("messages"), kind=kind)
        ):
            raise SM121AgentAdmissionRequestError("body")
        if kind in {"agentic", "long_context"} and (
            body.get("tool_choice") != "auto"
            or body.get("tools") != _expected_tools(self._case_id, self._variant)
        ):
            raise SM121AgentAdmissionRequestError("body")
        if kind == "long_context" and (
            body.get("cache_prompt") is not False
            or body.get("return_cached_tokens_details") is not True
        ):
            raise SM121AgentAdmissionRequestError("body")

    def _serialized_body(self, *, messages: list[dict[str, Any]]) -> bytes:
        try:
            body = self._body_object(messages=messages)
            encoded = json.dumps(
                body,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            if not encoded or len(encoded) > _MAX_BODY_BYTES:
                raise ValueError("body size")
            decoded = json.loads(
                encoded.decode("ascii"),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
            self._validate_final_body(decoded)
        except (
            SM121AgentAdmissionRequestError,
            TypeError,
            UnicodeError,
            ValueError,
            RecursionError,
        ) as error:
            if isinstance(error, SM121AgentAdmissionRequestError):
                raise
            raise SM121AgentAdmissionRequestError("body") from None
        return encoded

    def _record_validated_body(self) -> None:
        self._validated_low_thinking_body_count += 1
        if self._kind() in {"agentic", "long_context"}:
            self._validated_tool_body_count += 1
        if self._kind() == "long_context":
            self._validated_cache_zero_body_count += 1

    def _send(self, *, messages: list[dict[str, Any]], request_id: str) -> RequestResult:
        if not _validate_string(request_id):
            raise SM121AgentAdmissionRequestError("body")
        body = self._serialized_body(messages=messages)
        self._record_validated_body()
        request = urllib.request.Request(
            SM121_AGENT_ADMISSION_LOOPBACK_ENDPOINT + "/chat/completions",
            data=body,
            headers=self._headers,
        )
        self._outbound_body_count += 1
        self._transport_attempt_count += 1
        started_wall_ns = time.time_ns()
        started = time.perf_counter()
        first_output_at: float | None = None
        usage: dict[str, Any] | None = None
        finish_reason: str | None = None
        response_model: str | None = None
        emission_events = 0
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        response_bytes = 0
        response_events = 0
        try:
            with _open_loopback_request(
                request, timeout_s=SM121_AGENT_ADMISSION_REQUEST_TIMEOUT_S
            ) as response:
                for raw_line in response:
                    if (
                        not isinstance(raw_line, bytes)
                        or len(raw_line) > _MAX_SSE_LINE_BYTES
                    ):
                        raise SM121AgentAdmissionRequestError("response")
                    response_bytes += len(raw_line)
                    if response_bytes > _MAX_RESPONSE_BYTES:
                        raise SM121AgentAdmissionRequestError("response")
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    raw_data = line[5:].lstrip()
                    if raw_data == "[DONE]":
                        continue
                    response_events += 1
                    if response_events > _MAX_RESPONSE_EVENTS:
                        raise SM121AgentAdmissionRequestError("response")
                    try:
                        event = json.loads(
                            raw_data,
                            object_pairs_hook=_unique_json_object,
                            parse_constant=_reject_json_constant,
                        )
                    except (ValueError, json.JSONDecodeError, RecursionError):
                        raise SM121AgentAdmissionRequestError("response") from None
                    if type(event) is not dict:
                        raise SM121AgentAdmissionRequestError("response")
                    model_name = event.get("model")
                    if model_name is not None:
                        if model_name != SM121_AGENT_ADMISSION_SERVED_NAME:
                            raise SM121AgentAdmissionRequestError("response")
                        response_model = SM121_AGENT_ADMISSION_SERVED_NAME
                    observed_usage = event.get("usage")
                    if observed_usage is not None:
                        if type(observed_usage) is not dict:
                            raise SM121AgentAdmissionRequestError("response")
                        usage = observed_usage
                    choices = event.get("choices") or []
                    if type(choices) is not list:
                        raise SM121AgentAdmissionRequestError("response")
                    for choice in choices:
                        if type(choice) is not dict:
                            raise SM121AgentAdmissionRequestError("response")
                        choice_finish = choice.get("finish_reason")
                        if choice_finish is not None and not isinstance(
                            choice_finish, str
                        ):
                            raise SM121AgentAdmissionRequestError("response")
                        if choice_finish:
                            finish_reason = choice_finish
                        delta = choice.get("delta") or {}
                        if type(delta) is not dict:
                            raise SM121AgentAdmissionRequestError("response")
                        content = delta.get("content")
                        reasoning = delta.get("reasoning")
                        if reasoning is None:
                            reasoning = delta.get("reasoning_content")
                        calls = delta.get("tool_calls") or []
                        if (
                            content is not None
                            and not isinstance(content, str)
                            or reasoning is not None
                            and not isinstance(reasoning, str)
                            or type(calls) is not list
                        ):
                            raise SM121AgentAdmissionRequestError("response")
                        if content or reasoning or calls:
                            emission_events += 1
                            if first_output_at is None:
                                first_output_at = time.perf_counter()
                        if isinstance(content, str):
                            content_parts.append(content)
                        if isinstance(reasoning, str):
                            reasoning_parts.append(reasoning)
                        for tool_delta in calls:
                            if type(tool_delta) is not dict:
                                raise SM121AgentAdmissionRequestError("response")
                            index = tool_delta.get("index", 0)
                            if (
                                isinstance(index, bool)
                                or not isinstance(index, int)
                                or index < 0
                                or index > 64
                            ):
                                raise SM121AgentAdmissionRequestError("response")
                            current = tool_calls.setdefault(
                                index,
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            call_id = tool_delta.get("id")
                            if call_id is not None:
                                if not isinstance(call_id, str):
                                    raise SM121AgentAdmissionRequestError("response")
                                if (
                                    len(current["id"].encode("utf-8"))
                                    + len(call_id.encode("utf-8"))
                                    > _MAX_TOOL_FIELD_BYTES
                                ):
                                    raise SM121AgentAdmissionRequestError("response")
                                current["id"] += call_id
                            function = tool_delta.get("function") or {}
                            if type(function) is not dict:
                                raise SM121AgentAdmissionRequestError("response")
                            name = function.get("name")
                            arguments = function.get("arguments")
                            if name is not None:
                                if not isinstance(name, str):
                                    raise SM121AgentAdmissionRequestError("response")
                                if (
                                    len(current["function"]["name"].encode("utf-8"))
                                    + len(name.encode("utf-8"))
                                    > _MAX_TOOL_FIELD_BYTES
                                ):
                                    raise SM121AgentAdmissionRequestError("response")
                                current["function"]["name"] += name
                            if arguments is not None:
                                if not isinstance(arguments, str):
                                    raise SM121AgentAdmissionRequestError("response")
                                if (
                                    len(
                                        current["function"]["arguments"].encode(
                                            "utf-8"
                                        )
                                    )
                                    + len(arguments.encode("utf-8"))
                                    > _MAX_TOOL_FIELD_BYTES
                                ):
                                    raise SM121AgentAdmissionRequestError("response")
                                current["function"]["arguments"] += arguments
        except SM121AgentAdmissionRequestError:
            raise
        except urllib.error.HTTPError as error:
            error.close()
            raise SM121AgentAdmissionRequestError("transport") from None
        except (OSError, urllib.error.URLError, TimeoutError, UnicodeError):
            raise SM121AgentAdmissionRequestError("transport") from None
        finished = time.perf_counter()
        if type(usage) is not dict or response_model != SM121_AGENT_ADMISSION_SERVED_NAME:
            raise SM121AgentAdmissionRequestError("response")
        prompt_tokens = _exact_nonnegative_int(usage.get("prompt_tokens"))
        completion_tokens = _exact_nonnegative_int(usage.get("completion_tokens"))
        if (
            prompt_tokens is None
            or completion_tokens is None
            or completion_tokens <= 0
            or first_output_at is None
        ):
            raise SM121AgentAdmissionRequestError("response")
        reasoning_tokens = _exact_nonnegative_int(usage.get("reasoning_tokens"))
        if reasoning_tokens is None:
            details = usage.get("completion_tokens_details")
            if type(details) is dict:
                reasoning_tokens = _exact_nonnegative_int(
                    details.get("reasoning_tokens")
                )
        elapsed_s = finished - started
        decode_s = max(finished - first_output_at, 1e-9)
        result = RequestResult(
            request_id=request_id,
            started_at_ns=started_wall_ns,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            ttft_s=first_output_at - started,
            elapsed_s=elapsed_s,
            decode_s=decode_s,
            decode_tps=max(completion_tokens - 1, 0) / decode_s,
            output_tps=completion_tokens / max(elapsed_s, 1e-9),
            emission_events=emission_events,
            finish_reason=finish_reason,
            response_model=response_model,
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=[tool_calls[index] for index in sorted(tool_calls)],
            cached_prompt_tokens=_cached_prompt_tokens(usage),
        )
        self._successful_response_count += 1
        return result

    def run_quality_turn(self, *, prompt: str) -> RequestResult:
        if self._kind() != "quality" or not _validate_string(prompt):
            raise SM121AgentAdmissionRequestError("body")
        return self._send(
            messages=[{"role": "user", "content": prompt}],
            request_id=secrets.token_hex(16),
        )

    def run_agentic(self) -> AgenticRunResult:
        if self._kind() != "agentic":
            raise SM121AgentAdmissionRequestError("body")
        return agentic_tools.run_agentic_scenario(
            scenario_id=self._case_id,
            variant=self._variant,
            request_function=self._agentic_turn,
            request_kwargs={},
            request_id_prefix=secrets.token_hex(16),
            max_turns=6,
            max_output_tokens=_CASE_OUTPUT_TOKENS[self._case_id],
            temperature=0.0,
            extra_body={"chat_template_kwargs": dict(_LOW_THINKING)},
        )

    def _agentic_turn(self, **arguments: object) -> RequestResult:
        if self._kind() != "agentic" or set(arguments) != {
            "prompt",
            "max_tokens",
            "temperature",
            "request_id",
            "extra_body",
        }:
            raise SM121AgentAdmissionRequestError("body")
        scenario = agentic_tools._scenario(self._case_id, self._variant)
        if (
            arguments.get("prompt") != scenario.prompt
            or arguments.get("max_tokens") != _CASE_OUTPUT_TOKENS[self._case_id]
            or not _is_zero_temperature(arguments.get("temperature"))
            or not _validate_string(arguments.get("request_id"))
        ):
            raise SM121AgentAdmissionRequestError("body")
        extra_body = arguments.get("extra_body")
        if type(extra_body) is not dict or set(extra_body) != {
            "chat_template_kwargs",
            "messages",
            "tools",
            "tool_choice",
        }:
            raise SM121AgentAdmissionRequestError("body")
        if (
            not _is_low_thinking(extra_body.get("chat_template_kwargs"))
            or extra_body.get("tool_choice") != "auto"
            or extra_body.get("tools") != _expected_tools(self._case_id, self._variant)
            or type(extra_body.get("messages")) is not list
        ):
            raise SM121AgentAdmissionRequestError("body")
        return self._send(
            messages=extra_body["messages"],
            request_id=arguments["request_id"],
        )

    def run_long_context_turn(self) -> RequestResult:
        if self._kind() != "long_context" or self._outbound_body_count != 0:
            raise SM121AgentAdmissionRequestError("body")
        result = self._send(
            messages=[{"role": "user", "content": _long_context_prompt()}],
            request_id=secrets.token_hex(16),
        )
        self._long_context_prompt_tokens = result.prompt_tokens
        self._long_context_cached_prompt_tokens = result.cached_prompt_tokens
        self._long_context_response_semantics_verified = (
            result.finish_reason == "stop"
            and not result.tool_calls
            and result.content == _LONG_CONTEXT_EXPECTED_CONTENT
        )
        return result

    def long_context_receipt(self) -> dict[str, bool]:
        """Project one successful first-turn long request to scalar gates.

        Fresh-lifetime ownership and metric/guardrail observations remain the
        dedicated controller's responsibility. This client binds the rendered
        60K body, its returned prompt-token lower bound, its exact no-tool
        final-answer result, and its response cache counter without retaining
        prompt or response text in the receipt.
        """

        if self._kind() != "long_context":
            raise SM121AgentAdmissionRequestError("body")
        tokens = self._long_context_prompt_tokens
        cached = self._long_context_cached_prompt_tokens
        input_tokenization_verified = (
            type(tokens) is int
            and tokens >= SM121_AGENT_ADMISSION_LONG_CONTEXT_MIN_PROMPT_TOKENS
        )
        return {
            "input_tokenization_verified": input_tokenization_verified,
            "context_fit": (
                input_tokenization_verified
                and tokens + _CASE_OUTPUT_TOKENS[self._case_id]
                <= SM121_STORAGE_CONTEXT_LENGTH
            ),
            "zero_response_cache_hits": type(cached) is int and cached == 0,
            "response_semantics_verified": (
                self._long_context_response_semantics_verified
            ),
            "first_turn_only": (
                self._outbound_body_count == 1
                and self._successful_response_count == 1
            ),
        }


def create_sm121_agent_admission_client(
    *, server: object, model: object, case_id: str, variant: int = 0
) -> _SM121AgentAdmissionClient:
    """Bind one C1 client to the frozen server/model/case contract.

    The caller cannot provide a transport, endpoint, authorization header,
    retry policy, request body, or payload observer.  The returned client is
    useful only to a future in-repository controller; no CLI reaches it.
    """

    return _SM121AgentAdmissionClient(
        server=server,
        model=model,
        case_id=case_id,
        variant=variant,
    )


def validate_c1_payload_diagnostics(value: object) -> dict[str, int | bool]:
    """Validate the scalar-only diagnostic shape before a controller journals it."""

    if type(value) is not dict or frozenset(value) != _PUBLIC_DIAGNOSTIC_FIELDS:
        raise SM121AgentAdmissionRequestError("body")
    result: dict[str, int | bool] = {}
    for field in _PUBLIC_DIAGNOSTIC_FIELDS:
        item = value[field]
        if field == "payload_contract_verified":
            if type(item) is not bool:
                raise SM121AgentAdmissionRequestError("body")
        elif type(item) is not int or item < 0:
            raise SM121AgentAdmissionRequestError("body")
        result[field] = item
    if result["transport_retry_count"] != 0 or (
        result["transport_attempt_count"] != result["outbound_body_count"]
    ) or (
        result["validated_low_thinking_body_count"]
        > result["outbound_body_count"]
    ) or (
        result["validated_tool_body_count"] > result["outbound_body_count"]
    ) or (
        result["validated_cache_zero_body_count"]
        > result["validated_tool_body_count"]
    ) or (
        result["payload_contract_verified"] is True
        and (
            result["outbound_body_count"] <= 0
            or result["validated_low_thinking_body_count"]
            != result["outbound_body_count"]
        )
    ):
        raise SM121AgentAdmissionRequestError("body")
    return result
