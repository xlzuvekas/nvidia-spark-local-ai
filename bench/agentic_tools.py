"""Deterministic, scalar-safe multi-turn tool-use evaluation.

The synthetic prompts, tool arguments, model messages, and tool results in this
module are deliberately ephemeral.  :class:`AgenticRunResult` is the only
persistable output and contains aggregate scalars rather than a trajectory.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import re
import time
from typing import Any, Callable, Mapping, Protocol


AGENTIC_SCENARIO_IDS = (
    "agentic-select-and-call",
    "agentic-no-tool",
    "agentic-two-hop",
    "agentic-tool-error-recovery",
)
AGENTIC_VARIANT_COUNT = 3
DEFAULT_MAX_TURNS = 6
MIN_OUTPUT_TOKENS = 2_048
MAX_TURNS = 8
MAX_TOOL_CALLS = 16
MAX_ARGUMENT_BYTES = 4_096

_FAILURE_CODES = frozenset(
    {
        "final_answer",
        "malformed_tool_call",
        "missing_final",
        "output_limit",
        "tool_call_limit",
        "tool_sequence",
        "turn_limit",
        "unknown_tool",
    }
)
_CONTROLLED_REQUEST_KEYS = frozenset(
    {"extra_body", "max_tokens", "prompt", "request_id", "temperature"}
)
_CONTROLLED_BODY_KEYS = frozenset({"messages", "tool_choice", "tools"})
_FINAL_ANSWER = re.compile(r"\s*FINAL\s*:\s*(?P<answer>[^\r\n]+?)\s*", re.IGNORECASE)

_SYSTEM_PROMPT = (
    "Complete the deterministic tool-use task. Choose tools only when they are "
    "needed, treat tool results as authoritative, and never guess a lookup "
    "result. If a tool reports transient_error, retry that same tool once with "
    "the same arguments. When the task is complete, reply with exactly "
    "FINAL: <answer> and no other text."
)

_SELECT_VARIANTS = ((6, 7), (8, 9), (12, 11))
_NO_TOOL_VARIANTS = ("ORCHID-27", "EMBER-41", "QUARTZ-63")
_TWO_HOP_VARIANTS = (("cedar", 3), ("amber", 4), ("cobalt", 2))
_LOOKUP_NUMBERS = {"cedar": 17, "amber": 23, "cobalt": 31}
_RECOVERY_VARIANTS = ("north", "east", "west")
_UNSTABLE_NUMBERS = {"north": 44, "east": 58, "west": 73}

_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "multiply": {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Return the exact product of two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    },
    "add_integers": {
        "type": "function",
        "function": {
            "name": "add_integers",
            "description": "Return the exact sum of two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    },
    "lookup_number": {
        "type": "function",
        "function": {
            "name": "lookup_number",
            "description": "Look up the integer stored under a registry label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": list(_LOOKUP_NUMBERS),
                    }
                },
                "required": ["label"],
                "additionalProperties": False,
            },
        },
    },
    "unstable_lookup": {
        "type": "function",
        "function": {
            "name": "unstable_lookup",
            "description": (
                "Look up the integer for a compass key. A transient failure may "
                "require one retry."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "enum": list(_UNSTABLE_NUMBERS),
                    }
                },
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    },
}


class AgenticScenarioError(RuntimeError):
    """A privacy-safe episode failure whose message never contains model data."""


class ChatResult(Protocol):
    """The subset of ``RequestResult`` consumed by the agentic loop."""

    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    elapsed_s: float
    emission_events: int
    finish_reason: str | None
    content: str
    tool_calls: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class AgenticRunResult:
    """Aggregate result safe to journal or pass to the evidence exporter."""

    schema_version: int
    scenario_id: str
    variant: int
    passed: bool
    failure_code: str | None
    max_turns: int
    max_output_tokens: int
    turns_used: int
    expected_tool_calls: int
    tool_calls_requested: int
    tool_calls_executed: int
    tool_calls_succeeded: int
    tool_errors: int
    malformed_tool_calls: int
    unknown_tool_calls: int
    final_answer_emitted: bool
    final_answer_correct: bool
    tool_sequence_correct: bool
    recovery_required: bool
    recovery_succeeded: bool
    turn_limit_reached: bool
    prompt_tokens: int
    completion_tokens: int
    emission_events: int
    first_turn_ttft_s: float | None
    request_elapsed_s: float
    wall_s: float
    length_terminated_turns: int
    elapsed_s: float
    ttft_s: float | None
    finish_reason: str | None
    output_tps: float
    decode_s: None
    decode_tps: None
    decode_metric_source: None

    def to_dict(self) -> dict[str, Any]:
        """Return the fixed scalar schema; no messages or payloads are included."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Scenario:
    id: str
    variant: int
    prompt: str
    expected_answer: str
    expected_calls: tuple[str, ...]
    tool_names: tuple[str, ...]
    oracle_values: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _ToolAttempt:
    name: str
    arguments: dict[str, Any]
    outcome: str
    value: int | None
    turn: int


@dataclass(slots=True)
class _ToolState:
    unstable_calls: int = 0


@dataclass(frozen=True, slots=True)
class _ToolOutcome:
    code: str
    content: dict[str, Any]
    value: int | None = None


def is_agentic_scenario(case_id: str) -> bool:
    """Return whether ``case_id`` is one of the fixed synthetic scenarios."""

    return case_id in AGENTIC_SCENARIO_IDS


def estimate_agentic_context_tokens(
    *, max_turns: int, max_output_tokens: int
) -> int:
    """Conservatively budget accumulated outputs plus prompts and tool schemas."""

    _validate_limits(max_turns=max_turns, max_output_tokens=max_output_tokens)
    return max_turns * max_output_tokens + 2_048


def _validate_limits(*, max_turns: int, max_output_tokens: int) -> None:
    if isinstance(max_turns, bool) or not isinstance(max_turns, int):
        raise ValueError("max_turns must be an integer")
    if not 1 <= max_turns <= MAX_TURNS:
        raise ValueError(f"max_turns must be between 1 and {MAX_TURNS}")
    if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int):
        raise ValueError("max_output_tokens must be an integer")
    if max_output_tokens < MIN_OUTPUT_TOKENS:
        raise ValueError(
            f"max_output_tokens must be at least {MIN_OUTPUT_TOKENS} per turn"
        )


def _scenario(scenario_id: str, variant: int) -> _Scenario:
    if scenario_id not in AGENTIC_SCENARIO_IDS:
        raise ValueError(f"unknown agentic scenario {scenario_id!r}")
    if isinstance(variant, bool) or not isinstance(variant, int):
        raise ValueError("variant must be an integer")
    if not 0 <= variant < AGENTIC_VARIANT_COUNT:
        raise ValueError(
            f"variant must be between 0 and {AGENTIC_VARIANT_COUNT - 1}"
        )

    if scenario_id == "agentic-select-and-call":
        left, right = _SELECT_VARIANTS[variant]
        return _Scenario(
            id=scenario_id,
            variant=variant,
            prompt=(
                f"Use the appropriate available tool to multiply {left} by "
                f"{right}, then report the exact product."
            ),
            expected_answer=str(left * right),
            expected_calls=("multiply",),
            tool_names=("multiply", "add_integers", "lookup_number"),
            oracle_values=(left, right),
        )
    if scenario_id == "agentic-no-tool":
        token = _NO_TOOL_VARIANTS[variant]
        return _Scenario(
            id=scenario_id,
            variant=variant,
            prompt=(
                "No calculation or lookup is required. Do not call a tool. "
                f"Report the literal token {token}."
            ),
            expected_answer=token,
            expected_calls=(),
            tool_names=("lookup_number", "multiply", "add_integers"),
            oracle_values=(),
        )
    if scenario_id == "agentic-two-hop":
        label, multiplier = _TWO_HOP_VARIANTS[variant]
        expected = _LOOKUP_NUMBERS[label] * multiplier
        return _Scenario(
            id=scenario_id,
            variant=variant,
            prompt=(
                f"Look up the unknown integer stored under registry label {label}. "
                f"Then use a tool to multiply the returned integer by {multiplier}. "
                "Report the product."
            ),
            expected_answer=str(expected),
            expected_calls=("lookup_number", "multiply"),
            tool_names=("lookup_number", "add_integers", "multiply"),
            oracle_values=(label, multiplier),
        )

    key = _RECOVERY_VARIANTS[variant]
    return _Scenario(
        id=scenario_id,
        variant=variant,
        prompt=(
            f"Use unstable_lookup to retrieve the integer for compass key {key}. "
            "If the tool reports transient_error, retry once with the same key. "
            "Report the retrieved integer."
        ),
        expected_answer=str(_UNSTABLE_NUMBERS[key]),
        expected_calls=("unstable_lookup", "unstable_lookup"),
        tool_names=("add_integers", "unstable_lookup", "multiply"),
        oracle_values=(key,),
    )


def _rotated_tool_schemas(scenario: _Scenario) -> list[dict[str, Any]]:
    names = list(scenario.tool_names)
    offset = scenario.variant % len(names)
    names = names[offset:] + names[:offset]
    return [deepcopy(_TOOL_SCHEMAS[name]) for name in names]


def _safe_call_id(*, turn: int, index: int) -> str:
    """Replace model-provided identifiers with bounded episode-local values."""

    return f"agentic_t{turn}_c{index}"


def _function_parts(call: Any) -> tuple[str | None, Any]:
    if not isinstance(call, Mapping):
        return None, None
    function = call.get("function")
    if not isinstance(function, Mapping):
        return None, None
    name = function.get("name")
    if not isinstance(name, str) or not name or len(name) > 64:
        return None, function.get("arguments")
    return name, function.get("arguments")


def _parse_arguments(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, Mapping):
        try:
            encoded = json.dumps(
                dict(raw), allow_nan=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            if len(encoded) > MAX_ARGUMENT_BYTES:
                return None
            value: Any = json.loads(encoded)
        except (TypeError, ValueError, RecursionError, UnicodeError):
            return None
    elif isinstance(raw, str):
        try:
            encoded = raw.encode("utf-8")
        except UnicodeError:
            return None
        if len(encoded) > MAX_ARGUMENT_BYTES:
            return None
        try:
            value = json.loads(
                raw,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, RecursionError, ValueError):
            return None
    else:
        return None
    return value if isinstance(value, dict) else None


def _integer_arguments(arguments: Mapping[str, Any]) -> bool:
    if set(arguments) != {"a", "b"}:
        return False
    values = (arguments["a"], arguments["b"])
    return all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and abs(value) <= 1_000_000
        for value in values
    )


def _arguments_valid(name: str, arguments: Mapping[str, Any]) -> bool:
    if name in {"multiply", "add_integers"}:
        return _integer_arguments(arguments)
    if name == "lookup_number":
        return (
            set(arguments) == {"label"}
            and isinstance(arguments["label"], str)
            and arguments["label"] in _LOOKUP_NUMBERS
        )
    if name == "unstable_lookup":
        return (
            set(arguments) == {"key"}
            and isinstance(arguments["key"], str)
            and arguments["key"] in _UNSTABLE_NUMBERS
        )
    return False


def _execute_tool(
    name: str, arguments: Mapping[str, Any], state: _ToolState
) -> _ToolOutcome:
    if name == "multiply":
        value = int(arguments["a"]) * int(arguments["b"])
        return _ToolOutcome("ok", {"ok": True, "value": value}, value)
    if name == "add_integers":
        value = int(arguments["a"]) + int(arguments["b"])
        return _ToolOutcome("ok", {"ok": True, "value": value}, value)
    if name == "lookup_number":
        value = _LOOKUP_NUMBERS[str(arguments["label"])]
        return _ToolOutcome("ok", {"ok": True, "value": value}, value)
    if name == "unstable_lookup":
        state.unstable_calls += 1
        if state.unstable_calls == 1:
            return _ToolOutcome(
                "transient_error",
                {"ok": False, "error": "transient_error", "retryable": True},
            )
        value = _UNSTABLE_NUMBERS[str(arguments["key"])]
        return _ToolOutcome("ok", {"ok": True, "value": value}, value)
    raise AssertionError("tool execution escaped the allowlist")


def _assistant_tool_call(
    *, call_id: str, name: str | None, arguments: Mapping[str, Any] | None
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name or "invalid_tool",
            "arguments": json.dumps(
                dict(arguments) if arguments is not None else {},
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    }


def _answer_matches(content: Any, expected: str) -> tuple[bool, bool]:
    if not isinstance(content, str):
        return False, False
    match = _FINAL_ANSWER.fullmatch(content)
    if match is None:
        return False, False
    return True, match.group("answer").strip() == expected


def _same_product_arguments(
    arguments: Mapping[str, Any], left: int, right: int
) -> bool:
    return sorted((arguments.get("a"), arguments.get("b"))) == sorted((left, right))


def _tool_sequence_correct(
    scenario: _Scenario,
    attempts: list[_ToolAttempt],
    *, malformed_tool_calls: int,
    unknown_tool_calls: int,
    call_limit_exceeded: bool,
) -> bool:
    if malformed_tool_calls or unknown_tool_calls or call_limit_exceeded:
        return False
    if len(attempts) != len(scenario.expected_calls):
        return False
    if tuple(attempt.name for attempt in attempts) != scenario.expected_calls:
        return False

    if scenario.id == "agentic-no-tool":
        return True
    if scenario.id == "agentic-select-and-call":
        left, right = scenario.oracle_values
        attempt = attempts[0]
        return (
            attempt.outcome == "ok"
            and _same_product_arguments(attempt.arguments, left, right)
            and attempt.value == left * right
        )
    if scenario.id == "agentic-two-hop":
        label, multiplier = scenario.oracle_values
        lookup, product = attempts
        lookup_value = _LOOKUP_NUMBERS[label]
        return (
            lookup.arguments == {"label": label}
            and lookup.outcome == "ok"
            and lookup.value == lookup_value
            and product.turn > lookup.turn
            and product.outcome == "ok"
            and _same_product_arguments(product.arguments, lookup_value, multiplier)
            and product.value == lookup_value * multiplier
        )

    (key,) = scenario.oracle_values
    first, retry = attempts
    return (
        first.arguments == {"key": key}
        and retry.arguments == {"key": key}
        and retry.turn > first.turn
        and first.outcome == "transient_error"
        and retry.outcome == "ok"
        and retry.value == _UNSTABLE_NUMBERS[key]
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a nonnegative finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return number


def _safe_finish_reason(value: Any) -> str | None:
    if value is None:
        return None
    if value in {"stop", "length", "tool_calls", "content_filter"}:
        return str(value)
    return "other"


def _failure_code(
    *,
    passed: bool,
    call_limit_exceeded: bool,
    malformed_tool_calls: int,
    unknown_tool_calls: int,
    turn_limit_reached: bool,
    length_terminated_turns: int,
    tool_sequence_correct: bool,
    final_answer_emitted: bool,
) -> str | None:
    if passed:
        return None
    if call_limit_exceeded:
        return "tool_call_limit"
    if length_terminated_turns:
        return "output_limit"
    if malformed_tool_calls:
        return "malformed_tool_call"
    if unknown_tool_calls:
        return "unknown_tool"
    if turn_limit_reached:
        return "turn_limit"
    if not tool_sequence_correct:
        return "tool_sequence"
    if not final_answer_emitted:
        return "missing_final"
    return "final_answer"


def run_agentic_scenario(
    *,
    scenario_id: str,
    variant: int,
    request_function: Callable[..., ChatResult],
    request_kwargs: Mapping[str, Any],
    request_id_prefix: str,
    max_turns: int,
    max_output_tokens: int,
    temperature: float = 0.0,
    extra_body: Mapping[str, Any] | None = None,
) -> AgenticRunResult:
    """Run one bounded tool loop and return aggregate, scalar-only evidence.

    ``request_kwargs`` contains transport-specific values such as ``base_url``,
    ``model``, authorization, or an Ollama context size. The harness exclusively
    controls prompt/messages, output budget, temperature, request IDs, tool
    schemas, and ``tool_choice``.
    """

    _validate_limits(max_turns=max_turns, max_output_tokens=max_output_tokens)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError("temperature must be numeric")
    if float(temperature) != 0.0:
        raise ValueError("agentic scenarios require temperature 0")
    if not isinstance(request_id_prefix, str) or not request_id_prefix:
        raise ValueError("request_id_prefix must be a non-empty string")
    request_overlap = _CONTROLLED_REQUEST_KEYS & set(request_kwargs)
    if request_overlap:
        raise ValueError(
            "request_kwargs cannot override controlled fields: "
            + ", ".join(sorted(request_overlap))
        )
    additions = dict(extra_body or {})
    body_overlap = _CONTROLLED_BODY_KEYS & set(additions)
    if body_overlap:
        raise ValueError(
            "extra_body cannot override agentic fields: "
            + ", ".join(sorted(body_overlap))
        )
    unknown_additions = set(additions) - {"chat_template_kwargs"}
    if unknown_additions:
        raise ValueError("extra_body contains unsupported agentic settings")
    if "chat_template_kwargs" in additions:
        template_options = additions["chat_template_kwargs"]
        if (
            not isinstance(template_options, Mapping)
            or set(template_options) != {"enable_thinking"}
            or not isinstance(template_options["enable_thinking"], bool)
        ):
            raise ValueError(
                "agentic chat_template_kwargs must contain one boolean enable_thinking"
            )

    scenario = _scenario(scenario_id, variant)
    tools = _rotated_tool_schemas(scenario)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": scenario.prompt},
    ]
    tool_state = _ToolState()
    attempts: list[_ToolAttempt] = []
    tool_calls_requested = 0
    tool_calls_executed = 0
    tool_calls_succeeded = 0
    tool_errors = 0
    malformed_tool_calls = 0
    unknown_tool_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    emission_events = 0
    request_elapsed_s = 0.0
    first_turn_ttft_s: float | None = None
    length_terminated_turns = 0
    final_answer_emitted = False
    final_answer_correct = False
    call_limit_exceeded = False
    last_turn_had_tools = False
    turns_used = 0
    finish_reason: str | None = None
    started = time.perf_counter()

    for turn in range(1, max_turns + 1):
        turns_used = turn
        request_body = dict(additions)
        request_body.update(
            {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
                "tool_choice": "auto",
            }
        )
        try:
            result = request_function(
                **dict(request_kwargs),
                prompt=scenario.prompt,
                max_tokens=max_output_tokens,
                temperature=0.0,
                request_id=f"{request_id_prefix}-t{turn}",
                extra_body=request_body,
            )
        except Exception as error:
            raise AgenticScenarioError("agentic model request failed") from error
        prompt_tokens += _nonnegative_int(
            getattr(result, "prompt_tokens", None), name="prompt_tokens"
        )
        completion_tokens += _nonnegative_int(
            getattr(result, "completion_tokens", None), name="completion_tokens"
        )
        emission_events += _nonnegative_int(
            getattr(result, "emission_events", None), name="emission_events"
        )
        request_elapsed_s += _nonnegative_float(
            getattr(result, "elapsed_s", None), name="elapsed_s"
        )
        if first_turn_ttft_s is None:
            first_turn_ttft_s = _nonnegative_float(
                getattr(result, "ttft_s", None), name="ttft_s"
            )
        if getattr(result, "finish_reason", None) == "length":
            length_terminated_turns += 1
        finish_reason = _safe_finish_reason(getattr(result, "finish_reason", None))

        raw_calls = getattr(result, "tool_calls", None)
        if not isinstance(raw_calls, list):
            raise ValueError("tool_calls must be a list")
        calls = list(raw_calls)
        if not calls:
            if finish_reason not in {"stop", "length"}:
                raise AgenticScenarioError(
                    "agentic final turn finish reason invalid"
                )
            last_turn_had_tools = False
            final_answer_emitted, final_answer_correct = _answer_matches(
                getattr(result, "content", ""), scenario.expected_answer
            )
            break

        last_turn_had_tools = True
        if finish_reason != "tool_calls":
            raise AgenticScenarioError("agentic tool turn finish reason invalid")
        tool_calls_requested += len(calls)
        if tool_calls_requested > MAX_TOOL_CALLS:
            call_limit_exceeded = True
            break

        assistant_calls: list[dict[str, Any]] = []
        tool_messages: list[dict[str, Any]] = []
        allowed_names = set(scenario.tool_names)
        for index, call in enumerate(calls):
            call_id = _safe_call_id(turn=turn, index=index)
            name, raw_arguments = _function_parts(call)
            arguments = _parse_arguments(raw_arguments)
            assistant_calls.append(
                _assistant_tool_call(
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                )
            )
            outcome_content: dict[str, Any]
            safe_name = name or "invalid_tool"
            if name is None or arguments is None:
                malformed_tool_calls += 1
                outcome_content = {"ok": False, "error": "invalid_arguments"}
            elif name not in allowed_names:
                unknown_tool_calls += 1
                outcome_content = {"ok": False, "error": "unknown_tool"}
            elif not _arguments_valid(name, arguments):
                malformed_tool_calls += 1
                outcome_content = {"ok": False, "error": "invalid_arguments"}
            else:
                tool_calls_executed += 1
                outcome = _execute_tool(name, arguments, tool_state)
                if outcome.code == "ok":
                    tool_calls_succeeded += 1
                else:
                    tool_errors += 1
                attempts.append(
                    _ToolAttempt(
                        name=name,
                        arguments=dict(arguments),
                        outcome=outcome.code,
                        value=outcome.value,
                        turn=turn,
                    )
                )
                outcome_content = outcome.content
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": safe_name,
                    "content": json.dumps(
                        outcome_content, separators=(",", ":"), sort_keys=True
                    ),
                }
            )
        messages.append(
            {
                "role": "assistant",
                "content": getattr(result, "content", "") or None,
                "tool_calls": assistant_calls,
            }
        )
        messages.extend(tool_messages)

    wall_s = time.perf_counter() - started
    turn_limit_reached = bool(
        not call_limit_exceeded
        and turns_used == max_turns
        and last_turn_had_tools
        and not final_answer_emitted
    )
    if call_limit_exceeded:
        finish_reason = "tool_call_limit"
    elif turn_limit_reached:
        finish_reason = "turn_limit"
    sequence_correct = _tool_sequence_correct(
        scenario,
        attempts,
        malformed_tool_calls=malformed_tool_calls,
        unknown_tool_calls=unknown_tool_calls,
        call_limit_exceeded=call_limit_exceeded,
    )
    recovery_required = scenario.id == "agentic-tool-error-recovery"
    recovery_succeeded = bool(
        recovery_required
        and sequence_correct
        and tool_errors == 1
        and tool_calls_succeeded == 1
    )
    passed = bool(
        sequence_correct
        and final_answer_emitted
        and final_answer_correct
        and not turn_limit_reached
        and length_terminated_turns == 0
    )
    failure_code = _failure_code(
        passed=passed,
        call_limit_exceeded=call_limit_exceeded,
        malformed_tool_calls=malformed_tool_calls,
        unknown_tool_calls=unknown_tool_calls,
        turn_limit_reached=turn_limit_reached,
        length_terminated_turns=length_terminated_turns,
        tool_sequence_correct=sequence_correct,
        final_answer_emitted=final_answer_emitted,
    )
    if failure_code is not None and failure_code not in _FAILURE_CODES:
        raise AssertionError("agentic result escaped the failure-code allowlist")
    return AgenticRunResult(
        schema_version=1,
        scenario_id=scenario.id,
        variant=scenario.variant,
        passed=passed,
        failure_code=failure_code,
        max_turns=max_turns,
        max_output_tokens=max_output_tokens,
        turns_used=turns_used,
        expected_tool_calls=len(scenario.expected_calls),
        tool_calls_requested=tool_calls_requested,
        tool_calls_executed=tool_calls_executed,
        tool_calls_succeeded=tool_calls_succeeded,
        tool_errors=tool_errors,
        malformed_tool_calls=malformed_tool_calls,
        unknown_tool_calls=unknown_tool_calls,
        final_answer_emitted=final_answer_emitted,
        final_answer_correct=final_answer_correct,
        tool_sequence_correct=sequence_correct,
        recovery_required=recovery_required,
        recovery_succeeded=recovery_succeeded,
        turn_limit_reached=turn_limit_reached,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        emission_events=emission_events,
        first_turn_ttft_s=first_turn_ttft_s,
        request_elapsed_s=request_elapsed_s,
        wall_s=wall_s,
        length_terminated_turns=length_terminated_turns,
        elapsed_s=wall_s,
        ttft_s=first_turn_ttft_s,
        finish_reason=finish_reason,
        output_tps=completion_tokens / max(wall_s, 1e-9),
        decode_s=None,
        decode_tps=None,
        decode_metric_source=None,
    )
