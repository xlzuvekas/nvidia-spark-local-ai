"""Scalar-only speculative-decoding audits for managed SGLang servers.

The pinned SGLang build returns speculative counters automatically in the
final native ``/generate`` response.  It does not accept a
``return_meta_info`` request field.  This module deliberately projects only
the cumulative per-request counters and discards generated text, token IDs,
request IDs, and every unrelated metadata field.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import math
import re
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


SGLANG_SPECULATIVE_SOURCE = "sglang_native_generate_per_request_counters"
SGLANG_SPECULATIVE_SCOPE = "explicit_sglang_native_audit_requests_only"
SGLANG_SPECULATIVE_METHOD = "NEXTN"
SGLANG_AUDIT_PROMPT = (
    "Write an unbroken numbered list of distinct two-word phrases. "
    "Continue until the output limit; do not conclude or summarize."
)

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_AUDIT_OUTPUT_TOKENS = 4096
_MAX_DRAFT_DEPTH = 64
_REPORTED_RATE_REL_TOLERANCE = 1e-6
_REPORTED_RATE_ABS_TOLERANCE = 1e-9
_INTEGER = re.compile(r"[0-9]+")
_CANONICAL_COUNTS = (
    "spec_num_correct_drafts",
    "spec_num_proposed_drafts",
    "spec_verify_ct",
)
_ALIASES = {
    "spec_accepted_drafts": "spec_num_correct_drafts",
    "spec_proposed_drafts": "spec_num_proposed_drafts",
}
_SNAPSHOT_KEYS = {
    "accepted_tokens_per_position",
    "configured_max_draft_tokens",
    "draft_acceptance_rate",
    "mean_accepted_length",
    "method",
    "num_accepted_tokens",
    "num_draft_tokens",
    "num_drafts",
    "requested",
    "scope",
    "source",
}


class SGLangSpeculativeAuditError(RuntimeError):
    """A bounded, public-safe failure from the native acceptance audit."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    _NoRedirectHandler(),
)


def _urlopen(request: urllib.request.Request, *, timeout: float) -> Any:
    return _OPENER.open(request, timeout=timeout)


def _positive_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise SGLangSpeculativeAuditError(f"{name} must be a positive integer")
    return value


def _count(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise SGLangSpeculativeAuditError(
            f"SGLang speculative field {name} must be a non-negative integer"
        )
    return value


def _finite_number(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SGLangSpeculativeAuditError(
            f"SGLang speculative field {name} must be finite"
        )
    return float(value)


def _canonical_root(base_url: str) -> str:
    if not isinstance(base_url, str):
        raise SGLangSpeculativeAuditError(
            "SGLang audit endpoint must be canonical loopback HTTP"
        )
    candidate = base_url.rstrip("/")
    try:
        parsed = urllib.parse.urlsplit(candidate)
        port = parsed.port
    except ValueError as error:
        raise SGLangSpeculativeAuditError(
            "SGLang audit endpoint must be canonical loopback HTTP"
        ) from None
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port is None
        or not 1 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/v1"
        or parsed.query
        or parsed.fragment
        or candidate != f"http://127.0.0.1:{port}/v1"
    ):
        raise SGLangSpeculativeAuditError(
            "SGLang audit endpoint must be canonical loopback HTTP"
        )
    return f"http://127.0.0.1:{port}"


def _headers(authorization: str) -> dict[str, str]:
    if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
        raise SGLangSpeculativeAuditError(
            "SGLang audit authorization is unavailable or malformed"
        )
    token = authorization.removeprefix("Bearer ")
    if (
        not token
        or any(character.isspace() for character in token)
        or any(ord(character) < 33 or ord(character) == 127 for character in token)
    ):
        raise SGLangSpeculativeAuditError(
            "SGLang audit authorization is unavailable or malformed"
        )
    return {
        "Authorization": authorization,
        "Content-Type": "application/json",
    }


def _post_json(
    url: str,
    *,
    body: Mapping[str, Any],
    authorization: str,
    timeout_s: float,
) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            dict(body),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError):
        raise SGLangSpeculativeAuditError(
            "SGLang native audit request body is invalid"
        ) from None
    request = urllib.request.Request(
        url,
        data=encoded,
        headers=_headers(authorization),
        method="POST",
    )
    try:
        with _urlopen(request, timeout=timeout_s) as response:
            status = getattr(response, "status", 200)
            if type(status) is not int or status != 200:
                raise SGLangSpeculativeAuditError(
                    "SGLang native audit returned a non-success status"
                )
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
    except SGLangSpeculativeAuditError:
        raise
    except urllib.error.HTTPError as error:
        status = error.code if type(error.code) is int else 0
        error.close()
        raise SGLangSpeculativeAuditError(
            f"SGLang native audit failed with HTTP status {status}"
        ) from None
    except (OSError, TimeoutError, urllib.error.URLError):
        raise SGLangSpeculativeAuditError(
            "SGLang native audit transport failed"
        ) from None
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise SGLangSpeculativeAuditError(
            "SGLang native audit response exceeded the safe size limit"
        )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SGLangSpeculativeAuditError(
            "SGLang native audit response was not valid JSON"
        ) from None
    if not isinstance(payload, dict):
        raise SGLangSpeculativeAuditError(
            "SGLang native audit response must be an object"
        )
    return payload


def _option_value(arguments: Sequence[Any], option: str) -> str | None:
    values: list[str] = []
    normalized = [str(argument) for argument in arguments]
    for index, argument in enumerate(normalized):
        if argument.startswith(option + "="):
            values.append(argument.split("=", 1)[1])
        elif argument == option:
            if index + 1 >= len(normalized) or normalized[index + 1].startswith("--"):
                raise SGLangSpeculativeAuditError(
                    f"SGLang NEXTN option {option} is missing its value"
                )
            values.append(normalized[index + 1])
    if len(values) > 1:
        raise SGLangSpeculativeAuditError(
            f"SGLang NEXTN option {option} must be configured exactly once"
        )
    return values[0] if values else None


def sglang_nextn_depth(arguments: Iterable[Any]) -> int | None:
    """Return the exact proposals-per-verify depth for one NEXTN profile."""

    values = tuple(arguments)
    algorithm = _option_value(values, "--speculative-algorithm")
    if algorithm is None or algorithm.upper() != SGLANG_SPECULATIVE_METHOD:
        return None
    configured = _option_value(values, "--speculative-num-draft-tokens")
    if configured is None or _INTEGER.fullmatch(configured) is None:
        raise SGLangSpeculativeAuditError(
            "SGLang NEXTN draft-token geometry is missing or malformed"
        )
    total_tokens = int(configured)
    depth = total_tokens - 1
    if not 1 <= depth <= _MAX_DRAFT_DEPTH:
        raise SGLangSpeculativeAuditError(
            "SGLang NEXTN draft-token geometry is outside the audit bounds"
        )
    return depth


def parse_sglang_speculative_response(
    payload: Any, *, expected_depth: int
) -> dict[str, Any]:
    """Project and validate exact cumulative counters from one final response."""

    depth = _positive_integer(expected_depth, name="SGLang NEXTN depth")
    if depth > _MAX_DRAFT_DEPTH:
        raise SGLangSpeculativeAuditError(
            "SGLang NEXTN depth is outside the audit bounds"
        )
    if not isinstance(payload, dict):
        raise SGLangSpeculativeAuditError(
            "SGLang native audit response must be an object"
        )
    meta_info = payload.get("meta_info")
    if not isinstance(meta_info, dict):
        raise SGLangSpeculativeAuditError(
            "SGLang native audit response has no metadata object"
        )
    missing = [name for name in _CANONICAL_COUNTS if name not in meta_info]
    if missing:
        raise SGLangSpeculativeAuditError(
            "SGLang NEXTN response is missing cumulative speculative counters"
        )

    accepted = _count(
        meta_info["spec_num_correct_drafts"],
        name="spec_num_correct_drafts",
    )
    proposed = _count(
        meta_info["spec_num_proposed_drafts"],
        name="spec_num_proposed_drafts",
    )
    verify_count = _count(meta_info["spec_verify_ct"], name="spec_verify_ct")
    if verify_count == 0 or proposed == 0:
        raise SGLangSpeculativeAuditError(
            "SGLang NEXTN response contains no speculative verification activity"
        )
    if proposed != verify_count * depth:
        raise SGLangSpeculativeAuditError(
            "SGLang NEXTN proposed-token geometry does not match the configured depth"
        )
    if accepted > proposed:
        raise SGLangSpeculativeAuditError(
            "SGLang NEXTN accepted-token count exceeds proposed tokens"
        )

    for alias, canonical in _ALIASES.items():
        if alias in meta_info:
            alias_value = _count(meta_info[alias], name=alias)
            if alias_value != meta_info[canonical]:
                raise SGLangSpeculativeAuditError(
                    "SGLang NEXTN compatibility counter disagrees with its canonical field"
                )

    raw_histogram = meta_info.get("spec_correct_drafts_histogram")
    if (
        not isinstance(raw_histogram, list)
        or not raw_histogram
        or len(raw_histogram) > depth + 1
    ):
        raise SGLangSpeculativeAuditError(
            "SGLang NEXTN acceptance histogram is missing or outside the configured depth"
        )
    histogram = [
        _count(value, name="spec_correct_drafts_histogram")
        for value in raw_histogram
    ]
    if "spec_accept_histogram" in meta_info:
        alias_histogram = meta_info["spec_accept_histogram"]
        if alias_histogram != raw_histogram:
            raise SGLangSpeculativeAuditError(
                "SGLang NEXTN compatibility histogram disagrees with its canonical field"
            )
    if sum(histogram) != verify_count:
        raise SGLangSpeculativeAuditError(
            "SGLang NEXTN histogram step count does not match verification count"
        )
    if sum(index * count for index, count in enumerate(histogram)) != accepted:
        raise SGLangSpeculativeAuditError(
            "SGLang NEXTN histogram accepted-token sum does not match its counter"
        )

    calculated_rate = accepted / proposed
    if "spec_accept_rate" in meta_info:
        reported_rate = _finite_number(
            meta_info["spec_accept_rate"], name="spec_accept_rate"
        )
        if not math.isclose(
            reported_rate,
            calculated_rate,
            rel_tol=_REPORTED_RATE_REL_TOLERANCE,
            abs_tol=_REPORTED_RATE_ABS_TOLERANCE,
        ):
            raise SGLangSpeculativeAuditError(
                "SGLang NEXTN reported acceptance rate disagrees with its counters"
            )
    if "spec_accept_length" in meta_info:
        reported_length = _finite_number(
            meta_info["spec_accept_length"], name="spec_accept_length"
        )
        if reported_length <= 0:
            raise SGLangSpeculativeAuditError(
                "SGLang NEXTN reported acceptance length must be positive"
            )

    per_position = {
        str(position): sum(histogram[position + 1 :])
        for position in range(depth)
    }
    if sum(per_position.values()) != accepted:
        raise SGLangSpeculativeAuditError(
            "SGLang NEXTN derived position counters are inconsistent"
        )
    return {
        "source": SGLANG_SPECULATIVE_SOURCE,
        "scope": SGLANG_SPECULATIVE_SCOPE,
        "requested": True,
        "method": SGLANG_SPECULATIVE_METHOD,
        "configured_max_draft_tokens": depth,
        "num_drafts": verify_count,
        "num_draft_tokens": proposed,
        "num_accepted_tokens": accepted,
        "accepted_tokens_per_position": per_position,
        "draft_acceptance_rate": calculated_rate,
        "mean_accepted_length": 1.0 + accepted / verify_count,
    }


def _validate_snapshot(
    snapshot: Any, *, expected_depth: int
) -> tuple[int, int, int, dict[str, int]]:
    if not isinstance(snapshot, dict) or set(snapshot) != _SNAPSHOT_KEYS:
        raise SGLangSpeculativeAuditError(
            "SGLang speculative audit snapshot does not match its scalar schema"
        )
    if (
        snapshot.get("source") != SGLANG_SPECULATIVE_SOURCE
        or snapshot.get("scope") != SGLANG_SPECULATIVE_SCOPE
        or snapshot.get("requested") is not True
        or snapshot.get("method") != SGLANG_SPECULATIVE_METHOD
        or snapshot.get("configured_max_draft_tokens") != expected_depth
    ):
        raise SGLangSpeculativeAuditError(
            "SGLang speculative audit snapshot provenance or geometry changed"
        )
    drafts = _count(snapshot.get("num_drafts"), name="num_drafts")
    proposed = _count(snapshot.get("num_draft_tokens"), name="num_draft_tokens")
    accepted = _count(
        snapshot.get("num_accepted_tokens"), name="num_accepted_tokens"
    )
    if drafts == 0 or proposed != drafts * expected_depth or accepted > proposed:
        raise SGLangSpeculativeAuditError(
            "SGLang speculative audit snapshot counters are inconsistent"
        )
    raw_positions = snapshot.get("accepted_tokens_per_position")
    expected_positions = {str(position) for position in range(expected_depth)}
    if not isinstance(raw_positions, dict) or set(raw_positions) != expected_positions:
        raise SGLangSpeculativeAuditError(
            "SGLang speculative audit position counters changed"
        )
    positions = {
        key: _count(value, name="accepted_tokens_per_position")
        for key, value in raw_positions.items()
    }
    ordered = [positions[str(position)] for position in range(expected_depth)]
    if (
        any(value > drafts for value in ordered)
        or any(left < right for left, right in zip(ordered, ordered[1:]))
        or sum(ordered) != accepted
    ):
        raise SGLangSpeculativeAuditError(
            "SGLang speculative audit position counters are inconsistent"
        )
    rate = _finite_number(
        snapshot.get("draft_acceptance_rate"), name="draft_acceptance_rate"
    )
    mean_length = _finite_number(
        snapshot.get("mean_accepted_length"), name="mean_accepted_length"
    )
    if not math.isclose(rate, accepted / proposed, rel_tol=1e-9, abs_tol=1e-12):
        raise SGLangSpeculativeAuditError(
            "SGLang speculative audit acceptance rate is inconsistent"
        )
    if not math.isclose(
        mean_length,
        1.0 + accepted / drafts,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise SGLangSpeculativeAuditError(
            "SGLang speculative audit mean acceptance length is inconsistent"
        )
    return drafts, proposed, accepted, positions


def aggregate_sglang_speculative_audits(
    snapshots: Iterable[dict[str, Any]], *, expected_depth: int | None = None
) -> dict[str, Any] | None:
    """Combine disjoint explicit native audit requests without raw payloads."""

    records = list(snapshots)
    if not records:
        return None
    if expected_depth is None:
        first_depth = records[0].get("configured_max_draft_tokens")
        expected_depth = _positive_integer(
            first_depth, name="SGLang NEXTN aggregate depth"
        )
    depth = _positive_integer(expected_depth, name="SGLang NEXTN aggregate depth")
    drafts = 0
    proposed = 0
    accepted = 0
    positions = {str(position): 0 for position in range(depth)}
    for record in records:
        current_drafts, current_proposed, current_accepted, current_positions = (
            _validate_snapshot(record, expected_depth=depth)
        )
        drafts += current_drafts
        proposed += current_proposed
        accepted += current_accepted
        for position, value in current_positions.items():
            positions[position] += value
    deepest = max(
        (int(position) for position, value in positions.items() if value > 0),
        default=None,
    )
    proposal_depth: dict[str, Any] = {
        "average_draft_tokens_per_draft": proposed / drafts,
        "configured_max_draft_tokens": depth,
        "passed": True,
        "reason": None,
    }
    if deepest is not None:
        proposal_depth.update(
            {
                "deepest_accepted_position": deepest,
                "deepest_accepted_draft_depth": deepest + 1,
            }
        )
    return {
        "source": SGLANG_SPECULATIVE_SOURCE,
        "scope": SGLANG_SPECULATIVE_SCOPE,
        "requested": True,
        "method": SGLANG_SPECULATIVE_METHOD,
        "configured_max_draft_tokens": depth,
        "num_drafts": drafts,
        "num_draft_tokens": proposed,
        "num_accepted_tokens": accepted,
        "accepted_tokens_per_position": positions,
        "draft_acceptance_rate": accepted / proposed,
        "mean_accepted_length": 1.0 + accepted / drafts,
        "snapshot_count": len(records),
        "proposal_depth": proposal_depth,
    }


def request_sglang_speculative_audit(
    *,
    base_url: str,
    model: str,
    authorization: str,
    expected_depth: int,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    prompt: str = SGLANG_AUDIT_PROMPT,
    max_new_tokens: int = 256,
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    """Run one authenticated chat-templated native audit and return scalars."""

    root = _canonical_root(base_url)
    depth = _positive_integer(expected_depth, name="SGLang NEXTN depth")
    if depth > _MAX_DRAFT_DEPTH:
        raise SGLangSpeculativeAuditError(
            "SGLang NEXTN depth is outside the audit bounds"
        )
    if not isinstance(model, str) or not model or "\r" in model or "\n" in model:
        raise SGLangSpeculativeAuditError("SGLang audit model name is invalid")
    if not isinstance(prompt, str) or not prompt:
        raise SGLangSpeculativeAuditError("SGLang audit prompt is invalid")
    if (
        type(max_new_tokens) is not int
        or not 2 <= max_new_tokens <= _MAX_AUDIT_OUTPUT_TOKENS
    ):
        raise SGLangSpeculativeAuditError(
            "SGLang audit output-token bound is invalid"
        )
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or not 0 < float(timeout_s) <= 3600
    ):
        raise SGLangSpeculativeAuditError("SGLang audit timeout is invalid")
    template_kwargs = (
        {"enable_thinking": False}
        if chat_template_kwargs is None
        else dict(chat_template_kwargs)
    )
    tokenized = _post_json(
        root + "/v1/tokenize",
        body={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": template_kwargs,
        },
        authorization=authorization,
        timeout_s=float(timeout_s),
    )
    tokens = tokenized.get("tokens")
    count = tokenized.get("count")
    max_model_len = tokenized.get("max_model_len")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int or token < 0 for token in tokens)
        or type(count) is not int
        or count != len(tokens)
        or type(max_model_len) is not int
        or max_model_len <= 0
    ):
        raise SGLangSpeculativeAuditError(
            "SGLang native tokenize response does not match its exact schema"
        )
    generated = _post_json(
        root + "/generate",
        body={
            "input_ids": tokens,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
            },
            "stream": False,
            "no_logs": True,
            "log_metrics": False,
        },
        authorization=authorization,
        timeout_s=float(timeout_s),
    )
    return parse_sglang_speculative_response(
        generated,
        expected_depth=depth,
    )
