from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

from bench.content_battery import (
    MIN_OUTPUT_TOKENS,
    PROBES,
    REPETITIONS_PER_PROMPT,
    ContentBatteryError,
    RequestMetrics,
    canonical_loopback_base_url,
    main,
    read_api_key,
    run_battery,
    stream_request,
    tagged_prompt,
    verify_served_model,
    write_result,
)


class _Response:
    def __init__(self, *, lines: list[bytes] | None = None, payload: object = None):
        self._lines = lines or []
        self._payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def __iter__(self):
        return iter(self._lines)

    def read(self, *_: object) -> bytes:
        return json.dumps(self._payload).encode()


class _Opener:
    def __init__(self, *responses: _Response):
        self.responses = list(responses)
        self.requests: list[urllib.request.Request] = []

    def open(self, request: urllib.request.Request, **_: object) -> _Response:
        self.requests.append(request)
        return self.responses.pop(0)


def _metrics(
    *, prompt_tokens: int = 100, completion_tokens: int = 80
) -> RequestMetrics:
    return RequestMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_s=0.25,
        e2e_s=4.0,
        decode_s=3.75,
        decode_tps=(completion_tokens - 1) / 3.75,
        output_tps=completion_tokens / 4.0,
        emission_events=completion_tokens,
    )


class ContentBatteryTests(unittest.TestCase):
    def test_frozen_prompts_match_upstream_v1_digest(self) -> None:
        digest = hashlib.sha256(
            "\0".join(probe.prompt for probe in PROBES).encode()
        ).hexdigest()
        self.assertEqual(
            digest,
            "87ccd7e8e05c0716f6b12812501eaf2697d62120c14cd9f83e9dcac2b4010097",
        )
        self.assertEqual(len(PROBES), 8)

    def test_tags_are_unique_equal_length_permutations(self) -> None:
        prompts = [
            tagged_prompt(probe_index, repetition)
            for repetition in range(REPETITIONS_PER_PROMPT)
            for probe_index in range(len(PROBES))
        ]
        self.assertEqual(len(prompts), 24)
        self.assertEqual(len(set(prompts)), 24)
        tags = [prompt.split(". Ignore", 1)[0].removeprefix("Benchmark tag ") for prompt in prompts]
        self.assertEqual({len(tag) for tag in tags}, {15})
        for tag in tags:
            self.assertEqual(sorted(tag.split()), list("01234567"))

    def test_only_literal_loopback_v1_endpoints_are_allowed(self) -> None:
        self.assertEqual(
            canonical_loopback_base_url("http://127.0.0.1:30000/v1/"),
            "http://127.0.0.1:30000/v1",
        )
        self.assertEqual(
            canonical_loopback_base_url("http://[::1]:8000/v1"),
            "http://[::1]:8000/v1",
        )
        for invalid in (
            "https://127.0.0.1:30000/v1",
            "http://localhost:30000/v1",
            "http://127.0.0.1/v1",
            "http://127.0.0.1:30000/v1/models",
            "http://127.0.0.1:30000/v1?x=1",
            "http://user@127.0.0.1:30000/v1",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ContentBatteryError):
                    canonical_loopback_base_url(invalid)

    def test_stream_request_uses_exact_usage_and_discards_text(self) -> None:
        events = [
            {"choices": [{"delta": {"reasoning_content": "private reasoning"}}]},
            {"choices": [{"delta": {"content": "generated answer"}}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 41, "completion_tokens": 80},
            },
        ]
        lines = [f"data: {json.dumps(event)}\n".encode() for event in events]
        lines.append(b"data: [DONE]\n")
        opener = _Opener(_Response(lines=lines))
        clock = iter((10.0, 10.25, 14.0)).__next__
        result = stream_request(
            base_url="http://127.0.0.1:30000/v1",
            model="qwen3.8-27b",
            prompt="secret prompt",
            max_tokens=680,
            timeout_s=10,
            api_key="top-secret-key",
            opener=opener,
            clock=clock,
        )
        self.assertEqual(result.prompt_tokens, 41)
        self.assertEqual(result.completion_tokens, 80)
        self.assertEqual(result.ttft_s, 0.25)
        self.assertEqual(result.e2e_s, 4.0)
        self.assertAlmostEqual(result.decode_tps, 79 / 3.75)
        self.assertEqual(result.emission_events, 2)
        self.assertNotIn("content", result.to_dict())
        sent = json.loads(opener.requests[0].data or b"{}")
        self.assertEqual(sent["temperature"], 0.0)
        self.assertEqual(sent["max_tokens"], 680)
        self.assertTrue(sent["stream_options"]["include_usage"])
        self.assertEqual(
            opener.requests[0].get_header("Authorization"),
            "Bearer top-secret-key",
        )
        self.assertNotIn("top-secret-key", json.dumps(result.to_dict()))

    def test_stream_request_fails_without_usage(self) -> None:
        event = {"choices": [{"delta": {"content": "answer"}}]}
        opener = _Opener(
            _Response(lines=[f"data: {json.dumps(event)}\n".encode(), b"data: [DONE]\n"])
        )
        clock = iter((0.0, 0.1, 1.0)).__next__
        with self.assertRaisesRegex(ContentBatteryError, "omitted exact token usage"):
            stream_request(
                base_url="http://127.0.0.1:30000/v1",
                model="qwen3.8-27b",
                prompt="prompt",
                max_tokens=680,
                timeout_s=10,
                opener=opener,
                clock=clock,
            )

    def test_model_verification_sends_bearer_without_persisting_it(self) -> None:
        opener = _Opener(
            _Response(payload={"data": [{"id": "qwen3.8-27b"}]})
        )
        verify_served_model(
            base_url="http://127.0.0.1:30000/v1",
            model="qwen3.8-27b",
            timeout_s=10,
            api_key="top-secret-key",
            opener=opener,
        )
        self.assertEqual(
            opener.requests[0].get_header("Authorization"),
            "Bearer top-secret-key",
        )

    def test_full_battery_is_interleaved_scalar_only_and_aggregated(self) -> None:
        prompts: list[str] = []
        observed_keys: list[object] = []

        def verify(**_: object) -> None:
            return None

        def request(**kwargs: object) -> RequestMetrics:
            prompt = str(kwargs["prompt"])
            prompts.append(prompt)
            observed_keys.append(kwargs.get("api_key"))
            if len(prompts) == 1:
                return _metrics(prompt_tokens=9, completion_tokens=2)
            # The fixed tag permutations preserve whitespace-token count for a
            # given base prompt, which emulates endpoint usage validation.
            return _metrics(
                prompt_tokens=len(prompt.split()),
                completion_tokens=80,
            )

        result = run_battery(
            base_url="http://127.0.0.1:30000/v1",
            model="qwen3.8-27b",
            api_key="top-secret-key",
            opener=object(),
            verify_function=verify,
            request_function=request,
        )
        self.assertEqual(len(prompts), 25)
        self.assertEqual(set(observed_keys), {"top-secret-key"})
        self.assertEqual(len(set(prompts[1:])), 24)
        self.assertEqual(result["summary"]["requests"], 24)
        self.assertEqual(result["summary"]["completion_tokens"], 1920)
        self.assertEqual(result["summary"]["aggregate_output_tps"], 20.0)
        self.assertEqual(len(result["probes"]), 8)
        for probe in result["probes"]:
            self.assertEqual(probe["summary"]["requests"], 3)
            self.assertEqual(len(probe["samples"]), 3)
        serialized = json.dumps(result)
        for probe in PROBES:
            self.assertNotIn(probe.prompt, serialized)
        self.assertNotIn("generated answer", serialized)
        self.assertNotIn("top-secret-key", serialized)
        self.assertNotIn("prompt", result["probes"][0]["samples"][0])

    def test_api_key_file_is_bounded_and_never_followed_through_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key_file = root / "key"
            key_file.write_text("secret-value\n")
            self.assertEqual(read_api_key(key_file), "secret-value")
            link = root / "link"
            link.symlink_to(key_file)
            with self.assertRaisesRegex(ContentBatteryError, "symbolic link"):
                read_api_key(link)
            empty = root / "empty"
            empty.write_text("")
            with self.assertRaisesRegex(ContentBatteryError, "invalid size"):
                read_api_key(empty)

    def test_short_measured_output_fails_the_run(self) -> None:
        calls = 0

        def request(**_: object) -> RequestMetrics:
            nonlocal calls
            calls += 1
            return _metrics(
                completion_tokens=1 if calls == 1 else MIN_OUTPUT_TOKENS - 1
            )

        with self.assertRaisesRegex(ContentBatteryError, "Short output"):
            run_battery(
                base_url="http://127.0.0.1:30000/v1",
                model="qwen3.8-27b",
                opener=object(),
                verify_function=lambda **_: None,
                request_function=request,
            )

    def test_token_length_mismatch_fails_instead_of_biasing_result(self) -> None:
        calls = 0

        def request(**_: object) -> RequestMetrics:
            nonlocal calls
            calls += 1
            if calls == 1:
                return _metrics(completion_tokens=1)
            # Repetition two for each probe occurs after the first eight
            # measured requests, creating a tokenizer-length mismatch.
            prompt_tokens = 100 if calls <= 9 else 101
            return _metrics(prompt_tokens=prompt_tokens)

        with self.assertRaisesRegex(ContentBatteryError, "prompt-token length"):
            run_battery(
                base_url="http://127.0.0.1:30000/v1",
                model="qwen3.8-27b",
                opener=object(),
                verify_function=lambda **_: None,
                request_function=request,
            )

    def test_write_result_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "result.json"
            write_result(path, {"scalar": 1})
            self.assertEqual(json.loads(path.read_text()), {"scalar": 1})
            with self.assertRaisesRegex(ContentBatteryError, "already exists"):
                write_result(path, {"scalar": 2})
            self.assertEqual(json.loads(path.read_text()), {"scalar": 1})

    def test_cli_requires_explicit_output_and_prints_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            key_file = Path(directory) / "api-key"
            key_file.write_text("secret-value\n")
            with (
                patch(
                    "bench.content_battery.run_battery", return_value={"ok": 1}
                ) as run,
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                status = main(
                    [
                        "--base-url",
                        "http://127.0.0.1:30000/v1",
                        "--model",
                        "qwen3.8-27b",
                        "--api-key-file",
                        str(key_file),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.read_text()), {"ok": 1})
            self.assertEqual(stdout.getvalue().strip(), str(output))
            self.assertEqual(run.call_args.kwargs["api_key"], "secret-value")
            self.assertNotIn("secret-value", output.read_text())

    def test_cli_refuses_existing_output_before_running(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text("original")
            with patch("bench.content_battery.run_battery") as run:
                with self.assertRaisesRegex(SystemExit, "already exists"):
                    main(
                        [
                            "--base-url",
                            "http://127.0.0.1:30000/v1",
                            "--model",
                            "qwen3.8-27b",
                            "--output",
                            str(output),
                        ]
                    )
            run.assert_not_called()
            self.assertEqual(output.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
