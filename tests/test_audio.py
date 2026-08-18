from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.client import (
    BenchmarkRequestError,
    RequestResult,
    stream_audio_chat_request,
)
from bench.journal import Journal
from bench.manifest import KNOWN_TASKS, load_suite
from bench.runner import (
    _AUDIO_EXPECTED_TRANSCRIPTION,
    _AUDIO_FIXTURE_PATH,
    _AUDIO_FIXTURE_SHA256,
    _AUDIO_LORA_NAME,
    _chat_request_function,
    _estimated_context_tokens,
    _execute_case,
    _request_arguments,
    _validate_capability,
)


ROOT = Path(__file__).resolve().parents[1]


def _case(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": "spark-tts-chinese-exact",
        "case_id": "spark-tts-chinese-exact--test",
        "kind": "capability",
        "requires": ["chat", "audio"],
        "warmups": 1,
        "repetitions": 2,
        "max_output_tokens": 128,
        "temperature": 0.0,
        "concurrency": 1,
        "prompt_repetitions": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _result(request_id: str, content: str) -> RequestResult:
    return RequestResult(
        request_id=request_id,
        started_at_ns=1,
        prompt_tokens=500,
        completion_tokens=64,
        reasoning_tokens=None,
        ttft_s=0.02,
        elapsed_s=0.08,
        decode_s=0.06,
        decode_tps=1000.0,
        output_tps=800.0,
        emission_events=8,
        finish_reason="stop",
        response_model="nvidia/Phi-4-multimodal-instruct-FP8",
        content=content,
        reasoning="",
        tool_calls=[],
    )


class _SseResponse:
    def __init__(self, content: str) -> None:
        events = [
            {
                "model": "phi-speech",
                "choices": [
                    {"delta": {"content": content}, "finish_reason": "stop"}
                ],
            },
            {
                "usage": {"prompt_tokens": 500, "completion_tokens": 64},
                "choices": [],
            },
        ]
        self.lines = [
            *(f"data: {json.dumps(event)}\n\n".encode() for event in events),
            b"data: [DONE]\n\n",
        ]

    def __enter__(self) -> _SseResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):
        return iter(self.lines)


class AudioWorkloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = SimpleNamespace(
            backend="sglang", base_url="http://127.0.0.1:30000/v1"
        )
        self.model = SimpleNamespace(
            served_name="nvidia/Phi-4-multimodal-instruct-FP8",
            max_context=32768,
            request_body_json=None,
            architecture="dense-vlm",
        )

    def test_suite_uses_explicit_audio_task_gate(self) -> None:
        suite = load_suite(ROOT / "manifests" / "suites" / "audio_asr.toml")

        self.assertIn("audio", KNOWN_TASKS)
        self.assertEqual(suite.id, "audio-asr")
        self.assertEqual(len(suite.cases), 1)
        case = suite.cases[0]
        self.assertEqual(case.id, "spark-tts-chinese-exact")
        self.assertEqual(case.requires, ("chat", "audio"))
        self.assertEqual(case.warmups, 1)
        self.assertEqual(case.repetitions, 3)
        self.assertEqual(case.concurrency, 1)
        self.assertEqual(case.max_output_tokens, 128)
        self.assertEqual(case.temperature, 0.0)
        self.assertIn("9.953313-second", suite.description)
        self.assertIn("never journaled", suite.description)

    def test_client_emits_rc0_audio_url_shape_and_selects_speech_lora(self) -> None:
        fixture = b"RIFF\x08\x00\x00\x00WAVEmock"
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "fixture.wav"
            audio_path.write_bytes(fixture)
            with patch(
                "bench.client.urllib.request.urlopen",
                return_value=_SseResponse(_AUDIO_EXPECTED_TRANSCRIPTION),
            ) as urlopen:
                result = stream_audio_chat_request(
                    base_url=self.server.base_url,
                    model=self.model.served_name,
                    prompt="Transcribe the audio clip into text.",
                    max_tokens=128,
                    temperature=0.0,
                    request_id="audio-request",
                    audio_path=audio_path,
                    expected_audio_sha256=hashlib.sha256(fixture).hexdigest(),
                    lora_path="speech",
                )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        content = payload["messages"][0]["content"]
        self.assertEqual(payload["lora_path"], "speech")
        self.assertEqual([part["type"] for part in content], ["audio_url", "text"])
        data_url = content[0]["audio_url"]["url"]
        self.assertTrue(data_url.startswith("data:audio/wav;base64,"))
        self.assertEqual(base64.b64decode(data_url.partition(",")[2]), fixture)
        self.assertEqual(content[1]["text"], "Transcribe the audio clip into text.")
        self.assertEqual(result.content, _AUDIO_EXPECTED_TRANSCRIPTION)

    def test_client_pins_fixture_hash_and_redacts_request_failures(self) -> None:
        fixture = b"RIFF\x08\x00\x00\x00WAVEmock"
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "fixture.wav"
            audio_path.write_bytes(fixture)
            arguments = {
                "base_url": self.server.base_url,
                "model": self.model.served_name,
                "prompt": "transcribe",
                "max_tokens": 128,
                "temperature": 0.0,
                "request_id": "redacted-audio-request",
                "audio_path": audio_path,
                "expected_audio_sha256": hashlib.sha256(fixture).hexdigest(),
                "lora_path": "speech",
            }
            with self.assertRaisesRegex(BenchmarkRequestError, "SHA-256"):
                stream_audio_chat_request(
                    **{**arguments, "expected_audio_sha256": "0" * 64}
                )
            with patch(
                "bench.client.stream_chat_request",
                side_effect=BenchmarkRequestError(
                    "HTTP 400 data:audio/wav;base64,SECRET-PAYLOAD"
                ),
            ):
                with self.assertRaises(BenchmarkRequestError) as raised:
                    stream_audio_chat_request(**arguments)

        message = str(raised.exception)
        self.assertIn("redacted-audio-request", message)
        self.assertNotIn("base64", message)
        self.assertNotIn("SECRET-PAYLOAD", message)

    def test_runner_keeps_audio_out_of_arguments_and_requires_sglang(self) -> None:
        arguments = _request_arguments(
            server=self.server,
            model=self.model,
            case=_case(),
            request_id="audio-arguments",
        )

        self.assertEqual(arguments["audio_path"], _AUDIO_FIXTURE_PATH)
        self.assertEqual(arguments["expected_audio_sha256"], _AUDIO_FIXTURE_SHA256)
        self.assertEqual(arguments["lora_path"], _AUDIO_LORA_NAME)
        self.assertNotIn("messages", arguments["extra_body"])
        self.assertNotIn("base64", repr(arguments))
        self.assertIs(_chat_request_function(self.server, _case()), stream_audio_chat_request)

        ollama = SimpleNamespace(
            backend="ollama", base_url="http://127.0.0.1:11434/v1"
        )
        with self.assertRaisesRegex(RuntimeError, "requires SGLang"):
            _request_arguments(
                server=ollama,
                model=self.model,
                case=_case(),
                request_id="unsupported-audio",
            )

    def test_validation_is_punctuation_insensitive_but_wording_exact(self) -> None:
        passing = (
            _AUDIO_EXPECTED_TRANSCRIPTION,
            f"  {_AUDIO_EXPECTED_TRANSCRIPTION}\n",
            _AUDIO_EXPECTED_TRANSCRIPTION.replace("，", " ").replace("。", ""),
        )
        failing = (
            "以下是转录：" + _AUDIO_EXPECTED_TRANSCRIPTION,
            _AUDIO_EXPECTED_TRANSCRIPTION[:-2],
            _AUDIO_EXPECTED_TRANSCRIPTION + "谢谢",
        )

        for content in passing:
            with self.subTest(content=content):
                self.assertTrue(
                    _validate_capability(_case(), _result("passing", content))["passed"]
                )
        for content in failing:
            with self.subTest(content=content):
                validation = _validate_capability(
                    _case(), _result("failing", content)
                )
                self.assertFalse(validation["passed"])
                self.assertIn("known fixture", validation["reason"])

    def test_workload_never_journals_audio_path_or_encoded_payload(self) -> None:
        case = _case()

        def measured(
            *,
            requests: list[dict[str, object]],
            concurrency: int,
            request_function: object,
        ) -> tuple[list[RequestResult], float]:
            self.assertEqual(concurrency, 1)
            self.assertEqual(len(requests), 1)
            self.assertNotIn("base64", repr(requests))
            return [
                _result(
                    str(requests[0]["request_id"]),
                    _AUDIO_EXPECTED_TRANSCRIPTION,
                )
            ], 0.08

        with tempfile.TemporaryDirectory() as directory:
            events_path = Path(directory) / "events.jsonl"
            journal = Journal(events_path)
            with (
                patch(
                    "bench.runner.stream_audio_chat_request",
                    return_value=_result("warmup", _AUDIO_EXPECTED_TRANSCRIPTION),
                ) as warmup,
                patch(
                    "bench.runner.concurrent_chat_requests", side_effect=measured
                ) as concurrent,
            ):
                _execute_case(
                    server=self.server,
                    model=self.model,
                    case=case,
                    journal=journal,
                    telemetry=Mock(),
                )
            events = journal.events()
            serialized_events = events_path.read_text()

        self.assertEqual(warmup.call_count, 1)
        self.assertEqual(concurrent.call_count, 2)
        self.assertNotIn("data:audio/", serialized_events)
        self.assertNotIn(str(_AUDIO_FIXTURE_PATH), serialized_events)
        request_events = [
            event for event in events if event["event"] == "request_complete"
        ]
        self.assertEqual(len(request_events), 2)
        self.assertTrue(all(event["validation"]["passed"] for event in request_events))

    def test_context_estimate_is_conservative_for_fixed_fixture(self) -> None:
        estimate, basis = _estimated_context_tokens(_case())

        self.assertEqual(estimate, 4096 + 128)
        self.assertEqual(basis, "fixed_9.953313s_audio_plus_margin")
        self.assertLess(estimate, self.model.max_context)


if __name__ == "__main__":
    unittest.main()
