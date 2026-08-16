from __future__ import annotations

import base64
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zlib

from bench.client import (
    BenchmarkRequestError,
    RequestResult,
    _ollama_messages,
    stream_ollama_chat_request,
)
from bench.journal import Journal
from bench.manifest import load_models, load_suite
from bench.runner import (
    _OCR_EXPECTED_TRANSCRIPTION,
    _OCR_IMAGE_HEIGHT,
    _OCR_IMAGE_WIDTH,
    _estimated_context_tokens,
    _execute_case,
    _request_arguments,
    _validate_capability,
)


ROOT = Path(__file__).resolve().parents[1]


def _case(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": "ocr-exact-token",
        "case_id": "ocr-exact-token--test",
        "kind": "capability",
        "requires": ["chat", "vision", "ocr"],
        "warmups": 1,
        "repetitions": 2,
        "max_output_tokens": 32,
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
        prompt_tokens=100,
        completion_tokens=4,
        ttft_s=0.02,
        elapsed_s=0.05,
        decode_s=0.03,
        decode_tps=100.0,
        output_tps=80.0,
        emission_events=4,
        finish_reason="stop",
        response_model="deepseek-ocr:latest",
        content=content,
        reasoning="",
        tool_calls=[],
    )


def _png_chunks(png: bytes) -> dict[bytes, list[bytes]]:
    chunks: dict[bytes, list[bytes]] = {}
    offset = 8
    while offset < len(png):
        length = struct.unpack(">I", png[offset : offset + 4])[0]
        kind = png[offset + 4 : offset + 8]
        data = png[offset + 8 : offset + 8 + length]
        chunks.setdefault(kind, []).append(data)
        offset += 12 + length
    return chunks


class _LineResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.lines = [json.dumps(payload).encode() + b"\n"]

    def __enter__(self) -> _LineResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):
        return iter(self.lines)


class OcrWorkloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = SimpleNamespace(
            backend="ollama", base_url="http://127.0.0.1:11434/v1"
        )
        self.model = SimpleNamespace(
            served_name="deepseek-ocr:latest", max_context=8192
        )

    def test_only_deepseek_declares_ocr_and_suite_uses_the_gate(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        ocr_models = [model.id for model in models.values() if "ocr" in model.tasks]
        suite = load_suite(ROOT / "manifests" / "suites" / "ocr.toml")

        self.assertEqual(ocr_models, ["ollama-deepseek-ocr-f16"])
        self.assertEqual(suite.id, "ocr")
        self.assertEqual(len(suite.cases), 1)
        case = suite.cases[0]
        self.assertEqual(case.id, "ocr-exact-token")
        self.assertEqual(case.requires, ("chat", "vision", "ocr"))
        self.assertEqual(case.warmups, 1)
        self.assertEqual(case.repetitions, 3)
        self.assertEqual(case.concurrency, 1)
        self.assertEqual(case.temperature, 0.0)
        self.assertIn("never journaled", suite.description)

    def test_request_contains_deterministic_high_contrast_png_for_both_adapters(
        self,
    ) -> None:
        first = _request_arguments(
            server=self.server,
            model=self.model,
            case=_case(),
            request_id="ocr-first",
        )
        second = _request_arguments(
            server=self.server,
            model=self.model,
            case=_case(),
            request_id="ocr-second",
        )
        first_content = first["extra_body"]["messages"][0]["content"]
        second_content = second["extra_body"]["messages"][0]["content"]
        image_url = first_content[1]["image_url"]["url"]

        self.assertEqual(image_url, second_content[1]["image_url"]["url"])
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        self.assertFalse(first["require_native_decode_timing"])
        self.assertNotIn(_OCR_EXPECTED_TRANSCRIPTION, first["prompt"])
        self.assertNotIn(_OCR_EXPECTED_TRANSCRIPTION, first_content[0]["text"])

        png = base64.b64decode(image_url.partition(",")[2], validate=True)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        chunks = _png_chunks(png)
        width, height = struct.unpack(">II", chunks[b"IHDR"][0][:8])
        self.assertEqual((width, height), (_OCR_IMAGE_WIDTH, _OCR_IMAGE_HEIGHT))
        scanlines = zlib.decompress(b"".join(chunks[b"IDAT"]))
        pixels = b"".join(
            scanlines[row * (width * 3 + 1) + 1 : (row + 1) * (width * 3 + 1)]
            for row in range(height)
        )
        self.assertIn(b"\x00\x00\x00", pixels)
        self.assertIn(b"\xff\xff\xff", pixels)

        translated = _ollama_messages(
            str(first["prompt"]), dict(first["extra_body"])
        )
        self.assertEqual(translated[0]["content"], first_content[0]["text"])
        self.assertEqual(translated[0]["images"], [image_url.partition(",")[2]])

    def test_only_vision_capabilities_relax_native_decode_timing(self) -> None:
        strict_cases = (
            _case(kind="decode", requires=["chat"]),
            _case(kind="concurrency", requires=["chat"]),
            _case(kind="prefill", requires=["chat"], prompt_repetitions=256),
            _case(kind="capability", requires=["chat", "json"]),
        )
        relaxed_cases = (
            _case(kind="capability", requires=["chat", "vision"]),
            _case(kind="capability", requires=["chat", "vision", "ocr"]),
        )

        for case in strict_cases:
            with self.subTest(kind=case.kind, requires=case.requires):
                arguments = _request_arguments(
                    server=self.server,
                    model=self.model,
                    case=case,
                    request_id="strict-timing",
                )
                self.assertTrue(arguments["require_native_decode_timing"])
        for case in relaxed_cases:
            with self.subTest(kind=case.kind, requires=case.requires):
                arguments = _request_arguments(
                    server=self.server,
                    model=self.model,
                    case=case,
                    request_id="relaxed-timing",
                )
                self.assertFalse(arguments["require_native_decode_timing"])

        vllm = SimpleNamespace(
            backend="vllm", base_url="http://127.0.0.1:8000/v1"
        )
        vllm_arguments = _request_arguments(
            server=vllm,
            model=self.model,
            case=relaxed_cases[1],
            request_id="vllm-unchanged",
        )
        self.assertNotIn("require_native_decode_timing", vllm_arguments)

    def test_ollama_ocr_missing_eval_duration_has_null_decode_metrics(self) -> None:
        final_record = {
            "model": "deepseek-ocr:latest",
            "message": {"content": _OCR_EXPECTED_TRANSCRIPTION},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 700,
            "prompt_eval_duration": 1_000_000_000,
            "eval_count": 6,
        }
        arguments = {
            "base_url": self.server.base_url,
            "model": self.model.served_name,
            "prompt": "transcribe",
            "max_tokens": 32,
            "temperature": 0.0,
            "request_id": "ocr-missing-eval-duration",
            "context_size": self.model.max_context,
        }

        with patch(
            "bench.client.urllib.request.urlopen",
            return_value=_LineResponse(final_record),
        ):
            result = stream_ollama_chat_request(
                **arguments, require_native_decode_timing=False
            )

        self.assertEqual(result.content, _OCR_EXPECTED_TRANSCRIPTION)
        self.assertIsNone(result.decode_s)
        self.assertIsNone(result.decode_tps)
        self.assertIsNone(result.decode_metric_source)
        self.assertIsNone(result.to_dict()["decode_tps"])
        self.assertGreater(result.output_tps, 0)

        with (
            patch(
                "bench.client.urllib.request.urlopen",
                return_value=_LineResponse(final_record),
            ),
            self.assertRaisesRegex(BenchmarkRequestError, "eval_duration"),
        ):
            stream_ollama_chat_request(**arguments)

    def test_validation_is_normalized_exact_match_not_substring(self) -> None:
        passing = (
            _OCR_EXPECTED_TRANSCRIPTION,
            f"  {_OCR_EXPECTED_TRANSCRIPTION}\n",
            "ＳＰＡＲＫＯＣＲ４８２７",
        )
        failing = (
            f"The token is {_OCR_EXPECTED_TRANSCRIPTION}",
            f"{_OCR_EXPECTED_TRANSCRIPTION}.",
            _OCR_EXPECTED_TRANSCRIPTION.lower(),
            _OCR_EXPECTED_TRANSCRIPTION[:-1],
            f"not {_OCR_EXPECTED_TRANSCRIPTION}",
            f"{_OCR_EXPECTED_TRANSCRIPTION}\n{_OCR_EXPECTED_TRANSCRIPTION}",
        )

        for content in passing:
            with self.subTest(content=content):
                validation = _validate_capability(
                    _case(), _result("passing", content)
                )
                self.assertTrue(validation["passed"])
        for content in failing:
            with self.subTest(content=content):
                validation = _validate_capability(
                    _case(), _result("failing", content)
                )
                self.assertFalse(validation["passed"])
                self.assertIn("exactly match", validation["reason"])

        hidden_only = _result("hidden-only", "wrong")
        hidden_only.reasoning = _OCR_EXPECTED_TRANSCRIPTION
        self.assertFalse(_validate_capability(_case(), hidden_only)["passed"])

    def test_workload_never_journals_image_or_base64_payload(self) -> None:
        case = _case()
        captured_image_url = ""

        def measured(
            *,
            requests: list[dict[str, object]],
            concurrency: int,
            request_function: object,
        ) -> tuple[list[RequestResult], float]:
            nonlocal captured_image_url
            self.assertEqual(concurrency, 1)
            self.assertEqual(len(requests), 1)
            content = requests[0]["extra_body"]["messages"][0]["content"]
            captured_image_url = content[1]["image_url"]["url"]
            self.assertTrue(captured_image_url.startswith("data:image/png;base64,"))
            return [
                _result(
                    str(requests[0]["request_id"]),
                    _OCR_EXPECTED_TRANSCRIPTION,
                )
            ], 0.05

        with tempfile.TemporaryDirectory() as directory:
            events_path = Path(directory) / "events.jsonl"
            journal = Journal(events_path)
            with (
                patch(
                    "bench.runner.stream_ollama_chat_request",
                    return_value=_result("warmup", _OCR_EXPECTED_TRANSCRIPTION),
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
        self.assertTrue(captured_image_url)
        self.assertNotIn("data:image/png;base64,", serialized_events)
        self.assertNotIn(captured_image_url.partition(",")[2], serialized_events)
        request_events = [
            event for event in events if event["event"] == "request_complete"
        ]
        self.assertEqual(len(request_events), 2)
        self.assertTrue(all(event["validation"]["passed"] for event in request_events))

    def test_ocr_context_estimate_uses_embedded_image_dimensions(self) -> None:
        estimate, basis = _estimated_context_tokens(_case())

        self.assertLess(estimate, 8192)
        self.assertEqual(basis, "ocr_image_patch14_plus_margin")


if __name__ == "__main__":
    unittest.main()
