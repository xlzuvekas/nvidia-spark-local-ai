from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.client import (
    BenchmarkRequestError,
    concurrent_chat_requests,
    embedding_request,
    multimodal_embedding_request,
    score_request,
    stream_chat_request,
    stream_ollama_chat_request,
)
from bench.runner import _validate_capability


def _event(payload: dict[str, object], *, space: bool = True) -> bytes:
    separator = " " if space else ""
    return f"data:{separator}{json.dumps(payload)}\n\n".encode()


class _SSEHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.received.append((self.path, json.loads(body)))  # type: ignore[attr-defined]
        authorization = self.headers.get("Authorization")
        self.server.received_authorizations.append(authorization)  # type: ignore[attr-defined]
        expected_authorization = self.server.expected_authorization  # type: ignore[attr-defined]
        authorized = (
            expected_authorization is None
            or authorization == expected_authorization
        )
        status = self.server.status if authorized else 401  # type: ignore[attr-defined]
        response_bodies = self.server.response_bodies  # type: ignore[attr-defined]
        response_body = (
            response_bodies.pop(0)
            if response_bodies
            else self.server.response_body  # type: ignore[attr-defined]
        )
        if not authorized:
            response_body = b'{"error":"unauthorized"}'
        self.send_response(status)
        self.send_header(
            "Content-Type",
            "text/event-stream" if status < 400 else "application/json",
        )
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format: str, *args: object) -> None:
        return


class StreamingClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _SSEHandler)
        self.server.received = []  # type: ignore[attr-defined]
        self.server.status = 200  # type: ignore[attr-defined]
        self.server.response_body = b""  # type: ignore[attr-defined]
        self.server.response_bodies = []  # type: ignore[attr-defined]
        self.server.expected_authorization = None  # type: ignore[attr-defined]
        self.server.received_authorizations = []  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}/v1"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, **overrides: object):
        arguments: dict[str, object] = {
            "base_url": self.base_url,
            "model": "test-model",
            "prompt": "hello",
            "max_tokens": 8,
            "temperature": 0.0,
            "request_id": "request-1",
        }
        arguments.update(overrides)
        return stream_chat_request(**arguments)  # type: ignore[arg-type]

    def test_parses_content_reasoning_usage_and_fragmented_tool_calls(self) -> None:
        events = [
            _event({"model": "served", "choices": [{"delta": {"role": "assistant"}}]}),
            _event({"choices": [{"delta": {"reasoning_content": "think "}}]}),
            _event({"choices": [{"delta": {"content": "hel"}}]}),
            _event({"choices": [{"delta": {"content": "lo"}}]}),
            _event(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_",
                                        "function": {"name": "multi", "arguments": '{"a":6,'},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            _event(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "1",
                                        "function": {"name": "ply", "arguments": '"b":7}'},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ),
            _event(
                {
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 6,
                        "reasoning_tokens": 12,
                    },
                    "choices": [],
                }
            ),
            b"data: [DONE]\n\n",
        ]
        self.server.response_body = b"".join(events)  # type: ignore[attr-defined]

        result = self._request(extra_body={"seed": 7})

        self.assertEqual(result.content, "hello")
        self.assertEqual(result.reasoning, "think ")
        self.assertEqual(result.prompt_tokens, 4)
        self.assertEqual(result.completion_tokens, 6)
        self.assertEqual(result.reasoning_tokens, 12)
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.response_model, "served")
        self.assertEqual(result.emission_events, 5)
        self.assertEqual(
            result.tool_calls,
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "multiply", "arguments": '{"a":6,"b":7}'},
                }
            ],
        )
        self.assertGreaterEqual(result.ttft_s, 0)
        self.assertGreater(result.output_tps, 0)
        path, request = self.server.received[0]  # type: ignore[attr-defined]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertTrue(request["stream"])
        self.assertEqual(request["stream_options"], {"include_usage": True})
        self.assertEqual(request["seed"], 7)
        self.assertEqual(
            self.server.received_authorizations, [None]  # type: ignore[attr-defined]
        )

    def test_concurrent_requests_complete_against_threaded_server(self) -> None:
        self.server.response_body = b"".join(  # type: ignore[attr-defined]
            [
                _event({"choices": [{"delta": {"content": "ok"}}]}),
                _event({"usage": {"prompt_tokens": 2, "completion_tokens": 1}, "choices": []}),
                b"data: [DONE]\n\n",
            ]
        )
        requests = [
            {
                "base_url": self.base_url,
                "model": "test-model",
                "prompt": f"prompt-{index}",
                "max_tokens": 1,
                "temperature": 0.0,
                "request_id": f"request-{index}",
            }
            for index in range(2)
        ]

        results, wall_s = concurrent_chat_requests(requests=requests, concurrency=2)

        self.assertEqual({result.request_id for result in results}, {"request-0", "request-1"})
        self.assertEqual(len(self.server.received), 2)  # type: ignore[attr-defined]
        self.assertGreater(wall_s, 0)

    def test_ephemeral_bearer_rejects_browser_and_wrong_key_requests(self) -> None:
        correct = "Bearer ephemeral-correct-key"
        wrong = "Bearer secret-wrong-key"
        self.server.expected_authorization = correct  # type: ignore[attr-defined]
        self.server.response_body = b"".join(  # type: ignore[attr-defined]
            [
                _event({"choices": [{"delta": {"content": "ok"}}]}),
                _event(
                    {
                        "usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 1,
                        },
                        "choices": [],
                    }
                ),
                b"data: [DONE]\n\n",
            ]
        )

        with self.assertRaisesRegex(BenchmarkRequestError, "HTTP 401"):
            self._request()
        with self.assertRaisesRegex(BenchmarkRequestError, "HTTP 401") as raised:
            self._request(authorization=wrong)
        self.assertNotIn("secret-wrong-key", str(raised.exception))

        result = self._request(authorization=correct)

        self.assertEqual(result.content, "ok")
        self.assertEqual(
            self.server.received_authorizations,  # type: ignore[attr-defined]
            [None, wrong, correct],
        )

    def test_invalid_json_and_http_errors_have_context(self) -> None:
        self.server.response_body = b"data: {not-json}\n\n"  # type: ignore[attr-defined]
        with self.assertRaisesRegex(BenchmarkRequestError, "Invalid SSE JSON"):
            self._request()

        self.server.status = 503  # type: ignore[attr-defined]
        self.server.response_body = b'{"error":"warming up"}'  # type: ignore[attr-defined]
        with self.assertRaisesRegex(BenchmarkRequestError, "503.*warming up"):
            self._request()

    def test_accepts_standard_sse_data_field_without_optional_space(self) -> None:
        self.server.response_body = b"".join(  # type: ignore[attr-defined]
            [
                _event({"choices": [{"delta": {"content": "ok"}}]}, space=False),
                _event(
                    {"usage": {"prompt_tokens": 1, "completion_tokens": 1}, "choices": []},
                    space=False,
                ),
                b"data:[DONE]\n\n",
            ]
        )

        result = self._request()

        self.assertEqual(result.content, "ok")
        self.assertEqual(result.completion_tokens, 1)

    def test_missing_reasoning_tokens_remain_unreported(self) -> None:
        self.server.response_body = b"".join(  # type: ignore[attr-defined]
            [
                _event({"choices": [{"delta": {"content": "ok"}}]},
                       space=False),
                _event(
                    {"usage": {"prompt_tokens": 1, "completion_tokens": 2}, "choices": []},
                    space=False,
                ),
                b"data:[DONE]\n\n",
            ]
        )

        result = self._request()

        self.assertIsNone(result.reasoning_tokens)

    def test_parses_nested_reasoning_tokens(self) -> None:
        self.server.response_body = b"".join(  # type: ignore[attr-defined]
            [
                _event({"choices": [{"delta": {"content": "ok"}}]}),
                _event(
                    {
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 2,
                            "reasoning_tokens": "untrusted",
                            "completion_tokens_details": {"reasoning_tokens": 7},
                        },
                        "choices": [],
                    }
                ),
                b"data: [DONE]\n\n",
            ]
        )

        result = self._request()

        self.assertEqual(result.reasoning_tokens, 7)

    def test_top_level_reasoning_tokens_preserve_reported_zero(self) -> None:
        self.server.response_body = b"".join(  # type: ignore[attr-defined]
            [
                _event({"choices": [{"delta": {"content": "ok"}}]}),
                _event(
                    {
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 2,
                            "reasoning_tokens": 0,
                            "completion_tokens_details": {"reasoning_tokens": 7},
                        },
                        "choices": [],
                    }
                ),
                b"data: [DONE]\n\n",
            ]
        )

        result = self._request()

        self.assertEqual(result.reasoning_tokens, 0)

    def test_invalid_reasoning_token_usage_remains_unreported(self) -> None:
        self.server.response_body = b"".join(  # type: ignore[attr-defined]
            [
                _event({"choices": [{"delta": {"content": "ok"}}]}),
                _event(
                    {
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 2,
                            "reasoning_tokens": True,
                            "completion_tokens_details": {"reasoning_tokens": "7"},
                        },
                        "choices": [],
                    }
                ),
                b"data: [DONE]\n\n",
            ]
        )

        result = self._request()

        self.assertIsNone(result.reasoning_tokens)

    def test_chat_connection_resets_are_wrapped_with_request_context(self) -> None:
        failures = (
            ConnectionResetError("peer reset the connection"),
            http.client.RemoteDisconnected("peer closed without a response"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                with patch("bench.client.urllib.request.urlopen", side_effect=failure):
                    with self.assertRaises(BenchmarkRequestError) as raised:
                        self._request(request_id="chat-reset-17")
                message = str(raised.exception)
                self.assertIn("chat-reset-17", message)
                self.assertIn("chat", message.lower())

    def test_embedding_disconnect_is_wrapped_with_request_context(self) -> None:
        with patch(
            "bench.client.urllib.request.urlopen",
            side_effect=http.client.RemoteDisconnected("peer closed"),
        ):
            with self.assertRaises(BenchmarkRequestError) as raised:
                embedding_request(
                    base_url=self.base_url,
                    model="embedding-model",
                    inputs=["one"],
                    request_id="embedding-reset-9",
                )

        message = str(raised.exception)
        self.assertIn("embedding-reset-9", message)
        self.assertIn("embedding", message.lower())

    def test_text_embedding_request_keeps_the_openai_input_shape(self) -> None:
        self.server.response_body = json.dumps(  # type: ignore[attr-defined]
            {
                "model": "text-embedding",
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [0.0, 1.0]},
                ],
                "usage": {"prompt_tokens": 4},
            }
        ).encode()

        result = embedding_request(
            base_url=self.base_url,
            model="text-embedding",
            inputs=["one", "two"],
            request_id="text-embedding-1",
        )

        path, payload = self.server.received[0]  # type: ignore[attr-defined]
        self.assertEqual(path, "/v1/embeddings")
        self.assertEqual(
            payload,
            {"model": "text-embedding", "input": ["one", "two"]},
        )
        self.assertEqual(result.batch_size, 2)
        self.assertEqual(result.dimension, 2)

    def test_score_request_posts_vllm_batch_and_builds_deterministic_ranking(self) -> None:
        self.server.response_body = json.dumps(  # type: ignore[attr-defined]
            {
                "model": "served-reranker",
                "data": [
                    {"index": 2, "object": "score", "score": 0.25},
                    {"index": 0, "object": "score", "score": 0.10},
                    {"index": 1, "object": "score", "score": 0.95},
                ],
                "usage": {"total_tokens": 17},
            }
        ).encode()

        result = score_request(
            base_url=self.base_url,
            model="reranker",
            query="query",
            candidates=["candidate zero", "candidate one", "candidate two"],
            request_id="rerank-1",
        )

        path, request = self.server.received[0]  # type: ignore[attr-defined]
        self.assertEqual(path, "/v1/score")
        self.assertEqual(
            request,
            {
                "model": "reranker",
                "queries": "query",
                "documents": [
                    "candidate zero",
                    "candidate one",
                    "candidate two",
                ],
            },
        )
        self.assertEqual(result.response_model, "served-reranker")
        self.assertEqual(result.prompt_tokens, 17)
        self.assertEqual(result.candidate_count, 3)
        self.assertEqual(result.scores, [0.10, 0.95, 0.25])
        self.assertEqual(result.ranking, [1, 2, 0])
        self.assertEqual(result.top_index, 1)
        self.assertTrue(result.finite)
        self.assertGreater(result.pairs_per_s, 0)
        self.assertEqual(result.to_dict()["tool_calls"], [])

    def test_score_request_rejects_incomplete_or_duplicate_indexes(self) -> None:
        invalid_data_sets = (
            [{"index": 0, "score": 0.1}],
            [{"index": 0, "score": 0.1}, {"index": 0, "score": 0.2}],
        )
        for index, data in enumerate(invalid_data_sets):
            with self.subTest(index=index):
                self.server.response_body = json.dumps(  # type: ignore[attr-defined]
                    {"model": "reranker", "data": data}
                ).encode()
                with self.assertRaisesRegex(BenchmarkRequestError, "Score response"):
                    score_request(
                        base_url=self.base_url,
                        model="reranker",
                        query="query",
                        candidates=["one", "two"],
                        request_id=f"rerank-invalid-{index}",
                    )

    def test_score_request_accepts_multimodal_documents_without_returning_payloads(
        self,
    ) -> None:
        self.server.response_body = json.dumps(  # type: ignore[attr-defined]
            {
                "model": "served-qwen3-vl-reranker",
                "data": [
                    {"index": 1, "object": "score", "score": 0.9},
                    {"index": 0, "object": "score", "score": 0.1},
                ],
                "usage": {"prompt_tokens": 42, "total_tokens": 42},
            }
        ).encode()
        documents = [
            {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,local-blue-image"
                        },
                    }
                ]
            },
            {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,local-red-image"
                        },
                    }
                ]
            },
        ]

        result = score_request(
            base_url=self.base_url,
            model="qwen3-vl-reranker",
            query="A solid red square.",
            candidates=documents,
            instruction="Retrieve images relevant to the user's query.",
            request_id="multimodal-rerank-1",
        )

        path, payload = self.server.received[0]  # type: ignore[attr-defined]
        self.assertEqual(path, "/v1/score")
        self.assertEqual(payload["documents"], documents)
        self.assertEqual(
            payload["instruction"],
            "Retrieve images relevant to the user's query.",
        )
        self.assertEqual(result.prompt_tokens, 42)
        self.assertEqual(result.scores, [0.1, 0.9])
        self.assertEqual(result.ranking, [1, 0])
        self.assertEqual(result.top_index, 1)
        serialized = json.dumps(result.to_dict())
        self.assertNotIn("local-blue-image", serialized)
        self.assertNotIn("local-red-image", serialized)

    def test_multimodal_score_http_error_does_not_expose_image_payload(self) -> None:
        self.server.status = 422  # type: ignore[attr-defined]
        self.server.response_body = (  # type: ignore[attr-defined]
            b'{"detail":"data:image/png;base64,sensitive-image-payload"}'
        )

        with self.assertRaises(BenchmarkRequestError) as raised:
            score_request(
                base_url=self.base_url,
                model="qwen3-vl-reranker",
                query="red",
                candidates=[
                    {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,sensitive-image-payload"
                                },
                            }
                        ]
                    }
                ],
                request_id="multimodal-rerank-error",
            )

        self.assertNotIn("sensitive-image-payload", str(raised.exception))
        self.assertIn("omitted", str(raised.exception))

    def test_multimodal_embedding_request_uses_chat_extension_and_compares_vectors(
        self,
    ) -> None:
        def response(vector: list[float], prompt_tokens: int) -> bytes:
            return json.dumps(
                {
                    "model": "served-qwen3-vl-embedding",
                    "data": [{"index": 0, "embedding": vector}],
                    "usage": {"prompt_tokens": prompt_tokens},
                }
            ).encode()

        self.server.response_bodies = [  # type: ignore[attr-defined]
            response([1.0, 0.0, 0.0], 5),
            response([0.8, 0.2, 0.0], 7),
            response([0.0, 1.0, 0.0], 9),
        ]
        image_data_url = "data:image/png;base64,local-red-image"

        result = multimodal_embedding_request(
            base_url=self.base_url,
            model="qwen3-vl-embedding",
            image_data_url=image_data_url,
            relevant_text="A solid red square.",
            unrelated_text="Snowy mountains below a blue sky.",
            request_id="multimodal-1",
        )

        self.assertEqual(result.response_model, "served-qwen3-vl-embedding")
        self.assertEqual(result.prompt_tokens, 21)
        self.assertEqual(result.dimension, 3)
        self.assertEqual(result.batch_size, 3)
        self.assertTrue(result.finite)
        self.assertGreater(
            result.relevant_similarity or 0, result.unrelated_similarity or 0
        )
        self.assertGreater(result.similarity_margin or 0, 0)
        self.assertGreaterEqual(result.image_latency_s, 0)
        self.assertGreaterEqual(result.relevant_text_latency_s, 0)
        self.assertGreaterEqual(result.unrelated_text_latency_s, 0)

        received = self.server.received  # type: ignore[attr-defined]
        self.assertEqual([path for path, _ in received], ["/v1/embeddings"] * 3)
        for _, payload in received:
            self.assertEqual(payload["encoding_format"], "float")
            self.assertTrue(payload["continue_final_message"])
            self.assertTrue(payload["add_special_tokens"])
            self.assertNotIn("input", payload)
            self.assertEqual(payload["messages"][0]["role"], "system")
            self.assertEqual(payload["messages"][-1]["role"], "assistant")
        image_content = received[0][1]["messages"][1]["content"]
        self.assertEqual(image_content[0]["image_url"]["url"], image_data_url)
        self.assertEqual(image_content[1], {"type": "text", "text": ""})
        self.assertEqual(
            received[1][1]["messages"][1]["content"],
            [{"type": "text", "text": "A solid red square."}],
        )
        serialized_result = json.dumps(result.to_dict())
        self.assertNotIn("embedding", result.to_dict())
        self.assertNotIn("local-red-image", serialized_result)

    def test_native_ollama_request_enforces_context_and_uses_server_timing(self) -> None:
        self.server.response_body = b"\n".join(  # type: ignore[attr-defined]
            [
                json.dumps(
                    {
                        "model": "native-model",
                        "message": {"content": "one "},
                        "done": False,
                    }
                ).encode(),
                json.dumps(
                    {
                        "model": "native-model",
                        "message": {
                            "content": "two",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "multiply",
                                        "arguments": {"a": 6, "b": 7},
                                    }
                                }
                            ],
                        },
                        "done": True,
                        "done_reason": "length",
                        "prompt_eval_count": 5,
                        "prompt_eval_duration": 50_000_000,
                        "eval_count": 2,
                        "eval_duration": 100_000_000,
                        "load_duration": 25_000_000,
                    }
                ).encode(),
            ]
        ) + b"\n"

        result = stream_ollama_chat_request(
            base_url=self.base_url,
            model="native-model",
            prompt="hello",
            max_tokens=2,
            temperature=0.0,
            request_id="ollama-native-1",
            context_size=32768,
            extra_body={
                "reasoning_effort": "none",
                "response_format": {"type": "json_object"},
                "tool_choice": "required",
                "tools": [{"type": "function", "function": {"name": "multiply"}}],
            },
        )

        path, request = self.server.received[0]  # type: ignore[attr-defined]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(request["options"]["num_ctx"], 32768)
        self.assertEqual(request["options"]["num_predict"], 2)
        self.assertEqual(request["format"], "json")
        self.assertFalse(request["think"])
        self.assertNotIn("tool_choice", request)
        self.assertEqual(result.content, "one two")
        self.assertIsNone(result.reasoning_tokens)
        self.assertEqual(result.decode_metric_source, "server_reported_eval_duration")
        self.assertAlmostEqual(result.decode_tps, 20.0)
        self.assertAlmostEqual(result.load_s or 0, 0.025)
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(
            json.loads(result.tool_calls[0]["function"]["arguments"]),
            {"a": 6, "b": 7},
        )

    def test_native_ollama_rejects_missing_decode_metrics(self) -> None:
        complete = {
            "model": "native-model",
            "message": {"content": "output"},
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 5,
            "prompt_eval_duration": 50_000_000,
            "eval_count": 8,
            "eval_duration": 100_000_000,
        }
        for missing_field in ("eval_count", "eval_duration"):
            with self.subTest(missing_field=missing_field):
                payload = {key: value for key, value in complete.items() if key != missing_field}
                self.server.response_body = json.dumps(payload).encode() + b"\n"  # type: ignore[attr-defined]

                with self.assertRaisesRegex(BenchmarkRequestError, missing_field):
                    stream_ollama_chat_request(
                        base_url=self.base_url,
                        model="native-model",
                        prompt="hello",
                        max_tokens=8,
                        temperature=0.0,
                        request_id=f"ollama-missing-{missing_field}",
                        context_size=32768,
                    )

    def test_native_ollama_short_length_completion_fails_validation(self) -> None:
        self.server.response_body = (  # type: ignore[attr-defined]
            json.dumps(
                {
                    "model": "native-model",
                    "message": {"content": "short output"},
                    "done": True,
                    "done_reason": "length",
                    "prompt_eval_count": 5,
                    "prompt_eval_duration": 50_000_000,
                    "eval_count": 7,
                    "eval_duration": 100_000_000,
                }
            ).encode()
            + b"\n"
        )
        result = stream_ollama_chat_request(
            base_url=self.base_url,
            model="native-model",
            prompt="hello",
            max_tokens=8,
            temperature=0.0,
            request_id="ollama-short-length",
            context_size=32768,
        )

        for kind in ("decode", "concurrency"):
            with self.subTest(kind=kind):
                validation = _validate_capability(
                    SimpleNamespace(kind=kind, max_output_tokens=8), result
                )
                self.assertIsNotNone(validation)
                self.assertFalse(validation["passed"])


if __name__ == "__main__":
    unittest.main()
