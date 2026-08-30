"""Offline contracts for the uninvoked SM121 C1 runtime identity reader."""

from __future__ import annotations

import io
import http.client
import json
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch
import urllib.error
import urllib.request

from bench import runtime
from bench.sglang_sm121_agent_admission import (
    SM121_AGENT_ADMISSION_ENDPOINT,
    SM121_AGENT_ADMISSION_RUNTIME_EXPECTED,
)
from bench.sglang_sm121_cache_semantic import SM121_CACHE_SEMANTIC_CACHE_ON_ARM


_SERVER_INFO_URL = SM121_AGENT_ADMISSION_ENDPOINT.removesuffix("/v1") + "/server_info"
_METRICS_URL = SM121_AGENT_ADMISSION_ENDPOINT.removesuffix("/v1") + "/metrics"
_STARTUP = (
    b"2026-08-29 00:00:00 Tree cache initialized: source=default "
    b"impl=UnifiedRadixCache hybrid_swa=False hybrid_ssm=True "
    b"hicache_attached=False streaming_wrapped=False\n"
)
_LIVE_GENERATION = ("2026-08-29T00:00:00.000000000Z", 1234)


class _StringSubclass(str):
    def __str__(self) -> str:
        return "Bearer " + "z" * 32


def _completed(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["synthetic"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class _Response:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int = 200,
        url: str = _SERVER_INFO_URL,
        content_length: bool = False,
    ) -> None:
        self._payload = payload
        self._offset = 0
        self.status = status
        self._url = url
        self.read_limits: list[int] = []
        self.socket_timeouts: list[float] = []
        self.fp = SimpleNamespace(
            raw=SimpleNamespace(
                _sock=SimpleNamespace(settimeout=self.socket_timeouts.append)
            )
        )
        self.length: int | None = len(payload) if content_length else None

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def geturl(self) -> str:
        return self._url

    def read1(self, limit: int = -1) -> bytes:
        self.read_limits.append(limit)
        if self._offset >= len(self._payload):
            return b""
        end = len(self._payload) if limit < 0 else self._offset + limit
        chunk = self._payload[self._offset:end]
        self._offset += len(chunk)
        if self.length is not None:
            self.length -= len(chunk)
            if self.length == 0:
                self.fp = None
        return chunk


def _server_info(**overrides: object) -> dict[str, object]:
    return {
        "disable_radix_cache": False,
        "mamba_radix_cache_strategy": "extra_buffer_lazy",
        "max_mamba_cache_size": 4,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
        "chunked_prefill_size": 4096,
        "max_running_requests": 1,
        "max_total_tokens": 65536,
        "context_length": 65536,
        **overrides,
    }


def _server() -> runtime.ManagedServer:
    key = "x" * 32
    return runtime.ManagedServer(
        backend="sglang",
        base_url=SM121_AGENT_ADMISSION_ENDPOINT,
        container_id="synthetic-container",
        run_identity="synthetic-run-identity",
        authorization="Bearer " + key,
        api_key=key,
    )


def _binding(
    *,
    authorization: str = "Bearer " + "x" * 32,
    generation: tuple[str, int] = _LIVE_GENERATION,
) -> object:
    """Return an in-memory-only C1 ownership binding for metrics tests."""

    return runtime._SM121AgentRuntimeBinding(
        "synthetic-container",
        "synthetic-run-identity",
        authorization,
        "x" * 32,
        generation,
    )


def _metrics(*, input_tokens: int = 17) -> str:
    """Produce the complete scalar-only C1 cache-on metric fixture."""

    scheduler = (
        'engine_type="prefill",model_name="synthetic",moe_ep_rank="0",'
        'pp_rank="0",tp_rank="0"'
    )

    def labels(selector: str = "") -> str:
        joined = ",".join(value for value in (scheduler, selector) if value)
        return "{" + joined + "}"

    input_mode = 'mode="input"'
    device_hit_mode = 'mode="device_hit"'
    host_hit_mode = 'mode="host_hit"'
    storage_hit_mode = 'mode="storage_hit"'
    return "\n".join(
        (
            "sglang:prefill_effective_tokens_total"
            f"{labels(input_mode)} {input_tokens}",
            "sglang:prefill_effective_tokens_total"
            f"{labels(device_hit_mode)} 0",
            "sglang:prefill_effective_tokens_total"
            f"{labels(host_hit_mode)} 0",
            "sglang:prefill_effective_tokens_total"
            f"{labels(storage_hit_mode)} 0",
            f"sglang:kv_available_tokens{labels()} 90",
            f"sglang:kv_evictable_tokens{labels()} 0",
            f"sglang:kv_used_tokens{labels()} 10",
            f"sglang:mamba_available_tokens{labels()} 80",
            f"sglang:mamba_evictable_tokens{labels()} 0",
            f"sglang:mamba_used_tokens{labels()} 20",
        )
    ) + "\n"


class SM121AgentAdmissionRuntimeTests(unittest.TestCase):
    def _inspect(
        self,
        payload: bytes | None = None,
        *,
        server: runtime.ManagedServer | None = None,
        response: _Response | None = None,
        logs: bytes = _STARTUP,
    ) -> tuple[dict[str, object], _Response, Mock, Mock]:
        server = server or _server()
        response = response or _Response(
            payload or json.dumps(_server_info()).encode("utf-8")
        )
        opener = Mock(return_value=response)
        log_reader = Mock(return_value=logs)
        with (
            patch.object(
                runtime.ManagedServer,
                "_require_live_owned_loopback_port",
                return_value=_LIVE_GENERATION,
            ) as require_live,
            patch.object(
                runtime,
                "_open_sm121_agent_runtime_server_info",
                opener,
            ),
            patch.object(
                runtime,
                "_read_sm121_agent_runtime_startup_logs",
                log_reader,
            ),
            patch.object(runtime, "inspect_sm121_cache_runtime_identity") as old_cache,
            patch.object(
                runtime,
                "inspect_sm121_chunked_prefill_runtime_identity",
            ) as old_chunked,
        ):
            result = runtime.inspect_sm121_agent_admission_runtime_identity(server)
        self.assertEqual(
            require_live.call_args_list,
            [
                call(server, host_port=30000, container_port=30000),
                call(server, host_port=30000, container_port=30000),
            ],
        )
        old_cache.assert_not_called()
        old_chunked.assert_not_called()
        return result, response, opener, log_reader

    def test_returns_only_the_exact_allowlisted_runtime_identity(self) -> None:
        result, response, opener, log_reader = self._inspect()

        self.assertEqual(result, SM121_AGENT_ADMISSION_RUNTIME_EXPECTED)
        self.assertEqual(opener.call_count, 1)
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, _SERVER_INFO_URL)
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.get_header("Authorization"), "Bearer " + "x" * 32)
        self.assertEqual(opener.call_args.kwargs, {"timeout_s": 15.0})
        self.assertEqual(
            response.read_limits,
            [runtime._SM121_AGENT_RUNTIME_RESPONSE_CHUNK_BYTES, 65536],
        )
        self.assertTrue(response.socket_timeouts)
        log_reader.assert_called_once_with(
            "synthetic-container",
            started_at=_LIVE_GENERATION[0],
        )

    def test_rejects_wrong_server_identity_before_any_runtime_read(self) -> None:
        for mutate in (
            lambda server: setattr(server, "backend", "vllm"),
            lambda server: setattr(server, "base_url", "http://127.0.0.1:30001/v1"),
            lambda server: setattr(server, "authorization", "Bearer " + "y" * 32),
            lambda server: setattr(server, "api_key", None),
            lambda server: setattr(server, "run_identity", ""),
            lambda server: setattr(
                server,
                "authorization",
                _StringSubclass("Bearer " + "x" * 32),
            ),
        ):
            with self.subTest(mutate=mutate):
                server = _server()
                mutate(server)
                opener = Mock()
                logs = Mock()
                with (
                    patch.object(
                        runtime.ManagedServer,
                        "_require_live_owned_loopback_port",
                        return_value=_LIVE_GENERATION,
                    ) as require_live,
                    patch.object(
                        runtime,
                        "_open_sm121_agent_runtime_server_info",
                        opener,
                    ),
                    patch.object(
                        runtime,
                        "_read_sm121_agent_runtime_startup_logs",
                        logs,
                    ),
                    self.assertRaisesRegex(
                        runtime.RuntimeErrorWithContext,
                        "not an owned server",
                    ),
                ):
                    runtime.inspect_sm121_agent_admission_runtime_identity(server)
                require_live.assert_not_called()
                opener.assert_not_called()
                logs.assert_not_called()

        with self.assertRaisesRegex(runtime.RuntimeErrorWithContext, "owned server"):
            runtime.inspect_sm121_agent_admission_runtime_identity(  # type: ignore[arg-type]
                SimpleNamespace()
            )

        class _ManagedServerSubclass(runtime.ManagedServer):
            pass

        source = _server()
        lookalike = _ManagedServerSubclass("", "")
        lookalike.__dict__.update(source.__dict__)
        with self.assertRaisesRegex(runtime.RuntimeErrorWithContext, "owned server"):
            runtime.inspect_sm121_agent_admission_runtime_identity(lookalike)

    def test_server_info_rejects_status_and_redirect_target_drift(self) -> None:
        for response in (
            _Response(b"{}", status=503),
            _Response(b"{}", url="http://127.0.0.1:30000/redirected"),
        ):
            with self.subTest(status=response.status, url=response.geturl()):
                server = _server()
                opener = Mock(return_value=response)
                with (
                    patch.object(
                        runtime.ManagedServer,
                        "_require_live_owned_loopback_port",
                        return_value=_LIVE_GENERATION,
                    ),
                    patch.object(
                        runtime,
                        "_open_sm121_agent_runtime_server_info",
                        opener,
                    ),
                    patch.object(runtime, "_read_sm121_agent_runtime_startup_logs") as logs,
                    self.assertRaisesRegex(
                        runtime.RuntimeErrorWithContext,
                        "attestation is invalid",
                    ),
                ):
                    runtime.inspect_sm121_agent_admission_runtime_identity(server)
                logs.assert_not_called()

    def test_server_info_rejects_malformed_or_oversized_bytes(self) -> None:
        malformed = (
            b"\xff",
            b"[]",
            b'{"reasoning_parser":"qwen3","reasoning_parser":"drift"}',
            b'{"value":NaN}',
            (
                json.dumps(_server_info()).removesuffix("}")
                + ',"unrelated":1e999999}'
            ).encode("utf-8"),
        )
        for payload in malformed:
            with self.subTest(payload=payload[:12]):
                server = _server()
                opener = Mock(return_value=_Response(payload))
                with (
                    patch.object(
                        runtime.ManagedServer,
                        "_require_live_owned_loopback_port",
                        return_value=_LIVE_GENERATION,
                    ),
                    patch.object(
                        runtime,
                        "_open_sm121_agent_runtime_server_info",
                        opener,
                    ),
                    patch.object(runtime, "_read_sm121_agent_runtime_startup_logs") as logs,
                    self.assertRaises(runtime.RuntimeErrorWithContext),
                ):
                    runtime.inspect_sm121_agent_admission_runtime_identity(server)
                logs.assert_not_called()

        server = _server()
        opener = Mock(return_value=_Response(b"{}"))
        with (
            patch.object(
                runtime.ManagedServer,
                "_require_live_owned_loopback_port",
                return_value=_LIVE_GENERATION,
            ),
            patch.object(
                runtime,
                "_open_sm121_agent_runtime_server_info",
                opener,
            ),
            patch.object(runtime, "_SM121_AGENT_RUNTIME_SERVER_INFO_MAX_BYTES", 1),
            patch.object(runtime, "_read_sm121_agent_runtime_startup_logs") as logs,
            self.assertRaisesRegex(runtime.RuntimeErrorWithContext, "invalid"),
        ):
            runtime.inspect_sm121_agent_admission_runtime_identity(server)
        logs.assert_not_called()

    def test_parser_and_limits_are_authoritative_top_level_scalars(self) -> None:
        cases = (
            _server_info(
                reasoning_parser="wrong",
                nested={"reasoning_parser": "qwen3"},
            ),
            _server_info(
                reasoning_parser=None,
                nested={"reasoning_parser": "qwen3"},
            ),
            _server_info(
                disable_radix_cache=True,
                nested={"disable_radix_cache": False},
            ),
            _server_info(max_running_requests=True),
            _server_info(chunked_prefill_size=2048),
        )
        for payload in cases:
            with self.subTest(payload=payload):
                server = _server()
                opener = Mock(return_value=_Response(json.dumps(payload).encode("utf-8")))
                with (
                    patch.object(
                        runtime.ManagedServer,
                        "_require_live_owned_loopback_port",
                        return_value=_LIVE_GENERATION,
                    ),
                    patch.object(
                        runtime,
                        "_open_sm121_agent_runtime_server_info",
                        opener,
                    ),
                    patch.object(
                        runtime,
                        "_read_sm121_agent_runtime_startup_logs",
                        return_value=_STARTUP,
                    ),
                    self.assertRaises(runtime.RuntimeErrorWithContext),
                ):
                    runtime.inspect_sm121_agent_admission_runtime_identity(server)

        result, _response, _opener, _logs = self._inspect(
            json.dumps(
                _server_info(
                    server_args={
                        "disable_radix_cache": True,
                        "mamba_radix_cache_strategy": "wrong",
                        "max_mamba_cache_size": 1,
                    }
                )
            ).encode("utf-8")
        )
        self.assertEqual(result, SM121_AGENT_ADMISSION_RUNTIME_EXPECTED)

    def test_startup_identity_is_one_bounded_unambiguous_event(self) -> None:
        for logs in (
            b"",
            _STARTUP + _STARTUP,
            _STARTUP.replace(b"UnifiedRadixCache", b"ChunkCache"),
            _STARTUP.rstrip() + b" trailing-spoof\n",
        ):
            with self.subTest(logs=logs[:16]):
                server = _server()
                opener = Mock(return_value=_Response(json.dumps(_server_info()).encode()))
                log_reader = Mock(return_value=logs)
                with (
                    patch.object(
                        runtime.ManagedServer,
                        "_require_live_owned_loopback_port",
                        return_value=_LIVE_GENERATION,
                    ),
                    patch.object(
                        runtime,
                        "_open_sm121_agent_runtime_server_info",
                        opener,
                    ),
                    patch.object(
                        runtime,
                        "_read_sm121_agent_runtime_startup_logs",
                        log_reader,
                    ),
                    self.assertRaises(runtime.RuntimeErrorWithContext),
                ):
                    runtime.inspect_sm121_agent_admission_runtime_identity(server)
                self.assertEqual(log_reader.call_count, 1)

    def test_startup_identity_rejects_a_newline_flood_before_splitting(self) -> None:
        server = _server()
        opener = Mock(return_value=_Response(json.dumps(_server_info()).encode()))
        logs = _STARTUP + b"\n" * (runtime._SM121_AGENT_RUNTIME_LOG_MAX_LINES + 1)
        with (
            patch.object(
                runtime.ManagedServer,
                "_require_live_owned_loopback_port",
                return_value=_LIVE_GENERATION,
            ),
            patch.object(
                runtime,
                "_open_sm121_agent_runtime_server_info",
                opener,
            ),
            patch.object(
                runtime,
                "_read_sm121_agent_runtime_startup_logs",
                return_value=logs,
            ),
            self.assertRaisesRegex(
                runtime.RuntimeErrorWithContext,
                "startup attestation is invalid",
            ),
        ):
            runtime.inspect_sm121_agent_admission_runtime_identity(server)

    def test_http_failure_closes_without_exposing_response_text(self) -> None:
        response_body = io.BytesIO(b"RUNTIME-RESPONSE-MARKER")
        error = urllib.error.HTTPError(
            _SERVER_INFO_URL,
            503,
            "synthetic",
            None,
            response_body,
        )
        server = _server()
        with (
            patch.object(
                runtime.ManagedServer,
                "_require_live_owned_loopback_port",
                return_value=_LIVE_GENERATION,
            ),
            patch.object(
                runtime,
                "_open_sm121_agent_runtime_server_info",
                side_effect=error,
            ),
            patch.object(runtime, "_read_sm121_agent_runtime_startup_logs") as logs,
            self.assertRaises(runtime.RuntimeErrorWithContext) as caught,
        ):
            runtime.inspect_sm121_agent_admission_runtime_identity(server)
        self.assertNotIn("RUNTIME-RESPONSE-MARKER", str(caught.exception))
        self.assertTrue(response_body.closed)
        logs.assert_not_called()

    def test_truncated_http_body_fails_with_a_fixed_error(self) -> None:
        response = _Response(b"")
        response.read1 = Mock(
            side_effect=http.client.IncompleteRead(b"PARTIAL-MARKER", 32)
        )
        server = _server()
        with (
            patch.object(
                runtime.ManagedServer,
                "_require_live_owned_loopback_port",
                return_value=_LIVE_GENERATION,
            ),
            patch.object(
                runtime,
                "_open_sm121_agent_runtime_server_info",
                return_value=response,
            ),
            patch.object(runtime, "_read_sm121_agent_runtime_startup_logs") as logs,
            self.assertRaisesRegex(
                runtime.RuntimeErrorWithContext,
                "Could not read the SM121 agent runtime attestation",
            ) as caught,
        ):
            runtime.inspect_sm121_agent_admission_runtime_identity(server)
        self.assertNotIn("PARTIAL-MARKER", str(caught.exception))
        logs.assert_not_called()

    def test_snapshot_values_bind_the_request_and_postcheck(self) -> None:
        server = _server()
        original_authorization = server.authorization

        def open_and_mutate(
            request: urllib.request.Request, *, timeout_s: float
        ) -> _Response:
            self.assertEqual(timeout_s, 15.0)
            self.assertEqual(request.get_header("Authorization"), original_authorization)
            server.authorization = "Bearer " + "y" * 32
            return _Response(json.dumps(_server_info()).encode("utf-8"))

        with (
            patch.object(
                runtime.ManagedServer,
                "_require_live_owned_loopback_port",
                return_value=_LIVE_GENERATION,
            ) as live,
            patch.object(
                runtime,
                "_open_sm121_agent_runtime_server_info",
                side_effect=open_and_mutate,
            ),
            patch.object(
                runtime,
                "_read_sm121_agent_runtime_startup_logs",
                return_value=_STARTUP,
            ) as logs,
            self.assertRaisesRegex(
                runtime.RuntimeErrorWithContext,
                "not an owned server",
            ),
        ):
            runtime.inspect_sm121_agent_admission_runtime_identity(server)
        self.assertEqual(live.call_count, 1)
        logs.assert_called_once_with(
            "synthetic-container",
            started_at=_LIVE_GENERATION[0],
        )

    def test_live_owned_loopback_port_requires_one_running_exact_binding(self) -> None:
        server = _server()
        with (
            patch.object(
                runtime.ManagedServer,
                "_require_owned_container",
            ) as require_owned,
            patch.object(
                runtime,
                "_run",
                return_value=_completed(
                    stdout="true 2026-08-29T00:00:00.000000000Z 1234 "
                    "127.0.0.1:30000;\n"
                ),
            ) as run,
        ):
            generation = server._require_live_owned_loopback_port(
                host_port=30000,
                container_port=30000,
            )
        self.assertEqual(generation, _LIVE_GENERATION)
        require_owned.assert_called_once_with(server)
        self.assertEqual(
            run.call_args.args[0][:3],
            ["docker", "inspect", "--format"],
        )
        self.assertEqual(run.call_args.args[0][-1], "synthetic-container")
        self.assertIn('index .NetworkSettings.Ports "30000/tcp"', run.call_args.args[0][3])
        self.assertIn("$binding.HostIP", run.call_args.args[0][3])
        self.assertEqual(run.call_args.kwargs, {"check": False, "timeout": 20})

        for result in (
            _completed(
                stdout="false 2026-08-29T00:00:00.000000000Z 1234 "
                "127.0.0.1:30000;\n"
            ),
            _completed(
                stdout="true 2026-08-29T00:00:00.000000000Z 1234 "
                "0.0.0.0:30000;\n"
            ),
            _completed(
                stdout="true 2026-08-29T00:00:00.000000000Z 1234 "
                "127.0.0.1:30001;\n"
            ),
            _completed(
                stdout="true invalid-time 1234 127.0.0.1:30000;\n"
            ),
            _completed(
                stdout="true 2026-08-29T00:00:00.000000000Z 0 "
                "127.0.0.1:30000;\n"
            ),
            _completed(
                stdout="true 2026-08-29T00:00:00.000000000Z 1234 "
                "127.0.0.1:30000;127.0.0.1:30001;\n"
            ),
            _completed(
                stdout="true 2026-08-29T00:00:00.000000000Z 1234 "
                "127.0.0.1:30000;\n",
                stderr="unexpected",
            ),
            _completed(returncode=1),
        ):
            with self.subTest(result=result):
                with (
                    patch.object(runtime.ManagedServer, "_require_owned_container"),
                    patch.object(runtime, "_run", return_value=result),
                    self.assertRaisesRegex(
                        runtime.RuntimeErrorWithContext,
                        "not live on the required loopback port",
                    ),
                ):
                    server._require_live_owned_loopback_port(
                        host_port=30000,
                        container_port=30000,
                    )

    def test_live_owned_check_runs_before_and_after_attestation(self) -> None:
        server = _server()
        live = Mock(
            side_effect=(
                _LIVE_GENERATION,
                ("2026-08-29T00:00:01.000000000Z", 1235),
            )
        )
        with (
            patch.object(
                runtime.ManagedServer,
                "_require_live_owned_loopback_port",
                live,
            ),
            patch.object(
                runtime,
                "_open_sm121_agent_runtime_server_info",
                return_value=_Response(json.dumps(_server_info()).encode("utf-8")),
            ),
            patch.object(
                runtime,
                "_read_sm121_agent_runtime_startup_logs",
                return_value=_STARTUP,
            ) as logs,
            self.assertRaisesRegex(
                runtime.RuntimeErrorWithContext,
                "attestation",
            ),
        ):
            runtime.inspect_sm121_agent_admission_runtime_identity(server)
        self.assertEqual(
            live.call_args_list,
            [
                call(server, host_port=30000, container_port=30000),
                call(server, host_port=30000, container_port=30000),
            ],
        )
        logs.assert_called_once_with(
            "synthetic-container",
            started_at=_LIVE_GENERATION[0],
        )

    def test_response_body_enforces_one_total_deadline(self) -> None:
        response = _Response(b"first-chunk")
        with (
            patch.object(
                runtime.time,
                "monotonic",
                side_effect=(0.0, 0.0, 16.0),
            ),
            self.assertRaisesRegex(
                runtime.RuntimeErrorWithContext,
                "Could not read the SM121 agent runtime attestation",
            ) as caught,
        ):
            runtime._read_sm121_agent_runtime_response_body(response)
        self.assertEqual(
            response.read_limits,
            [runtime._SM121_AGENT_RUNTIME_RESPONSE_CHUNK_BYTES],
        )
        self.assertEqual(response.socket_timeouts, [15.0])
        self.assertNotIn("first-chunk", str(caught.exception))

    def test_response_body_accepts_content_length_final_read_closure(self) -> None:
        response = _Response(b"{}", content_length=True)

        payload = runtime._read_sm121_agent_runtime_response_body(response)

        self.assertEqual(payload, b"{}")
        self.assertIsNone(response.fp)
        self.assertEqual(
            response.read_limits,
            [runtime._SM121_AGENT_RUNTIME_RESPONSE_CHUNK_BYTES],
        )

    def test_native_metrics_reader_uses_only_fixed_direct_transport(self) -> None:
        metrics = _metrics()
        response = _Response(metrics.encode("utf-8"), url=_METRICS_URL)
        opener = Mock(return_value=response)
        with (
            patch.object(
                runtime,
                "_open_sm121_agent_runtime_server_info",
                opener,
            ),
            patch.object(runtime.urllib.request, "urlopen") as generic_urlopen,
        ):
            observed = runtime._read_sm121_agent_admission_metrics(
                "Bearer " + "x" * 32
            )

        self.assertEqual(observed, metrics)
        generic_urlopen.assert_not_called()
        opener.assert_called_once()
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, _METRICS_URL)
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.get_header("Authorization"), "Bearer " + "x" * 32)
        self.assertEqual(
            opener.call_args.kwargs,
            {"timeout_s": runtime._SM121_AGENT_RUNTIME_RESPONSE_TIMEOUT_S},
        )
        self.assertEqual(
            response.read_limits,
            [
                runtime._SM121_AGENT_RUNTIME_RESPONSE_CHUNK_BYTES,
                runtime._SM121_AGENT_RUNTIME_RESPONSE_CHUNK_BYTES,
            ],
        )
        self.assertTrue(response.socket_timeouts)

    def test_native_metrics_reader_rejects_transport_and_exposition_drift(self) -> None:
        malformed_responses = (
            _Response(b"PRIVATE-METRICS-MARKER", status=503, url=_METRICS_URL),
            _Response(
                b"PRIVATE-METRICS-MARKER",
                url="http://127.0.0.1:30000/redirected",
            ),
            _Response(b"\xff", url=_METRICS_URL),
            _Response(
                b"\n" * (runtime._SM121_AGENT_METRICS_MAX_LINES + 1),
                url=_METRICS_URL,
            ),
        )
        for response in malformed_responses:
            with self.subTest(status=response.status, url=response.geturl()):
                with (
                    patch.object(
                        runtime,
                        "_open_sm121_agent_runtime_server_info",
                        return_value=response,
                    ),
                    self.assertRaisesRegex(
                        runtime.RuntimeErrorWithContext,
                        "attestation is invalid",
                    ) as caught,
                ):
                    runtime._read_sm121_agent_admission_metrics(
                        "Bearer " + "x" * 32
                    )
                self.assertNotIn("PRIVATE-METRICS-MARKER", str(caught.exception))

        with (
            patch.object(
                runtime,
                "_open_sm121_agent_runtime_server_info",
                return_value=_Response(b"ab", url=_METRICS_URL),
            ),
            patch.object(runtime, "_SM121_AGENT_RUNTIME_SERVER_INFO_MAX_BYTES", 1),
            self.assertRaisesRegex(
                runtime.RuntimeErrorWithContext,
                "attestation is invalid",
            ),
        ):
            runtime._read_sm121_agent_admission_metrics("Bearer " + "x" * 32)

    def test_native_metrics_reader_caps_open_and_body_to_settle_deadline(self) -> None:
        response = _Response(b"metric 1\n", url=_METRICS_URL)
        opener = Mock(return_value=response)
        with (
            patch.object(
                runtime,
                "_open_sm121_agent_runtime_server_info",
                opener,
            ),
            patch.object(
                runtime.time,
                "monotonic",
                side_effect=(100.0, 100.0, 100.0, 101.1),
            ),
            self.assertRaisesRegex(
                runtime.RuntimeErrorWithContext,
                "attestation",
            ),
        ):
            runtime._read_sm121_agent_admission_metrics(
                "Bearer " + "x" * 32, deadline=101.0
            )
        self.assertEqual(opener.call_args.kwargs, {"timeout_s": 1.0})
        self.assertEqual(response.socket_timeouts, [1.0])

    def test_native_metrics_snapshot_binds_request_and_postcheck(self) -> None:
        server = _server()
        binding = _binding()
        deadline = runtime.time.monotonic() + 10.0
        reader = Mock(return_value=_metrics())
        with (
            patch.object(
                runtime,
                "_require_sm121_agent_admission_server",
                side_effect=((server, binding), (server, binding)),
            ) as require_owned,
            patch.object(
                runtime,
                "_read_sm121_agent_admission_metrics",
                reader,
            ),
            patch.object(runtime.urllib.request, "urlopen") as generic_urlopen,
        ):
            snapshot, lease_binding = runtime._snapshot_sm121_agent_admission_metrics(
                server, deadline=deadline
            )

        self.assertEqual(lease_binding, binding)
        self.assertEqual(
            require_owned.call_args_list,
            [call(server), call(server)],
        )
        reader.assert_called_once_with("Bearer " + "x" * 32, deadline=deadline)
        generic_urlopen.assert_not_called()
        self.assertEqual(set(snapshot), set(runtime._sm121_cache_metric_defaults()))
        self.assertTrue(snapshot["available"])
        self.assertTrue(snapshot["guardrail_metrics_available"])
        self.assertEqual(snapshot["prefill_input_tokens"], 17)
        self.assertEqual(snapshot["prefill_device_hit_tokens"], 0)
        self.assertEqual(snapshot["evicted_tokens"], 0)
        self.assertEqual(snapshot["retracted_requests"], 0)

        replacement = _binding(
            generation=("2026-08-29T00:00:01.000000000Z", 1235)
        )
        with (
            patch.object(
                runtime,
                "_require_sm121_agent_admission_server",
                side_effect=((server, binding), (server, replacement)),
            ),
            patch.object(
                runtime,
                "_read_sm121_agent_admission_metrics",
                return_value=_metrics(),
            ),
            self.assertRaisesRegex(
                runtime.RuntimeErrorWithContext,
                "metrics attestation is invalid",
            ),
        ):
            runtime._snapshot_sm121_agent_admission_metrics(
                server, deadline=runtime.time.monotonic() + 10.0
            )

    def test_native_metrics_settle_requires_same_opaque_generation_lease(self) -> None:
        server = _server()
        first = runtime._parse_sm121_cache_observability_metrics(
            _metrics(), semantic_arm=SM121_CACHE_SEMANTIC_CACHE_ON_ARM
        )
        binding = _binding()
        with (
            patch.object(
                runtime,
                "_snapshot_sm121_agent_admission_metrics",
                side_effect=((first, binding), (first, binding)),
            ) as snapshot,
            patch.object(runtime.time, "sleep") as sleep,
        ):
            observed, lease, polls, settled = runtime.settle_sm121_agent_admission_metrics(
                server,
                deadline=runtime.time.monotonic() + 10.0,
                poll_interval_s=0.1,
            )

        self.assertEqual(observed, first)
        self.assertEqual(polls, 2)
        self.assertTrue(settled)
        self.assertIs(type(lease), runtime._SM121AgentNativeCacheLease)
        self.assertNotIn("synthetic-container", repr(lease))
        self.assertNotIn("synthetic-run-identity", repr(lease))
        self.assertEqual(snapshot.call_count, 2)
        self.assertTrue(
            all(
                item.args == (server,)
                and isinstance(item.kwargs.get("deadline"), float)
                for item in snapshot.call_args_list
            )
        )
        sleep.assert_called_once()

        replacement = _binding(
            generation=("2026-08-29T00:00:01.000000000Z", 1235)
        )
        with (
            patch.object(
                runtime,
                "_snapshot_sm121_agent_admission_metrics",
                side_effect=((first, binding), (first, replacement)),
            ) as snapshot,
            patch.object(runtime.time, "sleep") as sleep,
            self.assertRaisesRegex(
                runtime.RuntimeErrorWithContext,
                "metrics attestation is invalid",
            ),
        ):
            runtime.settle_sm121_agent_admission_metrics(
                server,
                deadline=runtime.time.monotonic() + 10.0,
                poll_interval_s=0.1,
            )
        self.assertEqual(snapshot.call_count, 2)
        self.assertTrue(
            all(item.args == (server,) for item in snapshot.call_args_list)
        )
        sleep.assert_called_once()

        expected_lease = runtime._SM121AgentNativeCacheLease(binding)
        with (
            patch.object(
                runtime,
                "_snapshot_sm121_agent_admission_metrics",
                return_value=(first, replacement),
            ) as snapshot,
            self.assertRaisesRegex(
                runtime.RuntimeErrorWithContext,
                "metrics attestation is invalid",
            ),
        ):
            runtime.settle_sm121_agent_admission_metrics(
                server,
                deadline=runtime.time.monotonic() + 10.0,
                poll_interval_s=0.1,
                expected_lease=expected_lease,
            )
        snapshot.assert_called_once()
        self.assertEqual(snapshot.call_args.args, (server,))
        self.assertIn("deadline", snapshot.call_args.kwargs)

    def test_fixed_opener_disables_proxies_and_redirects(self) -> None:
        request = urllib.request.Request(_SERVER_INFO_URL)
        opener = Mock()
        opener.open.return_value = "synthetic-response"
        with patch.object(
            runtime.urllib.request,
            "build_opener",
            return_value=opener,
        ) as build_opener:
            response = runtime._open_sm121_agent_runtime_server_info(
                request,
                timeout_s=3.0,
            )
        self.assertEqual(response, "synthetic-response")
        handlers = build_opener.call_args.args
        proxy = next(
            handler
            for handler in handlers
            if isinstance(handler, urllib.request.ProxyHandler)
        )
        self.assertEqual(proxy.proxies, {})
        self.assertTrue(
            any(
                isinstance(handler, runtime._SM121AgentRuntimeNoRedirect)
                for handler in handlers
            )
        )
        opener.open.assert_called_once_with(request, timeout=3.0)

        with self.assertRaises(urllib.error.URLError):
            runtime._SM121AgentRuntimeNoRedirect().redirect_request()

    def test_startup_log_reader_fails_closed_at_its_byte_cap(self) -> None:
        stream = Mock()
        stream.fileno.return_value = 7
        process = Mock(stdout=stream, stderr=Mock())
        process.poll.return_value = None
        selector = Mock()
        selector.get_map.return_value = {7: object()}
        selector.select.return_value = [(SimpleNamespace(fileobj=stream), None)]

        with (
            patch.object(
                runtime.subprocess,
                "Popen",
                return_value=process,
            ) as popen,
            patch.object(runtime.selectors, "DefaultSelector", return_value=selector),
            patch.object(runtime.os, "read", return_value=b"too-large"),
            patch.object(runtime, "_SM121_AGENT_RUNTIME_LOG_MAX_BYTES", 1),
            self.assertRaisesRegex(
                runtime.RuntimeErrorWithContext, "invalid"
            ) as caught,
        ):
            runtime._read_sm121_agent_runtime_startup_logs(
                "synthetic-container",
                started_at=_LIVE_GENERATION[0],
            )

        self.assertEqual(
            popen.call_args.args[0],
            [
                "docker",
                "logs",
                "--since",
                _LIVE_GENERATION[0],
                "--timestamps",
                "--tail",
                "1024",
                "synthetic-container",
            ],
        )
        process.kill.assert_called_once_with()
        self.assertNotIn("too-large", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
