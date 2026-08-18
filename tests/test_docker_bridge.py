from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import socket
import socketserver
import tempfile
import threading
import unittest

from bench.docker_bridge import (
    MAX_CONNECT_TIMEOUT_S,
    MAX_IDLE_TIMEOUT_S,
    MAX_WORKERS,
    TARGET_HOST,
    AuthenticatedHttpBridge,
    AuthenticatedUnixHttpBridge,
    BridgeConfig,
    BridgeError,
    UnixBridgeConfig,
    build_parser,
    read_api_key,
    validate_bridge_config,
    validate_unix_bridge_config,
)


FIXTURE_BEARER = "synthetic-test-bearer"


def _read_until(connection: socket.socket, marker: bytes) -> bytes:
    buffered = bytearray()
    while marker not in buffered:
        chunk = connection.recv(4096)
        if not chunk:
            break
        buffered.extend(chunk)
    return bytes(buffered)


def _read_all(connection: socket.socket) -> bytes:
    buffered = bytearray()
    while True:
        chunk = connection.recv(4096)
        if not chunk:
            return bytes(buffered)
        buffered.extend(chunk)


class _ThreadedServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _CountingHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        if not isinstance(server, _ThreadedServer):
            return
        server.accepted.set()  # type: ignore[attr-defined]
        _read_until(self.request, b"\r\n\r\n")
        self.request.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
            b"Connection: close\r\n\r\nok"
        )


class _StreamingHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        if not isinstance(server, _ThreadedServer):
            return
        initial = _read_until(self.request, b"\r\n\r\n")
        header, _, body = initial.partition(b"\r\n\r\n")
        server.received_header = header  # type: ignore[attr-defined]
        while len(body) < 6:
            body += self.request.recv(4096)
        self.request.sendall(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n4\r\npong\r\n"
        )
        while len(body) < 10:
            body += self.request.recv(4096)
        server.received_body = body[:10]  # type: ignore[attr-defined]
        self.request.sendall(b"4\r\ndone\r\n0\r\n\r\n")


class DockerBridgeTests(unittest.TestCase):
    def _start_target(
        self, handler: type[socketserver.BaseRequestHandler]
    ) -> tuple[_ThreadedServer, threading.Thread]:
        target = _ThreadedServer((TARGET_HOST, 0), handler)
        target.accepted = threading.Event()  # type: ignore[attr-defined]
        thread = threading.Thread(target=target.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2.0)
        self.addCleanup(target.server_close)
        self.addCleanup(target.shutdown)
        return target, thread

    def _start_bridge(
        self, target_port: int
    ) -> tuple[AuthenticatedHttpBridge, threading.Thread, int]:
        bridge = AuthenticatedHttpBridge(
            BridgeConfig(
                listen_host="127.0.0.1",
                listen_port=0,
                target_port=target_port,
                connect_timeout_s=1.0,
                idle_timeout_s=2.0,
            ),
            FIXTURE_BEARER,
            _allow_test_loopback=True,
        )
        endpoint = bridge.start()
        thread = threading.Thread(target=bridge.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2.0)
        self.addCleanup(bridge.close)
        return bridge, thread, endpoint.port

    def test_production_listener_accepts_only_explicit_private_ipv4(self) -> None:
        for host in (
            "10.0.0.1",
            "172.16.0.1",
            "172.31.255.254",
            "192.168.1.1",
            "169.254.10.2",
        ):
            with self.subTest(host=host):
                validate_bridge_config(BridgeConfig(host, target_port=8000))
        for host in (
            "0.0.0.0",
            "127.0.0.1",
            "8.8.8.8",
            "172.15.0.1",
            "100.64.0.1",
            "::1",
            "localhost",
            "192.168.1.0/24",
        ):
            with self.subTest(host=host):
                with self.assertRaises(BridgeError):
                    validate_bridge_config(BridgeConfig(host, target_port=8000))
        validate_bridge_config(
            BridgeConfig("127.0.0.1", target_port=8000),
            _allow_test_loopback=True,
        )

    def test_ports_timeouts_and_cli_surface_are_bounded(self) -> None:
        invalid = (
            BridgeConfig("10.0.0.1", target_port=0),
            BridgeConfig("10.0.0.1", target_port=65_536),
            BridgeConfig("10.0.0.1", target_port=8000, listen_port=-1),
            BridgeConfig("10.0.0.1", target_port=8000, listen_port=65_536),
            BridgeConfig("10.0.0.1", target_port=8000, connect_timeout_s=0),
            BridgeConfig(
                "10.0.0.1",
                target_port=8000,
                connect_timeout_s=MAX_CONNECT_TIMEOUT_S + 1,
            ),
            BridgeConfig("10.0.0.1", target_port=8000, idle_timeout_s=0),
            BridgeConfig(
                "10.0.0.1",
                target_port=8000,
                idle_timeout_s=MAX_IDLE_TIMEOUT_S + 1,
            ),
        )
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(BridgeError):
                    validate_bridge_config(config)

        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("--api-key-file", help_text)
        self.assertIn("--target-port", help_text)
        self.assertNotIn("--target-host", help_text)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "--listen-host",
                        "172.17.0.1",
                        "--target-port",
                        "8000",
                    ]
                )

    def test_signal_stop_never_reenters_the_bridge_lock(self) -> None:
        bridge = AuthenticatedHttpBridge(
            BridgeConfig(
                listen_host="127.0.0.1",
                target_port=8000,
            ),
            FIXTURE_BEARER,
            _allow_test_loopback=True,
        )
        with bridge._lock:
            bridge.signal_stop()
        self.assertTrue(bridge._stop_event.is_set())

    def test_api_key_file_is_private_bounded_and_never_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key_path = root / "bridge.key"
            key_path.write_text(FIXTURE_BEARER + "\n", encoding="ascii")
            os.chmod(key_path, 0o600)
            self.assertEqual(read_api_key(key_path), FIXTURE_BEARER)

            os.chmod(key_path, 0o640)
            with self.assertRaisesRegex(BridgeError, "mode 0600"):
                read_api_key(key_path)
            os.chmod(key_path, 0o600)

            link = root / "linked.key"
            link.symlink_to(key_path)
            with self.assertRaises(BridgeError):
                read_api_key(link)

            hardlink = root / "hardlinked.key"
            os.link(key_path, hardlink)
            with self.assertRaisesRegex(BridgeError, "exactly one hard link"):
                read_api_key(key_path)
            hardlink.unlink()

            real_parent = root / "real-parent"
            real_parent.mkdir()
            nested_key = real_parent / "nested.key"
            nested_key.write_text(FIXTURE_BEARER + "\n", encoding="ascii")
            os.chmod(nested_key, 0o600)
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaises(BridgeError):
                read_api_key(linked_parent / nested_key.name)

    def test_unix_listener_requires_private_real_parent_and_cleans_up(self) -> None:
        target, _ = self._start_target(_CountingHandler)
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "private"
            parent.mkdir(mode=0o700)
            socket_path = parent / "model.sock"
            config = UnixBridgeConfig(
                socket_path=socket_path,
                target_port=target.server_address[1],
                connect_timeout_s=1.0,
                idle_timeout_s=2.0,
            )
            self.assertEqual(validate_unix_bridge_config(config), socket_path)
            bridge = AuthenticatedUnixHttpBridge(config, FIXTURE_BEARER)
            endpoint = bridge.start()
            thread = threading.Thread(target=bridge.serve_forever, daemon=True)
            thread.start()
            try:
                self.assertEqual(endpoint.socket_path, socket_path)
                self.assertEqual(os.stat(socket_path).st_mode & 0o777, 0o600)
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(1.0)
                    client.connect(os.fspath(socket_path))
                    client.sendall(
                        b"GET /v1/models HTTP/1.1\r\nHost: fixture\r\n"
                        b"Authorization: Bearer "
                        + FIXTURE_BEARER.encode()
                        + b"\r\nConnection: close\r\n\r\n"
                    )
                    response = _read_all(client)
                self.assertIn(b"\r\n\r\nok", response)
            finally:
                bridge.close()
                thread.join(2.0)
            self.assertFalse(socket_path.exists())

            os.chmod(parent, 0o750)
            with self.assertRaisesRegex(BridgeError, "owner-private"):
                validate_unix_bridge_config(config)
            os.chmod(parent, 0o700)
            socket_path.write_text("occupied", encoding="ascii")
            with self.assertRaisesRegex(BridgeError, "must not already exist"):
                validate_unix_bridge_config(config)

            real_parent = Path(directory) / "real-socket-parent"
            real_parent.mkdir(mode=0o700)
            linked_parent = Path(directory) / "linked-socket-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(BridgeError, "securely validate"):
                validate_unix_bridge_config(
                    UnixBridgeConfig(
                        socket_path=linked_parent / "model.sock",
                        target_port=target.server_address[1],
                    )
                )

    def test_missing_and_wrong_bearers_never_reach_loopback_target(self) -> None:
        target, _ = self._start_target(_CountingHandler)
        _, _, bridge_port = self._start_bridge(target.server_address[1])
        for auth_header in (b"", b"Authorization: Bearer wrong\r\n"):
            with self.subTest(has_header=bool(auth_header)):
                with socket.create_connection(
                    ("127.0.0.1", bridge_port), timeout=1.0
                ) as client:
                    client.settimeout(1.0)
                    client.sendall(
                        b"GET /v1/models HTTP/1.1\r\nHost: fixture\r\n"
                        + auth_header
                        + b"Connection: close\r\n\r\n"
                    )
                    response = _read_all(client)
                self.assertTrue(response.startswith(b"HTTP/1.1 401 "))
        self.assertFalse(target.accepted.wait(0.1))  # type: ignore[attr-defined]

        with socket.create_connection(
            ("127.0.0.1", bridge_port), timeout=1.0
        ) as client:
            client.settimeout(1.0)
            client.sendall(
                b"GET /v1/models HTTP/1.1\r\nHost: fixture\r\n"
                b"Authorization: Bearer "
                + FIXTURE_BEARER.encode()
                + b"\r\nConnection: close\r\n\r\n"
            )
            response = _read_all(client)
        self.assertIn(b"\r\n\r\nok", response)
        self.assertTrue(target.accepted.wait(1.0))  # type: ignore[attr-defined]

    def test_partial_unauthenticated_connections_have_a_hard_worker_cap(self) -> None:
        target, _ = self._start_target(_CountingHandler)
        bridge, _, bridge_port = self._start_bridge(target.server_address[1])
        clients: list[socket.socket] = []
        self.addCleanup(lambda: [client.close() for client in clients])
        for _ in range(MAX_WORKERS):
            client = socket.create_connection(
                ("127.0.0.1", bridge_port), timeout=1.0
            )
            client.sendall(b"GET /v1/models HTTP/1.1\r\n")
            clients.append(client)
        deadline = threading.Event()
        for _ in range(100):
            with bridge._lock:
                worker_count = len(bridge._workers)
            if worker_count == MAX_WORKERS:
                break
            deadline.wait(0.01)
        self.assertEqual(worker_count, MAX_WORKERS)

        excess = socket.create_connection(
            ("127.0.0.1", bridge_port), timeout=1.0
        )
        clients.append(excess)
        excess.settimeout(1.0)
        excess.sendall(b"GET /v1/models HTTP/1.1\r\n")
        try:
            closed = excess.recv(1) == b""
        except ConnectionResetError:
            closed = True
        self.assertTrue(closed)
        self.assertFalse(target.accepted.wait(0.1))  # type: ignore[attr-defined]

    def test_authenticated_request_and_response_bodies_stream_both_ways(
        self,
    ) -> None:
        target, _ = self._start_target(_StreamingHandler)
        bridge, bridge_thread, bridge_port = self._start_bridge(
            target.server_address[1]
        )
        with socket.create_connection(
            ("127.0.0.1", bridge_port), timeout=1.0
        ) as client:
            client.settimeout(1.0)
            client.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: fixture\r\nContent-Length: 10\r\n"
                b"Authorization: Bearer "
                + FIXTURE_BEARER.encode()
                + b"\r\nConnection: close\r\n\r\nfirst-"
            )
            first_response = _read_until(client, b"pong\r\n")
            self.assertIn(b"pong", first_response)
            client.sendall(b"last")
            client.shutdown(socket.SHUT_WR)
            final_response = _read_all(client)
        self.assertIn(b"done", final_response)
        self.assertEqual(target.received_body, b"first-last")  # type: ignore[attr-defined]
        received_header = target.received_header.lower()  # type: ignore[attr-defined]
        self.assertNotIn(b"authorization:", received_header)
        self.assertIn(b"connection: close", received_header)
        self.assertNotIn(FIXTURE_BEARER.encode(), received_header)

        bridge.close()
        bridge_thread.join(2.0)
        self.assertFalse(bridge_thread.is_alive())
