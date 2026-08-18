from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
RELAY_ASSET = REPOSITORY / "bench" / "assets" / "harbor_uds_relay.js"
POLICY_ASSET = REPOSITORY / "bench" / "assets" / "harbor_no_network_policy.sh"
PLACEHOLDER = "sparkbench-relay-placeholder-v1"


def _unused_loopback_port() -> int:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])
    finally:
        listener.close()


def _recv_all(connection: socket.socket) -> bytes:
    blocks: list[bytes] = []
    while True:
        block = connection.recv(65536)
        if not block:
            return b"".join(blocks)
        blocks.append(block)


class HarborRelayAssetTests(unittest.TestCase):
    def test_policy_transitions_are_atomic_and_phase_scoped(self) -> None:
        policy = POLICY_ASSET.read_text()
        self.assertNotIn("delete table", policy)
        self.assertIn("policy drop", policy)
        self.assertIn("flush chain inet $NFTABLES_RULESET_NAME output", policy)
        self.assertIn('RELAY_SENTINEL=sparkbench-relay.invalid', policy)
        self.assertIn('[ "$#" -ne 2 ] || [ "$2" != "$RELAY_SENTINEL" ]', policy)
        self.assertIn(
            'ip daddr 127.0.0.1 tcp dport $RELAY_PORT accept', policy
        )
        self.assertIn(
            'ip daddr 127.0.0.1 tcp sport $RELAY_PORT ct state established accept',
            policy,
        )
        allow_all = policy.split("  allow-all)", 1)[1].split("    ;;", 1)[0]
        self.assertIn("exit 2", allow_all)
        self.assertNotIn("nft ", allow_all)
        deny_all = policy.split("install_deny_all()", 1)[1].split(
            "install_agent_relay_only()", 1
        )[0]
        self.assertNotIn("dport", deny_all)
        self.assertNotIn("ip6", policy)

    def test_relay_streams_half_close_and_bounds_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.chmod(0o700)
            socket_path = root / "model.sock"
            key_path = root / "internal-api-key"
            internal_key = "internal-" + "k" * 56
            key_path.write_text(internal_key)
            key_path.chmod(0o600)
            port = _unused_loopback_port()

            source = RELAY_ASSET.read_text()
            replacements = {
                'const LISTEN_PORT = 18080;': f"const LISTEN_PORT = {port};",
                'const SOCKET_PATH = "/run/sparkbench/model.sock";': (
                    f"const SOCKET_PATH = {json.dumps(str(socket_path))};"
                ),
                'const KEY_PATH = "/run/sparkbench/internal-api-key";': (
                    f"const KEY_PATH = {json.dumps(str(key_path))};"
                ),
            }
            for old, new in replacements.items():
                self.assertEqual(source.count(old), 1)
                source = source.replace(old, new)
            rendered = root / "relay.js"
            rendered.write_text(source)

            captured_request: list[bytes] = []
            upstream_failure: list[BaseException] = []
            upstream_ready = threading.Event()
            upstream_done = threading.Event()

            def serve_upstream() -> None:
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    listener.bind(str(socket_path))
                    listener.listen(1)
                    listener.settimeout(10)
                    upstream_ready.set()
                    connection, _ = listener.accept()
                    with connection:
                        connection.settimeout(10)
                        captured_request.append(_recv_all(connection))
                        connection.sendall(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n"
                            b"Connection: close\r\n\r\nhello "
                        )
                        time.sleep(0.02)
                        connection.sendall(b"world")
                except BaseException as error:
                    upstream_failure.append(error)
                finally:
                    listener.close()
                    upstream_done.set()

            upstream_thread = threading.Thread(target=serve_upstream, daemon=True)
            upstream_thread.start()
            self.assertTrue(upstream_ready.wait(2))
            process = subprocess.Popen(
                ["node", str(rendered)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            held: list[socket.socket] = []
            try:
                wrong: socket.socket | None = None
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    candidate.settimeout(0.2)
                    try:
                        candidate.connect(("127.0.0.1", port))
                    except OSError:
                        candidate.close()
                        time.sleep(0.02)
                    else:
                        wrong = candidate
                        break
                self.assertIsNotNone(wrong)
                assert wrong is not None
                with wrong:
                    wrong.settimeout(3)
                    wrong.sendall(
                        b"GET /v1/models HTTP/1.1\r\nHost: relay\r\n"
                        b"Authorization: Bearer wrong\r\n\r\n"
                    )
                    rejected = _recv_all(wrong)
                self.assertIn(b"401 Unauthorized", rejected)

                with socket.create_connection(("127.0.0.1", port), timeout=3) as client:
                    client.settimeout(10)
                    client.sendall(
                        b"POST /v1/chat/completions HTTP/1.1\r\nHost: relay\r\n"
                        + f"Authorization: Bearer {PLACEHOLDER}\r\n".encode()
                    )
                    time.sleep(0.02)
                    client.sendall(b"Content-Length: 2\r\n\r\n{}")
                    client.shutdown(socket.SHUT_WR)
                    response = _recv_all(client)
                self.assertTrue(upstream_done.wait(3))
                upstream_thread.join(1)
                self.assertFalse(upstream_failure)
                self.assertEqual(len(captured_request), 1)
                request = captured_request[0]
                self.assertIn(f"Authorization: Bearer {internal_key}".encode(), request)
                self.assertNotIn(PLACEHOLDER.encode(), request)
                self.assertIn(b"Connection: close", request)
                self.assertTrue(request.endswith(b"{}"))
                self.assertIn(b"hello world", response)
                self.assertNotIn(internal_key.encode(), response)

                for _ in range(16):
                    connection = socket.create_connection(
                        ("127.0.0.1", port), timeout=3
                    )
                    connection.settimeout(3)
                    connection.sendall(b"GET /held HTTP/1.1\r\n")
                    held.append(connection)
                excess = socket.create_connection(("127.0.0.1", port), timeout=3)
                with excess:
                    excess.settimeout(3)
                    over_cap = _recv_all(excess)
                self.assertIn(b"503 Service Unavailable", over_cap)
            finally:
                for connection in held:
                    connection.close()
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(stdout, b"")
            self.assertEqual(stderr, b"")
            self.assertNotIn(internal_key.encode(), stdout + stderr)


if __name__ == "__main__":
    unittest.main()
