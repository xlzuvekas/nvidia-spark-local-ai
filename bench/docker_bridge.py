"""Authenticated HTTP bridges from a private endpoint to loopback.

The bridge exposes an otherwise loopback-only model server to a Docker bridge
network without moving the model listener off ``127.0.0.1``.  It authenticates
the first HTTP request on every accepted connection, then relays bytes in both
directions.  It deliberately has no request, header, or payload logging.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hmac
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import socket
import stat
import threading
import time
from types import FrameType
from typing import Callable


TARGET_HOST = "127.0.0.1"
MAX_API_KEY_BYTES = 16 * 1024
MAX_HEADER_BYTES = 64 * 1024
MAX_PENDING_BYTES = 1024 * 1024
RELAY_CHUNK_BYTES = 64 * 1024
MAX_CONNECT_TIMEOUT_S = 30.0
MAX_IDLE_TIMEOUT_S = 3600.0
ACCEPT_POLL_S = 0.25
RELAY_POLL_S = 0.25
LISTEN_BACKLOG = 32
MAX_WORKERS = 32

_PRIVATE_LISTEN_NETWORKS = tuple(
    ipaddress.IPv4Network(network)
    for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
_LINK_LOCAL_NETWORK = ipaddress.IPv4Network("169.254.0.0/16")
_HEADER_NAME = re.compile(rb"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_METHOD = _HEADER_NAME
_BEARER = re.compile(rb"(?i:Bearer)[ \t]+([^ \t]+)")

_RESPONSES = {
    400: (
        b"HTTP/1.1 400 Bad Request\r\n"
        b"Content-Length: 0\r\nConnection: close\r\n"
        b"Cache-Control: no-store\r\n\r\n"
    ),
    401: (
        b"HTTP/1.1 401 Unauthorized\r\n"
        b"Content-Length: 0\r\nConnection: close\r\n"
        b"Cache-Control: no-store\r\nWWW-Authenticate: Bearer\r\n\r\n"
    ),
    408: (
        b"HTTP/1.1 408 Request Timeout\r\n"
        b"Content-Length: 0\r\nConnection: close\r\n"
        b"Cache-Control: no-store\r\n\r\n"
    ),
    431: (
        b"HTTP/1.1 431 Request Header Fields Too Large\r\n"
        b"Content-Length: 0\r\nConnection: close\r\n"
        b"Cache-Control: no-store\r\n\r\n"
    ),
    502: (
        b"HTTP/1.1 502 Bad Gateway\r\n"
        b"Content-Length: 0\r\nConnection: close\r\n"
        b"Cache-Control: no-store\r\n\r\n"
    ),
}


class BridgeError(RuntimeError):
    """Raised when a bridge configuration or private credential is unsafe."""


@dataclass(frozen=True, slots=True)
class BridgeConfig:
    """Bounded network settings for one authenticated loopback bridge."""

    listen_host: str
    target_port: int
    listen_port: int = 0
    connect_timeout_s: float = 5.0
    idle_timeout_s: float = 300.0


@dataclass(frozen=True, slots=True)
class UnixBridgeConfig:
    """Bounded settings for one owner-private Unix-socket bridge."""

    socket_path: Path
    target_port: int
    connect_timeout_s: float = 5.0
    idle_timeout_s: float = 300.0


@dataclass(frozen=True, slots=True)
class BoundEndpoint:
    """The concrete private address selected by ``start``."""

    host: str
    port: int


@dataclass(frozen=True, slots=True)
class BoundUnixEndpoint:
    """The owner-private Unix socket selected by ``start``."""

    socket_path: Path


def _valid_port(value: object, *, allow_zero: bool, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BridgeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if not minimum <= value <= 65_535:
        qualifier = "zero or " if allow_zero else ""
        raise BridgeError(f"{name} must be {qualifier}between 1 and 65535")
    return value


def _valid_timeout(value: object, *, maximum: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BridgeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0 or result > maximum:
        raise BridgeError(f"{name} must be greater than zero and at most {maximum:g}")
    return result


def validate_bridge_config(
    config: BridgeConfig, *, _allow_test_loopback: bool = False
) -> None:
    """Reject any production listener outside RFC1918 or IPv4 link-local space.

    ``_allow_test_loopback`` exists only for hermetic unit fixtures.  It does
    not permit wildcard, public, reserved, or non-IPv4 listeners, and the CLI
    never enables it.
    """

    try:
        address = ipaddress.IPv4Address(config.listen_host)
    except ipaddress.AddressValueError as error:
        raise BridgeError("listen_host must be one explicit IPv4 address") from error
    if str(address) != config.listen_host:
        raise BridgeError("listen_host must use canonical dotted-decimal IPv4")
    allowed = any(address in network for network in _PRIVATE_LISTEN_NETWORKS)
    allowed = allowed or address in _LINK_LOCAL_NETWORK
    if _allow_test_loopback:
        allowed = allowed or address.is_loopback
    if not allowed or address.is_unspecified or address.is_multicast:
        raise BridgeError(
            "listen_host must be one RFC1918 or IPv4 link-local address"
        )
    if address.is_loopback and not _allow_test_loopback:
        raise BridgeError("production listen_host must not be loopback")
    _valid_port(config.listen_port, allow_zero=True, name="listen_port")
    _valid_port(config.target_port, allow_zero=False, name="target_port")
    _valid_timeout(
        config.connect_timeout_s,
        maximum=MAX_CONNECT_TIMEOUT_S,
        name="connect_timeout_s",
    )
    _valid_timeout(
        config.idle_timeout_s,
        maximum=MAX_IDLE_TIMEOUT_S,
        name="idle_timeout_s",
    )


def validate_unix_bridge_config(config: UnixBridgeConfig) -> Path:
    """Return an absolute socket path below one owner-private real directory."""

    path = config.socket_path
    if not isinstance(path, Path) or not path.is_absolute():
        raise BridgeError("socket_path must be one absolute path")
    if path.name in {"", ".", ".."}:
        raise BridgeError("socket_path must name one socket")
    encoded = os.fsencode(path)
    if len(encoded) >= 100:
        raise BridgeError("socket_path is too long for a Unix-domain socket")
    _valid_port(config.target_port, allow_zero=False, name="target_port")
    _valid_timeout(
        config.connect_timeout_s,
        maximum=MAX_CONNECT_TIMEOUT_S,
        name="connect_timeout_s",
    )
    _valid_timeout(
        config.idle_timeout_s,
        maximum=MAX_IDLE_TIMEOUT_S,
        name="idle_timeout_s",
    )

    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_only = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_only is None:
        raise BridgeError("Secure Unix socket traversal is unavailable")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow | directory_only
    components = path.parent.parts[1:]
    directory_descriptor: int | None = None
    try:
        directory_descriptor = os.open("/", flags)
        for component in components:
            next_descriptor = os.open(
                component,
                flags,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        metadata = os.fstat(directory_descriptor)
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise BridgeError(
                "socket parent must be owned by this user and owner-private"
            )
        try:
            os.stat(path.name, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise BridgeError("socket_path must not already exist")
    except BridgeError:
        raise
    except OSError as error:
        raise BridgeError("Could not securely validate Unix socket path") from error
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
    return path


def _validate_api_key(api_key: str) -> bytes:
    try:
        encoded = api_key.encode("ascii")
    except UnicodeEncodeError as error:
        raise BridgeError("API key must contain only visible ASCII") from error
    if (
        not encoded
        or len(encoded) > MAX_API_KEY_BYTES
        or any(byte < 0x21 or byte > 0x7E for byte in encoded)
    ):
        raise BridgeError("API key must contain only visible ASCII")
    return encoded


def read_api_key(path: Path) -> str:
    """Read one private, bounded token without following path symlinks."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_only = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_only is None:
        raise BridgeError("Secure API key file traversal is unavailable")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow
    absolute = Path(os.path.abspath(path))
    components = absolute.parts[1:]
    if not components:
        raise BridgeError("API key path must name one regular file")
    descriptor: int | None = None
    directory_descriptor: int | None = None
    try:
        directory_descriptor = os.open(
            "/", flags | directory_only
        )
        for component in components[:-1]:
            next_descriptor = os.open(
                component,
                flags | directory_only,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        descriptor = os.open(
            components[-1], flags, dir_fd=directory_descriptor
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise BridgeError("API key path must be a regular file")
        if metadata.st_nlink != 1:
            raise BridgeError("API key file must have exactly one hard link")
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise BridgeError("API key file must be owned by this user and mode 0600")
        if metadata.st_size <= 0 or metadata.st_size > MAX_API_KEY_BYTES:
            raise BridgeError("API key file has an invalid size")
        payload = bytearray()
        while len(payload) <= MAX_API_KEY_BYTES:
            chunk = os.read(
                descriptor,
                min(4096, MAX_API_KEY_BYTES + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > MAX_API_KEY_BYTES:
            raise BridgeError("API key file has an invalid size")
    except BridgeError:
        raise
    except OSError as error:
        raise BridgeError("Could not securely read API key file") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
    try:
        bearer = bytes(payload).decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise BridgeError("API key must contain only visible ASCII") from error
    _validate_api_key(bearer)
    return bearer


def _authorization_status(header: bytes, expected_key: bytes) -> int:
    """Return 200 only for one well-formed request carrying the exact bearer."""

    if not header.endswith(b"\r\n\r\n"):
        return 400
    lines = header[:-4].split(b"\r\n")
    if not lines:
        return 400
    request_parts = lines[0].split(b" ")
    if (
        len(request_parts) != 3
        or not _METHOD.fullmatch(request_parts[0])
        or not request_parts[1]
        or any(byte <= 0x20 or byte == 0x7F for byte in request_parts[1])
        or request_parts[2] not in {b"HTTP/1.0", b"HTTP/1.1"}
    ):
        return 400
    authorizations: list[bytes] = []
    for line in lines[1:]:
        if not line or line.startswith((b" ", b"\t")):
            return 400
        name, separator, value = line.partition(b":")
        if not separator or not _HEADER_NAME.fullmatch(name):
            return 400
        if any(byte < 0x20 and byte != 0x09 or byte == 0x7F for byte in value):
            return 400
        if name.lower() == b"authorization":
            authorizations.append(value.strip(b" \t"))
    if len(authorizations) != 1:
        return 401
    match = _BEARER.fullmatch(authorizations[0])
    if match is None or not hmac.compare_digest(match.group(1), expected_key):
        return 401
    return 200


def _sanitize_authenticated_header(header: bytes) -> bytes:
    """Remove bridge credentials and disable upstream connection reuse."""

    lines = header[:-4].split(b"\r\n")
    sanitized = [lines[0]]
    for line in lines[1:]:
        name, _, _ = line.partition(b":")
        if name.lower() in {b"authorization", b"connection"}:
            continue
        sanitized.append(line)
    sanitized.append(b"Connection: close")
    return b"\r\n".join(sanitized) + b"\r\n\r\n"


def _send_status(client: socket.socket, status: int) -> None:
    try:
        client.sendall(_RESPONSES[status])
    except OSError:
        pass


class AuthenticatedHttpBridge:
    """Own a private HTTP listener and its loopback relay connections."""

    def __init__(
        self,
        config: BridgeConfig,
        api_key: str,
        *,
        _allow_test_loopback: bool = False,
    ) -> None:
        validate_bridge_config(
            config, _allow_test_loopback=_allow_test_loopback
        )
        self.config = config
        self._api_key = _validate_api_key(api_key)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._listener: socket.socket | None = None
        self._owned_sockets: set[socket.socket] = set()
        self._workers: set[threading.Thread] = set()
        self._closed = False

    def start(self) -> BoundEndpoint:
        """Bind and listen, returning the selected private port."""

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.config.listen_host, self.config.listen_port))
            listener.listen(LISTEN_BACKLOG)
            listener.settimeout(ACCEPT_POLL_S)
            with self._lock:
                if self._closed or self._listener is not None:
                    raise BridgeError("bridge cannot be started more than once")
                self._listener = listener
            host, port = listener.getsockname()[:2]
            return BoundEndpoint(str(host), int(port))
        except BaseException:
            with self._lock:
                owned = self._listener is listener
                if owned:
                    self._listener = None
            listener.close()
            raise

    def _track_socket(self, connection: socket.socket) -> bool:
        with self._lock:
            if self._closed or self._stop_event.is_set():
                return False
            self._owned_sockets.add(connection)
            return True

    def _forget_socket(self, connection: socket.socket) -> None:
        with self._lock:
            self._owned_sockets.discard(connection)

    def _read_authenticated_request(
        self, client: socket.socket
    ) -> bytes | None:
        client.settimeout(self.config.idle_timeout_s)
        buffered = bytearray()
        while True:
            header_end = buffered.find(b"\r\n\r\n")
            if header_end >= 0:
                header_end += 4
                if header_end > MAX_HEADER_BYTES:
                    _send_status(client, 431)
                    return None
                status = _authorization_status(
                    bytes(buffered[:header_end]), self._api_key
                )
                if status != 200:
                    _send_status(client, status)
                    return None
                sanitized = _sanitize_authenticated_header(
                    bytes(buffered[:header_end])
                )
                return sanitized + bytes(buffered[header_end:])
            if len(buffered) > MAX_HEADER_BYTES:
                _send_status(client, 431)
                return None
            try:
                chunk = client.recv(RELAY_CHUNK_BYTES)
            except socket.timeout:
                _send_status(client, 408)
                return None
            if not chunk:
                return None
            buffered.extend(chunk)

    @staticmethod
    def _shutdown_write(connection: socket.socket) -> None:
        try:
            connection.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    def _relay(self, client: socket.socket, upstream: socket.socket) -> None:
        peers = {client: upstream, upstream: client}
        sockets = (client, upstream)
        buffers = {client: bytearray(), upstream: bytearray()}
        read_open = {client: True, upstream: True}
        write_closed = {client: False, upstream: False}
        registrations: dict[socket.socket, int] = {}
        last_activity = time.monotonic()

        client.setblocking(False)
        upstream.setblocking(False)
        with selectors.DefaultSelector() as selector:
            while not self._stop_event.is_set():
                if all(not read_open[item] for item in sockets) and all(
                    not buffers[item] for item in sockets
                ):
                    return
                for connection in sockets:
                    peer = peers[connection]
                    events = 0
                    if (
                        read_open[connection]
                        and len(buffers[peer]) < MAX_PENDING_BYTES
                    ):
                        events |= selectors.EVENT_READ
                    if buffers[connection] and not write_closed[connection]:
                        events |= selectors.EVENT_WRITE
                    current = registrations.get(connection, 0)
                    if events and not current:
                        selector.register(connection, events)
                        registrations[connection] = events
                    elif events and events != current:
                        selector.modify(connection, events)
                        registrations[connection] = events
                    elif not events and current:
                        selector.unregister(connection)
                        registrations.pop(connection, None)
                if not registrations:
                    return
                remaining = self.config.idle_timeout_s - (
                    time.monotonic() - last_activity
                )
                if remaining <= 0.0:
                    return
                ready = selector.select(min(RELAY_POLL_S, remaining))
                for key, mask in ready:
                    connection = key.fileobj
                    if not isinstance(connection, socket.socket):
                        return
                    peer = peers[connection]
                    if mask & selectors.EVENT_READ and read_open[connection]:
                        available = MAX_PENDING_BYTES - len(buffers[peer])
                        try:
                            chunk = connection.recv(
                                min(RELAY_CHUNK_BYTES, available)
                            )
                        except BlockingIOError:
                            chunk = None
                        if chunk:
                            buffers[peer].extend(chunk)
                            last_activity = time.monotonic()
                        elif chunk == b"":
                            read_open[connection] = False
                            if not buffers[peer] and not write_closed[peer]:
                                self._shutdown_write(peer)
                                write_closed[peer] = True
                    if mask & selectors.EVENT_WRITE and buffers[connection]:
                        try:
                            sent = connection.send(buffers[connection])
                        except BlockingIOError:
                            sent = 0
                        if sent > 0:
                            del buffers[connection][:sent]
                            last_activity = time.monotonic()
                            if (
                                not buffers[connection]
                                and not read_open[peer]
                                and not write_closed[connection]
                            ):
                                self._shutdown_write(connection)
                                write_closed[connection] = True

    def _handle_client(self, client: socket.socket) -> None:
        upstream: socket.socket | None = None
        try:
            initial = self._read_authenticated_request(client)
            if initial is None or self._stop_event.is_set():
                return
            try:
                upstream = socket.create_connection(
                    (TARGET_HOST, self.config.target_port),
                    timeout=self.config.connect_timeout_s,
                )
            except OSError:
                _send_status(client, 502)
                return
            if not self._track_socket(upstream):
                return
            upstream.settimeout(self.config.idle_timeout_s)
            try:
                upstream.sendall(initial)
            except OSError:
                _send_status(client, 502)
                return
            self._relay(client, upstream)
        except (OSError, ValueError):
            return
        finally:
            for connection in (upstream, client):
                if connection is None:
                    continue
                self._forget_socket(connection)
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()

    def _worker_entry(self, client: socket.socket) -> None:
        try:
            self._handle_client(client)
        finally:
            with self._lock:
                self._workers.discard(threading.current_thread())

    def serve_forever(self) -> None:
        """Accept connections until ``request_stop`` or a termination signal."""

        with self._lock:
            listener = self._listener
        if listener is None:
            raise BridgeError("bridge must be started before serving")
        try:
            while not self._stop_event.is_set():
                try:
                    client, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop_event.is_set():
                        return
                    raise
                if not self._track_socket(client):
                    client.close()
                    return
                worker = threading.Thread(
                    target=self._worker_entry,
                    args=(client,),
                    name="sparkbench-http-bridge",
                    daemon=True,
                )
                with self._lock:
                    if len(self._workers) >= MAX_WORKERS:
                        self._owned_sockets.discard(client)
                        accepted = False
                    else:
                        self._workers.add(worker)
                        accepted = True
                if not accepted:
                    client.close()
                    continue
                worker.start()
        finally:
            self.request_stop()

    def request_stop(self) -> None:
        """Signal shutdown and close only sockets owned by this bridge."""

        self._stop_event.set()
        with self._lock:
            listener = self._listener
            self._listener = None
            connections = tuple(self._owned_sockets)
        if listener is not None:
            listener.close()
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def signal_stop(self) -> None:
        """Request signal-safe shutdown without acquiring the bridge lock."""

        self._stop_event.set()

    def close(self) -> None:
        """Idempotently stop the listener and reap its bounded worker set."""

        with self._lock:
            self._closed = True
        self.request_stop()
        deadline = time.monotonic() + self.config.connect_timeout_s + 1.0
        current = threading.current_thread()
        with self._lock:
            workers = tuple(self._workers)
        for worker in workers:
            if worker is current:
                continue
            worker.join(max(0.0, deadline - time.monotonic()))


class AuthenticatedUnixHttpBridge(AuthenticatedHttpBridge):
    """Own an authenticated Unix listener forwarding only to host loopback."""

    def __init__(self, config: UnixBridgeConfig, api_key: str) -> None:
        validate_unix_bridge_config(config)
        self.config = config  # type: ignore[assignment]
        self._api_key = _validate_api_key(api_key)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._listener: socket.socket | None = None
        self._owned_sockets: set[socket.socket] = set()
        self._workers: set[threading.Thread] = set()
        self._closed = False
        self._socket_identity: tuple[int, int] | None = None

    def start(self) -> BoundUnixEndpoint:  # type: ignore[override]
        """Bind one mode-0600 Unix socket below its validated private parent."""

        socket_path = validate_unix_bridge_config(self.config)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound = False
        try:
            listener.bind(os.fspath(socket_path))
            bound = True
            metadata = os.lstat(socket_path)
            with self._lock:
                self._socket_identity = (metadata.st_dev, metadata.st_ino)
            os.chmod(socket_path, 0o600, follow_symlinks=False)
            metadata = os.lstat(socket_path)
            if (
                not stat.S_ISSOCK(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise BridgeError("Unix listener did not retain owner-only socket mode")
            listener.listen(LISTEN_BACKLOG)
            listener.settimeout(ACCEPT_POLL_S)
            with self._lock:
                if self._closed or self._listener is not None:
                    raise BridgeError("bridge cannot be started more than once")
                self._listener = listener
            return BoundUnixEndpoint(socket_path)
        except BaseException:
            with self._lock:
                owned = self._listener is listener
                if owned:
                    self._listener = None
            listener.close()
            if bound:
                self._remove_owned_socket_path(socket_path)
            raise

    def _remove_owned_socket_path(self, socket_path: Path | None = None) -> None:
        path = socket_path or self.config.socket_path
        with self._lock:
            expected = self._socket_identity
            self._socket_identity = None
        if expected is None:
            return
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(metadata.st_mode) and (metadata.st_dev, metadata.st_ino) == expected:
            path.unlink()

    def request_stop(self) -> None:
        """Stop owned sockets and remove only this bridge's exact Unix socket."""

        super().request_stop()
        self._remove_owned_socket_path()


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone authenticated bridge CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Authenticate HTTP on one private IPv4 listener and forward it "
            "to a model server fixed on 127.0.0.1."
        )
    )
    parser.add_argument("--listen-host", required=True)
    parser.add_argument("--listen-port", type=int, default=0)
    parser.add_argument("--target-port", required=True, type=int)
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--connect-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--idle-timeout-seconds", type=float, default=300.0)
    return parser


def _install_signal_handlers(
    stop: Callable[[], None],
) -> dict[signal.Signals, signal.Handlers]:
    previous: dict[signal.Signals, signal.Handlers] = {}

    def handle_signal(_signum: int, _frame: FrameType | None) -> None:
        stop()

    for name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is not None:
            previous[signum] = signal.signal(signum, handle_signal)
    return previous


def main(argv: list[str] | None = None) -> int:
    """Run the authenticated bridge until interrupted."""

    parser = build_parser()
    args = parser.parse_args(argv)
    config = BridgeConfig(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        target_port=args.target_port,
        connect_timeout_s=args.connect_timeout_seconds,
        idle_timeout_s=args.idle_timeout_seconds,
    )
    try:
        validate_bridge_config(config)
        bearer = read_api_key(args.api_key_file)
        bridge = AuthenticatedHttpBridge(config, bearer)
        endpoint = bridge.start()
    except (BridgeError, OSError) as error:
        raise SystemExit(f"Could not start authenticated bridge: {error}") from error
    print(
        json.dumps(
            {
                "event": "ready",
                "listen_host": endpoint.host,
                "listen_port": endpoint.port,
                "target_host": TARGET_HOST,
                "target_port": config.target_port,
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    previous_handlers = _install_signal_handlers(bridge.signal_stop)
    try:
        bridge.serve_forever()
    finally:
        bridge.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
