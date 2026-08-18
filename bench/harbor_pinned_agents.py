"""Offline Harbor agents backed by immutable read-only tool prefixes.

Harbor imports these classes only inside its pinned virtual environment.  The
outer campaign lifecycle verifies the complete mounted trees before launch;
these subclasses deliberately replace the stock network installers with a
small PATH shim and exact runtime/version admission.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
from typing import override

from harbor.agents.installed.opencode import OpenCode
from harbor.agents.installed.qwen_code import QwenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


NODE_PREFIX = "/opt/sparkbench/node"
AGENT_PREFIX = "/opt/sparkbench/agent"
NODE_VERSION = "v22.22.1"
QWEN_CODE_VERSION = "0.21.13"
OPENCODE_VERSION = "1.18.18"
NETWORK_ADMISSION_FILENAME = "sparkbench-network-admission.json"


_NETWORK_PROBE_JS = r"""
"use strict";
const dns = require("dns");
const fs = require("fs");
const http = require("http");
const net = require("net");
const mode = process.argv[1];
const TIMEOUT_MS = 1200;
const RELAY_PORT = 18080;
const PLACEHOLDER = "sparkbench-relay-placeholder-v1";
const MODEL = "Qwen/Qwen3-Coder-Next";

function tcpOpen(host, port) {
  return new Promise((resolve) => {
    let settled = false;
    const socket = net.createConnection({host, port});
    const finish = (value) => {
      if (settled) return;
      settled = true;
      socket.destroy();
      resolve(value);
    };
    socket.setTimeout(TIMEOUT_MS, () => finish(false));
    socket.once("connect", () => finish(true));
    socket.once("error", () => finish(false));
  });
}

function httpResult(bearer) {
  return new Promise((resolve) => {
    let settled = false;
    let body = Buffer.alloc(0);
    const finish = (value) => {
      if (settled) return;
      settled = true;
      request.destroy();
      resolve(value);
    };
    const request = http.request({
      host: "127.0.0.1",
      port: RELAY_PORT,
      path: "/v1/models",
      method: "GET",
      agent: false,
      headers: {Authorization: `Bearer ${bearer}`, Connection: "close"},
    }, (response) => {
      response.on("data", (chunk) => {
        if (body.length + chunk.length > 65536) return finish(null);
        body = Buffer.concat([body, chunk], body.length + chunk.length);
      });
      response.once("end", () => finish({status: response.statusCode, body}));
      response.once("error", () => finish(null));
    });
    request.setTimeout(TIMEOUT_MS, () => finish(null));
    request.once("error", () => finish(null));
    request.end();
  });
}

function dnsBlocked() {
  return new Promise((resolve) => {
    const resolver = new dns.Resolver();
    resolver.setServers(["127.0.0.11"]);
    let settled = false;
    const finish = (blocked) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolver.cancel();
      resolve(blocked);
    };
    const timer = setTimeout(() => finish(true), TIMEOUT_MS);
    resolver.resolve4("example.com", (error) => finish(Boolean(error)));
  });
}

function defaultGateway() {
  try {
    for (const line of fs.readFileSync("/proc/net/route", "ascii").trim().split("\n").slice(1)) {
      const fields = line.trim().split(/\s+/);
      if (fields[1] !== "00000000" || fields[2].length !== 8) continue;
      return [6, 4, 2, 0].map((offset) => parseInt(fields[2].slice(offset, offset + 2), 16)).join(".");
    }
  } catch (_) {}
  return null;
}

function capabilitiesDropped() {
  try {
    const status = fs.readFileSync("/proc/self/status", "ascii");
    for (const name of ["CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"]) {
      const match = status.match(new RegExp(`^${name}:\\s*([0-9a-fA-F]+)$`, "m"));
      if (!match || !/^0+$/.test(match[1])) return false;
    }
    return /^NoNewPrivs:\s*1$/m.test(status);
  } catch (_) {
    return false;
  }
}

async function commonNegativeChecks() {
  const gateway = defaultGateway();
  const [other, gost, dnsDenied, gatewayOpen, publicOpen] = await Promise.all([
    tcpOpen("127.0.0.1", RELAY_PORT + 1),
    tcpOpen("127.0.0.1", 12345),
    dnsBlocked(),
    gateway === null ? Promise.resolve(false) : tcpOpen(gateway, 22),
    tcpOpen("1.1.1.1", 443),
  ]);
  return {
    other_loopback_rejected: !other,
    gost_rejected: !gost,
    dns_rejected: dnsDenied,
    gateway_rejected: !gatewayOpen,
    public_rejected: !publicOpen,
    capabilities_dropped: capabilitiesDropped(),
  };
}

(async () => {
  const negative = await commonNegativeChecks();
  if (mode === "setup") {
    const relayOpen = await tcpOpen("127.0.0.1", RELAY_PORT);
    process.stdout.write(JSON.stringify({
      schema_version: 1,
      setup_relay_rejected: !relayOpen,
      ...negative,
    }));
    return;
  }
  if (mode !== "agent") process.exit(2);
  const [valid, invalid] = await Promise.all([
    httpResult(PLACEHOLDER),
    httpResult(`${PLACEHOLDER}-wrong`),
  ]);
  let relayPassed = false;
  if (valid !== null && valid.status === 200) {
    try {
      const payload = JSON.parse(valid.body.toString("utf8"));
      relayPassed = Array.isArray(payload.data) && payload.data.some((item) => item && item.id === MODEL);
    } catch (_) {}
  }
  process.stdout.write(JSON.stringify({
    schema_version: 1,
    agent_relay_passed: relayPassed,
    wrong_auth_rejected: invalid !== null && invalid.status === 401,
    ...negative,
  }));
})().catch(() => process.exit(1));
""".strip()


_SETUP_KEYS = frozenset(
    {
        "schema_version",
        "setup_relay_rejected",
        "other_loopback_rejected",
        "gost_rejected",
        "dns_rejected",
        "gateway_rejected",
        "public_rejected",
        "capabilities_dropped",
    }
)
_AGENT_KEYS = frozenset(
    {
        "schema_version",
        "agent_relay_passed",
        "wrong_auth_rejected",
        "other_loopback_rejected",
        "gost_rejected",
        "dns_rejected",
        "gateway_rejected",
        "public_rejected",
        "capabilities_dropped",
    }
)


def _offline_install_command(*, executable: str, version: str) -> str:
    """Return a fixed shell program with no package manager or network action."""

    return (
        "set -euo pipefail; "
        f"test -x {NODE_PREFIX}/bin/node; "
        f"test -x {AGENT_PREFIX}/bin/{executable}; "
        'umask 077; mkdir -p "$HOME/.nvm"; '
        "printf '%s\\n' 'export PATH=\""
        f"{AGENT_PREFIX}/bin:{NODE_PREFIX}/bin:$PATH"
        "\"' > \"$HOME/.nvm/nvm.sh\"; "
        '. "$HOME/.nvm/nvm.sh"; '
        f'test "$(node --version)" = "{NODE_VERSION}"; '
        f'test "$({executable} --version)" = "{version}"'
    )


class _PinnedNetworkAdmission:
    """Run fixed phase probes and persist only host-authored booleans."""

    _setup_network_admission: dict[str, bool | int] | None = None
    _agent_network_admission: dict[str, bool | int] | None = None

    async def _network_probe(
        self, environment: BaseEnvironment, mode: str
    ) -> dict[str, bool | int]:
        if mode not in {"setup", "agent"}:
            raise RuntimeError("Pinned network probe phase is invalid")
        result = await self.exec_as_agent(  # type: ignore[attr-defined]
            environment,
            command=(
                f"{NODE_PREFIX}/bin/node -e "
                f"{shlex.quote(_NETWORK_PROBE_JS)} {mode}"
            ),
            timeout_sec=15,
        )
        try:
            parsed = json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("Pinned network probe did not return scalar JSON") from error
        expected = _SETUP_KEYS if mode == "setup" else _AGENT_KEYS
        if (
            not isinstance(parsed, dict)
            or frozenset(parsed) != expected
            or parsed.get("schema_version") != 1
            or any(parsed[key] is not True for key in expected - {"schema_version"})
        ):
            raise RuntimeError("Pinned network phase admission failed")
        return parsed

    def _write_network_admission(self) -> None:
        setup = self._setup_network_admission
        agent = self._agent_network_admission
        if setup is None or agent is None:
            return
        marker = {
            "schema_version": 1,
            "setup_relay_rejected": setup["setup_relay_rejected"],
            "agent_relay_passed": agent["agent_relay_passed"],
            "wrong_auth_rejected": agent["wrong_auth_rejected"],
            "other_loopback_rejected": (
                setup["other_loopback_rejected"]
                and agent["other_loopback_rejected"]
            ),
            "gost_rejected": setup["gost_rejected"] and agent["gost_rejected"],
            "dns_rejected": setup["dns_rejected"] and agent["dns_rejected"],
            "gateway_rejected": (
                setup["gateway_rejected"] and agent["gateway_rejected"]
            ),
            "public_rejected": (
                setup["public_rejected"] and agent["public_rejected"]
            ),
            "capabilities_dropped": (
                setup["capabilities_dropped"] and agent["capabilities_dropped"]
            ),
        }
        if any(value is not True for key, value in marker.items() if key != "schema_version"):
            raise RuntimeError("Pinned network phase marker is incomplete")
        path = Path(self.logs_dir).parent / NETWORK_ADMISSION_FILENAME  # type: ignore[attr-defined]
        payload = (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "ascii"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            written = 0
            while written < len(payload):
                written += os.write(descriptor, payload[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class PinnedQwenCode(_PinnedNetworkAdmission, QwenCode):
    """Qwen Code with its exact external prefix admitted offline."""

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if self.version() != QWEN_CODE_VERSION:
            raise RuntimeError("Pinned Qwen Code version does not match the campaign")
        await self.exec_as_agent(
            environment,
            command=_offline_install_command(
                executable="qwen", version=QWEN_CODE_VERSION
            ),
        )
        self._setup_network_admission = await self._network_probe(
            environment, "setup"
        )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._agent_network_admission = await self._network_probe(
            environment, "agent"
        )
        try:
            await super().run(instruction, environment, context)
        finally:
            self._write_network_admission()


class PinnedOpenCode(_PinnedNetworkAdmission, OpenCode):
    """OpenCode with its exact external prefix admitted offline."""

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if self.version() != OPENCODE_VERSION:
            raise RuntimeError("Pinned OpenCode version does not match the campaign")
        await self.exec_as_agent(
            environment,
            command=_offline_install_command(
                executable="opencode", version=OPENCODE_VERSION
            ),
        )
        self._setup_network_admission = await self._network_probe(
            environment, "setup"
        )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._agent_network_admission = await self._network_probe(
            environment, "agent"
        )
        try:
            await super().run(instruction, environment, context)
        finally:
            self._write_network_admission()
