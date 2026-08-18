"use strict";

/*
 * Deliberately tiny HTTP/1.1 credential boundary for the Harbor agent phase.
 *
 * The untrusted main container knows only PUBLIC_BEARER.  This relay is the
 * sole container with the owner-private run directory mounted.  It replaces
 * that fixed, non-secret marker with the per-run bearer read from KEY_PATH,
 * then connects to the host's mode-0600 AF_UNIX bridge.  The host bridge
 * validates and strips the internal bearer before forwarding to llama.cpp.
 *
 * There is intentionally no logging in this process: requests and the real
 * bearer are raw benchmark material and must never enter persisted output.
 */

const fs = require("fs");
const net = require("net");

const LISTEN_HOST = "127.0.0.1";
const LISTEN_PORT = 18080;
const SOCKET_PATH = "/run/sparkbench/model.sock";
const KEY_PATH = "/run/sparkbench/internal-api-key";
const PUBLIC_BEARER = "sparkbench-relay-placeholder-v1";
const MAX_CONNECTIONS = 16;
const MAX_HEADER_BYTES = 64 * 1024;
const MAX_PENDING_BYTES = 128 * 1024;
const IDLE_TIMEOUT_MS = 15 * 60 * 1000;
const REJECT_TIMEOUT_MS = 1000;

const internalBearer = fs.readFileSync(KEY_PATH, {encoding: "ascii"});
if (
  internalBearer.length < 32 ||
  internalBearer.length > 512 ||
  internalBearer.trim() !== internalBearer ||
  /[\x00-\x20\x7f]/.test(internalBearer)
) {
  process.exit(1);
}

const reject = (client, status) => {
  const message = Buffer.from(
    `HTTP/1.1 ${status}\r\nContent-Length: 0\r\nConnection: close\r\n` +
      "Cache-Control: no-store\r\n\r\n",
    "ascii",
  );
  const timer = setTimeout(() => client.destroy(), REJECT_TIMEOUT_MS);
  timer.unref();
  client.once("close", () => clearTimeout(timer));
  client.end(message, () => client.destroy());
};

const rewriteHeader = (headerBytes) => {
  const header = headerBytes.toString("latin1");
  const lines = header.split("\r\n");
  if (lines.length < 2 || !/^[A-Z]+ [^ ]+ HTTP\/1\.[01]$/.test(lines[0])) {
    return null;
  }
  let authorizationCount = 0;
  const output = [lines[0]];
  for (const line of lines.slice(1)) {
    if (line === "") continue;
    if (/^[ \t]/.test(line)) return null;
    const separator = line.indexOf(":");
    if (separator < 1) return null;
    const name = line.slice(0, separator);
    const value = line.slice(separator + 1).trim();
    if (!/^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/.test(name)) return null;
    const lower = name.toLowerCase();
    if (lower === "authorization") {
      authorizationCount += 1;
      if (value !== `Bearer ${PUBLIC_BEARER}`) return null;
      output.push(`Authorization: Bearer ${internalBearer}`);
    } else if (lower !== "connection") {
      output.push(`${name}: ${value}`);
    }
  }
  if (authorizationCount !== 1) return null;
  output.push("Connection: close", "", "");
  return Buffer.from(output.join("\r\n"), "latin1");
};

let activeConnections = 0;
const server = net.createServer({allowHalfOpen: true, pauseOnConnect: false}, (client) => {
  if (activeConnections >= MAX_CONNECTIONS) {
    reject(client, "503 Service Unavailable");
    return;
  }
  activeConnections += 1;
  let upstream = null;
  let pending = Buffer.alloc(0);
  let admitted = false;
  let released = false;
  let clientEnded = false;
  let clientClosed = false;
  let upstreamEnded = false;
  let upstreamClosed = true;
  let upstreamConnected = false;

  const releasePair = () => {
    if (!released && clientClosed && upstreamClosed) {
      released = true;
      activeConnections -= 1;
    }
  };
  const fail = () => {
    client.destroy();
    if (upstream !== null) upstream.destroy();
  };
  const rejectAndStop = (status) => {
    admitted = true;
    pending = Buffer.alloc(0);
    client.removeAllListeners("data");
    reject(client, status);
  };
  client.setTimeout(IDLE_TIMEOUT_MS, fail);
  client.once("end", () => {
    clientEnded = true;
    if (!admitted || upstream === null) {
      fail();
    } else if (upstreamConnected && !upstream.destroyed) {
      upstream.end();
    }
  });
  client.once("close", () => {
    clientClosed = true;
    if (upstream !== null && !upstreamClosed) upstream.destroy();
    releasePair();
  });
  client.on("error", fail);

  client.on("data", (chunk) => {
    if (admitted) return;
    if (pending.length + chunk.length > MAX_PENDING_BYTES) {
      rejectAndStop("431 Request Header Fields Too Large");
      return;
    }
    pending = Buffer.concat([pending, chunk], pending.length + chunk.length);
    const boundary = pending.indexOf("\r\n\r\n");
    if (boundary < 0) {
      if (pending.length > MAX_HEADER_BYTES) {
        rejectAndStop("431 Request Header Fields Too Large");
      }
      return;
    }
    if (boundary + 4 > MAX_HEADER_BYTES) {
      rejectAndStop("431 Request Header Fields Too Large");
      return;
    }
    const rewritten = rewriteHeader(pending.subarray(0, boundary + 4));
    if (rewritten === null) {
      rejectAndStop("401 Unauthorized");
      return;
    }

    admitted = true;
    client.pause();
    client.removeAllListeners("data");
    upstream = net.createConnection({path: SOCKET_PATH, allowHalfOpen: true});
    upstreamClosed = false;
    upstream.setTimeout(IDLE_TIMEOUT_MS, fail);
    upstream.on("error", fail);
    upstream.once("end", () => {
      upstreamEnded = true;
      client.end();
    });
    upstream.once("close", () => {
      upstreamClosed = true;
      if (!upstreamEnded) client.destroy();
      releasePair();
    });
    upstream.once("connect", () => {
      upstreamConnected = true;
      const bodyPrefix = pending.subarray(boundary + 4);
      pending = Buffer.alloc(0);
      upstream.write(rewritten);
      if (bodyPrefix.length) upstream.write(bodyPrefix);
      client.pipe(upstream);
      upstream.pipe(client);
      if (clientEnded) upstream.end();
      client.resume();
    });
  });
});

// Reserve exactly one transport slot for an explicit bounded rejection.  The
// kernel/Node transport drops any further excess without allocating a relay
// pair, while the application quota remains MAX_CONNECTIONS.
server.maxConnections = MAX_CONNECTIONS + 1;
server.on("error", () => process.exit(1));
server.listen({host: LISTEN_HOST, port: LISTEN_PORT, exclusive: true});

const stop = () => {
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(1), 5000).unref();
};
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
