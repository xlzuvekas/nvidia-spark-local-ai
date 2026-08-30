# Current-SM121 Pi/cowork agent-admission preflight — 2026-08-29

## Result

The exact current SM121 storage image has passed a **static parser/CLI
preflight** for the prospective C1 Pi/cowork profile. The image-local Python
both exposes and initializes the required entries:

- Qwen reasoning parser: `qwen3`;
- Qwen Coder tool-call parser: `qwen3_coder`.

It also accepts the exact frozen C1 argv with a dummy model and retains the
intended resolved scalar values: 4,096-token chunked prefill, one running
request, and 65,536-token total/context limits. The dummy-model path exits
before model or GPU initialization, so this is a CLI-contract check rather
than a serving test.

The check is intentionally not a server or model admission. It starts an
ephemeral no-network, read-only Docker process with no model mount, no GPU
request, no published port, no host weight mount or model loading, and no
inference request. It returns only the exact local image/source identity,
parser registry/initialization booleans, and resolved parser/limit scalars. It
emits or persists no timing metric, prompt, completion, reasoning, tool
payload, or result directory.

Run it with:

```bash
python3 sparkbench.py sm121-agent-parser-preflight
```

The checked image is `local/sglang:sm121-storage-274ee330-runtime`, local ID
`sha256:b14c39fb7cb2e0b82f2f8cae1e115a55f2bb69b5ec6fd7ccc4099b219d1096b0`,
and source tree `274ee330db7ea9653807b868c0fb8693d50ed7b2`.

## What changed

`qwen38-flash-next-nvfp4-sm121-triton-storage-agent-admission-sglang` is now
an exact, **tombstoned** prospective profile. It preserves the public retained
current-SM121 C1 geometry: 64K context, Triton attention,
`flashinfer_cutlass`, NVFP4, io_uring PLE, lazy Mamba state, 4K chunked
prefill, one running request, and disabled CUDA graphs. Its only intended
serving deltas are:

- `tasks = ["chat", "json", "thinking", "tools"]`;
- `{"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"low"}}`;
- `--reasoning-parser qwen3`; and
- `--tool-call-parser qwen3_coder`.

The prospective profile also intentionally tightens its future host-safety
admission gates to 14 GiB `MemAvailable` and 64 MiB starting/growth swap.

It explicitly does not import retired QSA/TRT-LLM overlays, MTP, mmap, or the
vLLM-only `--enable-auto-tool-choice` flag. Generic plan, benchmark, and matrix
paths reject the profile until a dedicated controller exists.

## Private C1 plan and audit scaffold

The repository now has an exact six-case private C1 plan contract and a
read-only audit shape under the ignored `logs/agent-admissions/` tree. The
plan freezes the prospective profile with these ordered gates:

1. exact-answer quality;
2. deterministic select/call, no-tool, two-hop, and tool-error-recovery
   scenarios; and
3. a first-turn 60K low-thinking-plus-tools long-context/cache-zero probe.

The plan uses no generic benchmark route and is deliberately not evidence. It
can be frozen with:

```bash
python3 sparkbench.py sm121-agent-admission-plan
```

There is currently **no live C1 execution command or latent server path**. The
scaffold intentionally contains no caller-supplied request hooks: such hooks
could fabricate a scalar success record without actually proving the request
body, parser state, tool semantics, or cache behavior. Its read-only auditor
can validate record shape, but cannot accept any record as an admission while
the controller is unimplemented. The prospective model remains
runtime-tombstoned even if an internal caller sets an authorization-like
attribute. The corresponding static parser preflight was rechecked on the
pinned image during this continuation and again passed without starting an
inference server or leaving a container behind.

The next private building block is a byte-bound direct client, still unreachable
from every CLI and runner. It owns final JSON serialization, validates the
serialized low-thinking/tool/cache-zero body, sends it once through fixed
loopback/no-proxy/no-redirect transport, and exposes only bounded scalar
diagnostics to a future controller. Its internally rendered long case requires
at least 60,000 returned input tokens, a cache-zero counter, one request on
that client, and an exact `LONG-CONTEXT-READY` no-tool final answer. It does not
establish that this is the first request of a freshly started server or that
native metric cache counters are zero; those remain mandatory controller-owned
proofs. Neither the client nor its standalone scalar validator is an admission
proof without that controller.

The repository also freezes the one allowed scalar runtime-identity projection:
the cache-on unified Radix/Mamba-lazy state, 4K chunked prefill, Qwen reasoning
and tool parsers, one running request, and both 64K limits. An uninvoked
private inspector now reads that identity from one owned C1 server using fixed
loopback/no-proxy/no-redirect transport, a bounded strict finite-JSON
`/server_info` response with a total read deadline, and a bounded one-event
startup-log projection scoped to that container generation. It checks
ownership, running state, process generation, and the sole `127.0.0.1:30000`
to container-`30000` binding before and after those reads. The pinned image
source expands resolved server configuration into `/server_info`'s top-level
object, so cache, parser, and limit fields have no recursive fallback. No
runner or CLI calls this inspector; it cannot start a server or turn a forged
record into an admission.

This is deliberately a planning and audit hardening step, not an agent
admission, performance result, or permission to run Pi.

## What remains before an agent result

The parser/CLI check and private plan only remove static and topology
uncertainty.
A reviewed in-repository execution adapter must still prove, in fresh C1
lifetimes:

1. parser initialization and exact low-effort payload after client/provider
   transformation;
2. exact-answer quality, strict tool-call semantics, and bounded tool-error
   recovery;
3. rendered low-thinking-plus-tools long-context fit, exact no-tool result,
   fresh-lifetime and cache-zero-first-turn semantics; and
4. the stronger 14 GiB available-memory and 64 MiB starting/growth swap gates.

The adapter must construct and observe the final direct-client payload itself,
inspect only allowlisted runtime parser/limit fields, own its no-retry tool
loop, and clean up verified containers and API-key files on both complete and
partial paths. It must not accept caller-defined hooks as proof.

Only after those gates can the offline Pi-core wrapper be admitted. Its default
transport cannot be used as-is because it retries requests; the wrapper needs
a pinned custom no-retry transport. A successful future agent admission would
still not establish coding/cowork speed: it is only the prerequisite for
separately frozen fresh-lifetime Pi coding and cowork measurements.
