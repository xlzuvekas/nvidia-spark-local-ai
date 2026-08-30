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

The private controller has a separate, target-snapshot tokenizer preflight
before it can start a C1 server. It mounts only the exact cached target snapshot
read-only in an offline, CPU-only probe, renders the fixed low-thinking request
and canonical tool schema, and pins the raw-prompt, tools, tokenizer,
chat-template, and rendered-prompt hashes. The exact rendered input is 60,489
tokens; with its 128-token output reservation, the request budget is 60,617 of
65,536 tokens, leaving 4,919 tokens of headroom. This is not the parser CLI
command above: it never starts SGLang or uses a GPU, but it does verify the
actual target tokenizer rather than assuming a 60K filler prompt will fit.

## What changed

`qwen38-flash-next-nvfp4-sm121-triton-storage-agent-admission-sglang` is now
an exact, **admission-only** prospective profile. It preserves the public retained
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
paths remain blocked; only the dedicated controller may execute the profile.

## Private C1 controller and audit

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

The repository now has one dedicated live command:

```bash
python3 sparkbench.py sm121-agent-admission
```

It first freezes a new private plan, then runs exactly three fresh SGLang
lifetimes: exact-answer quality, four deterministic tool-loop scenarios, and a
first-turn 60K low-thinking-plus-tools probe. It never resumes a partial plan,
never uses generic benchmark/matrix execution, and accepts no caller-supplied
hooks, transports, payloads, or retry policy. The client constructs and checks
the final serialized request bytes in memory, and the quality gate now uses the
same explicit `FINAL: <answer>` grammar as the exact-answer validator instead
of the former bare-question form.

The controller authenticates the frozen plan, image identity, and run nonce,
then registers no launch authority until its top-level clean-start, host,
parser, and exact-tokenizer gates pass. Every lifetime independently rechecks
clean start and receives one registry-backed lease tied to the exact in-memory
model identity and frozen-plan hash. The lease is consumed immediately before
the Docker launch and revoked during lifetime cleanup, so it cannot be reused
to start a second server. Each lifetime separately checks source and runtime
parser/limit identity, owns cleanup, and records only allowlisted scalars under
the ignored private logs tree. The long-context lifetime takes two settled
fixed-loopback native metric views before and after its single request; a
same-generation cache-zero receipt must prove positive native input and zero
device/host/storage and response cache hits, evictions, and retractions. Metric
text, labels, endpoints, credentials, prompt, completion, reasoning, timing,
and container identity never enter the record.

The read-only auditor can now accept a structurally complete record, but this
change is not an admission result: no live C1 controller run was started for
this update. The current host must first satisfy both the general preflight
reserve and the stricter 14 GiB available-memory / 64 MiB starting-and-growth
swap gate. Before an admitted summary is written, the controller audits the
complete terminal event shape in memory; the CLI then invokes the independent
read-only audit and returns success only for an audited admitted record. A
forged authorization-like model attribute or a model-shaped binding cannot
bypass the controller because the runtime requires a registered one-shot lease.

This is deliberately an implementation hardening step, not an agent admission,
performance result, or permission to run Pi.

## What remains before an agent result

The parser/CLI check and controller implementation remove static and topology
uncertainty, but a successful live C1 run must still prove, in fresh lifetimes:

1. parser initialization and exact low-effort payload after client/provider
   transformation;
2. exact-answer quality, strict tool-call semantics, and bounded tool-error
   recovery;
3. the already-pinned rendered low-thinking-plus-tools long-context budget in
   a live first-turn request, exact no-tool result, fresh-lifetime, and
   cache-zero-first-turn semantics; and
4. the stronger 14 GiB available-memory and 64 MiB starting/growth swap gates.

Only after those gates can the offline Pi-core wrapper be admitted. Its default
transport cannot be used as-is because it retries requests; the wrapper needs
a pinned custom no-retry transport. A successful future agent admission would
still not establish coding/cowork speed: it is only the prerequisite for
separately frozen fresh-lifetime Pi coding and cowork measurements.
