# Memory-operation component results

Date: 2026-08-24 (MST)

## Conclusion

All four no-thinking deployment profiles passed all 33 exact operations in the
frozen component battery: 9/9 Graphiti-style resolver operations and 24/24
explicitly synthetic memory-transaction operations. This small panel therefore
did not separate their bounded decision accuracy. Laguna XS had the smallest
summed measured-request wall time at 42.05625 seconds, followed by Ornith with
reasoning disabled at 57.20794 seconds, Qwen3.6 at 65.23171 seconds, and Laguna
S at 148.96527 seconds.
The four profiles are tied on quality within this clarified battery; their
latency and completion-token differences are descriptive deployment
measurements, not a quality ranking.

The matched Ornith reasoning-on profile is exploratory. It retained 9/9 on the
Graphiti-style resolver but fell to 18/24 on the synthetic extension, for 27/33
overall. All six failures were the secret- and transcript-injection-refusal
variants, and the narrow protected-value canary recorded six emissions. The
same run produced 13,128 completion tokens and 196.45785 seconds of summed
request wall time, versus 3,170 tokens and 57.20794 seconds with reasoning
disabled. The runtime did not expose a separate reasoning-token count, so the
extra completion tokens cannot be labeled as measured reasoning tokens. The
protected-value scalar also does not partition matches by scanned response
surface.

These are component results, not end-to-end Graphiti, MemFS, retrieval, or
memory-agent scores. The [planning and source audit](laguna-graphiti-memory-plan-2026-08-24.md)
defines that boundary and the intended 72K-trace follow-up.

## Exact scalar results

Each row is one complete execution of 11 cases with three deterministic
variants per case. Counts and token totals are exact. Summed request wall times
are shown to five decimal places from the recorded scalar totals; they exclude
artifact validation, model startup, the startup probe, and shutdown.

| Profile | Thinking policy | Quantization | All operations | Graphiti resolver | Synthetic extension | Schema valid | Protected-value emissions | Prompt tokens | Completion tokens | Reasoning tokens | Summed request wall (s) | Publication status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Laguna XS 2.1 33B-A3B | off | Q4_K_M | 33/33 | 9/9 | 24/24 | 33/33 | 0 | 16,923 | 2,456 | unavailable | 42.05625 | complete |
| Ornith 1.5 35B-A3B | off | Q4_K_M | 33/33 | 9/9 | 24/24 | 33/33 | 0 | 17,104 | 3,170 | unavailable | 57.20794 | complete |
| Qwen3.6 35B-A3B | off | UD-Q4_K_XL | 33/33 | 9/9 | 24/24 | 33/33 | 0 | 17,104 | 3,238 | unavailable | 65.23171 | complete |
| Laguna S 2.1 118B-A8B | off | UD-Q4_K_XL | 33/33 | 9/9 | 24/24 | 33/33 | 0 | 16,923 | 2,918 | unavailable | 148.96527 | complete |
| Ornith 1.5 35B-A3B | on, exploratory | Q4_K_M | 27/33 | 9/9 | 18/24 | 33/33 | 6 | 17,038 | 13,128 | unavailable | 196.45785 | partial |

Every row completed all 11 cases and 33 requests without an unexpected tool
call. Prompt-cache reuse was disabled and the server reported zero cached
prompt tokens for all 165 measured requests. `partial` on the Ornith
reasoning-on row denotes exact-oracle failures, not an interrupted run. The
four no-thinking rows each passed 3/3 secret-refusal and 3/3
transcript-injection-refusal variants. Ornith reasoning-on passed 0/3 and 0/3,
respectively.

## What the battery measures

The 9 Graphiti-style operations cover semantic reuse, accepting a changed fact
while selecting its contradiction for invalidation, and accepting an unrelated
new fact. The model emits Graphiti's two-array edge-resolver shape and the
harness grades exact index sets. The test does not include extraction,
embedding candidate recall, entity resolution, graph mutation, retrieval, or a
reader.

The other 24 operations are an explicitly synthetic extension for add,
supersede, delete, duplicate no-op, temporal invalidation, tier placement,
secret refusal, and untrusted-transcript instruction refusal. They test an
exact bounded transaction proposal that deterministic code could validate and
apply. They do not exercise the current Letta Code MemFS tool-and-Git contract
or a writable filesystem.

All five profiles emitted schema-valid JSON on 33/33 requests. In this frozen
panel the no-thinking profiles also agreed with every exact oracle, so a larger
and more varied dataset is required to compare memory-decision quality. The
private Graphiti traces are useful for that next stage only after successful,
atomic pre/post transitions are isolated and split by group and connected
entity cluster.

## Reproducibility

The runs used one NVIDIA DGX Spark / GB10 and the same native
[`llama.cpp` b10453](https://github.com/ggml-org/llama.cpp/releases/tag/b10453)
server at source revision
`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`. The server binary was
`sha256:ae1bd49f869ff3397b2a5d757fcf010c6eaaf16c4e3071a15861312defcd4e40`.
Serving was serial, single-slot, full-GPU-offload inference at 32,768 allocated
context tokens with Q8 key/value cache, temperature zero, a 1,536-token output
cap, no per-case warm-up, no prompt-cache reuse, and no speculative draft.

The frozen suite schema version is 1 and its content-bound protocol digest is
`sha256:96df2d5d742c6f4863c77ec3c6cc980845d43900e25607d37fe0be361f0808f1`.
It binds the prompts, response schemas, deterministic variant construction,
limits, exact oracles, protected-value set, and grading contract into the
suite, plan fingerprint, model-bound case IDs, and evidence. Runs made before
that binding existed were diagnostic and are excluded from every number in
this report. A semantic protocol change requires a new digest and fresh plans;
reusing a case name is insufficient.
The model-source pins were:

- `poolside/Laguna-XS-2.1-GGUF@1a37c0a5fb8c7a18e6106decb6be6327d1b63fa6`;
- `ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb`;
- `unsloth/Qwen3.6-35B-A3B-MTP-GGUF@5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`; and
- `unsloth/Laguna-S-2.1-GGUF@750f92f90cf54159c4d7a610cb7b3e74498e75c6`.

The Ornith off/on rows use the same model artifact and runtime. Exact model
files, split-shard hashes, runtime provenance, lifecycle counts, and the
recomputed scalar summaries are retained in the
[sanitized evidence index](../evidence/index.json). Prompts, completions,
reasoning text, values, paths, nonces, and request identifiers are excluded.

## Limits

- Each profile ran once over three fixed variants per case. The variants add
  deterministic coverage; they are not independent repeated experiments.
  There is no basis here for statistical significance or a confidence
  interval.
- This is a deployment-artifact panel, not an architecture-only ablation.
  It mixes Q4_K_M and UD-Q4_K_XL, 33B/35B/118B total sizes, different active
  parameter counts, model families, tokenizers, and chat templates. Response
  lengths also differ, so wall-time differences cannot be assigned to any one
  factor.
- `completion_tokens` includes every token decoded by b10453. That runtime
  does not expose an exact reasoning partition, leaving `reasoning_tokens`
  unavailable even for the reasoning-on row. Its first-token time begins at
  the first reasoning-or-visible stream delta, not necessarily the first JSON
  token.
- Protected-value detection recognizes only a contiguous verbatim synthetic
  value after NFKC normalization and case folding across visible output,
  reasoning, and tool payloads. It does not establish general resistance to
  split, encoded, encrypted, or confusable-transformed exfiltration.
- Perfect bounded results do not establish durable memory quality, retrieval
  utility, safe mutation, broad instruction robustness, or performance on the
  unlabeled private trace distribution.

The complete protocol and evidence-publication rules are in the
[benchmark record](../BENCHMARK.md#memory-operation-component-protocol).
