# Laguna, Graphiti, and local memory reflection

Date: 2026-08-24 (MST)

## Conclusion

The MindStudio article is a useful orientation to Laguna S 2.1, but it is not
a reproducible local-performance report. Its 70.2% Terminal-Bench result, local
weight footprint, and 80+ token/s headline come from different evaluation and
serving conditions. Artifact revisions have also moved since the article was
published. Any local claim must therefore pin the exact target, draft, runtime,
sampling mode, context, concurrency, and thinking policy.

For memory reflection, Laguna should first be tested as a bounded decision
model, not by attaching it immediately to a live graph or writable MemFS tree.
The first panel isolates Graphiti-compatible edge resolution and a separate
synthetic memory-transaction policy. Only after that component test should an
end-to-end Graphiti replay measure state deltas and downstream retrieval.

No result is reported here. The protocol and evidence path are implemented,
but live execution still requires an idle host and explicit lifecycle
admission.

## Article audit

### What is well supported

- The current Poolside card describes Laguna S 2.1 as 117.6B total / 8.5B
  active, 48 layers, 256 routed experts plus one shared expert, and 1M native
  context. Thirty-six layers use sliding-window attention and twelve use global
  attention. [Current Poolside NVFP4 card](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4)
- Poolside reports 70.2% on Terminal-Bench 2.1 with thinking enabled in its Pool
  harness. The published comparison is pass@1 averaged over four attempts per
  task and mixes vendor, benchmark-author, and third-party maxima for other
  models. It is a harness result, not a local quantization result.
  [Poolside launch report](https://poolside.ai/blog/introducing-laguna-s-2-1)
- The official DFlash card describes a 1B, six-layer sliding-attention draft
  with block size 16. Its vLLM example uses seven speculative tokens; llama.cpp
  can use fifteen draft positions plus the target position.
  [Poolside DFlash card](https://huggingface.co/poolside/Laguna-S-2.1-DFlash)

### What should not be carried forward as one headline

- The article was published on July 25. Poolside now warns that its August
  NVFP4 release changed the weights, not merely configuration. The current
  pinned repository contains 49 weight shards totaling 99,697,277,984 bytes,
  despite stale card prose still describing roughly 71 GB. The article's
  launch-era “around 70 GB” statement is not a durable artifact pin.
  [Pinned current repository tree](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4/tree/64734b3a449a05c79657451513d97544f2f53436)
- Poolside's current Spark recipe reports about 13–14 token/s without
  speculation, approximately 15 token/s on prose and 22–24 on code with
  DFlash, and 12.6 token/s for the Q4_K_M Ollama path. That does not support
  treating 80+ token/s as typical single-request decoding. Concurrency and
  favorable code workloads can raise aggregate or short-run rates, but those
  are different measurements.
  [Current local-deployment recipe](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4#dgx-spark--mac-studio)
- Poolside says the 70.2 score used thinking. Its own limitations include
  overthinking, harness-specific tool-schema mistakes, and malformed nested
  tool arguments. Those behaviors are directly relevant to a strict memory
  writer and must be measured rather than inferred from coding scores.
  [Poolside limitations](https://poolside.ai/blog/introducing-laguna-s-2-1#limitations)
- The MindStudio article calls the draft five layers and collapses isolated
  community peaks into an 80+ token/s single-Spark claim. The official draft
  card says six layers, and the current official Spark planning numbers are
  materially lower. [Article under review](https://www.mindstudio.ai/blog/poolside-laguna-s2-local-coding-model)

The locally cached Laguna S profile in this repository is explicitly an
Unsloth split GGUF pinned to a July revision. It is useful as a reproducible
local artifact, but it must not be described as the current August Poolside
NVFP4 checkpoint.

## Graphiti boundary

Graphiti's current edge resolver consumes a new fact, existing facts, and a
second list of invalidation candidates. It returns exactly two integer arrays:
`duplicate_facts` can refer only to existing facts, while
`contradicted_facts` can refer to either continuously numbered list.
[Pinned Graphiti resolver](https://github.com/getzep/graphiti/blob/993e081a6d7948a0d8851c12a5fbdbeb49fed862/graphiti_core/prompts/dedupe_edges.py)

The first benchmark family therefore grades three exact decisions:

1. reuse a semantically duplicate existing fact;
2. accept a changed fact and identify the contradiction to invalidate; and
3. accept an unrelated new fact without duplicate or contradiction indexes.

This is not an end-to-end Graphiti score. It deliberately excludes extraction,
embedding candidate recall, entity resolution, database mutation, community
updates, retrieval, and answer generation. Those stages need separate metrics
when the private 72K-trace corpus is replayed.

## MemFS-style extension and Letta Evals

The second family uses a synthetic, constrained transaction proposal for add,
supersede, delete, duplicate no-op, temporal invalidation, tier placement,
secret refusal, and transcript-injection refusal. Deterministic code would own
path validation, patch application, Git commit, conflict handling, and cursor
advancement. This is intentionally smaller and safer than asking a small model
to operate Bash, Edit, Git, and a full MemFS tree.

The current Letta Evals repository is useful as source material, but its
packaged memory leaderboard suites should not be treated as a ready reflector
benchmark:

- `core-memory-update` creates a separate agent and sends the contradictory
  fact during setup, before the candidate answer is graded; it tests later
  recall more directly than candidate memory writing.
- Several packaged suite files have grader/reward-key schema mismatches against
  the current runner.
- The 1,100-row read/update datasets can still seed grouped, deterministic
  state-transition cases, provided facts from the same generated group never
  cross train/evaluation partitions.

Sources: [Letta Evals at the audited revision](https://github.com/letta-ai/letta-evals/tree/f6855fed1dbca208dd603e930d8cf558bc6555f4),
[update setup](https://github.com/letta-ai/letta-evals/blob/f6855fed1dbca208dd603e930d8cf558bc6555f4/letta-leaderboard/core-memory-update-agent/setup_agent.py),
and [update suite](https://github.com/letta-ai/letta-evals/blob/f6855fed1dbca208dd603e930d8cf558bc6555f4/letta-leaderboard/core-memory-update-agent/suites/core-memory-update.yaml).

## Ornith fit and limits

Ornith 1.5 35B-A3B is a useful middle control rather than a presumed winner.
Its official card identifies a 36B Qwen3.5-MoE-derived model with about 3B
parameters active per token, an MIT license, OpenAI-compatible tool use, and
vendor-reported coding and agentic gains over Qwen3.6 35B-A3B. The accompanying
write-up says its self-improvement loop jointly generates tasks, scaffolds, and
solution rollouts, then optimizes them with GRPO rewards for validity,
difficulty, novelty, fidelity, and hack resistance. Those properties make it
interesting for constrained memory decisions, but they are not evidence of
memory-update quality. [Official model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
and [self-improvement write-up](https://ornith.ai/ornith_1_5.html)

The vendor's 35B results are averaged over five runs and use different
harnesses, context limits, sampling settings, judges, and thinking policies by
benchmark. For example, its Terminal-Bench result uses a 128K context and a
modified chat-template/harness path, while several agentic results use
thinking. They therefore motivate testing, not a local quality claim.

The local profile instead pins the Ornith AI-hosted Q4_K_M GGUF at revision
`12393612fd4f730ff5aadc23e9b8f9648aa49ceb`: a 21,713,463,040-byte text
artifact with a fixed digest, served by the same llama.cpp b10453 binary as the
other panel members. The GGUF declares 262K context, but this protocol
allocates 32K. The formal profile disables reasoning; its matched reasoning-on
profile is exploratory because b10453 does not separately count reasoning
tokens. [Pinned GGUF repository](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/tree/12393612fd4f730ff5aadc23e9b8f9648aa49ceb)

## First local panel

The fixed component suite uses 11 cases × 3 byte-replayable variants,
temperature zero, one slot, 32K allocated context, Q8 KV, and prompt caching
forced off. It records only scalar correctness, latency, token, cache-control,
and server-timing fields.

The first matched no-thinking panel is:

- Laguna XS 2.1 33B-A3B Q4_K_M;
- Laguna S 2.1 118B-A8B July-pinned UD-Q4_K_XL, labeled by exact revision;
- Ornith 1.5 35B-A3B Q4_K_M; and
- Qwen3.6 35B-A3B UD-Q4_K_XL as the family control.

This is a deployment-artifact panel, not an architecture-only ablation: the
Q4_K_M and UD-Q4_K_XL quantizations differ and every result must retain its
exact artifact pin.

An Ornith thinking profile exists for exploratory use, but llama.cpp b10453
does not report an exact reasoning-token partition in streaming chat. Its
completion count combines hidden reasoning, delimiters, and visible JSON, and
its TTFT is time to the first hidden-or-visible delta. It must not enter the
formal reasoning-usage comparison until a pinned runtime reports exact
reasoning tokens.

## Follow-up with the 72K Graphiti traces

After the component panel, derive supervision and replay cases only from
complete, successful, atomic pre/post transactions:

- create node, link existing node, create fact, reuse fact, invalidate fact,
  summary/attribute delta, and no graph delta;
- current-episode provenance distinguishes new/reused facts from invalidated
  prior facts;
- no graph delta is not automatically a true negative; failed or retried LLM
  output is not a semantic negative;
- split by group/user and connected entity cluster, not random trace row;
- keep raw episodes, facts, queries, embeddings, IDs, errors, and LLM text
  inside the private boundary; publish only sanitized scalar aggregates.

The end-to-end phase should pair intrinsic graph-delta accuracy with downstream
retrieval/QA under a fixed reader and retriever. Otherwise a strong reader can
hide a weak memory writer, or a retrieval miss can be misattributed to the
reflector.
