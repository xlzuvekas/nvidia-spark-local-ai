# Qwen3.8-Flash-Next upstream qualification on GB10 — 2026-08-29

## Decision

No released or merged primary-source path currently qualifies an NVFP4
Qwen3.8-Flash-Next deployment on one NVIDIA GB10 / DGX Spark. This is a
support-status conclusion, not a claim that the locally cached Radix NVFP4
artifact or a custom current-SM121 runtime cannot work.

Qwen's public model materials establish why the single-Spark problem is
unusual: Flash-Next has a 125B main model plus a 51B N-gram embedding table,
with 6B activated parameters per token. Qwen states that the embedding table
can be host-offloaded and asynchronously prefetched. That makes both the
offload path and its interaction with the serving runtime material to any GB10
claim. [Qwen's repository](https://github.com/QwenLM/Qwen3.8-Flash-Next)
also documents SGLang, vLLM, TokenSpeed, Transformers, and llama.cpp local-use
paths, but does not provide a GB10 NVFP4 qualification.

The current upstream status is narrower than a model-card compatibility claim:

| Stack | What its primary source currently says | GB10 implication |
| --- | --- | --- |
| Qwen | The official collection publishes base and FP8 Flash-Next artifacts; the FP8 card describes fine-grained FP8 (block size 128). | The official low-precision alternative is FP8, not a Qwen-owned Flash-Next NVFP4 checkpoint. |
| SGLang | [Flash-Next support PR #36497](https://github.com/sgl-project/sglang/pull/36497) remains open; its cookbook points to the model-support work rather than a tagged release. | No released upstream GB10 qualification. A PR comment reports a GB10 run after consumer-Blackwell QSA dispatch changes and an explicit Triton attention setting, which is useful implementation evidence but not a merged or signed-off configuration. |
| vLLM | [Flash-Next support PR #53896](https://github.com/vllm-project/vllm/pull/53896) remains open. Its stated validation covers NVFP4 without N-gram offload on GB300, GB200, and H200; N-gram offload is BF16/FP8 only on GB200. | The obvious one-Spark NVFP4-plus-offload design is expressly outside that validation, and GB10 is absent. |
| Unsloth | The relevant Flash-Next releases are [GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) and FP8. | GGUF is a practical llama.cpp/Pi baseline, not evidence for Flash-Next NVFP4 or GB10 support. |

The upstream report in the open SGLang PR is especially actionable for our
current candidate: it says an SM121 capability gate left QSA without the
intended sparse-decode path, and that widening the gate plus choosing Triton
allowed server boot and correct serving. It also says explicit `qwen3`
reasoning-parser configuration mattered. Those observations reinforce, rather
than remove, this repository's runtime, parser, varied-token, long-context,
and cache-safety gates.

## Consequence for local work

Keep every Flash-Next NVFP4-on-Spark profile labeled an experimental local
candidate. A speed result is publishable only for its exact pinned runtime and
after the repository's local correctness, tool/parser, long-context, cache,
and memory admission—not because either upstream project has accepted a
generic GB10 recipe.

The lowest-risk practical comparison remains a local GGUF/llama.cpp baseline
versus an independently admitted custom NVFP4 stack. The current Pi/cowork
path must additionally pass its own immutable-prefix, isolated-wrapper, and
agent-semantics gates; neither this support review nor a static parser probe is
an agent benchmark.

## Primary sources

- [Qwen Flash-Next collection](https://huggingface.co/collections/Qwen/qwen38-flash-next)
- [Qwen Flash-Next FP8 model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
- [Qwen Flash-Next repository](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [SGLang Flash-Next cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
- [SGLang open Flash-Next support PR #36497](https://github.com/sgl-project/sglang/pull/36497)
- [vLLM Flash-Next recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)
- [vLLM open Flash-Next support PR #53896](https://github.com/vllm-project/vllm/pull/53896)
- [Unsloth Qwen3.8 collection](https://huggingface.co/collections/unsloth/qwen38)
