# SparkBench 24-Hour Campaign — 2026-08-15

## Window and Operating Rules

The campaign began with the frozen quick matrix at
`2026-08-16 02:10:55 UTC` (`2026-08-15 19:10:55 MST`) and ends exactly 24 hours
later. Treat its creation time as **T+0**. Run only one SparkBench process at a
time and omit `--allow-download` during the cached phase. Frozen plans, image
digests, and event journals are the record of truth; resume an interrupted run
with `python3 sparkbench.py run results/<run-directory>`.

Use these hard admission gates:

- **T+17 h:** stop the cached queue, preserving seven hours. Do not start an
  item if its upper estimate would cross this gate. If one overruns the gate,
  send the foreground process one `Ctrl-C`, wait for lifecycle cleanup, and
  resume it after the campaign. Never use `kill -9` or unload an Ollama model
  unless SparkBench proved ownership.
- **T+23 h:** start no new download, server, or benchmark. Use the last hour to
  drain the active run, summarize results, and verify cleanup.
- **T+24 h:** hard stop. Confirm `docker ps --filter
  label=ai.sparkbench.managed=true` is empty and inspect `ollama ps`; do not
  alter an unrelated user load.

## Cached-First Queue

Queue item 0 is already running and must not be restarted:

```text
python3 sparkbench.py matrix --task chat --suite manifests/suites/quick.toml
results/matrices/20260816T021055Z-quick
```

It froze 20 chat profiles before the two new throughput profiles were added.
Expected duration is 2.5–4.0 h. After it exits, run the following rows in order.
The exact command for each is `python3 sparkbench.py benchmark <model> --suite
manifests/suites/<suite>.toml`.

| # | Model | Suite | Estimate |
| -: | --- | --- | ---: |
| 1 | `qwen38-27b-bf16-throughput` | `quick` | 0.15–0.25 h |
| 2 | `qwen38-27b-nvfp4-throughput` | `quick` | 0.12–0.20 h |
| 3 | `qwen38-27b-bf16` | `core` | 2.3–2.7 h |
| 4 | `qwen38-27b-nvfp4` | `core` | 1.1–1.4 h |
| 5 | `qwen38-27b-nvfp4-mtp3` | `core` | 0.7–0.9 h |
| 6 | `qwen38-27b-bf16-throughput` | `core` | 1.2–1.6 h |
| 7 | `qwen38-27b-nvfp4-throughput` | `core` | 0.7–1.0 h |
| 8 | `qwen38-27b-nvfp4-mtp3-throughput` | `core` | 0.4–0.6 h |
| 9 | `gpt-oss-120b-mxfp4` | `core` | 0.4–0.7 h |
| 10 | `qwen3-coder-30b-a3b-bf16` | `core` | 0.4–0.7 h |
| 11 | `rlm-qwen3-8b-bf16` | `core` | 0.5–0.8 h |
| 12 | `gemma-3-4b-it-bf16` | `vision` | 0.15–0.25 h |
| 13 | `all-minilm-l6-v2-bf16` | `embeddings` | 0.02–0.04 h |
| 14 | `ollama-nomic-embed-text-f16` | `embeddings` | under 0.02 h |
| 15 | `ollama-qwen3.5-35b-a3b-q4-k-m` | `vision` | 0.05–0.10 h |
| 16 | `ollama-gemma3-12b-q4-k-m` | `vision` | 0.05–0.10 h |
| 17 | `ollama-gemma4-31b-q4-k-m` | `vision` | 0.08–0.15 h |
| 18 | `ollama-deepseek-ocr-f16` | `vision` | 0.05–0.10 h |
| 19 | `ollama-mistral-medium-3.5-128b-q4-k-m` | `vision` | 0.15–0.25 h |
| 20 | `ollama-qwen3-30b-a3b-q4-k-m` | `core` | 0.2–0.3 h |
| 21 | `ollama-nemotron-cascade-2-q4-k-m` | `core` | 0.2–0.3 h |
| 22 | `ollama-qwen3.5-35b-a3b-q4-k-m` | `core` | 0.25–0.4 h |
| 23 | `ollama-nemotron-3-super-q4-k-m` | `core` | 0.5–0.7 h |
| 24 | `ollama-gemma4-31b-q4-k-m` | `core` | 0.8–1.1 h |
| 25 | `ollama-llama3.3-70b-q4-k-m` | `core` | 1.9–2.4 h |

The complete cached queue is approximately 14.9–21.1 h including item 0, so
items 20–25 are explicitly opportunistic. The T+17 admission rule takes
precedence over finishing the table. Do not add core runs for Mistral, Gemma 3,
GLM, DeepSeek OCR, or either diffusion model: quick/vision coverage or a better
controlled anchor already exists, and their current adapter would not answer a
new question.

## Capability Reserve: T+17 to T+23

Spend the six-hour active reserve on one end-to-end missing capability, not on
more duplicate decode runs. The preferred order is:

1. Add multimodal pooling support and smoke
   `Qwen/Qwen3-VL-Embedding-2B` at its pinned revision.
2. Add a true rerank request/quality path, then smoke
   `Qwen/Qwen3-VL-Reranker-2B`; embedding similarity is not a valid substitute
   for cross-encoder reranking.
3. If at least three hours remain and audio fixtures are ready, trial
   `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`. Otherwise, implement
   sandboxed coding pass@1 on the cached Qwen Coder or ASR RTF/WER on the cached
   Whisper family.

Checkpoint details and pinned revisions are in
[`model-candidates-2026-08-15.md`](model-candidates-2026-08-15.md). Do not
download a model until its adapter, fixture, metric, and manifest entry exist.

## Interpretation and Known Gaps

This section records the campaign's starting assumptions. Several gaps below
were closed during execution; the dated [results report](benchmark-results-2026-08-16.md)
is authoritative for final coverage.

The Ollama service sets `OLLAMA_HOST` but not `OLLAMA_NUM_PARALLEL`. Its default
parallelism is one, so suite concurrency currently measures client queueing and
aggregate service behavior—not true multi-sequence scaling. Label those results
`queued`; changing parallelism also multiplies context memory. Compare genuine
concurrency only across the matched eight-sequence Qwen throughput profiles.
See the Ollama [concurrency documentation](https://github.com/ollama/ollama/blob/main/docs/faq.mdx#how-does-ollama-handle-concurrent-requests).

The harness still cannot make valid quality measurements for cached rerank,
ASR, diffusion/DFlash, or coding models. Thinking has no enabled correctness
suite; vision is only a generated red-image check; embeddings report shape and
speed rather than retrieval quality; MTP acceptance is not parsed. The smallest
high-value additions are, in order: queued-versus-parallel provenance, a
container-sandboxed coding suite, thinking exact-answer cases plus reasoning
tokens, MTP acceptance parsing, ASR WER/real-time factor, true rerank, and small
OCR/retrieval fixtures. SGLang/DFlash support is valuable but is a larger
lifecycle and protocol addition.

## Completion Criteria

The campaign is complete only when:

- every started plan has a terminal summary or a documented resumable abort;
- the 20-profile quick matrix and both added throughput quick gates are
  accounted for;
- the six Qwen core anchors, three distinct vLLM core models, vision probes,
  and both embedding suites either complete or have an evidence-backed failure;
- at least one reserve capability produces an executable suite and a measured
  artifact, or its exact adapter/runtime blocker is recorded;
- reports retain metric provenance and label Ollama concurrency as queued; and
- no SparkBench-owned server remains at T+24, with result journals, summaries,
  telemetry, and server provenance left intact.
