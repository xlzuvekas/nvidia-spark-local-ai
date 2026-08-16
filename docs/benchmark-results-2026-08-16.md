# DGX Spark Benchmark Campaign Results — 2026-08-16

This is an evidence-backed snapshot of the 24-hour campaign through
`results/20260816T214513Z-qwen38-27b-ud-q4-k-xl-llamacpp-mtp4-llamacpp-mtp-depth-confirm-383e6711`,
plus the completed media and training artifacts retained under `/tmp`. Tests
used one NVIDIA GB10 and one inference configuration at a time. Exact
revisions, image digests, prompts, and runtime arguments remain frozen in each
run's `plan.json`.

## Measurement Rules

Aggregate output throughput is the primary text-generation metric: completed
output tokens divided by measured case wall time. Per-request client decode
rates are secondary, especially when vLLM bundled multiple tokens into one SSE
emission (Qwen3.8 MTP, GPT-OSS, and DFlash). Prefill figures are prompt tokens
divided by client-observed TTFT, not isolated engine prefill, unless explicitly
reported by Ollama. The quick suite has only three decode repetitions and two
concurrency bursts, so it supports medians but not p95 claims.

Ollama ran with its default `OLLAMA_NUM_PARALLEL=1`. Its concurrency cases
therefore measure queued aggregate service behavior, not parallel sequence
scaling. All rates below are tokens/s unless another unit is shown.

## Cached 20-Profile Quick Matrix

The matrix at `results/matrices/20260816T021055Z-quick` reached a terminal state
for all 20 frozen profiles: 5 complete, 12 partial, and 3 startup failures, with
82 completed cases.

### Qwen3.8 Precision, MTP, and Scheduling

| Profile | Single | Concurrency 2 | Concurrency 4 | Approx. prefill 256 / 2K / 8K | 8K needle |
| --- | ---: | ---: | ---: | ---: | --- |
| BF16, 1 sequence | 4.195 | 4.203 | 4.186 | 841 / 1,163 / 1,222 | Pass |
| BF16, 8 sequences | 4.205 | 8.169 | 15.994 | 849 / 1,178 / 1,314 | Pass |
| NVFP4, 1 sequence | 9.069 | 8.948 | 8.955 | 1,380 / 2,139 / 2,061 | Pass |
| NVFP4, 8 sequences | 9.051 | 17.491 | 34.007 | 1,377 / 2,135 / 2,005 | Pass |
| NVFP4 + MTP3, 1 sequence | 15.573 | 14.387 | 14.304 | 1,168 / 1,923 / 1,882 | Pass |
| NVFP4 + MTP3, 8 sequences | 16.081 | 28.360 | **52.415** | 1,173 / 2,071 / 1,890 | Pass |

On this short suite, plain NVFP4 delivered 2.16 times BF16 single-stream
aggregate output. MTP3 raised that to 3.71 times BF16. The three one-sequence
profiles serialized concurrent requests, so their flat concurrency rates are
expected. The NVFP4+MTP eight-sequence profile is a valid scaling result:
concurrency 4 was 1.85 times concurrency 2.

The matched eight-sequence comparison isolates weight precision. NVFP4 was
2.15 times BF16 single-stream, 2.14 times BF16 at concurrency 2, and 2.13 times
BF16 at concurrency 4. Its client-TTFT prefill approximations were 1.62 / 1.81
/ 1.53 times BF16 at the 256 / 2K / 8K targets. Within the NVFP4 run,
concurrency 2 reached 1.93 times single-stream and concurrency 4 reached 3.76
times single-stream; concurrency 4 was 1.94 times concurrency 2. Adding MTP3
to the same eight-sequence NVFP4 setup added another 1.78 / 1.62 / 1.54 times
at single / concurrency 2 / concurrency 4 respectively. Actual NVFP4 prefill
prompt counts were 322 / 2,117 / 8,259 versus BF16's 324 / 2,117 / 8,261. Its
8,284-token needle passed with a 4.112-second TTFT, 34.6% below BF16's nearly
identical probe.

The added BF16 eight-sequence profile held single-stream output essentially
flat while scaling to 1.94 times single-stream at concurrency 2 and 3.80 times
at concurrency 4; concurrency 4 was 1.96 times concurrency 2. This isolates the
effect of scheduler capacity from precision. Its 324 / 2,117 / 8,261-token
prefill probes used client TTFT, and the 8,286-token needle passed with a
6.285-second TTFT.

The BF16 cached-weight process startup took 551.47 seconds and had no
interference annotation. That number includes the complete vLLM bring-up path
rather than isolated weight loading, and no second warm-start replicate was
measured.
Minimum sampled system `MemAvailable` was 18.34 GiB during startup and 16.09
GiB during measured cases, so the profile fit but retained limited memory
headroom. Sampled case telemetry peaked at 90.58 W and 85 °C; it is supporting
capacity evidence, not a board-total energy result.

The matched NVFP4 cached-weight startup took 425.27 seconds, 126.21 seconds
(22.9%) less than the preceding BF16 observation, and had no annotation. These
are single sequential process-start measurements that include full vLLM
bring-up; cache, compilation, page-cache, and thermal state were not reset, so
the difference is not a controlled cold-start claim. NVFP4 minimum sampled
system `MemAvailable` was 16.01 GiB during startup and 14.43 GiB during cases.
That is less headroom than the BF16 observation despite smaller weights,
confirming that host-wide `MemAvailable` cannot substitute for model-resident
memory in a cross-run footprint comparison. Both profiles fit, but both should
be treated as low-headroom configurations.

The MTP3 one-sequence startup interval is invalid because a CPU-only Docker
metadata probe overlapped it. The probe ended before `server_ready`; all seven
measured cases remain valid.

### Other Cached Chat Profiles

| Profile | Single | Concurrency 2 | Concurrency 4 | 8K needle | Interpretation |
| --- | ---: | ---: | ---: | --- | --- |
| GPT-OSS 120B MXFP4 | 34.001 | 51.756 | 93.717 | Fail | Initial one-token prefill and 32-token needle budgets were insufficient; repaired below |
| RLM Qwen3 8B BF16 | 12.986 | 27.190 | 54.396 | Fail | The 32-token needle truncated before the key; repaired below |
| Qwen3 Coder 30B-A3B BF16 | Excluded | 35.836 | 58.033 | Pass | Startup and decode had separate interference annotations; concurrency remained valid |

The initial Qwen Coder single-stream value is excluded because an external
`/metrics` GET overlapped repetition 2. A later clean, matched AR-control run is
reported below and supersedes it for single-stream comparisons.

The GPT-OSS 120B reasoning repair reserved 64 output tokens for prefill and 128
for the needle so low-effort reasoning could finish before the visible answer.
All seven cases then completed:

| Single | Concurrency 2 | Concurrency 4 | Approx. prefill 256 / 2K / 8K | 8K needle |
| ---: | ---: | ---: | ---: | --- |
| 33.398 | 45.488 | 89.951 | 1,196 / 3,858 / 4,252 | Pass |

All nine prefill requests reached a visible one-word final answer with
`finish_reason=stop`; actual prompt counts were 362 / 2,154 / 8,299. The
8,320-token needle used 35 output tokens, returned the complete
`SPARK-5F47DB67E9` key in final content, and stopped naturally. In the original
quick run, the 32-token limit left only a truncated key in visible final
content even though reasoning held the full value, so that exact-answer check
correctly failed. The repaired 128-token case is a correctness fix, not a
direct performance comparison with the truncated case.

Only decode and concurrency retain comparable request shapes across the two
runs, and even their prompts differed by one or two tokens per request. The
repair run's aggregate rates were 1.8% lower single-stream, 12.1% lower at
concurrency 2, and 4.0% lower at concurrency 4. With three decode repetitions,
two concurrency bursts, and no interleaved design, treat this as run-to-run
spread rather than a measured repair penalty. vLLM bundled multiple tokens per
SSE emission in both runs; aggregate output throughput is primary and the
client decode estimates are not used here.

The clean repaired run started from cached weights in 495.17 seconds. Minimum
sampled system `MemAvailable` was 10.56 GiB during startup and 9.94 GiB during
cases, confirming that this 120B profile fits but remains a low-headroom
deployment. As elsewhere, `MemAvailable` is host-wide sampled state, not a
model-footprint measurement.

The RLM Qwen3 8B reasoning repair likewise raised prefill completion budgets to
64 tokens and the needle budget to 128. All seven request cases are valid:

| Single | Concurrency 2 | Concurrency 4 | Approx. prefill 256 / 2K / 8K | 8K needle |
| ---: | ---: | ---: | ---: | --- |
| 12.939 | 27.756 | 54.291 | 1,926 / 3,724 / 4,076 | Pass |

All nine prefill requests returned visible final content and stopped naturally;
their actual prompt counts were 322 / 2,115 / 8,259. The 8,281-token needle
returned the complete `SPARK-D01A8E888C` key in visible content before reaching
the 128-token limit, so exact-answer validation passed; the response itself
later ended with `finish_reason=length`. The original 32-token case truncated
before showing any key and correctly failed. Because the repair changes the
completion budget, the two needle timings are not a matched performance
comparison.

Decode and concurrency retain comparable shapes, with only zero to one prompt
token difference per request. Against the original quick run, repaired
aggregate output was 0.4% lower single-stream, 2.1% higher at concurrency 2,
and 0.2% lower at concurrency 4—effectively unchanged at this sample size.

The recorded 120.80-second RLM startup is diagnostic only. After `run_complete`,
the journal appends a startup-scoped `measurement_annotation`: a source edit
and three CPU-only validation commands overlapped startup, ending before
`server_ready`. No request case overlapped that activity,
`measurement_invalid_cases` is empty, and all seven request measurements remain
valid.

| Ollama Q4_K_M profile | Single | Queued c2 | Queued c4 | 8K needle |
| --- | ---: | ---: | ---: | --- |
| Qwen3 30B-A3B | **67.634** | 64.556 | 66.164 | Pass |
| Nemotron Cascade 2 | 60.311 | 56.303 | 57.397 | Pass |
| GLM 4.7 Flash | 50.970 | 49.210 | 50.598 | Pass |
| Qwen3.5 35B-A3B | 46.678 | 44.209 | 45.155 | Pass |
| Gemma3 12B | 23.248 | 22.900 | 23.214 | Pass |
| Gemma4 31B | 9.307 | 9.208 | 9.239 | Pass |
| Llama3.3 70B | 4.318 | 4.311 | 4.305 | Pass |
| Nemotron 3 Super | Excluded | 14.844 | 14.941 | Pass |
| Mistral Medium 3.5 128B | Excluded | 2.385 | 2.402 | Pass |

Nemotron Super and Mistral did not produce the requested 384 decode tokens
(356 and 324 respectively), so their single-stream cases failed validation.
The other Ollama single-stream rates use native `eval_duration`. All nine
chat-capable Ollama profiles failed the quick prefill cases because a one-token
response omitted `eval_duration`; this is a suite/runtime reporting mismatch,
not a model-fit failure. The targeted two-token repair below resolved all 27
cases. DeepSeek OCR completed no generic chat cases; the targeted OCR sequence
below distinguishes adapter compatibility from recognition accuracy.

#### Ollama Native Prefill Repair

These are medians of three measured requests per target. Each rate uses
Ollama's native `prompt_eval_count / prompt_eval_duration`; it is not the
client-TTFT approximation used for vLLM rows elsewhere in this report.

| Ollama Q4_K_M profile | 256 target | 2K target | 8K target | Cold first load |
| --- | ---: | ---: | ---: | ---: |
| Nemotron Cascade 2 | 1,769.4 | 2,713.4 | **3,082.6** | 6.10 s |
| Qwen3 30B-A3B | **2,145.3** | **2,752.3** | 2,669.5 | 4.48 s |
| GLM 4.7 Flash | 1,832.8 | 2,194.3 | 1,821.5 | 4.54 s |
| Qwen3.5 35B-A3B | 1,216.5 | 1,537.5 | 1,608.7 | 6.49 s |
| Gemma3 12B | 1,773.1 | 1,784.6 | 1,799.6 | 3.72 s |
| Nemotron 3 Super | 433.2 | 650.1 | 739.8 | 15.56 s |
| Gemma4 31B | 647.0 | 651.2 | 622.0 | 5.72 s |
| Llama3.3 70B | 271.5 | 293.3 | 281.7 | 24.30 s |
| Mistral Medium 3.5 128B | 349.2 | 197.2 | 166.0 | 23.14 s |

The column names are suite repetition targets, not common tokenizer counts.
Actual native prompt counts for the first eight models were approximately
305–330 / 2,099–2,122 / 8,241–8,267; Mistral's chat template/tokenizer produced
852 / 2,647 / 8,789. Native timing isolates Ollama's prompt-evaluation phase,
whereas client TTFT also includes request, load, queue, and first-emission
overhead. The two metric families should not be ranked together.

The nine runs were sequential. Each journal marks SparkBench-owned unload,
records `server_stopped` before the next run, and gives the separately issued
`first_request_after_start` a nonzero cold-load duration. The cold-load column
reports that first request; it is not folded into the three measured native
prefill samples. All nine summaries are complete with no failed or invalid
cases. With only `n=3` per prompt size, the table supports medians, not tail
latency or variance claims.

#### DeepSeek OCR: Adapter Repair Versus Accuracy

The F16 `deepseek-ocr:latest` sequence used the same exact-token image fixture
and expected normalized transcription, `SPARKOCR4827`, across three attempts:

| Run directory | Prompt/adapter state | Three-request observation | Outcome |
| --- | --- | --- | --- |
| `results/20260816T054610Z-ollama-deepseek-ocr-f16-ocr-89ee5553` | Initial OCR adapter | Warm-up response omitted native `eval_duration`; the adapter rejected it before measurement | Adapter timing failure; accuracy not scored |
| `results/20260816T055102Z-ollama-deepseek-ocr-f16-ocr-89ee5553` | Repaired adapter, generic instruction | Empty visible content on all three requests | `0/3` exact; validation failed |
| `results/20260816T055210Z-ollama-deepseek-ocr-f16-ocr-89ee5553` | Repaired adapter, exact `Free OCR.` instruction | Stable `SPARKOOR+827` on all three requests | `0/3` exact; validation correctly failed |

The third prompt follows the exact `Free OCR.` instruction shown by the
[Ollama DeepSeek OCR model page](https://ollama.com/library/deepseek-ocr) and
the [upstream DeepSeek-OCR repository](https://github.com/deepseek-ai/DeepSeek-OCR).
All three responses stopped normally, but each substituted `O` for `C` and `+`
for `4`. Stable wrong text is an OCR accuracy failure on this synthetic fixture,
not an adapter failure. Conversely, the first attempt says nothing about model
accuracy: it exposed an overly strict native-timing requirement, which the
second and third runs repaired. The harness then retained empty or incorrect
content and correctly failed exact normalized validation instead of treating a
successful request as a successful transcription.

#### Vision Color and Image-Transport Screen

All five Ollama models completed the three-resolution solid-red-image suite.
Each resolution had three measured requests, and all `5 × 3 × 3 = 45` request
validations returned the expected color (`45/45`). Values are median seconds,
shown as `TTFT / E2E`:

| Model | 64 px | 512 px | 1024 px |
| --- | ---: | ---: | ---: |
| Mistral Medium 3.5 128B | 0.807 / 1.209 | 0.804 / 1.206 | 0.801 / 1.202 |
| Gemma4 31B | 0.347 / 0.454 | 0.322 / 0.430 | 0.337 / 0.442 |
| Qwen3.5 35B-A3B | 0.309 / 0.335 | 0.463 / 0.486 | 0.308 / 0.333 |
| Gemma3 12B | 0.259 / 0.304 | 0.254 / 0.297 | 0.279 / 0.321 |
| DeepSeek OCR F16 | 0.225 / 0.232 | **0.216 / 0.223** | 0.375 / 0.382 |

This validates image payload transport, multimodal request acceptance, and
simple dominant-color output only. It is not an OCR, document-understanding,
spatial-reasoning, or general vision-quality benchmark. In particular,
DeepSeek's color passes do not override the exact-token OCR failure above.

Case journals also expose a material distinction between measured medians and
first use at a new resolution. The approximate pre-measured gap below is
`first request_complete timestamp - first measured burst_elapsed - case_start`;
it contains the unjournaled case warmup and any first-use setup:

| Model | 64 px gap | 512 px gap | 1024 px gap |
| --- | ---: | ---: | ---: |
| Mistral Medium 3.5 128B | 1.513 s | 6.242 s | **134.368 s** |
| Gemma4 31B | 1.036 s | 1.008 s | 1.013 s |
| Qwen3.5 35B-A3B | 0.332 s | 0.511 s | 4.424 s |
| Gemma3 12B | 0.895 s | 0.893 s | 0.930 s |
| DeepSeek OCR F16 | 1.298 s | 1.308 s | 3.018 s |

Mistral's 1024 case began at `05:53:33.137Z`; its first measured completion was
not journaled until `05:55:48.730Z`, 135.593 seconds later. Subtracting that
measured burst's 1.225 seconds leaves about 134.368 seconds before measurement,
while the three measured requests themselves had a 1.202-second median E2E.
The model had already completed the run-level cold first request, so this is a
resolution-case warmup/first-use effect, not model load. Qwen3.5 and DeepSeek
also show smaller 1024-specific gaps; Gemma3 and Gemma4 are nearly flat. The
journal does not isolate preprocessing, runtime compilation, or another cause,
so none should be asserted. For interactive first-use planning, retain these
gaps alongside the intentionally post-warmup median table.

The repaired vLLM Gemma3 4B BF16 follow-up also passed all nine color
validations. The missing `preprocessor_config.json` was restored from
`google/gemma-3-4b-it` at the plan's pinned revision
`093f9f388b31de276ce2de164bdc2081324b9767`; the model source and weights were
not substituted. Its warm medians were nearly resolution-independent:

| Backend/model | 64 px | 512 px | 1024 px |
| --- | ---: | ---: | ---: |
| vLLM Gemma3 4B BF16 | 0.074 / 0.118 | **0.068 / 0.111** | 0.071 / 0.116 |

Each three-request case reported 852 aggregate prompt tokens (284 per request),
regardless of source image size. Startup was substantial: telemetry contains
149 startup samples, while the journal records 153.084 seconds wall-clock to
`server_ready`. Warm latency was lower than the Ollama Gemma3 12B row above,
but this is not a matched backend result: model size, serving stack, chat/image
processor, and token accounting all differ. It does not isolate a vLLM versus
Ollama speedup.

The three initial matrix startup failures were diagnostic: Gemma3 4B lacked
`preprocessor_config.json`; the cached Nemotron Diffusion 8B snapshot lacked
its remote modeling module; and the 14B diffusion checkpoint was resolved by
vLLM as pooling rather than diffusion generation. The supported direct 14B
path succeeded later in the campaign.

#### Gemma3 4B vLLM Capability Check

The repaired BF16 profile then completed the `capabilities` suite. These are
single requests without case warmups, so they establish correctness and a
latency observation, not a distribution:

| Case | Prompt / output tokens | TTFT | E2E | Aggregate output | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| JSON correctness | 78 / 14 | 0.766074 s | 1.352489 s | 10.241865 tok/s | Pass |
| 4K needle | 4,186 / 12 | 0.491729 s | 1.017648 s | 11.618835 tok/s | Pass |

JSON returned the exact object `{"benchmark": "spark", "value": 42}`. The
4K request returned the exact `SPARK-7CC32E68DE` key and stopped naturally.
The tools case was skipped before server startup because this profile does not
advertise the required `tools` capability; it is unsupported, not a failed
tool-call attempt. Startup to `server_ready` was 118.778897 seconds, followed
by a 0.405372-second first request.

## Speech Synthesis: Spark-TTS GPU

Spark-TTS required a four-attempt compatibility repair before producing audio:

| Temporary run | Runtime change | Result |
| --- | --- | --- |
| `/tmp/sparktts-bench.OuHJwV` | None | Complex-absolute requested unsupported NVRTC target `sm_121`; no trial |
| `/tmp/sparktts-bench.mHsEOu` | First spectrogram wrapper | Wrapper signature mismatch; no trial |
| `/tmp/sparktts-bench.8o2Bne` | Corrected real/imag magnitude fallback | TorchScript-fused Snake then requested unsupported `sm_121`; no trial |
| `/tmp/sparktts-bench.u6H1uC` | Real/imag square-root magnitude plus eager Snake | Load and all three trials completed |

The successful run loaded in 23.645655 seconds. Each seeded trial generated
4.42 seconds of audio and reached the same 4,345,788,416-byte (4.047 GiB) peak
Torch allocation:

| Trial | Inference | RTF | Real-time rate | Output SHA-256 |
| ---: | ---: | ---: | ---: | --- |
| 0, first use | 5.018990 s | 1.135518 | 0.880655× | `b1f4a431f018664d9aaca12873baf7fdf2b89fdb6e44ac0ee0704b8ee311a95c` |
| 1, warm | 3.770920 s | 0.853149 | 1.172128× | `b1f42b27a31202877c47ee23c529fcc475dddc71c8eb56957d954458fc47719f` |
| 2, warm | **3.760285 s** | **0.850743** | **1.175443×** | `e1fdd6e516593b1a892ab174b10c6c9e36c5e53b654ad782c0f703e504c855a4` |

Thus the two warm trials were narrowly faster than real time: RTF
0.850743–0.853149, or 1.172128–1.175443× real time. The spectrogram fallback
computes magnitude as `sqrt(real² + imag²)`; the Snake fallback expands the
same real-valued operations eagerly. Both are runtime monkeypatches and did not
further edit the checkout. However, the source was already locally modified at
commit `2f1ea9082400547242641f5271b6f941c9f439d1` (`cli/SparkTTS.py` and
`requirements.txt`), so this is not a pristine-upstream measurement. Principal
weight hashes were BiCodec
`e9940cd48d4446e4340ced82d234bf5618350dd9f5db900ebe47a4fdb03867ec`,
LLM `54825baf0a2f6076eb3c78fa1d22a95aee225f59070a8b295f8169db860eb109`,
and Wav2Vec2
`314340227371a608f71adcd5f0de5933824fe77e55822aa4b24dba9c1c364dcb`.

PyTorch 2.10.0+cu128 warned that its compiled capability support ends at 12.0
while GB10 reports 12.1. The final fallbacks demonstrate successful execution,
not native SM121 kernel coverage or a speech-quality judgment; the distinct
seeded WAV hashes establish retained outputs, not perceptual equivalence.

## Speech Recognition: Whisper PyTorch GPU

The offline Whisper run used the same 9.953313-second, 16 kHz mono Chinese
fixture documented in `docs/cached-media-capabilities-2026-08-15.md`, SHA-256
`335e7f7789b231cd90d9670292d561ecfe6a6bdd5e737a7bc6c29730741852de`.
All eight records are present: first-use trial 0 and warm trial 1 for each of
four multilingual checkpoints. The table reports warm trial 1; load happened
once before the pair:

| Model | Load | Warm inference | RTF | Real-time rate | CER | Peak Torch allocation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 1.260 s | **0.781 s** | **0.07846** | **12.746×** | 34.55% | **0.358 GiB** |
| small | 1.967 s | 1.962 s | 0.19716 | 5.072× | 25.45% | 1.056 GiB |
| medium | 4.817 s | 5.413 s | 0.54382 | 1.839× | 3.64% | 3.137 GiB |
| large-v3-turbo | 5.177 s | 1.913 s | 0.19222 | 5.202× | **0.00%** | 3.263 GiB |

`large-v3-turbo` was the only exact transcript and remained 5.2× faster than
real time; the smaller models traded accuracy for less Torch allocation.
Trial 0 retains first-use overhead and is not mixed into the warm table. Peak
Torch allocation is `torch.cuda.max_memory_allocated`, not total unified-memory
use. PyTorch emitted a compatibility warning for the GB10's compute capability
12.1 (`sm_121`). All CUDA trials nevertheless completed, but these observations
do not prove that the installed build used fully optimized SM121-native kernels.

## Fine-Tuning: SmolLM2 LoRA with Unsloth

Three offline attempts separated runtime compatibility from artifact handling:

| Temporary run | Outcome |
| --- | --- |
| `/tmp/smollm2-unsloth.Va7qUG` | PEFT's optional TorchAO dispatcher rejected the pinned TorchAO 0.14 API; no benchmark metrics |
| `/tmp/smollm2-unsloth.GjjBcx` | Runtime TorchAO availability bypass enabled inference, 16 updates, and save; host hashing could not read root-owned mode-0600 `adapter_model.safetensors` |
| `/tmp/smollm2-unsloth.xYctAZ` | Same narrow BF16 LoRA bypass, successful compute, and mode-0644 readable adapter files with complete hashes |

The final run pinned `HuggingFaceTB/SmolLM2-135M` revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2` and model-weight SHA-256
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
inside image
`sha256:98261f554d5061eb8e3c05a94689d212567fa9d565c861539ed1c0ed61a96720`.
The runtime bypass only makes PEFT report TorchAO unavailable for this ordinary
BF16 LoRA path; it does not replace the base linear layers or mutate packages.

Load took 2.489498 seconds. Raw greedy inference generated 64 tokens from 13
prompt tokens in 1.875807 seconds, or 34.118657 generated tok/s. Rank-8,
alpha-8 LoRA setup took 0.931184 seconds and exposed 2,442,240 trainable of
136,957,248 total parameters (1.783%). Training results were:

| Steps / shape | End-to-end | Step-0-excluded steady rate | Loss | Peak Torch memory |
| --- | ---: | ---: | ---: | ---: |
| 16; batch 4 × sequence 256 | 3.088630 s; 5,283.896 predicted tok/s | 15,263.633 predicted tok/s | 0.924001 → 0.381207 | 921.84 MiB allocated / 1,018 MiB reserved |

All 16 losses were finite and decreased monotonically; the fitted slope was
-0.037266 per step. Step 0 took 2.085459 seconds, versus roughly 0.065–0.077
seconds thereafter, hence the required end-to-end and compilation-excluded
views. The fixture contributed 16,320 predicted tokens and is pinned by
SHA-256 `b420d5ba89b11d2c35f9b7114c5bb1b00c4ed9efbba487374d15390883db3af8`.
Load, inference, and LoRA-setup peak allocations were 258.59, 280.45, and
288.03 MiB respectively. After `empty_cache`, Torch still reported 232.63 MiB
allocated and 422 MiB reserved; the artifact records CUDA-context release on
container exit. These are Torch allocator figures, not total unified memory.

Saving took 0.184095 seconds. `SHA256SUMS` verifies every final adapter and
tokenizer artifact. The two principal adapter hashes are
`adapter_model.safetensors`:
`b7ad28b23da8117d0f347d9cdd024a0e9de6b410c8cfeb0988a3cf53c84f2c87`
and `adapter_config.json`:
`8c9c386bdca58dba8cd73321029d0ba77940a530f6bcd366b2229382f6ade01d`.
The retained adapter files are host-readable but remain root-owned; no cleanup
was performed on the temporary output. Finally, the repeated synthetic fixture
and falling loss validate the training/save plumbing only—not generalization,
alignment, or useful fine-tuning quality.

## Newly Acquired Text Models

### Qwen3 8B: NVFP4 Versus FP8

| Quantization | Single | Concurrency 2 | Concurrency 4 | Approx. prefill 256 / 2K / 8K | 8K needle |
| --- | ---: | ---: | ---: | ---: | --- |
| NVFP4 | Excluded | 77.444 | **149.272** | 6,625 / 10,904 / 7,661 | Pass |
| FP8 | 24.477 | 49.698 | 97.981 | 4,429 / 8,150 / 7,019 | Pass |

NVFP4 exceeded FP8 aggregate throughput by 55.8% at concurrency 2 and 52.3%
at concurrency 4. Its raw single-stream result is not used: a read-only host
inventory probe overlapped the measured decode case. The later concurrency,
prefill, and needle cases are valid. The FP8 run completed all seven cases
without annotations.

GPT-OSS 20B MXFP4 completed the reasoning-aware quick suite cleanly. Aggregate
output was 47.236 single-stream, 77.962 at concurrency 2, and 159.678 at
concurrency 4; concurrency 4 was 2.05 times concurrency 2. Approximate prefill
was 2,420 / 6,158 / 6,652 at the 256 / 2K / 8K targets. Its 8,320-token needle
case passed, returning 40 output tokens. SSE emissions were bundled, so the
aggregate rates—not client decode estimates—are the comparison values.

### Qwen3 Coder DFlash15

The matched AR control and DFlash15 profiles used the same pinned Qwen3 Coder
30B-A3B BF16 checkpoint and served context.

| Profile | Single | Concurrency 2 | Concurrency 4 | Approx. prefill 256 / 2K / 8K | 8K needle |
| --- | ---: | ---: | ---: | ---: | --- |
| AR control | 28.700 | **42.154** | 70.695 | **1,631 / 4,769 / 5,141** | Pass |
| DFlash15 | **31.408** | 39.798 | **98.103** | 1,310 / 3,971 / 4,461 | Pass |

DFlash improved single-stream aggregate output by 9.4% and concurrency-4 by
38.8%, but reduced concurrency-2 by 5.6% and each measured prefill tier by
13–20%. Its cumulative vLLM counters, whose scope includes prime, warm-up, and
measured requests, recorded 445 drafts, 6,675 draft tokens, 1,000 accepted
tokens, a 14.98% draft-token acceptance rate, and mean accepted length 3.247.
These counters must not be interpreted as measured-case-only acceptance.

## Embedding and Multimodal Retrieval

| Model and path | Dimensions | Batch 1 | Batch 8 | Batch 32 | Longer-input result |
| --- | ---: | ---: | ---: | ---: | --- |
| all-MiniLM-L6-v2, vLLM | 384 | 197.16 items/s | 737.92 | **1,050.68** | 512-repeat case correctly context-limited by the 256-token serving profile |
| nomic-embed-text F16, Ollama | 768 | 60.24 items/s | 169.21 | 201.03 | 37.03 items/s at length 512, batch 1 |

Qwen3-VL-Embedding-2B produced finite, normalized 2,048-dimensional image and
text vectors. On the five measured synthetic red-image requests, median
throughput was 43.34 items/s and median image latency was 24.64 ms. The relevant
text/image similarity was 0.7341 versus 0.2799 for the unrelated text, a 0.4541
margin. This validates the cross-modal adapter and fixture, not broad retrieval
quality.

Qwen3-VL-Reranker-2B ranked the red-image candidate first on every measured
request with finite scores. It processed 10 measured pairs at 42.00 aggregate
pairs/s; the median per-request rate was 76.15 pairs/s. The first post-start
request scored the relevant image 0.7744 versus 0.2248–0.2807 for the other
candidates. This is a deterministic capability check, not an IR benchmark.

## Direct Diffusion-Language Generation

The pinned Nemotron Labs Diffusion 14B Transformers adapter completed both
block-generation cases with a 140.62-second load and 25.23 GiB peak allocated
CUDA memory during generation.

| Output/request | Aggregate output | Blocks | NFE | NFE/block | Stable across 3 repetitions |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128 tokens | 2.810 tokens/s | 12 | 243 | 20.250 | Yes |
| 256 tokens | 5.081 tokens/s | 24 | 267 | 11.125 | Yes |

The worker was reaped successfully and its post-delete allocation fell to
8.13 MiB. These are end-to-end block-generation measurements; they are not
autoregressive decode TPS, TPOT, ITL, or TTFT.

## Extended Core and Capability Campaign

The post-quick campaign added 21 `core`/`reasoning-core` runs, one Qwen3.6
vision run, and seven focused capability/quality runs. Rates below are valid
aggregate output tokens per measured case wall time. A dash means completion-
length validation failed, so the raw rate is not ranked.

### Qwen3.6 35B-A3B NVFP4 MTP3

The core run completed 14 clean cases. Decode reached 94.846 tok/s at 256
tokens and 106.694 at 1,024; C1/C2/C4/C8 aggregate throughput was 97.489 /
146.479 / 257.517 / 388.017 tok/s. Client-TTFT prefill approximations were
2,095 / 5,237 / 6,408 / 5,955 prompt tok/s at 128 / 1K / 4K / 16K. The 32K
case was context-limited. JSON, the 16K needle, all four exact-answer items,
and core vision passed; embeddings/reranking were unsupported. Tool calling
hit an HTTP 500 because the stack could not import `normalize_tool_choice`
from `xgrammar`, an integration failure rather than a failed tool answer.

The MTP3 server-lifetime snapshot (prime, warmups, and measurements) recorded
19,336 accepted of 30,270 draft tokens (63.878%), mean accepted length 2.916.
Its 243.751-second core startup is diagnostic only: a read-only audit ended
before `server_ready` and touched no request case; all case measurements remain
valid. The separate vision run passed all nine requests:

| Resolution | Prompt tokens/request | Median TTFT | Median E2E |
| ---: | ---: | ---: | ---: |
| 64 px | 93 | 0.117677 s | 0.145805 s |
| 512 px | 285 | 0.154267 s | 0.183141 s |
| 1024 px | 1,053 | 0.251772 s | 0.281237 s |

Vision startup was a valid 116.827 seconds. Its small MTP sample accepted 40/45
draft tokens (88.889%, mean 3.667). This red-image suite validates transport
and dominant-color output, not general vision or OCR.

### vLLM Core Throughput and Prefill

| Model/profile | D256 | D1024 | C1 | C2 | C4 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B NVFP4 MTP3 | 94.846 | 106.694 | 97.489 | 146.479 | 257.517 | 388.017 |
| Qwen3.8 27B BF16 | 4.218 | — | 4.219 | 8.229 | 16.232 | 31.498 |
| Qwen3.8 27B NVFP4 | 9.098 | 9.113 | 9.090 | 18.109 | 35.079 | 67.326 |
| Qwen3.8 27B NVFP4 MTP3 | 16.669 | 19.418 | 16.737 | 34.045 | 64.966 | 112.610 |
| GPT-OSS 20B / 120B MXFP4 | 48.134 / 34.399 | 48.400 / 34.788 | 48.109 / 34.370 | 86.547 / 58.749 | 158.262 / 95.902 | 275.218 / 158.329 |
| Qwen3 8B NVFP4 / FP8 | 37.005 / 24.402 | 36.971 / — | 37.006 / 24.369 | 78.490 / 51.022 | 156.110 / 100.994 | 306.310 / 199.226 |
| RLM Qwen3 8B BF16 | 13.110 | — | 13.123 | 28.042 | 55.795 | 110.147 |
| Coder 30B AR / DFlash15 | 29.254 / 31.563 | — / — | 29.274 / 39.419 | 46.667 / 54.032 | 70.887 / 89.382 | 107.682 / 149.681 |
| Gemma3 4B BF16 | 22.333 | 22.095 | 22.391 | 55.088 | 108.424 | 212.052 |

| Model/profile | Prefill 128 / 1K / 4K / 16K / 32K prompt tok/s |
| --- | --- |
| Qwen3.6 MTP3 | 2,095 / 5,237 / 6,408 / 5,955 / CL |
| Qwen3.8 BF16 | 595 / 1,266 / 1,216 / 1,221 / CL |
| Qwen3.8 NVFP4 | 1,100 / 1,926 / 2,089 / 1,957 / CL |
| Qwen3.8 NVFP4 MTP3 | 799 / 1,176 / 2,044 / 1,781 / CL |
| GPT-OSS 20B / 120B | 1,679 / 5,173 / 6,983 / 5,301 / CL; 872 / 2,937 / 4,392 / 3,613 / CL |
| Qwen3 8B NVFP4 / FP8 | 4,268 / 10,795 / 9,324 / 6,189 / CL; 3,297 / 8,066 / 7,851 / 5,885 / CL |
| RLM Qwen3 8B | 1,215 / 3,863 / 3,940 / 3,586 / 2,884 |
| Coder AR / DFlash15 | 1,043 / 3,647 / 5,103 / 4,187 / CL; 836 / 2,888 / 4,116 / 3,513 / CL |
| Gemma3 4B | 3,402 / 8,254 / 8,761 / 8,773 / CL |

These are client-TTFT approximations, not server-native prefill counters.
Matched core results confirm the quick-run direction: Qwen3.8 NVFP4 beat BF16
by 114–120% across valid D256/C1–C8 cells, MTP3 added 67–113% over NVFP4,
Qwen3 8B NVFP4 beat FP8 by 52–55%, and Coder DFlash beat AR by 7.9% at D256
and 15.8–39.0% at C1–C8. Speculation did not improve Qwen3.8 prefill. Qwen3.8
MTP accepted 19,736/29,699 draft tokens (66.453%, mean 2.994); DFlash accepted
20,592/108,977 (18.896%, mean 3.830), including prime and warmups.

| Model/profile | JSON | Tool | 16K needle | Exact quality | Vision |
| --- | --- | --- | --- | --- | --- |
| Qwen3.6 MTP3 | Pass | Stack error | Pass | 4/4 | Pass |
| Qwen3.8 BF16 / NVFP4 / MTP3 | Pass | Pass | Pass | 3/4 / 3/4 / 2/4 | Unsupported |
| GPT-OSS 20B / 120B | Pass | Pass | Pass | 3/4 / 4/4 | Unsupported |
| Qwen3 8B NVFP4 / FP8 | Pass | Stack error | Fail | 3/4 / 3/4 | Unsupported |
| RLM Qwen3 8B | Pass | Unsupported | Pass | 3/4 | Unsupported |
| Coder AR / DFlash | Pass | Stack error | Pass | 3/4 / 3/4 | Unsupported |
| Gemma3 4B | Pass | Unsupported | Pass | 2/4 | Pass |

The other vLLM tool errors were the same `xgrammar` import failure. The exact-
answer screen has only four deterministic items and is not a general score.

### Ollama Core Sweep

| Q4_K_M model | D256 | D1024 | C1 | C2 | C4 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 30B-A3B | 71.113 | — | 70.963 | 72.363 | 72.759 | 72.630 |
| Nemotron Cascade 2 | 63.825 | — | 63.633 | — | 64.875 | — |
| Qwen3.5 35B-A3B | 49.747 | — | 50.087 | 50.433 | 50.656 | 51.019 |
| Nemotron 3 Super | — | — | — | — | — | 18.185 |
| Gemma4 31B | 9.474 | 9.399 | 9.474 | 9.492 | 9.501 | 9.510 |
| Llama3.3 70B | — | — | 4.363 | — | — | — |
| GLM 4.7 Flash | 54.013 | 54.288 | 54.224 | 54.733 | 55.055 | 55.217 |
| Gemma3 12B | 23.706 | — | 23.856 | 24.002 | 24.093 | 24.131 |
| Mistral Medium 3.5 128B | — | — | 2.453 | — | — | — |

Flat C1–C8 rates again confirm serialized Ollama queues, not batch scaling.
Missing cells stopped before every request reached its token budget. All 36
prefill cases failed at warmup because one-token replies omitted
`eval_duration`; the earlier two-token native-prefill repairs remain the valid
evidence. Each 32K target was plan-limited by configured context.

| Q4_K_M model | JSON | Tool | 16K needle | Exact quality | Vision |
| --- | --- | --- | --- | --- | --- |
| Qwen3 30B-A3B | Pass | Pass | Fail | 3/4 | Unsupported |
| Nemotron Cascade 2 | Pass | Pass | Pass | 2/4 | Unsupported |
| Qwen3.5 35B-A3B | Pass | Pass | Pass | 3/4 | Pass |
| Nemotron 3 Super | Pass | Pass | Pass | 4/4 | Unsupported |
| Gemma4 31B | Fail | Pass | Pass | 3/4 | Pass |
| Llama3.3 70B | Pass | Pass | Pass | 2/4 | Unsupported |
| GLM 4.7 Flash | Fail | Pass | Fail | 3/4 | Unsupported |
| Gemma3 12B | Pass | Unsupported | Pass | 2/4 | Pass |
| Mistral Medium 3.5 128B | Pass | Pass | Pass | 3/4 | Pass |

Qwen3 30B and GLM missed the 16K needle on all three 32-token completions.
The focused short runs agreed with later core results: GLM passed tool/4K
needle but failed strict JSON because it fenced the correct object, and scored
3/4 quality (code miss); Mistral passed JSON/tool/4K and scored 3/4 (code);
Gemma3 12B passed JSON/4K with tools unsupported and scored 2/4 (arithmetic,
code); Gemma3 4B scored 2/4 (instruction, code). Median quality E2E was 0.327,
3.018, 0.438, and 0.298 seconds respectively. These screens are too small for
broad quality claims.

Exact new evidence directories span
`results/20260816T062133Z-qwen36-35b-a3b-nvfp4-mtp3-core-795163dc` through
`results/20260816T143735Z-ollama-mistral-medium-3.5-128b-q4-k-m-core-68dcaf64`;
all 29 are enumerated in the ledger below.

### Phi-4 Multimodal: SGLang FP8 Usable, NVFP4 Invalid

The seven-run Phi sequence separates server compatibility from output
correctness. The vLLM 26.07 NVFP4 attempt, pinned to checkpoint revision
`617cfabb9ad6c2c6e318fd21c1961536b84f65a1`, aborted during load with the
observed tied-weight `NotImplementedError`; its retained bounded error tail
starts at the final bare exception, and it reached neither `server_ready` nor a
request case. The first SGLang NVFP4 attempt also stopped before serving, this
time at `Phi4MMConfig` because Transformers exposed the checkpoint's legacy
LongRoPE fields while SGLang expected `rope_parameters`.

The subsequent frozen plans re-expressed the checkpoint-derived 48-element
`long_factor` and `short_factor` arrays as `type=longrope` with
`rope_theta=10000.0` through `--json-model-override-args`; both `plan.json` and
`server/provenance.json` retain the exact value. They used the same multimodal
checkpoint ID, `--trust-remote-code`, and SGLang's inferred `phi-4-mm` template,
not a separate image model or custom template. This matches NVIDIA's
[DGX Spark SGLang instructions](https://build.nvidia.com/spark/sglang/instructions)
and SGLang's exact
[v0.5.10rc0 Phi4MM implementation](https://github.com/sgl-project/sglang/blob/v0.5.10rc0/python/sglang/srt/models/phi4mm.py);
NVFP4 used the documented `modelopt_fp4` loader.

With that schema migration, NVFP4 became ready in 80.531 seconds but remained
semantically unusable. Its cold first response and measured chat response were
gibberish. The chat case's permissive nonempty validator passed and recorded
48.118 aggregate tok/s (52.764 client-estimated decode tok/s), but it is not a
valid quality or performance result. The red-image request also produced
gibberish and correctly failed. SGLang warned that the checkpoint scale format
was not `ue8m0` and “might cause accuracy degradation” on Blackwell; the run
does not prove that warning caused the corruption.

FP8, at pinned revision `d822efce23f65f86c165aeed435cc27092e21d60`,
served coherent text and correct red-image answers:

| FP8 run | Startup | Cold first TTFT | Result |
| --- | ---: | ---: | --- |
| Smoke | 88.545 s | 21.582 s | Chat 38.006 aggregate tok/s; image pass, each n=1 |
| Vision | 82.542 s | 19.226 s | 9/9 measured image requests passed |
| Quick | 82.538 s | 19.293 s | Six cases complete; 8K prefill warmup and needle failed |
| Reasoning core | 88.568 s | 21.329 s | Eleven cases complete; two prefill warmups and three validations failed |

Core decode and concurrency results were:

| Workload | Requests | Aggregate tok/s | Median client decode | Median TTFT | Median E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| D256 | 5 | 38.858 | 39.061 | 0.041618 s | 6.569755 s |
| D1024 | 5 | — | — | 0.043440 s | 26.571468 s |
| C1 | 5 | 38.712 | 38.835 | 0.041611 s | 6.607818 s |
| C2 | 10 | 83.836 | 42.226 | 0.048170 s | 6.092489 s |
| C4 | 20 | 164.202 | 41.451 | 0.065948 s | 6.217570 s |
| C8 | 40 | 318.351 | 40.341 | 0.073751 s | 6.395616 s |

D1024 is deliberately unranked: one request stopped at 649 tokens, leaving
4,745/5,120 requested tokens. Relative to C1, aggregate C2/C4/C8 scaling was
2.166x / 4.242x / 8.224x while per-request median decode stayed 38.835–42.226
tok/s. The shorter quick run agreed: D128 was 38.740 aggregate tok/s and C2/C4
were 82.404 / 161.853 tok/s.

Prefill evidence is a client-TTFT approximation, not a server-native counter.
Combining only the valid quick/core cases gives 3,073 / 5,070 / 10,445 / 11,015
prompt tok/s at 128 / 256 / 1K / 2K targets (n=5/3/5/3; actual prompt tokens
per request 170/300/1,066/2,089). The 4K, 8K, and 16K warmups emitted neither
content nor reasoning, so they have no rate; 32K was plan-context-limited. The
quick 8K needle returned repeated `archive` tokens (8,257 prompt tokens,
TTFT/E2E 1.123798/2.063852 seconds) and failed. All three core 16K needles did
the same (49,343 total prompt and 384 output tokens; median TTFT/E2E
2.516627/7.041738 seconds). Exact-answer quality was 2/4: arithmetic and logic
passed; instruction following and code reasoning failed. JSON, tools,
embeddings, and reranking were unsupported by this profile.

The dedicated vision medians were:

| Resolution | Prompt tokens/request | Median TTFT | Median E2E | Aggregate output tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 64 px | 564 | 0.057074 s | 0.083246 s | 17.139 |
| 512 px | 1,348 | 0.073107 s | 0.100186 s | 17.489 |
| 1024 px | 2,644 | 0.094290 s | 0.121878 s | 13.598 |

Each row is n=3 and passed 3/3. Core's separate n=3 red-image check also
passed (564 prompt tokens/request; median TTFT/E2E 0.061663/0.088970 seconds).
These tests validate image transport and dominant-color recognition, not OCR
or general visual reasoning.

Startup telemetry sampled minimum host-wide `MemAvailable` of 51.651 GiB for
the ready NVFP4 run and 52.306–52.876 GiB for FP8. These are neither GPU-only
allocation nor model footprint; the two early aborts retained much higher
105.013/111.238 GiB minima and are not comparable. Startup spans 78–86 samples,
but several short vision/prefill cases have zero or one telemetry sample and
therefore no defensible energy estimate. Warm case sizes are also small, and
the ready runs' 18.5–21.6-second cold first TTFTs are excluded from their
medians.

Finally, NVIDIA's [NVFP4](https://huggingface.co/nvidia/Phi-4-multimodal-instruct-NVFP4)
and [FP8](https://huggingface.co/nvidia/Phi-4-multimodal-instruct-FP8) model
cards specify the NVIDIA Open Model License and deployment geography as global
except the European Union. That is a deployment/license constraint to review,
not a benchmark finding.

### Phi-4 FP8 Audio: TensorRT-LLM Pipeline Pass, Exact-ASR Miss

The final offline audio run used NVIDIA's
[TensorRT-LLM for DGX Spark instructions](https://build.nvidia.com/spark/trt-llm/instructions)
and the tagged
[v1.3.0rc13 multimodal LLM-API example](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc13/examples/llm-api/quickstart_multimodal.py).
It pinned
[`nvidia/Phi-4-multimodal-instruct-FP8`](https://huggingface.co/nvidia/Phi-4-multimodal-instruct-FP8)
revision
`d822efce23f65f86c165aeed435cc27092e21d60` and image
`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` at digest
`sha256:4f30c464ead64fb9727a24064b25057dacc07bef848022421108e544c91f0965`.
The container reported TensorRT-LLM 1.3.0rc13, PyTorch
2.11.0a0+eb65b36914.nv26.02, and Transformers 4.57.3.

The 9.953313-second Chinese fixture received one warmup and three measured
requests. The LLM constructor took 92.467995 seconds and multimodal input
preparation took 0.790822 seconds. Measured results, independently recomputed
from `request_complete` events and the 5.255344-second measured worker wall
time, were (the 8.370614-second warmup is excluded):

| Requests | Tokens/request | Median latency | Median output rate | Aggregate output rate | Median RTF |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 53 | 1.750606 s | 30.275 tok/s | 30.255 tok/s | 0.175882 |

An RTF of 0.175882 is about 5.69 times faster than the clip duration, but it
does not imply an accuracy pass. All three measured requests stopped normally
and produced identical output, normalized-output, and token-ID hashes. The
warmup produced those same hashes. Each output normalized to 54 characters
against the 55-character reference: character edit distance was exactly 1 and
CER was `1/55 = 1.818%` on every trial. Exact transcription therefore scored
`0/3`; the summary is correctly `partial`, with three per-request
`transcription_mismatch` validations and a terminal `run_complete`, not an
infrastructure abort. This is a successful, deterministic audio pipeline with
a stable one-character quality miss on one fixture, not evidence of general
ASR quality.

The bounded compatibility progression explains why earlier artifacts are not
benchmark results:

| Run timestamp (UTC) | Terminal observation | Interpretation |
| --- | --- | --- |
| 17:43:41 | SGLang rejected the speech LoRA because it adds vocabulary tokens | Backend incompatibility; no ready server or request |
| 18:06:52 | The container UID could not create the result artifact | Harness permission failure; no trustworthy worker result |
| 18:08:05 | Overriding the image entrypoint omitted `libnvonnxparser.so.10` from the runtime environment | Container launch failure |
| 18:10:11 | The executor tried to create `/root/.triton` on a read-only root | Cache-routing failure after model loading began |
| 18:12:36 | A hand-authored special-token prompt yielded zero audio tokens for 125 embeddings | Input-contract failure; the official loader must own templating |
| 18:19:31 | One warmup and three requests completed, but the old validator turned the exact miss into `run_aborted` | Harness classification failure, repaired by the final run |

The 18:16:54 launch has no result, cleanup, or terminal journal event and is
excluded rather than inferred. Every terminal direct attempt recorded a reaped
worker and an absent container. The final worker returned zero, did not time
out, required no termination or kill, and confirmed the auto-removed container
absent. Its 120 worker-phase telemetry samples span 122.430 seconds: average
GPU utilization/power were 11.825%/16.773 W, peak power/temperature were
74.03 W/68 °C, peak SM clock was 2,535 MHz, trapezoidal sampled energy over
119 intervals was 2,062.491 J, and minimum host-wide `MemAvailable` was
78.499 GiB. There were no GPU-query errors or missing power samples. The lone
idle sample was 2% utilization, 6.47 W, and 48 °C, so it is context rather than
a measured baseline. These samples cover the whole worker lifecycle, not just
measured ASR. Model allocation is deliberately reported unavailable:
TensorRT-LLM allocates in an executor child outside the driver's PyTorch
allocator, and host `MemAvailable` is not model-resident memory.

For privacy, retained plans, journals, and summaries contain fixture/output
SHA-256 identities, token counts, normalized lengths, edit distances, and CER,
but neither audio bytes nor transcript text. The audio and model repositories
were read-only inputs; only the isolated worker artifact directory was
writable, and the container had no network.

### Phi-4 Reasoning Plus FP8 on vLLM

Two annotation-free runs used `nvcr.io/nvidia/vllm:26.07-py3` and pinned
`nvidia/Phi-4-reasoning-plus-FP8` revision
`18abf8a59bd8ff0b79ec712863a153becc6cdaeb`. Smoke startup took 219.662
seconds; its one measured chat request produced 32 tokens at 14.016 aggregate
tok/s, with median client-estimated decode 14.187 tok/s, TTFT 0.083505 seconds,
and E2E 2.268653 seconds. That n=1 request passed the generic nonempty
validator, but exhausted its budget while emitting `<think>` content and did
not establish answer quality. JSON, tools, vision, embeddings, and reranking
were profile-declared unsupported and were not exercised.

The reasoning-core startup took 213.628 seconds. Its valid token-throughput
results were:

| Workload | Requests | Aggregate tok/s | Median client decode | Median TTFT | Median E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| D256 | 5 | 14.574 | 14.594 | 0.086269 s | 17.558701 s |
| D1024 | 5 | 14.572 | 14.571 | 0.084299 s | 70.293364 s |
| C1 | 5 | 14.571 | 14.568 | 0.084460 s | 17.588946 s |
| C2 | 10 | 28.822 | 14.501 | 0.090400 s | 17.722485 s |
| C4 | 20 | 56.588 | 14.241 | 0.175578 s | 18.054482 s |
| C8 | 40 | 112.903 | 14.211 | 0.200381 s | 18.080803 s |

C2/C4/C8 delivered 1.978x / 3.884x / 7.749x C1 throughput, or 96.9% of
ideal eight-way scaling at C8. Every decode/concurrency request reached its
token budget and ended `length`, but every response began with `<think>` while
the API's separate `reasoning` field stayed empty. These are real generated-
token rates, not evidence that the model reached a user-visible final answer.

The valid prefill cases were all client-TTFT approximations:

| Target | Requests | Total prompt tokens | Median prompt tok/s | Median TTFT | Median E2E |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 5 | 1,987 | 4,046.763 | 0.098150 s | 4.419469 s |
| 1K | 5 | 6,483 | 4,681.538 | 0.277046 s | 0.555705 s |
| 4K | 5 | 21,829 | 4,850.832 | 0.900052 s | 5.323033 s |
| 16K | 5 | 83,264 | 3,701.514 | 4.498970 s | 9.266610 s |

The 32K target was context-limited by the 32,768-token serving cap plus output
budget. Output behavior also limits interpretation: every 1K request stopped
after five tokens (`: assistant`), while 4K/16K responses filled 64 tokens by
repeating `measurement`. TTFT remains measurable, but this is not a content-
quality pass.

Both evaluated quality paths failed. All three 16K needles returned 128
repeated `archive` tokens instead of the key (50,021 total prompt tokens;
median TTFT/E2E 4.515728/14.111916 seconds). Exact-answer quality was 0/4:
each item consumed all 64 tokens on a truncated `<think>` trace and never
reached the required final answer. The score therefore describes this output
budget and parsing contract, not necessarily the model with a larger reasoning
budget. Core JSON, tools, vision, embeddings, and reranking were likewise
profile-declared unsupported.

Both startups are marked valid and have no measurement annotations. Their
minimum host-wide `MemAvailable` was 43.856/43.272 GiB, and cold first-request
TTFT was 0.212686/0.144292 seconds for smoke/core. Startup telemetry had
212/206 samples, but the smoke and 1K-prefill cases had only two samples each;
short-case power and energy are correspondingly coarse. Direct comparison with
the SGLang multimodal Phi runs is inappropriate because both the model and
serving backend changed.

### Qwen3.8 27B Unsloth Q4 GGUF on Managed llama.cpp

Six annotation-free native-subprocess runs completed for controlled smoke,
quick, and core baseline/MTP comparisons. Both profiles used the same cached
[`unsloth/Qwen3.8-27B-GGUF` snapshot](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe),
revision `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`, and
`Qwen3.8-27B-UD-Q4_K_XL.gguf`: 17,923,394,624 bytes at
`sha256:bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`.
The native runtime was
[`llama.cpp` b10453](https://github.com/ggml-org/llama.cpp/releases/tag/b10453),
commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`; its 58,085,600-byte
`llama-server` binary was pinned at
`sha256:ae1bd49f869ff3397b2a5d757fcf010c6eaaf16c4e3071a15861312defcd4e40`.
This follows NVIDIA's pinned
[DGX Spark llama.cpp playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/1fb66f059ee427c5a3678b3117ef73aab042b458/nvidia/llama-cpp/README.md).

Both profiles offloaded all layers, enabled flash attention, exposed eight
parallel slots, used 8,192/512 batch/ubatch sizes and Q8_0 KV caches, disabled
reasoning, and allocated 262,144 total context tokens across eight 32,768-token
slots. The MTP profile changed only the architecture label and native
speculation flags: `draft-mtp`, maximum draft depth three, Q8_0 draft KV, and
backend sampling. Thus this is a matched runtime/artifact comparison, not a
cross-backend comparison with the earlier vLLM NVFP4 runs.

The four smoke/quick launches reached readiness and terminal completion:

| Run | Server startup | First-request TTFT | First-request E2E | Supported cases completed |
| --- | ---: | ---: | ---: | ---: |
| Baseline smoke | 4.050260 s | 0.291704 s | 1.053923 s | 3/3 |
| MTP3 smoke | 6.050201 s | 0.379223 s | 0.777117 s | 3/3 |
| Baseline quick | 4.048156 s | 0.283660 s | 0.961819 s | 7/7 |
| MTP3 quick | 4.047258 s | 0.383151 s | 0.906198 s | 7/7 |

The smoke chat request produced 32 tokens at 9.301705 aggregate tok/s without
MTP and 18.665528 tok/s with MTP; median E2E was 3.425610/1.699369 seconds.
Both JSON and tool-call fixtures passed for both profiles. Vision, embeddings,
and reranking were profile-declared unsupported and skipped, rather than
attempted failures. These are one-request protocol checks, not broad quality
evaluations.

The matched quick generation cases were independently recomputed from
`request_complete` and `case_complete` journal events:

| Case | Requests | Baseline aggregate | MTP3 aggregate | MTP3 change | Baseline / MTP3 median client decode | Baseline / MTP3 median TTFT | Baseline / MTP3 median E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D128 (C1) | 3 each | 10.292859 | 20.678449 | +100.901% | 10.542649 / 22.734785 | 0.368026 / 0.407833 s | 12.414334 / 5.993986 s |
| C2 | 4 each | 18.360362 | 31.074434 | +69.247% | 9.893563 / 18.592670 | 0.585085 / 0.639019 s | 6.952907 / 4.033852 s |
| C4 | 8 each | 31.575320 | 50.036640 | +58.468% | 8.856777 / 15.838235 | 0.968662 / 1.043930 s | 8.081689 / 5.095380 s |

Aggregate and decode columns are tokens/s. Every request reached its 64- or
128-token budget and passed validation. MTP roughly doubled single-stream
aggregate throughput and improved C2/C4 aggregate throughput, while median
client TTFT rose 7.8–10.8%. The lower median E2E reflects faster decoding after
that first token.

Prefill remained a client-TTFT approximation and did not improve with MTP:

| Target | Requests | Baseline / MTP3 total prompt tokens | Baseline / MTP3 median TTFT | Baseline / MTP3 approximate prompt tok/s | MTP3 change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 3 each | 969 / 975 | 0.634324 / 0.704704 s | 509.203702 / 461.186367 | -9.430% |
| 2,048 | 3 each | 6,348 / 6,354 | 3.050239 / 3.240916 s | 693.716212 / 653.518888 | -5.794% |
| 8,192 | 3 each | 24,780 / 24,783 | 11.195577 / 11.672199 s | 737.791349 / 707.750077 | -4.072% |

Each prefill request emitted only one measured token and had no content-quality
validator. The slightly different actual prompt totals come from the frozen
rendered requests, so the table uses observed token counts rather than target
labels in its calculations.

Both one-request 8K needle cases passed their validators. Baseline used 8,285
prompt tokens, with 11.356242-second TTFT, 12.638608-second E2E, and 1.106555
aggregate tok/s for 14 output tokens. MTP used 8,283 prompt tokens, with
11.796261-second TTFT, 12.350166-second E2E, and 1.132079 aggregate tok/s for
the same output-token count. MTP therefore had 3.875% slower needle TTFT but
2.282% faster E2E in this single trial.

Native Prometheus counters prove that speculation was active:

| MTP3 run | Drafts | Draft tokens | Accepted tokens | Recomputed acceptance | Accepted at positions 0 / 1 / 2 | Mean accepted length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Smoke | 26 | 78 | 61 | 78.205128% | 23 / 20 / 18 | 3.346154 |
| Quick | 507 | 1,479 | 906 | 61.257606% | 376 / 276 / 254 | 2.786982 |

The baseline snapshots contained zero drafts and accepted tokens. Acceptance
is `accepted_tokens / draft_tokens`; the reported mean accepted length also
recomputes as `(accepted_tokens + drafts) / drafts`. Each snapshot is a
cumulative counter over one server lifetime, including the first request,
warmups, and measured requests. It cannot defensibly assign acceptance to an
individual D128, concurrency, prefill, or needle case.

Telemetry contains 23/20 samples for baseline/MTP smoke and 173/135 for the
quick pair, with no missing GPU-utilization, power, clock, or temperature
fields. Minimum host-wide `MemAvailable` was 82.808/77.115 GiB for smoke and
75.386/68.036 GiB for quick; peak power was 54.63/61.62 W and 89.72/90.12 W,
and peak temperature was 61/62 °C and 84/85 °C. These lifecycle samples include
startup, first request, warmups, cases, and shutdown; host `MemAvailable` is
not model-resident memory, and averages across the unequal phase mixes would
be misleading. Each journal records `server_stopped` followed by terminal
`run_complete`, with no run error or measurement annotation. These were owned
native processes with `keep_server_requested=false`, not Docker containers.

Finally, the sample sizes are deliberately small: smoke is n=1, D128 and each
prefill are n=3, concurrency is two bursts, and the needle is n=1. No p95 or
statistical-significance claim follows. Client TTFT includes the loopback HTTP
and streaming path, and client decode is an estimate from one-token emission
chunks; the prefill rates are especially not engine-native prompt timings.

#### Matched Core Pair

The core baseline and MTP3 runs used the same pins and arguments above. Both
reached terminal `run_complete` with no run error or measurement annotation.
Each completed 14 cases, skipped three profile-unsupported cases, and skipped
one context-limited case. Their summaries are correctly `partial`, while
`run_completion_status` is `completed`, because D1024 failed its semantic
validation rather than because the server or harness failed.

The valid fixed-budget generation results were:

| Case | Requests | Baseline aggregate | MTP3 aggregate | MTP3 change | Baseline / MTP3 median client decode | Baseline / MTP3 median TTFT | Baseline / MTP3 median E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D256 | 5 each | 10.412715 | 22.076504 | +112.015% | 10.528125 / 22.815651 | 0.381371 / 0.412230 s | 24.591620 / 11.586655 s |
| C1 | 5 each | 10.423285 | 21.308735 | +104.434% | 10.555662 / 21.892791 | 0.384936 / 0.425577 s | 24.544665 / 12.079437 s |
| C2 | 10 each | 10.317788 | 21.478372 | +108.168% | 5.241760 / 11.239559 | 0.786013 / 0.883477 s | 49.474563 / 23.673731 s |
| C4 | 20 each | 34.168821 | 59.425994 | +73.919% | 8.856797 / 16.191336 | 1.177935 / 1.155266 s | 29.942310 / 16.827985 s |
| C8 | 40 each | 53.566635 | 87.224987 | +62.835% | 7.076732 / 12.459683 | 2.314002 / 2.471834 s | 38.336522 / 22.733354 s |

Aggregate and decode columns are tokens/s. Every listed request reached 256
tokens and passed validation. Relative to each profile's C1, aggregate C2/C4/C8
were 0.990x/3.278x/5.139x for baseline and 1.008x/2.789x/4.093x for MTP3.
The non-ideal C2 behavior is an observed scheduler result, not an assumption
that the two requests ran serially. MTP3 still improved aggregate throughput
by 62.8–108.2% across C1–C8, while client TTFT was slightly lower only at C4.

D1024 is intentionally absent from the valid-throughput table. All five
baseline requests ended with `finish_reason=stop` at 593–619 tokens (3,004
total), and all five MTP3 requests stopped at 593–598 tokens (2,976 total),
well before the 1,024-token budget. Every per-request validator therefore
reported `generation ended with 'stop'`. Median baseline/MTP3 TTFT was
0.374805/0.409301 seconds and median E2E was 56.640509/26.480144 seconds, but
the summary deliberately suppresses aggregate and median-decode throughput for
both cases. Publishing rates from unequal early-stop lengths would turn a
semantic validation miss into a misleading performance result. The preserved
timings diagnose the stop behavior; they are not valid D1024 benchmark rates.

The valid prefill cases again used client-observed TTFT:

| Target | Requests | Baseline / MTP3 total prompt tokens | Baseline / MTP3 median TTFT | Baseline / MTP3 approximate prompt tok/s | MTP3 change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 5 each | 975 / 985 | 0.493510 / 0.562725 s | 395.128747 / 350.082215 | -11.400% |
| 1,024 | 5 each | 5,470 / 5,460 | 1.729421 / 1.880356 s | 632.581475 / 580.741033 | -8.195% |
| 4,096 | 5 each | 20,825 / 20,830 | 5.757236 / 6.086410 s | 723.437475 / 684.475752 | -5.386% |
| 16,384 | 5 each | 82,265 / 82,265 | 22.798407 / 23.685721 s | 721.673225 / 694.637912 | -3.746% |

Each request emitted one measured token and intentionally had no quality
validator. MTP3 was 3.7–11.4% slower on this approximation. The 32K target was
not attempted: both journals estimate 32,909 required tokens against the
32,768-token per-slot/model cap and record `case_skipped_context_limit` before
any warmup or request. The native server's 262,144-token allocation is divided
across eight 32,768-token slots, so it does not make a 32,909-token single
request admissible.

Capability and quality validation remained intact:

| Case | Validation, baseline / MTP3 | Baseline / MTP3 aggregate tok/s | Baseline / MTP3 median TTFT | Baseline / MTP3 median E2E |
| --- | ---: | ---: | ---: | ---: |
| 16K needle | 3/3 / 3/3 | 0.547019 / 0.506827 | 23.033030 / 23.732657 s | 24.119258 / 24.190471 s |
| JSON | 5/5 / 5/5 | 8.498528 / 14.438256 | 0.394835 / 0.433607 s | 1.634355 / 0.956839 s |
| Tool call | 5/5 / 5/5 | 9.810166 / 22.814045 | 1.009909 / 0.649611 s | 3.648848 / 1.562551 s |
| Exact-answer quality | 4/4 / 4/4 | 5.871772 / 7.366043 | 0.423875 / 0.460643 s | 0.764811 / 0.617952 s |

The needle output counts differed slightly—40 baseline versus 37 MTP3 tokens—
so its aggregate-output comparison is less meaningful than the 3/3 validation
result. Exact-answer quality passed one item each for arithmetic, logic,
instruction following, and code reasoning. Vision, embeddings, and reranking
were declared unsupported and skipped. Per run, 97 of 122 measured requests
passed validation, five D1024 requests failed, and 20 one-token prefill requests
had no validator; there were no other validation failures.

The MTP3 core snapshot recorded 9,025 drafts, 26,926 draft tokens, and 17,420
accepted tokens: `17,420 / 26,926 = 64.695833%`. Accepted-position counters
were 7,503/5,181/4,736 and mean accepted length was 2.930194, which also
recomputes as `(17,420 + 9,025) / 9,025`. The baseline counters were all zero.
As with quick, this is one cumulative snapshot over the first request,
warmups, and all measured cases; it proves MTP activity but cannot allocate
acceptance to an individual workload.

Process readiness took 4.048050 seconds for baseline and 6.053039 seconds for
MTP3. Their first-request TTFT/E2E was 0.286274/0.955763 seconds and
0.380616/0.917606 seconds respectively. For these runs, the
`server_startup` telemetry phase began before exact SHA-256 validation of the
17,923,394,624-byte artifact, so its 11/13 samples include validation time.
The journal's process `startup_s` clock begins only after artifact validation
and therefore excludes it; subtracting these coarse sampled spans is not an
exact checksum timer.

| Run | Telemetry samples | Minimum host `MemAvailable` | Average GPU utilization | Average power | Peak power | Peak temperature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline core | 1,624 | 74.078 GiB | 93.387% | 61.388 W | 91.58 W | 87 °C |
| MTP3 core | 959 | 66.675 GiB | 90.160% | 69.338 W | 91.05 W | 87 °C |

All telemetry records contain GPU-utilization, power, clock, and temperature;
peak SM clock was 2,515 MHz in both runs. The sample totals and whole-run
averages cover different-duration startup, warmup, measured-case, and shutdown
mixtures. Host `MemAvailable` is not model-resident memory, so neither the
7.403 GiB minimum-memory difference nor the averages isolate MTP overhead.

These runs predate the later CORS hardening, although both servers were
loopback-bound, offline, and UI-disabled. Server-log task accounting rules out
unexplained generation traffic: each log contains exactly 137 unique parent
task launches, 137 prompt-evaluation records, 137 total-time records, and 137
releases. That equals one fixed first request, 14 planned warmups, and 122
measured journal requests. The warmups are four decode, four prefill, four
single-request concurrency primes, one JSON, and one tool request. There are no
extra generation tasks in either log. This does not enumerate harmless
readiness or metrics HTTP probes; it specifically audits requests that entered
a generation slot. Both journals then record `server_stopped` followed by
terminal `run_complete` with `keep_server_requested=false`.

#### Post-Hardening GGUF Quantization and Vision Sweep

Nine later terminal runs exercised the hardened native path and expanded the
quantization and vision coverage. All retained the same Unsloth revision and
`llama.cpp` b10453 runtime/binary digest documented above. Exact model artifacts
were:

- Q4_K_XL: 17,923,394,624 bytes at
  `sha256:bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`;
- Q8_0: 29,047,086,048 bytes at
  `sha256:a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348`;
- UD-IQ2_XXS: 9,010,048,064 bytes at
  `sha256:8d1b37297d6cf98303cd396896f35e01089ddcc904053a9c6997f7a1c35b8524`;
- vision `mmproj-F16.gguf`: 927,607,488 bytes at
  `sha256:cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e`.

Q8 is 62.062% larger than Q4; IQ2 is 49.730% smaller than Q4, while Q8 is
3.224x the IQ2 file size. Every run records an `artifact_validation_complete`
event with the expected model, runtime, and, for vision, projector SHA. Unlike
the preceding core pair, checksum work now has a distinct
`artifact_validation` telemetry phase and exact elapsed field rather than
being grouped under `server_startup`. Process `startup_s` remains separately
timed after validation.

| Run | Artifact validation | Process startup | Minimum host `MemAvailable` | Terminal status |
| --- | ---: | ---: | ---: | --- |
| Q4 hardened smoke | 8.507540 s | 4.048850 s | 83.119 GiB | Complete |
| Q4 MTP3 hardened smoke | 8.487868 s | 6.050979 s | 77.392 GiB | Complete |
| Q4 vision | 8.767855 s | 4.047399 s | 81.635 GiB | Complete |
| Q8 smoke | 13.555300 s | 30.088293 s | 73.133 GiB | Complete |
| Q8 quick | 13.534328 s | 12.059043 s | 64.125 GiB | Complete |
| Q8 core | 13.557539 s | 10.061691 s | 64.241 GiB | Partial/completed |
| IQ2 smoke | 4.349877 s | 4.046590 s | 91.983 GiB | Partial/completed |
| IQ2 quick | 4.246855 s | 4.048392 s | 83.537 GiB | Complete |
| IQ2 core | 4.359427 s | 4.045306 s | 83.237 GiB | Partial/completed |

The stable checksum durations scale with artifact size. Q8 process startup did
not: its 30.088-second smoke launch fell to 12.059/10.062 seconds in quick/core,
so the cold smoke value is not a fixed model-load constant. `MemAvailable` is
host-wide and includes different run phases; it is useful as a pressure bound,
not an allocation measurement.

All nine provenance records now freeze `--cors-origins localhost` and
`--no-cors-credentials` in addition to loopback binding, offline mode, and a
disabled UI. This is the relevant distinction from the preceding pair, whose
logs were audited for unexpected generation but whose provenance predates the
CORS flags. The hardened logs contain 4/4/13/4/32/137/4/32 generation-slot
tasks for Q4 smoke/MTP/vision, Q8 smoke/quick/core, and IQ2 smoke/quick,
followed by 137 for IQ2 core—367 total—and exactly the same number of releases.
Those counts equal the fixed first requests, planned warmups, and 315 measured
journal requests, with no unexplained generation task. Every run then records
`server_stopped` and terminal `run_complete`.

The hardened smoke comparison was:

| Profile | Chat aggregate tok/s | Chat median TTFT | Chat median E2E | JSON | Tools |
| --- | ---: | ---: | ---: | --- | --- |
| Q4 | 9.709843 | 0.377129 s | 3.281090 s | Pass | Pass |
| Q4 MTP3 | 17.694089 | 0.480172 s | 1.794456 s | Pass | Pass |
| Q8 | 6.793473 | 0.448869 s | 4.695755 s | Pass | Pass |
| IQ2 | 16.453750 | 0.319319 s | 1.930040 s | **Fail** | Pass |

Q4 MTP3 again showed real speculation: 61 of 78 draft tokens were accepted
(`78.205128%`), with position counts 23/20/18 and mean accepted length
3.346154. The new proposal-depth check also passed: every draft proposed the
configured three tokens and the deepest accepted position was two, proving all
three draft positions were exercised. Baseline/Q8/IQ2 correctly reported that
speculation was not requested. IQ2's JSON request stopped normally but was not
valid JSON, so the smoke result is `partial`; chat and the tool-call protocol
still passed. This is concrete format-quality loss on one fixture, not a claim
about general IQ2 accuracy.

The Q4 vision profile added the exact F16 projector above and passed all nine
solid-red image validations:

| Input size | Requests | Total prompt tokens | Aggregate tok/s | Median TTFT | Median E2E | Validation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 px | 3 | 114 | 7.498338 | 0.155521 s | 0.251288 s | 3/3 |
| 512 px | 3 | 855 | 7.454773 | 0.157234 s | 0.253874 s | 3/3 |
| 1,024 px | 3 | 3,159 | 7.105849 | 0.171671 s | 0.268477 s | 3/3 |

Each response used two output tokens. The 9/9 result validates the projector,
image transport, and dominant-red fixture across three sizes; it is not a
general visual-reasoning or OCR score.

For the matched quick suite, the earlier non-MTP Q4 run is retained as the
middle-quantization anchor. Q8 and IQ2 used the hardened path, so small
run-order effects remain possible, but the same model family, runtime, suite,
and serving geometry make the direction clear:

| Case | Q4 aggregate | Q8 aggregate | Q8 vs Q4 | IQ2 aggregate | IQ2 vs Q4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D128 | 10.292859 | 7.070759 | -31.304% | 18.436103 | +79.115% |
| C2 | 18.360362 | 12.536213 | -31.721% | 30.168622 | +64.314% |
| C4 | 31.575320 | 21.396566 | -32.236% | 46.656897 | +47.764% |

All 15 measured decode/concurrency requests per quantization reached their
token budgets and passed. Client-TTFT prefill showed a smaller spread:

| Target | Q4 approximate prompt tok/s | Q8 | Q8 vs Q4 | IQ2 | IQ2 vs Q4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 509.203702 | 438.058854 | -13.972% | 519.053661 | +1.934% |
| 2,048 | 693.716212 | 641.289752 | -7.557% | 664.276387 | -4.244% |
| 8,192 | 737.791349 | 687.545678 | -6.810% | 695.980693 | -5.667% |

All three one-request 8K needles passed. Q4/Q8/IQ2 median TTFT was
11.356242/12.197608/12.080339 seconds and median E2E was
12.638608/13.876281/12.641243 seconds. Output counts differed at 14/13/11, so
their aggregate output rates are not a clean quality or decode comparison.
The quick suite has no JSON or exact-answer case: IQ2's complete quick result
does not negate its smoke JSON failure.

Q8 core reproduced the same semantic shape as Q4 core. Valid fixed-budget
generation was about one-third slower:

| Case | Q4 aggregate | Q8 aggregate | Q8 change | Q8 median client decode | Q8 median TTFT | Q8 median E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D256 | 10.412715 | 6.980862 | -32.958% | 7.102248 | 0.463862 s | 36.366691 s |
| C1 | 10.423285 | 7.072987 | -32.142% | 7.133668 | 0.493668 s | 36.197773 s |
| C2 | 10.317788 | 7.017274 | -31.989% | 3.547717 | 0.972074 s | 72.870853 s |
| C4 | 34.168821 | 22.521189 | -34.088% | 5.782403 | 1.306641 s | 45.405909 s |
| C8 | 53.566635 | 35.174646 | -34.335% | 4.569833 | 2.342640 s | 58.378101 s |

All 80 requests in that table reached 256 tokens and passed. Q8 core prefill
was also lower than Q4:

| Target | Q4 approximate prompt tok/s | Q8 | Q8 change | Q8 median TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 395.128747 | 323.793096 | -18.054% | 0.602236 s |
| 1,024 | 632.581475 | 566.140554 | -10.503% | 1.930616 s |
| 4,096 | 723.437475 | 672.276911 | -7.072% | 6.196851 s |
| 16,384 | 721.673225 | 677.086270 | -6.178% | 24.302664 s |

The Q8 D1024 case is correctly unranked: all five requests stopped at 592–654
tokens (3,066 total), so all five semantic validations failed and aggregate
and median-decode rates are suppressed. Median TTFT/E2E was
0.476457/86.379863 seconds.
The 32K case was again skipped at an estimated 32,909 tokens against the
32,768 per-slot cap. Q8 nevertheless passed the 16K needle 3/3, JSON 5/5,
tools 5/5, and all four exact-answer categories. Thus Q8 supplied no measured
quality advantage over Q4 on this deliberately tiny screen while using a
larger artifact and running slower; that is not evidence of general quality
equivalence. Q8 core is `partial/completed` only because of D1024 and the
declared context/capability skips.

IQ2 core completed the same frozen suite and preserved its speed advantage on
valid fixed-budget generation:

| Case | Requests | Aggregate tok/s | Median client decode | Median TTFT | Median E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| D256 | 5 | 18.873221 | 19.272907 | 0.323548 s | 13.551251 s |
| C1 | 5 | 18.716919 | 19.064998 | 0.325566 s | 13.709214 s |
| C2 | 10 | 18.328012 | 9.394433 | 0.688723 s | 27.840447 s |
| C4 | 20 | 53.218560 | 14.088656 | 1.137205 s | 19.257254 s |
| C8 | 40 | 74.250681 | 10.031032 | 2.188310 s | 27.561176 s |

All 80 requests reached 256 tokens and passed validation. Relative to the Q4
core anchor, IQ2 aggregate throughput was 81.252%/79.568%/77.635%/55.752%/
38.614% higher for D256/C1/C2/C4/C8. Its client-TTFT prefill estimates for
128/1,024/4,096/16,384 targets were 425.819236/602.205475/675.688221/
678.498276 prompt tok/s, from median TTFTs of 0.460289/1.813335/6.165566/
24.252088 seconds. These are performance results for this quantized artifact,
not evidence that IQ2 preserved general model quality.

The IQ2 D1024 case is also correctly suppressed: all five requests stopped
semantically early at 292–597 tokens (2,370 total), so validation was 0/5 and
both aggregate and median-decode rates are null. The retained median TTFT/E2E
was 0.321040/31.301940 seconds. IQ2 also repeated its smoke JSON failure in all
five core repetitions: each normal-stop response was the same 25-token
Markdown-fenced object rather than valid top-level JSON, for 0/5 validation.
That reproducible format failure is distinct from pipeline failure.

The remaining semantic checks passed: the 16K needle was 3/3 with 49,432
total prompt tokens, 0.490894 aggregate tok/s, and 24.275516/24.916863-second
median TTFT/E2E; tools were 5/5 at 17.044216 aggregate tok/s; and arithmetic,
logic, instruction-following, and code exact answers were 4/4 at 7.603424
aggregate tok/s. The 32K request was skipped before inference at an estimated
32,909 tokens against the 32,768 cap, while vision, embeddings, and reranking
were profile-declared unsupported. The summary is therefore
`partial/completed`: 92 measured validations passed, ten failed (five D1024
and five JSON), and 20 one-token prefill requests had no quality validator.

IQ2 core validated its 9,010,048,064-byte artifact in 4.359427 seconds and
reached process readiness in 4.045306 seconds. First-request TTFT/E2E was
0.273284/0.646547 seconds; the phase was shorter than the telemetry sampler
interval and consequently has no sample. The full telemetry journal has 1,033
samples, minimum host `MemAvailable` 83.237160 GiB, average GPU utilization
91.866409%, average power 73.740242 W, peak power 91.66 W, peak temperature
87 °C, and peak SM clock 2,515 MHz. Every sample contains all four GPU
scalars. Its server log has exactly 137 parent-task launches, prompt-timing
records, total-time records, and releases: one fixed prime, 14 warmups, and
122 measured requests, with no unexplained generation task. The owned process
then stopped before terminal `run_complete`.

The nine telemetry files contain 3,864 samples with no missing GPU
utilization, power, clock, or temperature scalar. Per-run counts are
23/21/21 for Q4 smoke/MTP/vision, 57/232/2,325 for Q8 smoke/quick/core, and
16/136/1,033 for IQ2 smoke/quick/core. Peak power across this sweep was
93.35 W and peak temperature was 87 °C, both in terminal runs; phase mixes
and durations differ, so whole-run average power is not a quantization-
efficiency result.

#### Matched MTP Draft-Depth Sweep

The frozen matrix at `results/matrices/20260816T212553Z-llamacpp-mtp-depth`
completed all six profiles, bracketed by independent non-MTP controls before
and after it. Every profile retained the Q4 artifact, runtime digest, serving
geometry, Q8_0 KV caches, offline loopback binding, and hardened CORS settings
documented above; only the requested MTP maximum draft length changed. Each
matrix/control run used two warmups and five measured D256 requests.

The opening control produced 10.379663 aggregate tok/s, 10.483883 median
client-decode tok/s, and 0.383704/24.713894-second median TTFT/E2E. The closing
control produced 10.414178, 10.576687, and 0.386176/24.495805 seconds. Closing
aggregate throughput was 0.332526% higher—**0.333% control drift** when rounded—
and all ten measured requests reached 256 tokens and passed. This is a useful
stability bound for the short sweep, not a confidence interval. Both controls
correctly reported speculation unrequested and zero drafts, draft tokens, and
accepted tokens.

The six profiles executed in the frozen order shown here:

| Order | Profile (configured maximum) | Aggregate tok/s | Median client decode | Median TTFT | Median E2E | Validation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | MTP3 | 21.493041 | 22.223815 | 0.419432 s | 11.893943 s | 5/5 |
| 2 | MTP6 | 22.257576 | 23.434015 | 0.416165 s | 11.297618 s | 5/5 |
| 3 | MTP1 | 16.711829 | 17.184165 | 0.409952 s | 15.253386 s | 5/5 |
| 4 | MTP5 | **23.701124** | 25.027339 | 0.403681 s | 10.590513 s | 5/5 |
| 5 | MTP2 | 19.834056 | 20.516821 | 0.415847 s | 12.837050 s | 5/5 |
| 6 | MTP4 | 22.949409 | 23.985603 | 0.413944 s | 11.043727 s | 5/5 |

Every row is terminal `complete`, measurement-valid, and totals 1,280 output
tokens; observed prompt totals were 395–405 tokens. Ranked by this preliminary
five-repetition aggregate, the order was MTP5, MTP4, MTP6, MTP3, MTP2, MTP1.
In particular, MTP5 measured **23.701** tok/s versus **22.949** for MTP4, a
3.275534% lead before confirmation.

Native cumulative counters for those same server lifetimes were:

| Max | Drafts | Draft tokens | Accepted | Acceptance | Mean accepted length | Average proposed/draft | Accepted positions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 955 | 955 | 833 | 87.225131% | 1.872251 | 1.000000 | 833 |
| 2 | 728 | 1,451 | 1,062 | 73.190903% | 2.458791 | 1.993132 | 597 / 465 |
| 3 | 614 | 1,833 | 1,172 | 63.938898% | 2.908795 | 2.985342 | 481 / 363 / 328 |
| 4 | 533 | 2,110 | 1,256 | 59.526066% | 3.356473 | 3.958724 | 369 / 306 / 302 / 279 |
| 5 | 487 | 2,409 | 1,299 | 53.922790% | 3.667351 | 4.946612 | 319 / 302 / 295 / 273 / 110 |
| 6 | 487 | 2,888 | 1,301 | 45.048476% | 3.671458 | 5.930185 | 311 / 297 / 295 / 267 / 103 / 28 |

Here “MTP5” or depth five means the **configured maximum draft length**, not
that every proposal or acceptance had length five. The average-proposal column
reports the observed value, and every proposal-depth check reached and accepted
the configured deepest position. Acceptance is accepted/draft tokens; mean
accepted length recomputes as `(accepted + drafts) / drafts`. All counters are
cumulative across the fixed prime request, both warmups, and all five measured
requests. They prove activity and depth coverage, but are not measured-only or
per-request acceptance rates.

The leading two depths then ran the 20-repetition confirmation suite:

| Profile | Aggregate tok/s | Median client decode | Median / p95 TTFT | Median / p95 E2E | Drafts | Accepted / draft tokens | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP5 | **23.799848** | 25.552286 | 0.408432 / 0.430412 s | 10.383478 / 11.819780 s | 1,517 | 4,087 / 7,483 | 54.617132% |
| MTP4 | 23.269441 | 24.274008 | 0.416272 / 0.423137 s | 10.919224 / 11.668320 s | 1,658 | 3,956 / 6,549 | 60.406169% |

Both runs generated 5,120 measured tokens and passed all 20 validations. MTP5
finished in 215.127420 seconds versus 220.031069 for MTP4, so its confirmed
aggregate lead is **2.279416%** (23.799848 versus 23.269441 tok/s). MTP5's mean
accepted length and average proposal were 3.694133/4.932762, with accepted
position counts 995/949/924/857/362. MTP4 recorded 3.386007/3.949940 and
1,158/976/947/875. These counters again include one prime and two warmups in
addition to the 20 measured requests. The matched confirmation supports MTP5
over MTP4 for this D256 setup; it does not establish a universal optimal depth.

Lifecycle and telemetry accounting for the controls, matrix, and confirmations
was:

| Run | Artifact validation | Process startup | First TTFT / E2E | Samples | Minimum `MemAvailable` | Average GPU / power | Peak W / °C | Parent tasks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control open | 8.598664 s | 4.046931 s | 0.281639 / 0.948173 s | 178 | 82.754776 GiB | 88.477528% / 47.504888 W | 52.62 / 79 | 8 |
| MTP3 | 8.561316 s | 4.040490 s | 0.382662 / 0.907068 s | 94 | 76.879147 GiB | 81.734043% / 58.526702 W | 69.18 / 83 | 8 |
| MTP6 | 8.572177 s | 6.043045 s | 0.376114 / 0.846332 s | 93 | 72.714363 GiB | 79.225806% / 65.839140 W | 80.91 / 86 | 8 |
| MTP1 | 8.563896 s | 4.042409 s | 0.379610 / 0.935669 s | 116 | 78.664268 GiB | 83.827586% / 49.243448 W | 55.74 / 79 | 8 |
| MTP5 | 8.387622 s | 6.046941 s | 0.379686 / 0.832771 s | 89 | 73.914001 GiB | 77.651685% / 62.429438 W | 76.85 / 86 | 8 |
| MTP2 | 8.391386 s | 6.046773 s | 0.386029 / 0.851623 s | 102 | 77.440441 GiB | 81.911765% / 52.932549 W | 62.63 / 80 | 8 |
| MTP4 | 8.590874 s | 6.052863 s | 0.388008 / 0.957827 s | 90 | 75.088749 GiB | 79.433333% / 59.329778 W | 72.00 / 83 | 8 |
| Control close | 8.576899 s | 4.049254 s | 0.288606 / 0.956289 s | 178 | 82.103329 GiB | 88.370787% / 46.439326 W | 50.76 / 75 | 8 |
| MTP5 confirm | 8.565582 s | 6.047696 s | 0.379452 / 0.973897 s | 243 | 69.703293 GiB | 87.613169% / 69.622634 W | 78.08 / 86 | 23 |
| MTP4 confirm | 8.378307 s | 6.057060 s | 0.375849 / 0.938820 s | 247 | 70.159081 GiB | 87.724696% / 66.972348 W | 73.82 / 86 | 23 |

All ten artifact events matched the frozen Q4 and runtime SHA-256 digests; each
artifact-validation phase contains eight telemetry samples and is separate
from process `startup_s`. The 1,430 telemetry samples have no missing GPU,
power, clock, or temperature scalar. Host `MemAvailable` and whole-run averages
mix validation, startup, warmup, measured, and shutdown phases, so they do not
isolate model memory or per-token efficiency.

Each control/matrix server log has exactly eight parent launches, prompt-time
records, total-time records, and releases (`1 + 2 + 5`); each confirmation has
23 of each (`1 + 2 + 20`). Across the ten runs, all 110 generation tasks are
therefore explained. Every journal records an owned-server stop followed by
terminal `run_complete`, and the matrix ledger reports all six children
`complete`.

## Validity and Evidence Ledger

Four interferences are preserved in `measurement_annotations`: Qwen3.8 NVFP4
MTP3 startup only; initial Qwen Coder startup plus one decode repetition; the
new Qwen3 8B NVFP4 decode case; and RLM Qwen3 8B repair startup only. The
affected numbers are excluded above, while non-overlapping cases remain usable.
Serialized Ollama queues, client-TTFT prefill approximations, and bundled SSE
are labeled wherever they affect interpretation.

Primary evidence paths are:

- `results/matrices/20260816T021055Z-quick`
- `results/20260816T035020Z-qwen3-vl-embedding-2b-bf16-multimodal-embeddings-0fe7a49c`
- `results/20260816T035133Z-qwen3-vl-reranker-2b-bf16-multimodal-rerank-9bc3f63e`
- `results/20260816T040209Z-qwen3-8b-nvfp4-quick-d12b0ad8`
- `results/20260816T041014Z-qwen3-8b-fp8-quick-eda9e3c2`
- `results/20260816T042133Z-gpt-oss-20b-mxfp4-reasoning-quick-ce877963`
- `results/20260816T042615Z-nemotron-labs-diffusion-14b-transformers-direct-diffusion-direct-e062a2de`
- `results/20260816T043515Z-all-minilm-l6-v2-bf16-embeddings-f4e55e1a`
- `results/20260816T043542Z-ollama-nomic-embed-text-f16-embeddings-c4ceca04`
- `results/20260816T043554Z-qwen3-coder-30b-a3b-bf16-ar-control-quick-2193523e`
- `results/20260816T044521Z-qwen3-coder-30b-a3b-bf16-dflash15-quick-a8dfd578`
- `results/20260816T045458Z-qwen38-27b-bf16-throughput-quick-bb4bb479`
- `results/20260816T050847Z-qwen38-27b-nvfp4-throughput-quick-b8d27b4f`
- `results/20260816T051818Z-gpt-oss-120b-mxfp4-reasoning-quick-e40d9ffa`
- `results/20260816T052746Z-rlm-qwen3-8b-bf16-reasoning-quick-c7491fa4`
- `results/20260816T053203Z-ollama-nemotron-cascade-2-q4-k-m-ollama-prefill-repair-1b66f9b2`
- `results/20260816T053245Z-ollama-qwen3-30b-a3b-q4-k-m-ollama-prefill-repair-c484aa21`
- `results/20260816T053308Z-ollama-glm-4.7-flash-q4-k-m-ollama-prefill-repair-81fd6d05`
- `results/20260816T053338Z-ollama-qwen3.5-35b-a3b-q4-k-m-ollama-prefill-repair-7757ad19`
- `results/20260816T053416Z-ollama-gemma3-12b-q4-k-m-ollama-prefill-repair-241a44ae`
- `results/20260816T053448Z-ollama-nemotron-3-super-q4-k-m-ollama-prefill-repair-e005e606`
- `results/20260816T053610Z-ollama-gemma4-31b-q4-k-m-ollama-prefill-repair-d92d9223`
- `results/20260816T053732Z-ollama-llama3.3-70b-q4-k-m-ollama-prefill-repair-2cab8179`
- `results/20260816T054034Z-ollama-mistral-medium-3.5-128b-q4-k-m-ollama-prefill-repair-5e2876a4`
- `results/20260816T054610Z-ollama-deepseek-ocr-f16-ocr-89ee5553`
- `results/20260816T055102Z-ollama-deepseek-ocr-f16-ocr-89ee5553`
- `results/20260816T055210Z-ollama-deepseek-ocr-f16-ocr-89ee5553`
- `results/matrices/20260816T055256Z-vision`
- `results/20260816T055710Z-gemma-3-4b-it-bf16-vision-0017fbd0`
- `results/20260816T060004Z-gemma-3-4b-it-bf16-capabilities-53d5590d`
- `/tmp/sparktts-bench.OuHJwV/metrics.jsonl`
- `/tmp/sparktts-bench.mHsEOu/metrics.jsonl`
- `/tmp/sparktts-bench.8o2Bne/metrics.jsonl`
- `/tmp/sparktts-bench.u6H1uC/metrics.jsonl`
- `/tmp/whisper-bench.pKabZq/metrics.jsonl`
- `/tmp/smollm2-unsloth.Va7qUG/console.log`
- `/tmp/smollm2-unsloth.GjjBcx/metrics.json`
- `/tmp/smollm2-unsloth.GjjBcx/SHA256SUMS`
- `/tmp/smollm2-unsloth.xYctAZ/metrics.json`
- `/tmp/smollm2-unsloth.xYctAZ/SHA256SUMS`
- `results/20260816T062133Z-qwen36-35b-a3b-nvfp4-mtp3-core-795163dc`
- `results/20260816T062941Z-qwen36-35b-a3b-nvfp4-mtp3-vision-01724f08`
- `results/20260816T063208Z-qwen38-27b-bf16-throughput-core-138da570`
- `results/20260816T073258Z-qwen38-27b-nvfp4-throughput-core-649479d8`
- `results/20260816T080900Z-qwen38-27b-nvfp4-mtp3-throughput-core-3ac7df1a`
- `results/20260816T083223Z-gpt-oss-20b-mxfp4-reasoning-core-7382f497`
- `results/20260816T084122Z-gpt-oss-120b-mxfp4-reasoning-core-c6157712`
- `results/20260816T085950Z-qwen3-8b-nvfp4-core-bddaa03b`
- `results/20260816T090922Z-qwen3-8b-fp8-core-0a71f1b4`
- `results/20260816T092132Z-rlm-qwen3-8b-bf16-reasoning-core-2be66127`
- `results/20260816T094442Z-qwen3-coder-30b-a3b-bf16-ar-control-core-ccc29e71`
- `results/20260816T100225Z-qwen3-coder-30b-a3b-bf16-dflash15-core-fdc2fd73`
- `results/20260816T101928Z-ollama-qwen3-30b-a3b-q4-k-m-core-958161dd`
- `results/20260816T102945Z-ollama-nemotron-cascade-2-q4-k-m-core-b9bd5241`
- `results/20260816T103720Z-ollama-qwen3.5-35b-a3b-q4-k-m-core-a45b2e2a`
- `results/20260816T104925Z-ollama-nemotron-3-super-q4-k-m-core-88a64c71`
- `results/20260816T111355Z-ollama-gemma4-31b-q4-k-m-core-f540100e`
- `results/20260816T120827Z-ollama-llama3.3-70b-q4-k-m-core-2003e97f`
- `results/20260816T134826Z-ollama-glm-4.7-flash-q4-k-m-capabilities-d4cd1195`
- `results/20260816T134842Z-ollama-glm-4.7-flash-q4-k-m-chat-quality-3e480903`
- `results/20260816T134855Z-ollama-mistral-medium-3.5-128b-q4-k-m-capabilities-f780c99e`
- `results/20260816T135021Z-ollama-mistral-medium-3.5-128b-q4-k-m-chat-quality-9f5c19fb`
- `results/20260816T135109Z-ollama-gemma3-12b-q4-k-m-capabilities-0924c860`
- `results/20260816T135124Z-ollama-gemma3-12b-q4-k-m-chat-quality-eddc7f3d`
- `results/20260816T135140Z-gemma-3-4b-it-bf16-chat-quality-73920897`
- `results/20260816T135413Z-ollama-glm-4.7-flash-q4-k-m-core-13071433`
- `results/20260816T140431Z-gemma-3-4b-it-bf16-core-86c5102d`
- `results/20260816T141751Z-ollama-gemma3-12b-q4-k-m-core-d3a9c272`
- `results/20260816T143735Z-ollama-mistral-medium-3.5-128b-q4-k-m-core-68dcaf64`
- `results/20260816T162204Z-phi-4-multimodal-instruct-nvfp4-smoke-4b67f818`
- `results/20260816T163510Z-phi-4-multimodal-instruct-nvfp4-smoke-faa306b5`
- `results/20260816T163733Z-phi-4-multimodal-instruct-nvfp4-smoke-9adcae20`
- `results/20260816T164642Z-phi-4-multimodal-instruct-fp8-smoke-7d4c8353`
- `results/20260816T164847Z-phi-4-multimodal-instruct-fp8-vision-eb08efc1`
- `results/20260816T165046Z-phi-4-multimodal-instruct-fp8-quick-65e98f95`
- `results/20260816T165332Z-phi-4-multimodal-instruct-fp8-reasoning-core-7a48a87e`
- `results/20260816T171328Z-phi-4-reasoning-plus-fp8-smoke-89328647`
- `results/20260816T171724Z-phi-4-reasoning-plus-fp8-reasoning-core-c2ff5771`
- `results/20260816T174341Z-phi-4-multimodal-instruct-fp8-audio-audio-asr-56e2f4bb`
- `results/20260816T180652Z-phi-4-multimodal-instruct-fp8-trtllm-audio-audio-asr-44167e8d`
- `results/20260816T180805Z-phi-4-multimodal-instruct-fp8-trtllm-audio-audio-asr-778d04b9`
- `results/20260816T181011Z-phi-4-multimodal-instruct-fp8-trtllm-audio-audio-asr-1d006c80`
- `results/20260816T181236Z-phi-4-multimodal-instruct-fp8-trtllm-audio-audio-asr-7c232733`
- `results/20260816T181654Z-phi-4-multimodal-instruct-fp8-trtllm-audio-audio-asr-c1f49ed1`
- `results/20260816T181931Z-phi-4-multimodal-instruct-fp8-trtllm-audio-audio-asr-fe899e08`
- `results/20260816T182551Z-phi-4-multimodal-instruct-fp8-trtllm-audio-audio-asr-09e4e739`
- `results/20260816T185014Z-qwen38-27b-ud-q4-k-xl-llamacpp-smoke-806b0a41`
- `results/20260816T185053Z-qwen38-27b-ud-q4-k-xl-llamacpp-mtp3-smoke-74fd0198`
- `results/20260816T185130Z-qwen38-27b-ud-q4-k-xl-llamacpp-quick-e4bfaebd`
- `results/20260816T185444Z-qwen38-27b-ud-q4-k-xl-llamacpp-mtp3-quick-bf786900`
- `results/20260816T185730Z-qwen38-27b-ud-q4-k-xl-llamacpp-core-6c40ac8d`
- `results/20260816T192615Z-qwen38-27b-ud-q4-k-xl-llamacpp-mtp3-core-d2eedbb5`
- `results/20260816T195615Z-qwen38-27b-ud-q4-k-xl-llamacpp-smoke-4fe645a9`
- `results/20260816T201207Z-qwen38-27b-ud-q4-k-xl-llamacpp-mtp3-smoke-e7688286`
- `results/20260816T201245Z-qwen38-27b-ud-q4-k-xl-llamacpp-vision-vision-8ee8caa8`
- `results/20260816T201352Z-qwen38-27b-q8-0-llamacpp-smoke-ce2a61fe`
- `results/20260816T201534Z-qwen38-27b-q8-0-llamacpp-quick-384958f4`
- `results/20260816T202020Z-qwen38-27b-q8-0-llamacpp-core-96642439`
- `results/20260816T210122Z-qwen38-27b-ud-iq2-xxs-llamacpp-smoke-dc5303be`
- `results/20260816T210149Z-qwen38-27b-ud-iq2-xxs-llamacpp-quick-f50b7851`
- `results/20260816T210420Z-qwen38-27b-ud-iq2-xxs-llamacpp-core-a7798fa4`
- `results/20260816T212232Z-qwen38-27b-ud-q4-k-xl-llamacpp-llamacpp-mtp-depth-d48bca41`
- `results/matrices/20260816T212553Z-llamacpp-mtp-depth`
- `results/20260816T213709Z-qwen38-27b-ud-q4-k-xl-llamacpp-llamacpp-mtp-depth-d48bca41`
- `results/20260816T214033Z-qwen38-27b-ud-q4-k-xl-llamacpp-mtp5-llamacpp-mtp-depth-confirm-a24560cf`
- `results/20260816T214513Z-qwen38-27b-ud-q4-k-xl-llamacpp-mtp4-llamacpp-mtp-depth-confirm-383e6711`

A filesystem-only recomputation from `events.jsonl` matched all 80 available
matrix aggregate-rate fields, all 92 aggregate-rate fields across the 20
post-matrix runs with completed request cases, and both diffusion aggregate
rates exactly (maximum absolute delta `0`). The six matched BF16/NVFP4
throughput prefill medians, three GPT-OSS 120B repaired prefill medians, three
RLM repaired prefill medians, 27 Ollama server-native prefill medians, and nine
embedding/rerank medians independently recomputed from request events also
matched their summaries exactly. The first OCR adapter attempt had no completed
case or aggregate field; the other two OCR aggregate calculations matched
exactly, and their journals confirm three identical empty or incorrect outputs.
The Ollama vision journals independently confirm all 45 request validations
and all 15 summary median TTFT/E2E pairs exactly. The vLLM Gemma3 follow-up adds
nine passed validations and three more independently matched median pairs
(maximum absolute delta `0`). The two Gemma3 capability aggregate-output rates
also recomputed exactly from the journal, and both validations passed; the
third case is explicitly unsupported. All eight Whisper records parsed into
the expected four trial pairs; RTF and reciprocal real-time arithmetic matched,
and CER was independently recomputed from the normalized 55-character reference
with no differences. The three failed Spark-TTS artifacts each contain a load
but no trial; the successful artifact contains one load and three trials. All
three RTF/real-time calculations and independently hashed WAV outputs matched
their records exactly. Final SmolLM2 inference and both training-rate formulas
also recomputed exactly; 16 finite losses and all ten listed output hashes were
verified, including eight readable adapter/tokenizer files. The 29 later runs
add 272 summarized cases and 2,398 request-complete events;
1,978 recomputed token counts, medians, elapsed/rate fields, and aggregate rates
matched their summaries exactly (maximum absolute delta `0`).

The seven Phi journals add 24 completed cases, three failed warmup cases, and
140 measured request completions. All 23 numeric aggregate-output rates were
recomputed as completion tokens divided by case wall time, and all 58 numeric
summary TTFT/E2E/decode medians were independently sorted from request events; every
value matched its summary exactly (maximum absolute delta `0`). The journals
also contain eight request-level validation failures: one NVFP4 vision, one
quick needle, one truncated D1024 decode, three core needles, and two exact-
answer items.

The two Phi-4 Reasoning Plus journals add 13 completed cases and 113 measured
request completions, with no failed cases. All 13 aggregate-output rates and
all 37 numeric TTFT/E2E/decode/prefill medians independently recomputed from
request events matched their summaries exactly (maximum absolute delta `0`).
Seven requests failed validation: three needles and four exact-answer items.
The final TRT-LLM audio journal adds three measured request completions and one
terminal partial case. Recomputing from those events gives 53 tokens per
request, median latency/output rate of 1.750606 seconds/30.275 tok/s, median RTF
0.175882, and three identical one-edit results at 1.818% CER. The independently
integrated 120-sample worker telemetry gives 2,062.491 J and matches the
summary; cleanup records a zero return code, reaped worker, and absent
container. The five terminal pre-final direct attempts are retained as bounded
compatibility or harness failures, while the nonterminal 18:16:54 artifact is
explicitly excluded.

The four managed llama.cpp journals add 20 completed cases and 56 measured
request completions. All 20 aggregate rates, all 40 median TTFT/E2E fields,
eight applicable client-decode medians, and six client-TTFT prefill medians
independently recomputed from request and case events match their summaries
exactly (maximum absolute delta `0`). Thirty-eight request validations passed;
the remaining 18 one-token prefill requests intentionally had no quality
validator, and no request failed validation. The MTP acceptance ratios also
recompute exactly from the cumulative counters: `61/78 = 78.205128%` in smoke
and `906/1,479 = 61.257606%` in quick. Across 351 lifecycle telemetry samples,
all GPU-utilization, power, clock, and temperature fields are present. Every
run records an orderly native-server stop and terminal completion.

The two managed llama.cpp core journals add 28 completed cases and 244
measured request completions. Twenty-six valid aggregate rates, all 56 median
TTFT/E2E fields, ten applicable client-decode medians, and eight client-TTFT
prefill medians independently recompute exactly from request and case events
(maximum absolute delta `0`). The two D1024 summaries correctly suppress their
aggregate/decode rates after all ten requests stopped before budget. Across
the pair, 194 request validations passed, those ten D1024 validations failed,
and 40 one-token prefill requests intentionally had no validator. The journals
also record two 32K context skips and six unsupported capability skips. The
core MTP counters independently give `17,420 / 26,926 = 64.695833%` acceptance.
Both server logs account for all 274 generation-slot tasks exactly, and all
2,583 lifecycle telemetry samples contain the four GPU scalar fields.

The nine post-hardening journals add 57 completed cases and 315 measured
request completions: six runs are complete and Q8 core/IQ2 smoke/IQ2 core are
terminal partial results. Fifty-five numeric aggregate rates, all 114 median
TTFT/E2E fields, 20 applicable client-decode medians, and 14 client-TTFT prefill
medians independently recompute from journal events exactly (maximum absolute
delta `0`). There are 241 passed request validations, ten semantically
early-stopped D1024 failures, six invalid IQ2 JSON failures, and 58 one-token
prefill requests with no quality validator. The journals also record two
context skips and 18 profile-unsupported skips. All nine artifact-validation
events match their frozen digests; all 367 generation-slot tasks and releases
are accounted for; and all 3,864 telemetry samples contain the four GPU scalar
fields. Each run has an orderly native-server stop and terminal completion.

The two controls, six matrix children, and two confirmation journals add ten
completed cases and 80 measured request completions, all validation-passing.
All ten aggregate rates, 20 median TTFT/E2E fields, ten client-decode medians,
and four confirmation p95 fields recompute from the journals; every summary
value agrees. All eight MTP counter snapshots independently satisfy accepted-
position sums, acceptance, mean-length, and proposal-average arithmetic. Ten
artifact validations match the frozen digests, all 110 parent tasks have
matching prompt/total timing and release records, and all 1,430 telemetry
samples contain the four GPU scalar fields. All ten runs are terminal complete.

For the eleven runs newly reconciled in this final pass (IQ2 core plus that
depth experiment), the ledger totals are 24 cases, 202 measured requests, 172
passed validations, ten failures, 20 unvalidated one-token prefill requests,
one context skip, three unsupported skips, eleven artifact validations, 247
fully accounted generation tasks, and 2,463 telemetry samples. Recomputing 190
numeric/count fields and MTP invariants from the journals produced a maximum
absolute difference of `2.22e-16`, floating-point round-off.

Generated artifacts remain untracked as required; this report records stable
run IDs and the retained temporary artifact paths for local audit.
