# Single-Spark Model Candidates — 2026-08-15

## Scope and Method

This gap analysis compares the 2026-08-14 local inventory with NVIDIA's current
single-DGX-Spark [vLLM](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md#model-support-matrix),
[TensorRT-LLM](https://build.nvidia.com/spark/trt-llm/overview), and
[SGLang](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/sglang/README.md#model-support-matrix)
matrices. NVIDIA calls matrix entries “supported” and “ready to use”; this is
not a formal certification. Revisions are Hugging Face heads observed on
2026-08-15 and should be pinned before retrieval. Download sizes sum model
weight blobs and exclude containers. Fit ranges below are conservative starting
estimates, not local measurements.

The already-cached `nvcr.io/nvidia/vllm:26.07-py3` is vLLM 0.24.0 and should be
pinned as `sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268`.
It avoids another approximately 22 GB expanded image for most candidates.

## Ranked Downloads

| Rank | Exact checkpoint and revision | New coverage | Weights / initial fit | Terms and access |
| ---: | --- | --- | --- | --- |
| 1 | [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` | Text, image, screenshot, and video embeddings; 32K; 64–2048 dimensions | 3.963 GiB / under 16 GiB at low concurrency | Apache-2.0; ungated |
| 2 | [`Qwen/Qwen3-VL-Reranker-2B`](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) `4bd860ac4f15ad1897a214615cccc700f8f71818` | Completes a multimodal retrieve-then-rerank pipeline over the same modalities | 3.963 GiB / under 16 GiB | Apache-2.0; ungated |
| 3 | [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4) `dc5f0b0bfddf8b6e0f5891475be9af05b80126fe` | Unified video, audio, image, and text reasoning; ASR, OCR, documents, and GUI workflows; 256K native | 20.870 GiB / roughly 45–90 GiB at 32K–128K | NVIDIA Open Model Agreement; ungated; global |
| 4 | [`nvidia/Phi-4-multimodal-instruct-NVFP4`](https://huggingface.co/nvidia/Phi-4-multimodal-instruct-NVFP4) `617cfabb9ad6c2c6e318fd21c1961536b84f65a1` | Compact joint text, image, and speech model; 128K | 8.306 GiB / roughly 20–35 GiB at 32K | NVIDIA Open Model License plus MIT; ungated; card excludes EU deployment |
| 5 | [`nvidia/diffusiongemma-26B-A4B-it-NVFP4`](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4) `ec4ff3df205028f4e81c954c2227f9312b3ec2ea` | Discrete-diffusion block decoding, image/video understanding, tools, and JSON; 256K | 17.531 GiB / roughly 40–80 GiB at 32K | Apache-2.0 plus Gemma Terms and Prohibited Use Policy; ungated |
| 6 | [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) `491c2f1ea524c639598bf8fa787a93fed5a6fbce` | Official Spark MTP, agentic, image/video, and 262K comparator | 21.816 GiB / recipe budgets about 64 GiB | Apache-2.0; ungated; global |
| 7 | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) `6cee5e81ee83917806bbde320786a8fb61efebee` | Low-latency and fine-tunable MoE control for the cached 120B model | 12.817 GiB for vLLM shards / officially within 16 GB; reserve 20–32 GiB | Apache-2.0; ungated |
| 8 | [`nvidia/Qwen3-8B-NVFP4`](https://huggingface.co/nvidia/Qwen3-8B-NVFP4) `ccd10a893cbca613259517c3efe08e151ddf2b8e` plus [`nvidia/Qwen3-8B-FP8`](https://huggingface.co/nvidia/Qwen3-8B-FP8) `2cebc4c89e25abc17668c81b01dceaf3d8b914d5` | Controlled FP8-versus-NVFP4 and backend anchor; 131K native | 5.958 + 8.788 GiB / roughly 16–32 GiB each at 32K | Apache-2.0; ungated |

Ranks 1–3 maximize genuinely missing capability. Rank 4 is cheaper than rank 3,
but its narrower modality set and geographic license restriction reduce its
operational value. Ranks 6–8 add unusually useful controls, but overlap cached
model families.

## Deployment Profiles

For the Qwen3-VL pair, use the pinned local NGC image with a pooling runner,
BF16, and `--trust-remote-code`. The official embedding example specifies
`runner="pooling"`; the reranker additionally requires the vLLM
`qwen3_vl_reranker` template and these Hugging Face overrides:

```text
architectures = ["Qwen3VLForSequenceClassification"]
classifier_from_token = ["no", "yes"]
is_original_qwen3_reranker = true
```

The Omni card provides a Spark-specific `vllm/vllm-openai:v0.20.0` profile.
Install `vllm[audio]` in the container and start conservatively at 32K and
`--gpu-memory-utilization 0.70`; its published 128K profile uses:

```text
--trust-remote-code --max-model-len 131072 --max-num-seqs 8
--limit-mm-per-prompt '{"video":1,"image":1,"audio":1}'
--media-io-kwargs '{"video":{"fps":2,"num_frames":256}}'
--enable-prefix-caching --max-num-batched-tokens 32768
--reasoning-parser nemotron_v3 --enable-auto-tool-choice
--tool-call-parser qwen3_coder
```

Phi-4 multimodal is listed in the Spark vLLM matrix and explicitly requires
`--trust-remote-code`; use the pinned local NGC image and begin at 32K.

DiffusionGemma uses `vllm/vllm-openai:gemma`,
`VLLM_USE_V2_MODEL_RUNNER=1`, and:

```text
--trust-remote-code --attention-backend TRITON_ATTN --max-num-seqs 4
--diffusion-config '{"canvas_length":256}'
--override-generation-config '{"max_new_tokens":null}'
--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
--default-chat-template-kwargs '{"enable_thinking":true}'
```

Its card still labels the serving command tentative pending a supporting public
image, so verify and pin the image digest before downloading weights.

The official [Qwen3.6 recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B)
uses `vllm/vllm-openai:v0.24.0-ubuntu2404`, FP8 KV cache, Marlin MoE,
`--gpu-memory-utilization 0.5`, 262K context, eight sequences, chunked prefill,
async scheduling, prefix caching, fastsafetensors, Qwen3 reasoning/tools, and
MTP3 with a Triton MoE backend. The cached NGC 0.24 image is a sensible
zero-download compatibility trial.

```text
--trust-remote-code --kv-cache-dtype fp8 --moe-backend marlin
--gpu-memory-utilization 0.5 --max-model-len 262144 --max-num-seqs 8
--max-num-batched-tokens 8192 --enable-chunked-prefill
--async-scheduling --enable-prefix-caching --load-format fastsafetensors
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
--reasoning-parser qwen3 --tool-call-parser qwen3_coder
--enable-auto-tool-choice
```

GPT-OSS-20B should reuse the validated 120B NGC profile and Harmony format.
Download only root `model-*.safetensors` and required configuration: an
indiscriminate repository snapshot is 38.439 GiB because root, `original/`, and
Metal layouts duplicate the weights. Start with 32K,
`--gpu-memory-utilization 0.85`, eight sequences, chunked prefill, prefix caching,
`--reasoning-parser openai_gptoss`, and `--tool-call-parser openai` with
automatic tool choice.

## Deferred Candidates

| Candidate | Weights | Why defer |
| --- | ---: | --- |
| [`nvidia/Qwen2.5-VL-7B-Instruct-NVFP4`](https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-NVFP4) `d13bb1f2d8fbbd9f16cbb7688c8d6b6932d797d7` | 6.710 GiB | Existing VLMs cover image input; NVIDIA terms exclude EU. Use pinned NGC vLLM and 32K. |
| [`nvidia/Phi-4-reasoning-plus-NVFP4`](https://huggingface.co/nvidia/Phi-4-reasoning-plus-NVFP4) `cb950fd61cdbfa8e0f467ace3087c7d32ea8a47b` | 9.056 GiB | Dense reasoning overlaps GPT-OSS/Qwen; MIT card still excludes EU. Use NGC vLLM, 64K maximum. |
| [`nvidia/Qwen3-32B-NVFP4`](https://huggingface.co/nvidia/Qwen3-32B-NVFP4) `16426c6eb87be9e27c14cc9fb318f9c7a5f8588c` | 19.247 GiB | Valuable dense control, not a new task. Apache-2.0/ungated; generic NGC vLLM profile. |
| [`nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4`](https://huggingface.co/nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4) `9417590c9bc99b359f3ac66d1346ecd54249888e` | 60.860 GiB | TRT-LLM-only Spark matrix entry; gated contact sharing, NVIDIA plus Llama terms, attribution, and EU exclusion. Its 1M model limit is not a 1M single-Spark fit guarantee. Use `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` and start at 32K. |
| [`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) `4f0cf9daaeb7a4d5e23f80a00e7ed15f0e03caf6` | 74.802 GiB | The cached 80 GiB Ollama variant already covers its core capability. The exact NVFP4 checkpoint is mainly an MTP/backend boundary. NVIDIA Nemotron Open Model License; ungated/global. Official Spark image is `vllm/vllm-openai:v0.20.0`. |

Do not schedule the matrix's Qwen3-235B-A22B NVFP4 checkpoint: NVIDIA marks it
as a **two-Spark-only** TensorRT-LLM target.

## Pipeline Gates

Implement a multimodal pooling workload before rank 1, the currently missing
rerank adapter before rank 2, audio/video fixtures and task metrics before rank
3, and diffusion-block-aware timing before rank 5. This code-gated sequence
prevents another large collection of downloaded but unmeasurable assets.

Outside the LLM/VLM scope, image generation is the clearest device-level gap:
NVIDIA's [Spark ComfyUI playbook](https://build.nvidia.com/spark/comfy-ui/overview)
uses an approximately 2 GB Stable Diffusion checkpoint, while its
[TensorRT multimodal playbook](https://build.nvidia.com/spark/multi-modal-inference/overview)
covers FLUX.1 and SDXL in FP16, FP8, and FP4. Those require a separate
image-quality and latency pipeline rather than token metrics.
