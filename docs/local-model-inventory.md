# Local Model Inventory

> **Point-in-time snapshot:** 2026-08-14. This is a discovery record, not a
> live inventory. Sizes are approximate on-disk values and may include shared
> blobs, alternate layouts, metadata, or runtime caches.

## Storage Summary

| Store | Approximate size | Location |
| --- | ---: | --- |
| Project Hugging Face cache | 77 GiB | `data/huggingface/` |
| Project vLLM cache | 971 MiB | `data/vllm/` |
| User Hugging Face cache | 249 GiB | `~/.cache/huggingface/` |
| Ollama model store | 309 GiB | `/usr/share/ollama/.ollama/models/` |
| OpenAI Whisper cache | 3.7 GiB | `~/.cache/whisper/` |
| Detached Ollama Docker volume | 65.37 GB | `open-webui-ollama` |

The detached volume had no linked container and could not be inspected with
the current user's host permissions, so its model IDs remain unknown.

## Project Checkpoints

| Model | Format | Weight size | Reuse |
| --- | --- | ---: | --- |
| `Qwen/Qwen3.8-27B` | BF16 safetensors | 51.75 GiB | Dense baseline; multimodal architecture; 262K context |
| `Inferact/Qwen3.8-27B-NVFP4` | ModelOpt NVFP4 safetensors | 24.57 GiB | Quantized comparison; includes an MTP asset |

These checkpoints support BF16-versus-NVFP4 and MTP-on/off experiments without
another download. Preserve the mounted vLLM cache to reuse compiled graphs and
FlashInfer autotuning.

## User Hugging Face Cache

| Model | Format | Approximate weights | Primary workload |
| --- | --- | ---: | --- |
| `openai/gpt-oss-120b` | MXFP4 MoE | 121.54 GiB cached | 120B-class text and long context |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | BF16 MoE | 56.87 GiB | Coding and tool use |
| `nvidia/Nemotron-Labs-Diffusion-14B` | BF16 | 25.33 GiB | Diffusion language generation |
| `nvidia/Nemotron-Labs-Diffusion-8B` | BF16 | 15.95 GiB | Diffusion language generation |
| `mit-oasys/rlm-qwen3-8b-v0.1` | BF16 | 15.26 GiB | Recursive/long-form reasoning |
| `google/gemma-3-4b-it` | BF16 | 8.01 GiB | Small multimodal baseline |
| `z-lab/Qwen3-Coder-30B-A3B-DFlash` | BF16 draft model | 0.88 GiB | Speculative decoding with Qwen Coder |

The GPT-OSS snapshot contains two alternative 60.77 GiB weight layouts (root
and `original/`); both belong to the MXFP4 checkpoint and are not BF16 variants.
No incomplete Hugging Face downloads were found.

Smaller reusable assets include `SmolLM2-135M`, MiniLM embeddings in
safetensors and Q8_0 GGUF, Faster-Whisper tiny/base/small CTranslate2 models,
and Moonshine ONNX. `SparkAudio/Spark-TTS-0.5B` and `moonshine-base` contain
only references or metadata, not complete weights. The 785 MiB
`kernels-community/flash-attn3` entry is a kernel artifact rather than a model.

## Ollama Library

Ollama 0.20.6 reported no loaded model at capture time. Its ready-to-run store
contained:

| Model | Disk size | Quantization | Capabilities |
| --- | ---: | --- | --- |
| `mistral-medium-3.5:128b` | 80 GB | Q4_K_M | Vision, tools, thinking |
| `nemotron-3-super` | 86 GB | Q4_K_M | Tools, thinking |
| `llama3.3:70b-instruct-q4_K_M` | 42 GB | Q4_K_M | Text, tools |
| `nemotron-cascade-2` | 24 GB | Q4_K_M | Tools, thinking |
| `qwen3.5:35b-a3b` | 23 GB | Q4_K_M | Vision, tools, thinking |
| `gemma4:31b` | 19 GB | Q4_K_M | Vision, tools, thinking |
| `glm-4.7-flash` | 19 GB | Q4_K_M | Tools, thinking |
| `qwen3:30b-a3b-instruct-2507-q4_K_M` | 18 GB | Q4_K_M | Text, tools |
| `gemma3:12b` | 8.1 GB | Q4_K_M | Vision |
| `deepseek-ocr` | 6.7 GB | F16 | Vision and OCR |
| `dengcao/bge-reranker-v2-m3` | 1.2 GB | F16 | Reranking |
| `nomic-embed-text` | 274 MB | F16 | Embeddings |

## Speech and Runtime Assets

The Whisper PyTorch cache contains `large-v3-turbo.pt` (1.51 GiB), `medium.pt`
(1.42 GiB), `small.pt` (461 MiB), and base/base.en checkpoints (about 139 MiB
each). These enable a separate ASR latency and real-time-factor track.

Local images are already available for vLLM (`qwen38`, NVIDIA 26.07, and
latest) and SGLang 0.5.10rc0. No populated LM Studio, ModelScope, NIM, or SGLang
model cache was found. Benchmark manifests should therefore prefer the exact
cache paths above, distinguish runtime-image size from model size, and mark any
new download explicitly.
