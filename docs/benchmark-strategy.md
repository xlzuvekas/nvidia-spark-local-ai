# DGX Spark Benchmark Strategy

## Goal

Measure the useful limits of one DGX Spark, not just peak tokens per second. The
suite should separate model architecture, precision, serving engine, context
length, concurrency, and capability quality. Run one server at a time and record
the exact checkpoint revision, container digest, launch flags, software versions,
and system state with every result.

NVIDIA describes checkpoints in its Spark matrices as *supported* and “ready to
use,” not formally certified. Use these support labels in manifests and reports:

- `spark_vllm_matrix`: listed in NVIDIA's DGX Spark vLLM support matrix.
- `spark_vllm_recipe`: covered by a current Spark-specific vLLM recipe.
- `spark_other_backend`: validated on Spark with TRT-LLM, SGLang, or llama.cpp,
  but not in the current Spark vLLM matrix.
- `exploratory`: upstream-supported or locally working, without current official
  Spark validation. Local success does not promote this label automatically.

## Staged Plan

### 0. Inventory and cached-first run

Inventory Hugging Face and vLLM caches, Ollama models, Docker images, free disk,
and active services. Treat a Hugging Face model as available only when its exact
revision directory is present. Reject weight and image retrieval by default;
enable downloads explicitly only after producing a gap report with checkpoint
sizes. First run cached configurations, beginning with the repository's Qwen3.8
BF16, NVFP4, and NVFP4+MTP baselines. This validates the pipeline without
consuming bandwidth or changing the model inventory.

The 2026-08-14 cached Ollama sweep is complete at a fit-safe 32K served context;
see [its smoke results](benchmark-results-2026-08-14.md). Ollama's automatic 262K
allocation for the 128B Mistral artifact required about 165 GiB and OOMed. The
native API adapter now freezes `num_ctx` and verifies unload before advancing.

### 1. Architecture and precision anchors

Compare FP8 and NVFP4 on Qwen3-8B, then dense and MoE behavior with Qwen3-32B
and GPT-OSS-20B. Measure cold start, warm start, peak unified memory, prompt
throughput, decode throughput, TTFT, inter-token latency, energy, and thermal
stability. Treat weight precision and KV-cache precision as independent fields.

### 2. Capability tracks

Run task-specific protocols for reasoning, coding/tool use, image and document
understanding, multimodal retrieval, text embedding/reranking, and diffusion
decoding. Quality checks belong beside performance results; a faster malformed
tool call or incorrect retrieval result is not a win.

### 3. Backend comparison

Avoid a full model-by-backend Cartesian product. Use Qwen3-8B as the dense anchor
and GPT-OSS-20B as the MoE anchor across vLLM, TRT-LLM, and SGLang. Compare
Qwen3.6 MTP through vLLM and llama.cpp, noting that GGUF and NVFP4 weights are
different experiments. Use Ollama or LM Studio for deployment/API usability
smoke tests, not the core engine leaderboard.

### 4. Long-context and fit limits

Advance through 8K, 32K, 128K, and 262K prompts; attempt 1M only for checkpoints
that claim it. Record actual admitted tokens, KV allocation, prefill time, decode
degradation, and correctness with deterministic retrieval probes. Finish with the
70B dense and 120B MoE models. Stop before swap thrashing; an OOM or inability to
reserve a useful KV cache is a valid boundary result.

## Representative Checkpoint Matrix

| Track | Exact checkpoint | Support | Preferred backend |
| --- | --- | --- | --- |
| Dense precision anchor | `nvidia/Qwen3-8B-FP8`; `nvidia/Qwen3-8B-NVFP4` | `spark_vllm_matrix` | vLLM; TRT/SGLang cross-check |
| Mid-size dense | `nvidia/Qwen3-32B-NVFP4` | `spark_vllm_matrix` | vLLM |
| Large dense | `nvidia/Llama-3.3-70B-Instruct-NVFP4` | `spark_vllm_matrix` | vLLM |
| Latest dense, MTP, 262K | `Qwen/Qwen3.8-27B`; `Inferact/Qwen3.8-27B-NVFP4` | `exploratory` (locally validated) | vLLM |
| Agentic MoE, MTP, 262K | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `spark_vllm_recipe` | vLLM |
| Small reasoning MoE | `openai/gpt-oss-20b` | `spark_vllm_matrix` | vLLM/TRT/SGLang |
| Dense reasoning | `nvidia/Phi-4-reasoning-plus-NVFP4` | `spark_vllm_matrix` | vLLM |
| Coding specialist | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | `exploratory` | vLLM |
| Lightweight VLM | `nvidia/Qwen2.5-VL-7B-Instruct-NVFP4` | `spark_vllm_matrix` | vLLM |
| Omni reasoning | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | `spark_vllm_matrix` | vLLM |
| Multimodal retrieval | `Qwen/Qwen3-VL-Embedding-2B`; `Qwen/Qwen3-VL-Reranker-2B` | `spark_vllm_matrix` | vLLM |
| Text retrieval | `Qwen/Qwen3-Embedding-4B`; `Qwen/Qwen3-Reranker-4B` | `exploratory` | SentenceTransformers or pooling server |
| Diffusion LM | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | `spark_vllm_matrix` | vLLM |
| 120B fit/reasoning | `openai/gpt-oss-120b` | `spark_vllm_matrix` | vLLM/TRT |
| 120B, MTP, up to 1M | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | `spark_vllm_matrix` | vLLM/TRT |
| TRT multimodal alternate | `nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4` | `spark_other_backend` | TRT-LLM |

The official TRT-LLM matrix places `nvidia/Qwen3-235B-A22B-NVFP4` on **two
Sparks only**. Do not schedule it as a single-device target.

## Protocol Boundaries

- Pin prompt tokens, requested output tokens, sampling settings, concurrency,
  parsers, and chat template per test. Save raw responses and token counts.
- Report cold, warm, cached-prefix, and uncached-prefix results separately. Use
  unique prefill prompts when testing without prefix-cache benefit.
- Benchmark MTP off and on as separate configurations and report draft acceptance,
  not merely effective output rate.
- Do not compare text generation, VLM, embedding, and reranking throughput in one
  ranking. Use tokens/s, images/s, documents/s, vectors/s, or pairs/s as appropriate.
- A backend comparison requires equivalent model weights and precision. Otherwise
  label it a deployment comparison and include the quantization difference.
- Exclude model downloads, graph compilation, and cache flushing from steady-state
  timing, while retaining them as separate startup measurements.
- Capture competing processes, memory pressure, power mode, temperature, and clocks.
  Spark's CPU and GPU share the 128 GB memory pool and 273 GB/s memory interface.
- Treat integrated power samples as estimates: report sampled joules and the
  workload-specific unit per sampled joule, never as wall-power measurements.
- Use immutable v2 plans for new runs. Resume only canonical case identities and
  retain failed, unsupported, and context-limited boundaries in the journal.

## Primary Sources

- [NVIDIA DGX Spark vLLM support matrix](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md#model-support-matrix)
- [NVIDIA vLLM agent-ready recommendation](https://build.nvidia.com/spark/vllm/agent-ready-models)
- [Qwen3.6 Spark vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.8 vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [NVIDIA Spark TensorRT-LLM matrix](https://build.nvidia.com/spark/trt-llm/overview)
- [NVIDIA Spark SGLang matrix](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/sglang/README.md#model-support-matrix)
- [NVIDIA Spark llama.cpp playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/llama-cpp/README.md)
- [DGX Spark hardware specifications](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [Nemotron-3-Super model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)
- [GPT-OSS-120B model card](https://huggingface.co/openai/gpt-oss-120b)
- [Qwen3-Coder-30B model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
- [Qwen3 text embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- [Qwen3-VL embedding model card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
