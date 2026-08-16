# Cached Training Capability — 2026-08-15

This track exercises raw causal generation and a 16-step LoRA update using only
artifacts already cached on the DGX Spark. It has not been GPU-run yet because
the active SparkBench matrix owns `results/.sparkbench.lock`. The command below
fails immediately when that lock is unavailable, cannot use the network, and
has no writable bind mount except a unique directory under `/tmp`.

## Pinned inputs

| Component | Exact local identity |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-135M` at revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| Weights | `model.safetensors`, 272 tensors, BF16, SHA-256 `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1` |
| Config | SHA-256 `1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843` |
| Tokenizer | `tokenizer.json` SHA-256 `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`; `tokenizer_config.json` SHA-256 `4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424` |
| Logical snapshot size | 272,437,465 bytes |
| Runtime image | `unsloth-dgx-spark:latest`, pinned below by local image ID `sha256:98261f554d5061eb8e3c05a94689d212567fa9d565c861539ed1c0ed61a96720` |

The model is a 30-layer, 576-hidden-size base causal LM with an 8,192-token
context window; it is not instruction-tuned. Mount the complete repository
cache, not only the snapshot directory, because snapshot entries are symlinks
to the sibling `blobs/` directory.

A network-disabled metadata check found Python 3.12.3, Torch
`2.10.0a0+b558c986e8.nv25.11`, Transformers 4.56.2, Unsloth 2026.6.1,
Unsloth Zoo 2026.6.1, PEFT 0.19.1, Accelerate 1.13.0, Datasets 4.3.0,
TRL 0.22.2, BitsAndBytes 0.49.2, Xformers
`0.0.33+aa7bc36.d20260604`, Triton `3.4.0+gitc5d671f9`, and
Safetensors 0.8.0rc1. Importing Unsloth without a GPU correctly stopped with
“cannot find any torch accelerator”; the actual CUDA path remains to be
validated by this track.

## Offline inference and LoRA benchmark

Run this only after the matrix is complete. It performs deterministic greedy
generation, then trains rank-8 adapters on a fixed in-memory fixture. The
training rate counts shifted, non-padding causal targets and includes a CUDA
synchronization after every step so per-step timings are meaningful.

```bash
cd /home/xlz/bb-experiments/local-llm
flock -n results/.sparkbench.lock bash <<'SH'
set -euo pipefail

IMAGE='sha256:98261f554d5061eb8e3c05a94689d212567fa9d565c861539ed1c0ed61a96720'
MODEL_CACHE='/home/xlz/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M'
REV='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
TRAIN_OUT="$(mktemp -d /tmp/smollm2-unsloth.XXXXXX)"
CONTAINER="sparkbench-unsloth-smol-$$"

test -f "$MODEL_CACHE/snapshots/$REV/model.safetensors"
cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

set +e
/usr/bin/timeout --signal=TERM --kill-after=60s 30m \
  docker run --rm -i --init --name "$CONTAINER" --pull never --gpus all \
    --network none --read-only --pids-limit 1024 \
    --memory 24g --memory-swap 24g --shm-size 2g \
    --tmpfs /tmp:rw,nosuid,nodev,exec,size=8g \
    -e HOME=/tmp/home -e XDG_CACHE_HOME=/tmp/cache \
    -e HF_HOME=/tmp/hf -e HF_DATASETS_CACHE=/tmp/hf-datasets \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HF_DATASETS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONHASHSEED=3407 \
    -e CUDA_VISIBLE_DEVICES=0 -e CUDA_CACHE_PATH=/tmp/cuda-cache \
    -e TRITON_CACHE_DIR=/tmp/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    --mount "type=bind,src=$MODEL_CACHE,dst=/model-cache,readonly" \
    --mount "type=bind,src=$TRAIN_OUT,dst=/output" \
    --workdir /tmp --entrypoint /usr/bin/timeout "$IMAGE" \
    --signal=TERM --kill-after=30s 25m python - <<'PY' \
  | tee "$TRAIN_OUT/console.log"
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
import random
import time
from pathlib import Path

import torch
from unsloth import FastLanguageModel
import peft.tuners.lora.torchao as peft_torchao

# The pinned image carries torchao 0.14 while PEFT's optional dispatcher now
# requires >0.16. This BF16 base model does not use TorchAO, so fail that
# optional availability check closed instead of rejecting ordinary nn.Linear.
peft_torchao.is_torchao_available = lambda: False

SEED = 3407
STEPS = 16
BATCH = 4
SEQ = 256
REV = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL = Path("/model-cache/snapshots") / REV
EXPECTED_FILES = {
    "model.safetensors": "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
    "config.json": "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843",
    "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "tokenizer_config.json": "4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424",
}
IMAGE_ID = "sha256:98261f554d5061eb8e3c05a94689d212567fa9d565c861539ed1c0ed61a96720"
OUT = Path("/output")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

assert os.environ.get("HF_HUB_OFFLINE") == "1"
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, "expected one visible GPU"
assert torch.cuda.get_device_capability(0) == (12, 1), "not a GB10 Spark GPU"
file_hashes = {name: sha256(MODEL / name) for name in EXPECTED_FILES}
assert file_hashes == EXPECTED_FILES, file_hashes

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
started = time.perf_counter()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(MODEL),
    max_seq_length=SEQ,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    full_finetuning=False,
    local_files_only=True,
)
torch.cuda.synchronize()
load_s = time.perf_counter() - started
load_peak_allocated = torch.cuda.max_memory_allocated()
load_peak_reserved = torch.cuda.max_memory_reserved()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
prompt = "The DGX Spark is useful for local language-model experiments because"
encoded = tokenizer(prompt, return_tensors="pt")
encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
model.eval()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
started = time.perf_counter()
with torch.inference_mode():
    generated = model.generate(
        **encoded,
        do_sample=False,
        min_new_tokens=64,
        max_new_tokens=64,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
torch.cuda.synchronize()
inference_s = time.perf_counter() - started
new_tokens = generated.shape[1] - encoded["input_ids"].shape[1]
completion = tokenizer.decode(
    generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True
)
inference_peak_allocated = torch.cuda.max_memory_allocated()
inference_peak_reserved = torch.cuda.max_memory_reserved()

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
started = time.perf_counter()
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,
    random_state=SEED,
    max_seq_length=SEQ,
    use_rslora=False,
    loftq_config=None,
)
model.train()
model.config.use_cache = False
torch.cuda.synchronize()
lora_setup_s = time.perf_counter() - started
lora_setup_peak_allocated = torch.cuda.max_memory_allocated()

fixture = (
    "Local language model benchmarks must pin weights, runtime, prompts, "
    "seeds, and hardware. Measurements should separate cold startup from "
    "warm inference and should never overlap GPU workloads. "
) * 80
fixture_hash = hashlib.sha256(fixture.encode()).hexdigest()
tokens = tokenizer(fixture, return_tensors="pt", add_special_tokens=True)["input_ids"]
assert tokens.shape[1] >= SEQ
input_ids = tokens[:, :SEQ].repeat(BATCH, 1).to("cuda:0")
attention_mask = torch.ones_like(input_ids)
labels = input_ids.clone()
trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
optimizer = torch.optim.AdamW(
    trainable, lr=5e-4, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=0.0, foreach=False,
)
optimizer.zero_grad(set_to_none=True)

losses = []
step_seconds = []
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
train_started = time.perf_counter()
for step in range(STEPS):
    torch.cuda.synchronize()
    step_started = time.perf_counter()
    loss = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    ).loss
    assert torch.isfinite(loss), f"non-finite loss at step {step}"
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    step_seconds.append(time.perf_counter() - step_started)
    losses.append(float(loss.detach().cpu()))
train_s = time.perf_counter() - train_started
train_peak_allocated = torch.cuda.max_memory_allocated()
train_peak_reserved = torch.cuda.max_memory_reserved()

adapter_dir = OUT / "adapter"
started = time.perf_counter()
model.save_pretrained(adapter_dir, safe_serialization=True)
tokenizer.save_pretrained(adapter_dir)
save_s = time.perf_counter() - started
# The container runs as root, but the host-side integrity step must be able to
# read the saved adapter without changing ownership of the mounted temp root.
for path in adapter_dir.rglob("*"):
    path.chmod(0o755 if path.is_dir() else 0o644)
adapter_dir.chmod(0o755)

predicted_tokens_per_step = BATCH * (SEQ - 1)
total_predicted_tokens = STEPS * predicted_tokens_per_step
steady_s = sum(step_seconds[1:])
x_mean = (STEPS - 1) / 2
y_mean = sum(losses) / STEPS
loss_slope = sum(
    (index - x_mean) * (value - y_mean) for index, value in enumerate(losses)
) / sum((index - x_mean) ** 2 for index in range(STEPS))
packages = {
    name: metadata.version(name)
    for name in (
        "accelerate", "bitsandbytes", "datasets", "peft", "safetensors",
        "transformers", "triton", "trl", "unsloth", "unsloth_zoo", "xformers",
    )
}
metrics = {
    "schema_version": 1,
    "image_id": IMAGE_ID,
    "model": {
        "repo": "HuggingFaceTB/SmolLM2-135M",
        "revision": REV,
        "file_sha256": file_hashes,
        "snapshot_path_in_container": str(MODEL),
    },
    "runtime": {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torchao_dispatch_disabled_for_bf16_lora": True,
        "packages": packages,
    },
    "load": {
        "elapsed_s": load_s,
        "peak_allocated_bytes": load_peak_allocated,
        "peak_reserved_bytes": load_peak_reserved,
    },
    "inference": {
        "mode": "raw_causal_greedy",
        "prompt": prompt,
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
        "generated_tokens": int(new_tokens),
        "elapsed_s": inference_s,
        "generated_tokens_s": float(new_tokens) / inference_s,
        "peak_allocated_bytes": inference_peak_allocated,
        "peak_reserved_bytes": inference_peak_reserved,
        "completion": completion,
    },
    "lora_setup": {
        "elapsed_s": lora_setup_s,
        "rank": 8,
        "alpha": 8,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_bytes": lora_setup_peak_allocated,
    },
    "training": {
        "steps": STEPS,
        "batch_size": BATCH,
        "sequence_length": SEQ,
        "learning_rate": 5e-4,
        "fixture_sha256": fixture_hash,
        "predicted_tokens": total_predicted_tokens,
        "elapsed_s": train_s,
        "predicted_tokens_s": total_predicted_tokens / train_s,
        "steady_predicted_tokens_s_excluding_step_0": (
            (STEPS - 1) * predicted_tokens_per_step / steady_s
        ),
        "step_seconds": step_seconds,
        "losses": losses,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_delta_last_minus_first": losses[-1] - losses[0],
        "loss_slope_per_step": loss_slope,
        "loss_decreased": losses[-1] < losses[0],
        "peak_allocated_bytes": train_peak_allocated,
        "peak_reserved_bytes": train_peak_reserved,
    },
    "save": {"elapsed_s": save_s, "adapter_dir": str(adapter_dir)},
}

del optimizer, trainable, labels, attention_mask, input_ids, generated, model
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
metrics["cleanup"] = {
    "allocated_bytes_after_empty_cache": torch.cuda.memory_allocated(),
    "reserved_bytes_after_empty_cache": torch.cuda.memory_reserved(),
    "container_exit_releases_cuda_context": True,
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics))
PY
pipeline_status=("${PIPESTATUS[@]}")
set -e

cleanup
trap - EXIT INT TERM
if (( pipeline_status[0] != 0 )); then
  echo "benchmark failed with status ${pipeline_status[0]}; outputs=$TRAIN_OUT" >&2
  exit "${pipeline_status[0]}"
fi
if (( pipeline_status[1] != 0 )); then
  echo "tee failed with status ${pipeline_status[1]}; outputs=$TRAIN_OUT" >&2
  exit "${pipeline_status[1]}"
fi
test -s "$TRAIN_OUT/metrics.json"
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "container cleanup failed: $CONTAINER" >&2
  exit 1
fi
find "$TRAIN_OUT" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > "$TRAIN_OUT/SHA256SUMS"
echo "outputs=$TRAIN_OUT"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
SH
```

The output directory contains `metrics.json`, the raw console log, a saved
adapter/tokenizer, and `SHA256SUMS`. Retain it only long enough to review or
copy the desired aggregate into a tracked result note; deleting that one
printed `/tmp/smollm2-unsloth.*` directory removes every persistent benchmark
artifact.

## Acceptance and interpretation

A valid run must preserve the pinned image, revision, and weight hash; report a
GB10 compute capability of 12.1; generate exactly 64 tokens; complete all 16
finite-loss updates; write the adapter; and leave no named container. Report
both end-to-end and step-0-excluded training rates because the first step can
include Triton compilation. Treat a falling loss as a plumbing signal only:
the repeated synthetic fixture is intentionally tiny and says nothing about
generalization or model quality. Peak Torch allocation is comparable within
this command but is not total system or unified-memory use.

The earlier package-metadata container overlapped the MTP3 matrix startup for
about 3.4 seconds after preflight. Label that MTP3 `startup_s` measurement
**diagnostically contaminated**. Preserve its measured request cases only if
the event timeline confirms the metadata probe exited before `server_ready`;
the training command above must not be launched until the matrix releases the
lock.
