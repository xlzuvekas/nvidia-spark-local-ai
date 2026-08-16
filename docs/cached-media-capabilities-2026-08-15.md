# Cached Media Capabilities — 2026-08-15

This is a read-only audit of media-capable artifacts already on the DGX Spark.
No GPU media inference was run because the SparkBench matrix held
`results/.sparkbench.lock`. Run the commands below only after the lock can be
acquired; each command holds it for its full lifetime and disables network
access at the library level.

## Verified inventory

| Capability | Local artifact | Offline execution status |
| --- | --- | --- |
| TTS and voice cloning | Spark-TTS 0.5B, revision `642071559bfc6346c2359d19dcb6be3f9dd8a05d`, 3,945,429,446 bytes | GPU-ready in the Homebrew venv |
| PyTorch ASR | Whisper `base`, `base.en`, `small`, `medium`, and `large-v3-turbo`, 3.7 GiB total | GPU-ready in the Homebrew venv |
| CTranslate2 ASR | Faster-Whisper `tiny.en`, `base.en`, and `small.en` | CPU-only: cached CTranslate2 has no CUDA/cuBLAS dependency |
| ONNX ASR | Moonshine base ONNX plus its separate config/tokenizer snapshot | CPU-only: cached ONNX Runtime has no CUDA provider |
| Fish Audio S2 | `fish-speech-spark:latest` | Blocked: image has bootstrap scripts, but source and `s2-pro/codec.pth` are absent |

The runnable environment is `/home/xlz/voice-cloning/.venv` (Homebrew Python
3.14, `openai-whisper==20250625`, `torch==2.10.0+cu128`,
`torchaudio==2.10.0+cu128`, `transformers==4.57.6`). The similarly named
`/home/xlz/whisper-cuda-venv` is **not** runnable: it contains only
`torch==2.10.0`, with no Whisper, Torchaudio, tokenizer, or FFmpeg bindings.

The shared fixture is
`/home/xlz/voice-cloning/Spark-TTS/example/prompt_audio.wav`: PCM S16LE,
16 kHz, mono, 9.953313 seconds, SHA-256
`335e7f7789b231cd90d9670292d561ecfe6a6bdd5e737a7bc6c29730741852de`.
Its reference transcript is:

> 吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。

## Spark-TTS GPU benchmark

The source checkout is commit `2f1ea9082400547242641f5271b6f941c9f439d1`
with local modifications to `cli/SparkTTS.py` and `requirements.txt`; record
those patches with any result. The three principal weight hashes are:

- BiCodec: `e9940cd48d4446e4340ced82d234bf5618350dd9f5db900ebe47a4fdb03867ec`
- LLM: `54825baf0a2f6076eb3c78fa1d22a95aee225f59070a8b295f8169db860eb109`
- Wav2Vec2: `314340227371a608f71adcd5f0de5933824fe77e55822aa4b24dba9c1c364dcb`

This produces three seeded clone trials and reports load time, inference time,
audio duration, real-time factor (RTF), throughput versus real time, and Torch
peak allocation. Results remain under `/tmp`, not in the repository.

Torch 2.10+cu128 supports kernels only through compute capability 12.0 and its
complex-absolute and TorchScript-fused Snake activation paths ask NVRTC for
unsupported `sm_121` on GB10. The benchmark therefore computes the equivalent
spectrogram magnitude as `sqrt(real^2 + imag^2)` and expands Snake into eager
real-valued operations. These narrow runtime monkeypatches leave the checked-out
Spark-TTS source unchanged and are recorded in the load event.

```bash
cd /home/xlz/bb-experiments/local-llm
flock -n results/.sparkbench.lock bash <<'SH'
set -euo pipefail
export MEDIA_OUT="$(mktemp -d /tmp/sparktts-bench.XXXXXX)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
cd /home/xlz/voice-cloning/Spark-TTS
/usr/bin/timeout --signal=TERM --kill-after=30s 30m \
  /home/xlz/voice-cloning/.venv/bin/python - <<'PY' \
  | tee "$MEDIA_OUT/metrics.jsonl"
import gc, hashlib, json, os, time
from pathlib import Path
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from cli.SparkTTS import SparkTTS
from sparktts.modules.blocks.layers import Snake1d

original_spectrogram = audio_functional.spectrogram
def sm121_spectrogram(
    waveform, pad, window, n_fft, hop_length, win_length, power, normalized,
    center=True, pad_mode="reflect", onesided=True, return_complex=None,
):
    complex_spectrogram = original_spectrogram(
        waveform, pad, window, n_fft, hop_length, win_length, None, normalized,
        center, pad_mode, onesided, return_complex,
    )
    if power is None:
        return complex_spectrogram
    magnitude = torch.sqrt(
        complex_spectrogram.real.square() + complex_spectrogram.imag.square()
    )
    return magnitude if power == 1.0 else magnitude.pow(power)
audio_functional.spectrogram = sm121_spectrogram

def sm121_snake_forward(self, x):
    scaled = self.alpha * x
    sine = torch.sin(scaled)
    squared = sine * sine
    inverse = torch.reciprocal(self.alpha + 1e-9)
    return x + inverse * squared
Snake1d.forward = sm121_snake_forward

model_dir = "/home/xlz/voice-cloning/pretrained_models/Spark-TTS-0.5B"
prompt_wav = "/home/xlz/voice-cloning/Spark-TTS/example/prompt_audio.wav"
prompt_text = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"
target_text = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。"
out = Path(os.environ["MEDIA_OUT"])
assert torch.cuda.is_available(), "CUDA is unavailable"
t0 = time.perf_counter()
model = SparkTTS(model_dir, torch.device("cuda:0"))
torch.cuda.synchronize()
print(json.dumps({"event": "load", "seconds": time.perf_counter() - t0,
                  "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                  "spectrogram_magnitude": "real_imag_sqrt_sm121_fallback",
                  "snake_activation": "eager_real_ops_sm121_fallback"}))
for trial in range(3):
    torch.manual_seed(3407 + trial)
    torch.cuda.manual_seed_all(3407 + trial)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        wav = model.inference(target_text, prompt_wav, prompt_text=prompt_text)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    path = out / f"trial-{trial}.wav"
    sf.write(path, wav, 16000)
    audio_s = float(wav.shape[-1]) / 16000
    print(json.dumps({"event": "trial", "trial": trial, "inference_s": elapsed,
          "audio_s": audio_s, "rtf": elapsed / audio_s,
          "audio_x_realtime": audio_s / elapsed,
          "peak_torch_bytes": torch.cuda.max_memory_allocated(),
          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "path": str(path)}))
del model
gc.collect()
torch.cuda.empty_cache()
PY
echo "outputs=$MEDIA_OUT"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
SH
```

Trial 0 includes first-use kernel effects; trials 1–2 are the warm comparison.
Spark-TTS samples tokens, so the fixed seeds are part of the fixture.

## Whisper PyTorch GPU benchmark

All cached files match the SHA-256 embedded in OpenAI Whisper's model URLs:

- `base`: `ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e`
- `base.en`: `25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead`
- `small`: `9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794`
- `medium`: `345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1`
- `large-v3-turbo`: `aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a`

Use multilingual models for this Chinese fixture; `base.en` is intentionally
excluded. Loading absolute checkpoint paths prevents a cache miss from becoming
a download.

```bash
cd /home/xlz/bb-experiments/local-llm
WHISPER_MODELS="base small medium large-v3-turbo" \
flock -n results/.sparkbench.lock bash <<'SH'
set -euo pipefail
export MEDIA_OUT="$(mktemp -d /tmp/whisper-bench.XXXXXX)"
export WHISPER_MODELS="${WHISPER_MODELS:-base}"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
/usr/bin/timeout --signal=TERM --kill-after=30s 45m \
  /home/xlz/voice-cloning/.venv/bin/python - <<'PY' \
  | tee "$MEDIA_OUT/metrics.jsonl"
import gc, json, os, time, unicodedata, wave
from pathlib import Path
import torch, whisper

audio = Path("/home/xlz/voice-cloning/Spark-TTS/example/prompt_audio.wav")
reference = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"
cache = Path("/home/xlz/.cache/whisper")
files = {name: cache / f"{name}.pt" for name in os.environ["WHISPER_MODELS"].split()}
with wave.open(str(audio), "rb") as f:
    audio_s = f.getnframes() / f.getframerate()

def normalize(text):
    return "".join(c.lower() for c in text if not unicodedata.category(c).startswith(("P", "Z")))

def distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        nxt = [i]
        for j, cb in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = nxt
    return row[-1]

assert torch.cuda.is_available(), "CUDA is unavailable"
for name, path in files.items():
    assert path.is_file(), path
    t0 = time.perf_counter()
    model = whisper.load_model(str(path), device="cuda")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    for trial in range(2):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = model.transcribe(str(audio), language="zh", task="transcribe",
            fp16=True, temperature=0.0, beam_size=5,
            condition_on_previous_text=False, verbose=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        hyp, ref = normalize(result["text"]), normalize(reference)
        print(json.dumps({"model": name, "trial": trial, "load_s": load_s,
              "inference_s": elapsed, "audio_s": audio_s,
              "rtf": elapsed / audio_s, "audio_x_realtime": audio_s / elapsed,
              "cer": distance(ref, hyp) / len(ref), "text": result["text"].strip(),
              "peak_torch_bytes": torch.cuda.max_memory_allocated()}, ensure_ascii=False))
    del model
    gc.collect()
    torch.cuda.empty_cache()
PY
echo "outputs=$MEDIA_OUT"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
SH
```

Trial 0 includes first-use overhead and trial 1 is warm. CER removes Unicode
punctuation and whitespace but otherwise preserves characters and digits.

## Process lifecycle and remaining assets

Both commands run one foreground Python process. Whisper invokes FFmpeg
synchronously to decode audio; neither path launches a server or daemon. Normal
exit or the enforced timeout destroys the CUDA context. The final `nvidia-smi`
query must not show the benchmark Python PID. If it does, investigate before the
next SparkBench model; do not start overlapping inference.

Other offline opportunities remain outside the media run: SmolLM2-135M can
exercise Unsloth inference/LoRA; the MiniLM Q8 GGUF can be imported into an
isolated Ollama embedding store; and the cached Qwen DFlash draft exactly pairs
with the cached Qwen3-Coder target in the SGLang image. The cached standalone
FlashAttention-3 binary is unusable on this machine: it contains only `sm_80`
and `sm_90a` cubins, no PTX, while GB10 reports compute capability 12.1.
