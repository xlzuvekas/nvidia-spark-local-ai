"""Plan expansion and resumable execution for text-generation benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import base64
import fcntl
import functools
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import struct
import time
from types import SimpleNamespace
from typing import Any
import unicodedata
import uuid
import zlib

from .client import (
    concurrent_chat_requests,
    concurrent_multimodal_embedding_requests,
    concurrent_score_requests,
    embedding_request,
    multimodal_embedding_request,
    score_request,
    stream_audio_chat_request,
    stream_chat_request,
    stream_ollama_chat_request,
)
from .journal import Journal, content_hash, utc_now, write_json
from .llamacpp_metrics import (
    llamacpp_dflash_requested,
    llamacpp_mtp_requested,
    require_llamacpp_dflash_evidence,
    require_llamacpp_mtp_evidence,
    require_mtp_activity,
    require_speculative_activity,
    snapshot_llamacpp_spec_decode_metrics,
)
from .report import summarize_run
from .runtime import (
    RuntimeErrorWithContext,
    capture_server_provenance,
    ollama_model_loaded,
    recover_owned_llamacpp,
    recover_owned_sglang,
    recover_owned_vllm,
    save_server_logs,
    start_server,
    validate_llamacpp_artifacts,
)
from .telemetry import TelemetrySampler
from .vllm_metrics import snapshot_vllm_spec_decode_metrics


class PreflightError(RuntimeError):
    pass


def _command_output(command: list[str]) -> str | None:
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    return result.stdout.strip() if result.returncode == 0 else None


def _image_digest(image: str | None) -> str | None:
    if not image:
        return None
    output = _command_output(
        ["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"]
    )
    if not output:
        return None
    try:
        digests = json.loads(output)
    except json.JSONDecodeError:
        return None
    return digests[0] if digests else None


def _host_snapshot() -> dict[str, Any]:
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, raw = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            meminfo[f"{name.lower()}_kib"] = int(raw.strip().split()[0])
    return {
        "captured_at": utc_now(),
        "uname": _command_output(["uname", "-a"]),
        "python": _command_output(["python3", "--version"]),
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "docker": _command_output(["docker", "version", "--format", "{{.Server.Version}}"]),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_status": _command_output(["git", "status", "--short"]),
        **meminfo,
    }


def _canonical_case(model: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    payload = {"model": model, "case": case}
    return {**case, "case_id": f"{case['id']}--{content_hash(payload, 12)}"}


def create_plan(
    *, model: Any, suite: Any, results_root: Path, models_path: Path, suite_path: Path
) -> Path:
    if str(getattr(model, "support_status", "")) == "incompatible":
        raise RuntimeError("Incompatible model profiles cannot be planned")
    direct_commands = {
        "transformers": "diffusion-direct",
        "trtllm": "trtllm-direct",
    }
    direct_command = direct_commands.get(str(model.backend))
    if direct_command:
        raise RuntimeError(
            f"{model.backend} direct profiles require the {direct_command} command"
        )
    model_data = asdict(model)
    suite_data = asdict(suite)
    cases = [_canonical_case(model_data, case) for case in suite_data["cases"]]
    resolved_image = _image_digest(model.image)
    if model.image_digest and (
        not resolved_image or not resolved_image.endswith("@" + model.image_digest)
    ):
        raise RuntimeError(
            f"Local image digest for {model.image} does not match manifest {model.image_digest}"
        )
    resolved: dict[str, Any] = {"image_digest": resolved_image}
    if str(model.backend) == "llamacpp":
        resolved["llamacpp"] = validate_llamacpp_artifacts(
            model, workspace=models_path.resolve().parents[1]
        )
    fingerprint = content_hash(
        {"model": model_data, "suite": suite_data, "resolved": resolved}
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_root / f"{stamp}-{model.id}-{suite.id}-{fingerprint[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema_version": 2,
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "models_manifest": str(models_path),
        "suite_manifest": str(suite_path),
        "model": model_data,
        "suite": {**suite_data, "cases": cases},
        "resolved": resolved,
        "host_at_plan": _host_snapshot(),
    }
    plan["integrity_hash"] = content_hash(plan, 64)
    write_json(run_dir / "plan.json", plan)
    write_json(run_dir / "inventory.json", _host_snapshot())
    return run_dir


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {key: _plain(item) for key, item in vars(value).items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _preflight(model: SimpleNamespace) -> None:
    running = _command_output(["docker", "ps", "--format", "{{.Names}}"])
    if running is None:
        raise PreflightError("Could not verify running Docker containers")
    backend = str(getattr(model, "backend", ""))
    managed_name = (
        f"sparkbench-{backend}" if backend in {"sglang", "vllm"} else None
    )
    containers = [
        name for name in running.splitlines() if name != managed_name
    ]
    if containers:
        raise PreflightError("Unrelated containers are running: " + ", ".join(containers))
    compute_apps = None
    for attempt in range(11):
        compute_apps = _command_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ]
        )
        if compute_apps is None:
            raise PreflightError("Could not verify active GPU compute processes")
        if not compute_apps:
            break
        if attempt < 10:
            time.sleep(0.5)
    if compute_apps:
        raise PreflightError("Unrelated GPU compute processes are active: " + compute_apps)
    available_kib = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available_kib = int(line.split()[1])
            break
    required_gib = float(getattr(model, "estimated_ram_gib", 0) or 0) + 8
    if available_kib / 1024**2 < required_gib:
        raise PreflightError(
            f"Only {available_kib / 1024**2:.1f} GiB available; model plus reserve needs {required_gib:.1f} GiB"
        )


def _needle(nonce: str) -> str:
    return "SPARK-" + hashlib.sha256(nonce.encode()).hexdigest()[:10].upper()


def _solid_color_png_data_url(
    red: int, green: int, blue: int, size: int = 64
) -> str:
    width = height = max(16, min(size, 2048))
    pixel = bytes((red, green, blue))
    raw = b"".join(b"\x00" + pixel * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _solid_red_png_data_url(size: int = 64) -> str:
    return _solid_color_png_data_url(255, 0, 0, size)


def _solid_blue_png_data_url(size: int = 64) -> str:
    return _solid_color_png_data_url(0, 0, 255, size)


_AUDIO_FIXTURE_PATH = Path(
    "/home/xlz/voice-cloning/Spark-TTS/example/prompt_audio.wav"
)
_AUDIO_FIXTURE_SHA256 = (
    "335e7f7789b231cd90d9670292d561ecfe6a6bdd5e737a7bc6c29730741852de"
)
_AUDIO_EXPECTED_TRANSCRIPTION = (
    "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。"
    "豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"
)
_AUDIO_PROMPT = "Transcribe the audio clip into text."
_AUDIO_LORA_NAME = "speech"


def _normalize_audio_transcription(content: str) -> str:
    """Normalize punctuation and width while preserving transcript wording."""

    normalized = unicodedata.normalize("NFKC", content)
    return "".join(character for character in normalized if character.isalnum()).casefold()


_OCR_EXPECTED_TRANSCRIPTION = "SPARKOCR4827"
# DeepSeek-OCR is unusually sensitive to this exact, punctuated prompt; Ollama's
# model card and the upstream reference both use it for layout-free OCR.
_OCR_PROMPT = "Free OCR."
_OCR_IMAGE_WIDTH = 1024
_OCR_IMAGE_HEIGHT = 256
_OCR_GLYPH_WIDTH = 5
_OCR_GLYPH_HEIGHT = 7
_OCR_GLYPH_SCALE = 12
_OCR_GLYPHS = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("11111", "10000", "10000", "11111", "00001", "00001", "11111"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "4": ("00100", "01100", "10100", "11111", "00100", "00100", "00100"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
}


@functools.lru_cache(maxsize=1)
def _ocr_png_data_url() -> str:
    """Render the fixed OCR token into a dependency-free, deterministic PNG."""

    pixels = bytearray(b"\xff" * (_OCR_IMAGE_WIDTH * _OCR_IMAGE_HEIGHT * 3))
    token_width = (
        len(_OCR_EXPECTED_TRANSCRIPTION) * _OCR_GLYPH_WIDTH
        + len(_OCR_EXPECTED_TRANSCRIPTION)
        - 1
    ) * _OCR_GLYPH_SCALE
    start_x = (_OCR_IMAGE_WIDTH - token_width) // 2
    start_y = (
        _OCR_IMAGE_HEIGHT - _OCR_GLYPH_HEIGHT * _OCR_GLYPH_SCALE
    ) // 2
    for character_index, character in enumerate(_OCR_EXPECTED_TRANSCRIPTION):
        glyph = _OCR_GLYPHS[character]
        glyph_x = start_x + character_index * (
            _OCR_GLYPH_WIDTH + 1
        ) * _OCR_GLYPH_SCALE
        for glyph_y, row in enumerate(glyph):
            for glyph_column, enabled in enumerate(row):
                if enabled != "1":
                    continue
                left = glyph_x + glyph_column * _OCR_GLYPH_SCALE
                top = start_y + glyph_y * _OCR_GLYPH_SCALE
                for pixel_y in range(top, top + _OCR_GLYPH_SCALE):
                    offset = (pixel_y * _OCR_IMAGE_WIDTH + left) * 3
                    pixels[offset : offset + _OCR_GLYPH_SCALE * 3] = (
                        b"\x00" * (_OCR_GLYPH_SCALE * 3)
                    )

    raw = b"".join(
        b"\x00"
        + pixels[
            row * _OCR_IMAGE_WIDTH * 3 : (row + 1) * _OCR_IMAGE_WIDTH * 3
        ]
        for row in range(_OCR_IMAGE_HEIGHT)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            struct.pack(
                ">IIBBBBB",
                _OCR_IMAGE_WIDTH,
                _OCR_IMAGE_HEIGHT,
                8,
                2,
                0,
                0,
                0,
            ),
        )
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _normalize_ocr_transcription(content: str) -> str:
    """Apply Unicode normalization and trim framing whitespace only."""

    return unicodedata.normalize("NFKC", content).strip()


_RERANK_QUERY = "Which planet is known as the Red Planet?"
_RERANK_RELEVANT_INDEX = 1
_RERANK_BASE_CANDIDATES = (
    "Venus is the second planet from the Sun and has a dense atmosphere.",
    "Mars is widely known as the Red Planet because iron oxides color its surface.",
    "Jupiter is the largest planet in the Solar System and is a gas giant.",
    "Neptune is an ice giant with strong winds and a deep blue appearance.",
)
_RERANK_DISTRACTORS = (
    "Saturn has a prominent ring system made mostly of ice particles.",
    "Mercury is the closest planet to the Sun and has a cratered surface.",
    "Uranus rotates on its side and is classified as an ice giant.",
    "Earth has liquid surface oceans and a nitrogen-rich atmosphere.",
)
_MULTIMODAL_RELEVANT_TEXT = "A solid red square with no other colors or objects."
_MULTIMODAL_UNRELATED_TEXT = (
    "A photograph of snow-covered mountains beneath a cloudy blue sky."
)
_MULTIMODAL_RERANK_QUERY = "A solid red square with no other colors or objects."
_MULTIMODAL_RERANK_INSTRUCTION = (
    "Retrieve images relevant to the user's query."
)
_MULTIMODAL_RERANK_CANDIDATE_COUNT = 2
_MULTIMODAL_RERANK_RELEVANT_INDEX = 1


@dataclass(frozen=True, slots=True)
class QualityItem:
    """One embedded exact-answer item; model output is always treated as text."""

    id: str
    category: str
    question: str
    expected_answer: str


_QUALITY_ITEMS = (
    QualityItem(
        id="arithmetic-01",
        category="arithmetic",
        question="What is (17 * 6) - 19?",
        expected_answer="83",
    ),
    QualityItem(
        id="logic-01",
        category="logic",
        question=(
            "All flerns are torps. No torps are silver. Can any flern be silver? "
            "Answer yes or no."
        ),
        expected_answer="no",
    ),
    QualityItem(
        id="instruction-01",
        category="instruction_following",
        question="Return the third word in this sequence: amber cobalt silver jade.",
        expected_answer="silver",
    ),
    QualityItem(
        id="code-01",
        category="code_reasoning",
        question=(
            "Without running it, what integer does this Python snippet print?\n"
            "```python\n"
            "values = [2, 5, 8]\n"
            "total = 0\n"
            "for value in values:\n"
            "    if value % 2 == 0:\n"
            "        total += value\n"
            "    else:\n"
            "        total -= 1\n"
            "print(total)\n"
            "```"
        ),
        expected_answer="9",
    ),
)
_QUALITY_FINAL_PATTERN = re.compile(
    r"^\s*(?:\*\*|__|`)?final"
    r"[ \t]*"
    r"(?::(?:\*\*|__|`)[ \t]*|:[ \t]*|(?:\*\*|__|`)[ \t]*:)"
    r"\s*(?P<answer>.+?)\s*$",
    re.IGNORECASE,
)
_QUALITY_FINAL_OCCURRENCE_PATTERN = re.compile(
    r"\bfinal(?::\s*(?:\*\*|__|`)?|(?:\*\*|__|`)?\s*:)",
    re.IGNORECASE,
)
_NUMERIC_ANSWER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def _is_multimodal_embedding_case(case: SimpleNamespace) -> bool:
    return {"embeddings", "vision"}.issubset(set(case.requires))


def _is_multimodal_rerank_case(case: SimpleNamespace) -> bool:
    return {"rerank", "vision"}.issubset(
        set(getattr(case, "requires", ()))
    )


def _is_diffusion_model(model: Any) -> bool:
    """Return whether generation uses non-autoregressive diffusion semantics."""

    return str(getattr(model, "architecture", "")).strip().lower() == "diffusion-lm"


def _request_result_payload(model: Any, result: Any) -> dict[str, Any]:
    """Serialize a result without assigning AR timing semantics to diffusion output."""

    payload = result.to_dict()
    if not _is_diffusion_model(model):
        return payload
    payload.update(
        {
            "time_to_first_emission_s": payload.get("ttft_s"),
            "block_generation_output_tps": payload.get("output_tps"),
            "block_generation_metric_source": (
                "client_completion_tokens_per_end_to_end_request_elapsed"
            ),
            "ttft_s": None,
            "decode_s": None,
            "decode_tps": None,
            "decode_metric_source": None,
            "output_tps": None,
        }
    )
    return payload


def _rerank_inputs(case: SimpleNamespace) -> tuple[str, list[str], int]:
    requested = int(case.prompt_repetitions)
    candidate_count = max(requested, 2) if requested else len(_RERANK_BASE_CANDIDATES)
    candidates = list(_RERANK_BASE_CANDIDATES[:candidate_count])
    while len(candidates) < candidate_count:
        index = len(candidates)
        distractor = _RERANK_DISTRACTORS[
            (index - len(_RERANK_BASE_CANDIDATES)) % len(_RERANK_DISTRACTORS)
        ]
        candidates.append(f"{distractor} Candidate record {index:03d}.")
    return _RERANK_QUERY, candidates, _RERANK_RELEVANT_INDEX


def _multimodal_rerank_inputs(
    case: SimpleNamespace,
) -> tuple[str, list[dict[str, Any]], int]:
    image_size = int(case.prompt_repetitions) or 64
    documents = [
        {
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _solid_blue_png_data_url(image_size)},
                }
            ]
        },
        {
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _solid_red_png_data_url(image_size)},
                }
            ]
        },
    ]
    return (
        _MULTIMODAL_RERANK_QUERY,
        documents,
        _MULTIMODAL_RERANK_RELEVANT_INDEX,
    )


def _quality_prompt(item: QualityItem, nonce: str) -> str:
    return (
        f"Benchmark exact-answer item {item.id}; nonce {nonce}.\n"
        f"{item.question}\n"
        "Return one line exactly in the form `FINAL: <answer>`. "
        "Do not include an explanation."
    )


def _extract_quality_answer(content: str) -> tuple[str | None, str | None]:
    """Extract an explicit final answer without interpreting or executing output."""

    if len(_QUALITY_FINAL_OCCURRENCE_PATTERN.findall(content)) > 1:
        return None, "response contained multiple FINAL markers"
    lines = content.strip().splitlines()
    if (
        len(lines) >= 2
        and re.fullmatch(r"\s*```[A-Za-z0-9_-]*\s*", lines[0])
        and re.fullmatch(r"\s*```\s*", lines[-1])
    ):
        lines = lines[1:-1]
    nonempty = [line.strip() for line in lines if line.strip()]
    marked: list[str] = []
    for line in nonempty:
        marker_line = line
        for wrapper in ("**", "__", "`"):
            if len(marker_line) > 2 * len(wrapper) and marker_line.startswith(
                wrapper
            ) and marker_line.endswith(wrapper):
                marker_line = marker_line[len(wrapper) : -len(wrapper)].strip()
                break
        match = _QUALITY_FINAL_PATTERN.fullmatch(marker_line)
        if match:
            marked.append(match.group("answer").strip())
    if marked:
        return marked[0], None
    if len(nonempty) == 1:
        return nonempty[0], None
    return None, "response had neither one FINAL marker nor one bare-answer line"


def _normalize_quality_answer(answer: str) -> str:
    value = " ".join(answer.strip().split())
    paired_wrappers = (
        ("**", "**"),
        ("__", "__"),
        ("`", "`"),
        ('"', '"'),
        ("'", "'"),
    )
    for _ in range(3):
        previous = value
        if value.endswith("."):
            value = value[:-1].rstrip()
        for opening, closing in paired_wrappers:
            if len(value) > len(opening) + len(closing) and value.startswith(
                opening
            ) and value.endswith(closing):
                value = value[len(opening) : -len(closing)].strip()
                break
        if value == previous:
            break
    return value.casefold()


def _quality_answers_match(extracted: str, expected: str) -> bool:
    actual = _normalize_quality_answer(extracted)
    target = _normalize_quality_answer(expected)
    if _NUMERIC_ANSWER_PATTERN.fullmatch(actual) and _NUMERIC_ANSWER_PATTERN.fullmatch(
        target
    ):
        try:
            return Decimal(actual) == Decimal(target)
        except InvalidOperation:
            return False
    return actual == target


def _validate_quality_item(item: QualityItem, result: Any) -> dict[str, Any]:
    extracted, extraction_error = _extract_quality_answer(str(result.content))
    passed = bool(
        extraction_error is None
        and extracted is not None
        and _quality_answers_match(extracted, item.expected_answer)
    )
    reason = extraction_error
    if reason is None and not passed:
        reason = "extracted answer did not match the exact answer key"
    return {
        "passed": passed,
        "reason": reason,
        "quality_item_id": item.id,
        "quality_category": item.category,
        "expected_answer": item.expected_answer,
        "extracted_answer": extracted[:160] if extracted is not None else None,
    }


def _prompt(case: SimpleNamespace, nonce: str) -> str:
    kind = str(case.kind)
    prompt_repetitions = int(case.prompt_repetitions)
    prefix = f"Benchmark nonce {nonce}. "
    if kind == "prefill":
        return prefix + "Read all text and reply with exactly one word. " + (
            "measurement " * prompt_repetitions
        )
    if kind == "capability" and "json" in case.requires:
        return prefix + 'Return only a JSON object with keys "benchmark" set to "spark" and "value" set to 42.'
    if kind == "capability" and "tools" in case.requires:
        return prefix + "Use the multiply tool to multiply 6 by 7. Do not answer without calling the tool."
    if kind == "capability" and str(case.id).startswith("long-context-needle"):
        filler = "archive " * max(prompt_repetitions, 1)
        midpoint = len(filler) // 2
        needle = _needle(nonce)
        return (
            prefix
            + filler[:midpoint]
            + f" The unique benchmark key is {needle}. "
            + filler[midpoint:]
            + " What is the unique benchmark key? Reply with only the key."
        )
    return (
        prefix
        + "Write an unbroken numbered list of distinct two-word phrases. "
        "Continue until the output limit; do not conclude or summarize."
    )


def _estimated_context_tokens(case: SimpleNamespace) -> tuple[int, str]:
    """Return a conservative workload estimate without model-specific tokenizers."""

    output_tokens = int(case.max_output_tokens)
    if str(case.kind) == "quality":
        prompt_words = max(
            len(_quality_prompt(item, "context-estimate").split())
            for item in _QUALITY_ITEMS
        )
        return prompt_words + output_tokens + 128, "quality_prompt_words_plus_margin"
    if "audio" in case.requires:
        return output_tokens + 4096, "fixed_9.953313s_audio_plus_margin"
    if "ocr" in case.requires:
        patch_tokens = (
            (_OCR_IMAGE_WIDTH + 13) // 14
        ) * ((_OCR_IMAGE_HEIGHT + 13) // 14)
        return patch_tokens + output_tokens + 256, "ocr_image_patch14_plus_margin"
    if "vision" in case.requires:
        image_size = max(16, min(int(case.prompt_repetitions) or 64, 2048))
        patch_tokens = ((image_size + 13) // 14) ** 2
        return patch_tokens + output_tokens + 256, "clamped_vision_patch14_plus_margin"
    if "embeddings" in case.requires:
        return (
            max(int(case.prompt_repetitions), 1) + output_tokens + 32,
            "embedding_words_plus_margin",
        )
    if "rerank" in case.requires:
        query, candidates, _ = _rerank_inputs(case)
        longest_pair_words = len(query.split()) + max(
            len(candidate.split()) for candidate in candidates
        )
        return longest_pair_words + 128, "rerank_pair_words_plus_margin"
    prompt_words = len(_prompt(case, "context-estimate").split())
    request_overhead = 128
    estimate_basis = "prompt_words_plus_request_margin"
    if "tools" in case.requires:
        request_overhead += 256
        estimate_basis = "prompt_words_plus_tool_schema_margin"
    if "json" in case.requires:
        request_overhead += 64
        if estimate_basis == "prompt_words_plus_request_margin":
            estimate_basis = "prompt_words_plus_json_schema_margin"
    return (
        prompt_words + output_tokens + request_overhead,
        estimate_basis,
    )


def _request_arguments(
    *, server: Any, model: SimpleNamespace, case: SimpleNamespace, request_id: str
) -> dict[str, Any]:
    max_tokens = int(case.max_output_tokens)
    audio_case = str(case.kind) == "capability" and "audio" in case.requires
    if audio_case and server.backend != "sglang":
        raise RuntimeError(
            "The audio adapter requires SGLang with a registered speech LoRA"
        )
    request_body_json = getattr(model, "request_body_json", None)
    extra_body = json.loads(request_body_json) if request_body_json else {}
    if str(case.kind) == "capability" and "json" in case.requires:
        extra_body["response_format"] = {"type": "json_object"}
    if str(case.kind) == "capability" and "tools" in case.requires:
        extra_body.update(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "multiply",
                            "description": "Multiply two integers.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer"},
                                    "b": {"type": "integer"},
                                },
                                "required": ["a", "b"],
                            },
                        },
                    }
                ],
                "tool_choice": "required",
            }
        )
    if audio_case:
        if "messages" in extra_body or "lora_path" in extra_body:
            raise RuntimeError(
                "Audio cases reserve messages and lora_path request fields"
            )
    elif str(case.kind) == "capability" and "ocr" in case.requires:
        extra_body["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": _ocr_png_data_url()},
                    },
                ],
            }
        ]
    elif str(case.kind) == "capability" and "vision" in case.requires:
        extra_body["messages"] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the dominant color of this image? Reply with one color word.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _solid_red_png_data_url(
                                int(case.prompt_repetitions) or 64
                            )
                        },
                    },
                ],
            }
        ]
    arguments = {
        "base_url": server.base_url,
        "model": str(model.served_name),
        "prompt": (
            _AUDIO_PROMPT
            if audio_case
            else _OCR_PROMPT
            if "ocr" in case.requires
            else _prompt(case, request_id)
        ),
        "max_tokens": max_tokens,
        "temperature": float(case.temperature),
        "request_id": request_id,
        "extra_body": extra_body,
    }
    if audio_case:
        arguments.update(
            {
                "audio_path": _AUDIO_FIXTURE_PATH,
                "expected_audio_sha256": _AUDIO_FIXTURE_SHA256,
                "lora_path": _AUDIO_LORA_NAME,
            }
        )
    if server.backend == "ollama":
        arguments["context_size"] = int(model.max_context)
        arguments["require_native_decode_timing"] = not (
            str(case.kind) == "capability" and "vision" in case.requires
        )
    elif getattr(server, "authorization", None):
        arguments["authorization"] = str(server.authorization)
    return arguments


def _quality_request_arguments(
    *,
    server: Any,
    model: SimpleNamespace,
    case: SimpleNamespace,
    item: QualityItem,
    request_id: str,
) -> dict[str, Any]:
    arguments = _request_arguments(
        server=server,
        model=model,
        case=case,
        request_id=request_id,
    )
    arguments["prompt"] = _quality_prompt(item, request_id)
    return arguments


def _chat_request_function(server: Any, case: SimpleNamespace | None = None):
    if case is not None and "audio" in case.requires:
        if server.backend != "sglang":
            raise RuntimeError(
                "The audio adapter requires SGLang with a registered speech LoRA"
            )
        return stream_audio_chat_request
    return stream_ollama_chat_request if server.backend == "ollama" else stream_chat_request


def _authorization_argument(server: Any) -> dict[str, str]:
    authorization = getattr(server, "authorization", None)
    return {"authorization": str(authorization)} if authorization else {}


def _rerank_request_arguments(
    *, server: Any, model: SimpleNamespace, case: SimpleNamespace, request_id: str
) -> dict[str, Any]:
    if server.backend != "vllm":
        raise RuntimeError("The rerank adapter requires a vLLM /score endpoint")
    if _is_multimodal_rerank_case(case):
        query, candidates, _ = _multimodal_rerank_inputs(case)
        instruction = _MULTIMODAL_RERANK_INSTRUCTION
    else:
        query, candidates, _ = _rerank_inputs(case)
        instruction = None
    arguments = {
        "base_url": server.base_url,
        "model": str(model.served_name),
        "query": query,
        "candidates": candidates,
        "request_id": request_id,
    }
    if getattr(server, "authorization", None):
        arguments["authorization"] = str(server.authorization)
    if instruction is not None:
        arguments["instruction"] = instruction
    return arguments


def _multimodal_embedding_request_arguments(
    *, server: Any, model: SimpleNamespace, case: SimpleNamespace, request_id: str
) -> dict[str, Any]:
    if server.backend != "vllm":
        raise RuntimeError(
            "Multimodal embeddings require the vLLM Chat Embeddings endpoint"
        )
    image_size = int(case.prompt_repetitions) or 64
    arguments = {
        "base_url": server.base_url,
        "model": str(model.served_name),
        "image_data_url": _solid_red_png_data_url(image_size),
        "relevant_text": _MULTIMODAL_RELEVANT_TEXT,
        "unrelated_text": _MULTIMODAL_UNRELATED_TEXT,
        "request_id": request_id,
    }
    if getattr(server, "authorization", None):
        arguments["authorization"] = str(server.authorization)
    return arguments


def _validate_capability(
    case: SimpleNamespace, result: Any, model: Any | None = None
) -> dict[str, Any] | None:
    if str(case.kind) in {"decode", "concurrency"}:
        expected_tokens = int(case.max_output_tokens)
        actual_tokens = int(result.completion_tokens)
        diffusion_generation = model is not None and _is_diffusion_model(model)
        output_rate = getattr(result, "output_tps", None)
        rate_valid = (
            isinstance(output_rate, (int, float))
            and math.isfinite(float(output_rate))
            and float(output_rate) > 0
        )
        passed = (
            result.finish_reason == "length"
            and actual_tokens == expected_tokens
            and (rate_valid or not diffusion_generation)
        )
        if result.finish_reason != "length":
            reason = f"generation ended with {result.finish_reason!r}"
        elif actual_tokens != expected_tokens:
            reason = (
                f"generation reported {actual_tokens} completion tokens; "
                f"expected {expected_tokens}"
            )
        elif diffusion_generation and not rate_valid:
            reason = "end-to-end block-generation output rate was not positive and finite"
        else:
            reason = None
        validation = {
            "passed": passed,
            "reason": reason,
        }
        if diffusion_generation:
            validation.update(
                {
                    "generation_mode": "diffusion_block_generation",
                    "throughput_metric": (
                        "completion_tokens_per_end_to_end_request_elapsed"
                    ),
                }
            )
        return validation
    if str(case.kind) != "capability":
        return None
    if "json" in case.requires:
        try:
            payload = json.loads(result.content)
        except json.JSONDecodeError:
            return {"passed": False, "reason": "response was not valid JSON"}
        passed = payload.get("benchmark") == "spark" and payload.get("value") == 42
        return {"passed": passed, "reason": None if passed else "JSON values differed"}
    if "tools" in case.requires:
        for call in result.tool_calls:
            function = call.get("function") or {}
            if function.get("name") != "multiply":
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            passed = {arguments.get("a"), arguments.get("b")} == {6, 7}
            return {"passed": passed, "reason": None if passed else "tool arguments differed"}
        return {"passed": False, "reason": "multiply tool was not called"}
    if _is_multimodal_embedding_case(case):
        if not result.finite or int(result.dimension) <= 0:
            return {
                "passed": False,
                "reason": "multimodal embedding vectors were not finite and nonzero",
            }
        relevant = result.relevant_similarity
        unrelated = result.unrelated_similarity
        if not isinstance(relevant, (int, float)) or not isinstance(
            unrelated, (int, float)
        ):
            return {
                "passed": False,
                "reason": "multimodal embedding similarities were unavailable",
            }
        passed = math.isfinite(float(relevant)) and math.isfinite(
            float(unrelated)
        ) and float(relevant) > float(unrelated)
        return {
            "passed": passed,
            "reason": (
                None
                if passed
                else "red image did not rank the relevant red text higher"
            ),
        }
    if "audio" in case.requires:
        transcription = _normalize_audio_transcription(str(result.content))
        expected = _normalize_audio_transcription(_AUDIO_EXPECTED_TRANSCRIPTION)
        passed = transcription == expected
        return {
            "passed": passed,
            "reason": (
                None
                if passed
                else "normalized transcription did not exactly match the known fixture"
            ),
            "expected_transcription": expected,
            "normalized_transcription": transcription[:160],
        }
    if "ocr" in case.requires:
        transcription = _normalize_ocr_transcription(str(result.content))
        passed = transcription == _OCR_EXPECTED_TRANSCRIPTION
        return {
            "passed": passed,
            "reason": (
                None
                if passed
                else "normalized transcription did not exactly match the expected token"
            ),
            "expected_transcription": _OCR_EXPECTED_TRANSCRIPTION,
            "normalized_transcription": transcription[:160],
        }
    if "vision" in case.requires and "rerank" not in case.requires:
        passed = "red" in result.content.lower()
        return {"passed": passed, "reason": None if passed else "dominant color was not red"}
    if "rerank" in case.requires:
        if _is_multimodal_rerank_case(case):
            candidate_count = _MULTIMODAL_RERANK_CANDIDATE_COUNT
            relevant_index = _MULTIMODAL_RERANK_RELEVANT_INDEX
        else:
            _, candidates, relevant_index = _rerank_inputs(case)
            candidate_count = len(candidates)
        expected_indexes = list(range(candidate_count))
        if not result.finite:
            return {"passed": False, "reason": "rerank scores were not finite"}
        if int(result.candidate_count) != candidate_count:
            return {
                "passed": False,
                "reason": "rerank response returned the wrong candidate count",
            }
        scores = list(result.scores)
        ranking = list(result.ranking)
        if len(scores) != candidate_count:
            return {
                "passed": False,
                "reason": "rerank response returned the wrong score count",
            }
        if sorted(ranking) != expected_indexes:
            return {"passed": False, "reason": "rerank ranking was not a permutation"}
        if not ranking or int(ranking[0]) != relevant_index:
            return {
                "passed": False,
                "reason": f"rerank top candidate was not index {relevant_index}",
            }
        relevant_score = float(scores[relevant_index])
        passed = all(
            relevant_score > float(score)
            for index, score in enumerate(scores)
            if index != relevant_index
        )
        return {
            "passed": passed,
            "reason": None if passed else "rerank top score was not uniquely highest",
        }
    if "embeddings" in case.requires:
        passed = bool(result.finite and result.dimension > 0 and result.batch_size > 0)
        return {"passed": passed, "reason": None if passed else "embedding vector validation failed"}
    if str(case.id).startswith("long-context-needle"):
        passed = _needle(result.request_id) in result.content
        return {"passed": passed, "reason": None if passed else "needle was not returned"}
    return {"passed": False, "reason": "capability adapter is not implemented"}


def _run_warmups(server: Any, model: SimpleNamespace, case: SimpleNamespace) -> None:
    for index in range(int(case.warmups)):
        if str(case.kind) == "quality":
            item = _QUALITY_ITEMS[index % len(_QUALITY_ITEMS)]
            request_id = (
                f"warmup-{case.case_id}-{index}-{item.id}-{time.time_ns()}"
            )
            _chat_request_function(server, case)(
                **_quality_request_arguments(
                    server=server,
                    model=model,
                    case=case,
                    item=item,
                    request_id=request_id,
                )
            )
            continue
        request_id = f"warmup-{case.case_id}-{index}-{time.time_ns()}"
        if _is_multimodal_embedding_case(case):
            multimodal_embedding_request(
                **_multimodal_embedding_request_arguments(
                    server=server,
                    model=model,
                    case=case,
                    request_id=request_id,
                )
            )
        elif "rerank" in case.requires:
            score_request(
                **_rerank_request_arguments(
                    server=server,
                    model=model,
                    case=case,
                    request_id=request_id,
                )
            )
        elif "embeddings" in case.requires:
            text = "measurement " * max(int(case.prompt_repetitions), 1)
            embedding_request(
                base_url=server.base_url,
                model=str(model.served_name),
                inputs=[f"{text} batch item {item}" for item in range(int(case.concurrency))],
                request_id=request_id,
                **_authorization_argument(server),
            )
        else:
            _chat_request_function(server, case)(
                **_request_arguments(server=server, model=model, case=case, request_id=request_id)
            )


def _prime_model(server: Any, model: SimpleNamespace) -> Any:
    request_id = f"first-request-after-start-{time.time_ns()}"
    if "rerank" in model.tasks and "chat" not in model.tasks:
        prime_case = SimpleNamespace(prompt_repetitions=0)
        return score_request(
            **_rerank_request_arguments(
                server=server,
                model=model,
                case=prime_case,
                request_id=request_id,
            )
        )
    if {"embeddings", "vision"}.issubset(set(model.tasks)) and "chat" not in model.tasks:
        prime_case = SimpleNamespace(prompt_repetitions=64, requires=["embeddings", "vision"])
        return multimodal_embedding_request(
            **_multimodal_embedding_request_arguments(
                server=server,
                model=model,
                case=prime_case,
                request_id=request_id,
            )
        )
    if "embeddings" in model.tasks and "chat" not in model.tasks:
        return embedding_request(
            base_url=server.base_url,
            model=str(model.served_name),
            inputs=["Spark benchmark model-load probe."],
            request_id=request_id,
            **_authorization_argument(server),
        )
    prime_case = SimpleNamespace(
        id="first-request-after-start",
        case_id="first-request-after-start",
        kind="decode",
        requires=["chat"],
        prompt_repetitions=0,
        max_output_tokens=8,
        temperature=0.0,
    )
    return _chat_request_function(server)(
        **_request_arguments(
            server=server,
            model=model,
            case=prime_case,
            request_id=request_id,
        )
    )


def _execute_case(
    *,
    server: Any,
    model: SimpleNamespace,
    case: SimpleNamespace,
    journal: Journal,
    telemetry: TelemetrySampler,
) -> None:
    attempt_id = uuid.uuid4().hex
    journal.append(
        {
            "event": "case_start",
            "case_id": case.case_id,
            "attempt_id": attempt_id,
            "kind": case.kind,
            "concurrency": case.concurrency,
        }
    )
    telemetry.set_phase(f"warmup:{case.case_id}:{attempt_id}")
    started = time.perf_counter()
    try:
        _run_warmups(server, model, case)
        telemetry.set_phase(f"case:{case.case_id}:{attempt_id}")
        measured_started = time.perf_counter()
        validation_results: list[dict[str, Any]] = []
        for repetition in range(int(case.repetitions)):
            quality_items_by_request_id: dict[str, QualityItem] = {}
            quality_bursts_by_request_id: dict[str, float] = {}
            if str(case.kind) == "quality":
                results = []
                configured_concurrency = int(case.concurrency)
                for offset in range(0, len(_QUALITY_ITEMS), configured_concurrency):
                    batch_items = _QUALITY_ITEMS[
                        offset : offset + configured_concurrency
                    ]
                    requests = []
                    for item in batch_items:
                        request_id = (
                            f"{case.case_id}-r{repetition}-{item.id}-{time.time_ns()}"
                        )
                        quality_items_by_request_id[request_id] = item
                        requests.append(
                            _quality_request_arguments(
                                server=server,
                                model=model,
                                case=case,
                                item=item,
                                request_id=request_id,
                            )
                        )
                    batch_results, batch_s = concurrent_chat_requests(
                        requests=requests,
                        concurrency=len(requests),
                        request_function=_chat_request_function(server, case),
                    )
                    if len(batch_results) != len(requests):
                        raise RuntimeError(
                            "Quality batch returned the wrong number of responses"
                        )
                    for result in batch_results:
                        if result.request_id not in quality_items_by_request_id:
                            raise RuntimeError(
                                "Quality response request_id did not match its request"
                            )
                        quality_bursts_by_request_id[result.request_id] = batch_s
                    results.extend(batch_results)
                burst_s = 0.0
            elif _is_multimodal_embedding_case(case):
                requests = [
                    _multimodal_embedding_request_arguments(
                        server=server,
                        model=model,
                        case=case,
                        request_id=(
                            f"{case.case_id}-r{repetition}-w{worker}-{time.time_ns()}"
                        ),
                    )
                    for worker in range(int(case.concurrency))
                ]
                results, burst_s = concurrent_multimodal_embedding_requests(
                    requests=requests,
                    concurrency=int(case.concurrency),
                )
            elif "rerank" in case.requires:
                requests = [
                    _rerank_request_arguments(
                        server=server,
                        model=model,
                        case=case,
                        request_id=(
                            f"{case.case_id}-r{repetition}-w{worker}-{time.time_ns()}"
                        ),
                    )
                    for worker in range(int(case.concurrency))
                ]
                results, burst_s = concurrent_score_requests(
                    requests=requests,
                    concurrency=int(case.concurrency),
                )
            elif "embeddings" in case.requires:
                request_id = f"{case.case_id}-r{repetition}-w0-{time.time_ns()}"
                text = "measurement " * max(int(case.prompt_repetitions), 1)
                burst_started = time.perf_counter()
                results = [
                    embedding_request(
                        base_url=server.base_url,
                        model=str(model.served_name),
                        inputs=[
                            f"{text} batch item {item} nonce {request_id}"
                            for item in range(int(case.concurrency))
                        ],
                        request_id=request_id,
                        **_authorization_argument(server),
                    )
                ]
                burst_s = time.perf_counter() - burst_started
            else:
                requests = [
                    _request_arguments(
                        server=server,
                        model=model,
                        case=case,
                        request_id=f"{case.case_id}-r{repetition}-w{worker}-{time.time_ns()}",
                    )
                    for worker in range(int(case.concurrency))
                ]
                results, burst_s = concurrent_chat_requests(
                    requests=requests,
                    concurrency=int(case.concurrency),
                    request_function=_chat_request_function(server, case),
                )
            for result in results:
                if str(case.kind) == "quality":
                    validation = _validate_quality_item(
                        quality_items_by_request_id[result.request_id], result
                    )
                    result_burst_s = quality_bursts_by_request_id[result.request_id]
                else:
                    validation = _validate_capability(case, result, model=model)
                    result_burst_s = burst_s
                if validation is not None:
                    validation_results.append(validation)
                journal.append(
                    {
                        "event": "request_complete",
                        "case_id": case.case_id,
                        "attempt_id": attempt_id,
                        "kind": case.kind,
                        "repetition": repetition,
                        "burst_elapsed_s": result_burst_s,
                        "result": _request_result_payload(model, result),
                        "validation": validation,
                    }
                )
        elapsed_s = time.perf_counter() - measured_started
    except Exception as error:
        journal.append(
            {
                "event": "case_failed",
                "case_id": case.case_id,
                "attempt_id": attempt_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_s": time.perf_counter() - started,
            }
        )
        raise
    else:
        journal.append(
            {
                "event": "case_complete",
                "case_id": case.case_id,
                "attempt_id": attempt_id,
                "kind": case.kind,
                "concurrency": case.concurrency,
                "elapsed_s": elapsed_s,
                "validation_passed": (
                    all(item["passed"] for item in validation_results)
                    if validation_results
                    else None
                ),
            }
        )
    finally:
        telemetry.set_phase("between_cases")


def _recover_pending_lifecycle(
    *, model: SimpleNamespace, journal: Journal, run_dir: Path, workspace: Path
) -> bool:
    """Resolve an earlier crashed run's lifecycle before resume or finalization."""

    events = journal.events()
    last_start = max(
        (index for index, event in enumerate(events) if event.get("event") == "run_start"),
        default=-1,
    )
    if last_start < 0:
        return False
    run_finished = any(
        index > last_start and event.get("event") == "run_complete"
        for index, event in enumerate(events)
    )
    ready_indexes = [
        index
        for index, event in enumerate(events)
        if index > last_start and event.get("event") == "server_ready"
    ]
    if not ready_indexes:
        backend = str(model.backend)
        if backend == "llamacpp" and not run_finished:
            action = recover_owned_llamacpp(
                model,
                workspace=workspace,
                run_identity=str(model.run_identity),
                process_state_path=run_dir / "server" / "process.json",
            )
            journal.append(
                {
                    "event": "server_stopped",
                    "backend": backend,
                    "recovered": True,
                    "recovery_action": action,
                }
            )
            return True
        if backend in {"sglang", "vllm"} and not run_finished:
            if backend == "sglang":
                action = recover_owned_sglang(
                    str(model.run_identity),
                    api_key_path=run_dir / "server" / "api-key",
                )
            else:
                action = recover_owned_vllm(str(model.run_identity))
            journal.append(
                {
                    "event": "server_stopped",
                    "backend": backend,
                    "recovered": True,
                    "recovery_action": action,
                }
            )
            return True
        return False
    last_ready = ready_indexes[-1]
    if any(
        index > last_ready and event.get("event") in {"server_stopped", "server_kept"}
        for index, event in enumerate(events)
    ):
        return False

    ready_event = events[last_ready]
    backend = str(model.backend)
    if backend == "llamacpp":
        action = recover_owned_llamacpp(
            model,
            workspace=workspace,
            run_identity=str(model.run_identity),
            process_state_path=run_dir / "server" / "process.json",
        )
        journal.append(
            {
                "event": "server_stopped",
                "backend": backend,
                "recovered": True,
                "recovery_action": action,
            }
        )
        return True
    if backend in {"sglang", "vllm"}:
        if backend == "sglang":
            action = recover_owned_sglang(
                str(model.run_identity),
                api_key_path=run_dir / "server" / "api-key",
            )
        else:
            action = recover_owned_vllm(str(model.run_identity))
        journal.append(
            {
                "event": "server_stopped",
                "backend": backend,
                "recovered": True,
                "recovery_action": action,
            }
        )
        return True

    if backend != "ollama":
        return False

    cleanup_failed = any(
        index > last_ready and event.get("event") == "cleanup_failed"
        for index, event in enumerate(events)
    )
    unload_owned = ready_event.get("ollama_unload_owned")
    if run_finished and not cleanup_failed:
        # Older completed journals predate explicit server_stopped events. Do
        # not reinterpret a later user load as benchmark-owned.
        return False
    if unload_owned is False:
        return False

    try:
        loaded = ollama_model_loaded(str(model.endpoint), str(model.source))
    except Exception as error:
        journal.append(
            {
                "event": "recovery_cleanup_pending",
                "backend": backend,
                "model": str(model.source),
                "reason": f"loaded state could not be verified: {error}",
            }
        )
        raise
    if loaded:
        journal.append(
            {
                "event": "recovery_cleanup_pending",
                "backend": backend,
                "model": str(model.source),
                "reason": "model remains loaded and crash-time ownership cannot be proven",
            }
        )
        raise RuntimeErrorWithContext(
            f"Ollama model {model.source} may still be owned by the crashed run; "
            "refusing to finalize or unload it automatically"
        )
    journal.append(
        {
            "event": "server_stopped",
            "backend": backend,
            "recovered": True,
            "recovery_action": "observed_model_absent",
        }
    )
    return True


def _record_run_aborted(
    journal: Journal, error: BaseException, *, stage: str
) -> None:
    """Persist one terminal abort record for the current run attempt."""

    events = journal.events()
    last_start = max(
        (
            index
            for index, event in enumerate(events)
            if event.get("event") == "run_start"
        ),
        default=-1,
    )
    if last_start < 0:
        return
    already_recorded = any(
        index > last_start and event.get("event") == "run_aborted"
        for index, event in enumerate(events)
    )
    if already_recorded:
        return
    journal.append(
        {
            "event": "run_aborted",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )


def execute_plan(
    run_dir: Path,
    *,
    workspace: Path,
    allow_download: bool = False,
    keep_server: bool = False,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    plan_path = run_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    suite_without_case_ids = {
        **plan["suite"],
        "cases": [
            {key: value for key, value in case.items() if key != "case_id"}
            for case in plan["suite"]["cases"]
        ],
    }
    schema_version = int(plan.get("schema_version", 1))
    if schema_version not in {1, 2}:
        raise RuntimeError(f"Unsupported frozen plan schema version: {schema_version}")
    if schema_version >= 2:
        integrity_hash = plan.get("integrity_hash")
        integrity_payload = {
            key: value for key, value in plan.items() if key != "integrity_hash"
        }
        expected_fingerprint = content_hash(
            {
                "model": plan["model"],
                "suite": suite_without_case_ids,
                "resolved": plan.get("resolved", {}),
            }
        )
        integrity_length = len(str(integrity_hash)) if integrity_hash else 64
        integrity_valid = bool(integrity_hash) and (
            content_hash(integrity_payload, integrity_length) == integrity_hash
        )
    else:
        expected_fingerprint = content_hash(
            {"model": plan["model"], "suite": suite_without_case_ids}
        )
        expected_digest = plan["model"].get("image_digest")
        resolved_image = plan.get("resolved", {}).get("image_digest")
        integrity_valid = not expected_digest or (
            isinstance(resolved_image, str)
            and resolved_image.endswith("@" + expected_digest)
        )
    if not integrity_valid or expected_fingerprint != plan["fingerprint"]:
        raise RuntimeError("Frozen plan fingerprint does not match its contents")
    for case in plan["suite"]["cases"]:
        case_without_id = {key: value for key, value in case.items() if key != "case_id"}
        expected_case_id = _canonical_case(plan["model"], case_without_id)["case_id"]
        if case.get("case_id") != expected_case_id:
            raise RuntimeError("Frozen plan case identity does not match its contents")
    model = _namespace(plan["model"])
    model.resolved_image = plan.get("resolved", {}).get("image_digest")
    model.run_identity = f"{plan['fingerprint']}-{run_dir.name}"
    if str(getattr(model, "support_status", "")) == "incompatible":
        raise PreflightError(
            "This frozen model profile is marked incompatible and cannot be executed"
        )
    direct_commands = {
        "transformers": "diffusion-direct",
        "trtllm": "trtllm-direct",
    }
    direct_command = direct_commands.get(str(model.backend))
    if direct_command:
        raise PreflightError(
            f"{model.backend} direct profiles require the {direct_command} command"
        )
    cases = [_namespace(case) for case in plan["suite"]["cases"]]
    journal = Journal(run_dir / "events.jsonl")
    completed = journal.completed_cases()
    terminal = journal.terminal_cases()
    lock_path = results_lock_path(workspace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry = TelemetrySampler(run_dir / "telemetry.jsonl")
    server = None
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another SparkBench run holds the benchmark lock") from error
        try:
            lifecycle_changed = _recover_pending_lifecycle(
                model=model,
                journal=journal,
                run_dir=run_dir,
                workspace=workspace,
            )
        except BaseException as error:
            _record_run_aborted(journal, error, stage="lifecycle_recovery")
            try:
                summarize_run(run_dir)
            except Exception as summary_error:
                journal.append(
                    {
                        "event": "summary_failed",
                        "error_type": type(summary_error).__name__,
                        "error": str(summary_error),
                    }
                )
            raise
        if str(model.backend) == "llamacpp":
            try:
                require_llamacpp_mtp_evidence(
                    getattr(model, "args", ()), journal.events()
                )
                if llamacpp_dflash_requested(getattr(model, "args", ())):
                    require_llamacpp_dflash_evidence(
                        getattr(model, "args", ()), journal.events()
                    )
            except BaseException as error:
                stage = (
                    "dflash_evidence"
                    if llamacpp_dflash_requested(getattr(model, "args", ()))
                    else "mtp_evidence"
                )
                _record_run_aborted(journal, error, stage=stage)
                try:
                    summarize_run(run_dir)
                except Exception as summary_error:
                    journal.append(
                        {
                            "event": "summary_failed",
                            "error_type": type(summary_error).__name__,
                            "error": str(summary_error),
                        }
                    )
                raise
        if cases and all(case.case_id in terminal for case in cases):
            existing_summary = summarize_run(run_dir)
            if lifecycle_changed:
                existing_summary = summarize_run(run_dir)
            if existing_summary["status"] in {"aborted", "incomplete", "not_started"}:
                journal.append(
                    {
                        "event": "run_complete",
                        "status": "recovered_all_terminal" if completed else "no_work",
                    }
                )
                return summarize_run(run_dir)
            return existing_summary
        journal.append({"event": "run_start", "completed_cases_at_resume": sorted(completed)})
        if keep_server and str(model.backend) == "llamacpp":
            error = PreflightError(
                "--keep-server is not supported for managed llama.cpp processes"
            )
            _record_run_aborted(journal, error, stage="preflight")
            raise error
        runnable = []
        for case in cases:
            if case.case_id in terminal:
                continue
            missing = set(case.requires) - set(model.tasks)
            if missing:
                journal.append(
                    {
                        "event": "case_skipped_unsupported",
                        "case_id": case.case_id,
                        "missing_capabilities": sorted(missing),
                    }
                )
                continue
            unavailable_adapters: set[str] = set()
            adapter_reasons: list[str] = []
            if str(model.backend) != "vllm" and "rerank" in case.requires:
                unavailable_adapters.add("rerank")
                adapter_reasons.append("rerank requires the vLLM /score adapter")
            if str(model.backend) != "sglang" and "audio" in case.requires:
                unavailable_adapters.add("audio")
                adapter_reasons.append(
                    "audio requires SGLang with a registered speech LoRA"
                )
            if str(model.backend) != "vllm" and _is_multimodal_embedding_case(case):
                unavailable_adapters.update({"embeddings", "vision"})
                adapter_reasons.append(
                    "multimodal embeddings require the vLLM Chat Embeddings adapter"
                )
            if unavailable_adapters:
                journal.append(
                    {
                        "event": "case_skipped_adapter_unimplemented",
                        "case_id": case.case_id,
                        "capabilities": sorted(unavailable_adapters),
                        "backend": str(model.backend),
                        "reason": "; ".join(adapter_reasons),
                    }
                )
                continue
            approximate_tokens, estimate_basis = _estimated_context_tokens(case)
            if approximate_tokens > int(model.max_context):
                journal.append(
                    {
                        "event": "case_skipped_context_limit",
                        "case_id": case.case_id,
                        "approximate_required_tokens": approximate_tokens,
                        "estimate_basis": estimate_basis,
                        "model_max_context": model.max_context,
                    }
                )
                continue
            runnable.append(case)
        if not runnable:
            journal.append({"event": "run_complete", "status": "no_work"})
            return summarize_run(run_dir)

        failure_stage = "preflight"
        try:
            _preflight(model)
            failure_stage = "telemetry_start"
            telemetry.start()
            primary_error: BaseException | None = None
            try:
                validated_llamacpp_artifacts: dict[str, Any] | None = None
                artifact_validation_s: float | None = None
                if str(model.backend) == "llamacpp":
                    failure_stage = "artifact_validation"
                    telemetry.set_phase("artifact_validation")
                    validation_started = time.perf_counter()
                    validated_llamacpp_artifacts = validate_llamacpp_artifacts(
                        model, workspace=workspace
                    )
                    artifact_validation_s = time.perf_counter() - validation_started
                    artifact_event = {
                        "event": "artifact_validation_complete",
                        "backend": str(model.backend),
                        "elapsed_s": artifact_validation_s,
                        "runtime_binary_sha256": (
                            validated_llamacpp_artifacts[
                                "runtime_binary_sha256"
                            ]
                        ),
                        "model_sha256": validated_llamacpp_artifacts[
                            "model_sha256"
                        ],
                    }
                    if "mmproj_sha256" in validated_llamacpp_artifacts:
                        artifact_event["mmproj_sha256"] = (
                            validated_llamacpp_artifacts["mmproj_sha256"]
                        )
                    if "draft_model_sha256" in validated_llamacpp_artifacts:
                        artifact_event["draft_model_sha256"] = (
                            validated_llamacpp_artifacts["draft_model_sha256"]
                        )
                    journal.append(artifact_event)
                telemetry.set_phase("server_startup")
                failure_stage = "server_start"
                server = start_server(
                    model,
                    workspace=workspace,
                    allow_download=allow_download,
                    server_log_path=run_dir / "server" / "server.log",
                    process_state_path=run_dir / "server" / "process.json",
                    validated_llamacpp_artifacts=validated_llamacpp_artifacts,
                    artifact_validation_s=artifact_validation_s,
                )
                failure_stage = "server_provenance"
                provenance = capture_server_provenance(server)
                write_json(
                    run_dir / "server" / "provenance.json",
                    provenance,
                )
                if server.backend == "llamacpp":
                    Journal(run_dir / "server" / "provenance.jsonl").append(
                        {"event": "server_provenance", **provenance}
                    )
                journal.append(
                    {
                        "event": "server_ready",
                        "startup_s": server.startup_s,
                        "backend": server.backend,
                        "container_id": getattr(server, "container_id", None),
                        "process_pid": getattr(
                            getattr(server, "process", None), "pid", None
                        ),
                        "ollama_model": getattr(server, "ollama_model", None),
                        "ollama_unload_owned": bool(
                            getattr(server, "unload_ollama", False)
                        ),
                        "keep_server_requested": keep_server,
                    }
                )
                failure_stage = "first_request"
                telemetry.set_phase("first_request_after_start")
                first_request = _prime_model(server, model)
                journal.append(
                    {
                        "event": "first_request_complete",
                        "backend": server.backend,
                        "result": _request_result_payload(model, first_request),
                    }
                )
                failure_stage = "case_execution"
                for case in runnable:
                    try:
                        _execute_case(
                            server=server,
                            model=model,
                            case=case,
                            journal=journal,
                            telemetry=telemetry,
                        )
                    except Exception:
                        if not continue_on_error:
                            raise
            except BaseException as error:
                primary_error = error
                _record_run_aborted(journal, error, stage=failure_stage)
                raise
            finally:
                failure_stage = "server_cleanup"
                telemetry.set_phase("server_shutdown")
                try:
                    if server:
                        try:
                            # A vLLM metrics scrape is valid only for a server that this
                            # run actually started.  In particular, never probe a
                            # loopback endpoint supplied by an external or mocked
                            # server object: another benchmark may own that port.
                            if (
                                server.backend == "vllm"
                                and getattr(server, "container_id", None)
                                and getattr(server, "run_identity", None)
                                == str(model.run_identity)
                            ):
                                spec_decode_metrics = (
                                    snapshot_vllm_spec_decode_metrics(server.base_url)
                                )
                                if spec_decode_metrics is not None:
                                    journal.append(
                                        {
                                            "event": "vllm_spec_decode_metrics_snapshot",
                                            "backend": server.backend,
                                            "metrics": spec_decode_metrics,
                                        }
                                    )
                            if server.backend == "llamacpp":
                                spec_decode_metrics = (
                                    snapshot_llamacpp_spec_decode_metrics(
                                        server.base_url
                                    )
                                )
                                if spec_decode_metrics is not None:
                                    journal.append(
                                        {
                                            "event": (
                                                "llamacpp_spec_decode_metrics_snapshot"
                                            ),
                                            "backend": server.backend,
                                            "metrics": spec_decode_metrics,
                                        }
                                    )
                                if llamacpp_mtp_requested(
                                    getattr(model, "args", ())
                                ):
                                    require_mtp_activity(spec_decode_metrics)
                                if llamacpp_dflash_requested(
                                    getattr(model, "args", ())
                                ):
                                    require_speculative_activity(
                                        spec_decode_metrics, method="DFlash"
                                    )
                            save_server_logs(
                                server, run_dir / "server" / "server.log"
                            )
                        finally:
                            server.stop(keep_server=keep_server)
                        journal.append(
                            {
                                "event": (
                                    "server_kept" if keep_server else "server_stopped"
                                ),
                                "backend": server.backend,
                            }
                        )
                except Exception as cleanup_error:
                    journal.append(
                        {
                            "event": "cleanup_failed",
                            "error_type": type(cleanup_error).__name__,
                            "error": str(cleanup_error),
                        }
                    )
                    if primary_error is None:
                        raise
                finally:
                    telemetry.stop()
            failure_stage = "mtp_evidence"
            if str(model.backend) == "llamacpp":
                require_llamacpp_mtp_evidence(
                    getattr(model, "args", ()), journal.events()
                )
                if llamacpp_dflash_requested(getattr(model, "args", ())):
                    failure_stage = "dflash_evidence"
                    require_llamacpp_dflash_evidence(
                        getattr(model, "args", ()), journal.events()
                    )
            journal.append(
                {
                    "event": "run_complete",
                    "status": (
                        "completed_server_kept" if keep_server else "completed"
                    ),
                }
            )
        except BaseException as error:
            _record_run_aborted(journal, error, stage=failure_stage)
            try:
                summarize_run(run_dir)
            except Exception as summary_error:
                journal.append(
                    {
                        "event": "summary_failed",
                        "error_type": type(summary_error).__name__,
                        "error": str(summary_error),
                    }
                )
            raise
    return summarize_run(run_dir)


def results_lock_path(workspace: Path) -> Path:
    return workspace / "results" / ".sparkbench.lock"
