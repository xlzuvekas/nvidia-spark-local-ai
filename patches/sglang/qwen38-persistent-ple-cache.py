#!/usr/bin/env python3
"""Add fail-closed persistent read-only PLE reuse to the pinned mmap patch."""

from __future__ import annotations

import sys


HELPER_START = "_PLE_MMAP_DIR = None\n"
HELPER_END = "class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):"
LOADER_ANCHOR = """            emb = ple_mod.ngram_embedding
            if (
                loaded_weight.dtype == torch.float8_e4m3fn"""
LOADER_REPLACEMENT = """            emb = ple_mod.ngram_embedding
            if _ple_cache_is_readonly():
                # The offline materializer already concatenated and verified all
                # numeric shard_N weights. Keep loading the small weight_scale and
                # other PLE buffers through their normal paths.
                if ple_num_sync_shards != 128:
                    raise ValueError("persistent PLE cache requires 128 shards")
                if shard_idx < 0 or shard_idx >= 128:
                    raise ValueError(f"persistent PLE shard ID out of range: {shard_idx}")
                if shard_idx in ple_cache_seen_shards:
                    raise ValueError(f"duplicate persistent PLE shard ID: {shard_idx}")
                if loaded_weight.dtype != torch.float8_e4m3fn:
                    raise ValueError(
                        f"persistent PLE shard {shard_idx} has dtype "
                        f"{loaded_weight.dtype}, expected float8_e4m3fn"
                    )
                if tuple(loaded_weight.shape) != (2500012, 160):
                    raise ValueError(
                        f"persistent PLE shard {shard_idx} has shape "
                        f"{tuple(loaded_weight.shape)}, expected (2500012, 160)"
                    )
                ple_cache_seen_shards.add(shard_idx)
                loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
                return True
            if (
                loaded_weight.dtype == torch.float8_e4m3fn"""
LOADER_STATE_ANCHOR = """        loaded_buffers: Set[str] = set()
        loaded_shard_params: Set[str] = set()
        skipped_visual_count = 0"""
LOADER_STATE_REPLACEMENT = """        loaded_buffers: Set[str] = set()
        loaded_shard_params: Set[str] = set()
        ple_cache_seen_shards: Set[int] = set()
        skipped_visual_count = 0"""
LOADER_FINAL_ANCHOR = """        loaded_params.update(loaded_buffers)
        loaded_params.update(loaded_shard_params)

        if skipped_visual_count > 0:"""
LOADER_FINAL_REPLACEMENT = """        loaded_params.update(loaded_buffers)
        loaded_params.update(loaded_shard_params)

        if _ple_cache_is_readonly():
            expected_ple_shards = set(range(128))
            if ple_cache_seen_shards != expected_ple_shards:
                missing = sorted(expected_ple_shards - ple_cache_seen_shards)
                extra = sorted(ple_cache_seen_shards - expected_ple_shards)
                raise ValueError(
                    f"persistent PLE shard set mismatch: {missing=} {extra=}"
                )
            if len(ple_modules) != 1:
                raise ValueError(
                    f"persistent PLE cache requires one PLE module, got {len(ple_modules)}"
                )
            expected_ple_scales = {
                f"{prefix}.ngram_embedding.weight_scale" for prefix in ple_modules
            }
            if not expected_ple_scales.issubset(loaded_buffers):
                raise ValueError("persistent PLE cache weight_scale was not loaded")

        if skipped_visual_count > 0:"""

HELPER = '''_PLE_MMAP_DIR = None
_PLE_CACHE_MODE = None
_PLE_CACHE_VALIDATED = False


def _ple_cache_is_readonly():
    """Whether this process must reuse a completed persistent PLE cache."""
    import os

    global _PLE_CACHE_MODE
    if _PLE_CACHE_MODE is None:
        _PLE_CACHE_MODE = os.environ.get(
            "SGLANG_QWEN4_PLE_CACHE_MODE", ""
        ).strip()
        if _PLE_CACHE_MODE not in ("", "readonly"):
            raise RuntimeError(
                "unsupported SGLANG_QWEN4_PLE_CACHE_MODE: %r" % _PLE_CACHE_MODE
            )
        if _PLE_CACHE_MODE == "readonly" and os.environ.get(
            "SGLANG_QWEN4_PLE_CACHE_LOADER_CONTRACT", ""
        ) != "auto-mmap-no-prefetch":
            raise RuntimeError(
                "persistent PLE reuse requires the auto mmap/no-prefetch loader"
            )
    return _PLE_CACHE_MODE == "readonly"


def _validate_readonly_ple_cache(path, shape, dtype, nbytes):
    """Validate the host-admitted marker before mapping the immutable table."""
    import hashlib
    import json
    import os
    import stat

    global _PLE_CACHE_VALIDATED
    if _PLE_CACHE_VALIDATED:
        return
    if dtype != torch.float8_e4m3fn:
        raise RuntimeError("persistent PLE cache requires torch.float8_e4m3fn")
    if tuple(int(value) for value in shape) != (320001536, 160):
        raise RuntimeError("persistent PLE cache shape mismatch")
    if nbytes != 51200245760:
        raise RuntimeError("persistent PLE cache byte-size mismatch")
    if os.path.islink(path):
        raise RuntimeError("persistent PLE cache payload must not be a symlink")
    status = os.stat(path)
    if not stat.S_ISREG(status.st_mode) or status.st_size != nbytes:
        raise RuntimeError("persistent PLE cache payload is missing or partial")

    marker_path = os.path.join(os.path.dirname(path), "ple-cache-v1.json")
    if os.path.islink(marker_path):
        raise RuntimeError("persistent PLE cache marker must not be a symlink")
    with open(marker_path, "rb") as stream:
        marker_bytes = stream.read()
    expected_marker_sha256 = os.environ.get(
        "SGLANG_QWEN4_PLE_CACHE_MARKER_SHA256", ""
    ).strip()
    if len(expected_marker_sha256) != 64 or hashlib.sha256(
        marker_bytes
    ).hexdigest() != expected_marker_sha256:
        raise RuntimeError("persistent PLE cache marker digest mismatch")
    marker = json.loads(marker_bytes.decode("utf-8"))
    if marker.get("schema_version") != 1 or marker.get("kind") != (
        "sparkbench-qwen4-ple-cache"
    ):
        raise RuntimeError("persistent PLE cache marker schema mismatch")
    if marker.get("artifact") != {
        "source": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
        "revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
    }:
        raise RuntimeError("persistent PLE cache artifact mismatch")
    tensor = marker.get("tensor")
    if not isinstance(tensor, dict) or (
        tensor.get("file") != os.path.basename(path)
        or tensor.get("dtype") != "F8_E4M3"
        or tensor.get("shape") != [320001536, 160]
        or tensor.get("shard_count") != 128
        or tensor.get("size_bytes") != nbytes
        or tensor.get("sha256")
        != "sha256:b070f9644adf93794d8a1030584ab705809387e64396a9327a68fa3a3a6666b3"
    ):
        raise RuntimeError("persistent PLE cache tensor marker mismatch")
    _PLE_CACHE_VALIDATED = True


def _alloc_ple_table(shape, dtype):
    """Allocate a writable one-shot mmap or reuse the verified persistent one."""
    import logging
    import os

    global _PLE_MMAP_DIR
    if _PLE_MMAP_DIR is None:
        _PLE_MMAP_DIR = os.environ.get("SGLANG_QWEN4_PLE_MMAP_DIR", "").strip()
    if not _PLE_MMAP_DIR:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)

    numel = 1
    for d in shape:
        numel *= int(d)
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    path = os.path.join(_PLE_MMAP_DIR, "ple_table_%d_%d.bin" % (numel, nbytes))
    readonly = _ple_cache_is_readonly()
    if readonly:
        _validate_readonly_ple_cache(path, shape, dtype, nbytes)
    else:
        os.makedirs(_PLE_MMAP_DIR, exist_ok=True)
        if not os.path.exists(path) or os.path.getsize(path) != nbytes:
            with open(path, "wb") as f:
                f.truncate(nbytes)
    logging.getLogger(__name__).info(
        "PLE table -> mmap %s (%.1f GiB, dtype=%s, readonly=%s)",
        path,
        nbytes / 2**30,
        dtype,
        readonly,
    )
    storage = torch.from_file(
        path, shared=not readonly, size=nbytes, dtype=torch.uint8
    )

    # MADV_RANDOM: the table is purely random access (16 rows of 160 B per
    # token). Without it the kernel reads its whole readahead window around each
    # row. Measured cold: 1.4 MB of disk per token, ~560x more than is used.
    # Cheap in decode; a prefill chunk touches tens of thousands of rows.
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        MADV_RANDOM = 1
        rc = libc.madvise(
            ctypes.c_void_p(storage.data_ptr()),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(MADV_RANDOM),
        )
        logging.getLogger(__name__).info(
            "PLE table: madvise(MADV_RANDOM) %s", "ok" if rc == 0 else "failed"
        )
    except Exception as exc:  # not critical: affects I/O only, never correctness
        logging.getLogger(__name__).warning("PLE table: madvise not applied (%s)", exc)

    return storage.view(dtype).view(*shape)


'''


def transform(source: str) -> str:
    if "_validate_readonly_ple_cache" in source:
        return source
    if source.count(HELPER_START) != 1 or source.count(HELPER_END) != 1:
        raise ValueError("expected one public PLE mmap helper block")
    start = source.index(HELPER_START)
    end = source.index(HELPER_END, start)
    source = source[:start] + HELPER + source[end:]
    if source.count(LOADER_ANCHOR) != 1:
        raise ValueError("expected one Qwen4 PLE shard loader anchor")
    source = source.replace(LOADER_ANCHOR, LOADER_REPLACEMENT, 1)
    if source.count(LOADER_STATE_ANCHOR) != 1:
        raise ValueError("expected one Qwen4 loader-state anchor")
    source = source.replace(LOADER_STATE_ANCHOR, LOADER_STATE_REPLACEMENT, 1)
    if source.count(LOADER_FINAL_ANCHOR) != 1:
        raise ValueError("expected one Qwen4 loader-finalization anchor")
    return source.replace(LOADER_FINAL_ANCHOR, LOADER_FINAL_REPLACEMENT, 1)


def main(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as stream:
            source = stream.read()
        patched = transform(source)
    except (OSError, UnicodeError, ValueError) as error:
        print("ERROR:", error)
        return 1
    if patched == source:
        print("ALREADY PATCHED:", path)
        return 0
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(patched)
    print("PATCHED:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
