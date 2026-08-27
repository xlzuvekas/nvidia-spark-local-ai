from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from bench.qwen38_ple_cache import (
    PLECacheError,
    PLELayout,
    PLESourceFile,
    expected_marker_bytes,
    expected_marker_sha256,
    materialize_ple_cache,
    validate_ple_cache,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(document: object) -> bytes:
    return (
        json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _safetensors_bytes(tensors: list[tuple[str, bytes]]) -> bytes:
    offset = 0
    header: dict[str, object] = {}
    payload = bytearray()
    for name, data in tensors:
        header[name] = {
            "data_offsets": [offset, offset + len(data)],
            "dtype": "F8_E4M3",
            "shape": [2, 3],
        }
        payload.extend(data)
        offset += len(data)
    encoded_header = json.dumps(
        header, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return struct.pack("<Q", len(encoded_header)) + encoded_header + payload


class TinySnapshot:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.prefix = "model.layers.1.ple.ngram_embedding.shard_"
        self.shards = tuple(bytes(str(index) * 6, "ascii") for index in range(4))
        source_payloads = {
            "ple-a.safetensors": _safetensors_bytes(
                [
                    (f"{self.prefix}2.weight", self.shards[2]),
                    (f"{self.prefix}0.weight", self.shards[0]),
                ]
            ),
            "ple-b.safetensors": _safetensors_bytes(
                [
                    (f"{self.prefix}3.weight", self.shards[3]),
                    (f"{self.prefix}1.weight", self.shards[1]),
                ]
            ),
        }
        for name, data in source_payloads.items():
            (root / name).write_bytes(data)

        config = _json_bytes(
            {
                "text_config": {
                    "ple_embedding_dtype": "float8_e4m3fn",
                    "split_ngram_parts": 4,
                }
            }
        )
        index = _json_bytes(
            {
                "weight_map": {
                    f"{self.prefix}0.weight": "ple-a.safetensors",
                    f"{self.prefix}1.weight": "ple-b.safetensors",
                    f"{self.prefix}2.weight": "ple-a.safetensors",
                    f"{self.prefix}3.weight": "ple-b.safetensors",
                }
            }
        )
        (root / "config.json").write_bytes(config)
        (root / "model.safetensors.index.json").write_bytes(index)
        self.payload = b"".join(self.shards)
        self.layout = PLELayout(
            model_id="synthetic-ple",
            source="synthetic/model",
            revision="1" * 40,
            recipe_source="synthetic/recipe",
            recipe_revision="2" * 40,
            config_sha256=_sha256(config),
            index_sha256=_sha256(index),
            tensor_prefix=self.prefix,
            dtype="F8_E4M3",
            shard_count=4,
            rows_per_shard=2,
            embedding_dim=3,
            cache_file="ple_table_24_24.bin",
            marker_file="ple-cache-v1.json",
            payload_sha256=_sha256(self.payload),
            source_files=tuple(
                PLESourceFile(name, len(data), _sha256(data))
                for name, data in sorted(source_payloads.items())
            ),
        )


class Qwen38PLECacheTests(unittest.TestCase):
    def _tiny(self, root: Path) -> TinySnapshot:
        snapshot_root = root / "snapshot"
        snapshot_root.mkdir()
        return TinySnapshot(snapshot_root)

    def test_materializes_numeric_order_and_reuses_exact_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._tiny(root)
            cache = root / "cache"
            progress: list[str] = []
            first = materialize_ple_cache(
                snapshot=fixture.root,
                cache=cache,
                layout=fixture.layout,
                progress=progress.append,
            )
            self.assertEqual(
                (cache / fixture.layout.cache_file).read_bytes(),
                fixture.payload,
            )
            self.assertEqual(first.payload_sha256, _sha256(fixture.payload))
            self.assertEqual(
                first.marker_sha256,
                expected_marker_sha256(fixture.layout),
            )
            self.assertEqual(
                (cache / fixture.layout.marker_file).read_bytes(),
                expected_marker_bytes(fixture.layout),
            )
            self.assertEqual(os.stat(cache).st_mode & 0o777, 0o500)
            self.assertEqual(
                os.stat(cache / fixture.layout.cache_file).st_mode & 0o777,
                0o400,
            )
            self.assertEqual(
                os.stat(cache / fixture.layout.marker_file).st_mode & 0o777,
                0o400,
            )
            second = materialize_ple_cache(
                snapshot=fixture.root,
                cache=cache,
                layout=fixture.layout,
                progress=progress.append,
            )
            self.assertEqual(second, first)
            self.assertIn("verified existing completed PLE cache", progress)
            self.assertEqual(
                validate_ple_cache(
                    cache, layout=fixture.layout, verify_payload=True
                ),
                first,
            )

    def test_adopts_only_a_full_exact_unmarked_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._tiny(root)
            cache = root / "cache"
            cache.mkdir(mode=0o700)
            payload = cache / fixture.layout.cache_file
            payload.write_bytes(fixture.payload)

            materialize_ple_cache(
                snapshot=fixture.root,
                cache=cache,
                layout=fixture.layout,
            )

            self.assertEqual(payload.read_bytes(), fixture.payload)
            self.assertTrue((cache / fixture.layout.marker_file).is_file())
            self.assertEqual(os.stat(cache).st_mode & 0o777, 0o500)

    def test_refuses_wrong_unmarked_payload_without_overwriting_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._tiny(root)
            cache = root / "cache"
            cache.mkdir(mode=0o700)
            payload = cache / fixture.layout.cache_file
            wrong = b"x" * len(fixture.payload)
            payload.write_bytes(wrong)

            with self.assertRaisesRegex(PLECacheError, "payload digest mismatch"):
                materialize_ple_cache(
                    snapshot=fixture.root,
                    cache=cache,
                    layout=fixture.layout,
                )

            self.assertEqual(payload.read_bytes(), wrong)
            self.assertFalse((cache / fixture.layout.marker_file).exists())

    def test_stale_marker_and_writable_completion_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._tiny(root)
            cache = root / "cache"
            materialize_ple_cache(
                snapshot=fixture.root,
                cache=cache,
                layout=fixture.layout,
            )
            marker = cache / fixture.layout.marker_file
            cache.chmod(0o700)
            marker.chmod(0o600)
            marker.write_bytes(
                expected_marker_bytes(
                    replace(fixture.layout, revision="3" * 40)
                )
            )
            marker.chmod(0o400)
            cache.chmod(0o500)
            with self.assertRaisesRegex(PLECacheError, "marker is stale"):
                validate_ple_cache(cache, layout=fixture.layout)

            cache.chmod(0o700)
            with self.assertRaisesRegex(PLECacheError, "mode 0500"):
                validate_ple_cache(cache, layout=fixture.layout)

    def test_ownership_failure_is_explicit_and_never_auto_corrected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._tiny(root)
            cache = root / "cache"
            cache.mkdir(mode=0o700)
            payload = cache / fixture.layout.cache_file
            payload.write_bytes(fixture.payload)
            original_chmod = Path.chmod

            def guarded_chmod(path: Path, mode: int) -> None:
                if path == payload and mode == 0o400:
                    raise PermissionError("synthetic foreign owner")
                original_chmod(path, mode)

            with (
                patch.object(
                    Path,
                    "chmod",
                    autospec=True,
                    side_effect=guarded_chmod,
                ),
                self.assertRaisesRegex(
                    PLECacheError,
                    "change ownership of this one payload",
                ),
            ):
                materialize_ple_cache(
                    snapshot=fixture.root,
                    cache=cache,
                    layout=fixture.layout,
                )
            self.assertFalse((cache / fixture.layout.marker_file).exists())


if __name__ == "__main__":
    unittest.main()
