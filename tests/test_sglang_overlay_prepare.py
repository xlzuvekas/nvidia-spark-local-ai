from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from bench import sglang_overlay_prepare as prepare


PLE_SOURCE = r'''import torch


class nn:
    class Parameter:
        pass


class VocabParallelEmbedding:
    pass


class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):
    def load(self, source_weight):
        cpu_weight = nn.Parameter(
            torch.empty(
                source_weight.shape,
                dtype=source_weight.dtype,
                device="cpu",
                pin_memory=True,
            ),
            requires_grad=False,
        )
        return cpu_weight


class Qwen4Exp:
    def load_weights(self, weights):
        def load_qwen4_exp_ple_shard(name, loaded_weight):
            import re

            match = re.search(r"\.ngram_embedding\.shard_(\d+)\.weight$", name)
            shard_idx = int(match.group(1))
            mod_prefix = "model.layers.1.ple.ple_embedding"
            ple_mod = ple_modules.get(mod_prefix)
            emb = ple_mod.ngram_embedding
            if (
                loaded_weight.dtype == torch.float8_e4m3fn
                and emb.weight.dtype != torch.float8_e4m3fn
            ):
                pass
            return True

        ple_modules = {}
        ple_num_sync_shards = 128
        loaded_params: Set[str] = set()
        loaded_buffers: Set[str] = set()
        loaded_shard_params: Set[str] = set()
        skipped_visual_count = 0
        for name, loaded_weight in weights:
            if load_qwen4_exp_ple_shard(name, loaded_weight):
                continue
        loaded_params.update(loaded_buffers)
        loaded_params.update(loaded_shard_params)

        if skipped_visual_count > 0:
            pass
        return loaded_params
'''

QSA_SOURCE = '''def _resolve_trtllm_sparse_decode():
    from sglang.srt.utils import is_sm100_supported

    if not is_sm100_supported():
        return None
    return object()
'''


def _completed(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_specs() -> tuple[prepare.OverlaySpec, ...]:
    sources = {
        "qwen4_exp.py": PLE_SOURCE,
        "qwen_sparse_attn_backend.py": QSA_SOURCE,
    }
    specs: list[prepare.OverlaySpec] = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for spec in prepare.MODULE_OVERLAYS:
            source = root / spec.output_name
            source.write_text(sources[spec.output_name], encoding="utf-8")
            patcher_names = [spec.patcher_name]
            if spec.post_patcher_name is not None:
                patcher_names.append(spec.post_patcher_name)
            for patcher_name in patcher_names:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(prepare.PATCHER_ROOT / patcher_name),
                        str(source),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise AssertionError(result.stdout + result.stderr)
            specs.append(replace(spec, output_sha256=_sha256(source)))
    return tuple(specs)


class FakeDocker:
    def __init__(
        self,
        specs: tuple[prepare.OverlaySpec, ...],
        *,
        fail_cp: bool = False,
        discovered_paths: dict[str, str] | None = None,
    ) -> None:
        self.specs = specs
        self.fail_cp = fail_cp
        self.discovered_paths = discovered_paths or {
            spec.module: spec.container_path for spec in specs
        }
        self.calls: list[tuple[str, ...]] = []
        self.container_id = "a" * 64

    def __call__(
        self, arguments: list[str], *, timeout: int = 120
    ) -> subprocess.CompletedProcess[str]:
        del timeout
        args = tuple(arguments)
        self.calls.append(args)
        if args[:2] == ("image", "inspect"):
            return _completed(stdout=json.dumps([prepare.PINNED_IMAGE]) + "\n")
        if args[0] == "run":
            payload = json.dumps(self.discovered_paths, sort_keys=True)
            return _completed(
                stdout=f"import noise\n{prepare.MODULE_PATH_MARKER}{payload}\n"
            )
        if args[0] == "create":
            return _completed(stdout=self.container_id + "\n")
        if args[0] == "cp":
            if self.fail_cp:
                return _completed(returncode=1, stderr="synthetic cp failure")
            source = args[1].split(":", 1)[1]
            output = Path(args[2])
            if source.endswith("/qwen4_exp.py"):
                output.write_text(PLE_SOURCE, encoding="utf-8")
            elif source.endswith("/qwen_sparse_attn_backend.py"):
                output.write_text(QSA_SOURCE, encoding="utf-8")
            else:
                raise AssertionError(f"unexpected Docker cp source: {source}")
            return _completed()
        if args[:2] == ("container", "rm"):
            self.assert_own_container(args[2])
            return _completed(stdout=args[2] + "\n")
        raise AssertionError(f"unexpected Docker command: {args}")

    def assert_own_container(self, container_id: str) -> None:
        if container_id != self.container_id:
            raise AssertionError(f"unexpected container ID: {container_id}")


class SGLangOverlayPreparationTests(unittest.TestCase):
    def test_vendored_patchers_and_pinned_outputs_have_exact_digests(
        self,
    ) -> None:
        expected = {
            "bf2b7c75-ple_mmap.py": (
                "eeabdde061631c9b606d4ccc7371ff8f"
                "b01c6cc034dfe6bad1e4f29a8aa21555"
            ),
            "bf2b7c75-qsa_trtllm_sm120.py": (
                "f60ccb9f9e350a43155a1a7a20d154b"
                "e0b7e93c29dacb3db95d397ba910090b2"
            ),
            "qwen38-persistent-ple-cache.py": (
                "bf47f244406e149a3c7fe51d42d326d6"
                "3a008733d55868b51a73112052e3bcdf"
            ),
        }
        for spec in prepare.MODULE_OVERLAYS:
            patcher = prepare.PATCHER_ROOT / spec.patcher_name
            self.assertEqual(_sha256(patcher), expected[spec.patcher_name])
            self.assertEqual(spec.patcher_sha256, expected[spec.patcher_name])
            if spec.post_patcher_name is not None:
                post_patcher = prepare.PATCHER_ROOT / spec.post_patcher_name
                self.assertEqual(
                    _sha256(post_patcher), expected[spec.post_patcher_name]
                )
                self.assertEqual(
                    spec.post_patcher_sha256,
                    expected[spec.post_patcher_name],
                )
        self.assertEqual(
            {spec.output_name: spec.output_sha256 for spec in prepare.MODULE_OVERLAYS},
            {
                "qwen4_exp.py": (
                    "0b513b4dc4f2394f6b1733bb0b74fa40"
                    "ab59f4a04f6b33601350b2a606c67804"
                ),
                "qwen_sparse_attn_backend.py": (
                    "e30566492e1502f94a4c7fed42d90b5"
                    "23bbb662580c628459e6e63c7b5263c75"
                ),
            },
        )

    def test_prepares_with_offline_cpu_only_docker_and_is_idempotent(
        self,
    ) -> None:
        specs = _fixture_specs()
        fake = FakeDocker(specs)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with (
                patch.object(prepare, "MODULE_OVERLAYS", specs),
                patch.object(prepare, "_docker", side_effect=fake),
            ):
                target = prepare.prepare_overlays(workspace)
                self.assertEqual(
                    target,
                    workspace
                    / "results/runtime-overlays/"
                    "qwen38-flash-next-bf2b7c75-persistent-ple-v1",
                )
                self.assertEqual(
                    {entry.name for entry in target.iterdir()},
                    {spec.output_name for spec in specs},
                )
                for spec in specs:
                    self.assertEqual(
                        _sha256(target / spec.output_name), spec.output_sha256
                    )

                first_call_count = len(fake.calls)
                self.assertEqual(prepare.prepare_overlays(workspace), target)
                self.assertEqual(len(fake.calls), first_call_count + 1)
                self.assertEqual(fake.calls[-1][:2], ("image", "inspect"))

        flattened = [argument for call in fake.calls for argument in call]
        self.assertNotIn("pull", flattened)
        self.assertNotIn("--gpus", flattened)
        self.assertNotIn("stop", flattened)
        run = next(call for call in fake.calls if call[0] == "run")
        self.assertIn("--pull=never", run)
        self.assertIn("--network=none", run)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", run)
        self.assertIn(
            ("container", "rm", fake.container_id), fake.calls
        )

    def test_existing_mismatch_is_never_overwritten(self) -> None:
        specs = _fixture_specs()
        fake = FakeDocker(specs)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            target = (
                workspace
                / "results/runtime-overlays/"
                "qwen38-flash-next-bf2b7c75-persistent-ple-v1"
            )
            target.mkdir(parents=True)
            for spec in specs:
                (target / spec.output_name).write_text(
                    "mismatch\n", encoding="utf-8"
                )
            with (
                patch.object(prepare, "MODULE_OVERLAYS", specs),
                patch.object(prepare, "_docker", side_effect=fake),
                self.assertRaisesRegex(
                    prepare.OverlayPreparationError,
                    "refusing to overwrite",
                ),
            ):
                prepare.prepare_overlays(workspace)
            self.assertTrue(
                all(
                    (target / spec.output_name).read_text(encoding="utf-8")
                    == "mismatch\n"
                    for spec in specs
                )
            )
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(fake.calls[0][:2], ("image", "inspect"))

    def test_missing_image_fails_without_pull_or_container_mutation(self) -> None:
        calls: list[tuple[str, ...]] = []

        def missing_image(
            arguments: list[str], *, timeout: int = 120
        ) -> subprocess.CompletedProcess[str]:
            del timeout
            calls.append(tuple(arguments))
            return _completed(returncode=1)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(prepare, "_docker", side_effect=missing_image),
            self.assertRaises(prepare.OverlayPreparationError),
        ):
            prepare.prepare_overlays(Path(directory))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:2], ("image", "inspect"))
        self.assertNotIn("pull", calls[0])

    def test_discovery_path_mismatch_fails_before_extraction(self) -> None:
        specs = _fixture_specs()
        paths = {spec.module: spec.container_path for spec in specs}
        paths[specs[0].module] = "/tmp/qwen4_exp.py"
        fake = FakeDocker(specs, discovered_paths=paths)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(prepare, "MODULE_OVERLAYS", specs),
            patch.object(prepare, "_docker", side_effect=fake),
            self.assertRaisesRegex(
                prepare.OverlayPreparationError, "module path mismatch"
            ),
        ):
            prepare.prepare_overlays(Path(directory))
        self.assertFalse(any(call[0] == "create" for call in fake.calls))

    def test_extract_failure_removes_only_its_inert_container(self) -> None:
        specs = _fixture_specs()
        fake = FakeDocker(specs, fail_cp=True)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(prepare, "MODULE_OVERLAYS", specs),
            patch.object(prepare, "_docker", side_effect=fake),
            self.assertRaises(prepare.OverlayPreparationError),
        ):
            prepare.prepare_overlays(Path(directory))
        self.assertIn(
            ("container", "rm", fake.container_id), fake.calls
        )
        self.assertFalse(any("stop" in call for call in fake.calls))
        self.assertFalse(any("--force" in call or "-f" in call for call in fake.calls))


if __name__ == "__main__":
    unittest.main()
