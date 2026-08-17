from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bench.llamacpp_metrics import (
    aggregate_llamacpp_spec_decode_metrics,
    assess_llamacpp_mtp_evidence,
    assess_llamacpp_mtp_proposal_depth,
    llamacpp_dflash_requested,
    llamacpp_mtp_depth,
    llamacpp_mtp_requested,
    parse_llamacpp_spec_decode_metrics,
    require_llamacpp_dflash_evidence,
    require_llamacpp_mtp_evidence,
    require_mtp_activity,
)
from bench.journal import Journal, content_hash
from bench.inventory import (
    HuggingFaceSnapshot,
    Inventory,
    assess_model_availability,
)
from bench.manifest import ManifestError, load_models, load_suite, validate_model
from bench.report import summarize_run
from bench.runner import _estimated_context_tokens, _request_arguments, execute_plan
from bench.runtime import (
    ManagedServer,
    RuntimeErrorWithContext,
    _llamacpp_alias_ready,
    _native_command,
    recover_owned_llamacpp,
    start_llamacpp,
    validate_llamacpp_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
EXPOSITION = """
# TYPE llamacpp:spec_decode_num_draft_tokens_total counter
llamacpp:spec_decode_num_draft_tokens_total 317
# TYPE llamacpp:spec_decode_num_accepted_tokens_total counter
llamacpp:spec_decode_num_accepted_tokens_total 148
# TYPE llamacpp:spec_decode_num_drafts_total counter
llamacpp:spec_decode_num_drafts_total 106
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 75
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 46
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="2"} 27
"""


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _completed(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class LlamaCppManifestTests(unittest.TestCase):
    def test_muse_glimmer_profiles_are_exact_matched_dflash_pair(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        baseline = models["muse-glimmer-30b-ud-q4-k-xl-llamacpp"]
        dflash = models[
            "muse-glimmer-30b-ud-q4-k-xl-llamacpp-dflash15"
        ]

        for model in (baseline, dflash):
            with self.subTest(model=model.id):
                self.assertEqual(model.backend, "llamacpp")
                self.assertEqual(model.lifecycle, "subprocess")
                self.assertEqual(
                    model.source, "unsloth/Muse-Glimmer-30B-GGUF"
                )
                self.assertEqual(
                    model.revision,
                    "faa5b025c584459c13febfa5c59883516710ae39",
                )
                self.assertEqual(
                    model.model_file,
                    "Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
                )
                self.assertEqual(model.model_size_bytes, 15_878_222_368)
                self.assertIsNone(model.weight_size_bytes)
                self.assertEqual(
                    model.model_digest,
                    "sha256:82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38",
                )
                self.assertEqual(model.runtime_parallel, 4)
                self.assertEqual(model.max_context, 32_768)
                self.assertEqual(model.native_context, 131_072)
                self.assertEqual(model.tasks, ("chat", "json", "tools"))
                self.assertNotIn("vision", model.tasks)
                self.assertNotIn("--spec-draft-model", model.args)

        self.assertFalse(llamacpp_dflash_requested(baseline.args))
        self.assertTrue(llamacpp_dflash_requested(dflash.args))
        self.assertEqual(
            dflash.draft_source, "meta-models/Muse-Glimmer-30B-GGUF"
        )
        self.assertEqual(
            dflash.draft_revision,
            "43c7eadd41352a299ea8e0a36b3157978dd63596",
        )
        self.assertEqual(
            dflash.draft_model_file,
            "dflash-Muse-Glimmer-30B-Q4_K_M.gguf",
        )
        self.assertEqual(dflash.draft_model_size_bytes, 1_631_208_128)
        self.assertIsNone(dflash.draft_weight_size_bytes)
        self.assertEqual(
            dflash.draft_model_digest,
            "sha256:b2e808bf656086fe86bd0d0bd990f01d33e377537a07c02d45371517c8b264ef",
        )
        self.assertEqual(
            dflash.args[len(baseline.args) :],
            (
                "--spec-type",
                "draft-dflash",
                "--spec-draft-n-max",
                "15",
            ),
        )
        self.assertEqual(
            replace(
                dflash,
                id=baseline.id,
                description=baseline.description,
                architecture=baseline.architecture,
                quantization=baseline.quantization,
                estimated_ram_gib=baseline.estimated_ram_gib,
                args=baseline.args,
                draft_source=None,
                draft_revision=None,
                draft_weight_size_bytes=None,
                draft_model_file=None,
                draft_model_digest=None,
                draft_model_size_bytes=None,
            ),
            baseline,
        )

    def test_llamacpp_dflash_contract_is_complete_and_runtime_owned(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        baseline = models["muse-glimmer-30b-ud-q4-k-xl-llamacpp"]
        dflash = models[
            "muse-glimmer-30b-ud-q4-k-xl-llamacpp-dflash15"
        ]
        invalid = (
            replace(dflash, draft_model_digest=None),
            replace(dflash, draft_model_file="../draft.gguf"),
            replace(dflash, draft_model_size_bytes=None),
            replace(dflash, args=baseline.args),
            replace(
                dflash,
                args=(*baseline.args, "--spec-type", "draft-dflash"),
            ),
        )
        for profile in invalid:
            with self.subTest(profile=profile):
                with self.assertRaises(ManifestError):
                    validate_model(profile)
        with self.assertRaisesRegex(ManifestError, "runtime-owned"):
            validate_model(
                replace(
                    dflash,
                    args=(*dflash.args, "--spec-draft-model", "/tmp/draft.gguf"),
                )
            )

    def test_dflash_native_command_injects_only_the_verified_sidecar(self) -> None:
        model = load_models(ROOT / "manifests" / "models.toml")[
            "muse-glimmer-30b-ud-q4-k-xl-llamacpp-dflash15"
        ]
        command = _native_command(
            model,
            {
                "runtime_binary": "/runtime/llama-server",
                "model_path": "/verified/target.gguf",
                "draft_model_path": "/verified/draft.gguf",
            },
            port=8000,
        )
        self.assertEqual(command.count("--spec-draft-model"), 1)
        self.assertEqual(
            command[command.index("--spec-draft-model") + 1],
            "/verified/draft.gguf",
        )
        self.assertLess(
            command.index("--spec-draft-model"), command.index("--alias")
        )
        self.assertNotIn(str(model.draft_source), command)

    def test_dflash_inventory_requires_both_exact_snapshots(self) -> None:
        profile = load_models(ROOT / "manifests" / "models.toml")[
            "muse-glimmer-30b-ud-q4-k-xl-llamacpp-dflash15"
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            draft = root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / str(profile.model_file)).write_bytes(b"target")
            runtime = root / "llama-server"
            runtime.write_bytes(b"runtime")
            profile = replace(
                profile, cache_dir="project", runtime_binary=str(runtime)
            )

            def inventory(*snapshots: HuggingFaceSnapshot) -> Inventory:
                return Inventory(
                    collected_at="now",
                    python_version="3",
                    platform="test",
                    machine="aarch64",
                    huggingface_snapshots=snapshots,
                    docker_images=(),
                    ollama_models=(),
                )

            target_snapshot = HuggingFaceSnapshot(
                source=profile.source,
                revision=str(profile.revision),
                path=target,
            )
            draft_snapshot = HuggingFaceSnapshot(
                source=str(profile.draft_source),
                revision=str(profile.draft_revision),
                path=draft,
            )
            availability = assess_model_availability(
                {profile.id: profile}, inventory(target_snapshot)
            )[profile.id]
            self.assertFalse(availability.source_available)
            self.assertIn(
                "exact draft GGUF file is not cached", availability.details
            )

            (draft / str(profile.draft_model_file)).write_bytes(b"draft")
            availability = assess_model_availability(
                {profile.id: profile},
                inventory(target_snapshot, draft_snapshot),
            )[profile.id]
            self.assertTrue(availability.available)

    def test_qwen36_profiles_are_exact_single_slot_mtp_pair(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        baseline = models["qwen36-35b-a3b-ud-q4-k-xl-llamacpp"]
        mtp2 = models["qwen36-35b-a3b-ud-q4-k-xl-llamacpp-mtp2"]

        for model in (baseline, mtp2):
            with self.subTest(model=model.id):
                self.assertEqual(model.backend, "llamacpp")
                self.assertEqual(model.lifecycle, "subprocess")
                self.assertEqual(
                    model.source, "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
                )
                self.assertEqual(
                    model.revision,
                    "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
                )
                self.assertEqual(
                    model.model_file,
                    "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
                )
                self.assertEqual(
                    model.fetch_allow_patterns, (model.model_file,)
                )
                self.assertEqual(
                    model.model_digest,
                    "sha256:55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1",
                )
                self.assertEqual(model.model_size_bytes, 22_853_663_008)
                self.assertEqual(
                    model.runtime_digest,
                    "sha256:ae1bd49f869ff3397b2a5d757fcf010c6eaaf16c4e3071a15861312defcd4e40",
                )
                self.assertEqual(
                    model.runtime_revision,
                    "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70",
                )
                self.assertEqual(model.tasks, ("chat", "json", "tools"))
                self.assertEqual(model.architecture, "qwen35moe")
                self.assertEqual(model.quantization, "ud-q4_k_xl")
                self.assertEqual(model.runtime_parallel, 1)
                self.assertEqual(model.max_context, 262_144)
                self.assertEqual(model.native_context, 262_144)
                self.assertEqual(model.estimated_ram_gib, 100.0)
                self.assertIsNone(model.mmproj_file)
                self.assertNotIn("--parallel", model.args)

        self.assertFalse(llamacpp_mtp_requested(baseline.args))
        self.assertTrue(llamacpp_mtp_requested(mtp2.args))
        self.assertEqual(llamacpp_mtp_depth(mtp2.args), 2)
        self.assertEqual(mtp2.args[: len(baseline.args)], baseline.args)
        self.assertEqual(
            mtp2.args[len(baseline.args) :],
            (
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                "2",
                "--spec-draft-type-k",
                "q8_0",
                "--spec-draft-type-v",
                "q8_0",
                "--spec-draft-backend-sampling",
            ),
        )
        self.assertEqual(
            replace(
                mtp2,
                id=baseline.id,
                description=baseline.description,
                args=baseline.args,
            ),
            baseline,
            "Qwen3.6 matched profiles drifted outside identity text and MTP args",
        )

    def test_repository_profiles_pin_runtime_and_each_gguf(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        expected = {
            "qwen38-27b-ud-q4-k-xl-llamacpp": (
                "Qwen3.8-27B-UD-Q4_K_XL.gguf",
                "sha256:bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372",
                17_923_394_624,
                False,
            ),
            "qwen38-27b-ud-q4-k-xl-llamacpp-mtp3": (
                "Qwen3.8-27B-UD-Q4_K_XL.gguf",
                "sha256:bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372",
                17_923_394_624,
                True,
            ),
            "qwen38-27b-ud-q5-k-xl-llamacpp": (
                "Qwen3.8-27B-UD-Q5_K_XL.gguf",
                "sha256:176a6a3f034e9cdc447c10cd00329fc9b31002e6589b9295f2ad4f1eefe0f6ab",
                20_218_178_624,
                False,
            ),
            "qwen38-27b-q8-0-llamacpp": (
                "Qwen3.8-27B-Q8_0.gguf",
                "sha256:a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348",
                29_047_086_048,
                False,
            ),
            "qwen38-27b-ud-iq2-xxs-llamacpp": (
                "Qwen3.8-27B-UD-IQ2_XXS.gguf",
                "sha256:8d1b37297d6cf98303cd396896f35e01089ddcc904053a9c6997f7a1c35b8524",
                9_010_048_064,
                False,
            ),
        }
        for model_id, (filename, digest, size, mtp) in expected.items():
            with self.subTest(model=model_id):
                model = models[model_id]
                self.assertEqual(model.backend, "llamacpp")
                self.assertEqual(model.lifecycle, "subprocess")
                self.assertEqual(model.model_file, filename)
                self.assertEqual(model.fetch_allow_patterns, (filename,))
                self.assertEqual(model.model_digest, digest)
                self.assertEqual(model.model_size_bytes, size)
                self.assertEqual(
                    model.runtime_digest,
                    "sha256:ae1bd49f869ff3397b2a5d757fcf010c6eaaf16c4e3071a15861312defcd4e40",
                )
                self.assertEqual(model.max_context, 32768)
                self.assertEqual(model.native_context, 262144)
                self.assertEqual(model.runtime_parallel, 8)
                self.assertNotIn("--parallel", model.args)
                self.assertIn("--reasoning", model.args)
                self.assertEqual("draft-mtp" in model.args, mtp)

    def test_q5_profile_is_a_matched_non_mtp_anchor_between_q4_and_q8(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        q4 = models["qwen38-27b-ud-q4-k-xl-llamacpp"]
        q5 = models["qwen38-27b-ud-q5-k-xl-llamacpp"]
        q8 = models["qwen38-27b-q8-0-llamacpp"]

        self.assertEqual(q5.tasks, ("chat", "json", "tools"))
        self.assertEqual(q5.quantization, "ud-q5_k_xl")
        self.assertFalse(llamacpp_mtp_requested(q5.args))
        self.assertIsNone(q5.mmproj_file)
        self.assertEqual(q5.estimated_ram_gib, 72.0)
        self.assertLess(q4.estimated_ram_gib, q5.estimated_ram_gib)
        self.assertLess(q5.estimated_ram_gib, q8.estimated_ram_gib)
        self.assertEqual(
            replace(
                q5,
                id=q4.id,
                description=q4.description,
                quantization=q4.quantization,
                fetch_allow_patterns=q4.fetch_allow_patterns,
                model_file=q4.model_file,
                model_digest=q4.model_digest,
                model_size_bytes=q4.model_size_bytes,
                estimated_ram_gib=q4.estimated_ram_gib,
            ),
            q4,
            "Q5 profile drifted outside its identity, artifact, quantization, "
            "and RAM estimate",
        )

    def test_mtp_depth_sweep_profiles_are_matched_and_nonmonotonic(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        stem = "qwen38-27b-ud-q4-k-xl-llamacpp-mtp"
        depths = (3, 6, 1, 5, 2, 4)
        expected_ids = tuple(f"{stem}{depth}" for depth in depths)
        sweep_ids = tuple(model_id for model_id in models if model_id in expected_ids)
        self.assertEqual(sweep_ids, expected_ids)

        normalized = []
        for depth, model_id in zip(depths, expected_ids, strict=True):
            with self.subTest(model=model_id):
                model = models[model_id]
                self.assertTrue(llamacpp_mtp_requested(model.args))
                self.assertEqual(llamacpp_mtp_depth(model.args), depth)
                self.assertEqual(model.args.count("--spec-type"), 1)
                self.assertEqual(model.args.count("--spec-draft-n-max"), 1)
                self.assertEqual(model.architecture, f"dense+mtp{depth}")
                self.assertEqual(model.quantization, f"ud-q4_k_xl+mtp{depth}")
                self.assertIsNone(model.mmproj_file)

                arguments = list(model.args)
                depth_index = arguments.index("--spec-draft-n-max") + 1
                arguments[depth_index] = "<depth>"
                normalized.append(
                    replace(
                        model,
                        id="<id>",
                        description="<description>",
                        architecture="<architecture>",
                        quantization="<quantization>",
                        args=tuple(arguments),
                    )
                )

        self.assertTrue(
            all(profile == normalized[0] for profile in normalized[1:]),
            "MTP sweep profiles drifted outside their depth-specific labels/argument",
        )

    def test_mtp_depth_suite_exactly_matches_core_decode_256(self) -> None:
        sweep = load_suite(
            ROOT / "manifests" / "suites" / "llamacpp_mtp_depth.toml"
        )
        core = load_suite(ROOT / "manifests" / "suites" / "core.toml")
        core_decode = next(case for case in core.cases if case.id == "decode-256")
        self.assertEqual(sweep.id, "llamacpp-mtp-depth")
        self.assertEqual(sweep.cases, (core_decode,))

    def test_long_context_profiles_change_only_geometry_and_mtp_fields(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        profile_pairs = (
            (
                models["qwen38-27b-ud-q4-k-xl-llamacpp"],
                models["qwen38-27b-ud-q4-k-xl-llamacpp-long-context"],
                96.0,
            ),
            (
                models["qwen38-27b-ud-q4-k-xl-llamacpp-mtp5"],
                models[
                    "qwen38-27b-ud-q4-k-xl-llamacpp-mtp5-long-context"
                ],
                100.0,
            ),
        )
        for current, long_context, estimated_ram_gib in profile_pairs:
            with self.subTest(model=long_context.id):
                self.assertEqual(long_context.runtime_parallel, 1)
                self.assertEqual(long_context.max_context, 262_144)
                self.assertEqual(long_context.native_context, 262_144)
                self.assertEqual(
                    long_context.max_context * long_context.runtime_parallel,
                    long_context.native_context,
                )
                self.assertEqual(
                    long_context.estimated_ram_gib, estimated_ram_gib
                )
                self.assertEqual(
                    replace(
                        long_context,
                        id=current.id,
                        description=current.description,
                        runtime_parallel=current.runtime_parallel,
                        max_context=current.max_context,
                        estimated_ram_gib=current.estimated_ram_gib,
                    ),
                    current,
                    "long-context profile drifted outside identity, slot geometry, "
                    "and conservative RAM estimate",
                )

        baseline = profile_pairs[0][1]
        mtp5 = profile_pairs[1][1]
        self.assertEqual(mtp5.args[: len(baseline.args)], baseline.args)
        self.assertEqual(
            mtp5.args[len(baseline.args) :],
            (
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                "5",
                "--spec-draft-type-k",
                "q8_0",
                "--spec-draft-type-v",
                "q8_0",
                "--spec-draft-backend-sampling",
            ),
        )
        self.assertEqual(
            replace(
                mtp5,
                id=baseline.id,
                description=baseline.description,
                architecture=baseline.architecture,
                quantization=baseline.quantization,
                estimated_ram_gib=baseline.estimated_ram_gib,
                args=baseline.args,
            ),
            baseline,
            "matched long-context profiles drifted outside MTP labels/arguments",
        )

    def test_long_context_suite_is_exact_key_single_slot_and_fits(self) -> None:
        suite = load_suite(
            ROOT / "manifests" / "suites" / "llamacpp_long_context.toml"
        )
        self.assertEqual(suite.id, "llamacpp-long-context")
        expected = (
            (32_768, 3),
            (65_536, 3),
            (131_072, 3),
            (245_760, 1),
        )
        self.assertEqual(
            tuple(
                (case.prompt_repetitions, case.repetitions)
                for case in suite.cases
            ),
            expected,
        )

        models = load_models(ROOT / "manifests" / "models.toml")
        long_context_profiles = (
            models["qwen38-27b-ud-q4-k-xl-llamacpp-long-context"],
            models["qwen38-27b-ud-q4-k-xl-llamacpp-mtp5-long-context"],
        )
        for case in suite.cases:
            with self.subTest(case=case.id):
                self.assertEqual(
                    case.id,
                    f"long-context-needle-{case.prompt_repetitions}",
                )
                self.assertEqual(case.kind, "capability")
                self.assertEqual(case.requires, ("chat",))
                self.assertEqual(case.warmups, 0)
                self.assertEqual(case.max_output_tokens, 32)
                self.assertEqual(case.temperature, 0.0)
                self.assertEqual(case.concurrency, 1)
                estimated_tokens, basis = _estimated_context_tokens(case)
                self.assertEqual(basis, "prompt_words_plus_request_margin")
                self.assertGreater(estimated_tokens, case.prompt_repetitions)
                for profile in long_context_profiles:
                    self.assertLessEqual(estimated_tokens, profile.max_context)

    def test_long_context_runtime_command_changes_only_parallel_slots(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        artifacts = {
            "runtime_binary": "/runtime/llama-server",
            "model_path": "/models/qwen38-q4.gguf",
        }
        profile_pairs = (
            (
                models["qwen38-27b-ud-q4-k-xl-llamacpp"],
                models["qwen38-27b-ud-q4-k-xl-llamacpp-long-context"],
            ),
            (
                models["qwen38-27b-ud-q4-k-xl-llamacpp-mtp5"],
                models[
                    "qwen38-27b-ud-q4-k-xl-llamacpp-mtp5-long-context"
                ],
            ),
        )
        for current, long_context in profile_pairs:
            with self.subTest(model=long_context.id):
                current_command = _native_command(
                    current, artifacts, port=8000
                )
                long_command = _native_command(
                    long_context, artifacts, port=8000
                )
                current_parallel = current_command.index("--parallel") + 1
                long_parallel = long_command.index("--parallel") + 1
                current_context = current_command.index("--ctx-size") + 1
                long_context_size = long_command.index("--ctx-size") + 1
                self.assertEqual(current_command[current_context], "262144")
                self.assertEqual(long_command[long_context_size], "262144")
                self.assertEqual(current_command[current_parallel], "8")
                self.assertEqual(long_command[long_parallel], "1")

                normalized_current = list(current_command)
                normalized_long = list(long_command)
                normalized_current[current_parallel] = "<parallel>"
                normalized_long[long_parallel] = "<parallel>"
                self.assertEqual(
                    normalized_long,
                    normalized_current,
                    "native launch command drifted outside the parallel-slot count",
                )

    def test_vision_profile_pins_exact_projector_and_capability(self) -> None:
        model = load_models(ROOT / "manifests" / "models.toml")[
            "qwen38-27b-ud-q4-k-xl-llamacpp-vision"
        ]
        self.assertEqual(model.tasks, ("chat", "json", "vision", "tools"))
        self.assertEqual(
            model.fetch_allow_patterns,
            ("Qwen3.8-27B-UD-Q4_K_XL.gguf", "mmproj-F16.gguf"),
        )
        self.assertEqual(model.mmproj_file, "mmproj-F16.gguf")
        self.assertEqual(model.mmproj_size_bytes, 927_607_488)
        self.assertEqual(
            model.mmproj_digest,
            "sha256:cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e",
        )
        self.assertNotIn("--mmproj", model.args)
        self.assertNotIn("draft-mtp", model.args)

    def test_projector_contract_is_complete_safe_and_vision_only(self) -> None:
        models = load_models(ROOT / "manifests" / "models.toml")
        baseline = models["qwen38-27b-ud-q4-k-xl-llamacpp"]
        vision = models["qwen38-27b-ud-q4-k-xl-llamacpp-vision"]

        with self.assertRaisesRegex(ManifestError, "must be set together"):
            validate_model(replace(baseline, mmproj_file="mmproj.gguf"))
        with self.assertRaisesRegex(ManifestError, "vision task requires"):
            validate_model(replace(baseline, tasks=(*baseline.tasks, "vision")))
        with self.assertRaisesRegex(ManifestError, "requires the vision task"):
            validate_model(
                replace(
                    vision,
                    tasks=tuple(
                        task for task in vision.tasks if task != "vision"
                    ),
                )
            )
        with self.assertRaisesRegex(ManifestError, "safe GGUF filename"):
            validate_model(replace(vision, mmproj_file="../mmproj.gguf"))
        with self.assertRaisesRegex(ManifestError, "differ from model_file"):
            validate_model(replace(vision, mmproj_file=vision.model_file))
        with self.assertRaisesRegex(ManifestError, "fetch_allow_patterns"):
            validate_model(
                replace(vision, fetch_allow_patterns=(str(vision.model_file),))
            )

    def test_reserved_runtime_argument_is_rejected(self) -> None:
        model = load_models(ROOT / "manifests" / "models.toml")[
            "qwen38-27b-ud-q4-k-xl-llamacpp"
        ]
        with self.assertRaisesRegex(ManifestError, "runtime-owned"):
            validate_model(replace(model, args=(*model.args, "--host")))

        for argument in (
            "-mu",
            "-hfr",
            "--parallel=4",
            "--cors-origins",
            "-mm",
            "--mmproj=/tmp/unpinned.gguf",
            "--mmproj-url",
            "--no-mmproj",
            "--no-mmproj-offload",
        ):
            with self.subTest(argument=argument):
                with self.assertRaisesRegex(ManifestError, "runtime-owned"):
                    validate_model(replace(model, args=(*model.args, argument)))

    def test_unsafe_native_runtime_arguments_are_rejected(self) -> None:
        model = load_models(ROOT / "manifests" / "models.toml")[
            "qwen38-27b-ud-q4-k-xl-llamacpp"
        ]
        for argument in (
            "--rpc",
            "--reuse-port",
            "--tools=all",
            "--mcp-servers-config",
            "--agent",
            "--props",
            "--models-dir",
            "--models-autoload",
            "--log-file",
            "--log-prompts-dir=/tmp/prompts",
            "--verbosity",
            "--media-path",
            "--mtp",
            "--dflash",
            "--eagle3",
        ):
            with self.subTest(argument=argument):
                with self.assertRaisesRegex(ManifestError, "unsafe llamacpp"):
                    validate_model(replace(model, args=(*model.args, argument)))

    def test_llamacpp_parallel_is_typed_and_positive(self) -> None:
        model = load_models(ROOT / "manifests" / "models.toml")[
            "qwen38-27b-ud-q4-k-xl-llamacpp"
        ]
        for value in (None, 0, -1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ManifestError, "runtime_parallel"):
                    validate_model(replace(model, runtime_parallel=value))

        with self.assertRaisesRegex(ManifestError, "native_context.*max_context"):
            validate_model(replace(model, native_context=model.native_context - 1))

    def test_llamacpp_endpoint_rejects_userinfo(self) -> None:
        model = load_models(ROOT / "manifests" / "models.toml")[
            "qwen38-27b-ud-q4-k-xl-llamacpp"
        ]
        with self.assertRaisesRegex(ManifestError, "canonical"):
            validate_model(
                replace(model, endpoint="http://userinfo@127.0.0.1:8000/v1")
            )


class LlamaCppVisionContractTests(unittest.TestCase):
    def test_openai_vision_payload_is_typed_inline_png(self) -> None:
        arguments = _request_arguments(
            server=SimpleNamespace(
                backend="llamacpp", base_url="http://127.0.0.1:8000/v1"
            ),
            model=SimpleNamespace(
                served_name="example/vision",
                request_body_json=None,
                max_context=32768,
            ),
            case=SimpleNamespace(
                id="vision-smoke",
                max_output_tokens=32,
                kind="capability",
                requires=("chat", "vision"),
                prompt_repetitions=64,
                temperature=0.0,
            ),
            request_id="vision-contract",
        )
        content = arguments["extra_body"]["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(
            content[1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )

    def test_multimodal_readiness_requires_reported_capability(self) -> None:
        def response(capabilities: list[str]) -> io.BytesIO:
            return io.BytesIO(
                json.dumps(
                    {
                        "data": [
                            {
                                "id": "example/vision",
                            }
                        ],
                        "models": [
                            {
                                "name": "example/vision",
                                "model": "example/vision",
                                "capabilities": capabilities,
                            }
                        ],
                    }
                ).encode()
            )

        with patch(
            "bench.runtime.urllib.request.urlopen",
            side_effect=lambda *_args, **_kwargs: response(
                ["completion", "multimodal"]
            ),
        ):
            self.assertTrue(
                _llamacpp_alias_ready(
                    "http://127.0.0.1:8000/v1",
                    "example/vision",
                    require_multimodal=True,
                )
            )
        with patch(
            "bench.runtime.urllib.request.urlopen",
            side_effect=lambda *_args, **_kwargs: response(["completion"]),
        ):
            self.assertFalse(
                _llamacpp_alias_ready(
                    "http://127.0.0.1:8000/v1",
                    "example/vision",
                    require_multimodal=True,
                )
            )
            self.assertTrue(
                _llamacpp_alias_ready(
                    "http://127.0.0.1:8000/v1", "example/vision"
                )
            )

    def test_inventory_requires_both_model_and_projector(self) -> None:
        profile = load_models(ROOT / "manifests" / "models.toml")[
            "qwen38-27b-ud-q4-k-xl-llamacpp-vision"
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / str(profile.model_file)).write_bytes(b"model")
            runtime = root / "llama-server"
            runtime.write_bytes(b"runtime")
            profile = replace(
                profile, cache_dir="project", runtime_binary=str(runtime)
            )
            inventory = Inventory(
                collected_at="now",
                python_version="3",
                platform="test",
                machine="aarch64",
                huggingface_snapshots=(
                    HuggingFaceSnapshot(
                        source=profile.source,
                        revision=str(profile.revision),
                        path=snapshot,
                    ),
                ),
                docker_images=(),
                ollama_models=(),
            )
            availability = assess_model_availability(
                {profile.id: profile}, inventory
            )[profile.id]
            self.assertFalse(availability.source_available)
            self.assertIn(
                "exact multimodal projector is not cached", availability.details
            )

            (snapshot / str(profile.mmproj_file)).write_bytes(b"projector")
            availability = assess_model_availability(
                {profile.id: profile}, inventory
            )[profile.id]
            self.assertTrue(availability.available)


class LlamaCppMetricsTests(unittest.TestCase):
    def test_parser_persists_acceptance_and_positions(self) -> None:
        metrics = parse_llamacpp_spec_decode_metrics(EXPOSITION)
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertEqual(metrics["num_draft_tokens"], 317)
        self.assertEqual(metrics["num_accepted_tokens"], 148)
        self.assertEqual(metrics["num_drafts"], 106)
        self.assertEqual(
            metrics["accepted_tokens_per_position"],
            {"0": 75, "1": 46, "2": 27},
        )
        self.assertAlmostEqual(metrics["draft_acceptance_rate"], 148 / 317)
        self.assertAlmostEqual(metrics["mean_accepted_length"], 1 + 148 / 106)
        require_mtp_activity(metrics)

    def test_mtp_requires_positive_drafted_and_accepted_activity(self) -> None:
        baseline = parse_llamacpp_spec_decode_metrics(
            EXPOSITION.replace("317", "0").replace("148", "0").replace("106", "0")
        )
        with self.assertRaisesRegex(RuntimeError, "requested"):
            require_mtp_activity(baseline)

    def test_mtp_proposal_depth_is_parsed_proved_and_bounded(self) -> None:
        parsed = parse_llamacpp_spec_decode_metrics(EXPOSITION)
        assert parsed is not None
        self.assertEqual(
            llamacpp_mtp_depth(
                ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
            ),
            3,
        )
        self.assertEqual(llamacpp_mtp_depth(["--spec-draft-n-max=3"]), 3)
        self.assertIsNone(
            llamacpp_mtp_depth(
                ["--spec-draft-n-max", "3", "--spec-draft-n-max=3"]
            )
        )
        self.assertIsNone(llamacpp_mtp_depth(["--spec-draft-n-max", "0"]))

        evidence = assess_llamacpp_mtp_proposal_depth(
            parsed, configured_depth=3
        )
        self.assertTrue(evidence["passed"])
        self.assertAlmostEqual(
            evidence["average_draft_tokens_per_draft"], 317 / 106
        )
        self.assertEqual(evidence["deepest_accepted_position"], 2)
        self.assertEqual(evidence["deepest_accepted_draft_depth"], 3)

        not_exercised = assess_llamacpp_mtp_proposal_depth(
            parsed, configured_depth=4
        )
        self.assertFalse(not_exercised["passed"])
        self.assertIn("do not prove", not_exercised["reason"])

        out_of_range = {
            **parsed,
            "accepted_tokens_per_position": {
                **parsed["accepted_tokens_per_position"],
                "3": 1,
            },
        }
        invalid = assess_llamacpp_mtp_proposal_depth(
            out_of_range, configured_depth=3
        )
        self.assertFalse(invalid["passed"])
        self.assertIn("exceeds", invalid["reason"])

    def test_report_aggregates_resumed_llamacpp_lifetimes(self) -> None:
        parsed = parse_llamacpp_spec_decode_metrics(EXPOSITION)
        assert parsed is not None
        combined = aggregate_llamacpp_spec_decode_metrics([parsed, parsed])
        assert combined is not None
        self.assertEqual(combined["num_draft_tokens"], 634)
        self.assertEqual(combined["accepted_tokens_per_position"]["2"], 54)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "events.jsonl").write_text(
                json.dumps(
                    {
                        "event": "llamacpp_spec_decode_metrics_snapshot",
                        "metrics": parsed,
                    }
                )
                + "\n"
            )
            summary = summarize_run(run_dir)
        self.assertEqual(summary["speculative_decoding"]["num_draft_tokens"], 317)

    def test_mtp_evidence_is_required_for_each_contributing_lifetime(self) -> None:
        parsed = parse_llamacpp_spec_decode_metrics(EXPOSITION)
        assert parsed is not None
        arguments = [
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "3",
        ]
        self.assertTrue(llamacpp_mtp_requested(arguments))
        self.assertTrue(
            llamacpp_mtp_requested(["--spec-type=draft-mtp,ngram-cache"])
        )

        missing = [
            {"event": "run_start"},
            {"event": "case_complete", "case_id": "one"},
        ]
        with self.assertRaisesRegex(RuntimeError, "no later"):
            require_llamacpp_mtp_evidence(arguments, missing)

        zero = parse_llamacpp_spec_decode_metrics(
            EXPOSITION.replace("317", "0")
            .replace("148", "0")
            .replace("106", "0")
        )
        inactive = [
            *missing,
            {
                "event": "llamacpp_spec_decode_metrics_snapshot",
                "metrics": zero,
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "not all positive"):
            require_llamacpp_mtp_evidence(arguments, inactive)

        mixed_lifetimes = [
            *missing,
            {"event": "run_start"},
            {"event": "case_complete", "case_id": "two"},
            {
                "event": "llamacpp_spec_decode_metrics_snapshot",
                "metrics": parsed,
            },
        ]
        evidence = assess_llamacpp_mtp_evidence(
            mixed_lifetimes, requested=True
        )
        self.assertFalse(evidence["passed"])
        self.assertEqual(evidence["contributing_lifetimes"], 2)
        self.assertEqual(evidence["validated_lifetimes"], 0)

        complete = [
            *missing,
            {
                "event": "llamacpp_spec_decode_metrics_snapshot",
                "metrics": parsed,
            },
            {"event": "run_start"},
            {"event": "case_complete", "case_id": "two"},
            {
                "event": "llamacpp_spec_decode_metrics_snapshot",
                "metrics": parsed,
            },
        ]
        evidence = require_llamacpp_mtp_evidence(arguments, complete)
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["contributing_lifetimes"], 2)
        self.assertEqual(evidence["validated_lifetimes"], 2)
        self.assertEqual(evidence["proposal_depth_validated_lifetimes"], 2)

    def test_dflash_evidence_requires_positive_per_lifetime_activity(self) -> None:
        parsed = parse_llamacpp_spec_decode_metrics(EXPOSITION)
        assert parsed is not None
        arguments = [
            "--spec-type",
            "draft-dflash",
            "--spec-draft-n-max",
            "15",
        ]
        self.assertTrue(llamacpp_dflash_requested(arguments))
        self.assertFalse(llamacpp_mtp_requested(arguments))
        missing = [
            {"event": "run_start"},
            {"event": "case_complete", "case_id": "one"},
        ]
        with self.assertRaisesRegex(RuntimeError, "DFlash evidence"):
            require_llamacpp_dflash_evidence(arguments, missing)
        evidence = require_llamacpp_dflash_evidence(
            arguments,
            [
                *missing,
                {
                    "event": "llamacpp_spec_decode_metrics_snapshot",
                    "metrics": parsed,
                },
            ],
        )
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["validated_lifetimes"], 1)
        self.assertEqual(evidence["proposal_depth_validated_lifetimes"], 0)

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "plan.json").write_text(
                json.dumps(
                    {
                        "model": {"backend": "llamacpp", "args": arguments},
                        "suite": {"id": "fixture"},
                    }
                )
            )
            (run_dir / "events.jsonl").write_text(
                "\n".join(
                    json.dumps(event)
                    for event in (
                        {"event": "run_start"},
                        {
                            "event": "case_complete",
                            "case_id": "one",
                            "attempt_id": "attempt",
                            "kind": "decode",
                            "elapsed_s": 1.0,
                        },
                        {
                            "event": "llamacpp_spec_decode_metrics_snapshot",
                            "metrics": parsed,
                        },
                        {"event": "run_complete", "status": "completed"},
                    )
                )
                + "\n"
            )
            summary = summarize_run(run_dir)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(
            summary["speculative_decoding"]["method"], "draft-dflash"
        )
        self.assertEqual(
            summary["speculative_decoding"]["configured_max_draft_tokens"],
            15,
        )
        self.assertTrue(summary["llamacpp_dflash_evidence"]["passed"])
        self.assertIsNone(summary["llamacpp_mtp_evidence"])

    def test_report_records_and_enforces_configured_mtp_depth(self) -> None:
        parsed = parse_llamacpp_spec_decode_metrics(EXPOSITION)
        assert parsed is not None
        for depth, expected_status in ((3, "complete"), (4, "partial")):
            with self.subTest(depth=depth), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                (run_dir / "plan.json").write_text(
                    json.dumps(
                        {
                            "model": {
                                "backend": "llamacpp",
                                "args": [
                                    "--spec-type",
                                    "draft-mtp",
                                    "--spec-draft-n-max",
                                    str(depth),
                                ],
                            },
                            "suite": {"id": "fixture"},
                        }
                    )
                )
                (run_dir / "events.jsonl").write_text(
                    "\n".join(
                        json.dumps(event)
                        for event in (
                            {"event": "run_start"},
                            {
                                "event": "case_complete",
                                "case_id": "decode-256",
                                "attempt_id": "attempt",
                                "kind": "decode",
                                "elapsed_s": 1.0,
                            },
                            {
                                "event": "llamacpp_spec_decode_metrics_snapshot",
                                "metrics": parsed,
                            },
                            {"event": "run_complete", "status": "completed"},
                        )
                    )
                    + "\n"
                )
                summary = summarize_run(run_dir)

            speculative = summary["speculative_decoding"]
            self.assertEqual(speculative["configured_max_draft_tokens"], depth)
            self.assertEqual(
                speculative["proposal_depth"]["passed"], depth == 3
            )
            evidence = summary["llamacpp_mtp_evidence"]
            self.assertEqual(evidence["configured_max_draft_tokens"], depth)
            self.assertEqual(evidence["passed"], depth == 3)
            self.assertEqual(summary["status"], expected_status)

    def test_report_fails_closed_when_mtp_evidence_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "plan.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "backend": "llamacpp",
                            "args": ["--spec-type", "draft-mtp"],
                        },
                        "suite": {"id": "fixture"},
                    }
                )
            )
            (run_dir / "events.jsonl").write_text(
                "\n".join(
                    json.dumps(event)
                    for event in (
                        {"event": "run_start"},
                        {
                            "event": "case_complete",
                            "case_id": "one",
                            "attempt_id": "attempt",
                            "kind": "decode",
                            "elapsed_s": 1.0,
                        },
                        {"event": "run_complete", "status": "completed"},
                    )
                )
                + "\n"
            )
            summary = summarize_run(run_dir)
        self.assertEqual(summary["status"], "partial")
        self.assertFalse(summary["llamacpp_mtp_evidence"]["passed"])


class LlamaCppRuntimeTests(unittest.TestCase):
    def _write_plan(
        self, root: Path, *, mtp: bool, dflash: bool = False
    ) -> tuple[Path, str]:
        if mtp and dflash:
            raise ValueError("fixture cannot request MTP and DFlash together")
        arguments = (
            ["--spec-type", "draft-dflash", "--spec-draft-n-max", "15"]
            if dflash
            else (["--spec-type", "draft-mtp"] if mtp else [])
        )
        model = {
            "id": "llamacpp-fixture",
            "backend": "llamacpp",
            "source": "example/model",
            "served_name": "example/model",
            "tasks": ["chat"],
            "max_context": 1024,
            "endpoint": "http://127.0.0.1:8000/v1",
            "args": arguments,
        }
        case = {
            "id": "decode",
            "kind": "decode",
            "requires": ["chat"],
            "warmups": 0,
            "repetitions": 1,
            "max_output_tokens": 8,
            "temperature": 0.0,
            "concurrency": 1,
            "prompt_repetitions": 0,
        }
        suite = {
            "id": "suite",
            "description": "",
            "schema_version": 1,
            "cases": [case],
        }
        case_id = f"decode--{content_hash({'model': model, 'case': case}, 12)}"
        plan = {
            "fingerprint": content_hash({"model": model, "suite": suite}),
            "model": model,
            "suite": {**suite, "cases": [{**case, "case_id": case_id}]},
            "resolved": {},
        }
        run_dir = root / "run"
        run_dir.mkdir()
        (run_dir / "plan.json").write_text(json.dumps(plan))
        return run_dir, case_id

    def _fixture(
        self, root: Path, *, vision: bool = False, dflash: bool = False
    ) -> tuple[Path, SimpleNamespace]:
        source_dir = root / "llama.cpp"
        binary = source_dir / "build" / "bin" / "llama-server"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"static llama server")
        binary.chmod(0o755)
        revision = "a" * 40
        model_file = "model.gguf"
        gguf = (
            root
            / "data"
            / "huggingface"
            / "hub"
            / "models--example--gguf"
            / "snapshots"
            / revision
            / model_file
        )
        gguf.parent.mkdir(parents=True)
        gguf.write_bytes(b"tiny gguf fixture")
        mmproj = gguf.with_name("mmproj.gguf")
        if vision:
            mmproj.write_bytes(b"tiny projector fixture")
        draft_revision = "c" * 40
        draft = (
            root
            / "data"
            / "huggingface"
            / "hub"
            / "models--example--draft"
            / "snapshots"
            / draft_revision
            / "dflash.gguf"
        )
        if dflash:
            draft.parent.mkdir(parents=True)
            draft.write_bytes(b"tiny dflash fixture")
        model = SimpleNamespace(
            backend="llamacpp",
            source="example/gguf",
            revision=revision,
            served_name="example/gguf",
            tasks=["chat", "vision"] if vision else ["chat"],
            cache_dir="project",
            model_file=model_file,
            model_digest=_digest(gguf),
            model_size_bytes=gguf.stat().st_size,
            mmproj_file=mmproj.name if vision else None,
            mmproj_digest=_digest(mmproj) if vision else None,
            mmproj_size_bytes=mmproj.stat().st_size if vision else None,
            draft_source="example/draft" if dflash else None,
            draft_revision=draft_revision if dflash else None,
            draft_weight_size_bytes=None,
            draft_model_file=draft.name if dflash else None,
            draft_model_digest=_digest(draft) if dflash else None,
            draft_model_size_bytes=draft.stat().st_size if dflash else None,
            runtime_binary=str(binary),
            runtime_digest=_digest(binary),
            runtime_source_dir=str(source_dir),
            runtime_revision="b" * 40,
            runtime_parallel=8,
            max_context=32768,
            native_context=262144,
            endpoint="http://127.0.0.1:8000/v1",
            startup_timeout_s=5,
            run_identity="frozen-run-1",
            args=(
                [
                    "--reasoning",
                    "off",
                    "--spec-type",
                    "draft-dflash",
                    "--spec-draft-n-max",
                    "15",
                ]
                if dflash
                else ["--reasoning", "off"]
            ),
        )
        return gguf, model

    def test_launch_is_exact_offline_loopback_and_secret_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _, model = self._fixture(workspace)
            process = Mock(pid=4242)
            process.poll.return_value = None
            git_results = [
                _completed(stdout=model.runtime_revision + "\n"),
                _completed(),
            ]
            with (
                patch("bench.runtime._run", side_effect=git_results),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime.subprocess.Popen", return_value=process) as popen,
                patch("bench.runtime._proc_start_ticks", return_value=123456),
                patch("bench.runtime.os.getpgid", return_value=4242),
                patch("bench.runtime.wait_for_llamacpp", return_value=4.25),
                patch.dict(
                    os.environ,
                    {
                        "HF_TOKEN": "secret",
                        "OPENAI_API_KEY": "secret",
                        "LD_LIBRARY_PATH": "/untrusted/runtime",
                    },
                    clear=False,
                ),
            ):
                server = start_llamacpp(
                    model,
                    workspace=workspace,
                    allow_download=True,
                    server_log_path=workspace / "run" / "server.log",
                    process_state_path=workspace / "run" / "process.json",
                )

            command = popen.call_args.args[0]
            kwargs = popen.call_args.kwargs
            self.assertEqual(command[0], str(Path(model.runtime_binary).resolve()))
            self.assertIn("--model", command)
            self.assertIn("--host", command)
            self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
            self.assertIn("--offline", command)
            self.assertIn("--metrics", command)
            self.assertIn("--no-ui", command)
            self.assertEqual(command[command.index("--ctx-size") + 1], "262144")
            self.assertEqual(command[command.index("--parallel") + 1], "8")
            self.assertEqual(
                command[command.index("--cors-origins") + 1], "localhost"
            )
            self.assertIn("--no-cors-credentials", command)
            self.assertTrue(kwargs["start_new_session"])
            self.assertFalse(kwargs["shell"])
            self.assertNotIn("HF_TOKEN", kwargs["env"])
            self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
            self.assertNotIn("LD_LIBRARY_PATH", kwargs["env"])
            self.assertEqual(kwargs["env"]["SPARKBENCH_RUN_ID"], "frozen-run-1")
            state = json.loads((workspace / "run" / "process.json").read_text())
            self.assertEqual(state["pid"], 4242)
            self.assertEqual(state["start_ticks"], 123456)
            self.assertEqual(server.startup_s, 4.25)
            assert server.native_provenance is not None
            self.assertEqual(server.native_provenance["runtime_parallel"], 8)
            self.assertEqual(
                server.native_provenance["runtime_total_context"], 262144
            )
            self.assertEqual(
                server.native_provenance["served_max_context"], 32768
            )
            self.assertEqual(
                server.native_provenance["cors_origins"], "localhost"
            )
            self.assertFalse(server.native_provenance["cors_credentials"])
            self.assertIsInstance(
                server.native_provenance["artifact_validation_s"], float
            )
            assert server.process_log is not None
            server.process_log.close()

    def test_hash_mismatch_fails_before_process_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _, model = self._fixture(workspace)
            model.model_digest = "sha256:" + "0" * 64
            with (
                patch(
                    "bench.runtime._run",
                    side_effect=[
                        _completed(stdout=model.runtime_revision + "\n"),
                        _completed(),
                    ],
                ),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime.subprocess.Popen") as popen,
            ):
                with self.assertRaisesRegex(RuntimeErrorWithContext, "GGUF SHA-256"):
                    start_llamacpp(
                        model,
                        workspace=workspace,
                        server_log_path=workspace / "run" / "server.log",
                        process_state_path=workspace / "run" / "process.json",
                    )
            popen.assert_not_called()

    def test_vision_launch_injects_only_verified_projector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            gguf, model = self._fixture(workspace, vision=True)
            process = Mock(pid=4243)
            process.poll.return_value = None
            with (
                patch(
                    "bench.runtime._run",
                    side_effect=[
                        _completed(stdout=model.runtime_revision + "\n"),
                        _completed(),
                    ],
                ),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime.subprocess.Popen", return_value=process) as popen,
                patch("bench.runtime._proc_start_ticks", return_value=123457),
                patch("bench.runtime.os.getpgid", return_value=4243),
                patch(
                    "bench.runtime.wait_for_llamacpp", return_value=5.0
                ) as wait,
            ):
                server = start_llamacpp(
                    model,
                    workspace=workspace,
                    server_log_path=workspace / "run" / "server.log",
                    process_state_path=workspace / "run" / "process.json",
                )

            command = popen.call_args.args[0]
            expected_mmproj = str(gguf.with_name("mmproj.gguf"))
            self.assertEqual(
                command[command.index("--mmproj") + 1], expected_mmproj
            )
            self.assertNotIn("--mmproj-url", command)
            wait.assert_called_once_with(
                server,
                served_name=model.served_name,
                timeout_s=float(model.startup_timeout_s),
                require_multimodal=True,
            )
            state = json.loads((workspace / "run" / "process.json").read_text())
            self.assertEqual(state["mmproj_path"], expected_mmproj)
            self.assertEqual(state["mmproj_digest"], model.mmproj_digest)
            assert server.native_provenance is not None
            self.assertEqual(
                server.native_provenance["mmproj_sha256"], model.mmproj_digest
            )
            self.assertEqual(
                server.native_provenance["mmproj_size_bytes"],
                model.mmproj_size_bytes,
            )
            self.assertTrue(server.native_provenance["multimodal"])
            assert server.process_log is not None
            server.process_log.close()

    def test_projector_hash_mismatch_fails_before_process_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _, model = self._fixture(workspace, vision=True)
            model.mmproj_digest = "sha256:" + "0" * 64
            with (
                patch(
                    "bench.runtime._run",
                    side_effect=[
                        _completed(stdout=model.runtime_revision + "\n"),
                        _completed(),
                    ],
                ),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime.subprocess.Popen") as popen,
            ):
                with self.assertRaisesRegex(
                    RuntimeErrorWithContext, "mmproj GGUF SHA-256"
                ):
                    start_llamacpp(
                        model,
                        workspace=workspace,
                        server_log_path=workspace / "run" / "server.log",
                        process_state_path=workspace / "run" / "process.json",
                    )
            popen.assert_not_called()

    def test_dflash_launch_validates_and_owns_exact_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _, model = self._fixture(workspace, dflash=True)
            process = Mock(pid=4244)
            process.poll.return_value = None
            with (
                patch(
                    "bench.runtime._run",
                    side_effect=[
                        _completed(stdout=model.runtime_revision + "\n"),
                        _completed(),
                    ],
                ),
                patch("bench.runtime._port_is_free", return_value=True),
                patch(
                    "bench.runtime.subprocess.Popen", return_value=process
                ) as popen,
                patch("bench.runtime._proc_start_ticks", return_value=123458),
                patch("bench.runtime.os.getpgid", return_value=4244),
                patch("bench.runtime.wait_for_llamacpp", return_value=5.0),
            ):
                server = start_llamacpp(
                    model,
                    workspace=workspace,
                    server_log_path=workspace / "run" / "server.log",
                    process_state_path=workspace / "run" / "process.json",
                )

            command = popen.call_args.args[0]
            expected_draft = str(
                workspace
                / "data"
                / "huggingface"
                / "hub"
                / "models--example--draft"
                / "snapshots"
                / str(model.draft_revision)
                / str(model.draft_model_file)
            )
            self.assertEqual(
                command[command.index("--spec-draft-model") + 1],
                expected_draft,
            )
            state = json.loads((workspace / "run" / "process.json").read_text())
            self.assertEqual(state["draft_model_path"], expected_draft)
            self.assertEqual(
                state["draft_model_digest"], model.draft_model_digest
            )
            assert server.native_provenance is not None
            self.assertEqual(
                server.native_provenance["draft_model_sha256"],
                model.draft_model_digest,
            )
            assert server.process_log is not None
            server.process_log.close()

    def test_dflash_hash_mismatch_fails_before_process_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _, model = self._fixture(workspace, dflash=True)
            model.draft_model_digest = "sha256:" + "0" * 64
            with (
                patch(
                    "bench.runtime._run",
                    side_effect=[
                        _completed(stdout=model.runtime_revision + "\n"),
                        _completed(),
                    ],
                ),
                patch("bench.runtime._port_is_free", return_value=True),
                patch("bench.runtime.subprocess.Popen") as popen,
            ):
                with self.assertRaisesRegex(
                    RuntimeErrorWithContext, "draft GGUF SHA-256"
                ):
                    validate_llamacpp_artifacts(model, workspace=workspace)
            popen.assert_not_called()

    def test_terminal_mtp_resume_refuses_missing_lifetime_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_dir, case_id = self._write_plan(workspace, mtp=True)
            journal = Journal(run_dir / "events.jsonl")
            journal.append({"event": "run_start"})
            journal.append(
                {
                    "event": "case_complete",
                    "case_id": case_id,
                    "attempt_id": "attempt",
                    "kind": "decode",
                    "elapsed_s": 1.0,
                }
            )

            with (
                patch(
                    "bench.runner._recover_pending_lifecycle",
                    return_value=False,
                ),
                patch("bench.runner._preflight") as preflight,
                patch("bench.runner.start_server") as start_server,
            ):
                with self.assertRaisesRegex(RuntimeError, "MTP evidence"):
                    execute_plan(run_dir, workspace=workspace)

            preflight.assert_not_called()
            start_server.assert_not_called()
            events = Journal(run_dir / "events.jsonl").events()
            self.assertFalse(
                any(event.get("event") == "run_complete" for event in events)
            )
            aborted = [
                event for event in events if event.get("event") == "run_aborted"
            ]
            self.assertEqual(aborted[-1]["stage"], "mtp_evidence")
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["status"], "aborted")
            self.assertFalse(summary["llamacpp_mtp_evidence"]["passed"])

    def test_terminal_dflash_resume_refuses_missing_lifetime_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_dir, case_id = self._write_plan(
                workspace, mtp=False, dflash=True
            )
            journal = Journal(run_dir / "events.jsonl")
            journal.append({"event": "run_start"})
            journal.append(
                {
                    "event": "case_complete",
                    "case_id": case_id,
                    "attempt_id": "attempt",
                    "kind": "decode",
                    "elapsed_s": 1.0,
                }
            )

            with (
                patch(
                    "bench.runner._recover_pending_lifecycle",
                    return_value=False,
                ),
                patch("bench.runner._preflight") as preflight,
                patch("bench.runner.start_server") as start_server,
            ):
                with self.assertRaisesRegex(RuntimeError, "DFlash evidence"):
                    execute_plan(run_dir, workspace=workspace)

            preflight.assert_not_called()
            start_server.assert_not_called()
            events = Journal(run_dir / "events.jsonl").events()
            aborted = [
                event for event in events if event.get("event") == "run_aborted"
            ]
            self.assertEqual(aborted[-1]["stage"], "dflash_evidence")
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["status"], "aborted")
            self.assertFalse(summary["llamacpp_dflash_evidence"]["passed"])

    def test_runner_separates_artifact_validation_from_server_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_dir, _ = self._write_plan(workspace, mtp=False)
            artifacts = {
                "runtime_binary_sha256": "sha256:" + "1" * 64,
                "model_sha256": "sha256:" + "2" * 64,
                "mmproj_sha256": "sha256:" + "3" * 64,
            }
            telemetry = Mock()
            server = SimpleNamespace(
                backend="llamacpp",
                base_url="http://127.0.0.1:8000/v1",
                startup_s=0.25,
                container_id=None,
                process=None,
                process_log=None,
                native_provenance={
                    **artifacts,
                    "artifact_validation_s": 0.01,
                },
                ollama_model=None,
                unload_ollama=False,
                stop=Mock(),
            )
            prime_result = Mock()
            prime_result.to_dict.return_value = {"elapsed_s": 0.01}

            def complete_case(**kwargs: object) -> None:
                case = kwargs["case"]
                journal = kwargs["journal"]
                journal.append(
                    {
                        "event": "case_complete",
                        "case_id": case.case_id,
                        "attempt_id": "attempt",
                        "kind": case.kind,
                        "elapsed_s": 0.1,
                    }
                )

            with (
                patch("bench.runner._preflight"),
                patch("bench.runner.TelemetrySampler", return_value=telemetry),
                patch(
                    "bench.runner.validate_llamacpp_artifacts",
                    return_value=artifacts,
                ) as validate,
                patch(
                    "bench.runner.start_server", return_value=server
                ) as start_server,
                patch("bench.runner._prime_model", return_value=prime_result),
                patch("bench.runner._execute_case", side_effect=complete_case),
                patch(
                    "bench.runner.snapshot_llamacpp_spec_decode_metrics",
                    return_value=None,
                ),
            ):
                summary = execute_plan(run_dir, workspace=workspace)

            validate.assert_called_once()
            self.assertEqual(
                start_server.call_args.kwargs["validated_llamacpp_artifacts"],
                artifacts,
            )
            self.assertIsInstance(
                start_server.call_args.kwargs["artifact_validation_s"], float
            )
            phases = [call.args[0] for call in telemetry.set_phase.call_args_list]
            self.assertLess(
                phases.index("artifact_validation"), phases.index("server_startup")
            )
            artifact_event = next(
                event
                for event in Journal(run_dir / "events.jsonl").events()
                if event.get("event") == "artifact_validation_complete"
            )
            self.assertGreaterEqual(artifact_event["elapsed_s"], 0)
            self.assertEqual(
                artifact_event["model_sha256"], artifacts["model_sha256"]
            )
            self.assertEqual(
                artifact_event["mmproj_sha256"], artifacts["mmproj_sha256"]
            )
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(
                summary["artifact_validation"]["model_sha256"],
                artifacts["model_sha256"],
            )
            self.assertEqual(
                summary["artifact_validation"]["mmproj_sha256"],
                artifacts["mmproj_sha256"],
            )
            server.stop.assert_called_once_with(keep_server=False)

    def test_recovery_uses_exact_state_and_keep_server_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _, model = self._fixture(workspace)
            state_path = workspace / "process.json"
            state_path.write_text(json.dumps({"run_identity": model.run_identity}))
            with patch(
                "bench.runtime._stop_native_state",
                return_value="stopped_owned_process",
            ) as stop:
                action = recover_owned_llamacpp(
                    model,
                    workspace=workspace,
                    run_identity=model.run_identity,
                    process_state_path=state_path,
                )
            self.assertEqual(action, "stopped_owned_process")
            stop.assert_called_once_with(state_path, model.run_identity)

        server = ManagedServer(
            backend="llamacpp",
            base_url="http://127.0.0.1:8000/v1",
            run_identity="run",
            process_state_path=Path("/mock/process.json"),
        )
        with self.assertRaisesRegex(RuntimeErrorWithContext, "keep-server"):
            server.stop(keep_server=True)


if __name__ == "__main__":
    unittest.main()
