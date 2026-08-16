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
    llamacpp_mtp_depth,
    llamacpp_mtp_requested,
    parse_llamacpp_spec_decode_metrics,
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
from bench.runner import _request_arguments, execute_plan
from bench.runtime import (
    ManagedServer,
    RuntimeErrorWithContext,
    _llamacpp_alias_ready,
    recover_owned_llamacpp,
    start_llamacpp,
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
    def _write_plan(self, root: Path, *, mtp: bool) -> tuple[Path, str]:
        model = {
            "id": "llamacpp-fixture",
            "backend": "llamacpp",
            "source": "example/model",
            "served_name": "example/model",
            "tasks": ["chat"],
            "max_context": 1024,
            "endpoint": "http://127.0.0.1:8000/v1",
            "args": ["--spec-type", "draft-mtp"] if mtp else [],
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
        self, root: Path, *, vision: bool = False
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
            args=["--reasoning", "off"],
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
