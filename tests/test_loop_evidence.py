from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from bench.evidence import (
    EvidenceError,
    LOOP_EVIDENCE_KIND,
    SCHEMA_VERSION,
    _LOOP_MEASUREMENT_FIELDS,
    _export_loop_campaign,
    _project_loop_model,
    _validate_projected_sglang_provenance,
    _validate_runtime_overlay_tree,
    _verify_simple_bundle,
)
from bench.loop_campaign import (
    WORKER_IMAGE,
    _case_id,
    _content_hash,
    _rlm_compaction_admission,
    build_cases,
    load_campaign_manifest,
    summarize_campaign,
)
from bench.manifest import load_models, model_spec_to_dict


REPOSITORY = Path(__file__).resolve().parents[1]
CAMPAIGN = REPOSITORY / "manifests" / "campaigns" / "rlm_halo_overnight.toml"
MODELS = REPOSITORY / "manifests" / "models.toml"
SOURCE_GROUP = "loop-smokes"
RAW_SENTINEL = "RAW_LOOP_PROMPT_AND_COMPLETION_SENTINEL"
RAW_HOST_PATH = "/home/private-user/loop/prompt.json"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical(value) for value in values))


def _rehash_plan(plan: dict[str, object]) -> dict[str, object]:
    plan.pop("integrity_hash", None)
    plan.pop("fingerprint", None)
    plan["fingerprint"] = _content_hash(plan)
    plan["integrity_hash"] = _content_hash(plan)
    return plan


def _model_record(profile: object) -> dict[str, object]:
    # Keep this conversion independent of the evidence projector while matching the
    # controller's frozen-plan representation.
    record = model_spec_to_dict(profile)
    record["tasks"] = list(profile.tasks)
    record["args"] = list(profile.args)
    record["model_shards"] = [asdict(shard) for shard in profile.model_shards]
    record["sglang_source_overlays"] = [
        asdict(overlay) for overlay in profile.sglang_source_overlays
    ]
    return record


def _fixture_plan(variant: str = "current_v2") -> dict[str, object]:
    config = load_campaign_manifest(CAMPAIGN)
    profiles = load_models(MODELS)
    rlm_profile = str(config["rlm"]["model_profile"])
    halo_profile = str(config["halo"]["model_profiles"][0])
    cases = build_cases(config)
    selected_cases = [
        copy.deepcopy(next(case for case in cases if case["phase"] == phase))
        for phase in ("rlm", "halo")
    ]
    models = {
        profile_id: _model_record(profiles[profile_id])
        for profile_id in (rlm_profile, halo_profile)
    }
    # These raw values are intentionally sensitive-looking and must be dropped, not
    # copied or recursively sanitized into tracked evidence.
    models[rlm_profile]["description"] = RAW_SENTINEL
    models[rlm_profile]["args"] = [*models[rlm_profile]["args"], RAW_HOST_PATH]
    models[halo_profile]["request_body_json"] = RAW_SENTINEL

    rlm = copy.deepcopy(config["rlm"])
    halo = copy.deepcopy(config["halo"])
    plan: dict[str, object] = {
        "schema_version": 2,
        "protocol_version": 2,
        "campaign_id": "synthetic-loop-evidence",
        "description": RAW_SENTINEL,
        "created_at": "2026-08-26T01:02:03+00:00",
        "window": {
            "rlm_stop_at": "2026-08-26T02:00:00+00:00",
            "measurement_stop_at": "2026-08-26T03:00:00+00:00",
            "hard_stop_at": "2026-08-26T04:00:00+00:00",
            "cleanup_reserve_s": 60,
        },
        "upstreams": copy.deepcopy(config["upstreams"]),
        "rlm": rlm,
        "rlm_compaction_admission": _rlm_compaction_admission(
            rlm, served_context_tokens=int(profiles[rlm_profile].max_context)
        ),
        "halo": halo,
        "models": models,
        "dataset": {
            "source": config["upstreams"]["babilong_source"],
            "revision": config["upstreams"]["babilong_revision"],
            "rows_per_split": 100,
            "selected_files": [
                {
                    "context_length": "8k",
                    "task": "qa1",
                    "size_bytes": 1234,
                    "sha256": "sha256:" + "a" * 64,
                }
            ],
        },
        "worker": {"isolation": "docker", "image": WORKER_IMAGE},
        "repository": {"clean": True, "revision": "b" * 40},
        "cases": selected_cases,
    }
    if variant in {"legacy_v2", "legacy_v1"}:
        plan.pop("rlm_compaction_admission")
        for model in models.values():
            for field in (
                "sglang_ple_cache_marker_digest",
                "sglang_ple_cache_mode",
                "sglang_ple_cache_payload_digest",
                "sglang_ple_mmap",
                "sglang_ple_omitted",
                "sglang_source_overlays",
            ):
                model.pop(field, None)
        for key in ("compaction", "compaction_threshold_pct"):
            rlm.pop(key)
        for case in selected_cases:
            case.pop("admission_status", None)
            case.pop("compaction", None)
            case.pop("compaction_threshold_pct", None)
            case["case_id"] = _case_id(
                {key: value for key, value in case.items() if key != "case_id"}
            )
    if variant == "legacy_v1":
        plan["schema_version"] = 1
        plan["protocol_version"] = 1
        rlm.pop("reasoning_control")
        halo.pop("reasoning_effort")
        for case in selected_cases:
            case.pop("reasoning_control", None)
            case.pop("reasoning_effort", None)
            case["case_id"] = _case_id(
                {key: value for key, value in case.items() if key != "case_id"}
            )
    if variant not in {"current_v2", "legacy_v2", "legacy_v1"}:
        raise AssertionError(f"unknown synthetic fixture variant: {variant}")
    return _rehash_plan(plan)


def _add_readonly_sglang_provenance(
    plan: dict[str, object], results: Path
) -> tuple[str, list[dict[str, str]]]:
    profile_id = str(plan["rlm"]["model_profile"])
    model = plan["models"][profile_id]
    overlay_dir = results / "runtime-overlays" / "synthetic-loop-sglang"
    overlay_dir.mkdir(parents=True)
    files = {
        "qwen4_exp.py": (
            "MODEL_KIND = 'synthetic-loop'\n",
            "/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py",
        ),
        "qwen_sparse_attn_backend.py": (
            "BACKEND_KIND = 'synthetic-loop'\n",
            "/sgl-workspace/sglang/python/sglang/srt/layers/attention/"
            "qwen_sparse_attn_backend.py",
        ),
    }
    overlays: list[dict[str, str]] = []
    for basename, (source, container_path) in files.items():
        path = overlay_dir / basename
        path.write_text(source, encoding="utf-8")
        overlays.append(
            {
                "container_path": container_path,
                "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                "host_path": (
                    "results/runtime-overlays/synthetic-loop-sglang/" + basename
                ),
            }
        )
    model.update(
        {
            "backend": "sglang",
            "sglang_ple_cache_marker_digest": "sha256:" + "c" * 64,
            "sglang_ple_cache_mode": "readonly",
            "sglang_ple_cache_payload_digest": "sha256:" + "d" * 64,
            "sglang_ple_mmap": True,
            "sglang_source_overlays": overlays,
        }
    )
    _rehash_plan(plan)
    return profile_id, overlays


def _materialize_source(
    root: Path, plan: dict[str, object], *, lifecycle: bool = False
) -> Path:
    fingerprint = str(plan["fingerprint"])
    run_id = f"20260826T010203Z-synthetic-loop-{fingerprint[:8]}"
    run_dir = root / SOURCE_GROUP / run_id
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "plan.json", plan)
    if lifecycle:
        (run_dir / "private").mkdir()
        (run_dir / "private" / "prompt.json").write_text(
            f"{RAW_SENTINEL}\n{RAW_HOST_PATH}\n", encoding="utf-8"
        )
        (run_dir / "server").mkdir()
        (run_dir / "server" / "runtime.log").write_text(
            RAW_SENTINEL + "\n", encoding="utf-8"
        )
        cases = list(plan["cases"])
        rlm_profile = str(plan["rlm"]["model_profile"])
        halo_profile = str(plan["halo"]["model_profiles"][0])
        events: list[dict[str, object]] = [
            {
                "event": "campaign_started",
                "timestamp": "2026-08-26T01:02:03+00:00",
                "plan_fingerprint": fingerprint,
                "repository_revision": "b" * 40,
            }
        ]
        for index, (case, profile_id) in enumerate(
            zip(cases, (rlm_profile, halo_profile), strict=True), start=1
        ):
            started = f"2026-08-26T01:02:{index * 10:02d}+00:00"
            finished = f"2026-08-26T01:02:{index * 10 + 1:02d}+00:00"
            events.append(
                {
                    "event": "case_started",
                    "timestamp": started,
                    "case_id": case["case_id"],
                    "attempt": 1,
                    "profile_id": profile_id,
                }
            )
            dimensions = {
                key: value
                for key, value in case.items()
                if key in _LOOP_MEASUREMENT_FIELDS
            }
            measurement: dict[str, object] = {
                **dimensions,
                "event": "case_complete",
                "timestamp": finished,
                "attempt": 1,
                "profile_id": profile_id,
                "vllm_prompt_tokens": 100 * index,
                "vllm_cached_prompt_tokens": 40 * index,
                "vllm_generation_tokens": 20 * index,
                "vllm_successful_requests": index + 1,
                "wall_s": 2.0 * index,
                "effective_generation_tps": 10.0,
            }
            if case["phase"] == "rlm":
                measurement.update({"correct": True, "reported_calls": 2})
            else:
                measurement.update(
                    {
                        "json_valid": True,
                        "citation_precision": 0.75,
                        "mean_count_accuracy": 0.5,
                        "family_f1": 0.625,
                    }
                )
            events.append(measurement)
        events.extend(
            [
                {
                    "event": "campaign_cleanup_verified",
                    "timestamp": "2026-08-26T01:02:40+00:00",
                },
                {
                    "event": "campaign_finished",
                    "timestamp": "2026-08-26T01:02:41+00:00",
                    "status": "complete",
                    "completed_cases": len(cases),
                },
            ]
        )
        _write_jsonl(run_dir / "journal.jsonl", events)
        summarize_campaign(run_dir)
        _write_jsonl(
            run_dir / "telemetry.jsonl",
            [
                {
                    "cached_kib": 10,
                    "gpu_timestamp": "2026/08/26 01:02:03.000",
                    "gpu_util_pct": 25.0,
                    "memavailable_kib": 1000,
                    "memfree_kib": 900,
                    "memory_util_pct": 50.0,
                    "phase": "rlm_cases",
                    "power_w": 100.0,
                    "sm_clock_mhz": 1000.0,
                    "swapfree_kib": 800,
                    "swaptotal_kib": 1000,
                    "temperature_c": 40.0,
                    "timestamp": "2026-08-26T01:02:03+00:00",
                },
                {
                    "cached_kib": 11,
                    "gpu_timestamp": "2026/08/26 01:02:04.000",
                    "gpu_util_pct": 50.0,
                    "memavailable_kib": 990,
                    "memfree_kib": 890,
                    "memory_util_pct": 60.0,
                    "phase": "rlm_cases",
                    "power_w": 110.0,
                    "sm_clock_mhz": 1100.0,
                    "swapfree_kib": 800,
                    "swaptotal_kib": 1000,
                    "temperature_c": 41.0,
                    "timestamp": "2026-08-26T01:02:04+00:00",
                },
                {
                    "cached_kib": 12,
                    "gpu_timestamp": "2026/08/26 01:02:20.000",
                    "gpu_util_pct": 75.0,
                    "memavailable_kib": 980,
                    "memfree_kib": 880,
                    "memory_util_pct": 70.0,
                    "phase": "halo_cases",
                    "power_w": 120.0,
                    "sm_clock_mhz": 1200.0,
                    "swapfree_kib": 800,
                    "swaptotal_kib": 1000,
                    "temperature_c": 42.0,
                    "timestamp": "2026-08-26T01:02:20+00:00",
                },
            ],
        )
    return run_dir


def _bundle_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _refresh_bundle_checksums(root: Path, entry: dict[str, object]) -> None:
    bundle = root / "campaigns" / str(entry["campaign_id"])
    checksums = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(bundle.iterdir())
        if path.is_file() and path.name != "checksums.json"
    }
    checksum_data = _canonical(
        {"files": checksums, "schema_version": SCHEMA_VERSION}
    )
    (bundle / "checksums.json").write_bytes(checksum_data)
    entry["bundle_sha256"] = hashlib.sha256(checksum_data).hexdigest()


class LoopEvidenceTests(unittest.TestCase):
    def test_plan_variants_export_deterministically_without_private_values(self) -> None:
        for variant in ("legacy_v1", "legacy_v2", "current_v2"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                results = root / "results"
                run_dir = _materialize_source(results, _fixture_plan(variant))
                first = root / "first"
                second = root / "second"

                first_entry = _export_loop_campaign(
                    run_dir, results, first, source_group=SOURCE_GROUP
                )
                second_entry = _export_loop_campaign(
                    run_dir, results, second, source_group=SOURCE_GROUP
                )

                self.assertEqual(first_entry, second_entry)
                self.assertEqual(_bundle_snapshot(first), _bundle_snapshot(second))
                rendered = b"".join(_bundle_snapshot(first).values())
                self.assertNotIn(RAW_SENTINEL.encode(), rendered)
                self.assertNotIn(RAW_HOST_PATH.encode(), rendered)
                self.assertNotIn(b"prompt.json", rendered)
                self.assertNotIn(b"runtime.log", rendered)
                _verify_simple_bundle(
                    first,
                    first_entry,
                    category="campaigns",
                    identity_key="campaign_id",
                )
                bundle = first / "campaigns" / run_dir.name
                manifest = json.loads((bundle / "manifest.json").read_text())
                summary = json.loads((bundle / "summary.json").read_text())
                outcomes = json.loads((bundle / "outcomes.json").read_text())
                self.assertEqual(first_entry["evidence_kind"], LOOP_EVIDENCE_KIND)
                self.assertEqual(summary["aggregates"]["status"], "planned")
                self.assertEqual(outcomes["outcome_count"], 2)
                self.assertTrue(
                    all(row["outcome"] == "not_started" for row in outcomes["outcomes"])
                )
                if variant == "legacy_v1":
                    self.assertNotIn("reasoning_control", manifest["protocol"]["rlm"])
                    self.assertNotIn("reasoning_effort", manifest["protocol"]["halo"])
                elif variant == "legacy_v2":
                    self.assertIsNone(
                        manifest["protocol"]["rlm_compaction_admission"]
                    )
                else:
                    self.assertIsNotNone(
                        manifest["protocol"]["rlm_compaction_admission"]
                    )

    def test_pre_omission_loop_model_schemas_remain_exportable(self) -> None:
        for include_cache_fields in (False, True):
            with (
                self.subTest(include_cache_fields=include_cache_fields),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                results = root / "results"
                plan = _fixture_plan()
                for model in plan["models"].values():
                    model.pop("sglang_ple_omitted")
                    if not include_cache_fields:
                        for field in (
                            "sglang_ple_cache_marker_digest",
                            "sglang_ple_cache_mode",
                            "sglang_ple_cache_payload_digest",
                        ):
                            model.pop(field)
                _rehash_plan(plan)
                run_dir = _materialize_source(results, plan)

                entry = _export_loop_campaign(
                    run_dir,
                    results,
                    root / "evidence",
                    source_group=SOURCE_GROUP,
                )

                self.assertEqual(entry["status"], "planned")

    def test_ple_study_recipe_revision_survives_loop_projection(self) -> None:
        profiles = load_models(MODELS)
        profile_ids = (
            "qwen38-flash-next-nvfp4-mtp3-c8-lazy-ple-mapped-sglang",
            "qwen38-flash-next-nvfp4-mtp3-c8-lazy-ple-omitted-sglang",
        )
        for profile_id in profile_ids:
            with self.subTest(profile_id=profile_id):
                projected = _project_loop_model(
                    _model_record(profiles[profile_id]), profile_id=profile_id
                )
                self.assertEqual(
                    "bf2b7c75870d3703730b6bd8f3bb93dc622c278d",
                    projected["recipe_revision"],
                )
                _validate_projected_sglang_provenance(projected, projected)

                projected.pop("recipe_revision")
                with self.assertRaisesRegex(
                    EvidenceError, "recipe identity changed"
                ):
                    _validate_projected_sglang_provenance(projected, projected)

    def test_sglang_ple_and_overlay_provenance_survives_loop_projection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            plan = _fixture_plan()
            profile_id, overlays = _add_readonly_sglang_provenance(plan, results)
            run_dir = _materialize_source(results, plan)
            output = root / "evidence"

            entry = _export_loop_campaign(
                run_dir, results, output, source_group=SOURCE_GROUP
            )

            bundle = output / "campaigns" / run_dir.name
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            model = next(
                value for value in manifest["models"] if value["id"] == profile_id
            )
            self.assertEqual(2, model["sglang_provenance_version"])
            self.assertIs(model["sglang_ple_omitted"], False)
            self.assertIs(model["sglang_ple_mmap"], True)
            self.assertEqual("readonly", model["sglang_ple_cache_mode"])
            self.assertEqual("c" * 64, model["sglang_ple_cache_marker_sha256"])
            self.assertEqual("d" * 64, model["sglang_ple_cache_payload_sha256"])
            self.assertEqual(
                [
                    {
                        "sha256": overlay["digest"].removeprefix("sha256:"),
                        "target": Path(overlay["host_path"]).name,
                    }
                    for overlay in sorted(
                        overlays,
                        key=lambda value: Path(value["host_path"]).name,
                    )
                ],
                model["sglang_source_overlay_artifacts"],
            )
            rendered = manifest_path.read_text(encoding="utf-8")
            self.assertNotIn("runtime-overlays", rendered)
            self.assertNotIn("sgl-workspace", rendered)
            _verify_simple_bundle(
                output,
                entry,
                category="campaigns",
                identity_key="campaign_id",
            )

            for key in tuple(model):
                if key.startswith("sglang_"):
                    del model[key]
            _write_json(manifest_path, manifest)
            _refresh_bundle_checksums(output, entry)
            with self.assertRaisesRegex(EvidenceError, "provenance is required"):
                _verify_simple_bundle(
                    output,
                    entry,
                    category="campaigns",
                    identity_key="campaign_id",
                )

    def test_overlay_tree_admits_loop_declared_runtime_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            plan = _fixture_plan()
            _add_readonly_sglang_provenance(plan, results)
            run_dir = _materialize_source(results, plan)

            self.assertTrue(
                _validate_runtime_overlay_tree(
                    results,
                    [],
                    [run_dir],
                )
            )

            overlay_dir = (
                results / "runtime-overlays" / "synthetic-loop-sglang"
            )
            (overlay_dir / "undeclared.py").write_text(
                "UNDECLARED = True\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(EvidenceError, "file set changed"):
                _validate_runtime_overlay_tree(
                    results,
                    [],
                    [run_dir],
                )

    def test_completed_projection_is_scalar_verifiable_and_records_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            run_dir = _materialize_source(results, _fixture_plan(), lifecycle=True)
            output = root / "evidence"

            entry = _export_loop_campaign(
                run_dir, results, output, source_group=SOURCE_GROUP
            )
            _verify_simple_bundle(
                output, entry, category="campaigns", identity_key="campaign_id"
            )

            bundle = output / "campaigns" / run_dir.name
            rendered = b"".join(_bundle_snapshot(output).values())
            measurements = json.loads((bundle / "measurements.json").read_text())
            outcomes = json.loads((bundle / "outcomes.json").read_text())
            telemetry = json.loads((bundle / "telemetry.json").read_text())
            self.assertNotIn(RAW_SENTINEL.encode(), rendered)
            self.assertNotIn(RAW_HOST_PATH.encode(), rendered)
            self.assertNotIn(b"prompt.json", rendered)
            self.assertNotIn(b"runtime.log", rendered)
            self.assertEqual(entry["status"], "complete")
            self.assertEqual(measurements["measurement_count"], 2)
            self.assertEqual([row["sample_index"] for row in measurements["measurements"]], [1, 2])
            self.assertEqual(
                [row["outcome"] for row in outcomes["outcomes"]],
                ["complete", "complete"],
            )
            self.assertEqual(telemetry["sample_count"], 3)
            self.assertEqual(telemetry["segment_count"], 2)

    def test_source_plan_hash_case_identity_and_unknown_fields_fail_closed(self) -> None:
        mutations = {
            "integrity": lambda plan: plan.__setitem__("description", "changed"),
            "case identity": lambda plan: (
                plan["cases"][0].__setitem__("case_id", "rlm-0000000000000000"),
                _rehash_plan(plan),
            ),
            "unknown field": lambda plan: (
                plan.__setitem__("raw_prompt", RAW_SENTINEL),
                _rehash_plan(plan),
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                plan = _fixture_plan()
                mutate(plan)
                run_dir = _materialize_source(root / "results", plan)
                with self.assertRaises(EvidenceError):
                    _export_loop_campaign(
                        run_dir,
                        root / "results",
                        root / "evidence",
                        source_group=SOURCE_GROUP,
                    )

    def test_unknown_journal_field_and_foreign_source_file_fail_closed(self) -> None:
        for mutation in ("journal", "layout"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                results = root / "results"
                run_dir = _materialize_source(results, _fixture_plan(), lifecycle=True)
                if mutation == "journal":
                    events = [
                        json.loads(line)
                        for line in (run_dir / "journal.jsonl").read_text().splitlines()
                    ]
                    events[0]["raw_prompt"] = RAW_SENTINEL
                    _write_jsonl(run_dir / "journal.jsonl", events)
                else:
                    (run_dir / "trace.json").write_text(RAW_SENTINEL, encoding="utf-8")
                with self.assertRaises(EvidenceError):
                    _export_loop_campaign(
                        run_dir,
                        results,
                        root / "evidence",
                        source_group=SOURCE_GROUP,
                    )

    def test_semantic_tampering_fails_after_checksums_are_refreshed(self) -> None:
        mutators = {
            "manifest unknown field": lambda bundle: _mutate_document(
                bundle / "manifest.json", lambda value: value.__setitem__("trace", "synthetic")
            ),
            "measurement case binding": lambda bundle: _mutate_document(
                bundle / "measurements.json",
                lambda value: value["measurements"][0].__setitem__(
                    "case_id", value["measurements"][1]["case_id"]
                ),
            ),
            "summary aggregate": lambda bundle: _mutate_document(
                bundle / "summary.json",
                lambda value: value["aggregates"]["groups"][0].__setitem__(
                    "effective_generation_tps", 999.0
                ),
            ),
            "outcome identity": lambda bundle: _mutate_document(
                bundle / "outcomes.json",
                lambda value: value["outcomes"][0].__setitem__(
                    "case_id", "rlm-0000000000000000"
                ),
            ),
            "foreign bundle file": lambda bundle: (bundle / "trace.json").write_text(
                "synthetic trace", encoding="utf-8"
            ),
        }
        for name, mutate in mutators.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                results = root / "results"
                run_dir = _materialize_source(results, _fixture_plan(), lifecycle=True)
                output = root / "evidence"
                entry = _export_loop_campaign(
                    run_dir, results, output, source_group=SOURCE_GROUP
                )
                bundle = output / "campaigns" / run_dir.name
                mutate(bundle)
                _refresh_bundle_checksums(output, entry)
                with self.assertRaises(EvidenceError):
                    _verify_simple_bundle(
                        output,
                        entry,
                        category="campaigns",
                        identity_key="campaign_id",
                    )


def _mutate_document(
    path: Path, mutate: Callable[[dict[str, object]], object]
) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    _write_json(path, value)


if __name__ == "__main__":
    unittest.main()
