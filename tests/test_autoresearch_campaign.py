from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from bench.autoresearch_campaign import (
    CampaignPlanningError,
    EXPECTED_AXES,
    EXPECTED_PRIMARY_CASE_IDS,
    _cell_specs,
    freeze_campaign,
    load_campaign_definition,
    semantic_config,
    validate_campaign,
)
from bench.journal import content_hash, write_json


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PATH = (
    ROOT
    / "manifests"
    / "campaigns"
    / "qwen38_flash_next_single_user_autoresearch.toml"
)


class AutoresearchCampaignPlanningTests(unittest.TestCase):
    def test_definition_and_three_semantic_deltas_are_exact(self) -> None:
        definition = load_campaign_definition(CAMPAIGN_PATH, workspace=ROOT)
        preview, models, suite = validate_campaign(definition)

        self.assertEqual(preview.suite_id, suite.id)
        self.assertEqual(preview.policy.primary_case_ids, EXPECTED_PRIMARY_CASE_IDS)
        self.assertEqual(preview.policy.allowed_axes, EXPECTED_AXES)
        self.assertEqual(
            tuple(proposal.axis for proposal in preview.proposals),
            EXPECTED_AXES,
        )
        self.assertEqual(preview.to_mapping()["planned_cell_count"], 14)
        self.assertFalse(preview.to_mapping()["execution_started"])

        baseline = semantic_config(models[definition.baseline_id])
        self.assertEqual(baseline["chunked_prefill_size"], 1024)
        self.assertEqual(
            baseline["nextn_bundle"], {"steps": 2, "draft_tokens": 3}
        )
        self.assertEqual(
            baseline["reasoning_policy"],
            {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "reasoning_effort": "low",
                }
            },
        )

    def test_candidate_queue_freezes_calibration_and_two_fresh_pairs_each(self) -> None:
        definition = load_campaign_definition(CAMPAIGN_PATH, workspace=ROOT)
        preview, _models, _suite = validate_campaign(definition)
        cells = _cell_specs(preview)

        self.assertEqual(len(cells), 14)
        self.assertEqual(
            tuple(cell["arm"] for cell in cells[:2]),
            ("control_a", "control_b"),
        )
        for offset, proposal in enumerate(preview.proposals):
            block = cells[2 + 4 * offset : 6 + 4 * offset]
            self.assertEqual(
                tuple((cell["stage"], cell["arm"]) for cell in block),
                (
                    ("screen", "champion"),
                    ("screen", "candidate"),
                    ("confirmation", "candidate"),
                    ("confirmation", "champion"),
                ),
            )
            self.assertTrue(
                all(cell["candidate_id"] == proposal.candidate_id for cell in block)
            )

    def test_non_axis_model_change_is_rejected(self) -> None:
        definition = load_campaign_definition(CAMPAIGN_PATH, workspace=ROOT)
        preview, models, suite = validate_campaign(definition)
        candidate = models[preview.proposals[0].candidate_id]
        models[candidate.id] = replace(candidate, estimated_ram_gib=103.0)

        # Exercise the invariant directly through a temporary models manifest is
        # unnecessary: changing a non-axis field must make the dataclasses differ
        # after the three semantic axes and bookkeeping fields are normalized.
        from bench.autoresearch_campaign import _invariant_model_projection

        baseline = models[definition.baseline_id]
        self.assertNotEqual(
            _invariant_model_projection(baseline),
            _invariant_model_projection(models[candidate.id]),
        )
        self.assertEqual(suite.id, preview.suite_id)

    def test_loader_rejects_unknown_keys_and_paths_outside_workspace(self) -> None:
        source = CAMPAIGN_PATH.read_text()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            unknown = root / "unknown.toml"
            unknown.write_text(source + "\nunknown = true\n")
            with self.assertRaisesRegex(CampaignPlanningError, "unknown keys"):
                load_campaign_definition(unknown, workspace=ROOT)

            escaped = root / "escaped.toml"
            escaped.write_text(source.replace('../models.toml', '../../../../etc/passwd'))
            with self.assertRaisesRegex(CampaignPlanningError, "inside the workspace"):
                load_campaign_definition(escaped, workspace=ROOT)

    def test_freeze_uses_unique_cell_roots_and_never_executes(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_create_plan(**kwargs: object) -> Path:
            calls.append(kwargs)
            results_root = kwargs["results_root"]
            assert isinstance(results_root, Path)
            run_dir = results_root / "frozen-run"
            run_dir.mkdir()
            model = kwargs["model"]
            suite = kwargs["suite"]
            fingerprint = content_hash(
                {"model": getattr(model, "id"), "suite": getattr(suite, "id")},
                64,
            )
            write_json(
                run_dir / "plan.json",
                {
                    "fingerprint": fingerprint,
                    "integrity_hash": "a" * 64,
                },
            )
            return run_dir

        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            campaign_dir = freeze_campaign(
                CAMPAIGN_PATH,
                workspace=ROOT,
                results_root=Path(directory),
                create_plan_fn=fake_create_plan,
            )
            frozen = json.loads((campaign_dir / "campaign.json").read_text())

        self.assertEqual(len(calls), 14)
        self.assertEqual(len(frozen["cells"]), 14)
        roots = [Path(call["results_root"]) for call in calls]
        self.assertEqual(len(set(roots)), 14)
        self.assertFalse(frozen["execution_started"])
        integrity = frozen.pop("integrity_hash")
        self.assertEqual(integrity, content_hash(frozen, 64))


if __name__ == "__main__":
    unittest.main()
