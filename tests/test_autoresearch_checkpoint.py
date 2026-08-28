from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from bench.autoresearch_checkpoint import (
    ACK_SCHEMA_VERSION,
    CampaignBinding,
    CheckpointAcknowledgement,
    CheckpointError,
    EvidenceCell,
    EvidenceProof,
    PairCompletion,
    RepositoryProof,
    acknowledge_checkpoint,
    autoresearch_published_run_id,
    checkpoint_gate,
    checkpoint_state_path,
    load_acknowledgement,
    prove_evidence,
    prove_repository,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
OID_A = "a" * 40
OID_B = "b" * 40


def _campaign() -> CampaignBinding:
    return CampaignBinding(
        campaign_id="synthetic-autoresearch",
        campaign_integrity_sha256=SHA_A,
        preview_sha256=SHA_B,
        policy_sha256=SHA_C,
    )


def _completion(
    *, sequence: int = 1, observation: str = SHA_D
) -> PairCompletion:
    if sequence == 1:
        return PairCompletion(
            sequence=1,
            pair_kind="calibration",
            candidate_id="control",
            search_pair_index=None,
            ordered_cell_ids=("calibration-control-a", "calibration-control-b"),
            ordered_evidence_run_ids=("published-control-a", "published-control-b"),
            cell_plan_integrity_sha256s=(SHA_A, SHA_B),
            observation_sha256=observation,
        )
    return PairCompletion(
        sequence=sequence,
        pair_kind="screen" if sequence == 2 else "confirmation",
        candidate_id="candidate-a",
        search_pair_index=sequence - 2,
        ordered_cell_ids=("candidate-a-champion", "candidate-a-candidate"),
        ordered_evidence_run_ids=("published-champion", "published-candidate"),
        cell_plan_integrity_sha256s=(SHA_A, SHA_B),
        observation_sha256=observation,
    )


def _evidence(completion: PairCompletion, *, corpus: str = SHA_A) -> EvidenceProof:
    cells = tuple(
        EvidenceCell(
            cell_id=cell_id,
            published_run_id=run_id,
            bundle_sha256=SHA_B if index == 0 else SHA_C,
            status="complete",
            measurement_terminal=True,
        )
        for index, (cell_id, run_id) in enumerate(
            zip(
                completion.ordered_cell_ids,
                completion.ordered_evidence_run_ids,
                strict=True,
            )
        )
    )
    return EvidenceProof(
        index_sha256=corpus,
        checksums_sha256=SHA_D,
        cells=cells,
    )


def _repository(*, commit: str = OID_A) -> RepositoryProof:
    return RepositoryProof(
        head_commit=commit,
        local_branch_ref="refs/heads/main",
        upstream_ref="refs/remotes/origin/main",
        upstream_commit=commit,
        remote_name="origin",
        remote_ref="refs/heads/main",
        remote_commit=commit,
    )


def _write_journal(path: Path, *, value: str = "campaign-started") -> None:
    path.write_text(
        json.dumps(
            {
                "event": "autoresearch_campaign_started",
                "transition_id": value,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


class AutoresearchCheckpointTests(unittest.TestCase):
    def _acknowledge(
        self,
        workspace: Path,
        completion: PairCompletion,
        *,
        evidence: EvidenceProof | None = None,
        repository: RepositoryProof | None = None,
        completion_reader=None,
    ) -> CheckpointAcknowledgement:
        journal = workspace / "events.jsonl"
        if not journal.exists():
            _write_journal(journal)
        evidence = evidence or _evidence(completion)
        repository = repository or _repository()
        reader = completion_reader or (lambda: completion)
        return acknowledge_checkpoint(
            workspace=workspace,
            campaign=_campaign(),
            journal_path=journal,
            completion_reader=reader,
            evidence_verifier=lambda item: evidence,
            evidence_snapshot_reader=lambda item: evidence,
            repository_verifier=lambda: repository,
            repository_snapshot_reader=lambda: repository,
            now=lambda: datetime(2026, 8, 28, 1, 2, 3, tzinfo=timezone.utc),
        )

    def test_first_pair_is_not_blocked_without_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            gate = checkpoint_gate(
                workspace=Path(directory),
                campaign=_campaign(),
                completion=None,
                journal_path=Path(directory) / "absent.jsonl",
                evidence_reader=lambda _item: self.fail("evidence read before first pair"),
                repository_reader=lambda: self.fail("Git read before first pair"),
            )
        self.assertTrue(gate.ready)
        self.assertEqual(gate.sequence, 0)

    def test_ack_is_private_integrity_bound_and_contains_no_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            completion = _completion()
            acknowledgement = self._acknowledge(workspace, completion)
            path = checkpoint_state_path(workspace, _campaign())
            raw = json.loads(path.read_text())
            loaded = load_acknowledgement(path)

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
            self.assertEqual(loaded, acknowledgement)
            self.assertEqual(raw["schema_version"], ACK_SCHEMA_VERSION)
            self.assertEqual(raw["pair_state_sha256"], completion.digest)
            self.assertNotIn(str(workspace), path.read_text())
            self.assertNotIn("run_dir", path.read_text())
            self.assertNotIn("remote_url", path.read_text())
            self.assertEqual(path.parent, workspace / "logs" / "autoresearch-checkpoints")

    def test_ack_is_idempotent_for_the_same_bound_proofs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            completion = _completion()
            first = self._acknowledge(workspace, completion)
            path = checkpoint_state_path(workspace, _campaign())
            first_bytes = path.read_bytes()
            evidence = _evidence(completion)
            repository = _repository()
            second = acknowledge_checkpoint(
                workspace=workspace,
                campaign=_campaign(),
                journal_path=workspace / "events.jsonl",
                completion_reader=lambda: completion,
                evidence_verifier=lambda _item: evidence,
                evidence_snapshot_reader=lambda _item: evidence,
                repository_verifier=lambda: repository,
                repository_snapshot_reader=lambda: repository,
                now=lambda: datetime(2027, 1, 1, tzinfo=timezone.utc),
            )

            self.assertEqual(second, first)
            self.assertEqual(path.read_bytes(), first_bytes)

    def test_ack_rejects_no_pair_and_a_verification_race(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            journal = workspace / "events.jsonl"
            _write_journal(journal)
            with self.assertRaises(CheckpointError) as missing:
                acknowledge_checkpoint(
                    workspace=workspace,
                    campaign=_campaign(),
                    journal_path=journal,
                    completion_reader=lambda: None,
                    evidence_verifier=lambda _item: self.fail("unexpected evidence read"),
                    repository_verifier=lambda: self.fail("unexpected Git read"),
                )
            self.assertEqual(missing.exception.code, "no_completed_pair")

            before = _completion()
            after = _completion(sequence=2)
            reads = iter((before, after))
            evidence_before = _evidence(before)
            evidence_after = _evidence(after)
            with self.assertRaises(CheckpointError) as raced:
                acknowledge_checkpoint(
                    workspace=workspace,
                    campaign=_campaign(),
                    journal_path=journal,
                    completion_reader=lambda: next(reads),
                    evidence_verifier=lambda _item: evidence_before,
                    evidence_snapshot_reader=lambda _item: evidence_after,
                    repository_verifier=_repository,
                    repository_snapshot_reader=_repository,
                )
            self.assertEqual(raced.exception.code, "checkpoint_race")
            self.assertFalse(checkpoint_state_path(workspace, _campaign()).exists())

    def test_gate_requires_ack_then_accepts_an_appended_journal_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            completion = _completion()
            evidence = _evidence(completion)
            repository = _repository()
            journal = workspace / "events.jsonl"
            _write_journal(journal)
            missing = checkpoint_gate(
                workspace=workspace,
                campaign=_campaign(),
                completion=completion,
                journal_path=journal,
                evidence_reader=lambda _item: evidence,
                repository_reader=lambda: repository,
            )
            self.assertEqual(missing.reason, "missing")

            self._acknowledge(
                workspace, completion, evidence=evidence, repository=repository
            )
            with journal.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "event": "autoresearch_candidate_started",
                            "transition_id": "candidate-started",
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
            gate = checkpoint_gate(
                workspace=workspace,
                campaign=_campaign(),
                completion=completion,
                journal_path=journal,
                evidence_reader=lambda _item: evidence,
                repository_reader=lambda: repository,
            )
            self.assertTrue(gate.ready)

    def test_gate_detects_new_pair_evidence_repository_and_prefix_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            completion = _completion()
            evidence = _evidence(completion)
            repository = _repository()
            self._acknowledge(
                workspace, completion, evidence=evidence, repository=repository
            )
            journal = workspace / "events.jsonl"

            newer = checkpoint_gate(
                workspace=workspace,
                campaign=_campaign(),
                completion=_completion(sequence=2),
                journal_path=journal,
                evidence_reader=lambda _item: self.fail("new pair should short-circuit"),
                repository_reader=lambda: self.fail("new pair should short-circuit"),
            )
            self.assertEqual(newer.reason, "new_pair")

            changed_evidence = checkpoint_gate(
                workspace=workspace,
                campaign=_campaign(),
                completion=completion,
                journal_path=journal,
                evidence_reader=lambda _item: _evidence(completion, corpus=SHA_B),
                repository_reader=lambda: repository,
            )
            self.assertEqual(changed_evidence.reason, "evidence_changed")

            stale_results = checkpoint_gate(
                workspace=workspace,
                campaign=_campaign(),
                completion=completion,
                journal_path=journal,
                evidence_reader=lambda _item: (_ for _ in ()).throw(
                    CheckpointError(
                        "evidence_not_current", "synthetic ignored-results drift"
                    )
                ),
                repository_reader=lambda: repository,
            )
            self.assertEqual(stale_results.reason, "evidence_changed")

            changed_repository = checkpoint_gate(
                workspace=workspace,
                campaign=_campaign(),
                completion=completion,
                journal_path=journal,
                evidence_reader=lambda _item: evidence,
                repository_reader=lambda: _repository(commit=OID_B),
            )
            self.assertEqual(changed_repository.reason, "repository_changed")

            _write_journal(journal, value="rewritten-prefix")
            with self.assertRaises(CheckpointError) as changed_prefix:
                checkpoint_gate(
                    workspace=workspace,
                    campaign=_campaign(),
                    completion=completion,
                    journal_path=journal,
                    evidence_reader=lambda _item: evidence,
                    repository_reader=lambda: repository,
                )
            self.assertEqual(changed_prefix.exception.code, "journal_prefix_changed")

    def test_state_rejects_unknown_fields_bad_mode_links_and_rehashed_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            completion = _completion()
            self._acknowledge(workspace, completion)
            path = checkpoint_state_path(workspace, _campaign())

            os.chmod(path, 0o644)
            with self.assertRaises(CheckpointError) as bad_mode:
                load_acknowledgement(path)
            self.assertEqual(bad_mode.exception.code, "checkpoint_state_invalid")
            os.chmod(path, 0o600)

            value = json.loads(path.read_text())
            value["unknown"] = True
            path.write_text(json.dumps(value))
            os.chmod(path, 0o600)
            with self.assertRaises(CheckpointError) as unknown:
                load_acknowledgement(path)
            self.assertEqual(unknown.exception.code, "checkpoint_state_invalid")

            path.unlink()
            self._acknowledge(workspace, completion)
            value = json.loads(path.read_text())
            value["pair_state_sha256"] = SHA_A
            payload = {key: item for key, item in value.items() if key != "integrity_hash"}
            value["integrity_hash"] = hashlib.sha256(
                json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest()
            path.write_text(json.dumps(value))
            os.chmod(path, 0o600)
            with self.assertRaises(CheckpointError) as rebound:
                load_acknowledgement(path)
            self.assertEqual(rebound.exception.code, "checkpoint_state_invalid")

            path.unlink()
            target = workspace / "target"
            target.write_text("{}")
            os.chmod(target, 0o600)
            path.symlink_to(target)
            with self.assertRaises(CheckpointError) as symlink:
                load_acknowledgement(path)
            self.assertEqual(symlink.exception.code, "checkpoint_state_invalid")
            path.unlink()
            os.link(target, path)
            with self.assertRaises(CheckpointError) as hardlink:
                load_acknowledgement(path)
            self.assertEqual(hardlink.exception.code, "checkpoint_state_invalid")

    def test_pair_and_evidence_schemas_fail_closed(self) -> None:
        with self.assertRaises(CheckpointError):
            PairCompletion(
                sequence=1,
                pair_kind="screen",
                candidate_id="candidate-a",
                search_pair_index=0,
                ordered_cell_ids=("a", "b"),
                ordered_evidence_run_ids=("c", "d"),
                cell_plan_integrity_sha256s=(SHA_A, SHA_B),
                observation_sha256=SHA_C,
            )
        completion = _completion()
        with self.assertRaises(CheckpointError) as incomplete:
            EvidenceCell(
                cell_id=completion.ordered_cell_ids[0],
                published_run_id=completion.ordered_evidence_run_ids[0],
                bundle_sha256=SHA_A,
                status="partial",
                measurement_terminal=True,
            )
        self.assertEqual(incomplete.exception.code, "evidence_incomplete")

    def test_repository_proof_requires_clean_local_and_live_remote_equality(self) -> None:
        outputs = {
            ("status", "--porcelain=v2", "-z", "--untracked-files=all"): b"",
            ("symbolic-ref", "-q", "HEAD"): b"refs/heads/main\n",
            ("rev-parse", "--verify", "HEAD^{commit}"): (OID_A + "\n").encode(),
            ("config", "--get", "branch.main.remote"): b"origin\n",
            ("config", "--get", "branch.main.merge"): b"refs/heads/main\n",
            ("rev-parse", "--symbolic-full-name", "@{upstream}"): b"refs/remotes/origin/main\n",
            ("rev-parse", "--verify", "@{upstream}^{commit}"): (OID_A + "\n").encode(),
            ("ls-remote", "--exit-code", "origin", "refs/heads/main"): (
                OID_A + "\trefs/heads/main\n"
            ).encode(),
        }

        def runner(_workspace: Path, arguments: tuple[str, ...]) -> bytes:
            return outputs[arguments]

        with tempfile.TemporaryDirectory() as directory:
            proof = prove_repository(Path(directory), git_runner=runner)
            self.assertEqual(proof, _repository())

            dirty = dict(outputs)
            dirty[("status", "--porcelain=v2", "-z", "--untracked-files=all")] = b"1 .M file\0"
            with self.assertRaises(CheckpointError) as changed:
                prove_repository(
                    Path(directory),
                    git_runner=lambda _workspace, arguments: dirty[arguments],
                )
            self.assertEqual(changed.exception.code, "repository_dirty")

            remote_changed = dict(outputs)
            remote_changed[("ls-remote", "--exit-code", "origin", "refs/heads/main")] = (
                OID_B + "\trefs/heads/main\n"
            ).encode()
            with self.assertRaises(CheckpointError) as not_pushed:
                prove_repository(
                    Path(directory),
                    git_runner=lambda _workspace, arguments: remote_changed[arguments],
                )
            self.assertEqual(not_pushed.exception.code, "repository_not_pushed")

    def test_published_evidence_ids_accept_exporter_timestamp_case(self) -> None:
        exported_id = autoresearch_published_run_id(
            campaign_id="synthetic-autoresearch",
            cell_id="calibration-control-a",
            ordinal=1,
            created_at="2026-08-28T01:02:03+00:00",
        )
        completion = PairCompletion(
            sequence=1,
            pair_kind="calibration",
            candidate_id="control",
            search_pair_index=None,
            ordered_cell_ids=("calibration-control-a", "calibration-control-b"),
            ordered_evidence_run_ids=(
                exported_id,
                "20260828T010203000000Z-autoresearch-control-b",
            ),
            cell_plan_integrity_sha256s=(SHA_A, SHA_B),
            observation_sha256=SHA_C,
        )
        _evidence(completion).require_completion(completion)

    def test_evidence_proof_requires_unchanged_verified_terminal_bundles(self) -> None:
        completion = _completion()
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            evidence = workspace / "evidence"
            evidence.mkdir()
            runs = [
                {
                    "bundle_sha256": SHA_B if index == 0 else SHA_C,
                    "measurement_terminal": True,
                    "run_id": run_id,
                    "status": "complete",
                }
                for index, run_id in enumerate(completion.ordered_evidence_run_ids)
            ]
            (evidence / "index.json").write_text(json.dumps({"runs": runs}))
            (evidence / "checksums.json").write_text(json.dumps({"files": {}}))
            calls: list[str] = []

            proof = prove_evidence(
                completion,
                workspace=workspace,
                results_root=workspace / "results",
                evidence_root=evidence,
                exporter=lambda **_kwargs: calls.append("export") or {"changed": False},
                verifier=lambda _path: calls.append("verify") or {"status": "verified"},
                staged_verifier=lambda **_kwargs: calls.append("staged")
                or {"status": "staged_verified"},
            )
            proof.require_completion(completion)
            self.assertEqual(calls, ["export", "verify", "staged"])

            with self.assertRaises(CheckpointError) as stale:
                prove_evidence(
                    completion,
                    workspace=workspace,
                    results_root=workspace / "results",
                    evidence_root=evidence,
                    exporter=lambda **_kwargs: {"changed": True},
                    verifier=lambda _path: {"status": "verified"},
                    staged_verifier=lambda **_kwargs: {"status": "staged_verified"},
                )
            self.assertEqual(stale.exception.code, "evidence_not_current")

    def test_evidence_proof_never_creates_a_missing_evidence_tree(self) -> None:
        completion = _completion()
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            evidence = workspace / "evidence"
            calls: list[str] = []

            with self.assertRaises(CheckpointError) as missing:
                prove_evidence(
                    completion,
                    workspace=workspace,
                    results_root=workspace / "results",
                    evidence_root=evidence,
                    exporter=lambda **_kwargs: calls.append("export")
                    or {"changed": False},
                    verifier=lambda _path: calls.append("verify")
                    or {"status": "verified"},
                    staged_verifier=lambda **_kwargs: calls.append("staged")
                    or {"status": "staged_verified"},
                )

            self.assertEqual(missing.exception.code, "evidence_not_current")
            self.assertEqual(calls, [])
            self.assertFalse(evidence.exists())


if __name__ == "__main__":
    unittest.main()
