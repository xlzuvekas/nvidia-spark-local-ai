from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import sparkbench


class PiCorePrefixCliTests(unittest.TestCase):
    """CLI contracts that never touch a lockfile or npm content store."""

    def test_parser_registers_the_offline_closure_commands(self) -> None:
        source_lock = Path("synthetic-candidate.package-lock.json")
        freeze = sparkbench.build_parser().parse_args(
            ["pi-core-closure-freeze", "--source-lock", str(source_lock)]
        )
        audit = sparkbench.build_parser().parse_args(["pi-core-closure-audit"])
        prefix_parent = Path("/synthetic/private-prefix-parent")
        materialize = sparkbench.build_parser().parse_args(
            ["pi-core-prefix-materialize", "--prefix-parent", str(prefix_parent)]
        )
        prefix = Path("/synthetic/retained-pi-prefix")
        admit = sparkbench.build_parser().parse_args(
            ["pi-core-prefix-admit", "--prefix", str(prefix)]
        )

        self.assertIs(freeze.function, sparkbench.command_pi_core_closure_freeze)
        self.assertEqual(freeze.source_lock, source_lock)
        self.assertIs(audit.function, sparkbench.command_pi_core_closure_audit)
        self.assertEqual(
            audit.cache_sha512_content_store,
            sparkbench.DEFAULT_NPM_SHA512_CONTENT_STORE,
        )
        self.assertIs(
            materialize.function, sparkbench.command_pi_core_prefix_materialize
        )
        self.assertEqual(materialize.prefix_parent, prefix_parent)
        self.assertEqual(
            materialize.cache_sha512_content_store,
            sparkbench.DEFAULT_NPM_SHA512_CONTENT_STORE,
        )
        self.assertIs(admit.function, sparkbench.command_pi_core_prefix_admit)
        self.assertEqual(admit.prefix, prefix)

    def test_freeze_writes_a_missing_fixed_output_target_and_prints_scalars(self) -> None:
        source_lock = Path("synthetic-candidate.package-lock.json")
        args = sparkbench.build_parser().parse_args(
            ["pi-core-closure-freeze", "--source-lock", str(source_lock)]
        )
        expected = {
            "protocol": "synthetic-pi-core-v1",
            "status": "frozen",
            "frozen_lock_sha256": "sha256:" + "a" * 64,
            "package_count": 4,
            "integrity_count": 4,
            "unique_artifact_count": 3,
            "install_script_package_count": 1,
            "optional_package_count": 1,
            "origin_lock_sha256": "sha256:" + "b" * 64,
        }
        summary = Mock()
        summary.scalar.return_value = expected
        target = Mock(spec=Path)
        target.exists.return_value = False
        target.is_symlink.return_value = False
        output = io.StringIO()
        with (
            patch("sparkbench.freeze_pinned_pi_core_lock", return_value=summary) as freeze,
            patch("sparkbench.write_new_frozen_pi_core_lock") as write,
            patch(
                "sparkbench.load_frozen_pi_core_lock",
                side_effect=sparkbench.PiCorePrefixError("synthetic missing closure"),
            ) as load,
            patch("sparkbench.audit_pi_core_cache") as audit,
            patch.object(sparkbench, "DEFAULT_PI_CORE_CLOSURE_LOCK", target),
            patch.object(
                sparkbench, "PI_CORE_CANDIDATE_LOCK_SHA256", expected["origin_lock_sha256"]
            ),
            redirect_stdout(output),
        ):
            status = args.function(args)

        self.assertEqual(status, 0)
        freeze.assert_called_once_with(source_lock)
        load.assert_called_once_with(target)
        target.exists.assert_called_once_with()
        target.is_symlink.assert_called_once_with()
        write.assert_called_once_with(target, summary)
        summary.scalar.assert_called_once_with(
            status="frozen", origin_lock_sha256=expected["origin_lock_sha256"]
        )
        audit.assert_not_called()
        self.assertEqual(json.loads(output.getvalue()), expected)
        self.assertNotIn(str(source_lock), output.getvalue())
        self.assertNotIn(str(target), output.getvalue())

    def test_freeze_reuses_a_matching_fixed_output_target_without_writing(self) -> None:
        source_lock = Path("synthetic-candidate.package-lock.json")
        args = sparkbench.build_parser().parse_args(
            ["pi-core-closure-freeze", "--source-lock", str(source_lock)]
        )
        expected = {
            "protocol": "synthetic-pi-core-v1",
            "status": "already_frozen",
            "frozen_lock_sha256": "sha256:" + "a" * 64,
            "package_count": 4,
            "integrity_count": 4,
            "unique_artifact_count": 3,
            "install_script_package_count": 1,
            "optional_package_count": 1,
            "origin_lock_sha256": "sha256:" + "b" * 64,
        }
        summary = Mock()
        summary.frozen_lock_sha256 = expected["frozen_lock_sha256"]
        summary.scalar.return_value = expected
        existing = Mock()
        existing.frozen_lock_sha256 = expected["frozen_lock_sha256"]
        target = Mock(spec=Path)
        output = io.StringIO()
        with (
            patch("sparkbench.freeze_pinned_pi_core_lock", return_value=summary) as freeze,
            patch("sparkbench.write_new_frozen_pi_core_lock") as write,
            patch("sparkbench.load_frozen_pi_core_lock", return_value=existing) as load,
            patch("sparkbench.audit_pi_core_cache") as audit,
            patch.object(sparkbench, "DEFAULT_PI_CORE_CLOSURE_LOCK", target),
            patch.object(
                sparkbench, "PI_CORE_CANDIDATE_LOCK_SHA256", expected["origin_lock_sha256"]
            ),
            redirect_stdout(output),
        ):
            status = args.function(args)

        self.assertEqual(status, 0)
        freeze.assert_called_once_with(source_lock)
        load.assert_called_once_with(target)
        write.assert_not_called()
        audit.assert_not_called()
        summary.scalar.assert_called_once_with(
            status="already_frozen", origin_lock_sha256=expected["origin_lock_sha256"]
        )
        self.assertEqual(json.loads(output.getvalue()), expected)
        self.assertNotIn(str(source_lock), output.getvalue())
        self.assertNotIn(str(target), output.getvalue())

    def test_audit_uses_the_fixed_frozen_lock_and_prints_only_scalars(self) -> None:
        cache_root = Path("synthetic-cache")
        args = sparkbench.build_parser().parse_args(
            ["pi-core-closure-audit", "--cache-sha512-content-store", str(cache_root)]
        )
        summary = Mock()
        expected = {
            "protocol": "synthetic-pi-core-v1",
            "status": "cache_complete",
            "frozen_lock_sha256": "sha256:" + "c" * 64,
            "package_count": 4,
            "artifact_count": 3,
            "artifact_size_bytes": 1234,
        }
        audit_result = Mock()
        audit_result.scalar.return_value = expected
        output = io.StringIO()
        with (
            patch("sparkbench.freeze_pinned_pi_core_lock") as freeze,
            patch("sparkbench.write_new_frozen_pi_core_lock") as write,
            patch("sparkbench.load_frozen_pi_core_lock", return_value=summary) as load,
            patch("sparkbench.audit_pi_core_cache", return_value=audit_result) as audit,
            redirect_stdout(output),
        ):
            status = args.function(args)

        self.assertEqual(status, 0)
        load.assert_called_once_with(sparkbench.DEFAULT_PI_CORE_CLOSURE_LOCK)
        audit.assert_called_once_with(summary, cache_sha512_root=cache_root)
        audit_result.scalar.assert_called_once_with()
        freeze.assert_not_called()
        write.assert_not_called()
        self.assertEqual(json.loads(output.getvalue()), expected)
        self.assertNotIn(str(cache_root), output.getvalue())
        self.assertNotIn(str(sparkbench.DEFAULT_PI_CORE_CLOSURE_LOCK), output.getvalue())

    def test_materialize_uses_the_fixed_closure_and_prints_only_scalars(self) -> None:
        cache_root = Path("synthetic-cache")
        prefix_parent = Path("/synthetic/private-prefix-parent")
        args = sparkbench.build_parser().parse_args(
            [
                "pi-core-prefix-materialize",
                "--prefix-parent",
                str(prefix_parent),
                "--cache-sha512-content-store",
                str(cache_root),
            ]
        )
        summary = Mock()
        expected = {
            "protocol": "synthetic-pi-prefix-v1",
            "status": "materialized",
            "frozen_lock_sha256": "sha256:" + "d" * 64,
            "package_count": 4,
            "artifact_count": 3,
            "artifact_size_bytes": 1234,
            "tree_digest": "sha256:" + "e" * 64,
            "tree_entries": 12,
            "tree_files": 8,
            "tree_size_bytes": 5678,
            "prefix_directory_name": "synthetic-prefix",
        }
        result = Mock()
        result.scalar.return_value = expected
        output = io.StringIO()
        with (
            patch("sparkbench.freeze_pinned_pi_core_lock") as freeze,
            patch("sparkbench.write_new_frozen_pi_core_lock") as write,
            patch("sparkbench.audit_pi_core_cache") as audit,
            patch("sparkbench.load_frozen_pi_core_lock", return_value=summary) as load,
            patch("sparkbench.materialize_pi_core_prefix", return_value=result) as materialize,
            redirect_stdout(output),
        ):
            status = args.function(args)

        self.assertEqual(status, 0)
        load.assert_called_once_with(sparkbench.DEFAULT_PI_CORE_CLOSURE_LOCK)
        materialize.assert_called_once_with(
            summary,
            cache_sha512_root=cache_root,
            prefix_parent=prefix_parent,
            repo_root=sparkbench.WORKSPACE,
        )
        result.scalar.assert_called_once_with()
        freeze.assert_not_called()
        write.assert_not_called()
        audit.assert_not_called()
        self.assertEqual(json.loads(output.getvalue()), expected)
        self.assertNotIn(str(cache_root), output.getvalue())
        self.assertNotIn(str(prefix_parent), output.getvalue())

    def test_admit_uses_fixed_policy_and_prints_only_scalars(self) -> None:
        prefix = Path("/synthetic/retained-pi-prefix")
        args = sparkbench.build_parser().parse_args(
            ["pi-core-prefix-admit", "--prefix", str(prefix)]
        )
        summary = Mock()
        summary.frozen_lock_sha256 = "sha256:" + "a" * 64
        pin = Mock()
        expected = {
            "protocol": "synthetic-pi-prefix-admission-v1",
            "status": "admitted",
            "frozen_lock_sha256": summary.frozen_lock_sha256,
            "tree_protocol": "synthetic-tree-v1",
            "tree_digest": "sha256:" + "b" * 64,
            "tree_entries": 12,
            "tree_files": 8,
            "tree_links": 0,
            "tree_size_bytes": 5678,
            "entrypoint_digest": "sha256:" + "c" * 64,
            "entrypoint_size_bytes": 210,
            "entrypoint_mode": "0444",
        }
        result = Mock()
        result.scalar.return_value = expected
        output = io.StringIO()
        with (
            patch("sparkbench.load_frozen_pi_core_lock", return_value=summary) as load,
            patch(
                "sparkbench.load_pi_core_prefix_admission_pin", return_value=pin
            ) as load_pin,
            patch("sparkbench.admit_pi_core_prefix", return_value=result) as admit,
            redirect_stdout(output),
        ):
            status = args.function(args)

        self.assertEqual(status, 0)
        load.assert_called_once_with(sparkbench.DEFAULT_PI_CORE_CLOSURE_LOCK)
        load_pin.assert_called_once_with(sparkbench.DEFAULT_PI_CORE_PREFIX_ADMISSION_PIN)
        admit.assert_called_once_with(
            prefix,
            repo_root=sparkbench.WORKSPACE,
            frozen_lock_sha256=summary.frozen_lock_sha256,
            pin=pin,
        )
        result.scalar.assert_called_once_with()
        self.assertEqual(json.loads(output.getvalue()), expected)
        self.assertNotIn(str(prefix), output.getvalue())
        self.assertNotIn(str(sparkbench.DEFAULT_PI_CORE_CLOSURE_LOCK), output.getvalue())
        self.assertNotIn(
            str(sparkbench.DEFAULT_PI_CORE_PREFIX_ADMISSION_PIN), output.getvalue()
        )


if __name__ == "__main__":
    unittest.main()
