from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from bench.harbor_runtime_assets import TREE_PROTOCOL, inspect_normalized_tree
from bench.pi_core_prefix import (
    PI_CORE_FROZEN_LOCK_SHA256,
    PI_CORE_PREFIX_DIRECTORY_NAME,
)
from bench.pi_core_prefix_admission import (
    PI_CORE_PREFIX_ADMISSION_PROTOCOL,
    PI_CORE_PREFIX_ADMISSION_SCHEMA_VERSION,
    PI_CORE_PREFIX_ENTRYPOINT,
    PiCorePrefixAdmissionError,
    PiCorePrefixAdmissionPin,
    admit_pi_core_prefix,
    load_pi_core_prefix_admission_pin,
)


ROOT = Path(__file__).resolve().parents[1]


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


class PiCorePrefixAdmissionTests(unittest.TestCase):
    """Synthetic read-only prefixes exercise no npm, Node, Pi, or runtime."""

    @staticmethod
    def _make_writable(root: Path) -> None:
        for current, _directories, files in os.walk(root, topdown=False):
            directory = Path(current)
            os.chmod(directory, 0o700)
            for name in files:
                os.chmod(directory / name, 0o600)

    @staticmethod
    def _normalize(root: Path) -> None:
        for current, _directories, files in os.walk(root, topdown=False):
            directory = Path(current)
            for name in files:
                os.chmod(directory / name, 0o444)
            os.chmod(directory, 0o555)

    def _prefix(self, parent: Path, *, name: str = PI_CORE_PREFIX_DIRECTORY_NAME) -> Path:
        parent.chmod(0o700)
        prefix = parent / name
        entrypoint = prefix / PI_CORE_PREFIX_ENTRYPOINT
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_bytes(b"export { Agent } from './agent.js';\n")
        extra = prefix / "node_modules" / "@mariozechner" / "pi-ai" / "dist" / "index.js"
        extra.parent.mkdir(parents=True)
        extra.write_bytes(b"export const synthetic = true;\n")
        self._normalize(prefix)
        return prefix

    def _pin(self, prefix: Path) -> PiCorePrefixAdmissionPin:
        tree = inspect_normalized_tree(prefix, repo_root=ROOT)
        entrypoint = prefix / PI_CORE_PREFIX_ENTRYPOINT
        payload = entrypoint.read_bytes()
        return PiCorePrefixAdmissionPin(
            protocol=PI_CORE_PREFIX_ADMISSION_PROTOCOL,
            frozen_lock_sha256=PI_CORE_FROZEN_LOCK_SHA256,
            prefix_directory_name=PI_CORE_PREFIX_DIRECTORY_NAME,
            tree_protocol=TREE_PROTOCOL,
            tree_digest="sha256:" + tree.digest,
            tree_entries=tree.entries,
            tree_files=tree.files,
            tree_links=tree.links,
            tree_size_bytes=tree.size_bytes,
            entrypoint_relative_path=PI_CORE_PREFIX_ENTRYPOINT,
            entrypoint_digest="sha256:" + hashlib.sha256(payload).hexdigest(),
            entrypoint_size_bytes=len(payload),
            entrypoint_mode=0o444,
        )

    @staticmethod
    def _pin_document(pin: PiCorePrefixAdmissionPin) -> dict[str, object]:
        return {
            "schema_version": PI_CORE_PREFIX_ADMISSION_SCHEMA_VERSION,
            "protocol": pin.protocol,
            "frozen_lock_sha256": pin.frozen_lock_sha256,
            "prefix_directory_name": pin.prefix_directory_name,
            "tree": {
                "protocol": pin.tree_protocol,
                "digest": pin.tree_digest,
                "entries": pin.tree_entries,
                "files": pin.tree_files,
                "links": pin.tree_links,
                "size_bytes": pin.tree_size_bytes,
            },
            "entrypoint": {
                "relative_path": pin.entrypoint_relative_path,
                "digest": pin.entrypoint_digest,
                "size_bytes": pin.entrypoint_size_bytes,
                "mode": pin.entrypoint_mode,
            },
        }

    @staticmethod
    def _write_pin(path: Path, document: object) -> None:
        path.write_bytes(_canonical_json_bytes(document))
        path.chmod(0o600)

    def test_tracked_pin_is_canonical_and_binds_the_observed_smoke_identity(self) -> None:
        pin = load_pi_core_prefix_admission_pin(
            ROOT / "manifests" / "prefixes" / "pi-core-0.57.1.admission.json"
        )
        self.assertEqual(pin.protocol, PI_CORE_PREFIX_ADMISSION_PROTOCOL)
        self.assertEqual(pin.frozen_lock_sha256, PI_CORE_FROZEN_LOCK_SHA256)
        self.assertEqual(pin.prefix_directory_name, PI_CORE_PREFIX_DIRECTORY_NAME)
        self.assertEqual(pin.tree_protocol, TREE_PROTOCOL)
        self.assertEqual(
            pin.tree_digest,
            "sha256:aebaccc9fa0c58d9ef15a8b718b08f700d2564cbcc31b518c492a6e993964ac8",
        )
        self.assertEqual((pin.tree_entries, pin.tree_files, pin.tree_links), (15563, 13828, 0))
        self.assertEqual(pin.tree_size_bytes, 75042106)
        self.assertEqual(pin.entrypoint_relative_path, PI_CORE_PREFIX_ENTRYPOINT)
        self.assertEqual(
            pin.entrypoint_digest,
            "sha256:3cb4d4c12c9f19b9113c10ad7d3451837a5d1b92040747c7e1251f1f41ac3687",
        )
        self.assertEqual((pin.entrypoint_size_bytes, pin.entrypoint_mode), (210, 0o444))

    def test_admits_an_exact_synthetic_normalized_prefix_with_scalar_only_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = self._prefix(Path(temporary))
            pin = self._pin(prefix)
            result = admit_pi_core_prefix(
                prefix,
                repo_root=ROOT,
                frozen_lock_sha256=PI_CORE_FROZEN_LOCK_SHA256,
                pin=pin,
            )
            self.assertEqual(result.protocol, PI_CORE_PREFIX_ADMISSION_PROTOCOL)
            self.assertEqual(result.tree_digest, pin.tree_digest)
            self.assertEqual(result.entrypoint_digest, pin.entrypoint_digest)
            scalar = result.scalar()
            self.assertEqual(scalar["status"], "admitted")
            self.assertEqual(scalar["entrypoint_mode"], "0444")
            self.assertNotIn(str(prefix), json.dumps(scalar, sort_keys=True))
            self.assertNotIn(PI_CORE_PREFIX_ENTRYPOINT, json.dumps(scalar, sort_keys=True))

    def test_rejects_tree_and_entrypoint_drift_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = self._prefix(Path(temporary))
            pin = self._pin(prefix)
            entrypoint = prefix / PI_CORE_PREFIX_ENTRYPOINT
            self._make_writable(prefix)
            entrypoint.write_bytes(b"export const drift = true;\n")
            self._normalize(prefix)
            with self.assertRaises(PiCorePrefixAdmissionError):
                admit_pi_core_prefix(
                    prefix,
                    repo_root=ROOT,
                    frozen_lock_sha256=PI_CORE_FROZEN_LOCK_SHA256,
                    pin=pin,
                )

            tree = inspect_normalized_tree(prefix, repo_root=ROOT)
            tree_matched_but_entrypoint_stale = replace(
                pin,
                tree_digest="sha256:" + tree.digest,
                tree_entries=tree.entries,
                tree_files=tree.files,
                tree_links=tree.links,
                tree_size_bytes=tree.size_bytes,
            )
            with self.assertRaises(PiCorePrefixAdmissionError):
                admit_pi_core_prefix(
                    prefix,
                    repo_root=ROOT,
                    frozen_lock_sha256=PI_CORE_FROZEN_LOCK_SHA256,
                    pin=tree_matched_but_entrypoint_stale,
                )

    def test_rejects_wrong_name_symlink_repo_path_and_unsafe_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            prefix = self._prefix(parent)
            pin = self._pin(prefix)
            alias = parent / "alias"
            alias.symlink_to(prefix.name)
            with self.assertRaises(PiCorePrefixAdmissionError):
                admit_pi_core_prefix(
                    alias,
                    repo_root=ROOT,
                    frozen_lock_sha256=PI_CORE_FROZEN_LOCK_SHA256,
                    pin=pin,
                )
            with self.assertRaises(PiCorePrefixAdmissionError):
                admit_pi_core_prefix(
                    ROOT,
                    repo_root=ROOT,
                    frozen_lock_sha256=PI_CORE_FROZEN_LOCK_SHA256,
                    pin=pin,
                )
            prefix.chmod(0o755)
            with self.assertRaises(PiCorePrefixAdmissionError):
                admit_pi_core_prefix(
                    prefix,
                    repo_root=ROOT,
                    frozen_lock_sha256=PI_CORE_FROZEN_LOCK_SHA256,
                    pin=pin,
                )

        with tempfile.TemporaryDirectory() as temporary:
            wrong_name = self._prefix(Path(temporary), name="wrong-prefix")
            pin = self._pin(wrong_name)
            with self.assertRaises(PiCorePrefixAdmissionError):
                admit_pi_core_prefix(
                    wrong_name,
                    repo_root=ROOT,
                    frozen_lock_sha256=PI_CORE_FROZEN_LOCK_SHA256,
                    pin=pin,
                )

    def test_rejects_a_caller_built_pin_that_allows_internal_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = self._prefix(Path(temporary))
            self._make_writable(prefix)
            link = prefix / "internal-link"
            link.symlink_to("node_modules/@mariozechner/pi-agent-core/dist/index.js")
            self._normalize(prefix)
            pin = self._pin(prefix)
            self.assertEqual(pin.tree_links, 1)
            with self.assertRaises(PiCorePrefixAdmissionError):
                admit_pi_core_prefix(
                    prefix,
                    repo_root=ROOT,
                    frozen_lock_sha256=PI_CORE_FROZEN_LOCK_SHA256,
                    pin=pin,
                )

    def test_loader_rejects_unknown_noncanonical_and_symlink_pins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            prefix = self._prefix(parent)
            pin = self._pin(prefix)
            document = self._pin_document(pin)
            candidate = parent / "admission.json"

            document["unexpected"] = True
            self._write_pin(candidate, document)
            with self.assertRaises(PiCorePrefixAdmissionError):
                load_pi_core_prefix_admission_pin(candidate)

            document.pop("unexpected")
            candidate.write_bytes(
                json.dumps(document, ensure_ascii=True, sort_keys=True).encode("ascii")
            )
            candidate.chmod(0o600)
            with self.assertRaises(PiCorePrefixAdmissionError):
                load_pi_core_prefix_admission_pin(candidate)

            self._write_pin(candidate, document)
            alias = parent / "admission-link.json"
            alias.symlink_to(candidate.name)
            with self.assertRaises(PiCorePrefixAdmissionError):
                load_pi_core_prefix_admission_pin(alias)

    def test_rejects_policy_lock_drift_before_prefix_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = self._prefix(Path(temporary))
            pin = self._pin(prefix)
            with self.assertRaises(PiCorePrefixAdmissionError):
                admit_pi_core_prefix(
                    prefix,
                    repo_root=ROOT,
                    frozen_lock_sha256="sha256:" + "0" * 64,
                    pin=pin,
                )
            with self.assertRaises(PiCorePrefixAdmissionError):
                admit_pi_core_prefix(
                    prefix,
                    repo_root=ROOT,
                    frozen_lock_sha256=PI_CORE_FROZEN_LOCK_SHA256,
                    pin=replace(pin, entrypoint_mode=0o555),
                )


if __name__ == "__main__":
    unittest.main()
