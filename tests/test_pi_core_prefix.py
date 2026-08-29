from __future__ import annotations

import base64
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench import pi_core_prefix as prefix


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def _integrity(payload: bytes) -> str:
    return "sha512-" + base64.b64encode(hashlib.sha512(payload).digest()).decode("ascii")


def _entry(
    label: str,
    *,
    version: str = "1.0.0",
    dependencies: dict[str, str] | None = None,
    optional_dependencies: dict[str, str] | None = None,
    peer_dependencies: dict[str, str] | None = None,
    peer_dependencies_meta: dict[str, dict[str, bool]] | None = None,
    optional: bool = False,
    has_install_script: bool = False,
    source_metadata: bool = False,
) -> tuple[dict[str, object], bytes]:
    """Build one synthetic npm-lock entry and its matching fake tarball bytes."""

    payload = ("synthetic-pi-tarball:" + label).encode("ascii")
    value: dict[str, object] = {
        "version": version,
        "resolved": (
            "https://registry.npmjs.org/synthetic-" + label + "/-/" + label + ".tgz"
        ),
        "integrity": _integrity(payload),
    }
    if dependencies is not None:
        value["dependencies"] = dependencies
    if optional_dependencies is not None:
        value["optionalDependencies"] = optional_dependencies
    if peer_dependencies is not None:
        value["peerDependencies"] = peer_dependencies
    if peer_dependencies_meta is not None:
        value["peerDependenciesMeta"] = peer_dependencies_meta
    if optional:
        value["optional"] = True
    if has_install_script:
        value["hasInstallScript"] = True
    if source_metadata:
        # These are valid in an npm source lock but intentionally excluded from
        # the frozen output schema.
        value["name"] = "synthetic-" + label
        value["license"] = "UNLICENSED"
    return value, payload


def _source_document(packages: dict[str, object]) -> dict[str, object]:
    return {
        "lockfileVersion": 3,
        "name": "ambient-synthetic-project",
        "packages": {"": {"name": "ambient-synthetic-project"}, **packages},
        "requires": True,
        "version": "1.0.0",
    }


class PiCorePrefixTests(unittest.TestCase):
    """All fixtures are locally generated synthetic JSON and content blobs."""

    def _basic_packages(
        self,
    ) -> tuple[dict[str, object], dict[str, bytes], tuple[str, ...]]:
        core_path = "node_modules/" + prefix.PI_AGENT_CORE_PACKAGE
        ai_path = "node_modules/" + prefix.PI_AI_PACKAGE
        core, core_payload = _entry(
            "agent-core",
            version=prefix.PI_CORE_VERSION,
            dependencies={prefix.PI_AI_PACKAGE: "^0.57.1"},
            source_metadata=True,
        )
        ai, ai_payload = _entry(
            "pi-ai",
            version=prefix.PI_CORE_VERSION,
            dependencies={"required": "^1.0.0"},
            optional_dependencies={"optional": "^1.0.0"},
            source_metadata=True,
        )
        required, required_payload = _entry("required", has_install_script=True)
        optional, optional_payload = _entry("optional", optional=True)
        packages: dict[str, object] = {
            core_path: core,
            ai_path: ai,
            "node_modules/required": required,
            "node_modules/optional": optional,
        }
        payloads = {
            str(core["integrity"]): core_payload,
            str(ai["integrity"]): ai_payload,
            str(required["integrity"]): required_payload,
            str(optional["integrity"]): optional_payload,
        }
        return packages, payloads, tuple(packages)

    @contextmanager
    def _patched_contract(
        self,
        source_bytes: bytes,
        packages: dict[str, object],
        selected_paths: tuple[str, ...],
    ):
        core = packages["node_modules/" + prefix.PI_AGENT_CORE_PACKAGE]
        ai = packages["node_modules/" + prefix.PI_AI_PACKAGE]
        self.assertIsInstance(core, dict)
        self.assertIsInstance(ai, dict)
        selected_entries = [packages[path] for path in selected_paths]
        self.assertTrue(all(isinstance(entry, dict) for entry in selected_entries))
        entries = [entry for entry in selected_entries if isinstance(entry, dict)]
        integrities = {str(entry["integrity"]) for entry in entries}
        install_script_count = sum(entry.get("hasInstallScript") is True for entry in entries)
        optional_count = sum(entry.get("optional") is True for entry in entries)
        paths = sorted(selected_paths)
        path_anchor = prefix._sha256(("\n".join(paths) + "\n").encode("utf-8"))
        record_by_path: dict[str, str] = {}
        for path in paths:
            entry = packages[path]
            self.assertIsInstance(entry, dict)
            record_by_path[path] = f"{path}\t{entry['version']}\t{entry['integrity']}"
        record_anchor = prefix._sha256(
            ("\n".join(record_by_path[path] for path in paths) + "\n").encode("utf-8")
        )
        frozen_packages: dict[str, object] = {"": prefix._root_package_entry()}
        for path in paths:
            entry = packages[path]
            self.assertIsInstance(entry, dict)
            frozen_packages[path] = {
                key: entry[key]
                for key in sorted(prefix._OUTPUT_ENTRY_KEYS)
                if key in entry
            }
        frozen_document = {
            "lockfileVersion": prefix.PI_CORE_LOCKFILE_VERSION,
            "name": prefix.PI_CORE_PREFIX_NAME,
            "packages": frozen_packages,
            "requires": True,
            "version": prefix.PI_CORE_VERSION,
        }
        with patch.multiple(
            prefix,
            PI_CORE_CANDIDATE_LOCK_SHA256=prefix._sha256(source_bytes),
            PI_AGENT_CORE_INTEGRITY=str(core["integrity"]),
            PI_AI_INTEGRITY=str(ai["integrity"]),
            PI_CORE_CLOSURE_PACKAGE_COUNT=len(selected_paths),
            PI_CORE_UNIQUE_ARTIFACT_COUNT=len(integrities),
            PI_CORE_INSTALL_SCRIPT_PACKAGE_COUNT=install_script_count,
            PI_CORE_OPTIONAL_PACKAGE_COUNT=optional_count,
            PI_CORE_PATH_LIST_SHA256=path_anchor,
            PI_CORE_PATH_VERSION_INTEGRITY_SHA256=record_anchor,
            PI_CORE_FROZEN_LOCK_SHA256=prefix._sha256(
                _canonical_json_bytes(frozen_document)
            ),
        ):
            yield

    @staticmethod
    def _write_private(path: Path, payload: bytes) -> None:
        path.write_bytes(payload)
        os.chmod(path, 0o600)

    def _freeze(
        self,
        directory: Path,
        packages: dict[str, object],
        selected_paths: tuple[str, ...],
    ) -> tuple[prefix.PiCoreLockSummary, bytes]:
        source_bytes = _canonical_json_bytes(_source_document(packages))
        source_path = directory / "candidate.package-lock.json"
        self._write_private(source_path, source_bytes)
        with self._patched_contract(source_bytes, packages, selected_paths):
            return prefix.freeze_pinned_pi_core_lock(source_path), source_bytes

    def test_freeze_strips_source_metadata_and_round_trips_canonical_lock(self) -> None:
        packages, _payloads, selected_paths = self._basic_packages()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_bytes = _canonical_json_bytes(_source_document(packages))
            source_path = directory / "candidate.package-lock.json"
            self._write_private(source_path, source_bytes)
            with self._patched_contract(source_bytes, packages, selected_paths):
                summary = prefix.freeze_pinned_pi_core_lock(source_path)
                frozen_core = summary.document["packages"][
                    "node_modules/" + prefix.PI_AGENT_CORE_PACKAGE
                ]
                self.assertNotIn("name", frozen_core)
                self.assertNotIn("license", frozen_core)
                self.assertEqual(summary.package_count, 4)
                self.assertEqual(summary.unique_artifact_count, 4)
                self.assertEqual(summary.install_script_package_count, 1)
                self.assertEqual(summary.optional_package_count, 1)

                target = directory / "pi-core.package-lock.json"
                prefix.write_new_frozen_pi_core_lock(target, summary)
                self.assertEqual(target.read_bytes(), _canonical_json_bytes(summary.document))
                loaded = prefix.load_frozen_pi_core_lock(target)

            self.assertEqual(loaded, summary)
            scalar = loaded.scalar(status="frozen", origin_lock_sha256="sha256:" + "a" * 64)
            self.assertNotIn(str(directory), json.dumps(scalar, sort_keys=True))

    def test_load_rejects_noncanonical_and_source_only_frozen_fields(self) -> None:
        packages, _payloads, selected_paths = self._basic_packages()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_bytes = _canonical_json_bytes(_source_document(packages))
            source_path = directory / "candidate.package-lock.json"
            self._write_private(source_path, source_bytes)
            with self._patched_contract(source_bytes, packages, selected_paths):
                summary = prefix.freeze_pinned_pi_core_lock(source_path)
                target = directory / "pi-core.package-lock.json"
                prefix.write_new_frozen_pi_core_lock(target, summary)

                self._write_private(target, target.read_bytes() + b"\n")
                with self.assertRaisesRegex(prefix.PiCorePrefixError, "not canonical"):
                    prefix.load_frozen_pi_core_lock(target)

                injected = copy.deepcopy(summary.document)
                frozen_core = injected["packages"][
                    "node_modules/" + prefix.PI_AGENT_CORE_PACKAGE
                ]
                self.assertIsInstance(frozen_core, dict)
                frozen_core["name"] = "source-field-must-not-survive"
                self._write_private(target, _canonical_json_bytes(injected))
                with self.assertRaisesRegex(prefix.PiCorePrefixError, "package entry"):
                    prefix.load_frozen_pi_core_lock(target)

    def test_write_refuses_to_overwrite_an_existing_lock(self) -> None:
        packages, _payloads, selected_paths = self._basic_packages()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary, source_bytes = self._freeze(directory, packages, selected_paths)
            target = directory / "pi-core.package-lock.json"
            with self._patched_contract(source_bytes, packages, selected_paths):
                prefix.write_new_frozen_pi_core_lock(target, summary)
                with self.assertRaisesRegex(prefix.PiCorePrefixError, "already exists"):
                    prefix.write_new_frozen_pi_core_lock(target, summary)

    def test_closure_resolves_nested_and_ancestor_package_locations(self) -> None:
        core_path = "node_modules/" + prefix.PI_AGENT_CORE_PACKAGE
        ai_path = "node_modules/" + prefix.PI_AI_PACKAGE
        nested_shared_path = ai_path + "/node_modules/shared"
        ancestor_path = ai_path + "/node_modules/ancestor"
        core, _ = _entry(
            "agent-core-resolution",
            version=prefix.PI_CORE_VERSION,
            dependencies={prefix.PI_AI_PACKAGE: "^0.57.1"},
        )
        ai, _ = _entry(
            "pi-ai-resolution",
            version=prefix.PI_CORE_VERSION,
            dependencies={"shared": "^1.0.0"},
        )
        shared, _ = _entry("shared-resolution", dependencies={"ancestor": "^1.0.0"})
        ancestor, _ = _entry("ancestor-resolution")
        packages: dict[str, object] = {
            core_path: core,
            ai_path: ai,
            nested_shared_path: shared,
            ancestor_path: ancestor,
        }
        selected_paths = tuple(packages)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_bytes = _canonical_json_bytes(_source_document(packages))
            source_path = directory / "candidate.package-lock.json"
            self._write_private(source_path, source_bytes)
            with self._patched_contract(source_bytes, packages, selected_paths):
                summary = prefix.freeze_pinned_pi_core_lock(source_path)

        self.assertEqual(
            set(summary.document["packages"]),
            {"", core_path, ai_path, nested_shared_path, ancestor_path},
        )

    def test_optional_peer_can_be_unresolved_but_required_peer_cannot(self) -> None:
        core_path = "node_modules/" + prefix.PI_AGENT_CORE_PACKAGE
        ai_path = "node_modules/" + prefix.PI_AI_PACKAGE
        core, _ = _entry(
            "agent-core-peer",
            version=prefix.PI_CORE_VERSION,
            dependencies={prefix.PI_AI_PACKAGE: "^0.57.1"},
        )
        optional_ai, _ = _entry(
            "pi-ai-optional-peer",
            version=prefix.PI_CORE_VERSION,
            peer_dependencies={"uninstalled-peer": "^1.0.0"},
            peer_dependencies_meta={"uninstalled-peer": {"optional": True}},
        )
        optional_packages: dict[str, object] = {core_path: core, ai_path: optional_ai}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_bytes = _canonical_json_bytes(_source_document(optional_packages))
            source_path = directory / "optional-peer.package-lock.json"
            self._write_private(source_path, source_bytes)
            with self._patched_contract(source_bytes, optional_packages, tuple(optional_packages)):
                summary = prefix.freeze_pinned_pi_core_lock(source_path)
        self.assertEqual(summary.package_count, 2)

        required_ai = copy.deepcopy(optional_ai)
        self.assertIsInstance(required_ai, dict)
        required_ai.pop("peerDependenciesMeta")
        required_packages: dict[str, object] = {core_path: core, ai_path: required_ai}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_bytes = _canonical_json_bytes(_source_document(required_packages))
            source_path = directory / "required-peer.package-lock.json"
            self._write_private(source_path, source_bytes)
            with self._patched_contract(source_bytes, required_packages, tuple(required_packages)):
                with self.assertRaisesRegex(prefix.PiCorePrefixError, "unresolved dependency"):
                    prefix.freeze_pinned_pi_core_lock(source_path)

    def test_freeze_rejects_missing_required_dependency_and_invalid_entries(self) -> None:
        core_path = "node_modules/" + prefix.PI_AGENT_CORE_PACKAGE
        ai_path = "node_modules/" + prefix.PI_AI_PACKAGE
        core, _ = _entry(
            "agent-core-invalid",
            version=prefix.PI_CORE_VERSION,
            dependencies={prefix.PI_AI_PACKAGE: "^0.57.1"},
        )
        ai, _ = _entry(
            "pi-ai-invalid",
            version=prefix.PI_CORE_VERSION,
            dependencies={"not-present": "^1.0.0"},
        )
        missing_packages: dict[str, object] = {core_path: core, ai_path: ai}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_bytes = _canonical_json_bytes(_source_document(missing_packages))
            source_path = directory / "missing.package-lock.json"
            self._write_private(source_path, source_bytes)
            with self._patched_contract(source_bytes, missing_packages, tuple(missing_packages)):
                with self.assertRaisesRegex(prefix.PiCorePrefixError, "unresolved dependency"):
                    prefix.freeze_pinned_pi_core_lock(source_path)

        invalid_variants: list[tuple[str, dict[str, object]]] = []
        bad_url_core = copy.deepcopy(core)
        self.assertIsInstance(bad_url_core, dict)
        bad_url_core["resolved"] = "file:/synthetic/private/core.tgz"
        invalid_variants.append(("resolved URL", {core_path: bad_url_core, ai_path: ai}))
        bad_spec_core = copy.deepcopy(core)
        self.assertIsInstance(bad_spec_core, dict)
        bad_spec_core["dependencies"] = {prefix.PI_AI_PACKAGE: "file:../pi-ai"}
        invalid_variants.append(("dependency specification", {core_path: bad_spec_core, ai_path: ai}))
        extra_field_core = copy.deepcopy(core)
        self.assertIsInstance(extra_field_core, dict)
        extra_field_core["unexpected"] = True
        invalid_variants.append(("unknown entry field", {core_path: extra_field_core, ai_path: ai}))
        for label, variant in invalid_variants:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                source_bytes = _canonical_json_bytes(_source_document(variant))
                source_path = directory / "invalid.package-lock.json"
                self._write_private(source_path, source_bytes)
                with self._patched_contract(source_bytes, variant, tuple(variant)):
                    with self.assertRaises(prefix.PiCorePrefixError):
                        prefix.freeze_pinned_pi_core_lock(source_path)

    def test_freeze_rejects_a_source_lock_with_the_wrong_identity(self) -> None:
        packages, _payloads, selected_paths = self._basic_packages()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_bytes = _canonical_json_bytes(_source_document(packages))
            source_path = directory / "candidate.package-lock.json"
            self._write_private(source_path, source_bytes)
            with self._patched_contract(source_bytes, packages, selected_paths), patch.object(
                prefix, "PI_CORE_CANDIDATE_LOCK_SHA256", "sha256:" + "0" * 64
            ):
                with self.assertRaisesRegex(prefix.PiCorePrefixError, "identity changed"):
                    prefix.freeze_pinned_pi_core_lock(source_path)

    def _write_cache(
        self, root: Path, payloads: dict[str, bytes]
    ) -> None:
        root.mkdir(mode=0o700)
        for integrity, payload in payloads.items():
            artifact = prefix._cache_blob_path(root, integrity)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            self._write_private(artifact, payload)

    def test_cache_audit_hashes_every_synthetic_blob_and_counts_unique_artifacts(self) -> None:
        packages, payloads, selected_paths = self._basic_packages()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary, source_bytes = self._freeze(directory, packages, selected_paths)
            cache_root = directory / "content-v2" / "sha512"
            cache_root.parent.mkdir(mode=0o700)
            self._write_cache(cache_root, payloads)
            with self._patched_contract(source_bytes, packages, selected_paths):
                audit = prefix.audit_pi_core_cache(summary, cache_sha512_root=cache_root)

        self.assertEqual(audit.artifact_count, 4)
        self.assertEqual(audit.package_count, 4)
        self.assertEqual(audit.artifact_size_bytes, sum(map(len, payloads.values())))
        self.assertEqual(audit.scalar()["status"], "cache_complete")

    def test_cache_audit_deduplicates_package_records_by_integrity(self) -> None:
        packages, payloads, selected_paths = self._basic_packages()
        required = packages["node_modules/required"]
        optional = packages["node_modules/optional"]
        self.assertIsInstance(required, dict)
        self.assertIsInstance(optional, dict)
        original_optional_integrity = str(optional["integrity"])
        optional["integrity"] = required["integrity"]
        optional["resolved"] = required["resolved"]
        payloads.pop(original_optional_integrity)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary, source_bytes = self._freeze(directory, packages, selected_paths)
            cache_root = directory / "content-v2" / "sha512"
            cache_root.parent.mkdir(mode=0o700)
            self._write_cache(cache_root, payloads)
            with self._patched_contract(source_bytes, packages, selected_paths):
                audit = prefix.audit_pi_core_cache(summary, cache_sha512_root=cache_root)

        self.assertEqual(summary.package_count, 4)
        self.assertEqual(summary.unique_artifact_count, 3)
        self.assertEqual(audit.artifact_count, 3)
        self.assertEqual(audit.artifact_size_bytes, sum(map(len, payloads.values())))

    def test_cache_audit_rejects_missing_and_digest_mismatched_blobs(self) -> None:
        packages, payloads, selected_paths = self._basic_packages()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary, source_bytes = self._freeze(directory, packages, selected_paths)
            cache_root = directory / "content-v2" / "sha512"
            cache_root.parent.mkdir(mode=0o700)
            with self._patched_contract(source_bytes, packages, selected_paths):
                with self.assertRaisesRegex(prefix.PiCorePrefixError, "unavailable"):
                    prefix.audit_pi_core_cache(summary, cache_sha512_root=cache_root)

                self._write_cache(cache_root, payloads)
                first_integrity = next(iter(payloads))
                tampered = prefix._cache_blob_path(cache_root, first_integrity)
                self._write_private(tampered, b"synthetic-but-wrong")
                with self.assertRaisesRegex(prefix.PiCorePrefixError, "digest changed"):
                    prefix.audit_pi_core_cache(summary, cache_sha512_root=cache_root)

    def test_cache_audit_accepts_a_group_writable_external_content_store(self) -> None:
        packages, payloads, selected_paths = self._basic_packages()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary, source_bytes = self._freeze(directory, packages, selected_paths)
            cache_root = directory / "content-v2" / "sha512"
            cache_root.parent.mkdir(mode=0o700)
            self._write_cache(cache_root, payloads)
            os.chmod(cache_root, 0o770)
            with self._patched_contract(source_bytes, packages, selected_paths):
                audit = prefix.audit_pi_core_cache(summary, cache_sha512_root=cache_root)

        self.assertEqual(audit.artifact_count, 4)

    def test_write_revalidates_forged_and_mutated_summaries_before_publishing(self) -> None:
        packages, _payloads, selected_paths = self._basic_packages()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary, source_bytes = self._freeze(directory, packages, selected_paths)
            forged = prefix.PiCoreLockSummary(
                protocol=summary.protocol,
                frozen_lock_sha256=summary.frozen_lock_sha256,
                package_count=summary.package_count + 1,
                integrity_count=summary.integrity_count,
                unique_artifact_count=summary.unique_artifact_count,
                install_script_package_count=summary.install_script_package_count,
                optional_package_count=summary.optional_package_count,
                document=copy.deepcopy(summary.document),
            )
            mutated = copy.deepcopy(summary)
            mutated_core = mutated.document["packages"][
                "node_modules/" + prefix.PI_AGENT_CORE_PACKAGE
            ]
            self.assertIsInstance(mutated_core, dict)
            mutated_core["resolved"] = (
                "https://registry.npmjs.org/synthetic-alternate/-/alternate.tgz"
            )
            with self._patched_contract(source_bytes, packages, selected_paths):
                forged_target = directory / "forged.package-lock.json"
                with self.assertRaisesRegex(prefix.PiCorePrefixError, "summary changed"):
                    prefix.write_new_frozen_pi_core_lock(forged_target, forged)
                self.assertFalse(forged_target.exists())

                mutated_target = directory / "mutated.package-lock.json"
                with self.assertRaisesRegex(prefix.PiCorePrefixError, "identity changed"):
                    prefix.write_new_frozen_pi_core_lock(mutated_target, mutated)
                self.assertFalse(mutated_target.exists())

    def test_cache_audit_revalidates_forged_and_mutated_summaries_before_io(self) -> None:
        packages, _payloads, selected_paths = self._basic_packages()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary, source_bytes = self._freeze(directory, packages, selected_paths)
            forged = prefix.PiCoreLockSummary(
                protocol=summary.protocol,
                frozen_lock_sha256="sha256:" + "0" * 64,
                package_count=summary.package_count,
                integrity_count=summary.integrity_count,
                unique_artifact_count=summary.unique_artifact_count,
                install_script_package_count=summary.install_script_package_count,
                optional_package_count=summary.optional_package_count,
                document=copy.deepcopy(summary.document),
            )
            mutated = copy.deepcopy(summary)
            mutated_ai = mutated.document["packages"][
                "node_modules/" + prefix.PI_AI_PACKAGE
            ]
            self.assertIsInstance(mutated_ai, dict)
            mutated_ai["resolved"] = (
                "https://registry.npmjs.org/synthetic-alternate/-/alternate.tgz"
            )
            nonexistent_cache = directory / "not-a-cache"
            with self._patched_contract(source_bytes, packages, selected_paths):
                with self.assertRaisesRegex(prefix.PiCorePrefixError, "summary changed"):
                    prefix.audit_pi_core_cache(forged, cache_sha512_root=nonexistent_cache)
                with self.assertRaisesRegex(prefix.PiCorePrefixError, "identity changed"):
                    prefix.audit_pi_core_cache(mutated, cache_sha512_root=nonexistent_cache)

    def test_write_rejects_a_group_writable_destination_parent(self) -> None:
        packages, _payloads, selected_paths = self._basic_packages()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary, source_bytes = self._freeze(directory, packages, selected_paths)
            destination_parent = directory / "group-writable"
            destination_parent.mkdir(mode=0o700)
            os.chmod(destination_parent, 0o770)
            target = destination_parent / "pi-core.package-lock.json"
            with self._patched_contract(source_bytes, packages, selected_paths):
                with self.assertRaisesRegex(prefix.PiCorePrefixError, "destination is unsafe"):
                    prefix.write_new_frozen_pi_core_lock(target, summary)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
