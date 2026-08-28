from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from bench import seccomp_profile_contract as contract


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SECCOMP_DIRECTORY = Path("patches/sglang/seccomp")
CONTRACT_PATH = SECCOMP_DIRECTORY / "qwen38-io-uring-docker-v29.2.1.toml"
BASELINE_PATH = SECCOMP_DIRECTORY / "moby-profiles-seccomp-v0.1.0-default.json"
DERIVED_PATH = SECCOMP_DIRECTORY / "qwen38-io-uring-docker-v29.2.1.json"


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class SeccompProfileContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "repository"
        target = self.root / SECCOMP_DIRECTORY
        target.parent.mkdir(parents=True)
        shutil.copytree(REPOSITORY_ROOT / SECCOMP_DIRECTORY, target)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _path(self, relative: Path) -> Path:
        return self.root / relative

    def _derived_object(self) -> dict[str, object]:
        return json.loads(self._path(DERIVED_PATH).read_text(encoding="utf-8"))

    def _write_derived(self, value: object) -> None:
        self._path(DERIVED_PATH).write_bytes(_canonical_json(value))

    def test_production_artifacts_are_exactly_pinned_and_derivable(self) -> None:
        result = contract.verify_seccomp_profile_contract(self.root)
        baseline = json.loads(self._path(BASELINE_PATH).read_text(encoding="utf-8"))
        derived = self._derived_object()

        self.assertEqual(result.candidate_id, contract.CONTRACT_CANDIDATE_ID)
        self.assertEqual(result.status, contract.CONTRACT_STATUS)
        self.assertEqual(result.baseline_sha256, contract.BASELINE_SHA256)
        self.assertEqual(result.derived_sha256, contract.DERIVED_SHA256)
        self.assertEqual(result.engine_version, "29.2.1")
        self.assertEqual(result.engine_source_commit, contract.DOCKER_ENGINE_COMMIT)
        self.assertEqual(result.profiles_source_commit, contract.MOBY_PROFILES_COMMIT)
        self.assertTrue(result.as_dict()["verified"])
        self.assertNotIn("path", result.as_dict())
        self.assertEqual(len(derived["syscalls"]), len(baseline["syscalls"]) + 1)
        self.assertEqual(
            derived["syscalls"][-1],
            {"action": "SCMP_ACT_ALLOW", "names": list(contract.IO_URING_NAMES)},
        )

    def test_baseline_is_raw_pinned_bytes_and_derived_is_canonical(self) -> None:
        baseline = self._path(BASELINE_PATH).read_bytes()
        derived = self._path(DERIVED_PATH).read_bytes()
        expected = contract._expected_derived_profile(
            contract.strict_json_loads(baseline)
        )

        self.assertEqual(
            contract.BASELINE_SHA256,
            hashlib.sha256(baseline).hexdigest(),
        )
        self.assertEqual(derived, _canonical_json(expected))
        self.assertEqual(
            contract.DERIVED_SHA256,
            hashlib.sha256(derived).hexdigest(),
        )
        self.assertFalse(baseline.endswith(b"\n"))
        self.assertTrue(derived.endswith(b"\n"))

    def test_rejects_duplicate_json_keys_at_every_depth(self) -> None:
        cases = (
            b'{"syscalls":[],"syscalls":[]}',
            b'{"syscalls":[{"names":["read"],"action":"SCMP_ACT_ALLOW","action":"SCMP_ACT_ERRNO"}]}',
        )
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(contract.SeccompProfileContractError):
                    contract.strict_json_loads(value)

        self._path(DERIVED_PATH).write_bytes(cases[0])
        with self.assertRaises(contract.SeccompProfileContractError):
            contract.verify_seccomp_profile_contract(self.root)

    def test_rejects_nonfinite_json_constants(self) -> None:
        for value in (b'{"value":NaN}', b'{"value":Infinity}', b'{"value":-Infinity}'):
            with self.subTest(value=value):
                with self.assertRaises(contract.SeccompProfileContractError):
                    contract.strict_json_loads(value)

    def test_rejects_any_derived_change_beyond_one_allow_group(self) -> None:
        changed_default = self._derived_object()
        changed_default["defaultAction"] = "SCMP_ACT_KILL"
        self._write_derived(changed_default)
        with self.assertRaises(contract.SeccompProfileContractError):
            contract.verify_seccomp_profile_contract(self.root)

        original = json.loads((REPOSITORY_ROOT / DERIVED_PATH).read_text(encoding="utf-8"))
        extra_group = deepcopy(original)
        extra_group["syscalls"].append(
            {"action": "SCMP_ACT_ALLOW", "names": ["getpid"]}
        )
        self._write_derived(extra_group)
        with self.assertRaises(contract.SeccompProfileContractError):
            contract.verify_seccomp_profile_contract(self.root)

    def test_rejects_conditioned_or_overscoped_io_uring_group(self) -> None:
        altered = self._derived_object()
        group = altered["syscalls"][-1]
        self.assertIsInstance(group, dict)
        group["args"] = []
        self._write_derived(altered)
        with self.assertRaises(contract.SeccompProfileContractError):
            contract.verify_seccomp_profile_contract(self.root)

        altered = json.loads(
            (REPOSITORY_ROOT / DERIVED_PATH).read_text(encoding="utf-8")
        )
        group = altered["syscalls"][-1]
        self.assertIsInstance(group, dict)
        group["names"] = [*contract.IO_URING_NAMES, "getpid"]
        self._write_derived(altered)
        with self.assertRaises(contract.SeccompProfileContractError):
            contract.verify_seccomp_profile_contract(self.root)

    def test_rejects_noncanonical_derived_serialization(self) -> None:
        document = self._derived_object()
        self._path(DERIVED_PATH).write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(contract.SeccompProfileContractError) as raised:
            contract.verify_seccomp_profile_contract(self.root)
        self.assertIn("canonical", str(raised.exception))

    def test_rejects_baseline_that_already_admits_io_uring(self) -> None:
        baseline = {
            "syscalls": [
                {
                    "action": "SCMP_ACT_ALLOW",
                    "names": ["io_uring_setup"],
                }
            ]
        }
        with self.assertRaises(contract.SeccompProfileContractError):
            contract._expected_derived_profile(baseline)

    def test_rejects_unknown_or_duplicate_toml_configuration(self) -> None:
        path = self._path(CONTRACT_PATH)
        original = path.read_text(encoding="utf-8")
        for suffix in ("\n[unexpected]\nvalue = true\n", "\nstatus = \"wrong\"\n"):
            with self.subTest(suffix=suffix):
                path.write_text(original + suffix, encoding="utf-8")
                with self.assertRaises(contract.SeccompProfileContractError):
                    contract.load_seccomp_profile_contract(self.root)
                path.write_text(original, encoding="utf-8")

    def test_rejects_symlinks_and_unsafe_relative_paths(self) -> None:
        derived = self._path(DERIVED_PATH)
        copied = self._path(SECCOMP_DIRECTORY / "copied.json")
        shutil.copyfile(derived, copied)
        derived.unlink()
        derived.symlink_to(copied.name)
        with self.assertRaises(contract.SeccompProfileContractError):
            contract.verify_seccomp_profile_contract(self.root)

        with self.assertRaises(contract.SeccompProfileContractError):
            contract._read_regular_repository_file(self.root, "../outside.json", "fixture")

    def test_rejects_symlinked_contract_and_baseline(self) -> None:
        for relative in (CONTRACT_PATH, BASELINE_PATH):
            with self.subTest(relative=relative):
                original = self._path(relative)
                copied = original.with_name("copy-" + original.name)
                shutil.copyfile(original, copied)
                original.unlink()
                original.symlink_to(copied.name)
                with self.assertRaises(contract.SeccompProfileContractError):
                    contract.verify_seccomp_profile_contract(self.root)
                original.unlink()
                copied.replace(original)

    def test_errors_do_not_reflect_untrusted_configuration_text(self) -> None:
        secret = "hf_example_profile_secret"
        path = self._path(CONTRACT_PATH)
        path.write_text(
            path.read_text(encoding="utf-8")
            + f"\n[unexpected]\nsecret = \"{secret}\"\n",
            encoding="utf-8",
        )
        with self.assertRaises(contract.SeccompProfileContractError) as raised:
            contract.load_seccomp_profile_contract(self.root)
        self.assertNotIn(secret, str(raised.exception))

    def test_cli_is_read_only_and_path_free(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = contract.main(["--repository-root", str(self.root)])
        self.assertEqual(result, 0, stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["verified"])
        self.assertNotIn(str(self.root), stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
