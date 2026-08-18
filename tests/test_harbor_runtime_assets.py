from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from bench.harbor_runtime_assets import (
    RuntimeAssetError,
    TREE_PROTOCOL,
    _same_inode,
    stage_immutable_asset,
    verify_admitted_file,
    verify_admitted_symlink,
    verify_normalized_tree,
)


def _canonical_fixture_digest(root: Path) -> tuple[str, int, int, int, int]:
    records: list[tuple[bytes, bytes]] = []
    entries = files = links = size_bytes = 0
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        metadata = os.lstat(path)
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            record = f"D\0{relative}\0{mode:04o}\n".encode()
        elif stat.S_ISREG(metadata.st_mode):
            payload = path.read_bytes()
            record = (
                f"F\0{relative}\0{mode:04o}\0{len(payload)}\0"
                f"{hashlib.sha256(payload).hexdigest()}\n"
            ).encode()
            files += 1
            size_bytes += len(payload)
        elif stat.S_ISLNK(metadata.st_mode):
            record = f"L\0{relative}\0{os.readlink(path)}\n".encode()
            links += 1
        else:
            raise AssertionError("fixture special file")
        entries += 1
        records.append((relative.encode(), record))
    records.sort()
    digest = hashlib.sha256(b"".join(record for _, record in records)).hexdigest()
    return digest, entries, files, links, size_bytes


class HarborRuntimeAssetTests(unittest.TestCase):
    def _tree(self, parent: Path) -> Path:
        root = parent / "tree"
        binary_dir = root / "bin"
        library_dir = root / "lib"
        binary_dir.mkdir(parents=True)
        library_dir.mkdir()
        binary = binary_dir / "tool"
        binary.write_bytes(b"tool-v1\n")
        data = library_dir / "data.json"
        data.write_bytes(b"{}\n")
        (root / "tool").symlink_to("bin/tool")
        binary.chmod(0o555)
        data.chmod(0o444)
        binary_dir.chmod(0o555)
        library_dir.chmod(0o555)
        root.chmod(0o555)
        return root

    def test_exact_fd_relative_tree_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            parent.chmod(0o700)
            root = self._tree(parent)
            digest, entries, files, links, size = _canonical_fixture_digest(root)
            admission = verify_normalized_tree(
                root,
                repo_root=Path.cwd(),
                expected_digest=f"sha256:{digest}",
                expected_size_bytes=size,
                expected_entries=entries,
                expected_files=files,
                expected_links=links,
            )
            self.assertEqual(admission.protocol, TREE_PROTOCOL)
            self.assertEqual(admission.digest, digest)
            self.assertEqual(admission.resolved_path, root.resolve())
            binary = root / "bin" / "tool"
            file_admission = verify_admitted_file(
                admission,
                "bin/tool",
                expected_digest=hashlib.sha256(binary.read_bytes()).hexdigest(),
                expected_size_bytes=binary.stat().st_size,
            )
            self.assertEqual(file_admission.resolved_path, binary)
            link_admission = verify_admitted_symlink(
                admission, "tool", expected_target="bin/tool"
            )
            self.assertEqual(link_admission.resolved_target_path, binary)
            with self.assertRaises(RuntimeAssetError):
                verify_admitted_file(
                    admission,
                    "tool",
                    expected_digest=file_admission.digest,
                    expected_size_bytes=file_admission.size_bytes,
                )

    def test_lexical_symlink_and_escaping_tree_link_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            parent.chmod(0o700)
            root = self._tree(parent)
            alias = parent / "alias"
            alias.symlink_to(root.name)
            digest, entries, files, links, size = _canonical_fixture_digest(root)
            kwargs = {
                "repo_root": Path.cwd(),
                "expected_digest": digest,
                "expected_size_bytes": size,
                "expected_entries": entries,
                "expected_files": files,
                "expected_links": links,
            }
            with self.assertRaises(RuntimeAssetError):
                verify_normalized_tree(alias, **kwargs)

            root.chmod(0o755)
            (root / "escape").symlink_to("../../outside")
            root.chmod(0o555)
            with self.assertRaises(RuntimeAssetError):
                verify_normalized_tree(root, **kwargs)

    def test_hardlinks_and_directory_relist_changes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            parent.chmod(0o700)
            root = parent / "hardlinks"
            root.mkdir()
            first = root / "first"
            first.write_bytes(b"same")
            os.link(first, root / "second")
            first.chmod(0o444)
            root.chmod(0o555)
            with self.assertRaises(RuntimeAssetError):
                verify_normalized_tree(
                    root,
                    repo_root=Path.cwd(),
                    expected_digest="0" * 64,
                    expected_size_bytes=8,
                )

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            parent.chmod(0o700)
            root = parent / "tree"
            root.mkdir()
            payload = root / "file"
            payload.write_bytes(b"x")
            payload.chmod(0o444)
            root.chmod(0o555)
            real_listdir = os.listdir
            root_fd_calls = 0

            def changing_listdir(value):
                nonlocal root_fd_calls
                result = real_listdir(value)
                if isinstance(value, int):
                    root_fd_calls += 1
                    if root_fd_calls == 2:
                        return [*result, "late"]
                return result

            with mock.patch(
                "bench.harbor_runtime_assets.os.listdir",
                side_effect=changing_listdir,
            ):
                with self.assertRaises(RuntimeAssetError):
                    verify_normalized_tree(
                        root,
                        repo_root=Path.cwd(),
                        expected_digest="0" * 64,
                        expected_size_bytes=1,
                    )

    def test_identity_comparison_covers_all_security_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "file"
            path.write_bytes(b"identity")
            baseline = os.stat(path, follow_symlinks=False)
            self.assertTrue(_same_inode(baseline, baseline))
            fields = (
                "st_dev",
                "st_ino",
                "st_uid",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            for field in fields:
                changed = mock.Mock(wraps=baseline)
                for copied in fields:
                    setattr(changed, copied, getattr(baseline, copied))
                setattr(changed, field, getattr(baseline, field) + 1)
                with self.subTest(field=field):
                    self.assertFalse(_same_inode(baseline, changed))

    def test_tracked_asset_is_staged_with_a_stricter_mode(self) -> None:
        source = Path(__file__).resolve().parents[1] / "bench" / "assets" / "harbor_uds_relay.js"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            parent.chmod(0o700)
            staged = stage_immutable_asset(
                source,
                parent / "relay.js",
                repo_root=Path.cwd(),
                expected_digest=f"sha256:{digest}",
                expected_source_mode=0o644,
                output_mode=0o444,
            )
            self.assertEqual(stat.S_IMODE(staged.stat().st_mode), 0o444)
            self.assertEqual(staged.read_bytes(), source.read_bytes())
            with self.assertRaises(RuntimeAssetError):
                stage_immutable_asset(
                    source,
                    parent / "again.js",
                    repo_root=Path.cwd(),
                    expected_digest="0" * 64,
                    expected_source_mode=0o644,
                    output_mode=0o444,
                )


if __name__ == "__main__":
    unittest.main()
