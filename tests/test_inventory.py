"""Tests for read-only local artifact discovery."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from bench.inventory import discover_docker_images


class DockerInventoryTests(unittest.TestCase):
    def test_digest_only_image_without_tag_is_discovered(self) -> None:
        row = {
            "Repository": "lmsysorg/sglang",
            "Tag": "<none>",
            "Digest": "sha256:" + "a" * 64,
            "ID": "sha256:" + "b" * 64,
            "Size": "30.2GB",
        }
        with patch(
            "bench.inventory._run", return_value=json.dumps(row)
        ) as run:
            images = discover_docker_images()

        run.assert_called_once_with(
            "docker",
            [
                "image",
                "ls",
                "--all",
                "--digests",
                "--no-trunc",
                "--format",
                "{{json .}}",
            ],
            timeout_s=15.0,
        )
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].repository, "lmsysorg/sglang")
        self.assertIsNone(images[0].tag)
        self.assertEqual(images[0].digest, row["Digest"])
        self.assertEqual(images[0].reference, "lmsysorg/sglang")


if __name__ == "__main__":
    unittest.main()
