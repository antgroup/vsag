import re
import subprocess
import unittest
from unittest import mock

import changed_source_filter


class ChangedSourceFilterTest(unittest.TestCase):
    def test_build_filter_returns_empty_pattern_for_no_sources(self):
        self.assertEqual(changed_source_filter.build_filter([]), "")

    def test_build_filter_escapes_paths_and_anchors_them(self):
        paths = [
            "src/factory/factory.cpp",
            "src/storage/file+[draft].cpp",
        ]

        pattern = changed_source_filter.build_filter(paths)

        self.assertIsNotNone(re.search(pattern, "/workspace/src/factory/factory.cpp"))
        self.assertIsNotNone(re.search(pattern, "/workspace/src/storage/file+[draft].cpp"))
        self.assertIsNone(re.search(pattern, "/workspace/src/factory/factory.cpp.tmp"))
        self.assertIsNone(re.search(pattern, "/workspace/src/storage/fileeeeeed.cpp"))

    @mock.patch("changed_source_filter.subprocess.run")
    def test_get_changed_sources_uses_nul_delimited_git_diff(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=b"src/factory/factory.cpp\0src/storage/tlv_section.cpp\0",
        )

        paths = changed_source_filter.get_changed_sources("origin/main", "HEAD")

        self.assertEqual(
            paths,
            ["src/factory/factory.cpp", "src/storage/tlv_section.cpp"],
        )
        run.assert_called_once_with(
            [
                "git",
                "diff",
                "--name-only",
                "-z",
                "origin/main..HEAD",
                "--",
                "src/**/*.cpp",
                ":(exclude)src/**/*_test.cpp",
            ],
            check=True,
            stdout=subprocess.PIPE,
        )

    @mock.patch("changed_source_filter.subprocess.run")
    def test_get_changed_sources_propagates_git_failure(self, run):
        run.side_effect = subprocess.CalledProcessError(128, ["git", "diff"])

        with self.assertRaises(subprocess.CalledProcessError):
            changed_source_filter.get_changed_sources("origin/main", "HEAD")


if __name__ == "__main__":
    unittest.main()
