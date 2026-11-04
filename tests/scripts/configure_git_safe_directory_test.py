#!/usr/bin/env python3

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts/ci/configure-git-safe-directory.sh"


class ConfigureGitSafeDirectoryTest(unittest.TestCase):
    def run_helper(
        self, workspace: Path, *, github_actions: str | None = "true"
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["GITHUB_WORKSPACE"] = str(workspace)
        if github_actions is None:
            environment.pop("GITHUB_ACTIONS", None)
        else:
            environment["GITHUB_ACTIONS"] = github_actions
        return subprocess.run(
            [str(HELPER)],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )

    def test_refuses_to_change_local_developer_config(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_helper(Path(directory), github_actions=None)

        self.assertEqual(result.returncode, 1)
        self.assertIn("may only be configured in GitHub Actions", result.stderr)

    def test_requires_an_absolute_workspace_path(self):
        result = self.run_helper(Path("relative/workspace"))

        self.assertEqual(result.returncode, 1)
        self.assertIn("GITHUB_WORKSPACE must be an absolute path", result.stderr)

    def test_makes_only_the_exact_workspace_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            workspace = temporary / "workspace"
            home = temporary / "home"
            workspace.mkdir()
            home.mkdir()
            subprocess.run(["git", "init", "-q", str(workspace)], check=True)
            (workspace / "tracked.txt").write_text("tracked\n")
            subprocess.run(["git", "-C", str(workspace), "add", "tracked.txt"], check=True)

            environment = os.environ.copy()
            environment.update(
                {
                    "GITHUB_ACTIONS": "true",
                    "GITHUB_WORKSPACE": str(workspace),
                    "GIT_CONFIG_GLOBAL": str(home / ".gitconfig"),
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1",
                }
            )

            unsafe = subprocess.run(
                ["git", "-C", str(workspace), "ls-files"],
                text=True,
                capture_output=True,
                check=False,
                env=environment,
            )
            configured = subprocess.run(
                [str(HELPER)],
                text=True,
                capture_output=True,
                check=False,
                env=environment,
            )
            safe = subprocess.run(
                ["git", "-C", str(workspace), "ls-files"],
                text=True,
                capture_output=True,
                check=False,
                env=environment,
            )
            safe_directories = subprocess.run(
                ["git", "config", "--global", "--get-all", "safe.directory"],
                text=True,
                capture_output=True,
                check=True,
                env=environment,
            )

        self.assertEqual(unsafe.returncode, 128)
        self.assertIn("dubious ownership", unsafe.stderr)
        self.assertEqual(configured.returncode, 0, configured.stderr)
        self.assertEqual(safe.returncode, 0, safe.stderr)
        self.assertEqual(safe.stdout, "tracked.txt\n")
        self.assertEqual(safe_directories.stdout.splitlines(), [str(workspace)])


if __name__ == "__main__":
    unittest.main()
