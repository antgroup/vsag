#!/usr/bin/env python3
"""Check the Aarch64 ASan download contract without executing ARM binaries."""

from pathlib import Path
import os
import re
import subprocess
import tempfile
import textwrap
import unittest

from tests.scripts.test_dependency_workflows import job_text


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/main-ci.yml"
# v6's toolkit rejects its inactivity timeout; v5 can exit zero with a pending promise.
DOWNLOAD_ACTION = "actions/download-artifact@018cc2cf5baa6db3ef3c5f8a56943fffe632ef53"
JOBS = ("platforms-test-asan-aarch64", "platforms-test-example-aarch64")


def validation_script(job: str) -> str:
    match = re.search(
        r"(?ms)^      - name: Validate Test Artifact\n        run: \|\n"
        r"(?P<script>.*?)(?=^      - |\Z)",
        job_text(WORKFLOW, job),
    )
    if match is None:
        raise AssertionError(f"{job} must validate downloaded binaries before running tests")
    return textwrap.dedent(match.group("script"))


class AsanArtifactWorkflowTest(unittest.TestCase):
    def test_consumers_use_timeout_rejecting_action_and_validate_before_execution(self):
        for name in JOBS:
            with self.subTest(job=name):
                job = job_text(WORKFLOW, name)
                self.assertIn(f"uses: {DOWNLOAD_ACTION}", job)
                self.assertLess(job.index("name: Download Test"), job.index("name: Validate Test Artifact"))
                self.assertLess(job.index("name: Validate Test Artifact"), job.index("name: Do "))
                if name == "platforms-test-example-aarch64":
                    self.assertIn("inputs.artifact-name || format('test_aarch64-{0}', github.run_id)", job)
                    action = (ROOT / ".github/actions/asan-build-and-test-test-example-aarch64/action.yml").read_text()
                    self.assertIn("artifact-name:\n    description: Optional caller-provided build artifact\n    required: false\n    default: ''", action)
                else:
                    self.assertIn("name: test_aarch64-${{ github.run_id }}", job)
                self.assertIn("path: ./build", job)

    def check_artifact(self, files, expected_success):
        for job in JOBS:
            with self.subTest(job=job), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name, contents in files.items():
                    file = root / name
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.write_bytes(contents)
                result = subprocess.run(
                    ["sh", "-e", "-c", validation_script(job)],
                    cwd=root, text=True, capture_output=True, timeout=10,
                    env=dict(os.environ, LC_ALL="C"),
                )
                self.assertEqual(result.returncode == 0, expected_success, result.stdout + result.stderr)
                if not expected_success:
                    self.assertIn("::error::Aarch64 test artifact is incomplete", result.stdout)
                    self.assertIn("Filesystem", result.stdout)  # disk diagnostic survived missing build/

    def test_complete_artifact_passes_without_executing_binaries(self):
        self.check_artifact({"build/tests/unittests": b"ARM fixture", "build/tests/functests": b"ARM fixture"}, True)

    def test_absent_download_fails(self):
        self.check_artifact({}, False)

    def test_partial_download_fails(self):
        self.check_artifact({"build/tests/functests": b"ARM fixture"}, False)

    def test_empty_binary_fails(self):
        self.check_artifact({"build/tests/unittests": b"", "build/tests/functests": b"ARM fixture"}, False)

    def test_nested_build_layout_fails(self):
        self.check_artifact({"build/build/tests/unittests": b"ARM fixture", "build/build/tests/functests": b"ARM fixture"}, False)


if __name__ == "__main__":
    unittest.main()
