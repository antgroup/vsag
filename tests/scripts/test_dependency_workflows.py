#!/usr/bin/env python3
"""Regression checks for dependency workflow trust and source guards."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


def job_text(workflow: Path, job_name: str) -> str:
    text = workflow.read_text(encoding="utf-8")
    match = re.search(
        rf"(?ms)^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)", text
    )
    if match is None:
        raise AssertionError(f"job {job_name} not found in {workflow}")
    return match.group("body")


class DependencyWorkflowTest(unittest.TestCase):
    def test_metrics_helpers_preserve_dedicated_coverage_checks(self) -> None:
        workflow = ROOT / ".github/workflows/pr-ci.yml"
        coverage = job_text(workflow, "coverage-config")
        asan = job_text(workflow, "build-asan-x86")

        self.assertIn("tests/scripts/coverage_diagnostics_test.py", coverage)
        self.assertIn("scripts/coverage/collect_cpp_coverage.sh", coverage)
        self.assertNotIn("tests/scripts/coverage_diagnostics_test.py", asan)
        self.assertIn("tests/scripts/test_collect_build_metrics.py", asan)
        self.assertIn("tests/scripts/test_dependency_workflows.py", asan)

    def test_daily_x86_build_explicitly_disables_system_openblas(self) -> None:
        daily_x86 = job_text(
            ROOT / ".github/workflows/daily_test.yml", "daily_build_asan_x86"
        )

        self.assertIn("-DVSAG_USE_SYSTEM_OPENBLAS=OFF", daily_x86)
        self.assertNotIn("-DVSAG_USE_SYSTEM_OPENBLAS=AUTO", daily_x86)

    def test_pull_request_archive_cache_is_restore_only(self) -> None:
        pr_ci = (ROOT / ".github/workflows/pr-ci.yml").read_text(encoding="utf-8")

        self.assertIn("uses: actions/cache/restore@v4", pr_ci)
        self.assertNotIn("uses: actions/cache@v4", pr_ci)
        self.assertNotIn("uses: actions/cache/save@v4", pr_ci)
        self.assertIn("dependency-cache-write-policy restore-only", pr_ci)

    def test_external_archives_keep_hash_validation_and_pin_overrides(self) -> None:
        for dependency in ("antlr4", "hdf5", "openblas"):
            recipe = (ROOT / f"extern/{dependency}/{dependency}.cmake").read_text(
                encoding="utf-8"
            )
            with self.subTest(dependency=dependency):
                self.assertRegex(recipe, r"URL_HASH\s+(?:MD5|SHA256)=")
                self.assertIn("vsag_resolve_thirdparty_override", recipe)


if __name__ == "__main__":
    unittest.main()
