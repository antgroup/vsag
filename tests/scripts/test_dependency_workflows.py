#!/usr/bin/env python3
"""Regression checks for dependency workflow trust and source guards."""

from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
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
        workflow = ROOT / ".github/workflows/pr-ci-community.yml"
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
        pr_ci = (ROOT / ".github/workflows/pr-ci-community.yml").read_text(encoding="utf-8")

        self.assertIn("uses: actions/cache/restore@v4", pr_ci)
        self.assertNotIn("uses: actions/cache@v4", pr_ci)
        self.assertNotIn("uses: actions/cache/save@v4", pr_ci)
        benchmark = (ROOT / ".github/workflows/build_performance.yml").read_text(encoding="utf-8")
        self.assertIn("dependency-cache-write-policy restore-only", benchmark)

    def test_archive_directory_is_exported_for_the_entire_metrics_job(self) -> None:
        asan = job_text(ROOT / ".github/workflows/pr-ci-community.yml", "build-asan-x86")
        job_configuration, steps = asan.split("    steps:\n", 1)
        self.assertIn(
            "      VSAG_THIRDPARTY_DOWNLOAD_DIR: ${{ github.workspace }}/.ci-downloads\n",
            job_configuration,
        )
        self.assertIn("          path: .ci-downloads\n", steps)

    def test_team_compiler_cache_is_private_and_never_published_by_pr(self) -> None:
        team = job_text(ROOT / ".github/workflows/pr-ci-team.yml", "build")
        self.assertIn('cache="$RUNNER_TEMP/team-ccache"', team)
        self.assertIn('team_cache.py restore', team)
        self.assertIn('--source /__ci/cache/snapshot --manifest /__ci/cache/manifest.json', team)
        self.assertIn('--namespace-file ci-logs/cache-namespace.json', team)
        self.assertIn('CCACHE_BASEDIR=$GITHUB_WORKSPACE', team)
        self.assertIn('CCACHE_COMPILERCHECK=content', team)
        self.assertNotIn('team_cache.py manifest', team)
        self.assertNotIn('actions/cache/save', team)
        self.assertNotIn('actions/cache@', team)

    def test_external_archives_keep_hash_validation_and_pin_overrides(self) -> None:
        for dependency in ("antlr4", "hdf5", "openblas"):
            recipe = (ROOT / f"extern/{dependency}/{dependency}.cmake").read_text(
                encoding="utf-8"
            )
            with self.subTest(dependency=dependency):
                self.assertRegex(recipe, r"URL_HASH\s+(?:MD5|SHA256)=")
                self.assertIn("vsag_resolve_thirdparty_override", recipe)


class TsanCleanupTest(unittest.TestCase):
    def cleanup_script(self, sandbox: Path) -> str:
        workflow = ROOT / ".github/workflows/tsan_build_and_test.yml"
        build = job_text(workflow, "build_tsan")
        self.assertIn("        - /opt:/useless\n", build)
        match = re.search(
            r"(?ms)^      - name: Free Disk Space\n        run: \|\n"
            r"(?P<script>.*?)(?=^      - |\Z)", build
        )
        self.assertIsNotNone(match)
        # Execute the workflow block itself. Only map its destructive target to
        # our temporary directory; diagnostic commands never touch the host.
        script = textwrap.dedent(match.group("script"))
        self.assertIn("node24_diagnostics before\n", script)
        self.assertIn("node24_diagnostics after\n", script)
        diagnostics = "\n".join(
            f"{command}() {{ return 1; }}"
            for command in ("date", "ls", "readlink", "stat", "awk", "timeout")
        )
        return diagnostics + "\n" + script.replace("/useless", str(sandbox))

    def run_cleanup(self, sandbox: Path, prefix: str = "") -> subprocess.CompletedProcess:
        return subprocess.run(
            ["sh", "-e", "-c", prefix + self.cleanup_script(sandbox)],
            text=True, capture_output=True, timeout=10,
        )

    def test_preserves_runner_runtime_and_hidden_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = Path(temporary)
            node = sandbox / "runner-cache/externals/node24/bin/node"
            node.parent.mkdir(parents=True)
            node.write_text("runtime\n")
            for name in ("tools", "directory with spaces", "runner-cache-old"):
                directory = sandbox / name
                directory.mkdir()
                (directory / "payload").write_text("delete me\n")
            for name in ("file", "file with spaces", "-dash", ".hidden"):
                (sandbox / name).write_text("payload\n")
            hidden = sandbox / ".hidden-directory"
            hidden.mkdir()
            (hidden / "payload").write_text("keep me\n")

            result = self.run_cleanup(sandbox)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(node.read_text(), "runtime\n")
            self.assertEqual(
                {path.name for path in sandbox.iterdir()},
                {"runner-cache", ".hidden", ".hidden-directory"},
            )
            self.assertEqual((hidden / "payload").read_text(), "keep me\n")
            self.assertIn("(before cleanup)", result.stdout)
            self.assertIn("(after cleanup)", result.stdout)

    def test_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = Path(temporary)
            result = self.run_cleanup(sandbox)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(sandbox.iterdir()), [])
            self.assertIn("(after cleanup)", result.stdout)

    def test_cleanup_failure_propagates_before_next_entry_or_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = Path(temporary)
            (sandbox / "a").mkdir()
            (sandbox / "b").mkdir()
            result = self.run_cleanup(
                sandbox, 'rm() { echo "cleanup failed: $*" >&2; return 23; };\n'
            )
            self.assertEqual(result.returncode, 23, result.stderr)
            self.assertEqual(result.stderr.count("cleanup failed:"), 1)
            self.assertIn(str(sandbox / "a"), result.stderr)
            self.assertNotIn("(after cleanup)", result.stdout)
            self.assertEqual({path.name for path in sandbox.iterdir()}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
