"""Daily wiring and default-scope regressions without depending on PyYAML."""
import ast
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from tests.scripts.test_dependency_workflows import job_text

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/daily_test.yml"


class DailyWorkflowTests(unittest.TestCase):
    def test_regression_schedule_stays_off(self):
        text = WORKFLOW.read_text()
        self.assertEqual(re.findall(r'(?m)^    - cron: ["\x27]([^"\x27]+)', text),
                         ["01 08 * * 1-5", "23 3 * * 1"])
        self.assertIn("github.event_name != 'schedule'", job_text(WORKFLOW, "daily-plan"))

    def test_context_and_evidence_survive_failed_job_reruns(self):
        text = job_text(WORKFLOW, "daily-plan")
        self.assertIn("getWorkflowRun", text)
        self.assertIn("run.data.created_at", text)
        self.assertNotIn("github.run_attempt", text)
        self.assertIn("overwrite: true", text)
        for job in ("daily_test_asan_x86", "daily_test_tsan"):
            lane = job_text(WORKFLOW, job)
            self.assertNotIn("github.run_attempt", lane)
            self.assertIn("overwrite: true", lane)
        summary = job_text(WORKFLOW, "daily-summary")
        self.assertIn("daily-results-*-${{ github.run_id }}", summary)
        self.assertNotIn("datetime.now", text)

    def test_simd_selects_one_mode_but_keeps_integration_sentinels(self):
        build = job_text(WORKFLOW, "daily_build_simd")
        tests = job_text(WORKFLOW, "daily_test_asan_simd")
        self.assertIn("fromJSON(needs.daily-plan.outputs.simd_options)", build)
        self.assertIn("--target unittests_simd", build)
        self.assertIn("fromJSON(needs.daily-plan.outputs.simd_keys)", tests)
        self.assertIn("platform_smoke.py functests", tests)
        self.assertNotIn("test_type: [unittests, functests]", tests)

    def test_all_test_lanes_have_frozen_context_and_failure_evidence(self):
        text = job_text(WORKFLOW, "daily-plan")
        expected = ast.literal_eval(re.search(r"expected-lanes.json', (\[.*\])\)", text).group(1))
        self.assertEqual(len(expected), len(set(expected)))
        workflow = WORKFLOW.read_text()
        names = re.findall(r"(?m)^  ([a-z0-9_-]+):$", workflow.split("jobs:\n", 1)[1])
        for name in names:
            job = job_text(WORKFLOW, name)
            if "VSAG_DAILY_CONTEXT:" in job:
                self.assertIn("Download frozen Daily plan", job, name)
                self.assertIn("Upload Daily evidence", job, name)
                self.assertRegex(job, r"Upload Daily evidence\n\s+if: always\(\)")
                self.assertTrue(any(lane.startswith(name + "-") for lane in expected), name)
        summary = job_text(WORKFLOW, "daily-summary")
        self.assertIn("always()", summary)
        self.assertIn("--expected-file", summary)

    def test_summary_waits_for_lint_and_receives_its_result(self):
        summary = job_text(WORKFLOW, "daily-summary")
        dependencies = re.search(r"needs: \[(.*?)\]", summary).group(1)
        self.assertIn("full-lint-clang-tidy-check", dependencies.split(", "))
        self.assertIn("always()", summary)
        self.assertIn("NEEDS_JSON: ${{ toJSON(needs) }}", summary)

    def test_examples_use_manifest_not_shell_glob(self):
        for arch in ("x86", "aarch64"):
            action = (ROOT / f".github/actions/asan-build-and-test-test-example-{arch}/action.yml").read_text()
            self.assertIn("platform_smoke.py all-examples", action)
            self.assertNotIn("[[", action)
            self.assertNotIn("for example in", action)

    def test_reusable_performance_caller_grants_read_only_actions(self):
        caller = job_text(ROOT / ".github/workflows/manual-checks.yml", "performance")
        self.assertIn("actions: read", caller)
        self.assertIn("contents: read", caller)
        self.assertNotIn("actions: write", WORKFLOW.read_text())
        self.assertNotIn("actions: write", caller)

    def test_suppressions_survive_isolated_working_directories(self):
        sources = [WORKFLOW] + [ROOT / f".github/actions/asan-build-and-test-test-asan-{arch}/action.yml"
                                for arch in ("x86", "aarch64")]
        assignments = []
        for source in sources:
            assignments += re.findall(r"export (?:LSAN|TSAN)_OPTIONS=.*", source.read_text())
        self.assertEqual(len(assignments), 6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "omp.supp").write_text("leak:libomp\n")
            (root / "isolated").mkdir()
            for assignment in assignments:
                variable = assignment.split()[1].split("=", 1)[0]
                script = assignment + '\ncd isolated\nvalue="${' + variable + '}"\ntest -f "${value#suppressions=}"'
                process = subprocess.run(["bash", "-c", script], cwd=root, capture_output=True)
                self.assertEqual(process.returncode, 0, (assignment, process.stderr))

    def test_non_daily_measure_preserves_exit_and_working_directory(self):
        import sys
        script = ROOT / "scripts/testing/daily_measure.py"
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.run(
                [sys.executable, str(script), "--name", "check", "--owner", "simd",
                 "--cwd", directory, "--", sys.executable, "-c",
                 "import pathlib;pathlib.Path('marker').write_text('ok');raise SystemExit(7)"],
                capture_output=True)
            self.assertEqual(process.returncode, 7)
            self.assertEqual((Path(directory) / "marker").read_text(), "ok")


if __name__ == "__main__":
    unittest.main()
