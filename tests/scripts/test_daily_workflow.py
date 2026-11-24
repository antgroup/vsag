"""Daily wiring and default-scope regressions without depending on PyYAML."""
import ast
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.testing import daily_entry, daily_plan
from tests.scripts.test_dependency_workflows import job_text

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/daily_test.yml"


def job_dependencies(body):
    body = re.sub(r"(?m)[ \t]+#.*$", "", body)
    match = re.search(r"(?m)^    needs:[ \t]*(\[[^]]*\]|(?:\n      - [^\n]+)+|[^\n]*)", body)
    return set(re.findall(r"[a-z0-9_][a-z0-9_-]*", match[1])) if match else set()


class DailyWorkflowTests(unittest.TestCase):
    def test_extra_stage_is_separate_bounded_and_reuses_the_original_build(self):
        plan = job_text(WORKFLOW, "daily-extra-plan")
        self.assertIn("daily_test_asan_x86", plan)
        self.assertIn("full-coverage-coverage", plan)
        self.assertIn("full-lint-clang-tidy-check", plan)
        self.assertNotIn("always() &&", plan)
        extra = job_text(WORKFLOW, "daily-extra-x86")
        self.assertIn("daily_test_x86-${{ github.run_id }}", extra)
        self.assertIn("daily-extra-plan", extra)
        self.assertIn("timeout-minutes: 270", extra)
        self.assertNotIn("make asan", extra)
        self.assertNotIn("continue-on-error", extra)
        summary = job_text(WORKFLOW, "daily-summary")
        self.assertIn("daily-extra-x86", summary)
        self.assertIn("--extra-proposal", summary)
        download = summary.split("- name: Download extra quota plan\n", 1)[1].split("- name:", 1)[0]
        self.assertIn("if: ${{ always() && needs.daily-extra-plan.result != 'skipped' }}", download)
        self.assertNotIn("continue-on-error", download)
        self.assertIn("--job-reports", summary)

    def test_job_dependencies_accepts_scalar_block_and_multiline_flow(self):
        for needs, expected in (
                ("daily-plan", {"daily-plan"}),
                ("\n      - daily-plan\n      - daily_build_asan_x86", {"daily-plan", "daily_build_asan_x86"}),
                ("[daily-plan, daily_build_asan_x86]", {"daily-plan", "daily_build_asan_x86"}),
                ("[daily-plan,\n      daily_build_asan_x86 # build status\n    ]",
                 {"daily-plan", "daily_build_asan_x86"})):
            with self.subTest(needs=needs):
                self.assertEqual(job_dependencies(f"    needs: {needs}\n    steps:\n"), expected)
        self.assertEqual(job_dependencies("    steps:\n"), set())

    def test_summary_receives_all_supplemental_prerequisite_results(self):
        def dependencies(name):
            return job_dependencies(job_text(WORKFLOW, name))

        prerequisites = dependencies("daily-extra-x86")
        pending = list(prerequisites)
        while pending:
            unseen = dependencies(pending.pop()) - prerequisites
            prerequisites.update(unseen)
            pending.extend(unseen)
        self.assertIn("daily_build_asan_x86", prerequisites)
        self.assertIn("full-platform-build-asan-macos-arm64", prerequisites)
        self.assertLessEqual(prerequisites | {"daily-extra-x86"}, dependencies("daily-summary"))

    def test_only_final_regression_job_publishes_cache_summary(self):
        workflow = WORKFLOW.read_text()
        self.assertIn("VSAG_DAILY_AGGREGATE:", workflow)
        for p in [WORKFLOW] + list((ROOT / '.github/actions').glob('*/action.yml')):
            text = p.read_text()
            if 'hendrikmuhs/ccache-action@v1.2.25' in text:
                self.assertIn("job-summary: ${{ env.VSAG_DAILY_AGGREGATE != '1'", text, str(p))
        self.assertIn('name: Nightly Summary', job_text(WORKFLOW,'daily-summary'))
        self.assertIn('name: Coverage — C++ lines and branches',job_text(WORKFLOW,'full-coverage-coverage'))
        for name in ('performance-performance', 'build-performance-benchmark-x86'):
            self.assertNotIn('daily-job-report', job_text(WORKFLOW, name))
        readme=(ROOT/'README.md').read_text()
        self.assertIn('Nightly-10h%20regression',readme)
        self.assertNotIn('img.shields.io/circleci/',readme)

    def test_source_preflight_gates_heavy_regression_jobs(self):
        plan = job_text(WORKFLOW, "daily-plan")
        self.assertIn("run: python3 scripts/testing/daily_parameters.py", plan)
        self.assertLess(plan.index("Preflight Nightly parameter inventory"),
                        plan.index("Freeze source, date and matrix"))
        self.assertNotIn("continue-on-error", plan)
        for name in ("daily_build_asan_x86", "daily_build_asan_aarch64", "daily_build_simd",
                     "daily_build_tsan", "full-platform-build-asan-x86",
                     "full-platform-build-asan-macos-arm64", "full-python-build-python-wheel-x86",
                     "full-python-build-python-wheel-aarch64", "full-lint-clang-tidy-check",
                     "full-coverage-coverage", "full-compatibility-compatibility"):
            body = job_text(WORKFLOW, name)
            needs = re.search(r"(?m)^    needs:([^\n]*(?:\n      - [^\n]+)*)", body)
            self.assertIsNotNone(needs, name)
            self.assertIn("daily-plan", needs[1], name)

    def test_shared_arm_artifact_and_x86_shard_artifact_names(self):
        workflow = WORKFLOW.read_text()
        self.assertNotIn("  full-platform-build-asan-aarch64:", workflow)
        self.assertNotIn("  full-platform-test-asan-aarch64:", workflow)
        self.assertNotIn("  asan_test_clean_up:", workflow)
        example = job_text(WORKFLOW, "full-platform-test-example-aarch64")
        self.assertIn("daily_build_asan_aarch64", example)
        self.assertIn("artifact-name: daily_test_aarch64-${{ github.run_id }}", workflow)
        x86 = job_text(WORKFLOW, "daily_test_asan_x86")
        self.assertIn("shard: ${{ fromJSON(needs.daily-plan.outputs.x86_shards) }}", x86)
        self.assertIn("timeout-minutes: 300", x86)
        self.assertIn("VSAG_DAILY_SHARD_COUNT: ${{ needs.daily-plan.outputs.x86_shard_count }}", x86)
        self.assertIn("-shard${{ matrix.shard }}-${{ github.run_id }}", x86)
        self.assertIn("VSAG_ENABLE_INTEL_MKL=ON", (ROOT / ".github/actions/asan-build-and-test-build-asan-x86/action.yml").read_text())
        self.assertIn("-DVSAG_USE_SYSTEM_OPENBLAS=OFF", job_text(WORKFLOW, "daily_build_asan_x86"))

    def test_x86_functional_workflow_enables_rotation_timeout_warnings(self):
        x86 = job_text(WORKFLOW, "daily_test_asan_x86")
        env = dict(re.findall(r"(?m)^      (VSAG_DAILY_(?:LANE|ENVIRONMENT)): ([^\s]+)$", x86))
        shard_count = daily_plan.policy()["x86_shards"]
        # daily_run omits sharding metadata for an unsharded plan.
        self.assertGreater(shard_count, 1)
        self.assertIn("VSAG_DAILY_SHARD_INDEX: ${{ matrix.shard }}", x86)
        self.assertIn("VSAG_DAILY_SHARD_COUNT: ${{ needs.daily-plan.outputs.x86_shard_count }}", x86)
        env.update(VSAG_DAILY_CONTEXT="context.json", VSAG_DAILY_SHARD_INDEX="0",
                   VSAG_DAILY_SHARD_COUNT=str(shard_count))
        timeout = {"status": "timeout", "timeout_scope": "case", "exit_code": -signal.SIGTERM}
        with (patch.dict(os.environ, env, clear=True),
              patch.object(daily_entry.platform, "system", return_value="Linux"),
              patch.object(daily_entry.platform, "machine", return_value="x86_64")):
            for binary, advisory in (("functests", True), ("unittests", False)):
                with self.subTest(binary=binary):
                    argv = daily_entry.command(binary)
                    plan = {"environment": argv[argv.index("--environment") + 1],
                            "sharding": {"index": int(argv[argv.index("--shard-index") + 1])}}
                    lane = argv[argv.index("--lane") + 1]
                    self.assertEqual(
                        daily_plan.advisory_rotation_timeout(plan, lane, {"mode": "rotation"}, timeout),
                        advisory,
                    )

    def test_long_build_and_lint_stages_use_remaining_window(self):
        text = WORKFLOW.read_text()
        for name in ("daily_build_asan_x86", "daily_build_asan_aarch64", "daily_build_simd",
                     "daily_build_tsan", "full-platform-build-asan-x86", "full-platform-build-asan-macos-arm64",
                     "full-python-build-python-wheel-x86", "full-python-build-python-wheel-aarch64",
                     "full-lint-clang-tidy-check"):
            body = re.search(r"(?ms)^  " + name + r":\n.*?(?=^  [a-zA-Z0-9_-]+:|^permissions:)", text).group()
            self.assertIn("id: window", body, name)
            self.assertIn("timeout-minutes: ${{ fromJSON(steps.window.outputs.remaining_minutes) }}", body, name)
        build = job_text(WORKFLOW, "daily_build_asan_x86")
        self.assertLess(build.index("Load Cache"), build.index("Check build/lint execution window"))

    def test_nightly_schedule_and_display_name(self):
        text = WORKFLOW.read_text()
        self.assertEqual(re.findall(r'(?m)^    - cron: ["\x27]([^"\x27]+)', text),
                         ["0 14 * * *", "23 3 * * 1"])
        self.assertTrue(text.startswith("name: Nightly CI\n"))
        self.assertNotIn("01 08 * * 1-5", text)

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
        expected += [f'daily_test_asan_x86-{binary}-shard{i}'
                     for binary in ('unittests', 'functests') for i in range(json.loads((ROOT/'scripts/testing/daily_budget_policy.json').read_text())['x86_shards'])]
        from scripts.ci import python_ci
        expected += python_ci.lanes()
        self.assertEqual(len(expected), len(set(expected)))
        workflow = WORKFLOW.read_text()
        names = re.findall(r"(?m)^  ([a-z0-9_-]+):$", workflow.split("jobs:\n", 1)[1])
        for name in names:
            job = job_text(WORKFLOW, name)
            if "VSAG_DAILY_CONTEXT:" in job:
                self.assertIn("Download frozen Nightly plan", job, name)
                self.assertIn("uses: ./.github/actions/daily-execution-window", job, name)
                self.assertIn("Upload Nightly evidence", job, name)
                self.assertRegex(job, r"Upload Nightly evidence\n\s+if: always\(\)")
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
        self.assertEqual(len(assignments), 7)
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
