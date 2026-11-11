"""Execute selection/summary/smoke contracts without building the C++ library."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/ci" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class MainPolicyTest(unittest.TestCase):
    def test_classification(self):
        m = module("main_ci_policy")
        for paths in ([], ["docs/guide.txt", "README.md"]):
            self.assertEqual(m.classify(paths)["heavy"], "false")
        for paths in (["include/vsag/index.h"], ["src/CMakeLists.txt"], ["unknown"], ["README.md", "cmake/toolchain.cmake"], ["src/impl/detail.hpp"], ["src/odd\nname.cpp"]):
            self.assertEqual(m.classify(paths)["lint"], "full")
        self.assertEqual(m.classify(["tests/test_factory.cpp"])["lint"], "none")
        with patch.object(m.Path, "is_file", return_value=True):
            out = m.classify(["src/a.cpp", "src/b_test.cpp"])
            self.assertEqual(out["lint"], "changed")
            self.assertEqual(out["pattern"], r"src/a\.cpp$")

    def test_missing_base_and_transport_failure_select_full(self):
        m = module("main_ci_policy")
        self.assertEqual(m.select("0" * 40, "a" * 40)["lint"], "full")
        with patch.object(m.subprocess, "check_output", side_effect=subprocess.CalledProcessError(1, "git")):
            self.assertEqual(m.select("b" * 40, "a" * 40)["lint"], "full")

    def test_whole_push_includes_earlier_commit_and_deleted_file(self):
        m = module("main_ci_policy")
        with tempfile.TemporaryDirectory() as tmp:
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=tmp, stderr=subprocess.DEVNULL, text=True).strip()
            git("init"); git("config", "user.name", "Test"); git("config", "user.email", "test@example.invalid")
            Path(tmp, "README.md").write_text("base"); git("add", "."); git("commit", "-m", "base"); base = git("rev-parse", "HEAD")
            Path(tmp, "CMakeLists.txt").write_text("change"); git("add", "."); git("commit", "-m", "build")
            Path(tmp, "README.md").write_text("docs"); git("add", "."); git("commit", "-m", "docs"); head = git("rev-parse", "HEAD")
            old = os.getcwd()
            try:
                os.chdir(tmp)
                self.assertEqual(m.select(base, head)["lint"], "full")
                Path("CMakeLists.txt").unlink(); git("add", "."); git("commit", "-m", "delete")
                self.assertEqual(m.select(head, git("rev-parse", "HEAD"))["lint"], "full")
            finally:
                os.chdir(old)

    def test_summary_rejects_failed_cancelled_missing_or_skipped_selected(self):
        m = module("main_ci_summary")
        needs = {"changes": {"result": "success", "outputs": {"heavy": "true", "python": "true", "lint": "changed"}}}
        needs.update({n: {"result": "success"} for n in ("platforms", "python", "tsan", "lint")})
        self.assertTrue(m.validate(needs))
        for name in needs:
            for state in ("failure", "cancelled", "skipped"):
                bad = copy.deepcopy(needs); bad[name]["result"] = state
                self.assertFalse(m.validate(bad))
            bad = copy.deepcopy(needs); del bad[name]; self.assertFalse(m.validate(bad))
        needs["changes"]["outputs"] = {"heavy": "false", "python": "false", "lint": "none"}
        for n in ("platforms", "python", "tsan", "lint"): needs[n]["result"] = "skipped"
        self.assertTrue(m.validate(needs))

    def test_main_python_pause_preserves_other_selection_and_manual_checks(self):
        m = module("main_ci_policy")
        for heavy in ("true", "false"):
            policy = {"heavy": heavy, "lint": "full" if heavy == "true" else "none", "pattern": ""}
            for event, ref, paused in (("push", "refs/heads/main", True),
                                       ("push", "refs/heads/0.18", False),
                                       ("workflow_dispatch", "refs/heads/main", False)):
                with patch.object(m, "select", return_value=dict(policy)):
                    result = m.select_for_event("a" * 40, "b" * 40, event, ref)
                self.assertEqual(result["python"], "false" if paused else heavy)
                self.assertEqual({k: result[k] for k in policy}, policy)
        # A missing diff still retains other heavy checks; the intentional pause remains.
        result = m.select_for_event("0" * 40, "a" * 40, "push", "refs/heads/main")
        self.assertEqual((result["heavy"], result["lint"], result["python"]), ("true", "full", "false"))

    def test_paused_python_summary_requires_explicit_skip_and_other_success(self):
        m = module("main_ci_summary")
        needs = {"changes": {"result": "success", "outputs": {"heavy": "true", "python": "false", "lint": "full"}},
                 "python": {"result": "skipped"}}
        needs.update({n: {"result": "success"} for n in ("platforms", "tsan", "lint")})
        self.assertTrue(m.validate(needs))
        for name in ("python", "platforms", "tsan", "lint"):
            for state in ("failure", "cancelled", "missing"):
                bad = copy.deepcopy(needs)
                bad[name]["result"] = state
                self.assertFalse(m.validate(bad))
        for policy in (None, "", "invalid"):
            bad = copy.deepcopy(needs)
            bad["changes"]["outputs"]["python"] = policy
            self.assertFalse(m.validate(bad))
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp, "summary.md")
            with patch.dict(os.environ, MAIN_NEEDS=json.dumps(needs), GITHUB_STEP_SUMMARY=str(summary)):
                m.main()
            self.assertIn("temporarily paused", summary.read_text())
            self.assertIn("skipped is not validation success", summary.read_text())

    def test_smoke_report_cannot_pass_empty_skipped_or_failed(self):
        m = module("platform_smoke")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, "result.xml")
            p.write_text('<testsuites><testsuite><testcase name="actual"/></testsuite></testsuites>'); m.verify_report(p)
            for xml in ('<testsuites/>', '<testsuite><testcase><skipped/></testcase></testsuite>', '<testsuite><testcase><failure/></testcase></testsuite>', '<testsuite><testcase><error/></testcase></testsuite>'):
                p.write_text(xml)
                with self.assertRaises(ValueError): m.verify_report(p)

    def test_smoke_missing_binary_and_failing_process_are_fatal(self):
        m = module("platform_smoke")
        with tempfile.TemporaryDirectory() as tmp:
            old = os.getcwd()
            try:
                os.chdir(tmp)
                with self.assertRaises(ValueError): m.run("unittests")
                Path("build/tests").mkdir(parents=True); Path("build/tests/unittests").write_text("fake")
                with patch.object(m.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "test")):
                    with self.assertRaises(subprocess.CalledProcessError): m.run("unittests")
            finally: os.chdir(old)

    def test_smoke_inventory_exists_in_source(self):
        data = json.loads((ROOT / "scripts/ci/platform_smoke.json").read_text())
        for kind, folder in (("unittests", "src"), ("functests", "tests")):
            source = "\n".join(p.read_text() for p in (ROOT / folder).rglob("*test*.cpp"))
            for name in data[kind]: self.assertIn('"' + name + '"', source)
        for name in data["examples"]: self.assertTrue((ROOT / "examples/cpp" / (name + ".cpp")).is_file())

    def test_daily_full_regression_remains_manual(self):
        text = (ROOT / ".github/workflows/daily_test.yml").read_text()
        self.assertIn("github.event_name != 'schedule'", text)
        for profile in ('full-platform', 'full-python', 'full-lint', 'full-coverage', 'full-compatibility'):
            self.assertIn('  ' + profile + '-', text)
        self.assertNotRegex(text, r"(?m)^    - cron: [\"']0 14 ")

    def test_main_uses_only_light_contract_and_no_branch_wide_cancellation(self):
        text = (ROOT / ".github/workflows/main-ci.yml").read_text()
        self.assertRegex(text, r"(?s)push:.*?branches:.*?main.*?0\.\*")
        self.assertNotIn('\nconcurrency:', text)
        self.assertIn("inputs-smoke:", text)
        self.assertIn("python-build-python-wheel-aarch64:", text)
        self.assertIn("needs.entry-scope.outputs.profile == 'full-branch'", text)
        self.assertIn("github.event.before", text)

    def test_actual_macos_shell_selects_exactly_one_scope(self):
        import textwrap
        from tests.scripts.test_build_benchmark_workflow import job, step
        from tests.scripts.test_dependency_workflows import job_text
        workflow = ROOT / ".github/workflows/main-ci.yml"
        shell = textwrap.dedent(step(job_text(workflow, "platforms-test-asan-macos-arm64"), "Do Asan Test In ${{ matrix.test_type }}").split("run: |\n", 1)[1]).replace("${{ matrix.test_type }}", "unittests")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, "scripts/testing"); p.mkdir(parents=True)
            script = p / "test_parallel_by_name.sh"; script.write_text("#!/bin/sh\necho FULL\n"); script.chmod(0o755)
            stub = 'chmod() { :; }; python3() { echo "$*"; };\n'
            for smoke, full_examples, full, calls in (("true", "false", False, 2), ("false", "false", True, 0), ("false", "true", True, 1)):
                out = subprocess.check_output(["bash", "-eu", "-c", stub + shell], cwd=tmp, env=dict(os.environ, SMOKE=smoke, FULL_EXAMPLES=full_examples), text=True)
                self.assertEqual("FULL" in out, full)
                self.assertEqual(out.count("platform_smoke.py"), calls)

    def test_version_and_manual_keep_full_profiles(self):
        branch = (ROOT / ".github/workflows/main-ci.yml").read_text()
        manual = (ROOT / ".github/workflows/manual-checks.yml").read_text()
        self.assertIn("needs.entry-scope.outputs.version == 'true'", branch)
        for profile in ('asan', 'python', 'lint', 'coverage', 'compatibility', 'tsan'):
            self.assertIn("needs.entry-scope.outputs.profile == '" + profile + "'", branch)
            self.assertIn(profile, manual)

    def test_full_examples_uses_independent_complete_inventory(self):
        m = module("platform_smoke")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "103_index_hgraph").write_text("binary")
            manifest = directory / "ci-example-targets.txt"
            with self.assertRaises(FileNotFoundError): m.full_examples(directory)
            manifest.write_text("103_index_hgraph\n105_index_brute_force\n")
            with self.assertRaisesRegex(ValueError, "Missing expected example"):
                m.full_examples(directory)
            (directory / "105_index_brute_force").write_text("binary")
            self.assertEqual(m.full_examples(directory), ["103_index_hgraph", "105_index_brute_force"])
            for bad in ("", "../escape\n", "103_index_hgraph\n103_index_hgraph\n"):
                manifest.write_text(bad)
                with self.assertRaises(ValueError): m.full_examples(directory)

    def test_cmake_inventory_contains_all_configured_numbered_targets(self):
        import shutil
        if not shutil.which("cmake"):
            self.skipTest("cmake unavailable")
        source = (ROOT / "examples/cpp/CMakeLists.txt").read_text()
        fragment = source[source.index("get_property(VSAG_CI_EXAMPLE_TARGETS"):]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "main.cpp").write_text("int main() { return 0; }\n")
            (p / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.18)\nproject(inventory CXX)\nadd_executable(103_first main.cpp)\nadd_executable(105_second main.cpp)\nadd_library(helper INTERFACE)\n" + fragment)
            subprocess.run(["cmake", "-S", tmp, "-B", str(p / "build")], check=True, capture_output=True)
            self.assertEqual((p / "build/ci-example-targets.txt").read_text().splitlines(), ["103_first", "105_second"])
            self.assertFalse((p / "build/103_first").exists())

    def test_failure_and_timeout_print_diagnostics_without_commands(self):
        import contextlib
        import io
        m = module("platform_smoke")
        for error in (subprocess.CalledProcessError(1, "test"), subprocess.TimeoutExpired("test", 300)):
            with tempfile.TemporaryDirectory() as tmp:
                old = os.getcwd()
                try:
                    os.chdir(tmp)
                    Path("build/tests").mkdir(parents=True); Path("build/tests/unittests").write_text("fake")
                    def fail(command, **kwargs):
                        kwargs["stdout"].write("::error::simulated sanitizer detail\n")
                        Path(command[-1]).write_text('<testsuite><testcase><failure>assertion detail</failure></testcase></testsuite>')
                        raise error
                    log = io.StringIO()
                    with patch.object(m.subprocess, "run", side_effect=fail), contextlib.redirect_stdout(log):
                        with self.assertRaises(type(error)): m.run("unittests")
                    self.assertIn("| ::error::simulated sanitizer detail", log.getvalue())
                    self.assertIn("assertion detail", log.getvalue())
                    self.assertNotIn("\n::error::", log.getvalue())
                finally: os.chdir(old)

    def test_suppression_precedes_example_smoke_and_logs_survive(self):
        from tests.scripts.test_build_benchmark_workflow import job
        from tests.scripts.test_dependency_workflows import job_text
        workflow = ROOT / ".github/workflows/main-ci.yml"
        body = job_text(workflow, "platforms-test-example-aarch64")
        self.assertLess(body.index('export LSAN_OPTIONS='), body.index('if [ "$SMOKE"'))
        self.assertIn('suppressions=$RUNNER_TEMP/', body)
        self.assertIn('if: always() && fromJSON(', body)
        for name in ("test_asan_aarch64", "test_asan_macos_arm64"):
            self.assertIn('name: Upload Log\n        if: always()', job_text(workflow, "platforms-" + name.replace("_", "-")))
        data = json.loads((ROOT / "scripts/ci/platform_smoke.json").read_text())
        self.assertNotIn("Test CPU Info", data["functests"])
        self.assertIn("Factory rejects removed index names", data["functests"])

    def test_pyramid_smoke_requires_search_assertions(self):
        inventory = json.loads((ROOT / "scripts/ci/platform_smoke.json").read_text())
        self.assertNotIn("107_index_pyramid", inventory["examples"])
        self.assertIn("Pyramid half input Build Add and search roundtrip", inventory["functests"])
        source = (ROOT / "tests/test_pyramid_fp16_poc.cpp").read_text()
        case = source.split('TEST_CASE("Pyramid half input Build Add and search roundtrip"', 1)[1].split('TEST_CASE(', 1)[0]
        self.assertIn("REQUIRE(built.has_value())", case)
        self.assertIn("CheckHalfSearch(index, data, BUILD_COUNT)", case)
        self.assertIn("CheckHalfRoundTrip(index, param, data, HALF_COUNT)", case)


if __name__ == "__main__":
    unittest.main()
