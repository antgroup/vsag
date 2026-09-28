#!/usr/bin/env python3

import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/coverage/compare_simd_coverage.py"
spec = importlib.util.spec_from_file_location("compare_simd", SCRIPT)
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)
SOURCE = "src/simd/example.cpp"


class ComparisonTest(unittest.TestCase):
    def test_exact_patch_threshold(self):
        lines = {line: line != 5 for line in range(1, 6)}
        result = coverage.compare({SOURCE: lines}, {SOURCE: lines}, {SOURCE: set(lines)})
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["patch"], {"covered": 4, "measured": 5})

    def test_patch_failure_even_when_scoped_total_improves(self):
        result = coverage.compare({SOURCE: {1: False, 2: False}},
                                  {SOURCE: {1: True, 2: False}}, {SOURCE: {2}})
        self.assertEqual(result["errors"], ["SIMD patch coverage is below 80%"])

    def test_non_regression_uses_unrounded_ratios(self):
        before = {line: line < 10000 for line in range(1, 10001)}
        after = dict(before)
        after[10001] = False
        self.assertIn("SIMD scoped line coverage regressed",
                      coverage.compare({SOURCE: before}, {SOURCE: after}, {})["errors"])

    def test_new_file_is_in_denominator_and_patch(self):
        new = "src/simd/new.cpp"
        result = coverage.compare({SOURCE: {1: True}},
                                  {SOURCE: {1: True}, new: {1: False}}, {new: {1}})
        self.assertEqual(len(result["errors"]), 2)
        self.assertEqual(result["new_files"], [new])

    def test_unmeasured_new_file_fails(self):
        result = coverage.compare({SOURCE: {1: True}}, {SOURCE: {1: True}},
                                  {"src/simd/new.h": {1}})
        self.assertIn("Unmeasured changed file", result["errors"][0])

    def test_only_actual_deletions_can_disappear(self):
        other = "src/simd/other.cpp"
        base = {SOURCE: {1: True}, other: {1: True}}
        head = {other: {1: True}}
        self.assertTrue(coverage.compare(base, head, {SOURCE: set()})["errors"])
        self.assertFalse(coverage.compare(base, head, {SOURCE: set()}, [SOURCE])["errors"])

    def test_no_measured_patch_is_not_100_percent(self):
        result = coverage.compare({SOURCE: {1: True}}, {SOURCE: {1: True}}, {SOURCE: {2}})
        self.assertEqual(result["patch"], {"covered": 0, "measured": 0})

    def test_unsupported_scope_is_explicit(self):
        result = coverage.compare({SOURCE: {1: True}}, {SOURCE: {1: True}},
                                  {"src/algorithm/new.cpp": {1}})
        self.assertEqual(result["unsupported"], ["src/algorithm/new.cpp"])

    def test_trace_deduplicates_instantiations_and_excludes_other_scopes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.info"
            path.write_text(f"SF:{SOURCE}\nDA:1,0\nend_of_record\n"
                            f"SF:{SOURCE}\nDA:1,2\nend_of_record\n"
                            "SF:src/other.cpp\nDA:1,3\nend_of_record\n")
            self.assertEqual(coverage.read_trace(path), {SOURCE: {1: True}})

    def test_missing_empty_and_invalid_traces_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.info"
            with self.assertRaises(OSError):
                coverage.read_trace(path)
            for content in ("", "SF:/tmp/source.cpp\n", "SF:src/../bad.cpp\n",
                            "DA:1,3\n", f"SF:{SOURCE}\nDA:1,-1\n"):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    coverage.read_trace(path)

    def test_git_diff_new_renamed_deleted_and_spaced_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def git(*args):
                return coverage.git(root, *args).decode().strip()
            git("init", "-q")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@example.com")
            old = root / SOURCE
            old.parent.mkdir(parents=True)
            old.write_text("old\n")
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            new = old.with_name("new file.cpp")
            old.rename(new)
            new.write_text("new\nsecond\n")
            git("add", "-A")
            git("commit", "-qm", "head")
            result = coverage.changed_lines(root, base, "HEAD")
            self.assertEqual(result, {SOURCE: set(), "src/simd/new file.cpp": {1, 2}})

    def test_metadata_mismatch_and_missing_baseline_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            reports = Path(temp)
            for metadata in (None, {"sha": "wrong", "profile": coverage.PROFILE}):
                if metadata:
                    (reports / "base.json").write_text(json.dumps(metadata))
                result = subprocess.run(["python3", str(SCRIPT), "--root", str(ROOT),
                                         "--base", "base", "--head", "head", "--reports", temp],
                                        capture_output=True)
                self.assertEqual(result.returncode, 1)
                self.assertIn("No passing coverage result", (reports / "summary.md").read_text())

    def test_cli_writes_a_scoped_result_for_real_git_revisions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def git(*args):
                return coverage.git(root, *args).decode().strip()
            git("init", "-q")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@example.com")
            source = root / SOURCE
            source.parent.mkdir(parents=True)
            source.write_text("before\n")
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            source.write_text("after\n")
            git("add", ".")
            git("commit", "-qm", "head")
            head = git("rev-parse", "HEAD")
            reports = root / "reports"
            reports.mkdir()
            for role, sha in (("base", base), ("head", head)):
                (reports / f"{role}.json").write_text(json.dumps({"sha": sha, "profile": coverage.PROFILE}))
                (reports / f"{role}.info").write_text(f"SF:{SOURCE}\nDA:1,1\nend_of_record\n")
            result = subprocess.run(["python3", str(SCRIPT), "--root", str(root),
                                     "--base", base, "--head", head, "--reports", str(reports)],
                                    capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((reports / "result.json").read_text())
            self.assertEqual(data["patch"], {"covered": 1, "measured": 1})
            self.assertIn("not full-project", (reports / "summary.md").read_text())

    def test_dependency_changes_stop_before_running_base_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def git(*args):
                return coverage.git(root, *args).decode().strip()
            git("init", "-q")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@example.com")
            dependency = root / "extern/dependency.cmake"
            dependency.parent.mkdir()
            dependency.write_text("before\n")
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            dependency.write_text("after\n")
            git("add", ".")
            git("commit", "-qm", "head")
            result = subprocess.run(["bash", str(ROOT / "scripts/coverage/run_simd_coverage.sh"),
                                     base, git("rev-parse", "HEAD")], cwd=root, capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertFalse((root / "simd-base").exists())
            self.assertIn("dependency definitions differ", (root / "simd-coverage/summary.md").read_text())

    def test_workflow_triggers_for_build_inputs_but_not_unrelated_paths(self):
        workflow = (ROOT / ".github/workflows/pr-simd-coverage.yml").read_text()
        path_block = workflow.split("    paths:\n", 1)[1].split("\npermissions:", 1)[0]
        patterns = re.findall(r"^      - '([^']+)'$", path_block, re.MULTILINE)

        def triggers(path):
            # The workflow uses literal paths, ** and **/ only. The latter
            # matches zero or more directories, including src/CMakeLists.txt.
            for pattern in patterns:
                expression = re.escape(pattern).replace(r"\*\*/", "(?:.*/)?")
                expression = expression.replace(r"\*\*", ".*")
                if re.fullmatch(expression, path):
                    return True
            return False

        for path in (
            "src/simd/fp32_simd.cpp", "src/simd/fp32_simd_test.cpp",
            "Makefile", "CMakeLists.txt", "cmake/VSAGCompilerConfig.cmake",
            "cmake/VSAGThirdPartyOverride.cmake", "cmake/CheckSIMDCompilerFlag.cmake",
            "cmake/GenerateVersionHeader.cmake", "cmake/vsagConfig.cmake.in",
            "src/CMakeLists.txt", "src/quantization/CMakeLists.txt",
            "src/impl/blas/CMakeLists.txt", "src/version.h.in",
            "tests/CMakeLists.txt", "tests/test_main.cpp",
            "tests/fixtures/CMakeLists.txt", "tests/fixtures/unittest.h",
            "tests/fixtures/framework/CMakeLists.txt",
            "extern/catch2/catch2.cmake", "extern/antlr4/antlr4.cmake",
            ".github/actions/prepare-fetchcontent/action.yml",
            ".github/workflows/pr-simd-coverage.yml",
            "scripts/coverage/verify_cpp_coverage.py",
            "tests/scripts/simd_coverage_test.py",
        ):
            with self.subTest(path=path):
                self.assertTrue(triggers(path))
        for path in (
            "README.md", "docs/docs/en/src/development/testing.md",
            "src/algorithm/hnswlib/hnswalg.cpp", "src/impl/diskann.cpp",
            "tests/test_index.cpp", "python/setup.py", "examples/cpp/CMakeLists.txt",
            "tools/CMakeLists.txt", "scripts/testing/test_parallel_bg.sh",
            ".github/actions/setup-cache/action.yml", ".github/workflows/release.yml",
        ):
            with self.subTest(path=path):
                self.assertFalse(triggers(path))

    def test_workflow_has_no_privileged_pr_or_cache_upload_path(self):
        workflow = (ROOT / ".github/workflows/pr-simd-coverage.yml").read_text()
        for forbidden in ("pull_request_target", "workflow_run", "secrets.", "actions/cache",
                          "ccache-action", "codecov-action", "id-token: write"):
            self.assertNotIn(forbidden, workflow)
        for required in ("contents: read", "persist-credentials: false", "timeout-minutes: 90",
                         "github.event.pull_request.base.sha", "github.sha"):
            self.assertIn(required, workflow)
        runner = (ROOT / "scripts/coverage/run_simd_coverage.sh").read_text()
        self.assertIn("COMPILE_JOBS=4", runner)
        self.assertIn("timeout 15m", runner)
        self.assertNotIn("--allow-running-no-tests", runner)

    def test_runner_compares_distinct_revisions_with_container_ownership(self):
        # Exercise real Git/worktrees, metadata, instrumentation verification and
        # comparison. Only the expensive build, test executable and collector are
        # fixtures; this is orchestration coverage, not production C++ coverage.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def git(*args):
                return coverage.git(root, *args).decode().strip()
            git("init", "-q")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@example.com")
            scripts = root / "scripts/coverage"
            scripts.mkdir(parents=True)
            for name in ("run_simd_coverage.sh", "compare_simd_coverage.py",
                         "verify_cpp_coverage.py"):
                shutil.copyfile(ROOT / "scripts/coverage" / name, scripts / name)
            source = root / SOURCE
            source.parent.mkdir(parents=True)
            source.write_text("before\n")
            (root / "Makefile").write_text("cov:\n\tpython3 fixture.py\n")
            (root / "fixture.py").write_text(
                "import json\nfrom pathlib import Path\n"
                "root = Path.cwd()\n"
                "assert not (root / 'build').exists()\n"
                "(root / 'build/tests').mkdir(parents=True)\n"
                "binary = root / 'build/tests/unittests'\n"
                "binary.write_text('#!/bin/sh\\ngit rev-parse --verify HEAD\\n')\n"
                "binary.chmod(0o755)\n"
                "(root / 'build/compile_commands.json').write_text(json.dumps([\n"
                f"    {{'file': '{SOURCE}', 'directory': str(root),\n"
                "     'arguments': ['c++', '--coverage']}] ))\n")
            (scripts / "collect_cpp_coverage.sh").write_text(
                "mkdir coverage\n"
                f"printf 'SF:{SOURCE}\\nDA:1,1\\nend_of_record\\n' > coverage/coverage.info\n")
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            source.write_text("after\n")
            git("add", ".")
            git("commit", "-qm", "head")
            head = git("rev-parse", "HEAD")
            env = dict(os.environ, GIT_TEST_ASSUME_DIFFERENT_OWNER="1",
                       GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="safe.directory",
                       GIT_CONFIG_VALUE_0="")
            untrusted = subprocess.run(["git", "diff", "--quiet", base, head, "--", "extern"],
                                       cwd=root, env=env, capture_output=True)
            self.assertNotEqual(untrusted.returncode, 0)
            result = subprocess.run(["bash", str(scripts / "run_simd_coverage.sh"), base, head],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            data = json.loads((root / "simd-coverage/result.json").read_text())
            self.assertEqual((data["base_sha"], data["head_sha"]), (base, head))
            self.assertEqual(data["patch"], {"covered": 1, "measured": 1})
            for role in ("base", "head"):
                self.assertEqual(data[role], {"covered": 1, "measured": 1})
                self.assertFalse((root / f"simd-{role}").exists())
            invalid = subprocess.run(["bash", str(scripts / "run_simd_coverage.sh"),
                                      "missing-revision", head], cwd=root, env=env,
                                     capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)
            summary = (root / "simd-coverage/summary.md").read_text()
            self.assertIn("repository or revision validation failed", summary)
            self.assertNotIn("dependency definitions differ", summary)


if __name__ == "__main__":
    unittest.main()
