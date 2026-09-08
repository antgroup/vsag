#!/usr/bin/env python3

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_RUNNER = ROOT / "scripts/ci/run-with-diagnostics.sh"
COVERAGE_COLLECTOR = ROOT / "scripts/coverage/collect_cpp_coverage.sh"
COVERAGE_CHECKER = ROOT / "scripts/coverage/check_cov.sh"
COVERAGE_VERIFIER = ROOT / "scripts/coverage/verify_cpp_coverage.py"
CODECOV_CONFIG = ROOT / ".github/codecov.yml"
PARALLEL_TEST_RUNNER = ROOT / "scripts/testing/test_parallel_bg.sh"
PR_CI_WORKFLOW = ROOT / ".github/workflows/pr-ci.yml"


class DiagnosticRunnerTest(unittest.TestCase):
    def run_command(
        self, log_file: Path, *command: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(DIAGNOSTIC_RUNNER), str(log_file), *command],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_records_successful_command_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            log_file = Path(directory) / "diagnostics/success.log"
            result = self.run_command(log_file, "bash", "-c", "printf 'success output\\n'")
            log = log_file.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("success output", log)
        self.assertIn("Command exit status: 0", log)
        self.assertIn("Diagnostic capture exit status: 0", log)

    def test_propagates_failure_and_repeats_useful_tail(self):
        command = (
            "for value in {1..250}; do printf 'noise %s\\n' \"$value\"; done; "
            "printf 'causal detail\\n'; "
            "for value in {1..10}; do printf 'cleanup %s\\n' \"$value\"; done; "
            "exit 23"
        )
        with tempfile.TemporaryDirectory() as directory:
            log_file = Path(directory) / "diagnostics/failure.log"
            result = self.run_command(log_file, "bash", "-c", command)
            log = log_file.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        self.assertIn("causal detail", log)
        self.assertIn("Command exit status: 23", log)
        self.assertIn("::error title=Command failed::Exit status 23", result.stdout)
        failure_tail = result.stdout.split("::group::Failure output", maxsplit=1)[1]
        self.assertIn("causal detail", failure_tail)
        self.assertIn("Command failed with exit status 23", result.stdout)

    def test_distinguishes_a_capture_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            log_file = Path(directory) / "existing-directory"
            log_file.mkdir()
            result = self.run_command(log_file, "bash", "-c", "printf 'command ran\\n'")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("command ran", result.stdout)
        self.assertIn("Diagnostic capture failed", result.stdout)


class CoverageCollectorTest(unittest.TestCase):
    def run_collector(
        self, final_source: str = "src/example.cpp", include_branch: bool = True
    ) -> tuple[subprocess.CompletedProcess[str], list[list[str]], str]:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            collector_dir = temporary_root / "scripts/coverage"
            collector_dir.mkdir(parents=True)
            collector = collector_dir / COVERAGE_COLLECTOR.name
            shutil.copy2(COVERAGE_COLLECTOR, collector)
            shutil.copy2(COVERAGE_VERIFIER, collector_dir / COVERAGE_VERIFIER.name)

            (temporary_root / "src").mkdir()
            (temporary_root / "include").mkdir()
            (temporary_root / "src/example.cpp").write_text("int example;\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=temporary_root, check=True)
            subprocess.run(
                ["git", "add", "src/example.cpp"], cwd=temporary_root, check=True
            )

            fake_bin = temporary_root / "fake-bin"
            fake_bin.mkdir()
            fake_lcov = fake_bin / "lcov"
            fake_lcov.write_text(
                """#!/usr/bin/env bash
for argument in "$@"; do
    printf '%s\\t' "$argument" >> "$LCOV_CALLS"
done
printf '\\n' >> "$LCOV_CALLS"

output_file=""
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--output-file" ]]; then
        shift
        output_file="$1"
    fi
    shift
done
if [[ -n "$output_file" ]]; then
    if [[ "$output_file" == "$LCOV_ROOT/coverage/coverage.info" ]]; then
        source_path="$LCOV_FINAL_SOURCE"
    else
        source_path="$LCOV_ROOT/src/example.cpp"
    fi
    {
        printf 'TN:\\n'
        printf 'SF:%s\\n' "$source_path"
        printf 'DA:1,1\\n'
        if [[ "$LCOV_INCLUDE_BRANCH" == "true" ]]; then
            printf 'BRDA:1,0,0,1\\n'
        fi
        printf 'end_of_record\\n'
    } > "$output_file"
fi
""",
                encoding="utf-8",
            )
            fake_lcov.chmod(0o755)

            calls_file = temporary_root / "lcov-calls"
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            environment["LCOV_CALLS"] = str(calls_file)
            environment["LCOV_FINAL_SOURCE"] = final_source
            environment["LCOV_INCLUDE_BRANCH"] = str(include_branch).lower()
            environment["LCOV_ROOT"] = str(temporary_root)
            result = subprocess.run(
                ["bash", str(collector)],
                cwd=temporary_root,
                text=True,
                capture_output=True,
                check=False,
                env=environment,
            )
            calls = [
                [argument for argument in line.split("\t") if argument]
                for line in calls_file.read_text(encoding="utf-8").splitlines()
            ]
            final_trace = temporary_root / "coverage/coverage.info"
            trace_contents = (
                final_trace.read_text(encoding="utf-8") if final_trace.exists() else ""
            )

        return result, calls, trace_contents

    def test_scopes_trace_and_preserves_branch_configuration(self):
        result, calls, trace_contents = self.run_collector()

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls), 4)
        for call in calls:
            branch_setting = [
                call[index + 1]
                for index, argument in enumerate(call[:-1])
                if argument == "--rc" and call[index + 1] == "branch_coverage=1"
            ]
            self.assertEqual(branch_setting, ["branch_coverage=1"], call)

        capture = calls[0]
        self.assertIn("--capture", capture)
        exclude_index = capture.index("--exclude")
        self.assertEqual(capture[exclude_index + 1], "/usr/*")
        includes = [
            capture[index + 1]
            for index, argument in enumerate(capture[:-1])
            if argument == "--include"
        ]
        self.assertEqual(len(includes), 2)
        self.assertTrue(includes[0].endswith("/src/*"), includes)
        self.assertTrue(includes[1].endswith("/include/*"), includes)
        self.assertNotIn("inconsistent,inconsistent", capture)

        self.assertIn("--remove", calls[1])
        self.assertTrue(any(value.endswith("/src/version.h") for value in calls[1]))
        self.assertTrue(any(value.endswith("/include/vsag/expected.hpp") for value in calls[1]))
        self.assertNotIn("*/avx512.cpp", calls[1])

        self.assertIn("--extract", calls[2])
        extract_index = calls[2].index("--extract")
        self.assertEqual(calls[2][extract_index + 2 : extract_index + 4], ["src/*", "include/*"])
        self.assertIn("--substitute", calls[2])
        self.assertIn("--list", calls[3])
        self.assertIn("SF:src/example.cpp", trace_contents)
        self.assertIn("BRDA:1,0,0,1", trace_contents)

    def test_rejects_non_repository_trace_source(self):
        result, _, _ = self.run_collector(".ci-fetchcontent/dependency.cpp")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-production sources", result.stderr)

        result, _, _ = self.run_collector("src/../.ci-fetchcontent/dependency.cpp")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-production sources", result.stderr)

    def test_rejects_trace_without_branch_records(self):
        result, _, _ = self.run_collector(include_branch=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no branch records", result.stderr)

    def test_rejects_generated_or_untracked_trace_source(self):
        result, _, _ = self.run_collector("src/generated.cpp")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("generated or untracked sources", result.stderr)


class CoverageCMakeTest(unittest.TestCase):
    @staticmethod
    def coverage_scopes(cmake: str, target: str) -> list[str]:
        scopes = []
        pattern = rf"target_link_libraries\s*\(\s*{re.escape(target)}\b(?P<body>.*?)\)"
        for match in re.finditer(pattern, cmake, flags=re.DOTALL):
            current_scope = ""
            for token in match.group("body").split():
                if token in {"PRIVATE", "PUBLIC", "INTERFACE"}:
                    current_scope = token
                elif token == "coverage_config":
                    scopes.append(current_scope)
                else:
                    variable = re.fullmatch(r"\$\{([A-Za-z0-9_]+)\}", token)
                    if variable:
                        definition = re.search(
                            rf"set\s*\(\s*{variable.group(1)}\b(?P<body>.*?)\)",
                            cmake,
                            flags=re.DOTALL,
                        )
                        if definition and "coverage_config" in definition.group("body").split():
                            scopes.append(current_scope)
        return scopes

    def test_every_production_target_owns_private_instrumentation(self):
        production_targets: dict[str, Path] = {}
        cmake_by_file: dict[Path, str] = {}
        for cmake_path in (ROOT / "src").rglob("CMakeLists.txt"):
            cmake = cmake_path.read_text(encoding="utf-8")
            cmake_by_file[cmake_path] = cmake
            targets = re.findall(
                r"add_library\s*\(\s*([A-Za-z0-9_]+)\s+(?:OBJECT|STATIC|SHARED)\b",
                cmake,
            )
            for target in targets:
                if not target.endswith("_test") and target != "vsag_test":
                    production_targets[target] = cmake_path

        incorrect_scopes = {
            target: self.coverage_scopes(cmake_by_file[path], target)
            for target, path in production_targets.items()
            if self.coverage_scopes(cmake_by_file[path], target) != ["PRIVATE"]
        }

        self.assertTrue(production_targets)
        self.assertFalse(incorrect_scopes, incorrect_scopes)

    def test_coverage_test_executables_link_runtime_privately(self):
        cmake = (ROOT / "tests/CMakeLists.txt").read_text(encoding="utf-8")

        self.assertEqual(self.coverage_scopes(cmake, "unittests"), ["PRIVATE"])
        self.assertEqual(self.coverage_scopes(cmake, "functests"), ["PRIVATE"])


class CoverageVerifierTest(unittest.TestCase):
    def run_verifier(
        self, source_root: Path, entries: list[dict[str, object]]
    ) -> subprocess.CompletedProcess[str]:
        compile_commands = source_root / "compile_commands.json"
        compile_commands.write_text(json.dumps(entries), encoding="utf-8")
        return subprocess.run(
            [
                "python3",
                str(COVERAGE_VERIFIER),
                "instrumentation",
                str(compile_commands),
                "--source-root",
                str(source_root),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_accepts_instrumented_production_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory)
            (source_root / "src").mkdir()
            source = source_root / "src/example.cpp"
            source.write_text("int example;\n", encoding="utf-8")
            result = self.run_verifier(
                source_root,
                [
                    {
                        "directory": str(source_root),
                        "file": "src/example.cpp",
                        "arguments": ["c++", "--coverage", "-c", str(source)],
                    }
                ],
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("verified coverage instrumentation for 1", result.stdout)

    def test_rejects_uninstrumented_production_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory)
            (source_root / "src").mkdir()
            source = source_root / "src/example.cpp"
            source.write_text("int example;\n", encoding="utf-8")
            result = self.run_verifier(
                source_root,
                [
                    {
                        "directory": str(source_root),
                        "file": str(source),
                        "command": f"c++ -c {source}",
                    }
                ],
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("src/example.cpp", result.stderr)


class CoverageCheckerTest(unittest.TestCase):
    def test_uses_branch_aware_summary_with_consistency_allowance(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            coverage_dir = temporary_root / "coverage"
            coverage_dir.mkdir()
            (coverage_dir / "coverage.info").write_text("TN:\n", encoding="utf-8")
            fake_bin = temporary_root / "fake-bin"
            fake_bin.mkdir()
            fake_lcov = fake_bin / "lcov"
            fake_lcov.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" > \"$LCOV_CALL\"\n"
                "printf '  lines.......: 90.0%% (90 of 100 lines)\\n'\n",
                encoding="utf-8",
            )
            fake_lcov.chmod(0o755)
            call_file = temporary_root / "lcov-call"
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            environment["LCOV_CALL"] = str(call_file)
            result = subprocess.run(
                ["bash", str(COVERAGE_CHECKER)],
                cwd=temporary_root,
                text=True,
                capture_output=True,
                check=False,
                env=environment,
            )
            call = call_file.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--rc branch_coverage=1", call)
        self.assertIn("--ignore-errors inconsistent,inconsistent", call)


class ParallelTestRunnerTest(unittest.TestCase):
    def test_passes_and_reports_explicit_random_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            tests_dir = temporary_root / "build/tests"
            tests_dir.mkdir(parents=True)
            fake_test = tests_dir / "fake-test"
            fake_test.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$TEST_CALLS\"\n",
                encoding="utf-8",
            )
            fake_test.chmod(0o755)
            (tests_dir / "unittests").symlink_to(fake_test)
            (tests_dir / "functests").symlink_to(fake_test)

            calls_file = temporary_root / "test-calls"
            environment = os.environ.copy()
            environment["TEST_CALLS"] = str(calls_file)
            environment["VSAG_TEST_SEED"] = "424242"
            result = subprocess.run(
                ["bash", str(PARALLEL_TEST_RUNNER)],
                cwd=temporary_root,
                text=True,
                capture_output=True,
                check=False,
                env=environment,
            )
            calls = calls_file.read_text(encoding="utf-8").splitlines()

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("test random seed: 424242", result.stdout)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all("--rng-seed 424242" in call for call in calls), calls)


class CodecovComponentTest(unittest.TestCase):
    def component_patterns(self) -> dict[str, list[str]]:
        components: dict[str, list[str]] = {}
        component_id = ""
        in_components = False
        for line in CODECOV_CONFIG.read_text(encoding="utf-8").splitlines():
            if line.startswith("  individual_components:"):
                in_components = True
                continue
            if not in_components:
                continue
            component_match = re.fullmatch(r"    - component_id: (\S+)", line)
            if component_match:
                component_id = component_match.group(1)
                components[component_id] = []
                continue
            path_match = re.fullmatch(r"        - (\S+)", line)
            if path_match and component_id:
                components[component_id].append(path_match.group(1))
        return components

    @staticmethod
    def matches(pattern: str, path: str) -> bool:
        if pattern.startswith("^"):
            return re.fullmatch(pattern, path) is not None
        if pattern.endswith("/**"):
            return path.startswith(pattern.removesuffix("**"))
        return path == pattern

    def test_every_maintained_source_has_one_component_owner(self):
        patterns = self.component_patterns()
        tracked_files = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files", "--", "src", "include"], text=True
        ).splitlines()
        maintained_sources = [
            path
            for path in tracked_files
            if Path(path).suffix in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}
            and "_test." not in Path(path).name
            and path != "include/vsag/expected.hpp"
        ]

        ownership = {
            path: [
                component
                for component, component_patterns in patterns.items()
                if any(self.matches(pattern, path) for pattern in component_patterns)
            ]
            for path in maintained_sources
        }
        incorrectly_owned = {
            path: owners for path, owners in ownership.items() if len(owners) != 1
        }

        self.assertTrue(patterns)
        self.assertFalse(incorrectly_owned, incorrectly_owned)


class CoverageWorkflowTest(unittest.TestCase):
    def test_premerge_check_is_configuration_only(self):
        workflow = PR_CI_WORKFLOW.read_text(encoding="utf-8")
        job_match = re.search(
            r"^  coverage-config:\n(?P<job>.*?)(?=^  [a-z][a-z0-9-]+:\n)",
            workflow,
            flags=re.MULTILINE | re.DOTALL,
        )

        self.assertIsNotNone(job_match)
        job = job_match.group("job")
        self.assertIn("needs.changes.outputs.coverage == 'true'", job)
        self.assertIn("coverage_diagnostics_test.py", job)
        self.assertIn("bash -n scripts/testing/test_parallel_bg.sh", job)
        self.assertNotIn("make cov", job)
        self.assertNotIn("VSAG_TEST_SEED", job)


if __name__ == "__main__":
    unittest.main()
