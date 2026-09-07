#!/usr/bin/env python3

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_RUNNER = ROOT / "scripts/ci/run-with-diagnostics.sh"
COVERAGE_COLLECTOR = ROOT / "scripts/coverage/collect_cpp_coverage.sh"


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
    def test_excludes_system_headers_during_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            collector_dir = temporary_root / "scripts/coverage"
            collector_dir.mkdir(parents=True)
            collector = collector_dir / COVERAGE_COLLECTOR.name
            shutil.copy2(COVERAGE_COLLECTOR, collector)

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
    printf 'TN:\\n' > "$output_file"
fi
""",
                encoding="utf-8",
            )
            fake_lcov.chmod(0o755)

            calls_file = temporary_root / "lcov-calls"
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            environment["LCOV_CALLS"] = str(calls_file)
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

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls), 3)
        capture = calls[0]
        self.assertIn("--capture", capture)
        exclude_index = capture.index("--exclude")
        self.assertEqual(capture[exclude_index + 1], "/usr/*")
        self.assertNotIn("inconsistent,inconsistent", capture)
        self.assertIn("--remove", calls[1])
        self.assertIn("/usr/*", calls[1])


if __name__ == "__main__":
    unittest.main()
