#!/usr/bin/env python3
"""Regression checks for TSAN runner-cache cleanup."""

from pathlib import Path
import os
import re
import shlex
import shutil
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
            for command in ("date", "ls", "readlink", "stat", "awk")
        )
        diagnostics += '\ntimeout() { return "${PROBE_STATUS:-1}"; }'
        return diagnostics + "\n" + script.replace("/useless", str(sandbox))

    def run_cleanup(self, sandbox: Path, env=None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["sh", "-e", "-c", self.cleanup_script(sandbox)],
            env={**os.environ, **(env or {})},
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

    def test_cleanup_failure_attempts_later_entries_and_prints_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sandbox = root / "cleanup"
            sandbox.mkdir()
            node = sandbox / "runner-cache/externals/node24/bin/node"
            node.parent.mkdir(parents=True)
            node.write_text("runtime\n")
            for name in ("a", "b", "c"):
                (sandbox / name).mkdir()
                (sandbox / name / "payload").write_text("delete me\n")
            stub_dir = root / "bin"
            stub_dir.mkdir()
            stub = stub_dir / "rm"
            real_rm = shlex.quote(shutil.which("rm"))
            failed_path = shlex.quote(str(sandbox / "b"))
            stub.write_text(
                "#!/bin/sh\n"
                f'if [ "$3" = {failed_path} ]; then\n'
                '  echo "cleanup failed: $3" >&2\n'
                "  exit 23\n"
                "fi\n"
                f'exec {real_rm} "$@"\n'
            )
            stub.chmod(0o755)

            result = self.run_cleanup(
                sandbox, {"PATH": str(stub_dir) + os.pathsep + os.environ["PATH"]}
            )

            self.assertNotEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr.count("cleanup failed:"), 1)
            self.assertIn(str(sandbox / "b"), result.stderr)
            self.assertIn("(before cleanup)", result.stdout)
            self.assertIn("(after cleanup)", result.stdout)
            self.assertEqual({path.name for path in sandbox.iterdir()}, {"b", "runner-cache"})
            self.assertEqual((sandbox / "b/payload").read_text(), "delete me\n")
            self.assertEqual(node.read_text(), "runtime\n")

    def test_failed_node_probes_report_status_and_allow_cleanup(self) -> None:
        for status in (124, 127):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                sandbox = Path(temporary)
                (sandbox / "tools").mkdir()

                result = self.run_cleanup(sandbox, {"PROBE_STATUS": str(status)})

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(list(sandbox.iterdir()), [])
                before, after = result.stdout.split("(after cleanup)")
                self.assertIn("(before cleanup)", before)
                self.assertEqual(before.count(f"node --version exit={status}"), 1)
                self.assertEqual(after.count(f"node --version exit={status}"), 1)


if __name__ == "__main__":
    unittest.main()
