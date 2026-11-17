#!/usr/bin/env python3
"""Check ASan artifact and build diagnostics contracts without compiling binaries."""

from pathlib import Path
import os
import importlib.util
import json
import signal
import sys
import time
import re
import subprocess
import tempfile
import textwrap
import unittest
from unittest import mock

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



SCRIPT = ROOT / 'scripts/ci/monitor_build.py'
SPEC = importlib.util.spec_from_file_location('monitor', SCRIPT)
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


class BuildMonitorTest(unittest.TestCase):
    def run_build(self, directory, code):
        return subprocess.run([sys.executable, str(SCRIPT), '--output-dir', str(directory),
                               '--interval', '0.05', '--', sys.executable, '-c', code],
                              capture_output=True, text=True, timeout=10)

    def test_success_and_failure_are_not_masked(self):
        for status in (0, 7, 137):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                result = self.run_build(directory, f'import sys; print("build output"); sys.exit({status})')
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertIn('build output', result.stdout)
                summary = json.loads((Path(directory) / 'summary.json').read_text())
                self.assertEqual(summary['exitCode'], status)
                self.assertIsNone(summary['receivedSignal'])
                self.assertIn('does not prove OOM', summary['note'])

    def test_killed_build_reports_signal_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_build(directory, 'import os,signal; os.kill(os.getpid(), signal.SIGKILL)')
            self.assertEqual(result.returncode, 137, result.stderr)
            summary = json.loads((Path(directory) / 'summary.json').read_text())
            self.assertEqual(summary['commandReturnCode'], -9)
            self.assertIsNone(summary['receivedSignal'])

    def test_failed_command_cannot_leave_a_build_child_running(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'leaked'
            child = 'import time; from pathlib import Path; time.sleep(1); Path(' + repr(str(marker)) + ').touch()'
            code = ('import subprocess,sys; subprocess.Popen([sys.executable,"-c",' +
                    repr(child) + ']); raise SystemExit(7)')
            result = self.run_build(directory, code)
            self.assertEqual(result.returncode, 7, result.stderr)
            time.sleep(1.1)
            self.assertFalse(marker.exists())

    def test_invalid_intervals_do_not_launch_the_build(self):
        for interval in ('0', '-1', 'nan', 'inf'):
            with self.subTest(interval=interval), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), '--output-dir', directory,
                     '--interval', interval, '--', sys.executable, '-c', 'print("launched")'],
                    capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn('launched', result.stdout)

    def test_unwritable_capture_does_not_mask_command_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'not-a-directory'
            file.write_text('occupied')
            result = self.run_build(file / 'capture', 'raise SystemExit(7)')
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertIn('file capture unavailable', result.stdout)
            self.assertIn('build-resource-result', result.stdout)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux procfs sampling')
    def test_live_samples_are_streamed_and_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_build(directory, 'import time; time.sleep(0.2)')
            self.assertEqual(result.returncode, 0, result.stderr)
            samples = [json.loads(s) for s in (Path(directory) / 'samples.jsonl').read_text().splitlines()]
            self.assertGreaterEqual(len(samples), 2)
            self.assertTrue(any(s['processCount'] > 0 for s in samples))
            self.assertIn('MemAvailable', samples[0]['memory'])
            self.assertIn('[build-resource]', result.stdout)

    def test_sigterm_is_forwarded_and_descendants_are_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready, leaked = root / 'ready', root / 'leaked'
            child = 'import time; from pathlib import Path; time.sleep(1); Path(' + repr(str(leaked)) + ').touch()'
            code = ('import subprocess,sys,time; from pathlib import Path; '
                    'subprocess.Popen([sys.executable,"-c",' + repr(child) + ']); '
                    'Path(' + repr(str(ready)) + ').touch(); time.sleep(30)')
            proc = subprocess.Popen([sys.executable, str(SCRIPT), '--output-dir', directory,
                                     '--interval', '0.05', '--', sys.executable, '-c', code],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(ready.exists())
                proc.send_signal(signal.SIGTERM)
                _, errors = proc.communicate(timeout=5)
                self.assertEqual(proc.returncode, 143, errors)
                self.assertEqual(json.loads((root / 'summary.json').read_text())['receivedSignal'], 15)
                time.sleep(1.1)
                self.assertFalse(leaked.exists(), 'cancelled build descendant survived')
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()

    def test_cgroup_v2_host_and_container_namespace_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc, cg = root / 'proc', root / 'cg'
            (proc / 'self').mkdir(parents=True)
            cg.mkdir()
            for mount_root, group, expected in (
                ('/', '/build', cg / 'build'), ('/docker/job', '/docker/job', cg),
                ('/', '/', cg),
            ):
                (proc / 'self/cgroup').write_text('0::' + group + '\n')
                (proc / 'self/mountinfo').write_text(f'1 0 0:1 {mount_root} {cg} rw - cgroup2 cgroup rw\n')
                self.assertEqual(monitor.cgroup_directories(proc, cg), [expected])

    def test_cgroup_v1_and_unavailable_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc, cg = root / 'proc', root / 'cg'
            (proc / 'self').mkdir(parents=True)
            cg.mkdir()
            self.assertEqual(monitor.cgroup_directories(proc, cg), [])
            (proc / 'self/cgroup').write_text('4:memory:/build\n')
            (proc / 'self/mountinfo').write_text(f'1 0 0:1 / {cg}/memory rw - cgroup cgroup rw,memory\n')
            self.assertEqual(monitor.cgroup_directories(proc, cg), [cg / 'memory/build'])

    def test_private_cgroup_namespace_resolves_readable_controller_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc, cg = root / 'proc', root / 'cg'
            (proc / 'self').mkdir(parents=True)
            for kind, membership, options, mountpoint, counter in (
                ('cgroup', '4:memory:', 'rw,memory', cg / 'memory', 'memory.failcnt'),
                ('cgroup2', '0::', 'rw', cg, 'memory.events'),
            ):
                for group in ('/', '/build'):
                    with self.subTest(kind=kind, group=group):
                        candidate = mountpoint / group.lstrip('/')
                        candidate.mkdir(parents=True, exist_ok=True)
                        (candidate / counter).write_text('0')
                        (proc / 'self/cgroup').write_text(membership + group + '\n')
                        (proc / 'self/mountinfo').write_text(
                            f'1 0 0:1 /docker/job {mountpoint} rw - {kind} cgroup {options}\n')
                        self.assertEqual(monitor.cgroup_directories(proc, cg), [candidate])
                        (candidate / counter).unlink()
                        self.assertEqual(monitor.cgroup_directories(proc, cg), [])
                for group in ('/../escape', 'relative'):
                    (proc / 'self/cgroup').write_text(membership + group + '\n')
                    self.assertEqual(monitor.cgroup_directories(proc, cg), [])

    def test_process_sampling_filters_group_and_does_not_dump_secret_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pid, group, name in ((1, 42, 'cc1plus'), (2, 42, 'make'), (3, 99, 'cc1plus')):
                p = root / str(pid)
                p.mkdir()
                fields = ['S', '0', str(group)] + ['0'] * 20
                (p / 'stat').write_text(f'{pid} ({name}) ' + ' '.join(fields))
                (p / 'status').write_text('VmRSS:\t2048 kB\n')
                (p / 'comm').write_text(name)
                (p / 'cmdline').write_text(name + '\0-DTOKEN=secret\0src/example.cpp\0')
            # Equal RSS does not impose an order on directory iteration.
            for order in ((1, 2, 3), (3, 2, 1)):
                with self.subTest(order=order), mock.patch.object(
                        Path, 'iterdir', return_value=iter(root / str(pid) for pid in order)):
                    result = monitor.processes(42, root)
                by_pid = {p['pid']: p for p in result}
                self.assertEqual(set(by_pid), {1, 2})
                self.assertEqual(by_pid[1]['sources'], ['src/example.cpp'])
                self.assertNotIn('sources', by_pid[2])
                self.assertNotIn('secret', json.dumps(result))

    def test_arm_action_retains_build_scope_and_always_uploads(self):
        text = (ROOT / '.github/actions/asan-build-and-test-build-asan-aarch64/action.yml').read_text()
        self.assertIn('make asan COMPILE_JOBS=3 DATACELL_COMPILE_JOBS=1 VSAG_ENABLE_EXAMPLES=ON', text)
        self.assertIn('python3 scripts/ci/monitor_build.py', text)
        self.assertIn('name: Upload ARM build diagnostics\n      if: always()', text)
        self.assertIn('${{ github.run_attempt }}', text)
        build_step = text.split('    - name: Make Asan\n', 1)[1].split('    - name:', 1)[0]
        self.assertNotIn('continue-on-error', build_step)
        self.assertIn('if: always()\n      continue-on-error: true', text)


class NightlySimdBuildTest(unittest.TestCase):
    def build_job(self):
        return job_text(ROOT / '.github/workflows/daily_test.yml', 'daily_build_simd')

    def test_monitor_wraps_effective_limits_and_preserves_failures(self):
        step = self.build_job().split('      - name: Make Asan\n', 1)[1].split('      - name:', 1)[0]
        self.assertNotIn('continue-on-error', step)
        script = textwrap.dedent(step.split('        run: |\n', 1)[1])
        script = script.replace('${{ matrix.simd_option }}', 'sse;-DDISABLE_AVX_FORCE=1')
        for make_exit, simd_exit in ((0, 0), (7, 0), (137, 0), (0, 9)):
            with self.subTest(make_exit=make_exit, simd_exit=simd_exit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bindir = root / 'bin'
                bindir.mkdir()
                (bindir / 'python3').symlink_to(sys.executable)
                shim = '#!' + sys.executable + '\n' + textwrap.dedent('''\
                    import json, os, sys
                    from pathlib import Path
                    tool = Path(sys.argv[0]).name
                    with Path(os.environ['COMMAND_LOG']).open('a') as output:
                        output.write(json.dumps({'tool': tool, 'args': sys.argv[1:],
                            'mkl': os.environ.get('VSAG_ENABLE_INTEL_MKL'),
                            'generator': os.environ.get('CMAKE_GENERATOR'),
                            'simd': os.environ.get('EXTRA_DEFINED')}) + '\\n')
                    raise SystemExit(int(os.environ['MAKE_EXIT' if tool == 'make' else 'SIMD_EXIT']))
                    ''')
                for tool in ('make', 'cmake'):
                    path = bindir / tool
                    path.write_text(shim)
                    path.chmod(0o755)
                env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'],
                           RUNNER_TEMP=str(root), GITHUB_ENV=str(root / 'github-env'),
                           COMMAND_LOG=str(root / 'commands.jsonl'), MAKE_EXIT=str(make_exit),
                           SIMD_EXIT=str(simd_exit), COMPILE_JOBS='24', DATACELL_COMPILE_JOBS='24')
                result = subprocess.run(['sh', '-e', '-c', script], cwd=ROOT, env=env,
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, make_exit or simd_exit, result.stdout + result.stderr)
                calls = [json.loads(line) for line in (root / 'commands.jsonl').read_text().splitlines()]
                self.assertEqual(calls[0]['args'], ['asan', 'COMPILE_JOBS=3', 'DATACELL_COMPILE_JOBS=1'])
                self.assertEqual(calls[0]['mkl'], 'ON')
                self.assertEqual(calls[0]['generator'], 'Ninja')
                self.assertEqual(calls[0]['simd'], '-DDISABLE_AVX_FORCE=1')
                self.assertEqual(len(calls), 1 if make_exit else 2)
                if not make_exit:
                    self.assertEqual(calls[1]['args'],
                                     ['--build', 'build', '--target', 'unittests_simd', '--parallel', '3'])
                for stage, status in [('asan', make_exit)] + ([] if make_exit else [('simd', simd_exit)]):
                    summary = json.loads((root / 'simd-build-diagnostics' / stage / 'summary.json').read_text())
                    self.assertEqual(summary['exitCode'], status)
                self.assertIn('[build-resource-result]', result.stdout)

    def test_diagnostics_upload_is_failure_safe_and_matrix_unique(self):
        step = self.build_job().split('      - name: Upload SIMD build diagnostics\n', 1)[1].split('      - name:', 1)[0]
        self.assertIn('if: always()', step)
        self.assertIn('continue-on-error: true', step)
        self.assertIn('uses: actions/upload-artifact@v5', step)
        self.assertIn('path: ${{ runner.temp }}/simd-build-diagnostics/', step)
        self.assertIn('name: simd-build-diagnostics-${{ strategy.job-index }}-${{ github.run_id }}-${{ github.run_attempt }}', step)
        self.assertIn('retention-days: 7', step)
        self.assertIn('if-no-files-found: warn', step)

    def test_simd_matrix_and_functional_sentinels_remain_required(self):
        workflow = ROOT / '.github/workflows/daily_test.yml'
        build = self.build_job()
        tests = job_text(workflow, 'daily_test_asan_simd')
        self.assertIn('simd_option: ${{ fromJSON(needs.daily-plan.outputs.simd_options) }}', build)
        self.assertIn('test_type: [unittests_simd]', tests)
        self.assertIn('simd_key: ${{ fromJSON(needs.daily-plan.outputs.simd_keys) }}', tests)
        self.assertIn('python3 scripts/testing/daily_entry.py unittests_simd simd', tests)
        self.assertIn('python3 scripts/testing/daily_measure.py --name sentinels --owner simd -- python3 scripts/ci/platform_smoke.py functests', tests)
        self.assertNotIn('continue-on-error', tests)


if __name__ == "__main__":
    unittest.main()
