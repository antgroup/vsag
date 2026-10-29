import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import time
import unittest
from unittest import mock

import prepare_lint_base as helper

BASE = 'a' * 40


class RetryTest(unittest.TestCase):
    def result(self, error=''):
        return subprocess.CompletedProcess([], 128 if error else 0, 'ok', error)

    def test_deadline_kills_transport_children_holding_pipes(self):
        command = [sys.executable, '-c',
                   "import subprocess,sys,time; "
                   "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                   "time.sleep(30)"]
        started = time.monotonic()
        with mock.patch.object(helper, 'TIMEOUT_SECONDS', 0.2):
            with self.assertRaises(subprocess.TimeoutExpired):
                helper.run_with_deadline(command, dict(os.environ))
        self.assertLess(time.monotonic() - started, 5)

    def test_timeout_allows_lock_cleanup_before_next_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / 'shallow.lock'
            code = """import pathlib, signal, sys, time
lock = pathlib.Path(sys.argv[1])
def cleanup(signum, frame):
    lock.unlink()
    sys.exit(128 + signum)
signal.signal(signal.SIGTERM, cleanup)
lock.write_text('owned by this process')
time.sleep(30)
"""
            with mock.patch.object(helper, 'TIMEOUT_SECONDS', 1):
                with self.assertRaises(subprocess.TimeoutExpired):
                    helper.run_with_deadline([sys.executable, '-c', code, str(lock)],
                                             dict(os.environ))
            self.assertFalse(lock.exists())
            # A subsequent attempt can acquire the lock; never unlink another
            # process's lock from the retry wrapper.
            with lock.open('x') as output:
                output.write('next attempt')

    def test_stubborn_transport_is_killed_after_bounded_grace(self):
        command = [sys.executable, '-c',
                   'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                   'time.sleep(30)']
        started = time.monotonic()
        with mock.patch.object(helper, 'TIMEOUT_SECONDS', 0.5), \
             mock.patch.object(helper, 'TERMINATE_GRACE_SECONDS', 0.2):
            with self.assertRaises(subprocess.TimeoutExpired):
                helper.run_with_deadline(command, dict(os.environ))
        self.assertLess(time.monotonic() - started, 5)

    @mock.patch.object(helper.time, 'sleep')
    @mock.patch.object(helper, 'run_with_deadline')
    def test_tls_failure_then_success(self, run, sleep):
        run.side_effect = [self.result('GnuTLS recv error (-110)'), self.result()]
        self.assertEqual(helper.network_git(['fetch', 'origin', BASE]), 'ok')
        self.assertEqual(run.call_count, 2)
        self.assertGreaterEqual(sleep.call_args.args[0], 2)
        self.assertLessEqual(sleep.call_args.args[0], 3)
        for call in run.call_args_list:
            self.assertEqual(call.args[1]['GIT_TERMINAL_PROMPT'], '0')
            self.assertEqual(call.args[1]['LC_ALL'], 'C')
        self.assertEqual(helper.TIMEOUT_SECONDS, 60)

    @mock.patch.object(helper.time, 'sleep')
    @mock.patch.object(helper, 'run_with_deadline')
    def test_exhaustion_and_timeouts_are_bounded(self, run, sleep):
        for failure in (self.result('Connection reset by peer'),
                        subprocess.TimeoutExpired(['git'], 60)):
            with self.subTest(failure=failure):
                run.reset_mock(); sleep.reset_mock()
                if isinstance(failure, Exception):
                    run.side_effect = failure
                else:
                    run.side_effect = None; run.return_value = failure
                with self.assertRaises(RuntimeError):
                    helper.network_git(['fetch', 'origin', BASE])
                self.assertEqual(run.call_count, 4)
                self.assertEqual(sleep.call_count, 3)

    @mock.patch.object(helper.time, 'sleep')
    @mock.patch.object(helper, 'run_with_deadline')
    def test_permanent_unknown_and_certificate_errors_do_not_retry(self, run, sleep):
        for error in ('Authentication failed', 'not our ref', 'unknown failure',
                      'SSL certificate problem', 'HTTP/2 stream: requested URL returned error: 403'):
            run.reset_mock(); sleep.reset_mock()
            run.return_value = self.result(error + ' https://secret@example.invalid')
            log = io.StringIO()
            with contextlib.redirect_stderr(log), self.assertRaises(RuntimeError):
                helper.network_git(['fetch', 'origin', BASE])
            self.assertEqual(run.call_count, 1)
            sleep.assert_not_called()
            self.assertNotIn('secret', log.getvalue())


class PreparationTest(unittest.TestCase):
    def test_missing_output_is_diagnosed_before_fetch(self):
        for value in (None, ''):
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(helper, 'prepare_base') as prepare, \
                 contextlib.redirect_stderr(io.StringIO()) as diagnostic:
                if value is not None:
                    os.environ['GITHUB_OUTPUT'] = value
                self.assertEqual(helper.main(), 1)
                self.assertIn('::error::', diagnostic.getvalue())
                self.assertIn('GITHUB_OUTPUT is not set', diagnostic.getvalue())
                prepare.assert_not_called()

    @mock.patch.object(helper, 'network_git')
    def test_missing_or_invalid_pr_base_never_uses_main(self, network):
        for base in ('', 'main', '-bad', 'a' * 39, 'a' * 40 + '\n'):
            with self.assertRaises(ValueError):
                helper.prepare_base(base, 'pull_request')
        network.assert_not_called()

    @mock.patch.object(helper.subprocess, 'check_output', return_value=BASE + '\n')
    @mock.patch.object(helper, 'network_git')
    def test_manual_dispatch_pins_resolution(self, network, verify):
        network.side_effect = [BASE + '\trefs/heads/main\n', '']
        self.assertEqual(helper.prepare_base('', 'workflow_dispatch'), BASE)
        self.assertEqual(network.call_args_list, [
            mock.call(['ls-remote', '--exit-code', 'origin', 'refs/heads/main']),
            mock.call(['fetch', '--no-tags', '--depth=1', 'origin', BASE]),
        ])

    @mock.patch.object(helper.subprocess, 'check_output', return_value='b' * 40 + '\n')
    @mock.patch.object(helper, 'network_git')
    def test_wrong_object_fails(self, network, verify):
        with self.assertRaisesRegex(ValueError, 'expected commit'):
            helper.prepare_base(BASE, 'pull_request')

    def test_failed_preparation_or_diff_does_not_publish_outputs(self):
        for stage in ('prepare_base', 'get_changed_sources'):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(stage=stage):
                output = Path(tmp) / 'output'
                with mock.patch.dict(os.environ, GITHUB_OUTPUT=str(output)), \
                     mock.patch.object(helper, 'prepare_base', return_value=BASE), \
                     mock.patch.object(helper, 'get_changed_sources', return_value=[]), \
                     mock.patch.object(helper, stage, side_effect=RuntimeError('failed')):
                    self.assertEqual(helper.main(), 1)
                self.assertFalse(output.exists())

    def test_success_and_intentionally_empty_diff_publish(self):
        for paths in ([], ['src/new.cpp']):
            with tempfile.TemporaryDirectory() as tmp, \
                 mock.patch.object(helper, 'prepare_base', return_value=BASE), \
                 mock.patch.object(helper, 'get_changed_sources', return_value=paths) as diff:
                output = Path(tmp) / 'output'
                with mock.patch.dict(os.environ, GITHUB_OUTPUT=str(output)):
                    self.assertEqual(helper.main(), 0)
                self.assertEqual(output.read_text(),
                                 f'base_sha={BASE}\npattern={helper.build_filter(paths)}\n')
                diff.assert_called_once_with(BASE, 'HEAD')

    def test_real_git_pins_old_base_even_when_main_has_advanced(self):
        def git(root, *args):
            return subprocess.check_output(['git', '-C', str(root), *args],
                                           stderr=subprocess.PIPE, text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            remote = Path(tmp) / 'remote'; remote.mkdir()
            git(remote, 'init', '-b', 'main')
            git(remote, 'config', 'user.name', 'Test')
            git(remote, 'config', 'user.email', 'test@example.invalid')
            (remote / 'src/module').mkdir(parents=True)
            file = remote / 'src/module/a.cpp'; file.write_text('base\n')
            git(remote, 'add', '.'); git(remote, 'commit', '-m', 'base')
            base = git(remote, 'rev-parse', 'HEAD')
            file.write_text('changed\n'); git(remote, 'commit', '-am', 'advance main')
            head = git(remote, 'rev-parse', 'HEAD')
            checkout = Path(tmp) / 'checkout'
            subprocess.run(['git', 'clone', '--depth=1', remote.as_uri(), str(checkout)],
                           check=True, capture_output=True)
            env = dict(os.environ, GITHUB_EVENT_NAME='pull_request', LINT_BASE_SHA=base,
                       GITHUB_OUTPUT=str(Path(tmp) / 'output'))
            result = subprocess.run(['python3', str(Path(helper.__file__).resolve())],
                                    cwd=checkout, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(git(checkout, 'rev-parse', 'HEAD'), head)
            self.assertEqual(Path(env['GITHUB_OUTPUT']).read_text(),
                             f'base_sha={base}\npattern=src/module/a\\.cpp$\n')

    def test_both_workflows_prepare_before_compilation(self):
        root = Path(__file__).resolve().parents[2]
        for filename, job, name in (
            ('pr-ci-team.yml', 'build', 'Find changed source files'),
            ('pr-ci-community.yml', 'build-asan-x86', 'Get changed source files'),
        ):
            text = (root / '.github/workflows' / filename).read_text()
            from tests.scripts.test_build_benchmark_workflow import job as workflow_job
            body = workflow_job(text, job)
            # Only the build job contains this named preparation step.
            self.assertLess(body.index('name: ' + name), body.index('make asan'))
            self.assertIn('LINT_BASE_SHA: ${{ github.event.pull_request.base.sha }}', body)
            self.assertIn('python3 scripts/linters/prepare_lint_base.py', body)
            self.assertNotIn('git fetch origin "$BASE_BRANCH"', body)


if __name__ == '__main__':
    unittest.main()
