"""Exercise wheel provenance, matrix completeness and installed-test evidence."""
import ast
import copy
import importlib.util
import json
from io import BytesIO
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.ci import python_ci as ci
from scripts.ci import macos_wheel_ci as mac
from tests.scripts.test_workflow_entries import expression, job

ROOT = Path(__file__).resolve().parents[2]
SHA = 'a' * 40


class PythonCITests(unittest.TestCase):
    def test_supported_matrix_and_unique_lanes(self):
        envs = ci.environments()
        self.assertEqual(len(ci.lanes()), 18)
        self.assertEqual(len(set(ci.lanes())), 18)
        self.assertEqual(len(envs), 9)
        self.assertEqual([e['python'] for e in envs if e['numpy'].startswith('1.')],
                         ['3.10', '3.11', '3.12'])
        self.assertEqual([e for e in envs if e['mode'] == 'openmp'],
                         [dict(python='3.12', numpy='2.*', mode='openmp', key='py312-np2-openmp')])

    def test_wheel_manifest_rejects_wrong_source_arch_content_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel = root / 'pyvsag-1-py3-none-manylinux_2_28_x86_64.whl'
            wheel.write_bytes(b'binary')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(ci.wheel_manifest(root, SHA, 'x86_64')))
            self.assertEqual(ci.verify(root, manifest, SHA, 'x86_64'), wheel.resolve())
            for sha, arch in [('b' * 40, 'x86_64'), (SHA, 'aarch64'), ('main', 'x86_64')]:
                with self.assertRaises(ValueError):
                    ci.verify(root, manifest, sha, arch)
            wheel.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                ci.verify(root, manifest, SHA, 'x86_64')
            (root / 'other_x86_64.whl').write_bytes(b'binary')
            with self.assertRaises(ValueError):
                ci.wheel_manifest(root, SHA, 'x86_64')

    def test_junit_failure_skip_and_empty_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp, 'report.xml')
            xml.write_text('<testsuites><testsuite><testcase/><testcase><failure/></testcase>'
                           '<testcase><error/></testcase><testcase><skipped/></testcase></testsuite></testsuites>')
            self.assertEqual(ci.junit_counts(xml, 'ordinary'),
                             dict(collected=4, passed=1, failed=2, skipped=1))
            for cases in ['', '<testcase/>', '<testcase/>' * 5 + '<testcase><skipped/></testcase>']:
                xml.write_text('<testsuite>' + cases + '</testsuite>')
                with self.assertRaises(ValueError):
                    ci.junit_counts(xml, 'openmp')
            xml.write_text('<testsuite>' + '<testcase/>' * 6 + '</testsuite>')
            self.assertEqual(ci.junit_counts(xml, 'openmp')['passed'], 6)

    def test_runner_preserves_exit_checks_installed_import_and_openmp_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp, 'results.json')
            output.with_suffix('.xml').write_text('<testsuite><testcase/></testsuite>')
            package = SimpleNamespace(__file__='/opt/venv/site-packages/pyvsag/__init__.py', __version__='2.0')
            with patch.object(ci.platform, 'platform', return_value='test-linux'), \
                    patch.dict(os.environ, VSAG_CI_SOURCE_SHA=SHA), \
                    patch.object(ci.importlib, 'import_module', return_value=package), \
                    patch.object(ci.subprocess, 'run', return_value=SimpleNamespace(returncode=7)) as run:
                self.assertEqual(ci.run_tests('ordinary', output), 7)
                self.assertEqual(run.call_args.kwargs['cwd'], ROOT / 'tests/python')
                self.assertIn('--ignore=test_openmp_coexistence.py', run.call_args.args[0])
                self.assertEqual(json.loads(output.read_text())['status'], 'failed')
                run.return_value.returncode = 0
                self.assertEqual(ci.run_tests('ordinary', output), 0)
                package.__file__ = str(ROOT / '.venv/lib/python3.12/site-packages/pyvsag/__init__.py')
                self.assertEqual(ci.run_tests('ordinary', output), 0)
                output.with_suffix('.xml').write_text('<testsuite><testcase><skipped/></testcase></testsuite>')
                self.assertNotEqual(ci.run_tests('ordinary', output), 0)
                output.with_suffix('.xml').write_text('<testsuite><testcase/></testsuite>')
                package.__file__ = str(ROOT / 'python/pyvsag/__init__.py')
                self.assertNotEqual(ci.run_tests('ordinary', output), 0)
            def imports(name):
                if name == 'torch':
                    raise ImportError('missing torch')
                return SimpleNamespace(__file__='/opt/venv/site-packages/pyvsag/__init__.py', __version__='2.0')
            with patch.object(ci.platform, 'platform', return_value='test-linux'), \
                    patch.dict(os.environ, VSAG_CI_SOURCE_SHA=SHA), \
                    patch.object(ci.importlib, 'import_module', side_effect=imports), \
                    patch.object(ci.subprocess, 'run') as run:
                self.assertNotEqual(ci.run_tests('openmp', output), 0)
                run.assert_not_called()
                self.assertIn('missing torch', json.loads(output.read_text())['error'])

    def evidence(self):
        reports = []
        for arch, tag in [('x86_64', 'x86'), ('aarch64', 'aarch64')]:
            for env in ci.environments():
                reports.append(dict(source_sha=SHA, status='passed', mode=env['mode'],
                    lane=f"full-python-test-python-{tag}-{env['key']}", python=env['python'] + '.1',
                    dependencies={'numpy': '1.26.4' if env['numpy'][0] == '1' else '2.3.1'},
                    wheel=dict(source_sha=SHA, arch=arch, sha256='f' * 64),
                    collected=6, passed=6, failed=0, skipped=0, seconds=10.0))
        return reports

    def collect(self, reports):
        with tempfile.TemporaryDirectory() as tmp:
            for i, report in enumerate(reports):
                directory = Path(tmp, str(i))
                directory.mkdir()
                (directory / 'results.json').write_text(json.dumps(report))
            return ci.load_reports(tmp, SHA)

    def test_summary_requires_complete_matching_matrix(self):
        reports = self.evidence()
        self.assertEqual(len(self.collect(reports)), 18)
        variants = [reports[:-1], reports + [reports[0]]]
        for field, value in [('source_sha', 'b' * 40), ('status', 'failed'),
                             ('lane', reports[1]['lane']), ('python', '3.14.0'),
                             ('passed', 5), ('failed', 1), ('mode', 'openmp')]:
            bad = copy.deepcopy(reports); bad[0][field] = value; variants.append(bad)
        for field, value in [('arch', 'aarch64'), ('sha256', 'e' * 64), ('source_sha', 'b' * 40)]:
            bad = copy.deepcopy(reports); bad[0]['wheel'][field] = value; variants.append(bad)
        bad = copy.deepcopy(reports); bad[0]['dependencies']['numpy'] = '1.25.0'; variants.append(bad)
        bad = copy.deepcopy(reports); bad[-1].update(skipped=1, passed=5); variants.append(bad)
        for variant in variants:
            with self.assertRaises(ValueError):
                self.collect(variant)
        self.assertEqual(ci.markdown([]), [])
        text = '\n'.join(ci.markdown(reports))
        self.assertIn('3.12.1 / 2.3.1', text)
        self.assertIn('macOS Python is deferred', text)

    def test_parallel_python_cost_is_not_sum_as_wall_or_claimed_complete(self):
        cost = ci.timing([
            dict(name='Python Wheel Build X86', started_at='2026-11-16T00:00:00Z', completed_at='2026-11-16T00:20:00Z'),
            dict(name='Python Wheel Build Aarch64', started_at='2026-11-16T00:00:00Z', completed_at='2026-11-16T00:30:00Z'),
            dict(name='Python x86 py312-np2', started_at='2026-11-16T00:25:00Z', completed_at='2026-11-16T00:26:00Z'),
            dict(name='Unrelated', started_at='2026-11-15T00:00:00Z', completed_at='2026-11-16T00:00:00Z')])
        self.assertEqual(cost['wall_seconds'], 30 * 60)
        self.assertEqual(cost['runner_seconds'], 51 * 60)
        self.assertFalse(cost['complete'])
        self.assertIsNone(ci.timing([])['wall_seconds'])

    def test_cache_save_only_trusted_main_and_publish_artifacts_separate(self):
        build = (ROOT / '.github/actions/python-wheel-build/action.yml').read_text()
        save = build.split('name: Save trusted compiler cache', 1)[1]
        condition = re.search(r'if: (.*)', save)[1].replace('success()', 'true')
        for event in ('pull_request', 'pull_request_target', 'push', 'schedule', 'workflow_dispatch'):
            for repo in ('antgroup/vsag', 'fork/vsag'):
                for ref in ('refs/heads/main', 'refs/pull/1/merge', 'refs/heads/test'):
                    for workflow in ('Nightly CI', 'Branch CI', 'pyvsag TestPyPI', 'PR CI'):
                        for hit in ('true', 'false'):
                            actual = expression(condition, {'github.event_name': event, 'github.repository': repo,
                                'github.ref': ref, 'github.workflow': workflow, 'steps.cache.outputs.cache-hit': hit})
                            self.assertEqual(actual, repo == 'antgroup/vsag' and ref == 'refs/heads/main'
                                and workflow == 'Nightly CI' and event in ('schedule', 'workflow_dispatch') and hit != 'true')
        cache = build.split('name: Restore trusted compiler cache', 1)[1].split('    - name:', 1)[0]
        self.assertIn('steps.cache-epoch.outputs.week', cache)
        self.assertNotIn('github.run_id', cache)
        self.assertNotIn('inputs.source-sha', cache)
        self.assertIn('CCACHE_MAXSIZE=3G', build)
        self.assertIn('name: provenance-${{ inputs.artifact }}', build)
        self.assertIn('name: pyvsag-wheels-x86_64', job('pyvsag-testpypi.yml', 'publish-linux'))
        self.assertNotIn('CIBW_AFTER_BUILD', build)  # Not a supported cibuildwheel option.
        self.assertIn('python-build-report:/ci-report', build)
        self.assertIn('CIBW_BEFORE_TEST: bash /project/scripts/ci/collect_python_cache.sh /ci-report', build)
        self.assertIn('/ci-report/build-tests.json', build)
        self.assertIn('VSAG_USE_SYSTEM_DEPS=OFF', build)
        self.assertIn('VSAG_OPENBLAS_TARGET=GENERIC', build)

    def test_before_all_resets_mounted_cache_without_inherited_environment(self):
        build = (ROOT / '.github/actions/python-wheel-build/action.yml').read_text()
        command = re.search(r'(?m)^        CIBW_BEFORE_ALL: (.+)$', build)[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, script in {
                'yum': '#!/bin/sh\nexit 0\n',
                'ccache': '#!/bin/sh\nprintf "%s %s\\n" "$CCACHE_DIR" "$1"\n',
            }.items():
                executable = root / name
                executable.write_text(script)
                executable.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k != 'CCACHE_DIR'}
            env['PATH'] = str(root) + os.pathsep + env['PATH']
            result = subprocess.check_output(['bash', '-e', '-c', command], env=env, text=True)
            self.assertEqual(result.strip(), '/ccache -z')

    def test_cache_diagnostics_are_optional_and_never_mislabel_fallback(self):
        script = ROOT / 'scripts/ci/collect_python_cache.sh'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / 'ccache'
            executable.write_text('#!/bin/sh\ncase "$1" in\n'
                '  --version) echo "ccache version old";;\n'
                '  --print-stats) echo "unsupported option" >&2; exit 1;;\n'
                '  -s) echo "cache hit (direct) 5";;\nesac\n')
            executable.chmod(0o755)
            output = root / 'report'
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'])
            subprocess.run(['bash', str(script), str(output)], env=env, check=True)
            self.assertFalse((output / 'ccache.tsv').exists())
            self.assertIn('cache hit (direct) 5', (output / 'ccache.txt').read_text())
            # Even total ccache failure cannot fail the wheel build.
            executable.write_text('#!/bin/sh\nexit 7\n')
            subprocess.run(['bash', str(script), str(output)], env=env, check=True)
            self.assertFalse((output / 'ccache.tsv').exists())

    def test_wheel_library_gate_executes_before_upload_and_propagates_failures(self):
        build = (ROOT / '.github/actions/python-wheel-build/action.yml').read_text()
        self.assertLess(build.index('name: Check wheel libraries'), build.index('uses: actions/upload-artifact'))
        step = build.split('name: Check wheel libraries', 1)[1].split('    - ', 1)[0]
        command = step.split('run: ', 1)[1].strip()
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp, 'scripts/python/check_wheel_libs.py')
            script.parent.mkdir(parents=True)
            for exit_code in (0, 1):
                script.write_text(f'import sys; sys.exit({exit_code})\n')
                self.assertEqual(subprocess.run(['bash', '-e', '-c', command], cwd=tmp).returncode, exit_code)

    def test_numpy_input_explicitly_accepts_only_versions_or_patterns(self):
        for action in ('python-wheel-test', 'python-build-and-test-test-python-x86',
                       'python-build-and-test-test-python-aarch64'):
            text = (ROOT / '.github/actions' / action / 'action.yml').read_text()
            self.assertIn('numpy-version:', text)
            self.assertIn('not a comparison specifier', text)
            self.assertNotIn('inputs.numpy }}', text)

    def test_all_callers_share_build_and_nightly_waits_for_mandatory_matrix(self):
        for arch in ('x86', 'aarch64'):
            action = (ROOT / f'.github/actions/python-build-and-test-build-python-wheel-{arch}/action.yml').read_text()
            self.assertIn('uses: ./.github/actions/python-wheel-build', action)
            nightly = job('daily_test.yml', f'full-python-test-python-{arch}')
            self.assertNotIn('concurrency:', nightly)
            self.assertIn('fail-fast: false', nightly)
            self.assertIn('matrix.key', nightly)
            self.assertIn('matrix.python', nightly)
            self.assertIn(f'full-python-test-python-{arch}', job('daily_test.yml', 'daily-extra-plan'))
        self.assertIn('uses: ./.github/actions/python-wheel-build', job('pr-ci-community.yml', 'build-python-x86'))
        self.assertIn('uses: ./.github/actions/python-wheel-build', job('pyvsag-testpypi.yml', 'build'))
        self.assertIn('python_ci.lanes()', job('daily_test.yml', 'daily-plan'))
        self.assertIn('--python-reports', job('daily_test.yml', 'daily-summary'))
        test = (ROOT / '.github/actions/python-wheel-test/action.yml').read_text()
        self.assertIn('source "$SOURCE_SHA"', test)
        self.assertIn('if-no-files-found: error', test)
        self.assertNotIn('continue-on-error', test)


class MacOSWheelCITests(unittest.TestCase):
    def fixture(self, root):
        wheel = root / 'pyvsag-1.0.0.dev1-py3-none-macosx_15_0_arm64.whl'
        wheel.write_bytes(b'wheel content')
        manifest = root / 'wheel-manifest.json'
        manifest.write_text(json.dumps(ci.wheel_manifest(root, SHA, 'arm64')))
        return wheel, manifest

    def test_matrix_covers_supported_numpy_and_both_openmp_lanes(self):
        matrix = mac.environments()
        self.assertEqual(len(matrix), 10)
        self.assertEqual(len({e['key'] for e in matrix}), 10)
        self.assertEqual([e for e in matrix if e['mode'] == 'ordinary'],
                         [e for e in ci.environments() if e['mode'] == 'ordinary'])
        self.assertEqual({(e['python'], e['numpy']) for e in matrix if e['mode'] == 'openmp'},
                         {('3.12', '1.26.4'), ('3.12', '2.*')})

    def test_download_exact_testpypi_wheel_retries_and_checks_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel, manifest = self.fixture(root)
            calls = []
            def fetch(command, **kwargs):
                calls.append(command)
                if len(calls) == 1:
                    return SimpleNamespace(returncode=1)
                Path(command[command.index('--dest') + 1], wheel.name).write_bytes(wheel.read_bytes())
                return SimpleNamespace(returncode=0)
            with patch.object(mac.subprocess, 'run', side_effect=fetch), \
                    patch.object(mac.time, 'sleep') as sleep:
                result = mac.download(manifest, SHA, root / 'download')
                self.assertEqual(result.read_bytes(), wheel.read_bytes())
                sleep.assert_called_once_with(15)
            command = calls[-1]
            self.assertIn('--isolated', command)
            self.assertIn('--no-deps', command)
            self.assertIn('--no-cache-dir', command)
            self.assertIn('https://test.pypi.org/simple/', command)
            self.assertNotIn('--extra-index-url', command)
            self.assertEqual(command[-1], 'pyvsag==1.0.0.dev1')
            def corrupt(command, **kwargs):
                Path(command[command.index('--dest') + 1], wheel.name).write_bytes(b'wrong wheel')
                return SimpleNamespace(returncode=0)
            with patch.object(mac.subprocess, 'run', side_effect=corrupt) as run:
                with self.assertRaises(ValueError):
                    mac.download(manifest, SHA, root / 'corrupt')
                self.assertEqual(run.call_count, 1)
            with patch.object(mac.subprocess, 'run') as run:
                with self.assertRaises(ValueError):
                    mac.download(manifest, 'b' * 40, root / 'wrong-source')
                with self.assertRaises(ValueError):
                    mac.download(manifest, SHA, root / 'download')
                run.assert_not_called()

    def test_test_environment_is_clean_and_failures_propagate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel, manifest = self.fixture(root)
            with patch.object(mac.platform, 'system', return_value='Darwin'), \
                    patch.object(mac.platform, 'machine', return_value='arm64'), \
                    patch.object(mac.subprocess, 'check_output', return_value=SHA), \
                    patch.dict(os.environ, PYVSAG_LIBRARY='/wrong/lib.dylib', PYTHONPATH='/wrong'), \
                    patch.object(mac.subprocess, 'run') as run:
                mac.test_wheel(root, manifest, SHA, '1.26.4', 'openmp', root / 'report')
                commands = [call.args[0] for call in run.call_args_list]
                self.assertIn('venv', commands[0])
                self.assertIn(str(wheel.resolve()), commands[1])
                for package in ['numpy==1.26.4', *mac.OPENMP_PACKAGES]:
                    self.assertIn(package, commands[1])
                self.assertIn('openmp', commands[-1])
                for call in run.call_args_list:
                    self.assertTrue(call.kwargs['check'])
                    self.assertNotIn('PYVSAG_LIBRARY', call.kwargs['env'])
                    self.assertNotIn('PYTHONPATH', call.kwargs['env'])
                    self.assertEqual(call.kwargs['env']['KMP_DUPLICATE_LIB_OK'], 'FALSE')
                run.reset_mock()
                with patch.object(mac.subprocess, 'check_output', return_value='b' * 40):
                    with self.assertRaisesRegex(ValueError, 'checkout differs'):
                        mac.test_wheel(root, manifest, SHA, '2.*', 'ordinary', root / 'mismatch')
                    run.assert_not_called()
                run.side_effect = subprocess.CalledProcessError(1, 'test')
                with self.assertRaises(subprocess.CalledProcessError):
                    mac.test_wheel(root, manifest, SHA, '2.*', 'ordinary', root / 'failed')

    def test_retry_window_exhaustion_and_timeout_are_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest = self.fixture(root)
            for timeout in (False, True):
                now = [0.0]
                def sleep(seconds):
                    now[0] += seconds
                def run(command, **kwargs):
                    if timeout:
                        now[0] += kwargs['timeout']
                        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
                    return SimpleNamespace(returncode=1)
                with patch.object(mac.time, 'monotonic', side_effect=lambda: now[0]), \
                        patch.object(mac.time, 'sleep', side_effect=sleep), \
                        patch.object(mac.subprocess, 'run', side_effect=run) as runner:
                    with self.assertRaisesRegex(RuntimeError, 'after 600s retry window'):
                        mac.download(manifest, SHA, root / 'missing')
                    self.assertEqual(now[0], 600)
                    self.assertLessEqual(runner.call_count, 12)
                    self.assertGreater(runner.call_count, 1)

    def test_existing_filename_fails_before_upload_but_linux_files_do_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel, manifest = self.fixture(Path(tmp))
            for files in ([], [{'filename': 'pyvsag-1-py3-none-manylinux_2_28_aarch64.whl'}]):
                with patch.object(mac, 'urlopen', return_value=BytesIO(json.dumps({'urls': files}).encode())):
                    mac.require_unpublished(manifest, SHA)
            with patch.object(mac, 'urlopen', return_value=BytesIO(json.dumps({
                    'urls': [{'filename': wheel.name}]}).encode())):
                with self.assertRaisesRegex(ValueError, 'already exists on TestPyPI.*original upload run'):
                    mac.require_unpublished(manifest, SHA)
            for status in (404, 403, 503):
                with patch.object(mac, 'urlopen', side_effect=mac.HTTPError('url', status, 'error', {}, None)), \
                        patch.object(mac.time, 'sleep'):
                    if status == 404:
                        mac.require_unpublished(manifest, SHA)
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'publication blocked'):
                            mac.require_unpublished(manifest, SHA)

    def test_publication_lookup_retries_transient_errors_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest = self.fixture(Path(tmp))
            failures = [mac.URLError('network unavailable'), TimeoutError('timed out'),
                        mac.HTTPError('url', 503, 'unavailable', {}, None),
                        mac.HTTPError('url', 429, 'rate limited', {}, None),
                        b'<html>CDN error</html>', b'{"urls": null}', b'{"urls": [null]}']
            for failure in failures:
                def endpoint(*args, **kwargs):
                    if isinstance(failure, bytes):
                        return BytesIO(failure)
                    raise failure
                with self.subTest(failure=failure), \
                        patch.object(mac, 'urlopen', side_effect=endpoint) as request, \
                        patch.object(mac.time, 'sleep') as sleep:
                    with self.assertRaisesRegex(RuntimeError, 'after 3 attempts; publication blocked'):
                        mac.require_unpublished(manifest, SHA)
                    self.assertEqual(request.call_count, 3)
                    self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 10])
            with patch.object(mac, 'urlopen', side_effect=[mac.URLError('temporary'),
                    BytesIO(b'{"urls": []}')]) as request, patch.object(mac.time, 'sleep'):
                mac.require_unpublished(manifest, SHA)
                self.assertEqual(request.call_count, 2)

    def test_openmp_pins_match_documented_versions(self):
        for document in ['docs/agents/ci-routing.md', 'docs/docs/en/src/guide/pyvsag.md',
                         'docs/docs/zh/src/guide/pyvsag.md']:
            text = (ROOT / document).read_text()
            for requirement in mac.OPENMP_PACKAGES:
                self.assertRegex(requirement, r'^[a-z-]+==[0-9]+\.[0-9]+\.[0-9]+$')
                self.assertIn(f'`{requirement}`', text)

    def test_deployment_target_matches_setup_action_and_filename(self):
        setup = (ROOT / 'python/setup.py').read_text()
        target = re.search(r'setdefault\("MACOSX_DEPLOYMENT_TARGET", "([\d.]+)"\)', setup)[1]
        self.assertEqual(target, '15.0')
        self.assertEqual(target, mac.DEPLOYMENT_TARGET)
        self.assertIn(f'wheel default: {target}', (ROOT / 'cmake/VSAGMacOSOpenMP.cmake').read_text())
        action = (ROOT / '.github/actions/python-wheel-build-macos/action.yml').read_text()
        self.assertIn(f"MACOSX_DEPLOYMENT_TARGET: '{target}'", action)
        self.assertNotIn('brew install', action)
        with tempfile.TemporaryDirectory() as tmp:
            wheel, manifest = self.fixture(Path(tmp))
            record = json.loads(manifest.read_text())
            record['filename'] = f"pyvsag-1-py3-none-macosx_{target.replace('.', '_')}_arm64.whl"
            manifest.write_text(json.dumps(record))
            self.assertEqual(mac.published_version(manifest, SHA), '1')
            record['filename'] = record['filename'].replace('15_0', '14_0')
            manifest.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                mac.published_version(manifest, SHA)

    def test_native_wheel_tag_does_not_inherit_universal_python_architecture(self):
        # Execute the real wheel command class without invoking setup or compiling VSAG.
        setup = ROOT / 'python/setup.py'
        definition = next(node for node in ast.parse(setup.read_text()).body
                          if isinstance(node, ast.ClassDef) and node.name == 'PlatformWheel')
        for system, arch, original, expected in [
                ('darwin', 'arm64', 'macosx_15_0_universal2', 'macosx_15_0_arm64'),
                ('darwin', 'arm64', 'macosx_15_0_arm64', 'macosx_15_0_arm64'),
                ('darwin', 'arm64', 'macosx_26_0_universal2', 'macosx_26_0_arm64'),
                ('darwin', 'x86_64', 'macosx_15_0_universal2', 'macosx_15_0_x86_64'),
                ('linux', 'x86_64', 'linux_x86_64', 'linux_x86_64')]:
            base = type('Wheel', (), {'get_tag': lambda self: ('cp312', 'cp312', original)})
            namespace = {'bdist_wheel': base, 'sys': SimpleNamespace(platform=system),
                         'platform': SimpleNamespace(machine=lambda: arch)}
            exec(compile(ast.Module(body=[definition], type_ignores=[]), str(setup), 'exec'), namespace)
            with self.subTest(original=original, arch=arch):
                self.assertEqual(namespace['PlatformWheel']().get_tag(), ('py3', 'none', expected))

    def test_build_manifest_rejects_wrong_platform_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel, manifest = self.fixture(root)
            manifest.unlink()
            mac.write_manifest(root, manifest, SHA)
            self.assertEqual(mac.published_version(manifest, SHA), '1.0.0.dev1')
            manifest.unlink()
            wheel.rename(root / wheel.name.replace('15_0', '26_0'))
            with self.assertRaisesRegex(ValueError, 'MACOSX_DEPLOYMENT_TARGET=15.0.*macosx_26_0'):
                mac.write_manifest(root, manifest, SHA)
            self.assertFalse(manifest.exists())
        action = (ROOT / '.github/actions/python-wheel-build-macos/action.yml').read_text()
        self.assertIn('macos_wheel_ci.py manifest', action)

    def test_publication_is_independent_and_main_only_with_dry_run_support(self):
        for name, dependencies in [('publish-linux', ['build']),
                                   ('publish-macos', ['build-macos', 'test-macos'])]:
            publish = job('pyvsag-testpypi.yml', name)
            expected_needs = dependencies if name == 'publish-linux' else ['resolve-source', *dependencies]
            self.assertIn('needs: [' + ', '.join(expected_needs) + ']', publish)
            self.assertNotIn('pattern:', publish)
            self.assertIn('skip-existing: true', publish)
            condition = re.search(r'(?m)^    if: (.*)', publish)[1]
            for event in ('workflow_dispatch', 'schedule'):
                for enabled in (True, False):
                    for ref in ('refs/heads/main', 'refs/heads/test'):
                        for dependency in dependencies:
                            for result in ('success', 'failure', 'skipped', 'cancelled'):
                                context = {'github.event_name': event, 'github.ref': ref,
                                           'inputs.publish': enabled}
                                context.update({f'needs.{dep}.result': 'success' for dep in dependencies})
                                context[f'needs.{dependency}.result'] = result
                                actual = expression(condition.replace('vars.PYVSAG_NIGHTLY_PUBLISH',
                                                    repr('true' if enabled else 'false')), context)
                                self.assertEqual(actual, enabled and result == 'success' and ref == 'refs/heads/main')
        publish = job('pyvsag-testpypi.yml', 'publish-macos')
        self.assertLess(publish.index('check-unpublished'), publish.index('uses: pypa/'))
        source = job('pyvsag-testpypi.yml', 'resolve-source')
        self.assertIn('ref: ${{ github.sha }}', source)
        self.assertIn('test "$(git rev-parse HEAD)" = "$SOURCE_SHA"', source)
        guard = source.split('        run: |', 1)[1].split('      - uses:', 1)[0]
        for ref in ('refs/heads/main', 'refs/heads/test'):
            for publish in ('true', 'false'):
                result = subprocess.run(['bash', '-e', '-c', guard],
                                        env=dict(os.environ, GITHUB_REF=ref, PUBLISH=publish),
                                        capture_output=True)
                self.assertEqual(result.returncode == 0, ref == 'refs/heads/main' or publish == 'false')
        test = job('pyvsag-testpypi.yml', 'test-macos')
        self.assertIn('needs: [resolve-source, build-macos]', test)
        self.assertIn('fail-fast: false', test)
        self.assertNotIn('continue-on-error', test)
        self.assertIn('fromJSON(needs.resolve-source.outputs.macos-matrix)', test)
        roundtrip = job('pyvsag-testpypi.yml', 'verify-testpypi-macos')
        self.assertIn('needs: [resolve-source, publish-macos]', roundtrip)
        self.assertNotIn('always() &&', roundtrip)
        self.assertIn('name: provenance-pyvsag-wheels-macos-arm64', roundtrip)
        self.assertNotIn('name: pyvsag-wheels-macos-arm64', roundtrip)
        self.assertIn('check_macos_wheel.py', roundtrip)
        build = (ROOT / '.github/actions/python-wheel-build-macos/action.yml').read_text()
        self.assertLess(build.index('make pyvsag'), build.index('check_macos_wheel.py'))
        self.assertLess(build.index('check_macos_wheel.py'), build.index('actions/upload-artifact'))


if __name__ == '__main__':
    unittest.main()
