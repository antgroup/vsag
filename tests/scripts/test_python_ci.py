"""Exercise wheel provenance, matrix completeness and installed-test evidence."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.ci import python_ci as ci
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
        self.assertIn('pattern: pyvsag-wheels-*', job('pyvsag-testpypi.yml', 'publish'))
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


if __name__ == '__main__':
    unittest.main()
