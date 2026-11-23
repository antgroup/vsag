"""Entry consolidation contracts: selection, dependencies, gates and local actions."""
import copy
import importlib.util
import json
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/ci'))
from branch_entry_policy import select
from check_job_results import validate
from checks_policy import matches


def job(file, name):
    text = (ROOT / '.github/workflows' / file).read_text()
    match = re.search(r'(?ms)^  ' + re.escape(name) + r':\n(.*?)(?=^  [\w-]+:\n|\Z)', text)
    if not match:
        raise AssertionError(f'Missing job {name} in {file}')
    return match[1]


def condition(file, name, values):
    body = job(file, name)
    text = re.search(r'(?m)^    if: (.+)$', body)[1].strip()
    return bool(expression(text, values))


def expression(text, values):
    text = text.removeprefix('${{').removesuffix('}}').strip()
    text = text.replace('always()', 'True').replace('cancelled()', repr(bool(values.get('cancelled()', False))))
    # Evaluate this documented condition subset; never execute workflow shell code.
    token = r"'(?:[^']|'')*'|\b(?:needs|github|inputs|steps)\.[\w.-]+|\b(?:true|false)\b"
    def value(match):
        word = match[0]
        if word.startswith("'"):
            return word
        if word in ('true', 'false'):
            return word.title()
        return repr(values.get(word, ''))
    text = re.sub(token, value, text)
    text = text.replace('&&', ' and ').replace('||', ' or ')
    text = re.sub(r'!(?!=)', ' not ', text)
    return eval(text, {'__builtins__': {}}, {'startsWith': lambda value, prefix: value.startswith(prefix),
                                                 'contains': lambda value, item: item in value})


class EntryTests(unittest.TestCase):
    def test_community_cancels_with_superseded_run_but_keeps_manual_checks(self):
        for cancelled in (False, True):
            for route in ('success', 'skipped', 'failure', 'cancelled'):
                for selected in ('success', 'skipped', 'failure', 'cancelled'):
                    values = {'cancelled()': cancelled, 'needs.route.result': route,
                              'needs.select-checks.result': selected}
                    with self.subTest(cancelled=cancelled, route=route, selected=selected):
                        self.assertEqual(condition('pr-ci.yml', 'community', values),
                                         not cancelled and selected == 'success'
                                         and route in ('success', 'skipped'))
                        # Final reporting must still expose incomplete/cancelled results.
                        self.assertTrue(condition('pr-ci.yml', 'gate', values))
        self.assertIn('cancel-in-progress: true', (ROOT / '.github/workflows/pr-ci.yml').read_text())

    def test_python_pr_paths_select_build_inputs_not_unrelated_ci_helpers(self):
        changes = job('pr-ci-community.yml', 'changes')
        def paths(name):
            block = re.search(r'(?ms)^            ' + name + r':\n(.*?)(?=^            [\w-]+:|\Z)', changes)[1]
            return re.findall(r"^              - '([^']+)'", block, re.M)
        filters = {name: paths(name) for name in ('python', 'cmake', 'wheel', 'cpp')}
        selected = [
            'python/pyproject.toml', 'python/setup.py', 'python/setup.cfg',
            'python/MANIFEST.in', 'python/requirements-build.txt',
            'python/pyvsag/_index.py', 'python/pyvsag/_version.py.in', 'python/LICENSE',
            'python/NOTICE', 'python/licenses/LLVM-OpenMP.txt', 'README.md',
            'tests/python/test_index.py', 'include/vsag/vsag_c_api.h', 'src/vsag_c_api.cpp',
            'include/vsag/index.h', 'include/vsag/expected.hpp', 'src/version.h.in',
            'cmake/vsag_c.map', 'CMakeLists.txt', 'Makefile', 'cmake/VSAGOptions.cmake',
            'extern/openblas/openblas.cmake', 'src/CMakeLists.txt', 'src/index/CMakeLists.txt',
            'src/index/options.cmake', '.github/actions/python-wheel-build/action.yml',
            '.github/actions/python-wheel-test/action.yml', 'scripts/python/check_wheel_libs.py',
            'scripts/python/prepare_python_build.sh', 'scripts/python/local_build_wheel.sh',
            'scripts/deps/install_deps_centos.sh', 'scripts/ci/python_ci.py',
            'scripts/ci/collect_python_cache.sh', '.github/workflows/pr-ci-community.yml',
        ]
        unrelated = [
            'python/README.md', 'python/docs/guide.md', 'docs/docs/en/src/guide/pyvsag.md',
            'scripts/testing/daily_measure.py', 'scripts/testing/daily_plan.py',
            'scripts/testing/team_test_shards.py', 'scripts/testing/daily_report.py',
            'scripts/testing/team_test_timings.py', 'tests/scripts/test_python_ci.py',
            'tests/scripts/test_python_license.py', 'tests/scripts/test_local_build_wheel.py',
            'tests/scripts/test_daily_plan.py', 'tests/scripts/test_daily_workflow.py',
            'scripts/deps/install_deps_ubuntu.sh',
            'scripts/python/check_macos_wheel.py', 'docker/Dockerfile.x86.base',
            'scripts/ci/daily_budget.py', 'scripts/coverage/check_cov.sh',
            'scripts/linters/prepare_lint_base.py', 'src/algorithm/hgraph.cpp',
            'src/algorithm/hgraph_test.cpp', 'tests/test_hgraph.cpp', 'examples/cpp/101_index.cpp',
            '.github/workflows/pr-ci.yml',
            '.github/workflows/main-ci.yml', '.github/workflows/daily_test.yml',
            '.github/workflows/pyvsag-testpypi.yml', '.github/workflows/pr-ci-team.yml',
            '.github/workflows/release.yml', '.github/workflows/pr-issue-link.yml',
            '.github/actions/python-macos-wheel-build/action.yml',
            '.github/actions/check-format/action.yml', 'docs/agents/ci-routing.md',
        ]
        # Use the existing matcher on the actual positive paths-filter globs. A mixed
        # list selects if any path matches; status (added/modified/deleted) is unrestricted.
        cases = [(True, [p]) for p in selected] + [(False, [p]) for p in unrelated]
        cases += [(False, unrelated), (True, unrelated + ['python/setup.cfg']), (False, [])]
        for expected, files in cases:
            for team in (False, True):
                outputs = {'github.event_name': 'pull_request', 'github.base_ref': 'main',
                           'inputs.team-linux': team, 'needs.changes.result': 'success'}
                for name, patterns in filters.items():
                    outputs[f'needs.changes.outputs.{name}'] = str(any(
                        matches(path, pattern) for path in files for pattern in patterns)).lower()
                outputs['needs.format.result'] = ('success' if condition(
                    'pr-ci-community.yml', 'format', outputs) else 'skipped')
                outputs['needs.build-python-x86.result'] = 'success' if expected else 'skipped'
                for name in ('build-python-x86', 'test-python-x86'):
                    with self.subTest(files=files, team=team, job=name):
                        self.assertEqual(condition('pr-ci-community.yml', name, outputs), expected)
                        for failure in ('failure', 'cancelled', 'skipped'):
                            self.assertFalse(condition('pr-ci-community.yml', name,
                                dict(outputs, **{'needs.changes.result': failure})))
                        self.assertFalse(condition('pr-ci-community.yml', name,
                                                   dict(outputs, **{'cancelled()': True})))
                for failed in ('failure', 'cancelled'):
                    self.assertFalse(condition('pr-ci-community.yml', 'build-python-x86',
                        dict(outputs, **{'needs.format.result': failed})))
                for failed in ('failure', 'cancelled', 'skipped'):
                    self.assertFalse(condition('pr-ci-community.yml', 'test-python-x86',
                        dict(outputs, **{'needs.build-python-x86.result': failed})))

    def test_python_gate_requires_selected_jobs_and_accepts_inapplicable_skips(self):
        body = job('pr-ci-community.yml', 'python-result')
        self.assertIn('needs: [changes, build-python-x86, test-python-x86]', body)
        self.assertTrue(condition('pr-ci-community.yml', 'python-result', {'cancelled()': True}))
        script = textwrap.dedent(body.split('        run: |\n', 1)[1])
        selection = re.search(r'RUN_PYTHON: (.+)', body)[1]
        for core, selected in ((True, True), (True, False), (False, False)):
            values = {'github.event_name': 'pull_request', 'github.base_ref': 'main',
                      'needs.changes.outputs.wheel': str(selected).lower(),
                      # Legacy filters must not reselect an excluded main PR.
                      'needs.changes.outputs.python': 'true', 'needs.changes.outputs.cmake': 'true'}
            enabled = expression(selection, values)
            expected = 'success' if core and selected else 'skipped'
            env = dict(os.environ, CORE=str(core).lower(), RUN_PYTHON=str(enabled).lower(),
                       CHANGES_RESULT='success' if core else 'skipped',
                       BUILD_RESULT=expected, TEST_RESULT=expected)
            def run(overrides):
                return subprocess.run(['bash', '-e', '-c', script], env={**env, **overrides},
                                      capture_output=True, timeout=10).returncode
            self.assertEqual(run({}), 0)
            for key in ('BUILD_RESULT', 'TEST_RESULT'):
                for bad in ('failure', 'cancelled', '', 'skipped' if selected else 'success'):
                    self.assertNotEqual(run({key: bad}), 0)
            if core:
                for bad in ('failure', 'cancelled', 'skipped', ''):
                    self.assertNotEqual(run({'CHANGES_RESULT': bad}), 0)

    def test_wheel_helper_regressions_run_without_a_native_artifact(self):
        body = job('pr-ci.yml', 'routing-tests')
        for module in ('test_python_ci', 'test_workflow_entries', 'test_daily_plan', 'test_daily_workflow',
                       'test_team_test_shards', 'test_local_build_wheel', 'test_python_license'):
            self.assertIn('tests/scripts/' + module + '.py', body)
        self.assertNotIn('download-artifact', body)
        self.assertNotIn('python-wheel-build', body)
        self.assertNotIn('needs:', body)
        team = job('pr-ci-team.yml', 'checks')
        self.assertIn('python3 -m unittest tests/scripts/test_daily_workflow.py', team)

    def test_renamed_and_deleted_wheel_inputs_remain_selected(self):
        # The existing no-renames diff contract presents both endpoints, including
        # deleted paths. Exercise it with a real repository instead of basename guesses.
        from scripts.ci.checks_policy import changed_paths
        with tempfile.TemporaryDirectory() as tmp:
            def git(*args):
                return subprocess.check_output(['git', '-C', tmp, *args], text=True).strip()
            git('init', '-q')
            git('config', 'user.name', 'Test')
            git('config', 'user.email', 'test@example.com')
            Path(tmp, 'python').mkdir()
            Path(tmp, 'python/setup.cfg').write_text('metadata\n')
            git('add', '.')
            git('commit', '-qm', 'base')
            base = git('rev-parse', 'HEAD')
            Path(tmp, 'python/setup.cfg').rename(Path(tmp, 'notes.md'))
            git('add', '-A')
            git('commit', '-qm', 'move out of scope')
            head = git('rev-parse', 'HEAD')
            event = {'action': 'synchronize', 'pull_request': {
                'base': {'sha': base}, 'head': {'sha': head}}}
            with patch('scripts.ci.checks_policy.git',
                       side_effect=lambda *args: subprocess.check_output(['git', '-C', tmp, *args])):
                paths = changed_paths('pull_request', event)
            self.assertEqual(set(paths), {'python/setup.cfg', 'notes.md'})
            block = job('pr-ci-community.yml', 'changes').split('            wheel:\n', 1)[1]
            patterns = re.findall(r"^              - '([^']+)'", block, re.M)
            self.assertTrue(any(matches(path, pat) for path in paths for pat in patterns))

    def test_exact_workflow_files_and_local_workflow_targets_exist(self):
        expected = {'pr-ci.yml', 'pr-ci-team.yml', 'pr-ci-community.yml', 'pr-issue-link.yml',
                    'main-ci.yml', 'daily_test.yml', 'nightly-coverage-badge.yml', 'manual-checks.yml', 'release.yml',
                    'maintenance.yml', 'update-ci-image.yml', 'pyvsag-testpypi.yml'}
        files = list((ROOT / '.github/workflows').glob('*.y*ml'))
        self.assertEqual({p.name for p in files}, expected)
        for p in files:
            for name in re.findall(r'uses: \./\.github/workflows/([^\s]+)', p.read_text()):
                self.assertIn(name, expected)
            for name in re.findall(r'uses: \./(\.github/actions/[^\s]+)', p.read_text()):
                self.assertTrue((ROOT / name / 'action.yml').is_file(), name)

    def test_pyvsag_testpypi_workflow_shape(self):
        text = (ROOT / '.github/workflows/pyvsag-testpypi.yml').read_text()
        self.assertEqual(re.findall(r"cron: '([^']+)'", text), ['17 19 * * *'])
        self.assertRegex(text, r'publish:\n(?:.*\n)*?\s+default: true')
        self.assertIn("vars.PYVSAG_NIGHTLY_PUBLISH != 'false'", text)
        self.assertNotRegex(text, r'^\s+arch:\s*\n\s+description', re.M)
        # A nightly is skipped when the last successful nightly already built this main commit.
        self.assertIn('event=schedule&status=success', text)
        self.assertIn("needs.resolve-source.outputs.skip != 'true'", text)
        # The matrix context is not available in jobs.<id>.if; GitHub rejects the workflow.
        for condition in re.findall(r'^    if: (.*)$', text, re.M):
            self.assertNotIn('matrix', condition)
        for arch in ('x86_64', 'aarch64'):
            self.assertIn('{arch: %s,' % arch, text)
        # Callers share container construction, but retain their own source selection.
        self.assertIn("uses: ./.github/actions/python-wheel-build", text)
        self.assertIn("source-sha: ${{ needs.resolve-source.outputs.sha }}", text)
        # Trusted Publishing: the environment name must match the TestPyPI publisher entry.
        for name in ('publish-linux', 'publish-macos'):
            body = job('pyvsag-testpypi.yml', name)
            self.assertIn('name: testpypi', body)
            self.assertIn('id-token: write', body)
            self.assertNotIn('password:', body)
            self.assertNotIn('secrets.', body)
        text = (ROOT / ".github/actions/python-wheel-build/action.yml").read_text()
        self.assertIn('scripts/python/check_wheel_libs.py', text)
        self.assertIn('VSAG_USE_SYSTEM_DEPS=OFF', text)
        self.assertLess(text.index('name: Check wheel libraries'), text.index('name: Record wheel provenance'))
        # The ccache directory must be mounted into the container and enabled for the build.
        self.assertIn('create_args: -v ${{ runner.temp }}/pyvsag-ccache:/ccache', text)
        self.assertIn('VSAG_ENABLE_CCACHE=ON CCACHE_DIR=/ccache', text)
        self.assertIn('CCACHE_BASEDIR=/project CCACHE_NOHASHDIR=1', text)
        # Bound shared cache size; only the Nightly workflow publishes weekly snapshots.
        self.assertIn('CCACHE_MAXSIZE=3G CCACHE_COMPRESS=1', text)

    def test_pypi_publish_action_uses_existing_ref(self):
        # pypa/gh-action-pypi-publish has no `v1` tag; the moving ref is the release/v1 branch.
        for workflow in (ROOT / '.github/workflows').glob('*.y*ml'):
            with self.subTest(workflow=workflow.name):
                self.assertNotRegex(workflow.read_text(), r'pypa/gh-action-pypi-publish@v1\b(?!\.)')

    def test_consumers_do_not_pull_shared_architecture_tags(self):
        legacy_tag = r'(?:vsaglib/vsag|ghcr\.io/antgroup/vsag):ci-(?:x86|aarch64)-'
        for workflow in (ROOT / '.github/workflows').glob('*.y*ml'):
            if workflow.name == 'update-ci-image.yml':
                continue  # Local aliases in the publisher connect consecutive Dockerfile builds.
            with self.subTest(workflow=workflow.name):
                self.assertNotRegex(workflow.read_text(), legacy_tag)

    def test_branch_scope_original_version_docs_and_simd(self):
        event = {'ref': 'refs/heads/0.18'}
        self.assertEqual(select('push', event, 'full-branch', ['docs/a.md']),
                         dict(profile='full-branch', version=True, non_docs=False, simd=False))
        self.assertTrue(select('push', event, 'full-branch', ['README.md'])['non_docs'])
        for ref in ('refs/heads/main', 'refs/heads/0.18'):
            self.assertTrue(select('push', {'ref': ref}, 'full-branch', ['src/simd/a.cpp'])['simd'])
        self.assertFalse(select('push', dict(event, deleted=True), 'full-branch', ['src/a.cpp'])['version'])
        with self.assertRaises(ValueError):
            select('workflow_dispatch', {}, 'typo', [])

    def context(self, version=False, heavy=True, profile='full-branch'):
        return {'github.event_name': 'push', 'github.ref': 'refs/heads/0.18' if version else 'refs/heads/main',
                'needs.entry-scope.result': 'success', 'needs.entry-scope.outputs.profile': profile,
                'needs.entry-scope.outputs.version': str(version).lower(),
                'needs.entry-scope.outputs.non_docs': str(heavy).lower(),
                'needs.entry-scope.outputs.simd': 'false',
                'needs.changes.result': 'skipped' if version else 'success',
                'needs.changes.outputs.heavy': '' if version else str(heavy).lower(),
                'needs.changes.outputs.python': '' if version else str(heavy).lower(),
                'needs.changes.outputs.lint': '' if version else ('full' if heavy else 'none')}

    def test_main_keeps_smoke_and_python_x86_only(self):
        values = self.context()
        for name in ('platforms-build-asan-aarch64', 'platforms-build-asan-macos-arm64',
                     'lint-clang-tidy-check', 'python-build-python-wheel-x86'):
            self.assertTrue(condition('main-ci.yml', name, values), name)
        for name in ('platforms-build-asan-x86', 'python-build-python-wheel-aarch64',
                     'coverage-changes', 'compatibility-compatibility', 'simd-build-asan-simd'):
            self.assertFalse(condition('main-ci.yml', name, values), name)

    def test_python_main_x86_and_manual_version_scope(self):
        names = ('python-build-python-wheel-x86', 'python-build-python-wheel-aarch64',
                 'python-test-python-x86', 'python-test-python-aarch64', 'python-clean-up', 'python')
        for scope in ('main', 'version', 'manual', 'manual-full'):
            values = self.context(version=scope == 'version', profile='python' if scope == 'manual' else 'full-branch')
            if scope.startswith('manual'):
                values.update({'github.event_name': 'workflow_dispatch',
                               'needs.changes.outputs.python': 'true'})
            for name in names:
                values['needs.' + name + '.result'] = 'success'
            for name in names:
                expected = scope in ('version', 'manual') or (scope in ('main', 'manual-full') and name in
                    ('python-build-python-wheel-x86', 'python-test-python-x86', 'python-clean-up', 'python'))
                self.assertEqual(condition('main-ci.yml', name, values), expected, (scope, name))
        self.assertIn('python: ${{ steps.policy.outputs.python }}', job('main-ci.yml', 'changes'))

    def test_main_docs_skips_heavy_and_version_keeps_full(self):
        for version, heavy in [(False, False), (True, True)]:
            values = self.context(version, heavy)
            for name in ('platforms-build-asan-x86', 'platforms-build-asan-aarch64',
                         'python-build-python-wheel-aarch64', 'lint-clang-tidy-check',
                         'compatibility-compatibility'):
                self.assertEqual(condition('main-ci.yml', name, values), version, name)

    def test_manual_profiles_select_only_requested_group(self):
        for profile in ('asan', 'python', 'tsan', 'lint', 'coverage', 'compatibility', 'simd'):
            values = self.context(profile=profile)
            values.update({'github.event_name': 'workflow_dispatch', 'needs.changes.result': 'skipped',
                           'needs.changes.outputs.heavy': '', 'needs.changes.outputs.python': '', 'needs.changes.outputs.lint': ''})
            for selected, name in [('asan', 'platforms-build-asan-x86'), ('python', 'python-build-python-wheel-aarch64'),
                                   ('tsan', 'tsan-changes'), ('lint', 'lint-clang-tidy-check'),
                                   ('coverage', 'coverage-changes'), ('compatibility', 'compatibility-compatibility'),
                                   ('simd', 'simd-build-asan-simd')]:
                self.assertEqual(condition('main-ci.yml', name, values), selected == profile, name)

    def test_consumers_do_not_start_after_build_failure(self):
        values = self.context()
        for state in ('failure', 'cancelled', 'skipped', ''):
            values['needs.platforms-build-asan-aarch64.result'] = state
            self.assertFalse(condition('main-ci.yml', 'platforms-test-asan-aarch64', values))
        values['needs.platforms-build-asan-aarch64.result'] = 'success'
        self.assertTrue(condition('main-ci.yml', 'platforms-test-asan-aarch64', values))

    def test_nightly_schedule_selects_complete_regression_only(self):
        text = (ROOT / '.github/workflows/daily_test.yml').read_text()
        self.assertEqual(re.findall(r"(?m)^    - cron: ['\"]([^'\"]+)", text),
                         ['0 14 * * *', '23 3 * * 1'])
        names = re.findall(r'(?m)^  ([\w-]+):$', text.split('jobs:\n', 1)[1])
        benchmarks = {'performance-performance', 'build-performance-benchmark-x86'}
        regression = set(names) - benchmarks
        aggregate = re.search(r'(?m)^  VSAG_DAILY_AGGREGATE: (.+)$', text)[1]
        for cron, repo, ref, selected in (
            ('0 14 * * *', 'antgroup/vsag', 'refs/heads/main', regression),
            ('23 3 * * 1', 'antgroup/vsag', 'refs/heads/main', {'build-performance-benchmark-x86'}),
            ('01 08 * * 1-5', 'antgroup/vsag', 'refs/heads/main', set()),
            ('0 15 * * *', 'antgroup/vsag', 'refs/heads/main', set()),
            ('0 14 * * *', 'contributor/vsag', 'refs/heads/main', set()),
            ('0 14 * * *', 'antgroup/vsag', 'refs/heads/1.1', set()),
            ('23 3 * * 1', 'contributor/vsag', 'refs/heads/main', set()),
        ):
            with self.subTest(cron=cron, repo=repo, ref=ref):
                values = {'github.event_name': 'schedule', 'github.event.schedule': cron,
                          'github.ref': ref, 'github.repository': repo}
                # Evaluate routing with successful dependencies; dependency failures
                # are separately checked below and by the final result reconciler.
                values.update({f'needs.{name}.result': 'success' for name in names})
                for name in names:
                    self.assertEqual(condition('daily_test.yml', name, values), name in selected, name)
                self.assertEqual(expression(aggregate, values), '1' if selected == regression else '0')

    def test_nightly_openblas_uses_its_own_cache_and_only_canonical_schedule_can_save(self):
        body = job('daily_test.yml', 'daily_build_asan_x86')
        cache = body.split('- name: Load Cache\n', 1)[1].split('- name:', 1)[0]
        key = re.search(r'(?m)^          key: (.+)$', cache)[1]
        self.assertTrue(key.startswith('build-x86-asan-openblas-source-'), key)
        self.assertIn("hashFiles('./CMakeLists.txt')", key)
        self.assertIn("hashFiles('./.circleci/fresh_ci_cache.commit')", key)
        self.assertNotIn('restore-keys:', cache)
        save = re.search(r'(?m)^          save: (.+)$', cache)[1]
        for event, repository, ref, expected in (
                ('schedule', 'antgroup/vsag', 'refs/heads/main', True),
                ('workflow_dispatch', 'antgroup/vsag', 'refs/heads/main', False),
                ('pull_request', 'antgroup/vsag', 'refs/heads/main', False),
                ('push', 'antgroup/vsag', 'refs/heads/main', False),
                ('schedule', 'contributor/vsag', 'refs/heads/main', False),
                ('schedule', 'antgroup/vsag', 'refs/heads/1.2', False),
                ('schedule', 'antgroup/vsag', 'refs/tags/main', False)):
            with self.subTest(event=event, repository=repository, ref=ref):
                values = {'github.event_name': event, 'github.repository': repository, 'github.ref': ref}
                self.assertIs(expression(save, values), expected)

    def test_nightly_manual_and_reusable_profiles_keep_their_scope(self):
        text = (ROOT / '.github/workflows/daily_test.yml').read_text()
        names = re.findall(r'(?m)^  ([\w-]+):$', text.split('jobs:\n', 1)[1])
        benchmarks = {'performance-performance', 'build-performance-benchmark-x86'}
        for event in ('workflow_dispatch', 'push'):
            for profile in ('', 'regression', 'performance', 'build-performance'):
                with self.subTest(event=event, profile=profile):
                    selected = (set(names) - benchmarks if profile in ('', 'regression') else
                                {'performance-performance' if profile == 'performance' else
                                 'build-performance-benchmark-x86'})
                    values = {'github.event_name': event, 'inputs.profile': profile,
                              'github.repository': 'antgroup/vsag', 'github.ref': 'refs/heads/main',
                              'needs.full-coverage-changes.outputs.non_markdown': 'true'}
                    values.update({f'needs.{name}.result': 'success' for name in names})
                    for name in names:
                        self.assertEqual(condition('daily_test.yml', name, values), name in selected, name)
        # Manual regression replay of a PR/validation branch remains available.
        values.update({'github.event_name': 'workflow_dispatch', 'inputs.profile': 'regression',
                       'github.ref': 'refs/heads/validation'})
        for name in names:
            self.assertEqual(condition('daily_test.yml', name, values), name not in benchmarks, name)

    def test_nightly_summary_is_selected_even_after_upstream_failure(self):
        values = {'github.event_name': 'schedule', 'github.event.schedule': '0 14 * * *',
                  'github.repository': 'antgroup/vsag', 'github.ref': 'refs/heads/main'}
        for result in ('failure', 'cancelled', 'skipped'):
            values['needs.daily-plan.result'] = result
            values['needs.daily-extra-x86.result'] = result
            self.assertTrue(condition('daily_test.yml', 'daily-summary', values))
        self.assertIn('always()', job('daily_test.yml', 'daily-summary'))
        self.assertIn('NEEDS_JSON: ${{ toJSON(needs) }}', job('daily_test.yml', 'daily-summary'))
        self.assertIn('--expected-file', job('daily_test.yml', 'daily-summary'))

    def test_gate_rejects_missing_failed_cancelled_and_unexpected_skip(self):
        good = {'build': {'result': 'success'}, 'tests': {'result': 'success'}, 'unused': {'result': 'skipped'}}
        self.assertTrue(validate(good, ['build', 'tests']))
        for state in ('failure', 'cancelled', 'skipped', 'abandoned'):
            changed = copy.deepcopy(good); changed['tests']['result'] = state
            self.assertFalse(validate(changed, ['build', 'tests']))
        self.assertFalse(validate(good, ['missing']))
        self.assertFalse(validate(good, []))
        self.assertFalse(validate(good, ['build', 'build']))

    def test_version_tsan_respects_caller_core_scope(self):
        for target, core, expected in [('main', True, False), ('0.18', True, True),
                                       ('0.x/nested', False, False), ('0.x/nested', True, False),
                                       ('feature', False, False)]:
            values = {'github.event_name': 'pull_request', 'github.base_ref': target,
                      'inputs.core': core, 'needs.version-tsan-changes.result': 'success',
                      'needs.version-tsan-changes.outputs.non_markdown': 'true',
                      'needs.version-tsan-build-tsan.result': 'success'}
            for name in ('version-tsan-changes', 'version-tsan-build-tsan', 'version-tsan-test-tsan'):
                self.assertEqual(condition('pr-ci-community.yml', name, values), expected, (target, name))

    def test_tsan_gate_accepts_only_intentional_markdown_skip(self):
        body = job('main-ci.yml', 'tsan')
        required_expression = re.search(r'(?m)^          REQUIRED: (.+)$', body)[1]
        for event, non_markdown, build_required in [('pull_request', 'false', False),
                                                   ('pull_request', 'true', True),
                                                   ('push', 'false', True),
                                                   ('workflow_dispatch', '', True)]:
            required = json.loads(expression(required_expression, {
                'github.event_name': event, 'needs.tsan-changes.outputs.non_markdown': non_markdown}))
            self.assertEqual(required, ['tsan-changes'] +
                             (['tsan-build-tsan', 'tsan-test-tsan'] if build_required else []))
            results = {'tsan-changes': {'result': 'success'},
                       'tsan-build-tsan': {'result': 'skipped'}, 'tsan-test-tsan': {'result': 'skipped'}}
            self.assertEqual(validate(results, required), not build_required)
            results['tsan-changes']['result'] = 'failure'
            self.assertFalse(validate(results, required))

    def test_benchmark_uses_call_input_even_when_event_ref_differs(self):
        body = job('daily_test.yml', 'build-performance-benchmark-x86')
        requested = re.search(r'(?m)^          REQUESTED_REF: (.+)$', body)[1]
        sha = 'a' * 40
        for event_ref in ('', 'main', 'b' * 40):
            self.assertEqual(expression(requested, {'inputs.ref': sha,
                             'github.event.inputs.ref': event_ref}), sha)
        self.assertEqual(expression(requested, {}), 'main')
        for locale in ('en', 'zh'):
            docs = (ROOT / f'docs/docs/{locale}/src/development/building.md').read_text()
            self.assertNotIn('build_performance.yml', docs)
            self.assertIn('gh workflow run daily_test.yml --ref main -f profile=build-performance -f ref=', docs)

    def test_runtime_benchmark_requires_trusted_main_context(self):
        for event in ('workflow_dispatch', 'push', 'pull_request', 'schedule'):
            for ref in ('refs/heads/main', 'refs/heads/feature', 'refs/pull/1/merge'):
                for repo in ('antgroup/vsag', 'contributor/vsag'):
                    values = {'github.event_name': event, 'github.ref': ref,
                              'github.repository': repo, 'inputs.profile': 'performance',
                              'github.event.schedule': '01 08 * * 1-5'}
                    expected = event in ('workflow_dispatch', 'push') and ref == 'refs/heads/main' and repo == 'antgroup/vsag'
                    self.assertEqual(condition('daily_test.yml', 'performance-performance', values), expected)

    def test_runtime_benchmark_uses_shared_ref_validation_and_independent_runs(self):
        runtime = job('daily_test.yml', 'performance-performance')
        build = job('daily_test.yml', 'build-performance-benchmark-x86')
        self.assertIn('run: &approved-benchmark-ref |', runtime)
        self.assertIn('run: *approved-benchmark-ref', build)
        for body in (runtime, build):
            self.assertIn('repository: antgroup/vsag', body)
            self.assertIn('ref: main', body)
            self.assertIn('fetch-depth: 0', body)
            self.assertIn('persist-credentials: false', body)
            self.assertIn("REQUESTED_REF: ${{ inputs.ref || 'main' }}", body)
        self.assertIn('group: runtime-performance-${{ github.run_id }}-${{ github.run_attempt }}', runtime)
        self.assertIn('cancel-in-progress: false', runtime)
        before_report, report = runtime.split('      - name: Send Report', 1)
        self.assertNotIn('secrets.', before_report)
        self.assertIn('secrets.PERF_DINGDING_ACCESS_TOKEN', report)
        self.assertLess(runtime.index('run: &approved-benchmark-ref'), runtime.index('- name: Install Deps'))

    def test_team_group_and_gate_preserved(self):
        text = (ROOT / '.github/workflows/pr-ci.yml').read_text()
        for path in ('pr-ci-team.yml', 'pr-ci-community.yml'):
            self.assertIn('uses: ./.github/workflows/' + path, text)
        self.assertIn('PR CI Gate', text)
        self.assertNotRegex(text, r'(?m)^\s+types:.*edited')
        self.assertNotIn('pull_request_target', text)

    def test_maintenance_not_triggered_by_prs_and_docs_require_diff(self):
        text = (ROOT / '.github/workflows/maintenance.yml').read_text()
        self.assertNotRegex(text, r'(?m)^  pull_request')
        self.assertIn("needs.scope.outputs.docs == 'true'", job('maintenance.yml', 'sync'))
        self.assertIn("github.event_name == 'push'", job('maintenance.yml', 'contributors'))


if __name__ == '__main__':
    unittest.main()
