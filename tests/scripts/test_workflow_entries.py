"""Entry consolidation contracts: selection, dependencies, gates and local actions."""
import copy
import importlib.util
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/ci'))
from branch_entry_policy import select
from check_job_results import validate


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
    text = text.replace('always()', 'True').replace('cancelled()', 'False')
    # Evaluate this documented condition subset; never execute workflow shell code.
    token = r"'(?:[^']|'')*'|\b(?:needs|github|inputs)\.[\w.-]+|\b(?:true|false)\b"
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
    def test_exact_ten_files_and_local_workflow_targets_exist(self):
        expected = {'pr-ci.yml', 'pr-ci-team.yml', 'pr-ci-community.yml', 'pr-issue-link.yml',
                    'main-ci.yml', 'daily_test.yml', 'manual-checks.yml', 'release.yml',
                    'maintenance.yml', 'update-ci-image.yml'}
        files = list((ROOT / '.github/workflows').glob('*.y*ml'))
        self.assertEqual({p.name for p in files}, expected)
        for p in files:
            for name in re.findall(r'uses: \./\.github/workflows/([^\s]+)', p.read_text()):
                self.assertIn(name, expected)
            for name in re.findall(r'uses: \./(\.github/actions/[^\s]+)', p.read_text()):
                self.assertTrue((ROOT / name / 'action.yml').is_file(), name)

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
                'needs.changes.outputs.lint': '' if version else ('full' if heavy else 'none')}

    def test_main_keeps_smoke_and_omits_full_x86_and_arm_python(self):
        values = self.context()
        for name in ('platforms-build-asan-aarch64', 'platforms-build-asan-macos-arm64',
                     'python-build-python-wheel-x86', 'lint-clang-tidy-check'):
            self.assertTrue(condition('main-ci.yml', name, values), name)
        for name in ('platforms-build-asan-x86', 'python-build-python-wheel-aarch64',
                     'coverage-changes', 'compatibility-compatibility', 'simd-build-asan-simd'):
            self.assertFalse(condition('main-ci.yml', name, values), name)

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
                           'needs.changes.outputs.heavy': '', 'needs.changes.outputs.lint': ''})
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

    def test_schedules_select_performance_only(self):
        text = (ROOT / '.github/workflows/daily_test.yml').read_text()
        self.assertEqual(re.findall(r'(?m)^    - cron: [\"\x27]([^\"\x27]+)', text),
                         ['01 08 * * 1-5', '23 3 * * 1'])
        for cron, selected in [('01 08 * * 1-5', 'performance-performance'),
                               ('23 3 * * 1', 'build-performance-benchmark-x86')]:
            values = {'github.event_name': 'schedule', 'github.event.schedule': cron,
                      'github.ref': 'refs/heads/main', 'github.repository': 'antgroup/vsag'}
            for name in re.findall(r'(?m)^  ([\w-]+):$', text.split('jobs:\n', 1)[1]):
                self.assertEqual(condition('daily_test.yml', name, values), name == selected, name)

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
                    expected = event != 'pull_request' and ref == 'refs/heads/main' and repo == 'antgroup/vsag'
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
