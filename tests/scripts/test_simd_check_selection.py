#!/usr/bin/env python3
"""Content-aware SIMD routing must spare Actions upgrades without hiding builds."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.ci import checks_policy as policy
from scripts.ci.action_revision_diff import action_revision_only

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = '''name: Checks
on: pull_request
jobs:
  simd:
    runs-on: ubuntu-24.04
    container: {image: example/build:main}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          persist-credentials: false
      - run: make cov-simd COMPILE_JOBS=4
      - uses: actions/upload-artifact@v5
        with: {path: report, name: simd}
'''
COMPOSITE = '''name: Dependencies
runs:
  using: composite
  steps:
    - uses: actions/checkout@v4
      with:
        repository: example/library
        ref: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        path: .ci-fetchcontent/library
        persist-credentials: false
'''


class ActionRevisions(unittest.TestCase):
    def test_versions_and_sha_pins(self):
        for before, after in [
            (WORKFLOW, WORKFLOW.replace('@v4', '@v5').replace('@v5\n        with: {', '@v6\n        with: {')),
            (COMPOSITE, COMPOSITE.replace('@v4', '@v5')),
            (WORKFLOW, WORKFLOW.replace('@v4', '@v1.2.3-rc.1+build.5')),
            (WORKFLOW.replace('@v4', '@v1.2.3-rc.1+build.5'), WORKFLOW.replace('@v4', '@v1.2.3+build.6')),
            (WORKFLOW, WORKFLOW.replace('@v4', '@' + 'a' * 40)),
            (WORKFLOW.replace('@v4', '@' + 'a' * 40), WORKFLOW.replace('@v4', '@' + 'b' * 40)),
        ]:
            self.assertTrue(action_revision_only(before, after))
        self.assertFalse(action_revision_only(WORKFLOW, WORKFLOW))

    def test_meaningful_changes_are_retained_with_version_bump(self):
        for old, new in [
            ('COMPILE_JOBS=4', 'COMPILE_JOBS=8'),
            ('example/build:main', 'example/build:other'),
            ('fetch-depth: 0', 'fetch-depth: 1'),
            ('persist-credentials: false', 'persist-credentials: true'),
            ('actions/checkout', 'someone/checkout'),
            ('path: report', 'path: other'),
            ('ubuntu-24.04', 'ubuntu-26.04'),
            ('on: pull_request', 'on: push'),
        ]:
            with self.subTest(new=new):
                self.assertFalse(action_revision_only(WORKFLOW, WORKFLOW.replace('@v4', '@v5').replace(old, new)))
        for old, new in [('example/library', 'example/other'), ('a' * 40, 'b' * 40),
                         ('.ci-fetchcontent/library', '.ci-fetchcontent/other')]:
            self.assertFalse(action_revision_only(COMPOSITE, COMPOSITE.replace('@v4', '@v5').replace(old, new)))

    def test_run_text_is_not_an_action(self):
        before = WORKFLOW + '      - run: |\n          uses: actions/checkout@v4\n'
        self.assertFalse(action_revision_only(before, before.replace('@v4', '@v5')))
        before = WORKFLOW + '    env:\n      SPEC: "uses: actions/checkout@v4"\n'
        self.assertFalse(action_revision_only(before, before.replace('@v4', '@v5')))

    def test_ambiguous_or_unknown_content_is_not_exempt(self):
        for text in ['', 'jobs: [', 'jobs: {}\njobs: {}\n',
                     WORKFLOW + 'name: Duplicate\n',
                     WORKFLOW.replace('steps:', 'steps: &steps'),
                     WORKFLOW + '  another:\n    steps: *steps\n',
                     WORKFLOW.replace('run: make', 'run: !custom make'),
                     WORKFLOW.replace('@v4', '@main'),
                     WORKFLOW.replace('actions/checkout@v4', './local-action'),
                     WORKFLOW.replace('actions/checkout@v4', 'docker://image:v4')]:
            with self.subTest(text=text[:60]):
                self.assertFalse(action_revision_only(text, text.replace('v4', 'v5')))
        self.assertFalse(action_revision_only(WORKFLOW, WORKFLOW + '      - run: echo added\n'))
        self.assertFalse(action_revision_only(WORKFLOW, WORKFLOW.replace('      - run: make cov-simd COMPILE_JOBS=4\n', '')))

    def test_missing_parser_keeps_coverage(self):
        with patch.dict('sys.modules', {'yaml': None}):
            self.assertFalse(action_revision_only(WORKFLOW, WORKFLOW.replace('@v4', '@v5')))


class Routing(unittest.TestCase):
    def test_native_and_measurement_paths_always_select(self):
        event = {'action': 'synchronize', 'pull_request': {'base': {'ref': 'main'}}}
        for path in ['src/simd/fp32_simd.cpp', 'src/simd/fp32_simd_test.cpp',
                     'src/quantization/scalar_quantization/local_scalar_quantizer_test.cpp',
                     'src/quantization/scalar_quantization/local_scalar_quantizer.h',
                     'include/vsag/vsag.h', 'Makefile', 'cmake/Options.cmake',
                     'src/simd/CMakeLists.txt', 'extern/catch2/catch2.cmake',
                     'scripts/coverage/run_simd_coverage.sh', 'tests/fixtures/example.h',
                     'scripts/ci/checks_policy.py', 'scripts/ci/action_revision_diff.py',
                     'tests/scripts/test_simd_check_selection.py']:
            with self.subTest(path=path):
                self.assertTrue(policy.select('pull_request', event, [path])['simd'])
        path = '.github/workflows/pr-ci-community.yml'
        before = policy.select('pull_request', event, [path])
        after = policy.select('pull_request', event, [path], {path})
        self.assertTrue(before['simd'])
        self.assertFalse(after['simd'])
        self.assertEqual({k: v for k, v in before.items() if k != 'simd'},
                         {k: v for k, v in after.items() if k != 'simd'})
        self.assertTrue(policy.select('pull_request', event, [path, 'src/simd/a.cpp'], {path})['simd'])

    def test_workflow_installs_parser_and_runs_regressions(self):
        import yaml
        workflow = yaml.load((ROOT / '.github/workflows/pr-ci.yml').read_text(), Loader=yaml.BaseLoader)
        jobs = workflow['jobs']
        tests = next(step for step in jobs['routing-tests']['steps']
                     if step.get('name') == 'Test SIMD check selection')
        self.assertIn('tests.scripts.test_simd_check_selection', tests['run'])
        self.assertIn('scripts/ci/check-selection-requirements.txt', tests['run'])
        select = next(step for step in jobs['select-checks']['steps'] if step.get('id') == 'select')
        self.assertEqual(select['run'].split(), ['bash', 'scripts/ci/select_checks.sh'])



class GitIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.git('init', '-q')
        self.git('config', 'user.name', 'Test Fixture')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.path = '.github/workflows/pr-ci-community.yml'
        self.write(self.path, WORKFLOW)
        self.write('.github/actions/prepare-fetchcontent/action.yml', COMPOSITE)
        self.git('add', '.')
        self.git('commit', '-qm', 'base')
        self.base = self.git('rev-parse', 'HEAD').decode().strip()

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], stderr=subprocess.DEVNULL)

    def write(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def run_policy(self, bootstrap_failure=None):
        self.git('add', '-A')
        self.git('commit', '-qm', 'candidate')
        head = self.git('rev-parse', 'HEAD').decode().strip()
        event = {'action': 'synchronize', 'pull_request': {
            'base': {'ref': 'main', 'sha': self.base}, 'head': {'sha': head}}}
        event_file = self.root / 'event.json'
        event_file.write_text(json.dumps(event))
        output = self.root / 'outputs'
        command = [os.sys.executable, str(ROOT / 'scripts/ci/checks_policy.py')]
        extra_env = {}
        if bootstrap_failure:
            # Execute the real bootstrap with controlled venv/pip failures, no network.
            self.write('scripts/ci/checks_policy.py', (ROOT / 'scripts/ci/checks_policy.py').read_text())
            self.write('scripts/ci/checks_paths.json', (ROOT / 'scripts/ci/checks_paths.json').read_text())
            bindir = self.root / 'bin'
            bindir.mkdir()
            fake = bindir / 'python3'
            fake.write_text('#!/bin/bash\nif [[ "$1 $2" == "-m venv" ]]; then\n' +
                            ('exit 1\n' if bootstrap_failure == 'venv' else 'mkdir -p "$3/bin"\nexit 0\n') +
                            'fi\nexec "' + os.sys.executable + '" "$@"\n')
            fake.chmod(0o755)
            if bootstrap_failure == 'pip':
                self.write('check-selection/bin/pip', '#!/bin/bash\nexit 1\n')
                (self.root / 'check-selection/bin/pip').chmod(0o755)
            extra_env = {'PATH': str(bindir) + ':' + os.environ['PATH'], 'RUNNER_TEMP': str(self.root)}
            command = ['bash', str(ROOT / 'scripts/ci/select_checks.sh')]
        result = subprocess.run(command,
                                cwd=self.root, capture_output=True, text=True,
                                env={**os.environ, **extra_env, 'GITHUB_EVENT_NAME': 'pull_request',
                                     'GITHUB_EVENT_PATH': str(event_file), 'GITHUB_OUTPUT': str(output)})
        self.assertEqual(result.returncode, 0, result.stderr)
        if bootstrap_failure:
            self.assertIn('::warning::', result.stdout)
        return dict(line.split('=', 1) for line in output.read_text().splitlines())

    def test_all_relevant_workflows_revision_only_skip(self):
        self.write(self.path, WORKFLOW.replace('@v4', '@v5'))
        self.write('.github/actions/prepare-fetchcontent/action.yml', COMPOSITE.replace('@v4', '@v5'))
        self.assertEqual(self.run_policy()['simd'], 'false')

    def test_venv_failure_retains_original_selection(self):
        self.write(self.path, WORKFLOW.replace('@v4', '@v5'))
        result = self.run_policy('venv')
        self.assertEqual(result['simd'], 'true')
        self.assertEqual(result['format'], 'false')

    def test_pip_failure_retains_original_selection(self):
        self.write(self.path, WORKFLOW.replace('@v4', '@v5'))
        self.assertEqual(self.run_policy('pip')['simd'], 'true')

    def test_parser_failure_does_not_select_unrelated_docs(self):
        self.write('docs/note.md', 'Notes\n')
        result = self.run_policy('pip')
        self.assertEqual(result['simd'], 'false')
        self.assertEqual(result['docs'], 'true')

    def test_mixed_native_change_runs(self):
        self.write(self.path, WORKFLOW.replace('@v4', '@v5'))
        self.write('src/simd/a.cpp', 'int a;\n')
        self.assertEqual(self.run_policy()['simd'], 'true')

    def test_dependency_pin_change_runs(self):
        self.write('.github/actions/prepare-fetchcontent/action.yml', COMPOSITE.replace('@v4', '@v5').replace('a' * 40, 'b' * 40))
        self.assertEqual(self.run_policy()['simd'], 'true')

    def test_deleted_file_runs(self):
        (self.root / self.path).unlink()
        self.assertEqual(self.run_policy()['simd'], 'true')

    def test_added_file_runs(self):
        self.write('.github/workflows/pr-simd-coverage.yml', WORKFLOW)
        self.assertEqual(self.run_policy()['simd'], 'true')

    def test_rename_out_of_scope_runs(self):
        (self.root / self.path).rename(self.root / 'unrelated.yml')
        self.assertEqual(self.run_policy()['simd'], 'true')

    def test_unreadable_blob_keeps_selection(self):
        event = {'action': 'synchronize', 'pull_request': {
            'base': {'sha': self.base}, 'head': {'sha': self.base}}}
        with patch.object(policy, 'git', side_effect=[self.base.encode(), subprocess.CalledProcessError(128, 'git')]):
            self.assertEqual(policy.action_revision_paths('pull_request', event, [self.path]), set())


if __name__ == '__main__':
    unittest.main()
