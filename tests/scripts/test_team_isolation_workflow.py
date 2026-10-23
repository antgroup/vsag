"""Exercise the standalone source-handoff probe without GitHub or Kubernetes."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[2]
CALLER = (ROOT / '.github/workflows/team-isolation-probe.yml').read_text()
WORKER = (ROOT / '.github/workflows/team-isolation-worker.yml').read_text()


def step(name):
    start = WORKER.index('      - name: ' + name + '\n')
    end = WORKER.find('\n      - ', start + 1)
    return WORKER[start:end if end != -1 else len(WORKER)]


def python_step(name):
    block = step(name)
    match = re.search(r"^([ ]+)python3 (?:-I )?- <<'PY'\n(.*?)^\1PY$", block, re.MULTILINE | re.DOTALL)
    if not match:
        raise AssertionError('Missing inline Python')
    return textwrap.dedent(match[2])


class IsolationWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo = self.root / 'producer'
        (self.repo / 'scripts/ci').mkdir(parents=True)
        shutil.copyfile(ROOT / 'scripts/ci/team_source.py', self.repo / 'scripts/ci/team_source.py')
        (self.repo / 'fixture.bin').write_bytes(b'\x00\xffsource fixture')
        (self.repo / 'executable.sh').write_text('#!/bin/sh\necho fixture\n')
        (self.repo / 'executable.sh').chmod(0o755)
        subprocess.run(['git', 'init', '-q', '--template=', str(self.repo)], check=True)
        subprocess.run(['git', '-C', str(self.repo), 'add', '.'], check=True)
        subprocess.run(['git', '-c', 'core.hooksPath=' + os.devnull, '-C', str(self.repo),
                        '-c', 'user.name=Test Fixture', '-c', 'user.email=fixture@example.invalid',
                        'commit', '-qm', 'fixture'], check=True)
        self.sha = subprocess.check_output(['git', '-C', str(self.repo), 'rev-parse', 'HEAD'], text=True).strip()
        self.artifacts = self.root / 'artifacts'
        self.artifacts.mkdir()
        self.env = dict(os.environ, TEAM_ARTIFACTS=str(self.artifacts), TEAM_REPOSITORY_ID='123',
                        GITHUB_RUN_ID='456', GITHUB_RUN_ATTEMPT='1', TEAM_HEAD_SHA='a' * 40,
                        TEAM_TEST_SHA=self.sha, GITHUB_WORKSPACE=str(self.repo),
                        TEAM_IMAGE='example.invalid/test@sha256:' + 'c' * 64)

    def run_inline(self, name, cwd, mode='rw', env=None):
        source = python_step(name)
        wrapper = '''import errno, os, pathlib
from unittest.mock import patch
original_read = pathlib.Path.read_text
original_write = pathlib.Path.write_text
def read_text(path, *args, **kwargs):
    if str(path) == '/proc/self/mountinfo':
        return '1 2 3 / ' + os.environ['TEAM_ARTIFACTS'] + ' MODE - ext4 /dev/disk MODE\\n'
    return original_read(path, *args, **kwargs)
def write_text(path, *args, **kwargs):
    if path.parent == pathlib.Path(os.environ['TEAM_ARTIFACTS']) and path.name.startswith('forbidden-'):
        raise OSError(errno.EROFS, 'mock read-only mount')
    return original_write(path, *args, **kwargs)
with patch('os.path.ismount', return_value=True), patch.object(pathlib.Path, 'read_text', read_text), patch.object(pathlib.Path, 'write_text', write_text):
    exec(SOURCE)
'''.replace('MODE', mode).replace('SOURCE', repr(source))
        return subprocess.run([sys.executable, '-I', '-c', wrapper], cwd=cwd, env=env or self.env,
                              capture_output=True, text=True, timeout=20)

    def produce(self):
        prepared = self.run_inline('Prepare source handoff identity', self.repo)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        # Execute the real publish command, including the copied committed helper.
        shell = textwrap.dedent(step('Publish exact-commit source').split('run: |\n')[1])
        published = subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', shell], cwd=self.repo,
                                   env=self.env, capture_output=True, text=True, timeout=20)
        self.assertEqual(published.returncode, 0, published.stderr)
        sealed = self.run_inline('Seal small run artifact', self.repo)
        self.assertEqual(sealed.returncode, 0, sealed.stderr)

    def test_exact_three_branch_allowlist_in_both_caller_and_producer(self):
        conditions = [re.search(r'^    if: (.*)$', source, re.MULTILINE)[1] for source in (CALLER, WORKER)]
        self.assertEqual(conditions[0], conditions[1])
        expected = {'codex/team-ci-isolation-probe', 'codex/team-ci-isolation-probe-b', 'codex/team-ci-isolation-probe-c'}
        for repo in ('wxyucs/vsag', 'someone/vsag'):
            for branch in sorted(expected | {'codex/team-ci-isolation-probe-anything', 'main'}):
                expression = conditions[0].replace('github.event.pull_request.head.repo.full_name', repr(repo))
                expression = expression.replace('github.event.pull_request.head.ref', repr(branch))
                expression = expression.replace('&&', 'and').replace('||', 'or')
                self.assertEqual(eval(expression, {'__builtins__': {}}, {}), repo == 'wxyucs/vsag' and branch in expected)
        self.assertIn('queue: max', CALLER)
        self.assertIn('uses: ./.github/workflows/team-isolation-worker.yml', CALLER)
        self.assertNotIn('concurrency:', WORKER)

    def test_all_inline_python_compiles_and_consumers_have_no_checkout_or_build(self):
        blocks = list(re.finditer(r"^([ ]+)python3 (?:-I )?- <<'PY'\n(.*?)^\1PY$", WORKER, re.MULTILINE | re.DOTALL))
        self.assertEqual(len(blocks), 5)
        for index, block in enumerate(blocks):
            compile(textwrap.dedent(block[2]), 'probe-inline-' + str(index), 'exec')
        self.assertEqual(WORKER.count('actions/checkout@'), 1)
        self.assertIn('ref: ${{ github.sha }}', WORKER)
        self.assertIn('persist-credentials: false', WORKER)
        self.assertNotIn('make asan', WORKER)
        self.assertNotIn('team_test_shards.py run', WORKER)
        self.assertIn('no C++ test coverage', WORKER)

    def test_actual_producer_and_four_private_consumers_match_artifact_digest(self):
        self.produce()
        expected = (self.repo / 'probe-digest.txt').read_text()
        source_hash = hashlib.sha256((self.artifacts / 'meta/source.json').read_bytes()).hexdigest()
        for index in range(4):
            workspace = self.root / ('consumer-' + str(index))
            workspace.mkdir()
            env = dict(self.env, GITHUB_WORKSPACE=str(workspace), PROBE_SHARD=str(index))
            copied = self.run_inline('Verify read-only artifacts and execution identity', workspace, 'ro', env)
            self.assertEqual(copied.returncode, 0, copied.stderr)
            result = self.run_inline('Validate read-only shared artifact', workspace, 'ro', env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((workspace / 'consumer-digest.txt').read_text(), expected)
            self.assertEqual((workspace / 'fixture.bin').read_bytes(), (self.repo / 'fixture.bin').read_bytes())
            self.assertEqual((workspace / 'executable.sh').stat().st_mode & 0o777, 0o755)
            self.assertFalse((workspace / '.git').exists())
            self.assertTrue((workspace / ('probe-private-' + str(index))).is_file())
            self.assertFalse((self.artifacts / 'source' / ('probe-private-' + str(index))).exists())
        self.assertEqual(hashlib.sha256((self.artifacts / 'meta/source.json').read_bytes()).hexdigest(), source_hash)
        inventory = json.loads((self.artifacts / 'meta/inventory.json').read_text())
        self.assertEqual(inventory['kind'], 'source-handoff-probe')
        self.assertEqual(inventory['tests'], [])
        summary = self.root / 'summary'
        downloads = summary / 'probe-results'
        (downloads / 'producer-digest').mkdir(parents=True)
        shutil.copyfile(self.repo / 'probe-digest.txt', downloads / 'producer-digest/probe-digest.txt')
        for index in range(4):
            target = downloads / ('consumer-' + str(index))
            target.mkdir()
            shutil.copyfile(self.root / ('consumer-' + str(index)) / 'consumer-digest.txt', target / 'consumer-digest.txt')
        result = self.run_inline('Validate artifact roundtrip', summary)
        self.assertEqual(result.returncode, 0, result.stderr)
        (downloads / 'consumer-2/consumer-digest.txt').write_text('wrong digest\n')
        self.assertNotEqual(self.run_inline('Validate artifact roundtrip', summary).returncode, 0)

    def test_wrong_identity_and_tampered_probe_fail_closed(self):
        self.produce()
        workspace = self.root / 'consumer'
        workspace.mkdir()
        env = dict(self.env, GITHUB_WORKSPACE=str(workspace), PROBE_SHARD='0', GITHUB_RUN_ATTEMPT='2')
        result = self.run_inline('Verify read-only artifacts and execution identity', workspace, 'ro', env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('mismatch', result.stderr)
        self.assertEqual(list(workspace.iterdir()), [])
        env['GITHUB_RUN_ATTEMPT'] = '1'
        self.assertEqual(self.run_inline('Verify read-only artifacts and execution identity', workspace, 'ro', env).returncode, 0)
        path = self.artifacts / 'probe.json'
        payload = json.loads(path.read_text()); payload['source_manifest_sha256'] = 'f' * 64
        path.write_text(json.dumps(payload))
        result = self.run_inline('Validate read-only shared artifact', workspace, 'ro', env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Probe identity/source manifest mismatch', result.stderr)
        self.assertFalse((workspace / 'consumer-digest.txt').exists())


if __name__ == '__main__':
    unittest.main()
