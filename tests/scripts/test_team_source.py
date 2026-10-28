"""Exercise exact-commit source publication and the pre-source workflow bootstrap."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

from tests.scripts.test_build_benchmark_workflow import python_step

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('team_source', ROOT / 'scripts/ci/team_source.py')
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)
WORKFLOW = (ROOT / '.github/workflows/pr-ci-team.yml').read_text()
BOOTSTRAP = python_step('Verify read-only artifacts and execution identity', WORKFLOW)


class TeamSourceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.git('init', '-q', '--template=')
        (self.repo / 'scripts').mkdir()
        (self.repo / 'scripts/check.sh').write_text('#!/bin/sh\nprintf source-ok\\n\n')
        (self.repo / 'scripts/check.sh').chmod(0o755)
        (self.repo / 'fixtures').mkdir()
        (self.repo / 'fixtures/data.bin').write_bytes(b'\0\1fixture\xff')
        # Export attributes must not hide tracked fixtures from consumers.
        (self.repo / '.gitattributes').write_text('fixtures/data.bin export-ignore\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Test Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
        self.sha = self.git('rev-parse', 'HEAD').strip()
        self.artifacts = self.root / 'artifacts'
        (self.artifacts / 'meta').mkdir(parents=True)
        (self.artifacts / 'build').mkdir()
        self.context = {'repository_id': '123', 'run_id': '456', 'run_attempt': '1',
                        'head_sha': 'a' * 40, 'test_sha': self.sha,
                        'image': 'example.invalid/image@sha256:' + 'c' * 64}
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir()

    def git(self, *args):
        env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1')
        return subprocess.check_output(['git', '-c', 'core.hooksPath=' + os.devnull, '-C', str(self.repo), *args],
                                       env=env, text=True, stderr=subprocess.PIPE)

    def publish(self):
        return SOURCE.publish(self.repo, self.artifacts / 'source', self.artifacts / 'meta/source.json', self.context)

    def seal(self):
        meta = self.artifacts / 'meta'
        inventory = {'context': self.context}
        for name, value in {'context.json': self.context, 'inventory.json': inventory,
                            'plan.json': {'inventory': inventory}}.items():
            (meta / name).write_text(json.dumps(value))
        self.reseal()

    def reseal(self):
        meta = self.artifacts / 'meta'
        complete = {name: hashlib.sha256((meta / name).read_bytes()).hexdigest()
                    for name in ('context.json', 'inventory.json', 'plan.json', 'source.json')}
        (meta / 'complete.json').write_text(json.dumps(complete))

    def bootstrap(self, **overrides):
        env = dict(os.environ, TEAM_ARTIFACTS=str(self.artifacts), GITHUB_WORKSPACE=str(self.workspace),
                   TEAM_REPOSITORY_ID='123', GITHUB_RUN_ID='456', GITHUB_RUN_ATTEMPT='1',
                   TEAM_HEAD_SHA='a' * 40, TEAM_TEST_SHA=self.sha, TEAM_IMAGE=self.context['image'])
        env.update(overrides)
        wrapper = """import os, pathlib
from unittest.mock import patch
original = pathlib.Path.read_text
def read_text(path, *args, **kwargs):
    if str(path) == '/proc/self/mountinfo':
        return '1 2 3 / ' + os.environ['TEAM_ARTIFACTS'] + ' ro - ext4 /dev/disk ro\\n'
    return original(path, *args, **kwargs)
with patch('os.path.ismount', return_value=True), patch.object(pathlib.Path, 'read_text', read_text):
    exec(SCRIPT)
""".replace('SCRIPT', repr(BOOTSTRAP))
        return subprocess.run([sys.executable, '-I', '-c', wrapper], cwd=self.workspace, env=env,
                              capture_output=True, text=True, timeout=10)

    def test_real_git_export_and_bootstrap_keep_complete_tracked_source_without_credentials(self):
        original = (self.repo / 'fixtures/data.bin').read_bytes()
        (self.repo / 'fixtures/data.bin').write_text('uncommitted modification')
        (self.repo / 'ci-logs').mkdir()
        (self.repo / 'ci-logs/private.log').write_text('not a tracked fixture')
        (self.repo / '.git/config').write_text((self.repo / '.git/config').read_text() + '\n[credential]\n\thelper = do-not-export\n')
        result = self.publish()
        self.assertEqual(result['files'], 3)
        self.assertEqual((self.artifacts / 'source/fixtures/data.bin').read_bytes(), original)
        self.seal()
        process = self.bootstrap()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual((self.workspace / 'fixtures/data.bin').read_bytes(), original)
        self.assertEqual(stat.S_IMODE((self.workspace / 'scripts/check.sh').stat().st_mode), 0o755)
        self.assertFalse((self.workspace / '.git').exists())
        self.assertFalse((self.workspace / 'ci-logs').exists())
        self.assertEqual((self.workspace / 'build').resolve(), self.artifacts / 'build')
        (self.workspace / 'fixtures/data.bin').write_text('private mutation')
        self.assertEqual((self.artifacts / 'source/fixtures/data.bin').read_bytes(), original)
        self.assertEqual((self.workspace / 'fixtures/data.bin').stat().st_nlink, 1)

    def test_publisher_refuses_wrong_sha_republication_and_symlink_parent(self):
        with self.assertRaisesRegex(ValueError, 'tested SHA'):
            SOURCE.publish(self.repo, self.artifacts / 'source', self.artifacts / 'meta/source.json',
                           dict(self.context, test_sha='f' * 40))
        alias = self.root / 'alias'
        alias.symlink_to(self.artifacts, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'canonical'):
            SOURCE.publish(self.repo, alias / 'source', self.artifacts / 'meta/source.json', self.context)
        self.publish()
        with self.assertRaisesRegex(ValueError, 'replace'):
            self.publish()

    def test_size_bound_interrupts_export_without_a_selectable_publication(self):
        from unittest.mock import patch
        with patch.object(SOURCE, 'MAX_TOTAL_BYTES', 1):
            with self.assertRaisesRegex(ValueError, 'size bound'):
                self.publish()
        self.assertFalse((self.artifacts / 'source').exists())
        self.assertFalse((self.artifacts / 'meta/source.json').exists())
        self.assertEqual(list(self.artifacts.glob('.team-source-*')), [])
        self.assertFalse((self.artifacts / 'meta/complete.json').exists())

    def test_tracked_symlink_and_submodule_are_rejected(self):
        for mode in ('120000', '160000'):
            with self.subTest(mode=mode):
                oid = self.sha
                if mode == '120000':
                    (self.repo / 'unsafe').symlink_to('/etc/passwd')
                    self.git('add', 'unsafe')
                else:
                    self.git('update-index', '--add', '--cacheinfo', mode + ',' + oid + ',unsafe')
                tree = self.git('write-tree').strip()
                env = dict(os.environ, GIT_AUTHOR_NAME='Test Fixture', GIT_AUTHOR_EMAIL='fixture@example.invalid',
                           GIT_COMMITTER_NAME='Test Fixture', GIT_COMMITTER_EMAIL='fixture@example.invalid')
                sha = subprocess.check_output(['git', '-C', str(self.repo), 'commit-tree', tree, '-m', 'unsafe fixture'], env=env, text=True).strip()
                self.git('update-ref', 'HEAD', sha)
                with self.assertRaisesRegex(ValueError, 'regular tracked'):
                    SOURCE.publish(self.repo, self.artifacts / 'source', self.artifacts / 'meta/source.json',
                                   dict(self.context, test_sha=sha))
                self.git('reset', '--hard', '-q', self.sha)

    def test_bootstrap_rejects_each_unsafe_tree_without_running_exported_code(self):
        self.publish()
        self.seal()
        source = self.artifacts / 'source'
        baseline = source / 'fixtures/data.bin'
        original = baseline.read_bytes()
        marker = self.root / 'executed'
        attacks = ('modified', 'symlink', 'hardlink', 'extra', 'missing', 'directory-link', 'empty-directory', 'mode')
        for attack in attacks:
            with self.subTest(attack=attack):
                backup = self.root / 'backup'
                shutil.copytree(source, backup)
                if attack == 'modified':
                    baseline.write_bytes(b'bad prefix')
                elif attack == 'symlink':
                    baseline.unlink(); baseline.symlink_to(self.repo / 'fixtures/data.bin')
                elif attack == 'hardlink':
                    baseline.unlink(); os.link(self.repo / 'fixtures/data.bin', baseline)
                elif attack == 'extra':
                    (source / 'os.py').write_text('open(' + repr(str(marker)) + ', "w").write("bad")')
                elif attack == 'missing':
                    baseline.unlink()
                elif attack == 'directory-link':
                    shutil.rmtree(source / 'fixtures'); (source / 'fixtures').symlink_to(self.repo / 'fixtures')
                elif attack == 'empty-directory':
                    (source / 'unexpected').mkdir()
                else:
                    baseline.chmod(0o6755)
                process = self.bootstrap()
                self.assertNotEqual(process.returncode, 0, attack)
                self.assertFalse(marker.exists())
                self.assertEqual(list(self.workspace.iterdir()), [])
                shutil.rmtree(source)
                backup.rename(source)
        self.assertEqual(baseline.read_bytes(), original)

    def test_bootstrap_rejects_resealed_path_traversal_and_wrong_source_identity(self):
        self.publish()
        self.seal()
        path = self.artifacts / 'meta/source.json'
        original = json.loads(path.read_text())
        for name in ('../escape', '/absolute', 'a/../escape', 'a//b', '.git/config', 'a/.GiT/config',
                     'build/override', 'x\\y', 'x\x00y', 'x\ny'):
            with self.subTest(name=name):
                value = dict(original, files={name: next(iter(original['files'].values()))})
                path.write_text(json.dumps(value)); self.reseal()
                self.assertNotEqual(self.bootstrap().returncode, 0)
                self.assertEqual(list(self.workspace.iterdir()), [])
        value = dict(original, context=dict(self.context, run_attempt='2'))
        path.write_text(json.dumps(value)); self.reseal()
        result = self.bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Source run/attempt/SHA mismatch', result.stderr)

    def test_file_directory_conflict_is_rejected_before_copy_or_source_execution(self):
        self.publish()
        self.seal()
        path = self.artifacts / 'meta/source.json'
        value = json.loads(path.read_text())
        value['files']['fixtures'] = value['files']['fixtures/data.bin']
        path.write_text(json.dumps(value))
        self.reseal()
        result = self.bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Source inventory mismatch', result.stderr)
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_resealed_size_mismatch_is_not_accepted_as_a_matching_digest(self):
        self.publish()
        self.seal()
        path = self.artifacts / 'meta/source.json'
        value = json.loads(path.read_text())
        value['files']['fixtures/data.bin']['size'] += 1
        path.write_text(json.dumps(value))
        self.reseal()
        result = self.bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Source size mismatch', result.stderr)
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_bootstrap_requires_empty_canonical_private_destination(self):
        self.publish()
        self.seal()
        (self.workspace / '.git').mkdir()
        result = self.bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must be empty', result.stderr)
        (self.workspace / '.git').rmdir()
        actual = self.workspace
        self.workspace = self.root / 'alias-workspace'
        self.workspace.symlink_to(actual, target_is_directory=True)
        result = self.bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Invalid private workspace', result.stderr)

    def test_incomplete_publication_and_metadata_symlink_are_rejected(self):
        self.publish()
        self.seal()
        complete = self.artifacts / 'meta/complete.json'
        data = complete.read_bytes()
        complete.unlink()
        self.assertNotEqual(self.bootstrap().returncode, 0)
        other = self.root / 'complete.json'
        other.write_bytes(data)
        complete.symlink_to(other)
        self.assertNotEqual(self.bootstrap().returncode, 0)


if __name__ == '__main__':
    unittest.main()
