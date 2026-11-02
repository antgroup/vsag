#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual workflow snippet with an isolated synthetic commit DAG."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

WORKFLOW = Path(__file__).resolve().parents[2] / '.github/workflows/pr-simd-coverage.yml'

def snippet():
    text = WORKFLOW.read_text()
    section = text.split('      - name: Compare base and merge candidate\n', 1)[1]
    run = section.split('        run: |\n', 1)[1].split('      - name:', 1)[0]
    return '\n'.join(line[10:] for line in run.splitlines() if line.strip())

class MergeBaseSelection(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='vsag-ci-merge-test-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env = dict(os.environ, GIT_AUTHOR_NAME='CI Fixture',
                        GIT_AUTHOR_EMAIL='fixture@example.invalid',
                        GIT_COMMITTER_NAME='CI Fixture',
                        GIT_COMMITTER_EMAIL='fixture@example.invalid')
        self.git('init', '--quiet')
        tree = self.git('mktree', input='')
        self.base = self.git('commit-tree', tree, '-m', 'base')
        self.pr = self.git('commit-tree', tree, '-p', self.base, '-m', 'PR')
        self.new_main = self.git('commit-tree', tree, '-p', self.base, '-m', 'main advanced')
        self.merge = self.git('commit-tree', tree, '-p', self.base, '-p', self.pr, '-m', 'merge')
        scripts = self.root / 'scripts/coverage'
        scripts.mkdir(parents=True)
        (scripts / 'run_simd_coverage.sh').write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$1" "$2"\n')

    def git(self, *arguments, input=None):
        return subprocess.check_output(['git', *arguments], cwd=self.root,
                                       env=self.env, input=input, text=True).strip()

    def run_selection(self, head, pr, **extra_env):
        env = dict(self.env, HEAD_SHA=head, PR_HEAD_SHA=pr, BASE_SHA=self.new_main)
        env.update(extra_env)
        return subprocess.run(['bash', '-c', snippet()], cwd=self.root,
                              env=env, text=True, capture_output=True)

    def test_stale_event_base_uses_actual_merge_parent(self):
        result = self.run_selection(self.merge, self.pr)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [self.base, self.merge])

    def test_container_ownership_is_scoped_to_checkout(self):
        result = self.run_selection(self.merge, self.pr, GIT_TEST_ASSUME_DIFFERENT_OWNER='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [self.base, self.merge])

    def test_wrong_pr_head_is_rejected(self):
        result = self.run_selection(self.merge, self.new_main)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('expected PR head', result.stdout)

    def test_non_merge_is_rejected(self):
        result = self.run_selection(self.pr, self.pr)
        self.assertNotEqual(result.returncode, 0)

if __name__ == '__main__':
    unittest.main()
