# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_pinned_full_acceptance
import build_fresh_full_reference

from verify_pinned_full_acceptance import verify
from run_pinned_full_acceptance import validate_fresh_receipt


class FullAuditTests(unittest.TestCase):

    def preflight_fixture(self, root):
        feature = root / 'feature'
        build = root / 'build'
        full = root / 'full'
        dataset = root / 'dataset'
        raw = root / 'raw'
        for path in [feature / 'include', build / 'include', full / 'include',
                     full / 'lib', build / 'lib', dataset, raw]:
            path.mkdir(parents=True, exist_ok=True)
        for path in [feature / 'LICENSE', build / 'libvsag-lite.so',
                     build / 'lib/libvsag-lite.so', full / 'lib/libvsag.so', full / 'include/vsag.h',
                     *[dataset / name for name in ['base.fvecs', 'queries.fvecs', 'groundtruth.ivecs']],
                     *[raw / name for name in ['base-100k.fbin', 'queries.fbin']]]:
            path.write_bytes(b'fixture')
        receipt = root / 'receipt.json'
        receipt.write_text(json.dumps(dict(
            full_source_head='dbb2fe0bec685ae7a343b1d2060d0553a98e20d2', fresh_core=True,
            ccache=False, prior_object_count=0, core_object_count=100, install=str(full),
            library_sha256='a' * 64, full_headers_sha256={'vsag.h': 'b' * 64},
            full_build_boundary='fixture', full_suite='not rerun')))
        return feature, build, full, dataset, raw, receipt

    def invoke_preflight(self, root, failure=None, existing=False):
        feature, build, full, dataset, raw, receipt = self.preflight_fixture(root)
        output = root / 'new-parent/output'
        if existing:
            output.mkdir(parents=True)
            (output / 'preserved.txt').write_bytes(b'original result')
        if failure == 'malformed receipt':
            receipt.write_text('{invalid')
        elif failure == 'invalid receipt':
            receipt.write_text('{}')
        elif failure == 'missing dataset':
            (dataset / 'base.fvecs').unlink()
        elif failure == 'empty dataset':
            (dataset / 'base.fvecs').write_bytes(b'')
        elif failure == 'missing raw':
            (raw / 'queries.fbin').unlink()
        elif failure == 'missing lite library':
            (build / 'libvsag-lite.so').unlink()
            (build / 'lib/libvsag-lite.so').unlink()
        elif failure == 'missing library':
            (full / 'lib/libvsag.so').unlink()
        elif failure == 'missing headers':
            (full / 'include/vsag.h').unlink()
            (full / 'include').rmdir()
        archive = root / 'source.tar.gz'
        archive.write_bytes(b'fixture archive')
        argv = ['runner', '--feature', str(feature), '--feature-build', str(build), '--source-repo', str(feature), '--full-install', str(full), '--dataset', 'Fixture=' + str(dataset), '--fresh-full-receipt', str(receipt), '--output', str(output)]

        if failure == 'duplicate dataset':
            argv += ['--dataset', 'Fixture=' + str(dataset)]
        elif failure == 'unsafe dataset label':
            argv[argv.index('--dataset') + 1] = '../escape=' + str(dataset)
        elif failure == 'legacy replay':
            offset = argv.index('--fresh-full-receipt')
            del argv[offset:offset + 2]

        def digest(path):
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            if failure == 'wrong library' and path.name == 'libvsag.so':
                return '0' * 64
            if path.name == 'libvsag.so':
                return '9aeaa550703f116bef2959769d4f2f03da254b07c906ba941d8186224cc70c72' if failure == 'legacy replay' else 'a' * 64
            if path.name == 'vsag.h':
                return 'b' * 64
            if path.name == 'libvsag-lite.so':
                return '3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330'
            if path == archive:
                return '0' * 64 if failure == 'wrong archive' else '0c6697115a21dc210ab63cc0dfd659c53614d39ffc89115292765a1565ca9b54'
            return 'c' * 64

        def command(argv, **kwargs):
            if argv[:2] == ['git', 'rev-parse']:
                return 'wrong' if failure == 'wrong feature' else 'dbb2fe0bec685ae7a343b1d2060d0553a98e20d2'
            if argv[:2] == ['git', 'status']:
                return b' M tracked' if failure == 'dirty feature' else b''
            if argv[:2] == ['git', 'show']:
                if failure == 'missing helper':
                    raise subprocess.CalledProcessError(128, argv)
                if argv[2].endswith('/provenance.json'):
                    return receipt.read_bytes()
                return b'fixture historical source'
            return 'fixture tool version'

        # Stop before compilation or measurement; this fixture tests ordering only.
        with mock.patch.object(run_pinned_full_acceptance, 'sha', side_effect=digest), \
             mock.patch.object(run_pinned_full_acceptance.subprocess, 'check_output', side_effect=command), \
             mock.patch.object(run_pinned_full_acceptance.subprocess, 'run') as run, \
             mock.patch.object(run_pinned_full_acceptance.shutil, 'which', return_value=None if failure == 'missing tool' else '/tool'), \
             mock.patch.object(run_pinned_full_acceptance.tempfile, 'TemporaryDirectory', side_effect=RuntimeError('measurement boundary')), \
             mock.patch.object(sys, 'argv', argv):
            try:
                run_pinned_full_acceptance.main()
            finally:
                run.assert_not_called()
        return output

    def test_preflight_failures_do_not_create_output(self):
        for failure in ['wrong feature', 'dirty feature', 'missing dataset', 'empty dataset', 'missing lite library', 'malformed receipt', 'invalid receipt', 'missing library', 'wrong library', 'missing headers', 'missing helper', 'missing tool', 'duplicate dataset', 'unsafe dataset label']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises((AssertionError, FileNotFoundError, ValueError, KeyError,
                                        subprocess.CalledProcessError)):
                    self.invoke_preflight(root, failure)
                self.assertFalse((root / 'new-parent').exists())

    def test_legacy_replay_preflight_reaches_measurement_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, '^measurement boundary$'):
                self.invoke_preflight(root, 'legacy replay')
            self.assertEqual(list((root / 'new-parent/output').iterdir()), [])

    def test_existing_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileExistsError):
                self.invoke_preflight(root, existing=True)
            output = root / 'new-parent/output'
            self.assertEqual([p.name for p in output.iterdir()], ['preserved.txt'])
            self.assertEqual((output / 'preserved.txt').read_bytes(), b'original result')

    def test_valid_preflight_reaches_measurement_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, '^measurement boundary$'):
                self.invoke_preflight(root)
            self.assertTrue((root / 'new-parent/output').is_dir())

    def test_fresh_receipt_guards(self):
        # A current source needs both fresh receipts; a historical library hash
        # must never be relabeled by changing only the command-line commit.
        receipt = dict(full_source_head='feature', fresh_core=True, ccache=False,
            prior_object_count=0, core_object_count=100, install='/tmp/full-reference',
            library_sha256='a' * 64, full_headers_sha256={'vsag.h': 'b' * 64})
        validate_fresh_receipt(receipt, 'feature', Path('/tmp/full-reference'))
        for field, invalid in [('full_source_head', 'historical'), ('fresh_core', False),
                ('ccache', True), ('prior_object_count', 1), ('core_object_count', 1),
                ('install', '/tmp/another-reference'), ('library_sha256', 'short'),
                ('full_headers_sha256', {})]:
            with self.subTest(field=field):
                altered = dict(receipt, **{field: invalid})
                with self.assertRaises(AssertionError):
                    validate_fresh_receipt(altered, 'feature', Path('/tmp/full-reference'))

    def test_nonhex_receipt_digest_rejected(self):
        with self.assertRaises(ValueError):
            validate_fresh_receipt(dict(full_source_head='feature', fresh_core=True,
                ccache=False, prior_object_count=0, core_object_count=100,
                install='/tmp/full-reference', library_sha256='z' * 64), 'feature', Path('/tmp/full-reference'))

    def fixture(self, directory, ef=512, digest='full'):
        root = Path(directory)
        (root / 'measurements').mkdir()
        (root / 'provenance.json').write_text(json.dumps(
            dict(full_library_sha256='full', lite_library_sha256='lite')))
        (root / 'manifest.json').write_text('{}')
        (root / 'measurements/identity.json').write_text(json.dumps(dict(
            arguments=dict(ef=ef, full_ef=512), reference_library_sha256=digest,
            lite_library_sha256='lite')))
        return root

    def test_unequal_query_budget_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory, ef=128)
            with self.assertRaises(AssertionError):
                verify(root)

    def test_wrong_reference_library_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory, digest='old-lite')
            with self.assertRaises(AssertionError):
                verify(root)

    def test_manifest_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            (root / 'manifest.json').write_text(json.dumps({'provenance.json': 'incorrect'}))
            with self.assertRaises(AssertionError):
                verify(root)


class CurrentSourceTests(unittest.TestCase):
    def test_full_source_identity_checks_clean_tracked_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/current.cpp').write_text('// source\n')
            (root / 'CMakeLists.txt').write_text('# fixture\n')
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', 'add', '.'], cwd=root, check=True)
            subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                            '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture'], cwd=root, check=True)
            head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
            identity = build_fresh_full_reference.validate_source_checkout(root, head)
            self.assertEqual(set(identity), {'src/current.cpp', 'CMakeLists.txt'})
            for invalid in ['0' * 40, 'main', head.upper()]:
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    build_fresh_full_reference.validate_source_checkout(root, invalid)
            (root / 'untracked.h').write_text('// overlay\n')
            with self.assertRaises(ValueError):
                build_fresh_full_reference.validate_source_checkout(root, head)
            (root / 'untracked.h').unlink()
            (root / 'src/current.cpp').write_text('// changed\n')
            with self.assertRaises(ValueError):
                build_fresh_full_reference.validate_source_checkout(root, head)

    def test_current_feature_cannot_use_unreceipted_full(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = FullAuditTests()
            feature, build, full, dataset, raw, receipt = helper.preflight_fixture(root)
            output = root / 'new/output'
            args = ['runner', '--feature', str(feature), '--feature-sha', 'c' * 40,
                    '--feature-build', str(build), '--source-repo', str(feature),
                    '--full-install', str(full), '--dataset', 'Fixture=' + str(dataset),
                    '--output', str(output)]
            with mock.patch.object(sys, 'argv', args), self.assertRaisesRegex(ValueError, 'fresh Full'):
                run_pinned_full_acceptance.main()
            self.assertFalse(output.parent.exists())

    def test_current_lite_receipt_is_checked_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = FullAuditTests()
            feature, build, full, dataset, raw, receipt = helper.preflight_fixture(root)
            output = root / 'new/output'
            args = ['runner', '--feature', str(feature), '--feature-sha', 'c' * 40,
                    '--feature-build', str(build), '--source-repo', str(feature),
                    '--full-install', str(full), '--dataset', 'Fixture=' + str(dataset),
                    '--fresh-full-receipt', str(receipt), '--output', str(output)]
            with mock.patch.object(sys, 'argv', args), self.assertRaisesRegex(ValueError, 'build-source receipt'):
                run_pinned_full_acceptance.main()
            self.assertFalse(output.parent.exists())


if __name__ == '__main__':
    unittest.main()
