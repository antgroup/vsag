# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import io
import json
import os
from pathlib import Path
import tarfile
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import measure_runtime_package

from measure_runtime_package import archive_bundle, inventory, resolved_dependencies, safe_bundle_unpack
from run_mixed_cpu_acceptance import sha


class RuntimePackageTests(unittest.TestCase):

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
        argv = ['runner', '--feature', str(feature), '--lite-install', str(build), '--full-install', str(full), '--dataset', str(dataset), '--source-archive', str(archive), '--output', str(output)]

        def digest(path):
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            if failure == 'wrong library' and path.name == 'libvsag.so':
                return '0' * 64
            if path.name == 'libvsag.so':
                return '9aeaa550703f116bef2959769d4f2f03da254b07c906ba941d8186224cc70c72'
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
                return b'fixture historical source'
            return 'fixture tool version'

        # Stop before compilation or measurement; this fixture tests ordering only.
        with mock.patch.object(measure_runtime_package, 'sha', side_effect=digest), \
             mock.patch.object(measure_runtime_package.subprocess, 'check_output', side_effect=command), \
             mock.patch.object(measure_runtime_package.subprocess, 'run') as run, \
             mock.patch.object(measure_runtime_package.shutil, 'which', return_value=None if failure == 'missing tool' else '/tool'), \
             mock.patch.object(measure_runtime_package.tempfile, 'TemporaryDirectory', side_effect=RuntimeError('measurement boundary')), \
             mock.patch.object(sys, 'argv', argv):
            try:
                measure_runtime_package.main()
            finally:
                run.assert_not_called()
        return output

    def test_preflight_failures_do_not_create_output(self):
        for failure in ['missing dataset', 'empty dataset', 'missing lite library', 'missing library', 'wrong library', 'wrong archive', 'missing headers', 'missing tool']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises((AssertionError, FileNotFoundError, ValueError, KeyError,
                                        subprocess.CalledProcessError)):
                    self.invoke_preflight(root, failure)
                self.assertFalse((root / 'new-parent').exists())

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

    def test_dependency_aliases_and_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'real.so').write_bytes(b'elf')
            (root/'alias.so').symlink_to('real.so')
            parsed = resolved_dependencies('alias.so => '+str(root/'alias.so')+' (0x123)\n')
            self.assertEqual(parsed, {'alias.so':(root/'real.so').resolve()})
            with self.assertRaises(ValueError):
                resolved_dependencies('unknown.so => not found')

    def test_deterministic_pack_and_safe_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'bundle'
            (source/'lib').mkdir(parents=True)
            (source/'lib/real.so').write_bytes(b'elf')
            (source/'lib/alias.so').symlink_to('real.so')
            archive_bundle(source,root/'first.gz')
            os.utime(source/'lib/real.so',(20,20))
            archive_bundle(source,root/'second.gz')
            self.assertEqual(sha(root/'first.gz'),sha(root/'second.gz'))
            safe_bundle_unpack(root/'first.gz',root/'relocated')
            self.assertEqual(inventory(source),inventory(root/'relocated'))

    def test_unsafe_members_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name,kind,target in [('../escape',tarfile.REGTYPE,''),
                                     ('lib/alias',tarfile.SYMTYPE,'../../escape'),
                                     ('lib/alias',tarfile.SYMTYPE,'/outside')]:
                with tarfile.open(root/'bad.gz','w:gz') as stream:
                    member=tarfile.TarInfo(name)
                    member.type=kind
                    member.linkname=target
                    stream.addfile(member,io.BytesIO(b''))
                with self.assertRaises(ValueError):
                    safe_bundle_unpack(root/'bad.gz',root/'out')


class CurrentPackageTests(unittest.TestCase):
    def test_current_package_cannot_relabel_legacy_installs(self):
        helper = RuntimePackageTests()
        original = sys.argv
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Reuse the real preflight fixture, adding only an explicit new SHA.
            old_main = measure_runtime_package.main
            def current_main():
                with mock.patch.object(sys, 'argv', [*sys.argv, '--feature-sha', 'c' * 40]):
                    old_main()
            with mock.patch.object(measure_runtime_package, 'main', side_effect=current_main):
                with self.assertRaisesRegex(ValueError, 'requires --lite-build'):
                    helper.invoke_preflight(root)
            self.assertFalse((root / 'new-parent').exists())
        self.assertEqual(sys.argv, original)

    def test_wrong_current_lite_library_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = RuntimePackageTests()
            feature, build, full, dataset, raw, receipt = helper.preflight_fixture(root)
            fake = dict(library_sha256='0' * 64)
            with mock.patch.object(measure_runtime_package, 'validate_build', return_value=fake):
                with self.assertRaises(AssertionError):
                    measure_runtime_package.validate_current_installs(feature,
                        'dbb2fe0bec685ae7a343b1d2060d0553a98e20d2', build, build, full, receipt)


if __name__ == '__main__':
    unittest.main()
