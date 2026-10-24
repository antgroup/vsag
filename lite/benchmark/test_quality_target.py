# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_quality_target
from run_quality_target import audit_query
from verify_quality_target import quantile, verify


class QualityTargetTests(unittest.TestCase):

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
        argv = ['runner', '--feature', str(feature), '--lite-build', str(build), '--full-install', str(full), '--source-repo', str(feature), '--full-receipt', str(receipt), '--raw-gist', str(raw), '--prior-dataset', str(dataset), '--output', str(output)]

        def digest(path):
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            if failure == 'wrong library' and path.name == 'libvsag.so':
                return '0' * 64
            if path.name == 'libvsag.so':
                return 'a' * 64
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
        with mock.patch.object(run_quality_target, 'sha', side_effect=digest), \
             mock.patch.object(run_quality_target.subprocess, 'check_output', side_effect=command), \
             mock.patch.object(run_quality_target.subprocess, 'run') as run, \
             mock.patch.object(run_quality_target.shutil, 'which', return_value=None if failure == 'missing tool' else '/tool'), \
             mock.patch.object(run_quality_target.tempfile, 'TemporaryDirectory', side_effect=RuntimeError('measurement boundary')), \
             mock.patch.object(sys, 'argv', argv):
            try:
                run_quality_target.main()
            finally:
                run.assert_not_called()
        return output

    def test_preflight_failures_do_not_create_output(self):
        for failure in ['wrong feature', 'dirty feature', 'missing dataset', 'empty dataset', 'missing lite library', 'missing raw', 'malformed receipt', 'invalid receipt', 'missing library', 'wrong library', 'missing helper', 'missing tool']:
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

    def test_explicit_receipt_required_before_output_creation(self):
        script = Path(__file__).with_name('run_quality_target.py')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            for options in [[], ['--full-receipt', str(Path(directory) / 'missing.json')]]:
                with self.subTest(options=options):
                    result = subprocess.run([sys.executable, str(script), '--output', str(output), *options],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn('--full-receipt', result.stderr)
                    self.assertFalse(output.exists())

    def fixture(self, directory, corruption=None):
        stem=Path(directory)/'sample'
        Path(str(stem)+'.queries.csv').write_text('query,latency_us\n0,1\n1,2\n')
        lines=['query,rank,id,distance']
        for query in range(2):
            for rank in range(10):
                identifier=rank
                distance=float(rank).hex()
                if corruption=='duplicate' and rank==9: identifier=0
                if corruption=='nonfinite' and rank==9: distance='nan'
                if corruption=='ordering' and rank==9: distance='0x0p+0'
                lines.append(f'{query},{rank},{identifier},{distance}')
        Path(str(stem)+'.neighbors.csv').write_text('\n'.join(lines)+'\n')
        return stem,[set(range(10)),set(range(10))]

    def test_valid_raw_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            stem,truths=self.fixture(directory)
            self.assertEqual(audit_query(stem,truths,0,2)['hits'],20)

    def test_raw_corruptions_rejected(self):
        for corruption in ['duplicate','nonfinite','ordering']:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                stem,truths=self.fixture(directory,corruption)
                with self.assertRaises(AssertionError): audit_query(stem,truths,0,2)

    def test_query_interval_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            stem,truths=self.fixture(directory)
            with self.assertRaises(AssertionError): audit_query(stem,truths,1,2)

    def test_quantile_nearest_rank(self):
        self.assertEqual(quantile(list(range(1,901)),.99),891)
        self.assertEqual(quantile(list(range(1,901)),.5),450)

    def test_manifest_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'protocol.json').write_text('{}')
            (root/'manifest.json').write_text(json.dumps({'protocol.json':'invalid'}))
            with self.assertRaises(AssertionError): verify(root)

    def test_duplicate_result_query_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            stem,truths=self.fixture(directory)
            path=Path(str(stem)+'.neighbors.csv')
            path.write_text(path.read_text().replace('1,0,0,','0,0,0,'))
            with self.assertRaises(AssertionError): audit_query(stem,truths,0,2)

if __name__=='__main__': unittest.main()
