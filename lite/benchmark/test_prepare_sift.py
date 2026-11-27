# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

import prepare_sift


class SiftPreparationTests(unittest.TestCase):
    def prepare(self, root, counts=(8,), invalid=None):
        base = np.arange(16 * 128, dtype=np.float32).reshape(16, 128)
        queries = base[:2].copy()
        if invalid == 'base':
            base[0, 0] = np.nan
        elif invalid == 'query':
            queries[0, 0] = np.inf
        elif invalid == 'rank':
            base = base.reshape(-1)
        source = root / 'input.hdf5'
        with h5py.File(source, 'w') as stream:
            stream.attrs['distance'] = 'euclidean'
            stream.create_dataset('train', data=base)
            stream.create_dataset('test', data=queries)
        argv = ['prepare', str(source), str(root / 'output'), '--counts',
                *map(str, counts), '--queries', '2', '--k', '2']
        with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            prepare_sift.main()

    def test_finite_subsets_preserve_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root, counts=(8, 16))
            for count in [8, 16]:
                output = root / 'output' / ('scale-' + str(count))
                manifest = json.loads((output / 'manifest.json').read_text())
                self.assertEqual(manifest['base_prefix'], count)
                self.assertEqual(manifest['queries_prefix'], 2)
                self.assertEqual(set(manifest['files_sha256']),
                                 {'base.fvecs', 'queries.fvecs', 'groundtruth.ivecs'})

    def test_duplicate_counts_fail_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(SystemExit):
                self.prepare(root, counts=(8, 8))
            self.assertFalse((root / 'output').exists())

    def test_nonfinite_inputs_fail_before_output(self):
        for invalid in ['base', 'query']:
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(SystemExit):
                    self.prepare(root, invalid=invalid)
                self.assertFalse((root / 'output').exists())

    def test_invalid_rank_fails_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(SystemExit):
                self.prepare(root, invalid='rank')
            self.assertFalse((root / 'output').exists())
