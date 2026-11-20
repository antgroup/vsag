"""Cohere preparation: normalization, truth, IDs and pre-write rejections."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import h5py
    OPTIONAL_DEPENDENCIES = True
except ImportError:
    OPTIONAL_DEPENDENCIES = False


@unittest.skipUnless(OPTIONAL_DEPENDENCIES, 'Cohere preparation requires numpy, pyarrow and h5py')
class CoherePreparationTests(unittest.TestCase):
    @staticmethod
    def vector(x, y):
        return [x, y] + [0.0] * 766

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.rows = [self.vector(2, 0), self.vector(0, 3), self.vector(-4, 0), self.vector(0, -5)]
        self.write_train(self.rows)
        pq.write_table(pa.table({'id': [11, 22, 33], 'emb': [self.vector(9, 0), self.vector(0, 7), self.vector(-3, 0)]}),
                       self.source / 'test.parquet')

    def write_train(self, values, ids=None):
        pq.write_table(pa.table({'id': ids if ids is not None else [100, 5, 700, -8], 'emb': values}),
                       self.source / 'train.parquet')

    def prepare(self, output, extra=()):
        script = Path(__file__).with_name('prepare_cohere.py')
        return subprocess.run([sys.executable, str(script), str(self.source), str(output),
            '--counts', '4', '--queries', '1', '--query-offset', '1', '--k', '2', *extra],
            capture_output=True, text=True)

    def test_normalization_truth_source_ids_and_existing_output(self):
        output = self.root / 'valid'
        result = self.prepare(output)
        self.assertEqual(result.returncode, 0, result.stderr)
        directory = output / 'scale-4'
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(manifest['query_rows'], [1, 2])
        self.assertEqual(manifest['rounding_audit']['ordered_topk_exact_cosine_matches'], 1)
        base = np.fromfile(directory / 'base.fvecs', dtype='<f4').reshape(4, 769)[:, 1:]
        np.testing.assert_array_equal(np.sum(base * base, axis=1), np.ones(4))
        self.assertEqual(np.fromfile(directory / 'groundtruth.ivecs', dtype='<i4').tolist(), [2, 1, 0])
        self.assertEqual(np.fromfile(directory / 'source-base-ids.i64', dtype='<i8').tolist(), [100, 5, 700, -8])
        self.assertEqual(np.fromfile(directory / 'source-query-ids.i64', dtype='<i8').tolist(), [22])
        self.assertNotEqual(self.prepare(output).returncode, 0)

    def test_invalid_ranges_rejected_before_write(self):
        options = [['--query-offset', '-1'], ['--query-offset', '3'], ['--counts', '5'],
                   ['--counts', '4', '4'], ['--k', '5']]
        for index, extra in enumerate(options):
            with self.subTest(extra=extra):
                output = self.root / f'range-{index}'
                self.assertNotEqual(self.prepare(output, extra).returncode, 0)
                self.assertFalse(output.exists())

    def test_invalid_vectors_rejected_before_write(self):
        for index, bad in enumerate([self.vector(0, 0), self.vector(float('nan'), 0), [1.0] * 767]):
            with self.subTest(index=index):
                self.write_train([bad, *self.rows[1:]])
                output = self.root / f'vector-{index}'
                self.assertNotEqual(self.prepare(output).returncode, 0)
                self.assertFalse(output.exists())

    def test_duplicate_source_ids_rejected_before_write(self):
        self.write_train(self.rows, [1, 1, 2, 3])
        output = self.root / 'duplicate-id'
        self.assertNotEqual(self.prepare(output).returncode, 0)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
