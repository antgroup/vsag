"""Verify normalized metric, source-ID mapping and pre-write rejections."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import h5py
except ImportError as error:
    raise unittest.SkipTest('Cohere preparation requires optional numpy, pyarrow and h5py') from error

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    source = root / 'source'
    source.mkdir()
    def vector(x, y):
        return [x, y] + [0.0] * 766
    rows = [vector(2, 0), vector(0, 3), vector(-4, 0), vector(0, -5)]
    def write_train(values, ids=None):
        pq.write_table(pa.table({'id': ids or [100, 5, 700, -8], 'emb': values}), source / 'train.parquet')
    write_train(rows)
    pq.write_table(pa.table({'id': [11, 22, 33], 'emb': [vector(9, 0), vector(0, 7), vector(-3, 0)]}),
                   source / 'test.parquet')
    prepare = Path(__file__).with_name('prepare_cohere.py')
    def run(output, extra=None):
        return subprocess.run([sys.executable, str(prepare), str(source), str(output),
                               '--counts', '4', '--queries', '1', '--query-offset', '1', '--k', '2',
                               *(extra or [])], capture_output=True, text=True)
    output = root / 'valid'
    result = run(output)
    assert result.returncode == 0, result.stderr
    directory = output / 'scale-4'
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['query_rows'] == [1, 2]
    assert manifest['rounding_audit']['ordered_topk_exact_cosine_matches'] == 1
    base = np.fromfile(directory / 'base.fvecs', dtype='<f4').reshape(4, 769)[:, 1:]
    assert np.array_equal(np.sum(base * base, axis=1), np.ones(4))
    assert np.fromfile(directory / 'groundtruth.ivecs', dtype='<i4').tolist() == [2, 1, 0]
    assert np.fromfile(directory / 'source-base-ids.i64', dtype='<i8').tolist() == [100, 5, 700, -8]
    assert np.fromfile(directory / 'source-query-ids.i64', dtype='<i8').tolist() == [22]
    assert run(output).returncode != 0
    for i, extra in enumerate([['--query-offset', '-1'], ['--query-offset', '3'], ['--counts', '5'],
                               ['--counts', '4', '4'], ['--k', '5']]):
        invalid = root / f'range-{i}'
        assert run(invalid, extra).returncode != 0 and not invalid.exists()
    for i, bad in enumerate([vector(0, 0), vector(float('nan'), 0), [1.0] * 767]):
        write_train([bad, *rows[1:]])
        invalid = root / f'vector-{i}'
        assert run(invalid).returncode != 0 and not invalid.exists()
    write_train(rows, [1, 1, 2, 3])
    invalid = root / 'duplicate-id'
    assert run(invalid).returncode != 0 and not invalid.exists()
print('Cohere normalization, exact truth/ties, source IDs, offsets and rejection checks passed')
