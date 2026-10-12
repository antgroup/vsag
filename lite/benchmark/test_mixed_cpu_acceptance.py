# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import csv
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from run_mixed_cpu_acceptance import percentile, sha, verify_run


class AcceptanceTests(unittest.TestCase):
    def test_nearest_rank_and_invalid_values(self):
        self.assertEqual(percentile(list(range(1, 101)), .99), 99)
        self.assertEqual(percentile([3, 1, 2], .5), 2)
        for values in [[], [-1], [float('nan')], [float('inf')]]:
            with self.assertRaises(ValueError):
                percentile(values, .5)

    def fixture(self, folder):
        with (folder / 'samples.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['round', 'id', 'operation', 'wall_us', 'user_us',
                             'system_us', 'voluntary', 'involuntary', 'peak_kib'])
            for operation, count in dict(build=1, add=1, remove=1, update=1,
                                         query=3, save=1, load=1).items():
                for _ in range(count):
                    writer.writerow([1, 0, operation, 2, 1, 0, 0, 0, 100])
        (folder / 'quality.csv').write_text('round,query,hits,k\n0,0,10,10\n1,0,9,10\n')
        (folder / 'states.csv').write_text(
            'round,count,rss_kib,peak_kib,snapshot_bytes\n0,100,90,100,0\n1,100,95,100,20\n')

    def test_valid_raw_and_missing_call(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.fixture(folder)
            result = verify_run(folder, 1, 1, 1)
            self.assertEqual(result['recall'], {'0': 1, '1': .9})
            self.assertEqual(result['query']['call_only_ops_per_second'], 500000)
            text = (folder / 'samples.csv').read_text().splitlines(True)
            (folder / 'samples.csv').write_text(''.join(text[:-1]))
            with self.assertRaises(AssertionError):
                verify_run(folder, 1, 1, 1)

    def test_count_drift_and_duplicate_quality_query(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.fixture(folder)
            (folder / 'states.csv').write_text(
                (folder / 'states.csv').read_text().replace('1,100,', '1,99,'))
            with self.assertRaises(AssertionError):
                verify_run(folder, 1, 1, 1)
            self.fixture(folder)
            (folder / 'quality.csv').write_text('round,query,hits,k\n0,0,10,10\n1,1,9,10\n')
            with self.assertRaises(AssertionError):
                verify_run(folder, 1, 1, 1)

    @unittest.skipUnless(os.environ.get('VSAG_MIXED_PROBE'), 'Compiled C++ probe required')
    def test_resident_and_reloaded_lifecycle_parity(self):
        binary = os.environ['VSAG_MIXED_PROBE']
        with tempfile.TemporaryDirectory(dir='/dev/shm') as directory:
            root = Path(directory)
            dataset = root / 'input'
            dataset.mkdir()
            for name, count in [('base.fvecs', 96), ('queries.fvecs', 5)]:
                with (dataset / name).open('wb') as stream:
                    for row in range(count):
                        stream.write(struct.pack('<i17f', 17,
                            *[math.sin((row + 1) * (d + 3) * .17) for d in range(17)]))
            for storage in ['fp32', 'fp16', 'rabitq8']:
                receipts = []
                for resident in [False, True]:
                    tag = storage + ('-resident' if resident else '-reload')
                    output = root / tag
                    snapshots = root / (tag + '-snapshots')
                    snapshots.mkdir()
                    command = [binary, str(dataset), str(output), storage, '3', '7', str(snapshots)]
                    if resident:
                        command.append('resident')
                    result = subprocess.run(command, capture_output=True, text=True, check=True)
                    self.assertIn('resident=' + str(int(resident)), result.stdout)
                    raw = verify_run(output, 3, 7, 5)
                    self.assertTrue(all(int(s['threads']) == 1 for s in raw['states']))
                    receipts.append([(p.name, sha(p)) for p in sorted(snapshots.iterdir())])
                self.assertEqual(receipts[0], receipts[1])

    @unittest.skipUnless(os.environ.get('VSAG_MIXED_PROBE'), 'Compiled C++ probe required')
    def test_bad_lifecycle_rejected(self):
        result = subprocess.run([os.environ['VSAG_MIXED_PROBE'], 'missing', 'missing-output',
            'fp32', '1', '1', 'missing-snapshot', 'invalid'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid lifecycle mode', result.stderr)


if __name__ == '__main__':
    unittest.main()
