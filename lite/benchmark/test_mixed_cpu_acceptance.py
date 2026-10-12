# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import csv
from pathlib import Path
import tempfile
import unittest

from run_mixed_cpu_acceptance import percentile, verify_run


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


if __name__ == '__main__':
    unittest.main()
