# Copyright 2024-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run with: python3 tools/rpt_bench/test_cli.py /path/to/rpt_bench."""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np


class BucketSizesCliTest(unittest.TestCase):
    @staticmethod
    def create_dataset(directory):
        dataset = pathlib.Path(directory) / "tiny.hdf5"
        with h5py.File(dataset, "w") as fixture:
            fixture.attrs["distance"] = "euclidean"
            fixture["train"] = np.array(
                [[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32
            )
            fixture["test"] = np.array([[0, 0]], dtype=np.float32)
            fixture["neighbors"] = np.array([[0]], dtype=np.int64)
            fixture["distances"] = np.array([[0]], dtype=np.float32)
        return dataset

    def test_invalid_bucket_sizes(self):
        cases = [
            ("100,abc", "abc"),
            ("100abc", "100abc"),
            ("0", "0"),
            ("-1", "-1"),
            ("+1", "+1"),
            ("1.5", "1.5"),
            ("18446744073709551616", "18446744073709551616"),
            ("", ""),
            ("   ", ""),
            (",100", ""),
            ("100,,200", ""),
            ("100, ,200", ""),
            ("100,", ""),
        ]
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.create_dataset(directory)
            for value, invalid_item in cases:
                with self.subTest(bucket_sizes=value):
                    result = subprocess.run(
                        [
                            str(self.binary),
                            "--dataset",
                            str(dataset),
                            "--bucket_sizes",
                            value,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertIn(
                        f"invalid --bucket_sizes value '{invalid_item}'", result.stderr
                    )
                    self.assertIn(
                        "expected an integer between 1 and 18446744073709551615",
                        result.stderr,
                    )

            with self.subTest(dataset_and_bucket_sizes_invalid=True):
                result = subprocess.run(
                    [
                        str(self.binary),
                        "--dataset",
                        str(pathlib.Path(directory) / "missing.hdf5"),
                        "--bucket_sizes",
                        "0",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("invalid --bucket_sizes value '0'", result.stderr)

    def test_valid_bucket_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.create_dataset(directory)
            report = pathlib.Path(directory) / "report.json"

            result = subprocess.run(
                [
                    str(self.binary),
                    "--dataset",
                    str(dataset),
                    "--bucket_sizes",
                    " 1, 2 , 18446744073709551615 ",
                    "--strategies",
                    "random",
                    "--output",
                    str(report),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            runs = json.loads(report.read_text())["runs"]
            self.assertEqual(
                [run["bucket_size"] for run in runs],
                [1, 2, 18446744073709551615],
            )
            self.assertTrue(all(run["uniformity"]["satisfies_bounds"] for run in runs))


if __name__ == "__main__":
    BucketSizesCliTest.binary = pathlib.Path(sys.argv.pop(1)).resolve()
    unittest.main()
