#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Tests for query subset preparation used by calibration benchmarks."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/eval/prepare_query_split.py"
SPEC = importlib.util.spec_from_file_location("prepare_query_split", SCRIPT)
SPLIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPLIT)


class QuerySplitTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.hdf5"
        self.output = self.root / "split"
        self.rows = {"calibration": [5, 1, 3], "validation": [4, 0]}
        self.make_source()

    def make_source(self, dtype="float32", neighbors_dtype="int64"):
        self.train = np.arange(32, dtype=dtype).reshape(8, 4)
        self.queries = np.arange(24, dtype=dtype).reshape(6, 4)
        squared = ((self.queries[:, None, :].astype("float64") - self.train[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argsort(squared, axis=1)[:, :3]
        self.neighbors = nearest.astype(neighbors_dtype)
        self.distances = np.sqrt(np.take_along_axis(squared, nearest, axis=1)).astype("float32")
        with h5py.File(self.source, "w") as data:
            data.attrs["distance"] = np.bytes_("euclidean")
            data.attrs["type"] = "dense"
            data.attrs["description"] = "query subset fixture"
            data.create_dataset("train", data=self.train)
            data.create_dataset("test", data=self.queries).attrs["source"] = "fixture"
            data.create_dataset("neighbors", data=self.neighbors)
            data.create_dataset("distances", data=self.distances)

    def assert_split(self, output, rows):
        for role, indices in rows.items():
            with h5py.File(output / f"{role}.hdf5", "r") as data:
                self.assertIsInstance(data.get("train", getlink=True), h5py.ExternalLink)
                self.assertFalse(Path(data.get("train", getlink=True).filename).is_absolute())
                np.testing.assert_array_equal(data["train"][:], self.train)
                np.testing.assert_array_equal(data["test"][:], self.queries[indices])
                np.testing.assert_array_equal(data["neighbors"][:], self.neighbors[indices])
                np.testing.assert_array_equal(data["distances"][:], self.distances[indices])
                self.assertEqual(data["test"].dtype, self.queries.dtype)
                self.assertEqual(data["neighbors"].dtype, self.neighbors.dtype)
                self.assertEqual(data.attrs["distance"], b"euclidean")
                self.assertEqual(data.attrs["description"], "query subset fixture")
                self.assertEqual(data["test"].attrs["source"], "fixture")
        manifest = json.loads((output / "query_rows.json").read_text())
        self.assertEqual(manifest["query_rows"], rows)
        self.assertEqual((output / manifest["source"]).resolve(), (output.parent / "input.hdf5").resolve())

    def test_preserves_order_types_attributes_and_relative_links(self):
        # ann-benchmarks files such as sift-128-euclidean.hdf5 store neighbors as int32.
        cases = [("float32", "int64"), ("int8", "int64"), (">f4", "int64"),
                 ("float32", "int32"), ("float32", "uint64")]
        # Two float32 query rows per read, so the requested order must survive block boundaries.
        with mock.patch.object(SPLIT, "BLOCK_BYTES", 32):
            for dtype, neighbors_dtype in cases:
                with self.subTest(dtype=dtype, neighbors_dtype=neighbors_dtype):
                    self.make_source(dtype, neighbors_dtype)
                    output = self.root / f"split-{dtype}-{neighbors_dtype}".replace(">", "big")
                    SPLIT.prepare_split(self.source, self.rows, output)
                    self.assert_split(output, self.rows)
        moved = self.root / "relocated"
        moved.mkdir()
        shutil.move(str(self.source), moved / self.source.name)
        shutil.move(str(output), moved / output.name)
        self.assert_split(moved / output.name, self.rows)

    def test_numpy_integer_rows_preserve_data_order_and_json_manifest(self):
        rows = {"calibration": [np.int64(5), np.uint32(1), 3],
                "validation": [np.int32(4), np.uint64(0)]}
        SPLIT.prepare_split(self.source, rows, self.output)
        self.assert_split(self.output, self.rows)

    def test_rejects_invalid_selection_without_creating_output(self):
        choices = [None, {}, {"calibration": [], "validation": [0]},
                   {"calibration": [0, 0], "validation": [1]},
                   {"calibration": [0], "validation": [0]},
                   {"calibration": [True], "validation": [2]},
                   {"calibration": [np.bool_(True)], "validation": [2]},
                   {"calibration": [np.int64(-1)], "validation": [2]},
                   {"calibration": [np.uint64(2**64 - 1)], "validation": [2]},
                   {"calibration": [1.0], "validation": [2]},
                   {"calibration": [6], "validation": [2]}]
        for rows in choices:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                SPLIT.prepare_split(self.source, rows, self.output)
            self.assertFalse(self.output.exists())

    def test_rejects_unsupported_or_inconsistent_data_without_creating_output(self):
        def replace(name, values):
            def mutate(data):
                del data[name]
                data.create_dataset(name, data=values)
            return mutate

        def assign(name, index, value):
            def mutate(data):
                data[name][index] = value
            return mutate

        def signed_zero_across_splits(data):
            data["test"][5] = 0.0
            data["test"][4] = -0.0

        cases = [
            (lambda data: data.attrs.modify("type", "sparse"), "only dense"),
            (lambda data: data.create_dataset("test_labels", data=np.zeros(6, dtype="int64")), "no filters"),
            (replace("distances", np.zeros((5, 3), dtype="float32")), "align with query rows"),
            (replace("neighbors", self.neighbors.astype("float64")), "integer row indices"),
            (replace("neighbors", np.full((6, 3), -1, dtype="int32")), "outside train"),
            (replace("neighbors", np.full((6, 3), 8, dtype="int32")), "outside train"),
            (replace("neighbors", np.full((6, 3), 2**64 - 1, dtype="uint64")), "outside train"),
            (assign("test", (5, 0), np.nan), "non-finite query"),
            (assign("distances", (5, 0), np.inf), "non-finite ground-truth"),
            (lambda data: data.attrs.create("object", data["test"].ref), "attributes are unsupported"),
            (assign("test", 4, self.queries[5]), "identical queries cross splits"),
            (signed_zero_across_splits, "identical queries cross splits"),
        ]
        for case, (mutate, message) in enumerate(cases):
            with self.subTest(case=case, message=message):
                self.make_source()
                with h5py.File(self.source, "r+") as data:
                    mutate(data)
                with self.assertRaisesRegex(ValueError, message):
                    SPLIT.prepare_split(self.source, self.rows, self.output)
                self.assertFalse(self.output.exists())

    def test_duplicate_content_within_one_split_keeps_its_weight(self):
        with h5py.File(self.source, "r+") as data:
            data["test"][3] = data["test"][5]
        self.queries[3] = self.queries[5]
        SPLIT.prepare_split(self.source, self.rows, self.output)
        self.assert_split(self.output, self.rows)

    def test_cli_writes_once_and_never_replaces_a_destination(self):
        selection = self.root / "rows.json"
        selection.write_text(json.dumps(self.rows))
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        dangling = self.root / "broken-link"
        dangling.symlink_to(self.root / "missing")

        def run(output):
            command = [sys.executable, str(SCRIPT), str(self.source), str(selection), str(output)]
            return subprocess.run(command, cwd=self.root, capture_output=True, text=True)

        done = run(self.output)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assert_split(self.output, self.rows)
        contents = {path.name: path.read_bytes() for path in self.output.iterdir()}
        # The existing check runs before resolve(), so a dangling symlink is rejected too.
        for output in [self.output, alias / "split", Path("alias/split"), dangling]:
            with self.subTest(output=output):
                failed = run(output)
                self.assertNotEqual(failed.returncode, 0)
                self.assertIn(f"output directory already exists: {output}", failed.stderr)
                self.assertNotIn("Traceback", failed.stderr)
                self.assertEqual({path.name: path.read_bytes() for path in self.output.iterdir()}, contents)
                self.assertEqual(list(self.root.glob(".query-split-*")), [])
        self.assertTrue(dangling.is_symlink())
        self.assertFalse((self.root / "missing").exists())

    @unittest.skipUnless(os.environ.get("VSAG_EVAL_BINARY"), "set VSAG_EVAL_BINARY for native integration")
    def test_native_eval_reuses_the_index_with_both_subsets(self):
        # int32, as in ann-benchmarks files; eval_performance widens it to int64 on read.
        self.make_source(neighbors_dtype="int32")
        SPLIT.prepare_split(self.source, self.rows, self.output)
        index = self.root / "hgraph.index"
        create = {"dim": 4, "dtype": "float32", "metric_type": "l2", "index_param": {
            "base_quantization_type": "fp32", "max_degree": 16, "ef_construction": 64,
            "build_thread_count": 1, "use_reorder": False}}
        binary = str(Path(os.environ["VSAG_EVAL_BINARY"]).resolve())
        for role in ["build", *self.rows]:
            result = self.root / f"{role}.json"
            config = {"type": "build" if role == "build" else "search", "index_name": "hgraph",
                      "datapath": str(self.source if role == "build" else self.output / f"{role}.hdf5"),
                      "index_path": str(index), "create_params": json.dumps(create),
                      "search_params": json.dumps({"hgraph": {"ef_search": 64}}),
                      "search_mode": "knn", "topk": 3,
                      "search_query_count": 6 if role == "build" else len(self.rows[role]),
                      "num_threads_building": 1, "num_threads_searching": 2,
                      "disable_memory": True, "disable_qps": True,
                      "disable_latency": True, "disable_percent_latency": True}
            request = {"global": {"exporters": {"file": {"to": "file://" + str(result), "format": "json"}}},
                       role: config}
            request_path = self.root / f"{role}.yaml"
            request_path.write_text(json.dumps(request))
            done = subprocess.run([binary, str(request_path)], cwd=self.root,
                                  capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            value = json.loads(result.read_text())[role]
            if role != "build":
                self.assertEqual(value["statistics_query_count"], len(self.rows[role]))
                self.assertEqual(value["recall_avg"], 1.0)


if __name__ == "__main__":
    unittest.main()
