#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Tests for query subset preparation used by calibration benchmarks."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

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

    def make_source(self, dtype="float32", count=6, dim=4):
        self.train = np.arange(8 * dim, dtype=dtype).reshape(8, dim)
        self.queries = np.arange(count * dim, dtype=dtype).reshape(count, dim)
        self.neighbors = np.column_stack((np.arange(count) % 8, (np.arange(count) + 1) % 8)).astype("int64")
        self.distances = np.arange(count * 2, dtype="float32").reshape(count, 2) / 100
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
                self.assertEqual(data.attrs["distance"], b"euclidean")
                self.assertEqual(data.attrs["description"], "query subset fixture")
                self.assertEqual(data["test"].attrs["source"], "fixture")
        manifest = json.loads((output / "query_rows.json").read_text())
        self.assertEqual(manifest["query_rows"], rows)
        self.assertEqual((output / manifest["source"]).resolve(), (output.parent / "input.hdf5").resolve())

    def test_preserves_order_truth_attributes_source_and_relative_links(self):
        for dtype in ["float32", "int8", ">f4"]:
            with self.subTest(dtype=dtype):
                self.make_source(dtype)
                before = hashlib.sha256(self.source.read_bytes()).digest()
                output = self.root / ("split-" + dtype.replace(">", "big"))
                SPLIT.prepare_split(self.source, self.rows, output)
                self.assert_split(output, self.rows)
                self.assertEqual(hashlib.sha256(self.source.read_bytes()).digest(), before)
        moved = self.root / "relocated"
        moved.mkdir()
        shutil.move(str(self.source), moved / self.source.name)
        shutil.move(str(output), moved / output.name)
        self.assert_split(moved / output.name, self.rows)

    def test_large_subsets_preserve_order_across_copy_blocks(self):
        self.make_source(count=10000, dim=1024)
        rows = {"calibration": list(range(9999, 0, -2)), "validation": list(range(0, 10000, 2))}
        SPLIT.prepare_split(self.source, rows, self.output)
        self.assert_split(self.output, rows)

    def test_rejects_invalid_selection_without_creating_output(self):
        choices = [None, {}, {"calibration": [], "validation": [0]},
                   {"calibration": [0, 0], "validation": [1]},
                   {"calibration": [0], "validation": [0]},
                   {"calibration": [True], "validation": [1]},
                   {"calibration": [1.0], "validation": [2]},
                   {"calibration": [-1], "validation": [2]},
                   {"calibration": [6], "validation": [2]}]
        for rows in choices:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                SPLIT.prepare_split(self.source, rows, self.output)
            self.assertFalse(self.output.exists())

    def test_rejects_duplicate_query_content_across_splits(self):
        with h5py.File(self.source, "r+") as data:
            data["test"][4] = data["test"][5]
        with self.assertRaisesRegex(ValueError, "identical queries cross splits"):
            SPLIT.prepare_split(self.source, self.rows, self.output)
        self.assertFalse(self.output.exists())

    def test_duplicate_content_within_one_split_keeps_its_weight(self):
        with h5py.File(self.source, "r+") as data:
            data["test"][3] = data["test"][5]
        self.queries[3] = self.queries[5]
        SPLIT.prepare_split(self.source, self.rows, self.output)
        self.assert_split(self.output, self.rows)

    def test_signed_zero_does_not_evade_content_overlap_check(self):
        with h5py.File(self.source, "r+") as data:
            data["test"][5] = np.zeros(4, dtype="float32")
            data["test"][4] = np.full(4, -0.0, dtype="float32")
        with self.assertRaisesRegex(ValueError, "identical queries cross splits"):
            SPLIT.prepare_split(self.source, self.rows, self.output)

    def test_rejects_unsupported_or_misaligned_data(self):
        def mutate(kind):
            with h5py.File(self.source, "r+") as data:
                if kind == "sparse":
                    data.attrs["type"] = "sparse"
                elif kind == "labels":
                    data.create_dataset("test_labels", data=np.zeros(6, dtype="int64"))
                elif kind == "shape":
                    del data["distances"]
                    data.create_dataset("distances", data=np.zeros((5, 2), dtype="float32"))
                elif kind == "query_nan":
                    data["test"][5, 0] = np.nan
                elif kind == "gt_id":
                    data["neighbors"][5, 0] = 8
                elif kind == "gt_distance":
                    data["distances"][5, 0] = np.inf
                elif kind == "reference_attribute":
                    data.attrs["object"] = data["test"].ref
        for kind in ["sparse", "labels", "shape", "query_nan", "gt_id", "gt_distance", "reference_attribute"]:
            with self.subTest(kind=kind):
                self.make_source()
                mutate(kind)
                with self.assertRaises(ValueError):
                    SPLIT.prepare_split(self.source, self.rows, self.output)
                self.assertFalse(self.output.exists())

    def test_existing_destination_is_never_replaced(self):
        self.output.mkdir()
        sentinel = self.output / "user.txt"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            SPLIT.prepare_split(self.source, self.rows, self.output)
        self.assertEqual(sentinel.read_text(), "keep")
        link = self.root / "broken-link"
        link.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            SPLIT.prepare_split(self.source, self.rows, link)
        self.assertTrue(link.is_symlink())

    @unittest.skipUnless(os.environ.get("VSAG_EVAL_BINARY"), "set VSAG_EVAL_BINARY for native integration")
    def test_native_eval_reuses_the_index_with_both_subsets(self):
        squared = ((self.queries[:, None, :].astype("float64") - self.train[None, :, :]) ** 2).sum(axis=2)
        self.neighbors = np.argsort(squared, axis=1)[:, :3].astype("int64")
        self.distances = np.sqrt(np.take_along_axis(squared, self.neighbors, axis=1)).astype("float32")
        with h5py.File(self.source, "r+") as data:
            for name, values in [("neighbors", self.neighbors), ("distances", self.distances)]:
                del data[name]
                data.create_dataset(name, data=values)
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

    def test_cli_runs_from_another_directory_and_reports_errors(self):
        selection = self.root / "rows.json"
        selection.write_text(json.dumps(self.rows))
        command = [sys.executable, str(SCRIPT), str(self.source), str(selection), str(self.output)]
        done = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assert_split(self.output, self.rows)
        failed = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("output directory already exists", failed.stderr)
        self.assertNotIn("Traceback", failed.stderr)


if __name__ == "__main__":
    unittest.main()
