#!/usr/bin/env python3
"""Save and overwrite-safety checks for five-stage initial-index snapshots."""

import csv
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import sweep_mci_thresholds as common


class InitialIndexTests(unittest.TestCase):
    def test_save_and_reject_overwrite(self):
        binary = common.REPO / "build-release/tools/eval/mci_mutation_benchmark"
        if not binary.exists():
            self.skipTest("build mci_mutation_benchmark first")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(19)
            vectors = rng.normal(size=(400, 16)).astype("float32")
            queries = vectors[:4].copy()
            labels = np.arange(400, dtype="int64") % 4
            normalized = vectors / np.linalg.norm(vectors, axis=1)[:, None]
            truth = []
            for q in range(4):
                eligible = np.flatnonzero(labels == q)
                order = np.argsort(1 - normalized[eligible] @ normalized[q])[:10]
                truth.append(eligible[order])
            dataset = root / "fixture.hdf5"
            with h5py.File(dataset, "w") as data:
                data.attrs["type"], data.attrs["distance"] = "dense", "angular"
                data["train"], data["test"] = vectors, queries
                data["train_labels"], data["test_labels"] = labels, np.arange(4, dtype="int64")
                data["neighbors"] = np.array(truth, dtype="int64")
                data["distances"] = np.zeros((4, 10), dtype="float32")
                data["valid_ratios"] = np.full(4, .25, dtype="float32")
            saved = root / "initial.index"
            command = [str(binary), "--dataset", str(dataset), "--force-remove",
                       "--query-count", "4", "--search-count", "20", "--warmup-count", "1",
                       "--build-threads", "2", "--search-threads", "2",
                       "--mutation-batch-size", "40", "--ef-search-values", "40,80"]

            def run(name, *options):
                return subprocess.run(command + ["--output", str(root / f"{name}.csv"), *options],
                                      capture_output=True, text=True, timeout=90)

            built = run("built", "--save-initial-index", str(saved))
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            identity = json.loads(Path(str(saved) + ".json").read_text())
            self.assertEqual(identity["state"], "initial_100pct")
            original_hash = hashlib.sha256(saved.read_bytes()).hexdigest()
            self.assertGreater(saved.stat().st_size, 400 * 16 * 4)
            self.assertEqual(identity["base_count"], 400)
            self.assertLess(built.stdout.index("[index-save]"), built.stdout.index("[stage]"))
            with (root / "built.csv").open() as source:
                rows = list(csv.DictReader(source))
            common.validate_rows(rows, [40, 80], "force", common.BASELINE)
            overwrite = run("overwrite", "--save-initial-index", str(saved))
            self.assertNotEqual(overwrite.returncode, 0)
            self.assertIn("refusing to overwrite", overwrite.stderr)
            self.assertEqual(original_hash, hashlib.sha256(saved.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
