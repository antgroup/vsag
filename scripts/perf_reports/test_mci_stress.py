#!/usr/bin/env python3
"""Integration coverage for random churn and exact live-set ground truth."""

import csv
import subprocess
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

import run_mci_stress as stress


class StressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = stress.sweep.REPO / "build-release/tools/eval/mci_mutation_benchmark"
        if not cls.binary.exists():
            raise unittest.SkipTest("build mci_mutation_benchmark first")

    def execute(self, root, mode, seed=123):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(400, 16)).astype("float32")
        queries = rng.normal(size=(4, 16)).astype("float32")
        labels = np.arange(400, dtype="int64") % 4
        dataset = root / "fixture.hdf5"
        with h5py.File(dataset, "w") as data:
            data.attrs["type"] = "dense"
            data.attrs["distance"] = "angular"
            data["train"], data["test"] = train, queries
            data["train_labels"] = labels
            data["test_labels"] = np.arange(4, dtype="int64")
            data["neighbors"] = np.zeros((4, 10), dtype="int64")
            data["distances"] = np.zeros((4, 10), dtype="float32")
            data["valid_ratios"] = np.full(4, 0.25, dtype="float32")
        cmd = [str(self.binary), "--dataset", str(dataset), "--output", str(root / "curve.csv"),
               "--stress-rounds", "4", "--stress-initial-count", "200",
               "--stress-step-count", "60", "--stress-mode", mode,
               "--query-count", "4", "--search-count", "20", "--warmup-count", "1",
               "--ef-search-values", "20,40", "--build-threads", "1",
               "--search-threads", "1", "--mutation-batch-size", "60",
               "--random-seed", str(seed), "--force-remove"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        events, rows = stress.read_completed(root)
        args = SimpleNamespace(rounds=4, initial_count=200, step_count=60,
                               mode=mode, efs=[20, 40])
        stress.validate(events, rows, args)
        with (root / "curve.csv.ids.csv").open() as source:
            audit = list(csv.DictReader(source))
        with (root / "curve.csv.truth.csv").open() as source:
            truths = list(csv.DictReader(source))
        changes, true_ids = defaultdict(list), defaultdict(list)
        for item in audit:
            changes[item["stage"]].append((item["operation"], int(item["id"])))
        for item in truths:
            true_ids[item["stage"], int(item["query_id"])].append(int(item["id"]))
        active, rounds = set(), defaultdict(set)
        train64 = train.astype("float64")
        normalized = train64 / np.linalg.norm(train64, axis=1)[:, None]
        for event in events:
            stage = event["stage"]
            ids = changes[stage]
            self.assertEqual(len(ids), int(event["count"]))
            self.assertEqual(len({id_ for _, id_ in ids}), len(ids))
            for operation, id_ in ids:
                if event["round"] != "0":
                    self.assertNotIn(id_, rounds[event["round"]])
                    rounds[event["round"]].add(id_)
                if operation in ("build", "add"):
                    self.assertNotIn(id_, active)
                    active.add(id_)
                else:
                    self.assertIn(id_, active)
                    active.remove(id_)
            self.assertEqual(len(active), int(event["active_vectors"]))
            for q, query in enumerate(queries):
                eligible = sorted(id_ for id_ in active if labels[id_] == q)
                distances = 1 - normalized[eligible] @ (query/np.linalg.norm(query))
                expected = [eligible[i] for i in np.argsort(distances)[:10]]
                self.assertEqual(true_ids[stage, q], expected)
        # Truth changes are expected; no nearest-neighbor protection is applied.
        self.assertTrue(any(true_ids[events[0]["stage"], q] != true_ids[events[-1]["stage"], q]
                            for q in range(4)))
        self.assertEqual(stress.plot(root), len(events))
        self.assertGreater((root / "statistics.png").stat().st_size, 10000)
        self.assertGreater((root / "qps-recall.png").stat().st_size, 10000)
        return audit

    def test_toggle_exact_truth_counts_and_reproducible_ids(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = self.execute(Path(a), "toggle")
            second = self.execute(Path(b), "toggle")
            self.assertEqual(first, second)

    def test_alternate_exact_truth_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            self.execute(Path(directory), "alternate")

    def test_invalid_stress_mode_is_rejected(self):
        result = subprocess.run([str(self.binary), "--stress-mode", "invalid"],
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stress-mode", result.stderr)


if __name__ == "__main__":
    unittest.main()
