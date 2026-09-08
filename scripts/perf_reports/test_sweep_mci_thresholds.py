#!/usr/bin/env python3
"""Unit tests for experiment planning, validation, aggregation, and safe resumption."""

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sweep_mci_thresholds as sweep


def fixture_rows(mode="force"):
    result = []
    for i, stage in enumerate(sweep.STAGES):
        live = (100, 90, 80, 90, 100)[i]
        slots = live if mode == "force" else (100, 100, 100, 110, 120)[i]
        for ef in (40, 80):
            row = {
                "stage": stage, "ef_search": str(ef), "remove_mode": mode,
                "active_vectors": str(live), "index_elements": str(live),
                "mci_total_nodes": str(slots), "mci_covered_nodes": str(live),
                "inactive_nodes": str(slots-live), "qps": str(10000/ef),
                "recall_at_k": str(0.95 if ef == 40 else 0.99),
                "mci_route_ratio": "1", "mci_raw_float_ratio": "1",
                "avg_dist_cmp": "80", "avg_hops": "20", "avg_seed_count": "3",
                "mci_avg_clique_size": "4", "mci_total_memberships": str(live*2),
                "index_memory_bytes": "4096", "mutation_seconds": "0.2",
                "mci_delta_cliques": "3", "mci_delta_extra_memberships": "5",
            }
            row.update({column: str(sweep.BASELINE[key])
                        for key, column in sweep.PARAM_COLUMNS.items()})
            result.append(row)
    return result


class SweepTests(unittest.TestCase):
    def test_default_oat_and_grid(self):
        args = sweep.parse_args([])
        configs = sweep.configurations("oat", args.ranges)
        self.assertEqual(args.ranges["delete_size"], [3, 4, 5, 6])
        self.assertEqual(len(configs), 12)
        self.assertEqual(configs[0]["id"], "baseline")
        for config in configs[1:]:
            self.assertEqual(sum(config["parameters"][k] != v
                                 for k, v in sweep.BASELINE.items()), 1)
        self.assertEqual(len(sweep.configurations("grid", args.ranges)), 324)

    def test_validation_of_cli_ranges(self):
        for arguments in (["--join-ratios", "nan"], ["--join-ratios", "1.2"],
                          ["--added-mcts", "0"], ["--clique-maxes", "1"],
                          ["--delete-sizes", "0"], ["--timeout", "nan"],
                          ["--repeats", "0"], ["--query-count", "-1"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    sweep.parse_args(arguments)

    def test_dry_run_needs_no_binary_or_dataset(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = sweep.main(["--dry-run", "--dataset", "/missing/data",
                               "--binary", "/missing/binary"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["runs"], 36)

    def test_command_controls_incremental_not_initial_cap(self):
        args = sweep.parse_args([])
        config = {"id": "custom", "parameters": {**sweep.BASELINE, "clique_max": 100}}
        cmd = sweep.command(args, config, 2, Path("/tmp/result.csv"))
        self.assertNotIn("--mci-clique-max", cmd)
        self.assertEqual(cmd[cmd.index("--mci-incremental-clique-max")+1], "100")
        self.assertEqual(cmd[cmd.index("--random-seed")+1], "20260909")
        self.assertIn("--force-remove", cmd)
        args.remove_mode = "mark"
        self.assertNotIn("--force-remove", sweep.command(args, config, 2, Path("x")))

    def test_force_and_mark_counts(self):
        for mode in ("force", "mark"):
            rows = fixture_rows(mode)
            self.assertIs(sweep.validate_rows(rows, [40, 80], mode, sweep.BASELINE), rows)

    def test_invalid_measurements_rejected(self):
        for key, value in (("mci_total_nodes", "999"), ("mci_covered_nodes", "0"),
                           ("recall_at_k", "nan"), ("qps", "0"),
                           ("mci_raw_float_ratio", "0"), ("incremental_added_mct", "6")):
            with self.subTest(key=key):
                rows = fixture_rows()
                rows[-1][key] = value
                with self.assertRaises(ValueError):
                    sweep.validate_rows(rows, [40, 80], "force", sweep.BASELINE)
        rows = fixture_rows()
        for broken in (rows[:-1], rows+[rows[-1]], [rows[0]]*len(rows)):
            with self.assertRaises(ValueError):
                sweep.validate_rows(broken, [40, 80], "force", sweep.BASELINE)

    def test_paired_recall_deltas(self):
        a, b = fixture_rows(), fixture_rows()
        a[0]["recall_at_k"], a[-2]["recall_at_k"] = "0.95", "0.93"
        b[0]["recall_at_k"], b[-2]["recall_at_k"] = "0.90", "0.91"
        raw, summary, _ = sweep.aggregate([("x", 0, a), ("x", 1, b)], [0.98], 2)
        item = next(r for r in summary if r["stage"] == sweep.STAGES[-1]
                    and r["ef_search"] == 40)
        self.assertAlmostEqual(item["recall_delta_pp_median"], -0.5)
        self.assertEqual(len(raw), 20)
        self.assertEqual(item["memberships_per_live_node_median"], 2)

    def test_target_recall_no_extrapolation_or_missing_run_success(self):
        runs = [("x", 0, fixture_rows())]
        _, _, targets = sweep.aggregate(runs, [0.94, 0.98, 1.0], 1)
        initial = {r["target_recall"]: r for r in targets if r["stage"] == sweep.STAGES[0]}
        self.assertEqual(initial[0.94]["selected_efs"], "40")
        self.assertEqual(initial[0.98]["selected_efs"], "80")
        self.assertEqual(initial[1.0]["qps_median_if_all_reached"], "")
        _, _, incomplete = sweep.aggregate(runs, [0.94], 2)
        self.assertTrue(all(r["qps_median_if_all_reached"] == "" for r in incomplete))

    def test_trial_resume_and_timeout_preserve_attempts(self):
        args = sweep.parse_args(["--ef-search-values", "40,80"])
        config = {"id": "baseline", "parameters": dict(sweep.BASELINE)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(sweep.subprocess, "run",
                                   side_effect=subprocess.TimeoutExpired("fake", 1)):
                with self.assertRaises(RuntimeError):
                    sweep.run_trial(args, root, config, 0)
            failed = root / "baseline/repeat-1/attempt-1/status.json"
            self.assertFalse(json.loads(failed.read_text())["ok"])

            def execute(cmd, **kwargs):
                output = Path(cmd[cmd.index("--output")+1])
                sweep.write_csv(output, fixture_rows())
                return subprocess.CompletedProcess(cmd, 0)

            with mock.patch.object(sweep.subprocess, "run", side_effect=execute) as process:
                rows = sweep.run_trial(args, root, config, 0)
                self.assertEqual(len(rows), 10)
                sweep.run_trial(args, root, config, 0)
                self.assertEqual(process.call_count, 1)
            self.assertTrue(failed.exists())
            self.assertTrue((root / "baseline/repeat-1/attempt-2/curve.csv").exists())

    def test_run_limit_before_execution(self):
        with mock.patch.object(sweep.subprocess, "run") as process:
            with self.assertRaisesRegex(ValueError, "exceed"):
                sweep.main(["--design", "grid"])
            process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
