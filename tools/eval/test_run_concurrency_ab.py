import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_concurrency_ab import build_case_yaml, normalize_spec, run_suite


class ConcurrencyAbHarnessTest(unittest.TestCase):
    def make_spec(self, root: Path):
        dataset = root / "queries.hdf5"
        shared_index = root / "shared.index"
        for path in (dataset, shared_index):
            path.write_bytes(b"fixture")
        baseline_eval = self.make_fake_evaluator(root, "baseline")
        candidate_eval = self.make_fake_evaluator(root, "candidate")
        return {
            "datapath": str(dataset),
            "index_path": str(shared_index),
            "index_name": "hgraph",
            "create_params": {"dim": 4, "dtype": "float32", "metric_type": "l2"},
            "search_params": {"hgraph": {"ef_search": 8}},
            "topk": 2,
            "search_query_count": 4,
            "concurrency": [1, 2, 4],
            "rounds": 1,
            "baseline": {"eval_binary": str(baseline_eval)},
            "candidate": {"eval_binary": str(candidate_eval)},
        }

    def make_fake_evaluator(
        self,
        root: Path,
        marker: str,
        fail_at: int = 0,
        qps_base=None,
        omit_metric: str = "",
        qps_thread_factor: int = 1,
    ) -> Path:
        evaluator = root / f"{marker}_eval.py"
        template = """#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

config = Path(sys.argv[1]).read_text(encoding='utf-8')
threads = int(re.search(r'num_threads_searching: (\\d+)', config).group(1))
index_path = re.search(r\"index_path: '([^']+)'\", config).group(1)
result_uri = re.search(r\"to: 'file://([^']+)'\", config).group(1)
result_path = Path('/' + result_uri)
print('INFO __MARKER__ evaluator log before JSON export')
metrics = {
        'qps': __QPS__ + __QPS_THREAD_FACTOR__ * threads,
        'latency_detail(ms)': {'p50': 1.0 + threads, 'p99': 2.0 + threads},
        'recall_avg': 0.9,
        'measurement_sample_count': 4,
        'measurement_successful_query_count': 4,
        'evaluator': '__MARKER__',
        'index_path': index_path,
}
if '__OMIT_METRIC__':
    metrics.pop('__OMIT_METRIC__', None)
result_path.write_text(json.dumps({'concurrency_run': metrics}), encoding='utf-8')
print('INFO __MARKER__ evaluator log after JSON export')
if threads == __FAIL_AT__:
    print('synthetic failure', file=sys.stderr)
    sys.exit(7)
        """
        script = (
            template.replace("__MARKER__", marker)
            .replace(
                "__QPS__",
                str(
                    (1000 if marker == "baseline" else 2000)
                    if qps_base is None
                    else qps_base
                ),
            )
            .replace("__FAIL_AT__", str(fail_at))
            .replace("__OMIT_METRIC__", omit_metric)
            .replace("__QPS_THREAD_FACTOR__", str(qps_thread_factor))
        )
        evaluator.write_text(script, encoding="utf-8")
        evaluator.chmod(evaluator.stat().st_mode | stat.S_IXUSR)
        return evaluator

    def test_build_case_yaml_uses_file_export_and_shared_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = self.make_spec(Path(temp_dir))
            normalized = normalize_spec(spec)
            yaml = build_case_yaml(normalized, "candidate", 8, Path("/tmp/result.json"))
            self.assertIn("num_threads_searching: 8", yaml)
            self.assertIn("type: \"search\"", yaml)
            self.assertIn("format: \"json\"", yaml)
            self.assertIn("to: 'file://tmp/result.json'", yaml)
            self.assertIn("index_path: '" + spec["index_path"] + "'", yaml)
            self.assertIn("set_immutable: false", yaml)
            self.assertIn("search_query_count: 4", yaml)

    def test_set_immutable_is_shared_in_yaml_and_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self.make_spec(root)
            spec["set_immutable"] = True
            normalized = normalize_spec(spec)
            self.assertTrue(normalized["set_immutable"])
            yaml = build_case_yaml(normalized, "baseline", 1, root / "result.json")
            self.assertIn("set_immutable: true", yaml)

            report = run_suite(spec)
            self.assertTrue(report["fixed_input"]["set_immutable"])

    def test_acceptance_pairs_changes_and_medians_without_changing_harness_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = self.make_spec(Path(temp_dir))
            spec["acceptance"] = {"target_concurrency": [1, 2, 4]}
            report = run_suite(spec)

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["errors"], 0)
            self.assertEqual(report["acceptance"]["status"], "pass")
            self.assertEqual(report["acceptance"]["thresholds"]["qps_min_pct"], 15.0)
            self.assertEqual(
                report["acceptance"]["thresholds"]["p99_max_regression_pct"], 10.0
            )
            self.assertEqual(
                report["acceptance"]["thresholds"]["recall_max_abs_change"], 0.01
            )
            self.assertEqual(
                report["acceptance"]["target_concurrency"], [1, 2, 4]
            )
            self.assertEqual(len(report["acceptance"]["pairs"]), 3)
            self.assertEqual({run["pair_number"] for run in report["runs"]}, {1, 2, 3})
            for pair in report["acceptance"]["pairs"]:
                self.assertTrue(pair["valid"])
                self.assertIsNotNone(pair["qps_change_pct"])
                self.assertEqual(pair["p99_change_pct"], 0.0)
                self.assertEqual(pair["recall_abs_change"], 0.0)
            levels = {
                entry["outer_concurrency"]: entry
                for entry in report["acceptance"]["by_concurrency"]
            }
            self.assertEqual({entry["status"] for entry in levels.values()}, {"pass"})
            self.assertAlmostEqual(levels[1]["qps_change_pct"], (2001 / 1001 - 1) * 100)
            self.assertGreater(levels[4]["qps_change_pct"], 15.0)

    def test_acceptance_failure_is_separate_from_harness_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = self.make_spec(Path(temp_dir))
            spec["acceptance"] = {
                "qps_min_pct": 150,
                "target_concurrency": [1, 2, 4],
            }
            report = run_suite(spec)

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["errors"], 0)
            self.assertEqual(report["acceptance"]["status"], "fail")
            self.assertTrue(
                any("qps_change_pct median" in reason for reason in report["acceptance"]["reasons"])
            )

    def test_acceptance_reports_failed_missing_and_zero_baseline_reasons(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self.make_spec(root)
            spec["acceptance"] = {"target_concurrency": [1, 2, 4]}
            spec["candidate"]["eval_binary"] = str(
                self.make_fake_evaluator(root, "candidate_missing", omit_metric="qps")
            )
            report = run_suite(spec)

            self.assertEqual(report["status"], "failed")
            self.assertGreater(report["errors"], 0)
            self.assertEqual(report["acceptance"]["status"], "fail")
            self.assertTrue(
                any("candidate run" in reason and "failed" in reason
                    for reason in report["acceptance"]["reasons"])
            )

            spec = self.make_spec(root)
            spec["acceptance"] = {"target_concurrency": [1, 2, 4]}
            spec["baseline"]["eval_binary"] = str(
                self.make_fake_evaluator(
                    root, "baseline_zero", qps_base=0, qps_thread_factor=0
                )
            )
            report = run_suite(spec)

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["errors"], 0)
            self.assertEqual(report["acceptance"]["status"], "fail")
            self.assertTrue(
                any("baseline qps is zero" in reason for reason in report["acceptance"]["reasons"])
            )

    def test_default_concurrency_matrix_matches_acceptance_levels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = self.make_spec(Path(temp_dir))
            del spec["concurrency"]
            self.assertEqual(normalize_spec(spec)["concurrency"], [1, 2, 4, 8, 16, 32])

    def test_uses_two_evaluators_shared_index_and_ignores_stdout_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self.make_spec(root)
            spec["hash_files"] = True
            output = root / "report.json"
            report = run_suite(spec, output)

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["errors"], 0)
            self.assertEqual(
                report["variants"]["baseline"]["index_path"],
                report["variants"]["candidate"]["index_path"],
            )
            self.assertNotEqual(
                report["variants"]["baseline"]["eval_binary"],
                report["variants"]["candidate"]["eval_binary"],
            )
            self.assertEqual(
                {run["raw_result"]["concurrency_run"]["evaluator"] for run in report["runs"]},
                {"baseline", "candidate"},
            )
            self.assertEqual(
                {run["raw_result"]["concurrency_run"]["index_path"] for run in report["runs"]},
                {spec["index_path"]},
            )
            self.assertTrue(
                all("INFO " in run.get("stdout", "") for run in report["runs"])
            )
            self.assertEqual(
                set(report["fixed_input"]["sha256"]["eval_binaries"]),
                {"baseline", "candidate"},
            )
            self.assertNotEqual(
                report["fixed_input"]["sha256"]["eval_binaries"]["baseline"],
                report["fixed_input"]["sha256"]["eval_binaries"]["candidate"],
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["errors"], 0)

    def test_different_index_hashes_are_exposed_and_warned(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self.make_spec(root)
            candidate_index = root / "candidate.index"
            candidate_index.write_bytes(b"different index")
            spec["candidate"]["index_path"] = str(candidate_index)
            spec["hash_files"] = True

            report = run_suite(spec)

            self.assertNotEqual(report["variants"]["baseline"]["index_path"], report["variants"]["candidate"]["index_path"])
            self.assertNotEqual(
                report["fixed_input"]["sha256"]["indexes"]["baseline"],
                report["fixed_input"]["sha256"]["indexes"]["candidate"],
            )
            self.assertTrue(
                any("index SHA-256 hashes differ" in warning for warning in report["warnings"])
            )

    def test_process_failure_is_kept_in_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self.make_spec(root)
            spec["candidate"]["eval_binary"] = str(
                self.make_fake_evaluator(root, "candidate_failure", fail_at=2)
            )
            report = run_suite(spec)

            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["errors"], 1)
            failed = [run for run in report["runs"] if run["status"] == "process_error"]
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]["errors"], 1)
            self.assertIn("synthetic failure", failed[0]["stderr"])


if __name__ == "__main__":
    unittest.main()
