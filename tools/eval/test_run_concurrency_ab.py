import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

import run_concurrency_ab as harness
from run_concurrency_ab import HarnessError, build_case_yaml, main, normalize_spec, run_suite


class ConcurrencyAbHarnessTest(unittest.TestCase):
    def make_spec(self, root: Path):
        dataset = root / "queries.hdf5"
        shared_index = root / "shared.index"
        baseline_library = root / "baseline_libvsag.so.0.0.0"
        candidate_library = root / "candidate_libvsag.so.0.0.0"
        for path, content in (
            (dataset, b"fixture"),
            (shared_index, b"fixture"),
            (baseline_library, b"baseline library"),
            (candidate_library, b"candidate library"),
        ):
            path.write_bytes(content)
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
            "warmup_query_count": 10000,
            "concurrency": [1, 2, 4],
            "rounds": 1,
            "baseline": {
                "eval_binary": str(baseline_eval),
                "shared_library": str(baseline_library),
            },
            "candidate": {
                "eval_binary": str(candidate_eval),
                "shared_library": str(candidate_library),
            },
        }

    def make_fake_evaluator(
        self,
        root: Path,
        marker: str,
        fail_at: int = 0,
        qps_base=None,
        omit_metric: str = "",
        qps_thread_factor: int = 1,
        sample_count: int = 4,
        successful_count: int = 4,
        error_count: int = 0,
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
result_path = Path(result_uri)
print('INFO __MARKER__ evaluator log before JSON export')
metrics = {
        'qps': __QPS__ + __QPS_THREAD_FACTOR__ * threads,
        'latency_detail(ms)': {'p50': 1.0 + threads, 'p99': 2.0 + threads},
        'recall_avg': 0.9,
        'measurement_sample_count': __SAMPLE_COUNT__,
        'measurement_successful_query_count': __SUCCESSFUL_COUNT__,
        'error_count': __ERROR_COUNT__,
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
            .replace("__SAMPLE_COUNT__", str(sample_count))
            .replace("__SUCCESSFUL_COUNT__", str(successful_count))
            .replace("__ERROR_COUNT__", str(error_count))
        )
        evaluator.write_text(script, encoding="utf-8")
        evaluator.chmod(evaluator.stat().st_mode | stat.S_IXUSR)
        return evaluator

    @staticmethod
    def fake_preflight_variant(variant, binary, shared_library):
        binary_path = Path(binary).resolve()
        library_path = Path(shared_library).resolve()
        binary_hash = harness._sha256(binary_path)
        library_hash = harness._sha256(library_path)
        return {
            "eval_binary": binary,
            "evaluator_realpath": str(binary_path),
            "evaluator_sha256": binary_hash,
            "shared_library": shared_library,
            "shared_library_realpath": str(library_path),
            "shared_library_sha256": library_hash,
            "resolved_libvsag_realpath": str(library_path),
            "resolved_libvsag_sha256": library_hash,
            "ld_library_path": str(library_path.parent),
            "environment": {
                "PATH": harness.os.environ.get("PATH", ""),
                "LD_LIBRARY_PATH": str(library_path.parent),
            },
        }

    def run_fake_suite(self, spec, output_path=None, cpu_info=None):
        with patch.object(harness, "_preflight_variant", side_effect=self.fake_preflight_variant):
            cpu_patch = patch.object(harness, "_cpu_info", return_value=cpu_info) if cpu_info else None
            if cpu_patch:
                cpu_patch.start()
            try:
                return run_suite(spec, output_path)
            finally:
                if cpu_patch:
                    cpu_patch.stop()

    def test_build_case_yaml_uses_absolute_file_uri_and_shared_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = normalize_spec(self.make_spec(Path(temp_dir)))
            yaml = build_case_yaml(spec, "candidate", 8, Path("/tmp/result.json"))
            self.assertIn("num_threads_searching: 8", yaml)
            self.assertIn('format: "json"', yaml)
            self.assertIn("to: 'file:///tmp/result.json'", yaml)
            self.assertIn("index_path: '" + spec["index_path"] + "'", yaml)
            self.assertIn("set_immutable: false", yaml)
            self.assertIn("warmup_query_count: 10000", yaml)

    def test_file_export_target_makes_relative_path_absolute(self):
        target = harness._file_export_target(Path("result.json"))
        self.assertTrue(target.startswith("file:///"))
        self.assertTrue(Path(target[len("file://") :]).is_absolute())

    def test_strict_variant_schema_requires_shared_library(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = self.make_spec(Path(temp_dir))
            del spec["candidate"]["shared_library"]
            with self.assertRaises(HarnessError):
                normalize_spec(spec)

    def test_same_evaluator_content_different_paths_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self.make_spec(root)
            candidate = Path(spec["candidate"]["eval_binary"])
            candidate.write_bytes(Path(spec["baseline"]["eval_binary"]).read_bytes())
            with patch.object(harness, "_preflight_variant", side_effect=self.fake_preflight_variant):
                with self.assertRaisesRegex(HarnessError, "evaluator SHA-256"):
                    run_suite(spec)

    def test_reports_distinct_library_identity_and_cpu_oversubscription(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = self.make_spec(Path(temp_dir))
            spec["hash_files"] = True
            report = self.run_fake_suite(
                spec,
                cpu_info={
                    "sched_affinity": [2, 4],
                    "affinity_cpu_count": 2,
                    "logical_cpu_count": 4,
                    "reason": None,
                },
            )
            self.assertEqual(
                report["variants"]["baseline"]["resolved_libvsag_realpath"],
                str(Path(spec["baseline"]["shared_library"]).resolve()),
            )
            self.assertNotEqual(
                report["variants"]["baseline"]["resolved_libvsag_sha256"],
                report["variants"]["candidate"]["resolved_libvsag_sha256"],
            )
            self.assertEqual(report["cpu"]["sched_affinity"], [2, 4])
            self.assertEqual(
                {warning["outer_concurrency"] for warning in report["warning_records"]},
                {4},
            )
            fingerprints = report["fixed_input"]["sha256"]
            self.assertEqual(set(fingerprints["eval_binaries"]), {"baseline", "candidate"})
            self.assertNotEqual(
                fingerprints["eval_binaries"]["baseline"],
                fingerprints["eval_binaries"]["candidate"],
            )

    def test_runs_use_distinct_evaluators_shared_index_and_ignore_stdout_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self.make_spec(root)
            report = self.run_fake_suite(spec)

            self.assertEqual({run["variant"] for run in report["runs"]}, {"baseline", "candidate"})
            self.assertEqual(
                {run["raw_result"]["concurrency_run"]["evaluator"] for run in report["runs"]},
                {"baseline", "candidate"},
            )
            self.assertEqual(
                {run["index_path"] for run in report["runs"]},
                {spec["index_path"]},
            )
            self.assertTrue(
                all(
                    "evaluator log before JSON export" in run["stdout"]
                    and "evaluator log after JSON export" in run["stdout"]
                    for run in report["runs"]
                )
            )
            self.assertTrue(all(run["status"] == "ok" for run in report["runs"]))

    def test_ldd_parser_requires_one_resolved_libvsag(self):
        output = "libvsag.so.0 => /opt/vsag/lib/libvsag.so.0 (0x00007f)\n"
        self.assertEqual(
            harness._parse_ldd_libvsag(output, "baseline"),
            "/opt/vsag/lib/libvsag.so.0",
        )
        with self.assertRaisesRegex(HarnessError, "not found"):
            harness._parse_ldd_libvsag("libvsag.so.0 => not found\n", "candidate")
        with self.assertRaisesRegex(HarnessError, "exactly one"):
            harness._parse_ldd_libvsag("libc.so.6 => /lib/libc.so.6 (0x00007f)\n", "candidate")

    def test_set_immutable_and_warmup_are_shared_in_yaml_and_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self.make_spec(root)
            spec["set_immutable"] = True
            normalized = normalize_spec(spec)
            yaml = build_case_yaml(normalized, "baseline", 1, root / "result.json")
            self.assertIn("set_immutable: true", yaml)
            self.assertIn("warmup_query_count: 10000", yaml)
            report = self.run_fake_suite(spec)
            self.assertEqual(report["fixed_input"]["warmup_query_count"], 10000)

    def test_acceptance_pairs_changes_and_medians(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = self.make_spec(Path(temp_dir))
            spec["acceptance"] = {
                "target_concurrency": [1, 2, 4],
                "min_valid_pairs": 1,
            }
            report = self.run_fake_suite(spec)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["errors"], 0)
            self.assertEqual(report["acceptance"]["status"], "pass")
            self.assertEqual(report["acceptance"]["thresholds"]["min_valid_pairs"], 1)
            self.assertEqual(len(report["acceptance"]["pairs"]), 3)
            for pair in report["acceptance"]["pairs"]:
                self.assertTrue(pair["valid"])
                self.assertEqual(pair["measurement"]["baseline"]["error_count"], 0)
            self.assertGreater(
                report["acceptance"]["by_concurrency"][-1]["qps_change_pct"], 15.0
            )

    def test_sample_count_mismatch_invalidates_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self.make_spec(root)
            spec["acceptance"] = {"target_concurrency": [1, 2, 4], "min_valid_pairs": 1}
            spec["candidate"]["eval_binary"] = str(
                self.make_fake_evaluator(root, "candidate_short", sample_count=3)
            )
            report = self.run_fake_suite(spec)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["acceptance"]["status"], "fail")
            self.assertTrue(
                any("measurement_sample_count" in reason for reason in report["acceptance"]["reasons"])
            )

    def test_nonzero_error_count_invalidates_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self.make_spec(root)
            spec["acceptance"] = {"target_concurrency": [1, 2, 4], "min_valid_pairs": 1}
            spec["candidate"]["eval_binary"] = str(
                self.make_fake_evaluator(root, "candidate_errors", error_count=1)
            )
            report = self.run_fake_suite(spec)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["errors"], 0)
            self.assertEqual(report["acceptance"]["status"], "fail")
            self.assertTrue(
                any("paired error_count must be zero" in reason
                    for reason in report["acceptance"]["reasons"])
            )

    def test_acceptance_reports_missing_metric_and_zero_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self.make_spec(root)
            spec["acceptance"] = {"target_concurrency": [1, 2, 4], "min_valid_pairs": 1}
            spec["candidate"]["eval_binary"] = str(
                self.make_fake_evaluator(root, "candidate_missing", omit_metric="qps")
            )
            report = self.run_fake_suite(spec)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(
                any("candidate run" in reason and "failed" in reason
                    for reason in report["acceptance"]["reasons"])
            )

            spec = self.make_spec(root)
            spec["acceptance"] = {"target_concurrency": [1, 2, 4], "min_valid_pairs": 1}
            spec["baseline"]["eval_binary"] = str(
                self.make_fake_evaluator(root, "baseline_zero", qps_base=0, qps_thread_factor=0)
            )
            report = self.run_fake_suite(spec)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["acceptance"]["status"], "fail")
            self.assertTrue(
                any("baseline qps is zero" in reason
                    for reason in report["acceptance"]["reasons"])
            )

    def test_acceptance_requires_minimum_valid_pairs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = self.make_spec(Path(temp_dir))
            spec["acceptance"] = {"target_concurrency": [1, 2, 4], "min_valid_pairs": 3}
            report = self.run_fake_suite(spec)
            self.assertEqual(report["acceptance"]["status"], "fail")
            self.assertTrue(
                any("below minimum 3" in reason for reason in report["acceptance"]["reasons"])
            )

    def test_timeout_must_be_finite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self.make_spec(root)
            for value in (
                float("nan"),
                float("inf"),
                float("-inf"),
                "NaN",
                "Infinity",
                10**10000,
            ):
                spec["timeout_seconds"] = value
                with self.assertRaises(HarnessError):
                    normalize_spec(spec)

            spec_path = root / "non_finite_timeout.json"
            spec["timeout_seconds"] = float("nan")
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaises(HarnessError):
                normalize_spec(harness._load_spec(spec_path))

    def test_acceptance_exit_code_is_distinct_from_execution_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self.make_spec(root)
            spec["acceptance"] = {
                "qps_min_pct": 150,
                "target_concurrency": [1, 2, 4],
                "min_valid_pairs": 1,
            }
            spec_path = root / "spec.json"
            output_path = root / "report.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with patch.object(harness, "_preflight_variant", side_effect=self.fake_preflight_variant):
                self.assertEqual(
                    main(["--spec", str(spec_path), "--output", str(output_path)]),
                    2,
                )
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["acceptance"]["status"], "fail")

    def test_process_failure_is_kept_in_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self.make_spec(root)
            spec["candidate"]["eval_binary"] = str(
                self.make_fake_evaluator(root, "candidate_failure", fail_at=2)
            )
            report = self.run_fake_suite(spec)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["errors"], 1)
            failed = [run for run in report["runs"] if run["status"] == "process_error"]
            self.assertEqual(len(failed), 1)
            self.assertIn("synthetic failure", failed[0]["stderr"])


if __name__ == "__main__":
    unittest.main()
