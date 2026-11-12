"""Regression coverage for CSV measurements, failed runs and historical estimates."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.scripts import test_team_test_shards as fixture
from tests.scripts.test_build_benchmark_workflow import step

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/testing/team_test_timings.py"
sys.path.insert(0, str(SCRIPT.parent))
try:
    SPEC = importlib.util.spec_from_file_location("timings", SCRIPT)
    TIMINGS = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(TIMINGS)
finally:
    sys.path.pop(0)
SHARDS = fixture.SHARDS


def reseal(value):
    return SHARDS.sealed({k: v for k, v in value.items() if k != "digest"})


def data():
    inventory = fixture.fixture_example_inventory(["100_example", "101_example"])
    inventory["context"] = dict(repository_id="123", run_id="456", run_attempt="1",
                                head_sha="a" * 40, test_sha="b" * 40, image="image@sha256:" + "c" * 64)
    inventory = reseal(inventory)
    plan = SHARDS.make_plan(inventory, 2)
    tests = {test["id"]: test for test in inventory["tests"]}
    results = []
    for assignment in plan["shards"]:
        batches = []
        for index, test_id in enumerate(assignment["test_ids"]):
            case = tests[test_id]
            batches.append(dict(index=index, binary=case["binary"], test_ids=[test_id], status="passed",
                                exit_code=0, seconds=2.25, tests=[dict(case, success=True, skips=0, seconds=.25)]))
        examples = [dict(binary=binary, **inventory["examples"][binary], status="passed",
                         exit_code=0, seconds=3.5) for binary in assignment["example_binaries"]]
        results.append(SHARDS.sealed(dict(schema_version=SHARDS.SCHEMA, kind="result",
            plan_digest=plan["digest"], context=inventory["context"], shard=assignment["index"],
            test_ids=assignment["test_ids"], seed=inventory["seed"], workers=4, batch_size=32,
            batch_target_seconds=25, finished_at=1794484800, batches=batches, examples=examples,
            example_binaries=assignment["example_binaries"], success=True)))
    return plan, results


def rows(plan=None, results=None):
    if plan is None:
        plan, results = data()
    return TIMINGS.observations(plan, results, "linux-amd64", "asan", "2", "2")


class TimingsTest(unittest.TestCase):
    def test_complete_inventory_and_measured_duration_not_batch_wall(self):
        plan, results = data()
        self.assertTrue(SHARDS.merge_results(plan, results)["success"])
        output = rows(plan, results)
        self.assertEqual(len(output), 6)
        self.assertEqual(len({r["record_id"] for r in output}), 6)
        self.assertEqual(set(output[0]), set(TIMINGS.FIELDS))
        for row in output:
            self.assertEqual(row["status"], "passed")
            self.assertEqual(row["commit_sha"], "b" * 40)
            if row["kind"] == "case":
                self.assertEqual(row["duration_seconds"], .25)
                self.assertEqual(row["batch_seconds"], 2.25)
            else:
                self.assertEqual(row["duration_seconds"], 3.5)
        self.assertEqual(len(TIMINGS.estimates(output)), 6)

    def test_hint_survives_line_change_but_strict_execution_id_does_not(self):
        case = fixture.fixture_case('名称, "quoted"')
        moved = dict(case, source_line=500)
        self.assertEqual(TIMINGS.hint_id("case", "a", case), TIMINGS.hint_id("case", "a", moved))
        self.assertNotEqual(SHARDS.stable_id("a", case), SHARDS.stable_id("a", moved))
        for changed in (dict(case, name="other"), dict(case, tags="[other]"),
                        dict(case, source_file="other.cpp")):
            self.assertNotEqual(TIMINGS.hint_id("case", "a", case), TIMINGS.hint_id("case", "a", changed))

    def test_missing_shard_is_explicit_and_never_zero_duration(self):
        plan, results = data()
        output = rows(plan, results[:1])
        missing = [r for r in output if r["shard"] == 1]
        self.assertTrue(missing)
        self.assertTrue(all(r["status"] == "missing" and r["duration_seconds"] == "" for r in missing))
        self.assertEqual(len(TIMINGS.estimates(output)), len(output) - len(missing))

    def test_failed_skipped_timeout_and_missing_xml_not_estimated(self):
        for state in ("failed", "skipped", "timeout", "cancelled", "launch_error", "unavailable"):
            with self.subTest(state=state):
                plan, results = data()
                batch = results[0]["batches"][0]
                if state == "failed":
                    batch["tests"][0]["success"] = False
                    batch["status"] = "failed"
                elif state == "skipped":
                    batch["tests"][0]["skips"] = 1
                else:
                    batch["tests"] = []
                    batch["status"] = "passed" if state == "unavailable" else state
                results[0] = reseal(results[0])
                output = rows(plan, results)
                affected = next(r for r in output if r["execution_id"] == batch["test_ids"][0])
                self.assertEqual(affected["status"], state)
                if not batch["tests"]:
                    self.assertEqual(affected["duration_seconds"], "")
                self.assertNotIn(affected["case_id"], {r["case_id"] for r in TIMINGS.estimates(output)})

    def test_passed_batch_with_nonzero_exit_cannot_supply_samples(self):
        plan, results = data()
        results[0]["batches"][0]["exit_code"] = 1
        results[0] = reseal(results[0])
        with self.assertRaisesRegex(ValueError, "exit code"):
            rows(plan, results)

    def test_passed_case_in_failed_batch_is_not_a_normal_sample(self):
        plan, results = data()
        results[0]["batches"][0]["status"] = "failed"
        results[0] = reseal(results[0])
        output = rows(plan, results)
        self.assertEqual(len(TIMINGS.estimates(output)), 5)

    def test_failed_example_duration_retained_but_not_estimated(self):
        plan, results = data()
        example = results[0]["examples"][0]
        example.update(status="timeout", exit_code=-15)
        results[0] = reseal(results[0])
        output = rows(plan, results)
        failed = next(r for r in output if r["execution_id"] == example["id"])
        self.assertEqual(failed["duration_seconds"], 3.5)
        self.assertEqual(len(TIMINGS.estimates(output)), 5)

    def test_unstarted_examples_have_no_process_wall_measurement(self):
        for status, seconds in (("cancelled", 0.0), ("launch_error", 0.0),
                                ("launch_error", .012), ("failed", 0.0)):
            with self.subTest(status=status, seconds=seconds):
                plan, results = data()
                example = results[0]["examples"][0]
                example.update(status=status, exit_code=None, seconds=seconds)
                results[0] = reseal(results[0])
                output = rows(plan, results)
                observed = next(r for r in output if r["execution_id"] == example["id"])
                self.assertEqual(observed["status"], status)
                self.assertEqual(observed["duration_seconds"], "")
                self.assertEqual(observed["measurement"], "unavailable")
                self.assertEqual(len(TIMINGS.estimates(output)), 5)
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "timings.csv"
                    TIMINGS.write_csv(path, TIMINGS.FIELDS, output)
                    restored = next(r for r in TIMINGS.read_csv(path)
                                    if r["execution_id"] == example["id"])
                    self.assertEqual(restored["duration_seconds"], "")
                    self.assertEqual(restored["measurement"], "unavailable")

    def test_started_cancelled_examples_preserve_elapsed_duration(self):
        plan, results = data()
        example = results[0]["examples"][0]
        example.update(status="cancelled", exit_code=-15, seconds=1.25)
        results[0] = reseal(results[0])
        output = rows(plan, results)
        observed = next(r for r in output if r["execution_id"] == example["id"])
        self.assertEqual(observed["duration_seconds"], 1.25)
        self.assertEqual(observed["measurement"], "process_wall")
        self.assertEqual(len(TIMINGS.estimates(output)), 5)

    def test_started_zero_duration_example_remains_a_valid_measurement(self):
        plan, results = data()
        example = results[0]["examples"][0]
        example.update(status="passed", exit_code=0, seconds=0.0)
        results[0] = reseal(results[0])
        output = rows(plan, results)
        observed = next(r for r in output if r["execution_id"] == example["id"])
        self.assertEqual(observed["duration_seconds"], 0.0)
        self.assertEqual(observed["measurement"], "process_wall")
        estimate = next(r for r in TIMINGS.estimates(output) if r["case_id"] == observed["case_id"])
        self.assertEqual(estimate["median_seconds"], 0.0)

    def test_wrong_context_duplicates_identity_and_tampering_rejected(self):
        for mutate in (lambda r: r[0].update(context={}),
                       lambda r: r.append(r[0]),
                       lambda r: r[0]["batches"].append(r[0]["batches"][0]),
                       lambda r: r[0]["batches"][0]["tests"][0].update(source_line=99),
                       lambda r: r[0]["examples"][0].update(sha256="0" * 64)):
            with self.subTest(mutate=mutate):
                plan, results = data()
                mutate(results)
                results = [reseal(r) for r in results]
                with self.assertRaises(ValueError):
                    rows(plan, results)
        plan, results = data()
        results[0]["workers"] = 99
        with self.assertRaises(ValueError):
            rows(plan, results)

    def test_nan_negative_and_infinite_durations_rejected(self):
        for value in (float("nan"), float("inf"), -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TIMINGS.number(value)

    def test_dedup_window_median_p90_and_environment_separation(self):
        base = rows()[0]
        samples = [dict(base, record_id=str(i), run_id=str(i), duration_seconds=i,
                        observed_at=f"2026-11-12T08:00:{i:02d}Z") for i in range(1, 11)]
        result = TIMINGS.estimates(samples + [samples[0]], window=4)[0]
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["median_seconds"], 8.5)
        self.assertEqual(result["p90_seconds"], 10)
        self.assertEqual(result["min_seconds"], 7)
        for field, value in (("platform", "linux-arm64"), ("build_profile", "tsan"),
                             ("image", "another-image"), ("workers", 8), ("omp_num_threads", 1),
                             ("repository_id", "another-repo")):
            with self.subTest(field=field):
                different = dict(samples[0], record_id="new", **{field: value})
                self.assertEqual(len(TIMINGS.estimates(samples + [different])), 2)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            TIMINGS.estimates(samples + [dict(samples[0], duration_seconds=100)])

    def test_zero_measured_duration_is_not_missing(self):
        row = dict(rows()[0], duration_seconds=0)
        self.assertEqual(TIMINGS.estimates([row])[0]["median_seconds"], 0)

    def test_csv_roundtrip_unicode_quotes_newlines_and_formula_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timings.csv"
            samples = [dict(rows()[0], case_name=name, record_id=str(i)) for i, name in enumerate(
                ['名称, "quoted"\nsection', '=HYPERLINK("evil")', "  @name", "'literal", "+name", "-name"])]
            TIMINGS.write_csv(path, TIMINGS.FIELDS, samples)
            parsed = TIMINGS.read_csv(path)
            self.assertEqual([r["case_name"] for r in parsed], [r["case_name"] for r in samples])
            with path.open(newline="") as stream:
                cells = list(csv.DictReader(stream))
            self.assertTrue(cells[1]["case_name"].startswith("'="))
            self.assertEqual(TIMINGS.estimates(parsed), TIMINGS.estimates(samples))
            path.write_text("wrong,header\n1,2\n")
            with self.assertRaisesRegex(ValueError, "schema"):
                TIMINGS.read_csv(path)

    def test_export_and_aggregate_cli_without_executing_binaries(self):
        plan, results = data()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "plan.json").write_text(json.dumps(plan))
            for i, result in enumerate(results):
                folder = root / "results" / str(i)
                folder.mkdir(parents=True)
                (folder / "result.json").write_text(json.dumps(result))
            command = [sys.executable, str(SCRIPT), "export", "--plan", str(root / "plan.json"),
                       "--results-dir", str(root / "results"), "--platform", "linux-amd64",
                       "--build-profile", "asan", "--output-dir", str(root / "out")]
            proc = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            raw = root / "out/test-timings.csv"
            self.assertEqual(len(TIMINGS.read_csv(raw)), 6)
            output = root / "combined.csv"
            proc = subprocess.run([sys.executable, str(SCRIPT), "estimate", "--input", str(raw),
                                   "--input", str(raw), "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with output.open() as stream:
                self.assertTrue(all(row["samples"] == "1" for row in csv.DictReader(stream)))

    def test_workflow_exports_after_failed_verification_and_uploads_both_csvs(self):
        workflow = (ROOT / ".github/workflows/pr-ci-team.yml").read_text()
        export = step(workflow, "Export case and example timing CSVs")
        upload = step(workflow, "Upload case and example timing CSVs")
        self.assertIn("always()", export)
        self.assertIn("always()", upload)
        self.assertIn("--platform linux-amd64 --build-profile asan", export)
        self.assertIn("ci-summary/timings/*.csv", upload)
        self.assertIn("retention-days: 30", upload)
        self.assertIn("test_team_test_timings.py", workflow)
        self.assertIn("assert summary['success']", workflow)
        self.assertIn('test "$TEST_RESULT" = success', workflow)


if __name__ == "__main__":
    unittest.main()
