"""Daily budget/rotation invariants; no network or large builds."""
import copy
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/testing"))
import daily_plan as dp
import daily_parameters as parameters
import daily_report as report
import daily_run as run
from tests.scripts.test_team_test_shards import MOCK


class DailyPlanTests(unittest.TestCase):
    def context(self, date="2026-11-02"):
        return dp.context("a" * 40, "2026-11-01T18:00:00Z", date)

    def unit(self, key="core", owner="hgraph"):
        return {"key": key, "owner": owner, "mode": "core"}

    def result(self):
        plan = dp.build_plan(self.context(), "linux/gcc13/asan", [self.unit()], [])
        return dp.sealed({"schema_version": 1, "kind": "daily-result", "lane": "linux",
                          "plan": plan, "records": [{"key": "core", "status": "passed", "seconds": 20}]})

    def test_budget_is_exact_and_subbudgets_do_not_add_twice(self):
        p = dp.policy()
        self.assertEqual(sum(p["minutes"].values()), 360)
        self.assertEqual(p["minutes"]["hgraph"], 85)

    def test_timezone_week_and_replay(self):
        context = dp.context("a" * 40, "2026-11-01T18:00:00Z")
        self.assertEqual((context["logical_date"], context["slot"]), ("2026-11-02", 0))
        self.assertEqual(dp.validate_context(context), context)
        for day in range(7):
            date = (dt.date(2026, 11, 2) + dt.timedelta(days=day)).isoformat()
            self.assertEqual(self.context(date)["slot"], day)

    def test_invalid_sha_and_naive_time_fail(self):
        for sha, created in [("main", "2026-11-02T00:00:00Z"), ("a" * 40, "2026-11-02T00:00:00")]:
            with self.assertRaises(dp.PlanError):
                dp.context(sha, created)

    def test_week_is_disjoint_complete_and_deterministic(self):
        units = [{**self.unit(str(i)), "estimate_seconds": i + 1} for i in range(99)]
        week = dp.partition(units)
        self.assertEqual(week, dp.partition(list(reversed(units))))
        selected = [u["key"] for day in week for u in day]
        self.assertEqual(set(selected), {str(i) for i in range(99)})
        self.assertEqual(len(selected), len(set(selected)))
        loads = [sum(u["estimate_seconds"] for u in day) for day in week]
        self.assertLessEqual(max(loads) - min(loads), 99)

    def test_missing_history_is_unknown_not_zero(self):
        plan = dp.build_plan(self.context(), "new-image", [self.unit()], [])
        self.assertIsNone(plan["core"][0]["estimate_seconds"])
        self.assertEqual(plan["unknown_timings"], 1)

    def test_history_ignores_timeout_and_other_environment(self):
        history = {"samples": [{"key": "core", "environment": "a", "seconds": 100, "status": "timeout"},
                                {"key": "core", "environment": "b", "seconds": 1, "status": "passed"}]}
        self.assertIsNone(dp.estimate(self.unit(), "a", history))

    def test_modified_context_and_duplicate_units_fail(self):
        context = self.context()
        context["slot"] = 4
        with self.assertRaises(dp.PlanError):
            dp.validate_context(context)
        with self.assertRaises(dp.PlanError):
            dp.build_plan(self.context(), "linux", [self.unit()], [self.unit()])

    def test_unknown_owner_and_ambiguous_owner_fail(self):
        for tags in ("[unknown]", "[hgraph][sindi]"):
            with self.assertRaises(dp.PlanError):
                dp.owner({"name": "case", "tags": tags}, "functional")
        self.assertEqual(dp.owner({"name": "case", "tags": "[concurrent][hgraph]"}, "functional"), "concurrent")

    def test_nested_finite_generator_arguments(self):
        source = 'TEST_CASE("name", "[pyramid]") {\n auto v = DAILY_GENERATE(std::make_pair(1, 2), std::make_pair(3, 4));\n SECTION("path") { auto b = DAILY_GENERATE(false, true); }\n}'
        axes, configs = parameters.configurations(source, 1)
        self.assertEqual([a["choices"] for a in axes], [2, 2])
        self.assertEqual(len(configs), 4)
        self.assertEqual(configs[0], {"2": 0, "3": 0})

    def test_raw_strings_cannot_hide_later_generator_axes(self):
        source = 'TEST_CASE("x", "[sindi]") { auto json = R"tag({"x": "}", "fake":"DAILY_GENERATE(1,2)"})tag";\n auto value = DAILY_GENERATE(false, true); }'
        self.assertEqual(len(parameters.configurations(source, 1)[1]), 2)

    def test_comments_strings_and_unknown_generators(self):
        source = 'TEST_CASE("brace {", "[sindi]") {\n // DAILY_GENERATE(1,2)\n const char* s = "DAILY_GENERATE(1,2)";\n auto x = DAILY_GENERATE(false, true);\n}'
        self.assertEqual(len(parameters.configurations(source, 1)[1]), 2)
        for text in ('GENERATE(1, 2)', 'DAILY_GENERATE(range(0, 9))'):
            with self.assertRaises(dp.PlanError):
                parameters.configurations('TEST_CASE("x","") {' + text + ';}', 1)

    def test_result_reconciliation_rejects_missing_duplicate_and_mixed_context(self):
        valid = self.result()
        self.assertEqual(report.summarize([valid])["status"], "passed")
        for records in ([], valid["records"] * 2, [{"key": "wrong", "seconds": 1, "status": "passed"}]):
            invalid = dp.sealed({**{k: v for k, v in valid.items() if k != "digest"}, "records": records})
            with self.assertRaises(dp.PlanError):
                report.summarize([invalid])
        with self.assertRaises(dp.PlanError):
            report.summarize([valid, valid])

    def test_overbudget_warns_but_failure_or_skip_never_passes(self):
        base = self.result()
        for status in ("passed", "failed", "timeout", "skipped"):
            data = {k: v for k, v in copy.deepcopy(base).items() if k != "digest"}
            data["records"][0].update(status=status, seconds=6000)
            summary = report.summarize([dp.sealed(data)])
            self.assertEqual(summary["status"] == "passed", status == "passed")
            self.assertIn("over budget", report.markdown(summary))

    def test_forged_missing_or_negative_times_fail(self):
        base = self.result()
        base["records"][0]["seconds"] = -1
        with self.assertRaises(dp.PlanError):
            report.summarize([base])
        data = {k: v for k, v in base.items() if k != "digest"}
        with self.assertRaises(dp.PlanError):
            report.summarize([dp.sealed(data)])

    def test_configuration_events_include_catch_captured_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            log, xml = Path(directory) / "log", Path(directory) / "report.xml"
            log.write_text("startup\n")
            xml.write_text('<Catch2TestRun><TestCase><OverallResult><StdErr>DAILY_CHOICE 10 1</StdErr></OverallResult></TestCase></Catch2TestRun>')
            self.assertIn("DAILY_CHOICE 10 1", run.execution_events(log, xml))

    def test_real_child_exit_and_exact_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "mock.py"
            binary.write_text("#!" + sys.executable + "\n" + MOCK)
            binary.chmod(0o755)
            test = {"name": "case", "tags": "[hgraph]", "source_file": "test.cpp", "source_line": 1,
                    "binary": "mock.py", "id": "test"}
            binary.with_suffix(".json").write_text(json.dumps({"cases": [test]}))
            unit = {**self.unit(), "binary": str(binary), "tests": [test], "seed": 42, "env": {}}
            result = run.run_unit(unit, 0, root, 5, threading.Event())
            self.assertEqual(result["status"], "passed", result)

    def test_timeout_keeps_lower_bound_and_does_not_require_complete_xml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "mock.py"
            binary.write_text("#!" + sys.executable + "\n" + MOCK)
            binary.chmod(0o755)
            test = {"name": "case", "tags": "[hgraph]", "source_file": "test.cpp",
                    "source_line": 1, "binary": "mock.py", "id": "test", "sleep": 5}
            binary.with_suffix(".json").write_text(json.dumps({"cases": [test]}))
            unit = {**self.unit(), "binary": str(binary), "tests": [test], "seed": 42,
                    "env": {"VSAG_DAILY_HGRAPH": "{}"}}
            result = run.run_unit(unit, 0, root, 0.1, threading.Event())
            self.assertEqual(result["status"], "timeout", result)
            self.assertGreaterEqual(result["seconds"], 0.1)

    def test_lint_result_controls_summary_without_charging_test_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dp.write(root / "result.json", self.result())
            for status in ("success", "failure", "cancelled", "skipped"):
                with self.subTest(status=status):
                    env = {**os.environ, "NEEDS_JSON": json.dumps({
                        "full-lint-clang-tidy-check": {"result": status}})}
                    process = subprocess.run(
                        [sys.executable, str(ROOT / "scripts/testing/daily_report.py"),
                         str(root / "result.json"), "--expected-lanes", "linux",
                         "--summary", str(root / "summary.md"),
                         "--json", str(root / "summary.json")], env=env, capture_output=True)
                    actual = dp.load(root / "summary.json")
                    self.assertEqual(process.returncode, 0 if status == "success" else 1,
                                     process.stderr)
                    self.assertEqual(actual["total_seconds"], 20)
                    self.assertEqual(actual["status"], "passed" if status == "success" else "failed")
                    if status != "success":
                        self.assertIn("Workflow dependency full-lint-clang-tidy-check: " + status,
                                      actual["failures"])
                        self.assertIn("Correctness/completeness: **failed**", (root / "summary.md").read_text())

    def test_missing_lane_writes_failure_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dp.write(root / "expected.json", ["missing"])
            command = [sys.executable, str(ROOT / "scripts/testing/daily_report.py"),
                       "--expected-file", str(root / "expected.json"),
                       "--summary", str(root / "summary.md"), "--json", str(root / "summary.json")]
            process = subprocess.run(command, capture_output=True)
            self.assertEqual(process.returncode, 1)
            self.assertTrue(dp.load(root / "summary.json")["incomplete"])
            self.assertIn("incomplete evidence", (root / "summary.md").read_text())

    def test_capability_exclusions_are_exact_and_fail_closed(self):
        gpu = {"name": "Kmeans bounds the resident pass by a named budget only",
               "source_file": "/checkout/src/impl/cluster/kmeans_cluster_test.cpp"}
        allocator = {"name": "DefaultAllocator Reallocate Failure Keeps Tracking",
                     "source_file": "src/impl/allocator/default_allocator_test.cpp"}
        other = {**gpu, "source_file": "tests/another.cpp"}
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            binary = build / "bin/unittests"
            cases = [gpu, allocator]
            selected, evidence = run.capability_inventory(binary, cases)
            self.assertEqual(selected, cases)  # Unknown capability never removes a test.
            self.assertEqual(evidence["exclusions"], [])
            for asan, cuda, excluded in (("ON", "OFF", [gpu, allocator]),
                                         ("OFF", "OFF", [gpu]), ("ON", "ON", [allocator]),
                                         ("OFF", "ON", [])):
                (build / "CMakeCache.txt").write_text(
                    f"ENABLE_ASAN:BOOL={asan}\nENABLE_CUDA:BOOL={cuda}\n")
                selected, evidence = run.capability_inventory(binary, cases)
                self.assertEqual([e["test"] for e in evidence["exclusions"]], excluded)
                self.assertEqual(selected, [t for t in cases if t not in excluded])
                self.assertEqual(len(evidence["build_capabilities"]["cache_sha256"]), 64)
                self.assertEqual(run.capability_inventory(binary, [other])[0], [other])
            (build / "CMakeCache.txt").write_text("ENABLE_CUDA:BOOL=OFF\n")
            # Exact Catch2 filters cannot split case-insensitive name collisions.
            with self.assertRaises(dp.PlanError):
                run.capability_inventory(binary, [gpu, {**gpu, "name": gpu["name"].upper()}])

    def test_capability_exclusions_are_sealed_and_visible(self):
        evidence = {"exclusions": [{"test": {"name": "GPU case"},
                                     "reason": "CUDA disabled; not validated"}]}
        plan = dp.build_plan(self.context(), "cpu", [self.unit()], [], inventory=evidence)
        result = dp.sealed({"schema_version": 1, "kind": "daily-result", "lane": "cpu", "plan": plan,
                            "records": [{"key": "core", "status": "passed", "seconds": 1}]})
        summary = report.summarize([result])
        self.assertEqual(summary["status"], "passed")
        self.assertIn("GPU case: CUDA disabled; not validated", report.markdown(summary))
        plan["inventory"]["exclusions"] = []
        with self.assertRaises(dp.PlanError):
            dp.verify(plan)

    def test_mixed_pass_and_skip_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "mock.py"
            binary.write_text("#!" + sys.executable + "\n" + MOCK)
            binary.chmod(0o755)
            first = {"name": "case", "tags": "[hgraph]", "source_file": "test.cpp",
                     "source_line": 1, "binary": "mock.py", "id": "first"}
            second = {**first, "name": "CASE", "source_line": 2, "id": "second", "skip": True}
            binary.with_suffix(".json").write_text(json.dumps({"cases": [first, second]}))
            unit = {**self.unit(), "binary": str(binary), "tests": [first, second],
                    "seed": 42, "env": {}}
            result = run.run_unit(unit, 0, root, 5, threading.Event())
            self.assertEqual(result["status"], "skipped", result)
            self.assertEqual(len(result["tests"]), 2)
            self.assertEqual(sum(t["skips"] for t in result["tests"]), 1)

    def test_raw_string_other_delimiters_do_not_end_body(self):
        source = 'TEST_CASE("x", "[sindi]") { auto text = R"outer()inner"; DAILY_GENERATE(1,2,3); })outer";\n auto value = DAILY_GENERATE(false, true); }'
        self.assertEqual(len(parameters.configurations(source, 1)[1]), 2)

    def test_default_shell_paths_are_opt_in(self):
        for name in ("test_parallel_bg.sh", "test_parallel_by_name.sh"):
            text = (ROOT / "scripts/testing" / name).read_text()
            self.assertIn('if [ -n "${VSAG_DAILY_CONTEXT:-}" ]; then', text)
            self.assertIn("--allow-running-no-tests", text)  # Old non-Daily path stays intact.


if __name__ == "__main__":
    unittest.main()
