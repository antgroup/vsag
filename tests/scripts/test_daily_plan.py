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
import daily_window as window_helper
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

    def test_exact_timing_beats_slow_sibling_and_timeout_stays_a_lower_bound(self):
        original = dp.load
        history = {"samples": [
            {"key": "fast", "case": "same", "environment": "env", "status": "passed", "seconds": 10},
            {"key": "slow", "case": "same", "environment": "env", "status": "timeout", "seconds": 1800}]}
        def load(path):
            return history if Path(path).name == "daily_timing_history.json" else original(path)
        units = [{**self.unit(key), "tests": [{"name": "same"}]} for key in ("fast", "slow", "unknown")]
        with patch.object(dp, "load", side_effect=load):
            plan = dp.build_plan(self.context(), "env", units, [])
        fast, slow, unknown = plan["core"]
        self.assertEqual(fast["order_seconds"], 12)
        self.assertEqual(slow["order_seconds"], 1800)
        self.assertIsNone(slow["estimate_seconds"])
        self.assertEqual(unknown["order_seconds"], 1800)
        self.assertIsNone(unknown["estimate_seconds"])

    def test_quarantine_is_exact_rotation_only_and_visible_not_passed(self):
        rule = dp.load(dp.HERE / "daily_matrix.json")["quarantine"][0]
        bad = {**self.unit(rule["key"]), "mode": "rotation", "configuration": {"metric": "l2", "dim": 1536},
               "tests": [{"name": rule["case"]}]}
        other = {**bad, "key": "other"}
        plan = dp.build_plan(self.context(), "env", [self.unit()], [bad, other])
        self.assertEqual([u["key"] for day in plan["rotation"] for u in day], ["other"])
        self.assertEqual(plan["quarantines"][0]["status"], "not_validated")
        self.assertEqual(plan["quarantines"][0]["configuration"], bad["configuration"])
        result = dp.sealed({"schema_version": 1, "kind": "daily-result", "plan": plan, "lane": "test",
                            "records": [{"key": u["key"], "status": "passed", "seconds": 1}
                                        for u in plan["selected"]]})
        summary = report.summarize([result])
        text = report.markdown(summary)
        self.assertIn("Quarantined configurations", text)
        self.assertIn(rule["key"], text)
        self.assertIn("not validated", text)
        # Quarantine is not an execution result, does not hide unexpected failures,
        # and never disables PR core or the exhaustive coverage profile.
        self.assertEqual(summary["status"], "passed")
        result["records"][0]["status"] = "failed"
        result = dp.sealed({k: v for k, v in result.items() if k != "digest"})
        self.assertEqual(report.summarize([result])["status"], "failed")
        core_plan = dp.build_plan(self.context(), "env", [{**bad, "mode": "core"}], [])
        self.assertEqual(len(core_plan["selected"]), 1)
        full = dp.build_plan(self.context(), "env", [self.unit()], [bad], full_coverage=True)
        self.assertFalse(full["quarantines"])
        self.assertEqual(sum(map(len, full["rotation"])), 1)
        with self.assertRaisesRegex(dp.PlanError, "identity mismatch"):
            dp.quarantine_extensions([{**bad, "tests": [{"name": "unrelated"}]}], [rule])
        for invalid in ({**rule, "key": "*"}, {**rule, "case": "(PR) HGraph Duplicate"},
                        {**rule, "tracking": ""}, {**rule, "reason": ""}):
            with self.assertRaises(dp.PlanError):
                dp.quarantine_extensions([bad], [invalid])
        with self.assertRaises(dp.PlanError):
            dp.quarantine_extensions([bad], [rule, rule])

    def test_quarantine_is_not_duplicated_in_four_shard_summary(self):
        rule = dp.load(dp.HERE / "daily_matrix.json")["quarantine"][0]
        bad = {**self.unit(rule["key"]), "mode": "rotation", "tests": [{"name": rule["case"]}]}
        parent = dp.build_plan(self.context(), "env", [self.unit(str(i)) for i in range(4)], [bad])
        results = []
        for i in range(4):
            plan = dp.shard_plan(parent, i, 4)
            results.append(dp.sealed({"schema_version": 1, "kind": "daily-result", "plan": plan,
                                     "lane": f"test-shard{i}", "records": [
                                         {"key": u["key"], "status": "passed", "seconds": 1}
                                         for u in plan["selected"]]}))
        self.assertEqual(len(report.summarize(results)["quarantines"]), 1)

    def test_multiline_macro_aliases_keep_one_canonical_axis(self):
        source = 'TEST_CASE("x", "[pyramid]") {\n auto x = DAILY_GENERATE(1,\n 2,\n 3);\n}'
        axes, choices = parameters.configurations(source, 1)
        self.assertEqual((axes[0]["line"], axes[0]["end_line"]), ("2", "4"))
        self.assertEqual(len(choices), 3)
        for i, choice in enumerate(choices):
            self.assertEqual(parameters.runtime_choices(axes, choice), {"2": i, "4": i})
        with self.assertRaises(dp.PlanError):
            parameters.runtime_choices([{"line": "2", "end_line": "4"}], {"2": 0, "4": 1})
        with self.assertRaises(dp.PlanError):
            parameters.runtime_choices([{"line": "2", "end_line": "4"},
                                        {"line": "4", "end_line": "4"}], {"2": 0, "4": 0})

    def test_arm_scope_is_explicit_and_x86_remains_complete(self):
        units = [{**self.unit(str(i)), "tests": [{"source_file": "test_hgraph.cpp", "name": str(i)}]}
                 for i in range(100)]
        x86 = dp.build_plan(self.context(), "lane/Linux/x86_64", units[:2], units[2:])
        self.assertEqual(len(x86["core"]), 2)
        self.assertEqual(sum(map(len, x86["rotation"])), 98)
        arm = dp.build_plan(self.context(), "lane/Linux/aarch64", units[:2], units[2:])
        self.assertEqual(len(arm["core"]), 1)
        self.assertEqual(arm["architecture_scope"]["extension_inventory"], 99)
        self.assertEqual(sum(map(len, arm["rotation"])),
                         arm["architecture_scope"]["weekly_extension"])
        selected = {u["key"] for day in arm["rotation"] for u in day}
        self.assertLess(len(selected), 30)
        smoke = dp.build_plan(self.context(), "lane/Linux/aarch64", units[:2], units[2:],
                              core_only=True)
        self.assertEqual(sum(map(len, smoke["rotation"])), 0)
        self.assertEqual(len(smoke["core"]), 1)
        mac = dp.build_plan(self.context(), "lane/Darwin/arm64", units[:2], units[2:])
        self.assertIsNone(mac["architecture_scope"])
        coverage = dp.build_plan(self.context(), "coverage/Linux/aarch64", units[:2], units[2:],
                                 full_coverage=True)
        self.assertIsNone(coverage["architecture_scope"])
        self.assertEqual(len(coverage["core"]), 2)
        self.assertEqual(sum(map(len, coverage["rotation"])), 98)
        self.assertEqual(sum(map(len, mac["rotation"])), 98)
        seen = []
        for week in range(10):
            date = (dt.date(2026, 11, 2) + dt.timedelta(weeks=week)).isoformat()
            plan = dp.build_plan(self.context(date), "lane/Linux/aarch64", units[:2], units[2:])
            seen += [u["key"] for day in plan["rotation"] for u in day]
        self.assertEqual(len(seen), 99)
        self.assertEqual(len(set(seen)), 99)
        self.assertEqual(arm, dp.build_plan(self.context(), "lane/Linux/aarch64",
                                          list(reversed(units[:2])), units[2:]))

    def test_arm_build_preference_and_protected_core_survive_pruning(self):
        def unit(key, source, owner="hgraph", tags="", name=None):
            return {**self.unit(key, owner), "tests": [{"name": name or key,
                    "source_file": source, "tags": tags}]}
        units = [unit("a-normal", "hgraph.cpp"), unit("z-build", "hgraph.cpp", tags="[build]"),
                 unit("r", "hgraph.cpp", tags="[regression]"),
                 unit("g", "hgraph.cpp", name="gh#123"), unit("source", "clone.cpp"),
                 unit("owner", "hgraph.cpp", owner="pyramid")]
        for core_only in (False, True):
            plan = dp.build_plan(self.context(), "lane/Linux/aarch64", units, [], core_only=core_only)
            self.assertEqual({u["key"] for u in plan["core"]}, {"z-build", "r", "g", "source", "owner"})
            if core_only:
                self.assertEqual(sum(map(len, plan["rotation"])), 0)

    def test_shards_are_balanced_disjoint_complete_and_ordered(self):
        units = [{**self.unit(str(i)), "order_seconds": cost} for i, cost in enumerate([30, 25, 20, 15, 10, 5])]
        plan = dp.sealed({"schema_version": 1, "selected": units})
        shards = [dp.shard_plan(plan, i, 3) for i in range(3)]
        keys = [u["key"] for p in shards for u in p["selected"]]
        self.assertEqual(sorted(keys), sorted(u["key"] for u in units))
        loads = [sum(u["order_seconds"] for u in p["selected"]) for p in shards]
        self.assertEqual(loads, [35, 35, 35])
        for p in shards:
            costs = [u["order_seconds"] for u in p["selected"]]
            self.assertEqual(costs, sorted(costs, reverse=True))
        with self.assertRaises(dp.PlanError): dp.shard_plan(plan, 3, 3)
        with self.assertRaises(dp.PlanError): dp.shard_plan(plan, 0, 0)

    def test_report_rejects_missing_duplicate_or_tampered_shards(self):
        parent = self.result()["plan"]
        results = [dp.sealed({"schema_version": 1, "kind": "daily-result", "lane": f"lane-shard{i}",
                    "plan": dp.shard_plan(parent, i, 2),
                    "records": [{"key": u["key"], "status": "passed", "seconds": 1}
                                for u in dp.shard_plan(parent, i, 2)["selected"]]}) for i in range(2)]
        summary = report.summarize(results)
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["unknown_timings"], parent["unknown_timings"])
        self.assertEqual(len(summary["weekly"]), 1)
        with self.assertRaises(dp.PlanError): report.summarize(results[:1])
        with self.assertRaises(dp.PlanError): report.summarize(results + results[:1])
        bad = copy.deepcopy(results); bad[0]["plan"]["selected"] = []
        bad[0]["plan"] = dp.sealed({k:v for k,v in bad[0]["plan"].items() if k != "digest"})
        bad[0] = dp.sealed({k:v for k,v in bad[0].items() if k != "digest"})
        with self.assertRaises(dp.PlanError): report.summarize(bad)

    def test_execution_window_reserves_cleanup_and_is_fresh_per_attempt(self):
        w = dp.policy()["window"]
        self.assertEqual(w["wall_minutes"], 600)
        self.assertEqual(w["execution_minutes"] + w["cleanup_minutes"], 600)
        window = dp.sealed({"schema_version": 1, "kind": "daily-window", "source_sha": "a"*40,
                            "started_at": "2026-11-04T14:00:00Z", "job_started_at": "2026-11-04T19:00:00Z",
                            "attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1"))})
        self.assertEqual(dp.execution_deadline(self.context(), window).isoformat(), "2026-11-04T23:30:00+00:00")
        window["source_sha"] = "b" * 40
        with self.assertRaises(dp.PlanError): dp.execution_deadline(self.context(), window)

    def test_budget_exhaustion_never_launches_or_passes_a_selected_case(self):
        unit = {**self.unit(), "binary": "/missing", "tests": [], "seed": 1, "env": {}}
        with tempfile.TemporaryDirectory() as folder, patch.object(run.shards, "listing") as listing:
            result = run.run_unit(unit, 0, Path(folder), 30, threading.Event(),
                                  dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc))
            self.assertEqual(result["status"], "budget_exhausted")
            listing.assert_not_called()
            self.assertEqual(json.loads((Path(folder)/"case-00000/result.json").read_text())["status"],
                             "budget_exhausted")

    def test_parent_timeout_observation_orders_but_does_not_predict_new_tuple(self):
        original = dp.load
        def load(path):
            if Path(path).name == "daily_timing_history.json":
                return {"samples": [{"key": "old", "case": "slow", "environment": "env",
                                     "seconds": 1800, "status": "timeout"}]}
            return original(path)
        with patch.object(dp, "load", side_effect=load):
            unit = {**self.unit("new"), "tests": [{"name": "slow"}]}
            plan = dp.build_plan(self.context(), "env", [unit], [])
        self.assertIsNone(plan["core"][0]["estimate_seconds"])
        self.assertEqual(plan["core"][0]["order_seconds"], 1800)
        self.assertEqual(plan["unknown_timings"], 1)

    def test_expansion_share_is_measured_and_scope_is_explicit(self):
        results = []
        for arch, seconds in (("x86_64", 90), ("aarch64", 10)):
            item = self.result()
            parent = {k:v for k,v in item["plan"].items() if k != "digest"}
            parent["environment"] = "lane/Linux/" + arch
            parent["selected"][0]["mode"] = "rotation"
            item["plan"] = dp.sealed(parent)
            item["lane"] = arch
            item["records"][0]["seconds"] = seconds
            results.append(dp.sealed({k:v for k,v in item.items() if k != "digest"}))
        summary = report.summarize(results)
        self.assertEqual(summary["expansion_seconds"], {"x86": 90, "arm": 10})
        self.assertIn("**10.0%**", report.markdown(summary))
        self.assertIn("aspirational", report.markdown(summary))

    def test_elapsed_window_is_separate_from_cumulative_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dp.write(root/"result.json", self.result())
            for minutes, expected in ((120, 0), (601, 1)):
                end = dt.datetime(2026, 11, 4, 14, tzinfo=dt.timezone.utc) + dt.timedelta(minutes=minutes)
                dp.write(root/"metrics.json", {"started_at": "2026-11-04T14:00:00Z",
                        "observed_at": end.isoformat(), "jobs": []})
                process = subprocess.run([sys.executable, str(ROOT/"scripts/testing/daily_report.py"),
                    str(root/"result.json"), "--metrics", str(root/"metrics.json"),
                    "--summary", str(root/"summary.md"), "--json", str(root/"summary.json")],
                    capture_output=True)
                self.assertEqual(process.returncode, expected, process.stderr)
                result = json.loads((root/"summary.json").read_text())
                self.assertEqual(result["elapsed_seconds"], minutes*60)
                self.assertEqual(result["total_seconds"], 20)

    def test_failed_only_retry_refreshes_clock_without_changing_plan_context(self):
        ctx = self.context()
        original = copy.deepcopy(ctx)
        attempt = {"started_at": "2026-11-04T14:00:00Z", "source_sha": "a"*40, "attempt": 2, "job_started_at": "2026-11-04T14:00:00Z"}
        with patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "2"}):
            window = window_helper.refresh(ctx, attempt, 2)
            self.assertEqual(ctx, original)
            self.assertEqual(dp.execution_deadline(ctx, window).isoformat(), "2026-11-04T18:30:00+00:00")
            stale = dp.sealed({"schema_version": 1, "kind": "daily-window", **attempt, "attempt": 1})
            with self.assertRaises(dp.PlanError): dp.execution_deadline(ctx, stale)
            with self.assertRaises(dp.PlanError): window_helper.refresh(ctx, {**attempt, "source_sha": "b"*40}, 2)
            with self.assertRaises(dp.PlanError): window_helper.refresh(ctx, attempt, 1)

    def test_four_shards_execute_real_children_and_reconcile_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root/"mock.py"
            binary.write_text("#!" + sys.executable + "\n" + MOCK)
            binary.chmod(0o755)
            cases = [{"name": "case" + str(i), "tags": "[hgraph]", "source_file": "fixture.cpp",
                      "source_line": i+1, "binary": "mock.py", "id": str(i)} for i in range(8)]
            binary.with_suffix(".json").write_text(json.dumps({"cases": cases}))
            units = [{**self.unit(str(i)), "binary": str(binary), "tests": [case], "seed": 42, "env": {}}
                     for i, case in enumerate(cases)]
            parent = dp.build_plan(self.context(), "mock", units, [])
            results = []
            for i in range(4):
                selected = dp.shard_plan(parent, i, 4)
                folder = root/str(i); folder.mkdir()
                records = [run.run_unit(unit, n, folder, 5, threading.Event())
                           for n, unit in enumerate(selected["selected"])]
                results.append(dp.sealed({"schema_version": 1, "kind": "daily-result",
                    "plan": selected, "lane": "lane-shard"+str(i), "records": records}))
            self.assertEqual(report.summarize(results)["status"], "passed")
            self.assertEqual(sum(len(r["records"]) for r in results), 8)
            with self.assertRaises(dp.PlanError): report.summarize(results[:-1])

    def test_non_catch_command_obeys_execution_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dp.write(root/"context.json", self.context())
            dp.write(root/"window.json", dp.sealed({"schema_version":1, "kind":"daily-window",
                "source_sha":"a"*40, "started_at":"2000-01-01T00:00:00Z", "job_started_at":"2000-01-01T00:00:00Z",
                "attempt":int(os.environ.get("GITHUB_RUN_ATTEMPT", "1"))}))
            env = dict(os.environ, VSAG_DAILY_CONTEXT=str(root/"context.json"),
                       VSAG_DAILY_LANE="measurement", VSAG_DAILY_ENVIRONMENT="local")
            command = [sys.executable, str(ROOT/"scripts/testing/daily_measure.py"), "--name", "example",
                       "--owner", "other", "--", sys.executable, "-c",
                       "from pathlib import Path; Path('must-not-run').touch()"]
            process = subprocess.run(command, cwd=root, env=env, capture_output=True)
            self.assertEqual(process.returncode, 1, process.stderr)
            self.assertFalse((root/"must-not-run").exists())
            result = json.loads((root/"daily-results/measurement-example/daily-result.json").read_text())
            self.assertEqual(result["records"][0]["status"], "budget_exhausted")

    def test_job_deadline_is_shared_after_setup_and_between_binaries(self):
        ctx = self.context()
        window = dp.sealed({"schema_version":1, "kind":"daily-window", "source_sha":"a"*40,
            "attempt":int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
            "started_at":"2026-11-04T14:00:00Z", "job_started_at":"2026-11-04T14:00:00Z"})
        cutoff = dp.execution_deadline(ctx, window)
        self.assertEqual(cutoff.isoformat(), "2026-11-04T18:30:00+00:00")
        after_setup = dt.datetime(2026, 11, 4, 15, tzinfo=dt.timezone.utc)
        after_first_binary = dt.datetime(2026, 11, 4, 17, 30, tzinfo=dt.timezone.utc)
        self.assertEqual((cutoff-after_setup).total_seconds(), 210*60)
        self.assertEqual((dp.execution_deadline(ctx, window)-after_first_binary).total_seconds(), 60*60)
        with self.assertRaises(dp.PlanError):
            run.inventory_timeout(dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc))

    def test_parent_inventory_counts_once_but_execution_counts_every_shard(self):
        core = [self.unit("core")]
        extension = [self.unit(str(i)) for i in range(99)]
        parent = dp.build_plan(self.context(), "x86", core, extension,
                    inventory={"exclusions":[{"test":{"name":"unsupported"},"reason":"disabled"}]})
        results = []
        for i in range(4):
            plan = dp.shard_plan(parent,i,4)
            results.append(dp.sealed({"schema_version":1,"kind":"daily-result","plan":plan,
                 "lane":f"lane-shard{i}","records":[{"key":u["key"],"status":"passed","seconds":1}
                 for u in plan["selected"]]}))
        summary = report.summarize(results)
        self.assertEqual(summary["unknown_timings"], 100)
        self.assertEqual(len(summary["weekly"]), 1)
        self.assertEqual(len(summary["exclusions"]), 1)
        self.assertEqual(summary["total_seconds"], len(parent["selected"]))

    def test_policy_clocks_and_shard_count_are_validated(self):
        baseline = dp.policy()
        original = dp.load
        for changes in ({"start":"8:00"}, {"start":"24:00"}, {"end":"09:00"}, {"timezone":"UTC"}):
            data = copy.deepcopy(baseline); data["window"].update(changes)
            def load(path):
                return data if Path(path).name == "daily_budget_policy.json" else original(path)
            with patch.object(dp, "load", side_effect=load), self.assertRaises(dp.PlanError): dp.policy()
        for count in (0, 33, True):
            data = copy.deepcopy(baseline); data["x86_shards"] = count
            with patch.object(dp, "load", return_value=data), self.assertRaises(dp.PlanError): dp.policy()

    def test_window_report_renders_policy_clocks(self):
        summary = report.summarize([self.result()])
        policy = copy.deepcopy(dp.policy())
        policy["window"].update(start="23:00", end="09:00")
        with patch.object(dp, "policy", return_value=policy):
            text = report.markdown(summary)
        self.assertIn("23:00–09:00", text)
        self.assertNotIn("22:00–08:00", text)

    def test_stage_guard_exports_only_remaining_job_minutes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); dp.write(root/"context.json", self.context())
            now = dt.datetime.now(dt.timezone.utc)
            for job_age, expected in ((60, 0), (300, 1)):
                attempt = {"source_sha":"a"*40, "attempt":1,
                    "started_at":(now-dt.timedelta(hours=6)).isoformat(),
                    "job_started_at":(now-dt.timedelta(minutes=job_age)).isoformat()}
                dp.write(root/"attempt.json", attempt)
                (root/"output").write_text("")
                process = subprocess.run([sys.executable, str(ROOT/"scripts/testing/daily_window.py"),
                    "--context", str(root/"context.json"), "--attempt", str(root/"attempt.json")],
                    env=dict(os.environ, GITHUB_RUN_ATTEMPT="1", GITHUB_OUTPUT=str(root/"output")),
                    capture_output=True)
                self.assertEqual(process.returncode, expected, process.stderr)
                text = (root/"output").read_text()
                if expected == 0:
                    minutes = int(text.strip().split("=")[1])
                    self.assertGreater(minutes, 208); self.assertLessEqual(minutes, 210)
                else:
                    self.assertEqual(text, "")

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
            self.assertIn("above reference", report.markdown(summary))

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
            interrupted = MOCK.replace(
                "root = ET.Element('Catch2TestRun')",
                "Path(os.environ['VSAG_DAILY_PHASE_FILE']).write_text("
                "json.dumps({'phase': 'Build', 'event': 'start', 'details': {'quantizer': 'fp32'}}) + '\\n')\n"
                "report.write_text('<Catch2TestRun>')\n"
                "root = ET.Element('Catch2TestRun')")
            binary.write_text("#!" + sys.executable + "\n" + interrupted)
            binary.chmod(0o755)
            test = {"name": "case", "tags": "[hgraph]", "source_file": "test.cpp",
                    "source_line": 1, "binary": "mock.py", "id": "test", "sleep": 5}
            binary.with_suffix(".json").write_text(json.dumps({"cases": [test]}))
            unit = {**self.unit(), "binary": str(binary), "tests": [test], "seed": 42,
                    "env": {"VSAG_DAILY_HGRAPH": "{}"}}
            result = run.run_unit(unit, 0, root, 0.5, threading.Event())
            self.assertEqual(result["status"], "timeout", result)
            self.assertGreaterEqual(result["seconds"], 0.5)
            self.assertEqual(result["phase_events"], 1)
            self.assertEqual(result["active_phases"], [{"phase": "Build", "event": "start",
                                                      "details": {"quantizer": "fp32"}}])
            self.assertTrue((root / result["directory"] / result["phase_trace"]).is_file())

    def test_phase_trace_preserves_nested_progress_with_truncated_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "phases.jsonl"
            self.assertEqual(run.phase_diagnostics(trace), {})
            events = [{"phase": "Quantizer", "event": "start"},
                      {"phase": "Build", "event": "start"},
                      {"phase": "Build", "event": "end", "seconds": 12},
                      {"phase": "TestGeneral", "event": "start"}]
            trace.write_text("".join(json.dumps(event) + "\n" for event in events) + '{"phase":')
            result = run.phase_diagnostics(trace)
            self.assertEqual(result["phase_events"], 4)
            self.assertEqual(result["active_phases"], [events[0], events[3]])
            self.assertEqual(result["last_phase_event"], events[3])
            self.assertIn("phase_trace_error", result)
            with patch.object(Path, "read_text", side_effect=PermissionError("unreadable")):
                self.assertIn("Cannot read phase trace", run.phase_diagnostics(trace)["phase_trace_error"])

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
                        self.assertIn("Selected-check correctness/completeness: **failed**", (root / "summary.md").read_text())

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



class DailyNamedCountTests(unittest.TestCase):
    context = DailyPlanTests.context
    unit = DailyPlanTests.unit
    result = DailyPlanTests.result

    def named_result(self, lane="x86", status="passed"):
        tests = [{"name": "case", "source_file": "/checkout/tests/test_x.cpp"}]
        core = [{**self.unit(), "tests": tests}]
        extension = [{**self.unit("rotation"), "tests": tests, "mode": "rotation"}]
        plan = dp.build_plan(self.context(), lane, core, extension,
                             inventory={"named_cases": tests + [{"name": "other", "source_file": "tests/test_x.cpp"}]})
        return dp.sealed({"schema_version": 1, "kind": "daily-result", "lane": lane,
                          "plan": plan, "records": [{"key": u["key"], "status": status, "seconds": 1}
                                                     for u in plan["selected"]]})

    def test_counts_deduplicate_platforms_and_configurations(self):
        a = self.named_result()
        b = self.named_result("mac", "timeout")
        counts = report.summarize([a, b])["case_counts"]
        self.assertEqual((counts["passed"], counts["failed"], counts["selected"], counts["global"]), (0, 1, 1, 2))
        self.assertFalse(counts["legacy_inventory"])

    def test_missing_execution_is_pending_not_passed(self):
        r = self.named_result()
        r["records"] = []
        r = dp.sealed({k: v for k, v in r.items() if k != "digest"})
        summary = report.summarize([r], allow_incomplete=True)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["case_counts"]["pending"], 1)
        self.assertEqual(summary["case_counts"]["passed"], 0)

    def test_shard_order_and_missing_sibling_do_not_inflate_passes(self):
        r = self.named_result()
        # Put two configurations of the same named case in the core.
        core = [{**self.unit(key), "tests": r["plan"]["core"][0]["tests"]} for key in ("a", "b")]
        parent = dp.build_plan(self.context(), "env", core, [], inventory=r["plan"]["inventory"])
        shards = []
        for i in range(2):
            plan = dp.shard_plan(parent, i, 2)
            shards.append(dp.sealed({"schema_version": 1, "kind": "daily-result", "lane": f"test-shard{i}",
                                    "plan": plan, "records": [{"key": u["key"], "status": "passed", "seconds": 1} for u in plan["selected"]]}))
        self.assertEqual(report.summarize(shards)["case_counts"], report.summarize(shards[::-1])["case_counts"])
        self.assertEqual(report.summarize(shards)["case_counts"]["passed"], 1)
        partial = report.summarize(shards[:1], allow_incomplete=True)
        self.assertEqual(partial["case_counts"]["pending"], 1)
        self.assertEqual(partial["status"], "failed")

    def test_non_catch_command_is_not_a_named_case(self):
        r = self.result()
        counts = report.summarize([r])["case_counts"]
        self.assertEqual(counts["selected"], 0)
        self.assertEqual(counts["command_groups"], {"linux": "passed"})

    def test_missing_lane_preserves_counts_and_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dp.write(root / "result.json", self.named_result())
            proc = subprocess.run([sys.executable, str(ROOT / "scripts/testing/daily_report.py"),
                                   str(root / "result.json"), "--expected-lanes", "x86", "missing",
                                   "--summary", str(root / "summary.md"), "--json", str(root / "summary.json")],
                                  capture_output=True)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            summary = dp.load(root / "summary.json")
            self.assertTrue(summary["incomplete"])
            self.assertEqual(summary["case_counts"]["selected"], 1)
            self.assertIn("Passed cases", (root / "summary.md").read_text())
            self.assertIn("Incomplete evidence", (root / "summary.md").read_text())

    def test_missing_manifest_or_metrics_keeps_verified_counts_but_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dp.write(root / "result.json", self.named_result())
            proc = subprocess.run([sys.executable, str(ROOT / "scripts/testing/daily_report.py"),
                                   str(root / "result.json"), "--expected-file", str(root / "missing.json"),
                                   "--metrics", str(root / "missing-metrics.json"),
                                   "--summary", str(root / "summary.md"), "--json", str(root / "summary.json")],
                                  capture_output=True)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            summary = dp.load(root / "summary.json")
            self.assertTrue(summary["incomplete"])
            self.assertEqual(summary["case_counts"]["selected"], 1)
            self.assertEqual(len(summary["coverage_gaps"]), 2)

    def test_macos_examples_have_independent_failure_safe_step(self):
        action = (ROOT / ".github/actions/asan-build-and-test-test-asan-macos-arm64/action.yml").read_text()
        test, examples = action.split("    - name: Run full examples independently\n")
        examples = examples.split("    - name: Upload Log")[0]
        self.assertNotIn("daily_measure.py", test)
        self.assertIn("!cancelled()", examples)
        self.assertIn("steps.download-tests.outcome == 'success'", examples)
        self.assertIn("inputs.matrix-test-type == 'unittests'", examples)
        self.assertIn("daily_measure.py --name examples", examples)

    def test_counts_expose_skips_and_budget_exhaustion_as_not_executed(self):
        for status in ("skipped", "budget_exhausted", "cancelled", "launch_error"):
            counts = report.summarize([self.named_result(status=status)])["case_counts"]
            self.assertEqual((counts["passed"], counts["failed"], counts["pending"]), (0, 0, 1))

    def test_corrupt_input_has_unknown_counts_and_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = self.named_result()
            value["records"][0]["status"] = "failed"  # Do not reseal: reject corruption.
            dp.write(root / "result.json", value)
            proc = subprocess.run([sys.executable, str(ROOT / "scripts/testing/daily_report.py"),
                                   str(root / "result.json"), "--summary", str(root / "summary.md"),
                                   "--json", str(root / "summary.json")], capture_output=True)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("| — | — | — | — |", (root / "summary.md").read_text())

    def test_absolute_build_paths_do_not_duplicate_global_cases(self):
        import daily_counts
        self.assertEqual(daily_counts.case_identity({"source_file": "/runner/build/src/a_test.cpp", "name": "a"}),
                         daily_counts.case_identity({"source_file": "src/a_test.cpp", "name": "a"}))
        self.assertNotEqual(daily_counts.case_identity({"source_file": "src/a_test.cpp", "name": "a"}),
                            daily_counts.case_identity({"source_file": "tests/a_test.cpp", "name": "a"}))

    def test_second_quarantine_keeps_other_duplicate_configurations(self):
        rules = dp.load(dp.HERE / "daily_matrix.json")["quarantine"]
        cosine = next(r for r in rules if r["key"] == "df2ccbce1f3092b1dc00a580b4e6e2176e55e694eca7ed7c222ae100d1618bb5")
        unit = {**self.unit(cosine["key"]), "mode": "rotation", "tests": [{"name": cosine["case"]}]}
        kept, excluded = dp.quarantine_extensions([unit, {**unit, "key": "different-configuration"}], rules)
        self.assertEqual([u["key"] for u in kept], ["different-configuration"])
        self.assertEqual([u["key"] for u in excluded], [cosine["key"]])

    def test_group_failure_uses_individual_case_results(self):
        import daily_counts
        tests = [{"name": name, "source_file": "tests/x.cpp"} for name in ("SIMD", "simd")]
        unit = {"tests": tests}
        record = {"tests": [{**tests[0], "success": True, "skips": 0},
                             {**tests[1], "success": False, "skips": 0}]}
        self.assertEqual(daily_counts.named_status(unit, record, tests[0], "failed"), "passed")
        self.assertEqual(daily_counts.named_status(unit, record, tests[1], "failed"), "failed")
        self.assertEqual(daily_counts.named_status(unit, {}, tests[0], "timeout"), "not_run")
        self.assertEqual(daily_counts.named_status(unit, {**record, "error": "invalid plan"}, tests[0], "failed"), "not_run")


if __name__ == "__main__":
    unittest.main()
