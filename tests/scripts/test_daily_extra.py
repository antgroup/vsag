"""Supplemental selection/accounting is independent from mandatory Daily scope."""
import copy
import datetime as dt
import io
import json
import os
import signal
import subprocess
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/testing"))
import daily_plan as dp
import daily_extra as extra
import daily_report as report
import daily_run as run
import daily_job_report as jobs


class ExtraTests(unittest.TestCase):
    def source_results(self, group_rotation_cases=False):
        now = dt.datetime.now(dt.timezone.utc)
        ctx = dp.context("a" * 40, now.isoformat(), "2026-11-06")
        def unit(i, mode):
            cases = [{"name": "case" + str(i), "source_file": "tests/fake.cpp"}]
            if group_rotation_cases and mode == "rotation":
                cases.append({**cases[0], "name": cases[0]["name"].upper()})
            return {"key": str(i), "owner": "hgraph", "mode": mode, "tests": cases}
        parent = dp.build_plan(ctx, "daily_test_asan_x86/Linux/x86_64",
                               [unit(i, "core") for i in range(4)],
                               [unit(i, "rotation") for i in range(4, 32)],
                               inventory={"named_cases": [], "exclusions": []})
        results = []
        for i in range(4):
            plan = dp.shard_plan(parent, i, 4)
            results.append(dp.sealed({"schema_version": 1, "kind": "daily-result", "plan": plan,
                "lane": f"daily_test_asan_x86-functests-shard{i}", "records": [
                    {"key": u["key"], "status": "passed", "seconds": 10} for u in plan["selected"]]}))
        return results

    def proposal(self, hours=4):
        now = dt.datetime.now(dt.timezone.utc)
        return extra.propose(self.source_results(), (now-dt.timedelta(hours=hours)).isoformat(), now, 1, "123")

    def test_candidates_disjoint_from_daily_and_across_shards(self):
        p = self.proposal()
        extra.validate(p)
        keys = [u["key"] for shard in p["shards"] for u in shard]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertFalse(set(keys) & set(p["mandatory_keys"]))
        self.assertEqual(len(keys) + len(p["mandatory_keys"]), 32)
        self.assertEqual(p["budget_seconds"], 240*60)
        self.assertEqual(p["workers_per_shard"], 2)

    def test_attempt_clock_reserves_one_hour_and_delayed_queue_gets_zero(self):
        p = self.proposal(hours=8)
        self.assertAlmostEqual(p["budget_seconds"], 3600, places=1)
        p = self.proposal(hours=10)
        self.assertEqual(p["budget_seconds"], 0)

    def test_missing_or_failed_base_shards_are_not_a_topup_license(self):
        results = self.source_results()
        now = dt.datetime.now(dt.timezone.utc)
        with self.assertRaises(dp.PlanError):
            extra.propose(results[:-1], now.isoformat(), now, 1, "123")
        results[0]["records"][0]["status"] = "failed"
        results[0] = dp.sealed({k:v for k,v in results[0].items() if k != "digest"})
        with self.assertRaises(dp.PlanError):
            extra.propose(results, now.isoformat(), now, 1, "123")

    def update_record(self, results, mode, **fields):
        for index, result in enumerate(results):
            units = {u["key"]: u for u in result["plan"]["selected"]}
            for record in result["records"]:
                if units[record["key"]]["mode"] == mode:
                    record.update(fields)
                    results[index] = dp.sealed({k: v for k, v in result.items() if k != "digest"})
                    return index, record["key"]
        self.fail("No selected configuration for " + mode)

    def test_rotation_timeout_allows_disjoint_supplemental_plan_and_warns_in_summary(self):
        results = self.source_results()
        _, key = self.update_record(results, "rotation", status="timeout",
                                    timeout_scope="case", exit_code=-signal.SIGTERM, seconds=1800)
        now = dt.datetime.now(dt.timezone.utc)
        proposal = extra.propose(results, now.isoformat(), now, 1, "123")
        self.assertIn(key, proposal["mandatory_keys"])
        self.assertNotIn(key, {u["key"] for shard in proposal["shards"] for u in shard})

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            paths = [root / f"{i}.json" for i in range(len(results))]
            for path, result in zip(paths, results):
                dp.write(path, result)
            summary, artifact = root / "summary.md", root / "summary.json"
            proc = subprocess.run([sys.executable, str(ROOT / "scripts/testing/daily_report.py"),
                                   *map(str, paths), "--summary", str(summary), "--json", str(artifact)],
                                  env={**os.environ, "NEEDS_JSON": "{}"}, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            actual = dp.load(artifact)
            self.assertEqual(actual["status"], "passed_with_warnings")
            self.assertEqual(actual["failures"], [])
            self.assertEqual(actual["case_counts"]["failed"], 1)
            self.assertEqual(actual["case_counts"]["configurations"]["failed"], 1)
            self.assertEqual([(r["key"], r["status"]) for r in actual["rotation_timeouts"]],
                             [(key, "timeout")])
            self.assertIn("Rotation timeouts (warning)", summary.read_text())

            missing = subprocess.run([sys.executable, str(ROOT / "scripts/testing/daily_report.py"),
                                      *map(str, paths[:-1]), "--summary", str(summary), "--json", str(artifact)],
                                     env={**os.environ, "NEEDS_JSON": "{}"}, capture_output=True)
            self.assertEqual(missing.returncode, 1, missing.stderr)
            self.assertTrue(dp.load(artifact)["incomplete"])

        self.update_record(results, "core", status="failed")
        self.assertEqual(report.summarize(results)["status"], "failed")
        with self.assertRaises(dp.PlanError):
            extra.propose(results, now.isoformat(), now, 1, "123")

    def test_grouped_rotation_timeout_counts_configuration_failed_and_cases_pending(self):
        results = self.source_results(group_rotation_cases=True)
        self.update_record(results, "rotation", status="timeout", timeout_scope="case",
                           exit_code=-signal.SIGTERM, seconds=1800)
        summary = report.summarize(results)
        self.assertEqual(summary["status"], "passed_with_warnings")
        counts = summary["case_counts"]
        self.assertEqual(counts["configurations"]["failed"], 1)
        self.assertEqual(counts["failed"], 0)
        self.assertEqual(counts["pending"], 2)

    def test_timeout_policy_rejects_core_window_and_recorded_failure(self):
        now = dt.datetime.now(dt.timezone.utc)
        for mode, fields in (
                ("core", {}),
                ("rotation", {"timeout_scope": "window"}),
                ("rotation", {"timeout_scope": None}),
                ("rotation", {"exit_code": -signal.SIGSEGV}),
                ("rotation", {"error": "Test failure recorded before timeout"})):
            with self.subTest(mode=mode, fields=fields):
                results = self.source_results()
                timeout = {"status": "timeout", "timeout_scope": "case",
                           "exit_code": -signal.SIGTERM, "seconds": 1800}
                self.update_record(results, mode, **{**timeout, **fields})
                summary = report.summarize(results)
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(summary["rotation_timeouts"], [])
                with self.assertRaises(dp.PlanError):
                    extra.propose(results, now.isoformat(), now, 1, "123")

    def test_other_lanes_keep_per_case_rotation_timeouts_blocking(self):
        results = self.source_results()
        index, key = self.update_record(results, "rotation", status="timeout",
                                        timeout_scope="case", exit_code=-signal.SIGTERM, seconds=1800)
        for lane in ("daily_test_asan_x86-unittests-shard0", "daily-extra-x86-functests-shard0",
                     "daily_test_asan_aarch64-functests"):
            result = {**results[index], "lane": lane}
            result = dp.sealed({k: v for k, v in result.items() if k != "digest"})
            changed = [result if i == index else value for i, value in enumerate(results)]
            with self.subTest(lane=lane):
                summary = report.summarize(changed)
                self.assertEqual(summary["failures"], [key + ": timeout"])
                self.assertEqual(summary["rotation_timeouts"], [])

    def test_runner_reports_timeout_only_as_warning_and_preserves_mixed_failure(self):
        results = self.source_results()
        index, timeout_key = self.update_record(results, "rotation", status="timeout",
                                      timeout_scope="case", exit_code=-signal.SIGTERM, seconds=1800)
        plan = results[index]["plan"]
        parent = {k: v for k, v in plan.items() if k not in ("digest", "sharding")}
        parent.update(selected=plan["sharding"]["full_selected"], digest=plan["sharding"]["parent_digest"])
        records = {r["key"]: r for r in results[index]["records"]}
        self.assertGreater(len(records), 1)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            context, binary = root / "context.json", root / "functests"
            dp.write(context, plan["context"])
            binary.touch()
            for mixed in (False, True):
                with self.subTest(mixed=mixed):
                    if mixed:
                        next(r for r in records.values() if r["status"] == "passed")["status"] = "failed"
                    out = root / str(mixed)
                    lane = results[index]["lane"]
                    argv = ["daily_run.py", "--binary", str(binary), "--context", str(context),
                            "--lane", lane, "--environment", plan["environment"],
                            "--profile", "functional", "--output", str(out),
                            "--shard-index", str(index), "--shard-count", "4"]
                    stdout = io.StringIO()
                    with (redirect_stdout(stdout),
                          patch.object(sys, "argv", argv),
                          patch.object(run, "inventory_units", return_value=([], [], {})),
                          patch.object(dp, "build_plan", return_value=parent),
                          patch.object(dp, "execution_deadline", return_value=None),
                          patch.object(signal, "signal"),
                          patch.object(run, "run_unit", side_effect=lambda unit, *args: records[unit["key"]])):
                        self.assertEqual(run.main(), int(mixed))
                    self.assertEqual(stdout.getvalue().splitlines(), [
                        f"::warning::Nightly rotation timed out in {lane}: {timeout_key}"
                    ])
                    recorded = dp.load(out / "daily-result.json")
                    self.assertIn("timeout", [r["status"] for r in recorded["records"]])

    def test_adaptive_execution_reports_used_and_budget_deferred_separately(self):
        p = self.proposal()
        def fake(unit, number, output, timeout, cancel, deadline):
            return {"key": unit["key"], "status": "passed", "seconds": 2}
        with tempfile.TemporaryDirectory() as d:
            results = [extra.execute(p, i, Path(d), run_unit=fake, enabled=i!=3) for i in range(4)]
        extra.reconcile(p, results)
        summary = report.summarize(results)
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["total_seconds"], 0)  # core quota excludes extra
        cost = extra.accounting(p, results, [])
        self.assertEqual(cost["selected"] + cost["deferred"], sum(map(len,p["shards"])))
        self.assertEqual(cost["deferred"], len(p["shards"][3]))
        self.assertEqual(cost["test_seconds"], cost["selected"]*2)
        self.assertIn("Budget-deferred", "\n".join(report.extra_markdown(cost)))
        # Missing one supplemental shard is missing evidence, not budget deferral.
        with self.assertRaises(dp.PlanError):
            extra.reconcile(p,results[:-1])

    def test_extra_report_cannot_hide_or_invent_candidates(self):
        p = self.proposal()
        with tempfile.TemporaryDirectory() as d:
            results = [extra.execute(p, i, Path(d), enabled=False) for i in range(4)]
        extra.reconcile(p, results)
        bad = copy.deepcopy(results)
        bad[0]["plan"]["extra"]["deferred_keys"].pop()
        bad[0]["plan"] = dp.sealed({k:v for k,v in bad[0]["plan"].items() if k!='digest'})
        bad[0] = dp.sealed({k:v for k,v in bad[0].items() if k!='digest'})
        with self.assertRaises(dp.PlanError):extra.reconcile(p,bad)

    def test_supplemental_failure_still_fails_final_report(self):
        p = self.proposal()
        def fail(unit, *args):
            return {"key": unit["key"], "status": "failed", "seconds": 1}
        with tempfile.TemporaryDirectory() as d:
            results = [extra.execute(p, i, Path(d), run_unit=fail) for i in range(4)]
        extra.reconcile(p,results)
        self.assertEqual(report.summarize(results)["status"], "failed")

    def test_expired_budget_never_launches_test(self):
        p=self.proposal(hours=10)
        def fail(*args):
            self.fail("Should not start a test after cutoff")
        with tempfile.TemporaryDirectory() as d:
            r=extra.execute(p,0,Path(d),run_unit=fail)
        self.assertEqual(r["records"],[])
        self.assertEqual(len(r["plan"]["extra"]["deferred_keys"]),len(p["shards"][0]))

    def test_manifest_rejects_overlap_and_wrong_resources(self):
        for mutate in (lambda p:p.update(workers_per_shard=8),
                       lambda p:p.update(budget_seconds=30000),
                       lambda p:p["mandatory_keys"].append(p["shards"][0][0]["key"])):
            p=self.proposal();mutate(p)
            p=dp.sealed({k:v for k,v in p.items() if k!='digest'})
            with self.assertRaises(dp.PlanError):extra.validate(p)


class JobDiagnosticsTests(unittest.TestCase):
    def test_missing_diagnostic_identity_is_explicit_and_not_silently_accepted(self):
        required = {"GITHUB_SHA": "a" * 40, "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1",
                    "GITHUB_JOB": "test", "DAILY_JOB_STATUS": "success"}
        for name in required:
            with patch.dict(os.environ, {k: v for k, v in required.items() if k != name}, clear=True):
                with self.assertRaisesRegex(ValueError, name):
                    jobs.collect()
        with patch.dict(os.environ, required, clear=True), patch.object(jobs.subprocess, "run", side_effect=FileNotFoundError), patch.object(jobs.Path, "is_file", return_value=False):
            data = jobs.collect()
            self.assertEqual(data["run_id"], "123")
            self.assertFalse(data["cache"]["available"])
            self.assertIsNone(data["matrix"])

    def test_container_cache_statistics_override_unrelated_host_cache(self):
        required = {"GITHUB_SHA": "a" * 40, "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1",
                    "GITHUB_JOB": "wheel", "DAILY_JOB_STATUS": "success"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "python-build-report").mkdir()
            (root / "python-build-report/ccache.tsv").write_text(
                "cache_hit_direct 8\ncache_hit_preprocessed 1\ncache_miss 1\n")
            previous = os.getcwd()
            try:
                os.chdir(root)
                with patch.dict(os.environ, required, clear=True), patch.object(
                        jobs.subprocess, "run", side_effect=FileNotFoundError):
                    self.assertEqual(jobs.collect()["cache"]["hit_percent"], 90)
            finally:
                os.chdir(previous)

    def test_legacy_container_stats_and_missing_diagnostics_do_not_use_host_cache(self):
        required = {"GITHUB_SHA": "a" * 40, "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1",
                    "GITHUB_JOB": "wheel", "DAILY_JOB_STATUS": "success"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'python-build-report'
            folder.mkdir()
            previous = os.getcwd()
            try:
                os.chdir(root)
                with patch.dict(os.environ, required, clear=True), patch.object(jobs.subprocess, 'run') as run:
                    run.return_value.stdout = 'cache_hit_direct 100\ncache_hit_preprocessed 0\ncache_miss 0\n'
                    self.assertFalse(jobs.collect()['cache']['available'])
                    (folder / 'ccache.txt').write_text('cache hit (direct) 2\ncache hit (preprocessed) 1\ncache miss 7\n')
                    self.assertEqual(jobs.collect()['cache']['hit_percent'], 30)
                    (folder / 'ccache.txt').write_text('ccache: command not found')
                    self.assertFalse(jobs.collect()['cache']['available'])
            finally:
                os.chdir(previous)

    def test_cache_rate_uses_cacheable_calls_and_no_calls_is_unmeasured(self):
        c=jobs.cache_statistics("cache_hit_direct 2\ncache_hit_preprocessed 1\ncache_miss 7\ncalled_for_link 4\n")
        self.assertEqual(c["hit_percent"],30)
        self.assertEqual(c["cacheable_calls"],10)
        self.assertIsNone(jobs.cache_statistics("cache_hit_direct 0\ncache_hit_preprocessed 0\ncache_miss 0")["hit_percent"])
        self.assertFalse(jobs.cache_statistics("unrelated 0")["available"])

    def test_current_cache_counters_feed_summary_and_require_complete_evidence(self):
        machine_stats = "direct_cache_hit\t86\npreprocessed_cache_hit\t2\ncache_miss\t18313\ncalled_for_link\t327\n"
        cache = jobs.cache_statistics(machine_stats)
        self.assertTrue(cache["available"])
        self.assertEqual((cache["hits"], cache["misses"], cache["cacheable_calls"]), (88, 18313, 18401))
        self.assertAlmostEqual(cache["hit_percent"], 100 * 88 / 18401)
        report = {"job": "daily_build_asan_x86", "matrix": None, "coverage": {"available": False},
                  "status_before_upload": "success", "cache": cache}
        self.assertIn("0.48%", "\n".join(jobs.markdown([report])))

        self.assertFalse(jobs.cache_statistics("direct_cache_hit 86\ncache_miss 18313\n")["available"])
        self.assertFalse(jobs.cache_statistics("direct_cache_hit 86\npreprocessed_cache_hit 2\n")["available"])
        self.assertIsNone(jobs.cache_statistics(
            "direct_cache_hit 0\npreprocessed_cache_hit 0\ncache_miss 0\n")["hit_percent"])

    def test_scoped_coverage_is_separate_from_job_success(self):
        c=jobs.coverage_statistics("SF:src/x.cpp\nLF:100\nLH:85\nend_of_record\n")
        self.assertEqual(c["line_percent"],85)
        self.assertFalse(c["meets_target"])
        text="\n".join(jobs.markdown([{"job":"full-coverage-coverage","matrix":None,
            "coverage":c,"status_before_upload":"success","cache":{"available":False}}]))
        self.assertIn("below target",text)
        for bad in ("SF:/extern/x.cpp\nLF:10\nLH:10\nend_of_record", "SF:src/x.cpp\nLF:1\nLH:2"):
            with self.assertRaises(ValueError):jobs.coverage_statistics(bad)

    def test_duplicate_or_mixed_job_reports_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'job-report.json'
            r={"schema_version":1,"source_sha":"a"*40,"run_id":"1","job":"build","matrix":None}
            p.write_text(json.dumps(r))
            self.assertEqual(len(jobs.load_reports(d,'a'*40,'1')),1)
            with self.assertRaises(ValueError):jobs.load_reports(d,'b'*40,'1')
            q=Path(d)/'duplicate';q.mkdir();(q/'job-report.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError):jobs.load_reports(d,'a'*40,'1')


if __name__ == '__main__':unittest.main()
