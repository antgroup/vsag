#!/usr/bin/env python3
"""Strict selected/executed reconciliation; budget excess is only a warning."""
from __future__ import annotations

import argparse
from collections import Counter
import math
import os
from pathlib import Path
import sys

import daily_plan as dp


def summarize(results):
    if not results:
        raise dp.PlanError("No daily results")
    ctx = None
    identities = set()
    totals = {b: 0.0 for b in dp.BUCKETS}
    failures = []
    incomplete = []
    unknown = 0
    slowest = []
    weekly = []
    exclusions = []
    for result in results:
        dp.verify(result)
        if result.get("kind") != "daily-result":
            raise dp.PlanError("Wrong result kind")
        plan = result["plan"]
        dp.verify(plan)
        if plan.get("kind") != "daily-plan":
            raise dp.PlanError("Wrong plan kind")
        dp.validate_context(plan["context"])
        if ctx is None:
            ctx = plan["context"]
        if ctx != plan["context"]:
            raise dp.PlanError("Mixed source/date/rotation context")
        identity = (plan["environment"], result["lane"])
        if identity in identities:
            raise dp.PlanError("Duplicate lane result")
        identities.add(identity)
        expected = {u["key"]: u for u in plan["selected"]}
        for exclusion in plan.get("inventory", {}).get("exclusions", []):
            exclusions.append({"lane": result["lane"], **exclusion})
        records = result["records"]
        if Counter(r["key"] for r in records) != Counter(expected.keys()):
            raise dp.PlanError("Missing, duplicate or unexpected execution result")
        unknown += plan["unknown_timings"]
        weekly.append({"lane": result["lane"], "core": len(plan["core"]),
                       "rotation": [len(day) for day in plan["rotation"]]})
        for record in records:
            seconds = record["seconds"]
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or seconds < 0):
                raise dp.PlanError("Invalid elapsed time")
            unit = expected[record["key"]]
            totals[unit["owner"]] += seconds
            slowest.append({"lane": result["lane"], "key": record["key"], "seconds": seconds,
                            "name": unit.get("tests", [{}])[0].get("name", result["lane"]),
                            "mode": unit.get("mode", "core"), "status": record["status"]})
            if record["status"] != "passed":
                failures.append(record["key"] + ": " + record["status"])
        incomplete.extend(result.get("coverage_gaps", []))
    budgets = dp.policy()["minutes"]
    return {"source_sha": ctx["source_sha"], "date": ctx["logical_date"], "slot": ctx["slot"],
            "seconds": totals, "total_seconds": sum(totals.values()), "budgets_minutes": budgets,
            "exclusions": exclusions, "failures": failures, "coverage_gaps": incomplete, "unknown_timings": unknown, "weekly": weekly,
            "slowest": sorted(slowest, key=lambda row: -row["seconds"])[:20],
            "status": "failed" if failures or incomplete else "passed"}


def markdown(report):
    lines = ["# Daily regression", "",
             f"Source: `{report['source_sha']}`; date: {report['date']}; slot: {report['slot']}.",
             "", "Six hours means cumulative test execution, including parallel workers and repeats.",
             "Compilation, lint, queueing and artifact transfer are outside this test budget.", "",
             "| Area | Budget (min) | Actual (min) | State |",
             "| --- | ---: | ---: | --- |"]
    for key, budget in report["budgets_minutes"].items():
        actual = report["seconds"][key] / 60
        ratio = actual / budget
        state = "🟢" if ratio < .9 else "🟡" if ratio <= 1 else "🔴 over budget"
        lines.append(f"| {key} | {budget} | {actual:.2f} | {state} |")
    lines += ["", f"Total: **{report['total_seconds'] / 60:.2f} / 360 min**.",
              f"Correctness/completeness: **{report['status']}**.",
              f"Unmeasured configuration groups: **{report['unknown_timings']}**.",
              "Unknown estimates are not evidence that the budget will fit.",
              "", "<details><summary>Failures and coverage gaps</summary>", ""]
    lines.extend("- " + x for x in report["failures"] + report["coverage_gaps"])
    lines += ["", "</details>", "", "A single day does not prove full weekly or release coverage.", ""]
    lines += ["<details><summary>Capability exclusions (not validated in these lanes)</summary>", ""]
    for item in report["exclusions"]:
        lines.append("- " + item["lane"] + " / " + item["test"]["name"] + ": " + item["reason"])
    lines += ["", "Unexpected runtime skips still fail the run.", "", "</details>", ""]
    lines += ["<details><summary>Slowest configurations (all lanes)</summary>", "",
              "| Lane / case | Core or rotation | Seconds | Status |", "| --- | --- | ---: | --- |"]
    for row in report["slowest"]:
        name = (row["lane"] + " / " + row["name"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {name} | {row['mode']} | {row['seconds']:.2f} | {row['status']} |")
    lines += ["", "</details>", "", "<details><summary>Seven-day planned configuration counts</summary>", ""]
    for row in report["weekly"]:
        lines.append(f"- {row['lane']}: core {row['core']}; rotation {row['rotation']}")
    lines += ["", "These are planned counts, not seven-day pass evidence.", "", "</details>", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="*")
    p.add_argument("--collect")
    p.add_argument("--expected-file")
    p.add_argument("--summary", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--expected-lanes", nargs="*", default=[])
    args = p.parse_args()
    try:
        paths = args.results + ([str(path) for path in Path(args.collect).rglob("daily-result.json")] if args.collect else [])
        results = [dp.load(path) for path in paths]
        if args.expected_file:
            args.expected_lanes = dp.load(args.expected_file)
        if args.expected_lanes and Counter(r["lane"] for r in results) != Counter(args.expected_lanes):
            raise dp.PlanError("Selected workflow lanes did not all report")
        report = summarize(results)
        needs = __import__("json").loads(os.environ.get("NEEDS_JSON", "{}"))
        for name, job in needs.items():
            if job.get("result") != "success":
                report["failures"].append("Workflow dependency " + name + ": " + str(job.get("result")))
                report["status"] = "failed"
        dp.write(args.json, report)
        Path(args.summary).write_text(markdown(report))
    except (dp.PlanError, OSError, KeyError, ValueError) as error:
        message = str(error)
        dp.write(args.json, {"status": "failed", "incomplete": True, "error": message})
        Path(args.summary).write_text(
            "# Daily regression: incomplete evidence\n\n"
            "Validation failed: " + message.replace("\n", " ") + "\n\n"
            "Missing or invalid results are not a pass or zero test cost.\n")
        print("Daily report failed: " + message, file=sys.stderr)
        return 1
    for b, seconds in report["seconds"].items():
        if seconds > report["budgets_minutes"][b] * 60:
            print(f"::warning::Daily {b} exceeded its test-time budget")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (dp.PlanError, OSError, KeyError, ValueError) as error:
        print("Daily report failed: " + str(error), file=sys.stderr)
        sys.exit(1)
