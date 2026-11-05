#!/usr/bin/env python3
"""Strict selected/executed reconciliation; budget excess is only a warning."""
from __future__ import annotations

import argparse
from collections import Counter
import math
import os
import re
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
    architecture_scopes = []
    shard_sets = {}
    seen_metadata = set()
    expansion_seconds = {"x86": 0.0, "arm": 0.0}
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
        if plan.get("sharding"):
            shard = plan["sharding"]
            parent = {k: v for k, v in plan.items() if k not in ("digest", "sharding")}
            parent["selected"] = shard["full_selected"]
            parent["digest"] = shard["parent_digest"]
            dp.verify(parent)
            if dp.shard_plan(parent, shard["index"], shard["count"]) != plan:
                raise dp.PlanError("Invalid shard assignment")
            shard_sets.setdefault((plan["environment"], parent["digest"]), []).append(shard)
        expected = {u["key"]: u for u in plan["selected"]}
        metadata_id = ((plan["environment"], plan["sharding"]["parent_digest"])
                       if plan.get("sharding") else (plan["environment"], result["lane"], plan["digest"]))
        if metadata_id not in seen_metadata:
            seen_metadata.add(metadata_id)
            label = re.sub(r"-shard[0-9]+$", "", result["lane"]) if plan.get("sharding") else result["lane"]
            for exclusion in plan.get("inventory", {}).get("exclusions", []):
                exclusions.append({"lane": label, **exclusion})
            if plan.get("architecture_scope"):
                architecture_scopes.append({"lane": label, **plan["architecture_scope"]})
            unknown += plan["unknown_timings"]
            weekly.append({"lane": label, "core": len(plan["core"]),
                           "rotation": [len(day) for day in plan["rotation"]]})
        records = result["records"]
        if Counter(r["key"] for r in records) != Counter(expected.keys()):
            raise dp.PlanError("Missing, duplicate or unexpected execution result")
        for record in records:
            seconds = record["seconds"]
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or seconds < 0):
                raise dp.PlanError("Invalid elapsed time")
            unit = expected[record["key"]]
            totals[unit["owner"]] += seconds
            if unit.get("mode") == "rotation":
                env = plan["environment"].lower()
                architecture = "x86" if env.endswith("/linux/x86_64") else "arm" if env.endswith(("/linux/aarch64", "/linux/arm64")) else None
                if architecture:
                    expansion_seconds[architecture] += seconds
            slowest.append({"lane": result["lane"], "key": record["key"], "seconds": seconds,
                            "name": unit.get("tests", [{}])[0].get("name", result["lane"]),
                            "mode": unit.get("mode", "core"), "status": record["status"]})
            if record["status"] != "passed":
                failures.append(record["key"] + ": " + record["status"])
        incomplete.extend(result.get("coverage_gaps", []))
    for shards in shard_sets.values():
        count = shards[0]["count"]
        if any(s["count"] != count for s in shards) or Counter(s["index"] for s in shards) != Counter(range(count)):
            raise dp.PlanError("Missing/duplicate/incompatible test shards")
    budgets = dp.policy()["minutes"]
    return {"source_sha": ctx["source_sha"], "date": ctx["logical_date"], "slot": ctx["slot"],
            "seconds": totals, "total_seconds": sum(totals.values()), "budgets_minutes": budgets,
            "expansion_seconds": expansion_seconds,
            "architecture_scopes": architecture_scopes,
            "exclusions": exclusions, "failures": failures, "coverage_gaps": incomplete, "unknown_timings": unknown, "weekly": weekly,
            "slowest": sorted(slowest, key=lambda row: -row["seconds"])[:20],
            "status": "failed" if failures or incomplete else "passed"}


def markdown(report):
    window = dp.policy()["window"]
    lines = ["# Daily regression", "",
             f"Source: `{report['source_sha']}`; date: {report['date']}; slot: {report['slot']}.",
             "", f"Target operating window: **{window['start']}–{window['end']} {window['timezone']} ({window['wall_minutes'] / 60:g} hours)**.",
             f"Execution budget: **{window['execution_minutes'] / 60:g} hours**, with **{window['cleanup_minutes']} minutes** reserved for cleanup and reporting.",
             "Cumulative test execution below is a separate resource-consumption diagnostic.",
             "Compilation, lint, queueing and transfer count toward elapsed time, but not cumulative test consumption.", "",
             "| Area | Legacy reference (min) | Cumulative actual (min) | Diagnostic |",
             "| --- | ---: | ---: | --- |"]
    for key, budget in report["budgets_minutes"].items():
        actual = report["seconds"][key] / 60
        ratio = actual / budget
        state = "🟢" if ratio < .9 else "🟡" if ratio <= 1 else "🔴 above reference"
        lines.append(f"| {key} | {budget} | {actual:.2f} | {state} |")
    lines += ["", f"Cumulative test consumption: **{report['total_seconds'] / 60:.2f} min** (not workflow elapsed time).",
              f"Correctness/completeness: **{report['status']}**.",
              f"Unmeasured configuration groups: **{report['unknown_timings']}**.",
              "Unknown estimates are not evidence that the budget will fit.",
              "", "<details><summary>Failures and coverage gaps</summary>", ""]
    lines.extend("- " + x for x in report["failures"] + report["coverage_gaps"])
    lines += ["", "</details>", "", "### Architecture scope", "",
              "x86 retains complete seven-day rotation. ARM runs per-source core smoke plus "
              "approximately 10% of extensions per week; complete matrices require release validation."]
    expansion = report.get("expansion_seconds", {"x86": 0, "arm": 0})
    total = sum(expansion.values())
    target = dp.load(dp.HERE / "daily_matrix.json")["architecture_policy"]["aspirational_expansion_weights"]
    actual = "unmeasured" if total == 0 else f"{100 * expansion['arm'] / total:.1f}%"
    lines.append(f"Measured ARM share of Linux expansion test time: **{actual}**; "
                 f"aspirational target **{100 * target['arm'] / sum(target.values()):.1f}%**. "
                 "This measures observed test-process consumption, not CPU allocation or core coverage.")
    for scope in report.get("architecture_scopes", []):
        lines.append(f"- {scope['lane']}: weekly extensions {scope['weekly_extension']} / "
                     f"{scope['extension_inventory']}; outside this week "
                     f"{scope['release_only_this_week']}. This is not full ARM coverage.")
    lines += ["", "A single day does not prove full weekly or release coverage.", ""]
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
    elapsed = report.get("elapsed_seconds")
    lines += ["", "### Workflow elapsed time", "",
              "Not measured locally." if elapsed is None else f"This attempt: **{elapsed / 60:.2f} / {window['wall_minutes']} minutes**.",
              "", "<details><summary>Job timings (setup/build/test/transfer included)</summary>", "",
              "| Job | Elapsed min | Result |", "| --- | ---: | --- |"]
    import datetime as dt
    for job in report.get("job_metrics", []):
        if job.get("started_at") and job.get("completed_at"):
            seconds = (dt.datetime.fromisoformat(job["completed_at"].replace("Z", "+00:00")) -
                       dt.datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))).total_seconds()
            name = job["name"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {name} | {max(0, seconds) / 60:.2f} | {job.get('conclusion')} |")
    lines += ["", "Queue time is not inferred from dependency waiting; job elapsed values are not summed as workflow time.",
              "", "</details>"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="*")
    p.add_argument("--collect")
    p.add_argument("--expected-file")
    p.add_argument("--metrics")
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
            actual = Counter(r["lane"] for r in results)
            expected = Counter(args.expected_lanes)
            raise dp.PlanError("Selected workflow lanes did not all report; missing=" +
                               str(list((expected - actual).elements())) + "; unexpected=" +
                               str(list((actual - expected).elements())))
        report = summarize(results)
        needs = __import__("json").loads(os.environ.get("NEEDS_JSON", "{}"))
        for name, job in needs.items():
            if job.get("result") != "success":
                report["failures"].append("Workflow dependency " + name + ": " + str(job.get("result")))
                report["status"] = "failed"
        if args.metrics:
            metrics = dp.load(args.metrics)
            import datetime as dt
            start = dt.datetime.fromisoformat(metrics["started_at"].replace("Z", "+00:00"))
            end = dt.datetime.fromisoformat(metrics["observed_at"].replace("Z", "+00:00"))
            elapsed = (end - start).total_seconds()
            if start.tzinfo is None or end.tzinfo is None or elapsed < 0:
                raise dp.PlanError("Invalid elapsed workflow clock")
            report["elapsed_seconds"] = elapsed
            report["job_metrics"] = metrics["jobs"]
            if elapsed > dp.policy()["window"]["wall_minutes"] * 60:
                report["failures"].append("Workflow exceeded ten-hour window")
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
            print(f"::warning::Daily {b} exceeded its legacy cumulative reference")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (dp.PlanError, OSError, KeyError, ValueError) as error:
        print("Daily report failed: " + str(error), file=sys.stderr)
        sys.exit(1)
