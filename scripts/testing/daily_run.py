#!/usr/bin/env python3
"""Execute a frozen Daily plan, with exact Catch2 result reconciliation."""
from __future__ import annotations
import argparse
import datetime as dt
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import xml.etree.ElementTree as ET

import daily_parameters as params
import daily_plan as dp
import team_test_shards as shards


def test_source(root, test):
    source = test["source_file"].replace("\\", "/")
    relative = "tests/" + source.rsplit("/tests/", 1)[-1] if "/tests/" in source else ("src/" + source.rsplit("/src/", 1)[-1] if "/src/" in source else source)
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise dp.PlanError("Source for inventory is missing: " + relative)
    return path


def capability_inventory(binary, tests):
    """Exclude only exact tests unsupported by explicit build capabilities.

    Missing build metadata never authorizes an exclusion. Such tests still run,
    and any unexpected runtime SKIP fails reconciliation.
    """
    cache = binary.parent.parent / "CMakeCache.txt"
    flags = {}
    evidence = {"cache_sha256": None, "flags": flags}
    if cache.is_file():
        contents = cache.read_bytes()
        evidence["cache_sha256"] = hashlib.sha256(contents).hexdigest()
        for key, value in re.findall(r"^(ENABLE_ASAN|ENABLE_CUDA):BOOL=(ON|OFF)$",
                                     contents.decode(), re.MULTILINE):
            if key in flags:
                raise dp.PlanError("Duplicate build capability: " + key)
            flags[key] = value == "ON"
    rules = [
        ("Kmeans bounds the resident pass by a named budget only",
         "src/impl/cluster/kmeans_cluster_test.cpp", "ENABLE_CUDA", False,
         "CUDA disabled in this build; GPU resident-budget behavior is not validated"),
        ("DefaultAllocator Reallocate Failure Keeps Tracking",
         "src/impl/allocator/default_allocator_test.cpp", "ENABLE_ASAN", True,
         "ASan aborts allocation-failure probes; this case remains selected in non-ASan builds"),
    ]
    selected, excluded = [], []
    for test in tests:
        source = test["source_file"].replace("\\", "/")
        match = next((r for r in rules if test["name"] == r[0]
                      and (source == r[1] or source.endswith("/" + r[1]))
                      and flags.get(r[2]) is r[3]), None)
        if match:
            excluded.append({"test": test, "reason": match[4],
                             "capability": match[2], "value": match[3]})
        else:
            selected.append(test)
    if {t["name"].lower() for t in selected} & {e["test"]["name"].lower() for e in excluded}:
        raise dp.PlanError("Capability exclusion splits a case-insensitive Catch2 group")
    return selected, {"build_capabilities": evidence, "exclusions": excluded}


def inventory_timeout(deadline, maximum=120):
    if deadline is None:
        return maximum
    remaining = (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()
    if remaining <= 0:
        raise dp.PlanError("Job/attempt execution window exhausted during inventory")
    return min(maximum, remaining)


def inventory_units(binary, root, profile, output, seed, cancelled, deadline=None):
    tests = shards.listing(binary, "~[.]", seed, output / "inventory", inventory_timeout(deadline), cancelled)
    for test in tests:
        test["binary"] = binary.name
        test["id"] = shards.stable_id(binary.name, test)
    tests, inventory = capability_inventory(binary, tests)
    core, extension = [], []
    for group in shards.case_groups(tests):
        inventory_timeout(deadline)
        owners = {dp.owner(t, profile) for t in group}
        if len(owners) != 1:
            raise dp.PlanError("A case-insensitive Catch2 group crosses budget owners")
        base = {"tests": group, "owner": owners.pop(), "binary": str(binary),
                "seed": seed, "env": {}, "contract": "All original assertions and SECTION branches"}
        key = dp.digest([[t["name"], str(test_source(root, t).relative_to(root))]
                         for t in group])
        # Coverage remains exhaustive on its existing supported test set.
        if profile == "unit-coverage":
            if any("[daily]" in t["tags"].lower() for t in group):
                continue
            core.append({**base, "key": key, "mode": "core"})
        elif len(group) == 1 and test_source(root, group[0]).name in ("test_pyramid.cpp", "test_sindi.cpp"):
            test = group[0]
            axes, choices = params.configurations(test_source(root, test).read_text(), test["source_line"])
            for i, choice in enumerate(choices):
                unit = {**base, "key": dp.digest([key, axes, choice]), "axes": axes,
                        "configuration": choice, "mode": "core" if i == 0 else "rotation",
                        "env": {"VSAG_DAILY_CHOICES": json.dumps(params.runtime_choices(axes, choice))}}
                (core if i == 0 else extension).append(unit)
        elif len(group) == 1 and group[0]["name"].startswith("(Daily) HGraph "):
            directory = output / ("enumerate-" + key[:16])
            directory.mkdir(parents=True)
            (directory / "tmp").mkdir()
            manifest = directory / "groups.json"
            command = ["env", "VSAG_DAILY_ENUMERATE=" + str(manifest), str(binary),
                       shards.exact_filter([group[0]["name"]]), "--rng-seed", str(seed)]
            state = shards.execute(command, directory, directory / "tmp",
                                   directory / "enumeration.log", inventory_timeout(deadline), cancelled)
            if state["exit_code"] != 4 or not manifest.is_file():
                raise dp.PlanError("HGraph inventory failed: " + group[0]["name"])
            configurations = dp.load(manifest)
            if not configurations or len({dp.digest(c) for c in configurations}) != len(configurations):
                raise dp.PlanError("Empty/duplicate HGraph configuration inventory")
            for configuration in configurations:
                extension.append({**base, "key": dp.digest([key, configuration]),
                                  "mode": "rotation", "configuration": configuration,
                                  "env": {"VSAG_DAILY_HGRAPH": json.dumps(configuration)}})
        else:
            # Non-adapted tests are indivisible exhaustive groups; never silently sample them.
            target = extension if all("[daily]" in t["tags"].lower() for t in group) else core
            target.append({**base, "key": key, "mode": "rotation" if target is extension else "core"})
    if not tests:
        raise dp.PlanError("Empty binary inventory")
    return core, extension, inventory


def execution_events(logfile, report):
    events = Path(logfile).read_text(errors="replace")
    if Path(report).is_file():
        # Catch2 XML captures streams in OverallResult/StdOut and StdErr.
        tree = ET.parse(report).getroot()
        events += "\n" + "\n".join(node.text or "" for node in tree.iter()
                                  if node.tag in ("StdOut", "StdErr"))
    return events


def run_unit(unit, number, output, timeout, cancelled, deadline=None):
    directory = output / ("case-%05d" % number)
    directory.mkdir()
    cwd, temporary = directory / "cwd", directory / "tmp"
    cwd.mkdir()
    temporary.mkdir()
    binary = Path(unit["binary"])
    record = {"key": unit["key"], "status": "failed", "seconds": 0, "directory": directory.name}
    try:
        listing_timeout = 120
        if deadline is not None:
            remaining = (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()
            if remaining <= 0:
                record.update(status="budget_exhausted", error="Execution window exhausted")
                dp.write(directory / "result.json", record)
                return record
            listing_timeout = min(120, max(1, math.floor(remaining)))
            timeout = min(timeout, max(1, math.floor(remaining)))
        selected = shards.listing(binary, shards.exact_filter([t["name"] for t in unit["tests"]]),
                                  unit["seed"], directory / "selection", listing_timeout, cancelled)
        if Counter(map(shards.case_identity, selected)) != Counter(map(shards.case_identity, unit["tests"])):
            raise dp.PlanError("Exact filter did not select the expected tests")
        if deadline is not None:
            remaining = (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()
            if remaining <= 0:
                raise dp.PlanError("Execution window exhausted during test listing")
            timeout = min(timeout, max(1, math.floor(remaining)))
        report = directory / "report.xml"
        command = ["env", "VSAG_DAILY_EVENTS=1", "VSAG_TEST_SEED=" + str(unit["seed"])]
        command += [k + "=" + v for k, v in sorted(unit["env"].items())]
        command += [str(binary), shards.exact_filter([t["name"] for t in unit["tests"]]),
                    "--reporter", "xml", "--out", str(report), "--durations", "yes",
                    "--order", "lex", "--rng-seed", str(unit["seed"])]
        record.update(shards.execute(command, cwd, temporary, directory / "output.log", timeout, cancelled))
        if record["status"] not in ("timeout", "cancelled", "launch_error"):
            executed, passed, _ = shards.parse_execution_xml(report, unit["tests"])
            record["tests"] = executed
            if any(test["skips"] for test in executed):
                record["status"] = "skipped"
            elif not passed or record["exit_code"] != 0:
                record["status"] = "failed"
            else:
                record["status"] = "passed"
        if record["status"] not in ("timeout", "cancelled", "launch_error"):
            # Log executed generator choices and SECTIONs in the XML, not just top-level names.
            events = execution_events(directory / "output.log", report)
            record["choices_seen"] = sorted(set(line for line in events.splitlines()
                                                 if line.startswith("DAILY_CHOICE ")))
            if unit.get("axes"):
                expected_choices = json.loads(unit["env"]["VSAG_DAILY_CHOICES"])
                if not record["choices_seen"]:
                    raise dp.PlanError("No evidence that the planned generator adapter executed")
                for event in record["choices_seen"]:
                    _, line, index = event.split()
                    if expected_choices.get(line) != int(index):
                        raise dp.PlanError("Executed generator choice differs from plan")
            if "VSAG_DAILY_HGRAPH" in unit["env"]:
                actual = [json.loads(line[len("DAILY_HGRAPH "):]) for line in events.splitlines()
                          if line.startswith("DAILY_HGRAPH ")]
                if not actual or any(value != json.loads(unit["env"]["VSAG_DAILY_HGRAPH"]) for value in actual):
                    raise dp.PlanError("Executed HGraph configuration differs from plan")
    except (dp.PlanError, shards.ShardError, OSError, ValueError, ET.ParseError) as error:
        record["error"] = str(error)
        record["status"] = "failed"
    dp.write(directory / "result.json", record)
    print(json.dumps({"case": unit["tests"][0]["name"], "key": unit["key"],
                      "status": record["status"], "seconds": record["seconds"]}), flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--binary", required=True)
    p.add_argument("--lane", required=True)
    p.add_argument("--environment", required=True, help="Platform, compiler/image, sanitizer and ISA")
    p.add_argument("--profile", choices=["functional", "unit", "unit-coverage", "concurrent", "simd"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--timeout", type=int, default=dp.policy()["case_timeout_seconds"])
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--core-only", action="store_true")
    args = p.parse_args()
    if args.workers < 1 or args.timeout < 1:
        raise dp.PlanError("Workers and timeout must be positive")
    ctx = dp.validate_context(dp.load(args.context))
    root = Path(args.root).resolve()
    binary = Path(args.binary).resolve()
    if not binary.is_file():
        raise dp.PlanError("Test binary is missing")
    binary.chmod(binary.stat().st_mode | 0o100)
    output = Path(args.output).resolve()
    if output.exists():
        raise dp.PlanError("Use a fresh output directory; do not mix attempts")
    output.mkdir(parents=True)
    cancel = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: cancel.set())
    seed = int(dp.digest([ctx["week"], "daily-seed-v1"])[:8], 16) or 1
    window_path = Path(args.context).with_name("window.json")
    deadline = dp.execution_deadline(ctx, dp.load(window_path) if window_path.exists() else None)
    try:
        core, extension, inventory = inventory_units(binary, root, args.profile, output, seed, cancel, deadline)
    except (dp.PlanError, shards.ShardError) as error:
        dp.write(output / "inventory-error.json", {"status": "failed", "error": str(error)})
        raise
    if args.profile == "concurrent":
        core = [u for u in core if any("[concurrent]" in t["tags"].lower() for t in u["tests"])]
        extension = [u for u in extension if any("[concurrent]" in t["tags"].lower() for t in u["tests"])]
    plan = dp.build_plan(ctx, args.environment, core, extension, inventory=inventory, core_only=args.core_only,
                         full_coverage=args.profile == "unit-coverage")
    if args.shard_count != 1 or args.shard_index != 0:
        plan = dp.shard_plan(plan, args.shard_index, args.shard_count)
    dp.write(output / "plan.json", plan)
    if args.plan_only:
        return 0
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(run_unit, u, i, output, args.timeout, cancel, deadline)
                   for i, u in enumerate(plan["selected"])]
        for future in as_completed(pending):
            records.append(future.result())
    result = dp.sealed({"schema_version": 1, "kind": "daily-result", "plan": plan,
                        "lane": args.lane, "records": records})
    dp.write(output / "daily-result.json", result)
    return 0 if all(r["status"] == "passed" for r in records) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (dp.PlanError, shards.ShardError, OSError, ValueError, ET.ParseError) as error:
        print("Daily execution failed: " + str(error), file=sys.stderr)
        sys.exit(1)
