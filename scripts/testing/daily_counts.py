#!/usr/bin/env python3
"""Named-case counts, independent of configuration multiplicity and lane timing."""
from collections import defaultdict


def case_identity(test):
    source = test.get("source_file", "").replace("\\", "/")
    for prefix in ("src/", "tests/"):
        if "/" + prefix in source:
            source = prefix + source.rsplit("/" + prefix, 1)[1]
            break
    return source, test["name"]


def named_status(unit, record, test, status):
    # Catch2 can execute distinct names together (case-insensitive matching).
    # A group failure is not proof that every member failed.
    if len(unit["tests"]) <= 1 or status == "passed":
        return status
    actual = [t for t in record.get("tests", []) if case_identity(t) == case_identity(test)]
    if any(t.get("success") is False for t in actual):
        return "failed"
    if (actual and all(t.get("success") is True and not t.get("skips") for t in actual)
            and not record.get("error") and status in ("failed", "not_run")):
        return "passed"
    return "not_run"


def count_cases(results):
    """Call only after result/plan integrity validation.

    Missing selected executions stay pending; one failed configuration makes the
    named case fail. Non-Catch commands are groups, never invented testcase counts.
    """
    inventory = set()
    selected = defaultdict(dict)
    commands = defaultdict(dict)
    legacy_inventory = False
    for result in results:
        plan = result["plan"]
        meta = plan.get("inventory", {})
        legacy_inventory |= bool(plan.get("selected")) and "named_cases" not in meta and any(
            u.get("tests") for u in plan["selected"])
        inventory.update(case_identity(t) for t in meta.get("named_cases", []))
        inventory.update(case_identity(x["test"]) for x in meta.get("exclusions", []))
        universe = plan["core"] + [u for day in plan["rotation"] for u in day]
        for unit in universe:
            inventory.update(case_identity(t) for t in unit.get("tests", []))
        # Sibling shards carry the full selection. Missing siblings must not turn
        # partial case coverage into a passing named case.
        units = plan.get("sharding", {}).get("full_selected", plan["selected"])
        lane = result["lane"]
        if plan.get("sharding"):
            scope = (plan["environment"], plan["sharding"]["parent_digest"])
        else:
            scope = (plan["environment"], lane)
        records = {r["key"]: r for r in result["records"]}
        owned = {u["key"] for u in plan["selected"]}
        for unit in units:
            execution = (*scope, unit["key"])
            record = records.get(unit["key"], {}) if unit["key"] in owned else {}
            status = record.get("status", "not_run")
            if status not in ("passed", "failed", "timeout", "error"):
                status = "not_run"
            if unit.get("tests"):
                for test in unit["tests"]:
                    identity = case_identity(test)
                    inventory.add(identity)
                    states = selected[identity]
                    test_status = named_status(unit, record, test, status)
                    if execution not in states or test_status != "not_run":
                        states[execution] = test_status
            else:
                states = commands[lane]
                states[execution] = status
    passed = failed = 0
    for states in selected.values():
        if any(s in ("failed", "timeout", "error") for s in states.values()):
            failed += 1
        elif states and all(s == "passed" for s in states.values()):
            passed += 1
    return {"passed": passed, "failed": failed, "selected": len(selected),
            "global": len(inventory), "pending": len(selected) - passed - failed,
            "legacy_inventory": legacy_inventory,
            "command_groups": {name: ("passed" if all(s == "passed" for s in states.values())
                                       else "not_passed") for name, states in sorted(commands.items())}}


def markdown(counts):
    if counts is None:
        return ["| Passed cases | Failed cases | Selected cases this run | Global discovered cases |",
                "| ---: | ---: | ---: | ---: |", "| — | — | — | — |", "",
                "No trustworthy case counts are available; missing evidence is not zero."]
    lines = ["| Passed cases | Failed cases | Selected cases this run | Global discovered cases |",
             "| ---: | ---: | ---: | ---: |",
             "| {passed} | {failed} | {selected} | {global} |".format(**counts), "",
             "Named cases are deduplicated by repository-relative source file and exact name, across configurations and platforms.",
             "Any failed/timed-out single-case execution fails the case; grouped executions use individual results and remain pending when attribution is unavailable. All selected executions must pass to count a case as passed.",
             f"Pending/not executed cases: **{counts['pending']}**. Global inventory is the active named inventory discovered by this source's test binaries, not assertion or parameter counts."]
    if counts["legacy_inventory"]:
        lines += ["Legacy artifacts have no complete pre-selection inventory: the global count is a known lower bound."]
    if counts["command_groups"]:
        lines += ["", "<details><summary>Non-Catch command groups (excluded from case counts)</summary>", ""]
        lines += [f"- {name}: {state}" for name, state in counts["command_groups"].items()]
        lines += ["", "Examples, Python and compatibility commands may contain multiple checks; a group is not counted as one testcase.", "", "</details>"]
    return lines
