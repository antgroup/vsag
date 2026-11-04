#!/usr/bin/env python3
"""Versioned daily core/rotation planning. No network or test execution."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
BUCKETS = {"hgraph", "pyramid", "sindi", "other", "concurrent", "simd",
           "unit-coverage", "bindings-examples-compatibility", "reserve"}


class PlanError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sealed(value):
    return {**value, "digest": digest(value)}


def verify(value):
    if value.get("schema_version") != 1 or value.get("digest") != digest(
            {k: v for k, v in value.items() if k != "digest"}):
        raise PlanError("Invalid schema or digest")
    return value


def load(path):
    def reject(value):
        raise PlanError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def policy():
    p = load(HERE / "daily_budget_policy.json")
    if (p.get("schema_version") != 1 or set(p["minutes"]) != BUCKETS
            or any(type(v) is not int or v <= 0 for v in p["minutes"].values())
            or sum(p["minutes"].values()) != 360):
        raise PlanError("Daily budget must have exactly 360 minutes")
    for name in p["core_minutes"]:
        if sum(p[k][name] for k in ("core_minutes", "rotation_minutes", "reserve_minutes")) != p["minutes"][name]:
            raise PlanError("Invalid sub-budget: " + name)
    return p


def context(source_sha, created_at, logical_date=None):
    if not re.fullmatch("[0-9a-f]{40}", source_sha):
        raise PlanError("Expected full source SHA")
    started = dt.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    if started.tzinfo is None:
        raise PlanError("Run creation time must have a timezone")
    day = dt.date.fromisoformat(logical_date) if logical_date else started.astimezone(
        ZoneInfo("Asia/Shanghai")).date()
    matrix = load(HERE / "daily_matrix.json")
    p = policy()
    history = load(HERE / "daily_timing_history.json")
    return sealed({"schema_version": 1, "kind": "daily-context", "source_sha": source_sha,
                   "created_at": created_at, "logical_date": day.isoformat(),
                   "week": "%04d-W%02d" % day.isocalendar()[:2], "slot": day.weekday(),
                   "policy_digest": digest(p), "matrix_digest": digest(matrix),
                   "history_digest": digest(history),
                   "simd_mode": matrix["simd_rotation"][day.weekday()]})


def validate_context(ctx):
    verify(ctx)
    if ctx.get("kind") != "daily-context":
        raise PlanError("Wrong context kind")
    expected = context(ctx["source_sha"], ctx["created_at"], ctx["logical_date"])
    if ctx != expected:
        raise PlanError("Context/checked-out policy mismatch; rerun original source and plan")
    return ctx


def owner(test, profile):
    if profile == "unit":
        return "unit-coverage"
    if profile in ("simd", "concurrent", "unit-coverage", "bindings-examples-compatibility"):
        return profile
    tags = set(re.findall(r"\[([^\]]+)\]", test["tags"].lower()))
    if "concurrent" in tags:
        return "concurrent"
    matrix = load(HERE / "daily_matrix.json")
    found = {value for key, value in matrix["index_tags"].items() if key in tags}
    if len(found) == 1:
        return found.pop()
    if len(found) > 1:
        raise PlanError("Ambiguous index ownership: " + test["name"])
    if test["name"] in matrix["case_owners"]:
        return matrix["case_owners"][test["name"]]
    found = {value for key, value in matrix["support_tags"].items() if key in tags}
    if len(found) == 1:
        return found.pop()
    # Catch2 unit inventory has a separate, explicitly declared binary/profile.
    raise PlanError("Unclassified functional test: " + test["name"] + " " + test["tags"])


def estimate(unit, environment, history):
    samples = [s["seconds"] for s in history["samples"]
               if s["key"] == unit["key"] and s["environment"] == environment
               and s["status"] == "passed"]
    if not samples:
        return None
    if any(isinstance(s, bool) or not isinstance(s, (float, int))
           or not math.isfinite(s) or s <= 0 for s in samples):
        raise PlanError("Invalid measured timing")
    return max(samples) * 1.2


def partition(units):
    """Whole configuration groups, disjoint and exhaustive, stable on unknowns."""
    if len({u["key"] for u in units}) != len(units):
        raise PlanError("Duplicate configuration group")
    slots = [[] for _ in range(7)]
    loads = [{b: 0.0 for b in BUCKETS} for _ in range(7)]
    for u in sorted(units, key=lambda u: (-(u["estimate_seconds"] or 60.0), u["key"])):
        b = u["owner"]
        slot = min(range(7), key=lambda i: (loads[i][b], i))
        slots[slot].append(u)
        loads[slot][b] += u["estimate_seconds"] or 60.0
    return slots


def build_plan(ctx, environment, core, extension, inventory=None):
    validate_context(ctx)
    if not environment or not core:
        raise PlanError("Environment and daily core cannot be empty")
    history = load(HERE / "daily_timing_history.json")
    units = core + extension
    if len({u["key"] for u in units}) != len(units):
        raise PlanError("Core/extension IDs overlap")
    for u in units:
        if u["owner"] not in BUCKETS - {"reserve"}:
            raise PlanError("Invalid owner")
        u["estimate_seconds"] = estimate(u, environment, history)
    week = partition(extension)
    return sealed({"schema_version": 1, "kind": "daily-plan", "context": ctx,
                   "environment": environment, "core": core, "rotation": week,
                   "inventory": inventory or {"exclusions": []},
                   "selected": core + week[ctx["slot"]],
                   "unknown_timings": sum(u["estimate_seconds"] is None for u in units)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-sha", required=True)
    p.add_argument("--created-at", required=True, help="Original workflow run created_at, not rerun time")
    p.add_argument("--logical-date")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    write(args.output, context(args.source_sha, args.created_at, args.logical_date))


if __name__ == "__main__":
    main()
