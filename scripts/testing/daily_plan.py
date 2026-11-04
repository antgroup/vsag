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
        raise PlanError("Invalid legacy diagnostic allocation")
    w = p["window"]
    if (w["wall_minutes"] != 600 or w["execution_minutes"] + w["cleanup_minutes"] != 600
            or w["execution_minutes"] <= 0 or w["cleanup_minutes"] < 1
            or type(w["job_execution_minutes"]) is not int
            or not 0 < w["job_execution_minutes"] < 300):
        raise PlanError("Invalid ten-hour execution/cleanup window")
    if type(p["x86_shards"]) is not int or not 1 <= p["x86_shards"] <= 32:
        raise PlanError("Invalid x86 shard count")
    if p["timezone"] != "Asia/Shanghai" or w["timezone"] != p["timezone"]:
        raise PlanError("Daily business timezone must be Asia/Shanghai")
    if any(not isinstance(w.get(k), str) or not re.fullmatch(r"[0-9]{2}:[0-9]{2}", w[k])
           for k in ("start", "end")):
        raise PlanError("Window clock must use HH:MM")
    try:
        start, end = (dt.time.fromisoformat(w[k]) for k in ("start", "end"))
    except ValueError as error:
        raise PlanError("Invalid window clock") from error
    if ((end.hour * 60 + end.minute - start.hour * 60 - start.minute) % 1440) != w["wall_minutes"]:
        raise PlanError("Window clocks disagree with elapsed duration")
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
    for u in sorted(units, key=lambda u: (-(u.get("order_seconds") or u["estimate_seconds"] or 60.0), u["key"])):
        b = u["owner"]
        slot = min(range(7), key=lambda i: (loads[i][b], i))
        slots[slot].append(u)
        loads[slot][b] += u.get("order_seconds") or u["estimate_seconds"] or 60.0
    return slots


def build_plan(ctx, environment, core, extension, inventory=None, core_only=False, full_coverage=False):
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
        names = {t["name"] for t in u.get("tests", [])}
        observations = [s["seconds"] for s in history["samples"]
                        if s.get("case") in names and s["environment"] == environment
                        and s["status"] in ("passed", "timeout")]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in observations):
            raise PlanError("Invalid scheduling observation")
        # Parent-group/timeout observations order work, never certify a new tuple's estimate.
        u["order_seconds"] = max([u["estimate_seconds"] or 60.0] + observations)
    architecture_scope = None
    if not full_coverage and environment.lower().endswith(("/linux/aarch64", "/linux/arm64")):
        # Keep one stable core smoke per source file/owner. Other existing core
        # cases join the extension inventory rather than disappearing silently.
        smoke, rest, seen = [], [], set()
        for unit in sorted(core, key=lambda u: (
                not any("[build]" in t.get("tags", "").lower() for t in u["tests"]), u["key"])):
            family = (unit["owner"], tuple(sorted({t["source_file"] for t in unit["tests"]})))
            protected = any("[regression]" in t.get("tags", "").lower() or
                            t["name"].startswith("gh#") for t in unit["tests"])
            if family in seen and not protected:
                rest.append({**unit, "mode": "rotation"})
            else:
                seen.add(family)
                smoke.append(unit)
        core = smoke
        extension = rest + extension
        # A fixed weekly bucket gives approximately one tenth of the extension
        # inventory to ARM. Full matrices remain a release validation obligation.
        date = dt.date.fromisoformat(ctx["logical_date"])
        monday = date - dt.timedelta(days=date.weekday())
        strata = load(HERE / "daily_matrix.json")["architecture_policy"]["arm_extension_strata"]
        if type(strata) is not int or not 1 <= strata <= 70:
            raise PlanError("Invalid ARM extension strata")
        bucket = (monday.toordinal() // 7) % strata
        eligible = ([] if core_only else [u for u in extension
                    if int(digest(u["key"])[:8], 16) % strata == bucket])
        architecture_scope = {"architecture": "arm", "extension_fraction": f"1/{strata}",
                              "extension_inventory": len(extension),
                              "weekly_extension": len(eligible),
                              "release_only_this_week": len(extension) - len(eligible)}
        extension = eligible
    week = partition([] if core_only else extension)
    return sealed({"schema_version": 1, "kind": "daily-plan", "context": ctx,
                   "environment": environment, "architecture_scope": architecture_scope,
                   "core": core, "rotation": week,
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


def shard_plan(plan, index, count):
    verify(plan)
    if type(index) is not int or type(count) is not int or not 1 <= count <= 32 or not 0 <= index < count:
        raise PlanError("Invalid shard index/count")
    bins = [[] for _ in range(count)]
    loads = [0.0] * count
    for unit in sorted(plan["selected"], key=lambda u: (-u.get("order_seconds", 60.0), u["key"])):
        target = min(range(count), key=lambda i: (loads[i], i))
        bins[target].append(unit)
        loads[target] += unit.get("order_seconds", 60.0)
    return sealed({**{k: v for k, v in plan.items() if k != "digest"},
                   "selected": bins[index],
                   "sharding": {"index": index, "count": count, "parent_digest": plan["digest"],
                                "full_selected": plan["selected"]}})


def execution_deadline(ctx, window=None):
    now = dt.datetime.now(dt.timezone.utc)
    if window is not None:
        verify(window)
        if window.get("kind") != "daily-window" or window.get("source_sha") != ctx["source_sha"]:
            raise PlanError("Execution window/source mismatch")
        import os
        if os.environ.get("GITHUB_RUN_ATTEMPT") and window.get("attempt") != int(os.environ["GITHUB_RUN_ATTEMPT"]):
            raise PlanError("Stale execution window from an earlier attempt")
        if not isinstance(window.get("started_at"), str) or not isinstance(window.get("job_started_at"), str):
            raise PlanError("Missing attempt/job start clock")
        started = dt.datetime.fromisoformat(window["started_at"].replace("Z", "+00:00"))
        job_start = dt.datetime.fromisoformat(window["job_started_at"].replace("Z", "+00:00"))
        if started.tzinfo is None or job_start.tzinfo is None or job_start < started:
            raise PlanError("Invalid attempt/job start clock")
    else:
        started = job_start = now
    limits = policy()["window"]
    return min(started + dt.timedelta(minutes=limits["execution_minutes"]),
               job_start + dt.timedelta(minutes=limits["job_execution_minutes"]))
