#!/usr/bin/env python3
"""Budgeted, separately accounted x86 top-up after mandatory Daily checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import math
import os
from pathlib import Path
import signal
import threading

import daily_plan as dp
import daily_run
import daily_report


def timestamp(value):
    date = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise dp.PlanError("Extra coverage clock requires a timezone")
    return date


def propose(results, started_at, now, attempt, run_id):
    # Require complete base evidence. Only the shared rotation-timeout policy
    # can allow an unvalidated configuration without blocking supplemental work.
    checked = daily_report.summarize(results)
    if checked["status"] not in ("passed", "passed_with_warnings"):
        raise dp.PlanError("Mandatory functional shards did not pass")
    first = results[0]["plan"]
    shard = first["sharding"]
    if (len(results) != dp.policy()["x86_shards"]
            or any(r["plan"]["environment"] != dp.X86_ASAN_ENVIRONMENT for r in results)
            or any(r["plan"]["sharding"]["parent_digest"] != shard["parent_digest"] for r in results)):
        raise dp.PlanError("Expected one complete x86 functional plan")
    parent = {k: v for k, v in first.items() if k not in ("digest", "sharding")}
    parent["selected"] = shard["full_selected"]
    parent["digest"] = shard["parent_digest"]
    dp.verify(parent)
    used = {u["key"] for u in parent["selected"]}
    observations = {}
    for result in results:
        units = {u["key"]: u for u in result["plan"]["selected"]}
        for record in result["records"]:
            names = tuple(t["name"] for t in units[record["key"]]["tests"])
            observations[names] = max(observations.get(names, 0), record["seconds"])
    candidates = []
    for offset in range(1, 7):
        for u in parent["rotation"][(parent["context"]["slot"] + offset) % 7]:
            if u["key"] in used:
                raise dp.PlanError("Extra rotation overlaps today's mandatory plan")
            # Same-case measured costs guide selection, not a runtime guarantee.
            names = tuple(t["name"] for t in u["tests"])
            observed = observations.get(names)
            seconds = max(60, observed * 1.5) if observed is not None else 1800
            candidates.append({**u, "extra_estimate_seconds": seconds,
                               "order_seconds": seconds, "extra_day_offset": offset})
    if len({u["key"] for u in candidates}) != len(candidates):
        raise dp.PlanError("Duplicate extra candidate")
    candidate_plan = dp.sealed({**{k: v for k, v in parent.items() if k != "digest"},
                               "selected": candidates})
    count = dp.policy()["x86_shards"]
    bins = [dp.shard_plan(candidate_plan, i, count)["selected"] for i in range(count)]
    # 9 hours is a target cutoff, leaving an hour inside the 10-hour envelope.
    # Jobs also retain their existing 270-minute limit; extra runs use <=240.
    start = timestamp(started_at)
    if now < start:
        raise dp.PlanError("Extra stage precedes attempt start")
    cutoff = min(start + dt.timedelta(hours=9), now + dt.timedelta(minutes=240))
    budget = max(0, (cutoff - now).total_seconds())
    return dp.sealed({"schema_version": 1, "kind": "daily-extra-proposal",
                      "context": parent["context"], "environment": parent["environment"],
                      "inventory": parent["inventory"], "source_plan_digest": parent["digest"],
                      "run_id": str(run_id), "attempt": int(attempt),
                      "planned_at": now.isoformat(), "cutoff": cutoff.isoformat(),
                      "budget_seconds": budget, "workers_per_shard": 2, "shards": bins,
                      "mandatory_keys": sorted(used)})


def validate(proposal):
    dp.verify(proposal)
    dp.validate_context(proposal["context"])
    if proposal.get("kind") != "daily-extra-proposal":
        raise dp.PlanError("Wrong extra proposal kind")
    if len(proposal["shards"]) != dp.policy()["x86_shards"]:
        raise dp.PlanError("Wrong extra shard count")
    if proposal.get("workers_per_shard") != 2 or proposal.get("environment") != dp.X86_ASAN_ENVIRONMENT:
        raise dp.PlanError("Unexpected supplemental execution resources")
    if not 0 <= proposal["budget_seconds"] <= 240 * 60:
        raise dp.PlanError("Extra budget exceeds phase limit")
    for units in proposal["shards"]:
        for unit in units:
            cost = unit.get("extra_estimate_seconds")
            if type(cost) not in (int, float) or not math.isfinite(cost) or cost <= 0:
                raise dp.PlanError("Invalid extra timing estimate")
    keys = [u["key"] for units in proposal["shards"] for u in units]
    if len(set(keys)) != len(keys) or set(keys) & set(proposal["mandatory_keys"]):
        raise dp.PlanError("Extra candidates overlap")
    quarantines = {r["key"] for r in dp.load(dp.HERE / "daily_matrix.json").get("quarantine", [])}
    if set(keys) & quarantines:
        raise dp.PlanError("Extra stage cannot restore quarantined configurations")
    return proposal


def execute(proposal, index, output, now=None, run_unit=daily_run.run_unit, enabled=True):
    validate(proposal)
    if type(index) is not int or not 0 <= index < len(proposal["shards"]):
        raise dp.PlanError("Invalid extra shard")
    started = now or dt.datetime.now(dt.timezone.utc)
    deadline = min(timestamp(proposal["cutoff"]), started + dt.timedelta(minutes=240))
    candidates = list(proposal["shards"][index])
    # Keep next rotation days ahead of later days. Interleave index families to
    # avoid spending all optional capacity on the largest HGraph family.
    buckets = {}
    for u in sorted(candidates, key=lambda u: (u["extra_day_offset"], -u["order_seconds"], u["key"])):
        buckets.setdefault(u["owner"], []).append(u)
    pending = []
    priority = ["hgraph", "pyramid", "sindi", "concurrent", "other"]
    owners = priority + sorted(set(buckets) - set(priority))
    while any(buckets.values()):
        for owner in owners:
            if buckets.get(owner):
                pending.append(buckets[owner].pop(0))
    lock, cancelled = threading.Lock(), threading.Event()
    selected, records = [], []
    # Keep time for uploads. Never count budget-deferred candidates as passed.
    def worker():
        while not cancelled.is_set():
            with lock:
                remaining = (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds() - 300
                position = next((i for i, u in enumerate(pending)
                                 if enabled and u["extra_estimate_seconds"] <= remaining), None)
                if position is None:
                    return
                unit = pending.pop(position)
                number = len(selected)
                selected.append(unit)
            record = run_unit(unit, number, output, dp.policy()["case_timeout_seconds"], cancelled, deadline)
            with lock:
                records.append(record)
    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda *_: cancelled.set())
    try:
        with ThreadPoolExecutor(max_workers=proposal["workers_per_shard"]) as pool:
            list(pool.map(lambda _: worker(), range(proposal["workers_per_shard"])))
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    metadata = {"proposal_digest": proposal["digest"], "shard": index,
                "candidate_keys": [u["key"] for u in candidates],
                "deferred_keys": [u["key"] for u in pending],
                "budget_seconds": proposal["budget_seconds"],
                "started_at": started.isoformat(), "cutoff": deadline.isoformat(),
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "workers": proposal["workers_per_shard"], "cancelled": cancelled.is_set()}
    plan = dp.sealed({"schema_version": 1, "kind": "daily-plan", "context": proposal["context"],
                      "environment": proposal["environment"], "architecture_scope": None,
                      "quarantines": [], "core": [], "rotation": [[] for _ in range(7)],
                      "inventory": proposal["inventory"], "selected": selected,
                      "unknown_timings": 0, "extra": metadata})
    return dp.sealed({"schema_version": 1, "kind": "daily-result", "plan": plan,
                      "lane": f"daily-extra-x86-functests-shard{index}", "records": records})


def reconcile(proposal, results):
    validate(proposal)
    found = set()
    for r in results:
        dp.verify(r)
        plan = dp.verify(r["plan"])
        extra = plan.get("extra")
        if extra is None:
            continue
        i = extra["shard"]
        if type(i) is not int or not 0 <= i < len(proposal["shards"]) or i in found:
            raise dp.PlanError("Duplicate or invalid extra shard")
        found.add(i)
        if (extra["proposal_digest"] != proposal["digest"] or plan["context"] != proposal["context"]
                or plan["environment"] != proposal["environment"]):
            raise dp.PlanError("Extra result/proposal mismatch")
        assigned = {u["key"]: u for u in proposal["shards"][i]}
        selected = [u["key"] for u in plan["selected"]]
        deferred = extra["deferred_keys"]
        if (set(extra["candidate_keys"]) != set(assigned)
                or len(selected + deferred) != len(set(selected + deferred))
                or set(selected + deferred) != set(assigned)
                or any(u != assigned[u["key"]] for u in plan["selected"])):
            raise dp.PlanError("Extra selected/deferred inventory mismatch")
        if extra["cancelled"]:
            raise dp.PlanError("Extra stage was cancelled")
    if found != set(range(len(proposal["shards"]))):
        raise dp.PlanError("Missing extra shard evidence")


def accounting(proposal, results, jobs):
    extra = [r for r in results if r["plan"].get("extra")]
    records = [record for r in extra for record in r["records"]]
    workers = [j for j in jobs if j["name"].startswith("X86 Extra Coverage shard ")]
    runner_seconds = sum(max(0, (timestamp(j["completed_at"]) - timestamp(j["started_at"])).total_seconds())
                         for j in workers if j.get("completed_at") and j.get("started_at"))
    end = max([timestamp(r["plan"]["extra"]["finished_at"]) for r in extra]
              + [timestamp(j["completed_at"]) for j in workers if j.get("completed_at")])
    return {"budget_seconds": proposal["budget_seconds"],
            "wall_seconds": max(0, (end - timestamp(proposal["planned_at"])).total_seconds()),
            "runner_seconds": runner_seconds, "test_seconds": sum(r["seconds"] for r in records),
            "selected": sum(len(r["plan"]["selected"]) for r in extra),
            "passed": sum(r["status"] == "passed" for r in records),
            "failed": sum(r["status"] != "passed" for r in records),
            "deferred": sum(len(r["plan"]["extra"]["deferred_keys"]) for r in extra)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["plan", "run"])
    parser.add_argument("--collect")
    parser.add_argument("--clock")
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.mode == "plan":
        results = [dp.load(p) for p in Path(args.collect).rglob("daily-result.json")]
        clock = dp.load(args.clock)
        proposal = propose(results, clock["started_at"], dt.datetime.now(dt.timezone.utc),
                           os.environ["GITHUB_RUN_ATTEMPT"], os.environ["GITHUB_RUN_ID"])
        validate(proposal)
        dp.write(args.proposal, proposal)
    else:
        proposal = validate(dp.load(args.proposal))
        if (proposal["context"]["source_sha"] != os.environ["GITHUB_SHA"]
                or proposal["run_id"] != os.environ["GITHUB_RUN_ID"]
                or proposal["attempt"] != int(os.environ["GITHUB_RUN_ATTEMPT"])):
            raise dp.PlanError("Stale extra run/source/attempt")
        binary = Path("build/tests/functests").resolve()
        enabled = os.environ.get("EXTRA_AVAILABLE", "true") == "true"
        if enabled and not binary.is_file():
            raise dp.PlanError("Extra test binary is missing")
        if enabled and any(Path(u["binary"]) != binary for u in proposal["shards"][args.shard]):
            raise dp.PlanError("Extra binary differs from the approved build path")
        if enabled:
            binary.chmod(binary.stat().st_mode | 0o100)
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        result = execute(proposal, args.shard, output, enabled=enabled)
        dp.write(output / "daily-result.json", result)
        return 0 if not result["plan"]["extra"]["cancelled"] and all(r["status"] == "passed" for r in result["records"]) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
