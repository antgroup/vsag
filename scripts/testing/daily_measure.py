#!/usr/bin/env python3
"""Account an indivisible non-Catch test command once, excluding compile/download."""
import argparse
import os
from pathlib import Path
import signal
import sys
import threading

import daily_plan as dp
import team_test_shards as shards


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--owner", choices=sorted(dp.BUCKETS - {"reserve"}), required=True)
    p.add_argument("--cwd", default=".")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout < 1:
        raise dp.PlanError("Expected test command and positive timeout")
    if not os.environ.get("VSAG_DAILY_CONTEXT"):
        os.chdir(args.cwd)
        os.execvp(command[0], command)
    ctx = dp.validate_context(dp.load(os.environ["VSAG_DAILY_CONTEXT"]))
    lane = os.environ["VSAG_DAILY_LANE"] + "-" + args.name
    output = Path("daily-results", lane).resolve()
    if output.exists():
        raise dp.PlanError("Duplicate measurement")
    output.mkdir(parents=True)
    temporary = output / "tmp"
    temporary.mkdir()
    cancel = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: cancel.set())
    unit = {"key": dp.digest([lane, command]), "owner": args.owner, "mode": "core",
            "command": command, "contract": "Full non-Catch command; detailed inventory in command logs"}
    plan = dp.build_plan(ctx, os.environ["VSAG_DAILY_ENVIRONMENT"], [unit], [])
    dp.write(output / "plan.json", plan)
    record = shards.execute(command, Path(args.cwd).resolve(), temporary, output / "output.log", args.timeout, cancel)
    record["key"] = unit["key"]
    dp.write(output / "daily-result.json",
             dp.sealed({"schema_version": 1, "kind": "daily-result", "plan": plan,
                        "lane": lane, "records": [record]}))
    for line in (output / "output.log").read_text(errors="replace").splitlines():
        print("| " + line)
    return 0 if record["status"] == "passed" else 1

if __name__ == "__main__":
    sys.exit(main())
