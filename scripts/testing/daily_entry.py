#!/usr/bin/env python3
"""Opt-in bridge for existing test scripts; absent Daily context leaves callers unchanged."""
import os
import platform
from pathlib import Path
import subprocess
import sys

def command(binary, profile=None):
    profile = profile or ("unit" if binary.startswith("unittests") else "functional")
    if os.environ.get("VSAG_DAILY_COVERAGE") == "1":
        profile = "unit-coverage"
    lane = os.environ["VSAG_DAILY_LANE"] + "-" + binary
    args = [sys.executable, str(Path(__file__).with_name("daily_run.py")),
            "--context", os.environ["VSAG_DAILY_CONTEXT"], "--binary", "build/tests/" + binary,
            "--lane", lane, "--environment", os.environ["VSAG_DAILY_ENVIRONMENT"] + "/" + platform.system() + "/" + platform.machine(),
            "--profile", profile, "--output", "daily-results/" + lane,
            "--workers", os.environ.get("VSAG_DAILY_WORKERS", "2")]
    if os.environ.get("VSAG_DAILY_CORE_ONLY") == "1":
        args.append("--core-only")
    return args

if __name__ == "__main__":
    if sys.argv[1] == "coverage":
        codes = [subprocess.call(command(binary, "unit-coverage")) for binary in ("unittests", "functests")]
        sys.exit(1 if any(codes) else 0)
    os.execv(sys.executable, command(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
