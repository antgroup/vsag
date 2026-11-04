#!/usr/bin/env python3
"""Run an explicit smoke inventory; missing/empty/skipped cases are failures."""
import json
import re
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def verify_report(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    if not cases or any(any(c.find(tag) is not None for tag in ("skipped", "failure", "error")) for c in cases):
        raise ValueError("Smoke tests must execute and pass; empty/skipped reports are invalid")


def full_examples(directory):
    selected = (directory / "ci-example-targets.txt").read_text().splitlines()
    if not selected or len(selected) != len(set(selected)) or any(
            not re.fullmatch(r"[0-9]{3}_[A-Za-z0-9_]+", name) for name in selected):
        raise ValueError("Invalid expected example target manifest")
    # Check every expected binary before executing the first one.
    for name in selected:
        binary = directory / name
        if not binary.is_file() or binary.stat().st_size == 0:
            raise ValueError(f"Missing expected example: {name}")
    return selected


def emit_diagnostics(*paths):
    for path in paths:
        if path.is_file():
            print(f"--- {path} (last 64 KiB) ---", flush=True)
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 65536))
                # Prefix lines so test output cannot become Actions commands.
                for line in stream.read().decode(errors="replace").splitlines():
                    print("| " + line, flush=True)


def run(kind):
    inventory = json.loads(Path(__file__).with_suffix(".json").read_text())
    examples = kind in ("examples", "all-examples")
    selected = full_examples(Path("build/examples/cpp")) if kind == "all-examples" else inventory[kind]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Empty or duplicated smoke inventory")
    logs = Path("log") / ("smoke-" + kind)
    logs.mkdir(parents=True, exist_ok=True)
    for number, case in enumerate(selected):
        binary = Path("build/examples/cpp") / case if examples else Path("build/tests") / kind
        if not binary.is_file() or binary.stat().st_size == 0:
            raise ValueError(f"Missing smoke binary: {binary}")
        binary.chmod(binary.stat().st_mode | 0o100)
        command = [str(binary.resolve())]
        report = logs / f"{number}.xml"
        if not examples:
            command += [case, "--rng-seed", "424242", "--reporter", "junit", "--out", str(report.resolve())]
        print("Smoke:", case, flush=True)
        logfile = logs / f"{number}.log"
        try:
            with logfile.open("w") as output:
                subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=1800 if kind == "all-examples" else 300)
            if not examples:
                verify_report(report)
        except (OSError, ValueError, ET.ParseError, subprocess.SubprocessError):
            emit_diagnostics(logfile, report)
            raise


if __name__ == "__main__":
    run(sys.argv[1])
