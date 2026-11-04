#!/usr/bin/env python3
"""Conservative main-push selection; compare the entire before..after range."""
import json
import os
from pathlib import Path
import re
import subprocess


def classify(paths):
    if any("\n" in p or "\r" in p for p in paths):
        return {"heavy": "true", "lint": "full", "pattern": ""}
    docs = lambda p: p.startswith("docs/") or p.lower().endswith(".md")
    changed = [p for p in paths if not docs(p)]
    if not changed:
        return {"heavy": "false", "lint": "none", "pattern": ""}
    # Header/build/unknown changes can affect any translation unit.
    def implementation(p):
        return p.startswith(("src/", "tests/", "tools/", "examples/", "python/", "python_bindings/")) and p.endswith((".cpp", ".py"))
    if not all(implementation(p) for p in changed):
        return {"heavy": "true", "lint": "full", "pattern": ""}
    sources = [p for p in changed if p.startswith("src/") and p.endswith(".cpp") and not p.endswith("_test.cpp") and Path(p).is_file()]
    return {"heavy": "true", "lint": "changed" if sources else "none",
            "pattern": "|".join(re.escape(p) + "$" for p in sources)}


def select(base, head):
    try:
        if not all(re.fullmatch(r"[0-9a-f]{40}", v or "") and v != "0" * 40 for v in (base, head)):
            raise ValueError("missing immutable push endpoints")
        paths = subprocess.check_output(["git", "diff", "--name-only", "--no-renames", "-z", base, head, "--"], timeout=60).decode().split("\0")
        return classify([p for p in paths if p])
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"heavy": "true", "lint": "full", "pattern": ""}


def main():
    result = select(os.environ.get("BEFORE_SHA"), os.environ.get("AFTER_SHA"))
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        for key, value in result.items():
            output.write(f"{key}={value}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
