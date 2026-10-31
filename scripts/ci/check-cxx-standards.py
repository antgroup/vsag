#!/usr/bin/env python3
"""Audit actual Full VSAG compile commands, including optional CUDA/Node targets."""
import json
from pathlib import Path
import shlex
import sys


def main():
    build = Path(sys.argv[1]).resolve()
    root = Path(__file__).resolve().parents[2]
    counts = {"cpu": 0, "cuda": 0, "node": 0}
    for entry in json.loads((build / "compile_commands.json").read_text()):
        source = Path(entry["file"])
        if not source.is_absolute():
            source = Path(entry["directory"]) / source
        source = source.resolve()
        relative = source.relative_to(root) if root in source.parents else None
        if relative is None:
            continue
        if relative.parts[0] == "src":
            kind = "cuda" if source.suffix == ".cu" else "cpu"
        elif relative.parts[0] == "node_bindings":
            kind = "node"
        else:
            continue
        if source.suffix not in {".cpp", ".cu"}:
            continue
        arguments = entry.get("arguments") or shlex.split(entry.get("command") or "")
        standards = [arg.split("=", 1)[1] for arg in arguments if arg.startswith("-std=")]
        expected = "20" if kind == "cpu" else "17"
        accepted = {"c++" + expected, "gnu++" + expected}
        # GCC 9 names its C++20 draft mode c++2a; CMake's CXX_STANDARD is still 20.
        if kind == "cpu":
            accepted.update({"c++2a", "gnu++2a"})
        else:
            accepted.update({"c++1z", "gnu++1z"})
        if not standards or standards[-1] not in accepted:
            raise SystemExit(f"{source}: expected C++{expected}, found {standards}")
        counts[kind] += 1
    if not counts["cpu"]:
        raise SystemExit("No internal CPU compilation found")
    print(f"Compile standard audit passed: {counts}")


if __name__ == "__main__":
    main()
