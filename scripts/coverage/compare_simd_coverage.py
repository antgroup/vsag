#!/usr/bin/env python3
"""Compare like-for-like SIMD traces; never treat them as full-suite coverage."""

import argparse
from fractions import Fraction
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx"}
PROFILE = "simd-v1:seed=424242:filter=[simd]~[!benchmark]:order=lex"
# gcov emits line records only for sources that define something. A changed file
# holding nothing but declarations (prototypes, aliases, macros) can never show
# up in a trace, so it must not be reported as an unmeasured changed file. The
# check errs towards "defines code", which keeps the gate fail-closed.
HEADER_SUFFIXES = {".h", ".hpp", ".hxx"}
FUNCTION_BODY = re.compile(r"\)[^;{}()]*\{")
# Steps that implement one instruction set can only produce covered lines on a
# runner whose CPU actually exposes that set, so a runner that cannot execute
# them must not be asked to cover them. The measurement log prints the decision
# for every set as "cpu <isa> >> dist_support:Y + platform:Y = using:Y".
RUNNER_ISA_PATTERN = re.compile(
    r"cpu\s+([a-z0-9]+)\s*>>\s*dist_support:([YN])\s*\+\s*platform:([YN])\s*=\s*using:([YN])")
ISA_IMPLEMENTATIONS = {
    "avx": {"src/simd/avx.cpp"},
    "avx2": {"src/simd/avx2.cpp"},
    "avx512f": {"src/simd/avx512.cpp"},
    "avx512vpopcntdq": {"src/simd/avx512vpopcntdq.cpp"},
    "neon": {"src/simd/neon.cpp"},
    "sve": {"src/simd/sve.cpp"},
}
# Files that need more than the single set they are listed under above.
ISA_REQUIREMENTS = {
    "src/simd/avx512vpopcntdq.cpp": ("avx512f", "avx512vpopcntdq"),
}


def production(path):
    return (PurePosixPath(path).suffix in SUFFIXES
            and path.startswith(("src/", "include/"))
            and not path.endswith("_test.cpp")
            and path not in {"src/version.h", "include/vsag/expected.hpp"})


def scoped(path):
    return production(path) and path.startswith("src/simd/")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def changed_lines(root, base, head):
    # Disable rename detection: new destinations must earn their own coverage.
    paths = git(root, "diff", "--no-renames", "--name-only", "-z", base, head).decode().split("\0")
    result = {}
    for path in filter(production, paths):
        patch = git(root, "diff", "--no-ext-diff", "--no-renames", "--unified=0",
                    base, head, "--", path).decode()
        added = set()
        for start, count in re.findall(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", patch, re.M):
            added.update(range(int(start), int(start) + int(count or 1)))
        result[path] = added
    return result


def defines_executable_code(text):
    """Return True when the source text can produce gcov line records."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    text = re.sub(r"\\\n", " ", text)
    text = re.sub(r"(?m)^[ \t]*#.*$", " ", text)
    return FUNCTION_BODY.search(text) is not None


def unavailable_isa_files(log):
    """Changed steps whose instruction set the measured runner does not enable.

    Only a set the log explicitly reports as unusable is dropped; a set that is
    missing from the log keeps the previous, stricter behaviour, so a truncated
    or fixture log can never weaken the gate silently.
    """
    if not log.is_file():
        return set()
    enabled = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RUNNER_ISA_PATTERN.search(line)
        if match:
            enabled[match.group(1)] = match.group(4) == "Y"
    result = set()
    for isa, paths in ISA_IMPLEMENTATIONS.items():
        for path in paths:
            required = ISA_REQUIREMENTS.get(path, (isa,))
            if not all(enabled.get(key, True) for key in required):
                result.add(path)
    return result


def unmeasurable_changes(root, changes, head):
    """Changed scoped headers that define no code and therefore cannot be measured.

    Only headers qualify: an unmeasured changed translation unit is always a real
    failure, because it means the file was not compiled into the measured build.
    """
    result = set()
    for path, added in changes.items():
        if not scoped(path) or not added or path in head:
            continue
        if PurePosixPath(path).suffix not in HEADER_SUFFIXES:
            continue
        source = root / path
        if not source.is_file():
            continue
        if not defines_executable_code(source.read_text(encoding="utf-8", errors="replace")):
            result.add(path)
    return result


def read_trace(path):
    files = {}
    source = None
    for line in path.read_text().splitlines():
        if line.startswith("SF:"):
            source = line[3:]
            if PurePosixPath(source).is_absolute() or ".." in PurePosixPath(source).parts:
                raise ValueError("trace contains unsafe source paths")
        elif line.startswith("DA:"):
            if source is None:
                raise ValueError("line record without source")
            number, hits, *_ = line[3:].split(",")
            number, hits = int(number), int(hits)
            if number <= 0 or hits < 0:
                raise ValueError("invalid line record")
            if scoped(source):
                lines = files.setdefault(source, {})
                lines[number] = lines.get(number, False) or hits > 0
        elif line == "end_of_record":
            source = None
    if not files:
        raise ValueError("missing or empty SIMD baseline/head measurement")
    return files


def totals(files, skip=()):
    kept = [(path, lines) for path, lines in files.items() if path not in skip]
    return sum(sum(lines.values()) for _, lines in kept), sum(len(lines) for _, lines in kept)


def compare(base, head, changes, deleted=(), unmeasurable=(), unavailable=()):
    errors = []
    patch = []
    declaration_only = []
    isa_unavailable = []
    for path, added in changes.items():
        if not scoped(path) or not added:
            continue
        if path in unavailable:
            isa_unavailable.append(path)
        elif path in unmeasurable:
            declaration_only.append(path)
        elif path not in head:
            errors.append(f"Unmeasured changed file: {path}")
        else:
            patch.extend(head[path][line] for line in added if line in head[path])
    # A surviving source must not silently disappear from the denominator.
    for path in base.keys() - head.keys():
        if path not in deleted:
            errors.append(f"Baseline source disappeared from head trace: {path}")
    before, before_total = totals(base, unavailable)
    after, after_total = totals(head, unavailable)
    if Fraction(after, after_total) < Fraction(before, before_total):
        errors.append("SIMD scoped line coverage regressed")
    if patch and Fraction(sum(patch), len(patch)) < Fraction(4, 5):
        errors.append("SIMD patch coverage is below 80%")
    return {
        "scope": "SIMD only; not full-project coverage",
        "base": {"covered": before, "measured": before_total},
        "head": {"covered": after, "measured": after_total},
        "patch": {"covered": sum(patch), "measured": len(patch)},
        "declaration_only": sorted(declaration_only),
        "isa_unavailable": sorted(isa_unavailable),
        "unsupported": sorted(path for path in changes if not scoped(path)),
        "new_files": sorted(head.keys() - base.keys()),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    try:
        for role, sha in (("base", args.base), ("head", args.head)):
            metadata = json.loads((args.reports / f"{role}.json").read_text())
            if metadata != {"sha": sha, "profile": PROFILE}:
                raise ValueError(f"{role} SHA/profile mismatch; baseline is not comparable")
        changes = changed_lines(args.root, args.base, args.head)
        head_trace = read_trace(args.reports / "head.info")
        deleted = git(args.root, "diff", "--no-renames", "--diff-filter=D", "--name-only",
                      "-z", args.base, args.head).decode().split("\0")
        result = compare(read_trace(args.reports / "base.info"), head_trace, changes, deleted,
                         unmeasurable_changes(args.root, changes, head_trace),
                         unavailable_isa_files(args.reports / "head.log"))
        result.update(base_sha=args.base, head_sha=args.head, profile=PROFILE)
        (args.reports / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        lines = ["## Pre-merge SIMD coverage", "", f"Base: `{args.base}`; merge candidate: `{args.head}`.",
                 "", "Same runner, build options, test filter and seed. This is not full-project coverage.", ""]
        for role in ("base", "head", "patch"):
            entry = result[role]
            value = (f"{entry['covered']}/{entry['measured']} ({100 * entry['covered'] / entry['measured']:.2f}%)"
                     if entry["measured"] else "N/A: no measured added executable lines")
            lines.append(f"- {role}: {value}")
        lines += ["", "Patch target: 80%; SIMD line coverage must not regress (unrounded comparison).",
                  "Other C++ paths are unsupported by this stage; full-project non-regression remains deferred."]
        if result["isa_unavailable"]:
            lines += ["", "Changed files for instruction sets this runner does not enable",
                      "(excluded from the patch gate and from both totals):"]
            lines += [f"- `{p}`" for p in result["isa_unavailable"]]
        if result["declaration_only"]:
            lines += ["", "Declaration-only changed files (no executable lines; not measurable):"]
            lines += [f"- `{p}`" for p in result["declaration_only"]]
        if result["unsupported"]:
            lines += ["", "Unsupported changed files:"] + [f"- `{p}`" for p in result["unsupported"]]
        if result["new_files"]:
            lines += ["", "New measured files have no per-file baseline; included in head totals and patch gate:"]
            lines += [f"- `{p}`" for p in result["new_files"]]
        lines += ["", *result["errors"]]
        (args.reports / "summary.md").write_text("\n".join(lines) + "\n")
        return bool(result["errors"])
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        (args.reports / "summary.md").write_text(
            f"## Pre-merge SIMD coverage unavailable\n\n{error}\n\nNo passing coverage result is claimed.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
