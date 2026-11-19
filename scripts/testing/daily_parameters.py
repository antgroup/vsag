#!/usr/bin/env python3
"""Finite explicit Catch2 generator axes; reject syntax we cannot enumerate.

SECTION branches remain intact. The product is conservative: choices in mutually
exclusive branches can produce duplicate exercised paths, never omitted paths.
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import daily_plan as dp

PARAMETER_SOURCES = ("tests/test_pyramid.cpp", "tests/test_sindi.cpp")

# Ignore comments/literals while locating punctuation, preserving line offsets.
TOKEN = re.compile(r'R"(?P<raw_delim>[^ ()\t\r\n]{0,16})\(.*?\)(?P=raw_delim)"|//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_][A-Za-z_0-9]*|[^\s]', re.S)


def tokens(source):
    return [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(source)
            if not m.group().startswith(("//", "/*"))]


def matching(ts, start, left, right):
    depth = 0
    for i in range(start, len(ts)):
        if ts[i][0] == left:
            depth += 1
        elif ts[i][0] == right:
            depth -= 1
            if depth == 0:
                return i
    raise dp.PlanError("Unbalanced source syntax")


def axes_for(source, source_line):
    ts = tokens(source)
    offset = sum(len(s) for s in source.splitlines(keepends=True)[:source_line - 1])
    begin = next(i for i, (_, p, _) in enumerate(ts) if p >= offset)
    body = next(i for i in range(begin, len(ts)) if ts[i][0] == "{")
    end = matching(ts, body, "{", "}")
    axes = []
    i = body + 1
    while i < end:
        if ts[i][0] == "GENERATE":
            raise dp.PlanError("Unadapted GENERATE at line " +
                               str(source.count("\n", 0, ts[i][1]) + 1) +
                               "; use DAILY_GENERATE with the same finite choices")
        if ts[i][0] != "DAILY_GENERATE":
            i += 1
            continue
        if ts[i + 1][0] != "(":
            raise dp.PlanError("Invalid DAILY_GENERATE")
        close = matching(ts, i + 1, "(", ")")
        depth = 0
        count = 1
        has_value = False
        line = source.count("\n", 0, ts[i][1]) + 1
        for token, _, _ in ts[i + 2:close]:
            if token == "," and depth == 0:
                if not has_value:
                    raise dp.PlanError(f"Empty DAILY_GENERATE value at line {line}")
                has_value = False
            else:
                has_value = True
            if token in ("(", "{", "["):
                depth += 1
            elif token in (")", "}", "]"):
                depth -= 1
            elif token == "," and depth == 0:
                count += 1
        if not has_value:
            raise dp.PlanError(f"Empty DAILY_GENERATE value at line {line}")
        expression = source[ts[i + 1][2]:ts[close][1]]
        # These adapters accept finite values, not random/range generators.
        if re.search(r"\b(from_range|range|random|take|filter|map|values)\s*\(", expression):
            raise dp.PlanError("Generator adapter needs an explicit inventory")
        line = source.count("\n", 0, ts[i][1]) + 1
        end_line = source.count("\n", 0, ts[close][1]) + 1
        axes.append({"line": str(line), "end_line": str(end_line),
                     "choices": count, "expression": expression.strip()})
        i = close + 1
    if len({a["line"] for a in axes}) != len(axes):
        raise dp.PlanError("Multiple generators on one line are unsupported")
    return axes


def configurations(source, source_line, *, source_file="<source>", case_name="<case>"):
    try:
        axes = axes_for(source, source_line)
        count = 1
        for axis in axes:
            count *= axis["choices"]
        if count > 4096:
            raise dp.PlanError("Parameter product exceeds 4096; explicitly group it before execution")
        return axes, [
            {a["line"]: index for a, index in zip(axes, indices)}
            for indices in itertools.product(*(range(a["choices"]) for a in axes))
        ]
    except (dp.PlanError, StopIteration, IndexError) as error:
        raise dp.PlanError(f"{source_file}:{source_line}: {case_name}: {error}") from error


def runtime_choices(axes, choice):
    """Accept opening/closing macro locations used by GCC and Clang respectively."""
    result = dict(choice)
    owners = {}
    for axis in axes:
        for location in {axis["line"], axis.get("end_line", axis["line"])}:
            if location in owners and owners[location] != axis["line"]:
                raise dp.PlanError("Generator line aliases overlap")
            owners[location] = axis["line"]
        alias = axis.get("end_line", axis["line"])
        value = choice[axis["line"]]
        if alias in result and result[alias] != value:
            raise dp.PlanError("Generator line aliases overlap")
        result[alias] = value
    return result


def source_cases(source):
    """Read declarations in adapted files without executing or preprocessing C++."""
    ts = tokens(source)
    i = 0
    while i < len(ts):
        macro = ts[i][0]
        if not macro.startswith("TEST_CASE"):
            i += 1
            continue
        line = source.count("\n", 0, ts[i][1]) + 1
        if macro not in ("TEST_CASE", "TEST_CASE_PERSISTENT_FIXTURE"):
            raise dp.PlanError(f"Unsupported test declaration {macro} at line {line}")
        if i + 1 >= len(ts) or ts[i + 1][0] != "(":
            raise dp.PlanError(f"Invalid test declaration at line {line}")
        end = matching(ts, i + 1, "(", ")")
        if end + 1 >= len(ts) or ts[end + 1][0] != "{":
            raise dp.PlanError(f"Missing test body at line {line}")
        # The first string literal is the name; fixture types are identifiers.
        try:
            name = next((json.loads(t) for t, _, _ in ts[i + 2:end]
                         if t.startswith('"')), None)
        except json.JSONDecodeError as error:
            raise dp.PlanError(f"Invalid literal test name at line {line}: {error.msg}") from error
        if name is None:
            raise dp.PlanError(f"Missing literal test name at line {line}")
        yield {"name": name, "source_line": line}
        i = matching(ts, end + 1, "{", "}") + 1


def preflight(root):
    """Validate every adapted source case, including currently inactive #if branches.

    This inexpensive check does not replace the compiled inventory: the latter
    still decides which cases exist on each platform and validates the plan.
    """
    cases, errors = [], []
    for relative in PARAMETER_SOURCES:
        try:
            source = (root / relative).read_text()
            declarations = list(source_cases(source))
            if not declarations:
                raise dp.PlanError("No test declarations found")
        except (OSError, UnicodeError, dp.PlanError) as error:
            errors.append(f"{relative}: {error}")
            continue
        for test in declarations:
            try:
                axes, choices = configurations(source, test["source_line"],
                                              source_file=relative, case_name=test["name"])
                for choice in choices:
                    runtime_choices(axes, choice)
                cases.append({**test, "source_file": relative,
                              "axes": len(axes), "configurations": len(choices)})
            except dp.PlanError as error:
                errors.append(str(error))
    if errors:
        raise dp.PlanError("Daily parameter preflight failed:\n" + "\n".join(errors))
    return cases


def main():
    parser = argparse.ArgumentParser(description="Preflight Daily parameter adapters without a build")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        cases = preflight(args.root)
    except dp.PlanError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"Daily parameter preflight passed: {len(cases)} cases, "
          f"{sum(c['configurations'] for c in cases)} finite configurations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
