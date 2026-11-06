#!/usr/bin/env python3
"""Finite explicit Catch2 generator axes; reject syntax we cannot enumerate.

SECTION branches remain intact. The product is conservative: choices in mutually
exclusive branches can produce duplicate exercised paths, never omitted paths.
"""
import itertools
import re
from pathlib import Path

import daily_plan as dp

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
            raise dp.PlanError("Unadapted GENERATE in a planned parameter testcase")
        if ts[i][0] != "DAILY_GENERATE":
            i += 1
            continue
        if ts[i + 1][0] != "(":
            raise dp.PlanError("Invalid DAILY_GENERATE")
        close = matching(ts, i + 1, "(", ")")
        depth = 0
        count = 1
        for token, _, _ in ts[i + 2:close]:
            if token in ("(", "{", "["):
                depth += 1
            elif token in (")", "}", "]"):
                depth -= 1
            elif token == "," and depth == 0:
                count += 1
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


def configurations(source, source_line):
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
