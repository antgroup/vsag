#!/usr/bin/env python3
"""Report testcase counts, modeled budgets and measured case time without running tests."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
import zipfile


MAX_MEMBER_BYTES = 64 * 1024 * 1024
CASE_FIELDS = ("name", "tags", "source_file", "source_line")
CATEGORIES = (
    "tune", "recovery", "serialization", "mutation_concurrency",
    "search_filter", "build_train", "data_quant_simd", "other",
)
# This ordered, mutually exclusive classification is a reporting heuristic.
# The per-case output retains the assignment so reviewers can inspect it.
CATEGORY_RULES = (
    ("tune", r"\btun(?:e|ing|index)\b|autotune"),
    ("recovery", r"recover|crash|rollback|checkpoint recovery|restart"),
    ("serialization", r"serializ|round.?trip|restore|persist|checkpoint|export|import|reader|\bio\b"),
    ("mutation_concurrency", r"concurren|parallel|\b(?:add|insert|remove|delete|update|merge|mutation|duplicate)\b|continueadd"),
    ("search_filter", r"search|query|filter|recall|knn|rerank|\brange\b|distance"),
    ("build_train", r"build|train|factory|construct|\bcreate\b"),
    ("data_quant_simd", r"quant|simd|dataset|vector|codec|encode|decode|kmeans|scalar|fp16|bf16|rabitq|allocat|memory"),
)
FAMILY_RULES = (
    ("hgraph", r"hgraph"), ("ivf", r"\bivf\b|ivf_"),
    ("pyramid", r"pyramid"), ("hnsw", r"hnsw"),
    ("sindi", r"sindi"), ("brute_force", r"brute.?force"),
    ("diskann", r"diskann"), ("odescent", r"odescent"),
    ("pipnn", r"pipnn"),
)


class ReportError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def parse_json(raw):
    def reject_constant(value):
        raise ReportError("Non-finite JSON number: " + value)

    try:
        return json.loads(raw, parse_constant=reject_constant)
    except (ValueError, UnicodeError) as error:
        raise ReportError("Invalid JSON: " + str(error)) from error


def read_bytes(path):
    path = Path(path)
    if path.stat().st_size > MAX_MEMBER_BYTES:
        raise ReportError("Input exceeds the 64 MiB limit: " + str(path))
    return path.read_bytes()


def member_bytes(archive, name):
    if archive.getinfo(name).file_size > MAX_MEMBER_BYTES:
        raise ReportError("Archive member exceeds the 64 MiB limit: " + name)
    return archive.read(name)


class NoDoctypeTreeBuilder(ET.TreeBuilder):
    def doctype(self, name, pubid, system):
        # Parser callbacks see declarations after XML encoding detection, so
        # UTF-16 cannot bypass this check as it can a raw-byte substring scan.
        raise ReportError("XML entities and DTDs are not accepted")


def xml_root(path, expected):
    raw = read_bytes(path)
    try:
        root = ET.fromstring(raw, parser=ET.XMLParser(target=NoDoctypeTreeBuilder()))
    except ET.ParseError as error:
        raise ReportError("Invalid Catch2 XML: " + str(error)) from error
    if root.tag != expected:
        raise ReportError("Expected " + expected + " XML")
    return root


def normalize_source(source, roots=(), require_relative=False):
    if not isinstance(source, str) or not source:
        raise ReportError("Missing source path")
    source = posixpath.normpath(source)
    if posixpath.isabs(source) and (roots or require_relative):
        matches = {posixpath.relpath(source, root) for root in roots
                   if source.startswith(root.rstrip("/") + "/")}
        if len(matches) != 1:
            raise ReportError("Absolute source needs one matching --source-root")
        source = matches.pop()
    # Catch may preserve compiler paths such as ../../../__w/vsag/vsag/tests/x.cpp.
    # This is identity data only: it is never opened or resolved on the filesystem.
    if source in (".", ".."):
        raise ReportError("Source must include a file identity")
    return source


def verify_seal(value):
    if not isinstance(value, dict) or value.get("digest") != digest(
            {key: item for key, item in value.items() if key != "digest"}):
        raise ReportError("Missing or invalid manifest digest")


def identity(case):
    if not isinstance(case, dict):
        raise ReportError("Case identity must be an object")
    for field in CASE_FIELDS[:-1]:
        if not isinstance(case.get(field), str):
            raise ReportError("Missing case identity field: " + field)
    if not case["name"] or not case["source_file"]:
        raise ReportError("Empty case identity")
    if type(case.get("source_line")) is not int or case["source_line"] <= 0:
        raise ReportError("Invalid source line")
    return tuple(case[field] for field in CASE_FIELDS)


def stable_id(binary, case):
    return digest([binary, *identity(case)])


def load_inventory(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if Path(name).name == "inventory.json"]
            if len(names) != 1:
                raise ReportError("Inventory ZIP must contain exactly one inventory.json")
            value = parse_json(member_bytes(archive, names[0]))
    else:
        value = parse_json(read_bytes(path))
    if isinstance(value, dict) and value.get("kind") == "plan":
        verify_seal(value)
        value = value.get("inventory")
    verify_seal(value)
    if value.get("kind") != "inventory" or value.get("schema_version") != 3:
        raise ReportError("Expected a frozen schema 3 Team inventory")
    tests, binaries = value.get("tests"), value.get("binaries")
    if not isinstance(tests, list) or not tests or not isinstance(binaries, dict):
        raise ReportError("Inventory has no cases or binaries")
    seen = set()
    for case in tests:
        binary = case.get("binary")
        if not isinstance(binary, str) or binary not in binaries:
            raise ReportError("Case binary is absent from inventory")
        case_id = stable_id(binary, case)
        if case.get("id") != case_id or case_id in seen:
            raise ReportError("Invalid or duplicate inventory case ID")
        seen.add(case_id)
    return value


def list_inventory(arguments, source_roots=()):
    """Create a local identity inventory; XML hashes do not attest binary contents."""
    cases, binaries, sources, seen = [], {}, [], set()
    roots = tuple(posixpath.normpath(str(root)) for root in source_roots)
    for argument in arguments:
        binary, path = binary_path_argument(argument)
        if binary in binaries:
            raise ReportError("Duplicate --list-tests binary")
        root = xml_root(path, "MatchingTests")
        for node in root.findall("TestCase"):
            try:
                case = {"binary": binary, "name": node.findtext("Name"),
                        "tags": node.findtext("Tags", ""),
                        "source_file": normalize_source(node.findtext("SourceInfo/File"), roots),
                        "source_line": int(node.findtext("SourceInfo/Line", ""))}
                case["id"] = stable_id(binary, case)
            except ValueError as error:
                raise ReportError("Invalid list-tests identity: " + str(error)) from error
            if case["id"] in seen:
                raise ReportError("Duplicate list-tests case identity")
            seen.add(case["id"])
            cases.append(case)
        binaries[binary] = {"list_xml_sha256": hashlib.sha256(read_bytes(path)).hexdigest(),
                            "binary_sha256": None}
        sources.append(path)
    if not cases:
        raise ReportError("List-tests inventory has no cases")
    value = {"kind": "inventory", "schema_version": 3,
             "context": {"input_kind": "catch-list-tests"}, "seed": None,
             "filter": "exactly as listed; no automatic exclusions",
             "tests": cases, "binaries": binaries, "list_sources": sources}
    return dict(value, digest=digest(value))


def finite_number(value, label, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ReportError("Invalid " + label)
    return value


def classify(case):
    text = (case["name"] + " " + case["tags"]).lower()
    for category, pattern in CATEGORY_RULES:
        if re.search(pattern, text):
            return category
    if re.search(r"/(?:simd|quantization|datacell)/", case["source_file"].lower()):
        return "data_quant_simd"
    return "other"


def family(case):
    # A source-owned family is less ambiguous than a case comparing two indexes.
    for text in (case["source_file"].lower(), (case["name"] + " " + case["tags"]).lower()):
        for label, pattern in FAMILY_RULES:
            if re.search(pattern, text):
                return label
    return "other"


class Inputs:
    def __init__(self, inventory, source_roots=()):
        self.inventory = inventory
        self.cases = {case["id"]: case for case in inventory["tests"]}
        self.timings = {}
        self.costs = {}
        self.cost_plans = {}
        self.source_roots = tuple(posixpath.normpath(str(root)) for root in source_roots)
        self.sources = []
        self.result_success = []

    def resolve(self, binary, case):
        case_id = stable_id(binary, case)
        if case_id not in self.cases:
            raise ReportError("Case does not match the exact inventory identity: " + case["name"])
        if "id" in case and case["id"] != case_id:
            raise ReportError("Case ID disagrees with its identity")
        return case_id

    def timing(self, binary, case):
        case_id = self.resolve(binary, case)
        if case_id in self.timings:
            raise ReportError("Duplicate measured case; do not pass both results and their XML")
        seconds = finite_number(case.get("seconds"), "case seconds")
        if type(case.get("success")) is not bool:
            raise ReportError("Missing case success flag")
        skips = case.get("skips")
        if type(skips) is not int or skips < 0:
            raise ReportError("Invalid skip count")
        self.timings[case_id] = {"seconds": seconds, "success": case["success"], "skips": skips}

    def result(self, value):
        verify_seal(value)
        if value.get("kind") != "result" or value.get("schema_version") != 3:
            raise ReportError("Expected a schema 3 shard result")
        if value.get("context") != self.inventory.get("context"):
            raise ReportError("Result belongs to a different run context")
        if value.get("seed") != self.inventory.get("seed"):
            raise ReportError("Result seed differs from inventory")
        if type(value.get("success")) is not bool or not isinstance(value.get("batches"), list):
            raise ReportError("Invalid shard result")
        self.result_success.append(value["success"])
        for batch in value["batches"]:
            if not isinstance(batch.get("tests"), list):
                raise ReportError("Invalid batch cases")
            for case in batch["tests"]:
                self.timing(batch.get("binary"), case)

    def result_path(self, path):
        self.sources.append(str(path))
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                names = [name for name in archive.namelist() if Path(name).name == "result.json"]
                if not names:
                    raise ReportError("Result ZIP contains no result.json")
                for name in names:
                    self.result(parse_json(member_bytes(archive, name)))
        else:
            self.result(parse_json(read_bytes(path)))

    def xml_path(self, binary, path):
        self.sources.append(str(path))
        try:
            root = xml_root(path, "Catch2TestRun")
            for node in root.findall("TestCase"):
                overall = node.find("OverallResult")
                if overall is None or overall.get("success") not in ("true", "false"):
                    raise ReportError("Incomplete Catch2 case result")
                case = {
                    "name": node.get("name"), "tags": node.get("tags", ""),
                    "source_file": normalize_source(node.get("filename")),
                    "source_line": int(node.get("line", "")),
                    "seconds": float(overall.get("durationInSeconds", "nan")),
                    "success": overall.get("success") == "true", "skips": int(overall.get("skips", "0")),
                }
                if stable_id(binary, case) not in self.cases:
                    case["source_file"] = normalize_source(case["source_file"], self.source_roots)
                self.timing(binary, case)
        except (ET.ParseError, ValueError) as error:
            raise ReportError("Invalid Catch2 XML: " + str(error)) from error

    def cost(self, binary, value):
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ReportError("Expected schema 1 testcase cost record")
        case = value.get("case")
        if isinstance(case, dict) and type(case.get("source_line")) is int and case["source_line"] == 0:
            # The lightweight Catch bridge knows the current name and __FILE__,
            # but not registration tags/line. Never guess by basename.
            if case.get("tags") != "" or not isinstance(case.get("source_file"), str):
                raise ReportError("Invalid partial cost identity")
            source = normalize_source(case["source_file"])
            candidates = [item for item in self.cases.values()
                          if item["binary"] == binary and item["name"] == case.get("name")]
            matches = [item["id"] for item in candidates
                       if normalize_source(item["source_file"]) == source]
            if not matches:
                source = normalize_source(case["source_file"], self.source_roots,
                                          require_relative=True)
                matches = [item["id"] for item in candidates
                           if normalize_source(item["source_file"], self.source_roots,
                                               require_relative=True) == source]
            if len(matches) != 1:
                raise ReportError("Partial cost identity must match exactly one inventory case")
            case_id = matches[0]
            if "id" in case and case["id"] != case_id:
                raise ReportError("Case ID disagrees with its identity")
        else:
            case_id = self.resolve(binary, case)
        plan_bytes = canonical(value)
        if case_id in self.cost_plans:
            if plan_bytes != self.cost_plans[case_id]:
                raise ReportError("Differing cost plans for the same case")
            self.costs[case_id]["identical_plan_repetitions"] += 1
            return
        for field in ("model_version", "unit", "policy"):
            if not isinstance(value.get(field), str) or not value[field]:
                raise ReportError("Missing cost field: " + field)
        cap = finite_number(value.get("cap_points"), "cap points", positive=True)
        total = finite_number(value.get("total_points"), "estimated cost points")
        self.cost_plans[case_id] = plan_bytes
        self.costs[case_id] = {"model_version": value["model_version"], "unit": value["unit"],
                               "policy": value["policy"], "cap_points": cap, "total_points": total,
                               "over_cap": total > cap, "seed": value.get("seed"),
                               "identical_plan_repetitions": 1}

    def cost_path(self, binary, path):
        self.sources.append(str(path))
        raw = read_bytes(path)
        probe = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
        # Keep XML bytes intact for the parser's encoding detection and DTD
        # guard; only JSONL/plain logs require UTF-8 decoding. NUL bytes allow
        # the opening marker in BOM-less UTF-16 to be detected as well.
        is_xml = (Path(path).suffix.lower() == ".xml"
                  or probe.startswith((b"\xff\xfe", b"\xfe\xff"))
                  or probe.lstrip(b"\x00 \t\r\n").startswith(b"<"))
        if is_xml:
            root = xml_root(path, "Catch2TestRun")
            # ElementTree decodes names such as "Build & ContinueAdd" before JSON
            # parsing. Only captured stdout may contain explicit budget records.
            lines = [line.strip() for node in root.iter("StdOut")
                     for line in "".join(node.itertext()).splitlines() if line.strip()]
        else:
            lines = [line.strip() for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
        prefixed = [line[len("[test-budget]"):].strip() for line in lines if line.startswith("[test-budget]")]
        # In a mixed test log only explicitly marked records are input. Without
        # markers a text file must be JSONL. XML without budget records leaves
        # cases unmodeled, rather than interpreting unrelated stdout as JSON.
        for line in prefixed if is_xml or prefixed else lines:
            self.cost(binary, parse_json(line))


def ratio(value, total):
    return value / total if total else None


def metric(cases, groups, values):
    selected = {case["id"]: values[case["id"]] for case in cases if case["id"] in values}
    total = sum(selected.values()) if selected else None
    rows = []
    for label in groups:
        members = [case for case in cases if case["group"] == label]
        known = [selected[case["id"]] for case in members if case["id"] in selected]
        value = sum(known) if known else None
        rows.append({"group": label, "known_count": len(known), "unknown_count": len(members) - len(known),
                     "value": value, "share_of_known_total": ratio(value, total) if value is not None else None})
    return {"known_count": len(selected), "unknown_count": len(cases) - len(selected),
            "total": total, "groups": rows}


def make_report(inputs, binaries=None, group_by="category"):
    binaries = set(binaries or ())
    known_binaries = set(inputs.inventory["binaries"])
    if not binaries <= known_binaries:
        raise ReportError("Requested binary is absent from inventory")
    cases = []
    for original in inputs.inventory["tests"]:
        if binaries and original["binary"] not in binaries:
            continue
        case = dict(original, category=classify(original), family=family(original))
        case["group"] = case[group_by]
        case["timing"] = inputs.timings.get(case["id"])
        case["cost"] = inputs.costs.get(case["id"])
        cases.append(case)
    groups = [group for group in CATEGORIES if any(case["group"] == group for case in cases)] if group_by == "category" else sorted({case["group"] for case in cases})
    counts = Counter(case["group"] for case in cases)
    models = sorted({(case["cost"]["model_version"], case["cost"]["unit"], case["cost"]["policy"])
                     for case in cases if case["cost"] is not None})
    modeled = []
    for model, unit, policy in models:
        records = {case["id"]: case["cost"] for case in cases if case["cost"] is not None
                   and (case["cost"]["model_version"], case["cost"]["unit"], case["cost"]["policy"]) == (model, unit, policy)}
        modeled.append({"model_version": model, "unit": unit, "policy": policy,
                        "cap": metric(cases, groups, {key: item["cap_points"] for key, item in records.items()}),
                        "estimated_cost": metric(cases, groups, {key: item["total_points"] for key, item in records.items()})})
    return {
        "schema_version": 1, "kind": "test-budget-report", "classification": "ordered-name-tags-source-v1",
        "group_by": group_by, "inventory_digest": inputs.inventory["digest"],
        "context": inputs.inventory.get("context"), "seed": inputs.inventory.get("seed"),
        "filter": inputs.inventory.get("filter"), "sources": inputs.sources,
        "inventory_case_count": len(inputs.cases), "selected_case_count": len(cases),
        "binary_counts": dict(sorted(Counter(case["binary"] for case in cases).items())),
        "case_counts": [{"group": group, "count": counts[group], "share": ratio(counts[group], len(cases))} for group in groups],
        "case_seconds": metric(cases, groups, {key: item["seconds"] for key, item in inputs.timings.items()}),
        "modeled_costs": modeled,
        "cost_known_count": sum(case["cost"] is not None for case in cases),
        "cost_unknown_count": sum(case["cost"] is None for case in cases),
        "over_cap_count": sum(bool(case["cost"] and case["cost"]["over_cap"]) for case in cases),
        "measured_failed_cases": sum(bool(case["timing"] and not case["timing"]["success"]) for case in cases),
        "measured_cases_with_skip": sum(bool(case["timing"] and case["timing"]["skips"]) for case in cases),
        "failed_input_results": inputs.result_success.count(False),
        "notes": ["Case counts retain binary identity, including cases present in two binaries.",
                  "Cap and estimated cost shares use only known records of the same model, unit and policy.",
                  "Missing cost or timing is unknown, never zero. Zero totals have undefined shares.",
                  "Identical whole-case plans repeated by Catch re-entry count once; repetitions are reported per case.",
                  "Sum of case seconds is not elapsed CI time or CPU time; cases can execute concurrently.",
                  "Categories are mutually exclusive heuristics, not claims about assertion coverage.",
                  "This is a report, not a CI pass/fail gate or a test runner."],
        "cases": sorted(cases, key=lambda case: (case["binary"], case["name"], case["id"])),
    }


def markdown(report):
    def percent(value):
        return "unknown" if value is None else f"{100 * value:.2f}%"

    def number(value):
        return "unknown" if value is None else f"{value:.3f}"

    lines = ["# Testcase budget report", "", f"Selected cases: **{report['selected_case_count']}** / inventory {report['inventory_case_count']}.",
             f"Cost records: {report['cost_known_count']} known, **{report['cost_unknown_count']} unknown**.", "",
             "Run context: `" + json.dumps(report["context"], sort_keys=True) + "`", "",
             "Case-second sums are **not wall time or CPU time**. Classification is an ordered heuristic.", "",
             "| Group | Cases | Case share | Timed / unknown | Case seconds | Share of known seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    timings = {row["group"]: row for row in report["case_seconds"]["groups"]}
    for row in report["case_counts"]:
        timing = timings[row["group"]]
        lines.append(f"| {row['group']} | {row['count']} | {percent(row['share'])} | {timing['known_count']} / {timing['unknown_count']} | {number(timing['value'])} | {percent(timing['share_of_known_total'])} |")
    for model in report["modeled_costs"]:
        lines.extend(["", f"## {model['model_version']} / {model['unit']} / {model['policy']}", "",
                      f"Known modeled cases: {model['cap']['known_count']}; unmodeled for this model: {model['cap']['unknown_count']}.", "",
                      "| Group | Modeled / unknown | Cap points | Cap share | Estimated points | Estimated share |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for cap, cost in zip(model["cap"]["groups"], model["estimated_cost"]["groups"]):
            lines.append(f"| {cap['group']} | {cap['known_count']} / {cap['unknown_count']} | {number(cap['value'])} | {percent(cap['share_of_known_total'])} | {number(cost['value'])} | {percent(cost['share_of_known_total'])} |")
    if not report["modeled_costs"]:
        lines.extend(["", "Cap and estimated-cost shares are **unavailable**: no cost records were supplied."])
    lines.extend(["", f"Over cap: {report['over_cap_count']}; measured failed cases: {report['measured_failed_cases']}; cases with skip: {report['measured_cases_with_skip']}; failed input results: {report['failed_input_results']}.", ""])
    lines.extend("- " + note for note in report["notes"])
    return "\n".join(lines) + "\n"


def binary_path_argument(value):
    binary, separator, path = value.partition("=")
    if not separator or not binary or not path:
        raise ReportError("Expected BINARY=PATH")
    return binary, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inventory_group = parser.add_mutually_exclusive_group(required=True)
    inventory_group.add_argument("--inventory", help="Frozen schema 3 inventory/plan JSON or Team plan ZIP")
    inventory_group.add_argument("--list-tests", action="append", metavar="BINARY=PATH", help="Catch2 --list-tests --reporter xml output; repeat for each binary")
    parser.add_argument("--result", action="append", default=[], help="Shard result.json or Team results ZIP; repeatable")
    parser.add_argument("--xml", action="append", default=[], metavar="BINARY=PATH", help="Catch2 execution XML; do not duplicate --result cases")
    parser.add_argument("--cost", action="append", default=[], metavar="BINARY=PATH", help="Complete-case cost JSONL, prefixed log, or Catch2 execution XML StdOut")
    parser.add_argument("--source-root", action="append", default=[], help="Exact checkout root for absolute __FILE__ paths in partial cost identities")
    parser.add_argument("--binary", action="append", help="Report only this inventory binary; repeatable")
    parser.add_argument("--group-by", choices=("category", "family"), default="category")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", help="Output file; default stdout")
    args = parser.parse_args(argv)
    try:
        inventory = load_inventory(args.inventory) if args.inventory else list_inventory(args.list_tests, args.source_root)
        inputs = Inputs(inventory, args.source_root)
        inputs.sources.extend(inventory.get("list_sources", []))
        for path in args.result:
            inputs.result_path(path)
        for item in args.xml:
            inputs.xml_path(*binary_path_argument(item))
        for item in args.cost:
            inputs.cost_path(*binary_path_argument(item))
        report = make_report(inputs, args.binary, args.group_by)
        rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n" if args.format == "json" else markdown(report)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (ReportError, OSError, zipfile.BadZipFile) as error:
        print("test-budget-report: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
