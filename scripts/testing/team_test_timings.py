#!/usr/bin/env python3
"""Export Team test observations and estimate durations; never publish trusted hints.

Consumes existing sealed manifests only: no test execution or extra per-case probes.
CSV is UTF-8, seconds are numeric, timestamps are UTC, and missing timings are empty.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
import sys

import team_test_shards as shards

SCHEMA = "1"
ENVIRONMENT = ["platform", "build_profile", "image", "workers", "batch_size",
               "batch_target_seconds", "omp_num_threads", "openblas_num_threads"]
FIELDS = ["schema_version", "record_id", "repository_id", "run_id", "run_attempt",
          "head_sha", "commit_sha", "plan_digest", "case_id", "execution_id", "kind",
          "binary", "case_name", "tags", "source_file", "source_line", *ENVIRONMENT,
          "shard", "seed", "observed_at", "duration_seconds", "measurement", "status",
          "batch_status", "batch_seconds"]
ESTIMATE_FIELDS = ["schema_version", "repository_id", "case_id", "kind", "binary",
                   "case_name", *ENVIRONMENT, "samples", "median_seconds", "p90_seconds",
                   "min_seconds", "max_seconds", "last_seen"]
STATUSES = {"passed", "failed", "skipped", "timeout", "cancelled", "launch_error",
            "unavailable", "missing"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(not isinstance(value, bool), "Boolean is not a duration")
    result = float(value)
    require(math.isfinite(result) and result >= 0, "Invalid duration")
    return result


def hint_id(kind, binary, case=None):
    # A performance hint survives source-line movement. Strict execution IDs and
    # their source-line/hash verification remain unchanged in team_test_shards.
    parts = ["timing-v1", kind, binary]
    if case is not None:
        parts.extend(case[key] for key in ("name", "tags", "source_file"))
    return shards.digest(parts)


def utc(timestamp):
    return datetime.fromtimestamp(number(timestamp), timezone.utc).isoformat().replace("+00:00", "Z")


def observations(plan, results, platform, profile, omp="", openblas=""):
    shards.validate_plan(plan)
    context = plan["inventory"]["context"]
    for key in ("repository_id", "run_id", "run_attempt", "head_sha", "test_sha", "image"):
        require(isinstance(context.get(key), str) and context[key], "Missing context: " + key)
    require(platform and profile, "Platform and build profile are required")
    by_shard = {}
    for result in results:
        shards.verify_seal(result, "result")
        require(result["plan_digest"] == plan["digest"] and result["context"] == context,
                "Result belongs to a different plan/context")
        index = result["shard"]
        require(type(index) is int and 0 <= index < len(plan["shards"]), "Invalid shard")
        require(index not in by_shard, "Duplicate shard result")
        require(result["test_ids"] == plan["shards"][index]["test_ids"], "Case manifest mismatch")
        require(result["seed"] == plan["inventory"]["seed"], "Seed mismatch")
        by_shard[index] = result
    inventory = {case["id"]: case for case in plan["inventory"]["tests"]}
    rows = []
    for assignment in plan["shards"]:
        index = assignment["index"]
        result = by_shard.get(index)
        common = dict(schema_version=SCHEMA, repository_id=context["repository_id"],
                      run_id=context["run_id"], run_attempt=context["run_attempt"],
                      head_sha=context["head_sha"], commit_sha=context["test_sha"],
                      image=context["image"], plan_digest=plan["digest"], platform=platform,
                      build_profile=profile, shard=index, seed=plan["inventory"]["seed"],
                      omp_num_threads=omp, openblas_num_threads=openblas)
        for field in ("workers", "batch_size", "batch_target_seconds"):
            common[field] = result.get(field, "") if result else ""
        common["observed_at"] = utc(result["finished_at"]) if result else ""
        cases, batches = {}, {}
        examples = {}
        if result:
            for batch in result["batches"]:
                require(batch["status"] in STATUSES, "Unknown batch status")
                if batch["status"] == "passed":
                    skipped_exit = (batch.get("exit_code") == 4 and batch.get("all_skipped") is True
                                    and bool(batch["tests"]) and all(
                                        test.get("success") is True and test.get("skips", 0) > 0
                                        for test in batch["tests"]))
                    require(type(batch.get("exit_code")) is int
                            and (batch["exit_code"] == 0 or skipped_exit),
                            "Passed batch has an invalid exit code")
                for test_id in batch["test_ids"]:
                    require(test_id in assignment["test_ids"] and test_id not in batches,
                            "Unexpected or duplicate batch case")
                    require(batch["binary"] == inventory[test_id]["binary"], "Wrong batch binary")
                    batches[test_id] = batch
                for case in batch["tests"]:
                    test_id = case["id"]
                    require(test_id in batch["test_ids"] and test_id not in cases,
                            "Unexpected or duplicate measured case")
                    require(shards.case_identity(case) == shards.case_identity(inventory[test_id]),
                            "Measured case identity mismatch")
                    require(type(case["success"]) is bool and type(case["skips"]) is int
                            and case["skips"] >= 0, "Invalid case outcome")
                    number(case["seconds"])
                    cases[test_id] = case
            expected_examples = shards.assigned_examples(plan, index)
            require(result.get("example_binaries", []) == assignment.get("example_binaries", []),
                    "Example manifest mismatch")
            for example in result.get("examples", []):
                binary = example["binary"]
                require(binary in expected_examples and binary not in examples,
                        "Unexpected or duplicate example")
                require(all(example.get(k) == v for k, v in expected_examples[binary].items()),
                        "Example identity mismatch")
                require(example["status"] in STATUSES, "Unknown example status")
                number(example["seconds"])
                examples[binary] = example
        for test_id in assignment["test_ids"]:
            case, measured, batch = inventory[test_id], cases.get(test_id), batches.get(test_id)
            status = "missing"
            if batch:
                status = batch["status"] if batch["status"] != "passed" else "unavailable"
            if measured:
                status = "failed" if not measured["success"] else (
                    "skipped" if measured["skips"] else "passed")
            rows.append(dict(common, case_id=hint_id("case", case["binary"], case),
                             execution_id=test_id, kind="case", binary=case["binary"],
                             case_name=case["name"], tags=case["tags"], source_file=case["source_file"],
                             source_line=case["source_line"], status=status,
                             duration_seconds=measured["seconds"] if measured else "",
                             measurement="catch2_case" if measured else "unavailable",
                             batch_status=batch["status"] if batch else "missing",
                             batch_seconds=number(batch["seconds"]) if batch else ""))
        for binary, metadata in shards.assigned_examples(plan, index).items():
            example = examples.get(binary)
            status = example["status"] if example else "missing"
            if example and status == "passed":
                require(type(example["exit_code"]) is int and example["exit_code"] == 0,
                        "Passed example has a nonzero exit code")
            # execute() has no exit code when cancellation precedes Popen or
            # launch fails. Its seconds value is not a process-wall measurement.
            measured = example is not None and example["exit_code"] is not None
            rows.append(dict(common, case_id=hint_id("example", binary), execution_id=metadata["id"],
                             kind="example", binary=binary, case_name=Path(binary).name,
                             tags="", source_file="", source_line="", status=status,
                             duration_seconds=example["seconds"] if measured else "",
                             measurement="process_wall" if measured else "unavailable",
                             batch_status=status, batch_seconds=""))
    for row in rows:
        row["record_id"] = shards.digest([context["repository_id"], context["run_id"],
                                         context["run_attempt"], plan["digest"], row["execution_id"]])
    require(len({r["case_id"] for r in rows}) == len(rows), "Ambiguous performance hint identity")
    return rows


def spreadsheet_escape(value):
    text = str(value) if value is not None else ""
    if text.startswith("'") or text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def spreadsheet_unescape(text):
    if text.startswith("'") and (text[1:].startswith("'") or
                                text[1:].lstrip().startswith(("=", "+", "-", "@"))):
        return text[1:]
    return text


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: spreadsheet_escape(row.get(field, "")) for field in fields})


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        require(reader.fieldnames == FIELDS, "Unexpected timing CSV schema")
        rows = []
        for row in reader:
            require(None not in row and None not in row.values(), "Malformed timing CSV row")
            rows.append({k: spreadsheet_unescape(v) for k, v in row.items()})
        return rows


def estimates(rows, window=20):
    require(type(window) is int and window > 0, "Window must be a positive sample count")
    unique, groups = {}, defaultdict(list)
    for raw in rows:
        row = {k: "" if v is None else str(v) for k, v in raw.items()}
        require(set(row) == set(FIELDS) and row["schema_version"] == SCHEMA,
                "Invalid observation schema")
        require(row["status"] in STATUSES and row["batch_status"] in STATUSES,
                "Invalid observation status")
        record_id = row["record_id"]
        require(record_id, "Missing record ID")
        if record_id in unique:
            require(unique[record_id] == row, "Conflicting duplicate observation")
            continue
        unique[record_id] = row
        require(row["kind"] in ("case", "example"), "Unknown observation kind")
        if row["duration_seconds"]:
            number(row["duration_seconds"])
        if row["status"] != "passed" or row["batch_status"] != "passed":
            continue
        require(row["measurement"] == ("catch2_case" if row["kind"] == "case" else "process_wall"),
                "Passed sample lacks the correct timing measurement")
        number(row["duration_seconds"])
        require(row["observed_at"].endswith("Z"), "Sample timestamp must be UTC")
        timestamp = datetime.fromisoformat(row["observed_at"].replace("Z", "+00:00"))
        key = tuple(row[k] for k in ["repository_id", "case_id", *ENVIRONMENT])
        groups[key].append((timestamp, record_id, row))
    output = []
    for key, samples in sorted(groups.items()):
        recent = sorted(samples, key=lambda item: item[:2])[-window:]
        last = recent[-1][2]
        values = sorted(number(item[2]["duration_seconds"]) for item in recent)
        output.append(dict({k: last[k] for k in ["schema_version", "repository_id", "case_id",
                                                 "kind", "binary", "case_name", *ENVIRONMENT]},
                           samples=len(values), median_seconds=statistics.median(values),
                           p90_seconds=values[math.ceil(len(values) * .9) - 1],
                           min_seconds=values[0], max_seconds=values[-1], last_seen=last["observed_at"]))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Export all planned cases/examples, including missing results")
    export.add_argument("--plan", required=True)
    export.add_argument("--results-dir", required=True)
    export.add_argument("--platform", required=True)
    export.add_argument("--build-profile", required=True)
    export.add_argument("--omp-num-threads", default="")
    export.add_argument("--openblas-num-threads", default="")
    export.add_argument("--output-dir", required=True)
    aggregate = commands.add_parser("estimate", help="Deduplicate and aggregate one or more raw timing CSVs")
    aggregate.add_argument("--input", action="append", required=True)
    aggregate.add_argument("--window", type=int, default=20)
    aggregate.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            plan = shards.load_json(args.plan)
            results = [shards.load_json(path) for path in sorted(Path(args.results_dir).rglob("result.json"))]
            rows = observations(plan, results, args.platform, args.build_profile,
                                args.omp_num_threads, args.openblas_num_threads)
            directory = Path(args.output_dir)
            write_csv(directory / "test-timings.csv", FIELDS, rows)
            write_csv(directory / "test-duration-estimates.csv", ESTIMATE_FIELDS, estimates(rows))
            print(f"Exported {len(rows)} observations; {sum(r['status'] == 'missing' for r in rows)} missing")
        else:
            rows = [row for path in args.input for row in read_csv(path)]
            write_csv(args.output, ESTIMATE_FIELDS, estimates(rows, args.window))
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        print(f"team_test_timings: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
