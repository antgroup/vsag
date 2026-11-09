#!/usr/bin/env python3
"""Small per-job diagnostics for the single final Daily Summary (no network)."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def cache_statistics(text):
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            values[parts[0]] = int(parts[1])
    required = ("cache_hit_direct", "cache_hit_preprocessed", "cache_miss")
    if not all(k in values for k in required):
        return {"available": False, "reason": "Missing ccache counters"}
    hits = values[required[0]] + values[required[1]]
    calls = hits + values[required[2]]
    return {"available": True, "hits": hits, "misses": values[required[2]],
            "cacheable_calls": calls, "hit_percent": 100 * hits / calls if calls else None,
            "counters": values}


def coverage_statistics(text):
    records = text.split("end_of_record")
    found = hit = count = 0
    seen = set()
    for record in records:
        fields = {}
        for line in record.splitlines():
            key, sep, value = line.partition(":")
            if sep and key in ("SF", "LF", "LH"):
                if key in fields:
                    raise ValueError("Duplicate LCOV field")
                fields[key] = value
        if not fields:
            continue
        source = fields.get("SF", "")
        if not source.startswith(("src/", "include/")) or ".." in source.split("/"):
            raise ValueError("Expected scoped repository-relative LCOV source")
        if source in seen:
            raise ValueError("Duplicate LCOV source")
        seen.add(source)
        lf, lh = int(fields["LF"]), int(fields["LH"])
        if not 0 <= lh <= lf:
            raise ValueError("Invalid LCOV line totals")
        found += lf
        hit += lh
        count += 1
    if not found:
        raise ValueError("No instrumented lines")
    return {"available": True, "files": count, "lines_found": found, "lines_hit": hit,
            "line_percent": 100 * hit / found, "target_percent": 90,
            "meets_target": hit * 100 >= found * 90}


def collect():
    # These identify evidence, not optional measurements. Do not invent defaults
    # that could attribute a report to the wrong run or hide missing evidence.
    required = ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB", "DAILY_JOB_STATUS")
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        raise ValueError("Missing required Daily diagnostic context: " + ", ".join(missing))
    cache = {"available": False, "reason": "ccache unavailable or not used"}
    try:
        result = subprocess.run(["ccache", "--print-stats"], capture_output=True,
                                text=True, timeout=10, check=True)
        cache = cache_statistics(result.stdout)
    except (OSError, subprocess.SubprocessError):
        pass
    coverage = {"available": False, "reason": "No scoped LCOV trace in this job"}
    path = Path("coverage/coverage.info")
    if path.is_file():
        try:
            coverage = coverage_statistics(path.read_text())
        except (OSError, ValueError, KeyError) as error:
            coverage["reason"] = str(error)
    return {"schema_version": 1, "source_sha": os.environ["GITHUB_SHA"],
            "run_id": os.environ["GITHUB_RUN_ID"], "attempt": os.environ["GITHUB_RUN_ATTEMPT"],
            "job": os.environ["GITHUB_JOB"], "matrix": json.loads(os.environ.get("DAILY_JOB_MATRIX", "null")),
            "status_before_upload": os.environ["DAILY_JOB_STATUS"],
            "cache": cache, "coverage": coverage}


def load_reports(directory, source_sha, run_id=None):
    reports, seen = [], set()
    for path in sorted(Path(directory).rglob("job-report.json")):
        report = json.loads(path.read_text())
        if (report.get("schema_version") != 1 or report.get("source_sha") != source_sha
                or (run_id is not None and report.get("run_id") != run_id)):
            raise ValueError("Daily job diagnostic source/run mismatch")
        identity = (report["job"], json.dumps(report["matrix"], sort_keys=True))
        if identity in seen:
            raise ValueError("Duplicate Daily job diagnostic")
        seen.add(identity)
        reports.append(report)
    return reports


def markdown(reports):
    lines = ["### Coverage", ""]
    coverage = [r for r in reports if r["job"] == "full-coverage-coverage"]
    if not coverage or not coverage[0]["coverage"].get("available"):
        lines += ["**Not measured / missing report.** A successful Job or Codecov upload is not proof of coverage."]
    else:
        c = coverage[0]["coverage"]
        state = "🟢 target met" if c["meets_target"] else "🔴 below target"
        lines += [f"Scoped C++ line coverage: **{c['line_percent']:.2f}% / {c['target_percent']}%** — {state}.",
                  f"{c['lines_hit']:,} / {c['lines_found']:,} instrumented lines; {c['files']} files.",
                  "Coverage target status is separate from test execution and upload success."]
    lines += ["", "<details><summary>Per-job diagnostics and compiler caches</summary>", "",
              "| Job / matrix | State before final upload | Hits / cacheable calls | Hit rate |",
              "| --- | --- | ---: | ---: |"]
    for r in reports:
        name = (r["job"] + (" / " + json.dumps(r["matrix"], sort_keys=True) if r["matrix"] else ""))
        name = name.replace("|", "\\|").replace("\n", " ")
        cache = r["cache"]
        calls = cache.get("cacheable_calls", 0)
        counts = f"{cache['hits']} / {calls}" if cache.get("available") and calls else "—"
        ratio = f"{cache['hit_percent']:.2f}%" if cache.get("available") and calls else "not measured / no calls"
        lines.append(f"| {name} | {r['status_before_upload']} | {counts} | {ratio} |")
    lines += ["", "Counters are captured before post-job cache saving. No compiler calls is not a 0% hit rate.",
              "The job-timing table below reports final GitHub job outcomes, including post steps.", "", "</details>", ""]
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.write_text(json.dumps(collect(), indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError) as error:
        parser.exit(1, "Daily diagnostics failed: " + str(error) + "\n")
