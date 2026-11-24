"""Publish the README badge from a completed Nightly Coverage job report."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo


def write_badge(event_path: Path, report_path: Path, destination: Path) -> bool:
    run = json.loads(event_path.read_text())["workflow_run"]
    run_id, run_number, attempt = (run[key] for key in ("id", "run_number", "run_attempt"))
    sha = run["head_sha"]
    if any(type(value) is not int or value <= 0 for value in (run_id, run_number, attempt)):
        raise ValueError("Invalid Nightly run identity")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Invalid Nightly source SHA")

    source_path = destination / "source.json"
    if source_path.exists():
        previous = json.loads(source_path.read_text())
        if (previous["run_number"], previous["run_attempt"]) >= (run_number, attempt):
            return False

    report = json.loads(report_path.read_text())
    if (
        report["schema_version"] != 1
        or report["job"] != "full-coverage-coverage"
        or report["status_before_upload"] != "success"
        or report["source_sha"] != sha
        or report["run_id"] != str(run_id)
        or report["attempt"] != str(attempt)
    ):
        raise ValueError("Coverage report does not match the successful Nightly job attempt")

    coverage = report["coverage"]
    hit, found = coverage["lines_hit"], coverage["lines_found"]
    if (
        coverage["available"] is not True
        or type(hit) is not int
        or type(found) is not int
        or found <= 0
        or not 0 <= hit <= found
    ):
        raise ValueError("Coverage report has invalid LCOV line counts")

    date = (
        datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
        .astimezone(ZoneInfo("Asia/Shanghai"))
        .date()
        .isoformat()
    )
    percent = f"{hit / found:.2%}"
    badge = {
        "schemaVersion": 1,
        "label": "Nightly C++ line coverage",
        "message": percent,
        "color": "blue",
    }
    source = {"run_id": run_id, "run_number": run_number, "run_attempt": attempt, "source_sha": sha}
    (destination / "badge.json").write_text(json.dumps(badge, indent=2) + "\n")
    (destination / "README.md").write_text(
        "# Nightly C++ line coverage\n\n"
        f"**{percent}** ({hit:,} / {found:,} lines covered by LCOV)\n\n"
        f"Scheduled Nightly run created on {date} (Beijing time), "
        f"commit `{sha}`.\n\n"
        f"[GitHub Actions run, attempt {attempt}]"
        f"(https://github.com/antgroup/vsag/actions/runs/{run_id}/attempts/{attempt})\n"
    )
    source_path.write_text(json.dumps(source, indent=2) + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    write_badge(args.event, args.report, args.destination)
