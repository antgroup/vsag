#!/usr/bin/env python3
# Copyright 2024-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Summarize reproducible Full/Lite benchmark artifacts."""

import json
import pathlib
import re
import sys


def peak_rss(path: pathlib.Path) -> int:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", path.read_text())
    return int(match.group(1)) if match else 0


def percentage(lite: float, full: float) -> float:
    return 0.0 if full == 0 else (lite - full) * 100.0 / full


def main() -> None:
    output = pathlib.Path(sys.argv[1])
    rows = {}
    for variant in ("full", "lite"):
        values = json.loads((output / f"{variant}.json").read_text())
        values["library_bytes"] = int((output / f"{variant}.library_bytes").read_text())
        values["peak_rss_kb"] = peak_rss(output / f"{variant}.time")
        rows[variant] = values

    keys = (
        "library_bytes",
        "peak_rss_kb",
        "build_ms",
        "deserialize_ms",
        "index_bytes",
        "qps",
        "p50_ms",
        "p99_ms",
        "recall_at_10",
    )
    print("| Metric | Full | Lite | Lite vs Full |")
    print("| --- | ---: | ---: | ---: |")
    for key in keys:
        full = rows["full"][key]
        lite = rows["lite"][key]
        print(f"| {key} | {full:.6g} | {lite:.6g} | {percentage(lite, full):+.2f}% |")


if __name__ == "__main__":
    main()
