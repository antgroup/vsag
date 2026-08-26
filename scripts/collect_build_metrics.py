#!/usr/bin/env python3
"""Collect repeatable CMake, Ninja, and ccache build-performance metrics."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any


def elapsed_seconds(start_ns: int, end_ns: int) -> float:
    return round((end_ns - start_ns) / 1_000_000_000, 3)


def load_ccache_stats(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        values = {}
        for line in text.splitlines():
            fields = line.split("\t", 1)
            if len(fields) == 2 and fields[1].isdigit():
                values[fields[0]] = int(fields[1])
        if not values:
            return {"available": False, "raw": text.strip()}
        return {
            "available": True,
            "cache_hit": values.get("direct_cache_hit", 0)
            + values.get("preprocessed_cache_hit", 0),
            "cache_miss": values.get("cache_miss", 0),
            "raw": values,
        }
    stats = data.get("stats", data)
    return {
        "available": True,
        "cache_hit": stats.get("cache_hit", stats.get("cache_hit_direct", 0)),
        "cache_miss": stats.get("cache_miss", 0),
        "raw": data,
    }


def read_compile_commands(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    commands = json.loads(path.read_text(encoding="utf-8"))
    sources: dict[str, str] = {}
    for entry in commands:
        arguments = entry.get("arguments")
        if arguments is None:
            arguments = shlex.split(entry.get("command", ""))
        try:
            output = arguments[arguments.index("-o") + 1]
        except (ValueError, IndexError):
            output = entry.get("output")
        if output:
            sources[str(Path(output))] = entry.get("file", "")
            sources[Path(output).name] = entry.get("file", "")
    return sources


def is_dependency_output(normalized: str) -> bool:
    return normalized.startswith(("_deps/", "extern/", "hdf5", "openblas")) or any(
        part in normalized for part in ("/_deps/", "/extern/", "/hdf5", "/openblas")
    )


def classify_edge(output: str, link_outputs: set[str] | None = None) -> str:
    normalized = output.replace("\\", "/").lower()
    if normalized.endswith((".o", ".obj")):
        if is_dependency_output(normalized):
            return "dependency_compile"
        if normalized.startswith("tests/") or any(
            part in normalized for part in ("/tests/", "test.dir/", "_test.dir/")
        ):
            return "test_compile"
        if normalized.startswith("src/cmakefiles/") or "/src/cmakefiles/" in normalized:
            return "production_compile"
        return "other_compile"
    if is_dependency_output(normalized):
        return "dependency_build"
    if link_outputs and output in link_outputs:
        return "link"
    if normalized.endswith((".a", ".so", ".dylib", ".dll", ".exe")):
        return "link"
    return "other"


def read_link_outputs(build_dir: Path) -> set[str]:
    result = subprocess.run(
        ["ninja", "-C", str(build_dir), "-t", "targets", "all"],
        text=True,
        capture_output=True,
        check=False,
    )
    outputs = set()
    for line in result.stdout.splitlines():
        output, separator, rule = line.rpartition(": ")
        if separator and "LINKER" in rule:
            outputs.add(output)
    return outputs


def parse_ninja_log(
    path: Path, compile_commands: dict[str, str], link_outputs: set[str] | None = None
) -> dict[str, Any]:
    categories: dict[str, dict[str, float | int]] = {}
    translation_units: list[dict[str, Any]] = []
    if not path.is_file():
        return {"available": False, "categories": categories, "slowest_translation_units": []}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        try:
            start_ms, end_ms = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        output = fields[3]
        duration = max(0, end_ms - start_ms) / 1000
        category = classify_edge(output, link_outputs)
        current = categories.setdefault(category, {"edges": 0, "cumulative_seconds": 0.0})
        current["edges"] = int(current["edges"]) + 1
        current["cumulative_seconds"] = round(float(current["cumulative_seconds"]) + duration, 3)
        if output.lower().endswith((".o", ".obj")):
            source = compile_commands.get(output, compile_commands.get(Path(output).name, output))
            translation_units.append(
                {"source": source, "output": output, "seconds": round(duration, 3), "category": category}
            )
    translation_units.sort(key=lambda item: item["seconds"], reverse=True)
    return {
        "available": True,
        "categories": categories,
        "slowest_translation_units": translation_units[:20],
    }


def parse_peak_rss(path: Path) -> int | None:
    if not path.is_file():
        return None
    match = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        path.read_text(encoding="utf-8", errors="replace"),
    )
    return int(match.group(1)) if match else None


def parse_ninja_stats(path: Path) -> list[str]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().startswith("metric") and "count" in line and "avg (us)" in line
        ),
        None,
    )
    if start is None:
        return []
    return [line for line in lines[start : start + 20] if line.strip()]


class Collector:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path.cwd().resolve()
        self.build_dir = (self.root / args.build_dir).resolve()
        self.output_dir = (self.root / args.output_dir).resolve()
        self.phases: list[dict[str, Any]] = []
        self.failure_code = 0
        if self.root not in self.build_dir.parents or self.build_dir == self.root:
            raise ValueError("build directory must be a child of the repository")
        if self.root not in self.output_dir.parents or self.output_dir == self.root:
            raise ValueError("output directory must be a child of the repository")
        if self.build_dir == self.output_dir or self.build_dir in self.output_dir.parents:
            raise ValueError("output directory must not be inside the build directory")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_logged(self, name: str, command: list[str]) -> dict[str, Any]:
        log_path = self.output_dir / f"{name}.log"
        time_path = self.output_dir / f"{name}.time.txt"
        timed_command = command
        time_binary = Path("/usr/bin/time")
        if time_binary.is_file():
            timed_command = [str(time_binary), "-v", "-o", str(time_path), *command]
        started = time.monotonic_ns()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                timed_command,
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
            return_code = process.wait()
        finished = time.monotonic_ns()
        phase = {
            "name": name,
            "command": command,
            "elapsed_seconds": elapsed_seconds(started, finished),
            "exit_code": return_code,
            "peak_rss_kib": parse_peak_rss(time_path),
            "ninja_stats": parse_ninja_stats(log_path),
        }
        self.phases.append(phase)
        if return_code and not self.failure_code:
            self.failure_code = return_code
        return phase

    def ccache(self, *arguments: str) -> dict[str, Any]:
        result = subprocess.run(
            ["ccache", *arguments], cwd=self.root, text=True, capture_output=True, check=False
        )
        if "--print-stats" in arguments:
            return load_ccache_stats(result.stdout)
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

    def capture_build(self, name: str) -> None:
        ninja_log = self.build_dir / ".ninja_log"
        ninja_log.unlink(missing_ok=True)
        self.ccache("--zero-stats")
        phase = self.run_logged(
            name,
            [
                "make",
                "build-asan",
                f"COMPILE_JOBS={self.args.jobs}",
                "CMAKE_BUILD_ARGS=-d stats",
            ],
        )
        captured_log = self.output_dir / f"{name}.ninja_log"
        if ninja_log.is_file():
            shutil.copy2(ninja_log, captured_log)
        phase["ccache"] = self.ccache("--print-stats")
        phase["ninja"] = parse_ninja_log(
            captured_log,
            read_compile_commands(self.build_dir / "compile_commands.json"),
            read_link_outputs(self.build_dir),
        )

    def collect(self) -> int:
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        if self.args.clear_ccache:
            self.ccache("--clear")
        self.ccache("--zero-stats")
        configure = self.run_logged(
            "configure",
            ["make", "configure-asan", f"COMPILE_JOBS={self.args.jobs}"],
        )
        configure["ccache"] = self.ccache("--print-stats")
        if not self.failure_code:
            self.capture_build("cold_build")
        if not self.failure_code:
            self.run_logged(
                "clean_for_warm_build",
                ["cmake", "--build", str(self.build_dir), "--target", "clean"],
            )
            if not self.failure_code:
                self.capture_build("warm_ccache_build")
        if not self.failure_code:
            self.capture_build("noop_incremental_build")
        self.write_reports()
        return self.failure_code

    def write_reports(self) -> None:
        report = {
            "schema_version": 1,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "failed" if self.failure_code else "success",
            "base_sha": self.args.base_sha,
            "commit_sha": self.args.commit_sha,
            "configuration": {
                "jobs": self.args.jobs,
                "build_dir": str(self.build_dir.relative_to(self.root)),
                "cold_ccache_cleared": self.args.clear_ccache,
                "compiler": self.args.compiler,
                "compiler_cache_key": self.args.compiler_cache_key,
                "dependency_cache_key": self.args.dependency_cache_key,
                "dependency_preparation_seconds": self.args.dependency_preparation_seconds,
            },
            "phases": self.phases,
        }
        (self.output_dir / "build-metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        detailed = render_markdown(report, concise=False)
        summary = render_markdown(report, concise=True)
        (self.output_dir / "build-metrics.md").write_text(detailed, encoding="utf-8")
        (self.output_dir / "job-summary.md").write_text(summary, encoding="utf-8")


def metric(phase: dict[str, Any], category: str) -> float:
    return float(
        phase.get("ninja", {})
        .get("categories", {})
        .get(category, {})
        .get("cumulative_seconds", 0)
    )


def render_markdown(report: dict[str, Any], concise: bool) -> str:
    config = report["configuration"]
    phases = {phase["name"]: phase for phase in report["phases"]}
    lines = ["## Build performance metrics", ""]
    lines.extend(
        [
            f"Status: **{report['status']}** · compiler: `{config['compiler']}` · parallel jobs: `{config['jobs']}`",
            "",
            "| Phase | Wall time | Peak RSS | ccache hits | ccache misses |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, label in (
        ("configure", "CMake configure"),
        ("cold_build", "Cold build"),
        ("warm_ccache_build", "Warm ccache rebuild"),
        ("noop_incremental_build", "No-op incremental"),
    ):
        phase = phases.get(name)
        if phase is None:
            continue
        ccache = phase.get("ccache", {})
        peak = phase.get("peak_rss_kib")
        peak_text = f"{peak / 1024:.1f} MiB" if peak is not None else "n/a"
        lines.append(
            f"| {label} | {phase['elapsed_seconds']:.3f} s | {peak_text} | "
            f"{ccache.get('cache_hit', 'n/a')} | {ccache.get('cache_miss', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            f"Dependency source preparation: **{config['dependency_preparation_seconds']:.3f} s**",
            "",
            f"Compiler cache key: `{config['compiler_cache_key']}`  ",
            f"Dependency cache key: `{config['dependency_cache_key']}`",
            "",
        ]
    )
    cold = phases.get("cold_build", {})
    if cold:
        lines.extend(
            [
                "Cold-build cumulative Ninja edge time (parallel edges overlap):",
                "",
                f"- Dependencies: {metric(cold, 'dependency_compile') + metric(cold, 'dependency_build'):.3f} s",
                f"- VSAG production compile: {metric(cold, 'production_compile'):.3f} s",
                f"- Test compile: {metric(cold, 'test_compile'):.3f} s",
                f"- Link: {metric(cold, 'link'):.3f} s",
                "",
            ]
        )
    if concise:
        lines.append("The `build-performance-metrics` artifact contains JSON, complete logs, Ninja statistics, and the slowest translation units.")
        return "\n".join(lines) + "\n"
    slowest = cold.get("ninja", {}).get("slowest_translation_units", [])
    lines.extend(
        [
            "### Slowest translation units (cold build)",
            "",
            "| Translation unit | Category | Time |",
            "| --- | --- | ---: |",
        ]
    )
    for unit in slowest:
        lines.append(f"| `{unit['source']}` | {unit['category']} | {unit['seconds']:.3f} s |")
    for phase in report["phases"]:
        if phase.get("ninja_stats"):
            lines.extend(
                [
                    "",
                    f"### Ninja scheduling statistics: {phase['name']}",
                    "",
                    "```text",
                    *phase["ninja_stats"],
                    "```",
                ]
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--output-dir", default="build-metrics")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--commit-sha", default="")
    parser.add_argument("--compiler", default="unknown")
    parser.add_argument("--compiler-cache-key", default="unknown")
    parser.add_argument("--dependency-cache-key", default="unknown")
    parser.add_argument("--dependency-preparation-seconds", type=float, default=0)
    parser.add_argument("--clear-ccache", action="store_true")
    return parser.parse_args()


def main() -> int:
    return Collector(parse_args()).collect()


if __name__ == "__main__":
    raise SystemExit(main())
