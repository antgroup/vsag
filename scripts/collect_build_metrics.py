#!/usr/bin/env python3
"""Collect repeatable CMake, Ninja, and ccache build-performance metrics."""

from __future__ import annotations

import argparse
from collections import Counter
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


DEPENDENCY_ALIASES = {
    "nlohmann_json": "json",
    "nlohmann-json": "json",
    "thread-pool": "thread_pool",
    "yaml-cpp": "yaml_cpp",
}
DEPENDENCY_DISPLAY_NAMES = {
    "antlr4": "ANTLR4",
    "argparse": "argparse",
    "catch2": "Catch2",
    "cpuinfo": "CPUinfo",
    "fmt": "fmt",
    "hdf5": "HDF5",
    "httplib": "cpp-httplib",
    "json": "nlohmann/json",
    "mkl": "Intel MKL",
    "openblas": "OpenBLAS",
    "roaringbitmap": "CRoaring",
    "tabulate": "tabulate",
    "thread_pool": "thread-pool",
    "tsl": "tsl::robin_map",
    "yaml_cpp": "yaml-cpp",
}
FETCHCONTENT_DIRECTORY_NAMES = {
    "argparse": "argparse",
    "catch2": "catch2",
    "cpuinfo": "cpuinfo",
    "fmt": "fmt",
    "httplib": "httplib",
    "json": "nlohmann_json",
    "pybind11": "pybind11",
    "roaringbitmap": "roaringbitmap",
    "tabulate": "tabulate",
    "thread_pool": "thread_pool",
    "tsl": "tsl",
    "yaml_cpp": "yaml-cpp",
}
EXTERNAL_PROJECT_DEPENDENCIES = {"antlr4", "hdf5", "openblas"}
EXTERNAL_PROJECT_STAGES = (
    "mkdir",
    "download",
    "update",
    "patch",
    "configure",
    "build",
    "install",
    "test",
    "done",
)


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


def canonical_dependency(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_")
    return DEPENDENCY_ALIASES.get(normalized, normalized)


def normalize_output(output: str) -> str:
    return output.replace("\\", "/").lower().removeprefix("./")


def external_project_stage(output: str) -> tuple[str, str] | None:
    normalized = normalize_output(output)
    if "/.vsag-build-info/" not in f"/{normalized}" and "-stamp/" not in normalized:
        return None
    basename = normalized.rsplit("/", 1)[-1]
    for stage in EXTERNAL_PROJECT_STAGES:
        suffix = f"-{stage}"
        if basename.endswith(suffix) and len(basename) > len(suffix):
            dependency = canonical_dependency(basename[: -len(suffix)])
            return dependency, stage
    return None


def dependency_from_output(output: str) -> str | None:
    normalized = normalize_output(output)
    stage = external_project_stage(normalized)
    if stage is not None:
        return stage[0]
    components = [component for component in normalized.split("/") if component]
    for marker in (".ci-fetchcontent", "_deps"):
        if marker not in components:
            continue
        marker_index = components.index(marker)
        if marker_index + 1 >= len(components):
            continue
        directory = components[marker_index + 1]
        match = re.fullmatch(r"(.+)-(?:build|src|subbuild)", directory)
        if match:
            return canonical_dependency(match.group(1))
    if "extern" in components:
        extern_index = components.index("extern")
        if extern_index + 1 < len(components):
            return canonical_dependency(components[extern_index + 1])
    for dependency in EXTERNAL_PROJECT_DEPENDENCIES:
        if dependency in components:
            return dependency
    return None


def is_build_maintenance_output(normalized: str) -> bool:
    components = normalize_output(normalized).split("/")
    return components[-1] == "cmake.verify_globs" or (
        len(components) >= 2
        and components[-2] == "cmakefiles"
        and components[-1] == "version"
    )


def classify_output(
    output: str, normalized_link_outputs: set[str] | None = None
) -> dict[str, str | None]:
    normalized = normalize_output(output)
    stage = external_project_stage(normalized)
    dependency = dependency_from_output(normalized)
    if is_build_maintenance_output(normalized):
        return {"category": "build_maintenance", "dependency": None, "stage": None}
    if stage is not None:
        return {
            "category": "dependency_build",
            "dependency": stage[0],
            "stage": stage[1],
            "external_project_stage": stage[1],
        }
    if normalized.endswith((".o", ".obj")):
        if dependency is not None:
            return {
                "category": "dependency_compile",
                "dependency": dependency,
                "stage": "compile",
            }
        if normalized.startswith("tests/") or any(
            part in normalized for part in ("/tests/", "test.dir/", "_test.dir/")
        ):
            return {"category": "test_compile", "dependency": None, "stage": None}
        if normalized.startswith("src/cmakefiles/") or "/src/cmakefiles/" in normalized:
            return {"category": "production_compile", "dependency": None, "stage": None}
        return {"category": "other_compile", "dependency": None, "stage": None}
    if dependency is not None:
        return {"category": "dependency_build", "dependency": dependency, "stage": "build"}
    if normalized_link_outputs and normalized in normalized_link_outputs:
        return {"category": "link", "dependency": None, "stage": None}
    if normalized.endswith((".a", ".so", ".dylib", ".dll", ".exe")):
        return {"category": "link", "dependency": None, "stage": None}
    return {"category": "other", "dependency": None, "stage": None}


def classify_edge(output: str, link_outputs: set[str] | None = None) -> str:
    normalized_link_outputs = {normalize_output(item) for item in link_outputs or set()}
    return str(classify_output(output, normalized_link_outputs)["category"])


def classify_outputs(
    outputs: list[str], normalized_link_outputs: set[str] | None = None
) -> dict[str, str | None]:
    details = [classify_output(output, normalized_link_outputs) for output in outputs]
    external_stage = next(
        (detail for detail in details if detail.get("external_project_stage") is not None),
        None,
    )
    if external_stage is not None:
        return external_stage
    priority = {
        "dependency_compile": 0,
        "test_compile": 1,
        "production_compile": 2,
        "other_compile": 3,
        "dependency_build": 4,
        "link": 5,
        "build_maintenance": 6,
        "other": 7,
    }
    return min(details, key=lambda detail: priority[str(detail["category"])])


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
    dependencies: dict[str, dict[str, Any]] = {}
    translation_units: list[dict[str, Any]] = []
    if not path.is_file():
        return {
            "available": False,
            "categories": categories,
            "dependencies": dependencies,
            "slowest_translation_units": [],
            "raw_log_records": 0,
            "edge_count": 0,
            "build_edge_count": 0,
            "deduplicated_output_records": 0,
        }
    edge_records: list[dict[str, Any]] = []
    raw_log_records = 0
    normalized_link_outputs = {normalize_output(item) for item in link_outputs or set()}
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
        raw_log_records += 1
        output = fields[3]
        if len(fields) >= 5 and fields[4]:
            edge_key: tuple[Any, ...] = (start_ms, end_ms, fields[2], fields[4])
        else:
            edge_key = (start_ms, end_ms, fields[2], output)
        if edge_records and edge_records[-1]["key"] == edge_key:
            record = edge_records[-1]
        else:
            record = {
                "key": edge_key,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "outputs": [],
            }
            edge_records.append(record)
        if output not in record["outputs"]:
            record["outputs"].append(output)

    for record in edge_records:
        outputs = record["outputs"]
        duration = max(0, record["end_ms"] - record["start_ms"]) / 1000
        detail = classify_outputs(outputs, normalized_link_outputs)
        category = str(detail["category"])
        current = categories.setdefault(category, {"edges": 0, "cumulative_seconds": 0.0})
        current["edges"] = int(current["edges"]) + 1
        current["cumulative_seconds"] = round(float(current["cumulative_seconds"]) + duration, 3)

        dependency = detail["dependency"]
        stage = detail["stage"]
        if dependency is not None and stage is not None:
            dependency_metrics = dependencies.setdefault(
                str(dependency), {"edges": 0, "cumulative_seconds": 0.0, "stages": {}}
            )
            dependency_metrics["edges"] += 1
            dependency_metrics["cumulative_seconds"] = round(
                dependency_metrics["cumulative_seconds"] + duration, 3
            )
            stage_metrics = dependency_metrics["stages"].setdefault(
                str(stage), {"edges": 0, "cumulative_seconds": 0.0}
            )
            stage_metrics["edges"] += 1
            stage_metrics["cumulative_seconds"] = round(
                stage_metrics["cumulative_seconds"] + duration, 3
            )

        for output in outputs:
            if not output.lower().endswith((".o", ".obj")):
                continue
            source = compile_commands.get(output, compile_commands.get(Path(output).name, output))
            translation_units.append(
                {"source": source, "output": output, "seconds": round(duration, 3), "category": category}
            )
    translation_units.sort(key=lambda item: item["seconds"], reverse=True)
    return {
        "available": True,
        "categories": categories,
        "dependencies": dependencies,
        "slowest_translation_units": translation_units[:20],
        "raw_log_records": raw_log_records,
        "edge_count": len(edge_records),
        "build_edge_count": len(edge_records)
        - int(categories.get("build_maintenance", {}).get("edges", 0)),
        "deduplicated_output_records": raw_log_records - len(edge_records),
    }


def read_ninja_log_records(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [
        line
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line and not line.startswith("#")
    ]


def read_ninja_log_header(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("# ninja log v"):
            return line
    return None


def ninja_log_delta(before: list[str], after: list[str]) -> list[str]:
    previous = Counter(before)
    added = []
    for line in after:
        if previous[line]:
            previous[line] -= 1
        else:
            added.append(line)
    return added


def parse_dependency_resolutions(text: str) -> dict[str, dict[str, Any]]:
    resolutions: dict[str, dict[str, Any]] = {}
    pattern = re.compile(
        r"Third-party override: dependency=([^,]+), pin=([^,]+), source=([^,]+),\s*"
        r"variable=([^;\n]+);?([^\n]*)"
    )
    for match in pattern.finditer(text):
        dependency = canonical_dependency(match.group(1))
        resolutions[dependency] = {
            "resolution": "source",
            "pin": match.group(2).strip(),
            "source_origin": match.group(3).strip(),
            "override_variable": match.group(4).strip(),
            "fallback": match.group(5).strip().strip("; ") or "none reported",
        }

    if "Using system OpenBLAS as BLAS backend" in text:
        resolutions["openblas"] = {
            "resolution": "system",
            "pin": "system",
            "source_origin": "system",
            "fallback": "bundled source fallback available when policy is AUTO",
        }
    elif "Building OpenBLAS from source" in text:
        resolutions.setdefault(
            "openblas",
            {
                "resolution": "source",
                "pin": "unknown",
                "source_origin": "default",
                "fallback": "system dependency was disabled or unavailable",
            },
        )

    if "Using pre-existing fmt::fmt target" in text or "Found fmt via find_package" in text:
        resolutions["fmt"] = {
            "resolution": "system",
            "pin": "system",
            "source_origin": "system",
            "fallback": "bundled source fallback available when policy is AUTO",
        }
    if "Enabled Intel MKL as BLAS backend" in text:
        resolutions["mkl"] = {
            "resolution": "system",
            "pin": "system",
            "source_origin": "system",
            "fallback": "OpenBLAS fallback is architecture-dependent",
        }
    if "Intel MKL is not supported on this architecture" in text:
        openblas = resolutions.setdefault(
            "openblas",
            {
                "resolution": "source",
                "pin": "unknown",
                "source_origin": "default",
            },
        )
        openblas["fallback"] = "selected after Intel MKL was unavailable on this architecture"
    return resolutions


def directory_size(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path, followlinks=False):
        for filename in files:
            candidate = Path(root) / filename
            try:
                if not candidate.is_symlink():
                    total += candidate.stat().st_size
            except FileNotFoundError:
                continue
    return total


def read_cmake_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")):
            continue
        declaration, separator, value = line.partition("=")
        if not separator or ":" not in declaration:
            continue
        name, _ = declaration.split(":", 1)
        values[name] = value
    return values


def local_dependency_path(root: Path, value: str | None, default: Path) -> Path | None:
    candidate = Path(value).resolve() if value else default.resolve()
    if root.resolve() in candidate.parents:
        return candidate
    return None


def dependency_storage(
    build_dir: Path, fetchcontent_dir: Path | None, archive_dir: Path | None
) -> dict[str, int]:
    external_build_bytes = sum(
        directory_size(build_dir / dependency) for dependency in EXTERNAL_PROJECT_DEPENDENCIES
    )
    return {
        "fetchcontent_bytes": directory_size(fetchcontent_dir),
        "archive_bytes": directory_size(archive_dir),
        "external_build_bytes": external_build_bytes,
    }


def build_dependency_report(
    resolutions: dict[str, dict[str, Any]],
    ninja_dependencies: dict[str, dict[str, Any]],
    build_dir: Path,
    fetchcontent_dir: Path | None,
    cmake_cache: dict[str, str],
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for dependency in sorted(set(resolutions) | set(ninja_dependencies)):
        details = dict(resolutions.get(dependency, {}))
        details.setdefault("resolution", "source")
        details.setdefault("pin", "unknown")
        details.setdefault("source_origin", "unknown")
        details.setdefault("fallback", "none reported")
        details["display_name"] = DEPENDENCY_DISPLAY_NAMES.get(dependency, dependency)
        details["stages"] = ninja_dependencies.get(dependency, {}).get("stages", {})

        source_bytes = 0
        build_bytes = 0
        install_bytes = 0
        if details["resolution"] == "system":
            details["preparation"] = "host dependency"
        elif dependency in EXTERNAL_PROJECT_DEPENDENCIES:
            dependency_dir = build_dir / dependency
            source_bytes = directory_size(dependency_dir / "source")
            install_bytes = directory_size(dependency_dir / "install")
            build_bytes = max(
                0, directory_size(dependency_dir) - source_bytes - install_bytes
            )
            details["preparation"] = "verified ExternalProject archive"
        elif dependency in FETCHCONTENT_DIRECTORY_NAMES and fetchcontent_dir is not None:
            directory_name = FETCHCONTENT_DIRECTORY_NAMES[dependency]
            source_bytes = directory_size(fetchcontent_dir / f"{directory_name}-src")
            build_bytes = directory_size(fetchcontent_dir / f"{directory_name}-build")
            if fetchcontent_dir.name == ".ci-fetchcontent":
                details["preparation"] = "pinned FetchContent checkout"
            else:
                details["preparation"] = "FetchContent source tree"
        else:
            details["preparation"] = f"{details['resolution']} preparation not classified"
            if details["fallback"] == "none reported":
                details["fallback"] = "classification path unknown; metrics may be incomplete"

        policy = cmake_cache.get(
            f"VSAG_USE_SYSTEM_{dependency.upper()}", cmake_cache.get("VSAG_USE_SYSTEM_DEPS", "AUTO")
        )
        if dependency == "yaml_cpp":
            policy = cmake_cache.get("VSAG_USE_SYSTEM_YAML_CPP", policy)
        details["system_policy"] = policy or cmake_cache.get("VSAG_USE_SYSTEM_DEPS", "AUTO")
        if details["resolution"] == "system" and details["system_policy"] == "ON":
            details["fallback"] = "none; system dependency required"
        elif details["resolution"] == "source" and details["system_policy"] == "OFF":
            details["fallback"] = "system dependency disabled; source hash validation required"
        details["source_bytes"] = source_bytes
        details["build_bytes"] = build_bytes + install_bytes
        report[dependency] = details
    return report


def format_bytes(byte_count: int) -> str:
    if byte_count < 1024:
        return f"{byte_count} B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f} KiB"
    if byte_count < 1024 * 1024 * 1024:
        return f"{byte_count / (1024 * 1024):.1f} MiB"
    return f"{byte_count / (1024 * 1024 * 1024):.2f} GiB"


def empty_ninja_metrics(available: bool) -> dict[str, Any]:
    return {
        "available": available,
        "categories": {},
        "dependencies": {},
        "slowest_translation_units": [],
        "raw_log_records": 0,
        "edge_count": 0,
        "build_edge_count": 0,
        "deduplicated_output_records": 0,
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

    def capture_build(self, name: str, measurement_mode: str) -> None:
        ninja_log = self.build_dir / ".ninja_log"
        before_header = read_ninja_log_header(ninja_log)
        before_records = read_ninja_log_records(ninja_log)
        self.ccache("--zero-stats")
        phase = self.run_logged(
            name,
            [
                "make",
                "build-asan",
                f"DEBUG_BUILD_DIR={self.build_dir}",
                f"COMPILE_JOBS={self.args.jobs}",
                "CMAKE_BUILD_ARGS=-d stats",
            ],
        )
        captured_log = self.output_dir / f"{name}.ninja_log"
        captured_log.unlink(missing_ok=True)
        if ninja_log.is_file():
            added_records = ninja_log_delta(before_records, read_ninja_log_records(ninja_log))
            header = read_ninja_log_header(ninja_log) or before_header
            captured_log.write_text(
                (f"{header}\n" if header else "")
                + "\n".join(added_records)
                + ("\n" if added_records else ""),
                encoding="utf-8",
            )
        phase["ccache"] = self.ccache("--print-stats")
        if captured_log.is_file():
            phase["ninja"] = parse_ninja_log(
                captured_log,
                read_compile_commands(self.build_dir / "compile_commands.json"),
                read_link_outputs(self.build_dir),
            )
        else:
            phase["ninja"] = empty_ninja_metrics(available=False)
        phase["measurement_mode"] = measurement_mode
        available = phase["ninja"]["available"]
        phase["ninja_edges"] = phase["ninja"]["edge_count"] if available else None
        phase["build_edges"] = phase["ninja"]["build_edge_count"] if available else None
        if measurement_mode == "no-op":
            phase["true_noop"] = available and phase["exit_code"] == 0 and phase["build_edges"] == 0

    def collect(self) -> int:
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        if self.args.clear_ccache:
            self.ccache("--clear")
        self.ccache("--zero-stats")
        configure = self.run_logged(
            "configure",
            [
                "make", "configure-asan",
                f"DEBUG_BUILD_DIR={self.build_dir}",
                f"COMPILE_JOBS={self.args.jobs}",
            ],
        )
        configure["ccache"] = self.ccache("--print-stats")
        if not self.failure_code:
            self.capture_build("clean_build", "clean")
        if not self.failure_code:
            self.run_logged(
                "prepare_warm_cache_build",
                ["cmake", "--build", str(self.build_dir), "--target", "clean"],
            )
            if not self.failure_code:
                self.capture_build("warm_cache_build", "warm-cache")
        if not self.failure_code:
            self.capture_build("noop_build", "no-op")
        self.write_reports()
        return self.failure_code

    def write_reports(self) -> None:
        fetchcontent_dir = local_dependency_path(
            self.root,
            os.environ.get("VSAG_FETCHCONTENT_BASE_DIR"),
            self.build_dir / "_deps",
        )
        archive_dir = local_dependency_path(
            self.root,
            os.environ.get("VSAG_THIRDPARTY_DOWNLOAD_DIR"),
            self.build_dir / ".vsag-downloads",
        )
        configure_log = self.output_dir / "configure.log"
        configure_text = (
            configure_log.read_text(encoding="utf-8", errors="replace")
            if configure_log.is_file()
            else ""
        )
        clean_phase = next(
            (phase for phase in self.phases if phase["name"] == "clean_build"), {}
        )
        cmake_cache = read_cmake_cache(self.build_dir / "CMakeCache.txt")
        dependencies = build_dependency_report(
            parse_dependency_resolutions(configure_text),
            clean_phase.get("ninja", {}).get("dependencies", {}),
            self.build_dir,
            fetchcontent_dir,
            cmake_cache,
        )
        storage = dependency_storage(self.build_dir, fetchcontent_dir, archive_dir)
        source_preparation_seconds = getattr(
            self.args, "dependency_source_preparation_seconds", 0.0
        )
        cache_restore_seconds = getattr(self.args, "dependency_cache_restore_seconds", 0.0)
        preparation_seconds = getattr(self.args, "dependency_preparation_seconds", 0.0)
        if not preparation_seconds:
            preparation_seconds = source_preparation_seconds + cache_restore_seconds
        report = {
            "schema_version": 2,
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
                "dependency_cache_matched_key": getattr(
                    self.args, "dependency_cache_matched_key", ""
                ),
                "dependency_cache_state": getattr(
                    self.args, "dependency_cache_state", "unknown"
                ),
                "dependency_cache_write_policy": getattr(
                    self.args, "dependency_cache_write_policy", "unknown"
                ),
                "dependency_preparation_seconds": preparation_seconds,
                "dependency_source_preparation_seconds": source_preparation_seconds,
                "dependency_cache_restore_seconds": cache_restore_seconds,
                "dependency_storage": storage,
            },
            "dependencies": dependencies,
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


def dependency_stage_text(stages: dict[str, dict[str, Any]]) -> str:
    if not stages:
        return "n/a"
    order = [
        "mkdir",
        "download",
        "update",
        "patch",
        "configure",
        "compile",
        "build",
        "install",
        "test",
        "done",
    ]
    names = [stage for stage in order if stage in stages]
    names.extend(sorted(set(stages) - set(names)))
    return "; ".join(
        f"{stage} {float(stages[stage]['cumulative_seconds']):.3f} s" for stage in names
    )


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict[str, Any], concise: bool) -> str:
    config = report["configuration"]
    phases = {phase["name"]: phase for phase in report["phases"]}
    lines = ["## Build performance metrics", ""]
    lines.extend(
        [
            f"Status: **{report['status']}** · compiler: `{config['compiler']}` · parallel jobs: `{config['jobs']}`",
            "",
            "| Phase | Wall time | Peak RSS | ccache hits | ccache misses | Ninja edges |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    clean_cache_state = (
        "cold ccache" if config.get("cold_ccache_cleared") else "preserved ccache"
    )
    for name, label in (
        ("configure", "CMake configure"),
        ("clean_build", f"Clean build ({clean_cache_state})"),
        ("warm_cache_build", "Warm-cache rebuild (clean outputs)"),
        ("noop_build", "No-op build"),
    ):
        phase = phases.get(name)
        if phase is None:
            continue
        if name == "noop_build":
            if phase.get("true_noop"):
                label = "True no-op build (verified; maintenance only)"
            elif phase.get("ninja_edges") is not None:
                label = f"Expected no-op build ({phase.get('build_edges', 'unknown')} build edges ran)"
        ccache = phase.get("ccache", {})
        peak = phase.get("peak_rss_kib")
        peak_text = f"{peak / 1024:.1f} MiB" if peak is not None else "n/a"
        edge_count: Any = phase.get("ninja_edges")
        if edge_count is None:
            edge_count = "n/a"
        build_edges = phase.get("build_edges")
        if build_edges is not None and build_edges != edge_count:
            edge_count = f"{edge_count} total / {build_edges} build"
        lines.append(
            f"| {label} | {phase['elapsed_seconds']:.3f} s | {peak_text} | "
            f"{ccache.get('cache_hit', 'n/a')} | {ccache.get('cache_miss', 'n/a')} | "
            f"{edge_count} |"
        )

    storage = config.get("dependency_storage", {})
    cache_state = config.get("dependency_cache_state", "unknown").replace("-", " ")
    write_policy = config.get("dependency_cache_write_policy", "unknown").replace("-", " ")
    lines.extend(
        [
            "",
            "### Dependency preparation",
            "",
            "| Item | State | Wall time | Current size |",
            "| --- | --- | ---: | ---: |",
            f"| Pinned FetchContent sources | prepared checkout | {config.get('dependency_source_preparation_seconds', 0.0):.3f} s | {format_bytes(int(storage.get('fetchcontent_bytes', 0)))} |",
            f"| ExternalProject source archives | {cache_state}; {write_policy} | {config.get('dependency_cache_restore_seconds', 0.0):.3f} s | {format_bytes(int(storage.get('archive_bytes', 0)))} |",
            f"| ExternalProject source/build/install trees | source build | included in build phases | {format_bytes(int(storage.get('external_build_bytes', 0)))} |",
            "",
            f"Total dependency preparation before configure: **{config['dependency_preparation_seconds']:.3f} s**",
            "",
        ]
    )
    matched_key = config.get("dependency_cache_matched_key", "")
    lines.append(f"Compiler cache key: `{config['compiler_cache_key']}`  ")
    lines.append(f"Dependency archive cache key: `{config['dependency_cache_key']}`  ")
    if matched_key:
        lines.append(f"Dependency archive matched key: `{matched_key}`")
    lines.append("")

    dependencies = report.get("dependencies", {})
    if dependencies:
        lines.extend(
            [
                "### Dependency resolution",
                "",
                "| Dependency | Pin | Resolution | Preparation / selection | Cumulative Ninja stages | Local size | Fallback |",
                "| --- | --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for dependency in sorted(
            dependencies.values(), key=lambda item: item["display_name"].lower()
        ):
            source_origin = dependency.get("source_origin", "unknown")
            preparation = dependency.get("preparation", "unknown")
            if dependency.get("resolution") == "source":
                preparation = f"{preparation} ({source_origin})"
            local_size = int(dependency.get("source_bytes", 0)) + int(
                dependency.get("build_bytes", 0)
            )
            lines.append(
                f"| {markdown_cell(dependency['display_name'])} | "
                f"`{markdown_cell(dependency.get('pin', 'unknown'))}` | "
                f"{markdown_cell(dependency.get('resolution', 'unknown'))} | "
                f"{markdown_cell(preparation)} | "
                f"{markdown_cell(dependency_stage_text(dependency.get('stages', {})))} | "
                f"{format_bytes(local_size)} | {markdown_cell(dependency.get('fallback', 'none reported'))} |"
            )
        lines.append("")

    clean = phases.get("clean_build", {})
    if clean:
        lines.extend(
            [
                "Clean-build cumulative Ninja edge time (parallel edges overlap):",
                "",
                f"- Dependencies: {metric(clean, 'dependency_compile') + metric(clean, 'dependency_build'):.3f} s",
                f"- VSAG production compile: {metric(clean, 'production_compile'):.3f} s",
                f"- Test compile: {metric(clean, 'test_compile'):.3f} s",
                f"- Link: {metric(clean, 'link'):.3f} s",
                f"- Multi-output records deduplicated: {clean.get('ninja', {}).get('deduplicated_output_records', 0)}",
                "",
            ]
        )
    if concise:
        lines.append("The `build-performance-metrics` artifact contains JSON, complete logs, Ninja statistics, and the slowest translation units.")
        return "\n".join(lines) + "\n"
    slowest = clean.get("ninja", {}).get("slowest_translation_units", [])
    lines.extend(
        [
            "### Slowest translation units (clean build)",
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
    parser.add_argument("--dependency-cache-matched-key", default="")
    parser.add_argument("--dependency-cache-state", default="unknown")
    parser.add_argument("--dependency-cache-write-policy", default="unknown")
    parser.add_argument("--dependency-preparation-seconds", type=float, default=0)
    parser.add_argument("--dependency-source-preparation-seconds", type=float, default=0)
    parser.add_argument("--dependency-cache-restore-seconds", type=float, default=0)
    parser.add_argument("--clear-ccache", action="store_true")
    return parser.parse_args()


def main() -> int:
    return Collector(parse_args()).collect()


if __name__ == "__main__":
    raise SystemExit(main())
