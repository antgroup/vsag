#!/usr/bin/env python3

import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys


PRODUCTION_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}


def fail(message: str) -> None:
    print(f"coverage verification failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def verify_trace(trace_file: Path, source_root: Path) -> None:
    if not trace_file.is_file() or trace_file.stat().st_size == 0:
        fail(f"trace file is missing or empty: {trace_file}")

    source_files: list[str] = []
    branch_records = 0
    with trace_file.open(encoding="utf-8") as trace:
        for raw_line in trace:
            line = raw_line.rstrip("\n")
            if line.startswith("SF:"):
                source_files.append(line.removeprefix("SF:"))
            elif line.startswith("BRDA:"):
                branch_records += 1

    if not source_files:
        fail("trace file contains no source records")

    absolute_sources = [source for source in source_files if PurePosixPath(source).is_absolute()]
    if absolute_sources:
        fail("trace contains absolute source paths: " + ", ".join(absolute_sources[:5]))

    unexpected_sources = []
    for source in source_files:
        path = PurePosixPath(source)
        if not path.parts or path.parts[0] not in {"src", "include"} or ".." in path.parts:
            unexpected_sources.append(source)
    if unexpected_sources:
        fail("trace contains non-production sources: " + ", ".join(unexpected_sources[:5]))

    try:
        tracked_sources = set(
            subprocess.check_output(
                ["git", "-C", str(source_root), "ls-files", "--", "src", "include"],
                text=True,
            ).splitlines()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"cannot read tracked source inventory from {source_root}: {error}")

    untracked_sources = [source for source in source_files if source not in tracked_sources]
    if untracked_sources:
        fail("trace contains generated or untracked sources: " + ", ".join(untracked_sources[:5]))

    if branch_records == 0:
        fail("trace contains no branch records; branch coverage was lost during filtering")

    print(
        f"verified {len(source_files)} repository production source records "
        f"and {branch_records} branch records"
    )


def source_path(entry: dict[str, object], compile_commands: Path) -> Path:
    file_value = entry.get("file")
    if not isinstance(file_value, str):
        fail(f"compile command without a file in {compile_commands}")

    path = Path(file_value)
    if path.is_absolute():
        return path.resolve()

    directory_value = entry.get("directory")
    if not isinstance(directory_value, str):
        fail(f"relative compile command without a directory in {compile_commands}")
    return (Path(directory_value) / path).resolve()


def command_arguments(entry: dict[str, object], compile_commands: Path) -> list[str]:
    arguments = entry.get("arguments")
    if isinstance(arguments, list) and all(isinstance(argument, str) for argument in arguments):
        return arguments

    command = entry.get("command")
    if isinstance(command, str):
        return shlex.split(command)

    fail(f"compile command has neither arguments nor command in {compile_commands}")


def verify_instrumentation(compile_commands: Path, source_root: Path) -> None:
    try:
        database = json.loads(compile_commands.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {compile_commands}: {error}")

    if not isinstance(database, list):
        fail(f"compile command database is not a list: {compile_commands}")

    production_root = (source_root / "src").resolve()
    production_commands: list[tuple[Path, list[str]]] = []
    for raw_entry in database:
        if not isinstance(raw_entry, dict):
            fail(f"compile command entry is not an object in {compile_commands}")
        path = source_path(raw_entry, compile_commands)
        try:
            path.relative_to(production_root)
        except ValueError:
            continue
        if path.suffix not in PRODUCTION_SOURCE_SUFFIXES or path.name.endswith("_test.cpp"):
            continue
        production_commands.append((path, command_arguments(raw_entry, compile_commands)))

    if not production_commands:
        fail(f"no maintained production compile commands found below {production_root}")

    missing = [
        str(path.relative_to(source_root))
        for path, arguments in production_commands
        if "--coverage" not in arguments
    ]
    if missing:
        fail("maintained production sources lack --coverage: " + ", ".join(missing[:10]))

    print(f"verified coverage instrumentation for {len(production_commands)} compile commands")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    trace_parser = subparsers.add_parser("trace")
    trace_parser.add_argument("trace_file", type=Path)
    trace_parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[2]
    )

    instrumentation_parser = subparsers.add_parser("instrumentation")
    instrumentation_parser.add_argument("compile_commands", type=Path)
    instrumentation_parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.mode == "trace":
        verify_trace(arguments.trace_file, arguments.source_root.resolve())
    else:
        verify_instrumentation(arguments.compile_commands, arguments.source_root.resolve())


if __name__ == "__main__":
    main()
