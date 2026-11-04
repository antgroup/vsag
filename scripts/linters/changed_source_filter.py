#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence


def get_changed_sources(base: str, head: str) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "-z",
            f"{base}..{head}",
            "--",
            "src/**/*.cpp",
            ":(exclude)src/**/*_test.cpp",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [path.decode("utf-8") for path in result.stdout.split(b"\0") if path]


def build_filter(paths: Sequence[str]) -> str:
    return "|".join(f"{re.escape(path)}$" for path in paths)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a clang-tidy file filter for changed production C++ sources."
    )
    parser.add_argument("base", help="Base Git revision")
    parser.add_argument("head", nargs="?", default="HEAD", help="Head Git revision")
    args = parser.parse_args()

    try:
        paths = get_changed_sources(args.base, args.head)
    except subprocess.CalledProcessError as error:
        print(f"Failed to determine changed source files: {error}", file=sys.stderr)
        return 1

    if paths:
        print(f"Linting {len(paths)} changed source file(s):", file=sys.stderr)
        for path in paths:
            print(f"  {path}", file=sys.stderr)
    else:
        print("No changed production C++ source files to lint.", file=sys.stderr)

    pattern = build_filter(paths)
    if paths and not pattern:
        print("Changed source files were found, but the lint filter is empty.", file=sys.stderr)
        return 1

    print(pattern)
    return 0


if __name__ == "__main__":
    sys.exit(main())
