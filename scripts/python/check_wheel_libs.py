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

"""Fails if a pyvsag wheel bundles or links libraries it must not depend on.

The wheel has to work on a plain manylinux system, so it must not ship or require
libgfortran, libquadmath, libaio or liburing (only libgomp is vendored on purpose).
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

FORBIDDEN = ("libgfortran", "libquadmath", "libaio", "liburing")
NEEDED_RE = re.compile(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]")


def is_forbidden(name):
    base = os.path.basename(name)
    return any(base.startswith(prefix) for prefix in FORBIDDEN)


def needed_libraries(readelf_output):
    return NEEDED_RE.findall(readelf_output)


def check_wheel(path):
    """Returns a list of problems found in one wheel."""
    problems = []
    readelf = shutil.which("readelf")
    with zipfile.ZipFile(path) as wheel, tempfile.TemporaryDirectory() as tmp:
        for name in wheel.namelist():
            if is_forbidden(name):
                problems.append(f"{os.path.basename(path)}: bundles {name}")
            if ".so" not in os.path.basename(name):
                continue
            if readelf is None:
                problems.append(f"readelf not found; could not verify NEEDED entries of {name}")
                continue
            extracted = wheel.extract(name, tmp)
            env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
            proc = subprocess.run(
                [readelf, "-d", extracted], capture_output=True, text=True, check=False, env=env
            )
            if proc.returncode != 0:
                problems.append(f"{os.path.basename(path)}: readelf failed on {name}")
                continue
            for lib in needed_libraries(proc.stdout):
                if is_forbidden(lib):
                    problems.append(f"{os.path.basename(path)}: {name} needs {lib}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", help="wheel files or glob patterns")
    args = parser.parse_args()

    paths = []
    for pattern in args.wheels:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        print(f"error: no wheel matches {args.wheels}")
        return 1

    problems = []
    for path in paths:
        found = check_wheel(path)
        print(f"{os.path.basename(path)}: {'FAILED' if found else 'ok'}")
        problems.extend(found)
    for problem in problems:
        print(f"error: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
