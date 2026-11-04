#!/bin/bash

# Smoke-test pyvsag.analyze_index() and the pyvsag-analyze-index command of a built wheel in
# clean environments, outside the source tree:
#   1. a fresh virtual environment that contains only the wheel and its dependencies, run with
#      an empty environment and PATH limited to that virtual environment;
#   2. a python:<version>-slim container, which has no NumPy, CMake or C++ compiler.
#
# Usage: scripts/python/wheel_smoke_test.sh WHEELHOUSE [PYTHON]
# The container step runs when Docker is available; set VSAG_SMOKE_REQUIRE_DOCKER=1 to make a
# missing Docker daemon an error.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 WHEELHOUSE [PYTHON]" >&2
    exit 2
fi

wheelhouse=$(cd "$1" && pwd)
python=${2:-python3}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
smoke_script="$repo/tests/python/analyze_index_smoke.py"
version=$("$python" -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))')
tag="cp${version/./}"

wheels=("$wheelhouse"/pyvsag-*-"$tag"-*.whl)
if [ ! -f "${wheels[0]}" ]; then
    echo "No pyvsag wheel for $tag in $wheelhouse" >&2
    exit 1
fi
wheel=${wheels[0]}
echo "Smoke-testing $(basename "$wheel")"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# Index and query files are written in a separate environment that also has NumPy.
"$python" -m venv "$work/builder"
"$work/builder/bin/pip" install --quiet --disable-pip-version-check "$wheel" numpy
(cd "$work" && "$work/builder/bin/python" -I "$smoke_script" fixtures "$work/fixtures")

"$python" -m venv "$work/clean"
"$work/clean/bin/pip" install --quiet --disable-pip-version-check "$wheel"
(cd "$work" && env -i HOME="$work" PATH="$work/clean/bin" \
    "$work/clean/bin/python" -I "$smoke_script" check "$work/fixtures" --bare)

if command -v docker > /dev/null 2>&1 && docker info > /dev/null 2>&1; then
    docker run --rm \
        -v "$wheelhouse:/wheelhouse:ro" \
        -v "$work/fixtures:/fixtures:ro" \
        -v "$smoke_script:/analyze_index_smoke.py:ro" \
        "python:${version}-slim" \
        sh -c "pip install --quiet --disable-pip-version-check /wheelhouse/$(basename "$wheel") \
               && python -I /analyze_index_smoke.py check /fixtures --bare"
elif [ "${VSAG_SMOKE_REQUIRE_DOCKER:-0}" = "1" ]; then
    echo "Docker is required for the container smoke test but is not available" >&2
    exit 1
else
    echo "Docker is not available; skipped the python:${version}-slim container check"
fi
