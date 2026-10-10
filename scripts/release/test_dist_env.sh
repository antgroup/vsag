#!/bin/bash
# Focused packaging argument tests; no Docker daemon or package build required.
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
tmp=$(mktemp -d)
trap 'chmod -R u+rwX "$tmp"; rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin" "$tmp/work space"
export DIST_TEST_LOG="$tmp/docker.jsonl"
export PATH="$tmp/bin:$PATH"

# Model Docker's literal env-file parsing and image-default replacement. Only
# explicit run options cross the container boundary; host variables do not.
cat > "$tmp/bin/docker" <<'MOCK'
#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
record = {"args": args}
if args[0] == "run":
    values = {"VSAG_THIRDPARTY_FMT": "https://image.invalid/fmt"}
    i = 1
    while i < len(args):
        arg = args[i]
        if arg in ("-u", "-v"):
            i += 2
        elif arg == "--rm":
            i += 1
        elif arg == "--env-file":
            for line in Path(args[i + 1]).read_text().splitlines():
                line = line.lstrip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    values[key] = value
            i += 2
        else:
            break
    assert args[i].startswith("vsag-builder-")
    assert args[i + 1:i + 3] == ["bash", "-c"]
    assert "make run-dist-tests" in args[i + 3]
    record["env"] = values
    Path("dist").mkdir(exist_ok=True)
    Path("dist/vsag.tar.gz").touch()
with open(os.environ["DIST_TEST_LOG"], "a") as log:
    log.write(json.dumps(record) + "\n")
MOCK
cat > "$tmp/bin/git" <<'MOCK'
#!/bin/bash
printf '%s\n' v-test
MOCK
chmod +x "$tmp/bin/docker" "$tmp/bin/git"
cd "$tmp/work space"

run_dist() {
    : > "$DIST_TEST_LOG"
    bash "$repo/scripts/release/dist.sh" > "$tmp/output" 2>&1
}

export VSAG_THIRDPARTY_FMT=https://host.invalid/fmt
export VSAG_THIRDPARTY_FMT_10_2_1=https://host.invalid/pinned
unset VSAG_THIRDPARTY_ENV_FILE
run_dist
python3 - <<'PY'
import json, os
records = [json.loads(line) for line in open(os.environ["DIST_TEST_LOG"])]
assert [r["args"][0] for r in records] == ["build", "run", "build", "run"]
for r in records:
    assert "--env-file" not in r["args"]
    if r["args"][0] == "run":
        assert r["env"] == {"VSAG_THIRDPARTY_FMT": "https://image.invalid/fmt"}
PY

export VSAG_THIRDPARTY_ENV_FILE="$tmp/dependency urls.env"
cat > "$VSAG_THIRDPARTY_ENV_FILE" <<'ENV'
# Dummy URLs only. Shell-looking text must remain literal.

VSAG_THIRDPARTY_FMT=https://mirror.invalid/fmt?x=a=b&y=c#fragment
VSAG_THIRDPARTY_FMT_10_2_1=https://mirror.invalid/pinned?x=$HOME&y=$(touch${IFS}sentinel)#hash
VSAG_THIRDPARTY_SPDLOG_COMMIT_0123456789AB=https://mirror.invalid/commit
VSAG_THIRDPARTY_SPDLOG_TAG_TEST_H0123456789AB=https://mirror.invalid/tag
ENV
# CRLF and a final line without a newline are both supported by Docker.
printf '\r\nVSAG_THIRDPARTY_OPENBLAS_0_3_34=https://mirror.invalid/blas?x=1&y=2#hash' \
    >> "$VSAG_THIRDPARTY_ENV_FILE"
run_dist
[[ ! -e sentinel ]]
[[ "$VSAG_THIRDPARTY_FMT" == https://host.invalid/fmt ]]
[[ "$VSAG_THIRDPARTY_FMT_10_2_1" == https://host.invalid/pinned ]]
python3 - <<'PY'
import json, os
records = [json.loads(line) for line in open(os.environ["DIST_TEST_LOG"])]
assert [r["args"][0] for r in records] == ["build", "run", "build", "run"]
expected = dict(line.split("=", 1) for line in
                open(os.environ["VSAG_THIRDPARTY_ENV_FILE"]).read().splitlines()
                if line and not line.startswith("#"))
for r in records:
    args = r["args"]
    if args[0] == "build":
        assert "--env-file" not in args
    else:
        assert args[args.index("--env-file") + 1] == os.environ["VSAG_THIRDPARTY_ENV_FILE"]
        assert args[args.index("-v") + 1] == os.getcwd() + ":/work"
        assert r["env"] == expected
PY
! grep -q 'mirror.invalid' "$tmp/output"

# Exercise the actual CMake resolver with the environments captured at both
# container boundaries, including its pinned > legacy > default precedence.
cat > "$tmp/check.cmake" <<'CMAKE'
include("$ENV{DIST_TEST_REPO}/cmake/VSAGThirdPartyOverride.cmake")
set(urls "https://default.invalid/fmt")
vsag_resolve_thirdparty_override("fmt" "10.2.1" urls)
list(GET urls 0 actual)
if(NOT actual STREQUAL "$ENV{DIST_TEST_EXPECTED}")
    message(FATAL_ERROR "Unexpected dependency URL precedence")
endif()
list(GET urls -1 fallback)
if(NOT fallback STREQUAL "https://default.invalid/fmt")
    message(FATAL_ERROR "Default download fallback lost")
endif()
CMAKE
DIST_TEST_REPO="$repo" DIST_TEST_CMAKE="$tmp/check.cmake" python3 - <<'PYTEST'
import json, os, subprocess
for record in map(json.loads, open(os.environ["DIST_TEST_LOG"])):
    if "env" not in record:
        continue
    env = {"PATH": os.environ["PATH"], "DIST_TEST_REPO": os.environ["DIST_TEST_REPO"]}
    env.update(record["env"])
    for key in ("VSAG_THIRDPARTY_FMT_10_2_1", "VSAG_THIRDPARTY_FMT", None):
        env["DIST_TEST_EXPECTED"] = env[key] if key else "https://default.invalid/fmt"
        result = subprocess.run(["cmake", "-P", os.environ["DIST_TEST_CMAKE"]],
                                env=env, capture_output=True)
        assert result.returncode == 0, "CMake precedence check failed"
        if key:
            del env[key]
PYTEST

expect_failure() {
    if run_dist; then
        echo "Expected env-file validation failure" >&2
        exit 1
    fi
    [[ ! -s "$DIST_TEST_LOG" ]]
    grep -q 'VSAG_THIRDPARTY_ENV_FILE' "$tmp/output"
    ! grep -q 'private.invalid' "$tmp/output"
}

VSAG_THIRDPARTY_ENV_FILE="$tmp/missing" expect_failure
VSAG_THIRDPARTY_ENV_FILE="$tmp" expect_failure
VSAG_THIRDPARTY_ENV_FILE='' expect_failure
mkfifo "$tmp/pipe"
VSAG_THIRDPARTY_ENV_FILE="$tmp/pipe" expect_failure
chmod 000 "$VSAG_THIRDPARTY_ENV_FILE"
if [[ $(id -u) != 0 ]]; then
    expect_failure
else
    echo 'SKIP unreadable-file check: root can read mode-000 files'
fi
chmod 600 "$VSAG_THIRDPARTY_ENV_FILE"

for entry in \
    'VSAG_THIRDPARTY_FMT' \
    'VSAG_THIRDPARTY_FMT=' \
    'export VSAG_THIRDPARTY_FMT=https://private.invalid/a' \
    'OTHER=https://private.invalid/a' \
    'VSAG_THIRDPARTY_fmt=https://private.invalid/a' \
    'VSAG_THIRDPARTY_FMT="https://private.invalid/a"' \
    'VSAG_THIRDPARTY_FMT=https://private.invalid/a #comment' \
    ' VSAG_THIRDPARTY_FMT=https://private.invalid/a'; do
    printf '%s\n' "$entry" > "$VSAG_THIRDPARTY_ENV_FILE"
    expect_failure
done
printf '%s\n' 'VSAG_THIRDPARTY_FMT=https://private.invalid/a' \
    'VSAG_THIRDPARTY_FMT=https://private.invalid/b' > "$VSAG_THIRDPARTY_ENV_FILE"
expect_failure
printf 'VSAG_THIRDPARTY_FMT=https://private.invalid/\0hidden\n' > "$VSAG_THIRDPARTY_ENV_FILE"
expect_failure
printf '\377\n' > "$VSAG_THIRDPARTY_ENV_FILE"
expect_failure
python3 - <<'PY'
import os
with open(os.environ["VSAG_THIRDPARTY_ENV_FILE"], "w") as f:
    f.write("VSAG_THIRDPARTY_FMT=https://private.invalid/" + "a" * 65536)
PY
expect_failure
: > "$VSAG_THIRDPARTY_ENV_FILE"
run_dist
printf '%s\n' 'dist env-file tests passed'
