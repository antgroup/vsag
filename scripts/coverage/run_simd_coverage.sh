#!/usr/bin/env bash
set -euo pipefail

# Run twice sequentially on one runner, with separate clean build trees. No gcda,
# dependency archives or compiler caches are restored from another run.
root=$(pwd -P)
base=$1
head=$2
mkdir -p simd-coverage
# Container checkouts can belong to the runner's UID. Checkout's temporary Git
# config is not inherited here. Trust only these paths for this process and its
# children (including the comparator), without changing global configuration.
for directory in "$root" "$root/simd-base" "$root/simd-head"; do
    config_index=${GIT_CONFIG_COUNT:-0}
    export "GIT_CONFIG_KEY_$config_index=safe.directory"
    export "GIT_CONFIG_VALUE_$config_index=$directory"
    export GIT_CONFIG_COUNT=$((config_index + 1))
done
if ! git rev-parse --show-toplevel ||
   ! git rev-parse --verify "$base^{commit}" ||
   ! git rev-parse --verify "$head^{commit}"; then
    echo 'SIMD coverage unavailable: repository or revision validation failed.' > simd-coverage/summary.md
    exit 1
fi
# Both builds consume the pinned source checkout prepared by the workflow. If
# those pins change, reusing it for the base would measure the wrong dependency.
if git diff --quiet "$base" "$head" -- extern .github/actions/prepare-fetchcontent; then
    :
else
    diff_status=$?
    if [ "$diff_status" -eq 1 ]; then
        echo 'SIMD coverage unavailable: dependency definitions differ; a comparable baseline requires separate dependency preparation.' > simd-coverage/summary.md
    else
        echo 'SIMD coverage unavailable: Git dependency comparison failed.' > simd-coverage/summary.md
    fi
    exit "$diff_status"
fi
for role in base head; do
    sha=$base
    if [ "$role" = head ]; then sha=$head; fi
    git worktree add --detach "simd-$role" "$sha"
    (
        cd "simd-$role"
        export CCACHE_DISABLE=1
        export VSAG_FETCHCONTENT_BASE_DIR="$root/.ci-fetchcontent"
        export VSAG_THIRDPARTY_DOWNLOAD_DIR="$PWD/.ci-downloads"
        export CMAKE_GENERATOR=Ninja
        export EXTRA_DEFINED=-DVSAG_USE_SYSTEM_OPENBLAS=ON
        make cov COMPILE_JOBS=4
        python3 "$root/scripts/coverage/verify_cpp_coverage.py" instrumentation \
            build/compile_commands.json --source-root "$PWD"
        timeout 15m ./build/tests/unittests '[simd]~[!benchmark]' --rng-seed 424242 --order lex
        bash scripts/coverage/collect_cpp_coverage.sh
        cp coverage/coverage.info "$root/simd-coverage/$role.info"
        python3 - "$root" "$role" "$sha" <<'PY'
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "scripts/coverage"))
from compare_simd_coverage import PROFILE
path = Path(sys.argv[1]) / "simd-coverage" / (sys.argv[2] + ".json")
path.write_text(json.dumps({"sha": sys.argv[3], "profile": PROFILE}) + "\n")
PY
    ) 2>&1 | tee "simd-coverage/$role.log"
    # Retain the trace/log, not two multi-gigabyte build trees on the runner.
    git worktree remove --force "simd-$role"
done
python3 scripts/coverage/compare_simd_coverage.py --root "$root" \
    --base "$base" --head "$head" --reports simd-coverage
