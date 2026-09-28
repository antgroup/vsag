#!/usr/bin/env bash
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

# Runs the rpt_bench experiment matrix. Usage:
#
#   tools/rpt_bench/run_experiments.sh [experiment ...]
#
# With no arguments every experiment runs. Experiments: main seeds scale dims deep10m
# peakmem ksens. Override paths with DATA_DIR, OUT_DIR and RPT_BENCH.

set -euo pipefail

DATA_DIR="${DATA_DIR:-$HOME/code/datasets}"
OUT_DIR="${OUT_DIR:-$HOME/code/rpt-results}"
# Resolve through symlinks so the script works when invoked via a shortcut.
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
RPT_BENCH="${RPT_BENCH:-$SCRIPT_DIR/../../build/tools/rpt_bench/rpt_bench}"

SIFT="$DATA_DIR/sift-128-euclidean.hdf5"
GIST="$DATA_DIR/gist-960-euclidean.hdf5"
DEEP="$DATA_DIR/deep-image-96-angular.hdf5"

mkdir -p "$OUT_DIR"

run() {
    # run <output-name> <rpt_bench args...>
    local name="$1"
    shift
    echo "==> $name"
    "$RPT_BENCH" "$@" -o "$OUT_DIR/$name.json" 2>&1 | grep -vE "^cpu |====|running on|instance spec|\[info\]|\[debug\]"
    echo
}

# 1. Main comparison: two datasets x three bucket sizes x three balanced strategies,
#    plus KMeans as the locality upper bound that ignores size constraints.
exp_main() {
    run sift1m -d "$SIFT" -L 100,1000,10000 -s rpt,random,single_dim -k 10
    run gist1m -d "$GIST" -L 100,1000,10000 -s rpt,random,single_dim -k 10
    run sift1m-kmeans -d "$SIFT" -L 1000,10000 -s kmeans -k 10
}

# 2. Seed stability: the same configuration under five seeds. Report mean and stddev of
#    the locality metrics; the uniformity metrics must be identical across seeds.
exp_seeds() {
    for seed in 0 1 2 3 4; do
        run "sift1m-seed$seed" -d "$SIFT" -L 1000 -s rpt,random -k 10 --seed "$seed"
        run "gist1m-seed$seed" -d "$GIST" -L 1000 -s rpt,random -k 10 --seed "$seed"
    done
}

# 3. Scale: build time and memory as the number of vectors grows (SIFT prefix subsets).
exp_scale() {
    for n in 100000 250000 500000 1000000; do
        run "sift-scale-$n" -d "$SIFT" -L 1000 -s rpt,random,single_dim -k 10 --max_base "$n"
    done
}

# 4. Dimensionality: the same N=1M and L across 96, 128 and 960 dimensions.
exp_dims() {
    run deep1m -d "$DEEP" -L 1000,10000 -s rpt,random,single_dim -k 10 --max_base 1000000
    # SIFT (128) and GIST (960) at 1M come from the main experiment.
}

# 5. Ten million vectors: the full deep-image subset. Needs roughly 6 GB of RAM.
exp_deep10m() {
    run deep10m -d "$DEEP" -L 10000 -s rpt,random,single_dim -k 10
}

# 6. Exact peak memory: one strategy per process, because the kernel only exposes a
#    high-water mark and later strategies in the same process would be under-reported.
exp_peakmem() {
    for s in rpt random single_dim kmeans; do
        run "sift1m-peak-$s" -d "$SIFT" -L 1000 -s "$s" -k 10
    done
}

# 7. Sensitivity to k: the locality conclusion should hold for 10, 50 and 100 neighbours.
exp_ksens() {
    for k in 10 50 100; do
        run "sift1m-k$k" -d "$SIFT" -L 1000 -s rpt,random,single_dim -k "$k"
    done
}

if [ "$#" -eq 0 ]; then
    set -- main seeds scale dims deep10m peakmem ksens
fi
for exp in "$@"; do
    "exp_$exp"
done
echo "all results in $OUT_DIR"
