#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-release-mci-add}"
DATA="${DATA:-/root/data/codefilter-3m-384-angular-f32.hdf5}"
KNNG_PATH="${KNNG_PATH:-/root/data/codefilter-3m-384-angular/clique_index/knng200_cliqueMax50_mcs200/knng_200.bin}"
LOG="${LOG:-/tmp/hgraph_mci_half_build_add.log}"

LIMIT="${LIMIT:-0}"
QUERY_COUNT="${QUERY_COUNT:-0}"
BUILD_RATIO="${BUILD_RATIO:-0.5}"
ADD_BATCH_SIZE="${ADD_BATCH_SIZE:-100000}"
BUILD_THREAD_COUNT="${BUILD_THREAD_COUNT:-100}"
THREADS="${THREADS:-16}"
EF_SEARCH="${EF_SEARCH:-120}"
SEARCH_MCI_SEED_RATIO="${SEARCH_MCI_SEED_RATIO:-0.002}"
MCI_MCS="${MCI_MCS:-200}"
MCI_CLIQUE_MAX="${MCI_CLIQUE_MAX:-50}"
MCI_ALPHA="${MCI_ALPHA:-1.2}"
MAX_DEGREE="${MAX_DEGREE:-32}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-400}"
GRAPH_ITER_TURN="${GRAPH_ITER_TURN:-30}"
EXACT_FILTERED_GT_LIMIT="${EXACT_FILTERED_GT_LIMIT:-200000}"
COMPARE_FULL_BUILD="${COMPARE_FULL_BUILD:-1}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_EXAMPLES=ON \
    -DENABLE_TESTS=OFF

cmake --build "${BUILD_DIR}" --target 328_hgraph_mci_add_compare_hdf5 -j "${THREADS}"

cmd=(
    "${BUILD_DIR}/examples/cpp/328_hgraph_mci_add_compare_hdf5"
    --data "${DATA}"
    --quantization fp32
    --base-io-type memory_io
    --limit "${LIMIT}"
    --query-count "${QUERY_COUNT}"
    --build-ratio "${BUILD_RATIO}"
    --add-batch-size "${ADD_BATCH_SIZE}"
    --build-thread-count "${BUILD_THREAD_COUNT}"
    --threads "${THREADS}"
    --ef-search "${EF_SEARCH}"
    --search-mci-seed-ratio "${SEARCH_MCI_SEED_RATIO}"
    --mci-mcs "${MCI_MCS}"
    --mci-clique-max "${MCI_CLIQUE_MAX}"
    --mci-alpha "${MCI_ALPHA}"
    --max-degree "${MAX_DEGREE}"
    --ef-construction "${EF_CONSTRUCTION}"
    --graph-iter-turn "${GRAPH_ITER_TURN}"
    --exact-filtered-gt-limit "${EXACT_FILTERED_GT_LIMIT}"
)

if [[ -n "${KNNG_PATH}" ]]; then
    cmd+=(--mci-knng-path "${KNNG_PATH}")
fi

if [[ "${COMPARE_FULL_BUILD}" == "0" ]]; then
    cmd+=(--no-full-build)
fi

echo "running:"
printf ' %q' env "LD_LIBRARY_PATH=${BUILD_DIR}/src:${LD_LIBRARY_PATH:-}" "${cmd[@]}"
echo

env "LD_LIBRARY_PATH=${BUILD_DIR}/src:${LD_LIBRARY_PATH:-}" "${cmd[@]}" 2>&1 | tee "${LOG}"

echo "log saved to ${LOG}"
