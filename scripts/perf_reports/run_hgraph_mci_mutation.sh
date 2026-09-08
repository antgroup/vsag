#!/usr/bin/env bash

set -euo pipefail

MCI_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MCI_BUILD_DIR="${MCI_BUILD_DIR:-${MCI_REPO_ROOT}/build-release}"
MCI_DATASET_PATH="${MCI_DATASET_PATH:-/root/data/codefilter-10k-384-angular-f32.hdf5}"
MCI_RESULT_DIR="${MCI_RESULT_DIR:-/tmp/vsag_mci_mutation}"
MCI_CSV_PATH="${MCI_CSV_PATH:-${MCI_RESULT_DIR}/codefilter_mci_mutation.csv}"
MCI_PLOT_PATH="${MCI_PLOT_PATH:-${MCI_RESULT_DIR}/codefilter_mci_mutation.png}"
MCI_BUILD_JOBS="${MCI_BUILD_JOBS:-$(nproc)}"

mkdir -p "${MCI_RESULT_DIR}"

if [[ "${MCI_SKIP_BUILD:-0}" != "1" ]]; then
    cmake -S "${MCI_REPO_ROOT}" -B "${MCI_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_CCACHE=ON \
        -DENABLE_TESTS=OFF \
        -DENABLE_TOOLS=ON
    cmake --build "${MCI_BUILD_DIR}" \
        --target mci_mutation_benchmark \
        -j"${MCI_BUILD_JOBS}"
fi

"${MCI_BUILD_DIR}/tools/eval/mci_mutation_benchmark" \
    --dataset "${MCI_DATASET_PATH}" \
    --output "${MCI_CSV_PATH}" \
    "$@"

if python3 -c 'import matplotlib' >/dev/null 2>&1; then
    python3 "${MCI_REPO_ROOT}/scripts/perf_reports/plot_mci_mutation_curve.py" \
        "${MCI_CSV_PATH}" \
        --output "${MCI_PLOT_PATH}"
else
    echo "matplotlib is unavailable; CSV result: ${MCI_CSV_PATH}"
fi
