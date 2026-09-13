#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: run_baseline.sh LITE_BENCHMARK OUTPUT_DIRECTORY" >&2
    exit 1
fi

benchmark=$1
output_dir=$2
if [[ -e "${output_dir}" ]]; then
    echo "output directory already exists: ${output_dir}" >&2
    exit 1
fi
if [[ ! -f "${benchmark}" || ! -x "${benchmark}" ]]; then
    echo "benchmark executable does not exist or is not executable: ${benchmark}" >&2
    exit 1
fi
mkdir -p "${output_dir}"

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
{
    echo "commit=$(git -C "${repo_root}" rev-parse HEAD)"
    echo "benchmark_sha256=$(sha256sum "${benchmark}" | cut -d' ' -f1)"
    echo "uname=$(uname -a)"
    echo "compiler=$(c++ --version | head -n 1)"
    echo "cmake=$(cmake --version | head -n 1)"
    echo "fresh_process_page_cache_control=none"
    lscpu
} >"${output_dir}/environment.txt"

run_case() {
    local name=$1
    shift
    /usr/bin/time -v -o "${output_dir}/${name}.time.txt" \
        "${benchmark}" "$@" "${output_dir}/${name}.snapshot" \
        >"${output_dir}/${name}.csv"
}

run_case scale-10k 10000 128 64 10 500 20260909
run_case scale-100k 100000 128 32 10 1000 20260909

run_load_case() {
    local name=$1
    local snapshot=$2
    local queries=$3
    for run in {1..7}; do
        /usr/bin/time -v -o "${output_dir}/${name}-run${run}.time.txt" \
            "${benchmark}" load "${snapshot}" 128 0 1 "${queries}" 10 20260909 \
            >"${output_dir}/${name}-run${run}.csv"
    done
}

run_load_case fresh-process-load-10k "${output_dir}/scale-10k.snapshot" 64
run_load_case fresh-process-load-100k "${output_dir}/scale-100k.snapshot" 32

/usr/bin/time -v -o "${output_dir}/crud-stability.time.txt" \
    "${benchmark}" stability 10000 128 20 500 32 10 20260909 \
    "${output_dir}/crud-stability-snapshot" \
    >"${output_dir}/crud-stability.csv"
