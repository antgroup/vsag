#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo "usage: run_rabitq_reference.sh BENCHMARK FULL_LIBRARY PREPARED_DATASET_DIRECTORY OUTPUT_DIRECTORY" >&2
    exit 1
fi
benchmark=$1
full_library=$2
dataset=$3
output_dir=$4
if [[ ! -f "${benchmark}" || ! -x "${benchmark}" ]]; then
    echo "benchmark is missing or not executable: ${benchmark}" >&2
    exit 1
fi
if [[ ! -f "${full_library}" ]]; then
    echo "library is missing: ${full_library}" >&2
    exit 1
fi
for count in 10000 100000; do
    for input in base.fvecs queries.fvecs groundtruth.ivecs manifest.json; do
        if [[ ! -f "${dataset}/scale-${count}/${input}" ]]; then
            echo "missing input: ${dataset}/scale-${count}/${input}" >&2
            exit 1
        fi
    done
done
if [[ -e "${output_dir}" ]]; then
    echo "output directory already exists: ${output_dir}" >&2
    exit 1
fi
mkdir -p "${output_dir}"
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
{
    echo "commit=$(git -C "${repo_root}" rev-parse HEAD)"
    echo "benchmark_sha256=$(sha256sum "${benchmark}" | cut -d' ' -f1)"
    echo "full_library_sha256=$(sha256sum "${full_library}" | cut -d' ' -f1)"
    echo "uname=$(uname -a)"
    echo "compiler=$(c++ --version | head -n 1)"
    echo "cmake=$(cmake --version | head -n 1)"
    lscpu
} >"${output_dir}/environment.txt"
for count in 10000 100000; do
    cp "${dataset}/scale-${count}/manifest.json" "${output_dir}/scale-${count}-manifest.json"
done
modes=(fp32 rabitq1 rabitq3x5)
for run in {1..7}; do
    start=$(((run - 1) % 3))
    order=("${modes[@]:start}" "${modes[@]:0:start}")
    for mode in "${order[@]}"; do
        for count in 10000 100000; do
            name="${mode}-scale-${count}-run${run}"
            /usr/bin/time -v -o "${output_dir}/${name}.time.txt"                 "${benchmark}" "${dataset}/scale-${count}"                 "${output_dir}/${name}.snapshot" "${mode}"                 >"${output_dir}/${name}.stdout.txt" 2>"${output_dir}/${name}.stderr.txt"
            awk -F, '$1 == "mode" && $2 == "base_count" {
                if (++found != 1) exit 1
                columns = NF; pending = 1; print; next
            }
            pending {
                if (NF != columns || $1 !~ /^(fp32|rabitq1|rabitq3x5)$/) exit 1
                print; pending = 0; next
            }
            END { if (found != 1 || pending) exit 1 }'                 "${output_dir}/${name}.stdout.txt" >"${output_dir}/${name}.csv"
        done
    done
done
