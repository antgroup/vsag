#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 6 ]]; then
    echo "usage: run_sift_comparison.sh FULL_DATASET_BENCHMARK LITE_DATASET_BENCHMARK FULL_LIBRARY LITE_LIBRARY PREPARED_DATASET_DIRECTORY OUTPUT_DIRECTORY" >&2
    exit 1
fi
full_benchmark=$1
lite_benchmark=$2
full_library=$3
lite_library=$4
dataset=$5
output_dir=$6
for benchmark in "${full_benchmark}" "${lite_benchmark}"; do
    if [[ ! -f "${benchmark}" || ! -x "${benchmark}" ]]; then
        echo "benchmark is missing or not executable: ${benchmark}" >&2
        exit 1
    fi
done
for library in "${full_library}" "${lite_library}"; do
    if [[ ! -f "${library}" ]]; then
        echo "library is missing: ${library}" >&2
        exit 1
    fi
done
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
    echo "full_benchmark_sha256=$(sha256sum "${full_benchmark}" | cut -d' ' -f1)"
    echo "lite_benchmark_sha256=$(sha256sum "${lite_benchmark}" | cut -d' ' -f1)"
    echo "full_library_sha256=$(sha256sum "${full_library}" | cut -d' ' -f1)"
    echo "lite_library_sha256=$(sha256sum "${lite_library}" | cut -d' ' -f1)"
    echo "uname=$(uname -a)"
    echo "compiler=$(c++ --version | head -n 1)"
    echo "cmake=$(cmake --version | head -n 1)"
    lscpu
} >"${output_dir}/environment.txt"
for count in 10000 100000; do
    cp "${dataset}/scale-${count}/manifest.json" "${output_dir}/scale-${count}-manifest.json"
done
for run in {1..7}; do
    if ((run % 2 == 1)); then
        order=(full lite)
    else
        order=(lite full)
    fi
    for implementation in "${order[@]}"; do
        if [[ "${implementation}" == full ]]; then
            benchmark=${full_benchmark}
        else
            benchmark=${lite_benchmark}
        fi
        for count in 10000 100000; do
            name="${implementation}-scale-${count}-run${run}"
            /usr/bin/time -v -o "${output_dir}/${name}.time.txt" \
                "${benchmark}" "${dataset}/scale-${count}" "${output_dir}/${name}.snapshot" \
                >"${output_dir}/${name}.stdout.txt" 2>"${output_dir}/${name}.stderr.txt"
            tail -n 2 "${output_dir}/${name}.stdout.txt" >"${output_dir}/${name}.csv"
        done
    done
done
