#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: run_sift.sh LITE_DATASET_BENCHMARK PREPARED_DATASET_DIRECTORY OUTPUT_DIRECTORY" >&2
    exit 1
fi
benchmark=$1
dataset=$2
output_dir=$3
if [[ ! -f "${benchmark}" || ! -x "${benchmark}" ]]; then
    echo "benchmark is missing or not executable: ${benchmark}" >&2
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
    echo "uname=$(uname -a)"
    echo "compiler=$(c++ --version | head -n 1)"
    echo "cmake=$(cmake --version | head -n 1)"
    lscpu
} >"${output_dir}/environment.txt"
for count in 10000 100000; do
    cp "${dataset}/scale-${count}/manifest.json" "${output_dir}/scale-${count}-manifest.json"
    for run in {1..7}; do
        name="scale-${count}-run${run}"
        /usr/bin/time -v -o "${output_dir}/${name}.time.txt" \
            "${benchmark}" "${dataset}/scale-${count}" "${output_dir}/${name}.snapshot" \
            >"${output_dir}/${name}.csv" 2>"${output_dir}/${name}.stderr.txt"
    done
done
