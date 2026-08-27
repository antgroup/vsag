#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
data_root=${1:-/tmp/vsag-saq-data}
result_root=${2:-${repo_root}/benchs/saq/results/ablation-local}
build_dir=${SAQ_BUILD_DIR:-${repo_root}/build-release}
build_jobs=${SAQ_BUILD_JOBS:-6}
train_count=${SAQ_TRAIN_COUNT:-65536}
encode_count=${SAQ_ENCODE_COUNT:-1000000}
exact_rabitq=${SAQ_EXACT_RABITQ:-1}
extra_c_flags=${SAQ_C_FLAGS:--Wno-error=stringop-overflow}
extra_cxx_flags=${SAQ_CXX_FLAGS:--Wno-error=stringop-overflow}
run_started_epoch=$(date +%s)

mkdir -p "${data_root}" "${result_root}"
environment_path="${result_root}/environment.txt"
: > "${environment_path}"

record_completion() {
    local status=$?
    trap - EXIT
    {
        echo "run_finished=$(date --iso-8601=seconds)"
        echo "run_exit_status=${status}"
        echo "run_elapsed_seconds=$(($(date +%s) - run_started_epoch))"
    } >> "${environment_path}"
    exit "${status}"
}
trap record_completion EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cmake -S "${repo_root}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="${extra_c_flags}" \
    -DCMAKE_CXX_FLAGS="${extra_cxx_flags}" -DENABLE_TOOLS=ON -DENABLE_TESTS=OFF
cmake --build "${build_dir}" --target saq_quantization_benchmark --parallel "${build_jobs}"

quant_binary="${build_dir}/tools/saq_quantization_benchmark/saq_quantization_benchmark"

download_dataset() {
    local filename=$1
    local url=$2
    if [[ ! -s "${data_root}/${filename}" ]]; then
        curl --fail --location --retry 3 --output "${data_root}/${filename}.part" "${url}"
        mv "${data_root}/${filename}.part" "${data_root}/${filename}"
    fi
}

run_dataset() {
    local name=$1
    local filename=$2
    local output_dir="${result_root}/${name}"
    mkdir -p "${output_dir}"
    local quantizer_flags=(--ablations)
    if [[ "${exact_rabitq}" == "1" ]]; then
        quantizer_flags+=(--exact-rabitq)
    fi
    "${quant_binary}" "${data_root}/${filename}" "${output_dir}/quantization.json" 4 \
        "${train_count}" "${encode_count}" "${quantizer_flags[@]}" \
        2>&1 | tee "${output_dir}/quantization.stdout.txt"
}

{
    echo "run_started=$(date --iso-8601=seconds)"
    echo "git_branch=$(git -C "${repo_root}" branch --show-current)"
    echo "git_tracked_status_begin"
    git -C "${repo_root}" status --short --untracked-files=no
    echo "git_tracked_status_end"
    echo "data_root=$(realpath "${data_root}")"
    echo "result_root=$(realpath "${result_root}")"
    echo "build_dir=$(realpath "${build_dir}")"
    echo "build_jobs=${build_jobs}"
    echo "train_count=${train_count}"
    echo "encode_count=${encode_count}"
    echo "exact_rabitq=${exact_rabitq}"
    echo "average_bits_per_dimension=4"
    echo "ablation_fixed_segment_count=1,2"
    echo "saq_random_rotation_seed=20260825"
    echo "reconstruction_count_limit=10000"
    echo "extra_c_flags=${extra_c_flags}"
    echo "extra_cxx_flags=${extra_cxx_flags}"
    echo "cmake_build_type=Release"
    uname -a
    cat /etc/os-release
    cmake --version
    c++ --version
    lscpu
    free -h
} > "${environment_path}"

download_dataset "sift-128-euclidean.hdf5" \
    "https://ann-benchmarks.com/sift-128-euclidean.hdf5"
download_dataset "gist-960-euclidean.hdf5" \
    "https://ann-benchmarks.com/gist-960-euclidean.hdf5"
ldd "${quant_binary}" >> "${environment_path}"

run_dataset "sift1m" "sift-128-euclidean.hdf5"
run_dataset "gist1m" "gist-960-euclidean.hdf5"
python3 "${script_dir}/summarize.py" "${result_root}"
