#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: run_comparison.sh FULL_BENCHMARK LITE_BENCHMARK FULL_LIBRARY LITE_LIBRARY OUTPUT_DIRECTORY" >&2
    exit 1
fi

full_benchmark=$1
lite_benchmark=$2
full_library=$3
lite_library=$4
output_dir=$5
if [[ -e "${output_dir}" ]]; then
    echo "output directory already exists: ${output_dir}" >&2
    exit 1
fi
for input in "${full_benchmark}" "${lite_benchmark}" "${full_library}" "${lite_library}"; do
    if [[ ! -f "${input}" ]]; then
        echo "input file does not exist: ${input}" >&2
        exit 1
    fi
done
for benchmark in "${full_benchmark}" "${lite_benchmark}"; do
    if [[ ! -x "${benchmark}" ]]; then
        echo "benchmark is not executable: ${benchmark}" >&2
        exit 1
    fi
done
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
    echo "comparison_scope=fp32_squared_l2_bruteforce"
    lscpu
} >"${output_dir}/environment.txt"

measure_library() {
    local implementation=$1
    local library=$2
    local stripped="${output_dir}/${implementation}-stripped.so"
    cp "${library}" "${stripped}"
    strip --strip-unneeded "${stripped}"
    printf '%s,%s,%s\n' "${implementation}" "$(stat -c %s "${library}")" \
        "$(stat -c %s "${stripped}")" >>"${output_dir}/library-size.csv"
}

echo "implementation,unstripped_bytes,stripped_bytes" >"${output_dir}/library-size.csv"
measure_library full "${full_library}"
measure_library lite "${lite_library}"

run_case() {
    local implementation=$1
    local benchmark=$2
    local name=$3
    shift 3
    /usr/bin/time -v -o "${output_dir}/${implementation}-${name}.time.txt" \
        "${benchmark}" "$@" "${output_dir}/${implementation}-${name}.snapshot" \
        >"${output_dir}/${implementation}-${name}.stdout.txt"
    tail -n 2 "${output_dir}/${implementation}-${name}.stdout.txt" \
        >"${output_dir}/${implementation}-${name}.csv"
}

for run in {1..7}; do
    if ((run % 2 == 1)); then
        order=(full lite)
    else
        order=(lite full)
    fi
    for implementation in "${order[@]}"; do
        if [[ "${implementation}" == "full" ]]; then
            benchmark=${full_benchmark}
        else
            benchmark=${lite_benchmark}
        fi
        run_case "${implementation}" "${benchmark}" "scale-10k-run${run}" \
            10000 128 64 10 500 20260909
        run_case "${implementation}" "${benchmark}" "scale-100k-run${run}" \
            100000 128 32 10 1000 20260909
    done
done
