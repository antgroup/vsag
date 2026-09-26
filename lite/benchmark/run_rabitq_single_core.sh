#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: run_rabitq_single_core.sh BENCHMARK PREPARED_DATASET OUTPUT_DIRECTORY CPU" >&2
    exit 1
fi

benchmark=$1
dataset=$2
output_dir=$3
cpu=$4
runs=${VSAG_SINGLE_CORE_RUNS:-7}
crud_ops=${VSAG_SINGLE_CORE_CRUD_OPS:-20}
queries=${VSAG_SINGLE_CORE_QUERIES:-100}
max_degree=${VSAG_SINGLE_CORE_MAX_DEGREE:-16}
ef_search=${VSAG_SINGLE_CORE_EF_SEARCH:-128}

if [[ ! -x "${benchmark}" ]]; then
    echo "benchmark is missing or not executable: ${benchmark}" >&2
    exit 1
fi
if [[ ! "${cpu}" =~ ^[0-9]+$ ]]; then
    echo "CPU must be a non-negative integer" >&2
    exit 1
fi
for value in "${runs}" "${crud_ops}" "${queries}" "${max_degree}" "${ef_search}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "runner settings must be positive integers" >&2
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
thread_env=(
    OMP_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1
    MKL_NUM_THREADS=1
    BLIS_NUM_THREADS=1
    VECLIB_MAXIMUM_THREADS=1
    NUMEXPR_NUM_THREADS=1
)
perf_available=false
if command -v perf >/dev/null 2>&1 &&
    perf stat -e task-clock true >/dev/null 2>&1; then
    perf_available=true
fi

{
    echo "commit=$(git -C "${repo_root}" rev-parse HEAD)"
    echo "benchmark_sha256=$(sha256sum "${benchmark}" | cut -d' ' -f1)"
    echo "cpu=${cpu}"
    echo "runs=${runs}"
    echo "crud_ops=${crud_ops}"
    echo "queries=${queries}"
    echo "max_degree=${max_degree}"
    echo "ef_search=${ef_search}"
    echo "perf_available=${perf_available}"
    echo "uname=$(uname -a)"
    echo "compiler=$(c++ --version | head -n 1)"
    echo "cmake=$(cmake --version | head -n 1)"
    taskset -c "${cpu}" grep -E "^(Cpus_allowed|Cpus_allowed_list):" /proc/self/status
    lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
} >"${output_dir}/environment.txt"

for count in 10000 100000; do
    cp "${dataset}/scale-${count}/manifest.json" "${output_dir}/scale-${count}-manifest.json"
    for run in $(seq 1 "${runs}"); do
        name="scale-${count}-run${run}"
        snapshot="${output_dir}/${name}-v3.bin"
        command=(
            env "${thread_env[@]}"
            taskset -c "${cpu}"
            "${benchmark}"
            --crud "${dataset}/scale-${count}" "${snapshot}"
            1 "${crud_ops}" "${queries}" "${max_degree}" "${ef_search}"
        )
        if [[ "${perf_available}" == true ]]; then
            /usr/bin/time -v -o "${output_dir}/${name}.time.txt" \
                perf stat -x, -o "${output_dir}/${name}.perf.csv" \
                -e task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions,branches,branch-misses,cache-references,cache-misses \
                -- "${command[@]}" >"${output_dir}/${name}.csv" \
                2>"${output_dir}/${name}.stderr.txt"
        else
            /usr/bin/time -v -o "${output_dir}/${name}.time.txt" \
                "${command[@]}" >"${output_dir}/${name}.csv" \
                2>"${output_dir}/${name}.stderr.txt"
        fi
        if [[ -s "${output_dir}/${name}.stderr.txt" ]]; then
            echo "benchmark wrote stderr: ${output_dir}/${name}.stderr.txt" >&2
            exit 1
        fi
        awk -F, '
            NR == 1 {
                if ($1 != "round" || ++headers != 1) exit 1
                columns = NF
                print
                next
            }
            NR == 2 {
                if (NF != columns || $1 != "1") exit 1
                rows++
                print
                next
            }
            { exit 1 }
            END { if (headers != 1 || rows != 1) exit 1 }
        ' "${output_dir}/${name}.csv" >"${output_dir}/${name}.validated.csv"
    done
done

(
    cd "${output_dir}"
    sha256sum -- * >SHA256SUMS
    sha256sum -c SHA256SUMS
)
