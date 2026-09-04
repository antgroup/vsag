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

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output_dir="${1:-${root_dir}/benchmark-results/lite-comparison}"
count="${COUNT:-10000}"
dimension="${DIMENSION:-64}"
queries="${QUERIES:-100}"
jobs="${JOBS:-4}"
mkdir -p "${output_dir}"

extra_args=()
if [[ -n "${CMAKE_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "${CMAKE_EXTRA_ARGS}"
fi

build_variant() {
    local variant="$1"
    local lite="$2"
    local build_dir="${root_dir}/build-benchmark-${variant}"
    cmake -S "${root_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_LITE="${lite}" -DENABLE_LITE_BENCHMARK=ON "${extra_args[@]}"
    cmake --build "${build_dir}" --target vsag_lite_benchmark --parallel "${jobs}"
    stat -c '%s' "${build_dir}/src/libvsag_static.a" > "${output_dir}/${variant}.library_bytes"
    /usr/bin/time -v -o "${output_dir}/${variant}.time" \
        "${build_dir}/benchs/lite/vsag_lite_benchmark" \
        "${count}" "${dimension}" "${queries}" \
        2> "${output_dir}/${variant}.log" |
        tail -n 1 > "${output_dir}/${variant}.json"
}

build_variant full OFF
build_variant lite ON
python3 "${root_dir}/scripts/benchmark/summarize_lite_comparison.py" "${output_dir}"
