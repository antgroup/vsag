# Copyright 2026-present the vsag project
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

if (NOT DEFINED VSAG_SOURCE_DIR)
    message (FATAL_ERROR "VSAG_SOURCE_DIR is required")
endif ()

set (fixture_source "${VSAG_SOURCE_DIR}/tests/cmake/openblas_build_config_fixture")
set (fixture_build_root "${CMAKE_CURRENT_BINARY_DIR}/openblas-build-config-fixture")

function (configure_fixture processor output)
    set (fixture_build "${fixture_build_root}/${processor}")
    execute_process (
        COMMAND ${CMAKE_COMMAND} -S ${fixture_source} -B ${fixture_build}
                -DVSAG_SOURCE_DIR=${VSAG_SOURCE_DIR}
                -DVSAG_TARGET_PROCESSOR=${processor}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    if (NOT result EQUAL 0)
        message (FATAL_ERROR
                 "OpenBLAS build configuration fixture failed for ${processor}:\n"
                 "${stdout}\n${stderr}")
    endif ()
    file (READ "${fixture_build}/external-project-args.txt" external_project_args)
    set (${output} "${external_project_args}" PARENT_SCOPE)
endfunction ()

configure_fixture (x86_64 external_project_args)
string (REGEX MATCHALL "NO_PARALLEL_MAKE=1" no_parallel_args "${external_project_args}")
list (LENGTH no_parallel_args no_parallel_count)
if (NOT no_parallel_count EQUAL 2)
    message (FATAL_ERROR
             "Expected NO_PARALLEL_MAKE=1 in bundled OpenBLAS build and install commands:\n"
             "${external_project_args}")
endif ()

file (REMOVE_RECURSE "${fixture_build_root}")
message (STATUS "OpenBLAS build configuration checks passed")
