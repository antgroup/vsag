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

cmake_minimum_required (VERSION 3.18)
project (openblas_build_config_fixture LANGUAGES C CXX)

if (NOT DEFINED VSAG_SOURCE_DIR OR NOT DEFINED VSAG_TARGET_PROCESSOR)
    message (FATAL_ERROR "VSAG_SOURCE_DIR and VSAG_TARGET_PROCESSOR are required")
endif ()

set (BUILD_INFO_DIR "${CMAKE_BINARY_DIR}/build-info")
set (DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloads")
set (VSAG_THIRDPARTY_C_FLAGS "-O2 -fPIC")
set (VSAG_THIRDPARTY_CXX_FLAGS "-O2 -fPIC")
set (common_configure_envs ${CMAKE_COMMAND} -E env)
set (VSAG_USE_SYSTEM_OPENBLAS "")

function (vsag_get_system_dep_policy dependency output)
    set (${output} OFF PARENT_SCOPE)
endfunction ()

function (vsag_resolve_thirdparty_override dependency pin urls_variable)
endfunction ()

function (ExternalProject_Add)
    file (WRITE "${CMAKE_BINARY_DIR}/external-project-args.txt" "${ARGV}")
endfunction ()

include (${VSAG_SOURCE_DIR}/extern/openblas/openblas.cmake)
