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
project (compile_flag_scope_fixture LANGUAGES C CXX ASM)

if (NOT DEFINED VSAG_SOURCE_DIR)
    message (FATAL_ERROR "VSAG_SOURCE_DIR is required")
endif ()

include (${VSAG_SOURCE_DIR}/cmake/VSAGHelpers.cmake)

vsag_add_compile_flag (-DVSAG_C_AND_CXX_OPTION)
vsag_add_c_compile_flag (-DVSAG_C_OPTION)
vsag_add_cxx_compile_flag (-DVSAG_CXX_OPTION)
add_link_options (-Wl,--as-needed)

vsag_include_without_directory_options (thirdparty.cmake)
add_subdirectory (owned)
