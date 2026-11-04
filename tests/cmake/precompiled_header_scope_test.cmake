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

if (NOT VSAG_SOURCE_DIR)
    message (FATAL_ERROR "VSAG_SOURCE_DIR is required")
endif ()

file (READ "${VSAG_SOURCE_DIR}/CMakeLists.txt" root_cmake)
set (pch_guard "if (ENABLE_TESTS AND NOT CMAKE_DISABLE_PRECOMPILE_HEADERS)")
string (FIND "${root_cmake}" "${pch_guard}" block_start)
if (block_start EQUAL -1)
    message (FATAL_ERROR "The unit-test PCH block must honor CMake's standard opt-out")
endif ()

string (SUBSTRING "${root_cmake}" ${block_start} -1 block_remainder)
string (FIND "${block_remainder}" "\nendif ()" block_end)
if (block_end EQUAL -1)
    message (FATAL_ERROR "The unit-test PCH block is not terminated")
endif ()
string (SUBSTRING "${block_remainder}" 0 ${block_end} pch_block)

function (assert_target target expect_pch)
    string (REGEX MATCHALL "(^|[ \t\r\n])${target}([ \t\r\n)]|$)" matches "${pch_block}")
    list (LENGTH matches match_count)
    if (expect_pch AND NOT match_count EQUAL 1)
        message (FATAL_ERROR "Expected exactly one PCH entry for ${target}")
    elseif (NOT expect_pch AND NOT match_count EQUAL 0)
        message (FATAL_ERROR "Unexpected PCH entry for ${target}")
    endif ()
endfunction ()

set (pch_targets
     algorithm_test
     datacell_test
     impl_test
     io_test
     quantizer_test
     simd_test)
foreach (target IN LISTS pch_targets)
    assert_target (${target} TRUE)
endforeach ()

set (excluded_targets
     attr_test
     factory_test
     layout_test
     storage_test
     utils_test
     vsag_test)
foreach (target IN LISTS excluded_targets)
    assert_target (${target} FALSE)
endforeach ()

string (REGEX MATCHALL "target_precompile_headers" pch_commands "${pch_block}")
list (LENGTH pch_commands pch_command_count)
if (NOT pch_command_count EQUAL 1)
    message (FATAL_ERROR "Unit-test PCH configuration must stay in one target-local command")
endif ()
if (NOT pch_block MATCHES "tests/fixtures/unittest\\.h")
    message (FATAL_ERROR "Unit-test PCH must use the shared fixture header")
endif ()
if (pch_block MATCHES "CMAKE_UNITY_BUILD|UNITY_BUILD")
    message (FATAL_ERROR "Unity Build must not be enabled by the PCH configuration")
endif ()

message (STATUS "Unit-test PCH scope checks passed")
