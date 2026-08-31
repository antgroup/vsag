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

function (get_make_commands output_var)
    execute_process (
        COMMAND ${CMAKE_COMMAND} -E env --unset=VSAG_ENABLE_CCACHE
                make --no-print-directory --dry-run ${ARGN}
        WORKING_DIRECTORY "${VSAG_SOURCE_DIR}"
        RESULT_VARIABLE make_result
        OUTPUT_VARIABLE make_output
        ERROR_VARIABLE make_error
    )
    if (NOT make_result EQUAL 0)
        message (FATAL_ERROR "make ${ARGN} failed:\n${make_error}")
    endif ()
    set (${output_var} "${make_output}" PARENT_SCOPE)
endfunction ()

function (assert_command_contains commands expected)
    string (FIND "${commands}" "${expected}" match_position)
    if (match_position EQUAL -1)
        message (FATAL_ERROR "Expected command fragment '${expected}' in:\n${commands}")
    endif ()
endfunction ()

get_make_commands (release_commands release)
assert_command_contains ("${release_commands}" "-B\"./build-release/\"")
assert_command_contains ("${release_commands}" "-DCMAKE_BUILD_TYPE=Release")
assert_command_contains ("${release_commands}" "-DENABLE_CCACHE=OFF")

get_make_commands (perf_release_commands release-perf)
assert_command_contains ("${perf_release_commands}" "-B\"./build-release-perf/\"")
assert_command_contains ("${perf_release_commands}" "-DCMAKE_BUILD_TYPE=Release")
assert_command_contains ("${perf_release_commands}" "-DENABLE_CCACHE=ON")

get_make_commands (release_override_commands release VSAG_ENABLE_CCACHE=ON)
assert_command_contains ("${release_override_commands}" "-DENABLE_CCACHE=ON")

get_make_commands (perf_release_override_commands release-perf VSAG_ENABLE_CCACHE=OFF)
assert_command_contains ("${perf_release_override_commands}" "-DENABLE_CCACHE=OFF")

get_make_commands (distribution_commands dist-cxx11-abi)
assert_command_contains ("${distribution_commands}" "-DCMAKE_BUILD_TYPE=Release")
assert_command_contains ("${distribution_commands}" "-DENABLE_CCACHE=OFF")

message (STATUS "Release build mode checks passed")
