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

find_program (MAKE_EXECUTABLE NAMES gmake make REQUIRED)
find_program (UNAME_EXECUTABLE NAMES uname REQUIRED)

string (RANDOM LENGTH 12 ALPHABET 0123456789abcdef fixture_suffix)
set (fixture_root "/tmp/vsag-generator-selection-${fixture_suffix}")
set (ninja_path "${fixture_root}/with-ninja")
set (fallback_path "${fixture_root}/without-ninja")

function (create_tool_path path include_ninja)
    file (MAKE_DIRECTORY "${path}")
    file (CREATE_LINK "${CMAKE_COMMAND}" "${path}/cmake" SYMBOLIC)
    file (CREATE_LINK "${MAKE_EXECUTABLE}" "${path}/make" SYMBOLIC)
    file (CREATE_LINK "${UNAME_EXECUTABLE}" "${path}/uname" SYMBOLIC)
    if (include_ninja)
        # CMake accepts --version, so its executable is a deterministic stand-in for Ninja here.
        file (CREATE_LINK "${CMAKE_COMMAND}" "${path}/ninja" SYMBOLIC)
    endif ()
endfunction ()

function (run_make output_variable tool_path)
    execute_process (
        COMMAND ${CMAKE_COMMAND} -E env --unset=CMAKE_GENERATOR "PATH=${tool_path}"
                ${MAKE_EXECUTABLE} --no-print-directory --dry-run debug ${ARGN}
        WORKING_DIRECTORY "${VSAG_SOURCE_DIR}"
        RESULT_VARIABLE make_result
        OUTPUT_VARIABLE make_output
        ERROR_VARIABLE make_error)
    if (NOT make_result EQUAL 0)
        file (REMOVE_RECURSE "${fixture_root}")
        message (FATAL_ERROR "make ${ARGN} failed:\n${make_output}\n${make_error}")
    endif ()
    set (${output_variable} "${make_output}" PARENT_SCOPE)
endfunction ()

function (run_make_with_environment output_variable tool_path generator)
    execute_process (
        COMMAND ${CMAKE_COMMAND} -E env "PATH=${tool_path}" "CMAKE_GENERATOR=${generator}"
                ${MAKE_EXECUTABLE} --no-print-directory --dry-run debug
        WORKING_DIRECTORY "${VSAG_SOURCE_DIR}"
        RESULT_VARIABLE make_result
        OUTPUT_VARIABLE make_output
        ERROR_VARIABLE make_error)
    if (NOT make_result EQUAL 0)
        file (REMOVE_RECURSE "${fixture_root}")
        message (FATAL_ERROR "make with CMAKE_GENERATOR=${generator} failed:\n"
                            "${make_output}\n${make_error}")
    endif ()
    set (${output_variable} "${make_output}" PARENT_SCOPE)
endfunction ()

function (assert_generator commands expected description)
    string (FIND "${commands}" "-G \"${expected}\"" match_position)
    if (match_position EQUAL -1)
        file (REMOVE_RECURSE "${fixture_root}")
        message (FATAL_ERROR "${description}: expected generator '${expected}' in:\n${commands}")
    endif ()
endfunction ()

file (REMOVE_RECURSE "${fixture_root}")
create_tool_path ("${ninja_path}" TRUE)
create_tool_path ("${fallback_path}" FALSE)

run_make (ninja_commands "${ninja_path}")
assert_generator ("${ninja_commands}" "Ninja" "Ninja auto-selection")

run_make (fallback_commands "${fallback_path}")
assert_generator ("${fallback_commands}" "Unix Makefiles" "fallback selection")

run_make (command_line_override "${ninja_path}" "CMAKE_GENERATOR=Unix Makefiles")
assert_generator ("${command_line_override}" "Unix Makefiles" "command-line precedence")

run_make_with_environment (environment_override "${ninja_path}" "Unix Makefiles")
assert_generator ("${environment_override}" "Unix Makefiles" "environment precedence")

file (REMOVE_RECURSE "${fixture_root}")
message (STATUS "CMake generator selection checks passed")
