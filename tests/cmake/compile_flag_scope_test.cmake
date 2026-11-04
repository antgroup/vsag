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

string (RANDOM LENGTH 12 ALPHABET 0123456789abcdef fixture_suffix)
set (fixture_source "${VSAG_SOURCE_DIR}/tests/cmake/compile_flag_scope_fixture")
set (fixture_build "/tmp/vsag-compile-flag-scope-${fixture_suffix}")

execute_process (
    COMMAND ${CMAKE_COMMAND}
            -S ${fixture_source}
            -B ${fixture_build}
            -G Ninja
            -DVSAG_SOURCE_DIR=${VSAG_SOURCE_DIR}
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_stdout
    ERROR_VARIABLE configure_stderr)
if (NOT configure_result EQUAL 0)
    file (REMOVE_RECURSE "${fixture_build}")
    message (FATAL_ERROR
             "Flag-scope fixture configuration failed:\n${configure_stdout}\n${configure_stderr}")
endif ()

file (READ "${fixture_build}/compile_commands.json" compile_commands)
file (READ "${fixture_build}/build.ninja" ninja_content)

function (get_compile_command source output_variable)
    string (REGEX MATCH
            "[^\n]*${source}[^\n]*"
            command "${compile_commands}")
    if (NOT command)
        file (REMOVE_RECURSE "${fixture_build}")
        message (FATAL_ERROR "No compile command found for ${source}")
    endif ()
    set (${output_variable} "${command}" PARENT_SCOPE)
endfunction ()

function (require_flag command flag description)
    string (FIND "${command}" "${flag}" position)
    if (position EQUAL -1)
        file (REMOVE_RECURSE "${fixture_build}")
        message (FATAL_ERROR "${description} is missing ${flag}:\n${command}")
    endif ()
endfunction ()

function (reject_flag command flag description)
    string (FIND "${command}" "${flag}" position)
    if (NOT position EQUAL -1)
        file (REMOVE_RECURSE "${fixture_build}")
        message (FATAL_ERROR "${description} unexpectedly has ${flag}:\n${command}")
    endif ()
endfunction ()

get_compile_command (thirdparty_probe.c thirdparty_command)
reject_flag ("${thirdparty_command}" "VSAG_C_AND_CXX_OPTION" "third-party C target")
reject_flag ("${thirdparty_command}" "VSAG_C_OPTION" "third-party C target")

get_compile_command (owned_c_probe.c owned_c_command)
require_flag ("${owned_c_command}" "VSAG_C_AND_CXX_OPTION" "VSAG-owned C target")
require_flag ("${owned_c_command}" "VSAG_C_OPTION" "VSAG-owned C target")
reject_flag ("${owned_c_command}" "VSAG_CXX_OPTION" "VSAG-owned C target")

get_compile_command (owned_cxx_probe.cpp owned_cxx_command)
require_flag ("${owned_cxx_command}" "VSAG_C_AND_CXX_OPTION" "VSAG-owned C++ target")
require_flag ("${owned_cxx_command}" "VSAG_CXX_OPTION" "VSAG-owned C++ target")
reject_flag ("${owned_cxx_command}" "VSAG_C_OPTION" "VSAG-owned C++ target")

get_compile_command (owned_asm_probe.S owned_asm_command)
reject_flag ("${owned_asm_command}" "VSAG_C_AND_CXX_OPTION" "VSAG-owned ASM target")
reject_flag ("${owned_asm_command}" "VSAG_C_OPTION" "VSAG-owned ASM target")
reject_flag ("${owned_asm_command}" "VSAG_CXX_OPTION" "VSAG-owned ASM target")

string (REGEX MATCH "LINK_FLAGS = [^\n]*--as-needed" owned_link_options "${ninja_content}")
if (NOT owned_link_options)
    file (REMOVE_RECURSE "${fixture_build}")
    message (FATAL_ERROR "VSAG-owned link options were not restored")
endif ()

file (REMOVE_RECURSE "${fixture_build}")
message (STATUS "Compile flag scope checks passed")
