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

include_guard (GLOBAL)

function (vsag_set_default variable value)
    if (NOT DEFINED ${variable} OR "${${variable}}" STREQUAL "")
        set (${variable} "${value}" PARENT_SCOPE)
    endif ()
endfunction ()

function (vsag_add_compile_flag flag)
    add_compile_options ("${flag}")
    if (DEFINED VSAG_EXTERNAL_C_FLAGS AND NOT "${VSAG_EXTERNAL_C_FLAGS}" STREQUAL "")
        set (_vsag_external_c_flags "${VSAG_EXTERNAL_C_FLAGS} ${flag}")
    else ()
        set (_vsag_external_c_flags "${flag}")
    endif ()
    if (DEFINED VSAG_EXTERNAL_CXX_FLAGS AND NOT "${VSAG_EXTERNAL_CXX_FLAGS}" STREQUAL "")
        set (_vsag_external_cxx_flags "${VSAG_EXTERNAL_CXX_FLAGS} ${flag}")
    else ()
        set (_vsag_external_cxx_flags "${flag}")
    endif ()

    set (VSAG_EXTERNAL_C_FLAGS "${_vsag_external_c_flags}" PARENT_SCOPE)
    set (VSAG_EXTERNAL_CXX_FLAGS "${_vsag_external_cxx_flags}" PARENT_SCOPE)
endfunction ()

function (vsag_add_c_compile_flag flag)
    add_compile_options ($<$<COMPILE_LANGUAGE:C>:${flag}>)
    if (DEFINED VSAG_EXTERNAL_C_FLAGS AND NOT "${VSAG_EXTERNAL_C_FLAGS}" STREQUAL "")
        set (_vsag_external_c_flags "${VSAG_EXTERNAL_C_FLAGS} ${flag}")
    else ()
        set (_vsag_external_c_flags "${flag}")
    endif ()
    set (VSAG_EXTERNAL_C_FLAGS "${_vsag_external_c_flags}" PARENT_SCOPE)
endfunction ()

function (vsag_add_cxx_compile_flag flag)
    add_compile_options ($<$<COMPILE_LANGUAGE:CXX>:${flag}>)
    if (DEFINED VSAG_EXTERNAL_CXX_FLAGS AND NOT "${VSAG_EXTERNAL_CXX_FLAGS}" STREQUAL "")
        set (_vsag_external_cxx_flags "${VSAG_EXTERNAL_CXX_FLAGS} ${flag}")
    else ()
        set (_vsag_external_cxx_flags "${flag}")
    endif ()
    set (VSAG_EXTERNAL_CXX_FLAGS "${_vsag_external_cxx_flags}" PARENT_SCOPE)
endfunction ()

function (vsag_add_linker_flag flag)
    add_link_options ("${flag}")
    if (DEFINED VSAG_EXTERNAL_EXE_LINKER_FLAGS AND NOT "${VSAG_EXTERNAL_EXE_LINKER_FLAGS}" STREQUAL "")
        set (_vsag_external_exe_linker_flags "${VSAG_EXTERNAL_EXE_LINKER_FLAGS} ${flag}")
    else ()
        set (_vsag_external_exe_linker_flags "${flag}")
    endif ()
    if (DEFINED VSAG_EXTERNAL_SHARED_LINKER_FLAGS AND NOT "${VSAG_EXTERNAL_SHARED_LINKER_FLAGS}" STREQUAL "")
        set (_vsag_external_shared_linker_flags "${VSAG_EXTERNAL_SHARED_LINKER_FLAGS} ${flag}")
    else ()
        set (_vsag_external_shared_linker_flags "${flag}")
    endif ()

    set (VSAG_EXTERNAL_EXE_LINKER_FLAGS "${_vsag_external_exe_linker_flags}" PARENT_SCOPE)
    set (VSAG_EXTERNAL_SHARED_LINKER_FLAGS "${_vsag_external_shared_linker_flags}" PARENT_SCOPE)
endfunction ()

function (maybe_add_dependencies depender)
    if (TARGET ${depender})
        foreach (dependee ${ARGN})
            if (NOT TARGET ${dependee})
                continue ()
            endif ()
            get_target_property (_dependee_aliased_target ${dependee} ALIASED_TARGET)
            get_target_property (_dependee_imported ${dependee} IMPORTED)
            if (NOT _dependee_aliased_target AND NOT _dependee_imported)
                add_dependencies (${depender} ${dependee})
            endif ()
        endforeach ()
    endif ()
endfunction ()

function (vsag_validate_system_dep_policy policy variable)
    string (TOUPPER "${policy}" _policy)
    if (NOT _policy STREQUAL "AUTO" AND NOT _policy STREQUAL "ON" AND NOT _policy STREQUAL "OFF")
        message (FATAL_ERROR "${variable} must be one of AUTO, ON, or OFF; got '${policy}'.")
    endif ()
endfunction ()

function (vsag_get_system_dep_policy dep out_var)
    string (TOUPPER "${dep}" _dep)
    vsag_validate_system_dep_policy ("${VSAG_USE_SYSTEM_DEPS}" "VSAG_USE_SYSTEM_DEPS")
    set (_policy "${VSAG_USE_SYSTEM_DEPS}")
    if (DEFINED VSAG_USE_SYSTEM_${_dep} AND NOT "${VSAG_USE_SYSTEM_${_dep}}" STREQUAL "")
        vsag_validate_system_dep_policy ("${VSAG_USE_SYSTEM_${_dep}}" "VSAG_USE_SYSTEM_${_dep}")
        set (_policy "${VSAG_USE_SYSTEM_${_dep}}")
    endif ()
    string (TOUPPER "${_policy}" _policy)
    set (${out_var} "${_policy}" PARENT_SCOPE)
endfunction ()

function (vsag_note_system_dep dep target)
    message (STATUS "Using external ${dep}: ${target}")
endfunction ()

function (vsag_fail_missing_system_dep dep package targets)
    message (FATAL_ERROR
             "VSAG_USE_SYSTEM_${dep}=ON or VSAG_USE_SYSTEM_DEPS=ON, but external ${dep} was not found. "
             "Provide one of these targets before including VSAG (${targets}), or install/find package '${package}', "
             "or set VSAG_USE_SYSTEM_${dep}=OFF to use the bundled copy.")
endfunction ()

function (vsag_alias_library alias target)
    if (NOT TARGET ${alias} AND TARGET ${target})
        add_library (${alias} ALIAS ${target})
    endif ()
endfunction ()

function (vsag_add_interface_alias alias target)
    if (NOT TARGET ${alias})
        add_library (${alias} INTERFACE)
        target_link_libraries (${alias} INTERFACE ${target})
    endif ()
endfunction ()
