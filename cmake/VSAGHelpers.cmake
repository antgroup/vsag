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
            get_target_property (_dependee_aliased ${dependee} ALIASED_TARGET)
            get_target_property (_dependee_imported ${dependee} IMPORTED)
            if (NOT _dependee_aliased AND NOT _dependee_imported)
                add_dependencies (${depender} ${dependee})
            endif ()
        endforeach ()
    endif ()
endfunction ()

# Validate that a VSAG_USE_SYSTEM_* value is one of AUTO / ON / OFF.
function (vsag_validate_system_dep_policy policy variable)
    string (TOUPPER "${policy}" _policy)
    if (NOT _policy STREQUAL "AUTO"
            AND NOT _policy STREQUAL "ON"
            AND NOT _policy STREQUAL "OFF")
        message (FATAL_ERROR
                 "${variable} must be one of AUTO, ON, or OFF; got '${policy}'.")
    endif ()
endfunction ()

# Resolve the effective system-dependency policy for ${dep}, returning AUTO / ON / OFF
# in ${out_var}. The per-dependency override VSAG_USE_SYSTEM_<DEP> wins when set to a
# non-empty value; otherwise the global VSAG_USE_SYSTEM_DEPS is used.
function (vsag_get_system_dep_policy dep out_var)
    string (TOUPPER "${dep}" _dep)
    vsag_validate_system_dep_policy ("${VSAG_USE_SYSTEM_DEPS}" "VSAG_USE_SYSTEM_DEPS")
    set (_policy "${VSAG_USE_SYSTEM_DEPS}")
    if (DEFINED VSAG_USE_SYSTEM_${_dep}
            AND NOT "${VSAG_USE_SYSTEM_${_dep}}" STREQUAL "")
        vsag_validate_system_dep_policy ("${VSAG_USE_SYSTEM_${_dep}}"
                                         "VSAG_USE_SYSTEM_${_dep}")
        set (_policy "${VSAG_USE_SYSTEM_${_dep}}")
    endif ()
    string (TOUPPER "${_policy}" _policy)
    set (${out_var} "${_policy}" PARENT_SCOPE)
endfunction ()

# Abort configuration when a system-only dependency cannot be found.
function (vsag_fail_missing_system_dep dep package hint)
    message (FATAL_ERROR
             "VSAG was asked to use a system copy of ${dep} (VSAG_USE_SYSTEM_${dep}=ON "
             "or VSAG_USE_SYSTEM_DEPS=ON), but it was not found. "
             "Install package '${package}', expose target ${hint} before configuring VSAG, "
             "or set VSAG_USE_SYSTEM_${dep}=OFF to use the bundled copy.")
endfunction ()

# Return TRUE when ${target}'s declared include directories contain ${header}.
function (vsag_target_has_header target header out_var)
    set (_include_dirs "")
    get_target_property (_target_includes ${target} INTERFACE_INCLUDE_DIRECTORIES)
    if (_target_includes)
        list (APPEND _include_dirs ${_target_includes})
    endif ()

    get_target_property (_target_system_includes ${target} INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
    if (_target_system_includes)
        list (APPEND _include_dirs ${_target_system_includes})
    endif ()

    set (_found FALSE)
    foreach (_include_dir ${_include_dirs})
        if (_include_dir MATCHES "^\\$<BUILD_INTERFACE:(.*)>$")
            set (_include_dir "${CMAKE_MATCH_1}")
        elseif (_include_dir MATCHES "^\\$<INSTALL_INTERFACE:.*>$")
            continue ()
        endif ()
        if (_include_dir AND EXISTS "${_include_dir}/${header}")
            set (_found TRUE)
            break ()
        endif ()
    endforeach ()
    set (${out_var} ${_found} PARENT_SCOPE)
endfunction ()
