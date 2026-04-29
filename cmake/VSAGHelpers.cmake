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
            if (TARGET ${dependee})
                add_dependencies (${depender} ${dependee})
            endif ()
        endforeach ()
    endif ()
endfunction ()

function (vsag_link_openmp_private target)
    if (TARGET ${target} AND TARGET vsag_openmp)
        target_link_libraries (${target} PRIVATE vsag_openmp)
    endif ()
endfunction ()
