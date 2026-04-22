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

set (_vsag_external_c_flags "${VSAG_EXTERNAL_C_FLAGS}")
set (_vsag_external_cxx_flags "${VSAG_EXTERNAL_CXX_FLAGS}")
set (_vsag_external_exe_linker_flags "${VSAG_EXTERNAL_EXE_LINKER_FLAGS}")
set (_vsag_external_shared_linker_flags "${VSAG_EXTERNAL_SHARED_LINKER_FLAGS}")

foreach (_warning_flag -Werror=suggest-override -Werror)
    string (REPLACE "${_warning_flag}" "" _vsag_external_c_flags "${_vsag_external_c_flags}")
    string (REPLACE "${_warning_flag}" "" _vsag_external_cxx_flags "${_vsag_external_cxx_flags}")
endforeach ()

foreach (_sanitizer_pattern "-fsanitize=[^ ]+" "-fsanitize-recover=[^ ]+" "-fno-sanitize=[^ ]+" -fno-omit-frame-pointer -fno-optimize-sibling-calls -static-libasan)
    string (REGEX REPLACE "${_sanitizer_pattern}" "" _vsag_external_c_flags "${_vsag_external_c_flags}")
    string (REGEX REPLACE "${_sanitizer_pattern}" "" _vsag_external_cxx_flags "${_vsag_external_cxx_flags}")
    string (REGEX REPLACE "${_sanitizer_pattern}" "" _vsag_external_exe_linker_flags "${_vsag_external_exe_linker_flags}")
    string (REGEX REPLACE "${_sanitizer_pattern}" "" _vsag_external_shared_linker_flags "${_vsag_external_shared_linker_flags}")
endforeach ()

string (REGEX REPLACE "[ ]+" " " _vsag_external_c_flags "${_vsag_external_c_flags}")
string (REGEX REPLACE "[ ]+" " " _vsag_external_cxx_flags "${_vsag_external_cxx_flags}")
string (REGEX REPLACE "[ ]+" " " _vsag_external_exe_linker_flags "${_vsag_external_exe_linker_flags}")
string (REGEX REPLACE "[ ]+" " " _vsag_external_shared_linker_flags "${_vsag_external_shared_linker_flags}")

if (NOT WIN32 AND NOT MSVC)
    if (NOT _vsag_external_c_flags MATCHES "(^| )-fPIC($| )")
        string (APPEND _vsag_external_c_flags " -fPIC")
    endif ()
    if (NOT _vsag_external_cxx_flags MATCHES "(^| )-fPIC($| )")
        string (APPEND _vsag_external_cxx_flags " -fPIC")
    endif ()
endif ()

string (STRIP "${_vsag_external_c_flags}" VSAG_THIRDPARTY_C_FLAGS)
string (STRIP "${_vsag_external_cxx_flags}" VSAG_THIRDPARTY_CXX_FLAGS)
string (STRIP "${_vsag_external_exe_linker_flags}" VSAG_THIRDPARTY_EXE_LINKER_FLAGS)
string (STRIP "${_vsag_external_shared_linker_flags}" VSAG_THIRDPARTY_SHARED_LINKER_FLAGS)

if (WIN32)
    set (ld_flags "")
else ()
    set (vsag_ld_flags
        "-L${CMAKE_INSTALL_PREFIX}/lib"
        "-L${CMAKE_INSTALL_PREFIX}/lib64")

    if (CMAKE_LIBRARY_ARCHITECTURE AND EXISTS "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
        list (APPEND vsag_ld_flags "-L/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
    endif ()
    if (CMAKE_LIBRARY_ARCHITECTURE AND EXISTS "/usr/lib64/${CMAKE_LIBRARY_ARCHITECTURE}")
        list (APPEND vsag_ld_flags "-L/usr/lib64/${CMAKE_LIBRARY_ARCHITECTURE}")
    endif ()
    if (EXISTS "/opt/alibaba-cloud-compiler/lib64/")
        list (APPEND vsag_ld_flags "-L/opt/alibaba-cloud-compiler/lib64/")
    endif ()
    string (JOIN " " ld_flags ${vsag_ld_flags})
endif ()

set (VSAG_EXTERNAL_PATH "$ENV{PATH}")
if (DEFINED BUILDING_PATH AND NOT BUILDING_PATH STREQUAL "")
    set (VSAG_EXTERNAL_PATH "${BUILDING_PATH}:$ENV{PATH}")
endif ()

if (WIN32)
    file (TO_CMAKE_PATH "${CMAKE_MAKE_PROGRAM}" _cmake_make_program_fwd)
    set (common_cmake_args
        "-G" "Ninja"
        "-DCMAKE_MAKE_PROGRAM=${_cmake_make_program_fwd}"
        "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DCMAKE_RC_COMPILER=rc"
        "-DCMAKE_CXX_FLAGS=${VSAG_THIRDPARTY_CXX_FLAGS} /EHsc /GR -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS -DWIN32_LEAN_AND_MEAN"
        "-DCMAKE_C_FLAGS=${VSAG_THIRDPARTY_C_FLAGS}"
        "-DCMAKE_EXE_LINKER_FLAGS=${VSAG_THIRDPARTY_EXE_LINKER_FLAGS}"
        "-DCMAKE_SHARED_LINKER_FLAGS=${VSAG_THIRDPARTY_SHARED_LINKER_FLAGS}"
        "-DCMAKE_INCLUDE_PATH=${CMAKE_INSTALL_PREFIX}/include"
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
        "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL")
else ()
    set (common_cmake_args
        "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
        "-DCMAKE_CXX_FLAGS=${VSAG_THIRDPARTY_CXX_FLAGS}"
        "-DCMAKE_C_FLAGS=${VSAG_THIRDPARTY_C_FLAGS}"
        "-DCMAKE_EXE_LINKER_FLAGS=${VSAG_THIRDPARTY_EXE_LINKER_FLAGS}"
        "-DCMAKE_SHARED_LINKER_FLAGS=${VSAG_THIRDPARTY_SHARED_LINKER_FLAGS}"
        "-DCMAKE_INCLUDE_PATH=${CMAKE_INSTALL_PREFIX}/include"
        "-DCMAKE_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:/opt/alibaba-cloud-compiler/lib64")
endif ()

if (NOT WIN32)
    set (common_configure_envs
        "env"
        "CC=${CMAKE_C_COMPILER}"
        "CXX=${CMAKE_CXX_COMPILER}"
        "CFLAGS=${VSAG_THIRDPARTY_C_FLAGS} -D_DEFAULT_SOURCE -D_GNU_SOURCE"
        "CXXFLAGS=${VSAG_THIRDPARTY_CXX_FLAGS} -D_DEFAULT_SOURCE -D_GNU_SOURCE"
        "CPPFLAGS=-isystem ${CMAKE_INSTALL_PREFIX}/include"
        "LDFLAGS=${ld_flags_workaround} ${ld_flags} ${VSAG_THIRDPARTY_SHARED_LINKER_FLAGS}"
        "PATH=${VSAG_EXTERNAL_PATH}")

    if (DEFINED ACLOCAL_PATH AND NOT ACLOCAL_PATH STREQUAL "")
        list (APPEND common_configure_envs "ACLOCAL_PATH=${ACLOCAL_PATH}")
    endif ()
endif ()
