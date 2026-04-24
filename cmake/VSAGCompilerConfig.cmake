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

set (CMAKE_POSITION_INDEPENDENT_CODE ON)

set (VSAG_EXTERNAL_C_FLAGS "")
set (VSAG_EXTERNAL_CXX_FLAGS "")
set (VSAG_EXTERNAL_EXE_LINKER_FLAGS "")
set (VSAG_EXTERNAL_SHARED_LINKER_FLAGS "")

if (ENABLE_FUZZ_TEST)
    message (WARNING "ENABLE_FUZZ_TEST is set, but no top-level fuzz targets are currently defined.")
endif ()

if (NOT APPLE)
    if (ENABLE_CXX11_ABI)
        set (VSAG_GLIBCXX_USE_CXX11_ABI 1)
        message (STATUS "Using libstdc++ with ${Yellow}C++11 ABI${CR}")
    else ()
        set (VSAG_GLIBCXX_USE_CXX11_ABI 0)
        message (STATUS "Using libstdc++ with ${Yellow}pre-C++11 ABI${CR}")
    endif ()
    add_compile_definitions (_GLIBCXX_USE_CXX11_ABI=${VSAG_GLIBCXX_USE_CXX11_ABI})
endif ()

if (ENABLE_WERROR)
    vsag_add_compile_flag (-Werror)
endif ()

if (NOT APPLE)
    vsag_add_linker_flag (-static-libstdc++)
endif ()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    vsag_add_c_compile_flag (-gdwarf-4)
    vsag_add_cxx_compile_flag (-gdwarf-4)

    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15.0.0)
        vsag_add_cxx_compile_flag (-Wno-deprecated-builtins)
        message (STATUS "Clang >=15 detected: -Wno-deprecated-builtins enabled")
    endif ()
elseif (ENABLE_LIBCXX)
    message (FATAL_ERROR "gcc does not support using libc++")
endif ()

if (ENABLE_LIBCXX OR APPLE)
    message (STATUS "Using libc++ with C++11 ABI")
    vsag_add_cxx_compile_flag (-stdlib=libc++)

    if (APPLE)
        vsag_add_linker_flag (-stdlib=libc++)
    else ()
        set (LIBCXX_SEARCH_PATH /opt/alibaba-cloud-compiler/lib64)
        find_library (LIBCXX_STATIC libc++.a PATHS ${LIBCXX_SEARCH_PATH})
        find_library (LIBCXXABI_STATIC libc++abi.a PATHS ${LIBCXX_SEARCH_PATH})
        if (LIBCXX_STATIC AND LIBCXXABI_STATIC)
            get_filename_component (LIBCXX_DIR "${LIBCXX_STATIC}" DIRECTORY)
            message (STATUS "Found libc++ at ${LIBCXX_STATIC}")
            message (STATUS "Found libc++abi at ${LIBCXXABI_STATIC}")

            vsag_add_linker_flag (-fuse-ld=lld)
            vsag_add_linker_flag ("-Wl,-rpath,${LIBCXX_DIR}")
            set (CMAKE_CXX_STANDARD_LIBRARIES
                 "${CMAKE_CXX_STANDARD_LIBRARIES} ${LIBCXX_STATIC} ${LIBCXXABI_STATIC}")
        else ()
            message (FATAL_ERROR "libc++ or libc++abi not found")
        endif ()
    endif ()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    vsag_add_cxx_compile_flag (-stdlib=libstdc++)
endif ()

set (CMAKE_CXX_STANDARD 17)
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 17)
    set (CMAKE_CXX_STANDARD 20)
endif ()
set (CMAKE_CXX_STANDARD_REQUIRED ON)

if (ENABLE_FRAME_POINTER)
    vsag_add_compile_flag (-fno-omit-frame-pointer)
endif ()

vsag_add_cxx_compile_flag (-Werror=suggest-override)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    vsag_add_compile_flag (-g)
    vsag_add_compile_flag (-O0)
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    vsag_add_compile_flag (-g)
    vsag_add_compile_flag (-Ofast)
elseif (CMAKE_BUILD_TYPE STREQUAL "Sanitize")
    vsag_add_compile_flag (-g)
    vsag_add_compile_flag (-O2)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-var-tracking-assignments")
    endif ()
endif ()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    vsag_add_c_compile_flag (-Wno-incompatible-pointer-types-discards-qualifiers)
endif ()

if (ENABLE_THIN_LTO)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        vsag_add_compile_flag (-flto=thin)
        vsag_add_linker_flag (-flto=thin)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        vsag_add_compile_flag (-flto)
        vsag_add_linker_flag (-flto)
    else ()
        message (FATAL_ERROR "ENABLE_THIN_LTO requires Clang or GCC")
    endif ()
endif ()

add_library (coverage_config INTERFACE)
if (ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32 AND NOT APPLE)
    target_compile_options (coverage_config INTERFACE
        -O0
        -g
        --coverage
        -fprofile-update=atomic)
    target_link_options (coverage_config INTERFACE --coverage)
endif ()

if (ENABLE_ASAN AND ENABLE_TSAN)
    message (FATAL_ERROR "ENABLE_TSAN cannot be combined with ENABLE_ASAN")
endif ()

if (ENABLE_ASAN)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
        OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0))
        message (STATUS "Compiling with AddressSanitizer and UndefinedBehaviorSanitizer")
        set (VSAG_SANITIZER_FLAGS
            -g
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
            -fno-optimize-sibling-calls
            -fsanitize-recover=address
            -fno-sanitize=vptr)
    else ()
        message (STATUS "Compiling with AddressSanitizer")
        set (VSAG_SANITIZER_FLAGS
            -g
            -fsanitize=address
            -fno-omit-frame-pointer
            -static-libasan)
    endif ()

    foreach (flag ${VSAG_SANITIZER_FLAGS})
        vsag_add_compile_flag (${flag})
        vsag_add_linker_flag (${flag})
    endforeach ()
endif ()

if (ENABLE_TSAN)
    set (CMAKE_REQUIRED_FLAGS "-fsanitize=thread")
    set (ENV{TSAN_OPTIONS} "report_atomic_races=0")
    foreach (flag -fsanitize=thread -g)
        vsag_add_compile_flag (${flag})
        vsag_add_linker_flag (${flag})
    endforeach ()
endif ()
