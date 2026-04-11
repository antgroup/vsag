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

# Build options.
option (ENABLE_JEMALLOC "Whether to link jemalloc to all executables" OFF)
option (ENABLE_ASAN "Whether to turn AddressSanitizer ON or OFF" OFF)
option (ENABLE_COVERAGE "Whether to turn unit test coverage ON or OFF" OFF)
option (ENABLE_FUZZ_TEST "Whether to turn Fuzz Test ON or OFF" OFF)
option (ENABLE_WERROR "Whether to error on warnings" ON)
option (ENABLE_TSAN "Whether to turn Thread Sanitizer ON or OFF" OFF)
option (ENABLE_FRAME_POINTER "Whether to build with -fno-omit-frame-pointer" ON)
option (ENABLE_THIN_LTO "Whether to build with thin lto -flto=thin" OFF)
option (ENABLE_CCACHE "Whether to enable ccache" OFF)
option (ENABLE_INTEL_MKL "Enable intel-mkl (x86 platform only)" ON)
option (ENABLE_CXX11_ABI "Use CXX11 ABI" ON)
option (ENABLE_LIBCXX "Use libc++ instead of libstdc++" OFF)
option (ENABLE_TOOLS "Whether compile vsag tools" OFF)
option (ENABLE_EXAMPLES "Whether compile examples" ON)
option (ENABLE_TESTS "Whether compile vsag tests" ON)
option (ENABLE_PYBINDS "Whether compile Python bindings" OFF)
option (ENABLE_NODE_BINDS "Whether compile Node.js bindings" OFF)
option (ENABLE_LIBAIO "Whether to enable libaio support" ON)
option (DISABLE_SSE_FORCE "Force disable sse and higher instructions" OFF)
option (DISABLE_AVX_FORCE "Force disable avx and higher instructions" OFF)
option (DISABLE_AVX2_FORCE "Force disable avx2 and higher instructions" OFF)
option (DISABLE_AVX512_FORCE "Force disable avx512 instructions" OFF)
option (DISABLE_AVX512VPOPCNTDQ_FORCE "Force disable avx512vpopcntdq instructions" OFF)
option (DISABLE_NEON_FORCE "Force disable neon instructions" OFF)
option (DISABLE_SVE_FORCE "Force disable sve instructions" OFF)

set (NUM_BUILDING_JOBS "4" CACHE STRING "Default compilation parallelism for third-party builds.")
set (BUILD_INFO_DIR "${CMAKE_BINARY_DIR}/.vsag-build-info" CACHE PATH "Metadata directory for ExternalProject state.")
set (DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/.vsag-downloads" CACHE PATH "Download cache directory for ExternalProject archives.")
set (BUILDING_PATH "" CACHE STRING "Optional PATH prefix for third-party build tools.")

set (_default_aclocal_path "")
if (DEFINED ENV{ACLOCAL_PATH} AND NOT "$ENV{ACLOCAL_PATH}" STREQUAL "")
    set (_default_aclocal_path "$ENV{ACLOCAL_PATH}")
endif ()
set (ACLOCAL_PATH "${_default_aclocal_path}" CACHE STRING "Optional ACLOCAL_PATH for third-party builds.")
unset (_default_aclocal_path)

file (MAKE_DIRECTORY "${BUILD_INFO_DIR}")
file (MAKE_DIRECTORY "${DOWNLOAD_DIR}")
