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

include (cmake/CheckSIMDCompilerFlag.cmake)
include (ExternalProject)

include (extern/json/json.cmake)
include (extern/antlr4/antlr4.cmake)
include (extern/boost/boost.cmake)

set (VSAG_BLAS_BACKEND "openblas")
if (VSAG_TARGET_PROCESSOR STREQUAL "x86_64" AND ENABLE_INTEL_MKL)
    set (VSAG_BLAS_BACKEND "mkl")
    include (extern/mkl/mkl.cmake)
else ()
    if (ENABLE_INTEL_MKL)
        message (WARNING
                 "Intel MKL is not supported on this architecture (${VSAG_TARGET_PROCESSOR}). "
                 "Falling back to OpenBLAS.")
    endif ()
    include (extern/openblas/openblas.cmake)
endif ()

include (extern/diskann/diskann.cmake)
include (extern/catch2/catch2.cmake)
include (extern/cpuinfo/cpuinfo.cmake)
include (extern/fmt/fmt.cmake)
include (extern/thread_pool/thread_pool.cmake)
include (extern/tsl/tsl.cmake)
include (extern/roaringbitmap/roaringbitmap.cmake)

if (ENABLE_TOOLS AND ENABLE_CXX11_ABI)
    include (extern/hdf5/hdf5.cmake)
    include (extern/argparse/argparse.cmake)
    include (extern/yaml-cpp/yaml-cpp.cmake)
    include (extern/tabulate/tabulate.cmake)
    include (extern/httplib/httplib.cmake)
endif ()
