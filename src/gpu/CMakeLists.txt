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

if (ENABLE_CUDA)
    add_library (vsag_gpu STATIC
        cuda_backend.cu
        cuda_assign.cu
        cuda_seed.cu
        cuda_accumulate.cu)
    # The tree's C/C++ flags are language-scoped and do not reach nvcc; the
    # host pass follows the tree's C++ standard, and everything else is
    # CMake's per-configuration CUDA default.
    set_target_properties (vsag_gpu PROPERTIES
        CUDA_ARCHITECTURES "${VSAG_CUDA_ARCHITECTURES}"
        CUDA_STANDARD ${CMAKE_CXX_STANDARD}
        CUDA_STANDARD_REQUIRED ON)
    # Private: no other translation unit needs the macro or the toolkit's
    # headers. Everything from the toolkit is linked statically, so a build with
    # the backend on still loads on a machine that has no CUDA installed and
    # reports no device, which is the contract the header states. cuBLAS pulls
    # in cublasLt and culibos. It costs 108 MiB, 1151 MiB against 1042, measured
    # on the Release build with CUDA 12.4; a toolkit change moves that, so the
    # number is worth re-measuring rather than trusting. It is the price of the
    # contract either way: a build that links cuBLAS dynamically cannot load at
    # all where the toolkit is absent, which would mean shipping two builds
    # instead of deciding at runtime.
    target_compile_definitions (vsag_gpu PRIVATE VSAG_ENABLE_CUDA=1)
    target_link_libraries (vsag_gpu PRIVATE
        CUDA::cudart_static
        CUDA::cublas_static
        CUDA::cublasLt_static
        CUDA::culibos)
else ()
    add_library (vsag_gpu STATIC cuda_backend_stub.cpp)
    # Only the stub is instrumented; nvcc does not accept --coverage.
    target_link_libraries (vsag_gpu PRIVATE coverage_config)
endif ()

if (ENABLE_TESTS)
    file (GLOB GPU_TESTS "*_test.cpp")
    add_library (gpu_test STATIC ${GPU_TESTS})
    target_link_libraries (gpu_test PRIVATE Catch2::Catch2 vsag_gpu unittest_deps)
    if (ENABLE_CUDA)
        # The test switches on the same macro and talks to the runtime itself.
        target_compile_definitions (gpu_test PRIVATE VSAG_ENABLE_CUDA=1)
        target_link_libraries (gpu_test PRIVATE CUDA::cudart_static)
    endif ()
    add_dependencies (gpu_test Catch2)
endif ()
