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
    add_library (vsag_gpu STATIC cuda_backend.cu)
    # The tree's C/C++ flags are language-scoped and do not reach nvcc; the
    # host pass follows the tree's C++ standard, and everything else is
    # CMake's per-configuration CUDA default.
    set_target_properties (vsag_gpu PROPERTIES
        CUDA_ARCHITECTURES "${VSAG_CUDA_ARCHITECTURES}"
        CUDA_STANDARD ${CMAKE_CXX_STANDARD}
        CUDA_STANDARD_REQUIRED ON)
    # Private: no other translation unit needs the macro or the toolkit's
    # headers. The static runtime keeps libcudart.so off the result.
    target_compile_definitions (vsag_gpu PRIVATE VSAG_ENABLE_CUDA=1)
    target_link_libraries (vsag_gpu PRIVATE CUDA::cudart_static)
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
