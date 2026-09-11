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
    set_target_properties (vsag_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES "${VSAG_CUDA_ARCHITECTURES}")
    target_compile_options (vsag_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>)
    target_compile_definitions (vsag_gpu PUBLIC VSAG_ENABLE_CUDA=1)
    target_link_libraries (vsag_gpu PUBLIC CUDA::cudart)
else ()
    add_library (vsag_gpu STATIC cuda_backend_stub.cpp)
    set_target_properties (vsag_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # Only the stub is instrumented: it is ordinary C++ and the tests below
    # exercise it. nvcc does not accept --coverage, and a device-less CI could
    # not run the other branch anyway.
    target_link_libraries (vsag_gpu PRIVATE coverage_config)
endif ()

target_include_directories (vsag_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (ENABLE_TESTS)
    file (GLOB GPU_TESTS "*_test.cpp")
    add_library (gpu_test STATIC ${GPU_TESTS})
    # vsag_gpu carries VSAG_ENABLE_CUDA as a PUBLIC definition and the stub
    # cases are guarded on it. Linking it here is what makes that guard work:
    # without it those cases compile into a CUDA build and fail, because they
    # assert the backend is absent while the real one is linked in.
    target_link_libraries (gpu_test PRIVATE Catch2::Catch2 vsag_gpu vsag_static
                                            unittest_deps coverage_config)
    vsag_link_openmp_private (gpu_test)
    add_dependencies (gpu_test Catch2)
endif ()
