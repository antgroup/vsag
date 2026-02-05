
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


#  unittests
add_executable (unittests
        test_main.cpp
        fixtures/fixtures.cpp
        fixtures/test_logger.cpp
)

if (APPLE)
        target_link_libraries (unittests
                        PRIVATE
                        Catch2::Catch2 
                        simd_test vsag_test algorithm_test factory_test attr_test
                        datacell_test quantizer_test storage_test io_test utils_test impl_test
                        vsag
        )
        # Ensure static test libraries are not dead-stripped on macOS.
        target_link_options (unittests PRIVATE
        "-Wl,-force_load,$<TARGET_FILE:simd_test>"
        "-Wl,-force_load,$<TARGET_FILE:vsag_test>"
        "-Wl,-force_load,$<TARGET_FILE:algorithm_test>"
        "-Wl,-force_load,$<TARGET_FILE:factory_test>"
        "-Wl,-force_load,$<TARGET_FILE:attr_test>"
        "-Wl,-force_load,$<TARGET_FILE:datacell_test>"
        "-Wl,-force_load,$<TARGET_FILE:quantizer_test>"
        "-Wl,-force_load,$<TARGET_FILE:storage_test>"
        "-Wl,-force_load,$<TARGET_FILE:io_test>"
        "-Wl,-force_load,$<TARGET_FILE:utils_test>"
        "-Wl,-force_load,$<TARGET_FILE:impl_test>"
        )
else()
        target_link_libraries (unittests
                        PRIVATE
                        Catch2::Catch2 
                        "-Wl,--whole-archive"
                        simd_test vsag_test algorithm_test factory_test attr_test
                        datacell_test quantizer_test storage_test io_test utils_test impl_test
                        "-Wl,--no-whole-archive"
                        vsag
        )
endif()

add_dependencies (unittests spdlog Catch2 vsag)

# function tests
file (GLOB FUNCTION_TESTS "test_*.cpp")
add_executable (functests
        ${FUNCTION_TESTS}
        fixtures/fixtures.cpp
        fixtures/test_dataset.cpp
        fixtures/test_dataset_pool.cpp
        fixtures/test_logger.cpp
)
target_include_directories (functests PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/spdlog/install/include
        ${HDF5_INCLUDE_DIRS})
target_link_libraries (functests PRIVATE Catch2::Catch2 vsag simd)
add_dependencies (functests spdlog Catch2)
