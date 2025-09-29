
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

target_link_libraries (unittests
                       PRIVATE
                       Catch2::Catch2 
                       "-Wl,--whole-archive"
                       simd_test vsag_test algorithm_test factory_test attr_test
                       datacell_test quantizer_test storage_test io_test utils_test impl_test
                       "-Wl,--no-whole-archive"
                       vsag
)
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
