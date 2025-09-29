
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


add_subdirectory (ivf_partition)
add_subdirectory (hnswlib)
add_subdirectory (sindi)

file (GLOB ALGORITHM_SRCS "*.cpp")
list (FILTER ALGORITHM_SRCS EXCLUDE REGEX "_test.cpp")

add_library(algorithm OBJECT ${ALGORITHM_SRCS})
target_link_libraries (algorithm PUBLIC coverage_config)
maybe_add_dependencies(algorithm spdlog fmt::fmt antlr4)


set (ALGORITHM_LIBS
        algorithm
        hnswlib
        ivf_partition
        sindi
        PARENT_SCOPE)

if (ENABLE_TESTS)
    file (GLOB_RECURSE ALGORITHM_TESTS "*_test.cpp")
    add_library (algorithm_test STATIC ${ALGORITHM_TESTS})
    target_link_libraries (algorithm_test PRIVATE Catch2::Catch2 vsag)
    add_dependencies (algorithm_test spdlog Catch2 antlr4)
endif ()
