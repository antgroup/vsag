
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


add_subdirectory (thread_pool)
add_subdirectory (allocator)
add_subdirectory (filter)
add_subdirectory (heap)
add_subdirectory (transform)
add_subdirectory (bitset)
add_subdirectory (cluster)
add_subdirectory (logger)
add_subdirectory (searcher)
add_subdirectory (odescent)
add_subdirectory (reorder)

file (GLOB IMPL_SRCS "*.cpp")
list (FILTER IMPL_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER IMPL_SRCS EXCLUDE REGEX kmeans_cluster.cpp)

add_library (impl OBJECT ${IMPL_SRCS})
target_link_libraries (impl PUBLIC transform thread_pool allocator heap 
    bitset logger filter cluster searcher odescent reorder fmt::fmt coverage_config)
maybe_add_dependencies (impl spdlog)

set (IMPL_LIBS transform thread_pool allocator heap bitset 
    logger filter impl cluster searcher odescent reorder PARENT_SCOPE)

if (ENABLE_TESTS)
    file (GLOB_RECURSE IMPL_TESTS "*_test.cpp")
    add_library (impl_test STATIC ${IMPL_TESTS})
    target_link_libraries (impl_test PRIVATE Catch2::Catch2 vsag)
    add_dependencies (impl_test spdlog Catch2)
endif ()
