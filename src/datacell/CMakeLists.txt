
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


file (GLOB DATACELL_SRCS "*.cpp")
list (FILTER DATACELL_SRCS EXCLUDE REGEX "_test.cpp")

set (RABITQ_SPLIT_FACTORY_SRCS
     "${CMAKE_CURRENT_SOURCE_DIR}/rabitq_split_datacell_factory_l2.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/rabitq_split_datacell_factory_ip.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/rabitq_split_datacell_factory_cosine.cpp")
list (REMOVE_ITEM DATACELL_SRCS ${RABITQ_SPLIT_FACTORY_SRCS})

add_library (rabitq_split_datacell_factory OBJECT ${RABITQ_SPLIT_FACTORY_SRCS})
target_link_libraries (
  rabitq_split_datacell_factory
  PRIVATE transform quantizer io attr coverage_config vsag_src_common)

add_library (datacell OBJECT ${DATACELL_SRCS})
target_link_libraries (datacell PRIVATE transform quantizer io attr coverage_config vsag_src_common)
if (NOT VSAG_DATACELL_COMPILE_JOBS STREQUAL "0")
    # Both object libraries share one budget on memory-limited builders.
    # With the default 0, all sources follow the build's global parallelism.
    set_property (GLOBAL APPEND PROPERTY JOB_POOLS
                  vsag_datacell_compile_pool=${VSAG_DATACELL_COMPILE_JOBS})
    set_property (TARGET datacell rabitq_split_datacell_factory
                  PROPERTY JOB_POOL_COMPILE vsag_datacell_compile_pool)
endif ()

if (ENABLE_TESTS)
    file (GLOB_RECURSE DATACELL_TESTS "*_test.cpp")
    add_library (datacell_test STATIC ${DATACELL_TESTS})
    target_link_libraries (datacell_test PRIVATE Catch2::Catch2 vsag_static unittest_deps)
    add_dependencies (datacell_test Catch2)
endif ()
