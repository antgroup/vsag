
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

add_library (datacell OBJECT ${DATACELL_SRCS})
target_link_libraries(datacell PRIVATE transform quantizer io attr coverage_config)
add_dependencies (datacell spdlog fmt::fmt)

if (ENABLE_TESTS)
    file (GLOB_RECURSE DATACELL_TESTS "*_test.cpp")
    add_library (datacell_test STATIC ${DATACELL_TESTS})
    target_link_libraries (datacell_test PRIVATE Catch2::Catch2 vsag)
    add_dependencies (datacell_test spdlog Catch2)
endif ()
