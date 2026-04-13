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

add_subdirectory(allocator)
add_subdirectory(core)
add_subdirectory(data)
add_subdirectory(framework)

# Common dependencies for unit tests (no precompiled header)
add_library(unittest_deps STATIC unittest.cpp)
target_include_directories(unittest_deps PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(unittest_deps PUBLIC Catch2::Catch2 vsag_src_common)
