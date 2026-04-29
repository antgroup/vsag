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

add_library(fixtures_core OBJECT
    common.cpp
    temp_dir.cpp
    types.cpp
)

target_include_directories(fixtures_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/tests/fixtures
)

target_link_libraries(fixtures_core PUBLIC vsag_src_common)
vsag_link_openmp_private(fixtures_core)

set_target_properties(fixtures_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
