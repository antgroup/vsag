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


set (ANALYZER_SRC
        hgraph_analyzer.cpp
        hgraph_analyzer.h
        analyzer.cpp
        analyzer.h
)

add_library (analyzer OBJECT ${ANALYZER_SRC})
target_link_libraries (analyzer PRIVATE vsag_src_common)

if (ENABLE_TESTS)
    file (GLOB ANALYZER_TESTS "*_test.cpp")
    add_library (analyzer_test STATIC ${ANALYZER_TESTS})
    target_include_directories (analyzer_test PRIVATE ${CMAKE_SOURCE_DIR}/tests/fixtures)
    target_link_libraries (analyzer_test PRIVATE Catch2::Catch2 vsag)
    add_dependencies (analyzer_test Catch2)
endif ()