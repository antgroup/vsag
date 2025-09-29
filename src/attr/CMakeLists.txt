
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


file (GLOB ATTR_SRCS "executor/*.cpp")
list (FILTER ATTR_SRCS EXCLUDE REGEX "_test.cpp")

list (APPEND ATTR_SRCS
        expression_visitor.cpp
        attribute.cpp
        attr_type_schema.cpp
        multi_bitset_manager.cpp
        attr_value_map.cpp
        argparse.cpp
)

add_library (attr OBJECT ${ATTR_SRCS})
target_link_libraries (attr PUBLIC coverage_config)

add_dependencies (attr spdlog fmt::fmt antlr4)

if (ENABLE_TESTS)
    file (GLOB_RECURSE ATTR_TESTS "*_test.cpp")
    add_library (attr_test STATIC ${ATTR_TESTS})
    target_link_libraries (attr_test PRIVATE Catch2::Catch2 vsag)
    add_dependencies (attr_test spdlog Catch2 antlr4)
endif ()
