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

include_guard (GLOBAL)

if (NOT ENABLE_NODE_BINDS)
    return ()
endif ()

# Locate Node.js and node-addon-api headers.
find_program (NODE_EXECUTABLE node REQUIRED)

# Get Node.js include directory.
execute_process (
    COMMAND ${NODE_EXECUTABLE} -p "require('node-addon-api').include_dir"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/typescript
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _napi_result)
if (NOT _napi_result EQUAL 0)
    message (FATAL_ERROR "Failed to locate node-addon-api. Run 'npm install' in typescript/.")
endif ()

execute_process (
    COMMAND ${NODE_EXECUTABLE} -e "console.log(require('path').dirname(process.execPath) + '/../include/node')"
    OUTPUT_VARIABLE NODE_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

message (STATUS "Node.js found:")
message (STATUS "  Executable: ${NODE_EXECUTABLE}")
message (STATUS "  Node include: ${NODE_INCLUDE_DIR}")
message (STATUS "  node-addon-api: ${NODE_ADDON_API_DIR}")

add_library (vsag_node SHARED
    node_bindings/module.cpp
    node_bindings/index_binding.cpp
    node_bindings/logging_binding.cpp)

set_target_properties (vsag_node PROPERTIES
    PREFIX ""
    SUFFIX ".node"
    CXX_STANDARD 17)

target_include_directories (vsag_node PRIVATE
    ${NODE_INCLUDE_DIR}
    ${NODE_ADDON_API_DIR})

target_compile_definitions (vsag_node PRIVATE
    NAPI_VERSION=8
    NODE_ADDON_API_DISABLE_DEPRECATED
    BUILDING_NODE_EXTENSION)

target_compile_options (vsag_node PRIVATE -fopenmp)
target_link_libraries (vsag_node PRIVATE vsag)

if (NOT APPLE)
    target_link_options (vsag_node PRIVATE -static-libgcc)
else ()
    # On macOS, mark Node.js symbols as undefined-dynamic-lookup.
    target_link_options (vsag_node PRIVATE -undefined dynamic_lookup)
endif ()

message (STATUS "Building vsag_node native addon")
