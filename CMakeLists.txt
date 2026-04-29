
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


cmake_minimum_required (VERSION 3.18)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy (SET CMP0135 NEW)
endif ()
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.30.0")
    cmake_policy (SET CMP0169 OLD)
endif ()

project (VSAG LANGUAGES C CXX)

include (GNUInstallDirs)               # Standard install directory variables.
include (cmake/ColorizeMessage.cmake)  # Colored status output helpers.
include (cmake/VSAGHelpers.cmake)      # Shared helper functions.
include (cmake/VSAGOptions.cmake)      # Cache options and build defaults.

set (VSAG_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if (NOT VSAG_TARGET_PROCESSOR)
    set (VSAG_TARGET_PROCESSOR "${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif ()

message (STATUS "CPU ARCH: ${Yellow}${VSAG_TARGET_PROCESSOR}${CR}")

include (cmake/VSAGCompilerConfig.cmake)        # Compiler, ABI, sanitizer, and linker setup.
include (cmake/VSAGSystemDeps.cmake)            # System dependency discovery.
include (cmake/DarwinDep.cmake)                 # macOS-specific dependency handling.
include (cmake/VSAGExternalProjectConfig.cmake) # Shared ExternalProject flags and env.
include (cmake/VSAGThirdParty.cmake)            # Third-party dependency declarations.
include (cmake/VSAGDependencyTargets.cmake)     # Shared interface/header dependency targets.

include_directories (${PROJECT_SOURCE_DIR}/include)
include_directories (${PROJECT_SOURCE_DIR}/src)

if (ENABLE_TESTS)
    # Create unittest_deps first so src test targets can use it
    include_directories (${PROJECT_SOURCE_DIR}/tests/fixtures)
    add_subdirectory (tests/fixtures)
endif ()

add_subdirectory (src)
if (ENABLE_MOCKIMPL)
    add_subdirectory (mockimpl)
endif ()
if (ENABLE_EXAMPLES)
    add_subdirectory (examples/cpp)
endif ()
if (ENABLE_TOOLS AND ENABLE_CXX11_ABI)
    add_subdirectory (tools)
endif ()
if (ENABLE_TESTS)
    add_subdirectory (tests)
    target_compile_definitions (vsag PRIVATE ENABLE_TESTS)
endif ()

include (cmake/VSAGPythonBindings.cmake)  # Optional Python bindings.
include (cmake/VSAGNodeBindings.cmake)    # Optional Node.js bindings.
include (cmake/VSAGInstall.cmake)         # Install rules.
include (cmake/VSAGVersion.cmake)         # Generated version header target.
