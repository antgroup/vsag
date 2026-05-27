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

find_package (Git)

# ---- Configure-time version detection ---------------------------------------
#
# Resolve VSAG_VERSION (full git-describe string) and the parsed numeric
# triple VSAG_VERSION_MAJOR/MINOR/PATCH up-front so they can drive
# set_target_properties(... VERSION SOVERSION) on the library and
# write_basic_package_version_file() on the installed CMake config package.
#
# The richer git-describe string (including -dirty / -gSHA suffix) is still
# regenerated at build time via the `version` custom target below so the
# in-binary VSAG_VERSION reflects the exact source tree.
#
# `VSAG_VERSION` may be supplied externally (e.g. `-DVSAG_VERSION=v1.2.3` for
# package maintainers building from a source tarball without a .git tree).
# Only fall back to `git describe` when no value has been provided.
# Use ${PROJECT_SOURCE_DIR} so the lookup still resolves to the vsag tree
# when this project is consumed via add_subdirectory()/FetchContent.

if (NOT DEFINED VSAG_VERSION OR VSAG_VERSION STREQUAL "")
    if (GIT_EXECUTABLE)
        execute_process (
            COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty --match "v*"
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            OUTPUT_VARIABLE _vsag_git_describe
            RESULT_VARIABLE _vsag_git_describe_rc
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if (NOT _vsag_git_describe_rc)
            set (VSAG_VERSION "${_vsag_git_describe}")
        endif ()
    endif ()
endif ()

if (NOT DEFINED VSAG_VERSION OR VSAG_VERSION STREQUAL "")
    set (VSAG_VERSION "v0.0.0-unknown")
    message (WARNING
        "Failed to determine VSAG_VERSION from Git tags. "
        "Using default version \"${VSAG_VERSION}\".")
endif ()

string (REGEX MATCH "^v?([0-9]+)\\.([0-9]+)\\.([0-9]+)" _vsag_ver_match "${VSAG_VERSION}")
if (_vsag_ver_match)
    set (VSAG_VERSION_MAJOR "${CMAKE_MATCH_1}")
    set (VSAG_VERSION_MINOR "${CMAKE_MATCH_2}")
    set (VSAG_VERSION_PATCH "${CMAKE_MATCH_3}")
else ()
    set (VSAG_VERSION_MAJOR 0)
    set (VSAG_VERSION_MINOR 0)
    set (VSAG_VERSION_PATCH 0)
endif ()

set (VSAG_VERSION_TRIPLE
    "${VSAG_VERSION_MAJOR}.${VSAG_VERSION_MINOR}.${VSAG_VERSION_PATCH}")

message (STATUS "vsag version: ${VSAG_VERSION} (parsed ${VSAG_VERSION_TRIPLE})")

# ---- Build-time regeneration of src/version.h -------------------------------
#
# The custom target itself does not depend on any library target, so it can
# be created here as part of configure-time setup. It is wired as a
# build-order dependency of `vsag` / `vsag_static` next to those targets'
# declarations in src/CMakeLists.txt.

add_custom_target (version
    ${CMAKE_COMMAND}
    -D SRC=${PROJECT_SOURCE_DIR}/src/version.h.in
    -D DST=${PROJECT_SOURCE_DIR}/src/version.h
    -D GIT_EXECUTABLE=${GIT_EXECUTABLE}
    -D VSAG_VERSION=${VSAG_VERSION}
    -P ${PROJECT_SOURCE_DIR}/cmake/GenerateVersionHeader.cmake)
