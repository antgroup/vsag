
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


if (GIT_EXECUTABLE)
  get_filename_component (SRC_DIR ${SRC} DIRECTORY)
  # Generate a git-describe version string from Git repository tags
  execute_process (
    COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty --match "v*"
    WORKING_DIRECTORY ${SRC_DIR}
    OUTPUT_VARIABLE GIT_DESCRIBE_VERSION
    RESULT_VARIABLE GIT_DESCRIBE_ERROR_CODE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  if (NOT GIT_DESCRIBE_ERROR_CODE)
    set (VSAG_VERSION ${GIT_DESCRIBE_VERSION})
  endif ()
endif ()

# Final fallback: Just use a bogus version string that is semantically older
# than anything else and spit out a warning to the developer.
if (NOT DEFINED VSAG_VERSION)
  set (VSAG_VERSION v0.0.0-unknown)
  message (WARNING "Failed to determine VSAG_VERSION from Git tags. Using default version \"${VSAG_VERSION}\".")
endif ()

message (STATUS "vsag version: ${VSAG_VERSION}")
configure_file (${SRC} ${DST} @ONLY)
