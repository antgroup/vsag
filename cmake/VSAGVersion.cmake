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

add_custom_target (version
    ${CMAKE_COMMAND}
    -D SRC=${CMAKE_CURRENT_SOURCE_DIR}/src/version.h.in
    -D DST=${CMAKE_CURRENT_SOURCE_DIR}/src/version.h
    -D GIT_EXECUTABLE=${GIT_EXECUTABLE}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateVersionHeader.cmake)

if (TARGET vsag)
    add_dependencies (vsag version)
endif ()
if (TARGET vsag_static)
    add_dependencies (vsag_static version)
endif ()
