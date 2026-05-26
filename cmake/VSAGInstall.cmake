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

include (CMakePackageConfigHelpers)

# Where the CMake package config files will live, relative to the install
# prefix. Picked to match the convention used by most distros and what
# CMake searches first via CMAKE_PREFIX_PATH.
set (VSAG_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/vsag")

install (DIRECTORY include/
         DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# Only the shared library participates in the exported package. The static
# library currently aggregates a number of internal helper targets
# (`factory`, `algorithm`, `simd`, ...) that are not themselves part of the
# public install surface, so exporting it would force every helper to be
# installed and namespaced as well. The static archive is still installed
# as a file for users who build vsag in-tree, but it is not advertised as
# a `find_package()` imported target.
install (TARGETS vsag
         EXPORT vsagTargets
         LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
         RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
         ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
         INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install (TARGETS vsag_static
         ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

install (EXPORT vsagTargets
         FILE vsagTargets.cmake
         NAMESPACE vsag::
         DESTINATION "${VSAG_INSTALL_CMAKEDIR}")

configure_package_config_file (
    "${CMAKE_CURRENT_LIST_DIR}/vsagConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/vsagConfig.cmake"
    INSTALL_DESTINATION "${VSAG_INSTALL_CMAKEDIR}")

write_basic_package_version_file (
    "${CMAKE_CURRENT_BINARY_DIR}/vsagConfigVersion.cmake"
    VERSION "${VSAG_VERSION_TRIPLE}"
    COMPATIBILITY SameMajorVersion)

install (FILES
    "${CMAKE_CURRENT_BINARY_DIR}/vsagConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/vsagConfigVersion.cmake"
    DESTINATION "${VSAG_INSTALL_CMAKEDIR}")
