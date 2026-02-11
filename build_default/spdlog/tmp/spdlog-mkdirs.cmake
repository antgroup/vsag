# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/runner/work/vsag/vsag/build_default/spdlog/source")
  file(MAKE_DIRECTORY "/home/runner/work/vsag/vsag/build_default/spdlog/source")
endif()
file(MAKE_DIRECTORY
  "/home/runner/work/vsag/vsag/build_default/spdlog/src/spdlog-build"
  "/home/runner/work/vsag/vsag/build_default/spdlog"
  "/home/runner/work/vsag/vsag/build_default/spdlog/tmp"
  "/home/runner/work/vsag/vsag/build_default/spdlog/src/spdlog-stamp"
  "/home/runner/work/vsag/vsag/build_default/spdlog/src"
  "/home/runner/work/vsag/vsag/build_default/spdlog/src/spdlog-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/runner/work/vsag/vsag/build_default/spdlog/src/spdlog-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/runner/work/vsag/vsag/build_default/spdlog/src/spdlog-stamp${cfgdir}") # cfgdir has leading slash
endif()
