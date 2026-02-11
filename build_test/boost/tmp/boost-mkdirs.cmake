# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/runner/work/vsag/vsag/build_test/boost/source")
  file(MAKE_DIRECTORY "/home/runner/work/vsag/vsag/build_test/boost/source")
endif()
file(MAKE_DIRECTORY
  "/home/runner/work/vsag/vsag/build_test/boost/src/boost-build"
  "/home/runner/work/vsag/vsag/build_test/boost"
  "/home/runner/work/vsag/vsag/build_test/boost/tmp"
  "/home/runner/work/vsag/vsag/build_test/boost/src/boost-stamp"
  "/home/runner/work/vsag/vsag/build_test/boost/src"
  "/home/runner/work/vsag/vsag/build_test/boost/src/boost-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/runner/work/vsag/vsag/build_test/boost/src/boost-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/runner/work/vsag/vsag/build_test/boost/src/boost-stamp${cfgdir}") # cfgdir has leading slash
endif()
