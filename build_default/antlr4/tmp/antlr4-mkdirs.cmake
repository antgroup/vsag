# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/runner/work/vsag/vsag/build_default/antlr4/source")
  file(MAKE_DIRECTORY "/home/runner/work/vsag/vsag/build_default/antlr4/source")
endif()
file(MAKE_DIRECTORY
  "/home/runner/work/vsag/vsag/build_default/antlr4/source/runtime/Cpp"
  "/home/runner/work/vsag/vsag/build_default/antlr4"
  "/home/runner/work/vsag/vsag/build_default/antlr4/tmp"
  "/home/runner/work/vsag/vsag/build_default/antlr4/src/antlr4-stamp"
  "/home/runner/work/vsag/vsag/build_default/antlr4/src"
  "/home/runner/work/vsag/vsag/build_default/antlr4/src/antlr4-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/runner/work/vsag/vsag/build_default/antlr4/src/antlr4-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/runner/work/vsag/vsag/build_default/antlr4/src/antlr4-stamp${cfgdir}") # cfgdir has leading slash
endif()
