cmake_minimum_required(VERSION 3.18)
project(vsag_lite VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# A separate entry point deliberately avoids the Full build's dependency closure.
get_filename_component(VSAG_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
option(ENABLE_TESTS "Build Lite tests with the repository's Catch2" OFF)
option(ENABLE_COVERAGE "Instrument Lite sources for gcov" OFF)

add_library(vsag_lite SHARED "${VSAG_ROOT}/src/lite/index.cpp")
add_library(vsag::lite ALIAS vsag_lite)
set_target_properties(vsag_lite PROPERTIES OUTPUT_NAME vsag-lite EXPORT_NAME lite)
target_compile_features(vsag_lite PUBLIC cxx_std_17)
target_include_directories(vsag_lite PUBLIC
    $<BUILD_INTERFACE:${VSAG_ROOT}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE "${VSAG_ROOT}/src")
target_compile_options(vsag_lite PRIVATE -Wall -Wextra -Wpedantic)
if(ENABLE_COVERAGE)
    target_compile_options(vsag_lite PRIVATE --coverage -O0 -g)
    target_link_options(vsag_lite PRIVATE --coverage)
endif()

if(ENABLE_TESTS)
    include("${VSAG_ROOT}/cmake/VSAGThirdPartyOverride.cmake")
    include("${VSAG_ROOT}/extern/catch2/catch2.cmake")
    add_executable(lite_tests "${VSAG_ROOT}/src/lite/index_test.cpp")
    target_link_libraries(lite_tests PRIVATE vsag::lite Catch2::Catch2WithMain)
endif()

include(GNUInstallDirs)
install(TARGETS vsag_lite EXPORT vsag-lite-config
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES "${VSAG_ROOT}/include/vsag/lite/index.h" DESTINATION include/vsag/lite)
# Header-only error contract dependencies; no Full runtime is linked.
install(FILES
    "${VSAG_ROOT}/include/vsag/errors.h"
    "${VSAG_ROOT}/include/vsag/expected.hpp"
    "${VSAG_ROOT}/include/vsag/options.h"
    "${VSAG_ROOT}/include/vsag/allocator.h"
    "${VSAG_ROOT}/include/vsag/logger.h"
    DESTINATION include/vsag)
install(EXPORT vsag-lite-config NAMESPACE vsag:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vsag-lite)
