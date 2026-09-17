cmake_minimum_required(VERSION 3.18)
project(vsag_lite VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# A separate entry point deliberately avoids the Full build's dependency closure.
get_filename_component(VSAG_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
option(ENABLE_TESTS "Build Lite tests with the repository's Catch2" OFF)
option(ENABLE_COVERAGE "Instrument Lite sources for gcov" OFF)

set(VSAG_LITE_SOURCES
    "${VSAG_ROOT}/src/lite/index.cpp"
    "${VSAG_ROOT}/src/lite/fp32_distance.cpp")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$"
   AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
    # Only these small Lite translation units include ISA-specific VSAG traits.
    list(APPEND VSAG_LITE_SOURCES
        "${VSAG_ROOT}/src/lite/fp32_distance_sse.cpp"
        "${VSAG_ROOT}/src/lite/fp32_distance_avx2.cpp"
        "${VSAG_ROOT}/src/lite/fp32_distance_avx512.cpp")
    set_source_files_properties("${VSAG_ROOT}/src/lite/fp32_distance_sse.cpp"
        PROPERTIES COMPILE_FLAGS
        "-msse4.1")
    set_source_files_properties("${VSAG_ROOT}/src/lite/fp32_distance_avx2.cpp"
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
    set_source_files_properties("${VSAG_ROOT}/src/lite/fp32_distance_avx512.cpp"
        PROPERTIES COMPILE_FLAGS
        "-mavx512f -mavx512dq -mavx512bw -mavx512vl")
    set(VSAG_LITE_HAS_X86_SIMD ON)
endif()
add_library(vsag_lite_objects OBJECT ${VSAG_LITE_SOURCES})
add_library(vsag_lite SHARED $<TARGET_OBJECTS:vsag_lite_objects>)
add_library(vsag_lite_static STATIC $<TARGET_OBJECTS:vsag_lite_objects>)
add_library(vsag::lite ALIAS vsag_lite)
set_target_properties(vsag_lite_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(vsag_lite PROPERTIES OUTPUT_NAME vsag-lite EXPORT_NAME lite)
set_target_properties(vsag_lite_static PROPERTIES OUTPUT_NAME vsag-lite)
foreach(target vsag_lite_objects vsag_lite vsag_lite_static)
    target_compile_features(${target} PUBLIC cxx_std_17)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${VSAG_ROOT}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE "${VSAG_ROOT}/src")
endforeach()
target_compile_options(vsag_lite_objects PRIVATE -Wall -Wextra -Wpedantic)
if(VSAG_LITE_HAS_X86_SIMD)
    target_compile_definitions(vsag_lite_objects PRIVATE VSAG_LITE_HAS_X86_SIMD)
endif()
if(ENABLE_COVERAGE)
    target_compile_options(vsag_lite_objects PRIVATE --coverage -O0 -g)
    target_link_options(vsag_lite PRIVATE --coverage)
endif()

if(ENABLE_TESTS)
    include("${VSAG_ROOT}/cmake/VSAGThirdPartyOverride.cmake")
    include("${VSAG_ROOT}/extern/catch2/catch2.cmake")
    add_executable(lite_tests "${VSAG_ROOT}/src/lite/index_test.cpp")
    target_link_libraries(lite_tests PRIVATE vsag::lite Catch2::Catch2WithMain)
    if(VSAG_LITE_HAS_X86_SIMD)
        target_compile_definitions(lite_tests PRIVATE VSAG_LITE_HAS_X86_SIMD)
    endif()
    enable_testing()
    add_test(NAME lite_unit COMMAND lite_tests "[lite]")
    add_test(NAME lite_scale COMMAND lite_tests "[lite-scale]")
endif()

include(GNUInstallDirs)
install(TARGETS vsag_lite EXPORT vsag-lite-config
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS vsag_lite_static ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
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
