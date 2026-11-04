# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
find_package (Python3 COMPONENTS Interpreter REQUIRED)
set (_pipnn_cache_generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
execute_process (
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate.py"
            --repo "${PROJECT_SOURCE_DIR}" --out "${_pipnn_cache_generated}"
    RESULT_VARIABLE _pipnn_cache_generation_result)
if (NOT _pipnn_cache_generation_result EQUAL 0)
    message (FATAL_ERROR "PiPNN cache experiment source generation failed")
endif ()
set_property (DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    generate.py bench.cpp derived.cpp warm_final.cpp
    "${PROJECT_SOURCE_DIR}/src/impl/pipnn/pipnn_graph_builder.cpp")

function (pipnn_cache_executable name source)
    add_executable (${name} "${source}")
    target_link_libraries (${name} PRIVATE vsag_static)
    target_include_directories (${name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${_pipnn_cache_generated}"
        "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/src/impl/pipnn")
    set_target_properties (${name} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
endfunction ()

pipnn_cache_executable (pipnn_cache_deletion bench.cpp)
pipnn_cache_executable (pipnn_cache_derived derived.cpp)
pipnn_cache_executable (pipnn_cache_exact final_reuse.cpp)
pipnn_cache_executable (pipnn_cache_warm warm_final.cpp)
pipnn_cache_executable (pipnn_cache_search "${_pipnn_cache_generated}/search_qps_driver.cpp")
pipnn_cache_executable (pipnn_cache_correctness correctness.cpp)
pipnn_cache_executable (pipnn_cache_certificate_check certificate_check.cpp)
