
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


include (${CMAKE_CURRENT_SOURCE_DIR}/eval_core_sources.cmake)

set (eval_srcs
    ${VSAG_EVAL_CORE_SRCS}
    exporter/exporter.cpp
    exporter/formatter.cpp
    monitor/http_server_monitor.cpp
    eval_config_loader.cpp
    eval_dataset_hdf5.cpp
    eval_job.cpp)

add_library (vsag_eval_common INTERFACE)
target_compile_options (vsag_eval_common INTERFACE -fopenmp)
target_link_options (vsag_eval_common INTERFACE -fopenmp)
target_link_libraries (vsag_eval_common INTERFACE
    pthread
    yaml-cpp
    argparse::argparse
    tabulate
    vsag_httplib
    vsag_src_common)
if (TARGET vsag_hdf5_headers)
    target_link_libraries (vsag_eval_common INTERFACE vsag_hdf5_headers)
endif ()

add_library (eval_obj OBJECT ${eval_srcs})
target_link_libraries (eval_obj PRIVATE vsag_eval_common)
add_dependencies (eval_obj hdf5 yaml-cpp tabulate)

add_executable (eval_performance main.cpp)
target_link_libraries (eval_performance PRIVATE
    eval_obj
    simd
    vsag
    vsag_eval_common
    ${HDF5_CPP_STATIC_LIBRARY}
    ${HDF5_C_STATIC_LIBRARY}
    z dl)
add_dependencies (eval_performance hdf5)

if (ENABLE_TESTS)
    add_executable (eval_dataset_test
        eval_dataset_test.cpp)
    target_include_directories (eval_dataset_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries (eval_dataset_test PRIVATE
        Catch2::Catch2WithMain
        eval_obj
        vsag
        simd
        vsag_eval_common
        ${HDF5_CPP_STATIC_LIBRARY}
        ${HDF5_C_STATIC_LIBRARY}
        z dl)
    add_dependencies (eval_dataset_test hdf5 Catch2)

    add_executable (vsag_eval_c_api_test
        vsag_eval_c_api_test.cpp
        ${VSAG_EVAL_CORE_SRCS}
        vsag_eval_c_api.cpp)
    target_include_directories (vsag_eval_c_api_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries (vsag_eval_c_api_test PRIVATE
        Catch2::Catch2WithMain
        vsag
        simd
        vsag_eval_common
        vsag_src_common)
    add_dependencies (vsag_eval_c_api_test Catch2)
endif ()
