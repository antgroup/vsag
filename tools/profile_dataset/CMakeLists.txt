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

set (profile_dataset_srcs
    profile_types.cpp
    deterministic_random.cpp
    statistics.cpp
    sha256.cpp
    vector_math.cpp
    hdf5_source.cpp
    neighbor_search.cpp
    linear_algebra.cpp
    kmeans.cpp
    quantization_simulation.cpp
    dense_profiler.cpp
    sparse_profiler.cpp
    dataset_profiler.cpp
    profile_signals.cpp
    svg_chart.cpp
    html_report.cpp
    ai_analysis.cpp
    http_transport.cpp
    profile_cli.cpp)

add_library (profile_dataset_obj OBJECT ${profile_dataset_srcs})
target_include_directories (profile_dataset_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Only the generated src/version.h is used from the library sources. The tool does not link
# libvsag, so it has no runtime library dependency and prints no VSAG init banner.
target_include_directories (profile_dataset_obj PUBLIC ${PROJECT_SOURCE_DIR}/src)
add_dependencies (profile_dataset_obj version)
# Release builds use -Ofast. Reports must be reproducible and must count NaN/Inf values, so the
# profiler opts out of fast-math (reassociation, finite-math assumptions) and FMA contraction.
target_compile_options (profile_dataset_obj PRIVATE -fopenmp -fno-fast-math -ffp-contract=off)
target_link_libraries (profile_dataset_obj PUBLIC
    nlohmann_json::nlohmann_json
    argparse::argparse
    vsag_httplib)
if (TARGET vsag_hdf5_headers)
    target_link_libraries (profile_dataset_obj PUBLIC vsag_hdf5_headers)
endif ()
add_dependencies (profile_dataset_obj hdf5)

# HTTPS for the optional AI providers. cpp-httplib requires OpenSSL 3; without it only http://
# endpoints (for example a local gateway) are available and the tool says so.
find_package (OpenSSL 3.0 QUIET)
if (OPENSSL_FOUND)
    message (STATUS "profile_dataset: HTTPS AI endpoints enabled (OpenSSL ${OPENSSL_VERSION})")
    target_compile_definitions (profile_dataset_obj PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
    target_link_libraries (profile_dataset_obj PUBLIC OpenSSL::SSL OpenSSL::Crypto)
else ()
    message (STATUS "profile_dataset: OpenSSL 3 not found, AI endpoints limited to http://")
endif ()

set (profile_dataset_libs
    profile_dataset_obj
    ${HDF5_CPP_STATIC_LIBRARY}
    ${HDF5_C_STATIC_LIBRARY}
    z
    ${CMAKE_DL_LIBS}
    pthread)

add_executable (profile_dataset main.cpp)
target_link_libraries (profile_dataset PRIVATE ${profile_dataset_libs})
target_link_options (profile_dataset PRIVATE -fopenmp)

if (ENABLE_TESTS)
    add_executable (profile_dataset_test
        profile_core_test.cpp
        profile_reader_test.cpp
        profile_report_test.cpp
        profile_ai_test.cpp)
    target_compile_options (profile_dataset_test PRIVATE -fopenmp)
    target_link_libraries (profile_dataset_test PRIVATE Catch2::Catch2WithMain ${profile_dataset_libs})
    target_link_options (profile_dataset_test PRIVATE -fopenmp)
    add_dependencies (profile_dataset_test hdf5 Catch2)
endif ()
