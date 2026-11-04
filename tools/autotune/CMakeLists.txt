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

add_library (vsag_autotune SHARED
    autotune.cpp
    autotune_candidate.cpp
    autotune_evaluation.cpp
    $<TARGET_OBJECTS:eval_obj>)
add_library (vsag::autotune ALIAS vsag_autotune)
target_include_directories (vsag_autotune PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (vsag_autotune PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../eval)
target_link_libraries (vsag_autotune PUBLIC nlohmann_json::nlohmann_json vsag)
target_link_libraries (vsag_autotune PRIVATE
    simd
    vsag_eval_common
    ${HDF5_CPP_STATIC_LIBRARY}
    ${HDF5_C_STATIC_LIBRARY}
    z
    ${CMAKE_DL_LIBS})
add_dependencies (vsag_autotune hdf5)

add_executable (autotune main.cpp)
target_include_directories (autotune PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (autotune PRIVATE vsag_autotune vsag)

if (ENABLE_TESTS)
    add_executable (autotune_test autotune_test.cpp)
    target_include_directories (autotune_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../eval)
    target_link_libraries (autotune_test PRIVATE
        Catch2::Catch2WithMain
        vsag_autotune
        vsag
        vsag_eval_common
        ${HDF5_CPP_STATIC_LIBRARY}
        ${HDF5_C_STATIC_LIBRARY}
        z
        ${CMAKE_DL_LIBS})
    add_dependencies (autotune_test hdf5 Catch2)
endif ()
