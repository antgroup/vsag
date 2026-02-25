
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


add_subdirectory (simd)
add_subdirectory (io)
add_subdirectory (quantization)
add_subdirectory (impl)
add_subdirectory (storage)
add_subdirectory (attr)
add_subdirectory (datacell)
add_subdirectory (algorithm)
add_subdirectory (utils)
add_subdirectory (factory)
add_subdirectory (analyzer)

file (GLOB CPP_SRCS "*.cpp")
list (FILTER CPP_SRCS EXCLUDE REGEX "_test.cpp")

file (GLOB CPP_INDEX_SRCS "index/*.cpp")
list (FILTER CPP_INDEX_SRCS EXCLUDE REGEX "_test.cpp")

set (VSAG_SRCS ${CPP_SRCS} ${CPP_INDEX_SRCS})

add_library (vsag SHARED ${VSAG_SRCS})
add_library (vsag_static STATIC ${VSAG_SRCS})

set (VSAG_DEP_LIBS antlr4-autogen antlr4-runtime diskann simd io quantizer storage utils factory analyzer
      datacell ${IMPL_LIBS} ${ALGORITHM_LIBS} attr pthread m dl fmt::fmt nlohmann_json::nlohmann_json roaring ${BLAS_LIBRARIES})

target_link_libraries (vsag ${VSAG_DEP_LIBS} coverage_config)
target_link_libraries (vsag_static ${VSAG_DEP_LIBS} coverage_config)

add_dependencies (vsag spdlog)
add_dependencies (vsag_static spdlog)

maybe_add_dependencies (vsag antlr4 spdlog roaring openblas boost mkl)
maybe_add_dependencies (vsag_static antlr4 spdlog roaring openblas boost mkl)

if (ENABLE_TESTS)
    file (GLOB VSAG_TESTS "*_test.cpp")
    file (GLOB INDEX_TESTS "index/*_test.cpp")
    list (APPEND VSAG_TESTS ${INDEX_TESTS})
    add_library (vsag_test STATIC ${VSAG_TESTS})
    target_link_libraries (vsag_test PRIVATE Catch2::Catch2 vsag)
    add_dependencies (vsag_test spdlog Catch2)
endif ()
