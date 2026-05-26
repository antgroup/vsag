
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

# Trigger build-time regeneration of src/version.h before vsag's TUs compile.
add_dependencies (vsag version)
add_dependencies (vsag_static version)

set_target_properties (vsag PROPERTIES
    C_VISIBILITY_PRESET default
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
    VERSION "${VSAG_VERSION_TRIPLE}"
    SOVERSION "${VSAG_VERSION_MAJOR}")

# Public include directory: usable both during the build (BUILD_INTERFACE)
# and after `make install` from a downstream find_package() consumer
# (INSTALL_INTERFACE).
foreach (_lib vsag vsag_static)
    target_include_directories (${_lib}
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
endforeach ()

set (VSAG_DEP_LIBS antlr4-autogen antlr4-runtime diskann simd io quantizer storage utils factory analyzer
      datacell ${IMPL_LIBS} ${ALGORITHM_LIBS} attr pthread m dl fmt::fmt nlohmann_json::nlohmann_json roaring ${BLAS_LIBRARIES})

# `vsag_src_common` aggregates third-party include paths and link
# dependencies (fmt::fmt, nlohmann_json::nlohmann_json, tsl::robin_map, and
# several internal *_headers INTERFACE libraries) that vsag's own source
# files use during compilation. None of these targets appear in vsag's
# public API headers, so they should not be propagated to install-side
# consumers via the exported target file. They are still propagated to
# in-tree consumers (examples, tests, tools) that include vsag headers
# directly via add_subdirectory(), since some of those targets transitively
# rely on, e.g., <nlohmann/json.hpp>. Wrapping the link in BUILD_INTERFACE
# is what gives us both: PUBLIC propagation in-tree, no leakage on install.
target_link_libraries (vsag
    PUBLIC $<BUILD_INTERFACE:vsag_src_common>
    PRIVATE ${VSAG_DEP_LIBS} coverage_config)
target_link_libraries (vsag_static
    PUBLIC $<BUILD_INTERFACE:vsag_src_common>
    PRIVATE ${VSAG_DEP_LIBS} coverage_config)

maybe_add_dependencies (vsag antlr4 roaring boost)
maybe_add_dependencies (vsag_static antlr4 roaring boost)
if (VSAG_BLAS_BACKEND STREQUAL "openblas")
    maybe_add_dependencies (vsag openblas)
    maybe_add_dependencies (vsag_static openblas)
elseif (VSAG_BLAS_BACKEND STREQUAL "mkl")
    maybe_add_dependencies (vsag mkl)
    maybe_add_dependencies (vsag_static mkl)
endif ()

if (ENABLE_TESTS)
    file (GLOB VSAG_TESTS "*_test.cpp")
    file (GLOB INDEX_TESTS "index/*_test.cpp")
    list (APPEND VSAG_TESTS ${INDEX_TESTS})
    add_library (vsag_test STATIC ${VSAG_TESTS})
    target_link_libraries (vsag_test PRIVATE Catch2::Catch2 vsag_static unittest_deps)
    add_dependencies (vsag_test Catch2)
endif ()
