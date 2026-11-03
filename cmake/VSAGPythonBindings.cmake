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

include_guard (GLOBAL)

if (NOT ENABLE_PYBINDS)
    return ()
endif ()

# pyvsag loads this library through cffi, so it only needs the C ABI: no Python headers and no
# dependency on the Python version. Only the vsag_* C symbols are exported.
add_library (vsag_c SHARED $<TARGET_OBJECTS:vsag_objects>)
add_dependencies (vsag_c version)
target_link_libraries (vsag_c PRIVATE vsag_src_common ${VSAG_DEP_LIBS} coverage_config)
vsag_link_openmp_private (vsag_c)
maybe_add_dependencies (vsag_c antlr4 roaring)
if (VSAG_BLAS_BACKEND STREQUAL "openblas")
    maybe_add_dependencies (vsag_c openblas)
elseif (VSAG_BLAS_BACKEND STREQUAL "mkl")
    maybe_add_dependencies (vsag_c mkl)
endif ()
if (APPLE)
    target_link_options (vsag_c PRIVATE
        -Wl,-exported_symbol,_vsag_*)
else ()
    target_link_options (vsag_c PRIVATE
        -static-libgcc
        -Wl,--version-script=${PROJECT_SOURCE_DIR}/cmake/vsag_c.map)
endif ()

# find_package(Python3) will find the python interpreter.
# To use a specific python, set the CMake variable Python3_EXECUTABLE.
# For example: cmake -DPython3_EXECUTABLE=/usr/bin/python3 ...
find_package (Python3 REQUIRED COMPONENTS Interpreter Development.Module)

message (STATUS "Python3 found:")
message (STATUS "  Executable: ${Python3_EXECUTABLE}")
message (STATUS "  Version: ${Python3_VERSION}")
message (STATUS "  Include dirs: ${Python3_INCLUDE_DIRS}")
message (STATUS "  Module suffix: ${Python3_SOABI}")

include (extern/pybind11/pybind11.cmake)

pybind11_add_module (_pyvsag
    python_bindings/module.cpp
    python_bindings/index_binding.cpp
    python_bindings/logging_binding.cpp
    python_bindings/analyze_index_binding.cpp)
target_compile_options (_pyvsag PRIVATE -fopenmp)
target_link_libraries (_pyvsag PRIVATE pybind11::module vsag)
if (NOT APPLE)
    target_link_options (_pyvsag PRIVATE -static-libgcc)
endif ()
