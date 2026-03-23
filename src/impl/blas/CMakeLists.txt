
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


add_library (vsag_blas OBJECT blas_function.cpp)
if (VSAG_BLAS_BACKEND STREQUAL "openblas")
    maybe_add_dependencies (vsag_blas openblas)
elseif (VSAG_BLAS_BACKEND STREQUAL "mkl")
    maybe_add_dependencies (vsag_blas mkl)
    target_include_directories (vsag_blas SYSTEM BEFORE PRIVATE ${MKL_INCLUDE_PATH})
    target_compile_definitions (vsag_blas PRIVATE VSAG_USE_MKL_HEADERS)
else ()
    message (FATAL_ERROR "Unsupported VSAG_BLAS_BACKEND='${VSAG_BLAS_BACKEND}'. Expected 'openblas' or 'mkl'.")
endif ()
