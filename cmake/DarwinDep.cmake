
# Copyright 2025-present the vsag project
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

if (APPLE)
    set (ld_flags_workaround "-Wl,-rpath,@loader_path")
    if (NOT TARGET vsag_openmp)
        add_library (vsag_openmp INTERFACE)
    endif ()

    option (VSAG_OPENMP_STATIC "Link the macOS wheel's OpenMP runtime statically" OFF)

    # Prefer Homebrew's libomp when using AppleClang.
    find_program (BREW_EXECUTABLE brew)
    set (LIBOMP_PREFIX "" CACHE PATH "Prefix of a custom macOS OpenMP installation")
    if (BREW_EXECUTABLE AND NOT LIBOMP_PREFIX AND NOT VSAG_OPENMP_STATIC)
        execute_process (
            COMMAND ${BREW_EXECUTABLE} --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            RESULT_VARIABLE LIBOMP_PREFIX_RESULT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if (NOT LIBOMP_PREFIX_RESULT EQUAL 0)
            set (LIBOMP_PREFIX "")
        endif ()
    endif ()

    # The Python wheel hides all but vsag_* symbols. Link libomp into that same image so
    # Mach-O weak C++ symbols cannot coalesce with another package's OpenMP runtime.
    # Normal C++ builds retain the dynamic runtime.
    if (NOT VSAG_OPENMP_STATIC)
        find_package (OpenMP QUIET COMPONENTS CXX)
    endif ()
    if (VSAG_OPENMP_STATIC)
        if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message (FATAL_ERROR "VSAG_OPENMP_STATIC requires Clang on macOS")
        endif ()
        include (${CMAKE_CURRENT_LIST_DIR}/VSAGMacOSOpenMP.cmake)
        vsag_configure_private_openmp ()
    elseif (OpenMP_CXX_FOUND)
        message (STATUS "Found OpenMP via CMake: ${OpenMP_CXX_LIB_NAMES}")
        target_link_libraries (vsag_openmp INTERFACE OpenMP::OpenMP_CXX)
        foreach (_openmp_lib IN LISTS OpenMP_CXX_LIBRARIES)
            if (IS_ABSOLUTE "${_openmp_lib}" AND EXISTS "${_openmp_lib}")
                get_filename_component (_openmp_lib_dir "${_openmp_lib}" DIRECTORY)
                target_link_options (vsag_openmp INTERFACE "-Wl,-rpath,${_openmp_lib_dir}")
            endif ()
        endforeach ()
        if (LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
            target_link_options (vsag_openmp INTERFACE "-Wl,-rpath,${LIBOMP_PREFIX}/lib")
        endif ()
    elseif (LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
        message (STATUS "Configuring OpenMP from Homebrew libomp: ${LIBOMP_PREFIX}")
        target_compile_options (vsag_openmp INTERFACE -Xclang -fopenmp)
        target_include_directories (vsag_openmp INTERFACE "${LIBOMP_PREFIX}/include")
        target_link_libraries (vsag_openmp INTERFACE "${LIBOMP_PREFIX}/lib/libomp.dylib")
        target_link_options (vsag_openmp INTERFACE "-Wl,-rpath,${LIBOMP_PREFIX}/lib")
    else ()
        message (FATAL_ERROR "OpenMP not found on macOS. Run brew install libomp, or configure with -DLIBOMP_PREFIX=/path/to/libomp (containing include/omp.h and lib/libomp.dylib).")
    endif ()

    # Find LAPACK - will automatically use Accelerate framework on macOS
    find_package (LAPACK)
    if (LAPACK_FOUND)
        message (STATUS "Found LAPACK (using Accelerate framework on macOS)")
        # LAPACK libraries are in LAPACK_LIBRARIES variable
    else ()
        message (WARNING "LAPACK not found")
    endif ()

    # Find gfortran and its library path for OpenBLAS
    find_program (GFORTRAN_EXECUTABLE NAMES gfortran)
    if (GFORTRAN_EXECUTABLE)
        execute_process (
            COMMAND ${GFORTRAN_EXECUTABLE} -print-file-name=libgfortran.dylib
            OUTPUT_VARIABLE GFORTRAN_LIB
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if (EXISTS "${GFORTRAN_LIB}" AND NOT IS_DIRECTORY "${GFORTRAN_LIB}")
            get_filename_component (GFORTRAN_LIB_DIR "${GFORTRAN_LIB}" DIRECTORY)
            list (APPEND CMAKE_INSTALL_RPATH "${GFORTRAN_LIB_DIR}")
        else ()
            unset (GFORTRAN_LIB)
            message (WARNING
                     "gfortran found but libgfortran.dylib not found via -print-file-name; "
                     "OpenBLAS link may fail")
        endif ()
    else ()
        message (WARNING "gfortran not found; OpenBLAS/LAPACKE features may not link on macOS")
    endif ()

    # Fixup: some scripts (e.g. OpenBLAS fallback) set BLAS_LIBRARIES/LAPACK_LIBRARIES to include
    # a bare `gfortran` token (expands to -lgfortran). On macOS libgfortran is often not in the
    # default linker search paths, causing: `ld: library 'gfortran' not found`.
    function (vsag_darwin_fixup_blas_lapack_libs)
        if (NOT APPLE)
            return ()
        endif ()
        if (NOT DEFINED GFORTRAN_LIB OR NOT EXISTS "${GFORTRAN_LIB}")
            return ()
        endif ()

        foreach (_var BLAS_LIBRARIES LAPACK_LIBRARIES)
            if (DEFINED ${_var})
                set (_new_list "")
                foreach (_item IN LISTS ${_var})
                    if (_item STREQUAL "gfortran")
                        list (APPEND _new_list "${GFORTRAN_LIB}")
                    else ()
                        list (APPEND _new_list "${_item}")
                    endif ()
                endforeach ()
                # Keep cache in sync so downstream includes/targets see the rewritten list.
                set (${_var} "${_new_list}" CACHE STRING "" FORCE)
            endif ()
        endforeach ()
    endfunction ()

    # Run after the whole top-level configure has defined BLAS/LAPACK variables (order independent).
    cmake_language (DEFER CALL vsag_darwin_fixup_blas_lapack_libs)
else ()
    set (ld_flags_workaround "-Wl,-rpath=\\$\\$ORIGIN")
endif ()
