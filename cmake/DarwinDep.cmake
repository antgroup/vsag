
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

if(APPLE)
    set(ld_flags_workaround "-Wl,-rpath,@loader_path")
    
    # Find OpenMP (libomp) on macOS.
    # Try to detect Homebrew prefix automatically
    if (NOT DEFINED OpenMP_ROOT)
        # Try to get Homebrew prefix
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE BREW_LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if (BREW_LIBOMP_PREFIX AND EXISTS "${BREW_LIBOMP_PREFIX}")
            set(OpenMP_ROOT "${BREW_LIBOMP_PREFIX}")
            message(STATUS "Auto-detected OpenMP from Homebrew: ${OpenMP_ROOT}")
        endif()
    endif()

    set(OpenMP_C_FOUND FALSE)
    set(OpenMP_CXX_FOUND FALSE)

    # Search for libomp - CMake will automatically search standard system paths
    find_library(VSAG_LIBOMP_LIBRARY NAMES omp libomp
        HINTS "${OpenMP_ROOT}"
        PATH_SUFFIXES lib
    )
    find_path(VSAG_LIBOMP_INCLUDE_DIR NAMES omp.h
        HINTS "${OpenMP_ROOT}"
        PATH_SUFFIXES include
    )

    if (VSAG_LIBOMP_LIBRARY AND VSAG_LIBOMP_INCLUDE_DIR)
        # For LLVM clang on macOS, we need both compile and link flags
        set(OpenMP_C_FLAGS "-fopenmp")
        set(OpenMP_CXX_FLAGS "-fopenmp")
        set(OpenMP_C_FOUND TRUE)
        set(OpenMP_CXX_FOUND TRUE)

        # Add the libomp library path to linker search paths
        get_filename_component(VSAG_LIBOMP_LIB_DIR "${VSAG_LIBOMP_LIBRARY}" DIRECTORY)
        link_directories("${VSAG_LIBOMP_LIB_DIR}")

        message(STATUS "Found OpenMP (explicit libomp): ${VSAG_LIBOMP_LIBRARY}")

        if (NOT TARGET OpenMP::OpenMP_C)
            add_library(OpenMP::OpenMP_C INTERFACE IMPORTED)
        endif()
        if (NOT TARGET OpenMP::OpenMP_CXX)
            add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
        endif()

        # Set both compile and link options
        set_target_properties(OpenMP::OpenMP_C PROPERTIES
            INTERFACE_COMPILE_OPTIONS "${OpenMP_C_FLAGS}"
            INTERFACE_INCLUDE_DIRECTORIES "${VSAG_LIBOMP_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${VSAG_LIBOMP_LIBRARY}"
        )
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            INTERFACE_COMPILE_OPTIONS "${OpenMP_CXX_FLAGS}"
            INTERFACE_INCLUDE_DIRECTORIES "${VSAG_LIBOMP_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${VSAG_LIBOMP_LIBRARY}"
        )
    else()
        find_package(OpenMP QUIET) # Try default FindOpenMP if explicit fails
        if (OpenMP_CXX_FOUND)
            message(STATUS "Found OpenMP (FindOpenMP): ${OpenMP_CXX_INCLUDE_DIRS}")
            # LLVM clang can use -fopenmp directly
            if (" ${OpenMP_CXX_FLAGS} " MATCHES " -fopenmp ")
                message(STATUS "Using LLVM clang OpenMP flags: ${OpenMP_CXX_FLAGS}")
            endif()
        else()
            message(FATAL_ERROR "OpenMP not found on macOS. Set OpenMP_ROOT (e.g. /opt/homebrew/opt/libomp).")
        endif()
    endif()
    
    # Find LAPACK - will automatically use Accelerate framework on macOS
    find_package(LAPACK)
    if (LAPACK_FOUND)
        message(STATUS "Found LAPACK (using Accelerate framework on macOS)")
        # LAPACK libraries are in LAPACK_LIBRARIES variable
    else()
        message(WARNING "LAPACK not found")
    endif()

    # Find gfortran and its library path for OpenBLAS
    find_program(GFORTRAN_EXECUTABLE NAMES gfortran)
    if(GFORTRAN_EXECUTABLE)
        execute_process(
            COMMAND ${GFORTRAN_EXECUTABLE} -print-file-name=libgfortran.dylib
            OUTPUT_VARIABLE GFORTRAN_LIB
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(EXISTS "${GFORTRAN_LIB}" AND NOT IS_DIRECTORY "${GFORTRAN_LIB}")
            get_filename_component(GFORTRAN_LIB_DIR "${GFORTRAN_LIB}" DIRECTORY)
            list(APPEND CMAKE_INSTALL_RPATH "${GFORTRAN_LIB_DIR}")
        else()
            unset(GFORTRAN_LIB)
            message(WARNING "gfortran found but libgfortran.dylib not found via -print-file-name; OpenBLAS link may fail")
        endif()
    else()
        message(WARNING "gfortran not found; OpenBLAS/LAPACKE features may not link on macOS")
    endif()

    # Fixup: some scripts (e.g. OpenBLAS fallback) set BLAS_LIBRARIES/LAPACK_LIBRARIES to include
    # a bare `gfortran` token (expands to -lgfortran). On macOS libgfortran is often not in the
    # default linker search paths, causing: `ld: library 'gfortran' not found`.
    function(vsag_darwin_fixup_blas_lapack_libs)
        if (NOT APPLE)
            return()
        endif()
        if (NOT DEFINED GFORTRAN_LIB OR NOT EXISTS "${GFORTRAN_LIB}")
            return()
        endif()

        foreach(_var BLAS_LIBRARIES LAPACK_LIBRARIES)
            if(DEFINED ${_var})
                set(_new_list "")
                foreach(_item IN LISTS ${_var})
                    if(_item STREQUAL "gfortran")
                        list(APPEND _new_list "${GFORTRAN_LIB}")
                    else()
                        list(APPEND _new_list "${_item}")
                    endif()
                endforeach()
                # Keep cache in sync so downstream includes/targets see the rewritten list.
                set(${_var} "${_new_list}" CACHE STRING "" FORCE)
            endif()
        endforeach()
    endfunction()

    # Run after the whole top-level configure has defined BLAS/LAPACK variables (order independent).
    cmake_language(DEFER CALL vsag_darwin_fixup_blas_lapack_libs)
else()
    set(ld_flags_workaround "-Wl,-rpath=\\$\\$ORIGIN")
endif()