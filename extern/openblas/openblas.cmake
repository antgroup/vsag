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

set (name openblas)
set (source_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/source)
set (install_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/install)

# Legacy switch kept for backward compatibility: USE_SYSTEM_OPENBLAS=ON keeps
# its previous "try system, fall back to bundled" semantics, i.e. it maps to
# AUTO (not ON) when the new override is empty.
option (USE_SYSTEM_OPENBLAS "(deprecated) Try system OpenBLAS, fall back to bundled" OFF)

vsag_get_system_dep_policy (OPENBLAS _openblas_policy)
if (USE_SYSTEM_OPENBLAS AND "${VSAG_USE_SYSTEM_OPENBLAS}" STREQUAL ""
        AND _openblas_policy STREQUAL "OFF")
    set (_openblas_policy "AUTO")
endif ()

set (_openblas_system_paths
    /usr/lib
    /usr/lib64
    /usr/lib/x86_64-linux-gnu
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib
    /usr/local/lib64
    /opt/homebrew/lib)
set (_openblas_system_includes
    /usr/include
    /usr/include/openblas
    /usr/include/x86_64-linux-gnu
    /usr/include/aarch64-linux-gnu
    /usr/local/include
    /usr/local/include/openblas
    /opt/homebrew/include)

set (OPENBLAS_FOUND FALSE)
set (_openblas_uses_imported_target FALSE)
set (OPENBLAS_INCLUDE_DIRS "")

if (NOT _openblas_policy STREQUAL "OFF")
    # 1) Reuse an existing OpenBLAS::OpenBLAS target if a parent project already
    #    provided one.
    if (TARGET OpenBLAS::OpenBLAS)
        set (OPENBLAS_FOUND TRUE)
        set (_openblas_uses_imported_target TRUE)
        set (BLAS_LIBRARIES OpenBLAS::OpenBLAS)
        message (STATUS "Using pre-existing OpenBLAS::OpenBLAS target")
    endif ()

    # 2) Try find_package(OpenBLAS CONFIG ...) which is what modern OpenBLAS
    #    installs ship.
    if (NOT OPENBLAS_FOUND)
        find_package (OpenBLAS CONFIG QUIET)
        if (TARGET OpenBLAS::OpenBLAS)
            set (OPENBLAS_FOUND TRUE)
            set (_openblas_uses_imported_target TRUE)
            set (BLAS_LIBRARIES OpenBLAS::OpenBLAS)
            message (STATUS "Found OpenBLAS via find_package(OpenBLAS CONFIG)")
        endif ()
    endif ()

    # 3) Fall back to a manual search for libopenblas + cblas.h + lapacke.h.
    #    Search VSAG's known list first, then fall through to CMake's default
    #    paths (including CMAKE_PREFIX_PATH, multiarch dirs, etc.) so users on
    #    less common architectures can point us at a system install.
    if (NOT OPENBLAS_FOUND)
        find_library (OPENBLAS_LIB
            NAMES openblas
            PATHS ${_openblas_system_paths})
        find_path (OPENBLAS_INCLUDE
            NAMES cblas.h
            PATHS ${_openblas_system_includes})
        find_path (LAPACKE_INCLUDE
            NAMES lapacke.h
            PATHS ${_openblas_system_includes})

        if (OPENBLAS_LIB AND OPENBLAS_INCLUDE AND LAPACKE_INCLUDE)
            set (OPENBLAS_FOUND TRUE)
            message (STATUS "Found system OpenBLAS library: ${OPENBLAS_LIB}")
            message (STATUS "Found OpenBLAS include directory: ${OPENBLAS_INCLUDE}")
            message (STATUS "Found LAPACKE include directory: ${LAPACKE_INCLUDE}")

            find_library (LAPACKE_LIB
                NAMES lapacke
                PATHS ${_openblas_system_paths})
            if (LAPACKE_LIB)
                message (STATUS "Found LAPACKE library: ${LAPACKE_LIB}")
            else ()
                message (STATUS
                         "LAPACKE library not found as separate library; "
                         "assuming it is bundled inside OpenBLAS")
            endif ()

            set (OPENBLAS_INCLUDE_DIRS ${OPENBLAS_INCLUDE})
            if (NOT "${OPENBLAS_INCLUDE}" STREQUAL "${LAPACKE_INCLUDE}")
                list (APPEND OPENBLAS_INCLUDE_DIRS ${LAPACKE_INCLUDE})
            endif ()

            set (BLAS_LIBRARIES ${OPENBLAS_LIB})
            if (LAPACKE_LIB)
                list (APPEND BLAS_LIBRARIES ${LAPACKE_LIB})
            endif ()
        endif ()
    endif ()

    # Discovery is done; honour the policy.
    if (NOT OPENBLAS_FOUND AND _openblas_policy STREQUAL "ON")
        message (STATUS "  OPENBLAS_LIB: ${OPENBLAS_LIB}")
        message (STATUS "  OPENBLAS_INCLUDE: ${OPENBLAS_INCLUDE}")
        message (STATUS "  LAPACKE_INCLUDE: ${LAPACKE_INCLUDE}")
        vsag_fail_missing_system_dep (OPENBLAS OpenBLAS OpenBLAS::OpenBLAS)
    endif ()
endif ()

if (OPENBLAS_FOUND)
    # Append gfortran + OpenMP runtime when we are taking the system path that
    # only resolves to a raw library (the imported CONFIG target already pulls
    # transitive deps in).
    if (NOT _openblas_uses_imported_target)
        if (APPLE AND DEFINED GFORTRAN_LIB AND EXISTS "${GFORTRAN_LIB}")
            list (APPEND BLAS_LIBRARIES "${GFORTRAN_LIB}")
        else ()
            list (APPEND BLAS_LIBRARIES gfortran)
        endif ()
        if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            list (PREPEND BLAS_LIBRARIES omp)
        else ()
            list (PREPEND BLAS_LIBRARIES gomp)
        endif ()
    endif ()

    if (NOT TARGET ${name})
        add_custom_target (${name})
    endif ()
    message (STATUS "Using system OpenBLAS as BLAS backend")
else ()
    set (BLAS_LIBRARIES ${install_dir}/lib/libopenblas.a)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list (PREPEND BLAS_LIBRARIES omp)
    else ()
        list (PREPEND BLAS_LIBRARIES gomp)
    endif ()
    set (OPENBLAS_INCLUDE_DIRS ${install_dir}/include)
    message (STATUS "Enable bundled OpenBLAS as BLAS backend")
endif ()

add_library (vsag_openblas_headers INTERFACE)
if (OPENBLAS_INCLUDE_DIRS)
    target_include_directories (vsag_openblas_headers INTERFACE ${OPENBLAS_INCLUDE_DIRS})
endif ()
if (_openblas_uses_imported_target)
    target_link_libraries (vsag_openblas_headers INTERFACE OpenBLAS::OpenBLAS)
endif ()

# Publish BLAS_LIBRARIES in the cache so downstream / superbuild consumers see
# the same variable shape on both BLAS backends (the MKL path does the same).
set (BLAS_LIBRARIES "${BLAS_LIBRARIES}" CACHE STRING
     "Final list of BLAS libraries to link against." FORCE)

if (NOT OPENBLAS_FOUND)
    # Build OpenBLAS from source
    message (STATUS "Building OpenBLAS from source")

    set (openblas_urls
        https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.23/OpenBLAS-0.3.23.tar.gz
        # this url is maintained by the vsag project, if it's broken, please try
        #  the latest commit or contact the vsag project
        https://vsagcache.oss-rg-china-mainland.aliyuncs.com/openblas/OpenBLAS-0.3.23.tar.gz
    )
    if (DEFINED ENV{VSAG_THIRDPARTY_OPENBLAS})
        message (STATUS "Using local path for openblas: $ENV{VSAG_THIRDPARTY_OPENBLAS}")
        list (PREPEND openblas_urls "$ENV{VSAG_THIRDPARTY_OPENBLAS}")
    endif ()

    ExternalProject_Add (
        ${name}
        URL ${openblas_urls}
        URL_HASH MD5=115634b39007de71eb7e75cf7591dfb2
        DOWNLOAD_NAME OpenBLAS-v0.3.23.tar.gz
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${name}
        TMP_DIR ${BUILD_INFO_DIR}
        STAMP_DIR ${BUILD_INFO_DIR}
        DOWNLOAD_DIR ${DOWNLOAD_DIR}
        SOURCE_DIR ${source_dir}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND
            ${common_configure_envs}
            OMP_NUM_THREADS=1
            PATH=/usr/lib/ccache:$ENV{PATH}
            LD_LIBRARY_PATH=/opt/alibaba-cloud-compiler/lib64/:$ENV{LD_LIBRARY_PATH}
            make USE_THREAD=0 USE_LOCKING=1 DYNAMIC_ARCH=1 NOFORTRAN=1 -j${NUM_BUILDING_JOBS}
        INSTALL_COMMAND
            make DYNAMIC_ARCH=1 NOFORTRAN=1 PREFIX=${install_dir} install
        BUILD_IN_SOURCE 1
        LOG_CONFIGURE TRUE
        LOG_BUILD TRUE
        LOG_INSTALL TRUE
        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30

        BUILD_BYPRODUCTS
            ${install_dir}/lib/libopenblas.a
    )

    if (NOT APPLE)
        set (OPENBLAS_LIBRARY_DIRS ${install_dir}/lib ${install_dir}/lib64)
    else ()
        set (OPENBLAS_LIBRARY_DIRS ${install_dir}/lib)
    endif ()

    file (GLOB LIB_DIR_EXIST CHECK_DIRECTORIES LIST_DIRECTORIES true ${install_dir}/lib)
    if (LIB_DIR_EXIST)
        file (GLOB LIB_FILES ${install_dir}/lib/lib*.a)
        foreach (lib_file ${LIB_FILES})
            install (FILES ${lib_file}
                     DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
        endforeach ()
    endif ()

    if (NOT APPLE)
        file (GLOB LIB64_DIR_EXIST CHECK_DIRECTORIES LIST_DIRECTORIES true ${install_dir}/lib64)
        if (LIB64_DIR_EXIST)
            file (GLOB LIB64_FILES ${install_dir}/lib64/lib*.a)
            foreach (lib64_file ${LIB64_FILES})
                install (FILES ${lib64_file}
                         DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
            endforeach ()
        endif ()
    endif ()
else ()
    set (OPENBLAS_LIBRARY_DIRS "")
endif ()
