# Copyright 2026-present the vsag project
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include_guard (GLOBAL)
include (ExternalProject)
include (${CMAKE_CURRENT_LIST_DIR}/VSAGThirdPartyOverride.cmake)

function (vsag_configure_private_openmp)
    if (NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        message (FATAL_ERROR "Private OpenMP requires CMAKE_OSX_DEPLOYMENT_TARGET (wheel default: 14.0)")
    endif ()
    if (LIBOMP_PREFIX)
        set (_include "${LIBOMP_PREFIX}/include")
        set (_library "${LIBOMP_PREFIX}/lib/libomp.a")
        if (NOT EXISTS "${_include}/omp.h" OR NOT EXISTS "${_library}")
            message (FATAL_ERROR
                "LIBOMP_PREFIX must contain include/omp.h and lib/libomp.a. "
                "Unset LIBOMP_PREFIX to build the pinned OpenMP source automatically, or "
                "install libomp (brew install libomp) and pass -DLIBOMP_PREFIX=$(brew --prefix libomp).")
        endif ()
        # A newer Homebrew bottle cannot be made compatible by changing the wheel tag.
        execute_process (COMMAND otool -l "${_library}" OUTPUT_VARIABLE _load_commands
                         RESULT_VARIABLE _otool_result ERROR_QUIET)
        # Check each LC_BUILD_VERSION block, including mixed-platform archives.
        # Archive paths may contain CMake list delimiters; load-command fields do not.
        string (REPLACE ";" "_" _commands "${_load_commands}")
        string (REPLACE "[" "_" _commands "${_commands}")
        string (REPLACE "]" "_" _commands "${_commands}")
        string (REPLACE "Load command" ";Load command" _commands "${_commands}")
        foreach (_command IN LISTS _commands)
            if (_command MATCHES "cmd LC_BUILD_VERSION" AND
                NOT _command MATCHES "platform (1|MACOS)[\r\n ]")
                message (FATAL_ERROR "Custom libomp.a contains a non-macOS or unknown platform")
            endif ()
        endforeach ()
        string (REGEX MATCHALL "minos [0-9]+\\.[0-9]+(\\.[0-9]+)?|cmd LC_VERSION_MIN_MACOSX[\r\n ]+cmdsize [0-9]+[\r\n ]+version [0-9]+\\.[0-9]+(\\.[0-9]+)?" _versions "${_load_commands}")
        if (NOT _otool_result EQUAL 0 OR NOT _versions)
            message (FATAL_ERROR "Cannot inspect deployment target of ${_library} with otool")
        endif ()
        foreach (_entry IN LISTS _versions)
            string (REGEX REPLACE "^.*(minos|version) " "" _version "${_entry}")
            if (_version VERSION_GREATER CMAKE_OSX_DEPLOYMENT_TARGET)
                message (FATAL_ERROR
                    "Custom libomp.a requires macOS ${_version}, newer than target ${CMAKE_OSX_DEPLOYMENT_TARGET}. "
                    "Unset LIBOMP_PREFIX to build a compatible private runtime from source.")
            endif ()
        endforeach ()
    else ()
        if (CMAKE_VERSION VERSION_LESS 3.20)
            message (FATAL_ERROR "Building the wheel's LLVM OpenMP requires CMake 3.20 or later")
        endif ()
        find_package (Perl QUIET)
        if (NOT PERL_FOUND)
            message (FATAL_ERROR "Building private OpenMP requires Perl. Install perl (brew install perl) and ensure it is on PATH.")
        endif ()
        set (_root "${CMAKE_BINARY_DIR}/private-openmp")
        set (_include "${_root}/install/include")
        set (_library "${_root}/install/lib/libomp.a")
        set (_release "https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.8")
        set (_openmp_urls "${_release}/openmp-20.1.8.src.tar.xz")
        set (_cmake_urls "${_release}/cmake-20.1.8.src.tar.xz")
        vsag_resolve_thirdparty_override (openmp 20.1.8 _openmp_urls)
        vsag_resolve_thirdparty_override (llvm_cmake 20.1.8 _cmake_urls)
        ExternalProject_Add (vsag_openmp_cmake
            URL ${_cmake_urls}
            URL_HASH SHA256=3319203cfd1172bbac50f06fa68e318af84dcb5d65353310c0586354069d6634
            SOURCE_DIR "${_root}/src/cmake"
            CONFIGURE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND "")
        ExternalProject_Add (vsag_private_openmp_build
            DEPENDS vsag_openmp_cmake
            URL ${_openmp_urls}
            URL_HASH SHA256=b21c04ee9cbe56e200c5d83823765a443ee6389bbc3f64154c96e94016e6cee9
            SOURCE_DIR "${_root}/src/openmp"
            BINARY_DIR "${_root}/build"
            CMAKE_ARGS
                "-DCMAKE_INSTALL_PREFIX=${_root}/install"
                "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
                "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                "-DCMAKE_BUILD_TYPE=Release"
                "-DPERL_EXECUTABLE=${PERL_EXECUTABLE}"
                "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}"
                "-DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}"
                "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}"
                "-DLIBOMP_ENABLE_SHARED=OFF"
                "-DOPENMP_ENABLE_LIBOMPTARGET=OFF"
                "-DOPENMP_ENABLE_OMPT_TOOLS=OFF"
            BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel ${NUM_BUILDING_JOBS}
            BUILD_BYPRODUCTS "${_library}")
        file (MAKE_DIRECTORY "${_include}")
        add_library (vsag_private_openmp STATIC IMPORTED GLOBAL)
        set_target_properties (vsag_private_openmp PROPERTIES IMPORTED_LOCATION "${_library}")
        add_dependencies (vsag_private_openmp vsag_private_openmp_build)
    endif ()
    message (STATUS "Private OpenMP deployment target: macOS ${CMAKE_OSX_DEPLOYMENT_TARGET}")
    target_compile_options (vsag_openmp INTERFACE -Xclang -fopenmp)
    target_include_directories (vsag_openmp SYSTEM INTERFACE "${_include}")
    if (TARGET vsag_private_openmp)
        target_link_libraries (vsag_openmp INTERFACE vsag_private_openmp)
    else ()
        target_link_libraries (vsag_openmp INTERFACE "${_library}")
    endif ()
endfunction ()
