if(NOT DEFINED VSAG_SOURCE_DIR)
    message(FATAL_ERROR "VSAG_SOURCE_DIR is required")
endif()
unset(ENV{VSAG_THIRDPARTY_OPENBLAS_0_3_34})
unset(ENV{VSAG_THIRDPARTY_OPENBLAS})
set(build_dir ${CMAKE_CURRENT_BINARY_DIR}/build/openblas-config-fixture)
execute_process(COMMAND ${CMAKE_COMMAND}
    -S ${VSAG_SOURCE_DIR}/tests/cmake/openblas_fixture -B ${build_dir}
    -DVSAG_SOURCE_DIR=${VSAG_SOURCE_DIR}
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "OpenBLAS fixture failed: ${output}\n${error}")
endif()
file(READ ${build_dir}/commands.txt commands)
if(commands MATCHES "-Ofast|-ffast-math")
    message(FATAL_ERROR "OpenBLAS CPU detection received fast-math flags: ${commands}")
endif()
string(REGEX MATCHALL "NOFORTRAN=1" nofortran_settings "${commands}")
list(LENGTH nofortran_settings nofortran_count)
if(NOT nofortran_count EQUAL 2)
    message(FATAL_ERROR "Both build and install must select C LAPACK")
endif()
foreach(required "USE_THREAD=0" "USE_LOCKING=1" "DYNAMIC_ARCH=1" "NOFORTRAN=1"
    "-j8" "MAKE_NB_JOBS=8" "-O2"
    "SHA256=cd7e129868320cc2d033afa920e31202dfe0b8066a5b66661900ccc0f197dfed"
    "OpenBLAS-v0.3.34.tar.gz")
    if(NOT commands MATCHES "${required}")
        message(FATAL_ERROR "Missing OpenBLAS setting ${required}: ${commands}")
    endif()
endforeach()
execute_process(COMMAND ${CMAKE_COMMAND}
    -S ${VSAG_SOURCE_DIR}/tests/cmake/openblas_fixture -B ${build_dir}-gnu15
    -DVSAG_SOURCE_DIR=${VSAG_SOURCE_DIR} -DTEST_GNU15_FLAGS=ON
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "GNU C17 configuration fixture failed: ${output}\n${error}")
endif()
file(READ ${build_dir}-gnu15/commands.txt commands)
if(NOT commands MATCHES "-std=gnu17")
    message(FATAL_ERROR "GCC 15 configuration needs GNU C17")
endif()
message(STATUS "OpenBLAS generated-command checks passed")
