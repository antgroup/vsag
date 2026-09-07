if (NOT DEFINED VSAG_SOURCE_DIR)
    message (FATAL_ERROR "VSAG_SOURCE_DIR is required")
endif ()

set (fixture "${VSAG_SOURCE_DIR}/tests/cmake/openblas_target_fixture.cmake")

function (assert_match_count content pattern expected description)
    string (REGEX MATCHALL "${pattern}" matches "${content}")
    list (LENGTH matches actual)
    if (NOT actual EQUAL expected)
        message (FATAL_ERROR
                 "${description}: expected ${expected} matches, got ${actual}")
    endif ()
endfunction ()

function (run_fixture target expected_result expected_pattern)
    unset (ENV{VSAG_OPENBLAS_TARGET})
    if (NOT "${target}" STREQUAL "<UNSET>")
        set (ENV{VSAG_OPENBLAS_TARGET} "${target}")
    endif ()

    execute_process (
        COMMAND "${CMAKE_COMMAND}" "-DVSAG_SOURCE_DIR:PATH=${VSAG_SOURCE_DIR}" -P "${fixture}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    set (output "${stdout}\n${stderr}")
    if (expected_result STREQUAL "PASS" AND NOT result EQUAL 0)
        message (FATAL_ERROR "OpenBLAS target fixture failed (${result}):\n${output}")
    elseif (expected_result STREQUAL "FAIL" AND result EQUAL 0)
        message (FATAL_ERROR "OpenBLAS target fixture unexpectedly passed:\n${output}")
    endif ()
    if (NOT output MATCHES "${expected_pattern}")
        message (FATAL_ERROR
                 "OpenBLAS target fixture output did not match '${expected_pattern}':\n${output}")
    endif ()
endfunction ()

run_fixture ("<UNSET>" PASS "OPENBLAS_TARGET_ARG=")
run_fixture ("  GENERIC  " PASS "OPENBLAS_TARGET_ARG=TARGET=GENERIC")
run_fixture ("RISCV64_GENERIC" PASS "OPENBLAS_TARGET_ARG=TARGET=RISCV64_GENERIC")
run_fixture ("GENERIC;NOFORTRAN=0" FAIL "must contain only letters, digits, or underscores")
run_fixture ("GENERIC NOFORTRAN=0" FAIL "must contain only letters, digits, or underscores")

unset (ENV{VSAG_OPENBLAS_TARGET})

file (READ "${VSAG_SOURCE_DIR}/extern/openblas/openblas.cmake" openblas_cmake)
assert_match_count ("${openblas_cmake}" "\\$\\{_openblas_target_arg\\}" 2
                    "OpenBLAS build and install target propagation")
assert_match_count ("${openblas_cmake}" "DYNAMIC_ARCH=1" 2
                    "OpenBLAS dynamic architecture preservation")

file (READ "${VSAG_SOURCE_DIR}/.github/workflows/python_build_and_test.yml" python_workflow)
assert_match_count ("${python_workflow}" "VSAG_OPENBLAS_TARGET=GENERIC" 2
                    "Python x86 wheel target propagation")

file (READ "${VSAG_SOURCE_DIR}/.github/workflows/build_release_wheel.yml" release_workflow)
assert_match_count ("${release_workflow}"
                    "matrix.arch == 'x86_64' && 'VSAG_OPENBLAS_TARGET=GENERIC'" 3
                    "x86-only release wheel target propagation")

message (STATUS "OpenBLAS target tests passed")
