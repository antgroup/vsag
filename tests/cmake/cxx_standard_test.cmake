if (NOT DEFINED VSAG_SOURCE_DIR)
    message (FATAL_ERROR "VSAG_SOURCE_DIR is required")
endif ()
string (RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set (cuda_args)
if (DEFINED VSAG_CUDA_ARCHITECTURES)
    list (APPEND cuda_args "-DVSAG_CUDA_ARCHITECTURES:STRING=${VSAG_CUDA_ARCHITECTURES}")
endif ()
foreach (build_type IN ITEMS Debug Release Sanitize)
    set (tsan OFF)
    if (build_type STREQUAL "Sanitize")
        set (tsan ON)
    endif ()
    set (build "/tmp/vsag-cxx-standard-${suffix}-${build_type}")
    execute_process (
        COMMAND ${CMAKE_COMMAND} -S ${VSAG_SOURCE_DIR}/tests/cmake/cxx_standard_fixture
                -B ${build} -G Ninja -DVSAG_SOURCE_DIR=${VSAG_SOURCE_DIR}
                -DCMAKE_BUILD_TYPE=${build_type}
                -DENABLE_WERROR=ON -DENABLE_TSAN:BOOL=${tsan}
                -DVSAG_TEST_CUDA:BOOL=${VSAG_TEST_CUDA} ${cuda_args}
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors)
    if (tsan)
        set (targets internal_probe tsan_fence_probe)
    else ()
        set (targets internal_probe release_math_probe)
    endif ()
    if (result EQUAL 0)
        execute_process (
            COMMAND ${CMAKE_COMMAND} --build ${build} --target ${targets}
            RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors)
    endif ()
    if (result EQUAL 0 AND NOT tsan)
        execute_process (COMMAND ${build}/release_math_probe
            RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors)
    endif ()
    file (REMOVE_RECURSE ${build})
    if (NOT result EQUAL 0)
        message (FATAL_ERROR "C++ standard/math fixture (${build_type}) failed:\n${output}\n${errors}")
    endif ()
endforeach ()
message (STATUS "C++20, GPU stub, nonfinite math and TSAN fence compilation checks passed")
