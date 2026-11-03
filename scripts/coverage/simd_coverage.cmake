# CI-only augmentation, applied to both the measured base and head revisions.
# Catch2 tags cross module boundaries: Local SQ owns SIMD reference tests too.
if (CMAKE_VERSION VERSION_LESS "3.19")
    message (FATAL_ERROR "SIMD coverage target completion requires CMake 3.19 or newer")
endif ()

set (VSAG_SIMD_COVERAGE_HELPER_DIR "${CMAKE_CURRENT_LIST_DIR}")

function (vsag_complete_simd_coverage_target)
    # Preserve the aggregate binary's library bootstrap and ISA diagnostic output.
    target_sources (unittests_simd PRIVATE "${VSAG_SIMD_COVERAGE_HELPER_DIR}/simd_coverage_init.cpp")
    set (local_sq_tests
         "${CMAKE_SOURCE_DIR}/src/quantization/scalar_quantization/local_scalar_quantizer_test.cpp")
    if (EXISTS "${local_sq_tests}")
        # Match quantizer_test: instrument production code, not the test source.
        # Attaching this source directly would inherit coverage_config and change
        # the measured inline-header line set.
        add_library (vsag_simd_coverage_reference OBJECT "${local_sq_tests}")
        target_link_libraries (vsag_simd_coverage_reference PRIVATE unittest_deps)
        target_sources (unittests_simd PRIVATE $<TARGET_OBJECTS:vsag_simd_coverage_reference>)
    endif ()
endfunction ()

# The project's module targets are defined after project(VSAG).
cmake_language (DEFER CALL vsag_complete_simd_coverage_target)
