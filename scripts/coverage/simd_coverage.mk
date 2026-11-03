# Append to the measured revision's Makefile, including older PR bases.
# Keep the same configuration as make cov; only restrict the build target.
SIMD_COVERAGE_CMAKE := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))simd_coverage.cmake)
.PHONY: cov-simd
cov-simd:
	cmake ${VSAG_CMAKE_ARGS} -B"${DEBUG_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON -DENABLE_CCACHE=ON -DENABLE_ASAN=OFF -DENABLE_TESTS=ON -DCMAKE_PROJECT_VSAG_INCLUDE="${SIMD_COVERAGE_CMAKE}"
	cmake --build "${DEBUG_BUILD_DIR}" --target unittests_simd --parallel ${COMPILE_JOBS}
