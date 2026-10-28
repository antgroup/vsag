cmake_minimum_required (VERSION 3.18)
project (cxx_standard_fixture LANGUAGES C CXX)
include (${VSAG_SOURCE_DIR}/cmake/VSAGHelpers.cmake)
set (ENABLE_CXX11_ABI ON)
include (${VSAG_SOURCE_DIR}/cmake/VSAGCompilerConfig.cmake)
file (WRITE "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp" "static_assert(__cplusplus > 201703L);\n")
add_library (internal_probe OBJECT "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp")
file (WRITE "${CMAKE_CURRENT_BINARY_DIR}/math_probe.cpp" [=[
#include <cmath>
#include <limits>
int main() {
    volatile double infinity = std::numeric_limits<double>::infinity();
    volatile double nan = std::numeric_limits<double>::quiet_NaN();
    return std::isinf(infinity) && !std::isfinite(infinity) &&
           std::isnan(nan) && !std::isfinite(nan) ? 0 : 1;
}
]=])
add_executable (release_math_probe "${CMAKE_CURRENT_BINARY_DIR}/math_probe.cpp")
if (ENABLE_TSAN)
    file (WRITE "${CMAKE_CURRENT_BINARY_DIR}/fence_probe.cpp" [=[
#include <atomic>
void publish_fence() {
    std::atomic_thread_fence(std::memory_order_release);
}
]=])
    add_library (tsan_fence_probe OBJECT "${CMAKE_CURRENT_BINARY_DIR}/fence_probe.cpp")
endif ()
get_target_property (internal_standard internal_probe CXX_STANDARD)
if (NOT internal_standard EQUAL 20 OR NOT CMAKE_CXX_STANDARD_REQUIRED)
    message (FATAL_ERROR "Internal targets must require C++20")
endif ()
option (VSAG_TEST_CUDA "Verify actual CUDA target with an installed toolkit" OFF)
set (ENABLE_CUDA ${VSAG_TEST_CUDA})
if (ENABLE_CUDA)
    enable_language (CUDA)
    find_package (CUDAToolkit 11.8 REQUIRED)
    # Compile-only default; override for a toolkit or GPU with a different target.
    set (VSAG_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures for the standard fixture")
endif ()
set (ENABLE_TESTS OFF)
add_subdirectory (${VSAG_SOURCE_DIR}/src/gpu gpu)
if (ENABLE_CUDA)
    get_target_property (cuda_standard vsag_gpu CUDA_STANDARD)
    get_target_property (cuda_required vsag_gpu CUDA_STANDARD_REQUIRED)
    if (NOT cuda_standard EQUAL 17 OR NOT cuda_required)
        message (FATAL_ERROR "CUDA must require C++17 independently of CPU C++20")
    endif ()
else ()
    get_target_property (stub_standard vsag_gpu CXX_STANDARD)
    if (NOT stub_standard EQUAL 20)
        message (FATAL_ERROR "GPU CPU stub must remain C++20")
    endif ()
endif ()
