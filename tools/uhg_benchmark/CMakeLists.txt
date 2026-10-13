cmake_minimum_required(VERSION 3.18)
project(uhg_benchmark LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
get_filename_component(VSAG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(VSAG_BUILD "${VSAG_ROOT}/build-release" CACHE PATH "Optimized VSAG build")
set(VSAG_DEPS "/tmp/vsag-deps" CACHE PATH "VSAG dependency source directory")
add_executable(uhg_benchmark main.cpp)
target_include_directories(uhg_benchmark PRIVATE
    ${VSAG_ROOT}/include ${VSAG_ROOT}/src
    ${VSAG_DEPS}/nlohmann_json-src/include ${VSAG_DEPS}/fmt-src/include
    ${VSAG_DEPS}/thread_pool-src ${VSAG_DEPS}/tsl-src/include
    ${VSAG_DEPS}/cpuinfo-src/include
    ${HDF5_INCLUDE_DIRS})
target_compile_options(uhg_benchmark PRIVATE -O3 -march=native)
target_link_libraries(uhg_benchmark PRIVATE
    ${VSAG_BUILD}/src/libvsag.so ${HDF5_LIBRARIES} ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)
set_target_properties(uhg_benchmark PROPERTIES BUILD_RPATH "${VSAG_BUILD}/src")
