cmake_minimum_required(VERSION 3.18)
project(vsag_full_baseline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(vsag CONFIG REQUIRED)

add_executable(full_benchmark ../full_main.cpp)
target_link_libraries(full_benchmark PRIVATE vsag::vsag)
target_compile_options(full_benchmark PRIVATE -Wall -Wextra -Wpedantic)
