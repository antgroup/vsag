
add_subdirectory (eval)

add_executable (test_performance test_performance.cpp)
target_link_libraries (test_performance PRIVATE vsag eval_obj simd libhdf5_cpp.a libhdf5.a z)

add_executable (eval_performance eval_performance.cpp)
target_link_libraries (eval_performance PRIVATE vsag eval_obj argparse::argparse simd libhdf5_cpp.a libhdf5.a z)
