
# deprecated
add_executable (test_performance main.cpp)
target_link_libraries (test_performance PRIVATE vsag eval_obj yaml-cpp simd libhdf5_cpp.a libhdf5.a z)
