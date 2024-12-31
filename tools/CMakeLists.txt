
add_library (eval_dataset OBJECT eval_dataset.cpp eval_dataset.h)
add_dependencies (eval_dataset hdf5 spdlog)

add_executable (test_performance test_performance.cpp)
target_link_libraries (test_performance PRIVATE vsag eval_dataset simd libhdf5_cpp.a libhdf5.a z)
