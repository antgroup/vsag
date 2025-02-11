
set (EVAL_SRC
        eval_case.cpp
        search_eval_case.cpp
        build_eval_case.cpp
        eval_dataset.cpp
        eval_config.cpp
        monitor/monitor.cpp
        monitor/latency_monitor.cpp
        monitor/recall_monitor.cpp
        monitor/memory_peak_monitor.cpp
        monitor/duration_monitor.cpp)
add_library (eval_obj OBJECT ${EVAL_SRC})
add_dependencies (eval_obj hdf5 spdlog yaml-cpp)

add_executable (eval_performance main.cpp)
target_link_libraries (eval_performance
  PRIVATE
  vsag
  eval_obj
  yaml-cpp
  argparse::argparse
  simd
  libhdf5_cpp.a
  libhdf5.a
  z
)
