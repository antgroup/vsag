
set (EVAL_SRC
        eval_case.cpp
        search_eval_case.cpp
        build_eval_case.cpp
        monitor.cpp
        eval_dataset.cpp
        latency_monitor.cpp
        recall_monitor.cpp
        memory_peak_monitor.cpp
        duration_monitor.cpp)
add_library (eval_obj OBJECT ${EVAL_SRC})
add_dependencies (eval_obj hdf5)
