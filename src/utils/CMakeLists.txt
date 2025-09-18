
set (UTILS_SRC
        util_functions.cpp
        linear_congruential_generator.cpp
        visited_list.cpp
        slow_task_timer.cpp
        timer.cpp
        window_result_queue.cpp
        sparse_vector_transform.cpp
        prefetch.cpp
        lock_strategy.cpp
)

add_library (utils OBJECT ${UTILS_SRC})
target_link_libraries (utils PUBLIC coverage_config)
add_dependencies (utils spdlog fmt::fmt)
