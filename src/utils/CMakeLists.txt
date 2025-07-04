
set (UTILS_SRC
        util_functions.cpp
        linear_congruential_generator.cpp
        visited_list.cpp
        slow_task_timer.cpp
        timer.cpp
        window_result_queue.cpp
)

add_library (utils OBJECT ${UTILS_SRC})
add_dependencies (utils spdlog fmt::fmt)
