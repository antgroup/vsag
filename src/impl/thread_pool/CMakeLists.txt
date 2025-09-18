
set (THREAD_POOL_SRC
    default_thread_pool.cpp
    default_thread_pool.h
    safe_thread_pool.h
)

add_library (thread_pool OBJECT ${THREAD_POOL_SRC})
target_link_libraries (thread_pool PRIVATE fmt::fmt coverage_config)
maybe_add_dependencies (thread_pool spdlog)
