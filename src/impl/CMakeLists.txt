
add_subdirectory (thread_pool)
add_subdirectory (allocator)
add_subdirectory (filter)
add_subdirectory (heap)
add_subdirectory (transform)
add_subdirectory (bitset)
add_subdirectory (cluster)
add_subdirectory (logger)
add_subdirectory (searcher)
add_subdirectory (odescent)

file (GLOB IMPL_SRCS "*.cpp")
list (FILTER IMPL_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER IMPL_SRCS EXCLUDE REGEX kmeans_cluster.cpp)

add_library (impl OBJECT ${IMPL_SRCS})
target_link_libraries (impl PUBLIC transform thread_pool allocator heap 
    bitset logger filter cluster searcher odescent fmt::fmt coverage_config)
maybe_add_dependencies (impl spdlog)

set (IMPL_LIBS transform thread_pool allocator heap bitset 
    logger filter impl cluster searcher odescent PARENT_SCOPE)
