
add_subdirectory (allocator)
add_subdirectory (filter)
add_subdirectory (heap)
add_subdirectory (transform)
add_subdirectory (bitset)

file (GLOB IMPL_SRCS "*.cpp")
list (FILTER IMPL_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER IMPL_SRCS EXCLUDE REGEX kmeans_cluster.cpp)

add_library (kmeans OBJECT kmeans_cluster.cpp)
target_link_libraries (kmeans PRIVATE fmt::fmt)
maybe_add_dependencies (kmeans spdlog openblas mkl)

add_library (impl OBJECT ${IMPL_SRCS})
target_link_libraries (impl PUBLIC transform allocator heap bitset filter kmeans fmt::fmt coverage_config)
maybe_add_dependencies (impl spdlog)

set (IMPL_LIBS transform allocator heap bitset filter impl kmeans PARENT_SCOPE)
