
add_library (cluster OBJECT kmeans_cluster.cpp)
target_link_libraries (cluster PRIVATE fmt::fmt)
maybe_add_dependencies (cluster spdlog openblas mkl)

