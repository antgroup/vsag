
set (HNSWLIB_SRCS
        block_manager.cpp
        algorithm_interface.cpp
        hnswalg.cpp
)

add_library (hnswlib OBJECT ${HNSWLIB_SRCS})
target_link_libraries (hnswlib PUBLIC coverage_config)
add_dependencies (hnswlib spdlog fmt::fmt)
