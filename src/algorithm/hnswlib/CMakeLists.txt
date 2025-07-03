
set (HNSWLIB_SRCS
        block_manager.cpp
        algorithm_interface.cpp
        hnswalg.cpp
)

add_library (hnswlib OBJECT ${HNSWLIB_SRCS})
add_dependencies (hnswlib spdlog fmt::fmt)
