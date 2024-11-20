
set (HNSWLIB_SRCS
        algorithm_interface.cpp
        block_manager.cpp
        hnswalg.cpp
)
set (HNSWLIB_HEADERS
        algorithm_interface.h
        block_manager.h
        hnswalg.h
        hnswlib.h
        hnswalg_static.h
        space_interface.h
        space_ip.h
        space_l2.h
        visited_list_pool.h
)
add_library (hnsw ${HNSWLIB_HEADERS} ${HNSWLIB_SRCS})
target_link_libraries (hnsw PRIVATE roaring)
