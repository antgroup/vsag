
set (DISKANN_INDEX_SRCS
        diskann.cpp
        diskann_zparameters.cpp
)
set (DISKANN_INDEX_HEADERS
        diskann.h
        diskann_zparameters.h
)
add_library (diskann_index OBJECT ${DISKANN_INDEX_HEADERS} ${DISKANN_INDEX_SRCS})
target_link_libraries (diskann_index diskann)

set (HNSW_INDEX_SRCS
        hnsw.cpp
        hnsw_zparameters.cpp
)
set (HNSW_INDEX_HEADERS
        hnsw.h
        hnsw_zparameters.h
)
add_library (hnsw_index OBJECT ${HNSW_INDEX_HEADERS} ${HNSW_INDEX_SRCS})
target_link_libraries (hnsw_index PRIVATE hnsw roaring)

add_library (index index_common_param.cpp index_common_param.h)
target_link_libraries (index PRIVATE hnsw_index diskann_index)
