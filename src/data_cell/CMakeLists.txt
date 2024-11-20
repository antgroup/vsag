set (DATACELL_SRCS
        flatten_interface.cpp
        graph_interface.cpp
        sparse_graph_datacell.cpp
)

set (DATACELL_HEADERS
        flatten_datacell.h
        flatten_interface.h
        graph_datacell.h
        graph_interface.h
        sparse_graph_datacell.h
)

add_library (datacell ${DATACELL_SRCS} ${DATACELL_HEADERS})
target_link_libraries (datacell roaring)
