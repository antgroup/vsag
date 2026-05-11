set(HGRAPH_SRCS
        hgraph.cpp
        hgraph_parameter.cpp
        hgraph_serializer.cpp
)

add_library(hgraph OBJECT ${HGRAPH_SRCS})
target_link_libraries(hgraph PRIVATE coverage_config vsag_src_common)
