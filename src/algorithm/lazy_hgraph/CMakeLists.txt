set(LAZY_HGRAPH_SRCS
    lazy_hgraph.cpp
    lazy_hgraph_parameter.cpp
)

add_library(lazy_hgraph OBJECT ${LAZY_HGRAPH_SRCS})
target_link_libraries(lazy_hgraph PRIVATE coverage_config vsag_src_common)
