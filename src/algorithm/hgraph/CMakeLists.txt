set(HGRAPH_SRCS
    hgraph.cpp
    hgraph_build.cpp
    hgraph_modify.cpp
    hgraph_parameter.cpp
    hgraph_param_mapping.cpp
    hgraph_search.cpp
    hgraph_serialize.cpp
    hgraph_cache.cpp
)

add_library(hgraph OBJECT ${HGRAPH_SRCS})
target_link_libraries(hgraph PRIVATE coverage_config vsag_src_common)
