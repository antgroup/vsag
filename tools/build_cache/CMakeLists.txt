add_library (build_cache_tool_common OBJECT
    build_cache_tool_common.cpp)
target_link_libraries (build_cache_tool_common PRIVATE
    vsag)

add_executable (export_hgraph_build_cache export_hgraph_build_cache.cpp)
target_sources (export_hgraph_build_cache PRIVATE
    $<TARGET_OBJECTS:build_cache_tool_common>)
target_link_libraries (export_hgraph_build_cache PRIVATE
    argparse::argparse
    vsag)

add_executable (build_hgraph_with_cache build_hgraph_with_cache.cpp)
target_sources (build_hgraph_with_cache PRIVATE
    $<TARGET_OBJECTS:build_cache_tool_common>)
target_link_libraries (build_hgraph_with_cache PRIVATE
    argparse::argparse
    vsag)

add_executable (compare_hgraph_ids compare_hgraph_ids.cpp)
target_link_libraries (compare_hgraph_ids PRIVATE
    argparse::argparse
    vsag)
