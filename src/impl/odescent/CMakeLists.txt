
set (ODESCENT_SRC
    odescent_graph_builder.cpp
    odescent_graph_builder.h
    odescent_graph_parameter.cpp
    odescent_graph_parameter.h
)

add_library (odescent OBJECT ${ODESCENT_SRC})
target_link_libraries (odescent PRIVATE fmt::fmt coverage_config)
maybe_add_dependencies (odescent spdlog)
