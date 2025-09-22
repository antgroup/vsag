
set (SEARCHER_SRC
    basic_searcher.cpp
    basic_searcher.h
    parallel_searcher.cpp
    parallel_searcher.h
)

add_library (searcher OBJECT ${SEARCHER_SRC})
target_link_libraries (searcher PRIVATE fmt::fmt coverage_config)
maybe_add_dependencies (searcher spdlog)
