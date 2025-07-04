
set (FILTER_SRC
        black_list_filter.h
        black_list_filter.cpp
        extrainfo_wrapper_filter.h
        extrainfo_wrapper_filter.cpp
        inner_id_wrapper_filter.h
        white_list_filter.h
        white_list_filter.cpp
)
add_library (filter OBJECT ${FILTER_SRC})
target_link_libraries (filter PRIVATE fmt::fmt)
maybe_add_dependencies (filter spdlog)
