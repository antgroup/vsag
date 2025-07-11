
set (HEAP_SRC
        distance_heap.cpp
        memmove_heap.cpp
        standard_heap.cpp
)

add_library (heap OBJECT ${HEAP_SRC})
target_link_libraries (heap PUBLIC coverage_config)

maybe_add_dependencies (heap spdlog fmt::fmt)
