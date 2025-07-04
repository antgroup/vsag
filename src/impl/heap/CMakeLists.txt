
set (HEAP_SRC
        distance_heap.cpp
        memmove_heap.cpp
        standard_heap.cpp
)

add_library (heap OBJECT ${HEAP_SRC})
maybe_add_dependencies (heap spdlog fmt::fmt)
