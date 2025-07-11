
set (ALLOCATOR_SRC
        default_allocator.cpp
        default_allocator.h
        safe_allocator.h
        allocator_wrapper.h
)

add_library (allocator OBJECT ${ALLOCATOR_SRC})
target_link_libraries (allocator PRIVATE fmt::fmt coverage_config)
maybe_add_dependencies (allocator spdlog)
