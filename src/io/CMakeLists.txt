
set (IO_SRC
        io_parameter.cpp
        memory_io_parameter.cpp
        memory_block_io_parameter.cpp
        buffer_io_parameter.cpp
        buffer_io.cpp
        async_io_parameter.cpp
        async_io.cpp
        mmap_io_parameter.cpp
        memory_block_io.cpp
)

add_library (io OBJECT ${IO_SRC})
target_link_libraries (io PUBLIC fmt::fmt aio)
maybe_add_dependencies (io spdlog)
