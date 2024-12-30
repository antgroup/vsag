
set (IO_SRC
        io_parameter.cpp
        memory_io_parameter.cpp
        memory_block_io_parameter.cpp
        buffer_io_parameter.cpp
)

add_library (io OBJECT ${IO_SRC})
