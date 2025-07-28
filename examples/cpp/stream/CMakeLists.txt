set (STREAM_SRC append_buffer.cpp
        meta_table.cpp read_stream.cpp record_write_stream.cpp sender.cpp receiver.cpp)

add_library (stream SHARED ${STREAM_SRC})

add_executable (example1 example1.cpp)
target_link_libraries (example1 PUBLIC vsag stream)
