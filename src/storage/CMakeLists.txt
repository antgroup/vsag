
set (STORAGE_SRC
  footer.cpp
  serialization.cpp
  stream_reader.cpp
  stream_writer.cpp
)

add_library (storage OBJECT ${STORAGE_SRC})
target_link_libraries (storage PUBLIC fmt::fmt-header-only)
add_dependencies (storage spdlog)
