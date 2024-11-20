set (FACTORY_SRCS factory.cpp)
add_library (factory ${FACTORY_SRCS})
target_link_libraries (factory roaring index)
