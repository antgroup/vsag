
file (GLOB DATACELL_SRCS "*.cpp")
list (FILTER DATACELL_SRCS EXCLUDE REGEX "_test.cpp")

add_library (datacell OBJECT ${DATACELL_SRCS})
target_link_libraries(datacell PRIVATE transform quantizer io attr)
add_dependencies (datacell spdlog fmt::fmt)
