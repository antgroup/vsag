
add_subdirectory (ivf_partition)
add_subdirectory (hnswlib)

file (GLOB ALGORITHM_SRCS "*.cpp")
list (FILTER ALGORITHM_SRCS EXCLUDE REGEX "_test.cpp")

add_library(algorithm OBJECT ${ALGORITHM_SRCS})
target_link_libraries (algorithm PUBLIC coverage_config)
maybe_add_dependencies(algorithm spdlog fmt::fmt antlr4)


set (ALGORITHM_LIBS
        algorithm
        hnswlib
        ivf_partition
        PARENT_SCOPE)
