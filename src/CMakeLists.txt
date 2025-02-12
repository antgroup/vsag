
add_subdirectory (simd)
add_subdirectory (io)
add_subdirectory (quantization)

file (GLOB CPP_SRCS "*.cpp")
file (GLOB CPP_FACTORY_SRCS "factory/*.cpp")
file (GLOB CPP_CONJUGATE_GRAPH_SRCS "impl/*.cpp")
file (GLOB CPP_INDEX_SRCS "index/*.cpp")
file (GLOB CPP_HNSWLIB_SRCS "algorithm/hnswlib/*.cpp")
file (GLOB CPP_DATA_CELL_SRCS "data_cell/*.cpp")
file (GLOB CPP_ALGORITHM_SRCS "algorithm/*.cpp")
list (FILTER CPP_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_FACTORY_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_CONJUGATE_GRAPH_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_INDEX_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_DATA_CELL_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_ALGORITHM_SRCS EXCLUDE REGEX "_test.cpp")

file (GLOB CPP_UTILS_SRCS "utils/*.cpp")
list (FILTER CPP_UTILS_SRCS EXCLUDE REGEX "_test.cpp")

set (VSAG_SRCS ${CPP_SRCS} ${CPP_FACTORY_SRCS} ${CPP_INDEX_SRCS} ${CPP_CONJUGATE_GRAPH_SRCS} ${CPP_HNSWLIB_SRCS}
        ${CPP_HNSWLIB_SRCS} ${CPP_DATA_CELL_SRCS} ${CPP_ALGORITHM_SRCS} ${CPP_UTILS_SRCS})

add_library (vsag SHARED ${VSAG_SRCS})
add_library (vsag_static STATIC ${VSAG_SRCS})

set (VSAG_DEP_LIBS diskann pthread m dl simd io quantizer fmt::fmt-header-only nlohmann_json::nlohmann_json roaring)
target_link_libraries (vsag ${VSAG_DEP_LIBS} coverage_config)
target_link_libraries (vsag_static ${VSAG_DEP_LIBS} coverage_config)

maybe_add_dependencies (vsag spdlog roaring openblas boost mkl)
maybe_add_dependencies (vsag_static spdlog roaring openblas boost mkl)
