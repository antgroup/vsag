
add_subdirectory (simd)
add_subdirectory (io)
add_subdirectory (quantization)

file (GLOB CPP_SRCS "*.cpp")
list (FILTER CPP_SRCS EXCLUDE REGEX "_test.cpp")

file(GLOB CPP_ATTR_SRCS "attr/*.cpp")
list(FILTER CPP_ATTR_SRCS EXCLUDE REGEX "_test.cpp")

file (GLOB CPP_IMPL_SRCS "impl/*.cpp")
file (GLOB CPP_IMPL_EXTRA_SRCS "impl/**/*.cpp")
list (APPEND CPP_IMPL_SRCS ${CPP_IMPL_EXTRA_SRCS})
list (FILTER CPP_IMPL_SRCS EXCLUDE REGEX "_test.cpp")

file (GLOB CPP_INDEX_SRCS "index/*.cpp")
list (FILTER CPP_INDEX_SRCS EXCLUDE REGEX "_test.cpp")

file (GLOB CPP_DATA_CELL_SRCS "data_cell/*.cpp")
list (FILTER CPP_DATA_CELL_SRCS EXCLUDE REGEX "_test.cpp")

file (GLOB CPP_ALGORITHM_SRCS "algorithm/*.cpp")
file (GLOB CPP_ALGORITHM_EXTRA_SRCS "algorithm/**/*.cpp")
list (APPEND CPP_ALGORITHM_SRCS ${CPP_ALGORITHM_EXTRA_SRCS})
list (FILTER CPP_ALGORITHM_SRCS EXCLUDE REGEX "_test.cpp")

file (GLOB CPP_UTILS_SRCS "utils/*.cpp")
list (FILTER CPP_UTILS_SRCS EXCLUDE REGEX "_test.cpp")

set(VSAG_SRCS ${CPP_SRCS} ${CPP_ATTR_SRCS} ${CPP_INDEX_SRCS} ${CPP_IMPL_SRCS}
        ${CPP_DATA_CELL_SRCS} ${CPP_ALGORITHM_SRCS} ${CPP_UTILS_SRCS})

add_library (vsag SHARED ${VSAG_SRCS})
add_library (vsag_static STATIC ${VSAG_SRCS})

set(VSAG_DEP_LIBS antlr4-autogen antlr4-runtime diskann pthread m dl simd io quantizer fmt::fmt-header-only nlohmann_json::nlohmann_json roaring)
target_link_libraries (vsag ${VSAG_DEP_LIBS} coverage_config)
target_link_libraries (vsag_static ${VSAG_DEP_LIBS} coverage_config)
add_dependencies (vsag spdlog)
add_dependencies (vsag_static spdlog)

maybe_add_dependencies(vsag antlr4 spdlog roaring openblas boost mkl)
maybe_add_dependencies(vsag_static antlr4 spdlog roaring openblas boost mkl)
