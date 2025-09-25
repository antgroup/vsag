
add_subdirectory (simd)
add_subdirectory (io)
add_subdirectory (quantization)
add_subdirectory (impl)
add_subdirectory (storage)
add_subdirectory (attr)
add_subdirectory (datacell)
add_subdirectory (algorithm)
add_subdirectory (utils)
add_subdirectory (factory)

file (GLOB CPP_SRCS "*.cpp")
list (FILTER CPP_SRCS EXCLUDE REGEX "_test.cpp")

file (GLOB CPP_INDEX_SRCS "index/*.cpp")
list (FILTER CPP_INDEX_SRCS EXCLUDE REGEX "_test.cpp")

set (VSAG_SRCS ${CPP_SRCS} ${CPP_INDEX_SRCS})

add_library (vsag SHARED ${VSAG_SRCS})
add_library (vsag_static STATIC ${VSAG_SRCS})

set (VSAG_DEP_LIBS antlr4-autogen antlr4-runtime diskann simd io quantizer storage utils factory
      datacell ${IMPL_LIBS} ${ALGORITHM_LIBS} attr pthread m dl fmt::fmt nlohmann_json::nlohmann_json roaring)

target_link_libraries (vsag ${VSAG_DEP_LIBS} coverage_config)
target_link_libraries (vsag_static ${VSAG_DEP_LIBS} coverage_config)

add_dependencies (vsag spdlog)
add_dependencies (vsag_static spdlog)

maybe_add_dependencies (vsag antlr4 spdlog roaring openblas boost mkl)
maybe_add_dependencies (vsag_static antlr4 spdlog roaring openblas boost mkl)
