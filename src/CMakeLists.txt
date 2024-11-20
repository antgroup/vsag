
add_subdirectory (simd)
add_subdirectory (data_cell)
add_subdirectory (algorithm/hnswlib)
add_subdirectory (index)
add_subdirectory (factory)
add_subdirectory (impl)

file (GLOB CPP_SRCS "*.cpp")
list (FILTER CPP_SRCS EXCLUDE REGEX "_test.cpp")

set (VSAG_SRCS ${CPP_SRCS})
add_library (vsag SHARED ${VSAG_SRCS})
add_library (vsag_static STATIC ${VSAG_SRCS})

set (VSAG_DEP_LIBS pthread m dl fmt::fmt-header-only nlohmann_json::nlohmann_json roaring
        simd datacell index factory conjugate_graph)
target_link_libraries (vsag ${VSAG_DEP_LIBS})
target_link_libraries (vsag_static ${VSAG_DEP_LIBS})
maybe_add_dependencies (vsag spdlog roaring openblas boost mkl)
maybe_add_dependencies (vsag_static spdlog roaring openblas boost mkl)
