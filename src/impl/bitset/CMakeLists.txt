
set (BITSET_SRC
        bitset.cpp
        computable_bitset.cpp
        computable_bitset.h
        fast_bitset.cpp
        fast_bitset.h
        sparse_bitset.cpp
        sparse_bitset.h
)

add_library (bitset OBJECT ${BITSET_SRC})
target_link_libraries (bitset PRIVATE roaring)
maybe_add_dependencies (bitset spdlog)
