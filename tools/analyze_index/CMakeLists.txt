
add_executable (analyze_index analyze_index.cpp)
target_link_libraries (analyze_index
        PRIVATE
        argparse::argparse
        vsag
)


