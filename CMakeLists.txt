add_executable (create_old_version_index create_old_version_index.cpp)
target_link_libraries (create_old_version_index
        PRIVATE
        vsag
)
