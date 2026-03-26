include_guard (GLOBAL)

add_library (vsag_src_common INTERFACE)
target_link_libraries (vsag_src_common INTERFACE
    fmt::fmt
    nlohmann_json::nlohmann_json
    tsl::robin_map)

if (TARGET vsag_cpuinfo_headers)
    target_link_libraries (vsag_src_common INTERFACE vsag_cpuinfo_headers)
endif ()
if (TARGET vsag_diskann_headers)
    target_link_libraries (vsag_src_common INTERFACE vsag_diskann_headers)
endif ()
if (TARGET vsag_thread_pool_headers)
    target_link_libraries (vsag_src_common INTERFACE vsag_thread_pool_headers)
endif ()
if (TARGET vsag_antlr4_runtime_headers)
    target_link_libraries (vsag_src_common INTERFACE vsag_antlr4_runtime_headers)
endif ()
if (TARGET vsag_antlr4_autogen_headers)
    target_link_libraries (vsag_src_common INTERFACE vsag_antlr4_autogen_headers)
endif ()
if (TARGET vsag_roaring_headers)
    target_link_libraries (vsag_src_common INTERFACE vsag_roaring_headers)
endif ()
