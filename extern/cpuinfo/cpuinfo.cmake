include (FetchContent)

vsag_get_system_dep_policy (CPUINFO _vsag_dep_policy)
if (TARGET cpuinfo::cpuinfo OR TARGET cpuinfo)
    if (NOT TARGET cpuinfo AND TARGET cpuinfo::cpuinfo)
        add_library (cpuinfo ALIAS cpuinfo::cpuinfo)
    endif ()
    if (NOT TARGET vsag_cpuinfo_headers)
        add_library (vsag_cpuinfo_headers INTERFACE)
    endif ()
    target_link_libraries (vsag_cpuinfo_headers INTERFACE cpuinfo)
    vsag_note_system_dep (cpuinfo cpuinfo)
    return ()
elseif (NOT _vsag_dep_policy STREQUAL "OFF")
    find_path (CPUINFO_INCLUDE_DIR NAMES cpuinfo.h)
    find_library (CPUINFO_LIBRARY NAMES cpuinfo)
    if (CPUINFO_INCLUDE_DIR AND CPUINFO_LIBRARY)
        add_library (cpuinfo UNKNOWN IMPORTED GLOBAL)
        set_target_properties (cpuinfo PROPERTIES
            IMPORTED_LOCATION ${CPUINFO_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${CPUINFO_INCLUDE_DIR})
        if (NOT TARGET cpuinfo::cpuinfo)
            add_library (cpuinfo::cpuinfo ALIAS cpuinfo)
        endif ()
        if (NOT TARGET vsag_cpuinfo_headers)
            add_library (vsag_cpuinfo_headers INTERFACE)
        endif ()
        target_link_libraries (vsag_cpuinfo_headers INTERFACE cpuinfo)
        vsag_note_system_dep (cpuinfo cpuinfo)
        return ()
    elseif (_vsag_dep_policy STREQUAL "ON")
        vsag_fail_missing_system_dep (CPUINFO cpuinfo "cpuinfo::cpuinfo, cpuinfo")
    endif ()
endif ()

set (cpuinfo_urls
    https://github.com/pytorch/cpuinfo/archive/ca678952a9a8eaa6de112d154e8e104b22f9ab3f.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/cpuinfo/ca678952a9a8eaa6de112d154e8e104b22f9ab3f.tar.gz
)
if (DEFINED ENV{VSAG_THIRDPARTY_CPUINFO})
    message (STATUS "Using local path for cpuinfo: $ENV{VSAG_THIRDPARTY_CPUINFO}")
    list (PREPEND cpuinfo_urls "$ENV{VSAG_THIRDPARTY_CPUINFO}")
endif ()
FetchContent_Declare (
    cpuinfo
    URL ${cpuinfo_urls}
    URL_HASH MD5=a72699bc703dfea4ab2c9c01025e46e9
    DOWNLOAD_NO_PROGRESS 1
    INACTIVITY_TIMEOUT 5
    TIMEOUT 30
)

set (CPUINFO_BUILD_TOOLS OFF CACHE BOOL "Disable some option in the library" FORCE)
set (CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "Disable some option in the library" FORCE)
set (CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "Disable some option in the library" FORCE)
set (CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "Disable some option in the library" FORCE)
set (CPUINFO_BUILD_PKG_CONFIG OFF CACHE BOOL "Disable some option in the library" FORCE)
set (CPUINFO_LIBRARY_TYPE "static")

# exclude cpuinfo in vsag installation
FetchContent_GetProperties (cpuinfo)
if (NOT cpuinfo_POPULATED)
    FetchContent_Populate (cpuinfo)
    add_subdirectory (${cpuinfo_SOURCE_DIR} ${cpuinfo_BINARY_DIR} EXCLUDE_FROM_ALL)
endif ()

if (NOT TARGET vsag_cpuinfo_headers)
    add_library (vsag_cpuinfo_headers INTERFACE)
endif ()
target_include_directories (vsag_cpuinfo_headers INTERFACE ${cpuinfo_SOURCE_DIR}/include)

install (
    TARGETS cpuinfo
    ARCHIVE DESTINATION lib)
