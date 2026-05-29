include (FetchContent)

vsag_get_system_dep_policy (ROARING _vsag_dep_policy)
if (TARGET roaring::roaring OR TARGET roaring)
    if (NOT TARGET roaring AND TARGET roaring::roaring)
        add_library (roaring ALIAS roaring::roaring)
    endif ()
    if (NOT TARGET vsag_roaring_headers)
        add_library (vsag_roaring_headers INTERFACE)
    endif ()
    target_link_libraries (vsag_roaring_headers INTERFACE roaring)
    vsag_note_system_dep (roaring roaring)
    return ()
elseif (NOT _vsag_dep_policy STREQUAL "OFF")
    find_path (ROARING_INCLUDE_DIR NAMES roaring.hh)
    find_library (ROARING_LIBRARY NAMES roaring)
    if (ROARING_INCLUDE_DIR AND ROARING_LIBRARY)
        add_library (roaring UNKNOWN IMPORTED GLOBAL)
        set_target_properties (roaring PROPERTIES
            IMPORTED_LOCATION ${ROARING_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${ROARING_INCLUDE_DIR})
        if (NOT TARGET roaring::roaring)
            add_library (roaring::roaring ALIAS roaring)
        endif ()
        if (NOT TARGET vsag_roaring_headers)
            add_library (vsag_roaring_headers INTERFACE)
        endif ()
        target_link_libraries (vsag_roaring_headers INTERFACE roaring)
        vsag_note_system_dep (roaring roaring)
        return ()
    elseif (_vsag_dep_policy STREQUAL "ON")
        vsag_fail_missing_system_dep (ROARING roaring "roaring::roaring, roaring")
    endif ()
endif ()

set (roaringbitmap_urls
    https://github.com/RoaringBitmap/CRoaring/archive/refs/tags/v3.0.1.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/roaringbitmap/v3.0.1.tar.gz
)
if (DEFINED ENV{VSAG_THIRDPARTY_ROARINGBITMAP})
    message (STATUS "Using local path for roaringbitmap: $ENV{VSAG_THIRDPARTY_ROARINGBITMAP}")
    list (PREPEND roaringbitmap_urls "$ENV{VSAG_THIRDPARTY_ROARINGBITMAP}")
endif ()
FetchContent_Declare (
    roaringbitmap
    URL ${roaringbitmap_urls}
    URL_HASH MD5=463db911f97d5da69393d4a3190f9201
    DOWNLOAD_NO_PROGRESS 0
    INACTIVITY_TIMEOUT 5
    # filesize ~= 90MiB
    TIMEOUT 90
)

set (ROARING_USE_CPM OFF)
set (ENABLE_ROARING_TESTS OFF)

if (DISABLE_AVX_FORCE OR NOT COMPILER_AVX_SUPPORTED)
  set(ROARING_DISABLE_AVX ON)
endif ()

if (DISABLE_AVX512_FORCE OR NOT COMPILER_AVX512_SUPPORTED)
  set (ROARING_DISABLE_AVX512 ON)
endif ()

# exclude roaringbitmap in vsag installation
FetchContent_GetProperties (roaringbitmap)
if (NOT roaringbitmap_POPULATED)
    FetchContent_Populate (roaringbitmap)
    add_subdirectory (${roaringbitmap_SOURCE_DIR} ${roaringbitmap_BINARY_DIR} EXCLUDE_FROM_ALL)
    target_compile_options (roaring PRIVATE -Wno-unused-function)
endif ()

if (NOT TARGET vsag_roaring_headers)
    add_library (vsag_roaring_headers INTERFACE)
endif ()
target_include_directories (vsag_roaring_headers INTERFACE
    ${roaringbitmap_SOURCE_DIR}/include
    ${roaringbitmap_SOURCE_DIR}/cpp)
