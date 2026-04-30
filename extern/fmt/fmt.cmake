
include (FetchContent)

vsag_get_system_dep_policy (FMT _vsag_dep_policy)
if (TARGET fmt::fmt OR TARGET fmt::fmt-header-only)
    if (NOT TARGET fmt::fmt AND TARGET fmt::fmt-header-only)
        add_library (fmt::fmt ALIAS fmt::fmt-header-only)
    endif ()
    if (NOT TARGET fmt::fmt-header-only AND TARGET fmt::fmt)
        add_library (fmt::fmt-header-only ALIAS fmt::fmt)
    endif ()
    vsag_note_system_dep (fmt fmt::fmt)
    return ()
elseif (NOT _vsag_dep_policy STREQUAL "OFF")
    find_path (FMT_INCLUDE_DIR NAMES fmt/format.h)
    find_library (FMT_LIBRARY NAMES fmt)
    if (FMT_INCLUDE_DIR AND FMT_LIBRARY)
        add_library (fmt::fmt UNKNOWN IMPORTED GLOBAL)
        set_target_properties (fmt::fmt PROPERTIES
            IMPORTED_LOCATION ${FMT_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${FMT_INCLUDE_DIR})
        add_library (fmt::fmt-header-only ALIAS fmt::fmt)
        vsag_note_system_dep (fmt fmt::fmt)
        return ()
    elseif (_vsag_dep_policy STREQUAL "ON")
        vsag_fail_missing_system_dep (FMT fmt "fmt::fmt, fmt::fmt-header-only")
    endif ()
endif ()

# suppress "stringop-overflow" warning which caused by a compiler bug in gcc 10 or earlier
# ref: https://github.com/fmtlib/fmt/issues/2708
set (FMT_SYSTEM_HEADERS ON)

set (fmt_urls
    https://github.com/fmtlib/fmt/archive/refs/tags/10.2.1.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/fmt/10.2.1.tar.gz
)
if (DEFINED ENV{VSAG_THIRDPARTY_FMT})
    message (STATUS "Using local path for fmt: $ENV{VSAG_THIRDPARTY_FMT}")
    list (PREPEND fmt_urls "$ENV{VSAG_THIRDPARTY_FMT}")
endif ()
FetchContent_Declare (
    fmt
    URL ${fmt_urls}
    URL_HASH MD5=dc09168c94f90ea890257995f2c497a5
    DOWNLOAD_NO_PROGRESS 1
    INACTIVITY_TIMEOUT 5
    TIMEOUT 30
)

# exclude fmt in vsag installation
FetchContent_GetProperties (fmt)
if (NOT fmt_POPULATED)
    FetchContent_Populate (fmt)
    add_subdirectory (${fmt_SOURCE_DIR} ${fmt_BINARY_DIR} EXCLUDE_FROM_ALL)
endif ()
