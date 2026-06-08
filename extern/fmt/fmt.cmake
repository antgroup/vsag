
include (FetchContent)

# suppress "stringop-overflow" warning which caused by a compiler bug in gcc 10 or earlier
# ref: https://github.com/fmtlib/fmt/issues/2708
set (FMT_SYSTEM_HEADERS ON)

vsag_get_system_dep_policy (FMT _fmt_policy)
set (FMT_FOUND FALSE)

if (NOT _fmt_policy STREQUAL "OFF")
    if (TARGET fmt::fmt)
        vsag_target_has_header (fmt::fmt fmt/format.h _fmt_has_header)
        if (_fmt_has_header)
            set (FMT_FOUND TRUE)
            message (STATUS "Using pre-existing fmt::fmt target")
        else ()
            message (STATUS "Ignoring pre-existing fmt::fmt target without fmt/format.h")
        endif ()
    endif ()

    if (NOT FMT_FOUND)
        set (_fmt_include_hints "")
        if (DEFINED fmt_DIR AND NOT "${fmt_DIR}" STREQUAL "")
            get_filename_component (_fmt_prefix "${fmt_DIR}/../../.." ABSOLUTE)
            list (APPEND _fmt_include_hints "${_fmt_prefix}" "${_fmt_prefix}/include")
        endif ()

        unset (_fmt_include_dir CACHE)
        unset (_vsag_fmt_include_dir CACHE)
        find_path (_vsag_fmt_include_dir
            NAMES fmt/format.h
            HINTS ${_fmt_include_hints}
            PATH_SUFFIXES include)
        set (_fmt_include_dir "${_vsag_fmt_include_dir}")
        unset (_vsag_fmt_include_dir CACHE)

        if (_fmt_include_dir)
            find_package (fmt CONFIG QUIET)
            if (TARGET fmt::fmt)
                vsag_target_has_header (fmt::fmt fmt/format.h _fmt_has_header)
                if (_fmt_has_header)
                    set (FMT_FOUND TRUE)
                    message (STATUS "Found fmt via find_package(fmt CONFIG)")
                else ()
                    message (STATUS "Ignoring fmt package without fmt/format.h")
                endif ()
            else ()
                message (STATUS "Ignoring fmt package without fmt::fmt target")
            endif ()
        else ()
            message (STATUS "fmt/format.h was not found in system include paths")
        endif ()
    endif ()

    if (NOT FMT_FOUND AND _fmt_policy STREQUAL "ON")
        vsag_fail_missing_system_dep (FMT fmt fmt::fmt)
    endif ()
endif ()

if (FMT_FOUND)
    if (NOT TARGET fmt::fmt-header-only)
        add_library (fmt::fmt-header-only INTERFACE IMPORTED GLOBAL)
        target_link_libraries (fmt::fmt-header-only INTERFACE fmt::fmt)
    endif ()
    return ()
endif ()

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
