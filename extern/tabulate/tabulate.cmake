include (FetchContent)

vsag_get_system_dep_policy (TABULATE _vsag_dep_policy)
if (TARGET tabulate::tabulate OR TARGET tabulate)
    if (NOT TARGET tabulate AND TARGET tabulate::tabulate)
        add_library (tabulate ALIAS tabulate::tabulate)
    endif ()
    vsag_note_system_dep (tabulate tabulate)
    return ()
elseif (NOT _vsag_dep_policy STREQUAL "OFF")
    find_package (tabulate CONFIG QUIET)
    if (TARGET tabulate::tabulate OR TARGET tabulate)
        if (NOT TARGET tabulate AND TARGET tabulate::tabulate)
            add_library (tabulate ALIAS tabulate::tabulate)
        endif ()
        vsag_note_system_dep (tabulate tabulate)
        return ()
    elseif (_vsag_dep_policy STREQUAL "ON")
        vsag_fail_missing_system_dep (TABULATE tabulate "tabulate::tabulate, tabulate")
    endif ()
endif ()

set (tabulate_urls
    https://github.com/p-ranav/tabulate/archive/3a58301067bbc03da89ae5a51b3e05b7da719d38.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    # the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/tabulate/3a58301067bbc03da89ae5a51b3e05b7da719d38.tar.gz
)
if (DEFINED ENV{VSAG_THIRDPARTY_TABULATE})
    message (STATUS "Using local path for tabulate: $ENV{VSAG_THIRDPARTY_TABULATE}")
    list (PREPEND tabulate_urls "$ENV{VSAG_THIRDPARTY_TABULATE}")
endif ()
FetchContent_Declare (
    tabulate
    URL ${tabulate_urls}
    URL_HASH MD5=9d396f30fcc513abbb970773c8ddf8ff
    DOWNLOAD_NO_PROGRESS 1
    INACTIVITY_TIMEOUT 5
    TIMEOUT 30)

FetchContent_MakeAvailable (tabulate)
