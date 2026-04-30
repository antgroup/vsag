include (FetchContent)

vsag_get_system_dep_policy (ARGPARSE _vsag_dep_policy)
if (TARGET argparse::argparse)
    vsag_note_system_dep (argparse argparse::argparse)
    return ()
elseif (NOT _vsag_dep_policy STREQUAL "OFF")
    find_package (argparse CONFIG QUIET)
    if (TARGET argparse::argparse)
        vsag_note_system_dep (argparse argparse::argparse)
        return ()
    elseif (_vsag_dep_policy STREQUAL "ON")
        vsag_fail_missing_system_dep (ARGPARSE argparse "argparse::argparse")
    endif ()
endif ()

set (argparse_urls
    https://github.com/p-ranav/argparse/archive/refs/tags/v3.1.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/argparse/v3.1.tar.gz
)
if (DEFINED ENV{VSAG_THIRDPARTY_ARGPARSE})
    message (STATUS "Using local path for argparse: $ENV{VSAG_THIRDPARTY_ARGPARSE}")
    list (PREPEND argparse_urls "$ENV{VSAG_THIRDPARTY_ARGPARSE}")
endif ()
FetchContent_Declare (
    argparse
    URL ${argparse_urls}
    URL_HASH MD5=11822ccbe1bd8d84c948450d24281b67
    DOWNLOAD_NO_PROGRESS 1
    INACTIVITY_TIMEOUT 5
    TIMEOUT 30)

FetchContent_MakeAvailable (argparse)
