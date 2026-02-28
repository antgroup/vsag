include (FetchContent)

set(argparse_urls
    https://github.com/p-ranav/argparse/archive/refs/tags/v3.1.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/argparse/v3.1.tar.gz
)
if(DEFINED ENV{VSAG_THIRDPARTY_ARGPARSE})
  message(STATUS "Using local path for argparse: $ENV{VSAG_THIRDPARTY_ARGPARSE}")
  list(PREPEND argparse_urls "$ENV{VSAG_THIRDPARTY_ARGPARSE}")
endif()
FetchContent_Declare (
        argparse
        URL ${argparse_urls}
        URL_HASH MD5=11822ccbe1bd8d84c948450d24281b67
        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30
)

FetchContent_MakeAvailable (argparse)
include_directories (${argparse_SOURCE_DIR}/include)
