include (FetchContent)

set(argparse_urls
    https://github.com/p-ranav/argparse/archive/refs/tags/v3.1.tar.gz
)
vsag_resolve_thirdparty_override (ARGPARSE v3.1 argparse_urls)
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
