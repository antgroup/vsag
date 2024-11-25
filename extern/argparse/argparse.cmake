include (FetchContent)

FetchContent_Declare (
        argparse
        URL https://github.com/p-ranav/argparse/archive/refs/tags/v3.1.tar.gz
        # this url is maintained by the vsag project, if it's broken, please try
        #  the latest commit or contact the vsag project
        http://vsagcache.oss-rg-china-mainland.aliyuncs.com/argparse/v3.1.tar.gz
        URL_HASH MD5=11822ccbe1bd8d84c948450d24281b67
        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30
)

FetchContent_MakeAvailable (argparse)
include_directories (${argparse_SOURCE_DIR}/include)
