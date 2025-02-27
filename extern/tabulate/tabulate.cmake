include (FetchContent)

FetchContent_Declare (
        tabulate
        URL https://github.com/p-ranav/tabulate/archive/3a58301067bbc03da89ae5a51b3e05b7da719d38.tar.gz
        # this url is maintained by the vsag project, if it's broken, please try
        # the latest commit or contact the vsag project
        http://vsagcache.oss-rg-china-mainland.aliyuncs.com/tabulate/3a58301067bbc03da89ae5a51b3e05b7da719d38.tar.gz
        URL_HASH MD5=9d396f30fcc513abbb970773c8ddf8ff
        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30
)

FetchContent_MakeAvailable (tabulate)
include_directories (${tabulate_SOURCE_DIR}/single_include)
