
include(FetchContent)

FetchContent_Declare(
        thread_pool
        URL https://github.com/log4cplus/ThreadPool/archive/3507796e172d36555b47d6191f170823d9f6b12c.tar.gz
            # this url is maintained by the vsag project, if it's broken, please try
            #  the latest commit or contact the vsag project
            https://vsagcache.oss-rg-china-mainland.aliyuncs.com/thread_pool/3507796e172d36555b47d6191f170823d9f6b12c.tar.gz
            URL_HASH MD5=e5b67a770f9f37500561a431d1dc1afe

        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30
)

FetchContent_MakeAvailable(thread_pool)
include_directories(${thread_pool_SOURCE_DIR}/)
