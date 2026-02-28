
include(FetchContent)

set(thread_pool_urls
    https://github.com/log4cplus/ThreadPool/archive/3507796e172d36555b47d6191f170823d9f6b12c.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/thread_pool/3507796e172d36555b47d6191f170823d9f6b12c.tar.gz
)
if(DEFINED ENV{VSAG_THIRDPARTY_THREAD_POOL})
  message(STATUS "Using local path for thread_pool: $ENV{VSAG_THIRDPARTY_THREAD_POOL}")
  list(PREPEND thread_pool_urls "$ENV{VSAG_THIRDPARTY_THREAD_POOL}")
endif()
FetchContent_Declare(
        thread_pool
        URL ${thread_pool_urls}
        URL_HASH MD5=e5b67a770f9f37500561a431d1dc1afe
        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30
)

FetchContent_MakeAvailable(thread_pool)
include_directories(${thread_pool_SOURCE_DIR}/)
