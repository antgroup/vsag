
include (FetchContent)

set (thread_pool_urls
    https://github.com/log4cplus/ThreadPool/archive/3507796e172d36555b47d6191f170823d9f6b12c.tar.gz
)
vsag_resolve_thirdparty_override (
    THREAD_POOL 3507796e172d36555b47d6191f170823d9f6b12c thread_pool_urls)
FetchContent_Declare (
    thread_pool
    URL ${thread_pool_urls}
    URL_HASH MD5=e5b67a770f9f37500561a431d1dc1afe
    DOWNLOAD_NO_PROGRESS 1
    INACTIVITY_TIMEOUT 5
    TIMEOUT 30)

FetchContent_MakeAvailable (thread_pool)

if (NOT TARGET vsag_thread_pool_headers)
    add_library (vsag_thread_pool_headers INTERFACE)
endif ()
target_include_directories (vsag_thread_pool_headers INTERFACE ${thread_pool_SOURCE_DIR})
