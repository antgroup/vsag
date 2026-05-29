
include (FetchContent)

vsag_get_system_dep_policy (THREAD_POOL _vsag_dep_policy)
if (TARGET vsag_thread_pool_headers)
    vsag_note_system_dep (thread_pool vsag_thread_pool_headers)
    return ()
elseif (DEFINED VSAG_THREAD_POOL_INCLUDE_DIR AND EXISTS "${VSAG_THREAD_POOL_INCLUDE_DIR}/ThreadPool.h")
    add_library (vsag_thread_pool_headers INTERFACE)
    target_include_directories (vsag_thread_pool_headers INTERFACE ${VSAG_THREAD_POOL_INCLUDE_DIR})
    vsag_note_system_dep (thread_pool "${VSAG_THREAD_POOL_INCLUDE_DIR}")
    return ()
elseif (_vsag_dep_policy STREQUAL "ON")
    vsag_fail_missing_system_dep (THREAD_POOL VSAG_THREAD_POOL_INCLUDE_DIR "vsag_thread_pool_headers")
endif ()

set (thread_pool_urls
    https://github.com/log4cplus/ThreadPool/archive/3507796e172d36555b47d6191f170823d9f6b12c.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/thread_pool/3507796e172d36555b47d6191f170823d9f6b12c.tar.gz
)
if (DEFINED ENV{VSAG_THIRDPARTY_THREAD_POOL})
    message (STATUS "Using local path for thread_pool: $ENV{VSAG_THIRDPARTY_THREAD_POOL}")
    list (PREPEND thread_pool_urls "$ENV{VSAG_THIRDPARTY_THREAD_POOL}")
endif ()
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
