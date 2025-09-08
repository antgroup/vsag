include(FetchContent)
FetchContent_Declare(
        cpr
        URL https://github.com/libcpr/cpr/archive/refs/tags/1.11.2.tar.gz
        https://vsagcache.oss-rg-china-mainland.aliyuncs.com/cpr/1.11.2.tar.gz
        URL_HASH MD5=639cff98d5124cf06923a0975fb427d8
        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30
)

FetchContent_GetProperties(cpr)
if (NOT cpr_POPULATED)
    FetchContent_Populate(cpr)

endif ()

set(CPR_ENABLE_SSL OFF)
add_subdirectory(${cpr_SOURCE_DIR} ${cpr_BINARY_DIR})
