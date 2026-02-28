
include(FetchContent)

set(tsl_urls
    https://github.com/Tessil/robin-map/archive/refs/tags/v1.4.0.tar.gz
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/robin-map/v1.4.0.tar.gz
)
if(DEFINED ENV{VSAG_THIRDPARTY_TSL})
  message(STATUS "Using local path for tsl: $ENV{VSAG_THIRDPARTY_TSL}")
  list(PREPEND tsl_urls "$ENV{VSAG_THIRDPARTY_TSL}")
endif()
FetchContent_Declare(
        tsl
        URL ${tsl_urls}
        URL_HASH MD5=d56a879c94e021c55d8956e37deb3e4f
        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30
)

FetchContent_MakeAvailable(tsl)
include_directories(${tsl_SOURCE_DIR}/include)
