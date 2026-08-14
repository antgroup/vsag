
include(FetchContent)

set(tsl_urls
    https://github.com/Tessil/robin-map/archive/refs/tags/v1.4.0.tar.gz
)
vsag_resolve_thirdparty_override (TSL v1.4.0 tsl_urls)
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
