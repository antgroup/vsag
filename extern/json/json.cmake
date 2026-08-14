
include(FetchContent)

set(nlohmann_json_urls
    https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
)
vsag_resolve_thirdparty_override (JSON v3.11.3 nlohmann_json_urls)
FetchContent_Declare(
    nlohmann_json
    URL ${nlohmann_json_urls}
    URL_HASH MD5=d603041cbc6051edbaa02ebb82cf0aa9
    DOWNLOAD_NO_PROGRESS 1
    INACTIVITY_TIMEOUT 5
    TIMEOUT 30
)

FetchContent_MakeAvailable(nlohmann_json)
include_directories(${nlohmann_json_SOURCE_DIR}/include)
