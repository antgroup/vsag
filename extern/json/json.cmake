
include(FetchContent)

set(nlohmann_json_urls
    https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/nlohmann_json/v3.11.3.tar.gz
)
if(DEFINED ENV{VSAG_THIRDPARTY_JSON})
  message(STATUS "Using local path for nlohmann_json: $ENV{VSAG_THIRDPARTY_JSON}")
  list(PREPEND nlohmann_json_urls "$ENV{VSAG_THIRDPARTY_JSON}")
endif()
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
