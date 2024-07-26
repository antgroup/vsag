
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    URL_HASH MD5=d603041cbc6051edbaa02ebb82cf0aa9
)

FetchContent_MakeAvailable(nlohmann_json)
include_directories(${nlohmann_json_SOURCE_DIR}/include)
