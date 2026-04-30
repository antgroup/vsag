
include (FetchContent)

vsag_get_system_dep_policy (NLOHMANN_JSON _vsag_dep_policy)
if (TARGET nlohmann_json::nlohmann_json)
    vsag_note_system_dep (nlohmann_json nlohmann_json::nlohmann_json)
    return ()
elseif (NOT _vsag_dep_policy STREQUAL "OFF")
    find_package (nlohmann_json CONFIG QUIET)
    if (TARGET nlohmann_json::nlohmann_json)
        vsag_note_system_dep (nlohmann_json nlohmann_json::nlohmann_json)
        return ()
    elseif (_vsag_dep_policy STREQUAL "ON")
        vsag_fail_missing_system_dep (NLOHMANN_JSON nlohmann_json "nlohmann_json::nlohmann_json")
    endif ()
endif ()

set (nlohmann_json_urls
    https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/nlohmann_json/v3.11.3.tar.gz
)
if (DEFINED ENV{VSAG_THIRDPARTY_JSON})
    message (STATUS "Using local path for nlohmann_json: $ENV{VSAG_THIRDPARTY_JSON}")
    list (PREPEND nlohmann_json_urls "$ENV{VSAG_THIRDPARTY_JSON}")
endif ()
FetchContent_Declare (
    nlohmann_json
    URL ${nlohmann_json_urls}
    URL_HASH MD5=d603041cbc6051edbaa02ebb82cf0aa9
    DOWNLOAD_NO_PROGRESS 1
    INACTIVITY_TIMEOUT 5
    TIMEOUT 30
)

FetchContent_MakeAvailable (nlohmann_json)
