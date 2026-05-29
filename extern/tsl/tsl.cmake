
include (FetchContent)

vsag_get_system_dep_policy (TSL _vsag_dep_policy)
if (TARGET tsl::robin_map)
    vsag_note_system_dep (tsl tsl::robin_map)
    return ()
elseif (NOT _vsag_dep_policy STREQUAL "OFF")
    find_package (tsl-robin-map CONFIG QUIET)
    if (TARGET tsl::robin_map)
        vsag_note_system_dep (tsl tsl::robin_map)
        return ()
    elseif (_vsag_dep_policy STREQUAL "ON")
        vsag_fail_missing_system_dep (TSL tsl-robin-map "tsl::robin_map")
    endif ()
endif ()

set (tsl_urls
    https://github.com/Tessil/robin-map/archive/refs/tags/v1.4.0.tar.gz
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/robin-map/v1.4.0.tar.gz
)
if (DEFINED ENV{VSAG_THIRDPARTY_TSL})
    message (STATUS "Using local path for tsl: $ENV{VSAG_THIRDPARTY_TSL}")
    list (PREPEND tsl_urls "$ENV{VSAG_THIRDPARTY_TSL}")
endif ()
FetchContent_Declare (
    tsl
    URL ${tsl_urls}
    URL_HASH MD5=d56a879c94e021c55d8956e37deb3e4f
    DOWNLOAD_NO_PROGRESS 1
    INACTIVITY_TIMEOUT 5
    TIMEOUT 30)

FetchContent_MakeAvailable (tsl)
