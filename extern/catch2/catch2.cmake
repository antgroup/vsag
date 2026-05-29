Include(FetchContent)

vsag_get_system_dep_policy (CATCH2 _vsag_dep_policy)
if (TARGET Catch2::Catch2 OR TARGET Catch2::Catch2WithMain)
    if (NOT TARGET Catch2)
        add_custom_target (Catch2)
    endif ()
    vsag_note_system_dep (catch2 Catch2::Catch2)
    return ()
elseif (NOT _vsag_dep_policy STREQUAL "OFF")
    find_package (Catch2 CONFIG QUIET)
    if (TARGET Catch2::Catch2 OR TARGET Catch2::Catch2WithMain)
        if (NOT TARGET Catch2)
            add_custom_target (Catch2)
        endif ()
        vsag_note_system_dep (catch2 Catch2::Catch2)
        return ()
    elseif (_vsag_dep_policy STREQUAL "ON")
        vsag_fail_missing_system_dep (CATCH2 Catch2 "Catch2::Catch2, Catch2::Catch2WithMain")
    endif ()
endif ()

set(catch2_urls
    https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/catch2/v3.7.1.tar.gz
)
if(DEFINED ENV{VSAG_THIRDPARTY_CATCH2})
  message(STATUS "Using local path for catch2: $ENV{VSAG_THIRDPARTY_CATCH2}")
  list(PREPEND catch2_urls "$ENV{VSAG_THIRDPARTY_CATCH2}")
endif()
FetchContent_Declare(
  Catch2
  URL      ${catch2_urls}
  URL_HASH MD5=9fcbec1dc95edcb31c6a0d6c5320e098
  DOWNLOAD_NO_PROGRESS 1
  INACTIVITY_TIMEOUT 5
  TIMEOUT 30
)

FetchContent_MakeAvailable(Catch2)
