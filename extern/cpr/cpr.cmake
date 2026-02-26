include(FetchContent)
set(cpr_urls
    https://github.com/libcpr/cpr/archive/refs/tags/1.11.2.tar.gz
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/cpr/1.11.2.tar.gz
)
if(DEFINED ENV{VSAG_THIRDPARTY_CPR})
  message(STATUS "Using local path for cpr: $ENV{VSAG_THIRDPARTY_CPR}")
  list(PREPEND cpr_urls "$ENV{VSAG_THIRDPARTY_CPR}")
endif()
FetchContent_Declare(
        cpr
        URL ${cpr_urls}
        URL_HASH MD5=639cff98d5124cf06923a0975fb427d8
        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30
)

FetchContent_GetProperties(cpr)
if (NOT cpr_POPULATED)
    FetchContent_Populate(cpr)

    # check patch marker
    if (NOT EXISTS "${cpr_SOURCE_DIR}/patch_applied_marker")
        set(PATCH_FILE "${CMAKE_SOURCE_DIR}/extern/cpr/fix_curl.patch")
        execute_process(
                COMMAND patch -p1
                INPUT_FILE ${PATCH_FILE}
                WORKING_DIRECTORY ${cpr_SOURCE_DIR}
                RESULT_VARIABLE PATCH_RESULT
        )
        if (NOT PATCH_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to apply patch CURL, please make clean and retry.")
        endif ()

        # create a marker
        file(WRITE "${cpr_SOURCE_DIR}/patch_applied_marker" "Patch applied")
        message(STATUE "Write patch marker in ${cpr_SOURCE_DIR}/patch_applied_marker")
    endif ()
endif ()

set(CPR_ENABLE_SSL OFF)
add_subdirectory(${cpr_SOURCE_DIR} ${cpr_BINARY_DIR})
