include(FetchContent)

function(fetch_and_patch trial)
    set(TARGET_NAME "cpr_${trial}")
    set(PATCH_FILE "${CMAKE_SOURCE_DIR}/extern/cpr/fix_curl.patch")

    FetchContent_Declare(
            ${TARGET_NAME}
            URL https://github.com/libcpr/cpr/archive/refs/tags/1.11.2.tar.gz
            https://vsagcache.oss-rg-china-mainland.aliyuncs.com/cpr/1.11.2.tar.gz
            URL_HASH MD5=639cff98d5124cf06923a0975fb427d8
            DOWNLOAD_NO_PROGRESS 1
            INACTIVITY_TIMEOUT 5
            TIMEOUT 30
    )

    unset(${TARGET_NAME}_POPULATED CACHE)
    set(${TARGET_NAME}_POPULATED 0)

    FetchContent_GetProperties(${TARGET_NAME})
    if (NOT ${TARGET_NAME}_POPULATED)
        FetchContent_Populate(${TARGET_NAME})
    endif ()

    find_package(Git REQUIRED)

    message(STATUS "cpr start ${trial} patch in ${${TARGET_NAME}_SOURCE_DIR}")
    execute_process(
            COMMAND patch -p1
            INPUT_FILE ${PATCH_FILE}
            WORKING_DIRECTORY ${${TARGET_NAME}_SOURCE_DIR}
            RESULT_VARIABLE PATCH_RESULT
    )
    set(PATCH_RESULT ${PATCH_RESULT} PARENT_SCOPE)
    message(STATUS "cpr patch result: ${PATCH_RESULT}")

    if (NOT PATCH_RESULT EQUAL 0)
        set(${TARGET_NAME}_SUBBUILD_DIR "${CMAKE_BINARY_DIR}/_deps/${TARGET_NAME}-subbuild")

        if (DEFINED ${TARGET_NAME}_SOURCE_DIR AND EXISTS "${${TARGET_NAME}_SOURCE_DIR}")
            message(STATUS "removing ${${TARGET_NAME}_SOURCE_DIR}")
            file(REMOVE_RECURSE "${${TARGET_NAME}_SOURCE_DIR}")
        endif ()

        if (DEFINED ${TARGET_NAME}_BINARY_DIR AND EXISTS "${${TARGET_NAME}_BINARY_DIR}")
            message(STATUS "removing ${${TARGET_NAME}_BINARY_DIR}")
            file(REMOVE_RECURSE "${${TARGET_NAME}_BINARY_DIR}")
        endif ()

        if (EXISTS "${${TARGET_NAME}_SUBBUILD_DIR}")
            message(STATUS "removing ${${TARGET_NAME}_SUBBUILD_DIR}")
            file(REMOVE_RECURSE "${${TARGET_NAME}_SUBBUILD_DIR}")
        endif ()
    else ()
        set(${TARGET_NAME}_SOURCE_DIR ${${TARGET_NAME}_SOURCE_DIR} PARENT_SCOPE)
        set(${TARGET_NAME}_BINARY_DIR ${${TARGET_NAME}_BINARY_DIR} PARENT_SCOPE)
    endif ()
endfunction()


fetch_and_patch("first")
set(CPR_ENABLE_SSL OFF)
if (NOT PATCH_RESULT EQUAL 0)
    message(WARNING "Failed to apply patch CURL. Clean cache and retrying...")
    fetch_and_patch("retry")
    if (NOT PATCH_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to apply patch CURL after re-downloading CPR source.")
    else ()
        add_subdirectory(${cpr_retry_SOURCE_DIR} ${cpr_retry_BINARY_DIR})
        message(STATUS "Patch of CURL applied successfully after re-downloading in ${cpr_retry_SOURCE_DIR}.")
    endif ()
else ()
    add_subdirectory(${cpr_first_SOURCE_DIR} ${cpr_first_BINARY_DIR})
    message(STATUS "Patch of CURL applied successfully on the first attempt in ${cpr_first_SOURCE_DIR}.")
endif ()
