
include (FetchContent)
FetchContent_Declare (
  cpr
  URL https://github.com/libcpr/cpr/archive/refs/tags/1.11.2.tar.gz
      https://vsagcache.oss-rg-china-mainland.aliyuncs.com/cpr/1.11.2.tar.gz
  URL_HASH MD5=639cff98d5124cf06923a0975fb427d8
  DOWNLOAD_NO_PROGRESS 1
  INACTIVITY_TIMEOUT 5
  TIMEOUT 30
  )

FetchContent_GetProperties(cpr)
if(NOT cpr_POPULATED)
  FetchContent_Populate(cpr)
endif()

set(PATCH_FILE "${CMAKE_SOURCE_DIR}/extern/cpr/fix_curl.patch")

find_package(Git REQUIRED)
execute_process(
        COMMAND ${GIT_EXECUTABLE} apply --ignore-whitespace --whitespace=fix ${PATCH_FILE}
        WORKING_DIRECTORY ${cpr_SOURCE_DIR}
        RESULT_VARIABLE PATCH_RESULT
)

if(NOT PATCH_RESULT EQUAL 0)
  message(WARNING "Failed to apply CURL modification patch to CPR CMakeLists.txt")
else()
  message(STATUS "CURL Modification Patch applied successfully")
endif()

set (CPR_USE_SYSTEM_CURL OFF)
set (CPR_ENABLE_SSL OFF)

add_subdirectory(${cpr_SOURCE_DIR} ${cpr_BINARY_DIR})
