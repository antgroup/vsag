Include (FetchContent)
set(cpuinfo_urls
    https://github.com/pytorch/cpuinfo/archive/ca678952a9a8eaa6de112d154e8e104b22f9ab3f.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/cpuinfo/ca678952a9a8eaa6de112d154e8e104b22f9ab3f.tar.gz
)
if(DEFINED ENV{VSAG_THIRDPARTY_CPUINFO})
  message(STATUS "Using local path for cpuinfo: $ENV{VSAG_THIRDPARTY_CPUINFO}")
  list(PREPEND cpuinfo_urls "$ENV{VSAG_THIRDPARTY_CPUINFO}")
endif()
FetchContent_Declare (
  cpuinfo
  URL      ${cpuinfo_urls}
  URL_HASH MD5=a72699bc703dfea4ab2c9c01025e46e9
  DOWNLOAD_NO_PROGRESS 1
  INACTIVITY_TIMEOUT 5
  TIMEOUT 30
)

set (CPUINFO_BUILD_TOOLS OFF CACHE BOOL "Disable some option in the library" FORCE)
set (CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "Disable some option in the library" FORCE)
set (CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "Disable some option in the library" FORCE)
set (CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "Disable some option in the library" FORCE)
set (CPUINFO_BUILD_PKG_CONFIG OFF CACHE BOOL "Disable some option in the library" FORCE)
set (CPUINFO_LIBRARY_TYPE "static")

# exclude cpuinfo in vsag installation
FetchContent_GetProperties (cpuinfo)
if (NOT cpuinfo_POPULATED)
  FetchContent_Populate (cpuinfo)
  add_subdirectory (${cpuinfo_SOURCE_DIR} ${cpuinfo_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

include_directories (${cpuinfo_SOURCE_DIR}/include)

install (
  TARGETS cpuinfo
  ARCHIVE DESTINATION lib
)
