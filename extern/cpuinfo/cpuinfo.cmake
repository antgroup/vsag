Include (FetchContent)
set(cpuinfo_urls
    https://github.com/pytorch/cpuinfo/archive/ca678952a9a8eaa6de112d154e8e104b22f9ab3f.tar.gz
)
vsag_resolve_thirdparty_override (
    CPUINFO ca678952a9a8eaa6de112d154e8e104b22f9ab3f cpuinfo_urls)
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
