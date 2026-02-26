include(FetchContent)

set(pybind11_urls
    https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/pybind11/v2.11.1.tar.gz
)
if(DEFINED ENV{VSAG_THIRDPARTY_PYBIND11})
  message(STATUS "Using local path for pybind11: $ENV{VSAG_THIRDPARTY_PYBIND11}")
  list(PREPEND pybind11_urls "$ENV{VSAG_THIRDPARTY_PYBIND11}")
endif()
FetchContent_Declare(
        pybind11
        URL ${pybind11_urls}
        URL_HASH MD5=49e92f92244021912a56935918c927d0
        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30
)

FetchContent_MakeAvailable(pybind11)
