Include(FetchContent)
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
