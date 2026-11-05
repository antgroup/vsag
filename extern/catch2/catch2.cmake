Include(FetchContent)
set(catch2_urls
    https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
)
vsag_resolve_thirdparty_override (CATCH2 v3.7.1 catch2_urls)
FetchContent_Declare(
  Catch2
  URL      ${catch2_urls}
  URL_HASH MD5=9fcbec1dc95edcb31c6a0d6c5320e098
  DOWNLOAD_NO_PROGRESS 1
  INACTIVITY_TIMEOUT 5
  TIMEOUT 30
)

FetchContent_MakeAvailable(Catch2)
