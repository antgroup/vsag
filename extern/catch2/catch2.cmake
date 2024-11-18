Include(FetchContent)
FetchContent_Declare(
  Catch2
  URL      https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
            # this url is maintained by the vsag project, if it's broken, please try
            #  the latest commit or contact the vsag project
           http://vsagcache.oss-rg-china-mainland.aliyuncs.com/catch2/v3.7.1.tar.gz
  URL_HASH MD5=9fcbec1dc95edcb31c6a0d6c5320e098
  DOWNLOAD_NO_PROGRESS 1
  INACTIVITY_TIMEOUT 5
  TIMEOUT 30
)

FetchContent_MakeAvailable(Catch2)
