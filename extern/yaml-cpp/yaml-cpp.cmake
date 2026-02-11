include (FetchContent)

FetchContent_Declare (
        yaml-cpp
        URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.9.0.tar.gz
        # this url is maintained by the vsag project, if it's broken, please try
        #  the latest commit or contact the vsag project
        http://vsagcache.oss-rg-china-mainland.aliyuncs.com/yaml-cpp/0.9.0.tar.gz
        URL_HASH MD5=3be7b8b182ccd96e48989b4e57311193
        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30
)

FetchContent_MakeAvailable (yaml-cpp)
include_directories (${yaml-cpp_SOURCE_DIR}/include)
