include (FetchContent)

FetchContent_Declare (
        yaml-cpp
        URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.9.0.tar.gz
        # this url is maintained by the vsag project, if it's broken, please try
        #  the latest commit or contact the vsag project
        http://vsagcache.oss-rg-china-mainland.aliyuncs.com/yaml-cpp/yaml-cpp-0.9.0.tar.gz
        URL_HASH MD5=7d17de1b2a4b1d2776181f67c940bcdf
        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30
)

FetchContent_MakeAvailable (yaml-cpp)
include_directories (${yaml-cpp_SOURCE_DIR}/include)
