include (FetchContent)

set(yaml_cpp_urls
    https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/yaml-cpp/0.8.0.tar.gz
)
if(NOT "$ENV{VSAG_THIRDPARTY_YAML_CPP}" STREQUAL "")
  message(STATUS "Using override URL/archive for yaml-cpp: $ENV{VSAG_THIRDPARTY_YAML_CPP}")
  list(PREPEND yaml_cpp_urls "$ENV{VSAG_THIRDPARTY_YAML_CPP}")
endif()
FetchContent_Declare (
        yaml-cpp
        URL ${yaml_cpp_urls}
        URL_HASH MD5=1d2c7975edba60e995abe3c4af6480e5
        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30
)

FetchContent_MakeAvailable (yaml-cpp)
include_directories (${yaml-cpp_SOURCE_DIR}/include)
