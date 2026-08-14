include (FetchContent)

set(yaml_cpp_urls
    https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
)
vsag_resolve_thirdparty_override (YAML_CPP 0.8.0 yaml_cpp_urls)
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
