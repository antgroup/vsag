include (FetchContent)

set (YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Disable yaml-cpp contrib targets" FORCE)
set (YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable yaml-cpp utility targets" FORCE)
set (YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests" FORCE)

set (yaml_cpp_urls
    https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.9.0.tar.gz
)
vsag_resolve_thirdparty_override (YAML_CPP yaml-cpp-0.9.0 yaml_cpp_urls)
FetchContent_Declare (
    yaml-cpp
    URL ${yaml_cpp_urls}
    URL_HASH MD5=7d17de1b2a4b1d2776181f67c940bcdf
    DOWNLOAD_NO_PROGRESS 1
    INACTIVITY_TIMEOUT 5
    TIMEOUT 30)

FetchContent_MakeAvailable (yaml-cpp)
