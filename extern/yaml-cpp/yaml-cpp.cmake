include (FetchContent)

vsag_get_system_dep_policy (YAML_CPP _vsag_dep_policy)
if (TARGET yaml-cpp::yaml-cpp OR TARGET yaml-cpp)
    if (NOT TARGET yaml-cpp AND TARGET yaml-cpp::yaml-cpp)
        add_library (yaml-cpp ALIAS yaml-cpp::yaml-cpp)
    endif ()
    vsag_note_system_dep (yaml-cpp yaml-cpp)
    return ()
elseif (NOT _vsag_dep_policy STREQUAL "OFF")
    find_package (yaml-cpp CONFIG QUIET)
    if (TARGET yaml-cpp::yaml-cpp OR TARGET yaml-cpp)
        if (NOT TARGET yaml-cpp AND TARGET yaml-cpp::yaml-cpp)
            add_library (yaml-cpp ALIAS yaml-cpp::yaml-cpp)
        endif ()
        vsag_note_system_dep (yaml-cpp yaml-cpp)
        return ()
    elseif (_vsag_dep_policy STREQUAL "ON")
        vsag_fail_missing_system_dep (YAML_CPP yaml-cpp "yaml-cpp::yaml-cpp, yaml-cpp")
    endif ()
endif ()

set (YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Disable yaml-cpp contrib targets" FORCE)
set (YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable yaml-cpp utility targets" FORCE)
set (YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests" FORCE)

set (yaml_cpp_urls
    https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.9.0.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/yaml-cpp/yaml-cpp-0.9.0.tar.gz
)
if (DEFINED ENV{VSAG_THIRDPARTY_YAML_CPP})
    message (STATUS "Using local path for yaml-cpp: $ENV{VSAG_THIRDPARTY_YAML_CPP}")
    list (PREPEND yaml_cpp_urls "$ENV{VSAG_THIRDPARTY_YAML_CPP}")
endif ()
FetchContent_Declare (
    yaml-cpp
    URL ${yaml_cpp_urls}
    URL_HASH MD5=7d17de1b2a4b1d2776181f67c940bcdf
    DOWNLOAD_NO_PROGRESS 1
    INACTIVITY_TIMEOUT 5
    TIMEOUT 30)

FetchContent_MakeAvailable (yaml-cpp)
