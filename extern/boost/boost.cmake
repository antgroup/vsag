# Copyright (c) 2022 Ant Group. All rights reserved.

cmake_policy (SET CMP0114 NEW)

vsag_get_system_dep_policy (BOOST _vsag_dep_policy)
if (TARGET Boost::headers OR TARGET Boost::boost)
    if (NOT TARGET boost)
        add_custom_target (boost)
    endif ()
    message (STATUS "Using external boost headers")
    return ()
elseif (NOT _vsag_dep_policy STREQUAL "OFF")
    find_package (Boost QUIET)
    if (TARGET Boost::headers OR TARGET Boost::boost OR Boost_INCLUDE_DIRS)
        if (NOT TARGET boost)
            add_custom_target (boost)
        endif ()
        message (STATUS "Using external boost headers")
        return ()
    elseif (_vsag_dep_policy STREQUAL "ON")
        vsag_fail_missing_system_dep (BOOST Boost "Boost::headers, Boost::boost")
    endif ()
endif ()

set (name boost)
set (source_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/source)
set (install_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/install)
if (APPLE)
    set (BOOST_TOOLSET "clang")
else ()
    set (BOOST_TOOLSET "gcc")
endif ()
get_filename_component (compiler_path ${CMAKE_CXX_COMPILER} DIRECTORY)

set (boost_urls
    https://archives.boost.io/release/1.67.0/source/boost_1_67_0.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/boost/boost_1_67_0.tar.gz
)
if (DEFINED ENV{VSAG_THIRDPARTY_BOOST})
    message (STATUS "Using local path for boost: $ENV{VSAG_THIRDPARTY_BOOST}")
    list (PREPEND boost_urls "$ENV{VSAG_THIRDPARTY_BOOST}")
endif ()

ExternalProject_Add (
    ${name}
    URL ${boost_urls}
    URL_HASH SHA256=8aa4e330c870ef50a896634c931adf468b21f8a69b77007e45c444151229f665
    DOWNLOAD_NAME boost_1_67_0.tar.gz
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${name}
    TMP_DIR ${BUILD_INFO_DIR}
    STAMP_DIR ${BUILD_INFO_DIR}
    DOWNLOAD_DIR ${DOWNLOAD_DIR}
    SOURCE_DIR ${source_dir}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND
        ${CMAKE_COMMAND} -E copy_directory boost ${install_dir}/include/boost
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    LOG_CONFIGURE TRUE
    LOG_BUILD TRUE
    LOG_INSTALL TRUE
    DOWNLOAD_NO_PROGRESS 0
    INACTIVITY_TIMEOUT 5
    # filesize ~= 90MiB
    TIMEOUT 90
)

ExternalProject_Add_Step (${name} clean
    EXCLUDE_FROM_MAIN TRUE
    ALWAYS TRUE
    DEPENDEES configure
    COMMAND ./b2 clean
    COMMAND rm -f ${BUILD_INFO_DIR}/${name}-build
    WORKING_DIRECTORY ${source_dir}
)

ExternalProject_Add_StepTargets (${name} clean)
