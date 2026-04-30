
include (ExternalProject)

vsag_get_system_dep_policy (HDF5 _vsag_dep_policy)
if (TARGET HDF5::HDF5 OR TARGET hdf5::hdf5_cpp OR TARGET hdf5_cpp)
    add_library (vsag_hdf5_headers INTERFACE)
    if (TARGET HDF5::HDF5)
        target_link_libraries (vsag_hdf5_headers INTERFACE HDF5::HDF5)
        set (HDF5_CPP_STATIC_LIBRARY HDF5::HDF5)
        set (HDF5_C_STATIC_LIBRARY "")
    elseif (TARGET hdf5::hdf5_cpp)
        target_link_libraries (vsag_hdf5_headers INTERFACE hdf5::hdf5_cpp)
        set (HDF5_CPP_STATIC_LIBRARY hdf5::hdf5_cpp)
        set (HDF5_C_STATIC_LIBRARY "")
    else ()
        target_link_libraries (vsag_hdf5_headers INTERFACE hdf5_cpp)
        set (HDF5_CPP_STATIC_LIBRARY hdf5_cpp)
        set (HDF5_C_STATIC_LIBRARY "")
    endif ()
    if (NOT TARGET hdf5)
        add_custom_target (hdf5)
    endif ()
    vsag_note_system_dep (hdf5 vsag_hdf5_headers)
    return ()
elseif (NOT _vsag_dep_policy STREQUAL "OFF")
    find_package (HDF5 QUIET COMPONENTS C CXX)
    if (TARGET HDF5::HDF5 OR TARGET hdf5::hdf5_cpp OR TARGET hdf5_cpp OR HDF5_FOUND)
        add_library (vsag_hdf5_headers INTERFACE)
        if (TARGET HDF5::HDF5)
            target_link_libraries (vsag_hdf5_headers INTERFACE HDF5::HDF5)
            set (HDF5_CPP_STATIC_LIBRARY HDF5::HDF5)
            set (HDF5_C_STATIC_LIBRARY "")
        elseif (TARGET hdf5::hdf5_cpp)
            target_link_libraries (vsag_hdf5_headers INTERFACE hdf5::hdf5_cpp)
            set (HDF5_CPP_STATIC_LIBRARY hdf5::hdf5_cpp)
            set (HDF5_C_STATIC_LIBRARY "")
        elseif (TARGET hdf5_cpp)
            target_link_libraries (vsag_hdf5_headers INTERFACE hdf5_cpp)
            set (HDF5_CPP_STATIC_LIBRARY hdf5_cpp)
            set (HDF5_C_STATIC_LIBRARY "")
        else ()
            target_include_directories (vsag_hdf5_headers INTERFACE ${HDF5_INCLUDE_DIRS})
            set (HDF5_CPP_STATIC_LIBRARY ${HDF5_CXX_LIBRARIES})
            set (HDF5_C_STATIC_LIBRARY ${HDF5_C_LIBRARIES})
        endif ()
        if (NOT TARGET hdf5)
            add_custom_target (hdf5)
        endif ()
        vsag_note_system_dep (hdf5 vsag_hdf5_headers)
        return ()
    elseif (_vsag_dep_policy STREQUAL "ON")
        vsag_fail_missing_system_dep (HDF5 HDF5 "HDF5::HDF5, hdf5::hdf5_cpp, hdf5_cpp")
    endif ()
endif ()

set (name hdf5)
set (source_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/source)
set (install_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/install)
set (HDF5_CPP_STATIC_LIBRARY ${install_dir}/lib/libhdf5_cpp.a)
set (HDF5_C_STATIC_LIBRARY ${install_dir}/lib/libhdf5.a)

set (hdf5_urls
    https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_1.14.4.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/hdf5/hdf5_1.14.4.tar.gz
)
if (DEFINED ENV{VSAG_THIRDPARTY_HDF5})
    message (STATUS "Using local path for hdf5: $ENV{VSAG_THIRDPARTY_HDF5}")
    list (PREPEND hdf5_urls "$ENV{VSAG_THIRDPARTY_HDF5}")
endif ()

ExternalProject_Add (
    ${name}
    URL ${hdf5_urls}
    URL_HASH MD5=fdea52afcce07ed6c3e2a36e7fa11f21
    DOWNLOAD_NAME hdf5_1.14.4.tar.gz
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${name}
    TMP_DIR ${BUILD_INFO_DIR}
    STAMP_DIR ${BUILD_INFO_DIR}
    DOWNLOAD_DIR ${DOWNLOAD_DIR}
    SOURCE_DIR ${source_dir}
    CONFIGURE_COMMAND
        cmake ${common_cmake_args} -DHDF5_ENABLE_NONSTANDARD_FEATURE_FLOAT16=OFF -DCMAKE_INSTALL_PREFIX=${install_dir} -DHDF5_BUILD_CPP_LIB=ON -S. -Bbuild
    BUILD_COMMAND
        cmake --build build --target install --parallel ${NUM_BUILDING_JOBS}
    INSTALL_COMMAND
        ""
    BUILD_IN_SOURCE 1
    LOG_CONFIGURE TRUE
    LOG_BUILD TRUE
    LOG_INSTALL TRUE
    DOWNLOAD_NO_PROGRESS 1
    INACTIVITY_TIMEOUT 5
    TIMEOUT 30

    BUILD_BYPRODUCTS
        ${HDF5_CPP_STATIC_LIBRARY}
        ${HDF5_C_STATIC_LIBRARY}
)

add_library (vsag_hdf5_headers INTERFACE)
target_include_directories (vsag_hdf5_headers INTERFACE ${install_dir}/include)
