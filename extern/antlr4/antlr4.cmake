set(name antlr4)
set(source_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/source)
set(binary_dir ${source_dir}/runtime/Cpp)
set(install_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/install)

ExternalProject_Add(
        ${name}
        URL http://aivolvo-dev.cn-hangzhou-alipay-b.oss-cdn.aliyun-inc.com/tbase/third-party/antlr4_4.13.2.tar.gz
        #https://github.com/antlr/antlr4/archive/refs/tags/4.13.2.tar.gz
        URL_HASH MD5=3b75610fc8a827119258cba09a068be5
        DOWNLOAD_NAME antlr4_4.13.2.tar.gz
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${name}
        TMP_DIR ${BUILD_INFO_DIR}
        STAMP_DIR ${BUILD_INFO_DIR}
        DOWNLOAD_DIR ${DOWNLOAD_DIR}
        SOURCE_DIR ${source_dir}
        BINARY_DIR ${binary_dir}
        BUILD_IN_SOURCE 0
        CONFIGURE_COMMAND
        cmake ${common_cmake_args} -DCMAKE_INSTALL_PREFIX=${install_dir} -DANTLR_BUILD_SHARED=ON -DANTLR_BUILD_STATIC=OFF -DWITH_DEMO=False -DANTLR_BUILD_CPP_TESTS=OFF -S . -B build
        BUILD_COMMAND
        cmake --build build --target install --parallel ${NUM_BUILDING_JOBS}
        INSTALL_COMMAND cmake --install build
        LOG_CONFIGURE TRUE
        LOG_BUILD TRUE
        LOG_INSTALL TRUE
        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30
)
# http://www.antlr.org/download/antlr-4.13.2-complete.jar
if (NOT EXISTS "${install_dir}/antlr-4.13.2-complete.jar")
    file(DOWNLOAD
            http://aivolvo-dev.cn-hangzhou-alipay-b.oss-cdn.aliyun-inc.com/tbase/third-party/antlr-4.13.2-complete.jar
            ${install_dir}/antlr-4.13.2-complete.jar
            TIMEOUT 60
            STATUS status
            TLS_VERIFY OFF
    )

    list(GET status 0 error_code)
    list(GET status 1 error_msg)

    if (error_code)
        message(FATAL_ERROR "Failed to download antlr-4.13.2-complete.jar: ${error_msg}")
    else ()
        message(STATUS "Downloaded antlr-4.13.2-complete.jar successfully")
    endif ()
endif ()

include_directories(${install_dir}/include/antlr4-runtime)
link_directories(${install_dir}/lib)
link_directories(${install_dir}/lib64)
