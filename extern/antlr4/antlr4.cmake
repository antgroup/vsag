set(name antlr4)
set(source_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/source)
set(binary_dir ${source_dir}/runtime/Cpp)
set(install_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/install)

ExternalProject_Add(
        ${name}
        URL https://github.com/antlr/antlr4/archive/refs/tags/4.13.2.tar.gz
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
        TIMEOUT 60
)

include_directories(${install_dir}/include/antlr4-runtime)
link_directories(${install_dir}/lib)
link_directories(${install_dir}/lib64)

include_directories(extern/antlr4/fc)
file(GLOB ANTLR4_GEN_SRC "extern/antlr4/fc/*.cpp")
add_library(antlr4-autogen SHARED ${ANTLR4_GEN_SRC})
add_dependencies(antlr4-autogen antlr4)
target_compile_options(antlr4-autogen PRIVATE ${common_cmake_args})
set_property(TARGET antlr4-autogen PROPERTY CXX_STANDARD 17)
target_link_libraries(antlr4-autogen antlr4-runtime)
