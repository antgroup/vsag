set (name antlr4)
set (source_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/source)
set (binary_dir ${source_dir}/runtime/Cpp)
set (install_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/install)
if (WIN32)
    set (antlr4_runtime_library ${install_dir}/lib/antlr4-runtime-static.lib)
else ()
    set (antlr4_runtime_library ${install_dir}/lib/libantlr4-runtime.a)
endif ()

if (WIN32)
    set (VSAG_ANTLR4_CXX11_ABI "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS -DWIN32_LEAN_AND_MEAN /FIchrono /EHsc /GR")
elseif (ENABLE_CXX11_ABI)
    set (VSAG_ANTLR4_CXX11_ABI "-D_GLIBCXX_USE_CXX11_ABI=1")
else ()
    set (VSAG_ANTLR4_CXX11_ABI "-D_GLIBCXX_USE_CXX11_ABI=0")
endif ()

set (FULL_CXX_FLAGS "${VSAG_THIRDPARTY_CXX_FLAGS} ${VSAG_ANTLR4_CXX11_ABI}")

set (antlr4_urls
    https://github.com/antlr/antlr4/archive/refs/tags/4.13.2.tar.gz
    # this url is maintained by the vsag project, if it's broken, please try
    #  the latest commit or contact the vsag project
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/antlr4/v4.13.2.tar.gz
)
if (DEFINED ENV{VSAG_THIRDPARTY_ANTLR4})
    message (STATUS "Using local path for antlr4: $ENV{VSAG_THIRDPARTY_ANTLR4}")
    list (PREPEND antlr4_urls "$ENV{VSAG_THIRDPARTY_ANTLR4}")
endif ()

ExternalProject_Add (
        ${name}
        URL ${antlr4_urls}
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
        cmake ${common_cmake_args} -DCMAKE_CXX_FLAGS=${FULL_CXX_FLAGS} -DCMAKE_INSTALL_PREFIX=${install_dir} -DANTLR_BUILD_SHARED=OFF -DANTLR_BUILD_STATIC=ON -DWITH_DEMO=False -DANTLR_BUILD_CPP_TESTS=OFF -DWITH_STATIC_CRT=OFF -S . -B build
        BUILD_COMMAND
        cmake --build build --target install --parallel ${NUM_BUILDING_JOBS}
        INSTALL_COMMAND cmake --install build
        LOG_CONFIGURE TRUE
        LOG_BUILD TRUE
        LOG_INSTALL TRUE
        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30

        BUILD_BYPRODUCTS
        ${antlr4_runtime_library}
)

add_library (vsag_antlr4_runtime_headers INTERFACE)
target_include_directories (vsag_antlr4_runtime_headers INTERFACE ${install_dir}/include/antlr4-runtime)

if (NOT TARGET antlr4-runtime)
    add_library (antlr4-runtime STATIC IMPORTED GLOBAL)
    set_target_properties (antlr4-runtime PROPERTIES IMPORTED_LOCATION ${antlr4_runtime_library})
endif ()
add_dependencies (antlr4-runtime antlr4)

add_library (vsag_antlr4_autogen_headers INTERFACE)
target_include_directories (vsag_antlr4_autogen_headers INTERFACE
    ${install_dir}/include
    ${install_dir}/include/antlr4-autogen)

file (GLOB ANTLR4_GEN_SRC "extern/antlr4/fc/*.cpp")
add_library (antlr4-autogen STATIC ${ANTLR4_GEN_SRC})
add_dependencies (antlr4-autogen antlr4)
target_link_libraries (antlr4-autogen PUBLIC vsag_antlr4_runtime_headers vsag_antlr4_autogen_headers)
target_include_directories (antlr4-autogen PRIVATE extern/antlr4/fc)
target_compile_options (antlr4-autogen PRIVATE ${VSAG_ANTLR4_CXX11_ABI})
if (WIN32)
    target_compile_definitions (antlr4-autogen PUBLIC ANTLR4CPP_STATIC)
endif ()
set_property (TARGET antlr4-autogen PROPERTY CXX_STANDARD 17)
target_link_libraries (antlr4-autogen PRIVATE antlr4-runtime)

file (COPY extern/antlr4/fc/
      DESTINATION ${install_dir}/include/antlr4-autogen
      FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.hpp")
