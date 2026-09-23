
set(name openblas)
set(source_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/source)
set(install_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/install)

set(openblas_urls
    https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.34/OpenBLAS-0.3.34.tar.gz
    https://sourceforge.net/projects/openblas/files/v0.3.34/OpenBLAS-0.3.34.tar.gz/download
)
vsag_resolve_thirdparty_override (OPENBLAS v0.3.34 openblas_urls)

# Keep the existing environment, but build OpenBLAS CPU detection tools with strict FP.
set(openblas_configure_envs)
foreach(entry IN LISTS common_configure_envs)
    if(entry MATCHES "^(CFLAGS|CXXFLAGS)=")
        string(REPLACE "-Ofast" "-O2" entry "${entry}")
        string(REPLACE "-ffast-math" "" entry "${entry}")
        # The translated C LAPACK sources require pre-C23 empty parameter lists.
        if(entry MATCHES "^CFLAGS=" AND CMAKE_C_COMPILER_ID STREQUAL "GNU"
           AND CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
            string(APPEND entry " -std=gnu17")
        endif()
    endif()
    list(APPEND openblas_configure_envs "${entry}")
endforeach()

ExternalProject_Add(
    ${name}
    URL ${openblas_urls}
    URL_HASH SHA256=cd7e129868320cc2d033afa920e31202dfe0b8066a5b66661900ccc0f197dfed
    DOWNLOAD_NAME OpenBLAS-v0.3.34.tar.gz
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${name}
    TMP_DIR ${BUILD_INFO_DIR}
    STAMP_DIR ${BUILD_INFO_DIR}
    DOWNLOAD_DIR ${DOWNLOAD_DIR}
    SOURCE_DIR ${source_dir}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND
        ${openblas_configure_envs}
        OMP_NUM_THREADS=1
        PATH=/usr/lib/ccache:$ENV{PATH}
        LD_LIBRARY_PATH=/opt/alibaba-cloud-compiler/lib64/:$ENV{LD_LIBRARY_PATH}
        make USE_THREAD=0 USE_LOCKING=1 DYNAMIC_ARCH=1 NOFORTRAN=1
             MAKE_NB_JOBS=${NUM_BUILDING_JOBS} -j${NUM_BUILDING_JOBS}
    INSTALL_COMMAND
        make DYNAMIC_ARCH=1 NOFORTRAN=1 PREFIX=${install_dir} install
    BUILD_BYPRODUCTS ${install_dir}/lib/libopenblas.a
    BUILD_IN_SOURCE 1
    LOG_CONFIGURE TRUE
    LOG_BUILD TRUE
    LOG_INSTALL TRUE
    LOG_OUTPUT_ON_FAILURE TRUE
    DOWNLOAD_NO_PROGRESS 1
    INACTIVITY_TIMEOUT 5
    TIMEOUT 30
)

include_directories(${install_dir}/include)
link_directories (${install_dir}/lib)
link_directories (${install_dir}/lib64)

file(GLOB LIB_DIR_EXIST CHECK_DIRECTORIES LIST_DIRECTORIES true ${install_dir}/lib)
if(LIB_DIR_EXIST)
    file(GLOB LIB_FILES ${install_dir}/lib/lib*.a)
    foreach(lib_file ${LIB_FILES})
        install(FILES ${lib_file}
                DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    )
    endforeach()
endif()

file(GLOB LIB64_DIR_EXIST CHECK_DIRECTORIES LIST_DIRECTORIES true ${install_dir}/lib64)
if(LIB64_DIR_EXIST)
    file(GLOB LIB64_FILES ${install_dir}/lib64/lib*.a)
    foreach(lib64_file ${LIB64_FILES})
        install(FILES ${lib64_file}
                DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    )
    endforeach()
endif()
