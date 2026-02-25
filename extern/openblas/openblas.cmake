
set(name openblas)
set(source_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/source)
set(install_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/install)

option(USE_SYSTEM_OPENBLAS "Use system-installed OpenBLAS instead of building from source" OFF)

set(OPENBLAS_FOUND FALSE)

if(USE_SYSTEM_OPENBLAS)
    # Try to find system-installed OpenBLAS
    find_library(OPENBLAS_LIB
        NAMES openblas
        PATHS
            /usr/lib
            /usr/lib64
            /usr/lib/x86_64-linux-gnu
            /usr/lib/aarch64-linux-gnu
            /usr/local/lib
            /usr/local/lib64
            /opt/homebrew/lib
        NO_DEFAULT_PATH
    )
    
    find_path(OPENBLAS_INCLUDE
        NAMES cblas.h
        PATHS
            /usr/include
            /usr/include/openblas
            /usr/include/x86_64-linux-gnu
            /usr/include/aarch64-linux-gnu
            /usr/local/include
            /usr/local/include/openblas
            /opt/homebrew/include
        NO_DEFAULT_PATH
    )
    
    find_path(LAPACKE_INCLUDE
        NAMES lapacke.h
        PATHS
            /usr/include
            /usr/include/openblas
            /usr/include/x86_64-linux-gnu
            /usr/include/aarch64-linux-gnu
            /usr/local/include
            /usr/local/include/openblas
            /opt/homebrew/include
        NO_DEFAULT_PATH
    )
    
    if(OPENBLAS_LIB AND OPENBLAS_INCLUDE AND LAPACKE_INCLUDE)
        set(OPENBLAS_FOUND TRUE)
        message(STATUS "Found system OpenBLAS library: ${OPENBLAS_LIB}")
        message(STATUS "Found OpenBLAS include directory: ${OPENBLAS_INCLUDE}")
        message(STATUS "Found LAPACKE include directory: ${LAPACKE_INCLUDE}")
        
        # Try to find LAPACKE library (separate from OpenBLAS on some systems)
        find_library(LAPACKE_LIB
            NAMES lapacke
            PATHS
                /usr/lib
                /usr/lib64
                /usr/lib/x86_64-linux-gnu
                /usr/lib/aarch64-linux-gnu
                /usr/local/lib
                /usr/local/lib64
                /opt/homebrew/lib
            NO_DEFAULT_PATH
        )
        
        if(LAPACKE_LIB)
            message(STATUS "Found LAPACKE library: ${LAPACKE_LIB}")
            set(OPENBLAS_LAPACKE_LIB ${LAPACKE_LIB})
        else()
            message(STATUS "LAPACKE library not found as separate library, assuming it's included in OpenBLAS")
            set(OPENBLAS_LAPACKE_LIB "")
        endif()
        
        # Set install_dir to a dummy value for compatibility
        set(install_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/system)
        
        # Add include directories
        include_directories(${OPENBLAS_INCLUDE})
        if(NOT "${OPENBLAS_INCLUDE}" STREQUAL "${LAPACKE_INCLUDE}")
            include_directories(${LAPACKE_INCLUDE})
        endif()
        
        # Create a dummy target for consistency with dependencies
        add_custom_target(${name})
    else()
        message(WARNING "System OpenBLAS not found (USE_SYSTEM_OPENBLAS=ON). Falling back to building from source.")
        message(STATUS "  OPENBLAS_LIB: ${OPENBLAS_LIB}")
        message(STATUS "  OPENBLAS_INCLUDE: ${OPENBLAS_INCLUDE}")
        message(STATUS "  LAPACKE_INCLUDE: ${LAPACKE_INCLUDE}")
    endif()
endif()

if(NOT OPENBLAS_FOUND)
    # Build OpenBLAS from source
    message(STATUS "Building OpenBLAS from source")
    
    ExternalProject_Add(
        ${name}
        URL https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.23/OpenBLAS-0.3.23.tar.gz
            # this url is maintained by the vsag project, if it's broken, please try
            #  the latest commit or contact the vsag project
            http://vsagcache.oss-rg-china-mainland.aliyuncs.com/openblas/OpenBLAS-0.3.23.tar.gz
        URL_HASH MD5=115634b39007de71eb7e75cf7591dfb2
        DOWNLOAD_NAME OpenBLAS-v0.3.23.tar.gz
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${name}
        TMP_DIR ${BUILD_INFO_DIR}
        STAMP_DIR ${BUILD_INFO_DIR}
        DOWNLOAD_DIR ${DOWNLOAD_DIR}
        SOURCE_DIR ${source_dir}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND
            ${common_configure_envs}
            OMP_NUM_THREADS=1
            PATH=/usr/lib/ccache:$ENV{PATH}
            LD_LIBRARY_PATH=/opt/alibaba-cloud-compiler/lib64/:$ENV{LD_LIBRARY_PATH}
            make USE_THREAD=0 USE_LOCKING=1 DYNAMIC_ARCH=1 -j${NUM_BUILDING_JOBS}
        INSTALL_COMMAND
            make DYNAMIC_ARCH=1 PREFIX=${install_dir} install
        BUILD_IN_SOURCE 1
        LOG_CONFIGURE TRUE
        LOG_BUILD TRUE
        LOG_INSTALL TRUE
        DOWNLOAD_NO_PROGRESS 1
        INACTIVITY_TIMEOUT 5
        TIMEOUT 30
    )
    
    include_directories(${install_dir}/include)
    link_directories (${install_dir}/lib)
    if (NOT APPLE)
        link_directories (${install_dir}/lib64)
    endif()
    
    file(GLOB LIB_DIR_EXIST CHECK_DIRECTORIES LIST_DIRECTORIES true ${install_dir}/lib)
    if(LIB_DIR_EXIST)
        file(GLOB LIB_FILES ${install_dir}/lib/lib*.a)
        foreach(lib_file ${LIB_FILES})
            install(FILES ${lib_file}
                    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        )
        endforeach()
    endif()
    
    if (NOT APPLE)
        file(GLOB LIB64_DIR_EXIST CHECK_DIRECTORIES LIST_DIRECTORIES true ${install_dir}/lib64)
        if(LIB64_DIR_EXIST)
            file(GLOB LIB64_FILES ${install_dir}/lib64/lib*.a)
            foreach(lib64_file ${LIB64_FILES})
                install(FILES ${lib64_file}
                        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            )
            endforeach()
        endif()
    endif()
endif()
