
option(MKL_STATIC_LINK "Set to ON to link Intel MKL statically." OFF)

if (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64" AND ENABLE_INTEL_MKL)

 
    if(MKL_STATIC_LINK)
        message(STATUS "Configuring Intel MKL with STATIC linking.")

        
        find_path(MKL_PATH
            NAMES libmkl_core.a
            HINTS
                "/opt/intel/oneapi/mkl/latest/lib/intel64"
                "/usr/lib/x86_64-linux-gnu"
                "/opt/intel/mkl/lib/intel64"
        )

       
        find_path(OMP_PATH
            NAMES libiomp5.a
            HINTS
                "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin"
                "/opt/intel/lib/intel64_lin"
                "/opt/intel/compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin"
                "/usr/local/lib"
                "/usr/lib/x86_64-linux-gnu"
        )

        
        find_path(MKL_INCLUDE_PATH
            NAMES mkl.h
            HINTS
                "/opt/intel/oneapi/mkl/latest/include"
                "/usr/include/mkl"
                "/opt/intel/mkl/include"
        )

        if(NOT MKL_PATH OR NOT OMP_PATH OR NOT MKL_INCLUDE_PATH)
            message(FATAL_ERROR "Could not find Intel MKL (static) or OpenMP libraries/headers. "
                                "Use -DMKL_PATH, -DOMP_PATH, and -DMKL_INCLUDE_PATH to specify their locations.")
        else()
            message(STATUS "Found MKL static libraries in: ${MKL_PATH}")
            message(STATUS "Found MKL include path: ${MKL_INCLUDE_PATH}")
            message(STATUS "Found OpenMP static library in: ${OMP_PATH}")
        endif()

       
        include_directories(${MKL_INCLUDE_PATH})
        link_directories(${MKL_PATH} ${OMP_PATH})

        
        set(BLAS_LIBRARIES
            "-Wl,--start-group"
            "${MKL_PATH}/libmkl_intel_lp64.a"
            "${MKL_PATH}/libmkl_intel_thread.a"
            "${MKL_PATH}/libmkl_core.a"
            "-Wl,--end-group"
            "${OMP_PATH}/libiomp5.a"
            "pthread"
            "m"
            "dl"
        )
        message(STATUS "Enabled Intel MKL as BLAS backend (STATIC linking).")

 
    else()
        message(STATUS "Configuring Intel MKL with DYNAMIC linking (default).")

        
        find_path(MKL_PATH
            NAMES libmkl_core.so
            HINTS
                "/opt/intel/oneapi/mkl/latest/lib"
                "/opt/intel/oneapi/mkl/latest/lib/intel64"
                "/usr/lib/x86_64-linux-gnu"
                "/opt/intel/mkl/lib/intel64"
        )

      
        find_path(OMP_PATH
            NAMES libiomp5.so
            HINTS
                "/opt/intel/oneapi/compiler/2024.2/lib"
                "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin"
                "/usr/lib/x86_64-linux-gnu"
                "/opt/intel/lib/intel64_lin"
        )

       
        find_path(MKL_INCLUDE_PATH
            NAMES mkl.h
            HINTS
                "/opt/intel/oneapi/mkl/latest/include"
                "/usr/include/mkl"
                "/opt/intel/mkl/include"
        )

        if(NOT MKL_PATH OR NOT OMP_PATH OR NOT MKL_INCLUDE_PATH)
            message(FATAL_ERROR "Could not find Intel MKL (dynamic) or OpenMP libraries/headers. "
                                "Please check your MKL installation or disable ENABLE_INTEL_MKL.")
        else()
            message(STATUS "Found MKL dynamic libraries in: ${MKL_PATH}")
            message(STATUS "Found MKL include path: ${MKL_INCLUDE_PATH}")
            message(STATUS "Found OpenMP dynamic library in: ${OMP_PATH}")
        endif()

       
        include_directories(${MKL_INCLUDE_PATH})
        link_directories(${MKL_PATH} ${OMP_PATH})

       
        set(BLAS_LIBRARIES
            "${MKL_PATH}/libmkl_intel_lp64.so"
            "${MKL_PATH}/libmkl_intel_thread.so"
            "${MKL_PATH}/libmkl_core.so"
            "${OMP_PATH}/libiomp5.so"
        )

      
        foreach(mkllib ${BLAS_LIBRARIES})
            if(EXISTS ${mkllib})
                install(FILES ${mkllib} DESTINATION ${CMAKE_INSTALL_LIBDIR})
            endif()
        endforeach()
        message(STATUS "Enabled Intel MKL as BLAS backend (DYNAMIC linking).")
    endif()


    foreach(mkllib ${BLAS_LIBRARIES})
        install(FILES ${mkllib} DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endforeach()
    message (STATUS "enable ${Yellow}intel-mkl${CR} as blas backend")

else()
    if(ENABLE_INTEL_MKL)
        message(WARNING "Intel MKL is not supported on this architecture (${CMAKE_HOST_SYSTEM_PROCESSOR}). Falling back to OpenBLAS.")
    endif()

    # Check if we're using system-installed OpenBLAS
    if(USE_SYSTEM_OPENBLAS AND OPENBLAS_FOUND)
        # Use the system OpenBLAS library
        set(BLAS_LIBRARIES ${OPENBLAS_LIB})
        
        # Add LAPACKE library if found separately
        if(DEFINED OPENBLAS_LAPACKE_LIB AND OPENBLAS_LAPACKE_LIB)
            list(APPEND BLAS_LIBRARIES ${OPENBLAS_LAPACKE_LIB})
        endif()
        
        # Add gfortran dependency
        if (APPLE AND DEFINED GFORTRAN_LIB AND EXISTS "${GFORTRAN_LIB}")
            list(APPEND BLAS_LIBRARIES "${GFORTRAN_LIB}")
        else()
            list(APPEND BLAS_LIBRARIES gfortran)
        endif()
        
        # Add OpenMP library
        if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            list(PREPEND BLAS_LIBRARIES omp)
        else()
            list(PREPEND BLAS_LIBRARIES gomp)
        endif()
        
        message(STATUS "Using system OpenBLAS as BLAS backend: ${OPENBLAS_LIB}")
    else()
        # Use bundled/built OpenBLAS
        # OpenBLAS on macOS typically requires libgfortran at link time, but `-lgfortran` often fails
        # because libgfortran is not in the default linker search paths. If DarwinDep.cmake already
        # detected the full path to libgfortran.dylib, prefer that; otherwise fall back to `gfortran`.
        if (APPLE AND DEFINED GFORTRAN_LIB AND EXISTS "${GFORTRAN_LIB}")
            set(BLAS_LIBRARIES libopenblas.a "${GFORTRAN_LIB}")
        else()
            set(BLAS_LIBRARIES libopenblas.a gfortran)
        endif()
        if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            list(PREPEND BLAS_LIBRARIES omp)
        else()
            list(PREPEND BLAS_LIBRARIES gomp)
        endif()
        message ("enable openblas as blas backend")
    endif()
endif()

set(BLAS_LIBRARIES "${BLAS_LIBRARIES}" CACHE STRING "Final list of BLAS libraries to link against.")
