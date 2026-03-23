
option(MKL_STATIC_LINK "Set to ON to link Intel MKL statically." OFF)

if(NOT TARGET mkl)
    add_custom_target(mkl)
endif()

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
    set(MKL_INSTALL_LIBS
        "${MKL_PATH}/libmkl_intel_lp64.a"
        "${MKL_PATH}/libmkl_intel_thread.a"
        "${MKL_PATH}/libmkl_core.a"
        "${OMP_PATH}/libiomp5.a"
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
        "${MKL_PATH}/libmkl_rt.so"
        "${OMP_PATH}/libiomp5.so"
    )
    set(MKL_INSTALL_LIBS ${BLAS_LIBRARIES})

    foreach(mkllib ${MKL_INSTALL_LIBS})
        if(EXISTS ${mkllib})
            install(FILES ${mkllib} DESTINATION ${CMAKE_INSTALL_LIBDIR})
        endif()
    endforeach()
    message(STATUS "Enabled Intel MKL as BLAS backend (DYNAMIC linking).")
endif()

foreach(mkllib ${MKL_INSTALL_LIBS})
    install(FILES ${mkllib} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endforeach()
message(STATUS "enable ${Yellow}intel-mkl${CR} as blas backend")

set(BLAS_LIBRARIES "${BLAS_LIBRARIES}" CACHE STRING "Final list of BLAS libraries to link against." FORCE)
