include_guard (GLOBAL)

include (cmake/CheckSIMDCompilerFlag.cmake)
include (ExternalProject)

include (extern/json/json.cmake)
include (extern/antlr4/antlr4.cmake)
include (extern/boost/boost.cmake)

set (VSAG_BLAS_BACKEND "openblas")
if (VSAG_TARGET_PROCESSOR STREQUAL "x86_64" AND ENABLE_INTEL_MKL)
    set (VSAG_BLAS_BACKEND "mkl")
    include (extern/mkl/mkl.cmake)
else ()
    if (ENABLE_INTEL_MKL)
        message (WARNING
                 "Intel MKL is not supported on this architecture (${VSAG_TARGET_PROCESSOR}). "
                 "Falling back to OpenBLAS.")
    endif ()
    include (extern/openblas/openblas.cmake)
endif ()

include (extern/diskann/diskann.cmake)
include (extern/catch2/catch2.cmake)
include (extern/cpuinfo/cpuinfo.cmake)
include (extern/fmt/fmt.cmake)
include (extern/thread_pool/thread_pool.cmake)
include (extern/tsl/tsl.cmake)
include (extern/roaringbitmap/roaringbitmap.cmake)

if (ENABLE_TOOLS AND ENABLE_CXX11_ABI)
    include (extern/hdf5/hdf5.cmake)
    include (extern/argparse/argparse.cmake)
    include (extern/yaml-cpp/yaml-cpp.cmake)
    include (extern/tabulate/tabulate.cmake)
    include (extern/httplib/httplib.cmake)
endif ()
