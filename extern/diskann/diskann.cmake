set (DISKANN_SOURCES
    extern/diskann/DiskANN/src/abstract_data_store.cpp
    extern/diskann/DiskANN/src/ann_exception.cpp
    extern/diskann/DiskANN/src/disk_utils.cpp
    extern/diskann/DiskANN/src/distance.cpp
    extern/diskann/DiskANN/src/index.cpp
    extern/diskann/DiskANN/src/in_mem_graph_store.cpp
    extern/diskann/DiskANN/src/in_mem_data_store.cpp
    extern/diskann/DiskANN/src/local_file_reader.cpp
    extern/diskann/DiskANN/src/math_utils.cpp
    extern/diskann/DiskANN/src/natural_number_map.cpp
    extern/diskann/DiskANN/src/natural_number_set.cpp
    extern/diskann/DiskANN/src/memory_mapper.cpp
    extern/diskann/DiskANN/src/partition.cpp
    extern/diskann/DiskANN/src/pq.cpp
    extern/diskann/DiskANN/src/pq_flash_index.cpp
    extern/diskann/DiskANN/src/scratch.cpp
    extern/diskann/DiskANN/src/logger.cpp
    extern/diskann/DiskANN/src/utils.cpp
    extern/diskann/DiskANN/src/filter_utils.cpp
    extern/diskann/DiskANN/src/index_factory.cpp
    extern/diskann/DiskANN/src/abstract_index.cpp
)

if (NOT TARGET vsag_diskann_headers)
    add_library (vsag_diskann_headers INTERFACE)
endif ()
target_include_directories (vsag_diskann_headers INTERFACE
    extern/diskann
    extern/diskann/DiskANN/include)
if (TARGET Boost::headers)
    target_link_libraries (vsag_diskann_headers INTERFACE Boost::headers)
elseif (TARGET Boost::boost)
    target_link_libraries (vsag_diskann_headers INTERFACE Boost::boost)
elseif (Boost_INCLUDE_DIRS)
    target_include_directories (vsag_diskann_headers INTERFACE ${Boost_INCLUDE_DIRS})
else ()
    target_include_directories (vsag_diskann_headers INTERFACE
        ${CMAKE_CURRENT_BINARY_DIR}/boost/install/include)
endif ()

# not working without FMA
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2")

add_library (diskann STATIC ${DISKANN_SOURCES})
target_link_libraries (diskann PUBLIC vsag_diskann_headers)
target_compile_options (diskann PRIVATE
    -fdata-sections
    -ffunction-sections
    -ftree-vectorize
    -fno-builtin-malloc
    -fno-builtin-calloc
    -fno-builtin-realloc
    -fno-builtin-free
    -fopenmp
    -fopenmp-simd
    -funroll-loops
    -Wfatal-errors)
target_compile_definitions (diskann PRIVATE ENABLE_CUSTOM_LOGGER=1)
target_link_options (diskann PRIVATE -Wl,--gc-sections)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options (diskann PRIVATE -g1)
endif ()
if (ENABLE_ASAN)
    target_compile_options (diskann PRIVATE -Wno-pass-failed)
endif ()
if (VSAG_BLAS_BACKEND STREQUAL "openblas")
    if (TARGET vsag_openblas_headers)
        target_link_libraries (diskann PRIVATE vsag_openblas_headers)
    endif ()
    maybe_add_dependencies (diskann openblas)
elseif (VSAG_BLAS_BACKEND STREQUAL "mkl")
    if (TARGET vsag_mkl_headers)
        target_link_libraries (diskann PRIVATE vsag_mkl_headers)
    endif ()
    target_compile_definitions (diskann PRIVATE VSAG_USE_MKL_HEADERS)
    maybe_add_dependencies (diskann mkl)
else ()
    message (FATAL_ERROR "Unsupported VSAG_BLAS_BACKEND='${VSAG_BLAS_BACKEND}'. Expected 'openblas' or 'mkl'.")
endif ()
set_property (TARGET diskann PROPERTY CXX_STANDARD 17)
maybe_add_dependencies (diskann boost)
target_link_libraries (diskann PRIVATE ${BLAS_LIBRARIES})

install (
    TARGETS diskann
    ARCHIVE DESTINATION lib)
