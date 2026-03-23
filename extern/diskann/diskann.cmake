include_directories(${CMAKE_CURRENT_BINARY_DIR}/boost/install/include
  ${AIO_INCLUDE_DIR}
  )
if(DEFINED OPENBLAS_INCLUDE_DIRS)
  include_directories(${OPENBLAS_INCLUDE_DIRS})
endif()
include_directories(extern/diskann)

if(VSAG_BLAS_BACKEND STREQUAL "openblas" AND NOT USE_SYSTEM_OPENBLAS)
  link_directories(${CMAKE_CURRENT_BINARY_DIR}/openblas/install/lib)
endif()

include_directories(extern/diskann/DiskANN/include)
set(DISKANN_SOURCES
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

# not working without FMA
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2")

add_library(diskann STATIC ${DISKANN_SOURCES})
target_compile_options(diskann PRIVATE -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DENABLE_CUSTOM_LOGGER=1)
if(ENABLE_ASAN)
  target_compile_options(diskann PRIVATE -Wno-pass-failed)
endif()
if(VSAG_BLAS_BACKEND STREQUAL "mkl")
  target_include_directories(diskann SYSTEM BEFORE PRIVATE ${MKL_INCLUDE_PATH})
  target_compile_definitions(diskann PRIVATE VSAG_USE_MKL_HEADERS)
endif()
set_property(TARGET diskann PROPERTY CXX_STANDARD 17)
add_dependencies(diskann boost)
if(VSAG_BLAS_BACKEND STREQUAL "openblas")
  add_dependencies(diskann openblas)
elseif(VSAG_BLAS_BACKEND STREQUAL "mkl")
  add_dependencies(diskann mkl)
else()
  message(FATAL_ERROR "Unsupported VSAG_BLAS_BACKEND='${VSAG_BLAS_BACKEND}'. Expected 'openblas' or 'mkl'.")
endif()
target_link_libraries(diskann
  ${BLAS_LIBRARIES}
)

install (
  TARGETS diskann
  ARCHIVE DESTINATION lib
  )
