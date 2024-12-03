set (SIMD_SRCS
        generic.cpp
        simd.cpp
        fp32_simd.cpp
        sq8_simd.cpp
        sq4_simd.cpp
        sq4_uniform_simd.cpp
        sq8_uniform_simd.cpp
        sse.cpp
        avx.cpp
        avx512.cpp
)
if (DIST_CONTAINS_SSE)
    set (SIMD_SRCS ${SIMD_SRCS} sse.cpp)
    set_source_files_properties (
            sse.cpp
            PROPERTIES COMPILE_FLAGS
            "-msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2")
endif ()
if (DIST_CONTAINS_AVX)
    set (SIMD_SRCS ${SIMD_SRCS} avx.cpp)
    set_source_files_properties (avx.cpp PROPERTIES COMPILE_FLAGS "-mavx")
endif ()
# FIXME(LHT): cause illegal instruction on platform which has avx only
#if (DIST_CONTAINS_AVX2)
#    set_source_files_properties (avx.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
#endif ()
if (DIST_CONTAINS_AVX512)
    set (SIMD_SRCS ${SIMD_SRCS} avx512.cpp
            normalize.cpp)
    set_source_files_properties (
            avx512.cpp
            PROPERTIES
            COMPILE_FLAGS
            "-mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512ifma -mavx512vbmi"
    )
endif ()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-calloc")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-realloc")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-free")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fopenmp-simd")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")

add_library (simd ${SIMD_SRCS})

macro (simd_add_definitions flag1 flag2)
    if (${flag1})
        target_compile_definitions (simd PUBLIC ${flag2})
    endif ()
endmacro ()

simd_add_definitions (DIST_CONTAINS_SSE -DENABLE_SSE=1)
simd_add_definitions (DIST_CONTAINS_AVX -DENABLE_AVX=1)
#simd_add_definitions (DIST_CONTAINS_AVX2 -DENABLE_AVX2=1)
simd_add_definitions (DIST_CONTAINS_AVX512 -DENABLE_AVX512=1)

target_link_libraries (simd PRIVATE cpuinfo)
install (TARGETS simd ARCHIVE DESTINATION lib)
