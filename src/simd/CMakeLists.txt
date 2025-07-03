set (SIMD_SRCS
        generic.cpp
        sse.cpp
        avx.cpp
        avx2.cpp
        avx512.cpp
        avx512vpopcntdq.cpp
        neon.cpp
        simd.cpp
        simd_status.cpp
        basic_func.cpp
        bit_simd.cpp
        fp32_simd.cpp
        fp16_simd.cpp
        bf16_simd.cpp
        pqfs_simd.cpp
        sq8_simd.cpp
        sq4_simd.cpp
        sq4_uniform_simd.cpp
        sq8_uniform_simd.cpp
        rabitq_simd.cpp
        normalize.cpp
)
if (DIST_CONTAINS_SSE)
    set_source_files_properties (
            sse.cpp
            PROPERTIES COMPILE_FLAGS
            "-msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2")
endif ()
if (DIST_CONTAINS_AVX)
    set_source_files_properties (avx.cpp PROPERTIES COMPILE_FLAGS "-mavx -mf16c")
endif ()
if (DIST_CONTAINS_AVX2)
    set_source_files_properties (avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c")
endif ()
if (DIST_CONTAINS_AVX512)
    set_source_files_properties (
            avx512.cpp
            PROPERTIES
            COMPILE_FLAGS
            "-mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512ifma -mavx512vbmi"
    )
endif ()
if (DIST_CONTAINS_AVX512VPOPCNTDQ)
    set_source_files_properties (
            avx512vpopcntdq.cpp
            PROPERTIES
            COMPILE_FLAGS
            "-mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512ifma -mavx512vbmi -mavx512vpopcntdq"
    )
endif ()
if (DIST_CONTAINS_NEON)
    set_source_files_properties (neon.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
endif ()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-calloc")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-realloc")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-free")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fopenmp-simd")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")

add_library (simd OBJECT ${SIMD_SRCS})

macro (simd_add_definitions flag1 flag2)
    if (${flag1})
        target_compile_definitions (simd PUBLIC ${flag2})
    endif ()
endmacro ()

simd_add_definitions (DIST_CONTAINS_SSE -DENABLE_SSE=1)
simd_add_definitions (DIST_CONTAINS_AVX -DENABLE_AVX=1)
simd_add_definitions (DIST_CONTAINS_AVX2 -DENABLE_AVX2=1)
simd_add_definitions (DIST_CONTAINS_AVX512 -DENABLE_AVX512=1)
simd_add_definitions (DIST_CONTAINS_AVX512VPOPCNTDQ -DENABLE_AVX512VPOPCNTDQ=1)
simd_add_definitions (DIST_CONTAINS_NEON -DENABLE_NEON=1)

target_link_libraries (simd INTERFACE cpuinfo coverage_config)
# install (TARGETS simd ARCHIVE DESTINATION lib)
