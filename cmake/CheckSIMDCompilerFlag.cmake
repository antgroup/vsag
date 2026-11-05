
# Copyright 2024-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx512vpopcntdq.cpp "#include <immintrin.h>\nint main() { __m512i a, b; b = _mm512_popcnt_epi64(a); return 0; }")
try_compile(COMPILER_AVX512VPOPCNTDQ_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx512vpopcntdq
    ${CMAKE_BINARY_DIR}/instructions_test_avx512vpopcntdq.cpp
    COMPILE_DEFINITIONS "-mavx512vpopcntdq"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx512.cpp "#include <immintrin.h>\nint main() { __m512 a, b; a = _mm512_sub_ps(a, b); return 0; }")
try_compile(COMPILER_AVX512_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx512
    ${CMAKE_BINARY_DIR}/instructions_test_avx512.cpp
    COMPILE_DEFINITIONS "-mavx512f"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx2.cpp "#include <immintrin.h>\nint main() { __m256 a, b, c; c = _mm256_fmadd_ps(a, b, c); return 0; }")
try_compile(COMPILER_AVX2_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx2
    ${CMAKE_BINARY_DIR}/instructions_test_avx2.cpp
    COMPILE_DEFINITIONS "-mavx2 -mfma"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx.cpp "#include <immintrin.h>\nint main() { __m256 a, b; a = _mm256_sub_ps(a, b); return 0; }")
try_compile(COMPILER_AVX_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx
    ${CMAKE_BINARY_DIR}/instructions_test_avx.cpp
    COMPILE_DEFINITIONS "-mavx"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_sse.cpp "#include <immintrin.h>\nint main() { __m128 a, b; a = _mm_sub_ps(a, b); return 0; }")
try_compile(COMPILER_SSE_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_sse
    ${CMAKE_BINARY_DIR}/instructions_test_sse.cpp
    COMPILE_DEFINITIONS "-msse"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_sve.cpp
    "#include <arm_sve.h>\n"
    "int main() {\n"
    "    svbool_t pg = svptrue_b32();\n"
    "    svfloat32_t a = svdup_f32(1.0f);\n"
    "    svfloat32_t b = svdup_f32(2.0f);\n"
    "    a = svadd_f32_x(pg, a, b);\n"
    "    return 0;\n"
    "}"
)
try_compile(COMPILER_SVE_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_sve
    ${CMAKE_BINARY_DIR}/instructions_test_sve.cpp
    COMPILE_DEFINITIONS "-march=armv8-a+sve" 
    OUTPUT_VARIABLE COMPILE_OUTPUT
)

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_neon.cpp "#include <arm_neon.h>\nint main() { float32x4_t a, b; a = vdupq_n_f32(1.0f); b = vdupq_n_f32(2.0f); a = vaddq_f32(a, b); return 0; }")
try_compile(COMPILER_NEON_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_neon
    ${CMAKE_BINARY_DIR}/instructions_test_neon.cpp
    COMPILE_DEFINITIONS "-march=armv8-a"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx512vpopcntdq.cpp "#include <immintrin.h>\nint main() { __m512i a, b; b = _mm512_popcnt_epi64(a); return 0; }")
try_compile(RUNTIME_AVX512VPOPCNTDQ_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx512vpopcntdq
    ${CMAKE_BINARY_DIR}/instructions_test_avx512vpopcntdq.cpp
    COMPILE_DEFINITIONS "-march=native"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx512.cpp "#include <immintrin.h>\nint main() { __m512 a, b; a = _mm512_sub_ps(a, b); return 0; }")
try_compile(RUNTIME_AVX512_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx512
    ${CMAKE_BINARY_DIR}/instructions_test_avx512.cpp
    COMPILE_DEFINITIONS "-march=native"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx2.cpp "#include <immintrin.h>\nint main() { __m256 a, b, c; c = _mm256_fmadd_ps(a, b, c); return 0; }")
try_compile(RUNTIME_AVX2_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx2
    ${CMAKE_BINARY_DIR}/instructions_test_avx2.cpp
    COMPILE_DEFINITIONS "-march=native"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_avx.cpp "#include <immintrin.h>\nint main() { __m256 a, b; a = _mm256_sub_ps(a, b); return 0; }")
try_compile(RUNTIME_AVX_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_avx
    ${CMAKE_BINARY_DIR}/instructions_test_avx.cpp
    COMPILE_DEFINITIONS "-march=native"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_sse.cpp "#include <immintrin.h>\nint main() { __m128 a, b; a = _mm_sub_ps(a, b); return 0; }")
try_compile(RUNTIME_SSE_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_sse
    ${CMAKE_BINARY_DIR}/instructions_test_sse.cpp
    COMPILE_DEFINITIONS "-march=native"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

try_compile(RUNTIME_SVE_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_sve
    ${CMAKE_BINARY_DIR}/instructions_test_sve.cpp
    COMPILE_DEFINITIONS "-march=native" 
    OUTPUT_VARIABLE COMPILE_OUTPUT
)

file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_neon.cpp "#include <arm_neon.h>\nint main() { float32x4_t a, b; a = vdupq_n_f32(1.0f); b = vdupq_n_f32(2.0f); a = vaddq_f32(a, b); return 0; }")
try_compile(RUNTIME_NEON_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_neon
    ${CMAKE_BINARY_DIR}/instructions_test_neon.cpp
    COMPILE_DEFINITIONS "-march=armv8-a"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )


file(WRITE ${CMAKE_BINARY_DIR}/instructions_test_amx.cpp "#include <immintrin.h>
int main() {
    _tile_loadconfig((const void*)0);
    _tile_zero(0);
    _tile_dpbuud(0, 1, 2);
    _tile_dpbf16ps(0, 1, 2);
    __m512bh a = (__m512bh)_mm512_setzero_si512();
    __m512bh b = (__m512bh)_mm512_setzero_si512();
    __m512  c = _mm512_setzero_ps();
    c = _mm512_dpbf16_ps(c, a, b);
    _tile_release();
    return 0;
}")
# Probe with the same flag set that src/simd/CMakeLists.txt applies to
# amx.cpp. The AMX translation unit uses AMX_INT8 (tdpbuud), AMX_BF16
# (tdpbf16ps), AVX-512 BF16 helpers, and VNNI lane ops, so a probe that
# only covered -mamx-tile -mamx-int8 would let configure succeed on
# toolchains where the build later fails with "target specific option
# mismatch" on the BF16 intrinsics.
try_compile(COMPILER_AMX_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_amx
    ${CMAKE_BINARY_DIR}/instructions_test_amx.cpp
    COMPILE_DEFINITIONS "-mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512vnni -mavx512bf16 -mamx-tile -mamx-int8 -mamx-bf16"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

try_compile(RUNTIME_AMX_SUPPORTED
    ${CMAKE_BINARY_DIR}/instructions_test_amx
    ${CMAKE_BINARY_DIR}/instructions_test_amx.cpp
    COMPILE_DEFINITIONS "-march=native"
    OUTPUT_VARIABLE COMPILE_OUTPUT
    )

# determine which instructions can be package into distribution
set (COMPILER_SUPPORTED "compiler support instructions: ")
if (COMPILER_SSE_SUPPORTED)
  set (COMPILER_SUPPORTED "${COMPILER_SUPPORTED} SSE")
endif ()
if (COMPILER_AVX_SUPPORTED)
  set (COMPILER_SUPPORTED "${COMPILER_SUPPORTED} AVX")
endif ()
if (COMPILER_AVX2_SUPPORTED)
  set (COMPILER_SUPPORTED "${COMPILER_SUPPORTED} AVX2")
endif ()
if (COMPILER_AVX512_SUPPORTED)
  set (COMPILER_SUPPORTED "${COMPILER_SUPPORTED} AVX512")
endif ()
if (COMPILER_AVX512VPOPCNTDQ_SUPPORTED)
  set (COMPILER_SUPPORTED "${COMPILER_SUPPORTED} AVX512VPOPCNTDQ")
endif ()
if (COMPILER_AMX_SUPPORTED)
  set (COMPILER_SUPPORTED "${COMPILER_SUPPORTED} AMX")
endif ()
if (COMPILER_NEON_SUPPORTED)
  set (COMPILER_SUPPORTED "${COMPILER_SUPPORTED} NEON")
endif ()
if (COMPILER_SVE_SUPPORTED)
  set (COMPILER_SUPPORTED "${COMPILER_SUPPORTED} SVE")
endif ()

message (STATUS ${COMPILER_SUPPORTED})

# RUNTIME just output for debugging
set (RUNTIME_SUPPORTED "runtime support instructions: ")
if (RUNTIME_SSE_SUPPORTED)
  set (RUNTIME_SUPPORTED "${RUNTIME_SUPPORTED} SSE")
endif ()
if (RUNTIME_AVX_SUPPORTED)
  set (RUNTIME_SUPPORTED "${RUNTIME_SUPPORTED} AVX")
endif ()
if (RUNTIME_AVX2_SUPPORTED)
  set (RUNTIME_SUPPORTED "${RUNTIME_SUPPORTED} AVX2")
endif ()
if (RUNTIME_AVX512_SUPPORTED)
  set (RUNTIME_SUPPORTED "${RUNTIME_SUPPORTED} AVX512")
endif ()
if (RUNTIME_AVX512VPOPCNTDQ_SUPPORTED)
  set (RUNTIME_SUPPORTED "${RUNTIME_SUPPORTED} AVX512VPOPCNTDQ")
endif ()
if (RUNTIME_AMX_SUPPORTED)
  set (RUNTIME_SUPPORTED "${RUNTIME_SUPPORTED} AMX")
endif ()
if (RUNTIME_NEON_SUPPORTED)
  set (RUNTIME_SUPPORTED "${RUNTIME_SUPPORTED} NEON")
endif ()
if (RUNTIME_SVE_SUPPORTED)
  set (RUNTIME_SUPPORTED "${RUNTIME_SUPPORTED} SVE")
endif ()

message (STATUS ${RUNTIME_SUPPORTED})

# important distribution logic:
#       (not force disable) and (compiler support) and (lower instructions contained)
set (DIST_CONTAINS_INSTRUCTIONS "distribution contain instructions: ")
if (NOT DISABLE_SSE_FORCE AND COMPILER_SSE_SUPPORTED)
  set (DIST_CONTAINS_SSE ON)
  set (DIST_CONTAINS_INSTRUCTIONS "${DIST_CONTAINS_INSTRUCTIONS} SSE")
endif ()
if (NOT DISABLE_AVX_FORCE AND COMPILER_AVX_SUPPORTED AND DIST_CONTAINS_SSE)
  set (DIST_CONTAINS_AVX ON)
  set (DIST_CONTAINS_INSTRUCTIONS "${DIST_CONTAINS_INSTRUCTIONS} AVX")
endif ()
if (NOT DISABLE_AVX2_FORCE AND COMPILER_AVX2_SUPPORTED AND DIST_CONTAINS_AVX)
  set (DIST_CONTAINS_AVX2 ON)
  set (DIST_CONTAINS_INSTRUCTIONS "${DIST_CONTAINS_INSTRUCTIONS} AVX2")
endif ()
if (NOT DISABLE_AVX512_FORCE AND COMPILER_AVX512_SUPPORTED AND DIST_CONTAINS_AVX2)
  set (DIST_CONTAINS_AVX512 ON)
  set (DIST_CONTAINS_INSTRUCTIONS "${DIST_CONTAINS_INSTRUCTIONS} AVX512")
endif ()
if (NOT DISABLE_AVX512VPOPCNTDQ_FORCE AND COMPILER_AVX512VPOPCNTDQ_SUPPORTED AND DIST_CONTAINS_AVX512)
  set (DIST_CONTAINS_AVX512VPOPCNTDQ ON)
  set (DIST_CONTAINS_INSTRUCTIONS "${DIST_CONTAINS_INSTRUCTIONS} AVX512VPOPCNTDQ")
endif ()
if (NOT DISABLE_AMX_FORCE AND COMPILER_AMX_SUPPORTED AND DIST_CONTAINS_AVX512)
  set (DIST_CONTAINS_AMX ON)
  set (DIST_CONTAINS_INSTRUCTIONS "${DIST_CONTAINS_INSTRUCTIONS} AMX")
endif ()
if (NOT DISABLE_NEON_FORCE AND COMPILER_NEON_SUPPORTED)
  set (DIST_CONTAINS_NEON ON)
  set (DIST_CONTAINS_INSTRUCTIONS "${DIST_CONTAINS_INSTRUCTIONS} NEON")
endif ()
if (NOT DISABLE_SVE_FORCE AND COMPILER_SVE_SUPPORTED)
  set (DIST_CONTAINS_SVE ON)
  set (DIST_CONTAINS_INSTRUCTIONS "${DIST_CONTAINS_INSTRUCTIONS} SVE")
endif ()

message (STATUS ${DIST_CONTAINS_INSTRUCTIONS})
