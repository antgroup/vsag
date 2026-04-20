
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "simd_status.h"

namespace vsag {

// Macros that capture the standard SIMD dispatch pattern.
//
// Historically every dispatch function in src/simd/*_simd.cpp selects a
// function pointer based on the best available ISA, falling back from
// the highest instruction set to the generic scalar implementation:
//
//     if (SupportAVX512()) {
//     #if defined(ENABLE_AVX512)
//         return avx512::Name;
//     #endif
//     } else if (SupportAVX2()) {
//     #if defined(ENABLE_AVX2)
//         return avx2::Name;
//     #endif
//     } /* ... sse, sve, neon ... */
//     return generic::Name;
//
// The same cascade repeats ~47 times across 13 files. The macros below
// replace each cascade with a single one-line registration while
// preserving two important properties:
//
//   1. Symbols from an ISA that is not enabled at compile time must not
//      appear in the translation unit. Each VSAG_SIMD_DISPATCH_BODY_<ISA>
//      expands to the return statement only when ENABLE_<ISA> is
//      defined, and to nothing otherwise, so a build without
//      ENABLE_AVX2 never mentions avx2::Name.
//
//   2. When Support<ISA>() is true at runtime but the corresponding
//      ENABLE_<ISA> is not set at compile time, control falls through
//      to the next 'if' clause and eventually to the generic fallback,
//      matching the existing hand-written behavior.
//
// Non-standard dispatch functions (for example those that also try
// AVX512VPOPCNTDQ, or that unconditionally return the generic
// implementation) are intentionally left outside of these macros.

#if defined(ENABLE_AVX512)
#define VSAG_SIMD_DISPATCH_BODY_AVX512(fn) return avx512::fn;
#else
#define VSAG_SIMD_DISPATCH_BODY_AVX512(fn)
#endif

#if defined(ENABLE_AVX512VPOPCNTDQ)
#define VSAG_SIMD_DISPATCH_BODY_AVX512VPOPCNTDQ(fn) return avx512vpopcntdq::fn;
#else
#define VSAG_SIMD_DISPATCH_BODY_AVX512VPOPCNTDQ(fn)
#endif

#if defined(ENABLE_AVX2)
#define VSAG_SIMD_DISPATCH_BODY_AVX2(fn) return avx2::fn;
#else
#define VSAG_SIMD_DISPATCH_BODY_AVX2(fn)
#endif

#if defined(ENABLE_AVX)
#define VSAG_SIMD_DISPATCH_BODY_AVX(fn) return avx::fn;
#else
#define VSAG_SIMD_DISPATCH_BODY_AVX(fn)
#endif

#if defined(ENABLE_SSE)
#define VSAG_SIMD_DISPATCH_BODY_SSE(fn) return sse::fn;
#else
#define VSAG_SIMD_DISPATCH_BODY_SSE(fn)
#endif

#if defined(ENABLE_SVE)
#define VSAG_SIMD_DISPATCH_BODY_SVE(fn) return sve::fn;
#else
#define VSAG_SIMD_DISPATCH_BODY_SVE(fn)
#endif

#if defined(ENABLE_NEON)
#define VSAG_SIMD_DISPATCH_BODY_NEON(fn) return neon::fn;
#else
#define VSAG_SIMD_DISPATCH_BODY_NEON(fn)
#endif

// Register a standard seven-way dispatch. Picks the best available
// implementation among avx512/avx2/avx/sse/sve/neon and falls back to
// generic::FnName. Defines a static Get##FnName() helper and a global
// FnName variable of type FnType.
//
// Usage:
//     VSAG_DEFINE_SIMD_DISPATCH(FP32ComputeIP, FP32ComputeType);
#define VSAG_DEFINE_SIMD_DISPATCH(FnName, FnType)  \
    static FnType Get##FnName() {                  \
        if (SimdStatus::SupportAVX512()) {         \
            VSAG_SIMD_DISPATCH_BODY_AVX512(FnName) \
        }                                          \
        if (SimdStatus::SupportAVX2()) {           \
            VSAG_SIMD_DISPATCH_BODY_AVX2(FnName)   \
        }                                          \
        if (SimdStatus::SupportAVX()) {            \
            VSAG_SIMD_DISPATCH_BODY_AVX(FnName)    \
        }                                          \
        if (SimdStatus::SupportSSE()) {            \
            VSAG_SIMD_DISPATCH_BODY_SSE(FnName)    \
        }                                          \
        if (SimdStatus::SupportSVE()) {            \
            VSAG_SIMD_DISPATCH_BODY_SVE(FnName)    \
        }                                          \
        if (SimdStatus::SupportNEON()) {           \
            VSAG_SIMD_DISPATCH_BODY_NEON(FnName)   \
        }                                          \
        return generic::FnName;                    \
    }                                              \
    FnType FnName = Get##FnName()

// Register a dispatch that additionally prefers AVX512VPOPCNTDQ over
// AVX512. Used by rabitq binary inner-product routines. AVX2/AVX/SSE
// are intentionally skipped because no implementations exist for them.
//
// Usage:
//     VSAG_DEFINE_SIMD_DISPATCH_VPOPCNTDQ(RaBitQSQ4UBinaryIP,
//                                         RaBitQSQ4UBinaryType);
#define VSAG_DEFINE_SIMD_DISPATCH_VPOPCNTDQ(FnName, FnType) \
    static FnType Get##FnName() {                           \
        if (SimdStatus::SupportAVX512VPOPCNTDQ()) {         \
            VSAG_SIMD_DISPATCH_BODY_AVX512VPOPCNTDQ(FnName) \
        }                                                   \
        if (SimdStatus::SupportAVX512()) {                  \
            VSAG_SIMD_DISPATCH_BODY_AVX512(FnName)          \
        }                                                   \
        if (SimdStatus::SupportSVE()) {                     \
            VSAG_SIMD_DISPATCH_BODY_SVE(FnName)             \
        }                                                   \
        if (SimdStatus::SupportNEON()) {                    \
            VSAG_SIMD_DISPATCH_BODY_NEON(FnName)            \
        }                                                   \
        return generic::FnName;                             \
    }                                                       \
    FnType FnName = Get##FnName()

// Register a narrow dispatch for prefetch-like routines that only have
// SSE / SVE / NEON implementations. AVX512/AVX2/AVX are intentionally
// skipped because no implementations exist for them.
//
// Usage:
//     VSAG_DEFINE_SIMD_DISPATCH_PREFETCH(Prefetch, PrefetchFuncType);
#define VSAG_DEFINE_SIMD_DISPATCH_PREFETCH(FnName, FnType) \
    static FnType Get##FnName() {                          \
        if (SimdStatus::SupportSSE()) {                    \
            VSAG_SIMD_DISPATCH_BODY_SSE(FnName)            \
        }                                                  \
        if (SimdStatus::SupportSVE()) {                    \
            VSAG_SIMD_DISPATCH_BODY_SVE(FnName)            \
        }                                                  \
        if (SimdStatus::SupportNEON()) {                   \
            VSAG_SIMD_DISPATCH_BODY_NEON(FnName)           \
        }                                                  \
        return generic::FnName;                            \
    }                                                      \
    FnType FnName = Get##FnName()

}  // namespace vsag
