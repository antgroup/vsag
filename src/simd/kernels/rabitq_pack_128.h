// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstring>

#include "simd/kernels/rabitq_pack.h"
#include "simd/rabitq_simd.h"

#if defined(__aarch64__)
#include <arm_neon.h>
namespace vsag::neon {
#else
#include <emmintrin.h>
namespace vsag::sse {
#endif
namespace {

#if defined(__aarch64__)
using Bytes = uint8x16_t;
using Floats = float32x4_t;
inline Bytes
Load(const uint8_t* p) {
    return vld1q_u8(p);
}
inline void
Store(uint8_t* p, Bytes v) {
    vst1q_u8(p, v);
}
inline Bytes
And(Bytes a, uint8_t mask) {
    return vandq_u8(a, vdupq_n_u8(mask));
}
inline Bytes
Or(Bytes a, Bytes b) {
    return vorrq_u8(a, b);
}
template <int N>
Bytes
ShiftRight(Bytes a) {
    return vshrq_n_u8(a, N);
}
template <int N>
Bytes
ShiftLeft(Bytes a) {
    return vshlq_n_u8(a, N);
}
inline Bytes
ExpandPlane(uint8_t lo, uint8_t hi, uint8_t weight) {
    const uint8_t masks[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    const auto mask = vld1_u8(masks);
    return vandq_u8(vtstq_u8(vcombine_u8(vdup_n_u8(lo), vdup_n_u8(hi)), vcombine_u8(mask, mask)),
                    vdupq_n_u8(weight));
}
inline uint16_t
PackPlane(Bytes bytes, uint32_t bit) {
    const uint8_t weights[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const auto bits = vandq_u8(vshlq_u8(bytes, vdupq_n_s8(-static_cast<int>(bit))), vdupq_n_u8(1));
    const auto weighted = vmulq_u8(bits, vld1q_u8(weights));
    return static_cast<uint16_t>(vaddv_u8(vget_low_u8(weighted))) |
           static_cast<uint16_t>(vaddv_u8(vget_high_u8(weighted))) << 8U;
}
inline void
Widen(Bytes bytes, Floats (&f)[4]) {
    const auto lo = vmovl_u8(vget_low_u8(bytes));
    const auto hi = vmovl_u8(vget_high_u8(bytes));
    f[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
    f[1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo)));
    f[2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
    f[3] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi)));
}
inline Floats
LoadFloat(const float* p) {
    return vld1q_f32(p);
}
inline Floats
Splat(float v) {
    return vdupq_n_f32(v);
}
inline Floats
Sub(Floats a, Floats b) {
    return vsubq_f32(a, b);
}
inline Floats
Add(Floats a, Floats b) {
    return vaddq_f32(a, b);
}
inline Floats
MulAdd(Floats accum, Floats a, Floats b) {
    return vfmaq_f32(accum, a, b);
}
inline float
Sum(Floats v) {
    return vaddvq_f32(v);
}
inline void
StoreFloat(float* p, Floats v) {
    vst1q_f32(p, v);
}
inline bool
Finite(Floats v) {
    const auto exponent = vdupq_n_u32(0x7f800000U);
    return vmaxvq_u32(vceqq_u32(vandq_u32(vreinterpretq_u32_f32(v), exponent), exponent)) == 0U;
}
inline void
AddMoments(Bytes codes, uint64_t& sum, uint64_t* sq) {
    sum += vaddlvq_u8(codes);
    if (sq != nullptr) {
        const auto low = vget_low_u8(codes);
        const auto high = vget_high_u8(codes);
        *sq += vaddlvq_u16(vmull_u8(low, low)) + vaddlvq_u16(vmull_u8(high, high));
    }
}
#else
using Bytes = __m128i;
using Floats = __m128;
inline Bytes
Load(const uint8_t* p) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}
inline void
Store(uint8_t* p, Bytes v) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
}
inline Bytes
And(Bytes a, uint8_t mask) {
    return _mm_and_si128(a, _mm_set1_epi8(mask));
}
inline Bytes
Or(Bytes a, Bytes b) {
    return _mm_or_si128(a, b);
}
template <int N>
Bytes
ShiftRight(Bytes a) {
    return And(_mm_srli_epi16(a, N), 0xffU >> N);
}
template <int N>
Bytes
ShiftLeft(Bytes a) {
    return And(_mm_slli_epi16(a, N), (0xffU << N) & 0xffU);
}
inline Bytes
ExpandPlane(uint8_t lo, uint8_t hi, uint8_t weight) {
    const auto masks = _mm_set1_epi64x(0x8040201008040201ULL);
    const auto packed = _mm_unpacklo_epi64(_mm_set1_epi8(lo), _mm_set1_epi8(hi));
    return _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(packed, masks), masks),
                         _mm_set1_epi8(weight));
}
inline uint16_t
PackPlane(Bytes bytes, uint32_t bit) {
    return static_cast<uint16_t>(_mm_movemask_epi8(
        _mm_sll_epi16(And(bytes, static_cast<uint8_t>(1U << bit)), _mm_cvtsi32_si128(7U - bit))));
}
inline void
Widen(Bytes bytes, Floats (&f)[4]) {
    const auto zero = _mm_setzero_si128();
    const auto lo = _mm_unpacklo_epi8(bytes, zero);
    const auto hi = _mm_unpackhi_epi8(bytes, zero);
    f[0] = _mm_cvtepi32_ps(_mm_unpacklo_epi16(lo, zero));
    f[1] = _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo, zero));
    f[2] = _mm_cvtepi32_ps(_mm_unpacklo_epi16(hi, zero));
    f[3] = _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi, zero));
}
inline Floats
LoadFloat(const float* p) {
    return _mm_loadu_ps(p);
}
inline Floats
Splat(float v) {
    return _mm_set1_ps(v);
}
inline Floats
Sub(Floats a, Floats b) {
    return _mm_sub_ps(a, b);
}
inline Floats
Add(Floats a, Floats b) {
    return _mm_add_ps(a, b);
}
inline Floats
MulAdd(Floats accum, Floats a, Floats b) {
    return _mm_add_ps(accum, _mm_mul_ps(a, b));
}
inline float
Sum(Floats v) {
    alignas(16) float lanes[4];
    _mm_store_ps(lanes, v);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}
inline void
StoreFloat(float* p, Floats v) {
    _mm_storeu_ps(p, v);
}
inline bool
Finite(Floats v) {
    const auto exponent = _mm_set1_epi32(0x7f800000U);
    return _mm_movemask_epi8(
               _mm_cmpeq_epi32(_mm_and_si128(_mm_castps_si128(v), exponent), exponent)) == 0;
}
inline void
AddMoments(Bytes codes, uint64_t& sum, uint64_t* sq) {
    const auto zero = _mm_setzero_si128();
    const auto sums = _mm_sad_epu8(codes, zero);
    sum += static_cast<uint64_t>(_mm_cvtsi128_si64(sums)) +
           static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_srli_si128(sums, 8)));
    if (sq != nullptr) {
        const auto lo = _mm_unpacklo_epi8(codes, zero);
        const auto hi = _mm_unpackhi_epi8(codes, zero);
        const auto squares = _mm_add_epi32(_mm_madd_epi16(lo, lo), _mm_madd_epi16(hi, hi));
        const auto pairs = _mm_add_epi32(squares, _mm_srli_si128(squares, 8));
        *sq += static_cast<uint32_t>(
            _mm_cvtsi128_si32(_mm_add_epi32(pairs, _mm_srli_si128(pairs, 4))));
    }
}
#endif

inline void
StorePlane(uint8_t* out, uint16_t bits) {
    out[0] = static_cast<uint8_t>(bits);
    out[1] = static_cast<uint8_t>(bits >> 8U);
}

// Compact supplement layout and legacy ExCode7 share the low-six layout, but
// the two high bit planes have different numbering and must not be conflated.
inline void
UnpackLowSix(const uint8_t* packed, Bytes (&codes)[4]) {
    const auto a = Load(packed);
    const auto b = Load(packed + 16);
    const auto c = Load(packed + 32);
    codes[0] = And(a, 0x3f);
    codes[1] = And(b, 0x3f);
    codes[2] = And(c, 0x3f);
    codes[3] = Or(Or(ShiftRight<6>(a), And(ShiftRight<4>(b), 0x0c)), And(ShiftRight<2>(c), 0x30));
}

inline void
UnpackCompact(const uint8_t* packed, uint32_t bits, Bytes (&codes)[4]) {
    if (bits == 5U) {
        const auto a = Load(packed);
        const auto b = Load(packed + 16);
        codes[0] = And(a, 0x0f);
        codes[1] = ShiftRight<4>(a);
        codes[2] = And(b, 0x0f);
        codes[3] = ShiftRight<4>(b);
    } else {
        UnpackLowSix(packed, codes);
    }
    if (bits != 6U) {
        const auto* plane = packed + (bits == 5U ? 32U : 48U);
        const auto weight = static_cast<uint8_t>(bits == 5U ? 16U : 64U);
        for (unsigned g = 0; g < 4; ++g) {
            codes[g] = Or(codes[g], ExpandPlane(plane[2 * g], plane[2 * g + 1], weight));
        }
    }
}

inline void
PackCompact(const uint8_t* input, uint8_t* out, uint32_t bits) {
    const Bytes codes[4] = {Load(input), Load(input + 16), Load(input + 32), Load(input + 48)};
    if (bits == 5U) {
        Store(out, Or(And(codes[0], 15), ShiftLeft<4>(codes[1])));
        Store(out + 16, Or(And(codes[2], 15), ShiftLeft<4>(codes[3])));
    } else {
        Store(out, Or(And(codes[0], 63), ShiftLeft<6>(codes[3])));
        Store(out + 16, Or(And(codes[1], 63), ShiftLeft<6>(ShiftRight<2>(codes[3]))));
        Store(out + 32, Or(And(codes[2], 63), ShiftLeft<6>(ShiftRight<4>(codes[3]))));
    }
    if (bits != 6U) {
        auto* plane = out + (bits == 5U ? 32U : 48U);
        for (unsigned g = 0; g < 4; ++g) {
            StorePlane(plane + 2 * g, PackPlane(codes[g], bits - 1U));
        }
    }
}

inline void
UnpackLegacy(const uint8_t* packed, Bytes (&codes)[4]) {
    UnpackLowSix(packed, codes);
    const auto* top = packed + 48;
    for (unsigned g = 0; g < 4; ++g) {
        // Legacy byte j carries lane j from each successive 8-lane group.
#if defined(__aarch64__)
        const auto upper = vld1_u8(top);
        const auto lower = vtst_u8(upper, vdup_n_u8(1U << (2 * g)));
        const auto higher = vtst_u8(upper, vdup_n_u8(2U << (2 * g)));
        codes[g] = Or(codes[g], And(vcombine_u8(lower, higher), 64));
#else
        const auto upper = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(top));
        const auto lanes = _mm_unpacklo_epi64(upper, upper);
        const auto low_mask = _mm_set1_epi8(1U << (2 * g));
        const auto high_mask = _mm_set1_epi8(2U << (2 * g));
        const auto masks = _mm_unpacklo_epi64(low_mask, high_mask);
        codes[g] = Or(codes[g], And(_mm_cmpeq_epi8(_mm_and_si128(lanes, masks), masks), 64));
#endif
    }
}

inline void
AddDot(Bytes codes, const float* query, Floats (&sum)[4]) {
    Floats f[4];
    Widen(codes, f);
    for (unsigned i = 0; i < 4; ++i) {
        sum[i] = MulAdd(sum[i], f[i], LoadFloat(query + i * 4));
    }
}

inline float
ReduceDot(const Floats (&sum)[4]) {
    return Sum(Add(Add(sum[0], sum[1]), Add(sum[2], sum[3])));
}

}  // namespace

float
RaBitQFloatPackedSupplementCodeIP(const float* query,
                                  const uint8_t* packed,
                                  uint64_t dim,
                                  uint32_t bits) {
    if (bits > 8U) {
        return generic::RaBitQFloatPackedSupplementCodeIP(query, packed, dim, bits);
    }
    const simd::RaBitQPackedSupplementLayout layout{dim, bits};
    if (layout.PackedSize() == 0U) {
        return 0.0F;
    }
    Floats sums[4] = {Splat(0), Splat(0), Splat(0), Splat(0)};
    for (uint64_t chunk = 0; chunk < layout.FullChunkCount(); ++chunk) {
        Bytes codes[4];
        UnpackCompact(packed + chunk * layout.ChunkBytes(), bits, codes);
        for (unsigned g = 0; g < 4; ++g) {
            AddDot(codes[g], query + 64 * chunk + 16 * g, sums);
        }
    }
    const uint8_t* tail = packed + layout.FullChunksSize();
    const uint64_t tail_dim = layout.TailDimension();
    uint64_t lane = 0;
    for (; lane + 16 <= tail_dim; lane += 16) {
        auto codes = ExpandPlane(0, 0, 0);
        for (uint32_t bit = 0; bit < bits; ++bit) {
            const auto* plane = tail + bit * layout.TailPlaneBytes() + lane / 8;
            codes = Or(codes, ExpandPlane(plane[0], plane[1], 1U << bit));
        }
        AddDot(codes, query + layout.FullDimension() + lane, sums);
    }
    float result = ReduceDot(sums);
    for (; lane < tail_dim; ++lane) {
        uint32_t code = 0;
        for (uint32_t bit = 0; bit < bits; ++bit) {
            const auto* plane = tail + bit * layout.TailPlaneBytes();
            code |= ((plane[lane / 8] >> (lane % 8)) & 1U) << bit;
        }
        result += query[layout.FullDimension() + lane] * static_cast<float>(code);
    }
    return result;
}

#if !defined(__aarch64__)
void
RaBitQPackScalarToSplitPlanes(const uint8_t* scalar_codes,
                              uint8_t* filter,
                              uint8_t* supplement,
                              uint64_t dim,
                              uint32_t total_bits,
                              uint32_t filter_bits) {
    const uint64_t plane_bytes = (dim + 7U) / 8U;
    uint64_t lane = 0;
    for (; lane + 16 <= dim; lane += 16) {
        const auto codes = Load(scalar_codes + lane);
        for (uint32_t bit = 0; bit < total_bits; ++bit) {
            auto* plane = simd::RaBitQGetSplitPlane(
                filter, supplement, plane_bytes, total_bits, filter_bits, bit);
            StorePlane(plane + lane / 8U, PackPlane(codes, bit));
        }
    }
    simd::RaBitQPackScalarToSplitPlanesTail(
        scalar_codes, filter, supplement, dim, total_bits, filter_bits, lane);
}
#endif

void
RaBitQPackScalarToSplitCode(const uint8_t* scalar_codes,
                            uint8_t* filter,
                            uint8_t* packed,
                            uint64_t dim,
                            uint32_t total_bits,
                            uint32_t filter_bits) {
    const simd::RaBitQPackedSupplementLayout layout{dim, total_bits - filter_bits};
    if (dim == 0U) {
        return;
    }
    const uint64_t plane_bytes = layout.PlaneBytes();
    if (filter_bits != 0U) {
        uint64_t lane = 0;
        for (; lane + 16 <= dim; lane += 16) {
            const auto codes = Load(scalar_codes + lane);
            for (uint32_t plane = 0; plane < filter_bits; ++plane) {
                StorePlane(filter + plane * plane_bytes + lane / 8,
                           PackPlane(codes, total_bits - 1U - plane));
            }
        }
        for (uint32_t plane = 0; plane < filter_bits; ++plane) {
            std::memset(filter + plane * plane_bytes + lane / 8U, 0, (dim - lane + 7U) / 8U);
        }
        simd::RaBitQPackScalarFilterPlanesTail(
            scalar_codes, filter, dim, total_bits, filter_bits, lane);
    }
    if (layout.PackedSize() == 0U) {
        return;
    }
    for (uint64_t chunk = 0; chunk < layout.FullChunkCount(); ++chunk) {
        PackCompact(scalar_codes + chunk * layout.CHUNK_DIM,
                    packed + chunk * layout.ChunkBytes(),
                    layout.bits);
    }
    auto* tail = packed + layout.FullChunksSize();
    const auto* tail_codes = scalar_codes + layout.FullDimension();
    const uint64_t tail_dim = layout.TailDimension();
    const uint64_t tail_plane_bytes = layout.TailPlaneBytes();
    for (uint32_t bit = 0; bit < layout.bits; ++bit) {
        auto* plane = tail + bit * tail_plane_bytes;
        uint64_t lane = 0;
        for (; lane + 16 <= tail_dim; lane += 16) {
            StorePlane(plane + lane / 8U, PackPlane(Load(tail_codes + lane), bit));
        }
        for (; lane < tail_dim; lane += 8) {
            uint8_t byte = 0;
            for (uint64_t j = 0; j < 8 && lane + j < tail_dim; ++j) {
                byte |= ((tail_codes[lane + j] >> bit) & 1U) << j;
            }
            plane[lane / 8U] = byte;
        }
    }
}

float
RaBitQFloatExCode7IP(const float* query, const uint8_t* packed, uint64_t dim) {
    if ((dim & 63U) != 0U) {
        return 0.0F;
    }
    Floats sums[4] = {Splat(0), Splat(0), Splat(0), Splat(0)};
    for (uint64_t block = 0; block < dim; block += 64, packed += 56) {
        Bytes codes[4];
        UnpackLegacy(packed, codes);
        for (unsigned g = 0; g < 4; ++g) {
            AddDot(codes[g], query + block + 16 * g, sums);
        }
    }
    return ReduceDot(sums);
}

bool
RaBitQExCode7ToBytes(const uint8_t* filter,
                     const uint8_t* packed,
                     uint64_t dim,
                     uint8_t* out,
                     uint64_t* code_sum,
                     uint64_t* code_sq_sum) {
    if (filter == nullptr || packed == nullptr || out == nullptr || code_sum == nullptr ||
        !RaBitQExCode7SupportedDim(dim)) {
        return false;
    }
    uint64_t sum = 0;
    uint64_t squares = 0;
    for (uint64_t block = 0; block < dim; block += 64, packed += 56, filter += 8) {
        Bytes codes[4];
        UnpackLegacy(packed, codes);
        for (unsigned g = 0; g < 4; ++g) {
            const auto full = Or(codes[g], ExpandPlane(filter[2 * g], filter[2 * g + 1], 128));
            Store(out + block + 16 * g, full);
            AddMoments(full, sum, code_sq_sum != nullptr ? &squares : nullptr);
        }
    }
    *code_sum = sum;
    if (code_sq_sum != nullptr) {
        *code_sq_sum = squares;
    }
    return true;
}

bool
RaBitQExCode7ToVector(const uint8_t* filter,
                      const uint8_t* packed,
                      const float* centroid,
                      float residual_scale,
                      float full_center,
                      uint64_t dim,
                      float* out) {
    if (filter == nullptr || packed == nullptr || centroid == nullptr || out == nullptr ||
        !RaBitQExCode7SupportedDim(dim)) {
        return false;
    }
    const auto scale = Splat(residual_scale);
    const auto center = Splat(full_center);
    bool finite = true;
    for (uint64_t block = 0; block < dim; block += 64, packed += 56, filter += 8) {
        Bytes codes[4];
        UnpackLegacy(packed, codes);
        for (unsigned g = 0; g < 4; ++g) {
            Floats f[4];
            Widen(Or(codes[g], ExpandPlane(filter[2 * g], filter[2 * g + 1], 128)), f);
            for (unsigned i = 0; i < 4; ++i) {
                const uint64_t lane = block + g * 16 + i * 4;
                const auto value = MulAdd(LoadFloat(centroid + lane), scale, Sub(f[i], center));
                StoreFloat(out + lane, value);
                finite &= Finite(value);
            }
        }
    }
    return finite;
}

}  // namespace vsag::{neon,sse}
