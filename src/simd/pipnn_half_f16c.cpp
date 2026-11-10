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

#include <immintrin.h>

#include "pipnn_half_scalar.h"

namespace vsag::pipnn_half::detail {

void
DecodeRowFP16(const uint16_t* source, float* destination, uint64_t dim) {
    uint64_t k = 0;
    for (; dim - k >= 8; k += 8) {
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + k));
        const __m128i exponent = _mm_set1_epi16(0x7c00);
        const __m128i nonfinite = _mm_cmpeq_epi16(_mm_and_si128(raw, exponent), exponent);
        // F16C quiets signaling NaNs. Do not convert exceptional lanes: retain the
        // scalar decoder's exact sign/payload bits, without raising invalid for sNaNs.
        const __m128i finite = _mm_andnot_si128(nonfinite, raw);
        _mm256_storeu_ps(destination + k, _mm256_cvtph_ps(finite));
        if (_mm_movemask_epi8(nonfinite) != 0) {
            for (uint64_t lane = 0; lane < 8; ++lane) {
                if ((source[k + lane] & 0x7c00U) == 0x7c00U) {
                    destination[k + lane] = Decode(source[k + lane], Format::FP16);
                }
            }
        }
    }
    DecodeRowScalar<Format::FP16>(source + k, destination + k, dim - k);
}

// Four-lane accumulation matches the baseline FP32 preparation reduction on
// baseline x86 builds. Widen only live registers; no FP32 input arrays.
template <bool square, Format format>
float
PreparationDot(const uint16_t* a, const float* b, uint64_t dim) {
    __m128 sum = _mm_setzero_ps();
    uint64_t k = 0;
    for (; dim - k >= 4; k += 4) {
        uint64_t bits;
        std::memcpy(&bits, a + k, sizeof(bits));
        const __m128i raw = _mm_cvtsi64_si128(static_cast<int64_t>(bits));
        __m128 av;
        if constexpr (format == Format::FP16) {
            av = _mm_cvtph_ps(raw);
        } else {
            av = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(raw), 16));
        }
        const __m128 bv = square ? av : _mm_loadu_ps(b + k);
        sum = _mm_add_ps(sum, _mm_mul_ps(av, bv));
    }
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
    float result = _mm_cvtss_f32(sum);
    for (; k < dim; ++k) {
        const float av = Decode(a[k], format);
        result += av * (square ? av : b[k]);
    }
    return result;
}
float
PreparationAVX2(const uint16_t* a, const float* b, uint64_t dim, Format format) {
    if (b == nullptr) {
        return format == Format::FP16 ? PreparationDot<true, Format::FP16>(a, b, dim)
                                      : PreparationDot<true, Format::BF16>(a, b, dim);
    }
    return format == Format::FP16 ? PreparationDot<false, Format::FP16>(a, b, dim)
                                  : PreparationDot<false, Format::BF16>(a, b, dim);
}

}  // namespace vsag::pipnn_half::detail
