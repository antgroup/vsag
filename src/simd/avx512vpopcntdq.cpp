
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
#if defined(ENABLE_AVX512VPOPCNTDQ)
#include <immintrin.h>
#endif

#include "simd.h"

namespace vsag::avx512vpopcntdq {

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim) {
    // require dim align with 512
#if defined(ENABLE_AVX512VPOPCNTDQ)
    if (dim == 0) {
        return 0;
    }

    uint32_t result = 0;
    uint64_t num_bytes = (dim + 7) / 8;

    for (uint64_t bit_pos = 0; bit_pos < 4; ++bit_pos) {
        uint64_t i = 0;

        __m512i acc = _mm512_setzero_si512();
        const uint8_t* cur = codes + bit_pos * num_bytes;
        for (; i + 64 <= num_bytes; i += 64) {
            __m512i vec_codes = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(cur + i));
            __m512i vec_bits = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(bits + i));

            __m512i and_result = _mm512_and_si512(vec_codes, vec_bits);
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(and_result));
        }
        uint64_t sum = _mm512_reduce_add_epi64(acc);

        for (; i < num_bytes; ++i) {
            uint8_t bitwise_and = cur[i] & bits[i];
            sum += __builtin_popcount(bitwise_and);
        }

        result += sum << bit_pos;
    }

    return result;
#else
    return avx512::RaBitQSQ4UBinaryIP(codes, bits, dim);
#endif
}

}  // namespace vsag::avx512vpopcntdq
