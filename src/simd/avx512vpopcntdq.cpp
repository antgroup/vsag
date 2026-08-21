
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

uint64_t
RaBitQSQ4UBinaryIPWithBaseSum(const uint8_t* codes, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_AVX512VPOPCNTDQ)
    const uint64_t num_bytes = (dim + 7) / 8;
    __m512i base_acc = _mm512_setzero_si512();
    __m512i inner_acc[4] = {_mm512_setzero_si512(),
                            _mm512_setzero_si512(),
                            _mm512_setzero_si512(),
                            _mm512_setzero_si512()};
    uint64_t offset = 0;
    for (; offset + 64 <= num_bytes; offset += 64) {
        const auto base = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(bits + offset));
        base_acc = _mm512_add_epi64(base_acc, _mm512_popcnt_epi64(base));
        for (uint32_t bit = 0; bit < 4; ++bit) {
            const auto query = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(codes + bit * num_bytes + offset));
            inner_acc[bit] = _mm512_add_epi64(inner_acc[bit],
                                              _mm512_popcnt_epi64(_mm512_and_si512(query, base)));
        }
    }
    uint32_t base_sum = static_cast<uint32_t>(_mm512_reduce_add_epi64(base_acc));
    uint32_t inner_product = 0;
    for (uint32_t bit = 0; bit < 4; ++bit) {
        inner_product += static_cast<uint32_t>(_mm512_reduce_add_epi64(inner_acc[bit])) << bit;
    }
    for (; offset < num_bytes; ++offset) {
        const auto base = bits[offset];
        base_sum += static_cast<uint32_t>(__builtin_popcount(base));
        for (uint32_t bit = 0; bit < 4; ++bit) {
            inner_product +=
                static_cast<uint32_t>(__builtin_popcount(codes[bit * num_bytes + offset] & base))
                << bit;
        }
    }
    return static_cast<uint64_t>(inner_product) | (static_cast<uint64_t>(base_sum) << 32U);
#else
    return avx2::RaBitQSQ4UBinaryIPWithBaseSum(codes, bits, dim);
#endif
}

void
RaBitQSQ4UBinaryIPWithBaseSumBatch4(const uint8_t* codes,
                                    const uint8_t* bits1,
                                    const uint8_t* bits2,
                                    const uint8_t* bits3,
                                    const uint8_t* bits4,
                                    uint64_t dim,
                                    uint64_t* results) {
#if defined(ENABLE_AVX512VPOPCNTDQ)
    const uint64_t num_bytes = (dim + 7) / 8;
    const uint8_t* bases[4] = {bits1, bits2, bits3, bits4};
    __m512i base_acc[4];
    __m512i inner_acc[4][4];
    for (uint32_t base_id = 0; base_id < 4; ++base_id) {
        base_acc[base_id] = _mm512_setzero_si512();
        for (uint32_t bit = 0; bit < 4; ++bit) {
            inner_acc[base_id][bit] = _mm512_setzero_si512();
        }
    }

    uint64_t offset = 0;
    for (; offset + 64 <= num_bytes; offset += 64) {
        __m512i base_values[4];
        for (uint32_t base_id = 0; base_id < 4; ++base_id) {
            base_values[base_id] =
                _mm512_loadu_si512(reinterpret_cast<const __m512i*>(bases[base_id] + offset));
            base_acc[base_id] =
                _mm512_add_epi64(base_acc[base_id], _mm512_popcnt_epi64(base_values[base_id]));
        }
        for (uint32_t bit = 0; bit < 4; ++bit) {
            const auto query = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(codes + bit * num_bytes + offset));
            for (uint32_t base_id = 0; base_id < 4; ++base_id) {
                inner_acc[base_id][bit] = _mm512_add_epi64(
                    inner_acc[base_id][bit],
                    _mm512_popcnt_epi64(_mm512_and_si512(query, base_values[base_id])));
            }
        }
    }

    uint32_t inner_products[4] = {0, 0, 0, 0};
    uint32_t base_sums[4] = {0, 0, 0, 0};
    for (uint32_t base_id = 0; base_id < 4; ++base_id) {
        base_sums[base_id] = static_cast<uint32_t>(_mm512_reduce_add_epi64(base_acc[base_id]));
        for (uint32_t bit = 0; bit < 4; ++bit) {
            inner_products[base_id] +=
                static_cast<uint32_t>(_mm512_reduce_add_epi64(inner_acc[base_id][bit])) << bit;
        }
    }
    for (; offset < num_bytes; ++offset) {
        for (uint32_t base_id = 0; base_id < 4; ++base_id) {
            const auto base = bases[base_id][offset];
            base_sums[base_id] += static_cast<uint32_t>(__builtin_popcount(base));
            for (uint32_t bit = 0; bit < 4; ++bit) {
                inner_products[base_id] += static_cast<uint32_t>(__builtin_popcount(
                                               codes[bit * num_bytes + offset] & base))
                                           << bit;
            }
        }
    }
    for (uint32_t i = 0; i < 4; ++i) {
        results[i] =
            static_cast<uint64_t>(inner_products[i]) | (static_cast<uint64_t>(base_sums[i]) << 32U);
    }
#else
    avx2::RaBitQSQ4UBinaryIPWithBaseSumBatch4(codes, bits1, bits2, bits3, bits4, dim, results);
#endif
}

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
