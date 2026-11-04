
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
#include "simd/kernels/hamming_compute.h"

namespace vsag::avx512vpopcntdq {

#if defined(ENABLE_AVX512VPOPCNTDQ)
namespace {

constexpr uint64_t K_BYTES_PER_VECTOR = 64;
constexpr uint64_t K_BYTES_PER_LANE = sizeof(uint64_t);

__mmask8
lower_lane_mask(uint64_t lane_count) {
    return static_cast<__mmask8>((uint32_t{1} << lane_count) - 1U);
}

}  // namespace
#endif

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
    for (; offset + K_BYTES_PER_VECTOR <= num_bytes; offset += K_BYTES_PER_VECTOR) {
        const auto base = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(bits + offset));
        base_acc = _mm512_add_epi64(base_acc, _mm512_popcnt_epi64(base));
        for (uint32_t bit = 0; bit < 4; ++bit) {
            const auto query = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(codes + bit * num_bytes + offset));
            inner_acc[bit] = _mm512_add_epi64(inner_acc[bit],
                                              _mm512_popcnt_epi64(_mm512_and_si512(query, base)));
        }
    }
    const auto remaining_lanes = (num_bytes - offset) / K_BYTES_PER_LANE;
    if (remaining_lanes > 0) {
        const auto mask = lower_lane_mask(remaining_lanes);
        const auto base = _mm512_maskz_loadu_epi64(mask, bits + offset);
        base_acc = _mm512_add_epi64(base_acc, _mm512_popcnt_epi64(base));
        for (uint32_t bit = 0; bit < 4; ++bit) {
            const auto query = _mm512_maskz_loadu_epi64(mask, codes + bit * num_bytes + offset);
            inner_acc[bit] = _mm512_add_epi64(inner_acc[bit],
                                              _mm512_popcnt_epi64(_mm512_and_si512(query, base)));
        }
        offset += remaining_lanes * K_BYTES_PER_LANE;
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
    for (; offset + K_BYTES_PER_VECTOR <= num_bytes; offset += K_BYTES_PER_VECTOR) {
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

    const auto remaining_lanes = (num_bytes - offset) / K_BYTES_PER_LANE;
    if (remaining_lanes > 0) {
        const auto mask = lower_lane_mask(remaining_lanes);
        __m512i base_values[4];
        for (uint32_t base_id = 0; base_id < 4; ++base_id) {
            base_values[base_id] = _mm512_maskz_loadu_epi64(mask, bases[base_id] + offset);
            base_acc[base_id] =
                _mm512_add_epi64(base_acc[base_id], _mm512_popcnt_epi64(base_values[base_id]));
        }
        for (uint32_t bit = 0; bit < 4; ++bit) {
            const auto query = _mm512_maskz_loadu_epi64(mask, codes + bit * num_bytes + offset);
            for (uint32_t base_id = 0; base_id < 4; ++base_id) {
                inner_acc[base_id][bit] = _mm512_add_epi64(
                    inner_acc[base_id][bit],
                    _mm512_popcnt_epi64(_mm512_and_si512(query, base_values[base_id])));
            }
        }
        offset += remaining_lanes * K_BYTES_PER_LANE;
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
        for (; i + K_BYTES_PER_VECTOR <= num_bytes; i += K_BYTES_PER_VECTOR) {
            __m512i vec_codes = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(cur + i));
            __m512i vec_bits = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(bits + i));

            __m512i and_result = _mm512_and_si512(vec_codes, vec_bits);
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(and_result));
        }
        const auto remaining_lanes = (num_bytes - i) / K_BYTES_PER_LANE;
        if (remaining_lanes > 0) {
            const auto mask = lower_lane_mask(remaining_lanes);
            const auto vec_codes = _mm512_maskz_loadu_epi64(mask, cur + i);
            const auto vec_bits = _mm512_maskz_loadu_epi64(mask, bits + i);
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_and_si512(vec_codes, vec_bits)));
            i += remaining_lanes * K_BYTES_PER_LANE;
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

#if defined(ENABLE_AVX512VPOPCNTDQ)
namespace {
struct HammingVpopcntdqTag {};

// Number of set bits per 64-bit lane.
inline __m512i
hamming_count_lanes(__m512i diff) {
    return _mm512_popcnt_epi64(diff);
}

// Selects the first `count` bytes of a 64-byte block (count <= 64). Byte masks need AVX-512BW,
// which the dispatcher requires together with VPOPCNTDQ.
inline __mmask64
hamming_byte_mask(uint64_t count) {
    return count >= 64 ? ~__mmask64{0} : static_cast<__mmask64>((uint64_t{1} << count) - 1U);
}

// Valid bits of a block that ends with the last `count` bytes of a vector: every byte is 0xFF
// except the partial last byte, which keeps only its high dim % 8 bits.
inline __m512i
hamming_tail_bits(uint64_t count, uint64_t dim) {
    const __m512i all = _mm512_set1_epi8(static_cast<char>(0xFF));
    if (dim % 8 == 0) {
        return all;
    }
    const auto last = static_cast<char>(simd::HammingTailMask(dim));
    return _mm512_mask_set1_epi8(all, __mmask64{1} << (count - 1), last);
}

// Horizontal sums of four accumulators: element k of the result is the sum of the eight 64-bit
// lanes of a<k>, which replaces four separate reductions.
inline __m256i
hamming_reduce4(__m512i a0, __m512i a1, __m512i a2, __m512i a3) {
    // Each 128-bit block b of p01 holds [a0[2b] + a0[2b + 1], a1[2b] + a1[2b + 1]].
    const __m512i p01 =
        _mm512_add_epi64(_mm512_unpacklo_epi64(a0, a1), _mm512_unpackhi_epi64(a0, a1));
    const __m512i p23 =
        _mm512_add_epi64(_mm512_unpacklo_epi64(a2, a3), _mm512_unpackhi_epi64(a2, a3));
    // The blocks of s hold [a0, a1] over lanes 0-3, [a0, a1] over lanes 4-7, then a2, a3 likewise.
    const __m512i s = _mm512_add_epi64(_mm512_shuffle_i64x2(p01, p23, _MM_SHUFFLE(2, 0, 2, 0)),
                                       _mm512_shuffle_i64x2(p01, p23, _MM_SHUFFLE(3, 1, 3, 1)));
    const __m512i halves = _mm512_shuffle_i64x2(s, s, _MM_SHUFFLE(3, 1, 2, 0));
    return _mm256_add_epi64(_mm512_castsi512_si256(halves), _mm512_extracti64x4_epi64(halves, 1));
}
}  // namespace
#endif

uint64_t
HammingDistance(const uint8_t* x, const uint8_t* y, uint64_t dim) {
#if defined(ENABLE_AVX512VPOPCNTDQ)
    const uint64_t full_bytes = simd::HammingFullBytes(dim);
    const uint64_t total_bytes = (dim + 7) / 8;
    if (dim % 64 == 0 and dim <= 128) {
        // One or two whole 64-bit words: scalar POPCNT is as fast and skips the reduction.
        return simd::HammingScalarRange<HammingVpopcntdqTag>(x, y, 0, dim);
    }
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    uint64_t i = 0;
    for (; i + 2 * K_BYTES_PER_VECTOR <= full_bytes; i += 2 * K_BYTES_PER_VECTOR) {
        acc1 = _mm512_add_epi64(acc1,
                                hamming_count_lanes(_mm512_xor_si512(_mm512_loadu_si512(x + i),
                                                                     _mm512_loadu_si512(y + i))));
        acc2 = _mm512_add_epi64(
            acc2,
            hamming_count_lanes(_mm512_xor_si512(_mm512_loadu_si512(x + i + K_BYTES_PER_VECTOR),
                                                 _mm512_loadu_si512(y + i + K_BYTES_PER_VECTOR))));
    }
    if (i + K_BYTES_PER_VECTOR <= full_bytes) {
        acc1 = _mm512_add_epi64(acc1,
                                hamming_count_lanes(_mm512_xor_si512(_mm512_loadu_si512(x + i),
                                                                     _mm512_loadu_si512(y + i))));
        i += K_BYTES_PER_VECTOR;
    }
    if (i < total_bytes) {
        // The remainder, including a partial last byte, is one masked block: masked-out bytes
        // load as zero and never fault, and the padding bits are cleared before counting.
        const uint64_t count = total_bytes - i;
        const __mmask64 mask = hamming_byte_mask(count);
        const __m512i diff =
            _mm512_and_si512(_mm512_xor_si512(_mm512_maskz_loadu_epi8(mask, x + i),
                                              _mm512_maskz_loadu_epi8(mask, y + i)),
                             hamming_tail_bits(count, dim));
        acc2 = _mm512_add_epi64(acc2, hamming_count_lanes(diff));
    }
    return static_cast<uint64_t>(_mm512_reduce_add_epi64(_mm512_add_epi64(acc1, acc2)));
#else
    return avx512::HammingDistance(x, y, dim);
#endif
}

void
HammingDistanceBatch4(const uint8_t* query,
                      const uint8_t* codes1,
                      const uint8_t* codes2,
                      const uint8_t* codes3,
                      const uint8_t* codes4,
                      uint64_t dim,
                      uint64_t* results) {
#if defined(ENABLE_AVX512VPOPCNTDQ)
    const uint64_t full_bytes = simd::HammingFullBytes(dim);
    const uint64_t total_bytes = (dim + 7) / 8;
    if (dim % 64 == 0 and dim <= 128) {
        results[0] = 0;
        results[1] = 0;
        results[2] = 0;
        results[3] = 0;
        simd::HammingScalarRangeBatch4<HammingVpopcntdqTag>(
            query, codes1, codes2, codes3, codes4, 0, dim, results);
        return;
    }
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    __m512i acc3 = _mm512_setzero_si512();
    __m512i acc4 = _mm512_setzero_si512();
    uint64_t i = 0;
    for (; i + K_BYTES_PER_VECTOR <= full_bytes; i += K_BYTES_PER_VECTOR) {
        const __m512i q = _mm512_loadu_si512(query + i);
        acc1 = _mm512_add_epi64(
            acc1, hamming_count_lanes(_mm512_xor_si512(q, _mm512_loadu_si512(codes1 + i))));
        acc2 = _mm512_add_epi64(
            acc2, hamming_count_lanes(_mm512_xor_si512(q, _mm512_loadu_si512(codes2 + i))));
        acc3 = _mm512_add_epi64(
            acc3, hamming_count_lanes(_mm512_xor_si512(q, _mm512_loadu_si512(codes3 + i))));
        acc4 = _mm512_add_epi64(
            acc4, hamming_count_lanes(_mm512_xor_si512(q, _mm512_loadu_si512(codes4 + i))));
    }
    if (i < total_bytes) {
        const uint64_t count = total_bytes - i;
        const __mmask64 mask = hamming_byte_mask(count);
        const __m512i q = _mm512_maskz_loadu_epi8(mask, query + i);
        const __m512i valid = hamming_tail_bits(count, dim);
        acc1 = _mm512_add_epi64(
            acc1,
            hamming_count_lanes(_mm512_and_si512(
                _mm512_xor_si512(q, _mm512_maskz_loadu_epi8(mask, codes1 + i)), valid)));
        acc2 = _mm512_add_epi64(
            acc2,
            hamming_count_lanes(_mm512_and_si512(
                _mm512_xor_si512(q, _mm512_maskz_loadu_epi8(mask, codes2 + i)), valid)));
        acc3 = _mm512_add_epi64(
            acc3,
            hamming_count_lanes(_mm512_and_si512(
                _mm512_xor_si512(q, _mm512_maskz_loadu_epi8(mask, codes3 + i)), valid)));
        acc4 = _mm512_add_epi64(
            acc4,
            hamming_count_lanes(_mm512_and_si512(
                _mm512_xor_si512(q, _mm512_maskz_loadu_epi8(mask, codes4 + i)), valid)));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(results),
                        hamming_reduce4(acc1, acc2, acc3, acc4));
#else
    avx512::HammingDistanceBatch4(query, codes1, codes2, codes3, codes4, dim, results);
#endif
}

#if defined(ENABLE_AVX512VPOPCNTDQ)
namespace {

// Horizontal sums of eight accumulators: lane k of the result is the sum of the lanes of a<k>.
inline __m512i
hamming_reduce8(__m512i a0,
                __m512i a1,
                __m512i a2,
                __m512i a3,
                __m512i a4,
                __m512i a5,
                __m512i a6,
                __m512i a7) {
    const __m512i p01 =
        _mm512_add_epi64(_mm512_unpacklo_epi64(a0, a1), _mm512_unpackhi_epi64(a0, a1));
    const __m512i p23 =
        _mm512_add_epi64(_mm512_unpacklo_epi64(a2, a3), _mm512_unpackhi_epi64(a2, a3));
    const __m512i p45 =
        _mm512_add_epi64(_mm512_unpacklo_epi64(a4, a5), _mm512_unpackhi_epi64(a4, a5));
    const __m512i p67 =
        _mm512_add_epi64(_mm512_unpacklo_epi64(a6, a7), _mm512_unpackhi_epi64(a6, a7));
    const __m512i s0123 = _mm512_add_epi64(_mm512_shuffle_i64x2(p01, p23, _MM_SHUFFLE(2, 0, 2, 0)),
                                           _mm512_shuffle_i64x2(p01, p23, _MM_SHUFFLE(3, 1, 3, 1)));
    const __m512i s4567 = _mm512_add_epi64(_mm512_shuffle_i64x2(p45, p67, _MM_SHUFFLE(2, 0, 2, 0)),
                                           _mm512_shuffle_i64x2(p45, p67, _MM_SHUFFLE(3, 1, 3, 1)));
    // Block b of the sum holds the totals of a<2b> and a<2b + 1>.
    return _mm512_add_epi64(_mm512_shuffle_i64x2(s0123, s4567, _MM_SHUFFLE(2, 0, 2, 0)),
                            _mm512_shuffle_i64x2(s0123, s4567, _MM_SHUFFLE(3, 1, 3, 1)));
}

// Differing bits of one 64-byte block of a code against the matching query block, per lane.
template <bool PartialByte>
inline __m512i
hamming_masked_count(__m512i query_block, __mmask64 mask, const uint8_t* code, __m512i valid) {
    __m512i diff = _mm512_xor_si512(query_block, _mm512_maskz_loadu_epi8(mask, code));
    if constexpr (PartialByte) {
        diff = _mm512_and_si512(diff, valid);
    }
    return hamming_count_lanes(diff);
}

// Scan of codes of any size, eight codes at a time: whole 64-byte blocks first, then the
// remaining bytes as one masked block, and a single eight-way reduction per group.
template <bool PartialByte>
inline void
hamming_scan_blocks(
    const uint8_t* query, const uint8_t* codes, uint64_t dim, uint64_t count, float* dists) {
    const uint64_t code_size = (dim + 7) / 8;
    const uint64_t body = simd::HammingFullBytes(dim) / K_BYTES_PER_VECTOR * K_BYTES_PER_VECTOR;
    const uint64_t tail = code_size - body;
    const __mmask64 tail_mask = hamming_byte_mask(tail);
    const __m512i tail_query = _mm512_maskz_loadu_epi8(tail_mask, query + body);
    const __m512i valid = tail > 0 ? hamming_tail_bits(tail, dim) : _mm512_setzero_si512();
    uint64_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const uint8_t* c0 = codes + i * code_size;
        const uint8_t* c1 = c0 + code_size;
        const uint8_t* c2 = c1 + code_size;
        const uint8_t* c3 = c2 + code_size;
        const uint8_t* c4 = c3 + code_size;
        const uint8_t* c5 = c4 + code_size;
        const uint8_t* c6 = c5 + code_size;
        const uint8_t* c7 = c6 + code_size;
        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();
        __m512i acc4 = _mm512_setzero_si512();
        __m512i acc5 = _mm512_setzero_si512();
        __m512i acc6 = _mm512_setzero_si512();
        __m512i acc7 = _mm512_setzero_si512();
        for (uint64_t j = 0; j < body; j += K_BYTES_PER_VECTOR) {
            const __m512i q = _mm512_loadu_si512(query + j);
            acc0 = _mm512_add_epi64(
                acc0, hamming_count_lanes(_mm512_xor_si512(q, _mm512_loadu_si512(c0 + j))));
            acc1 = _mm512_add_epi64(
                acc1, hamming_count_lanes(_mm512_xor_si512(q, _mm512_loadu_si512(c1 + j))));
            acc2 = _mm512_add_epi64(
                acc2, hamming_count_lanes(_mm512_xor_si512(q, _mm512_loadu_si512(c2 + j))));
            acc3 = _mm512_add_epi64(
                acc3, hamming_count_lanes(_mm512_xor_si512(q, _mm512_loadu_si512(c3 + j))));
            acc4 = _mm512_add_epi64(
                acc4, hamming_count_lanes(_mm512_xor_si512(q, _mm512_loadu_si512(c4 + j))));
            acc5 = _mm512_add_epi64(
                acc5, hamming_count_lanes(_mm512_xor_si512(q, _mm512_loadu_si512(c5 + j))));
            acc6 = _mm512_add_epi64(
                acc6, hamming_count_lanes(_mm512_xor_si512(q, _mm512_loadu_si512(c6 + j))));
            acc7 = _mm512_add_epi64(
                acc7, hamming_count_lanes(_mm512_xor_si512(q, _mm512_loadu_si512(c7 + j))));
        }
        if (tail > 0) {
            acc0 = _mm512_add_epi64(
                acc0, hamming_masked_count<PartialByte>(tail_query, tail_mask, c0 + body, valid));
            acc1 = _mm512_add_epi64(
                acc1, hamming_masked_count<PartialByte>(tail_query, tail_mask, c1 + body, valid));
            acc2 = _mm512_add_epi64(
                acc2, hamming_masked_count<PartialByte>(tail_query, tail_mask, c2 + body, valid));
            acc3 = _mm512_add_epi64(
                acc3, hamming_masked_count<PartialByte>(tail_query, tail_mask, c3 + body, valid));
            acc4 = _mm512_add_epi64(
                acc4, hamming_masked_count<PartialByte>(tail_query, tail_mask, c4 + body, valid));
            acc5 = _mm512_add_epi64(
                acc5, hamming_masked_count<PartialByte>(tail_query, tail_mask, c5 + body, valid));
            acc6 = _mm512_add_epi64(
                acc6, hamming_masked_count<PartialByte>(tail_query, tail_mask, c6 + body, valid));
            acc7 = _mm512_add_epi64(
                acc7, hamming_masked_count<PartialByte>(tail_query, tail_mask, c7 + body, valid));
        }
        const __m512i sums = hamming_reduce8(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
        _mm256_storeu_ps(dists + i, _mm512_cvtepi64_ps(sums));
    }
    for (; i < count; ++i) {
        dists[i] = simd::HammingToFloat(HammingDistance(query, codes + i * code_size, dim));
    }
}

// Scan of 8-, 16- or 32-byte codes: each 64-byte load holds 64 / CodeSize whole codes, so eight
// codes take CodeSize / 8 loads and their sums need only a partial reduction and a permutation.
template <uint64_t CodeSize, bool PartialByte>
inline void
hamming_scan_packed(
    const uint8_t* query, const uint8_t* codes, uint64_t dim, uint64_t count, float* dists) {
    static_assert(CodeSize == 8 or CodeSize == 16 or CodeSize == 32);
    // The query, and the mask of its valid bits, repeated in every CodeSize-byte slot.
    alignas(64) uint8_t query_slots[K_BYTES_PER_VECTOR];
    alignas(64) uint8_t valid_slots[K_BYTES_PER_VECTOR];
    for (uint64_t k = 0; k < K_BYTES_PER_VECTOR; ++k) {
        query_slots[k] = query[k % CodeSize];
        valid_slots[k] = 0xFF;
        if (PartialByte and k % CodeSize == CodeSize - 1) {
            valid_slots[k] = static_cast<uint8_t>(simd::HammingTailMask(dim));
        }
    }
    const __m512i q = _mm512_load_si512(query_slots);
    const __m512i valid = _mm512_load_si512(valid_slots);
    const auto count_block = [&](const uint8_t* block) {
        __m512i diff = _mm512_xor_si512(q, _mm512_loadu_si512(block));
        if constexpr (PartialByte) {
            diff = _mm512_and_si512(diff, valid);
        }
        return hamming_count_lanes(diff);
    };
    uint64_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const uint8_t* block = codes + i * CodeSize;
        __m512i sums;
        if constexpr (CodeSize == 8) {
            sums = count_block(block);
        } else if constexpr (CodeSize == 16) {
            // a holds codes 0-3 and b codes 4-7, two lanes per code. Adding the lane pairs gives
            // the codes in the order 0, 4, 1, 5, 2, 6, 3, 7.
            const __m512i a = count_block(block);
            const __m512i b = count_block(block + K_BYTES_PER_VECTOR);
            const __m512i mixed =
                _mm512_add_epi64(_mm512_unpacklo_epi64(a, b), _mm512_unpackhi_epi64(a, b));
            sums = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), mixed);
        } else {
            // Each register holds two codes with four lanes per code. Two pairing steps give the
            // codes in the order 0, 2, 1, 3, 4, 6, 5, 7.
            const __m512i a = count_block(block);
            const __m512i b = count_block(block + K_BYTES_PER_VECTOR);
            const __m512i c = count_block(block + 2 * K_BYTES_PER_VECTOR);
            const __m512i d = count_block(block + 3 * K_BYTES_PER_VECTOR);
            const __m512i ab =
                _mm512_add_epi64(_mm512_unpacklo_epi64(a, b), _mm512_unpackhi_epi64(a, b));
            const __m512i cd =
                _mm512_add_epi64(_mm512_unpacklo_epi64(c, d), _mm512_unpackhi_epi64(c, d));
            const __m512i mixed =
                _mm512_add_epi64(_mm512_shuffle_i64x2(ab, cd, _MM_SHUFFLE(2, 0, 2, 0)),
                                 _mm512_shuffle_i64x2(ab, cd, _MM_SHUFFLE(3, 1, 3, 1)));
            sums = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 1, 3, 4, 6, 5, 7), mixed);
        }
        _mm256_storeu_ps(dists + i, _mm512_cvtepi64_ps(sums));
    }
    for (; i < count; ++i) {
        dists[i] = simd::HammingToFloat(HammingDistance(query, codes + i * CodeSize, dim));
    }
}

template <bool PartialByte>
inline void
hamming_scan(
    const uint8_t* query, const uint8_t* codes, uint64_t dim, uint64_t count, float* dists) {
    switch ((dim + 7) / 8) {
        case 8:
            hamming_scan_packed<8, PartialByte>(query, codes, dim, count, dists);
            return;
        case 16:
            hamming_scan_packed<16, PartialByte>(query, codes, dim, count, dists);
            return;
        case 32:
            hamming_scan_packed<32, PartialByte>(query, codes, dim, count, dists);
            return;
        default:
            hamming_scan_blocks<PartialByte>(query, codes, dim, count, dists);
    }
}
}  // namespace
#endif

void
HammingDistanceScan(
    const uint8_t* query, const uint8_t* codes, uint64_t dim, uint64_t count, float* dists) {
#if defined(ENABLE_AVX512VPOPCNTDQ)
    if (dim % 8 == 0) {
        hamming_scan<false>(query, codes, dim, count, dists);
    } else {
        hamming_scan<true>(query, codes, dim, count, dists);
    }
#else
    avx512::HammingDistanceScan(query, codes, dim, count, dists);
#endif
}
}  // namespace vsag::avx512vpopcntdq
