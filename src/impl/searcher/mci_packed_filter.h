// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

#include "typing.h"

namespace vsag {

// Thread-private scratch owns its memory independently of index/query allocators.
// Filter preserves CSR order; all source slots are still examined once.
class MCIPackedFilter {
public:
    struct Members {
        const InnerIdType* data;
        uint64_t count;
        [[nodiscard]] const InnerIdType*
        // NOLINTNEXTLINE(readability-identifier-naming): range-for protocol
        begin() const {
            return data;
        }
        [[nodiscard]] const InnerIdType*
        // NOLINTNEXTLINE(readability-identifier-naming): range-for protocol
        end() const {
            // An empty vector may have null data; avoid arithmetic on that pointer.
            return count == 0 ? data : data + count;
        }
    };

    void
    Build(uint64_t total, const uint8_t* bitmap, const InnerIdType* ids, uint64_t count) {
        total_ = total;
        words_.assign((total + 31) / 32, 0);
        if (ids != nullptr) {
            for (uint64_t i = 0; i < count; ++i) {
                if (ids[i] < total) {
                    words_[ids[i] / 32] |= uint32_t{1} << (ids[i] % 32);
                }
            }
        } else {
            for (uint64_t i = 0; i < total; ++i) {
                if (bitmap[i] != 0) {
                    words_[i / 32] |= uint32_t{1} << (i % 32);
                }
            }
        }
    }

    [[nodiscard]] bool
    IsValid(InnerIdType id) const {
        return id < total_ and (words_[id / 32] & (uint32_t{1} << (id % 32))) != 0;
    }

    Members
    Filter(const InnerIdType* members, uint64_t count, uint64_t total, bool simd = true) {
        total = std::min(total, total_);
        if (selected_.size() < count) {
            selected_.resize(count);
        }
        uint64_t selected = 0;
        uint64_t offset = 0;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
        if (simd and __builtin_cpu_supports("avx512f")) {
            offset = filter512(members, count, total, selected);
        } else if (simd and __builtin_cpu_supports("avx2")) {
            offset = filter256(members, count, total, selected);
        }
#else
        (void)simd;
#endif
        for (; offset < count; ++offset) {
            const auto id = members[offset];
            if (id < total and this->IsValid(id)) {
                selected_[selected++] = id;
            }
        }
        return {selected_.data(), selected};
    }

private:
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __attribute__((target("avx512f"))) uint64_t
    filter512(const InnerIdType* members, uint64_t count, uint64_t total, uint64_t& selected) {
        uint64_t offset = 0;
        const auto ones = _mm512_set1_epi32(1);
        const auto low = _mm512_set1_epi32(31);
        const auto upper = _mm512_set1_epi32(static_cast<int32_t>(total));
        for (; offset + 16 <= count; offset += 16) {
            const auto ids = _mm512_loadu_si512(members + offset);
            const auto in_range = _mm512_cmplt_epu32_mask(ids, upper);
            const auto words = _mm512_mask_i32gather_epi32(
                _mm512_setzero_si512(), in_range, _mm512_srli_epi32(ids, 5), words_.data(), 4);
            const auto shifted = _mm512_srlv_epi32(words, _mm512_and_si512(ids, low));
            auto mask = static_cast<uint32_t>(_mm512_test_epi32_mask(shifted, ones) & in_range);
            while (mask != 0) {
                selected_[selected++] = members[offset + __builtin_ctz(mask)];
                mask &= mask - 1;
            }
        }
        return offset;
    }

    __attribute__((target("avx2"))) uint64_t
    filter256(const InnerIdType* members, uint64_t count, uint64_t total, uint64_t& selected) {
        uint64_t offset = 0;
        const auto ones = _mm256_set1_epi32(1);
        const auto low = _mm256_set1_epi32(31);
        for (; offset + 8 <= count; offset += 8) {
            const auto ids = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(members + offset));
            const auto sign = _mm256_set1_epi32(static_cast<int32_t>(0x80000000U));
            const auto upper = _mm256_set1_epi32(static_cast<int32_t>(total));
            const auto in_range =
                _mm256_cmpgt_epi32(_mm256_xor_si256(upper, sign), _mm256_xor_si256(ids, sign));
            const auto words =
                _mm256_mask_i32gather_epi32(_mm256_setzero_si256(),
                                            reinterpret_cast<const int*>(words_.data()),
                                            _mm256_srli_epi32(ids, 5),
                                            in_range,
                                            4);
            const auto shifted = _mm256_srlv_epi32(words, _mm256_and_si256(ids, low));
            const auto valid = _mm256_cmpeq_epi32(_mm256_and_si256(shifted, ones), ones);
            auto mask = static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(valid)));
            while (mask != 0) {
                selected_[selected++] = members[offset + __builtin_ctz(mask)];
                mask &= mask - 1;
            }
        }
        return offset;
    }
#endif
    uint64_t total_{0};
    std::vector<uint32_t> words_;
    std::vector<InnerIdType> selected_;
};

}  // namespace vsag
