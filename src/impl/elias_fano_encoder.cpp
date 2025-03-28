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

#include "elias_fano_encoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace vsag {

size_t
EliasFanoEncoder::ctzll(uint64_t x) const {
#ifdef __GNUC__
    return __builtin_ctzll(x);
#else
    if (x == 0)
        return 64;
    int count = 0;
    while ((x & 1) == 0) {
        x >>= 1;
        count++;
    }
    return count;
#endif
}

void
EliasFanoEncoder::set_low_bits(size_t index, InnerIdType value) {
    if (low_bits_width_ == 0)
        return;

    size_t bit_pos = index * low_bits_width_;
    size_t word_pos = bit_pos >> 6;
    size_t shift = bit_pos & 63;
    uint64_t mask = ((1ULL << low_bits_width_) - 1) << shift;
    low_bits_[word_pos] = (low_bits_[word_pos] & ~mask) | ((uint64_t)value << shift);

    // Handle word boundary crossing
    if (shift + low_bits_width_ > 64 && word_pos + 1 < low_bits_.size()) {
        size_t remaining_bits = shift + low_bits_width_ - 64;
        mask = (1ULL << remaining_bits) - 1;
        low_bits_[word_pos + 1] =
            (low_bits_[word_pos + 1] & ~mask) | (value >> (low_bits_width_ - remaining_bits));
    }
}

InnerIdType
EliasFanoEncoder::get_low_bits(size_t index) const {
    if (low_bits_width_ == 0)
        return 0;

    size_t bit_pos = index * low_bits_width_;
    size_t word_pos = bit_pos >> 6;
    size_t shift = bit_pos & 63;
    InnerIdType value = (low_bits_[word_pos] >> shift) & ((1ULL << low_bits_width_) - 1);

    // Handle word boundary crossing
    if (shift + low_bits_width_ > 64 && word_pos + 1 < low_bits_.size()) {
        size_t remaining_bits = shift + low_bits_width_ - 64;
        value |= (low_bits_[word_pos + 1] & ((1ULL << remaining_bits) - 1))
                 << (low_bits_width_ - remaining_bits);
    }
    return value;
}

void
EliasFanoEncoder::encode(const Vector<InnerIdType>& values, InnerIdType max_value) {
    clear();
    if (values.empty())
        return;

    // Check if number of elements exceeds uint8_t maximum
    if (values.size() > UINT8_MAX) {
        num_elements_ = UINT8_MAX;
        throw std::runtime_error("Error: Elias-Fano encoder, number of elements exceeds 255.");
    } else {
        num_elements_ = static_cast<uint8_t>(values.size());
    }

    InnerIdType universe = max_value + 1;

    // Calculate low bits width
    low_bits_width_ =
        static_cast<uint32_t>(std::floor(std::log2(static_cast<double>(universe) / num_elements_)));

    // Allocate space for high bits
    const size_t high_bits_count = (max_value >> low_bits_width_) + num_elements_ + 1;
    high_bits_.resize((high_bits_count + 63) / 64, 0);

    // Allocate space for low bits
    size_t total_low_bits = num_elements_ * low_bits_width_;
    low_bits_.resize(std::max<size_t>(1, (total_low_bits + 63) / 64), 0);

    // Encode each value
    for (size_t i = 0; i < num_elements_; ++i) {
        InnerIdType x = values[i];
        InnerIdType high = x >> low_bits_width_;
        InnerIdType low = x & ((1U << low_bits_width_) - 1);

        set_high_bit(high_bits_, i + high);
        set_low_bits(i, low);
    }
}

Vector<InnerIdType>
EliasFanoEncoder::decompress_all(Allocator* allocator) const {
    Vector<InnerIdType> result(allocator);
    result.reserve(num_elements_);

    // Decompress all values at once
    size_t count = 0;

    for (size_t i = 0; i < high_bits_.size() && count < num_elements_; ++i) {
        uint64_t word = high_bits_[i];

        // Use ctzll to find position of 1
        while (word && count < num_elements_) {
            size_t bit = ctzll(word);
            // Found 1, calculate corresponding value
            InnerIdType high = (i * 64 + bit) - count;
            InnerIdType low = get_low_bits(count);
            result.push_back((high << low_bits_width_) | low);
            count++;
            // Delete lowest 1
            word &= (word - 1);
        }
    }

    return result;
}

}  // namespace vsag