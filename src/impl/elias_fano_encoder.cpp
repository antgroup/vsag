
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

#include <cmath>
#include <iostream>

namespace vsag {

// Cross-platform implementation of ctzll (count trailing zeros)
static inline size_t
ctzll(uint64_t x) {
#ifdef __GNUC__
    return __builtin_ctzll(x);
#else
    if (x == 0) {
        return 64;
    }
    int count = 0;
    while ((x & 1) == 0) {
        x >>= 1;
        count++;
    }
    return count;
#endif
}

static inline void
set_high_bit(uint64_t* bits_array, size_t pos, size_t low_bits_size) {
    bits_array[low_bits_size + (pos >> 6)] |= (1ULL << (pos & 63));
}

// requires "const" to pass lint check
void
EliasFanoEncoder::set_low_bits(size_t index, InnerIdType value) const {
    if (low_bits_width == 0) {
        return;
    }

    size_t bit_pos = index * low_bits_width;
    size_t word_pos = bit_pos >> 6;
    size_t shift = bit_pos & 63;
    uint64_t mask = ((1ULL << low_bits_width) - 1) << shift;
    bits[word_pos] = (bits[word_pos] & ~mask) | ((uint64_t)value << shift);

    // Handle word boundary crossing
    if (shift + low_bits_width > 64 && word_pos + 1 < low_bits_size) {
        size_t remaining_bits = shift + low_bits_width - 64;
        mask = (1ULL << remaining_bits) - 1;
        bits[word_pos + 1] =
            (bits[word_pos + 1] & ~mask) | (value >> (low_bits_width - remaining_bits));
    }
}

InnerIdType
EliasFanoEncoder::get_low_bits(size_t index) const {
    if (low_bits_width == 0) {
        return 0;
    }

    size_t bit_pos = index * low_bits_width;
    size_t word_pos = bit_pos >> 6;
    size_t shift = bit_pos & 63;
    InnerIdType value = (bits[word_pos] >> shift) & ((1ULL << low_bits_width) - 1);

    // Handle word boundary crossing
    if (shift + low_bits_width > 64 && word_pos + 1 < low_bits_size) {
        size_t remaining_bits = shift + low_bits_width - 64;
        value |= (bits[word_pos + 1] & ((1ULL << remaining_bits) - 1))
                 << (low_bits_width - remaining_bits);
    }
    return value;
}

void
EliasFanoEncoder::Encode(const Vector<InnerIdType>& values,
                         InnerIdType max_value,
                         Allocator* allocator) {
    Clear(allocator);
    if (values.empty()) {
        return;
    }

    // Check if number of elements exceeds uint8_t maximum
    if (values.size() <= UINT8_MAX) {
        num_elements = static_cast<uint8_t>(values.size());
    } else {
        throw std::runtime_error("Error: Elias-Fano encoder, number of elements exceeds 255.");
    }

    InnerIdType universe = max_value + 1;

    // Calculate low bits width
    low_bits_width =
        static_cast<uint32_t>(std::floor(std::log2(static_cast<double>(universe) / num_elements)));

    // Calculate the size of high bits
    const size_t high_bits_count = (max_value >> low_bits_width) + num_elements + 1;
    high_bits_size = (high_bits_count + 63) / 64;

    // Calculate the size of low bits
    size_t total_low_bits = static_cast<size_t>(num_elements) * low_bits_width;
    low_bits_size = std::max<size_t>(1, (total_low_bits + 63) / 64);

    // Allocate combined space for both low and high bits
    bits = static_cast<uint64_t*>(
        allocator->Allocate((low_bits_size + high_bits_size) * sizeof(uint64_t)));
    std::fill(bits, bits + low_bits_size + high_bits_size, 0);

    // Encode each value
    for (size_t i = 0; i < num_elements; ++i) {
        InnerIdType x = values[i];
        InnerIdType high = x >> low_bits_width;
        InnerIdType low = x & ((1U << low_bits_width) - 1);

        set_high_bit(bits, i + high, low_bits_size);
        set_low_bits(i, low);
    }
}

void
EliasFanoEncoder::DecompressAll(Vector<InnerIdType>& neighbors) const {
    neighbors.resize(num_elements);

    // Decompress all values at once
    size_t count = 0;
    for (size_t i = 0; i < high_bits_size && count < num_elements; ++i) {
        uint64_t word = bits[low_bits_size + i];

        // Use ctzll to find position of 1
        while (word != 0U && count < num_elements) {
            size_t bit = ctzll(word);
            // Found 1, calculate corresponding value
            InnerIdType high = (i * 64 + bit) - count;
            InnerIdType low = get_low_bits(count);
            neighbors[count] = ((high << low_bits_width) | low);
            count++;
            // Delete lowest 1
            word &= (word - 1);
        }
    }
}

}  // namespace vsag
