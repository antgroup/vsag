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

#include "typing.h"

namespace vsag {

/// @brief Elias-Fano Encoder for *ordered* adjacency list
/// @note Our adjacency list size is mostly no more than 255, so we use uint8_t to store the number of elements
class EliasFanoEncoder {
private:
    Vector<uint64_t> high_bits_;  // Bitmap storage for high bits
    Vector<uint64_t> low_bits_;   // Compressed storage for low bits
    uint8_t num_elements_{0};     // Number of elements, max 255
    uint8_t low_bits_width_{0};   // Width of low bits

    // Cross-platform implementation of ctzll (count trailing zeros)
    size_t
    ctzll(uint64_t x) const;

    inline void
    set_high_bit(Vector<uint64_t>& vec, size_t pos) {
        vec[pos >> 6] |= (1ULL << (pos & 63));
    }

    void
    set_low_bits(size_t index, InnerIdType value);

    InnerIdType
    get_low_bits(size_t index) const;

public:
    EliasFanoEncoder(Allocator* allocator) : high_bits_(allocator), low_bits_(allocator) {
    }

    // Encode ordered sequence
    void
    encode(const Vector<InnerIdType>& values, InnerIdType max_value);

    // Decompress all values
    Vector<InnerIdType>
    decompress_all(Allocator* allocator) const;

    void
    clear() {
        high_bits_.clear();
        low_bits_.clear();
        num_elements_ = 0;
        low_bits_width_ = 0;
    }

    size_t
    size_in_bytes() const {
        return high_bits_.size() * sizeof(uint64_t) + low_bits_.size() * sizeof(uint64_t) +
               sizeof(num_elements_) + sizeof(low_bits_width_);
    }

    uint8_t
    size() const {
        return num_elements_;
    }
};

}  // namespace vsag