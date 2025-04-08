
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
public:
    EliasFanoEncoder(Allocator* allocator) : allocator_(allocator) {
    }

    ~EliasFanoEncoder() {
        Clear();
    }

    // Encode ordered sequence
    void
    Encode(const Vector<InnerIdType>& values, InnerIdType max_value);

    // Decompress all values
    Vector<InnerIdType>
    DecompressAll(Allocator* allocator) const;

    void
    Clear() {
        if (bits_ != nullptr) {
            allocator_->Deallocate(bits_);
            bits_ = nullptr;
        }
        num_elements_ = 0;
        low_bits_width_ = 0;
        low_bits_size_ = 0;
        high_bits_size_ = 0;
    }

    [[nodiscard]] size_t
    SizeInBytes() const {
        return sizeof(EliasFanoEncoder) + (low_bits_size_ + high_bits_size_) * sizeof(uint64_t);
    }

    [[nodiscard]] uint8_t
    Size() const {
        return num_elements_;
    }

private:
    void
    set_low_bits(size_t index, InnerIdType value);

    [[nodiscard]] InnerIdType
    get_low_bits(size_t index) const;

    Allocator* allocator_;
    uint64_t* bits_{nullptr};    // Combined storage for low bits and high bits
    uint8_t num_elements_{0};    // Number of elements, max 255
    uint8_t low_bits_width_{0};  // Width of low bits
    uint8_t low_bits_size_{0};   // Size of low_bits_ array
    uint8_t high_bits_size_{0};  // Size of high_bits_ array
};

}  // namespace vsag
